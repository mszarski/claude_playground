#include "sonar/live/udp_source.hpp"

#include "sonar/live/board_merge.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace sonar::live {
namespace {

constexpr std::size_t kHeader = 128;

template <typename T>
T get(const unsigned char* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));   // the layout is little-endian, as are the machines this runs on
  return v;
}

}  // namespace

double block_full_scale(std::int64_t bytes) {
  if (bytes != 2 && bytes != 3 && bytes != 4) throw std::invalid_argument("stream: sample_bytes must be 2, 3 or 4");
  return std::ldexp(1.0, static_cast<int>(8 * bytes - 1));
}

PingAssembler::PingAssembler(const PingRecord& template_record, const DecimateOptions* decimate, const Reference* reference, int sample_bytes)
    : template_(template_record), sample_bytes_(sample_bytes) {
  const Samples& s = template_.samples;
  if (s.is_complex()) throw std::invalid_argument("PingAssembler: the template must describe real ADC samples");
  window_start_ = static_cast<std::int64_t>(std::llround(s.start_time * s.sample_rate));
  window_length_ = s.n_samples;
  n_channels_ = s.n_channels;
  const double full_scale = template_.metadata.adc_full_scale;
  volts_per_count_ = (std::isfinite(full_scale) ? full_scale : 1.0) / block_full_scale(sample_bytes_);
  template_.samples.data.clear();
  if (decimate != nullptr && decimate->enabled) decimator_ = std::make_shared<PingDecimator>(template_, *decimate, reference);
}

void PingAssembler::reset() {
  started_ = have_first_tx_ = have_max_ = false;
  frontier_ = origin_ = stream_ns_ = first_tx_ = 0;
  holes_.clear();
  expired_.clear();
  lost_.clear();
  pings_.clear();
  owner_ = nullptr;
  released_.clear();
  deltas_.clear();
}

bool PingAssembler::in_holes(std::int64_t a, std::int64_t b) const {
  for (const Hole& h : holes_) {
    if (h.a <= a && a < h.b) {
      a = h.b;
      if (a >= b) return true;
    }
  }
  return a >= b;
}

bool PingAssembler::holds(std::int64_t a, std::int64_t b) const {
  if (!started_ || b > frontier_) return true;
  for (const Hole& h : holes_) {
    if (std::min(b, h.b) > std::max(a, h.a)) return true;
  }
  return false;
}

void PingAssembler::lose(std::int64_t a, std::int64_t b) {
  lost_samples += b - a;
  expired_.push_back({a, b});
  lost_.push_back({a, b});
  if (lost_.size() > 64) lost_.erase(lost_.begin());
}

void PingAssembler::give_up(std::int64_t limit) {
  std::vector<Hole> keep;
  for (const Hole& h : holes_) {
    bool held = false;
    for (const Ping& p : pings_) held = held || (p.state == State::kOpen && h.a < p.w0 + window_length_ && h.b > p.w0);
    if (h.b <= limit && !held) lose(h.a, h.b);
    else keep.push_back(h);
  }
  holes_.swap(keep);
}

void PingAssembler::give_up_to(std::int64_t end, std::int64_t n) {
  std::vector<Hole> keep;
  for (const Hole& h : holes_) {
    if (h.a >= end) {
      keep.push_back(h);
      continue;
    }
    const std::int64_t cut = n > 0 ? std::min(h.b, h.a + (end - h.a + n - 1) / n * n) : h.b;
    lose(h.a, cut);
    if (cut < h.b) keep.push_back({cut, h.b});
  }
  holes_.swap(keep);
}

bool PingAssembler::take_expired(std::int64_t a, std::int64_t b) {
  bool hit = false;
  std::vector<Hole> keep;
  for (const Hole& h : expired_) {
    const std::int64_t x = std::max(a, h.a), y = std::min(b, h.b);
    if (y <= x) {
      keep.push_back(h);
      continue;
    }
    hit = true;
    if (h.a < x) keep.push_back({h.a, x});
    if (y < h.b) keep.push_back({y, h.b});
  }
  expired_.swap(keep);
  return hit;
}

bool PingAssembler::known(std::int64_t tx, std::int64_t index) const {
  for (const Ping& p : pings_) {
    if (p.tx == tx && p.index == index) return true;
  }
  return std::find(released_.begin(), released_.end(), std::make_pair(tx, index)) != released_.end();
}

bool PingAssembler::fits(const Header& h) const {
  if (h.start_ns != stream_ns_) return false;
  if (h.first >= frontier_) {                            // ahead: names the latest transmit, or a later one
    if (h.tx < 0) return !have_max_;
    return !have_max_ || h.tx > max_named_.first || std::make_pair(h.tx, h.index) == max_named_;
  }
  bool in_hole = false;                                  // a block in a hole still open always may come
  for (const Hole& o : holes_) in_hole = in_hole || std::min(h.first + h.n, o.b) > std::max(h.first, o.a);
  if ((h.first < frontier_ - span(h.n) && !in_hole) || h.first < origin_ - reorder_blocks * h.n) return false;
  if (h.tx < 0) return !have_first_tx_ || h.first < first_tx_;
  if (known(h.tx, h.index)) return true;
  if (h.tx > h.first + h.n || !have_max_ || h.tx > max_named_.first) return false;
  for (const Ping& p : pings_) {
    if (p.tx == h.tx) return false;                      // a heard transmit under another index
  }
  for (const auto& r : released_) {
    if (r.first == h.tx) return false;
  }
  // A transmit not heard yet: its index must fit between those of the transmits heard around it.
  bool have_before = false, have_after = false;
  std::pair<std::int64_t, std::int64_t> before{0, 0}, after{0, 0};
  auto consider = [&](std::int64_t tx, std::int64_t index) {
    if (tx < h.tx && (!have_before || tx > before.first)) {
      before = {tx, index};
      have_before = true;
    }
    if (tx > h.tx && (!have_after || tx < after.first)) {
      after = {tx, index};
      have_after = true;
    }
  };
  for (const auto& r : released_) consider(r.first, r.second);
  for (const Ping& p : pings_) consider(p.tx, p.index);
  const std::int64_t half = std::int64_t{1} << 31;
  if (have_before) {
    const std::int64_t d = (h.index - before.second) & 0xFFFFFFFFLL;
    if (!(d > 0 && d < half)) return false;
  }
  if (have_after) {
    const std::int64_t d = (after.second - h.index) & 0xFFFFFFFFLL;
    if (!(d > 0 && d < half)) return false;
  }
  return true;
}

void PingAssembler::name(const Header& h) {
  if (!released_.empty() && h.tx <= released_.back().first) return;   // too late: a later ping was already decided
  if (owner_ != nullptr && owner_->tx > h.tx) return;                  // too late for the decimator, on a later ping
  if (!have_first_tx_) {
    first_tx_ = h.tx;
    have_first_tx_ = true;
  }
  if (!have_max_ || h.tx > max_named_.first) {
    max_named_ = {h.tx, h.index};
    have_max_ = true;
  }
  Ping p;
  p.tx = h.tx;
  p.index = h.index;
  p.start_ns = h.start_ns;
  p.w0 = h.tx + window_start_;
  p.state = h.first <= p.w0 || in_holes(p.w0, h.first) ? State::kOpen : State::kDropped;   // else joined too late
  if (p.state == State::kOpen && !decimator_) p.buffer.assign(static_cast<std::size_t>(window_length_ * n_channels_), 0);
  p.record = template_;
  auto at = pings_.end();
  while (at != pings_.begin() && std::prev(at)->tx > h.tx) --at;
  const auto it = pings_.insert(at, std::move(p));
  // Transmitted before the previous window closed: the previous ping is given up, as it always was. Likewise this
  // one, if the next was transmitted inside its window.
  if (it != pings_.begin()) {
    Ping& before = *std::prev(it);
    if (before.state == State::kOpen && h.tx < before.w0 + window_length_) drop(before);
  }
  if (std::next(it) != pings_.end() && it->state == State::kOpen && std::next(it)->tx < it->w0 + window_length_) drop(*it);
  std::vector<Ping*> opened;
  for (Ping& q : pings_) {
    if (q.state == State::kOpen && q.covered < window_length_) opened.push_back(&q);
  }
  for (std::size_t k = 0; k + kMaxOpenPings < opened.size(); ++k) drop(*opened[k]);   // more open than the grace allows
}

void PingAssembler::drop(Ping& p) {
  p.state = State::kDropped;
  p.buffer.clear();
  p.buffer.shrink_to_fit();
  p.pending.clear();
  p.held = 0;
  if (owner_ == &p) owner_ = nullptr;
}

void PingAssembler::drain(Ping& p) {
  if (owner_ != &p) return;
  for (auto it = p.pending.find(p.fed); it != p.pending.end(); it = p.pending.find(p.fed)) {
    const auto rows = static_cast<std::int64_t>(it->second.size()) / n_channels_;
    decimator_->feed(it->second.data(), rows);
    p.held -= static_cast<std::int64_t>(it->second.size() * sizeof(float));
    p.fed += rows;
    p.pending.erase(it);
  }
}

void PingAssembler::place(Ping& p, const Header& h, std::int64_t a, std::int64_t b) {
  a = std::max(a, p.w0);
  b = std::min(b, p.w0 + window_length_);
  if (b <= a) return;
  const unsigned char* payload = h.bytes.data() + kHeader;
  const std::int64_t ch = n_channels_;
  if (decimator_) rows_.resize(static_cast<std::size_t>((b - a) * ch));
  for (std::int64_t i = a; i < b; ++i) {
    for (std::int64_t c = 0; c < ch; ++c) {
      const std::int32_t v = block_sample(payload + ((i - h.first) * ch + c) * sample_bytes_, sample_bytes_);
      if (decimator_) rows_[static_cast<std::size_t>((i - a) * ch + c)] = static_cast<float>(static_cast<double>(v) * volts_per_count_);
      else p.buffer[static_cast<std::size_t>((i - p.w0) * ch + c)] = v;
    }
  }
  if (decimator_) {          // filtered and decimated as it arrives, in order: rows behind a hole wait for it
    p.held += static_cast<std::int64_t>(rows_.size() * sizeof(float));
    p.pending[a - p.w0] = rows_;
    drain(p);
  }
  p.covered += b - a;
}

PingRecord PingAssembler::make_record(Ping& p) {
  PingRecord rec = std::move(p.record);
  rec.metadata.sequence = p.index;
  // clock_ppm 0: capture start + tx / fs, as always; else the board clock's rate taken out (sonar-ubg, board_merge.hpp).
  rec.transmit.time_ns = clock_ppm == 0.0
                             ? p.start_ns + static_cast<std::int64_t>(std::llround(static_cast<double>(p.tx) / rec.samples.sample_rate * 1e9))
                             : p.start_ns + stamp_offset_ns(p.tx, rec.samples.sample_rate, clock_ppm);
  if (decimator_) {
    decimator_->finish(rec);
    return rec;
  }
  rec.samples.resize(n_channels_, window_length_, SampleKind::RealPassband);
  rec.samples.start_time = static_cast<double>(window_start_) / rec.samples.sample_rate;
  for (std::int64_t c = 0; c < n_channels_; ++c) {
    float* dst = rec.samples.channel(c);
    for (std::int64_t i = 0; i < window_length_; ++i) {
      dst[i] = static_cast<float>(static_cast<double>(p.buffer[static_cast<std::size_t>(i * n_channels_ + c)]) * volts_per_count_);
    }
  }
  p.buffer.clear();
  p.buffer.shrink_to_fit();
  return rec;
}

bool PingAssembler::gap_open(const Ping& p) const {
  if (released_.empty()) return false;
  if (((p.index - released_.back().second) & 0xFFFFFFFFLL) < 2) return false;
  for (const Hole& h : holes_) {
    if (h.a < p.tx && h.b > released_.back().first) return true;
  }
  return false;
}

void PingAssembler::settle_pings() {
  for (Ping& p : pings_) {
    if (p.state != State::kOpen) continue;
    if (!decimator_) {
      if (p.covered == window_length_) {
        p.record = make_record(p);
        p.state = State::kComplete;
      }
      continue;
    }
    if (owner_ != &p) {
      if (owner_ != nullptr || gap_open(p)) break;     // an earlier ping has the decimator, or may still be named
      owner_ = &p;
      decimator_->open();
      drain(p);
    }
    if (p.covered < window_length_) break;
    p.record = make_record(p);
    p.state = State::kComplete;
    owner_ = nullptr;
  }
  std::int64_t held = 0, open = 0;
  for (const Ping& p : pings_) {
    held += p.held;
    open += p.state == State::kOpen ? 1 : 0;
  }
  held_peak_bytes = std::max(held_peak_bytes, held);
  open_peak = std::max(open_peak, open);
}

void PingAssembler::release(std::vector<PingRecord>& out, bool wait) {
  while (!pings_.empty() && pings_.front().state != State::kOpen) {
    Ping& p = pings_.front();
    if (wait && gap_open(p)) break;
    if (!released_.empty()) {
      const std::int64_t d = (p.index - released_.back().second) & 0xFFFFFFFFLL;
      const std::int64_t gap = p.tx - released_.back().first;
      if (d == 1) {
        deltas_.push_back(gap);
        if (deltas_.size() > 5) deltas_.erase(deltas_.begin());
      } else if (d >= 2 && !deltas_.empty()) {
        std::vector<std::int64_t> sorted = deltas_;
        std::sort(sorted.begin(), sorted.end());
        const std::int64_t interval = sorted[sorted.size() / 2];
        // |gap - d interval| < interval / 2; d beyond gap / interval + 1 cannot be (and d * interval could overflow).
        if (interval > 0 && gap >= 0 && d <= gap / interval + 1) {
          const std::int64_t off = gap - d * interval;
          // ... and longer than d - 1 of the longest recent intervals by half an interval
          const std::int64_t longest = *std::max_element(deltas_.begin(), deltas_.end());
          // ... and samples lost between the two transmits: the blocks of a ping lost whole name it, so they are missing
          // from the counter; a sender that skipped an index leaves none missing (sonar-2ht)
          const std::int64_t tx = released_.back().first;
          bool lost_between = false;
          for (const std::vector<Hole>* v : {&lost_, &holes_}) {
            for (const Hole& h : *v) lost_between = lost_between || (h.a < p.tx && h.b > tx);
          }
          if ((off < 0 ? -off : off) * 2 < interval && (gap - (d - 1) * longest) * 2 > interval && lost_between) dropped_pings += d - 1;
        }
      }
    }
    released_.emplace_back(p.tx, p.index);
    if (released_.size() > 8) released_.erase(released_.begin());
    const std::int64_t after = p.tx;   // later pings are transmitted after this one
    lost_.erase(std::remove_if(lost_.begin(), lost_.end(), [after](const Hole& h) { return h.b <= after; }), lost_.end());
    if (p.state == State::kComplete) out.push_back(std::move(p.record));
    else ++dropped_pings;
    pings_.pop_front();
  }
}

void PingAssembler::close(std::vector<PingRecord>& out, bool drop_open) {
  for (Ping& p : pings_) {
    if (p.state != State::kOpen || p.covered >= window_length_) continue;
    bool cut = drop_open;
    for (const Hole& h : holes_) cut = cut || (h.a < p.w0 + window_length_ && h.b > p.w0);
    if (cut) drop(p);
  }
  for (const Hole& h : holes_) lose(h.a, h.b);
  holes_.clear();
  settle_pings();                                        // complete pings that waited for the decimator
  pings_.remove_if([](const Ping& p) { return p.state == State::kOpen; });
  owner_ = nullptr;
  release(out, false);
}

void PingAssembler::finish(std::vector<PingRecord>& out) {
  repeated_blocks += static_cast<std::int64_t>(suspects_.size());
  suspects_.clear();
  held_run_ = 0;
  if (started_) {
    close(out, false);
    reset();
  }
}

PingAssembler::Block PingAssembler::take(const Header& h, std::vector<PingRecord>& out) {
  const std::int64_t first = h.first, n = h.n, end = h.first + h.n;
  if (!started_) {
    started_ = true;
    frontier_ = origin_ = first;
    stream_ns_ = h.start_ns;
  }
  if (first < origin_) {                                 // just before the first block heard: a late block
    holes_.insert(holes_.begin(), {first, origin_});
    origin_ = first;
  }
  const std::int64_t frontier = frontier_;
  Block kind = Block::kNone;
  std::vector<Hole> fresh;                               // received for the first time
  if (first < frontier) {                                // the parts of the block in open holes, taken out of them
    const std::int64_t b = std::min(end, frontier);
    std::vector<Hole> keep;
    for (const Hole& o : holes_) {
      const std::int64_t x = std::max(first, o.a), y = std::min(b, o.b);
      if (y <= x) {
        keep.push_back(o);
        continue;
      }
      fresh.push_back({x, y});
      if (o.a < x) keep.push_back({o.a, x});
      if (y < o.b) keep.push_back({y, o.b});
    }
    holes_.swap(keep);
  }
  if (end > frontier) {
    if (first > frontier) {
      holes_.push_back({frontier, first});
      kind = Block::kGap;
    }
    fresh.push_back({std::max(first, frontier), end});
  }
  if (fresh.empty()) {
    if (take_expired(first, end)) {
      ++late_blocks;
      ++too_late_blocks;
      return Block::kTooLate;
    }
    ++repeated_blocks;
    return Block::kRepeated;
  }
  if (first < frontier && fresh.front().a < frontier) {
    ++late_blocks;
    if (kind == Block::kNone) kind = Block::kLate;
  }
  ++used_blocks;
  frontier_ = std::max(frontier, end);
  if (h.flags & 1) ++overflow_blocks;
  if (h.tx >= 0 && !known(h.tx, h.index)) name(h);
  for (Ping& p : pings_) {
    if (p.state != State::kOpen) continue;
    for (const Hole& f : fresh) place(p, h, f.a, f.b);
  }
  settle_pings();
  const std::int64_t horizon = reorder_blocks * n;
  for (Ping& p : pings_) {
    if (p.state == State::kOpen && frontier_ >= p.w0 + window_length_ + horizon) {
      drop(p);                                           // given up: its holes with it
      give_up_to(p.w0 + window_length_, n);
    }
  }
  settle_pings();
  for (;;) {                                             // rows held behind holes, all pings together, past the bound
    std::int64_t held = 0;
    Ping* first_open = nullptr;
    for (Ping& p : pings_) {
      held += p.held;
      if (first_open == nullptr && p.state == State::kOpen) first_open = &p;
    }
    if (held <= max_held_bytes || first_open == nullptr) break;
    drop(*first_open);
    give_up_to(first_open->w0 + window_length_, n);
    settle_pings();
  }
  give_up(frontier_ - horizon);
  release(out, true);
  const std::int64_t low = frontier_ - span(n);
  expired_.erase(std::remove_if(expired_.begin(), expired_.end(), [low](const Hole& o) { return o.b <= low; }), expired_.end());
  if (expired_.size() > 64) expired_.erase(expired_.begin(), expired_.end() - 64);
  return kind == Block::kNone ? Block::kInOrder : kind;
}

bool PingAssembler::feed(const unsigned char* data, std::size_t size, std::vector<PingRecord>& out) {
  // A per-board block (flag bit 1, sonar-6f9) is not part of a whole-array stream: merged first (board_merge.hpp).
  if (size < kHeader || std::memcmp(data, "GSFB", 4) != 0 || get<std::uint16_t>(data + 4) != 1 || (get<std::uint16_t>(data + 34) & 2) != 0) {
    ++bad_blocks;
    return false;
  }
  const auto ch = static_cast<std::int64_t>(get<std::uint16_t>(data + 6));
  const auto n = static_cast<std::int64_t>(get<std::uint32_t>(data + 8));
  const auto first = get<std::uint64_t>(data + 12);
  const auto tx = get<std::int64_t>(data + 20);
  if (ch != n_channels_ || size != kHeader + static_cast<std::size_t>(n * ch * sample_bytes_) ||
      first >= static_cast<std::uint64_t>(kMaxSampleIndex) || tx < -1 || tx >= kMaxSampleIndex ||
      tx >= static_cast<std::int64_t>(first) + n) {      // a transmit beyond the block's own end is refused too
    ++bad_blocks;
    return false;
  }
  if (reorder_blocks < 0) throw std::invalid_argument("PingAssembler: reorder_blocks must be 0 or more");
  Header h;
  h.first = static_cast<std::int64_t>(first);
  h.n = n;
  h.tx = tx;
  h.index = static_cast<std::int64_t>(get<std::uint32_t>(data + 28));
  h.flags = get<std::uint16_t>(data + 34);
  h.start_ns = static_cast<std::int64_t>(get<std::uint64_t>(data + 36));
  h.bytes.assign(data, data + size);
  if (!started_ || fits(h)) {
    ++fitting_;
    held_run_ = 0;
    const std::int64_t fitting = fitting_, stale_after = reorder_blocks + 2;
    const auto stale = [&](const std::pair<std::int64_t, Header>& s) { return fitting - s.first > stale_after; };
    repeated_blocks += std::count_if(suspects_.begin(), suspects_.end(), stale);
    suspects_.erase(std::remove_if(suspects_.begin(), suspects_.end(), stale), suspects_.end());
    last_block = take(h, out);
    return true;
  }
  if (std::find(left_.begin(), left_.end(), h.start_ns) != left_.end()) {   // a capture already left: stale
    ++repeated_blocks;
    last_block = Block::kRepeated;
    return true;
  }
  ++held_run_;
  std::vector<const Header*> run = restart_run(h);
  if (run.empty()) {
    suspects_.emplace_back(fitting_, std::move(h));
    if (suspects_.size() > static_cast<std::size_t>(kMaxHeldBlocks)) {
      suspects_.pop_front();
      ++repeated_blocks;
    }
    last_block = Block::kHeld;
    return true;
  }
  ++restarts;                                            // a restart: the old stream ends here
  close(out, true);
  if (stream_ns_ != h.start_ns) {
    left_.push_back(stream_ns_);
    if (left_.size() > 8) left_.erase(left_.begin());
  }
  reset();
  std::vector<Header> fresh;
  for (const Header* r : run) fresh.push_back(r == &h ? h : *r);
  for (const auto& s : suspects_) {
    if (std::find(run.begin(), run.end(), &s.second) == run.end()) ++repeated_blocks;
  }
  suspects_.clear();
  held_run_ = 0;
  std::stable_sort(fresh.begin(), fresh.end(), [](const Header& x, const Header& y) { return x.first < y.first; });
  for (const Header& b : fresh) {
    if (!started_ || fits(b)) take(b, out);
    else ++repeated_blocks;
  }
  last_block = Block::kRestart;
  return true;
}

std::vector<const PingAssembler::Header*> PingAssembler::restart_run(const Header& h) const {
  // (a) a contiguous run of one capture: two of another capture start, or, of this one (a board without a clock),
  // two that end before the first sample this stream heard by more than the restart span; (b) kMaxHeldBlocks
  // held in a row, none fitting in between, of one capture start and consistent with each other. As in flank_stream.py.
  std::vector<const Header*> held;
  for (const auto& s : suspects_) {
    if (s.second.start_ns == h.start_ns) held.push_back(&s.second);
  }
  held.push_back(&h);
  std::vector<const Header*> chain{&h};
  for (bool grown = true; grown;) {
    grown = false;
    for (const Header* c : held) {
      if (std::find(chain.begin(), chain.end(), c) != chain.end()) continue;
      bool next = false;
      for (const Header* x : chain) next = next || c->first + c->n == x->first || x->first + x->n == c->first;
      if (next) {
        chain.push_back(c);
        grown = true;
      }
    }
  }
  if (chain.size() >= 2) {
    if (h.start_ns != stream_ns_) return chain;
    bool before = true;
    for (const Header* c : chain) before = before && c->first + c->n <= origin_ - span(h.n);
    if (before) return chain;
  }
  if (held_run_ >= kMaxHeldBlocks && held.size() >= static_cast<std::size_t>(kMaxHeldBlocks)) {
    std::vector<const Header*> run = held;
    std::stable_sort(run.begin(), run.end(), [](const Header* x, const Header* y) { return x->first < y->first; });
    bool consistent = true;
    for (std::size_t k = 1; k < run.size(); ++k) {
      consistent = consistent && run[k - 1]->first + run[k - 1]->n <= run[k]->first && run[k - 1]->tx <= run[k]->tx;
    }
    if (consistent) return run;
  }
  return {};
}

StreamAssembler::StreamAssembler(const PingRecord& template_record, std::int64_t chunk_samples, const DecimateOptions* decimate,
                                 int sample_bytes)
    : template_(template_record), chunk_(chunk_samples), sample_bytes_(sample_bytes) {
  if (template_.samples.is_complex()) throw std::invalid_argument("StreamAssembler: the template must describe real ADC samples");
  if (chunk_ < 1) throw std::invalid_argument("StreamAssembler: chunk_samples must be positive");
  n_channels_ = template_.samples.n_channels;
  const double full_scale = template_.metadata.adc_full_scale;
  volts_per_count_ = (std::isfinite(full_scale) ? full_scale : 1.0) / block_full_scale(sample_bytes_);
  template_.samples.data.clear();
  if (decimate != nullptr && decimate->enabled) {
    plan_ = plan_decimation(template_, *decimate);
    decimator_ = std::make_shared<StreamDecimator>(plan_, n_channels_);
    decimator_->open(0.0);
    pending_.assign(static_cast<std::size_t>(n_channels_), {});
    return;
  }
  buffer_.assign(static_cast<std::size_t>(n_channels_ * chunk_), 0.0F);
}

bool StreamAssembler::feed(const unsigned char* data, std::size_t size, std::vector<PingRecord>& out) {
  if (size < kHeader || std::memcmp(data, "GSFB", 4) != 0 || get<std::uint16_t>(data + 4) != 1 || (get<std::uint16_t>(data + 34) & 2) != 0) {
    ++bad_blocks;
    return false;
  }
  const auto ch = static_cast<std::int64_t>(get<std::uint16_t>(data + 6));
  const auto n = static_cast<std::int64_t>(get<std::uint32_t>(data + 8));
  const auto first = static_cast<std::int64_t>(get<std::uint64_t>(data + 12));
  const auto start_ns = static_cast<std::int64_t>(get<std::uint64_t>(data + 36));
  if (ch != n_channels_ || size != kHeader + static_cast<std::size_t>(n * ch * sample_bytes_)) {
    ++bad_blocks;
    return false;
  }
  if (have_next_ && first != next_sample_) {
    lost_samples += std::max<std::int64_t>(first - next_sample_, 0);
    if (filled_ > 0) ++dropped_chunks;
    filled_ = 0;
  }
  // Gain of sample i of this block: zero from the last transmit the block marks until the blanking ends.
  const auto tx = get<std::int64_t>(data + 20);
  const double raw_fs = decimator_ ? plan_.input_rate : template_.samples.sample_rate;
  auto gain = [&](std::int64_t i) {
    if (!(blank_until_ > 0.0) || tx < 0 || first + i < tx) return 1.0;
    const double out = static_cast<double>(first + i - tx) / raw_fs - blank_until_;
    if (out >= blank_taper_) return 1.0;
    ++blanked_samples;
    return out <= 0.0 ? 0.0 : 0.5 - 0.5 * std::cos(3.14159265358979323846 * out / blank_taper_);
  };
  if (decimator_ && (!have_next_ || first != next_sample_)) {
    // The start, or a gap: the filter starts again and what was waiting is dropped.
    if (have_next_ && !pending_[0].empty()) ++dropped_chunks;
    decimator_->open(0.0);
    for (auto& v : pending_) v.clear();
    stretch_first_ = first;
    emitted_ = 0;
    start_ns_ = start_ns;
  }
  have_next_ = true;
  next_sample_ = first + n;
  const unsigned char* payload = data + kHeader;
  if (decimator_) {
    rows_.resize(static_cast<std::size_t>(n * ch));
    for (std::int64_t i = 0; i < n * ch; ++i) {
      const std::int32_t v = block_sample(payload + i * sample_bytes_, sample_bytes_);
      rows_[static_cast<std::size_t>(i)] = static_cast<float>(static_cast<double>(v) * volts_per_count_ * gain(i / ch));
    }
    decimator_->feed(rows_.data(), n, pending_);
    while (static_cast<std::int64_t>(pending_[0].size()) >= chunk_) {
      PingRecord rec = template_;
      rec.metadata.sequence = chunks++;
      const double raw_rate = rec.samples.sample_rate;
      rec.samples.resize(n_channels_, chunk_, SampleKind::ComplexBaseband);
      rec.samples.start_time = 0.0;
      rec.samples.center_frequency = plan_.carrier;
      rec.samples.sample_rate = plan_.output_rate;
      rec.samples.storage = StorageType::Float32;
      rec.samples.storage_scale = 1.0;
      rec.transmit.time_ns = start_ns_ + static_cast<std::int64_t>(std::llround(
          static_cast<double>(stretch_first_ + emitted_ * chunk_ * plan_.factor) / raw_rate * 1e9));
      for (std::int64_t c = 0; c < n_channels_; ++c) {
        auto& v = pending_[static_cast<std::size_t>(c)];
        std::copy_n(v.begin(), chunk_, rec.samples.channel_complex(c));
        v.erase(v.begin(), v.begin() + chunk_);
      }
      ++emitted_;
      out.push_back(std::move(rec));
    }
    return true;
  }
  for (std::int64_t i = 0; i < n; ++i) {
    if (filled_ == 0) {
      first_ = first + i;
      start_ns_ = start_ns;
    }
    for (std::int64_t c = 0; c < ch; ++c) {
      const std::int32_t v = block_sample(payload + (i * ch + c) * sample_bytes_, sample_bytes_);
      buffer_[static_cast<std::size_t>(c * chunk_ + filled_)] = static_cast<float>(static_cast<double>(v) * volts_per_count_ * (blank_until_ > 0.0 ? gain(i) : 1.0));
    }
    if (++filled_ == chunk_) {
      PingRecord rec = template_;
      rec.metadata.sequence = chunks++;
      rec.samples.resize(n_channels_, chunk_, SampleKind::RealPassband);
      rec.samples.start_time = 0.0;
      rec.transmit.time_ns = start_ns_ + static_cast<std::int64_t>(std::llround(static_cast<double>(first_) / rec.samples.sample_rate * 1e9));
      for (std::int64_t c = 0; c < n_channels_; ++c) {
        std::copy_n(buffer_.data() + c * chunk_, chunk_, rec.samples.channel(c));
      }
      out.push_back(std::move(rec));
      filled_ = 0;
    }
  }
  return true;
}

UdpSocket::UdpSocket(const std::string& host, int port, double timeout_s, std::int64_t buffer_bytes) {
  fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd_ < 0) throw std::runtime_error("UdpSocket: cannot create a socket");
  asked_ = buffer_bytes > 0 ? buffer_bytes : (8 << 20);
  int size = static_cast<int>(std::min<std::int64_t>(asked_, 0x7fffffff));
  ::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
  int got = 0;
  socklen_t got_len = sizeof(got);
  if (::getsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &got, &got_len) == 0) {
#ifdef __linux__
    granted_ = got / 2;        // Linux doubles the value for its own bookkeeping and reports the doubled one
#else
    granted_ = got;
#endif
  }
  timeval tv{};
  tv.tv_sec = static_cast<time_t>(timeout_s);
  tv.tv_usec = static_cast<suseconds_t>((timeout_s - std::floor(timeout_s)) * 1e6);
  ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) throw std::runtime_error("UdpSocket: bad address " + host);
  if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd_);
    throw std::runtime_error("UdpSocket: cannot bind " + host + ":" + std::to_string(port));
  }
  socklen_t len = sizeof(addr);
  ::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
  port_ = ntohs(addr.sin_port);
}

UdpSocket::~UdpSocket() {
  if (fd_ >= 0) ::close(fd_);
}

std::size_t UdpSocket::receive(std::vector<unsigned char>& buffer) {
  buffer.resize(65536);
  const auto n = ::recv(fd_, buffer.data(), buffer.size(), 0);
  buffer.resize(n > 0 ? static_cast<std::size_t>(n) : 0);
  return buffer.size();
}

bool UdpSocket::send_to(const std::string& host, int port, const std::string& text) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(port));
  // A text too long for a datagram is not sent at all: half a JSON line is worse than none (Report::datagrams splits one).
  if (text.size() > kMaxDatagram || ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    ++refused;
    return false;
  }
  ::sendto(fd_, text.data(), text.size(), 0, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));   // nobody listening is not an error
  return true;
}

void nav_to_array(const NavOptions& options, double* v) {
  const Vec3d& l = options.lever_arm;
  const Vec3d& a = options.mount_rpy;
  const bool lever = l.x != 0.0 || l.y != 0.0 || l.z != 0.0;
  const bool mount = a.x != 0.0 || a.y != 0.0 || a.z != 0.0;
  if (!lever && !mount) return;                 // the default: the fix as it came, bit for bit
  const Mat3d r = rotation_ned(v[3], v[4], v[5]);
  if (lever) {
    const Vec3d d = r * l;                      // the lever arm in north-east-down
    v[0] += d.x;
    v[1] += d.y;
    v[2] += d.z;
  }
  if (mount) {
    // Array axes -> unit axes -> NED; its Eq. 2 angles (live.py and navadapter.py euler_zyx read them the same way).
    const Mat3d m = r * rotation_ned(a.x, a.y, a.z);
    v[3] = std::atan2(m.m[2][1], m.m[2][2]);
    v[4] = -std::asin(std::clamp(m.m[2][0], -1.0, 1.0));
    v[5] = std::atan2(m.m[1][0], m.m[0][0]);
  }
}

bool Navigation::push(const std::string& json) {
  auto find = [&](const char* key, double& value) {
    const std::string k = std::string("\"") + key + "\"";
    const auto at = json.find(k);
    if (at == std::string::npos) return false;
    const auto colon = json.find(':', at + k.size());
    if (colon == std::string::npos) return false;
    const char* begin = json.c_str() + colon + 1;
    char* end = nullptr;
    const double v = std::strtod(begin, &end);
    if (end == begin) return false;               // null, true, false: not a number
    value = v;
    return true;
  };
  Fix fix{};
  const std::string k = "\"time_ns\"";
  const auto at = json.find(k);
  if (at == std::string::npos) return false;
  fix.time_ns = std::strtoll(json.c_str() + json.find(':', at) + 1, nullptr, 10);
  const char* keys[kFields] = {"x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "sound_speed", "water_depth", "altitude",
                               "bottom_lock", "roll_sigma", "pitch_sigma", "yaw_sigma", "position_sigma", "velocity_sigma",
                               "depth_sigma", "altitude_sigma", "water_depth_sigma", "sound_speed_sigma"};
  fix.velocity = true;
  for (int i = 0; i < kFields; ++i) {
    fix.v[i] = i < 9 ? 0.0 : std::nan("");
    const bool sent = find(keys[i], fix.v[i]);
    if (i >= 6 && i < 9 && !(sent && std::isfinite(fix.v[i]))) {
      // A velocity component not sent, or not a finite number, is 0 as before; the fix is marked (velocity_given).
      fix.v[i] = 0.0;
      fix.velocity = false;
    }
  }
  if (json.find("\"bottom_lock\": true") != std::string::npos || json.find("\"bottom_lock\":true") != std::string::npos) fix.v[kLock] = 1.0;
  if (json.find("\"bottom_lock\": false") != std::string::npos || json.find("\"bottom_lock\":false") != std::string::npos) fix.v[kLock] = 0.0;
  const bool lock = std::isnan(fix.v[kLock]) ? true : fix.v[kLock] != 0.0;
  // The water depth and the (depth, sound speed) samples from the unit's own position, as before the lever arm existed.
  fix.water = water_.update(fix.time_ns, fix.v[0], fix.v[1], fix.v[2] - opt_.surface_z, fix.v[kAltitude], lock, fix.v[kAltitudeSigma],
                            fix.v[kDepthSigma]);
  if (std::isfinite(fix.v[kSoundSpeed]) && fix.v[kSoundSpeed] > 0.0 &&
      (samples_.empty() || std::abs(fix.v[2] - samples_.back().z) >= opt_.profile_step_m)) {
    samples_.push_back({fix.time_ns, fix.v[0], fix.v[1], fix.v[2], fix.v[kSoundSpeed]});
    if (samples_.size() > 2000) samples_.pop_front();
  }
  nav_to_array(opt_, fix.v);                    // the pose kept is the array's (sonar-3rv)
  fixes_.push_back(fix);
  const auto keep_ns = static_cast<std::int64_t>(opt_.keep_s * 1e9);
  while (fixes_.size() > 1 && (fixes_.front().time_ns < fix.time_ns - keep_ns || fixes_.size() > 200000)) fixes_.pop_front();
  return true;
}

bool NavCounts::apply(Navigation& nav, PingRecord& record) {
  if (!nav.apply(record)) {
    ++without_nav;
    return false;
  }
  if (nav.velocity_given()) return false;
  return ++without_velocity == 1;
}

bool Navigation::apply(PingRecord& record) {
  if (fixes_.empty()) return false;
  const std::int64_t t = record.transmit.time_ns;
  double v[9] = {};
  std::size_t i = 0;
  while (i < fixes_.size() && fixes_[i].time_ns < t) ++i;
  if (i == 0 || i == fixes_.size()) {
    // The pose is the first nine of a fix's values (kFields in all): only those are copied.
    std::copy_n(std::begin(fixes_[std::min(i, fixes_.size() - 1)].v), 9, v);
    velocity_given_ = fixes_[std::min(i, fixes_.size() - 1)].velocity;
  } else {
    const Fix& a = fixes_[i - 1];
    const Fix& b = fixes_[i];
    velocity_given_ = a.velocity && b.velocity;
    const double w = static_cast<double>(t - a.time_ns) / static_cast<double>(std::max<std::int64_t>(b.time_ns - a.time_ns, 1));
    for (int k = 0; k < 9; ++k) v[k] = (1.0 - w) * a.v[k] + w * b.v[k];
    for (int k = 3; k < 6; ++k) v[k] = a.v[k] + w * std::atan2(std::sin(b.v[k] - a.v[k]), std::cos(b.v[k] - a.v[k]));   // the short way round
  }
  for (Pose* pose : {&record.geometry.tx_pose, &record.geometry.rx_pose}) {
    pose->position = {v[0], v[1], v[2]};
    pose->rpy = {v[3], v[4], v[5]};
    pose->velocity = {v[6], v[7], v[8]};
  }
  // The nearest fix's sound speed and water depth, and the fixes around the receive window.
  const Fix* nearest = &fixes_.front();
  for (const Fix& fx : fixes_) {
    if (std::llabs(fx.time_ns - t) < std::llabs(nearest->time_ns - t)) nearest = &fx;
  }
  const double* nv = nearest->v;
  std::string facts;
  auto fact = [&](const char* key, double value) {
    char text[64];
    std::snprintf(text, sizeof(text), "%s\"%s\": %.10g", facts.empty() ? "" : ", ", key, value);
    facts += text;
  };
  const bool measured = std::isfinite(nv[kSoundSpeed]) && nv[kSoundSpeed] > 0.0;
  if (measured) record.environment.sound_speed = nv[kSoundSpeed];
  if (measured) fact("sound_speed_measured", 1.0);
  if (std::isfinite(nv[kWaterDepth]) && nv[kWaterDepth] > 0.0) {
    record.environment.water_depth = nv[kWaterDepth];
    if (std::isfinite(nv[kWaterDepthSigma])) fact("water_depth_sigma_m", nv[kWaterDepthSigma]);
  } else if (std::isfinite(nearest->water.depth)) {
    // From the Doppler log's altitude and the vehicle's depth (held when bottom lock is lost).
    record.environment.water_depth = nearest->water.depth;
    fact("water_depth_sigma_m", nearest->water.sigma);
    fact("water_depth_held_s", nearest->water.held ? nearest->water.held_s : 0.0);
  }
  const char* names[] = {"roll_sigma", "pitch_sigma", "yaw_sigma", "position_sigma", "velocity_sigma", "depth_sigma", "altitude_sigma",
                         nullptr, "sound_speed_sigma"};
  for (int q = kRollSigma; q < kFields; ++q) {
    if (names[q - kRollSigma] != nullptr && std::isfinite(nv[q])) fact(names[q - kRollSigma], nv[q]);
  }
  const Samples& s = record.samples;
  const auto window_ns = static_cast<std::int64_t>((s.start_time + static_cast<double>(s.n_samples) / s.sample_rate) * 1e9);
  std::vector<InsSample> history;
  std::vector<DvlSample> dvl;
  std::vector<DepthSample> depth;
  for (const Fix& fx : fixes_) {
    const auto margin_ns = static_cast<std::int64_t>(opt_.history_margin_s * 1e9);
    if (fx.time_ns < t - margin_ns || fx.time_ns > t + window_ns + margin_ns) continue;
    InsSample a{};
    a.t = static_cast<double>(fx.time_ns - t) * 1e-9;
    a.position = {fx.v[0], fx.v[1], fx.v[2]};
    a.rpy = {fx.v[3], fx.v[4], fx.v[5]};
    a.velocity = {fx.v[6], fx.v[7], fx.v[8]};
    history.push_back(a);
    depth.push_back({a.t, fx.v[2] - opt_.surface_z});
    if (std::isfinite(fx.v[kAltitude]) || !std::isnan(fx.v[kLock])) {
      // Velocity over ground in the vehicle's own axes, and the altitude (NaN without bottom lock).
      Pose p{};
      p.rpy = a.rpy;
      const bool lock = (std::isnan(fx.v[kLock]) || fx.v[kLock] != 0.0) && std::isfinite(fx.v[kAltitude]) && fx.v[kAltitude] > 0.0;
      dvl.push_back({a.t, {dot(a.velocity, world_direction(p, {1.0, 0.0, 0.0})), dot(a.velocity, world_direction(p, {0.0, 1.0, 0.0})),
                           dot(a.velocity, world_direction(p, {0.0, 0.0, 1.0}))}, lock ? fx.v[kAltitude] : std::nan("")});
    }
  }
  if (history.size() > 1) {            // a fixed pose has one fix and no history
    record.navigation.ins = std::move(history);
    record.navigation.depth = std::move(depth);
    if (!dvl.empty()) record.navigation.dvl = std::move(dvl);
  }
  if (!samples_.empty()) {
    std::string rows = "[";
    for (const Sample& q : samples_) {
      char text[160];
      std::snprintf(text, sizeof(text), "%s[%.6f, %.3f, %.3f, %.3f, %.4f]", rows.size() > 1 ? ", " : "", static_cast<double>(q.time_ns - t) * 1e-9,
                    q.x, q.y, q.z, q.speed);
      rows += text;
    }
    facts += std::string(facts.empty() ? "" : ", ") + "\"sound_speed_samples\": " + rows + "]";
    samples_.clear();
  }
  if (!facts.empty()) extra_set(record.metadata.extra_json, "nav", "{" + facts + "}");
  return true;
}

}  // namespace sonar::live
