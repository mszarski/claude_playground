// The C++ twin of python/gpusonar/sources/board_merge.py: the same rules, in the same order, so that both engines merge
// the same board blocks into the same bytes with the same counts (tests/python/test_board_streams.py). Compiled with
// floating-point contraction off (CMakeLists.txt): the alignment's arithmetic must round as Python's does.
#include "sonar/live/board_merge.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "sonar/live/udp_source.hpp"

namespace sonar::live {
namespace {

constexpr std::size_t kHeader = 128;
constexpr std::size_t kMaxLeft = 8, kMaxDone = 64, kMaxNamed = 8;

template <typename T>
T get(const unsigned char* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}
template <typename T>
void put(std::vector<unsigned char>& b, std::size_t at, T v) {
  std::memcpy(b.data() + at, &v, sizeof(T));
}

}  // namespace

std::int64_t block_samples_for(std::int64_t n_channels, std::int64_t sample_bytes) {
  return std::max<std::int64_t>(kStreamPayloadBytes / (n_channels * sample_bytes), 1);
}

std::int64_t half_up(double x) { return static_cast<std::int64_t>(std::floor(x + 0.5)); }

std::vector<unsigned char> encode_block(const std::int32_t* counts, std::int64_t n, std::int64_t n_channels, std::int64_t first,
                                        std::int64_t tx, std::int64_t index, std::uint16_t gain, std::uint16_t flags,
                                        std::int64_t start_ns, std::int64_t sample_bytes, int board) {
  std::vector<unsigned char> b(kHeader + static_cast<std::size_t>(n * n_channels * sample_bytes), 0);
  std::memcpy(b.data(), "GSFB", 4);
  put<std::uint16_t>(b, 4, 1);
  put<std::uint16_t>(b, 6, static_cast<std::uint16_t>(n_channels));
  put<std::uint32_t>(b, 8, static_cast<std::uint32_t>(n));
  put<std::uint64_t>(b, 12, static_cast<std::uint64_t>(first));
  put<std::int64_t>(b, 20, tx);
  put<std::uint32_t>(b, 28, static_cast<std::uint32_t>(index));
  put<std::uint16_t>(b, 32, gain);
  put<std::uint16_t>(b, 34, static_cast<std::uint16_t>(board >= 0 ? (flags | kFlagBoard) : flags));
  put<std::uint64_t>(b, 36, static_cast<std::uint64_t>(start_ns));
  if (board >= 0) put<std::uint16_t>(b, 44, static_cast<std::uint16_t>(board));
  unsigned char* q = b.data() + kHeader;
  for (std::int64_t k = 0; k < n * n_channels; ++k) {
    const auto v = static_cast<std::uint32_t>(counts[k]);
    for (std::int64_t i = 0; i < sample_bytes; ++i) *q++ = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
  }
  return b;
}

std::vector<BoardSpec> parse_board_channels(const std::string& text) {
  std::vector<BoardSpec> out;
  std::stringstream s(text);
  std::string item;
  while (std::getline(s, item, ',')) {
    if (item.empty()) continue;
    BoardSpec b;
    char c1 = 0, c2 = 0;
    long long id = 0, first = 0, count = 0;
    std::istringstream in(item);
    if (!(in >> id >> c1 >> first >> c2 >> count) || c1 != ':' || c2 != ':' || !(in >> std::ws).eof()) {
      throw std::invalid_argument("stream_board_channels: " + item + ": id:first:count");
    }
    b.id = static_cast<int>(id);
    b.first = first;
    b.count = count;
    out.push_back(b);
  }
  return out;
}

std::vector<BoardSpec> boards_from_template(const PingRecord& record) {
  const auto& cb = record.metadata.channel_board;
  std::vector<BoardSpec> out;
  for (std::size_t c = 0; c < cb.size(); ++c) {
    const int b = cb[c];
    if (!out.empty() && out.back().id == b) {
      ++out.back().count;
      continue;
    }
    for (const BoardSpec& x : out) {
      if (x.id == b) throw std::invalid_argument("stream_boards: the template's board " + std::to_string(b) + " is not one run of channels; give stream_board_channels");
    }
    out.push_back({b, static_cast<std::int64_t>(c), 1});
  }
  if (out.empty()) out.push_back({0, 0, static_cast<std::int64_t>(record.samples.n_channels)});
  return out;
}

BoardMerger::BoardMerger(const BoardMergeOptions& o, std::int64_t n_channels, double sample_rate)
    : n_channels_(n_channels), sample_bytes_(o.sample_bytes), sample_rate_(sample_rate), zero_(o.zero) {
  const std::string where = "boards.";
  block_full_scale(sample_bytes_);   // 2, 3 or 4, or an error
  if (o.boards.empty()) throw std::invalid_argument(where + "channels: no board");
  std::vector<int> on(static_cast<std::size_t>(std::max<std::int64_t>(n_channels, 0)), 0);
  for (const BoardSpec& b : o.boards) {
    if (b.id < 0 || b.id >= (1 << 16)) {
      throw std::invalid_argument(where + "channels: board id " + std::to_string(b.id) + " does not fit the block header (0 to 65535)");
    }
    for (const Board& x : boards_) {
      if (x.spec.id == b.id) throw std::invalid_argument(where + "channels: board " + std::to_string(b.id) + " given twice");
    }
    if (b.count < 1 || b.first < 0 || b.first + b.count > n_channels) {
      throw std::invalid_argument(where + "channels: board " + std::to_string(b.id) + ": channels " + std::to_string(b.first) + " to " +
                                  std::to_string(b.first + b.count - 1) + " are not channels of the array's " + std::to_string(n_channels));
    }
    for (std::int64_t c = b.first; c < b.first + b.count; ++c) ++on[static_cast<std::size_t>(c)];
    Board board;
    board.spec = b;
    boards_.push_back(board);
  }
  for (std::size_t c = 0; c < on.size(); ++c) {
    if (on[c] != 1) {
      throw std::invalid_argument(where + "channels: channel " + std::to_string(c) + " is on " + (on[c] == 0 ? "no" : "more than one") + " board");
    }
  }
  const int reference = o.reference < 0 ? boards_.front().spec.id : o.reference;
  ref_ = boards_.size();
  for (std::size_t i = 0; i < boards_.size(); ++i) {
    if (boards_[i].spec.id == reference) ref_ = i;
  }
  if (ref_ == boards_.size()) throw std::invalid_argument(where + "reference: board " + std::to_string(reference) + " is not in the board map");
  if (!std::isfinite(o.wait_s) || o.wait_s < 0.0) throw std::invalid_argument(where + "wait_s: a number of seconds, 0 or more");
  wait_ = half_up(o.wait_s * sample_rate_);
  block_samples_ = o.block_samples > 0 ? o.block_samples : block_samples_for(n_channels_, sample_bytes_);
}

int BoardMerger::reference() const { return boards_[ref_].spec.id; }

bool BoardMerger::refuse(const std::string& message, Board* board) {
  ++bad_blocks;
  if (board != nullptr) ++board->bad;
  last_error = message;
  return false;
}

std::int64_t BoardMerger::offset(std::int64_t start_ns, std::int64_t /*anchor*/) const {
  const double d = static_cast<double>(boards_[ref_].start_ns - start_ns) * 1e-9 * sample_rate_;
  return -half_up(d);
}

std::vector<BoardMerger::Span> BoardMerger::spans(const Board& b, std::int64_t anchor) const {
  std::vector<Span> out;
  out.reserve(b.blocks.size());
  for (const auto& kv : b.blocks) {
    const std::int64_t f = kv.first.second + offset(kv.first.first, anchor);
    out.push_back({f, f + kv.second.n, kv.first});
  }
  std::sort(out.begin(), out.end(), [](const Span& x, const Span& y) { return std::tie(x.f, x.e, x.key) < std::tie(y.f, y.e, y.key); });
  return out;
}

bool BoardMerger::frontier(const Board& b, std::int64_t anchor, std::int64_t& out) const {
  if (!b.have_frontier) return false;
  out = b.frontier + offset(b.start_ns, anchor);
  return true;
}

bool BoardMerger::feed(const unsigned char* data, std::size_t size, std::vector<std::vector<unsigned char>>& out) {
  if (size < kHeader) return refuse("block shorter than its header");
  if (std::memcmp(data, "GSFB", 4) != 0) return refuse("bad magic");
  const auto version = get<std::uint16_t>(data + 4);
  if (version != 1) return refuse("unsupported block version " + std::to_string(version));
  const auto ch = static_cast<std::int64_t>(get<std::uint16_t>(data + 6));
  const auto n = static_cast<std::int64_t>(get<std::uint32_t>(data + 8));
  if (size != kHeader + static_cast<std::size_t>(n * ch * sample_bytes_)) {
    return refuse("block length does not match its header for samples of " + std::to_string(sample_bytes_) + " bytes");
  }
  const auto flags = get<std::uint16_t>(data + 34);
  if (!(flags & kFlagBoard)) return refuse("a whole-array block (no board id) where per-board streams are merged");
  const int bid = get<std::uint16_t>(data + 44);
  Board* board = nullptr;
  for (Board& b : boards_) {
    if (b.spec.id == bid) board = &b;
  }
  if (board == nullptr) {
    std::string ids;
    for (const Board& b : boards_) ids += (ids.empty() ? "" : ", ") + std::to_string(b.spec.id);
    return refuse("board " + std::to_string(bid) + " is not in the board map (boards " + ids + ")");
  }
  if (ch != board->spec.count) {
    return refuse("board " + std::to_string(bid) + " sent " + std::to_string(ch) + " channels, the board map gives it " +
                  std::to_string(board->spec.count), board);
  }
  const auto ufirst = get<std::uint64_t>(data + 12);
  const auto tx = get<std::int64_t>(data + 20);
  if (n < 1 || ufirst >= static_cast<std::uint64_t>(kMaxSampleIndex) || tx < -1 || tx >= kMaxSampleIndex ||
      tx >= static_cast<std::int64_t>(ufirst) + n) {
    return refuse("board " + std::to_string(bid) + ": a block without samples, or its sample index or transmit out of range", board);
  }
  const auto first = static_cast<std::int64_t>(ufirst);
  const auto ns = static_cast<std::int64_t>(get<std::uint64_t>(data + 36));
  Board& ref = boards_[ref_];
  if (!board->started) {
    board->started = true;
    board->start_ns = ns;
  } else if (ns != board->start_ns) {
    if (std::find(board->left.begin(), board->left.end(), ns) != board->left.end()) {   // a capture this board has left
      ++board->repeated_blocks;
      return true;
    }
    ++board->restarts;
    board->left.push_back(board->start_ns);
    if (board->left.size() > kMaxLeft) board->left.erase(board->left.begin());
    if (board == &ref) {
      decide(true, out);               // the old capture's spans, with what the boards gave
      pending_.clear();
      done_.clear();
      named_.clear();
      have_ref_frontier_ = have_decided_ = false;
    }
    board->start_ns = ns;
    board->have_frontier = false;
  }
  ++board->n_blocks;
  // Samples, sign-extended, [n][count].
  std::vector<std::int32_t> counts(static_cast<std::size_t>(n * ch));
  const unsigned char* q = data + kHeader;
  for (std::size_t k = 0; k < counts.size(); ++k, q += sample_bytes_) counts[k] = block_sample(q, sample_bytes_);
  if (board == &ref) {
    if (std::find(done_.begin(), done_.end(), first) != done_.end() ||
        std::any_of(pending_.begin(), pending_.end(), [&](const Pending& p) { return p.first == first; })) {
      ++board->repeated_blocks;
      return true;
    }
    Pending p;
    p.first = first;
    p.n = n;
    p.tx = tx;
    p.index = static_cast<std::int64_t>(get<std::uint32_t>(data + 28));
    p.gain = get<std::uint16_t>(data + 32);
    p.flags = flags & kFlagOverflow;
    p.start_ns = ns;
    p.counts = std::move(counts);
    p.anchor = tx >= 0 ? tx : first;
    std::size_t at = pending_.size();
    while (at > 0 && pending_[at - 1].first > first) --at;
    anchor_ = p.anchor;
    if (tx >= 0) {
      const std::pair<std::int64_t, std::int64_t> t{tx, p.index};
      if (std::find(named_.begin(), named_.end(), t) == named_.end()) {
        named_.push_back(t);
        std::sort(named_.begin(), named_.end());
        if (named_.size() > kMaxNamed) named_.erase(named_.begin());
      }
    }
    pending_.insert(pending_.begin() + static_cast<std::ptrdiff_t>(at), std::move(p));
    ref_frontier_ = have_ref_frontier_ ? std::max(ref_frontier_, first + n) : first + n;
    have_ref_frontier_ = true;
  } else {
    const std::pair<std::int64_t, std::int64_t> key{ns, first};
    if (board->blocks.count(key) != 0) {
      ++board->repeated_blocks;
      return true;
    }
    if (!wanted(*board, ns, first, n)) {
      ++board->late_blocks;
      return true;
    }
    Block blk;
    blk.n = n;
    blk.counts = std::move(counts);
    blk.flags = flags & kFlagOverflow;
    board->blocks.emplace(key, std::move(blk));
    board->frontier = board->have_frontier ? std::max(board->frontier, first + n) : first + n;
    board->have_frontier = true;
  }
  decide(false, out);
  prune();
  return true;
}

void BoardMerger::finish(std::vector<std::vector<unsigned char>>& out) {
  decide(true, out);
  prune();
}

bool BoardMerger::wanted(const Board& b, std::int64_t ns, std::int64_t first, std::int64_t n) const {
  if (!boards_[ref_].started) return true;
  const std::int64_t f = first + offset(ns, anchor_), e = f + n;
  for (const Pending& p : pending_) {
    if (std::min(e, p.first + p.n) > std::max(f, p.first)) return true;
  }
  std::int64_t front = 0;
  if (frontier(b, anchor_, front) && e <= front - wait_) return false;
  return !(have_decided_ && e <= decided_to_);
}

std::pair<std::int64_t, std::uint16_t> BoardMerger::cover(Board& b, std::int64_t a, std::int64_t e, std::int64_t anchor, std::int32_t* dest) {
  std::int64_t pos = a, covered = 0;
  std::uint16_t flags = 0;
  const std::int64_t count = b.spec.count;
  for (const Span& s : spans(b, anchor)) {
    if (s.f >= e) break;
    const std::int64_t x0 = std::max(s.f, pos), x1 = std::min(s.e, e);
    if (x1 <= x0) continue;
    Block& blk = b.blocks.at(s.key);
    if (dest != nullptr) {
      std::copy(blk.counts.begin() + (x0 - s.f) * count, blk.counts.begin() + (x1 - s.f) * count, dest + (x0 - a) * count);
      if (!blk.used) {
        blk.used = true;
        ++b.used_blocks;
      }
    }
    covered += x1 - x0;
    flags = static_cast<std::uint16_t>(flags | blk.flags);
    pos = x1;
  }
  return {covered, flags};
}

int BoardMerger::status(const Pending& p, Board& b, bool force) {   // 0 have, 1 wait, 2 missing
  const std::int64_t end = p.first + p.n;
  if (b.started && cover(b, p.first, end, p.anchor, nullptr).first == p.n) return 0;
  std::int64_t front = 0;
  const bool have_front = b.started && frontier(b, p.anchor, front);
  if (force || ref_frontier_ >= end + wait_ || (have_front && front >= end + wait_)) return 2;
  return 1;
}

void BoardMerger::decide(bool force, std::vector<std::vector<unsigned char>>& out) {
  while (!pending_.empty()) {
    bool wait = false;
    for (std::size_t i = 0; i < boards_.size() && !wait; ++i) {
      if (i != ref_ && status(pending_.front(), boards_[i], force) == 1) wait = true;
    }
    if (wait) break;
    Pending p = std::move(pending_.front());
    pending_.erase(pending_.begin());
    done_.push_back(p.first);
    if (done_.size() > kMaxDone) done_.erase(done_.begin());
    const std::int64_t end = p.first + p.n;
    decided_to_ = have_decided_ ? std::max(decided_to_, end) : end;
    have_decided_ = true;
    bool short_ = false;
    for (std::size_t i = 0; i < boards_.size(); ++i) {
      if (i == ref_) continue;
      Board& b = boards_[i];
      const std::int64_t miss = p.n - (b.started ? cover(b, p.first, end, p.anchor, nullptr).first : 0);
      if (miss != 0) {
        b.missing_samples += miss;
        short_ = true;
      }
    }
    if (short_ && !zero_) {
      ++dropped_blocks;
      continue;
    }
    if (short_) ++zero_filled_blocks;
    ++merged_blocks;
    ++boards_[ref_].used_blocks;
    emit(p, out);
  }
}

void BoardMerger::emit(const Pending& p, std::vector<std::vector<unsigned char>>& out) {
  const Board& ref = boards_[ref_];
  for (std::int64_t k0 = 0; k0 < p.n; k0 += block_samples_) {
    const std::int64_t k1 = std::min(k0 + block_samples_, p.n), m = k1 - k0;
    const std::int64_t p0 = p.first + k0, p1 = p.first + k1;
    std::vector<std::int32_t> counts(static_cast<std::size_t>(m * n_channels_), 0);
    for (std::int64_t r = 0; r < m; ++r) {
      std::copy(p.counts.begin() + (k0 + r) * ref.spec.count, p.counts.begin() + (k0 + r + 1) * ref.spec.count,
                counts.begin() + r * n_channels_ + ref.spec.first);
    }
    std::uint16_t flags = p.flags;
    for (std::size_t i = 0; i < boards_.size(); ++i) {
      if (i == ref_) continue;
      Board& b = boards_[i];
      std::vector<std::int32_t> part(static_cast<std::size_t>(m * b.spec.count), 0);
      flags = static_cast<std::uint16_t>(flags | cover(b, p0, p1, p.anchor, part.data()).second);
      for (std::int64_t r = 0; r < m; ++r) {
        std::copy(part.begin() + r * b.spec.count, part.begin() + (r + 1) * b.spec.count, counts.begin() + r * n_channels_ + b.spec.first);
      }
    }
    std::int64_t tx = -1, index = p.index;
    if (p.tx >= 0 && p.tx < p1) {
      tx = p.tx;
    } else {
      for (const auto& t : named_) {
        if (t.first < p1) {
          tx = t.first;
          index = t.second;
        }
      }
    }
    out.push_back(encode_block(counts.data(), m, n_channels_, p0, tx, index, p.gain, flags, p.start_ns, sample_bytes_));
  }
}

void BoardMerger::prune() {
  if (!boards_[ref_].started) return;
  for (std::size_t i = 0; i < boards_.size(); ++i) {
    if (i == ref_) continue;
    Board& b = boards_[i];
    if (b.blocks.empty()) continue;
    std::int64_t front = 0;
    const bool have_front = frontier(b, anchor_, front);
    for (const Span& s : spans(b, anchor_)) {
      bool needed = false;
      for (const Pending& p : pending_) needed = needed || std::min(s.e, p.first + p.n) > std::max(s.f, p.first);
      if (needed) continue;
      if ((have_front && s.e <= front - wait_) || (have_decided_ && s.e <= decided_to_)) b.blocks.erase(s.key);
    }
  }
}

std::string BoardMerger::stats_json() const {
  std::ostringstream s;
  s << "{\"boards\": {";
  for (std::size_t i = 0; i < boards_.size(); ++i) {
    const Board& b = boards_[i];
    s << (i ? ", " : "") << "\"" << b.spec.id << "\": {\"blocks\": " << b.n_blocks << ", \"used_blocks\": " << b.used_blocks
      << ", \"late_blocks\": " << b.late_blocks << ", \"repeated_blocks\": " << b.repeated_blocks << ", \"restarts\": " << b.restarts
      << ", \"missing_samples\": " << b.missing_samples << ", \"bad_blocks\": " << b.bad << ", \"start_ns\": ";
    if (b.started) s << b.start_ns;
    else s << "null";
    s << "}";
  }
  s << "}, \"merged_blocks\": " << merged_blocks << ", \"dropped_blocks\": " << dropped_blocks << ", \"zero_filled_blocks\": "
    << zero_filled_blocks << ", \"bad_blocks\": " << bad_blocks << ", \"pending\": " << pending_.size() << "}";
  return s.str();
}

}  // namespace sonar::live
