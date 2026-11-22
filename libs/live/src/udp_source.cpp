#include "sonar/live/udp_source.hpp"

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

bool PingAssembler::feed(const unsigned char* data, std::size_t size, std::vector<PingRecord>& out) {
  if (size < kHeader || std::memcmp(data, "GSFB", 4) != 0 || get<std::uint16_t>(data + 4) != 1) {
    ++bad_blocks;
    return false;
  }
  const auto ch = static_cast<std::int64_t>(get<std::uint16_t>(data + 6));
  const auto n = static_cast<std::int64_t>(get<std::uint32_t>(data + 8));
  const auto first = static_cast<std::int64_t>(get<std::uint64_t>(data + 12));
  const auto tx = get<std::int64_t>(data + 20);
  const auto ping_index = static_cast<std::int64_t>(get<std::uint32_t>(data + 28));
  const auto flags = get<std::uint16_t>(data + 34);
  const auto start_ns = static_cast<std::int64_t>(get<std::uint64_t>(data + 36));
  if (ch != n_channels_ || size != kHeader + static_cast<std::size_t>(n * ch * sample_bytes_)) {
    ++bad_blocks;
    return false;
  }
  if (have_next_ && first != next_sample_) {
    // A gap (or a restart). Whatever ping it cuts through is lost.
    lost_samples += std::max<std::int64_t>(first - next_sample_, 0);
    if (open_) {
      const std::int64_t w0 = tx_ + window_start_;
      if (first > w0 + filled_ || first < next_sample_) {
        ++dropped_pings;
        done_ping_ = index_;
        have_done_ = true;
        open_ = false;
      }
    }
  }
  have_next_ = true;
  next_sample_ = first + n;
  if (flags & 1) ++overflow_blocks;
  if (tx >= 0 && !(have_done_ && ping_index == done_ping_) && (!open_ || index_ != ping_index)) {
    if (open_) ++dropped_pings;                    // the previous window never completed
    if (first > tx + window_start_) {              // joined too late for this ping
      ++dropped_pings;
      done_ping_ = ping_index;
      have_done_ = true;
      open_ = false;
    } else {
      open_ = true;
      tx_ = tx;
      index_ = ping_index;
      start_ns_ = start_ns;
      filled_ = 0;
      if (decimator_) decimator_->open();
      else buffer_.assign(static_cast<std::size_t>(window_length_ * n_channels_), 0);
    }
  }
  if (!open_) return true;
  const std::int64_t w0 = tx_ + window_start_;
  const std::int64_t a = std::max(first, w0 + filled_), b = std::min(first + n, w0 + window_length_);
  if (b > a) {
    if (a != w0 + filled_) {
      ++dropped_pings;
      done_ping_ = index_;
      have_done_ = true;
      open_ = false;
      return true;
    }
    const unsigned char* payload = data + kHeader;
    if (decimator_) rows_.resize(static_cast<std::size_t>((b - a) * ch));
    for (std::int64_t i = a; i < b; ++i) {
      for (std::int64_t c = 0; c < ch; ++c) {
        const std::int32_t v = block_sample(payload + ((i - first) * ch + c) * sample_bytes_, sample_bytes_);
        if (decimator_) rows_[static_cast<std::size_t>((i - a) * ch + c)] = static_cast<float>(static_cast<double>(v) * volts_per_count_);
        else buffer_[static_cast<std::size_t>((i - w0) * ch + c)] = v;
      }
    }
    if (decimator_) decimator_->feed(rows_.data(), b - a);    // filtered and decimated as it arrives: the raw ping is never held
    filled_ = b - w0;
  }
  if (filled_ == window_length_) {
    PingRecord rec = template_;
    rec.metadata.sequence = index_;
    rec.transmit.time_ns = start_ns_ + static_cast<std::int64_t>(std::llround(static_cast<double>(tx_) / rec.samples.sample_rate * 1e9));
    if (decimator_) {
      decimator_->finish(rec);
      out.push_back(std::move(rec));
      done_ping_ = index_;
      have_done_ = true;
      open_ = false;
      return true;
    }
    rec.samples.resize(n_channels_, window_length_, SampleKind::RealPassband);
    rec.samples.start_time = static_cast<double>(window_start_) / rec.samples.sample_rate;
    for (std::int64_t c = 0; c < n_channels_; ++c) {
      float* dst = rec.samples.channel(c);
      for (std::int64_t i = 0; i < window_length_; ++i) {
        dst[i] = static_cast<float>(static_cast<double>(buffer_[static_cast<std::size_t>(i * n_channels_ + c)]) * volts_per_count_);
      }
    }
    out.push_back(std::move(rec));
    done_ping_ = index_;
    have_done_ = true;
    open_ = false;
  }
  return true;
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
  if (size < kHeader || std::memcmp(data, "GSFB", 4) != 0 || get<std::uint16_t>(data + 4) != 1) {
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
  fix.water = water_.update(fix.time_ns, fix.v[0], fix.v[1], fix.v[2] - opt_.surface_z, fix.v[kAltitude], lock, fix.v[kAltitudeSigma],
                            fix.v[kDepthSigma]);
  fixes_.push_back(fix);
  const auto keep_ns = static_cast<std::int64_t>(opt_.keep_s * 1e9);
  while (fixes_.size() > 1 && (fixes_.front().time_ns < fix.time_ns - keep_ns || fixes_.size() > 200000)) fixes_.pop_front();
  if (std::isfinite(fix.v[kSoundSpeed]) && fix.v[kSoundSpeed] > 0.0 &&
      (samples_.empty() || std::abs(fix.v[2] - samples_.back().z) >= opt_.profile_step_m)) {
    samples_.push_back({fix.time_ns, fix.v[0], fix.v[1], fix.v[2], fix.v[kSoundSpeed]});
    if (samples_.size() > 2000) samples_.pop_front();
  }
  return true;
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
