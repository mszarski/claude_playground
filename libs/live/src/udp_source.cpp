#include "sonar/live/udp_source.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
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

PingAssembler::PingAssembler(const PingRecord& template_record) : template_(template_record) {
  const Samples& s = template_.samples;
  if (s.is_complex()) throw std::invalid_argument("PingAssembler: the template must describe real ADC samples");
  window_start_ = static_cast<std::int64_t>(std::llround(s.start_time * s.sample_rate));
  window_length_ = s.n_samples;
  n_channels_ = s.n_channels;
  const double full_scale = template_.metadata.adc_full_scale;
  volts_per_count_ = (std::isfinite(full_scale) ? full_scale : 1.0) / 8388608.0;
  template_.samples.data.clear();
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
  if (ch != n_channels_ || size != kHeader + static_cast<std::size_t>(n * ch * 3)) {
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
      buffer_.assign(static_cast<std::size_t>(window_length_ * n_channels_), 0);
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
    for (std::int64_t i = a; i < b; ++i) {
      for (std::int64_t c = 0; c < ch; ++c) {
        const unsigned char* q = payload + ((i - first) * ch + c) * 3;
        std::int32_t v = static_cast<std::int32_t>(q[0]) | (static_cast<std::int32_t>(q[1]) << 8) | (static_cast<std::int32_t>(q[2]) << 16);
        if (v & 0x800000) v -= 0x1000000;        // sign-extend 24 bits
        buffer_[static_cast<std::size_t>((i - w0) * ch + c)] = v;
      }
    }
    filled_ = b - w0;
  }
  if (filled_ == window_length_) {
    PingRecord rec = template_;
    rec.metadata.sequence = index_;
    rec.transmit.time_ns = start_ns_ + static_cast<std::int64_t>(std::llround(static_cast<double>(tx_) / rec.samples.sample_rate * 1e9));
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

StreamAssembler::StreamAssembler(const PingRecord& template_record, std::int64_t chunk_samples)
    : template_(template_record), chunk_(chunk_samples) {
  if (template_.samples.is_complex()) throw std::invalid_argument("StreamAssembler: the template must describe real ADC samples");
  if (chunk_ < 1) throw std::invalid_argument("StreamAssembler: chunk_samples must be positive");
  n_channels_ = template_.samples.n_channels;
  const double full_scale = template_.metadata.adc_full_scale;
  volts_per_count_ = (std::isfinite(full_scale) ? full_scale : 1.0) / 8388608.0;
  template_.samples.data.clear();
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
  if (ch != n_channels_ || size != kHeader + static_cast<std::size_t>(n * ch * 3)) {
    ++bad_blocks;
    return false;
  }
  if (have_next_ && first != next_sample_) {
    lost_samples += std::max<std::int64_t>(first - next_sample_, 0);
    if (filled_ > 0) ++dropped_chunks;
    filled_ = 0;
  }
  have_next_ = true;
  next_sample_ = first + n;
  const unsigned char* payload = data + kHeader;
  for (std::int64_t i = 0; i < n; ++i) {
    if (filled_ == 0) {
      first_ = first + i;
      start_ns_ = start_ns;
    }
    for (std::int64_t c = 0; c < ch; ++c) {
      const unsigned char* q = payload + (i * ch + c) * 3;
      std::int32_t v = static_cast<std::int32_t>(q[0]) | (static_cast<std::int32_t>(q[1]) << 8) | (static_cast<std::int32_t>(q[2]) << 16);
      if (v & 0x800000) v -= 0x1000000;
      buffer_[static_cast<std::size_t>(c * chunk_ + filled_)] = static_cast<float>(static_cast<double>(v) * volts_per_count_);
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

UdpSocket::UdpSocket(const std::string& host, int port, double timeout_s) {
  fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd_ < 0) throw std::runtime_error("UdpSocket: cannot create a socket");
  int size = 8 << 20;
  ::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
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

void UdpSocket::send_to(const std::string& host, int port, const std::string& text) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<std::uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) return;
  ::sendto(fd_, text.data(), std::min<std::size_t>(text.size(), 65000), 0, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
}

bool Navigation::push(const std::string& json) {
  auto find = [&](const char* key, double& value) {
    const std::string k = std::string("\"") + key + "\"";
    const auto at = json.find(k);
    if (at == std::string::npos) return false;
    const auto colon = json.find(':', at + k.size());
    if (colon == std::string::npos) return false;
    value = std::strtod(json.c_str() + colon + 1, nullptr);
    return true;
  };
  Fix fix{};
  const std::string k = "\"time_ns\"";
  const auto at = json.find(k);
  if (at == std::string::npos) return false;
  fix.time_ns = std::strtoll(json.c_str() + json.find(':', at) + 1, nullptr, 10);
  const char* keys[9] = {"x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"};
  for (int i = 0; i < 9; ++i) find(keys[i], fix.v[i]);
  fixes_.push_back(fix);
  if (fixes_.size() > 600) fixes_.pop_front();
  return true;
}

bool Navigation::apply(PingRecord& record) const {
  if (fixes_.empty()) return false;
  const std::int64_t t = record.transmit.time_ns;
  double v[9];
  std::size_t i = 0;
  while (i < fixes_.size() && fixes_[i].time_ns < t) ++i;
  if (i == 0 || i == fixes_.size()) {
    std::copy(std::begin(fixes_[std::min(i, fixes_.size() - 1)].v), std::end(fixes_[std::min(i, fixes_.size() - 1)].v), v);
  } else {
    const Fix& a = fixes_[i - 1];
    const Fix& b = fixes_[i];
    const double w = static_cast<double>(t - a.time_ns) / static_cast<double>(std::max<std::int64_t>(b.time_ns - a.time_ns, 1));
    for (int k = 0; k < 9; ++k) v[k] = (1.0 - w) * a.v[k] + w * b.v[k];
    for (int k = 3; k < 6; ++k) v[k] = a.v[k] + w * std::atan2(std::sin(b.v[k] - a.v[k]), std::cos(b.v[k] - a.v[k]));   // the short way round
  }
  for (Pose* pose : {&record.geometry.tx_pose, &record.geometry.rx_pose}) {
    pose->position = {v[0], v[1], v[2]};
    pose->rpy = {v[3], v[4], v[5]};
    pose->velocity = {v[6], v[7], v[8]};
  }
  return true;
}

}  // namespace sonar::live
