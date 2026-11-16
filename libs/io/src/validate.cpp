#include "sonar/io/validate.hpp"

#include <cmath>
#include <exception>
#include <sstream>

#include "sonar/io/record_file.hpp"

namespace sonar::io {
namespace {

class Checker {
 public:
  explicit Checker(ValidationReport& report, std::string prefix = "")
      : report_(report), prefix_(std::move(prefix)) {}

  void error(const std::string& field, const std::string& message) {
    report_.issues.push_back({Severity::Error, prefix_ + field, message});
  }
  void warn(const std::string& field, const std::string& message) {
    report_.issues.push_back({Severity::Warning, prefix_ + field, message});
  }
  bool finite(const std::string& field, double v) {
    if (std::isfinite(v)) return true;
    error(field, "must be finite");
    return false;
  }
  bool finite(const std::string& field, const Vec3d& v) {
    if (std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)) return true;
    error(field, "must be finite");
    return false;
  }
  void unit(const std::string& field, const Vec3d& v) {
    if (!finite(field, v)) return;
    if (std::abs(norm(v) - 1.0) > 1e-6) error(field, "must be a unit vector");
  }
  void pose(const std::string& field, const Pose& p) {
    finite(field + ".position", p.position);
    finite(field + ".rpy", p.rpy);
    finite(field + ".velocity", p.velocity);
    finite(field + ".rates", p.rates);
    if (std::abs(p.rpy.y) > 1.5533) warn(field + ".rpy", "pitch is within 1 degree of gimbal lock");
  }
  void beamwidth(const std::string& field, double bw) {
    if (std::isnan(bw)) return;  // omnidirectional
    if (!(bw > 0.0) || bw > 2.0 * 3.14159265358979323846 + 1e-9) {
      error(field, "beamwidth must be in (0, 2*pi] rad or NaN");
    }
  }
  void element(const std::string& field, const Element& e) {
    finite(field + ".lever_arm", e.lever_arm);
    unit(field + ".boresight", e.boresight);
    beamwidth(field + ".beamwidth_az", e.beamwidth_az);
    beamwidth(field + ".beamwidth_el", e.beamwidth_el);
  }
  template <typename T>
  void sized(const std::string& field, const std::vector<T>& v, std::size_t n) {
    if (v.size() != n) {
      error(field, "has " + std::to_string(v.size()) + " entries, expected " + std::to_string(n) +
                       " (one per channel)");
    }
  }

 private:
  ValidationReport& report_;
  std::string prefix_;
};

void validate_into(const PingRecord& r, ValidationReport& report, const std::string& prefix) {
  Checker c(report, prefix);
  const Samples& s = r.samples;
  const auto n = static_cast<std::size_t>(s.n_channels > 0 ? s.n_channels : 0);

  // --- samples
  if (s.n_channels <= 0) c.error("samples.n_channels", "must be positive");
  if (s.n_samples <= 0) c.error("samples.n_samples", "must be positive");
  if (s.data.size() != s.expected_size()) {
    c.error("samples.data", "size " + std::to_string(s.data.size()) +
                                " does not match channels x samples x stride = " +
                                std::to_string(s.expected_size()));
  }
  if (static_cast<int>(s.kind) < 0 || static_cast<int>(s.kind) > 2) {
    c.error("samples.kind", "unknown sample kind");
  }
  if (!(s.sample_rate > 0.0) || !std::isfinite(s.sample_rate)) {
    c.error("samples.sample_rate", "must be positive and finite");
  }
  if (!(s.center_frequency >= 0.0) || !std::isfinite(s.center_frequency)) {
    c.error("samples.center_frequency", "must be non-negative and finite");
  }
  c.finite("samples.start_time", s.start_time);
  if (s.storage != StorageType::Float32 && !(s.storage_scale > 0.0)) {
    c.error("samples.storage_scale", "must be positive for integer storage");
  }
  bool bad_sample = false;
  for (float v : s.data) {
    if (!std::isfinite(v)) {
      bad_sample = true;
      break;
    }
  }
  if (bad_sample) c.error("samples.data", "contains NaN or infinity");

  // --- transmit and band against sample rate
  const Waveform& w = r.transmit.waveform;
  if (static_cast<int>(w.type) < 0 || static_cast<int>(w.type) > 3) {
    c.error("transmit.waveform.type", "unknown waveform type");
  }
  if (!(w.duration > 0.0) || !std::isfinite(w.duration)) {
    c.error("transmit.waveform.duration", "must be positive and finite");
  }
  if (!(w.f_start > 0.0) || !(w.f_stop > 0.0)) {
    c.error("transmit.waveform.f_start", "f_start and f_stop must be positive");
  }
  if (w.type == WaveformType::Cw && w.f_start != w.f_stop) {
    c.error("transmit.waveform.f_stop", "must equal f_start for a CW pulse");
  }
  if (w.window_alpha < 0.0 || w.window_alpha > 1.0) {
    c.error("transmit.waveform.window_alpha", "must be in [0, 1]");
  }
  if (w.type == WaveformType::Arbitrary) {
    if (w.replica.empty()) c.error("transmit.waveform.replica", "is required for an arbitrary waveform");
    if (!(w.replica_sample_rate > 0.0)) {
      c.error("transmit.waveform.replica_sample_rate", "must be positive for an arbitrary waveform");
    }
  }
  if (r.transmit.time_ns < 0) c.error("transmit.time_ns", "must be non-negative");
  if (r.transmit.kind == TransmitKind::PlaneWave) c.unit("transmit.direction", r.transmit.direction);
  if (s.sample_rate > 0.0 && w.f_start > 0.0 && w.f_stop > 0.0) {
    const double f_hi = std::max(w.f_start, w.f_stop);
    const double f_lo = std::min(w.f_start, w.f_stop);
    if (s.kind == SampleKind::RealPassband) {
      if (f_hi > 0.5 * s.sample_rate) {
        // Legitimate for deliberate bandpass sampling, so a warning.
        c.warn("samples.sample_rate", "transmit band extends above Nyquist for real sampling");
      }
    } else if (s.kind == SampleKind::ComplexBaseband) {
      const double half = std::max(std::abs(f_hi - s.center_frequency),
                                   std::abs(f_lo - s.center_frequency));
      if (half > 0.5 * s.sample_rate) {
        c.error("samples.sample_rate", "transmit band does not fit in the complex baseband sample rate");
      }
    }
  }

  // --- geometry
  c.pose("geometry.tx_pose", r.geometry.tx_pose);
  c.pose("geometry.rx_pose", r.geometry.rx_pose);
  c.element("geometry.tx", r.geometry.tx);
  if (r.geometry.rx.size() != n) {
    c.error("geometry.rx", "has " + std::to_string(r.geometry.rx.size()) + " elements, expected " +
                               std::to_string(n) + " (one per channel)");
  }
  for (std::size_t i = 0; i < r.geometry.rx.size(); ++i) {
    c.element("geometry.rx[" + std::to_string(i) + "]", r.geometry.rx[i]);
  }
  const Vec3d& lla = r.geometry.frame_origin_lla;
  if (!std::isnan(lla.x) && (std::abs(lla.x) > 90.0 || std::abs(lla.y) > 180.0)) {
    c.error("geometry.frame_origin_lla", "latitude/longitude out of range");
  }

  // --- navigation: times must be non-decreasing
  auto monotonic = [&](const std::string& field, auto&& samples) {
    double prev = -std::numeric_limits<double>::infinity();
    for (const auto& a : samples) {
      if (!std::isfinite(a.t)) {
        c.error(field, "contains a non-finite time");
        return;
      }
      if (a.t < prev) {
        c.error(field, "times must be non-decreasing");
        return;
      }
      prev = a.t;
    }
  };
  monotonic("navigation.ins", r.navigation.ins);
  monotonic("navigation.dvl", r.navigation.dvl);
  monotonic("navigation.gps", r.navigation.gps);
  monotonic("navigation.depth", r.navigation.depth);

  // --- calibration
  c.sized("calibration.gain", r.calibration.gain, n);
  c.sized("calibration.phase", r.calibration.phase, n);
  c.sized("calibration.delay", r.calibration.delay, n);
  c.sized("calibration.position_offset", r.calibration.position_offset, n);
  for (double g : r.calibration.gain) {
    if (!(g > 0.0) || !std::isfinite(g)) {
      c.error("calibration.gain", "gains must be positive and finite");
      break;
    }
  }
  for (double p : r.calibration.phase) {
    if (!std::isfinite(p)) {
      c.error("calibration.phase", "must be finite");
      break;
    }
  }
  for (double d : r.calibration.delay) {
    if (!std::isfinite(d)) {
      c.error("calibration.delay", "must be finite");
      break;
    }
  }

  // --- environment
  const Environment& e = r.environment;
  if (!(e.sound_speed > 0.0) || !std::isfinite(e.sound_speed)) {
    c.error("environment.sound_speed", "must be positive and finite");
  } else if (e.sound_speed < 1300.0 || e.sound_speed > 1700.0) {
    c.warn("environment.sound_speed", "is outside 1300-1700 m/s; is this water?");
  }
  if (e.profile_depth.size() != e.profile_speed.size()) {
    c.error("environment.profile_speed", "profile depth and speed tables differ in length");
  }
  for (std::size_t i = 1; i < e.profile_depth.size(); ++i) {
    if (!(e.profile_depth[i] > e.profile_depth[i - 1])) {
      c.error("environment.profile_depth", "must be strictly increasing");
      break;
    }
  }

  // --- metadata
  const Metadata& m = r.metadata;
  if (m.array_id.empty()) c.warn("metadata.array_id", "is empty");
  if (m.mode == Mode::Unknown) c.warn("metadata.mode", "is unknown");
  c.sized("metadata.channel_element", m.channel_element, n);
  c.sized("metadata.channel_board", m.channel_board, n);
  if (m.boards.empty()) c.error("metadata.boards", "at least one board is required");
  for (auto b : m.channel_board) {
    if (b < 0 || static_cast<std::size_t>(b) >= m.boards.size()) {
      c.error("metadata.channel_board", "refers to a board that does not exist");
      break;
    }
  }
  for (std::size_t i = 0; i < m.boards.size(); ++i) {
    const int ref = m.boards[i].reference_channel;
    if (ref >= static_cast<int>(n)) {
      c.error("metadata.boards[" + std::to_string(i) + "].reference_channel",
              "is not a valid channel index");
    }
  }
  if (m.tvg_time.size() != m.tvg_gain_db.size()) {
    c.error("metadata.tvg_gain_db", "gain ramp time and gain tables differ in length");
  }
  for (std::size_t i = 1; i < m.tvg_time.size(); ++i) {
    if (!(m.tvg_time[i] > m.tvg_time[i - 1])) {
      c.error("metadata.tvg_time", "must be strictly increasing");
      break;
    }
  }
  if (m.adc_bits < 0 || m.adc_bits > 32) c.error("metadata.adc_bits", "must be in [0, 32]");
  if (m.extra_json.empty() || m.extra_json.front() != '{' || m.extra_json.back() != '}') {
    c.error("metadata.extra_json", "must be a JSON object");
  }
  if (std::isfinite(m.adc_full_scale) && m.adc_full_scale > 0.0 && !bad_sample && !s.data.empty()) {
    std::size_t clipped = 0;
    const auto limit = static_cast<float>(0.999 * m.adc_full_scale);
    for (float v : s.data) clipped += (std::abs(v) >= limit) ? 1U : 0U;
    if (clipped > 0) {
      const double pct = 100.0 * static_cast<double>(clipped) / static_cast<double>(s.data.size());
      std::ostringstream os;
      os << clipped << " values (" << pct << "%) are at ADC full scale";
      c.warn("samples.data", os.str());
    }
  }
}

}  // namespace

bool ValidationReport::ok() const { return error_count() == 0; }

std::size_t ValidationReport::error_count() const {
  std::size_t n = 0;
  for (const auto& i : issues) n += i.severity == Severity::Error ? 1U : 0U;
  return n;
}

std::size_t ValidationReport::warning_count() const { return issues.size() - error_count(); }

std::string ValidationReport::to_string() const {
  std::ostringstream os;
  for (const auto& i : issues) {
    os << (i.severity == Severity::Error ? "error: " : "warning: ") << i.field << ": " << i.message
       << "\n";
  }
  return os.str();
}

ValidationReport validate(const PingRecord& record) {
  ValidationReport report;
  validate_into(record, report, "");
  return report;
}

ValidationReport validate_file(const std::string& path) {
  ValidationReport report;
  try {
    RecordReader reader(path);
    if (reader.size() == 0) {
      report.issues.push_back({Severity::Warning, "pings", "file contains no pings"});
    }
    std::int64_t last_sequence = 0;
    std::int64_t last_time = 0;
    for (std::int64_t i = 0; i < reader.size(); ++i) {
      const PingRecord rec = reader.read(i);
      const std::string prefix = "pings[" + std::to_string(i) + "].";
      validate_into(rec, report, prefix);
      if (i > 0) {
        if (rec.transmit.time_ns < last_time) {
          report.issues.push_back(
              {Severity::Error, prefix + "transmit.time_ns", "transmit times must not go backwards"});
        }
        if (rec.metadata.sequence < last_sequence) {
          report.issues.push_back(
              {Severity::Warning, prefix + "metadata.sequence", "sequence number went backwards"});
        }
      }
      last_sequence = rec.metadata.sequence;
      last_time = rec.transmit.time_ns;
    }
  } catch (const std::exception& e) {
    report.issues.push_back({Severity::Error, "file", e.what()});
  }
  return report;
}

}  // namespace sonar::io
