#include "sonar/live/calibrate.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/conditioning.hpp"
#include "sonar/dsp/fft.hpp"
#include "sonar/live/beamform.hpp"

namespace sonar::live {
namespace {

constexpr double kPi = 3.14159265358979323846;
using Cf = std::complex<float>;
using Cd = std::complex<double>;

// Median with the two middle values averaged for an even count (as numpy.median).
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

std::vector<Cd> channel_responses(const PingRecord& record, const Vec3d& source) {
  const Samples& s = record.samples;
  if (!s.is_complex()) throw std::invalid_argument("channel_responses: the record must be complex baseband and pulse compressed (live::prepare)");
  if (static_cast<std::int64_t>(record.geometry.rx.size()) != s.n_channels || s.n_samples < 1 || s.data.size() != s.expected_size()) {
    throw std::invalid_argument("channel_responses: one receive element per channel and a full sample block are required");
  }
  const std::int64_t n = s.n_samples;
  const double c = record.environment.sound_speed;
  const Vec3d tx = world_position(record.geometry.tx_pose, record.geometry.tx.lever_arm);
  const double out_leg = norm(source - tx);
  const double df = s.sample_rate / static_cast<double>(n);
  std::vector<Cd> v(static_cast<std::size_t>(s.n_channels));
  parallel_for_host(s.n_channels, [&](std::int64_t ch) {
    Vec3d lever = record.geometry.rx[static_cast<std::size_t>(ch)].lever_arm;
    if (static_cast<std::size_t>(ch) < record.calibration.position_offset.size()) lever += record.calibration.position_offset[static_cast<std::size_t>(ch)];
    const double tau = (out_leg + norm(source - world_position(record.geometry.rx_pose, lever))) / c;
    // Exact band-limited value at the (fractional) delay: the channel's Fourier series
    // evaluated there. Linear interpolation would cost up to a few percent of amplitude,
    // differently per channel, and read as gain error.
    std::vector<Cf> spectrum(s.channel_complex(ch), s.channel_complex(ch) + n);
    dsp::fft(spectrum.data(), static_cast<std::size_t>(n), false);
    const double t = tau - s.start_time;
    Cd acc = 0.0;
    for (std::int64_t k = 0; k < n; ++k) {
      const double f = static_cast<double>(k < (n + 1) / 2 ? k : k - n) * df;
      acc += Cd(spectrum[static_cast<std::size_t>(k)]) * std::polar(1.0, 2.0 * kPi * f * t);
    }
    v[static_cast<std::size_t>(ch)] = acc / static_cast<double>(n) * std::polar(1.0, 2.0 * kPi * s.center_frequency * tau);
  });
  return v;
}

ChannelErrors estimate_channel_errors(const std::vector<const PingRecord*>& records, const std::vector<Vec3d>& sources, bool planar) {
  if (records.empty()) throw std::invalid_argument("estimate_channel_errors: no calibration pings");
  if (records.size() != sources.size()) throw std::invalid_argument("one source position per calibration ping is required");
  const PingRecord& rec0 = *records[0];
  const ArrayAxes f = array_axes(rec0);
  const auto n_src = static_cast<Eigen::Index>(records.size());
  const auto n_ch = static_cast<Eigen::Index>(rec0.samples.n_channels);
  const double wavelength = rec0.environment.sound_speed / rec0.transmit.waveform.center();

  Eigen::MatrixXcd v(n_src, n_ch);
  for (Eigen::Index k = 0; k < n_src; ++k) {
    const PingRecord& r = *records[static_cast<std::size_t>(k)];
    if (r.samples.n_channels != n_ch) throw std::invalid_argument("estimate_channel_errors: the pings differ in channel count");
    const std::vector<Cd> row = channel_responses(r, sources[static_cast<std::size_t>(k)]);
    for (Eigen::Index c = 0; c < n_ch; ++c) v(k, c) = row[static_cast<std::size_t>(c)];
  }

  // Gain: each ping's magnitudes relative to that ping's median channel (the source level
  // drops out), then the median over pings, which one poor ping cannot pull.
  ChannelErrors out;
  out.gain.resize(static_cast<std::size_t>(n_ch));
  std::vector<double> ping_median(static_cast<std::size_t>(n_src));
  for (Eigen::Index k = 0; k < n_src; ++k) {
    std::vector<double> mags(static_cast<std::size_t>(n_ch));
    for (Eigen::Index c = 0; c < n_ch; ++c) mags[static_cast<std::size_t>(c)] = std::abs(v(k, c));
    ping_median[static_cast<std::size_t>(k)] = median(std::move(mags));
  }
  double gain_mean = 0.0;
  for (Eigen::Index c = 0; c < n_ch; ++c) {
    std::vector<double> rel(static_cast<std::size_t>(n_src));
    for (Eigen::Index k = 0; k < n_src; ++k) rel[static_cast<std::size_t>(k)] = std::abs(v(k, c)) / ping_median[static_cast<std::size_t>(k)];
    out.gain[static_cast<std::size_t>(c)] = median(std::move(rel));
    gain_mean += out.gain[static_cast<std::size_t>(c)] / static_cast<double>(n_ch);
  }
  for (double& g : out.gain) g /= gain_mean;

  // Phase of each channel relative to the array's coherent mean for that ping.
  Eigen::MatrixXd psi(n_src, n_ch);
  for (Eigen::Index k = 0; k < n_src; ++k) {
    const Cd mean = v.row(k).mean();
    for (Eigen::Index c = 0; c < n_ch; ++c) psi(k, c) = std::arg(v(k, c) * std::conj(mean));
  }
  std::vector<Vec3d> basis{f.across, f.forward};
  if (!planar) basis.push_back(f.normal);
  const auto n_par = static_cast<Eigen::Index>(1 + basis.size());
  if (n_src < n_par) throw std::invalid_argument("need at least " + std::to_string(n_par) + " sources at different bearings");
  // A source further along u is reached sooner by an element displaced along u: the two-way
  // path shortens by u . dp (receive leg only), so the restored phase rises.
  const double k_wave = 2.0 * kPi / wavelength;
  Eigen::MatrixXd design(n_src, n_par);
  for (Eigen::Index k = 0; k < n_src; ++k) {
    const Vec3d u = normalized(sources[static_cast<std::size_t>(k)] - f.origin);
    design(k, 0) = 1.0;
    for (std::size_t i = 0; i < basis.size(); ++i) design(k, static_cast<Eigen::Index>(1 + i)) = k_wave * dot(u, basis[i]);
  }
  // Minimum-norm least squares (as numpy.linalg.lstsq), so sources that do not span every
  // unknown still give the smallest correction that explains them.
  const Eigen::MatrixXd coeffs = design.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(psi);   // [1 + dims, channels]

  const double phase_mean = coeffs.row(0).mean();
  out.phase.resize(static_cast<std::size_t>(n_ch));
  std::vector<Vec3d> world(static_cast<std::size_t>(n_ch));
  Vec3d world_mean;
  for (Eigen::Index c = 0; c < n_ch; ++c) {
    out.phase[static_cast<std::size_t>(c)] = coeffs(0, c) - phase_mean;
    Vec3d p;
    for (std::size_t i = 0; i < basis.size(); ++i) p += basis[i] * coeffs(static_cast<Eigen::Index>(1 + i), c);
    world[static_cast<std::size_t>(c)] = p;
    world_mean += p / static_cast<double>(n_ch);
  }
  const Vec3d rpy = rec0.geometry.rx_pose.rpy;
  const Mat3d to_body = rotation_ned(rpy.x, rpy.y, rpy.z).transposed();
  out.position.resize(static_cast<std::size_t>(n_ch));
  for (Eigen::Index c = 0; c < n_ch; ++c) out.position[static_cast<std::size_t>(c)] = to_body * (world[static_cast<std::size_t>(c)] - world_mean);
  const Eigen::MatrixXd residual = psi - design * coeffs;
  out.residual_rms_deg = std::sqrt(residual.array().square().mean()) * 180.0 / kPi;
  return out;
}

void store_calibration(PingRecord& record, const ChannelErrors& errors) {
  const auto n = static_cast<std::size_t>(record.samples.n_channels);
  if (errors.gain.size() != n || errors.phase.size() != n || errors.position.size() != n) {
    throw std::invalid_argument("store_calibration: the errors are for a different number of channels than the record has");
  }
  record.calibration.gain = errors.gain;
  record.calibration.phase = errors.phase;
  record.calibration.position_offset = errors.position;
}

void apply_channel_errors(PingRecord& record, const ChannelErrors& errors) {
  store_calibration(record, errors);
  // The delay table is read alongside gain and phase; a record that never had one gets zeros.
  if (record.calibration.delay.size() != errors.gain.size()) record.calibration.delay.assign(errors.gain.size(), 0.0);
  dsp::apply_calibration(record);
}

}  // namespace sonar::live
