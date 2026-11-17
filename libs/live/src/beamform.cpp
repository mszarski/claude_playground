#include "sonar/live/beamform.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/exec_host.hpp"
#include "sonar/dsp/fft.hpp"
#include "sonar/live/beamform_batched.hpp"

namespace sonar::live {

#if SONAR_HAVE_CUDA
// beamform_cuda.cu: the batched formulation on the device.
void farfield_cuda(const float* samples, const float* w, const double* tau, std::int64_t n, std::int64_t n_el,
                   std::int64_t n_b, double df, double fc, float* out, int device);
MvdrBatchedStatus mvdr_cuda(const MvdrBatchedInput& in, float* out, int device);
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;
using Cf = std::complex<float>;
using Cd = std::complex<double>;
using MatCf = Eigen::Matrix<Cf, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatCd = Eigen::MatrixXcd;

void require_conditioned(const PingRecord& record, const char* who) {
  const Samples& s = record.samples;
  if (!s.is_complex()) throw std::invalid_argument(std::string(who) + ": the record must be complex baseband and pulse compressed (live::prepare)");
  if (s.n_channels < 2 || s.n_samples < 2) throw std::invalid_argument(std::string(who) + ": needs at least two channels and two samples");
  if (static_cast<std::int64_t>(record.geometry.rx.size()) != s.n_channels) {
    throw std::invalid_argument(std::string(who) + ": one receive element per channel is required");
  }
  if (s.data.size() != s.expected_size()) throw std::invalid_argument(std::string(who) + ": sample block does not match its shape");
}

// Direction of largest extent of the points about their mean (unit vector, sign arbitrary).
Vec3d principal_axis(const std::vector<Vec3d>& positions, const Vec3d& centre) {
  Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
  for (const Vec3d& p : positions) {
    const Eigen::Vector3d d(p.x - centre.x, p.y - centre.y, p.z - centre.z);
    scatter += d * d.transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(scatter);
  const Eigen::Vector3d v = eig.eigenvectors().col(2);   // eigenvalues come in increasing order
  return {v[0], v[1], v[2]};
}

// Frequency of FFT bin k as a signed multiple of the bin spacing (the order numpy.fft.fftfreq uses).
std::int64_t signed_bin(std::int64_t k, std::int64_t n) { return k < (n + 1) / 2 ? k : k - n; }

// Every channel's spectrum, [channels][samples] row major.
std::vector<Cf> spectra(const Samples& s) {
  const auto n = static_cast<std::size_t>(s.n_samples);
  std::vector<Cf> out(static_cast<std::size_t>(s.n_channels) * n);
  parallel_for_host(s.n_channels, [&](std::int64_t c) {
    Cf* row = out.data() + static_cast<std::size_t>(c) * n;
    std::copy(s.channel_complex(c), s.channel_complex(c) + n, row);
    dsp::fft(row, n, false);
  });
  return out;
}

BeamMap empty_map(const PingRecord& record, const std::vector<double>& bearings) {
  const Samples& s = record.samples;
  BeamMap m;
  m.n_bearings = static_cast<std::int64_t>(bearings.size());
  m.n_samples = s.n_samples;
  m.bearings = bearings;
  m.data.assign(bearings.size() * static_cast<std::size_t>(s.n_samples), Cf{});
  m.times.resize(static_cast<std::size_t>(s.n_samples));
  m.ranges.resize(m.times.size());
  for (std::size_t i = 0; i < m.times.size(); ++i) {
    m.times[i] = s.start_time + static_cast<double>(i) / s.sample_rate;
    m.ranges[i] = 0.5 * record.environment.sound_speed * m.times[i];
  }
  // The replica hangs over the end of the record for the last pulse length.
  const auto pulse = static_cast<std::int64_t>(std::ceil(record.transmit.waveform.duration * s.sample_rate));
  m.n_valid = std::max<std::int64_t>(s.n_samples - pulse, 1);
  return m;
}

}  // namespace

ArrayAxes array_axes(const PingRecord& record) {
  const auto n = record.geometry.rx.size();
  if (n < 2) throw std::invalid_argument("array_axes: needs at least two receive elements");
  ArrayAxes f;
  f.positions.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    Vec3d lever = record.geometry.rx[i].lever_arm;
    if (i < record.calibration.position_offset.size()) lever += record.calibration.position_offset[i];
    f.positions.push_back(world_position(record.geometry.rx_pose, lever));
    f.origin += f.positions.back() / static_cast<double>(n);
  }
  f.forward = world_direction(record.geometry.rx_pose, record.geometry.rx[0].boresight);
  Vec3d a = principal_axis(f.positions, f.origin);
  a = a - f.forward * dot(a, f.forward);
  if (!(norm(a) > 0.0)) throw std::invalid_argument("array_axes: the elements run along the boresight");
  a = normalized(a);
  f.across = dot(f.positions[n - 1] - f.positions[0], a) < 0.0 ? -a : a;
  f.normal = cross(f.forward, f.across);
  return f;
}

std::vector<float> aperture_weights(const PingRecord& record, const std::string& window) {
  if (window != "rect" && window != "hann" && window != "hamming") {
    throw std::invalid_argument("unknown aperture window '" + window + "' (rect, hann or hamming)");
  }
  const ArrayAxes f = array_axes(record);
  const auto n = f.positions.size();
  std::vector<float> w(n, 1.0F);
  if (window == "rect") return w;
  const Vec3d axis = principal_axis(f.positions, f.origin);
  std::vector<double> u(n);
  double lo = 1e300, hi = -1e300;
  for (std::size_t i = 0; i < n; ++i) {
    u[i] = dot(f.positions[i] - f.origin, axis);
    lo = std::min(lo, u[i]);
    hi = std::max(hi, u[i]);
  }
  const double span = hi - lo, pitch = span / static_cast<double>(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = u[i] / (span + pitch);   // in (-0.5, 0.5)
    const double c = std::cos(kPi * x);
    w[i] = static_cast<float>(window == "hann" ? c * c : 0.54 + 0.46 * std::cos(2.0 * kPi * x));
  }
  return w;
}

std::vector<double> steering_delays(const PingRecord& record, const std::vector<double>& bearings) {
  const ArrayAxes f = array_axes(record);
  const auto n = f.positions.size();
  std::vector<double> tau(bearings.size() * n);
  for (std::size_t b = 0; b < bearings.size(); ++b) {
    const Vec3d u = f.forward * std::cos(bearings[b]) + f.across * std::sin(bearings[b]);
    for (std::size_t e = 0; e < n; ++e) tau[b * n + e] = dot(u, f.positions[e] - f.origin) / record.environment.sound_speed;
  }
  return tau;
}

BeamMap BeamMap::valid() const {
  const std::int64_t n = n_valid > 0 ? std::min(n_valid, n_samples) : n_samples;
  BeamMap m;
  m.n_bearings = n_bearings;
  m.n_samples = m.n_valid = n;
  m.bearings = bearings;
  m.times.assign(times.begin(), times.begin() + n);
  m.ranges.assign(ranges.begin(), ranges.begin() + n);
  m.data.resize(static_cast<std::size_t>(n_bearings * n));
  for (std::int64_t b = 0; b < n_bearings; ++b) std::copy(row(b), row(b) + n, m.data.begin() + b * n);
  return m;
}

BeamMap farfield_beamform(const PingRecord& record, const std::vector<double>& bearings, const std::string& window,
                          Backend backend, int device, bool batched) {
  require_conditioned(record, "farfield_beamform");
  const Samples& s = record.samples;
  const std::int64_t n = s.n_samples, n_el = s.n_channels;
  const std::vector<float> w = aperture_weights(record, window);
  const std::vector<double> tau = steering_delays(record, bearings);
  const double df = s.sample_rate / static_cast<double>(n), fc = s.center_frequency;
  BeamMap out = empty_map(record, bearings);
  (void)device;  // used by the CUDA backend only
#if !SONAR_HAVE_CUDA
  if (backend == Backend::Cuda) throw std::runtime_error("farfield_beamform: built without the CUDA backend");
#endif
  if (backend == Backend::Cuda || batched) {
    auto* data = reinterpret_cast<float*>(out.data.data());
    if (out.n_bearings == 0) return out;
    if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
      farfield_cuda(s.data.data(), w.data(), tau.data(), n, n_el, out.n_bearings, df, fc, data, device);
#endif
    } else {
      dsp::HostExec ex;
      farfield_batched(ex, s.data.data(), w.data(), tau.data(), n, n_el, out.n_bearings, df, fc, data);
    }
    return out;
  }
  const std::vector<Cf> x = spectra(s);
  parallel_for_host(out.n_bearings, [&](std::int64_t b) {
    Cf* acc = out.data.data() + b * n;
    for (std::int64_t e = 0; e < n_el; ++e) {
      const double t = tau[static_cast<std::size_t>(b * n_el + e)];
      const float we = w[static_cast<std::size_t>(e)];
      const Cf* xe = x.data() + e * n;
      // The steering phase is linear in the bin index: step a phasor instead of taking a sine
      // and cosine per bin, and set it afresh every so often so rounding cannot build up.
      const Cd step = std::polar(1.0, -2.0 * kPi * df * t);
      Cd phasor;
      for (std::int64_t k = 0; k < n; ++k) {
        if ((k & 63) == 0 || k == (n + 1) / 2) {
          phasor = std::polar(1.0, -2.0 * kPi * (fc + static_cast<double>(signed_bin(k, n)) * df) * t);
        } else {
          phasor *= step;
        }
        acc[k] += we * xe[k] * Cf(static_cast<float>(phasor.real()), static_cast<float>(phasor.imag()));
      }
    }
    dsp::fft(acc, static_cast<std::size_t>(n), true);
  });
  return out;
}

BeamMap mvdr_beamform(const PingRecord& record, const std::vector<double>& bearings, const MvdrOptions& o, Backend backend,
                      int device, bool batched) {
  require_conditioned(record, "mvdr_beamform");
  (void)device;  // used by the CUDA backend only
#if !SONAR_HAVE_CUDA
  if (backend == Backend::Cuda) throw std::runtime_error("mvdr_beamform: built without the CUDA backend");
#endif
  if (o.n_subbands < 1 || o.block < 1 || o.guard_blocks < 0 || !(o.loading >= 0.0)) {
    throw std::invalid_argument("mvdr_beamform: n_subbands and block must be positive, guard_blocks and loading not negative");
  }
  const Samples& s = record.samples;
  const std::int64_t n = s.n_samples, n_el = s.n_channels;
  const auto n_b = static_cast<std::int64_t>(bearings.size());
  const std::int64_t n_blocks = (n + o.block - 1) / o.block;
  if (n_blocks < 2 * o.guard_blocks + 3) {
    throw std::invalid_argument("mvdr_beamform: record too short for this block size: no training data would be left");
  }
  const double fc = s.center_frequency;
  const Waveform& w = record.transmit.waveform;
  const double lo = std::min(w.f_start, w.f_stop) - fc, hi = std::max(w.f_start, w.f_stop) - fc;
  // Bin frequencies and sub-band edges are formed the way NumPy forms them (fftfreq, linspace),
  // so a bin that sits on an edge falls into the same sub-band here as in the Python stage.
  const double bin_hz = 1.0 / (static_cast<double>(n) * (1.0 / s.sample_rate));
  const double edge_step = (hi - lo) / static_cast<double>(o.n_subbands);
  auto edge = [&](std::int64_t k) { return k == o.n_subbands ? hi : lo + static_cast<double>(k) * edge_step; };

  const std::vector<double> tau = steering_delays(record, bearings);
  BeamMap out = empty_map(record, bearings);
  if ((backend == Backend::Cuda || batched) && n_b > 0) {
    // The batched formulation: the bins of each sub-band are found here, by the same
    // comparisons as below; the executor does the rest.
    std::vector<std::int32_t> band(static_cast<std::size_t>(n), -1), sub;
    std::vector<double> f_abs;
    for (std::int64_t k = 0; k < o.n_subbands; ++k) {
      bool any = false;
      for (std::int64_t i = 0; i < n; ++i) {
        const double f = static_cast<double>(signed_bin(i, n)) * bin_hz;
        if (f >= edge(k) && f < edge(k + 1)) {
          band[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(k);
          any = true;
        }
      }
      if (any) {
        sub.push_back(static_cast<std::int32_t>(k));
        f_abs.push_back(fc + 0.5 * (edge(k) + edge(k + 1)));
      }
    }
    if (sub.empty()) return out;
    const MvdrBatchedInput in{s.data.data(), tau.data(), band.data(), sub.data(), f_abs.data(),
                              n,             n_el,       n_b,         static_cast<std::int64_t>(sub.size()),
                              o.block,       o.guard_blocks,          o.loading};
    auto* data = reinterpret_cast<float*>(out.data.data());
    MvdrBatchedStatus status = MvdrBatchedStatus::Ok;
    if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
      status = mvdr_cuda(in, data, device);
#endif
    } else {
      dsp::HostExec ex;
      status = mvdr_batched(ex, in, data);
    }
    if (status == MvdrBatchedStatus::Singular) {
      throw std::runtime_error("mvdr_beamform: singular covariance (no signal in the training data and no loading)");
    }
    if (status == MvdrBatchedStatus::Ok) return out;
    // Not positive definite somewhere: the reference code below, which falls back to LU.
    std::fill(out.data.begin(), out.data.end(), Cf{});
  }
  const std::vector<Cf> x = spectra(s);
  Eigen::Map<MatCf> out_map(out.data.data(), n_b, n);
  MatCf xk(n_el, n);                              // one sub-band of every channel, in time
  MatCd a(n_el, n_b);                             // steering vectors, one column per bearing
  std::vector<MatCd> outer(static_cast<std::size_t>(n_blocks));
  std::vector<MatCf> coeff(static_cast<std::size_t>(n_blocks));   // [bearing][element]: n_el conj(w)
  auto block_len = [&](std::int64_t i) { return std::min(o.block, n - i * o.block); };
  // The weights of a block are applied a few bearings at a time, so that the loop has many
  // more iterations than threads and every one of them writes its own part of the map.
  const std::int64_t group = 8, n_groups = (n_b + group - 1) / group;
  std::atomic<bool> singular{false};

  for (std::int64_t k = 0; k < o.n_subbands; ++k) {
    std::vector<std::int64_t> bins;
    for (std::int64_t i = 0; i < n; ++i) {
      const double f = static_cast<double>(signed_bin(i, n)) * bin_hz;
      if (f >= edge(k) && f < edge(k + 1)) bins.push_back(i);
    }
    if (bins.empty()) continue;
    parallel_for_host(n_el, [&](std::int64_t c) {
      Cf* row = xk.data() + c * n;
      std::fill(row, row + n, Cf{});
      for (std::int64_t i : bins) row[i] = x[static_cast<std::size_t>(c * n + i)];
      dsp::fft(row, static_cast<std::size_t>(n), true);
    });
    const double f_abs = fc + 0.5 * (edge(k) + edge(k + 1));
    for (std::int64_t b = 0; b < n_b; ++b) {
      for (std::int64_t e = 0; e < n_el; ++e) a(e, b) = std::polar(1.0, 2.0 * kPi * f_abs * tau[static_cast<std::size_t>(b * n_el + e)]);
    }
    parallel_for_host(n_blocks, [&](std::int64_t i) {
      const auto blk = xk.middleCols(i * o.block, block_len(i));
      outer[static_cast<std::size_t>(i)] = (blk * blk.adjoint()).cast<Cd>();
    });
    MatCd total = MatCd::Zero(n_el, n_el);
    for (const MatCd& m : outer) total += m;
    parallel_for_host(n_blocks, [&](std::int64_t i) {
      // Training data: every block but this one and its guards. With a strong echo in its own
      // training data MVDR takes the echo's small mismatch to the steering vector for
      // interference and cancels the echo.
      const std::int64_t j0 = std::max<std::int64_t>(i - o.guard_blocks, 0), j1 = std::min(i + o.guard_blocks + 1, n_blocks);
      MatCd r = total;
      std::int64_t count = n;
      for (std::int64_t j = j0; j < j1; ++j) {
        r -= outer[static_cast<std::size_t>(j)];
        count -= block_len(j);
      }
      r /= static_cast<double>(count);
      const double power = r.trace().real() / static_cast<double>(n_el);
      r.diagonal().array() += o.loading * power;
      MatCd ra;                                  // columns: R^-1 a
      const Eigen::LLT<MatCd> llt(r);
      if (llt.info() == Eigen::Success) {
        ra = llt.solve(a);
      } else {
        // Without loading, differences of sample covariances need not be positive definite.
        ra = r.partialPivLu().solve(a);
      }
      MatCf& cf = coeff[static_cast<std::size_t>(i)];
      cf.resize(n_b, n_el);
      for (std::int64_t b = 0; b < n_b; ++b) {
        const Cd gain = a.col(b).dot(ra.col(b));  // a^H R^-1 a
        if (!(std::abs(gain) > 0.0) || !std::isfinite(std::abs(gain))) {
          singular = true;
          cf.row(b).setZero();
          continue;
        }
        cf.row(b) = (static_cast<double>(n_el) * (ra.col(b) / gain).conjugate().transpose()).cast<Cf>();
      }
    });
    parallel_for_host(n_blocks * n_groups, [&](std::int64_t job) {
      const std::int64_t i = job / n_groups, b0 = (job % n_groups) * group, rows = std::min(group, n_b - b0);
      out_map.block(b0, i * o.block, rows, block_len(i)).noalias() +=
          coeff[static_cast<std::size_t>(i)].middleRows(b0, rows) * xk.middleCols(i * o.block, block_len(i));
    });
  }
  // Thrown here, not in the loops: an exception must not leave a parallel region.
  if (singular) throw std::runtime_error("mvdr_beamform: singular covariance (no signal in the training data and no loading)");
  return out;
}

}  // namespace sonar::live
