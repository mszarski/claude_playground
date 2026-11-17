#include "sonar/live/doa.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/fft.hpp"

namespace sonar::live {
namespace {

using Cd = std::complex<double>;
using Mat = Eigen::MatrixXcd;
constexpr double kPi = 3.14159265358979323846;

// One channel's block as complex samples, Hann weighted, transformed. Real records give the
// positive-frequency half; `f0` is the absolute frequency of bin 0 and `df` the bin spacing.
void block_spectrum(const PingRecord& rec, std::int64_t channel, std::int64_t first, std::int64_t n_fft,
                    std::vector<std::complex<float>>& work) {
  const Samples& s = rec.samples;
  work.assign(static_cast<std::size_t>(n_fft), {0.0F, 0.0F});
  for (std::int64_t k = 0; k < n_fft; ++k) {
    const std::int64_t i = first + k;
    if (i < 0 || i >= s.n_samples) continue;
    const auto w = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * (static_cast<double>(k) + 0.5) / static_cast<double>(n_fft)));
    work[static_cast<std::size_t>(k)] = s.is_complex() ? s.channel_complex(channel)[i] * w
                                                       : std::complex<float>(s.channel(channel)[i] * w, 0.0F);
  }
  dsp::fft(work.data(), work.size(), false);
}

// Bins of an n_fft transform whose absolute frequency lies in [f_low, f_high].
std::vector<std::int64_t> bins_in_band(const PingRecord& rec, std::int64_t n_fft, double f_low, double f_high,
                                       std::vector<double>& freqs) {
  const Samples& s = rec.samples;
  std::vector<std::int64_t> bins;
  for (std::int64_t k = 0; k < n_fft; ++k) {
    double f;
    if (s.is_complex()) {
      const std::int64_t signed_k = k <= n_fft / 2 ? k : k - n_fft;
      f = s.center_frequency + static_cast<double>(signed_k) * s.sample_rate / static_cast<double>(n_fft);
    } else {
      if (k > n_fft / 2) break;
      f = static_cast<double>(k) * s.sample_rate / static_cast<double>(n_fft);
    }
    if (f >= f_low && f <= f_high && f > 0.0) {
      bins.push_back(k);
      freqs.push_back(f);
    }
  }
  return bins;
}

// Steering matrix [N, M] at frequency f: a plane wave from bearing b reaches an element at
// offset u earlier by u sin(b) / c.
Mat steering(const std::vector<double>& u, const std::vector<double>& bearings, double f, double c) {
  Mat a(static_cast<Eigen::Index>(u.size()), static_cast<Eigen::Index>(bearings.size()));
  for (Eigen::Index m = 0; m < a.cols(); ++m) {
    const double k = 2.0 * kPi * f * std::sin(bearings[static_cast<std::size_t>(m)]) / c;
    for (Eigen::Index n = 0; n < a.rows(); ++n) a(n, m) = std::polar(1.0, k * u[static_cast<std::size_t>(n)]);
  }
  return a;
}

Eigen::Map<const Mat> snapshots_of(const DoaFrame& frame, std::size_t f) {
  // Stored [F][L][N]: a column-major N x L matrix per frequency.
  return {frame.snapshots.data() + f * static_cast<std::size_t>(frame.n_snapshots * frame.n_channels), frame.n_channels,
          frame.n_snapshots};
}

// Which of a frame's frequencies to use: all, or `limit` spread evenly over the band.
std::vector<std::size_t> chosen_frequencies(const DoaFrame& frame, std::int32_t limit) {
  const std::size_t n = frame.frequencies.size();
  std::vector<std::size_t> pick;
  if (limit <= 0 || static_cast<std::size_t>(limit) >= n) {
    for (std::size_t f = 0; f < n; ++f) pick.push_back(f);
  } else {
    for (std::int32_t i = 0; i < limit; ++i) {
      pick.push_back(static_cast<std::size_t>((static_cast<double>(i) + 0.5) * static_cast<double>(n) / static_cast<double>(limit)));
    }
  }
  return pick;
}

// Local maxima of a spectrum, strongest first, each refined by the centroid of its three bins.
// Peaks closer than `min_separation` to a stronger one are folded into it: a source between two
// grid bearings puts power on both, and nothing nearer than a fraction of a beamwidth is a
// separate source the array could have resolved.
std::vector<DoaPeak> find_peaks(const std::vector<double>& g, const std::vector<double>& bearings, std::int32_t max_peaks,
                                double min_separation = 0.0) {
  std::vector<std::pair<double, std::size_t>> cand;
  for (std::size_t m = 0; m < g.size(); ++m) {
    const double left = m > 0 ? g[m - 1] : -1.0, right = m + 1 < g.size() ? g[m + 1] : -1.0;
    if (g[m] > 0.0 && g[m] >= left && g[m] > right) cand.emplace_back(g[m], m);
  }
  std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<DoaPeak> peaks;
  for (std::size_t i = 0; i < cand.size() && static_cast<std::int32_t>(peaks.size()) < max_peaks; ++i) {
    const std::size_t m = cand[i].second;
    bool folded = false;
    for (DoaPeak& p : peaks) {
      if (std::abs(p.bearing - bearings[m]) < min_separation) {
        p.power += g[m];
        folded = true;
        break;
      }
    }
    if (folded) continue;
    double w = g[m], wb = g[m] * bearings[m];
    if (m > 0) {
      w += g[m - 1];
      wb += g[m - 1] * bearings[m - 1];
    }
    if (m + 1 < g.size()) {
      w += g[m + 1];
      wb += g[m + 1] * bearings[m + 1];
    }
    peaks.push_back({wb / w, w, 0.0});
  }
  return peaks;
}

void finish(DoaResult& out, const std::vector<double>& g, const std::vector<double>& bearings, double noise, std::int64_t n,
            const DoaOptions& o, double min_separation = 0.0) {
  out.spectrum.assign(g.begin(), g.end());
  out.noise = noise;
  auto peaks = find_peaks(g, bearings, o.max_sources, min_separation);
  for (auto& p : peaks) {
    p.snr_db = 10.0 * std::log10(std::max(static_cast<double>(n) * p.power / std::max(noise, 1e-300), 1e-30));
    if (p.power >= peaks.front().power * std::pow(10.0, -o.dynamic_range_db / 10.0) && p.snr_db >= o.min_snr_db) {
      out.peaks.push_back(p);
    }
  }
}

}  // namespace

std::vector<double> element_offsets(const PingRecord& record) {
  const auto n = record.geometry.rx.size();
  if (n < 2) throw std::invalid_argument("element_offsets: need at least two receive elements");
  std::vector<Vec3d> pos(n);
  Vec3d centre{0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < n; ++i) {
    pos[i] = world_position(record.geometry.rx_pose, record.geometry.rx[i].lever_arm);
    centre = centre + pos[i] / static_cast<double>(n);
  }
  const Vec3d axis = normalized(pos[n - 1] - pos[0]);
  std::vector<double> u(n);
  for (std::size_t i = 0; i < n; ++i) u[i] = dot(pos[i] - centre, axis);
  return u;
}

DoaFrame gate_frame(const PingRecord& record, double two_way_time, std::int64_t n_fft) {
  const Samples& s = record.samples;
  if (!s.is_complex()) throw std::invalid_argument("gate_frame: the record must be conditioned (complex baseband)");
  const Waveform& w = record.transmit.waveform;
  const double f_low = std::min(w.f_start, w.f_stop), f_high = std::max(w.f_start, w.f_stop);
  DoaFrame frame;
  frame.n_channels = s.n_channels;
  frame.n_snapshots = 1;
  frame.time = two_way_time;
  const auto bins = bins_in_band(record, n_fft, f_low, f_high, frame.frequencies);
  const auto centre = static_cast<std::int64_t>(std::llround((two_way_time - s.start_time) * s.sample_rate));
  const std::int64_t first = centre - n_fft / 2;
  frame.snapshots.assign(bins.size() * static_cast<std::size_t>(s.n_channels), Cd{});
  std::vector<std::complex<float>> work;
  for (std::int64_t c = 0; c < s.n_channels; ++c) {
    block_spectrum(record, c, first, n_fft, work);
    for (std::size_t f = 0; f < bins.size(); ++f) {
      // Phase of each bin referred to the block's first sample, so that it is the absolute
      // frequency's phase at that instant on every channel (the common factor is harmless).
      frame.snapshots[f * static_cast<std::size_t>(s.n_channels) + static_cast<std::size_t>(c)] =
          Cd(work[static_cast<std::size_t>(bins[f])]);
    }
  }
  return frame;
}

std::vector<DoaFrame> passive_frames(const PingRecord& record, std::int64_t n_fft, std::int64_t hop,
                                     std::int64_t n_average, double f_low, double f_high) {
  const Samples& s = record.samples;
  if (n_fft < 8 || hop < 1 || n_average < 1) throw std::invalid_argument("passive_frames: bad block sizes");
  std::vector<double> freqs;
  const auto bins = bins_in_band(record, n_fft, f_low, f_high, freqs);
  if (bins.empty()) throw std::invalid_argument("passive_frames: no FFT bin inside the band asked for");
  const std::int64_t span = n_fft + (n_average - 1) * hop;
  std::vector<DoaFrame> frames;
  for (std::int64_t start = 0; start + span <= s.n_samples; start += span) {
    DoaFrame frame;
    frame.n_channels = s.n_channels;
    frame.n_snapshots = n_average;
    frame.frequencies = freqs;
    frame.time = s.start_time + (static_cast<double>(start) + 0.5 * static_cast<double>(span)) / s.sample_rate;
    frame.snapshots.assign(bins.size() * static_cast<std::size_t>(n_average * s.n_channels), Cd{});
    frames.push_back(std::move(frame));
  }
  parallel_for_host(static_cast<std::int64_t>(frames.size()) * s.n_channels, [&](std::int64_t job) {
    const std::int64_t i = job / s.n_channels, c = job % s.n_channels;
    std::vector<std::complex<float>> work;
    DoaFrame& frame = frames[static_cast<std::size_t>(i)];
    for (std::int64_t l = 0; l < n_average; ++l) {
      block_spectrum(record, c, i * span + l * hop, n_fft, work);
      for (std::size_t f = 0; f < bins.size(); ++f) {
        frame.snapshots[(f * static_cast<std::size_t>(n_average) + static_cast<std::size_t>(l)) * static_cast<std::size_t>(s.n_channels) +
                        static_cast<std::size_t>(c)] = Cd(work[static_cast<std::size_t>(bins[f])]);
      }
    }
  });
  return frames;
}

DoaResult conventional(const DoaFrame& frame, const std::vector<double>& u, double c, const std::vector<double>& bearings,
                       const DoaOptions& o) {
  const auto n = frame.n_channels;
  std::vector<double> g(bearings.size(), 0.0);
  double total = 0.0;
  const auto pick = chosen_frequencies(frame, o.max_frequencies);
  for (std::size_t f : pick) {
    const Mat a = steering(u, bearings, frame.frequencies[f], c);
    const auto y = snapshots_of(frame, f);
    const Eigen::VectorXd p = (a.adjoint() * y).cwiseAbs2().rowwise().sum();
    for (std::size_t m = 0; m < g.size(); ++m) g[m] += p[static_cast<Eigen::Index>(m)];
    total += y.cwiseAbs2().sum();
  }
  const double norm = static_cast<double>(pick.size() * static_cast<std::size_t>(frame.n_snapshots)) * static_cast<double>(n * n);
  for (double& v : g) v /= norm;
  DoaResult out;
  // The mean element power stands in for the noise: a conventional spectrum cannot separate them.
  finish(out, g, bearings, total / (norm / static_cast<double>(n)), n, o);
  return out;
}

DoaResult sbl(const DoaFrame& frame, const std::vector<double>& u, double c, const std::vector<double>& bearings,
              const DoaOptions& o) {
  const Eigen::Index n = frame.n_channels, m_grid = static_cast<Eigen::Index>(bearings.size());
  const auto pick = chosen_frequencies(frame, o.max_frequencies);
  const std::size_t n_f = pick.size();
  const auto l = static_cast<double>(frame.n_snapshots);
  if (static_cast<std::size_t>(n) != u.size()) throw std::invalid_argument("sbl: one element offset per channel is needed");
  if (n_f == 0 || m_grid < 3) throw std::invalid_argument("sbl: empty frame or bearing grid");
  std::vector<Mat> a(n_f);
  std::vector<double> power(n_f), sigma2(n_f);
  Eigen::VectorXd gamma = Eigen::VectorXd::Zero(m_grid);
  for (std::size_t f = 0; f < n_f; ++f) {
    a[f] = steering(u, bearings, frame.frequencies[pick[f]], c);
    const auto y = snapshots_of(frame, pick[f]);
    power[f] = y.cwiseAbs2().sum() / (l * static_cast<double>(n));
    sigma2[f] = 0.1 * power[f];
    // Start from the conventional beam power.
    gamma += (a[f].adjoint() * y).cwiseAbs2().rowwise().sum() / (l * static_cast<double>(n * n) * static_cast<double>(n_f));
  }
  DoaResult out;
  std::vector<double> g(static_cast<std::size_t>(m_grid));
  // Bearings still in play. Most gammas fall towards zero within a few iterations; once one is far
  // below anything that could be reported it is set to zero and left out, which is where the time goes.
  // Coarse to fine: the first iterations run on every `stride`-th bearing (about half a beamwidth
  // apart for a half-wavelength array), which is enough to see where the energy is; the grid asked
  // for is then switched on around those bearings only.
  const Eigen::Index stride = std::max<Eigen::Index>(m_grid / (2 * n), 1);
  const std::int32_t coarse_iterations = stride > 1 ? 6 : 0;
  std::vector<Eigen::Index> active;
  for (Eigen::Index i = (m_grid - 1) % stride / 2; i < m_grid; i += stride) active.push_back(i);
  if (stride > 1) {
    for (Eigen::Index i = 0; i < m_grid; ++i) gamma[i] = (i - active[0]) % stride == 0 ? gamma[i] : 0.0;
  }
  const double prune = std::pow(10.0, -(o.dynamic_range_db + 15.0) / 10.0);
  std::vector<Eigen::VectorXd> num(n_f), den(n_f);
  for (out.iterations = 1; out.iterations <= o.max_iterations; ++out.iterations) {
    const auto n_act = static_cast<Eigen::Index>(active.size());
    Eigen::VectorXd g_act(n_act);
    for (Eigen::Index k = 0; k < n_act; ++k) g_act[k] = gamma[active[static_cast<std::size_t>(k)]];
    // Every frequency is independent given gamma.
    parallel_for_host(static_cast<std::int64_t>(n_f), [&](std::int64_t fi) {
      const auto f = static_cast<std::size_t>(fi);
      const auto y = snapshots_of(frame, pick[f]);
      Mat aa(n, n_act);
      for (Eigen::Index k = 0; k < n_act; ++k) aa.col(k) = a[f].col(active[static_cast<std::size_t>(k)]);
      Mat cov = (aa * g_act.asDiagonal()) * aa.adjoint();
      cov.diagonal().array() += sigma2[f];
      const Eigen::LLT<Mat> llt(cov);
      const Mat b = llt.solve(aa);                                    // Sigma^-1 A
      num[f] = (b.adjoint() * y).cwiseAbs2().rowwise().sum() / l;     // |y^H Sigma^-1 a|^2 per bearing
      den[f] = (aa.conjugate().cwiseProduct(b)).colwise().sum().real().transpose();   // a^H Sigma^-1 a
    });
    Eigen::VectorXd top = Eigen::VectorXd::Zero(n_act), bottom = Eigen::VectorXd::Zero(n_act);
    for (std::size_t f = 0; f < n_f; ++f) {
      top += num[f];
      bottom += den[f];
    }
    const Eigen::VectorXd next = g_act.cwiseProduct((top.cwiseQuotient(bottom.cwiseMax(1e-300))).cwiseSqrt());
    const double change = (next - g_act).cwiseAbs().sum() / std::max(g_act.cwiseAbs().sum(), 1e-300);
    const double floor = out.iterations >= 4 && out.iterations != coarse_iterations + 1 ? prune * next.maxCoeff() : -1.0;
    std::vector<Eigen::Index> kept;
    for (Eigen::Index k = 0; k < n_act; ++k) {
      const Eigen::Index m = active[static_cast<std::size_t>(k)];
      if (next[k] > floor) {
        gamma[m] = next[k];
        kept.push_back(m);
      } else {
        gamma[m] = 0.0;
      }
    }
    if (kept.size() >= 3) active = std::move(kept);
    if (out.iterations == coarse_iterations) {
      // Switch on the fine grid within two coarse steps of every bearing still alive, sharing each
      // coarse bearing's power among the fine ones it stood for.
      std::vector<char> on(static_cast<std::size_t>(m_grid), 0);
      Eigen::VectorXd fine = Eigen::VectorXd::Zero(m_grid);
      for (Eigen::Index m : active) {
        for (Eigen::Index i = std::max<Eigen::Index>(m - 2 * stride, 0); i <= std::min<Eigen::Index>(m + 2 * stride, m_grid - 1); ++i) {
          on[static_cast<std::size_t>(i)] = 1;
          if (std::abs(i - m) <= stride / 2) fine[i] = gamma[m] / static_cast<double>(stride);
        }
      }
      active.clear();
      for (Eigen::Index i = 0; i < m_grid; ++i) {
        if (!on[static_cast<std::size_t>(i)]) continue;
        active.push_back(i);
        gamma[i] = std::max(fine[i], 1e-3 * gamma.maxCoeff() / static_cast<double>(stride));
      }
    }
    // Noise: what is left of each frequency's data outside the span of the strongest peaks.
    for (Eigen::Index i = 0; i < m_grid; ++i) g[static_cast<std::size_t>(i)] = gamma[i];
    const auto peaks = find_peaks(g, bearings, std::min<std::int32_t>(o.max_sources, static_cast<std::int32_t>(n) - 1));
    if (!peaks.empty()) {
      for (std::size_t f = 0; f < n_f; ++f) {
        Mat am(n, static_cast<Eigen::Index>(peaks.size()));
        for (std::size_t k = 0; k < peaks.size(); ++k) {
          const double kk = 2.0 * kPi * frame.frequencies[pick[f]] * std::sin(peaks[k].bearing) / c;
          for (Eigen::Index i = 0; i < n; ++i) am(i, static_cast<Eigen::Index>(k)) = std::polar(1.0, kk * u[static_cast<std::size_t>(i)]);
        }
        const auto y = snapshots_of(frame, pick[f]);
        const Mat proj = am * am.completeOrthogonalDecomposition().solve(y);
        const double left = (y - proj).cwiseAbs2().sum() / (l * static_cast<double>(n - static_cast<Eigen::Index>(peaks.size())));
        sigma2[f] = std::max(left, 1e-6 * power[f]);
      }
    }
    if (change < o.tolerance && out.iterations > coarse_iterations + 2) break;
  }
  out.iterations = std::min(out.iterations, o.max_iterations);
  double noise = 0.0;
  for (double s2 : sigma2) noise += s2 / static_cast<double>(n_f);
  // A fifth of a beamwidth at the top of the band.
  const double f_top = *std::max_element(frame.frequencies.begin(), frame.frequencies.end());
  const double length = *std::max_element(u.begin(), u.end()) - *std::min_element(u.begin(), u.end());
  finish(out, g, bearings, noise, n, o, 0.2 * c / (f_top * std::max(length, 1e-9)));
  return out;
}

}  // namespace sonar::live
