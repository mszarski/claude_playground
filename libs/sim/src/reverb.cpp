#include "sonar/sim/reverb.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/fft.hpp"
#include "sonar/sim/reverb_kernel.hpp"

namespace sonar::sim {

#if SONAR_HAVE_CUDA
void reverb_steer_cuda(const float* spectra, const float* lead, const float* omega, float* total,
                       std::int64_t n_sectors, std::int64_t n_channels, std::int64_t n_low, bool uniform,
                       int device);
#endif

namespace {

using cf = std::complex<float>;

constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

/// Sector spectra held at once: bounds the host and device memory of one batch.
constexpr std::int64_t kBatchBytes = std::int64_t{64} << 20;

void check(const ReverbParams& p) {
  if (!(p.sample_rate > 0.0)) throw std::invalid_argument("reverb: sample_rate must be positive");
  if (p.n_samples < 0 || p.n_low <= 0) throw std::invalid_argument("reverb: n_samples >= 0 and n_low >= 1");
  if (p.n_azimuth <= 0) throw std::invalid_argument("reverb: n_azimuth must be positive");
  if (p.decimation < 1) throw std::invalid_argument("reverb: decimation must be at least 1");
  if (!(p.sound_speed > 0.0)) throw std::invalid_argument("reverb: sound_speed must be positive");
}

/// numpy.interp at x with zero outside the grid: linear between the neighbouring nodes.
/// `j` is where the search starts and is left at the node at or below x; the x of
/// successive calls increases, so the search is a step or two.
double interp(double x, const double* xp, const double* fp, std::int64_t n, std::int64_t& j) {
  if (!(x >= xp[0] && x <= xp[n - 1])) return 0.0;
  while (j + 1 < n && xp[j + 1] <= x) ++j;  // xp[j] <= x < xp[j + 1]
  if (j == n - 1 || xp[j] == x) return fp[j];
  const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  return slope * (x - xp[j]) + fp[j];
}

}  // namespace

namespace {

void check(const ReverbPower& power, const ReverbParams& params) {
  if (power.n_coarse < 1 || power.coarse == nullptr || power.density == nullptr) {
    throw std::invalid_argument("reverb: empty power grid");
  }
  if ((power.direction == nullptr) != (power.doppler == nullptr)) {
    throw std::invalid_argument("reverb: components need both a direction and a Doppler shift");
  }
  if (power.direction != nullptr) {
    if (power.n_rows < 1) throw std::invalid_argument("reverb: no components");
    return;
  }
  if (power.n_rows != 1 && power.n_rows != params.n_azimuth) {
    throw std::invalid_argument("reverb: the density must have one row, or one per azimuth sector");
  }
}

/// One row of sigma, [n_low].
void sigma_row(const ReverbParams& params, const ReverbPower& power, std::int64_t row, double* sigma) {
  const double fs_low = params.sample_rate / static_cast<double>(params.decimation);
  const double dphi = kTwoPi / static_cast<double>(params.n_azimuth);
  const double* density = power.density + row * power.n_coarse;
  std::int64_t node = 0;
  for (std::int64_t k = 0; k < params.n_low; ++k) {
    const double t = params.time0 + static_cast<double>(k) / fs_low;
    sigma[k] = std::sqrt(interp(t, power.coarse, density, power.n_coarse, node) / fs_low * dphi / 2.0);
  }
}

/// The sectors with power, in order of azimuth: the ones that take random numbers.
std::vector<std::int64_t> active_sectors(const ReverbParams& params, const ReverbPower& power) {
  const std::vector<std::uint8_t> rows = reverb_active_rows(params, power);
  std::vector<std::int64_t> sectors;
  if (power.direction != nullptr) {
    // Component mode: a "sector" is a row of the density.
    for (std::int64_t r = 0; r < power.n_rows; ++r) {
      if (rows[r] != 0) sectors.push_back(r);
    }
    return sectors;
  }
  for (std::int64_t m = 0; m < params.n_azimuth; ++m) {
    if (rows[power.n_rows == 1 ? 0 : m] != 0) sectors.push_back(m);
  }
  return sectors;
}

}  // namespace

std::vector<double> reverb_sigma(const ReverbParams& params, const ReverbPower& power) {
  check(params);
  check(power, params);
  const std::int64_t n_low = params.n_low;
  std::vector<double> sigma(static_cast<std::size_t>(power.n_rows * n_low));
  parallel_for_host(power.n_rows, [&](std::int64_t r) { sigma_row(params, power, r, sigma.data() + r * n_low); });
  return sigma;
}

std::vector<std::uint8_t> reverb_active_rows(const ReverbParams& params, const ReverbPower& power) {
  check(params);
  check(power, params);
  const double fs_low = params.sample_rate / static_cast<double>(params.decimation);
  const double dphi = kTwoPi / static_cast<double>(params.n_azimuth);
  std::vector<std::uint8_t> active(static_cast<std::size_t>(power.n_rows), 0);
  parallel_for_host(power.n_rows, [&](std::int64_t r) {
    const double* density = power.density + r * power.n_coarse;
    std::int64_t node = 0;
    for (std::int64_t k = 0; k < params.n_low; ++k) {
      const double t = params.time0 + static_cast<double>(k) / fs_low;
      // The value sigma_row takes the root of: not zero (or not a number) means power.
      if (!(interp(t, power.coarse, density, power.n_coarse, node) / fs_low * dphi / 2.0 == 0.0)) {
        active[r] = 1;
        return;
      }
    }
  });
  return active;
}

std::int64_t reverb_active_sectors(const ReverbParams& params, const ReverbPower& power) {
  return static_cast<std::int64_t>(active_sectors(params, power).size());
}

bool reverb_uniform_line(const double* offsets, std::int64_t n_channels) {
  if (n_channels <= 2) return false;
  for (std::int64_t ch = 1; ch < n_channels; ++ch) {
    for (int a = 0; a < 3; ++a) {
      const double along = offsets[3 * ch + a] - offsets[a];
      const double expected = static_cast<double>(ch) * (offsets[3 + a] - offsets[a]);
      // numpy.allclose(along, expected, atol=1e-9) with its default rtol.
      if (!(std::abs(along - expected) <= 1e-9 + 1e-5 * std::abs(expected))) return false;
    }
  }
  return true;
}

void reverb_accumulate(Backend backend, const ReverbParams& params, const ReverbPower& power, std::int64_t first,
                       const double* normals, std::int64_t n_sectors, const std::complex<double>* pulse,
                       std::int64_t n_pulse, const double* offsets, std::int64_t n_channels, float* total,
                       int device) {
  check(params);
  check(power, params);
  if (n_channels < 1 || n_pulse < 1) throw std::invalid_argument("reverb: channels and a pulse are required");
  (void)device;  // used by the CUDA backend only
#if !SONAR_HAVE_CUDA
  if (backend == Backend::Cuda) throw std::runtime_error("reverb: built without the CUDA backend");
#endif
  const std::int64_t n_low = params.n_low;
  const double fc = params.center_frequency, c = params.sound_speed;
  const double fs_low = params.sample_rate / static_cast<double>(params.decimation);
  const double dphi = kTwoPi / static_cast<double>(params.n_azimuth);
  const std::vector<std::int64_t> sectors = active_sectors(params, power);
  if (first < 0 || n_sectors < 0 || first + n_sectors > static_cast<std::int64_t>(sectors.size())) {
    throw std::invalid_argument("reverb: normals must cover sectors with power only");
  }
  if (n_sectors == 0) return;
  const bool components = power.direction != nullptr;
  // With one density for all sectors, sigma is worked out once.
  const bool shared = power.n_rows == 1 && !components;
  std::vector<double> shared_sigma;
  if (shared) {
    shared_sigma.resize(static_cast<std::size_t>(n_low));
    sigma_row(params, power, 0, shared_sigma.data());
  }

  // omega[f] = 2 pi (f + fc) over the FFT bins of the reduced-rate series (numpy.fft.fftfreq).
  std::vector<float> omega(static_cast<std::size_t>(n_low));
  {
    const double val = 1.0 / (static_cast<double>(n_low) * (1.0 / fs_low));
    const std::int64_t n_positive = (n_low - 1) / 2 + 1;
    for (std::int64_t k = 0; k < n_low; ++k) {
      const double f = static_cast<double>(k < n_positive ? k : k - n_low) * val;
      omega[k] = static_cast<float>(kTwoPi * (f + fc));
    }
  }
  const std::int64_t n_taps = std::min(n_pulse, n_low);
  std::vector<cf> pulse_spectrum(static_cast<std::size_t>(n_low));
  for (std::int64_t k = 0; k < n_taps; ++k) pulse_spectrum[k] = cf(pulse[k]);
  dsp::fft(pulse_spectrum, false);

  const bool uniform = reverb_uniform_line(offsets, n_channels);
  const std::int64_t batch = std::max<std::int64_t>(1, kBatchBytes / (8 * n_low));
  std::vector<float> spectra;
  std::vector<float> lead;
  for (std::int64_t begin = 0; begin < n_sectors; begin += batch) {
    const std::int64_t count = std::min(batch, n_sectors - begin);
    spectra.resize(static_cast<std::size_t>(2 * count * n_low));
    lead.resize(static_cast<std::size_t>(count * n_channels));
    // Step 1: each sector's sequence, shifted by the Doppler of the sonar's own speed
    // towards that azimuth, times the pulse (shifted alike), in frequency.
    parallel_for_host(count, [&](std::int64_t j) {
      const std::int64_t m = sectors[first + begin + j];
      const double azimuth = (static_cast<double>(m) + 0.5) * dphi;
      // The arrival direction: the sector's azimuth in the horizontal plane, or the
      // component's own direction (which may have elevation).
      const double ux = components ? power.direction[3 * m] : std::cos(azimuth);
      const double uy = components ? power.direction[3 * m + 1] : std::sin(azimuth);
      const double uz = components ? power.direction[3 * m + 2] : 0.0;
      std::vector<double> own_sigma;
      if (!shared) {
        own_sigma.resize(static_cast<std::size_t>(n_low));
        sigma_row(params, power, m, own_sigma.data());
      }
      const double* s = shared ? shared_sigma.data() : own_sigma.data();
      const double* g_re = normals + (begin + j) * 2 * n_low;
      const double* g_im = g_re + n_low;
      auto* b = reinterpret_cast<cf*>(spectra.data() + 2 * j * n_low);
      const double doppler =
          components ? power.doppler[m] : 2.0 * (params.velocity.x * ux + params.velocity.y * uy) * fc / c;
      if (doppler != 0.0) {
        const double rate = kTwoPi * doppler;
        std::vector<cf> shifted(static_cast<std::size_t>(n_low));
        for (std::int64_t k = 0; k < n_low; ++k) {
          const double dt = (params.time0 + static_cast<double>(k) / fs_low) - params.time0;
          const std::complex<double> v =
              std::complex<double>(s[k] * g_re[k], s[k] * g_im[k]) * std::complex<double>(std::cos(rate * dt), std::sin(rate * dt));
          b[k] = cf(v);
        }
        const double per_sample = 1.0 / fs_low;
        for (std::int64_t k = 0; k < n_taps; ++k) {
          const double a = rate * static_cast<double>(k) * per_sample;
          shifted[k] = cf(pulse[k] * std::complex<double>(std::cos(a), std::sin(a)));
        }
        dsp::fft(b, static_cast<std::size_t>(n_low), false);
        dsp::fft(shifted, false);
        for (std::int64_t k = 0; k < n_low; ++k) b[k] *= shifted[k];
      } else {
        for (std::int64_t k = 0; k < n_low; ++k) {
          b[k] = cf(static_cast<float>(s[k] * g_re[k]), static_cast<float>(s[k] * g_im[k]));
        }
        dsp::fft(b, static_cast<std::size_t>(n_low), false);
        for (std::int64_t k = 0; k < n_low; ++k) b[k] *= pulse_spectrum[k];
      }
      for (std::int64_t ch = 0; ch < n_channels; ++ch) {
        const double* o = offsets + 3 * ch;
        // Sector mode keeps the two-term sum of the NumPy code (uz is zero there).
        lead[j * n_channels + ch] = components ? static_cast<float>((o[0] * ux + o[1] * uy + o[2] * uz) / c)
                                               : static_cast<float>((o[0] * ux + o[1] * uy) / c);
      }
    });
    // Step 2: every sector into every channel with its plane-wave lead.
    if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
      reverb_steer_cuda(spectra.data(), lead.data(), omega.data(), total, count, n_channels, n_low,
                        uniform, device);
#endif
    } else {
      const ReverbSteerKernel k{spectra.data(), lead.data(), omega.data(), total, count,
                                n_channels,     n_low,       kReverbMaxBlock, uniform};
      parallel_for_host(k.size(), k);
    }
  }
}

void reverb_finish(const ReverbParams& params, const float* total, std::int64_t n_channels, float* out) {
  check(params);
  const std::int64_t n_low = params.n_low, n = params.n_samples;
  const double fs = params.sample_rate, fc = params.center_frequency;
  // Step 3: band-limited interpolation to the record's rate (zero-pad the spectrum),
  // the record's window, and the carrier.
  const std::int64_t n_full = n_low * params.decimation;
  const std::int64_t half = n_low / 2;
  // Sample k of the series is at time0 + k / fs; the record starts one pulse later.
  const auto start = static_cast<std::int64_t>(std::nearbyint((params.start_time - params.time0) * fs));
  const auto gain = static_cast<float>(params.decimation);
  const double carrier_rate = kTwoPi * fc;
  parallel_for_host(n_channels, [&](std::int64_t ch) {
    const auto* row = reinterpret_cast<const cf*>(total + 2 * ch * n_low);
    std::vector<cf> series(static_cast<std::size_t>(n_full));
    std::copy(row, row + half, series.begin());
    std::copy(row + half, row + n_low, series.end() - (n_low - half));
    dsp::fft(series, true);
    for (std::int64_t k = 0; k < n; ++k) {
      const std::int64_t at = start + k;
      cf v = (at >= 0 && at < n_full) ? series[at] : cf{};
      if (params.decimation > 1) v *= gain;
      if (params.carrier) {
        const double a = carrier_rate * (params.start_time + static_cast<double>(k) / fs);
        const std::complex<double> z = std::complex<double>(v) * std::complex<double>(std::cos(a), std::sin(a));
        if (params.real_output) {
          out[ch * n + k] = static_cast<float>(z.real());
          continue;
        }
        v = cf(z);
      } else if (params.real_output) {
        out[ch * n + k] = v.real();
        continue;
      }
      out[2 * (ch * n + k)] = v.real();
      out[2 * (ch * n + k) + 1] = v.imag();
    }
  });
}

void reverb_series(Backend backend, const ReverbParams& params, const ReverbPower& power, const double* normals,
                   std::int64_t n_active, const std::complex<double>* pulse, std::int64_t n_pulse,
                   const double* offsets, std::int64_t n_channels, float* out, int device) {
  if (reverb_active_sectors(params, power) != n_active) {
    throw std::invalid_argument("reverb: normals must cover exactly the sectors with power");
  }
  if (n_channels < 1) throw std::invalid_argument("reverb: channels are required");
  std::vector<float> total(static_cast<std::size_t>(2 * n_channels * params.n_low), 0.0F);
  reverb_accumulate(backend, params, power, 0, normals, n_active, pulse, n_pulse, offsets, n_channels, total.data(),
                    device);
  reverb_finish(params, total.data(), n_channels, out);
}

}  // namespace sonar::sim
