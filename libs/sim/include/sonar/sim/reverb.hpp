// Boundary reverberation by the scattering-function method: the per-ping generation of
// python/gpusonar/sim/reverb.py (step 3 and 4 of its header). Given the reverberation
// power against time and azimuth, it makes the time series every hydrophone receives:
//
//   1. each azimuth sector's complex Gaussian sequence, with the time-varying standard
//      deviation of that sector, convolved with the transmitted pulse (in frequency);
//   2. the sum over sectors at each hydrophone with the plane-wave lead of each
//      direction (reverb_kernel.hpp: the one part that is a per-element kernel);
//   3. band-limited interpolation to the record's rate, the cut to the record's window
//      and the carrier.
//
// Random numbers are not drawn here. The caller passes standard normal deviates (the
// Python simulator draws them from its NumPy generator, in the order the NumPy
// implementation consumes them), so a ping is reproducible from the scenario seed on
// either engine and the two engines can be compared sample for sample.
//
// What runs where: steps 1 and 3 are FFTs and run on the host on both backends
// (sonar::dsp::fft). Step 2 runs on the host or on the CUDA device.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/vec.hpp"

namespace sonar::sim {

struct ReverbParams {
  double sample_rate = 0.0;       ///< of the record, Hz
  std::int64_t n_samples = 0;     ///< of the record, per channel
  double start_time = 0.0;        ///< of the record's sample 0 after transmit, s
  /// Time of sample 0 of the impulse response the pulse is convolved with: one pulse
  /// length before the record starts.
  double time0 = 0.0;
  double center_frequency = 0.0;  ///< the carrier the baseband series refers to, Hz
  double sound_speed = 1500.0;
  std::int64_t n_azimuth = 0;     ///< sectors over the full circle
  /// The series is generated at sample_rate / decimation with n_low samples and
  /// interpolated to the record's rate.
  std::int64_t decimation = 1;
  std::int64_t n_low = 0;
  Vec3d velocity;                 ///< of the sonar (world): Doppler of each sector, two-way (sector mode)
  bool carrier = false;           ///< put the series on the carrier (real or analytic records)
  bool real_output = false;       ///< keep the real part: `out` is then [channels, n_samples] floats
};

/// The reverberation power the sequences are drawn with: a density (power per radian of
/// azimuth and per second) on a coarse time grid, for each sector or one for all.
struct ReverbPower {
  const double* coarse = nullptr;   ///< [n_coarse] times after transmit, increasing, s
  std::int64_t n_coarse = 0;
  const double* density = nullptr;  ///< [n_rows, n_coarse]
  std::int64_t n_rows = 0;          ///< n_azimuth, or 1 when every sector has the same density
  /// Component mode (both set, or neither). Each row of the density is then one component
  /// of the reverberation with its own arrival direction and Doppler shift, in place of
  /// the azimuth sector of that index: n_rows is any number, ReverbParams::velocity is not
  /// used, and n_azimuth only scales the density (sigma^2 = density / fs_low * dphi / 2).
  /// This is how bistatic geometry, the Doppler of both platforms and the arrival
  /// elevation reach the generator (python/gpusonar/sim/reverb_bistatic.py).
  const double* direction = nullptr;  ///< [n_rows, 3] unit vector from the array towards the arrival (world)
  const double* doppler = nullptr;    ///< [n_rows] frequency shift of the component, Hz
};

/// Standard deviation of each sector's sequence at the reduced rate, [n_rows, n_low]:
/// sqrt(density(t) / fs_low * dphi / 2), the density interpolated linearly in time and
/// zero outside the grid.
std::vector<double> reverb_sigma(const ReverbParams& params, const ReverbPower& power);

/// Which rows of the density give a sequence that is not zero throughout, [n_rows]. A
/// sector without power is skipped and consumes no random numbers.
std::vector<std::uint8_t> reverb_active_rows(const ReverbParams& params, const ReverbPower& power);

/// Number of sectors with power (in component mode: of components with power).
std::int64_t reverb_active_sectors(const ReverbParams& params, const ReverbPower& power);

/// True when the receive elements (offsets [n_channels, 3]) lie on a uniformly spaced
/// line, for which the steering needs one complex exponential per sector.
bool reverb_uniform_line(const double* offsets, std::int64_t n_channels);

/// Steps 1 and 2 for a run of sectors: add them into `total`, the spectrum of every
/// channel's reduced-rate series ([n_channels, n_low] interleaved complex, zero before
/// the first call). Sectors may be added over several calls, in order, so the caller
/// never holds the random numbers of more than one run.
///
/// `normals` holds, for `n_sectors` sectors with power starting at the `first`-th of them
/// (in increasing order of azimuth), n_low deviates for the real part followed by n_low
/// for the imaginary part: [n_sectors, 2, n_low]. `pulse` is the transmitted pulse at the
/// reduced rate, complex baseband about center_frequency. `offsets` [n_channels, 3] are
/// the element positions about the array centre (world).
void reverb_accumulate(Backend backend, const ReverbParams& params, const ReverbPower& power, std::int64_t first,
                       const double* normals, std::int64_t n_sectors, const std::complex<double>* pulse,
                       std::int64_t n_pulse, const double* offsets, std::int64_t n_channels, float* total,
                       int device = 0);

/// Step 3: from `total` to the time series of every channel in the record's window,
/// written to `out`: [n_channels, n_samples] interleaved complex, or real floats with
/// real_output. Runs on the host.
void reverb_finish(const ReverbParams& params, const float* total, std::int64_t n_channels, float* out);

/// All sectors at once: reverb_accumulate with `normals` [n_active, 2, n_low] for every
/// sector with power, then reverb_finish.
void reverb_series(Backend backend, const ReverbParams& params, const ReverbPower& power, const double* normals,
                   std::int64_t n_active, const std::complex<double>* pulse, std::int64_t n_pulse,
                   const double* offsets, std::int64_t n_channels, float* out, int device = 0);

}  // namespace sonar::sim
