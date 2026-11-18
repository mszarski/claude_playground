// Scattering-function reverberation in its batched form: what the CUDA backend runs.
//
// The reference host code (reverb.cpp) transforms one component at a time on a CPU thread
// (its sequence, and with a Doppler shift its pulse too) and hands the device only the
// steering sum. Those transforms are nearly all of the work: two transforms of the
// reduced-rate length per component, a length that is rarely a product of small primes.
// Here a whole run of components is on the device from the shaped random sequences on:
//
//   host    the normal deviates of each component -> single precision (page-locked
//           memory), its row of the power density, the plane-wave lead of each channel
//   device  sigma(t) x deviates, Doppler shift        ReverbShapeKernel
//           transform of every sequence               one batched FFT
//           shifted, zero-padded pulses + transform   ReverbPulseKernel, one batched FFT
//           sequence spectrum x pulse spectrum        ReverbProductKernel
//           steering into the channels and the sum    ReverbSteerKernel (as on the host)
//   and, once per call of the generator, the interpolation to the record's rate:
//           zero-padded spectrum, inverse transform   ReverbPadKernel, one batched FFT
//           window, gain, carrier                     ReverbWindowKernel
//
// The sequence is a template over the executor (sonar/dsp/exec_host.hpp): CudaExec in
// reverb_cuda.cu, and HostExec in reverb.cpp so that this formulation is tested on the
// host. Components are added in their order in every channel and bin, as on the host, so
// the two differ only by the rounding of single-precision transforms and phases.
//
// A generator call that adds its components over several calls of reverb_accumulate can
// leave the accumulated spectrum on the executor between them (ReverbBatchSetup::from_host,
// to_host), so it crosses to the device once and back once.
//
// C++17: this header is compiled by nvcc.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "sonar/sim/reverb_kernel.hpp"

namespace sonar::sim {

/// Seconds spent in each step of the batched generator since the last reset, and counts.
struct ReverbProfile {
  double shape = 0.0;     ///< host: deviates to single precision, density rows, leads
  double upload = 0.0;    ///< deviates, density rows, leads, shifts to the device
  double shift = 0.0;     ///< ReverbShapeKernel
  double transform = 0.0; ///< the transforms of the sequences
  double pulse = 0.0;     ///< shifted pulses and their transforms
  double product = 0.0;   ///< ReverbProductKernel
  double steer = 0.0;     ///< ReverbSteerKernel
  double download = 0.0;  ///< the accumulated spectrum back to the host (and its upload)
  double finish = 0.0;    ///< interpolation to the record's rate, all of it
  double active = 0.0;    ///< finding the rows with power (both formulations)
  std::int64_t calls = 0, components = 0;
};

class ReverbStopwatch {
 public:
  ReverbStopwatch() : t_(std::chrono::steady_clock::now()) {}
  /// Add the time since the last call (or construction) to `to`.
  void lap(double& to) {
    const auto now = std::chrono::steady_clock::now();
    to += std::chrono::duration<double>(now - t_).count();
    t_ = now;
  }

 private:
  std::chrono::steady_clock::time_point t_;
};

/// What stays the same for every run of components of one call.
struct ReverbBatchSetup {
  const float* omega;           ///< [n_low] 2 pi (f + fc)
  const float* pulse;           ///< [n_taps] interleaved complex: the pulse at the reduced rate
  const float* pulse_spectrum;  ///< [n_low] interleaved complex: its transform, for components without a shift
  const double* coarse;         ///< [n_coarse] times of the power density, s
  std::int64_t n_low, n_taps, n_channels, n_coarse;
  double fs_low, time0, dphi;
  bool uniform;
  bool shared;                  ///< one density row for every component
  /// The accumulated spectrum comes from the host at the start of the call and goes back at
  /// its end. A caller that adds over several calls clears from_host after the first and
  /// to_host before the last: the spectrum then stays on the executor between them.
  bool from_host, to_host;
};

/// The accumulated spectrum left on an executor between calls.
template <class Exec>
struct ReverbResident {
  typename Exec::template Buffer<float> total;
  const float* host = nullptr;
  std::size_t size = 0;
};
template <class Exec>
ReverbResident<Exec>& reverb_resident() {
  static ReverbResident<Exec> resident;
  return resident;
}

/// One call of the generator on an executor: construct with the spectrum accumulated so
/// far, add runs of components, then download().
template <class Exec>
class ReverbBatched {
 public:
  ReverbBatched(Exec& ex, const ReverbBatchSetup& setup, const float* total, ReverbProfile& profile)
      : ex_(ex), s_(setup), profile_(profile) {
    ReverbStopwatch w;
    omega_ = ex_.upload(s_.omega, static_cast<std::size_t>(s_.n_low));
    pulse_ = ex_.upload(s_.pulse, static_cast<std::size_t>(2 * s_.n_taps));
    pulse_spectrum_ = ex_.upload(s_.pulse_spectrum, static_cast<std::size_t>(2 * s_.n_low));
    coarse_ = ex_.upload(s_.coarse, static_cast<std::size_t>(s_.n_coarse));
    const auto size = static_cast<std::size_t>(2 * s_.n_channels * s_.n_low);
    ReverbResident<Exec>& resident = reverb_resident<Exec>();
    if (s_.from_host) {
      total_ = ex_.upload(total, size);
    } else {
      if (resident.host != total || resident.size != size) {
        throw std::runtime_error("reverb: no accumulated spectrum of this call was left on the backend");
      }
      total_ = std::move(resident.total);
    }
    resident.host = nullptr;
    resident.size = 0;
    w.lap(profile_.download);
  }

  /// Add `count` components: `deviates` [count, n_low] interleaved complex (standard normal),
  /// `density` [count, n_coarse] (one row when shared), `lead` [count, n_channels] seconds,
  /// `doppler` [count] Hz.
  void add(const float* deviates, const double* density, const float* lead, const double* doppler, std::int64_t count) {
    if (count <= 0) return;
    const std::int64_t n_low = s_.n_low;
    const auto points = static_cast<std::size_t>(count * n_low);
    bool any_shift = false;
    for (std::int64_t j = 0; j < count; ++j) any_shift = any_shift || doppler[j] != 0.0;
    ReverbStopwatch w;
    auto x = ex_.upload(deviates, 2 * points);
    auto d_density = ex_.upload(density, static_cast<std::size_t>((s_.shared ? 1 : count) * s_.n_coarse));
    auto d_lead = ex_.upload(lead, static_cast<std::size_t>(count * s_.n_channels));
    auto d_doppler = ex_.upload(doppler, static_cast<std::size_t>(count));
    w.lap(profile_.upload);
    ex_.run(count * n_low, ReverbShapeKernel{x.data(), coarse_.data(), d_density.data(), d_doppler.data(), n_low,
                                             s_.n_coarse, s_.time0, s_.fs_low, s_.dphi, s_.shared});
    w.lap(profile_.shift);
    ex_.fft(x.data(), n_low, count, false);
    w.lap(profile_.transform);
    if (any_shift) {
      auto p = ex_.template alloc<float>(2 * points);
      ex_.run(count * n_low, ReverbPulseKernel{pulse_.data(), d_doppler.data(), p.data(), n_low, s_.n_taps, s_.fs_low});
      ex_.fft(p.data(), n_low, count, false);
      w.lap(profile_.pulse);
      ex_.run(count * n_low, ReverbProductKernel{x.data(), p.data(), n_low, false});
    } else {
      ex_.run(count * n_low, ReverbProductKernel{x.data(), pulse_spectrum_.data(), n_low, true});
    }
    w.lap(profile_.product);
    const ReverbSteerKernel steer{x.data(), d_lead.data(), omega_.data(), total_.data(), count,
                                  s_.n_channels, n_low,         1,             s_.uniform};
    ex_.run(steer.size(), steer);
    w.lap(profile_.steer);
    profile_.components += count;
  }

  /// The end of the call: the spectrum back to the host, or left on the executor.
  void finish(float* total) {
    ReverbStopwatch w;
    if (s_.to_host) {
      ex_.download(total_, total);
    } else {
      ReverbResident<Exec>& resident = reverb_resident<Exec>();
      resident.size = total_.size();
      resident.host = total;
      resident.total = std::move(total_);
    }
    w.lap(profile_.download);
  }

 private:
  Exec& ex_;
  ReverbBatchSetup s_;
  ReverbProfile& profile_;
  typename Exec::template Buffer<float> omega_, pulse_, pulse_spectrum_, total_;
  typename Exec::template Buffer<double> coarse_;
};

/// Fills the host arrays of components begin .. begin + count of a call: `deviates`
/// [count, n_low] interleaved complex, `density` [count, n_coarse] (one row when shared),
/// `lead` [count, n_channels], `doppler` [count].
using ReverbShapeFn = void (*)(void* context, std::int64_t begin, std::int64_t count, float* deviates, double* density,
                               float* lead, double* doppler);

/// Complex samples of the sequences held on the executor at once.
constexpr std::int64_t kReverbBatchPoints = std::int64_t{4} << 20;

/// All the components of one call, in runs of at most kReverbBatchPoints samples.
template <class Exec>
void reverb_accumulate_batched(Exec& ex, const ReverbBatchSetup& setup, std::int64_t n_components, ReverbShapeFn shape,
                               void* context, float* total, ReverbProfile& profile) {
  ReverbBatched<Exec> batched(ex, setup, total, profile);
  const std::int64_t most = kReverbBatchPoints / setup.n_low;
  const std::int64_t batch = most < 1 ? 1 : most;
  for (std::int64_t begin = 0; begin < n_components; begin += batch) {
    const std::int64_t count = n_components - begin < batch ? n_components - begin : batch;
    auto* deviates = ex.template staging<float>(0, static_cast<std::size_t>(2 * count * setup.n_low));
    auto* lead = ex.template staging<float>(1, static_cast<std::size_t>(count * setup.n_channels));
    auto* doppler = ex.template staging<double>(2, static_cast<std::size_t>(count));
    auto* density = ex.template staging<double>(3, static_cast<std::size_t>((setup.shared ? 1 : count) * setup.n_coarse));
    ReverbStopwatch w;
    shape(context, begin, count, deviates, density, lead, doppler);
    w.lap(profile.shape);
    batched.add(deviates, density, lead, doppler, count);
  }
  batched.finish(total);
}

/// The interpolation to the record's rate (step 3 of reverb.hpp) on an executor. `out` is
/// [n_channels, n] interleaved complex, or real with real_output.
template <class Exec>
void reverb_finish_batched(Exec& ex, const float* total, std::int64_t n_channels, std::int64_t n_low, std::int64_t n,
                           std::int64_t decimation, std::int64_t start, double sample_rate, double start_time,
                           double center_frequency, bool carrier, bool real_output, float* out) {
  const std::int64_t n_full = n_low * decimation;
  auto d_total = ex.upload(total, static_cast<std::size_t>(2 * n_channels * n_low));
  auto series = ex.template alloc<float>(static_cast<std::size_t>(2 * n_channels * n_full));
  ex.run(n_channels * n_full, ReverbPadKernel{d_total.data(), series.data(), n_low, n_full});
  ex.fft(series.data(), n_full, n_channels, true);
  auto d_out = ex.template alloc<float>(static_cast<std::size_t>((real_output ? 1 : 2) * n_channels * n));
  ex.run(n_channels * n, ReverbWindowKernel{series.data(), d_out.data(), n, n_full, start, decimation, sample_rate,
                                            start_time, center_frequency, carrier, real_output});
  ex.download(d_out, out);
}

}  // namespace sonar::sim
