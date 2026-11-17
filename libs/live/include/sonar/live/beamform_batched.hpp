// The far-field beamformers in their batched form: what the CUDA backend runs.
//
// The reference host code (beamform.cpp) works bearing by bearing and sub-band by sub-band
// on the CPU threads. Here the same sums are a short sequence of whole-array steps on an
// executor (sonar/dsp/exec_host.hpp): CudaExec in beamform_cuda.cu, and HostExec in
// beamform.cpp so that this formulation is tested on a machine without a device.
//
//   farfield   transform of every channel; the weighted, steered sum for every bearing and
//              bin (FarfieldSumKernel); inverse transform of every beam.
//   mvdr       transform of every channel; then, for as many sub-bands at a time as the
//              memory budget allows: the sub-band of every channel in time (MvdrBandKernel,
//              one batched inverse transform, MvdrTransposeKernel); x x^H of every range
//              block and their sum (MvdrOuterKernel, MvdrTotalKernel); the training covariance
//              of every block with its loading and its Cholesky factor (MvdrFactorKernel);
//              the weights of every block and bearing (MvdrSolveKernel); the weights applied
//              and the sub-bands summed (MvdrApplyKernel).
//
// The covariances are accumulated in double precision from the single-precision samples
// (the reference forms each block's product in single precision), so the two agree to the
// rounding of the reference, magnified by the conditioning of the loaded covariance.
//
// C++17: this header is compiled by nvcc.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sonar/live/beamform_kernel.hpp"

namespace sonar::live {

/// `samples` [n_el][n] interleaved complex (conditioned), `w` [n_el], `tau` [n_b][n_el];
/// `out` [n_b][n] interleaved complex.
template <class Exec>
void farfield_batched(Exec& ex, const float* samples, const float* w, const double* tau, std::int64_t n,
                      std::int64_t n_el, std::int64_t n_b, double df, double fc, float* out) {
  auto x = ex.upload(samples, static_cast<std::size_t>(2 * n_el * n));
  auto d_w = ex.upload(w, static_cast<std::size_t>(n_el));
  auto d_tau = ex.upload(tau, static_cast<std::size_t>(n_b * n_el));
  auto beams = ex.template alloc<float>(static_cast<std::size_t>(2 * n_b * n));
  ex.fft(x.data(), n, n_el, false);
  ex.run(n_b * n, FarfieldSumKernel{x.data(), d_w.data(), d_tau.data(), beams.data(), n, n_el, df, fc,
                                    1.0F / static_cast<float>(n)});
  ex.fft(beams.data(), n, n_b, true);
  ex.download(beams, out);
}

struct MvdrBatchedInput {
  const float* samples;       ///< [n_el][n] interleaved complex (conditioned)
  const double* tau;          ///< [n_b][n_el]
  const std::int32_t* band;   ///< [n] sub-band of each frequency bin, or -1
  const std::int32_t* sub;    ///< [n_sub] the sub-bands that have bins, increasing
  const double* f_abs;        ///< [n_sub] their centre frequencies, Hz
  std::int64_t n, n_el, n_b, n_sub;
  std::int64_t block, guard_blocks;
  double loading;
};

/// Device memory the sub-band arrays of one run may take, bytes.
constexpr std::int64_t kMvdrBatchBytes = std::int64_t{1} << 30;

/// Result of mvdr_batched.
enum class MvdrBatchedStatus {
  Ok,
  NotPositiveDefinite,  ///< a covariance has no Cholesky factor: the caller uses the reference code (LU)
  Singular,             ///< a^H R^-1 a is zero or not finite somewhere
};

/// `out` [n_b][n] interleaved complex, host memory, written.
template <class Exec>
MvdrBatchedStatus mvdr_batched(Exec& ex, const MvdrBatchedInput& in, float* out) {
  const std::int64_t n = in.n, n_el = in.n_el, n_b = in.n_b;
  const std::int64_t n_blocks = (n + in.block - 1) / in.block;
  const std::int64_t nn = n_el * n_el;
  auto x = ex.upload(in.samples, static_cast<std::size_t>(2 * n_el * n));
  auto d_tau = ex.upload(in.tau, static_cast<std::size_t>(n_b * n_el));
  auto d_band = ex.upload(in.band, static_cast<std::size_t>(n));
  auto d_out = ex.template alloc<float>(static_cast<std::size_t>(2 * n_b * n));
  ex.zero(d_out);
  ex.fft(x.data(), n, n_el, false);

  // Bytes of the arrays below for one sub-band.
  const std::int64_t per_sub = 2 * (8 * n_el * n) + 2 * (16 * n_blocks * nn) + 16 * nn + 24 * n_blocks * n_b * n_el +
                               4 * n_blocks * (n_b + 1);
  std::int64_t run = kMvdrBatchBytes / per_sub;
  if (run < 1) run = 1;
  if (run > in.n_sub) run = in.n_sub;
  const auto count = [&](std::int64_t per) { return static_cast<std::size_t>(run * per); };
  auto y = ex.template alloc<float>(count(2 * n_el * n));
  auto xt = ex.template alloc<float>(count(2 * n_el * n));
  auto outer = ex.template alloc<double>(count(2 * n_blocks * nn));
  auto total = ex.template alloc<double>(count(2 * nn));
  auto r = ex.template alloc<double>(count(2 * n_blocks * nn));
  auto work = ex.template alloc<double>(count(2 * n_blocks * n_b * n_el));
  auto coeff = ex.template alloc<float>(count(2 * n_blocks * n_b * n_el));
  auto status = ex.template alloc<std::int32_t>(count(n_blocks));
  auto singular = ex.template alloc<std::int32_t>(count(n_blocks * n_b));
  std::vector<std::int32_t> h_status(count(n_blocks)), h_singular(count(n_blocks * n_b));
  const std::int64_t n_groups = (n_b + kMvdrApplyGroup - 1) / kMvdrApplyGroup;

  MvdrBatchedStatus result = MvdrBatchedStatus::Ok;
  for (std::int64_t first = 0; first < in.n_sub; first += run) {
    const std::int64_t n_sub = in.n_sub - first < run ? in.n_sub - first : run;
    auto d_sub = ex.upload(in.sub + first, static_cast<std::size_t>(n_sub));
    auto d_f = ex.upload(in.f_abs + first, static_cast<std::size_t>(n_sub));
    ex.run(n_sub * n_el * n, MvdrBandKernel{x.data(), d_band.data(), d_sub.data(), y.data(), n, n_el,
                                            1.0F / static_cast<float>(n)});
    ex.fft(y.data(), n, n_sub * n_el, true);
    ex.run(n_sub * n * n_el, MvdrTransposeKernel{y.data(), xt.data(), n, n_el});
    ex.run(n_sub * n_blocks * nn, MvdrOuterKernel{xt.data(), outer.data(), n, n_el, n_blocks, in.block});
    ex.run(n_sub * nn, MvdrTotalKernel{outer.data(), total.data(), n_el, n_blocks});
    ex.run(n_sub * n_blocks, MvdrFactorKernel{outer.data(), total.data(), r.data(), status.data(), n, n_el, n_blocks,
                                              in.block, in.guard_blocks, in.loading});
    ex.run(n_sub * n_blocks * n_b, MvdrSolveKernel{r.data(), status.data(), d_tau.data(), d_f.data(), work.data(),
                                                   coeff.data(), singular.data(), n_el, n_blocks, n_b});
    ex.download(status, h_status.data());
    ex.download(singular, h_singular.data());
    for (std::int64_t k = 0; k < n_sub * n_blocks; ++k) {
      if (h_status[static_cast<std::size_t>(k)] != 0) return MvdrBatchedStatus::NotPositiveDefinite;
    }
    for (std::int64_t k = 0; k < n_sub * n_blocks * n_b; ++k) {
      if (h_singular[static_cast<std::size_t>(k)] != 0) result = MvdrBatchedStatus::Singular;
    }
    ex.run(n_groups * n, MvdrApplyKernel{xt.data(), coeff.data(), d_out.data(), n, n_el, n_blocks, n_b, n_sub, in.block});
  }
  ex.download(d_out, out);
  return result;
}

}  // namespace sonar::live
