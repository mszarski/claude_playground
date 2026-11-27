// D8 of docs/tasks/DOPPLER.md: the sweep's Doppler bank and the tone's Doppler filter bank in their batched form, the
// form the CUDA backend runs (doppler.hpp says what the two banks compute and why).
//
// The reference is the per-hypothesis loop of chain.cpp (sweep bank) and the loops of doppler.cpp tone_stage (tone), on
// the host. Here the same operations are a short sequence of whole-array steps on an executor (sonar/dsp/exec_host.hpp):
// CudaExec in doppler_cuda.cu, HostExec in doppler_batched.cpp, so that this formulation is tested on a machine without a
// device (`doppler.batched: on` on the host backend; tests/python/test_doppler_device.py).
//
// The kernels this file shares with the default chain (recon::BpKernel, the detector's three, the far-field beamformer's
// sum) are NOT instantiated for CUDA here: on the device they are launched through launch_on_device in their own .cu
// files (run_shared below; backproject_kernel.hpp says why), so the default chain's device code is the one it was.
//
//   sweep bank   the baseband record uploaded once a ping, its channels' spectra once per transform length (one length
//                for every hypothesis of the shipped sweeps); then, for as many hypotheses at a time as memory allows
//                (`batch`): each hypothesis' filter (dsp::compression_filter, built on the host) times every channel's
//                spectrum (BankMultiplyKernel), one batched inverse transform, the crop (BankCropKernel), the map of each
//                hypothesis by THE backprojection kernel (recon::BpKernel, unchanged, launched once per hypothesis on the
//                device-resident compressed samples; its group carries the hypothesis' read shift in start_time), and
//                the fold into the running maximum (BankFoldMaxKernel), or, with combine cfar, each (beam, hypothesis)'s
//                cell means (BankCellMeanKernel), their background by the detector's own kernels (detector_kernel.hpp,
//                one row per (hypothesis, beam)) and the normalised fold (BankNormFoldKernel). At the end the cells of the
//                combined map (BankCellsOfMaxKernel, or BankNormMapKernel) and one download of what the detector takes.
//   tone bank    the beams of the far-field beamformer (farfield_batched_device: they stay where they are made), each
//                beam's spectrum, the band cut (ToneCutKernel), one batched inverse transform, the matched filters of
//                every (beam, hypothesis, lag) (ToneBankKernel), the CFAR along rate (ToneCfarKernel) and the leakage rule
//                (ToneLeakKernel). One download of power, ratio and the flags; the grouping of flagged cells stays on the
//                host (it is a walk over a short list).
//
// Each kernel does the reference's operations in the reference's order, one thread per item, and a fold over hypotheses
// visits them in increasing order (a tie keeps the earlier one, as the reference). With HostExec the result is the
// reference's bit for bit (held by the default-off proof and test_doppler_device.py); on the device the transforms are
// cuFFT's and the maths library the device's, so the maps differ from the host's in the last bits (doppler_cuda.cu is
// compiled with --fmad=false, as detector_cuda.cu, so that no product is contracted into a fused multiply-add).
//
// C++17: this header is compiled by nvcc.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/vec.hpp"
#include "sonar/live/beamform_batched.hpp"
#include "sonar/live/detector_kernel.hpp"
#include "sonar/recon/backproject_kernel.hpp"

namespace sonar::live {

/// Seconds in each part of the Doppler stages since the last reset (doppler_profile), both formulations. The loop's
/// `compress` and `map` are its pulse_compress and beamformer calls; its `fold` the host fold.
struct DopplerProfile {
  double prepare = 0.0;    ///< host: the filters, the geometry, the batch's size
  double transfer = 0.0;   ///< uploads, allocations and downloads (batched)
  double compress = 0.0;   ///< the channels' transforms, the filter products, the inverse transforms, the crop
  double map = 0.0;        ///< the backprojection of every hypothesis
  double fold = 0.0;       ///< fold (max) or background and normalisation (cfar), and the combined map's cells
  double tone_beams = 0.0; ///< the tone path's beamformer
  double tone_bank = 0.0;  ///< the tone path's transforms, band cut and matched filters
  double tone_cfar = 0.0;  ///< the tone path's CFAR along rate and its leakage rule
  std::int64_t pings = 0, hypotheses = 0, batches = 0, tone_calls = 0, batched_calls = 0;
};
DopplerProfile doppler_profile();
void doppler_profile_reset();
DopplerProfile& doppler_profile_accumulator();   ///< where the stages add their seconds (not thread-safe, as BeamformProfile)

/// Runs a kernel the default chain also runs: on the host as any other, on the device through its own translation unit's
/// launch_on_device (found by argument-dependent lookup: recon::BpKernel, the detector's kernels).
template <class Exec, class K>
void run_shared(Exec& ex, std::int64_t n, const K& kernel) {
  if constexpr (Exec::on_device) {
    launch_on_device(kernel, n);
  } else {
    ex.run(n, kernel);
  }
}

// ------------------------------------------------------------------------------------------------------------------------
// Memory and the batch (host code; tests/python/test_doppler_device.py holds the arithmetic).

/// Bytes the batched sweep bank takes on its executor: `fixed` whatever the batch, `per_hypothesis` for each hypothesis
/// a batch holds. Counts every buffer sweep_bank_batched allocates (the transforms' own work memory is cuFFT's: the
/// caller leaves room for it, doppler_cuda.cu).
struct SweepBankBytes {
  std::int64_t fixed = 0, per_hypothesis = 0;
};
SweepBankBytes sweep_bank_bytes(std::int64_t n_channels, std::int64_t n_samples, std::int64_t max_fft, std::int64_t n_hypotheses,
                                std::int64_t n_bearings, std::int64_t n_cells, std::int64_t cell_samples, std::int64_t skip,
                                std::int64_t train, bool cfar, bool sloped);
/// Hypotheses a batch holds: `asked` when above 0 (at most n_hypotheses), else as many as fit in `budget_bytes` after the
/// fixed part, at least 1 and at most n_hypotheses.
std::int64_t sweep_bank_batch(const SweepBankBytes& bytes, std::int64_t n_hypotheses, std::int64_t budget_bytes, std::int64_t asked);
/// The batches in hypothesis order: runs of consecutive hypotheses with the same transform length, each at most `batch`
/// long. [first, count] pairs.
std::vector<std::int64_t> sweep_bank_runs(const std::vector<std::int64_t>& n_fft, std::int64_t batch);

// ------------------------------------------------------------------------------------------------------------------------
// Kernels of the sweep bank. Complex arrays are interleaved (re, im) floats.

/// [rows][n] -> zero-padded [rows][n_fft] (the reference's copy into its zeroed transform buffer).
struct BankPadKernel {
  const float* in;
  float* x;
  std::int64_t n, n_fft;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t k = i % n_fft, r = i / n_fft;
    if (k < n) {
      x[2 * i] = in[2 * (r * n + k)];
      x[2 * i + 1] = in[2 * (r * n + k) + 1];
    } else {
      x[2 * i] = 0.0F;
      x[2 * i + 1] = 0.0F;
    }
  }
};

/// y[t][c][k] = x[c][k] h[t][k]: every channel's spectrum times hypothesis t's filter (pulse_compress's x[i] *= h[i]).
struct BankMultiplyKernel {
  const float* x;   ///< [n_ch][n_fft]
  const float* h;   ///< [hc][n_fft]
  float* y;         ///< [hc][n_ch][n_fft]
  std::int64_t n_ch, n_fft;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t k = i % n_fft, c = (i / n_fft) % n_ch, t = i / (n_fft * n_ch);
    const float a = x[2 * (c * n_fft + k)], b = x[2 * (c * n_fft + k) + 1];
    const float hr = h[2 * (t * n_fft + k)], hi = h[2 * (t * n_fft + k) + 1];
    y[2 * i] = a * hr - b * hi;
    y[2 * i + 1] = a * hi + b * hr;
  }
};

/// The first n samples of each padded row with the inverse's 1 / n_fft (the reference's scaled inverse transform).
struct BankCropKernel {
  const float* y;   ///< [rows][n_fft]
  float* out;       ///< [rows][n]
  std::int64_t n, n_fft;
  float scale;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t k = i % n, r = i / n;
    out[2 * i] = y[2 * (r * n_fft + k)] * scale;
    out[2 * i + 1] = y[2 * (r * n_fft + k) + 1] * scale;
  }
};

/// The map's points, bearing major: origin + dirs[b] x ranges[q] (chain.cpp range_bearing_image's).
struct BankPointsKernel {
  Vec3d origin;
  const Vec3d* dirs;      ///< [n_b]
  const double* ranges;   ///< [n_r]
  Vec3d* points;          ///< [n_b][n_r]
  std::int64_t n_r;
  SONAR_HD void operator()(std::int64_t i) const { points[i] = origin + dirs[i / n_r] * ranges[i % n_r]; }
};

struct BankFillKernel {
  float* v;
  std::int32_t* w;   ///< may be null
  float value;
  SONAR_HD void operator()(std::int64_t i) const {
    v[i] = value;
    if (w != nullptr) w[i] = -1;
  }
};

/// The power of one map sample, as the reference's std::norm of a std::complex<float>.
SONAR_HD inline float bank_power(const float* image, std::int64_t i) {
  const float re = image[2 * i], im = image[2 * i + 1];
  return re * re + im * im;
}

/// combine max: per map sample, the running maximum over hypotheses h0 .. h0 + hc - 1 of the beam's own (fold_hypothesis).
struct BankFoldMaxKernel {
  const float* image;           ///< [hc][n_points] complex
  const std::int64_t* first;    ///< [n_b]
  std::int64_t per_beam, h0, hc, n_points, stride;
  float* best;                  ///< [n_points]
  std::int32_t* which;          ///< [n_points]
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t f = first[i / stride];
    for (std::int64_t t = 0; t < hc; ++t) {
      const std::int64_t h = h0 + t;
      if (h < f || h >= f + per_beam) continue;
      const float p = bank_power(image, t * n_points + i);
      if (p > best[i]) {           // strictly: a tie keeps the earlier hypothesis
        best[i] = p;
        which[i] = static_cast<std::int32_t>(h);
      }
    }
  }
};

/// combine cfar: the mean power of each kept cell of each (hypothesis t, beam b) row, in double then float
/// (fold_normalised's `cells`). Rows t * n_b + b, `kept` cells each.
struct BankCellMeanKernel {
  const float* image;   ///< [hc][n_b * stride] complex
  std::int64_t n_b, stride, k, skip, kept;
  float* cells;         ///< [hc * n_b][kept]
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t j = i % kept, r = i / kept;
    const std::int64_t base = r * stride + (skip + j) * k;   // row r = t * n_b + b: image [t][b][...]
    double sum = 0.0;
    for (std::int64_t q = 0; q < k; ++q) sum += static_cast<double>(bank_power(image, base + q));
    cells[i] = static_cast<float>(sum / static_cast<double>(k));
  }
};

/// combine cfar: per (beam, kept cell), the fold over hypotheses h0 .. h0 + hc - 1 of the beam's own (fold_normalised's
/// loop, the same comparisons): the centre hypothesis' background, the cell's and each sample's ratio to it.
struct BankNormFoldKernel {
  const float* image;           ///< [hc][n_b * stride] complex
  const double* level;          ///< [hc * n_b][kept]: each row's background (DetectorCellKernel's level)
  const float* cells;           ///< [hc * n_b][kept]
  const std::int64_t* first;    ///< [n_b]
  std::int64_t per_beam, h0, hc, n_b, stride, n_cells, k, skip, kept;
  float* ratio;                 ///< [n_b][stride]
  float* cell_ratio;            ///< [n_b][n_cells]
  double* reference;            ///< [n_b][n_cells]
  std::int32_t* which;          ///< [n_b][stride]
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t b = i / kept, j = i % kept, f = first[b];
    const std::int64_t cell = b * n_cells + skip + j;
    for (std::int64_t t = 0; t < hc; ++t) {
      const std::int64_t h = h0 + t;
      if (h < f || h >= f + per_beam) continue;
      const std::int64_t at = (t * n_b + b) * kept + j;
      const double lev = level[at];
      const double floor = lev < 1e-300 ? 1e-300 : lev;             // std::max(level, 1e-300)
      if (h == f + (per_beam - 1) / 2) reference[cell] = lev;
      const auto c = static_cast<float>(static_cast<double>(cells[at]) / floor);
      if (c > cell_ratio[cell]) cell_ratio[cell] = c;
      for (std::int64_t q = 0; q < k; ++q) {
        const std::int64_t idx = b * stride + (skip + j) * k + q;
        const auto r = static_cast<float>(static_cast<double>(bank_power(image, t * n_b * stride + idx)) / floor);
        if (r > ratio[idx]) {        // strictly: a tie keeps the earlier hypothesis
          ratio[idx] = r;
          which[idx] = static_cast<std::int32_t>(h);
        }
      }
    }
  }
};

/// combine max: the cells of the maximum map, their mean power in double then float (chain.cpp's loop).
struct BankCellsOfMaxKernel {
  const float* best;   ///< [n_b][stride]
  std::int64_t stride, k, n_cells;
  float* cells;        ///< [n_b][n_cells]
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t b = i / n_cells, j = i % n_cells;
    double p = 0.0;
    for (std::int64_t q = 0; q < k; ++q) p += static_cast<double>(best[b * stride + j * k + q]);
    cells[i] = static_cast<float>(p / static_cast<double>(k));
  }
};

/// combine cfar: the map the detector is handed (normalised_map): ratio x reference per sample, cell_ratio x reference per
/// cell, 0 in the skipped cells. `fine` holds the ratios on entry (in place).
struct BankNormMapKernel {
  float* fine;                 ///< [n_b][stride]
  const float* cell_ratio;     ///< [n_b][n_cells]
  const double* reference;     ///< [n_b][n_cells]
  float* cells;                ///< [n_b][n_cells]
  std::int64_t stride, k, n_cells, skip;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t b = i / n_cells, j = i % n_cells;
    float* row = fine + b * stride + j * k;
    if (j < skip) {
      cells[i] = 0.0F;
      for (std::int64_t q = 0; q < k; ++q) row[q] = 0.0F;
      return;
    }
    const double ref = reference[i];
    const float cr = cell_ratio[i];
    cells[i] = static_cast<float>(static_cast<double>(cr < 0.0F ? 0.0F : cr) * ref);   // std::max(cell_ratio, 0.0F)
    for (std::int64_t q = 0; q < k; ++q) {
      const float r = row[q];
      row[q] = static_cast<float>(static_cast<double>(r < 0.0F ? 0.0F : r) * ref);
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------------
// The sweep bank, batched.

struct SweepBankBatchedInput {
  const float* samples = nullptr;              ///< the baseband record before compression, [n_ch][n] complex
  std::int64_t n_ch = 0, n = 0;
  std::int64_t n_h = 0;                        ///< hypotheses (DopplerPlan::rates)
  const std::int64_t* n_fft = nullptr;         ///< [n_h] each hypothesis' transform length (compression_filter)
  const float* const* filters = nullptr;       ///< [n_h] host arrays of n_fft[h] complex: each hypothesis' filter
  const recon::BpGroup* groups = nullptr;      ///< [n_h] the record's group with start_time + the hypothesis' shift
  const recon::BpRecord* records = nullptr;    ///< [n_ch]
  recon::BpOptions bp;
  Vec3d origin;
  const Vec3d* dirs = nullptr;                 ///< [n_b] the map's bearing directions
  const double* ranges = nullptr;              ///< [n_cells * k] the fine ranges
  std::int64_t n_b = 0, n_cells = 0, k = 1;
  const std::int64_t* first = nullptr;         ///< [n_b] DopplerPlan::first
  std::int64_t per_beam = 1;
  bool cfar = false;
  std::int64_t skip = 0, guard = 0, train = 0; ///< cfar: blanked cells, the detector's window
  const double* offset = nullptr;              ///< cfar: [n_cells - skip] detector_background_window's offset
  bool sloped = false;
  std::int64_t batch = 1;                      ///< hypotheses a batch holds (sweep_bank_batch)
};

struct SweepBankBatchedOutput {
  float* fine = nullptr;           ///< [n_b][n_cells * k]: the maximum (max) or the normalised map (cfar)
  std::int32_t* which = nullptr;   ///< [n_b][n_cells * k]
  float* cells = nullptr;          ///< [n_b][n_cells]
  double* level = nullptr;         ///< cfar: [n_b][n_cells], the centre hypotheses' background (0 in skipped cells)
};

template <class Exec>
void sweep_bank_batched(Exec& ex, const SweepBankBatchedInput& in, const SweepBankBatchedOutput& out, DopplerProfile& prof) {
  BeamformStopwatch watch;
  const auto sz = [](std::int64_t v) { return static_cast<std::size_t>(v > 0 ? v : 0); };
  const std::int64_t stride = in.n_cells * in.k, n_pts = in.n_b * stride, kept = in.n_cells - in.skip;
  const std::int64_t n_ch = in.n_ch, n = in.n;
  std::vector<std::int64_t> lengths(in.n_fft, in.n_fft + in.n_h);
  std::int64_t max_fft = 1;
  for (std::int64_t v : lengths) max_fft = std::max(max_fft, v);
  const std::int64_t batch = std::clamp<std::int64_t>(in.batch, 1, std::max<std::int64_t>(in.n_h, 1));
  const std::vector<std::int64_t> runs = sweep_bank_runs(lengths, batch);
  watch.lap(prof.prepare);

  auto d_in = ex.upload(in.samples, sz(2 * n_ch * n));
  auto d_groups = ex.upload(in.groups, sz(in.n_h));
  auto d_records = ex.upload(in.records, sz(n_ch));
  auto d_dirs = ex.upload(in.dirs, sz(in.n_b));
  auto d_ranges = ex.upload(in.ranges, sz(stride));
  auto d_first = ex.upload(in.first, sz(in.n_b));
  auto d_points = ex.template alloc<Vec3d>(sz(n_pts));
  auto d_x = ex.template alloc<float>(sz(2 * n_ch * max_fft));
  auto d_y = ex.template alloc<float>(sz(2 * batch * n_ch * max_fft));
  auto d_c = ex.template alloc<float>(sz(2 * batch * n_ch * n));
  auto d_img = ex.template alloc<float>(sz(2 * batch * n_pts));
  auto d_best = ex.template alloc<float>(sz(n_pts));
  auto d_which = ex.template alloc<std::int32_t>(sz(n_pts));
  auto d_cells = ex.template alloc<float>(sz(in.n_b * in.n_cells));
  // combine cfar: the fold's state and one batch's backgrounds (one row per (hypothesis, beam)).
  const std::int64_t rows = in.cfar ? batch * in.n_b : 0, kc = in.cfar ? std::max<std::int64_t>(kept, 0) : 0;
  auto d_cell_ratio = ex.template alloc<float>(sz(in.cfar ? in.n_b * in.n_cells : 0));
  auto d_reference = ex.template alloc<double>(sz(in.cfar ? in.n_b * in.n_cells : 0));
  auto d_cellmean = ex.template alloc<float>(sz(rows * kc));
  auto d_scratch = ex.template alloc<double>(sz(in.cfar && in.sloped ? 4 * rows * in.train : 0));
  auto d_median = ex.template alloc<double>(sz(4 * rows));
  auto d_csum = ex.template alloc<double>(sz(rows * (kc + 1)));
  auto d_slope = ex.template alloc<double>(sz(2 * rows));
  auto d_level = ex.template alloc<double>(sz(rows * kc));
  auto d_bg = ex.template alloc<float>(sz(rows * kc));
  auto d_peak = ex.template alloc<float>(sz(rows * kc));
  auto d_flag = ex.template alloc<unsigned char>(sz(rows * kc));
  const std::vector<double> zeros(sz(kc), 0.0);
  const std::int64_t scale_cells = 0;
  auto d_offset = ex.upload(in.cfar ? in.offset : zeros.data(), sz(kc));
  auto d_threshold = ex.upload(zeros.data(), sz(kc));   // one test, threshold 0: only the background is used
  auto d_scales = ex.upload(&scale_cells, 1);
  watch.lap(prof.transfer);

  ex.run(n_pts, BankPointsKernel{in.origin, d_dirs.data(), d_ranges.data(), d_points.data(), stride});
  ex.run(n_pts, BankFillKernel{d_best.data(), d_which.data(), -1.0F});
  if (in.cfar) {
    ex.run(in.n_b * in.n_cells, BankFillKernel{d_cell_ratio.data(), nullptr, -1.0F});
    ex.zero(d_reference);
    ex.zero(d_median);
  }
  watch.lap(prof.fold);

  std::int64_t x_fft = -1;
  for (std::size_t r = 0; r + 1 < runs.size(); r += 2) {
    const std::int64_t h0 = runs[r], hc = runs[r + 1], nf = lengths[sz(h0)];
    // The batch's filters (built on the host, dsp::compression_filter), staged and uploaded together.
    float* stage = ex.template staging<float>(0, sz(2 * hc * nf));
    for (std::int64_t t = 0; t < hc; ++t) std::memcpy(stage + 2 * t * nf, in.filters[h0 + t], sz(2 * nf) * sizeof(float));
    auto d_h = ex.upload(stage, sz(2 * hc * nf));
    watch.lap(prof.transfer);
    if (nf != x_fft) {             // every channel's spectrum at this length: once a ping for the shipped sweeps
      ex.run(n_ch * nf, BankPadKernel{d_in.data(), d_x.data(), n, nf});
      ex.fft(d_x.data(), nf, n_ch, false);
      x_fft = nf;
    }
    ex.run(hc * n_ch * nf, BankMultiplyKernel{d_x.data(), d_h.data(), d_y.data(), n_ch, nf});
    ex.fft(d_y.data(), nf, hc * n_ch, true);
    ex.run(hc * n_ch * n, BankCropKernel{d_y.data(), d_c.data(), n, nf, 1.0F / static_cast<float>(nf)});
    watch.lap(prof.compress);
    ex.zero(d_img);
    for (std::int64_t t = 0; t < hc; ++t) {
      // THE backprojection kernel, as range_bearing_image calls it on the host, on this hypothesis' compressed samples.
      const recon::BpKernel bk{d_groups.data() + h0 + t, d_records.data(), d_c.data() + 2 * t * n_ch * n, d_points.data(), nullptr,
                               d_img.data() + 2 * t * n_pts, nullptr, 1, in.bp, nullptr, recon::TrajectoryAccess{}, 0, 0};
      run_shared(ex, n_pts, bk);
    }
    watch.lap(prof.map);
    if (!in.cfar) {
      ex.run(n_pts, BankFoldMaxKernel{d_img.data(), d_first.data(), in.per_beam, h0, hc, n_pts, stride, d_best.data(), d_which.data()});
    } else if (kc > 0) {
      const std::int64_t nr = hc * in.n_b;
      ex.run(nr * kc, BankCellMeanKernel{d_img.data(), in.n_b, stride, in.k, in.skip, kc, d_cellmean.data()});
      if (in.sloped) run_shared(ex, 4 * nr, DetectorMedianKernel{d_cellmean.data(), d_scratch.data(), d_median.data(), kc, in.train});
      run_shared(ex, nr, DetectorRowKernel{d_cellmean.data(), d_median.data(), d_csum.data(), d_slope.data(), kc, in.train, in.sloped});
      run_shared(ex, nr * kc, DetectorCellKernel{d_cellmean.data(), kc, 0, d_csum.data(), d_slope.data(), d_offset.data(), d_threshold.data(),
                                         d_scales.data(), 1, kc, 1, in.guard, in.train, d_level.data(), d_bg.data(), d_peak.data(),
                                         d_flag.data()});
      ex.run(in.n_b * kc, BankNormFoldKernel{d_img.data(), d_level.data(), d_cellmean.data(), d_first.data(), in.per_beam, h0, hc, in.n_b,
                                             stride, in.n_cells, in.k, in.skip, kc, d_best.data(), d_cell_ratio.data(),
                                             d_reference.data(), d_which.data()});
    }
    watch.lap(prof.fold);
    prof.hypotheses += hc;
    ++prof.batches;
  }
  if (!in.cfar) {
    ex.run(in.n_b * in.n_cells, BankCellsOfMaxKernel{d_best.data(), stride, in.k, in.n_cells, d_cells.data()});
  } else {
    ex.run(in.n_b * in.n_cells, BankNormMapKernel{d_best.data(), d_cell_ratio.data(), d_reference.data(), d_cells.data(), stride, in.k,
                                                  in.n_cells, in.skip});
  }
  watch.lap(prof.fold);
  ex.download(d_best, out.fine);
  ex.download(d_which, out.which);
  ex.download(d_cells, out.cells);
  if (in.cfar && out.level != nullptr) ex.download(d_reference, out.level);
  watch.lap(prof.transfer);
  ++prof.batched_calls;
}

// ------------------------------------------------------------------------------------------------------------------------
// Kernels of the tone bank (tone_stage's steps 1 to 3b; doppler.hpp).

/// Step 1, the band cut: dec[b][m] = spectrum[b][(shift + q) mod N] for the bins q = -h .. h about the tone (m = q mod M),
/// zero elsewhere (tone_stage's loop over q into its zeroed `dec`).
struct ToneCutKernel {
  const float* spec;   ///< [n_b][N]
  float* dec;          ///< [n_b][M]
  std::int64_t N, M, h, shift;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t m = i % M, b = i / M;
    std::int64_t q = 0;
    bool in_band = true;
    if (m <= h) {
      q = m;
    } else if (m >= M - h) {
      q = m - M;
    } else {
      in_band = false;
    }
    if (!in_band) {
      dec[2 * i] = 0.0F;
      dec[2 * i + 1] = 0.0F;
      return;
    }
    const std::int64_t src = ((shift + q) % N + N) % N;
    dec[2 * i] = spec[2 * (b * N + src)];
    dec[2 * i + 1] = spec[2 * (b * N + src) + 1];
  }
};

/// x scale in place (the reference's scaled inverse transform, dsp::fft(..., true): 1 / M).
struct ToneScaleKernel {
  float* x;
  float scale;
  SONAR_HD void operator()(std::int64_t i) const { x[i] = x[i] * scale; }
};

/// Step 2, the bank: power of (beam b, own hypothesis j, lag l), |sum_m z[l H + m] ph_k[m]|^2 (M / N)^2 / sum w^2, in
/// double as the reference (its std::complex<double> products and sum, the same order).
struct ToneBankKernel {
  const float* dec;            ///< [n_b][M] the decimated, scaled beams
  const double* phasor;        ///< [n_r][L] complex (re, im)
  const std::int64_t* first;   ///< [n_b]
  std::int64_t M, L, H, J, NL;
  double scale, sw2;
  float* power;                ///< [n_b][J][NL]
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t l = i % NL, j = (i / NL) % J, b = i / (NL * J);
    const std::int64_t k = first[b] + j;
    const float* z = dec + 2 * (b * M + l * H);
    const double* ph = phasor + 2 * k * L;
    double ar = 0.0, ai = 0.0;
    for (std::int64_t m = 0; m < L; ++m) {
      const auto zr = static_cast<double>(z[2 * m]), zi = static_cast<double>(z[2 * m + 1]);
      const double pr = ph[2 * m], pi = ph[2 * m + 1];
      ar += zr * pr - zi * pi;
      ai += zr * pi + zi * pr;
    }
    power[i] = static_cast<float>((ar * ar + ai * ai) * scale * scale / sw2);
  }
};

/// Step 3, the CFAR along rate per (beam, lag, own hypothesis), notch excluded: the ratio to the training mean (0 where
/// not tested) and the flag against the cell-averaging factor of the training count (`factor` [2 train + 1], made on the
/// host by the reference's function).
struct ToneCfarKernel {
  const float* power;              ///< [n_b][J][NL]
  const unsigned char* notched;    ///< [n_b][J]
  const double* factor;            ///< [2 train + 1], by training count
  std::int64_t J, NL, guard, train;
  float* ratio;                    ///< [n_b][J][NL]
  unsigned char* flag;             ///< [n_b][J][NL]
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t l = i % NL, j = (i / NL) % J, b = i / (NL * J);
    ratio[i] = 0.0F;
    flag[i] = 0;
    if (notched[b * J + j]) return;
    double sum = 0.0;
    std::int64_t count = 0;
    for (std::int64_t d = guard + 1; d <= guard + train; ++d) {
      for (int side = 0; side < 2; ++side) {
        const std::int64_t q = side == 0 ? j - d : j + d;
        if (q < 0 || q >= J || notched[b * J + q]) continue;
        sum += static_cast<double>(power[(b * J + q) * NL + l]);
        ++count;
      }
    }
    if (count == 0 || !(sum > 0.0)) return;
    const double mean = sum / static_cast<double>(count);
    const double x = static_cast<double>(power[i]) / mean;
    ratio[i] = static_cast<float>(x);
    flag[i] = x > factor[count] ? 1 : 0;
  }
};

/// Step 3b, the leakage rule on a flagged cell: kept when its power is more than `margin` times the largest leak of a
/// stronger source of its beam and lag (a notched hypothesis or a tested cell) through the window's spectrum envelope.
struct ToneLeakKernel {
  const float* power;              ///< [n_b][J][NL]
  const float* ratio;              ///< [n_b][J][NL]
  const unsigned char* notched;    ///< [n_b][J]
  const double* env;               ///< [J + 1]
  std::int64_t J, NL;
  double margin;
  unsigned char* flag;             ///< [n_b][J][NL], in place
  SONAR_HD void operator()(std::int64_t i) const {
    if (!flag[i]) return;
    const std::int64_t l = i % NL, j = (i / NL) % J, b = i / (NL * J);
    const auto me = static_cast<double>(power[i]);
    double leak = 0.0;
    for (std::int64_t q = 0; q < J; ++q) {
      if (q == j) continue;
      const std::int64_t at = (b * J + q) * NL + l;
      const bool source = notched[b * J + q] != 0 || static_cast<double>(ratio[at]) > 0.0;
      if (!source) continue;
      const auto other = static_cast<double>(power[at]);
      if (other <= me) continue;
      const std::int64_t d = q > j ? q - j : j - q;
      const double v = other * env[d];
      leak = leak < v ? v : leak;      // std::max(leak, v)
    }
    flag[i] = me > margin * leak ? 1 : 0;
  }
};

struct ToneBankBatchedInput {
  /// The beams [n_b][n] complex: `beams` on the host (uploaded), or `device_beams` already in the executor's memory (the
  /// cuda wrapper makes them there from `samples` [n_el][n], `w` [n_el], `tau` [n_b][n_el], `df`, `fc` by the far-field
  /// beamformer in beamform_cuda.cu, farfield_cuda_device).
  const float* beams = nullptr;
  const float* device_beams = nullptr;
  const float* samples = nullptr;
  const float* w = nullptr;
  const double* tau = nullptr;
  std::int64_t n_el = 0;
  double df = 0.0, fc = 0.0;
  std::int64_t n_b = 0, n = 0;
  std::int64_t N = 0, M = 0, h = 0, shift = 0, L = 0, H = 0, J = 0, NL = 0, n_r = 0;
  double scale = 1.0, sw2 = 1.0;
  const double* phasor = nullptr;        ///< [n_r][L] complex
  const std::int64_t* first = nullptr;   ///< [n_b]
  const unsigned char* notched = nullptr;   ///< [n_b][J]
  const double* factor = nullptr;        ///< [2 train + 1]
  std::int64_t guard = 0, train = 0;
  const double* env = nullptr;           ///< [J + 1]
  double margin = 1.0;
};

/// power, ratio, flag: host arrays of n_b x J x NL.
template <class Exec>
void tone_bank_batched(Exec& ex, const ToneBankBatchedInput& in, float* power, float* ratio, unsigned char* flag, DopplerProfile& prof) {
  BeamformStopwatch watch;
  const auto sz = [](std::int64_t v) { return static_cast<std::size_t>(v > 0 ? v : 0); };
  const std::int64_t cells = in.n_b * in.J * in.NL;
  auto d_buf = ex.template alloc<float>(sz(2 * in.n_b * in.N));
  if (in.device_beams != nullptr) {
    ex.run(in.n_b * in.N, BankPadKernel{in.device_beams, d_buf.data(), in.n, in.N});
  } else {
    auto d_beams = ex.upload(in.beams, sz(2 * in.n_b * in.n));
    watch.lap(prof.transfer);
    ex.run(in.n_b * in.N, BankPadKernel{d_beams.data(), d_buf.data(), in.n, in.N});
  }
  auto d_dec = ex.template alloc<float>(sz(2 * in.n_b * in.M));
  auto d_phasor = ex.upload(in.phasor, sz(2 * in.n_r * in.L));
  auto d_first = ex.upload(in.first, sz(in.n_b));
  auto d_notched = ex.upload(in.notched, sz(in.n_b * in.J));
  auto d_factor = ex.upload(in.factor, sz(2 * in.train + 1));
  auto d_env = ex.upload(in.env, sz(in.J + 1));
  auto d_power = ex.template alloc<float>(sz(cells));
  auto d_ratio = ex.template alloc<float>(sz(cells));
  auto d_flag = ex.template alloc<unsigned char>(sz(cells));
  watch.lap(prof.transfer);
  ex.fft(d_buf.data(), in.N, in.n_b, false);
  ex.run(in.n_b * in.M, ToneCutKernel{d_buf.data(), d_dec.data(), in.N, in.M, in.h, in.shift});
  ex.fft(d_dec.data(), in.M, in.n_b, true);
  ex.run(2 * in.n_b * in.M, ToneScaleKernel{d_dec.data(), 1.0F / static_cast<float>(in.M)});
  ex.run(cells, ToneBankKernel{d_dec.data(), d_phasor.data(), d_first.data(), in.M, in.L, in.H, in.J, in.NL, in.scale, in.sw2, d_power.data()});
  watch.lap(prof.tone_bank);
  ex.run(cells, ToneCfarKernel{d_power.data(), d_notched.data(), d_factor.data(), in.J, in.NL, in.guard, in.train, d_ratio.data(), d_flag.data()});
  ex.run(cells, ToneLeakKernel{d_power.data(), d_ratio.data(), d_notched.data(), d_env.data(), in.J, in.NL, in.margin, d_flag.data()});
  watch.lap(prof.tone_cfar);
  ex.download(d_power, power);
  ex.download(d_ratio, ratio);
  ex.download(d_flag, flag);
  watch.lap(prof.transfer);
}

}  // namespace sonar::live
