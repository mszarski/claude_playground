// D8: the host side of the batched Doppler stages (doppler_batched.hpp): the memory and batch arithmetic, the inputs the
// kernels take (each hypothesis' filter, the backprojection's groups), and the executor: HostExec here, CudaExec in
// doppler_cuda.cu.
#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "sonar/dsp/exec_host.hpp"
#include "sonar/dsp/pulse_compression.hpp"
#include "sonar/live/detector.hpp"
#include "sonar/live/doppler.hpp"
#include "sonar/live/doppler_batched.hpp"
#include "sonar/recon/backproject.hpp"

namespace sonar::live {

namespace {
DopplerProfile g_doppler_profile;
}  // namespace

DopplerProfile doppler_profile() { return g_doppler_profile; }
void doppler_profile_reset() { g_doppler_profile = DopplerProfile{}; }
DopplerProfile& doppler_profile_accumulator() { return g_doppler_profile; }

#if SONAR_HAVE_CUDA
// doppler_cuda.cu: the same templates on the device.
void sweep_bank_cuda(const SweepBankBatchedInput& in, const SweepBankBatchedOutput& out, DopplerProfile& prof, int device);
std::int64_t doppler_cuda_free_bytes(int device);
#endif

SweepBankBytes sweep_bank_bytes(std::int64_t n_ch, std::int64_t n, std::int64_t max_fft, std::int64_t n_h, std::int64_t n_b,
                                std::int64_t n_cells, std::int64_t k, std::int64_t skip, std::int64_t train, bool cfar, bool sloped) {
  // Every buffer of sweep_bank_batched, in bytes (8 a complex float, 4 a float or int32, 8 a double, 24 a Vec3d).
  const std::int64_t stride = n_cells * k, n_pts = n_b * stride, kept = std::max<std::int64_t>(n_cells - skip, 0);
  SweepBankBytes b;
  b.fixed = 8 * n_ch * n                  // the record
            + 8 * n_ch * max_fft          // its channels' spectra
            + static_cast<std::int64_t>(sizeof(recon::BpGroup)) * n_h + static_cast<std::int64_t>(sizeof(recon::BpRecord)) * n_ch
            + 24 * n_b + 8 * stride + 8 * n_b   // directions, ranges, first
            + 24 * n_pts                  // points
            + 4 * n_pts + 4 * n_pts       // best (or ratio), which
            + 4 * n_b * n_cells;          // cells
  if (cfar) b.fixed += 4 * n_b * n_cells + 8 * n_b * n_cells + 8 * kept * 2 + 8;   // cell_ratio, reference, offset, threshold
  b.per_hypothesis = 8 * max_fft          // its filter
                     + 8 * n_ch * max_fft // its products (then inverse transforms)
                     + 8 * n_ch * n       // its compressed samples
                     + 8 * n_pts;         // its map
  if (cfar) {
    b.per_hypothesis += n_b * (4 * kept                     // cell means
                               + (sloped ? 8 * 4 * train : 0) + 8 * 4 + 8 * (kept + 1) + 8 * 2   // median scratch, medians, csum, slope
                               + 8 * kept + 4 * kept + 4 * kept + kept);                       // level, background, peak, flag
  }
  return b;
}

std::int64_t sweep_bank_batch(const SweepBankBytes& bytes, std::int64_t n_h, std::int64_t budget, std::int64_t asked) {
  if (n_h <= 0) return 0;
  if (asked > 0) return std::min(asked, n_h);
  if (bytes.per_hypothesis <= 0) return n_h;
  const std::int64_t room = budget - bytes.fixed;
  return std::clamp<std::int64_t>(room / bytes.per_hypothesis, 1, n_h);
}

std::vector<std::int64_t> sweep_bank_runs(const std::vector<std::int64_t>& n_fft, std::int64_t batch) {
  std::vector<std::int64_t> runs;
  const auto n_h = static_cast<std::int64_t>(n_fft.size());
  const std::int64_t b = std::max<std::int64_t>(batch, 1);
  for (std::int64_t h0 = 0; h0 < n_h;) {
    std::int64_t hc = 1;
    while (h0 + hc < n_h && hc < b && n_fft[static_cast<std::size_t>(h0 + hc)] == n_fft[static_cast<std::size_t>(h0)]) ++hc;
    runs.push_back(h0);
    runs.push_back(hc);
    h0 += hc;
  }
  return runs;
}

SweepBankMaps sweep_bank_maps(const PingRecord& base, const DopplerPlan& plan, const MapGeometry& geometry, const std::vector<double>& fine_ranges,
                              std::int64_t n_cells, std::int64_t k, std::int64_t skip, std::int64_t guard, std::int64_t train, Backend backend,
                              int device, std::int64_t batch) {
  DopplerProfile& prof = doppler_profile_accumulator();
  const auto t0 = std::chrono::steady_clock::now();
  const Samples& s = base.samples;
  if (s.kind != SampleKind::ComplexBaseband) throw std::invalid_argument("sweep_bank_maps: the record must be complex baseband before compression");
  const auto n_h = static_cast<std::int64_t>(plan.rates.size());
  const auto n_b = static_cast<std::int64_t>(geometry.dirs.size());
  if (static_cast<std::int64_t>(fine_ranges.size()) != n_cells * k) throw std::invalid_argument("sweep_bank_maps: fine_ranges must hold n_cells x k ranges");
  if (static_cast<std::int64_t>(plan.first.size()) != n_b) throw std::invalid_argument("sweep_bank_maps: the plan's beams and the geometry's differ");
  if (n_h < 1) throw std::invalid_argument("sweep_bank_maps: the plan has no hypothesis");
  // Each hypothesis' filter, as pulse_compress makes it with its doppler_scale (the zero one's is exactly 1).
  std::vector<std::vector<std::complex<float>>> filters(static_cast<std::size_t>(n_h));
  std::vector<const float*> filter_ptr(static_cast<std::size_t>(n_h));
  std::vector<std::int64_t> n_fft(static_cast<std::size_t>(n_h));
  for (std::int64_t h = 0; h < n_h; ++h) {
    dsp::PulseCompressionOptions pc;
    pc.doppler_scale = plan.scales[static_cast<std::size_t>(h)];
    filters[static_cast<std::size_t>(h)] =
        dsp::compression_filter(base.transmit.waveform, s.sample_rate, s.center_frequency, s.n_samples, pc, n_fft[static_cast<std::size_t>(h)]);
    filter_ptr[static_cast<std::size_t>(h)] = reinterpret_cast<const float*>(filters[static_cast<std::size_t>(h)].data());
  }
  // The backprojection's inputs of the record (samples aside: the kernel reads each hypothesis' compressed ones), one group
  // per hypothesis, read later by its shift as the loop's `trial.samples.start_time += shifts[h]` (doppler.hpp).
  const std::vector<const PingRecord*> pings{&base};
  const std::vector<std::vector<float>> weights{geometry.weights};
  const recon::BpInputs bp = recon::make_inputs(pings, &weights);
  if (bp.groups.size() != 1 || bp.groups[0].sample_offset != 0) throw std::logic_error("sweep_bank_maps: one ping's inputs expected");
  std::vector<recon::BpGroup> groups(static_cast<std::size_t>(n_h), bp.groups[0]);
  for (std::int64_t h = 0; h < n_h; ++h) {
    double t = base.samples.start_time;
    if (h != plan.zero) t += plan.shifts[static_cast<std::size_t>(h)];
    groups[static_cast<std::size_t>(h)].start_time = t;
  }
  recon::BpOptions o;                       // range_bearing_image's
  o.interp = recon::Interp::Linear;
  o.device = device;

  SweepBankBatchedInput in;
  in.samples = s.data.data();
  in.n_ch = s.n_channels;
  in.n = s.n_samples;
  in.n_h = n_h;
  in.n_fft = n_fft.data();
  in.filters = filter_ptr.data();
  in.groups = groups.data();
  in.records = bp.records.data();
  in.bp = o;
  in.origin = geometry.origin;
  in.dirs = geometry.dirs.data();
  in.ranges = fine_ranges.data();
  in.n_b = n_b;
  in.n_cells = n_cells;
  in.k = k;
  in.first = plan.first.data();
  in.per_beam = plan.per_beam;
  in.cfar = plan.combine == DopplerCombine::Cfar;
  in.skip = std::clamp<std::int64_t>(skip, 0, n_cells);
  in.guard = guard;
  in.train = train;
  std::vector<double> offset;
  if (in.cfar) detector_background_window(n_cells - in.skip, guard, train, offset, in.sloped);
  in.offset = offset.data();
  std::int64_t max_fft = 1;
  for (std::int64_t v : n_fft) max_fft = std::max(max_fft, v);
  const SweepBankBytes bytes = sweep_bank_bytes(in.n_ch, in.n, max_fft, n_h, n_b, n_cells, k, in.skip, train, in.cfar, in.sloped);

  SweepBankMaps m;
  const auto n_fine = static_cast<std::size_t>(n_b * n_cells * k);
  m.fine.resize(n_fine);
  m.which.resize(n_fine);
  m.cells.resize(static_cast<std::size_t>(n_b * n_cells));
  if (in.cfar) m.level.resize(static_cast<std::size_t>(n_b * n_cells));
  SweepBankBatchedOutput out{m.fine.data(), m.which.data(), m.cells.data(), in.cfar ? m.level.data() : nullptr};
  ++prof.pings;
  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    // Half of what the device has free: the rest is cuFFT's work memory (its plans for the batch's transforms) and
    // the chain's other stages.
    const std::int64_t budget = doppler_cuda_free_bytes(device) / 2;
    in.batch = sweep_bank_batch(bytes, n_h, budget, batch);
    m.batch = in.batch;
    prof.prepare += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    sweep_bank_cuda(in, out, prof, device);
    return m;
#else
    throw std::runtime_error("sweep_bank_maps: built without the CUDA backend");
#endif
  }
  // On the host the formulation is for tests: a budget of 1 GiB unless the batch is given.
  in.batch = sweep_bank_batch(bytes, n_h, std::int64_t{1} << 30, batch);
  m.batch = in.batch;
  prof.prepare += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  dsp::HostExec ex;
  sweep_bank_batched(ex, in, out, prof);
  return m;
}

}  // namespace sonar::live
