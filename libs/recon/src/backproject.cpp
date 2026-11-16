#include "sonar/recon/backproject.hpp"

#include <cmath>
#include <stdexcept>

#include "sonar/core/directivity.hpp"
#include "sonar/core/parallel.hpp"
#include "sonar/recon/backproject_kernel.hpp"
#include "sonar/recon/sinc_table.hpp"

namespace sonar::recon {

#if SONAR_HAVE_CUDA
void backproject_cuda(const BpInputs& inputs, const std::vector<Vec3d>& points,
                      const BpOptions& options, const std::vector<float>& sinc_table,
                      std::complex<float>* image, float* weight_sum, BpStats* stats);
#endif

namespace {

bool culling_enabled(const BpOptions& o, std::size_t n_points) {
  if (o.cull_block <= 0) return false;
  if (!(o.fov_factor > 0.0F)) {
    throw std::invalid_argument("backproject: culling needs a field-of-view limit (fov_factor > 0)");
  }
  if (o.grid_rows <= 0 || o.grid_cols <= 0 ||
      static_cast<std::size_t>(o.grid_rows * o.grid_cols) != n_points) {
    throw std::invalid_argument("backproject: culling needs grid_rows x grid_cols == number of points");
  }
  return true;
}

}  // namespace

float aperture_radius(const BpInputs& inputs) {
  float r = 0.0F;
  for (const BpRecord& e : inputs.records) {
    const float d = norm(e.d);
    if (d > r) r = d;
  }
  return r;
}

std::vector<unsigned char> cull_mask(const BpInputs& inputs, const std::vector<Vec3d>& points,
                                     const BpOptions& options) {
  if (!culling_enabled(options, points.size())) return {};
  const std::int64_t br = block_count(options.grid_rows, options.cull_block);
  const std::int64_t bc = block_count(options.grid_cols, options.cull_block);
  const auto n_groups = static_cast<std::int64_t>(inputs.groups.size());
  std::vector<unsigned char> mask(static_cast<std::size_t>(n_groups * br * bc));
  const CullKernel k{inputs.groups.data(), inputs.records.data(), points.data(), mask.data(),
                     options.grid_rows, options.grid_cols, options.cull_block, br, bc,
                     options.fov_factor, aperture_radius(inputs)};
  parallel_for_host(n_groups * br * bc, k);
  return mask;
}

BpInputs make_inputs(const std::vector<const PingRecord*>& pings,
                     const std::vector<std::vector<float>>* weights) {
  BpInputs in;
  std::size_t total_samples = 0, total_records = 0;
  for (const PingRecord* p : pings) {
    if (p == nullptr) throw std::invalid_argument("backproject: null ping");
    if (!p->samples.is_complex()) {
      throw std::invalid_argument(
          "backproject: samples must be complex (baseband or analytic); run to_baseband first");
    }
    if (p->samples.data.size() != p->samples.expected_size()) {
      throw std::invalid_argument("backproject: ping samples are not allocated");
    }
    if (static_cast<std::int64_t>(p->geometry.rx.size()) != p->samples.n_channels) {
      throw std::invalid_argument("backproject: one receive element per channel is required");
    }
    total_samples += static_cast<std::size_t>(p->samples.n_channels * p->samples.n_samples);
    total_records += static_cast<std::size_t>(p->samples.n_channels);
  }
  if (weights != nullptr && weights->size() != pings.size()) {
    throw std::invalid_argument("backproject: weights must have one entry per ping");
  }
  in.groups.reserve(pings.size());
  in.records.reserve(total_records);
  in.samples.resize(total_samples);

  std::int64_t sample_cursor = 0;
  for (std::size_t ip = 0; ip < pings.size(); ++ip) {
    const PingRecord& p = *pings[ip];
    const auto n = static_cast<std::size_t>(p.samples.n_channels);
    const Pose& rx_pose = p.geometry.rx_pose;

    // Element positions (with calibrated offsets) and their mean as the reference point.
    std::vector<Vec3d> lever(n);
    Vec3d mean_lever{};
    for (std::size_t i = 0; i < n; ++i) {
      lever[i] = p.geometry.rx[i].lever_arm;
      if (i < p.calibration.position_offset.size()) lever[i] += p.calibration.position_offset[i];
      mean_lever += lever[i];
    }
    mean_lever = mean_lever / static_cast<double>(n);

    BpGroup g;
    g.tx_pos = world_position(p.geometry.tx_pose, p.geometry.tx.lever_arm);
    g.tx_plane_wave = p.transmit.kind == TransmitKind::PlaneWave ? 1 : 0;
    g.tx_dir = world_direction(p.geometry.tx_pose, p.transmit.direction);
    g.rx_pos = rx_pose.position;
    g.rx_vel = rx_pose.velocity;
    g.rx_rpy = rx_pose.rpy;
    g.rx_rates = rx_pose.rates;
    g.ref_lever = mean_lever;
    g.sound_speed = p.environment.sound_speed;
    g.start_time = p.samples.start_time;
    g.sample_rate = p.samples.sample_rate;
    g.carrier = p.samples.kind == SampleKind::ComplexBaseband ? p.samples.center_frequency : 0.0;
    g.first_record = static_cast<std::int64_t>(in.records.size());
    g.n_records = static_cast<std::int64_t>(n);
    g.sample_offset = sample_cursor;
    g.n_samples = p.samples.n_samples;
    const ElementAxes<double> tx_axes = element_axes(p.geometry.tx.boresight);
    g.tx_boresight = Vec3f(world_direction(p.geometry.tx_pose, tx_axes.boresight));
    g.tx_az = Vec3f(world_direction(p.geometry.tx_pose, tx_axes.az));
    g.tx_el = Vec3f(world_direction(p.geometry.tx_pose, tx_axes.el));
    g.tx_bw_az = static_cast<float>(p.geometry.tx.beamwidth_az);
    g.tx_bw_el = static_cast<float>(p.geometry.tx.beamwidth_el);
    in.groups.push_back(g);

    if (weights != nullptr && (*weights)[ip].size() != n) {
      throw std::invalid_argument("backproject: weights must have one value per channel");
    }
    for (std::size_t i = 0; i < n; ++i) {
      BpRecord e;
      e.d = Vec3f(lever[i] - mean_lever);
      const ElementAxes<double> axes = element_axes(p.geometry.rx[i].boresight);
      e.boresight = Vec3f(axes.boresight);
      e.az = Vec3f(axes.az);
      e.el = Vec3f(axes.el);
      e.bw_az = static_cast<float>(p.geometry.rx[i].beamwidth_az);
      e.bw_el = static_cast<float>(p.geometry.rx[i].beamwidth_el);
      e.weight = weights != nullptr ? (*weights)[ip][i] : 1.0F;
      in.records.push_back(e);
    }

    const auto count = static_cast<std::size_t>(p.samples.n_channels * p.samples.n_samples);
    const auto* src = reinterpret_cast<const std::complex<float>*>(p.samples.data.data());
    std::copy(src, src + count, in.samples.begin() + sample_cursor);
    sample_cursor += static_cast<std::int64_t>(count);
  }
  return in;
}

void backproject(Backend backend, const BpInputs& inputs, const std::vector<Vec3d>& points,
                 const BpOptions& options, std::complex<float>* image, float* weight_sum,
                 BpStats* stats) {
  if (points.empty() || inputs.groups.empty()) return;
  const std::vector<float> table = make_sinc_table();
  const bool cull = culling_enabled(options, points.size());
  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    backproject_cuda(inputs, points, options, table, image, weight_sum, stats);
    return;
#else
    throw std::runtime_error("backproject: built without the CUDA backend");
#endif
  }
  BpKernel k{inputs.groups.data(),
             inputs.records.data(),
             reinterpret_cast<const float*>(inputs.samples.data()),
             points.data(),
             table.data(),
             reinterpret_cast<float*>(image),
             weight_sum,
             static_cast<std::int64_t>(inputs.groups.size()),
             options};
  std::vector<unsigned char> mask;
  if (cull) {
    mask = cull_mask(inputs, points, options);
    k.mask = mask.data();
    k.blocks_c = block_count(options.grid_cols, options.cull_block);
    k.n_blocks = block_count(options.grid_rows, options.cull_block) * k.blocks_c;
  }
  if (stats != nullptr) {
    stats->ping_blocks_total = static_cast<std::int64_t>(mask.size());
    stats->ping_blocks_kept = 0;
    for (unsigned char m : mask) stats->ping_blocks_kept += m;
  }
  parallel_for_host(static_cast<std::int64_t>(points.size()), k);
}

void debug_delays(const BpInputs& inputs, const std::vector<Vec3d>& points, const BpOptions& options,
                  std::vector<double>& sample_position, std::vector<double>& phase_cycles) {
  const std::size_t n_rec = inputs.records.size();
  sample_position.assign(points.size() * n_rec, 0.0);
  phase_cycles.assign(points.size() * n_rec, 0.0);
  for (std::size_t ip = 0; ip < points.size(); ++ip) {
    for (const BpGroup& g : inputs.groups) {
      const RefDelay ref = reference_delay(g, points[ip], options.stop_and_hop);
      for (std::int64_t ir = 0; ir < g.n_records; ++ir) {
        const auto rec = static_cast<std::size_t>(g.first_record + ir);
        Vec3f to_point;
        float r_elem;
        const float dtau = element_delay(g, inputs.records[rec], ref, to_point, r_elem);
        // Reassemble exactly what the kernel uses: integer + float fraction.
        const float pos = ref.frac0 + dtau * static_cast<float>(g.sample_rate);
        sample_position[ip * n_rec + rec] = static_cast<double>(ref.n0) + static_cast<double>(pos);
        float cyc = ref.phase0 + static_cast<float>(g.carrier) * dtau;
        cyc -= std::floor(cyc);
        phase_cycles[ip * n_rec + rec] = static_cast<double>(cyc);
      }
    }
  }
}

}  // namespace sonar::recon
