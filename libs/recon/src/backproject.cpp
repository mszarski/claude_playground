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
                      std::complex<float>* image, float* weight_sum, BpStats* stats,
                      const BpResident* resident);
BpResident* bp_resident_create(const BpInputs& inputs, const std::vector<float>& sinc_table, int device);
void bp_resident_destroy(BpResident* resident);
int bp_resident_device(const BpResident* resident);
std::size_t bp_resident_bytes(const BpResident* resident);
#endif

namespace {

void check_options(const BpOptions& o) {
  if (o.refraction && (!(o.sediment_speed > 0.0) || !std::isfinite(o.sediment_speed) || !std::isfinite(o.interface_z))) {
    throw std::invalid_argument(
        "backproject: refraction needs the sediment's sound speed and the interface depth; there is no default");
  }
}

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

namespace {

/// Catmull-Rom interpolation of irregularly indexed samples at fractional index u.
Vec3d cubic(const std::vector<Vec3d>& v, double u) {
  const auto n = static_cast<std::int64_t>(v.size());
  auto i = static_cast<std::int64_t>(std::floor(u));
  if (i > n - 2) i = n - 2;
  if (i < 0) i = 0;
  const double f = u - static_cast<double>(i);
  const Vec3d& p1 = v[static_cast<std::size_t>(i)];
  const Vec3d& p2 = v[static_cast<std::size_t>(i + 1)];
  // One-sided ends: extrapolate the missing neighbour linearly.
  const Vec3d p0 = i > 0 ? v[static_cast<std::size_t>(i - 1)] : p1 * 2.0 - p2;
  const Vec3d p3 = i + 2 < n ? v[static_cast<std::size_t>(i + 2)] : p2 * 2.0 - p1;
  const double f2 = f * f, f3 = f2 * f;
  return (p0 * (-f3 + 2.0 * f2 - f) + p1 * (3.0 * f3 - 5.0 * f2 + 2.0) +
          p2 * (-3.0 * f3 + 4.0 * f2 + f) + p3 * (f3 - f2)) * 0.5;
}

/// Residual table of one ping from its INS samples, which must be uniformly spaced.
void append_trajectory(const PingRecord& p, int upsample, BpGroup& g, std::vector<Vec3f>& table) {
  const auto& ins = p.navigation.ins;
  if (upsample <= 0 || ins.size() < 2) return;
  const double dt = (ins.back().t - ins.front().t) / static_cast<double>(ins.size() - 1);
  for (std::size_t k = 1; k < ins.size(); ++k) {
    if (std::abs((ins[k].t - ins[k - 1].t) - dt) > 1e-6 * dt) {
      throw std::invalid_argument("backproject: INS samples must be uniformly spaced to build the "
                                  "trajectory table; resample them first");
    }
  }
  std::vector<Vec3d> residual(ins.size());
  for (std::size_t k = 0; k < ins.size(); ++k) {
    residual[k] = ins[k].position - (p.geometry.rx_pose.position + p.geometry.rx_pose.velocity * ins[k].t);
  }
  const std::size_t n_out = (ins.size() - 1) * static_cast<std::size_t>(upsample) + 1;
  g.traj_offset = static_cast<std::int64_t>(table.size());
  g.traj_count = static_cast<std::int32_t>(n_out);
  g.traj_t0 = ins.front().t;
  g.traj_dt = dt / upsample;
  for (std::size_t k = 0; k < n_out; ++k) {
    const Vec3f r(cubic(residual, static_cast<double>(k) / upsample));
    const float size = norm(r);
    if (size > g.traj_residual_max) g.traj_residual_max = size;
    table.push_back(r);
  }
}

}  // namespace

BpInputs make_inputs(const std::vector<const PingRecord*>& pings,
                     const std::vector<std::vector<float>>* weights, int trajectory_upsample,
                     float trajectory_allowance) {
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
    if (!(p->environment.sound_speed > 0.0) || !std::isfinite(p->environment.sound_speed)) {
      throw std::invalid_argument("backproject: a ping has no sound speed (environment.sound_speed)");
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
    g.rot0 = rotation_ned(rx_pose.rpy.x, rx_pose.rpy.y, rx_pose.rpy.z);
    g.static_attitude = (rx_pose.rates.x == 0.0 && rx_pose.rates.y == 0.0 && rx_pose.rates.z == 0.0) ? 1 : 0;
    g.speed = static_cast<float>(norm(rx_pose.velocity));
    g.turn_rate = static_cast<float>(norm(rx_pose.rates));
    for (std::size_t i = 0; i < n; ++i) {
      const auto r = static_cast<float>(norm(lever[i] - mean_lever));
      if (r > g.aperture) g.aperture = r;
    }
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
    append_trajectory(p, trajectory_upsample, g, in.trajectory);
    if (!std::isfinite(trajectory_allowance)) {
      throw std::invalid_argument("backproject: trajectory_allowance must be a number (negative: from the table)");
    }
    g.traj_allowance = trajectory_allowance >= 0.0F
                           ? trajectory_allowance
                           : (g.traj_residual_max > kTrajectoryAllowance ? g.traj_residual_max : kTrajectoryAllowance);
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

namespace {

void backproject_host(const BpInputs& inputs, const std::vector<float>& table,
                      const std::vector<Vec3d>& points, const BpOptions& options,
                      std::complex<float>* image, float* weight_sum, BpStats* stats) {
  const bool cull = culling_enabled(options, points.size());
  BpKernel k{inputs.groups.data(),
             inputs.records.data(),
             reinterpret_cast<const float*>(inputs.samples.data()),
             points.data(),
             table.data(),
             reinterpret_cast<float*>(image),
             weight_sum,
             static_cast<std::int64_t>(inputs.groups.size()),
             options,
             nullptr,
             TrajectoryAccess{},
             0,
             0};
  k.traj.table = inputs.trajectory.data();
  k.traj.mode = inputs.trajectory.empty() ? PositionLookup::Analytic : options.position_lookup;
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

}  // namespace

void backproject(Backend backend, const BpInputs& inputs, const std::vector<Vec3d>& points,
                 const BpOptions& options, std::complex<float>* image, float* weight_sum,
                 BpStats* stats) {
  if (points.empty() || inputs.groups.empty()) return;
  check_options(options);
  const std::vector<float> table = make_sinc_table();
  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    backproject_cuda(inputs, points, options, table, image, weight_sum, stats, nullptr);
    return;
#else
    throw std::runtime_error("backproject: built without the CUDA backend");
#endif
  }
  backproject_host(inputs, table, points, options, image, weight_sum, stats);
}

BpSession::BpSession(BpInputs inputs) : inputs_(std::move(inputs)), sinc_table_(make_sinc_table()) {}

BpSession::~BpSession() {
#if SONAR_HAVE_CUDA
  for (BpResident* r : resident_) bp_resident_destroy(r);
#endif
}

std::size_t BpSession::resident_bytes() const {
  std::size_t n = 0;
#if SONAR_HAVE_CUDA
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const BpResident* r : resident_) n += bp_resident_bytes(r);
#endif
  return n;
}

void BpSession::backproject(Backend backend, const std::vector<Vec3d>& points, const BpOptions& options,
                            std::complex<float>* image, float* weight_sum, BpStats* stats) {
  if (points.empty() || inputs_.groups.empty()) return;
  check_options(options);
  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    const BpResident* resident = nullptr;
    {
      // Upload once per device; other devices wait only while one is being filled.
      const std::lock_guard<std::mutex> lock(mutex_);
      for (const BpResident* r : resident_) {
        if (bp_resident_device(r) == options.device) resident = r;
      }
      if (resident == nullptr) {
        resident_.push_back(bp_resident_create(inputs_, sinc_table_, options.device));
        resident = resident_.back();
      }
    }
    backproject_cuda(inputs_, points, options, sinc_table_, image, weight_sum, stats, resident);
    return;
#else
    throw std::runtime_error("backproject: built without the CUDA backend");
#endif
  }
  backproject_host(inputs_, sinc_table_, points, options, image, weight_sum, stats);
}

void debug_delays(const BpInputs& inputs, const std::vector<Vec3d>& points, const BpOptions& options,
                  std::vector<double>& sample_position, std::vector<double>& phase_cycles) {
  const std::size_t n_rec = inputs.records.size();
  sample_position.assign(points.size() * n_rec, 0.0);
  phase_cycles.assign(points.size() * n_rec, 0.0);
  for (std::size_t ip = 0; ip < points.size(); ++ip) {
    for (const BpGroup& g : inputs.groups) {
      const RefDelay ref = reference_delay(g, points[ip], options.stop_and_hop);  // analytic path
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
