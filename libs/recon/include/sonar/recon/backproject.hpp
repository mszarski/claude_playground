// Time-domain backprojection: the one kernel behind FLS, SAS and flank imaging.
//
// Input is a list of records (transmit position, receive position, time series)
// grouped by ping, and a list of output points. Output is the coherent sum per
// point. See DECISIONS.md D-01 (equation), D-05 (interpolation), D-06 (precision).
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"
#include "sonar/core/vec.hpp"

namespace sonar::recon {

enum class Interp : std::int32_t { Nearest = 0, Linear = 1, Sinc = 2 };

/// How the receive platform's position during reception is obtained (DECISIONS.md D-07).
enum class PositionLookup : std::int32_t {
  Analytic = 0,       ///< first-order pose model only
  SoftwareTable = 1,  ///< plus a tabulated residual, linear interpolation in float32
  Texture = 2,        ///< plus the residual from GPU texture memory with hardware filtering
                      ///< (on the host backend: the same 8-bit-weight arithmetic, emulated)
};

inline constexpr int kSincTaps = 8;
inline constexpr int kSincPhases = 128;
inline constexpr double kSincKaiserBeta = 6.0;

/// One ping: the reference geometry held in double precision.
struct BpGroup {
  Vec3d tx_pos;                 ///< transmitter (or virtual source) at transmit, world
  Vec3d tx_dir;                 ///< unit direction of travel, plane-wave transmit only
  std::int32_t tx_plane_wave = 0;
  // Receive body at the transmit instant. The reference point is the body point
  // `ref_lever` (the array centre); its position at time t after transmit is
  // rx_pos + rx_vel t + R(rx_rpy + rx_rates t) ref_lever, evaluated exactly.
  Vec3d rx_pos, rx_vel, rx_rpy, rx_rates;
  Vec3d ref_lever;
  double sound_speed = 1500.0;
  double start_time = 0.0;      ///< time of sample 0 after transmit
  double sample_rate = 0.0;
  double carrier = 0.0;         ///< Hz removed from the samples; 0 for analytic input
  std::int64_t first_record = 0;
  std::int64_t n_records = 0;
  std::int64_t sample_offset = 0;  ///< index of this group's first complex sample
  std::int64_t n_samples = 0;      ///< samples per record
  // Trajectory residual table: position minus the first-order model, world frame.
  std::int64_t traj_offset = 0;    ///< first entry of this ping in BpInputs::trajectory
  std::int32_t traj_count = 0;     ///< entries; 0 = no table
  double traj_t0 = 0.0;            ///< time of entry 0 after transmit, s
  double traj_dt = 1.0;            ///< spacing, s
  // Transmit beam, for the field-of-view test and directivity weighting.
  Vec3f tx_boresight, tx_az, tx_el;
  float tx_bw_az = 0.0F, tx_bw_el = 0.0F;  ///< NaN = omnidirectional
};

/// One receive element of one ping, relative to its group's reference point.
struct BpRecord {
  Vec3f d;        ///< offset from the group reference point, body frame, m
  Vec3f boresight, az, el;  ///< body frame
  float bw_az = 0.0F, bw_el = 0.0F;  ///< NaN = omnidirectional
  float weight = 1.0F;               ///< aperture weighting
};

struct BpOptions {
  Interp interp = Interp::Sinc;
  bool stop_and_hop = false;       ///< freeze the receiver at its transmit-time position
  float fov_factor = 0.0F;         ///< accept directions within fov_factor * FWHM / 2; 0 = no limit
  bool directivity_weight = false; ///< weight by the element patterns
  std::int32_t spreading = 0;      ///< multiply by (R_tx R_rx)^(spreading/2): 0 none, 1, 2
  PositionLookup position_lookup = PositionLookup::Analytic;
  /// Refraction at a flat sediment-water interface (ASASIN Sec. III-B, DECISIONS.md D-23).
  /// Points with z > interface_z are in the sediment; sound travels at each ping's sound
  /// speed above the interface and at sediment_speed below it, along the Snell path.
  bool refraction = false;
  double interface_z = 0.0;      ///< m, NED (z down)
  double sediment_speed = 1700.0;  ///< m/s
  /// Ray culling (ASASIN Sec. II-C, DECISIONS.md D-09). The output points must form a
  /// grid of grid_rows x grid_cols in row-major order. Blocks of cull_block x cull_block
  /// points whose corners all lie outside a ping's transmit or receive beam are skipped
  /// for that ping. Needs fov_factor > 0. cull_block = 0 disables culling.
  std::int32_t cull_block = 0;
  std::int64_t grid_rows = 0;
  std::int64_t grid_cols = 0;
  std::int32_t device = 0;  ///< CUDA device index (multi-GPU: one call per device, per image tile)
};

/// What culling did, for the processing log.
struct BpStats {
  std::int64_t ping_blocks_total = 0;  ///< pings x blocks
  std::int64_t ping_blocks_kept = 0;   ///< those the kernel actually visited
};

/// Everything the kernel needs, in flat arrays.
struct BpInputs {
  std::vector<BpGroup> groups;
  std::vector<BpRecord> records;
  std::vector<std::complex<float>> samples;  ///< all records, concatenated
  std::vector<Vec3f> trajectory;             ///< residual tables of all pings, concatenated
};

/// Build kernel inputs from conditioned ping records (complex baseband or analytic).
/// `weights`, when given, holds one aperture weight per channel of each record.
/// `trajectory_upsample` > 0 builds each ping's residual table from its INS samples,
/// upsampled by that factor with cubic interpolation (the paper upsamples before
/// loading the texture).
BpInputs make_inputs(const std::vector<const PingRecord*>& pings,
                     const std::vector<std::vector<float>>* weights = nullptr,
                     int trajectory_upsample = 0);

/// Coherent sum at each output point, added into `image` (size points.size()).
/// `weight_sum`, when not null, receives the sum of weights that contributed per point.
void backproject(Backend backend, const BpInputs& inputs, const std::vector<Vec3d>& points,
                 const BpOptions& options, std::complex<float>* image, float* weight_sum = nullptr,
                 BpStats* stats = nullptr);

/// The culling mask on its own: one byte per (ping, block), 1 = visit. Host backend.
std::vector<unsigned char> cull_mask(const BpInputs& inputs, const std::vector<Vec3d>& points,
                                     const BpOptions& options);

/// Test hook: the delay each record contributes at each point, as the kernel computes it.
/// Returns, per (point, record): sample position (double precision equivalent) and
/// carrier phase in cycles, modulo 1.
void debug_delays(const BpInputs& inputs, const std::vector<Vec3d>& points, const BpOptions& options,
                  std::vector<double>& sample_position, std::vector<double>& phase_cycles);

}  // namespace sonar::recon
