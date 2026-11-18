// Time-domain backprojection: the one kernel behind FLS, SAS and flank imaging.
//
// Input is a list of records (transmit position, receive position, time series)
// grouped by ping, and a list of output points. Output is the coherent sum per
// point. See DECISIONS.md D-01 (equation), D-05 (interpolation), D-06 (precision).
//
// Units and conventions: SI; world frame north-east-down (z positive down), element
// vectors in the body frame; beamwidths are full widths at half maximum in radians;
// delays are two-way through a transmitter and a receiver that may differ (bistatic).
// The output keeps the units of the input samples: no level convention enters here.
// The sound speed of each ping and the sediment speed of a refracting interface have no
// default; `make_inputs` and `backproject` throw when one is not stated.
#pragma once

#include <complex>
#include <cstdint>
#include <mutex>
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
  // Derived on the host so the kernel can avoid float64 work (see group_sum):
  Mat3d rot0;                      ///< body-to-world rotation at transmit
  std::int32_t static_attitude = 0;  ///< 1 when the attitude rates are zero: rot0 holds throughout
  float aperture = 0.0F;           ///< largest element offset from the reference point, m
  float speed = 0.0F;              ///< |rx_vel|, m/s
  float turn_rate = 0.0F;          ///< |rx_rates|, rad/s
  double sound_speed = kNaN;    ///< m/s; from the record, which must state it
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
  /// How far the platform may be from its first-order pose during this ping, m: what the
  /// cheap field-of-view pre-test of the kernel adds to the reach of the array before it
  /// rejects a (point, ping) pair. Set by make_inputs (see BpInputs::kTrajectoryAllowance).
  float traj_allowance = 0.02F;
  float traj_residual_max = 0.0F;  ///< largest norm in this ping's residual table, m; 0 without a table
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
  double sediment_speed = kNaN;  ///< m/s; no default: required when `refraction` is set
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

/// The allowance for the trajectory residual in the kernel's field-of-view pre-test, m.
/// 2 cm is what the kernel has always added: enough for the millimetres to centimetres a
/// short ping strays from its first-order pose, not for a long ping or a turning platform
/// with a navigation table (docs/HARDCODED_AUDIT.md R-1), where pairs the exact test would
/// keep are then rejected at the beam edges.
inline constexpr float kTrajectoryAllowance = 0.02F;

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
/// `trajectory_allowance` (m) is the allowance for the trajectory residual in the kernel's
/// field-of-view pre-test: a value >= 0 is used for every ping (the default is the 2 cm
/// the kernel has always used); a negative value takes it from each ping's own residual
/// table, max(kTrajectoryAllowance, largest residual in the table), which never rejects a
/// pair the fixed 2 cm keeps and stops rejecting pairs of a ping that strays further.
BpInputs make_inputs(const std::vector<const PingRecord*>& pings,
                     const std::vector<std::vector<float>>* weights = nullptr,
                     int trajectory_upsample = 0, float trajectory_allowance = kTrajectoryAllowance);

/// Coherent sum at each output point, added into `image` (size points.size()).
/// `weight_sum`, when not null, receives the sum of weights that contributed per point.
void backproject(Backend backend, const BpInputs& inputs, const std::vector<Vec3d>& points,
                 const BpOptions& options, std::complex<float>* image, float* weight_sum = nullptr,
                 BpStats* stats = nullptr);

/// Per-collection kernel inputs kept on a CUDA device between calls (docs/DECISIONS.md D-40).
struct BpResident;

/// A collection prepared once and imaged many times: tiles, depth layers, autofocus
/// iterations, several devices. On the CUDA backend the samples, ping and element tables
/// and the interpolation table are uploaded to a device on first use there and stay until
/// the session is destroyed; each call then moves only its points and its image. On the
/// host backend a session just saves rebuilding the inputs.
class BpSession {
 public:
  explicit BpSession(BpInputs inputs);
  ~BpSession();
  BpSession(const BpSession&) = delete;
  BpSession& operator=(const BpSession&) = delete;

  const BpInputs& inputs() const { return inputs_; }
  /// Same contract as the free function `backproject`. Safe to call from several threads
  /// with different `options.device`.
  void backproject(Backend backend, const std::vector<Vec3d>& points, const BpOptions& options,
                   std::complex<float>* image, float* weight_sum = nullptr, BpStats* stats = nullptr);
  /// Bytes held on all devices so far.
  std::size_t resident_bytes() const;

 private:
  BpInputs inputs_;
  std::vector<float> sinc_table_;
  mutable std::mutex mutex_;
  std::vector<BpResident*> resident_;
};

/// The culling mask on its own: one byte per (ping, block), 1 = visit. Host backend.
std::vector<unsigned char> cull_mask(const BpInputs& inputs, const std::vector<Vec3d>& points,
                                     const BpOptions& options);

/// Test hook: the delay each record contributes at each point, as the kernel computes it.
/// Returns, per (point, record): sample position (double precision equivalent) and
/// carrier phase in cycles, modulo 1.
void debug_delays(const BpInputs& inputs, const std::vector<Vec3d>& points, const BpOptions& options,
                  std::vector<double>& sample_position, std::vector<double>& phase_cycles);

}  // namespace sonar::recon
