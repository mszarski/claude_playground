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
};

/// Everything the kernel needs, in flat arrays.
struct BpInputs {
  std::vector<BpGroup> groups;
  std::vector<BpRecord> records;
  std::vector<std::complex<float>> samples;  ///< all records, concatenated
};

/// Build kernel inputs from conditioned ping records (complex baseband or analytic).
/// `weights`, when given, holds one aperture weight per channel of each record.
BpInputs make_inputs(const std::vector<const PingRecord*>& pings,
                     const std::vector<std::vector<float>>* weights = nullptr);

/// Coherent sum at each output point, added into `image` (size points.size()).
/// `weight_sum`, when not null, receives the sum of weights that contributed per point.
void backproject(Backend backend, const BpInputs& inputs, const std::vector<Vec3d>& points,
                 const BpOptions& options, std::complex<float>* image, float* weight_sum = nullptr);

/// Test hook: the delay each record contributes at each point, as the kernel computes it.
/// Returns, per (point, record): sample position (double precision equivalent) and
/// carrier phase in cycles, modulo 1.
void debug_delays(const BpInputs& inputs, const std::vector<Vec3d>& points, const BpOptions& options,
                  std::vector<double>& sample_position, std::vector<double>& phase_cycles);

}  // namespace sonar::recon
