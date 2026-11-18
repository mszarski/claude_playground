// The ping record: the one in-memory format every source produces and every
// stage consumes. The on-disk form is described in docs/RECORD_FORMAT.md and
// implemented in libs/io. Schema changes need the owner's sign-off.
//
// Units are SI: metres, seconds, radians, hertz, pascals. Frames are NED.
// Times inside a record are seconds relative to the transmit instant unless a
// field name ends in _ns (integer nanoseconds since the Unix epoch, UTC).
//
// Conventions a consumer may rely on (docs/HARDCODED_AUDIT.md section 4):
//  - World frame: north, east, down. Body frame: x ahead, y to starboard, z down;
//    attitude is roll, pitch, yaw.
//  - Levels: `Waveform::source_level_db` and `Waveform::amplitude` are PEAK values (the
//    amplitude of the sinusoid, not its rms). Whether a datasheet figure entered in a
//    sonar config is peak or rms is settled where the config is read (sonar.py), not here.
//  - Signal sign: exp(+i w t); a baseband record has had `center_frequency` removed.
//  - Sample times are seconds after transmit; ranges derived from them are two-way
//    (range = c t / 2) unless a stage says otherwise.
//  - Facts that have no physical default are NaN until a source sets them: the sound
//    speed and the element boresights. libs/io validation and the entry points of the
//    kernels reject a record that still holds them (audit K-2, K-3).
#pragma once

#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "sonar/core/vec.hpp"

namespace sonar {

inline constexpr const char* kSchemaName = "gpusonar.record";
inline constexpr int kSchemaMajor = 1;
inline constexpr int kSchemaMinor = 0;
inline constexpr int kSchemaPatch = 0;

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

enum class SampleKind : std::int32_t {
  RealPassband = 0,     ///< real samples at the carrier
  ComplexBaseband = 1,  ///< complex, carrier `center_frequency` removed
  ComplexAnalytic = 2,  ///< complex analytic signal, carrier still present
};

enum class Mode : std::int32_t { Unknown = 0, Fls = 1, Sas = 2, Flank = 3 };

enum class WaveformType : std::int32_t {
  Cw = 0,         ///< gated tone at f_start
  Lfm = 1,        ///< linear sweep f_start -> f_stop
  Gaussian = 2,   ///< Gaussian-enveloped tone centred on (f_start+f_stop)/2; f_stop - f_start is the -6 dB bandwidth
  Arbitrary = 3,  ///< replica samples supplied
};

enum class TransmitKind : std::int32_t {
  Point = 0,      ///< diverging wave from a (possibly virtual) point source
  PlaneWave = 1,  ///< plane wave travelling along `direction`
};

/// How samples are stored on disk. In memory they are always float32.
enum class StorageType : std::int32_t { Float32 = 0, Int16 = 1, Int32 = 2 };

/// Channels x samples block. Complex data is interleaved (re, im).
struct Samples {
  SampleKind kind = SampleKind::RealPassband;
  std::int64_t n_channels = 0;
  std::int64_t n_samples = 0;
  double sample_rate = 0.0;       ///< Hz
  double center_frequency = 0.0;  ///< Hz; the removed carrier for baseband data
  double start_time = 0.0;        ///< time of sample 0 relative to transmit, s
  StorageType storage = StorageType::Float32;
  double storage_scale = 1.0;  ///< physical value = stored integer * storage_scale
  std::vector<float> data;     ///< size n_channels * n_samples * (complex ? 2 : 1)

  bool is_complex() const { return kind != SampleKind::RealPassband; }
  std::size_t stride() const { return is_complex() ? 2 : 1; }
  std::size_t expected_size() const {
    return static_cast<std::size_t>(n_channels) * static_cast<std::size_t>(n_samples) * stride();
  }
  float* channel(std::int64_t c) { return data.data() + c * n_samples * stride(); }
  const float* channel(std::int64_t c) const { return data.data() + c * n_samples * stride(); }
  /// Complex view of one channel. Only valid when is_complex().
  std::complex<float>* channel_complex(std::int64_t c) {
    return reinterpret_cast<std::complex<float>*>(channel(c));
  }
  const std::complex<float>* channel_complex(std::int64_t c) const {
    return reinterpret_cast<const std::complex<float>*>(channel(c));
  }
  void resize(std::int64_t channels, std::int64_t samples, SampleKind k) {
    kind = k;
    n_channels = channels;
    n_samples = samples;
    data.assign(expected_size(), 0.0F);
  }
};

struct Waveform {
  WaveformType type = WaveformType::Cw;
  double f_start = 0.0;   ///< Hz
  double f_stop = 0.0;    ///< Hz (equal to f_start for CW)
  double duration = 0.0;  ///< s
  double window_alpha = 0.0;  ///< Tukey taper fraction on the envelope, 0 = rectangular
  double source_level_db = kNaN;  ///< dB re 1 uPa at 1 m, NaN if unknown
  double amplitude = 1.0;  ///< peak amplitude at 1 m in the record's sample units
  double replica_sample_rate = 0.0;  ///< Hz, for Arbitrary
  std::vector<float> replica;        ///< real replica samples, for Arbitrary

  double center() const { return 0.5 * (f_start + f_stop); }
  double bandwidth() const { return f_stop > f_start ? f_stop - f_start : f_start - f_stop; }
};

/// Rigid-body state at the transmit instant.
struct Pose {
  Vec3d position;  ///< NED, m
  Vec3d rpy;       ///< roll, pitch, yaw, rad (ASASIN Eq. 2 conventions)
  Vec3d velocity;  ///< NED, m/s
  Vec3d rates;     ///< d(roll, pitch, yaw)/dt, rad/s
};

/// One transducer element on a body.
struct Element {
  Vec3d lever_arm;                 ///< body frame, m
  /// Body-frame unit vector of the main response axis. No default (NaN): a record built
  /// without one fails validation instead of silently looking to starboard.
  Vec3d boresight{kNaN, kNaN, kNaN};
  double beamwidth_az = kNaN;      ///< FWHM, rad; NaN = omnidirectional
  double beamwidth_el = kNaN;      ///< FWHM, rad; NaN = omnidirectional
};

struct Transmit {
  Waveform waveform;
  std::int64_t time_ns = 0;  ///< transmit instant, ns since Unix epoch (UTC)
  TransmitKind kind = TransmitKind::Point;
  Vec3d direction{kNaN, kNaN, kNaN};  ///< body-frame unit vector, PlaneWave only; no default
};

struct Geometry {
  Pose tx_pose;  ///< body carrying the transmitter
  Pose rx_pose;  ///< body carrying the receive array (same as tx_pose when monostatic)
  Element tx;    ///< transmitter (or virtual source / plane-wave phase reference)
  std::vector<Element> rx;  ///< one per channel
  Vec3d frame_origin_lla{kNaN, kNaN, kNaN};  ///< lat (deg), lon (deg), alt (m) of the NED origin
};

/// Navigation samples that fall in this ping's interval. Times are relative to transmit.
struct InsSample {
  double t;
  Vec3d position;  ///< NED, m
  Vec3d rpy;
  Vec3d velocity;  ///< NED, m/s
  Vec3d rates;
};
struct DvlSample {
  double t;
  Vec3d velocity_body;  ///< m/s
  double altitude;      ///< m above the seafloor, NaN if no bottom lock
};
struct GpsSample {
  double t;
  double latitude, longitude, altitude;  ///< deg, deg, m
  double horizontal_accuracy;            ///< m, NaN if unknown
};
struct DepthSample {
  double t;
  double depth;  ///< m, positive down
};
struct Navigation {
  std::vector<InsSample> ins;
  std::vector<DvlSample> dvl;
  std::vector<GpsSample> gps;
  std::vector<DepthSample> depth;
};

/// Per-channel corrections. Conditioning divides by gain*exp(j*phase), advances
/// by delay, and adds position_offset to the nominal lever arm.
struct Calibration {
  std::vector<double> gain;   ///< linear amplitude gain of the channel
  std::vector<double> phase;  ///< rad at center_frequency
  std::vector<double> delay;  ///< s
  std::vector<Vec3d> position_offset;  ///< body frame, m
};

struct Environment {
  double sound_speed = kNaN;    ///< m/s at the sonar; no default: the source must state it
  double temperature = kNaN;    ///< deg C
  double salinity = kNaN;       ///< PSU
  double water_depth = kNaN;    ///< m
  std::vector<double> profile_depth;  ///< m, increasing
  std::vector<double> profile_speed;  ///< m/s
};

/// One capture board with its own sample clock (for multi-board arrays).
struct Board {
  double clock_offset = 0.0;     ///< s, this board's time minus the common timeline, at transmit
  double clock_drift_ppm = 0.0;  ///< sample-clock frequency error
  double pps_sample = kNaN;      ///< sample index of the 1PPS edge in this ping, NaN if none
  std::int32_t reference_channel = -1;  ///< channel carrying the shared reference signal, -1 if none
  std::int32_t synchronised = 0;        ///< 1 once the sync stage has resampled this board
};

struct Metadata {
  std::string array_id;
  Mode mode = Mode::Unknown;
  std::int64_t sequence = 0;       ///< ping counter from the source
  std::int32_t sub_sequence = 0;   ///< index within a multi-transmit scan (mux phase, compounding angle)
  std::vector<std::int32_t> channel_element;  ///< index of each channel's element in the full array
  std::vector<std::int32_t> channel_board;    ///< index into `boards` for each channel
  std::vector<Board> boards;
  double hardware_gain_db = 0.0;   ///< fixed analogue gain
  std::vector<double> tvg_time;    ///< hardware gain ramp knots: time since transmit, s
  std::vector<double> tvg_gain_db; ///< hardware gain ramp knots: gain, dB (added to hardware_gain_db)
  std::int32_t adc_bits = 0;       ///< 0 if not applicable
  double adc_full_scale = kNaN;    ///< sample value at ADC full scale, NaN if unknown
  std::string extra_json = "{}";   ///< source-specific settings, JSON object
};

struct PingRecord {
  Samples samples;
  Transmit transmit;
  Geometry geometry;
  Navigation navigation;
  Calibration calibration;
  Environment environment;
  Metadata metadata;
};

// --- Geometry helpers -------------------------------------------------------

/// World position of a body point `lever_arm` at time `t` after transmit, using the
/// pose's first-order model: position + velocity*t, attitude + rates*t.
Vec3d world_position(const Pose& pose, const Vec3d& lever_arm, double t = 0.0);
/// World direction of a body-frame unit vector at time `t` after transmit.
Vec3d world_direction(const Pose& pose, const Vec3d& body_dir, double t = 0.0);
/// World velocity of a body point at transmit (translation plus rotation).
Vec3d world_velocity(const Pose& pose, const Vec3d& lever_arm);

/// Fill calibration and channel maps with neutral defaults sized to the samples.
void set_default_channel_tables(PingRecord& record);

}  // namespace sonar
