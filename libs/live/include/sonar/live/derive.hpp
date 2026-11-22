// The one place the chain's sizes come from when the configuration does not give them.
//
// A setting left out of the configuration (NaN in ChainOptions, 0 in PassiveOptions) is resolved
// here from the first record the chain sees: its waveform, its array and its sound speed. Each
// derived setting is
//
//     the larger of   a standing value (kStanding below: the numbers the chain has always used,
//                     tuned on kilometre-range arrays with a band of 2 kHz or more)
//               and   a floor that follows from the sonar (range resolution, beamwidth, the time
//                     sound takes to cross the array)
//
// so that an array the standing values suit gets exactly them, and a narrower band, a shorter or
// lower-frequency array, or a longer array gets settings that grow with it. The smaller of the two
// is used for the one setting that is an upper limit (the row tolerance). A value given in the
// configuration always wins. What was resolved, and why, is `ChainDefaults::json()`: the programs
// print it once, and tests read it.
//
// Not derived, and why: the false-alarm rate, the detection threshold, the tracker's confirmation
// counts, speeds and process noise are choices about targets and false alarms, not properties of the
// sonar; the multipath excess range belongs to the site (water depth and range). They stay
// configuration keys with their defaults in ChainOptions.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::live {

struct ChainOptions;
struct PassiveOptions;

/// "Not given: derive it" for a setting that is a number.
constexpr double kDerive = std::numeric_limits<double>::quiet_NaN();
inline bool is_given(double v) { return v == v; }

/// The standing values (see the top of this file). One table: nothing else in the chain, the
/// bindings or the Python package types these numbers.
struct Standing {
  double range_cell_m = 2.0;                             ///< m
  double cfar_guard_m = 100.0, cfar_train_m = 300.0;     ///< m
  double tracker_range_sigma = 15.0;                     ///< m
  double tracker_bearing_sigma = 0.017453292519943295;   ///< rad (1 deg)
  double multipath_bearing = 0.05235987755982988;        ///< rad (3 deg)
  double classify_max_fit_rms = 60.0;                    ///< m
  double row_tolerance_m = 0.02;                         ///< m (an upper limit)
  std::int64_t passive_n_fft = 1024;                     ///< samples
  double stream_buffer_bytes = 8.0 * 1024.0 * 1024.0;    ///< bytes
};
constexpr Standing kStanding{};

// The floors, as multiples of what the sonar gives.
constexpr double kCellsPerResolution = 1.0;      ///< range cell >= this many range resolutions c / 2B
constexpr double kGuardCells = 4.0;              ///< CFAR guard >= this many range resolutions
constexpr double kTrainPerGuard = 3.0;           ///< CFAR training >= this many guard floors
/// Tracker bearing sigma >= beamwidth / this. A centroid on a beam of width w at a signal-to-noise ratio s
/// is good to about w / sqrt(2 s) (s as a power ratio): a sixteenth of a beam at 21 dB. Every array shipped
/// today (beams of 1 to 14.5 degrees) stays under the standing 1 degree with it; a beam wider than 16 degrees
/// does not. Kept at the standing 1 degree for the Popoto array (floor 0.22 degrees) although a point target's bearing
/// at 12 km scatters by 0.03 to 0.05 degrees in simulation (sonar-ym1): the report's line goes through the detections,
/// not the filtered state (D-157), so the sigma no longer moves it (tests/python/test_detect.py
/// test_the_12km_report_does_not_depend_on_the_trackers_bearing_sigma), and a ship's own detections scatter by more
/// (23 to 66 m along track side-on at 12 km, 0.1 to 0.3 degrees). The sigma still sets the gate (D-99).
constexpr double kBeamwidthsPerSigma = 16.0;
constexpr double kMultipathSigmas = 3.0;         ///< multipath bearing gate >= this many bearing-sigma floors
constexpr double kFitRmsSigmas = 4.0;            ///< classifier's fit limit >= this many range-sigma floors
/// ... and >= this many of the pooled scatter across the line of sight at the track's range, R x bearing floor / sqrt(2)
/// (the fit's RMS pools two coordinates; sonar-4jm): classify.max_fit_rms_bearing = kFitRmsSigmas x floor / sqrt(2).
constexpr double kRowToleranceWavelengths = 0.25;  ///< row tolerance <= this many wavelengths at the top of the band
constexpr double kPassiveTransits = 2.0;         ///< passive block >= this many array transit times
constexpr double kStreamBufferSeconds = 0.5;     ///< socket buffer >= this much of the board stream

/// What the derivation reads from a record.
struct SonarFacts {
  double sound_speed = 0.0;        ///< m/s (the record's environment)
  double f_low = 0.0, f_high = 0.0, band = 0.0;   ///< Hz; band is 1 / duration for a pulse that does not sweep
  double pulse = 0.0;              ///< s
  double range_resolution = 0.0;   ///< m: c / 2B
  double wavelength = 0.0;         ///< m at the band centre
  double wavelength_min = 0.0;     ///< m at the top of the band
  std::int64_t n_channels = 0, n_rows = 0, row_elements = 0;
  double aperture = 0.0;           ///< m: one row, element count x pitch
  double beamwidth = 0.0;          ///< rad: wavelength / aperture (unweighted)
  double sample_rate = 0.0;        ///< Hz, of the record as it is
  double window_m = 0.0;           ///< m of range the receive window covers
};

/// The record's waveform, array and sound speed. `row_tolerance` and `min_row_elements` find the rows
/// (NaN: the derived tolerance).
SonarFacts sonar_facts(const PingRecord& record, double row_tolerance = kDerive, std::int64_t min_row_elements = 3);

/// Elements within this of one line are one row: the given value, or the smaller of the standing
/// 2 cm and a quarter of the shortest wavelength (rows are at least half of one apart).
double row_tolerance(const PingRecord& record, double given = kDerive);

struct ChainDefaults {
  SonarFacts facts;
  double range_cell_m = 0.0, cfar_guard_m = 0.0, cfar_train_m = 0.0;
  double tracker_range_sigma = 0.0, tracker_bearing_sigma = 0.0;
  double multipath_bearing = 0.0, classify_max_fit_rms = 0.0, row_tolerance_m = 0.0;
  double classify_max_fit_rms_bearing = 0.0;   ///< rad: the fit limit per metre of range (sonar-4jm)
  std::vector<std::string> given;      ///< settings the configuration gave
  std::vector<std::string> scaled;     ///< settings the sonar's floor decided, not the standing value
  std::vector<std::string> warnings;   ///< what a new sonar's owner should look at
  std::string json() const;            ///< one line
};

/// Fill every setting of `options` that was left to be derived, from `record` (after the chain has
/// set its sound speed and dropped dead channels, before conditioning). Returns what it used.
ChainDefaults resolve(ChainOptions& options, const PingRecord& record);

/// Passive block sizes left at 0: n_fft the larger of 1024 and the next power of two above two array
/// transit times at the record's sample rate; hop half of it.
void resolve(PassiveOptions& options, const PingRecord& record);

/// The tolerance two boards must agree within, s: `given_us` (microseconds) if positive, else a
/// twentieth of a period at `top_frequency`.
double sync_tolerance(double given_us, double top_frequency);

/// Bytes per second of the board stream of this sonar: channels x sample rate x bytes per sample.
double stream_bytes_per_second(const PingRecord& template_record, int sample_bytes);

/// The receive buffer to ask the system for: `given` if positive, else the larger of 8 MB and half a
/// second of the stream.
std::int64_t stream_buffer_bytes(const PingRecord& template_record, int sample_bytes, std::int64_t given = 0);

}  // namespace sonar::live
