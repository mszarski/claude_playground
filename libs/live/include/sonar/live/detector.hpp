// The chain's detector at the range resolution: `cell_statistic: multiscale`.
//
// The map is computed finer than the range resolution and was always averaged into range cells
// (2 m for the long-range arrays) before detection. An echo one resolution long then reads
// 10 log10(samples per cell x 0.75) below its peak (7.7 dB at 4-6 kHz), and the cell-averaging
// threshold for ONE exponential sample was applied to a mean of several, so the false-alarm
// probability asked for was never the one obtained (docs/DEPLOYMENT.md, "What snr_db is").
//
// Here every range cell is tested at several integration lengths, each against the threshold of its
// own statistic (detect/statistic.hpp):
//
//     its strongest SAMPLE           (a point echo: nothing is averaged away)
//     its MEAN                       (an echo that fills the cell)
//     and, when asked for (`cfar.integrate_cells`), the mean of several cells (an echo tens of metres long)
//
// over the same background as before (the mean of the cells `cfar.train_m` either side beyond
// `cfar.guard_m`), corrected where that window is one-sided (below). The false-alarm budget is stated
// per PING (`cfar.false_alarms_per_ping`) and shared between the lengths and the map's independent
// cells, so the thresholds follow from the map's size instead of a typed per-cell probability.
//
// A cell that passes any test is flagged. Flagged cells that a stronger cell's own sidelobes explain
// (the compressed pulse's along range, the aperture window's along bearing, with a margin) are
// dropped: a strong echo is one detection, not a mainlobe and a fringe of sidelobe fragments. Flagged
// cells are then grouped: two belong together when they are no further apart in range than the
// tracker can tell apart (`cfar.merge_m`, default its range sigma) and no further apart in bearing
// than half the beam's half-power width (touching cells always do), so a target's highlights, the
// arrivals of one multipath group and the two halves of a weak echo's beam are ONE detection, and
// two targets further apart than that in either are two. It reports its strongest sample over the
// background (`snr_db`), its energy-weighted range, its bearing (the bearings of the group weighted
// by their energy above half the strongest one's) and its extent in range.
//
// The background at the ends of the map. Near either end the training window has cells on one side
// only, and reverberation falls with range: the mean of cells that all lie beyond the cell under test
// is then below the level at the cell (1.7 dB at the near end of the flank_patrol scenario). With a
// threshold of 10.7 dB on a statistic that needed 6 that went unnoticed; with a threshold matched to
// the statistic it is a row of false alarms. So where the window is not balanced the mean is moved
// along the slope of the background, exp(slope x offset of the training cells' centroid), the slope
// being that between the medians of the first (last) two training lengths of the same bearing.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"
#include "sonar/live/derive.hpp"
#include "sonar/live/detections.hpp"

namespace sonar::live {

/// What a detection's range is (DetectorOptions::range_estimate).
///   Centroid  the energy-weighted mean range of the group's samples (the default; the detector as it was).
///   First     the leading edge: the nearest sample of the group whose power over the background of its cell exceeds
///             the single-sample threshold (scales[0].factor, the test its `arrivals` are held to), on the strongest
///             of the group's bearings at each sample; the nearest cell's centre when no sample does (a group flagged
///             by a longer integration only). In a channel the direct arrival comes first and the surface and bottom
///             arrivals after it, so the centroid of the group lies behind the target (popoto_12km: +49.5 m, the
///             leading edge -0.8 m; sonar-2hz). Ours (D-157), opt-in.
enum class RangeEstimate { Centroid, First };
/// "centroid" or "first"; throws std::invalid_argument for anything else.
RangeEstimate parse_range_estimate(const std::string& name);
const char* range_estimate_name(RangeEstimate r);

struct DetectorOptions {
  /// Expected false detections per ping in Gaussian noise, all integration lengths and the whole map together.
  double false_alarms_per_ping = 1e-3;
  /// Integration lengths beyond the single sample, in range cells. Empty: the cell itself. Longer ones (4, 16) gain
  /// 3 to 6 dB on an echo 5 to 100 m long in stationary noise, but their thresholds are only 1.5 to 3 dB over the
  /// background, which a slope in reverberation or a strong echo's bearing sidelobes reach: off unless asked for.
  std::vector<std::int64_t> integrate_cells;
  /// Flagged cells of one bearing no further apart than this in range are one detection, m. NaN: the tracker's range sigma.
  double merge_m = kDerive;
  /// A flagged cell is put down to a stronger cell's sidelobes when it is no more than this above the sidelobe
  /// envelope at its offset. 10 dB: a sidelobe plus noise exceeds the envelope by that much about once in 1e5 cells.
  double sidelobe_margin_db = 10.0;
  /// The bearing sidelobes of a real array do not fall below what its gain and phase errors leave: the envelope's
  /// floor, dB below the peak. -40 dB is errors of about 5 % across 32 hydrophones (an assumption, not a measurement).
  double sidelobe_floor_db = -40.0;
  /// NaN (default): the number of independent tests follows the threshold and the rows, with the measured constants
  /// of kSearchSample / kSearchMean below. A number: every test counts that many per range resolution (or cell) and
  /// beamwidth, whatever the threshold and the rows (2.5 is the detector as it was before the table).
  double search_factor = kDerive;
  /// A detection's range: the centroid of its samples (default) or its leading edge (RangeEstimate). It places the
  /// detection and gives its time, transmit + range / c (D-156).
  RangeEstimate range_estimate = RangeEstimate::Centroid;
};

/// How many separate false detections a map of noise gives: the measured part.
///
/// The map is sampled finer than a range resolution and a beamwidth, the largest sample is searched for, and flagged
/// cells are grouped, so the number of independent tests is not the number of cells. Above a threshold x (over the
/// mean noise power) on a statistic that is a sum of a = rows x looks unit exponentials over a, u = a x:
///   the single sample     separate maxima of a field smooth in range and in bearing: their number over the
///                         exceedance probability grows as u (the region above the threshold about a maximum of
///                         height h has area pi (h - u) / (u sqrt|L|), L the field's roughness; PHYSICS_NOTES Sec. 29.3)
///                             tests = kSearchSample[rows] x beams x extent / resolution x u
///   a mean over cells     independent from cell to cell in range and smooth in bearing only: upcrossings along
///                         bearing over the exceedance probability (Rice), sqrt(u) x the hazard of the statistic
///                             tests = kSearchMean[rows] x beams x extent / length x sqrt(u) x (-d ln P(> x) / dx) / a
/// The constants are MEASURED: tools/detector_search_factor.py, 400 noise-only pings of the long-range array for each
/// of one, two and three rows (0.8 km window, bearings -60 to 60 degrees every 0.5, 2 m cells of 8 samples, Hann
/// aperture), each test alone at 3, 1, 0.3 and 0.1 false detections a ping, 1203 to 2204 detections per entry (2.4 to
/// 2.9 % each). With the growth above they do not depend on the budget within those errors (the single sample: 0.176 to
/// 0.211 over all twelve entries), and the rows are in u: a flat factor of 2.5 read 0.86, 1.16 and 1.24 of the budget
/// for one, two and three rows at these budgets, and more as the budget tightens. More than three rows: the three-row
/// value (not measured). Another map (bearing step, samples per cell, window) needs the tool run again.
inline constexpr double kSearchSample[3] = {0.191, 0.208, 0.192};
inline constexpr double kSearchMean[3] = {0.515, 0.532, 0.598};

struct DetectorScale {
  std::int64_t cells = 0;       ///< 0: the strongest sample of the cell; m >= 1: the mean of m cells about it
  double length_m = 0.0;        ///< what is integrated, m (the range resolution for the sample)
  double looks = 1.0;           ///< independent looks of one row's statistic (variance-equivalent)
  double tests = 0.0;           ///< independent tests of this length in the map, at this test's threshold
  double search = 0.0;          ///< the constant they were counted with (kSearchSample / kSearchMean, or the option's)
  double pfa = 0.0;             ///< false-alarm probability per test
  double factor = 0.0;          ///< threshold over the true mean noise power
  std::vector<double> weights;  ///< the statistic's weights (statistic.hpp)
};

struct DetectorPlan {
  std::vector<DetectorScale> scales;
  std::int64_t rows = 1;                 ///< maps added in power (stacked rows)
  std::int64_t samples_per_cell = 1;
  double cell_m = 0.0, step_m = 0.0, resolution_m = 0.0;
  double beams = 0.0;                    ///< independent beams in the bearing sector
  double train_looks_per_cell = 1.0;     ///< independent exponential looks one training cell holds (rows included)
  double margin = 10.0;                  ///< sidelobe margin, power ratio
  double aperture_m = 0.0;
  std::int64_t merge_cells = 0;          ///< flagged cells up to this many cells + 1 apart in range may be one detection
  double merge_u = 0.0;                  ///< ... when no further apart than this in sin(bearing): half the half-power beamwidth
  /// Range sidelobe envelope by cells apart (index 0 and 1: the mainlobe, 1.0).
  std::vector<double> range_envelope;
  /// Bearing sidelobe envelope against |sin(bearing) - sin(bearing of the stronger cell)|, sampled every `u_step`;
  /// 1.0 inside the mainlobe.
  std::vector<double> bearing_envelope;
  double u_step = 0.0;
  /// The source ratio (strongest sample over background) below which a cell's sidelobes cannot reach the threshold.
  double source_ratio = 0.0;
  /// DetectorOptions::range_estimate: how a detection's range is taken.
  RangeEstimate range_estimate = RangeEstimate::Centroid;
  std::string json() const;
  /// The threshold of scale `s` over the background estimated from `count` training cells.
  double threshold(std::size_t s, double count) const;
};

/// The amplitude correlation of matched-filtered white noise between map samples `step` metres apart, lags 0..n-1:
/// the autocorrelation of the replica the chain compresses with (the pulse with its taper), which is the transform of
/// that replica's own power spectrum. For a flat band it would be sinc(lag step / resolution) and for an untapered
/// tone the triangle of its length; a taper narrows the band and so lowers the looks a cell mean holds (a 10 % Tukey
/// taper on a sweep: 5.47 looks in a 2 m cell at 4 to 6 kHz where the sinc gives 5.80, and 5.42 measured on the map).
std::vector<double> sample_correlation(const Waveform& waveform, double sound_speed, double step, std::int64_t n);

/// The detector's plan for a map of `n_cells` range cells of `samples_per_cell` samples `step` apart and the
/// bearings given, `rows` maps added in power, a background of `train_cells` cells either side. `hann`: the
/// aperture window of the map. `record` gives the band, the pulse and the array. `lines`: the channels of each line
/// whose map is added (stacked rows); empty: every channel, one line. The bearing sidelobe envelope is that of the
/// hydrophones the record has, where they are: a dead board leaves a gap in a row and sidelobes the envelope must know.
DetectorPlan plan_detector(const PingRecord& record, const SonarFacts& facts, const DetectorOptions& options,
                           std::int64_t n_cells, std::int64_t samples_per_cell, double step, const std::vector<double>& bearings,
                           std::int64_t rows, std::int64_t guard_cells, std::int64_t train_cells, bool hann, double merge_m,
                           const std::vector<std::vector<std::int64_t>>& lines = {});

/// The detector's background of every cell of `cell_mean` [n_bearing][n_cells]: the mean of the `train` cells either
/// side beyond `guard`, moved along the background's slope where that window is not balanced (see the top of this
/// file). `train_cells` [n_cells] receives the number of independent training cells the estimate is worth.
void detector_background(const float* cell_mean, std::int64_t n_bearing, std::int64_t n_cells, std::int64_t guard, std::int64_t train,
                         std::vector<double>& background, std::vector<double>& train_cells);

/// The detector's statistic, cell by cell [n_bearing * n_cells]: what multiscale_detect tests before it drops sidelobes
/// and groups.
struct DetectorCells {
  /// false: the device path leaves `level` empty (one array fewer brought back; the chain does not read it).
  bool with_level = true;
  std::vector<double> level;          ///< the background (detector_background)
  std::vector<float> background;      ///< the same in float, as the detector reports it
  std::vector<float> peak;            ///< the cell's strongest sample
  std::vector<unsigned char> flag;    ///< 1 where any test of the plan passes
};

/// The statistic of multiscale_detect on `backend`. Backend::Host is the reference (the loops of detector.cpp);
/// Backend::Cuda runs the same operations on the device (detector_kernel.hpp), one thread per cell, and gives the
/// reference's flags: its background differs from the host's by the last bits of exp and log (tests/cpp/test_detector.cpp,
/// DetectorDevice; tests/python/test_detector_device.py). `fine` and `cell_mean` are host memory either way.
/// Not held: a map with a NaN in the first or last two training lengths of a bearing (the slope's medians), where the
/// host's std::sort and the device's insertion sort order the NaN differently (detector_kernel.hpp).
void multiscale_cells(Backend backend, const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean,
                      std::int64_t n_bearing, std::int64_t n_cells, std::int64_t guard_cells, std::int64_t train_cells,
                      const DetectorPlan& plan, DetectorCells& out, int device = 0);

/// The kernels of the device path run on host threads instead: what the CPU build can test of them against the reference.
void multiscale_cells_kernels_on_host(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean,
                                      std::int64_t n_bearing, std::int64_t n_cells, std::int64_t guard_cells, std::int64_t train_cells,
                                      const DetectorPlan& plan, DetectorCells& out);

/// Detect on `fine` [n_bearing][fine_stride] (power; the cells start at sample `fine_offset` of each row) with
/// `cell_mean` [n_bearing][n_cells] its cell means. `background` receives the background of each cell
/// [n_bearing * n_cells]. Detections carry range (m), bearing (rad), snr_db, cells, first_range and extent.
/// `backend`: where the statistic (multiscale_cells) runs; dropping sidelobes and grouping run on the host.
std::vector<Detection> multiscale_detect(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean,
                                         std::int64_t n_bearing, std::int64_t n_cells, const std::vector<double>& fine_ranges,
                                         const std::vector<double>& cell_ranges, const std::vector<double>& bearings,
                                         std::int64_t guard_cells, std::int64_t train_cells, const DetectorPlan& plan,
                                         std::vector<float>& background, Backend backend = Backend::Host, int device = 0);

/// multiscale_detect after its statistic: sidelobes dropped and flagged cells grouped, from `statistic` (its flags are
/// changed). `background` receives statistic.background.
std::vector<Detection> multiscale_group(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, std::int64_t n_bearing,
                                        std::int64_t n_cells, const std::vector<double>& fine_ranges, const std::vector<double>& cell_ranges,
                                        const std::vector<double>& bearings, const DetectorPlan& plan, DetectorCells& statistic,
                                        std::vector<float>& background);

}  // namespace sonar::live
