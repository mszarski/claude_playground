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
// background (`snr_db`), its energy-weighted position and its extent in range.
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

#include "sonar/core/record.hpp"
#include "sonar/live/derive.hpp"
#include "sonar/live/detections.hpp"

namespace sonar::live {

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
  /// How many independent tests one range resolution x one beamwidth of the map counts as. The map is sampled finer
  /// than both and its largest sample is searched for: measured on noise maps (tests/python/test_detector.py).
  double search_factor = 2.0;
};

struct DetectorScale {
  std::int64_t cells = 0;       ///< 0: the strongest sample of the cell; m >= 1: the mean of m cells about it
  double length_m = 0.0;        ///< what is integrated, m (the range resolution for the sample)
  double looks = 1.0;           ///< independent looks of one row's statistic (variance-equivalent)
  double tests = 0.0;           ///< independent tests of this length in the map
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
  std::string json() const;
  /// The threshold of scale `s` over the background estimated from `count` training cells.
  double threshold(std::size_t s, double count) const;
};

/// The amplitude correlation of the compressed pulse between map samples `step` metres apart, lags 0..n-1:
/// sinc(lag step / resolution) for a swept pulse (flat band), a triangle of the pulse's length for one that is not.
std::vector<double> sample_correlation(const Waveform& waveform, double sound_speed, double step, std::int64_t n);

/// The detector's plan for a map of `n_cells` range cells of `samples_per_cell` samples `step` apart and the
/// bearings given, `rows` maps added in power, a background of `train_cells` cells either side. `hann`: the
/// aperture window of the map. `record` gives the band, the pulse and the array.
DetectorPlan plan_detector(const PingRecord& record, const SonarFacts& facts, const DetectorOptions& options,
                           std::int64_t n_cells, std::int64_t samples_per_cell, double step, const std::vector<double>& bearings,
                           std::int64_t rows, std::int64_t guard_cells, std::int64_t train_cells, bool hann, double merge_m);

/// The detector's background of every cell of `cell_mean` [n_bearing][n_cells]: the mean of the `train` cells either
/// side beyond `guard`, moved along the background's slope where that window is not balanced (see the top of this
/// file). `train_cells` [n_cells] receives the number of independent training cells the estimate is worth.
void detector_background(const float* cell_mean, std::int64_t n_bearing, std::int64_t n_cells, std::int64_t guard, std::int64_t train,
                         std::vector<double>& background, std::vector<double>& train_cells);

/// Detect on `fine` [n_bearing][fine_stride] (power; the cells start at sample `fine_offset` of each row) with
/// `cell_mean` [n_bearing][n_cells] its cell means. `background` receives the background of each cell
/// [n_bearing * n_cells]. Detections carry range (m), bearing (rad), snr_db, cells, first_range and extent.
std::vector<Detection> multiscale_detect(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean,
                                         std::int64_t n_bearing, std::int64_t n_cells, const std::vector<double>& fine_ranges,
                                         const std::vector<double>& cell_ranges, const std::vector<double>& bearings,
                                         std::int64_t guard_cells, std::int64_t train_cells, const DetectorPlan& plan,
                                         std::vector<float>& background);

}  // namespace sonar::live
