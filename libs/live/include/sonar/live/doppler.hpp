// The sweep's Doppler bank in the live chain (D4 of docs/tasks/DOPPLER.md; gpusonar.live `doppler`), opt-in.
//
// A long linear sweep loses matched-filter gain on an echo whose range changes: the echo is the pulse scaled in time by
// eta = (c - r) / (c + r) (r the range rate, opening positive), its sweep rate is k eta^2 against the replica's k, and
// after the delay that matches the frequency offset (the range-Doppler coupling) a quadratic phase error is left whose
// value at the ends of the pulse is
//
//     phi = pi B T |r| / c                                                       (ours, from the LFM's phase; B the band, T the pulse)
//
// and the compressed peak falls to M(phi) = max over x0 of |integral_{-1/2}^{1/2} exp(i 4 phi (u - x0)^2) du| of its matched
// value (a Fresnel integral; x0, where the residual chirp's stationary point lies, is the delay the matched filter's peak
// is free to take: for phi above about 2 it is not the centre). Ours; tests/python/test_doppler_bank.py derives it and
// holds it against an ideal sweep and the simulator: 11.7 dB at 2 m/s on a 1 s, 4 kHz sweep (the simulator's Tukey 0.05
// pulse: 11.4 dB). The design note's tolerance is phi = pi / 2, |r| = c / (2 B T) (docs/design/projector_selection.md,
// "Doppler": 0.375 m/s for 2 kHz and 1 s), where the loss is 0.97 dB.
//
// The bank compresses the record once per range-rate hypothesis (each replica the pulse time-scaled exactly: the sweep is
// wideband, a frequency shift would not do), maps each compression with the chain's beamformer and keeps, per map sample,
// the largest power over the hypotheses that belong to that sample's beam. Hypotheses are r_k = k x step for integer k:
//   - step, unless given: c / (2 B T), so that no echo is further than step / 2 from a hypothesis: phi <= pi / 4 and the
//     worst loss between hypotheses is -20 log10 M(pi / 4) = 0.24 dB (ours; the test holds it);
//   - the own ship's part is taken out per beam first: a still scatterer seen along beam b (the unit vector of the map's
//     rows, forward cos b + across sin b) has the range rate -v . u_b, v the mean of the record's transmit and receive pose
//     velocities (geometry.tx_pose.velocity, geometry.rx_pose.velocity: navigation's). Beam b takes the 2K + 1 hypotheses
//     nearest that rate, K = ceil(max_range_rate / step) (or (hypotheses - 1) / 2): every beam has the same count, so one
//     threshold shift holds for the map, and the bank spans the target speeds only;
//   - a velocity that is not finite is not known: every beam is centred on 0 and K grows by unknown_ownship / step (the
//     bank spans the vehicle's own speed too, at the cost of more hypotheses and a higher background): a ping without a
//     velocity is processed, not refused, because a live chain must not stop on one bad navigation fix (ours).
// The map of the zero hypothesis is the chain's map without the bank, sample for sample: compressed by the same call with
// doppler_scale exactly 1. With max_range_rate 0 (one hypothesis, a shift of 0 dB) the chain is the chain without it.
//
// The maximum over N hypotheses is a different statistic from one map: the background the detector estimates from the
// maximum is higher (a mean of maxima: 2.1 to 5.6 dB for 3 to 33 hypotheses, measured) and its tail is another. Measured
// on noise-only maps (tools/doppler_bank_threshold.py), the detector with its thresholds unchanged gives fewer false
// detections on the maximum than configured; the shift that would restore the configured rate is negative and could not
// be measured at the default budget here, so the chain applies none by default (threshold_db, bank_threshold_db).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sonar/core/record.hpp"
#include "sonar/live/derive.hpp"

namespace sonar::live {

enum class DopplerMode { Off, SweepBank };
DopplerMode parse_doppler_mode(const std::string& name);   ///< off | sweep_bank
const char* doppler_mode_name(DopplerMode m);

struct DopplerOptions {
  DopplerMode mode = DopplerMode::Off;
  /// m/s: the target range rates (over ground, along each beam) the bank spans either side of the beam's own-ship rate.
  double max_range_rate = 3.0;
  /// m/s between hypotheses; NaN: c / (2 B T) of the record's sweep (the worst loss between two hypotheses 0.24 dB).
  double step = kDerive;
  /// > 0: hypotheses per beam (odd), instead of max_range_rate: K = (hypotheses - 1) / 2 either side.
  std::int64_t hypotheses = 0;
  /// m/s: added to the span when the record's velocity is not finite (the bank then spans the vehicle's own speed too).
  /// 2.0: the scenes' vehicle makes 1.5 m/s; set it to the vehicle's top speed (an assumption, configuration).
  double unknown_ownship = 2.0;
  /// dB every threshold of the detector (and min_snr_db) moves by; NaN: bank_threshold_db (0, measured: doppler.cpp).
  double threshold_db = kDerive;
  /// Report each detection's range rate (`range_rate_m_s`, `range_rate_step_m_s`) on the report line: off by default
  /// (docs/tasks/DOPPLER.md section 5, question 6: the report line is a contract).
  bool report = false;
};

/// What the bank does to one record: its hypotheses and which belong to each beam.
struct DopplerPlan {
  bool on = false;
  double step = 0.0;                     ///< m/s
  std::vector<double> rates;             ///< m/s, range rate relative to the platform, opening positive, ascending: k x step
  std::vector<double> scales;            ///< doppler_scale of each (pulse_compress): (c - r) / (c + r); exactly 1 for r = 0
  /// s added to each compressed hypothesis' samples.start_time, 2 f0 T r / (B c) (the LFM coupling of its own rate,
  /// lfm_range_doppler_factor): every hypothesis then shows an echo where the zero hypothesis shows it, so the maximum
  /// takes the matched peak at today's (coupled) range instead of smearing it along the range-Doppler ridge. 0 for r = 0.
  std::vector<double> shifts;
  std::int64_t zero = -1;                ///< the index of the zero hypothesis in `rates`; -1 when no beam uses it
  std::int64_t per_beam = 1;             ///< hypotheses each beam takes (2K + 1)
  std::vector<std::int64_t> first;       ///< per beam: the index of its first hypothesis in `rates`
  std::vector<double> ownship;           ///< per beam: the range rate of a still scatterer along it (-v . u_b), m/s; 0 unknown
  bool velocity_known = true;
  double baseband_max_range_rate = 0.0;  ///< what to_baseband is given: 0 when the default band holds every shift
  double threshold_db = 0.0;             ///< the detector's thresholds move by this (dB)
  /// m: a mismatched hypothesis spreads a strong echo over +-|mismatch| T about its range (the residual chirp); the
  /// detector's range sidelobe envelope is raised to that pedestal out to per_beam x step x T (DetectorOptions::pedestal_m).
  double pedestal_m = 0.0;
  std::string json() const;
};

/// The compressed peak of a linear sweep left with a quadratic phase error phi at the ends of the pulse, over its matched
/// peak: max over x0 of |integral_{-1/2}^{1/2} exp(i 4 phi (u - x0)^2) du| (ours; the top of this file).
double sweep_mismatch(double phi);

/// The shift (dB) the chain applies to the multiscale detector's thresholds on the maximum of `per_beam` hypotheses: 0, on
/// the safe side of what was measured (doppler.cpp: on the maximum the unchanged thresholds give fewer false detections
/// than configured at every budget counted).
double bank_threshold_db(std::int64_t per_beam);

/// The bank for `record` (its sweep, sound speed and pose velocities) and the map's bearings (rad from broadside, the
/// chain's convention; `starboard` as ChainOptions). Throws std::invalid_argument for a pulse that is not a linear sweep
/// (D3's tone bank is another stage), a step that is not positive, or an even hypothesis count. The hypotheses' maps are
/// aligned by `shifts` on one range grid: the chain takes the bank with its backprojection beamformer only.
DopplerPlan plan_doppler(const PingRecord& record, const std::vector<double>& bearings, const DopplerOptions& options);

/// The largest power over each beam's hypotheses, folded in one hypothesis at a time: `fine` [n_bearing][n] the map of
/// hypothesis `h`; `best` and `which` [n_bearing][n] the running maximum and the hypothesis that holds it (start them
/// with fold_start). A beam that does not take `h` is left alone. A tie keeps the earlier hypothesis.
void fold_start(std::int64_t n_bearing, std::int64_t n, std::vector<float>& best, std::vector<std::int32_t>& which);
void fold_hypothesis(const DopplerPlan& plan, std::int64_t h, const float* fine, std::int64_t n_bearing, std::int64_t n,
                     std::vector<float>& best, std::vector<std::int32_t>& which);

/// The hypothesis a detection takes: that of the strongest sample of the maximum map along its nearest bearing row over
/// [lo, hi] m of `fine_ranges` (the row's whole map when nothing lies there). -1 when the map holds none.
std::int64_t detection_hypothesis(const float* best, const std::int32_t* which, std::int64_t fine_stride, std::int64_t fine_offset,
                                  std::int64_t n_fine, const std::vector<double>& fine_ranges, const std::vector<double>& bearings,
                                  double bearing, double lo, double hi);

}  // namespace sonar::live
