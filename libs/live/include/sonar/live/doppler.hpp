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

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"
#include "sonar/live/derive.hpp"
#include "sonar/live/detections.hpp"

namespace sonar::live {

enum class DopplerMode { Off, SweepBank };
DopplerMode parse_doppler_mode(const std::string& name);   ///< off | sweep_bank
const char* doppler_mode_name(DopplerMode m);

/// How the sweep bank's hypotheses become one map (D3 of docs/tasks/DOPPLER.md).
///   Max   the largest power over each beam's hypotheses, then the detector as it is (D4): the detector's background is
///         estimated from that maximum, which rises 4.8 to 5.6 dB on noise for 17 to 33 hypotheses (D4, measured) and
///         costs every target that much margin.
///   Cfar  each (beam, hypothesis) map normalised by ITS OWN background first (the detector's background estimate,
///         detector_background, on that hypothesis' cell means), then the largest normalised value over the beam's
///         hypotheses, tested against thresholds planned for a budget divided by the hypotheses per beam (the union
///         bound: P(max over N > x) <= N P(one > x); ours). The map handed to the detector is that normalised maximum
///         times the background of the beam's centre (own-ship) hypothesis, which the detector is given as its
///         background instead of estimating one from the maximum (multiscale_detect's `level`): a detection's snr_db
///         is its value over its own hypothesis' background.
enum class DopplerCombine { Max, Cfar };
DopplerCombine parse_doppler_combine(const std::string& name);   ///< max | cfar
const char* doppler_combine_name(DopplerCombine c);

/// Where the bank's hypotheses and the tone's filter bank are computed (D8 of docs/tasks/DOPPLER.md; doppler_batched.hpp).
///   Auto  the batched formulation on the cuda backend (every hypothesis on the device from one upload of the baseband
///         record to the combined map), the per-hypothesis loop on the host (the code before D8, byte for byte);
///   On    the batched formulation on either backend (on the host: the same kernels on CPU threads, dsp::HostExec; how it
///         is tested on a machine without a device);
///   Off   the loop on either backend (on cuda: each hypothesis' compression and map on the device, the fold on the host).
enum class DopplerBatched { Auto, On, Off };
DopplerBatched parse_doppler_batched(const std::string& name);   ///< auto | on | off
const char* doppler_batched_name(DopplerBatched b);

/// The tone path's notch about the own-motion ridge seen through each beam's grating lobes (plan_tone, step 4b):
///   Auto  on wherever a beam has a visible grating lobe at the tone's frequency (nothing changes where none is);
///   Off   the steered direction's notch alone (the tone path before 2026-11-27: for comparison).
enum class ToneLobeNotch { Auto, Off };
ToneLobeNotch parse_tone_lobe_notch(const std::string& name);   ///< auto | off
const char* tone_lobe_notch_name(ToneLobeNotch n);

/// D3 part B: the tone's Doppler filter bank (tone_stage below), opt-in. Every NaN is derived (plan_tone says from what).
struct ToneOptions {
  bool enabled = false;               ///< doppler_tone = bank
  double frequency = kDerive;         ///< Hz; NaN: the record's (a CW record); required for a train (an arbitrary record)
  double start = 0.0;                 ///< s from the start of the pulse to the tone's start (a train's segment table)
  double duration = kDerive;          ///< s; NaN: the record's (a CW record)
  double window_alpha = kDerive;      ///< the tone's Tukey taper (1: Hann); NaN: the record's (a CW record)
  double max_range_rate = kDerive;    ///< m/s either side of each beam's own-ship rate; NaN: DopplerOptions::max_range_rate
  double step = kDerive;              ///< m/s between hypotheses; NaN: c / (4 f T), half the resolution c / (2 f T)
  double hop = 0.25;                  ///< lags this fraction of the tone's length apart
  double notch = kDerive;             ///< m/s half-width about each beam's own-ship rate not tested; NaN: derived (plan_tone)
  double notch_cells = 3.0;           ///< resolution cells added to the beam's own-motion span in the derived notch
  std::int64_t guard = 0;             ///< hypotheses either side kept out of a cell's background; 0: 2 resolution cells
  std::int64_t train = 32;            ///< hypotheses either side in a cell's background (beyond the guard)
  double false_alarms_per_ping = kDerive;  ///< the tone path's budget; NaN: the detector's (cfar.false_alarms_per_ping)
  double gate_range = kDerive;        ///< m: a tone and a sweep detection this close in range are one; NaN: c T / 4
  double gate_bearing = kDerive;      ///< rad: ... and this close in bearing; NaN: 2 degrees
  double range_sigma = kDerive;       ///< m: the tracker's range sigma for a tone-only detection; NaN: c T / 8
  double sidelobe_db = kDerive;       ///< a weaker group under a stronger one's bearing sidelobes; NaN: -25 dB (Hann), -7 dB (rect)
  /// dB: a cell no more than this above what a stronger cell of its beam and lag (the ridge in the notch, or a stronger
  /// tested cell) leaks into it through the tone's spectrum is not a detection (tone_stage step 3b). 10: the detector's
  /// sidelobe margin (detector.hpp).
  double leak_margin_db = 10.0;
  /// The grating lobes' notches (ToneLobeNotch; plan_tone step 4b). Auto: on with the tone path.
  ToneLobeNotch lobe_notch = ToneLobeNotch::Auto;
  /// dB: a beam whose response towards endfire (either side; the aperture window's array factor at the tone, over its
  /// main lobe's) is at least this gets the endfire ridge notched too (step 4b: the skirt of a lobe centred just beyond
  /// endfire). -40: ours (doppler.hpp step 4b says why).
  double lobe_floor_db = -40.0;
};

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
  /// How the hypotheses are combined (DopplerCombine): max (D4, the default of the bank) or cfar (D3).
  DopplerCombine combine = DopplerCombine::Max;
  /// dB every threshold of the detector (and min_snr_db) moves by; NaN: bank_threshold_db (0, measured: doppler.cpp).
  /// With combine cfar it is added to the union bound's correction (the budget divided by the hypotheses per beam).
  double threshold_db = kDerive;
  /// The tone's Doppler filter bank (D3 part B), beside the sweep path (on with mode off too).
  ToneOptions tone;
  /// D8: the batched formulation (DopplerBatched) and the hypotheses one batch holds (0: as many as the device's free
  /// memory allows; doppler_batched.hpp sweep_bank_batch). Neither changes a result on the host backend with Auto.
  DopplerBatched batched = DopplerBatched::Auto;
  std::int64_t batch = 0;
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
  DopplerCombine combine = DopplerCombine::Max;
  /// The detector's false-alarm budget is divided by this: 1 with max; the hypotheses per beam with cfar (union bound).
  double budget_divisor = 1.0;
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

/// combine cfar: the running maximum over each beam's hypotheses of the map normalised by its own background.
struct NormalisedFold {
  std::int64_t n_bearing = 0, stride = 0, n_cells = 0, cell_samples = 1, skip = 0;
  std::vector<float> ratio;          ///< [n_bearing][stride] sample / the background of its cell in its hypothesis; max over h
  std::vector<float> cell_ratio;     ///< [n_bearing][n_cells] cell mean / its background; max over h
  std::vector<double> reference;     ///< [n_bearing][n_cells] the background of each beam's centre hypothesis
  std::vector<std::int32_t> which;   ///< [n_bearing][stride] the hypothesis that holds `ratio`
};
/// Starts the fold for a map of `n_cells` cells of `cell_samples` samples per bearing row (stride n_cells x cell_samples);
/// the first `skip` cells (blanked) are left out of every background and hold 0.
void normalised_start(std::int64_t n_bearing, std::int64_t n_cells, std::int64_t cell_samples, std::int64_t skip, NormalisedFold& fold);
/// Folds in hypothesis `h`'s map `fine` [n_bearing][stride]: its cell means (the mean of each cell's samples, in double),
/// their background by detector_background(guard, train) over the cells from `skip` on, and each sample's and cell's ratio
/// to it; a beam that does not take `h` is left alone; a tie keeps the earlier hypothesis. A background of 0 counts as
/// 1e-300 (as the detector's floor).
void fold_normalised(const DopplerPlan& plan, std::int64_t h, const float* fine, std::int64_t guard, std::int64_t train, NormalisedFold& fold);
/// The map the detector is handed: `fine` = ratio x reference (per cell), `cells` = cell_ratio x reference, `level` =
/// reference (0 in the skipped cells; the chain drops them).
void normalised_map(const NormalisedFold& fold, std::vector<float>& fine, std::vector<float>& cells, std::vector<double>& level);

/// D8: the geometry of the chain's backprojection map (chain.cpp range_bearing_image): the aperture weights of the
/// channels, the array's origin and each bearing's direction; point (b, q) is origin + dirs[b] x range q.
struct MapGeometry {
  std::vector<float> weights;
  Vec3d origin;
  std::vector<Vec3d> dirs;
};

/// D8: the sweep bank's combined map, as the chain's per-hypothesis loop leaves it: `fine` and `which` [n_bearing][n_cells
/// x k] (the maximum, or with combine cfar the normalised map), `cells` [n_bearing][n_cells] (the detector's cell means),
/// `level` (cfar: the background the detector is given, 0 in the skipped cells). `batch`: the hypotheses a batch held.
struct SweepBankMaps {
  std::vector<float> fine, cells;
  std::vector<std::int32_t> which;
  std::vector<double> level;
  std::int64_t batch = 0;
};

/// D8: every hypothesis of `plan` compressed from the baseband record `base` (before compression), mapped onto
/// `fine_ranges` (n_cells x k, the zero hypothesis' grid) by the backprojection kernel and folded, by the batched
/// formulation (doppler_batched.hpp): on the device with Backend::Cuda, on CPU threads otherwise. combine cfar: `skip`
/// blanked cells and the detector's window (`guard`, `train`). `batch` 0: as many hypotheses at a time as memory allows.
SweepBankMaps sweep_bank_maps(const PingRecord& base, const DopplerPlan& plan, const MapGeometry& geometry, const std::vector<double>& fine_ranges,
                              std::int64_t n_cells, std::int64_t k, std::int64_t skip, std::int64_t guard, std::int64_t train, Backend backend,
                              int device, std::int64_t batch);

// ------------------------------------------------------------------------------------------------------------------------
// D3 part B: the tone's Doppler filter bank (docs/tasks/DOPPLER.md D3; gpusonar/doppler.py tone_stage), opt-in.
//
// For a transmit with a tone (a CW record, or a train whose tone segment the configuration names: frequency f, start t1,
// length T, taper), per beam of the conventional far-field beamformer (farfield_beamform, the chain's bearings) on the
// complex baseband record BEFORE pulse compression:
//   1. the beam is mixed to the tone and decimated by an exact band-limited FFT cut to the band the hypotheses need,
//      +-(2 r_max f / c + 8 / T) about the tone (r_max the largest |rate| of any hypothesis);
//   2. for each lag l (every hop x T) and range-rate hypothesis r_k the matched filter of the tone at that rate,
//      X = sum_m z[lH + m] w[m] exp(-i 2 pi nu_k (lH + m) / fs), w the tone's Tukey envelope (the replica's), nu_k the
//      echo's frequency f (c - r_k) / (c + r_k) in the decimated frame (a narrowband echo: the envelope's own time scaling,
//      under T |r| / c = 2 ms at 3 m/s and 1 s, is left out; ours); power |X|^2 / sum w^2;
//   3. hypotheses r_k = k x step, step c / (4 f T) (half the resolution c / (2 f T): the worst loss between two, a Hann
//      tone, is under 1 dB; ours), each beam taking the 2K + 1 nearest its own-ship rate -v . u_b as the sweep bank does;
//   4. a NOTCH: the hypotheses within `notch` of the beam's own-ship rate are not tested and not used as background: the
//      stationary reverberation (and every stationary target) lies there. Derived: the own-ship rate's spread over the
//      beam's half-power width (|v| |du/db| x half the width 0.886 lambda / (L cos b), ours) plus `notch_cells` resolution
//      cells (the Hann tone's main lobe is +-2 cells; 3 leaves its first sidelobes, ours). The sweep path keeps the
//      stationary targets;
//   4b. GRATING LOBES (tone_lobe_notch auto, the default with the tone path): a uniform line of pitch d steered to b has,
//      at the tone's wavelength lambda, a grating lobe wherever sin g = sin b + m lambda / d (m a non-zero integer) is
//      visible, |sin g| <= 1 (the array factor of the far-field beamformer is periodic in sin with period lambda / d: the
//      lobe is the main lobe again, at its level for omnidirectional elements; the aperture window shapes both alike and
//      lowers neither; an element pattern would, and none is configured: ours). Above the sweep's band (the tone 0.5 kHz
//      above it, the shipped rule) the pitch is more than half a wavelength: 0.65 at 6.5 kHz on popoto_long_range's
//      0.15 m, lobes from |b| = 32.6 degrees (c 1500 m/s; the endfire rule below from 24); 0.66 at 10.5 kHz on
//      flank_32's 0.094 m, from 31.3 degrees (the endfire rule from 23).
//      The still water seen through the lobe lies at -v . u_g, not at the beam's -v . u_b: for motion along the line
//      the two differ by |v| m lambda / d (2.31 m/s at 1.5 m/s and 6.5 kHz), well inside the bank's span, and the
//      lobe's own-motion ridge is tested as a target unless it is notched too (measured: 3 to 5 false tone detections
//      on every tone ping of popoto_12km_large_submerged, DOPPLER.md "Large submerged at 12 km" LS.1 item 0). So each
//      visible lobe's ridge gets a notch of the same rule as step 4 at g: |v . du_g| x half the width 0.886 lambda /
//      (L max(|cos g|, 0.05)) plus notch_cells resolution cells (or tone_notch_m_s when given). A lobe centred just
//      BEYOND endfire still receives the endfire water through its skirt, where the water is densest in sine (an
//      interval ds of sine spans dtheta = ds / cos theta): measured on a still field all round, beams at 24 to 31.5
//      degrees whose lobe centre is invisible reported the endfire ridge (rates 1.41 to 1.55 m/s, 19 to 39 dB) once
//      the visible lobes were notched (ours, tests/python/test_doppler_tone_lobes.py's record). So a beam whose
//      response towards endfire (the array factor of its aperture weights at the tone, either side, over its main
//      lobe's) is at least tone_lobe_floor_db (-40 dB) gets the endfire ridge notched too, by the same rule at
//      g = +-90 degrees. -40 dB, ours: a ridge up to about 55 dB over the noise in the beam's own lobe then stays
//      under the CFAR's 15 dB; the Hann aperture's response towards endfire is -44 to -160 dB on every beam within
//      +-50 degrees where no lobe is near (flank_32 at 8 kHz, popoto_long_range at 5 kHz), -38 to -0.1 dB from 24
//      degrees at 6.5 kHz on popoto_long_range. Lobe directions are taken in the beam's half-plane (forward
//      cos g + across sin g) as step 4 takes the beam's: for a velocity across the line the mirror half differs by
//      2 v_across cos g, not notched (small near endfire, where the lobes fall). Only a UNIFORM line has lobes here:
//      the hydrophones' coordinates along the line (rows that share them count once) equally spaced to 1e-3 of the
//      pitch; any other array takes no lobe notch (TonePlan::pitch 0). With the velocity unknown every beam's ridge
//      is 0 and so is every lobe's: nothing is added. Where a lobe's ridge falls inside the beam's own notch (a still
//      platform: every ridge at 0) nothing changes, bit for bit. The cost: a target whose rate lies in a lobe's notch
//      is not reported by the tone (the sweep path still sees it); TonePlan::lobe_hidden says how much per beam;
//   5. the background of a cell (b, k, l) is the mean power over the beam's hypotheses `guard` + 1 to `guard` + `train`
//      away on either side at the same lag, notch excluded: a CFAR along the RATE axis. Not along range: a ping holds
//      too few lags for that (flank_patrol: 13 at a quarter of 1 s), and the background that matters here is flat in rate
//      and not in range (noise; the sweep's broadband reverberation when the tone is inside its band, D2). Ours;
//   6. a cell is flagged when its power over that background exceeds the cell-averaging factor
//      a = N (pfa^(-1/N) - 1) (exponential power; N the training cells times step / (enbw x resolution), enbw the
//      window's equivalent noise bandwidth in bins: neighbouring hypotheses are not independent; ours), with pfa the
//      budget over every cell tested (the union bound: on the safe side, cells overlap);
//   6b. a flagged cell no more than `leak_margin_db` above what a stronger cell of its beam and lag leaks into it through
//      the matched filter's own spectrum (the window's power spectrum, its envelope) is dropped: the ridge's leakage (a
//      strong still echo or the near reverberation outside the notch) and a strong mover's own Doppler sidelobes;
//   7. flagged cells touching in (beam, hypothesis, lag) are one detection at its strongest cell, rate, bearing and lag
//      refined by a parabola through the log power of the neighbours; a group no stronger than `sidelobe_db` under a
//      stronger one's power at a nearby rate and lag is that one's bearing sidelobe and dropped.
// A detection's range is c (t_l - t1) / 2 (t_l the lag's start after the transmit): COARSE, the tone's c T / 2 cell; its
// rate is relative to the platform, opening positive (the sweep bank's convention). The chain (chain.cpp, live.py) merges
// them with the sweep's detections (DopplerOptions::tone gates).

/// A visible grating lobe of one beam at the tone's frequency and the notch about its own-motion ridge (step 4b).
struct ToneLobe {
  std::int64_t beam = 0;
  double direction = 0.0;            ///< rad from broadside, the map's convention (clamped to +-pi/2 at endfire)
  double ownship = 0.0;              ///< m/s: the range rate of still water along the lobe, -v . u_g
  double notch = 0.0;                ///< m/s half-width about it not tested
  bool endfire = false;              ///< true: added for the beam's response towards endfire (lobe_floor_db), not a visible lobe's centre
  double response_db = 0.0;          ///< the beam's response in that direction over its main lobe's (0 for a visible lobe's centre)
};

/// The directions (rad from broadside) of the visible grating lobes of a uniform line of `pitch` m steered to `bearing`
/// rad at `wavelength` m: asin(sin b + m lambda / d) for every non-zero integer m with |sin b + m lambda / d| <= 1 + slack
/// (sine clamped to +-1: a lobe just beyond endfire counts at endfire), in increasing m. Empty for a pitch or wavelength
/// that is not positive and finite. A pitch of half a wavelength or less has none for |bearing| < pi/2 with slack 0.
std::vector<double> grating_lobes(double bearing, double pitch, double wavelength, double slack = 0.0);

/// The pitch of a uniform line from the hydrophones' coordinates along it (m; any order; coordinates within 1e-6 of the
/// span count once: rows of one line): the mean spacing when every spacing is within `tolerance` of it (relative), else
/// 0 (not uniform, or fewer than two distinct coordinates).
double uniform_pitch(std::vector<double> along, double tolerance = 1e-3);

struct TonePlan {
  bool on = false;                   ///< false: this record has no tone (a sweep-only ping)
  bool tone_only = false;            ///< the tone is the whole pulse: the sweep path has nothing to say (its detections dropped)
  double frequency = 0.0, start = 0.0, duration = 0.0, window_alpha = 0.0;
  double step = 0.0, resolution = 0.0;   ///< m/s
  std::vector<double> rates;         ///< m/s, ascending, k x step
  std::int64_t per_beam = 1;
  std::vector<std::int64_t> first;   ///< per beam, index into rates
  std::vector<double> ownship;       ///< per beam, m/s
  std::vector<double> notch;         ///< per beam, m/s half-width
  double hop = 0.0;                  ///< s between lags
  std::int64_t guard = 0, train = 0;
  double budget = 0.0;               ///< false detections a ping
  double baseband_max_range_rate = 0.0;   ///< what to_baseband is given so that the tone and its shifts are kept
  double gate_range = 0.0, gate_bearing = 0.0, range_sigma = 0.0, sidelobe_db = 0.0;
  /// Step 4b: the line's pitch (m; 0: not a uniform line, no lobe notch), every beam's visible lobes with their notches
  /// (none with lobe_notch off or the velocity unknown), per beam the rate interval the lobes' notches hide beyond the
  /// beam's own notch (m/s: its hypotheses notched by a lobe only, times step), and the beams with any.
  ToneLobeNotch lobe_notch = ToneLobeNotch::Auto;
  double pitch = 0.0;
  std::vector<ToneLobe> lobes;
  std::vector<double> lobe_hidden;
  std::int64_t lobe_beams = 0;
  std::string json() const;
};

/// The notch of every (beam, own hypothesis) of `plan`, [n_beams][per_beam], 1 not tested: the beam's own (step 4,
/// |r - ownship| <= notch) or any of its lobes' (step 4b). tone_stage's, on either backend.
std::vector<unsigned char> tone_notch_mask(const TonePlan& plan);

/// The tone path for `record` (before baseband or after: the allowance needs the record's waveform and sound speed) and
/// the map's bearings. Off (on = false) when the tone bank is not enabled or the record is a linear sweep (the other ping of
/// an alternating pair). Throws std::invalid_argument for a train (an arbitrary record) without `tone.frequency`, a tone
/// that does not fit the pulse, or numbers out of range.
/// `default_budget`: the tone path's false-alarm budget when ToneOptions::false_alarms_per_ping is not given (the chain
/// passes its detector's).
/// `hann`: the aperture window the tone's beams take (tone_stage's; step 4b's endfire response uses it).
TonePlan plan_tone(const PingRecord& record, const std::vector<double>& bearings, const DopplerOptions& options, double default_budget = 1e-3,
                   bool hann = true);

struct ToneDetection {
  double range = 0.0;        ///< m, c (t_l - t1) / 2 at the refined lag
  double bearing = 0.0;      ///< rad
  double range_rate = 0.0;   ///< m/s, refined
  double snr_db = 0.0;       ///< the strongest cell over its background
  std::int64_t cells = 0;
  std::int64_t beam = 0, hypothesis = 0, lag = 0;   ///< the strongest cell (hypothesis: index into rates)
};

struct ToneStage {
  TonePlan plan;
  std::vector<ToneDetection> detections;
  std::int64_t n_lags = 0, tests = 0;
  double factor = 0.0;                ///< the threshold of a cell with a full background (power over its mean)
  double fs = 0.0;                    ///< Hz, the decimated rate
  std::vector<double> lag_ranges;     ///< m
  /// With keep_map: [n_bearing][per_beam][n_lags] power and power over its background (0 where not tested).
  std::vector<float> power, ratio;
};

/// The tone path on `record` (complex baseband, NOT pulse compressed; the chain keeps it before compression): the plan,
/// the beams (farfield_beamform with the chain's aperture window), the bank, the CFAR along rate and the detections.
ToneStage tone_stage(const PingRecord& record, const std::vector<double>& bearings, const DopplerOptions& options, bool hann,
                     double default_budget = 1e-3, Backend backend = Backend::Host, int device = 0, bool keep_map = false);

/// The sweep path's replica on a train with the tone path on (D3): the record's replica (an arbitrary waveform: the train as
/// sent) with the tone segment's samples set to zero, emission times [start, start + duration] of the plan. The tone's
/// echo is Doppler-shifted off the zero-Doppler sweep replica's tone part (a 1 s tone tolerates c / (2 f T): 0.07 to 0.15 m/s),
/// so with the whole train as the replica the sweep path pays the tone's half of the energy in its noise and gains nothing
/// for a mover (measured: popoto_12km_doppler's ship 0 of 6 pings through the chain with the whole train; DOPPLER.md D3).
/// Leaves anything that is not an arbitrary waveform with a replica alone; returns the samples zeroed.
std::int64_t sweep_only_replica(Waveform& waveform, const TonePlan& plan);

/// The chain's merge of the two paths (D3). Tone detections, strongest first, each take the strongest sweep detection not
/// yet taken within the plan's gates (|range difference| <= gate_range, |bearing difference| <= gate_bearing): that sweep
/// detection keeps its range and bearing (the sweep's resolution) and takes the tone's range rate and step (doppler_path
/// 2). A tone detection that finds none is returned as a detection of its own (doppler_path 1, its coarse range, the
/// plan's range_sigma for the tracker). Sweep detections no tone detection takes are left as they are (path 0): a
/// stationary target seen only by the sweep and a mover seen only by the tone both survive.
std::vector<Detection> merge_tone(std::vector<Detection>& sweep, const ToneStage& tone);

/// The hypothesis a detection takes: that of the strongest sample of the maximum map along its nearest bearing row over
/// [lo, hi] m of `fine_ranges` (the row's whole map when nothing lies there). -1 when the map holds none.
std::int64_t detection_hypothesis(const float* best, const std::int32_t* which, std::int64_t fine_stride, std::int64_t fine_offset,
                                  std::int64_t n_fine, const std::vector<double>& fine_ranges, const std::vector<double>& bearings,
                                  double bearing, double lo, double hi);

}  // namespace sonar::live
