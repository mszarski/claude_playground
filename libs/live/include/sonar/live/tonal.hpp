// Passive tonal analysis of the flank array: Goertzel lines and DEMON.
//
// Narrowband lines (machinery tonals): a small bank of Goertzel filters around each frequency
// asked for, on the array steered at a bearing or on the hydrophones; per block of data the
// line's frequency (interpolated between bins), amplitude, phase, signal to noise against the
// bins either side, the bearing it is strongest at, and a track of its frequency (Doppler).
//
// DEMON (detection of envelope modulation on noise): per bearing, the beam in a band, its
// envelope, decimated; the envelope's spectrum averaged over time and normalised by its local
// background; lines above a threshold set from a false-alarm rate; shaft rate, blade rate and
// blade count from the lines' harmonic relations. The normalised spectra over bearing are the
// display product (DEMONgram).
//
// TonalAnalyzer::push takes stretches of data as they arrive and returns what each completes;
// cutting a recording into pieces gives the same results as passing it whole. The NumPy
// reference, with the derivations, is python/gpusonar/tonal.py; tests/python/test_tonal.py
// holds one against the other.
//
// Formulation read in and re-implemented (no code copied): the Goertzel recurrence as in
// arlpy `signal.goertzel` (org-arl, BSD 3-clause) and SignalAnalysis.jl `goertzel` (org-arl,
// MIT); DEMON as in SignalAnalysis.jl `demon` (envelope, reduced to its RMS over each
// decimation block, high-passed). See tonal.py for what differs and why.
#pragma once

#include <complex>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::live {

struct TonalOptions {
  bool enabled = false;
  double bearing_first_deg = -80.0, bearing_last_deg = 80.0, bearing_step_deg = 1.0;
  bool hann = true;               ///< aperture weighting of each row of hydrophones
  double row_tolerance_m = std::numeric_limits<double>::quiet_NaN();  ///< hydrophones within this of one line are one row; NaN: derived (derive.hpp)
  double attach_deg = 3.0;        ///< a result within this of a passive bearing is reported as that bearing's
  struct Goertzel {
    bool enabled = false;
    std::vector<double> lines;    ///< Hz, absolute
    bool beams = true;            ///< false: on the hydrophones, channel powers added
    double block_s = 1.0, hop_s = 0.0;  ///< hop 0: the block length
    std::int64_t search_bins = 4, guard_bins = 3, noise_bins = 8;
    double min_snr_db = 10.0;
    bool track = true;
    double alpha = 0.5, beta = 0.1;
    std::int64_t max_misses = 3;
    double max_shift_hz = 0.0;    ///< 0: no limit
  } goertzel;
  struct Demon {
    bool enabled = false;
    double f_low = 0.0, f_high = 0.0;  ///< Hz, absolute; both 0: the band given to the analyzer, else the transmit band
    bool square = true;                ///< false: the root of the band power ("hilbert")
    double envelope_rate_hz = 100.0;
    std::int64_t window_hops = 2;
    double segment_s = 8.0, segment_overlap = 0.5;
    std::int64_t n_average = 8, min_segments = 4;
    double f_min_hz = 1.0, f_max_hz = 0.0;  ///< 0: 0.8 of half the envelope rate
    std::int64_t background_bins = 16, guard_bins = 2;
    double clip = 3.0, pfa = 1e-6;
    std::int64_t max_lines = 12, max_sources = 4;
    double min_separation_deg = 4.0;
    double shaft_min_hz = 0.5, shaft_max_hz = 30.0;
    std::int64_t max_harmonic = 12;
    double harmonic_tolerance_bins = 1.0, score_fraction = 0.9;
    std::int64_t blades_min = 2, blades_max = 9;
    bool gram = false;                 ///< put the DEMONgram in every result's JSON
  } demon;
};

/// Set one `tonal...` option from its key=value text; false if the key is not a tonal option.
bool set_tonal_option(TonalOptions& options, const std::string& key, const std::string& value);

struct TonalLine {
  double nominal = 0.0, frequency = 0.0, amplitude = 0.0, phase = 0.0, snr_db = 0.0, noise_db = 0.0;
  double bearing = 0.0;        ///< rad; meaningful with has_bearing
  double track = 0.0, rate = 0.0;
  bool detected = false, has_bearing = false, tracking = false;
};

struct DemonLine {
  double frequency = 0.0, snr_db = 0.0, modulation = 0.0, bearing = 0.0;
  std::int64_t harmonic = 0;   ///< of the source's shaft rate; 0: not one
};

struct DemonSource {
  double bearing = 0.0;        ///< rad
  double shaft_hz = 0.0, blade_hz = 0.0, confidence = 0.0;  ///< rates 0: not known
  std::int64_t blades = 0;     ///< 0: not known
  std::vector<DemonLine> lines;
};

struct TonalEvent {
  bool demon = false;
  std::int64_t time_ns = 0;    ///< of the last sample used
  double block_s = 0.0;
  std::vector<TonalLine> lines;
  std::int64_t segments = 0;
  double independent = 0.0, threshold_db = 0.0;  ///< spectra averaged; the independent ones they stand for
  std::vector<DemonSource> sources;
  std::vector<float> gram_db;  ///< with demon.gram: this result's DEMONgram, [n_bearing][n_frequency]
};

/// The Fourier sum of x at one frequency by the Goertzel recurrence, phase referred to the first
/// sample: sum_n x[n] exp(-2 pi j f n / fs).
std::complex<double> goertzel(const float* x, std::int64_t n, double frequency, double sample_rate);
std::complex<double> goertzel(const std::complex<float>* x, std::int64_t n, double frequency, double sample_rate);

/// Shaft rate, blade rate and blade count from one source's modulation lines (frequency and
/// harmonic of each line are read and written); `df` is the modulation bin width.
void harmonic_set(DemonSource& source, double df, const TonalOptions::Demon& options);

/// Ratio to the background a bin of the averaged envelope spectrum of noise exceeds with
/// probability `pfa`, the background being the mean of `background` such bins (0: known exactly);
/// `independent` receives the number of independent spectra the average stands for.
double demon_threshold(double pfa, std::int64_t n_spectra, double correlation, std::int64_t background = 0,
                       double* independent = nullptr);

class TonalAnalyzer {
 public:
  /// `f_low`, `f_high`: the passive band, used for DEMON when its own band is not set.
  explicit TonalAnalyzer(const TonalOptions& options, double f_low = 0.0, double f_high = 0.0);
  /// Results completed by this record's samples, in time order.
  std::vector<TonalEvent> push(const PingRecord& record);
  const TonalOptions& options() const { return opt_; }
  // The latest DEMONgram: dB above the background, [n_bearing][n_frequency] row major.
  std::vector<double> bearings, gram_frequencies;
  std::vector<float> gram_db;
  /// Results not yet written into a passive frame (passive_tonal_json).
  std::vector<TonalEvent> pending;

 private:
  struct LineState {
    double nominal = 0.0, frequency = 0.0, rate = 0.0;
    bool tracking = false;
    std::int64_t misses = 0;
  };
  void setup(const PingRecord& record);
  TonalEvent goertzel_block(std::int64_t first, std::int64_t end_index);
  void demon_blocks(std::int64_t n_blocks, std::vector<TonalEvent>& events);
  TonalEvent demon_result(std::int64_t end_index);
  std::int64_t time_ns(std::int64_t index) const;

  TonalOptions opt_;
  double band_low_ = 0.0, band_high_ = 0.0;
  // the stream
  bool have_key_ = false, is_complex_ = false;
  double fs_ = 0.0, carrier_ = 0.0, c_ = 0.0, key_center_ = 0.0;
  std::int64_t n_channels_ = 0, first_ = 0, run_ns_ = 0, next_ns_ = 0;
  std::vector<std::vector<float>> re_, im_;        ///< [channel][buffered samples]
  // the array
  std::vector<std::vector<std::int64_t>> rows_;
  std::vector<double> u_, weights_;
  // Goertzel
  std::vector<LineState> lines_;
  std::int64_t g_n_ = 0, g_hop_ = 0, g_next_ = 0;
  // DEMON
  std::int64_t d_hop_ = 0, d_n_ = 0, d_next_ = 0, s_n_ = 0, s_hop_ = 0, k_lo_ = 0, k_hi_ = 0;
  double d_rate_ = 0.0, s_df_ = 0.0, s_correlation_ = 0.0;
  std::vector<std::int64_t> d_bins_;
  std::vector<double> d_window_, s_window_;
  std::vector<double> d_steer_re_, d_steer_im_;    ///< [bearing][channel][bin], weights included
  std::vector<double> env_;                        ///< [envelope sample][bearing]
  std::deque<std::vector<double>> spectra_;        ///< each [frequency][bearing]
};

/// Events as a JSON array, every line and source in full (what core.TonalAnalyzerCpp.push returns).
std::string tonal_events_json(const std::vector<TonalEvent>& events, const TonalAnalyzer& analyzer);

struct PassiveFrame;
/// The passive report lines of one record (passive_json) with each tonal result put into the
/// frame during which it was completed, as "tonal": {"goertzel": [...], "demon": [...]}; every
/// line and source carries "peak", the index of the frame's bearing it lies within `attach_deg`
/// of, or -1. Frames without a result are exactly passive_json's. `span` is the frame length in
/// samples (n_fft + (n_average - 1) hop).
std::string passive_tonal_json(const PingRecord& record, const std::vector<PassiveFrame>& frames, std::int64_t span,
                               TonalAnalyzer& analyzer);

}  // namespace sonar::live
