// Decimate first: the boards' real stream band-selected and decimated to complex baseband
// before anything else looks at it, so that every later stage (and, if wanted, the recording)
// handles an integer factor fewer samples.
//
//   real samples at the board rate
//     -> [reference inputs: the raw stretches around the two reference bursts are kept aside]
//     -> mix to the centre of the band, low-pass (one linear-phase FIR), keep every D-th sample
//     -> at the end of the ping: board clocks estimated from the kept stretches (sync.hpp) and
//        the baseband samples of every board resampled onto the first board's timeline
//     -> a complex-baseband record, marked as synchronised
//
// Order of operations. The reference chirp lies above the acoustic band, so the filter removes
// it: the boards cannot be synchronised after decimation from the decimated data. Both bursts
// (the second closes the window) are needed for offset and drift, so they cannot be corrected
// before decimation either without holding the whole ping at the full rate. The reference is
// therefore carried through: only the raw samples around its two bursts are kept (a few
// milliseconds per board), and the correction is applied to the baseband samples, where a
// delay tau is a resampling by tau and a turn of the carrier phase by 2 pi f_c tau.
//
// What the chain does itself (dsp::to_baseband) is the same conversion, once per ping on the
// whole block by FFT. This stage does it once, early and streaming; a record that arrives
// complex is not converted again (live::prepare skips it).
//
// Nothing here is specific to a sonar: the band defaults to the template record's transmit
// band, the output rate to the same rule dsp::to_baseband uses, and the input rate is the
// record's.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "sonar/core/record.hpp"
#include "sonar/live/sync.hpp"

namespace sonar::live {

struct DecimateOptions {
  bool enabled = false;
  double f_low = 0.0, f_high = 0.0;  ///< Hz, the band of interest; both 0: the record's transmit band
  double band_margin = 1.5;          ///< the band kept flat is this x (f_high - f_low) about its centre (as dsp::to_baseband)
  double output_rate = 0.0;          ///< Hz wanted; 0: oversample x the kept band. Realised: input rate / integer, never below what is wanted
  double oversample = 2.0;
  double ripple_db = 0.1;            ///< pass-band ripple allowed, dB peak to peak
  double stopband_db = 80.0;         ///< rejection of everything that would alias into the kept band, dB
  std::int64_t taps = 0;             ///< FIR length (odd); 0: from Kaiser's estimate for the ripple, rejection and transition
  /// Blanking (chain.hpp BlankingOptions, mode zero), done here on the raw samples, after the reference
  /// stretches are taken and before the filter: input up to this long after the transmit is zeroed, with
  /// a raised-cosine return over the taper. 0: none.
  double blank_until_s = 0.0, blank_taper_s = 0.0;
};

struct DecimatePlan {
  std::int64_t factor = 1;           ///< D: one output sample per D input samples
  double input_rate = 0.0, output_rate = 0.0;
  double carrier = 0.0;              ///< Hz: the centre of the band, exactly
  double passband = 0.0;             ///< Hz from the carrier: flat to here
  double stopband = 0.0;             ///< Hz from the carrier: rejected from here (output_rate - passband: no alias reaches the kept band)
  std::vector<double> taps;          ///< the low-pass, unit gain at DC, odd length, symmetric
  double delay_s() const { return 0.5 * static_cast<double>(taps.size() - 1) / input_rate; }  ///< what a streaming receiver waits for
};

/// The filter and factor for a band at an input rate. Throws when the band does not fit.
DecimatePlan plan_decimation(double input_rate, double f_low, double f_high, const DecimateOptions& options);
/// The same for a record: the options' band, or the record's transmit band.
DecimatePlan plan_decimation(const PingRecord& record, const DecimateOptions& options);

/// One receive window (or one continuous stretch), sample by sample. Output sample m is the
/// baseband value at input sample m D, phase referenced to the time origin given to open();
/// input outside what was fed is taken as zero (the first and last half filter length of a
/// window are edge-affected, as with any filter).
class StreamDecimator {
 public:
  StreamDecimator(const DecimatePlan& plan, std::int64_t n_channels);
  /// Start a stretch whose first input sample is at `t0` seconds on the phase reference's clock.
  void open(double t0);
  /// `n` consecutive samples, row major [n][n_channels], in volts. Completed output samples are
  /// appended to out[channel].
  void feed(const float* rows, std::int64_t n, std::vector<std::vector<std::complex<float>>>& out);
  /// The outputs still waiting for input that will not come (zeros are fed): after this the
  /// stretch holds floor((fed - 1) / D) + 1 samples per channel.
  void flush(std::vector<std::vector<std::complex<float>>>& out);
  std::int64_t fed() const { return count_; }

 private:
  void emit(std::vector<std::vector<std::complex<float>>>& out);
  DecimatePlan plan_;
  std::int64_t n_channels_ = 0, n_taps_ = 0, half_ = 0, count_ = 0, next_ = 0, pos_ = 0;
  double t0_ = 0.0;
  std::vector<std::complex<float>> coeff_;   ///< taps x the mixer, oldest sample first
  std::vector<float> history_;               ///< [channel][2 n_taps]: every sample written twice, so the last n_taps are contiguous
};

/// A whole ping: decimation of every channel and, with a reference, the boards' clocks.
class PingDecimator {
 public:
  /// `template_record` describes the raw stream (rate, channels, window, boards). `reference`:
  /// the shared reference signal when the boards run on their own clocks, else null.
  PingDecimator(const PingRecord& template_record, const DecimateOptions& options, const Reference* reference = nullptr);
  const DecimatePlan& plan() const { return plan_; }
  void open();
  /// `n` consecutive samples of the window, row major [n][n_channels], volts.
  void feed(const float* rows, std::int64_t n);
  /// The complex-baseband record of what was fed (geometry, waveform and metadata from `like`,
  /// which is usually the template with this ping's sequence and time): boards synchronised,
  /// dedicated reference inputs removed. The result has the facts the chain reports.
  SyncResult finish(PingRecord& like);

 private:
  DecimatePlan plan_;
  StreamDecimator stream_;
  bool sync_ = false;
  Reference reference_;
  std::int64_t n_channels_ = 0, window_ = 0, fed_ = 0;
  double start_time_ = 0.0, plan_blank_ = 0.0, plan_taper_ = 0.0;
  std::vector<float> blanked_;
  std::vector<std::int32_t> reference_channel_;   ///< per board
  BurstWindows bursts_;
  std::vector<std::vector<std::complex<float>>> out_;
};

/// A raw record converted in place by the same stage (for a recording replayed, or when the raw
/// ping is recorded first and decimated after). A complex record is left as it is.
SyncResult decimate_record(PingRecord& record, const DecimateOptions& options, const Reference* reference = nullptr);

/// Baseband samples of every board moved onto the first board's timeline, in place: channel
/// samples resampled by the board's offset and drift and turned by the carrier phase the
/// nominal-time mixer got wrong.
void correct_baseband(PingRecord& record, const std::vector<double>& offset, const std::vector<double>& drift_ppm);

/// What decimate_record / PingDecimator left in a record for the chain to report: true when
/// the record says its boards were synchronised by this stage.
bool decimated_sync_facts(const PingRecord& record, SyncResult& facts);

}  // namespace sonar::live
