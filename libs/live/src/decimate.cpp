#include "sonar/live/decimate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "sonar/core/parallel.hpp"
#include "sonar/live/environment.hpp"

namespace sonar::live {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Modified Bessel function of the first kind, order zero: its power series.
double bessel_i0(double x) {
  double sum = 1.0, term = 1.0;
  for (int k = 1; k < 200; ++k) {
    term *= (0.5 * x / k) * (0.5 * x / k);
    sum += term;
    if (term < 1e-18 * sum) break;
  }
  return sum;
}

}  // namespace

DecimatePlan plan_decimation(double fs, double f_low, double f_high, const DecimateOptions& o) {
  if (!(fs > 0.0) || !(f_high > f_low) || !(f_low >= 0.0)) throw std::invalid_argument("decimate: needs a band f_low < f_high and a sample rate");
  DecimatePlan p;
  p.input_rate = fs;
  p.carrier = 0.5 * (f_low + f_high);
  const double kept = o.band_margin * (f_high - f_low);
  p.passband = 0.5 * kept;
  const double wanted = o.output_rate > 0.0 ? o.output_rate : o.oversample * kept;
  if (!(wanted > kept)) throw std::invalid_argument("decimate: the output rate must exceed the band kept (band_margin x the band)");
  p.factor = std::max<std::int64_t>(static_cast<std::int64_t>(std::floor(fs / wanted + 1e-9)), 1);
  p.output_rate = fs / static_cast<double>(p.factor);
  // Whatever lies further than this from the carrier folds onto the kept band at the output rate.
  p.stopband = p.output_rate - p.passband;
  if (p.carrier + p.passband >= 0.5 * fs) throw std::invalid_argument("decimate: the band does not fit under half the input rate");
  // The real input's image sits 2 x carrier below the band: it must be in the stop band too
  // (it is for any band that is not at the very bottom of the spectrum).
  if (2.0 * p.carrier - p.passband < p.stopband) {
    throw std::invalid_argument("decimate: the band is too close to zero frequency for one real-to-baseband filter at this output rate");
  }
  // Kaiser-window low-pass. The window's shape parameter and the length for a wanted ripple and
  // transition width are Kaiser's empirical formulas as scipy.signal states them (kaiser_beta,
  // kaiser_atten: a = 2.285 (N - 1) pi width + 7.95 with the width as a fraction of the Nyquist
  // frequency; scipy/signal/_fir_filter_design.py, scipy 1.18.1, citing Oppenheim and Schafer,
  // "Discrete-Time Signal Processing", pp. 475-476). tests/python/test_decimate.py measures the
  // ripple and rejection of the result against what was asked.
  const double ripple = std::pow(10.0, o.ripple_db / 20.0);
  const double delta_pass = (ripple - 1.0) / (ripple + 1.0);
  const double a = std::max(o.stopband_db, -20.0 * std::log10(std::max(delta_pass, 1e-12)));
  const double beta = a > 50.0 ? 0.1102 * (a - 8.7) : a > 21.0 ? 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0) : 0.0;
  const double width = (p.stopband - p.passband) / (0.5 * fs);
  std::int64_t n = o.taps > 0 ? o.taps : static_cast<std::int64_t>(std::ceil((a - 7.95) / (2.285 * kPi * width))) + 1;
  if (n % 2 == 0) ++n;
  if (n < 3) n = 3;
  const double cutoff = 0.5 * (p.passband + p.stopband) / fs;   // cycles per sample: the middle of the transition
  const auto half = static_cast<double>(n - 1) / 2.0;
  p.taps.resize(static_cast<std::size_t>(n));
  double sum = 0.0;
  for (std::int64_t k = 0; k < n; ++k) {
    const double x = static_cast<double>(k) - half;
    const double sinc = x == 0.0 ? 2.0 * cutoff : std::sin(2.0 * kPi * cutoff * x) / (kPi * x);
    const double r = x / half;
    p.taps[static_cast<std::size_t>(k)] = sinc * bessel_i0(beta * std::sqrt(std::max(1.0 - r * r, 0.0))) / bessel_i0(beta);
    sum += p.taps[static_cast<std::size_t>(k)];
  }
  for (double& t : p.taps) t /= sum;
  return p;
}

DecimatePlan plan_decimation(const PingRecord& record, const DecimateOptions& o) {
  const Waveform& w = record.transmit.waveform;
  double lo = o.f_low, hi = o.f_high;
  if (lo <= 0.0 && hi <= 0.0) {
    lo = std::min(w.f_start, w.f_stop);
    hi = std::max(w.f_start, w.f_stop);
    if (!(hi > lo)) {                    // a tone: the band the pulse length gives, as dsp::to_baseband takes it
      const double band = 4.0 / w.duration / o.band_margin;
      lo -= 0.5 * band;
      hi += 0.5 * band;
    }
  }
  return plan_decimation(record.samples.sample_rate, lo, hi, o);
}

StreamDecimator::StreamDecimator(const DecimatePlan& plan, std::int64_t n_channels) : plan_(plan), n_channels_(n_channels) {
  n_taps_ = static_cast<std::int64_t>(plan_.taps.size());
  half_ = (n_taps_ - 1) / 2;
  coeff_.resize(static_cast<std::size_t>(n_taps_));
  // history[k] holds input sample (centre - half + k): its tap is taps[n - 1 - k] (symmetric), its
  // mixer phase -2 pi f_c (k - half) / fs relative to the centre. Twice: the analytic half of a real signal.
  for (std::int64_t k = 0; k < n_taps_; ++k) {
    const double ph = -2.0 * kPi * plan_.carrier * static_cast<double>(k - half_) / plan_.input_rate;
    const double h = 2.0 * plan_.taps[static_cast<std::size_t>(n_taps_ - 1 - k)];
    coeff_[static_cast<std::size_t>(k)] = {static_cast<float>(h * std::cos(ph)), static_cast<float>(h * std::sin(ph))};
  }
  history_.assign(static_cast<std::size_t>(n_channels_ * 2 * n_taps_), 0.0F);
}

void StreamDecimator::open(double t0) {
  t0_ = t0;
  count_ = next_ = pos_ = 0;
  std::fill(history_.begin(), history_.end(), 0.0F);
}

void StreamDecimator::emit(std::vector<std::vector<std::complex<float>>>& out) {
  // The mixer's phase at the centre sample, in double: f_c t over a 17 s window is 1e5 cycles.
  const double cycles = plan_.carrier * (t0_ + static_cast<double>(next_ * plan_.factor) / plan_.input_rate);
  const double ph = -2.0 * kPi * (cycles - std::floor(cycles));
  const std::complex<float> rot(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
  const std::complex<float>* g = coeff_.data();
  const auto n = static_cast<std::size_t>(n_taps_);
  for (std::int64_t c = 0; c < n_channels_; ++c) {
    const float* x = history_.data() + static_cast<std::size_t>(c * 2 * n_taps_ + pos_);   // the last n_taps, oldest first
    float re = 0.0F, im = 0.0F;
    for (std::size_t k = 0; k < n; ++k) {
      re += x[k] * g[k].real();
      im += x[k] * g[k].imag();
    }
    out[static_cast<std::size_t>(c)].push_back(std::complex<float>(re, im) * rot);
  }
  ++next_;
}

void StreamDecimator::feed(const float* rows, std::int64_t n, std::vector<std::vector<std::complex<float>>>& out) {
  if (static_cast<std::int64_t>(out.size()) != n_channels_) out.resize(static_cast<std::size_t>(n_channels_));
  for (std::int64_t i = 0; i < n; ++i) {
    const float* row = rows ? rows + i * n_channels_ : nullptr;
    for (std::int64_t c = 0; c < n_channels_; ++c) {
      float* h = history_.data() + static_cast<std::size_t>(c * 2 * n_taps_);
      h[pos_] = h[pos_ + n_taps_] = row ? row[c] : 0.0F;
    }
    pos_ = (pos_ + 1) % n_taps_;
    ++count_;
    if (count_ - 1 == next_ * plan_.factor + half_) emit(out);
  }
}

void StreamDecimator::flush(std::vector<std::vector<std::complex<float>>>& out) {
  const std::int64_t fed = count_;
  if (fed <= 0) return;
  const std::int64_t total = (fed - 1) / plan_.factor + 1;
  while (next_ < total) feed(nullptr, 1, out);
  count_ = fed;
}

PingDecimator::PingDecimator(const PingRecord& t, const DecimateOptions& options, const Reference* reference)
    : plan_(plan_decimation(t, options)), stream_(plan_, t.samples.n_channels) {
  if (t.samples.is_complex()) throw std::invalid_argument("decimate: the stream must be real samples at the boards' rate");
  n_channels_ = t.samples.n_channels;
  window_ = t.samples.n_samples;
  start_time_ = t.samples.start_time;
  plan_blank_ = options.blank_until_s;
  plan_taper_ = options.blank_taper_s;
  if (reference != nullptr) {
    sync_ = true;
    reference_ = *reference;
    for (const Board& b : t.metadata.boards) {
      if (b.reference_channel < 0) throw std::invalid_argument("decimate: a board has no reference channel to synchronise on");
      reference_channel_.push_back(b.reference_channel);
    }
    if (reference_channel_.empty()) throw std::invalid_argument("decimate: the record names no boards to synchronise");
    if (reference_.min_quality > 0.0 && !(reference_.clip_level > 0.0) && std::isfinite(t.metadata.adc_full_scale)) {
      reference_.clip_level = t.metadata.adc_full_scale;
    }
  }
  open();
}

void PingDecimator::open() {
  stream_.open(start_time_);
  fed_ = 0;
  out_.assign(static_cast<std::size_t>(n_channels_), {});
  const auto expect = static_cast<std::size_t>((window_ - 1) / plan_.factor + 1);
  for (auto& v : out_) v.reserve(expect);
  if (sync_) bursts_ = burst_windows(reference_, plan_.input_rate, window_, reference_channel_.size());
}

void PingDecimator::feed(const float* rows, std::int64_t n) {
  if (sync_) {
    // The raw stretches around the two reference bursts, as they go by.
    for (int which = 0; which < 2; ++which) {
      const std::int64_t first = which == 0 ? bursts_.first_a : bursts_.first_b;
      auto& kept = which == 0 ? bursts_.a : bursts_.b;
      const std::int64_t lo = std::max(fed_, first), hi = std::min(fed_ + n, first + bursts_.length);
      for (std::int64_t i = lo; i < hi; ++i) {
        for (std::size_t b = 0; b < reference_channel_.size(); ++b) {
          kept[b][static_cast<std::size_t>(i - first)] = rows[(i - fed_) * n_channels_ + reference_channel_[b]];
        }
      }
    }
  }
  if (plan_blank_ > 0.0) {
    // Blanked on the way in: a saturated stretch must not ring through the filter into what follows.
    const double fs = plan_.input_rate;
    const auto last = static_cast<std::int64_t>(std::ceil((plan_blank_ + plan_taper_ - start_time_) * fs)) + 1;
    if (fed_ < last) {
      blanked_.assign(rows, rows + n * n_channels_);
      for (std::int64_t i = 0; i < n && fed_ + i < last; ++i) {
        const double out = start_time_ + static_cast<double>(fed_ + i) / fs - plan_blank_;
        const double gain = out <= 0.0 ? 0.0 : out >= plan_taper_ ? 1.0 : 0.5 - 0.5 * std::cos(kPi * out / plan_taper_);
        for (std::int64_t c = 0; c < n_channels_; ++c) blanked_[static_cast<std::size_t>(i * n_channels_ + c)] *= static_cast<float>(gain);
      }
      rows = blanked_.data();
    }
  }
  stream_.feed(rows, n, out_);
  fed_ += n;
}

namespace {

// The record's samples replaced by complex baseband at the plan's rate.
void store(PingRecord& rec, const DecimatePlan& plan, const std::vector<std::vector<std::complex<float>>>& out, double start_time) {
  Samples& s = rec.samples;
  const auto n_out = static_cast<std::int64_t>(out.empty() ? 0 : out[0].size());
  s.resize(static_cast<std::int64_t>(out.size()), n_out, SampleKind::ComplexBaseband);
  for (std::size_t c = 0; c < out.size(); ++c) std::copy(out[c].begin(), out[c].end(), s.channel_complex(static_cast<std::int64_t>(c)));
  s.sample_rate = plan.output_rate;
  s.center_frequency = plan.carrier;
  s.start_time = start_time;
  s.storage = StorageType::Float32;
  s.storage_scale = 1.0;
}

SyncResult finish_sync(PingRecord& rec, const DecimatePlan& plan, bool sync, const Reference& reference, const BurstWindows& bursts,
                       std::int64_t window, double blanked) {
  SyncResult r;
  char note[320];
  char tail[96];
  // What the chain needs to know of this stage: how long the filter rings, and what it blanked.
  std::snprintf(tail, sizeof(tail), ", \"delay_s\": %.6g, \"blanked_s\": %.9g}", plan.delay_s(), blanked);
  if (sync) {
    const Waveform& w = rec.transmit.waveform;
    r = estimate_from_bursts(bursts, reference, plan.input_rate, window, std::max(w.f_start, w.f_stop));
    correct_baseband(rec, r.offset, r.drift_ppm);
    double worst = 0.0;
    for (std::size_t b = 0; b < r.offset.size() && b < rec.metadata.boards.size(); ++b) {
      rec.metadata.boards[b].clock_offset = r.offset[b];
      rec.metadata.boards[b].clock_drift_ppm = r.drift_ppm[b];
      rec.metadata.boards[b].synchronised = 1;
      worst = std::max(worst, r.residual[b]);
    }
    if (reference.dedicated) {
      std::vector<std::int64_t> refs;
      for (const Board& b : rec.metadata.boards) {
        if (b.reference_channel >= 0) refs.push_back(b.reference_channel);
      }
      drop_channels(rec, refs);
    }
    std::string flags;
    for (std::size_t b = 0; b < r.flagged.size(); ++b) flags += (b ? ", " : "") + std::to_string(r.flagged[b]);
    std::snprintf(note, sizeof(note), "{\"factor\": %lld, \"sync_ok\": %d, \"sync_residual_us\": %.6g, \"sync_tolerance_us\": %.6g, \"sync_flagged\": [[%s]]",
                  static_cast<long long>(plan.factor), r.ok ? 1 : 0, worst * 1e6, r.tolerance * 1e6, flags.c_str());
  } else {
    std::snprintf(note, sizeof(note), "{\"factor\": %lld", static_cast<long long>(plan.factor));
  }
  extra_set(rec.metadata.extra_json, "decimated", std::string(note) + tail);
  return r;
}

}  // namespace

SyncResult PingDecimator::finish(PingRecord& like) {
  stream_.flush(out_);
  store(like, plan_, out_, start_time_);
  return finish_sync(like, plan_, sync_, reference_, bursts_, window_, plan_blank_ > 0.0 ? plan_blank_ + plan_taper_ : 0.0);
}

SyncResult decimate_record(PingRecord& record, const DecimateOptions& options, const Reference* reference) {
  Samples& s = record.samples;
  if (s.is_complex()) return {};
  const DecimatePlan plan = plan_decimation(record, options);
  const std::int64_t n = s.n_samples, n_ch = s.n_channels;
  BurstWindows bursts;
  if (reference != nullptr) {
    bursts = burst_windows(*reference, plan.input_rate, n, record.metadata.boards.size());
    for (std::size_t b = 0; b < record.metadata.boards.size(); ++b) {
      const std::int32_t row = record.metadata.boards[b].reference_channel;
      if (row < 0) throw std::invalid_argument("decimate: a board has no reference channel to synchronise on");
      const float* x = s.channel(row);
      for (int which = 0; which < 2; ++which) {
        const std::int64_t first = which == 0 ? bursts.first_a : bursts.first_b;
        auto& kept = which == 0 ? bursts.a[b] : bursts.b[b];
        for (std::int64_t i = std::max<std::int64_t>(first, 0); i < std::min(first + bursts.length, n); ++i) kept[static_cast<std::size_t>(i - first)] = x[i];
      }
    }
  }
  Reference ref;
  if (reference != nullptr) {
    ref = *reference;
    if (ref.min_quality > 0.0 && !(ref.clip_level > 0.0) && std::isfinite(record.metadata.adc_full_scale)) ref.clip_level = record.metadata.adc_full_scale;
  }
  if (options.blank_until_s > 0.0) blank_interval(record, -1e300, options.blank_until_s, options.blank_taper_s);
  // Channel by channel through the same filter as the streaming path (one decimator per channel).
  std::vector<std::vector<std::complex<float>>> out(static_cast<std::size_t>(n_ch));
  const double start = s.start_time;
  parallel_for_host(n_ch, [&](std::int64_t c) {
    StreamDecimator one(plan, 1);
    one.open(start);
    std::vector<std::vector<std::complex<float>>> y(1);
    y[0].reserve(static_cast<std::size_t>((n - 1) / plan.factor + 1));
    one.feed(s.channel(c), n, y);
    one.flush(y);
    out[static_cast<std::size_t>(c)] = std::move(y[0]);
  });
  store(record, plan, out, start);
  return finish_sync(record, plan, reference != nullptr, ref, bursts, n, options.blank_until_s > 0.0 ? options.blank_until_s + options.blank_taper_s : 0.0);
}

void correct_baseband(PingRecord& record, const std::vector<double>& offset, const std::vector<double>& drift_ppm) {
  Samples& s = record.samples;
  if (!s.is_complex()) throw std::invalid_argument("correct_baseband: needs complex baseband samples");
  const std::int64_t n = s.n_samples;
  const double fs = s.sample_rate, fc = s.center_frequency;
  constexpr int taps = 16, half = taps / 2;
  parallel_for_host(s.n_channels, [&](std::int64_t c) {
    const auto b = static_cast<std::size_t>(record.metadata.channel_board[static_cast<std::size_t>(c)]);
    const double o = offset[b], d = drift_ppm[b] * 1e-6;
    if (o == 0.0 && d == 0.0) return;
    std::complex<float>* x = s.channel_complex(c);
    std::vector<std::complex<float>> y(static_cast<std::size_t>(n));
    for (std::int64_t m = 0; m < n; ++m) {
      // Nominal sample m is wanted at time m / fs after the window opens; this board holds that instant at
      // index `pos`, mixed with the phase of its own nominal time pos / fs: (m - pos) / fs too early.
      const double pos = (static_cast<double>(m) / fs - o) * fs * (1.0 + d);
      const auto base = static_cast<std::int64_t>(std::floor(pos));
      const double frac = pos - static_cast<double>(base);
      std::complex<double> acc = 0.0;
      for (int k = -half + 1; k <= half; ++k) {
        const std::int64_t idx = base + k;
        if (idx < 0 || idx >= n) continue;
        const double q = static_cast<double>(k) - frac;
        const double sinc = std::abs(q) < 1e-12 ? 1.0 : std::sin(kPi * q) / (kPi * q);
        acc += sinc * (0.5 + 0.5 * std::cos(kPi * q / half)) * std::complex<double>(x[idx]);
      }
      const double cycles = fc * (static_cast<double>(m) - pos) / fs;
      const double ph = -2.0 * kPi * (cycles - std::floor(cycles));
      y[static_cast<std::size_t>(m)] = std::complex<float>(acc * std::complex<double>(std::cos(ph), std::sin(ph)));
    }
    std::copy(y.begin(), y.end(), x);
  });
}

bool decimated_sync_facts(const PingRecord& record, SyncResult& facts) {
  const std::string& extra = record.metadata.extra_json;
  if (extra.find("\"sync_residual_us\"") == std::string::npos) return false;
  facts = {};
  facts.ok = extra_number(extra, "sync_ok", 0.0) != 0.0;
  for (const auto& row : extra_rows(extra, "sync_flagged", record.metadata.boards.size())) {
    for (double v : row) facts.flagged.push_back(static_cast<int>(v));
  }
  facts.tolerance = extra_number(extra, "sync_tolerance_us", 0.0) * 1e-6;
  for (const Board& b : record.metadata.boards) {
    if (!b.synchronised) return false;
    facts.offset.push_back(b.clock_offset);
    facts.drift_ppm.push_back(b.clock_drift_ppm);
    facts.residual.push_back(extra_number(extra, "sync_residual_us", 0.0) * 1e-6);
  }
  return !facts.offset.empty();
}

}  // namespace sonar::live
