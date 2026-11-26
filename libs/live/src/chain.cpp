#include "sonar/live/chain.hpp"

#include "sonar/live/raytrace.hpp"
#include "sonar/core/platform.hpp"

#include <utility>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/detect/cfar.hpp"
#include "sonar/live/detector.hpp"
#include "sonar/dsp/baseband.hpp"
#include "sonar/dsp/pulse_compression.hpp"
#include "sonar/recon/backproject.hpp"

namespace sonar::live {
namespace {

constexpr double kDeg = 3.14159265358979323846 / 180.0;

std::vector<double> grid(double first_deg, double last_deg, double step_deg) {
  std::vector<double> g;
  for (double b = first_deg; b <= last_deg + 0.5 * step_deg; b += step_deg) g.push_back(b * kDeg);
  return g;
}

struct Frame {
  Vec3d origin, forward, across;
  std::vector<Vec3d> positions;
};

Frame array_frame(const PingRecord& rec) {
  Frame f;
  const auto n = rec.geometry.rx.size();
  f.origin = Vec3d{0.0, 0.0, 0.0};
  for (const Element& e : rec.geometry.rx) {
    f.positions.push_back(world_position(rec.geometry.rx_pose, e.lever_arm));
    f.origin = f.origin + f.positions.back() / static_cast<double>(n);
  }
  f.forward = world_direction(rec.geometry.rx_pose, rec.geometry.rx[0].boresight);
  Vec3d a = f.positions[n - 1] - f.positions[0];
  a = a - f.forward * dot(a, f.forward);
  f.across = normalized(a);
  return f;
}

void num(std::ostringstream& s, double v) {
  if (std::isfinite(v)) s << v; else s << "null";
}

}  // namespace

std::vector<float> range_bearing_power(const PingRecord& record, const std::vector<double>& ranges,
                                       const std::vector<double>& bearings, bool hann, Backend backend, int device);

namespace {

// Cell-averaged power over bearing x range for one line of hydrophones, by the beamformer chosen.
// `fine_out` (when not null) receives the map before the averaging, [n_bearing][n_cells * cell_samples], and its ranges.
void line_map(const ChainOptions& opt_, const std::vector<double>& bearings, const PingRecord& record,
              std::vector<float>& last_power, std::vector<double>& last_ranges, std::int64_t& n_cells, std::int64_t& cell_samples,
              std::vector<float>* fine_out = nullptr, std::vector<double>* fine_ranges_out = nullptr, double grid_start = kDerive) {
  const double c = record.environment.sound_speed;
  const Waveform& w = record.transmit.waveform;
  const Samples& s = record.samples;
  const double band = std::abs(w.f_stop - w.f_start) > 0.0 ? std::abs(w.f_stop - w.f_start) : 1.0 / w.duration;
  const auto n_b = static_cast<std::int64_t>(bearings.size());
  if (opt_.beamformer == "backprojection") {
    // The range grid from the record's window, or from `grid_start` (the Doppler bank: every hypothesis on the grid of the
    // zero one, its own samples read later by its coupling shift).
    const double t0 = is_given(grid_start) ? grid_start : s.start_time;
    const double r0 = 0.5 * c * t0;
    const double r1 = 0.5 * c * (t0 + static_cast<double>(s.n_samples - 1) / s.sample_rate - w.duration);
    const auto k = std::max<std::int64_t>(static_cast<std::int64_t>(std::ceil(opt_.range_cell_m / (0.75 * c / (2.0 * band)))), 1);
    const double step = opt_.range_cell_m / static_cast<double>(k);
    cell_samples = k;
    n_cells = static_cast<std::int64_t>((r1 - r0) / step) / k;
    if (n_cells < 3) throw std::invalid_argument("LiveChain: the receive window is shorter than three range cells");
    std::vector<double> fine(static_cast<std::size_t>(n_cells * k));
    for (std::size_t i = 0; i < fine.size(); ++i) fine[i] = r0 + step * static_cast<double>(i);
    const std::vector<float> fine_power = range_bearing_power(record, fine, bearings, opt_.hann, opt_.backend, opt_.device);
    last_power.assign(static_cast<std::size_t>(n_b * n_cells), 0.0F);
    last_ranges.assign(static_cast<std::size_t>(n_cells), 0.0);
    for (std::int64_t j = 0; j < n_cells; ++j) {
      double r = 0.0;
      for (std::int64_t q = 0; q < k; ++q) r += fine[static_cast<std::size_t>(j * k + q)];
      last_ranges[static_cast<std::size_t>(j)] = r / static_cast<double>(k);
      for (std::int64_t b = 0; b < n_b; ++b) {
        double p = 0.0, top = 0.0;
        for (std::int64_t q = 0; q < k; ++q) {
          const auto v = static_cast<double>(fine_power[static_cast<std::size_t>(b * n_cells * k + j * k + q)]);
          p += v;
          top = std::max(top, v);
        }
        last_power[static_cast<std::size_t>(b * n_cells + j)] = static_cast<float>(opt_.cell_peak ? top : p / static_cast<double>(k));
      }
    }
    if (fine_out != nullptr) {
      *fine_out = fine_power;
      *fine_ranges_out = fine;
    }
  } else {
    // The far-field beamformers give every sample of the record: the part with a fully
    // overlapped replica is averaged into range cells of whole samples.
    const BeamMap map = opt_.beamformer == "mvdr"
                            ? mvdr_beamform(record, bearings, opt_.mvdr, opt_.backend, opt_.device)
                            : farfield_beamform(record, bearings, opt_.hann ? "hann" : "rect", opt_.backend, opt_.device);
    const double sample_m = map.ranges[1] - map.ranges[0];
    const auto k = std::max<std::int64_t>(std::llround(opt_.range_cell_m / sample_m), 1);
    cell_samples = k;
    n_cells = std::min(map.n_valid, map.n_samples) / k;
    if (n_cells < 3) throw std::invalid_argument("LiveChain: the receive window is shorter than three range cells");
    last_power.assign(static_cast<std::size_t>(n_b * n_cells), 0.0F);
    last_ranges.assign(static_cast<std::size_t>(n_cells), 0.0);
    for (std::int64_t j = 0; j < n_cells; ++j) {
      double r = 0.0;
      for (std::int64_t q = 0; q < k; ++q) r += map.ranges[static_cast<std::size_t>(j * k + q)];
      last_ranges[static_cast<std::size_t>(j)] = r / static_cast<double>(k);
    }
    if (fine_out != nullptr) {
      fine_out->assign(static_cast<std::size_t>(n_b * n_cells * k), 0.0F);
      fine_ranges_out->assign(map.ranges.begin(), map.ranges.begin() + n_cells * k);
    }
    parallel_for_host(n_b, [&](std::int64_t b) {
      const std::complex<float>* row = map.row(b);
      for (std::int64_t j = 0; j < n_cells; ++j) {
        double p = 0.0, top = 0.0;
        for (std::int64_t q = 0; q < k; ++q) {
          const auto v = static_cast<double>(std::norm(row[j * k + q]));
          if (fine_out != nullptr) (*fine_out)[static_cast<std::size_t>(b * n_cells * k + j * k + q)] = static_cast<float>(v);
          p += v;
          top = std::max(top, v);
        }
        last_power[static_cast<std::size_t>(b * n_cells + j)] = static_cast<float>(opt_.cell_peak ? top : p / static_cast<double>(k));
      }
    });
  }
}

}  // namespace

void prepare(PingRecord& record, Backend backend, int device) {
  if (!record.samples.is_complex()) dsp::to_baseband(record, {}, backend, device);
  // The Python stages mark a compressed record in metadata.extra_json; honour and keep that mark.
  std::string& extra = record.metadata.extra_json;
  if (extra.find("\"compressed\": true") != std::string::npos) return;
  dsp::pulse_compress(record, {}, backend, device);
  const std::string mark = "\"gpusonar\": {\"compressed\": true, \"replica_window\": \"none\"}";
  const auto brace = extra.find('{');
  if (extra.empty() || brace == std::string::npos) {
    extra = "{" + mark + "}";
  } else if (extra.find("\"gpusonar\"") == std::string::npos) {
    const bool empty_object = extra.find_first_not_of(" \t\n", brace + 1) == extra.find('}');
    extra.insert(brace + 1, mark + (empty_object ? "" : ", "));
  } else {
    const auto g = extra.find('{', extra.find("\"gpusonar\""));
    extra.insert(g + 1, "\"compressed\": true, ");
  }
}

namespace {
std::vector<std::complex<float>> range_bearing_image(const PingRecord& record, const std::vector<double>& ranges,
                                                     const std::vector<double>& bearings, bool hann, Backend backend, int device);
}  // namespace

std::vector<float> range_bearing_power(const PingRecord& record, const std::vector<double>& ranges,
                                       const std::vector<double>& bearings, bool hann, Backend backend, int device) {
  const std::vector<std::complex<float>> image = range_bearing_image(record, ranges, bearings, hann, backend, device);
  std::vector<float> power(image.size());
  for (std::size_t i = 0; i < power.size(); ++i) power[i] = std::norm(image[i]);
  return power;
}

std::vector<std::complex<float>> beam_samples(const PingRecord& record, const std::vector<double>& ranges, double bearing, bool hann,
                                              Backend backend, int device) {
  return range_bearing_image(record, ranges, {bearing}, hann, backend, device);
}

namespace {
std::vector<std::complex<float>> range_bearing_image(const PingRecord& record, const std::vector<double>& ranges,
                                                     const std::vector<double>& bearings, bool hann, Backend backend, int device) {
  const Frame f = array_frame(record);
  const auto n = f.positions.size();
  std::vector<std::vector<float>> weights(1, std::vector<float>(n, 1.0F));
  if (hann) {
    // The window is laid across the aperture and reaches zero half a pitch beyond its ends.
    std::vector<double> u(n);
    double lo = 1e300, hi = -1e300;
    for (std::size_t i = 0; i < n; ++i) {
      u[i] = dot(f.positions[i] - f.origin, normalized(f.positions[n - 1] - f.positions[0]));
      lo = std::min(lo, u[i]);
      hi = std::max(hi, u[i]);
    }
    const double span = hi - lo, pitch = span / static_cast<double>(std::max<std::size_t>(n - 1, 1));
    for (std::size_t i = 0; i < n; ++i) {
      const double c = std::cos(3.14159265358979323846 * u[i] / (span + pitch));
      weights[0][i] = static_cast<float>(c * c);
    }
  }
  std::vector<Vec3d> points;
  points.reserve(ranges.size() * bearings.size());
  for (double b : bearings) {          // bearing major: the detector wants rows of constant bearing
    const Vec3d dir = f.forward * std::cos(b) + f.across * std::sin(b);
    for (double r : ranges) points.push_back(f.origin + dir * r);
  }
  const std::vector<const PingRecord*> pings{&record};
  const recon::BpInputs inputs = recon::make_inputs(pings, &weights);
  recon::BpOptions o;
  o.interp = recon::Interp::Linear;
  o.device = device;
  std::vector<std::complex<float>> image(points.size());
  recon::backproject(backend, inputs, points, o, image.data());
  return image;
}
}  // namespace

RangeDoppler parse_range_doppler(const std::string& name) {
  if (name == "off") return RangeDoppler::Off;
  if (name == "ownship") return RangeDoppler::Ownship;
  if (name == "full") return RangeDoppler::Full;
  throw std::invalid_argument("range_doppler '" + name + "': one of off, ownship, full");
}

const char* range_doppler_name(RangeDoppler r) {
  return r == RangeDoppler::Full ? "full" : r == RangeDoppler::Ownship ? "ownship" : "off";
}

double lfm_range_doppler_factor(const Waveform& w) {
  const double sweep = w.f_stop - w.f_start;
  if (w.type != WaveformType::Lfm || sweep == 0.0) return 0.0;
  return 0.5 * (w.f_start + w.f_stop) * w.duration / sweep;
}

RangeEstimate chain_range_estimate(const ChainOptions& c) {
  if (c.range_estimate) return *c.range_estimate;
  return c.cell_multiscale ? RangeEstimate::First : RangeEstimate::Centroid;
}

LiveChain::LiveChain(const ChainOptions& options)
    : opt_(options), tracker_(options.tracker), ladder_(options.elevation.profile), multipath_(options.multipath_depth), coverage_(options.coverage) {
  bearings = grid(opt_.bearing_first_deg, opt_.bearing_last_deg, opt_.bearing_step_deg);
  // classify.py `detections` rules: the row rule acts only in the corridor test, which the chain never runs.
  if (!(opt_.classify.row_snr_db < 0.0)) throw std::invalid_argument("LiveChain: classify row_snr_db: the live chain never runs the row (corridor) test, so this rule would do nothing there; it applies only to classify_tracks with linear (the gallery)");
  if (!(std::isfinite(opt_.tracker_ping_interval_s) && opt_.tracker_ping_interval_s >= 0.0)) {
    throw std::invalid_argument("LiveChain: tracker_ping_interval_s: seconds above 0, or 0 (off)");
  }
  if (std::isnan(opt_.classify.min_mover_cells)) throw std::invalid_argument("LiveChain: classify min_mover_cells is NaN");
  if (!(opt_.classify.outlier_alpha >= 0.0 && opt_.classify.outlier_alpha < 1.0)) throw std::invalid_argument("LiveChain: classify outlier_alpha: from 0 (off) to below 1");
  // NaN: derived from the first record (derive.hpp); given, a finite angle, 0 (off) or more.
  if (is_given(opt_.classify.max_fit_rms_bearing) && !(std::isfinite(opt_.classify.max_fit_rms_bearing) && opt_.classify.max_fit_rms_bearing >= 0.0)) {
    throw std::invalid_argument("LiveChain: classify max_fit_rms_bearing: a finite angle, 0 (off) or more");
  }
  // A setting of the multiscale detector only (live.py check_choices refuses it the same way).
  // An explicit `first` without it is refused; unset, the chain takes `first` with multiscale and the centroid
  // otherwise (chain_range_estimate, D-158). The detector's own field is not the chain's setting.
  if (opt_.detector.range_estimate != RangeEstimate::Centroid) {
    throw std::invalid_argument("LiveChain: set the range estimate in ChainOptions::range_estimate (cfar_range_estimate), not detector.range_estimate");
  }
  if (opt_.range_estimate && *opt_.range_estimate != RangeEstimate::Centroid && !opt_.cell_multiscale) {
    throw std::invalid_argument(std::string("LiveChain: cfar_range_estimate ") + range_estimate_name(*opt_.range_estimate) + " is a setting of cell_statistic multiscale");
  }
  if (opt_.doppler.mode != DopplerMode::Off && !opt_.cell_multiscale) {
    throw std::invalid_argument("LiveChain: doppler sweep_bank needs cell_statistic multiscale (its thresholds carry the bank's shift)");
  }
  if (opt_.doppler.tone.enabled && opt_.range_doppler != RangeDoppler::Off) {
    throw std::invalid_argument("LiveChain: doppler tone needs range_doppler off (a tone detection's range has no sweep coupling to correct)");
  }
  if (opt_.doppler.tone.enabled && opt_.elevation.enabled) {
    throw std::invalid_argument("LiveChain: doppler tone is not made for the elevation option yet (a tone detection's coarse range has no elevation search)");
  }
  if (opt_.doppler.mode != DopplerMode::Off && opt_.beamformer != "backprojection") {
    throw std::invalid_argument("LiveChain: doppler sweep_bank needs beamformer backprojection (its hypotheses' maps share one range grid)");
  }
  if (opt_.coverage.enabled && !coverage_site_missing(opt_.coverage).empty()) throw std::invalid_argument("LiveChain: " + coverage_site_missing(opt_.coverage));
  if (opt_.beamformer != "backprojection" && opt_.beamformer != "farfield" && opt_.beamformer != "mvdr") {
    throw std::invalid_argument("LiveChain: beamformer '" + opt_.beamformer + "' (backprojection, farfield or mvdr)");
  }
  // Asking for CUDA means CUDA: never a silent run on the host (auto is the choice that falls back).
  if (opt_.backend == Backend::Cuda && !cuda_available()) {
    throw std::runtime_error("LiveChain: backend cuda, but no CUDA device is usable here (built without CUDA or no device); "
                             "use backend auto to run on the host when there is none");
  }
}

Report LiveChain::process(PingRecord& record) {
  const auto t_start = std::chrono::steady_clock::now();
  Report rep;
  rep.ping = pings_++;
  rep.sequence = record.metadata.sequence;
  rep.time_ns = record.transmit.time_ns;
  double burst_lo = -1.0, burst_hi = -1.0;       // ranges the reference bursts occupy, when they stay in the data
  // Decimate first: band, baseband and board clocks in one early stage (decimate.hpp).
  const BlankingOptions& bl = opt_.blanking;
  const double blank_until = bl.enabled ? bl.until(record) : 0.0;
  if (opt_.decimate.enabled && !record.samples.is_complex()) {
    DecimateOptions d = opt_.decimate;
    if (bl.enabled && bl.zero) {            // blanked inside the stage: after the reference is read, before the filter
      d.blank_until_s = blank_until;
      d.blank_taper_s = bl.taper_s;
    }
    decimate_record(record, d, opt_.sync ? &opt_.reference : nullptr);
  }
  if (opt_.sync) {
    // Boards run on their own clocks: line them up before anything is summed across them
    // (a record the decimating stage has already synchronised says so, and what it found).
    SyncResult r;
    if (!decimated_sync_facts(record, r)) r = synchronise(record, opt_.reference);
    rep.synced = true;
    rep.sync_ok = r.ok;
    for (std::size_t b = 0; b < r.flagged.size(); ++b) {
      if (r.flagged[b] != 0) rep.sync_flagged.push_back(r.flagged[b] == 1 ? static_cast<std::int64_t>(b) : -static_cast<std::int64_t>(b) - 1);
    }
    for (std::size_t b = 0; b < r.offset.size(); ++b) {
      rep.sync_offsets_us.push_back(r.offset[b] * 1e6);
      rep.sync_drift_ppm.push_back(r.drift_ppm[b]);
      rep.sync_worst_residual_us = std::max(rep.sync_worst_residual_us, r.residual[b] * 1e6);
    }
    if (!opt_.reference.dedicated) {
      const Samples& raw = record.samples;
      const double t_close = raw.start_time + static_cast<double>(raw.n_samples) / raw.sample_rate;
      const double cs = record.environment.sound_speed, pulse = record.transmit.waveform.duration;
      burst_lo = 0.5 * cs * (raw.start_time + opt_.reference.lead + opt_.reference.duration);
      burst_hi = 0.5 * cs * (t_close - opt_.reference.tail - pulse - opt_.reference.duration);
    }
  }
  double blank_edge = 0.0;       // s after the transmit: echoes that begin before this are not whole
  if (bl.enabled) {
    // A record the decimating stage blanked says so; its filter spreads the edge by its own delay.
    const std::string& note = record.metadata.extra_json;
    const double done = extra_number(note, "blanked_s", 0.0), ring = record.samples.is_complex() ? extra_number(note, "delay_s", 0.0) : 0.0;
    if (bl.zero && !(done > 0.0)) blank_interval(record, -1e300, blank_until, bl.taper_s);
    blank_edge = blank_until + bl.taper_s + ring;
  }
  if (!opt_.dead_channels.empty()) {
    for (std::int64_t r : opt_.dead_channels) {
      if (r < 0 || r >= record.samples.n_channels) throw std::invalid_argument("LiveChain: dead_channels names a channel the record does not have");
    }
    drop_channels(record, opt_.dead_channels);
  }
  if (opt_.sound_speed > 0.0) record.environment.sound_speed = opt_.sound_speed;
  // D3: a ping whose pulse is the tone alone does not size the chain (its band is 1 / T): the sizes come from the first
  // sweep ping, and a tone-only ping before it is reported but not tracked (live.py the same).
  const bool tone_only_ping = opt_.doppler.tone.enabled && record.transmit.waveform.type != WaveformType::Lfm &&
                              plan_tone(record, bearings, opt_.doppler, opt_.detector.false_alarms_per_ping).tone_only;
  if (!resolved_ && !tone_only_ping) {
    // Every size the configuration left out, from this sonar's band, pulse and array (derive.hpp): once, at the first ping.
    defaults_ = resolve(opt_, record);
    tracker_ = Tracker(opt_.tracker);
    resolved_ = true;
  }
  // The sweep's Doppler bank (doppler.hpp; D4), opt-in: its plan from this record, the baseband with the room its
  // hypotheses need, and the baseband record kept before compression for the other hypotheses.
  const bool tone_on = opt_.doppler.tone.enabled;       // the tone's Doppler filter bank (D3), opt-in
  // With the tone path on, the sweep bank runs on the pings that are sweeps (the other ping of an alternating pair, or a
  // train's tone-only ping, has none to bank); without it a pulse that is not a sweep is refused as before (plan_doppler).
  const bool bank_on = opt_.doppler.mode == DopplerMode::SweepBank && !(tone_on && record.transmit.waveform.type != WaveformType::Lfm);
  doppler_ = DopplerPlan{};
  tone_ = ToneStage{};
  which_.clear();
  PingRecord base;
  TonePlan tone_plan;
  if ((bank_on || tone_on) && record.metadata.extra_json.find("\"compressed\": true") != std::string::npos) {
    throw std::invalid_argument(std::string("LiveChain: doppler ") + (bank_on ? "sweep_bank" : "tone") + " needs the record before pulse compression");
  }
  if (tone_on) tone_plan = plan_tone(record, bearings, opt_.doppler, opt_.detector.false_alarms_per_ping);
  if (bank_on || tone_plan.on) {
    if (bank_on) doppler_ = plan_doppler(record, bearings, opt_.doppler);
    if (!record.samples.is_complex()) {
      dsp::BasebandOptions bo;
      // 0 (the default band) unless the bank's shifts or the tone and its shifts need more
      bo.max_range_rate = std::max(doppler_.baseband_max_range_rate, tone_plan.baseband_max_range_rate);
      dsp::to_baseband(record, bo, opt_.backend, opt_.device);
    }
    base = record;
    // The tone's bank on the baseband record before compression (doppler.hpp tone_stage): its detections join the sweep's below.
    if (tone_plan.on) tone_ = tone_stage(base, bearings, opt_.doppler, opt_.hann, opt_.detector.false_alarms_per_ping, opt_.backend, opt_.device);
  }
  // A ping whose pulse is the tone alone (D3): the sweep path makes no map (a tone's zero-Doppler map is its reverberation
  // at c T / 2); its detections are the tone path's alone (live.py the same).
  const bool tone_only = tone_.plan.on && tone_.plan.tone_only;
  const auto n_b = static_cast<std::int64_t>(bearings.size());
  std::int64_t n_cells = 0;
  std::vector<std::vector<std::int64_t>> rows;
  bool stacked = false;
  const bool multiscale = opt_.cell_multiscale;
  std::vector<float> ratio;
  std::vector<Detection> hits;
  if (!tone_only) {   // --- the sweep path: compression, map, detector, SBL (not re-indented: as before D3) ---
  prepare(record, opt_.backend, opt_.device);

  // Range grid: finer than the range resolution, then averaged into cells.
  const double c = record.environment.sound_speed;
  const Waveform& w = record.transmit.waveform;
  const Samples& s = record.samples;
  const double band = std::abs(w.f_stop - w.f_start) > 0.0 ? std::abs(w.f_stop - w.f_start) : 1.0 / w.duration;
  if (opt_.elevation.enabled) {
    rows = find_rows(record, opt_.elevation.row_tolerance_m, opt_.elevation.min_row_elements);
    if (rows.size() < 2) rows.clear();           // one row: exactly the chain without the option
  }
  stacked = !rows.empty();
  if (multiscale && opt_.cell_peak) throw std::invalid_argument("LiveChain: cell_statistic is one of mean, peak, multiscale");
  const double grid_start = record.samples.start_time;
  const auto map_of = [&](const PingRecord& prepared) {
    if (!stacked) {
      line_map(opt_, bearings, prepared, last_power, last_ranges, n_cells, cell_samples_, multiscale ? &fine_ : nullptr, &fine_ranges_,
               grid_start);
    } else {
      // Each row beamformed on its own, the rows' power added: a row's bearing response does not
      // depend on the elevation the arrival comes from.
      std::vector<float> row_power, row_fine;
      for (std::size_t k = 0; k < rows.size(); ++k) {
        const PingRecord sub = row_record(prepared, rows[k]);
        line_map(opt_, bearings, sub, k == 0 ? last_power : row_power, last_ranges, n_cells, cell_samples_,
                 multiscale ? (k == 0 ? &fine_ : &row_fine) : nullptr, &fine_ranges_, grid_start);
        if (k > 0) {
          for (std::size_t i = 0; i < last_power.size(); ++i) last_power[i] += row_power[i];
          if (multiscale) {
            for (std::size_t i = 0; i < fine_.size(); ++i) fine_[i] += row_fine[i];
          }
        }
      }
    }
  };
  // One hypothesis, the zero one: the map without the bank, the same calls.
  const bool banked = bank_on && !(doppler_.rates.size() == 1 && doppler_.zero == 0);
  const bool normalised = banked && doppler_.combine == DopplerCombine::Cfar;
  std::vector<double> given_level;           // combine cfar: the background the detector tests against (doppler.hpp)
  if (!banked) {
    map_of(record);
  } else if (normalised) {
    // D3, combine cfar: each hypothesis' map normalised by its own background (the detector's estimate on its own cell
    // means, over the cells the detector will keep), the largest normalised value per sample over the beam's hypotheses
    // (fold_normalised); the detector is handed that maximum times the beam's centre hypothesis' background, and that
    // background as its own (live.py the same).
    NormalisedFold fold;
    std::int64_t guard_n = 0, train_n = 0;
    for (std::size_t h = 0; h < doppler_.rates.size(); ++h) {
      if (static_cast<std::int64_t>(h) == doppler_.zero) {
        map_of(record);
      } else {
        PingRecord trial = base;
        dsp::PulseCompressionOptions pc;
        pc.doppler_scale = doppler_.scales[h];
        dsp::pulse_compress(trial, pc, opt_.backend, opt_.device);
        trial.samples.start_time += doppler_.shifts[h];
        map_of(trial);
      }
      if (h == 0) {
        // The cells the detector keeps (blanking, as below) and its window (cfar_window, as below).
        std::int64_t skip = 0;
        if (bl.enabled) {
          const double min_range = 0.5 * c * blank_edge;
          while (skip < n_cells - 3 && last_ranges[static_cast<std::size_t>(skip)] < min_range) ++skip;
        }
        detect::CfarWindow win;
        win.guard_m = opt_.cfar_guard_m;
        win.train_m = opt_.cfar_train_m;
        const detect::CfarOptions cw = detect::cfar_window(last_ranges[1] - last_ranges[0], c / (2.0 * band), win, opt_.cfar_pfa);
        guard_n = cw.guard;
        train_n = cw.train;
        normalised_start(n_b, n_cells, cell_samples_, skip, fold);
      }
      fold_normalised(doppler_, static_cast<std::int64_t>(h), fine_.data(), guard_n, train_n, fold);
    }
    normalised_map(fold, fine_, last_power, given_level);
    which_ = std::move(fold.which);
  } else {
    // Each hypothesis compressed from the same baseband record (the zero one is `record` itself, compressed by
    // prepare with doppler_scale exactly 1) and mapped by the chain's beamformer; each beam keeps the largest power
    // over its own hypotheses (fold_hypothesis). A loop over the existing kernels: on the CUDA backend each
    // compression and each map runs on the device as it does without the bank (cost: hypotheses x one ping's map).
    std::vector<float> best;
    for (std::size_t h = 0; h < doppler_.rates.size(); ++h) {
      if (static_cast<std::int64_t>(h) == doppler_.zero) {
        map_of(record);
      } else {
        PingRecord trial = base;
        dsp::PulseCompressionOptions pc;
        pc.doppler_scale = doppler_.scales[h];
        dsp::pulse_compress(trial, pc, opt_.backend, opt_.device);
        trial.samples.start_time += doppler_.shifts[h];   // the echo where the zero hypothesis shows it (doppler.hpp)
        map_of(trial);
      }
      const std::int64_t stride = n_b > 0 ? static_cast<std::int64_t>(fine_.size()) / n_b : 0;
      if (h == 0) fold_start(n_b, stride, best, which_);
      fold_hypothesis(doppler_, static_cast<std::int64_t>(h), fine_.data(), n_b, stride, best, which_);
    }
    fine_ = std::move(best);
    // The cells of the maximum map: their mean power, accumulated as line_map does.
    const std::int64_t stride = n_b > 0 ? static_cast<std::int64_t>(fine_.size()) / n_b : 0;
    for (std::int64_t b = 0; b < n_b; ++b) {
      for (std::int64_t j = 0; j < n_cells; ++j) {
        double p = 0.0;
        for (std::int64_t q = 0; q < cell_samples_; ++q) p += static_cast<double>(fine_[static_cast<std::size_t>(b * stride + j * cell_samples_ + q)]);
        last_power[static_cast<std::size_t>(b * n_cells + j)] = static_cast<float>(p / static_cast<double>(cell_samples_));
      }
    }
  }
  const std::int64_t fine_stride = n_cells * cell_samples_;      // before any blanked cells are dropped
  std::int64_t fine_offset = 0;
  if (bl.enabled) {
    // The map starts where whole echoes do: the detector neither reports nearer nor trains on the blanked cells.
    rep.blanking_min_range_m = 0.5 * c * blank_edge;
    rep.blanking_zero = bl.zero;
    std::int64_t skip = 0;
    while (skip < n_cells - 3 && last_ranges[static_cast<std::size_t>(skip)] < rep.blanking_min_range_m) ++skip;
    if (skip > 0) {
      std::vector<float> kept(static_cast<std::size_t>(n_b * (n_cells - skip)));
      for (std::int64_t b = 0; b < n_b; ++b) {
        std::copy_n(last_power.begin() + b * n_cells + skip, n_cells - skip, kept.begin() + b * (n_cells - skip));
      }
      last_power = std::move(kept);
      if (normalised) {
        std::vector<double> level_kept(static_cast<std::size_t>(n_b * (n_cells - skip)));
        for (std::int64_t b = 0; b < n_b; ++b) {
          std::copy_n(given_level.begin() + b * n_cells + skip, n_cells - skip, level_kept.begin() + b * (n_cells - skip));
        }
        given_level = std::move(level_kept);
      }
      last_ranges.erase(last_ranges.begin(), last_ranges.begin() + skip);
      n_cells -= skip;
      fine_offset = skip * cell_samples_;
    }
  }
  const double cell = last_ranges[1] - last_ranges[0];
  // The window in cells of this map, by the library's one conversion (detect::cfar_window: rounded up, the guard at
  // least one range resolution): the Python engine calls the same function.
  detect::CfarWindow window;
  window.guard_m = opt_.cfar_guard_m;
  window.train_m = opt_.cfar_train_m;
  const detect::CfarOptions co = detect::cfar_window(cell, c / (2.0 * band), window, opt_.cfar_pfa);
  std::vector<unsigned char> det;
  std::vector<Detection> found;
  if (multiscale) {
    // Each cell on its strongest sample, its mean and the means of several cells, against the thresholds of those
    // statistics; sidelobes of a stronger cell dropped; touching cells one detection (detector.hpp).
    const auto n_rows = static_cast<std::int64_t>(stacked ? rows.size() : 1);
    if (!planned_ || plan_cells_ != n_cells || plan_train_ != co.train || plan_rows_ != n_rows || plan_offset_db_ != doppler_.threshold_db ||
        plan_divisor_ != doppler_.budget_divisor) {
      DetectorOptions detector = opt_.detector;
      detector.false_alarms_per_ping /= doppler_.budget_divisor;   // combine cfar: the union bound over the hypotheses; 1 otherwise
      detector.range_estimate = chain_range_estimate(opt_);
      detector.threshold_offset_db = doppler_.threshold_db;        // the Doppler bank's shift (doppler.hpp); 0 without it
      detector.pedestal_m = doppler_.pedestal_m;                   // and its pedestal along range; 0 without it
      plan_ = plan_detector(record, defaults_.facts, detector, n_cells, cell_samples_, cell / static_cast<double>(cell_samples_), bearings,
                            n_rows, co.guard, co.train, opt_.hann && opt_.beamformer != "mvdr",
                            is_given(opt_.detector.merge_m) ? opt_.detector.merge_m : opt_.tracker.range_sigma, rows);
      planned_ = true;
      plan_cells_ = n_cells;
      plan_train_ = co.train;
      plan_rows_ = n_rows;
      plan_offset_db_ = doppler_.threshold_db;
      plan_divisor_ = doppler_.budget_divisor;
    }
    // The statistic on the chain's backend (the device gives the host's flags: detector.hpp, multiscale_cells).
    found = multiscale_detect(fine_.data(), fine_stride, fine_offset, last_power.data(), n_b, n_cells, fine_ranges_, last_ranges, bearings,
                              co.guard, co.train, plan_, background_, opt_.backend, opt_.device, normalised ? given_level.data() : nullptr);
    if (bank_on) {
      // Each detection's coarse range rate: the hypothesis of the strongest sample of the maximum map along its nearest
      // bearing, over the range it covers (doppler.hpp detection_hypothesis; live.py the same).
      const double half_cell = 0.5 * (last_ranges[1] - last_ranges[0]);
      for (Detection& d : found) {
        const double from = std::isfinite(d.first_range) ? d.first_range : d.range;
        const double lo = from - half_cell, hi = from + (std::isfinite(d.extent) ? d.extent : 0.0) + half_cell;
        std::int64_t h = doppler_.zero;
        if (banked) {
          h = detection_hypothesis(fine_.data(), which_.data(), fine_stride, fine_offset, n_cells * cell_samples_, fine_ranges_, bearings,
                                   d.bearing, lo, hi);
        }
        if (h >= 0) {
          d.range_rate = doppler_.rates[static_cast<std::size_t>(h)];
          d.range_rate_step = doppler_.step;
        }
      }
    }
  } else {
    det.resize(last_power.size());
    ratio.resize(last_power.size());
    detect::cfar(opt_.backend, last_power.data(), n_b, n_cells, co, det.data(), ratio.data(), opt_.device);
    found = cluster(det.data(), last_power.data(), ratio.data(), n_b, n_cells, last_ranges, bearings);
  }
  for (const Detection& h : found) {
    // Both filters judge a detection by its centroid, whichever range it reports (D-158): their limits were set for it,
    // and with the leading edge a detection would pass the far limit up to the echo's spread later than before.
    const double at = std::isfinite(h.centroid_range) ? h.centroid_range : h.range;
    // A reference summed into hydrophone inputs leaves its bursts in the data near both ends of
    // the window: no detections are taken there.
    if (burst_lo >= 0.0 && (at <= burst_lo || at >= burst_hi)) continue;
    if (bl.enabled && at < rep.blanking_min_range_m) continue;
    if (h.snr_db > opt_.min_snr_db + doppler_.threshold_db) hits.push_back(h);   // the bank's shift (0 without it)
  }

  if (opt_.doa == "sbl" && !hits.empty()) {
    // The map says at which ranges something is; SBL says from which bearings, at each of them.
    // SBL's model is one line: the first row of a stacked array, with or without the elevation option
    // (all rows as one line would be more channels than a line has and phases the model does not hold).
    PingRecord first_row;
    const std::vector<std::vector<std::int64_t>> lines =
        stacked ? rows : find_rows(record, opt_.elevation.row_tolerance_m, opt_.elevation.min_row_elements);
    const bool several = lines.size() > 1;
    if (several) first_row = row_record(record, lines[0]);
    const PingRecord& line = several ? first_row : record;
    const std::vector<double> u = element_offsets(line);
    const double length = u.back() - u.front();
    std::int64_t n_fft = opt_.gate_fft;
    if (n_fft <= 0) {
      const double want = s.sample_rate * (2.0 * length / c + 4.0 / band);
      n_fft = 16;
      while (static_cast<double>(n_fft) < want) n_fft *= 2;
    }
    const double gate_m = 0.25 * c * static_cast<double>(n_fft) / s.sample_rate;   // half a gate, in range
    const std::vector<double> sbl_grid = grid(opt_.bearing_first_deg, opt_.bearing_last_deg, opt_.sbl_grid_step_deg);
    // Strongest first; each gate takes every detection within half a gate of its range. Where SBL
    // finds discrete arrivals they replace those detections; where it finds none (a diffuse
    // field: reverberation, an edge the detector tripped on) the map's detections are kept.
    std::vector<Detection> by_snr = hits;
    std::sort(by_snr.begin(), by_snr.end(), [](const Detection& a, const Detection& b) { return a.snr_db > b.snr_db; });
    // Gates are grouped and centred on each detection's centroid, as before the leading edge became the default range
    // (D-158): a gate centred on the leading edge holds less of the echo.
    const auto centre = [](const Detection& d) { return std::isfinite(d.centroid_range) ? d.centroid_range : d.range; };
    std::vector<char> taken(by_snr.size(), 0);
    std::vector<Detection> resolved;
    std::int64_t gates = 0;
    for (std::size_t i = 0; i < by_snr.size(); ++i) {
      if (taken[i]) continue;
      const Detection& h = by_snr[i];
      std::vector<std::size_t> group;
      for (std::size_t j = i; j < by_snr.size(); ++j) {
        if (!taken[j] && std::abs(centre(by_snr[j]) - centre(h)) < gate_m) group.push_back(j);
      }
      for (std::size_t j : group) taken[j] = 1;
      std::vector<DoaPeak> peaks;
      if (gates < opt_.sbl_max_gates) {
        ++gates;
        const DoaFrame frame = gate_frame(line, 2.0 * centre(h) / c, n_fft);
        // The conventional spectrum of the gate costs one pass; SBL is only worth running where
        // that spectrum stands clear of its own median, which a diffuse field's does not.
        DoaOptions solver = opt_.sbl;          // the solver runs where the rest of the chain does
        solver.backend = opt_.backend;
        solver.device = opt_.device;
        const DoaResult look = conventional(frame, u, c, sbl_grid, solver);
        std::vector<float> sorted = look.spectrum;
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2), sorted.end());
        const double median = static_cast<double>(sorted[sorted.size() / 2]);
        const double peak = static_cast<double>(*std::max_element(look.spectrum.begin(), look.spectrum.end()));
        if (median > 0.0 && 10.0 * std::log10(peak / median) >= opt_.sbl_min_contrast_db) {
          DoaResult solved = sbl(frame, u, c, sbl_grid, solver);
          peaks = std::move(solved.peaks);
          rep.sbl_fallback = rep.sbl_fallback || solved.fallback;   // more channels than the kernels take: said in the report
        }
      }
      if (peaks.empty()) {
        for (std::size_t j : group) resolved.push_back(by_snr[j]);
      } else {
        // Each arrival SBL finds is a detection at the gate's range: the leader's reported range and its centroid (the
        // elevation is searched about it). Its first_range, extent and arrivals are not carried: they were measured
        // along the leader's bearing of the map, not along the bearing SBL found (as before D-158).
        for (const DoaPeak& p : peaks) {
          Detection d{h.range, p.bearing, p.snr_db, 1};
          d.centroid_range = h.centroid_range;
          d.range_rate = h.range_rate;             // the map's (the bank's) hypothesis at the leader
          d.range_rate_step = h.range_rate_step;
          resolved.push_back(d);
        }
      }
    }
    hits = std::move(resolved);
  }
  }   // --- the sweep path ---
  if (tone_.plan.on) {
    // D3: the tone path's detections join the sweep's (doppler.hpp merge_tone). On a ping whose pulse is the tone alone
    // the zero-Doppler map is the tone's reverberation at c T / 2: its detections are not reported.
    for (const Detection& t : merge_tone(hits, tone_)) {
      if (bl.enabled && t.range < 0.5 * record.environment.sound_speed * blank_edge) continue;
      if (burst_lo >= 0.0 && (t.range <= burst_lo || t.range >= burst_hi)) continue;
      hits.push_back(t);
    }
  }

  // World position of each detection, from the array's own position and heading in the record.
  Frame f = array_frame(record);
  Vec3d span = f.positions.back() - f.positions.front();
  ElevationOptions eo = opt_.elevation;
  std::vector<ElevationEstimate> els(hits.size());
  double water = eo.water_depth > 0.0 ? eo.water_depth : record.environment.water_depth;
  // What navigation said about its own numbers for this ping (metadata.extra_json "nav"; absent: the options).
  const std::string& said = record.metadata.extra_json;
  if (!(eo.water_depth > 0.0)) eo.water_depth_sigma = std::hypot(eo.water_depth_sigma, extra_number(said, "water_depth_sigma_m", 0.0));
  eo.roll_sigma = std::max(eo.roll_sigma, extra_number(said, "roll_sigma", 0.0));
  LadderProfile ladder;
  bool laddered = false;
  const double sound = record.environment.sound_speed;
  std::vector<double> profile_depth = eo.ssp_depth, profile_speed = eo.ssp_speed;   // depth along a ray instead of the straight line
  std::vector<double> limit_depth, limit_speed;   // the water the multipath term of the depth sigma is worked out in (elevation.multipath_sigma)
  double limit_range_resolution = 0.0, limit_aperture = 0.0;
  if (stacked) {
    // Stacked rows: the elevation of the strongest detections, from the whole array.
    const ElevationFrame ef = elevation_frame(record, rows, opt_.starboard);
    f.origin = ef.origin;
    span = ef.along;
    std::vector<std::size_t> order(hits.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return hits[a].snr_db > hits[b].snr_db; });
    if (static_cast<std::int64_t>(order.size()) > eo.max_detections) order.resize(static_cast<std::size_t>(std::max<std::int64_t>(eo.max_detections, 0)));
    std::vector<ElevationHit> at;
    std::vector<double> background;
    for (std::size_t i : order) {
      std::int64_t ib = 0, ir = 0;
      for (std::int64_t q = 1; q < n_b; ++q) {
        if (std::abs(bearings[static_cast<std::size_t>(q)] - hits[i].bearing) < std::abs(bearings[static_cast<std::size_t>(ib)] - hits[i].bearing)) ib = q;
      }
      // The elevation is searched about the cluster's centroid, whichever range the detection reports (D-158): from the
      // leading edge the search, 1.5 cells beyond it, can stop short of the echo's peak.
      const double el_range = std::isfinite(hits[i].centroid_range) ? hits[i].centroid_range : hits[i].range;
      for (std::int64_t q = 1; q < n_cells; ++q) {
        if (std::abs(last_ranges[static_cast<std::size_t>(q)] - el_range) < std::abs(last_ranges[static_cast<std::size_t>(ir)] - el_range)) ir = q;
      }
      const auto cell_at = static_cast<std::size_t>(ib * n_cells + ir);
      // The mean noise power of the map: with cells holding their largest sample the detector's background is the
      // mean of such maxima, the k-th harmonic number above it.
      double looks = 1.0;
      if (opt_.cell_peak) {
        looks = 0.0;
        for (std::int64_t q = 1; q <= cell_samples_; ++q) looks += 1.0 / static_cast<double>(q);
      }
      if (multiscale) {
        background.push_back(static_cast<double>(background_[cell_at]));
      } else {
        background.push_back(ratio[cell_at] > 0.0F ? static_cast<double>(last_power[cell_at]) / static_cast<double>(ratio[cell_at]) / looks : 0.0);
      }
      at.push_back({el_range, hits[i].bearing, hits[i].first_range});
    }
    auto climb = [&]() {
      // The best rung of the sound-speed ladder this ping: fed from the record, so a replay climbs it the same way.
      laddered = true;
      const bool measured = ladder_.absorb(record, eo.surface_z) || opt_.sound_speed > 0.0;
      ladder = ladder_.evaluate(record.transmit.time_ns, f.origin.x, f.origin.y, f.origin.z - eo.surface_z, sound, measured, water,
                                eo.ssp_depth, eo.ssp_speed, eo.use_gradient ? eo.gradient : 0.0, eo.gradient_sigma);
    };
    if (eo.sector_from_rays || (eo.multipath_sigma && eo.depth)) {
      // hook (opt-in): the elevations sound can arrive from, and the echoes that share a range cell, by rays through
      // the water held this ping
      if (eo.profile.enabled && eo.depth) climb();
      const MultipathProfile through = water_profile(eo.profile, laddered ? &ladder : nullptr, eo.ssp_depth, eo.ssp_speed, eo.use_gradient, eo.gradient,
                                                     eo.gradient_sigma, f.origin.z - eo.surface_z, sound, water);
      limit_depth = through.depth;
      limit_speed = through.speed;
      if (eo.sector_from_rays) {
        eo.sector_ray_depth = through.depth;
        eo.sector_ray_speed = through.speed;
      }
    }
    {
      // What the rows cannot separate: the range resolution of the pulse, and the lobe width of the rows' height.
      const Waveform& pulse = record.transmit.waveform;
      const double f_lo = std::min(pulse.f_start, pulse.f_stop), f_hi = std::max(pulse.f_start, pulse.f_stop);
      const double pulse_band = f_hi > f_lo ? f_hi - f_lo : 1.0 / pulse.duration;
      const auto [h_lo, h_hi] = std::minmax_element(ef.heights.begin(), ef.heights.end());
      limit_range_resolution = 0.5 * sound / pulse_band;
      limit_aperture = *h_hi > *h_lo ? sound / (0.5 * (f_lo + f_hi)) / (*h_hi - *h_lo) : 2.0;
    }
    double noise_weight = 0.0;
    for (const auto& r : rows) noise_weight += weight_power(static_cast<std::int64_t>(r.size()), opt_.hann && opt_.beamformer != "mvdr");
    const std::vector<ElevationEstimate> got = estimate_elevation(record, rows, at, background, ef, eo, noise_weight, opt_.range_cell_m,
                                                                  opt_.backend, opt_.device);
    for (std::size_t q = 0; q < order.size(); ++q) els[order[q]] = got[q];
    rep.rows = static_cast<std::int64_t>(rows.size());
    if (eo.profile.enabled && eo.depth) {
      if (!laddered) climb();
      profile_depth = ladder.depth;
      profile_speed = ladder.speed;
      rep.profile_rung = ladder.rung;
      rep.profile_bins = ladder.bins;
      rep.profile_lo = ladder.span_lo;
      rep.profile_hi = ladder.span_hi;
    }
    if (profile_depth.empty() && eo.use_gradient) {
      // c(z) = local speed + gradient (z - own depth), from the surface down.
      const double own = f.origin.z - eo.surface_z;
      const double bottom = std::max({std::isfinite(water) ? water : 0.0, own, 0.0}) + 10000.0;
      profile_depth = {0.0, bottom};
      profile_speed = {sound - eo.gradient * own, sound + eo.gradient * (bottom - own)};
    }
  }
  // --- hook: the water this ping's rays go through (raytrace.hpp), opt-in: the multipath depth with `rays`, the
  // coverage picture, and the profile ensemble. One row has no elevation, but it has a ladder all the same.
  const bool want_rays = (opt_.multipath_depth.enabled && opt_.multipath_depth.rays) || opt_.coverage.enabled;
  MultipathProfile rays_water;
  if (want_rays || (stacked && eo.depth && eo.profile.ensemble)) {
    if (!laddered && eo.profile.enabled && want_rays) {
      laddered = true;
      const bool measured = ladder_.absorb(record, eo.surface_z) || opt_.sound_speed > 0.0;
      ladder = ladder_.evaluate(record.transmit.time_ns, f.origin.x, f.origin.y, f.origin.z - eo.surface_z, sound, measured, water, eo.ssp_depth,
                                eo.ssp_speed, eo.use_gradient ? eo.gradient : 0.0, eo.gradient_sigma);
      rep.profile_rung = ladder.rung;
      rep.profile_bins = ladder.bins;
      rep.profile_lo = ladder.span_lo;
      rep.profile_hi = ladder.span_hi;
    }
    rays_water = water_profile(eo.profile, laddered ? &ladder : nullptr, eo.ssp_depth, eo.ssp_speed, eo.use_gradient, eo.gradient, eo.gradient_sigma,
                               f.origin.z - eo.surface_z, sound, water);
  }
  // The stacked rows' depth back along a ray: the stepped ray as before, or (elevation.ray_arcs) the chain's one ray calculation.
  auto back_trace = [&](double level, double range) {
    const double own = f.origin.z - eo.surface_z;
    return eo.ray_arcs ? ray_depth_arcs(own, level, range, profile_depth, profile_speed, water) : ray_depth(own, level, range, profile_depth, profile_speed, water);
  };
  // The same arrival through every member of the profile ensemble: where the depth may lie.
  auto ensemble_span = [&](double level, double range, double depth_m) {
    std::string out;
    if (rays_water.ensemble.empty()) return out;
    double lo = depth_m, hi = depth_m;
    for (const std::vector<double>& speeds : rays_water.ensemble) {
      const double z = eo.surface_z + ray_depth_arcs(f.origin.z - eo.surface_z, level, range, rays_water.depth, speeds, water);
      lo = std::min(lo, z);
      hi = std::max(hi, z);
    }
    std::ostringstream text;
    text << std::setprecision(10) << "\"depth_ensemble_m\": [" << lo << ", " << hi << "]";
    return text.str();
  };
  // --- end hook ---
  const Vec3d axis = normalized(Vec3d{span.x, span.y, 0.0});
  const double side = opt_.starboard ? 1.0 : -1.0;
  const Vec3d broadside{-side * axis.y, side * axis.x, 0.0};
  if (!have_origin_) {
    origin_ = f.origin;
    t0_ns_ = record.transmit.time_ns;
    have_origin_ = true;
  } else if (record.transmit.time_ns <= last_ns_) {
    // No time since the ping before (a source without timestamps; sonar-4bs): counted, and with a ping interval given the
    // tracker's clock advances by it. Ordinary pings (increasing transmit times) never come here: their clock is as before.
    ++pings_without_time_;
    if (opt_.tracker_ping_interval_s > 0.0) {
      clock_offset_ns_ += last_ns_ - record.transmit.time_ns + static_cast<std::int64_t>(std::llround(opt_.tracker_ping_interval_s * 1e9));
    }
  }
  last_ns_ = record.transmit.time_ns;
  rep.pings_without_time = pings_without_time_;
  const double t = static_cast<double>(record.transmit.time_ns - t0_ns_ + clock_offset_ns_) * 1e-9;
  // A monostatic echo shows the target where it was when the sound reached it: transmit + range / c, with the c that
  // turned the two-way time into range (the record's, after `sound_speed`). D-156; python/gpusonar/live.py the same.
  const double c_range = record.environment.sound_speed;
  // What the angle cannot know: how the ray bent on the way. The prior alone, or what the profile's rung claims.
  auto bend = [&](double level, double range) {
    return laddered && !ladder.depth.empty()
               ? profile_depth_sigma(f.origin.z - eo.surface_z, level, range, ladder.depth, ladder.speed, ladder.sigma, eo.gradient_sigma, water, eo.ray_arcs)
               : refraction_sigma(range, sound, eo.gradient_sigma);
  };
  // What echoes sharing the range cell add to a depth read at `level` with the angle's own sigma `sigma` (rad): the
  // term in metres, and its members appended to the report's extra ones (elevation.multipath_sigma; 0 when off).
  auto multipath_term = [&](double level, double range, double sigma, double depth_m, std::string& extra) {
    if (!eo.multipath_sigma || limit_depth.empty()) return 0.0;
    const MultipathLimit lim = multipath_depth_limit(limit_depth, limit_speed, water, f.origin.z - eo.surface_z, range, level, sigma, depth_m - eo.surface_z,
                                                     limit_range_resolution, limit_aperture, eo.first_arrival, eo.first_arrival_db, eo.ambiguity_sigma,
                                                     static_cast<int>(eo.multipath_sigma_bounces), eo.multipath_sigma_max_depth);
    const std::string members = multipath_limit_members(lim);
    extra = extra.empty() ? members : extra + ", " + members;
    return lim.extra;
  };
  std::vector<Detection> local;
  const bool placed_by_elevation = chain_range_estimate(opt_) == RangeEstimate::First;
  // sonar-fyj (opt-in): the LFM coupling of the ownship's own motion, factor x its closing speed along each line of sight
  // (the mean of the transmit and receive poses' velocities, from navigation), added to the range. live.py the same.
  // A velocity that is not finite is not known: no correction then (the factor 0), rather than a NaN range.
  const Vec3d v_own = (record.geometry.tx_pose.velocity + record.geometry.rx_pose.velocity) * 0.5;
  const bool v_known = std::isfinite(v_own.x) && std::isfinite(v_own.y) && std::isfinite(v_own.z);
  // With the Doppler bank on, every hypothesis' map is aligned to the zero one (doppler.hpp, shifts): a detection shows
  // where today's chain shows it, coupling and all, and carries the bank's range rate. The correction then takes the
  // bank's estimate as each detection's closing speed (-range_rate: the target's part and the own ship's together), and
  // `full` moves no track further: its target part is in every detection already (no double correction; live.py the same).
  const double lfm_factor = opt_.range_doppler != RangeDoppler::Off ? lfm_range_doppler_factor(record.transmit.waveform) : 0.0;
  const double rd_factor = v_known || bank_on ? lfm_factor : 0.0;
  const auto closing = [&](const Detection& h, double ux, double uy, double uz) {
    if (bank_on) return std::isfinite(h.range_rate) ? -h.range_rate : 0.0;
    return v_own.x * ux + v_own.y * uy + v_own.z * uz;
  };
  for (std::size_t i = 0; i < hits.size(); ++i) {
    const Detection& h = hits[i];
    const double sb = std::sin(h.bearing), cb = std::cos(h.bearing);
    const double lx = sb * axis.x + cb * broadside.x, ly = sb * axis.y + cb * broadside.y;
    const double h_range = rd_factor != 0.0 ? h.range + rd_factor * closing(h, lx, ly, 0.0) : h.range;
    double px = f.origin.x - origin_.x + h_range * lx;
    double py = f.origin.y - origin_.y + h_range * ly;
    ReportDetection d{h_range, h.bearing / kDeg, h.snr_db};
    if (opt_.doppler.report && (bank_on || tone_on)) {     // opt-in: the report line is a contract (DOPPLER.md question 6)
      d.has_range_rate = true;
      d.range_rate_m_s = h.range_rate;
      d.range_rate_step_m_s = h.range_rate_step;
      if (tone_on) d.doppler_path = h.doppler_path;        // which path found it (D3)
    }
    if (std::isfinite(h.extent)) {
      d.extent_m = h.extent;
      d.first_range_m = h.first_range;
      d.arrivals = h.arrivals;
    }
    Detection m{0.0, 0.0, h.snr_db, h.cells};
    m.time = t + h_range / c_range;
    m.range_sigma = h.range_sigma;                         // a tone's coarse range (D3); NaN: the tracker's
    const ElevationEstimate& e = els[i];
    if (stacked && e.valid) {
      const Vec3d& u = e.world;
      const double level = std::asin(std::clamp(-u.z, -1.0, 1.0));   // elevation above the horizontal, roll and pitch applied
      d.elevation_valid = true;
      d.elevation_deg = level / kDeg;
      d.elevation_sigma_deg = e.sigma / kDeg;
      d.elevation_quality = e.quality;
      d.elevation_ambiguous = e.ambiguous;
      // An ambiguous elevation is one of several candidates, and which is picked changes from ping to ping: placing
      // the detection by it would move it by the range times the difference between their cosines (a kilometre at
      // 12 km) and break its track. The row's own bearing and the slant range are kept then, and the report says so.
      d.bearing_uncorrected = eo.correct_bearing && e.ambiguous;
      if (eo.correct_bearing && !e.ambiguous) {
        // The line array gave the cone angle; with the elevation the horizontal bearing follows.
        const double e_range = rd_factor != 0.0 ? e.range + rd_factor * closing(h, u.x, u.y, u.z) : e.range;   // sonar-fyj
        px = f.origin.x - origin_.x + e_range * u.x;
        py = f.origin.y - origin_.y + e_range * u.y;
        if (placed_by_elevation) {
          // D-158: placed at the elevation's range (the earliest arrival of the window searched about the centroid,
          // elevation.arrival first; else its strongest): its range_m and time are that range too. With centroid given,
          // as before D-158: placed there, reported and timed at the centroid. live.py the same.
          d.range_m = e_range;
          m.time = t + e_range / c_range;
        }
        d.bearing_corrected = true;
        d.cone_bearing_deg = h.bearing / kDeg;
        d.bearing_deg = std::atan2(u.x * axis.x + u.y * axis.y, u.x * broadside.x + u.y * broadside.y) / kDeg;
      }
      if (e.ambiguous) {
        for (const Vec3d& alt : e.alternatives) d.elevation_alternatives_deg.push_back(std::asin(std::clamp(-alt.z, -1.0, 1.0)) / kDeg);
      }
      if (eo.depth) {
        d.has_depth = true;
        d.image_depth_m = f.origin.z + e.range * u.z;                // straight-line depth: outside the water for a bounce
        const double sigma_z = e.range * std::cos(level) * e.sigma;  // from the angle: noise, roll, attitude timing
        if (eo.fold) {
          const auto folded = fold_depth(d.image_depth_m, water, eo.surface_z);
          d.depth_m = folded.first;
          d.path = folded.second;
        } else {
          d.depth_m = d.image_depth_m;
          d.path = "direct";
        }
        // Back along the ray through the profile, not the straight line.
        if (!profile_depth.empty()) d.depth_m = eo.surface_z + back_trace(level, e.range);
        d.extra = ensemble_span(level, e.range, d.depth_m);
        // What the angle cannot know: the gradient along the path, and the water depth a bottom bounce is folded about.
        d.depth_refraction_sigma_m = bend(level, e.range);
        if (laddered) d.profile_source = ladder.rung;
        const double fold_sigma = (d.path == "bottom" || d.path == "multiple") ? 2.0 * eo.water_depth_sigma : 0.0;
        // What the rows cannot know: which of the echoes sharing the range cell they read (opt-in). It is the same
        // boundary ping after ping, so the track's filter is not given it to average: the track's own sigma gains it below.
        const double multipath_sigma = multipath_term(level, e.range, e.sigma, d.depth_m, d.extra);
        d.depth_sigma_m = std::sqrt(sigma_z * sigma_z + d.depth_refraction_sigma_m * d.depth_refraction_sigma_m + fold_sigma * fold_sigma +
                                    multipath_sigma * multipath_sigma);
        m.depth = d.image_depth_m;
        m.depth_sigma = sigma_z;
        if (e.ambiguous) {
          // The track, if it has a depth already, takes the candidate that fits it.
          m.depth_ambiguous = true;
          for (const Vec3d& alt : e.alternatives) m.depth_alternatives.push_back(f.origin.z + e.range * alt.z);
        }
      }
    }
    // The tracker measures (range, bearing) about the frame origin; its state is (east, north).
    m.range = std::hypot(px, py);
    m.bearing = std::atan2(py, px);
    local.push_back(m);
    rep.detections.push_back(d);
  }
  const auto confirmed = resolved_ ? tracker_.update(t, local) : std::vector<const Track*>{};   // D3: see tone_only_ping
  ClassifyOptions co2 = opt_.classify;
  co2.origin_x = f.origin.y - origin_.y;
  co2.origin_y = f.origin.x - origin_.x;
  co2.time = t;   // positions at the report's time (the ping's transmit time), not at each track's last detection (D-156)
  // Echo grouping after the elevation step below, which finds the bounces (sonar-bal; python/gpusonar/live.py the same).
  const bool group = co2.multipath;
  co2.multipath = false;
  std::vector<TrackClass> classes = classify_tracks(confirmed, co2);
  // sonar-fyj: the target's own part, from its fitted velocity, on the reported position (before anything reads it).
  if (opt_.range_doppler == RangeDoppler::Full && !bank_on) shift_for_target_doppler(classes, lfm_factor, co2.origin_x, co2.origin_y);
  std::vector<std::int64_t> bounces;   // tracks whose straight-line depth lies outside the water
  for (const TrackClass& tc : classes) {
    ReportTrack rt{tc.id, tc.parent, kind_name(tc.kind), tc.y + origin_.x, tc.x + origin_.y, tc.vy, tc.vx, tc.speed,
                   std::hypot(tc.y - co2.origin_y, tc.x - co2.origin_x), tc.mean_snr_db, tc.seen};
    if (stacked && eo.depth) {
      rt.depth_reported = true;
      for (const Track* tr : confirmed) {
        if (tr->id != tc.id || !std::isfinite(tr->depth)) continue;
        rt.has_depth = true;
        const double slant = std::hypot(rt.range_m, tr->depth - f.origin.z);
        const double bent = bend(std::asin(std::clamp((f.origin.z - tr->depth) / slant, -1.0, 1.0)), slant);
        if (laddered) rt.profile_source = ladder.rung;
        if (eo.fold) {
          const auto folded = fold_depth(tr->depth, water, eo.surface_z);
          rt.depth_m = folded.first;
          rt.path = folded.second;
        } else {
          rt.depth_m = tr->depth;
          rt.path = "direct";
        }
        const double fold_sigma = (rt.path == "bottom" || rt.path == "multiple") ? 2.0 * eo.water_depth_sigma : 0.0;
        // The filter averages what differs from ping to ping; the bending of the ray does not.
        rt.depth_sigma_m = std::sqrt(tr->depth_var + bent * bent + fold_sigma * fold_sigma);
        if (!profile_depth.empty()) {
          // The track's straight-line depth as an arrival angle at its range, then along the ray.
          rt.depth_m = eo.surface_z + back_trace(std::asin((f.origin.z - tr->depth) / slant), slant);
        }
        rt.extra = ensemble_span(std::asin(std::clamp((f.origin.z - tr->depth) / slant, -1.0, 1.0)), slant, rt.depth_m);
        {
          // The same boundary echoes ping after ping: what they add is not averaged away by the filter.
          const double level = std::asin(std::clamp((f.origin.z - tr->depth) / slant, -1.0, 1.0));
          const double more = multipath_term(level, slant, std::sqrt(tr->depth_var) / (slant * std::max(std::cos(level), 1e-6)), rt.depth_m, rt.extra);
          rt.depth_sigma_m = std::hypot(rt.depth_sigma_m, more);
        }
        // Its straight-line depth is above the surface or below the bottom: a bounce.
        if (eo.classify && outside_water(tr->depth, std::hypot(std::sqrt(tr->depth_var), bent), water, eo.surface_z, eo.classify_sigma,
                                         eo.water_depth_sigma)) {
          bounces.push_back(tc.id);
        }
      }
    }
    rep.tracks.push_back(rt);
  }
  // A bounce is multipath; before echo grouping, so that it is never a group's object (sonar-bal: made multipath after
  // grouping, an object left its members a parent that was itself multipath).
  if (group) {
    co2.multipath = true;
    group_echoes(classes, co2, bounces);
  } else {
    for (TrackClass& tc : classes) {
      if (std::find(bounces.begin(), bounces.end(), tc.id) != bounces.end()) tc.kind = Kind::Multipath;
    }
  }
  for (std::size_t i = 0; i < classes.size(); ++i) {
    rep.tracks[i].kind = kind_name(classes[i].kind);
    rep.tracks[i].parent = classes[i].parent;
  }
  // --- hook: depth from multipath timing (multipath_depth.hpp), opt-in; adds report members only ---
  if (opt_.multipath_depth.enabled) {
    std::vector<MultipathHit> mh(hits.size());
    for (std::size_t i = 0; i < hits.size(); ++i) {
      const ReportDetection& d = rep.detections[i];
      // The centroid when the detector gave one (D-158): what this stage was built on (with first_range and extent known
      // it takes their middle anyway; an SBL detection has only its range).
      const double mh_range = std::isfinite(hits[i].centroid_range) ? hits[i].centroid_range : hits[i].range;
      mh[i] = {mh_range, hits[i].bearing, hits[i].first_range, hits[i].extent, local[i].range * std::sin(local[i].bearing),
               local[i].range * std::cos(local[i].bearing), hits[i].snr_db, d.has_depth, d.depth_m - eo.surface_z, d.depth_sigma_m, d.depth_refraction_sigma_m, {}};
      for (double z : local[i].depth_alternatives) mh[i].depth_alternatives.push_back(fold_depth(z, water, eo.surface_z).first - eo.surface_z);
    }
    std::vector<MultipathTrack> mt;
    for (const ReportTrack& t : rep.tracks) mt.push_back({t.id, t.y - origin_.y, t.x - origin_.x, t.has_depth, t.depth_m - eo.surface_z, t.depth_sigma_m});
    if (opt_.multipath_depth.rays) multipath_.set_profile(rays_water);
    const MultipathReport mr = multipath_.process(record, f.origin, mh, mt, rows, opt_.backend, opt_.device);
    auto add = [](std::string& extra, const std::string& more) {
      if (!more.empty()) extra = extra.empty() ? more : extra + ", " + more;
    };
    for (std::size_t i = 0; i < mh.size(); ++i) add(rep.detections[i].extra, mr.detections[i]);
    for (std::size_t i = 0; i < mt.size(); ++i) add(rep.tracks[i].extra, mr.tracks[i]);
  }
  // --- end hook ---
  // --- hook: the coverage picture (coverage.hpp), opt-in; one member of the report line, refreshed when something changed ---
  if (opt_.coverage.enabled) rep.extra = coverage_.update(record, f.origin, water, rays_water);
  // --- end hook ---
  rep.ownship = f.origin;
  rep.processing_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  return rep;
}

std::string Report::json() const { return json_part(0, detections.size(), 0, tracks.size(), 0, 0); }

std::string Report::json_part(std::size_t det_first, std::size_t det_last, std::size_t track_first, std::size_t track_last,
                              std::size_t part, std::size_t parts) const {
  std::ostringstream s;
  s << std::setprecision(10);
  s << "{\"ping\": " << ping << ", \"sequence\": " << sequence << ", \"time_ns\": " << time_ns << ", \"ownship_xyz\": ["
    << ownship.x << ", " << ownship.y << ", " << ownship.z << "], \"pings_without_nav\": " << pings_without_nav;
  // Only when some ping had no velocity from navigation (sonar-8is): every line without it is the one it was.
  if (pings_without_velocity > 0) s << ", \"pings_without_velocity\": " << pings_without_velocity;
  if (pings_without_time > 0) s << ", \"pings_without_time\": " << pings_without_time;   // sonar-4bs
  s << ", \"detections\": [";
  for (std::size_t i = det_first; i < det_last; ++i) {
    const auto& d = detections[i];
    s << (i > det_first ? ", " : "") << "{\"range_m\": " << d.range_m << ", \"bearing_deg\": " << d.bearing_deg << ", \"snr_db\": ";
    num(s, d.snr_db);
    if (d.extent_m >= 0.0) {
      s << ", \"extent_m\": " << d.extent_m << ", \"first_range_m\": " << d.first_range_m;
      if (d.arrivals.size() > 1) {
        s << ", \"arrivals\": [";
        for (std::size_t q = 0; q < d.arrivals.size(); ++q) s << (q ? ", " : "") << "[" << d.arrivals[q].range << ", " << d.arrivals[q].snr_db << "]";
        s << "]";
      }
    }
    if (rows > 0) {
      s << ", \"elevation_deg\": ";
      if (!d.elevation_valid) {
        s << "null";
      } else {
        s << d.elevation_deg << ", \"elevation_sigma_deg\": " << d.elevation_sigma_deg << ", \"elevation_quality\": "
          << d.elevation_quality << ", \"elevation_ambiguous\": " << (d.elevation_ambiguous ? "true" : "false");
        if (!d.elevation_alternatives_deg.empty()) {
          s << ", \"elevation_alternatives_deg\": [";
          for (std::size_t q = 0; q < d.elevation_alternatives_deg.size(); ++q) s << (q ? ", " : "") << d.elevation_alternatives_deg[q];
          s << "]";
        }
        if (d.bearing_corrected) s << ", \"cone_bearing_deg\": " << d.cone_bearing_deg;
        if (d.bearing_uncorrected) s << ", \"bearing_corrected\": false";
        if (d.has_depth) {
          s << ", \"depth_m\": " << d.depth_m << ", \"depth_sigma_m\": " << d.depth_sigma_m << ", \"depth_refraction_sigma_m\": "
            << d.depth_refraction_sigma_m << ", \"image_depth_m\": "
            << d.image_depth_m << ", \"path\": \"" << d.path << "\"";
          if (!d.profile_source.empty()) s << ", \"depth_profile_source\": \"" << d.profile_source << "\"";
        }
      }
    }
    if (d.has_range_rate) {
      s << ", \"range_rate_m_s\": ";
      num(s, d.range_rate_m_s);
      s << ", \"range_rate_step_m_s\": ";
      num(s, d.range_rate_step_m_s);
      if (d.doppler_path >= 0) s << ", \"doppler_path\": \"" << (d.doppler_path == 1 ? "tone" : d.doppler_path == 2 ? "both" : "sweep") << "\"";
    }
    if (!d.extra.empty()) s << ", " << d.extra;
    s << "}";
  }
  s << "], \"tracks\": [";
  for (std::size_t i = track_first; i < track_last; ++i) {
    const auto& t = tracks[i];
    s << (i > track_first ? ", " : "") << "{\"id\": " << t.id << ", \"kind\": \"" << t.kind << "\", \"parent\": ";
    if (t.parent < 0) s << "null"; else s << t.parent;
    s << ", \"x\": " << t.x << ", \"y\": " << t.y << ", \"vx\": " << t.vx << ", \"vy\": " << t.vy << ", \"speed_m_s\": " << t.speed
      << ", \"range_m\": " << t.range_m << ", \"mean_snr_db\": ";
    num(s, t.mean_snr_db);
    s << ", \"seen\": " << t.seen;
    if (t.depth_reported) {
      s << ", \"depth_m\": ";
      if (!t.has_depth) s << "null";
      else s << t.depth_m << ", \"depth_sigma_m\": " << t.depth_sigma_m << ", \"path\": \"" << t.path << "\"";
      if (t.has_depth && !t.profile_source.empty()) s << ", \"depth_profile_source\": \"" << t.profile_source << "\"";
    }
    if (!t.extra.empty()) s << ", " << t.extra;
    s << "}";
  }
  s << "], \"sync\": ";
  if (!synced) {
    s << "null";
  } else {
    s << "{\"ok\": " << (sync_ok ? "true" : "false") << ", \"worst_residual_us\": " << sync_worst_residual_us << ", \"offsets_us\": [";
    for (std::size_t i = 0; i < sync_offsets_us.size(); ++i) s << (i ? ", " : "") << sync_offsets_us[i];
    s << "], \"drift_ppm\": [";
    for (std::size_t i = 0; i < sync_drift_ppm.size(); ++i) s << (i ? ", " : "") << sync_drift_ppm[i];
    s << "]";
    if (!sync_flagged.empty()) {
      s << ", \"flagged\": [";
      for (std::size_t i = 0; i < sync_flagged.size(); ++i) s << (i ? ", " : "") << sync_flagged[i];
      s << "]";
    }
    s << "}";
  }
  if (blanking_min_range_m >= 0.0) {
    s << ", \"blanking\": {\"min_range_m\": " << blanking_min_range_m << ", \"mode\": \"" << (blanking_zero ? "zero" : "ignore") << "\"}";
  }
  if (rows > 0) s << ", \"rows\": " << rows;
  if (!profile_rung.empty()) {
    s << ", \"profile\": {\"rung\": \"" << profile_rung << "\"";
    if (profile_bins > 0) s << ", \"bins\": " << profile_bins << ", \"span_m\": [" << profile_lo << ", " << profile_hi << "]";
    s << "}";
  }
  if (sbl_fallback) s << ", \"sbl_fallback\": true";
  if (!extra.empty()) s << ", " << extra;
  s << ", \"processing_seconds\": " << processing_seconds;
  if (parts > 0) s << ", \"part\": [" << part << ", " << parts << "]";
  s << "}";
  return s.str();
}

std::vector<std::string> Report::datagrams(std::size_t max_bytes) const {
  std::string whole = json();
  if (whole.size() <= max_bytes) return {std::move(whole)};
  // Shares of the detections, then of the tracks, each as large as fits beside the fields every part repeats.
  const std::size_t n_det = detections.size(), n_trk = tracks.size();
  const std::size_t head = json_part(0, 0, 0, 0, 999, 999).size();
  std::vector<std::size_t> size(n_det + n_trk);   // an item with its separator
  for (std::size_t i = 0; i < n_det; ++i) size[i] = json_part(i, i + 1, 0, 0, 999, 999).size() - head + 2;
  for (std::size_t i = 0; i < n_trk; ++i) size[n_det + i] = json_part(0, 0, i, i + 1, 999, 999).size() - head + 2;
  std::vector<std::pair<std::size_t, std::size_t>> cuts;   // [first, last) over detections then tracks
  std::size_t first = 0, used = head;
  for (std::size_t i = 0; i < size.size(); ++i) {
    if (i > first && used + size[i] > max_bytes) {
      cuts.emplace_back(first, i);
      first = i;
      used = head;
    }
    used += size[i];
  }
  cuts.emplace_back(first, size.size());
  std::vector<std::string> out;
  for (std::size_t k = 0; k < cuts.size(); ++k) {
    const std::size_t a = cuts[k].first, b = cuts[k].second;
    out.push_back(json_part(std::min(a, n_det), std::min(b, n_det), std::max(a, n_det) - n_det, std::max(b, n_det) - n_det, k + 1, cuts.size()));
  }
  return out;
}

std::vector<std::string> line_datagrams(const std::string& lines, std::size_t max_bytes) {
  std::vector<std::string> out;
  if (lines.size() <= max_bytes) {
    out.push_back(lines);
    return out;
  }
  std::string now;
  std::size_t at = 0;
  while (at < lines.size()) {
    const std::size_t end = std::min(lines.find('\n', at), lines.size() - 1) + 1;
    if (!now.empty() && now.size() + (end - at) > max_bytes) {
      out.push_back(std::move(now));
      now.clear();
    }
    now.append(lines, at, end - at);
    at = end;
  }
  if (!now.empty()) out.push_back(std::move(now));
  return out;
}

std::int64_t passive_span(const PassiveOptions& options, const PingRecord& record) {
  PassiveOptions o = options;
  resolve(o, record);
  return o.n_fft + (o.n_average - 1) * o.hop;
}

std::vector<PassiveFrame> passive_bearings(const PingRecord& record, const PassiveOptions& options, std::vector<double>* bearings_out) {
  PassiveOptions o = options;
  resolve(o, record);                 // block sizes left to be derived
  const Waveform& w = record.transmit.waveform;
  double f_low = o.f_low, f_high = o.f_high;
  if (f_low <= 0.0 && f_high <= 0.0) {
    f_low = std::min(w.f_start, w.f_stop);
    f_high = std::max(w.f_start, w.f_stop);
  }
  const std::vector<double> bearings = grid(o.bearing_first_deg, o.bearing_last_deg, o.bearing_step_deg);
  if (bearings_out) *bearings_out = bearings;
  // The rows are looked for with or without the elevation option: the bearing solve's model is one
  // line, and the hydrophones of several rows taken as one line are not one (the phases between rows
  // depend on the elevation, which that model does not have: it then finds a wrong bearing or none).
  std::vector<std::vector<std::int64_t>> rows = find_rows(record, row_tolerance(record, o.elevation.row_tolerance_m), o.elevation.min_row_elements);
  if (rows.size() < 2) rows.clear();
  if (rows.empty()) {
    const std::vector<double> u = element_offsets(record);
    const std::vector<DoaFrame> frames = passive_frames(record, o.n_fft, o.hop, o.n_average, f_low, f_high);
    std::vector<PassiveFrame> out(frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i) {    // each solve is already parallel over frequency
      const DoaResult r = o.use_sbl ? sbl(frames[i], u, record.environment.sound_speed, bearings, o.sbl)
                                    : conventional(frames[i], u, record.environment.sound_speed, bearings, o.sbl);
      out[i] = {frames[i].time, r.peaks, r.spectrum, r.noise, {}, {}};
      out[i].sbl_fallback = r.fallback;
    }
    return out;
  }
  // Stacked rows. Bearings: every row is a line with the same response over bearing, so the rows
  // of equal layout are further snapshots of one line (the others are left out of the bearing
  // solve). Elevation: the rows' beams at each bearing found, steered over elevation.
  std::vector<PingRecord> subs;
  for (const auto& r : rows) {
    if (r.size() == rows[0].size()) subs.push_back(row_record(record, r));
  }
  const std::vector<double> u = element_offsets(subs[0]);
  std::vector<DoaFrame> frames = passive_frames(subs[0], o.n_fft, o.hop, o.n_average, f_low, f_high);
  for (std::size_t k = 1; k < subs.size(); ++k) {
    const std::vector<DoaFrame> more = passive_frames(subs[k], o.n_fft, o.hop, o.n_average, f_low, f_high);
    for (std::size_t i = 0; i < frames.size(); ++i) {
      // [F][L][N] -> [F][L + L'][N]
      DoaFrame& a = frames[i];
      const DoaFrame& b = more[i];
      const auto n_f = a.frequencies.size();
      const auto la = static_cast<std::size_t>(a.n_snapshots * a.n_channels), lb = static_cast<std::size_t>(b.n_snapshots * b.n_channels);
      std::vector<std::complex<double>> merged;
      merged.reserve(n_f * (la + lb));
      for (std::size_t fi = 0; fi < n_f; ++fi) {
        merged.insert(merged.end(), a.snapshots.begin() + static_cast<std::ptrdiff_t>(fi * la), a.snapshots.begin() + static_cast<std::ptrdiff_t>((fi + 1) * la));
        merged.insert(merged.end(), b.snapshots.begin() + static_cast<std::ptrdiff_t>(fi * lb), b.snapshots.begin() + static_cast<std::ptrdiff_t>((fi + 1) * lb));
      }
      a.snapshots = std::move(merged);
      a.n_snapshots += b.n_snapshots;
    }
  }
  const auto n_lines = static_cast<std::int64_t>(subs.size());
  subs.clear();
  const ElevationFrame ef = elevation_frame(record, rows, o.starboard);
  const std::int64_t span = o.n_fft + (o.n_average - 1) * o.hop;
  std::vector<PassiveFrame> out(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const DoaResult r = o.use_sbl ? sbl(frames[i], u, record.environment.sound_speed, bearings, o.sbl)
                                  : conventional(frames[i], u, record.environment.sound_speed, bearings, o.sbl);
    out[i] = {frames[i].time, r.peaks, r.spectrum, r.noise, {}, {}};
    out[i].sbl_fallback = r.fallback;
    if (!o.elevation.enabled) {
      // Without the option: cone bearings from the rows as snapshots, and the frame says so.
      out[i].rows_as_snapshots = n_lines;
      continue;
    }
    for (const DoaPeak& peak : r.peaks) {
      const PassiveElevation pe = passive_elevation(record, rows, peak.bearing, f_low, f_high, o.n_fft, o.hop,
                                                    static_cast<std::int64_t>(i) * span, o.n_average, o.elevation.first_deg,
                                                    o.elevation.last_deg, o.elevation.step_deg, o.starboard);
      const Vec3d dir = arrival_direction(ef, std::sin(peak.bearing), std::sin(pe.elevation));
      out[i].elevations.push_back(std::asin(std::clamp(-dir.z, -1.0, 1.0)));   // level-referenced: the record's roll and pitch applied
      out[i].elevation_quality.push_back(pe.quality);
    }
  }
  return out;
}

std::string passive_json(const std::vector<PassiveFrame>& frames, std::int64_t record_time_ns) {
  std::ostringstream s;
  s << std::setprecision(10);
  for (const PassiveFrame& f : frames) {
    s << "{\"time_ns\": " << record_time_ns + static_cast<std::int64_t>(std::llround(f.time * 1e9)) << ", \"time_s\": " << f.time
      << ", \"noise_db\": " << 10.0 * std::log10(std::max(f.noise, 1e-300)) << ", \"bearings_deg\": [";
    for (std::size_t i = 0; i < f.peaks.size(); ++i) s << (i ? ", " : "") << f.peaks[i].bearing / kDeg;
    s << "], \"power_db\": [";
    for (std::size_t i = 0; i < f.peaks.size(); ++i) s << (i ? ", " : "") << 10.0 * std::log10(std::max(f.peaks[i].power, 1e-300));
    s << "], \"snr_db\": [";
    for (std::size_t i = 0; i < f.peaks.size(); ++i) s << (i ? ", " : "") << f.peaks[i].snr_db;
    if (!f.elevations.empty()) {
      s << "], \"elevations_deg\": [";
      for (std::size_t i = 0; i < f.elevations.size(); ++i) s << (i ? ", " : "") << f.elevations[i] / kDeg;
      s << "], \"elevation_quality\": [";
      for (std::size_t i = 0; i < f.elevation_quality.size(); ++i) s << (i ? ", " : "") << f.elevation_quality[i];
    }
    s << "]";
    if (f.rows_as_snapshots > 0) s << ", \"rows_as_snapshots\": " << f.rows_as_snapshots;
    if (f.sbl_fallback) s << ", \"sbl_fallback\": true";
    s << "}\n";
  }
  return s.str();
}

}  // namespace sonar::live
