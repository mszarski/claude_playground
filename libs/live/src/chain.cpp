#include "sonar/live/chain.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/detect/cfar.hpp"
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

std::vector<float> range_bearing_power(const PingRecord& record, const std::vector<double>& ranges,
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
  std::vector<float> power(points.size());
  for (std::size_t i = 0; i < power.size(); ++i) power[i] = std::norm(image[i]);
  return power;
}

LiveChain::LiveChain(const ChainOptions& options) : opt_(options), tracker_(options.tracker) {
  bearings = grid(opt_.bearing_first_deg, opt_.bearing_last_deg, opt_.bearing_step_deg);
  if (opt_.beamformer != "backprojection" && opt_.beamformer != "farfield" && opt_.beamformer != "mvdr") {
    throw std::invalid_argument("LiveChain: beamformer '" + opt_.beamformer + "' (backprojection, farfield or mvdr)");
  }
}

Report LiveChain::process(PingRecord& record) {
  const auto t_start = std::chrono::steady_clock::now();
  Report rep;
  rep.ping = pings_++;
  rep.sequence = record.metadata.sequence;
  rep.time_ns = record.transmit.time_ns;
  double burst_lo = -1.0, burst_hi = -1.0;       // ranges the reference bursts occupy, when they stay in the data
  if (opt_.sync) {
    // Boards run on their own clocks: line them up before anything is summed across them.
    const SyncResult r = synchronise(record, opt_.reference);
    rep.synced = true;
    rep.sync_ok = r.ok;
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
  prepare(record, opt_.backend, opt_.device);

  // Range grid: finer than the range resolution, then averaged into cells.
  const double c = record.environment.sound_speed;
  const Waveform& w = record.transmit.waveform;
  const Samples& s = record.samples;
  const double band = std::abs(w.f_stop - w.f_start) > 0.0 ? std::abs(w.f_stop - w.f_start) : 1.0 / w.duration;
  const auto n_b = static_cast<std::int64_t>(bearings.size());
  std::int64_t n_cells = 0;
  if (opt_.beamformer == "backprojection") {
    const double r0 = 0.5 * c * s.start_time;
    const double r1 = 0.5 * c * (s.start_time + static_cast<double>(s.n_samples - 1) / s.sample_rate - w.duration);
    const auto k = std::max<std::int64_t>(static_cast<std::int64_t>(std::ceil(opt_.range_cell_m / (0.75 * c / (2.0 * band)))), 1);
    const double step = opt_.range_cell_m / static_cast<double>(k);
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
        double p = 0.0;
        for (std::int64_t q = 0; q < k; ++q) p += static_cast<double>(fine_power[static_cast<std::size_t>(b * n_cells * k + j * k + q)]);
        last_power[static_cast<std::size_t>(b * n_cells + j)] = static_cast<float>(p / static_cast<double>(k));
      }
    }
  } else {
    // The far-field beamformers give every sample of the record: the part with a fully
    // overlapped replica is averaged into range cells of whole samples.
    const BeamMap map = opt_.beamformer == "mvdr" ? mvdr_beamform(record, bearings, opt_.mvdr)
                                                  : farfield_beamform(record, bearings, opt_.hann ? "hann" : "rect");
    const double sample_m = map.ranges[1] - map.ranges[0];
    const auto k = std::max<std::int64_t>(std::llround(opt_.range_cell_m / sample_m), 1);
    n_cells = std::min(map.n_valid, map.n_samples) / k;
    if (n_cells < 3) throw std::invalid_argument("LiveChain: the receive window is shorter than three range cells");
    last_power.assign(static_cast<std::size_t>(n_b * n_cells), 0.0F);
    last_ranges.assign(static_cast<std::size_t>(n_cells), 0.0);
    for (std::int64_t j = 0; j < n_cells; ++j) {
      double r = 0.0;
      for (std::int64_t q = 0; q < k; ++q) r += map.ranges[static_cast<std::size_t>(j * k + q)];
      last_ranges[static_cast<std::size_t>(j)] = r / static_cast<double>(k);
    }
    parallel_for_host(n_b, [&](std::int64_t b) {
      const std::complex<float>* row = map.row(b);
      for (std::int64_t j = 0; j < n_cells; ++j) {
        double p = 0.0;
        for (std::int64_t q = 0; q < k; ++q) p += static_cast<double>(std::norm(row[j * k + q]));
        last_power[static_cast<std::size_t>(b * n_cells + j)] = static_cast<float>(p / static_cast<double>(k));
      }
    });
  }
  const double cell = last_ranges[1] - last_ranges[0];
  detect::CfarOptions co;
  co.pfa = opt_.cfar_pfa;
  co.guard = std::max<std::int32_t>(static_cast<std::int32_t>(std::lround(opt_.cfar_guard_m / cell)), 1);
  co.train = std::max<std::int32_t>(static_cast<std::int32_t>(std::lround(opt_.cfar_train_m / cell)), 1);
  std::vector<unsigned char> det(last_power.size());
  std::vector<float> ratio(last_power.size());
  detect::cfar(opt_.backend, last_power.data(), n_b, n_cells, co, det.data(), ratio.data(), opt_.device);
  std::vector<Detection> hits;
  for (const Detection& h : cluster(det.data(), last_power.data(), ratio.data(), n_b, n_cells, last_ranges, bearings)) {
    // A reference summed into hydrophone inputs leaves its bursts in the data near both ends of
    // the window: no detections are taken there.
    if (burst_lo >= 0.0 && (h.range <= burst_lo || h.range >= burst_hi)) continue;
    if (h.snr_db > opt_.min_snr_db) hits.push_back(h);
  }

  if (opt_.doa == "sbl" && !hits.empty()) {
    // The map says at which ranges something is; SBL says from which bearings, at each of them.
    const std::vector<double> u = element_offsets(record);
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
    std::vector<char> taken(by_snr.size(), 0);
    std::vector<Detection> resolved;
    std::int64_t gates = 0;
    for (std::size_t i = 0; i < by_snr.size(); ++i) {
      if (taken[i]) continue;
      const Detection& h = by_snr[i];
      std::vector<std::size_t> group;
      for (std::size_t j = i; j < by_snr.size(); ++j) {
        if (!taken[j] && std::abs(by_snr[j].range - h.range) < gate_m) group.push_back(j);
      }
      for (std::size_t j : group) taken[j] = 1;
      std::vector<DoaPeak> peaks;
      if (gates < opt_.sbl_max_gates) {
        ++gates;
        const DoaFrame frame = gate_frame(record, 2.0 * h.range / c, n_fft);
        // The conventional spectrum of the gate costs one pass; SBL is only worth running where
        // that spectrum stands clear of its own median, which a diffuse field's does not.
        const DoaResult look = conventional(frame, u, c, sbl_grid, opt_.sbl);
        std::vector<float> sorted = look.spectrum;
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2), sorted.end());
        const double median = static_cast<double>(sorted[sorted.size() / 2]);
        const double peak = static_cast<double>(*std::max_element(look.spectrum.begin(), look.spectrum.end()));
        if (median > 0.0 && 10.0 * std::log10(peak / median) >= opt_.sbl_min_contrast_db) peaks = sbl(frame, u, c, sbl_grid, opt_.sbl).peaks;
      }
      if (peaks.empty()) {
        for (std::size_t j : group) resolved.push_back(by_snr[j]);
      } else {
        for (const DoaPeak& p : peaks) resolved.push_back({h.range, p.bearing, p.snr_db, 1});
      }
    }
    hits = std::move(resolved);
  }

  // World position of each detection, from the array's own position and heading in the record.
  const Frame f = array_frame(record);
  const Vec3d span = f.positions.back() - f.positions.front();
  const Vec3d axis = normalized(Vec3d{span.x, span.y, 0.0});
  const double side = opt_.starboard ? 1.0 : -1.0;
  const Vec3d broadside{-side * axis.y, side * axis.x, 0.0};
  if (!have_origin_) {
    origin_ = f.origin;
    t0_ns_ = record.transmit.time_ns;
    have_origin_ = true;
  }
  const double t = static_cast<double>(record.transmit.time_ns - t0_ns_) * 1e-9;
  std::vector<Detection> local;
  for (const Detection& h : hits) {
    const double sb = std::sin(h.bearing), cb = std::cos(h.bearing);
    const double px = f.origin.x - origin_.x + h.range * (sb * axis.x + cb * broadside.x);
    const double py = f.origin.y - origin_.y + h.range * (sb * axis.y + cb * broadside.y);
    // The tracker measures (range, bearing) about the frame origin; its state is (east, north).
    local.push_back({std::hypot(px, py), std::atan2(py, px), h.snr_db, h.cells});
    rep.detections.push_back({h.range, h.bearing / kDeg, h.snr_db});
  }
  const auto confirmed = tracker_.update(t, local);
  ClassifyOptions co2 = opt_.classify;
  co2.origin_x = f.origin.y - origin_.y;
  co2.origin_y = f.origin.x - origin_.x;
  for (const TrackClass& tc : classify_tracks(confirmed, co2)) {
    rep.tracks.push_back({tc.id, tc.parent, kind_name(tc.kind), tc.y + origin_.x, tc.x + origin_.y, tc.vy, tc.vx, tc.speed,
                          std::hypot(tc.y - co2.origin_y, tc.x - co2.origin_x), tc.mean_snr_db, tc.seen});
  }
  rep.ownship = f.origin;
  rep.processing_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  return rep;
}

std::string Report::json() const {
  std::ostringstream s;
  s << std::setprecision(10);
  s << "{\"ping\": " << ping << ", \"sequence\": " << sequence << ", \"time_ns\": " << time_ns << ", \"ownship_xyz\": ["
    << ownship.x << ", " << ownship.y << ", " << ownship.z << "], \"detections\": [";
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const auto& d = detections[i];
    s << (i ? ", " : "") << "{\"range_m\": " << d.range_m << ", \"bearing_deg\": " << d.bearing_deg << ", \"snr_db\": ";
    num(s, d.snr_db);
    s << "}";
  }
  s << "], \"tracks\": [";
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const auto& t = tracks[i];
    s << (i ? ", " : "") << "{\"id\": " << t.id << ", \"kind\": \"" << t.kind << "\", \"parent\": ";
    if (t.parent < 0) s << "null"; else s << t.parent;
    s << ", \"x\": " << t.x << ", \"y\": " << t.y << ", \"vx\": " << t.vx << ", \"vy\": " << t.vy << ", \"speed_m_s\": " << t.speed
      << ", \"range_m\": " << t.range_m << ", \"mean_snr_db\": ";
    num(s, t.mean_snr_db);
    s << ", \"seen\": " << t.seen << "}";
  }
  s << "], \"sync\": ";
  if (!synced) {
    s << "null";
  } else {
    s << "{\"ok\": " << (sync_ok ? "true" : "false") << ", \"worst_residual_us\": " << sync_worst_residual_us << ", \"offsets_us\": [";
    for (std::size_t i = 0; i < sync_offsets_us.size(); ++i) s << (i ? ", " : "") << sync_offsets_us[i];
    s << "], \"drift_ppm\": [";
    for (std::size_t i = 0; i < sync_drift_ppm.size(); ++i) s << (i ? ", " : "") << sync_drift_ppm[i];
    s << "]}";
  }
  s << ", \"processing_seconds\": " << processing_seconds << "}";
  return s.str();
}

std::vector<PassiveFrame> passive_bearings(const PingRecord& record, const PassiveOptions& o, std::vector<double>* bearings_out) {
  const Waveform& w = record.transmit.waveform;
  double f_low = o.f_low, f_high = o.f_high;
  if (f_low <= 0.0 && f_high <= 0.0) {
    f_low = std::min(w.f_start, w.f_stop);
    f_high = std::max(w.f_start, w.f_stop);
  }
  const std::vector<double> bearings = grid(o.bearing_first_deg, o.bearing_last_deg, o.bearing_step_deg);
  if (bearings_out) *bearings_out = bearings;
  const std::vector<double> u = element_offsets(record);
  const std::vector<DoaFrame> frames = passive_frames(record, o.n_fft, o.hop, o.n_average, f_low, f_high);
  std::vector<PassiveFrame> out(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {    // each solve is already parallel over frequency
    const DoaResult r = o.use_sbl ? sbl(frames[i], u, record.environment.sound_speed, bearings, o.sbl)
                                  : conventional(frames[i], u, record.environment.sound_speed, bearings, o.sbl);
    out[i] = {frames[i].time, r.peaks, r.spectrum, r.noise};
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
    s << "]}\n";
  }
  return s.str();
}

}  // namespace sonar::live
