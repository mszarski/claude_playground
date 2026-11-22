#include "sonar/live/detector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/detect/statistic.hpp"
#include "sonar/live/detector_kernel.hpp"
#include "sonar/dsp/pulse_compression.hpp"

namespace sonar::live {
namespace {

constexpr double kPi = 3.14159265358979323846;
// How finely the replica is sampled for its autocorrelation (sample_correlation): this many samples per 1 / band, and
// at least this many over the pulse (a tone has no band to go by). Sampling choices, not properties of a sonar.
constexpr double kReplicaSamplesPerBand = 4.0, kReplicaSamplesPerPulse = 256.0;
// No test is run at a false-alarm probability above this, however loose the budget (as before the table).
constexpr double kLargestTestProbability = 0.1;

}  // namespace

std::vector<double> sample_correlation(const Waveform& w, double sound_speed, double step, std::int64_t n) {
  // White noise through the matched filter has the replica's own power spectrum, so its amplitude correlation between
  // two map samples is the replica's autocorrelation at their two-way time apart. The replica is the one the chain
  // compresses with (dsp::make_replica: the pulse with its taper, at baseband about the band's centre, where a
  // symmetric pulse's autocorrelation is real), sampled a whole number of times per map sample and finely enough for
  // the band and for the pulse's length. A flat band would give sinc(2 B x / c) and a rectangular tone the triangle of
  // its length; a swept pulse with a 10 % Tukey taper is 6 % narrower in band than that sinc assumes.
  std::vector<double> r(static_cast<std::size_t>(std::max<std::int64_t>(n, 1)), 0.0);
  r[0] = 1.0;
  const double tau = 2.0 * step / sound_speed;
  if (!(tau > 0.0 && w.duration > 0.0) || r.size() == 1) return r;
  const double band = std::abs(w.f_stop - w.f_start);
  const double finest = std::max(kReplicaSamplesPerBand * band, kReplicaSamplesPerPulse / w.duration);
  const auto per_step = static_cast<std::int64_t>(std::max(std::ceil(finest * tau), 1.0));
  const std::vector<std::complex<float>> h = dsp::make_replica(w, static_cast<double>(per_step) / tau, w.center());
  double energy = 0.0;
  for (const std::complex<float>& v : h) energy += std::norm(std::complex<double>(v));
  if (!(energy > 0.0)) return r;
  for (std::size_t d = 1; d < r.size(); ++d) {
    const std::size_t lag = d * static_cast<std::size_t>(per_step);
    std::complex<double> sum{0.0, 0.0};
    for (std::size_t k = lag; k < h.size(); ++k) sum += std::complex<double>(h[k]) * std::conj(std::complex<double>(h[k - lag]));
    r[d] = sum.real() / energy;
  }
  return r;
}

double DetectorPlan::threshold(std::size_t s, double count) const {
  const DetectorScale& sc = scales.at(s);
  return sc.factor * detect::training_correction(sc.pfa, count * train_looks_per_cell);
}

void check_plan(const DetectorPlan& plan) {
  if (plan.scales.empty() || plan.scales[0].cells != 0) {
    throw std::invalid_argument("detector plan: its first scale must be the single-sample test (cells 0); plan_detector puts it there");
  }
  for (std::size_t s = 1; s < plan.scales.size(); ++s) {
    if (plan.scales[s].cells < 1) throw std::invalid_argument("detector plan: scales after the first are means of one cell or more");
  }
}

double DetectorPlan::sample_factor() const {
  check_plan(*this);
  return scales[0].factor;
}

RangeEstimate parse_range_estimate(const std::string& name) {
  if (name == "centroid") return RangeEstimate::Centroid;
  if (name == "first") return RangeEstimate::First;
  throw std::invalid_argument("cfar range_estimate '" + name + "': one of centroid, first");
}

const char* range_estimate_name(RangeEstimate r) { return r == RangeEstimate::First ? "first" : "centroid"; }

std::string DetectorPlan::json() const {
  std::ostringstream s;
  s << std::setprecision(6);
  s << "{\"rows\": " << rows << ", \"samples_per_cell\": " << samples_per_cell << ", \"cell_m\": " << cell_m << ", \"step_m\": " << step_m
    << ", \"resolution_m\": " << resolution_m << ", \"beams\": " << beams << ", \"train_looks_per_cell\": " << train_looks_per_cell
    << ", \"merge_cells\": " << merge_cells << ", \"merge_deg\": " << std::asin(std::min(merge_u, 1.0)) * 180.0 / kPi << ", \"sidelobe_margin_db\": " << 10.0 * std::log10(margin) << ", \"source_snr_db\": " << 10.0 * std::log10(source_ratio) << ", \"scales\": [";
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const DetectorScale& c = scales[i];
    s << (i ? ", " : "") << "{\"cells\": " << c.cells << ", \"length_m\": " << c.length_m << ", \"looks\": " << c.looks << ", \"tests\": " << c.tests
      << ", \"search\": " << c.search
      << ", \"pfa\": " << c.pfa << ", \"threshold_db\": " << 10.0 * std::log10(c.factor) << "}";
  }
  s << "]";
  // Said only when not the default, so that the default plan's line is the one it was.
  if (range_estimate != RangeEstimate::Centroid) s << ", \"range_estimate\": \"" << range_estimate_name(range_estimate) << "\"";
  s << "}";
  return s.str();
}

DetectorPlan plan_detector(const PingRecord& record, const SonarFacts& facts, const DetectorOptions& options, std::int64_t n_cells,
                           std::int64_t samples_per_cell, double step, const std::vector<double>& bearings, std::int64_t rows,
                           std::int64_t guard_cells, std::int64_t train_cells, bool hann, double merge_m,
                           const std::vector<std::vector<std::int64_t>>& lines) {
  if (n_cells < 1 || samples_per_cell < 1 || !(step > 0.0) || bearings.empty() || rows < 1 || train_cells < 1) {
    throw std::invalid_argument("plan_detector: an empty map");
  }
  if (!(options.false_alarms_per_ping > 0.0)) throw std::invalid_argument("cfar.false_alarms_per_ping must be positive");
  const Waveform& w = record.transmit.waveform;
  const double c = facts.sound_speed;
  DetectorPlan p;
  p.range_estimate = options.range_estimate;
  p.rows = rows;
  p.samples_per_cell = samples_per_cell;
  p.step_m = step;
  p.cell_m = step * static_cast<double>(samples_per_cell);
  p.resolution_m = facts.range_resolution;
  p.margin = std::pow(10.0, options.sidelobe_margin_db / 10.0);
  p.aperture_m = facts.aperture;
  p.merge_cells = merge_m > 0.0 ? static_cast<std::int64_t>(std::floor(merge_m / p.cell_m + 1e-9)) : 0;
  // Independent beams across the sector: its width in sin(bearing) over one beamwidth (wavelength / aperture).
  const double sector = std::abs(std::sin(bearings.back()) - std::sin(bearings.front()));
  p.beams = std::clamp(facts.beamwidth > 0.0 ? sector / facts.beamwidth + 1.0 : 1.0, 1.0, static_cast<double>(bearings.size()));

  std::vector<std::int64_t> lengths = options.integrate_cells;
  if (lengths.empty()) lengths = {1};
  std::sort(lengths.begin(), lengths.end());
  lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
  std::vector<std::int64_t> kept;
  for (std::int64_t m : lengths) {
    if (m < 1) throw std::invalid_argument("cfar.integrate_cells: lengths of one cell or more");
    // A length the guard does not cover would have its own echo in its background; one the map cannot hold is no test.
    if (m == 1 || (m <= guard_cells && 2 * m <= n_cells)) kept.push_back(m);
  }
  const std::int64_t longest = kept.empty() ? 1 : kept.back();
  const std::vector<double> corr = sample_correlation(w, c, step, longest * samples_per_cell);
  const double extent = p.cell_m * static_cast<double>(n_cells);
  const auto n_scales = static_cast<double>(kept.size() + 1);
  const auto add = [&](std::int64_t cells, double length, std::vector<double> weights) {
    DetectorScale s;
    s.cells = cells;
    s.length_m = length;
    s.looks = detect::equivalent_looks(weights);
    const double places = p.beams * extent / std::max(length, p.resolution_m);   // range resolutions (or cells) x beamwidths
    const double share = options.false_alarms_per_ping / n_scales;
    if (is_given(options.search_factor)) {
      s.search = options.search_factor;
      s.tests = std::max(s.search * places, 1.0);
      s.pfa = std::min(share / s.tests, kLargestTestProbability);
      s.factor = detect::weighted_exponential_threshold(weights, rows, s.pfa);
    } else {
      // The tests grow with the threshold (detector.hpp, kSearchSample): the threshold x at which
      // tail(x) x tests(x) is this test's share of the budget. The product falls with x beyond the statistic's mean.
      const auto row = static_cast<std::size_t>(std::min<std::int64_t>(rows, 3) - 1);
      s.search = cells == 0 ? kSearchSample[row] : kSearchMean[row];
      const double a = static_cast<double>(rows) * s.looks;
      const auto count = [&](double x) {
        const double u = a * x;
        double growth = u;
        if (cells > 0) {
          // The hazard of the statistic itself, per unit of u: -d ln P(> x) / dx / a.
          const double h = 1e-4 * x;
          const double hazard = (std::log(detect::weighted_exponential_tail(weights, rows, x - h)) -
                                 std::log(detect::weighted_exponential_tail(weights, rows, x + h))) / (2.0 * h * a);
          growth = std::sqrt(u) * std::max(hazard, 0.0);
        }
        return std::max(s.search * places * growth, 1.0);
      };
      double lo = detect::weighted_exponential_threshold(weights, rows, kLargestTestProbability), hi = lo;
      if (detect::weighted_exponential_tail(weights, rows, lo) * count(lo) > share) {
        while (detect::weighted_exponential_tail(weights, rows, hi) * count(hi) > share) hi *= 2.0;
        for (int it = 0; it < 80; ++it) {
          const double mid = 0.5 * (lo + hi);
          (detect::weighted_exponential_tail(weights, rows, mid) * count(mid) > share ? lo : hi) = mid;
        }
      }
      s.factor = 0.5 * (lo + hi);
      s.tests = count(s.factor);
      s.pfa = detect::weighted_exponential_tail(weights, rows, s.factor);
    }
    s.weights = std::move(weights);
    p.scales.push_back(std::move(s));
  };
  add(0, p.resolution_m, {1.0});
  for (std::int64_t m : kept) add(m, p.cell_m * static_cast<double>(m), detect::mean_power_weights(corr, m * samples_per_cell));
  p.train_looks_per_cell = static_cast<double>(rows) * detect::equivalent_looks(detect::mean_power_weights(corr, samples_per_cell));

  // Range sidelobes of the compressed pulse, by whole cells between two cells' nearest edges.
  const bool swept = std::abs(w.f_stop - w.f_start) > 0.0;
  p.range_envelope.assign(static_cast<std::size_t>(n_cells + 1), 1.0);
  for (std::int64_t d = 2; d <= n_cells; ++d) {
    const double x = static_cast<double>(d - 1) * p.cell_m / p.resolution_m;
    p.range_envelope[static_cast<std::size_t>(d)] = x < 1.0 ? 1.0 : (swept ? std::min(1.0, 1.0 / (kPi * kPi * x * x)) : 1e-12);
  }
  // Bearing sidelobes: the pattern of each line of hydrophones as the record has them (a dead board is a gap), with the
  // map's aperture window laid as the map lays it, averaged over the band (which fills the nulls) and over the lines
  // (their maps are added in power).
  std::vector<std::vector<std::int64_t>> use = lines;
  if (use.empty()) {
    use.emplace_back();
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(record.geometry.rx.size()); ++i) use[0].push_back(i);
  }
  const std::int64_t n_u = 2048;
  p.u_step = 2.0 / static_cast<double>(n_u);
  std::vector<double> pattern(static_cast<std::size_t>(n_u + 1), 0.0);
  const int n_f = 9;
  for (const std::vector<std::int64_t>& line : use) {
    const auto n_el = line.size();
    if (n_el == 0) throw std::invalid_argument("plan_detector: an empty line of hydrophones");
    const auto at = [&](std::size_t i) { return record.geometry.rx.at(static_cast<std::size_t>(line[i])).lever_arm; };
    Vec3d centre{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < n_el; ++i) centre = centre + at(i) / static_cast<double>(n_el);
    const Vec3d end = at(n_el - 1) - at(0);
    const Vec3d axis = n_el > 1 && dot(end, end) > 0.0 ? normalized(end) : Vec3d{1.0, 0.0, 0.0};
    std::vector<double> x_el(n_el), weight(n_el, 1.0);
    double lo = 1e300, hi = -1e300, w_sum = 0.0;
    for (std::size_t i = 0; i < n_el; ++i) {
      x_el[i] = dot(at(i) - centre, axis);
      lo = std::min(lo, x_el[i]);
      hi = std::max(hi, x_el[i]);
    }
    const double span = hi - lo, pitch = span / static_cast<double>(std::max<std::size_t>(n_el - 1, 1));
    for (std::size_t i = 0; i < n_el; ++i) {
      if (hann && span > 0.0) {
        const double cw = std::cos(kPi * x_el[i] / (span + pitch));
        weight[i] = cw * cw;
      }
      w_sum += weight[i];
    }
    for (std::int64_t iu = 0; iu <= n_u; ++iu) {
      const double u = p.u_step * static_cast<double>(iu);
      double acc = 0.0;
      for (int q = 0; q < n_f; ++q) {
        const double f = facts.f_low + (facts.f_high - facts.f_low) * (static_cast<double>(q) + 0.5) / n_f;
        std::complex<double> z(0.0, 0.0);
        for (std::size_t i = 0; i < n_el; ++i) z += weight[i] * std::polar(1.0, 2.0 * kPi * f / c * x_el[i] * u);
        acc += std::norm(z) / (w_sum * w_sum);
      }
      pattern[static_cast<std::size_t>(iu)] += acc / n_f / static_cast<double>(use.size());
    }
  }
  // Cells closer in bearing than half the half-power beamwidth are one echo's beam (a weak echo's beam can be flagged in
  // two pieces); never less than one step of the bearing grid, so touching cells always belong together.
  std::int64_t half = 1;
  while (half < n_u && pattern[static_cast<std::size_t>(half)] >= 0.5) ++half;
  p.merge_u = p.u_step * static_cast<double>(half);
  for (std::size_t b = 1; b < bearings.size(); ++b) p.merge_u = std::max(p.merge_u, std::abs(std::sin(bearings[b]) - std::sin(bearings[b - 1])));
  // The detection itself is the half-power part of the beam (and the cells next to it in range): the envelope is 1
  // there. Beyond, it is the beam's own skirt and then its sidelobe peaks (the largest value from here to one
  // beamwidth further out, so that the nulls between sidelobes are bridged), never below the floor. Two echoes at one
  // range are therefore two detections when their half-power beams are further apart than merge_u: more than one and
  // a half half-power beamwidths between them.
  const std::int64_t main = half;
  const auto lobe = std::max<std::int64_t>(std::llround(facts.beamwidth / p.u_step), 1);
  const double floor = std::pow(10.0, options.sidelobe_floor_db / 10.0);
  p.bearing_envelope.assign(pattern.size(), 1.0);
  double out_max = floor;
  for (std::int64_t iu = main; iu <= n_u; ++iu) {
    double top = 0.0;
    for (std::int64_t q = iu; q <= std::min(iu + lobe, n_u); ++q) top = std::max(top, pattern[static_cast<std::size_t>(q)]);
    p.bearing_envelope[static_cast<std::size_t>(iu)] = std::clamp(top, floor, 1.0);
    out_max = std::max(out_max, p.bearing_envelope[static_cast<std::size_t>(iu)]);
  }
  out_max = std::max(out_max, n_cells >= 2 ? p.range_envelope[2] : 0.0);
  p.source_ratio = p.scales[0].factor / (p.margin * std::min(out_max, 1.0));
  return p;
}

namespace {

double median(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t h = v.size() / 2;
  return v.size() % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
}

// Training cells of each range cell (the same for every bearing): [j - guard - train, j - guard) and (j + guard, j + guard + train];
// the offset of their centroid from the cell, and what the estimate is worth in independent training cells.
struct BackgroundWindow {
  std::vector<std::int64_t> lo0, lo1, hi0, hi1;
  std::vector<double> offset;
  bool sloped = false;
};

BackgroundWindow background_window(std::int64_t n_cells, std::int64_t guard, std::int64_t train, std::vector<double>& train_cells) {
  BackgroundWindow bw;
  train_cells.assign(static_cast<std::size_t>(n_cells), 1.0);
  bw.lo0.assign(static_cast<std::size_t>(n_cells), 0);
  bw.lo1 = bw.lo0;
  bw.hi0 = bw.lo0;
  bw.hi1 = bw.lo0;
  bw.offset.assign(static_cast<std::size_t>(n_cells), 0.0);     // of the training cells' centroid from the cell, cells
  std::vector<std::int64_t>&lo0 = bw.lo0, &lo1 = bw.lo1, &hi0 = bw.hi0, &hi1 = bw.hi1;
  std::vector<double>& offset = bw.offset;
  const bool sloped = n_cells >= 2 * train && train >= 8;
  bw.sloped = sloped;
  const auto t = static_cast<double>(train);
  for (std::int64_t j = 0; j < n_cells; ++j) {
    const auto q = static_cast<std::size_t>(j);
    lo0[q] = std::clamp<std::int64_t>(j - guard - train, 0, n_cells);
    lo1[q] = std::clamp<std::int64_t>(j - guard, 0, n_cells);
    hi0[q] = std::clamp<std::int64_t>(j + guard + 1, 0, n_cells);
    hi1[q] = std::clamp<std::int64_t>(j + guard + 1 + train, 0, n_cells);
    const auto n_lo = static_cast<double>(lo1[q] - lo0[q]), n_hi = static_cast<double>(hi1[q] - hi0[q]);
    const double count = std::max(n_lo + n_hi, 1.0);
    const auto g = static_cast<double>(guard);
    if (sloped) offset[q] = (n_hi * (g + 0.5 * (n_hi + 1.0)) - n_lo * (g + 0.5 * (n_lo + 1.0))) / count;
    // What the slope costs: the two medians it is the ratio of (pi / 2 of a mean's variance each), and that one of
    // them shares its cells with the mean.
    const double r = std::abs(offset[q]) / t;
    train_cells[q] = 1.0 / (1.0 / count + (3.14159265358979323846 * r * r + 2.0 * r) / t);
  }
  return bw;
}

}  // namespace

void detector_background(const float* cell_mean, std::int64_t n_b, std::int64_t n_cells, std::int64_t guard, std::int64_t train,
                         std::vector<double>& background, std::vector<double>& train_cells) {
  background.assign(static_cast<std::size_t>(n_b * n_cells), 0.0);
  const BackgroundWindow bw = background_window(n_cells, guard, train, train_cells);
  const std::vector<std::int64_t>&lo0 = bw.lo0, &lo1 = bw.lo1, &hi0 = bw.hi0, &hi1 = bw.hi1;
  const std::vector<double>& offset = bw.offset;
  const bool sloped = bw.sloped;
  const auto t = static_cast<double>(train);
  std::vector<double> csum(static_cast<std::size_t>(n_cells + 1)), block;
  for (std::int64_t b = 0; b < n_b; ++b) {
    const float* row = cell_mean + b * n_cells;
    csum[0] = 0.0;
    for (std::int64_t j = 0; j < n_cells; ++j) csum[static_cast<std::size_t>(j + 1)] = csum[static_cast<std::size_t>(j)] + static_cast<double>(row[j]);
    double slope_near = 0.0, slope_far = 0.0;                             // d ln(power) / d cell, positive when it falls with range
    if (sloped) {
      const auto med = [&](std::int64_t first) {
        block.assign(row + first, row + first + train);
        return median(block);
      };
      const double a0 = med(0), a1 = med(train), z0 = med(n_cells - 2 * train), z1 = med(n_cells - train);
      if (a0 > 0.0 && a1 > 0.0) slope_near = std::log(a0 / a1) / t;
      if (z0 > 0.0 && z1 > 0.0) slope_far = std::log(z0 / z1) / t;
    }
    for (std::int64_t j = 0; j < n_cells; ++j) {
      const auto q = static_cast<std::size_t>(j);
      const std::int64_t count = std::max<std::int64_t>((lo1[q] - lo0[q]) + (hi1[q] - hi0[q]), 1);
      double mean = (csum[static_cast<std::size_t>(lo1[q])] - csum[static_cast<std::size_t>(lo0[q])] + csum[static_cast<std::size_t>(hi1[q])] -
                     csum[static_cast<std::size_t>(hi0[q])]) / static_cast<double>(count);
      // The training cells' centroid lies `offset` cells beyond the cell: the level there is exp(-slope offset) of the cell's.
      if (offset[q] != 0.0) mean *= std::exp(std::clamp((offset[q] > 0.0 ? slope_near : slope_far) * offset[q], -2.3, 2.3));
      background[static_cast<std::size_t>(b * n_cells + j)] = mean;
    }
  }
}

#if SONAR_HAVE_CUDA
// detector_cuda.cu: the kernels of detector_kernel.hpp on the device.
void multiscale_cells_cuda(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean, std::int64_t n_b,
                           std::int64_t n_cells, std::int64_t guard, std::int64_t train, bool sloped, const std::vector<double>& offset,
                           const std::vector<double>& threshold, const std::vector<std::int64_t>& cells, std::int64_t k, double* level,
                           float* background, float* peak, unsigned char* flag, int device);
#endif

namespace {

// Each test's threshold over the background by range cell, [n_scales * n_cells] (multiscale_detect's thr[s][j]), and the
// background window's geometry.
BackgroundWindow cell_thresholds(const DetectorPlan& plan, std::int64_t n_cells, std::int64_t guard, std::int64_t train,
                                 std::vector<double>& threshold, std::vector<std::int64_t>& cells) {
  std::vector<double> worth;
  BackgroundWindow bw = background_window(n_cells, guard, train, worth);
  threshold.assign(plan.scales.size() * static_cast<std::size_t>(n_cells), 0.0);
  cells.clear();
  for (std::size_t s = 0; s < plan.scales.size(); ++s) {
    cells.push_back(plan.scales[s].cells);
    for (std::int64_t j = 0; j < n_cells; ++j) {
      threshold[s * static_cast<std::size_t>(n_cells) + static_cast<std::size_t>(j)] = plan.threshold(s, worth[static_cast<std::size_t>(j)]);
    }
  }
  return bw;
}

}  // namespace

void multiscale_cells_kernels_on_host(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean,
                                      std::int64_t n_b, std::int64_t n_cells, std::int64_t guard, std::int64_t train,
                                      const DetectorPlan& plan, DetectorCells& out) {
  check_plan(plan);
  std::vector<double> threshold;
  std::vector<std::int64_t> cells;
  const BackgroundWindow bw = cell_thresholds(plan, n_cells, guard, train, threshold, cells);
  const auto n = static_cast<std::size_t>(n_b * n_cells);
  std::vector<double> scratch(bw.sloped ? static_cast<std::size_t>(n_b * 4 * train) : 0U), median(static_cast<std::size_t>(4 * n_b), 0.0);
  std::vector<double> csum(static_cast<std::size_t>(n_b * (n_cells + 1))), slope(static_cast<std::size_t>(2 * n_b));
  out.level.assign(n, 0.0);
  out.background.assign(n, 0.0F);
  out.peak.assign(n, 0.0F);
  out.flag.assign(n, 0);
  if (bw.sloped) parallel_for_host(4 * n_b, DetectorMedianKernel{cell_mean, scratch.data(), median.data(), n_cells, train});
  parallel_for_host(n_b, DetectorRowKernel{cell_mean, median.data(), csum.data(), slope.data(), n_cells, train, bw.sloped});
  parallel_for_host(n_b * n_cells, DetectorCellKernel{fine, fine_stride, fine_offset, csum.data(), slope.data(), bw.offset.data(), threshold.data(),
                                                      cells.data(), static_cast<std::int64_t>(cells.size()), n_cells, plan.samples_per_cell, guard,
                                                      train, out.level.data(), out.background.data(), out.peak.data(), out.flag.data()});
}

void multiscale_cells(Backend backend, const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean,
                      std::int64_t n_b, std::int64_t n_cells, std::int64_t guard, std::int64_t train, const DetectorPlan& plan,
                      DetectorCells& out, int device) {
  (void)device;  // used by the CUDA backend only
  check_plan(plan);
  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    std::vector<double> threshold;
    std::vector<std::int64_t> cells;
    const BackgroundWindow bw = cell_thresholds(plan, n_cells, guard, train, threshold, cells);
    const auto n = static_cast<std::size_t>(n_b * n_cells);
    out.level.assign(out.with_level ? n : 0U, 0.0);
    out.background.assign(n, 0.0F);
    out.peak.assign(n, 0.0F);
    out.flag.assign(n, 0);
    multiscale_cells_cuda(fine, fine_stride, fine_offset, cell_mean, n_b, n_cells, guard, train, bw.sloped, bw.offset, threshold, cells,
                          plan.samples_per_cell, out.with_level ? out.level.data() : nullptr, out.background.data(), out.peak.data(),
                          out.flag.data(), device);
    return;
#else
    throw std::runtime_error("multiscale_cells: built without the CUDA backend");
#endif
  }
  // The reference.
  const std::int64_t k = plan.samples_per_cell;
  const auto n = static_cast<std::size_t>(n_b * n_cells);
  std::vector<double>& level = out.level;
  std::vector<double> worth;
  detector_background(cell_mean, n_b, n_cells, guard, train, level, worth);
  std::vector<float>& background = out.background;
  background.assign(n, 0.0F);
  std::vector<float>& peak = out.peak;
  peak.assign(n, 0.0F);
  std::vector<unsigned char>& flag = out.flag;
  flag.assign(n, 0);
  std::vector<std::vector<double>> thr(plan.scales.size(), std::vector<double>(static_cast<std::size_t>(n_cells)));
  for (std::int64_t j = 0; j < n_cells; ++j) {
    for (std::size_t s = 0; s < plan.scales.size(); ++s) thr[s][static_cast<std::size_t>(j)] = plan.threshold(s, worth[static_cast<std::size_t>(j)]);
  }
  std::vector<double> csum(static_cast<std::size_t>(n_cells + 1));
  for (std::int64_t b = 0; b < n_b; ++b) {
    const float* row = cell_mean + b * n_cells;
    const float* fine_row = fine + b * fine_stride + fine_offset;
    csum[0] = 0.0;
    for (std::int64_t j = 0; j < n_cells; ++j) csum[static_cast<std::size_t>(j + 1)] = csum[static_cast<std::size_t>(j)] + static_cast<double>(row[j]);
    for (std::int64_t j = 0; j < n_cells; ++j) {
      const auto q = static_cast<std::size_t>(j);
      const auto i = static_cast<std::size_t>(b * n_cells + j);
      const double mean = level[i];
      background[i] = static_cast<float>(mean);
      float top = 0.0F;
      for (std::int64_t s = 0; s < k; ++s) top = std::max(top, fine_row[j * k + s]);
      peak[i] = top;
      const double floor = std::max(mean, 1e-300);
      bool hit = static_cast<double>(top) > thr[0][q] * floor;
      for (std::size_t s = 1; s < plan.scales.size() && !hit; ++s) {
        const std::int64_t m = plan.scales[s].cells, a = j - (m - 1) / 2, e = a + m;
        if (a < 0 || e > n_cells) continue;
        hit = (csum[static_cast<std::size_t>(e)] - csum[static_cast<std::size_t>(a)]) / static_cast<double>(m) > thr[s][q] * floor;
      }
      flag[i] = hit ? 1 : 0;
    }
  }
}

std::vector<Detection> multiscale_detect(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean,
                                         std::int64_t n_b, std::int64_t n_cells, const std::vector<double>& fine_ranges,
                                         const std::vector<double>& cell_ranges, const std::vector<double>& bearings,
                                         std::int64_t guard, std::int64_t train, const DetectorPlan& plan, std::vector<float>& background,
                                         Backend backend, int device) {
  DetectorCells statistic;
  statistic.with_level = false;
  multiscale_cells(backend, fine, fine_stride, fine_offset, cell_mean, n_b, n_cells, guard, train, plan, statistic, device);
  return multiscale_group(fine, fine_stride, fine_offset, n_b, n_cells, fine_ranges, cell_ranges, bearings, plan, statistic, background);
}

std::vector<Detection> multiscale_group(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, std::int64_t n_b,
                                        std::int64_t n_cells, const std::vector<double>& fine_ranges, const std::vector<double>& cell_ranges,
                                        const std::vector<double>& bearings, const DetectorPlan& plan, DetectorCells& statistic,
                                        std::vector<float>& background) {
  // The arrivals and the leading edge are held to the single-sample threshold (sonar-d99: never another scale's).
  const double sample_factor = plan.sample_factor();
  const std::int64_t k = plan.samples_per_cell;
  const auto n = static_cast<std::size_t>(n_b * n_cells);
  background = statistic.background;
  const std::vector<float>& peak = statistic.peak;
  std::vector<unsigned char>& flag = statistic.flag;

  // Cells that a stronger cell's sidelobes explain are dropped, strongest source first.
  std::vector<std::int64_t> flagged, sources;
  for (std::int64_t i = 0; i < n_b * n_cells; ++i) {
    if (!flag[static_cast<std::size_t>(i)]) continue;
    flagged.push_back(i);
    if (static_cast<double>(peak[static_cast<std::size_t>(i)]) >= plan.source_ratio * static_cast<double>(background[static_cast<std::size_t>(i)])) sources.push_back(i);
  }
  std::stable_sort(sources.begin(), sources.end(), [&](std::int64_t a, std::int64_t b) { return peak[static_cast<std::size_t>(a)] > peak[static_cast<std::size_t>(b)]; });
  std::vector<double> sin_b(bearings.size());
  for (std::size_t b = 0; b < bearings.size(); ++b) sin_b[b] = std::sin(bearings[b]);
  const auto n_env = static_cast<std::int64_t>(plan.bearing_envelope.size());
  // A cell inside the core of a stronger source (its half-power beam, the next range cell) is that source's own: it is
  // not dropped by another source's sidelobe rule. Without this every strong cell of one echo ate into the others'
  // side of the beam and the group, and its bearing, came out lopsided. It is still a source itself: a second arrival a
  // cell behind the first has sidelobes of its own.
  std::vector<unsigned char> owned(n, 0);
  for (std::int64_t src : sources) {
    if (!flag[static_cast<std::size_t>(src)]) continue;
    const std::int64_t b0 = src / n_cells, j0 = src % n_cells;
    const double p0 = static_cast<double>(peak[static_cast<std::size_t>(src)]);
    for (std::int64_t i : flagged) {
      if (!flag[static_cast<std::size_t>(i)] || i == src) continue;
      const double pq = static_cast<double>(peak[static_cast<std::size_t>(i)]);
      if (pq >= p0) continue;
      const std::int64_t b = i / n_cells, j = i % n_cells;
      const double du = std::abs(sin_b[static_cast<std::size_t>(b)] - sin_b[static_cast<std::size_t>(b0)]);
      const double e_b = plan.bearing_envelope[static_cast<std::size_t>(std::min<std::int64_t>(std::llround(du / plan.u_step), n_env - 1))];
      // Off the source's bearing its energy is spread in range by the array's own length: that many cells are let off.
      const std::int64_t smear = static_cast<std::int64_t>(std::floor(0.25 * plan.aperture_m * du / plan.cell_m + 0.5));   // whole cells, to the nearest
      const std::int64_t dc = std::max<std::int64_t>(std::abs(j - j0) - smear, 0);
      const double e_r = plan.range_envelope[static_cast<std::size_t>(std::min<std::int64_t>(dc, n_cells))];
      if (e_r >= 1.0 && e_b >= 1.0) {                                // inside the core: part of this detection
        owned[static_cast<std::size_t>(i)] = 1;
        continue;
      }
      if (!owned[static_cast<std::size_t>(i)] && pq <= plan.margin * e_r * e_b * p0) flag[static_cast<std::size_t>(i)] = 0;
    }
  }

  // One detection per group of flagged cells that belong together: no further apart than merge_cells + 1 in range and
  // merge_u in sin(bearing). Touching cells (diagonals included) always are.
  std::vector<std::int64_t> b_lo(static_cast<std::size_t>(n_b)), b_hi(b_lo);
  for (std::int64_t b = 0; b < n_b; ++b) {
    std::int64_t lo = b, hi = b;
    while (lo > 0 && std::abs(sin_b[static_cast<std::size_t>(lo - 1)] - sin_b[static_cast<std::size_t>(b)]) <= plan.merge_u + 1e-12) --lo;
    while (hi < n_b - 1 && std::abs(sin_b[static_cast<std::size_t>(hi + 1)] - sin_b[static_cast<std::size_t>(b)]) <= plan.merge_u + 1e-12) ++hi;
    b_lo[static_cast<std::size_t>(b)] = lo;
    b_hi[static_cast<std::size_t>(b)] = hi;
  }
  const std::int64_t reach = plan.merge_cells + 1;
  std::vector<double> row_energy(static_cast<std::size_t>(n_b), 0.0);
  std::vector<std::int64_t> rows_touched;
  std::vector<std::pair<std::int64_t, double>> profile;                // (sample along range, power over background) of a group
  std::vector<Detection> out;
  std::vector<unsigned char> seen(n, 0);
  std::vector<std::int64_t> stack;
  for (std::int64_t start : flagged) {
    if (!flag[static_cast<std::size_t>(start)] || seen[static_cast<std::size_t>(start)]) continue;
    double w_sum = 0.0, r_sum = 0.0, best = 0.0, nearest = 1e300, farthest = -1e300;
    rows_touched.clear();
    profile.clear();
    std::int64_t cells = 0;
    stack.assign(1, start);
    seen[static_cast<std::size_t>(start)] = 1;
    while (!stack.empty()) {
      const std::int64_t i = stack.back();
      stack.pop_back();
      const std::int64_t row = i / n_cells, col = i % n_cells;
      const float* fine_row = fine + row * fine_stride + fine_offset + col * k;
      if (row_energy[static_cast<std::size_t>(row)] == 0.0) rows_touched.push_back(row);
      for (std::int64_t s = 0; s < k; ++s) {
        const auto w = static_cast<double>(fine_row[s]);
        w_sum += w;
        r_sum += w * fine_ranges[static_cast<std::size_t>(fine_offset + col * k + s)];
        row_energy[static_cast<std::size_t>(row)] += w;
        profile.emplace_back(col * k + s, w / std::max(static_cast<double>(background[static_cast<std::size_t>(i)]), 1e-300));
      }
      nearest = std::min(nearest, cell_ranges[static_cast<std::size_t>(col)]);
      farthest = std::max(farthest, cell_ranges[static_cast<std::size_t>(col)]);
      const double ratio = static_cast<double>(peak[static_cast<std::size_t>(i)]) / std::max(static_cast<double>(background[static_cast<std::size_t>(i)]), 1e-300);
      best = std::max(best, ratio);
      ++cells;
      for (std::int64_t r2 = b_lo[static_cast<std::size_t>(row)]; r2 <= b_hi[static_cast<std::size_t>(row)]; ++r2) {
        for (std::int64_t c2 = std::max<std::int64_t>(col - reach, 0); c2 <= std::min(col + reach, n_cells - 1); ++c2) {
          const std::int64_t j = r2 * n_cells + c2;
          if (flag[static_cast<std::size_t>(j)] && !seen[static_cast<std::size_t>(j)]) {
            seen[static_cast<std::size_t>(j)] = 1;
            stack.push_back(j);
          }
        }
      }
    }
    // Range: the energy-weighted mean of the group's samples. Bearing: the mean over its bearings weighted by what each
    // holds, over all its range cells, above half the strongest one's energy. The group ends at the beam's half-power
    // points on the bearing grid, and a plain energy-weighted mean over a beam cut there is pulled towards the grid;
    // weights that fall to nothing at the cut are not. (The energy, not the strongest sample: that scallops by a
    // decibel from beam to beam with where the echo falls between samples, which was 0.3 degrees of bearing.)
    // The level is half the strongest bearing's energy, or what the group's outermost bearings hold when that is more
    // (a beam wider than the modelled one is cut above its half-power points): the weights are zero at both ends.
    std::sort(rows_touched.begin(), rows_touched.end());
    double top = 0.0, bw_sum = 0.0, b_sum = 0.0, plain_sum = 0.0, plain_b = 0.0;
    for (std::int64_t r : rows_touched) top = std::max(top, row_energy[static_cast<std::size_t>(r)]);
    const double level = std::max({0.5 * top, row_energy[static_cast<std::size_t>(rows_touched.front())],
                                   row_energy[static_cast<std::size_t>(rows_touched.back())]});
    for (std::int64_t r : rows_touched) {
      const double e = row_energy[static_cast<std::size_t>(r)], w = std::max(e - level, 0.0);
      bw_sum += w;
      b_sum += w * bearings[static_cast<std::size_t>(r)];
      plain_sum += e;
      plain_b += e * bearings[static_cast<std::size_t>(r)];
      row_energy[static_cast<std::size_t>(r)] = 0.0;
    }
    // One or two bearings, or the strongest at an end: nothing stands above the level, the plain mean is all there is.
    Detection d{r_sum / w_sum, bw_sum > 0.0 ? b_sum / bw_sum : plain_b / plain_sum, 10.0 * std::log10(best), cells};
    d.first_range = nearest;
    d.extent = farthest - nearest + plan.cell_m;
    // Its arrivals: along range, the strongest sample over the group's bearings; the local peaks of that above the
    // single-sample threshold (a run of samples the group does not hold ends a peak), the 8 strongest, nearest first.
    std::sort(profile.begin(), profile.end());
    std::vector<std::pair<std::int64_t, double>> line;
    for (const auto& q : profile) {
      if (!line.empty() && line.back().first == q.first) line.back().second = std::max(line.back().second, q.second);
      else line.push_back(q);
    }
    std::vector<std::pair<double, std::int64_t>> peaks;                // (ratio, sample)
    for (std::size_t q = 0; q < line.size(); ++q) {
      const bool left = q > 0 && line[q - 1].first == line[q].first - 1, right = q + 1 < line.size() && line[q + 1].first == line[q].first + 1;
      if (line[q].second <= sample_factor) continue;
      if ((left && line[q - 1].second >= line[q].second) || (right && line[q + 1].second > line[q].second)) continue;
      peaks.emplace_back(line[q].second, line[q].first);
    }
    std::sort(peaks.begin(), peaks.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (peaks.size() > 8) peaks.resize(8);
    std::sort(peaks.begin(), peaks.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    for (const auto& q : peaks) d.arrivals.push_back({fine_ranges[static_cast<std::size_t>(fine_offset + q.second)], 10.0 * std::log10(q.first)});
    if (plan.range_estimate == RangeEstimate::First) {
      // The leading edge (RangeEstimate::First): the nearest sample above the single-sample threshold, else the nearest
      // cell's centre. Its placement and its time (transmit + range / c) follow from it in the chain.
      d.range = nearest;
      for (const auto& q : line) {
        if (q.second > sample_factor) {
          d.range = fine_ranges[static_cast<std::size_t>(fine_offset + q.first)];
          break;
        }
      }
    }
    out.push_back(d);
  }
  return out;
}

}  // namespace sonar::live
