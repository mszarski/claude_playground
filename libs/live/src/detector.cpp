#include "sonar/live/detector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "sonar/detect/statistic.hpp"

namespace sonar::live {
namespace {

constexpr double kPi = 3.14159265358979323846;

double sinc(double x) { return std::abs(x) < 1e-12 ? 1.0 : std::sin(kPi * x) / (kPi * x); }

}  // namespace

std::vector<double> sample_correlation(const Waveform& w, double sound_speed, double step, std::int64_t n) {
  std::vector<double> r(static_cast<std::size_t>(std::max<std::int64_t>(n, 1)));
  const double band = std::abs(w.f_stop - w.f_start);
  for (std::size_t d = 0; d < r.size(); ++d) {
    const double x = static_cast<double>(d) * step;
    // Swept: a flat band, whose compressed pulse is a sinc with its first zero one resolution (c / 2B) out.
    // Not swept: the compressed pulse is the triangle of the pulse's own length.
    r[d] = band > 0.0 ? sinc(x * 2.0 * band / sound_speed) : std::max(1.0 - x / (0.5 * sound_speed * w.duration), 0.0);
  }
  return r;
}

double DetectorPlan::threshold(std::size_t s, double count) const {
  const DetectorScale& sc = scales.at(s);
  return sc.factor * detect::training_correction(sc.pfa, count * train_looks_per_cell);
}

std::string DetectorPlan::json() const {
  std::ostringstream s;
  s << std::setprecision(6);
  s << "{\"rows\": " << rows << ", \"samples_per_cell\": " << samples_per_cell << ", \"cell_m\": " << cell_m << ", \"step_m\": " << step_m
    << ", \"resolution_m\": " << resolution_m << ", \"beams\": " << beams << ", \"train_looks_per_cell\": " << train_looks_per_cell
    << ", \"merge_cells\": " << merge_cells << ", \"sidelobe_margin_db\": " << 10.0 * std::log10(margin) << ", \"source_snr_db\": " << 10.0 * std::log10(source_ratio) << ", \"scales\": [";
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const DetectorScale& c = scales[i];
    s << (i ? ", " : "") << "{\"cells\": " << c.cells << ", \"length_m\": " << c.length_m << ", \"looks\": " << c.looks << ", \"tests\": " << c.tests
      << ", \"pfa\": " << c.pfa << ", \"threshold_db\": " << 10.0 * std::log10(c.factor) << "}";
  }
  s << "]}";
  return s.str();
}

DetectorPlan plan_detector(const PingRecord& record, const SonarFacts& facts, const DetectorOptions& options, std::int64_t n_cells,
                           std::int64_t samples_per_cell, double step, const std::vector<double>& bearings, std::int64_t rows,
                           std::int64_t guard_cells, std::int64_t train_cells, bool hann, double merge_m) {
  if (n_cells < 1 || samples_per_cell < 1 || !(step > 0.0) || bearings.empty() || rows < 1 || train_cells < 1) {
    throw std::invalid_argument("plan_detector: an empty map");
  }
  if (!(options.false_alarms_per_ping > 0.0)) throw std::invalid_argument("cfar.false_alarms_per_ping must be positive");
  const Waveform& w = record.transmit.waveform;
  const double c = facts.sound_speed;
  DetectorPlan p;
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
    s.tests = std::max(options.search_factor * p.beams * extent / std::max(length, p.resolution_m), 1.0);
    s.pfa = std::min(options.false_alarms_per_ping / n_scales / s.tests, 0.1);
    s.factor = detect::weighted_exponential_threshold(weights, rows, s.pfa);
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
  // Bearing sidelobes of one row with the map's aperture window, averaged over the band (which fills the nulls).
  const std::int64_t n_el = std::max<std::int64_t>(facts.row_elements, 1);
  const double pitch = facts.aperture / static_cast<double>(n_el);
  std::vector<double> weight(static_cast<std::size_t>(n_el), 1.0), x_el(static_cast<std::size_t>(n_el));
  double w_sum = 0.0;
  for (std::int64_t i = 0; i < n_el; ++i) {
    x_el[static_cast<std::size_t>(i)] = (static_cast<double>(i) - 0.5 * static_cast<double>(n_el - 1)) * pitch;
    if (hann) {
      const double cw = std::cos(kPi * x_el[static_cast<std::size_t>(i)] / facts.aperture);
      weight[static_cast<std::size_t>(i)] = cw * cw;
    }
    w_sum += weight[static_cast<std::size_t>(i)];
  }
  const std::int64_t n_u = 2048;
  p.u_step = 2.0 / static_cast<double>(n_u);
  std::vector<double> pattern(static_cast<std::size_t>(n_u + 1), 0.0);
  const int n_f = 9;
  for (std::int64_t iu = 0; iu <= n_u; ++iu) {
    const double u = p.u_step * static_cast<double>(iu);
    double acc = 0.0;
    for (int q = 0; q < n_f; ++q) {
      const double f = facts.f_low + (facts.f_high - facts.f_low) * (static_cast<double>(q) + 0.5) / n_f;
      std::complex<double> z(0.0, 0.0);
      for (std::int64_t i = 0; i < n_el; ++i) {
        z += weight[static_cast<std::size_t>(i)] * std::polar(1.0, 2.0 * kPi * f / c * x_el[static_cast<std::size_t>(i)] * u);
      }
      acc += std::norm(z) / (w_sum * w_sum);
    }
    pattern[static_cast<std::size_t>(iu)] = acc / n_f;
  }
  // The detection itself is the part of the beam within the margin of its peak (and the cells next to it in range):
  // the envelope is 1 there. Beyond, it is the beam's own skirt and then its sidelobe peaks (the largest value from
  // here to one beamwidth further out, so that the nulls between sidelobes are bridged), never below the floor.
  std::int64_t main = 1;
  while (main < n_u && pattern[static_cast<std::size_t>(main)] >= 1.0 / p.margin) ++main;
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

}  // namespace

void detector_background(const float* cell_mean, std::int64_t n_b, std::int64_t n_cells, std::int64_t guard, std::int64_t train,
                         std::vector<double>& background, std::vector<double>& train_cells) {
  background.assign(static_cast<std::size_t>(n_b * n_cells), 0.0);
  train_cells.assign(static_cast<std::size_t>(n_cells), 1.0);
  // Training cells of each range cell (the same for every bearing): [j - guard - train, j - guard) and (j + guard, j + guard + train].
  std::vector<std::int64_t> lo0(static_cast<std::size_t>(n_cells)), lo1(lo0), hi0(lo0), hi1(lo0);
  std::vector<double> offset(static_cast<std::size_t>(n_cells), 0.0);     // of the training cells' centroid from the cell, cells
  const bool sloped = n_cells >= 2 * train && train >= 8;
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

std::vector<Detection> multiscale_detect(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean,
                                         std::int64_t n_b, std::int64_t n_cells, const std::vector<double>& fine_ranges,
                                         const std::vector<double>& cell_ranges, const std::vector<double>& bearings,
                                         std::int64_t guard, std::int64_t train, const DetectorPlan& plan, std::vector<float>& background) {
  const std::int64_t k = plan.samples_per_cell;
  const auto n = static_cast<std::size_t>(n_b * n_cells);
  std::vector<double> level, worth;
  detector_background(cell_mean, n_b, n_cells, guard, train, level, worth);
  background.assign(n, 0.0F);
  std::vector<float> peak(n, 0.0F);
  std::vector<unsigned char> flag(n, 0);
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
      const std::int64_t smear = static_cast<std::int64_t>(std::ceil(0.25 * plan.aperture_m * du / plan.cell_m - 1e-9));
      const std::int64_t dc = std::max<std::int64_t>(std::abs(j - j0) - smear, 0);
      const double e_r = plan.range_envelope[static_cast<std::size_t>(std::min<std::int64_t>(dc, n_cells))];
      if (e_r >= 1.0 && e_b >= 1.0) continue;                        // inside the mainlobe: part of the detection
      if (pq <= plan.margin * e_r * e_b * p0) flag[static_cast<std::size_t>(i)] = 0;
    }
  }

  // Flagged cells of one bearing no further apart than the tracker can tell apart are joined (the cells between
  // them become part of the group): the arrivals of one multipath group, a target's highlights.
  if (plan.merge_cells > 0) {
    bool any = false;
    for (std::int64_t b = 0; b < n_b; ++b) {
      std::int64_t last = -1;
      for (std::int64_t j = 0; j < n_cells; ++j) {
        if (!flag[static_cast<std::size_t>(b * n_cells + j)]) continue;
        if (last >= 0 && j - last > 1 && j - last - 1 <= plan.merge_cells) {
          for (std::int64_t q = last + 1; q < j; ++q) flag[static_cast<std::size_t>(b * n_cells + q)] = 1;
          any = true;
        }
        last = j;
      }
    }
    if (any) {
      flagged.clear();
      for (std::int64_t i = 0; i < n_b * n_cells; ++i) {
        if (flag[static_cast<std::size_t>(i)]) flagged.push_back(i);
      }
    }
  }

  // One detection per group of touching flagged cells (diagonals included).
  std::vector<Detection> out;
  std::vector<unsigned char> seen(n, 0);
  std::vector<std::int64_t> stack;
  for (std::int64_t start : flagged) {
    if (!flag[static_cast<std::size_t>(start)] || seen[static_cast<std::size_t>(start)]) continue;
    double w_sum = 0.0, r_sum = 0.0, b_sum = 0.0, best = 0.0, nearest = 1e300, farthest = -1e300;
    std::int64_t cells = 0;
    stack.assign(1, start);
    seen[static_cast<std::size_t>(start)] = 1;
    while (!stack.empty()) {
      const std::int64_t i = stack.back();
      stack.pop_back();
      const std::int64_t row = i / n_cells, col = i % n_cells;
      const float* fine_row = fine + row * fine_stride + fine_offset + col * k;
      for (std::int64_t s = 0; s < k; ++s) {
        const auto w = static_cast<double>(fine_row[s]);
        w_sum += w;
        r_sum += w * fine_ranges[static_cast<std::size_t>(fine_offset + col * k + s)];
        b_sum += w * bearings[static_cast<std::size_t>(row)];
      }
      nearest = std::min(nearest, cell_ranges[static_cast<std::size_t>(col)]);
      farthest = std::max(farthest, cell_ranges[static_cast<std::size_t>(col)]);
      const double ratio = static_cast<double>(peak[static_cast<std::size_t>(i)]) / std::max(static_cast<double>(background[static_cast<std::size_t>(i)]), 1e-300);
      best = std::max(best, ratio);
      ++cells;
      for (std::int64_t dr = -1; dr <= 1; ++dr) {
        for (std::int64_t dc = -1; dc <= 1; ++dc) {
          const std::int64_t r2 = row + dr, c2 = col + dc;
          if (r2 < 0 || r2 >= n_b || c2 < 0 || c2 >= n_cells) continue;
          const std::int64_t j = r2 * n_cells + c2;
          if (flag[static_cast<std::size_t>(j)] && !seen[static_cast<std::size_t>(j)]) {
            seen[static_cast<std::size_t>(j)] = 1;
            stack.push_back(j);
          }
        }
      }
    }
    Detection d{r_sum / w_sum, b_sum / w_sum, 10.0 * std::log10(best), cells};
    d.first_range = nearest;
    d.extent = farthest - nearest + plan.cell_m;
    out.push_back(d);
  }
  return out;
}

}  // namespace sonar::live
