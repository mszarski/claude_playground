#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "sonar/live/detections.hpp"

namespace sonar::live {
namespace {

// Student's t over the normal at the 99.95% point, for 1 to 60 degrees of freedom: how much
// larger a speed-over-sigma ratio must be when sigma itself comes from few points.
constexpr double kTRatio[60] = {193.47032, 9.60304, 3.92763, 2.61669, 2.08746, 1.81090, 1.64347, 1.53207, 1.45293, 1.39397, 1.34841, 1.31219, 1.28272, 1.25830, 1.23772, 1.22017, 1.20501, 1.19180, 1.18018, 1.16988, 1.16069, 1.15244, 1.14499, 1.13824, 1.13208, 1.12645, 1.12128, 1.11651, 1.11210, 1.10802, 1.10422, 1.10068, 1.09737, 1.09427, 1.09136, 1.08863, 1.08605, 1.08362, 1.08132, 1.07915, 1.07709, 1.07513, 1.07327, 1.07150, 1.06981, 1.06821, 1.06667, 1.06520, 1.06379, 1.06245, 1.06116, 1.05992, 1.05873, 1.05759, 1.05649, 1.05543, 1.05441, 1.05343, 1.05248, 1.05156};

double t_ratio(std::int64_t dof) {
  if (dof < 1) dof = 1;
  return dof <= 60 ? kTRatio[dof - 1] : 1.0 + 1.9 / static_cast<double>(dof);
}

struct Fit {
  double x, y, vx, vy, rms, sigma;
};

// Some row records a measured position (classify.py: `measured` not all NaN).
bool any_measured(const std::vector<HistoryPoint>& h) {
  for (const auto& p : h) {
    if (!std::isnan(p.mx) || !std::isnan(p.my)) return true;
  }
  return false;
}

// The points the line is fitted through (classify.py fit_velocity, D-157): the measured positions of the rows that have
// one (mx, my finite: the detection the track took there), the rows of misses left out; a history that records none
// (every row NaN: built by hand from positions) is fitted through its filtered x, y as before.
std::vector<HistoryPoint> fit_points(const std::vector<HistoryPoint>& h) {
  std::vector<HistoryPoint> pts;
  for (const auto& p : h) {
    if (!(std::isfinite(p.mx) && std::isfinite(p.my))) continue;
    HistoryPoint q{p.time, p.mx, p.my};
    q.ping = p.ping;
    pts.push_back(q);
  }
  return any_measured(h) ? pts : h;
}

// The fitted points' times differ by their echo delays as much as by their pings (sonar-054, ours; classify.py
// _delays_only): each time is the ping's transmit time plus R/c (D-156), and the transmit times span no more than the
// delays (time - ping) do. A line through such points measures range against R/c, a radial speed of the order of the
// sound speed (pings without timestamps, all at one time, gave tens of km/s). A real target's delays span
// ptp(R)/c <= (|dR/dt| / c) ptp(pings), less than ptp(pings) at any radial speed below c. A row without a ping time (NaN,
// a history built by hand): not tested.
bool delays_only(const std::vector<HistoryPoint>& h) {
  double p_min = std::numeric_limits<double>::infinity(), p_max = -p_min, d_min = p_min, d_max = -p_min;
  for (const auto& p : h) {
    if (!std::isfinite(p.ping)) return false;
    p_min = std::min(p_min, p.ping);
    p_max = std::max(p_max, p.ping);
    d_min = std::min(d_min, p.time - p.ping);
    d_max = std::max(d_max, p.time - p.ping);
  }
  return !h.empty() && p_max - p_min <= d_max - d_min;
}

// P(|T| > t) for Student's t with an integer nu >= 1 degrees of freedom (classify.py t_two_sided): the finite sums of
// Abramowitz and Stegun 26.7.3 (nu odd) and 26.7.4 (nu even) with theta = atan(t / sqrt(nu)).
double t_two_sided(double t, std::int64_t nu) {
  if (!std::isfinite(t)) return 0.0;
  const double th = std::atan(std::abs(t) / std::sqrt(static_cast<double>(nu)));
  const double c = std::cos(th), sn = std::sin(th), c2 = c * c;
  double a = 0.0;
  if (nu % 2) {
    double total = 0.0, term = c;
    for (std::int64_t j = 1; j <= (nu - 1) / 2; ++j) {   // powers 1, 3, ..., nu - 2
      total += term;
      term *= c2 * (2.0 * static_cast<double>(j)) / (2.0 * static_cast<double>(j) + 1.0);
    }
    a = (2.0 / 3.14159265358979323846) * (th + sn * total);
  } else {
    double total = 0.0, term = 1.0;
    for (std::int64_t j = 0; j < nu / 2; ++j) {          // powers 0, 2, ..., nu - 2
      total += term;
      term *= c2 * (2.0 * static_cast<double>(j) + 1.0) / (2.0 * static_cast<double>(j) + 2.0);
    }
    a = sn * total;
  }
  return std::max(0.0, 1.0 - a);
}

// The one point of a line fit through h that is an outlier, or -1 (classify.py _outlier, sonar-ljh after review; ours,
// from the deleted-residual results of least squares). The residuals in range and cross-range: one rotation onto u, from
// (ox, oy) (the sonar) to the points' mean, and its normal; at the mean itself the frame's axes. Per axis a and point i
// t_ai = r_ai / sqrt(s2_ai (1 - h_i)), s2_ai = (SS_a - r_ai^2 / (1 - h_i)) / (n - 3), Student's t with n - 3 degrees of
// freedom. The largest |t_ai| (ties: the earliest point, range first) is an outlier when P(|T| > |t_ai|) < alpha / m,
// m = n x the axes tested. An axis is tested when its RMS is above 1e-9 of the largest coordinate (or 1e-9 m) and the
// other axis's scatter times the largest sine of the turn of a point's line of sight from u is at most half its own
// (scatter sqrt(SS / (n - 2))). Four points at least; no candidate where 1 - h_i <= 1e-12; alpha <= 0 or NaN: off.
std::int64_t outlier(const std::vector<HistoryPoint>& h, double alpha, double ox, double oy) {
  const std::size_t n = h.size();
  if (!(alpha > 0.0) || n < 4) return -1;
  const double dn = static_cast<double>(n);
  double tm = 0.0, xm = 0.0, ym = 0.0, big = 0.0;
  for (const auto& p : h) {
    tm += p.time;
    xm += p.x;
    ym += p.y;
    big = std::max({big, std::abs(p.x), std::abs(p.y)});
  }
  tm /= dn;
  xm /= dn;
  ym /= dn;
  double stt = 0.0, stx = 0.0, sty = 0.0;
  for (const auto& p : h) {
    const double tc = p.time - tm;
    stt += tc * tc;
    stx += tc * p.x;
    sty += tc * p.y;
  }
  const double vx = stx / stt, vy = sty / stt;
  const double dx = xm - ox, dy = ym - oy, dist = std::hypot(dx, dy);
  const bool aimed = dist > 0.0 && std::isfinite(dist);
  const double ux = aimed ? dx / dist : 1.0, uy = aimed ? dy / dist : 0.0;
  std::vector<double> ra[2] = {std::vector<double>(n), std::vector<double>(n)};
  double ss[2] = {0.0, 0.0}, turn = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double tc = h[i].time - tm;
    const double rx = h[i].x - (xm + tc * vx), ry = h[i].y - (ym + tc * vy);
    ra[0][i] = rx * ux + ry * uy;   // range
    ra[1][i] = rx * uy - ry * ux;   // cross-range
    ss[0] += ra[0][i] * ra[0][i];
    ss[1] += ra[1][i] * ra[1][i];
    const double px = h[i].x - ox, py = h[i].y - oy, pd = std::hypot(px, py);
    if (aimed && pd > 0.0) turn = std::max(turn, std::abs(px * uy - py * ux) / pd);
  }
  const double tol = 1e-9 * std::max(1.0, big);
  const double sd[2] = {std::sqrt(ss[0] / (dn - 2.0)), std::sqrt(ss[1] / (dn - 2.0))};
  bool live[2];
  int tested = 0;
  for (int k = 0; k < 2; ++k) {
    live[k] = std::sqrt(ss[k] / dn) > tol && sd[1 - k] * turn <= 0.5 * sd[k];
    tested += live[k] ? 1 : 0;
  }
  if (tested == 0) return -1;
  const double nu = dn - 3.0;
  std::int64_t best = -1;
  double best_t2 = -1.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double tc = h[i].time - tm;
    const double one_minus_h = 1.0 - (1.0 / dn + tc * tc / stt);
    if (!(one_minus_h > 1e-12)) continue;
    for (int k = 0; k < 2; ++k) {
      if (!live[k]) continue;
      const double d2 = ra[k][i] * ra[k][i] / one_minus_h, rest = (ss[k] - d2) / nu;
      const double t2 = rest > 0.0 ? d2 / rest : (d2 > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
      if (t2 > best_t2) {
        best = static_cast<std::int64_t>(i);
        best_t2 = t2;
      }
    }
  }
  if (best < 0) return -1;
  return t_two_sided(std::sqrt(best_t2), static_cast<std::int64_t>(n) - 3) < alpha / (dn * static_cast<double>(tested)) ? best : -1;
}

// Least-squares straight line through (time, x, y) of fit_points(h); the position on it at `at` (NaN, or any value that
// is not finite: at the time of the last row of `history`, the last ping). With fewer than three points, or no time between them (max - min, as
// numpy.ptp), or transmit times (HistoryPoint::ping) that span no more than the echo delays (delays_only, sonar-054),
// no fitted velocity and the last row of `h` (its filtered position), moved to `at` by (vx, vy), the track's
// Kalman state's velocity (classify.py fit_velocity). Through measured positions, the one point `outlier` finds at
// `outlier_alpha` about the sonar at (ox, oy) is left out (sonar-ljh). `n_fit` receives the number of points fitted.
Fit fit_velocity(const std::vector<HistoryPoint>& history, double at, double vx, double vy, double outlier_alpha, double ox, double oy,
                 std::int64_t& n_fit) {
  std::vector<HistoryPoint> h = fit_points(history);
  const bool from_measured = any_measured(history);
  const std::size_t n = h.size();
  n_fit = static_cast<std::int64_t>(n);
  Fit f{history.back().x, history.back().y, 0.0, 0.0, 0.0, std::numeric_limits<double>::infinity()};
  double t_min = std::numeric_limits<double>::infinity(), t_max = -std::numeric_limits<double>::infinity();
  for (const auto& p : h) {
    t_min = std::min(t_min, p.time);
    t_max = std::max(t_max, p.time);
  }
  if (n < 3 || t_max - t_min <= 0.0 || delays_only(h)) {
    if (std::isfinite(at) && std::isfinite(vx) && std::isfinite(vy)) {
      f.x += (at - history.back().time) * vx;
      f.y += (at - history.back().time) * vy;
    }
    return f;
  }
  if (from_measured) {
    const std::int64_t out = outlier(h, outlier_alpha, ox, oy);
    if (out >= 0) {
      h.erase(h.begin() + out);
      n_fit = static_cast<std::int64_t>(h.size());
    }
  }
  const std::size_t m = h.size();
  double tm = 0.0, xm = 0.0, ym = 0.0;
  for (const auto& p : h) {
    tm += p.time;
    xm += p.x;
    ym += p.y;
  }
  tm /= static_cast<double>(m);
  xm /= static_cast<double>(m);
  ym /= static_cast<double>(m);
  double stt = 0.0, stx = 0.0, sty = 0.0;
  for (const auto& p : h) {
    const double tc = p.time - tm;
    stt += tc * tc;
    stx += tc * p.x;
    sty += tc * p.y;
  }
  f.vx = stx / stt;
  f.vy = sty / stt;
  double ss = 0.0;
  for (const auto& p : h) {
    const double tc = p.time - tm;
    const double rx = p.x - (xm + tc * f.vx), ry = p.y - (ym + tc * f.vy);
    ss += rx * rx + ry * ry;
  }
  // No time given: the last history row's (the last ping), also when the measured rows end earlier (classify.py).
  const double tc_last = (std::isfinite(at) ? at : history.back().time) - tm;
  f.x = xm + tc_last * f.vx;
  f.y = ym + tc_last * f.vy;
  f.rms = std::sqrt(ss / static_cast<double>(std::max<std::int64_t>(2 * static_cast<std::int64_t>(m) - 4, 1)));
  f.sigma = f.rms / std::sqrt(stt);
  return f;
}

// Median of the detections' cell counts, as numpy.median (mean of the middle two); NaN when there are none.
double median_cells(const std::vector<std::int64_t>& cells) {
  if (cells.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::vector<std::int64_t> c(cells);
  std::sort(c.begin(), c.end());
  const std::size_t n = c.size();
  return n % 2 ? static_cast<double>(c[n / 2]) : 0.5 * (static_cast<double>(c[n / 2 - 1]) + static_cast<double>(c[n / 2]));
}

}  // namespace

const char* kind_name(Kind kind) {
  switch (kind) {
    case Kind::Mover: return "mover";
    case Kind::Static: return "static";
    case Kind::Line: return "line";
    case Kind::Multipath: return "multipath";
    default: return "uncertain";
  }
}

void group_echoes(std::vector<TrackClass>& out, const ClassifyOptions& o, const std::vector<std::int64_t>& bounces) {
  // A bounce (the chain's elevation step: a straight-line depth outside the water; sonar-bal) is multipath before
  // grouping: never an anchor nor an object; a member of a group like any other track, else multipath of nothing.
  for (TrackClass& c : out) {
    if (std::find(bounces.begin(), bounces.end(), c.id) != bounces.end()) {
      c.kind = Kind::Multipath;
      c.parent = -1;
    }
  }
  // Echo grouping (ours, D-155; classify.py has the reasons): from each established track, strongest first, a group
  // grown transitively from the tracks not yet grouped: a track joins when it is within multipath_max_excess_range in
  // range (either side), the bearing gate and the velocity gate of ANY member, until none joins. The group's object is
  // its established member nearest in range (ties: the lowest id), every other member is multipath of it. A member
  // farther than multipath_max_excess_range from the object in range is left out of the group and free for one of
  // its own; when it is the anchor, the anchor groups again from what is left (sonar-a5x: grown transitively, a chain
  // of arrivals linked two objects farther apart than one gate).
  std::vector<std::size_t> order(out.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  auto key = [&](std::size_t i) { return std::isfinite(out[i].mean_snr_db) ? out[i].mean_snr_db : -std::numeric_limits<double>::infinity(); };
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return key(a) > key(b); });
  std::vector<double> range(out.size()), bearing(out.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double dx = out[i].x - o.origin_x, dy = out[i].y - o.origin_y;
    range[i] = std::hypot(dx, dy);
    bearing[i] = std::atan2(dx, dy);
  }
  auto established = [&](std::size_t i) { return out[i].kind == Kind::Mover || out[i].kind == Kind::Static || out[i].kind == Kind::Line; };
  auto linked = [&](std::size_t i, std::size_t j) {   // false when anything is NaN, as in classify.py
    const double db = std::abs(std::atan2(std::sin(bearing[i] - bearing[j]), std::cos(bearing[i] - bearing[j])));
    const double dv = std::hypot(out[i].vx - out[j].vx, out[i].vy - out[j].vy);
    return std::abs(range[i] - range[j]) <= o.multipath_max_excess_range && db <= o.multipath_bearing && dv <= o.multipath_velocity;
  };
  std::vector<bool> taken(out.size(), false), member(out.size(), false);
  for (const std::size_t a : order) {
    if (!established(a)) continue;
    if (!linked(a, a)) continue;   // NaN range, bearing or velocity: nothing to group by; left as it is
    while (!taken[a]) {            // again when the anchor itself lay beyond its group's object
      std::vector<std::size_t> group{a};
      std::fill(member.begin(), member.end(), false);
      member[a] = true;
      for (bool grew = true; grew;) {
        grew = false;
        for (std::size_t i = 0; i < out.size(); ++i) {
          if (taken[i] || member[i]) continue;
          for (const std::size_t g : group) {
            if (linked(i, g)) {
              group.push_back(i);
              member[i] = true;
              grew = true;
              break;
            }
          }
        }
      }
      // The anchor is established, so an established member exists.
      std::size_t obj = a;
      for (const std::size_t i : group) {
        if (established(i) && (range[i] < range[obj] || (range[i] == range[obj] && out[i].id < out[obj].id))) obj = i;
      }
      for (const std::size_t i : group) {
        if (std::abs(range[i] - range[obj]) > o.multipath_max_excess_range) continue;   // not its echo (sonar-a5x)
        taken[i] = true;
        if (i != obj) {
          out[i].kind = Kind::Multipath;
          out[i].parent = out[obj].id;
        }
      }
    }
  }
}

namespace {
// D5: the filter's velocity as a speed test (classify.py filter_speed): v^T P^-1 v (0 when P is not positive definite),
// |v|, and sqrt(u^T P u) with u = v / |v| (at v = 0 the larger axis' sigma); v = state[2:4], P = cov[2:4, 2:4].
struct FilterSpeed {
  double chi2 = 0.0, speed = 0.0, sigma = 0.0, vx = 0.0, vy = 0.0;
};
FilterSpeed filter_speed(const Track& t) {
  FilterSpeed f;
  f.vx = t.state[2];
  f.vy = t.state[3];
  const double a = t.cov[10], b = t.cov[11], d = t.cov[15];
  const double det = a * d - b * b;
  f.chi2 = det > 0.0 && a > 0.0 ? (d * f.vx * f.vx - 2.0 * b * f.vx * f.vy + a * f.vy * f.vy) / det : 0.0;
  f.speed = std::hypot(f.vx, f.vy);
  if (f.speed > 0.0) {
    const double ux = f.vx / f.speed, uy = f.vy / f.speed;
    f.sigma = std::sqrt(std::max(ux * ux * a + 2.0 * ux * uy * b + uy * uy * d, 0.0));
  } else {
    f.sigma = std::sqrt(std::max({a, d, 0.0}));
  }
  return f;
}
}  // namespace

std::vector<TrackClass> classify_tracks(const std::vector<const Track*>& tracks, const ClassifyOptions& o,
                                        std::vector<ClassEvidence>* evidence) {
  const std::size_t first_evidence = evidence ? evidence->size() : 0;
  std::vector<TrackClass> out;
  for (const Track* tr : tracks) {
    std::int64_t n_fit = 0;
    const Fit f = fit_velocity(tr->history, o.time, tr->state[2], tr->state[3], o.outlier_alpha, o.origin_x, o.origin_y, n_fit);
    TrackClass c;
    c.id = tr->id;
    c.x = f.x;
    c.y = f.y;
    c.vx = f.vx;
    c.vy = f.vy;
    c.speed = std::hypot(f.vx, f.vy);
    c.speed_sigma = f.sigma;
    c.fit_rms = f.rms;
    const auto n = static_cast<std::int64_t>(tr->history.size());
    c.seen = static_cast<double>(tr->hits) / static_cast<double>(std::max<std::int64_t>(n, 1));
    double snr = 0.0;
    for (double s : tr->snr) snr += s;
    c.mean_snr_db = tr->snr.empty() ? std::numeric_limits<double>::quiet_NaN() : snr / static_cast<double>(tr->snr.size());
    // The fit's degrees of freedom count the points fitted (the measured rows, D-157); min_points and seen the pings.
    const double need = o.significance * t_ratio(std::max<std::int64_t>(2 * n_fit - 4, 1));
    // The fit limit, at least max_fit_rms_bearing x the track's range from the sonar (sonar-4jm).
    const double limit = o.max_fit_rms_bearing > 0.0 ? std::max(o.max_fit_rms, o.max_fit_rms_bearing * std::hypot(f.x - o.origin_x, f.y - o.origin_y))
                                                     : o.max_fit_rms;
    const bool steady = o.max_fit_rms < 0.0 || f.rms <= limit;
    // D5 (opt-in): the filter's velocity, which the range rate informs (classify.py `range_rate`).
    const bool rated = o.range_rate && tr->rate_updates > 0;
    const FilterSpeed fs = rated ? filter_speed(*tr) : FilterSpeed{};
    const bool rate_mover = rated && fs.speed >= o.min_speed && fs.chi2 >= o.significance * o.significance && c.seen >= o.min_seen;
    const bool fit_mover = c.speed >= o.min_speed && c.speed >= need * f.sigma && c.seen >= o.min_seen;
    if (n < o.min_points || !steady) {
      c.kind = Kind::Uncertain;
    } else if (fit_mover) {
      c.kind = Kind::Mover;
    } else if (rate_mover) {
      c.kind = Kind::Mover;   // mover_range_rate: the filter's velocity, speed and sigma
      c.vx = fs.vx;
      c.vy = fs.vy;
      c.speed = fs.speed;
      c.speed_sigma = fs.sigma;
    } else if (c.speed < o.static_speed && c.seen >= o.min_seen) {
      c.kind = Kind::Static;
    } else if (c.speed < need * f.sigma && c.seen >= o.min_seen && c.speed < 3.0 * o.static_speed) {
      c.kind = Kind::Static;
    } else {
      c.kind = Kind::Uncertain;
    }
    c.extent_cells = median_cells(tr->cells);
    out.push_back(c);
    if (evidence) {   // the tap: reads only
      ClassEvidence e;
      e.id = c.id;
      e.rows = n;
      e.fitted = n_fit;
      e.speed = c.speed;
      e.speed_sigma = c.speed_sigma;
      e.need = need;
      e.seen = c.seen;
      e.fit_rms = f.rms;
      e.limit = o.max_fit_rms < 0.0 ? std::numeric_limits<double>::quiet_NaN() : limit;
      e.steady = steady;
      e.extent_cells = c.extent_cells;
      e.kind = kind_name(c.kind);
      if (n < o.min_points) e.rule = "few_points";
      else if (!steady) e.rule = "scatter";
      else if (c.kind == Kind::Mover) e.rule = fit_mover ? "mover" : "mover_range_rate";
      else if (c.kind == Kind::Uncertain) e.rule = "uncertain_otherwise";
      else e.rule = (c.speed < o.static_speed && c.seen >= o.min_seen) ? "static_slow" : "static_insignificant";
      if (o.range_rate) {   // D5: written out only with the rate on
        e.has_rate = true;
        e.rate_updates = tr->rate_updates;
        e.filter_speed = rated ? fs.speed : std::numeric_limits<double>::quiet_NaN();
        e.filter_chi2 = rated ? fs.chi2 : std::numeric_limits<double>::quiet_NaN();
      }
      evidence->push_back(e);
    }
  }
  if (o.linear) {
    // Other tracks in a narrow corridor ahead of and astern of a mover, along its own velocity.
    std::vector<Kind> updated(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) updated[i] = out[i].kind;
    for (std::size_t i = 0; i < out.size(); ++i) {
      if (out[i].kind != Kind::Mover) continue;
      const double ux = out[i].vx / out[i].speed, uy = out[i].vy / out[i].speed;
      int ahead = 0, astern = 0;
      for (std::size_t j = 0; j < out.size(); ++j) {
        const double dx = out[j].x - out[i].x, dy = out[j].y - out[i].y;
        const double along = dx * ux + dy * uy, across = std::abs(dx * uy - dy * ux);
        if (std::abs(along) > o.linear_length || across > o.linear_width) continue;
        // Only echoes of about this one's strength make a row with it; a NaN SNR never does.
        if (o.row_snr_db >= 0.0 && !(out[j].mean_snr_db >= out[i].mean_snr_db - o.row_snr_db)) continue;
        if (along > 1e-6) ++ahead;
        if (along < -1e-6) ++astern;
      }
      if (ahead >= 1 && astern >= 1) {
        updated[i] = Kind::Line;
        if (evidence) (*evidence)[first_evidence + i].later.push_back("line");
      }
    }
    // Python changes kinds as it goes, and a track already made a line no longer counts as a mover
    // for later ones only through its own entry, which the loop above never reads: the same result.
    for (std::size_t i = 0; i < out.size(); ++i) out[i].kind = updated[i];
  }
  if (o.min_mover_cells >= 0.0) {
    // After the corridor test (a row of small echoes along a pipeline stays a line), before echo grouping (a track too
    // small for an object under way explains nothing). A glint or clutter joined into a line; NaN (no cells) leaves it.
    for (std::size_t i = 0; i < out.size(); ++i) {
      TrackClass& c = out[i];
      if (c.kind == Kind::Mover && c.extent_cells < o.min_mover_cells) {
        c.kind = Kind::Uncertain;
        if (evidence) (*evidence)[first_evidence + i].later.push_back("small_extent");
      }
    }
  }
  if (o.multipath) {
    std::vector<std::pair<Kind, std::int64_t>> before;
    for (const TrackClass& c : out) before.emplace_back(c.kind, c.parent);
    group_echoes(out, o);
    if (evidence) {
      std::vector<ClassEvidence> mine(evidence->begin() + static_cast<std::ptrdiff_t>(first_evidence), evidence->end());
      evidence_after_grouping(mine, before, out);
      std::copy(mine.begin(), mine.end(), evidence->begin() + static_cast<std::ptrdiff_t>(first_evidence));
    }
  }
  return out;
}

void evidence_after_grouping(std::vector<ClassEvidence>& evidence, const std::vector<std::pair<Kind, std::int64_t>>& before,
                             const std::vector<TrackClass>& classes, const std::vector<std::int64_t>& bounces) {
  for (std::size_t i = 0; i < evidence.size() && i < classes.size() && i < before.size(); ++i) {
    const TrackClass& c = classes[i];
    const bool bounce = std::find(bounces.begin(), bounces.end(), c.id) != bounces.end();
    if (bounce) evidence[i].later.push_back("bounce");
    if (c.kind == Kind::Multipath && (before[i].first != Kind::Multipath || before[i].second != c.parent) && !(bounce && c.parent < 0)) {
      evidence[i].later.push_back("multipath:" + (c.parent < 0 ? std::string("none") : std::to_string(c.parent)));
    }
  }
}

void shift_for_target_doppler(std::vector<TrackClass>& classes, double factor, double origin_x, double origin_y) {
  for (TrackClass& c : classes) {
    const double dx = c.x - origin_x, dy = c.y - origin_y;
    const double r = std::hypot(dx, dy);
    if (!(r > 0.0) || !std::isfinite(c.vx) || !std::isfinite(c.vy)) continue;
    const double ux = dx / r, uy = dy / r;
    const double shift = factor * -(c.vx * ux + c.vy * uy);
    c.x += shift * ux;
    c.y += shift * uy;
  }
}

}  // namespace sonar::live
