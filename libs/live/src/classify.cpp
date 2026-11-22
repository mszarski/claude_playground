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

// The points the line is fitted through (classify.py fit_velocity, D-157): the measured positions of the rows that have
// one (mx, my finite: the detection the track took there), the rows of misses left out; a history that records none
// (every row NaN: built by hand from positions) is fitted through its filtered x, y as before.
std::vector<HistoryPoint> fit_points(const std::vector<HistoryPoint>& h) {
  std::vector<HistoryPoint> pts;
  for (const auto& p : h) {
    if (std::isfinite(p.mx) && std::isfinite(p.my)) pts.push_back({p.time, p.mx, p.my});
  }
  bool any = false;
  for (const auto& p : h) any = any || !std::isnan(p.mx) || !std::isnan(p.my);
  return any ? pts : h;
}

// Least-squares straight line through (time, x, y) of fit_points(h); the position on it at `at` (NaN, or any value that
// is not finite: at the time of the last row of `history`, the last ping). With fewer than three points, or no time between them (max - min, as
// numpy.ptp), no fitted velocity and the last row of `h` (its filtered position), moved to `at` by (vx, vy), the track's
// Kalman state's velocity (classify.py fit_velocity). `n_fit` receives the number of points fitted.
Fit fit_velocity(const std::vector<HistoryPoint>& history, double at, double vx, double vy, std::int64_t& n_fit) {
  const std::vector<HistoryPoint> h = fit_points(history);
  const std::size_t n = h.size();
  n_fit = static_cast<std::int64_t>(n);
  Fit f{history.back().x, history.back().y, 0.0, 0.0, 0.0, std::numeric_limits<double>::infinity()};
  double t_min = std::numeric_limits<double>::infinity(), t_max = -std::numeric_limits<double>::infinity();
  for (const auto& p : h) {
    t_min = std::min(t_min, p.time);
    t_max = std::max(t_max, p.time);
  }
  if (n < 3 || t_max - t_min <= 0.0) {
    if (std::isfinite(at) && std::isfinite(vx) && std::isfinite(vy)) {
      f.x += (at - history.back().time) * vx;
      f.y += (at - history.back().time) * vy;
    }
    return f;
  }
  double tm = 0.0, xm = 0.0, ym = 0.0;
  for (const auto& p : h) {
    tm += p.time;
    xm += p.x;
    ym += p.y;
  }
  tm /= static_cast<double>(n);
  xm /= static_cast<double>(n);
  ym /= static_cast<double>(n);
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
  f.rms = std::sqrt(ss / static_cast<double>(std::max<std::int64_t>(2 * static_cast<std::int64_t>(n) - 4, 1)));
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

std::vector<TrackClass> classify_tracks(const std::vector<const Track*>& tracks, const ClassifyOptions& o) {
  std::vector<TrackClass> out;
  for (const Track* tr : tracks) {
    std::int64_t n_fit = 0;
    const Fit f = fit_velocity(tr->history, o.time, tr->state[2], tr->state[3], n_fit);
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
    const bool steady = o.max_fit_rms < 0.0 || f.rms <= o.max_fit_rms;
    if (n < o.min_points || !steady) {
      c.kind = Kind::Uncertain;
    } else if (c.speed >= o.min_speed && c.speed >= need * f.sigma && c.seen >= o.min_seen) {
      c.kind = Kind::Mover;
    } else if (c.speed < o.static_speed && c.seen >= o.min_seen) {
      c.kind = Kind::Static;
    } else if (c.speed < need * f.sigma && c.seen >= o.min_seen && c.speed < 3.0 * o.static_speed) {
      c.kind = Kind::Static;
    } else {
      c.kind = Kind::Uncertain;
    }
    c.extent_cells = median_cells(tr->cells);
    out.push_back(c);
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
      if (ahead >= 1 && astern >= 1) updated[i] = Kind::Line;
    }
    // Python changes kinds as it goes, and a track already made a line no longer counts as a mover
    // for later ones only through its own entry, which the loop above never reads: the same result.
    for (std::size_t i = 0; i < out.size(); ++i) out[i].kind = updated[i];
  }
  if (o.min_mover_cells >= 0.0) {
    // After the corridor test (a row of small echoes along a pipeline stays a line), before echo grouping (a track too
    // small for an object under way explains nothing). A glint or clutter joined into a line; NaN (no cells) leaves it.
    for (TrackClass& c : out) {
      if (c.kind == Kind::Mover && c.extent_cells < o.min_mover_cells) c.kind = Kind::Uncertain;
    }
  }
  if (o.multipath) {
    // Echo grouping (ours, D-155; classify.py has the reasons): from each established track, strongest first, a group
    // grown transitively from the tracks not yet grouped: a track joins when it is within multipath_max_excess_range in
    // range (either side), the bearing gate and the velocity gate of ANY member, until none joins. The group's object is
    // its established member nearest in range (ties: the lowest id), every other member is multipath of it.
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
      if (taken[a] || !established(a)) continue;
      if (!linked(a, a)) continue;   // NaN range, bearing or velocity: nothing to group by; left as it is
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
        taken[i] = true;
        if (i != obj) {
          out[i].kind = Kind::Multipath;
          out[i].parent = out[obj].id;
        }
      }
    }
  }
  return out;
}

}  // namespace sonar::live
