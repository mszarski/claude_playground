#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include <Eigen/Dense>

#include "sonar/live/detections.hpp"

namespace sonar::live {
namespace {

using M4 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
using M2 = Eigen::Matrix2d;
using V2 = Eigen::Vector2d;
using V4 = Eigen::Vector4d;
using H24 = Eigen::Matrix<double, 2, 4>;

V2 xy(const Detection& d) { return {d.range * std::sin(d.bearing), d.range * std::cos(d.bearing)}; }

// Polar measurement noise rotated into Cartesian.
M2 measurement_cov(const Detection& d, const TrackerOptions& o) {
  const double s = std::sin(d.bearing), c = std::cos(d.bearing);
  M2 j;
  j << s, d.range * c, c, -d.range * s;
  M2 p = M2::Zero();
  const double rs = std::isfinite(d.range_sigma) ? d.range_sigma : o.range_sigma;   // a tone's coarse range (D3); NaN: as before
  p(0, 0) = rs * rs;
  p(1, 1) = o.bearing_sigma * o.bearing_sigma;
  return j * p * j.transpose();
}

// A scalar Kalman update of the track's depth with a detection's, when it has one.
void take_depth(Track& t, const Detection& d) {
  if (!(std::isfinite(d.depth) && std::isfinite(d.depth_sigma) && d.depth_sigma > 0.0)) return;
  const double r = d.depth_sigma * d.depth_sigma;
  double z = d.depth;
  if (d.depth_ambiguous) {
    // Several candidates: the nearest to the depth the track has, if within three sigma.
    if (!std::isfinite(t.depth)) return;
    for (double c : d.depth_alternatives) {
      if (std::abs(c - t.depth) < std::abs(z - t.depth)) z = c;
    }
    if (std::abs(z - t.depth) > 3.0 * std::sqrt(t.depth_var + r)) return;
  }
  if (!std::isfinite(t.depth)) {
    t.depth = z;
    t.depth_var = r;
    return;
  }
  const double k = t.depth_var / (t.depth_var + r);
  t.depth += k * (z - t.depth);
  t.depth_var *= 1.0 - k;
}

}  // namespace

Association parse_association(const std::string& name) {
  if (name == "likelihood") return Association::Likelihood;
  if (name == "in_order") return Association::InOrder;
  throw std::invalid_argument("tracker association '" + name + "': one of likelihood, in_order");
}

namespace {

// A track's state, covariance and depth variance moved by the constant-velocity model from its time to `to`. `to` may
// be earlier than the track's time (Tracker::update no longer asks for that since sonar-uc5: an earlier detection is
// taken at the track's time). The process noise then still grows the uncertainty: Q = G G^T q^2 is an outer product,
// positive semi-definite for either sign of dt, and the depth's random walk grows by q_d^2 |dt| (ours: a variance never
// shrinks by predicting; the same for dt >= 0 as before). python/gpusonar/detect.py Tracker._predict is the same.
struct Prior {
  V4 x;
  M4 p;
  double depth_var;
};

Prior predict(const Track& t, double to, const TrackerOptions& o) {
  const double dt = to - t.time;
  M4 f = M4::Identity();
  f(0, 2) = f(1, 3) = dt;
  Eigen::Matrix<double, 4, 2> g;
  g << 0.5 * dt * dt, 0.0, 0.0, 0.5 * dt * dt, dt, 0.0, 0.0, dt;
  const M4 q = g * g.transpose() * (o.process_noise * o.process_noise);
  Eigen::Map<const V4> x(t.state);
  Eigen::Map<const M4> p(t.cov);
  Prior out;
  out.x = f * x;
  out.p = f * p * f.transpose() + q;
  out.depth_var = std::isfinite(t.depth_var) ? t.depth_var + o.depth_process_noise * o.depth_process_noise * std::abs(dt) : t.depth_var;
  return out;
}

TrackerTapTrack tap_track(const Track& t) {
  TrackerTapTrack o;
  o.id = t.id;
  std::copy(t.state, t.state + 4, o.state);
  std::copy(t.cov, t.cov + 16, o.cov);
  o.hits = t.hits;
  o.misses = t.misses;
  o.confirmed = t.confirmed;
  o.time = t.time;
  return o;
}

}  // namespace

std::vector<const Track*> Tracker::update(double time, const std::vector<Detection>& detections) {
  H24 h = H24::Zero();
  h(0, 0) = h(1, 1) = 1.0;
  // Each detection's own time (D-156): the ping's when it carries none (any time that is not finite, as detect.py).
  std::vector<double> when(detections.size());
  for (std::size_t j = 0; j < detections.size(); ++j) when[j] = std::isfinite(detections[j].time) ? detections[j].time : time;
  // The time a track compares with and takes a detection at: the detection's, or the track's own when the detection's is
  // earlier (sonar-uc5, ours; detect.py Tracker._not_before has the reasons): no history steps back in time.
  auto not_before = [&](std::size_t ti, double when) { return when < tracks_[ti].time ? tracks_[ti].time : when; };
  // Each track predicted to each distinct detection time, once.
  std::vector<std::vector<std::pair<double, Prior>>> cache(tracks_.size());
  auto predicted = [&](std::size_t ti, double when) -> const Prior& {
    const double to = not_before(ti, when);
    for (const auto& [at, pr] : cache[ti]) {
      if (at == to) return pr;
    }
    cache[ti].emplace_back(to, predict(tracks_[ti], to, opt_));
    return cache[ti].back().second;
  };
  // The position of the detection each track took this ping (NaN: none), for its history row (D-157).
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<V2> took(tracks_.size(), V2(nan, nan));
  // The Kalman update of one track, predicted to the detection's time, with that detection.
  auto take = [&](Track& t, const Detection& d, const Prior& prior, double at) {
    Eigen::Map<V4> x(t.state);
    Eigen::Map<M4> p(t.cov);
    x = prior.x;
    p = prior.p;
    t.depth_var = prior.depth_var;
    t.time = at < t.time ? t.time : at;   // not_before
    const M2 s = h * p * h.transpose() + measurement_cov(d, opt_);
    const Eigen::Matrix<double, 4, 2> k = p * h.transpose() * s.inverse();
    const V2 z = xy(d);
    x = x + k * (z - h * x);
    p = (M4::Identity() - k * h) * p;
    took[static_cast<std::size_t>(&t - tracks_.data())] = z;
    ++t.hits;
    t.snr.push_back(d.snr_db);
    t.cells.push_back(d.cells);
    take_depth(t, d);
    t.misses = 0;
    t.confirmed = t.confirmed || t.hits >= opt_.confirm_hits;
  };
  // The opt-in tap (read only): every pair compared, kept below when gated or a track's nearest.
  const bool tap = tap_;
  std::vector<TrackerTapPair> compared;
  std::vector<std::ptrdiff_t> took_index(tracks_.size(), -1);
  if (tap) {
    last_tap_ = TrackerTap{};
    last_tap_.time = time;
    for (const Track& t : tracks_) {
      TrackerTapTrack b;
      b.id = t.id;
      b.hits = t.hits;
      b.misses = t.misses;
      b.confirmed = t.confirmed;
      last_tap_.before.push_back(b);
    }
  }
  auto record = [&](std::size_t ti, std::size_t j, double d2, double cost, const Prior& pr, const M2& s) {
    TrackerTapPair q;
    q.track = tracks_[ti].id;
    q.detection = static_cast<std::int64_t>(j);
    q.d2 = d2;
    q.cost = cost;
    q.x = pr.x[0];
    q.y = pr.x[1];
    q.sxx = s(0, 0);
    q.sxy = s(0, 1);
    q.syy = s(1, 1);
    compared.push_back(q);
  };
  std::vector<std::size_t> unused;
  if (opt_.association == Association::Likelihood) {
    // Most probable gated pairs first, confirmed tracks before tentative ones (D-141).
    struct Pair {
      bool tentative;
      double cost;
      std::size_t track;
      std::size_t detection;
    };
    std::vector<Pair> pairs;
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
      const Track& t = tracks_[ti];
      for (std::size_t j = 0; j < detections.size(); ++j) {
        const Detection& d = detections[j];
        const Prior& pr = predicted(ti, when[j]);
        const M2 s = h * pr.p * h.transpose() + measurement_cov(d, opt_);
        const V2 y = xy(d) - h * pr.x;
        const double d2 = y.dot(s.inverse() * y);
        if (d2 < opt_.gate * opt_.gate) pairs.push_back({!t.confirmed, d2 + std::log(s.determinant()), ti, j});
        if (tap) record(ti, j, d2, d2 + std::log(s.determinant()), pr, s);
      }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
      return std::tie(a.tentative, a.cost, a.track, a.detection) < std::tie(b.tentative, b.cost, b.track, b.detection);
    });
    std::vector<std::ptrdiff_t> taken(tracks_.size(), -1);
    std::vector<char> used(detections.size(), 0);
    for (const Pair& pr : pairs) {
      if (taken[pr.track] >= 0 || used[pr.detection]) continue;
      taken[pr.track] = static_cast<std::ptrdiff_t>(pr.detection);
      used[pr.detection] = 1;
    }
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
      if (taken[ti] >= 0) {
        const auto j = static_cast<std::size_t>(taken[ti]);
        const Prior pr = predicted(ti, when[j]);
        took_index[ti] = static_cast<std::ptrdiff_t>(j);
        take(tracks_[ti], detections[j], pr, when[j]);
      } else {
        ++tracks_[ti].misses;
      }
    }
    for (std::size_t j = 0; j < detections.size(); ++j) {
      if (!used[j]) unused.push_back(j);
    }
  } else {
    unused.resize(detections.size());
    for (std::size_t j = 0; j < unused.size(); ++j) unused[j] = j;
    // Confirmed tracks choose first, each taking its nearest gated detection.
    std::vector<std::size_t> order(tracks_.size());
    for (std::size_t ti = 0; ti < order.size(); ++ti) order[ti] = ti;
    std::stable_sort(order.begin(), order.end(),
                     [this](std::size_t a, std::size_t b) { return tracks_[a].confirmed && !tracks_[b].confirmed; });
    for (std::size_t ti : order) {
      Track& t = tracks_[ti];
      double best_d2 = opt_.gate * opt_.gate;
      std::ptrdiff_t best = -1;
      for (std::size_t k = 0; k < unused.size(); ++k) {
        const Detection& d = detections[unused[k]];
        const Prior& pr = predicted(ti, when[unused[k]]);
        const M2 s = h * pr.p * h.transpose() + measurement_cov(d, opt_);
        const V2 y = xy(d) - h * pr.x;
        const double d2 = y.dot(s.inverse() * y);
        if (tap) record(ti, unused[k], d2, std::numeric_limits<double>::quiet_NaN(), pr, s);
        if (d2 < best_d2) {
          best = static_cast<std::ptrdiff_t>(k);
          best_d2 = d2;
        }
      }
      if (best < 0) {
        ++t.misses;
        continue;
      }
      const std::size_t j = unused[static_cast<std::size_t>(best)];
      unused.erase(unused.begin() + best);
      const Prior pr = predicted(ti, when[j]);
      took_index[ti] = static_cast<std::ptrdiff_t>(j);
      take(t, detections[j], pr, when[j]);
    }
  }
  for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
    Track& t = tracks_[ti];
    if (t.misses > 0 && t.time < time) {
      // It took nothing this ping: where it is at the ping. A track that already holds at a later time (its last
      // detection at transmit + R/c after this ping's transmit: a ping interval shorter than R/c) stays where it is, so
      // a track never moves back in time.
      const Prior pr = predicted(ti, time);
      Eigen::Map<V4>(t.state) = pr.x;
      Eigen::Map<M4>(t.cov) = pr.p;
      t.depth_var = pr.depth_var;
      t.time = time;
    }
    t.history.push_back({t.time, t.state[0], t.state[1], took[ti][0], took[ti][1], time});
  }
  if (tap) {
    // Kept: inside the gate, or a track's nearest (the first smallest d2), in the order compared (detect.py _tap_pairs_kept).
    std::vector<std::ptrdiff_t> nearest(tracks_.size(), -1);
    for (std::size_t i = 0; i < compared.size(); ++i) {
      for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (tracks_[ti].id != compared[i].track) continue;
        if (nearest[ti] < 0 || compared[i].d2 < compared[static_cast<std::size_t>(nearest[ti])].d2) nearest[ti] = static_cast<std::ptrdiff_t>(i);
      }
    }
    for (std::size_t i = 0; i < compared.size(); ++i) {
      TrackerTapPair q = compared[i];
      q.gated = q.d2 < opt_.gate * opt_.gate;
      q.nearest = std::find(nearest.begin(), nearest.end(), static_cast<std::ptrdiff_t>(i)) != nearest.end();
      if (q.gated || q.nearest) last_tap_.pairs.push_back(q);
    }
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
      const Track& t = tracks_[ti];
      if (took_index[ti] >= 0) last_tap_.took.emplace_back(t.id, static_cast<std::int64_t>(took_index[ti]));
    }
    for (const Track& t : tracks_) {
      if (t.misses > 0) last_tap_.missed.push_back(t.id);
    }
    for (const Track& t : tracks_) {
      if (t.misses > opt_.max_misses) last_tap_.dropped.push_back(tap_track(t));
    }
    for (const Track& t : tracks_) {
      bool was = false;
      for (const TrackerTapTrack& b : last_tap_.before) {
        if (b.id == t.id) was = b.confirmed;
      }
      if (t.confirmed && !was) last_tap_.confirmed_now.push_back(t.id);
    }
  }
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [this](const Track& t) { return t.misses > opt_.max_misses; }),
                tracks_.end());
  for (std::size_t j : unused) {
    const Detection& d = detections[j];
    Track tr;
    tr.id = next_id_++;
    tr.time = when[j];
    const V2 p0 = xy(d);
    tr.state[0] = p0[0];
    tr.state[1] = p0[1];
    Eigen::Map<M4> p(tr.cov);
    p.setZero();
    p.topLeftCorner<2, 2>() = measurement_cov(d, opt_);
    p(2, 2) = p(3, 3) = opt_.initial_speed_sigma * opt_.initial_speed_sigma;
    tr.history.push_back({when[j], p0[0], p0[1], p0[0], p0[1], time});
    tr.snr.push_back(d.snr_db);
    tr.cells.push_back(d.cells);
    take_depth(tr, d);
    if (tap) last_tap_.born.emplace_back(tr.id, static_cast<std::int64_t>(j));
    tracks_.push_back(std::move(tr));
  }
  if (tap) {
    for (const Track& t : tracks_) last_tap_.after.push_back(tap_track(t));
  }
  std::vector<const Track*> confirmed;
  for (const Track& t : tracks_) {
    if (t.confirmed) confirmed.push_back(&t);
  }
  return confirmed;
}

}  // namespace sonar::live
