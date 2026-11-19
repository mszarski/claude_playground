#include <algorithm>
#include <cmath>
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
  p(0, 0) = o.range_sigma * o.range_sigma;
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

std::vector<const Track*> Tracker::update(double time, const std::vector<Detection>& detections) {
  const double dt = started_ ? time - time_ : 0.0;
  started_ = true;
  time_ = time;
  M4 f = M4::Identity();
  f(0, 2) = f(1, 3) = dt;
  Eigen::Matrix<double, 4, 2> g;
  g << 0.5 * dt * dt, 0.0, 0.0, 0.5 * dt * dt, dt, 0.0, 0.0, dt;
  const M4 q = g * g.transpose() * (opt_.process_noise * opt_.process_noise);
  H24 h = H24::Zero();
  h(0, 0) = h(1, 1) = 1.0;
  for (Track& t : tracks_) {
    Eigen::Map<V4> x(t.state);
    Eigen::Map<M4> p(t.cov);
    x = f * x;
    p = f * p * f.transpose() + q;
    if (std::isfinite(t.depth_var)) t.depth_var += opt_.depth_process_noise * opt_.depth_process_noise * dt;
  }
  // The Kalman update of one track with one detection.
  auto take = [&](Track& t, const Detection& d) {
    Eigen::Map<V4> x(t.state);
    Eigen::Map<M4> p(t.cov);
    const M2 s = h * p * h.transpose() + measurement_cov(d, opt_);
    const Eigen::Matrix<double, 4, 2> k = p * h.transpose() * s.inverse();
    x = x + k * (xy(d) - h * x);
    p = (M4::Identity() - k * h) * p;
    ++t.hits;
    t.snr.push_back(d.snr_db);
    take_depth(t, d);
    t.misses = 0;
    t.confirmed = t.confirmed || t.hits >= opt_.confirm_hits;
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
      Eigen::Map<const V4> x(t.state);
      Eigen::Map<const M4> p(t.cov);
      for (std::size_t j = 0; j < detections.size(); ++j) {
        const Detection& d = detections[j];
        const M2 s = h * p * h.transpose() + measurement_cov(d, opt_);
        const V2 y = xy(d) - h * x;
        const double d2 = y.dot(s.inverse() * y);
        if (d2 < opt_.gate * opt_.gate) pairs.push_back({!t.confirmed, d2 + std::log(s.determinant()), ti, j});
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
      if (taken[ti] >= 0) take(tracks_[ti], detections[static_cast<std::size_t>(taken[ti])]);
      else ++tracks_[ti].misses;
    }
    for (std::size_t j = 0; j < detections.size(); ++j) {
      if (!used[j]) unused.push_back(j);
    }
  } else {
    unused.resize(detections.size());
    for (std::size_t j = 0; j < unused.size(); ++j) unused[j] = j;
    // Confirmed tracks choose first, each taking its nearest gated detection.
    std::vector<Track*> order;
    for (Track& t : tracks_) order.push_back(&t);
    std::stable_sort(order.begin(), order.end(), [](const Track* a, const Track* b) { return a->confirmed && !b->confirmed; });
    for (Track* t : order) {
      Eigen::Map<V4> x(t->state);
      Eigen::Map<M4> p(t->cov);
      double best_d2 = opt_.gate * opt_.gate;
      std::ptrdiff_t best = -1;
      for (std::size_t k = 0; k < unused.size(); ++k) {
        const Detection& d = detections[unused[k]];
        const M2 s = h * p * h.transpose() + measurement_cov(d, opt_);
        const V2 y = xy(d) - h * x;
        const double d2 = y.dot(s.inverse() * y);
        if (d2 < best_d2) {
          best = static_cast<std::ptrdiff_t>(k);
          best_d2 = d2;
        }
      }
      if (best < 0) {
        ++t->misses;
        continue;
      }
      const Detection d = detections[unused[static_cast<std::size_t>(best)]];
      unused.erase(unused.begin() + best);
      take(*t, d);
    }
  }
  for (Track& t : tracks_) t.history.push_back({time, t.state[0], t.state[1]});
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [this](const Track& t) { return t.misses > opt_.max_misses; }),
                tracks_.end());
  for (std::size_t j : unused) {
    const Detection& d = detections[j];
    Track tr;
    tr.id = next_id_++;
    const V2 p0 = xy(d);
    tr.state[0] = p0[0];
    tr.state[1] = p0[1];
    Eigen::Map<M4> p(tr.cov);
    p.setZero();
    p.topLeftCorner<2, 2>() = measurement_cov(d, opt_);
    p(2, 2) = p(3, 3) = opt_.initial_speed_sigma * opt_.initial_speed_sigma;
    tr.history.push_back({time, p0[0], p0[1]});
    tr.snr.push_back(d.snr_db);
    take_depth(tr, d);
    tracks_.push_back(std::move(tr));
  }
  std::vector<const Track*> confirmed;
  for (const Track& t : tracks_) {
    if (t.confirmed) confirmed.push_back(&t);
  }
  return confirmed;
}

}  // namespace sonar::live
