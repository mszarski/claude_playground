#include <algorithm>
#include <cmath>

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

}  // namespace

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
  }
  std::vector<std::size_t> unused(detections.size());
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
    const M2 s = h * p * h.transpose() + measurement_cov(d, opt_);
    const Eigen::Matrix<double, 4, 2> k = p * h.transpose() * s.inverse();
    x = x + k * (xy(d) - h * x);
    p = (M4::Identity() - k * h) * p;
    ++t->hits;
    t->snr.push_back(d.snr_db);
    t->misses = 0;
    t->confirmed = t->confirmed || t->hits >= opt_.confirm_hits;
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
    tracks_.push_back(std::move(tr));
  }
  std::vector<const Track*> confirmed;
  for (const Track& t : tracks_) {
    if (t.confirmed) confirmed.push_back(&t);
  }
  return confirmed;
}

}  // namespace sonar::live
