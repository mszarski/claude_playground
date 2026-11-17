#include "sonar/sim/paths.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sonar/core/parallel.hpp"

namespace sonar::sim {

namespace {

using cd = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;

/// x^n for a small non-negative integer n by binary powering: the products NumPy and
/// Python form for an integer exponent.
cd powu(cd x, std::int64_t n) {
  cd r{1.0, 0.0};
  cd p = x;
  for (std::int64_t mask = 1; mask > 0 && n >= mask; mask <<= 1) {
    if ((n & mask) != 0) r = r * p;
    p = p * p;
  }
  return r;
}

double clip01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

/// Complex quotient by Smith's method with a reciprocal, the operation order NumPy uses,
/// so that amplitudes agree with the Python providers to the last bits.
cd divide(cd a, cd b) {
  if (std::abs(b.real()) >= std::abs(b.imag())) {
    const double rat = b.imag() / b.real();
    const double scl = 1.0 / (b.real() + b.imag() * rat);
    return {(a.real() + a.imag() * rat) * scl, (a.imag() - a.real() * rat) * scl};
  }
  const double rat = b.real() / b.imag();
  const double scl = 1.0 / (b.imag() + b.real() * rat);
  return {(a.real() * rat + a.imag()) * scl, (a.imag() * rat - a.real()) * scl};
}
cd divide(cd a, double b) { return divide(a, cd{b, 0.0}); }

/// f(i) for every field point. A point is a few dozen operations, so threads take
/// blocks of points, and only when there are enough of them to repay the start.
template <typename F>
void for_each_point(std::int64_t n, const F& f) {
  constexpr std::int64_t block = 256;
  if (n < 64 * block) {
    for (std::int64_t i = 0; i < n; ++i) f(i);
    return;
  }
  parallel_for_host((n + block - 1) / block, [&](std::int64_t b) {
    const std::int64_t end = std::min(n, (b + 1) * block);
    for (std::int64_t i = b * block; i < end; ++i) f(i);
  });
}

void check_points(const double* points, std::int64_t n_points) {
  if (n_points < 0 || (n_points > 0 && points == nullptr)) throw std::invalid_argument("paths: bad field points");
}

}  // namespace

void EigenraySet::resize(std::int64_t points, std::int64_t rays) {
  n_points = points;
  n_rays = rays;
  const auto n = static_cast<std::size_t>(points * rays);
  delay.assign(n, 0.0);
  amplitude.assign(n, cd{});
  launch.assign(3 * n, 0.0);
  arrival.assign(3 * n, 0.0);
  n_surface.assign(n, 0);
  n_bottom.assign(n, 0);
  valid.assign(n, 1);
}

EigenraySet free_field_eigenrays(double sound_speed, const Vec3d& source, const double* points,
                                 std::int64_t n_points) {
  check_points(points, n_points);
  EigenraySet out;
  out.sound_speed = sound_speed;
  out.resize(n_points, 1);
  for_each_point(n_points, [&](std::int64_t i) {
    const Vec3d d{points[3 * i] - source.x, points[3 * i + 1] - source.y, points[3 * i + 2] - source.z};
    const double r = norm(d);
    out.delay[i] = r / sound_speed;
    out.amplitude[i] = cd{1.0 / r, 0.0};
    for (int a = 0; a < 3; ++a) {
      out.launch[3 * i + a] = d[a] / r;
      out.arrival[3 * i + a] = d[a] / r;
    }
  });
  return out;
}

std::vector<ImageRay> image_series(const ImageSourceConfig& config) {
  std::vector<ImageRay> series;
  const auto with_walls = [&](const ImageRay& ray) {
    series.push_back(ray);
    for (std::size_t w = 0; w < config.walls.size(); ++w) {
      ImageRay walled = ray;
      walled.wall = static_cast<int>(w);
      series.push_back(walled);
    }
  };
  with_walls(ImageRay{});
  for (const bool surface_first : {true, false}) {
    if ((surface_first && !config.has_surface) || (!surface_first && !config.has_bottom)) continue;
    ImageRay ray;
    bool next_surface = surface_first;
    for (int order = 0; order < config.max_order; ++order) {
      if (next_surface) {
        if (!config.has_surface) break;
        ray.mirrors.push_back(config.surface_z);
        ++ray.n_surface;
      } else {
        if (!config.has_bottom) break;
        ray.mirrors.push_back(config.bottom_z);
        ++ray.n_bottom;
      }
      next_surface = !next_surface;
      with_walls(ray);
    }
  }
  return series;
}

std::complex<double> rayleigh_reflection(double grazing, double c_water, const FluidBottom& bottom) {
  const double delta = bottom.attenuation / (40.0 * kPi * std::log10(std::exp(1.0)));  // loss tangent
  const cd c_b = bottom.speed / cd{1.0, delta};
  const double m = bottom.density / bottom.water_density;
  const cd n = c_water / c_b;
  const double cg = std::cos(grazing);
  cd root = std::sqrt(n * n - cg * cg + cd{0.0, 0.0});
  if (root.imag() < 0.0) root = -root;  // decaying branch in the sediment
  const double ms = m * std::sin(grazing);
  return divide(ms - root, ms + root);
}

EigenraySet image_source_eigenrays(const ImageSourceConfig& config, const Vec3d& source,
                                   const double* points, std::int64_t n_points) {
  check_points(points, n_points);
  for (const Wall& w : config.walls) {
    if (w.axis != 0 && w.axis != 1) throw std::invalid_argument("image_source: wall axis must be 0 or 1");
  }
  const std::vector<ImageRay> series = image_series(config);
  const auto k = static_cast<std::int64_t>(series.size());
  const double c = config.sound_speed;
  // Constant coefficient of each ray. With a fluid bottom, a path that meets the bottom
  // takes surface^ns * R(grazing)^nb instead (as the Python provider does, the wall's
  // coefficient is not applied to such a path).
  std::vector<cd> coefficient(series.size());
  for (std::size_t j = 0; j < series.size(); ++j) {
    coefficient[j] = powu(config.surface_coefficient, series[j].n_surface) *
                     powu(config.bottom_coefficient, series[j].n_bottom);
    if (series[j].wall >= 0) coefficient[j] = coefficient[j] * config.walls[series[j].wall].coefficient;
  }
  EigenraySet out;
  out.sound_speed = c;
  out.resize(n_points, k);
  for_each_point(n_points, [&](std::int64_t i) {
    for (std::int64_t j = 0; j < k; ++j) {
      const ImageRay& ray = series[static_cast<std::size_t>(j)];
      double img[3] = {points[3 * i], points[3 * i + 1], points[3 * i + 2]};
      for (const double plane : ray.mirrors) img[2] = 2.0 * plane - img[2];
      if (ray.wall >= 0) {
        const Wall& w = config.walls[static_cast<std::size_t>(ray.wall)];
        img[w.axis] = 2.0 * w.position - img[w.axis];
      }
      const Vec3d d{img[0] - source.x, img[1] - source.y, img[2] - source.z};
      const double r = norm(d);
      const Vec3d u = d / r;
      const std::int64_t e = i * k + j;
      out.delay[e] = r / c;
      cd coef = coefficient[static_cast<std::size_t>(j)];
      if (config.has_fluid_bottom && ray.n_bottom > 0) {
        // In an isovelocity layer every bounce of one image path has the grazing angle
        // of the straight line to the image.
        const double grazing = std::asin(clip01(std::abs(u.z)));
        coef = powu(config.surface_coefficient, ray.n_surface) *
               powu(rayleigh_reflection(grazing, c, config.fluid_bottom), ray.n_bottom);
      }
      out.amplitude[e] = divide(coef, r);
      const bool odd = ((ray.n_surface + ray.n_bottom) % 2) == 1;
      for (int a = 0; a < 3; ++a) {
        out.launch[3 * e + a] = u[a];
        // Direction of travel at the real point: the vertical part mirrors once per bounce.
        out.arrival[3 * e + a] = (a == 2 && odd) ? -u[a] : u[a];
      }
      out.n_surface[e] = ray.n_surface;
      out.n_bottom[e] = ray.n_bottom;
    }
  });
  return out;
}

void snell_crossing(const Vec3d& source, const double* points, std::int64_t n_points, double c_water,
                    double c_sediment, double interface_z, double* time, double* crossing) {
  check_points(points, n_points);
  const double a = interface_z - source.z;
  for_each_point(n_points, [&](std::int64_t i) {
    const double px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
    const double b = pz - interface_z;
    const double hx = px - source.x, hy = py - source.y;
    const double dist = std::sqrt(hx * hx + hy * hy);
    if (!(b > 0.0 && a > 0.0)) {
      const Vec3d d{px - source.x, py - source.y, pz - source.z};
      time[i] = norm(d) / c_water;
      crossing[3 * i] = px;
      crossing[3 * i + 1] = py;
      crossing[3 * i + 2] = pz;
      return;
    }
    // Newton's method on Snell's law for the horizontal distance u to the crossing point.
    double u = dist * a / (a + b);
    for (int it = 0; it < 40; ++it) {
      const double l1 = std::sqrt(u * u + a * a);
      const double l2 = std::sqrt((dist - u) * (dist - u) + b * b);
      const double f = u / (c_water * l1) - (dist - u) / (c_sediment * l2);
      const double df = a * a / (c_water * std::pow(l1, 3.0)) + b * b / (c_sediment * std::pow(l2, 3.0));
      u = u - f / df;
      u = u < 0.0 ? 0.0 : (u > dist ? dist : u);
    }
    time[i] = std::sqrt(u * u + a * a) / c_water + std::sqrt((dist - u) * (dist - u) + b * b) / c_sediment;
    const double ux = dist > 0.0 ? hx / dist : 0.0;
    const double uy = dist > 0.0 ? hy / dist : 0.0;
    crossing[3 * i] = source.x + ux * u;
    crossing[3 * i + 1] = source.y + uy * u;
    crossing[3 * i + 2] = interface_z;
  });
}

EigenraySet sediment_eigenrays(const SedimentConfig& config, const Vec3d& source, const double* points,
                               std::int64_t n_points) {
  check_points(points, n_points);
  EigenraySet out;
  out.sound_speed = config.sound_speed;
  out.resize(n_points, 1);
  std::vector<double> crossing(static_cast<std::size_t>(3 * n_points));
  snell_crossing(source, points, n_points, config.sound_speed, config.sediment_speed, config.interface_z,
                 out.delay.data(), crossing.data());
  for_each_point(n_points, [&](std::int64_t i) {
    const Vec3d cross{crossing[3 * i], crossing[3 * i + 1], crossing[3 * i + 2]};
    const Vec3d point{points[3 * i], points[3 * i + 1], points[3 * i + 2]};
    const Vec3d l = cross - source;
    const Vec3d launch = l / norm(l);
    const Vec3d d = point - cross;
    const double length = norm(d);
    const Vec3d arrival = length > 0.0 ? d / std::max(length, 1e-300) : launch;
    for (int a = 0; a < 3; ++a) {
      out.launch[3 * i + a] = launch[a];
      out.arrival[3 * i + a] = arrival[a];
    }
    const bool below = point.z > config.interface_z;
    out.amplitude[i] = divide(below ? config.transmission : cd{1.0, 0.0}, config.sound_speed * out.delay[i]);
  });
  return out;
}

EigenrayView view(const EigenraySet& set) {
  return {set.n_points,         set.n_rays,        set.sound_speed,    set.delay.data(),
          set.amplitude.data(), set.launch.data(), set.arrival.data(), set.valid.data()};
}

bool any_moving(const double* velocities, std::int64_t n_points) {
  if (velocities == nullptr) return false;
  // NaN counts as moving, as numpy.any does.
  for (std::int64_t i = 0; i < 3 * n_points; ++i) {
    if (!(velocities[i] == 0.0)) return true;
  }
  return false;
}

std::int64_t two_way_paths(const EigenrayView& out, const EigenrayView& back, const Vec3d& tx_position,
                           const Vec3d& rx_reference, const std::complex<double>* amplitudes,
                           const double* velocities, const std::uint8_t* lambert, const double* normals,
                           double* q, double* p, std::complex<double>* amplitude, double* velocity) {
  if (out.n_points != back.n_points) throw std::invalid_argument("two_way_paths: the two eigenray sets differ in size");
  const std::int64_t s = out.n_points, ko = out.n_rays, kb = back.n_rays;
  if (s > 0 && amplitudes == nullptr) throw std::invalid_argument("two_way_paths: amplitudes are required");
  if (velocity != nullptr && velocities == nullptr) throw std::invalid_argument("two_way_paths: no velocities to copy");
  bool any_lambert = false;
  if (lambert != nullptr) {
    for (std::int64_t i = 0; i < s && !any_lambert; ++i) any_lambert = lambert[i] != 0;
  }
  // One pass in the order scatterer, outbound ray, return ray; kept pairs are written in
  // that order. The work per pair is a few multiplications: it is bound by memory, not
  // arithmetic, so it runs on one thread.
  std::int64_t at = 0;
  for (std::int64_t i = 0; i < s; ++i) {
    const bool weighted = any_lambert && lambert[i] != 0;
    bool has_normal = false;
    Vec3d n{};
    if (weighted && normals != nullptr) {
      n = Vec3d{normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]};
      has_normal = std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z);
    }
    // Lambert's law: the sine of each eigenray's grazing angle at a horizontal boundary,
    // or the cosine from the patch's own normal (a face turned away returns nothing).
    const auto weight = [&](const EigenrayView& set, std::int64_t e) {
      const double* arr = set.arrival + 3 * e;
      if (!has_normal) return std::abs(arr[2]);
      return std::max(-(arr[0] * n.x + arr[1] * n.y + arr[2] * n.z), 0.0);
    };
    for (std::int64_t a = 0; a < ko; ++a) {
      const std::int64_t eo = i * ko + a;
      const double r_out = out.sound_speed * out.delay[eo];
      const cd partial = amplitudes[i] * (out.amplitude[eo] * r_out);
      const double w_out = weighted ? weight(out, eo) : 0.0;
      for (std::int64_t b = 0; b < kb; ++b) {
        const std::int64_t eb = i * kb + b;
        const double r_back = back.sound_speed * back.delay[eb];
        cd value = partial * (back.amplitude[eb] * r_back);
        if (weighted) value = value * std::sqrt(w_out * weight(back, eb));
        if (out.valid[eo] == 0 || back.valid[eb] == 0 || value == cd{0.0, 0.0}) continue;
        for (int x = 0; x < 3; ++x) {
          q[3 * at + x] = tx_position[x] + r_out * out.launch[3 * eo + x];
          p[3 * at + x] = rx_reference[x] + r_back * back.launch[3 * eb + x];
          if (velocity != nullptr) velocity[3 * at + x] = velocities[3 * i + x];
        }
        amplitude[at] = value;
        ++at;
      }
    }
  }
  return at;
}

TwoWayPaths two_way_paths(const EigenrayView& out, const EigenrayView& back, const Vec3d& tx_position,
                          const Vec3d& rx_reference, const std::complex<double>* amplitudes,
                          const double* velocities, const std::uint8_t* lambert, const double* normals) {
  if (out.n_points != back.n_points) throw std::invalid_argument("two_way_paths: the two eigenray sets differ in size");
  const auto room = static_cast<std::size_t>(out.n_points * out.n_rays * back.n_rays);
  const bool moving = any_moving(velocities, out.n_points);
  TwoWayPaths paths;
  paths.q.resize(3 * room);
  paths.p.resize(3 * room);
  paths.amplitude.resize(room);
  if (moving) paths.velocity.resize(3 * room);
  const auto kept = static_cast<std::size_t>(
      two_way_paths(out, back, tx_position, rx_reference, amplitudes, velocities, lambert, normals, paths.q.data(),
                    paths.p.data(), paths.amplitude.data(), moving ? paths.velocity.data() : nullptr));
  paths.q.resize(3 * kept);
  paths.p.resize(3 * kept);
  paths.amplitude.resize(kept);
  if (moving) paths.velocity.resize(3 * kept);
  return paths;
}

}  // namespace sonar::sim
