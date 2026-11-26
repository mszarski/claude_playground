#include "sonar/sim/paths.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "sonar/core/parallel.hpp"
#include "sonar/core/reflection.hpp"

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
/// (sonar/core/reflection.hpp: one copy, shared with the elastic coefficient of the live chain)
cd divide(cd a, cd b) { return smith_divide(a, b); }
cd divide(cd a, double b) { return divide(a, cd{b, 0.0}); }

/// f(i) for every field point. A point is a few dozen operations, so threads take
/// blocks of points, and only when there are enough of them (`min_blocks`) to repay the start.
template <typename F>
void for_each_point(std::int64_t n, const F& f, std::int64_t min_blocks = 64) {
  constexpr std::int64_t block = 256;
  if (n < min_blocks * block) {
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

void check_speed(double c, const char* what) {
  if (!(c > 0.0) || !std::isfinite(c)) {
    throw std::invalid_argument(std::string(what) + " must be stated (positive and finite, m/s); there is no default");
  }
}

bool stated(std::complex<double> v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

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
  check_speed(sound_speed, "free_field: sound_speed");
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
  if (bottom.shear_speed == 0.0) return divide(ms - root, ms + root);
  // Elastic solid (A8): J&R (9.34)-(9.40), the one implementation the live chain calls too (sonar/core/reflection.hpp).
  return elastic_reflection(grazing, c_water, bottom.speed, m, bottom.attenuation, bottom.shear_speed, bottom.shear_attenuation);
}

void check_config(const ImageSourceConfig& config) {
  check_speed(config.sound_speed, "image_source: sound_speed");
  if (config.max_order < 0) {
    throw std::invalid_argument(
        "image_source: max_order (bounces along one path) must be stated; there is no default. Paths with "
        "more bounces are not computed: image_series_truncation gives what a choice leaves out");
  }
  for (const Wall& w : config.walls) {
    if (w.axis != 0 && w.axis != 1) throw std::invalid_argument("image_source: wall axis must be 0 or 1");
    if (!stated(w.coefficient)) {
      throw std::invalid_argument("image_source: a wall needs its reflection coefficient; there is no default");
    }
  }
  if (config.has_surface && !stated(config.surface_coefficient)) {
    throw std::invalid_argument("image_source: surface_coefficient is not a number");
  }
  if (config.has_bottom && config.max_order > 0) {
    if (config.has_fluid_bottom) {
      const FluidBottom& b = config.fluid_bottom;
      if (!(b.speed > 0.0) || !(b.density > 0.0) || !(b.water_density > 0.0) || !std::isfinite(b.speed) ||
          !std::isfinite(b.density) || !std::isfinite(b.attenuation) || b.attenuation < 0.0 ||
          !(b.shear_speed >= 0.0) || !std::isfinite(b.shear_speed) || !(b.shear_attenuation >= 0.0) ||
          !std::isfinite(b.shear_attenuation)) {
        throw std::invalid_argument(
            "image_source: a fluid bottom needs its speed (m/s) and density (ratio to water_density), and an "
            "attenuation >= 0 dB per wavelength (an elastic one a shear_speed and shear_attenuation >= 0); "
            "there is no default sediment");
      }
    } else if (!stated(config.bottom_coefficient)) {
      throw std::invalid_argument(
          "image_source: a bottom needs a reflection coefficient or a fluid sediment; there is no default");
    }
  }
}

double image_series_truncation(const ImageSourceConfig& config, double source_z, double receiver_z,
                               double range) {
  check_config(config);
  const double direct = std::sqrt(range * range + (receiver_z - source_z) * (receiver_z - source_z));
  if (!(direct > 0.0)) throw std::invalid_argument("image_series_truncation: source and receiver coincide");
  const int next = config.max_order + 1;
  double worst = 0.0;
  for (const bool surface_first : {true, false}) {
    if ((surface_first && !config.has_surface) || (!surface_first && !config.has_bottom)) continue;
    // The ray of this series with `next` bounces, if the series reaches that far.
    double z = receiver_z;
    std::int64_t ns = 0, nb = 0;
    bool next_surface = surface_first, ended = false;
    for (int order = 0; order < next; ++order) {
      if ((next_surface && !config.has_surface) || (!next_surface && !config.has_bottom)) {
        ended = true;
        break;
      }
      z = 2.0 * (next_surface ? config.surface_z : config.bottom_z) - z;
      ++(next_surface ? ns : nb);
      next_surface = !next_surface;
    }
    if (ended) continue;
    const double dz = z - source_z;
    const double r = std::sqrt(range * range + dz * dz);
    cd bottom = config.bottom_coefficient;
    if (config.has_fluid_bottom && nb > 0) {
      bottom = rayleigh_reflection(std::asin(clip01(std::abs(dz) / r)), config.sound_speed, config.fluid_bottom);
    }
    const double a = std::abs(powu(config.surface_coefficient, ns) * (nb > 0 ? powu(bottom, nb) : cd{1.0, 0.0})) / r;
    worst = std::max(worst, a * direct);
  }
  return worst;
}

EigenraySet image_source_eigenrays(const ImageSourceConfig& config, const Vec3d& source,
                                   const double* points, std::int64_t n_points) {
  check_points(points, n_points);
  check_config(config);
  const std::vector<ImageRay> series = image_series(config);
  const auto k = static_cast<std::int64_t>(series.size());
  const double c = config.sound_speed;
  // Constant coefficient of each ray. With a fluid bottom, a path that meets the bottom
  // takes surface^ns * R(grazing)^nb instead, times the wall's coefficient if it met one.
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
        if (ray.wall >= 0) coef = coef * config.walls[static_cast<std::size_t>(ray.wall)].coefficient;
      }
      out.amplitude[e] = divide(coef, r);
      const bool odd = ((ray.n_surface + ray.n_bottom) % 2) == 1;
      const int wall_axis = ray.wall >= 0 ? config.walls[static_cast<std::size_t>(ray.wall)].axis : -1;
      for (int a = 0; a < 3; ++a) {
        out.launch[3 * e + a] = u[a];
        // Direction of travel at the real point: the direction to the image mirrored in every plane met, the
        // vertical part once per surface or bottom bounce and the wall's horizontal part at its bounce (before
        // sonar-8qn9 the wall's was left out).
        const bool flip = (a == 2 && odd) || a == wall_axis;
        out.arrival[3 * e + a] = flip ? -u[a] : u[a];
      }
      out.n_surface[e] = ray.n_surface;
      out.n_bottom[e] = ray.n_bottom;
    }
  });
  return out;
}

namespace {

/// A point of the general construction is hundreds of operations a ray (every plane, every leg), so threads
/// start at four blocks of points rather than the 64 of the providers above.
constexpr std::int64_t kPlaneMinBlocks = 4;

/// The plane's axis when its normal has one non-zero component (it then mirrors as 2 c - x on that axis, the
/// arithmetic of the flat series, so a level bottom gives its images to the bit), else -1.
int plane_axis(const Plane& p) {
  int axis = -1, count = 0;
  for (int a = 0; a < 3; ++a) {
    if (p.normal[a] != 0.0) {
      axis = a;
      ++count;
    }
  }
  return count == 1 ? axis : -1;
}

/// The arithmetic of `propagation._Plane`, operation for operation (the dot product summed x, y, z).
struct PlaneOps {
  Plane plane;
  int axis = -1;

  double distance(const double* x) const {
    const double d0 = x[0] - plane.point.x, d1 = x[1] - plane.point.y, d2 = x[2] - plane.point.z;
    return d0 * plane.normal.x + d1 * plane.normal.y + d2 * plane.normal.z;
  }
  void mirror(const double* x, double* y) const {
    if (axis >= 0) {
      y[0] = x[0];
      y[1] = x[1];
      y[2] = x[2];
      y[axis] = 2.0 * plane.point[axis] - x[axis];
      return;
    }
    const double twice = 2.0 * distance(x);
    for (int a = 0; a < 3; ++a) y[a] = x[a] - twice * plane.normal[a];
  }
  bool on_face(const double* x) const {
    for (int a = 0; a < 3; ++a) {
      if (plane.has_bound[a] &&
          !(x[a] >= plane.lower[a] - kPlaneTolerance && x[a] <= plane.upper[a] + kPlaneTolerance)) {
        return false;
      }
    }
    return true;
  }
  /// The segment a -> b passes through the face (strictly: both ends off the plane).
  bool blocks(const double* a, const double* b) const {
    const double da = distance(a), db = distance(b);
    const bool cross = (da > kPlaneTolerance && db < -kPlaneTolerance) || (da < -kPlaneTolerance && db > kPlaneTolerance);
    if (!cross) return false;
    const double t = da / (da - db);
    double x[3];
    for (int k = 0; k < 3; ++k) x[k] = a[k] + t * (b[k] - a[k]);
    return on_face(x);
  }
};

/// One sequence of the construction: plane indices in the order met from the source, the image of every point
/// [S, 3], and where the sequence is possible [S].
struct PlaneNode {
  std::vector<int> seq;
  std::vector<double> image;
  std::vector<std::uint8_t> possible;
  int n_wall = 0;
  int n_vertical = 0;
};

}  // namespace

Plane make_plane(PlaneKind kind, const Vec3d& point, const Vec3d& normal, std::complex<double> coefficient) {
  Plane p;
  p.kind = kind;
  p.point = point;
  const double length = norm(normal);
  if (!(length > 0.0) || !std::isfinite(length)) throw std::invalid_argument("make_plane: the normal is a zero vector");
  p.normal = normal / length;
  p.coefficient = coefficient;
  return p;
}

void check_config(const PlaneImageConfig& config) {
  check_speed(config.sound_speed, "image_source: sound_speed");
  if (config.max_order < 0 || config.wall_order < 0) {
    throw std::invalid_argument(
        "image_source planes: max_order (surface and bottom bounces) and wall_order (wall bounces) must be stated "
        "(0 or more); there is no default");
  }
  bool bottom = false;
  for (const Plane& p : config.planes) {
    const double n2 = p.normal.x * p.normal.x + p.normal.y * p.normal.y + p.normal.z * p.normal.z;
    if (!std::isfinite(p.point.x) || !std::isfinite(p.point.y) || !std::isfinite(p.point.z) ||
        !(std::abs(n2 - 1.0) < 1e-12)) {
      throw std::invalid_argument("image_source planes: a plane needs a finite point and a unit normal (make_plane)");
    }
    for (int a = 0; a < 3; ++a) {
      if (p.has_bound[a] && !(p.lower[a] <= p.upper[a])) {
        throw std::invalid_argument("image_source planes: bounds need lower <= upper");
      }
    }
    if (p.kind == PlaneKind::Bottom) {
      bottom = true;
    } else if (!stated(p.coefficient)) {
      throw std::invalid_argument(p.kind == PlaneKind::Wall
                                      ? "image_source: a wall needs its reflection coefficient; there is no default"
                                      : "image_source: surface_coefficient is not a number");
    }
  }
  if (bottom && config.max_order > 0) {
    if (config.has_fluid_bottom) {
      const FluidBottom& b = config.fluid_bottom;
      if (!(b.speed > 0.0) || !(b.density > 0.0) || !(b.water_density > 0.0) || !std::isfinite(b.speed) ||
          !std::isfinite(b.density) || !std::isfinite(b.attenuation) || b.attenuation < 0.0 ||
          !(b.shear_speed >= 0.0) || !std::isfinite(b.shear_speed) || !(b.shear_attenuation >= 0.0) ||
          !std::isfinite(b.shear_attenuation)) {
        throw std::invalid_argument(
            "image_source: a fluid bottom needs its speed (m/s) and density (ratio to water_density), and an "
            "attenuation >= 0 dB per wavelength (an elastic one a shear_speed and shear_attenuation >= 0); "
            "there is no default sediment");
      }
    } else if (!stated(config.bottom_coefficient)) {
      throw std::invalid_argument(
          "image_source: a bottom needs a reflection coefficient or a fluid sediment; there is no default");
    }
  }
}

EigenraySet plane_image_eigenrays(const PlaneImageConfig& config, const Vec3d& source, const double* points,
                                  std::int64_t n_points) {
  check_points(points, n_points);
  check_config(config);
  const double c = config.sound_speed;
  const double tol = kPlaneTolerance;
  const auto s = static_cast<std::size_t>(n_points);
  std::vector<PlaneOps> planes;
  for (const Plane& p : config.planes) planes.push_back({p, plane_axis(p)});
  std::vector<int> unbounded, bounded;
  for (std::size_t k = 0; k < planes.size(); ++k) (planes[k].plane.bounded() ? bounded : unbounded).push_back(static_cast<int>(k));

  // The sequences (`_PlaneImages.sequences`): the direct path, then each level grown from the last, a node kept
  // when it is possible for some point.
  std::vector<PlaneNode> nodes(1);
  nodes[0].image.assign(points, points + 3 * s);
  nodes[0].possible.assign(s, 1);
  for_each_point(n_points, [&](std::int64_t i) {
    for (const int k : unbounded) {
      if (!(planes[static_cast<std::size_t>(k)].distance(points + 3 * i) >= -tol)) nodes[0].possible[i] = 0;
    }
  }, kPlaneMinBlocks);
  std::size_t begin = 0, end = 1;
  while (begin < end) {
    for (std::size_t f = begin; f < end; ++f) {
      for (std::size_t k = 0; k < planes.size(); ++k) {
        const PlaneNode& from = nodes[f];  // re-read: push_back below may move the nodes
        const bool wall = planes[k].plane.kind == PlaneKind::Wall;
        if (!from.seq.empty() && from.seq.front() == static_cast<int>(k)) continue;
        if (wall ? from.n_wall + 1 > config.wall_order : from.n_vertical + 1 > config.max_order) continue;
        PlaneNode grown;
        grown.seq.reserve(from.seq.size() + 1);
        grown.seq.push_back(static_cast<int>(k));
        grown.seq.insert(grown.seq.end(), from.seq.begin(), from.seq.end());
        grown.n_wall = from.n_wall + (wall ? 1 : 0);
        grown.n_vertical = from.n_vertical + (wall ? 0 : 1);
        grown.image.resize(3 * s);
        grown.possible.resize(s);
        const PlaneOps& plane = planes[k];
        for_each_point(n_points, [&](std::int64_t i) {
          const double* img = from.image.data() + 3 * i;
          grown.possible[i] = (from.possible[i] != 0 && plane.distance(img) >= -tol) ? 1 : 0;
          plane.mirror(img, grown.image.data() + 3 * i);
        }, kPlaneMinBlocks);
        if (std::any_of(grown.possible.begin(), grown.possible.end(), [](std::uint8_t v) { return v != 0; })) {
          nodes.push_back(std::move(grown));
        }
      }
    }
    begin = end;
    end = nodes.size();
  }

  // Each sequence traced back from the source (`_PlaneImages.eigenrays`).
  const auto k_rays = static_cast<std::int64_t>(nodes.size());
  EigenraySet out;
  out.sound_speed = c;
  out.resize(n_points, k_rays);
  bool source_in_water = true;
  {
    const double src[3] = {source.x, source.y, source.z};
    for (const int k : unbounded) source_in_water = source_in_water && planes[static_cast<std::size_t>(k)].distance(src) >= -tol;
  }
  for_each_point(n_points, [&](std::int64_t i) {
    const double* point = points + 3 * i;
    std::vector<double> vertices;  // the source, each specular point, the field point: [n + 2, 3]
    for (std::int64_t j = 0; j < k_rays; ++j) {
      const PlaneNode& node = nodes[static_cast<std::size_t>(j)];
      const double* image = node.image.data() + 3 * i;
      const Vec3d d{image[0] - source.x, image[1] - source.y, image[2] - source.z};
      const double r = norm(d);
      const double u[3] = {d.x / r, d.y / r, d.z / r};
      bool valid = node.possible[i] != 0 && source_in_water;
      double a[3] = {source.x, source.y, source.z};
      double target[3] = {image[0], image[1], image[2]};
      double direction[3] = {u[0], u[1], u[2]};
      cd coef{1.0, 0.0};
      vertices.assign(a, a + 3);
      std::int64_t n_surface = 0, n_bottom = 0;
      for (const int idx : node.seq) {
        const PlaneOps& plane = planes[static_cast<std::size_t>(idx)];
        const double da = plane.distance(a), dt = plane.distance(target);
        const double denom = da - dt;
        const double t = denom > 0.0 ? da / denom : -1.0;
        valid = valid && denom > 0.0 && t >= -1e-9 && t <= 1.0 + 1e-9;
        double x[3];
        for (int k = 0; k < 3; ++k) x[k] = a[k] + t * (target[k] - a[k]);
        valid = valid && plane.on_face(x);
        const Vec3d& n = plane.plane.normal;
        const double cos_n = direction[0] * n.x + direction[1] * n.y + direction[2] * n.z;  // < 0: into the plane
        const double sin_g = clip01(std::abs(cos_n));
        cd reflection = plane.plane.coefficient;
        if (plane.plane.kind == PlaneKind::Bottom) {
          reflection = config.has_fluid_bottom ? rayleigh_reflection(std::asin(sin_g), c, config.fluid_bottom)
                                               : config.bottom_coefficient;
          ++n_bottom;
        } else if (plane.plane.kind == PlaneKind::Surface) {
          ++n_surface;
        }
        coef = coef * reflection;
        const double twice = 2.0 * cos_n;
        for (int k = 0; k < 3; ++k) direction[k] = direction[k] - twice * n[k];
        double mirrored[3];
        plane.mirror(target, mirrored);  // the image of P for the bounces still to come
        for (int k = 0; k < 3; ++k) {
          target[k] = mirrored[k];
          a[k] = x[k];
        }
        vertices.insert(vertices.end(), x, x + 3);
      }
      vertices.insert(vertices.end(), point, point + 3);
      const std::size_t n_vertices = vertices.size() / 3;
      for (std::size_t v = 1; v + 1 < n_vertices && valid; ++v) {
        for (const int k : unbounded) valid = valid && planes[static_cast<std::size_t>(k)].distance(&vertices[3 * v]) >= -tol;
      }
      for (const int k : bounded) {
        for (std::size_t v = 0; v + 1 < n_vertices && valid; ++v) {
          valid = !planes[static_cast<std::size_t>(k)].blocks(&vertices[3 * v], &vertices[3 * v + 3]);
        }
      }
      const std::int64_t e = i * k_rays + j;
      out.delay[e] = r / c;
      out.amplitude[e] = valid ? divide(coef, r) : cd{0.0, 0.0};
      for (int k = 0; k < 3; ++k) {
        out.launch[3 * e + k] = u[k];
        out.arrival[3 * e + k] = direction[k];
      }
      out.n_surface[e] = n_surface;
      out.n_bottom[e] = n_bottom;
      out.valid[e] = valid ? 1 : 0;
    }
  }, kPlaneMinBlocks);
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
  check_speed(config.sound_speed, "sediment: sound_speed");
  check_speed(config.sediment_speed, "sediment: sediment_speed");
  double sqrt_rho = 1.0;
  double k_loss = 0.0;
  if (config.fluid) {
    const FluidBottom& b = config.bottom;
    if (!(b.speed > 0.0) || !(b.density > 0.0) || !(b.water_density > 0.0) || !std::isfinite(b.speed) ||
        !std::isfinite(b.density) || !std::isfinite(b.water_density) || !std::isfinite(b.attenuation) ||
        b.attenuation < 0.0) {
      throw std::invalid_argument(
          "sediment: fluid penetration needs the sediment's speed (m/s), density (ratio to water_density) and "
          "attenuation >= 0 dB per wavelength; none is assumed");
    }
    if (std::abs(b.speed - config.sediment_speed) > 1e-9 * b.speed) {
      throw std::invalid_argument("sediment: the bottom's speed differs from sediment_speed");
    }
    if (!(config.frequency > 0.0) || !std::isfinite(config.frequency)) {
      throw std::invalid_argument("sediment: fluid penetration needs the frequency (Hz) of its attenuation");
    }
    sqrt_rho = std::sqrt(b.density / b.water_density);
    const double delta = b.attenuation / (40.0 * kPi * std::log10(std::exp(1.0)));
    k_loss = 2.0 * kPi * config.frequency / config.sound_speed * delta / (b.speed / config.sound_speed);
  }
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
    if (config.fluid) {
      cd factor{1.0, 0.0};
      const Vec3d direct = point - source;
      const double sin_d = direct.z / norm(direct);
      const double c_ratio = config.sound_speed / config.bottom.speed;
      if (below && config.evanescent && config.bottom.speed > config.sound_speed && source.z < config.interface_z &&
          sin_d < std::sqrt(1.0 - c_ratio * c_ratio)) {
        // Below the critical angle: the plane wave's zeroth-order field (15.1), as the NumPy provider.
        const double g_d = std::asin(std::min(std::max(sin_d, 0.0), 1.0));
        const double delta = config.bottom.attenuation / (40.0 * kPi * std::log10(std::exp(1.0)));
        const cd kappa = cd{1.0, delta} / (config.bottom.speed / config.sound_speed);
        const double cg = std::cos(g_d);
        cd root = std::sqrt(kappa * kappa - cg * cg + cd{0.0, 0.0});
        if (root.imag() < 0.0) root = -root;
        const double k_w = 2.0 * kPi * config.frequency / config.sound_speed;
        factor = (1.0 + rayleigh_reflection(g_d, config.sound_speed, config.bottom)) / sqrt_rho *
                 std::exp(-k_w * root.imag() * (point.z - config.interface_z));
      } else if (below) {
        const double grazing = std::asin(std::min(std::max(std::abs(launch.z), 0.0), 1.0));
        // V_wp / sqrt(a_rho) exp(-k_p'' L), in the order of the NumPy provider.
        factor = (1.0 + rayleigh_reflection(grazing, config.sound_speed, config.bottom)) / sqrt_rho *
                 std::exp(-k_loss * length);
      }
      out.amplitude[i] = divide(factor, config.sound_speed * out.delay[i]);
      return;
    }
    out.amplitude[i] = divide(below ? config.transmission : cd{1.0, 0.0}, config.sound_speed * out.delay[i]);
  });
  if (config.rough) add_rough_scattering(config, source, points, n_points, out);  // G-13, penetration.cpp
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

namespace {

/// The pairing loop. Without rates: `velocity` (may be null) takes the scatterer's velocity.
/// With `doppler`: q_velocity and p_velocity take it (zero when `velocities` is null) plus
/// each leg's range rate along its launch direction.
std::int64_t pair_paths(const EigenrayView& out, const EigenrayView& back, const Vec3d& tx_position,
                        const Vec3d& rx_reference, const std::complex<double>* amplitudes,
                        const double* velocities, const std::uint8_t* lambert, const double* normals,
                        bool doppler, const RangeRates& rates, double* q, double* p,
                        std::complex<double>* amplitude, double* velocity, double* p_velocity) {
  if (out.n_points != back.n_points) throw std::invalid_argument("two_way_paths: the two eigenray sets differ in size");
  check_speed(out.sound_speed, "two_way_paths: the outbound set's sound_speed");
  check_speed(back.sound_speed, "two_way_paths: the return set's sound_speed");
  const std::int64_t s = out.n_points, ko = out.n_rays, kb = back.n_rays;
  if (s > 0 && amplitudes == nullptr) throw std::invalid_argument("two_way_paths: amplitudes are required");
  if (!doppler && velocity != nullptr && velocities == nullptr) {
    throw std::invalid_argument("two_way_paths: no velocities to copy");
  }
  if (doppler && (velocity == nullptr || p_velocity == nullptr)) {
    throw std::invalid_argument("two_way_paths: room for both velocities is required");
  }
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
          if (doppler) {
            const double v = velocities != nullptr ? velocities[3 * i + x] : 0.0;
            velocity[3 * at + x] = rates.out != nullptr ? v + rates.out[eo] * out.launch[3 * eo + x] : v;
            p_velocity[3 * at + x] = rates.back != nullptr ? v + rates.back[eb] * back.launch[3 * eb + x] : v;
          } else if (velocity != nullptr) {
            velocity[3 * at + x] = velocities[3 * i + x];
          }
        }
        amplitude[at] = value;
        ++at;
      }
    }
  }
  return at;
}

}  // namespace

std::int64_t two_way_paths(const EigenrayView& out, const EigenrayView& back, const Vec3d& tx_position,
                           const Vec3d& rx_reference, const std::complex<double>* amplitudes,
                           const double* velocities, const std::uint8_t* lambert, const double* normals,
                           double* q, double* p, std::complex<double>* amplitude, double* velocity) {
  return pair_paths(out, back, tx_position, rx_reference, amplitudes, velocities, lambert, normals, false, RangeRates{},
                    q, p, amplitude, velocity, nullptr);
}

std::int64_t two_way_paths_doppler(const EigenrayView& out, const EigenrayView& back, const Vec3d& tx_position,
                                   const Vec3d& rx_reference, const std::complex<double>* amplitudes,
                                   const double* velocities, const std::uint8_t* lambert, const double* normals,
                                   const RangeRates& rates, double* q, double* p, std::complex<double>* amplitude,
                                   double* q_velocity, double* p_velocity) {
  return pair_paths(out, back, tx_position, rx_reference, amplitudes, velocities, lambert, normals, true, rates, q, p,
                    amplitude, q_velocity, p_velocity);
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
