// The roughness-scattered field in the sediment (G-13, `penetration: rough`): see
// sonar/sim/penetration.hpp. Every step follows python/gpusonar/sim/penetration.py in the same
// order; the comments name the NumPy line each mirrors where the order is not obvious. The two
// engines differ only by the rounding of NumPy's vectorised exp, log, pow, hypot and complex
// product (it uses fused multiply-adds and its own polynomials; C++ here uses libm and plain
// products): measured in tests/python/test_sim_cpp.py.
#include "sonar/sim/penetration.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/sim/paths.hpp"

namespace sonar::sim {

namespace {

using cd = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;

/// Complex quotient by Smith's method with a reciprocal: NumPy's order (paths.cpp `divide`).
cd np_div(cd a, cd b) {
  if (std::abs(b.real()) >= std::abs(b.imag())) {
    const double rat = b.imag() / b.real();
    const double scl = 1.0 / (b.real() + b.imag() * rat);
    return {(a.real() + a.imag() * rat) * scl, (a.imag() - a.real() * rat) * scl};
  }
  const double rat = b.real() / b.imag();
  const double scl = 1.0 / (b.imag() + b.real() * rat);
  return {(a.real() * rat + a.imag()) * scl, (a.imag() * rat - a.real()) * scl};
}

/// Complex quotient as CPython forms it for Python complex scalars (Smith's method, dividing).
cd py_div(cd a, cd b) {
  if (std::abs(b.real()) >= std::abs(b.imag())) {
    const double ratio = b.imag() / b.real();
    const double denom = b.real() + b.imag() * ratio;
    return {(a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom};
  }
  const double ratio = b.real() / b.imag();
  const double denom = b.real() * ratio + b.imag();
  return {(a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom};
}

/// a * b written out, so that no library routine or contraction changes the products.
cd mul(cd a, cd b) {
  return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

cd branch(cd b) { return b.imag() < 0.0 ? -b : b; }  // Im >= 0 (penetration._branch)

/// |z|^2 as NumPy's np.abs(z) ** 2.
double abs2(cd z) {
  const double h = std::hypot(z.real(), z.imag());
  return h * h;
}

std::uint64_t splitmix(std::uint64_t x) {
  x = x + 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

/// The sediment in the book's notation, with the Python scalars penetration.py forms once.
struct Medium {
  double k_w = 0.0, a_rho = 0.0, nu = 0.0, delta = 0.0;
  cd a_p;      ///< nu / (1 + i delta) (8.25), a Python complex quotient
  cd rho_a_p;  ///< a_rho * a_p, a Python product
  cd c2;       ///< 1 / (a_p^2 a_rho), Python scalars (a_p ** 2 by binary powering)
  double kw2 = 0.0;
  double c1 = 0.0;  ///< 1 - 1 / a_rho

  Medium(double k_w_, double a_rho_, double nu_, double delta_) : k_w(k_w_), a_rho(a_rho_), nu(nu_), delta(delta_) {
    a_p = py_div(cd{nu, 0.0}, cd{1.0, delta});
    rho_a_p = mul(cd{a_rho, 0.0}, a_p);
    const cd sq = mul(cd{1.0, 0.0}, mul(a_p, a_p));
    c2 = py_div(cd{1.0, 0.0}, mul(sq, cd{a_rho, 0.0}));
    kw2 = std::pow(k_w, 2.0);
    c1 = 1.0 - 1.0 / a_rho;
  }
};

/// beta_w(K), beta_p(K) (penetration._betas).
void betas(double K, const Medium& m, cd& bw, cd& bp) {
  const double x = K / m.k_w;
  bw = branch(std::sqrt(cd{1.0 - x * x, 0.0}));
  const cd z = np_div(mul(m.a_p, cd{K, 0.0}), cd{m.k_w, 0.0});
  const cd z2 = mul(z, z);
  bp = branch(std::sqrt(cd{1.0 - z2.real(), (0.0 - z2.imag()) + 0.0}));
}

/// V_ww(K) (penetration.flat_reflection).
cd flat_reflection(double K, const Medium& m) {
  cd bw, bp;
  betas(K, m, bw, bp);
  const cd z = np_div(mul(m.rho_a_p, bw), bp);
  return np_div(cd{z.real() - 1.0, z.imag()}, cd{z.real() + 1.0, z.imag()});
}

/// A_wp (15.4) (penetration.a_wp, two-dimensional wave vectors).
cd a_wp(double ksx, double ksy, double kix, double kiy, const Medium& m) {
  const double dot = ksx * kix + ksy * kiy;
  const double ks = std::sqrt(ksx * ksx + ksy * ksy);
  const double ki = std::sqrt(kix * kix + kiy * kiy);
  cd bw_s, bp_s, bw_i, bp_i;
  betas(ks, m, bw_s, bp_s);
  betas(ki, m, bw_i, bp_i);
  const cd v_i = flat_reflection(ki, m), v_s = flat_reflection(ks, m);
  const cd q = np_div(mul(bp_i, bw_s), m.a_p);
  const cd inner{dot / m.kw2 + q.real(), q.imag()};
  const cd g = cd{m.c1 * inner.real() - 1.0 + m.c2.real(), m.c1 * inner.imag() + m.c2.imag()};
  return mul(mul(cd{0.5 * (1.0 + v_i.real()), 0.5 * v_i.imag()}, cd{1.0 + v_s.real(), v_s.imag()}), g);
}

double sigma_wp(double ksx, double ksy, double kix, double kiy, const Medium& m, double w2, double gamma2, double K0) {
  const double qx = ksx - kix, qy = ksy - kiy;
  const double Q = std::sqrt(qx * qx + qy * qy);
  const double ks = std::sqrt(ksx * ksx + ksy * ksy);
  const double amp2 = abs2(a_wp(ksx, ksy, kix, kiy, m));
  const double W = w2 / std::pow(Q * Q + std::pow(K0, 2.0), gamma2 / 2.0);       // (D.7)
  const double base = std::pow(m.k_w, 4.0) / std::pow(m.nu, 2.0) * amp2 * W;
  cd bw, bp;
  betas(ks, m, bw, bp);
  return base * abs2(bp) / abs2(bw);                                               // the obliquity factor (ours)
}

void check_roughness(const SedimentConfig& config) {
  const SedimentRoughness& r = config.roughness;
  if (!config.fluid || !config.evanescent) {
    throw std::invalid_argument(
        "sediment: penetration: rough adds the roughness-scattered field to the evanescent zeroth order of fluid "
        "penetration: it needs `fluid` and `evanescent`");
  }
  if (!(r.w2 >= 0.0) || !std::isfinite(r.w2) || !(r.gamma2 > 2.0 && r.gamma2 < 4.0) || !(r.K0 >= 0.0) ||
      !std::isfinite(r.K0)) {
    throw std::invalid_argument(
        "sediment: rough needs the relief spectrum stated: w2 >= 0 (m^(4 - gamma2)), 2 < gamma2 < 4, K0 >= 0 (1/m); "
        "there is no default");
  }
  for (const double v : {r.patch, r.refine, r.floor_db, r.bin}) {
    if (!(v > 0.0) || !std::isfinite(v)) {
      throw std::invalid_argument("sediment: rough needs patch (m), refine, floor_db (dB) and bin (s) stated and > 0");
    }
  }
  if (r.max_pairs < 1) throw std::invalid_argument("sediment: roughness.max_pairs must be >= 1 (ray pairs an echo)");
}

}  // namespace

std::complex<double> rough_patch_coefficient(std::uint64_t seed, std::int64_t level, std::int64_t i,
                                             std::int64_t j) {
  std::uint64_t h = splitmix(seed);
  for (const std::int64_t part : {level, i, j}) h = splitmix(h ^ static_cast<std::uint64_t>(part));
  const std::uint64_t h2 = splitmix(h);
  constexpr double kTwo53 = 9007199254740992.0;
  const double u1 = (static_cast<double>(h >> 11U) + 0.5) / kTwo53;
  const double u2 = static_cast<double>(h2 >> 11U) / kTwo53;
  // np.sqrt(-np.log(u1)) * np.exp(2j * np.pi * u2): the exponent is (0, 2 pi u2).
  const double r = std::sqrt(-std::log(u1));
  const cd e = std::exp(cd{0.0, (2.0 * kPi) * u2});
  return {r * e.real(), r * e.imag()};
}

QuadtreeCells rough_quadtree_cells(double px, double py, double radius, double finest, double refine) {
  QuadtreeCells out;
  const int top = std::max(0, static_cast<int>(std::ceil(std::log2(std::max(radius, finest) / finest))));
  double s = finest * std::ldexp(1.0, top);
  const auto i0 = static_cast<std::int64_t>(std::floor((px - radius) / s));
  const auto i1 = static_cast<std::int64_t>(std::floor((px + radius) / s));
  const auto j0 = static_cast<std::int64_t>(std::floor((py - radius) / s));
  const auto j1 = static_cast<std::int64_t>(std::floor((py + radius) / s));
  std::vector<std::int64_t> I, J;
  for (std::int64_t a = i0; a <= i1; ++a) {
    for (std::int64_t b = j0; b <= j1; ++b) {
      I.push_back(a);
      J.push_back(b);
    }
  }
  int level = top;
  while (!I.empty()) {
    s = finest * std::ldexp(1.0, level);
    std::vector<std::int64_t> si, sj;  // the cells split at this level, in order
    for (std::size_t k = 0; k < I.size(); ++k) {
      const double cx = (static_cast<double>(I[k]) + 0.5) * s;
      const double cy = (static_cast<double>(J[k]) + 0.5) * s;
      const double near = std::hypot(std::max(std::abs(px - cx) - s / 2, 0.0), std::max(std::abs(py - cy) - s / 2, 0.0));
      if (!(near <= radius)) continue;
      const bool split = level > 0 && std::hypot(px - cx, py - cy) < refine * s;
      if (split) {
        si.push_back(I[k]);
        sj.push_back(J[k]);
        continue;
      }
      out.level.push_back(level);
      out.i.push_back(I[k]);
      out.j.push_back(J[k]);
      out.cx.push_back(cx);
      out.cy.push_back(cy);
      out.side.push_back(s);
    }
    if (level == 0 || si.empty()) break;
    // np.concatenate over the children (a, b) = (0, 0), (0, 1), (1, 0), (1, 1), each over every split cell
    I.clear();
    J.clear();
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        for (std::size_t k = 0; k < si.size(); ++k) {
          I.push_back(2 * si[k] + a);
          J.push_back(2 * sj[k] + b);
        }
      }
    }
    --level;
  }
  return out;
}

double rough_sediment_radius(double depth, double k_loss, double floor_db) {
  double lo = depth;
  double hi = depth * std::pow(10.0, floor_db / 20.0) + 1e-9;
  for (int it = 0; it < 80; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (20.0 * std::log10(mid / depth) + 20.0 / std::log(10.0) * k_loss * (mid - depth) < floor_db) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return std::sqrt(std::max(std::pow(hi, 2.0) - std::pow(depth, 2.0), 0.0));
}

double rough_sigma_wp(double ks_x, double ks_y, double ki_x, double ki_y, double k_w, double a_rho, double nu,
                      double delta, double w2, double gamma2, double K0) {
  return sigma_wp(ks_x, ks_y, ki_x, ki_y, Medium(k_w, a_rho, nu, delta), w2, gamma2, K0);
}

std::string rough_cost_message(std::int64_t rays, std::int64_t point, std::int64_t max_pairs) {
  return "sediment: penetration: rough: sub-critical point " + std::to_string(point) + " has " + std::to_string(rays) +
         " rays a leg, about " + std::to_string(rays * rays) + " ray pairs an echo, more than roughness.max_pairs " +
         std::to_string(max_pairs) +
         ". Nothing is truncated; either raise roughness.max_pairs (the synthesis then takes that many pairs an echo), or "
         "give a coarser roughness.bin (fewer delay bins: fewer rays a leg)";
}

RoughBins rough_point_rays(const SedimentConfig& config, const Vec3d& source, const Vec3d& point) {
  check_roughness(config);
  const SedimentRoughness& r = config.roughness;
  const FluidBottom& b = config.bottom;
  const double c_w = config.sound_speed, c_s = b.speed, iz = config.interface_z;
  const double nu = b.speed / config.sound_speed;
  const double delta = b.attenuation / (40.0 * kPi * std::log10(std::exp(1.0)));
  const double k_w = 2.0 * kPi * config.frequency / c_w;
  const double k_loss = k_w * delta / nu;
  const double a_rho = b.density / b.water_density;
  const Medium m(k_w, a_rho, nu, delta);

  const double depth = point.z - iz;
  const double radius = rough_sediment_radius(depth, k_loss, r.floor_db);
  const QuadtreeCells cells = rough_quadtree_cells(point.x, point.y, radius, r.patch, r.refine);
  const std::size_t n = cells.cx.size();
  std::vector<cd> amp(n);
  std::vector<double> t(n), lau(3 * n), arr(3 * n);
  const double kn = k_w / nu;
  const double dz2 = std::pow(depth, 2.0);
  const double dzi2 = std::pow(iz - source.z, 2.0);
  const double sqrt_rho = std::sqrt(a_rho);
  for (std::size_t k = 0; k < n; ++k) {
    const double cx = cells.cx[k], cy = cells.cy[k];
    // _patch_geometry: the sediment leg from the cell to the point, the water leg from the source
    const double dxs = point.x - cx, dys = point.y - cy;
    const double L = std::sqrt(dxs * dxs + dys * dys + dz2);
    const double ksx = kn * dxs / L, ksy = kn * dys / L;
    const double dx = cx - source.x, dy = cy - source.y;
    const double r_i = std::sqrt(dx * dx + dy * dy + dzi2);
    const double kix = k_w * dx / r_i, kiy = k_w * dy / r_i;
    const double p_i = 1.0 / r_i;
    const double sig = sigma_wp(ksx, ksy, kix, kiy, m, r.w2, r.gamma2, r.K0);
    const double side = cells.side[k];
    const cd c = rough_patch_coefficient(r.seed, cells.level[k], cells.i[k], cells.j[k]);
    // c sqrt(sigma A) exp(-k_loss L) p_i / L / sqrt(a_rho), left to right
    const double sq = std::sqrt(sig * (side * side));
    const double e = std::exp(-k_loss * L);
    cd a{c.real() * sq, c.imag() * sq};
    a = cd{a.real() * e, a.imag() * e};
    a = cd{a.real() * p_i, a.imag() * p_i};
    a = np_div(a, cd{L, 0.0});
    amp[k] = np_div(a, cd{sqrt_rho, 0.0});
    t[k] = r_i / c_w + L / c_s;
    const double cell[3] = {cx, cy, iz};
    const double src[3] = {source.x, source.y, source.z};
    const double pt[3] = {point.x, point.y, point.z};
    for (int q = 0; q < 3; ++q) {
      lau[3 * k + q] = (cell[q] - src[q]) / r_i;
      arr[3 * k + q] = (pt[q] - cell[q]) / L;
    }
  }
  // One eigenray per delay bin on the global grid floor(t / bin), bins in increasing delay (np.unique).
  std::vector<std::int64_t> key(n);
  for (std::size_t k = 0; k < n; ++k) key[k] = static_cast<std::int64_t>(std::floor(t[k] / r.bin));
  std::vector<std::int64_t> bins = key;
  std::sort(bins.begin(), bins.end());
  bins.erase(std::unique(bins.begin(), bins.end()), bins.end());
  const std::size_t nb = bins.size();
  std::vector<std::size_t> inv(n);
  for (std::size_t k = 0; k < n; ++k) {
    inv[k] = static_cast<std::size_t>(std::lower_bound(bins.begin(), bins.end(), key[k]) - bins.begin());
  }
  std::vector<double> w(n), wsum(nb, 0.0), wt_sum(nb, 0.0), t_sum(nb, 0.0), count(nb, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    w[k] = abs2(amp[k]);
    wsum[inv[k]] += w[k];
    wt_sum[inv[k]] += w[k] * t[k];
    t_sum[inv[k]] += t[k];
    count[inv[k]] += 1.0;
  }
  RoughBins out;
  out.delay.resize(nb);
  for (std::size_t q = 0; q < nb; ++q) {
    // a bin of zero amplitude (w2 = 0) keeps the plain mean delay
    out.delay[q] = wsum[q] > 0.0 ? wt_sum[q] / wsum[q] : t_sum[q] / count[q];
  }
  const double omega = 2.0 * kPi * config.frequency;
  std::vector<double> re(nb, 0.0), im(nb, 0.0), ml(3 * nb, 0.0), ma(3 * nb, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t q = inv[k];
    const cd ph = mul(amp[k], std::exp(cd{0.0, -omega * (t[k] - out.delay[q])}));
    re[q] += ph.real();
    im[q] += ph.imag();
    const double wt = wsum[q] > 0.0 ? w[k] : 1.0;  // a bin of zero amplitude: plain means
    for (int c = 0; c < 3; ++c) {
      ml[3 * q + c] += wt * lau[3 * k + c];
      ma[3 * q + c] += wt * arr[3 * k + c];
    }
  }
  out.amplitude.resize(nb);
  out.launch.resize(3 * nb);
  out.arrival.resize(3 * nb);
  for (std::size_t q = 0; q < nb; ++q) {
    out.amplitude[q] = cd{re[q], im[q]};
    for (std::vector<double>* v : {&ml, &ma}) {
      const double* x = v->data() + 3 * q;
      const double norm = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      std::vector<double>& dst = v == &ml ? out.launch : out.arrival;
      for (int c = 0; c < 3; ++c) dst[3 * q + c] = x[c] / norm;
    }
  }
  return out;
}

void add_rough_scattering(const SedimentConfig& config, const Vec3d& source, const double* points,
                          std::int64_t n_points, EigenraySet& set) {
  check_roughness(config);
  if (set.n_points != n_points || set.n_rays != 1) {
    throw std::invalid_argument("sediment: rough widens the one-ray set of sediment_eigenrays");
  }
  // The points of the evanescent branch of sediment_eigenrays (SedimentProvider._with_scattered).
  std::vector<std::int64_t> sub;
  if (config.sediment_speed > config.sound_speed && source.z < config.interface_z) {
    const double c_ratio = config.sound_speed / config.sediment_speed;
    const double sin_c = std::sqrt(1.0 - c_ratio * c_ratio);
    for (std::int64_t i = 0; i < n_points; ++i) {
      const Vec3d point{points[3 * i], points[3 * i + 1], points[3 * i + 2]};
      const Vec3d direct = point - source;
      if (point.z > config.interface_z && direct.z / norm(direct) < sin_c) sub.push_back(i);
    }
  }
  std::vector<RoughBins> rays(sub.size());
  parallel_for_host(static_cast<std::int64_t>(sub.size()), [&](std::int64_t k) {
    const std::int64_t i = sub[static_cast<std::size_t>(k)];
    rays[static_cast<std::size_t>(k)] =
        rough_point_rays(config, source, Vec3d{points[3 * i], points[3 * i + 1], points[3 * i + 2]});
  });
  // The cost cap (penetration.rough_cost_check): the first point with the most rays a leg; raise, never truncate.
  std::size_t extra = 0, worst = 0;
  for (std::size_t s = 0; s < rays.size(); ++s) {
    if (rays[s].delay.size() > extra) {
      extra = rays[s].delay.size();
      worst = s;
    }
  }
  if (!rays.empty()) {
    const auto n = static_cast<std::int64_t>(1 + extra);
    if (n * n > config.roughness.max_pairs) throw std::invalid_argument(rough_cost_message(n, sub[worst], config.roughness.max_pairs));
  }
  if (extra == 0) return;
  const auto k = static_cast<std::int64_t>(1 + extra);
  EigenraySet out;
  out.sound_speed = set.sound_speed;
  out.resize(n_points, k);
  std::fill(out.valid.begin(), out.valid.end(), std::uint8_t{0});
  for (std::int64_t i = 0; i < n_points; ++i) {
    const std::int64_t row = i * k;
    out.delay[row] = set.delay[i];
    out.amplitude[row] = set.amplitude[i];
    out.valid[row] = 1;
    for (std::int64_t c = 0; c < k; ++c) {
      for (int a = 0; a < 3; ++a) {
        out.launch[3 * (row + c) + a] = set.launch[3 * i + a];
        out.arrival[3 * (row + c) + a] = set.arrival[3 * i + a];
      }
    }
  }
  for (std::size_t s = 0; s < sub.size(); ++s) {
    const std::int64_t row = sub[s] * k;
    const RoughBins& r = rays[s];
    for (std::size_t q = 0; q < r.delay.size(); ++q) {
      const auto c = static_cast<std::int64_t>(1 + q);
      out.delay[row + c] = r.delay[q];
      out.amplitude[row + c] = r.amplitude[q];
      out.valid[row + c] = 1;
      for (int a = 0; a < 3; ++a) {
        out.launch[3 * (row + c) + a] = r.launch[3 * q + a];
        out.arrival[3 * (row + c) + a] = r.arrival[3 * q + a];
      }
    }
  }
  set = std::move(out);
}

}  // namespace sonar::sim
