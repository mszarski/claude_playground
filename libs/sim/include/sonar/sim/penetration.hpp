// Penetration of a rough fluid seafloor (G-13): one seeded realisation of the first-order
// roughness-scattered field in the sediment as extra eigenrays of the `sediment` tier. The C++
// counterpart of python/gpusonar/sim/penetration.py (`RoughPatches`, `quadtree_cells`,
// `patch_coefficients`, `sigma_wp`), with the same arithmetic in the same order; that module's
// docstring says what is still wrong (first order, single scattering, a frozen realisation that is
// too coherent across a long aperture, a band-centre cross section, an isotropic spectrum, (15.11)
// read with the obliquity factor of (15.2)-(15.3): ours). Host only, like the rest of paths.hpp:
// the eigenrays feed the path pairing, and only the echo synthesis runs on a device.
//
// Formulas: Jackson and Richardson (2007) Ch. 15, (15.2)-(15.4), (15.11)-(15.16), as retyped in
// docs/references/jackson_richardson_2007/03_wave_theories_reflection_penetration.md Sec. 2.7, the
// spectrum (D.7) (02_roughness_heterogeneity_statistics.md Sec. 2.1); the patch sum, the quadtree,
// the hash and the delay bins are ours (docs/PHYSICS_NOTES.md section 31).
//
// The random interface is engine independent by construction: a cell (level, i, j) of the global
// quadtree carries the complex Gaussian deviate sqrt(-ln u1) exp(2 pi i u2), u1 and u2 the top 53
// bits of SplitMix64 chained over (seed, level, i, j) (`rough_patch_coefficient`). No random-number
// library enters, so both engines draw the same interface for a seed, bit for bit.
#pragma once

#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "sonar/core/vec.hpp"

namespace sonar::sim {

struct EigenraySet;
struct SedimentConfig;

/// The cost cap's default (`SedimentRoughness::max_pairs`): python/gpusonar/sim/penetration.py
/// DEFAULT_MAX_PAIRS, the same number (ours; that module says how it was chosen).
inline constexpr std::int64_t kRoughMaxPairs = 1000000;

/// The relief and the numerical knobs of `penetration: rough`. Nothing but `max_pairs` has a
/// default here: the Python provider states every field (its documented defaults are ours: patch
/// half a water wavelength, refine 3, floor_db 30 dB, bin a quarter period at the band centre,
/// max_pairs kRoughMaxPairs).
struct SedimentRoughness {
  double w2 = std::numeric_limits<double>::quiet_NaN();      ///< spectral strength, m^(4 - gamma2)
  double gamma2 = std::numeric_limits<double>::quiet_NaN();  ///< spectral exponent, 2 < gamma2 < 4
  double K0 = std::numeric_limits<double>::quiet_NaN();      ///< von Karman cutoff (D.7), 1/m; 0 the power law
  std::uint64_t seed = 0;                                    ///< the interface realisation
  double patch = std::numeric_limits<double>::quiet_NaN();     ///< finest cell side, m
  double refine = std::numeric_limits<double>::quiet_NaN();    ///< split while nearer than this many sides
  double floor_db = std::numeric_limits<double>::quiet_NaN();  ///< reach of the cells, dB of the sediment leg
  double bin = std::numeric_limits<double>::quiet_NaN();       ///< delay bin, s
  /// The cost cap (D-213; the owner, D-229: a cap that raises, never truncates): the most ray pairs
  /// one echo of a sub-critical point may need, estimated as the square of its leg's rays (the
  /// fluid ray and the delay bins). More raises std::invalid_argument (`rough_cost_message`)
  /// before any pairing; nothing is ever dropped. >= 1.
  std::int64_t max_pairs = kRoughMaxPairs;
};

/// The refusal of the cost cap, word for word python/gpusonar/sim/penetration.py
/// `rough_cost_message`: point `point` (its index in the points given) has `rays` rays a leg.
std::string rough_cost_message(std::int64_t rays, std::int64_t point, std::int64_t max_pairs);

/// The complex Gaussian deviate (<|c|^2> = 1) of interface cell (level, i, j) for `seed`:
/// h = SplitMix64(seed), then h = SplitMix64(h ^ part) for part = level, i, j (two's complement),
/// h2 = SplitMix64(h); u1 = ((h >> 11) + 0.5) / 2^53, u2 = (h2 >> 11) / 2^53;
/// c = sqrt(-ln u1) exp(2 pi i u2) (Box-Muller in polar form: |c|^2 is exponential with mean 1).
std::complex<double> rough_patch_coefficient(std::uint64_t seed, std::int64_t level, std::int64_t i, std::int64_t j);

/// Interface cells about (px, py): squares of a global quadtree whose finest side is `finest`,
/// split while the cell's centre is nearer than `refine` sides to the point, kept while any part
/// of the cell is within `radius`. Cell (level, i, j) covers [i s, (i + 1) s) x [j s, (j + 1) s),
/// s = finest 2^level. The order is penetration.quadtree_cells's.
struct QuadtreeCells {
  std::vector<std::int64_t> level, i, j;
  std::vector<double> cx, cy, side;
};
QuadtreeCells rough_quadtree_cells(double px, double py, double radius, double finest, double refine);

/// Horizontal reach about a point `depth` under the interface: where the sediment leg's spreading
/// and attenuation, 20 log10(L / depth) + (20 / ln 10) k_loss (L - depth), reach `floor_db` (ours).
double rough_sediment_radius(double depth, double k_loss, double floor_db);

/// The cross section per unit area and solid angle for scattering from the water into the sediment,
/// k_w^2 k_p'^2 |beta_p(K_s) / beta_w(K_s)|^2 |A_wp|^2 W(K_s - K_i) (ours, by stationary phase;
/// penetration.sigma_wp with obliquity=True), for horizontal wave vectors K_s, K_i (rad/m) and the
/// sediment (a_rho, nu, delta) of the book's notation.
double rough_sigma_wp(double ks_x, double ks_y, double ki_x, double ki_y, double k_w, double a_rho, double nu,
                      double delta, double w2, double gamma2, double K0);

/// The delay bins of one point: (delay [B], amplitude [B], launch [B, 3], arrival [B, 3]), bins in
/// increasing delay. `config` must be a fluid, evanescent sediment with `rough` (as
/// `sediment_eigenrays` checks).
struct RoughBins {
  std::vector<double> delay;
  std::vector<std::complex<double>> amplitude;
  std::vector<double> launch;
  std::vector<double> arrival;
};
RoughBins rough_point_rays(const SedimentConfig& config, const Vec3d& source, const Vec3d& point);

/// Widens `set` (the [S, 1] fluid rays of `sediment_eigenrays`) by the scattered bins of every
/// point under the interface whose straight line from `source` meets it below the critical angle:
/// column 0 stays the fluid ray, columns 1.. are the point's bins, the rest are invalid (zero delay
/// and amplitude, the fluid ray's directions), as SedimentProvider._with_scattered builds them.
/// Before widening, the cost cap: a point whose leg has n rays with n^2 > roughness.max_pairs
/// (the first with the most) throws std::invalid_argument(rough_cost_message(...)).
void add_rough_scattering(const SedimentConfig& config, const Vec3d& source, const double* points,
                          std::int64_t n_points, EigenraySet& set);

}  // namespace sonar::sim
