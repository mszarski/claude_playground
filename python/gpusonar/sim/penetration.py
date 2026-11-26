"""Penetration of a rough fluid seafloor: the first-order roughness-scattered field in the sediment (G-13).

Jackson and Richardson (2007), Ch. 15, as retyped in
docs/references/jackson_richardson_2007/03_wave_theories_reflection_penetration.md (cited [N3]) Sec. 2.7, with
the roughness spectrum of Appendix D ([N2] = 02_roughness_heterogeneity_statistics.md, Sec. 2.1). What is still
wrong or uncertain, first:

- **Single scattering, first order in the relief.** Second order improves agreement "noticeably" at 2.5 cm rms
  ripples (Lopes et al. 2005, [N3] Sec. 2.7); the flat-interface (zeroth-order) field is not reduced by the energy the
  roughness scatters (that is second order in the relief). Flat MEAN interface, fluid sediment, no layering, no
  volume scattering of the transmitted field.
- **(15.11) as retyped in [N3] lacks an obliquity factor that its own (15.3) implies.** Our cross section is the
  far field of (15.2)-(15.3) by stationary phase (ours):
  sigma_wp = k_w^2 k_p^2 |beta_p(K_s) / beta_w(K_s)|^2 |A_wp|^2 W(K_s - K_i), which is the retyped (15.11)
  (k_w^4 / nu_p^2) |A_wp|^2 W times |beta_p(K_s) / beta_w(K_s)|^2 (the two agree on the vertical and differ toward
  the interface). Fig. 15.8's reading (133-141 dB at 5-40 cm, [N3] Sec. 2.7) is reproduced with ours (141.6 to
  133.5 dB) and not with the retyped form (146.7 to 136.0 dB): `test_sediment_roughness.py`. The book's page was
  not reread to settle it.
- **(15.13) as retyped gives K_s = k_w (R_s - R) / |r_s - r|**; the plane wave that reaches a point in the sediment
  from a patch has the sediment's wavenumber, so we take K_s = k_p' (R_s - R) / |r_s - r|, k_p' = k_w / nu_p (ours;
  with k_w a sediment ray steeper than the critical angle would be evanescent).
- **(15.9) as retyped lacks the factor i of (15.3)**; `ripple_field` keeps (15.3)'s. It moves P_+- against P_0 by a
  quarter period along the ripple and changes no level range.
- **One frequency**: the cross section, the attenuation and the patch phases are taken at the band centre.
- **Isotropic spectrum** (power law or von Karman (D.7)); no ripple-specific (anisotropic) spectrum beyond the
  single sinusoid of `ripple_field`.
- **A realisation, not a mean** (`RoughPatches`): see its docstring for what that does to imaging.

Conventions ([N3] Sec. 2.1): exp(-i w t); grazing angles; horizontal wave vectors K in rad/m; beta_q(K) =
sqrt(1 - K^2 / k_q^2) with Im >= 0; a_p = nu / (1 + i delta) (8.25); a_rho the density ratio; the roughness
spectrum W(K) = w2 / (K^2 + K0^2)^(gamma2 / 2) (D.7) with w2 in m^(4 - gamma2) (Table 13.1's units are inferred,
[N4] Sec. 6; `configs/sediments.yaml` uses the same), K0 = 0 the pure power law (6.8). The perturbation amplitude
A_wp (15.4) equals the sediment-side amplitude of `bottom.py`'s first-order solution (B = i k_w A_wp / beta_w(K_s),
to 1e-15: `test_our_perturbation_sediment_amplitude_is_a_wp`).

Engines: `RoughPatches`, `quadtree_cells`, `patch_coefficients`, `_sediment_radius` and `sigma_wp` (obliquity=True)
have a C++ counterpart in libs/sim/src/penetration.cpp (sonar/sim/penetration.hpp) with the same operations in the
same order; `SedimentProvider` uses it with the C++ engine. The interface is the same on both for a seed: the cell
deviates come from an explicit SplitMix64 chain over (seed, level, i, j) and a polar Box-Muller step, no random
generator. The two differ only by the rounding of NumPy's vectorised exp, log, pow, hypot and complex product
(tests/python/test_sim_cpp.py measures it). Not ported (no caller in the simulator, tests only): `sigma_wp` with
one-dimensional wave numbers or obliquity=False, `ripple_*`, `mean_square_field`.
"""

from __future__ import annotations

import numpy as np

from . import bottom as bottom_model

ROUGHNESS_KEYS = ("sediment", "sediment_file", "w2", "gamma2", "K0", "seed", "patch", "refine", "floor_db", "bin")


def _branch(b):
    return np.where(b.imag < 0.0, -b, b)


def _betas(K, k_w, a_p):
    """beta_w(K), beta_p(K) (dimensionless vertical sines, Im >= 0) for horizontal wavenumbers |K| (rad/m)."""
    K = np.asarray(K, dtype=float)
    return (_branch(np.sqrt(1.0 - (K / k_w) ** 2 + 0j)), _branch(np.sqrt(1.0 - (a_p * K / k_w) ** 2 + 0j)))


def _a_p(nu: float, delta: float) -> complex:
    return nu / (1.0 + 1j * delta)                                      # (8.25)


def flat_reflection(K, k_w: float, a_rho: float, nu: float, delta: float):
    """V_ww(K) = (z_wp - 1) / (z_wp + 1), z_wp = a_rho a_p beta_w / beta_p ((8.49), (8.64))."""
    a_p = _a_p(nu, delta)
    bw, bp = _betas(K, k_w, a_p)
    z = a_rho * a_p * bw / bp
    return (z - 1.0) / (z + 1.0)


def a_wp(K_s, K_i, k_w: float, a_rho: float, nu: float, delta: float):
    """The first-order perturbation amplitude for scattering from the water into the sediment, (15.4):

        A_wp = (1/2) [1 + V(K_i)] [1 + V(K_s)] { (1 - 1/a_rho) [K_s.K_i / k_w^2 + beta_p(K_i) beta_w(K_s) / a_p]
               - 1 + 1 / (a_p^2 a_rho) }

    K_s, K_i: horizontal wave vectors, [..., 2] (or [...] scalars for propagation in one vertical plane), rad/m."""
    K_s, K_i = np.asarray(K_s, dtype=float), np.asarray(K_i, dtype=float)
    if K_s.ndim and K_s.shape[-1] == 2 and K_i.shape[-1:] == (2,):
        dot = (K_s * K_i).sum(axis=-1)
        ks, ki = np.linalg.norm(K_s, axis=-1), np.linalg.norm(K_i, axis=-1)
    else:
        dot, ks, ki = K_s * K_i, np.abs(K_s), np.abs(K_i)
    a_p = _a_p(nu, delta)
    bw_s = _betas(ks, k_w, a_p)[0]
    bp_i = _betas(ki, k_w, a_p)[1]
    v_i, v_s = flat_reflection(ki, k_w, a_rho, nu, delta), flat_reflection(ks, k_w, a_rho, nu, delta)
    g = (1.0 - 1.0 / a_rho) * (dot / k_w**2 + bp_i * bw_s / a_p) - 1.0 + 1.0 / (a_p**2 * a_rho)
    return 0.5 * (1.0 + v_i) * (1.0 + v_s) * g


def roughness_spectrum(Q, w2: float, gamma2: float, K0: float = 0.0):
    """W(Q) = w2 / (Q^2 + K0^2)^(gamma2 / 2): von Karman (D.7) ([N2] Sec. 2.1); K0 = 0 the power law (6.8)."""
    Q = np.asarray(Q, dtype=float)
    with np.errstate(divide="ignore"):
        return w2 / (Q**2 + K0**2) ** (gamma2 / 2.0)


def sigma_wp(K_s, K_i, k_w: float, a_rho: float, nu: float, delta: float, w2: float, gamma2: float, K0: float = 0.0,
             obliquity: bool = True):
    """Cross section per unit area and solid angle for scattering from the water into the sediment.

    obliquity=True (used everywhere): the far field of (15.2)-(15.3) by stationary phase (ours),
    k_w^2 k_p'^2 |beta_p(K_s) / beta_w(K_s)|^2 |A_wp|^2 W(K_s - K_i), k_p' = k_w / nu.
    obliquity=False: (15.11) as retyped in [N3] Sec. 2.7, (k_w^4 / nu^2) |A_wp|^2 W (for comparison only)."""
    K_s, K_i = np.asarray(K_s, dtype=float), np.asarray(K_i, dtype=float)
    two_d = K_s.ndim and K_s.shape[-1] == 2
    Q = np.linalg.norm(K_s - K_i, axis=-1) if two_d else np.abs(K_s - K_i)
    ks = np.linalg.norm(K_s, axis=-1) if two_d else np.abs(K_s)
    amp2 = np.abs(a_wp(K_s, K_i, k_w, a_rho, nu, delta)) ** 2
    base = k_w**4 / nu**2 * amp2 * roughness_spectrum(Q, w2, gamma2, K0)
    if not obliquity:
        return base
    bw, bp = _betas(ks, k_w, _a_p(nu, delta))
    return base * np.abs(bp) ** 2 / np.abs(bw) ** 2


# --- a sinusoidal ripple (Sect. 15.1.1) ------------------------------------------------------------------------

def ripple_diffraction_cos(grazing, nu: float, wavelength: float, ripple_wavelength: float):
    """cos(theta_ps) of the order diffracted at K_s = K_i - K0 (15.5), lossless, propagation normal to the crests:
    nu (cos theta_wi - lambda_w / lambda_0). A propagating order needs |cos| <= 1."""
    return nu * (np.cos(grazing) - wavelength / ripple_wavelength)


def ripple_cutoff_wavelength(grazing, nu: float, ripple_wavelength: float):
    """(15.6): the order of (15.5) propagates only for water wavelengths above lambda_0 (cos theta_wi - 1 / nu)
    (a high-frequency cutoff)."""
    return ripple_wavelength * (np.cos(grazing) - 1.0 / nu)


def ripple_field(x, depth, grazing: float, frequency: float, c_w: float, a_rho: float, nu: float, delta: float,
                 h0: float, ripple_wavelength: float, theta0: float = 0.0):
    """Pressure in the sediment under the relief f(x) = sqrt(2) h0 cos(K0 x + theta0) (15.7)-(15.8) for a unit plane
    wave at `grazing` travelling along +x (normal to the crests): P = P_0 + P_- + P_+ (15.10), P_0 the zeroth order
    (15.1), P_+- the orders at K_s = K_i +- K0 (15.9) with the factor i of (15.3). `x` (m) along the ripple, `depth`
    (m, > 0) under the mean interface; broadcast. h0 = 0 gives (15.1) exactly."""
    x, depth = np.asarray(x, dtype=float), np.asarray(depth, dtype=float)
    k_w = 2.0 * np.pi * frequency / c_w
    a_p = _a_p(nu, delta)
    k_p = k_w / a_p
    K_i = k_w * np.cos(grazing)
    K0 = 2.0 * np.pi / ripple_wavelength
    p = (1.0 + flat_reflection(K_i, k_w, a_rho, nu, delta)) * np.exp(1j * K_i * x + 1j * k_p * _betas(K_i, k_w, a_p)[1] * depth)
    if h0 == 0.0:
        return p
    for sign in (1.0, -1.0):
        K_s = K_i + sign * K0
        bw_s, bp_s = _betas(K_s, k_w, a_p)
        p = p + (1j * k_w * h0 / (np.sqrt(2.0) * bw_s) * a_wp(K_s, K_i, k_w, a_rho, nu, delta)
                 * np.exp(1j * sign * theta0 + 1j * K_s * x + 1j * k_p * bp_s * depth))
    return p


# --- the averaged field (Sect. 15.1.2) ---------------------------------------------------------------------------

def _patch_geometry(cx, cy, interface_z, point, k_w, nu, source=None, grazing=None):
    """Per interface patch centre: K_i (15.12), K_s (15.13 with k_p', see the module docstring), the water-side
    incident |P_i| for a unit source (1 / r_i) or a unit plane wave (1), the sediment path L and the delay (15.16)
    without the water leg's constant for a plane wave."""
    point = np.asarray(point, dtype=float)
    dxs, dys = point[0] - cx, point[1] - cy
    L = np.sqrt(dxs**2 + dys**2 + (point[2] - interface_z) ** 2)
    K_s = (k_w / nu) * np.stack([dxs, dys], axis=-1) / L[..., None]
    if source is not None:
        source = np.asarray(source, dtype=float)
        dx, dy = cx - source[0], cy - source[1]
        r_i = np.sqrt(dx**2 + dy**2 + (interface_z - source[2]) ** 2)
        K_i = k_w * np.stack([dx, dy], axis=-1) / r_i[..., None]
        return K_i, K_s, 1.0 / r_i, L, r_i
    K_i = np.broadcast_to(np.array([k_w * np.cos(grazing), 0.0]), K_s.shape)
    return K_i, K_s, np.ones_like(L), L, cx * np.cos(grazing)


def mean_square_field(point, frequency: float, c_w: float, interface_z: float, *, a_rho: float, nu: float,
                      delta: float, w2: float, gamma2: float, K0: float = 0.0, source=None, grazing=None,
                      radius: float = 3.0, n: int = 601, times=None, pulse_ts: float | None = None,
                      obliquity: bool = True):
    """<|P_s|^2> at `point` (NED, z down, under the interface at `interface_z`) by quadrature on an n x n grid of
    half-width `radius` about the point: (15.14) for a CW source, (15.15) for the pulse u(t) = exp(-t^2 / t_s^2)
    at `times` (s) with `pulse_ts`. Incident field: a point source at `source` (unit pressure at 1 m) or a unit
    plane wave at `grazing` (radians, travelling along +x). Sediment attenuation exp(-2 k_p'' L), k_p'' = k_w
    delta / nu ((8.18), (8.26)). Returns the CW value, or (CW, pulsed array)."""
    k_w = 2.0 * np.pi * frequency / c_w
    point = np.asarray(point, dtype=float)
    u = np.linspace(-radius, radius, n)
    cx, cy = np.meshgrid(point[0] + u, point[1] + u, indexing="ij")
    area = (u[1] - u[0]) ** 2
    K_i, K_s, p_i, L, water = _patch_geometry(cx, cy, interface_z, point, k_w, nu, source, grazing)
    sig = sigma_wp(K_s, K_i, k_w, a_rho, nu, delta, w2, gamma2, K0, obliquity)
    k_loss = k_w * delta / nu
    g = sig * p_i**2 * np.exp(-2.0 * k_loss * L) / L**2 * area
    cw = float(g.sum())
    if times is None:
        return cw
    t_d = (water + L / nu) / c_w                                              # (15.16)
    times = np.asarray(times, dtype=float)
    pulsed = np.array([float((g * np.exp(-2.0 * (t - t_d) ** 2 / pulse_ts**2)).sum()) for t in times.ravel()])
    return cw, pulsed.reshape(times.shape)


# --- a realisation for the eigenray simulator ---------------------------------------------------------------------

def _splitmix(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def patch_coefficients(seed: int, level, i, j):
    """Complex Gaussian deviates with <|c|^2> = 1, one per interface cell (level, i, j), the same wherever and
    whenever the cell is used (a frozen interface): a hash of (seed, level, i, j), engine independent.

    h = SplitMix64(seed mod 2^64), then h = SplitMix64(h xor part) for part = level, i, j (as 64-bit two's
    complement), h2 = SplitMix64(h); u1 = ((h >> 11) + 0.5) / 2^53 in (0, 1), u2 = (h2 >> 11) / 2^53 in [0, 1);
    c = sqrt(-ln u1) exp(2 pi i u2): |c|^2 is exponential with mean 1 and the phase uniform (Box-Muller in polar
    form). SplitMix64 with its usual published constants (unconfirmed: the source was not reread; the constants are
    the ones in the code, and both engines use these, so the realisation is fixed whatever their origin). The C++ copy: libs/sim/src/penetration.cpp `rough_patch_coefficient`."""
    with np.errstate(over="ignore"):
        key = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
        h = _splitmix(key)
        for part in (level, i, j):
            h = _splitmix(h ^ np.asarray(part, dtype=np.int64).astype(np.uint64))
        h2 = _splitmix(h)
    u1 = ((h >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0**53
    u2 = (h2 >> np.uint64(11)).astype(np.float64) / 2.0**53
    return np.sqrt(-np.log(u1)) * np.exp(2j * np.pi * u2)


def quadtree_cells(px: float, py: float, radius: float, finest: float, refine: float):
    """Interface cells about the point (px, py): squares of a global quadtree whose finest cells have side `finest`,
    split while the cell's centre is nearer than `refine` sides to the point, kept while any part is within
    `radius`. Returns (level, i, j, centre x, centre y, side) arrays; cell (level, i, j) covers
    [i s, (i + 1) s) x [j s, (j + 1) s), s = finest 2^level, the same square for every point that uses it."""
    top = max(0, int(np.ceil(np.log2(max(radius, finest) / finest))))
    s = finest * 2.0**top
    i = np.arange(int(np.floor((px - radius) / s)), int(np.floor((px + radius) / s)) + 1)
    j = np.arange(int(np.floor((py - radius) / s)), int(np.floor((py + radius) / s)) + 1)
    I, J = (a.ravel() for a in np.meshgrid(i, j, indexing="ij"))
    level = top
    out = []
    while I.size:
        s = finest * 2.0**level
        cx, cy = (I + 0.5) * s, (J + 0.5) * s
        near = np.hypot(np.maximum(np.abs(px - cx) - s / 2, 0.0), np.maximum(np.abs(py - cy) - s / 2, 0.0))
        keep = near <= radius
        I, J, cx, cy = I[keep], J[keep], cx[keep], cy[keep]
        split = (np.hypot(px - cx, py - cy) < refine * s) if level > 0 else np.zeros(I.size, dtype=bool)
        leaf = ~split
        out.append((np.full(int(leaf.sum()), level), I[leaf], J[leaf], cx[leaf], cy[leaf], np.full(int(leaf.sum()), s)))
        if level == 0 or not split.any():
            break
        I = np.concatenate([2 * I[split] + a for a in (0, 1) for _ in (0, 1)])
        J = np.concatenate([2 * J[split] + b for _ in (0, 1) for b in (0, 1)])
        level -= 1
    return tuple(np.concatenate([o[k] for o in out]) for k in range(6))


def _sediment_radius(depth: float, k_loss: float, floor_db: float) -> float:
    """Horizontal reach about a point at `depth`: where the sediment leg's spreading and attenuation,
    20 log10(L / depth) + (20 / ln 10) k_loss (L - depth), reach floor_db (ours)."""
    lo, hi = depth, depth * 10.0 ** (floor_db / 20.0) + 1e-9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 20.0 * np.log10(mid / depth) + 20.0 / np.log(10.0) * k_loss * (mid - depth) < floor_db:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(max(hi**2 - depth**2, 0.0)))


class RoughPatches:
    """The roughness-scattered field in the sediment as eigenrays: ONE REALISATION of a frozen rough interface.

    The interface is cut into cells of a global quadtree (`quadtree_cells`: fine near the point's projection,
    coarser away from it), each carrying a complex Gaussian deviate c drawn from (seed, cell) (`patch_coefficients`),
    so every point, every leg and every ping sees the same interface. A cell of area A at R gives, per leg from a
    source at r_i to the point r:

        c sqrt(sigma_wp(K_s, K_i) A) exp(-k_p'' |r - R|) / (|R - r_i| |r - R|) / sqrt(a_rho)

    with sigma_wp of `sigma_wp` at the band centre, K_i from the source to the cell (15.12), K_s from the cell to the
    point (15.13, k_p'), 1 / sqrt(a_rho) the half of the reciprocity factor that the sediment tier gives every leg
    (docs/PHYSICS_NOTES.md section 31), and the delay (15.16): |R - r_i| / c_w + |r - R| / c_s. Averaged over seeds
    the CW intensity is (15.14) on the cell grid (`test_realisations_average_to_the_mean_square_field`). Cells are
    merged into one eigenray per delay bin of width `bin` (on a global grid of delays): the bin's ray sums its
    cells' amplitudes with their phases at the band centre, exp(-i w0 (t - t_b)), at the intensity-weighted mean
    delay t_b and direction; at the band centre this is exact, elsewhere in the band the phase error is at most
    pi |f - f0| bin.

    What it means for imaging: the realisation is coherent from ping to ping (a frozen interface), so a synthetic
    aperture or 3-D focuser sums the scattered paths of a buried point coherently. What is wrong in that: (a) a
    cell's deviate does not change with the geometry, whereas the relief component that scatters (the Bragg
    component at K_s - K_i) does when K_i turns by more than about one over the cell size: across a long aperture
    the realisation is too coherent and the focused target too sharp; (b) neighbouring points share cells only
    where their quadtrees coincide; (c) the delay bin and the band-centre cross section blur the arrival structure
    at the scale of `bin`. Expect a buried point below the critical angle to image as a speckled, displaced and
    smeared copy of itself at about the mean level; a single ping's level is one draw of an exponential
    intensity (Rayleigh amplitude) about the mean.
    """

    def __init__(self, *, c_w: float, c_s: float, interface_z: float, a_rho: float, nu: float, delta: float,
                 frequency: float, w2: float, gamma2: float, K0: float, seed: int = 0, patch: float | None = None,
                 refine: float = 3.0, floor_db: float = 30.0, bin: float | None = None):
        self.c_w, self.c_s, self.interface_z = float(c_w), float(c_s), float(interface_z)
        self.a_rho, self.nu, self.delta = float(a_rho), float(nu), float(delta)
        self.frequency, self.w2, self.gamma2, self.K0 = float(frequency), float(w2), float(gamma2), float(K0)
        self.seed = int(seed)
        self.patch = float(patch) if patch is not None else c_w / (2.0 * frequency)     # half a water wavelength (ours)
        self.refine, self.floor_db = float(refine), float(floor_db)
        self.bin = float(bin) if bin is not None else 1.0 / (4.0 * frequency)           # a quarter period (ours)
        self.k_w = 2.0 * np.pi * self.frequency / self.c_w
        self.k_loss = self.k_w * self.delta / self.nu

    def cpp_args(self) -> tuple:
        """(w2, gamma2, K0, seed, patch, refine, floor_db, bin) for the C++ engine's `sim_sediment_eigenrays`
        (libs/sim/src/penetration.cpp), every value stated; the seed as its 64-bit pattern, as the hash takes it."""
        return (self.w2, self.gamma2, self.K0, self.seed & 0xFFFFFFFFFFFFFFFF, self.patch, self.refine, self.floor_db,
                self.bin)

    def point_rays(self, source, point):
        """(delay [B], amplitude [B] complex, launch [B, 3], arrival [B, 3]) of the delay bins for one point."""
        source, point = np.asarray(source, dtype=float), np.asarray(point, dtype=float)
        depth = point[2] - self.interface_z
        radius = _sediment_radius(depth, self.k_loss, self.floor_db)
        level, i, j, cx, cy, side = quadtree_cells(point[0], point[1], radius, self.patch, self.refine)
        K_i, K_s, p_i, L, r_i = _patch_geometry(cx, cy, self.interface_z, point, self.k_w, self.nu, source=source)
        sig = sigma_wp(K_s, K_i, self.k_w, self.a_rho, self.nu, self.delta, self.w2, self.gamma2, self.K0)
        amp = (patch_coefficients(self.seed, level, i, j) * np.sqrt(sig * side**2) * np.exp(-self.k_loss * L)
               * p_i / L / np.sqrt(self.a_rho))
        t = r_i / self.c_w + L / self.c_s
        cell = np.stack([cx, cy, np.full_like(cx, self.interface_z)], axis=1)
        launch = (cell - source) / r_i[:, None]
        arrival = (point - cell) / L[:, None]
        key = np.floor(t / self.bin).astype(np.int64)
        bins, inv = np.unique(key, return_inverse=True)
        w = np.abs(amp) ** 2
        wsum = np.bincount(inv, weights=w, minlength=bins.size)
        wsafe = np.where(wsum > 0, wsum, 1.0)
        t_b = np.where(wsum > 0, np.bincount(inv, weights=w * t, minlength=bins.size) / wsafe,
                       np.bincount(inv, weights=t, minlength=bins.size) / np.bincount(inv, minlength=bins.size))
        # a bin of zero amplitude (w2 = 0) keeps the plain mean delay
        omega = 2.0 * np.pi * self.frequency
        ph = amp * np.exp(-1j * omega * (t - t_b[inv]))
        a_b = np.bincount(inv, weights=ph.real, minlength=bins.size) + 1j * np.bincount(inv, weights=ph.imag, minlength=bins.size)

        wt = np.where(wsum[inv] > 0, w, 1.0)          # a bin of zero amplitude (w2 = 0): plain means

        def mean_dir(v):
            m = np.stack([np.bincount(inv, weights=wt * v[:, c], minlength=bins.size) for c in range(3)], axis=1)
            return m / np.linalg.norm(m, axis=1)[:, None]
        return t_b, a_b, mean_dir(launch), mean_dir(arrival)


def roughness_from_config(roughness: dict) -> dict:
    """{w2, gamma2, K0, ...} of a `roughness:` block: w2 and gamma2 given, or from `sediment: name` of
    configs/sediments.yaml (w2 in m^(4 - gamma2)); K0 (1/m, the von Karman cutoff (D.7); 0 for the power law) must
    be stated: none is assumed."""
    import difflib
    for key in roughness:
        if key not in ROUGHNESS_KEYS:
            near = difflib.get_close_matches(str(key), ROUGHNESS_KEYS, n=1)
            raise ValueError(f"sediment: roughness.{key} unknown" + (f"; did you mean {near[0]!r}?" if near else
                                                                       f" (known: {', '.join(ROUGHNESS_KEYS)})"))
    out = {k: v for k, v in roughness.items() if k not in ("sediment", "sediment_file")}
    if "sediment" in roughness:
        par = bottom_model.sediment(roughness["sediment"], roughness.get("sediment_file"))
        for k in ("w2", "gamma2"):
            if k in out and float(out[k]) != float(par[k]):
                raise ValueError(f"sediment: roughness.{k} {out[k]!r} differs from sediment {roughness['sediment']!r}'s "
                                 f"{par[k]!r}; give one")
            out[k] = par[k]
    for k in ("w2", "gamma2", "K0"):
        if out.get(k) is None:
            raise ValueError(f"sediment: roughness.{k} is needed for penetration: rough"
                             + (" (1/m, the von Karman cutoff of (D.7); 0 for a pure power law)" if k == "K0" else
                                " (or roughness.sediment: a name in configs/sediments.yaml)") + "; there is no default")
    w2, gamma2, K0 = float(out["w2"]), float(out["gamma2"]), float(out["K0"])
    if not (w2 >= 0.0 and 2.0 < gamma2 < 4.0 and K0 >= 0.0):
        raise ValueError(f"sediment: roughness w2 {w2!r} (>= 0, m^(4 - gamma2)), gamma2 {gamma2!r} (2 < gamma2 < 4), "
                         f"K0 {K0!r} (>= 0, 1/m)")
    for k in ("patch", "refine", "bin"):
        if k in out and not float(out[k]) > 0.0:
            raise ValueError(f"sediment: roughness.{k} must be > 0")
    if "floor_db" in out and not float(out["floor_db"]) > 0.0:
        raise ValueError("sediment: roughness.floor_db must be > 0 dB")
    res = {"w2": w2, "gamma2": gamma2, "K0": K0, "seed": int(out.get("seed", 0))}
    for k in ("patch", "refine", "floor_db", "bin"):
        if k in out:
            res[k] = float(out[k])
    return res
