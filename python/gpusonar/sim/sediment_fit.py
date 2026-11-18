"""Fitting the bottom scattering model (sim/bottom.py) to measured backscattering strength.

For when measurements exist: a calibrated seabed scattering strength against grazing angle
(and frequency) from a survey, or values typed in from a paper. The three parameters that no
table read so far supplies for most sediments are fitted,

    w2      roughness spectral strength, m^(4 - gamma2)
    gamma2  roughness spectral exponent
    sigma2  volume parameter

with the density ratio, speed ratio and loss parameter held (from a named row of
configs/sediments.yaml or given), or fitted too if asked. The fit is least squares in dB.

What a data set can and cannot tell is reported, not guessed (the convention of simcal.py):

- a parameter whose standard error is large or that sits on a search limit is listed as
  unconstrained (a strong correlation between two parameters is noted);
- the reason is given from the fitted model itself: the share of the roughness term and of
  the volume term in the intensity at each data point (a term that is the smaller part
  everywhere is poorly seen by the data), the number of frequencies, the angle span.

Why the coverage matters (first-order perturbation theory, bottom.py):

- the volume term is proportional to sigma2 and depends on frequency only through the
  angles, so sigma2 is fixed by the level wherever the volume term dominates: soft sediments,
  and intermediate grazing angles below the critical angle's neighbourhood;
- the roughness term is A(angle) k^4 w2 (2 k cos g)^(-gamma2) for backscatter. At one
  frequency a change of gamma2 is, to first order, a change of level (the factor
  (2 k)^(-gamma2)) plus a weak change of shape (cos g)^(-gamma2), which is nearly flat at low
  grazing angles. So one frequency over low angles fixes only the combination
  w2 (2 k)^(-gamma2); the two separate with a second frequency (the level then goes as
  f^(4 - gamma2)) or with angles near the specular direction, where cos g varies fast (and
  where a Kirchhoff or small-slope roughness model must be used, not perturbation theory).

Uncertainties are the standard errors of linearised least squares: covariance
s^2 (J^T J)^-1 with s the stated measurement error in dB, or the rms residual when none is
given. They do not include calibration bias: an error of x dB in the absolute level of the
data moves w2 and sigma2 by the factor 10^(x / 10) and is not visible to the fit.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import bottom

SCATTER_KEYS = ("w2", "gamma2", "sigma2")
GEO_KEYS = ("rho", "nu", "delta")
LOG_KEYS = ("w2", "sigma2", "delta")                       # fitted as log10: they span decades
BOUNDS = {"w2": (1e-9, 1e-1), "gamma2": (2.05, 3.95), "sigma2": (1e-6, 1.0),
          "rho": (1.01, 3.0), "nu": (0.9, 3.0), "delta": (1e-4, 0.1)}
"""Search limits (ours: wide enough for clay to rock; gamma2 must lie between 2 and 4 for the
power law to have a finite structure function, bottom.py)."""
MAX_FACTOR = 3.0            # a log-fitted parameter known to worse than this factor (1 sigma) is unconstrained
MAX_STD = {"gamma2": 0.5, "rho": 0.3, "nu": 0.1}           # the same for the linear ones
MAX_CORRELATION = 0.98
DOMINANT_SHARE = 0.5        # which term the data mostly are, for the explanation of an unconstrained parameter


@dataclass
class SedimentFit:
    params: dict                                 # the six parameters
    fitted: tuple                                # names of those that were fitted
    std: dict                                    # 1 sigma: decades (log10) for LOG_KEYS, the parameter's units otherwise
    correlation: np.ndarray                      # between the fitted parameters, in `fitted` order
    unconstrained: list                          # fitted parameters the data do not fix
    notes: list                                  # why, and other warnings
    rms_residual_db: float
    sigma_db: float                              # the measurement error used for the uncertainties
    n_points: int
    grazing_deg: tuple                           # (min, max) of the data
    frequencies: tuple                           # Hz, the distinct ones
    roughness_share: tuple                       # (min, max) over the data of roughness / total intensity
    roughness_model: str
    sound_speed: float
    residuals_db: np.ndarray = field(repr=False, default=None)

    def factor(self, key: str) -> float:
        """1 sigma as a multiplicative factor for a log-fitted parameter."""
        return float(10.0 ** self.std[key])

    def model_db(self, grazing_deg, frequency):
        return _model_db(self.params, np.radians(np.asarray(grazing_deg, dtype=float)), np.asarray(frequency, dtype=float),
                         self.sound_speed, self.roughness_model)

    def yaml_row(self, name: str, data_source: str) -> str:
        """An entry for configs/sediments.yaml. `status: fitted (...)`; parameters the data do
        not fix are listed in `unconfirmed_fields`, so that using the row warns about them."""
        p = self.params
        lines = [f"{name}:"] + [f"  {k}: {p[k]:.6g}" for k in bottom.SEDIMENT_KEYS]
        lines.append(f"  status: fitted ({data_source})")
        if self.unconstrained:
            lines.append(f"  unconfirmed_fields: [{', '.join(self.unconstrained)}]")
        held = [k for k in bottom.SEDIMENT_KEYS if k not in self.fitted]
        freqs = ", ".join(f"{f / 1e3:g}" for f in self.frequencies)
        lines += ["  source: >-",
                  f"    Fitted by tools/fit_sediment.py ({self.roughness_model} roughness model) to {self.n_points} values of",
                  f"    backscattering strength, {self.grazing_deg[0]:g} to {self.grazing_deg[1]:g} degrees grazing, at {freqs} kHz:",
                  f"    {data_source}. Held: {', '.join(held) if held else 'nothing'}.",
                  "  fit:",
                  f"    rms_residual_db: {self.rms_residual_db:.3g}",
                  f"    measurement_error_db: {self.sigma_db:.3g}",
                  "    one_sigma:"]
        for k in self.fitted:
            lines.append(f"      {k}: " + (f"factor {self.factor(k):.3g}" if k in LOG_KEYS else f"{self.std[k]:.3g}"))
        if self.notes:
            lines.append("    notes:")
            lines += [f"      - \"{n}\"" for n in self.notes]
        return "\n".join(lines) + "\n"


def _model_db(params: dict, grazing, frequency, sound_speed: float, roughness_model: str):
    out = np.empty(grazing.shape)
    for f in np.unique(frequency):
        m = frequency == f
        out[m] = bottom.backscattering_strength_db(grazing[m], float(f), sound_speed, roughness_model=roughness_model, **params)
    return out


def _shares(params: dict, grazing, frequency, sound_speed: float, roughness_model: str):
    """Roughness intensity / total intensity at each data point."""
    share = np.empty(grazing.shape)
    for f in np.unique(frequency):
        m = frequency == f
        r = np.asarray(bottom.roughness_strength(grazing[m], grazing[m], float(f), sound_speed, roughness_model=roughness_model, **params))
        v = np.asarray(bottom.volume_strength(grazing[m], grazing[m], float(f), sound_speed, **params))
        share[m] = r / np.maximum(r + v, 1e-300)
    return share


def _pack(params: dict, keys) -> np.ndarray:
    return np.array([np.log10(params[k]) if k in LOG_KEYS else params[k] for k in keys])


def _unpack(x, keys, held: dict) -> dict:
    out = dict(held)
    for k, v in zip(keys, x):
        out[k] = float(10.0 ** v) if k in LOG_KEYS else float(v)
    return out


def fit(grazing_deg, frequency, strength_db, *, rho: float, nu: float, delta: float, sound_speed: float = 1500.0,
        sigma_db=None, fit_geoacoustic: bool = False, roughness_model: str = "perturbation", start: dict | None = None,
        n_starts: int = 12, seed: int = 0) -> SedimentFit:
    """Fit w2, gamma2 and sigma2 (and rho, nu, delta with `fit_geoacoustic`) to measured
    backscattering strength `strength_db` (dB re 1 m^2 per m^2: 10 log10 of the cross section
    per unit area and solid angle) at grazing angles `grazing_deg` and frequencies `frequency`
    (Hz, one per value or a single number). `sigma_db`: the measurement's standard error in
    dB (a number or one per value); without it the rms residual is used for the uncertainties.
    Several starts are tried and the best kept, because the dB misfit has local minima.
    `roughness_model` is named here and recorded in the result: "perturbation" unless given,
    which is NOT the forward model's default since D-86 (small_slope). Below about 60 degrees
    the two agree to a few tenths of a dB; with a free exponent the small-slope table would be
    rebuilt at every step (minutes instead of seconds), so it is for data nearer the specular
    direction, by request."""
    g = np.radians(np.asarray(grazing_deg, dtype=float).ravel())
    y = np.asarray(strength_db, dtype=float).ravel()
    f = np.broadcast_to(np.asarray(frequency, dtype=float).ravel(), g.shape).copy()
    if not (len(g) == len(y) and len(g) > 0):
        raise ValueError("fit: grazing_deg and strength_db must have the same, non-zero length")
    if np.any(g <= 0.0) or np.any(g > 0.5 * np.pi) or np.any(f <= 0.0) or not np.all(np.isfinite(y)):
        raise ValueError("fit: grazing angles must be in (0, 90] degrees, frequencies positive, strengths finite")
    weight = np.ones(len(y)) if sigma_db is None else 1.0 / np.broadcast_to(np.asarray(sigma_db, dtype=float).ravel(), y.shape)
    keys = SCATTER_KEYS + (GEO_KEYS if fit_geoacoustic else ())
    held = {} if fit_geoacoustic else {"rho": float(rho), "nu": float(nu), "delta": float(delta)}
    if len(y) < len(keys):
        raise ValueError(f"fit: {len(y)} values cannot determine {len(keys)} parameters")
    lo = _pack({k: BOUNDS[k][0] for k in keys}, keys)
    hi = _pack({k: BOUNDS[k][1] for k in keys}, keys)

    def residual(x):
        return weight * (_model_db(_unpack(x, keys, held), g, f, sound_speed, roughness_model) - y)

    first = dict({"w2": 1e-5, "gamma2": 3.0, "sigma2": 1e-3, "rho": rho, "nu": nu, "delta": delta}, **(start or {}))
    rng = np.random.default_rng(seed)
    starts = [np.clip(_pack(first, keys), lo, hi)] + [rng.uniform(lo, hi) for _ in range(max(int(n_starts) - 1, 0))]
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in starts:
            try:
                sol = least_squares(residual, x0, bounds=(lo, hi), x_scale=np.maximum(hi - lo, 1e-3) / 10.0)
            except (ValueError, FloatingPointError):
                continue
            if best is None or sol.cost < best.cost:
                best = sol
    if best is None:
        raise RuntimeError("fit: no start converged")
    params = _unpack(best.x, keys, held)
    resid = _model_db(params, g, f, sound_speed, roughness_model) - y
    dof = max(len(y) - len(keys), 1)
    rms = float(np.sqrt(np.sum(resid**2) / dof))
    sigma_used = rms if sigma_db is None else float(np.mean(1.0 / weight))

    # Linearised covariance from a central-difference Jacobian in dB per unit of each fitted
    # variable (log10 for the ones that span decades).
    jac = np.empty((len(y), len(keys)))
    for j in range(len(keys)):
        h = 1e-4 * max(1.0, abs(best.x[j]))
        up, dn = best.x.copy(), best.x.copy()
        up[j], dn[j] = min(best.x[j] + h, hi[j]), max(best.x[j] - h, lo[j])
        jac[:, j] = (_model_db(_unpack(up, keys, held), g, f, sound_speed, roughness_model)
                     - _model_db(_unpack(dn, keys, held), g, f, sound_speed, roughness_model)) / (up[j] - dn[j])
    jw = jac * (weight[:, None] if sigma_db is not None else 1.0 / max(rms, 1e-6))
    # Pseudo-inverse: a direction the data do not see gets a huge, not an infinite, variance.
    u, s, vt = np.linalg.svd(jw, full_matrices=False)
    floor = 1e-6 * max(s.max(), 1e-300)
    cov = (vt.T / np.maximum(s, floor) ** 2) @ vt
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    share = _shares(params, g, f, sound_speed, roughness_model)
    freqs = tuple(float(v) for v in np.unique(f))
    unconstrained, notes = [], []

    def flag(key, why):
        if key in keys and key not in unconstrained:
            unconstrained.append(key)
        if why not in notes:
            notes.append(why)

    for j, k in enumerate(keys):
        limit = np.log10(MAX_FACTOR) if k in LOG_KEYS else MAX_STD.get(k, np.inf)
        if std[j] > limit:
            flag(k, f"{k}: 1 sigma is " + (f"a factor {10.0 ** min(std[j], 30.0):.3g}" if k in LOG_KEYS else f"{std[j]:.3g}")
                 + ", too wide to call the value known")
        if best.x[j] - lo[j] < 1e-6 * (hi[j] - lo[j]) or hi[j] - best.x[j] < 1e-6 * (hi[j] - lo[j]):
            flag(k, f"{k}: the fit ran to the search limit {BOUNDS[k][0] if best.x[j] - lo[j] < hi[j] - best.x[j] else BOUNDS[k][1]:g}")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if abs(corr[i, j]) > MAX_CORRELATION:
                # A note, not a verdict: strongly correlated parameters can still each be known
                # well (their standard errors, which include the correlation, decide that).
                notes.append(f"{keys[i]} and {keys[j]} trade off against each other (correlation {corr[i, j]:+.3f}): "
                             "their errors are not independent")
    # The reason, from the fitted model's own split of the intensity. (In a degenerate fit the
    # split itself is uncertain, so these explain a verdict the standard errors gave; they do not give one.)
    if ("w2" in unconstrained or "gamma2" in unconstrained) and share.max() < DOMINANT_SHARE:
        notes.append(f"the roughness term is the smaller part of the intensity at every data point (at most {share.max():.0%}): "
                     "the data are mostly volume scattering and do not fix w2 or gamma2")
    if "sigma2" in unconstrained and share.min() > DOMINANT_SHARE:
        notes.append(f"the volume term is the smaller part of the intensity at every data point (at most {1.0 - share.min():.0%}): "
                     "the data are mostly roughness scattering and give only an upper limit on sigma2")
    if len(freqs) == 1 and ("w2" in unconstrained or "gamma2" in unconstrained):
        notes.append("one frequency: the roughness level fixes w2 (2 k)^(-gamma2) only; a second frequency (level as "
                     "f^(4 - gamma2)) or angles near the specular direction separate w2 from gamma2")
    if roughness_model == "perturbation" and np.degrees(g.max()) > 60.0:
        notes.append("data above 60 degrees grazing with the perturbation roughness model, which is not valid there: "
                     "use roughness_model small_slope or combined")
    if sigma_db is not None and rms > 2.0 * sigma_used:
        notes.append(f"the rms residual ({rms:.2f} dB) is more than twice the stated measurement error ({sigma_used:.2f} dB): "
                     "the model does not describe these data to within their error, and the uncertainties are too small")
    return SedimentFit(params={k: params[k] for k in bottom.SEDIMENT_KEYS}, fitted=tuple(keys),
                       std={k: float(std[j]) for j, k in enumerate(keys)}, correlation=corr, unconstrained=unconstrained,
                       notes=notes, rms_residual_db=rms, sigma_db=sigma_used, n_points=len(y),
                       grazing_deg=(float(np.degrees(g.min())), float(np.degrees(g.max()))), frequencies=freqs,
                       roughness_share=(float(share.min()), float(share.max())), roughness_model=roughness_model,
                       sound_speed=float(sound_speed), residuals_db=resid)
