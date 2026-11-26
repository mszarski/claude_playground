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

The roughness model is `small_slope` unless named (`DEFAULT_ROUGHNESS_MODEL`, the forward model's
default since D-86; A10 of docs/references/CONFORMANCE.md): the lowest-order small-slope result, which
is first-order perturbation theory away from the specular direction and the Kirchhoff result in it.
Jackson and Richardson (2007) fitted with perturbation theory and show the two overlapping below about
70 degrees for medium sand ([N4] Sec. 2.5, docs/references/jackson_richardson_2007/04_*.md); above
that only small-slope is valid. `roughness_model="perturbation"` gives the fits made before A10, bit for
bit (tests/python/test_sediment_fit.py). Cost: every step with a new gamma2 builds a Kirchhoff table
(bottom._kirchhoff_table), so a small-slope fit takes seconds where a perturbation fit takes a fraction
of one (7.5 s against 0.2 s for 42 values, 12 starts, one core). What got worse: on volume-dominated
data at one frequency (very fine silt, 100 kHz, 0.5 dB noise) an unbounded small-slope fit can explain
part of the noise with a relief about 100 times rougher, flag nothing and put sigma2 at 0.65-0.78 of the
truth (5 of 10 noise draws; perturbation: none); the row's default prior (`sediment=name`, or
`bounds=row_prior_bounds(name)[0]`; the tool's default since 2026-11-26) removes it in all ten draws 0-9 (sigma2 0.85-1.05 of the truth), w2 or gamma2 then on a
prior bound in eight (tests/python/test_sediment_fit.py `test_a10_small_slope_on_volume_dominated_data_needs_prior_bounds`).

Why the coverage matters (first-order perturbation theory, bottom.py; the same for small-slope away
from the specular direction):

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

Opt-in (G-9 of docs/tasks/SST_GAPS_JACKSON.md; with none of these and `roughness_model="perturbation"`
the fit is bit-identical to before them, held by tests/python/test_sediment_fit.py):

- `independent_samples`: the number N of independent intensity samples behind each value (one
  number or one per value). The error of each value is then Jackson and Richardson's (2007)
  App. G.2.3 rule, sigma_db = 10 log10(1 + 1/sqrt(N)) (`gpusonar.measure`), and the least squares
  is weighted by it. That is the book's UPPER 1-sigma bar; the dB standard deviation of an
  N-sample intensity mean is somewhat larger (1.10 against 0.97 dB at N = 16) and the dB mean is
  low by about 4.34 / (2 N) dB (`measure.intensity_mean_db_bias`), neither corrected by this
  rule; `sample_error="gamma"` uses that standard deviation and removes the bias (ours). Measured
  coverage (synthetic N = 16 data, 100 sets, perturbation model): 55 % of the fitted parameters within 1 sigma with the
  book's rule, 61 % with "gamma", against 68 % nominal (tests/python/test_sediment_fit.py).
- `bounds`: search limits replacing `BOUNDS` for the keys named, e.g. `prior_bounds(phi)` (the
  spreads the reference notes give per sediment class: `PRIOR_BOUNDS`) or `site_bounds(...)` (the
  within-site coefficients of variation of App. C about measured values). A fit that ends on a
  prior bound is set by the prior, not by the data, and is listed as unconstrained; every parameter
  that ends on a bound or a search limit is in `SedimentFit.on_bound`. `fit` applies no prior
  unless given `bounds` or a sediment row (next item).
- `sediment` (the owner, 2026-11-26: "Library taking a row: yes"): the name of a row of
  configs/sediments.yaml (or of `sediment_file`). The fit then does what tools/fit_sediment.py does
  with `--sediment` (the tool calls `fit` with its row: one implementation): rho, nu, delta not given
  are the row's (`bottom.sediment_reflection(..., shear=False)`, the fluid model); unless `bounds` are
  given (they win, as the tool's `--prior-bounds PHI`), the row's default prior
  `row_prior_bounds(name)` is applied (w2, gamma2, sigma2 of its grain size; rho, nu, delta never: the
  owner's decision of the same day) and its source recorded in `bounds_source`; a row without a grain
  size gets none and `bounds_source` says why, in the words the tool prints after "bounds used: ".
  `row_prior=False` names the row but applies no prior (the tool's `--no-prior-bounds`). Without
  `sediment` every call is what it was, bit for bit (tests/python/test_sediment_fit.py).
- `hold`: parameters held at given values instead of fitted (e.g. gamma2 when the data are at
  one frequency and too few for three parameters).
With any of them the result also carries the extended report: the covariance of the fitted
variables, the error of the predicted strength (`SedimentFit.prediction_std_db`), and whether the
roughness models can be told apart on these data (`roughness_model_spread_db`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..acoustics import NOMINAL_SOUND_SPEED
from .. import measure
from scipy.optimize import least_squares

from . import bottom

SCATTER_KEYS = ("w2", "gamma2", "sigma2")
GEO_KEYS = ("rho", "nu", "delta")
LOG_KEYS = ("w2", "sigma2", "delta")                       # fitted as log10: they span decades
DEFAULT_ROUGHNESS_MODEL = bottom.DEFAULT_ROUGHNESS_MODEL
"""The roughness model of a fit that names none: the forward model's (`small_slope`, D-86; A10).
`perturbation` was the default before A10."""
BOUNDS = {"w2": (1e-9, 1e-1), "gamma2": (2.05, 3.95), "sigma2": (1e-6, 1.0),
          "rho": (1.01, 3.0), "nu": (0.9, 3.0), "delta": (1e-4, 0.1)}
"""Search limits (ours: wide enough for clay to rock; gamma2 must lie between 2 and 4 for the
power law to have a finite structure function, bottom.py)."""
MAX_FACTOR = 3.0            # a log-fitted parameter known to worse than this factor (1 sigma) is unconstrained
MAX_STD = {"gamma2": 0.5, "rho": 0.3, "nu": 0.1}           # the same for the linear ones
MAX_CORRELATION = 0.98
DOMINANT_SHARE = 0.5        # which term the data mostly are, for the explanation of an unconstrained parameter

# --- Opt-in prior bounds (G-9) ---------------------------------------------------------------------
# Every value from the reference notes on Jackson and Richardson (2007) ("J&R"); [N1] =
# docs/references/jackson_richardson_2007/01_sediment_properties.md, [N2] = 02_roughness_heterogeneity_statistics.md.
# Neither Table 6.1 nor Table C.1 is carried here (the notes ask that they not be): only the class summaries.

# sigma2, 1e-4 to 5e-3. The lower end and the old ceiling 4e-3: J&R p. 382 give 0.0001-0.004 as the typical range
# and call values above 0.004 a possible sign that the weak (single) scattering assumption fails (confirmed in the
# book via [N1 Sec. 4], [N2 Sec. 3.3]). The ceiling 5e-3 (the owner, 2026-11-26: "yes raise the ceiling to 0.005"):
# Jackson and Briggs (1992), in the text of their eq. (3) (sigma2 = sigma_v / alpha), put the fitted values reported
# in the work they cite at 0.0001 to 0.005, and print 0.005 as their own fit at the Arafura Sea site (Table V; the
# text, p. 974, says 0.004-0.006 fit reasonably) (confirmed via docs/references/bottom_backscatter_jackson_briggs_1992.md).
# That top value is ONE site, and the paper's own caveats on it: the sediment is bimodal (sand and gravel in a silty
# clay, so its mean grain size misleads) and 0.005 lies at the border of single scattering, where the first-order
# volume model may fail. The prior therefore admits values the book calls suspect; a fit that lands above 0.004 should
# be read with that warning. The search limit BOUNDS["sigma2"] (ours) and the soft rows' information key
# `sigma2_range: [0.0008, 0.005]` (configs/sediments.yaml) are separate and unchanged; the latter already ends at 0.005.
PRIOR_BOUNDS = {
    "rho": (1.1, 2.2),        # J&R Tables 5.1/5.4 ranges of measured sediments, "a_rho from 1.1 to about 2.2" [N1 Sec. 4]
    "nu": (0.97, 1.2),        # the same tables' range [N1 Sec. 4]
    "sigma2": (1e-4, 5e-3),   # two sources: the comment above
    "gamma2": (2.33, 3.95),   # J&R Table 6.1 (Briggs, 52 rows): gamma2 = gamma1 + 1 from 2.33 to 4.35 [N2 Sec. 2.2];
                              # the upper end capped at our search limit 3.95 (power law, bottom.py)
}
"""Bounds the notes give for every class. The TR 9407 grain-size rows are a model-fit convention ("ad hoc",
J&R p. 314), not measured properties: those coarser than about 1 phi have rho above 2.2 and nu above 1.2
(`bottom.GRAIN_SIZE_TABLE`), so these bounds do not suit a fit that starts from such a row."""

W2_SPREAD_DB = {"sand": (-15.3, 9.9), "intermediate": (-15.3, 14.5), "mud": (-7.1, 14.5)}
"""Range, in dB, of the roughness spectrum W(K) of J&R Table 6.1's rows about our grain-size rule
(`bottom.grain_size_scattering`) at the backscatter Bragg wavenumber K = 2 k cos(30 deg), the union over
10, 100 and 300 kHz: sands (Mz < 3 phi, 41 rows) -15.3 to +9.9 dB, Mz >= 5.5 (8 rows, 4 sites) -7.1 to
+14.5 dB [N2 Sec. 3.2] (the comparison is the note's, from the table with the corrected (D.19)). The three
rows between 3 and 5.5 phi give no spread: the union of the two is used there (ours). Applied to w2 as a
factor about the rule's w2: exact for W only at gamma2 = 3.25 (the rule's), approximate otherwise."""

WITHIN_SITE_CV = {"sand": {"rho": 0.03, "nu": 0.024, "delta": 0.48},
                  "mud": {"rho": 0.06, "nu": 0.02, "delta": 0.42}}
"""Within-site (core-to-core and depth) coefficients of variation, the upper end of the "mostly" range of
J&R App. C, Table C.1, as summarised in [N1 Sec. 2.9]: sands (Mz < 3 phi) bulk density 0.5-3 %, V_p
0.3-2.4 %, alpha_p 17-48 %; silts and clays (Mz > 5 phi) 2-6 %, 0.1-2 %, 12-42 %. The ratios' CVs equal
those of the quantities (the table's caption): rho from bulk density, nu from V_p. delta's is taken equal
to alpha_p's (ours: delta is proportional to alpha_p nu / f and nu's CV is two orders smaller). Between 3
and 5 phi the larger of the two (ours)."""


def _check_phi(phi: float) -> float:
    return bottom._check_grain_size(float(phi))


def prior_bounds(phi: float) -> dict:
    """Opt-in search limits for a site of mean grain size `phi` (phi units, -1 to 9): `PRIOR_BOUNDS` for
    rho, nu, sigma2, gamma2; w2 the grain-size rule's value times the class's `W2_SPREAD_DB`; delta the
    plain search limits (the notes give no class spread for it). Each bound is intersected with `BOUNDS`."""
    phi = _check_phi(phi)
    cls = "sand" if phi < 3.0 else ("mud" if phi >= 5.5 else "intermediate")
    w2 = bottom.grain_size_scattering(phi)["w2"]
    lo_db, hi_db = W2_SPREAD_DB[cls]
    out = dict(PRIOR_BOUNDS, w2=(w2 * 10.0 ** (lo_db / 10.0), w2 * 10.0 ** (hi_db / 10.0)), delta=BOUNDS["delta"])
    return {k: (max(lo, BOUNDS[k][0]), min(hi, BOUNDS[k][1])) for k, (lo, hi) in out.items()}


SEARCH_LIMITS_ONLY = "the search limits only (sediment_fit.BOUNDS, ours); no prior bounds applied: "
"""The start of what `SedimentFit.bounds_source` (and the tool's "bounds used:" line) says when a row was named and
no prior applied; the reason follows."""

DEFAULT_PRIOR_KEYS = SCATTER_KEYS
"""The keys of `prior_bounds` that `row_prior_bounds` (the default of the tool and of `fit(sediment=...)`) applies: the scattering parameters.
Not rho, nu, delta: the TR 9407 rows coarser than about 1 phi lie outside the rho and nu priors (the note on
`PRIOR_BOUNDS`), and without --fit-geoacoustic they are held anyway (ours)."""


def row_prior_bounds(name: str, path=None) -> tuple:
    """The default prior for the sediment row `name` (of configs/sediments.yaml or `path`) that `fit(sediment=name)`
    applies, and so tools/fit_sediment.py --sediment name: `(bounds, source)` with `bounds` the `DEFAULT_PRIOR_KEYS` of `prior_bounds(phi)`, phi the row's
    `mean_grain_size_phi`, and `source` saying so; `(None, reason)` when the row carries no grain size. The
    owner's decision (2026-11-26): "yes apply prior bounds by default" when the row named has a grain size.
    Where the bounds come from: `PRIOR_BOUNDS` (sigma2's lower end and gamma2's lower end from J&R, confirmed in
    the book via docs/references/; sigma2's ceiling 5e-3 from Jackson and Briggs's (1992) printed fits, confirmed
    via docs/references/, above J&R's typical 4e-3: the comment at `PRIOR_BOUNDS`; gamma2's upper end our search limit) and `W2_SPREAD_DB` (the spread of J&R
    Table 6.1's rows about our grain-size rule, our comparison) about `bottom.grain_size_scattering(phi)`
    (Snellen et al. 2013 eq. (5), via secondary): the widths are ours."""
    entry = bottom._entry(name, path)
    phi = entry.get("mean_grain_size_phi")
    if phi is None:
        return None, f"the row {name!r} carries no mean_grain_size_phi"
    phi = float(phi)
    full = prior_bounds(phi)
    bounds = {k: full[k] for k in DEFAULT_PRIOR_KEYS}
    return bounds, (f"default prior for the row {name!r}, mean grain size {phi:g} phi (sediment_fit.row_prior_bounds: "
                    f"prior_bounds of {', '.join(DEFAULT_PRIOR_KEYS)}; --no-prior-bounds turns it off)")


def site_bounds(rho: float, nu: float, delta: float, phi: float, n_sigma: float = 2.0) -> dict:
    """Opt-in search limits for rho, nu, delta measured at the site (cores) and fitted with
    `fit_geoacoustic`: each value -+ `n_sigma` times its within-site coefficient of variation
    (`WITHIN_SITE_CV`), intersected with `BOUNDS`."""
    phi = _check_phi(phi)
    sand, mud = WITHIN_SITE_CV["sand"], WITHIN_SITE_CV["mud"]
    cv = sand if phi < 3.0 else (mud if phi > 5.0 else {k: max(sand[k], mud[k]) for k in sand})
    out = {}
    for k, v in (("rho", rho), ("nu", nu), ("delta", delta)):
        lo, hi = v * (1.0 - n_sigma * cv[k]), v * (1.0 + n_sigma * cv[k])
        out[k] = (max(lo, BOUNDS[k][0]), min(hi, BOUNDS[k][1]))
    return out


def _refuse_unknown(given, known, where: str) -> None:
    """An unknown key is refused by name with the nearest known one (the rule of D-166/D-177)."""
    import difflib
    for key in given:
        if key not in known:
            near = difflib.get_close_matches(str(key), list(known), n=1)
            raise ValueError(f"{where}{key}: unknown key" + (f"; did you mean {near[0]!r}?" if near else f" (known: {', '.join(known)})"))


def roughness_model_spread_db(params: dict, grazing_deg, frequency, sound_speed: float = NOMINAL_SOUND_SPEED) -> float:
    """The largest difference, over the given points, between the backscattering strengths of the
    roughness models of `bottom.ROUGHNESS_MODELS` with the same six parameters, dB. Below the
    measurement error the data cannot tell the models apart (D-193 found 0.001 dB at its site).
    Evaluated at the given parameters: in a fit where w2 is unconstrained it says only what holds there.
    NaN when gamma2 is outside (2, 4), where only the perturbation model is defined (bottom.structure_constant):
    there is nothing to compare."""
    if not 2.0 < float(params["gamma2"]) < 4.0:
        return float("nan")
    g = np.radians(np.asarray(grazing_deg, dtype=float).ravel())
    f = np.broadcast_to(np.asarray(frequency, dtype=float).ravel(), g.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curves = np.array([_model_db(params, g, f, sound_speed, m) for m in bottom.ROUGHNESS_MODELS])
    return float(np.max(curves.max(axis=0) - curves.min(axis=0)))


def yaml_number(value: float, spec: str) -> str:
    """`value` formatted with `spec`, written so that PyYAML (YAML 1.1) reads it back as a number: `:g` gives
    `1e-05` for 0.00001, which YAML 1.1 reads as a STRING (a float there needs a dot in the mantissa); it is
    written `1.0e-05` (sonar-iwq)."""
    text = format(float(value), spec)
    mantissa, e, exponent = text.partition("e")
    return f"{mantissa}.0e{exponent}" if e and "." not in mantissa else text


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
    # Covariance of the fitted variables (log10 for LOG_KEYS), in `fitted` order: std = sqrt(diag).
    covariance: np.ndarray = field(repr=False, default=None)
    # Opt-in (G-9): None unless the fit was asked for them.
    independent_samples: tuple | None = None     # (min, max) of N over the data
    sample_error: str | None = None              # "book" or "gamma", with independent_samples
    bounds: dict | None = None                   # the search limits used, when other than BOUNDS
    bounds_source: str | None = None             # what they are, for the row; with a `sediment` row and no
                                                 # bounds, why none (SEARCH_LIMITS_ONLY + the reason)
    held: dict | None = None                     # parameters held by `hold`
    # Fitted parameters that ended on a bound: {key: {"bound": value, "side": "low" | "high", "kind": "prior bound"
    # | "search limit"}}. Such a value is set by the bound, not measured (also in `unconstrained` and `notes`).
    on_bound: dict = field(default_factory=dict)

    def prediction_std_db(self, grazing_deg, frequency) -> np.ndarray:
        """1 sigma of the predicted backscattering strength (dB) at the given angles and frequencies,
        from the fitted variables' covariance: sqrt(diag(J C J^T)), J the derivative of the model in dB
        with respect to the fitted variables (log10 for LOG_KEYS) at the fitted values (linearised;
        held parameters contribute nothing). Large where the data do not reach (other frequencies,
        angles) or where a parameter is unconstrained."""
        g = np.radians(np.asarray(grazing_deg, dtype=float).ravel())
        f = np.broadcast_to(np.asarray(frequency, dtype=float).ravel(), g.shape).copy()
        keys = self.fitted
        held = {k: v for k, v in self.params.items() if k not in keys}
        x = _pack(self.params, keys)
        jac = np.empty((len(g), len(keys)))
        for j in range(len(keys)):
            h = 1e-4 * max(1.0, abs(x[j]))
            up, dn = x.copy(), x.copy()
            up[j], dn[j] = x[j] + h, x[j] - h
            jac[:, j] = (_model_db(_unpack(up, keys, held), g, f, self.sound_speed, self.roughness_model)
                         - _model_db(_unpack(dn, keys, held), g, f, self.sound_speed, self.roughness_model)) / (2.0 * h)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", jac, self.covariance, jac), 0.0))

    def factor(self, key: str) -> float:
        """1 sigma as a multiplicative factor for a log-fitted parameter (inf beyond 1e300, where 10 **
        std overflowed before: a parameter the data do not see at all)."""
        return float(10.0 ** self.std[key]) if self.std[key] < 300.0 else float("inf")

    def model_db(self, grazing_deg, frequency):
        return _model_db(self.params, np.radians(np.asarray(grazing_deg, dtype=float)), np.asarray(frequency, dtype=float),
                         self.sound_speed, self.roughness_model)

    def yaml_row(self, name: str, data_source: str) -> str:
        """An entry for configs/sediments.yaml. `status: fitted (...)`; parameters the data do
        not fix are listed in `unconfirmed_fields`, so that using the row warns about them."""
        p = self.params
        lines = [f"{name}:"] + [f"  {k}: {yaml_number(p[k], '.6g')}" for k in bottom.SEDIMENT_KEYS]
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
                  f"    rms_residual_db: {yaml_number(self.rms_residual_db, '.3g')}",
                  f"    measurement_error_db: {yaml_number(self.sigma_db, '.3g')}"]
        if self.independent_samples is not None:
            lo, hi = self.independent_samples
            lines.append(f"    independent_samples: {yaml_number(lo, '.6g')}" if lo == hi else
                         f"    independent_samples: [{yaml_number(lo, '.6g')}, {yaml_number(hi, '.6g')}]")
            lines.append(f"    sample_error: {self.sample_error}")
        if self.held:
            lines.append(f"    held: [{', '.join(self.held)}]")
        if self.bounds is not None:
            lines.append(f"    bounds: \"{self.bounds_source or 'given'}\"")
            # Only in a bounded fit's row, so that an unbounded fit's row stays what it was (a search-limit hit
            # is in its notes and unconfirmed_fields as before).
            if self.on_bound:
                lines.append(f"    on_bound: [{', '.join(self.on_bound)}]   # set by the bound, not measured")
        lines.append("    one_sigma:")
        for k in self.fitted:
            lines.append(f"      {k}: " + (f"factor {self.factor(k):.3g}" if k in LOG_KEYS else yaml_number(self.std[k], '.3g')))
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


def _row(sediment, sediment_file, row_prior: bool, geo: dict, bounds, bounds_source):
    """What naming a row adds to a fit (the tool's --sediment, the module docstring): the held rho, nu, delta not
    given, and the default prior unless `bounds` are given. Returns (geo, bounds, bounds_source)."""
    if sediment is None:
        if sediment_file is not None:
            raise ValueError("fit: sediment_file given without a sediment row to read from it")
        return geo, bounds, bounds_source
    table = bottom.sediment_table(sediment_file)
    if sediment not in table:
        import difflib
        near = difflib.get_close_matches(str(sediment), [str(n) for n in table], n=1)
        raise ValueError(f"fit: sediment {sediment!r}: unknown row" + (f"; did you mean {near[0]!r}?" if near else
                         f" (known: {', '.join(sorted(map(str, table)))})"))
    if any(v is None for v in geo.values()):
        row = bottom.sediment_reflection(sediment, sediment_file, shear=False)      # the fluid model is fitted
        geo = {k: row[k] if v is None else v for k, v in geo.items()}
    if bounds is None:
        if not row_prior:
            bounds_source = SEARCH_LIMITS_ONLY + "the row's prior turned off (row_prior=False; the tool's --no-prior-bounds)"
        else:
            bounds, why = row_prior_bounds(sediment, sediment_file)
            if bounds is None:
                bounds_source = SEARCH_LIMITS_ONLY + why
            elif bounds_source is None:
                bounds_source = why
    return geo, bounds, bounds_source


def fit(grazing_deg, frequency, strength_db, *, rho: float | None = None, nu: float | None = None, delta: float | None = None,
        sound_speed: float = NOMINAL_SOUND_SPEED,
        sigma_db=None, fit_geoacoustic: bool = False, roughness_model: str = DEFAULT_ROUGHNESS_MODEL, start: dict | None = None,
        n_starts: int = 12, seed: int = 0, independent_samples=None, bounds: dict | None = None,
        bounds_source: str | None = None, hold: dict | None = None, sample_error: str = "book",
        sediment: str | None = None, sediment_file=None, row_prior: bool = True) -> SedimentFit:
    """Fit w2, gamma2 and sigma2 (and rho, nu, delta with `fit_geoacoustic`) to measured
    backscattering strength `strength_db` (dB re 1 m^2 per m^2: 10 log10 of the cross section
    per unit area and solid angle) at grazing angles `grazing_deg` and frequencies `frequency`
    (Hz, one per value or a single number). `sigma_db`: the measurement's standard error in
    dB (a number or one per value); without it the rms residual is used for the uncertainties.
    Several starts are tried and the best kept, because the dB misfit has local minima.
    `roughness_model` is recorded in the result and the row: `DEFAULT_ROUGHNESS_MODEL`
    ("small_slope", the forward model's default) unless given; "perturbation" was the default
    before A10 and reproduces those fits. Below about 60 degrees the two agree to a few tenths
    of a dB; above, only small-slope is valid. Every model but "perturbation" needs
    2 < gamma2 < 4: bounds or a held gamma2 outside that are refused. A small-slope fit builds a
    Kirchhoff table at every new gamma2: seconds, not a fraction of one.

    Opt-in (G-9; the module docstring): `independent_samples` N (a number or one per value; not
    with `sigma_db`) sets each value's error to 10 log10(1 + 1/sqrt(N)) (`measure.sigma_db_from_samples`,
    `sample_error` "book"), or, with `sample_error` "gamma", to the standard deviation of the dB of an
    N-sample intensity mean (`measure.intensity_mean_db_std`) and removes that dB value's expected
    bias (`measure.intensity_mean_db_bias`, -0.14 dB at N = 16) from the data before fitting (ours);
    `bounds` {key: (low, high)} replaces the search limits of the keys named (`prior_bounds`,
    `site_bounds`; `bounds_source` says what they are, for the row); `hold` {key: value} holds
    parameters instead of fitting them. With any of them the result's notes also say whether the
    roughness models can be told apart on these data, and which held values lie outside the bounds.

    Opt-in (the owner, 2026-11-26): `sediment` names a row (of configs/sediments.yaml, or of
    `sediment_file`), e.g. `fit(g, f, y, sigma_db=0.5, sediment="very_fine_silt")`: rho, nu, delta not
    given are the row's, and unless `bounds` are given the row's default prior (`row_prior_bounds`) is
    applied, as by tools/fit_sediment.py --sediment; `row_prior=False` applies none (--no-prior-bounds).
    rho, nu, delta are required unless a row is named."""
    geo, bounds, bounds_source = _row(sediment, sediment_file, row_prior, {"rho": rho, "nu": nu, "delta": delta},
                                      bounds, bounds_source)
    if any(v is None for v in geo.values()):
        raise TypeError("fit: give rho, nu and delta, or name a sediment row (sediment=...)")
    rho, nu, delta = geo["rho"], geo["nu"], geo["delta"]
    g = np.radians(np.asarray(grazing_deg, dtype=float).ravel())
    y = np.asarray(strength_db, dtype=float).ravel()
    f = np.broadcast_to(np.asarray(frequency, dtype=float).ravel(), g.shape).copy()
    if not (len(g) == len(y) and len(g) > 0):
        raise ValueError("fit: grazing_deg and strength_db must have the same, non-zero length")
    if np.any(g <= 0.0) or np.any(g > 0.5 * np.pi) or np.any(f <= 0.0) or not np.all(np.isfinite(y)):
        raise ValueError("fit: grazing angles must be in (0, 90] degrees, frequencies positive, strengths finite")
    extended = independent_samples is not None or bounds is not None or hold is not None
    n_used = None
    if independent_samples is not None:
        if sigma_db is not None:
            raise ValueError("fit: give sigma_db or independent_samples, not both")
        n_used = np.broadcast_to(np.asarray(independent_samples, dtype=float).ravel(), y.shape)
        if sample_error == "book":
            sigma_db = measure.sigma_db_from_samples(n_used)
        elif sample_error == "gamma":
            sigma_db = measure.intensity_mean_db_std(n_used)
            y = y - measure.intensity_mean_db_bias(n_used)
        else:
            raise ValueError(f"fit: sample_error {sample_error!r}: use 'book' or 'gamma'")
    if roughness_model in bottom.BACKSCATTER_ROUGHNESS_MODELS:
        raise ValueError(f"fit: roughness_model {roughness_model!r} is not fitted here: the Mourad-Jackson (1989) model pairs its "
                         "roughness term with a slope-averaged volume term (volume_model 'mourad_jackson_1989') that this fit does "
                         "not use, and it is defined for backscatter only. Fit with "
                         f"{', '.join(repr(m) for m in bottom.ROUGHNESS_MODELS)} and evaluate the 1989 model at the fitted "
                         "parameters if wanted")
    # The opt-in williams_jackson_1998 (Williams and Jackson 1998) is a model for any geometry with the same six
    # parameters, so it is offered too; it is not part of the model spread (bottom.OPT_IN_ROUGHNESS_MODELS).
    fittable = bottom.ROUGHNESS_MODELS + bottom.OPT_IN_ROUGHNESS_MODELS
    if roughness_model not in fittable:
        import difflib
        near = difflib.get_close_matches(str(roughness_model), fittable, n=1)
        raise ValueError(f"fit: roughness_model {roughness_model!r}: unknown" + (f"; did you mean {near[0]!r}?" if near else
                         f" (known: {', '.join(fittable)})"))
    hold = dict(hold or {})
    _refuse_unknown(hold, bottom.SEDIMENT_KEYS, "fit: hold.")
    active = dict(BOUNDS)
    if bounds is not None:
        _refuse_unknown(bounds, bottom.SEDIMENT_KEYS, "fit: bounds.")
        for k, (b_lo, b_hi) in bounds.items():
            if not (np.isfinite(b_lo) and np.isfinite(b_hi) and b_lo < b_hi) or (k in LOG_KEYS and b_lo <= 0.0):
                raise ValueError(f"fit: bounds.{k}: need finite low < high" + (" and low > 0" if k in LOG_KEYS else ""))
            active[k] = (float(b_lo), float(b_hi))
    if roughness_model != "perturbation":
        # The structure function of a power-law spectrum exists only for 2 < gamma2 < 4 (bottom.structure_constant):
        # outside it every step would raise, and the starts would fail one by one without saying why.
        g_lo, g_hi = (hold["gamma2"],) * 2 if "gamma2" in hold else active["gamma2"]
        if not 2.0 < g_lo <= g_hi < 4.0:
            raise ValueError(f"fit: roughness_model {roughness_model!r} needs 2 < gamma2 < 4; gamma2 is "
                             + (f"held at {g_lo:g}" if "gamma2" in hold else f"bounded by {g_lo:g} to {g_hi:g}")
                             + " (use roughness_model 'perturbation')")
    weight = np.ones(len(y)) if sigma_db is None else 1.0 / np.broadcast_to(np.asarray(sigma_db, dtype=float).ravel(), y.shape)
    keys = tuple(k for k in SCATTER_KEYS + (GEO_KEYS if fit_geoacoustic else ()) if k not in hold)
    held = {} if fit_geoacoustic else {"rho": float(rho), "nu": float(nu), "delta": float(delta)}
    held.update({k: float(v) for k, v in hold.items()})
    if len(y) < len(keys):
        raise ValueError(f"fit: {len(y)} values cannot determine {len(keys)} parameters")
    if not keys:
        raise ValueError("fit: every parameter is held; nothing to fit")
    lo = _pack({k: active[k][0] for k in keys}, keys)
    hi = _pack({k: active[k][1] for k in keys}, keys)

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
    # (The absolute floor 1e-150: with an all-zero Jacobian, e.g. a small-slope roughness term 1e-51 of the
    # intensity, 1e-300 squared underflowed to 0 and the errors came out NaN, which no test flags.)
    floor = max(1e-6 * s.max(), 1e-150)
    cov = (vt.T / np.maximum(s, floor) ** 2) @ vt
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    share = _shares(params, g, f, sound_speed, roughness_model)
    freqs = tuple(float(v) for v in np.unique(f))
    unconstrained, notes, on_bound = [], [], {}

    def flag(key, why):
        if key in keys and key not in unconstrained:
            unconstrained.append(key)
        if why not in notes:
            notes.append(why)

    for j, k in enumerate(keys):
        limit = np.log10(MAX_FACTOR) if k in LOG_KEYS else MAX_STD.get(k, np.inf)
        if not std[j] <= limit:                                   # NaN included
            flag(k, f"{k}: 1 sigma is " + (f"a factor {10.0 ** min(std[j], 30.0):.3g}" if k in LOG_KEYS else f"{std[j]:.3g}")
                 + ", too wide to call the value known")
        if best.x[j] - lo[j] < 1e-6 * (hi[j] - lo[j]) or hi[j] - best.x[j] < 1e-6 * (hi[j] - lo[j]):
            where = "prior bound" if bounds is not None and k in bounds else "search limit"
            side = "low" if best.x[j] - lo[j] < hi[j] - best.x[j] else "high"
            value = active[k][0] if side == "low" else active[k][1]
            on_bound[k] = {"bound": float(value), "side": side, "kind": where}
            # (The search-limit note is as before; the prior-bound one says what it means.)
            flag(k, f"{k}: the fit ran to the {where} {value:g}"
                 + (": the value is set by the bound, not measured" if where == "prior bound" else ""))
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
    if extended:
        if bounds is not None:
            for j, k in enumerate(keys):
                if k in bounds and 2.0 * std[j] > hi[j] - lo[j]:
                    flag(k, f"{k}: its 2-sigma interval is wider than the bounds {active[k][0]:g} to {active[k][1]:g}: "
                            "the value is set by the bounds, not by the data")
            for k, v in held.items():
                if k in bounds and not active[k][0] <= v <= active[k][1]:
                    notes.append(f"{k} is held at {v:g}, outside the bounds {active[k][0]:g} to {active[k][1]:g}")
        spread = roughness_model_spread_db(params, np.degrees(g), f, sound_speed)
        if spread < sigma_used:
            notes.append(f"the roughness models ({', '.join(bottom.ROUGHNESS_MODELS)}) give the same strengths to within "
                         f"{spread:.2g} dB at these data points with the fitted parameters, less than the measurement error "
                         f"({sigma_used:.2f} dB): these data cannot tell the roughness models apart")
    return SedimentFit(params={k: params[k] for k in bottom.SEDIMENT_KEYS}, fitted=tuple(keys),
                       std={k: float(std[j]) for j, k in enumerate(keys)}, correlation=corr, unconstrained=unconstrained,
                       notes=notes, rms_residual_db=rms, sigma_db=sigma_used, n_points=len(y),
                       grazing_deg=(float(np.degrees(g.min())), float(np.degrees(g.max()))), frequencies=freqs,
                       roughness_share=(float(share.min()), float(share.max())), roughness_model=roughness_model,
                       sound_speed=float(sound_speed), residuals_db=resid, covariance=cov,
                       independent_samples=None if n_used is None else (float(n_used.min()), float(n_used.max())),
                       sample_error=None if n_used is None else sample_error,
                       bounds=None if bounds is None else {k: active[k] for k in bounds}, bounds_source=bounds_source,
                       held=hold or None, on_bound=on_bound)
