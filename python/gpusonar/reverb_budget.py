"""The link budget's reverberation term (issue sonar-w2w3; D-198): what the boundaries scatter back at the delay of a
target, in the units of the sonar equation, so that signal excess is over the power sum of noise AND reverberation.

Why: `SonarEquation` and the coverage picture were noise-only. For `flank_patrol`'s small vehicle they gave about
30 dB where the simulated data give about 4 dB over the background: the bottom's reverberation (Lambert -27 dB) is
22 dB over the noise at 2.2 km (D-198).

What is computed. The scattering-function method of the simulator itself (sim/reverb.py `reverberation_power`; SST,
APL-UW TR 0702 Sec. 6; Jackson and Richardson 2007 (G.4), docs/references/jackson_richardson_2007/03, Sec. 2.9): the
boundary is cut into rings about the sonar, each pair of paths out to a ring and back adds
|a_out|^2 |a_back|^2 S(g_in, g_out) x ring area at its delay, per radian of azimuth, for a unit source. Then

    R(T) = Psi x sum_k P_k <|sinc(B (T - t_k))|^2>_bin k          (power re the source level)

the bins' power through the compressed pulse (the same |sinc|^2 of the band the coverage picture puts the target's
arrivals through, averaged over each bin: a bin wider than the resolution gives P_k / (B dt), one narrower its
value), times Psi, the beam's integral over azimuth re its axis (J&R (G.12)). For one direct path each way this is
J&R (G.11)-(G.13), monostatic: RL = SL + S_b - 40 log r - 2 alpha r + 10 log10(r Psi c tau / 2) with the pulse
length tau replaced by the compressed one, 1/B (for a CW pulse B = 1/T, so tau = T: (G.13) itself). The target's
echo after the same processing is SL - 2 TL + TS: both are compressed alike, so the matched filter's gain 10 log BT
cancels between them, and the array gain (against isotropic noise) is not applied to the reverberation; its beam is,
through Psi. The noise keeps its budget, NL - 10 log BT - AG, in the same units (`SonarEquation.noise_equivalent_db`).

The scattering strengths are the simulator's own: `sim.simulator.reverberation_item_strength` turns a scenario's
`reverberation:` item into its law (Lambert, Chapman-Harris, the fluid-sediment model, ...), and nothing here re-types a
law. The C++ coverage picture (libs/live coverage.hpp) takes the same laws as tables over grazing angle in x out
(`strength_tables`), and its rays in place of the eigenray provider.

Units: `level_db` returns dB re the source level (add SL for dB re 1 uPa^2 at the beamformer's output in the echo's
units). Not in it: the reverberation's statistics (non-Rayleigh, K) in the detection threshold, the Doppler that
separates a moving target from reverberation, the beam's vertical pattern and sidelobes beyond what `pattern` gives,
a boundary that changes with range.
"""

from __future__ import annotations

import numpy as np

GRAZING = np.radians(np.arange(0.0, 90.0 + 1e-9, 2.0))   # the tables' grid: 2 degrees (46 x 46 values a boundary)
SURFACE_MODELS = ("chapman_harris", "kraus_surface", "bubble_layer_surface")


def boundary_of(item: dict, surface_z: float = 0.0) -> str:
    """"bottom", "surface" or "volume": where a `reverberation:` item lies. `boundary:` when given; a surface law on the
    surface; a resolved item (with `boundary_z`) at the surface when its boundary_z is the surface's; else the bottom."""
    if item.get("model") == "volume":
        return "volume"
    if item.get("boundary") in ("surface", "bottom"):
        return str(item["boundary"])
    if item.get("model", "lambert") in SURFACE_MODELS:
        return "surface"
    if item.get("boundary_z") is not None and abs(float(item["boundary_z"]) - float(surface_z)) < 1e-9:
        return "surface"
    return "bottom"


def item_law(item: dict, frequency: float, sound_speed: float, provider=None):
    """(item, strength): the item as the simulator completes it and its scattering strength as a function of the
    incident and scattered grazing angles (rad; linear per unit area), offset included. The simulator's own dispatch
    (sim/simulator.py `reverberation_item_strength`): a Lambert item becomes mu sin(g_in) sin(g_out) with mu from
    `mu_db + offset_db`, exactly what the simulator hands sim/reverb.py."""
    from .sim.simulator import reverberation_item_strength
    item, strength, offset = reverberation_item_strength(dict(item), float(frequency), {"sound_speed": float(sound_speed)}, provider)
    if strength is None:
        mu = 10.0 ** ((float(item.get("mu_db", -29.0)) + offset) / 10.0)
        strength = lambda g_in, g_out, _mu=mu: _mu * np.sin(g_in) * np.sin(g_out)      # noqa: E731
    return item, strength


def strength_table(strength, grazing=GRAZING) -> np.ndarray:
    """[g_in, g_out] of a strength function on the grid (linear per unit area, finite, >= 0)."""
    g = np.asarray(grazing, dtype=float)
    t = np.asarray(strength(g[:, None] + 0.0 * g[None, :], g[None, :] + 0.0 * g[:, None]), dtype=float)
    return np.clip(np.where(np.isfinite(t), t, 0.0), 0.0, None)


def strength_tables(items, frequency: float, sound_speed: float, surface_z: float = 0.0, grazing=GRAZING) -> dict:
    """{bottom, surface}: the summed strength tables of `items` on each boundary (None where it has none). Volume
    items are refused: the coverage picture has no volume term."""
    out = {"bottom": None, "surface": None}
    for item in items or []:
        side = boundary_of(item, surface_z)
        if side == "volume":
            raise ValueError("reverberation: a volume item is not in the coverage picture's term (the link budget's Python reference has it)")
        _, law = item_law(item, frequency, sound_speed)
        t = strength_table(law, grazing)
        out[side] = t if out[side] is None else out[side] + t
    return out


def scenario_parts(cfg: dict) -> dict:
    """What a scenario's reverberation is made of, as run_scenario resolves it: {sonar, environment, items (the
    resolved `reverberation:` list), provider (the eigenray provider of its fidelity), boundaries ([(boundary_z,
    strength function)] of every item, volume layers at their centre depth), absorption_db_per_m, note}.
    `note` says why there is no term when the scenario gives no reverberation."""
    from . import acoustics
    from . import environment as site
    from .config import load_scenario_config
    from .sim.propagation import make_provider
    from .sonar import SonarModel
    cfg = load_scenario_config(cfg, None, expand=False) if not isinstance(cfg, dict) else cfg
    cfg = acoustics.resolve_sea_state(cfg)
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    env = site.resolve(cfg.get("environment"), strict=True)
    cfg = site.apply_to_scenario(cfg, env, sonar)
    provider = make_provider(cfg.get("fidelity", "free_field"), env["sound_speed"], cfg.get("propagation"))
    w = sonar.waveform_config
    fc = 0.5 * (float(w["f_start"]) + float(w.get("f_stop", w["f_start"])))
    items = list(cfg.get("reverberation") or [])
    boundaries = []
    for item in items:
        done, law = item_law(item, fc, env["sound_speed"], provider)
        boundaries.append((float(done["boundary_z"]), law))
    alpha = float(acoustics.absorption(fc, *site.water(env), cfg.get("absorption", site.DEFAULT_ABSORPTION)))
    return {"sonar": sonar, "environment": env, "items": items, "provider": provider, "boundaries": boundaries, "frequency": fc,
            "absorption_db_per_m": alpha, "config": cfg,
            "note": None if items else "not included: the scenario gives no reverberation (its `reverberation:` list is empty)"}


def sinc2_average(x, width: float, bandwidth: float):
    """<|sinc(B u)|^2> over u in [x - width/2, x + width/2] (s): Int sinc^2 = (Si(2 pi y) - sin^2(pi y) / (pi y)) / pi."""
    from scipy.special import sici
    x = np.asarray(x, dtype=float)
    if not width * bandwidth > 1e-3:
        return np.sinc(bandwidth * x) ** 2

    def F(y):
        a = np.pi * np.abs(y)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(a < 1e-9, np.abs(y), (sici(2.0 * a)[0] - np.sin(a) ** 2 / np.where(a > 0, a, 1.0)) / np.pi)
        return np.sign(y) * v
    return (F(bandwidth * (x + 0.5 * width)) - F(bandwidth * (x - 0.5 * width))) / (bandwidth * width)


def beam_integral(positions, frequency: float, sound_speed: float, steer_azimuth: float, weights=None, gain=None, n: int = 8192) -> float:
    """Psi (rad): Int |b(phi)|^2 dphi / |b(steer)|^2 over horizontal azimuth (J&R (G.12)) for hydrophones at `positions`
    ([N, 2 or 3], m; only x, y count), steered to `steer_azimuth` (rad from x towards y), with aperture `weights`
    (default none) and `gain` (a function of world unit directions [..., 3] giving the elements' and projector's
    power gain, as sim/reverb.element_power_pattern; default omnidirectional). One hydrophone: 2 pi x the pattern's."""
    p = np.asarray(positions, dtype=float).reshape(-1, np.asarray(positions).shape[-1])[:, :2]
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    k = 2.0 * np.pi * frequency / sound_speed
    extent = float(np.max(np.linalg.norm(p[:, None] - p[None], axis=2))) if len(p) > 1 else 0.0
    m = int(min(max(n, np.ceil(32.0 * k * extent)), 1 << 20))
    phi = 2.0 * np.pi * (np.arange(m) + 0.5) / m
    du = np.cos(phi) - np.cos(steer_azimuth)
    dv = np.sin(phi) - np.sin(steer_azimuth)
    af = np.abs(np.exp(1j * k * (np.outer(du, p[:, 0]) + np.outer(dv, p[:, 1]))) @ w) ** 2 / np.sum(w) ** 2
    if gain is not None:
        d = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
        g = np.asarray(gain(d), dtype=float)
        g0 = float(np.asarray(gain(np.array([[np.cos(steer_azimuth), np.sin(steer_azimuth), 0.0]])), dtype=float).ravel()[0])
        af = af * g / g0
    return float(np.sum(af) * 2.0 * np.pi / m)


def level_db(provider, source, boundaries, delays, bandwidth: float, psi: float, absorption_db_per_m: float = 0.0,
             step: float = 2e-3, tail: float | None = None):
    """Reverberation level re the source level (dB; -inf: none) at two-way `delays` (s), by the simulator's own integral
    (sim/reverb.py `reverberation_power`, monostatic: one eigenray set from `source` serves both legs, as there).
    `boundaries`: [(boundary_z, strength function), ...]; `step`: the time bins (s; rings 0.25 c step wide);
    `psi`: the beam's integral (`beam_integral`)."""
    from .sim.reverb import reverberation_power
    d = np.atleast_1d(np.asarray(delays, dtype=float))
    tail = 8.0 / bandwidth if tail is None else float(tail)
    t_max = float(np.nanmax(d)) + tail
    times = np.arange(0.5 * step, t_max + step, step)
    power = np.zeros(len(times))
    for z, law in boundaries:
        power = power + reverberation_power(provider, source, float(z), times, 0.0, absorption_db_per_m, strength=law)
    out = np.full(d.shape, -np.inf)
    for i, t in enumerate(d):
        if not np.isfinite(t):
            continue
        near = np.abs(times - t) <= tail + step
        s = float(np.sum(power[near] * sinc2_average(t - times[near], step, bandwidth)))
        out[i] = 10.0 * np.log10(psi * s) if s > 0.0 else -np.inf
    return out


def at_ranges(provider, source, boundaries, bandwidth: float, psi: float, sound_speed: float, absorption_db_per_m: float = 0.0,
              ranges=None, step: float = 2e-3):
    """A function of slant range (m) giving the reverberation level re the source level at the delay 2 r / c, made on
    `ranges` once and interpolated (what `SonarEquation(reverberation_db=...)` takes)."""
    r = np.asarray(ranges, dtype=float)
    lv = level_db(provider, source, boundaries, 2.0 * r / sound_speed, bandwidth, psi, absorption_db_per_m, step)

    def reverberation_db(range_m):
        return np.interp(np.asarray(range_m, dtype=float), r, lv)
    reverberation_db.ranges, reverberation_db.levels = r, lv
    return reverberation_db
