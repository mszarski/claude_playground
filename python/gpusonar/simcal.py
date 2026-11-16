"""Calibrating the simulator to real data (not to be confused with calibrate.py, which
estimates an array's channel errors).

Two kinds of real data, two methods.

Forward-looking sonar log (`calibrate_fls`). A real log's background is the sum of what
each boundary scatters back and of noise. The simulator's output is linear in each of
those: twice the scattering strength gives twice the power at every range. So each
contribution is simulated on its own, through the same imaging as the log, and the log's
range profile is fitted as a non-negative sum of the simulated profiles:

    observed(r) = sum_k a_k boundary_k(r) + b noise(r)

    10 log10(a_k)   how many dB boundary k's strength must move (`offset_db` in the scenario)
    10 log10(b)     how many dB the ambient noise level must move

With absolute levels (raw element data in ping records with the true gains, source level
and sensitivity in the sonar config) those are absolute corrections. With relative levels
(an unknown overall gain, as in most exported imagery) only the differences between them
mean anything; one term is held where the scenario has it (`anchor`) and the others are
moved relative to it. What a log cannot tell is reported, not guessed: a contribution that
never reaches a tenth of the profile anywhere is marked unconstrained.

SAS image (`calibrate_sas_image`). Exported SAS images are relative and usually flattened
in range, so the range profile says little. What survives is texture: the speckle of the
bare seabed tells how many scatterers share a resolution cell, and the depth of the
shadows tells how far the noise and sidelobe floor lies below the seabed.

Neither method touches objects in the scene: they set the background the objects sit in.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from . import _gpusonar as core
from . import recon
from .config import load_scenario_config


# --- forward-looking sonar ----------------------------------------------------------------------

@dataclass
class FlsObservation:
    """The background of a log: median power against range, over bearings and pings."""
    ranges: np.ndarray
    profile: np.ndarray
    bearings: np.ndarray
    levels: str = "absolute"                   # or "relative": an unknown gain multiplies everything
    pings: int = 0


def _profile(power: np.ndarray, ranges: np.ndarray, bins: int):
    """Median over bearings (robust to targets), then mean within `bins` range bins."""
    med = np.median(power, axis=1)
    n = len(ranges) // bins * bins
    return ranges[:n].reshape(bins, -1).mean(axis=1), med[:n].reshape(bins, -1).mean(axis=1)


def observe_records(records, ranges, bearings_rad, *, levels: str = "absolute", bins: int = 120,
                    window: str = "hann", backend=core.Backend.Host) -> FlsObservation:
    """Image each ping onto range x bearing and reduce it to a range profile."""
    ranges, bearings = np.asarray(ranges, dtype=float), np.asarray(bearings_rad, dtype=float)
    acc = None
    for rec in records:
        img = recon.fan_image([rec], ranges, bearings, interp="linear", window=window, backend=backend)
        r, p = _profile(np.abs(img.image) ** 2, ranges, bins)
        acc = p if acc is None else acc + p
    return FlsObservation(r, acc / len(records), bearings, levels, len(records))


def observe_frames(intensity, ranges, bearings_rad, *, levels: str = "relative", bins: int = 120) -> FlsObservation:
    """A log already beamformed by another system: intensity [frame, range, bearing] (power,
    not dB). Its processing gain is unknown, so treat it as relative unless it was made by
    this suite's own imaging."""
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim == 2:
        intensity = intensity[None]
    ranges = np.asarray(ranges, dtype=float)
    profiles = [_profile(frame, ranges, bins) for frame in intensity]
    return FlsObservation(profiles[0][0], np.mean([p for _, p in profiles], axis=0), np.asarray(bearings_rad, dtype=float),
                          levels, len(intensity))


@dataclass
class FlsCalibration:
    boundaries: list                           # per boundary: {boundary_z, model, offset_db, share, constrained}
    noise_offset_db: float | None
    noise_constrained: bool
    reference_noise_level_db: float            # the spectrum level the noise term was simulated at
    levels: str
    anchor: str | None                         # what was held fixed for a relative log
    fit_rms_db: float                          # RMS misfit of the fitted profile, dB
    ranges: np.ndarray = field(repr=False, default=None)
    observed_db: np.ndarray = field(repr=False, default=None)
    fitted_db: np.ndarray = field(repr=False, default=None)
    parts_db: dict = field(repr=False, default=None)

    def overrides(self) -> dict:
        """Scenario overrides that apply the calibration: `load_scenario_config(name, cal.overrides())`.
        Terms the log did not constrain are left as the scenario has them."""
        out = {"reverberation": [{k: v for k, v in b["item"].items()} for b in self.boundaries]}
        for o, b in zip(out["reverberation"], self.boundaries):
            if b["constrained"]:
                o["offset_db"] = float(b["item"].get("offset_db", 0.0) + b["offset_db"])
        if self.noise_constrained and self.noise_offset_db is not None:
            out["receiver"] = {"ambient_noise": {"level_db": float(self.reference_noise_level_db + self.noise_offset_db)}}
        return out

    def report(self) -> str:
        lines = [f"levels: {self.levels}" + (f" (held fixed: {self.anchor})" if self.anchor else ""),
                 f"fit: {self.fit_rms_db:.2f} dB RMS over {len(self.ranges)} range bins from {self.ranges[0]:.0f} to {self.ranges[-1]:.0f} m"]
        for b in self.boundaries:
            what = f"boundary at z = {b['boundary_z']:g} m ({b['model']})"
            if b["constrained"]:
                lines.append(f"{what}: move by {b['offset_db']:+.1f} dB (up to {100 * b['share']:.0f}% of the profile)")
            else:
                lines.append(f"{what}: not constrained by this log (at most {100 * b['share']:.0f}% of the profile)")
        if self.noise_constrained:
            lines.append(f"ambient noise: {self.reference_noise_level_db + self.noise_offset_db:.1f} dB re 1 uPa^2/Hz at the centre frequency"
                         + ("" if self.levels == "absolute" else " (relative to the held term)"))
        else:
            lines.append("ambient noise: not constrained by this log")
        return "\n".join(lines)


def _simulate_part(cfg: dict, ranges, bearings, bins, pings, backend, *, reverberation, noise_level_db):
    from .sim import run_scenario
    c = copy.deepcopy(cfg)
    c["scene"] = []
    c["reverberation"] = reverberation
    c["pings"] = dict(c.get("pings", {}), count=pings)
    rx = dict(c.get("receiver") or {})
    rx.pop("ambient_noise", None)
    rx["noise_rms"] = 0.0
    rx["adc"] = False
    if noise_level_db is not None:
        rx["ambient_noise"] = {"level_db": noise_level_db}
    c["receiver"] = rx
    records = run_scenario(c, backend=backend).records
    return observe_records(records, ranges, bearings, bins=bins, backend=backend).profile


def calibrate_fls(observation: FlsObservation, scenario, *, anchor: str = "noise", reference_noise_level_db: float = 50.0,
                  pings: int = 3, backend=core.Backend.Host) -> FlsCalibration:
    """Fit the scenario's boundaries and noise to an observed range profile.

    `scenario` names the sonar, the geometry (depth, attitude, water depth) and the
    boundaries of the log's setting: its `reverberation` list says which boundaries exist
    and with which model; its strengths are the starting point the offsets are relative to.
    `anchor` matters for a relative log only: "noise" keeps the scenario's noise and moves
    the boundaries, or "boundary:0" keeps the first boundary and moves the rest.
    """
    cfg = load_scenario_config(scenario) if isinstance(scenario, str) else copy.deepcopy(scenario)
    ranges_fine = None
    bins = len(observation.ranges)
    # The same fine range grid the observation was imaged on is not known for foreign frames;
    # a grid with eight samples per bin over the same span reduces to the same bins.
    step = (observation.ranges[-1] - observation.ranges[0]) / (bins - 1) / 8.0
    ranges_fine = observation.ranges[0] - 3.5 * step + step * np.arange(bins * 8)
    items = list(cfg.get("reverberation") or [])
    parts = {}
    for i, item in enumerate(items):
        parts[f"boundary:{i}"] = _simulate_part(cfg, ranges_fine, observation.bearings, bins, pings, backend,
                                                reverberation=[item], noise_level_db=None)
    parts["noise"] = _simulate_part(cfg, ranges_fine, observation.bearings, bins, pings, backend,
                                    reverberation=[], noise_level_db=reference_noise_level_db)
    names = list(parts)
    obs = np.asarray(observation.profile, dtype=float)
    ok = obs > 0
    A = np.stack([parts[n][ok] / obs[ok] for n in names], axis=1)      # relative error: every range counts the same
    coef, _ = nnls(A, np.ones(ok.sum()))
    fitted = sum(c * parts[n] for c, n in zip(coef, names))
    share = {n: float(np.max(c * parts[n][ok] / fitted[ok])) for c, n in zip(coef, names)}
    constrained = {n: share[n] >= 0.1 for n in names}
    gains_db = {n: (10 * np.log10(c) if c > 0 else -np.inf) for c, n in zip(coef, names)}
    held = None
    if observation.levels != "absolute":
        # An unknown gain multiplies the log: only differences between the terms are known.
        held = anchor if constrained.get(anchor, False) else next((n for n in names if constrained[n]), None)
        if held is None:
            raise ValueError("no term of the model reaches a tenth of the observed profile")
        # The held term keeps the scenario's value; for noise that is the scenario's own level.
        hold_at = 0.0
        if held == "noise":
            from .sim.simulator import _noise_spectrum
            spec = _noise_spectrum((cfg.get("receiver") or {}).get("ambient_noise"))
            if spec is None:
                raise ValueError("anchor 'noise' needs receiver.ambient_noise in the scenario")
            from .sonar import SonarModel
            w = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides")).config["transmit"]["waveform"]
            fc = 0.5 * (float(w["f_start"]) + float(w.get("f_stop", w["f_start"])))
            hold_at = float(spec(np.array([fc]))[0]) - reference_noise_level_db
        shift = hold_at - gains_db[held]
        gains_db = {n: g + shift for n, g in gains_db.items()}
    rms = float(np.sqrt(np.mean((10 * np.log10(fitted[ok] / obs[ok])) ** 2)))
    boundaries = [{"boundary_z": float(item["boundary_z"]), "model": item.get("model", "lambert"), "item": dict(item),
                   "offset_db": float(gains_db[f"boundary:{i}"]), "share": share[f"boundary:{i}"],
                   "constrained": bool(constrained[f"boundary:{i}"])} for i, item in enumerate(items)]
    with np.errstate(divide="ignore"):
        parts_db = {n: 10 * np.log10(np.maximum(c * parts[n], 1e-300)) for c, n in zip(coef, names)}
        return FlsCalibration(boundaries, float(gains_db["noise"]) if np.isfinite(gains_db["noise"]) else None,
                              bool(constrained["noise"]), reference_noise_level_db, observation.levels, held, rms,
                              observation.ranges, 10 * np.log10(np.maximum(obs, 1e-300)),
                              10 * np.log10(np.maximum(fitted, 1e-300)), parts_db)


# --- SAS images ------------------------------------------------------------------------------------

@dataclass
class SasCalibration:
    scintillation_index: float                 # variance of intensity over its mean squared, bare seabed
    cell_area_m2: float                        # correlation area of the speckle: one resolution cell
    scatterers_per_cell: float                 # inf when the speckle is fully developed
    density_per_m2: float                      # scatterers per square metre that give this texture
    shadow_below_seabed_db: float              # how far the darkest regions lie below the bare seabed
    bare_fraction: float                       # fraction of the image judged bare seabed
    blocks: int
    shadow_clipped: bool = False               # the darkest pixels sit at the image's floor: the true depth is at least this

    def overrides(self, seafloor_label: str | None = None, max_density: float = 5000.0) -> dict:
        """What to change in a scenario's `seafloor` item. Developed speckle needs no more than
        a few scatterers per cell; `max_density` caps what is asked of the simulator."""
        return {"seafloor": {"label": seafloor_label, "density": float(min(self.density_per_m2, max_density))},
                "shadow_below_seabed_db": self.shadow_below_seabed_db}

    def report(self) -> str:
        per_cell = "more than 20 (fully developed)" if self.scatterers_per_cell > 20 else f"{self.scatterers_per_cell:.2g}"
        return "\n".join([
            f"bare seabed: {100 * self.bare_fraction:.0f}% of {self.blocks} blocks",
            f"speckle: scintillation index {self.scintillation_index:.2f} (1.0 is fully developed)",
            f"resolution cell from the speckle's correlation: {self.cell_area_m2 * 1e4:.1f} cm^2",
            f"scatterers per cell: {per_cell}; density {self.density_per_m2:.3g} per m^2",
            f"shadows lie {self.shadow_below_seabed_db:.1f} dB below the bare seabed (noise and sidelobe floor)"
            + (": the image is clipped there, so at least that deep" if self.shadow_clipped else "")])


def scatterers_per_cell(scintillation_index: float) -> float:
    """For a Poisson number of scatterers with complex Gaussian amplitudes in a cell, the
    intensity has scintillation index 1 + 2 / mean count."""
    return float(np.inf) if scintillation_index <= 1.0 else 2.0 / (scintillation_index - 1.0)


def _correlation_area(block: np.ndarray, max_lag: int = 12) -> float:
    """Pixels in one speckle: the half-height widths of the intensity autocorrelation's
    central peak along each axis, as the area of the Gaussian with those widths."""
    b = block - block.mean()
    acf = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(b)) ** 2).real)
    acf /= acf.max()
    c = np.array(acf.shape) // 2

    def width(cut):
        k = 1
        while k < max_lag and cut[k] > 0.5:
            k += 1
        # Linear interpolation of the half-height crossing; both sides are the same.
        return 2.0 * (k - 1 + (cut[k - 1] - 0.5) / max(cut[k - 1] - cut[k], 1e-9))

    w0, w1 = width(acf[c[0]:c[0] + max_lag + 1, c[1]]), width(acf[c[0], c[1]:c[1] + max_lag + 1])
    return float(np.pi / (4.0 * np.log(2.0)) * w0 * w1)


def calibrate_sas_image(intensity, pixel_m, *, block: int = 48) -> SasCalibration:
    """Texture statistics of a SAS intensity image (power, linear; any overall gain).

    The image is cut into blocks. Bare seabed is taken to be the blocks whose mean lies
    within 1.5 dB of the median block mean and whose own statistics are typical (objects
    and shadow edges make a block's scintillation index jump). `pixel_m`: one number or
    (along track, across track).
    """
    img = np.asarray(intensity, dtype=float)
    px = (float(pixel_m), float(pixel_m)) if np.isscalar(pixel_m) else tuple(float(v) for v in pixel_m)
    n0, n1 = img.shape[0] // block, img.shape[1] // block
    tiles = img[:n0 * block, :n1 * block].reshape(n0, block, n1, block).swapaxes(1, 2).reshape(-1, block, block)
    tiles = tiles[np.all(tiles > 0, axis=(1, 2)) | (tiles.mean(axis=(1, 2)) > 0)]
    means = tiles.mean(axis=(1, 2))
    valid = means > 0
    tiles, means = tiles[valid], means[valid]
    med = np.median(means)
    level_ok = np.abs(10 * np.log10(means / med)) < 1.5
    si = tiles.var(axis=(1, 2)) / means**2
    si_med = np.median(si[level_ok])
    bare = level_ok & (si < 2.0 * si_med + 0.5)
    si_bare = float(np.median(si[bare]))
    area_px = float(np.median([_correlation_area(t) for t in tiles[bare][:64]]))
    cell = area_px * px[0] * px[1]
    lam = scatterers_per_cell(si_bare)
    # Shadows are smaller than the statistics blocks: look for them in 8 x 8 pixel means.
    m0, m1 = img.shape[0] // 8, img.shape[1] // 8
    small = img[:m0 * 8, :m1 * 8].reshape(m0, 8, m1, 8).mean(axis=(1, 3))
    shadow = float(10 * np.log10(max(np.percentile(small, 1.0), 1e-300) / np.median(means[bare])))
    clipped = bool(np.mean(img <= img.min() * (1.0 + 1e-9)) > 0.01)
    return SasCalibration(si_bare, cell, lam, (lam if np.isfinite(lam) else 20.0) / cell, shadow,
                          float(bare.mean()), int(len(tiles)), clipped)


def load_image(path) -> np.ndarray:
    """An intensity image from .npy (power), or from an 8- or 16-bit picture (PNG, TIFF),
    whose grey level is taken as amplitude. A picture that was log-compressed for display
    has lost its statistics; give the display's dB range to `from_display_db` instead."""
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(float)
    from PIL import Image
    a = np.asarray(Image.open(path).convert("F"), dtype=float)
    return a**2


def from_display_db(grey: np.ndarray, db_min: float, db_max: float, grey_max: float = 255.0) -> np.ndarray:
    """Undo a linear-in-dB grey scale: grey 0..grey_max maps to db_min..db_max. Returns power.
    Pixels at either end of the scale were clipped and are wrong by an unknown amount."""
    g = np.asarray(grey, dtype=float)
    return 10.0 ** ((db_min + (db_max - db_min) * g / grey_max) / 10.0)
