"""One arrival alone: the echo of one (outbound eigenray, return eigenray) pair of one target, synthesised by the
simulator's own kernel (docs/tasks/VISUALIZATION.md sections 4.4 and 14; package V3).

No simulator code changes. `capture_ping()` wraps, while it is active, `simulator.simulate_ping` and the name
`simulator.two_way_paths` (read only: each call goes on to the original with the same arguments and the same
provider results, so that the run and its record are what they would be without it). It keeps the first ping's
arguments, its random generator's state and the two eigenray sets the pairing was given. `single_pair_echoes()` then
calls `simulate_ping` again with those arguments (reverberation, sources and noise off; the same generator state, so
the same clock draws) and a pairing whose eigenray sets are masked to one outbound ray of one scatterer and one return
ray of the same scatterer: the "echoes" stage of that run is the pair's echo alone, from whichever engine the run used
(the masking is on the Python eigenray sets both engines' pairing reads). By linearity the pairs of a target sum to
that target's echoes (tests/python/test_scene_audio.py, both engines).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import tempfile

import numpy as np


class _Fixed:
    """A provider stand-in that returns the given eigenray sets in order (outbound, then return)."""

    def __init__(self, sets):
        self._sets = list(sets)

    def eigenrays(self, source, points):
        return self._sets.pop(0)


@contextlib.contextmanager
def capture_ping():
    """While active, keep the first simulated ping's call (`box["args"]`, `box["kwargs"]`, `box["rng_state"]`) and
    the outbound and return eigenray sets with the pairing's arguments (`box["out"]`, `box["back"]`, `box["pairing"]`).
    Yields the box."""
    from ..sim import simulator
    box: dict = {}
    sp, twp = simulator.simulate_ping, simulator.two_way_paths

    def simulate_ping(*args, **kwargs):
        first = "args" not in box
        if first:
            rng = kwargs.get("rng")
            box.update(args=args, kwargs=dict(kwargs), active=True,
                       rng_state=None if rng is None else copy.deepcopy(rng.bit_generator.state),
                       rng_kind=None if rng is None else type(rng.bit_generator))
        try:
            return sp(*args, **kwargs)
        finally:
            if first:
                box["active"] = False

    def two_way_paths(provider, tx_position, rx_reference, points, amplitudes, velocities=None, lambert=None,
                      normals=None, engine=None):
        if box.get("active") and "out" not in box:
            out = provider.eigenrays(tx_position, points)
            back = provider.eigenrays(rx_reference, points)
            box.update(out=out, back=back, pairing=dict(tx_position=np.array(tx_position, float),
                                                       rx_reference=np.array(rx_reference, float),
                                                       points=np.array(points, float), amplitudes=np.array(amplitudes),
                                                       velocities=None if velocities is None else np.array(velocities),
                                                       lambert=lambert, normals=normals))
            provider = _Fixed([out, back])
        return twp(provider, tx_position, rx_reference, points, amplitudes, velocities, lambert, normals, engine=engine)

    simulator.simulate_ping, simulator.two_way_paths = simulate_ping, two_way_paths
    try:
        yield box
    finally:
        simulator.simulate_ping, simulator.two_way_paths = sp, twp


def masked(eig, keep):
    """A copy of an EigenraySet in which only the (scatterer, ray) entries in `keep` (a bool [S, K] mask) are valid."""
    return dataclasses.replace(eig, valid=np.asarray(eig.valid, bool) & np.asarray(keep, bool))


def match_ray(eig, s: int, delay: float, tol: float = 1e-9) -> int | None:
    """The ray of scatterer `s` in `eig` with this delay (within `tol` s), or None."""
    d = np.where(eig.valid[s], np.abs(eig.delay[s] - delay), np.inf)
    k = int(np.argmin(d))
    return k if d[k] <= tol else None


def scatterer_of(box, position, tol: float = 1e-6) -> int | None:
    """The index of the scene point at `position` in the captured pairing (a trace target), or None."""
    pts = box["pairing"]["points"]
    d = np.linalg.norm(pts - np.asarray(position, float)[None, :], axis=1)
    i = int(np.argmin(d))
    return i if d[i] <= tol * max(1.0, float(np.linalg.norm(position))) else None


def echoes_with(box, out_keep, back_keep, engine=None) -> np.ndarray:
    """The "echoes" stage of the captured ping re-run with the eigenray sets masked to `out_keep` / `back_keep`
    ([S, K] bool), reverberation, sources and noise off, the generator in its captured state."""
    from .. import taps
    from ..sim import simulator
    if "out" not in box:
        raise ValueError("the captured ping had no echo pairing (no scene, or no transmit)")
    sets = [masked(box["out"], out_keep), masked(box["back"], back_keep)]
    pr = box["pairing"]
    twp = simulator.two_way_paths

    def fixed(provider, *args, engine=None, **kwargs):
        return twp(_Fixed(list(sets)), pr["tx_position"], pr["rx_reference"], pr["points"], pr["amplitudes"],
                   pr["velocities"], pr["lambert"], pr["normals"], engine=engine)
    kwargs = dict(box["kwargs"])
    kwargs.update(reverberation=None, sources=None, noise_rms=0.0, noise_spectrum=None, fitted_noise=None,
                  front_end_truth=None, adc=False)
    if box.get("rng_state") is not None:
        g = np.random.Generator(box["rng_kind"]())
        g.bit_generator.state = copy.deepcopy(box["rng_state"])
        kwargs["rng"] = g
    if engine is not None:
        kwargs["engine"] = engine
    simulator.two_way_paths = fixed
    try:
        with tempfile.TemporaryDirectory(prefix="gpusonar_pair_") as tmp:
            with taps.sim_capture(tmp):
                simulator.simulate_ping(*box["args"], **kwargs)
            return np.asarray(taps.read_sim_taps(tmp, 0)["echoes"])
    finally:
        simulator.two_way_paths = twp


def single_pair_echoes(box, s: int, k: int, j: int, engine=None) -> np.ndarray:
    """[C, N] echoes of the pair (outbound ray k, return ray j) of scene point s alone."""
    shape = box["out"].valid.shape, box["back"].valid.shape
    o, b = np.zeros(shape[0], bool), np.zeros(shape[1], bool)
    o[s, k] = True
    b[s, j] = True
    return echoes_with(box, o, b, engine)


def strongest_pairs(trace: dict, box, target: int = 0, n: int = 8) -> list[dict]:
    """The `n` strongest two-way pairs of trace target `target`, each mapped to the captured eigenray sets (scatterer
    index, outbound and return ray by their delays). Pairs that cannot be mapped are returned with `mapped: False`."""
    if not trace.get("targets") or target >= len(trace["targets"]):
        return []
    legs = trace["legs"][target]
    d_out = {r["k"]: r["delay_s"] for r in legs["out"]}
    d_back = {r["k"]: r["delay_s"] for r in legs["back"]}
    s = scatterer_of(box, trace["targets"][target]["position"])
    pairs = sorted([p for p in trace["pairs"] if p.get("target", 0) == target], key=lambda p: -p["level_db"])[:n]
    out = []
    for p in pairs:
        ko = d_out.get(p["out"])
        kb = d_back.get(p["back"])
        # The time-spread scenes' pairs are the wrapper's taps: their legs are not in `legs`; match the pair's
        # delay split by the captured sets instead.
        k = match_ray(box["out"], s, ko) if (s is not None and ko is not None) else None
        j = match_ray(box["back"], s, kb) if (s is not None and kb is not None) else None
        if (k is None or j is None) and s is not None:
            k, j = _split(box, s, p["delay_s"])
        out.append(dict(p, scatterer=s, k_captured=k, j_captured=j, mapped=k is not None and j is not None))
    return out


def _split(box, s, delay, tol=1e-9):
    """(k, j) of scatterer s whose outbound and return delays sum to `delay`, or (None, None)."""
    o, b = box["out"], box["back"]
    do = np.where(o.valid[s], o.delay[s], np.nan)
    db = np.where(b.valid[s], b.delay[s], np.nan)
    m = np.abs(do[:, None] + db[None, :] - delay)
    if not np.any(np.isfinite(m)):
        return None, None
    k, j = np.unravel_index(np.nanargmin(m), m.shape)
    return (int(k), int(j)) if m[k, j] <= tol else (None, None)
