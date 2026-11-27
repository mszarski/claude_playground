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

Which pairs (`strongest_pairs`): ranked by the synthesis's own amplitude, not the trace's level. While active,
`capture_ping()` also keeps the paths the synthesis kernel is handed for the first ping (`box["kernel"]`: q, p and the
amplitude after Lambert's law with the patch's normal, occlusion and the beam floor; read only). A pair's synthesis
level is that amplitude over the spherical spreading of its apparent path, 1 / (|q - tx| |p - rx|) (absorption left
out; ours); a pair the kernel was not handed (a face turned away, an occluded or beam-floored path) has none and is not
listed, so no listed arrival is silent. Without a kernel capture the trace's level ranks them, as before.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import tempfile

import numpy as np


class _NoBasin:
    """The run's provider without a basin's late tail (sim/basin.py), for the single-arrival runs: the tail is added
    after the echoes stage, so leaving it out changes nothing in the echoes and saves its synthesis."""

    def __init__(self, provider):
        self._p = provider

    basin = None

    def __getattr__(self, name):
        return getattr(self._p, name)


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
    from .. import _gpusonar as core
    from ..sim import simulator
    from ..sim import waveform as sampled
    box: dict = {}
    sp, twp = simulator.simulate_ping, simulator.two_way_paths
    syn, syn_s = core.synthesize_into, sampled.synthesize

    def keep_kernel(record, q, p, amp):
        if box.get("active") and "kernel" not in box:
            box["kernel"] = dict(q=np.array(q, float), p=np.array(p, float), amplitude=np.array(amp))

    def synthesize_into(record, q, p, amp, *args, **kwargs):
        keep_kernel(record, q, p, amp)
        return syn(record, q, p, amp, *args, **kwargs)

    def synthesize(record, waveform, q, p, amp, *args, **kwargs):
        keep_kernel(record, q, p, amp)
        return syn_s(record, waveform, q, p, amp, *args, **kwargs)

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
    core.synthesize_into, sampled.synthesize = synthesize_into, synthesize
    try:
        yield box
    finally:
        simulator.simulate_ping, simulator.two_way_paths = sp, twp
        core.synthesize_into, sampled.synthesize = syn, syn_s


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
    if getattr(kwargs.get("provider"), "basin", None) is not None:
        kwargs["provider"] = _NoBasin(kwargs["provider"])
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


def synthesis_level_db(box, s: int, k: int, j: int, rel: float = 1e-7) -> float | None:
    """dB of the amplitude the synthesis kernel was handed for the pair (outbound ray k, return ray j) of scene point s,
    over the spherical spreading of its apparent path (module docstring); None when the kernel was not handed it (or
    no kernel was captured)."""
    kern = box.get("kernel")
    if kern is None or not len(kern["q"]):
        return None
    pr = box["pairing"]
    if "_apparent" not in box:
        box["_apparent"] = (box["out"].apparent_positions(pr["tx_position"]), box["back"].apparent_positions(pr["rx_reference"]))
    q = box["_apparent"][0][s, k]
    p = box["_apparent"][1][s, j]
    scale = max(1.0, float(np.linalg.norm(q)), float(np.linalg.norm(p)))
    hit = np.flatnonzero((np.abs(kern["q"] - q).max(axis=1) <= rel * scale) & (np.abs(kern["p"] - p).max(axis=1) <= rel * scale))
    if not len(hit):
        return None
    a = float(np.abs(np.sum(kern["amplitude"][hit])))
    if a <= 0.0:
        return None
    spread = max(float(np.linalg.norm(q - pr["tx_position"])) * float(np.linalg.norm(p - pr["rx_reference"])), 1e-12)
    return 20.0 * float(np.log10(a / spread))


def strongest_pairs(trace: dict, box, target: int | None = 0, n: int = 8) -> list[dict]:
    """The `n` strongest two-way pairs of trace target `target` (None: of every target, ranked together), each mapped to the captured eigenray sets (scatterer
    index, outbound and return ray by their delays). With the kernel's paths captured (`capture_ping`) every pair of the
    target is mapped and ranked by its synthesis level (`synthesis_level_db`; `ranked_by` "synthesis"), and a pair the
    kernel was not handed is left out, so fewer than `n` may come back; otherwise the trace's `level_db` ranks them
    (`ranked_by` "trace") and pairs that cannot be mapped are returned with `mapped: False`."""
    if target is None:
        # Every target's pairs, ranked together (by the synthesis when the kernel was captured).
        allp = [q for t in range(len(trace.get("targets") or [])) for q in strongest_pairs(trace, box, t, n)]
        key = (lambda q: -q["synthesis_level_db"]) if box.get("kernel") is not None else (lambda q: -q["level_db"])
        return sorted(allp, key=key)[:n]
    if not trace.get("targets") or target >= len(trace["targets"]):
        return []
    legs = trace["legs"][target]
    d_out = {r["k"]: r["delay_s"] for r in legs["out"]}
    d_back = {r["k"]: r["delay_s"] for r in legs["back"]}
    s = scatterer_of(box, trace["targets"][target]["position"])
    by_synthesis = box.get("kernel") is not None
    pairs = sorted([p for p in trace["pairs"] if p.get("target", 0) == target], key=lambda p: -p["level_db"])
    if not by_synthesis:
        pairs = pairs[:n]
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
        mapped = k is not None and j is not None
        if not by_synthesis:
            out.append(dict(p, scatterer=s, k_captured=k, j_captured=j, mapped=mapped, ranked_by="trace", target=target))
            continue
        lvl = synthesis_level_db(box, s, k, j) if mapped else None
        if lvl is not None:
            out.append(dict(p, scatterer=s, k_captured=k, j_captured=j, mapped=True, ranked_by="synthesis",
                            synthesis_level_db=lvl, target=target))
    if by_synthesis:
        out = sorted(out, key=lambda q: -q["synthesis_level_db"])[:n]
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
