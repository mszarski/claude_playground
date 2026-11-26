#!/usr/bin/env python3
"""One-way transmission loss at each eigenray scenario's target: what the provider gives, and what it would give
under the alternatives measured for issue sonar-6c8 (docs/tasks/PROPAGATION_3D.md, section 2). Opt-in, diagnostic:
it changes nothing, writes nothing but a few small Bellhop files in a temporary directory, and is not run by tests.

    GPUSONAR_BELLHOP=.../bellhop.exe tools/eigenray_tl_check.py [--scenarios a,b] [--tmp DIR]

For each scenario at the `eigenray` tier (default: every one in configs/scenarios) the provider is built as
sim/simulator.py builds it (`make_provider` on the scenario's resolved `propagation`), from the platform position to
the first target point (the middle of a multi-point target). The provider looks a point up at its nearest grid node,
so Bellhop is run for that node only (and the grid's last range, which keeps Bellhop's automatic beam count and its
box as in the full run): a few receivers, a file of kilobytes. Columns: incoherent (sum of |a|^2) one-way loss, dB
re 1 m, at that one grid node (a point value: it carries the scatter of a single receiver, not a range average):

    shipped        the provider as the scenario uses it (its `max_arrivals`, or the 20 dB floor and cap of D-105)
    all arr        the same Bellhop arrivals, every one kept (no floor, no cap, no `max_arrivals`)
    TL run         Bellhop's own incoherent TL run ('I', same beams, same receiver) instead of summed arrivals
    and, for a scenario with a bathymetry profile, the same with the profile written as Bellhop's curvilinear 'C'
    instead of the piecewise-linear 'L' the provider writes ('C' columns): over relief, 'L' makes Bellhop's
    arrivals and coherent field louder than a parabolic equation (measured, sonar-6c8).

The wind-speed loss the provider applies to surface bounces is in `shipped` and `all arr` (as the provider applies
it) and not in `TL run`; the scenarios at this tier give the eigenrays a flat surface (wind_speed 0) except where
their file says otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def eigenray_scenarios():
    out = []
    for path in sorted((ROOT / "configs" / "scenarios").glob("*.yaml")):
        text = path.read_text()
        if any(line.split("#")[0].strip() == "fidelity: eigenray" for line in text.splitlines()):
            out.append(path.stem)
    return out


def build(name):
    """(provider, source position, target point) as sim/simulator.py run_scenario builds them."""
    import numpy as np
    from gpusonar import environment as site
    from gpusonar.config import load_scenario_config
    from gpusonar.sim.propagation import make_provider
    from gpusonar.sonar import SonarModel
    from gpusonar import acoustics
    cfg = acoustics.resolve_sea_state(load_scenario_config(name, None, expand=False))
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    env = site.resolve(cfg.get("environment"), strict=True)
    cfg = site.apply_to_scenario(cfg, env, sonar)
    provider = make_provider(cfg.get("fidelity"), env["sound_speed"], cfg.get("propagation"))
    item = next(i for i in cfg["scene"] if i.get("positions"))
    pts = np.asarray(item["positions"], dtype=float)
    return provider, np.asarray(cfg["platform"]["position"], dtype=float), pts[len(pts) // 2]


@contextlib.contextmanager
def curvilinear():
    """Write the provider's bathymetry file as Bellhop's curvilinear 'C' instead of 'L' (the points are the same)."""
    from gpusonar.sim import bellhop
    original = bellhop.write_env

    def write_env(path, *args, **kwargs):
        original(path, *args, **kwargs)
        bty = os.path.splitext(path)[0] + ".bty"
        if os.path.exists(bty):
            with open(bty) as f:
                rest = f.read().split("\n", 1)[1]
            with open(bty, "w") as f:
                f.write("'C'\n" + rest)
    bellhop.write_env = write_env
    try:
        yield
    finally:
        bellhop.write_env = original


def at_node(provider, source, point):
    """A copy of `provider` whose grid is the node `point` uses (and the last range): the same lookup, few receivers."""
    import numpy as np
    p = copy.copy(provider)
    rng = float(np.hypot(*(point[:2] - source[:2])))
    r = provider.ranges[int(np.clip(np.rint((rng - provider.ranges[0]) / (provider.ranges[1] - provider.ranges[0])), 0, len(provider.ranges) - 1))]
    dz = provider.depths[1] - provider.depths[0]
    z = provider.depths[int(np.clip(np.rint((point[2] - provider.depths[0]) / dz), 0, len(provider.depths) - 1))]
    p.ranges = np.array([r, provider.ranges[-1]]) if provider.ranges[-1] > r else np.array([r, r + 1.0])
    p.depths = np.array([z, z + dz]) if z + dz < provider.env.depth else np.array([z - dz, z])
    p._cache, p.arrivals_used, p.truncation, p.dropped_energy = {}, None, None, 0.0
    return p


def losses(provider, source, point, tmp):
    import numpy as np
    from gpusonar.sim import bellhop
    out = {}
    es = provider.eigenrays(source, point[None, :])
    a = np.where(es.valid, es.amplitude, 0.0)[0]
    out["shipped"] = -10 * np.log10(max(np.sum(np.abs(a) ** 2), 1e-300))
    every = copy.copy(provider)
    every.max_arrivals, every.arrival_floor_db, every.max_arrivals_cap, every.arrivals_used = None, 1e6, 10 ** 6, None
    ea = every.eigenrays(source, point[None, :])
    out["all arr"] = -10 * np.log10(max(np.sum(np.abs(np.where(ea.valid, ea.amplitude, 0.0)) ** 2), 1e-300))
    out["n arr"] = int(ea.valid.sum())
    run = os.path.join(tmp, "tl.env")
    bellhop.write_env(run, provider.env, provider.frequency, round(float(source[2]), 2), provider.depths, provider.ranges, "I",
                      provider.n_beams, provider.angles)
    subprocess.run([bellhop.find_bellhop(), "tl"], cwd=tmp, capture_output=True, check=True)
    out["TL run"] = -20 * np.log10(max(abs(read_shade(os.path.join(tmp, "tl.shd"))[0, 0]), 1e-30))
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    return out


def read_shade(path):
    """Pressure [n_rz, n_rr] of a Bellhop shade file (one source, one frequency, one bearing)."""
    import numpy as np
    raw = open(path, "rb").read()
    recl = 4 * int(np.frombuffer(raw, dtype="<i4", count=1)[0])
    rec = lambda i: raw[i * recl:(i + 1) * recl]                     # noqa: E731
    counts = np.frombuffer(rec(2), dtype="<i4", count=7)
    return np.stack([np.frombuffer(rec(10 + i), dtype="<c8", count=int(counts[6])) for i in range(int(counts[5]))])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", help="comma-separated scenario names (default: every eigenray scenario)")
    ap.add_argument("--tmp", help="directory for the temporary Bellhop files (default: the system's)")
    args = ap.parse_args(argv)
    import warnings
    import numpy as np
    from gpusonar.sim import find_bellhop
    if find_bellhop() is None:
        sys.exit("Bellhop is not installed: set GPUSONAR_BELLHOP")
    names = args.scenarios.split(",") if args.scenarios else eigenray_scenarios()
    head = f"{'scenario':32s} {'range':>7s} {'depth':>6s} {'n':>4s} {'shipped':>8s} {'all arr':>8s} {'TL run':>8s}"
    print(head + "   | with 'C' bed: shipped, all arr, TL run")
    for name in names:
        provider, source, point = build(name)
        rng = float(np.hypot(*(point[:2] - source[:2])))
        with tempfile.TemporaryDirectory(prefix="gpusonar_tlcheck_", dir=args.tmp) as tmp, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = at_node(provider, source, point)
            base = losses(p, source, point, tmp)
            line = (f"{name:32s} {rng:7.0f} {point[2]:6.1f} {base['n arr']:4d} {base['shipped']:8.2f} {base['all arr']:8.2f} "
                    f"{base['TL run']:8.2f}")
            if provider.env.bathymetry is not None:
                with curvilinear():
                    c = losses(at_node(provider, source, point), source, point, tmp)
                line += f"   | {c['shipped']:8.2f} {c['all arr']:8.2f} {c['TL run']:8.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
