#!/usr/bin/env python3
"""The threshold shift the live chain's Doppler bank needs (D4; libs/live doppler.hpp bank_threshold_db), measured.

    tools/doppler_bank_threshold.py [--sonar long_range|flank] [--pings 60] [--max-per-beam 33] [--budgets 3,1,0.3]
                                    [--seed 7000] [--json FILE]
    tools/doppler_bank_threshold.py --reverb flank_patrol [--pings 6] [--budgets 3,1] [--json FILE]

The bank keeps, per map sample, the largest power over 2K + 1 range-rate hypotheses one tolerance (c / (2 B T)) apart.
That maximum is not the statistic the multiscale detector's thresholds were planned for: its background (the mean of
training cells that are maxima too) is higher, and its tail is another. Whether the detector then gives more or fewer
false detections than asked, and how far every threshold must move to give the same number as the zero hypothesis
alone, is counted here, not assumed:

- noise-only pings (white noise on every hydrophone, nothing in the water; the platform still) of the long-range array
  (configs/sonars/popoto_long_range, 800 m of window at 8 kHz baseband, the map of tools/detector_search_factor.py) or
  of the flank array (flank_32, 16 kHz baseband), each through the Python chain's own map at every hypothesis (the bank
  with --max-per-beam hypotheses, its maps caught as the chain folds them);
- per ping and per count N = 1, 3, 5, ..., --max-per-beam: the maximum over the N hypotheses about the zero one, the
  chain's cell means, and the C++ detector (core.live_multiscale_detect) at loosened budgets (--budgets false detections
  a ping: at the default 1e-3 a few hundred pings give no count), its thresholds and the chain's min_snr_db floor moved
  by an offset on a grid; the detections are counted;
- the shift for N at a budget is the offset at which N's count equals the zero hypothesis' count at offset 0 (linear
  interpolation of log counts between grid points), with the counts and their Poisson errors.

`--reverb SCENE`: the same on reverberation-only pings of a scene (its targets removed, its own seed, platform as the
scene has it), the bank's shift taken from the code (bank_threshold_db): the counts with and without the bank at the
same budget, to see the false-alarm rate held where the background is reverberation (a few pings: coarse).

Output: one line per (N, budget) and a JSON file. The fit in libs/live doppler.hpp (kBankThresholdSlopeDb) is a straight
line in ln N through the long-range array's shifts at the budgets measured; run it again and compare before changing the
map, the detector or the hypothesis spacing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import core, detect, doppler, live  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402
from gpusonar.sim import Scene, make_pose, simulate_ping  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

SONARS = {"long_range": ("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1700.0]}}}),
          "flank": ("flank_32", {"receive": {"kind": "baseband", "sample_rate": 16000.0, "window": {"range": [900.0, 1700.0]}}})}
CHAIN = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "cell_statistic": "multiscale"}
NOISE_RMS = 2e-4
OFFSETS = np.round(np.arange(-4.0, 0.51, 0.25), 3)


def say(text: str) -> None:
    print(text, flush=True)


def noise_ping(sonar, seed: int, sequence: int = 0):
    """One ping with nothing in the water: white noise on every hydrophone (the target is 300 dB down)."""
    scene = Scene()
    scene.add_points([[0.0, 1300.0, 0.0]], target_strength_db=-300.0)
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=NOISE_RMS, rng=np.random.default_rng(seed), sequence=sequence,
                         environment={"sound_speed": 1500.0}, adc=False, absorption_model="none")


def caught_maps(cfg: dict, records):
    """For each record: the chain's map of every hypothesis {h: fine}, and the detector's arguments (fine_ranges,
    cell_ranges, bearings, guard, train, k, n) as the chain hands them over."""
    folds, seen = {}, []
    plain_fold, plain_detect = doppler.fold, detect.multiscale

    def spy_fold(p, h, fine, best, which):
        folds[h] = np.array(fine, dtype=np.float32)
        return plain_fold(p, h, fine, best, which)

    def spy_detect(*args):
        seen.append(args)
        return plain_detect(*args)

    doppler.fold, detect.multiscale = spy_fold, spy_detect
    try:
        with LiveChain(cfg) as chain:
            for rec in records:
                folds.clear()
                chain.process(rec)
                fine, cell_mean, fine_ranges, cell_ranges, bearings, guard, train, plan = seen.pop()
                yield dict(folds), {"fine_ranges": np.asarray(fine_ranges), "cell_ranges": np.asarray(cell_ranges), "bearings": np.asarray(bearings),
                                    "guard": int(guard), "train": int(train), "k": int(plan["samples_per_cell"]), "n": fine.shape[1],
                                    "zero": int(chain._bank["zero"]), "first": np.asarray(chain._bank["first"]), "plan": plan}
    finally:
        doppler.fold, detect.multiscale = plain_fold, plain_detect


def max_map(folds: dict, zero: int, K: int, first: np.ndarray, per_beam: int) -> np.ndarray:
    """The maximum over the 2K + 1 hypotheses about each beam's centre (its own first + (per_beam - 1) / 2)."""
    centre = first + (per_beam - 1) // 2
    out = None
    for h in sorted(folds):
        use = np.abs(h - centre) <= K
        if not use.any():
            continue
        f = np.where(use[:, None], folds[h], -1.0).astype(np.float32)
        out = f if out is None else np.maximum(out, f)
    return out


def count(fine, geo, plan, offset_db: float, min_snr_db: float) -> int:
    """Detections of the C++ detector on this map with every threshold and the min_snr floor moved by offset_db."""
    k, n = geo["k"], geo["n"]
    n_cells = n // k
    f = np.ascontiguousarray(fine[:, :n_cells * k], dtype=np.float32)
    cell_mean = f.reshape(f.shape[0], n_cells, k).mean(axis=2).astype(np.float32)
    scale = 10.0 ** (offset_db / 10.0)
    p = dict(plan, scales=[dict(s, factor=float(s["factor"]) * scale) for s in plan["scales"]])
    got = core.live_multiscale_detect(f, cell_mean, list(geo["fine_ranges"][:n_cells * k]), list(geo["cell_ranges"][:n_cells]), list(geo["bearings"]),
                                      geo["guard"], geo["train"], p)
    return sum(1 for d in got["detections"] if d["snr_db"] > min_snr_db + offset_db)


def crossing(offsets, counts, target: float) -> float:
    """The offset where the (falling) count reaches `target`: linear in log count between grid points."""
    c = np.maximum(np.asarray(counts, dtype=float), 0.5)
    t = max(target, 0.5)
    for i in range(len(offsets) - 1):
        if c[i] >= t >= c[i + 1]:
            if c[i] == c[i + 1]:
                return float(offsets[i])
            return float(offsets[i] + (offsets[i + 1] - offsets[i]) * (np.log(c[i]) - np.log(t)) / (np.log(c[i]) - np.log(c[i + 1])))
    return float("nan")


def plans_for(template, cfg: dict, budgets) -> dict:
    return {b: live.detector_plan(template, dict(cfg, cfar={"false_alarms_per_ping": b}), 1) for b in budgets}


def measure_noise(name: str, pings: int, max_per_beam: int, budgets, seed: int, log=say) -> dict:
    sonar_name, overrides = SONARS[name]
    sonar = SonarModel.load(sonar_name, overrides)
    template = sonar.make_record(core.Pose(), environment={"sound_speed": 1500.0})
    cfg = dict(CHAIN, engine="python", doppler={"mode": "sweep_bank", "hypotheses": max_per_beam, "threshold_db": 0.0})
    plans = plans_for(template, dict(CHAIN), budgets)
    Ks = [K for K in range(0, (max_per_beam - 1) // 2 + 1) if K in (0, 1, 2, 4, 8, 16) or K == (max_per_beam - 1) // 2]
    counts = {(K, b): np.zeros(len(OFFSETS), dtype=int) for K in Ks for b in budgets}
    level = {K: 0.0 for K in Ks}                 # the mean of the map, summed over pings: the background's rise
    floor = float(live.DEFAULTS["min_snr_db"])
    t0 = time.time()
    records = (noise_ping(sonar, seed + i, i) for i in range(pings))
    for i, (folds, geo) in enumerate(caught_maps(cfg, records)):
        for K in Ks:
            fine = max_map(folds, geo["zero"], K, geo["first"], max_per_beam)
            level[K] += float(np.mean(fine[:, :geo["n"]], dtype=np.float64))
            for b in budgets:
                offs = OFFSETS if K > 0 else [0.0]
                for j, o in enumerate(offs):
                    counts[(K, b)][j if K > 0 else int(np.argmin(np.abs(OFFSETS)))] += count(fine, geo, plans[b], float(o), floor)
        if (i + 1) % 10 == 0:
            log(f"{name}: {i + 1} pings in {time.time() - t0:.0f} s")
    out = {"sonar": sonar_name, "pings": pings, "max_per_beam": max_per_beam, "seed": seed, "offsets_db": OFFSETS.tolist(),
           "step_m_s": None, "by_count": {}}
    z0 = int(np.argmin(np.abs(OFFSETS)))
    for K in Ks:
        N = 2 * K + 1
        entry = {}
        for b in budgets:
            ref = int(counts[(0, b)][z0])
            c = counts[(K, b)]
            shift = 0.0 if K == 0 else crossing(OFFSETS, c, ref)
            at0 = int(c[z0])
            entry[str(b)] = {"zero_hypothesis_count": ref, "count_at_0_db": at0, "counts": c.tolist(), "shift_db": shift,
                             "ratio_at_0_db": at0 / max(ref, 1)}
            log(f"{name}: N {N:3d} budget {b:5.2f}: zero hypothesis {ref:5d} ({ref / (b * pings):.2f} of the budget), the maximum at 0 dB {at0:5d} "
                f"({at0 / max(ref, 1):.2f} of it); shift for the same count {shift:+.3f} dB")
        entry["background_rise_db"] = float(10.0 * np.log10(level[K] / level[0]))
        log(f"{name}: N {N:3d}: the map's mean over the zero hypothesis' {entry['background_rise_db']:+.3f} dB")
        out["by_count"][str(N)] = entry
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sonar", default="long_range", choices=sorted(SONARS))
    ap.add_argument("--pings", type=int, default=60)
    ap.add_argument("--max-per-beam", type=int, default=33)
    ap.add_argument("--budgets", default="3,0.3")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    budgets = [float(v) for v in a.budgets.split(",")]
    res = measure_noise(a.sonar, a.pings, a.max_per_beam, budgets, a.seed)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
