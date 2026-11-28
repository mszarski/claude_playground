#!/usr/bin/env python3
"""The threshold shift the live chain's Doppler bank needs (D4; libs/live doppler.hpp bank_threshold_db), measured.

    tools/doppler_bank_threshold.py [--sonar long_range|flank] [--pings 60] [--max-per-beam 33] [--budgets 3,1,0.3]
                                    [--seed 7000] [--json FILE]
    tools/doppler_bank_threshold.py --reverb flank_patrol [--pings 6] [--budgets 3,1] [--json FILE]
    tools/doppler_bank_threshold.py --at-default [--sonar long_range|flank] [--combines off,max,cfar] [--max-rates 3]
                                    [--pings 3000] [--budget 1e-3] [--backend auto] [--workers 4] [--max-seconds S]
                                    [--seed 30000] [--json FILE]

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

`--chain cfar` (D3): the live chain itself (C++ engine), without the bank and with `combine: cfar` (each (beam, hypothesis)
normalised by its own background, the thresholds planned for the budget divided by the hypotheses per beam: the union
bound), on the same noise-only pings at each budget and each --max-rates value: the counts side by side, the ratio, the
three-sigma Poisson interval of the ratio, and the single-sample threshold of each plan (the correction in dB). With
--reverb SCENE the same on the scene's reverberation-only pings (its targets removed, its own seed + 1000).

`--at-default` (for the GPU job, tools/gpu_job.sh GPUSONAR_JOB=doppler): the live chain itself (C++ engine, the map and
the bank's kernels on `--backend`) at ONE budget, the default (1e-3 false detections a ping, as shipped) unless --budget,
on the SAME noise-only pings for every combiner (white noise drawn as tools/detector_search_factor.py --at-default draws
it, on --workers threads; ping i the stream SeedSequence(seed).spawn(pings)[i]): without the bank (`off`), with D4's
maximum (`max`, the thresholds not moved) and with D3's `cfar` (the budget divided by the hypotheses a beam), each at
each --max-rates value. As in that tool the chain's 9 dB floor on what it reports (`min_snr_db`) is set to 0, so that the
count is the detector's own, and the detections above the floor are counted beside it. Per row: the count, the count a
ping over the budget, its exact 95 % Poisson interval, the single-sample threshold, the time a ping. A row stops at
--max-seconds (said, with the pings done). The JSON is written after each row and every 500 pings.
How many pings: at the budget, n pings expect n x 1e-3 false detections; a zero count bounds the rate under 3.7 / n
(exact two-sided 95 %), and the 95 % interval of a count at the budget spans 0.21 to 2.9 of it at 3000 pings (3
expected), 0.48 to 1.8 at 10 000, 0.71 to 1.4 at 40 000. On the shared 4-core CPU a ping took 0.86 s without the bank and
14.3 s with it at 3 m/s (17 a beam, max or cfar).

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

# D-256: popoto_long_range sends a 2 s hyperbolic sweep since, which the bank refuses: this tool's long-range sonar states the
# 1 s linear sweep it was measured with (and the bank needs).
SONARS = {"long_range": ("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1700.0]}},
                                               "transmit": {"waveform": {"type": "lfm", "source": None, "duration": 1.0}}}),
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


def chain_counts(records, budgets, max_rates, combine: str, bearings=None, log=say) -> dict:
    """Detections of the live chain (C++ engine) on `records` without the bank and with it (`combine`), per budget and
    largest rate; the single-sample threshold of each plan."""
    base = dict(CHAIN, engine="cpp") if bearings is None else {"engine": "cpp", "backend": "host"}
    out = {}
    for b in budgets:
        for rate in [None] + list(max_rates):
            cfg = dict(base, cfar={"false_alarms_per_ping": float(b)})
            if rate is not None:
                cfg["doppler"] = {"mode": "sweep_bank", "max_range_rate_m_s": float(rate), "combine": combine}
            t = time.time()
            with LiveChain(cfg) as chain:
                n = sum(len(chain.process(r.copy())["detections"]) for r in records)
                thr = float(chain.detector["scales"][0]["threshold_db"])
                per_beam = json.loads(chain._cpp.doppler()).get("per_beam", 1) if rate is not None else 1
            key = f"{b}/{'off' if rate is None else rate}"
            out[key] = {"budget": float(b), "max_range_rate_m_s": rate, "per_beam": int(per_beam), "detections": int(n), "pings": len(records),
                        "threshold_db": thr}
            log(f"budget {b:6.3f} {'off' if rate is None else f'cfar {rate:.1f} m/s, {per_beam} a beam'}: {n} detections on {len(records)} pings "
                f"({n / (b * len(records)):.2f} of the budget), single-sample threshold {thr:.3f} dB ({time.time() - t:.0f} s)")
    for b in budgets:
        ref = out[f"{b}/off"]
        for rate in max_rates:
            e = out[f"{b}/{rate}"]
            e["ratio_to_off"] = e["detections"] / max(ref["detections"], 1)
            e["threshold_correction_db"] = e["threshold_db"] - ref["threshold_db"]
            # Three-sigma interval of the ratio of two Poisson counts (normal approximation on the log): coarse.
            if e["detections"] > 0 and ref["detections"] > 0:
                sig = np.sqrt(1.0 / e["detections"] + 1.0 / ref["detections"])
                e["ratio_3sigma"] = [float(e["ratio_to_off"] * np.exp(-3 * sig)), float(e["ratio_to_off"] * np.exp(3 * sig))]
    return out


def at_default(name: str, combines, max_rates, pings: int, budget: float, seed: int, backend: str = "auto", workers: int = 4,
               max_seconds: float | None = None, store: Path | None = None, log=say) -> dict:
    """The live chain's false detections on noise-only pings at `budget` (the default as shipped), without the bank and
    with each combiner and rate, on the same pings: count, ratio to the budget, exact 95 % interval, time a ping."""
    sys.path.insert(0, str(ROOT / "tools"))
    from detector_search_factor import MIN_SNR_DB, noise_pings, poisson_interval
    sonar_name, overrides = SONARS[name]
    sonar = SonarModel.load(sonar_name, overrides)
    tpl = sonar.make_record(core.Pose(), environment={"sound_speed": 1500.0})
    arr = tpl.samples.array
    zero_hi = poisson_interval(0)[1]
    out = {"sonar": sonar_name, "budget": budget, "asked": pings, "seed": seed, "noise_rms": NOISE_RMS, "min_snr_db": 0.0,
           "floor_db": MIN_SNR_DB, "samples_shape": list(arr.shape),
           "resolves": {"zero_count_upper_95_per_ping": zero_hi / pings, "expected_at_budget": budget * pings,
                        "interval_at_budget_95": [v / pings for v in poisson_interval(int(round(budget * pings)))]},
           "rows": {}}
    variants = []
    for c in combines:
        if c == "off":
            variants.append(("off", None))
        else:
            for rate in max_rates:
                variants.append((f"{c}/{rate:g}", {"mode": "sweep_bank", "max_range_rate_m_s": float(rate), "combine": c}))

    def save():
        if store is not None:
            store.write_text(json.dumps(out, indent=1, default=float) + "\n")

    for key, dop in variants:
        cfg = dict(CHAIN, engine="cpp", backend=backend, min_snr_db=0.0)
        if budget != 1e-3:
            cfg["cfar"] = {"false_alarms_per_ping": float(budget)}
        if dop is not None:
            cfg["doppler"] = dop
        found, reported, done, detections = 0, 0, 0, []
        t0 = time.time()
        stopped = False

        def row(complete: bool) -> dict:
            lo, hi = poisson_interval(found)
            r_lo, r_hi = poisson_interval(reported)
            n = max(done, 1)
            return {"combine": key.split("/")[0], "doppler": dop, "per_beam": per_beam, "pings": done, "asked": pings, "complete": complete,
                    "stopped_for_time": stopped, "backend": used, "seconds": round(time.time() - t0, 1),
                    "seconds_a_ping": (time.time() - t0) / n, "found": found, "per_ping": found / n, "ratio": found / (budget * n),
                    "ratio_95": [lo / (budget * n), hi / (budget * n)], "reported": reported, "reported_ratio": reported / (budget * n),
                    "reported_ratio_95": [r_lo / (budget * n), r_hi / (budget * n)], "threshold_db": thr, "detections": detections}

        with LiveChain(cfg) as chain:
            used = chain.backend.name if hasattr(chain.backend, "name") else str(chain.backend)
            per_beam, thr = 1, None
            for i, x in enumerate(noise_pings(arr.shape, np.iscomplexobj(arr), pings, seed, workers)):
                rec = tpl.copy()
                rec.samples.array = x
                rec.transmit.time_ns = int(i) * 20_000_000_000            # a ping every 20 s
                dets = chain.process(rec)["detections"]
                if thr is None:
                    thr = float(chain.detector["scales"][0]["threshold_db"])
                    if dop is not None:
                        per_beam = int(json.loads(chain._cpp.doppler()).get("per_beam", 1))
                done += 1
                found += len(dets)
                reported += sum(1 for d in dets if d.get("snr_db") is not None and d["snr_db"] > MIN_SNR_DB)
                detections += [{"ping": i, "range_m": d.get("range_m"), "bearing_deg": d.get("bearing_deg"), "snr_db": d.get("snr_db"),
                                "range_rate_m_s": d.get("range_rate_m_s")} for d in dets]
                if done % 500 == 0:
                    out["rows"][key] = row(False)
                    save()
                    log(f"{key}: {done} pings, {found} false detections, {(time.time() - t0) / done:.3f} s a ping")
                if max_seconds is not None and time.time() - t0 > max_seconds:
                    stopped = True
                    log(f"{key}: stopped after {max_seconds:g} s at {done} pings")
                    break
        e = out["rows"][key] = row(not stopped)
        save()
        log(f"{sonar_name} {key} ({per_beam} a beam) on {used}: {found} false detections in {done} pings = {e['per_ping']:.2e} a ping, "
            f"{e['ratio']:.2f} of the budget (95 %: {e['ratio_95'][0]:.2f} to {e['ratio_95'][1]:.2f}); above the {MIN_SNR_DB:g} dB floor "
            f"{reported}; single-sample threshold {thr:.3f} dB; {e['seconds_a_ping']:.3f} s a ping")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sonar", default="long_range", choices=sorted(SONARS))
    ap.add_argument("--pings", type=int, help="default 60; --at-default 3000")
    ap.add_argument("--max-per-beam", type=int, default=33)
    ap.add_argument("--budgets", default="3,0.3")
    ap.add_argument("--seed", type=int, help="default 7000; --at-default 30000")
    ap.add_argument("--json")
    ap.add_argument("--chain", choices=["cfar"], help="count the live chain's detections without the bank and with this combiner")
    ap.add_argument("--max-rates", default="3", help="--chain: the bank's max_range_rate_m_s values, comma separated")
    ap.add_argument("--reverb", help="--chain: a scene whose reverberation-only pings are counted instead of noise")
    ap.add_argument("--at-default", action="store_true", help="the live chain at the default budget on the same noise pings: off, max, cfar")
    ap.add_argument("--combines", default="off,max,cfar", help="--at-default: off (no bank), max, cfar, comma separated")
    ap.add_argument("--budget", type=float, default=1e-3, help="--at-default: false detections a ping (default: as shipped)")
    ap.add_argument("--backend", default="auto", choices=["auto", "host", "cuda"], help="--at-default: the chain's backend")
    ap.add_argument("--workers", type=int, default=4, help="--at-default: threads drawing the noise")
    ap.add_argument("--max-seconds", type=float, help="--at-default: stop each row after this long")
    a = ap.parse_args(argv)
    budgets = [float(v) for v in a.budgets.split(",")]
    if a.at_default:
        at_default(a.sonar, [c for c in a.combines.split(",") if c], [float(v) for v in a.max_rates.split(",")],
                   3000 if a.pings is None else a.pings, a.budget, 30000 if a.seed is None else a.seed,
                   a.backend, a.workers, a.max_seconds, Path(a.json) if a.json else None)
        return 0
    if a.pings is None:
        a.pings = 60
    if a.seed is None:
        a.seed = 7000
    if a.chain:
        rates = [float(v) for v in a.max_rates.split(",")]
        if a.reverb:
            import copy
            import warnings
            from gpusonar.config import load_scenario_config
            from gpusonar.sim.simulator import run_scenario
            cfg = load_scenario_config(a.reverb, expand=False)
            cfg["scene"] = []
            cfg["pings"] = dict(cfg.get("pings", {}), count=int(a.pings))
            cfg["seed"] = int(cfg.get("seed", 0)) + 1000
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                records = run_scenario(copy.deepcopy(cfg)).records
            res = {"reverb": a.reverb, "seed": cfg["seed"], "combine": a.chain,
                   "counts": chain_counts(records, budgets, rates, a.chain, bearings="deployment")}
        else:
            sonar_name, overrides = SONARS[a.sonar]
            sonar = SonarModel.load(sonar_name, overrides)
            records = [noise_ping(sonar, a.seed + i, i) for i in range(a.pings)]
            res = {"sonar": sonar_name, "seed": a.seed, "combine": a.chain, "counts": chain_counts(records, budgets, rates, a.chain)}
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
        return 0
    res = measure_noise(a.sonar, a.pings, a.max_per_beam, budgets, a.seed)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
