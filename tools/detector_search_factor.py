#!/usr/bin/env python3
"""Count the multiscale detector's false detections on noise-only pings: where its measured constants come from.

    tools/detector_search_factor.py [--pings 400] [--rows 1,2,3] [--budgets 3,1,0.3,0.1] [--seed 4000] [--json FILE]
    tools/detector_search_factor.py --verify [--pings 200] [--budgets 3,1] [--seed 9000] [--json FILE]
    tools/detector_search_factor.py --search-factor 2.5 ...       the detector as it was: one flat factor for every test
    tools/detector_search_factor.py --at-default [--maps long_range,flank,static] [--rows 1,2,3] [--pings 100000]
                                    [--backend auto] [--workers 4] [--max-seconds S] [--seed 20000] [--json FILE]

The detector (libs/live detector.hpp) turns a budget of false detections per ping into a threshold for each of its
tests by counting the independent tests its map holds. The map is sampled finer than a range resolution and a
beamwidth, its largest sample is searched for, and a detection is a group of flagged cells, so that count cannot be
read off the grid. It is

    the single sample    kSearchSample[rows] x beams x extent / resolution x u
    the cell mean        kSearchMean[rows]   x beams x extent / cell       x sqrt(u) x hazard

with u = rows x looks x threshold (how the number of separate maxima of a smooth field, and of upcrossings along
bearing, grows with the threshold: docs/PHYSICS_NOTES.md Sec. 29.3), and the two constants per number of rows MEASURED
here.

What is done: noise-only pings of the long-range array with one, two and three rows (configs/sonars
popoto_long_range, _2row, _3row; 800 m of window, the deployed bearings -60 to 60 degrees every 0.5, white noise on
every hydrophone) go through the Python chain's own map; on each map the detector is run with ONE of its tests at a
time, at the threshold its plan gives for a stated number of false detections per ping from that test, and the
detections it reports are counted. The constant that would have made the count equal what was asked is the plan's own
times found / asked; it is printed per budget and pooled over the budgets of 3 a ping and fewer (at 10 a ping
neighbouring false detections begin to be merged), with the count and its Poisson error.

`--verify`: the whole detector as shipped (both tests, the constants in the code) at loosened budgets on other seeds:
false detections per ping against the budget.

The table in libs/live/include/sonar/live/detector.hpp was made with the default arguments, starting from a flat factor
of 2.5 (`--search-factor 2.5`, growth laws applied to the thresholds used); run it again and compare before changing
the map (bearing step, samples per cell, aperture window) or the grouping. Nothing is measured at the default budget of
one false detection in a thousand pings: that would take a hundred thousand pings.

`--at-default` is that count (issue sonar-ydi): the whole detector as shipped, through the C++ chain (engine cpp, the map
and the detector's statistic on the GPU with --backend cuda), at the default budget, on a hundred thousand noise-only
pings of each map and row count (DEFAULT_MAPS: D-134's map and a second one), with an exact 95 % Poisson interval for
false detections a ping over the budget. tools/gpu_job.sh runs it on a GPU (GPUSONAR_JOB=falsealarm).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import core, detect, live  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402
from gpusonar.sim import Scene, make_pose, simulate_ping  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

SONARS = {1: "popoto_long_range", 2: "popoto_long_range_2row", 3: "popoto_long_range_3row"}
WINDOW = {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1700.0]}}}
CHAIN = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "cell_statistic": "multiscale"}
NOISE_RMS = 2e-4
POOLED_BUDGETS = 3.0          # budgets up to this many false detections a ping go into the pooled constant
TESTS = ((0, "sample"), (1, "cell mean"))


def say(text: str) -> None:
    print(text, flush=True)


def noise_ping(sonar, seed: int, sequence: int = 0):
    """One ping with nothing in the water: white noise on every hydrophone (the target is 300 dB down)."""
    scene = Scene()
    scene.add_points([[0.0, 1300.0, 0.0]], target_strength_db=-300.0)
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=NOISE_RMS, rng=np.random.default_rng(seed), sequence=sequence,
                         environment={"sound_speed": 1500.0}, adc=False, absorption_model="none")


def chain_config(rows: int, **cfar) -> dict:
    cfg = dict(CHAIN, engine="python")
    if rows > 1:
        cfg["elevation"] = {}                       # the rows' maps are added in power
    cfar = {k: v for k, v in cfar.items() if v is not None}
    if cfar:
        cfg["cfar"] = cfar
    return cfg


def template(rows: int):
    return SonarModel.load(SONARS[rows], WINDOW).make_record(core.Pose(), environment={"sound_speed": 1500.0})


def maps(rows: int, pings: int, seed: int):
    """The chain's own map of each noise ping, as its detector is handed it: (fine, cell_mean, fine_ranges, cell_ranges,
    bearings, guard, train, plan)."""
    sonar = SonarModel.load(SONARS[rows], WINDOW)
    seen = []
    plain = detect.multiscale

    def spy(*args):
        seen.append(args)
        return plain(*args)

    detect.multiscale = spy
    try:
        with LiveChain(chain_config(rows)) as chain:
            for i in range(pings):
                chain.process(noise_ping(sonar, seed + i, i))
                yield seen.pop()
    finally:
        detect.multiscale = plain


def one_test(plan: dict, which: int) -> dict:
    """The plan with only test `which` (0: the sample, 1: the cell mean) switched on."""
    scales = [dict(s) for s in plan["scales"][:2]]
    scales[1 - which]["factor"] = float("inf")
    return dict(plan, scales=scales)


def growth(scale: dict, rows: int) -> float:
    """What the test's count of tests grows by with its threshold (detector.hpp): u for the sample, sqrt(u) x the
    statistic's hazard for a mean; u = rows x looks x threshold."""
    x, a = float(scale["factor"]), rows * float(scale["looks"])
    u = a * x
    if int(scale["cells"]) == 0:
        return u
    h = 1e-4 * x
    w = list(scale["weights"])
    hazard = (np.log(core.weighted_exponential_tail(w, rows, x - h)) - np.log(core.weighted_exponential_tail(w, rows, x + h))) / (2.0 * h * a)
    return float(np.sqrt(u) * hazard)


def measure(rows_list, pings: int, budgets, seed: int, search_factor=None, log=say, store: Path | None = None) -> dict:
    """Each test alone at each budget. `search_factor`: a flat factor for the thresholds used (None: the code's table)."""
    out = {}
    for rows in rows_list:
        t0 = time.time()
        record = template(rows)
        # A budget of 2 b shared between the two tests: b false detections a ping asked of each.
        plans = {b: live.detector_plan(record, chain_config(rows, false_alarms_per_ping=2.0 * b, search_factor=search_factor), rows) for b in budgets}
        counts = {(w, b): 0 for w, _ in TESTS for b in budgets}
        for args in maps(rows, pings, seed + 100000 * rows):
            for (which, b) in counts:
                counts[(which, b)] += len(detect.multiscale(*args[:-1], one_test(plans[b], which))[0])
        any_plan = plans[budgets[0]]
        res = {"rows": rows, "pings": pings, "seconds": round(time.time() - t0, 1), "beams": any_plan["beams"], "cell_m": any_plan["cell_m"],
               "resolution_m": any_plan["resolution_m"], "samples_per_cell": any_plan["samples_per_cell"], "tests": {}}
        for which, name in TESTS:
            entries, found, expected = [], 0, 0.0
            for b in budgets:
                s = plans[b]["scales"][which]
                n = counts[(which, b)]
                places = any_plan["beams"] * res["cell_m"] * (len(args[3])) / max(s["length_m"], res["resolution_m"])
                grows = growth(s, rows)
                # the constant that would have made the count what was asked, with the growth laws: tests x found / asked / (places x growth)
                constant = s["tests"] * n / (b * pings) / (places * grows)
                entries.append({"asked_per_ping": b, "found": n, "found_per_ping": n / pings, "ratio": n / (b * pings), "ratio_sigma": np.sqrt(max(n, 1)) / (b * pings),
                                "threshold_db": float(10.0 * np.log10(s["factor"])), "plan_tests": s["tests"], "plan_search": s["search"], "growth": grows,
                                "constant": constant})
                if b <= POOLED_BUDGETS:
                    found += n
                    expected += b * pings * places * grows / s["tests"]
                log(f"rows {rows} {name:9s} asked {b:5.2f} a ping at {entries[-1]['threshold_db']:5.2f} dB: found {n:5d} in {pings} pings = {n / pings:.3f} a ping, "
                    f"{n / (b * pings):.2f} +- {np.sqrt(max(n, 1)) / (b * pings):.2f} of what was asked; constant {constant:.3f}")
            pooled = found / expected if expected > 0 else float("nan")
            res["tests"][name] = {"looks": plans[budgets[0]]["scales"][which]["looks"], "by_budget": entries, "constant": pooled,
                                  "constant_sigma": pooled / np.sqrt(max(found, 1)), "detections": found}
            log(f"rows {rows} {name:9s} constant over the budgets of {POOLED_BUDGETS:g} a ping and fewer: {pooled:.3f} +- {pooled / np.sqrt(max(found, 1)):.3f} ({found} detections)")
        out[str(rows)] = res
        if store is not None:
            store.write_text(json.dumps(out, indent=1, default=float) + "\n")       # after every row count: a long run can be read while it goes
    return out


def verify(rows_list, pings: int, budgets, seed: int, search_factor=None, log=say, store: Path | None = None) -> dict:
    """The detector as shipped (both tests) at loosened budgets, counted on the chain's own maps."""
    out = {}
    for rows in rows_list:
        record = template(rows)
        plans = {b: live.detector_plan(record, chain_config(rows, false_alarms_per_ping=b, search_factor=search_factor), rows) for b in budgets}
        counts = {b: 0 for b in budgets}
        for args in maps(rows, pings, seed + 100000 * rows):
            for b in budgets:
                counts[b] += len(detect.multiscale(*args[:-1], plans[b])[0])
        out[str(rows)] = {str(b): {"found": n, "pings": pings, "per_ping": n / pings, "ratio": n / (b * pings), "ratio_sigma": np.sqrt(max(n, 1)) / (b * pings),
                                   "thresholds_db": [float(10.0 * np.log10(s["factor"])) for s in plans[b]["scales"]]} for b, n in counts.items()}
        for b, n in counts.items():
            log(f"rows {rows} budget {b:5.2f} a ping: {n} false detections in {pings} pings = {n / pings:.3f} a ping, {n / (b * pings):.2f} +- {np.sqrt(max(n, 1)) / (b * pings):.2f} of the budget")
        if store is not None:
            store.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


# --- the default budget, counted directly (issue sonar-ydi) -----------------------------------------------------------

# The maps the default budget is counted on: D-134's (the long-range array, 0.8 km of window) and a second one per row
# count. One row: the flank array (configs/sonars/flank_32: another sonar, a 6-10 kHz band, so 15 samples in a 2 m cell
# where D-134's map had 8, and a cell mean of 10.8 looks, not 5.5), at baseband with the band a quarter of the sample
# rate as in D-134's map, over the same 0.8 km. Two and three rows: no other sonar has stacked rows, so the static
# arrays (popoto_static_2row, _3row) over their whole deployed window, 75 m to 3 km: 3.7 times the cells of D-134's map
# (thresholds 0.2 dB higher) and its near end at 75 m. Bearings -60 to 60 degrees every 0.5, as deployed for all.
DEFAULT_MAPS = {
    "long_range": {r: (SONARS[r], WINDOW) for r in (1, 2, 3)},
    "flank": {1: ("flank_32", {"receive": {"kind": "baseband", "sample_rate": 16000.0, "window": {"range": [900.0, 1700.0]}}})},
    "static": {r: (f"popoto_static_{r}row", {"receive": {"kind": "baseband", "sample_rate": 8000.0}}) for r in (2, 3)},
}
DEFAULT_BUDGET = 1e-3          # cfar.false_alarms_per_ping as shipped (live.DEFAULTS, detector.hpp)


def poisson_interval(n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) two-sided interval for the mean of a Poisson count n."""
    from scipy.stats import chi2
    a = 1.0 - level
    lo = 0.0 if n == 0 else 0.5 * chi2.ppf(0.5 * a, 2 * n)
    return float(lo), float(0.5 * chi2.ppf(1.0 - 0.5 * a, 2 * n + 2))


def noise_pings(shape, complex_samples: bool, pings: int, seed: int, workers: int):
    """White Gaussian noise for `pings` records of `shape`, as receiver.add_noise draws it (NOISE_RMS per real component),
    drawn on `workers` threads (NumPy lets go of the interpreter while it draws), in order. Ping i always gets the stream
    SeedSequence(seed).spawn(pings)[i], whatever the number of workers."""
    from concurrent.futures import ThreadPoolExecutor
    streams = np.random.SeedSequence(seed).spawn(pings)

    def draw(i):
        rng = np.random.Generator(np.random.PCG64(streams[i]))
        if complex_samples:
            x = rng.standard_normal(tuple(shape) + (2,), dtype=np.float32)
            x *= np.float32(NOISE_RMS)
            return x.view(np.complex64)[..., 0]
        x = rng.standard_normal(tuple(shape), dtype=np.float32)
        x *= np.float32(NOISE_RMS)
        return x

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        ahead = 2 * max(workers, 1)
        futures = [pool.submit(draw, i) for i in range(min(ahead, pings))]
        for i in range(pings):
            x = futures[i].result()
            futures[i] = None
            if i + ahead < pings:
                futures.append(pool.submit(draw, i + ahead))
            yield x


def at_default(maps, rows_list, pings: int, seed: int, backend: str = "auto", budget: float = DEFAULT_BUDGET, workers: int = 4,
               max_seconds: float | None = None, log=say, store: Path | None = None) -> dict:
    """The detector as shipped (both tests, the constants in the code) at `budget` false detections a ping, through the
    C++ chain (engine cpp: the map and the detector's statistic on `backend`), on noise-only pings of each map: the
    false detections counted against the budget, with an exact 95 % Poisson interval. Each detection is put down to
    the single-sample test when its strongest sample over the background reaches that test's threshold (with the
    training correction of a full window), else to the cell mean. `max_seconds`: stop a map and row count early (said)."""
    out = {}
    for name in maps:
        for rows in rows_list:
            if rows not in DEFAULT_MAPS[name]:
                continue
            sonar_name, overrides = DEFAULT_MAPS[name][rows]
            sonar = SonarModel.load(sonar_name, overrides)
            tpl = sonar.make_record(core.Pose(), environment={"sound_speed": 1500.0})
            cfg = dict(CHAIN, engine="cpp", backend=backend)
            if rows > 1:
                cfg["elevation"] = {}
            if budget != DEFAULT_BUDGET:
                cfg["cfar"] = {"false_alarms_per_ping": budget}
            plan = live.detector_plan(tpl, cfg, rows)
            s0 = plan["scales"][0]
            sample_db = float(10.0 * np.log10(s0["factor"] * core.training_correction(s0["pfa"], plan["train_cells"] * plan["train_looks_per_cell"])))
            arr = tpl.samples.array
            t0 = time.time()
            found, done, per_ping, detections = 0, 0, {}, []
            with LiveChain(cfg) as chain:
                key = f"{name}/{rows}"
                for i, x in enumerate(noise_pings(arr.shape, np.iscomplexobj(arr), pings, seed + 1000 * rows + 100000 * list(DEFAULT_MAPS).index(name), workers)):
                    rec = tpl.copy()
                    rec.samples.array = x
                    rec.transmit.time_ns = int(i) * 20_000_000_000            # a ping every 20 s
                    dets = chain.process(rec)["detections"]
                    done += 1
                    found += len(dets)
                    per_ping[len(dets)] = per_ping.get(len(dets), 0) + 1
                    for d in dets:
                        detections.append({"ping": i, "range_m": d.get("range_m"), "bearing_deg": d.get("bearing_deg"), "snr_db": d.get("snr_db"),
                                           "extent_m": d.get("extent_m")})
                    if done % 10000 == 0:
                        log(f"{key}: {done} pings, {found} false detections, {(time.time() - t0) / done * 1e3:.1f} ms a ping")
                    if max_seconds is not None and time.time() - t0 > max_seconds:
                        log(f"{key}: stopped after {max_seconds:g} s at {done} pings")
                        break
                used = chain.backend.name if hasattr(chain.backend, "name") else str(chain.backend)
                thresholds = [s["threshold_db"] for s in chain.detector["scales"]]
            lo, hi = poisson_interval(found)
            by_sample = sum(1 for d in detections if d["snr_db"] is not None and d["snr_db"] >= sample_db)
            res = {"map": name, "sonar": sonar_name, "rows": rows, "pings": done, "asked": pings, "budget": budget, "backend": used,
                   "seconds": round(time.time() - t0, 1), "found": found, "per_ping": found / done, "ratio": found / (budget * done),
                   "ratio_95": [lo / (budget * done), hi / (budget * done)], "pings_by_count": {str(k): v for k, v in sorted(per_ping.items())},
                   "sample_test": by_sample, "cell_mean_only": found - by_sample, "sample_threshold_db": sample_db, "thresholds_db": thresholds,
                   "beams": plan["beams"], "cell_m": plan["cell_m"], "samples_per_cell": plan["samples_per_cell"], "resolution_m": plan["resolution_m"],
                   "looks": [s["looks"] for s in plan["scales"]], "tests": [s["tests"] for s in plan["scales"]], "search": [s["search"] for s in plan["scales"]],
                   "samples_shape": list(arr.shape), "detections": detections}
            out[f"{name}/{rows}"] = res
            log(f"{name} ({sonar_name}) rows {rows} on {used}: {found} false detections in {done} pings = {found / done:.2e} a ping, "
                f"{res['ratio']:.2f} of the budget (95 %: {res['ratio_95'][0]:.2f} to {res['ratio_95'][1]:.2f}); "
                f"single sample {by_sample}, cell mean only {found - by_sample}; thresholds {', '.join(f'{t:.2f}' for t in thresholds)} dB; {res['seconds']} s")
            if store is not None:
                store.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pings", type=int, default=None)
    ap.add_argument("--rows", default="1,2,3")
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--search-factor", type=float, default=None, help="a flat factor for the thresholds used (2.5: the detector as it was)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--at-default", action="store_true", help="the shipped detector at the default budget through the C++ chain")
    ap.add_argument("--maps", default=",".join(DEFAULT_MAPS), help="--at-default: maps (long_range, flank, static)")
    ap.add_argument("--backend", default="auto", help="--at-default: auto, host or cuda")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="--at-default: false detections a ping (default: as shipped)")
    ap.add_argument("--workers", type=int, default=4, help="--at-default: threads drawing the noise")
    ap.add_argument("--max-seconds", type=float, default=None, help="--at-default: stop each map and row count after this long")
    args = ap.parse_args(argv)
    rows = [int(r) for r in args.rows.split(",") if r]
    store = Path(args.json) if args.json else None
    if args.at_default:
        at_default([m for m in args.maps.split(",") if m], rows, args.pings or 100000, 20000 if args.seed is None else args.seed, args.backend,
                   args.budget, args.workers, args.max_seconds, store=store)
    elif args.verify:
        budgets = [float(b) for b in (args.budgets or "3,1").split(",") if b]
        verify(rows, args.pings or 200, budgets, 9000 if args.seed is None else args.seed, args.search_factor, store=store)
    else:
        budgets = [float(b) for b in (args.budgets or "3,1,0.3,0.1").split(",") if b]
        measure(rows, args.pings or 400, budgets, 4000 if args.seed is None else args.seed, args.search_factor, store=store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
