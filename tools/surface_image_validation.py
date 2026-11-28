#!/usr/bin/env python3
"""The surface-image factor of gpusonar.lloyd held against the simulator and against Bellhop.

    tools/surface_image_validation.py [--quick]      # writes docs/design/surface_image_validation.json

1. Simulator, straight-ray image-source paths (`run_scenario`, both engines), no noise: one point target at broadside,
   the long-range sonar's 4-6 kHz 1 s sweep. The echo is pulse-compressed and summed over the row; its peak and its
   energy with the surface, over the same without it, against `lloyd.two_way` (peak_db, energy_db), on a grid of own
   depth and target depth at 12 km and at 3 km.
2. The same with a rough surface (`time_spread.surface: {rms_height}`, coherent part only and with the scattered part)
   where the roughness matters (steeper grazing angles).
3. Through the C++ chain with noise: the SNR the detector reports for the target against own depth, less the SNR of the
   same target without a surface, against the model.
4. Bellhop (GPUSONAR_BELLHOP), one way: a deep layer whose bottom returns nothing (surface image only) and 100 m of water
   over sand (the image series), band energy and single-frequency loss against `lloyd.one_way`.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import lloyd  # noqa: E402
from gpusonar.config import load_scenario_config, load_yaml  # noqa: E402
from gpusonar.sim import run_scenario  # noqa: E402
from gpusonar.sim.bellhop import find_bellhop  # noqa: E402

F_LO, F_HI = 4000.0, 6000.0
PROJECTOR_BELOW = 1.5                                                   # popoto_long_range: transmit.lever_arm [0, 0, 1.5]
SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}            # the popoto_12km scenario's bottom (textbook sand)


def scenario(own: float, target: float, rng: float, surface: bool, bottom_z=None, spread=None, noise=False, ts=25.0, seed=3) -> dict:
    """One point target at broadside of the long-range row, straight rays; `surface` False: free field."""
    cfg = {"name": "surface_image_check", "sonar": "popoto_long_range", "fidelity": "image_source" if surface else "free_field",
           # D-256: popoto_long_range sends a 2 s hyperbolic sweep since; `echo` compresses with the 1 s linear sweep's replica, so the
           # check states that sweep (the model is the flat band's; the committed validation was made with it)
           "sonar_overrides": {"receive": {"kind": "baseband", "sample_rate": 8.0e3, "window": {"range": [rng - 400.0, rng + 400.0]}},
                               "transmit": {"waveform": {"type": "lfm", "source": None, "duration": 1.0}}},
           "environment": {"sound_speed": 1500.0, "temperature": 12.0, "salinity": 35.0},
           "platform": {"position": [0.0, 0.0, float(own)], "rpy": [0, 0, 0], "velocity": [0.0, 0.0, 0.0]},
           "pings": {"count": 1, "interval": 20.0},
           "scene": [{"type": "points", "label": "target", "positions": [[0.0, float(rng), float(target)]], "target_strength_db": float(ts)}],
           "absorption": "ainslie_mccolm", "receiver": {"adc": False}, "seed": seed}
    if surface:
        cfg["propagation"] = {"surface_z": 0.0, "bottom_z": bottom_z, "max_order": 1 if bottom_z is None else 2}
        if bottom_z is not None:
            cfg["propagation"]["bottom"] = dict(SAND)
    if spread:
        cfg["time_spread"] = spread
    if noise:
        cfg["receiver"]["ambient_noise"] = {"wind_speed": 5.0, "shipping": 0.5}
    return cfg


def echo(cfg: dict, engine: str):
    """(peak power, energy) of the compressed echo summed over the row (the target is at broadside)."""
    rec = run_scenario(copy.deepcopy(cfg), engine=engine).records[0]
    x = rec.samples.array.sum(axis=0)
    fs = rec.samples.sample_rate
    t = np.arange(int(round(fs * 1.0))) / fs                                  # the sonar's 1 s sweep at baseband (centre 5 kHz)
    rep = np.exp(1j * np.pi * (F_HI - F_LO) * (t - 0.5) ** 2)
    n = int(2 ** np.ceil(np.log2(x.size + rep.size)))
    y = np.fft.ifft(np.fft.fft(x, n) * np.conj(np.fft.fft(rep, n)))[:x.size]
    p = np.abs(y) ** 2
    return float(p.max()), float(p.sum())


def simulator_grid(rng: float, owns, targets, engines) -> list:
    rows = []
    for zt in targets:
        for zo in owns:
            model = lloyd.two_way(F_LO, F_HI, zo, zt, rng, n_f=257, projector_below_m=PROJECTOR_BELOW)
            row = {"range_m": rng, "own_depth_m": zo, "target_depth_m": zt, "model_peak_db": float(model["peak_db"]), "model_energy_db": float(model["energy_db"]),
                   "model_narrowband_db": float(model["narrowband_db"]), "textbook_db": float(0.5 * (lloyd.lloyd_two_way_db(5000.0, zo, zt, rng) + lloyd.lloyd_two_way_db(5000.0, zo + PROJECTOR_BELOW, zt, rng)))}
            for engine in engines:
                p1, e1 = echo(scenario(zo, zt, rng, True), engine)
                p0, e0 = echo(scenario(zo, zt, rng, False), engine)
                row[f"sim_{engine}_peak_db"] = 10 * np.log10(p1 / p0)
                row[f"sim_{engine}_energy_db"] = 10 * np.log10(e1 / e0)
            rows.append(row)
            print("  " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return rows


def rough(engine: str) -> list:
    """A rough surface where it matters: 1 km, both at 60 m (grazing 6.8 deg), rms heights 0.1 to 0.4 m."""
    rows = []
    rng, zo, zt = 1000.0, 60.0, 60.0
    p0, e0 = echo(scenario(zo, zt, rng, False), engine)
    for h in (0.0, 0.1, 0.2, 0.4):
        m = lloyd.two_way(F_LO, F_HI, zo, zt, rng, lloyd.Surface(rms_height=h), n_f=513, projector_below_m=PROJECTOR_BELOW)
        row = {"range_m": rng, "own_depth_m": zo, "target_depth_m": zt, "rms_height_m": h, "model_peak_db": float(m["peak_db"]), "model_energy_db": float(m["energy_db"]),
               "model_coherent_share_one_way": float(m["coherent_share"])}
        if h > 0:
            mss = 0.02                                                     # illustrative mean-square slope: sets only how the scattered part spreads in time
            surf = {"rms_height": h, "mean_square_slope": mss, "rms_vertical_speed": 0.0}
            pc, ec = echo(scenario(zo, zt, rng, True, spread={"surface": surf, "incoherent": False}), engine)
            es = [echo(scenario(zo, zt, rng, True, spread={"surface": surf, "taps": 6}, seed=s), engine)[1] for s in range(1, 9)]
            row.update(sim_coherent_peak_db=10 * np.log10(pc / p0), sim_coherent_energy_db=10 * np.log10(ec / e0),
                       sim_with_scattered_energy_db=10 * np.log10(np.mean(es) / e0), realisations=len(es))
        else:
            pc, ec = echo(scenario(zo, zt, rng, True), engine)
            row.update(sim_coherent_peak_db=10 * np.log10(pc / p0), sim_coherent_energy_db=10 * np.log10(ec / e0))
        rows.append(row)
        print("  " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return rows


def budget_gap(zo: float, zt: float, rng: float, ts: float, chain: dict, d: dict, budget_db: float, chain_db: float) -> dict:
    """Why the chain's `snr_db` for a point in free field reads below the link budget's SNR of one row, term by term (dB,
    negative: the chain has less). Closed forms from the budget's own models, and two measurements on simulated pings
    that carry no detector: the noise-free echo on the chain's own map, and noise alone through an unweighted and a
    windowed beamformer. `unexplained_db` is what is left of chain - budget after the terms.

    Terms the BUDGET leaves out (the simulator has them): the sweep's taper and the absorption's slope across the band, the
    noise spectrum's slope across the band, the row's gain against isotropic noise across the band. Terms the DETECTOR'S
    threshold already holds (buildspec.detection): the aperture window, where the echo falls between the map's samples
    (here the value for THIS target's place on the map; the threshold holds the mean over places)."""
    from gpusonar import acoustics, core, flank, live, recon
    from gpusonar.sim.noise import wenz_spectrum_db
    from gpusonar.sonar_equation import isotropic_noise_gain_db
    db = lambda x: float(10.0 * np.log10(x))                                           # noqa: E731
    pulse, n, pitch, c = d["pulse_s"], d["per_row"][0], d["pitch_m"], d["sound_speed"]
    w_cfg = d["sonar_config"]["transmit"]["waveform"]
    alpha = float(w_cfg.get("window_alpha", 0.0))
    slant = float(np.hypot(rng, zo - zt))
    # the sweep's taper (a Tukey window) and the absorption frequency by frequency, through a replica that has the taper
    t = (np.arange(4000) + 0.5) / 4000.0
    taper = np.ones_like(t)
    if alpha > 0:
        edge = t < alpha / 2
        taper[edge] = 0.5 * (1 - np.cos(2 * np.pi * t[edge] / alpha))
        taper[::-1][edge] = taper[edge]
    f = F_LO + (F_HI - F_LO) * t
    a = np.array([float(acoustics.absorption(x, 12.0, 35.0, 50.0, model="ainslie_mccolm")) for x in f])
    a0 = float(acoustics.absorption(0.5 * (F_LO + F_HI), 12.0, 35.0, 50.0, model="ainslie_mccolm"))
    amp = taper * 10.0 ** (-(a - a0) * 2.0 * slant / 20.0)
    terms = {"sweep_taper_db": db(np.mean(taper ** 2)),
             "taper_and_absorption_slope_db": db(np.mean(amp * taper) ** 2 / np.mean(taper ** 2))}
    terms["absorption_slope_across_band_db"] = terms["taper_and_absorption_slope_db"] - terms["sweep_taper_db"]
    band = np.linspace(F_LO, F_HI, 81)
    wind = 5.0
    terms["noise_across_band_db"] = -(db(np.mean(10.0 ** (wenz_spectrum_db(band, wind, 0.5) / 10.0))) - float(wenz_spectrum_db(np.array([0.5 * (F_LO + F_HI)]), wind, 0.5)[0]))
    pos = np.arange(n) * pitch
    centre = isotropic_noise_gain_db(pos, 0.5 * (F_LO + F_HI), c)
    terms["row_gain_across_band_db"] = db(1.0 / np.mean([10.0 ** (-isotropic_noise_gain_db(pos, x, c) / 10.0) for x in band])) - centre
    full = live._merge(live.DEFAULTS, chain)
    terms["aperture_window_db"] = -db(live._window_loss(n, full["window"]))
    # where this echo falls between the map's samples: the noise-free ping on the chain's own grid against a fine one
    quiet = run_scenario(copy.deepcopy(scenario(zo, zt, rng, False, noise=False, ts=ts))).records[0]
    recon.prepare([quiet], backend=core.Backend.Host)
    g = live._map_geometry(quiet, full)
    r0 = 0.5 * quiet.environment.sound_speed * quiet.samples.start_time
    grid = r0 + g["step"] * np.arange(int(800.0 / g["step"]))
    near = grid[np.abs(grid - slant) < 3.0]
    on_grid = np.abs(flank.backprojection_map(quiet, near, np.radians([-0.5, 0.0, 0.5]), window=full["window"], interp="linear", backend=core.Backend.Host).image) ** 2
    fine = np.abs(flank.backprojection_map(quiet, slant + np.arange(-2.0, 2.0, 0.005), np.radians([0.0]), window=full["window"], interp="linear", backend=core.Backend.Host).image) ** 2
    terms["between_samples_here_db"] = db(on_grid.max() / fine.max())
    mean_straddle = -live.snr_loss_db(quiet, g["cell"], full["window"], full["cell_statistic"], n)["cell_db"]
    leaves_out = ("taper_and_absorption_slope_db", "noise_across_band_db", "row_gain_across_band_db")
    in_threshold = ("aperture_window_db", "between_samples_here_db")
    total = sum(terms[k] for k in leaves_out + in_threshold)
    return {"terms_db": terms, "budget_leaves_out": list(leaves_out), "budget_leaves_out_db": sum(terms[k] for k in leaves_out),
            "in_the_detectors_threshold": list(in_threshold), "between_samples_mean_over_places_db": mean_straddle,
            "explained_db": total, "chain_minus_budget_db": chain_db - budget_db, "unexplained_db": (chain_db - budget_db) - total,
            "note": "the detector's threshold (buildspec.detection) holds the window and the MEAN loss between samples; this target's place costs "
                    f"{terms['between_samples_here_db'] - mean_straddle:+.2f} dB more than the mean"}


def through_chain(owns, zt=4.0, rng=12000.0, ts=55.0) -> dict:
    """SNR the C++ chain reports for a point at 12 km against own depth, with ambient noise at 5 m/s wind, less the SNR of the
    same point without a surface. The point is `ts` dB strong so that it is detected at every depth, nulls included; the
    free-field SNR is also set against the link budget's for the same target (what the chain's detector loses to it)."""
    from gpusonar import buildspec as bs
    from gpusonar.live import LiveChain
    chain = load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle.yaml")["chain"]
    chain.pop("sync", None)

    def snr(cfg):
        records = run_scenario(copy.deepcopy(cfg)).records
        with LiveChain(dict(chain, engine="cpp", backend="host")) as live:
            rep = live.run(records)[0]
        near = [d for d in rep["detections"] if d["snr_db"] is not None and abs(d["range_m"] - rng) < 30.0 and abs(d["bearing_deg"]) < 1.5]
        return max((d["snr_db"] for d in near), default=None)

    free = [snr(scenario(100.0, zt, rng, False, noise=True, ts=ts, seed=s)) for s in range(3, 15)]
    free_db = float(np.mean([x for x in free if x is not None])) if any(x is not None for x in free) else None
    d = bs.describe_shipped("popoto_long_range", "popoto_vehicle")
    perf = bs.performance(d, ranges=(rng,))
    budget = perf["row_snr_db"]["ship"][str(rng)] - 25.0 + ts
    rows = []
    for zo in owns:
        got = snr(scenario(zo, zt, rng, True, noise=True, ts=ts))
        m = lloyd.two_way(F_LO, F_HI, zo, zt, rng, n_f=257, projector_below_m=PROJECTOR_BELOW)
        rows.append({"own_depth_m": zo, "target_depth_m": zt, "range_m": rng, "chain_snr_db": got,
                     "chain_factor_db": None if got is None or free_db is None else got - free_db, "model_peak_db": float(m["peak_db"]), "model_energy_db": float(m["energy_db"])})
        print("  ", rows[-1], flush=True)
    out = {"target_strength_db": ts, "chain_snr_free_field_db": free_db, "chain_snr_free_field_pings": len(free), "chain_snr_free_field_std_db": float(np.std([x for x in free if x is not None])),
           "budget_snr_free_field_db": float(budget), "chain_minus_budget_db": None if free_db is None else free_db - float(budget),
           "accounting": None if free_db is None else budget_gap(100.0, zt, rng, ts, chain, d, float(budget), free_db), "rows": rows}
    print("  ", {k: v for k, v in out.items() if k != "rows"}, flush=True)
    return out


def bellhop_checks(quick: bool) -> dict:
    from gpusonar.sim.bellhop import OceanEnvironment, run_bellhop
    f = lloyd.band(F_LO, F_HI, 129)
    out = {}
    cases = {"surface image only (1000 m of water, a bottom that returns nothing)":
             (OceanEnvironment(depth=1000.0, sound_speed=1500.0, bottom_speed=1500.0, bottom_density=1.0, bottom_attenuation=0.0), lloyd.Layer(), [10.0, 30.0, 75.0, 150.0, 225.0, 300.0], (-20.0, 20.0)),
             "100 m of isovelocity water over sand (the image series)":
             (OceanEnvironment(depth=100.0, sound_speed=1500.0, bottom_speed=SAND["speed"], bottom_density=SAND["density"], bottom_attenuation=SAND["attenuation"]),
              lloyd.Layer(depth=100.0, bottom=SAND), [10.0, 30.0, 50.0, 70.0, 90.0], (-40.0, 40.0))}
    for label, (env, layer, owns, angles) in cases.items():
        rows = []
        for zt in ((4.0,) if quick else (4.0, 30.0)):
            t0 = time.time()
            arr = run_bellhop(env, 5000.0, zt, owns, [12000.0], angles=angles)        # reciprocity: the source at the target's depth
            for i, zo in enumerate(owns):
                hb, sb = lloyd.bellhop_transfer(arr, i, 0, f)
                hm, sm = lloyd.one_way(f, zo, zt, 12000.0, lloyd.FLAT, layer)
                b, m = lloyd.summarise(hb, sb), lloyd.summarise(hm, sm)
                rows.append({"own_depth_m": zo, "target_depth_m": zt, "arrivals": int(len(arr.delay[i][0])),
                             **{f"bellhop_{k}": float(b[k]) for k in ("one_way_energy_db", "one_way_narrowband_db", "energy_db", "peak_db")},
                             **{f"model_{k}": float(m[k]) for k in ("one_way_energy_db", "one_way_narrowband_db", "energy_db", "peak_db")}})
                print("  " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()), flush=True)
            print(f"  ({time.time() - t0:.0f} s)", flush=True)
        d = np.array([r["bellhop_one_way_energy_db"] - r["model_one_way_energy_db"] for r in rows])
        d2 = np.array([r["bellhop_peak_db"] - r["model_peak_db"] for r in rows])
        out[label] = {"rows": rows, "one_way_energy_difference_db": {"mean": float(d.mean()), "largest": float(np.abs(d).max())},
                      "two_way_peak_difference_db": {"mean": float(d2.mean()), "largest": float(np.abs(d2).max())}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--only-chain", action="store_true", help="redo the through-the-chain part only and replace it in the existing JSON")
    args = ap.parse_args(argv)
    if args.only_chain:
        path = ROOT / "docs" / "design" / "surface_image_validation.json"
        R = json.loads(path.read_text())
        R["through_chain"] = through_chain([10.0, 30.0, 75.0, 150.0, 225.0, 300.0, 450.0])
        print(json.dumps(R["through_chain"]["accounting"], indent=1))
        if not args.no_write:
            path.write_text(json.dumps(R, indent=2, default=float) + "\n")
        return 0
    engines = ("cpp", "numpy")
    owns = [10.0, 75.0, 225.0, 450.0] if args.quick else [10.0, 30.0, 75.0, 150.0, 225.0, 300.0, 450.0, 600.0]
    R = {"band_hz": [F_LO, F_HI], "what": __doc__.split("\n\n")[0]}
    print("simulator, 12 km")
    R["simulator_12km"] = simulator_grid(12000.0, owns, [4.0] if args.quick else [4.0, 30.0, 150.0], engines)
    print("simulator, 3 km")
    R["simulator_3km"] = simulator_grid(3000.0, [10.0, 56.0, 112.0] if args.quick else [10.0, 30.0, 56.0, 112.0, 150.0, 300.0], [4.0] if args.quick else [4.0, 60.0], engines)
    for key in ("simulator_12km", "simulator_3km"):
        for engine in engines:
            for kind in ("peak", "energy"):
                d = np.array([r[f"sim_{engine}_{kind}_db"] - r[f"model_{kind}_db"] for r in R[key]])
                R.setdefault("largest_difference_db", {})[f"{key}_{engine}_{kind}"] = float(np.abs(d).max())
    print("rough surface")
    R["rough_surface"] = rough("cpp")
    print("through the chain")
    R["through_chain"] = through_chain([10.0, 75.0, 150.0, 225.0] if args.quick else [10.0, 30.0, 75.0, 150.0, 225.0, 300.0, 450.0])
    if find_bellhop() is not None:
        print("Bellhop")
        R["bellhop"] = bellhop_checks(args.quick)
    else:
        R["bellhop"] = None
    print(json.dumps(R["largest_difference_db"], indent=1))
    if not args.no_write:
        (ROOT / "docs" / "design" / "surface_image_validation.json").write_text(json.dumps(R, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
