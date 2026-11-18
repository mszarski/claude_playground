#!/usr/bin/env python3
"""What the slow wander of the coherent surface arrival does to a long pulse and to ping-to-ping phase.

    tools/coherent_phase_study.py [--out docs/figures/coherent_phase_study.md] [--json ...]

For the long-range array's band (5 kHz) and a few geometries and seas, the phase law of
`time_spread.coherent_phase` (sim/timespread.py, item 8: the long waves heave the reflecting patch; class
`CoherentPhase`) is evaluated over an hour of surface and turned into the numbers a designer asks for. The path is
the echo that goes out and comes back by the surface (one surface bounce each way: twice the one-way phase), the
worst case; a path with one surface leg has half the phase and a quarter of the small losses.

    Theta             the Fresnel angle of the swell on this bounce (rad): below `fresnel_limit` (1) the patch moves
                      as one and the arrival wanders; above it the swell scatters instead and the arrival is steady
    rms phase         of the echo, rad, and the amplitude's rms relative change (focusing)
    loss, CW          coherent integration of a pulse of the stated length against a replica with no Doppler:
                      -10 log10 < |mean over the pulse of g exp(i phi)|^2 >
    loss, best shift  the same after the best constant frequency shift per ping (what a chirp's range-Doppler
                      coupling or a Doppler search absorbs): what is left is the phase's curvature within the pulse
    ping to ping      rms change of the echo's phase between pings the stated time apart, and the loss of adding
                      four such pings coherently

Seas: a low swell of the kind measured in a fjord on a calm night (0.8 cm rms, a line at 0.2 Hz: PHYSICS_NOTES
Sec. 21.6), two swells of 5 cm and 25 cm rms, and wind seas of states 2 to 4 with ALL of the wind law's height taken
as large-scale at the wind law's own rms wave frequency (v / 2 pi h, sim/acoustics.py): an upper bound, since much of
a wind sea's height is in short waves that scatter. Each with the waves running along the path and across it.

What the simulator does with this: the delay and its rate at mid-pulse (linear phase within a ping), exactly, on
both engines; the curvature within a ping is NOT in the simulated samples. The "best shift" column says how much
that leaves out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import acoustics  # noqa: E402
from gpusonar.sim.timespread import CoherentPhase  # noqa: E402

F, C = 5000.0, 1500.0
GEOMETRIES = [("measured fjord path, 274 m (20 m to 43 m)", 274.0, 20.0, 43.0),
              ("test AUV trial, 500 m (10 m to 15 m)", 500.0, 10.0, 15.0),
              ("static array trial, 2.6 km (30 m to 60 m)", 2600.0, 30.0, 60.0),
              ("long range, 12 km (10 m to 4 m)", 12000.0, 10.0, 4.0)]


def seas() -> list:
    out = [("fjord, calm night: 0.8 cm rms, 5 s", 0.008, 5.0), ("swell 5 cm rms, 8 s", 0.05, 8.0), ("swell 25 cm rms, 10 s", 0.25, 10.0)]
    for state in (2, 3, 4):
        u = acoustics.sea_state_wind_speed(state)
        h, v = acoustics.rms_wave_height(u), acoustics.rms_surface_vertical_speed(u)
        out.append((f"sea state {state} ({u:g} m/s): all {100 * h:.0f} cm rms as large-scale, {2 * np.pi * h / v:.1f} s", h, 2.0 * np.pi * h / v))
    return out


def case(range_m: float, z_source: float, z_point: float, height: float, period: float, direction: str, pulses=(0.1, 1.0), intervals=(5.0, 20.0)) -> dict:
    slant = np.hypot(range_m, z_source + z_point)
    sin_g = (z_source + z_point) / slant
    f1 = z_source / (z_source + z_point)
    rho = slant * f1 * (1.0 - f1)
    cp = CoherentPhase({"rms_height": height, "period_s": period, "direction_deg": 0.0 if direction == "along" else 90.0, "waves": 32, "seed": 5}, frequency=F)
    dt = 0.01
    t = np.arange(0.0, 3600.0, dt)
    ones = np.ones(len(t))
    m = cp.modulation(t, np.zeros((len(t), 2)), np.tile([1.0, 0.0], (len(t), 1)), sin_g * ones, rho * ones, C, height)
    phase = -2.0 * np.pi * F * 2.0 * m["delay"]                                   # out and back by the surface
    gain = ((1.0 + m["amplitude"]) / np.sqrt(1.0 + m["amplitude_variance"])) ** 2
    k = 2.0 * np.pi * F / C
    wave_k = (2.0 * np.pi / period) ** 2 / 9.81
    out = {"grazing_deg": float(np.degrees(np.arcsin(sin_g))), "fresnel_rad": float(rho * wave_k**2 * ((1.0 / sin_g**2) if direction == "along" else 1.0) / (2.0 * k)),
           "share_of_height_taking_part": float(np.mean(m["height2"]) / height**2), "rms_phase_rad": float(phase.std()), "rms_amplitude": float(np.sqrt(gain).std()),
           "gamma2_two_way_if_scattered": float(2.0 * (2.0 * k * height * sin_g) ** 2)}
    z = np.sqrt(gain) * np.exp(1j * phase)
    for pulse in pulses:
        n = int(round(pulse / dt))
        blocks = z[: len(z) // n * n].reshape(-1, n)
        cw = np.abs(blocks.mean(axis=1)) ** 2
        # the best constant frequency shift per pulse: a fine search
        tt = (np.arange(n) - 0.5 * (n - 1)) * dt
        rate = np.gradient(np.unwrap(np.angle(blocks), axis=1), dt, axis=1).mean(axis=1) / (2.0 * np.pi)
        best = np.zeros(len(blocks))
        for off in np.linspace(-0.3, 0.3, 13):
            best = np.maximum(best, np.abs((blocks * np.exp(-2j * np.pi * (rate + off / pulse)[:, None] * tt[None, :])).mean(axis=1)) ** 2)
        norm = (np.abs(blocks) ** 2).mean()
        out[f"loss_cw_db_{pulse:g}s"] = float(-10.0 * np.log10(max(cw.mean() / norm, 1e-12)))
        out[f"loss_best_shift_db_{pulse:g}s"] = float(-10.0 * np.log10(max(best.mean() / norm, 1e-12)))
        out[f"rms_doppler_hz_{pulse:g}s"] = float(rate.std())
    for interval in intervals:
        n = int(round(interval / dt))
        out[f"ping_to_ping_rms_rad_{interval:g}s"] = float((phase[n:] - phase[:-n]).std())
        four = z[: len(z) // (4 * n) * 4 * n: n].reshape(-1, 4)
        out[f"loss_four_pings_db_{interval:g}s"] = float(-10.0 * np.log10(max((np.abs(four.mean(axis=1)) ** 2).mean() / (np.abs(four) ** 2).mean(), 1e-12)))
    return out


def run() -> dict:
    res = {}
    for label, rng, zs, zp in GEOMETRIES:
        for sea, h, period in seas():
            for direction in ("along", "across"):
                res.setdefault(label, {})[f"{sea}; waves {direction} the path"] = case(rng, zs, zp, h, period, direction)
    return res


def markdown(res: dict) -> str:
    lines = ["# The wandering coherent surface arrival: long pulses and ping-to-ping phase at 5 kHz", "",
             "Made by `tools/coherent_phase_study.py` from the phase law of `time_spread.coherent_phase` (sim/timespread.py item 8). The echo that goes out and comes back by the surface "
             "(twice the one-way phase). Waves with a Fresnel angle above 1 rad take no part (they scatter; the arrival is then steady and the row reads zero): the last column is "
             "the two-way Gamma^2 that height has as scattering roughness instead. \"Best shift\": after the best constant frequency shift per pulse (what a chirp absorbs as a range "
             "offset): the curvature the simulator's linear-within-a-ping law leaves out.", ""]
    for label, rows in res.items():
        first = next(iter(rows.values()))
        lines += [f"## {label}; grazing {first['grazing_deg']:.1f} degrees", "",
                  "| Sea | Fresnel angle, rad | height taking part | rms phase, rad (amplitude) | loss of a 0.1 s pulse: CW / best shift, dB | loss of a 1 s pulse: CW / best shift, dB | rms Doppler over 1 s, Hz | "
                  "ping to ping, 5 s / 20 s: rms phase change, rad | four pings added coherently, 5 s / 20 s: loss, dB | two-way Gamma^2 if it scatters |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for sea, r in rows.items():
            lines.append(f"| {sea} | {r['fresnel_rad']:.3g} | {r['share_of_height_taking_part']:.2f} | {r['rms_phase_rad']:.3f} ({r['rms_amplitude']:.3f}) | "
                         f"{r['loss_cw_db_0.1s']:.2f} / {r['loss_best_shift_db_0.1s']:.2f} | {r['loss_cw_db_1s']:.2f} / {r['loss_best_shift_db_1s']:.2f} | {r['rms_doppler_hz_1s']:.3f} | "
                         f"{r['ping_to_ping_rms_rad_5s']:.2f} / {r['ping_to_ping_rms_rad_20s']:.2f} | {r['loss_four_pings_db_5s']:.2f} / {r['loss_four_pings_db_20s']:.2f} | {r['gamma2_two_way_if_scattered']:.3g} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    res = run()
    text = markdown(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
