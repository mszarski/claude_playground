#!/usr/bin/env python3
"""What time spreading does to a bounce-path echo: one target at 1 km in 100 m of isovelocity
water seen by the 32-hydrophone flank array, 60 pings 15 s apart, with the surface (a) flat,
(b) rough with the scattered part dropped (coherent loss only, what the simulator did before),
(c) rough with time spreading (sim/timespread.py). Prints the numbers quoted in docs/RESULTS.md.

    tools/timespread_study.py [--wind 6] [--pings 60]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from gpusonar import recon  # noqa: E402
from gpusonar.sim import ImageSourceProvider, Scene, make_pose, simulate_ping  # noqa: E402
from gpusonar.sim.timespread import wrap  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

C = 1500.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wind", type=float, default=6.0)
    ap.add_argument("--pings", type=int, default=60)
    args = ap.parse_args(argv)
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [900.0, 1150.0]}}})
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=100.0, max_order=2, bottom=dict(speed=1600.0, density=1.8, attenuation=0.5))
    scene = Scene().add_points([[0.0, 1000.0, 50.0]], target_strength_db=10.0)
    pose = make_pose(position=(0.0, 0.0, 30.0))
    spread = {"surface": {"wind_speed": args.wind}, "taps": 8, "seed": 11}
    cases = {"flat surface": inner,
             "coherent loss only": wrap(inner, dict(spread, incoherent=False), frequency=sonar.center_frequency),
             "time spreading": wrap(inner, spread, frequency=sonar.center_frequency)}
    par = cases["time spreading"].surface
    print(f"wind {args.wind} m/s: rms height {par['rms_height']:.3f} m, mean-square slope {par['mean_square_slope']:.4f}, "
          f"rms vertical speed {par['rms_vertical_speed']:.2f} m/s; centre frequency {sonar.center_frequency:.0f} Hz")
    fs = None
    out = {}
    for name, provider in cases.items():
        env = []
        for i in range(args.pings):
            rec = simulate_ping(sonar, pose, scene, provider=provider, environment={"sound_speed": C}, adc=False, noise_rms=0.0,
                                scenario_time=15.0 * i, sequence=i)
            recon.prepare([rec])
            s = rec.samples
            fs, t0 = s.sample_rate, s.start_time
            env.append(np.abs(s.array[16]) ** 2)                       # one hydrophone, after pulse compression
        out[name] = np.array(env)
    t = t0 + np.arange(out["flat surface"].shape[1]) / fs
    direct = 2.0 * np.hypot(1000.0, 20.0) / C
    res = 2.0 / abs(sonar.waveform_config["f_stop"] - sonar.waveform_config["f_start"])
    first = np.abs(t - direct) < res                                   # the direct echo: no bounce
    rest = t > direct + res                                            # every arrival with at least one bounce
    flat = out["flat surface"]
    print(f"{'case':20s} {'direct echo':>12s} {'bounce energy':>14s} {'strongest bounce peak':>22s} {'its scintillation index':>24s} {'energy later than 5 ms after it':>32s}")
    j = int(np.argmax(np.where(rest, flat.mean(axis=0), 0.0)))
    for name, e in out.items():
        peak = e[:, j - 2:j + 3].max(axis=1)
        late = e[:, (t > t[j] + 0.005) & (t < t[j] + 0.012)].sum(axis=1).mean() / max(flat[:, (t > t[j] + 0.005) & (t < t[j] + 0.012)].sum(axis=1).mean(), 1e-300)
        print(f"{name:20s} {10 * np.log10(e[:, first].sum(axis=1).mean() / flat[:, first].sum(axis=1).mean()):+10.2f} dB "
              f"{10 * np.log10(e[:, rest].sum(axis=1).mean() / flat[:, rest].sum(axis=1).mean()):+11.2f} dB "
              f"{10 * np.log10(peak.mean() / flat[:, j - 2:j + 3].max(axis=1).mean()):+19.2f} dB "
              f"{peak.var() / peak.mean() ** 2:24.2f} {10 * np.log10(late):+29.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
