#!/usr/bin/env python3
"""Calibrate the simulator to real data (gpusonar/simcal.py). Writes scenario overrides.

  Forward-looking sonar log, raw element data as a container file of ping records:
    tools/calibrate_sim.py fls --log dive12.h5 --scenario fls_120k_transit --levels absolute
  The same from frames beamformed by another system (an .npz with intensity [frame, range,
  bearing] as power, ranges in metres, bearings_deg):
    tools/calibrate_sim.py fls --frames dive12.npz --scenario fls_120k_transit --anchor boundary:0
  A SAS image (.npy power, or a picture; --display-db MIN MAX if the picture is linear in dB):
    tools/calibrate_sim.py sas --image mosaic.png --pixel 0.03 --display-db -15 25

--scenario names the setting the data was taken in: the sonar, its depth and attitude, the
water depth, and which boundaries can scatter. Edit a copy to match the dive first.
Use --levels absolute only when the records carry true gains and the sonar config the true
source level and sensitivity; otherwise only differences between terms are meaningful.
--out FILE.yaml writes the overrides; --plot FILE.png draws the fit.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import simcal  # noqa: E402
from gpusonar.config import dump_yaml, load_scenario_config  # noqa: E402
from gpusonar.sources import make_source  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fls")
    f.add_argument("--log")
    f.add_argument("--frames")
    f.add_argument("--scenario", required=True)
    f.add_argument("--levels", choices=["absolute", "relative"], default=None)
    f.add_argument("--anchor", default="noise")
    f.add_argument("--ranges", type=float, nargs=3, metavar=("FIRST", "LAST", "STEP"), help="imaging grid for --log (m)")
    f.add_argument("--bearings", type=float, nargs=3, default=[-40.0, 40.0, 1.0], metavar=("FIRST", "LAST", "STEP"))
    f.add_argument("--max-pings", type=int, default=20)
    s = sub.add_parser("sas")
    s.add_argument("--image", required=True)
    s.add_argument("--pixel", type=float, nargs="+", required=True, help="metres: one number, or along and across track")
    s.add_argument("--display-db", type=float, nargs=2, metavar=("MIN", "MAX"))
    s.add_argument("--block", type=int, default=48)
    for p in (f, s):
        p.add_argument("--out")
        p.add_argument("--plot")
    args = ap.parse_args(argv)

    if args.mode == "sas":
        if args.display_db:
            from PIL import Image
            grey = np.asarray(Image.open(args.image).convert("F"), dtype=float)
            image = simcal.from_display_db(grey, *args.display_db, grey_max=float(grey.max()))
        else:
            image = simcal.load_image(args.image)
        cal = simcal.calibrate_sas_image(image, args.pixel[0] if len(args.pixel) == 1 else tuple(args.pixel), block=args.block)
        print(cal.report())
        over = cal.overrides()
    else:
        cfg = load_scenario_config(args.scenario)
        bearings = np.radians(np.arange(args.bearings[0], args.bearings[1] + 0.5 * args.bearings[2], args.bearings[2]))
        if args.frames:
            z = np.load(args.frames)
            obs = simcal.observe_frames(z["intensity"], z["ranges"], np.radians(z["bearings_deg"]), levels=args.levels or "relative")
        else:
            with make_source({"type": "container", "path": args.log}) as src:
                records = [r for _, r in zip(range(args.max_pings), src)]
            if args.ranges:
                ranges = np.arange(*args.ranges)
            else:
                rec, c = records[0], records[0].environment.sound_speed
                r0 = 0.5 * c * rec.samples.start_time + 1.0
                r1 = 0.5 * c * (rec.samples.start_time + rec.samples.n_samples / rec.samples.sample_rate - rec.transmit.waveform.duration) - 1.0
                ranges = np.linspace(r0, r1, 120 * 36)
            obs = simcal.observe_records(records, ranges[:len(ranges) // 120 * 120], bearings, levels=args.levels or "absolute")
        cal = simcal.calibrate_fls(obs, cfg, anchor=args.anchor)
        print(cal.report())
        over = cal.overrides()
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
            ax.plot(cal.ranges, cal.observed_db, color="#0b0b0b", lw=2.2, label="log (median over bearings)")
            ax.plot(cal.ranges, cal.fitted_db, color="#eb6834", lw=1.4, ls="--", label=f"fitted sum ({cal.fit_rms_db:.2f} dB RMS)")
            for (name, part), col in zip(cal.parts_db.items(), ("#2a78d6", "#1baf7a", "#eda100", "#8b5fd6")):
                ax.plot(cal.ranges, part, color=col, lw=1.2, label=name)
            ax.set_ylim(cal.observed_db.min() - 15, cal.observed_db.max() + 5)
            ax.set_xlabel("range (m)")
            ax.set_ylabel("power (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False, fontsize=9)
            ax.set_title(f"{args.scenario}: simulated contributions fitted to the log ({cal.levels} levels)", loc="left", fontsize=10)
            fig.savefig(args.plot, dpi=110)
    if args.out:
        Path(args.out).write_text(dump_yaml(over))
        print(f"overrides written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
