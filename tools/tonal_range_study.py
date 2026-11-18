#!/usr/bin/env python3
"""How far away does the listening array detect a tonal, and a propeller's modulation?

MODEL FIGURES, NOT MEASUREMENTS. For a sonar's array and the `passive.tonal` settings this
prints the range at which

- a tonal of a given source level (dB re 1 uPa at 1 m) reaches the Goertzel detection
  threshold in its bin:   SNR = SL - TL - NL0 - 10 log10(1.5 / block_s) + AG
- a broadband source of a given spectrum level (dB re 1 uPa^2/Hz at 1 m, flat over the
  band) whose amplitude is modulated with a given index reaches the DEMON threshold:
  line / background = 1 + m^2 B T rows / 6,  m = 2 depth r / (1 + r),  r = (SLs - TL - NL0) x AG

(gpusonar/tonal.py: line_snr_db, demon_snr_db, detection_ranges; the two expressions are
checked against the analyzer on synthetic signals in tests/python/test_tonal.py). One-way
loss TL by gpusonar/sonar_equation.py: a spreading law plus absorption. Ambient noise NL0
from --noise-db, or the Wenz-type fit of gpusonar/sim/noise.py for --wind and --shipping.
Noise uncorrelated between hydrophones; no multipath, no directional noise, no fluctuation
of the source. The source levels are yours to give: none is assumed.

Usage: tools/tonal_range_study.py --tonal-sl 130 140 150 --broadband-sl 110 120 130 \\
           [--sonar popoto_long_range] [--tonal-hz 4700] [--depth 0.1] [--wind 5] [--config deploy.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import elevation, tonal  # noqa: E402
from gpusonar.sim import make_pose  # noqa: E402
from gpusonar.sim.noise import wenz_spectrum_db  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sonar", default="popoto_long_range")
    ap.add_argument("--tonal-sl", type=float, nargs="*", default=[], help="tonal source levels, dB re 1 uPa at 1 m")
    ap.add_argument("--tonal-hz", type=float, help="the tonal's frequency (default: the middle of the sonar's band)")
    ap.add_argument("--broadband-sl", type=float, nargs="*", default=[], help="broadband spectrum levels, dB re 1 uPa^2/Hz at 1 m")
    ap.add_argument("--depth", type=float, default=0.1, help="amplitude modulation index of the line looked for (default 0.1)")
    ap.add_argument("--noise-db", type=float, help="ambient noise spectrum level at the array, dB re 1 uPa^2/Hz")
    ap.add_argument("--wind", type=float, default=5.0, help="m/s, for the ambient noise fit when --noise-db is not given")
    ap.add_argument("--shipping", type=float, default=0.5, help="0..1, for the ambient noise fit")
    ap.add_argument("--spreading", type=float, default=20.0, help="TL = this x log10(R) + absorption (20: spherical)")
    ap.add_argument("--config", help="a deployment file: its passive.tonal section is used instead of the defaults")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not args.tonal_sl and not args.broadband_sl:
        ap.error("give --tonal-sl and/or --broadband-sl: no source level is assumed")
    sonar = SonarModel.load(args.sonar)
    cfg = None
    if args.config:
        from gpusonar.config import load_yaml
        cfg = ((load_yaml(args.config).get("passive") or {}).get("tonal"))
    f_line = args.tonal_hz or sonar.center_frequency
    noise = args.noise_db if args.noise_db is not None else float(wenz_spectrum_db(sonar.center_frequency, args.wind, args.shipping))
    rows = len(elevation.find_rows(sonar.make_record(make_pose())))
    out = tonal.detection_ranges(sonar, noise, tonal_levels_db=args.tonal_sl, tonal_hz=f_line, broadband_levels_db=args.broadband_sl,
                                 depth=args.depth, config=cfg, rows=rows, spreading=args.spreading)
    out.update(sonar=sonar.name, noise_spectrum_db=noise, spreading=args.spreading, model_figures=True)
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    c = tonal.configure(cfg if cfg is not None else {"goertzel": {"lines": [f_line]}, "demon": {}})
    g, d = c["goertzel"] or tonal.GOERTZEL, c["demon"] or tonal.DEMON
    print(f"MODEL FIGURES (sonar equation), not measurements. {sonar.name}: {sonar.n_elements} hydrophones in {rows} row(s), "
          f"array gain {out['array_gain_db']:.1f} dB ({c['window']} weighting, uncorrelated noise)")
    print(f"noise {noise:.1f} dB re 1 uPa^2/Hz" + ("" if args.noise_db is not None else f" (fit: wind {args.wind} m/s, shipping {args.shipping})")
          + f"; loss {args.spreading:g} log10(R) + absorption")
    if out["tonal"]:
        print(f"\ntonal at {f_line:.0f} Hz, Goertzel block {g['block_s']} s (bin noise bandwidth {1.5 / g['block_s']:.2f} Hz), "
              f"detected at {g['min_snr_db']} dB in the bin:")
        print("  source level   SNR at 1 km   detection range")
        for row in out["tonal"]:
            print(f"  {row['source_level_db']:8.1f} dB   {row['snr_at_1km_db']:8.1f} dB   {row['range_m'] / 1000.0:10.2f} km")
    if out["demon"]:
        lo, hi = out["band_hz"]
        print(f"\nbroadband source over {lo:.0f}-{hi:.0f} Hz, amplitude modulation index {args.depth}; DEMON segments {d['segment_s']} s, "
              f"{d['n_average']} averaged, pfa {d['pfa']:g}: line detected {out['demon'][0]['threshold_db']:.1f} dB above the background:")
        print("  spectrum level   line at 1 km   detection range")
        for row in out["demon"]:
            print(f"  {row['spectrum_level_db']:8.1f} dB     {row['snr_at_1km_db']:8.1f} dB   {row['range_m'] / 1000.0:10.2f} km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
