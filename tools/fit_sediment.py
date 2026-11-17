#!/usr/bin/env python3
"""Fit the bottom scattering model's roughness and volume parameters to measured
backscattering strength, and print a row for configs/sediments.yaml.

    PYTHONPATH=python python3 tools/fit_sediment.py --data site.csv --sediment fine_sand \\
        --name my_site --source "survey 2026-11, FLS 300 kHz, calibrated" [--out row.yaml] [--plot fit.png]

The data file is CSV with a header: grazing_deg, frequency_hz, bs_db and optionally sigma_db
(the standard error of each value in dB). bs_db is the backscattering strength in dB re 1 m^2
per m^2 (10 log10 of the cross section per unit area and solid angle), calibrated: an error
in its absolute level goes straight into w2 and sigma2.

The density ratio, speed ratio and loss parameter are held: from --sediment (a row of
configs/sediments.yaml, the reflection-only rows included) or from --rho --nu --delta. With
--fit-geoacoustic they are fitted as well (rarely identifiable from backscatter alone).

What the data cannot fix is listed (`unconfirmed_fields` in the row, with the reasons): see
python/gpusonar/sim/sediment_fit.py.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from gpusonar.sim import bottom, sediment_fit


def read_data(path: str):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(line for line in f if line.strip() and not line.lstrip().startswith("#"))]
    if not rows or not {"grazing_deg", "frequency_hz", "bs_db"} <= set(rows[0]):
        raise SystemExit(f"{path}: need a header with grazing_deg, frequency_hz, bs_db[, sigma_db]")
    col = lambda k: np.array([float(r[k]) for r in rows])               # noqa: E731
    return col("grazing_deg"), col("frequency_hz"), col("bs_db"), (col("sigma_db") if "sigma_db" in rows[0] else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV: grazing_deg, frequency_hz, bs_db[, sigma_db]")
    ap.add_argument("--sediment", help="row of the sediment table supplying rho, nu, delta")
    ap.add_argument("--sediment-file", help="another sediment table")
    ap.add_argument("--rho", type=float)
    ap.add_argument("--nu", type=float)
    ap.add_argument("--delta", type=float)
    ap.add_argument("--fit-geoacoustic", action="store_true", help="fit rho, nu, delta too (start from the held values)")
    ap.add_argument("--sound-speed", type=float, default=1500.0, help="water sound speed at the bottom, m/s")
    ap.add_argument("--sigma-db", type=float, help="measurement error of every value, dB (overrides a sigma_db column)")
    ap.add_argument("--roughness-model", default="perturbation", choices=bottom.ROUGHNESS_MODELS)
    ap.add_argument("--name", default="fitted_sediment", help="name of the output row")
    ap.add_argument("--source", required=True, help="where the data come from (goes into the row's status and source)")
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--out", help="write the row here (default: print it)")
    ap.add_argument("--plot", help="write a figure of data and fit")
    args = ap.parse_args(argv)

    grazing, frequency, bs, sigma = read_data(args.data)
    if args.sigma_db is not None:
        sigma = args.sigma_db
    geo = {}
    if args.sediment:
        geo = bottom.sediment_reflection(args.sediment, args.sediment_file)
    for k in ("rho", "nu", "delta"):
        if getattr(args, k) is not None:
            geo[k] = getattr(args, k)
    if set(geo) != {"rho", "nu", "delta"}:
        raise SystemExit("give --sediment, or --rho --nu --delta")
    result = sediment_fit.fit(grazing, frequency, bs, sound_speed=args.sound_speed, sigma_db=sigma,
                              fit_geoacoustic=args.fit_geoacoustic, roughness_model=args.roughness_model,
                              n_starts=args.starts, **geo)

    print(f"{result.n_points} values, {result.grazing_deg[0]:g} to {result.grazing_deg[1]:g} degrees grazing, "
          f"{', '.join(f'{f / 1e3:g}' for f in result.frequencies)} kHz; rms residual {result.rms_residual_db:.2f} dB "
          f"(measurement error used: {result.sigma_db:.2f} dB)", file=sys.stderr)
    print(f"roughness share of the intensity over the data: {result.roughness_share[0]:.1%} to {result.roughness_share[1]:.1%}", file=sys.stderr)
    for k in result.fitted:
        err = f"x/ {result.factor(k):.3g}" if k in sediment_fit.LOG_KEYS else f"+- {result.std[k]:.3g}"
        mark = "   UNCONSTRAINED" if k in result.unconstrained else ""
        print(f"  {k:<7s} {result.params[k]:.6g}  ({err}, 1 sigma){mark}", file=sys.stderr)
    for note in result.notes:
        print(f"  note: {note}", file=sys.stderr)
    row = result.yaml_row(args.name, args.source)
    if args.out:
        with open(args.out, "w") as f:
            f.write(row)
        print(f"row written to {args.out}", file=sys.stderr)
    else:
        print(row)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        frequency = np.broadcast_to(frequency, grazing.shape)
        for f in result.frequencies:
            m = frequency == f
            line, = ax.plot(grazing[m], bs[m], "o", label=f"data, {f / 1e3:g} kHz")
            a = np.linspace(grazing[m].min(), grazing[m].max(), 200)
            ax.plot(a, result.model_db(a, np.full_like(a, f)), "-", color=line.get_color())
        ax.set_xlabel("grazing angle, degrees")
        ax.set_ylabel("backscattering strength, dB")
        ax.set_title(f"{args.name}: fit ({result.roughness_model}), rms {result.rms_residual_db:.2f} dB")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
