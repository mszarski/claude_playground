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

  --roughness-model M       small_slope (the default, the simulator's own since D-86; A10 of
                            docs/references/CONFORMANCE.md), perturbation (the default before A10:
                            name it to reproduce earlier fits; not valid above about 60-70 degrees
                            grazing), kirchhoff, composite or combined. A small-slope fit takes
                            seconds (a Kirchhoff table per trial gamma2), a perturbation fit less
                            than one. The row records the model used.

Error bars, priors and the extended report (opt-in; without these options, and with
--roughness-model perturbation, the output is what it was before them, bit for bit):

  --independent-samples N   each value is the dB of an intensity average of N independent samples
                            (or `column`: an n_samples column per value); its error is then
                            10 log10(1 + 1/sqrt(N)) dB (Jackson and Richardson 2007, App. G.2.3:
                            0.97 dB at N = 16, 0.50 at 67, 0.26 at 260) and the fit is weighted by it.
                            N is (time window / resolution) x beams x pings, the resolution being the
                            pulse length (CW) or 1/bandwidth (FM), counting only beams that do not
                            overlap and pings that see a different seafloor (gpusonar.measure).
                            A value must be 10 log10 of a mean INTENSITY, never a mean of dB values
                            (that is 2.5 dB low for Rayleigh speckle).
  --sample-error gamma      instead of the book's +1 sigma, the standard deviation of the dB of an
                            N-sample mean (1.10 dB at 16) and its bias (-0.14 dB at 16) removed (ours)
  --prior-bounds PHI        search limits from the reference notes for a site of mean grain size PHI
                            (sim/sediment_fit.py PRIOR_BOUNDS, W2_SPREAD_DB); a parameter that ends on
                            a bound is set by the prior and is listed as unconstrained
  --site-spread K           with --fit-geoacoustic and --prior-bounds: rho, nu, delta held to within K
                            times their within-site coefficients of variation (App. C)
  --hold KEY=VALUE          hold a parameter instead of fitting it (repeatable)
  --report-covariance       the extended report alone: covariance of the fitted variables, the
                            error of the predicted strength, whether the roughness models differ

How many samples a stated error needs, and how a fit is judged: docs/tasks/REAL_DATA.md Sec. 5.
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


def read_column(path: str, name: str):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(line for line in f if line.strip() and not line.lstrip().startswith("#"))]
    if name not in rows[0]:
        raise SystemExit(f"{path}: no {name} column")
    return np.array([float(r[name]) for r in rows])


def parse_hold(items) -> dict:
    out = {}
    for item in items or ():
        key, eq, value = item.partition("=")
        if not eq:
            raise SystemExit(f"--hold {item}: use KEY=VALUE")
        out[key.strip()] = float(value)
    return out


def print_extended(result, grazing, frequency, roughness_model: str, file) -> None:
    """The opt-in report: what bounded the fit, the covariance of the fitted variables (log10 for
    w2, sigma2, delta), the error of the predicted strength over the data, the roughness models."""
    if result.independent_samples is not None:
        lo, hi = result.independent_samples
        print(f"independent samples per value: {lo:g}" + (f" to {hi:g}" if hi != lo else "") +
              f" ({result.sample_error} error rule)", file=file)
    if result.held:
        print("held: " + ", ".join(f"{k} = {v:g}" for k, v in result.held.items()), file=file)
    if result.bounds is not None:
        print(f"bounds ({result.bounds_source}):", file=file)
        for k, (lo, hi) in result.bounds.items():
            print(f"  {k:<7s} {lo:.4g} to {hi:.4g}" + ("" if k in result.fitted else "  (not fitted)"), file=file)
    names = [f"log10 {k}" if k in sediment_fit.LOG_KEYS else k for k in result.fitted]
    print("covariance of the fitted variables (" + ", ".join(names) + "):", file=file)
    for row in np.atleast_2d(result.covariance):
        print("  " + "  ".join(f"{v:11.4g}" for v in row), file=file)
    pred = result.prediction_std_db(grazing, np.broadcast_to(frequency, np.shape(grazing)))
    print(f"predicted strength, 1 sigma over the data points: {pred.min():.3g} to {pred.max():.3g} dB", file=file)
    spread = sediment_fit.roughness_model_spread_db(result.params, grazing, np.broadcast_to(frequency, np.shape(grazing)),
                                                    result.sound_speed)
    if np.isnan(spread):
        print(f"roughness models not compared: gamma2 = {result.params['gamma2']:g} is outside (2, 4), where only "
              f"perturbation is defined (this fit: {roughness_model})", file=file)
    else:
        print(f"roughness models with the fitted parameters differ by at most {spread:.3g} dB over the data "
              f"(this fit: {roughness_model})", file=file)


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
    ap.add_argument("--roughness-model", default=sediment_fit.DEFAULT_ROUGHNESS_MODEL, choices=bottom.ROUGHNESS_MODELS,
                    help=f"roughness model of the fit (default {sediment_fit.DEFAULT_ROUGHNESS_MODEL}; "
                         "perturbation was the default before A10)")
    ap.add_argument("--name", default="fitted_sediment", help="name of the output row")
    ap.add_argument("--source", required=True, help="where the data come from (goes into the row's status and source)")
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--out", help="write the row here (default: print it)")
    ap.add_argument("--plot", help="write a figure of data and fit")
    ap.add_argument("--independent-samples", metavar="N|column",
                    help="independent intensity samples behind each value: a number, or 'column' for an n_samples column")
    ap.add_argument("--sample-error", choices=("book", "gamma"), default="book",
                    help="with --independent-samples: the book's 10 log10(1 + 1/sqrt(N)) (default) or the gamma statistics")
    ap.add_argument("--prior-bounds", type=float, metavar="PHI", help="search limits from the notes for mean grain size PHI")
    ap.add_argument("--site-spread", type=float, metavar="K",
                    help="with --fit-geoacoustic and --prior-bounds: rho, nu, delta within K within-site CVs")
    ap.add_argument("--hold", action="append", metavar="KEY=VALUE", help="hold a parameter (repeatable)")
    ap.add_argument("--report-covariance", action="store_true", help="print the extended report")
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
    opt = {}                                    # the opt-in arguments (G-9); empty: the fit as before them
    if args.independent_samples is not None:
        if args.sigma_db is not None:
            raise SystemExit("give --sigma-db or --independent-samples, not both")
        n = read_column(args.data, "n_samples") if args.independent_samples == "column" else float(args.independent_samples)
        sigma = None
        opt.update(independent_samples=n, sample_error=args.sample_error)
    if args.prior_bounds is not None:
        bounds = sediment_fit.prior_bounds(args.prior_bounds)
        source = f"prior, mean grain size {args.prior_bounds:g} phi (sediment_fit.prior_bounds)"
        if args.site_spread is not None:
            if not args.fit_geoacoustic:
                raise SystemExit("--site-spread needs --fit-geoacoustic")
            bounds.update(sediment_fit.site_bounds(geo["rho"], geo["nu"], geo["delta"], args.prior_bounds, args.site_spread))
            source += f"; rho, nu, delta within {args.site_spread:g} within-site CVs (sediment_fit.site_bounds)"
        opt.update(bounds=bounds, bounds_source=source)
    elif args.site_spread is not None:
        raise SystemExit("--site-spread needs --prior-bounds PHI (the class)")
    if args.hold:
        opt["hold"] = parse_hold(args.hold)
    extended = bool(opt) or args.report_covariance
    result = sediment_fit.fit(grazing, frequency, bs, sound_speed=args.sound_speed, sigma_db=sigma,
                              fit_geoacoustic=args.fit_geoacoustic, roughness_model=args.roughness_model,
                              n_starts=args.starts, **geo, **opt)

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
    if extended:
        print_extended(result, grazing, frequency, args.roughness_model, file=sys.stderr)
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
