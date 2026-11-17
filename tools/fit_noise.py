#!/usr/bin/env python3
"""Fit a noise model to a passive (no transmit) multichannel recording and print the fit report.

    tools/fit_noise.py RECORDING --out model.h5 [--nfft 1024] [--smooth 1] [--tail auto|gaussian|k|sas]
                       [--channels 0,1,2] [--keep-transients] [--keep-tones] [--check]

RECORDING: a .wav, a WAV sidecar .yaml, or a record container .h5 (docs/RECORD_FORMAT.md).
The model file holds the cross-spectral matrix of the array on a frequency grid, the tail
parameters and the lines taken out; not the recording. A scenario names it with

    receiver: {ambient_noise: {model: fitted, file: model.h5}}

The report: spectrum level against frequency, coherence length, tail parameter, what was
excluded (transients, periodic bursts, lines), and how accurate the estimates are for this
record length. `--check` generates noise from the model, fits that, and prints how far the
second fit is from the first (spectrum, coherence, tail): the round trip on your own data.
Method and assumptions: python/gpusonar/sim/noise_fit.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np  # noqa: E402

from gpusonar.sim import noise_fit as nf  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--tail", default="auto", choices=["auto", "gaussian", "k", "sas"])
    ap.add_argument("--channels", default=None, help="comma-separated channel indices to keep")
    ap.add_argument("--keep-transients", action="store_true", help="fit everything, flagged blocks included")
    ap.add_argument("--keep-tones", action="store_true", help="leave lines in the spectrum")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    channels = None if args.channels is None else [int(c) for c in args.channels.split(",")]
    rec = nf.load_recording(args.recording, channels=channels)
    model = nf.fit(rec, nfft=args.nfft, smooth=args.smooth, tail=args.tail,
                   transient_threshold=None if args.keep_transients else 8.0,
                   tone_threshold_db=np.inf if args.keep_tones else None)
    model.save(args.out)
    print(nf.format_report(model))
    print(f"Model written to {args.out} ({Path(args.out).stat().st_size / 1e3:.1f} kB; the recording is "
          f"{sum(s.nbytes for s in rec.segments) / 1e6:.1f} MB in memory)")
    if args.check:
        n = int(min(rec.duration, 120.0) * rec.sample_rate)
        again = nf.fit(nf.Recording(segments=[model.generate(n, np.random.default_rng(args.seed))], sample_rate=rec.sample_rate,
                                    center_frequency=rec.center_frequency, positions=rec.positions), nfft=args.nfft, smooth=args.smooth)
        f, p0 = model.psd()
        _, p1 = again.psd()
        ok = p0.mean(axis=1) > 1e-4 * p0.mean(axis=1).max()
        d_db = 10.0 * np.log10(p1[ok] / p0[ok])
        dc = np.abs(again.coherence() - model.coherence())[1:model.nfft // 2][ok[: model.nfft // 2 - 1]] if not model.is_complex else np.abs(again.coherence() - model.coherence())
        print(f"Round trip ({n / rec.sample_rate:.1f} s regenerated and fitted again): spectrum {np.sqrt(np.mean(d_db**2)):.2f} dB rms, "
              f"{np.abs(d_db).max():.2f} dB worst; coherence {np.sqrt(np.mean(dc**2)):.3f} rms (estimator floor {again.report['coherence_floor']:.3f}); "
              f"tail {model.family} {model.tail_parameter:.2f} -> {again.family} {again.tail_parameter:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
