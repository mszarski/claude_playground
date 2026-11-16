#!/usr/bin/env python3
"""Look at ping records from a container file or a tap directory.

    tools/inspect_tap.py recording.h5                       # numbers per channel, first ping
    tools/inspect_tap.py taps/ --stage prepare --ping 2     # one stage of a tapped run
    tools/inspect_tap.py recording.h5 --plot ping.png       # waveform, spectrum, spectrogram, all channels
    tools/inspect_tap.py recording.h5 --wav ping.wav --npy ping.npy
    tools/inspect_tap.py taps/                              # what the directory holds

Works the same on simulated and recorded data: both are ping records.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import taps  # noqa: E402
from gpusonar.sources import make_source  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--stage")
    ap.add_argument("--ping", type=int, default=0)
    ap.add_argument("--channel", type=int, default=0, help="channel for the waveform and spectrogram panels")
    ap.add_argument("--plot")
    ap.add_argument("--wav")
    ap.add_argument("--npy")
    args = ap.parse_args(argv)
    path = Path(args.path)
    if path.is_dir():
        m = taps.manifest(path)
        if not args.stage:
            for e in m:
                print(json.dumps(e))
            return 0
        records = taps.read_tap(path, args.stage)
    else:
        with make_source({"type": "container", "path": str(path)}) as src:
            records = list(src)
    rec = records[args.ping]
    s = rec.samples
    x = s.array
    w = rec.transmit.waveform
    print(f"{len(records)} pings; ping {args.ping}: {s.n_channels} channels x {s.n_samples} samples at {s.sample_rate:.6g} Hz, "
          f"{str(s.kind).split('.')[-1]}, starts {s.start_time * 1e3:.3f} ms after transmit; "
          f"pulse {w.f_start / 1e3:.4g}-{w.f_stop / 1e3:.4g} kHz, {w.duration * 1e3:.4g} ms; array {rec.metadata.array_id}")
    st = taps.stats(rec)
    print("channel      rms         peak        clipped   strongest line (kHz)")
    for i, c in enumerate(st):
        print(f"{i:5d}  {c['rms']:11.4g} {c['peak']:11.4g}  {100 * c['clipped_fraction']:7.3f}%  {c['strongest_line_hz'] / 1e3:10.3f}")
    if args.wav:
        print("wav:", json.dumps(taps.export_wav(rec, args.wav)))
    if args.npy:
        np.save(args.npy, x)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = (s.start_time + np.arange(s.n_samples) / s.sample_rate) * 1e3
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        ch = x[args.channel]
        axes[0, 0].plot(t, ch.real, lw=0.6, color="#2a78d6", label="real")
        if np.iscomplexobj(ch):
            axes[0, 0].plot(t, np.abs(ch), lw=0.8, color="#eb6834", label="magnitude")
        axes[0, 0].set_title(f"channel {args.channel}", loc="left")
        axes[0, 0].set_xlabel("time after transmit (ms)")
        axes[0, 0].legend(frameon=False)
        f = np.fft.fftshift(np.fft.fftfreq(s.n_samples, 1.0 / s.sample_rate)) if np.iscomplexobj(x) else np.fft.rfftfreq(s.n_samples, 1.0 / s.sample_rate)
        spec = np.fft.fftshift(np.abs(np.fft.fft(x, axis=1)) ** 2, axes=1) if np.iscomplexobj(x) else np.abs(np.fft.rfft(x, axis=1)) ** 2
        offset = s.center_frequency if np.iscomplexobj(x) else 0.0
        axes[0, 1].plot((f + offset) / 1e3, 10 * np.log10(spec.mean(axis=0) / spec.mean(axis=0).max() + 1e-12), color="#0b0b0b", lw=0.8)
        axes[0, 1].set_title("spectrum, mean over channels (dB re peak)", loc="left")
        axes[0, 1].set_xlabel("frequency (kHz)")
        axes[0, 1].set_ylim(-90, 3)
        env = 20 * np.log10(np.abs(x) / np.abs(x).max() + 1e-9)
        m = axes[1, 0].imshow(env, aspect="auto", extent=[t[0], t[-1], s.n_channels - 0.5, -0.5], cmap="gray", vmin=-70, vmax=0, interpolation="nearest")
        fig.colorbar(m, ax=axes[1, 0], label="dB re peak")
        axes[1, 0].set_title("all channels, magnitude", loc="left")
        axes[1, 0].set_xlabel("time after transmit (ms)")
        axes[1, 0].set_ylabel("channel")
        n_fft = int(min(1024, 2 ** np.floor(np.log2(max(s.n_samples // 16, 16)))))
        axes[1, 1].specgram(ch, NFFT=n_fft, Fs=s.sample_rate / 1e3, noverlap=n_fft // 2, cmap="gray",
                            sides="twosided" if np.iscomplexobj(ch) else "onesided")
        axes[1, 1].set_title(f"spectrogram, channel {args.channel}" + (" (kHz about the carrier)" if np.iscomplexobj(ch) else " (kHz)"), loc="left")
        axes[1, 1].set_xlabel("time in the record (ms)")
        for ax in axes.ravel()[:2]:
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{path}" + (f", after {args.stage}" if args.stage else "") + f", ping {args.ping}")
        fig.savefig(args.plot, dpi=100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
