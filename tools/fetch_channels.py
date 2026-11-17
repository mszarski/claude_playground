#!/usr/bin/env python3
"""Download measured underwater acoustic channels from the Underwater Acoustic Channel
Repository (UACR) on Zenodo into a directory, with checksums.

    tools/fetch_channels.py <directory> [--set validation|purple|noise|all-small] [--files a.mat,b.mat] [--list]
    then:  GPUSONAR_UACR_DIR=<directory> tools/channel_validation.py
           GPUSONAR_UACR_DIR=<directory> pytest tests/python/test_channel_replay.py

Dataset: Z. Li, M. Chitre, D. A. Cuji, J. Preisig, A. C. Singer, M. Stojanovic, P. van Walree,
"Underwater Acoustic Channel Repository", version 1.0, Zenodo, 2026,
doi:10.5281/zenodo.21287414. Licence: Creative Commons Attribution 4.0 (CC BY 4.0): cite the
dataset and the paper (arXiv:2609.03207) when you use it. The files are NOT part of this
repository and must not be committed to it (the `purple` set alone is 1.6 GB).

The MD5 sums below are those Zenodo publishes for record 21287414 (read from
https://zenodo.org/api/records/21287414); a file whose sum differs is rejected.
"""

import argparse
import hashlib
import pathlib
import sys
import urllib.request

RECORD = "21287414"
BASE = f"https://zenodo.org/api/records/{RECORD}/files"

# name: (size in bytes, md5)
FILES = {
    "purple_1.mat": (712812521, "5a4261d9c51e35c322c92db3c497c793"),      # SPACE'08, 60 m, cross array (32)
    "purple_3.mat": (534541341, "a449f2256ccf59f0295bb8ce21c24590"),      # SPACE'08, 200 m, vertical (24)
    "purple_5.mat": (354079789, "f3f6c16a4139d2dc258a7cad295c75d6"),      # SPACE'08, 1 km, vertical (12)
    "purple_6.mat": (712812521, "2694d019f694a1e11e7a2799df426277"),      # the same systems, second recording
    "purple_8.mat": (534541341, "f6f3f3befbbfaba44d8bbe9fd30069c5"),
    "purple_10.mat": (354079789, "edd40719f8df6e225462d4f8eaaf9a2f"),
    "purple_11.mat": (712812521, "ed09ab649b89f127fe0b715a541c359b"),     # third recording
    "purple_13.mat": (534541341, "8c7030d26d33d5586e662e1a5965206b"),
    "purple_15.mat": (354079789, "3e19b310d857c0e9ce9c8912bf156473"),
    "red_1.mat": (191566442, "76ee45172cd0bf5deb4a7bfb04907884"),         # Singapore, 25 kHz, drifting
    "yellow_4.mat": (360679714, "538d4374097df37d13f952947c9ff6db"),      # KAM'11, 7 km
    "purple_noise_1.mat": (580006, "5300e408a09cf0662ab3f5ceb0f83b11"),
    "purple_noise_3.mat": (306031, "d340e5e50de3c3505c0d9a44fd579301"),
    "purple_noise_5.mat": (117081, "1d162196563dd4f392dd848675e81ec5"),
    "red_noise.mat": (5176, "0c525d72db00608ff77f3b8537bdf06f"),
    "blue_noise.mat": (155097, "039ffcd3d74cb87f418b052933e0b750"),
}
SETS = {
    "validation": ["purple_1.mat", "purple_3.mat", "purple_5.mat"],
    "purple": ["purple_1.mat", "purple_3.mat", "purple_5.mat", "purple_6.mat", "purple_8.mat", "purple_10.mat",
               "purple_11.mat", "purple_13.mat", "purple_15.mat"],
    "noise": ["purple_noise_1.mat", "purple_noise_3.mat", "purple_noise_5.mat", "red_noise.mat", "blue_noise.mat"],
    "all-small": ["purple_5.mat", "red_1.mat", "yellow_4.mat"],
}


def md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?")
    ap.add_argument("--set", default="validation", choices=sorted(SETS))
    ap.add_argument("--files", default=None, help="comma-separated file names instead of a set")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list or not args.directory:
        for name, (size, digest) in FILES.items():
            print(f"{name:22s} {size / 1e6:9.1f} MB  md5 {digest}")
        print("sets:", {k: len(v) for k, v in SETS.items()})
        return 0 if args.list else 2
    names = args.files.split(",") if args.files else SETS[args.set]
    out = pathlib.Path(args.directory)
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name in names:
        if name not in FILES:
            print(f"{name}: not in the checksum table of this script")
            bad += 1
            continue
        size, digest = FILES[name]
        dst = out / name
        if dst.exists() and dst.stat().st_size == size and md5(dst) == digest:
            print(f"have {dst}")
            continue
        print(f"fetching {name} ({size / 1e6:.0f} MB)")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(f"{BASE}/{name}/content", tmp)
        got = md5(tmp)
        if got != digest:
            print(f"{name}: MD5 {got} is not the published {digest}; kept as {tmp.name}")
            bad += 1
            continue
        tmp.replace(dst)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
