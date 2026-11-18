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

FILES = {   # every file of Zenodo record 21287414: name: (bytes, md5)
    "black.mat": (161710585, "16307b935cac01d02902e04a265f08cd"),
    "blue_1.mat": (206132594, "6776481ad636e40b1d6038fa37121fb7"),
    "blue_2.mat": (206132594, "60cc9bf8a5e9e64b75b5fcaf6b196993"),
    "blue_3.mat": (206132594, "4476d3f79d7eeb086574e0b3a3b74050"),
    "blue_4.mat": (206132594, "eaedef33a4bab70ffa6afaa835a897d1"),
    "blue_5.mat": (206132594, "262be5cf3cb79ede4cb04b56c4884a6f"),
    "blue_6.mat": (206132594, "be0b1923ad4d4aab5b6730bf5c3fb4d2"),
    "blue_7.mat": (206132594, "3ee1ab5b0862193fd470376601e6ff07"),
    "blue_8.mat": (206132594, "b15d1853c6808c6cb4f7fdb13a4aa1ea"),
    "blue_9.mat": (206132594, "3c78f02efccd985e5892176ecc58726e"),
    "blue_10.mat": (206132602, "0c5ee94f97430f1c412e3ba576fc7327"),
    "blue_11.mat": (206132602, "01d87d09cb1fea67bbf1eb9027100b83"),
    "blue_12.mat": (206132602, "6b6829dc9e7ecb4cc17b99a3ecfbdf2a"),
    "blue_13.mat": (206132602, "073b8d3dcdfc674fe930b18ef92b7082"),
    "blue_14.mat": (206132602, "249e57eac6cceaec64831269e489b29d"),
    "blue_15.mat": (206132602, "38f35c9185aace525296d817f5ac9d68"),
    "blue_16.mat": (206132602, "da08ff1d6635e95ebff0403a3510d566"),
    "blue_17.mat": (206132602, "8d57a6daceab00ce794bbc1846a341ad"),
    "blue_18.mat": (206132602, "e0da8f06b03ee4ccabd2b0f9b7f2242a"),
    "blue_19.mat": (206132602, "f778e04f493cd33b4f3722456cb8dc63"),
    "blue_20.mat": (206132602, "e10357d16a86c5ee996ca2710dcd32fe"),
    "brown.mat": (660486317, "c653cdf5f94b84715d6ac357f158aa82"),
    "green.mat": (1094029080, "7300ba02d788a551a3b59e164bfe59fb"),
    "pink_1.mat": (1496605964, "b276fb275a41fc75a286b30f3eafa999"),
    "pink_2.mat": (1496605964, "616da90fb3a8c5cf9633654f378aef08"),
    "pink_3.mat": (1440393938, "876e935e057e36391f5250e7dd2ac15e"),
    "purple_1.mat": (712812521, "5a4261d9c51e35c322c92db3c497c793"),
    "purple_2.mat": (712812521, "0707b7d09df3c592d4462135c6bc2019"),
    "purple_3.mat": (534541341, "a449f2256ccf59f0295bb8ce21c24590"),
    "purple_4.mat": (534541341, "ac50e6b05fad13729821d4835a9d0a2b"),
    "purple_5.mat": (354079789, "f3f6c16a4139d2dc258a7cad295c75d6"),
    "purple_6.mat": (712812521, "2694d019f694a1e11e7a2799df426277"),
    "purple_7.mat": (712812521, "1aabf286a99ec62d3264a1bd0c2c87a2"),
    "purple_8.mat": (534541341, "f6f3f3befbbfaba44d8bbe9fd30069c5"),
    "purple_9.mat": (534541341, "9a4e8018c447f3af521b8c62119f1fd3"),
    "purple_10.mat": (354079789, "edd40719f8df6e225462d4f8eaaf9a2f"),
    "purple_11.mat": (712812521, "ed09ab649b89f127fe0b715a541c359b"),
    "purple_12.mat": (712812521, "fb99c1a26e085791ae84bc8ea2a111b3"),
    "purple_13.mat": (534541341, "8c7030d26d33d5586e662e1a5965206b"),
    "purple_14.mat": (534541341, "b1baaa8d865083e17457488621d30d6d"),
    "purple_15.mat": (354079789, "3e19b310d857c0e9ce9c8912bf156473"),
    "red_1.mat": (191566442, "76ee45172cd0bf5deb4a7bfb04907884"),
    "red_2.mat": (191566442, "cb1b4566f5e9db67b93c072dbce8db06"),
    "red_3.mat": (191566442, "befb441590ddafc4383746cc0526d4b6"),
    "red_4.mat": (191566442, "bbab171a6af14632944d1fee157cde86"),
    "yellow_1.mat": (821046493, "55828835d6a1aed660f749b9c53e594e"),
    "yellow_2.mat": (821046493, "5c399ecf8f7001fec5248bbd36ac29d0"),
    "yellow_3.mat": (984922035, "b38c9ba7351ee735b00e7b2b4332693b"),
    "yellow_4.mat": (360679714, "538d4374097df37d13f952947c9ff6db"),
    "yellow_5.mat": (834576462, "281f2caded81d7cfbfbac2a60baaca1e"),
    "yellow_6.mat": (394887977, "a03482e6d624d0de3f19a26db64c86ef"),
    "black_noise.mat": (104793, "75f955908bfa7bf3261b0f855e8b0bbe"),
    "blue_noise.mat": (155097, "039ffcd3d74cb87f418b052933e0b750"),
    "green_noise.mat": (7416, "bfe52230fb115a1007fe68993f3f9088"),
    "pink_noise_1.mat": (614833, "69d9eea390c0e07c23931fe6c62fc433"),
    "pink_noise_2.mat": (614833, "5476a608fc72cbd6f2e0a8717bd7aa3f"),
    "pink_noise_3.mat": (514576, "dd85b83304be903ef9890f2ebdb9df9d"),
    "purple_noise_1.mat": (580006, "5300e408a09cf0662ab3f5ceb0f83b11"),
    "purple_noise_2.mat": (580006, "0091fcdb5677a954eebe3e3c96c7eb7b"),
    "purple_noise_3.mat": (306031, "d340e5e50de3c3505c0d9a44fd579301"),
    "purple_noise_4.mat": (306031, "00afbabffe2082ae09d03974fac9bcf1"),
    "purple_noise_5.mat": (117081, "1d162196563dd4f392dd848675e81ec5"),
    "red_noise.mat": (5176, "0c525d72db00608ff77f3b8537bdf06f"),
    "yellow_noise_1.mat": (614833, "c65a2596ff7c6a714e9750c64a02deb8"),
    "yellow_noise_2.mat": (614833, "645eeab85da7401c7b4b390fe5fb15e2"),
    "yellow_noise_3.mat": (313722, "8300c8384ad00efe679c1c38530e5200"),
    "yellow_noise_4.mat": (313722, "f8207560e1c689e56b36f57d525d2f7f"),
}
SETS = {
    "validation": ["purple_1.mat", "purple_3.mat", "purple_5.mat"],
    "purple": ["purple_1.mat", "purple_3.mat", "purple_5.mat", "purple_6.mat", "purple_8.mat", "purple_10.mat",
               "purple_11.mat", "purple_13.mat", "purple_15.mat"],
    # What tools/channel_validation.py --sites all uses (5.6 GB): SPACE'08, Norway'24, Mariana, Singapore'24, KAM'11, MACE'10.
    "all-sites": ["purple_1.mat", "purple_3.mat", "purple_5.mat", "purple_6.mat", "purple_8.mat", "purple_10.mat", "purple_11.mat",
                  "purple_13.mat", "green.mat", "black.mat", "red_1.mat", "red_2.mat", "yellow_1.mat", "yellow_4.mat", "blue_1.mat"],
    "noise": ["purple_noise_1.mat", "purple_noise_3.mat", "purple_noise_5.mat", "red_noise.mat", "blue_noise.mat"],
    "all-small": ["purple_5.mat", "red_1.mat", "yellow_4.mat"],
}


# The Watermark benchmark (FFI): "Free download" at https://www.ffi.no/en/research/watermark ; the archive
# carries no licence text. Cite van Walree, Socheleau, Otnes, Jenserud, IEEE J. Ocean. Eng. 42 (2017) 1007-1018.
# MD5 as downloaded on 2026-11-18 (FFI publishes none).
WATERMARK = ("https://www.ffi.no/en/research/watermark/_/attachment/download/"
             "cc46828f-0078-46a1-bf77-8cee63b50056:f3b02654a06b2405fa042a5d0a05ce1b77305e40/WatermarkV1.zip",
             770323858, "cd3478cb142a2f56a0feec407e7f3e05")


def fetch_watermark(out: pathlib.Path) -> int:
    """Download WatermarkV1.zip and unpack its channel files (NOF1, NCS1, BCH1, KAU1, KAU2) under `out`/Watermark."""
    import zipfile
    url, size, digest = WATERMARK
    dst = out / "WatermarkV1.zip"
    if not (dst.exists() and dst.stat().st_size == size and md5(dst) == digest):
        print(f"fetching WatermarkV1.zip ({size / 1e6:.0f} MB)")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research client)"})
        with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
            for block in iter(lambda: r.read(1 << 22), b""):
                f.write(block)
        if md5(dst) != digest:
            print(f"WatermarkV1.zip: MD5 {md5(dst)} is not the recorded {digest}")
            return 1
    with zipfile.ZipFile(dst) as z:
        names = [n for n in z.namelist() if n.endswith(".mat") or n.endswith("readme.txt")]
        z.extractall(out, members=names)
    print(f"unpacked {len(names)} files under {out / 'Watermark'}")
    return 0


def md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    import socket
    socket.setdefaulttimeout(120.0)           # a stalled connection fails instead of hanging; run again to resume from scratch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?")
    ap.add_argument("--set", default="validation", choices=sorted(SETS))
    ap.add_argument("--files", default=None, help="comma-separated file names instead of a set")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--watermark", action="store_true", help="the Watermark benchmark from www.ffi.no instead of the UACR files")
    args = ap.parse_args(argv)
    if args.list or not args.directory:
        for name, (size, digest) in FILES.items():
            print(f"{name:22s} {size / 1e6:9.1f} MB  md5 {digest}")
        print("sets:", {k: len(v) for k, v in SETS.items()})
        return 0 if args.list else 2
    if args.watermark:
        out = pathlib.Path(args.directory)
        out.mkdir(parents=True, exist_ok=True)
        return fetch_watermark(out)
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
