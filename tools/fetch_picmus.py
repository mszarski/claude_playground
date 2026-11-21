#!/usr/bin/env python3
"""Download the PICMUS files the benchmark uses (IQ, simulation) from the Hugging Face
mirror `zeahub/picmus` into a directory. Never commit the data.

Terms, as read on 2026-11-21 on the challenge site's Copyright page
(https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/node/48): "The datasets and code
provided on PICMUS are free of use. The only request is to refer properly to PICMUS - The
Plane Wave Imaging Challenge in Medical UltraSound and quote" the page's citation: Liebgott,
Rodriguez-Molares, Jensen and Bernard, "Plane-Wave Imaging Challenge in Medical Ultrasound",
IEEE IUS 2016 (the paper itself also lists Cervenansky).

--reference also takes, from the challenge's own archive on its site (archive_to_download.zip,
read member by member with HTTP range requests: the whole archive is 564 MB; it holds no
LICENSE file, so the page above is its only statement of terms), the challenge's reference
delay-and-sum images and the phantom and scan files of the two simulated phantoms.
tests/python/test_picmus.py checks our port of the challenge's evaluation against them.
Each file is written to <name>.part and renamed when complete.

Usage: tools/fetch_picmus.py <directory> [--reference]
Then: GPUSONAR_PICMUS_DIR=<directory> tools/test.sh
"""

import argparse
import io
import os
import pathlib
import shutil
import sys
import urllib.request
import zipfile

BASE = "https://huggingface.co/datasets/zeahub/picmus/resolve/main/database/simulation"
FILES = [
    "resolution_distorsion/resolution_distorsion_simu_dataset_iq/resolution_distorsion_simu_dataset_iq.hdf5",
    "contrast_speckle/contrast_speckle_simu_dataset_iq/contrast_speckle_simu_dataset_iq.hdf5",
]
ARCHIVE = ("https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/sites/"
           "www.creatis.insa-lyon.fr.Challenge.IEEE_IUS_2016/files/archive_to_download.zip")
REFERENCE = [
    "archive_to_download/reconstructed_image/simulation/contrast_speckle/contrast_speckle_simu_img_from_iq.hdf5",
    "archive_to_download/reconstructed_image/simulation/resolution_distorsion/resolution_distorsion_simu_img_from_iq.hdf5",
    "archive_to_download/database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5",
    "archive_to_download/database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5",
    "archive_to_download/database/simulation/resolution_distorsion/resolution_distorsion_simu_phantom.hdf5",
    "archive_to_download/database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5",
]


TIMEOUT_S = 60.0


class RangeReader(io.RawIOBase):
    """A remote file read with HTTP range requests, for zipfile. Every read must come back as 206 Partial Content
    with the Content-Range asked for: a server or proxy that ignores Range (200, the whole 564 MB archive) is refused."""

    def __init__(self, url: str, timeout: float = TIMEOUT_S):
        self.url = url
        self.timeout = timeout
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            length = r.headers.get("Content-Length")
        if length is None:
            raise RuntimeError(f"{url}: the server sent no Content-Length, so the archive cannot be read by ranges")
        self.size = int(length)
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        first, last = self.pos, self.pos + n - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={first}-{last}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            status = getattr(r, "status", None)
            got = r.headers.get("Content-Range", "")
            if status != 206 or not got.startswith(f"bytes {first}-{last}/"):
                raise RuntimeError(f"{self.url}: asked for bytes {first}-{last}, the server answered status {status} with "
                                   f"Content-Range {got!r}; it (or a proxy) does not honour range requests. Stopped "
                                   "rather than download the whole archive for every read.")
            data = r.read(n + 1)
        if len(data) != n:
            raise RuntimeError(f"{self.url}: asked for {n} bytes at {first}, got {len(data)}")
        buf[:n] = data
        self.pos += n
        return n


def write_atomically(dst: pathlib.Path, fill) -> None:
    """Write `dst` through `fill(file)` into `dst`.part, renamed to `dst` only when `fill` returns: a fetch that fails
    (network, CRC check) leaves no file that a later run would take as already fetched."""
    part = dst.with_name(dst.name + ".part")
    try:
        with open(part, "wb") as f:
            fill(f)
        os.replace(part, dst)
    finally:
        if part.exists():
            part.unlink()


def fetch_url(url: str, dst: pathlib.Path, timeout: float = TIMEOUT_S) -> None:
    def fill(f):
        with urllib.request.urlopen(url, timeout=timeout) as r:
            shutil.copyfileobj(r, f)
    write_atomically(dst, fill)


def fetch_member(archive, member: str, dst: pathlib.Path) -> None:
    """One member of an open zipfile (its CRC is checked as it is read) into `dst`."""
    def fill(f):
        with archive.open(member) as src:
            shutil.copyfileobj(src, f)
    write_atomically(dst, fill)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="where the files go (created if missing); a file already there is not fetched again")
    ap.add_argument("--reference", action="store_true",
                    help="also the challenge's reference images and simulated phantom and scan files (about 16 MB)")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.directory)
    out.mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        dst = out / pathlib.Path(rel).name
        if dst.exists():
            print(f"have {dst}")
            continue
        print(f"fetching {dst.name}")
        fetch_url(f"{BASE}/{rel}", dst)
    if args.reference:
        missing = [m for m in REFERENCE if not (out / pathlib.Path(m).name).exists()]
        if missing:
            with zipfile.ZipFile(io.BufferedReader(RangeReader(ARCHIVE), buffer_size=1 << 20)) as z:
                for member in missing:
                    dst = out / pathlib.Path(member).name
                    print(f"fetching {dst.name} from the challenge archive")
                    fetch_member(z, member, dst)
        else:
            print("have the reference files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
