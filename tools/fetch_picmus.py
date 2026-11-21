#!/usr/bin/env python3
"""Download the PICMUS files the benchmark uses (IQ, simulation) from the Hugging Face
mirror `zeahub/picmus` into a directory. Terms (the challenge site's Copyright page,
www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/, read 2026-11-21): "The datasets and code
provided on PICMUS are free of use. The only request is to refer properly to PICMUS ... and
quote" Liebgott, Rodriguez-Molares, Cervenansky, Jensen and Bernard, "Plane-Wave Imaging
Challenge in Medical Ultrasound", IEEE IUS 2016. Never commit the data.

--reference also takes, from the challenge's own archive on its site (archive_to_download.zip,
read member by member with HTTP range requests: the whole archive is 564 MB), the challenge's
reference delay-and-sum images and the phantom and scan files of the two simulated phantoms.
tests/python/test_picmus.py checks our port of the challenge's evaluation against them.

Usage: tools/fetch_picmus.py <directory> [--reference]
Then: GPUSONAR_PICMUS_DIR=<directory> tools/test.sh
"""

import argparse
import io
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


class RangeReader(io.RawIOBase):
    """A remote file read with HTTP range requests, for zipfile."""

    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
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
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


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
        urllib.request.urlretrieve(f"{BASE}/{rel}", dst)
    if args.reference:
        missing = [m for m in REFERENCE if not (out / pathlib.Path(m).name).exists()]
        if missing:
            with zipfile.ZipFile(io.BufferedReader(RangeReader(ARCHIVE), buffer_size=1 << 20)) as z:
                for member in missing:
                    dst = out / pathlib.Path(member).name
                    print(f"fetching {dst.name} from the challenge archive")
                    with z.open(member) as src, open(dst, "wb") as f:
                        shutil.copyfileobj(src, f)
        else:
            print("have the reference files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
