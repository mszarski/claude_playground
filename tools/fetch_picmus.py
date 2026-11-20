#!/usr/bin/env python3
"""Download the PICMUS files the benchmark uses (IQ, simulation) from the Hugging Face
mirror `zeahub/picmus` into a directory. Terms: free use, cite Liebgott et al., IEEE IUS 2016.

Usage: tools/fetch_picmus.py <directory>
Then: GPUSONAR_PICMUS_DIR=<directory> tools/test.sh
"""

import argparse
import pathlib
import sys
import urllib.request

BASE = "https://huggingface.co/datasets/zeahub/picmus/resolve/main/database/simulation"
FILES = [
    "resolution_distorsion/resolution_distorsion_simu_dataset_iq/resolution_distorsion_simu_dataset_iq.hdf5",
    "contrast_speckle/contrast_speckle_simu_dataset_iq/contrast_speckle_simu_dataset_iq.hdf5",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="where the files go (created if missing); a file already there is not fetched again")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
