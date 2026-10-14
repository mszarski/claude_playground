"""CAD-Recode v1.5 (filapro/cad-recode-v1.5, CC BY-NC 4.0): ~1M procedurally generated CadQuery programs.

The repo stores one small .py file per program; downloading them one by one is very slow, so this clones the repo
(one git pack) and writes every program into a single parquet: split, path, code.

  python scripts/fetch_cad_recode.py --out /work/data/cad_recode.parquet
"""
import argparse
import os
import subprocess
import tempfile

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://huggingface.co/datasets/filapro/cad-recode-v1.5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    subprocess.run(["git", "clone", "--depth", "1", a.repo, tmp + "/repo"], check=True, env=env)
    rows = []
    for split in ["train", "val"]:
        root = os.path.join(tmp, "repo", split)
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(dirpath, f)
                    with open(p) as fh:
                        rows.append((split, os.path.relpath(p, os.path.join(tmp, "repo")), fh.read()))
    df = pd.DataFrame(rows, columns=["split", "path", "code"]).sort_values("path").reset_index(drop=True)
    lfs = df.code.str.startswith("version https://git-lfs").sum()
    print(f"[cad_recode] {len(df)} programs ({df.split.value_counts().to_dict()}), lfs pointers: {lfs}, "
          f"mean len {df.code.str.len().mean():.0f} chars")
    df.to_parquet(a.out, index=False)
    print(f"[cad_recode] wrote {a.out}")


if __name__ == "__main__":
    main()
