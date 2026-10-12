"""Build a uid -> {level: prompt} text index from Text2CAD annotations (or any CSV/Parquet/JSON(L) of captions).

Text2CAD (NeurIPS 2024) annotates DeepCAD models with prompts at several levels (abstract, beginner,
intermediate, expert, ...). GenCAD also uses DeepCAD, so the two join on the DeepCAD uid ("0067/00675619").
The exact column layout differs between Text2CAD releases, so columns are detected, not hard-coded;
the script prints what it found. Override with --uid_col / --levels if detection is wrong.

Usage:
  # download annotation tables from the Hub, then index
  python scripts/build_text_index.py --hf_repo SadilKhan/Text2CAD --out data/text_index.json
  # or point at local files
  python scripts/build_text_index.py --files ann/*.csv --out data/text_index.json
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

import pandas as pd

UID_CANDIDATES = ["uid", "id", "model_id", "cad_id", "file_id", "name"]
LEVEL_CANDIDATES = ["abstract", "beginner", "intermediate", "expert", "description", "caption", "text",
                    "prompt", "keywords", "nli_data"]


def norm_uid(x):
    """'0067/00675619', '00675619', '0067_00675619', 'deepcad/0067/00675619.json' -> '0067/00675619'."""
    digits = re.findall(r"\d{8}", str(x))
    if not digits:
        return None
    d = digits[-1]
    return f"{d[:4]}/{d}"


def read_any(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict):  # {uid: {...}} or {uid: "text"}
            rows = [{"uid": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in obj.items()]
            return pd.DataFrame(rows)
        return pd.DataFrame(obj)
    if path.endswith(".pkl"):
        return pd.DataFrame(pd.read_pickle(path))
    raise ValueError(f"unsupported file {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--hf_patterns", nargs="*", default=["*.csv", "*.parquet", "*.jsonl"])
    ap.add_argument("--uid_col", default=None)
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--split_json", default="data/filtered_data.json")
    ap.add_argument("--out", default="data/text_index.json")
    a = ap.parse_args()

    files = []
    for f in a.files:
        files += glob.glob(f)
    if a.hf_repo:
        from huggingface_hub import snapshot_download
        d = snapshot_download(a.hf_repo, repo_type="dataset", allow_patterns=a.hf_patterns)
        for p in a.hf_patterns:
            files += glob.glob(os.path.join(d, "**", p), recursive=True)
    assert files, "no annotation files found"

    index = {}
    for path in files:
        try:
            df = read_any(path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {path}: {e}")
            continue
        cols = {c.lower(): c for c in df.columns}
        uid_col = a.uid_col or next((cols[c] for c in UID_CANDIDATES if c in cols), None)
        levels = a.levels or [cols[c] for c in LEVEL_CANDIDATES if c in cols]
        print(f"[file] {os.path.basename(path)}  rows={len(df)}  uid_col={uid_col}  levels={levels}")
        if uid_col is None or not levels:
            print(f"       columns: {list(df.columns)[:20]}")
            continue
        for _, row in df.iterrows():
            uid = norm_uid(row[uid_col])
            if uid is None:
                continue
            entry = index.setdefault(uid, {})
            for lv in levels:
                v = row[lv]
                if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
                    entry[lv.lower()] = v.strip()

    level_counts = Counter(k for e in index.values() for k in e)
    print(f"[index] {len(index)} uids; prompts per level: {dict(level_counts)}")
    if os.path.exists(a.split_json):
        with open(a.split_json) as f:
            splits = json.load(f)
        for s, uids in splits.items():
            hit = sum(u in index for u in uids)
            print(f"[join]  GenCAD {s}: {hit}/{len(uids)} uids have text ({hit / max(1, len(uids)):.1%})")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(index, f)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()
