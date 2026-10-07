"""Extract listening data from Meta's Seamless Interaction dataset (CC-BY-NC 4.0) for a learned listener.

  python deploy/hf_job.py --flavor cpu-upgrade --max-usd 1 --out-repo mszarski/reachy-listening --out-path seamless/v1 \
      -- python scripts/extract_listening.py --out /work/out --train-shards 80 --dev-shards 20

Each item of the dataset is one participant of a two-person conversation (their own microphone, denoised; their
voice activity; 30 Hz head rotation from the dataset's face tracker). Partners sit in different tar shards, so shards
are chosen greedily to complete as many conversations as possible, then streamed one at a time: the video (~85% of
each shard) is skipped, the rest is reduced to 25 Hz and the shard is deleted.

Writes ``{split}/{file_id}.npz`` with ``rot`` (T, 3; head rotation, rad, as the dataset gives it), ``valid`` (T,),
``db`` (T,; loudness of the participant's own voice, dBFS) and ``vad`` (T,; speaking), plus ``pairs.json``
(``[[split, file_a, file_b], ...]``). Nothing here is used by the default (rule-based, Apache-2.0) listener.
"""
import argparse
import collections
import io
import json
import os
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FPS = 25
FILELIST = "https://raw.githubusercontent.com/facebookresearch/seamless_interaction/main/assets/filelist.csv"
REPO = "facebook/seamless-interaction"


def choose_shards(rows, split, n):
    """Greedy: add the shard that completes the most conversations (both participants), ``n`` times."""
    by_key = collections.defaultdict(list)
    for r in rows:
        if r["split"] == split and r["has_imitator_movement"] == "1":
            by_key[r["file_id"].rsplit("_", 1)[0]].append(r)
    pairs = {k: v for k, v in by_key.items() if len(v) == 2}
    count = collections.Counter(tuple(sorted(r["shard"] for r in v)) for v in pairs.values())
    chosen = set()
    while len(chosen) < n:
        gain = collections.Counter()
        for (a, b), c in count.items():
            need = {a, b} - chosen
            if len(need) == 1:
                gain[next(iter(need))] += c
        if gain:
            chosen.add(gain.most_common(1)[0][0])
        else:
            (a, b), _ = max(((k, c) for k, c in count.items() if not set(k) <= chosen), key=lambda kc: kc[1])
            chosen |= {a, b}
    done = [[split] + [r["file_id"] for r in v] for v in pairs.values() if {r["shard"] for r in v} <= chosen]
    return sorted(chosen), done


def features(npz_bytes, wav_bytes, json_bytes):
    import soundfile as sf

    z = np.load(io.BytesIO(npz_bytes))
    rot = z["movement:alignment_head_rotation"].astype(np.float64)
    valid = z["movement:is_valid"][:, 0] > 0
    n30 = len(rot)
    T = int(n30 / 30 * FPS)
    t25, t30 = np.arange(T) / FPS, np.arange(n30) / 30
    rot25 = np.stack([np.interp(t25, t30, rot[:, k]) for k in range(3)], -1)
    valid25 = valid[np.minimum((t25 * 30).round().astype(int), n30 - 1)]
    x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    x = x.mean(1)
    hop = sr // FPS
    m = min(T, len(x) // hop)
    db = 10 * np.log10(np.mean(x[: m * hop].astype(np.float64).reshape(m, hop) ** 2, 1) + 1e-10)
    db = np.concatenate([db, np.full(T - m, -100.0)])
    vad = np.zeros(T, bool)
    for seg in json.loads(json_bytes)["metadata:vad"]:
        vad[int(seg["start"] * FPS): int(np.ceil(seg["end"] * FPS))] = True
    return {"rot": rot25.astype(np.float32), "valid": valid25, "db": db.astype(np.float32), "vad": vad}


def process(shard, wanted, out):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, f"{shard}.tar", repo_type="dataset", local_dir="/tmp/si")
    got = collections.defaultdict(dict)
    with tarfile.open(path) as tf:
        for m in tf:
            fid, ext = os.path.splitext(m.name)
            if fid in wanted and ext in (".npz", ".wav", ".json"):
                got[fid][ext] = tf.extractfile(m).read()
    os.remove(path)
    n = 0
    for fid, parts in got.items():
        if len(parts) == 3:
            np.savez_compressed(os.path.join(out, wanted[fid], fid + ".npz"),
                                **features(parts[".npz"], parts[".wav"], parts[".json"]))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-shards", type=int, default=80)
    ap.add_argument("--dev-shards", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import csv

    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(FILELIST).read().decode())))
    for r in rows:
        r["shard"] = f"{r['label']}/{r['split']}/{int(r['batch_idx']):04d}/{int(r['archive_idx']):04d}"
    shards, pairs, wanted = [], [], {}
    for split, n in (("train", a.train_shards), ("dev", a.dev_shards)):
        s, p = choose_shards(rows, split, n)
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        shards += s
        pairs += p
        wanted.update({f: split for _, *fs in p for f in fs})
        print(f"{split}: {len(s)} shards -> {len(p)} conversations", flush=True)
    with open(os.path.join(a.out, "pairs.json"), "w") as f:
        json.dump(pairs, f)
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(lambda s: process(s, wanted, a.out), shards)):
            done += n
            print(f"shard {i + 1}/{len(shards)}: {done}/{len(wanted)} participants", flush=True)


if __name__ == "__main__":
    main()
