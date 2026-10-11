"""Cut speech clips for the listening rating page from the AMI meeting corpus (edinburghcstr/ami, CC BY 4.0).

  python scripts/ami_clips.py --out runs/listen_rating/clips --n 8 --start-index 4

Each clip is ``--seconds`` long and has one person holding the floor: their own utterances cover at least
``--floor`` of the window and nobody else speaks more than ``--others`` of it. The clip is that speaker's headset
channel placed on the meeting's timeline, so their real pauses (where a listener nods) are kept. Clips are spread over
meetings and speakers (at most one per meeting and per speaker, alternating between speaker IDs starting with F and M), skip the
windows already in ``clips.json``, and are appended to it as c<start-index>, c<start-index + 1>, ...
"""
import argparse
import io
import json
import os
import random

import numpy as np

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/listen_rating/clips")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--start-index", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=16.5)
    ap.add_argument("--floor", type=float, default=0.6)
    ap.add_argument("--others", type=float, default=0.1)
    ap.add_argument("--shards", default="test-00001,test-00002,test-00003,validation-00000")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    meta_path = os.path.join(a.out, "clips.json")
    have = json.load(open(meta_path)) if os.path.exists(meta_path) else []
    used = {c["meeting"] for c in have}
    used_spk = {c["speaker"] for c in have}
    files = [f.path for f in HfApi().list_repo_tree("edinburghcstr/ami", path_in_repo="ihm", repo_type="dataset")]
    cand = []
    for shard in a.shards.split(","):
        path = next((f for f in files if os.path.basename(f).startswith(shard)), None)
        if not path:
            continue
        pf = pq.ParquetFile(hf_hub_download("edinburghcstr/ami", path, repo_type="dataset"))
        meta = pf.read(columns=["meeting_id", "speaker_id", "begin_time", "end_time", "text"]).to_pandas()
        meta["row"] = range(len(meta))
        for mid, m in meta.groupby("meeting_id"):
            if mid in used:
                continue
            m = m.sort_values("begin_time")
            for spk, s in m.groupby("speaker_id"):
                for t0 in s.begin_time.values:
                    t1 = t0 + a.seconds
                    def cover(df):
                        return float((np.minimum(df.end_time, t1) - np.maximum(df.begin_time, t0)).clip(lower=0).sum())
                    own, rest = cover(s), cover(m[m.speaker_id != spk])
                    if own >= a.floor * a.seconds and rest <= a.others * a.seconds:
                        rows = s[(s.end_time > t0) & (s.begin_time < t1)]
                        cand.append({"shard": path, "meeting": mid, "speaker": spk, "start": float(t0),
                                     "rows": rows.row.tolist(), "own": own,
                                     "text": " ".join(rows.text.str.lower())})
                        break                         # one window per speaker is enough
    rng = random.Random(a.seed)
    rng.shuffle(cand)
    picked, meetings, speakers = [], set(), set(used_spk)
    for want in "FM" * a.n:                           # alternate F / M speaker IDs; one clip per meeting and speaker
        for c in cand:
            if c["meeting"] not in meetings and c["speaker"] not in speakers and c["speaker"].startswith(want):
                picked.append(c)
                meetings.add(c["meeting"])
                speakers.add(c["speaker"])
                break
        if len(picked) == a.n:
            break
    tables = {}
    for i, c in enumerate(picked):
        pf = tables.setdefault(c["shard"], pq.ParquetFile(hf_hub_download("edinburghcstr/ami", c["shard"],
                                                                           repo_type="dataset")))
        tab = pf.read(columns=["audio", "begin_time"]).to_pandas()
        y = np.zeros(int(a.seconds * SR), np.float32)
        for r in c["rows"]:
            x, sr = sf.read(io.BytesIO(tab.audio.iloc[r]["bytes"]), dtype="float32")
            assert sr == SR
            off = int(round((tab.begin_time.iloc[r] - c["start"]) * SR))
            lo, hi = max(0, off), min(len(y), off + len(x))
            if hi > lo:
                y[lo:hi] += x[lo - off:hi - off]
        y /= max(1e-6, np.abs(y).max()) / 0.8
        name = f"c{a.start_index + i}"
        sf.write(os.path.join(a.out, name + ".wav"), y, SR)
        have.append({"clip": name, "meeting": c["meeting"], "speaker": c["speaker"],
                     "start": round(c["start"], 2), "text": c["text"]})
        print(name, c["meeting"], c["speaker"], f"{c['own']:.1f}s speech", "|", c["text"][:90])
    with open(meta_path, "w") as f:
        json.dump(have, f, indent=1)
    print(f"{len(picked)} new clips ({len(cand)} candidate windows) -> {a.out}")


if __name__ == "__main__":
    main()
