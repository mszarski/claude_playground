"""Give the rating page's real-voice items their audio: the CREMA-D clip each one was read from.

  python scripts/rating_audio.py --items runs/rating_clean/set/items.json --out runs/rating_clean/set/audio

Items on the tone test and the tone-only training clips come from real CREMA-D recordings (ODbL / Database Contents
License, CheyneyComputerScience/CREMA-D); hearing them lets a rater judge whether Reachy read the tone right, not
just its words. Writes ``<out>/<clip>.mp4`` (AAC audio, 48 kbps) and adds ``"audio": "<clip>.mp4"`` to those items.
Synthetic lines have no audio and stay text only.
"""
import argparse
import json
import os
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    items = json.load(open(a.items))
    want = {it["clip"].removeprefix("t_"): it for it in items if it["clip"].endswith(".wav")}
    os.makedirs(a.out, exist_ok=True)
    for i in range(2):
        pf = pq.ParquetFile(hf_hub_download("mteb/crema-d", f"data/train-0000{i}-of-00002.parquet", repo_type="dataset"))
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist():
                name = r["path"]
                if name not in want:
                    continue
                dst = os.path.join(a.out, name[:-4] + ".mp4")
                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                    tmp.write(r["bytes"])
                    tmp.flush()
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name, "-vn", "-c:a", "aac",
                                    "-b:a", "48k", dst], check=True)
                want[name]["audio"] = os.path.basename(dst)
    with open(a.items, "w") as f:
        json.dump(items, f, indent=1)
    print(f"{sum('audio' in it for it in items)} of {len(want)} real-voice items have audio -> {a.out}")


if __name__ == "__main__":
    main()
