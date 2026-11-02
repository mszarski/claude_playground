# /// script
# requires-python = ">=3.11"
# dependencies = ["kagglehub", "requests", "huggingface_hub>=0.30"]
# ///
"""
Pull the Kaggle marine radio chatter WhisperX transcripts (as one tar.gz), the full audio
file listing, and PER_FEED audio files per feed into a private Hub dataset (OUT_REPO).
Runs on HF Jobs (cpu-upgrade) because kaggle.com is blocked from the dev container:
    hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN --env OUT_REPO=<user>/marine-radio-chatter-sample \\
        --env PER_FEED=3 marine_vhf_asr/jobs/kaggle_sample_job.py
Broadcastify-derived audio: internal research only, do not republish.
"""
import collections, json, os, shutil, time
import kagglehub, requests


def get_json(url, params):
    """GET with retries: the listing endpoint sometimes returns an empty/HTML page (rate limit)."""
    for attempt in range(8):
        r = requests.get(url, params=params, timeout=60)
        try:
            return r.json()
        except ValueError:
            print(f"  listing retry {attempt}: HTTP {r.status_code} {r.text[:120]!r}", flush=True)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"listing failed: {url} {params}")
from huggingface_hub import HfApi
AUDIO = "linogova/marine-radio-chatter-bridge-2-bridge-communication"
TRANS = "linogova/marine-radio-chatter-bridge-to-bridge-transcript"
OUT = os.environ["OUT_REPO"]; PER_FEED = int(os.environ.get("PER_FEED", "3"))
api = HfApi(); api.create_repo(OUT, repo_type="dataset", private=True, exist_ok=True)

# Full audio file listing (paginated)
files, token = [], None
while True:
    params = {"pageSize": 200}
    if token: params["pageToken"] = token
    d = get_json(f"https://www.kaggle.com/api/v1/datasets/list/{AUDIO}", params)
    time.sleep(0.5)
    files += [(f.get("nameNullable") or f.get("name"), f.get("totalBytesNullable") or f.get("totalBytes")) for f in d.get("datasetFiles", [])]
    token = d.get("nextPageToken")
    if not token: break
by_feed = collections.defaultdict(list)
for name, size in files:
    by_feed[name.split("/")[1]].append((name, size))
summary = {feed: {"files": len(v), "bytes": sum(s or 0 for _, s in v), "first": min(v)[0], "last": max(v)[0]} for feed, v in by_feed.items()}
print("AUDIO_FILES", len(files), "FEEDS", len(by_feed), flush=True)
os.makedirs("/tmp/out", exist_ok=True)
json.dump(summary, open("/tmp/out/audio_feeds.json", "w"), indent=1)
json.dump(files, open("/tmp/out/audio_files.json", "w"))

# Transcripts (1.5 GB)
tpath = kagglehub.dataset_download(TRANS)
print("TRANSCRIPTS", tpath, flush=True)
# Tens of thousands of small files: ship them as one archive
shutil.make_archive("/tmp/out/transcripts", "gztar", root_dir=tpath)
print("TRANSCRIPT_FILES", sum(len(f) for _, _, f in os.walk(tpath)), flush=True)

# A few audio files per feed, spread over the period
os.makedirs("/tmp/out/audio", exist_ok=True)
for feed, v in sorted(by_feed.items()):
    v = sorted(v)
    picks = [v[int(i * (len(v) - 1) / max(1, PER_FEED - 1))] for i in range(min(PER_FEED, len(v)))]
    for name, _ in picks:
        try:
            p = kagglehub.dataset_download(AUDIO, path=name)
            dst = f"/tmp/out/audio/{name.split('/', 1)[1]}"
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(p, dst)
        except Exception as e:
            print("SKIP", name, repr(e)[:200], flush=True)
api.upload_folder(repo_id=OUT, repo_type="dataset", folder_path="/tmp/out",
                  commit_message="Marine radio chatter: feed listing, WhisperX transcripts archive, audio sample")
print("UPLOADED", flush=True)
