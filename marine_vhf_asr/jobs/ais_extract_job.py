# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "huggingface_hub", "requests"]
# ///
"""
Extract a regional AIS subset from NOAA MarineCadastre daily files and push it to
a private Hub dataset. Runs on HF Jobs (the files are ~0.5 GB/day for all US
waters; only the bounding box is kept).

    hf jobs uv run --flavor cpu-upgrade --timeout 2h --secrets HF_TOKEN \\
        --env OUT_REPO=<user>/marine-vhf-ais-nyc --env BBOX=40.35,-74.35,41.0,-73.6 \\
        --env DAYS=2024-07-15,2024-07-16 marine_vhf_asr/jobs/ais_extract_job.py
"""
import io
import os
import zipfile

import pandas as pd
import requests
from huggingface_hub import HfApi

lat0, lon0, lat1, lon1 = map(float, os.environ["BBOX"].split(","))
repo = os.environ["OUT_REPO"]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "VesselName", "IMO", "CallSign",
        "VesselType", "Status", "Length", "Width", "Draft", "TransceiverClass"]

for day in os.environ["DAYS"].split(","):
    y, m, d = day.split("-")
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m}_{d}.zip"
    print("Fetching", url, flush=True)
    r = requests.get(url, timeout=600)
    if r.status_code != 200:
        print("  HTTP", r.status_code, "- skipped", flush=True)
        continue
    print(f"  {len(r.content) / 1e6:.0f} MB", flush=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        parts = []
        with z.open(name) as f:
            for chunk in pd.read_csv(f, chunksize=1_000_000, usecols=lambda c: c in COLS, low_memory=False):
                keep = chunk[(chunk.LAT >= lat0) & (chunk.LAT <= lat1) & (chunk.LON >= lon0) & (chunk.LON <= lon1)]
                parts.append(keep)
    df = pd.concat(parts, ignore_index=True)
    print(f"  kept {len(df)} rows, {df.MMSI.nunique()} vessels, {df.VesselName.nunique()} names", flush=True)
    out = f"/tmp/ais_{day}.parquet"
    df.to_parquet(out, index=False)
    api.upload_file(path_or_fileobj=out, path_in_repo=f"data/{day}.parquet", repo_id=repo, repo_type="dataset",
                    commit_message=f"AIS {day} bbox {os.environ['BBOX']}")
print("DONE", flush=True)
