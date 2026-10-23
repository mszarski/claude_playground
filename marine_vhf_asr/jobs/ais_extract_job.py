# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow", "huggingface_hub", "requests"]
# ///
"""
Extract regional AIS subsets and push them to a private Hub dataset as
data/<region>/<day>.parquet, in NOAA MarineCadastre column names. Runs on HF Jobs.

Sources (by day prefix in DAYS):
    noaa:YYYY-MM-DD  NOAA MarineCadastre daily file, all US waters (~0.4 GB zipped)
    dk:YYYY-MM-DD    Danish Maritime Authority daily file (aisdata.ais.dk)

REGIONS is JSON: {"name": [lat0, lon0, lat1, lon1], ...}; each day is downloaded
once and every region inside it is extracted.

    hf jobs uv run --flavor cpu-upgrade --timeout 3h --secrets HF_TOKEN \\
        --env OUT_REPO=<user>/marine-vhf-ais --env 'REGIONS={"houston": [29.0, -95.3, 29.9, -94.4]}' \\
        --env DAYS=noaa:2024-07-15,dk:2024-07-15 marine_vhf_asr/jobs/ais_extract_job.py
"""
import io
import json
import os
import zipfile

import pandas as pd
import requests
from huggingface_hub import HfApi

COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "VesselName", "IMO", "CallSign",
        "VesselType", "Status", "Length", "Width", "Draft", "TransceiverClass"]

# Danish ship type text -> nearest MarineCadastre VesselType code
DK_TYPES = {"Fishing": 30, "Towing": 31, "Towing long/wide": 32, "Sailing": 36, "Pleasure": 37,
            "Pilot": 50, "SAR": 51, "Tug": 52, "Port tender": 53, "Law enforcement": 55,
            "Passenger": 60, "HSC": 40, "Cargo": 70, "Tanker": 80, "Dredging": 33, "Diving": 34,
            "Military": 35, "Other": 90, "Undefined": 0}
DK_RENAME = {"# Timestamp": "BaseDateTime", "Latitude": "LAT", "Longitude": "LON", "Name": "VesselName",
             "Callsign": "CallSign", "Ship type": "VesselType", "Type of mobile": "TransceiverClass",
             "Draught": "Draft"}


def chunks(source, day):
    y, m, d = day.split("-")
    if source == "noaa":
        url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m}_{d}.zip"
    else:
        url = f"http://aisdata.ais.dk/{y}/aisdk-{y}-{m}-{d}.zip"
    print("Fetching", url, flush=True)
    r = requests.get(url, timeout=1200)
    if r.status_code != 200:
        print("  HTTP", r.status_code, "- skipped", flush=True)
        return
    print(f"  {len(r.content) / 1e6:.0f} MB", flush=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            for chunk in pd.read_csv(f, chunksize=1_000_000, low_memory=False):
                if source == "dk":
                    chunk = chunk.rename(columns=DK_RENAME)
                    chunk["BaseDateTime"] = pd.to_datetime(chunk.BaseDateTime, format="%d/%m/%Y %H:%M:%S").astype(str)
                    chunk["VesselType"] = chunk.VesselType.map(DK_TYPES)
                    chunk["TransceiverClass"] = chunk.TransceiverClass.str.replace("Class ", "", regex=False)
                    chunk = chunk[chunk.TransceiverClass.isin(["A", "B"])]
                yield chunk[[c for c in COLS if c in chunk.columns]]


regions = json.loads(os.environ["REGIONS"])
repo = os.environ["OUT_REPO"]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for item in os.environ["DAYS"].split(","):
    source, day = item.split(":")
    parts = {name: [] for name in regions}
    for chunk in chunks(source, day):
        for name, (lat0, lon0, lat1, lon1) in regions.items():
            keep = chunk[(chunk.LAT >= lat0) & (chunk.LAT <= lat1) & (chunk.LON >= lon0) & (chunk.LON <= lon1)]
            if len(keep):
                parts[name].append(keep)
    for name, ps in parts.items():
        if not ps:
            continue
        df = pd.concat(ps, ignore_index=True)
        print(f"  {name}: {len(df)} rows, {df.MMSI.nunique()} vessels", flush=True)
        out = f"/tmp/{name}_{day}.parquet"
        df.to_parquet(out, index=False)
        api.upload_file(path_or_fileobj=out, path_in_repo=f"data/{name}/{day}.parquet", repo_id=repo,
                        repo_type="dataset", commit_message=f"AIS {name} {day}")
print("DONE", flush=True)
