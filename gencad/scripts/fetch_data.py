"""Fetch GenCAD data + checkpoints (Google Drive) and Text2CAD annotations (HF Hub); optionally mirror to a
private HF dataset repo so later GPU jobs pull from the Hub instead of Drive.

  python scripts/fetch_data.py --data data                         # Drive -> data/, data/ckpt/
  python scripts/fetch_data.py --data data --mirror me/gencad-data # ... and upload archives to the Hub
  python scripts/fetch_data.py --data data --from_mirror me/gencad-data   # later jobs: Hub -> data/

Drive folder ids are the ones linked from the upstream README.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

DATA_FOLDER = "1M0dPr5kILGY9HTRCHox1vLLDhhxJWl_C"
CKPT_FOLDER = "1Ej7wdtlqT5P-SoUf3gsZXD8b78XqhiI5"
MIRROR_DIRS = ["cad_vec", "images", "sketches", "ckpt"]


def sh(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def safe_extract(tar, dest):
    try:
        tar.extractall(dest, filter="data")
    except TypeError:  # Python without tarfile extraction filters
        tar.extractall(dest)


def extract_archives(root):
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.endswith(".zip"):
            print(f"[extract] {p}")
            with zipfile.ZipFile(p) as z:
                z.extractall(os.path.dirname(p))
            os.remove(p)
        elif p.endswith((".tar", ".tar.gz", ".tgz")):
            print(f"[extract] {p}")
            with tarfile.open(p) as t:
                safe_extract(t, os.path.dirname(p))
            os.remove(p)


def from_drive(data):
    sh(sys.executable, "-m", "pip", "install", "-q", "gdown")
    tmp = os.path.join(data, "_drive")
    os.makedirs(tmp, exist_ok=True)
    sh("gdown", "--folder", "--remaining-ok", f"https://drive.google.com/drive/folders/{DATA_FOLDER}",
       "-O", os.path.join(tmp, "dataset"))
    sh("gdown", "--folder", "--remaining-ok", f"https://drive.google.com/drive/folders/{CKPT_FOLDER}",
       "-O", os.path.join(data, "ckpt"))
    extract_archives(tmp)
    extract_archives(os.path.join(data, "ckpt"))
    # flatten: whatever top-level layout Drive had, move known dirs/files into data/
    for name in ["cad_vec", "images", "sketches", "multi_view_images", "filtered_data.json", "image_ids.json"]:
        hits = sorted(glob.glob(os.path.join(tmp, "**", name), recursive=True), key=len)
        if hits and not os.path.exists(os.path.join(data, name)):
            shutil.move(hits[0], os.path.join(data, name))
    print("[drive] data dir now contains:", sorted(os.listdir(data)))
    print("[drive] checkpoints:", sorted(os.listdir(os.path.join(data, "ckpt"))))


def mirror_to_hub(data, repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    os.makedirs("_mirror", exist_ok=True)
    for d in MIRROR_DIRS:
        src = os.path.join(data, d)
        if not os.path.isdir(src):
            continue
        arc = os.path.join("_mirror", f"{d}.tar")  # one tar per dir: 800k small PNGs upload badly as files
        print(f"[mirror] tar {src}")
        with tarfile.open(arc, "w") as t:
            t.add(src, arcname=d)
        api.upload_file(path_or_fileobj=arc, path_in_repo=f"{d}.tar", repo_id=repo, repo_type="dataset")
        os.remove(arc)
    for f in ["filtered_data.json", "image_ids.json", "text_index.json"]:
        if os.path.exists(os.path.join(data, f)):
            api.upload_file(path_or_fileobj=os.path.join(data, f), path_in_repo=f, repo_id=repo,
                            repo_type="dataset")
    print(f"[mirror] done: https://huggingface.co/datasets/{repo}")


def from_mirror(data, repo, include):
    from huggingface_hub import snapshot_download
    pats = [f"{d}.tar" for d in include] + ["*.json"]
    d = snapshot_download(repo, repo_type="dataset", allow_patterns=pats)
    os.makedirs(data, exist_ok=True)
    for p in glob.glob(os.path.join(d, "*.tar")):
        print(f"[mirror] extract {os.path.basename(p)}")
        with tarfile.open(p) as t:
            safe_extract(t, data)
    for p in glob.glob(os.path.join(d, "*.json")):
        shutil.copy(p, os.path.join(data, os.path.basename(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mirror", default=None, help="upload to this private HF dataset repo after fetching")
    ap.add_argument("--from_mirror", default=None, help="download from this HF dataset repo instead of Drive")
    ap.add_argument("--include", nargs="*", default=MIRROR_DIRS, help="with --from_mirror: which dirs to pull")
    ap.add_argument("--text2cad", default="SadilKhan/Text2CAD", help="HF dataset with text annotations ('' to skip)")
    a = ap.parse_args()
    os.makedirs(a.data, exist_ok=True)

    if a.from_mirror:
        from_mirror(a.data, a.from_mirror, a.include)
    else:
        from_drive(a.data)
        if a.text2cad and not os.path.exists(os.path.join(a.data, "text_index.json")):
            sh(sys.executable, os.path.join(os.path.dirname(__file__), "build_text_index.py"),
               "--hf_repo", a.text2cad, "--split_json", os.path.join(a.data, "filtered_data.json"),
               "--out", os.path.join(a.data, "text_index.json"))
        if a.mirror:
            mirror_to_hub(a.data, a.mirror)
    # upstream inference_gencad.py looks in model/ckpt
    if os.path.isdir(os.path.join(a.data, "ckpt")) and not os.path.exists("model/ckpt/ae_ckpt_epoch1000.pth"):
        for p in glob.glob(os.path.join(a.data, "ckpt", "**", "*.pt*"), recursive=True):
            dst = os.path.join("model", "ckpt", os.path.basename(p))
            if not os.path.exists(dst):
                os.makedirs("model/ckpt", exist_ok=True)
                os.symlink(os.path.abspath(p), dst)


if __name__ == "__main__":
    main()
