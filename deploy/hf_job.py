"""Run one command from this repo on a Hugging Face Jobs GPU, with a cost cap; upload /work/out when it ends.

  python deploy/hf_job.py --max-usd 1 --out-repo mszarski/reachy-voice --out-path listen/train \
      --packages "transformers>=5.5 safetensors soundfile pandas" -- python scripts/listen_meld.py --split train --out /work/out

The repo's code is uploaded to the private dataset repo used by the other jobs and installed with --no-deps; the
job installs torch, huggingface_hub and --packages. HF_TOKEN is passed as a secret. The timeout is --max-usd / the
flavor's price, so a job cannot cost more than that.
"""
import argparse
import os
import shlex
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCRIPT = r"""
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
upload() {{
  python - <<'PY' || true
import os
from huggingface_hub import HfApi
if os.path.isdir("/work/out") and os.listdir("/work/out"):
    api = HfApi(); api.create_repo(os.environ["OUT_REPO"], private=True, exist_ok=True)
    api.upload_folder(folder_path="/work/out", repo_id=os.environ["OUT_REPO"], path_in_repo=os.environ["OUT_PATH"],
                      commit_message="job output: " + os.environ["OUT_PATH"])
    print("uploaded", os.environ["OUT_PATH"])
PY
}}
trap upload EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv || true
uv pip install --system -q torch numpy scipy "huggingface_hub>=1.0" "reachy-mini-rust-kinematics>=1.0.3" {packages}
hf download {data_repo} --repo-type dataset --include "code/*" --local-dir /work
cd /work/code && uv pip install --system -q --no-deps -e .
mkdir -p /work/out
{command} 2>&1 | tee /work/out/job.log
"""


def main():
    from huggingface_hub import HfApi, list_jobs_hardware, run_job

    from rmr.planner.hfjob import CODE_PATTERNS, IMAGE

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--flavor", default="a10g-small")
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--packages", default="")
    ap.add_argument("--name", default="reachy-job")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    api = HfApi()
    data_repo = f"{api.whoami()['name']}/reachy-planner-sft"
    price = {h.name: h.unit_cost_usd for h in list_jobs_hardware()}[a.flavor] * 60
    api.upload_folder(folder_path=ROOT, repo_id=data_repo, repo_type="dataset", path_in_repo="code",
                      allow_patterns=CODE_PATTERNS + ["scripts/*.py"], commit_message=f"job code: {a.name}")
    script = SCRIPT.format(data_repo=data_repo, packages=" ".join(shlex.quote(p) for p in shlex.split(a.packages)),
                           command=" ".join(shlex.quote(c) for c in cmd))
    job = run_job(image=IMAGE, command=["bash", "-c", script], flavor=a.flavor, timeout=int(a.max_usd / price * 3600),
                  env={"OUT_REPO": a.out_repo, "OUT_PATH": a.out_path}, secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
                  name=a.name)
    print(f"job {job.id} on {a.flavor} (${price:.2f}/h), timeout {a.max_usd / price * 60:.0f} min = at most ${a.max_usd:.2f}")
    print(f"  {job.url}\n  output -> https://huggingface.co/{a.out_repo}/tree/main/{a.out_path}")


if __name__ == "__main__":
    main()
