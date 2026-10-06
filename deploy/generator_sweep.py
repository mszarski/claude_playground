"""Run rmr.generator.sweep on Hugging Face Jobs (billed to the token's account; private repos).

  python deploy/generator_sweep.py --max-usd 1 --args "--steps 300 --seeds 0 --configs reference"   # smoke
  python deploy/generator_sweep.py --max-usd 9                                                        # full sweep

Uploads this package to the private dataset repo used for planner jobs, runs the sweep on one GPU, and uploads
results.jsonl, summary.json, the training histories and the checkpoints to <model repo>/sweep/<tag>/ when the job
ends (also on failure). The timeout is --max-usd / the flavor's price.
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
    HfApi().upload_folder(folder_path="/work/out", repo_id=os.environ["OUT_REPO"], path_in_repo=os.environ["OUT_SUB"],
                          commit_message="generator sweep: " + os.environ["OUT_SUB"])
    print("uploaded", os.environ["OUT_SUB"])
PY
}}
trap upload EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv
uv pip install --system -q torch numpy scipy "huggingface_hub>=1.0" "reachy-mini-rust-kinematics>=1.0.3"
hf download {data_repo} --repo-type dataset --include "code/*" --local-dir /work
cd /work/code && uv pip install --system -q --no-deps -e .
mkdir -p /work/out
python -m rmr.generator.sweep --out /work/out {args} 2>&1 | tee /work/out/sweep.log
"""


def main():
    from huggingface_hub import HfApi, list_jobs_hardware, run_job

    from rmr.planner.hfjob import CODE_PATTERNS, IMAGE

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--flavor", default="a10g-small")
    ap.add_argument("--args", default="", help="arguments for python -m rmr.generator.sweep")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M"))
    a = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    data_repo, out_repo = f"{user}/reachy-planner-sft", f"{user}/reachy-motion-generator"
    price = {h.name: h.unit_cost_usd for h in list_jobs_hardware()}[a.flavor] * 60
    api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=ROOT, repo_id=data_repo, repo_type="dataset", path_in_repo="code",
                      allow_patterns=CODE_PATTERNS, commit_message="generator sweep: code")
    script = SCRIPT.format(data_repo=data_repo, args=" ".join(shlex.quote(x) for x in shlex.split(a.args)))
    job = run_job(image=IMAGE, command=["bash", "-c", script], flavor=a.flavor, timeout=int(a.max_usd / price * 3600),
                  env={"OUT_REPO": out_repo, "OUT_SUB": f"sweep/{a.tag}"}, secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
                  name="reachy-generator-sweep")
    print(f"job {job.id} on {a.flavor} (${price:.2f}/h), timeout {a.max_usd / price * 60:.0f} min = at most ${a.max_usd:.2f}")
    print(f"  {job.url}\n  output -> https://huggingface.co/{out_repo}/tree/main/sweep/{a.tag}")


if __name__ == "__main__":
    main()
