"""Fine-tune and sample a planner on Hugging Face Jobs (billed to the token's account).

  python -m rmr.planner.hfjob submit --data runs/sft --max-usd 2 --smoke     # ~15 min: environment check
  python -m rmr.planner.hfjob submit --data runs/sft --max-usd 7.5           # full run
  python -m rmr.planner.hfjob status <job id>
  python -m rmr.planner.hfjob fetch --out runs/4b                            # generations + logs from the repo

``submit`` uploads this package, data/teacher and the SFT set to a private dataset repo, then runs
``finetune train`` (+ merge) and ``finetune generate`` on one GPU, uploading the adapter, the merged model and
the generations to a private model repo. The job timeout is ``--max-usd`` / the flavor's price, so a job cannot
cost more than that even if it hangs.
"""
import argparse
import os
import shlex

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
PACKAGES = ["torch", "transformers>=5.5", "trl>=0.29", "peft>=0.18", "datasets", "accelerate", "bitsandbytes",
            "kernels", "huggingface_hub>=1.0", "numpy", "scipy"]

SCRIPT = r"""
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
upload() {{
  python - <<'PY' || true
import glob, os
from huggingface_hub import HfApi
api, repo, sub = HfApi(), os.environ["OUT_REPO"], os.environ["OUT_SUB"]
api.create_repo(repo, private=True, exist_ok=True)
pats = ["generations.json", "log_history.json", "adapter/*", "train.log"] + (["merged/*"] if os.environ.get("UPLOAD_MERGED") else [])
if any(glob.glob(f"/work/out/{{p}}") for p in pats):
    api.upload_folder(folder_path="/work/out", repo_id=repo, path_in_repo=sub or None, allow_patterns=pats,
                      commit_message=f"{{sub or 'planner'}}: job output")
    print("uploaded to", repo, sub)
PY
}}
trap upload EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv
uv pip install --system -q {packages}
hf download {data_repo} --repo-type dataset --local-dir /work
cd /work/code && uv pip install --system -q --no-deps -e .
mkdir -p /work/out
python -m rmr.planner.finetune train --data /work/sft --out /work/out --model {model} {train_args} 2>&1 | tee /work/out/train.log
python -m rmr.planner.finetune generate --model /work/out/merged --out /work/out/generations.json --samples {samples}
"""


def submit(a):
    from huggingface_hub import HfApi, list_jobs_hardware, run_job

    api = HfApi()
    user = api.whoami()["name"]
    data_repo, out_repo = a.data_repo or f"{user}/reachy-planner-sft", a.out_repo or f"{user}/reachy-mini-planner-4b"
    price = {h.name: h.unit_cost_usd for h in list_jobs_hardware()}[a.flavor] * 60          # USD per hour
    hours = a.max_usd / price
    api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=ROOT, repo_id=data_repo, repo_type="dataset", path_in_repo="code",
                      allow_patterns=["rmr/**/*.py", "rmr/assets/*", "pyproject.toml", "README.md", "NOTICE", "LICENSE",
                                      "data/teacher/val.jsonl", "data/teacher/eval_prompts.txt"],
                      commit_message="planner job: code")
    api.upload_folder(folder_path=a.data, repo_id=data_repo, repo_type="dataset", path_in_repo="sft",
                      allow_patterns=["train.jsonl", "val.jsonl", "leaked.json"], commit_message="planner job: SFT set")
    train_args = a.train_args
    if a.smoke:
        train_args += " --max-steps 20 --max-rows 1500 --eval-steps 10"
    script = SCRIPT.format(packages=" ".join(shlex.quote(p) for p in PACKAGES), data_repo=data_repo, model=a.model,
                           train_args=train_args, samples=1 if a.smoke else a.samples)
    env = {"OUT_REPO": out_repo, "OUT_SUB": "smoke" if a.smoke else ""}
    if not a.smoke:
        env["UPLOAD_MERGED"] = "1"
    job = run_job(image=IMAGE, command=["bash", "-c", script], flavor=a.flavor, timeout=int(hours * 3600),
                  env=env, secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
                  name="reachy-planner-smoke" if a.smoke else "reachy-planner-4b")
    print(f"job {job.id} on {a.flavor} (${price:.2f}/h), timeout {hours * 60:.0f} min = at most ${a.max_usd:.2f}")
    print(f"  {job.url}\n  output -> https://huggingface.co/{out_repo}")


def status(a):
    from huggingface_hub import fetch_job_logs, inspect_job

    j = inspect_job(job_id=a.job_id)
    print(j.status.stage, j.status.message or "")
    for line in list(fetch_job_logs(job_id=a.job_id))[-a.tail:]:
        print(line)


def fetch(a):
    from huggingface_hub import HfApi, snapshot_download

    repo = a.out_repo or f"{HfApi().whoami()['name']}/reachy-mini-planner-4b"
    pats = [f"{a.sub}generations.json", f"{a.sub}log_history.json", f"{a.sub}train.log"]
    print(snapshot_download(repo, allow_patterns=pats, local_dir=a.out))


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.planner.hfjob", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--data", required=True, help="SFT folder from rmr.planner.sft")
    s.add_argument("--max-usd", type=float, required=True, help="hard cost cap: sets the job timeout")
    s.add_argument("--flavor", default="a100-large")
    s.add_argument("--model", default="Qwen/Qwen3.5-4B")
    s.add_argument("--train-args", default="", help="extra arguments for finetune train")
    s.add_argument("--samples", type=int, default=12)
    s.add_argument("--smoke", action="store_true", help="20 steps on 1,500 rows, then merge and generate")
    s.add_argument("--data-repo")
    s.add_argument("--out-repo")
    st = sub.add_parser("status")
    st.add_argument("job_id")
    st.add_argument("--tail", type=int, default=30)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    f.add_argument("--out-repo")
    f.add_argument("--sub", default="", help="'smoke/' for the smoke run")
    a = ap.parse_args()
    {"submit": submit, "status": status, "fetch": fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
