"""Deploy the live viewer (python -m rmr.server) to a private Docker Space.

  python deploy/live_space.py --repo mszarski/reachy-mini-motions-live \
      --ckpt hf://mszarski/reachy-motion-generator/generator.pt

Uploads rmr/, pyproject.toml, visualizer/ and deploy/Dockerfile, sets GENERATOR_CKPT as a Space variable and the
caller's HF_TOKEN as a Space secret (the server needs it to download the private generator checkpoint and to call
the planner on Hugging Face Inference Providers, billed to that account).
"""
import argparse
import os
import shutil
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
CARD = """---
title: Reachy Mini Motions Live
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Type a prompt, watch Reachy Mini perform it in 3D
---

Type what Reachy Mini should express, or hold the mic button and talk to it, and watch it respond in 3D. A zero-shot LLM planner writes a motion recipe, a
21.8M flow-matching generator turns it into 25 Hz motion on CPU, and every frame is projected onto what the robot
can reach. Source: rmr/server.py in github.com/mszarski/claude_playground.
"""


def main():
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--planner-model", default=None)
    ap.add_argument("--responder-model", default="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    help="open-weight LLM that decides how Reachy responds to speech")
    ap.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.repo, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    api.add_space_secret(a.repo, "HF_TOKEN", os.environ["HF_TOKEN"])
    api.add_space_variable(a.repo, "GENERATOR_CKPT", a.ckpt)
    if a.planner_model:
        api.add_space_variable(a.repo, "PLANNER_MODEL", a.planner_model)
    api.add_space_variable(a.repo, "RESPONDER_MODEL", a.responder_model)
    api.add_space_variable(a.repo, "PRELOAD_VOICE", "1")
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(ROOT, "rmr"), os.path.join(d, "rmr"), ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(os.path.join(ROOT, "visualizer"), os.path.join(d, "visualizer"))
        for f in ["pyproject.toml", "NOTICE"]:
            shutil.copy(os.path.join(ROOT, f), d)
        shutil.copy(os.path.join(ROOT, "deploy", "Dockerfile"), d)
        with open(os.path.join(d, "README.md"), "w") as fh:
            fh.write(CARD)
        api.upload_folder(folder_path=d, repo_id=a.repo, repo_type="space", commit_message="Deploy live server")
    print(f"https://huggingface.co/spaces/{a.repo}")


if __name__ == "__main__":
    main()
