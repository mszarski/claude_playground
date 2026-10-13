"""Deploy the live viewer (python -m rmr.server) to a private Docker Space.

  python deploy/live_space.py --repo mszarski/reachy-mini-motions-live \
      --ckpt hf://mszarski/reachy-motion-generator/generator.pt --hardware l4x1 --sleep 1800

Uploads rmr/, pyproject.toml, visualizer/, deploy/Dockerfile and deploy/start.sh, sets GENERATOR_CKPT as a Space variable and the
caller's HF_TOKEN as a Space secret (the server needs it to download the private generator checkpoint and voice
student, and to call the planner on Hugging Face Inference Providers, billed to that account). Speech goes to the
licence-clean voice student (``--voice-model``, in-process on the Space's GPU) unless ``--voice-model ""``, which falls
back to ``--responder-model`` on Inference Providers. With ``--gguf`` (the default) the student runs as its 8-bit GGUF
build behind llama.cpp's CUDA server in the container instead of transformers (``--gguf ""`` to turn off). ``--sleep``: seconds without visitors before a paid Space sleeps.
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

Type what Reachy Mini should express, or hold the mic button and talk to it, and watch it listen and respond in 3D.
Speech: Whisper and speech-emotion models hear the words and the tone, and a fine-tuned 4B voice student answers with
a response and a motion recipe. Typed prompts: a zero-shot LLM planner writes the recipe. A 21.8M flow-matching
generator turns it into 25 Hz motion, and every frame is projected onto what the robot can reach. Source: rmr/server.py in github.com/mszarski/claude_playground.
"""


def main():
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--planner-model", default=None)
    ap.add_argument("--responder-model", default="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    help="open-weight LLM that decides how Reachy responds to speech")
    ap.add_argument("--voice-model", default="mszarski/reachy-voice:student/v8-dpo")
    ap.add_argument("--gguf", default="mszarski/reachy-voice:gguf/v8-dpo/reachy-voice-4b-Q8_0.gguf",
                    help="repo:path of the student's GGUF, served by llama.cpp in the Space")
    ap.add_argument("--feedback-repo", default="mszarski/reachy-feedback",
                    help="private HF dataset the thumbs up / down (rmr.feedback) are pushed to")
    ap.add_argument("--audio-model", default=None,
                    help="audio-native responder (rmr.audio_responder), e.g. mszarski/reachy-voice:audio/student/v1; "
                         "replaces the text responder (no llama.cpp), needs a GPU with ~20 GB")
    ap.add_argument("--hardware", default=None, help="e.g. l4x1, t4-small, cpu-upgrade (default: leave as is)")
    ap.add_argument("--sleep", type=int, default=None, help="sleep after this many idle seconds (paid hardware)")
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
    if a.voice_model:
        api.add_space_variable(a.repo, "VOICE_MODEL", a.voice_model)
    else:
        api.delete_space_variable(a.repo, "VOICE_MODEL")
    api.add_space_variable(a.repo, "FEEDBACK_DIR", "/tmp/feedback")
    if a.feedback_repo:
        api.create_repo(a.feedback_repo, repo_type="dataset", private=True, exist_ok=True)
        api.add_space_variable(a.repo, "FEEDBACK_REPO", a.feedback_repo)
    if a.audio_model:
        api.add_space_variable(a.repo, "AUDIO_MODEL", a.audio_model)
        a.gguf = ""                                   # the audio responder replaces the text one
    else:
        try:
            api.delete_space_variable(a.repo, "AUDIO_MODEL")
        except Exception:
            pass
    if a.gguf:
        api.add_space_variable(a.repo, "LLAMA_GGUF", a.gguf)
    else:
        api.delete_space_variable(a.repo, "LLAMA_GGUF")
    if a.hardware:
        api.request_space_hardware(a.repo, a.hardware, sleep_time=a.sleep)
    elif a.sleep:
        api.set_space_sleep_time(a.repo, a.sleep)
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(ROOT, "rmr"), os.path.join(d, "rmr"), ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(os.path.join(ROOT, "visualizer"), os.path.join(d, "visualizer"))
        for f in ["pyproject.toml", "NOTICE"]:
            shutil.copy(os.path.join(ROOT, f), d)
        for f in ["Dockerfile", "start.sh"]:
            shutil.copy(os.path.join(ROOT, "deploy", f), d)
        with open(os.path.join(d, "README.md"), "w") as fh:
            fh.write(CARD)
        api.upload_folder(folder_path=d, repo_id=a.repo, repo_type="space", commit_message="Deploy live server")
    print(f"https://huggingface.co/spaces/{a.repo}")


if __name__ == "__main__":
    main()
