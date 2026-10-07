"""Say something, see how Reachy responds: voice -> transcript + tone -> response -> reachable move (local).

  python -m rmr.converse clip.wav --out runs/converse                        # zero-shot LLM on PLANNER_BASE_URL
  PLANNER_BASE_URL=local python -m rmr.converse clip.wav --responder Qwen/Qwen3.5-4B \\
      --planner mszarski/reachy-mini-planner-4b --out runs/converse          # fully offline, open models only

Writes ``response.json`` (what it heard, its reading, the response and recipe) and ``motions/*.json``; ``--render``
adds MuJoCo videos. Everything runs on this machine except a remote LLM endpoint if ``PLANNER_BASE_URL`` names one.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.converse", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="hf://mszarski/reachy-motion-generator/generator_v2.pt")
    ap.add_argument("--model", help="zero-shot planner LLM (default: rmr.planner.llm.DEFAULT_MODEL)")
    ap.add_argument("--responder", help="LLM deciding the response (default: --model)")
    ap.add_argument("--planner", help="fine-tuned planner (dir or Hub repo id) instead of the zero-shot LLM")
    ap.add_argument("--voice-model", help="distilled responder: response + recipe in one local call "
                                          "(e.g. mszarski/reachy-voice:student/v3)")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    from .server import Engine

    eng = Engine(a.ckpt, a.model, a.planner, a.responder, a.voice_model)
    with open(a.audio, "rb") as f:
        r = eng.respond(f.read(), n=a.variants)
    mdir = os.path.join(a.out, "motions")
    os.makedirs(mdir, exist_ok=True)
    paths = []
    for i, m in enumerate(r.pop("moves")):
        paths.append(os.path.join(mdir, f"response__{i}.json"))
        json.dump(m, open(paths[-1], "w"))
    json.dump(r, open(os.path.join(a.out, "response.json"), "w"), indent=1)
    h = r["heard"]
    print(f'heard: "{h["text"]}"\nvoice: {h["emotion"]} ({h["confidence"]:.0%}), arousal {h.get("arousal", 0):.2f}, '
          f'valence {h.get("valence", 0):.2f}\nreading: {r["reading"]}\nresponse: {r["prompt"]}\nrecipe: {r["recipe"]}\n'
          f'timing: {r["timing_ms"]}')
    if a.render:
        from .renderer.outputs import videos
        videos(paths, os.path.join(a.out, "videos"))


if __name__ == "__main__":
    main()
