"""Zero-shot baseline: Claude (claude-opus-5-5) reads the same renders our models see and writes CadQuery code.

Like cadrille's GPT-4o / o4-mini baselines (arXiv 2505.22914, Tab. 7-8), but on our test uids and renders. Uses the
Message Batches API (50% price, results within ~1 h). Two steps:

  # 1. submit (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
  python scripts/claude_baseline.py submit --data data --uids_from results/baselines_vlm_final/preds.pkl \
      --modes view4 view1 view1+expert --out results/claude_baseline
  # 2. collect when the batch has ended: writes <out>/preds.pkl (code text per mode, same uids / dup mask)
  python scripts/claude_baseline.py collect --out results/claude_baseline

Score with scripts/cq_iou.py (same protocol for Claude's CadQuery and our models' DeepCAD programs).

Rich / real parts (scripts/gen_rich_data.py, scripts/prep_real.py layout): `submit --rich data/rich/real --modes view4
view1` uses a prompt that allows the full CadQuery feature set (fillets, chamfers, revolves, holes, shells, patterns);
score the collected preds.pkl with scripts/rich_iou.py.

Second attempt (teacher data): `retry --src <out of a scored --rich run> --below 0.8 --out <dir>` renders each weak
attempt with the input cameras (or keeps its build error) and asks Claude to correct it, given the input views and
the render of its own program; `collect` works on the result as usual (same uids layout, attempt kept in batch.json).
"""
import argparse
import base64
import io
import json
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "claude-opus-5-5"
VIEWS = ("The image is a 2x2 grid of orthographic renders of one part, all at the same scale: top-left looks from "
         "direction (+1,+1,+1) towards the origin (isometric, from above); top-right from (-1,-1,-1) (isometric, "
         "from below); bottom-left from -Y towards +Y (front view, Z up); bottom-right from +Z down (top view, "
         "Y up). Z is up in every view except the top view.")
VIEW1 = "The image is an orthographic isometric render of one part, viewed from direction (+1,+1,+1) (Z up)."
TASK = """Reconstruct this part as a CadQuery program. Requirements:
- Use only sketch-and-extrude modelling: cq.Workplane on the named planes 'XY', 'YZ', 'ZX' (and their reversed
  forms) with an origin, sketches made of .segment / .arc / .circle (or .rect), .extrude, and .union / .cut /
  .intersect between solids.
- Centre the part at the origin and scale it so its longest bounding-box side is 200 units. Keep the axes as shown.
- The program must start with `import cadquery as cq` and assign the final solid to a variable named `r`.
Return only the program in a single ```python code block."""
TASK_RICH = """Reconstruct this part as a CadQuery program. Requirements:
- Use any CadQuery features the part needs: sketches and extrudes, revolves, holes (hole / cboreHole / cskHole),
  fillets and chamfers, shells, patterns, boolean union / cut.
- Centre the part at the origin. Keep the axes and orientation as shown; only proportions matter, not absolute size.
- The program must start with `import cadquery as cq` and assign the final solid to a variable named `r`.
Return only the program in a single ```python code block."""
EXAMPLE = """Example of the expected style:
```python
import cadquery as cq
w0=cq.Workplane('XY',origin=(0,0,-10))
r=w0.sketch().segment((-100,-60),(100,-60)).segment((100,60)).segment((-100,60)).close().assemble().push([(0,0)]).circle(25,mode='s').finalize().extrude(20)
```"""


def grid_png(stem, mode):
    from PIL import Image
    from mm.vlm_cad import compose_views
    n = 4 if mode.startswith("view4") else 1
    ims = [Image.open(f"{stem}_{k}.png").convert("RGB") for k in range(n)]
    buf = io.BytesIO()
    compose_views(ims).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def request(stem, mode, prompt, effort, rich=False):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": grid_png(stem, mode)}}]
    text = (VIEWS if mode.startswith("view4") else VIEW1) + "\n\n"
    if prompt:
        text += f"The designer's description of the part:\n{prompt}\n\n"
    content.append({"type": "text", "text": text + (TASK_RICH if rich else TASK + "\n\n" + EXAMPLE)})
    return {"model": MODEL, "max_tokens": 16000, "output_config": {"effort": effort},
            "messages": [{"role": "user", "content": content}]}


def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return (m[-1] if m else text).strip() or None


def submit(a):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    extra, text_index = {}, {}
    if a.rich:
        import pandas as pd
        df = pd.concat([pd.read_parquet(os.path.join(a.rich, sh, "parts.parquet")).assign(shard=sh)
                        for sh in sorted(os.listdir(a.rich)) if os.path.exists(os.path.join(a.rich, sh, "parts.parquet"))],
                       ignore_index=True)
        if a.exclude_batch:                            # parts already sent in earlier batches
            done = set()
            for fn in a.exclude_batch:
                done |= set(json.load(open(fn))["uids"])
            df = df[~df.uid.isin(done)]
        df = (df.sample(n=min(a.n, len(df)), random_state=a.seed) if a.sample else df.head(a.n)).reset_index(drop=True)
        uids, dup = list(df.uid), [False] * len(df)
        stems = [os.path.join(a.rich, sh, "renders", u) for sh, u in zip(df.shard, df.uid)]
        extra = {"family": list(df.family), "gt_code": list(df.code), "stems": stems}
        a.modes = [m for m in a.modes if "expert" not in m]
    else:
        with open(a.uids_from, "rb") as f:
            P = pickle.load(f)
        uids, dup = P["uids"][: a.n], P["dup"][: a.n]
        stems = [os.path.join(a.data, "renders", u) for u in uids]
        with open(os.path.join(a.data, "text_index.json")) as f:
            text_index = json.load(f)
    reqs = []
    for i, u in enumerate(uids):
        if not os.path.exists(f"{stems[i]}_0.png"):
            continue                                   # GT program does not build: no render (scored as invalid)
        for mode in a.modes:
            prompt = text_index.get(u, {}).get("expert") if "expert" in mode else None
            if "expert" in mode and not prompt:
                continue
            reqs.append(Request(custom_id=f"{i}_{mode.replace('+', '-')}",   # API allows [a-zA-Z0-9_-] only
                                params=MessageCreateParamsNonStreaming(**request(stems[i], mode, prompt, a.effort,
                                                                                 bool(a.rich)))))
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=reqs)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "batch.json"), "w") as f:
        json.dump({"batch_id": batch.id, "uids": uids, "dup": list(map(bool, dup)), "modes": a.modes,
                   "n_requests": len(reqs), "model": MODEL, "effort": a.effort, **extra}, f)
    print(f"[claude] submitted batch {batch.id}: {len(reqs)} requests ({len(uids)} uids x {a.modes})")


def collect(a):
    import anthropic
    with open(os.path.join(a.out, "batch.json")) as f:
        meta = json.load(f)
    client = anthropic.Anthropic()
    ids = meta.get("batch_ids") or [meta["batch_id"]]
    while True:
        bs = [client.messages.batches.retrieve(i) for i in ids]
        if all(b.processing_status == "ended" for b in bs) or not a.wait:
            break
        print(f"[claude] {[b.processing_status for b in bs]}", flush=True)
        time.sleep(60)
    if not all(b.processing_status == "ended" for b in bs):
        print(f"[claude] batch not finished: {[(b.processing_status, b.request_counts) for b in bs]}")
        return
    n = len(meta["uids"])
    preds = {m: [None] * n for m in meta["modes"]}
    usage = {"input_tokens": 0, "output_tokens": 0}
    status = {}
    for r in (r for i in ids for r in client.messages.batches.results(i)):
        i, mode = r.custom_id.split("_", 1)
        mode = {m.replace("+", "-"): m for m in meta["modes"]}[mode]
        status[r.result.type] = status.get(r.result.type, 0) + 1
        if r.result.type != "succeeded":
            continue
        msg = r.result.message
        usage["input_tokens"] += msg.usage.input_tokens
        usage["output_tokens"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            continue
        text = "".join(blk.text for blk in msg.content if blk.type == "text")
        preds[mode][int(i)] = extract_code(text)
    with open(os.path.join(a.out, "preds.pkl"), "wb") as f:
        pickle.dump({"uids": meta["uids"], "dup": meta["dup"], "kind": "cadquery",
                     **{k: meta[k] for k in ("family", "gt_code", "stems") if k in meta},
                     "preds": {f"claude:{m}": v for m, v in preds.items()}}, f)
    # batch price: half of $4 / $20 per MTok for claude-opus-5-5
    cost = usage["input_tokens"] * 2e-6 + usage["output_tokens"] * 10e-6
    print(f"[claude] results {status}; tokens {usage}; batch cost ~${cost:.2f}; wrote {a.out}/preds.pkl")


RETRY = """A previous attempt at this part:
```python
{code}
```
{result}
Compare it with the part in the first image and write the corrected program (same requirements as above)."""


def retry(a):
    import anthropic
    from pebble import ProcessPool
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    from scripts.build_revision import work as render_attempt
    from mm.sysutil import usable_cpus
    P = pickle.load(open(os.path.join(a.src, "preds.pkl"), "rb"))
    R = pickle.load(open(os.path.join(a.src, "rich_rows.pkl"), "rb"))["rows"]["claude:view4"]
    codes = P["preds"]["claude:view4"]
    idx = [i for i, r in enumerate(R) if r is not None and r["own"] < a.below]
    print(f"[retry] {len(idx)} of {len(R)} attempts below {a.below}", flush=True)
    os.makedirs(os.path.join(a.out, "attempts"), exist_ok=True)
    fb = {}
    with ProcessPool(max_workers=usable_cpus()) as pool:
        futs = {i: pool.schedule(render_attempt, args=((str(i), codes[i] or "r = None", a.out),), timeout=120) for i in idx}
        for i, fu in futs.items():
            try:
                _, err, stem = fu.result()
            except Exception as e:  # noqa: BLE001
                err, stem = f"building it timed out ({type(e).__name__})", None
            fb[i] = (err, stem)
    reqs = []
    for i in idx:
        err, stem = fb[i]
        req = request(P["stems"][i], "view4", None, a.effort, True)
        if stem:
            req["messages"][0]["content"].insert(1, {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": grid_png(stem, "view4")}})
            result = ("The second image shows what this program builds, rendered with the same four cameras as the "
                      "first image.")
        else:
            result = f"It failed to build: {err or 'no program was produced'}"
        req["messages"][0]["content"][-1]["text"] += "\n\n" + RETRY.format(code=codes[i] or "(none)", result=result)
        reqs.append(Request(custom_id=f"{i}_view4", params=MessageCreateParamsNonStreaming(**req)))
    client = anthropic.Anthropic()
    ids = []
    for s0 in range(0, len(reqs), a.chunk):                # the batch API caps a batch at 256 MB
        ids.append(client.messages.batches.create(requests=reqs[s0:s0 + a.chunk]).id)
        print(f"[retry] submitted batch {ids[-1]} ({len(reqs[s0:s0 + a.chunk])} requests)", flush=True)
    meta = {"batch_ids": ids, "uids": P["uids"], "dup": P["dup"], "modes": ["view4"], "model": MODEL,
            "effort": a.effort, "family": P["family"], "gt_code": P["gt_code"], "stems": P["stems"],
            "attempt": {str(i): codes[i] for i in idx}, "attempt_stem": {str(i): fb[i][1] for i in idx},
            "attempt_error": {str(i): fb[i][0] for i in idx}, "src": a.src}
    json.dump(meta, open(os.path.join(a.out, "batch.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--data", default="data")
    s.add_argument("--uids_from", default=None, help="a baselines.py --save_preds file: same uids and dup mask")
    s.add_argument("--rich", default=None, help="rich/real part shards dir instead of DeepCAD (full-feature prompt)")
    s.add_argument("--sample", action="store_true", help="--rich: random --n parts across shards (teacher data)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--exclude_batch", nargs="*", default=None, help="batch.json files whose uids to skip")
    s.add_argument("--n", type=int, default=2000)
    s.add_argument("--modes", nargs="*", default=["view4", "view1", "view1+expert"])
    s.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--out", default="results/claude_baseline")
    c = sub.add_parser("collect")
    c.add_argument("--out", default="results/claude_baseline")
    c.add_argument("--wait", action="store_true")
    t = sub.add_parser("retry")
    t.add_argument("--src", required=True, help="a scored --rich run (preds.pkl + rich_rows.pkl)")
    t.add_argument("--below", type=float, default=0.8)
    t.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    t.add_argument("--chunk", type=int, default=2500)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    {"submit": submit, "collect": collect, "retry": retry}[a.cmd](a)


if __name__ == "__main__":
    main()
