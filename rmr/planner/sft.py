"""data/teacher/dataset.jsonl -> chat SFT set for the planner (train.jsonl, val.jsonl).

  python -m rmr.planner.sft --out runs/sft

- rows are repeated by their ``weight`` (build-up/release events x3); only the ``--sources`` the served models used
- rows whose prompt or family mentions a core out-of-distribution probe concept are dropped (``--block``), and so are
  prompts within cosine ``--leak`` of any evaluation or probe prompt (Qwen3-Embedding-0.6B), so the probes and the
  held-out emotions stay a real generalisation test
- the 39 prompts of data/teacher/val.jsonl are never trained on; val.jsonl holds their first teacher recipe
- every "word. sentence." training prompt is also trained as "word." and as "sentence." alone

Needs ``transformers`` and ``torch`` for the embedder (``--leak 1`` skips it).

Reference: ``planner/distill/sft.py`` and ``planner/distill/common.py`` in pham-tuan-binh/reachy-motion-generator
(Apache-2.0).
"""
import argparse
import json
import os
import random
import re

import numpy as np

from ..recipe import check
from . import read_lines
from .probes import PROBES
from .prompt import student_messages

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "teacher")
SOURCES = "claude,claude_events,seed,astra_lively"
BLOCK = r"sneez|startl|drunk|dizz|toddler|stalk|pounc|heartbr|ecstat"


class LocalEmbedder:
    """Qwen3-Embedding-0.6B, last-token pooling: the encoder the 0.72 leak threshold was calibrated on."""

    def __init__(self, name="Qwen/Qwen3-Embedding-0.6B", device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.m = AutoModel.from_pretrained(name, dtype=dtype).to(self.dev).eval()

    def __call__(self, texts, batch=128):
        out = []
        for i in range(0, len(texts), batch):
            b = self.tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=96).to(self.dev)
            with self.torch.no_grad():
                h = self.m(**b).last_hidden_state[:, -1].float()      # left padding -> last token
            out.append(self.torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out)


def write_jsonl(rows, path):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def build(dataset, val_file, eval_file, sources=SOURCES, leak=0.72, block=BLOCK, seed=0, embedder=None, log=print):
    """-> ``(train rows, val rows, leaked prompts)``; rows are ``{"messages", "source"}``."""
    keep, rows = set(filter(None, sources.split(","))), []
    for r in map(json.loads, open(dataset)):
        if not keep or r["source"] in keep:
            rows += [r] * int(r.get("weight", 1))
    n0 = len(rows)
    rows = [r for r in rows if not re.search(block, r["prompt"] + " " + r.get("family", ""), re.I)]
    log(f"blocklist: dropped {n0 - len(rows)} rows")
    rows = [r for r in rows if not check(r["recipe"])]

    leaked = set()
    if leak < 1:
        emb = embedder or LocalEmbedder()
        ps = sorted({r["prompt"] for r in rows})
        ev = read_lines(eval_file) + [p for p, _, _ in PROBES]
        sim = dict(zip(ps, (emb(ps) @ emb(ev).T).max(1)))
        leaked = {p for p in ps if sim[p] > leak}
        log(f"leak filter: dropped {len(leaked)} prompts > {leak} to an eval prompt, e.g. {sorted(leaked)[:6]}")

    val = [json.loads(l) for l in open(val_file)]
    val_p = {v["prompt"] for v in val}
    tr = [r for r in rows if r["prompt"] not in leaked and r["prompt"] not in val_p]
    extra = []
    for r in tr:
        w, _, rest = r["prompt"].partition(". ")
        if rest:
            extra += [dict(r, prompt=w + "."), dict(r, prompt=rest)]
    tr += list({(e["prompt"], e["recipe"]): e for e in extra}.values())     # repeated rows need not repeat variants
    random.Random(seed).shuffle(tr)
    train_rows = [dict(messages=student_messages(r["prompt"], r["idea"], r["recipe"]), source=r["source"]) for r in tr]
    val_rows = [dict(messages=student_messages(v["prompt"], v["labels"][0]["idea"], v["labels"][0]["recipe"]),
                     source="val") for v in val]
    return train_rows, val_rows, leaked


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.planner.sft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=os.path.join(DATA, "dataset.jsonl"))
    ap.add_argument("--val-file", default=os.path.join(DATA, "val.jsonl"))
    ap.add_argument("--eval", default=os.path.join(DATA, "eval_prompts.txt"), help="evaluation prompts, besides the probes")
    ap.add_argument("--sources", default=SOURCES, help="comma list of dataset sources to keep ('' = all)")
    ap.add_argument("--leak", type=float, default=0.72)
    ap.add_argument("--block", default=BLOCK)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tr, val, leaked = build(a.dataset, a.val_file, a.eval, a.sources, a.leak, a.block, a.seed)
    os.makedirs(a.out, exist_ok=True)
    write_jsonl(tr, f"{a.out}/train.jsonl")
    write_jsonl(val, f"{a.out}/val.jsonl")
    with open(f"{a.out}/leaked.json", "w") as f:
        json.dump(sorted(leaked), f, indent=1)
    print(f"train {len(tr)} rows | val {len(val)} prompts -> {a.out}")


if __name__ == "__main__":
    main()
