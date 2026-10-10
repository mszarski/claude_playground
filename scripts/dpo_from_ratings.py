"""Preference-tune the voice student on human ratings from the rating page (DPO).

  # 1. export the page's ratings (the ArtifactData tool, collection "ratings") to runs/rating/ratings.json
  # 2. build the pairs
  python scripts/dpo_from_ratings.py pairs --items runs/rating/items.json --ratings runs/rating/ratings.json \
      --heard runs/listen_train/heard.jsonl --out runs/rating/dpo.jsonl
  # 3. train (HF Jobs, like the SFT students)
  python scripts/dpo_from_ratings.py train --data runs/rating/dpo.jsonl --model mszarski/reachy-voice:student/v3 \
      --out /work/out

Only the page's *train* pool counts (MELD train clips, sampled answers of the student): the eval pool is MELD test
data and stays held out. A pick between two answers gives one {prompt, chosen, rejected} pair; best and worst among
k answers give 2k - 3 (best over each other, each other over worst). With several raters, a pair's majority wins and
ties are dropped.
"""
import argparse
import collections
import json


def pairs(a):
    from rmr.respond import ANSWER_ORDER, student_messages

    items = {it["id"]: it for it in json.load(open(a.items))}
    heard = {}
    for line in open(a.heard):
        r = json.loads(line)
        heard[r["clip"]] = r
    raw = json.load(open(a.ratings))
    docs = raw if isinstance(raw, list) else raw.get("documents", raw.get("docs", []))
    # each judgement -> ordered pairs (winner index, loser index): a/b votes give one; best-worst over k answers gives
    # best > every other and every other > worst (2k - 3 pairs). Several raters: a pair's majority wins, ties drop.
    prefs = collections.defaultdict(collections.Counter)
    for d in docs:
        body = d.get("data", d)
        for item_id, v in (body.get("votes") or {}).items():
            it = items.get(item_id)
            if not it or it["pool"] != "train" or v.get("same"):
                continue
            if "best" in v:
                k = len(it["answers"])
                b, w = v["best"], v["worst"]
                pairs_ = {(b, j) for j in range(k) if j != b} | {(j, w) for j in range(k) if j not in (b, w)}
            elif v.get("choice") in ("a", "b"):
                pairs_ = {(0, 1) if v["choice"] == "a" else (1, 0)}
            else:
                continue
            for x, y in pairs_:
                prefs[(item_id, min(x, y), max(x, y))][x] += 1
    out = []
    for (item_id, x, y), c in prefs.items():
        if c[x] == c[y]:
            continue
        it = items[item_id]
        answers = it.get("answers") or [it["a"], it["b"]]
        win, lose = (x, y) if c[x] > c[y] else (y, x)
        prompt = student_messages(heard[it["clip"]])
        ans = lambda j: json.dumps({f: answers[j][f] for f in ANSWER_ORDER})          # noqa: E731
        out.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": ans(win)}],
                    "rejected": [{"role": "assistant", "content": ans(lose)}], "clip": it["clip"],
                    "votes": {str(k): n for k, n in c.items()}})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in out)
    print(f"{len(out)} preference pairs from {len({k[0] for k in prefs})} rated training clips -> {a.out}")


def train(a):
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from rmr.planner.finetune import lm_class, resolve

    path = resolve(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    model = lm_class(path).from_pretrained(path, dtype="bfloat16")
    ds = load_dataset("json", data_files=a.data, split="train").remove_columns(["clip", "votes"])
    cfg = DPOConfig(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=4,
                    gradient_accumulation_steps=4, learning_rate=a.lr, beta=a.beta, logging_steps=5,
                    save_strategy="no", bf16=True, report_to=[])
    tr = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    tr.train()
    merged = tr.model.merge_and_unload()
    merged.save_pretrained(f"{a.out}/merged")
    tok.save_pretrained(f"{a.out}/merged")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--items", required=True)
    p.add_argument("--ratings", required=True)
    p.add_argument("--heard", default="runs/listen_train/heard.jsonl")
    p.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--lr", type=float, default=5e-6)
    t.add_argument("--beta", type=float, default=0.1)
    a = ap.parse_args()
    pairs(a) if a.cmd == "pairs" else train(a)


if __name__ == "__main__":
    main()
