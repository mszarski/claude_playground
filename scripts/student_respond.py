"""Answer heard clips with the distilled responder (one call: feeling, reading, response, recipe).

  python scripts/student_respond.py --model runs/student/merged --heard runs/respond_eval/heard.jsonl --out student.jsonl
  python scripts/student_respond.py --model http://localhost:8080/v1 ...    # the GGUF build behind llama-server

Writes one JSON line per clip in the shape scripts/eval_respond.py scores ({"clip", "label", "both": {...}, "recipe"}).
"""
import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (endpoint only)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips")
    ap.add_argument("--samples", type=int, default=1, help="answers per clip (endpoint only; use --temperature)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="with --limit: a random subset instead of the first N")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.heard)]
    if a.seed and a.limit:
        import random
        rows = random.Random(a.seed).sample(rows, a.limit)
    rows = rows[:a.limit]
    if a.model.startswith(("http://", "https://")):
        return endpoint(a, rows)
    import torch
    from transformers import AutoTokenizer

    from rmr.planner.finetune import lm_class, resolve
    from rmr.respond import student_messages

    path = resolve(a.model)
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = lm_class(path).from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    t0, outs = time.time(), []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        texts = [tok.apply_chat_template(student_messages(r), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for r, t in zip(chunk, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            outs.append(answer(r, t))
    write(a, outs, t0, "batched")


def answer(r, t):
    from rmr.planner.evaluate import parse_answer
    from rmr.planner.llm import parse_json
    from rmr.respond import FEELINGS

    try:
        d = parse_json(t)
    except Exception:
        d = {}
    feeling = str(d.get("feeling", "")).lower()
    recipe = d.get("recipe") if isinstance(d.get("recipe"), str) else None
    return {"clip": r["clip"], "label": r.get("label"), "raw": t,
            "both": {"feeling": feeling if feeling in FEELINGS else "invalid",
                     "reading": d.get("reading", ""), "response": d.get("response", "")},
            "recipe": recipe if recipe and parse_answer(json.dumps({"recipe": recipe})) else None}


def endpoint(a, rows):
    from concurrent.futures import ThreadPoolExecutor

    from rmr.respond import Student

    st = Student(a.model, temperature=a.temperature)
    rows = [r for r in rows for _ in range(a.samples)]

    def one(r):
        t = time.time()
        try:
            raw = st._generate(r)
        except Exception as e:
            raw = f"ERROR {type(e).__name__}: {e}"
        return {**answer(r, raw), "seconds": round(time.time() - t, 2)}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(one, rows))
    lat = sorted(o["seconds"] for o in outs)
    print(f"latency per answer: median {lat[len(lat) // 2]:.1f} s, max {lat[-1]:.1f} s ({a.workers} in parallel)")
    write(a, outs, t0, f"{a.workers} in parallel")


def write(a, outs, t0, how):
    with open(a.out, "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in outs)
    print(f"{len(outs)} answers in {time.time() - t0:.0f} s ({(time.time() - t0) / max(len(outs), 1):.2f} s each, batched) "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
