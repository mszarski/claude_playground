"""Answer heard clips with the distilled responder (one call: feeling, reading, response, recipe).

  python scripts/student_respond.py --model runs/student/merged --heard runs/respond_eval/heard.jsonl --out student.jsonl

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
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer

    from rmr.planner.evaluate import parse_answer
    from rmr.planner.finetune import lm_class, resolve
    from rmr.planner.llm import parse_json
    from rmr.respond import FEELINGS, student_messages

    path = resolve(a.model)
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = lm_class(path).from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    rows = [json.loads(l) for l in open(a.heard)]
    t0, outs = time.time(), []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        texts = [tok.apply_chat_template(student_messages(r), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for r, t in zip(chunk, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            try:
                d = parse_json(t)
            except Exception:
                d = {}
            feeling = str(d.get("feeling", "")).lower()
            recipe = d.get("recipe") if isinstance(d.get("recipe"), str) else None
            outs.append({"clip": r["clip"], "label": r.get("label"), "raw": t,
                         "both": {"feeling": feeling if feeling in FEELINGS else "invalid",
                                  "reading": d.get("reading", ""), "response": d.get("response", "")},
                         "recipe": recipe if recipe and parse_answer(json.dumps({"recipe": recipe})) else None})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in outs)
    print(f"{len(outs)} answers in {time.time() - t0:.0f} s ({(time.time() - t0) / max(len(outs), 1):.2f} s each, batched) "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
