#!/usr/bin/env python3
"""Score a hosted model zero-shot on the test split through any OpenAI-compatible endpoint.

Defaults to the Hugging Face Inference router (uses $HF_TOKEN). This sets the bar
that the fine-tuned small model has to beat.

    python scripts/eval_baseline.py --model meta-llama/Llama-3.3-70B-Instruct --n 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from design_edit.evaluate import evaluate, format_report  # noqa: E402


def chat(base_url: str, token: str, model: str, messages: list[dict], max_tokens: int = 8192) -> str:
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}).encode()
    for attempt in range(4):
        req = urllib.request.Request(f"{base_url}/chat/completions", data=body, method="POST", headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["choices"][0]["message"]["content"] or ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == 3:
                return f"// request failed: {e}"
            time.sleep(2 ** (attempt + 1))
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--base-url", default="https://router.huggingface.co/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="write per-row results as JSON")
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]

    rows = [json.loads(line) for line in open(args.data)][: args.n]

    def generate(prompts):
        with ThreadPoolExecutor(args.workers) as pool:
            return list(pool.map(lambda p: chat(args.base_url, token, args.model, p), prompts))

    result = evaluate(rows, generate, batch_size=args.workers * 4)
    print(f"model: {args.model}  (n={len(rows)})")
    print(format_report(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
