"""Greedy test-set predictions of a vlm_cq-style model with vLLM: same inputs, prompt format and output file as
train_vlm_cq.py --predict (view1 = render 0, view4 = 2x2 grid, + the expert prompt, or the prompt alone), ~10x faster.

  python scripts/predict_vllm.py --model results/vlm_dpo_v1/model --uids_from results/baselines_vlm_final/preds.pkl \
      --out results/vlm_dpo_v1_test/preds.pkl            # then scripts/cq_iou.py locally
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--uids_from", required=True)
    ap.add_argument("--modes", nargs="*", default=["view1", "view4", "view1+expert", "expert"])
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # FlashInfer JIT-compiles its sampler: needs nvcc
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from mm.vlm_cad import _messages, compose_views
    from train_vlm_cad import load_views

    with open(a.uids_from, "rb") as f:
        P = pickle.load(f)
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)
    render_dir = os.path.join(a.data, "renders")
    processor = AutoProcessor.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85, enable_prefix_caching=True, generation_config="vllm", seed=0)
    stop = [processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), processor.tokenizer.eos_token_id]
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, stop_token_ids=stop)
    max_prompt = a.max_len - a.max_new_tokens - 400
    preds = {}
    for mode in a.modes:
        t0 = time.time()
        reqs, idx = [], []
        for i, u in enumerate(P["uids"]):
            prompt = text_index.get(u, {}).get("expert") if "expert" in mode else None
            if prompt and len(processor.tokenizer(prompt)["input_ids"]) > max_prompt:
                prompt = None
            stem = os.path.join(render_dir, u)
            image = None
            if mode.startswith("view"):
                if not os.path.exists(f"{stem}_0.png"):
                    continue                          # GT does not build: no render (scored as invalid)
                image = compose_views(load_views("", stem, range(int(mode[4]))))
            elif not prompt:
                continue
            text = processor.apply_chat_template(_messages(image, prompt), tokenize=False, add_generation_prompt=True)
            reqs.append({"prompt": text, **({"multi_modal_data": {"image": image}} if image is not None else {})})
            idx.append(i)
        rows = [None] * len(P["uids"])
        for i, o in zip(idx, llm.generate(reqs, sp, use_tqdm=False)):
            rows[i] = o.outputs[0].text.strip() or None
        preds[f"vlm_cq:{mode}"] = rows
        print(f"[predict] {mode}: {len(reqs)} requests in {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"uids": P["uids"], "dup": P["dup"], "kind": "cadquery", "preds": preds}, f)
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
