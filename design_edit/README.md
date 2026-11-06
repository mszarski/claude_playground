# design_edit: training a small design-editing model

A small, open version of the recipe in Figma's
["How we're training a Figma-fluent editing model"](https://www.figma.com/blog/training-a-figma-fluent-editing-model/).
It generates a large set of synthetic (design, request) → edit examples, runs SFT on an open model, then runs RL (GRPO)
with a reward computed by actually applying each predicted edit to the design.

```
generator.py ──► designs ──► edits.py ──► (before, request, after) ──► schema.diff ──► reference patch
                                                     │
                       SFT (train_sft.py) ◄──────────┤
                       GRPO (train_grpo.py) ◄── rewards.score(apply(patch)) vs. after
```

## Pieces

| File | What it does |
|---|---|
| `design_edit/schema.py` | Design tree, a one-node-per-line **design DSL**, the **patch language** (`set` / `unset` / `delete` / `move` / `insert`), `apply_patch`, and `diff(a, b)`, which builds the canonical patch from `a` to `b`. |
| `design_edit/generator.py` | Procedural UI designs (cards, forms, navbars, pricing tables, settings lists) built on spacing, type and color scales. Nodes get semantic names that requests can refer to. |
| `design_edit/edits.py` | 18 edit types. **Forward** edits (recolor, retext, resize, spacing, radius, add, delete, reorder, layout, align, opacity) and **repairs** (break a clean design, ask for the fix: contrast, off-scale spacing or type, inconsistent buttons, missing label), plus multi-part `compose`. Vague requests carry `relax` rules ("bigger" means any larger value; "yellow" means any yellow). |
| `design_edit/rewards.py` | Scores a patch by its **effect**, not its text. Measures: valid, F1 over change facts, exact match, and new design-lint violations. Folds them into one scalar for RL. |
| `design_edit/evaluate.py` | Evaluation that works with any model, with a per-edit-type report. |
| `design_edit/render.py` | Renders a design to HTML or PNG (Playwright), for viewing data or for a VLM judge. |
| `scripts/build_dataset.py` | Builds train and test sets (test uses disjoint design seeds) and can push them to the Hub. |
| `scripts/eval_baseline.py` | Zero-shot baseline for any OpenAI-compatible endpoint (default: HF Inference router). |
| `scripts/train_sft.py` | Stage 1: LoRA SFT with loss on the patch only. Merges the adapter, pushes the model, evaluates. A PEP 723 script, runnable on HF Jobs. |
| `scripts/train_grpo.py` | Stage 2: GRPO from the SFT checkpoint, rewarded by `rewards.score`. |

Example input to the model:

```
Design:
frame#n1 "Page" layout=col w=1280 gap=32 pad=40 align=center fill=#F9FAFB
  frame#n10 "Card" layout=col w=360 gap=16 pad=24 fill=#FFFFFF stroke=#E5E7EB radius=10
    text#n11 "Title" text="Order confirmed" size=24 weight=600 color=#111827
    ...
Request: Insert a muted line of text below the Title: "Here's what happened while you were away.", and make the Settings 40% transparent
```

Expected output (a patch, not a rewritten document):

```
insert n10 1 text#n18 "Subtitle" text="Here's what happened while you were away." size=14 weight=400 color=#6B7280
set n2 opacity=0.6
```

## Running it

```bash
pip install datasets huggingface_hub   # data + baselines only; training runs on HF Jobs
pytest design_edit/tests -q

# 1. data (about 15 s for 20k examples)
python scripts/build_dataset.py --n-train 20000 --n-test 500 --out data/ --push <user>/design-edits

# 2. zero-shot baselines through the HF router
python scripts/eval_baseline.py --model meta-llama/Llama-3.3-70B-Instruct --data data/test.jsonl --n 150

# 3. SFT on HF Jobs (the script fetches design_edit/ from the dataset repo's code/ folder;
#    upload it with: hf upload <user>/design-edits . code --repo-type dataset --include "design_edit/*.py" "scripts/*.py")
hf jobs uv run --flavor a100-large --timeout 3h --secrets HF_TOKEN scripts/train_sft.py \
    --dataset <user>/design-edits --output <user>/design-edit-sft

# 4. GRPO from the SFT model
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN scripts/train_grpo.py \
    --dataset <user>/design-edits --model <user>/design-edit-sft --output <user>/design-edit-grpo
```

## Baselines (zero-shot, 150 test examples)

| model | valid | exact | F1 |
|---|---|---|---|
| Llama-3.3-70B-Instruct | 0.907 | 0.707 | 0.770 |
| Qwen3.8-27B (thinking) | 1.000 | 0.860 | 0.908 |

## Design decisions and next steps

- **Patches, not rewrites.** Output is about 1–5 lines instead of a 2k-token document. It is cheap to decode, easy to verify,
  and can't silently change unrelated nodes.
- **Labels come from `diff`, never hand-written.** Every SFT target applies cleanly by construction. The tests check
  `apply(before, diff(before, after)) == after` for every edit type.
- **Reward on effects.** Different-but-equivalent patches (another id for a new node, any yellow for "yellow") get full credit.
  This matters most for RL, where a text-match reward would punish correct answers.
- **Repairs are reverse edits.** Break a clean design in a known way, so the correct fix is known exactly. This is the cheapest
  source of "make it look right" data.

Not built yet:
- **Real designs.** Import Figma Community files through the REST API and convert them to the DSL. Synthetic templates are narrow.
- **More natural requests.** Have an LLM rewrite the template requests, and keep a held-out set of human-written requests.
- **VLM judge reward.** Use `render.screenshot` on before and after, then have a vision model grade against designer-written
  criteria. Add it as a third reward function in `train_grpo.py` for requests the programmatic checks can't verify.
- **Generalization eval.** Use `build_dataset.py --holdout-kinds` to measure transfer to unseen edit types.
