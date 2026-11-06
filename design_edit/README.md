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
| `design_edit/edits.py` | 18 edit types. **Forward** edits (recolor, retext, resize, spacing, radius, add, delete, reorder, layout, align, opacity) and **repairs** (break a clean design, ask for the fix: contrast, off-scale spacing or type, inconsistent buttons, missing label), plus multi-part `compose`. `NOVEL` holds 7 more edit types that are **eval-only**. Vague requests carry `relax` rules ("bigger" means any larger value; "yellow" means any yellow). |
| `design_edit/rewards.py` | Scores a patch by its **effect**, not its text. Measures: valid, F1 over change facts, exact match, and new design-lint violations. Folds them into one scalar for RL. |
| `design_edit/evaluate.py` | Evaluation that works with any model, with a per-edit-type report. |
| `design_edit/render.py` | Renders a design to HTML or PNG (Playwright), for viewing data or for a VLM judge. |
| `scripts/build_dataset.py` | Builds train and test sets (test uses disjoint design seeds) and can push them to the Hub. |
| `scripts/paraphrase.py` | Rewrites requests in varied human voices via the HF router. Keeps quoted text, numbers and hex values verbatim, and checks equivalence with an LLM judge. |
| `scripts/eval_model.py` | Scores a Hub model on any splits (PEP 723, runs on HF Jobs). |
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

## Results

Exact-match rate. The SFT model is Qwen3-4B-Instruct + LoRA, 1 epoch on 10k examples (about 51 min on one
RTX PRO 6000, about $3): [`mszarski/design-edit-sft`](https://huggingface.co/mszarski/design-edit-sft) (private),
with per-example outputs in its `eval_*.json` files. The baselines are zero-shot through the HF router, on the
first 150 rows of each split.

| split | what it tests | Llama-3.3-70B | Qwen3.8-27B (thinking) | **4B SFT** |
|---|---|---|---|---|
| `test` (500) | unseen designs, training templates | 0.707 | 0.860 | **0.990** |
| `test_paraphrased` (490) | same edits, requests rewritten by an LLM in human voices | 0.633 | 0.787 | **0.924** |
| `test_novel` (300) | 7 edit types never seen in training | 0.673 | **0.933** | 0.577 |
| `test_novel_paraphrased` (290) | both | 0.620 | **0.793** | 0.614 |

Patch validity is 1.00 for the SFT model on every split: it always produces a patch that parses and applies.

**Reworded requests: fine.** SFT loses 7 points (99 → 92) from rewording, the same drop as the large models,
and still beats them by 13+ points. The model learned to resolve messy references ("the text that says
'Join thousands…'", "the top bar thing"), not just the template wording.

**New edit types: the gap is data coverage.** Training targets contain only `set` / `insert` / `move` / `delete`,
and never edit `justify`, `weight`, `stroke` or `w`. Per novel edit type:

| edit | SFT exact | failure |
|---|---|---|
| `add_border` | 1.00 | (`stroke` unseen but named clearly: generalizes) |
| `set_width` | 1.00 | (`w` unseen: generalizes) |
| `set_weight` | 0.64 | uses 600 for "bold" (reference: 700); debatable |
| `darken_text` | 0.47 | no-op color, or `fill` instead of `color` on text |
| `promote_button` | 0.00 | sets the fill but never emits `unset stroke`; `unset` never appears in training |
| `spread_out` | 0.00 | "spread out" becomes a larger `gap`; `justify` edits never appear in training |
| `duplicate_setting` | 0.00 | copied subtree is right, but it is inserted under the wrong parent or at the wrong index |

The model generalizes to new *properties* when the request names them, but not to an unseen *operation*
(`unset`), an unseen mapping from intent to property ("spread out" means `justify`), or new insertion arithmetic.

**Label noise.** A hand-check of 30 random rewrites found about 1–2 that change the meaning, mostly
`spread_out` ("push apart" reworded as "increase the gap"). This is despite the value-preservation check and the
step-by-step equivalence judge in `scripts/paraphrase.py`.

## Design decisions and next steps

- **Patches, not rewrites.** Output is about 1–5 lines instead of a 2k-token document. It is cheap to decode, easy to verify,
  and can't silently change unrelated nodes.
- **Labels come from `diff`, never hand-written.** Every SFT target applies cleanly by construction. The tests check
  `apply(before, diff(before, after)) == after` for every edit type.
- **Reward on effects.** Different-but-equivalent patches (another id for a new node, any yellow for "yellow") get full credit.
  This matters most for RL, where a text-match reward would punish correct answers.
- **Repairs are reverse edits.** Break a clean design in a known way, so the correct fix is known exactly. This is the cheapest
  source of "make it look right" data.

Not built yet, in priority order:
- **Cover the operation space in training.** Add training edits that use `unset`, `justify` and `weight`, and
  subtree duplication. Keep a *different* set of novel edits held out, so `test_novel` stays a real generalization test.
- **Paraphrase part of the training set** with `scripts/paraphrase.py`, to close the 7-point rewording gap.
- **GRPO where the model is close.** RL needs some correct samples per group. Measure pass@8 on the
  failing edit types first. `promote_button` (one missing line) and `duplicate_setting` (right content, wrong
  index) are good candidates; `spread_out` probably needs data first.
- **Real designs.** Import Figma Community files through the REST API and convert them to the DSL. Synthetic templates are narrow.
- **More natural requests.** Have an LLM rewrite the template requests, and keep a held-out set of human-written requests.
- **VLM judge reward.** Use `render.screenshot` on before and after, then have a vision model grade against designer-written
  criteria. Add it as a third reward function in `train_grpo.py` for requests the programmatic checks can't verify.
