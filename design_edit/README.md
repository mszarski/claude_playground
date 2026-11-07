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

## v2: covering the missing operations

v2 adds 6 training edits that use exactly the operations v1 never trained on, through *different* requests than
the held-out novel edits: `remove_border` and `reset_opacity` (`unset`), `justify_items` (`justify=start/center/end`),
`set_weight_value` (400 or 500), `lighten_text`, and `add_field` (a 4-node subtree insert). 2,452 of the 12k
training rows are LLM-reworded. Same recipe (1 epoch, LoRA r=32, about 61 min, about $3):
[`mszarski/design-edit-sft-v2`](https://huggingface.co/mszarski/design-edit-sft-v2), data in
[`mszarski/design-edits-v2`](https://huggingface.co/datasets/mszarski/design-edits-v2).

| split | v1 SFT | **v2 SFT** |
|---|---|---|
| `test` | 0.990 | **0.998** |
| `test_paraphrased` | 0.924 | **0.945** |
| `test_coverage` (the 6 new edits) | – | **1.000** |
| `test_novel` | **0.577** | 0.553 |
| `test_novel_paraphrased` | **0.614** | 0.579 |

**Knowing an operation did not transfer to new requests that need it.** The model learned every new edit
perfectly (1.000), but the held-out ones didn't move:

| novel edit | v1 | v2 | what changed |
|---|---|---|---|
| `spread_out` | 0.00 | 0.00 | now reaches for `justify` in half the cases (v1: never), but picks `end`, the value from the new "push right" requests |
| `promote_button` | 0.00 | 0.00 | still never emits `unset`, though it does so perfectly for "remove the border" |
| `duplicate_setting` | 0.00 | 0.00 | content right, insert index still wrong |
| `darken_text` | 0.47 | 0.41 | **negative transfer**: "bump up the contrast" now often outputs `#6B7280`, the exact color `lighten_text` targets (a no-op) |

SFT learned *request template → patch template* mappings, including specific values, rather than composable
operations. This is the failure mode Figma cites for moving from imitation learning to RL. It makes the
held-out novel splits the right test for whether GRPO generalizes better than SFT. GRPO must train only on
training edit types, so the novel splits stay held out.

## Is RL viable? pass@8 before GRPO

`scripts/sample_model.py` draws 8 samples per request (vLLM, T=1.0) from v2 SFT. GRPO can only reinforce a
patch it actually samples, and only learns from groups whose rewards differ.

| held-out novel edit | pass@8 | GRPO can help? |
|---|---|---|
| `set_weight` | 0.80–0.93 | yes |
| `darken_text` | 0.67–0.70 | yes |
| `duplicate_setting` | 0.07 | barely |
| `spread_out` | 0.00–0.03 | no: `justify=between` is never sampled |
| `promote_button` | 0.00 | no: `unset stroke` is never sampled alongside the fill change |

For two of the three 0% edits, RL from this checkpoint *cannot* help: the correct patch is outside what the policy
samples, so the fix there is training data. On the training side, only 12.6% of the 1,500-prompt GRPO pool
(reworded and multi-part requests) has reward variance across 8 samples. The rest are solved every time and give
zero advantage, so GRPO trains only on the variance-bearing subset (`grpo_train`).

## GRPO from v2: did not generalize

GRPO (TRL 1.17, vLLM colocated, LoRA r=32, DAPO loss, no KL, 8 generations, T=1.0) from v2 SFT on the 974
training prompts with reward variance (`grpo_train`), 1.5 epochs, about 87 min on one RTX PRO 6000 (about $4):
[`mszarski/design-edit-grpo`](https://huggingface.co/mszarski/design-edit-grpo). Mean training reward on those
prompts went from 1.437 (first quarter of the run) to 1.530 (last quarter), out of a maximum of 1.6.

| split | v1 SFT | v2 SFT | **v2 + GRPO** |
|---|---|---|---|
| `test` | 0.990 | 0.998 | 0.994 |
| `test_paraphrased` | 0.924 | 0.945 | 0.941 |
| `test_coverage` | – | 1.000 | 1.000 |
| `test_novel` | 0.577 | 0.553 | 0.513 |
| `test_novel_paraphrased` | 0.614 | 0.579 | 0.562 |

| held-out novel edit | v2 | v2 + GRPO |
|---|---|---|
| `darken_text` | 0.41 | **0.20** |
| `set_weight` | 0.57 | 0.54 |
| `duplicate_setting` | 0.00 | 0.07 |
| `promote_button` / `spread_out` | 0.00 | 0.00 |

GRPO improved the prompts it trained on and was flat on in-distribution tests, but it **reduced** held-out
generalization. The clearest case is `darken_text`: 41 of 51 outputs are now `#6B7280`, the `lighten_text` value
(a no-op), against about 30 under v2. RL on the policy's own samples *sharpened* its most likely answer, and where
that answer was wrong for a new request, it got more wrong. Design-lint violations on the novel split dropped
(0.020 → 0.003), so the lint term of the reward did have an effect.

**What this says about the recipe.** At this scale, with a synthetic distribution the SFT model already solves,
GRPO polishes but does not add capabilities. pass@8 showed the missing behaviors (`unset` plus fill for "primary",
`justify=between` for "spread out") are never sampled, so no reward can reach them. Figma's claim that RL
generalizes better than imitation fits a setting with **diverse environments and expert-written criteria**,
where the policy explores across many tasks. Polishing a narrow set of templates is not that.

## v3: diversity instead of RL

v3 keeps the recipe (fresh LoRA SFT from Qwen3-4B, 1 epoch, about 61 min, about $3) and changes only the data:
target values are randomized (`lighten_text` now picks one of four grays, where it used to always be `#6B7280`),
and four broad edit families are added: `set_prop` and `unset_prop` (any property, value stated), `restyle_button`
(an explicit fill + label + border change), and `duplicate_node` (any subtree copied after itself). 28 training
edit types; 2,370 of 11,970 rows reworded. `justify=between` and goal-level "primary style" stay held out.
[`mszarski/design-edit-sft-v3`](https://huggingface.co/mszarski/design-edit-sft-v3),
data [`mszarski/design-edits-v3`](https://huggingface.co/datasets/mszarski/design-edits-v3).

| split | v1 SFT | v2 SFT | v2 + GRPO | **v3 SFT** |
|---|---|---|---|---|
| `test` | 0.990 | 0.998 | 0.994 | 0.996 |
| `test_paraphrased` | 0.924 | 0.945 | 0.941 | 0.943 |
| `test_novel` | 0.577 | 0.553 | 0.513 | **0.657** |
| `test_novel_paraphrased` | 0.614 | 0.579 | 0.562 | **0.652** |
| `test_coverage` / `test_coverage_v3` | – | 1.000 / – | 1.000 / – | 1.000 / 1.000 |

| held-out novel edit | v2 | GRPO | **v3** | v3 pass@8 |
|---|---|---|---|---|
| `darken_text` | 0.41 | 0.20 | **0.80** (0.91 reworded) | 0.92 |
| `duplicate_setting` | 0.00 | 0.07 | **0.64** | 0.79 |
| `set_weight` ("bold") | 0.57 | 0.54 | 0.61 | 0.87 |
| `promote_button` | 0.00 | 0.00 | 0.00 | 0.00 |
| `spread_out` | 0.00 | 0.00 | 0.00 | 0.00 |

**Diversity did what RL could not.** Removing the fixed `#6B7280` lifted `darken_text` from 0.41 to 0.80.
Generic `duplicate_node` (copy any subtree after itself) transferred to the held-out "duplicate this settings row,
rename it, switch it off": 0.00 → 0.64. It is the first held-out edit that went from impossible to mostly right.

**What still fails is turning a goal into concrete changes.** For "promote X to primary style", v3 changes only the fill,
usually to the wrong color (the brand color appears in 5 of 26 outputs). It never removes the border or whitens the
label, even though it does exactly that for `restyle_button` when each step is stated (1.000). "Spread out" still
becomes `gap` or `justify=start`. pass@8 is 0 for both, so this checkpoint still can't be RL-trained into them.
These two need goal-level training requests ("make it match the other primary buttons", "space them evenly"),
probably with designer-written criteria, which is where Figma puts its RL investment.

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
- **Goal-level requests in training** ("match the other primary buttons", "tidy this up"), whose patch needs the
  model to work out several concrete changes. v3 shows the operations transfer once they're varied, but the goal-to-changes step does not.
- **Real designs** (Figma REST API; needs a Figma token) for broader structure than the 5 templates.
- **GRPO from v3 on held-out-style goals.** v3's pass@8 is 0.79–0.92 on `duplicate_setting` / `darken_text`, so RL
  now has signal there. It still needs training goals that aren't the test set.
- **GRPO with entropy kept up**: use higher temperature, or a KL or entropy bonus, so RL doesn't collapse onto the SFT
  mode, and train on a broader prompt mix rather than only the variance-bearing subset.
- **More varied training values.** v2 shows the model copies specific values from templates (`#6B7280`, `end`).
  Randomizing target values within each edit type should push it toward reading the request.
- **GRPO where the model is close.** RL needs some correct samples per group. Measure pass@8 on the
  failing edit types first. `promote_button` (one missing line) and `duplicate_setting` (right content, wrong
  index) are good candidates; `spread_out` probably needs data first.
- **Real designs.** Import Figma Community files through the REST API and convert them to the DSL. Synthetic templates are narrow.
- **More natural requests.** Have an LLM rewrite the template requests, and keep a held-out set of human-written requests.
- **VLM judge reward.** Use `render.screenshot` on before and after, then have a vision model grade against designer-written
  criteria. Add it as a third reward function in `train_grpo.py` for requests the programmatic checks can't verify.
