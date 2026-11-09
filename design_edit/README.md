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

## v4: goal-level training requests

v4 adds four goal-level training edits, where the request names a goal or a reference element and the model
works out the concrete changes: `match_button` ("make A match B"), `demote_button` ("make X a secondary button",
the reverse of the held-out promote), `match_text_style`, `match_spacing`. A new held-out set `test_goal` (240 rows,
never in training) has `disable_button`, `highlight_plan` ("highlight the Free plan the way Pro is
highlighted") and `toggle_on` ("turn on Dark mode", where the on-color must be read off another toggle).
Same recipe, about 64 min, about $3: [`mszarski/design-edit-sft-v4`](https://huggingface.co/mszarski/design-edit-sft-v4).

**Scoring fix: alternative answers.** Qwen3.8-27B first scored 0.50 on `test_goal`, mostly from answers that
were valid but different from the reference (disabled = grey fill + grey label instead of lower opacity;
highlighting also restyles the plan's button). Examples now carry `alts`, other correct answers, and `score()` takes
the best match. `scripts/rescore.py` re-scores saved outputs offline. With this fix Qwen3.8-27B scores 0.94.

| split | v3 | **v4** | Qwen3.8-27B zero-shot |
|---|---|---|---|
| `test` | 0.996 | 0.992 | 0.860 |
| `test_paraphrased` | 0.943 | 0.943 | 0.787 |
| `test_novel` | 0.657 | **0.763** | 0.933 |
| `test_novel_paraphrased` | 0.652 | **0.759** | 0.793 |
| `test_goal` | **0.650** | 0.438 | 0.938 |

| held-out edit | v3 | v4 | v4 pass@8 |
|---|---|---|---|
| `promote_button` | 0.00 | **0.73** (0.81 reworded) | 0.77 |
| `set_weight` ("bold") | 0.61 | **0.82** | 0.98 |
| `darken_text` | 0.80 | 0.86 | 1.00 |
| `duplicate_setting` | 0.64 | 0.50 | 0.93 |
| `spread_out` | 0.00 | 0.00 | 0.00 |
| `disable_button` (goal) | 0.60 | 0.35 | 0.47 |
| `highlight_plan` (goal) | 0.89 | 0.57 | 0.75 |
| `toggle_on` (goal) | 0.50 | 0.61 | 0.74 |

**Goal training transfers to the reverse goal but leaks into nearby ones.** Training "make X secondary" taught the
reverse: "promote X to primary" went from 0.00 to 0.73 (correct fill, `unset stroke`, white label). But the new
recipes now fire on goals they don't fit:
- "show X as inactive" gets the *secondary-button* recipe (white fill, grey border; 24 of 149);
- "highlight the plan like the featured one" copies the brand color into the card's **fill** (the `match_button`
  copy-the-fill habit) instead of its border, creating unreadable text (lint +1.2 per row);
- "turn on X" still uses near-black rather than reading the on-color off another toggle.

So the model learns each goal as a fixed recipe and applies whichever recipe is nearest, rather than reading
the conventions in the design. Qwen3.8-27B, reasoning zero-shot, still leads on `test_goal` (0.94) and `test_novel`
(0.93). The 4B model wins only on the trained edit types and reworded requests. v4's pass@8 on the goal set is 0.58 (0.47–0.75 per type),
so a GRPO run on goal-*style* training prompts now has signal, but these results suggest the bigger gain is many
more goal types, each seen a few times, so no single recipe dominates.

## v5: distilled goals, and grading goals with rubrics

**Distillation** (`scripts/distill_goals.py`). An LLM proposes free-form goal requests for generated designs (14
angles: hierarchy, readability, consistency, decluttering, ...). Requests resembling held-out goals are rejected by
keyword. Qwen3.8-27B (thinking) writes the patch, kept only if it applies, changes something, adds no lint, and
a judge rates it GOOD. 2,004 of 3,500 kept. A second filter dropped 27 whose *patches* mimic held-out goals
(bordered button made filled, button opacity). v5 = v4 train + 1,977 distilled goals, same SFT recipe
(about 71 min): [`mszarski/design-edit-sft-v5`](https://huggingface.co/mszarski/design-edit-sft-v5).

**Rubric grading** (`scripts/judge_goal.py`). Goal requests have many valid answers, and hand-listing them
(`alts`) kept under-counting. Each goal type now has a short designer-written rubric ("reads as unavailable;
label still visible; must not look like an ordinary secondary button; nothing else changes"). gpt-oss-120b, a
different family from the Qwen teacher, grades each applied change. Sanity check: 18/18 reference answers pass,
18/18 wrong answers fail. By hand, its verdicts on v5's disabled outputs are sensible: gray text on same-gray
fill 0/32, classic disabled styling 28/28, secondary-button recipe 0/18.

| model | `test` | `test_novel` | `test_goal` strict | **`test_goal` rubric** |
|---|---|---|---|---|
| Qwen3.8-27B zero-shot | 0.860 | 0.933 | 0.938 | **0.992** |
| v3 SFT (no goal training) | 0.996 | 0.657 | 0.650 | **0.829** |
| v4 SFT (+ 4 goal recipes) | 0.992 | 0.763 | 0.438 | 0.646 |
| v5 SFT (+ 2k distilled goals) | 0.994 | **0.817** | 0.287 | 0.546 |

| rubric pass on `test_goal` | v3 | v4 | v5 |
|---|---|---|---|
| `disable_button` | **0.89** | 0.69 | 0.43 |
| `highlight_plan` | **0.89** | 0.57 | 0.64 |
| `toggle_on` | 0.50 | 0.61 | **0.87** |

On `test_novel`, v5 is the best 4B model yet (0.817: darken 1.00, bold 0.98, duplicate row 0.79; promote fell from
0.73 to 0.50, as expected after removing promote-like distilled rows).

**On held-out goals, every round of goal training made the 4B model worse** (rubric: 0.83 → 0.65 → 0.55),
even though each round helped some neighbor. v5's distilled "de-emphasize X" goals (grey fill + grey text)
turned "make it look disabled" into gray-on-gray (32 of 149 outputs). Each goal pattern learned by SFT becomes a recipe
that fires on whatever request resembles it. A 27B reasoning model reads the design's conventions and passes
99%. The 4B SFT student imitates the teacher's *outputs* and doesn't pick up that reasoning.

This is the strongest evidence in this repo for Figma's position: imitation (SFT, even distilled from a strong
teacher) generalizes poorly for goal-level editing. The obvious next step, now that a trustworthy rubric grader
exists, is **RL with rubric rewards**: GRPO where the reward is `judge_goal`'s verdict on goal-style
training prompts, not reference matching. v5's pass@8 on `test_goal` is 0.60, so there's signal.

## RL with an LLM-judge reward (4B, from v3)

GRPO from v3 (no goal training) on 748 *unlabeled* goal prompts (proposer only, no teacher answers; plan-emphasis
prompts filtered as too close to `highlight_plan`). Reward: a **Qwen3-235B judge** (generic 3-line rubric,
one-word PASS/FAIL) on the applied change, minus 0.25 per new lint violation; −1 for invalid patches; KL β=0.02.
1.5 epochs, about 76 min ($3.70): [`mszarski/design-edit-grpo-rubric`](https://huggingface.co/mszarski/design-edit-grpo-rubric).
Mean training reward rose from about 0.3 to about 0.6. Evaluation uses a *different* judge (gpt-oss-120b) with per-goal designer rubrics.

| split | v3 | v3 + judge-reward GRPO |
|---|---|---|
| `test` | 0.996 | 0.982 |
| `test_paraphrased` | 0.943 | 0.920 |
| `test_novel` | 0.657 | **0.740** |
| `test_goal` (rubric, gpt-oss-120b) | **0.829** | 0.696 |

RL helped the held-out *edit types* (+8 points) but hurt held-out *goals*. The policy moved toward
conspicuous **fill** changes: "highlight the plan like the featured one" now changes the card fill 22 times (v3: 6),
breaking readability, and "make it look disabled" uses fill-only changes far more than opacity. The cheap
training judge with a generic rubric rewards visible change; the stricter evaluation rubric does not. **RL optimizes
whatever the reward actually rewards**, which is why Figma's post emphasizes shaping reward models and expert-written criteria.
A per-goal-type rubric, a stronger or visual judge, and geometry checks in the reward are the next things to fix.

## Matching Figma's recipe: validators, self-healing teacher, Qwen3.8-27B

After reading the post itself, three pieces were added to match it more closely:

1. **Validators** (`geometry.py`, `visual_judge.py`). Designs render in Chromium. Each edit is checked for new
   clipping, sibling overlap, text overflow and off-page content (0 issues across 300 clean designs; fixed sizes no
   longer shrink silently). A vision judge (Llama-4-Maverick, a different family from the Qwen teacher) grades
   before/after screenshots on *fulfilled / preserved / legible / contained*. It agrees with the expected verdict on
   36/40 calibration cases: 19/20 bad edits rejected, 17/20 good ones passed.
2. **Self-healing teacher** (`distill_heal.py`). Qwen3.8-27B (thinking) attempts each proposed edit. The validators
   and visual judge send their specific problems back, and the teacher revises (up to 3 attempts). 1,543 edits
   passed everything, about 30% of them via revision; 1,490 remain after dropping held-out-like patches.
3. **Qwen3.8-27B student**, Figma's reported base. It's a vision-language model with hybrid linear attention:
   loaded text-only, trained and served with `enable_thinking=False` (fast mode), LoRA r=32, 1 epoch on 7,490 rows
   (6,000 template rows correct by construction + 1,490 self-healed teacher edits; the hand-written goal recipes that
   leaked in v4 were left out). About 78 min on one H200 (about $8): [`mszarski/design-edit-sft27b`](https://huggingface.co/mszarski/design-edit-sft27b).

Thinking-off baseline: Qwen3.8-27B with `enable_thinking=False` keeps nearly all of its goal-level judgment
(rubric 0.983 vs 0.992 with thinking) at a fraction of the latency. That's the fair starting point.

| model (same rows) | `test` | `test_novel` | `test_goal` (rubric) |
|---|---|---|---|
| Qwen3.8-27B thinking, zero-shot | 0.860 | 0.933 | 0.992 |
| Qwen3.8-27B no-think, zero-shot | 0.820 | 0.853 | 0.983 |
| **Qwen3.8-27B no-think + SFT** | **0.973** | 0.840 | **0.983** |
| best 4B (v3 / v5) | 0.996 | 0.817 | 0.829 |

Full splits for the 27B SFT: `test` 0.976, `test_paraphrased` 0.933, `test_novel` 0.823, `test_novel_paraphrased`
0.776, coverage 0.985 / 0.995.

**Scale preserved generalization under SFT.** The 27B learned the format and the precise template edits
(0.82 → 0.97) **without** losing its goal-level judgment (0.983, unchanged). Every 4B lost it under the same kind of
training (0.83 → 0.55). Held-out *edit types* didn't improve (0.85 → 0.84; `spread_out` still 0). That gap is what
the next RL stage targets.

### RL on the 27B with validator + visual-judge rewards

GRPO from the 27B SFT checkpoint on 240 prompts (`rl_mix`: half unlabeled goal prompts, half reworded or multi-part
template prompts). Reward: reference score where a reference exists; otherwise −1 invalid / 0 empty / the visual
judge's four dimensions gated on *fulfilled*, minus 0.25 per new lint or rendered-geometry issue. vLLM runs as a
server on a second H200 (the 27B doesn't fit training + sampling on one GPU); Chromium renders inside the job.
lr 5e-6, 60 steps, about 44 min of training: [`mszarski/design-edit-grpo27b`](https://huggingface.co/mszarski/design-edit-grpo27b).
Mean reward rose from about 0.9 to about 1.1; 1,003 patches judged.

| split | 27B + SFT | 27B + SFT + GRPO |
|---|---|---|
| `test` | 0.976 | 0.976 |
| `test_paraphrased` | 0.933 | 0.933 |
| `test_novel` | 0.823 | 0.830 |
| `test_novel_paraphrased` | 0.776 | 0.779 |
| `test_goal` (rubric) | 0.983 | 0.983 |

Unlike the 4B run with a weak text judge, this reward did no harm. It also barely helped: `duplicate_setting`
0.93 → 1.00, but `spread_out` stays at 0 and `promote_button` at about 0.5. The run was deliberately conservative
(low lr, 60 steps, 240 prompts) and the policy was already strong on its training prompts. Moving the held-out
gaps likely needs a larger and more varied RL environment (Figma's stated focus) and more steps.

**Engineering notes from this stage**
- *A reward-function resource leak hung the first 27B GRPO run at step 3 for about 2 h.* A new thread pool per
  reward call started a fresh Chromium per thread per step until memory ran out. Fixed with one persistent pool
  and a per-item timeout. Smoke tests should run past a few steps.
- Multimodal checkpoints (Qwen3.8) need the processor configs copied next to merged weights, or vLLM won't load them.
- In-script HF-generate eval on a 27B with the PyTorch linear-attention fallback is very slow. Use `--n-eval 0`
  and `eval_model.py --vllm`.

### Scaling up the 27B RL run: over-editing

The same reward at larger scale: 2,155 prompts (1,555 goal prompts incl. 807 new ones + 600 reworded/multi-part
template prompts), lr 1e-5, 420 steps, about 4.9 h of training on 2× H200 (about $52), 9,441 patches judged:
[`mszarski/design-edit-grpo27b-big`](https://huggingface.co/mszarski/design-edit-grpo27b-big).

| split | 27B SFT | + GRPO small (60 steps) | + GRPO big (420 steps) |
|---|---|---|---|
| `test` | 0.976 | 0.976 | 0.970 |
| `test_paraphrased` | 0.933 | 0.933 | 0.908 |
| `test_novel` | 0.823 | **0.830** | 0.673 |
| `test_novel_paraphrased` | 0.776 | **0.779** | 0.662 |
| `test_goal` strict / F1 | 0.629 / 0.699 | 0.654 / 0.732 | 0.375 / 0.793 |
| `test_goal` rubric | **0.983** | **0.983** | 0.971 |

More RL made the model **over-edit**: it adds unrequested emphasis to the element it was asked to change. "Bump up
the contrast on the Company link" became `weight=700 color=#111827` (darker *and* bold; the small run gave
`color=#374151`), and "show the button as inactive" also changes the label's weight. F1 rising while exact match falls
is the signature: the requested change is still there, plus extras. `promote_button` improved (0.50 → 0.65), and
`spread_out` is still 0 (never sampled).

Cause: the visual judge rewards edits that look *clearly done*, and its "preserved" check only looks for changes to
*other* elements, so extra changes to the same element cost nothing. 420 steps found and exploited that gap. Fixes for
a next run: a "minimal" rubric dimension (changes only what the request needs), an edit-size or extra-fact penalty,
a KL anchor to the SFT model (β > 0), and more reference-scored prompts, which do penalize extras.

## Where this ends up

| | `test` | `test_paraphrased` | `test_novel` | `test_goal` (rubric) |
|---|---|---|---|---|
| Llama-3.3-70B zero-shot | 0.707 | 0.633 | 0.673 | – |
| Qwen3.8-27B thinking zero-shot | 0.860 | 0.787 | **0.933** | **0.992** |
| Qwen3.8-27B no-think zero-shot | 0.820 | – | 0.853 | 0.983 |
| best 4B SFT (v3) | **0.996** | 0.943 | 0.657 | 0.829 |
| **Qwen3.8-27B no-think + SFT + GRPO (small)** | 0.976 | 0.933 | 0.830 | 0.983 |
| Qwen3.8-27B no-think + SFT + GRPO (big) | 0.970 | 0.908 | 0.673 | 0.971 |

(Zero-shot `test`/`test_novel` rows: first 150 per split; fine-tuned rows: full splits.)

The fine-tuned 27B in fast no-think mode is the best all-round model. It's nearly as precise on known edit types
as the best 4B, keeps the base model's goal-level judgment, and generalizes to new edit types far better than any
4B. The open gaps: new *edit types* that need a convention the model never saw (`spread_out` → `justify=between`),
where only the thinking base model succeeds, and a bigger RL stage to close them.

## Qwen 27B RL with the complete fixed reward: the first clear RL win

The reward now includes everything learned so far: the visual judge with *minimal* and placement checks, the edit-size
penalty, **structural placement checks** (`rewards.placement_violations`: "below X" / "above X" / "underneath it"
verified on the patched tree; it flags 10/10 of the Muse RL's misplaced duplicates and 1/1530 reference answers), and
reference matching on template prompts. GRPO from the 27B SFT checkpoint, KL β=0.02, 50/50 goal/reference prompts
(`rl_fixed`), lr 1e-5, 200 steps, about 2.6 h on 2× H200 (about $28): [`mszarski/design-edit-grpo27b-fixed`](https://huggingface.co/mszarski/design-edit-grpo27b-fixed).

| split | 27B SFT | GRPO small | GRPO big | **GRPO, full fixed reward** |
|---|---|---|---|---|
| `test` | **0.976** | **0.976** | 0.970 | 0.972 |
| `test_paraphrased` | **0.933** | **0.933** | 0.908 | 0.918 |
| `test_novel` | 0.823 | **0.830** | 0.673 | 0.800 |
| `test_goal` strict | 0.629 | 0.654 | 0.375 | **0.954** |
| `test_goal` rubric (current) | 0.950 | 0.925 | 0.512 | **0.996** |
| avg patch lines on goals | 1.43 | 1.51 | 1.71 | 1.51 |

- **Held-out goal edits: 0.950 → 0.996 rubric and 0.629 → 0.954 strict**, the best result in the repo, above
  Qwen3.8-27B with thinking (0.988), in fast no-think mode. "Make it look disabled" went 0.41 → 0.93 exact: the model
  now makes the minimal conventional edit instead of extra emphasis. Patch size didn't grow.
- **Placement held**: `duplicate_setting` stays 1.00 (the Muse run without the check fell to 0.21).
- **Remaining drift**: `promote_button` 0.46 → 0.15 exact (F1 0.80). RL learned to skip `unset stroke`, removing a
  1px light-gray border that is nearly invisible on a newly filled button, while still setting the fill and the white
  label. Again RL dropped the one change the reward couldn't see. `test_paraphrased` and `test_novel` dip 1.5–2.5 points.

**Takeaway:** RL works here once the reward covers what matters: deterministic structural checks for what screenshots
can't show (order, placement), a judge criterion for over-editing, and reference matching to anchor precision.
Each earlier failure pointed to the next missing piece.

## All goal-rubric numbers, re-graded with the *minimal* criterion

Every saved `test_goal` output re-judged with the current rubric (gpt-oss-120b, minimal-aware). Earlier tables in
this README use the old rubric. Re-scoring saved outputs with the current parser reproduced every reported
exact-match number, so only the rubric numbers changed.

| model | old rubric | **current rubric** |
|---|---|---|
| Qwen3.8-27B thinking, zero-shot | 0.992 | 0.988 |
| Qwen3.8-27B no-think, zero-shot | 0.983 | 0.983 |
| Muse-Glimmer-30B zero-shot | 0.979 | 0.979 |
| Muse-Spark-Open zero-shot | 0.988 | 0.975 |
| Gemma-4-31B / Gemma-4-26B-A4B zero-shot | about 0.94 / 0.933 | 0.929 / 0.908 |
| Inkling / Inkling-Small zero-shot | 0.850 / 0.867 | 0.838 / 0.829 |
| Granite-4.2-30B zero-shot | 0.588 | 0.404 |
| 4B SFT v3 / v4 / v5 | 0.829 / 0.646 / 0.546 | 0.833 / 0.658 / 0.537 |
| 4B v3 + judge-reward GRPO | 0.696 | 0.700 |
| Qwen 27B SFT / + GRPO small / + GRPO big | 0.983 / 0.983 / 0.971 | **0.950** / 0.925 / 0.512 |
| Muse-Glimmer SFT / + GRPO (fixed judge) | – | 0.921 / 0.908 |

Models that weren't RL-trained barely move (within the judge's ±0.02 noise; Granite's drop is where it was already
far behind). The *minimal* criterion mostly penalizes the RL-trained models, which is where over-editing happened.
Rankings are unchanged.

## Muse-Glimmer-30B (Meta, Apache-2.0): SFT and RL with the fixed judge

Same 7,490-row SFT recipe as the Qwen 27B, using `--template-kwargs '{"reasoning_strength": "none"}'` (the template's
fast mode). Muse-Glimmer uses a channel chat format (`to=self` reasoning / `to=user` answer); `strip_fences` now keeps
only the final `to=user` message, so evaluation and RL rewards parse it. SFT took about 61 min on one H200 (about $6).
RL is GRPO from the SFT checkpoint with the fixed reward (visual judge incl. *minimal*, edit-size penalty), KL β=0.02,
a 50/50 goal/reference prompt mix (`rl_fixed`), and 200 steps, about 2.2 h on 2× H200 (about $22), with checkpoints every 50 steps
in `mszarski/design-edit-grpo-glimmer-ckpt`.

| split | Qwen3.8-27B SFT | **Muse-Glimmer SFT** | Muse-Glimmer SFT + RL |
|---|---|---|---|
| `test` | 0.976 | **0.982** | 0.976 |
| `test_paraphrased` | 0.933 | **0.933** | 0.916 |
| `test_novel` | **0.823** | 0.773 | 0.767 |
| `test_novel_paraphrased` | **0.776** | 0.728 | 0.710 |
| `test_goal` strict | 0.629 | **0.775** | 0.750 |
| `test_goal` rubric (with *minimal*) | **0.950** | 0.921 | 0.908 |
| avg patch lines on goals | – | 1.14 | 1.46 |

- **A fully non-Chinese base gets within a few points of Qwen after the same SFT.** It's more precise on known edits,
  more often exactly right on goals, and weaker on unseen edit types (`add_border` 0.73, `promote_button` 0.46).
- **RL with the fixed judge still didn't help.** Over-editing is much milder than in the earlier big run (patch size
  1.14 → 1.46 lines vs a goal rubric collapse before), and `promote_button` improved (0.46 → 0.65). But
  `duplicate_setting` fell 1.00 → 0.21 from an **off-by-one**: "duplicate the row *underneath* it" now inserts the
  copy *above* it. Identical content in a different order is invisible to a screenshot judge, so RL drifted on
  exactly what the reward couldn't see.

Across five RL runs the pattern holds: **GRPO in this repo reliably finds and exploits the reward's blind spots**
(fill changes, extra emphasis, insertion order). With rewards this coarse, the SFT checkpoint is the best model. The next
improvement has to come from the reward side: structural checks on *where* things moved (the change-fact
machinery in `rewards.py` can verify order for reference prompts), or rubrics with per-request expected effects.

## Fixing the judges' over-editing blind spot

Both judges only checked for changes to *other* elements, so the big RL run's extra emphasis on the edited element
("darker **and** bold") passed. Added a **minimal** criterion to the visual judge (5th dimension) and to every goal
rubric, plus an edit-size penalty in the RL reward (0.1 per property change beyond 3 on goal prompts).

Calibration on a set built around the failure (Llama-4-Maverick visual judge):

| case | old 4-dim judge passes | new 5-dim judge passes |
|---|---|---|
| correct edits (should pass) | 16 / 20 | 16 / 20 |
| correct edit + extra emphasis on the same element (should fail) | **11 / 19** | **0 / 19** |
| other bad edits (should fail) | 1 / 20 | 0 / 20 |

Re-judging the saved 27B `test_goal` outputs with the minimal-aware rubric (gpt-oss-120b):

| model | old rubric | new rubric |
|---|---|---|
| 27B SFT | 0.983 | **0.950** |
| 27B SFT + GRPO small | 0.983 | 0.925 |
| 27B SFT + GRPO big | 0.971 | **0.512** |

The fixed judge exposes the big run's over-editing (`disable_button` 0.23) and shows the small run had started
to drift as well. **By the corrected measure, the 27B SFT checkpoint is the best model so far.** Earlier rubric
numbers in this README use the old rubric.

## Screening non-Chinese base models (zero-shot, no-think)

Every candidate run through the HF router in fast (thinking-off) mode on the first 150 rows of `test` and
`test_novel` and all 240 rows of `test_goal` (rubric-judged by gpt-oss-120b). "Lenient" rewrites `set #n5` to
`set n5` (a zero-shot ID-format habit that any SFT fixes) before scoring. Judge noise is about ±0.02.

| model | maker, license | trainable here? | valid | `test` (lenient) | `test_novel` (lenient) | `test_goal` rubric |
|---|---|---|---|---|---|---|
| Muse-Spark-Open-816B-A42B | Meta, custom (gated) | no (816B MoE) | 1.00 | 0.853 | 0.840 | **0.988** |
| **Muse-Glimmer-30B** | Meta, Apache-2.0 | **yes** (30B dense) | 0.97 | **0.860** | 0.760 | 0.979 |
| Gemma-4-31B-it | Google, Apache-2.0 | yes (31B dense) | 0.95 | 0.800 | 0.633 | 0.933–0.950 |
| Gemma-4-26B-A4B-it | Google, Apache-2.0 | yes (MoE, 4B active) | 0.85 | 0.600 | 0.513 | 0.933 |
| Inkling | Thinking Machines, Apache-2.0 | no (1.9 TB) | 0.79 | 0.773 | 0.713 | 0.850 |
| Inkling-Small | Thinking Machines, Apache-2.0 | no (532 GB) | 0.80 | 0.753 | 0.713 | 0.867 |
| Granite-4.2-30B | IBM, Apache-2.0 | yes | 0.37 | 0.400 | 0.460 | 0.588 |
| *Qwen3.8-27B (reference)* | *Alibaba, Apache-2.0* | *yes* | *0.99* | *0.820* | ***0.853*** | *0.983* |

- **Muse-Glimmer-30B is the best trainable non-Chinese base.** It roughly matches Qwen3.8-27B zero-shot
  (slightly better on `test`, behind on `test_novel`, equal on goals), it's Apache-2.0, and it's supported by transformers.
- **Muse-Spark-Open** is the strongest model overall but far too large to fine-tune here. It's a good teacher
  for the self-healing pipeline instead of Qwen.
- **Inkling** (named in Figma's post) underperforms here even after the format fix, mostly on goal-level edits.
- **Gemma-4-31B** is solid but weaker on unseen edit types; Granite struggles with the format.

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
- **Many more goal types, few examples each** (v4 shows 4 goal types become fixed recipes that leak into nearby
  goals). An LLM could propose goals and a strong model could write candidate patches, filtered by the reward.
- **GRPO with a rubric-judge reward** on distilled goal prompts (training goals, not test goals), from v3 or v5.
  Rubric pass on held-out goals is the metric to beat (v3 0.83, Qwen3.8-27B 0.99).
- **A larger student** (e.g. Qwen3.8-27B LoRA, as the Figma post suggests for its newer model). The 4B student can't
  absorb the teacher's reasoning through SFT.
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
