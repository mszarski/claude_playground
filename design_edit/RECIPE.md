# Recipe: training a small model to make verifiable edits to structured documents

This is the general method distilled from `design_edit/` (training Gemma-4-31B to edit UI designs from natural
requests, after Figma's "Figma-fluent editing model"). It applies to any task of the form
*(structured document, natural-language request) → edit*: UI designs, CAD models, spreadsheets, configs, slide decks,
code refactors. The README has the full experiment log; this file is the short version of what to do and why.

## The shape of the pipeline

```
 representation ─► eval (conventions, blind review, checks, held-out split)  ◄─ build this FIRST, and audit it often
       │
       ├─► data: procedural generator ─┐
       │         realistic LLM-written ├─► references by construction (diff) or validated by checks
       │         paraphrased requests ─┘
       │
       ├─► SFT on the model's OWN correct reasoning (rejection sampling, thinking kept on)
       │
       └─► RL (GRPO) with an effect-based, verifiable reward on prompts the model solves only sometimes
```

## 1. Representation: make the output small and checkable
- **Edit as a patch, not a rewrite.** A few lines of `set` / `insert` / `delete` / `move` instead of the whole
  document. It is cheap to decode, can't silently damage unrelated parts, and can be applied and checked.
- **One-line-per-node text format** for the document, with stable ids. The model reads it, the patch addresses it.
- **Labels by construction.** For generated data, write the after-state and derive the patch with `diff`, so every
  target applies cleanly. Test `apply(before, diff(before, after)) == after` for every edit type.
- **Put the conventions in context** (here, a design system). Without them, many requests have no defined answer and
  the model is graded on guessing the generator's taste.

## 2. The eval decides everything; build it first and distrust it
Most of the "progress" and "plateaus" in this project were measurement artifacts until the eval was fixed. In our
case the ceiling moved from 0.84 to 0.96 without a single training change.

**How to build a trustworthy eval (what eval v2.2 and v3 do):**
1. **Write grading conventions down** before writing items (`eval_v22/CONVENTIONS.md`): merge alternative solutions
   into one criterion; never require arbitrary choices (ids, indices, invented exact values for vague requests);
   say how ambiguous patterns are read ("next to", exclusive selection, system vs. local style); define when an
   item must be dropped (false premise, real ambiguity, bad reference).
2. **Criteria plus exact checks.** Natural-language criteria for an LLM judge, compiled into machine checks
   (keep-in / keep-out, with an `any` combinator for alternatives). Checks are exact and cheap; the judge covers
   genuinely subjective parts.
3. **A validator gate on every item:** the reference passes; at least one *different* valid answer passes; doing
   nothing fails; plausible wrong answers ("traps") fail. Items that can't meet this aren't gradable.
4. **Blind authoring and review by strong agents.** Item writers and reviewers never see model outputs, so the eval
   can't be tailored to a model. A second agent reviews each batch adversarially (try a fair alternative answer:
   if it fails, the checks are wrong). A consistency pass compares similar items across batches.
5. **Held-out split by source** (by screen), so checkpoint selection on dev doesn't leak into the reported test score.
6. **Calibrate the judge.** Compare judges on the same answers (and against the exact checks); avoid a judge from
   the same family as a contestant or as the item writer.
7. **Compare models item by item** (paired wins/losses with a confidence interval). On a few hundred items, ±3 points
   is noise.
8. **Re-derive everything after a change.** Our v2.1 rewrote criteria but kept checks compiled from the old ones,
   and the judge was shown stale results for half the items.
9. **Watch for saturation.** When the best models pass ~95% and only a handful of items fail for everyone, build a
   harder eval (v3: bigger screens, multi-part and propagating edits, patterns to follow, system audits).

**Symptoms that the eval, not the model, is the problem:** many items failed by *every* strong model; voting and
longer reasoning don't help; RL reward rises but eval doesn't; the judge disagrees with exact checks; the same answer
form passes on one item and fails on a near-identical one.

## 3. Data
- **A procedural generator** for coverage of edit types with exact references (cheap, unlimited, but narrow and
  template-flavored; models overfit to its specific values, so randomize targets).
- **Realistic LLM-written documents and requests** with a design system in context, validated by compiled checks.
  This is what moved realistic-eval scores most.
- **Paraphrased requests** (keep quoted text and numbers verbatim; check equivalence with a judge).
- **Repairs as reverse edits:** break a clean document in a known way and ask for the fix; the answer is exact.
- **Never train on eval seeds** (separate seed ranges; check for leaked rows — we lost a result to that once).

## 4. Model and SFT
- **Screen several base models zero-shot** on the real eval before committing; reasoning models start much higher.
- **Keep the base model's reasoning.** SFT without thinking erased Gemma's thinking mode. Instead, sample the base
  model's own reasoning traces, keep the correct ones (rejection sampling, graded by the checks so alternatives
  count), and train on them with the exact inference-time prompt rendering.
- **LoRA adapters, stacked.** Each stage trains a new adapter on top of the merged previous ones; keep every stage's
  adapter so any combination can be evaluated or resumed.
- SFT on own traces mostly makes the model *reliable* at what it can already do; by itself it barely beat the base
  model with thinking here (+1 ± 3 points). Its job is to give RL a good starting point.

## 5. RL
- **Reward the effect, not the text:** apply the patch, check the resulting document (checks, lint, a judge for
  subjective requests). Equivalent answers get equal reward. Text-match rewards punish correct answers; loose judges
  reward over-editing (penalize edit size, check placement).
- **Pick prompts the current model solves sometimes** (1–7 of 8). Groups with zero reward variance teach nothing.
- **RL was the step that mattered:** +8 ± 3 points over its SFT start, and the margin over the strong reasoning
  baseline. Further RL rounds, re-picked prompts and longer reasoning budgets were flat: once prompts are solved or
  never solved, more optimization on the same data doesn't help.
- **Audit the reward like the eval.** A lint that ignored each document's own design system penalized correct
  answers in the reward and in the trace filter.

## 6. What didn't help (so you can skip it)
- More RL steps on the same prompts after the first plateau.
- Self-consistency voting (errors were systematic, not sampling noise).
- Longer token budgets when few rollouts were truncated.
- SFT on reference-hinted ("rationalized") traces for never-solved prompts: +0.2 ± 2.5; it teaches arbitrary
  reference choices when items are noisy.
- A small student (4B) for goal-level judgment: it can't absorb the teacher's reasoning by SFT.

## 7. Operating it (cloud jobs)
- Verify every job gets past model loading before walking away; most lost money here was jobs that died at startup
  (missing dependencies) or at upload (Hub rejecting PEFT model-card metadata — skip the README on upload).
- Make long-running orchestration idempotent and resumable; schedule check-ins instead of long-lived watchers.
- Use agent workflows for judgment-heavy, parallel work over many items (eval repair, eval authoring, failure
  taxonomies) and scripts for bulk generation with a fixed prompt.
- Before training on another model's outputs, check that model's terms.

## Checklist
1. Patch-style output, stable ids, conventions in context.
2. Conventions doc → blind agent-written eval with a validator gate, traps and a held-out split.
3. Screen base models; pick a judge by calibration.
4. Data: generator + realistic + paraphrases; references by construction or by checks.
5. SFT on the model's own correct reasoning.
6. RL on sometimes-solved prompts with an effect-based reward.
7. After every plateau, audit the eval and the reward before changing the model.
8. When the eval saturates, make it harder.
