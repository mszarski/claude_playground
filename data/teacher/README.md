# Teacher data

From [pham-tuan-binh/reachy-motion-generator](https://github.com/pham-tuan-binh/reachy-motion-generator)
(`distill_data/`, `planner/examples/`), Apache-2.0.

- `dataset.jsonl`: 10,872 rows, `id, prompt, idea, recipe, source, family, weight` (`weight` = repeats in training).

  | source | rows | |
  |---|---|---|
  | `claude` | 520 | hand-written prompt → idea + recipe |
  | `claude_events` | 65 | build-up → release events and long multi-phase stories (weight 3) |
  | `seed` | 287 | the recipes of `recipes.json` (no idea) |
  | `astra` | 5,000 | scenarios over ~600 families, precise choreography; **not** used by the served models |
  | `astra_lively` | 5,000 | the same prompts re-authored in a livelier style |

- `val.jsonl`: 39 fixed validation prompts, each with two independent teacher recipes. Never train on these.
- `recipes.json`: 287 prompt → recipe seeds.
- `eval_prompts.txt`: the 12 held-out real emotions plus out-of-distribution probe prompts.
