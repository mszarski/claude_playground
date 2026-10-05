# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A recreation of the Reachy Mini text-to-motion system from
[pham-tuan-binh/reachy-motion-generator](https://github.com/pham-tuan-binh/reachy-motion-generator)
(the blog post *The best expressive harness for robots*). The pipeline:
text → **planner** (LLM) → **recipe** → `rmr.recipe.variants` → **plan** → **generator** (flow matching) →
25 Hz 9-DoF trajectory → `rmr.reach.Reach.project` → reachable **move** JSON.

README.md has the status table; docs/ROADMAP.md has the plan for each remaining step.

## Commands

```bash
pip install -e ".[ik,dev]"
pytest                                   # all tests
pytest tests/test_recipe.py -v           # one file
```

## Conventions

- `rmr/` modules are ports of the reference. Each docstring names the file it follows. Keep the semantics
  identical (units, channel order, limits, randomisation ranges), because the teacher data and published models
  depend on them; `tests/test_recipe.py::test_every_teacher_recipe_is_valid` guards the recipe language.
- Units: trajectories are SI (m, rad); plans and recipes are deg and mm. Plan channel order is `rmr.plan.CH`.
  The right antenna droops with negative angles, the left with positive.
- 25 Hz everywhere (`rmr.motion.FPS`). Library clips are ~100 Hz and must be resampled on their `time` array.
- `data/teacher/val.jsonl` and the held-out emotions are evaluation only: never train on them.
- Hugging Face and GPU-dependent work: put downloads behind the `hf` extra and keep the core importable without torch.

## Issue Tracking

Use the `bd` command for issue tracking instead of markdown TODOs:
```bash
bd create "Task description" -p 1 --json
bd ready --json
bd update <id> --status in_progress --json
bd show <id> --json
```
