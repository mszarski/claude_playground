# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python trajectory classification package (`trajectory_classifier`) for analyzing and classifying vehicle trajectory segments. It supports:
- Converting spherical coordinates (lat/lon/alt) to local Cartesian (ENU)
- Extracting geometric features (speed, heading, curvature, jerk)
- Classifying segments as STRAIGHT, TURN, or WIGGLE
- Computing trajectory similarity (DTW, Fréchet, Hausdorff distances)
- Learning-based trajectory ranking and quality scoring

## Build and Test Commands

```bash
# Install dependencies
pip install -r trajectory_classifier/requirements.txt

# Run all tests
pytest trajectory_classifier/tests/

# Run a single test file
pytest trajectory_classifier/tests/test_classifier.py -v

# Run a specific test
pytest trajectory_classifier/tests/test_classifier.py::TestClassifier::test_classify_straight_trajectory -v
```

## Architecture

### Core Modules

- **coordinates.py**: Geodetic (lat/lon/alt) → ECEF → ENU (East-North-Up) local Cartesian conversion using WGS84 ellipsoid
- **features.py**: Extracts per-point features (speed, acceleration, heading, curvature) and trajectory-level statistics (sinuosity, path efficiency)
- **classifier.py**: Sliding-window classifier that assigns STRAIGHT/TURN/WIGGLE labels based on heading change patterns
- **av_features.py**: AV-style metrics including longitudinal/lateral jerk decomposition, comfort scores, smoothness scores
- **similarity.py**: DTW (with Sakoe-Chiba band optimization), Fréchet distance, Hausdorff distance for trajectory comparison
- **ranking.py**: HeuristicRanker (weighted AV scores), LearnedRanker (sklearn-based pointwise/pairwise learning)

### Data Flow

1. Input: DataFrame with `timestamp`, `latitude`, `longitude`, `altitude` columns
2. `coordinates.to_local_cartesian()` → adds `x`, `y`, `z` (meters, ENU frame)
3. `features.extract_features()` → adds `speed`, `heading`, `curvature`, etc.
4. `classifier.classify_trajectory()` → returns `ClassificationResult` with per-point labels and contiguous `ClassifiedSegment` objects

### Classification Logic

- **STRAIGHT**: `|heading_change| < straight_threshold` (default 5°)
- **TURN**: Sustained `|heading_change| > turn_threshold` (default 15°)
- **WIGGLE**: Multiple sign changes in heading within window (oscillating/unstable)

## Issue Tracking

Use the `bd` command for issue tracking instead of markdown TODOs:
```bash
bd create "Task description" -p 1 --json
bd ready --json
bd update <id> --status in_progress --json
bd show <id> --json
```


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->
