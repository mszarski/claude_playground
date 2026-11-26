# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

**The GPU sonar suite is the main work here.** One GPU software suite for a forward-looking sonar (FLS), a
synthetic aperture sonar (SAS) and a long-range flank array ("Popoto"): a C++20/CUDA core (`libs/`), the
Python package `gpusonar` (`python/gpusonar/`), applications (`apps/`), tools (`tools/`), YAML configuration
(`configs/`), tests (`tests/cpp` GoogleTest, `tests/python` pytest) and documents (`docs/`). One record
format, one backprojection kernel, one simulator for all modes. The long-range live chain exists in two
engines, Python (`python/gpusonar/live.py`) and C++ (`libs/live`), held equal by tests. All of it is
simulation and analysis: nothing has run on a vehicle, the boards or real data.

## Read first

1. `docs/handoff/README.md`: reading orders, current state, risks, what waits on the owner.
2. `docs/HANDOFF.md`, all of it. Sections 2 (rules), 3 (build and test on this machine) and 11 (how to
   work here) are binding. Then `docs/handoff/SOFTWARE.md`, `TESTS.md`, `SIMULATION.md`, `HARDWARE.md`,
   `EXAMPLES.md`, `OUTPUTS.md` as the task needs.
3. `docs/STATUS.md`: what works, what is wrong, and "Open questions for the project owner".
   `docs/DECISIONS.md`: why. It is append-only: later entries amend earlier ones; add a new entry or a
   pointer, never rewrite an old one. Before changing a file, `grep -n "<file or key>" docs/DECISIONS.md`.
4. The task documents, which hold work that waits on the owner (questions first, then plan):
   `docs/tasks/VEHICLE_ADAPTERS.md`, `docs/tasks/REAL_DATA.md`, `docs/tasks/PROPAGATION_3D.md`,
   `docs/tasks/SST_GAPS_JACKSON.md` (with the reading notes in `docs/references/`; both may exist only on
   the session branch `claude/gpu-sonar-suite-u289p5`).
5. Sources: `docs/REFERENCES.md`, with its status words `confirmed` (read in the source), `via secondary`,
   `ours`, `unconfirmed`. Every change names its source in those words.

Where documents disagree, trust: code and configs, then `docs/DECISIONS.md`, `docs/STATUS.md`,
`docs/RESULTS.md`, the rest. Do not quote a number from memory: find it in `docs/RESULTS.md` or a JSON
under `docs/` and check the decision that last moved it. Current counts and timings live in
`docs/STATUS.md`, `docs/handoff/README.md` and `docs/handoff/TESTS.md`, not here.

## Rules that bind every change (the owner's)

- Tests before or alongside every change; a behaviour change has a test that fails before and passes
  after. Never call work done with a failing test. Never weaken a test without saying so plainly.
- The two live-chain engines stay equal: a behaviour change in `python/gpusonar/live.py`, `detect.py` or
  `classify.py` is made in `libs/live` in the same package, held by `tests/python/test_live_cpp.py`.
- Stop and ask the owner before: a default result moves (bring before -> after numbers on a
  representative case, worst first); changing the record schema (`docs/RECORD_FORMAT.md`, frozen since
  milestone 2); changing any FPGA code; adding a large or copyleft dependency; acting when a paper and a
  cited reference disagree in a way that changes results. New behaviour is opt-in unless the owner says
  otherwise; the defaults the owner chose are listed in `docs/HANDOFF.md` section 2.
- GPL/LGPL code is used only as a separate test oracle or a separately run tool, never linked or copied
  into the core. As used now: Bellhop (GPL-3) is the `eigenray` tier's program and Bellhop3D (GPL-3) the
  opt-in `eigenray3d` tier's (approved by the owner 2026-11-26), both run as separate processes;
  KRAKEN (GPL-3) is a test oracle run as a separate process; PyRAM (BSD-3) is a test oracle imported by
  `tests/python` only. Record every open-source decision in `docs/OPEN_SOURCE.md`.
- Unconfirmed hardware facts live in config, not code.
- Do not invent a number a paper reports.
- Licensed downloaded data sets are never committed or redistributed; only derived tables and figures.
- The distribution-limited report named in `docs/REFERENCES.md` is never fetched or read, from anywhere.
- Privacy: no personal details or project content to any outside service other than the GPU job service
  the owner approved (`docs/HANDOFF.md` sections 2 and 4); web searches use generic terms only; no
  accounts, forms or emails; paywalled or login-gated pages: move on. Never print or paste a token.
- Writing: plain statements; what failed or got worse first; every number names its source.

## Build and test (sonar suite)

The machine is small (4 cores, 15 GB, a few GB of disk, no GPU) and shared: build with `-j2` at most, `-j1`
when other builds run; run one pytest process at a time; never the whole suite at once for a quick check.

```bash
# CPU build in a worktree (options as in CMakeLists.txt); the Python module lands in <worktree>/python/gpusonar/
cmake -S <worktree> -B <build dir outside the worktree> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSONAR_BUILD_TESTS=ON -DSONAR_BUILD_PYTHON=ON -DSONAR_ENABLE_CUDA=OFF -DSONAR_BUILD_HOLOSCAN=OFF \
  -DSONAR_CERES_DIR=/opt/ceres-nosuitesparse -DSONAR_ALLOW_SUITESPARSE=OFF
cmake --build <build dir> -j1
export PYTHONPATH=<worktree>/python OMP_NUM_THREADS=1

# One test file, some tests of a file, one test
python3 -m pytest tests/python/test_live_cpp.py -q -p no:cacheprovider
python3 -m pytest tests/python/test_detector.py -q -k range_estimate
python3 -m pytest "tests/python/<file>.py::<test name>" -q -p no:cacheprovider

# C++ tests
ctest --test-dir <build dir> -E pytest --output-on-failure        # all GoogleTest cases
ctest --test-dir <build dir> -R 'Tracker\.' --output-on-failure    # one GoogleTest suite
```

- `tools/test.sh [extra ctest args]` configures, builds and runs everything (GoogleTest, and pytest as the
  single ctest entry `pytest`); `SONAR_BUILD_DIR`, `CMAKE_BUILD_TYPE`, `SONAR_CMAKE_ARGS` override. It
  builds at full width, so on this machine use the steps above.
- The main checkout's built module can be older than the C++ sources: rebuild before trusting a failure.
- The program tests in `tests/python/test_live_cpp.py` need `<worktree>/build/bin/sonar_live`: make a
  temporary `build` symlink to the build directory and remove it afterwards. The false failures it causes
  (and the others known) are in `docs/handoff/TESTS.md` section 4.
- GPU: the full GPU suite runs through `tools/gpu_job.sh` in the GPU job (`docs/handoff/TESTS.md`; launch
  mechanics in `docs/HANDOFF.md` section 4) and is run before every merge. `bash tools/gpu_job.sh --help`
  lists its switches and job modes (`GPUSONAR_JOB=benchmark|gallery|elevbench|liveablate|benchall|clseval|falsealarm|motion`).
- Oracles: `tools/build_oracles.sh <out-dir>` builds Bellhop, Bellhop3D and KRAKEN (needs `gfortran`) and
  prints the `export` line as its last line.

| Variable | What it enables |
| --- | --- |
| `GPUSONAR_BELLHOP=<path>/bellhop.exe` | the `eigenray` tier and the Bellhop-dependent tests (else they skip) |
| `GPUSONAR_BELLHOP3D=<path>/bellhop3d.exe` | Bellhop3D: the `eigenray3d` tier and its tests (else `bellhop3d.exe` beside `GPUSONAR_BELLHOP`) |
| `GPUSONAR_KRAKEN_DIR=<dir>` | the KRAKEN normal-mode cross-checks |
| `GPUSONAR_REQUIRE_ORACLES=1` | an oracle test that would skip fails instead (`tests/python/oracle_gate.py`) |
| `GPUSONAR_SLOW_ORACLES=1` | the slow `eigenray3d` acceptance against Bellhop3D (not part of the suite) |
| `GPUSONAR_HSB_STUDY=1`, `GPUSONAR_HSB_TIMING=1` | Sensor Bridge measurements in `test_hsb.py` (opt-in studies, not tests) |
| `GPUSONAR_SIM_ENGINE=numpy` or `cpp` | which simulator engine the tests use (unset: C++ when built, else NumPy with one warning); `cpp` on a module without the C++ simulator raises `EngineUnavailable`, which the tests turn into a skip (`sim/engine.py`, `tests/python/conftest.py`) |
| `GPUSONAR_ORACLES=1` (GPU job) | build Bellhop, Bellhop3D, KRAKEN, install PyRAM 1.3.0, set `GPUSONAR_REQUIRE_ORACLES=1` |
| `GPUSONAR_CERES=1` (GPU job) | build Ceres without SuiteSparse, so motion estimation is built and tested |
| `GPUSONAR_HOLOSCAN_CPP=1` (GPU job) | install the Holoscan C++ package and build the C++ operators |
| `GPUSONAR_HSB=1` (GPU job) | build Holoscan Sensor Bridge (implies `GPUSONAR_HOLOSCAN_CPP=1`) for `test_hsb.py` |

The full table, including the data-set variables, is `docs/handoff/TESTS.md` section 1.

## How work is done here (`docs/HANDOFF.md` section 11)

1. One `bd` issue per package; only the orchestrating session runs `bd`. An agent in a worktree does not,
   and lists follow-ups in its report.
2. One git worktree per package under `.claude/worktrees/<name>`, on its own branch.
3. Tests first, in both engines.
4. Before a merge: re-read the diff, rerun the affected test files, run the full GPU suite.
5. Ask before a default moves; new behaviour is opt-in.
6. Write it down: one `docs/DECISIONS.md` entry per package, leading with what got worse; update the
   state lines in `docs/STATUS.md`; file follow-ups.
7. Commit (when asked) with `git -c core.hooksPath=/dev/null commit`, staging explicit paths, never
   `.beads/`. Remove worktrees, build directories, large outputs and tarballs after use: the disk is small.
8. YAML: write `38.0e+3`, not `38.0e3` (PyYAML reads the latter as a string; `tests/python/test_yaml_numbers.py`).

## Issue Tracking

Use the `bd` command for issue tracking instead of markdown TODOs:
```bash
bd create "Task description" -p 1 --json
bd ready --json
bd update <id> --status in_progress --json
bd show <id> --json
```


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:1105d646 -->
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

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.

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
