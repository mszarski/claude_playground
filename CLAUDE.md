# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repository holds two independent projects.  **hydropt** (`hydropt/`,
`examples/`, `tests/`, `scripts/`, the root `README.md`) is the active one: a
differentiable 3-D underwater acoustic ray tracer and sonar simulator in
PyTorch.  `trajectory_classifier/` is a separate, older package with its own
tests, described at the end of this file.

# hydropt

## What it is, in one paragraph

A forward model from an ocean (sound-speed field, rough sea surface, rough
seabed, absorption) and a sonar (transmit fan, receive array, pulse) to what
the sonar records: traced rays, reverberation patches, target echoes off
point, analytic and triangle-mesh scatterers, coherently beamformed into a
bearing-range image and displayed in metres -- every stage differentiable, so
a loss on the picture reaches the scene's and the targets' parameters.
`README.md` is the long-form account: the physics, the validation, what each
piece is for and what it cost to get right.  Read the section for whatever
you touch before touching it; most non-obvious choices in the code have a
paragraph there explaining the bug that forced them.

## Commands

```bash
pip install -e '.[dev]' && pip install scipy   # torch >= 2.2; scipy for examples 19 and 25

python -m pytest tests -q                 # 574 tests, ~22 min on 4 cores
python -m pytest tests/test_beamform.py -q -x
python -m pytest tests -q -k "incoherent"

cd examples && python 21_long_range_300m.py        # figures -> examples/figures/ (gitignored)
cd examples && HYDROPT_SCENARIO=wake python 23_harbour_scenarios.py
cd examples && HYDROPT_SONAR=330 python 21_long_range_300m.py   # the 1.4 x 2.8 deg head
cd examples && HYDROPT_EXAMPLE_DTYPE=float64 python 22_inverse_fit_animation.py
cd examples && python ../scripts/timing_picture.py # stage timings of 21's picture
```

Switches every 21-derived example honours (21-29): `HYDROPT_SONAR` (`120`,
the default: 120 kHz, 3.0 x 4.8 deg beams, 300 m; or `330`: 330 kHz, 1.4 x
2.8 deg beams, 150 m, figures tagged `_330k`), `HYDROPT_FAR` (m),
`HYDROPT_NEAR`, `HYDROPT_BOAT` (range), `HYDROPT_HEADING` (deg, 40),
`HYDROPT_EXAMPLE_DTYPE` (`float32`/`float64`), `HYDROPT_SCENARIO` (one
scenario of 23-29), `HYDROPT_FRAMES` (28's and 29's pings).  22-29 lay their scenes out for the 120 kHz head's 300 m
and scale every absolute position by `FAR / 300` (`S` in each), so a new
scenario's positions go in as 300 m values times `S`.  Each example prints
`[PASS]`/`[FAIL]` lines for its acceptance criteria and exits non-zero on a
failure.

## Layout

See the `Layout` section at the end of `README.md` for the module map.  The
path a picture takes: `scene.py` (the container) -> `tracer.py` (RK4 fan) ->
`reverb.py` (patches from bounces) and `active.py` (target echoes: outbound
from the trace, return leg by `eigenray.py`) -> `beamform.py` (FFT or direct
delay-and-sum) -> `noise.py` (calibrate, receiver noise) -> the example's
`display` (median gain, in `21`) -> `examples/15`'s `to_cartesian`.  Targets
are `targets.py` (points, analytic patterns, `fish_school`) and `mesh.py`
(triangle meshes by physical optics, `load_obj`, generators, occlusion).
`sequence.py` wraps that path for a moving target (`PictureRenderer`,
`Trajectory`), `emission.py` is what a vessel radiates (a spoke, with
`propeller_directivity`: shielded forward by the hull, notched astern), and
`labels.py` reads a box, a mask and a class per target off the fields
(`picture(..., labels=True)`; the class is the target's `label` attribute;
`examples/LABELS.md` is the how-to and the record format).

## Conventions

* **Frame and units.**  Metres, seconds, kHz, dB.  `x` forward, `y` to port,
  `z` DOWN (depth).  Bearings positive to port.  A mesh's body frame is the
  same, its origin wherever the mesh's is.
* **Examples are the specification.**  Each is a numbered script with a
  docstring that says what it shows, why, and its acceptance criteria; the
  criteria are checked with `_common.check` and reported, and the figures
  are saved with `_common.save`.  Numbers quoted in a docstring or in the
  README come from a run and say so.
* **21-29 share one scene.**  22-29 import `21_long_range_300m.py` for the
  sonar, environment, boat and display (`_ex21()`), and 23-27 follow one
  pattern: the bare picture once, then each scenario as its own ping,
  measured against the bare picture, with a `bare | scenario | difference`
  figure under 21's own window (`vmin=THRESHOLD_DB`, `vmax` the bare
  picture's peak) and a gradient-liveness check on the scenario's own
  parameters.  Copy 23's skeleton for a new scenario example.  A moving
  thing is a sequence (`28`): `PictureRenderer` from 21's settings forms
  the background once, `Trajectory` gives the poses, `sequence(builder,
  trajectory, times, emitters=)` yields a picture per ping; hold the
  display gain from the first frame.  A moving SONAR is `29`:
  `ownship_sequence(world_targets, trajectory, times, scene_at=)`, the
  world's height fields built once at world scale and `reframe_height_field`
  onto 21's grids per pose, so the sea is re-traced and scrolls.
* **Float32 by default for pictures and fits** (`setup(double=False)`);
  float64 only where a wavelength-scale gradient is being *checked* against a
  finite difference (22 does this under `HYDROPT_EXAMPLE_DTYPE=float64` and
  skips it otherwise).  Measured: the coherent picture's loss in float32
  scatters by 2e-5 between points 0.05 mm apart, more than it changes over
  0.4 mm; the incoherent picture's is linear at that scale in both.
* **Every example's docstring has a "Construction and assumptions" block**:
  the sonar, the environment, the targets, the picture pipeline, the
  assumptions the physics makes, and what to vary -- so a variant is a
  matter of changing the named constants.  21's is the base the later
  ones cite; a new example gets its own.
* **Docstrings carry the reasoning.**  Every module, class and public
  function has one, and the long ones explain a measured failure.  Keep
  that: a change that reverses a documented decision must say what was
  measured.  Comments in the code are for the line they sit on.
* **Tests pin what was learned.**  A bug fixed gets a test that fails
  without the fix (`tests/test_beamform.py` has the pattern: the
  wavelength-scale gradient, the incoherent kernel, complex beams that add).
* **Beads (`bd`) for task tracking**, never markdown TODOs; `bd remember`
  for durable insight.  The stored memories (`bd prime` prints them) are
  the short list of things that were expensive to learn -- read them.

## Rules that were expensive to learn

* Never take `sqrt` of a beamformed image without a floor: an exactly-zero
  cell puts NaN into every parameter's gradient (`noise.py`,
  `active.compose_arrivals`).
* The eigenray solver never caps bracketed paths by miss distance (it
  dropped the direct path), and the spreading Jacobian is taken in the
  path's own launch and arrival frames, not the chord's tangent plane
  (bounce paths read +3 dB otherwise).
* A fit through the median-TVG display freezes the gain from the
  measurement (`display_gain`, `display(gain=)`): a median re-derived per
  trial image gives an analytic gradient six times the finite difference.
* Receiver noise goes on the complex field (`add_receiver_noise` with
  `complex_output=True` beams), not the power: the power route has a kink at
  every null of the field.
* A pose is fitted on the model's *incoherent* picture (`beamform(...,
  coherent=False)`) against the coherent measurement.  The coherent
  picture's gradient is exactly right and useless for descent: speckle in
  the pose, a cusp on a 2 m plateau.  Finish the blur schedule at ~0.5 m;
  coarser blurs have a biased minimum from the direct/surface-ghost fringe.
* `mesh_target(..., facet_chunk=4096, checkpoint=False)` for a hull against
  a few hundred direction pairs: 2.5x faster than the library's default
  chunking, 6 MB of working set.  The default stays conservative for big
  meshes against many directions.
* Precision is not where the time goes: float32 vs float64 is ~12 % of a
  fit step.  The trace (10 s at 300 m) and the physical-optics integral
  (1.6 s) are; see the README's "Where the time goes now".
* The method of images (`eigenray_arrivals_batched(method="auto")`) replaces
  the traced return leg whenever the profile is isovelocity: 17 s -> 2 s for
  a target's arrivals.  Keep the traced path for refracting profiles.
* Rx patterns see arrival directions: `rx_pattern=lambda d: pattern(-d)`
  when a transmit pattern is reused for receive.  And they are called once
  per highlight: anything expensive inside (21's `beam_3db_deg`, 70 ms) is
  multiplied by the point count -- cache it (`beam_tilts_deg` is).
* Physical optics per patch is a plane wave per patch: a patch larger than
  the Fresnel zone (`sqrt(lambda R)`, 1-2 m here) has its coherent part
  wrong.  A 4 m cylinder at 30 m needs `n_patches=1` (bead `cva`).
* An occluder is binary (`segment_mesh_transmission`); translucent things
  (kelp) are attenuated instead (`examples/26`).
* A label's mask is where the target's own power beats everything else by
  3 dB AND is within 35 dB of its own peak AND lies within its geometry
  plus two beams: without the second gate a +48 dB echo's Hamming
  sidelobes (-43 dB) label the whole swath at its range as the target.

## Operating notes

* Four cores.  Run examples one at a time and never alongside the test
  suite: a second 4-thread process oversubscribes OpenMP and both slow by
  10x or more, not 2x.  Background a long run, poll its log, and do the
  editing while it runs.
* Do not commit unexercised code: run the example (or the scenario) after
  editing it and read its `[PASS]`/`[FAIL]` lines and its figure before
  committing.  Library changes get the full suite.
* `examples/figures/*.png`, `*.gif` and `*.pt` are gitignored; send figures
  to the person, do not commit them.  `.beads/interactions.jsonl` is
  tracked but gitignored: `git add -f` it.
* Timings quoted anywhere come from `scripts/timing_picture.py` or an
  example's own stage timers, run alone.
* 21's speckle-sensitive numbers (the raw contrast, the swath cell count)
  repeat bit for bit between runs made alone, and differ between runs made
  under different load (float32 reduction order).  Compare a change
  against a run made alone in the same session, not against an old log.

## Handoff: where things stand, and what to pick up

Everything is committed on `claude/hydropt-acoustic-tracer-15kecr`; the suite
is green (574); examples 01-29 pass at 120 kHz and 21-29 at 330 kHz; the
figures are regenerated by a run.  The next pieces are beads with their
acceptance criteria written in (`bd show <id>`), in the order to take them:

* `claude_playground-oer` an external position source: `Trajectory` from GPS/AIS records
  and example 30 (reuse `trajectory_classifier/coordinates.py` for ENU);
* `claude_playground-d58` the randomised generator of labelled pictures
  (`scripts/generate_dataset.py`, 28 and 29 are the templates);
* `claude_playground-4qf` the GPU port (the README's "Running on a GPU" is the audit;
  nothing has run on a GPU yet, so expect device-less tensors and CPU
  generators to surface one at a time -- fix each where it is constructed);
* `claude_playground-2dd` the CV detector: trained on the generator, then attacked and
  rescored through the differentiable renderer (depends on the generator).

Read the bd memories first (`bd prime`); each one is a day that need not
be repeated.

## Next steps

The README's "Next steps" is the list; in short: a randomised generator of
labelled pictures on top of `PictureRenderer` (28 and 29 are the
templates, `examples/LABELS.md` the record), the GPU port that makes it
fast, the first real picture (fit the scene before the boat; the levels are
unvalidated), pose/track/identity by the inverse on real frames, a detector
trained through the simulator, and the physics owed in the open beads
(`bd list --status=open`: spherical-wave PO `cva` first).

## Adding an example

1. `bd create` an issue for it first.
2. Copy the nearest pattern: `23_harbour_scenarios.py` for a scenario on
   21's picture, `22_inverse_fit_animation.py` for a fit, `21` itself for a
   new sonar or environment.  Number it next in sequence.
3. Write the docstring first: what it shows, the physics in a paragraph
   each, the acceptance criteria as bullets.  Quantities in it should be
   ones the script prints.
4. Build the scenario from library pieces (`targets.py`, `mesh.py`,
   `wake.py`, `reverb.py` occluders).  A missing piece goes in the library
   with a docstring and a test, not in the example.
5. Metrics against the bare picture, `check(...)` for each criterion, a
   gradient-liveness check on the scenario's parameters, a figure per
   scenario under 21's window.  A sequence example renders through
   `PictureRenderer` with `labels=True`, gives every target a `label`
   (its class), draws the boxes with `draw_labels`, writes the JSON of
   records per frame, and checks the labels against the truth (28, 29).
6. Run it alone; read the numbers and look at the figure -- a passing check
   on a wrong picture is the usual failure.  Fix thresholds from what the
   physics allows, never from what the run gave.
7. Add its row to the README's examples table with the measured result,
   commit the example and the row, close the bead.

## Importing a mesh and placing it

`load_obj` -> re-frame to metres, `x` forward, `z` down -> check the winding
with `facet_geometry` (normals outward; `faces[:, [0, 2, 1]]` flips) ->
`mesh_target(verts, faces, position=(x, y, z), yaw=, pitch=, roll=,
n_patches=, diffuse_db=, learnable=True)`.  Position and orientation are
`nn.Parameter`s.  For an occluder, place the vertices in the world with the
`place` helper of `examples/23`-`25` and pass `(world_verts, faces)` to
`reverberation_arrivals(occluders=...)`.  The README's "Importing a mesh, and
placing it" has the worked snippet and the four things that go wrong.

## Performance and GPU

Float32, 4 cores, 21's 300 m picture: trace 10.4 s, reverberation 0.3 s,
boat echo 1.6 s, FFT beamform of 91k arrivals 0.85 s, display < 0.01 s; a
second picture of the same scene under 3 s; a fit step 2.6 s (coherent) or
4.2 s (incoherent), 22's 48 steps in 145 s.  hydropt has never run on a GPU;
what it needs (a device switch in `_common.setup`, per-device generators, 32
device-less tensor constructions that `torch.set_default_device` already
covers) and what to expect (the beamformer and physical optics 10-50x, the
tracer 3-5x until its step loop is fused, float32 only) are in the README's
"Running on a GPU".

---

# trajectory_classifier

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
