# Handoff: start here

The entry point for whoever continues this work: a hardware engineer, a software engineer,
or a coding model with no conversation history. Written 2026-11-24 at commit `933972b`,
decisions to D-191. Everything said here was checked against the repository at that commit;
where two documents disagree, trust in this order: code and configs, `docs/DECISIONS.md`
(later entries amend earlier ones), `docs/STATUS.md`, `docs/RESULTS.md`, the rest.

## What the project is

1. One GPU software suite for three sonars: a forward-looking sonar (FLS), a synthetic
   aperture sonar (SAS) and a long-range flank line array on "Popoto" capture boards.
2. A C++20/CUDA core (`libs/`), a Python package (`python/gpusonar/`), tools (`tools/`),
   YAML configuration (`configs/`) and tests (`tests/cpp`, `tests/python`); one record
   format, one backprojection kernel, one simulator for all modes.
3. A live chain for the long-range array (beamform, detect, track, classify, elevation and
   depth) in two engines, C++ and Python, held equal by tests, with a path through
   Holoscan and Holoscan Sensor Bridge.
4. A design for a 12 km array (32 hydrophones a row, 4 to 6 kHz, 197 dB, one to three
   rows) with its link budget, builds, parts and deployment files generated from specs.
5. **All of it is simulation and analysis. No sonar described here is built; nothing has
   run on a vehicle, the boards, a DGX Spark, a Jetson, Sensor Bridge hardware or real
   data.** Hardware facts are design values or placeholders, flagged `unconfirmed`.

## Start here

**A hardware engineer**

1. [HARDWARE.md](HARDWARE.md): the arrays, boards, projector, navigation unit, what is
   assumed and what is unconfirmed.
2. [`docs/STATUS.md`](../STATUS.md), "Open questions for the project owner", questions 1
   to 12: what the design is waiting for.
3. [`docs/REFERENCE_DESIGN.md`](../REFERENCE_DESIGN.md): from a spec sheet to a simulated
   and then a real system; the six builds in `configs/builds/`.
4. [`docs/DESIGN.md`](../DESIGN.md) (generated; link budgets, data rates, the 12 km design
   point) and the diagrams and parts lists in `docs/design/`.
5. [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md): what the vehicle has to provide, first time on
   the real hardware, the staged build, navigation and the sound-speed ladder.
6. [`docs/tasks/VEHICLE_ADAPTERS.md`](../tasks/VEHICLE_ADAPTERS.md): the 17 questions about
   the boards' output and the navigation interface.
7. [OUTPUTS.md](OUTPUTS.md), sections 1.8, 1.9 and 2.1: what the design does in simulation
   and where it fails.
8. [SIMULATION.md](SIMULATION.md): what the simulator behind those numbers models and what
   it leaves out, before trusting any of them for hardware.

**A software engineer**

1. [`docs/HANDOFF.md`](../HANDOFF.md): the rules in force, how to build and test here, GPU
   jobs, how the work has been done, the false failures.
2. [SOFTWARE.md](SOFTWARE.md): the architecture and where each thing lives.
3. [TESTS.md](TESTS.md): what is tested, how, and what is not. Then
   [EXAMPLES.md](EXAMPLES.md), twelve worked examples to run on the CPU, with what failed
   first, and [SIMULATION.md](SIMULATION.md) for the simulator's models, scenarios and limits.
4. [`docs/STATUS.md`](../STATUS.md), "Open and known wrong", then
   [`docs/DECISIONS.md`](../DECISIONS.md) from D-156 on (the live chain as it is now).
5. [OUTPUTS.md](OUTPUTS.md): what the system produces today and how to regenerate it.
6. [`docs/CONFIGURATION.md`](../CONFIGURATION.md) for keys;
   [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md), "The report line", for the output contract;
   [`docs/RECORD_FORMAT.md`](../RECORD_FORMAT.md) for the input contract (frozen).
7. [`docs/RESULTS.md`](../RESULTS.md) for a number,
   [`docs/PHYSICS_NOTES.md`](../PHYSICS_NOTES.md) for a derivation,
   [`docs/REFERENCES.md`](../REFERENCES.md) for a source and how far it is confirmed.

**A coding model**

1. `CLAUDE.md` and `AGENTS.md` at the repository root (issue tracking with `bd`; the
   conservative git policy).
2. [`docs/HANDOFF.md`](../HANDOFF.md), all of it. Sections 2 (rules), 3 (build and test on
   this machine) and 11 (how to work here) are binding.
3. This page's state table and risks, then `.beads/issues.jsonl` (or `bd ready --json`)
   for the open work.
4. For the task at hand: the task file if there is one
   ([`docs/tasks/VEHICLE_ADAPTERS.md`](../tasks/VEHICLE_ADAPTERS.md),
   [`docs/tasks/REAL_DATA.md`](../tasks/REAL_DATA.md); both say to ask the owner their
   questions before any code), else the `DECISIONS.md` entries that touch the files you
   will change (`grep -n "<file or key>" docs/DECISIONS.md`).
5. [SOFTWARE.md](SOFTWARE.md) and [TESTS.md](TESTS.md) for where things are and which
   tests hold them; [SIMULATION.md](SIMULATION.md) for the simulator; [EXAMPLES.md](EXAMPLES.md)
   for commands that are known to run on the CPU and what their output looks like.
6. Before quoting any number: find it in [`docs/RESULTS.md`](../RESULTS.md) or a JSON under
   `docs/`, and check the decision that last moved it. Do not quote from memory, and do not
   trust the main checkout's built module without rebuilding it (`docs/HANDOFF.md`,
   section 3).

## Current state

| Item | State | Source |
| --- | --- | --- |
| Branch | `claude/gpu-sonar-suite-u289p5`, HEAD `933972b` (2026-11-24) | `git log` |
| Latest decision | D-191: the C++ engine resamples a board's rows together (0.09 to 0.17 s a ping at 64 and 96 channels on an L4, from 1.05 to 1.83 s; outputs unchanged). D-190: the Python engine is real time at 96 channels through Sensor Bridge on an L4 (frame path and board resampling faster, outputs unchanged). D-189: a host check and runbook for the Sensor Bridge measurement that needs a real host (still not measured). D-188: the receiver names its damaged frames, and one damaged at the start of a Popoto-rate loopback is start-up loss. D-187: the Popoto-rate loss is at the start of the run; the receiver queue defaults to 4 s of stream. D-186: the Sensor Bridge loopback at Popoto rates loses frames; D-184 per-board streams merged; D-185 a board clock's rate. D-182 is this documentation set | `docs/DECISIONS.md` |
| Last full GPU suite | 2180 passed, 101 skipped, none failed, on an NVIDIA L4 (Hugging Face job 6b07440eb6a90bbe997e6027) | D-191 |
| Last whole CPU suite in one run | the tree of D-157, `605a4e1` (2026-11-22; 157 of 157 ctest entries); since then (D-158 to D-181) affected files per package only | `docs/STATUS.md`, the result paragraphs just before "Open questions for the project owner" |
| Gallery (`docs/figures/`, `gallery.json`) | regenerated at D-179 on an L4 (job 6b054768c97b8ed4fcd78015, tree `4a80876`); D-180 and D-181 move no default | D-179 |
| Milestones | 1 to 10 of 13 done; 11 (Holoscan), 12 (propagation), 13 (performance model, Jetson) open | `README.md`, `bd` |
| Open `bd` issues | 49 not closed of 126: **P1 8** (6 in progress, 2 open), **P2 0**, **P3 25**, **P4 16** | `.beads/issues.jsonl` at `933972b` |
| Run on hardware or real data | nothing | `docs/STATUS.md`, first paragraph |

The open issues by theme, with ids, are in [`docs/HANDOFF.md`](../HANDOFF.md), section 8.

## Top risks and unknowns

1. **Nothing has met hardware, water or recorded data.** Every result is simulated.
2. **The hardware facts are unconfirmed**: converter (102.4 kHz, 24 bit, simultaneous),
   board synchronisation to about 1 microsecond, the projector and its 197 dB, boards used
   at 4 to 6 kHz against a 10 to 100 kHz specification (STATUS questions 2 to 4).
3. **One row is 1.1 dB short at 12 km** (detection probability 0.22 a ping), on a +25 dB
   target strength that is a placeholder.
4. **The water decides more than the margin**: the same ship is not detected at 4 km in
   summer over mud, nor tracked at its true place in deep water over silt.
5. **Depth at long range is an indication only**: at 12 km the ship reads 8 to 35 m for a
   true 4 m; two rows at 3 km in 100 m read the surface or the bed.
6. **Track association among many multipath arrivals in one gate** is fragile at 12 km
   (seed 60: 65 m off; issue `sonar-tlc`). The gallery shows one good seed.
7. **False movers from a rough surface** (78 over ten seeds in one FLS scene) and **small
   targets missed** (kayak, small vehicle) are known limits with no fix.
8. **The Sensor Bridge path has only run against the board emulator** with a UDP patch;
   its real-time margin is one socket buffer, about 20 ms of stream.
9. **Navigation**: the lever arm is zero until surveyed, the array's velocity is not
   corrected for it, and a source without timestamps yields speed 0 for every track.
10. **Some generated files and documents are stale** (map below); the main checkout's
    built Python module was older than its sources on 2026-11-24.

## Waiting on the owner

| List | Where | Count | Blocks |
| --- | --- | --- | --- |
| Open questions for the project owner | [`docs/STATUS.md`](../STATUS.md), section of that name | 29 numbered; 1 to 12 are the Popoto array (rows, band, converter, projector, static frame, navigation unit, target classes, waters, dive, recording, receive windows, coverage site); 5 of the 29 are marked applied or answered | the hardware design and its defaults |
| Vehicle adapters | [`docs/tasks/VEHICLE_ADAPTERS.md`](../tasks/VEHICLE_ADAPTERS.md), section 3 | 17 | issue `sonar-o7m` (P1): board samples and navigation into the live chain |
| Real data | [`docs/tasks/REAL_DATA.md`](../tasks/REAL_DATA.md), section 3 | 11 | issue `sonar-sul` (P1): simulator calibration and the live chain on real data |

Standing rule: the owner approves each work package, and any change that moves a default
result, before it is merged (`docs/HANDOFF.md`, section 2).

## Document map

"Last commit" is `git log -1` of the file at `933972b`. "Past D-160" says whether the file
mentions any decision from D-160 on. A file can be old and still right; the last column
says what is known to be stale. The stale statements this map first listed in `STATUS.md`,
`RESULTS.md`, `DEPLOYMENT.md`, `SCENARIOS.md`, `REFERENCES.md`, `REFERENCE_DESIGN.md`,
`ARCHITECTURE.md`, `API_AUDIT.md`, the diagrams and the root `README.md` were corrected on
2026-11-24, after the map was made; the last column says so where it applies, and the "Last
commit" dates of those files are a day or less older than their present state.

| File | What it is | Last commit | Past D-160? | How current |
| --- | --- | --- | --- | --- |
| `handoff/README.md` | this page | 2026-11-24 | yes | current |
| `handoff/HARDWARE.md` | the hardware side: what is unknown or at risk, the system block diagram, a register of every assumption with its status (confirmed, via secondary, ours, unconfirmed), the six Popoto builds with their numbers (`hardware_builds.json`, `tools/handoff_hardware.py`), the link budget and the 12 km design point, the interfaces the hardware must meet (ping record, UDP block, WAV sidecar, timing, navigation datagram), a bring-up plan, open hardware decisions | 2026-11-24 | written with this set | current (D-181). Nothing in it is built or measured on hardware, as its first paragraph says |
| `handoff/SOFTWARE.md` | the software side: what is fragile or wrong, the layers and data flow, the repository map, the two live-chain engines, the live chain stage by stage with every default, its key and its decision, SAS, FLS and the simulator, configuration, tools and programs, how to add things without breaking the rules | 2026-11-24 | written with this set | current (D-181). Its section 10 lists stale statements found in other documents; those were corrected on 2026-11-24 after it was written (except root `CLAUDE.md`, the owner's file), so read that section as history |
| `handoff/TESTS.md` | how to run the tests (CPU, GPU job, oracles), a map of what each test file protects, the tests that pin decisions D-155 to D-191, known false failures and how to recognise them, what is not tested or only on simulated input | 2026-11-24 | written with this set | current (D-181). No per-file run time is recorded; the 93 skips of the last GPU job are derived, not read from its log |
| `handoff/SIMULATION.md` | the simulator: what it does not model, its pipeline and a table of its models with their sources and validation, all 76 scenarios by family, calibration to real data, determinism | written with this set | written with this set | written in parallel and merged after this row was written; described here from its author's summary, not read |
| `handoff/EXAMPLES.md` (with `handoff/examples/`: scripts, pictures, saved outputs) | twelve worked examples run on the CPU with their commands, run times, real outputs and pictures, led by what failed (`--emulate` on multi-board scenarios) | written with this set | written with this set | the same |
| `handoff/OUTPUTS.md` | pictures, headline numbers, a report line, how to regenerate | 2026-11-24 | yes | current |
| `HANDOFF.md` | rules, build and test, GPU jobs, open work, how to work here | 2026-11-24 | yes | current (refreshed to D-181) |
| `STATUS.md` | what works, how it was checked, what is wrong, owner questions | 2026-11-24 | yes (to D-181) | current. Corrected 2026-11-24: "State of verification of this tree" names the 2078-passed job and the last whole CPU run, "Where the code has run", "What is next" items 1 and 7, open item 14 and the list of stale generated files are brought to D-181 |
| `DECISIONS.md` | every decision D-1 to D-191 with what got worse | 2026-11-24 | yes | current; the authority after code and configs |
| `RESULTS.md` | every measured number and its source | 2026-11-24 | yes (to D-178) | mostly current. Corrected 2026-11-24: "Sensor Bridge loopback" carries the state since D-153, D-162, D-163 and D-173 beside its 2026-11-17 record; the `minefield_sas` and live-chain rows of "Every stage with a CUDA backend" are marked as old with the current figures beside them; the `sst_validation` sentence; "Not yet measured". Still: D-160, D-161, D-176 and D-181 are measured in their `DECISIONS.md` entries only; the `minefield_sas` row is not re-timed (issue `sonar-egj`) |
| `SCENARIOS.md` | every scene with figures and per-scene results | 2026-11-24 | yes (to D-178) | current with the D-179 gallery. CPU times in its table are from a much older tree |
| `CONFIGURATION.md` | every config key | 2026-11-24 | yes (to D-181) | current |
| `DEPLOYMENT.md` | running the live chain: what the vehicle provides, the report line, staged build, Sensor Bridge | 2026-11-24 | yes (to D-175) | current to D-181. Corrected 2026-11-24: the leading edge is the default since D-160; `chain.range_doppler` (D-181) is in the report-line table and in "Keys a new sonar must review"; the direct-sound figures, the GPU timings and the full-rate GPU ping are the files' |
| `DESIGN.md` | generated design report: requirements against configs, link budgets | 2026-11-20 | no | held to the configs by `tests/python/test_design.py`; regenerated 2026-11-24 for the full-rate GPU ping (measured; it had said unmeasured). **Stale:** its depth sections read `design/popoto_depth_planner.json` and its 12 km sections `design/popoto_12km.json`, both last regenerated 2026-11-19 and 20 (`STATUS.md`, list of stale generated files) |
| `REFERENCE_DESIGN.md` | from a spec sheet to a build: the build-spec fields and steps | 2026-11-18 | no | the method holds; since 2026-11-24 it points to the lever-arm keys of D-167 (not a spec field: generated deploy files do not carry them, issue `sonar-vjq`) |
| `ARCHITECTURE.md` | data flow, backends, where each stage runs | 2026-11-21 | no (to D-147) | structure holds; predates the Sensor Bridge receive work (D-161 to D-163, D-173) and the option-key refusals (D-166, D-177), and says so since 2026-11-24, when its source table, its `libs/motion` row and its diagrams (`backends.png`, `dataflow.png`, `live_chain.png`) were corrected. Prefer `handoff/SOFTWARE.md` for the live chain |
| `PHYSICS_NOTES.md` | derivations, Secs. 1 to 29 | 2026-11-20 | no | physics of the simulator and chain to D-144. The derivations of D-159 (outlier test), D-167 (lever arm), D-176 (parabola bearing) and D-181 (range-Doppler) are in their tests' docstrings and `DECISIONS.md`, not here |
| `PAPER_NOTES.md` | reading of the ASASIN paper | 2026-11-23 | yes (D-164) | current |
| `REFERENCES.md` | register of sources with status (confirmed, via secondary, ours, unconfirmed) | 2026-11-21 | no (to D-146) | registered 2026-11-24 in its section 12.7: the relations of D-159 (outlier test, fit limit), D-167 (navigation unit to array), D-176 (parabola vertex) and D-181 (LFM coupling), all "ours" or "unconfirmed" as their decisions give them |
| `RECORD_FORMAT.md` | the HDF5 record, schema 1.0.0 | 2026-11-18 | no | current: the schema is frozen and no decision since has changed it |
| `OPEN_SOURCE.md` | every open-source dependency and decision | 2026-11-21 | no | current as far as checked: no dependency was added from D-158 to D-181 |
| `HOLOSCAN_NOTES.md` | source-reading notes on Holoscan and Sensor Bridge | 2026-11-22 | no | reading notes; what was then built and measured is in `DECISIONS.md` D-151 to D-173 and `DEPLOYMENT.md` |
| `API_AUDIT.md` | audit of the API at `2c5bb8a` (2026-11-20), findings A1 to G5 | 2026-11-21 | no | historical: the findings are kept as found at `2c5bb8a`; the fixes followed (D-145 to D-147 and later packages) and its header says so since 2026-11-24. Do not read a finding as open without checking the code |
| `HARDCODED_AUDIT.md` | audit of hard-coded assumptions at `3113327` | 2026-11-18 | no | historical; not re-checked here |
| `ORG_ARL_SURVEY.md` | survey of the org-arl repositories | 2026-11-20 | no | historical reading survey, dated 2026-11-17 |
| `SIM_CALIBRATION.md` | calibrating the simulator to real data | 2026-11-16 | no | method only; never run on real data (issue `sonar-sul`) |
| `JETSON.md` | build steps for Jetson and DGX Spark | 2026-11-16 | no | **untested**, as its first line says |
| `tasks/VEHICLE_ADAPTERS.md` | brief for the vehicle adapters, with 17 owner questions | 2026-11-23 | no | current; its "current state" was checked at `6600df8`, before D-175 to D-181 |
| `tasks/REAL_DATA.md` | brief for real data, with 11 owner questions | 2026-11-23 | no | current; same note |
| `benchmarks/README.md` | which command wrote which benchmark file | 2026-11-20 | no | current for `bench_*`, `session_*`, `conditioning_*`; later files name their command in `RESULTS.md` |
| `benchmarks/*.json`, `*.md` (35 files) | raw benchmark, ablation and ten-seed evaluation output | newest 2026-11-22 | n/a | measurements at the commits in their names; none rerun at HEAD. No file for the ten-seed runs of D-172 to D-181 |
| `figures/` (88 files) | the gallery (57 files), validation tables and figures, study output | newest 2026-11-24 | n/a | gallery current (D-179); the other tables and figures date from 2026-11-18 to 20 |
| `scenarios/` (56 files) | geometry diagram per scenario, `transmission_loss.json` | newest 2026-11-20 | n/a | 55 of the 76 scenario files have a diagram (`SCENARIOS.md` names the 21 without). The diagrams of `wreck_sas` and `minefield_sas` were redrawn on 2026-11-24 with the wider seabed of D-171; the others date from 2026-11-20 or before |
| `design/` (38 files) | `derived.json`, study JSON, build and system diagrams, parts | newest 2026-11-20 | n/a | **stale:** `popoto_depth_planner.json` and `.png` (regenerated for D-134, item 37; not since) and `popoto_12km.json` (predates D-156 to D-181): `STATUS.md`, list of stale generated files. `popoto_parts.md` and `.json` and the `gpu` block of `popoto_stage2.json` were regenerated on 2026-11-24 |
| `architecture/` (6 files) | diagrams for `ARCHITECTURE.md`, `DEPLOYMENT.md`, `SIM_CALIBRATION.md` | 2026-11-19 | n/a | drawn by `tools/architecture_diagrams.py`; its backend table is checked by `tests/python/test_architecture.py`. `backends.png`, `dataflow.png` and `live_chain.png` redrawn 2026-11-24 (ten stages with a device build, not seven; the Sensor Bridge path; the multiscale detector and echo grouping) |
| `examples/reference_design/example_static_2x24.yaml` | the worked example of `REFERENCE_DESIGN.md` | 2026-11-24 | n/a | current (touched by D-180's YAML numbers) |
| `plan/GPU-FLS-Phased-Plan.pdf` | the owner's phased plan | 2026-11-16 | n/a | input document, unchanged |

Outside `docs/`: the repository `README.md` said "The full CPU suite and a GPU job have not
yet been rerun on the present tree"; corrected on 2026-11-24 (a GPU job has, the whole CPU
suite has not; state table above), and it now points here. The root `CLAUDE.md` describes
another package (`trajectory_classifier`), not this suite; it is the owner's instruction
file and is left as it is.
