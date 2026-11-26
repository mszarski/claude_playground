# Handoff: start here

The entry point for whoever continues this work: a hardware engineer, a software engineer,
or a coding model with no conversation history. Written 2026-11-24 at commit `933972b`
(decisions to D-181), brought to decisions D-203 (the opt-in 3-D propagation tier is D-202)
(`88e9b56`), then to D-225 (the merge `b88c458` and its documentation) and then to D-229 (the merge `61cfb66` of the second batch and its documentation) on 2026-11-26. Everything said here was checked against the repository then;
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
   the boards' output and the navigation interface. Since D-184 to D-200 each board may send its
   own stream (every receiver merges them); the boards' capture starts must agree with true time
   to about 0.8 ms, and a board must stream 0.25 s before the first transmit that matters.
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
   questions before any code; [`docs/tasks/PROPAGATION_3D.md`](../tasks/PROPAGATION_3D.md) for 3-D
   propagation, its packages done and its open questions), else the `DECISIONS.md` entries that touch the files you
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
| Branch | `claude/gpu-sonar-suite-u289p5`; this page brought to the merge `61cfb66` of the second 2026-11-26 batch and its documentation (D-229, 2026-11-26) | `git log` |
| Latest decision | D-229 (the merge of D-226 to D-228 and the owner's answers to `STATUS.md` questions 36 to 42). The second batch, one line each: D-226 Williams-Jackson (1998) and Jackson-Briggs (1992) as opt-ins, Python only (the default small-slope lies 6 to 7 dB above the published bistatic join at 20 to 45° for sands; it stays the default); D-227 the sites' boundary scattering in the depth plan and the coverage pictures, by one rule, with illustrative values (worst: the +5 dB class over the 100 m shelf reaches 2.56 km in the plan, 8.02 to 9.58 before; 1.24 to 1.37 km in the face files' coverage, 6.37 to 8.10 before; the +25 dB classes stay noise-limited at 12 km; the free-field budgets stay noise-only), and the term's defects fixed (`flank_patrol` worst margin error 4.7 → 3.4 dB); D-228 the Doppler workstream scoped (`docs/tasks/DOPPLER.md`, nothing in the chain changed; simulated surface reverberation has no Doppler spread at long range). D-225 (the merge of D-204 to D-224: the full GPU suite, the failures of the earlier suites and their fixes, corrections, what is stale). 2026-11-26, one line each: D-204 five 3-D scenes with N × 2-D twins (3-D matters behind a seamount's flank and down a canyon); D-205 the old wall form's arrival mirrored, the general image construction in C++; D-206 and D-207 the first wave of `docs/tasks/SST_GAPS_JACKSON.md` (the book's and the surface papers' equalities as tests, measurement estimators); D-208 the conformance matrix (`docs/references/CONFORMANCE.md`) and the owner's "apply every alignment"; D-209 grain-size relations by default, site rows; D-210 and D-211 the sediment fit with small-slope and the row's prior (sigma2 to 0.005); D-212 `penetration: fluid`, the two buried scenes switched; D-213 `penetration: rough` (opt-in, both engines); D-214 and D-215 elastic rock (simulator, Bellhop, live chain; shear unconfirmed); D-216 `dahl_1997`, the attenuated facet term, the geometric-mean bistatic rule (surface reverberation about 1 dB lower in multipath scenes); D-217 the large-scale slope law (SPACE'08 above 12°: 1.89), `asiaex_dahl_2004`; D-218 the opt-in exact Kirchhoff tail (it does not mend that residual); D-219 and D-220 the simulator engine never silent, recorded in every record; D-221 the reverberation term of the link budget (then no site stated scattering; since D-227 the sites do); D-222 to D-224 four papers the owner supplied, Wagner shadowing and the 1989 model opt-in, five site rows. Earlier: D-203 the documentation sweep; D-202 the opt-in 3-D tier (its work is described in [`docs/tasks/PROPAGATION_3D.md`](../tasks/PROPAGATION_3D.md), section 5). One line each, since this set was written (D-182): D-183 the receive window laid out at `source.sound_speed`; D-184 one stream per board, merged; D-185 a board clock's rate configured or estimated; D-186 to D-188 Sensor Bridge at Popoto rates loses its first frames at start-up, the queue holds 4 s of stream, damaged frames named; D-189 a host check and runbook (the host measurement still not made); D-190 to D-192 and D-195 the frame path and board resampling faster in both engines, outputs to the bit (C++ 0.048 / 0.054 / 0.065 s a ping at 32 / 64 / 96 channels on an L4); D-193 the boundary models against Gallaudet 2001 (the surface model lies above the data at 3 m/s); D-194 stale simulator statements corrected; D-196 the K reverberation shape from a density per cell (opt-in); D-197, corrected by D-199, the eigenray tier against Bellhop3D's N × 2-D mode explained; D-198 `flank_patrol`'s small vehicle is below the scene's reverberation, kept as a hard target, and the link budget has no reverberation term; D-200 every receiver merges per-board streams; D-201 image sources off a tilted bed and walls (opt-in), four multipath scenes, the FLS chain tracks ghosts, SAS sharpness falls to a third with bed multipath. 2026-11-26: the opt-in `eigenray3d` tier (Bellhop3D as a separate process, the owner's approval), the cross-slope warning, `flank_seamount_3d` | `docs/DECISIONS.md` |
| Scenario files | 91 in `configs/scenarios/` (counted): 76 before D-201, its 4 multipath scenes, `flank_seamount_3d`, the nine 3-D scenes and twins of D-204, `asiaex_dahl_2004` (D-217) | `docs/SCENARIOS.md`, "Every scenario file" |
| Last full GPU suite | 2823 passed, 103 skipped, 4 xfailed, none failed, on an NVIDIA L4 (Hugging Face job 6b080457d91f621b930a652a; Bellhop, Bellhop3D, KRAKEN, PyRAM, Ceres, the Holoscan C++ operators, Sensor Bridge, `GPUSONAR_REQUIRE_ORACLES=1`, `GPUSONAR_SLOW_ORACLES=1`), the second batch's tree `de3717b` (merged as `61cfb66`; the later session-branch commits were documents and the tracker only). The one more skip is `test_tools_help.py`'s "not a git checkout" for the new tool. Before it, the merged tree `b88c458`: 2763 passed, 102 skipped, 4 xfailed (job 6b07ea74d91f621b930a343d, D-225); D-202's tree: 2301 passed, 102 skipped (job 6b079ab0d91f621b93098915) | D-229 |
| Last whole CPU suite in one run | the tree of D-157, `605a4e1` (2026-11-22; 157 of 157 ctest entries); not rerun in one go since. From D-158 to D-201 and the 3-D work each package ran its affected files only | `docs/STATUS.md`, "State of verification of this tree" |
| Gallery (`docs/figures/`, `gallery.json`) | regenerated at D-179 on an L4 (job 6b054768c97b8ed4fcd78015, tree `4a80876`). **Stale since 2026-11-26:** `buried_objects` (D-212 switched it to `penetration: fluid`; cylinder over clutter 21.12 -> about 20.3 dB), and the Popoto and flank scenes' surface reverberation fell about 1 dB with D-216 (not re-measured). `sst_validation.{md,json}`'s pursuit rows and `sst_pursuit.png` predate D-216; `docs/handoff/examples/live_popoto_12km_report.jsonl` predates D-227. A GPU regeneration job for these is issue `sonar-tmvu` (D-229) | D-179, D-212, D-216, D-225, D-229 |
| Milestones | 1 to 10 of 13 done; 12 (propagation and reverberation) done as far as open literature allows (audited and closed 2026-11-25: `sonar-dzu`, `sonar-k7m`; follow-ups filed); 11 (Holoscan) and 13 (performance model, Jetson) open | `docs/STATUS.md`, "Where things stand"; `bd` |
| Open `bd` issues | 77 not closed of 176 when counted before the merge `b88c458`: **P1 6** (3 in progress, 3 open), **P2 7**, **P3 39** (1 in progress), **P4 25** (1 in progress); not recounted at D-225 or D-229 (the orchestrator files the batches' follow-ups) | `.beads/issues.jsonl`, 2026-11-26 |
| Run on hardware or real data | nothing | `docs/STATUS.md`, first paragraph |

The open issues by theme, with ids, are in [`docs/HANDOFF.md`](../HANDOFF.md), section 8.

## Top risks and unknowns

1. **Nothing has met hardware, water or recorded data.** Every result is simulated
   (`sonar-sul`, P1).
2. **The hardware facts are unconfirmed**: converter (102.4 kHz, 24 bit, simultaneous),
   board synchronisation to about 1 microsecond, the projector and its 197 dB, boards used
   at 4 to 6 kHz against a 10 to 100 kHz specification (STATUS questions 2 to 4). Per-board
   streams need the boards' capture starts true to about 0.8 ms and their clock rates known
   (`sonar-9a2`); a grating lobe is in view beyond ±42° at 6 kHz on five of six builds
   (`sonar-68e`); the synchronisation reference, input protection, power and mechanics are not
   designed (`sonar-mu3`).
3. **Reverberation cuts the small targets' reach to a few kilometres, on illustrative site values.**
   Since D-227 the sites state their boundary scattering and the depth plan and the coverage pictures
   include it: the +5 dB class over the 100 m shelf reaches 2.56 km in the plan (8.02 to 9.58 km
   before) and 1.24 to 1.37 km in the face deployment files' coverage (6.37 to 8.10 km before); the
   +25 dB classes stay noise-limited at 12 km. Every site value is ours (Lambert −27 dB on the shelf);
   real values per site are wanted (STATUS question 40). The free-field link budgets have no site and
   stay noise-only. With the term `flank_patrol`'s −3 dB small vehicle is predicted at 10.3 dB over the
   background against 6.9 dB measured (the noise-only budget gave 34.3 dB); worst margin error 3.4 dB,
   the rest the targets' motion and extent (D-227). The term still leaves out element patterns and a
   baffle, K-distributed reverberation in the threshold and Doppler (STATUS item 58).
4. **One row is 1.1 dB short at 12 km** (detection probability 0.22 a ping), on a +25 dB
   target strength that is a placeholder.
5. **The water decides more than the margin**: the same ship is not detected at 4 km in
   summer over mud, nor tracked at its true place in deep water over silt. Over relief that
   slopes across the look the shipped propagation is N × 2-D, 3 to 4 dB too loud over a
   seamount's flanks; the opt-in `eigenray3d` tier has no reverberation over relief.
6. **Depth at long range is an indication only**: at 12 km the ship reads 8 to 35 m for a
   true 4 m; two rows at 3 km in 100 m read the surface or the bed.
7. **Track association among many multipath arrivals in one gate** is fragile at 12 km
   (seed 60: 65 m off; issue `sonar-tlc`). The gallery shows one good seed. **The FLS chain
   tracks multipath ghosts as objects** off walls and a sloping bed (`sonar-izys`, D-201).
8. **False movers from a rough surface** (78 over ten seeds in one FLS scene) and **small
   targets missed** (the kayak; `flank_patrol`'s small vehicle, which is below the scene's
   reverberation by design, D-198) are known limits with no fix. **SAS results are the
   free-field case**: every shipped SAS scene but `sas_auv_multipath` has no multipath, and with
   bed multipath the image's sharpness falls to about a third (`sonar-ya2c`).
9. **The Sensor Bridge path has only run against the board emulator** with a UDP patch. Its
   real-time margin is one socket buffer (about 20 ms of `flank_32`'s stream, 4 to 13 ms at a
   Popoto build's rate). At Popoto rates the first 12 to 20 ms of a stream are lost at start-up,
   so a board must stream 0.25 s before the first transmit that matters, and nothing enforces it
   (`sonar-a75`). A host with Sensor Bridge's own socket setup has never been tried
   (`sonar-02c`); per-board frames have run only at `flank_32`'s rate (`sonar-pnyv`).
10. **Navigation**: the lever arm is zero until surveyed, the array's velocity is not
    corrected for it, and a source without timestamps yields speed 0 for every track.
11. **Some generated files and documents are stale** (map below): `popoto_12km.json` and
    `fjord_depth_plan.md` predate D-156 (the depth planner file was regenerated with D-227);
    `docs/handoff/examples/live_popoto_12km_report.jsonl` predates D-227 (`sonar-tmvu`); the benchmark files are of pings with ideal clocks (no
    board synchronisation; still right for what they measure); the main checkout's built
    Python module was older than its sources on 2026-11-24.

## Waiting on the owner

| List | Where | Count | Blocks |
| --- | --- | --- | --- |
| Open questions for the project owner | [`docs/STATUS.md`](../STATUS.md), section of that name | 50 numbered; 1 to 12 are the Popoto array (rows, band, converter, projector, static frame, navigation unit, target classes, waters, dive, recording, receive windows, coverage site); 30 to 34 were raised since D-182 (board timing, the grating lobe, the fjord beds, SAS with bed multipath, 3-D propagation); 36 to 42 were raised by the first 2026-11-26 merge and answered (D-229), with a measured rock shear speed (38) and real scattering values per site (40) still wanted; 43 to 49 are Doppler (from `docs/tasks/DOPPLER.md` section 5); 50, the default roughness model against the published bistatic join, is answered (D-229: small-slope stays); 17, 19, 22, 28, 29, 35 to 42 and 50 are marked applied or answered | the hardware design and its defaults |
| Vehicle adapters | [`docs/tasks/VEHICLE_ADAPTERS.md`](../tasks/VEHICLE_ADAPTERS.md), section 3 | 17 | issue `sonar-o7m` (P1): board samples and navigation into the live chain |
| Real data | [`docs/tasks/REAL_DATA.md`](../tasks/REAL_DATA.md), section 3 | 11 | issue `sonar-sul` (P1): simulator calibration and the live chain on real data |
| Doppler | [`docs/tasks/DOPPLER.md`](../tasks/DOPPLER.md), section 5 | 7 (the same as `STATUS.md` questions 43 to 49) | issue `sonar-u1fu`: packages D2 (a Doppler-resolving waveform), D6 (the budget's Doppler credit) and any default; the owner agreed that D4 (the sweep replica bank) and D2 go first, D3 (the chain's tone path) after (D-228) |
| 3-D propagation | [`docs/tasks/PROPAGATION_3D.md`](../tasks/PROPAGATION_3D.md), section 6 | 9; question 1 answered (Bellhop3D as an opt-in provider, a separate process, 2026-11-26); 5 answered by `sonar-6c8`; 6 done; 8 settled as built; 9 narrowed by D-199 | open: relief across the look in the owner's waters (2), ranges and slopes (3), run time and disk (4), walls and banks in the FLS and SAS use (7), the 2-row fjord scenario's 12-arrival limit (9, `sonar-2e9e`) |

Open P1 and P2 issues and what each waits on:

| Issue | Priority | Waits on |
| --- | --- | --- |
| `sonar-o7m` vehicle adapters | P1 | the owner's answers (VEHICLE_ADAPTERS section 3) |
| `sonar-sul` real data | P1 | data from the owner (REAL_DATA section 3) |
| `sonar-w2w3` reverberation in the link budget and coverage | P1 | work only (a package to approve) |
| `sonar-l59` M11 Holoscan, `sonar-3s7` M13 performance and Jetson, `sonar-k29` stacked rows | P1, in progress | hardware (a rig, a Jetson or DGX Spark) and the reference's SDK build |
| `sonar-9a2` board timing | P2 | the owner: how capture starts are stamped, the boards' rates, the channel map |
| `sonar-68e` grating lobe | P2 | the owner: limit the sector, change the pitch, or accept |
| `sonar-mu3` hardware design gaps | P2 | hardware design work and choices |
| `sonar-02c` Sensor Bridge on a real host, RoCE | P2 | a host with Sensor Bridge's socket setup; a RoCE build and a ConnectX card |
| `sonar-pnyv` clock-rate keys ignored for a whole-array stream | P2 | work only |
| `sonar-izys` FLS ghosts tracked | P2 | work; any chain default change goes to the owner |
| `sonar-ya2c` SAS with bed multipath | P2 | the owner: should SAS scenes carry bed multipath |

Standing rule: the owner approves each work package, and any change that moves a default
result, before it is merged (`docs/HANDOFF.md`, section 2).

## Document map

"Last commit" is `git log -1` of the file at `933972b` (2026-11-24); every file marked "swept
2026-11-26" in the last column was brought to D-201 and the 3-D work in one documentation sweep
that day (issue `sonar-tadd`). "Past D-160" says whether the file mentions any decision from
D-160 on. A file can be old and still right; the last column says what is known to be stale. The stale statements this map first listed in `STATUS.md`,
`RESULTS.md`, `DEPLOYMENT.md`, `SCENARIOS.md`, `REFERENCES.md`, `REFERENCE_DESIGN.md`,
`ARCHITECTURE.md`, `API_AUDIT.md`, the diagrams and the root `README.md` were corrected on
2026-11-24, after the map was made; the last column says so where it applies, and the "Last
commit" dates of those files are a day or less older than their present state.

| File | What it is | Last commit | Past D-160? | How current |
| --- | --- | --- | --- | --- |
| `handoff/README.md` | this page | 2026-11-24 | yes | current to D-201 and the 3-D work (swept 2026-11-26) |
| `handoff/HARDWARE.md` | the hardware side: what is unknown or at risk, the system block diagram, a register of every assumption with its status (confirmed, via secondary, ours, unconfirmed), the six Popoto builds with their numbers (`hardware_builds.json`, `tools/handoff_hardware.py`), the link budget and the 12 km design point, the interfaces the hardware must meet (ping record, UDP block, WAV sidecar, timing, navigation datagram), a bring-up plan, open hardware decisions | 2026-11-24 | written with this set | swept 2026-11-26 to D-201 (Sensor Bridge start-up loss, per-board streams, chain times with board synchronisation, the budget's missing reverberation term, issue ids); since D-227 its budget section says the free-field budget is noise-only and the plan and coverage at the builds' site include the boundaries' reverberation. Nothing in it is built or measured on hardware, as its first paragraph says |
| `handoff/SOFTWARE.md` | the software side: what is fragile or wrong, the layers and data flow, the repository map, the two live-chain engines, the live chain stage by stage with every default, its key and its decision, SAS, FLS and the simulator, configuration, tools and programs, how to add things without breaking the rules | 2026-11-24 | written with this set | swept 2026-11-26 to D-201 (sections 1, 5.1, 5.4, 6, 8), to D-225 and to D-229 (the link-budget and Doppler rows of section 1). Its section 10 lists stale statements found in other documents; those were corrected on 2026-11-24 after it was written (except root `CLAUDE.md`, the owner's file), so read that section as history |
| `handoff/TESTS.md` | how to run the tests (CPU, GPU job, oracles), a map of what each test file protects, the tests that pin decisions D-155 to D-227, known false failures and how to recognise them, what is not tested or only on simulated input | 2026-11-24 | written with this set | swept 2026-11-26: counts per file collected then, the new test files, D-183 to D-201 and the 3-D tier's tests, the switches; brought to D-225 and then to D-229 the same day (counts, the new files, D-204 to D-227 in section 3). No per-file run time is recorded; the 103 skips of the last GPU job are D-225's 102, derived, not read from its log, and the one D-229 names |
| `handoff/SIMULATION.md` | the simulator: what it does not model, its pipeline and a table of its models with their sources and validation, all 91 scenarios by family, calibration to real data, determinism | written with this set | written with this set | swept 2026-11-26: D-193, D-196, D-197/D-199, D-198, D-201 and the `eigenray3d` tier in its limits, models and validation tables; `flank_seamount_3d` in the catalogue; brought to D-225 the same day (sediment penetration, elastic rock, the surface and time-spread defaults, the engine, the fit tool, `asiaex_dahl_2004`) and to D-229 (the Williams-Jackson opt-ins, `reverberation: site`, no surface Doppler spread, the source level before the window) |
| `handoff/EXAMPLES.md` (with `handoff/examples/`: scripts, pictures, saved outputs) | twelve worked examples run on the CPU with their commands, run times, real outputs and pictures, led by what failed (`--emulate` on multi-board scenarios) | written with this set | written with this set | the same |
| `handoff/OUTPUTS.md` | pictures, headline numbers, a report line, how to regenerate | 2026-11-24 | yes | the gallery as of D-179; text swept 2026-11-26 (SAS free field, the small vehicle, Sensor Bridge at Popoto rates, chain times, the budget; the sites' reverberation in the plan and coverage, D-227) |
| `HANDOFF.md` | rules, build and test, GPU jobs, open work, how to work here | 2026-11-24 | yes | swept 2026-11-26 to D-201 (sections 2 to 11; section 8 rebuilt from the issue export) |
| `STATUS.md` | what works, how it was checked, what is wrong, owner questions | 2026-11-26 | yes (to D-229) | swept 2026-11-26: the state lines (D-201's GPU job, the whole CPU suite last at D-157), chain times with and without board synchronisation, milestones 11 and 12, coverage item 24 (no reverberation term), Sensor Bridge item 34, owner questions 30 to 35; brought to D-225 the same day (the last GPU suite, stale generated files, items 24, 27, 30, new items 46 to 57, owner questions 36 to 42); brought to D-229 the same day (the last GPU suite, items 24 and 50 with the sites' scattering, the `flank_patrol` margins, new items 58 to 63, questions 36 to 42 answered, new questions 43 to 50, the Doppler limits, what is next) |
| `DECISIONS.md` | every decision D-1 to D-229 with what got worse | 2026-11-26 | yes | current; the authority after code and configs. Append and amend only; pointers "(amended by D-xxx)" added 2026-11-26 where a later decision changed an entry (also on D-85, D-86, D-87, D-88, D-132, D-181, D-193, D-198, D-201, D-202 for D-204 to D-224) |
| `RESULTS.md` | every measured number and its source | 2026-11-26 | yes (to D-201) | swept 2026-11-26: "Performance" opens with the chain times with board synchronisation (D-190 to D-195); "Sensor Bridge at Popoto rates" carries D-188 to D-192 and D-200; the SAS sections say the scenes are free field; the link budget and coverage sections say they have no reverberation term (since D-227 "The link budget against the chain" carries the term's `flank_patrol` figures and says the ray count was not the cause of D-221's margin error); a summary table of D-193 to D-201 and the 3-D tier before "Not yet measured". Still: D-160, D-161, D-176 and D-181 are measured in their `DECISIONS.md` entries only; the `minefield_sas` row is not re-timed (issue `sonar-egj`) |
| `SCENARIOS.md` | every scene with figures and per-scene results | 2026-11-26 | yes (to D-201) | current with the D-179 gallery; `flank_patrol`'s small vehicle described as a hard target (D-198); the SAS scenes as free field. CPU times in its table are from a much older tree |
| `CONFIGURATION.md` | every config key | 2026-11-26 | yes (to D-229) | current: per-board and clock keys, the K density, image planes, the top-level `bathymetry` and the `eigenray3d` options are documented; `receive --queue` is listed beside the keys; since D-226 and D-227 the `williams_jackson_1998` models, `bottom_scattering`, `surface_scattering`, `reverberation: site` and the coverage block's new keys |
| `DEPLOYMENT.md` | running the live chain: what the vehicle provides, the report line, staged build, Sensor Bridge, the host check | 2026-11-24 | yes (to D-200) | swept 2026-11-26: chain times with board synchronisation beside the ideal-clock benchmark figures; the Popoto-rate section's superseded statements marked; "What `snr_db` is" says the budget has no reverberation term; since D-227 it says the table is noise-only and the coverage picture has the site's reverberation |
| `DESIGN.md` | generated design report: requirements against configs, link budgets | 2026-11-26 | no | held to the configs by `tests/python/test_design.py`. 2026-11-26: the generators (`tools/design_report.py`, `tools/reference_plan.py`) say the GPU timings are of pings with ideal clocks, and since D-227 that the free-field budget is noise-only (no site) while the depth plan and the coverage picture include the build's site's boundary scattering (illustrative values; the +5 dB class limited to a few kilometres, the +25 dB classes noise-limited at 12 km). Regenerated at `bf55a7b` with the depth planner file (D-227; the test passes). **Stale:** its 12 km sections read `design/popoto_12km.json`, last regenerated 2026-11-20 (`STATUS.md`, list of stale generated files) |
| `REFERENCE_DESIGN.md` | from a spec sheet to a build: the build-spec fields and steps | 2026-11-18 | no | the method holds; since 2026-11-24 it points to the lever-arm keys of D-167 (not a spec field: generated deploy files do not carry them, issue `sonar-vjq`) |
| `ARCHITECTURE.md` | data flow, backends, where each stage runs, the propagation tiers | 2026-11-26 | yes | structure holds; predates the Sensor Bridge receive work (D-161 to D-163, D-173) and the option-key refusals (D-166, D-177), and says so; since 2026-11-26 it names the five propagation tiers. Prefer `handoff/SOFTWARE.md` for the live chain |
| `PHYSICS_NOTES.md` | derivations, Secs. 1 to 33 (31 the sediment tier's penetration, 32 the link budget's reverberation, with the sites' one rule since D-227, 33 bottom scattering after the 1989 and 1967 papers, with 33.1 after Williams and Jackson 1998 and Jackson and Briggs 1992, D-226) | 2026-11-26 | yes | physics of the simulator and chain to D-144, with the Gallaudet comparison (D-193) and section 30, image sources off tilted planes and walls (D-201). The derivations of D-159 (outlier test), D-167 (lever arm), D-176 (parabola bearing) and D-181 (range-Doppler) are in their tests' docstrings and `DECISIONS.md`, not here |
| `PAPER_NOTES.md` | reading of the ASASIN paper | 2026-11-23 | yes (D-164) | current |
| `REFERENCES.md` | register of sources with status (confirmed, via secondary, ours, unconfirmed) | 2026-11-26 | yes | registered: section 12.7 (D-159, D-167, D-176, D-181); Gallaudet 2001's measurements (D-193); Abraham and Lyons 2002 (D-196); BELLHOP3D's role as the `eigenray3d` tier's program; Mopin 2023 and Steininger 2013 (2026-11-26, section 3.2); section 10 updated; after the merge `b88c458`: Mourad and Jackson 1989, Wagner 1967 (citation corrected), Williams and Jackson 1998, Jackson and Briggs 1992, SplitMix64 (unconfirmed), rock elastic; after the merge `61cfb66`: their opt-in uses (D-226). `docs/references/CONFORMANCE.md` (the conformance matrix and its apply list, applied except A11) |
| `RECORD_FORMAT.md` | the HDF5 record, schema 1.0.0 | 2026-11-18 | no | current: the schema is frozen and no decision since has changed it |
| `OPEN_SOURCE.md` | every open-source dependency and decision | 2026-11-26 | yes | current as far as checked: no dependency was added from D-158 to D-201; Bellhop3D's new role (the opt-in `eigenray3d` tier's program, a separate process, the owner's approval) is recorded in section 3.13 and the Acoustics Toolbox row |
| `HOLOSCAN_NOTES.md` | source-reading notes on Holoscan and Sensor Bridge | 2026-11-22 | no | reading notes; what was then built and measured is in `DECISIONS.md` D-151 to D-173 and `DEPLOYMENT.md` |
| `API_AUDIT.md` | audit of the API at `2c5bb8a` (2026-11-20), findings A1 to G5 | 2026-11-21 | no | historical: the findings are kept as found at `2c5bb8a`; the fixes followed (D-145 to D-147 and later packages) and its header says so since 2026-11-24. Do not read a finding as open without checking the code |
| `HARDCODED_AUDIT.md` | audit of hard-coded assumptions at `3113327` | 2026-11-18 | no | historical; not re-checked here |
| `ORG_ARL_SURVEY.md` | survey of the org-arl repositories | 2026-11-20 | no | historical reading survey, dated 2026-11-17 |
| `SIM_CALIBRATION.md` | calibrating the simulator to real data | 2026-11-16 | no | method only; never run on real data (issue `sonar-sul`) |
| `JETSON.md` | build steps for Jetson and DGX Spark | 2026-11-26 | no | **untested**, as its first line says; since 2026-11-26 it names the host check (D-189) and the L4 chain times to compare with |
| `tasks/VEHICLE_ADAPTERS.md` | brief for the vehicle adapters, with 17 owner questions | 2026-11-26 | yes | current; its "current state" was checked at `6600df8` and brought to D-201 on 2026-11-26 (per-board streams in every receiver, the 0.25 s lead, `source.sound_speed`, `sonar-57w`, `sonar-9a2`) |
| `tasks/DOPPLER.md` (with `tasks/doppler_baseline.json`) | the Doppler workstream (`sonar-u1fu`): what is not known or wrong, what exists, a measured baseline on existing scenes (`tools/doppler_study.py`), design options for the flank array, 7 owner questions, packages D1 to D9 | 2026-11-26 | yes | current; checked in the code at `f5054b4` (D-228). No default, scene, chain or record changed |
| `tasks/PROPAGATION_3D.md` | the scope of 3-D propagation (`sonar-rcq`): what is wrong, the `sonar-6c8` investigation, options, packages P0 to P6 (done 2026-11-26, with what differed from the plan), the cost of the `eigenray3d` tier, 9 owner questions | 2026-11-26 | yes | current; a state paragraph opens section 0; owner questions 6 to 9 annotated (question 9 corrected by D-199) |
| `tasks/REAL_DATA.md` | brief for real data, with 11 owner questions | 2026-11-26 | no | current; checked at `6600df8`, its self-check numbers corrected 2026-11-26 |
| `benchmarks/README.md` | which command wrote which benchmark file | 2026-11-20 | no | current for `bench_*`, `session_*`, `conditioning_*`; later files name their command in `RESULTS.md` |
| `benchmarks/*.json`, `*.md` (35 files) | raw benchmark, ablation and ten-seed evaluation output | newest 2026-11-22 | n/a | measurements at the commits in their names; none rerun at HEAD. The live-chain cases are of pings with ideal clocks (no board synchronisation stage), so D-190 to D-195 do not apply to them; the trials' chain times with board synchronisation are in the decisions only (D-191, D-192, D-195). No file for the ten-seed runs of D-172 to D-181 |
| `figures/` (88 files) | the gallery (57 files), validation tables and figures, study output | newest 2026-11-24 | n/a | gallery current (D-179); the other tables and figures date from 2026-11-18 to 20 |
| `scenarios/` (56 files) | geometry diagram per scenario, `transmission_loss.json` | newest 2026-11-24 | n/a | 55 of the 91 scenario files have a diagram (`SCENARIOS.md` names the 36 without). The diagrams of `wreck_sas` and `minefield_sas` were redrawn on 2026-11-24 with the wider seabed of D-171; the others date from 2026-11-20 or before |
| `design/` (38 files) | `derived.json`, study JSON, build and system diagrams, parts | newest 2026-11-26 | n/a | `popoto_depth_planner.json` and `.png` regenerated on the CPU at `bf55a7b` with the sites' boundary scattering (D-227; 83 of 117 plans moved), and the range study's note (free field, nothing to scatter). **Stale:** `popoto_12km.json` (predates D-156 to D-201): `STATUS.md`, list of stale generated files. `popoto_parts.md` and `.json` and the `gpu` block of `popoto_stage2.json` were regenerated on 2026-11-24; the parts list's GPU timing does not say its pings have ideal clocks (not regenerated) |
| `architecture/` (6 files) | diagrams for `ARCHITECTURE.md`, `DEPLOYMENT.md`, `SIM_CALIBRATION.md` | 2026-11-19 | n/a | drawn by `tools/architecture_diagrams.py`; its backend table is checked by `tests/python/test_architecture.py`. `backends.png`, `dataflow.png` and `live_chain.png` redrawn 2026-11-24 (ten stages with a device build, not seven; the Sensor Bridge path; the multiscale detector and echo grouping) |
| `examples/reference_design/example_static_2x24.yaml` | the worked example of `REFERENCE_DESIGN.md` | 2026-11-24 | n/a | current (touched by D-180's YAML numbers) |
| `plan/GPU-FLS-Phased-Plan.pdf` | the owner's phased plan | 2026-11-16 | n/a | input document, unchanged |

Outside `docs/`: the repository `README.md` was brought to D-201 on 2026-11-26, to D-225 the same
day (91 scenarios, the last GPU job, the reverberation term) and to D-229 (the last GPU job, the sites'
boundary scattering in the plans and coverage), and it points here. The root `CLAUDE.md` and `AGENTS.md` are the sonar suite's own instruction files since D-203
(what the repository is, what to read first, the owner's rules, build and test, how work is done).
