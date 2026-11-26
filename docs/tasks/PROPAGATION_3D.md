# Task: 3-D propagation for the eigenray tier

Issue `sonar-rcq` (P3), with `sonar-6c8` before it. Written 2026-11-26 at commit `48cc580`
(branch `fix-rcq`) as a scoping document: no code, config or test was changed. Read
`docs/HANDOFF.md` first, then this file. Status words are those of `docs/REFERENCES.md`
(**confirmed**: read in the opened source; **via secondary**; **ours**: our derivation or
measurement, stated as such; **unconfirmed**). "Measured here" means run on this machine on
2026-11-25/26 with the commands in section 8.

## 0. Answer first

**State on 2026-11-26, after this scoping was written** (read section 5's table for each
package): the owner answered question 1 (Bellhop3D as an opt-in provider, run as a separate
process). P1 (the cross-slope warning) and P2 to P6 (`sim/bellhop3d.py`, the opt-in `eigenray3d`
tier, its acceptance test behind `GPUSONAR_SLOW_ORACLES=1`, the top-level `bathymetry` key and the
illustrative `flank_seamount_3d`) are done, with what differs from the plan said in each row. The
tier has no reverberation over relief. Option F was filed and done as `sonar-0nsi` (D-201). D-199
corrects what this file said of the fjord beds (question 9). The decision entry for the 3-D tier
is D-202 in `docs/DECISIONS.md`; issue `sonar-rcq` is closed with it.

**Will 3-D propagation make FLS or SAS better? No.** Neither family uses the tier that 3-D
propagation would change. None of the 12 FLS scenarios and none of the 10 SAS scenarios is at
the `eigenray` tier (`docs/handoff/SIMULATION.md`, section 4). FLS uses `free_field` or
`image_source`, SAS uses `free_field` (and the buried-object scenes use `sediment`). Bellhop3D
would change no FLS or SAS record. The physics agrees, from the scenarios' own numbers (ours,
section 3):

- **Refraction is negligible at their ranges.** FLS sees 5 to 450 m and SAS sees 38 to 68 m slant.
  A strong vertical gradient (0.05 /s) bends a 300 m ray by 0.57° and sags it 1.5 m. This is a
  2-D effect, and it adds 0.8 µs of delay. A horizontal gradient of 0.005 /s, which is strong for
  a front, turns a 300 m ray by 0.06°. The FLS beam at 120 kHz is 0.93° wide (128 × 6 mm array,
  `configs/sonars/fls_120k.yaml`). Over a SAS path of 50 m the same gradient adds 4e-11 s, while
  SAS resolution is about 2 cm.
- **The 3-D effect that does matter at these frequencies is geometric: reflection off tilted
  planes.** One bounce off a bed tilted by α across the bearing, at grazing angle g, turns the ray
  out of its vertical plane by φ = atan(tan g · sin 2α). This is ours; the derivation is in
  section 3. FLS bounce paths to the bed reach it at 15 to 58° grazing (sonar 20 m deep in 60 m,
  ranges 50 to 300 m). A 5° cross-slope at those angles gives φ = 3.6 to 9.9°, which is 4 to 11
  FLS beamwidths. (Corrected 2026-11-26, `sonar-0nsi`: 3.6 and 9.9° are the formula's values at 20 and
  45°, as in the table of section 3; at 15 and 58° it gives 2.7 and 15.5°, 3 to 17 beamwidths. The
  formula itself was re-derived and is held by `tests/python/test_image_planes.py`.) A vertical quay or canyon wall is the extreme case. In water of one sound speed,
  **this is exactly an image source off a tilted plane, not something to ray-trace.** Bellhop3D
  is the wrong tool for it. Its seabed is one depth per (x, y) (`ReadBTY3D`, read for the format
  in `tests/python/bellhop3d_oracle.py`), so it cannot represent a vertical wall. Its cost is set
  by ray tracing a whole field, where FLS and SAS scenes need a few hundred metres of straight rays.
- **Where FLS and SAS realism is limited today** (and effort would pay off):
  - Nothing has met real data (`sonar-sul`, `docs/tasks/REAL_DATA.md`).
  - Every target strength is a placeholder (STATUS question 7).
  - Objects are Lambert point clouds without specular echoes in realistic scenes
    (`SIMULATION.md` section 1).
  - FLS scenes have no bounce paths to objects, and the pool scenes have no wall multipath,
    although `ImageSourceProvider` already takes axis-aligned vertical walls (`SCENARIOS.md`,
    "What no scenario has yet"; `sim/propagation.py`). (Corrected 2026-11-26, `sonar-0nsi`: the
    four FLS scenes at `image_source`, `fls_120k_transit`, `_170k_transit`, `_330k_transit` and
    `fls_170k_traffic`, do give their objects first-order surface and bottom bounce paths (three
    eigenrays a leg, nine pairs a scatterer); the free-field FLS scenes do not. `SCENARIOS.md` said
    otherwise.)
  - SAS scenes are `free_field`, with no surface or bottom multipath at all. (Since D-201,
    `sas_auv_multipath`, not in the gallery, has both; every other SAS scene is still free field.)
  - There are 78 false movers from a rough surface over ten seeds, and small targets such as
    the kayak are missed (`docs/handoff/README.md`, risk 7).
  - `fls_170k_transit` reverberation is 1 to 1.5 dB off (STATUS item 32).
  - SAS motion estimation has its own limits (STATUS, "Motion estimation limits so far").
  - Walls and tilted planes in `image_source` would give FLS and SAS their missing 3-D multipath
    at a small fraction of the cost (section 4, option F).

**What 3-D propagation would change: long-range work (Popoto, flank array) over relief that slopes
across the bearing, beyond a few water depths.** Measured against Bellhop3D (D-149,
`tests/python/test_seamount_3d.py`):

| Case | Size |
| --- | --- |
| Over a 180 m seamount in 200 m of water, 8 kHz, flank slope up to 12°, bearings over the flanks, averaged over 3.5 to 6.5 km | our N × 2-D tier is 3.8 dB too loud (converged beams) |
| Same, over the summit | 2.2 dB too loud |
| Same, beside the seamount | 0.0 and −0.7 dB |
| Same, 0.2 to 1.8 km | Bellhop3D's 3-D and N × 2-D modes agree within 0.07 dB |

**No shipped scenario has such relief.** All 22 `eigenray` scenarios have either a level bed or
(the three `fjord_*`) one range profile used on every bearing. Both are axisymmetric about the
sonar, so the bed has no slope across any bearing, and N × 2-D is the full answer in the
simulator's own world. A 3-D provider would move no shipped record through physics. Run in place
of Bellhop it would move records only through the differences between the two programs. Over a
level bed there are none: the 1.0 dB first reported as `sonar-6c8` was the oracle's beam count
(section 2, measured 2026-11-26).

**The scope therefore depends on the owner's waters.** If the operating areas have steep relief
across the look direction (banks, seamounts, canyons, fjord walls), the gap is 3 to 4 dB at a few
km, and an opt-in Bellhop3D provider is the right fix. It is about 4 to 6 weeks of work
(packages P2 to P6 in section 5); the owner has since approved its licence use (question 1). If the waters
are shelf-like (slopes of a degree or less), the effect is small. The first-principles estimate is
that φ is 0.2° per bounce at 5° grazing over a 1° cross-slope, against a 3.6° array beam. In that
case the right move is: add a warning where the cross-bearing slope is large, and keep the
characterisation test (`sonar-6c8` is explained in section 2 and needs no change to shipped runs
over a level bed).

**Recommended:**

1. `sonar-6c8` (P0) is done as an investigation (section 2): the 1.0 to 2.5 dB is mostly the
   oracle's beam count (Bellhop3D, 121 Gaussian elevation beams) and, over relief, Bellhop's
   arrivals output over a piecewise-linear bed. The 19 level-bed eigenray scenarios need no
   change; the three fjord scenarios' bed representation is owner question 9.
2. Add the cross-slope warning (P1, size S). It changes no record.
3. Build option A, Bellhop3D as an opt-in external provider, only after the owner answers
   questions 1 to 3 in section 6 with "yes" and "steep".

## 1. What is unknown, wrong or risky

1. **The owner's waters are unknown.** This is STATUS open questions 8 ("typical water depths and
   bottoms") and 12 (the coverage site is the illustrative `shelf_100m_sand`). The whole case for
   3-D propagation rests on cross-bearing slopes nobody has given us.
2. **Our 2-D tier and Bellhop3D's N × 2-D mode disagree by 1.0 to 2.5 dB on the seamount bearings,
   and by up to 1.0 dB over a level 200 m bed at 7 km** (`sonar-6c8`; the test docstring of
   `test_our_n_by_2d_tier_over_the_seamount`). Explained since (section 2): over a level bed it is
   all the oracle's beam count, and the two programs agree to 0.001 dB at equal beams. Over relief
   0.4 to 1.7 dB of it is Bellhop's arrivals output over our piecewise-linear (`'L'`) bed, which a
   parabolic equation does not support on the gentle bearings where it can referee. What remains
   risky: which bed representation is right for the fjord profiles (steep, coarse), where neither
   `'L'` nor `'C'` makes Bellhop's arrivals agree with its TL run.
3. **Bellhop3D's arrivals are not eigenrays.** Measured here: a 3-D arrivals run on the seamount
   with the test's fan returns up to 1,207 to 1,272 arrivals per receiver, and about 360 on
   average. Bellhop 2-D on one bearing returns at most 106. These are beam contributions that the
   program does not merge. Our energy rule (D-105: keep arrivals within 20 dB of the strongest,
   cap 64) would discard most of their energy unless the reader merges them first. This is the
   main technical risk of option A.
4. **A dense 3-D lookup does not fit on disk.** Measured here: 18,850 receivers wrote 1.1 GB of
   ASCII arrivals. The `popoto_12km` grid (range step 40 m, depth step 2 m, to 13.2 km) over a ±30°
   sector at 1° is about 1.0 M receivers, or about 60 GB at the same density (estimate). A
   provider must ask only for the receivers it needs: Bellhop3D's irregular-grid option, which its
   documentation says "has not been used much" (confirmed, `doc/bellhop3d.htm`). It also needs
   the binary output option (`'a'`, same page; not timed here) and to merge arrivals as it reads.
5. **Over 3-D relief the field depends on where the source is, not only its depth.** The 2-D
   provider caches one run per source depth because its ocean moves with the sonar (rings, D-110).
   A 3-D provider needs a run per source position (or per a few tens of metres of track). That
   multiplies the cost by the pings of a moving platform.
6. **Bellhop3D is GPL-3.** `docs/OPEN_SOURCE.md` rule 2 allows GPL code as a separate-process test
   oracle "or an optional external tool". Rule 3 requires the owner's sign-off for any copyleft
   item in a build, image or CI. Bellhop already runs this way as the eigenray tier's tracer.
   Bellhop3D is today an oracle only (D-149). The owner has approved it as an opt-in provider
   run as a separate process (2026-11-26, question 1); nothing ships it yet.
7. **Bellhop3D is "not independently re-validated"** (`OPEN_SOURCE.md`, Acoustics Toolbox row).
   Its own beam convergence on the seamount was measured: going from 121 × 361 to 241 × 721
   beams, the 3-D range averages move by up to 1.1 dB and the 3-D minus N × 2-D difference by
   0.6 dB (test docstring). Most of that is the elevation count: Gaussian beams need 481 over ±30°
   (section 2, `sonar-6c8`).

## 2. What exists

| Piece | Where | What it does | Right | Wrong or missing |
| --- | --- | --- | --- | --- |
| `BellhopProvider` | `python/gpusonar/sim/bellhop.py` | Runs `bellhop.exe` (GPL-3, separate process; `GPUSONAR_BELLHOP`). It keeps one arrivals lookup per source depth on a (range, depth) grid (`ranges`, `depths`: `range_step`, default `max_range / 200`; `depth_step`, default `depth / 50`). Lookup is to the nearest node, with a first-order delay correction `(dr cos a + dz sin a) / c` from the node to the point (D-36 conjugation, D-111 local speed) | Against KRAKEN at 8 kHz: at most 0.21 dB, 1 to 10 km (RES, `test_kraken_crosscheck.py`). Against PyRAM: 500 m windows within 1.42 dB, depth means within 0.44 dB (D-149) | One environment for every bearing ("rings about the sonar", D-110). Launch and arrival vectors lie in the vertical plane of the bearing (`unit` in `eigenrays`): no out-of-plane azimuth |
| Arrival keeping | same, `eigenrays` | Arrivals within `arrival_floor_db` (20) of the strongest, up to `max_arrivals_cap` (64), or a stated `max_arrivals`. It warns and records `truncation` and `dropped_energy` (D-39, D-105, D-137) | Says what it drops | `popoto_env_gravel_30m` drops 14 % (`sonar-9b8`) |
| `along_profile` | same | A copy of the provider whose `env.bathymetry` is the depth profile along one bearing; Bellhop traces over it (`'A~'` bottom, `.bty` file) | In-plane relief, including blocking by the relief | Used **only** by the reverberation over relief (`sim/reverb_relief.py`, D-73), one run per azimuth sector. It is N × 2-D about one ground point: a bistatic pair raises `NotImplementedError`. Target echoes do not use it: `simulator.py` calls `two_way_paths` with the base provider (`make_provider`), whose bed is the same on every bearing |
| What consumes eigenrays | `propagation.two_way_paths` (NumPy, and C++ `core.sim_two_way_paths`), `reverb.py`, `reverb_relief.py`, `timespread.TimeSpreadProvider`, `lloyd.py`, `depthplan.py` | `EigenraySet`: delay, amplitude, `launch` and `arrival` as **3-D unit vectors**, bounce counts, `valid`, `sound_speed` | The interface already carries 3-D directions, so a 3-D provider fills the same arrays. The synthesis kernels (`core.synthesize_into`, host and CUDA) take apparent positions and amplitudes and never see the lookup. The live chains (`live.py`, `libs/live`) read records, not providers | `TimeSpreadProvider` and the Lambert weighting in `two_way_paths` take a boundary's grazing angle from the arrival's z component, which assumes a horizontal boundary |
| Bellhop3D oracle | `tests/python/bellhop3d_oracle.py`, `test_seamount_3d.py` | `bellhop3d.exe` (Acoustics Toolbox 8b4682b, `tools/build_oracles.sh`; `GPUSONAR_BELLHOP3D`) in run type `I` (incoherent TL), Gaussian beams `B`, modes `2` (N × 2-D) and `3` (3-D), fan 121 × 361 (elevation ±30°, azimuth ±60°), bed on a 25 m grid of 361 × 401 nodes | Fixes the size of the 3-D effect: 3-D minus N × 2-D is +3.0 to +3.6 dB over the flanks and −1.1 to −1.2 dB beside them (D-149) | `test_our_n_by_2d_tier_over_the_seamount` holds the gap as a **characterisation**: ours must be more than 2.0 dB too loud over the flanks. Our tier against Bellhop3D N × 2-D: `abs(...) < 3.5` dB |

**The measured gap** (D-149; test docstring, converged beams [test beams]). This is our tier
against Bellhop3D 3-D, incoherent TL averaged over 3.5 to 6.5 km, source and receiver 50 m deep:

| Bearing | Where | Ours minus 3-D |
| --- | --- | --- |
| −4° and 16° | over the flanks | −3.8 dB [−4.5] (ours too loud) |
| 6° | over the summit | −2.2 dB [−3.4] |
| −20° | beside | 0.0 dB [−0.5] |
| 32° | beside | −0.7 dB [−1.2] |
| all, 0.2 to 1.8 km | before the seamount | 3-D and N × 2-D within 0.07 dB |

**`sonar-6c8`**: "ours is 1.0 to 2.5 dB louder than Bellhop3D's N×2-D on the seamount
bearings, up to 1.0 dB over a level 200 m bed at 7 km." Investigated 2026-11-26 (package 1 of
this thread); what follows is measured here (ours) unless a source is named, with the commands in
section 2.1.

**Answer.** Most of it is the oracle, not our runs. Over a level bed none of it is ours.

1. **Over a level bed Bellhop 2-D and Bellhop3D's N × 2-D mode are the same program.** At the
   same beam type and count they agree to 0.001 dB at every range; Bellhop3D's 3-D mode agrees
   with both to 0.003 dB beyond 2 km. The 1.0 dB at 7 km was the **oracle's beam count**: Gaussian
   beams (`'B'`) with 121 elevation beams over ±30° are not converged. Against the image-source
   closed form they are 1.12 dB too quiet at 7 km (0.67 at 5 km); 241 beams 0.26 dB, 481 beams
   0.00 dB. Hat beams (`'G'`, Bellhop's default and what our provider uses) are within 0.08 dB at
   121 and converged at 241. The 3-D mode behaves the same in elevation (121 × 121: 1.14 dB at
   7 km; 481 × 61: 0.04 dB); 21 to 61 azimuth beams over ±10° are enough on a level bed.
2. **Our provider is right over a level bed.** Its lookup (the seamount test's 100 m × 20 m grid,
   ±30°, default arrival selection) summed in power is within 0.19 dB of the image sources from 2
   to 7 km (mean +0.03 dB), 0.13 dB at 1 km. Nearer than 0.5 km the ±30° fan the test chose leaves
   out steep bottom paths (1.87 dB at 100 m); the provider's default fan is ±80°.
3. **KRAKEN is not a reference at the source's depth.** Its incoherent mode sum with the receiver
   at the source depth (50 m) is 1.68 to 1.75 dB louder than the image sources and every Bellhop
   run from 3 km; 20 m off the source depth it agrees with the image sources within 0.07 dB. The
   cause (ours): an incoherent mode sum weights each mode by φ(z_s)²φ(z_r)², whose mean over the
   modes at z_r = z_s is 3/8 against (1/2)² = 1/4 off it: 3/2, 1.76 dB. A range-averaged coherent
   field (PyRAM) shows the same 1.5 dB at z_r = z_s. `test_kraken_crosscheck.py` has z_r ≠ z_s and
   is not affected.
4. **Over relief, Bellhop's arrivals output depends on how the bed is written; its TL output does
   not.** On the bearings beside the seamount, Bellhop's own incoherent TL run (`'I'`) gives the
   same range average within 0.06 dB for hat or Gaussian beams from 481 up and the automatic
   count, and within 0.01 dB for the profile written `'L'` every 5, 25 or 100 m, or `'C'`. The arrivals (`'A'`, what the provider reads) summed in power
   are 1.2 to 1.7 dB louder than that TL run on the gently sloping bearings beside the seamount
   (−20 and 32°) with the provider's `'L'` bed, at 50, 90 and 130 m, and within 0.06 dB with `'C'`
   (Bellhop's curvilinear bed, same points). A parabolic equation (PyRAM 1.3.0, the inputs of
   `test_ram_crosscheck.py`) on the 32° profile sides with the TL run: within 0.07 to 0.29 dB at
   the three depths, the arrivals 1.6 to 1.7 dB louder. Bellhop's coherent field over `'L'` is
   also 1.9 to 2.5 dB louder than the PE there, over `'C'` within 0.05 to 0.5 dB; over the level
   bed the coherent field and the PE agree within 0.3 dB off the source depth. Over the steep flanks and
   summit (−4, 6, 16°: slopes to 12°) `'C'` does not bring the arrivals onto the TL run (they
   are then 0.1 to 0.8 dB quieter); the PE is not a referee there (one-way, stair-stepped bed;
   `test_ram_crosscheck.py` stays at 0.8°). The beam count matters only to the arrivals: arrivals
   minus TL run on the 32° bearing at 90 m over `'L'` is +0.88, −0.22, −0.95, −1.28, −1.46, −1.50 dB
   with 481, 1001, 2001, 4001, automatic (about 10,400) and 20,001 beams; over `'C'` +1.15, +0.58,
   +0.24, +0.08, +0.05, +0.04 dB; over the level bed +0.05 and +0.03 dB at 4001 and 20,001. A
   likely mechanism (ours, not traced in the source): a piecewise-linear bed's normal jumps at
   every node, the reflected beams fold there, and Bellhop's arrival merge (`ArrMod.f90`
   `AddArr`, opened: a contribution joins the previous arrival when its phase and its delay times
   ω are within 0.05 rad) adds folded contributions by amplitude.
5. **The floor, cap and lookup grid of the provider explain at most 0.10 dB** on the seamount
   bearings (provider against the same Bellhop arrivals at one receiver: −0.02 to +0.10 dB).

**How much each candidate explains on the seamount** (our tier minus Bellhop3D N × 2-D with the
test's beams, incoherent TL averaged over 3.5 to 6.5 km, 50 m; each column is one step, from the
oracle as the test runs it to our provider, and the steps add up to the total):

| Bearing | Total | 1. Oracle beams (B 121 → B 1921) | 2. Bellhop3D `'R'` 25 m grid → Bellhop 2-D TL run on our `'L'` profile | 3. TL run → arrivals summed in power | 4. Arrivals → provider (lookup, floor, cap) |
| --- | --- | --- | --- | --- | --- |
| −20° (beside) | −1.67 | −0.47 | −0.14 | −1.16 | +0.10 |
| −4° (flank) | −1.11 | −0.57 | −0.06 | −0.47 | −0.02 |
| 6° (summit) | −2.45 | −1.23 | −1.15 | −0.15 | +0.08 |
| 16° (flank) | −1.00 | −0.62 | −0.04 | −0.36 | +0.03 |
| 32° (beside) | −2.34 | −0.66 | +0.02 | −1.72 | +0.03 |

Negative: ours louder. Bellhop3D N × 2-D at 481 or 1921 Gaussian or 481 hat beams agree within
0.15 dB of each other on every bearing, so column 1 is the oracle's non-convergence. Column 2 is
small except over the summit, where the receivers are in the summit's shadow: there Bellhop3D's
own bed grid matters (its N × 2-D range average moves from 71.59 to 70.76 and 70.89 dB with a
25, 12.5 and 6.25 m grid; Bellhop 2-D's TL run on the `'L'` profile gives 69.99 to 70.72 dB by
beam type and count), so about 0.8 dB of the 1.15 is the oracle's 25 m grid and the rest is the
scatter of a shadow zone. Column 3 is item 4 above. Column 4 is the provider.

**So, against the questions of the issue:** (i) Bellhop3D's N × 2-D mode is not the cause: it
equals Bellhop 2-D. (ii) Non-convergence explains the level-bed 1 dB entirely and 0.5 to 1.2 dB
on the seamount: in **the oracle** (Bellhop3D, Gaussian beams, 121 in elevation), which needs 481
(seconds in N × 2-D mode; 3-D timings in section 2.1). Our provider's automatic count (about
10,400 over ±30° to 6.6 km) is converged for TL. (iii) Our provider's floor, cap and lookup: at
most 0.10 dB. What our provider does that matters is to read Bellhop's **arrivals** over a bed
written `'L'`: 0.4 to 1.7 dB over relief, none over a level bed. (iv) Unexplained: over the summit
about 0.3 dB (shadow-zone scatter), elsewhere under 0.15 dB.

**What this means for the shipped records.** 19 of the 22 `eigenray` scenarios have a level bed
(rings about the sonar, no `bathymetry`); for them the arrivals and Bellhop's own TL run agree at
the target within 0.14 dB (table below), and the provider is within 0.19 dB of the image sources
on the case above: nothing to change. The three `fjord_*` scenarios have a bathymetry profile
written `'L'` (100 m nodes, slopes to about 9°). There, writing the same points as `'C'` moves the
target's one-way loss by +5.1, +0.2 and +3.6 dB, but `'C'` does not make the arrivals agree with
the TL run either (they are then 1.8, 0.4 and 3.6 dB quieter than it), and on a coarse, steep
profile `'C'` is a different bed, not a smoother copy of the same one. **No change is
recommended until a referee exists for the fjord profile** (owner question 9, section 6).
`reverb_relief.py` (`along_profile`) also writes `'L'`; no shipped scenario uses it.

One-way incoherent loss at each scenario's target (dB re 1 m; the grid node the provider looks
the target up at; `tools/eigenray_tl_check.py`, 2 min for all 22). "Shipped" is the provider as
the scenario uses it; "all arrivals" keeps every arrival (no floor, cap or `max_arrivals`); "TL run"
is Bellhop's incoherent TL at the same receiver; the last three columns write the fjord beds `'C'`.
These are point values (one receiver), not range averages:

| Scenario | Range m | Depth m | Shipped | All arrivals | TL run | `'C'` shipped | `'C'` all | `'C'` TL run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fjord_static_trial_2row | 2600 | 60 | 58.29 | 55.77 | 55.57 | 63.38 | 58.82 | 57.04 |
| fjord_static_trial_3row | 150 | 18 | 40.03 | 40.03 | 39.71 | 40.20 | 40.19 | 39.80 |
| fjord_testauv_trial_1row | 500 | 15 | 50.63 | 50.48 | 49.38 | 54.18 | 54.09 | 50.47 |
| flank_patrol | 2961 | 60 | 58.26 | 58.26 | 58.26 | | | |
| flank_target_10km | 10000 | 50 | 66.06 | 66.06 | 66.06 | | | |
| flank_target_1km | 1000 | 50 | 52.09 | 52.09 | 52.09 | | | |
| flank_target_5km | 5000 | 50 | 61.50 | 61.50 | 61.50 | | | |
| popoto_12km (also `_2row`, `_3row`) | 12027 | 4 | 67.23 | 67.17 | 67.17 | | | |
| popoto_12km_two_ships | 12027 | 4 | 67.23 | 67.17 | 67.17 | | | |
| popoto_env_deep_silt_1000m | 10032 | 4 | 72.62 | 72.62 | 72.62 | | | |
| popoto_env_gravel_30m | 3105 | 3 | 53.51 | 52.87 | 52.76 | | | |
| popoto_env_summer_mud_60m | 4079 | 4 | 88.97 | 88.85 | 88.73 | | | |
| popoto_env_winter_sand_200m | 8040 | 4 | 64.19 | 64.19 | 64.05 | | | |
| popoto_far_mover | 6053 | 4 | 62.94 | 62.74 | 62.73 | | | |
| popoto_trial_1000m | 1000 | 15 | 48.52 | 48.52 | 48.52 | | | |
| popoto_trial_100m | 100 | 15 | 35.69 | 35.69 | 35.68 | | | |
| popoto_trial_2000m | 2000 | 15 | 52.92 | 52.91 | 52.91 | | | |
| ssp_front_static_trial_2row | 2600 | 60 | 59.16 | 57.97 | 57.78 | | | |
| ssp_front_testauv_trial_1row | 500 | 15 | 46.50 | 45.89 | 44.59 | | | |
| ssp_thermocline_static_trial_3row | 2600 | 60 | 58.72 | 57.81 | 57.06 | | | |

Read with care: "shipped" minus "all arrivals" is the stated selection (`max_arrivals: 12` in the
fjord and `ssp_*` scenarios: up to 2.5 dB at `fjord_static_trial_2row`; the cap of 64 at
`popoto_env_gravel_30m`: 0.64 dB, the 14 % of `sonar-9b8`). It is a scenario choice of D-105, not
part of `sonar-6c8`, and is not proposed for change here. The three `ssp_*` scenarios (water that
changes with range, no bathymetry) show arrivals 0.2 to 1.3 dB quieter than the TL run at their
targets: one receiver each, not investigated further (follow-up).

**What the later packages should take from this.**

- **The oracle's fan.** Bellhop3D runs used as references need at least 481 elevation beams with
  Gaussian beams (or 241 hat beams) over ±30°; the seamount test's 121 is 0.5 to 1.2 dB short of
  converged. Its bed grid needs 12.5 m or finer where receivers sit in a summit's shadow. The
  "1.1 dB between 121 × 361 and 241 × 721" quoted in section 1 item 7 and in the acceptance bound
  of section 5 is mostly this elevation count.
  With the elevation fan raised (seamount, 361 azimuth beams kept, 3.5 to 6.5 km, 50 m), the
  3-D effect stands: 3-D minus N × 2-D is +3.49 and +3.60 dB over the flanks (−4, 16°), +0.81
  over the summit, −1.10 and −1.12 beside (−20, 32°) with 481 Gaussian beams (208 s for both
  modes); +3.23, +3.95, +0.29, −1.18, −1.14 with 241 hat beams (105 s); the test's 121 gives
  +3.35, +3.51, +0.90, −1.11, −1.16 (66 s). Ours minus the 481-beam 3-D field: −3.99, −4.01 dB
  over the flanks, −2.17 over the summit, +0.01 and −0.62 beside; ours minus its N × 2-D: −0.51,
  −0.41, −1.36, −1.09, −1.73 (the last two are item 4). The azimuth count was not varied.
- **Package 2 (the cross-slope warning)** is not affected: it reads the bed, not the field.
- **An opt-in Bellhop3D provider (A):** compare it with our tier through Bellhop's TL output or
  with a `'C'` bed for smooth relief, never with arrivals summed over an `'L'` bed; over a level
  bed the two must agree within 0.2 dB at converged fans (they agree to 0.003 dB at equal beams).
  Its own arrivals need the same check against its own TL run before any merge is trusted: the
  arrival output is where the 2-D program goes wrong over relief, and Bellhop3D's 3-D arrivals
  were not checked (section 1 item 3).
- **The acceptance bound of section 5** loses the `sonar-6c8` allowance over a level bed. Over
  relief, "ours" must be produced with a bed representation whose arrivals agree with Bellhop's TL
  run (`'C'` for a smooth sampled bed such as the seamount), or the bound must carry the 1.7 dB of
  item 4.
- **The seamount test** (`test_our_n_by_2d_tier_over_the_seamount`, bound `abs(...) < 3.5` dB
  against N × 2-D) is explained, and could be tightened by the owner once the oracle's fan is
  raised; this package does not change it.

### 2.1 Commands and run times for `sonar-6c8`

All on 2026-11-26, Acoustics Toolbox 8b4682b built by `tools/build_oracles.sh` (the session's
scratch directory), PyRAM 1.3.0, on the shared 4-core machine. Scripts in the scratch directory;
all output files were deleted after reading, none larger than 2 MB.

- The pinned results: `tests/python/test_bellhop_nx2d_agreement.py` (six tests, 51 s in all;
  each skips without its oracle). Environment: `GPUSONAR_BELLHOP`, `GPUSONAR_BELLHOP3D`,
  `GPUSONAR_KRAKEN_DIR`, and `pyram` for the PE test.
- Level bed (200 m, 1500 m/s over 1700 m/s, 1.9 g/cm³, 0.5 dB/λ; 8 kHz; source and receiver
  50 m; 0.1 to 7 km every 100 m): `write_env` with run types `'I'`, `'IG'`, `'IB'` at 61 to 3841
  beams (0.0 to 1.5 s each); `run_bellhop3d` modes `2` (one bearing, 0.2 to 0.3 s) and `3` (121 ×
  61 over ±10°: 3 s; 481 × 61: 12 s); `run_kraken` incoherent at 50, 70 and 120 m (2 s);
  `ImageSourceProvider(..., max_order=400, engine="numpy")` summed in power; the provider as in
  `test_our_n_by_2d_tier_over_the_seamount`.
- Seamount (the case of `test_seamount_3d.py`): on −20, −4, 6, 16, 32° the provider through
  `along_profile` (7 to 9 s per bearing), `run_bellhop` arrivals at one receiver depth, Bellhop
  TL runs `'I'`, `'IG'` 481, `'IB'` 121, 481, 1921, Bellhop3D mode `2` with B 121, 481, 1921 and
  G 481 (one bearing each): 2 min 30 s for all. The bed-representation runs (`'L'` every 5, 25,
  100 m; `'C'` every 25 m; receivers every 10 m from 3.5 to 6.5 km at 50, 90, 130 m): 10 to 90 s
  per bed. PyRAM on 32°, 6° and −4° and the level bed: 13 to 16 s each.
- The scenario table: `GPUSONAR_BELLHOP=... tools/eigenray_tl_check.py` (1 min 58 s, four
  receivers per scenario, files of kilobytes).

Related open issue: `sonar-pup` (water that changes with range per bearing, a front across the
map). It is horizontal refraction by the water, not the bed, and it is small (section 3).

## 3. Where 3-D matters for this project

**What is unknown:** no scenario family has a measured site, so the "expected size" column is
for the scenario as shipped (where 3-D is zero by construction unless noted) and, separately,
for the same sonar over relief that might exist in the owner's waters.

**The formula (ours).** A ray in the vertical plane of its bearing, direction
(cos g, 0, sin g) with x along the bearing and z down, hits a bed whose depth rises across the
bearing with slope tan α. The normal into the water is (0, sin α, −cos α). On reflection the
direction becomes (cos g, sin g sin 2α, −sin g cos 2α). The ray leaves its plane at the azimuth
φ = atan(tan g · sin 2α), about 2gα for small angles. Over a further distance L it lands about
φL across the bearing. Cumulative over n bounces on a uniform cross-slope (same sign each time),
the displacement at range R is about Σ φ_i (R − r_i). Values of φ in degrees:

| Cross-slope α | g = 2° | 5° | 10° | 20° | 30° | 45° |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5° | 0.03 | 0.09 | 0.18 | 0.36 | 0.58 | 1.0 |
| 1° | 0.07 | 0.17 | 0.35 | 0.73 | 1.15 | 2.0 |
| 2° | 0.14 | 0.35 | 0.71 | 1.45 | 2.3 | 4.0 |
| 5° | 0.35 | 0.87 | 1.75 | 3.6 | 5.7 | 9.9 |
| 12° (the seamount's steepest) | 0.81 | 2.0 | 4.1 | 8.4 | 13.2 | 22.1 |

Horizontal refraction by the water (ours): across a front with sound speeds c and c + Δc, a ray
at angle ψ from the front's normal turns by about tan ψ · Δc / c. For Δc = 5 m/s at ψ = 45° that
is 0.19°; 6 km later it lies about 20 m off its bearing. Sustained curvature in a horizontal
gradient G has a radius of c / G. For G = 0.005 /s that is 300 km, which gives 0.06° over 300 m.

Beams and resolution cells (ours, λ / aperture at band centre):

| Sonar | Beam | Cross-range cell |
| --- | --- | --- |
| `fls_120k` | 0.93° | 4.9 m at 300 m |
| `flank_32` at 8 kHz | 3.6° | 312 m at 5 km |
| `popoto_long_range` at 5 kHz | 3.6° | 750 m at 12 km |
| `sas_auv`, `sas_minsas120` | ~2 cm along track | ~2 cm |

| Family (scenarios) | Tier | Band | Ranges | Depth, geometry | Relief in the scenario | 3-D effect as shipped | Over relief in real water | Basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FLS sea: `fls_120k_transit`, `_170k_transit`, `_330k_transit`, `_170k_traffic` | `image_source`, `max_order: 1` | 110–130, 160–180, 315–345 kHz | 20–450 m | 60 m, sonar 20 m deep | flat bed; a pinnacle of rocks as scatterers | none | Bounce paths off a bed tilted 5° across the bearing land 3.6 to 9.9° off (4 to 11 beams) at 15–58° grazing; at the level of a first-order bounce (the objects already have first-order bounce paths over the level bed; "no bounce paths to objects" was wrong, corrected by `sonar-0nsi`). Refraction: below 0.6° and 1.5 m vertical (2-D); horizontal 0.06° | formula above; geometry from the scenario files; refraction from c/G |
| FLS free-field: `fls_120k_cluttered_seabed`, `_rough_surface`, `seafloor_objects_fls`, `fls_glints_on_sediment`, `fls_highlight_target_aspect`, `harbour_fls_750k`, pool scenes (`pool_fls_*`, `pool_wires_l12_5d`) | `free_field` | 110 kHz to 6.25 MHz | 0.4–300 m | 40–60 m, harbour and pool | quay wall, pool walls as scatterers | none (no multipath at all) | Wall multipath: ghost targets "in the fan" (`SCENARIOS.md`). An image source off the wall, exact in water of one speed | `SCENARIOS.md` "What no scenario has yet" |
| SAS: `wreck_sas`, `minefield_sas`, `stripmap_points`, `sas_minsas_*` (4), rail scenes (4) | `free_field` | 180–220 and 310–350 kHz | 38–68 m slant | 12–15 m altitude, 10–23° grazing; the surface is not modelled | flat seabed (a `seafloor` can take `bathymetry`, unused here) | none | Direct paths: relief is in the scatterer positions already (D-73, one description). Multipath off a sloping bank: an image off a tilted plane. Refraction: delay error below 1e-10 s over 50 m | formula; c/G |
| Buried: `buried_cylinder`, `buried_objects` | `sediment` | 10–30 kHz | a few m | flat interface | none | none | Not a propagation-3-D question (a refracted ray into sediment) | — |
| Flank short, SST and staged trials: `flank_sst_*`, `sst_*`, `popoto_elevation_*`, `popoto_face_2row_shallow`, `popoto_testauv_*`, `popoto_static_*` (all at `image_source` or `free_field`) | `image_source` or `free_field` | 6–10, 20–35, 4–6, 38–48 kHz | 0.04–3.75 km | 20–1275 m, flat | none | none | Over relief they would need the eigenray tier first; N × 2-D is within 0.07 dB to 1.8 km over the seamount | D-149 near-range agreement |
| Flank eigenray: `flank_target_1km`, `_5km`, `_10km`, `flank_patrol` | `eigenray` | 6–10 kHz | 0.6–10.4 km | 100 m, sonar 30 m deep, coastal sand | flat | **none** (axisymmetric) | **0 to 4 dB** on bearings over relief beyond about 2 km: 3.0–3.6 dB at 3.5–6.5 km over 12° flanks at 8 kHz (measured, D-149). For shelf slopes of 1° or less: φ of 0.17° a bounce at 5° grazing; size in dB not measured | measured (seamount); formula otherwise |
| Popoto 27.5 kHz: `popoto_trial_100m`, `_1000m`, `_2000m`, `popoto_far_mover` | `eigenray` | 20–35 kHz | 0.02–6.8 km | 30 m and 100 m, flat | flat | **none** | As above; not measured at 27.5 kHz | formula |
| Popoto 12 km and environments: `popoto_12km`, `_2row`, `_3row`, `_two_ships`, `popoto_env_*` (4) | `eigenray` | 4–6 kHz | 2.3–13.6 km | 30–1000 m, flat (`slope_1000m_silt_profile` is flat despite its name: `configs/builds/environments/illustrative.yaml`) | flat | **none** | The largest exposure: many bottom bounces in 30–200 m water over 12 km. On a uniform 1° cross-slope at 5° grazing in 100 m water (about one bounce per 2.3 km): about 0.9° of turning and about 110 m at 12 km, against a 750 m cell. Over a seamount-like feature: 3 to 4 dB (8 kHz measured; 5 kHz not) | formula; measured at 8 kHz |
| Staged trials with range profiles: `fjord_*` (3), `ssp_*` (3) | `eigenray` | 38–48 kHz | 0.09–2.75 km | 30–100 m; fjord sills as one profile on every bearing | range-only profile | **none** (axisymmetric by construction) | A real fjord has side walls and sills that cross the bearing obliquely: steep, so 3-D is plausible (owner question 2). Walls as walls are an image-source question | formula |

## 4. Options

All options keep every shipped record unchanged unless the owner approves otherwise (new
behaviour opt-in, `docs/HANDOFF.md` section 2).

### A. Bellhop3D as an opt-in external provider (recommended, gated on the owner)

**What it is.** A `Bellhop3DProvider(EigenrayProvider)` in a new module
`python/gpusonar/sim/bellhop3d.py`. It runs `bellhop3d.exe` exactly as `bellhop.py` runs Bellhop:
it writes `.env` and a 3-D `.bty` (the oracle's writer moves into the library), runs the program
in a temporary directory, reads the arrivals, and keeps nothing of Bellhop3D in the library.

**Licence.** GPL-3, a separate process, never linked; the same position as Bellhop today
(`OPEN_SOURCE.md` rules 2 and 3). It needs the owner's explicit yes, because it moves Bellhop3D
from oracle to provider. `tools/build_oracles.sh` already builds it; a failure there only warns.

**How it plugs in.**

- **Lookup key.** One run per source position, rounded (x, y to a stated track tolerance; z to
  0.01 m as `arrivals_for` does). Each run gives arrivals at (bearing θ, depth z, range r) from
  that source. Bellhop3D's 3-D arrivals file carries, per arrival: amplitude, phase, delay,
  source declination and azimuth, receiver declination and azimuth, and surface and bottom bounce
  counts. This was confirmed by opening `Bellhop/ArrMod.f90`, `WriteArrivalsASCII3D`, at 8b4682b,
  read for the format only, as D-110 did for Bellhop, and in the header of a file written here.
- **Grid.** The provider asks only for what the scene needs: an irregular receiver list
  (Bellhop3D's `'I'` grid option) at the scatterer positions, or a sector of bearings with depths
  limited to the scatterers' and the bed's.
- **`eigenrays(source, points)`.** θ, r and z per point; the nearest node (or interpolation in θ).
  `launch` and `arrival` become true 3-D vectors from the two azimuths and declinations, where
  today they lie in the bearing's plane. The first-order delay correction uses the 3-D arrival
  vector. The energy rule of D-105 is applied after merging.
- **Merging (new, required).** Arrivals of one receiver that share bounce counts and lie within a
  delay tolerance (a fraction of the band's resolution) and an angle tolerance are summed into one
  eigenray: coherent amplitude sum, energy-weighted delay and angles. The tolerances are ours and
  must be tested.
- **Echo synthesis.** `two_way_paths` is unchanged: NumPy and C++ pairing take any `EigenraySet`,
  and the CUDA and host synthesis kernels see only apparent positions. No kernel, binding or
  live-chain change is needed. Engine parity (NumPy against C++ pairing) holds as it does for
  Bellhop today.
- **Reverberation over relief.** `reverb_relief._leg` already takes a `relief_aware` provider as
  one that traced the relief. The 3-D provider is `relief_aware` and has no `along_profile`, so
  the per-sector loop and the `separation > 1` `NotImplementedError` fall away, and bistatic
  becomes possible. Shadowing is then the provider's.
- **Config.** A new tier name (`fidelity: eigenray3d`, so `eigenray` stays byte-identical), and
  one `bathymetry` description (`sim/bathymetry.py`, e.g. `{seamount: ...}` or a grid) shared by
  the provider, the scene `seafloor` and the reverberation items (D-73's "one description").
  `make_provider` gains a branch. `docs/CONFIGURATION.md` gets the keys.

**Cost (measured here, one run each, load average 3.6 to 5.8 on a shared 4-core machine).** Seamount,
8 kHz, 121 × 361 Gaussian beams, 29 bearings × 65 ranges:

| Run | Receivers | Wall time | Output |
| --- | --- | --- | --- |
| Bellhop3D TL, mode 3 | 1 depth | 52.6 s (44.5 s user) | — |
| Bellhop3D TL, mode 2 | 1 depth | 4.4 s | — |
| Bellhop3D arrivals, mode 3 | 1 depth (1,885) | 61.8 s | 115.5 MB ASCII, 681,140 arrival lines, at most 1,207 per receiver |
| Bellhop3D arrivals, mode 3 | 10 depths (18,850) | 103.4 s (its own CPU time 88.7 s) | 1,099.7 MB, at most 1,272 per receiver |
| Bellhop3D arrivals, mode 2 | 1 depth | killed for memory after 9.7 s (exit −9; memory is shared with other agents; not repeated) | — |
| Bellhop 2-D arrivals, one bearing | 10 depths × 65 ranges | 5.2 s (automatic 10,400 beams) | 0.93 MB, at most 106 per receiver |

Estimates from these numbers (not measured):

- A 3-D run per source position costs about a minute at this fan. N × 2-D with Bellhop's
  automatic beams over the same 29 bearings costs about 2.5 minutes. **Tracing is not the
  problem; output and arrival counts are.**
- At about 170 bytes an ASCII arrival line and about 360 arrivals a receiver, a dense `popoto_12km`
  sector lookup (about 1.0 M receivers) is about 60 GB. An irregular grid at, say, 10^4
  scatterer and reverberation-cell positions is about 0.6 GB ASCII, and less binary.
- Memory in Bellhop3D grows with receivers × the largest arrival count (it stores arrivals per
  receiver: `Arr3D(itheta, id, ir, ...)` in `ArrMod.f90`). Not measured.
- With a moving platform, multiply by the pings or by track segments. `popoto_12km` has six
  pings 26 m apart (1.5 m/s × 17.3 s).
- Faster tracer: bellhopcxx3d (GPL-3, multithreaded, CUDA build possible; `OPEN_SOURCE.md` 2c
  and 3.6). Its 3-D accuracy against the Fortran is not checked here, and its README notes an open
  issue with 3-D runs in complex environments (via `OPEN_SOURCE.md` 3.6).

**Accuracy to expect.** The 3-D field is what Bellhop3D gives. Its beam convergence moves range
averages by up to 1.1 dB between 121 × 361 and 241 × 721 beams (measured, D-149). The plumbing
(merging, the floor, nearest-node lookup) adds an error to be measured in P3. Over a level bed the
tier would equal our Bellhop tier: the two programs agree to 0.001 dB at the same beams, and the
1.0 dB first reported was the oracle's beam count (`sonar-6c8`, section 2).

**Risks.**

- Arrival merging done wrong silently drops energy (section 1, item 3).
- The irregular-grid option is little used.
- Per-position runs make moving-platform scenes slow.
- Conventions: Bellhop3D bearings are "math convention with 0 degrees pointing along the
  x-axis" (`doc/bellhop3d.htm`, confirmed) in our NED x/y. The oracle test passed with x = north
  and θ from north towards east, but a provider needs its own mirror test.
- The grazing-angle uses in `two_way_paths` (Lambert) and `TimeSpreadProvider` assume a horizontal
  boundary and must take the facet normal over relief.
- The Bellhop3D bed is single-valued in (x, y): no walls, overhangs or caves.

**Tests that would hold it.** P2 to P5 below: reader fixtures; level-bed agreement with Bellhop3D
mode 2 and with our tier (within 0.2 dB at converged fans, section 2); a mirror test of the
seamount; unit 3-D vectors with out-of-plane azimuth on the flanks; the acceptance bound (section 5).

### B. Stay N × 2-D, with a warning or a first-order correction (ours)

**What it is.** In `reverb_relief` and anywhere a `bathymetry` reaches the eigenray tier: compute
the cross-bearing slope ∂D/∂n along each bearing's profile and the grazing angles of the paths
that hit the bed there. Warn, and record on the provider like `truncation`, where
φ = atan(tan g · sin 2α) exceeds a stated fraction of the receive beam, or where |α| exceeds a
threshold. The threshold would be ours, for example 2°.

- **What it can capture:** where the N × 2-D answer is suspect, with the size of the turning.
  It costs nothing at run time, changes no record and needs no licence decision.
- **What it cannot:** the level. The energy turned out of one bearing lands on others (+3 dB
  over the flanks, −1 dB beside them, measured). Redistributing it needs a 3-D transport: vertical
  modes with horizontal rays, or a 3-D tracer. A per-bearing level correction from φ alone would
  be a fit to one seamount, and we would not trust it elsewhere.

**Plug-in, cost, tests.** `bathymetry.Bathymetry.gradient` already exists; size S. Tests: the
seamount warns on the flank bearings and not beside; a level bed and the axisymmetric fjord
profile never warn.

### C. A 3-D Gaussian-beam tracer of our own

**What it is.** A clean-room ray and beam tracer in C++/CUDA. Licence: ours. **Size: large.**
Bellhop3D's Fortran is about 6,700 lines in the objects that build `bellhop3d.exe` (counted here,
`Bellhop/` only). Beam influence in 3-D, boundary interpolation over a grid, caustics and
convergence are where the work is. It must be clean-room: `OPEN_SOURCE.md` rule 2 forbids porting
Bellhop or bellhopcxx line by line. It would then need Bellhop3D as its oracle anyway.

**Why not.** All of that buys a dependency-free 3-D tier for a need the owner has not stated. It
duplicates a program we already run as a process. Its accuracy could not exceed the oracle's,
which is itself not independently re-validated.

### D. 3-D parabolic-equation or coupled-mode codes

**What was found (2026-11-26, generic web search, pages opened):**

- CAMBALA (github.com/Nauchnik/CAMBALA): "sound propagation in 3D-varying shallow water
  waveguides", coupled normal modes plus inversion. **No licence file or statement seen.**
- SMPE (github.com/tuhouwang/SMPE): a MATLAB wide-angle PE for laterally inhomogeneous oceans,
  apparently range and depth only. **No licence seen.**
- A Cartesian 3-D PE code from WHOI, described in a report on DTIC: MATLAB, licence not seen
  (search summary only, not opened).

None has a licence that allows use. Code without a licence cannot be used, even as an oracle,
without the authors' permission. PyRAM (BSD) is 2-D. A PE is also a field method at a frequency:
it gives TL, not the per-path delays, angles and bounce counts `EigenraySet` needs. **Not
suitable today.** Recheck if a permissively licensed 3-D PE appears.

### E. Do nothing; keep the characterisation test

Correct for every shipped scenario, since none has cross-bearing relief. It costs nothing. Its
risk is that a user adds a `bathymetry` with steep relief to a long-range scenario. They then get
levels 3 to 4 dB too high on bearings over it, without a warning. Option B's warning closes that
risk.

### F. (For FLS and SAS, not this issue) Image sources off walls and tilted planes

Already half there: `ImageSourceProvider(walls=[{axis, position, coefficient}])`, axis-aligned,
in NumPy and C++ (`libs/sim` `paths.cpp`). No shipped scene uses it.

- **The extension:** arbitrary planes, meaning a normal and a point, with a reflection
  coefficient at the local grazing angle. Exact in water of one speed. It is the right mechanism
  for quay walls, pool walls, canyon sides and a sloping bank under a SAS or FLS.
- **Licence:** ours. **Size:** M. It would need its own issue (see "Follow-ups").
- **Done (2026-11-26, `sonar-0nsi`), opt-in.** `propagation.bottom_slope: [dz/dx, dz/dy]` tilts the
  bottom and a wall may be `{point, normal, coefficient[, bounds]}`; either turns on one image
  construction over every plane with validity checks (specular point on the face, every vertex in the
  water, no leg through a bounded wall), Rayleigh at each bounce's own grazing angle to the tilted bed,
  `wall_order` for wall bounces (`docs/PHYSICS_NOTES.md` section 30; `CONFIGURATION.md`). NumPy only: the
  C++ provider keeps axis walls and a flat bottom. New scenes, nothing shipped changed:
  `fls_120k_quay_multipath`, `pool_fls_p4_2v_walls`, `sas_auv_multipath` and `sas_auv_multipath_free`
  (`SCENARIOS.md`). Not done: a wall's coefficient from a material (each is a constant, a placeholder);
  roughness (time spreading) on these paths; reverberation over a tilted bed; the arrival direction of
  the old `{axis, position}` walls, which is wrong in the horizontal (follow-up).

### Recommendation

**Do E + B now, with P0 (`sonar-6c8`) first; do A only if the owner says the waters have steep
relief and allows Bellhop3D as a provider.**

- A is the only option that gives the level right over relief, at a known accuracy (the
  oracle's), for about a minute of tracing per source position, with the existing interface
  carrying it unchanged.
- C costs far more for the same answer.
- D has nothing usable.
- B alone cannot get the level.
- E alone leaves a silent 3 to 4 dB error for anyone who adds relief.

## 5. Work plan

Each package is done and tested alone. Tests go first: each new test fails before its change. The
owner approves each package and any moved default before merging (`docs/HANDOFF.md` section 2).
Sizes: S is up to 2 days, M up to a week, L up to two weeks (ours).

| # | Package | Goal | Files | Tests first | Default that could move (owner first) | Oracle or GPU | Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | **`sonar-6c8`** (done 2026-11-26 as an investigation: section 2) | Explain the 1.0 to 2.5 dB between our 2-D runs and Bellhop3D mode 2 (1.0 dB on a level bed). Outcome: the oracle's elevation beam count (Gaussian, 121; needs 481) and, over relief, Bellhop's arrivals over our `'L'` bed; the provider's floor, cap and lookup at most 0.1 dB | `tests/python/test_bellhop_nx2d_agreement.py`, `tools/eigenray_tl_check.py` (opt-in diagnostic); no library change | Six characterisations (section 2.1): Bellhop 2-D and Bellhop3D N × 2-D within 0.05 dB at equal beams; the provider within 0.3 dB of the image sources from 2 km; arrivals over `'L'` louder than the TL run by more than 1 dB, over `'C'` within 0.2 dB | No: nothing shipped moved. A change to the fjord bed would move three scenarios (owner question 9) | Bellhop, Bellhop3D, KRAKEN, PyRAM on the CPU, 51 s for the tests | M |
| P1 | Cross-slope warning (option B) (done 2026-11-26: `reverb_relief.cross_slope_report`, `bathymetry.cross_bearing_slope`, `out_of_plane_turn`; flags a bearing where φ at 10° grazing exceeds 1.8°, half the 3.6° long-range beam; on the seamount: −16 to +2° and +10 to +30°, not −20, +6, +32°; no shipped scenario reaches it: none has relief reverberation, and the fjord profile is rings about the sonar; `tests/python/test_cross_slope.py`, 5 tests, 0.1 s) | Say where N × 2-D is suspect | `sim/reverb_relief.py`, `sim/bathymetry.py` (a cross-bearing slope helper), `docs/CONFIGURATION.md` if a threshold key is added | Warns on the seamount flanks, not beside; never on a level bed or the fjord profile; records the largest φ | None (a warning only) | none | S |
| P2 | Reader and merger (option A starts here: owner questions 1 to 3 answered) (done 2026-11-26: `sim/bellhop3d.py` `read_arrivals3d` (ASCII `'A'` and binary `'a'`; 171 and 52 bytes an arrival, measured), `merge_arrivals`, `write_env3d`, `write_bty3d` (the oracle now imports them; the `.bty` it writes is byte-identical, the `.env` differs only in writing the first profile depth `0.0000` for `0.0`); `tests/python/test_bellhop3d_reader.py`, 5 tests, 27 s with Bellhop3D. What differs from the plan, measured: (1) **Gaussian beams' arrivals do not hold the field**: summed in power they are 8.7 to 11 dB below the run's own TL on a level bed (up to 931 contributions a receiver), so the merger and provider use hat beams (`'G'`, 241 elevation beams: converged, P0). (2) The merge joins contributions with the same bounce counts whose **source** angles are neighbouring beams (within 1.5 beam spacings, transitively) and adds magnitudes (the hat weights of a family sum to one); joining by delay and arrival angle instead failed (families split or merged). On a level bed the merged arrivals equal the TL run within 0.05 dB at 1 to 5 km, 129 contributions become at most 30 eigenrays. (3) **Over relief the merged arrivals do not hold the TL run's energy**: on the seamount (hat 241 × 361, 50 m, −20, −4, 16, 32° at 2, 4, 6 km) they differ by −3.9 to +2.5 dB at single receivers (a family's hat weights no longer sum to one where the bed focuses and spreads its beams). P3 therefore runs Bellhop3D's TL at the same receivers and scales each receiver's eigenrays to it) | Read Bellhop3D 3-D arrivals (ASCII, then binary `'a'`) and merge beam contributions into eigenrays | new `sim/bellhop3d.py` (reader); the `.bty` and `.env` writers move from `tests/python/bellhop3d_oracle.py` (the oracle then imports them) | A hand-written fixture file read back exactly. Merging keeps total energy to 1e-6 and reduces the seamount's counts (up to 1,272) to below the cap of 64 with less than 1 % energy left out (target, ours) | None | Bellhop3D, CPU, about a minute; mind disk (1.1 GB ASCII at 18,850 receivers) | M |
| P3 | `Bellhop3DProvider` (done 2026-11-26: `sim/bellhop3d.py`, tier `eigenray3d` in `make_provider` and `environment.apply_to_scenario`; hat beams 241 × 361 by default; runs only the point's nodes as a product bearings × depths × ranges, binary arrivals AND the TL run, eigenrays scaled to the TL (P2 finding 3); refuses a run whose arrivals would exceed `max_file_mb` (300) at 10 kB a receiver (measured 3.6 to 4.8 kB); `tests/python/test_bellhop3d_provider.py`, 6 tests, 47 s with Bellhop3D: level bed within 0.16 dB of the image sources and the 2-D tier, mirror within 0.83 dB and 0.6°, out-of-plane arrivals 5 to 15° on the flanks, node delay within 5 µs, NumPy and C++ pairing equal. Differs from the plan: no irregular grid (the product of the points' nodes is small for targets; the size check guards the rest); a per-receiver TL scaling the plan did not foresee) | `eigenrays(source, points)` with 3-D vectors, per-position cache, irregular grid, the D-105 rule and its warning | `sim/bellhop3d.py`, `sim/propagation.py` (`make_provider`: tier `eigenray3d`) | (a) Level bed: equals Bellhop3D mode 2 TL within 0.3 dB. (b) Mirror: the seamount mirrored about the source's bearing gives mirrored azimuths. (c) Unit `launch`/`arrival`; out-of-plane azimuth on the flanks above 0.5° (formula: 0.8–4° at 2–10° grazing over 12°). (d) Delay of the node correction within 20 µs, as the 2-D lookup | None (new tier) | Bellhop3D, CPU, minutes | L |
| P4 | Acceptance: the seamount bound (done 2026-11-26: `test_seamount_3d.py::test_eigenray3d_tier_over_the_seamount`, the provider at hat 241 × 181 against Bellhop3D's converged 3-D and N × 2-D fields (Gaussian 481 × 361), bearings −24, −4, 16, 36°, 0.5 to 6.5 km every 500 m, 50 m. Measured: ours − 3-D −0.01, −0.10, −1.17, +0.01 dB; ours − N × 2-D −0.45, +3.37, +2.24, −0.71 dB; 0.05 dB at 0.5 and 1 km. Bounds as planned: 1.5 dB; > 2.0 dB over the flanks (16°: 0.24 dB of margin); 0.5 dB at single near ranges. **Differs from the plan: it takes 8 minutes** (three Bellhop3D runs at converged fans), so it runs only with `GPUSONAR_SLOW_ORACLES=1`; its skip does not name an oracle variable, so `GPUSONAR_REQUIRE_ORACLES=1` does not turn it into a failure. The N × 2-D characterisation of our 2-D tier is unchanged) | Turn `test_our_n_by_2d_tier_over_the_seamount` into an agreement bound | `tests/python/test_seamount_3d.py` | See the bound below | None | Bellhop3D, CPU; the converged oracle at 241 × 721 is about 4 × 53 s for mode 3 (estimate) | S |
| P5 | Scenario wiring and one shared bed (done 2026-11-26: the top-level `bathymetry` key in `environment.apply_to_scenario`; `configs/scenarios/flank_seamount_3d.yaml`; `tests/python/test_eigenray3d_scenario.py`, 3 tests, 2 s, no oracle: the bed's distribution, no other shipped scenario using the new keys, both engines equal on the host for 3-D eigenrays from a stand-in. Measured: the scenario runs in 231 s on the CPU (four Bellhop3D runs of 57 s: two source positions, the projector 0.5 m below the array), arrival files of 472 and 576 bytes, TL scaling 0.66 dB. **Differs from the plan**: no reverberation in the new scenario and no reverberation acceptance test against point scatterers: over relief the 3-D provider would need a receiver per scattering cell, which the node product and the disk cap refuse (an irregular receiver grid is the way: not done); no comparison with the scenario at `eigenray` (that tier's targets see a flat bed, not N × 2-D)) | `bathymetry` (seamount or grid) reaches the provider, the scene seafloor and the reverberation from one key. A new illustrative scenario (for example the flank array over a seamount, with `unconfirmed` relief), as `sonar-6c8` also asks | `sim/simulator.py` (`run_scenario`), `sim/scene.py`, `sim/reverb_relief.py`, `configs/scenarios/<new>.yaml`, `docs/CONFIGURATION.md`, `docs/SCENARIOS.md` | The new scenario runs; its flank bearings are 2–4 dB quieter than the same scenario at `eigenray`. Reverberation over relief is bistatic with the 3-D provider, against point scatterers on the same bed within 2 dB (the existing D-73 criterion) | None for shipped scenarios. A new scenario adds a record and a gallery entry only if the owner wants it | Bellhop3D CPU; one full GPU-suite run to confirm nothing moved | M |
| P6 | Grazing over relief, cost record (done 2026-11-26 as far as it is opt-in. **Lambert over relief needed no change**: `two_way_paths` (NumPy and C++) already takes each scatterer's facet normal where a scene `seafloor` has a `bathymetry` (D-73), whatever the tier, and no shipped scenario has such a seafloor, so no record can move; nothing was applied and there is no before → after. `TimeSpreadProvider`: the bottom's spreading still takes bounce grazing angles from the launch angle (the arrivals do not say where a ray met the bed); with `eigenray3d` over relief and a `time_spread.bottom` section it now warns (`test_eigenray3d_scenario.py::test_time_spread_over_relief_says_what_it_assumes`). The 2-D tier is not given the warning: the fjord scenarios would start warning only if they had a `time_spread.bottom`, which they do not; their reverberation is flat at the sonar's depth under a profile bed, a scenario question (owner question 9), not changed. Cost record (measured, shared 4-core CPU): see "Cost of the eigenray3d tier" below) | Lambert weighting in `two_way_paths` (NumPy and C++) and `TimeSpreadProvider` take the facet normal where a bed has relief. Measure and record run time and disk per scenario | `sim/propagation.py`, `libs/sim` `paths.cpp` (both engines, parity test), `sim/timespread.py`, `docs/RESULTS.md` | Engine parity on a sloped facet; a flat bed bit-identical | None on a flat bed (all shipped). Over relief it is new behaviour | GPU suite for the C++ change | M |

**Cost of the `eigenray3d` tier** (measured 2026-11-26; `Bellhop3DProvider.runs` records each run's receivers,
seconds and bytes). Two Bellhop3D runs (arrivals and TL) per source position:

| Case | Fan (hat) | Receivers | Seconds per run | Arrivals file |
| --- | --- | --- | --- | --- |
| level 200 m bed, one bearing, 1 to 5 km | 241 × 61 over 20° | 5 | 3.3 | 18 kB |
| centred seamount, 2 bearings, 3 and 4 km | 241 × 61 over 76° | 4 | 19 | 4.3 kB |
| seamount of `test_seamount_3d.py`, 5 bearings × 13 ranges | 241 × 361 | 65 | 219 | 274 kB |
| the same | 241 × 181 | 65 | 107 | 272 kB |
| the same, 4 bearings × 13 ranges (P4 test) | 241 × 181 | 52 | 148 | 223 kB |
| `flank_seamount_3d` (one target, two source positions) | 241 × 181 | 1 | 57 | 0.5 kB; scenario 231 s |

Bytes per receiver: 3.6 to 4.8 kB binary (52 bytes an arrival; ASCII would be 171). The provider refuses a run
whose estimate (10 kB a receiver) exceeds `max_file_mb` (300 MB, about 30,000 receivers). Time grows with rays
× range × the receivers' bearings, not with the file.

**The acceptance bound (P4).** Today the test asserts `to_full[flanks] < -2.0` (ours at least
2 dB too loud). It becomes three assertions, with the provider of P3 at its production fan
against the oracle's converged 3-D field (241 × 721), incoherent TL averaged over 3.5 to 6.5 km,
on the five bearings −20, −4, 6, 16 and 32°:

1. **|ours − Bellhop3D 3-D| < 1.5 dB on every bearing.** The basis is the oracle's own beam
   convergence: range averages move by up to 1.1 dB between the two fans (measured, docstring).
   Add 0.4 dB for merging and lookup (ours, to be confirmed in P3). 1.5 dB is half the smallest
   3-D effect measured over the flanks (3.0 dB), so the test cannot pass with the effect missing.
2. **ours − Bellhop3D N × 2-D > 2.0 dB over the flanks** (−4 and 16°): the energy is turned out.
   This is the same margin as `test_horizontal_refraction_by_the_seamount`.
3. **Before the seamount (0.2 to 1.8 km), |ours − Bellhop3D 3-D| < 0.3 dB**, the existing
   near-range criterion.

A plumbing-only variant with the provider and the oracle at the **same** fan should be within
0.5 dB (ours). It shows that merging loses nothing. Neither bound includes `sonar-6c8`, because
the oracle and the provider are the same program. Two findings of `sonar-6c8` apply to both
bounds (section 2): the oracle must run at least 481 Gaussian elevation beams (121 is 0.5 to
1.2 dB short), and "ours", if it sums arrivals, must not be read over a bed written `'L'`, or
the bound must carry up to 1.7 dB. The level-bed test of P0 holds the comparison
with our 2-D tier. If the owner chooses B + E only, the test stays a characterisation and gains
an assertion that P1's warning fires on the flank bearings.

## 6. Questions for the owner

1. May Bellhop3D (GPL-3, Acoustics Toolbox, run as a separate process as Bellhop is) be an
   **opt-in provider** of the simulator, not only a test oracle? Answered 2026-11-26: yes, as a
   separate process, opt-in, for the later packages.
2. Do the operating waters have **steep relief across the look direction**: banks, seamounts,
   canyon or fjord walls? This is STATUS question 8, waters and bottoms, and 12, the coverage site.
3. Which **ranges and cross-bearing sea-floor slopes** matter for the long-range array? For
   example, beyond 2 km over slopes above 2°? Below about 1°, option A is not worth building.
4. How much **run time and disk** is acceptable for one long-range scenario at the 3-D tier?
   About a minute of tracing per source position was measured. A moving platform multiplies it.
   Arrival files run to gigabytes unless the grid is restricted.
5. Answered by `sonar-6c8` (section 2): no change to beam type, count or run type is needed; the
   level-bed scenarios' levels stay.
6. Should the simulator warn where the bed slopes steeply across a bearing (P1)? It changes no
   record. Done 2026-11-26 with the approved packages (P1: `reverb_relief.cross_slope_report`, a
   `RuntimeWarning` only).
7. For FLS and SAS: are harbours, pools with walls, canyons or sloping banks part of the use? If
   so, wall and tilted-plane multipath by image sources (option F) would be filed as its own issue.
   Filed and done as `sonar-0nsi` (D-201, opt-in, four multipath scenes); whether the use includes
   such waters is still not recorded.
8. If A goes ahead: the Fortran `bellhop3d.exe`, or bellhopcxx3d (also GPL-3, faster, its 3-D
   accuracy not checked here)? As built (P2, P3), the provider runs the Fortran `bellhop3d.exe`
   (`GPUSONAR_BELLHOP3D`, or beside `GPUSONAR_BELLHOP`; `sim/bellhop3d.py` `find_bellhop3d`);
   bellhopcxx3d was not tried.
9. **The fjord bed** (`fjord_static_trial_2row`, `_3row`, `fjord_testauv_trial_1row`): its profile
   is written to Bellhop as piecewise linear (`'L'`), over which Bellhop's arrivals disagree with
   its own TL run. Writing the same points as curvilinear (`'C'`) would move the target's one-way
   loss by +5.1, +0.2 and +3.6 dB, and does not settle it. Should the fjord profile be read as
   linear between its points (keep), as a smooth bed (`'C'`), or be referred to a wave model before
   anything changes? Nothing is changed until you say. **Corrected by D-199:** on the bed as
   shipped the summed arrivals agree with Bellhop's own loss run within 0.2, 0.3 and 1.1 dB (2-row,
   3-row, 1-row); the 2-row scenario's 2.7 dB against the loss run is its limit of 12 arrivals, not
   the bed; the +5.1, +0.2, +3.6 dB are the curvilinear form's arrivals disagreeing with its own loss
   run. Recommendation: keep the beds; the question left is whether the 2-row scenario keeps its
   12-arrival limit (issue `sonar-2e9e`).

## 7. What was not verified

- The size of 3-D effects at 4–6 kHz and at 27.5 kHz, at 12 km, and over shelf-like slopes of
  1° or less. Only the 8 kHz seamount was measured (D-149). The other rows of section 3 are
  first-principles estimates (ours).
- The dB consequence of the out-of-plane turning φ for FLS and SAS bounce paths. The geometry is
  ours. No scene has those paths to measure. SCENARIOS.md says the FLS sea scenes have direct
  paths only to objects, while their config states `image_source` with `max_order: 1`. Which
  paths reach the objects was not checked in the code.
- Whether Bellhop3D's irregular-grid option and binary arrivals work as documented. Neither was
  run.
- Why the mode-2 arrivals run was killed: memory shared with other agents, or Bellhop3D's own
  allocation. It was not repeated.
- Bellhop3D's memory use for a large receiver grid.
- bellhopcxx3d against the Fortran in 3-D.
- The run times above are single runs on a loaded, shared 4-core machine. They are not
  repeatable benchmarks.
- The licences of CAMBALA, SMPE and the WHOI 3-D PE beyond what their pages show. The DTIC
  report was not opened.
- `sonar-6c8`: the five candidates of section 2 were tested (section 2.1). Not verified: the
  mechanism of the `'L'`-bed excess in Bellhop's arrivals (read from `ArrMod.f90`, not traced);
  which representation is right for the fjord profiles (no wave model was run on them: their
  slopes, to 9°, are beyond what PyRAM's stair-stepped bed is trusted for here); Bellhop3D's own
  arrivals against its TL run; the three `ssp_*` scenarios' 0.2 to 1.3 dB between arrivals and
  TL run at their targets (single receivers).
- The 0.4 dB merging allowance and the 0.5 dB same-fan bound: ours, to be measured in P3.
- No Bellhop3D publication was read. Its user guide is in the oracle build tree
  (`Bellhop/Bellhop3D User Guide 2024_8_28.pdf`) and was not opened. Only `doc/bellhop3d.htm` and
  `ArrMod.f90` (format only) were.

## 8. What was run for this document

All runs used Acoustics Toolbox 8b4682b, built by `tools/build_oracles.sh` into the session's
scratch directory. Scripts were in that directory, not in the repository, and their output files
were deleted after reading.

1. `tests/python/bellhop3d_oracle.run_bellhop3d`, the seamount of `test_seamount_3d.py` (same bed
   function, grid, fan, bearings and ranges), modes `2` and `3`, timed with `time`.
   - Range averages over 3.5 to 6.5 km at −4, 6 and 16°: mode 3 gives 67.76, 73.57, 68.57 dB;
     mode 2 gives 64.41, 72.67, 65.06 dB.
   - Differences: +3.35, +0.90, +3.51 dB, consistent with the docstring's +3.1 to +3.5 and
     0.9 dB.
2. The same `.env` with run type `'AB   3'` (3-D arrivals) for 1 and 10 receiver depths (10 to
   100 m), and `'AB   2'` for one depth (killed).
3. `bellhop.exe` 2-D, run type `'A'`, automatic beams, on the seamount profile along the 6°
   bearing (25 m steps, `'L'` bathymetry), 10 depths × 65 ranges.
4. `wc -l` of the Fortran sources named in `BELLHOP3D_OBJECTS` of `Bellhop/Makefile`.

## Follow-ups (for the orchestrator to file)

- FLS and SAS multipath off walls and tilted planes through `image_source` (option F), if the
  owner answers question 7 with yes.
- `sonar-6c8`: close as explained (section 2), with the fjord bed as owner question 9.
- Raise the seamount oracle's elevation fan to 481 (`test_seamount_3d.py` FAN), then tighten
  `test_our_n_by_2d_tier_over_the_seamount` (its 3.5 dB bound held the 1 dB level-bed allowance);
  an owner decision, as it changes a test's numbers.
- Range-dependent water (`ssp_*`): Bellhop's arrivals against its TL run at the targets differ by
  0.2 to 1.3 dB at single receivers; check with range averages and a PE (water fronts are gentle
  enough for one).
