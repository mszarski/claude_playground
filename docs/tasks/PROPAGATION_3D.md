# Task: 3-D propagation for the eigenray tier

Issue `sonar-rcq` (P3), with `sonar-6c8` before it. Written 2026-11-26 at commit `48cc580`
(branch `fix-rcq`) as a scoping document: no code, config or test was changed. Read
`docs/HANDOFF.md` first, then this file. Status words are those of `docs/REFERENCES.md`
(**confirmed**: read in the opened source; **via secondary**; **ours**: our derivation or
measurement, stated as such; **unconfirmed**). "Measured here" means run on this machine on
2026-11-25/26 with the commands in section 8.

## 0. Answer first

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
  FLS beamwidths. A vertical quay or canyon wall is the extreme case. In water of one sound speed,
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
    "What no scenario has yet"; `sim/propagation.py`).
  - SAS scenes are `free_field`, with no surface or bottom multipath at all.
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
of Bellhop it would move records only through the differences between the two programs. Those
differences are `sonar-6c8`: up to 1.0 dB over a level bed.

**The scope therefore depends on the owner's waters.** If the operating areas have steep relief
across the look direction (banks, seamounts, canyons, fjord walls), the gap is 3 to 4 dB at a few
km, and an opt-in Bellhop3D provider is the right fix. It is about 4 to 6 weeks of work
(packages P2 to P6 in section 5), and the owner's yes on its licence is needed first. If the waters
are shelf-like (slopes of a degree or less), the effect is small. The first-principles estimate is
that φ is 0.2° per bounce at 5° grazing over a 1° cross-slope, against a 3.6° array beam. In that
case the right move is: fix `sonar-6c8`, add a warning where the cross-bearing slope is large,
and keep the characterisation test.

**Recommended:**

1. Do `sonar-6c8` now (P0, size M). It affects all 22 eigenray scenarios.
2. Add the cross-slope warning (P1, size S). It changes no record.
3. Build option A, Bellhop3D as an opt-in external provider, only after the owner answers
   questions 1 to 3 in section 6 with "yes" and "steep".

## 1. What is unknown, wrong or risky

1. **The owner's waters are unknown.** This is STATUS open questions 8 ("typical water depths and
   bottoms") and 12 (the coverage site is the illustrative `shelf_100m_sand`). The whole case for
   3-D propagation rests on cross-bearing slopes nobody has given us.
2. **Our 2-D tier and Bellhop3D's N × 2-D mode disagree by 1.0 to 2.5 dB on the seamount bearings,
   and by up to 1.0 dB over a level 200 m bed at 7 km** (`sonar-6c8`, open, cause not found). This
   is in the test docstring of `test_our_n_by_2d_tier_over_the_seamount`. Until it is explained,
   nobody knows which program is right over a level bed. Any agreement bound between our tier and
   Bellhop3D carries this 1 dB.
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
   Bellhop3D is today an oracle only (D-149). Its use as a provider is the owner's decision.
7. **Bellhop3D is "not independently re-validated"** (`OPEN_SOURCE.md`, Acoustics Toolbox row).
   Its own beam convergence on the seamount was measured: going from 121 × 361 to 241 × 721
   beams, the 3-D range averages move by up to 1.1 dB and the 3-D minus N × 2-D difference by
   0.6 dB (test docstring).

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

**`sonar-6c8`** (open, P3): "ours is 1.0 to 2.5 dB louder than Bellhop3D's N×2-D on the seamount
bearings, up to 1.0 dB over a level 200 m bed at 7 km. Candidates: beam type G vs B, bed
interpolation. Also: a scenario YAML using {seamount: ...}." Other candidates seen while scoping
this task (ours, none tested):

- Our provider runs Bellhop in arrivals mode (`'A'`) with Bellhop's automatic beam count. On one
  seamount bearing it chose 10,400 beams over ±30° (its `.prt`, measured here). The oracle runs in
  TL mode (`'I'`) with 121 beams in elevation.
- Our incoherent level is a sum over arrivals after the 20 dB floor of D-105.
- Our `.bty` is `'L'` (piecewise linear along range). Bellhop3D's is `'R'` (a rectilinear grid,
  interpolated in 2-D).

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
| FLS sea: `fls_120k_transit`, `_170k_transit`, `_330k_transit`, `_170k_traffic` | `image_source`, `max_order: 1` | 110–130, 160–180, 315–345 kHz | 20–450 m | 60 m, sonar 20 m deep | flat bed; a pinnacle of rocks as scatterers | none | Bounce paths off a bed tilted 5° across the bearing land 3.6 to 9.9° off (4 to 11 beams) at 15–58° grazing; from 0 dB (no bounce paths to objects today) to the level of a first-order bounce. Refraction: below 0.6° and 1.5 m vertical (2-D); horizontal 0.06° | formula above; geometry from the scenario files; refraction from c/G |
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
tier would equal Bellhop3D, which differs from our Bellhop tier by up to 1.0 dB (`sonar-6c8`).

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
mode 2 and with our tier (within the `sonar-6c8` bound once fixed); a mirror test of the
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
| P0 | **`sonar-6c8`** | Explain the 1.0 to 2.5 dB between our 2-D runs and Bellhop3D mode 2 (1.0 dB on a level bed). Isolate, one at a time, on a level 200 m bed and then the seamount bearings: (1) beam type `G`/`B`; (2) beam count (automatic 10,400 against 121); (3) arrivals `'A'` summed against TL `'I'`; (4) the 20 dB floor of D-105 on the incoherent sum; (5) `.bty` `'L'` against `'R'`. Record which explains what | `tests/python/test_seamount_3d.py` (a level-bed test), possibly `sim/bellhop.py` | `test_level_bed_our_tier_against_bellhop3d_n_by_2d`: with matched beams and run type, within 0.3 dB (the two modes' near-range agreement); first written as the 1.0 dB characterisation | **Yes if** the fix changes `BellhopProvider`'s run type, beam type or beam count by default. That moves all 22 eigenray scenarios and the gallery (by up to about 1 dB, estimate) | Bellhop and Bellhop3D on the CPU, seconds per run; then the full GPU suite (`tools/gpu_job.sh`) for no regression | M |
| P1 | Cross-slope warning (option B) | Say where N × 2-D is suspect | `sim/reverb_relief.py`, `sim/bathymetry.py` (a cross-bearing slope helper), `docs/CONFIGURATION.md` if a threshold key is added | Warns on the seamount flanks, not beside; never on a level bed or the fjord profile; records the largest φ | None (a warning only) | none | S |
| P2 | Reader and merger (option A starts here: owner questions 1 to 3 answered) | Read Bellhop3D 3-D arrivals (ASCII, then binary `'a'`) and merge beam contributions into eigenrays | new `sim/bellhop3d.py` (reader); the `.bty` and `.env` writers move from `tests/python/bellhop3d_oracle.py` (the oracle then imports them) | A hand-written fixture file read back exactly. Merging keeps total energy to 1e-6 and reduces the seamount's counts (up to 1,272) to below the cap of 64 with less than 1 % energy left out (target, ours) | None | Bellhop3D, CPU, about a minute; mind disk (1.1 GB ASCII at 18,850 receivers) | M |
| P3 | `Bellhop3DProvider` | `eigenrays(source, points)` with 3-D vectors, per-position cache, irregular grid, the D-105 rule and its warning | `sim/bellhop3d.py`, `sim/propagation.py` (`make_provider`: tier `eigenray3d`) | (a) Level bed: equals Bellhop3D mode 2 TL within 0.3 dB. (b) Mirror: the seamount mirrored about the source's bearing gives mirrored azimuths. (c) Unit `launch`/`arrival`; out-of-plane azimuth on the flanks above 0.5° (formula: 0.8–4° at 2–10° grazing over 12°). (d) Delay of the node correction within 20 µs, as the 2-D lookup | None (new tier) | Bellhop3D, CPU, minutes | L |
| P4 | Acceptance: the seamount bound | Turn `test_our_n_by_2d_tier_over_the_seamount` into an agreement bound | `tests/python/test_seamount_3d.py` | See the bound below | None | Bellhop3D, CPU; the converged oracle at 241 × 721 is about 4 × 53 s for mode 3 (estimate) | S |
| P5 | Scenario wiring and one shared bed | `bathymetry` (seamount or grid) reaches the provider, the scene seafloor and the reverberation from one key. A new illustrative scenario (for example the flank array over a seamount, with `unconfirmed` relief), as `sonar-6c8` also asks | `sim/simulator.py` (`run_scenario`), `sim/scene.py`, `sim/reverb_relief.py`, `configs/scenarios/<new>.yaml`, `docs/CONFIGURATION.md`, `docs/SCENARIOS.md` | The new scenario runs; its flank bearings are 2–4 dB quieter than the same scenario at `eigenray`. Reverberation over relief is bistatic with the 3-D provider, against point scatterers on the same bed within 2 dB (the existing D-73 criterion) | None for shipped scenarios. A new scenario adds a record and a gallery entry only if the owner wants it | Bellhop3D CPU; one full GPU-suite run to confirm nothing moved | M |
| P6 | Grazing over relief, cost record | Lambert weighting in `two_way_paths` (NumPy and C++) and `TimeSpreadProvider` take the facet normal where a bed has relief. Measure and record run time and disk per scenario | `sim/propagation.py`, `libs/sim` `paths.cpp` (both engines, parity test), `sim/timespread.py`, `docs/RESULTS.md` | Engine parity on a sloped facet; a flat bed bit-identical | None on a flat bed (all shipped). Over relief it is new behaviour | GPU suite for the C++ change | M |

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
the oracle and the provider are the same program. The level-bed test of P0 holds the comparison
with our 2-D tier. If the owner chooses B + E only, the test stays a characterisation and gains
an assertion that P1's warning fires on the flank bearings.

## 6. Questions for the owner

1. May Bellhop3D (GPL-3, Acoustics Toolbox, run as a separate process as Bellhop is) be an
   **opt-in provider** of the simulator, not only a test oracle?
2. Do the operating waters have **steep relief across the look direction**: banks, seamounts,
   canyon or fjord walls? This is STATUS question 8, waters and bottoms, and 12, the coverage site.
3. Which **ranges and cross-bearing sea-floor slopes** matter for the long-range array? For
   example, beyond 2 km over slopes above 2°? Below about 1°, option A is not worth building.
4. How much **run time and disk** is acceptable for one long-range scenario at the 3-D tier?
   About a minute of tracing per source position was measured. A moving platform multiplies it.
   Arrival files run to gigabytes unless the grid is restricted.
5. If `sonar-6c8` shows that our Bellhop runs should change (beam type, count or run type), may
   the levels of the 22 eigenray scenarios and the gallery move by up to about 1 dB?
6. Should the simulator warn where the bed slopes steeply across a bearing (P1)? It changes no
   record.
7. For FLS and SAS: are harbours, pools with walls, canyons or sloping banks part of the use? If
   so, wall and tilted-plane multipath by image sources (option F) would be filed as its own issue.
8. If A goes ahead: the Fortran `bellhop3d.exe`, or bellhopcxx3d (also GPL-3, faster, its 3-D
   accuracy not checked here)?

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
- The candidates listed under `sonar-6c8`: none was tested.
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
- `sonar-6c8`: add the five candidates of section 2 to its description.
