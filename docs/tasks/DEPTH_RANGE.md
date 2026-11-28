# Depth against range on the three-row face (package `depthrange`, issue sonar-9035; 2026-11-28)

The owner, told that the depth estimate fails at 1.7 km and was never measured at 12 km: "do all 3"; this is the first of
the three. Depth is what tells a submerged target from a ship, and the main target is the large submerged class at 12 km,
30 to 300 m deep. Every number here is **ours** (a simulation through the chain, or a closed form), with the command that
made it. No default, chain behaviour, record or committed figure was changed by this package.

Tool: `tools/depth_range_sweep.py` (opt-in; `--physics` for the closed forms, a sweep for the measurement, `--table` for the
tables below); test `tests/python/test_depth_range_sweep.py`; GPU job `GPUSONAR_JOB=depthrange`. Measurements:
`docs/tasks/depth_range_sweep.json` (all CPU sweeps, merged).

## DR.1 What failed, is dishonest, uncertain or not run (read first)

F1. **The face's stacked-row depth is dishonest in 100 m of water from 1 to 2 km.** With the attitude known at the echo (as
   navigation will send it) the deployed chain, told the water, states a wrong depth with a small sigma on 22 of the 28
   depths it gave at 1, 1.7 and 2 km (own depth 30 m): 70.4 +- 3.5 m for a 60 m target at 1 km; 83.9 +- 11.6 m for 30 m at
   1.7 km; 7.1 +- 12.3 m for 60 m at 2 km (one ping called "surface"); **a ship 4 m deep at 1.7 km reads the sea bed,
   95.7 +- 27.4 m, and is called submerged on 4 of 4 pings** (the 2 s hyperbolic sweep the same: 99.8 +- 30.4 m), again at
   8 km on 2 of 4. Not noise: two seeds give the same depths to 0.1 m. The rows read a mixture of the direct echo with the
   direct-surface and direct-bottom pairs a few range cells behind it (DR.3), and the multipath term of the sigma (D-133,
   `libs/live` `multipath_depth_limit`, on in every stacked-row deployment file) either finds no echo that explains the
   reading and adds nothing (`depth_unresolved_sigma_m: null`; its stated design: "nothing is claimed") or finds a band
   that leaves the truth out (the ship: 23.6 and 25.7 m of term about the sea bed). In deep water at own depth 30 m it
   happens once: **a 30 m target at 1.7 km reads 3.4 +- 10.2 m and is called a surface target on 4 of 4 pings.**
   Held by `test_stacked_row_depth_is_honest_in_shallow_water` (3 cases, expected to fail, strict). Not fixed: any remedy
   moves a default result (the owner's).
F2. **The multipath-timing stage (opt-in; no shipped build enables it) gives a target shallower than its search's floor
   the floor's depth with a small sigma:** a ship at 4 m reads 5.1 +- 0.2 m at 1.7 km, 8.0 +- 0.3 m at 3 km (own depth
   225 m; at 3 km that is called submerged); a 30 m target from 30 m reads 38.3 +- 1.2 m at 1.7 km, 59.4 +- 2.2 m at 3 km.
   Held by `test_multipath_timing_is_honest_above_its_search_floor` (expected to fail, strict).
F3. **The same stage pairs the wrong arrivals in 100 m of water and slips fringes with sub-fringe sigma** (reported, no
   test yet): a ship at 2 km reads 67.0 +- 1.5 m (called submerged); 30 m at 1.7 km reads 77.6 and 37.5 +- 1.3 m; in deep
   water 300 m reads 304.5 +- 0.7 m at 1 km and 297.7 +- 0.2 m at 2 km, 150 m reads 148.8 +- 0.2 m at 2 km (errors of 1 to
   5 m, 2 to 10 sigma: harmless to the call, not to the sigma). Of the timing depths given, 16 of 28, 22 of 84 and 23 of 108
   are dishonest (100 m of water; deep at 30 m; deep at 225 m).
F4. **Ghost detections are dishonest.** A deep target's surface pair, 30 m behind it at 1 to 2 km from 225 m, is reported
   as a detection of its own with a folded depth and the angle's sigma (180.7 +- 3.3 m for a 150 m target; 374.5 +- 6.8 m
   for 300 m): 16 such pings.
F5. **As deployed, the stated sigma ends depth by 2 km.** The deployment file's gradient prior (0.05 / s, unconfirmed)
   adds r^2 sigma_g / 2c: 48 m at 1.7 km, 150 m at 3 km, 2400 m at 12 km. With it no target of the class is told from the
   surface beyond 2 km by the rows nor beyond 3 km by timing: honest, and uninformative. The measurements are with the
   water known (as the self-test runs); the deployed call is the same depth with that term added.

Not run: the 2 s hyperbolic sweep on the whole grid (a 2 s sampled pulse simulates 5 to 17 times slower here; 8 spot
cases, DR.4), the `eigenray` tier, a rough surface (every case reflects -1 from a flat surface, the worst case for the
surface pair), a target with extent or aspect, a profile, sea states. GPU-job lines in DR.7. Uncertain: at 12 km in deep
water 3 of 12 submerged pings lie 2.2 to 2.6 sigma from the truth (a quarter, where a Gaussian gives a twentieth), so the
angle's sigma there may be some 25 % small; four pings a case.

## DR.2 What estimates depth, and what the physics says before measuring

`PYTHONPATH=python python3 tools/depth_range_sweep.py --physics` (the build `popoto_face_3x32`; 4-6 kHz, c = 1500 m/s).

**(a) The stacked rows** (the only depth the deployed face reports; `chain.elevation`, both engines, the fold of
`elevation.fold_depth`). Rows at -0.75, -0.20, +0.75 m: an aperture of 1.5 m = 5.0 wavelengths at 5 kHz, a resolution
lambda / L = 11.5 deg. The direct and surface paths arrive 2 z_t / R apart in angle; the aperture separates them only for
z_t > R lambda / 2L = R / 10 (100 m at 1 km, 1200 m at 12 km): never for the class beyond 3 km. The build's own sigma budget
(`buildspec.performance`, a +25 dB point, free field):

| Range (km) | 1 | 1.7 | 2 | 3 | 5 | 8 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Depth sigma, water known (noise, roll 0.2 deg), m | 3.5 | 6.0 | 7.0 | 10.5 | 17.6 | 29.9 | 64.6 |
| Depth sigma, deployed gradient prior 0.05 / s, m | 17.0 | 48.5 | 67.0 | 150.4 | 417 | 1067 | 2401 |

What the rows read when the direct and surface echoes share a range cell: the receive field across the rows is
e^{ikh theta_d} + mu e^{i phi} e^{ikh theta_s}; with mu = -1 the phase slope gives Re[(theta_d - e^{i phi} theta_s) /
(1 - e^{i phi})] = (theta_d + theta_s) / 2 = z_own / R **for every phase phi**: the straight line at that angle reaches the
sea surface. A submerged target whose direct echo is not clear of its surface echo reads as a surface target (held by
`test_the_rows_read_the_surface_when_direct_and_surface_share_the_cell`). The direct echo leads the direct-surface pair by
z_own z_t / R; the range cell is c / 2B = 0.375 m. In shallow water the direct-bottom pair (reading the sea bed) is
(H - z_own)(H - z_t) / R behind it.

**(b) Multipath timing** (`chain.multipath_depth`, opt-in, one C++ stage for both engines; on in no shipped deployment
file). The surface echo's two-way delay tau = 4 z_own z_t / (R c); a depth is searched where tau B >= 3 (1.5 ms):
z_own z_t / R >= 0.5625 m. Two-way delay, ms (R = range resolved: lead of the direct echo >= one cell; M = searched):

| Own depth | Target | 1 km | 1.7 km | 2 km | 3 km | 5 km | 8 km | 12 km |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 m | 30 m | 0.40 | 0.24 | 0.20 | 0.13 | 0.08 | 0.05 | 0.03 |
| 5 m | 300 m | 4.0 RM | 2.35 RM | 2.0 RM | 1.33 R | 0.80 | 0.50 | 0.33 |
| 30 m | 30 m | 2.4 RM | 1.41 R | 1.20 R | 0.80 | 0.48 | 0.30 | 0.20 |
| 30 m | 60 m | 4.8 RM | 2.82 RM | 2.40 RM | 1.60 RM | 0.96 | 0.60 | 0.40 |
| 30 m | 150 m | 12 RM | 7.1 RM | 6.0 RM | 4.0 RM | 2.4 RM | 1.5 RM | 1.0 R |
| 30 m | 300 m | 24 RM | 14 RM | 12 RM | 8.0 RM | 4.8 RM | 3.0 RM | 2.0 RM |
| 100 m | 30 m | 8.0 RM | 4.7 RM | 4.0 RM | 2.7 RM | 1.6 RM | 1.0 R | 0.67 |
| 100 m | 60 m | 16 RM | 9.4 RM | 8.0 RM | 5.3 RM | 3.2 RM | 2.0 RM | 1.33 R |
| 225 m | 30 m | 18 RM | 10.6 RM | 9.0 RM | 6.0 RM | 3.6 RM | 2.25 RM | 1.50 RM |
| 225 m | 60 m | 36 RM | 21 RM | 18 RM | 12 RM | 7.2 RM | 4.5 RM | 3.0 RM |

(All own depths 5, 30, 100, 225 m and targets 30, 60, 150, 300 m are in the JSON.) Own depth is the lever: 225 m (the
planner's own depth at the deep site) keeps every target of the class searchable to 12 km; 30 m loses 30 m targets beyond
1 km and 60 m beyond 3 km. The unknown gradient adds z_t r^2 sigma_g / (2 c (z_own + z_t)) (RESULTS, "Depth from multipath
timing").

**(c) Anything else.** No stage uses a bottom-bounce pair or a Lloyd-mirror pattern along the track for depth; the timing
stage puts the bottom into its pattern when the water depth and bottom are given (`bottom_order` 2).

**Predicted, before measuring:** with the water known, the rows tell a submerged target from a ship only where the direct
echo is a few cells clear (z_own z_t / R well above 0.375 m) and the angle's sigma is small: with own depth 225 m in deep
water to about 3 km for 30 m and about 8 to 12 km for 150 to 300 m; at own depth 30 m not at 1.7 km for 30 m (lead 0.53 m);
in 100 m of water the bottom pairs add a second error. As deployed, nowhere past 2 km. Timing: wherever M holds, to 12 km at
225 m. When the chain cannot tell it should say so: a wide sigma or no depth; a wrong depth with a small sigma is the
dangerous case.

## DR.3 The 1.7 km failure of the rehearsal self-test, exactly

`tools/build_from_spec.py popoto_face_3x32 --out DIR --self-test --only rehearsal` (22 min on this CPU; C++ and Python
equal): free field passes; the rehearsal (100 m of water over sand, own depth 30 m, the target 60 m deep at 1.7 km, 3 deg of
roll at 1.5 deg/s, raw boards with their clocks) gives:

| Ping | Elevation read (level) | Depth +- sigma | Multipath term | Judged |
| --- | --- | --- | --- | --- |
| 1 | +1.94 deg | 28.0 +- 27.8 m (fold at the surface) | 27.1 m, multipath-limited | honest (1.15 sigma) |
| 2 | +6.53 deg | 35.4 +- 11.7 m (folded twice) | null: no echo explains it | **not honest (2.11 sigma)** |
| 3 | none (the rows' coherence under `min_quality`) | no depth | - | counted as not honest |

The stage is the stacked rows' elevation (the first-arrival rule, then the fold); multipath timing is not on. The arrivals
(flat surface, sand at 1650 m/s): direct at -1.01 deg; 2.8 range cells behind it the direct-surface pair, twice its
pressure, reading +1.01 deg (the surface); 4.4 cells behind the direct-bottom pair (|R| 0.58 at 3.7 deg grazing), reading
-2.36 deg (the sea bed); then the surface-surface echo (+3.03 deg), the surface-bottom pairs and more within 3.3 m
(`tools/depth_range_sweep.py --physics`: `rehearsal_echoes`, from `echoes()`). The rows read +1.9 and +6.5 deg, which is none of these: a phase mixture of the direct echo with
the pairs in the compressed pulse's skirt. At 1.0 km the same pairs are 4.8 and 7.4 cells behind and the bottom's
reflection is weaker (|R| 0.39 at 6.3 deg), and the rehearsal read 59, 60 and 63 m (D-256, HB.1 item 4). So: **a resolution limit of the
geometry the blind range forces, not a defect of the pairing and not the sweep** (both sweeps have the same band; with the
1 s linear sweep at 1.7 km the rehearsal gives no depth on any ping, D-256 HB.1 item 4).

Where the sigma came from: ping 2's 11.7 m is the roll's 0.2 deg (5.9 m at 1.7 km) and the fold through the sea bed (the
water depth's 5 m sigma, twice), in quadrature; the multipath term would have said more, but it found no echo that explains
+6.5 deg and added nothing (F1). The attitude is known at the echo (the record's navigation history covers the receive
window, which runs a pulse length past its last range). The cheap form (`tools/depth_range_sweep.py`, baseband, the same
geometry with the vehicle rolled 3 deg and not turning, an attitude history given) on two seeds: 1 s linear sweep at
1.0 km 68.5 to 72.7 +- 3.5 m, at 1.7 km 19.4 to 40.0 +- 11.7 m, all for 60 m (dishonest on 11 of 12 pings); without an
attitude history the chain charges 0.5 deg/s for every second since the transmit and states 43 to 51 m instead (1 s and
2 s sweeps: 19.4 to 44.6 m read), which is what made those runs look honest. What the self-test means by "depth honesty" and
what it should check: DR.6.

## DR.4 Measured: depth against range

`PYTHONPATH=python python3 tools/depth_range_sweep.py --ranges 1000,1700,2000,3000,5000,8000,12000 --depths 4,30,60,150,300
--own OWN --water WATER --waveform lfm1 --pings 2 --seeds 1,2 --multipath --json OUT`, then `--table OUT...`: image source,
a +25 dB point (the class's low end, assumed), flat surface, ambient noise at 5 m/s wind, the vehicle level and moving at
1.5 m/s, an attitude history to the echo's end, the face's deployed chain told the water (isovelocity, its depth); both
engines (equal to 0.003 m in depth on every ping: single-precision maps). "Told": "submerged" when depth - 2 sigma > 6 m
(the ship class's deepest), "surface" when depth + 2 sigma < 30 m; honest: the truth within 2 sigma. Four pings a case.
The 1 s linear sweep carries the grid (the depth physics is the band's, which both sweeps share; the 2 s hyperbolic sweep
is spot-checked below).

Worst first: where a wrong call or a dishonest depth was given (rows: stacked rows; timing: multipath timing).

| Water, own depth, pulse | Range (m) | Target (m) | Detected | Rows: depth +- sigma (median) | Honest / dishonest | Call (wrong) | Call, deployed sigma (wrong) | Ghosts dishonest | Timing: depth +- sigma | Timing honest / dishonest; no depth | Timing call (wrong) | Timing call, deployed sigma (wrong) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 m, 30 m, lfm1 | 1700 | 4 | 4/4 | 95.7 +- 27.4 | 0 / 4 | submerged 4 (4) | undecided 4 (0) | 0 | - +- - | 0 / 0; single_arrival 4 | none 4 (0) | none 4 (0) |
| deep, 30 m, lfm1 | 1700 | 30 | 4/4 | 3.4 +- 10.2 | 0 / 4 | surface 4 (4) | undecided 4 (0) | 0 | 38.3 +- 1.2 | 0 / 4; - | submerged 4 (0) | undecided 4 (0) |
| 100 m, 30 m, lfm1 | 2000 | 60 | 4/4 | 7.1 +- 12.3 | 0 / 4 | surface 1, undecided 3 (1) | undecided 4 (0) | 0 | 60.1 +- 1.5 | 4 / 0; - | submerged 4 (0) | undecided 4 (0) |
| 100 m, 30 m, lfm1 | 1000 | 30 | 4/4 | 99.2 +- 21.3 | 0 / 4 | submerged 4 (0) | submerged 4 (0) | 0 | - +- - | 0 / 0; single_arrival 4 | none 4 (0) | none 4 (0) |
| 100 m, 30 m, lfm1 | 1000 | 60 | 4/4 | 70.4 +- 3.5 | 0 / 4 | submerged 4 (0) | submerged 4 (0) | 0 | - +- - | 0 / 0; single_arrival 4 | none 4 (0) | none 4 (0) |
| deep, 225 m, lfm1 | 1000 | 150 | 4/4 | 150.0 +- 3.5 | 4 / 0 | submerged 4 (0) | submerged 4 (0) | 4 | - +- - | 0 / 0; window 4 | none 4 (0) | none 4 (0) |
| deep, 225 m, lfm1 | 1000 | 300 | 4/4 | 300.0 +- 3.6 | 4 / 0 | submerged 4 (0) | submerged 4 (0) | 4 | - +- - | 0 / 0; window 4 | none 4 (0) | none 4 (0) |
| 100 m, 30 m, lfm1 | 1700 | 30 | 4/4 | 83.9 +- 11.6 | 0 / 4 | submerged 4 (0) | undecided 4 (0) | 0 | 57.6 +- 1.3 | 0 / 4; - | submerged 4 (0) | submerged 2, undecided 2 (0) |
| deep, 225 m, lfm1 | 1700 | 300 | 4/4 | 299.9 +- 5.9 | 4 / 0 | submerged 4 (0) | submerged 4 (0) | 4 | 300.1 +- 0.2 | 4 / 0; - | submerged 4 (0) | submerged 4 (0) |
| deep, 225 m, lfm1 | 2000 | 300 | 4/4 | 300.0 +- 7.0 | 4 / 0 | submerged 4 (0) | submerged 4 (0) | 4 | 297.7 +- 0.2 | 2 / 2; - | submerged 4 (0) | submerged 4 (0) |
| 100 m, 30 m, lfm1 | 8000 | 4 | 4/4 | 95.3 +- 45.9 | 2 / 2 | submerged 2, undecided 2 (2) | undecided 4 (0) | 0 | - +- - | 0 / 0; merged 4 | none 4 (0) | none 4 (0) |
| 100 m, 30 m, hfm2 | 1700 | 4 | 2/2 | 99.8 +- 30.4 | 0 / 2 | submerged 2 (2) | undecided 2 (0) | 0 | - +- - | 0 / 0; single_arrival 2 | none 2 (0) | none 2 (0) |
| 100 m, 30 m, lfm1 | 1700 | 60 | 4/4 | 35.9 +- 11.7 | 2 / 2 | submerged 4 (0) | undecided 4 (0) | 0 | 59.9 +- 1.3 | 4 / 0; - | submerged 4 (0) | undecided 4 (0) |
| deep, 30 m, lfm1 | 1700 | 60 | 4/4 | 74.0 +- 8.2 | 2 / 2 | submerged 4 (0) | undecided 4 (0) | 0 | 60.1 +- 1.2 | 4 / 0; - | submerged 4 (0) | undecided 4 (0) |
| deep, 225 m, hfm2 | 1700 | 300 | 2/2 | 299.9 +- 5.9 | 2 / 0 | submerged 2 (0) | submerged 2 (0) | 2 | 300.0 +- 0.2 | 2 / 0; - | submerged 2 (0) | submerged 2 (0) |
| deep, 225 m, lfm1 | 12000 | 60 | 4/4 | 52.3 +- 80.3 | 3 / 1 | submerged 1, undecided 3 (0) | undecided 4 (0) | 0 | 60.0 +- 1.2 | 4 / 0; - | submerged 4 (0) | undecided 4 (0) |
| deep, 225 m, lfm1 | 12000 | 150 | 4/4 | 144.6 +- 85.8 | 3 / 1 | submerged 2, undecided 2 (0) | undecided 4 (0) | 0 | 150.0 +- 1.2 | 4 / 0; - | submerged 4 (0) | undecided 4 (0) |
| deep, 225 m, hfm2 | 1700 | 60 | 2/2 | 60.8 +- 6.0 | 2 / 0 | submerged 2 (0) | undecided 2 (0) | 1 | 60.2 +- 0.2 | 2 / 0; - | submerged 2 (0) | submerged 2 (0) |
| 100 m, 30 m, hfm2 | 1700 | 60 | 2/2 | 37.1 +- 11.7 | 1 / 1 | submerged 2 (0) | undecided 2 (0) | 0 | 60.0 +- 1.3 | 2 / 0; - | submerged 2 (0) | undecided 2 (0) |
| deep, 225 m, lfm1 | 12000 | 300 | 4/4 | 287.9 +- 86.1 | 3 / 1 | submerged 4 (0) | undecided 4 (0) | 0 | 300.1 +- 1.2 | 4 / 0; - | submerged 4 (0) | undecided 4 (0) |
| 100 m, 30 m, lfm1 | 2000 | 4 | 4/4 | - +- - | 0 / 0 | none 4 (0) | none 4 (0) | 0 | 67.0 +- 1.5 | 0 / 4; - | submerged 4 (4) | undecided 4 (0) |
| 100 m, 30 m, lfm1 | 2000 | 30 | 4/4 | 19.5 +- 12.2 | 4 / 0 | undecided 4 (0) | undecided 4 (0) | 0 | 40.8 +- 1.5 | 0 / 4; - | submerged 4 (0) | undecided 4 (0) |
| deep, 30 m, lfm1 | 2000 | 30 | 4/4 | 15.9 +- 9.7 | 4 / 0 | undecided 4 (0) | undecided 4 (0) | 0 | 39.7 +- 1.4 | 0 / 4; - | submerged 4 (0) | undecided 4 (0) |
| deep, 30 m, lfm1 | 3000 | 30 | 4/4 | 27.5 +- 14.7 | 4 / 0 | undecided 4 (0) | undecided 4 (0) | 0 | 59.4 +- 2.2 | 0 / 4; - | submerged 4 (0) | undecided 4 (0) |
| deep, 225 m, lfm1 | 8000 | 30 | 4/4 | 19.3 +- 31.8 | 4 / 0 | undecided 4 (0) | undecided 4 (0) | 0 | 30.2 +- 0.8 | 3 / 1; - | submerged 4 (0) | undecided 4 (0) |
| deep, 30 m, lfm1 | 1000 | 150 | 4/4 | 149.9 +- 3.6 | 4 / 0 | submerged 4 (0) | submerged 4 (0) | 0 | 152.7 +- 0.7 | 2 / 2; - | submerged 4 (0) | submerged 4 (0) |
| deep, 30 m, lfm1 | 1000 | 300 | 4/4 | 299.9 +- 3.4 | 4 / 0 | submerged 4 (0) | submerged 4 (0) | 0 | 304.5 +- 0.7 | 0 / 4; - | submerged 4 (0) | submerged 4 (0) |
| deep, 225 m, lfm1 | 1700 | 4 | 4/4 | 0.5 +- 6.0 | 4 / 0 | surface 4 (0) | undecided 4 (0) | 0 | 5.1 +- 0.2 | 0 / 4; - | surface 4 (0) | surface 4 (0) |
| deep, 225 m, hfm2 | 1700 | 4 | 2/2 | 0.1 +- 5.9 | 2 / 0 | surface 2 (0) | undecided 2 (0) | 0 | 5.1 +- 0.2 | 0 / 2; - | surface 2 (0) | surface 2 (0) |
| deep, 30 m, lfm1 | 2000 | 300 | 4/4 | 300.7 +- 7.1 | 4 / 0 | submerged 4 (0) | submerged 4 (0) | 0 | 308.3 +- 1.4 | 0 / 4; - | submerged 4 (0) | submerged 4 (0) |
| deep, 225 m, lfm1 | 2000 | 4 | 4/4 | 2.1 +- 7.0 | 4 / 0 | surface 4 (0) | undecided 4 (0) | 0 | 5.4 +- 0.2 | 0 / 4; - | surface 4 (0) | surface 4 (0) |
| deep, 225 m, lfm1 | 2000 | 150 | 4/4 | 149.9 +- 7.0 | 4 / 0 | submerged 4 (0) | submerged 4 (0) | 0 | 148.8 +- 0.2 | 0 / 4; - | submerged 4 (0) | submerged 4 (0) |
| 100 m, 30 m, lfm1 | 3000 | 30 | 4/4 | 60.3 +- 21.3 | 4 / 0 | submerged 4 (0) | undecided 4 (0) | 0 | 62.7 +- 2.2 | 0 / 4; - | submerged 4 (0) | undecided 4 (0) |
| deep, 225 m, lfm1 | 3000 | 4 | 4/4 | 3.9 +- 10.6 | 4 / 0 | surface 4 (0) | undecided 4 (0) | 0 | 8.0 +- 0.3 | 0 / 4; - | submerged 4 (4) | surface 4 (0) |
| deep, 225 m, lfm1 | 3000 | 300 | 4/4 | 300.3 +- 10.5 | 4 / 0 | submerged 4 (0) | undecided 4 (0) | 0 | 297.9 +- 0.3 | 0 / 4; - | submerged 4 (0) | submerged 4 (0) |

Where each target is told from a ship, with the water known and with the deployed gradient prior (ranges in km at which
every ping was told right):

| Water, own depth, pulse | Target (m) | Rows tell it at (km) | Rows, deployed sigma | Timing tells it at | Timing, deployed sigma | Rows dishonest at (m) | Timing dishonest at (m) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 m, 30 m, hfm2 | 4 | none | none | none | none | 1700 | none |
| 100 m, 30 m, hfm2 | 60 | 1.7 | none | 1.7 | none | 1700 | none |
| 100 m, 30 m, lfm1 | 4 | none | none | none | none | 1700, 8000 | 2000 |
| 100 m, 30 m, lfm1 | 30 | 1 1.7 3 | 1 | 1.7 2 3 | none | 1000, 1700 | 1700, 2000, 3000 |
| 100 m, 30 m, lfm1 | 60 | 1 1.7 | 1 | 1.7 2 3 | none | 1000, 1700, 2000 | none |
| deep, 30 m, lfm1 | 4 | 1 1.7 2 | none | none | none | none | none |
| deep, 30 m, lfm1 | 30 | 1 | none | 1 1.7 2 3 | 1 | 1700 | 1700, 2000, 3000 |
| deep, 30 m, lfm1 | 60 | 1 1.7 2 3 | 1 | 1 1.7 2 3 | 1 | 1700 | none |
| deep, 30 m, lfm1 | 150 | 1 1.7 2 3 5 8 | 1 1.7 2 | 1 1.7 2 3 5 8 | 1 1.7 2 | none | 1000 |
| deep, 30 m, lfm1 | 300 | 1 1.7 2 3 5 8 12 | 1 1.7 2 | 1 1.7 2 3 5 8 12 | 1 1.7 2 3 | none | 1000, 2000 |
| deep, 225 m, hfm2 | 4 | 1.7 | none | 1.7 | 1.7 | none | 1700 |
| deep, 225 m, hfm2 | 60 | 1.7 | none | 1.7 12 | 1.7 | none | none |
| deep, 225 m, hfm2 | 300 | 1.7 12 | 1.7 | 1.7 12 | 1.7 | none | none |
| deep, 225 m, lfm1 | 4 | 1 1.7 2 3 | none | 1.7 2 | 1.7 2 3 | none | 1700, 2000, 3000 |
| deep, 225 m, lfm1 | 30 | 1 1.7 2 | none | 1.7 2 3 5 8 12 | 1.7 2 | none | 8000 |
| deep, 225 m, lfm1 | 60 | 1 1.7 2 3 5 | 1 | 1.7 2 3 5 8 12 | 1.7 2 | 12000 | none |
| deep, 225 m, lfm1 | 150 | 1 1.7 2 3 5 8 | 1 1.7 2 | 1.7 2 3 5 8 12 | 1.7 2 3 | 12000 | 2000 |
| deep, 225 m, lfm1 | 300 | 1 1.7 2 3 5 8 12 | 1 1.7 2 | 1.7 2 3 5 8 12 | 1.7 2 3 | 12000 | 2000, 3000 |

The 2 s hyperbolic sweep (the baseline): the rows marked hfm2 above (one seed, two pings: `--waveform hfm2 --seeds 1` at
1.7 and 12 km in deep water at own depth 225 m, 1.7 km in 100 m of water). They read as the 1 s linear sweep does: the ship
in 100 m of water at 1.7 km 99.8 +- 30.4 m (called submerged), the 60 m target 35.2 and 39.1 +- 11.7 m; in deep water 0.1 +-
5.9 m for the ship and 60.8 +- 6.0 m for 60 m at 1.7 km, at 12 km 17.8 to 39.2 +- 62 to 68 m for 60 m and 240.9 to 297.9 +-
66 to 73 m for 300 m (the 1 s sweep: 46 to 95 m of sigma; the 2 s sweep's 3 dB more energy).

Totals (1 s sweep): in 100 m of water at own depth 30 m the rows gave a depth on 68 of 84 pings, **24 dishonest, 22 of them
at 1 to 2 km (of 28 depths there), 7 wrong calls**; in deep water at 30 m, 6 dishonest of 132 (all at 1.7 km), 4 wrong
calls (the 30 m target called surface); at 225 m, 3 dishonest of 140 (all at 12 km), no wrong call, and 16 dishonest ghost
detections at 1 to 2 km. Timing gave a depth on 28, 84 and 108 pings, of which 16, 22 and 23 dishonest; its wrong calls: a
ship called submerged at 2 km in 100 m of water (67.0 +- 1.5 m) and at 3 km in deep water (8.0 +- 0.3 m).

Not detected: at own depth 30 m in deep water the 4 m and 60 m targets at 12 km on no ping (the 30, 150 and 300 m ones on all):
the one-way phase 2 k z_own z_t / R is 2 pi at 60 m, a Lloyd-mirror null (and 0.4 rad at 4 m); the planner's own-depth
choice (225 m at the deep site) is about this.

## DR.5 Where depth stops being usable, and what would extend it

**Where it stops** (the operational question: is the target 30 m or more down, or at the surface?), measured, the water
known; the physics' prediction (DR.2) beside it:

| Case | Stacked rows (deployed stage) | Prediction | Multipath timing (opt-in, not deployed) | Prediction |
| --- | --- | --- | --- | --- |
| deep water, own depth 225 m | 30 m to 2 km (3 of 4 pings at 3 km), 60 m to 5 km, 150 m to 8 km, 300 m to 12 km; ship "surface" to 3 km | sigma-limited: 3 / 5 / 8 / 12 km | every depth 1.7 to 12 km (sigma 0.2 to 1.2 m; at 1 km no depth: `window`) | to 12 km (M at every range) |
| deep water, own depth 30 m | **30 m only at 1 km, called surface at 1.7 km**; 60 m to 3 km, 150 m to 8 km, 300 m to 12 km | 30 m lost at 1.7 km (lead 0.53 m, 1.4 cells) | 30 and 60 m to 3 km (30 m by the floor, F2), 150 m to 8 km, 300 m to 12 km | 30 m: 1 km; 60 m: 3 km; 150 m: 8 km |
| 100 m of water, own depth 30 m | **nowhere honestly**: right calls at 1 to 1.7 km with dishonest depths; a ship called submerged at 1.7 km | the bottom pairs 4 to 7 cells behind: mixed readings to about 2 km | 60 m to 3 km; 30 m and the ship mis-paired (F3); `unresolvable` / `merged` beyond 3 km | 60 m: 3 km; 30 m: 1 km |
| any, as deployed (gradient prior 0.05 / s) | no target past 2 km (150 and 300 m to 2 km, 60 m to 1 km, 30 m nowhere) | sigma 48 m at 1.7 km, 150 m at 3 km | to 2 or 3 km | z_t r^2 sigma_g / (2c (z_own + z_t)) |

So **for the main target at 12 km, as the face is deployed, depth cannot tell a submerged target from a ship**: the
stacked rows' own stated sigma is 2.4 km there (the gradient prior), and with the water known it still tells only the
deepest (300 m) from the surface; multipath timing would tell 30 to 300 m at 12 km in deep water from an own depth of
225 m, if the water is known and its own defects (F2, F3) are fixed. Inside 2 km in shallow water the deployed depth is
worse than uninformative: it is confidently wrong (F1).

**What would extend it**, each with its evidence or marked model:

1. **Knowing the water's sound speed through the column** (a cast or a dive; the owner: a dive must never be needed). The
   deployed sigma is the gradient prior's beyond 1 km (16.7 of 17.0 m at 1 km, 2400 of 2401 m at 12 km; `--physics`); the
   measured columns "water known" against "deployed sigma" show it is the term that ends depth at 2 km. With the profile
   known the measured reach is the table's (evidence: this sweep; the profile ladder's own study, RESULTS "Depth by rung").
2. **Multipath timing on the face, from a deep own depth** (the stage exists, opt-in; `chain.multipath_depth`). Measured
   0.2 to 1.2 m of sigma at 1.7 to 12 km in deep water at 225 m for 30 to 300 m targets. Before it can be deployed: the
   search floor (F2), the shallow-water pairing (F3) and fringe slips with sub-fringe sigma (F3) must be fixed, and its own
   gradient term (at 12 km, 0.05 / s: 282 m for 30 m, 1371 m for 300 m; 0.005 / s: a tenth) still needs the water known.
3. **Own depth.** Model and measured: the direct echo's lead over the surface pair is z_own z_t / R, so 225 m against 30 m
   is a factor 7.5 (deep water at 30 m loses the 30 m target at 1.7 km; at 225 m it holds to 2 to 3 km). The planner
   already chooses 225 m at the deep site for detection.
4. **More bandwidth** (model). Five cells of lead for the rows need B >= 5 R c / (2 z_own z_t): at 12 km, own 30 m,
   target 30 m, 50 kHz; the timing stage's floor B >= 3 R c / (4 z_own z_t): 15 kHz. Not from a 4-6 kHz transducer; from
   225 m the 2 kHz band already suffices for timing to 12 km.
5. **More vertical aperture** (model). Separating direct and surface by angle needs L >= lambda R / (2 z_t): 60 m at 12 km
   for 30 m: not from this face (2 m high). The noise part of the rows' sigma at 12 km (49 m of 65 m with the water known)
   halves with twice the aperture, but the roll term (0.2 deg: 42 m) needs a better attitude, not aperture.
6. **The bottom bounce in shallow water** (model). The direct-bottom pair carries (H - z_own)(H - z_t) / R; the timing
   stage already puts the bottom in its pattern (`bottom_order` 2) and in 100 m of water it mis-paired (F3): not usable
   until that is understood.
7. **Depth from the track (the Lloyd-mirror fading against range)** (model, not measured). The one-way factor |1 - e^{i phi}|,
   phi = 2 k z_own z_t / R, fades with a period in range of 2 pi R / phi. At 12 km from 30 m a 60 m target is in a null
   (phi = 2 pi: measured above, not detected on any ping) and the period is the whole track; from 225 m the period is 1.6 km
   but phi varies by +-20 % across the band (+-9 rad), which washes the fades out. With the target strength's own
   scintillation of several dB and a hull's extent, the pattern is not a usable depth estimator here.
8. **A short-range mode** (a short pulse for the first kilometres). It restores the 1 km behaviour (deep water: right at
   1 km for every depth here), not the shallow-water one (F1 holds at 1 km), and nothing at 12 km.

## DR.6 The build self-test: what the 3x32 rehearsal's depth check should check at 1.7 km

**What the check means now** (`tools/build_from_spec.py` `_check_depth`, D-133, D-143): on every ping with a detection of
the target, (i) a depth and a sigma are given, (ii) the truth lies within 2 stated sigma (for an ambiguous elevation, of one
candidate's fold), (iii) the multipath term does not exceed the water column about the reported depth. "Honest" is (ii)
and (iii); (i) makes a ping without a depth a failure.

**Why it fails at 1.7 km:** ping 2 fails (ii), a wrong depth with a stated sigma (35.4 +- 11.7 m for 60 m): **dishonesty,
the serious kind**, from F1. Ping 3 fails (i): no depth, which at 1.7 km is the honest answer when the rows' coherence is too low; that part
of the criterion assumed the 1 km geometry, where the rows always formed a reading. Ping 1 passes because its multipath term
(27 m) happened to be "explained".

**Proposed change** (not made; it does not loosen the honesty test):

1. Keep (ii) and (iii) as they are on every ping that gives a depth.
2. A ping with a detection and no depth (`elevation_valid` false, or withheld) counts as honest-and-uninformative, recorded
   (`no_depth` and the reason), not as a failure; the test fails if no ping of the trial gives a depth.
3. Add the operational check as a recorded criterion: the call (submerged / surface / undecided, DR.4) is never wrong.
   Undecided is allowed at 1.7 km.
4. Keep the target at 1.7 km (the face's nearest range with the baseline pulse). With 1 to 3 the self-test will go on
   failing at 1.7 km until F1 is remedied, which is right: the deployed depth is dishonest there.

## DR.7 GPU-job lines, commands, files

Not run here, for the GPU job (`tools/gpu_job.sh --help`; one JSON a sweep, uploaded after each):

    GPUSONAR_JOB=depthrange tools/gpu_job.sh
        # tools/depth_range_sweep.py --backend cuda --ranges 1700,2000,3000,5000,8000,12000 --depths 4,30,60,150,300
        #   --waveform hfm2 --pings 3 --seeds 1,2 --multipath, three sweeps: --water deep --own 225, --water deep --own 30,
        #   --water 100 --own 30; then the pooled tables (/work/depthrange/tables.md, pooled.json)
        # 180 cases of 3 pings; a case costs 2 to 8 min on this CPU, its cost on an L4 is not measured: if the 3 h limit
        # comes first, split the ranges over two jobs with GPUSONAR_DEPTHRANGE_ARGS (each sweep's JSON is uploaded as it ends)

The `eigenray` tier on this grid needs a fidelity option the tool does not have yet (a follow-up).

The rehearsal self-test with the baseline (22 min on this CPU): `tools/build_from_spec.py popoto_face_3x32 --out DIR
--self-test --only rehearsal`; the shelf at 12 km with Bellhop: `tests/python/test_buildspec.py -k 12km` (GPUSONAR_BELLHOP).

Files: `tools/depth_range_sweep.py` (new), `tests/python/test_depth_range_sweep.py` (new: 9 tests, 5 expected to fail,
strict), `tools/gpu_job.sh` (the `depthrange` mode), this document and `docs/tasks/depth_range_sweep.json` (the CPU
sweeps pooled; the commands inside). CPU times: a 1 s linear-sweep case of 2 pings, both engines, 10 to 40 s; a 2 s
hyperbolic one 2 to 8 min.
