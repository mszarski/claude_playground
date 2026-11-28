# Conformance of code, examples and expectations to the read references

**State 2026-11-26, after the merge `b88c458` (full GPU suite 2763 passed, 102 skipped, 4 xfailed; job
6b07ea74d91f621b930a343d):** the apply list is applied, A1-A10 and the first item of A13, except A11 (the matrix
wording of `docs/figures/sst_validation.md`, not done). Each item below carries its commit and decision; the
decisions are D-208 to D-225 in `docs/DECISIONS.md`. Three statements of this file were wrong and are corrected
at their items (A3, A4, A8). Matrix rows changed by the later reading of Mourad and Jackson (1989), Wagner (1967),
Williams and Jackson (1998) and Jackson and Briggs (1992) (D-222) are marked "2026-11-26 later"; the counts
below are the audit's and were not recounted.

**State 2026-11-26, after the merge `61cfb66` of the second batch (D-226 to D-229; full GPU suite 2823 passed, 103
skipped, 4 xfailed; job 6b080457d91f621b930a652a):** the Williams and Jackson (1998) join and spectral volume term,
the relief-exponent warning and the 1-D to 2-D relief helper are merged as opt-ins in Python (D-226; defaults
unchanged, small-slope kept, D-229); the sites state their boundary scattering and the depth plan and the coverage
pictures include it (D-227). The rows below that said "in progress, not merged" or "not done" for these carry a
"D-226" or "D-227" note.

**State 2026-11-27, after the merge `8581533` of the fourth batch (D-235 to D-240; full GPU suite 2933 passed, 113
skipped, 4 xfailed; job 6b08b9f554072e8b71681cd2; as three shards 2941 passed, 114 skipped, 4 xfailed):** D-235 to
D-240 name no matrix row and the batch did not edit this file. The `penetration: rough` cost cap (D-236,
`roughness.max_pairs`) raises before any pairing and never truncates, so it changes no conformance item under the
cap; the sea-state labels (D-235) and the rest of the batch (the rows' gain against reverberation, Doppler D3, POP-3,
the sharding) touch no conformance item.

**State 2026-11-28, after the merge `08ddfab` of the eighth batch (D-255 to D-261; full GPU suite as three shards
3494 passed, 124 skipped, 4 xfailed; jobs 6b0a30146c5d61fff18ac1ab, 6b0a301554072e8b716b9b00,
6b0a30156c5d61fff18ac1ae):** D-255 to D-261 name no matrix row and the batch did not edit this file. The FLS
deployment files and `look: boresight` (D-257), the tracker's range rate (D-258) and the gallery's evidence (D-259)
are chain and tool changes; the deep-site package (D-260) changed no simulator default: its two scenes state the
planner's site, and the site's `bubble_model` reaches the simulator only through the scene's
`time_spread.surface.bubble_loss_db` (a test holds the two equal). The outputs D-251 made stale were regenerated
(`sonar-rar0` closed). No row's verdict changes.

**State 2026-11-27, after the merge `e06fac8` of the seventh batch (D-251 to D-254; full GPU suite as three shards
3392 passed, 122 skipped, 4 xfailed; jobs 6b0947c254072e8b71697a54, 6b0947c254072e8b71697a58,
6b0947c36c5d61fff188c461):** D-251 to D-254 name no matrix row and the batch did not edit this file. **D-251 fixes the
defect D-249 found**: the boundary reverberation's Doppler is per frequency by default (`doppler: per_frequency`; each
sector's pulse time-scaled by 1 + d / fc as well as shifted, as TR 0702 eqs. (63)-(64), Sec. 8.3, confirmed, give for
the echo of a still scatterer; the scaled pulse and its interpolation ours; `docs/PHYSICS_NOTES.md` 14.7), and
`doppler: centre` keeps the old law bit for bit. No matrix row covers the reverberation's Doppler over a band, so no
row's verdict changes. Outputs made before it with a moving platform and boundary reverberation are stale (issue
`sonar-rar0`). The tone path's recount (D-252) and the chain view with its read-only taps (D-253) touch no conformance
item.

**State 2026-11-27, after the merge `bda7c7d` of the sixth batch (D-246 to D-250; full GPU suite as three shards
3321 passed, 121 skipped, 5 xfailed; jobs 6b0935f554072e8b71696d01, 6b0935f66c5d61fff188b937,
6b0935f754072e8b71696d05):** D-246 to D-250 name no matrix row and the batch did not edit this file. **Found, not
fixed (D-249, `sonar-l8yp`): the boundary reverberation's Doppler is one shift at the record's centre frequency**
(`python/gpusonar/sim/reverb.py` `_series_numpy`, `libs/sim/src/reverb.cpp`), where each frequency should shift by
2 (v.u) f / c; no matrix row covers the reverberation's Doppler over a band, so no row's verdict changes, but every
moving-platform reverberation measured near its Doppler ridges is in doubt until the fix (branch `revdop`; the owner:
"FIx it"; fixed by D-251, the paragraph above). The tone path's lobe notch (D-246), the transmit study and the hyperbolic sweep (D-247, no simulator change;
the HFM echo of a mover checked as the time-scaled pulse in both engines) and the visualization (D-248: read-only, with
opt-in taps that leave records bit-identical) touch no conformance item.

**State 2026-11-27, after the merge `f40d4c3` of the fifth batch (D-241 to D-245; full GPU suite as three shards
3002 passed, 117 skipped, 4 xfailed; jobs 6b090cd36c5d61fff188820e, 6b090cd354072e8b71692fa9,
6b090cd46c5d61fff1888214):** D-241 to D-245 name no matrix row and the batch did not edit this file or the
simulator's physics. The Doppler GPU job mode (D-241), the Doppler stages batched on the device (D-242), the two
12 km large-submerged scenes (D-243, opt-in) and the owner's decisions (D-244) touch no conformance item.

**State 2026-11-26, after the merge `4057d65` of the third batch (D-230 to D-234; full GPU suite 2886 passed, 111
skipped, 4 xfailed; job 6b084a4cd91f621b930b6331):** D-230 to D-233 name no matrix row and the batch did not edit this file. The library sediment fit takes a
row as the tool does (D-230, opt-in; A10 carries a note); the rest of the batch (pulse trains, the sweep's Doppler bank,
the 12 km trade study) touches no conformance item.

Written 2026-11-26 on branch `conform` (issue `sonar-jbnm`), after the owner's instruction "make sure our code and
examples and expectations match the references everywhere", amended the same day: "apply every alignment without
asking using your judgment to calibrate to real parameters, scenes, references". This file has three parts:

1. **Alignments to apply** (below): every change that matching a read reference requires and that would move a
   shipped result, in an order implementation agents can execute without further questions. Nothing in that list
   is implemented on this branch.
2. **The matrix** (section 2): one row per checked item, with what the source says, what ours does, the verdict and
   how it was checked, what depends on it, and the action.
3. What was fixed here (section 3: documents, docstrings, comments; no behaviour), what was not checked
   (section 4), the scripts (section 5).

Sources and tags as in `docs/tasks/SST_GAPS_JACKSON.md`: [N1]-[N4] the four Jackson and Richardson (2007) notes
(`docs/references/jackson_richardson_2007/01-04_*.md`), [NS] `surface_papers_mcdaniel_dahl.md` (McDaniel 1993
[McD93], Dahl et al. 1997 [D97], Dahl and Plant 1997 [DP97]), [NF] `surface_forward_scatter_dahl_1996_2004.md`
(Dahl 1996 [D96], Dahl 2004 [D04]), [NB] `surface_bistatic_dahl_1999_mcdaniel_1986.md` (Dahl 1999 [D99], McDaniel
1986 [McD86]). "TR 0702" is the open SST report; TR 9407 is never read. Status words as in `docs/REFERENCES.md`.
Every "ours" number below was computed for this file by the scripts of section 5 with the repository's own
functions; a number a note reports is cited to the note and was re-computed where the row says so.

**Counts (section 2, 149 rows):** MATCHES 71; DIFFERS 48; NO SOURCE READ 15; SOURCE MISPRINT 8 (ours follows the
corrected value, or does not use the misprinted form); NOT CHECKED 7. Of the 48 DIFFERS, 12 were only a document,
comment, docstring or citation and are fixed on this branch; the other 36 are model or scenario differences, each
in the apply list (A1-A11), kept with a stated reason (A13), in a plan package or a follow-up; several of those
also had their description fixed here.

---

## 1. Alignments to apply

The principle used (the owner's): a printed, read reference wins over ours; a value read from a figure or derived
by an agent does not override a printed class value but is recorded as a range or a named site; where no source
exists, the description says so and no number changes. Each item says how sure the reference is: **printed** (read
on the page), **figure** (read off a curve, about +-1 to 2 dB), **one site** (one data set), **ours** (an agent's
computation). Every item is a behaviour change: tests first (a test that fails before and passes after), both
engines where the item touches them, one decision entry, before -> after on a representative case, the full GPU
suite before merge.

### 1.0 Parallel sets (no two sets in one row touch the same file)

| Set | Items | Files | Can run beside |
|---|---|---|---|
| S1 surface | A1, A2, A3 (one agent, in that order) | `python/gpusonar/sim/surface.py`, `tests/python/test_gallaudet_boundaries.py`, `test_surface_dahl_1997.py`, `test_literature_models.py`, `test_surface_models.py`, `test_reverb_bistatic.py`, `docs/CONFIGURATION.md` (rows `bubble_layer_surface`, surface model) | S2, S3, S4, S5 |
| S2 forward scatter | A4 | `python/gpusonar/acoustics.py` (new function), `python/gpusonar/sim/timespread.py`, `tests/python/test_timespread.py`, `tests/python/test_forward_scatter_dahl.py` (new), `tools/sst_validation.py`, `tests/python/test_sst_validation.py`, `docs/figures/sst_validation.{md,json}`, `docs/figures/sst_asiaex.png` | S1, S3, S4 |
| S3 sediment table | A5, A6 | `configs/sediments.yaml`, `python/gpusonar/sim/bottom.py` (grain-size default and loader keys only), `tests/python/test_grain_size_relations.py`, `test_sediment_table.py`, `test_sediment_literature.py`, `test_gallaudet_boundaries.py` (one assertion; coordinate with S1: different test function) | S1, S2, S4 |
| S4 new scenario | A7 | `configs/scenarios/asiaex_dahl_2004.yaml` (new), `tests/python/test_forward_scatter_dahl.py` (shared with S2: S4 adds its own test file `test_asiaex_dahl_2004.py` instead), `docs/SCENARIOS.md` | S1, S3 |
| S5 after G-3 lands | A8 (rock), A9 (buried scenes, G-3's own agent) | `python/gpusonar/sim/bottom.py` (`reflection`), `python/gpusonar/sim/propagation.py` (`rayleigh_reflection`), `libs/sim/src/paths.cpp`, `python/gpusonar/sim/bellhop.py`, `configs/sediments.yaml` (rock, pebbles rows), `tests/python/test_sim_cpp.py`, `tests/cpp/test_sim_paths.cpp` | after G-3 and after S3 (shares `sediments.yaml`, `bottom.py`) |
| S6 after G-9 lands | A10 (fit tool default) | `tools/fit_sediment.py`, `python/gpusonar/sim/sediment_fit.py`, `tests/python/test_sediment_fit.py` | anything but G-9 |
| S7 documents (orchestrator) | A11, A12, A13 | `docs/RESULTS.md`, `docs/STATUS.md`, `docs/handoff/SIMULATION.md`, `docs/DECISIONS.md`, `tools/sst_validation.py` matrix wording (after S2) | after S2 for the tool |

### 1.1 The items, in the recommended order

**A1. `bubble_layer_surface`: `dahl_1997` becomes the default extinction** (set S1).
**Applied 2026-11-26: `be5c72b`, D-216.**
- Change: `surface.bubble_layer_optical_depth(..., extinction="apl_1994")` -> default `"dahl_1997"`; the same default
  in `bubble_backscatter`, `bubble_layer_surface_db`, `bubble_layer_surface_strength` and wherever
  `sim/simulator.py` passes the item's `extinction` (only the default of the keyword; the scenario key stays).
  `apl_1994` stays selectable. The `bubble_loss_db` laws of `time_spread` and `lloyd.py` are a different thing and
  do not change.
- Reference: [D97] Table II (printed: log10 beta_I = a + b log10 U10 at 26.5 / 50 / 70 kHz, 17 degrees, North Sea);
  one site and one angle; the log-f interpolation between the three fits and the f^0.8 scaling outside are ours
  (already in `dahl_1997_optical_depth`). A second, independent point agrees: [D99] beta_I <= 0.0012 at 30 kHz,
  about 4 m/s; `dahl_1997` gives 0.00142 (+0.7 dB), `apl_1994` 0.00395 (+5.2 dB) (computed).
- Before -> after: Gallaudet 68 kHz, 3 m/s band means (the D-193 test): -38.3 / -38.1 dB -> -44.4 / -43.7 dB at
  27-38 / 44-66 degrees (printed data -51.3 / -41.5; the excess +13.0 / +3.4 -> +6.9 / -2.2 dB; G-7's report,
  reproduced by `test_surface_with_the_published_extinction_law_at_3_m_s`). Wind exponent at 17 degrees, 2-7 m/s:
  U^4.25 / 4.06 / 3.91 -> U^5.37 / 6.20 / 5.10 (measured 5.5 / 6.4 / 5.3). At 8 m/s and above up to +3.7 dB
  ([NS] Sec. 3.2).
- Exposure: no shipped scenario uses `bubble_layer_surface` (all 90 expanded, `survey.py`). Tests that pin the
  `apl_1994` default and must be rewritten: `test_gallaudet_boundaries.py::test_surface_models_lie_above_the_printed_band_means_at_3_m_s`,
  `test_surface_dahl_1997.py::test_default_path_is_bit_identical_to_before` (becomes "the `apl_1994` path is
  bit-identical when named"), `test_literature_models.py::test_optical_depth_is_the_one_of_the_loss_law`,
  `::test_facet_term_and_the_whole_model`, `::test_scenario_key_for_the_bubble_layer_surface_model`,
  `test_surface_models.py` (re-run; only `saturation_crossing` uses the saturation level, which does not depend on
  beta_I). Documents: `docs/CONFIGURATION.md` row `bubble_layer_surface`, `docs/PHYSICS_NOTES.md` Sec. 23.3 and
  23.7, `docs/RESULTS.md` row "Boundary models against Gallaudet", `docs/REFERENCES.md` Sec. 4 Gallaudet row,
  `docs/handoff/SIMULATION.md` "Boundary models" row. GPU: nothing (Python strengths); run the suite.

**A2. Bubble attenuation of the facet term** (set S1, after A1).
**Applied 2026-11-26: `5ac3d09`, D-216** (key `facet_attenuation`).
- Change: `bubble_layer_surface_db`: total = bubble term + facet x exp(-2 beta_I / sin(t)) (the same beta_I as the
  bubble term).
- Reference: [D97] Sec. VI and [D99] eqs. (11)-(12) (printed). The exponent is one crossing per leg, the
  convention of [D99] (12) and of ours; [D96] eq. (17) prints twice it and [D99] calls that a misprint.
- Before -> after (with `apl_1994` beta_I, [NB] Sec. 3.2, not recomputed here): 0 dB at 45 degrees and below;
  30 kHz: -0.16 / -0.60 / -0.59 dB at 60 / 80 / 89 degrees (8 m/s) to -1.27 / -1.59 / -1.57 (15 m/s); 68 kHz up
  to -3.17 dB (80 degrees, 15 m/s). With `dahl_1997` beta_I (A1) larger at high wind: -2 to -9 dB at 80 degrees,
  10-12 m/s ([NB] Sec. 3.2).
- Exposure: none shipped. Tests: `test_literature_models.py::test_facet_term_and_the_whole_model`,
  `test_gallaudet_boundaries.py::test_surface_models_lie_above_the_printed_band_means_at_3_m_s` (44-66 degree band:
  check; under 0.2 dB at 3 m/s). Documents: as A1.

**A3. Bistatic surface strengths: the geometric mean of the two backscattering strengths** (set S1, after A2).
**Applied 2026-11-26: `ede3110`, D-216. Correction: "every monostatic use is unchanged / bit-identical" below is wrong:** the strength is evaluated for every pair of eigenrays, and a monostatic sonar over a bottom has pairs that meet the surface at two angles (a direct path out, a bottom-bounced path back); surface reverberation fell in every shipped scene with a surface law and multipath (`popoto_12km` -1.28 dB, `popoto_env_gravel_30m` -0.98, `flank_patrol` -0.91, `flank_sst_shallow` -0.08 overall and -1.43 in its worst twentieth, FLS scenes under 0.35 dB; one ping each). `sst_pursuit` streak 16.7 -> 15.6 dB.
- Change: `surface.empirical_surface_strength` (both laws, capped and not), `kraus_surface_strength`,
  `bubble_layer_surface_strength`, `searay_bottom_strength`: sigma(g_in, g_out) = sqrt(sigma(g_in) sigma(g_out))
  instead of sigma((g_in + g_out) / 2). Equal when g_in = g_out, so every monostatic use is bit-identical.
- Reference: TR 0702 Sec. 6.4.2 (printed: "monostatic models made bistatic by the geometric mean of the two
  backscattering strengths"). Supporting: of the two rules it is the closer to [D99] eq. (13), the one published
  bistatic surface term (by 0.3-5 dB; neither reproduces it, [NB] Sec. 3.2).
- Before -> after (computed, Chapman-Harris, 9.9 kHz, 8 m/s, `sst_pursuit`'s band and wind): pairs 5/5, 5/20,
  5/40, 10/30, 5/80, 2/30 degrees: 0, -1.7, -3.6, -1.1, -5.9, -5.7 dB. Only paths whose two legs meet the surface
  at different angles move: in `sst_pursuit` (transmitter and receiver on one vehicle) those with a bottom leg.
- Exposure: `sst_pursuit` (the only `geometry: bistatic` surface item in the 90 files); its tests
  `test_sst_validation.py::test_pursuit_surface_reverberation_dominates` (median ridge/noise 30.1 dB, band level
  over noise 15.6 dB), `::test_pursuit_bottom_bounce_streak_starts_when_geometry_says` (streak +16.7 dB over the
  run without a bottom: will fall), `::test_reverberation_gives_each_eigenray_pair_its_own_doppler`
  (+4.6 dB near 0.88 s); `tests/python/test_reverb_bistatic.py` (tests that evaluate a law at unequal angles:
  re-run, change the expected rule where one asserts the mean angle); `docs/figures/sst_validation.{md,json}`,
  `docs/figures/sst_pursuit.png`, `docs/SCENARIOS.md` pursuit paragraph, `docs/PHYSICS_NOTES.md` Sec. 14.2.
  The image-source and bistatic reverberation strengths are taken from Python by both engines and by the CUDA
  backend, so no C++ change; the GPU suite re-measures the pursuit records.

**A4. Forward scatter: the large-scale slope replaces the total Cox-Munk slope in time spreading** (set S2).
**Applied 2026-11-26: `1738009`, D-217.** Correction: the slope ratio is 0.477 at the fitted 9.14 m/s, not inside "0.55-0.74 at 3-9 m/s". The SPACE'08 residual above 12 degrees became 1.89 (re-simulated; "about 2" predicted); D-218 rejected the exact Kirchhoff tail as its cause.
- Change: new `acoustics.large_scale_mean_square_slope(U) = 0.0046 ln(2.1 U^2)` (U10 in m/s; clamp below at its
  1.5 m/s value 0.0071 with a warning, since it is negative below 0.69 m/s), and in
  `timespread.boundary_parameters` the `mean_square_slope` from `wind_speed` becomes this one. An explicit
  `mean_square_slope` still wins. `acoustics.mean_square_slope` itself, `surface.facet_backscatter` (backscatter
  facets, [McD93] eq. (7)) and the bottom are NOT changed. The `exclude_specular` lobe (`reverb_specular`) takes its
  slope from the item's own `mean_square_slope` or the time-spread section: it follows automatically when it reads
  the time-spread value; check.
- Reference: [D96] eq. (A7) (printed; Phillips' fit to Cox and Munk's slick-surface slopes, waves longer than about
  30 cm); [D99] prints s_L^2 = 0.017 at 4 m/s (runs 2-3; (A7) gives 0.0162 at 4 m/s, 5 % lower: consistent);
  [D04] the low-pass slope 0.152 rms (0.0231) at 7 m/s against (A7) 0.0213. With s_LA the closed form reproduces
  [D96] Table I coherence within 0-19 %; with the Cox-Munk total it is 21-39 % short at five sets and 67 % at
  1.5 m/s ([NF] Sec. 3.2, the closed-form columns re-derived here: q = sqrt(2) against 1.348, +4.9 %).
- Open contradiction to record, not to follow: [D99]'s measured angle spreads (runs 2-4) imply a mean-square slope
  of 0.008-0.011 at about 4 m/s, half of (A7)'s 0.0162 ([NB] Sec. 3.3, agents' computation); [D99]'s own
  explanation is that the late decay belongs to Bragg-scale waves a facet model lacks.
- **Evidence against, from our own data comparison (say it plainly):** the SPACE'08 comparison of
  `docs/PHYSICS_NOTES.md` Sec. 21.4 (item 5) found measured over model decay time (facet, total slope) median 0.99
  overall, 1.28 above 12 degrees and 0.59 below. Shortening the model by the (A7) ratio (0.55-0.74 at 3-9 m/s)
  moves those to about 1.5 overall and about 2 above 12 degrees, better below 12 degrees (about 0.9). Dahl's
  measurements are at 6-20 degrees grazing; SPACE'08's steep arrivals (21-32 degrees at 60 m) argue against the
  change there. Recommendation: apply it (printed law, two papers, the geometry already theirs), and state the
  steep-angle SPACE'08 residual in the decision entry; do not tune.
- Before -> after (computed, `slope_fls.py`): at 6 m/s (`flank_sst_shallow`, `flank_sst_spread`) mss 0.0337 ->
  0.0199; for their sonar-to-target surface bounce (30 m, 50 m, 1000 m, 4.57 degrees, 8 kHz) facet mean delay
  5.32 -> 3.14 ms and e-fold 10.57 -> 6.24 ms (x0.590), rms angle 5.58 -> 4.29 degrees in the vertical plane,
  0.445 -> 0.342 across (x0.768), coherence length across 3.84 -> 5.00 m, vertical 0.31 -> 0.40 m (x1.302). At
  3 m/s (`sst_asiaex`) mss 0.0184 -> 0.0135; surface bounce (9.09 degrees): TR 0702's rule-of-thumb time 2.91 ->
  2.14 ms (the quantity `test_asiaex_tails_decay_at_the_rule_of_thumb` compares the fit with), facet mean 1.49 ->
  1.10 ms (x0.736), angles x0.858, coherence x1.165. The Doppler widths do not change (vertical speed), nor the
  coherent fraction (height).
- Exposure: scenarios `flank_sst_shallow`, `flank_sst_spread`, `sst_asiaex`. Their tests:
  `test_timespread.py::test_the_sst_style_scenario_runs_and_its_bounce_arrivals_fade`,
  `::test_the_scenario_with_every_option_runs_on_both_engines`, `::test_per_ray_doppler_on_the_cuda_kernel_matches_the_host`
  (qualitative: re-run), every `test_timespread.py` test that builds `boundary_parameters({"wind_speed": ...})`
  and compares with a stored number or a closed form taking `acoustics.mean_square_slope` (grep for it; e.g.
  `test_the_options_off_give_the_output_of_the_version_before_them` uses wind 6 m/s: its reference must take the
  new slope or name `mean_square_slope` explicitly), `test_sst_validation.py::test_asiaex_tails_decay_at_the_rule_of_thumb`
  (the tool's `rule()` must take the same slope as the provider: `tools/sst_validation.py` line 757),
  `test_oneway_sources.py`, `test_sampled_waveform.py`, `test_image_planes.py`, `test_environment.py`,
  `test_reverb_specular.py` (each builds a time-spread section with a wind: re-run). Tools that call
  `mean_square_slope` directly for their own comparisons and must state which slope they use:
  `tools/channel_validation.py` (lines 244, 331), `tools/channel_smartocean.py` (667). Figures and documents:
  `docs/figures/sst_validation.{md,json}` rows Sec. 12.4 and 12.5, `docs/figures/sst_asiaex.png`,
  `docs/SCENARIOS.md` ASIAEX paragraph (surface -6 dB re peak, 1.88 dB), `docs/RESULTS.md` TR 0702 rows,
  `docs/PHYSICS_NOTES.md` Secs. 16, 21.4, 23.4, `docs/CONFIGURATION.md` `time_spread` rows. GPU: the CUDA test of
  the per-ray Doppler and the full suite; the time spreading itself is Python (taps to both engines).

**A5. Grain-size input: the printed relations become the default between table points** (set S3).
**Applied 2026-11-26: `efd2aff`, D-209.** Correction: a second tolerance (delta, 2e-5) would also have failed; no tolerance was widened: the strict check names `table_linear` and the relations got their own checks. C5 (D-223) later moved rho to 7.5e-5 off and widened that check from 6e-5 to 8e-5.
- Change: `bottom.sediment_from_grain_size(phi, method="table_linear")` -> default `method="relations"`
  (`apl_grain_size_relations`, with 1.0019 and 0.4556, the two printed constants read as misprints).
  `"table_linear"` stays selectable.
- Reference: J&R (11.1)-(11.3), pp. 313-314 (printed; the relations behind the table, attributed to Mourad and
  Jackson 1989 and TR 9407; "ad hoc" fits, not regressions).
- Before -> after: at the 20 table points rho within 5e-4, nu 1e-4, delta 0.5 %; between them at most 0.68 dB in
  the normal-incidence reflection (5.3 phi), under 0.1 dB outside 5-6 phi, rho up to 0.0106 (1.25 phi), delta up to
  7.6 % (near 5.8 phi) (`test_grain_size_relations.py::test_difference_from_todays_interpolation`). At Gallaudet's
  6.36 phi: rho 1.14828 -> 1.14825, nu 0.986436 -> 0.986430, delta 0.003284 -> 0.003267 (computed; his printed
  Table VI 1.1483, 0.9864, 0.0033).
- Exposure: no shipped scenario or library caller uses the function (grep). Tests:
  `test_grain_size_relations.py::test_the_default_is_todays_interpolation_bit_for_bit` and
  `::test_the_default_matches_values_saved_from_the_tree_before_the_change` (become tests of
  `method="table_linear"`), `test_gallaudet_boundaries.py::test_table_vi_geoacoustic_ratios_are_our_grain_size_interpolation`
  (tolerance 4e-5: rho now differs from the printed 1.1483 by 5.1e-5, inside the print's own rounding of 5e-5:
  widen to 6e-5 and say why, or call `method="table_linear"` there), `test_sediment_literature.py::test_sediment_from_grain_size_hits_the_classes_and_stays_between_them`
  (re-run: at the class points the relations differ from the rows by up to 5e-4). Documents: `docs/CONFIGURATION.md`
  if it names the method, `docs/PHYSICS_NOTES.md` Sec. 23.1. GPU: nothing.

**A6. The seven soft rows: keep the class values, add explicit ranges and named site rows** (set S3, after A5).
**Applied 2026-11-26: `0f02a2c`, D-209;** the range became [0.0008, 0.005] with Jackson and Briggs (1992) Table V's printed fits (`9585adc`, D-224).
- Change, per `docs/tasks/SEDIMENT_DEFAULTS_DECISION.md` option E: each of `medium_silt`, `sandy_mud`, `fine_silt`,
  `sandy_clay`, `very_fine_silt`, `silty_clay`, `clay` keeps sigma2 0.001 (0.000986 for `very_fine_silt`) and the
  w2 rule, and gains `sigma2_range: [0.0008, 0.0037]` with a `basis:` naming the five J&R Table 14.1 sites
  (agents' computation by (14.10): 0.0031, 0.0037, 0.0008, 0.0017, 0.0016) and Gallaudet's fit (0.003), and a
  `w2_note:` that Briggs' spectra (J&R Table 6.1) put silts and clays 1-7 dB above the rule at 10-100 kHz (+7.1 dB
  at 10 kHz, +2.2 at 100 kHz). New NAMED site rows (never "_measured" class rows): `gallaudet_2001_site` with his
  Table VI inputs (rho 1.1483, nu 0.9864, delta 0.0033, sigma2 0.003, gamma2 3.00, w2 0.000518 cm^4; Mz 6.36 from
  porosity), `status: confirmed` (printed); J&R Table 14.1 rows only if the owner allows the table in the
  repository (decision 11 of the plan; until then not added). The loader must accept and carry the new keys
  without using them.
- Record (owner's note, from the fit-tool agent's work, not re-derived here): Gallaudet's two band means cannot
  separate volume from roughness scattering, so his 0.003 is independent evidence for the volume parameter only if
  volume dominance is assumed (our model makes volume 20-30 dB above roughness there, `PHYSICS_NOTES` Sec. 23.2,
  which is the assumption). The range above therefore counts it as one fitted site.
- Before -> after: no number moves (the class values stay). Exposure: `test_sediment_table.py` (keys, loader),
  `test_gallaudet_boundaries.py::test_bottom_model_with_the_site_fit_against_the_printed_band_means` (could use the
  named row instead of its own copy), `docs/REFERENCES.md` Sec. 3.1. GPU: nothing.

**A7. A second ASIAEX scenario with the geometry and sea Dahl 2004 prints** (set S4). `sst_asiaex` stays TR 0702's
example with its quoted winds (3 m/s as used, 4 m/s as quoted).
**Applied 2026-11-26: `e0f97aa`, D-217.**
- Change: new `configs/scenarios/asiaex_dahl_2004.yaml`: source 25 m, one receiver at 52 m (the lower array, where
  the surface arrival is clear of the direct path) and optionally a second run at 26 m, horizontal range 467 m
  (Fig. 4 caption; 467 m reproduces the printed 9.4 degree iso-speed angle: 9.36 computed), water 105 m, 8 kHz
  (and 20 kHz as a second waveform or a variant), 3 ms CW, wind 7 m/s, `time_spread.surface.rms_height: 0.3`
  ([D04] H = 0.3 +- 0.1 m measured; the wind law gives 0.262 m at 7 m/s), sound speed 1525 m/s (the printed profile
  is 1521-1528 m/s, read from figure; `unconfirmed`), the slope from A4. `unconfirmed`: sediment (not printed),
  sound-speed profile, ping interval. Its own test file checks what [D04] prints and the geometry gives: the
  surface arrival's grazing angle 9.4 degrees at 52 m (iso-speed), the surface arrival at 3.62-3.68 ms after the
  direct one, the bottom arrival at 11.7-11.9 ms (computed; [D04] Fig. 3c shows the bottom arrival at 13.7 ms in
  the measured record, figure reading, with refraction: record the 2 ms difference, do not fit it).
- Reference: [D04] Sec. IV.A (printed). Not a reproduction of [D04]'s coherence curves (its small-slope model is in
  [D99] with an unprinted spectrum): the vertical coherence at 52 m, 20 kHz, 0.30 m is 0.69 measured (figure) and
  0.03-0.2 in our model ([NF] Sec. 3.2): an expected-failure test that records it.
- Why not change `sst_asiaex` instead (computed, `asiaex.py`): with [D04]'s layout the five arrivals cannot be
  TR 0702's five separate arrivals inside 35 ms. Receiver 26 m: surface 1.85 ms after the direct (the pulse is
  3 ms), surface-bottom and bottom-surface 0.55 ms apart. Receiver 52 m: bottom-surface at 37.3 ms (35 ms is the
  figure's window). `test_asiaex_scenario_holds_what_tr0702_states` would fail at either. At 7 m/s the surface
  bounce is no longer partly coherent (0.771 -> 0.000-0.026), and the p. 101 rule of TR 0702 read literally gives
  14.8-23.6 Hz, above TR 0702's "less than 10 Hz" (ours, 2 f v sin(g) / c, gives 0.5-0.8 Hz).
- Exposure: new file only; `docs/SCENARIOS.md` (count 90 -> 91 and its row), any test that counts or expands every
  scenario (`test_yaml_numbers.py`, scenario-listing tests: re-run). GPU: the suite runs every scenario test.

**A8. `rock`: elastic flat-interface reflection** (set S5, after G-3 and S3).
**Applied 2026-11-26: `e6020b5` (simulator, Bellhop; D-214) and `1013fea` (the live chain; D-215).** Corrections: the shear used is the printed generic rock of arXiv:1601.06078 Table II, 1900 m/s and 0.085 (1900/3820 = 0.50), not "0.5 V_p" by our choice; the "peak 16.7 dB at 35.7 degrees" below is the loss AT 35.7 degrees: the peak for this item's case is 28.76 dB at 33.43 degrees (for the row 27.9 dB at 33.1 degrees); Fig. 9.3 differs from ours by up to 0.506 dB (36.1 degrees), not under 0.5.
- Change: `configs/sediments.yaml` `rock` gains `shear_speed_ratio: 1.25` (V_s = 0.5 V_p = 1910 m/s on 1528 m/s
  water) and `shear_delta: 0.085`, both `unconfirmed_fields: [shear]` with the basis "class choice: middle of J&R
  Sect. 5.6's 40-60 % of V_p; shear loss parameter as the generic rock the notes attribute to Olson et al. (not
  re-read here)"; `bottom.reflection` (and its twins `propagation.rayleigh_reflection`, `libs/sim/src/paths.cpp`)
  take the fluid-solid coefficient when the row has shear (J&R (9.34)-(9.40); our check form
  R = (Z_p cos^2 2t_s + Z_s sin^2 2t_s - Z_1) / (... + Z_1), Z = rho w / k_z); `sim/bellhop.py` writes the shear
  column. `pebbles` stays a fluid (no source for its shear; [N1] Sec. 2.7: no gravel geoacoustic data in the book).
  The Lambert level stays -16 dB (D-88) with its range -23 to -10 dB stated, and J&R's own rock fit -5.6 dB
  recorded with the book's caveat that its rock data exceed the energy bound.
- Reference: J&R Sect. 5.6, p. 169 (printed range V_s 40-60 % of V_p; [N1] confirmed on the page image) and
  (9.34)-(9.40); the values chosen inside the range are a class choice (ours).
- Before -> after per bounce (computed, `rock.py`; our row rho 2.5, nu 2.5, delta 0.0137): fluid 0.01 / 0.02 /
  0.03 / 0.03 / 0.03 / 0.03 / 0.06 / 2.80 dB at 5 / 10 / 20 / 30 / 40 / 45 / 60 / 90 degrees -> elastic (0.5 V_p,
  shear delta 0.085) 1.55 / 2.86 / 5.29 / 13.68 / 8.96 / 6.27 / 4.52 / 2.80 dB (peak 16.7 dB at 35.7 degrees). With
  shear delta 0.1 (the [N3] case): 1.81 / 3.35 / 6.13 / 15.43 / 9.24 / 6.41 / 4.39 / 2.80. Over the book's range
  at 10 degrees: 6.83 (0.4 V_p), 3.35 (0.5), 2.05 (0.6) dB.
- Exposure: no shipped scenario has a rock or pebble bottom (`survey.py`). Tests that read the rock row:
  `test_sediment_table.py`, `test_sediment_literature.py`, `test_surface_models.py::test_reverberation_item_with_a_named_rock_is_lamberts_law_at_the_tabulated_value`
  (Lambert unchanged), `test_jackson_2007_bottom.py::test_flat_interface_reflection_is_8_49_to_8_53` (its fourth
  case is the rock row as a fluid: keep it as a fluid case by giving the numbers, not the row). Both simulator
  engines (`test_sim_cpp.py` parity). The live chain's reflection (`libs/live/src/raytrace.cpp`,
  `coverage.cpp`, `raytrace.py`, `lloyd.py`) only if the owner wants elastic bottoms in the chain's depth models:
  not in this item. GPU: the suite.

**A9. Buried targets: interface transmission and attenuation become the default** (G-3, its own agent, which will
also switch `buried_cylinder` and `buried_objects`). Before -> after from [N3] WP1 (agents' computation): targets
-6 dB two-way at 20 kHz, -9 dB at 30 kHz for 0.3 m of medium sand; the G-3 agent brings the scenes' own numbers.
Nothing to add here except: the scenes give a bare speed (1700 m/s); naming a sediment row is part of that switch.
**Applied 2026-11-26: `685ef49` (opt-in `penetration: fluid`) and `0935d88` (both buried scenes switched, the owner's decision); D-212.**

**A10. Sediment fit tool: small-slope as its default roughness model** (set S6, after G-9). `tools/fit_sediment.py`
and `sim/sediment_fit.py` default `roughness_model` -> `small_slope` (the simulator's default, D-86). J&R used
perturbation for its own fits and shows the two overlap below about 70 degrees for medium sand ([N4] Sec. 2.5): at
fit angles below 70 degrees the fitted values move little; above, small-slope is the valid one. The error-bar
default stays the book's rule (G.2.3: sigma_dB = 10 log10(1 + 1/sqrt(N))), with any corrected rule opt-in.
Exposure: `tests/python/test_sediment_fit.py` (the G-9 agent's), `docs/CONFIGURATION.md` fit-tool rows.
**Applied 2026-11-26: `a2e7fbe`, D-210;** the tool also applies the row's grain-size prior by default with sigma2 up to 0.005 (`e841add`, `1269be6`; D-211). D-230: the library `sediment_fit.fit(..., sediment=NAME)` applies the same row and prior (opt-in; without the argument a call is unchanged), and the tool calls it (one implementation).

**A11. TR 0702 matrix wording** (set S7, after S2; `tools/sst_validation.py` writes `docs/figures/sst_validation.md`).
**Not applied (2026-11-26):** row "Sec. 6.4.2" of `docs/figures/sst_validation.md` still says "Chapman-Harris at the MEAN grazing angle", false since A3; it waits for a regeneration of that file.
- Row "Sec. 6.4.2": after A3, "the geometric mean of the two strengths, as TR 0702 states"; before A3 it should
  read "Chapman-Harris at the MEAN grazing angle; TR 0702's geometric mean is 0 to 5.9 dB lower for the angle pairs
  of `docs/references/CONFORMANCE.md` A3" (it now says "not checked numerically").
- Rows "Sec. 6.4.3, Table 6" and "Sec. 12.2-12.5 ... TR 0702's is McDaniel's": add "McDaniel 1986 was read and is
  not `McDanielSurface` (TR 0702 cites ARL-PSU TM 90-88, not available)".
- Row "Sec. 12.6": add "Dahl 2004 prints 7 m/s for the 2001 sets 6-8; TR 0702's Set 22 is not said to be one of
  them (asiaex_dahl_2004.yaml, A7)".

**A12. Documents the orchestrator owns** (set S7). Proposed text in section 3.3.
**Done 2026-11-26 by the documentation commits after the merge (D-225).**

**A13. Blocked or not applied, with the reason** (no change):
**First item done 2026-11-26** (Wagner 1967 read, the sign settled, C1 `e037abd`, D-223); the last item (the link budget's reverberation term) done as `sonar-w2w3` (D-221); the others kept as stated.
- Composite and combined roughness without Wagner's shadowing: blocked on Wagner (1967) (not read; J&R's printed
  (13.82) is suspected of a sign error in pi^(1/2)). Stays as it is. Exposure 0 (no shipped item uses `composite`
  or `combined`).
- The five FLS scenarios' Chapman-Harris at 110-345 kHz: kept, with the warning and the descriptions this branch
  added (scenario files and `docs/SCENARIOS.md`). The numbers (end to end through the simulator, surface
  reverberation only, one ping, change of level against Chapman-Harris per range block; `fls_run.py`), for
  `fls_120k_transit` (`fls_120k_cluttered_seabed` the same within 1.3 dB): 20-30 m (overhead, 42-90 degrees)
  capped CH -4.9 / -4.6 dB, `bubble_layer_surface` +16.2 / +14.0 dB (the facet term near normal incidence),
  `dahl_1997` the same; 30-60 m -2.5 to -2.1 / -3.9 to -2.9 / -8.0 to -7.0; 60-300 m 0 to -0.5 / +2.6 to +5.1 /
  +0.4 to +3.0 dB. `fls_170k_transit` and `fls_170k_traffic` (identical): 30-450 m capped -4.9 to -0.3,
  bubble layer -3.2 to +3.0, `dahl_1997` -7.1 to +2.8 dB. `fls_330k_transit`: capped -8.6 to -5.1, bubble layer
  +6.3 (20-25 m) then -6.2 to -2.1, `dahl_1997` +6.2 then -10.0 to -1.4 dB. Every alternative moves the level by
  several dB, in opposite directions at different ranges, and none is better founded: the bubble model was
  compared with data at 12-70 kHz only, no read source holds any surface model above 70 kHz ([NS] Sec. 3.6), and
  the cap (`cap: bubble_saturation`, an opt-in that exists) is a physical ceiling, not a measurement. My numbers do
  not argue against the owner's default.
- Chapman-Harris at 6-10 kHz (`flank_patrol`, `flank_sst_shallow`, `flank_sst_spread`, `sst_pursuit`,
  `popoto_far_mover`): just above 6.4 kHz (the law grows by 1.6-2.6 dB from 6.4 to 10 kHz at 5 m/s); no read model
  covers 6.4-12 kHz better. Kept; the descriptions say so.
- Volume scattering isotropic away from backscatter (J&R (14.10) gives a pattern from heterogeneity spectra, not
  from the class parameters): no class value exists to apply. Kept; opt-in G-15 when wanted. Exposure 0.
- Default Lambert parameter -29 dB (D-89): J&R's Ch. 12 fits at 10-140 kHz are -19.7 (mud) and -20.2 dB (sand),
  printed class fits from the same book. No shipped item relies on the default (every Lambert item sets `mu_db`).
  Recommendation: keep the default (a different frequency band's printed value), add the book's two fits to the
  warning text it already gives at 10 kHz and above. Low value.
- K shape against patch count: D-196 follows the primary paper (Abraham and Lyons 2002 eq. (19)) against J&R
  p. 433's "2 nu": the primary print wins; kept.
- The link budget and coverage model have no reverberation term (`sonar-w2w3`): not an alignment of a value but a
  missing term; J&R (G.11)-(G.13) print the monostatic seafloor reverberation equation it would need (the plan's
  suggestion). A new package touching both live-chain engines; not in this list.

---

## 2. The matrix

Columns: item; what the source says (note and section, the source's equation, table or page); what ours says or does;
verdict and how it was checked; exposure (grep of `configs/scenarios`, `tests`, `docs`); action. "FIXED HERE"
names the commit of section 3.

### 2.1 Seawater

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| `acoustics.absorption_francois_garrison` constants | J&R (B.6)-(B.12), p. 461 [N1 2.8] | same formula | MATCHES: retyped from [N1] and compared over 0-30 deg C, 30-40 PSU, 0-4000 m, 0.1-1000 kHz: max relative difference 2.2e-16 (`check1.py`) | default absorption everywhere | NONE |
| F-G docstring "Valid 100 Hz to 1 MHz" | J&R p. 462: open ocean 30-40 ppt, "may not be appropriate" in low-salinity estuaries; no frequency range printed in what was read | the docstring claimed a frequency range with no source | DIFFERS (an unsourced range; the printed scope was missing) | docstring only | FIXED HERE (8b573ea) |
| F-G below 30 PSU (applied down to 5 PSU, boric dropped at 0) | outside the stated scope | ours | NO SOURCE READ for brackish water; would be confirmed by a print covering estuarine salinity | `fjord_*`, `ssp_*` if their salinity < 30 (not checked) | NONE |
| `absorption_ainslie_mccolm` | named only, J&R p. 461 | Col (2010) print | NO SOURCE READ beyond the via-secondary print; `test_literature_models.py::test_ainslie_mccolm_relaxation_frequencies_as_printed` holds the secondary | flank scenarios with `absorption: ainslie_mccolm` | NONE |
| `absorption_thorp` | TR 0702 eq. (42) (confirmed earlier, PHYSICS_NOTES Sec. 6.1) | same | MATCHES: `test_medium_models.py::test_thorp_absorption` (0.78 dB/km at 8 kHz; recomputed 0.780) | opt-in only | NONE |
| Thorp 100 kHz warning | TR 0702 states no upper frequency | a test comment said "outside Thorp's range" | DIFFERS (comment attributed our limit to the source) | `test_medium_models.py` | FIXED HERE (359ff57) |
| `water_density` | J&R (B.2)-(B.3), p. 456 (EOS 80, 1 atm) | arlpy coefficients | MATCHES: retyped, max 2.3e-13 kg/m3 over -2..40 deg C, 0..42 PSU (`check1.py`); no pressure term (+0.45 kg/m3 at 100 m, [N1 3.3]) | densities of all scenes | NONE (G-24 opt-in for pressure) |
| `sound_speed` (Mackenzie 1981) | not printed; Medwin (4.26), Del Grosso (B.5) are | Mackenzie | NO SOURCE READ for Mackenzie itself; ours within 0.253 m/s of Medwin (4.26) over 0-30 deg C, 30-40 PSU, 0-500 m (`check1.py`), within 0.39 m/s of Del Grosso ([N1 3.3]) | every environment from T, S | NONE; G-24 for check oracles |
| Brackish blend below 5 PSU | none | ours (D-146) | NO SOURCE READ (a pure-water pressure equation would confirm) | none shipped (D-146) | NONE |
| Seawater docstrings citing arlpy only | J&R prints both | docstrings named the secondary only | DIFFERS (citation) | docstrings | FIXED HERE (8b573ea) |

### 2.2 Reflection and bottom loss

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| `bottom.reflection` (fluid half-space) | J&R (8.49)-(8.53) [N3 2.2.3] | same | MATCHES: `test_jackson_2007_bottom.py::test_flat_interface_reflection_is_8_49_to_8_53` (1e-13) | every fluid bottom | NONE |
| `propagation.rayleigh_reflection` (the eigenray twin) | the same | same, attenuation in dB per wavelength | MATCHES: equal to `bottom.reflection` to 3.1e-16 for four bottoms 0.5-90 degrees (`check2.py`) | all image-source and Bellhop scenes | NONE (docstring still says "the dB/wavelength conversion is ours": proposed text 3.3; not edited, G-3 owns the file) |
| dB per wavelength to delta | J&R (2.6), (8.18), (8.25)-(8.27) | delta = alpha / (40 pi log10 e) | MATCHES: `test_loss_parameter_conversions_are_the_books` | all bottoms | REFERENCES status FIXED HERE (487c8b8) |
| C++ twins (`libs/sim/src/paths.cpp:140`, `libs/live` raytrace/coverage) | the same | same | NOT CHECKED here (C++ equality is the parity tests' job: `test_sim_cpp.py`, `test_raytrace_cpp.py`) | as above | NONE |
| Critical and intromission angles | J&R Fig. 8.3, (8.56)-(8.57) | same | MATCHES: `test_fig_8_3_critical_and_intromission_angles` | - | NONE |
| Coherent reflection exp(-Gamma^2 / 2) | J&R (13.14), p. 335 | `rough_coherent_reflection` | MATCHES: 0.24569 amplitude at 30 degrees, 40 kHz, 1 cm (`check1.py`, power 0.06036); `test_coherent_reflection_is_13_14` | `lloyd.py`, time spreading | REFERENCES status and docstring FIXED HERE |
| `rock` / `pebbles` reflection | elastic: J&R (9.34)-(9.40), Sect. 5.6 V_s 40-60 % V_p [N1 2.7], [N3 3.3] | fluid (nu 2.5 / 1.8) | DIFFERS: under 0.1 dB per bounce below 60 degrees against 1.6-16.7 dB elastic (A8 numbers, `rock.py`) | no shipped scenario | A8 |
| Note contradiction C1 (rock loss at 40-60 degrees) | [N1 3.4] 4.8-5.5 dB; [N3 3.3] 9.2 / 6.4 / 4.4 dB | - | MATCHES both notes once the unstated input is known (recomputed): [N3]'s are with shear delta 0.1 (9.24 / 6.41 / 4.39), [N1]'s with lossless shear (4.81 / 5.03 / 5.35); not an error, an unstated input (`rock.py`) | - | recorded in REFERENCES Sec. 11 (487c8b8) |
| Sediment shear (sands) | J&R Sect. 5.2, V_s 8-150 m/s | none | MATCHES within the book's own statement (under 0.2 dB, [N1 3.4], not recomputed) | - | NONE |
| `sediment` tier above critical (buried scenes) | transmission (8.55), attenuation (8.20)/(8.26), (8.68) [N3 3.5] | unit amplitude, no attenuation | DIFFERS: about -6 / -9 dB two-way at 20 / 30 kHz, 0.3 m ([N3 WP1], agents' computation; not recomputed: file in flight) | `buried_cylinder`, `buried_objects` | PACKAGE G-3 (in flight); A9 |
| `sediment` tier below critical | evanescent plus roughness (15.1)-(15.16) [N3 2.7] | a ray at the critical angle | DIFFERS (+20 to +30 dB, [N3 3.5]) | 0 shipped | PACKAGE G-13 |
| Constant delta (no dispersion) | (I.28) causal; SAX99 flat above 20 kHz | constant | DIFFERS by +0.24 dB, +2.1 degrees per decade for sand ([N1 3.4]); the book says data favour flat above 20 kHz | all bottoms, small | PACKAGE G-29 (opt-in); NONE by default (the data side with ours) |
| Sand as a simple fluid vs EDFM | (10.82), Table 10.1 | fluid | DIFFERS 1.0-1.6 dB loss for SAX99 sand ([N3 3.4]); must not be applied to TR 9407 rows (double count) | - | PACKAGE G-10 (opt-in, new row only) |
| Near-field validity of plane-wave coefficients | App. J.2 | plane-wave | NOT CHECKED ([N3 3.6] says the image-source distances satisfy it; not recomputed) | - | NONE |

### 2.3 Bottom roughness scattering

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| Perturbation (13.21)-(13.25) | J&R [N4 2.3] | `perturbation_strength` | MATCHES to 1e-13: `test_perturbation_kirchhoff_and_small_slope_are_the_books` | `perturbation` option | REFERENCES FIXED HERE |
| Kirchhoff (13.10), (13.12)-(13.17) | J&R; [D99] (1), (4) for the surface | `kirchhoff_strength` | MATCHES to 2e-7 (G-1 test); [D99] (4) to 0.001 dB ([NB 3.1]) | option | REFERENCES FIXED HERE |
| Small-slope (13.11) | J&R; [D99] (5)-(6) | default `small_slope` | MATCHES (G-1 test); [D99]'s g identity to 7e-16 ([NB 3.1]) | default for every `fluid_sediment` item (4 shipped files) | NONE |
| Default = book's SSA + volume | J&R tables [N4 3.2] | same | MATCHES under 0.001 dB: `test_default_model_equals_the_books_small_slope_plus_volume` | as above | NONE |
| Structure constant (D.11)-(D.13), `w2_from_cm4` (D.1.3) | J&R [N2 2.1] | same | MATCHES: `test_roughness_spectrum_conventions_structure_constant_and_unit_conversion` | all | NONE |
| (D.19), (D.23) as printed | misprinted (gamma2 - 2 for gamma2 - 1; F^(gamma1 - 1)) [N2 2.1] | not used | SOURCE MISPRINT (ours does not use them; quadrature to 1e-10 in [N2]). 2026-11-26 later: the corrected forms are confirmed by Jackson and Briggs (1992) eq. (4) (Table V from Table III to 0.1 % in 6 of 7 rows) | - | NONE (G-18 must use the corrected forms). D-226: `bottom.roughness_2d_from_1d` ([JB] eq. (4)), G-18 in part |
| Cut-off (13.72)-(13.75), large-scale slope (13.76)-(13.77) | J&R, attributed to D. Jackson et al. 1986a | `large_scale_slope` | MATCHES: per-axis rms slopes 0.0227 / 0.0595 / 0.1432 of Fig. 13.19 (`check2.py`: 2 s^2 = 0.00103, 0.00707, 0.0410); `test_composite_cut_off_and_large_scale_slope_13_72_to_13_77`. 2026-11-26 later: three prints differ: Mourad and Jackson (1989) eq. (18) implies 2^(-1/2), Jackson and Briggs (1992) text 1 and (A8) 2^(-1/2); at 5 degrees `composite` moves -1.2 to -1.4 dB (2^(-1/2)) or -2.0 to -2.8 dB (1); ours 1/2 kept | `composite`, `combined` | REFERENCES FIXED HERE |
| Composite shadowing (13.81)-(13.83) | J&R prints Wagner's factor | none; 2026-11-26 later: Wagner's R inside `composite` and `combined` | DIFFERS: ours +0.9 to +6.5 dB at 5 degrees ([N4 3.4]; computed with the misprinted pi^(+1/2)); ours re-computed here: medium sand -41.6 / -32.9 dB at 30 / 100 kHz, sandy gravel -21.4 dB at 100 kHz (`check2.py`, the plan's "before"). 2026-11-26 later: MATCHES Wagner (1967) (49), (20)-(22) with pi^(-1/2): -41.58 -> -41.69, -32.88 -> -33.56, -21.37 -> -23.75 dB; -0.05 to -2.39 dB at 5 degrees for sands and gravel (the plan's -42.5 / -35.9 / -27.8 used the misprint) | 0 shipped (all four `fluid_sediment` items use the default) | A13: blocked on Wagner 1967. 2026-11-26 later: DONE (C1, `e037abd`, D-223; `shadowing: none` the old models) |
| (13.82) pi^(+1/2) | probably pi^(-1/2) [N4 6] | - | SOURCE MISPRINT (suspected; not settled). 2026-11-26 later: SOURCE MISPRINT, settled: Wagner (1967) eq. (22) and Mourad and Jackson (1989) eq. (25) print pi^(-1/2) | - | blocked (G-11). 2026-11-26 later: done (D-223) |
| Composite validity (13.78) | sigma_sL < 0.1, theta < 70 degrees | not checked in code | DIFFERS (no warning; [N4 3.1] values 0.157-0.367 exceed it) | 0 shipped | PACKAGE G-11 warning part (opt-in, moves nothing) |
| Composite docstring "the recipe below is ours" | J&R (13.80) has the structure | - | DIFFERS (docstring outdated) | docstring | FIXED HERE (8b573ea) |
| `combined` | not in any source | ours | NO SOURCE READ (Williams-Jackson interpolation or Mourad-Jackson would replace it). 2026-11-26 later: DIFFERS: the Williams and Jackson (1998) join (eq. (10), eta = -2) is printed; ours (small-slope based) is -0 to +6.8 dB above it at 20-45 degrees for sands | 0 shipped | NONE. 2026-11-26 later: opt-in Williams-Jackson model (`roughness_model: williams_jackson_1998`), merged as an opt-in in Python (D-226); small-slope stays the default (D-222, D-229) |
| Interpolation Kirchhoff/composite | not printed (J&R p. 349; [McD93] none) | small-slope in its place (D-86) | NO SOURCE READ (the book recommends SSA, p. 336). 2026-11-26 later: printed in Mourad and Jackson (1989) eqs. (33)-(37) and, for the bistatic case, Williams and Jackson (1998) eq. (10): DIFFERS by design (small-slope kept as the default, the owner); the 1989 model is +1.6 to +11.5 dB above ours at 5 degrees for sands and gravel | - | NONE. 2026-11-26 later: the 1989 model as an opt-in (C4, `deee6f8`, D-223) |
| Composite / small-slope with gamma2 > 3.7 (2026-11-26 later: new row) | Jackson and Briggs (1992) p. 974: such exponents are unreliable | no warning | DIFFERS: +5.6 dB at 30 degrees, +10.3 at 40 against across strike at Quinault along strike (gamma2 3.92, 35 kHz; ours) | the Quinault along-strike values are only in a comment | a warning ([JB] 5.3): done, once per model, for an exponent between 3.7 and 4 (D-226; no shipped row is above 3.67) |
| (L.34)/(L.35) high-frequency Kirchhoff | J&R prints two forms; [McD86]+erratum and energy settle (13.62)/(13.63) | `reverb_specular.facet_lobe` = (13.62)/(13.63) | SOURCE MISPRINT ((L.34)/(L.35); ours follows (13.62)/(13.63), 5e-13 dB, [NB 3.4]) | `exclude_specular` (0 shipped) | REFERENCES Sec. 11 FIXED HERE |
| Medium sand row = Fig. 13.2 | J&R Fig. 13.2 "typical medium sand" | `medium_sand` | MATCHES: `test_medium_sand_row_is_fig_13_2` | `sst_sediment_reverb`, `sas_auv_multipath` | scenario comment FIXED HERE (1936f86) |
| Reciprocity, energy bound (2.18), (2.20) | J&R | small-slope | MATCHES (hemisphere 0.902-1.031 of |V|^2, below the bound, G-1) | - | NONE |
| Frequency law f^(4 - gamma2) (13.61) | J&R | perturbation | MATCHES: `test_perturbation_factor_identity_k_5_and_the_frequency_law_13_61` | - | NONE |

### 2.4 Sediment volume

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| Volume term (14.7)-(14.9) with (8.26) | J&R [N4 2.9] | `volume_strength` | MATCHES to 1e-11 (G-1); Fig. 14.2 by eye within 1 dB | all `fluid_sediment` items | REFERENCES FIXED HERE |
| Volume term isotropic in bistatic angle | J&R (14.10) pattern from heterogeneity spectra | isotropic | DIFFERS: +4 dB (rho_rk -1) or -10.6 dB (rho_rk 0) at 20 degrees forward ([N4 3.6]). 2026-11-26 later: Williams and Jackson (1998) (21)-(22) is (14.10) with rho_rk = -1 and their Table I mu negative; an Eckernfoerde volume null at phi = 39.3 degrees for 10/10 degrees (ours; Fig. 8 dips there) | 0 shipped bistatic sediment items | A13 (no class value); PACKAGE G-15: done in part (D-226: `volume_model: williams_jackson_1998`, opt-in, rho_rk = -1 only, refused for class rows; the general spectral form open) |
| Volume slope and shadow corrections | Sect. 14.1.2; Siemes 2009 | none | DIFFERS (not quantified). 2026-11-26 later: quantified at 5 degrees: +3.9 / +9.6 dB for medium sand at 20 / 100 kHz, +7.9 / +12.2 dB for sandy gravel, under 0.3 dB for silts; Mourad and Jackson's printed recipe (38) available as the opt-in `volume_model: mourad_jackson_1989` (monostatic) | - | PACKAGE G-16 (after G-11). 2026-11-26 later: done in part (C4, D-223); a bistatic slope-averaged volume term open |
| Mud sigma2 0.001 (7 rows, 5.5-9 phi) | printed class value (Snellen 2011, 2013; arXiv:1809.08459); sites imply 0.0008-0.0037 (J&R Table 14.1, agents' (14.10)), Gallaudet fit 0.003 | 0.001 | MATCHES the printed class value; the sites differ by -1.0 to +5.7 dB (range, not a class value). 2026-11-26 later: Jackson and Briggs (1992) Table V prints fits of 0.005 (Arafura) and 0.002 (San Francisco), +7.0 / +3.0 dB against 0.001, one site each, superseding the agents' derived 0.0031 / 0.0037; the range is [0.0008, 0.005] | 0 shipped rows 3 phi or finer (G-5 doc) | A6 (range and named site rows; values kept): applied (D-209, D-224) |
| Gallaudet 0.003 as evidence | his two band means | - | NOT CHECKED here (the fit-tool agent's finding, recorded): cannot separate volume from roughness; independent only under volume dominance | - | A6 note |
| Medium sand sigma2 0.001 (D-90) | no class value printed; range 0.0001-0.004 (J&R p. 382) | 0.001 (arXiv:1809.08459) | NO SOURCE READ for a class value; inside the printed range. 2026-11-26 later: Jackson and Briggs (1992) fit 0.001 at Quinault fine sand (one site; the fit insensitive): weak support | `sst_sediment_reverb` | NONE |

### 2.5 Sediment table and grain-size relations

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| `GRAIN_SIZE_TABLE` 20 rows | J&R (11.1)-(11.3) at the points | table | MATCHES (rho 5e-4, nu 1e-4, delta 0.5 %): `test_grain_size_table_is_11_1_to_11_3_at_its_points`, `test_the_relations_reproduce_every_tabulated_row` | named sediments | NONE |
| (11.2) constant 1.0091 | printed; 1.0019 meant (continuity, rows) | 1.0019 | SOURCE MISPRINT: `test_density_and_speed_are_continuous_at_the_joins_and_1_0091_is_not_used`. 2026-11-26 later: confirmed by the original, Mourad and Jackson (1989) eq. (2): 1.0018916; the relations take its seven-digit constants (C5, `6b3bafb`, D-223) | - | NONE |
| (11.3) constant 0.4566 | printed; 0.4556 meant (rows at -1, -0.5 phi agree to 2.5e-4 against 2.4e-3) | 0.4556 | SOURCE MISPRINT (G-6) | - | NONE |
| (11.3) steps at 2.6, 4.5, 6.0 phi (C2) | printed pieces | kept | MATCHES the print: +0.0022, -0.0009, +0.0013 (`test_the_printed_steps_of_11_3_contradiction_c2`); C2 settled for [N1] | - | NONE |
| Interpolation between rows (default) | the relations | linear in the table; 2026-11-26 later: the relations (A5) | DIFFERS up to 0.68 dB (5.3 phi). 2026-11-26 later: MATCHES (the default since A5) | no caller (grep) | A5: applied (`efd2aff`, D-209) |
| Fig. 11.2 normal-incidence loss | figure | relations + fluid reflection | MATCHES within figure reading: 6.12 / 14.77 / 22.15 dB at 0 / 3 / 5 phi (`test_fig_11_2_normal_incidence_loss`) | - | NONE |
| Gallaudet Table VI ratios = our grain size | printed 1.1483 / 0.9864 / 0.0033 | 1.14828 / 0.986436 / 0.003284 | MATCHES within 4e-5 (`test_table_vi_geoacoustic_ratios_are_our_grain_size_interpolation`; recomputed). 2026-11-26 later: with `method="table_linear"` only; the default relations miss rho by 5.08e-5 (A5) and 7.53e-5 (C5), delta by 3.3e-5 | - | A5 changes rho by 3e-5 (applied; the check of the relations is 8e-5 since C5, a widening) |
| w2 rule for silts and clays | Briggs' spectra (J&R Table 6.1) | Snellen 2013 rule | DIFFERS: 1-7 dB below at 10-100 kHz ([N2 3.2]); under 0.45 dB on the total below 60 degrees (G-5 doc) | 0 shipped | A6 (range note; rule kept) |
| Sands' w2 | Table 6.1 median | rule | MATCHES within 1.4 dB ([N2 3.2], not recomputed) | many scenes | NONE |
| Gamma2 3.25 for all | Snellen 2011 "works well" | 3.25 | MATCHES the secondary; NO SOURCE READ for a per-class value. 2026-11-26 later: MATCHES the original, Mourad and Jackson (1989) eq. (5) | all | NONE |
| Mourad and Jackson (1989) eq. (39), the volume term (2026-11-26 later: new row) | prints abs(1 - R)^2 | abs(1 - R^2)^2 (`volume_strength`) | SOURCE MISPRINT: the paper's own figures and J&R (14.8) use our form; ours equals the corrected (39) to 2e-14 dB | all `fluid_sediment` items | C3 pins it (`8fd1ec3`) |
| sigma2 step at 5.0 phi ([MJ89] (4)) against 5.5 phi (ours, Snellen) (2026-11-26 later: new row) | [MJ89] (4) steps at 5.0; its own Fig. 8 and the later prints use 5.5 | 5.5 | DIFFERS for `sandy_silt` only (-3.0 dB) | 0 shipped | NONE (kept) |
| Named site rows (2026-11-26 later: new row) | Jackson and Briggs (1992) Table V; Williams and Jackson (1998) Table I | `jackson_briggs_1992_quinault`, `_arafura`, `_san_francisco`, `williams_jackson_1998_panama_city` (sigma2 ours) | MATCHES the prints; our default model is within 0.2 dB of [JB] Fig. 15 (San Francisco) and 1 dB at Quinault and Arafura where the notes say it agrees; Panama City +5.8 dB above [WJ] Fig. 7 at 20 degrees (the published join) | used only when named | [JB] 5.1, 5.2, [WJ] 5.5 applied (`249cb07`, `9585adc`, D-224); Eckernfoerde not added (no printed sigma2) |
| Table 5.3 / 5.4 / 5.6 / (5.8) / Fig. 5.28 / Fig. 11.2 caption misprints | [N1 6] | not used | SOURCE MISPRINT (none used by us) | - | NONE (G-23 must avoid them) |
| `rock` row ratios 2.5 / 2.5 | Cintron Table 1; J&R p. 169: igneous 5-6 km/s, sedimentary 2-6 | 3820 m/s | MATCHES the secondary's class value; within J&R's sedimentary range | 0 shipped | NONE |

### 2.6 Sea-surface backscatter

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| `chapman_harris_db` formula | [McD93] eq. (27) | same | MATCHES to 1e-9 dB: `test_surface_papers_1997.py::test_chapman_harris_is_mcdaniel_eq27`, exponents 2.46 / 0.82 `::test_chapman_harris_exponents_at_30_degrees` | 18 shipped scenarios | NONE |
| Chapman-Harris range | 400-6400 Hz, to 15 m/s, "low grazing angles" | docstring correct (G-2); REFERENCES row "unconfirmed, below about 40 degrees"; CONFIGURATION.md "below about 40 degrees", "not confirmed"; PHYSICS_NOTES 23.3 "could not be confirmed" | DIFFERS (three documents stale) | documents | FIXED HERE (487c8b8, b42e1b7) |
| Chapman-Harris used at 110-345 kHz | nothing above 70 kHz in any source | 5 FLS scenarios | DIFFERS from its range (an extrapolation; alternatives -10 to +16 dB, A13) | `fls_120k_transit`, `fls_120k_cluttered_seabed`, `fls_170k_transit`, `fls_170k_traffic`, `fls_330k_transit`; tests `test_realistic_scenes.py` (those scenes), `test_detect.py`, `test_lfm_range_doppler.py`, `test_classify_rules_eval.py`, `test_simcal.py`, `test_taps.py`; gallery JSON | A13 kept; descriptions FIXED HERE (1936f86, 42f90fc) |
| Chapman-Harris at 6-10 kHz | range to 6.4 kHz | 5 scenarios | DIFFERS slightly (+1.6 to +2.6 dB growth 6.4 -> 10 kHz, `slope_fls.py`) | `flank_patrol`, `flank_sst_*`, `sst_pursuit`, `popoto_far_mover` | A13 kept; descriptions FIXED HERE |
| Chapman-Harris 4-6 kHz | inside range | 8 scenarios | MATCHES the stated range | `popoto_12km*` (4), `popoto_env_*` (4) | NONE |
| Bubble term (eq. 10) | [D97] (17)-(19), [McD93] (9)-(12) | `bubble_backscatter` | MATCHES to 1e-9: `test_bubble_term_is_d97_eq17_and_mcdaniel_eq9`, `test_bubble_term_limits_d97_eq18_eq19` | 0 shipped | NONE |
| Damping 2.55e-2 f_kHz^(1/3) | [D97] 2.55e-3 f_Hz^(1/3); [D96], [D99] 0.0792 at 30 kHz | same | MATCHES: 0.079234 at 30 kHz (`check1.py`); `test_damping_constant_is_d97_in_hz` | - | NONE |
| delta_r 0.0136 | [D96], [D97], [D99], [McD93] | 0.0136 | MATCHES | - | NONE |
| Saturation level | [D97] Table I -26.9 / -27.8 / -28.3 | -26.82 / -27.74 / -28.22 | MATCHES within 0.1 dB: `test_saturation_level_against_d97_table_i` | - | NONE |
| Facet term | [McD93] eq. (7), sigma_x'^2 = mss/2 | `facet_backscatter` | MATCHES to 1e-6 dB: `test_facet_term_is_mcdaniel_eq7` | - | NONE |
| Facet near normal incidence | Nuetzel 20 kHz (figure, [McD93] Fig. 20) | facet | DIFFERS 3.3-4.3 dB low at 82.5-88 degrees (figure, one data set, [NS 3.1]) | 0 shipped | PACKAGE G-22 (record only; McD86 does not settle it, [NB 4]) |
| Optical-depth convention | [D97] Sec. VI one crossing | same | MATCHES: `test_optical_depth_convention_is_d97_section_vi` | - | NONE |
| beta_I default `apl_1994` | [D97] Table II fit; [D99] <= 0.0012 at 30 kHz, 4 m/s | `apl_1994` | DIFFERS: +5.7 to +10.1 dB at 2-3 m/s (26.5 kHz) ([NS 3.2]); at 30 kHz, 4 m/s ours 0.00395 against 0.00142 (D97) and <= 0.0012 (D99) (`check1.py`) | 0 shipped; Gallaudet tests | A1 (opt-in exists: `extinction: dahl_1997`) |
| Wind exponent | [D97] U^5.3-6.4 | U^3.9-4.25 | DIFFERS (G-7 test `test_wind_exponent_at_17_degrees_against_d97`) | as above | A1 |
| Facet bubble attenuation | [D97] Sec. VI, [D99] (11)-(12) | none | DIFFERS 0 to -3.2 dB at 60-89 degrees ([NB 3.2]) | 0 shipped | A2 |
| Bistatic surface rule | TR 0702 Sec. 6.4.2 geometric mean; [D99] (13) | arithmetic mean angle | DIFFERS 0 to +5.9 dB (computed, `slope_fls.py`) | `sst_pursuit` | A3 |
| [D99] bistatic bubble term (13) vs [D97] (17) at backscatter | 1.76 dB apart (weak limit) | - | NOT CHECKED here: the two published forms do not join ([NB 3.2], not recomputed) | - | NONE |
| Small-scale roughness term | [McD93] (4), (8), (22)-(25); handbook G(U) not printed | omitted | NO SOURCE READ for G(U); [McD93]'s spectrum version contradicted by data at low wind ([NS 3.5]) | 0 shipped | PACKAGE G-20 (diagnostic) |
| "Lower bound" wording | holds at high wind only [NS 3.5] | surface.py corrected (G-7); CONFIGURATION.md "lower bound on the full model ... not on the sea" | MATCHES (both say what the note says) | docs | NONE |
| Coastal vs open ocean | about 10 dB ([McD93] p. 1917), no formula | none | NO SOURCE READ (no printed law) | - | PACKAGE G-21 |
| `kraus_surface_db` | Morency-Stilwell Sec. II-F, no range | same formula | MATCHES the print (`test_kraus_surface_formula_as_printed`); NO SOURCE READ for a range | 0 shipped | NONE |
| `searay_bottom_db` | same source | same | MATCHES the print (`test_searay_bottom_formula_as_printed`); range NO SOURCE READ | 0 shipped | NONE |
| Resonant radius | [McD93] 3.2/f, [D97] 3.25/f | Minnaert | MATCHES within 1.5 %: `test_resonant_radius_against_both_papers` | - | NONE |
| [McD93] eqs. (22), (23), (24) | misprints [NS 2.5] | not used | SOURCE MISPRINT (not used) | - | NONE |

### 2.7 Bubbles (loss per bounce, populations)

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| Loss per bounce convention exp(-2 beta_I / sin g) | [D97] Sec. VI; [D99] (12); [D96] (17) prints twice | ours | MATCHES [D97]/[D99]; [D96] (17) SOURCE MISPRINT by [D99]'s own statement: 2.09 dB vs 4.18 dB at beta 0.05, 12 degrees (`check1.py`) | `time_spread` bubble laws | comment FIXED HERE (8b573ea) |
| beta_I against forward-scatter papers | [D96] 0.05-0.10 at 7-10 m/s (30 kHz); [D04] 0.009 / 0.004 at 20 / 8 kHz, 7 m/s | `apl_1994` 0.055 (7 m/s, 30 kHz), 0.097 (10), 0.039 / 0.018 (7 m/s, 20 / 8 kHz) | MATCHES [D96]; DIFFERS from [D04] by x4.4-4.5 (6.4 dB) (`check1.py`); one site each | `apl_1994` bubble loss users (`lloyd.py`, time spreading with a model: 0 shipped; flank scenes use a fixed 0.5 dB) | NONE (site factor; G-21) |
| `apl_1994` loss law | arlpy / UnderwaterAcoustics.jl (via secondary), no range | same | MATCHES the secondaries; NO SOURCE READ for TR 9407 | `lloyd.py` default bubble model | NONE |
| Hall-Novarini population | arXiv:2603.23727 eqs. (20)-(23) | same | MATCHES: `test_hall_novarini_population_as_printed` | opt-in | NONE |
| `bubble_layer_surface_db` stated range 12-70 kHz, 3-15 m/s | Gallaudet (printed) | warns outside | MATCHES | - | NONE |

### 2.8 Forward scatter: time, angle, Doppler spreading

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| Rayleigh parameter, coherent split | [D96] eq. (6) chi = kH(sin + sin); TR 0702 | Gamma = 2 k h sin g | MATCHES (identical definition, [NF 3.1]) | all time spreading | NONE |
| Facet geometry: across-path spread | [D96] eq. (15) q = 1.348 | q = sqrt(2) | MATCHES within 4.9 % (closed form, `check1.py`) | `angle_spread` (`flank_sst_spread`) | docstring FIXED HERE |
| Coherence-length scaling | [D04] (7)-(9) | same scaling | MATCHES the form ([NF 3.1]) | as above | NONE |
| Slope for forward scatter | [D96] (A7) large-scale; [D04] 0.0231 low-pass at 7 m/s; [D99] 0.017 at 4 m/s | Cox-Munk total | DIFFERS: total/A7 = 1.50, 1.36, 1.45, 1.57, 1.69, 1.82, 1.90, 2.20 at 1.5, 3, 4, 5, 6, 7, 7.6, 10 m/s (`check1.py`) | 3 scenarios (A4) | A4; OPT-IN EXISTS (`time_spread.surface.mean_square_slope`, documented FIXED HERE b42e1b7) |
| [D99] implied slope 0.008-0.011 at 4 m/s | figure / agents' computation | - | NOT CHECKED here (figure readings and agents' computation); recorded as the open contradiction to (A7) | - | A4 note |
| Decay time | [D96] Fig. 2, [D04] Fig. 3c (figures) | facet default (D-85) | DIFFERS: facet 4-10 dB too slow, rule of thumb 8-16 dB ([NF 3.3], agents' computation, not recomputed) | 3 scenarios | A4 (slope part); PACKAGE: [NF] WP3 (exact Kirchhoff geometry, opt-in) |
| Rule of thumb against facet | data reject the rule of thumb | `decay_time: facet` default | MATCHES the direction of the data (facet better) | - | NONE |
| Vertical coherence at low grazing | [D04] Fig. 5 (figure) | facet | DIFFERS about x2 short (0.03-0.2 against 0.69 at 0.30 m; [NF 3.2]) | - | A7 records it (expected failure); blocked on [D99]'s spectrum |
| Doppler width 2 f v sin g / c | TR 0702 p. 101 prints 1/sin g; no paper gives one | ours | NO SOURCE READ for either reading (Dahl 2001 or TR 9407 wanted); TR 0702 p. 101 SOURCE MISPRINT is our reading only (PHYSICS_NOTES Sec. 11 item 23) | all Doppler spreading | NONE |
| Scintillation of forward arrival | [D96] "approximately exponentially distributed" (one set) | complex Gaussian taps (SI 1) | MATCHES that one set | - | NONE |
| Bistatic bubble scatter in forward scatter | [D96] (18), [D99] (13) | not modelled | DIFFERS; marginal at 7-8 m/s per both papers | - | NONE |
| `exclude_specular` lobe vs [D99] SSA lobe | [NB 3.5] (stand-in spectrum) | facet lobe | DIFFERS: 2.9 dB below at specular, over-removes 0.5-3.5 dB a few degrees off (agents' computation with a stand-in spectrum) | 0 shipped | NONE (no published spectrum) |
| Rms wave height (PM) | [D04] 0.3 m at 7 m/s; [D96] 0.5 m (swell); [D99] 0.16 m at 4 m/s | PM 0.262 / - / 0.086 m (`check1.py`) | DIFFERS (swell, fetch): supports the `rms_height` key (opt-in exists) | `sst_asiaex` (3 m/s) | OPT-IN EXISTS (`time_spread.surface.rms_height`); A7 uses 0.3 m |

### 2.9 Reverberation statistics

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| K PDF (16.33)-(16.35) | J&R | `shape` = nu | MATCHES ([N2 3.4]; simulated P_fa) | K items (0 shipped reverberation items with K) | REFERENCES FIXED HERE |
| Patch count factor | J&R p. 433 "2 nu"; Abraham-Lyons eq. (19) n | n (D-196) | DIFFERS from the book by 2; MATCHES the primary paper (`test_our_shape_is_the_papers_alpha`, `test_the_papers_worked_example` 1.685 / 2.694 dB) | opt-in densities | NONE (primary wins); REFERENCES Sec. 11 FIXED HERE |
| Cell-size dependence | McDaniel 1996 (16.59)-(16.69) | large-cell limit | DIFFERS (no saturation for small cells) | SAS, high-frequency FLS if K is used (0 shipped) | PACKAGE G-12 |
| (16.55) Gamma(n/s) | misprint for Gamma(n/2) | not used | SOURCE MISPRINT | - | NONE |
| [DP97] surface statistics | lognormal-modulated exponential | Rayleigh surface reverberation | DIFFERS (structural) | all surface items | PACKAGE G-19 |
| TR 0702 spread-coherence relations | eqs. (89)-(103) | generator | MATCHES: `test_reverb_statistics.py` (7.971 vs 7.958 Hz etc., feature report) | - | NONE |

### 2.10 Noise

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| `wenz_spectrum_db` constants | three secondary prints (Hurt 2005 eqs. (2.7)-(2.9) etc.) | same | MATCHES the prints: `test_literature_models.py::test_ambient_noise_fit_against_a_third_print`; Coates (1990) / Wenz (1962) NO SOURCE READ | `model: wenz` scenarios | NONE |
| Wenz fit's frequency/wind range | not stated in what was read | none stated | NO SOURCE READ | - | NONE |
| Isotropic and Cron-Sherman coherence | Shajahan 2022 eqs. (3)-(4) | same | MATCHES: `test_surface_noise_coherence_against_the_printed_closed_form` | - | NONE |
| K noise (TR 0702 eq. 27) | kurtosis 3/shape | same | MATCHES: `test_white_k_noise_as_in_tr0702_has_the_k_kurtosis_and_the_asked_level` | fjord scenarios (K ambient) | NONE |

### 2.11 Targets and propagation

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| Kirchhoff target amplitudes | Lavia et al. arXiv:1805.06945 | `sim/targets.py` | MATCHES ([REFERENCES 7], `test_scene_targets.py`); not re-checked here | target scenes | NONE |
| Cylinder obliquity (Urick) | not read | Kirchhoff integral | NO SOURCE READ (Urick Table 9.1 wanted) | - | NONE |
| TR 0702 eqs. (36)-(41) image vectors | printed | `ImageSourceProvider` | MATCHES: `test_image_equations_as_printed_match_the_provider` (0 m, 0 s) | image-source scenes | NONE |
| Munk profile | Acoustics Toolbox | same constants | MATCHES the secondary; Munk 1974 NO SOURCE READ | Bellhop scenes using it | NONE |
| KRAKEN cross-check conditions | KRAKEN incoherent mode sum is not a reference at the source's depth (1.68-1.75 dB above the ray sum there, a factor 3/2; `docs/DECISIONS.md` D-197) | `test_kraken_crosscheck.py` uses 30 m / 50 m | MATCHES the condition (the test avoids the source's depth; checked by reading the test constants) | oracle tests | NONE |
| Layered seafloor | not in J&R Chs. 8-11; Olson-Jackson arXiv:2009.12420 | none | NO SOURCE READ in the book; PACKAGE G-14 | - | NONE |

### 2.12 Sonar equation, coverage, link budget

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| `SonarEquation` (noise-limited) | TR 0702 Sec. 2.5, 5.1 units; echo level check | SL - 2TL + TS | MATCHES: `test_sst_validation.py::test_echo_level_is_the_sonar_equation` (0.004 dB) | `sonar_equation.py`, tools | NONE |
| Reverberation term in link budget / coverage | J&R (G.11)-(G.13) monostatic seafloor reverberation | none (D-198) | DIFFERS (missing term; `flank_patrol` small vehicle 30 dB budget against about 4 dB simulated, DESIGN.md) | live coverage (both engines), `docs/DESIGN.md` budget | NEW FOLLOW-UP (`sonar-w2w3`, A13). 2026-11-26 later: the term done (D-221); the sites state their scattering and the plan and coverage include it, the echo's images add as pressures (`flank_patrol` worst margin error 3.4 dB, was 4.7; D-227); the free-field budgets stay noise-only (no site) |
| Isotropic array gain | closed form (D-103) | same | MATCHES (ours, tested) | - | NONE |
| Matched-filter gain, 3 dB per octave of bandwidth | TR 0702 Sec. 10.6.1 | same | MATCHES: `test_signal_to_reverberation_after_pulse_compression` (+3.29, +2.70) | - | NONE |

### 2.13 Live-chain physical constants

| Item | Source says | Ours | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| Sound speed in the chain | navigation or the record | from data, not a formula | MATCHES (no source-bound constant: checked by grep of `live.py`, `libs/live`) | - | NONE |
| Coverage TL 20 log R + alpha R, absorption from config | spherical spreading | config value | NO SOURCE READ for the configured absorption values (config facts) | live coverage | NONE |
| Chain's reflection twins (`raytrace.py:186`, `lloyd.py:190`) | J&R (8.49)-(8.53) | call `propagation.rayleigh_reflection` | MATCHES (same function, row 2.2) | depth models | NONE |
| Surface image (`lloyd.py`) roughness factor | J&R (13.14) | `rough_coherent_reflection` | MATCHES (row 2.2) | depth planner | NONE |
| Baseband sign | J&R (E.1) p = Re{P exp(-i w_c t)} | analytic signal exp(+i 2 pi F t): our baseband is the conjugate of the book's P | DIFFERS in convention only (envelope statistics unaffected; [N2 3.4]); PHYSICS_NOTES line 1151 states the conjugate against TR 0702 but Sec. 15 does not name J&R | phases coded from the book | proposed PHYSICS_NOTES Sec. 15 sentence (3.3); NONE in code |

### 2.14 Source-bound scenarios against their own source

| Scenario | Its source says | The file does | Verdict | Exposure | Action |
|---|---|---|---|---|---|
| `sst_asiaex` vs TR 0702 Sec. 12.2-12.6 | 8 kHz, 3 ms, about 500 m, 20 pings, five arrivals in 35 ms, 3 m/s (4 quoted), bottom rms 0.012 m, mss 0.045 | all of these; depths, c, sediment ours (`unconfirmed`) | MATCHES what TR 0702 prints: `test_asiaex_scenario_holds_what_tr0702_states` | its tests, figure, matrix | NONE |
| `sst_asiaex` vs Dahl 2004 | source 25 m, arrays 26 / 52 m, 467-500 m, 105 m, 7 m/s, H 0.3 m (sets 6-8) | 30 / 50 m, 500 m, 100 m, 3 m/s | DIFFERS; but TR 0702's Set 22 is not said to be [D04]'s sets 6-8, and [D04]'s layout breaks TR 0702's five-arrival description (computed, A7) | - | A7 (new scenario); comment and SCENARIOS row FIXED HERE (1936f86, b42e1b7) |
| `sst_asiaex` surface slope | (A7) 0.0135 at 3 m/s | Cox-Munk 0.0184 | DIFFERS x1.36 | - | A4 |
| `sst_pursuit` vs TR 0702 Sec. 3 | target-radiated noise in SST's example; feature exists (D-123), no level printed | comment said "the simulator has no one-way source" | DIFFERS (stale comment; the matrix and SCENARIOS.md already said the right thing) | comment | FIXED HERE (1936f86) |
| `sst_pursuit` surface law | TR 0702 Sec. 6.4.2 geometric mean; Chapman-Harris range 0.4-6.4 kHz | mean angle at 9.9 kHz | DIFFERS (A3 numbers) | its tests, figure | A3; comment FIXED HERE |
| `sst_echo_level` | TR 0702 eqs. (36)-(38), (66), (68) | as stated | MATCHES: 0.004 / 0.005 dB, 0.13 us (`test_sst_validation.py`) | - | NONE |
| `sst_sediment_reverb` | "MediumSand"; medium sand of arXiv:1809.08459 = J&R Fig. 13.2 | `medium_sand` | MATCHES: `test_sediment_reverberation_follows_the_scattering_strength_integral` (+0.27 dB mean) | - | comment FIXED HERE |
| `flank_sst_shallow`, `flank_sst_spread` | "SST-style": TR 0702's time spreading; slope per Dahl | Cox-Munk total from 6 m/s | DIFFERS from the forward-scatter papers (x1.69) | their tests (qualitative) | A4; comments FIXED HERE |
| Five FLS scenarios | no source above 70 kHz | Chapman-Harris | DIFFERS from the law's range; three files said so, two did not | as 2.6 | FIXED HERE (descriptions); A13 kept |
| Buried scenes | J&R Ch. 8, 15 | unit transmission | DIFFERS (G-3) | 2 scenes | A9 |
| `popoto_env_gravel_30m` "gravel" | no gravel geoacoustics in J&R | fluid {1800, 2.0, 0.6} | NO SOURCE READ for a gravel class bottom (the file states its values) | 1 scene | NONE |
| `rail_objects` "pebbles" | scene points with own mu_db | - | NOT CHECKED against a source (not a seabed model) | - | NONE |

### 2.15 Documents' quoted numbers

| Statement | Source | Verdict | Action |
|---|---|---|---|
| CONFIGURATION.md Chapman-Harris "below about 40 degrees; not confirmed" | [McD93] p. 1918 | DIFFERS | FIXED HERE (b42e1b7) |
| CONFIGURATION.md `time_spread` slope not stated | [D96] (A7) | DIFFERS (omission) | FIXED HERE (b42e1b7) |
| PHYSICS_NOTES 23.3 "Chapman-Harris ... could not be confirmed"; 23.4 "Dahl papers could not be opened"; 6.4 Chapman and coherent-reflection rows | the notes | DIFFERS (stale) | FIXED HERE (b42e1b7) |
| REFERENCES.md Dahl 1996 "100, 2742" | [NF] top | DIFFERS (wrong citation) | FIXED HERE (487c8b8) |
| REFERENCES.md every status the plan's Sec. 5 lists | the notes | DIFFERS (not applied) | FIXED HERE (487c8b8), each verified against its note section |
| `docs/handoff/SIMULATION.md` "Chapman-Harris is unconfirmed"; "the roughness interpolation and composite split are ours" | [McD93]; J&R (13.72)-(13.77) | DIFFERS | proposed text 3.3 (not my file) |
| `docs/RESULTS.md` "Validation against TR 0702" ASIAEX surface tail numbers | TR 0702 | MATCHES today's records; will move with A4 | A4 |
| `docs/figures/sst_validation.md` Sec. 6.4.2 "not checked numerically" | TR 0702 | DIFFERS (now checked: 0 to 5.9 dB) | A11 |
| `docs/SCENARIOS.md` ASIAEX arrivals 0, 3.97, 9.20, 20.68, 30.57 ms | image closed forms | MATCHES (recomputed, `asiaex.py`) | NONE |
| `docs/handoff/EXAMPLES.md` | no source-bound numbers found (grep of authors, "printed", "Table") | NOT CHECKED beyond the grep | NONE |
| `docs/DESIGN.md` link budget | D-198 | MATCHES its own caveat | NONE |

---

## 3. Fixed on this branch (documents, docstrings, comments; no behaviour, no expectation)

| Commit | What |
|---|---|
| 487c8b8 | `docs/REFERENCES.md`: the plan's Sec. 5 status changes, each checked against its note section; new rows for J&R, [McD93], [D97]/[DP97], [D99]/[McD86], [D96]/[D04]; the Dahl 1996 citation; wanted rows; Sec. 11 rows (K factor, bubble exponent, rock C1 settled, (L.34)) |
| 8b573ea | docstrings and comments: `acoustics` (Francois-Garrison scope instead of an unsourced frequency range; density; coherent reflection; total vs large-scale slope), `bottom` (Lambert bound; composite structure), `surface` (three differences from [D96]/[D97]/[D99]/TR 0702), `timespread` (facet geometry = [D96] (15); slope differs) |
| b42e1b7 | `docs/PHYSICS_NOTES.md` (6.4, 14.2, 23.3, 23.4), `docs/CONFIGURATION.md` (Chapman-Harris range, time-spread slope), `docs/SCENARIOS.md` (flank_sst_*, sst_pursuit, sst_asiaex) |
| 1936f86 | scenario comments: `sst_pursuit` (stale one-way-source remark; surface law), `sst_asiaex` (Dahl 2004 layout), `flank_sst_shallow`, `flank_sst_spread`, `fls_120k_cluttered_seabed`, `fls_170k_traffic`, `sst_sediment_reverb`. Parsed configurations identical before and after (`snap.py`: JSON of `load_yaml` and `load_scenario_config`, byte-equal) |
| 359ff57 | `tests/python/test_medium_models.py` comment: the 100 kHz Thorp limit is ours (13 passed) |
| 42f90fc | `docs/SCENARIOS.md`: the five FLS rows say their surface level is an extrapolation |

No test expectation was changed: no expectation was found that pins a value a read source contradicts while the
code agrees with the source.

### 3.3 Proposed text for files this branch does not edit

- `python/gpusonar/sim/propagation.py` `rayleigh_reflection` docstring (after G-3 lands): "Sources: equal to
  Jackson and Richardson (2007) (8.49)-(8.53) and the conversion (2.6), (8.25)-(8.27) (confirmed; REFERENCES.md
  Sec. 12.4)."
- `docs/handoff/SIMULATION.md`, "Boundary models" row: replace "Chapman-Harris is **unconfirmed** (not read in any
  opened document)" by "Chapman-Harris is via secondary (McDaniel 1993 eq. (27); fitted at 0.4-6.4 kHz, low grazing
  angles: used above that in 10 shipped scenarios, an extrapolation)", and "the roughness interpolation and
  composite split are ours" by "the composite split is printed in Jackson and Richardson (2007) (13.72)-(13.77);
  the interpolation is ours (not printed anywhere read)".
- `docs/PHYSICS_NOTES.md` Sec. 15: "Jackson and Richardson (2007) App. E write p = Re{P exp(-i w_c t)}: our baseband
  is the complex conjugate of their P; a phase they write exp(+i w_c t_d) is exp(-i w_c t_d) here."
- `docs/STATUS.md` (new lines): "Conformance to the read references: `docs/references/CONFORMANCE.md` (149 items:
  71 match, 48 differ, 15 have no source read, 8 follow a corrected misprint, 7 not checked); its apply list (A1-A13)
  is the owner's 2026-11-26 instruction to apply every alignment."
- `docs/DECISIONS.md` (new entry): "D-2xx Conformance matrix and the apply list (sonar-jbnm; no default, scenario or
  result changed on the branch). Worse or uncertain first: the forward-scatter slope (A4) improves agreement with
  Dahl 1996/2004 at 6-20 degrees and worsens it with our SPACE'08 comparison above 12 degrees (measured/model 1.28
  -> about 2); the FLS surface level at 110-345 kHz stays an extrapolation with no source; [D99]'s angle spreads
  imply a slope half the printed law (open). Fixed: REFERENCES statuses and the Dahl 1996 citation; stale
  Chapman-Harris statements; sst_pursuit's stale comment; rock-loss contradiction C1 settled (shear loss
  parameter)."
- `docs/RESULTS.md`: nothing until A1-A4 land (their before -> after numbers above).

---

## 4. Not checked, and why

- C++ and CUDA twins of the reflection coefficient and the time-spread pairing: not run (the main checkout's
  module lacks this branch's newest bindings); held by the parity tests.
- [N3]'s buried-target numbers (G-3 owns the files), [NF]'s decay-time comparisons and [NB]'s SSA lobe (they need
  the figure readings and a stand-in spectrum): taken from the notes, labelled so.
- The record-level effect of A3 and A4 on `sst_pursuit` and the time-spreading scenarios: only closed forms here
  (A4) and strength differences (A3); the GPU suite re-measures them.
- `docs/handoff/EXAMPLES.md` beyond a grep; `fjord_*` and `ssp_*` salinities against Francois-Garrison's scope.
- TR 0702's p. 101 Doppler rule: neither reading has a read source beyond TR 0702 itself.

## 5. Scripts (session scratchpad, not committed)

`/tmp/claude-0/-home-user-claude-playground/32e65dcd-82ae-5f75-a363-7372d2e7f847/scratchpad/conform/`:
`check1.py` (seawater, slopes, bubble constants), `check2.py` (reflection twins, composite, rows), `asiaex.py`
(TR 0702 vs Dahl 2004 layouts, closed forms), `slope_fls.py` (slope law at the scenarios; surface laws at FLS
angles; bistatic rules), `fls_run.py` (the five FLS scenarios end to end with each surface model), `rock.py`
(fluid-solid reflection), `survey.py` (every scenario's surface, bottom and time-spread settings), `snap.py`
(scenario configs before and after the comment edits). Run with `PYTHONPATH=<worktree>/python` and the main
checkout's built module linked into the package.
