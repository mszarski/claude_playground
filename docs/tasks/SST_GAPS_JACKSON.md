# Task: close gaps to SST with Jackson and Richardson (2007), McDaniel (1993) and Dahl et al. (1997)

Issue `sonar-gm8e` (feeds `sonar-3k1`, `sonar-9le`, `sonar-6fc`, `sonar-6t5`, `sonar-sul`, `sonar-64sm`,
`sonar-5bf2`, `sonar-w2w3`). Written 2026-11-26 at commit `0e03c32` on branch `fix-jr`, from five reference
notes. Read `docs/HANDOFF.md` section 2 (rules in force) first, then this file, then the note sections that a
package names. "SST" is the sonar simulation toolset; APL-UW TR 0702 is its open description and
`docs/figures/sst_validation.md` our comparison with it (its Table 6 row: "APLBottom, JacksonBottom,
McDanielSurface ... cannot be checked"). TR 9407 is distribution-limited and is never read.

The notes, cited below by tag and section:

| Tag | File | Covers |
|---|---|---|
| [N1] | `docs/references/jackson_richardson_2007/01_sediment_properties.md` | Chs. 1-5, Apps. A, B, C, H, I; Ch. 11 pp. 312-314 |
| [N2] | `docs/references/jackson_richardson_2007/02_roughness_heterogeneity_statistics.md` | Chs. 6, 7, 16, Apps. D, E |
| [N3] | `docs/references/jackson_richardson_2007/03_wave_theories_reflection_penetration.md` | Chs. 8-11, 15, Apps. F, G, (13.14), L.3 |
| [N4] | `docs/references/jackson_richardson_2007/04_scattering_models.md` | Chs. 12-14, Apps. J-M |
| [NS] | `docs/references/surface_papers_mcdaniel_dahl.md` | McDaniel (1993), Dahl and Plant (1997), Dahl et al. (1997) |

Every number below is quoted from a note (tag and section) or from the repository (file named). Exposure
counts come from reading the 80 files in `configs/scenarios/` at `0e03c32`. Where two notes, or a note and
the repository, disagree, it is said where the number is used and collected in section 0.5.

---

## 0. Answer first

### 0.1 What is wrong (sizes from the notes; exposure from `configs/scenarios/`)

| # | What | Size | Shipped scenarios exposed | Source |
|---|---|---|---|---|
| W1 | `sediment` tier above the critical angle: no sediment attenuation, no interface transmission (unit amplitude, `SedimentProvider`, C++ `sediment_eigenrays`) | targets too loud by about 6 dB two-way at 20 kHz and 9 dB at 30 kHz for 0.3 m of medium sand; interface factor near vertical only +0.07 dB, growing toward the critical angle | **2**: `buried_cylinder`, `buried_objects` (`subbottom`, 10-30 kHz, 1.5 m up, raster +-0.9 m: steep, above the 28 deg critical angle of 1700/1500 m/s); their headers already say transmission and attenuation are missing | [N3 §3.5], [N3 WP1] |
| W2 | `sediment` tier below the critical angle: a ray at the critical angle running through the sediment with no attenuation (checked: 10 m up, 0.3 m deep, 40 m off: -31.9 dB) | order +20 to +30 dB too loud against the book's measured and modelled sub-critical field (about 10-30 dB below the incident field, Figs. 15.3, 15.8, 15.11); about -100 dB too quiet if only attenuation is added; wrong arrival structure | **0** (both buried scenes are above critical) | [N3 §3.5] |
| W3 | `rock` (and `pebbles`) reflect as fluids | `rock` with shear 40-60 % of compressional: 1.4-3.1 dB per bounce at 5 deg, 2-7 dB at 10 deg, peak 9-19 dB near the shear critical angle; fluid gives about 0 dB below 66 deg. [N1 §3.4] gives 4.8-5.5 dB at 40-60 deg for the same kind of case: **the notes disagree** (0.5 item C1) | **0** (no scenario uses `rock` or `pebbles` as a bottom; `popoto_env_gravel_30m` is a fluid `{speed 1800, density 2.0, attenuation 0.6}`; `rail_objects`' "pebbles" are scene points with their own `mu_db: -12`) | [N3 §3.3], [N1 §3.4] |
| W4 | `composite` and `combined` lack Wagner's shadowing factor (13.81)-(13.83) | ours higher at 5 deg: +0.9 dB (medium sand, 30 kHz), +3.0 dB (100 kHz), +3.4 / +6.5 dB (sandy gravel, 30 / 100 kHz); about a third of that in dB if Q has pi^(-1/2) instead of the printed pi^(+1/2) | **0** (the four `fluid_sediment` items in `flank_sst_shallow`, `flank_sst_spread`, `fls_glints_on_sediment`, `sst_sediment_reverb` all use the default `small_slope`) | [N4 §3.4], [N4 §2.6] |
| W5 | Composite validity (13.78) not checked: sigma_sL < 0.1, theta < 70 deg, K_c below Bragg | medium sand 100 kHz sigma_sL 0.157; sandy gravel 0.178 / 0.367 at 30 / 100 kHz: "corrections suspect" | **0** | [N4 §3.1] |
| W6 | Volume term isotropic in sigma_v | toward forward scatter (theta 20 deg, Fig. 14.3 inputs): ours 4 dB too high (rho_rk = -1) or 10.6 dB too low (rho_rk = 0) | **0** (the one bistatic item, in `sst_pursuit`, is a Chapman-Harris surface) | [N4 §3.6] |
| W7 | `bubble_layer_surface` (`apl_1994`) at low wind, open ocean | 6 to 12 dB above every open-ocean data set at 2.6-6.4 m/s and 5-45 deg (0-3 dB at 60 deg); wind exponent U^3.8-4.1 against measured U^5.3-6.4 at 17 deg, 26.5-70 kHz | **0** (no scenario uses `bubble_layer_surface` or `kraus_surface`); D-193's `test_gallaudet_boundaries.py` | [NS §3], [NS §3.2] |
| W8 | Facet term near normal incidence | 3.3-4.3 dB below Nuetzel's 20 kHz data at 82.5-88 deg (one data set, read from a figure) | **0** | [NS §3.1], [NS WP6] |

Uncertain, not shown wrong (owner decisions in section 3):

| # | What | Size | Exposure | Source |
|---|---|---|---|---|
| U1 | Volume parameter of mud rows (0.001 for Mz >= 5.5 phi) | sites imply 0.0031 (+4.9 dB) and 0.0037 (+5.7 dB) (Table 14.1, two sites); Gallaudet's fit 0.003 (`fine_silt` 5.0 dB low at 48-66 deg, D-193); three other Table 14.1 sites imply 0.0008, 0.0017, 0.0016 | **0** (no scenario names a row of 5.5 phi or finer) | [N2 §3.3], [N4 §3.6], `docs/DECISIONS.md` D-193 |
| U2 | Roughness rule for silts and clays | our w2 rule 1-7 dB below the median of Briggs' Table 6.1 at 10-100 kHz (+7.1 dB at 10 kHz, +2.2 at 100 kHz, -0.9 at 300 kHz; 8 rows, 4 sites, two glacial till and carbonate); sands at the median within 1.4 dB | **0** | [N2 §3.2] |
| U3 | `rock` Lambert -16 dB (D-88), `pebbles` -9 dB | book Ch. 12 fits: rock -5.6 dB (ours 10.4 dB lower), gravel -11.7 dB (ours 2.7 dB higher); the book says its rock data exceed the energy bound | **0** | [N4 §3.5] |
| U4 | Sand as a simple fluid | against EDFM/Biot with measured SAX99 parameters the fluid model has 1.2-1.6 dB too little loss at normal incidence, 1.0-1.2 dB at 45 deg, and 1-3 dB too much roughness backscatter; the TR 9407 rows already lower density and speed to fit reflection data | sand-like bottoms in many scenes, but the rows must not be changed this way (double count, section 1 G-10) | [N3 §3.4] |
| U5 | Chapman-Harris used outside its stated range (400-6400 Hz, low grazing) | in the five FLS scenarios at 120-330 kHz: Chapman-Harris minus `bubble_layer_surface` at 5 m/s from -5.4 to +6.6 dB at 120 kHz and +2.2 to +7.7 dB at 330 kHz; up to 11.7 dB (5 m/s) and 15.8 dB (10 m/s) above a [D97]-based bubble term; nothing in any source is validated above 70 kHz | **18** use `chapman_harris`: FLS 120 kHz (2), 170 kHz (2), 330 kHz (1); `flank_32` 6-10 kHz (4); `popoto_long_range*` 4-6 kHz (8); `popoto_far_mover` (1) | [NS §2.7], [NS §3.6], [NS §5] |
| U6 | Grain-size table interpolated linearly between half-phi rows | at most 0.68 dB in normal-incidence reflection (5.3 phi), under 0.1 dB below 5 and above 6 phi | **0** (no scenario or library caller uses `sediment_from_grain_size`; repository grep) | [N1 §3.2] |
| U7 | Constant delta (no causal dispersion) | +0.24 dB normal-incidence reflection and +2.1 deg critical angle per decade (sand); SAX99 data flat above 20 kHz | all fluid bottoms, small | [N1 §3.4] |
| U8 | K texture: large-cell limit, no saturation for small cells, no correlation between cells or pings | structural, no dB figure | **0** reverberation items with K statistics (the three fjord scenarios use K only on ambient noise) | [N2 §3.4] |
| U9 | Default Lambert -29 dB (D-89) | book mud and sand fits are -19.7 and -20.2 dB, 9 dB above the bare default at 10-140 kHz | **0** rely on the default: every Lambert item sets `mu_db` (-27 dB in 19 files; -30, -20, -35 in one each) | [N4 §4] |

### 0.2 What the sources confirmed in our simulator (no change needed; G-1, G-2 pin them)

- Conventions: exp(-i w t) with Im k > 0, grazing angles, a_rho, nu_p, delta_p and its conversions (2.6),
  (8.18), (8.25)-(8.27) (dB per wavelength 40 pi / ln 10), sigma and S_b, reciprocity (2.18), energy bound
  (2.20), Lambert bound mu < 1/pi (2.23) [N1 §2.1, §3.1], [N3 §2.1, §3.1].
- Fluid flat-interface reflection (8.49)-(8.53): ours equal to 1.8e-15 over 0.5-90 deg, four sediments [N3 §3.1].
- Coherent reflection exp(-2 k^2 h^2 sin^2) (13.14): equal (0.24569 both at 30 deg, 40 kHz, 1 cm) [N3 §3.1].
- Perturbation (13.21)-(13.25), Kirchhoff (13.10)/(13.12)-(13.17), small-slope (13.11), structure constant
  (D.12), volume term (14.8)/(14.9), composite cut-off (13.72)-(13.75) and large-scale slope (13.76)-(13.77):
  all equal to under 0.001 dB, backscatter and seven bistatic geometries; the book's printed slopes 0.0227,
  0.0595, 0.1432 (Fig. 13.19) reproduced [N4 §3.1-§3.3]. Our default (`small_slope` + volume) equals the
  book's SSA + volume to under 0.001 dB at every case tabulated [N4 §3.2].
- Roughness spectrum normalisation and `w2_from_cm4` (D.1.3) [N2 §3.1]; `medium_sand` roughness and
  geoacoustic inputs equal the book's Fig. 13.2 "typical medium sand" [N4 §3.1].
- The 20 grain-size rows equal the TR 9407 relations as the book prints them, eqs. (11.1)-(11.3), with two
  printed constants corrected (0.5 table) [N1 §3.2], [N3 §3.2].
- Seawater density at one atmosphere (B.2)-(B.3) and Francois-Garrison (B.6)-(B.12): every constant [N1 §3.3].
- K PDF (16.33)-(16.35): our `shape` is the book's nu; simulated P_fa matches; (16.26) behind the detector's
  effective-looks count [N2 §3.4]. `k_cell_area`'s cancelling cosines agree with App. G.2.1 [N3 §3.7].
- Surface: bubble term = [D97] eq. (17); damping 2.55e-3 f_Hz^(1/3) confirms our kHz form; saturation within
  0.1 dB of [D97] Table I; facet term = [McD93] eq. (7) to 1e-6 dB; Chapman-Harris formula identical to
  [McD93] eq. (27) (via secondary); optical-depth convention identical [NS §3.1].

### 0.3 What the sources do NOT contain that we hoped for

| Wanted | Result | Source |
|---|---|---|
| Kirchhoff / composite interpolation function (published APL-UW model) | not printed: "smooth interpolation" only (p. 349, Fig. 13.8); the book recommends small-slope as the single all-angle expression (p. 336), which is our D-66/D-86 choice. [McD93] prints none either | [N4 §2.7], [NS §2.2] |
| Empirical large-roughness law for gravel and rock | not printed; Lambert fits only; "no experiments ... on well-characterised gravel or rock" (p. 362) | [N4 §2.7] |
| Williams-Jackson bistatic interpolation | not printed; the general bistatic cross sections are | [N4 §2.7] |
| Layered-seafloor reflection recursion | not in Chs. 8-11; the book has (13.26) and (M.36)-(M.47), which need a layered V; the recursion in [N3 §2.2.9] is ours | [N3 §2.2.9], [N4 §2.3, §2.9] |
| Measured K shapes for any seabed | none printed (figures only); only McDaniel's fitted L = 1 m, gamma = 1.5 | [N2 §2.6] |
| Medium-sand volume parameter (D-90) | no named-sediment sigma_2 anywhere; typical range 0.0001-0.004 | [N1 §2.1], [N2 §3.3], [N4 §4] |
| Anything for the sea surface above about 70 kHz | nothing: data end at 60 kHz ([McD93]) and 70 kHz ([D97]); the book is about the seafloor | [NS §1, §3.6], [N2 §4] |
| A near-specular (forward) surface scattering model, coherent loss, time/angle spread | none in the three papers; candidate McDaniel (1986), not in hand | [NS §2.2, §4] |
| The handbook's G(U), Plant's spectrum, u*(U10), Cox-Munk constants | none printed | [NS §2.5, §2.8] |
| Ping-to-ping coherence of reverberation | no formula; seabed decorrelation about 1 day (roughness) to 1 month (volume) | [N2 §2.6, §3.4] |
| Mackenzie (1981) sound speed | not printed (Medwin and Del Grosso are) | [N1 §3.3] |

### 0.4 Misprints anyone implementing from these sources must know

| Source | Equation / table | As printed | Should be | How we know |
|---|---|---|---|---|
| J&R p. 314 | (11.2), Mz >= 5.3 | nu = -0.0024324 Mz + 1.0091 | + 1.0019 | continuity with the cubic at 5.3 (0.98901 both); our 7 rows and three earlier prints follow 1.0019 [N1 §3.2], [N3 §6]. Not proven: the original not read |
| J&R p. 314 | (11.3), -1 <= Mz < 0 | k = 0.4566 | 0.4556 | next piece starts at 0.4556; rows at -1 and -0.5 phi follow it [N3 §6]. [N1 §2.4] types 0.4566 without comment (0.5 item C3) |
| J&R p. 136 | Table 5.3 carbonate porosity regression | 11.760 | 1.760 | only 1.760 gives ratios near 1 [N1 §6] |
| J&R p. 136 | Table 5.3 both attenuation regressions | as printed | unusable | negative beyond 4.5 phi; 9.7 at 90 % porosity against data near 0.1 [N1 §6] |
| J&R p. 140 | Table 5.4 caption | rho in g/cm3 | kg/m3 | rho = 2000 gives 1.10, matching Fig. 5.7 [N1 §2.3] |
| J&R p. 150 | Table 5.6 validity | IOI in kg m^-1 s^-1 | g/cm3 (IOI as defined) | [N1 §6] |
| J&R pp. 167-169 | Fig. 5.28 axis | kg/m3 | g/cm3 | [N1 §6] |
| J&R p. 159 | (5.8) | exponent -1.1394 | -1.1394 e (void ratio) | (5.9) has it [N1 §6] |
| J&R Ch. 11 | Fig. 11.2 "standard" curve | "Eq. (5.15)" | (5.14) | (5.15) is Mz(rho) [N1 §6] |
| J&R p. 63 / p. 59 | anorbital ripple wavelength | 500 D (text) | 535 D (figure)? | unresolved [N1 §6] |
| J&R p. 115 | quartz grain density | 2560 kg/m3 | 2650 | 2650 on pp. 97, 114, Table 4.7 [N1 §6] |
| J&R p. 460 | (B.5) pressure unit | kg m^-2 | kg/cm^2 | p. 458 [N1 §6] |
| J&R p. 460 | App. B text | f1/f2 = MgSO4/boric "respectively" | the reverse | formulas (B.6), (B.9) [N1 §2.8] |
| J&R p. 478 | (D.19) | Gamma((gamma2 - 2)/2) | Gamma((gamma2 - 1)/2) | quadrature to 1e-10; as printed w2 is 1.5 times too small at gamma2 3.25 (1.8 dB), 3.0 times at 2.5 [N2 §2.1]. Fig. 6.18 may have been computed with the misprint (unresolved, [N2 §6]) |
| J&R p. 478 | (D.23) | F^(gamma1 - 1) | F^gamma1 | (6.11), (D.22), (D.24) [N2 §2.1] |
| J&R p. 196 | Sec. 6.7, Fig. 6.19 line | S(F0) = 7.1e-5 cm^3 | about 1.4 cm^3 | two points read off the line, +-20 % [N2 §2.3] |
| J&R p. 438 | (16.55) | Gamma(n/s) | Gamma(n/2) | chi-square form [N2 §2.6] |
| J&R p. 292-293 | (10.39); (10.47)-(10.48) | root printed as k_q; "c_t" | k_q^2; c_t is a slowness | reproduces the book's 20 kHz numbers exactly [N3 §6] |
| J&R p. 319 | rho_eff/rho at 250 kHz | 0.657 + i 0.0046 | about 0.863 + 0.006i? | (10.75), Table 10.1, Fig. 10.6; unresolved [N3 §6] |
| J&R p. 497 | (G.6)-(G.7) | "t - tau/2" on both bounds | upper bound "+" | [N3 §6] |
| J&R App. G.2 | (G.13) | -2 alpha_p r_t | water absorption | consistent with exp(-4 k_w'' r_t) in (G.11) [N3 §2.9] |
| J&R pp. 362-368 | (13.82) Wagner Q | pi^(+1/2) exp(-t^2) | pi^(-1/2)? | the usual Smith/Wagner form; 2-5 dB at 2 deg for rms slopes 0.05-0.35; **check Wagner (1967)** [N4 §6] |
| J&R p. 546 | (L.34), (L.35) | Delta k^2/Delta k_z^2; 1/(2 pi norm B_s) | (13.62)-(13.63) forms | Ch. 13 versions consistent with (13.67), (13.70) [N4 §6] |
| J&R p. 323 | Sect. 12.1.1 sand bound | "5 phi < Mz <= 0 phi" | 0 < Mz < 5 | inverted [N4 §6] |
| J&R p. 23 | seawater density | kg/m^2 | kg/m^3 | [N1 §6] |
| [McD93] p. 1910 | eq. (22) | Omega = g K (1 + tau K^2) | Omega^2 | dimensions [NS §2.5] |
| [McD93] p. 1910 | eq. (23), Gamma | exp[-(K/K_p)^(1/2)/(2 s^2)] | (sqrt(K/K_p) - 1)^2 in the exponent (Donelan) | probably; immaterial at Bragg wavenumbers [NS §2.5] |
| [McD93] p. 1911 | eq. (24) range | 0.942 cm | 0.942 cm^-1 | [NS §6] |
| [McD93] pp. 1918-1919 | Chapman-Harris | "Eq. (28)" | eq. (27) | no eq. (28) exists [NS §2.7] |
| [McD93] p. 1921 (reference list) | McDaniel (1986) eq. (18) | as published | right side times 2 | erratum printed in the review's reference list [NS §2.2] |

### 0.5 Where the notes contradict each other or the repository

| # | Item | Disagreement | What to do |
|---|---|---|---|
| C1 | Elastic rock loss at 40-60 deg | [N1 §3.4]: 4.8-5.5 dB at 40-60 deg (V_s = 0.5 V_p or Olson's generic rock). [N3 §3.3]: 9.2 / 6.4 / 4.4 dB at 40 / 45 / 60 deg (V_s = 0.5 V_p) and 9.0 / 6.4 / 4.5 (generic rock). [N1] does not state delta_t; [N3] uses 0.1 (0.085 generic) | not picked; G-4's first test recomputes the book's Fig. 9.4 case and both notes' cases and records the inputs |
| C2 | (11.3) at Mz = 6.0 | [N3 §2.5] "discontinuous at 6.0 as printed (0.338 against 0.140)"; [N1 WP-A] the printed jumps are 0.001-0.002 at 0, 2.6, 4.5, 6.0. Evaluating the two printed pieces at 6.0 while writing this plan gives 0.138 and 0.140 (this plan's arithmetic, not a note's) | G-6's continuity test settles it |
| C3 | (11.3) first constant 0.4566 | [N3]: misprint, 0.4556 is meant; [N1]: printed and "allowed" | same effect either way (delta 0.01709 vs 0.01705 at -1 phi, [N3 §3.2]); G-6 follows the rows (0.4556) and documents both |
| C4 | K shape and patch count | book p. 433: "2 nu" patches; D-196 (primary paper, Abraham and Lyons 2002): shape = n for n patches, n/2 for n scatterers | keep D-196; record the conflict ([N2] top, WP-2) |
| C5 | Implied mud sigma_2 | [N2 §3.3] five sites: 0.0031, 0.0037, 0.0008, 0.0017, 0.0016; [N4 §3.6] two sites (0.0031, 0.0037; finite-L_c rows not computed, Ch. 7 unread) and says they "agree with Gallaudet's 0.003"; [N2] calls the support "weak" | both reported in decision 1 |
| C6 | Rock and gravel exposure | [N3 §3.3]: "Popoto scenes over gravel/rock"; repository: no scenario has a rock or pebbles bottom; the gravel scene is a fluid half-space | exposure 0 (0.1 W3) |
| C7 | How much of the book's tables may be carried in tests | [N2 WP-3], [N2 §5]: never tabulate Table 6.1 or C.1; "a test may carry three or four rows". [N4 WP-10]: encode Table 13.1 (8 rows) and Table 14.1 (5 rows) as a test-data file | owner decision 11 |
| C8 | Rock properties, p. 169 | [N1 §2.7] confirmed on the page image; [N3 §2.3] from extracted text only | use [N1]'s reading |
| C9 | Cut-off attribution | `docs/REFERENCES.md` 3.2 credits the k h_s = 1/2 split to Fonseca 2025 / Novarini-Caruthers; the book attributes it to D. Jackson et al. 1986a | add the attribution (section 5) |

---

## 1. Work packages, merged and de-duplicated

Ordered by value to realism per effort. Flags: **OPT-IN** (adds an option, no default result moves; can be
done without asking), **OWNER FIRST** (would move a default result or a shipped record), **RECORD ONLY**
(documents, status, tests that pin a present equality). Every package: tests first (HANDOFF section 2), one
decision entry, proposed `REFERENCES.md`/`STATUS.md`/`DECISIONS.md` text in the report.

Engines (this plan's check of `libs/` at `0e03c32`): the scattering-strength models of `sim/bottom.py`,
`sim/surface.py` and `acoustics.py` have **no C++ or CUDA counterpart** (both simulator engines and the CUDA
reverberation backend take the same strengths from Python). Flat-bottom reflection has twins:
`propagation.rayleigh_reflection` and `libs/sim/src/paths.cpp:140` (simulator); the live chain has its own in
`libs/live/src/raytrace.cpp:170`, `libs/live/src/coverage.cpp:55` and Python `raytrace.py:186`, `lloyd.py:190`.
The sediment tier is `propagation.SedimentProvider` and C++ `sediment_eigenrays` (`libs/sim/src/paths.cpp:310`,
bound in `python/bindings/bind_simcpp.cpp`). `CellTexture` is drawn in Python and multiplies the deviates handed
to both engines (D-196). Bellhop's bottom line is written in `sim/bellhop.py` (about line 221) with shear `0.0`.

### G-1 Pin the book's bottom equalities as tests; relabel "ours" to "confirmed" — RECORD ONLY, S
- Gap: rows marked "ours" or "via secondary" that the book prints (0.2). Draws on [N1 §3.5], [N2 §3.5, WP-2],
  [N3 §3.1, §4], [N4 WP-1, WP-7, WP-9].
- Formulas: (13.21)-(13.25), (13.10)/(13.16)-(13.17), (13.11), (D.12), (13.75), (13.77), (14.8)/(14.9)/(8.26)
  [N4 §2.3-§2.9]; (8.49)-(8.53) [N3 §2.2.3]; (13.14) [N3 §2.6]; D.1.3 [N2 §2.1]; (2.18), (2.20), (L.26) [N4 §2.1].
- Tests first (new `tests/python/test_jackson_2007_bottom.py`): retyped A_ww against `_perturbation_factor` at
  Fig. 13.2 inputs (a_rho 1.845, nu 1.1782, delta 0.01624, w2 0.000141 m^0.75, gamma2 3.25), 5-85 deg and three
  bistatic geometries, 1e-6 relative; `large_scale_slope` at 20 kHz, gamma2 3.25, w2 0.0000422 / 0.0001406 /
  0.0004218 gives 0.0227 / 0.0595 / 0.1432 (printed, p. 368); sigma proportional to f^(4 - gamma2) (13.61);
  Fig. 14.2 critical angles 17.8 / 24.6 deg, Fig. 8.3 intromission angle 10.46 deg (|V| 0.0009) [N3 §2.2.3];
  Fig. 14.2 volume curve by eye within about 1 dB (-24.1 / -23.6 / -23.3 dB at 90 deg) [N1 §3.1]; reciprocity
  and hemisphere energy for `small_slope` [N4 WP-9]; `structure_constant` = (D.12), `w2_from_cm4` = D.1.3.
- Files: the new test file; `configs/sediments.yaml` comment text only (`medium_sand` provenance "J&R 2007
  Fig. 13.2"; the book's Lambert fits rock -5.6, gravel -11.7 dB in the `basis` text of `rock`, `pebbles` with
  the book's caveat, [N4 WP-7]); `bottom.py` docstring citations go with G-6 (same file). Proposed text for
  `REFERENCES.md` (section 5), PHYSICS_NOTES Sec. 15 (our baseband is the conjugate of the book's P, [N2 §3.4]).
- Engines: none. Size S. Depends on nothing.
- Must not: change any formula or value.
- **Done (2026-11-26, branch `g-bottom`)**: `tests/python/test_jackson_2007_bottom.py`, 15 tests, every book
  formula retyped. Everything the notes said agrees, agrees: perturbation to 1e-13, Kirchhoff and small-slope
  to 2e-7 (against an independent real-axis I_K), volume to 1e-11, (D.12) to 1e-14, (13.75)/(13.77) to 1e-12,
  Fig. 13.19 slopes 0.0227 / 0.0595 / 0.1432 printed digits, the notes' SSA + volume tables to their 0.01 dB.
  Small-slope's hemisphere integral is 0.902 / 1.031 / 1.005 |V|^2 at 20 / 40 / 70 deg (Fig. 13.2, 30 kHz):
  below the (2.20) bound, not exactly (L.26). `configs/sediments.yaml`: comments only (loads identically).
  The `bottom.py` docstring citations went with G-6. Nothing disagreed.

### G-2 Pin the surface papers' equalities; fix the docstrings — RECORD ONLY, S
- Gap: Chapman-Harris "unconfirmed"; its docstring range "below about 40 degrees" is not printed; the surface
  "lower bound" wording (D-193, `sonar-64sm`). Draws on [NS §3.1, §3.4, WP1].
- Tests first (new `tests/python/test_surface_papers_1997.py`): `bubble_saturation_db` at 17 deg against [D97]
  Table I (-26.9 / -27.8 / -28.3 dB at 26.5 / 50 / 70 kHz; ours -26.82 / -27.74 / -28.22; tolerance 0.15 dB);
  `bubble_damping_constant(f) == 2.55e-3 f_Hz^(1/3)`; `facet_backscatter` = [McD93] eq. (7) with
  sigma_x'^2 = mss/2 (1e-9 relative); `chapman_harris_db` = eq. (27) (1e-9 dB), exponents 2.46 (wind) and 0.82
  (frequency) at 30 deg; eq. (18) and (19) limits of `bubble_backscatter`; bracket peak +3.7 dB within [McD93]'s
  3-4 dB. A characterisation of the dB-mean bias of [DP97] eqs. (6)-(7): C_E k = 2.507 dB [NS §2.9].
- Files: the test file; `acoustics.chapman_harris_db` docstring (range: 400-6400 Hz, to 15 m/s, low grazing,
  explosive sources, March 1961, [McD93] p. 1918). The `surface.py` "lower bound" comment goes with G-7.
- Engines: none. Size S.
- **Done** (2026-11-26, branch `g-surface`): `tests/python/test_surface_papers_1997.py`, 12 tests, every
  equality of [NS §3.1] reproduced: Table I -26.82 / -27.74 / -28.22 against -26.9 / -27.8 / -28.3 dB
  (bound 0.1 dB); eq. (17) = [McD93] eq. (9) = ours to 1e-9; eqs. (18), (19) limits; bracket peak +3.73 dB;
  McDaniel's damping law -0.35 dB at 68 kHz (the note's table to 0.01 dB); [D97] Sec. VI convention for
  both laws; facet = eq. (7) to 1e-9; Chapman-Harris = eq. (27) to 1e-9 dB (1e-6 relative with the exact
  knot), exponents 2.459 / 0.820; [DP97] 2.507 dB and 5.57 dB. Printed resonant radii 121 / 64 / 46 um:
  ours 122.4 / 64.9 / 46.3 (1.5 %). Docstrings: `chapman_harris_db` (via secondary, the stated range;
  "below about 40 degrees" removed), `surface.py` (the kHz unit of delta and the bubble/facet terms now
  confirmed in print). No behaviour changed.

### G-3 Sediment tier above the critical angle: transmission and attenuation — OWNER FIRST (code may land opt-in), S
- Gap W1: 6 dB two-way at 20 kHz, 9 dB at 30 kHz, 0.3 m of medium sand [N3 §3.5]. Exposure: `buried_cylinder`,
  `buried_objects`. Draws on [N3 WP1], [N3 §2.2.3].
- Formulas: amplitude = V_wp(theta_w) (8.55) x exp(-k_p'' L_sed) x spreading, k_p'' = k_w delta/nu
  ((8.20)/(8.26)), reciprocity factor 1/a_rho on the return leg (8.68), sin theta_p (8.46) [N3 §2.2.3, §2.2.4].
  The sediment from a `configs/sediments.yaml` name instead of a bare `sediment_speed` (the scenes give
  1700 m/s; `medium_sand` is nu 1.178 on 1528 m/s [N1 §3.2]: which sediment is owner decision 6).
- Files: `python/gpusonar/sim/propagation.py` (`SedimentProvider`), `libs/sim/src/paths.cpp` and
  `include/sonar/sim/paths.hpp` (`sediment_eigenrays`), `python/bindings/bind_simcpp.cpp`, tests
  `tests/python/test_sim_cpp.py`, `tests/cpp/test_sim_paths.cpp`, a new Python test file.
- Tests first: energy identity 1 - |V|^2 = |V_wp|^2 Re{sin theta_p/(a_rho a_p)}/sin theta_w (lossless, 1e-12);
  V_wp = 1 + V; a_rho = 1, nu = 1 gives unit transmission; 0.3 m vertical at 20 kHz in medium sand = 3.0 dB one
  way; Python = C++ [N3 WP1].
- Before -> after the notes give: targets about -6 to -9 dB at 20-30 kHz, sediment clutter likewise [N3 WP1].
  To prepare: target peak levels of both scenes on both engines, and which tests pin them
  (`test_realistic_scenes.py`, `test_scene_objects.py` name the scenes).
- Must not: change the `transmission` default silently; claim sub-critical correctness (G-13).

#### G-3 done (opt-in code; the scene switch NOT made, see below)

- What exists: `propagation.penetration: fluid` on the `sediment` tier, both engines (`SedimentProvider`,
  C++ `sediment_eigenrays`; no CUDA copy: eigenrays are host-side for every backend), default `constant` = the tier
  before, bit-identical (sha256 of the default eigenrays on both engines and of one ping of `buried_cylinder`,
  `buried_objects`, `sst_echo_level`, `rail_objects`, before and after). Per leg V_wp(theta_w) / sqrt(a_rho)
  exp(-k_p'' L); the sediment from `bottom` (`{speed, density, attenuation}` or `{sediment: name}`, or the
  environment's bottom), an unstated attenuation refused; frequency = the sonar's band centre. Sub-critical points
  (straight line below arccos(c_w/c_s)): the evanescent field (15.1) by default (`subcritical: evanescent`), `ray`,
  or `refuse`; a warning once per provider. `docs/PHYSICS_NOTES.md` section 31, `docs/CONFIGURATION.md`.
- Disagreement with [N3 §3.5]: the two-way interface factor |V_wp(90)|^2 / a_rho = 1.017 is a PRESSURE factor,
  +0.146 dB, not the note's "+0.07 dB" (10 log10 of it). The worked case is otherwise reproduced: attenuation
  3.00 dB one way at 20 kHz, 6.00 / 9.00 dB two-way at 20 / 30 kHz; total against the old tier -5.85 / -8.85 dB.
  The evanescent rates 145 / 364 / 728 / 2183 dB/m (10 deg) are reproduced within 0.5 %.
- Owner's table (CPU, host backend; `tools/gallery.py` `run_buried` processing; noise-free scenes, so the figure of
  merit is the target over the median sediment clutter of the refraction-corrected volume, not an SNR; no detector
  runs on these scenes). "On" = `penetration: fluid, bottom: {sediment: medium_sand, speed: 1700}` (the scene's
  speed kept; medium sand's density ratio 1.845 and loss 0.884 dB/wavelength = 10.4 dB/m at 20 kHz at 1700 m/s).
  No point of either scene is sub-critical with it.

| Scene (raster) | Quantity | Off (today) | On | Change |
|---|---|---|---|---|
| `buried_objects` full (0.075 m, 625 pings, gallery) | cylinder peak voxel | ref | -5.18 dB | -5.18 dB |
| | sphere (0.22 m) peak voxel | ref | -3.63 dB | -3.63 dB |
| | median sediment clutter | ref | -4.38 dB | -4.38 dB |
| | `cylinder_over_clutter_db` | 21.12 | 20.32 | -0.80 dB |
| | sphere over clutter | 33.77 | 34.52 | +0.75 dB |
| | cylinder top depth (refraction on) | 0.26 m | 0.26 m | none |
| | sphere top depth | 0.14 m | 0.14 m | none |
| | nadir ping, record peak | 0.0656 | 0.0381 | -4.7 dB |
| `buried_objects` quick (0.15 m, 169 pings, the test) | cylinder peak voxel | ref | -5.85 dB | |
| | sphere peak voxel | ref | -3.62 dB | |
| | `cylinder_over_clutter_db` (test: > 12) | 17.53 | 16.16 | -1.37 dB |
| | cylinder / sphere top, refraction on (test: 0.24-0.31 / 0.14 +- 0.02) | 0.28 / 0.14 | 0.28 / 0.14 | none |
| | cylinder / sphere top, refraction off (test: cyl off < on) | 0.24 / 0.13 | 0.24 / 0.13 | none |
| `buried_cylinder` quick (169 pings) | cylinder peak voxel | ref | -6.50 dB | |
| | cylinder over clutter | 28.99 | 29.26 | +0.27 dB |
| | cylinder top depth | 0.30 m | 0.30 m | none |

  The interface's own (proud) scatterers give bit-identical records with the option on
  (`test_proud_scatterers_of_a_buried_scene_do_not_move`). Every assertion of `test_buried_objects_volume` still
  holds with the option on; `test_scene_objects.py`'s buried-cylinder check asserts only counts and positions.
  If switched, these would be stale until regenerated: `docs/figures/gallery.json` `buried_objects`
  (`cylinder_over_clutter_db` 21.12 -> about 20.3), `docs/figures/buried_objects.png` and `.gif` (drawn in dB re the
  sediment clutter, which drops with the targets: the cylinder moves -0.8 dB, the sphere +0.75 dB on that scale), the
  `docs/RESULTS.md` "3D near-field imaging of a buried object" bullet "Not modelled: transmission loss and attenuation
  in the sediment", the scene headers, and `docs/handoff/SIMULATION.md`'s sediment-tier row. With medium sand by
  name at 1767 m/s instead (quick): cylinder -6.39 dB, sphere -3.59 dB, `cylinder_over_clutter_db` 15.03, depths
  unchanged; 8 points sub-critical (measured with the earlier keep-the-ray policy).
- Not done: switching the two scenes (the owner's later instruction to switch them was refused by this session's
  permission system; the YAML lines to apply are in the G-3 report).

### G-4 Elastic flat-interface reflection (rock, optionally gravel) — OPT-IN, S-M
- Gap W3. Merges [N1 WP-C] and [N3 WP4]; formulas [N3 §2.3] (9.34)-(9.40), losses (9.23)-(9.24).
- Files: `python/gpusonar/sim/bottom.py` (`reflection`), `propagation.rayleigh_reflection`,
  `libs/sim/src/paths.cpp` (simulator engines, parity tests in `test_sim_cpp.py`), `sim/bellhop.py` (shear
  column and shear attenuation instead of `0.0`), `configs/sediments.yaml` (optional `shear_speed`,
  `shear_delta` keys; `unconfirmed_fields: [shear]`). The live chain's reflection (`libs/live`, `raytrace.py`,
  `lloyd.py`) only if the owner wants elastic bottoms in the chain's depth/coverage models: then both
  live-chain engines in one change.
- Tests first: nu_t -> 0 gives the fluid coefficient (1e-12); normal incidence independent of V_s; |R| = 1 below
  both critical angles for lossless rock; Fig. 9.4 critical angles arccos(1/2.5) = 66.4 deg and arccos(1/1.3) =
  39.7 deg (printed), minimum |V| 0.052 at 35.7 deg (recomputed by [N3]); Fig. 9.3 differences below 0.5 dB;
  energy with transmitted fluxes (lossless); Bellhop with shear against our coefficient on a flat bottom;
  both notes' rock cases recomputed with stated delta_t (C1).
- Before -> after (only when a row gets a shear speed): per-bounce table [N3 §3.3] (fluid 0.0 dB; elastic e.g.
  V_s = 0.5 V_p: 1.8 / 3.3 / 6.1 / 15.4 / 9.2 / 6.4 / 4.4 / 2.8 dB at 5 / 10 / 20 / 30 / 40 / 45 / 60 / 90 deg).
  No shipped scene moves (W3).
- Must not: give `pebbles` or any row a shear speed without a source; invent per-site rock V_s; change the
  rock Lambert scattering (U3, a separate decision). Sediment shear only if asked (under 0.2 dB, [N1 §3.4]).
- **Done 2026-11-26 (branch `a-rock`, CONFORMANCE.md A8, the owner's approval to align): rock is elastic by
  default.** `bottom.reflection(..., nu_t, delta_t)`, `propagation.rayleigh_reflection(..., shear_speed,
  shear_attenuation)`, C++ `FluidBottom::shear_speed` (equal to 4e-14 relative), Bellhop's and Bellhop3D's shear
  column; `rock` has 1900 m/s and shear delta 0.085 (Olson et al.'s generic rock, the notes' best-sourced value,
  `unconfirmed_fields: [shear]`), 0.50 V_p. Fluid rows bit-identical (sha256). C1 settled: [N3]'s numbers follow
  from its stated inputs (delta_t 0.1, or 0.085 for the generic rock); [N1]'s 4.8-5.5 dB at 40-60 degrees holds
  only from about 50 degrees up (ours, row: 8.8 / 6.3 / 5.3 / 4.8 / 4.5 dB at 40 / 45 / 50 / 55 / 60). Fig. 9.3's sand differs by up to 0.506 dB (not "below 0.5").
  Tests: `tests/python/test_elastic_bottom.py`, `Paths.ElasticRockCoefficient`, parity cases in `test_sim_cpp.py`.

### G-5 Mud volume parameter and silt roughness: the owner's table — OWNER FIRST, S
- Gap U1, U2. Draws on [N2 WP-4 (memo)], [N2 §3.2-§3.3], [N4 §3.6], D-193.
- Work: no code. With `bottom.named_backscattering_strength_db` compute, for every row of 5.5 phi and finer, the
  backscatter at 6, 30, 100 kHz and 30 / 45 deg with sigma_2 = 0.001 (now), 0.002 and 0.003. The notes already
  give: 0.002 is +3.0 dB on the volume term of every row >= 5.5 phi, nearly +3 dB total off specular [N2 WP-4];
  0.003 is worth 4.8 dB at the Gallaudet site (D-193). Add the roughness comparison of [N2 §3.2] (sands,
  fines, Mz >= 5.5 at 10 / 100 / 300 kHz).
- List which tests pin the present values (`test_gallaudet_boundaries.py`: `fine_silt` 5.0 / 0.8 dB low).
- Engines: none. Must not: change `sediments.yaml` or the 0.001/0.002 step without the owner [N2 §5].
- **Done 2026-11-26 (branch `g-mud`): `docs/tasks/SEDIMENT_DEFAULTS_DECISION.md`, the owner's table; nothing changed.**
  Found: no shipped scenario uses a row of 3 phi or finer (all 80 expanded); 0.002 / 0.003 move the seven rows
  by +3.0 / +4.8 dB at 5-60 deg, 8-330 kHz; the class-median roughness moves them by under 0.45 dB below 60 deg
  (volume dominates by at least 14.3 dB) and 1.3-2.5 dB at 75 deg; the five Table 14.1 values reproduce (C5: one
  table, agents' computations); 4 to 7 test functions pin values a volume change would move.

### G-6 Grain-size relations as printed formulas — OPT-IN, S
- Gap U6; closes the `REFERENCES.md` Sec. 10 "wanted" row. Merges [N1 WP-A] and [N3 WP5]; formulas (11.1)-(11.3)
  [N1 §2.4], delta from (2.6)/(8.26) with c_w = 1528 m/s [N1 §3.1].
- Files: `python/gpusonar/sim/bottom.py` (`apl_grain_size_relations(phi)`; `sediment_from_grain_size(phi,
  method="table_linear" | "relations")`, default unchanged; docstring citations of G-1), new
  `tests/python/test_grain_size_relations.py`.
- Tests first: all 20 rows reproduced (rho 5e-4, nu 1e-4, delta 0.5 %); continuity at 1 and 5.3 phi (rho, nu to
  1e-4); the printed (11.3) steps at 0, 2.6, 4.5, 6.0 documented (C2); the printed 1.0091 NOT used; Fig. 11.2
  normal-incidence loss 6.1 +- 0.5 dB at 0 phi, 14.8 +- 0.7 at 3, 22.2 +- 0.7 at 5 [N1 WP-A].
- Before -> after if made the default: up to 0.68 dB at 5.3 phi, under 0.1 dB outside 5-6 phi, rho up to 0.0106
  at 1.25 phi [N1 §3.2]; no shipped scenario calls the function.
- Must not: change the named rows; copy 1.0091 or 0.4566; call the relations "regressions" (the book: "ad hoc",
  not geoacoustic regressions, p. 314).
- **Done (2026-11-26, branch `g-bottom`)**: `bottom.apl_grain_size_relations(phi, water_sound_speed=1528)`;
  `sediment_from_grain_size(phi, method="table_linear" | "relations")`, default unchanged (bit-identical on a
  0.01 phi grid against values written before the change); `tests/python/test_grain_size_relations.py`, 8 tests.
  All of [N1 3.2]'s numbers reproduced: 0.678 dB at 5.3 phi, nu 0.0041, rho 0.0106 at 1.25 phi, delta 7.6 %
  near 5.79 phi, under 0.1 dB outside 5-6 phi; Fig. 11.2 loss 6.12 / 14.77 / 22.15 dB at 0 / 3 / 5 phi.
  C2 settled: the printed pieces of (11.3) at 6.0 phi give 0.1384 and 0.1397 (step 0.0013): [N1] is right,
  [N3]'s "0.338" is a slip. The other printed steps: +0.0022 at 2.6 (a little above [N1]'s "0.001-0.002"),
  -0.0009 at 4.5, 0 at 0 with 0.4556 (0.0010 with the printed 0.4566). C3: the rows at -1 and -0.5 phi agree
  with 0.4556 to 2.5e-4 and with 0.4566 only to 2.4e-3; 0.4556 used.

### G-7 `extinction: dahl_1997` for `bubble_layer_surface` — OPT-IN, M
- Gap W7 (`sonar-64sm`, `sonar-9le`). From [NS WP2]; formula beta_I = 10^(a + b log10 U10), [D97] Table II
  (26.5 kHz: a -6.16, b 5.50; 50 kHz: -6.70, 6.40; 70 kHz: -5.75, 5.35) [NS §2.8]; interpolation in log f
  between the three frequencies (ours), f^0.8 outside 26.5-70 kHz with a warning; a warning below 2.5 m/s.
- Files: `python/gpusonar/sim/surface.py` (`bubble_layer_optical_depth`; the "lower bound" comment near line
  387, `sonar-64sm`), `sim/simulator.py` (accept the key; shared file, minimal), new
  `tests/python/test_surface_dahl_1997.py`. Engines: Python only.
- Tests first: Table II at the three frequencies; eq. (17) inversion round trip; at 68 kHz, 3 m/s the Gallaudet
  bands -44.1 / -43.4 dB (from -38.3 / -38.1 with `apl_1994`); Urick-Hoover readings of [NS §3.3] within +8 dB;
  wind exponent between 5.0 and 6.3 over 2-7 m/s at 17 deg [NS WP2].
- Before -> after if made the default: the grid of [NS §3.2]: backscatter falls by up to 15.1 dB (50 kHz, 2 m/s)
  at low wind and rises by up to 3.7 dB above 8 m/s; no shipped scenario affected; `test_gallaudet_boundaries.py`
  values move.
- Must not: change the `extinction` default; tune a constant to one figure reading (+-1.5 dB).
- **Done, opt-in** (2026-11-26, branch `g-surface`): `surface.dahl_1997_optical_depth`, `extinction: dahl_1997`;
  `tests/python/test_surface_dahl_1997.py` (9 tests) and one new test in `test_gallaudet_boundaries.py` (its
  existing assertions unchanged). Found: the note's 68 kHz values -44.1 / -43.4 dB were computed with the 70 kHz
  fit scaled as f^0.8, not with the log-f interpolation this package specifies; with the interpolation the
  bands are -44.4 / -43.7 dB (+6.9 / -2.2 against Gallaudet; 0.3 dB apart; both held). Wind exponent at 17 deg:
  5.37 / 6.20 / 5.10 (measured 5.5 / 6.4 / 5.3); `apl_1994` 4.25 / 4.06 / 3.91 with a 51-point fit (the note's
  4.13 / 3.94 / 3.80 not reproduced: that law's exponent depends on the wind sampling, 3.4-4.25); Urick-Hoover
  readings all within +8 dB (max +7.0). Default bit-identical (float.hex values from before). Docs:
  CONFIGURATION.md (`bubble_layer_surface` row), PHYSICS_NOTES Sec. 23.7; the "lower bound" comment in
  `surface.py` corrected (holds at high wind only).

### G-8 Surface model of the FLS scenarios: prepare the decision — OWNER FIRST, S
- Gap U5. From [NS WP7]. No code. The notes give, `bubble_layer_surface` minus Chapman-Harris at 5 m/s: +5.4 /
  +3.0 / -0.4 / -2.6 / -5.0 / -6.6 dB at 5 / 10 / 20 / 30 / 45 / 60 deg at 120 kHz; -2.5 to -7.7 dB at 330 kHz
  [NS §3.6]. To prepare: the same at 170 kHz, and the records of `fls_120k_transit`, `fls_120k_cluttered_seabed`,
  `fls_170k_transit`, `fls_170k_traffic`, `fls_330k_transit` (surface reverberation level per range bin; tests
  that pin them). Must not: switch; claim either model is validated above 70 kHz.

### G-9 Measurement estimators, error bars and fit priors for real data — OPT-IN, S
- Gap: `tools/fit_sediment.py` takes `--sigma-db` with no guidance; no prior or bounds; `sonar-sul`, STATUS
  item 30. Merges [N3 WP6] and [N1 WP-E].
- Formulas: sigma_db = 10 log10(1 + 1/sqrt(N)) (G.2.3: 0.97 dB at N = 16, 0.5 at 67, 0.25 at 260); S_b from a
  calibrated sonar by (G.13) with Psi (G.28)-(G.29); reflection by averaging squared envelopes over patches,
  never dB (Sect. 11.1) [N3 §2.9]. Priors and bounds: density +-0.23 g/cm3 about (5.14) at fixed Mz (read from
  Fig. 5.28, approximate); App. C within-site CVs; a_rho about 1.1 to 2.2, nu 0.97-1.2, sigma_2 0.0001-0.004
  [N1 §4, §2.9]. (G.11)-(G.13) is also the monostatic seafloor reverberation equation that `sonar-w2w3`'s link
  budget lacks: that use is this plan's suggestion, not a note's.
- Files: `tools/fit_sediment.py` (`--independent-samples N`, opt-in `--bounds`), `python/gpusonar/sim/sediment_fit.py`,
  `tests/python/test_sediment_fit.py`. Engines: Python only.
- Tests first: N = 16 gives 0.97 dB; simulated Rayleigh speckle over N cells gives the stated spread; (G.13)
  inverts our own simulated reverberation to S_b within 0.3 dB [N3 WP6]; bounds keep the fit inside.
- Default moves: none unless bounds are made default (fit results where a bound binds: tell the owner).
- **Done (branch `g-fit`, 2026-11-26), opt-in; the default fit and the tool's default output are
  bit-identical to before (test with values from `c1d2b8f`).** New `python/gpusonar/measure.py` (G.2.3
  error, gamma-distribution bias and spread of an N-sample mean (ours), dB-mean bias -2.507 dB and
  5.57/sqrt(N), N from resolution/beams/pings and (16.26) effective looks, (G.13) with the absorption
  misprint corrected, (G.28)-(G.29)); `sediment_fit.fit(independent_samples=, sample_error=, bounds=,
  hold=)`, `prior_bounds(phi)`, `site_bounds(...)`, `roughness_model_spread_db`,
  `SedimentFit.covariance` and `prediction_std_db`; tool options `--independent-samples`,
  `--sample-error`, `--prior-bounds`, `--site-spread`, `--hold`, `--report-covariance`. Found:
  (G.13) inverts our own simulated reverberation to S_b within 0.20 dB at 10-50 deg (Psi ~ Psi_3: up
  to 1.6 dB); the book's +1 sigma used as a dB error under-covers (55 % within 1 sigma against 68 %
  nominal, N = 16; 61 % with the gamma rule); the note's "0.25 dB at N = 260" is 0.261; two Gallaudet
  band means fix neither w2 nor gamma2 (D-193 reproduced) and, with sigma2 free, are fitted better by
  roughness alone (w2 280 times the site's) than by the site's volume model: two values cannot tell
  the mechanisms apart; the prior for 6.36 phi excludes that. Not done: the near-vertical fit model
  (G.43)/(G.54); a fit of a full Rayleigh-statistics likelihood instead of weighted dB least squares.

### G-10 Effective-density fluid (EDFM) for sand — OPT-IN, S (simple) / M (full)
- Gap U4. From [N3 WP3]; formulas (10.82) simple; (10.70), (10.75)-(10.76), (10.81) with (10.13)-(10.16) full;
  Table 10.1 as the documented set "sax99_biot" [N3 §2.4].
- Files: `sim/bottom.py` (`effective_density(f, ...)`; use in `reflection`, perturbation, volume),
  `configs/sediments.yaml` (new row `sax99_sand`: density 2048 kg/m3, porosity 0.385, the Table 10.1 block);
  `paths.cpp` only if the image-source tier uses it; Bellhop takes real density (pass Re, or a reflection file).
- Tests first: F(eps) -> 1 at low frequency; rho_eff -> rho at 10 Hz; high-frequency limit 1755.5 kg/m3; Biot
  fast wave at 20 kHz 1763 m/s, 7.0 dB/m (p. 304; ours 1762.7, 6.96); normal-incidence loss at 20 kHz: EDFM
  9.42 dB, fluid 8.06 dB (Figs. 10.5, 10.7); simple EDFM = rho in a Wood mud; Fig. 10.8 +-0.5 dB for K_g 32-75 GPa.
- Before -> after on SAX99 sand: loss +1.2 to +1.6 dB at normal incidence, +1.0 to +1.2 dB at 45 deg;
  backscatter -0 to -2.9 dB [N3 §3.4].
- **Must not: apply EDFM to the TR 9407 rows** (double count: `medium_sand` would go 8.6 -> 13.0 dB at normal
  incidence, backscatter down to -7.4 dB; meaningless for fine sand) [N3 §3.4]; change the attenuation law of
  existing rows.

### G-11 Wagner shadowing and the book's composite form; validity warning — OPT-IN, S-M; blocked in part
- Gap W4, W5. From [N4 WP-2]; formulas (13.79)-(13.83) [N4 §2.6].
- **First obtain Wagner (1967)** (or another print) to settle pi^(+-1/2) in (13.82). The warning part (sigma_sL >
  0.1, theta > 70 deg, K_c above Bragg, (13.78)) can go ahead now.
- Files: `sim/bottom.py` (`composite_strength`, `combined_strength`: `shadowing="none" | "wagner"`, default
  "none"; optional `slope_average="book_1d"`), `tests/python/test_bottom_models.py`.
- Tests first: R -> 1 for large t, -> 0 as theta -> 0, monotonic; "none" bit-identical to today; composite with R
  against a direct (13.80) at Fig. 13.2 inputs; Fig. 13.19 slopes (G-1).
- Before -> after (opt-in model, composite roughness term at 5 deg): medium sand 30 kHz about -41.6 -> -42.5 dB;
  100 kHz -32.9 -> -35.9; sandy gravel 100 kHz -21.4 -> -27.8 (Q as printed) [N4 WP-2].
- **Must not: implement Q from the book's printed form before the primary paper is checked**; make
  `composite` or `combined` the default.

### G-12 Correlated cross-section texture (McDaniel 1996) — OPT-IN, M
- Gap U8 (`sonar-6fc`; `sonar-6t5` for the texture). From [N2 WP-1]; formulas (16.59)-(16.69) [N2 §2.6].
- Files: `sim/reverb.py` (`CellTexture` or a sibling; keys `model: correlated`, `correlation_length_m`,
  `cross_section_variance`), `docs/PHYSICS_NOTES.md` Sec. 14. Engines: drawn in Python, parity by construction
  (D-196); no C++ or CUDA change.
- Tests first: (16.68) against Monte Carlo for Gaussian Omega and C; the large-cell limit reproduces D-196's area
  rule; Fig. 16.7 plateaux about 3.12 (L_c 1 m) to 1.19 (16 m) at theta 30 deg, L 1 m, gamma 1.5 (ours 3.121,
  1.187); K-equivalent nu = 2/(sigma_I^2 - 1); a still sonar gives correlation 1 between pings, a moving one
  decorrelates over about L; adjacent-cell correlation against the overlap integral (ours to derive).
- Choose and document a positive field (gamma or lognormal; Gaussian with gamma 1.5 goes negative). Seed by
  position.
- Must not: present L = 1 m, gamma = 1.5 as a default for any seabed (one fit, via secondary).

### G-13 Sub-critical penetration: evanescent term plus roughness scattering — OPT-IN, M-L
- Gap W2. From [N3 WP2]; formulas (15.1)-(15.16) [N3 §2.7]; SAX99 inputs from Table 13.1 (a_rho 2.00, nu 1.16,
  delta 0.010, gamma2 3.04, w2 0.0000433) [N4 §2.3] with the von Karman (D.7), K0 = 10 1/m [N3 §2.7].
- Hint: prove our perturbation sediment amplitude B (`bottom.py` docstring) equals A_wp (15.4) first; (15.15) is
  a patch sum like `reverb_relief.py` with one leg in the sediment.
- Tests first: (15.5) diffraction angle and its cutoff (15.6); Fig. 15.6 qualitatively (h0 0.01 m, lambda0
  0.48 m, 10-40 kHz, 17 cm: 10-25 dB one-way below critical); Fig. 15.8 (t_s 100 us, SL 190 dB, 4 m up, 20 m off:
  peak 133-141 dB at 5-40 cm); h -> 0 gives zero; above critical the zeroth order dominates.
- Depends on G-3. Must not: use the coherent flat transmission below the critical angle.

### G-14 Layered or graded fluid seafloor — OPT-IN, M-L
- Gap (`sonar-3k1` layered part). Merges [N3 WP7] and [N4 WP-5]; reflection by the standard recursion of
  [N3 §2.2.9] (cite Olson and Jackson arXiv:2009.12420, not the book); roughness via (13.26); volume via
  (M.36)-(M.47) [N4 §2.3, §2.9]; profiles from (7.36) and Table 7.1 [N2 §2.5].
- Files: `sim/bottom.py` (`layers=`), `propagation.rayleigh_reflection` and `paths.cpp` if image-source paths use
  it (both simulator engines), `configs/sediments.yaml` (optional).
- Tests first: thickness 1e-9 m gives the half-space (1.6e-8); a layer equal to the half-space changes nothing;
  quarter-wave layer |V| 2e-17; a 5 m layer with delta 0.05 gives the layer's own coefficient [N3 §2.2.9];
  (13.26) reduces to (13.23) for a half-space [N4 §2.3]; energy (2.20) with the layered V. The book's 6 dB at
  20 kHz from a 3 cm transition layer is reproducible only if the Mourad-Jackson profile is found (a step layer
  gives 8.6 -> 10.2 dB, ours) [N3 §2.2.8-§2.2.9].
- Must not: treat an elastic top layer as the limit of a vanishing fluid layer (p. 341). Most relevant to the
  8 kHz flank array, which is below the book's 10 kHz-1 MHz range [N4 §2.13].

### G-15 Volume scattering from heterogeneity spectra — OPT-IN, M
- Gap W6, U1 (frequency dependence). Merges [N2 WP-4 (model)] and [N4 WP-4]; (14.10), (14.13)-(14.19),
  (14.26)-(14.28), (7.14)-(7.17), into (14.8).
- Files: `sim/bottom.py` (`volume_model="spectral"`: w3, gamma3, L_c, Lambda, mu, rho_rk), optional site rows from
  Table 14.1 (C7). Engines: Python only.
- Tests first: backscatter closed form (pi |k_p|^4 / 2)(mu^2 - 2 rho_rk mu + 1) W(|Delta k_p|); gamma3 = 3 nearly
  frequency independent (p. 387); the bistatic ratios of [N4 §3.6] (rho_rk -1: -0.5 / -1.2 / -2.2 / -3.4 /
  -4.0 dB at phi 120 / 90 / 60 / 30 / 10; rho_rk 0: up to +10.6); equivalent sigma_2 Arafura 20 kHz 0.0031, San
  Francisco 40 kHz 0.0037 (ours, labelled so); Fig. 14.3's forward null for mu 1, rho_rk -1.
- Must not: replace sigma_2 as default; present a Table 14.1 site value as a class default.

### G-16 Volume term averaged over large-scale slopes — OPT-IN, M
- Gap (`sonar-3k1` "slope and shadow corrections of the volume term"). From [N4 WP-3]; recipe Sect. 14.1.2
  (p. 383) via (13.80) and R from G-11.
- Tests first: zero slope gives exactly today's value; a sin-independent constant averages to itself times the
  shadow factor; reciprocity. Numbers to prepare: medium sand and very fine silt, 30/100 kHz, 5-30 deg.
- Depends on G-11.

### G-17 Elastic rough-interface perturbation (and later elastic volume) for rock — OPT-IN, M (L for volume)
- Gap U3 (a physical rock model instead of Lambert). From [N4 WP-6]; (13.27)-(13.34) with V from (9.34), in the
  small-slope form (13.11); later (14.49)-(14.57) [N4 §2.3, §2.9].
- Tests first: shear -> 0 gives the fluid A_ww; reciprocity; (K.5)/(L.9) identity A(K_i, K_i) = -2 beta^2 V; a
  dip near the Rayleigh angle (Fig. 13.5, description only).
- Depends on G-4. Must not: implement D1 (13.28) without re-reading the page (never evaluated, [N4 §6]); move
  the rock default (D-88) without the owner.

### G-18 1-D to 2-D roughness conversion helper — OPT-IN, S
- From [N2 WP-3]; (D.18), (D.24), (D.19) **corrected**, D.1.3; optional von Karman (D.7)-(D.8).
- Files: `sim/bottom.py` (`roughness_2d_from_1d(phi1, gamma1, units="cm")`), tests.
- Tests first: quadrature of the 2-D law over Ky equals W1 (1e-8); Phi1 integrates to h^2 (von Karman); SAX99
  c-c row by hand: gamma2 3.54, w2 7.54e-3 cm^(4 - gamma2) (ours).
- Must not: use (D.19), (D.23) or 7.1e-5 as printed; ship Table 6.1.

### G-19 Surface reverberation statistics from [DP97] — OPT-IN, M
- From [NS WP5]; lognormal mean with sigma_T = -0.18 U10 + 4.85 dB (3.8-11.6 m/s; clamp and warn), held about
  10 s; exponential within; pings independent beyond O(10) ms [NS §2.9].
- Files: `sim/reverb.py` (sibling of `CellTexture`). Engines: Python texture, parity by construction.
- Tests first: Var(s) = 5.57^2 + sigma_T^2 and E(s) = mu_T - 2.507 dB on 1e5 samples (0.1 dB); SI =
  2 exp((sigma_T/k)^2) - 1 [NS §3.7].
- Spatial correlation of the modulation is not given: say so (tie to G-12).

### G-20 Small-scale surface roughness term, diagnostic only — OPT-IN, M-L
- Gap (`sonar-9le` first item). From [NS WP4]; [McD93] eqs. (4), (8), (22)-(25) with the assumptions of [NS §3.5]
  stated in code (u*, tau, K_c by continuity, Gamma); two-way attenuation of [D97] Sec. VI.
- Tests first: spectrum against Fig. 10 readings (+66 dB re cm^3 at K_p, -17 dB at 1 cm^-1, cut-off 4-5.5 cm^-1,
  +-2 dB); eq. (4) = [D97] eq. (10)/(4 pi); K_L = 0 gives eq. (4); Ku comparison at 30 deg within 3 dB of -49 dB;
  an expected-failure test for the 11-18 dB excess over Urick-Hoover at 60 deg, 2.6-3.9 m/s.
- Depends on G-7. Must not: make it default (it worsens D-193 at 44-66 deg to -25.8 dB, [NS §3.4]).

### G-21 Coastal versus open-ocean surface parameter — OWNER FIRST, M
- From [NS WP3]: about 10 dB at 3-60 kHz before saturation, 5 dB of it attributed to whitecap coverage (factor
  3.5 at 9 m/s); no formula printed [NS §2.6-§2.7]. Only form supported: a named scalar multiplier on beta_I.
- Must not: add it silently or tune it to one data set.

### G-22 Facet term near normal incidence: record the gap — RECORD ONLY, S
- From [NS WP6]: a characterisation test recording Nuetzel 20 kHz 3-4 dB above our facet term at 82-88 deg; the
  "wanted" row for McDaniel (1986). The spectrum's slope variance is 1.5 times Cox-Munk at 30 kHz, 10 m/s
  [NS §3.5], so the fix is not "filter the slopes": read the 1986 paper first.

### G-23 Ch. 5 regressions as an alternative input path — OPT-IN, S-M
- From [N1 WP-B]; Table 5.4 (in situ, 38/58 kHz), Table 5.3 siliciclastic/all (core, 400 kHz), (5.14), Table 4.5,
  Table 5.6 [N1 §2.3]. `sediment_from_properties(porosity=..., density=..., grain_size=..., relations=...)`.
- Tests first: Table 5.5 Richardson-Briggs column (1.163 / 1.162 / 1.149; 1786 / 1785 / 1764 m/s with c_w 1536);
  Fig. 11.2 "standard" curve (7.3 dB at 0 phi, 14.3 at 8 phi, rho_w 1.024); errors outside the validity ranges;
  refuse Table 5.3's attenuation regressions.
- Must not: mix these with the named TR 9407 rows silently (up to 12.5 dB apart for silts, [N1 §3.2]).

### G-24 Seawater check oracles and the density pressure term — OPT-IN, S
- From [N1 WP-D]: Medwin (4.26) and Del Grosso (B.5) as oracles for `sound_speed` (0.3 / 0.4 m/s, 0-30 deg C,
  30-40 PSU, 0-500 m); opt-in `water_density(..., depth=)` by (B.1)/(B.4).
- Tests first: K(35, 25, 1000) = 27108.95 bar, K(0, 5, 0) = 20337.80 bar; depth 0 equals today exactly;
  +0.45 kg/m3 at 100 m (10 deg C, 35 PSU). Files: `acoustics.py`, `tests/python/test_medium_models.py`.

### G-25 Validation cases from the book's test sites — RECORD ONLY, M
- From [N4 WP-10], [N1 §4] (Tables 5.1-5.2 site means), [N2 §4]. Feeds `sonar-sul`. Site parameter sets (how
  many rows: decision 11) and feature tests only (critical-angle maximum present or absent; slope (13.61)) until
  the original papers give numbers. Expect 2-3 dB from measurement error alone (p. 351).
- Must not: digitise the book's figures.

### G-26 Discrete volume scatterers (shells, bubbles); gassy sediments — OPT-IN, M (gassy: L, research)
- From [N4 WP-8], [N1 WP-G]; (14.30)-(14.48), Table 14.2.
- Tests first: fluid-sphere high-frequency limit 0.420 a^2/4 for e = 15.7, h = 1.4 (Fig. 14.10); Rayleigh a^6 k^4;
  bubble-resonance iteration converges and shear raises w0; -66.0 dB re m^2 for a = 1 mm (p. 396).
- Gassy mud needs population data the book does not give (gas fraction, radii 0.25-10 mm, shape, [N1 §2.2]).

### G-27 Anisotropic (rippled) roughness spectrum — OPT-IN, M
- From [N2 WP-5], [N1 WP-G]. The form is ours (the book prints no parametric anisotropic spectrum); values must
  come from a site. Tests: isotropic limit bit-identical; the Bragg peak at 2k cos theta = K_r across crests.
- Must not: use Fig. 6.8 numbers; settle the 500 D / 535 D ripple constant first if Wiberg-Harris is used.

### G-28 K-aware detection threshold — OPT-IN, M
- From [N2 WP-6]; threshold from (16.35). Tests: 1.685 / 2.694 dB at 1e-6 (D-196); Rayleigh as nu -> inf.
  Touches detection: both live-chain engines if it enters the chain. Depends on G-12 for realistic nu.

### G-29 Causal dispersion option — OPT-IN, M (low value)
- From [N1 WP-F]; nu(f) = nu(f_ref)/(1 - (2 delta/pi) ln(f/f_ref)) from (I.28). Tests: Kramers-Kronig (I.25) by
  quadrature; f = f_ref unchanged. Per decade: +0.24 dB, +2.1 deg (sand). Mainly below 20 kHz (flank array).

### G-30 Biot reflection as a test oracle — OPT-IN (tests only), M (low)
- From [N3 WP8]; (10.66)-(10.69). Printed 20 kHz numbers: fast 1763 m/s, 7.0 dB/m; slow 141 m/s, 852 dB/m [N3 §2.4].
  Not for the runtime.

### Counts

| Flag | Packages | Count |
|---|---|---|
| RECORD ONLY | G-1, G-2, G-22, G-25 | 4 |
| OWNER FIRST | G-3, G-5, G-8, G-21 | 4 |
| OPT-IN | G-4, G-6, G-7, G-9 to G-20, G-23, G-24, G-26 to G-30 | 22 |

Merged from about 37 note packages: [N1] WP-A..G (7), [N2] WP-1..6 (6), [N3] WP1..8 (8), [N4] WP-1..10 (10),
[NS] WP1..7 (7). Merges: G-1 (N1 §3.5 + N2 WP-2 + N4 WP-1, WP-7, WP-9), G-4 (N1 WP-C + N3 WP4), G-6 (N1 WP-A +
N3 WP5), G-9 (N1 WP-E + N3 WP6), G-14 (N3 WP7 + N4 WP-5), G-15 (N2 WP-4 + N4 WP-4), G-26 (N4 WP-8 + N1 WP-G
gassy), G-27 (N2 WP-5 + N1 WP-G ripples).

### Must not, across all packages

- Apply EDFM to the TR 9407 rows (double count) [N3 §3.4].
- Implement Wagner's shadowing from the printed (13.82) before Wagner (1967) is checked [N4 §6].
- Use (D.19), (D.23), the 7.1e-5 constant, 1.0091 or 0.4566 as printed; Table 5.3's attenuation regressions.
- Change `sediments.yaml` values, the sigma_2 step, any default model or the `extinction` default without the
  owner; switch any shipped scenario's surface or sediment model without before -> after numbers.
- Present McDaniel's L, gamma, Table 14.1 site values or any figure reading as a class default; digitise the
  book's figures; tabulate Table 6.1 or C.1 in the repository.
- Call the TR 9407 relations "geoacoustic regressions"; describe our layered recursion as the book's.
- Give a row a shear speed without a source.

---

## 2. Recommended first wave

Seven packages, each one agent and one decision entry. The file sets do not overlap, so all seven can run in
parallel (shared file `sim/simulator.py` touched only by G-7, minimally).

| Order | Package | Why first | Files it touches |
|---|---|---|---|
| 1 | G-3 Sediment tier above critical (opt-in code; default switch asked with numbers) | the only error of the set in shipped scenes (2 scenes, 6-9 dB); small; both engines | `sim/propagation.py` (`SedimentProvider`), `libs/sim/src/paths.cpp`, `paths.hpp`, `bind_simcpp.cpp`, `test_sim_cpp.py`, `tests/cpp/test_sim_paths.cpp` |
| 2 | G-1 Pin bottom equalities, relabel | locks every confirmation of 0.2 at no risk; upgrades most "ours" rows | new `tests/python/test_jackson_2007_bottom.py`; `configs/sediments.yaml` comments |
| 3 | G-2 Pin surface equalities, Chapman-Harris docstring | same, surface side; corrects a range we printed without a source | new `tests/python/test_surface_papers_1997.py`; `acoustics.py` docstring |
| 4 | G-6 Grain-size relations as formulas | closes a "wanted" row; S; opt-in | `sim/bottom.py`; new `tests/python/test_grain_size_relations.py` |
| 5 | G-7 `dahl_1997` extinction | removes 5.8 of the 13.0 dB of D-193 at 27-38 deg and all 3.4 dB at 44-66 deg [NS §3.4], opt-in | `sim/surface.py`, `sim/simulator.py`, new test file |
| 6 | G-5 Mud volume parameter table | the largest class-default question (about 5 dB, U1, on the 7 rows from 5.5 to 9 phi [N3 §3.2]) needs only numbers | none (scratch script; numbers in the report) |
| 7 | G-9 Estimators and fit priors | `sonar-sul` (P1) needs error bars and bounds before any real data | `tools/fit_sediment.py`, `sim/sediment_fit.py`, `tests/python/test_sediment_fit.py` |

Second wave (after the first lands): G-4 (after G-3: same files `propagation.py`, `paths.cpp`), G-8 (decision
prep, any time), G-10, G-11's warning part, G-12, G-18. Then G-13 (after G-3), G-15, G-14, G-16 (after G-11),
G-17 (after G-4), G-19, G-20 (after G-7), the rest.

---

## 3. Decisions for the owner

1. **Mud volume parameter** (rows of 5.5 phi and finer, now 0.001): keep, 0.002 (+3.0 dB on the volume term,
   [N2 WP-4]) or 0.003 (+4.8 dB, D-193)? For: Gallaudet's fit 0.003 at 6.36 phi (D-193); Arafura 0.0031 (+4.9 dB)
   and San Francisco 0.0037 (+5.7 dB) by (14.10) [N4 §3.6]. Against or weaker: the last two come from one table;
   the same table's Orcas, Porto Venere and P. della Mariella imply 0.0008, 0.0017, 0.0016 [N2 §3.3]; Snellen 2013
   found fitted sigma_2 at 300 kHz about half the defaults (`REFERENCES.md` 3.1). No shipped scenario uses these
   rows. G-5 prepares the full table.
2. **Medium sand volume parameter (D-90)**: the sources give no new value (range 0.0001-0.004); keep D-90's
   0.001? [N1 §4], [N4 §4].
3. **Silt and clay roughness**: our rule is 1-7 dB below Briggs' data at 10-100 kHz (8 rows, 4 sites) [N2 §3.2];
   record as a known low bias, or change (the note recommends no change)?
4. **Rock reflection**: stay fluid, or elastic with a shear speed: book 40-60 % of V_p (p. 169) or the generic
   rock 1900 m/s (`sediments.yaml` header, via secondary)? Loss per bounce 1.4-7 dB at 5-10 deg, up to 9-19 dB
   near the shear critical angle [N3 §3.3]; [N1 §3.4] gives 4.8-5.5 dB at 40-60 deg (C1). No shipped scene moves.
5. **Rock and gravel Lambert**: keep -16 dB (D-88) or move toward the book's -5.6 dB (+10.4 dB), which the book
   says rests on data exceeding the energy bound; `pebbles` -9 against -11.7 (-2.7 dB) [N4 §3.5]? The note finds
   -16 defensible. No shipped scene moves.
6. **Buried-target tier**: switch `buried_cylinder` and `buried_objects` to the transmission-plus-attenuation
   amplitude (about -6 to -9 dB on the targets at 20-30 kHz, 0.3 m, [N3 WP1]), and name their sediment (they give only a
   speed, 1700 m/s)? G-3 prepares target peaks before -> after.
7. **Low-wind surface level**: keep `apl_1994` as the `bubble_layer_surface` default or make `dahl_1997` the
   default (68 kHz, 3 m/s: -38.3 -> -44.1 dB at 27-38 deg, -38.1 -> -43.4 at 44-66) [NS WP2]? No shipped scenario.
8. **Surface model of the five FLS scenarios** (120-330 kHz): keep Chapman-Harris (outside its 0.4-6.4 kHz range)
   or switch to `bubble_layer_surface` (+5.4 to -6.6 dB over 5-60 deg at 120 kHz; -2.5 to -7.7 dB at 330 kHz,
   5 m/s) [NS WP7]? Nothing is validated above 70 kHz; records change. G-8 prepares numbers.
9. **Coastal versus open-ocean surface** (about 10 dB at 3-60 kHz): add a named site multiplier on beta_I? [NS WP3]
10. **Grain-size table wording, and measured regressions as input**: describe the table everywhere as the TR 9407
    model-fit convention ("ad hoc", p. 314) rather than as sediment properties? Add the Ch. 5 regressions as an
    alternative input for sites with measured porosity or density (up to 12.5 dB apart in normal-incidence loss
    for silts) [N1 §3.2, WP-B]? Make the printed formulas the default between rows (up to 0.68 dB) [N1 §3.2]?
11. **How much of the book's tables may live in tests**: three or four rows per table with page citations [N2],
    or Table 13.1 and Table 14.1 in full as a test-data file [N4 WP-10] (C7)?
12. **Papers to obtain**: Wagner (1967) (blocks G-11), McDaniel (1986) (near-specular surface, G-22), Mourad and
    Jackson (1989), Williams and Jackson (1998), Lyons and Abraham (1999): obtain, or leave wanted? (section 4)
13. **Sediment fit tool** (STATUS item 30, `sonar-5bf2`): switch its model to small-slope? The book shows
    perturbation and small-slope overlap below about 70 deg for medium sand and used perturbation for its own
    model-data fits [N4 §2.5, §4].

---

## 4. What is still needed that these sources do not give

Bibliographic details are pointers: they come from the notes' "wanted" statements, from `docs/REFERENCES.md`
Sec. 10 and from reference lists, and were not checked against the documents, except where marked "read in a
reference list".

| Gap | Paper or data | What is needed |
|---|---|---|
| Bistatic interpolation | Williams and Jackson, J. Acoust. Soc. Am. 103, 169 (1998) (`REFERENCES.md` Sec. 10) | the bistatic cross section and its interpolation; the Fig. 13.8 model [N4 §2.7] |
| Kirchhoff/composite interpolation; origin of the grain-size relations and the 1.0019 constant; the 3 cm transition-layer profile behind "6 dB at 20 kHz" | Mourad and Jackson, Proc. OCEANS '89, 1168 (`REFERENCES.md` Sec. 10) | the interpolation formula; (11.1)-(11.2) as first printed; the density profile [N1 §6], [N3 §2.2.8], [N4 §4] |
| Cut-off rule origin; near-vertical treatment | D. Jackson, Winebrenner and Ishimaru, J. Acoust. Soc. Am. 79, 1410 (1986) (`REFERENCES.md` Sec. 10) | (13.72) as first printed [N4 §2.6] |
| Volume parameters and heterogeneity by site | D. Jackson and Briggs (1992) (cited by the book for Table 14.1 and Table 13.1 Quinault; details not in the notes) | sigma_2 or spectra per site; backscatter as numbers [N4 §2.9, WP-10] |
| Measured K shapes | Lyons and Abraham, J. Acoust. Soc. Am. 106, 1307 (1999) (`REFERENCES.md` Sec. 10) | shapes with beamwidth, bandwidth, grazing band, seabed (80 kHz; the book shows Fig. 16.3 only) [N2 §2.6] |
| Shadowing function sign | Wagner (1967) (cited by the book; details not in the notes); Smith's shadow function for comparison | Q with pi^(+1/2) or pi^(-1/2) [N4 §6] |
| Near-specular surface scattering | McDaniel, "Diffractive corrections to the high-frequency Kirchhoff approximation", J. Acoust. Soc. Am. 79, 952-957 (1986) (read in [McD93]'s reference list, with its factor-2 erratum) | the model; whether it is TR 0702's "McDaniel" model [NS §2.2, §6] |
| Correlated cross-section model | McDaniel (1996) (cited by the book; details not in the notes) | the model as first printed; how L and gamma were fitted to Gensane (1989) [N2 §2.6] |
| Surface reverberation fluctuations | McDaniel, "Sea surface reverberation fluctuations", J. Acoust. Soc. Am. 94, 1551-1559 (1993) (read in [McD93]'s reference list) | statistics to compare with [DP97] [NS §2.2] |
| Surface forward scatter | Dahl, J. Acoust. Soc. Am. 100, 2742 (1996); 105, 2155 (1999); 115, 589 (2004); IEEE J. Oceanic Eng. 26, 141 (2001); Preisig and Deane, J. Acoust. Soc. Am. 116, 2067 (2004) (`REFERENCES.md` Sec. 10) | coherent loss against the Rayleigh parameter; time and angle spread; Doppler spread [NS §4] |
| Handbook small-scale surface roughness | the handbook's G(U); Plant (1986) spectrum (cited in [D97]) | G(U10); the spectrum; a low-wind capillary cut-off law [NS §2.8, §4] |
| Backscatter at the book's test sites as numbers | Stanic et al. 1988, 1989, 1998; D. Jackson et al. 1996a; Williams et al. 2002b; Briggs et al. 2002a; Jones and Jackson 1997; Pouliquen and Lyons 2002 (as named in [N4 WP-10]) | measured curves as numbers [N4 WP-10] |
| Gravel and rock | **data**: the book says no experiment has characterised a gravel or rock seafloor (p. 362) | backscatter with roughness, heterogeneity and shear measured at one site [N4 §2.7], [N1 §2.7] |
| Surface above 70 kHz | **data** | backscatter against wind and grazing at 120-330 kHz with stated averaging [NS §3.6] |
| Scatterer or patch densities | **data** (Abraham and Lyons 2002 say they must be estimated from reverberation) | per m^2 for named seabeds (D-196) |
| Ping-to-ping coherence | **data** | decorrelation time of seafloor and surface reverberation for a still and a moving sonar; the book gives 1 day / 1 month seabed scales [N2 §2.4] and [DP97] O(10) ms at 50 kHz for the surface [NS §2.9] |
| Rock shear speed per site | **data** | V_s, delta_t with V_p, density |
| Millimetre speed and density profiles at 100-500 kHz | **data** (not available, p. 320) | transition layers for G-14 [N3 §2.5] |

---

## 5. Status changes to apply to `docs/REFERENCES.md` (after the documentation sweep, `sonar-tadd`, merges)

Not applied here. "J&R" = Jackson and Richardson (2007), confirmed on the page images unless said.

| Section and row | Now | New | Source that confirms it |
|---|---|---|---|
| 10 "Regressions of density, speed and loss against grain size" | wanted | found: J&R (11.1)-(11.3), pp. 313-314, attributed to Mourad and Jackson 1989 and TR 9407; two printed constants corrected (0.4) | [N1 §3.5], [N3 §4] |
| 3.1 UnderwaterAcoustics.jl / Cintron / Snellen rows (rho, nu, delta) | via secondary | add: equal to the relations as printed in J&R pp. 313-314 (still TR 9407's; "ad hoc" fits) | [N1 §3.5], [N3 §3.2] |
| 3.1 per-parameter w2 rule, gamma2 3.25 | via secondary | unchanged; add: sands at the median of J&R Table 6.1, fines 1-7 dB below | [N2 §3.2, §3.5] |
| 3.1 Bentrem row (`w2_from_cm4`) | confirmed | add second print J&R D.1.3, p. 479 | [N2 §3.1] |
| 3.2 "Our derivation ... perturbation ... volume term" | ours | confirmed: J&R (13.21)-(13.25), (14.8), (14.9) | [N2 §3.3], [N4 §3.1] |
| 3.2 "Kirchhoff prefactor, structure-function constant, ..., small-slope form, composite recipe, combined" | ours | Kirchhoff (13.10), structure constant (D.12), small-slope (13.11): confirmed; composite recipe: structure confirmed (13.80), ours lacks (13.81)-(13.83); combined: ours | [N4 §4] |
| 3.2 Fonseca row (scale split k h_s = 1/2) | confirmed | add: J&R (13.72)-(13.77), attributed to D. Jackson et al. 1986a; slopes of Fig. 13.19 reproduced | [N4 §2.6] |
| 3.2 Muggleworth (Lambert bound) | confirmed | add third print J&R (2.23), p. 26 | [N1 §3.5], [N4 §3.1] |
| 3.2 Olson, Lyons and Saebo (rock) | confirmed | add: J&R confirms the elastic approach for rock (pp. 267, 275-276); no rock roughness law; J&R Ch. 12 Lambert fits rock -5.6, gravel -11.7 dB as a further reference with the book's caveat | [N3 §3.3], [N4 §3.5] |
| 4 Chapman and Harris | unconfirmed | via secondary: [McD93] eq. (27), p. 1918; range 400-6400 Hz, to 15 m/s, low grazing; the docstring's "below about 40 degrees" is not printed | [NS §4] |
| 4 Gallaudet row (bubble and facet terms) | via secondary | the bubble formula confirmed in [D97] (17)-(19); the facet term equal to [McD93] eq. (7); delta in Hz confirmed by [D97] | [NS §3.1] |
| 4 Rayleigh-parameter coherent reflection exp(-Gamma^2/2) | ours | confirmed: J&R (13.14), p. 335, with its caveat | [N3 §3.1], [N4 §2.1] |
| 4 new rows [McD93], [D97], [DP97] | - | confirmed (read on the page images) | [NS §1] |
| 6 new: J&R (16.33)-(16.35) K PDF; McDaniel (1996) model and L = 1 m, gamma = 1.5 | - | confirmed (print); McDaniel's fit: via secondary | [N2 §3.4, §3.5] |
| 7 Francois and Garrison (1982) | via secondary | confirmed: J&R (B.6)-(B.12), p. 461; open-ocean scope S 30-40 ppt | [N1 §3.5] |
| 7 `water_density` | via secondary | confirmed: J&R (B.2)-(B.3), p. 456 (EOS 80, one atmosphere) | [N1 §3.5] |
| 7 Mackenzie (1981) | unconfirmed in print | unchanged; add Medwin (4.26) and Del Grosso (B.5) as check formulas (0.25 / 0.4 m/s) | [N1 §3.5] |
| 7 Hamilton shear regression | via secondary | J&R gives Richardson's (5.10)-(5.13), confirmed; Hamilton's own not printed | [N1 §3.5] |
| 10 "Interpolation between Kirchhoff and composite ...; cut-off; empirical expression" | wanted | cut-off settled ((13.72)-(13.75)); interpolation and large-roughness law stay wanted: J&R, named in the row, does not print them | [N4 §4] |
| 10 "Bistatic bottom model" | wanted | general bistatic formulas confirmed (13.9)-(13.11), (14.10); the interpolation stays wanted | [N4 §4] |
| 10 "Medium sand: volume parameter" | wanted | unchanged (no class table; range 0.0001-0.004, p. 382) | [N1 §3.5], [N4 §4] |
| 10 "K-distribution shape against cell size" | wanted | partly answered: McDaniel 1996 model; no measured shapes | [N2 §3.5] |
| 10 "The handbook's law for beta_I and G(U)" | wanted | beta_I: [D97] Table II, a published fit (whether the handbook's: not stated); G(U), interpolation: not in these papers | [NS §4] |
| 10 "Bubble loss per surface bounce" | wanted | partly: [D97] Sec. VI two-way loss 20 log10(e) beta_I / sin theta (a backscatter inversion) | [NS §4] |
| 10 "Chapman-Harris fit and its range" | wanted | via secondary in [McD93] (see Sec. 4 row) | [NS §4] |
| 11 new row | - | J&R p. 433 "2 nu" patches against D-196 (shape n for n patches): D-196 kept | [N2] top |
| 12.4 dB per wavelength to delta | ours; agrees with an implementation | confirmed: J&R (2.6), (8.18), (8.25)-(8.27), pp. 15, 251-252 | [N1 §3.5], [N3 §3.1] |
| 12.4 Li and Chitre (Rayleigh coefficient) | via secondary | add: equal to J&R (8.49)-(8.53) to 1.8e-15 | [N3 §3.1] |
| 12.4 Olson-Jackson (complex speed) | confirmed | add second print J&R (8.25) | [N3 §3.1] |

---

## 6. What was not verified (items that bear on a package)

| Item | Package | Source |
|---|---|---|
| 1.0091 and 0.4566 judged misprints by continuity and our rows only; TR 9407 / Mourad-Jackson not read | G-6 | [N1 §6], [N3 §6] |
| (11.3) at 6.0 phi: continuous or not (C2) | G-6 | [N3 §2.5], [N1 WP-A] |
| (13.82) sign of pi^(1/2); Wagner not opened | G-11, G-16 | [N4 §6] |
| Elastic D1 (13.28) and the Biot matrices (13.42)-(13.60) retyped, never evaluated | G-17, G-30 | [N4 §6] |
| Rock elastic loss at 40-60 deg differs between notes; [N1] does not state delta_t (C1) | G-4 | [N1 §3.4], [N3 §3.3] |
| Rock properties p. 169: [N3] from extracted text only | G-4 | [N3 §6] |
| rho_eff/rho at 250 kHz printed 0.657 + 0.0046i against 0.863 + 0.006i computed | G-10 | [N3 §6] |
| (D.19) misprint judged by calculation; whether Briggs' Fig. 6.18 used the misprinted form | G-18, U2 | [N2 §6] |
| Fig. 6.19 constant read off the figure (+-20 %) | G-18 | [N2 §6] |
| (D.35) subscript cut off; (7.38) depth units assumed cm | G-14, G-15 | [N2 §6] |
| Table 14.1 finite-L_c rows need Ch. 7's form ([N4] did not read it; [N2] did) | G-15, decision 1 | [N4 §6], [N2 §3.3] |
| Table 13.1 w2 units inferred (m^(4 - gamma2)) from the SAX99 row | G-13, G-25 | [N4 §6] |
| Fig. 14.2, 14.3, 13.18 and every figure-read value: by eye, about 0.5-1.5 dB | G-1, G-15 | [N2 §6], [N4 §6], [NS §6] |
| Fig. 13.1's w3 printed without units | G-15 | [N4 §6] |
| (G.6)-(G.7), (G.13) misprints; App. C, G.2.1, App. M read only in part by [N2] | G-9 | [N3 §6], [N2 §6] |
| Del Grosso pressure: gauge or absolute (0.22 / 0.39 m/s) | G-24 | [N1 §6] |
| Ripple constant 500 D or 535 D | G-27 | [N1 §6] |
| Figs. 5.26-5.27 read off log scales; Table 5.4 units; Fig. 5.28 bounds approximate | G-9, G-23, G-29 | [N1 §6] |
| [McD93] eq. (24): units of u*, log base, K_c not printed; our u*, tau, K_c are assumptions | G-20 | [NS §3.5, §6] |
| [D97] Table II: one site (North Sea), one angle (17 deg), data forced to saturation | G-7 | [NS §6] |
| Wind heights of Urick-Hoover, Garrison, Lilly-McConnell not given; Nuetzel at 47 m | G-7, G-22 | [NS §6] |
| Whether TR 0702's "McDaniel" near-specular model is McDaniel (1986) | G-22 | [NS §6] |
| How Gallaudet's band means were averaged (dB or intensity: 3.4-4.5 dB) | G-7, decision 7 | [NS §3.4], D-193 |
| Not read by any note: J&R App. J beyond [N4]'s pages, L.1-L.2 by [N3]; App. C Table C.1 by [N2]; nothing of TR 9407 | all | [N2 §1], [N3 §6] |
