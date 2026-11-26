# Sediment defaults: the mud volume parameter and the silt roughness (for the owner to decide)

Package G-5 of `docs/tasks/SST_GAPS_JACKSON.md` (its decisions 1, 2 and 3 in section 3; gaps U1 and U2).
Written 2026-11-26 on branch `g-mud`. **Nothing was changed**: no code, no `configs/sediments.yaml` value, no
default, no test. Every number below was computed for this document by the scripts in section 8, with the
repository's own functions (`python/gpusonar/sim/bottom.py`) and the values the notes record; each number names
its source. "J&R" is Jackson and Richardson (2007); [N2] and [N4] are
`docs/references/jackson_richardson_2007/02_roughness_heterogeneity_statistics.md` and `04_scattering_models.md`.

## 0. Answer first

What is uncertain, first:

- **Our seven mud, silt and clay rows (5.5 to 9 phi) are probably 2 to 5 dB too quiet at most measured sites, and
  nobody has measured the class value.** Every site number we have points above our volume parameter 0.001
  except one (Orcas, 0.0008), but the "site numbers" are one fitted value (Gallaudet, 0.003) and five values that
  agents computed from one table of the book (J&R Table 14.1); none is a printed volume parameter for a class.
- **Our 0.001 is the published model's own class value, printed in three open documents**: the step rule of
  Snellen et al. (2013), Snellen et al. (2011) Table II, and the arXiv:1809.08459 row for very fine silt (0.00099),
  all attributed to APL-UW TR 9407. Changing it makes the rows
  stop being the handbook's rows.
- **The roughness rule for silts and clays is 1 to 7 dB below the book's tabulated spectra ([N2] Sec. 3.2) but it
  hardly matters for these rows**: the volume term is at least 14.3 dB above the roughness term at 5 to 60 degrees
  and at least 5.3 dB at 75 degrees, at every frequency (table T4). Raising the roughness to the data's class
  median moves the total by at most 0.45 dB up to 60 degrees and by 1.3 to 2.5 dB at 75 degrees.

What moves, if anything is changed:

| Option | Backscatter of the 7 rows, 5-60 deg | 75 deg | Shipped scenarios | Tests that must be rewritten |
|---|---|---|---|---|
| volume parameter 0.002 | +2.9 to +3.1 dB, every frequency 8-330 kHz | +2.5 to +3.0 dB | **none** | 4 test functions (rows only) to 7 (rule and `very_fine_silt` too), section 4.2 |
| volume parameter 0.003 | +4.7 to +4.8 dB | +4.1 to +4.8 dB | **none** | the same |
| roughness at the data's class median | -0.02 to +0.45 dB | +1.3 to +2.5 dB | **none** | 2 test functions (3 with `very_fine_silt`) |
| both (0.003 and median roughness) | +4.7 to +4.9 dB | +4.7 to +5.7 dB | **none** | the union |

Bottom reverberation level moves by the same dB as the backscattering strength at the grazing angle of each range
cell (checked end to end through the simulator: 4.766 to 4.768 dB on all 12 range blocks for 0.001 -> 0.003, equal
to the strength change to 3e-5 dB; section 3.5).

**My recommendation (Claude's, not the owner's): keep the values and make the uncertainty explicit** (option E,
section 5): state on each of the seven rows that measured sites of this grain size imply 0.0008 to 0.0037
(-1.0 to +5.7 dB on the volume term) and one fitted site 0.003, so a user who needs a realistic mud gives
`sigma2` beside the name or fits the site. Leave the roughness rule as it is and record its low bias. Reason in
section 5.

## 1. The question

1. The seven rows of `configs/sediments.yaml` at 5.5 phi and finer (`medium_silt`, `sandy_mud`, `fine_silt`,
   `sandy_clay`, `very_fine_silt`, `silty_clay`, `clay`) carry a volume parameter of 0.001 (0.000986 for
   `very_fine_silt`), half of the 0.002 of every coarser row.
2. Their roughness is the grain-size rule (w2 0.0005175 cm^4, exponent 3.25 for every row from 5 phi), which lies
   below the measured spectra the book tabulates for silts and clays.
3. Should either move, for those seven rows, and how?

Every row of the table, with its three scattering parameters and where each came from by the file's own
`provenance` fields ("rule" = the regression or step of Snellen et al. 2013 eq. (5) and the text after it;
"one value" = Snellen, Siemes and Simons 2011 Sec. II.A; all three attributed by those papers to APL-UW TR 9407,
which was not read). Table T1:

| Row | Mz (phi) | sigma2 | w2 (m^(4-gamma2)) | w2 (cm^4) | gamma2 | sigma2 from | w2 from | gamma2 from | status | in the question |
|---|---|---|---|---|---|---|---|---|---|---|
| `medium_sand` | 1.5 | 0.001 | 0.000141 | 0.004459 | 3.25 | tabulated, arXiv:1809.08459 Table 1 | tabulated, arXiv:1809.08459 Table 1 | tabulated, arXiv:1809.08459 Table 1 | confirmed | no |
| `very_fine_silt` | 7.5 | 0.00098599 | 1.638e-05 | 0.000518 | 3.25 | tabulated, arXiv:1809.08459 Table 1 | tabulated, arXiv:1809.08459 Table 1 | tabulated, arXiv:1809.08459 Table 1 | confirmed | **yes (sigma2, w2)** |
| `placeholder_sand` | - | 0.002 | 2e-05 | 0.0006325 | 3.25 | invented (`status: unconfirmed`) | invented (`status: unconfirmed`) | invented (`status: unconfirmed`) | unconfirmed | no |
| `rock` | - | - | - | - | - | reflection only; Lambert mu_db -16.0 | - | - | confirmed | no |
| `pebbles` | - | - | - | - | - | reflection only; Lambert mu_db -9.0 | - | - | confirmed | no |
| `sandy_gravel` | -1.0 | 0.002 | 0.0004091 | 0.01294 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | no |
| `very_coarse_sand` | -0.5 | 0.002 | 0.0003343 | 0.01057 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | no |
| `muddy_sandy_gravel` | 0.0 | 0.002 | 0.000272 | 0.008602 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | no |
| `coarse_sand` | 0.5 | 0.002 | 0.00022 | 0.006957 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | no |
| `gravelly_muddy_sand` | 1.0 | 0.002 | 0.0001767 | 0.005587 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | no |
| `muddy_gravel` | 2.0 | 0.002 | 0.0001106 | 0.003498 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | no |
| `fine_sand` | 2.5 | 0.002 | 8.584e-05 | 0.002715 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | no |
| `muddy_sand` | 3.0 | 0.002 | 6.546e-05 | 0.00207 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | w2 only (fine class, 3 data rows) |
| `very_fine_sand` | 3.5 | 0.002 | 4.882e-05 | 0.001544 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | w2 only (fine class, 3 data rows) |
| `clayey_sand` | 4.0 | 0.002 | 3.539e-05 | 0.001119 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | w2 only (fine class, 3 data rows) |
| `coarse_silt` | 4.5 | 0.002 | 2.47e-05 | 0.0007811 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | w2 only (fine class, 3 data rows) |
| `sandy_silt` | 5.0 | 0.002 | 1.636e-05 | 0.0005175 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | w2 only (fine class, 3 data rows) |
| `medium_silt` | 5.5 | 0.001 | 1.636e-05 | 0.0005175 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | **yes (sigma2, w2)** |
| `sandy_mud` | 6.0 | 0.001 | 1.636e-05 | 0.0005175 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | **yes (sigma2, w2)** |
| `fine_silt` | 6.5 | 0.001 | 1.636e-05 | 0.0005175 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | **yes (sigma2, w2)** |
| `sandy_clay` | 7.0 | 0.001 | 1.636e-05 | 0.0005175 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | **yes (sigma2, w2)** |
| `silty_clay` | 8.0 | 0.001 | 1.636e-05 | 0.0005175 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | **yes (sigma2, w2)** |
| `clay` | 9.0 | 0.001 | 1.636e-05 | 0.0005175 | 3.25 | rule: Snellen 2013 step | rule: Snellen 2013 eq. (5) | one value for all (Snellen 2011) | confirmed | **yes (sigma2, w2)** |

Notes on the table. `very_fine_silt` is the one mud row whose numbers are tabulated for that sediment in a cited
source (arXiv:1809.08459 Table 1: -28.6 dB/m volume strength at 1.4 dB/m attenuation gives 0.000986; w2
1.638e-5 m^0.75), and the source agrees with the rule (0.001, 0.0005175 cm^4). `medium_sand` is the D-90 row
(question 2.1(e) below). `placeholder_sand`, `rock` and `pebbles` are not part of the question.

## 2. The evidence, stated without leaning

### 2.1 Volume parameter

(a) Every value we have, with what kind of number it is:

| Value | dB on the volume term vs 0.001 | What it is | Site, grain size, frequency | Where recorded | Independent of the others? |
|---|---|---|---|---|---|
| 0.001 (0.00099) | 0 | **printed class value** of the published model: step rule "0.001 from 5.5 phi" (Snellen 2013 text after eq. (5)); Snellen 2011 Table II prints 0.001 at 6, 7, 8, 9 phi; arXiv:1809.08459 prints very fine silt as -28.6 dB/m at 1.4 dB/m = 0.00099 | class, 5.5-9 phi, no frequency | `configs/sediments.yaml` provenance; `test_sediment_literature.py` | the three prints all go back to TR 9407 |
| 0.0031 | +4.9 | **agent's computation** from a measured heterogeneity spectrum, J&R Table 14.1 inputs, by (14.10) with rho_rk = -1, L_c infinite | Arafura Sea, 5.6 phi, 20 kHz | [N2] Sec. 3.3, [N4] Sec. 3.6; recomputed here: 0.00307 | one table (see C5 below) |
| 0.0037 | +5.7 | same | San Francisco, 6.4 phi, 40 kHz | [N2] Sec. 3.3, [N4] Sec. 3.6; recomputed: 0.00370 | one table |
| 0.0008 | -1.0 | same, finite L_c 0.032 m by (7.14) | Orcas, 8.1 phi, 40 kHz | [N2] Sec. 3.3 only; recomputed: 0.00082 | one table |
| 0.0017 | +2.3 | same, L_c 0.0014 m, gamma3 7.6 | Porto Venere, 9.1 phi, 140 kHz | [N2] Sec. 3.3 only; recomputed: 0.00169 | one table |
| 0.0016 | +2.1 | same, L_c 0.0012 m, gamma3 8.0 | P. della Mariella, 7.2 phi, 140 kHz | [N2] Sec. 3.3 only; recomputed: 0.00162 | one table |
| 0.003 | +4.8 | **printed fitted value** (Gallaudet 2001, Ch. 3 Table VI, p. 60: the composite model fitted to measured backscatter, exponent 3.00) | NE Gulf of Mexico, silt-sand, 6.36 phi (from porosity), 68 kHz | `docs/DECISIONS.md` D-193; `tests/python/test_gallaudet_boundaries.py` | yes (another source, another site) |
| 0.0001-0.004 | -10 to +6.0 | the book's **stated typical range**; above 0.004 "may indicate failure of the weak scattering assumption" | - | J&R p. 382, [N2] Sec. 2.5, [N4] Sec. 2.9 | - |
| 0.002 | +3.0 | the value **the book's example uses** (Fig. 14.2, 30 kHz, a_rho 1.5) | - | J&R p. 382, [N4] Sec. 2.9 | an example, not a measurement |

**Contradiction C5 of the plan, stated honestly.** [N4] Sec. 3.6 computed two sites (Arafura 0.0031, San Francisco
0.0037, the two rows with an infinite correlation length) and wrote that they "agree with Gallaudet's 0.003"; it
did not compute the three finite-L_c rows because it had not read Ch. 7. [N2] Sec. 3.3 computed all five and called
the support "weak". Both are right about what they computed. What this document adds: (1) all five values
reproduce to the notes' digits from the inputs [N2] Sec. 2.5 types, with (14.10) in backscatter and the spectrum
(7.14), Lambda = 1, and a water speed of 1500 m/s that is OUR assumption (`sites.py`; flat over 10-60 degrees to
1 %); (2) **the five are not five independent measurements of the class**: they are five rows of one table, pushed
through one model with assumptions the book itself calls unproven (rho_rk = -1 "never measured", p. 389; mu from
the speed/density variance ratio with rho_rp = 1 for three sites and eta = 1/3 assumed for two, [N2] Sec. 2.5);
(3) they are **agents' computations from measured spectra, not printed volume parameters**; the book prints no
sigma_2 for any sediment class ([N2] Sec. 3.3, [N4] Sec. 4). Arafura and San Francisco are therefore one
argument, not two; Gallaudet's fit is the only independent value, from one site.

Their median is 0.0017 (+2.3 dB); their geometric mean 0.0019; with Gallaudet's 0.003, 0.0021 (ours, simple
statistics of the six numbers above). Their spread, 0.0008 to 0.0037, is 6.7 dB: larger than the change in
question.

(b) The implied value depends on frequency (gamma3 is not 3). Extrapolating the two infinite-L_c spectra beyond the
frequency they were measured at (ours; not data; the book's model-data comparisons are 20-140 kHz, Fig. 14.4):
Arafura 0.0046 / 0.0026 / 0.0018 / 0.0014 / 0.0009 and San Francisco 0.0102 / 0.0044 / 0.0027 / 0.0019 / 0.0010
at 8 / 30 / 68 / 120 / 330 kHz. At the forward-looking sonars' 120-330 kHz both approach our 0.001; at 8 kHz San
Francisco exceeds the book's 0.004 multiple-scattering flag. The finite-L_c rows (gamma3 4.0 to 8.0) give
meaningless values far from their measurement frequency (`sites.py` prints them) and are not extrapolated here.

(c) Other numbers in the repository that bear on the level (not on sigma_2 itself):

- Measured level: "sandy mud and muddy very fine sand about -22 dB" at 45 degrees with a calibrated echosounder,
  35-440 kHz (Fezzani et al. 2025, as `docs/PHYSICS_NOTES.md` Sec. 23.1 reads it). Ours for `sandy_mud` at 45
  degrees: -28.3 dB now, -25.3 with 0.002, -23.6 with 0.003 (`pins.py`, 6-300 kHz alike).
- The book's Lambert fit for mud (Mz >= 5 phi, data at 20-140 kHz below 60 degrees, spread +-5-8 dB): -19.7 dB
  (J&R Ch. 12, [N4] Sec. 2.11). The Lambert parameter our rows amount to (mean of S_b - 20 log10 sin over 5-60
  degrees, ours, `lambert_eq.py`): -23.7 to -24.0 dB now, -20.7 to -21.0 with 0.002, -18.9 to -19.2 with 0.003,
  at 30 and 120 kHz alike.
- At Gallaudet's site (D-193 bands, 68 kHz): `fine_silt` now 5.0 dB low at 48-66 degrees and 0.8 dB low at 29-37;
  with 0.002 2.0 dB low and 2.2 dB high; with 0.003 0.2 dB low and 3.9 dB high (equal to the site fit within
  0.01 dB). The source's band means carry up to -9 dB of transmit-pattern bias and reverberation in both bands
  (D-193), so they cannot rank 0.002 against 0.003.
- Against: Snellen et al. (2013) found fitted sigma_2 about half the rule at 300 kHz, but for sediments coarser than
  5.5 phi (`bottom.py` comment above `GRAIN_SIZE_GAMMA2`); it says nothing about muds.

(d) The book's own words on what decides it: sigma_2 "has to be adjusted empirically" (GeoHab report, quoted in
`configs/sediments.yaml`); the book: heterogeneity inputs are uncertain (core-section bias, rho_rk unmeasured,
p. 389), model-data comparisons at five soft sites are "satisfactory" (Fig. 14.4, a figure only).

(e) **Medium sand (D-90): nothing new.** The book prints no sigma_2 for medium sand or any class; its typical range
0.0001-0.004 contains both 0.001 and 0.002; its example uses 0.002 (Fig. 14.2) ([N2] Sec. 3.3, [N4] Sec. 4). Its
Fig. 13.2 roughness inputs equal our `medium_sand` row exactly ([N4] Sec. 3.1). For reference, 0.002 on
`medium_sand` would move it by +0.1 to +2.4 dB (most at 8 kHz and 45 degrees; computed here), as D-90 says.

### 2.2 Roughness

All from [N2] Sec. 3.2 (an agent's comparison of our rule `grain_size_scattering(Mz)` with the 52 rows of Briggs'
J&R Table 6.1, converted to 2-D with the corrected (D.19), evaluated at the backscatter Bragg wavenumber
K = 2 k cos 30 deg; positive = the data rougher than our rule):

| Frequency (K) | Sands, Mz < 3 (41 rows): median (quartiles; range) | Fine, 3 <= Mz < 5.5 (3 rows) | Mz >= 5.5 (8 rows): median (range) |
|---|---|---|---|
| 10 kHz (73 rad/m) | +1.4 dB (-1.6 / +3.0; -12.4 to +9.5) | +7.8 | +7.1 (+2.9 to +14.5) |
| 100 kHz (726 rad/m) | -0.0 (-4.7 / +2.0; -13.1 to +7.4) | +7.0 | +2.2 (-2.4 to +13.4) |
| 300 kHz (2177 rad/m) | -0.6 (-6.3 / +2.2; -15.3 to +9.9) | +7.4 | -0.9 (-7.1 to +13.2) |

- Sands: our rule sits at the median within 1.4 dB from 10 to 300 kHz; the spread is about 20 dB (-13 to +10 dB).
- Silts and clays (Mz >= 5.5): our rule is 1 to 7 dB below the median at 10-100 kHz and 0.9 dB above it at 300 kHz.
  The eight rows come from four sites, two of them glacial till and carbonate: "weak evidence ... not a reason to
  change a default" ([N2] Sec. 3.2).
- Exponent: the data's median gamma2 is 3.43 (quartiles 3.18-3.71); 3.65 for Mz >= 5.5; ours 3.25 for all.
- The book: "no empirical relationships are evident" between roughness and grain size; "sediment mean grain size
  alone may never yield" predictive relations (J&R pp. 193-195, [N2] Sec. 2.3); its own measurement accuracy is
  +2/-3 dB, so differences up to 4 dB "do not necessarily imply disagreement" (p. 184, [N2] Sec. 2.2).
- Gallaudet's fit used our rule's w2 (0.000518 cm^4) with exponent 3.00, within Table 6.1's spread for muds.

## 3. Before -> after, computed now

### 3.1 How

`bottom.named_backscattering_strength_db(name, grazing, f, **override)` with the default roughness model
(`small_slope`, D-86), sound speed 1500 m/s (`NOMINAL_SOUND_SPEED`), grazing 5, 10, 20, 30, 45, 60, 75 degrees,
8, 30, 68, 120, 330 kHz; the two terms from `bottom.roughness_strength` and `bottom.volume_strength`. Options are
passed as keyword overrides (`sigma2=0.002`); the file is not touched.

**The class-median roughness used (ours, from [N2]'s three numbers only).** The note gives the median offset of the
Mz >= 5.5 data over our rule at three wavenumbers: +7.1 dB at K = 72.6 rad/m (10 kHz), +2.2 dB at 725.5 rad/m
(100 kHz), -0.9 dB at 2176.6 rad/m (300 kHz). One power law fitted through them by least squares in log K
(`tables.py`): offset = 17.14 dB - 5.34 dB x log10(K / 1 rad/m), residuals -0.11, +0.33, -0.22 dB. That is
w2' = 51.72 x the rule's w2 = 8.46e-4 m^(4 - gamma2') with gamma2' = 3.784 (the note's median exponent for this
class is 3.65). It reproduces the note's offsets at the three wavenumbers as +7.21, +1.87, -0.68 dB. Outside 73 to
2177 rad/m (8 kHz at steep angles, 330 kHz at low angles) it is an extrapolation. It is a construction for this
table, not a proposed row: Table 6.1 itself is not tabulated in the repository ([N2] WP-3).

### 3.2 Table T2: `fine_silt` (the class that holds Gallaudet's site), dB, change in brackets

| f (kHz) | case | 5 deg | 10 deg | 20 deg | 30 deg | 45 deg | 60 deg | 75 deg |
|---|---|---|---|---|---|---|---|---|
| 8 | now (sigma2 0.001, rule w2) | -41.6 | -35.8 | -31.8 | -30.0 | -28.3 | -27.4 | -26.9 |
| 8 | sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.4 (+3.0) | -23.9 (+3.0) |
| 8 | sigma2 0.003 | -36.8 (+4.8) | -31.1 (+4.8) | -27.1 (+4.8) | -25.2 (+4.8) | -23.6 (+4.8) | -22.7 (+4.8) | -22.1 (+4.7) |
| 8 | class-median roughness | -41.6 (+0.0) | -35.8 (+0.0) | -31.8 (+0.0) | -29.9 (+0.0) | -28.3 (+0.0) | -27.3 (+0.1) | -24.6 (+2.3) |
| 8 | median roughness + sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.4 (+3.1) | -22.6 (+4.3) |
| 8 | median roughness + sigma2 0.003 | -36.8 (+4.8) | -31.1 (+4.8) | -27.1 (+4.8) | -25.2 (+4.8) | -23.6 (+4.8) | -22.6 (+4.8) | -21.2 (+5.6) |
| 30 | now (sigma2 0.001, rule w2) | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.4 | -26.8 |
| 30 | sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.4 (+3.0) | -23.8 (+2.9) |
| 30 | sigma2 0.003 | -36.8 (+4.8) | -31.1 (+4.8) | -27.1 (+4.8) | -25.2 (+4.8) | -23.6 (+4.8) | -22.7 (+4.8) | -22.1 (+4.7) |
| 30 | class-median roughness | -41.6 (+0.0) | -35.8 (+0.0) | -31.8 (+0.0) | -29.9 (+0.0) | -28.3 (+0.0) | -27.2 (+0.2) | -24.4 (+2.4) |
| 30 | median roughness + sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.3 (+3.1) | -22.5 (+4.3) |
| 30 | median roughness + sigma2 0.003 | -36.8 (+4.8) | -31.1 (+4.8) | -27.0 (+4.8) | -25.2 (+4.8) | -23.6 (+4.8) | -22.6 (+4.8) | -21.1 (+5.6) |
| 68 | now (sigma2 0.001, rule w2) | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.4 | -26.6 |
| 68 | sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.4 (+3.0) | -23.8 (+2.8) |
| 68 | sigma2 0.003 | -36.8 (+4.8) | -31.1 (+4.8) | -27.1 (+4.8) | -25.2 (+4.8) | -23.6 (+4.8) | -22.6 (+4.7) | -22.1 (+4.6) |
| 68 | class-median roughness | -41.6 (+0.0) | -35.8 (+0.0) | -31.8 (+0.0) | -29.9 (+0.0) | -28.3 (+0.0) | -27.1 (+0.3) | -24.3 (+2.3) |
| 68 | median roughness + sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.3 (+3.1) | -22.4 (+4.2) |
| 68 | median roughness + sigma2 0.003 | -36.8 (+4.8) | -31.1 (+4.8) | -27.0 (+4.8) | -25.2 (+4.8) | -23.6 (+4.8) | -22.5 (+4.8) | -21.1 (+5.5) |
| 120 | now (sigma2 0.001, rule w2) | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.4 | -26.4 |
| 120 | sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.4 (+3.0) | -23.7 (+2.8) |
| 120 | sigma2 0.003 | -36.8 (+4.7) | -31.1 (+4.8) | -27.0 (+4.8) | -25.2 (+4.8) | -23.6 (+4.8) | -22.6 (+4.7) | -22.0 (+4.4) |
| 120 | class-median roughness | -41.5 (+0.0) | -35.8 (+0.0) | -31.8 (+0.0) | -29.9 (+0.0) | -28.3 (+0.0) | -27.0 (+0.3) | -24.4 (+2.1) |
| 120 | median roughness + sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.2 (+3.1) | -22.5 (+4.0) |
| 120 | median roughness + sigma2 0.003 | -36.8 (+4.8) | -31.0 (+4.8) | -27.0 (+4.8) | -25.2 (+4.8) | -23.6 (+4.8) | -22.5 (+4.8) | -21.1 (+5.3) |
| 330 | now (sigma2 0.001, rule w2) | -41.5 | -35.8 | -31.8 | -29.9 | -28.3 | -27.3 | -25.9 |
| 330 | sigma2 0.002 | -38.5 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.3 (+2.9) | -23.4 (+2.5) |
| 330 | sigma2 0.003 | -36.8 (+4.7) | -31.0 (+4.7) | -27.0 (+4.7) | -25.2 (+4.7) | -23.6 (+4.7) | -22.6 (+4.7) | -21.8 (+4.1) |
| 330 | class-median roughness | -41.5 (-0.0) | -35.8 (-0.0) | -31.8 (-0.0) | -29.9 (-0.0) | -28.3 (+0.0) | -26.8 (+0.4) | -24.5 (+1.4) |
| 330 | median roughness + sigma2 0.002 | -38.6 (+3.0) | -32.8 (+3.0) | -28.8 (+3.0) | -26.9 (+3.0) | -25.3 (+3.0) | -24.1 (+3.2) | -22.5 (+3.3) |
| 330 | median roughness + sigma2 0.003 | -36.8 (+4.7) | -31.0 (+4.7) | -27.0 (+4.7) | -25.2 (+4.7) | -23.5 (+4.7) | -22.5 (+4.8) | -21.2 (+4.7) |

The changes agree across the seven rows within 0.1 dB below 75 degrees and within 0.35 dB at 75 degrees (table T3;
Appendix A has every row in absolute dB).

### 3.3 Table T3: all seven rows, change against now, dB (min to max over the rows)

| f (kHz) | option | 5-45 deg | 60 deg | 75 deg |
|---|---|---|---|---|
| 8 | sigma2 0.002 | +3.01 to +3.07 | +3.01 to +3.07 | +2.98 to +3.04 |
| 8 | sigma2 0.003 | +4.77 to +4.83 | +4.77 to +4.83 | +4.73 to +4.79 |
| 8 | class-median roughness | +0.01 to +0.03 | +0.11 to +0.13 | +2.07 to +2.39 |
| 8 | median roughness + sigma2 0.002 | +3.01 to +3.08 | +3.06 to +3.13 | +4.15 to +4.34 |
| 8 | median roughness + sigma2 0.003 | +4.77 to +4.84 | +4.80 to +4.87 | +5.54 to +5.69 |
| 30 | sigma2 0.002 | +3.00 to +3.07 | +3.00 to +3.06 | +2.92 to +2.99 |
| 30 | sigma2 0.003 | +4.76 to +4.83 | +4.75 to +4.82 | +4.65 to +4.72 |
| 30 | class-median roughness | +0.01 to +0.03 | +0.18 to +0.21 | +2.17 to +2.49 |
| 30 | median roughness + sigma2 0.002 | +3.01 to +3.08 | +3.09 to +3.16 | +4.17 to +4.36 |
| 30 | median roughness + sigma2 0.003 | +4.77 to +4.84 | +4.82 to +4.88 | +5.53 to +5.67 |
| 68 | sigma2 0.002 | +3.00 to +3.07 | +2.99 to +3.05 | +2.84 to +2.91 |
| 68 | sigma2 0.003 | +4.75 to +4.82 | +4.74 to +4.80 | +4.54 to +4.62 |
| 68 | class-median roughness | +0.01 to +0.03 | +0.24 to +0.29 | +2.06 to +2.36 |
| 68 | median roughness + sigma2 0.002 | +3.01 to +3.08 | +3.11 to +3.18 | +4.05 to +4.22 |
| 68 | median roughness + sigma2 0.003 | +4.76 to +4.83 | +4.83 to +4.89 | +5.41 to +5.52 |
| 120 | sigma2 0.002 | +2.99 to +3.06 | +2.97 to +3.04 | +2.75 to +2.83 |
| 120 | sigma2 0.003 | +4.74 to +4.82 | +4.72 to +4.79 | +4.42 to +4.51 |
| 120 | class-median roughness | +0.01 to +0.04 | +0.29 to +0.35 | +1.88 to +2.15 |
| 120 | median roughness + sigma2 0.002 | +3.00 to +3.07 | +3.13 to +3.20 | +3.88 to +4.02 |
| 120 | median roughness + sigma2 0.003 | +4.75 to +4.83 | +4.83 to +4.90 | +5.24 to +5.34 |
| 330 | sigma2 0.002 | +2.97 to +3.05 | +2.93 to +3.00 | +2.49 to +2.57 |
| 330 | sigma2 0.003 | +4.71 to +4.80 | +4.67 to +4.74 | +4.06 to +4.16 |
| 330 | class-median roughness | -0.02 to +0.03 | +0.38 to +0.45 | +1.28 to +1.45 |
| 330 | median roughness + sigma2 0.002 | +2.96 to +3.05 | +3.14 to +3.21 | +3.31 to +3.38 |
| 330 | median roughness + sigma2 0.003 | +4.71 to +4.80 | +4.82 to +4.88 | +4.68 to +4.74 |

`very_fine_silt` moves 0.06 dB more than the others under 0.002 and 0.003 (its present value is 0.000986).

### 3.4 Table T4: which term dominates, volume minus roughness, dB (smallest over the seven rows)

| f (kHz) | case | 5 deg | 10 deg | 20 deg | 30 deg | 45 deg | 60 deg | 75 deg |
|---|---|---|---|---|---|---|---|---|
| 8 | now | +29.2 | +30.1 | +31.6 | +32.0 | +30.7 | +26.8 | +18.0 |
| 8 | median roughness, sigma2 0.001 | +21.8 | +22.7 | +24.0 | +24.0 | +21.6 | +14.9 | +1.2 |
| 8 | median roughness, sigma2 0.003 | +26.5 | +27.5 | +28.7 | +28.7 | +26.4 | +19.7 | +6.0 |
| 30 | now | +24.9 | +25.8 | +27.3 | +27.7 | +26.3 | +22.4 | +13.6 |
| 30 | median roughness, sigma2 0.001 | +20.5 | +21.4 | +22.7 | +22.6 | +20.1 | +12.5 | +0.7 |
| 30 | median roughness, sigma2 0.003 | +25.3 | +26.2 | +27.4 | +27.4 | +24.9 | +17.3 | +5.5 |
| 68 | now | +22.2 | +23.2 | +24.6 | +25.0 | +23.7 | +19.7 | +10.8 |
| 68 | median roughness, sigma2 0.001 | +19.7 | +20.7 | +21.9 | +21.8 | +19.1 | +11.0 | +0.6 |
| 68 | median roughness, sigma2 0.003 | +24.5 | +25.4 | +26.7 | +26.6 | +23.8 | +15.7 | +5.4 |
| 120 | now | +20.3 | +21.3 | +22.8 | +23.1 | +21.8 | +17.8 | +8.8 |
| 120 | median roughness, sigma2 0.001 | +19.2 | +20.1 | +21.3 | +21.2 | +18.3 | +9.9 | +0.7 |
| 120 | median roughness, sigma2 0.003 | +24.0 | +24.9 | +26.1 | +26.0 | +23.1 | +14.7 | +5.5 |
| 330 | now | +17.1 | +18.0 | +19.5 | +19.8 | +18.4 | +14.3 | +5.3 |
| 330 | median roughness, sigma2 0.001 | +18.3 | +19.2 | +20.3 | +20.1 | +16.8 | +8.2 | +0.9 |
| 330 | median roughness, sigma2 0.003 | +23.0 | +23.9 | +25.1 | +24.9 | +21.6 | +13.0 | +5.7 |

**The volume term dominates every row at every angle and frequency, before and after** every option. The margin is
14.3 to 32 dB below 60 degrees now; at 75 degrees it is 5.3 to 18 dB now and falls to 0.6 to 1.2 dB with the
class-median roughness alone (roughness nearly equal to volume there), 5.4 to 6.0 dB with median roughness and
0.003. That is why the roughness question moves the total only near 75 degrees.

### 3.5 Bottom reverberation level

Reverberation level is source level minus two-way transmission loss plus S_b plus 10 log10 of the ensonified area
(G.11-G.13 in J&R, [N3] Sec. 2.9); only S_b depends on the sediment row, so **the change in bottom reverberation
level from a range cell equals the change of S_b at that cell's grazing angle**, the numbers of 3.2-3.3. Checked end
to end (`rl_check.py`): the shipped closed-form scenario `sst_sediment_reverb` (20 kHz, sonar 20 m above a flat
bottom) run through `gpusonar.sim.run_scenario` with its bottom swapped to `fine_silt` by a loader override, once
as is and once with `sigma2: 0.003` in the reverberation item, same seed: the mean level of all 12 range blocks
(8.5 to 35.6 degrees grazing) rose by 4.766 to 4.768 dB, equal to the S_b change to 3e-5 dB; the run's level
against the scenario's closed-form integral is -0.18 to +0.58 dB (speckle, 6 pings), as for `medium_sand`.

### 3.6 Second computations (all in `tables.py` unless named)

| Check | Result |
|---|---|
| volume term change = 10 log10(new / old sigma_2), every row, frequency and angle (sigma_v is linear in sigma_2) | equal to 8.9e-15 dB |
| total change = 10 log10((r + v') / (r + v)) recomputed by hand from the two terms | equal to 8.0e-15 dB |
| where volume dominates by >= 10 dB, total change against 10 log10(ratio) | within 0.23 dB (the roughness share) |
| roughness term with w2 scaled by the note's offset (gamma2 fixed): perturbation theory is linear in w2 | perturbation: equal to 1.3e-15 dB; small-slope default: within 0.28 dB (its nonlinearity near 75 degrees) |
| the fitted median spectrum against a second construction (w2 scaled per frequency by the note's offset interpolated in log f, ends held), roughness term at 30 degrees, `fine_silt` | +8.0 / +5.0 / +3.2 / +1.9 / -0.3 dB against +7.1 / +4.8 / +3.0 / +1.7 / -0.9 dB at 8 / 30 / 68 / 120 / 330 kHz: within 0.9 dB, both outside the note's band at 8 and 330 kHz |
| the five Table 14.1 values by a second, repository-free implementation of (14.10) (`sites.py`) | 0.00307, 0.00370, 0.00082, 0.00169, 0.00162 against the notes' 0.0031, 0.0037, 0.0008, 0.0017, 0.0016 |
| `fine_silt` now against D-193's band means (`pins.py`, with the test's own band averaging) | -4.95 / -0.84 dB against D-193's 5.0 / 0.8 dB low |
| reverberation level change against S_b change through the simulator (`rl_check.py`) | 3e-5 dB |

### 3.7 Rows of 3 to 5 phi (outside the question, for completeness)

[N2]'s "fine" class (3 <= Mz < 5.5; three data rows only) sits +7.0 to +7.8 dB above our rule; fitted the same way:
w2 x 6.83, gamma2 3.285. On `muddy_sand` to `sandy_silt` (volume parameter 0.002) the total moves (table T5):

| row | 8 kHz: 5-60 deg / 75 deg | 30 kHz: 5-60 deg / 75 deg | 68 kHz: 5-60 deg / 75 deg | 120 kHz: 5-60 deg / 75 deg | 330 kHz: 5-60 deg / 75 deg |
|---|---|---|---|---|---|
| `muddy_sand` | +0.3 to +1.3 / +3.4 | +0.7 to +2.6 / +2.8 | +1.3 to +3.6 / +0.4 | +1.8 to +4.2 / -1.6 | -0.1 to +5.3 / -3.7 |
| `very_fine_sand` | +0.1 to +0.6 / +2.2 | +0.3 to +1.2 / +2.8 | +0.6 to +1.9 / +1.4 | +0.9 to +2.5 / -0.2 | +0.6 to +3.7 / -2.6 |
| `clayey_sand` | +0.1 to +0.2 / +1.2 | +0.1 to +0.5 / +2.2 | +0.3 to +0.9 / +1.9 | +0.4 to +1.2 / +0.9 | +0.8 to +2.0 / -1.4 |
| `coarse_silt` | +0.0 to +0.1 / +0.6 | +0.1 to +0.2 / +1.5 | +0.1 to +0.4 / +1.7 | +0.2 to +0.6 / +1.4 | +0.4 to +0.9 / -0.3 |
| `sandy_silt` | +0.0 to +0.0 / +0.3 | +0.0 to +0.1 / +0.7 | +0.0 to +0.2 / +1.1 | +0.1 to +0.3 / +1.3 | +0.2 to +0.5 / +0.6 |

Here roughness matters more (the coarser rows are not volume dominated); near 75 degrees the small-slope result
falls when the roughness rises (`muddy_sand` -3.7 dB at 330 kHz). Three data rows are not a basis for a change; this
is recorded, not proposed.

## 4. Exposure

### 4.1 Shipped scenarios: none uses an affected row

All 80 files in `configs/scenarios/` were expanded with the repository's loader
(`gpusonar.config.load_scenario_config`, which resolves `environment:` names and `base:` chains from
`configs/environments/` and `configs/builds/environments/` and fills the propagation and reverberation blocks);
every string under a `bottom` or `sediment` key, before and after expansion, was collected (`exposure.py`). The only
row of `configs/sediments.yaml` any scenario names is `medium_sand`. **No scenario uses a row of 5.5 phi or finer, and
none uses a row of 3 to 5 phi.** The two scenes with "mud" and "silt" in their names use typed three-parameter
half-spaces and Lambert reverberation, not the table.

What each scenario's bottom is (environment after expansion):

| Bottom | Scenarios | Names |
|---|---|---|
| sediment tier, sediment_speed 1700 m/s only | 2 | `buried_cylinder`, `buried_objects` |
| fluid half-space speed 1600 m/s, density 1.8, 0.5 dB/wavelength | 11 | `fjord_static_trial_2row`, `fjord_static_trial_3row`, `fjord_testauv_trial_1row`, `flank_patrol`, `flank_target_10km`, `flank_target_1km`, `flank_target_5km`, `popoto_far_mover`, `popoto_trial_1000m`, `popoto_trial_100m`, `popoto_trial_2000m` |
| fluid half-space speed 1650 m/s, density 1.9, 0.8 dB/wavelength | 25 | `flank_sst_shallow`, `flank_sst_spread`, `fls_120k_quay_multipath`, `fls_120k_transit`, `fls_170k_traffic`, `fls_170k_transit`, `fls_330k_transit`, `popoto_12km`, `popoto_12km_2row`, `popoto_12km_3row`, `popoto_12km_two_ships`, `popoto_elevation_rehearsal`, `popoto_elevation_shallow`, `popoto_env_winter_sand_200m`, `popoto_face_2row_shallow`, `popoto_static_passing`, `popoto_static_trial_2row`, `popoto_static_trial_3row`, `popoto_testauv_direct_overload`, `popoto_testauv_direct_overload_fast`, `popoto_testauv_direct_overload_gain_switch`, `popoto_testauv_trial_1row`, `ssp_front_static_trial_2row`, `ssp_front_testauv_trial_1row`, `ssp_thermocline_static_trial_3row` |
| no bottom in the environment or propagation (free field, or walls only; some carry a `seafloor` scene item, below) | 35 | `fls_120k_cluttered_seabed`, `fls_120k_rough_surface`, `fls_glints_on_sediment`, `fls_highlight_target_aspect`, `harbour_fls_750k`, `minefield_sas`, `point_grid_flank_32`, `point_grid_fls_300k`, `point_grid_fls_750k`, `point_grid_l12_5d`, `point_grid_p4_2v`, `point_grid_popoto_4x8`, `point_grid_popoto_8`, `point_grid_popoto_8_low`, `point_grid_sas_auv`, `point_grid_sas_rail`, `pool_fls_l12_5d`, `pool_fls_p4_2v`, `pool_fls_p4_2v_walls`, `pool_wires_l12_5d`, `popoto_elevation_free`, `popoto_static_free`, `rail_cylinder_glint`, `rail_objects`, `rail_points`, `rail_position_errors`, `sas_minsas_boulder_field`, `sas_minsas_rocky_ripples`, `sas_minsas_speckle`, `sas_minsas_survey`, `seafloor_objects_fls`, `sst_echo_level`, `sst_pursuit`, `stripmap_points`, `wreck_sas` |
| fluid half-space speed 1575 m/s, density 1.7, 1 dB/wavelength | 1 | `popoto_env_deep_silt_1000m` |
| fluid half-space speed 1800 m/s, density 2, 0.6 dB/wavelength | 1 | `popoto_env_gravel_30m` |
| fluid half-space speed 1500 m/s, density 1.5, 0.2 dB/wavelength | 1 | `popoto_env_summer_mud_60m` |
| named row `medium_sand` (environment.bottom) | 3 | `sas_auv_multipath`, `sas_auv_multipath_free`, `sst_sediment_reverb` |
| fluid half-space speed 1767 m/s, density 1.845, 0.88 dB/wavelength | 1 | `sst_asiaex` |

Bottom reverberation items:

| Item | Scenarios | Names |
|---|---|---|
| Lambert mu_db -27.0 | 19 | `flank_patrol`, `flank_target_10km`, `flank_target_1km`, `flank_target_5km`, `fls_120k_cluttered_seabed`, `fls_120k_rough_surface`, `fls_120k_transit`, `fls_170k_traffic`, `fls_170k_transit`, `fls_330k_transit`, `popoto_12km`, `popoto_12km_2row`, `popoto_12km_3row`, `popoto_12km_two_ships`, `popoto_env_winter_sand_200m`, `popoto_far_mover`, `popoto_trial_1000m`, `popoto_trial_100m`, `popoto_trial_2000m` |
| fluid_sediment, six explicit parameters (rho 1.9, nu 1.1, delta 0.0147, w2 2e-05, gamma2 3.25, sigma2 0.002) | 2 | `flank_sst_shallow`, `flank_sst_spread` |
| Lambert mu_db -30.0 | 1 | `popoto_env_deep_silt_1000m` |
| Lambert mu_db -20.0 | 1 | `popoto_env_gravel_30m` |
| Lambert mu_db -35.0 | 1 | `popoto_env_summer_mud_60m` |
| fluid_sediment, sediment `medium_sand` | 1 | `sst_sediment_reverb` |

Scene seafloor items (`type: seafloor`, 14 files): all Lambert (`lambert_mu_db`) except `fls_glints_on_sediment`,
whose item is `model: fluid_sediment` with the six explicit `PLACEHOLDER_SAND` values (sigma_2 0.002). C++ and CUDA:
the scattering-strength models have no counterpart (grep of `libs/` and `python/bindings/` for `sigma2`,
`volume_strength` and `sediments.yaml` finds only an unrelated noise variance in the DOA code), so no engine
parity is involved.

### 4.2 Tests that pin a value that would move

Found by grepping `tests/` for the seven row names, `grain_size_sigma2`, `grain_size_scattering` and
`sediment_from_grain_size`; each value recomputed under each option with the tests' own helpers (`pins.py`,
`fitcheck.py`; nothing run as a test).

| Test | What it pins | Now | 0.002 | 0.003 | Median roughness | Verdict |
|---|---|---|---|---|---|---|
| `test_gallaudet_boundaries.py::test_class_default_against_the_site` | `fine_silt`, `sandy_mud`, `silty_clay` band means minus printed: SF2 in (-5.4, -4.6), SF3 in (-1.2, -0.4); `sigma2 == 0.001`; gap to the site fit -4.77 +- 0.3 dB | SF2 -4.95 / -4.94 / -4.99; SF3 -0.84 / -0.82 / -0.90; gap -4.75 / -4.77 | SF2 -1.96 / -1.95 / -2.00; SF3 +2.16 / +2.18 / +2.10; gap -1.76 | SF2 -0.21 / -0.20 / -0.25; SF3 +3.92 / +3.94 / +3.86; gap -0.01 | SF2 -4.69 / -4.67 / -4.75; SF3 -0.83 / -0.80 / -0.89; gap SF2 -4.49 | volume options: rewrite (a characterisation: new bounds and docstring). Roughness: passes, by 0.02 dB on the gap |
| `test_gallaudet_boundaries.py::test_table_vi_geoacoustic_ratios_are_our_grain_size_interpolation` | `sediment_from_grain_size(6.36)`: sigma2 0.001, gamma2 3.25 | 0.001, 3.25 | moves only if the rule (`grain_size_sigma2`) changes, not the rows | same | moves only if `GRAIN_SIZE_GAMMA2` / the w2 rule change | rows-only change: passes |
| `test_sediment_literature.py::test_regression_reproduces_the_printed_table_to_its_digits` | `grain_size_sigma2` = Snellen 2011 Table II (0.001 at 6-9 phi); step at 5.5 | as printed | rows-only: passes; if the rule function changes it fails, and must: it checks a print | same | passes unless the w2 rule changes | keep the rule as printed whatever is decided |
| `test_sediment_literature.py::test_table_scattering_parameters_are_the_published_relations_at_the_rows_grain_size` | each rule row: sigma2 == rule, w2 within 2e-5, gamma2 3.25; `very_fine_silt` sigma2 within 2 % of the rule | equal | fails for 6 rows (and `very_fine_silt` if changed) | fails | fails (w2, gamma2) | rewrite: the rows would no longer be the relations |
| `test_sediment_literature.py::test_sediment_from_grain_size_hits_the_classes_and_stays_between_them` | rows equal the interpolated rule | equal | fails for 6 rows | fails | fails | rewrite or exempt the rows |
| `test_sediment_literature.py::test_family_trend_at_high_frequency` | `sandy_silt` minus `medium_silt` at 20 deg, 100/300 kHz = 3.01 +- 1.0 dB; spread of `medium_silt`, `fine_silt`, `clay` < 0.5 dB | +3.37 / +3.37; 0.29 | +0.37 / +0.38 (fails); 0.29 | -1.39 / -1.38 (fails: muds louder than sandy silt); 0.29 | +3.36 / +3.37; 0.29 | volume options: rewrite (the docstring's "steps down at 5.5 phi" would be false) |
| `test_bottom_models.py::test_published_sediment_rows_and_their_unit_conversions` | `very_fine_silt`: volume strength -28.6 dB/m at 20 kHz (1e-9), w2 == 1.638e-5, gamma2 == 3.25; volume > 100 x roughness at 20 deg, 100 kHz | -28.6; 241 x | -25.5 (fails) | -23.8 (fails) | w2 8.47e-4, gamma2 3.78 (fails); 157 x (passes) | only if `very_fine_silt` is changed; it pins the source's printed number, so D-90's rule says keep it |
| `test_sediment_fit.py::test_volume_dominated_data_do_not_fix_the_roughness` (and the "poor" fit in `test_a_fitted_row_loads_and_warns_only_about_what_the_data_left_open`) | `very_fine_silt` as the synthetic truth: w2, gamma2 unconstrained, sigma2 found within 25 % | holds (fit/truth 0.930) | holds (0.929) | holds (0.929) | holds (0.932) | unaffected (re-evaluated in `fitcheck.py`) |

Not pinned by any test: the `clay` row is used by `test_sediment_table.py` and `test_sediment_literature.py` only
for reflection (rho, nu, delta) or for self-consistency (`build_scene` params equal `bottom.sediment(name)`), which
no option moves. The full test files were not run (nothing changed); the counts above are from reading them.

### 4.3 Documents and the gallery

The gallery (`docs/figures/gallery.json`, figures) quotes no row: "mud" and "silt" appear only as the names of
the two Popoto scenes above. Documents that quote a value that would move:

| Where | Quoted | 0.002 | 0.003 | Median roughness |
|---|---|---|---|---|
| `docs/PHYSICS_NOTES.md` Sec. 23.1 family table, `medium_silt` 10 / 20 / 45 deg (6 kHz; 100 and 300 kHz within 0.05 dB of it) | -35.66 / -31.75 / -28.31 | -32.65 / -28.74 / -25.31 | -30.89 / -26.98 / -23.54 | within 0.03 dB |
| same, `clay` | -36.38 / -32.03 / -28.41 | -33.37 / -29.02 / -25.41 | -31.61 / -27.26 / -23.65 | within 0.03 dB |
| same, text: `sandy_mud` -28.3 dB at 45 deg against Fezzani's about -22 ("2.5 to 6 dB low"); "a 3 dB step down at 5.5 phi" | -28.3 | -25.3 (step gone) | -23.6 (step reversed: muds 1.4 dB above sandy silt) | -28.3 |
| `docs/PHYSICS_NOTES.md` Sec. 23.2, `docs/REFERENCES.md` Sec. 11 row "Default sediment parameters against measurements", D-193 | `fine_silt` 5.0 / 0.8 dB low at Gallaudet's site | 2.0 low / 2.2 high | 0.2 low / 3.9 high | 4.7 / 0.8 low |
| `docs/RESULTS.md` (boundary models study) and `docs/ORG_ARL_SURVEY.md` tables, `very_fine_silt` 20 kHz 2 / 10 / 20 / 30 / 45 / 75 deg | -53.23 / -36.08 / -31.96 / -30.05 / -28.42 / -26.89 | -50.16 / -33.01 / -28.89 / -26.98 / -25.36 / -23.89 | -48.40 / -31.25 / -27.13 / -25.22 / -23.60 / -22.15 | below 75 deg within 0.02 dB; 75 deg: -24.55 |
| `configs/sediments.yaml` header and the seven rows' `provenance.sigma2` ("0.001 from 5.5 phi"); `bottom.py` comments of the rule | the rule | text to change | text to change | text to change |

`docs/DECISIONS.md` D-193 is a dated record and would stay as written.

## 5. Options for the owner

| | Option | Consequence in one line |
|---|---|---|
| A | Keep everything | Nothing moves; the rows stay the published model's defaults and stay probably 2-5 dB quiet for real muds, as `REFERENCES.md` Sec. 11 already says ("+-3 to 6 dB"). |
| B | Volume parameter of the six rule rows (and optionally `very_fine_silt`) to 0.002 | +3.0 dB on every mud row at all angles and frequencies; the 3 dB step at 5.5 phi disappears; no scenario moves; 4 to 7 tests rewritten; the rows stop being the handbook's; `very_fine_silt` would contradict its own printed source (D-90's rule says keep it). |
| C | ... to 0.003 | +4.8 dB; the mud rows become 1.4 dB louder than `sandy_silt` (the family is no longer monotonic in grain size); matches Gallaudet's one site; the same test and provenance cost as B. |
| D | Roughness rule for Mz >= 5.5 set to the data's class median | Under 0.45 dB below 60 degrees, +1.3 to +2.5 dB at 75 degrees; rests on 8 rows from 4 sites ([N2]: "not a reason to change a default"); needs a w2/gamma2 law the book does not print (the construction of 3.1 is ours). |
| E | Make the uncertainty explicit: a stated range per row (e.g. `sigma2_range: [0.0008, 0.0037]` with its basis, and the roughness's known low bias in the note), carried by the documents and by the fit tool as a bound or prior (G-9) | Nothing moves; users see the likely error, and the per-scenario override (`sigma2:` beside the name) already exists. Sediment entries are not key-checked today, so a new key is read by nothing until code uses it: text in `note:` needs no code; a bound in the fit tool is G-9's work. |
| F | Add named alternative rows and keep the present ones | Nothing moves; but there is no measured class value to put in them: a row from Table 14.1 or Gallaudet is a SITE, and the plan forbids presenting a site value as a class default. Acceptable only as site rows named after their source (e.g. `gallaudet_2001_site`, the D-193 Table VI fit already typed in the test), not as `..._measured` classes. |

**My recommendation (Claude's): E, with F only as site rows if the owner wants a ready louder mud.** Reasons:
(1) the seven rows claim to be the handbook's defaults, and for muds the handbook's value is printed three times (Snellen
2013; Snellen 2011 Table II; arXiv:1809.08459's very fine silt, 0.00099), and the one value printed for a mud
sediment itself (very fine silt) agrees with the rule, so by D-90's principle (a value printed for the sediment
outranks a general rule) the rows are faithful as rows; (2) the evidence that real muds are louder is consistent in
direction but rests on one independent fitted site and five agent-computed values from one table, whose spread
(6.7 dB) is larger than the change; the central value of everything we have is about 0.002 (+3 dB), within the
error the documents already state; (3) no shipped scenario uses these rows, so a change buys no realism today and
costs four to seven rewritten tests and provenance that no longer matches its source; (4) the roughness option moves
less than the book's own measurement accuracy (+2/-3 dB) except at 75 degrees. If the owner prefers a change,
B (0.002) is the better-supported number than C: it is the median of the evidence and the book's own example,
while 0.003 rests on one site.

## 6. What is not known

- **No measured volume parameter is printed for any sediment class** in anything read ([N2] Sec. 3.3, [N4] Sec. 4);
  TR 9407, which holds the published model's own basis, was not read.
- The sites are few: five rows of one table (agents' computations through a model with an unmeasured correlation,
  rho_rk = -1, and assumed mu for two sites) and one fitted site (Gallaudet, one wind-and-wake-dominated survey,
  band means with up to -9 dB of transmit-pattern bias, D-193). The water speed behind Table 14.1's values (1500 m/s
  here) is our assumption.
- The finite-L_c rows need Ch. 7's spectral form, read by [N2] only ([N4] Sec. 6); the frequency dependence of
  the implied values outside the measured band is an extrapolation.
- The roughness comparison rests on [N2]'s typing of Table 6.1 and its corrected (D.19); whether Briggs used the
  misprinted form for Fig. 6.18 is unresolved ([N2] Sec. 6). The class-median spectrum of 3.1 is a construction
  from three medians, not a fit to the 8 rows.
- Fezzani's "about -22 dB" is read through `docs/PHYSICS_NOTES.md`; the paper's own numbers were not re-read for
  this document.
- Every value marked "ours" or recomputed here was derived by an agent, not printed in a source.

## 7. If the owner decides

- E: proposed text for each of the seven rows' `note` (or a new key): "Volume parameter: 0.001 is the published
  model's class value (Snellen et al. 2011 Table II; Snellen et al. 2013; arXiv:1809.08459 for very fine silt).
  Measured sites of this grain size imply 0.0008 to 0.0037 (J&R 2007 Table 14.1 through its eq. (14.10), our
  computation) and one fit 0.003 (Gallaudet 2001): the row may be up to 5.7 dB quiet on the volume term (median of
  the evidence about 0.002, +3 dB). Roughness: the rule lies 1 to 7 dB below J&R Table 6.1's median for Mz >= 5.5
  at 10-100 kHz; the total moves by under 0.5 dB below 60 degrees. Give `sigma2:` beside the name, or fit the site
  (tools/fit_sediment.py)."
- B or C: the files are `configs/sediments.yaml` (six or seven `sigma2` values, their `provenance.sigma2` and the
  header), the tests of 4.2, and the documents of 4.3; decide whether `grain_size_sigma2` (the printed rule) stays
  as printed (recommended: yes, so that `sediment_from_grain_size` keeps reproducing Gallaudet's Table VI inputs and
  Snellen's table).

## 8. Scripts (scratch, not committed)

All in `/tmp/claude-0/-home-user-claude-playground/32e65dcd-82ae-5f75-a363-7372d2e7f847/scratchpad/wave/mud/`, run
from the worktree root with `PYTHONPATH=python` (plus `:tests/python` or `:tools` where said):

| Script | Computes |
|---|---|
| `tables.py` | tables T2-T5 and Appendix A (`tables.json`), the fitted class-median roughness, and the checks of 3.6 |
| `gen_md.py` | the markdown of T1-T5 and Appendix A from `tables.json` and the sediment table |
| `sites.py` | the five Table 14.1 volume parameters, independent of the repository |
| `rl_check.py` | the end-to-end reverberation level check of 3.5 (`PYTHONPATH=python:tools`) |
| `pins.py`, `fitcheck.py` | the pinned and quoted values of 4.2 and 4.3 (`PYTHONPATH=python:tests/python`) |
| `exposure.py`, `group.py` | the scenario survey of 4.1 |
| `lambert_eq.py` | the equivalent Lambert parameters of 2.1(c) |

## Appendix A: every affected row, backscattering strength in dB (default small_slope, 1500 m/s)

Columns are grazing angles in degrees.

### `medium_silt` (rho 1.1490, nu 0.9885, delta 0.00676, w2 1.636e-05, gamma2 3.25, sigma2 0.001; class-median roughness: w2 0.0008464, gamma2 3.784)

| f (kHz) | case | 5 | 10 | 20 | 30 | 45 | 60 | 75 |
|---|---|---|---|---|---|---|---|---|
| 8 | now (sigma2 0.001, rule w2) | -41.3 | -35.7 | -31.7 | -29.9 | -28.3 | -27.4 | -26.9 |
| 8 | sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.4 | -23.9 |
| 8 | sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.1 | -23.5 | -22.6 | -22.1 |
| 8 | class-median roughness | -41.3 | -35.6 | -31.7 | -29.9 | -28.3 | -27.3 | -24.5 |
| 8 | median roughness + sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.3 | -22.5 |
| 8 | median roughness + sigma2 0.003 | -36.5 | -30.9 | -27.0 | -25.1 | -23.5 | -22.6 | -21.2 |
| 30 | now (sigma2 0.001, rule w2) | -41.3 | -35.6 | -31.7 | -29.9 | -28.3 | -27.4 | -26.7 |
| 30 | sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.4 | -23.8 |
| 30 | sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.1 | -23.5 | -22.6 | -22.1 |
| 30 | class-median roughness | -41.3 | -35.6 | -31.7 | -29.9 | -28.3 | -27.2 | -24.2 |
| 30 | median roughness + sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.3 | -22.4 |
| 30 | median roughness + sigma2 0.003 | -36.5 | -30.9 | -27.0 | -25.1 | -23.5 | -22.6 | -21.1 |
| 68 | now (sigma2 0.001, rule w2) | -41.3 | -35.6 | -31.7 | -29.9 | -28.3 | -27.4 | -26.6 |
| 68 | sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.4 | -23.7 |
| 68 | sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.1 | -23.5 | -22.6 | -22.0 |
| 68 | class-median roughness | -41.3 | -35.6 | -31.7 | -29.9 | -28.3 | -27.1 | -24.2 |
| 68 | median roughness + sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.2 | -22.4 |
| 68 | median roughness + sigma2 0.003 | -36.5 | -30.9 | -27.0 | -25.1 | -23.5 | -22.5 | -21.1 |
| 120 | now (sigma2 0.001, rule w2) | -41.3 | -35.6 | -31.7 | -29.9 | -28.3 | -27.3 | -26.4 |
| 120 | sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.4 | -23.6 |
| 120 | sigma2 0.003 | -36.5 | -30.9 | -27.0 | -25.1 | -23.5 | -22.6 | -22.0 |
| 120 | class-median roughness | -41.3 | -35.6 | -31.7 | -29.9 | -28.3 | -27.0 | -24.2 |
| 120 | median roughness + sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.2 | -22.4 |
| 120 | median roughness + sigma2 0.003 | -36.5 | -30.9 | -27.0 | -25.1 | -23.5 | -22.5 | -21.1 |
| 330 | now (sigma2 0.001, rule w2) | -41.2 | -35.6 | -31.7 | -29.9 | -28.3 | -27.2 | -25.8 |
| 330 | sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.3 | -23.3 |
| 330 | sigma2 0.003 | -36.5 | -30.9 | -27.0 | -25.1 | -23.5 | -22.6 | -21.7 |
| 330 | class-median roughness | -41.3 | -35.6 | -31.7 | -29.9 | -28.2 | -26.8 | -24.4 |
| 330 | median roughness + sigma2 0.002 | -38.3 | -32.6 | -28.7 | -26.9 | -25.3 | -24.1 | -22.4 |
| 330 | median roughness + sigma2 0.003 | -36.5 | -30.9 | -27.0 | -25.1 | -23.5 | -22.4 | -21.1 |

### `sandy_mud` (rho 1.1490, nu 0.9873, delta 0.00386, w2 1.636e-05, gamma2 3.25, sigma2 0.001; class-median roughness: w2 0.0008464, gamma2 3.784)

| f (kHz) | case | 5 | 10 | 20 | 30 | 45 | 60 | 75 |
|---|---|---|---|---|---|---|---|---|
| 8 | now (sigma2 0.001, rule w2) | -41.4 | -35.7 | -31.8 | -29.9 | -28.3 | -27.4 | -26.9 |
| 8 | sigma2 0.002 | -38.4 | -32.7 | -28.8 | -26.9 | -25.3 | -24.4 | -23.9 |
| 8 | sigma2 0.003 | -36.6 | -31.0 | -27.0 | -25.2 | -23.6 | -22.6 | -22.1 |
| 8 | class-median roughness | -41.4 | -35.7 | -31.8 | -29.9 | -28.3 | -27.3 | -24.5 |
| 8 | median roughness + sigma2 0.002 | -38.4 | -32.7 | -28.8 | -26.9 | -25.3 | -24.3 | -22.5 |
| 8 | median roughness + sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.2 | -23.6 | -22.6 | -21.2 |
| 30 | now (sigma2 0.001, rule w2) | -41.4 | -35.7 | -31.8 | -29.9 | -28.3 | -27.4 | -26.7 |
| 30 | sigma2 0.002 | -38.4 | -32.7 | -28.8 | -26.9 | -25.3 | -24.4 | -23.8 |
| 30 | sigma2 0.003 | -36.6 | -31.0 | -27.0 | -25.2 | -23.6 | -22.6 | -22.1 |
| 30 | class-median roughness | -41.3 | -35.7 | -31.8 | -29.9 | -28.3 | -27.2 | -24.3 |
| 30 | median roughness + sigma2 0.002 | -38.4 | -32.7 | -28.8 | -26.9 | -25.3 | -24.3 | -22.4 |
| 30 | median roughness + sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.2 | -23.5 | -22.6 | -21.1 |
| 68 | now (sigma2 0.001, rule w2) | -41.4 | -35.7 | -31.8 | -29.9 | -28.3 | -27.4 | -26.6 |
| 68 | sigma2 0.002 | -38.4 | -32.7 | -28.8 | -26.9 | -25.3 | -24.4 | -23.7 |
| 68 | sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.2 | -23.6 | -22.6 | -22.0 |
| 68 | class-median roughness | -41.3 | -35.7 | -31.8 | -29.9 | -28.3 | -27.1 | -24.3 |
| 68 | median roughness + sigma2 0.002 | -38.3 | -32.7 | -28.8 | -26.9 | -25.3 | -24.2 | -22.4 |
| 68 | median roughness + sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.1 | -23.5 | -22.5 | -21.1 |
| 120 | now (sigma2 0.001, rule w2) | -41.3 | -35.7 | -31.8 | -29.9 | -28.3 | -27.3 | -26.4 |
| 120 | sigma2 0.002 | -38.4 | -32.7 | -28.8 | -26.9 | -25.3 | -24.4 | -23.7 |
| 120 | sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.2 | -23.6 | -22.6 | -22.0 |
| 120 | class-median roughness | -41.3 | -35.7 | -31.8 | -29.9 | -28.3 | -27.0 | -24.3 |
| 120 | median roughness + sigma2 0.002 | -38.3 | -32.7 | -28.8 | -26.9 | -25.3 | -24.2 | -22.4 |
| 120 | median roughness + sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.1 | -23.5 | -22.5 | -21.1 |
| 330 | now (sigma2 0.001, rule w2) | -41.3 | -35.7 | -31.7 | -29.9 | -28.3 | -27.3 | -25.8 |
| 330 | sigma2 0.002 | -38.3 | -32.7 | -28.8 | -26.9 | -25.3 | -24.3 | -23.3 |
| 330 | sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.1 | -23.5 | -22.6 | -21.8 |
| 330 | class-median roughness | -41.3 | -35.7 | -31.7 | -29.9 | -28.2 | -26.8 | -24.4 |
| 330 | median roughness + sigma2 0.002 | -38.3 | -32.7 | -28.8 | -26.9 | -25.3 | -24.1 | -22.5 |
| 330 | median roughness + sigma2 0.003 | -36.6 | -30.9 | -27.0 | -25.1 | -23.5 | -22.4 | -21.1 |

### `fine_silt` (rho 1.1480, nu 0.9861, delta 0.00306, w2 1.636e-05, gamma2 3.25, sigma2 0.001; class-median roughness: w2 0.0008464, gamma2 3.784)

| f (kHz) | case | 5 | 10 | 20 | 30 | 45 | 60 | 75 |
|---|---|---|---|---|---|---|---|---|
| 8 | now (sigma2 0.001, rule w2) | -41.6 | -35.8 | -31.8 | -30.0 | -28.3 | -27.4 | -26.9 |
| 8 | sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.4 | -23.9 |
| 8 | sigma2 0.003 | -36.8 | -31.1 | -27.1 | -25.2 | -23.6 | -22.7 | -22.1 |
| 8 | class-median roughness | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.3 | -24.6 |
| 8 | median roughness + sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.4 | -22.6 |
| 8 | median roughness + sigma2 0.003 | -36.8 | -31.1 | -27.1 | -25.2 | -23.6 | -22.6 | -21.2 |
| 30 | now (sigma2 0.001, rule w2) | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.4 | -26.8 |
| 30 | sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.4 | -23.8 |
| 30 | sigma2 0.003 | -36.8 | -31.1 | -27.1 | -25.2 | -23.6 | -22.7 | -22.1 |
| 30 | class-median roughness | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.2 | -24.4 |
| 30 | median roughness + sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.3 | -22.5 |
| 30 | median roughness + sigma2 0.003 | -36.8 | -31.1 | -27.0 | -25.2 | -23.6 | -22.6 | -21.1 |
| 68 | now (sigma2 0.001, rule w2) | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.4 | -26.6 |
| 68 | sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.4 | -23.8 |
| 68 | sigma2 0.003 | -36.8 | -31.1 | -27.1 | -25.2 | -23.6 | -22.6 | -22.1 |
| 68 | class-median roughness | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.1 | -24.3 |
| 68 | median roughness + sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.3 | -22.4 |
| 68 | median roughness + sigma2 0.003 | -36.8 | -31.1 | -27.0 | -25.2 | -23.6 | -22.5 | -21.1 |
| 120 | now (sigma2 0.001, rule w2) | -41.6 | -35.8 | -31.8 | -29.9 | -28.3 | -27.4 | -26.4 |
| 120 | sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.4 | -23.7 |
| 120 | sigma2 0.003 | -36.8 | -31.1 | -27.0 | -25.2 | -23.6 | -22.6 | -22.0 |
| 120 | class-median roughness | -41.5 | -35.8 | -31.8 | -29.9 | -28.3 | -27.0 | -24.4 |
| 120 | median roughness + sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.2 | -22.5 |
| 120 | median roughness + sigma2 0.003 | -36.8 | -31.0 | -27.0 | -25.2 | -23.6 | -22.5 | -21.1 |
| 330 | now (sigma2 0.001, rule w2) | -41.5 | -35.8 | -31.8 | -29.9 | -28.3 | -27.3 | -25.9 |
| 330 | sigma2 0.002 | -38.5 | -32.8 | -28.8 | -26.9 | -25.3 | -24.3 | -23.4 |
| 330 | sigma2 0.003 | -36.8 | -31.0 | -27.0 | -25.2 | -23.6 | -22.6 | -21.8 |
| 330 | class-median roughness | -41.5 | -35.8 | -31.8 | -29.9 | -28.3 | -26.8 | -24.5 |
| 330 | median roughness + sigma2 0.002 | -38.6 | -32.8 | -28.8 | -26.9 | -25.3 | -24.1 | -22.5 |
| 330 | median roughness + sigma2 0.003 | -36.8 | -31.0 | -27.0 | -25.2 | -23.5 | -22.5 | -21.2 |

### `sandy_clay` (rho 1.1470, nu 0.9849, delta 0.00242, w2 1.636e-05, gamma2 3.25, sigma2 0.001; class-median roughness: w2 0.0008464, gamma2 3.784)

| f (kHz) | case | 5 | 10 | 20 | 30 | 45 | 60 | 75 |
|---|---|---|---|---|---|---|---|---|
| 8 | now (sigma2 0.001, rule w2) | -41.8 | -35.9 | -31.9 | -30.0 | -28.4 | -27.4 | -26.9 |
| 8 | sigma2 0.002 | -38.8 | -32.9 | -28.9 | -27.0 | -25.3 | -24.4 | -23.9 |
| 8 | sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.7 | -22.2 |
| 8 | class-median roughness | -41.8 | -35.9 | -31.9 | -30.0 | -28.3 | -27.3 | -24.6 |
| 8 | median roughness + sigma2 0.002 | -38.8 | -32.9 | -28.9 | -27.0 | -25.3 | -24.4 | -22.6 |
| 8 | median roughness + sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -21.3 |
| 30 | now (sigma2 0.001, rule w2) | -41.8 | -35.9 | -31.9 | -30.0 | -28.3 | -27.4 | -26.8 |
| 30 | sigma2 0.002 | -38.8 | -32.9 | -28.9 | -27.0 | -25.3 | -24.4 | -23.9 |
| 30 | sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.7 | -22.1 |
| 30 | class-median roughness | -41.8 | -35.9 | -31.8 | -30.0 | -28.3 | -27.2 | -24.4 |
| 30 | median roughness + sigma2 0.002 | -38.8 | -32.9 | -28.8 | -27.0 | -25.3 | -24.3 | -22.5 |
| 30 | median roughness + sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -21.2 |
| 68 | now (sigma2 0.001, rule w2) | -41.8 | -35.9 | -31.9 | -30.0 | -28.3 | -27.4 | -26.6 |
| 68 | sigma2 0.002 | -38.8 | -32.9 | -28.9 | -27.0 | -25.3 | -24.4 | -23.8 |
| 68 | sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.7 | -22.1 |
| 68 | class-median roughness | -41.8 | -35.9 | -31.8 | -29.9 | -28.3 | -27.1 | -24.4 |
| 68 | median roughness + sigma2 0.002 | -38.8 | -32.9 | -28.8 | -27.0 | -25.3 | -24.3 | -22.5 |
| 68 | median roughness + sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -21.2 |
| 120 | now (sigma2 0.001, rule w2) | -41.8 | -35.9 | -31.8 | -30.0 | -28.3 | -27.4 | -26.5 |
| 120 | sigma2 0.002 | -38.8 | -32.9 | -28.8 | -27.0 | -25.3 | -24.4 | -23.7 |
| 120 | sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -22.0 |
| 120 | class-median roughness | -41.8 | -35.9 | -31.8 | -29.9 | -28.3 | -27.0 | -24.4 |
| 120 | median roughness + sigma2 0.002 | -38.8 | -32.9 | -28.8 | -27.0 | -25.3 | -24.2 | -22.5 |
| 120 | median roughness + sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.5 | -21.2 |
| 330 | now (sigma2 0.001, rule w2) | -41.7 | -35.9 | -31.8 | -29.9 | -28.3 | -27.3 | -25.9 |
| 330 | sigma2 0.002 | -38.8 | -32.9 | -28.8 | -26.9 | -25.3 | -24.4 | -23.4 |
| 330 | sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -21.8 |
| 330 | class-median roughness | -41.8 | -35.9 | -31.8 | -29.9 | -28.3 | -26.9 | -24.5 |
| 330 | median roughness + sigma2 0.002 | -38.8 | -32.9 | -28.8 | -26.9 | -25.3 | -24.1 | -22.6 |
| 330 | median roughness + sigma2 0.003 | -37.0 | -31.2 | -27.1 | -25.2 | -23.6 | -22.5 | -21.2 |

### `very_fine_silt` (rho 1.1470, nu 0.9840, delta 0.00189, w2 1.638e-05, gamma2 3.25, sigma2 0.00098599; class-median roughness: w2 0.0008472, gamma2 3.784)

| f (kHz) | case | 5 | 10 | 20 | 30 | 45 | 60 | 75 |
|---|---|---|---|---|---|---|---|---|
| 8 | now (sigma2 0.001, rule w2) | -42.0 | -36.1 | -32.0 | -30.1 | -28.4 | -27.5 | -27.0 |
| 8 | sigma2 0.002 | -39.0 | -33.0 | -28.9 | -27.0 | -25.4 | -24.4 | -23.9 |
| 8 | sigma2 0.003 | -37.2 | -31.3 | -27.1 | -25.2 | -23.6 | -22.7 | -22.2 |
| 8 | class-median roughness | -42.0 | -36.1 | -31.9 | -30.0 | -28.4 | -27.4 | -24.7 |
| 8 | median roughness + sigma2 0.002 | -39.0 | -33.0 | -28.9 | -27.0 | -25.3 | -24.4 | -22.7 |
| 8 | median roughness + sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -21.3 |
| 30 | now (sigma2 0.001, rule w2) | -42.0 | -36.1 | -32.0 | -30.0 | -28.4 | -27.5 | -26.8 |
| 30 | sigma2 0.002 | -39.0 | -33.0 | -28.9 | -27.0 | -25.4 | -24.4 | -23.9 |
| 30 | sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.7 | -22.1 |
| 30 | class-median roughness | -42.0 | -36.1 | -31.9 | -30.0 | -28.4 | -27.3 | -24.5 |
| 30 | median roughness + sigma2 0.002 | -39.0 | -33.0 | -28.9 | -27.0 | -25.3 | -24.3 | -22.5 |
| 30 | median roughness + sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -21.2 |
| 68 | now (sigma2 0.001, rule w2) | -42.0 | -36.1 | -31.9 | -30.0 | -28.4 | -27.5 | -26.7 |
| 68 | sigma2 0.002 | -39.0 | -33.0 | -28.9 | -27.0 | -25.4 | -24.4 | -23.8 |
| 68 | sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.7 | -22.1 |
| 68 | class-median roughness | -42.0 | -36.1 | -31.9 | -30.0 | -28.4 | -27.2 | -24.5 |
| 68 | median roughness + sigma2 0.002 | -39.0 | -33.0 | -28.9 | -27.0 | -25.3 | -24.3 | -22.5 |
| 68 | median roughness + sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -21.2 |
| 120 | now (sigma2 0.001, rule w2) | -42.0 | -36.1 | -31.9 | -30.0 | -28.4 | -27.4 | -26.5 |
| 120 | sigma2 0.002 | -39.0 | -33.0 | -28.9 | -27.0 | -25.3 | -24.4 | -23.7 |
| 120 | sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.7 | -22.0 |
| 120 | class-median roughness | -42.0 | -36.0 | -31.9 | -30.0 | -28.4 | -27.1 | -24.5 |
| 120 | median roughness + sigma2 0.002 | -39.0 | -33.0 | -28.9 | -27.0 | -25.3 | -24.2 | -22.5 |
| 120 | median roughness + sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.5 | -21.2 |
| 330 | now (sigma2 0.001, rule w2) | -42.0 | -36.0 | -31.9 | -30.0 | -28.4 | -27.4 | -26.0 |
| 330 | sigma2 0.002 | -38.9 | -33.0 | -28.9 | -27.0 | -25.3 | -24.4 | -23.4 |
| 330 | sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.6 | -21.8 |
| 330 | class-median roughness | -42.0 | -36.0 | -31.9 | -30.0 | -28.3 | -26.9 | -24.6 |
| 330 | median roughness + sigma2 0.002 | -38.9 | -33.0 | -28.9 | -27.0 | -25.3 | -24.2 | -22.6 |
| 330 | median roughness + sigma2 0.003 | -37.2 | -31.2 | -27.1 | -25.2 | -23.6 | -22.5 | -21.2 |

### `silty_clay` (rho 1.1460, nu 0.9824, delta 0.00163, w2 1.636e-05, gamma2 3.25, sigma2 0.001; class-median roughness: w2 0.0008464, gamma2 3.784)

| f (kHz) | case | 5 | 10 | 20 | 30 | 45 | 60 | 75 |
|---|---|---|---|---|---|---|---|---|
| 8 | now (sigma2 0.001, rule w2) | -42.3 | -36.2 | -32.0 | -30.0 | -28.4 | -27.5 | -26.9 |
| 8 | sigma2 0.002 | -39.3 | -33.2 | -28.9 | -27.0 | -25.4 | -24.4 | -23.9 |
| 8 | sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.3 | -23.6 | -22.7 | -22.2 |
| 8 | class-median roughness | -42.3 | -36.2 | -31.9 | -30.0 | -28.4 | -27.3 | -24.8 |
| 8 | median roughness + sigma2 0.002 | -39.3 | -33.2 | -28.9 | -27.0 | -25.4 | -24.4 | -22.7 |
| 8 | median roughness + sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.2 | -23.6 | -22.7 | -21.3 |
| 30 | now (sigma2 0.001, rule w2) | -42.3 | -36.2 | -31.9 | -30.0 | -28.4 | -27.4 | -26.8 |
| 30 | sigma2 0.002 | -39.3 | -33.2 | -28.9 | -27.0 | -25.4 | -24.4 | -23.9 |
| 30 | sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.3 | -23.6 | -22.7 | -22.1 |
| 30 | class-median roughness | -42.3 | -36.1 | -31.9 | -30.0 | -28.4 | -27.3 | -24.6 |
| 30 | median roughness + sigma2 0.002 | -39.3 | -33.1 | -28.9 | -27.0 | -25.4 | -24.3 | -22.6 |
| 30 | median roughness + sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.2 | -23.6 | -22.6 | -21.2 |
| 68 | now (sigma2 0.001, rule w2) | -42.3 | -36.2 | -31.9 | -30.0 | -28.4 | -27.4 | -26.7 |
| 68 | sigma2 0.002 | -39.3 | -33.2 | -28.9 | -27.0 | -25.4 | -24.4 | -23.8 |
| 68 | sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.2 | -23.6 | -22.7 | -22.1 |
| 68 | class-median roughness | -42.2 | -36.1 | -31.9 | -30.0 | -28.3 | -27.2 | -24.5 |
| 68 | median roughness + sigma2 0.002 | -39.3 | -33.1 | -28.9 | -27.0 | -25.4 | -24.3 | -22.6 |
| 68 | median roughness + sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.2 | -23.6 | -22.6 | -21.2 |
| 120 | now (sigma2 0.001, rule w2) | -42.2 | -36.1 | -31.9 | -30.0 | -28.4 | -27.4 | -26.5 |
| 120 | sigma2 0.002 | -39.3 | -33.1 | -28.9 | -27.0 | -25.4 | -24.4 | -23.7 |
| 120 | sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.2 | -23.6 | -22.7 | -22.0 |
| 120 | class-median roughness | -42.2 | -36.1 | -31.9 | -30.0 | -28.3 | -27.1 | -24.6 |
| 120 | median roughness + sigma2 0.002 | -39.2 | -33.1 | -28.9 | -27.0 | -25.4 | -24.3 | -22.6 |
| 120 | median roughness + sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.2 | -23.6 | -22.6 | -21.2 |
| 330 | now (sigma2 0.001, rule w2) | -42.2 | -36.1 | -31.9 | -30.0 | -28.3 | -27.3 | -26.0 |
| 330 | sigma2 0.002 | -39.2 | -33.1 | -28.9 | -27.0 | -25.4 | -24.4 | -23.4 |
| 330 | sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.2 | -23.6 | -22.6 | -21.8 |
| 330 | class-median roughness | -42.2 | -36.1 | -31.9 | -30.0 | -28.3 | -26.9 | -24.7 |
| 330 | median roughness + sigma2 0.002 | -39.2 | -33.1 | -28.9 | -27.0 | -25.3 | -24.2 | -22.6 |
| 330 | median roughness + sigma2 0.003 | -37.5 | -31.4 | -27.2 | -25.2 | -23.6 | -22.5 | -21.3 |

### `clay` (rho 1.1450, nu 0.9800, delta 0.00148, w2 1.636e-05, gamma2 3.25, sigma2 0.001; class-median roughness: w2 0.0008464, gamma2 3.784)

| f (kHz) | case | 5 | 10 | 20 | 30 | 45 | 60 | 75 |
|---|---|---|---|---|---|---|---|---|
| 8 | now (sigma2 0.001, rule w2) | -42.7 | -36.4 | -32.0 | -30.1 | -28.4 | -27.5 | -26.9 |
| 8 | sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.1 | -25.4 | -24.5 | -24.0 |
| 8 | sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.7 | -22.2 |
| 8 | class-median roughness | -42.7 | -36.4 | -32.0 | -30.1 | -28.4 | -27.4 | -24.9 |
| 8 | median roughness + sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.1 | -25.4 | -24.4 | -22.8 |
| 8 | median roughness + sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.7 | -21.4 |
| 30 | now (sigma2 0.001, rule w2) | -42.7 | -36.4 | -32.0 | -30.1 | -28.4 | -27.5 | -26.8 |
| 30 | sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.1 | -25.4 | -24.5 | -23.9 |
| 30 | sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.7 | -22.2 |
| 30 | class-median roughness | -42.7 | -36.4 | -32.0 | -30.1 | -28.4 | -27.3 | -24.7 |
| 30 | median roughness + sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.0 | -25.4 | -24.4 | -22.7 |
| 30 | median roughness + sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.6 | -21.3 |
| 68 | now (sigma2 0.001, rule w2) | -42.7 | -36.4 | -32.0 | -30.1 | -28.4 | -27.4 | -26.7 |
| 68 | sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.1 | -25.4 | -24.5 | -23.8 |
| 68 | sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.7 | -22.1 |
| 68 | class-median roughness | -42.7 | -36.4 | -32.0 | -30.0 | -28.4 | -27.2 | -24.6 |
| 68 | median roughness + sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.0 | -25.4 | -24.3 | -22.7 |
| 68 | median roughness + sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.6 | -21.3 |
| 120 | now (sigma2 0.001, rule w2) | -42.7 | -36.4 | -32.0 | -30.1 | -28.4 | -27.4 | -26.5 |
| 120 | sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.1 | -25.4 | -24.4 | -23.8 |
| 120 | sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.7 | -22.1 |
| 120 | class-median roughness | -42.7 | -36.4 | -32.0 | -30.0 | -28.4 | -27.1 | -24.7 |
| 120 | median roughness + sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.0 | -25.4 | -24.3 | -22.7 |
| 120 | median roughness + sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.6 | -21.3 |
| 330 | now (sigma2 0.001, rule w2) | -42.6 | -36.3 | -32.0 | -30.0 | -28.4 | -27.4 | -26.0 |
| 330 | sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.0 | -25.4 | -24.4 | -23.5 |
| 330 | sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.7 | -21.9 |
| 330 | class-median roughness | -42.7 | -36.3 | -32.0 | -30.0 | -28.3 | -27.0 | -24.8 |
| 330 | median roughness + sigma2 0.002 | -39.7 | -33.4 | -29.0 | -27.0 | -25.4 | -24.2 | -22.7 |
| 330 | median roughness + sigma2 0.003 | -37.9 | -31.6 | -27.3 | -25.3 | -23.6 | -22.5 | -21.3 |
