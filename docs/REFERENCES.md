# References: what the suite's physics rests on

A register of every source behind a formula, table, constant or validation number in the
simulator, with how we got to it and how far it is confirmed; and, since 2026-11-20
(section 12), of the sources behind the processing chain's algorithms (detection,
beamforming, bearings, tracking, motion estimation) and the open-source oracles the
simulator is tested against. Written 2026-11-18 after the
owner allowed a wider literature search ("Propose to unlock literature and let's find more
references"); earlier rows come from `docs/PHYSICS_NOTES.md`, `docs/OPEN_SOURCE.md`,
`docs/ORG_ARL_SURVEY.md` and the code's docstrings. Software licences are in
`docs/OPEN_SOURCE.md`; this file is about physics and algorithms.

Rules that produced it: a value or formula is used only if it was read in a document opened in
a session of this project, and it is cited with its table, equation or page. Nothing is
written from memory as if it were sourced. APL-UW TR 9407 (limited distribution) was not
downloaded or read; where one of its values is used, the open document that prints it and
attributes it is the source (owner's ruling). Documents opened in the 2026-11 search are saved
outside the repository (section 9); nothing from them is committed except the numbers and
formulas cited here.

**Access**: `open` (read on the publisher's or author's open copy; licence where stated) ·
`open, PD` (US Government work, public release) · `via <source>` (not opened itself; the
value is the one an opened source prints and attributes to it) · `no` (could not be opened by
us: paywall, bot challenge, host not reachable).

**Status**: `confirmed` (read in the source itself) · `via secondary` (read in an opened
document that attributes it) · `ours` (our derivation, tested as stated) · `unconfirmed`
(in the code, not checked against any opened document).

---

## 1. Summary of what the 2026-11 search changed

| Topic | Before | Now |
|---|---|---|
| Sediment table | 2 complete rows, 20 reflection-only | 20 complete sediment rows (sandy gravel to clay) with per-parameter provenance; rock and pebbles carry an empirical Lambert parameter (section 3.1) |
| Sandy gravel sound speed | two values, unconfirmed | 1.337 (three independent prints against one); corrected |
| Volume scattering term | ours | equal to the published expression to 1e-10 |
| Composite-roughness split `small_scale_kh = 0.5` | our choice | the same number as a published criterion (Rayleigh parameter of the small scale = 1) |
| Small-slope and perturbation validity | rule of thumb | published limits for power-law roughness (section 3.2) |
| Surface backscatter above 10 kHz | empirical laws only | bubble-layer plus facet model from the published high-frequency model, opt-in; the empirical laws exceed its saturation level at high wind and frequency |
| Bubble damping constants | - | two printed constants agree with the module's independent bubble dynamics within 2 % |
| Pierson-Moskowitz constants, Cox-Munk slopes | from memory | read; ours agree within 0.4 % and 0.8 % |
| Francois-Garrison absorption | arlpy oracle only | within 0.6 % of 21 published values |
| Ambient noise fit | two secondary prints | a third, with attributions |
| Surface-noise coherence | isotropic only | Cron-Sherman added (function only) |
| Doppler width against grazing angle | our derivation against TR 0702's printed 1/sin | measured tables in an open thesis grow with steepness, as ours does |
| Not settled | | bubble loss per bounce (no measurement could be opened); coherent fraction, delay spread and scintillation of forward scatter (Dahl's papers not accessible); K-distribution shape; Urick's table; Hamilton's regressions; Chapman-Harris's range |

---

## 2. Framework

| Source | Access | Taken, and where it lives | Status |
|---|---|---|---|
| Goddard, "The Sonar Simulation Toolset, Release 4.6: Science, Mathematics, and Algorithms", APL-UW TR 0702 (2008) | open (extracted text used throughout) | structure of the simulator, eqs. (38), (42) Thorp, (89)-(103) spread and coherence relations, time-spread rule of thumb, validation numbers: `docs/PHYSICS_NOTES.md` Secs. 1-11, 18; `sim/timespread.py`, `acoustics.absorption_thorp` | confirmed |
| APL-UW High-Frequency Ocean Environmental Acoustic Models Handbook, APL-UW TR 9407 (1994) | **not read** (limited distribution) | nothing directly. Values attributed to it by open documents: sediment tables (3.1), sea states, bubble loss law, surface model terms (4) | via secondary, each row below |

## 3. Bottom

### 3.1 Sediment parameters (`configs/sediments.yaml`, `sim/bottom.py`)

| Source | Access | Taken, and where it lives | Status |
|---|---|---|---|
| UnderwaterAcoustics.jl (org-arl), `src/stdlib.jl:113-136`, commit 7b9c47e | open, MIT | density, sound speed, loss parameter of 22 seabeds for water 1023 kg/m3, 1528 m/s ("from APL-UW TR 9407 (1994), IV-6 Table 2"): the `reflection:` blocks | via secondary |
| Cintron, "Environmental impact on mine hunting in the Yellow Sea using the CASS/GRAB model", MS thesis, Naval Postgraduate School (2001), https://hdl.handle.net/10945/10842 | open, PD | Table 1, "APL/UW TR9407 Geo-acoustic parameters associated with bulk grain size index used by the CASS/GRAB model (From NAVOCEANO 1999)": 23 rows, name, grain size index, density ratio, speed ratio, loss. Gives every sediment name its mean grain size (`mean_grain_size_phi`), is a second print of all ratios (`bottom.GRAIN_SIZE_TABLE`), and has 1.337 for sandy gravel | via secondary |
| Snellen, Siemes and Simons, "Model-based sediment classification using single-beam echosounder signals", J. Acoust. Soc. Am. 129, 2878-2888 (2011); author copy https://repository.tudelft.nl/record/uuid:b635a0a8-2ca1-44ae-8fbc-0866b3304827 | open (repository copy; journal copyright) | Table II: rho, nu, delta, sigma2, w2 (cm^4) at Mz = -1 ... 9 (`bottom.GRAIN_SIZE_W2_TABLE_CM4`); Sec. II.A: spectral exponent 3.25 "found to work well for many types of sediments"; Table III: names against Mz for seven classes | via secondary |
| Snellen, Eleftherakis, Amiri-Simkooei, Koomans and Simons, "An inter-comparison of sediment classification methods ...", J. Acoust. Soc. Am. 134, 959-970 (2013); author copy https://repository.tudelft.nl/record/uuid:118445fc-6696-4fe2-abae-53272b80db43 | open (repository copy) | eq. (5): w2 = 0.00207 h^2 cm^4, h = (2.03846 - 0.26923 Mz) / (1 + 0.076923 Mz) for -1 <= Mz < 5, h = 0.5 for 5 <= Mz <= 9; sigma2 = 0.002 below 5.5 phi, 0.001 from 5.5 phi (`bottom.grain_size_w2_cm4`, `grain_size_sigma2`); the spectrum convention W = (h0 K)^-gamma w2, h0 = 1 cm; eq. (3), the volume backscattering expression (test: equal to ours); the finding that fitted w2 at 300 kHz is about twice and sigma2 about half the defaults | via secondary (relations); confirmed (their findings) |
| Brown, Johnson and Brownstead, arXiv:1809.08459, Table 1 | open (arXiv) | `medium_sand`, `very_fine_silt` in physical units, spectral strength in m^(4 - gamma): the check of the cm^4 conversion (0.3 %) | via secondary |
| Lurton, Lamarche et al., "Backscatter measurements by seafloor-mapping sonars: Guidelines and Recommendations", GeoHab BSWG (2015), doi:10.5281/zenodo.10089261 | open, CC BY 4.0 | Table 2-1: grain size ranges and a third print of density, velocity, absorption for nine classes (`grain_size_phi`); the remark that the volume parameter "has to be adjusted empirically" and that gravel and rock use "heuristic fits" | confirmed (as a report) |
| Bentrem, Sample, Kalcic and Duncan, arXiv:physics/0607146, eq. (7), Table II | open (arXiv) | the centimetre form of the spectrum (`bottom.w2_from_cm4`) | confirmed |
| Olson, Lyons and Saebo, "Measurements of high-frequency acoustic scattering from glacially eroded rock outcrops", arXiv:1601.06078 (J. Acoust. Soc. Am. 139, 1833, 2016) | open (arXiv) | rock: Lambert parameters -13 and -18 dB (plucked, 100 kHz), -33 to -26 dB at 20 degrees (abraded), earlier measurements -15 to -22 dB at 20 degrees, the handbook's rough-rock curve (-8 dB at 20 degrees) exceeding the largest Lambert law; elastic rock parameters: `rock.empirical_backscatter`, `bottom.LAMBERT_MAX_MU_DB`, comparison in `tools/boundary_models_study.py` | confirmed |
| Gruber, "Measurements of acoustic scattering from rocky outcrops in Monterey Bay", MS thesis, Naval Postgraduate School (2019), https://hdl.handle.net/10945/64176 | open, PD | rock at 200 kHz: Lambert parameter -24 to -22 dB with measured slopes (average -23 dB), -35 to -30 dB at 20 degrees; the list of the six rock measurements in the literature | confirmed |
| Fezzani, Berger, Le Bouffant and Lurton, "Multispectral backscatter-based characterization of seafloor sediments using calibrated multibeam and tilted single-beam echosounders", Front. Remote Sens. (2025), doi:10.3389/frsen.2025.1574996 | open, CC BY | calibrated levels at 45 degrees, 35-440 kHz, seven sites (Secs. 3.3-3.4): medium sand about -25 dB above 70 kHz; sandy mud and muddy very fine sand about -22 dB; rock about -13 dB above 80 kHz; maerl, pebbles, boulders -13 to -11 dB above 200 kHz: `pebbles.empirical_backscatter`, the measured rows of the family table | confirmed (text values; curves are figures) |

Per-parameter provenance of the table (each row's `provenance:` block repeats it):

| Rows | rho, nu, delta | mean grain size | w2 | gamma2 | sigma2 |
|---|---|---|---|---|---|
| 18 rows sandy_gravel ... clay (not the next two) | tabulated: UnderwaterAcoustics.jl; second print Cintron Table 1; nine also in GeoHab Table 2-1; whole-phi rows also in Snellen 2011 Table II | tabulated: Cintron Table 1 | regression: Snellen 2013 eq. (5), checked against Snellen 2011 Table II | one value for all: Snellen 2011 | step rule: Snellen 2013; Snellen 2011 Table II |
| medium_sand, very_fine_silt | tabulated: arXiv:1809.08459 Table 1 | tabulated: Cintron Table 1 | tabulated: arXiv:1809.08459 (regression agrees within 0.3 %) | tabulated: arXiv:1809.08459 | tabulated: arXiv:1809.08459 (medium sand: 0.001, the rule says 0.002) |
| rock, pebbles | tabulated: UnderwaterAcoustics.jl; Cintron Table 1 | none (not sediments) | none: Lambert parameter from measurements instead (rock: arXiv:1601.06078, Gruber 2019, Fezzani 2025; pebbles: one statement in Fezzani 2025, flagged unconfirmed) | | |

No row needed a fit to measured curves: no opened source prints a backscattering curve as
numbers against angle and frequency for a sediment class (they are figures). `tools/fit_sediment.py`
remains the route for a site's own data.

### 3.2 Scattering models (`sim/bottom.py`)

| Source | Access | Taken, and where it lives | Status |
|---|---|---|---|
| Our derivation of first-order perturbation theory and of the volume term | - | `perturbation_strength`, `volume_strength` | ours; perturbation factor equal to Jackson's form in arXiv:2009.12420 eqs. (13)-(16), (20) (1e-10); volume term equal to Snellen 2013 eq. (3) (1e-10) |
| Olson and Jackson, "Scattering from layered seafloors", arXiv:2009.12420 | open (arXiv) | check of the perturbation factor | confirmed |
| Olson, "A series approximation to the Kirchhoff integral", arXiv:2106.08287, eqs. (13), (24) | open (arXiv) | form of the Kirchhoff integral and the statement that small-slope and Kirchhoff share it | confirmed |
| Olson, "Validity criteria for the small slope approximation for backscattering of sound from rough surfaces with a power-law roughness spectrum", J. Acoust. Soc. Am. 160, 1918-1935 (2026), https://hdl.handle.net/10945/76181 | open, CC BY | validity limits of the lowest-order small-slope result against boundary-element solutions (1 dB criterion, 1-D surfaces): pressure release k h up to order 1 (0.1 to 2); sand-like fluid about 1.9 times that (0.6 to 3); mud-like up to about 11; rms slope limits about 10 to 60 degrees by exponent; "PT is accurate whenever SSA(2) is accurate" at small grazing angles; composite roughness "largely superseded by the SSA": `bottom.py` docstring | confirmed |
| Olson and Lyons, "Resolution dependence of rough surface scattering using a power law roughness spectrum", arXiv:2106.05909 | open (arXiv) | the composite model written as the tilted perturbation result averaged over large-scale slopes (their eq. (44), after McDaniel and Gorman 1983) and its conditions (k x small-scale rms height < 1; grazing above about twice the large-scale rms slope; shadowing below): supports `composite_strength`. Scintillation index trends (section 6) | confirmed |
| Fonseca, Lurton, Fezzani and Roche, "A simplified semi-empirical model for multifrequency seafloor backscattering angular response (ESAB)", Front. Remote Sens. (2025), doi:10.3389/frsen.2025.1619218 | open, CC BY | eqs. (6)-(8): the scale split where the Rayleigh parameter of the small scale is one, 4 k^2 zeta^2 = 1 ("following Novarini and Caruthers (1994)"), which is `small_scale_kh = 0.5`; small-scale variance and facet slope variance as the two integrals `large_scale_slope` uses. The ESAB model itself (facets + Bragg + volume, fitted across 35-440 kHz) is not implemented | confirmed |
| Siemes, Snellen, Simons and Hermand, "Inverting for bottom parameters in shallow-water soft sediment environments using MBES backscatter strength" (2009), https://repository.tudelft.nl/record/uuid:1d527d12-7168-4dc3-bdd5-d5991fb4c514; Simons and Snellen, "A comparison between modeled and measured high frequency bottom backscattering", Acoustics'08 Paris, https://repository.tudelft.nl/record/uuid:7257d901-e580-4968-a0d4-514850b3d5df | open (repository copies) | the regimes of the published model: "the Kirchhoff approximation for smooth and moderate rough bottoms at grazing angles between 40 and 90 degrees; the composite roughness approximation ... at grazing angles below 40 degrees; the large-roughness approximation for gravel and rock bottoms", joined "by appropriate interpolation" (not printed); the volume term "by applying corrections for slope effects and shadowing" (we apply none); the default sigma2 = 0.002 again; gamma taken as 3.25 | confirmed (statements); the interpolation itself remains wanted |
| Gallaudet, dissertation (2001), https://hdl.handle.net/10945/11015, pp. 47-48 and Table VI (p. 60) | open, PD | statement of the published bottom model's structure and range ("valid for frequencies between 10-100 kHz"; Kirchhoff near normal incidence, composite roughness elsewhere); a fit at 68 kHz on silty clay (Mz 6.36): w2 default 0.000518 cm^4, gamma 3.00, sigma2 0.003 | confirmed |
| Muggleworth, "Shallow water reverberation measurement and prediction", MS thesis, Naval Postgraduate School (1994), https://hdl.handle.net/10945/28529, p. 34, eq. (3.14) | open, PD | Lambert's law as S = mu_dB + 10 log10(sin(ti) sin(ts)); the bound mu = 1 / pi, -4.97 dB (a second print); "In 1961, K. V. Mackenzie reported success using Lambert's law with mu_dB = -29 dB to model bottom backscattering in deep water at 530 and 1030 Hz", for 20 to 60 degrees. The simulator's default for `model: lambert` was -27 dB, written from memory as "the Mackenzie value" (PHYSICS_NOTES Sec. 6.4); it is -29 dB, this print's value, since D-89 (`bottom.DEFAULT_LAMBERT_MU_DB`) | confirmed (law, bound, the value as quoted here); Mackenzie's own paper not opened |
| Morency and Stilwell, arXiv:2210.06535, Sec. II-E | open (arXiv) | SEARAY bottom type law (`surface.searay_bottom_db`); no stated range | via secondary (they quote lecture notes) |
| Kirchhoff cross section prefactor, structure-function constant, rotated-path Kirchhoff integral, small-slope form, composite recipe, "combined" | - | `kirchhoff_strength`, `small_slope_strength`, `composite_strength`, `combined_strength` | ours (tests in `test_bottom_models.py`) |

## 4. Sea surface (`sim/surface.py`, `acoustics.py`)

| Source | Access | Taken, and where it lives | Status |
|---|---|---|---|
| Gallaudet, "Shallow water acoustic backscatter and reverberation measurements using a 68-kHz cylindrical array", dissertation (2001), https://hdl.handle.net/10945/11015, pp. 124-125 (section "Acoustic Backscatter From the Sea Surface") | open, PD | eqs. (7)-(12): the published surface model's bubble term sigma_b = delta_r sin(t) / (8 pi delta) (1 + 8 b e^-2b - e^-4b), delta = 2.55e-2 f^(1/3), delta_r = 0.0136, its saturation and sparse limits; the facet term sec^4 / (4 pi s^2) exp(-tan^2 / s^2); stated range 12-70 kHz, 3-15 m/s, +-5 dB below 8 m/s: `surface.bubble_backscatter`, `facet_backscatter`, `bubble_layer_surface_db`, scenario `model: bubble_layer_surface` | via secondary (he summarises the handbook and McDaniel 1993) |
| arlpy 1.9.3 `uwa.bubble_surface_loss` (BSD) and UnderwaterAcoustics.jl `surface_reflection_coef` (MIT) | open | bubble loss per bounce 1.26e-3 U^1.57 f^0.85 / sin(t), exponential below 6 m/s ("APL model (1994)", "TR 9407 II-21"): `apl_1994_bubble_loss_db` | via secondary |
| Nennouche, Khalighi, Dowhuszko and Merad, arXiv:2603.23727v2, eqs. (20)-(23) | open (arXiv) | Hall-Novarini bubble population (after Keiffer, Novarini and Norton 1995): `hall_novarini_density` | via secondary |
| Beelen, Nijhof, de Jong, van Wijngaarden and Krug, arXiv:2410.14415, eqs. (6)-(9) | open (arXiv) | Commander-Prosperetti effective medium and bubble damping: `bubble_oscillator`, `bubble_layer_medium` | via secondary |
| arXiv:physics/0006060 (Mulhearn) | open (arXiv) | an independent air-fraction number for the population check (PHYSICS_NOTES Sec. 20) | confirmed |
| Morency and Stilwell, arXiv:2210.06535, Sec. II-F | open (arXiv) | empirical high-frequency surface law (`kraus_surface_db`); no stated range | via secondary |
| Chapman and Harris, J. Acoust. Soc. Am. 34 (1962) | no | `acoustics.chapman_harris_db`: the formula, and the docstring's range (0.4-6.4 kHz, below about 40 degrees) | **unconfirmed** (not read in any opened document; wanted) |
| Ead, "Predicting the effects of sea surface scatter on broad band pulse propagation with an ocean acoustic parabolic equation model", MS thesis, Naval Postgraduate School (2004), https://hdl.handle.net/10945/1160, eq. (5.1) | open, PD | Pierson-Moskowitz spectrum: alpha = 8.1e-3, beta = 0.74, w0 = g / U, U at 19.5 m: `rms_surface_vertical_speed` (second moment, exact), `rms_wave_height` (zeroth moment: significant height 0.2092 U^2 / g against the 0.21 in the code, 0.4 %) | via secondary; moments ours (tested numerically) |
| Guerin, Capelle and Hartmann, "Revisiting the Cox and Munk wave-slope statistics using IASI observations of the sea surface", arXiv:2210.05456v2, eq. (12) | open (arXiv) | Cox-Munk clean-surface slopes: upwind 3.16e-3 U, crosswind 3e-3 + 1.92e-3 U (U at 12.5 m, 1-14 m/s): `mean_square_slope_components`. Their sum is 0.8 % below the total regression in `mean_square_slope` (0.003 + 0.00512 U), which itself was not found printed | via secondary; total regression unconfirmed to 0.8 % |
| Rayleigh-parameter coherent reflection exp(-Gamma^2 / 2) | - | `rough_coherent_reflection` | ours (standard Gaussian-height result; not checked against an opened print) |
| Extinction by resonant bubbles, layer refraction, screened loss | - | `bubble_loss_db`, `bubble_layer_loss_db` | ours (tests in `test_bubble_layer.py`) |

## 5. Forward scatter: time, angle and Doppler spreading (`sim/timespread.py`; no change made)

| Source | Access | Taken | Status |
|---|---|---|---|
| TR 0702 eqs. (89)-(103), p. 98-104 | open | rule of thumb for the decay time; relations between spreads and coherence widths; "less than 10 Hz" | confirmed |
| Facet model: decay time, tap directions, Doppler width 2 f v sin(g) / c / sqrt(1 - exp(-Gamma^2)), multi-bounce convolution | - | `timespread.py` | ours |
| Li, Chitre, Cuji, Preisig, Singer, Stojanovic and van Walree, "Underwater Acoustic Channel Library", arXiv:2609.03207, with data Zenodo doi:10.5281/zenodo.21287414 (CC BY 4.0) | open | measured channels (SPACE'08 and others): the comparison of PHYSICS_NOTES Sec. 21 | confirmed (data) |
| Dessalermos, "Undersea acoustic propagation channel estimation", MS thesis, Naval Postgraduate School (2005), https://hdl.handle.net/10945/2163, Tables 4-13 | open, PD | measured Doppler spread of each resolved path, 12 kHz band (8-16 kHz probes), 30 m of water, source drifting, ten ranges 700 m to 6.55 km: e.g. 1.27 to 2.84 Hz with delay at 700 m, 0.89 to 2.85 Hz at 2300 m, 0.66 to 2.06 Hz at 4350 m. The spread grows with a path's excess delay, that is with its steepness and number of bounces | confirmed (wind not stated: a trend, not a test) |
| Dahl's forward-scatter papers (J. Acoust. Soc. Am. 100, 1996; 105, 1999; 115, 2004; IEEE J. Oceanic Eng. 26, 2001), Preisig and Deane (2004) | no | - | wanted |

## 6. Reverberation statistics (`statistics: {model: k, shape}`)

| Source | Access | Taken | Status |
|---|---|---|---|
| Olson and Lyons, arXiv:2106.05909, abstract and Sec. VIII | open (arXiv) | the scintillation index of backscatter from power-law roughness "increases as resolution increases, grazing angle decreases, and spectral strength increases" (100 and 10 kHz; exponents 1.5-2.5): there is no single shape value | confirmed |
| Olson, Lyons, Abraham and Saebo, "Scattering statistics of rock outcrops", arXiv:1901.08023 | open (arXiv) | "the K-distribution shape parameter is related to the effective number of scatterers within a resolution cell (Abraham and Lyons, 2002)"; single-component K fits fail on rock at 100 kHz, 3 cm resolution; mixtures needed | confirmed |
| Lehman, "Resolution dependence of acoustic scattering statistics for complex seafloors", MS thesis, Naval Postgraduate School (2022), https://hdl.handle.net/10945/70737 | open, PD | 100 kHz synthetic aperture sonar, 3.7 x 3.2 cm: rock, gravel, sand and mud all have scintillation index "much greater than 1 for most" resolutions, falling as the resolution is reduced and never reaching the Rayleigh value | confirmed (values are figures) |
| Abraham and Lyons, IEEE J. Oceanic Eng. 27 (2002) | no | the relation between shape and scatterers per cell | wanted |

## 7. Targets, noise, medium

| Source | Access | Taken, and where it lives | Status |
|---|---|---|---|
| Lavia, Gonzalez and Blanc, arXiv:1805.06945v2, eqs. (1), (5)-(7) | open (arXiv) | Kirchhoff backscattering amplitude of sphere, cylinder, disc: `sim/targets.py` | confirmed |
| Urick, *Principles of Underwater Sound*, table of simple forms | no | nothing (the cylinder obliquity factor cos against cos^2 stays as the Kirchhoff integral gives it) | wanted |
| Hurt, "Effect of wind noise on undersea acoustic network performance and design", MS thesis, Naval Postgraduate School (2005), https://hdl.handle.net/10945/1936, eqs. (2.7)-(2.9) and listing | open, PD | the four-term ambient noise fit with its attributions (wind: Wenz 1962; thermal, shipping: Urick 1983): third print of `noise.wenz_spectrum_db` | via secondary |
| ns-3 UAN noise model; arXiv:1801.03641 | open | first two prints of the same constants | via secondary |
| Coates (1990), Wenz (1962) | no | named in the docstring as the origin | unconfirmed attribution |
| Shajahan, Barclay and Lin, "Mapping of surface-generated noise coherence", Front. Mar. Sci. 9:1016702 (2022) | open, CC BY | eq. (3) isotropic sin(kd) / kd; eq. (4) Cron-Sherman vertical coherence; that the latter fails in shallow, range-dependent water (first zero crossing off by up to 30 %): `noise.isotropic_coherence`, `noise.surface_noise_coherence` | via secondary (closed form); general geometry ours |
| AlphaStableDistributions.jl (org-arl, MIT), UACR site models | open | symmetric alpha-stable generator; alpha = 1.7 for the Singapore site model | via secondary |
| Francois and Garrison (1982); Ainslie and McColm (1998) | no | `acoustics.absorption_francois_garrison`, `absorption_ainslie_mccolm` | via secondary: within 0.6 % (and 3 %) of 21 values "calculated using the Francois-Garrison equation" in the thesis below, whose eq. (2.3) prints the Ainslie-McColm form with our constants and two relaxation frequencies (0.769 and 41.372 kHz) that ours reproduce; also held to arlpy 1.9.3 from 10 kHz to 1 MHz (rel 1e-9) and to arlpy's printed 10.71 dB/km at 50 kHz |
| Col, "Fine-scale variability in temperature, salinity, and pH in the upper-ocean and the effects on acoustic transmission loss in the Western Arctic Ocean", MS thesis, Naval Postgraduate School (2010), https://hdl.handle.net/10945/5385, Tables 1-2, eq. (2.3) | open, PD | the table used above | confirmed |
| Mackenzie (1981); Marczak (1997) | no | `acoustics.sound_speed` | Mackenzie: unconfirmed in print, held to arlpy 1.9.3 to 1e-13 over a T/S/D grid and to arlpy's printed 1539.1 and 1534.6 m/s (`test_medium_models.py`); stated validity ranges not read. Marczak: unconfirmed, no published check value tested. Between 0 and 5 PSU: ours, a linear blend in salinity between the two (D-146) |
| arlpy `uwa.py:101-121`; UnderwaterAcoustics.jl `basic.jl:111-121` | open | sea water density at one atmosphere: `water_density`; Wood's equation: `bubbly_sound_speed` | via secondary |
| UnderwaterAcoustics.jl `stdlib.jl:411-422` | open, MIT | sea state to wind speed | via secondary |
| UnderwaterAcoustics.jl `basic.jl:43-54`, `stdlib.jl:236-252` | open, MIT | Hamilton's shear-speed regression and shear attenuations (data only, `docs/ORG_ARL_SURVEY.md` Sec. 9) | via secondary; Hamilton's papers not read |

## 8. Measured data that could serve as validation

| Data | Band | What it has | Availability |
|---|---|---|---|
| Fezzani et al. 2025 (doi:10.3389/frsen.2025.1574996) | 35-440 kHz | calibrated backscattering strength against angle (-9 to 75 degrees) on seven seabed types, with ground truth | article CC BY; "original contributions ... included in the article/Supplementary Material"; the curves are figures, numbers on request from the authors |
| Fonseca et al. 2025 (doi:10.3389/frsen.2025.1619218) | 35-440 kHz | the same campaign with model fits | article CC BY |
| Olson, Lyons and Saebo, arXiv:1601.06078 | 100 kHz | rock: scattering strength against angle, measured roughness spectra | paper only |
| Gruber 2019; Fromm 2020 (https://hdl.handle.net/10945/66640) | 200 kHz | rock outcrops, split-beam | theses, public domain; data not published |
| Gallaudet 2001 | 68 kHz | bottom (silty clay), surface and volume backscatter with a calibrated array; model fits | dissertation, public domain; data as figures and tables |
| Dessalermos 2005 | 8-16 kHz | per-path Doppler spread and shift at ten ranges in 30 m of water | thesis, public domain; tables |
| UACR, Zenodo doi:10.5281/zenodo.21287414 | 12.5 kHz and others | measured time-varying channels (already used, PHYSICS_NOTES Sec. 21) | CC BY 4.0 |
| Zenodo doi:10.5281/zenodo.17393662, "Interferometric Synthetic Aperture Sonar data from the FKt231024 expedition" | SAS | image products (TIFF) | CC BY 4.0 per the Zenodo listing; not inspected |
| GeoHab report (doi:10.5281/zenodo.10089261) | 30-300 kHz | reference curves and good practice | CC BY 4.0 |
| Lehman 2022 | 100 kHz SAS | statistics by seabed type | thesis public domain; the imagery (FFI) is not public |

No open measured set was found for bottom backscatter at 4-10 kHz, for surface reverberation
above 10 kHz as numbers, or for high-frequency forward scatter (coherent fraction and delay
spread against wind).

## 9. Where the opened documents are, and what refused access

Opened documents of the 2026-11-17/18 search are in the session's scratch area,
`scratchpad/lit/<name>/` (one directory each, with the URL, licence or terms as stated by the
host, and retrieval date in `SOURCE.json` or `SOURCE.txt`); they are not in the repository.
Directories: `snellen2011_sbes`, `snellen2013_intercomparison`, `nps_yellowsea2001`,
`gallaudet2001_nps`, `nps_ssa_validity2026`, `nps_rock_meas2019`, `arxiv_1601.06078`,
`frontiers_frsen2025_1574996`, `frontiers_esab2025`, `nps_resolution_stats2022`,
`arxiv_1901.08023`, `nps_windnoise2005`, `nps_arctic_ph2010`, `nps_surface_scatter_pe2004`,
`nps_channel_est2005`, `arxiv_2210.05456`, `frontiers_fmars2022_1016702`, and others opened
and not used.

Refused or unusable from this session (nothing was done to get around any of them):

| Host | What happened |
|---|---|
| www.mdpi.com | "Access Denied" (HTTP 403) to every article and PDF request, also after a retry |
| pubs.aip.org | browser challenge page (HTTP 403) |
| ieeexplore.ieee.org | not tried again after earlier refusals; OpenAlex lists no open copies of the papers wanted |
| link.springer.com | open-access articles redirect through idp.springer.com, which the session may not reach |
| www.ioa.org.uk | proceedings PDFs listed by OpenAlex return "Not found"; the site search is script-driven |
| jcaa.caa-aca.ca | connection closed without a reply |
| core.ac.uk | browser challenge page |
| dspace.mit.edu | search works; files are served from another host (not reachable) or answer "too many requests" |
| hal.science | files redirect to institutional subdomains that the session may not reach |
| api.openalex.org, api.semanticscholar.org | usable but rate-limited (shared anonymous budget); several hundred results were read |
| repository.tudelft.nl, calhoun.nps.edu, arxiv.org, export.arxiv.org, www.frontiersin.org, zenodo.org, github.com | worked |
| ocw.mit.edu, ocw.tudelft.nl, munin.uit.no, repository.gatech.edu, ar5iv.labs.arxiv.org | 2026-11-20: refused by the session's network proxy (not tried further) |
| apps.dtic.mil | 2026-11-20: an HTML page in place of the report PDF (not tried further) |

## 10. Wanted: documents that would settle open points

Each line says exactly what is needed, so that a page or a formula is enough. The
bibliographic details in this section are pointers only: they come from search listings, from
the reference lists of opened documents and from memory, and were not checked against the
documents themselves (which we could not open).

| Open point | Document | What is needed from it |
|---|---|---|
| Bubble loss per surface bounce: our population model is 2 to 3 times the adopted law in dB at 10-30 kHz | "Field measurements and modeling of attenuation from near-surface bubbles for frequencies 1-20 kHz", J. Acoust. Soc. Am. 124, EL163 (2008) (listed by OpenAlex; publisher copy behind a challenge page); Dahl's and Ainslie's papers on bubbles and surface loss (2003-2005) | measured loss in dB per bounce (or depth-integrated extinction beta_I) against wind speed at stated frequencies and grazing angles; any statement on patchiness |
| The handbook's law for beta_I and its small-scale roughness function G(U) | McDaniel, "Sea surface reverberation: a review", J. Acoust. Soc. Am. 94, 1905 (1993); Dahl et al., J. Acoust. Soc. Am. 101 (1997) | beta_I(U, f) as a formula; G(U10) of sigma_sc = G tan^4; the interpolation between facet and small-scale terms |
| Coherent fraction, delay spread, angle spread and scintillation of surface forward scatter (PHYSICS_NOTES Sec. 21: too coherent at low grazing; scintillation index above 1) | Dahl, J. Acoust. Soc. Am. 100, 2742 (1996); 105, 2155 (1999); 115, 589 (2004); IEEE J. Oceanic Eng. 26, 141 (2001); Preisig and Deane, J. Acoust. Soc. Am. 116, 2067 (2004) | measured coherent reflection loss against the Rayleigh parameter; time spread (e-folding) and angular spread against wind, range and frequency; the focusing explanation with numbers |
| Doppler spread against grazing angle (TR 0702 p. 101 prints 1 / sin) | TR 9407 "Eq. II-43a" as quoted in any open paper; or Dahl 2001 | the printed formula with its symbols defined |
| Chapman-Harris fit and its range | Chapman and Harris, J. Acoust. Soc. Am. 34, 1592 (1962) | the formula as printed and the stated ranges of frequency, wind and angle |
| Interpolation between Kirchhoff and composite results; the handbook's cut-off; the rough-bottom "empirical expression" for gravel and rock | Mourad and Jackson, Proc. OCEANS '89, 1168; Jackson, Winebrenner and Ishimaru, J. Acoust. Soc. Am. 79, 1410 (1986); Jackson and Richardson, *High-Frequency Seafloor Acoustics* (2007), Ch. 13 and Appendix | the interpolation formula; the cut-off wavenumber rule; the empirical large-roughness cross section with its constants |
| Bistatic bottom model | Williams and Jackson, J. Acoust. Soc. Am. 103, 169 (1998) | the bistatic cross section and its interpolation (to compare with ours away from backscatter) |
| Regressions of density, speed and loss against grain size (ours interpolates the half-phi table) | TR 9407 as quoted in any open paper; Hamilton, J. Acoust. Soc. Am. 68, 1313 (1980); Hamilton and Bachman (1982) | the piecewise polynomials rho(Mz), nu(Mz), delta(Mz) |
| Medium sand: volume parameter 0.001 or 0.002 | any second print of the handbook's sediment table with the volume parameter | the value for "medium sand" |
| K-distribution shape against cell size | Abraham and Lyons, IEEE J. Oceanic Eng. 27, 800 (2002); Lyons and Abraham, J. Acoust. Soc. Am. 106, 1307 (1999) | shape = f(number of scatterers per cell); measured shapes with beamwidth, bandwidth, range and seabed type |
| Target strength of simple forms | Urick (1983), Table 9.1 | the cylinder entry (cos or cos^2 of the angle from broadside) and its stated conditions |
| Lambert parameter for the deep sea floor | Mackenzie, J. Acoust. Soc. Am. 33 (1961) | the value of 10 log10(mu) as printed (-27 or -29 dB), with its frequencies and angle range |
| Ambient noise fit's origin | Coates, *Underwater Acoustic Systems* (1990) | the four expressions as first printed |
| Sound speed formula ranges | Mackenzie, J. Acoust. Soc. Am. 70, 807 (1981) | stated ranges of temperature, salinity and depth and the stated error |
| Calibrated bottom curves as numbers | Wendelboe, Mar. Geophys. Res. 39 (2018), doi:10.1007/s11001-018-9350-y (open access, but behind a redirect we cannot follow); Trzcinska et al., Remote Sens. 13, 4771 (2021) (MDPI) | tables or supplementary data of backscattering strength against angle and frequency on sand: input for `tools/fit_sediment.py` |

## 11. Differences between sources, and choices left to the owner

| Item | Evidence | Now | Proposal |
|---|---|---|---|
| `medium_sand` volume parameter | arXiv:1809.08459 Table 1: -20.0 dB/m at 10 dB/m, i.e. 0.0010. Snellen 2011 Table II and 2013 text: 0.002 below 5.5 phi. The same arXiv table agrees with the rule for very fine silt | 0.0010 kept (the row's own source) | decided (D-90): kept; a sourced tabulated value outranks a rule. 0.002 would make the family consistent and raise the row by 0 to 2.5 dB (most at 6 kHz and 45 degrees); a second print of the table is still wanted |
| `sandy_gravel` speed ratio | 1.357 in the Julia table; 1.337 in its tests, in Snellen 2011 Table II and in Cintron Table 1 | corrected to 1.337 | done (a typing error in one source); reflection of that row changes slightly |
| Default roughness model `perturbation` | Olson 2026: small-slope is "the current state of the art" and is valid wherever perturbation theory is, and beyond | `small_slope` is the default (D-86); `perturbation` selectable | done on the owner's decision |
| Default sediment parameters against measurements | Fezzani 2025: medium sand about -25 dB at 45 degrees, flat above 70 kHz; the defaults give -22.5 dB (100 kHz) and -19.7 dB (300 kHz). Snellen 2013 at 300 kHz: fitted w2 twice, sigma2 half the defaults. Gallaudet at 68 kHz on silty clay: sigma2 0.003 against 0.001. The published model's stated range is 10-100 kHz | defaults as published | none to the table; use a site fit above 100 kHz, and expect +-3 to 6 dB from class defaults |
| Rock | measured Lambert parameters span -23 to -10 dB; the type law `searay_bottom` rock (-7 dB at 20 degrees) and the handbook curve it resembles exceed the energy bound | `rock` carries mu = -16 dB with the range; `{boundary_z, sediment: rock}` uses it; `searay_bottom: rock` warns (D-88) | done on the owner's decision; no shipped scenario used `searay_bottom: rock`, so none moved |
| Surface backscatter above about 10 kHz | published bubble model saturates at delta_r sin(t) / (8 pi delta): -29 dB at 10 degrees and 30 kHz. Chapman-Harris extrapolated gives -19 dB at 68 kHz, 15 m/s, 10 degrees; the Kraus law -23 dB | the laws are unchanged; they warn once above the frequency where they exceed the saturation level, and take an opt-in `cap: bubble_saturation` (D-87) | done on the owner's decision: `bubble_layer_surface` is the documented choice for 12-70 kHz; no shipped scenario has a surface item in that band, so none was switched |
| Default Lambert parameter -27 dB (`reverberation` items without a model) | Muggleworth (1994) quotes Mackenzie (1961) as -29 dB at 530 and 1030 Hz; Mackenzie's paper itself was not opened | -29 dB (`bottom.DEFAULT_LAMBERT_MU_DB`, D-89), with a warning at 10 kHz and above | done on the owner's decision; in either case it is a deep-water value at about 1 kHz, not a high-frequency one (measured sands at 45 degrees and 100 kHz and above are near -25 dB, i.e. mu about -22 dB) |
| Doppler spectrum default `lorentzian` | measured SPACE'08 spectra are Gaussian (PHYSICS_NOTES Sec. 21); no opened paper contradicts | unchanged here (another package owns `timespread.py`) | as already proposed there |

## 12. Algorithms and oracles (processing chain; simulator test oracles)

Added 2026-11-20 for the audit's group C (C1-C8). Not settled first: the Swerling (1960),
Capon (1969), Lugannani-Rice (1980), Finn-Johnson (1968), Munk (1974) and Bar-Shalom
originals were not opened; Sæbø 2010 and Cook 2007 could not be reached (open copies are
listed by search engines, but the hosts are refused from this session, section 9); the
dB/wavelength conversion and the most-probable-first association are ours. Each
"confirmed" below names what was opened and what was compared with the code; documents
opened on 2026-11-20 are in the session's scratch area, `scratchpad/litc/<name>/` (with
`SOURCE.txt`), not in the repository. Numbers marked "ours" were computed here, not read.

### 12.1 Detection

| Source | Access | Taken, checked against the code | Status |
|---|---|---|---|
| Weinberg, "An Introduction to Radar Sliding Window Detectors", arXiv:1709.09786v1, eqs. (6)-(8) | open (arXiv) | cell-averaging CFAR: test Z0 > tau sum of N training cells, P_FA = (1 + tau)^(-N) in exponential clutter, so tau = P_FA^(-1/N) - 1. The code's factor multiplies the training MEAN: N (P_FA^(-1/N) - 1) x mean = tau x sum, the same test (`libs/detect` `cfar.cpp:16-19`, `cfar_kernel.hpp:29-36`; Python `detect.cfar_threshold_factor`). The factor is recomputed for the number of cells that exist at the map's edges (`cfar.cpp:46-50`), which is the same formula with that N. Example, ours: N = 64, P_FA = 1e-4 gives 9.906, and (1 + 9.906 / 64)^-64 = 1.0e-4 | confirmed (formula); Finn and Johnson, RCA Review (1968), the paper's ref. [1], not opened |
| Kato, Sekine and Yoshikawa, "Order Estimates for the Exact Lugannani-Rice Expansion", arXiv:1310.3347v3, eqs. (1.1), (1.2), (2.1), (2.3) and the expression for Psi_0 | open (arXiv) | P(X > x) ~ 1 - Phi(w) + phi(w) (1/u - 1/w), w = sgn(theta) sqrt(2 (x theta - K(theta))), u = theta sqrt(K''(theta)), K'(theta) = x: `libs/detect` `statistic.cpp:66-102` (`weighted_exponential_tail`) is this with K the cumulant generating function of the weighted sum of exponentials; above the mean theta > 0, so sgn = +1. Below the mean the code returns 0.5 (its own guard, not from the source). Ours: two equal weights, one row (Erlang 2, mean 1) at x = 3: 0.017411 against the exact e^-6 (1 + 6) = 0.017351 (0.3 %) | confirmed (formula); Lugannani and Rice, Adv. Appl. Prob. 12 (1980), not opened |
| Garcia, Flores Rodriguez and Fraidenraich, arXiv:2012.06878v1, Sec. I and eqs. (8)-(9); Geiger and Vogel, arXiv:1107.1972v2, eqs. (13)-(14), (19) | open (arXiv) | a steady echo's square-law statistic over N independent complex samples is noncentral chi-square with 2N degrees of freedom and noncentrality 2 x (signal power / total noise power) summed over the samples (P_D = Q_N(sqrt(2 zeta), sqrt(2 gamma)), zeta = sum xi / 2 sigma^2); squared magnitudes are normalised by the per-component variance. `live.detection_probability`: dof = 2 rows looks, threshold dof T, noncentrality dof s share (`live.py`, the `steady` helper) - the same with the noise power per sample normalised to 1. That an exponential power fluctuation is the Swerling I and II model: a statement in arXiv:2012.06878 Sec. I. The "swerling1" option averages over an exponential power with 48-point Gauss-Laguerre; ours: one sample, P_FA 1e-4, 13 dB gives 0.644301 against exp(-T / (1 + s)) = 0.644307 (our closed form). The 2 x rows x looks degrees of freedom for a correlated cell mean is an approximation for the signal's share (as the docstring says): ours | confirmed (distribution, dof, noncentrality); exponential model via secondary; Swerling, IRE Trans. IT-6 (1960), not opened |

### 12.2 Beamforming and bearings

| Source | Access | Taken, checked against the code | Status |
|---|---|---|---|
| Tsai, Su, Tsao and Wang, "Robust Beamforming Against Direction-of-Arrival Mismatch Using Subspace-Constrained Diagonal Loading", arXiv:1602.02690v2, eqs. (5), (7), (9) | open (arXiv) | MVDR weights w = R^-1 a / (a^H R^-1 a) (their eq. (7), the Capon beamformer of their ref. [1]); diagonal loading as minimising w^H (R + gamma I) w under the same constraint (eq. (9), after Carlson 1988, their ref. [4]). `libs/live` `beamform.cpp:332-352` and `flank.mvdr_beamform` compute R^-1 a by Cholesky (LU if not positive definite) and divide by a^H R^-1 a; the output is then multiplied by the number of elements (documented in `beamform.hpp`) | confirmed (formulas); Capon, Proc. IEEE 57 (1969) and Carlson, IEEE Trans. AES 24 (1988) not opened |
| The loading 3e-2 x the mean element power (trace R / N) | - | `MvdrOptions::loading`, `flank.mvdr_beamform(loading=3e-2)`; the opened source gives no value ("a way of choosing the diagonal loading factor is still lacking", its Sec. I). Training from the other range blocks with guard blocks, sub-band split: also ours | ours |
| Gerstoft, Mecklenbräuker and Xenaki, "Multi Snapshot Sparse Bayesian Learning for DOA Estimation", arXiv:1602.09120v1 (TeX source and PDF), eqs. (10), (SBL1), (26), (28) | open (arXiv) | gamma_new = (1/sqrt L) ||mu_m|| / sqrt(a_m^H Sigma_y^-1 a_m), mu_m = gamma_m a_m^H Sigma_y^-1 Y; noise (N - K)^-1 tr((I - A_M A_M^+) S_y); convergence ||gamma_new - gamma_old||_1 / ||gamma_old||_1. `libs/live` `doa.cpp:347-365` (one frequency: the same update), `doa.cpp:398-411` (the same noise estimate, per frequency, with a floor of 1e-6 of the data power: ours), `doa.cpp:366` (the same convergence measure). The journal version cited in `doa.hpp` (IEEE Signal Process. Lett. 23, 2016, with Nannuru) was not opened | confirmed (arXiv version) |
| Nannuru, Gemba, Gerstoft, Hodgkiss and Mecklenbräuker, "Sparse Bayesian learning with uncertainty models and multiple dictionaries", arXiv:1704.00436v2 (TeX source), the multi-dictionary update and the remark on the exponent b | open (arXiv) | gamma_new = gamma_old (sum_f a^H Sigma^-1 S_y Sigma^-1 a / sum_f a^H Sigma^-1 a)^b over frequencies with one gamma; "b = 0.5 gives the update equation in [Gerstoft 2016]", b = 1 Tipping's; the paper's own runs use b = 1. `doa.cpp:342-365` sums numerator and denominator over frequency and takes the square root: b = 0.5. The coarse-to-fine grid, the pruning and the peak-based noise estimate are ours | confirmed (arXiv version; Signal Processing 159 (2019) not opened) |

### 12.3 Tracking

| Source | Access | Taken, checked against the code | Status |
|---|---|---|---|
| Labbe, *Kalman and Bayesian Filters in Python*, GitHub rlabbe/Kalman-and-Bayesian-Filters-in-Python, notebooks 06 and 07 ("Piecewise White Noise Model") | open, CC BY 4.0 (README) | constant-velocity transition F = [[1, dt], [0, 1]]; piecewise-constant acceleration noise Q = Gamma sigma^2 Gamma^T, Gamma = [dt^2 / 2, dt]^T; S = H P H^T + R, K = P H^T S^-1, P = (I - K H) P. `detect.Tracker._predict`/`_take` and `libs/live` `tracker.cpp` (`predict()` and the `take` step of `Tracker::update`) use these per horizontal axis, sigma = `process_noise` | confirmed (as a textbook); Bar-Shalom et al. not opened |
| Gating at d^2 < gate^2, the polar-to-Cartesian measurement covariance (Jacobian of x = r sin b, y = r cos b), and the association (D-141): (track, detection) pairs inside the gate taken most probable first by d^2 + ln|S| (twice the Gaussian negative log-likelihood less a constant), confirmed tracks first, each used once | - | `detect.Tracker._associate_by_likelihood`, `tracker.cpp` (`Tracker::update`, the `Association::Likelihood` branch) | ours (a greedy global-nearest-neighbour rule; no published source compared) |

### 12.4 Propagation formulas and oracles

| Source | Access | Taken, checked against the code | Status |
|---|---|---|---|
| Li and Chitre, "Data-aided Underwater Acoustic Ray Propagation Modeling", arXiv:2205.06066, eqs. (25)-(26) | open (arXiv) | Rayleigh reflection coefficient (attributed to Brekhovskikh and Lysanov, *Fundamentals of Ocean Acoustics*, 2003): Gamma = (rho_r cos g - sqrt((delta_bar / c_r)^2 - sin^2 g)) / (rho_r cos g + ...), delta_bar = 1 + i delta, rho_r and c_r seabed over water; g the incidence angle (the form implies it is measured from the normal). With g = 90 deg - grazing it is the code's R = (m sin t - sqrt(n^2 - cos^2 t)) / (m sin t + ...), n = c_w (1 + i delta) / c_b: `libs/sim` `paths.cpp:140-150`, `propagation.rayleigh_reflection` | via secondary |
| Olson and Jackson, arXiv:2009.12420, eq. (1); Snellen et al. 2013 (section 3.1), definition of kappa | open (arXiv; repository copy) | complex sound speed c / (1 + i delta), delta the ratio of imaginary to real wavenumber: the code's `c_b = speed / (1 + i delta)` | confirmed |
| dB per wavelength to delta: delta = alpha / (40 pi log10 e) = alpha / 54.58 (one wavelength attenuates the amplitude by exp(-2 pi delta)) | open (code) | `paths.cpp:141`, `propagation.rayleigh_reflection`. Compared with the Acoustics Toolbox's `misc/AttenMod.f90` (HEAD 8b4682b, read only, GPL-3.0), unit 'W': alpha f / (8.6858896 c) neper/m, entering as an imaginary sound speed alpha_T c^2 / omega = c alpha / (2 pi 8.6858896): the same delta to first order; the Toolbox's form is linear in alpha, ours is exact for the stated loss per wavelength (they differ by the factor 1 + delta^2, below 1e-3 for 1.6 dB/wavelength); the sign of the imaginary part follows each code's time convention | ours; agrees with an opened implementation to first order |
| Acoustics Toolbox `misc/munk.f90` (HEAD 8b4682b, read only) | open, GPL-3.0 | Munk profile: c = 1500 (1 + 0.00737 (x - 1 + exp(-x))), x = 2 (z - 1300) / 1300: the same constants as `sim.bellhop.munk_profile`'s defaults (axis speed `NOMINAL_SOUND_SPEED` = 1500). The code's `axis_depth` moves only the axis; the scale depth stays 1300 m | via secondary (the Toolbox names it "Munk profile"); Munk, J. Acoust. Soc. Am. 55, 220 (1974), not opened |
| Acoustics Toolbox: BELLHOP, KRAKEN (github.com/oalib-acoustics/Acoustics-Toolbox) | open, GPL-3.0 | oracle only, run as a separate process; build, version and test cases in `docs/OPEN_SOURCE.md` Sec. 3.6; licence row in its table. Porter's manuals and Porter and Bucker (1987) not opened | software (licence and use: `OPEN_SOURCE.md`) |

### 12.6 Ultrasound benchmark (PICMUS)

| Source | Access | Taken | Status |
|---|---|---|---|
| Liebgott, Rodriguez-Molares, Cervenansky, Jensen and Bernard, "Plane-Wave Imaging Challenge in Medical Ultrasound", IEEE IUS 2016 | open (challenge site copy) | eq. (1), the CNR; the −6 dB width; it prints no result values | confirmed |
| PICMUS challenge archive `archive_to_download.zip` (creatis.insa-lyon.fr): `us_contrast.m`, `us_resolution.m` (Bernard, Rodriguez-Molares 2016), `das_iq.m` (F 1.75, Tukey 25 %), simulation phantom and scan files, `evaluation/*_simu_evaluation_from_iq.txt` | open; no licence file; site terms "free of use", cite the paper (read 2026-11-21) | the metric code (ported in `benchmarks.py`), the phantom geometry and the reference DAS outputs | confirmed |

### 12.5 Synthetic aperture processing (sources cited in `docs/PAPER_NOTES.md`)

| Source | Access | Taken | Status |
|---|---|---|---|
| Gerg, Brown, Wagner, Cook, O'Donnell, Benson and Montgomery, "GPU Acceleration for Synthetic Aperture Sonar Image Reconstruction", arXiv:2101.05888v2 | open (arXiv) | the reconstruction and motion-estimation pipeline, eqs. 3-8, and the speedups compared in the benchmarks: `docs/PAPER_NOTES.md` | confirmed |
| Brown, Gerg and Blanford, "Interpolation kernels for synthetic aperture sonar along-track motion estimation", IEEE J. Oceanic Eng. (2019), doi 10.1109/joe.2019.2921510 | not opened (IEEE paywall; open-copy hosts blocked from this session) | ASASIN's ref. [9], the along-track surge estimator; not implemented | unconfirmed (bibliographic entry from the ASASIN arXiv HTML) |
| Sæbø, "Seafloor depth estimation by means of interferometric synthetic aperture sonar", PhD dissertation, University of Tromsø (2010) | no (open copy listed at munin.uit.no; host refused from this session) | coarse delay by quadratic interpolation of the correlation peak, fine delay from the phase at that point: as ASASIN describes it (its ref. [11]); `motion/delay_estimation` | via secondary (ASASIN); the dissertation unconfirmed (not opened) |
| Cook, "Synthetic aperture sonar motion estimation and compensation", MS thesis, Georgia Institute of Technology (2007) | no (record at repository.gatech.edu; host refused from this session) | the displaced phase centre method that ASASIN's sway and heave estimate derives from (its ref. [5]) | via secondary (ASASIN); the thesis unconfirmed (not opened) |
