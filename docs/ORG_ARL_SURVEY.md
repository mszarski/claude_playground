# Survey of the org-arl repositories (arlpy and its Julia successors)

Date: 2026-11-17. Scope: what the Acoustic Research Laboratory (NUS) publishes under
`github.com/org-arl` that bears on underwater acoustics, compared item by item with this
repository at `claude/gpu-sonar-suite-u289p5` (HEAD `ed37689`). This is a reading survey: no
code was copied, nothing here changes the simulator. Adoption is a later, separate step.

Rules followed: APL-UW TR 9407 was not read or downloaded. Where an org-arl file says its
numbers come from TR 9407, the numbers below are quoted from that file (MIT or BSD), with the
attribution that file gives. Every formula and value quoted was read in the file and at the
lines stated. Julia could not be run here, so "their" numbers are hand evaluations of their
printed formulas in Python (`arlpy` 1.9.3 is installed and was called where it has the same
function).

Summary of what matters:

1. `UnderwaterAcoustics.jl` prints a 22-row seabed table (density, sound speed, loss parameter)
   that it attributes to TR 9407. It agrees with both of our "confirmed" rows and supplies the
   same three parameters for 20 more sediments. It does not carry roughness or volume
   parameters, so it settles forward reflection loss, not backscatter.
2. The bubble surface-loss law in arlpy is repeated in the Julia package with a more exact
   attribution ("TR 9407 II-21"). Ours is 1.3 to 2.3 times its dB at 4 to 10 kHz and 7.5 to 9
   times at 100 to 300 kHz (10 m/s, 10 degrees grazing).
3. A sea-state to wind-speed table, a sea water density equation, impulsive (alpha-stable)
   noise generators and a DEMON estimator are present there and absent here.
4. Nothing in org-arl gives a scattering strength law, a reverberation model, a rough-surface
   coherent reflection formula, an ambient noise spectrum (Wenz, Knudsen, wind, shipping,
   thermal) or an elastic or layered reflection coefficient of its own. On those items this
   repository already has more than they do.

## 1. What was opened

### 1.1 How the organisation was enumerated

The GitHub API for organisations is refused by this session's proxy (HTTP 403, "sessions are
bound to their configured repositories"). The list of repositories was read from the
organisation's web page instead (two pages, 44 public repositories, through the page-reading
tool, which returns a model-written extraction of the page, not the raw HTML). Repositories
were then cloned anonymously with `git clone --depth 1` into
`/tmp/claude-0/-home-user-claude-playground/32e65dcd-82ae-5f75-a363-7372d2e7f847/scratchpad/orgarl/`.
Licences were read from the LICENSE file in each tree. Nothing cloned was executed.

### 1.2 Repositories cloned and read

| Repository | Licence (file read) | Commit read | Committed | What was read |
|---|---|---|---|---|
| arlpy | BSD 3-clause (`LICENSE`; file headers say "Simplified BSD") | `2627a8c` | 2026-09-06 | `arlpy/uwa.py` in full; function lists and selected bodies of `signal.py`, `bf.py`, `stable.py`, `uwapm.py`, `comms.py`. Installed 1.9.3 has byte-identical `uwa.py`, `signal.py`, `bf.py`, `stable.py`, `comms.py`; `uwapm.py` differs |
| UnderwaterAcoustics.jl (v0.9.0) | MIT (`LICENSE.txt`) | `7b9c47e` | 2026-11-12 | `src/basic.jl`, `src/stdlib.jl`, `src/pekeris.jl`, `src/replay.jl` in full; `src/scatterers.jl` most; `src/adiabatic.jl` first 140 lines; `src/api.jl` and `src/reframe.jl` docstrings and selected bodies; `qdocs/` pages `basic`, `replay`, `scatterers`, `pekeris-rays`, `raysolver` (parts), `api` (boundary section), `references.bib`; `ROADMAP.md`; tests grepped |
| AcousticsToolbox.jl | MIT, except `src/bellhopjl/` and `src/krakenjl/` which are GPL-3.0-or-later (`LICENSE`, `LICENSE-GPL-3.0`, README "Licensing") | `82295d0` | 2026-11-05 | README; docstrings of `bellhop.jl`, `kraken.jl`, `orca.jl`; `common.jl` grepped; in the GPL part only `bellhopjl/attenuation.jl`, the head of `bellhopjl/reflection.jl`, `krakenjl/bcimp.jl` (head) and the two `PORTING_NOTES.md` (to know what exists) |
| AcousticRayTracers.jl | MIT | `4bade5e` | 2026-11-05 | `src/RaySolver.jl` lines 1-140 and a keyword search of the rest |
| SignalAnalysis.jl | MIT (`LICENSE.txt`) | `1aa1bd5` | 2026-11-17 | `src/generate.jl`, `src/rand.jl` in full; `src/array.jl` beamformers; `src/dsp.jl` docstrings; `src/tfa.jl` docstrings |
| NoiseModels.jl | MIT | `2000052` | 2024-05-31 | `src/gaussian.jl` in full, README |
| AlphaStableDistributions.jl | MIT (Copyright Bagge Carlson, Too Yuen Min) | `790607d` | 2024-12-29 | README; structure and reference comments of the source |
| DataDrivenAcoustics.jl | MIT | `1eeb2d7` | 2026-05-21 | README, `src/core.jl` first 50 lines |
| VirtualAcousticOcean.jl | MIT | `5cb5cea` | 2026-10-17 | README, keyword search of `src/sim.jl` |
| AdaptiveEstimators.jl | MIT | `f362540` | 2025-01-11 | README |
| UnetUtils.jl | MIT | `a9ccc9b` | 2026-10-16 | README (signal-dump reader; no acoustics) |
| ARLToolkit.jl | MIT | `de0a0aa` | 2026-09-18 | file list only (AIS, GPX, CTD, KML, bathymetry readers) |
| unet-doc | MIT | `c185061` | 2026-10-02 | `handbook-ed1v3/models.adoc` channel-model section |
| unet-contrib | BSD 3-clause | `aa92d70` | 2026-04-22 | keyword search only: no acoustics formulas found |
| fjage, Fjage.jl, UnetSockets.jl, InteractiveViz.jl, ProtocolDataUnits.jl | BSD / MIT | cloned | - | not relevant (agent framework, sockets, plotting, packet codecs); names and README titles only |

Listed on the organisation page but not cloned (judged from name and one-line description
only): UnetViz.jl, UnetTrace, unet-ros2, unet-dune, unet-blog, unetsockets, unet-support,
fjage-sentuator, QuartzHDL.jl, DirectSockets.jl, SignalScope.jl, AcousticAnnotations.jl,
UWRoboticsSimulator (Unity and ROS underwater robot simulator, C#), r2c2, jhwbus, jajub,
dronesim, leaflet-tracksymbol2, birdwatch-public, ipaomg, EE2111A-2022, MCC_Linux_Drivers,
jupyter-ieee-paper, netiquette-dashboard, unetstack.net. UWRoboticsSimulator is the only one
of these that might hold a sonar model; it was not opened.

No separate documentation site repository exists: the Julia documentation is the `qdocs/`
(Quarto source) and `docs/` (rendered) folders of UnderwaterAcoustics.jl, and it also documents
AcousticsToolbox.jl and AcousticRayTracers.jl.

### 1.3 Could not be reached

- `unetstack.net` (the rendered UnetStack handbook) does not resolve from this session; the
  handbook source in `unet-doc` was read instead.
- Papers on `arl.nus.edu.sg` (Chitre 2007, Chitre 2023, the alpha-sub-Gaussian paper) were not
  opened; they are listed as references only.
- UnetStack itself is closed source; only its handbook was read.

## 2. Inventory and comparison

Paths are relative to each repository. "UA.jl" is UnderwaterAcoustics.jl. Verdicts: **same**,
**theirs better**, **ours better**, **missing here**, **not relevant**.

### 2.1 Medium

| Item | Where in their code | Their cited reference | What we have | Verdict and numbers |
|---|---|---|---|---|
| Absorption, Francois-Garrison | arlpy `uwa.py:38-74`; UA.jl `src/basic.jl:81-102`; again in the GPL port `AcousticsToolbox.jl/src/bellhopjl/attenuation.jl:61-77` | "Francois and Garrison (1982)" | `acoustics.absorption_francois_garrison` | **Same.** dB/km at 10 degC, 35 ppt, 0 m, pH 8: 4 kHz 0.24498 / 0.24498; 10 kHz 0.96264 / 0.96264; 100 kHz 33.63032 / 33.63032; 300 kHz 73.13931 / 73.13931; 750 kHz 222.39202 / 222.39202 (ours / theirs). `arlpy.uwa.absorption(100e3, 1.0, 10, 35, 0, 8.0)` gives 0.0336303 dB/m. One difference: they switch the pure-water coefficient at `temperature < 20`, we at `t <= 20`, so only exactly 20 degC differs: 101.959 against 101.950 dB/km at 300 kHz. Immaterial. Their defaults are 27 degC and pH 8.1; ours 10 degC and pH 8.0 |
| Absorption, other formulas | Thorp in the GPL port only (`attenuation.jl:40-43`, "updated formula from JKPS Eq. 1.34"): `3.3e-3 + 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2) + 3.0e-4 f^2` dB/km, f in kHz | Jensen, Kuperman, Porter, Schmidt, *Computational Ocean Acoustics* | Francois-Garrison and Ainslie-McColm in code; Thorp (SST's form, eq. 42) only in `PHYSICS_NOTES.md` 6.1 | **Ours better** (two modern fits). Thorp against our Francois-Garrison at 10 degC: 4 kHz 0.283 vs 0.245 dB/km (+15 %), 10 kHz 1.190 vs 0.963 (+24 %), 100 kHz 34.3 vs 33.6, 750 kHz 212.5 vs 222.4. They have no Ainslie-McColm, Fisher-Simmons or Schulkin-Marsh |
| Absorption as a filter | arlpy `uwa.py:76-99` `absorption_filter` (FIR from `firwin2`) | - | per-frequency absorption table in `sim/waveform.py` | **Same idea**; ours is applied in the frequency domain |
| Sound speed | arlpy `uwa.py:17-36`; UA.jl `basic.jl:22-35` | "Mackenzie (1981)" | `acoustics.sound_speed` (Mackenzie; Marczak below 5 PSU) | **Same** nine coefficients. (10, 35, 0) gives 1489.8034 m/s in both (arlpy called). We add a fresh-water branch; they have no Chen-Millero, Del Grosso or TEOS-10 either |
| Sound speed in bubbly water | arlpy `uwa.py:215-231`; UA.jl `basic.jl:30-33` (keyword `γ`) | "Wood (1964) or Buckingham (1997)" | none | **Missing here.** `c_mix = 1 / ( (1/c) sqrt( (γ (c/c_a)^2 m + (1-γ)/m) (γ/m + (1-γ) m) ) )`, `m = sqrt(ρ_water/ρ_gas)`. Hand values for c = 1500, c_a = 340, density ratio 1000: void fraction 1e-6 gives 1485.6 m/s, 1e-5 1372.4, 1e-4 873.9. Only matters inside the bubble layer or a wake |
| Sea water density | arlpy `uwa.py:101-121`; UA.jl `basic.jl:111-121` | "Fofonoff (1985 - IES 80)" | none (1000 or 1025 kg/m3 assumed where needed) | **Missing here.** One-atmosphere polynomial in T and S. Hand values: (27, 35) 1022.72; (10, 35) 1026.95; (10, 0) 999.70 kg/m3; `arlpy.uwa.density(10, 35)` returns 1026.952 |
| Bubble resonance | arlpy `uwa.py:170-188`; UA.jl `basic.jl:234-242` | "Medwin & Clay (1998)" | `sim/surface.resonant_radius` (the inverse) | **Same** (Minnaert, adiabatic, no surface tension). Theirs: 100 um at the surface resonates at 32462 Hz; ours gives 99.9 um for 32462 Hz (we use 1025 kg/m3 and 101325 Pa, they 1022.72 and 1.013e5) |
| Doppler | arlpy `uwa.py:151-168`; UA.jl `basic.jl:213-218`: `(1 + v/c) f` | none | exact per-ray factor from platform, scatterer and boundary motion (`reverb_bistatic.doppler_factor`, `timespread.py`) | **Ours better** |

### 2.2 Sea surface

| Item | Where in their code | Their cited reference | What we have | Verdict and numbers |
|---|---|---|---|---|
| Bubble loss per bounce | arlpy `uwa.py:190-213` ("APL model (1994)"); UA.jl `basic.jl:192-204` `surface_reflection_coef` | UA.jl: "APL-UW Technical Report 9407 II-21" | `sim/surface.bubble_loss_db` (Hall-Novarini population with resonant extinction, D-68) | **Theirs better founded** (see 2.2.1). Another agent is adopting it; not duplicated here |
| Surface reflection coefficient | UA.jl `basic.jl:203`: `complex(-db2amp(-a), 0)`, a the bubble loss in dB; `stdlib.jl:397-409` `WindySurface` | same | pressure release (-1) times the coherent factor times the bubble loss | **Same structure** for the bubble part |
| Coherent loss from roughness | none of their own. `FluidBoundary` and `ElasticBoundary` carry an rms roughness `σ`, and `stdlib.jl:99` and `:229` warn "Roughness is currently ignored in computing reflection coefficient". `σ` is only written into Kraken and Orca input files (`AcousticsToolbox.jl/src/common.jl:74,108,127`); the GPL Kraken port names a "Kuperman-Ingenito interfacial-roughness perturbation" (`krakenjl/modes.jl:92`, not read further) | - | `acoustics.rough_coherent_reflection` (exp(-Γ²/2), Γ = 2 k h sin g), `rms_wave_height` (Pierson-Moskowitz), time and frequency spreading of the incoherent part (`sim/timespread.py`) | **Ours better.** Nothing there confirms or replaces our Rayleigh-parameter form |
| Surface wave statistics from wind | none (no wave height, slope or spectrum law) | - | Pierson-Moskowitz height and vertical speed, Cox-Munk slope | **Ours better**; still unconfirmed against a source for SST's `McDanielSurface` |
| Sea state to wind speed | UA.jl `stdlib.jl:411-422` | "WMO sea states from APL-UW TR 9407 (1994), II-4 Table 2 (median windspeed)" | none (wind speed is a number) | **Missing here.** Transcribed in 2.2.2 |
| Surface backscattering strength | none | - | Chapman-Harris, the Kraus empirical law | **Ours better** (and still not confirmed against the handbook) |
| Surface altimetry (range-dependent surface shape) | arlpy `uwapm.py:108,141-147` (`surface` profile passed to Bellhop); UA.jl `altimetry` field | Bellhop | `scene.wave_surface` for scatterers; flat surface for the image and Bellhop providers | Not relevant now (a Bellhop input feature) |

#### 2.2.1 Bubble loss: the formula and the numbers

As printed in both packages (arlpy `uwa.py:207-213`, UA.jl `basic.jl:196-203`), with U the wind
speed in m/s ("measured 10 m above the sea surface" in the arlpy docstring), f in kHz and β the
grazing angle (their argument is the angle of incidence from the normal, β = π/2 - angle):

    a [dB] = 1.26e-3 / sin β · U^1.57 · f^0.85                         U >= 6 m/s
    a [dB] = 1.26e-3 / sin β · 6^1.57 · f^0.85 · exp(1.2 (U - 6))      U <  6 m/s

Neither file states a range of validity in frequency, wind speed or angle, and neither caps
the loss. The arlpy docstring examples (`-1.44` and `-117.6` dB for 3 and 10 m/s at 10 kHz,
angle 0) do not match the code: calling installed arlpy 1.9.3 gives 0.0041 and 0.331 dB. The
docstring is stale; trust the formula, not the examples.

Loss per bounce in dB at 10 degrees grazing, theirs / ours (ours with its warnings silenced;
0 means the resonant bubble is outside the Hall-Novarini radius range):

| U (m/s) | 4 kHz | 7 kHz | 10 kHz | 30 kHz | 100 kHz | 300 kHz | 750 kHz |
|---|---|---|---|---|---|---|---|
| 3 | 0.011 / 0.021 | 0.017 / 0.046 | 0.023 / 0.075 | 0.059 / 0.354 | 0.166 / 1.57 | 0.421 / 4.72 | 0.918 / 0 |
| 5 | 0.118 / 0.096 | 0.190 / 0.211 | 0.258 / 0.348 | 0.656 / 1.64 | 1.83 / 7.29 | 4.64 / 21.9 | 10.1 / 0 |
| 10 | 0.876 / 1.17 | 1.41 / 2.60 | 1.91 / 4.35 | 4.86 / 21.6 | 13.5 / 102 | 34.4 / 305 | 74.9 / 0 |
| 15 | 1.66 / 5.63 | 2.66 / 12.9 | 3.61 / 21.9 | 9.18 / 119 | 25.5 / 651 | 65.0 / 1952 | 142 / 0 |

Reading: in the flank-array band (4 to 10 kHz) the two agree within a factor 2.3 at 5 and
10 m/s, both are below 0.1 dB at 3 m/s (where ours is 2 to 3 times theirs), and they diverge
by a factor 3 to 6 at 15 m/s (our cubic wind law against their U^1.57). In the
FLS/SAS band ours is unusable (hundreds of dB, then zero at 750 kHz); theirs still says a
surface bounce at 10 m/s and 10 degrees costs 13 to 75 dB, so the practical conclusion for
100 to 750 kHz is the same with either law: surface-bounce paths vanish in any real wind.

#### 2.2.2 Sea states (transcribed from UA.jl `src/stdlib.jl:413-422`)

| Sea state | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Wind speed, m/s | 0.8 | 2.6 | 4.4 | 6.9 | 9.8 | 12.6 | 19.3 | 26.5 | 30.6 | 32.9 |

Their stated source: "WMO sea states, from APL-UW TR 9407 (1994), II-4 Table 2 (median
windspeed)". For orientation with our own laws at 7 kHz and 5 degrees grazing: sea state 3
(6.9 m/s) gives rms height 0.255 m, coherent amplitude 0.43 and 1.57 dB of bubble loss by
their law; sea state 4 (9.8 m/s) gives 0.514 m, 0.032 and 2.72 dB.

### 2.3 Seabed

| Item | Where in their code | Their cited reference | What we have | Verdict and numbers |
|---|---|---|---|---|
| Fluid half-space reflection | arlpy `uwa.py:123-149`; UA.jl `basic.jl:135-142` | "Brekhovskikh & Lysanov"; loss parameter "based on APL-UW Technical Report 9407" | `propagation.rayleigh_reflection`, `bottom.reflection` | **Same.** For their MediumSand against water (1023, 1528): \|R\| at grazing 5, 20, 33, 45, 80 degrees is 0.97619, 0.93711, 0.74000, 0.47055, 0.37249 by both. The phases are conjugate (theirs +145.8, ours -145.8 degrees at 5 degrees): UA.jl uses `n = (1 - iδ)/c_r` with the comment "negative sign needed for agreement with Bellhop", we follow SST's exp(+iωt). Note arlpy's Python version uses `(1 + 1j*alpha)` (`uwa.py:144`), the opposite sign to its Julia successor |
| Loss parameter convention | UA.jl `basic.jl:171,179`: `δ = (dB/λ) / (40π / ln 10)`; `utils.jl:232-235`: dB/m/kHz times c/1000 gives dB/λ | "APL-UW TR 9407 (1994), IV-9 equation (4)" | the same conversion in `propagation.rayleigh_reflection` and `bottom.parameters_from_physical` | **Same.** This confirms the definition of δ we took from general knowledge |
| Named sediments | UA.jl `stdlib.jl:113-136` (22 rows) | "seabeds from APL-UW TR 9407 (1994), IV-6 Table 2; ratios evaluated wrt reference ρ=1023, c=1528" | 2 confirmed rows, 1 placeholder (`configs/sediments.yaml`) | **Theirs better, and the most valuable find.** See 2.3.1 |
| Elastic half-space | UA.jl `stdlib.jl:192-216` `ElasticBoundary`; `basic.jl:157-163` | - | none | **Missing in both.** Their own code says "WARNING: Fluid-solid reflection not implemented. Currently ignores shear." The type exists so that Kraken, Orca and Bellhop can be given shear parameters |
| Shear speed from compressional speed | UA.jl `basic.jl:43-54` | "regression relationship from Hamilton (1980)" | none | **Missing here**; only useful with an elastic bottom. Piecewise: `3.884 c - 5757` (1512 to 1555 m/s), `1.137 c - 1485` (to 1650), `991 - 1.136 c + 0.00047 c²` (to 2150), `0.78 c - 962` above; error below 1512. Hand values: fine sand 414 m/s, medium sand 469 m/s, rock 2018 m/s |
| Shear attenuation for named seabeds | UA.jl `stdlib.jl:236-252` | "shear absorption based on values recommended in Hamilton (1980)" | none | **Missing here.** 0.07 dB/m/kHz (rock, pebbles: "basalt"), 13.2 (sands and sandy gravel: "diluvial sand"), 4.8 (muddy types: "diluvial sand & clay"), 13.4 (coarse and sandy silt: "alluvial silt") |
| Layered seabed | UA.jl `stdlib.jl:331-389` `MultilayerElasticBoundary` | - | none | **Missing in both** as a formula: "Multilayer boundary reflection not implemented, using top layer only" (`stdlib.jl:382-385`). Layers are passed to Kraken and Orca |
| Tabulated reflection coefficient | GPL port only (`bellhopjl/reflection.jl`, Bellhop option 'F') | Bellhop | `Boundary` tables in the reverberation code | Not relevant |
| Bottom backscatter, roughness spectrum, sediment volume scattering | none | - | `sim/bottom.py` (perturbation, Kirchhoff, small slope, composite), SEARAY law | **Ours better.** They give nothing to confirm `w2`, `gamma2`, `sigma2` |

#### 2.3.1 The seabed table

Transcribed exactly from UA.jl `src/stdlib.jl:115-136` (also printed in `qdocs/api.qmd:86-107`):
`FluidBoundary(ρ kg/m3, c m/s, δ)`. The last three columns are computed here: ρ/1023, c/1528,
and δ·40π/ln 10 in dB per wavelength.

| Name | ρ | c | δ | ρ ratio | ν | dB/λ |
|---|---|---|---|---|---|---|
| Rock | 2557.50 | 3820.00 | 0.01374 | 2.5000 | 2.5000 | 0.750 |
| Pebbles | 2557.50 | 2750.40 | 0.01374 | 2.5000 | 1.8000 | 0.750 |
| SandyGravel | 2549.32 | 2073.50 | 0.01705 | 2.4920 | 1.3570 | 0.931 |
| VeryCoarseSand | 2455.20 | 1996.64 | 0.01667 | 2.4000 | 1.3067 | 0.910 |
| MuddySandyGravel | 2367.22 | 1952.48 | 0.0163 | 2.3140 | 1.2778 | 0.890 |
| CoarseSand | 2282.31 | 1910.46 | 0.01638 | 2.2310 | 1.2503 | 0.894 |
| GravellyMuddySand | 2200.47 | 1870.42 | 0.01645 | 2.1510 | 1.2241 | 0.898 |
| MediumSand | 1887.44 | 1800.29 | 0.01624 | 1.8450 | 1.1782 | 0.886 |
| MuddyGravel | 1652.15 | 1741.31 | 0.0161 | 1.6150 | 1.1396 | 0.879 |
| FineSand | 1484.37 | 1691.95 | 0.01602 | 1.4510 | 1.1073 | 0.874 |
| MuddySand | 1369.80 | 1650.24 | 0.01728 | 1.3390 | 1.0800 | 0.943 |
| VeryFineSand | 1297.16 | 1614.79 | 0.01875 | 1.2680 | 1.0568 | 1.023 |
| ClayeySand | 1252.15 | 1583.62 | 0.02019 | 1.2240 | 1.0364 | 1.102 |
| CoarseSilt | 1222.49 | 1555.35 | 0.02158 | 1.1950 | 1.0179 | 1.178 |
| SandySilt | 1195.89 | 1527.85 | 0.01261 | 1.1690 | 0.9999 | 0.688 |
| MediumSilt | 1175.43 | 1510.43 | 0.00676 | 1.1490 | 0.9885 | 0.369 |
| SandyMud | 1175.43 | 1508.59 | 0.00386 | 1.1490 | 0.9873 | 0.211 |
| FineSilt | 1174.40 | 1506.76 | 0.00306 | 1.1480 | 0.9861 | 0.167 |
| SandyClay | 1173.38 | 1504.93 | 0.00242 | 1.1470 | 0.9849 | 0.132 |
| VeryFineSilt | 1173.38 | 1503.09 | 0.00194 | 1.1470 | 0.9837 | 0.106 |
| SiltyClay | 1172.36 | 1501.11 | 0.00163 | 1.1460 | 0.9824 | 0.089 |
| Clay | 1171.34 | 1497.44 | 0.00148 | 1.1450 | 0.9800 | 0.081 |

What this confirms. The ratios come out as round three or four digit numbers, which shows the
file stores ratio times (1023, 1528). Against our two confirmed rows (read from Brown, Johnson
and Brownstead, arXiv:1809.08459, Table 1, who also attribute them to TR 9407):

| Sediment | Ours ρ, ν, δ | Theirs ρ, ν, δ | Difference |
|---|---|---|---|
| medium sand | 1.845, 1.178, 0.016189 | 1.8450, 1.1782, 0.01624 | ρ and ν identical to the printed digits; δ 0.3 % (ours is derived from "10 dB/m at 20 kHz", theirs is printed) |
| very fine silt | 1.147, 0.984, 0.001893 | 1.1470, 0.9837, 0.00194 | ρ and ν identical; δ 2.4 % (ours from "1.4 dB/m", a two-digit number) |

Two independent open sources now give the same density and speed ratios for these two rows,
both attributed to the handbook. That raises confidence in the other 20 rows of this table.
The arXiv paper's assumption about water (we took 1000 kg/m3 and 1500 m/s to form ratios)
is also resolved: the ratios are what the source tabulates, and they are independent of the
reference pair.

What it does not give. Roughness spectral strength, spectral exponent and volume parameter
are absent, so the 20 new rows can drive the Rayleigh forward reflection loss (image-source
`bottom:`, Bellhop bottom half-space, `bottom.reflection`) but not `bottom.strength`.

One inconsistency inside their repository. The table constant gives SandyGravel c = 2073.50,
that is ν = 1.3570. Their tests use ν = 1.3370 with the same density ratio and δ
(`test/test_pekeris.jl:229` `FluidBoundary(2.492 * 1023, 1.3370 * 1528, 0.01705)`;
`test/test_basic.jl:57` `reflection_coef(0.0, 2.492, 1.3370, 0.01705)`). One of the two is a
typing error; both values keep the column monotonic. The SandyGravel sound speed should be
marked uncertain (1.337 or 1.357) until a second source is found.

### 2.4 Propagation models

| Item | Where in their code | Their cited reference | What we have | Verdict |
|---|---|---|---|---|
| Isovelocity image (Pekeris ray) model | UA.jl `src/pekeris.jl:17-157` `PekerisRayTracer`; amplitude `1/D` times absorption at mid depth times `R_surface^s R_bottom^b` (`:133-135`) | "based on Chitre (2007)" (JASA 122, 2580-2586) | `propagation.ImageSourceProvider` (same series, plus walls, angle-dependent bottom, time and frequency spreading, per-ray Doppler, C++/CUDA engine) | **Same** physics; ours has more. Their default is 3 bounces |
| Arrivals to impulse response | UA.jl `pekeris.jl:108-120`: sinc times a Blackman window 2 ms wide at each fractional delay; arlpy `uwapm.py:416` | - | Hann or Kaiser windowed-sinc `var_delay` with measured accuracy (`PHYSICS_NOTES.md` 15) | **Same** idea; ours is characterised |
| Pekeris normal modes | UA.jl `pekeris.jl:181-468` `PekerisModeSolver`: characteristic equation `ρ_b γ cos γh + ρ sqrt(k1² - k2² - γ²) sin γh = 0`, analytic group velocity, modal sum with `sqrt(2π/R)` | Jensen et al. (COA) | none | **Not relevant** at our frequencies: the number of propagating modes is about 2 f h / c, roughly 900 at 7 kHz in 100 m and far more at 100 kHz. Their own limits: isovelocity, no bottom absorption, pressure-release surface, no leaky modes |
| Adiabatic modes (range-dependent depth) | UA.jl `src/adiabatic.jl` `AdiabaticExt` | - | none | **Not relevant** (same reason) |
| Gaussian beam tracer, native | AcousticRayTracers.jl `src/RaySolver.jl` (MIT): 2-D, sound-speed profile, range-dependent bathymetry, ODE solver, dynamic ray tracing, beam sum "[COA (3.72), (3.75)]" (`:205`), eigenray search grouped by bounce signature, reflection off scatterers, backscatter tracing | COA 2nd ed. ch. 3 | Bellhop as an external process (`sim/bellhop.py`) | **Missing here as in-house code.** A permissively licensed design to study if we ever want a tracer inside the package rather than a GPL executable beside it. Julia-only; it would be a re-implementation |
| Bellhop, Kraken, Orca front ends | arlpy `uwapm.py` (Bellhop, with surface altimetry, bottom roughness and transmit directivity inputs); AcousticsToolbox.jl `bellhop.jl`, `kraken.jl`, `orca.jl` (MIT wrappers writing input files for GPL binaries) | OALIB | `sim/bellhop.py` (same pattern: external process, nothing linked) | **Same** approach. They also drive Kraken and Orca, which we do not need |
| Native ports of Bellhop and Kraken | AcousticsToolbox.jl `src/bellhopjl/`, `src/krakenjl/` | derivative of GPL Bellhop / Kraken | - | **Not usable**: GPL-3.0-or-later |
| Differentiable models (ForwardDiff through all of the above) | UA.jl, RaySolver, BellhopJL; Chitre, "Differentiable Ocean Acoustic Propagation Modeling", OCEANS 2023 | - | none | Not relevant to a forward simulator; of interest only for geoacoustic inversion |
| Data-aided models | DataDrivenAcoustics.jl (neural network mapping position and wavenumber to complex field; "Data-Aided Underwater Acoustic Ray Propagation Modeling", IEEE JOE) | - | none | **Not relevant** |
| 2-D to 3-D reframing | UA.jl `src/reframe.jl` `Reframe2D` | - | our providers are 3-D in geometry already | Not relevant |
| Objects in the water column | UA.jl `src/scatterers.jl`: shapes (2-D `Ellipse`, parametric curve or surface) with a boundary condition, applied "under a local tangent-plane (Kirchhoff) approximation" (`:534`); only RaySolver uses them; the docs call it "2½D" (a 2-D shape is an object extended across the track) | - | `sim/targets.py`: 3-D sphere, cylinder, plate, disc glints, highlight targets, aspect tables | **Ours better** for sonar targets |
| Transmission loss helpers | UA.jl `api.jl:147-170` (`transmission_loss`, `acoustic_field`, coherent or incoherent sum); VirtualAcousticOcean `sim.jl:313` `D^-α` times absorption | - | `sonar_equation.SonarEquation` | **Same** |

### 2.5 Noise

| Item | Where in their code | Their cited reference | What we have | Verdict |
|---|---|---|---|---|
| Ambient spectra (wind, shipping, turbulence, thermal) | none anywhere (searched every cloned tree for Wenz, Knudsen: no hit) | - | `sim/noise.wenz_spectrum_db` (four-term fit) | **Ours better.** The UnetStack handbook's channel has a single noise level parameter (default 60 dB) |
| White, red, pink Gaussian | UA.jl `stdlib.jl:434-522`; SignalAnalysis.jl `src/rand.jl:59-83` (red: cumulative sum; pink: a 3-pole filter "based on" a CCRMA page) | pink in UA.jl: "as in the underwater acoustic channel repository (UACR) ... falls by 17 dB per decade" | Gaussian noise with any spectrum function (`noise.gaussian_noise`) | **Ours better** (arbitrary spectrum). Their fixed 17 dB/decade slope is a generic stand-in |
| Spatial coherence across an array | measured: NoiseModels.jl `gaussian.jl` fits mixing coefficients `α[i,j,lag]` to data so that `Σ α_i α_j` matches the measured covariance at lags up to 64 samples, then generates by filtering white noise; UA.jl `ReplayNoise` (`replay.jl:235-297`) plays UACR mixing coefficients `β[rx, rx, lag]` | UACR file specification | modelled: isotropic `sin(kd)/(kd)` coherence by a Cholesky factor per band | **Different thing.** Theirs reproduces a recorded field; ours a model field. **Missing here**: a noise generator driven by a recording |
| Impulsive (non-Gaussian) ambient noise | arlpy `stable.py:66-100` (`rnd`: alpha-stable with skew), `:36-64` (`sstabfit`); UA.jl `replay.jl:299-308` (symmetric alpha-stable innovations, scale 1/sqrt 2 so that α = 2 is the standard normal); AlphaStableDistributions.jl (alpha-stable, and "alpha sub-Gaussian with memory" for bursty noise) | "Chambers, Mallows & Stuck, JASA 71 (1976) 340-4"; fit by "McCulloch (1986)" and "Fama & Roll (1971)" fractiles; alpha-sub-Gaussian: an ARL paper linked from the README (file name `SigProc2016RandomVariate.pdf`, not opened) | K-distributed texture for reverberation only (`noise.k_texture`); ambient noise is Gaussian | **Missing here.** Symmetric case as printed in `replay.jl:303-305`: `x = sin(αu) / cos(u)^(1/α) · (cos(u - αu) / w)^((1-α)/α) / sqrt 2`, u uniform on (-π/2, π/2), w unit exponential |
| Impulse detection | arlpy `signal.py:441-465` (`detect_impulses`: envelope peaks above median + k times the median absolute deviation) | - | none | Missing here; small |
| Self noise, tonals | none | - | none for the platform. A scenario gets tonal lines only from a one-way radiating source (`sources: [{signal: {type: tonals}}]`, `sim/sources.py`) or from a fitted noise model's `include_tones`. The two library functions that were here (`sim/noise.self_noise_spectrum_db`, `add_tones`) were reachable from no scenario and were removed on 2026-11-20 (API audit G4) | **Missing here.** Platform self-noise cannot be configured in a scenario. (Corrections 2026-11-20: this row said "Ours better", then "Library only") |

### 2.6 Time variability and measured channels

| Item | Where in their code | Their cited reference | What we have | Verdict |
|---|---|---|---|---|
| Replay of measured time-varying impulse responses | UA.jl `src/replay.jl:10-233` `BasebandReplayChannel`: baseband, resample to the delay-axis rate, convolve with an impulse response that changes every sample (cubic spline between stored snapshots), apply the measured phase track `theta_hat` or delay track `phi_hat`, resample by a Doppler factor per receiver, return to passband | "popularized ... by the Watermark benchmark"; UACR (`uwa-channels`) file format | none; our time variability is modelled (surface spreading, Doppler) | **Missing here.** Useful as an independent check of `sim/timespread.py` against sea data in the bands the library covers (the documented example is 25 kHz ± 4.8 kHz, 3 hydrophones, 47.8 s) |
| Real-time network simulator | VirtualAcousticOcean.jl (streams samples to modems or software-defined modems over a network protocol) | - | - | **Not relevant** |
| Statistical link model | unet-doc `handbook-ed1v3/models.adoc:137-176`: "sound speed c [Mackenzie, JASA, 1981], transmission loss TL [Urick 3rd ed, p105-111]", `SNR = SL - TL - NL`, detection with `SNR + 10 log(Bt)`, Rician or Rayleigh fading (K = 10 by default) | Urick | sonar equation tool | **Not relevant** (communications link budget; nothing new) |

### 2.7 Signals, detection, array processing

| Item | Where in their code | What we have | Verdict |
|---|---|---|---|
| Linear and hyperbolic FM | SignalAnalysis.jl `generate.jl:55-76`: hyperbolic phase `-2π s f1 ln\|1 - t/s\|`, `s = f2 d/(f2 - f1)`; arlpy `signal.py:64` (`scipy.signal.chirp`: linear, quadratic, logarithmic, hyperbolic) | `waveform.py` lfm, hfm (`ln(1 + k f0 u)/k`), stepped FM | **Same.** For 4 to 10 kHz over 0.1 s the two hyperbolic phases differ by 2e-13 cycles at most. They also offer quadratic and logarithmic sweeps through SciPy |
| Codes | m-sequences (`dsp.jl:228`, arlpy `signal.py:96`), generalised m-sequence (`dsp.jl:295`, phase `θ = atan(sqrt(2^m - 1))`), Walsh-Hadamard (`dsp.jl:306`) | Barker, m-sequence, arbitrary chip list | **Same** for m-sequences. Generalised m-sequence and Hadamard **missing here**, low value for sonar |
| Matched filter, base-banding, resampling, fractional delay | `dsp.jl` `mfilter`, `upconvert`, `downconvert`, `delay!` (fractional and negative delays), arlpy `signal.py` `bb2pb`, `pb2bb` | all present (C++ and NumPy) | **Same** |
| Beamformers | SignalAnalysis.jl `array.jl:70-205`: plane-wave steering delays, time-domain delay-and-sum, narrowband Bartlett, Capon (with diagonal loading γ), MUSIC; arlpy `bf.py`: the same plus a broadband wrapper over an STFT (`:330`) and a Bartlett beam pattern | delay-and-sum, back-projection, sub-band MVDR with loading (`flank.py`), MUSIC (`libs/live`), pattern and ambiguity prediction | **Same or ours better.** Their code has "TODO: broadband beamformer" and "TODO: add support for shading" |
| DEMON | `dsp.jl:76-107`: envelope by RMS, mean or FIR decimation, then a high-pass | none | **Missing here**; relevant to the passive use of the flank array (propeller modulation lines) |
| Tone detection and tracking | Goertzel (`dsp.jl:356`, arlpy `signal.py:398`), phase-locked loop (`dsp.jl:398`) | none | Missing here; small |
| Arrival estimation | `findsignal` (`dsp.jl:759`: n strongest copies of a reference, optional sub-sample refinement by optimisation), `decompose` (`dsp.jl:861`: greedy decomposition into delayed copies) | none as a tool (`simcal.py` does its own fitting) | Missing here; could serve multipath extraction for simulator calibration |
| Time-frequency | spectrogram and Wigner-Ville (`tfa.jl`), inverse STFT, spectral whitening ("based on Lee 1986", `dsp.jl:597`), discrete Zak transform (`dsp.jl:929`) | spectrogram-level tools, TVG, whitening in the processing chain | Wigner-Ville and Zak **not relevant** |
| Adaptive filters | AdaptiveEstimators.jl: LMS, NLMS, RLS, decision-feedback equaliser | adaptive beamforming only | **Not relevant** (communications) |
| Modulation | arlpy `comms.py` (PSK, QAM, FSK, pulse shaping, BER) | - | **Not relevant** |

### 2.8 Sources, receivers, units and conventions

| Item | Theirs | Ours |
|---|---|---|
| Source | point source: position, one frequency, source level (`AcousticSource(pos, frequency; spl)`, `api.jl:470-498`); arlpy passes a transmit directivity table to Bellhop (`uwapm.py:111`) | arrays with element patterns, sampled waveforms, calibrated source level |
| Receiver | points and regular grids (`AcousticReceiverGrid2D/3D`); `arlpy.uwa.pressure` and `spl` convert volts to µPa with sensitivity and gain (`uwa.py:233-280`) | hydrophone arrays with sensitivity, gain, ADC, channel errors |
| Angles | boundary functions take the angle of **incidence from the normal**; arrivals report launch and arrival angles | **grazing** angles throughout |
| Axes | x, y horizontal, z up and negative downward | z up in scenes (see `PHYSICS_NOTES.md` 1.4) |
| Units | SI, dB re 1 µPa, salinity in ppt, loss as dimensionless δ or dB/m/kHz via `Unitful` | SI, dB re 1 µPa, loss in dB per wavelength or δ |
| Time convention | reflection phase chosen "for agreement with Bellhop"; mode fields conjugated "to match with Kraken" | exp(+iωt) as in SST; our reflection phase is the conjugate of theirs |
| Environment defaults | 27 degC, 35 ppt, pH 8.1, 100 m deep, rigid bottom, pressure-release surface (`api.jl:359-384`) | 10 degC, 35 PSU, pH 8.0 |

## 3. Recommended adoptions, ranked

Status (2026-11-17): items 1, 3, 5, 9, 10 and 11 are adopted in commit `cadc7ca`; see
section 8 for what each became. Items 2 and 4 are with other work packages; 6 to 8 wait for
the owner.

Effort is for implementation with tests in this repository. All are re-implementations from
the stated formula or table with attribution, not code copies.

1. **Seabed table: 20 more named sediments for reflection loss, and a cross-check of our two
   rows.** Source: UA.jl `src/stdlib.jl:113-136` (MIT), attributed there to TR 9407 IV-6
   Table 2. Add the rows to `configs/sediments.yaml` as `rho`, `nu`, `delta` with a new status
   (for example `reflection_only`: three of six parameters confirmed, roughness and volume
   absent), let `bottom.sediment()` return such rows and refuse them for `bottom.strength`,
   and let the image-source and Bellhop `bottom:` options take a sediment name. Add a test
   that our two arXiv rows agree with this table (ρ and ν exactly, δ within 3 %). Mark
   SandyGravel's speed uncertain (2.3.1). Touches `configs/sediments.yaml`, `sim/bottom.py`,
   `sim/propagation.py`, `sim/bellhop.py`, `docs/PHYSICS_NOTES.md` 13, `docs/DECISIONS.md`.
   Effort: half a day. Value: high; it removes "only two named sediments" for everything
   except backscatter and gives rock and gravel for the first time.
2. **Bubble loss law (in progress elsewhere).** What this survey adds to that work: the Julia
   package gives the section ("TR 9407 II-21") for the same formula; the angle argument is
   incidence from the normal; the arlpy docstring examples are wrong; no validity limits are
   printed, so ours must be chosen and documented (2.2.1 gives the comparison table to quote).
   Touches `sim/surface.py`. No further effort here.
3. **Sea-state table.** Accept `sea_state: 0..9` wherever a wind speed is accepted, mapped by
   the table in 2.2.2 with its attribution. Touches `sim/timespread.boundary_parameters`,
   scenario parsing in `sim/simulator.py`, `docs/CONFIGURATION.md`. Effort: one to two hours.
4. **Impulsive ambient noise.** A symmetric alpha-stable option for `sim/noise.py` (white
   innovations, then our existing spectral shaping in the time domain, because frequency
   domain synthesis makes the result Gaussian again, as `PHYSICS_NOTES.md` 7 notes for K
   noise), with the Chambers-Mallows-Stuck generator and a fractile fit as the test oracle
   (`arlpy.stable.sstabfit`, BSD, can be the oracle in a study tool as arlpy already is).
   Touches `sim/noise.py`, `sim/simulator.py` noise configuration. Effort: one day. Value:
   medium; snapping-shrimp noise in warm shallow water is the usual reason this matters
   [general knowledge, not read here]. The alpha-sub-Gaussian model with memory
   (AlphaStableDistributions.jl) is the next step if bursts must be correlated in time.
5. **Sea water density.** `acoustics.water_density(T, S)` from the one-atmosphere polynomial
   (arlpy `uwa.py:101-121`, "Fofonoff (1985 - IES 80)"), tested against the three values in
   2.1. Use it wherever a density ratio is formed from kg/m3 (`bottom.parameters_from_physical`
   default, Bellhop environment). Effort: one to two hours. Value: small but removes an
   assumed constant.
6. **DEMON and a Goertzel tone detector for the passive flank chain.** Re-implement from the
   description (envelope, decimate, high-pass, spectrum). Touches `gpusonar/detect.py` or
   `flank.py`, `sim/passive.py` for a modulated test source. Effort: one day. Value: medium
   if passive classification is in scope.
7. **Noise from a recording.** A generator that reproduces the lagged cross-covariance of a
   measured multichannel noise file (the mixing-coefficient form of NoiseModels.jl and the
   UACR noise files). Touches `sim/noise.py`. Effort: one to two days. Value: medium, only
   once sea recordings exist.
8. **Channel replay as a validation tool.** Read UACR files and pass our transmit waveforms
   through measured time-varying responses, then compare delay and Doppler spread with
   `sim/timespread.py`. A study tool under `tools/`, not part of the simulator. Effort: two
   to three days. Value: medium to low (bands are those of acoustic modems).
9. **Thorp absorption as a named model**, for like-for-like comparisons with SST and Bellhop
   runs. Take the formula from Jensen et al. eq. 1.34 (it was seen here only inside a GPL
   file, so cite and check against the book). Touches `acoustics.py`. Effort: under an hour.
10. **Elastic seabed parameters, recorded as data only.** Keep the shear-speed regression and
    the four shear attenuation values (2.3) in `docs/` until an elastic reflection coefficient
    exists here; org-arl has none to adopt.
11. **Wood's equation** for sound speed in bubbly water. Only if a bubble-layer refraction or
    wake model is ever wanted. Effort: one hour.

## 4. Not worth adopting, and why

- **Normal-mode solvers (Pekeris, Kraken, Orca, adiabatic).** Hundreds to thousands of modes at
  4 kHz and above; rays are the right tool for both of our bands.
- **BellhopJL and KrakenJL.** GPL-3.0-or-later derivatives; also Julia.
- **Differentiable and data-aided propagation.** For inversion and model fitting, not for
  generating sonar time series.
- **Their Pekeris ray model, Doppler, reflection coefficient, absorption, sound speed, FM
  pulses, m-sequences, beamformers.** We already have the same or more.
- **Their surface model as a whole.** It is the bubble law and nothing else: no roughness, no
  coherent loss, no scattering strength, no spreading.
- **Their scatterer objects.** Two-dimensional shapes for a 2-D ray tracer; our glint targets
  are three-dimensional.
- **Red and pink noise generators.** Special cases of our arbitrary-spectrum generator.
- **Communications tooling** (AdaptiveEstimators.jl, `arlpy.comms`, UnetStack channel models,
  VirtualAcousticOcean.jl, fjage, Unet*). Outside a sonar suite.
- **Wigner-Ville, Zak transform, generalised m-sequences, Hadamard codes.** No use identified.
- **Plotting, data readers** (`arlpy.plot`, `romanis`, `dtla`, ARLToolkit.jl, InteractiveViz.jl).

## 5. Licence notes

| Source | Licence | What may be done |
|---|---|---|
| arlpy | BSD 3-clause | Formulas and constants may be re-implemented with attribution ("as implemented in arlpy, which cites ..."). If code were ever copied, the copyright notice and licence text must be retained; this survey copies none. Already used as an optional oracle in `tools/boundary_models_study.py` |
| UnderwaterAcoustics.jl, AcousticRayTracers.jl, SignalAnalysis.jl, NoiseModels.jl, AlphaStableDistributions.jl, DataDrivenAcoustics.jl, VirtualAcousticOcean.jl, AdaptiveEstimators.jl, unet-doc | MIT | Same: re-implement with attribution; table values may be reproduced with the attribution the file gives |
| AcousticsToolbox.jl outside `src/bellhopjl/` and `src/krakenjl/` | MIT | Wrapper ideas (input-file writing) may be compared with `sim/bellhop.py` |
| AcousticsToolbox.jl `src/bellhopjl/`, `src/krakenjl/`; Bellhop, Kraken (OALIB); A-New-BellHope | GPL-3.0-or-later | Nothing into the core: no code, no translated code. Running the executable as a separate process (as `sim/bellhop.py` does) is unaffected. Formulas that are published elsewhere (Thorp, Francois-Garrison) are taken from the publication, not from these files |
| Values attributed to APL-UW TR 9407 | printed in MIT/BSD files | Used as "UnderwaterAcoustics.jl (MIT), `src/stdlib.jl`, which attributes them to TR 9407 section ..." The handbook itself stays unread. `status:` in `configs/sediments.yaml` should say the row was read from that package, as the existing rows say they were read from the arXiv paper |

This is a practical reading of the licences, not legal advice. Whatever the legal position
of individual numbers, the attribution chain above is what lets a reader trace each value,
so keep it in the YAML and in `PHYSICS_NOTES.md`.

## 6. Leads one level out (other authors; not surveyed in full)

| Lead | Licence | How far it was checked | Why it may matter |
|---|---|---|---|
| Underwater Acoustic Channel Repository toolboxes, `github.com/uwa-channels` (Python `uwa_channels`: `replay`, `noisegen`, `unpack`; MATLAB twin; documentation site) | MIT (LICENSE read in the Python, MATLAB and site repositories; commits `c392aea`, `2fc3460`, `b09eb57`; the Python README names the repository `uwa-channels/python`) | README, citation file, the site's noise and user-guide pages; code not read | Measured time-varying channels with site-specific noise models (coloured Gaussian; impulsive alpha-sub-Gaussian for one site), data on Zenodo. Python, so directly usable in a study tool (recommendations 7 and 8) |
| Watermark benchmark (FFI) | not checked | named in UA.jl `qdocs/replay.qmd:15` only | The original replay channel set |
| `ahmd-mahm/alpha-SGNm` (MATLAB) | BSD 3-clause (LICENSE read, commit `8b722f9`) | file list only | Reference implementation of alpha-sub-Gaussian noise with memory, cited by AlphaStableDistributions.jl |
| OALIB Acoustics Toolbox (Bellhop, Kraken, `misc/AttenMod.f90`, `Kraken/Scattering.f90`) | GPL | seen only through the Julia port's comments | Kuperman-Ingenito roughness loss for modes; the volume attenuation options of the toolbox. Read the cited papers, not the code |
| A-New-BellHope/bellhop | GPL-3 (per the AcousticsToolbox README) | not opened | Documented reproducibility fixes to Bellhop's boundary stepping (`bellhopjl/PORTING_NOTES.md` section 1). Relevant only as a choice of which Bellhop binary our external provider runs |
| ORCA normal-mode model | not checked | named in `orca.jl` only | Elastic layered bottoms in a mode model |
| DSP.jl (JuliaDSP), SciML solvers, Unitful, Lux | MIT | not opened | General libraries; no acoustics formulations |

References their files cite that could confirm things for us if a copy can be read through a
permitted route: Chitre, JASA 122 (2007) 2580-2586 (warm shallow water channel model, the basis
of their image model; PDF linked from `qdocs/pekeris-rays.qmd`); Hamilton (1980) (shear speed
regression and shear attenuation); Jensen et al., *Computational Ocean Acoustics* (2011)
(Thorp eq. 1.34, Gaussian beams eqs. 3.65, 3.72, 3.75); Medwin and Clay (1998); Fofonoff (1985);
Wood (1964) and Buckingham (1997); Brekhovskikh and Lysanov; Chambers, Mallows and Stuck
(1976); McCulloch (1986); Fama and Roll (1971); Lee (1986) (spectral whitening).

## 7. What this survey did not find

Stated so that nobody searches these repositories for them again: no surface or bottom
backscattering strength law; no Lambert, Chapman-Harris or handbook scattering model; no
reverberation; no target strength formulas; no rough-surface coherent reflection coefficient;
no wave-height or slope law from wind; no Wenz or Knudsen noise spectrum; no noise coherence
model (Cron-Sherman or isotropic); no fluid-solid or layered reflection coefficient of their
own; no sound-speed equation other than Mackenzie; no absorption formula other than
Francois-Garrison (plus Thorp inside the GPL port); no roughness spectrum or sediment volume
parameters for the named seabeds; no stated validity range for the bubble law.

## 8. Adoption record

Commit `cadc7ca`. Everything is opt-in; scenarios that do not use a new key give the records
they gave before.

| Item | What was adopted | Where | Source as cited in the code |
|---|---|---|---|
| 1 | 20 reflection-only seabeds (density, sound speed, loss parameter with the source's reference water 1023 kg/m3, 1528 m/s; the loader forms ratios). By name for bottom reflection: image-source `bottom: {sediment}`, Bellhop `environment: {sediment}`, `bottom.sediment_reflection`, `bottom.fluid_halfspace`. The scattering models refuse them unless the item supplies `w2`, `gamma2`, `sigma2`. `sandy_gravel` carries the table's speed with `unconfirmed_fields: [nu]` and a note giving both values. The two complete rows keep their arXiv numbers; their stated assumption (ratios formed with 1000 kg/m3 and 1500 m/s) is now recorded as supported by this table | `configs/sediments.yaml`, `sim/bottom.py`, `sim/propagation.py`, `sim/bellhop.py`, `sim/targets.py`, one argument in `sim/simulator.py`, `tools/boundary_models_study.py`, `tests/python/test_sediment_table.py` | UnderwaterAcoustics.jl (MIT), `src/stdlib.jl:113-136`, which attributes the table to APL-UW TR 9407 IV-6 Table 2 |
| 3 | `acoustics.SEA_STATE_WIND_SPEED`, `sea_state_wind_speed`, `resolve_sea_state`; `run_scenario` rewrites `sea_state` to `wind_speed` in every mapping of the scenario before anything reads it | `acoustics.py`, one line in `sim/simulator.py` | UnderwaterAcoustics.jl `src/stdlib.jl:411-422`, which attributes it to TR 9407 II-4 Table 2 |
| 5 | `acoustics.water_density(temperature, salinity)`, one atmosphere (the source has no pressure term). Not wired into any default | `acoustics.py` | arlpy (BSD) `uwa.py:101-121` and UnderwaterAcoustics.jl `src/basic.jl:111-121`, both citing "Fofonoff (1985 - IES 80)" |
| 9 | `absorption(model="thorp")`, the two-term form SST uses; warns above 100 kHz | `acoustics.py` | APL-UW TR 0702 eq. (42) (not the GPL file where the survey first saw a Thorp formula, and not the four-term form of Jensen et al.) |
| 10 | Section 9 below: data only, no code | this document | UnderwaterAcoustics.jl `src/basic.jl:43-54`, `src/stdlib.jl:236-252` |
| 11 | `acoustics.bubbly_sound_speed(void_fraction, ...)` | `acoustics.py` | arlpy `uwa.py:215-231`, citing Wood (1964) and Buckingham (1997) |

A correction to section 2.3.1: it says the ratios are "independent of the reference pair". More
exactly, the arXiv paper's kg/m3 and m/s reproduce the Julia table's ratios only when divided
by 1000 kg/m3 and 1500 m/s (1845 / 1000 = 1.845 = 1887.44 / 1023; 1767 / 1500 = 1.178,
1800.29 / 1528 = 1.1782). Divided by 1023 and 1528 they would be 1.803 and 1.156. So the
paper's authors formed their physical values from the ratios with 1000 and 1500, which is
what the earlier work assumed.

Thorp, measured: 0.780 dB/km at 8 kHz (0.683 for Francois-Garrison at 4 degC and 1000 m, the
pair our TR 0702 validation quotes as 0.78 and 0.68). The two-term form is 0.273 / 1.150 /
31.1 / 43.5 dB/km at 4 / 10 / 100 / 750 kHz against Francois-Garrison's 0.245 / 0.963 / 33.6 /
222.4 at 10 degC, so it is a low-frequency model only.

## 9. Elastic seabed parameters (data only; no code here uses them)

Recorded so that the values are at hand if an elastic reflection coefficient is written.
Read in UnderwaterAcoustics.jl (MIT); the attributions are that package's. Hamilton's paper
was not opened, so nothing here is checked against it, and the validity of the regression
(sediment type, depth below the sea floor, frequency) is not stated in the package.

Shear speed c_s (m/s) from compressional speed c (m/s), `src/basic.jl:43-54`, "a regression
relationship from Hamilton (1980)":

| Range of c | c_s |
|---|---|
| below 1512 | not valid (the function raises an error) |
| 1512 to 1555 | 3.884 c - 5757 |
| 1555 to 1650 | 1.137 c - 1485 |
| 1650 to 2150 | 991 - 1.136 c + 0.00047 c^2 |
| above 2150 | 0.78 c - 962 |

The table's sediments down to sandy silt (c = 1527.85 m/s, which gives c_s = 177 m/s by the
first branch) are in the valid range; medium silt (1510.43 m/s) and finer are below 1512 and
the package defines no elastic versions of them. Hand values: fine sand (1691.95) 414 m/s;
medium sand (1800.29) 469 m/s; rock (3820) 2018 m/s.

Shear attenuation, `src/stdlib.jl:236-252`, "shear absorption based on values recommended in
Hamilton (1980)", in dB/m/kHz, with the material named in the line's comment:

| Value | Seabeds | Comment there |
|---|---|---|
| 0.07 | Rock, Pebbles | basalt |
| 13.2 | SandyGravel, VeryCoarseSand, CoarseSand, MediumSand, FineSand, VeryFineSand | diluvial sand |
| 4.8 | MuddySandyGravel, GravellyMuddySand, MuddyGravel, MuddySand, ClayeySand | diluvial sand & clay |
| 13.4 | CoarseSilt, SandySilt | alluvial silt |

Conversion used there (`src/utils.jl:232-235`): dB/m/kHz times the wave's own speed / 1000
gives dB per wavelength, and that divided by 40 pi / ln 10 gives the dimensionless loss
parameter.

## 10. The backscatter parameters that are still missing, and how to get them

Question from the owner: which backscatter parameters are missing and how can they be found;
then: use literature to complete every sediment. This section records what was searched, what
was found, and what was not. Nothing was filled in without a source.

### 10.1 What is missing

The scattering model (`sim/bottom.py`) needs six numbers per sediment. Three (density ratio,
speed ratio, loss) set the reflection coefficient; three more set the scattered level:

| Parameter | Meaning | Have it for |
|---|---|---|
| `w2` | roughness spectral strength, m^(4 - gamma2) | medium sand, very fine silt |
| `gamma2` | roughness spectral exponent | medium sand, very fine silt |
| `sigma2` | volume scattering cross section per unit volume / attenuation in dB/m | medium sand, very fine silt |
| shear speed and shear loss | for rock and gravel, which are not fluids | data only (section 9); no elastic model here |

### 10.2 What was searched (2026-11-17)

Reachable from this session: arxiv.org, github.com, zenodo.org, huggingface.co. Not reachable
(name resolution refused by the session's network policy, or a browser challenge):
www.mdpi.com, jcaa.caa-aca.ca, repository.tudelft.nl, www.ioa.org.uk, core.ac.uk,
discloseweb.webhosting.rug.nl, apl.uw.edu, pubs.aip.org, ieeexplore.ieee.org. DTIC was not
tried, to stay clear of the handbook.

| Source opened | Licence | What it gives | What it does not |
|---|---|---|---|
| Lurton, Lamarche et al., "Backscatter measurements by seafloor-mapping sonars: Guidelines and Recommendations", GeoHab BSWG 2015, doi:10.5281/zenodo.10089261, Table 2-1 and section 2.5 | CC BY 4.0 | Density, velocity, absorption (dB per wavelength) and grain size range (phi) for nine classes, clay to coarse sand, "adapted from APL 1994 ... nominal values for c1 and rho1 (1500 m/s and 1000 kg/m3)". Agrees with the Julia table to the printed digits, except coarse silt's speed (1523 m/s against 1.0179 x 1500 = 1527). Its footnote: in the handbook's model "the volume contribution finally depends upon a volume parameter which has to be adjusted empirically" and "the rough seafloors (gravels, pebbles and rocks) are modeled using heuristic fits issued from experimental records, instead of theoretical formulas" | No w2, gamma2, sigma2; its model curves are figures only |
| Bentrem, Sample, Kalcic and Duncan, arXiv:physics/0607146, Tables II and III, section III.A | arXiv | The search ranges used in an inversion: gamma between 2.4 and 3.9, volume parameter between 0.00 and 0.02, spectral strength "within factor of 2 of APL-UW parameterization"; rho, nu, delta "parameterize[d] ... according to phi values" after Mourad and Jackson and the handbook | The parameterisation itself is not printed |
| Brown, Johnson and Brownstead, arXiv:1809.08459, Table 1 | arXiv | The two complete rows already in the table | Nothing new |
| Olson, Lyons and Saebo, arXiv:1601.06078, Tables II, III and text | arXiv | Rock: measured scattering strength at 100 kHz and 20 degrees grazing, -33 to -26 dB (glacially abraded, smooth) and -30 to -20 dB (plucked, very rough); earlier measurements on rock -15 to -22 dB; "the APL-UW measurement from 'rough rock' ... approximately -8 dB at the same angle"; measured roughness spectra (abraded: 4.847e-8 m^(4 - gamma2), gamma2 = 2.73, in cycles/m; plucked: gamma2 = 4.36, outside the model's range); elastic parameters of the site's rock and "generic rock" | One angle's range, not a curve; elastic, not fluid |
| Olson and Jackson, arXiv:2009.12420, Tables I, II | arXiv | Example layered bottoms for a numerical study (one-dimensional roughness) | Not named sediments; not two-dimensional spectra |
| Olson and Lyons, arXiv:2106.05909 | arXiv | Example spectral strengths for a simulation study | Chosen for the study, not sediment classes |
| arXiv:2003.11156, arXiv:2311.08337, arXiv:1901.08023; Zenodo 1274941 (Berkson and Matthews 1984, large-scale roughness), Zenodo 1278330 | various | Nothing usable | - |
| GitHub, Hugging Face | - | No implementation of the model with its grain-size relations was found by search | - |

Result: no open source that could be reached prints w2, gamma2 or sigma2 for any further named
sediment, a regression of them against grain size, or a measured backscattering curve for a
sediment class as numbers. The relations are known to exist in the literature this session
could not open: search results (not opened) describe papers in Canadian Acoustics, J. Mar.
Sci. Eng. (MDPI, open access) and TU Delft's repository as using "APL-UW methods" to get the
parameters from mean grain size, and Jackson and Richardson, *High-Frequency Seafloor
Acoustics* (2007) is the textbook.

### 10.3 Provenance of every parameter after this work

`J` = UnderwaterAcoustics.jl (MIT) `src/stdlib.jl:113-136`, attributed there to TR 9407 IV-6
Table 2. `G` = also in the GeoHab report's Table 2-1 (CC BY). `B` = arXiv:1809.08459 Table 1,
attributed there to TR 9407.

| Sediment | rho, nu, delta | grain size (phi) | w2 | gamma2 | sigma2 | Complete |
|---|---|---|---|---|---|---|
| rock | J | - | absent | absent | absent | no (and needs an elastic model) |
| pebbles | J | - | absent | absent | absent | no |
| sandy_gravel | J (nu unconfirmed: 1.3570 or 1.3370) | - | absent | absent | absent | no |
| very_coarse_sand | J | - | absent | absent | absent | no |
| muddy_sandy_gravel | J | - | absent | absent | absent | no |
| coarse_sand | J, G | 0 to 1 (G) | absent | absent | absent | no |
| gravelly_muddy_sand | J | - | absent | absent | absent | no |
| medium_sand | B, J, G | 1 to 2 (G) | B | B | B | yes |
| muddy_gravel | J | - | absent | absent | absent | no |
| fine_sand | J, G | 2 to 3 (G) | absent | absent | absent | no |
| muddy_sand | J | - | absent | absent | absent | no |
| very_fine_sand | J, G | 3 to 4 (G) | absent | absent | absent | no |
| clayey_sand | J | - | absent | absent | absent | no |
| coarse_silt | J, G (speed differs by 0.26 %) | 4 to 5 (G) | absent | absent | absent | no |
| sandy_silt | J | - | absent | absent | absent | no |
| medium_silt | J, G | 5 to 6 (G) | absent | absent | absent | no |
| sandy_mud | J | - | absent | absent | absent | no |
| fine_silt | J, G | 6 to 7 (G) | absent | absent | absent | no |
| sandy_clay | J | - | absent | absent | absent | no |
| very_fine_silt | B, J, G | 7 to 8 (G) | B | B | B | yes |
| silty_clay | J | - | absent | absent | absent | no |
| clay | J, G | above 8 (G) | absent | absent | absent | no |

Having the grain size for nine classes means that one opened regression (w2, sigma2 against
phi) would complete seven of them at once.

### 10.4 The two complete rows, for a sanity check of the family

Backscattering strength in dB from `bottom.backscattering_strength_db` (default perturbation
model; the `combined` model in brackets where it differs by more than 0.2 dB), and the
empirical type law already in the tree (`surface.searay_bottom_db`, from arXiv:2210.06535).

| Frequency | Row or law | 10 deg | 20 deg | 45 deg |
|---|---|---|---|---|
| 6 kHz | medium_sand | -41.2 | -33.1 | -26.7 |
| 6 kHz | very_fine_silt | -36.1 | -32.0 | -28.4 |
| 6 kHz | searay mud / sand / gravel / rock | -44.2 / -43.7 / -34.8 / -16.9 | -44.2 / -42.9 / -27.9 / -7.2 | -44.2 / -42.3 / -24.9 / -3.2 |
| 100 kHz | medium_sand | -33.3 (-29.5) | -25.3 | -22.5 (-21.4) |
| 100 kHz | very_fine_silt | -36.1 | -31.9 | -28.4 |
| 100 kHz | searay mud / sand / gravel / rock | -40.3 / -33.7 / -25.4 / -16.9 | -38.7 / -29.1 / -18.2 / -7.2 | -38.0 / -27.1 / -15.2 / -3.2 |
| 300 kHz | medium_sand | -29.9 (-24.0) | -21.8 | -19.7 (-18.4) |
| 300 kHz | very_fine_silt | -36.0 | -31.9 | -28.4 |
| 300 kHz | searay mud / sand / gravel / rock | -31.0 / -26.4 / -21.7 / -16.9 | -28.6 / -21.6 / -14.4 / -7.2 | -27.6 / -19.5 / -11.4 / -3.2 |

Reading. Sand: the model and the type law agree within 4 dB at 100 and 300 kHz and 20 to 45
degrees (-25.3 against -29.1; -21.8 against -21.6), and disagree by about 10 dB at 6 kHz,
where the type law is outside any stated range (its source states none). Silt against "mud":
the model row does not change with frequency (volume scattering with a loss per wavelength),
the type law rises 13 dB from 6 to 300 kHz; they cross near 200 kHz. At 6 kHz the model puts
very fine silt above medium sand at 10 degrees: with these two rows the silt is a volume
scatterer and the sand a roughness scatterer whose level falls as f^(4 - gamma2) = f^0.75.
Two rows cannot show a trend from clay to gravel; that needs the missing parameters. One
outside number supports the type law for rock: arXiv:1601.06078 quotes the handbook's 'rough
rock' measurement as "approximately -8 dB" at 20 degrees, and `searay_bottom_db` gives -7.2.

### 10.5 The rock measurement as a check

`tools/boundary_models_study.py` prints the fluid model for the table's `rock` with the
abraded surface's measured roughness (converted from cycles/m: w2 = phi2 (2 pi)^(gamma2 - 2)
= 1.85e-7; the normalisation of the paper's spectrum is assumed) and no volume term: -39.3 dB
at 20 degrees and 100 kHz, against -33 to -26 dB measured ("most of the points are clustered
at the lower end"). The fluid model is 6 dB below the measured range. The paper's agreement is
with an elastic small-slope model after calibrating the sonar on one of the rocks, so this is
not a failure of the roughness term so much as a reminder that rock is not a fluid; it is
recorded as a comparison, not used as a test.

### 10.6 The fitting tool

`tools/fit_sediment.py` (library: `python/gpusonar/sim/sediment_fit.py`, tests:
`tests/python/test_sediment_fit.py`) fits w2, gamma2 and sigma2 to backscattering strength
against grazing angle and frequency, holding rho, nu, delta from a named row (the
reflection-only ones serve) or fitting them on request. It reports one-sigma uncertainties,
lists the parameters the data cannot fix with the reason, and prints a row for
`configs/sediments.yaml` with `status: fitted (<data source>)`; unconstrained parameters go
into `unconfirmed_fields`, so using the row warns about them.

    PYTHONPATH=python python3 tools/fit_sediment.py --data site.csv --sediment fine_sand \
        --name my_site --source "survey 2026-11, FLS 300 kHz, calibrated" --out row.yaml --plot fit.png

Measured on synthetic curves from our own model with 0.5 dB noise (the tests):

| Data | Result |
|---|---|
| medium sand, 5 to 55 degrees, 20 and 200 kHz | all three recovered; one sigma: w2 a factor 1.18, gamma2 0.024, sigma2 a factor 1.10; errors within 4 sigma in four noise realisations |
| very fine silt, 8 to 44 degrees, 100 kHz | sigma2 within 25 %; w2 and gamma2 flagged unconstrained (the data are volume scattering) |
| medium sand, 5 to 25 degrees, 300 kHz only | all three flagged: one frequency at low angles fixes only w2 (2k)^(-gamma2) (recovered within 15 %), and the volume term is not seen |

What coverage fixes what: sigma2 from wherever volume scattering dominates (soft sediments;
low frequencies on sand); the roughness level w2 (2k)^(-gamma2) from one frequency; w2 and
gamma2 separately from two well separated frequencies (level as f^(4 - gamma2)) or from
angles near the specular direction with the `small_slope` or `combined` roughness model.

Getting a calibrated curve from this suite's own FLS or SAS data. Scattering strength is
SS = RL - SL + 2 TL - 10 log10(A), with RL the received level in dB re 1 uPa, SL the source
level, TL the one-way loss and A the insonified area of the resolution cell. From a record:

| Needed | In the ping record | Otherwise |
|---|---|---|
| source level | `source_level_db` (NaN if unknown) | must be measured |
| receive chain: hydrophone sensitivity | no | the sonar configuration (`sensitivity_db`), must be the real one |
| fixed gain and gain ramp (TVG) | `hardware_gain_db`, `tvg_time`, `tvg_gain_db` | - |
| per-channel gain, phase, delay | `calibration/` | `calibrate.py` |
| converter scale | `adc_full_scale`, `adc_bits` | - |
| transmit and receive beam patterns at the cell's angles | no | the sonar configuration's element patterns; for an absolute level they must be measured patterns |
| spreading and absorption | `environment/` sound speed, temperature, salinity | `acoustics.absorption` |
| insonified area: range resolution x beam footprint (FLS) or the image's resolution cell (SAS), and the grazing angle | geometry from the pose and the water depth (`dvl` altitude) | local slope from bathymetry if the bed is not flat |
| processing gains (pulse compression, beamforming, any image normalisation) | no | must be undone or computed for the processing used |

`simcal.calibrate_fls` already does the level bookkeeping in the other direction (it fits how
many dB each boundary's strength must move for the simulator to reproduce a log's range
profile, absolute when the record's gains, source level and sensitivity are the true ones).
A practical route to a curve is therefore: run `calibrate_sim.py fls` in absolute mode with a
Lambert bottom over several range bins, or bin the conditioned, beamformed intensity by
grazing angle and apply the sonar equation above; then give the binned SS against angle (and
each frequency band available: the FLS sonars span 120 to 750 kHz between them) to
`fit_sediment.py`. Exported, flattened or 8-bit imagery gives relative levels only: the shape
against angle can still constrain gamma2 weakly, but w2 and sigma2 need the absolute level.

### 10.7 Realistic ways to get each missing parameter

| Missing | For | Ways, in order of effort |
|---|---|---|
| w2, gamma2, sigma2 | the 20 reflection-only rows | (1) Allow the session to reach one open paper that prints the grain-size relations (www.mdpi.com, jcaa.caa-aca.ca or repository.tudelft.nl), or the owner supplies Jackson and Richardson (2007) or such a paper: with the grain size ranges now in the table, one regression completes the nine classes that have a phi range, and the rest by the grain size of their names. (2) A grab sample and a grain-size analysis of the survey site, then the same regression. (3) Fit to the owner's own calibrated FLS/SAS data with `fit_sediment.py` (needs two frequencies or near-specular angles for w2 and gamma2 separately). (4) Roughness directly: stereo photogrammetry or a laser profile of the bed gives w2 and gamma2 (as arXiv:1601.06078 did for rock); sigma2 still needs acoustics |
| sigma2 specifically | all | The GeoHab report notes it "has to be adjusted empirically" even in the handbook's model: expect to fit it, not to look it up |
| sandy_gravel sound speed | one row | a second source for the table |
| shear parameters and an elastic model | rock, pebbles, gravels | section 9 has the regression and attenuations; an elastic reflection and scattering model would be new work. Until then use the type law (`searay_bottom`, "rock"/"gravel") for these, which matches the one measured number found (-8 dB at 20 degrees for rough rock) |
