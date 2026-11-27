# Task: pictures and sound of every scenario — rays, bounces, beams, boundary models, hydrophones

Work package "viz" (tracker issue `sonar-6k79`). Written 2026-11-26/27 on branch `viz` (from `batch4` at `da015ba`).
The owner's requests, verbatim:

1. "i want a parallel workstream that does a visualization of the beams and their bouncing and the received audio
   at each hydrophone for all our scenarios. design that in parallel. ideally 3d but perhaps 2d could be useful to
   start."
2. "it should show all the special things we have from SST like the models for the surface and bed".
3. "add a 3d viewer. dependency is fine. you can render to html as well. and ideally it uses taps from the real
   thing as well so we can visualize how the tracker is working, but that can be later."

This document is the design. A 2-D and a 3-D prototype for two scenarios exist (section 11): an opt-in tool,
`tools/scenario_view.py`, and the package `python/gpusonar/viz/`. Nothing in the simulator, the processing chain,
the live chain, a default or the record format (`docs/RECORD_FORMAT.md`) was changed. Status words are those of
`docs/REFERENCES.md`; "ours" on a measured number means measured on this machine (4 cores, no GPU) on 2026-11-26.

## 0. What cannot be shown, or is uncertain (read first)

- **A picture is of the simulator, not of the sea.** Every ray, loss and level drawn is what the code computes; no
  real data has been through it (`docs/handoff/SIMULATION.md` section 1). A page must never look like a
  measurement. The "what is modelled here" list on every page (section 3.3) names each model with its status word.
- **Ray paths exist for only part of the catalogue.** Of the 96 scenario files (counted 2026-11-26 by
  `tests/python/test_scene_trace.py::test_every_scenario_has_a_models_list`): 34 are `free_field` (one straight
  direct path, no boundary: nothing bounces), 23 `image_source`, 31 `eigenray` (Bellhop, 2-D), 6 `eigenray3d`
  (Bellhop3D), 2 `sediment`. Today the prototype draws exact paths for free field, for the flat image tier and for
  `eigenray` (Bellhop's own ray file, matched to the provider's eigenrays). It does NOT draw: the image tier's
  general planes (2 scenarios: `fls_120k_quay_multipath`, `pool_fls_p4_2v_walls`, the only two with propagation
  walls; the pool scenes' other walls are scene objects, reached by straight rays), the sediment tier's refracted
  ray (2), and `eigenray3d` (6). Each needs new code (section 2.1), none of it in the simulator.
- **The `eigenray` tier's rays lie in one vertical plane per bearing.** Bellhop is run once per source depth and
  looked up on a range-depth grid (`sim/bellhop.py`): every bearing sees the same section. In 3-D such rays are
  N copies of a 2-D picture; 3-D adds the array, the beams and several targets' bearings, not new propagation
  (section 5.4). Only the six `eigenray3d` scenes have rays that leave the vertical plane.
- **Per-bounce losses are re-evaluated, not reported.** Bellhop writes one complex amplitude per arrival, not a loss
  per bounce. The page evaluates the same Rayleigh law of the same half-space at each drawn bounce's grazing angle
  (`viz/trace.py` `bounce_loss_db`). For the image tier this is exact: the product of the drawn bounce losses and
  the spherical spreading equals the provider's amplitude to 1e-9 dB (test
  `test_per_bounce_losses_make_up_the_image_rays_amplitude`). For Bellhop it is the same formula, not Bellhop's
  number. Time-spread taps (4 scenarios), the rough-surface coherent loss of `propagation.wind_speed` (no scenario
  sets it above 0) and a bubble loss per bounce are not drawn per bounce yet.
- **Reverberation is not rays, and its parts are not split.** The scattering-function generator adds all items to
  one buffer. The simulator's own taps give echoes, reverberation (all items) and noise separately; surface, bottom
  and volume reverberation are not separable without a per-item tap (section 3.2: Python-only, opt-in).
- **The "beams" are partly assumptions.** The simulator applies element and projector patterns (a separable sinc,
  `sim/reverb.py` `element_power_pattern`); most shipped projectors are `kind: point`. Steered receive beams are not
  in the simulator at all: they belong to the processing (FLS fan, SAS image, live chain `bearings_deg`, default
  -40 to 40 deg by 0.5, `python/gpusonar/live.py`). The prototype's plan view draws 9 analytic uniform-line-array
  beams (half-power width 0.886 lambda / (N d cos theta), ours), not the chain's configured beams.
- **The received record is often unreadable per channel.** In `flank_target_1km` the echo at a single hydrophone
  lies about 20 dB under the reverberation at 1.4 s (prototype page, lowest panel; ours): the per-channel matched
  filter shows no echo; the array gain is what finds it. A page therefore shows the echoes alone (the simulator's
  tap) beside the record, and a beamformed view belongs in V6 (live-chain taps).
- **Most bands are not audible.** Of the 96 scenarios, the flank array (6-10 kHz) and the long-range Popoto
  designs (4-6 kHz) are; the 27.5 kHz Popoto boards, every FLS (110 kHz to 6.25 MHz) and SAS (180 kHz to 3.7 MHz)
  band are not. Making them audible changes them (section 4.3).
- **A 3-D viewer is an assumption about who looks.** The owner approved a dependency and HTML output; whether the
  pages are for engineers debugging or for a demonstration still decides the defaults (question 1).

## 1. Questions for the owner (open), each with a recommendation

1. **Who looks at these pages: engineers debugging a scenario, or a demonstration?** Recommendation: engineers
   first. Pages carry every model and status word, losses in dB, the per-channel record as it is; a demonstration
   mode (fewer panels, larger 3-D, sound) can follow from the same trace.
2. **How should inaudible bands be made audible?** Options (section 4.3): shift the band down by a fixed number of
   hertz (keeps durations, Doppler in Hz; loses ratios and most of a wide band), time-stretch (keeps ratios and
   the sweep's shape; lengthens and lowers everything), or both. Recommendation: shift by default, keeping 3.6 kHz
   about the centre (the prototype's default), with time-stretch as an option for SAS and FLS echoes, and the
   method printed in the WAV's file name and on the page.
3. **Is per-arrival audio wanted** (one WAV per eigenray pair, and a page that plays an arrival when its ray is
   clicked)? It needs one synthesis per pair (section 4.4; no simulator change). Recommendation: yes for the
   strongest 8 pairs of one target per scenario, not for all (a 32-channel flank scene has 484 pairs a target).
4. **One page per scenario, or one viewer with a scenario picker?** Recommendation: one self-contained page per
   scenario (shareable, works from a file), plus a generated index page that links them (V2).
5. **Where do the 96 pages live?** Committed PNGs would add about 5 to 15 MB to a small repository (50 to 150 kB a
   page, ours). Recommendation: commit one page per family (12 pages, about 1.2 MB) under `docs/figures/scenes/`,
   upload the rest with the GPU job's dataset (the approved service, section 6.3), and never commit traces over
   100 kB or embedded-library HTML pages.
6. **Vendor three.js?** The viewer loads three.js 0.170.0 (MIT) from a pinned CDN by default (pages of 14 to 65 kB,
   need the network) or embeds a local copy (706 to 756 kB a page, offline). Recommendation: vendor the one file
   (0.69 MB, MIT, with its LICENSE) under `third_party/three/` so that offline pages can be built reproducibly
   without the network; record it in `docs/OPEN_SOURCE.md`. Not done in this package.
7. **Does "the special things we have from SST" mean more than the inventory of section 3?** Section 3 lists the
   boundary, medium, reverberation, noise and spreading models adopted from the TR 0702 family and the papers
   behind them, with where each is used. If the owner also means SST's own displays (for example its
   eigenray/arrival plots or its reverberation-versus-time plots in a particular layout), say which, and the page
   layout follows them. Recommendation: confirm section 3 is the scope.
8. **Live-chain taps (V6, later): which view of the tracker matters most?** Recommendation: the beam-range power map
   per ping with detections and gates, tracks with their covariance ellipses and history, in the same plan view as
   the scene's truth; the tracker's internals (gates, associations, covariance) need new opt-in taps in both
   engines (section 8).

Answered and no longer asked: a 3-D viewer is in scope; a permissive dependency is approved; HTML output is wanted
(the owner, 2026-11-26).

## 2. What can be drawn today, per tier and mode

### 2.1 By fidelity tier

| Tier (scenarios) | Eigenray data the provider returns | Ray path drawn | Bounce points | Status of the drawing |
| --- | --- | --- | --- | --- |
| `free_field` (34) | one direct ray: delay, 1/r, launch = arrival direction | straight segment, exact | none (no boundary) | prototype: done |
| `image_source`, flat (21) | per image: delay, amplitude with surface -1 and the bottom's Rayleigh or constant coefficient, launch and arrival, `n_surface`, `n_bottom` (`sim/propagation.py` `ImageSourceProvider`) | the image line folded back between the flat surface and bottom, exact (`viz/trace.py` `folded_path`) | exact, on the planes, with the grazing angle and the provider's own loss | prototype: done (held by `test_folded_image_paths_...`) |
| `image_source`, general planes (2) | the same, with plane sequences (`_PlaneImages.sequences`) | not drawn | not drawn | V1: specular points from the plane sequence (NumPy, ours); no simulator change |
| `sediment` (2) | one refracted ray (`snell_crossing`) with transmission | not drawn | the interface crossing | V2: polyline from `snell_crossing`; the `rough` sub-critical patch sum is not a ray (drawn as the patch) |
| `eigenray` (31) | Bellhop arrivals on a range-depth grid: delay, complex amplitude, launch and arrival angles, surface and bottom bounce counts (`sim/bellhop.py` `read_arrivals`); the runner requests run type `A` only | Bellhop's own ray file: run type `E` is written by the same `write_env` and run as a separate process; `viz/trace.py` `read_rays` reads it, cuts each ray at the target's range and matches it to the provider's eigenray by bounce counts, end depth and launch angle | Bellhop's boundary points (it writes the boundary point, often twice; deduplicated) with grazing angle and the re-evaluated Rayleigh loss | prototype: done. `flank_target_1km`: 44 of 44 eigenrays matched; travel time along the drawn ray within 1.2 ms of the provider's delay (ours; asserted under 2 ms) |
| `eigenray3d` (6) | Bellhop3D's merged arrivals, scaled to its TL run (`sim/bellhop3d.py`) | not drawn | not drawn | V4: Bellhop3D run type `E` (3-D ray file: x, y, z per step; a reader is new code, format to be read from the Toolbox source as `bellhop3d.py` was) |

Time-spread scenarios (4: `asiaex_dahl_2004`, `flank_sst_shallow`, `flank_sst_spread`, `sst_asiaex`): the
simulator's pairs are the wrapper's taps (spread copies of each bounce eigenray); the page draws the bounce
eigenrays under them and marks the taps' pairs in the arrival stems (prototype: done).

### 2.2 By family (the families of `docs/handoff/SIMULATION.md` section 4)

| Family (files) | Tiers | Modes and band | What its picture can contain today |
| --- | --- | --- | --- |
| Point targets (14) | free 14 | all sonars | geometry, straight rays to each grid point, beams; no bounce |
| Forward-looking sonar (14) | free 8, image 6 (2 general planes) | FLS, 110 kHz to 6.25 MHz | 4 transit/traffic scenes: surface and bottom images with Rayleigh losses; pool and harbour scenes: walls are scene objects (straight rays) |
| SAS and buried objects (12) | free 9, sediment 2, image 1 | SAS, 10 to 350 kHz and 1.7-3.7 MHz | `sas_auv_multipath`: flat images; buried: refracted ray (V2); the rest straight; one ping of a stripmap is not the SAS picture (V5 shows the track) |
| Flank array, 32 hydrophones (19, with `flank_patrol_doppler_tone`) | eigenray 11, eigenray3d 6, image 2 | 6-10 kHz | Bellhop rays with bounces (11), 3-D rays later (6), SST spread scenes (2) |
| TR 0702 validation, ASIAEX (5) | image 4, free 1 | flank_32 and popoto_8 | images; one-way sources (`sst_asiaex`, `asiaex_dahl_2004`, `sst_pursuit`) need a one-way leg drawn (V2) |
| Popoto 27.5 kHz and first long-range (5) | eigenray 5 | 20-35 kHz | Bellhop rays |
| The 12 km design (9) | eigenray 9 | 4-6 kHz | Bellhop rays over 4 to 12 km; profiles with ducts (`popoto_env_*`) |
| Stacked rows (4) | free 1, image 3 | 4-6 kHz | images; the rows' elevation is the point (3-D helps) |
| Staged builds (5) | image 4, free 1 | 4-6 kHz | images |
| Direct sound and overload (3) | image 3 | 4-6 kHz | images; the projector's direct arrival (a leg the trace does not draw yet, V2) |
| Fjord (3), water changing with range (3) | eigenray 6 | 4-6 kHz | Bellhop rays over a range-dependent bed (`.bty`) or water (`.ssp`): the section must show the bed and the water per range (V2) |

## 3. The boundaries and the medium: the SST-family models, and how each is shown

### 3.1 Inventory (counts: `models_in_force` over the 96 files, ours, 2026-11-26; sources and status as in `docs/handoff/SIMULATION.md` section 3)

| Area | Model (code) | Default or opt-in | Scenarios | Source, status |
| --- | --- | --- | --- | --- |
| Surface, propagation | pressure release, coefficient -1 (`ImageSourceProvider.surface_coefficient`; Bellhop `'V'` top) | default wherever a surface exists | 62 | pressure release; ours |
| Surface, propagation | rough-surface coherent loss exp(-Gamma^2/2) per bounce (`acoustics.surface_coherent_reflection`, `propagation.wind_speed`) | opt-in (0 = flat) | 0 above 0 (20 files name the key under `environment_overrides`, D-101) | ours |
| Surface, spreading | time, angle, Doppler spreading of bounce paths; bubble loss per bounce (`sim/timespread.py`, `time_spread`) | opt-in | 4 (`flank_sst_spread` with Doppler and angle spreading; 5 files name a bubble key) | TR 0702 eqs. 89-103 confirmed; facet model ours; Dahl 1996 (A7) confirmed |
| Surface, scattering | Chapman-Harris (`sim/surface.py`) | per item | 23 | McDaniel 1993 eq. (27); via secondary |
| Surface, scattering | `kraus_surface`; `bubble_layer_surface` with `extinction: dahl_1997` (default), `apl_1994`, `hall_novarini` | per item | 0 | via secondary; Dahl et al. 1997 Table II confirmed |
| Bottom, propagation | Rayleigh fluid half-space (`propagation.rayleigh_reflection`) | default with a described bottom | 57 | standard; sediment table via secondary |
| Bottom, propagation | elastic half-space (rock) | from the sediment row | 0 in the catalogue's propagation | J&R (2007) (9.34)-(9.40), confirmed; shear values unconfirmed |
| Bottom, propagation | constant coefficient | per scenario | 2 (`sst_echo_level`, `sst_pursuit`) | ours |
| Bottom, propagation | range-dependent bed (Bellhop `.bty`) | per scenario | 11 (fjord, 3-D twins) | Bellhop; ours |
| Bottom, penetration | `penetration: constant | fluid | rough` (`sim/penetration.py`) | `constant` default; `fluid` in both buried scenes | 2 | J&R (2007) through the notes, confirmed; split ours |
| Bottom, scattering | Lambert mu sin sin (`sim/reverb.py`) | default item model | 27 | as a thesis quotes it; via secondary |
| Bottom, scattering | rough fluid sediment, `roughness_model: small_slope` (default), perturbation, kirchhoff, composite, combined; `volume_model: stockhausen` | per item | 3 (`flank_sst_shallow`, `flank_sst_spread`, `sst_sediment_reverb`), all small-slope | ours = published forms; validity confirmed |
| Bottom, scattering | Williams-Jackson (1998), Mourad-Jackson (1989) | opt-in | 0 | confirmed (the papers) |
| Reverberation | scattering-function generator; bistatic geometry; K statistics; relief; specular exclusion | default generator; the rest opt-in | 31; bistatic 1 (`sst_pursuit`); K 0 in reverberation; relief 0 | TR 0702 Sec. 10-11 idea confirmed, generator ours; Abraham and Lyons 2002 confirmed |
| Water | sound speed (Mackenzie) | default | 96 | unconfirmed in print |
| Water | profile in depth; profile by range (Bellhop `'Q'`) | per scenario | 9; 3 | ours |
| Water | absorption Francois-Garrison (default), Ainslie-McColm, Thorp, none | per scenario | 31, 51, 0, 14 | via secondary; Thorp confirmed |
| Water | volume layer as a sheet | per item | 1 (`flank_sst_shallow`) | TR 0702 Sec. 6.5, confirmed |
| Water | ambient noise: Wenz fit, stated level, fitted to a recording; K or alpha-stable statistics | per scenario | 37 Wenz, 22 level, 3 name `statistics` (fjord) | via secondary; ours |
| Basin | late reverberation tail (`sim/basin.py`) | opt-in | 3 (fjord) | ours, one data set |
| Sources | one-way sources; projector direct sound and front-end overload | opt-in | 3; 3 | ours |

Not yet in `models_in_force` (V1): the sonar keys `transmit.direct_path` and `receive.overload` (the legend reads
the scenario, not the sonar), noise statistics as their own row, the relief and specular-exclusion options.

### 3.2 How each is shown

| Model | On the ray picture | Companion panel | Exists now / needs |
| --- | --- | --- | --- |
| Surface (flat, -1) | the line at z = 0, labelled; each surface bounce marked (triangle) with grazing angle and 0 dB | reflection loss vs grazing angle: a 0 dB line | prototype |
| Surface roughness (wind, rms height) | the surface drawn as a band of +-rms height (`time_spread.surface.rms_height`, or from wind by the model's law) with the wind and sea state written on it | coherent loss exp(-Gamma^2/2) vs grazing at the band's centre, the bounces' angles marked | V1 (the law is `acoustics.surface_coherent_reflection`; no new physics) |
| Bubble loss per bounce | a number at each surface bounce | bubble loss vs grazing (and vs depth for `bubble_layer_surface`'s extinction) | V1 for the constant `bubble_loss_db`; the depth form needs the extinction law evaluated (Python only) |
| Surface scattering strength | the insonified surface ring at the ping's time (3-D) | strength vs grazing, monostatic and, for a bistatic item, at the scene's azimuth | prototype (monostatic, every item, from `simulator.reverberation_item_strength`, the law the simulator builds) |
| Bottom half-space (fluid, elastic, constant) | the bed coloured by its sediment row, with speed, density, attenuation, shear | reflection loss vs grazing with the drawn bounces' angles marked; the critical angle marked | prototype (fluid, constant); V1 critical angle and elastic label |
| Range-dependent bed and water | the `.bty` bed and the `.ssp` section drawn in range | the profile per range (a small multiple) | V2 |
| Penetration (sediment tier) | the interface, the refracted leg below it, the buried object; the evanescent region shaded below the critical angle | transmission vs grazing | V2 (`snell_crossing`, `fluid_transmission`) |
| Bottom scattering (Lambert, rough fluid sediment, the opt-ins) | the bed's scattering cells at the ping's time (3-D) | strength vs grazing for the model in force | prototype (monostatic) |
| Reverberation vs time | — | level vs time beside the channels: echoes, reverberation, noise (prototype); split by item | prototype for the three parts. Per item: an opt-in tap per reverberation item in `simulate_ping`'s Python loop (`taps.sim_tap(record, f"reverberation[{i}]")` behind a `sim_capture(..., per_item=True)` switch). Each item runs in either engine inside that one Python loop, so one Python change covers both engines; a test sums the items to the existing `reverberation` stage in both engines |
| Time spreading | the spread taps as faint copies about each bounce ray, their Doppler lines in the stems | delay-Doppler scatter of the taps | V3 (read from the wrapper's EigenraySet: `range_rate`) |
| Sound speed | the profile beside the section (prototype); in 3-D a coloured water column | — | prototype |
| Absorption | — | dB per km at the band's centre in the legend | V1 |
| Ambient noise | — | noise level vs time (prototype); spectrum against the model's | V3 |

### 3.3 "What is modelled here"

Every page lists each model in force with its key, value, whether the scenario chose it or left it to the default,
its source and its status word (`viz/trace.py` `models_in_force`; the table it uses repeats
`docs/handoff/SIMULATION.md` section 3 and must follow it when that table changes). A model the scenario does not
use is never listed or drawn: the bottom drawn and its loss curve are the tier's (a constant coefficient where the
scenario gives one, not the site's sediment row: `test_models_in_force_name_only_what_the_scenario_uses`), and a
free-field page says that no surface or bottom is in the propagation.

## 4. The received sound at each hydrophone

### 4.1 What the record holds per channel

`samples` `[C, N]`, `sample_kind` 0 real passband or 1 complex baseband, `sample_rate`, `center_frequency` (the
removed carrier for baseband), `start_time` (`docs/RECORD_FORMAT.md`). Across the 96 scenarios (ours, from the
sonar configs): flank and Popoto real at 32 to 102.4 kHz or baseband at 8 to 20 kHz; FLS real at 0.5 to 50 MHz or
baseband at 40 to 400 kHz; SAS real at 1 or 50 MHz or baseband at 60 to 100 kHz. Bands: 4-6 kHz (27 files),
6-10 kHz (22), 20-35 kHz (5), FLS 110 kHz to 6.25 MHz (19), SAS 10 kHz to 3.7 MHz (17).

### 4.2 Presentation

- Per channel: matched-filtered envelope against time (the chain's own first stage: `core.to_baseband`,
  `core.pulse_compress`), each arrival's per-hydrophone delay marked and coloured by its bounce class
  (prototype); a click on a mark highlights its ray (V4, in the HTML page).
- Channel-by-time picture ("waterfall") of the record and, beside it, of the echoes alone (the simulator's tap)
  (prototype).
- Spectrogram of one channel with the transmit band and the arrivals' Doppler marked (V3).
- The parts against time: echoes, reverberation, noise (prototype), reverberation by item (V3).

### 4.3 Listening (`python/gpusonar/viz/audio.py`)

| Way | Used when | What it misrepresents |
| --- | --- | --- |
| as is (real passband at its own rate) | flank 6-10 kHz, Popoto 4-6 kHz, real | nothing; players may refuse 64 or 102.4 kHz files |
| remodulate (baseband back on its carrier, resampled to 2.5 x the top of the band) | the same bands, baseband | interpolation only |
| shift (band moved so its centre sits at 2 kHz; 3.6 kHz kept) | 20-35 kHz Popoto, FLS, SAS | equal shift in Hz: Doppler keeps its size in Hz, not its ratio; of a wide band only 3.6 kHz is kept, so the sweep's range resolution is lost |
| time-stretch (play at fs / k) | option (not built) | lowers every frequency by k and lengthens by k; keeps ratios and the sweep's shape |

WAV per hydrophone: 32-bit float, mono, the method written in the result (`write_wav`; `taps.export_wav` writes
all channels of a record unchanged). Tested: rate and duration kept, the tone inside the band, nothing folded
through 0 Hz after a shift (`test_audio_export_keeps_rate_and_duration`).

### 4.4 One arrival alone

The simulator can synthesise one path's contribution without change: a provider wrapper (Python, new, about 40
lines) returns the provider's EigenraySet with every ray masked out but ray k when called from the projector and
ray j when called from the array's reference point (`two_way_paths` calls the provider once from each), and the
scenario is run with reverberation and noise off. Both engines then synthesise the same masked set. Test: the sum
of the single-pair records equals the full echoes-only record to rounding, both engines (linearity of
synthesis; ours). Cost: one synthesis per pair; the strongest 8 pairs of one target are proposed (question 3).

## 5. The scene trace, the 2-D pages, the 3-D viewer

### 5.1 The scene trace (a new file kind; not the record format)

JSON, `kind: gpusonar.scene_trace`, `version: 1` (`viz/trace.py` `build_trace`). For one ping:

| Member | Content |
| --- | --- |
| `scenario`, `tier`, `ping`, `time_s` | what was traced |
| `sonar` | name, mode, band, centre frequency, sample kind and rate, start time, channel count, projector and hydrophone positions (m, NED), source level |
| `environment` | sound speed, surface and bottom depth (the tier's, or the site's marked as such), profile, bathymetry, the bottom the tier reflects from, wind, absorption |
| `models` | section 3.3 |
| `scattering` | strength vs grazing per reverberation item |
| `targets` | label, position, amplitude (point targets each; patch objects at their strongest patch this ping; seafloor, sea surface and volume items are not targets) |
| `legs[target].out/back[]` | per eigenray of each leg: index k, delay, amplitude, bounce counts, launch and arrival angles, `polyline` (m) or null with `path` saying why, `bounces[]` {xyz, boundary, grazing_deg, loss_db}; for Bellhop also the travel time along the drawn ray and its end-depth miss |
| `pairs[]` | every two-way pair: out k, back j, delay, `channel_delay_s[C]` (as the kernel places the echo: apparent points q, p), bounce counts, level re 1 uPa (spreading and reflection, no absorption or beam pattern) |
| `time_spread_taps` | true when the pairs are a time-spread wrapper's |
| `chain` | reserved for the live chain's taps (section 8); null |

Sizes (ours): 6 kB (`sst_echo_level`) to 0.69 MB (`flank_target_1km`, 484 pairs x 32 channel delays) and 0.77 MB
(`flank_sst_spread`, 1024 pairs). For committed traces the per-channel delays are dropped (recomputable from q, p)
or the trace is split into JSON plus a float32 NPZ (V2).

### 5.2 The 2-D page (`viz/page2d.py`, matplotlib only)

One PNG per scenario: profile; range-depth section with the eigenrays (strongest 8 per leg in colour by bounce
class with bounce markers, the rest grey), surface and bed labelled; reflection loss vs grazing with the drawn
bounces; scattering strength in force; plan view with bearings and a beam fan; arrival stems; record and
echoes-alone waterfalls with the arrivals marked; the parts against time; the models list. Colours: four bounce
classes checked with the dataviz palette validator (CVD and normal-vision separation pass; the green is below 3:1
on white, so every class also has its own marker and a legend). A page is 50 to 150 kB as a 256-colour PNG
(ours). Range is horizontal range from the sonar on each ray's own bearing.

### 5.3 The 3-D viewer (`viz/page3d.py`): chosen route

**Chosen: our own self-contained HTML page per scenario, drawn by three.js in the browser; no Python
dependency.** One mechanism serves both the interactive viewer (open the page) and the export (the page). three.js
is MIT; release 0.170.0's `three.module.min.js` is one file of 691,648 bytes. Loaded from a pinned CDN by default
(page 14 to 65 kB; needs the network); embedded with `--three PATH` (706 to 756 kB; works offline; question 6).
Checked first: none of plotly, pyvista, vispy, trimesh, k3d, pythreejs, pygltflib or vtk is installed here.
Rejected: Plotly 7.2.0 (MIT; 40.4 MB unpacked plus `narwhals`; 4.8 MB of script per embedded page; animation only
by precomputed frames, no audio sync); pyvista/VTK (BSD-3; large binary wheels; desktop window, HTML needs more
packages; size not measured); k3d, pythreejs (notebook widgets, not a standalone page). A neutral scene file
(glTF) is a later option for external viewers: three.js reads it, our trace writes the same geometry.
Recorded in `docs/OPEN_SOURCE.md` 2g and 3.14. The suite never imports any of it; the tests need no browser.

The page holds the thinned trace (strongest 8 rays a leg, 40 pairs) as JSON: surface and bed planes (the bed
marked as the site's depth when the tier has none), projector, hydrophones, targets, rays with bounce balls, the
receive-beam fan, a "what is modelled here" table; sliders for vertical exaggeration and time; play. Time: out
legs run from 0 to their delay, back legs from the earliest arrival at the target; a hydrophone lights while a
pair's arrival is within 0.4 % of the time span of its delay at that hydrophone. Audio in sync (V5): the page
carries the per-hydrophone WAV (base64, or beside it) and plays from the slider's time with the WebAudio API.

### 5.4 What 3-D adds, per mode

- Flank and Popoto at the `eigenray` tier: the array's line and rows, several targets' bearings, the beam fan, the
  bed and surface as planes, the ping going out and back. The rays themselves are 2-D sections: in 3-D they are
  decoration over the 2-D section, said so on the page.
- `eigenray3d` (6): real 3-D: rays bent across the bearing by a seamount, a ridge, a canyon. This is where 3-D
  shows something the 2-D page cannot.
- Stacked rows (4) and the 2- and 3-row builds: the elevation of arrivals across rows.
- FLS: the sector, the beams and the objects in the water column and on the bed; images of a quay wall and a tilted
  bed (once drawn, V1).
- SAS: the track over pings, the beam footprint on the bed, a buried object below the interface; one ping's rays
  are short and nearly straight.

## 6. The generator, cost and outputs

### 6.1 One tool

`tools/scenario_view.py <scenario...> | --all --out DIR [--wav C] [--html] [--three PATH] [--engine cpp|numpy]`.
It simulates the first ping only under the simulator's stage taps (`taps.sim_capture`, a temporary directory),
builds the trace, writes `NAME.trace.json`, `NAME.png`, and optionally `NAME.chC.wav` and `NAME.html`. One
scenario failing does not stop `--all`; each prints one JSON line with its timings.

### 6.2 Cost (ours, this CPU, one ping, NumPy engine except where noted)

| Scenario | Tier | Simulate | Trace | Page |
| --- | --- | --- | --- | --- |
| `sst_echo_level` | image | 0.02 s | 0.00 s | 1.3 s |
| `buried_cylinder` | sediment | 1.7 s | 0.05 s | 0.9 s |
| `popoto_trial_100m` | eigenray | 4.1 s | 1.0 s | 1.5 s |
| `seafloor_objects_fls` | free | 5.3 s | 0.01 s | 1.3 s |
| `fls_120k_transit` | image | 13 s | 0.02 s | 1.9 s |
| `flank_target_1km` | eigenray | 18 s | 2.3 s | 2.6 s |
| `flank_sst_spread` | image + spreading | 35 s | 0.06 s | 3.4 s |

`--all` on this CPU is not measured. Estimate (ours): most scenes take seconds to a minute for one ping; the six
`eigenray3d` scenes take 3 to 7 min each in their provider (`docs/handoff/SIMULATION.md` section 3); in all about
one hour. On the GPU job the synthesis is CUDA but Bellhop and Bellhop3D still run on the CPU, so the saving is in
the dense scenes (`sas_minsas_speckle`) only.

### 6.3 Outputs

Scratch outputs go under the session scratchpad or `DIR`. Committed (proposal, question 5): one 2-D page and one
CDN-loading HTML page per family under `docs/figures/scenes/` (about 1.5 MB in all). Uploaded with the GPU job's
dataset (its own folder): all 96 PNG pages, the traces, the embedded HTML pages and the WAVs.

## 7. Tests and rules

- The trace's delays are the simulator's: the pairs equal `two_way_paths` with the same provider, both engines, to
  1e-12 s (`test_trace_delays_are_the_simulators_pairs[numpy|cpp]`).
- The drawn image paths end on the target, bounce on the planes, and are as long as the image path
  (`test_folded_image_paths_...`); the bounce losses make up the ray's amplitude (`test_per_bounce_losses_...`).
- The marked arrivals sit on local maxima of the matched-filtered record within 1.5 samples, every channel
  (`test_marked_arrivals_sit_on_the_matched_filter_peaks`).
- Bellhop's rays: at least 90 % of the eigenrays matched, bounce counts equal, bounce points on the boundaries,
  travel time within 2 ms of the provider's delay at 1 km (`test_bellhop_rays_...`; skips without
  `GPUSONAR_BELLHOP`, fails under `GPUSONAR_REQUIRE_ORACLES=1` through `oracle_gate`).
- Audio keeps rate and duration; the models list is only what is in force; every one of the 96 files gives a
  models list; the 2-D page is under 300 kB and the HTML holds the rays it says (`test_pages_hold_what_they_draw`).
- To add per package: per-item reverberation taps sum to the stage (both engines); single-pair synthesis sums to the
  echoes (both engines); the general-plane specular points lie on their planes; Bellhop3D ray reader on a
  hand-made file; the HTML page loads without console errors (Playwright, opt-in like the HSB studies).
- Rules kept: Bellhop and Bellhop3D stay separate processes; no default, record field or simulator behaviour is
  changed; the viewer library is never imported by the suite; nothing is fetched at test time.

## 8. The live chain's taps (V6, later)

What the chain produces per ping, and where:

| Item | Python engine (`live.py`) | C++ engine (`libs/live`) | Report line (JSONL), both engines | Needs |
| --- | --- | --- | --- | --- |
| Records after raw, sync, prepare | taps `raw`, `sync`, `prepare` | `prepare` (the conditioned record, via the binding) | — | exists (`taps.py` TapWriter) |
| Beam-range power map | tap `beamform` (`power`, `ranges`, `bearings`) | tap `beamform` (`last_map()`) | — | exists in both |
| Detections: range, bearing, SNR, extent, arrivals; elevation and its sigma (rows); range rate and `doppler_path` (opt-in `doppler.report`) | in the report | in the report | yes | exists |
| Own-ship pose | `ownship_xyz` | same | yes | exists (attitude not in the line) |
| Tracks: id, kind, parent, x, y, vx, vy, speed, range, mean SNR, seen; depth and its sigma (rows) | in the report | in the report | yes | exists; history is rebuilt from successive lines by id |
| Track covariance, hits, misses, confirmed | `detect.Track` (`cov`, `hits`, `misses`, `confirmed`) | `tracker.cpp` | no | new opt-in tap in both engines (`tap.json("tracks", ...)`), held equal by `tests/python/test_live_cpp.py` |
| Gates and associations (which detection each track took, the d^2 of each gated pair) | inside `Tracker.update` | inside `tracker.cpp` | no | new opt-in tap in both engines, same test |
| Detection map cells, CFAR threshold | `Detection.cells`, the detector | the detector | no | new opt-in tap (both engines) |
| The truth to compare with | the scenario's scene and platform (`ScenarioResult`) | — | — | exists beside the records |

The trace's `chain` member is reserved for these per ping: `{ping, time_ns, ownship, map: {ranges, bearings,
power (dB, decimated)} | null, detections[], tracks[] (with cov and history when tapped), associations[] |
null}`. The viewer then draws, per ping, the power map in the plan view, detections as marks, tracks with
covariance ellipses and their history, gates as circles, and the truth's targets. Acceptance for V6: the viewer
reads a `sonar_live.py --scenario NAME --taps DIR` run of either engine; the new taps are opt-in, off by default,
equal in both engines on `popoto_12km` and `flank_patrol` (`test_live_cpp.py`), and the report line is unchanged.

## 9. Staged plan

| Package | Content | Acceptance | Effort | GPU job |
| --- | --- | --- | --- | --- |
| V1 trace + 2-D page, one family (flank) | the prototype hardened: general-plane image paths, critical angle, coherent surface loss band and bubble loss labels, absorption in the legend, sonar keys in the legend, colour/label fixes of section 11 | the 19 flank scenarios produce pages; every test of section 7 passes in both engines; pages under 300 kB | M (one agent, 1-2 days) | no (Bellhop on CPU) |
| V2 all scenarios + index | sediment refracted leg, one-way sources and the direct arrival, range-dependent bed and water drawn, trace split JSON + NPZ, index page | `--all` runs to the end with no exception on this CPU (6 `eigenray3d` excepted until V4); index links 96 pages | M | optional, to time `--all` |
| V3 audio and spectra | spectrogram panel, per-item reverberation taps (opt-in, Python loop, both engines tested), time-stretch option, single-pair synthesis for the strongest 8 pairs, noise spectrum vs model | per-item sum equals the stage; single pairs sum to the echoes (both engines); WAV rates and durations | M | no |
| V4 3-D | the page of section 5.3 for all tiers; Bellhop3D ray reader and 3-D rays; vendored three.js if the owner agrees | 96 pages load in headless Chromium with no console error; ray counts in the page equal the trace's | M-L | Bellhop3D runs: CPU, 3-7 min a scene |
| V5 animation and sound | time slider with ping fronts, hydrophone lighting, audio in sync from the WAVs; SAS/FLS multi-ping tracks | the lit hydrophone at time t matches the trace's per-channel delays (test on the page's data in Node) | M | no |
| V6 live-chain taps | section 8: new opt-in tracker and detector taps in both engines; the viewer's chain layer | `test_live_cpp.py` holds the taps equal; viewer reads both engines' taps | L | yes, before merge (the suite's rule) |

## 10. What this package changed

New: `python/gpusonar/viz/` (`trace.py`, `page2d.py`, `page3d.py`, `audio.py`), `tools/scenario_view.py`,
`tests/python/test_scene_trace.py`, this document, `docs/tasks/viz_prototype/`, rows 2g and 3.14 of
`docs/OPEN_SOURCE.md`. Unchanged: simulator, processing, live chain, record format, defaults, configs.

## 11. The prototype, and what is wrong in it

Files (`docs/tasks/viz_prototype/`): `sst_echo_level.png` and `flank_target_1km.png` (2-D pages),
`sst_echo_level_3d.png` and `flank_target_1km_3d.png` (headless Chromium screenshots of the 3-D pages, the second
at t = 0.637 s), `sst_echo_level.html` and `flank_target_1km.html` (3-D pages loading three.js from the CDN),
`sst_echo_level.ch0.wav` (hydrophone 0, remodulated, 40 kHz, 0.59 s), `sst_echo_level.trace.json`. Not committed
(sizes): the 0.53 MB WAV of `flank_target_1km` hydrophone 0 (as is, 64 kHz) and the offline pages with three.js
embedded (0.71 and 0.76 MB).

- `sst_echo_level` (image tier, the TR 0702 closed-form geometry): 3 eigenrays a leg, 9 pairs in 6 delay groups;
  the six marked arrivals sit on the six compressed peaks on all four hydrophones.
- `flank_target_1km` (Bellhop): 22 eigenrays a leg, 484 pairs; all 44 matched to Bellhop's rays; the bottom bounces'
  losses lie on the Rayleigh curve from 0.6 dB at 7 deg to 9.2 dB at 46 deg.

What looks wrong or is weak:
- In `flank_target_1km` the record's per-channel waterfall shows reverberation and noise, not the echo (about 20 dB
  under the reverberation per channel); correct, but a reader may think the marks are misplaced. The echoes-alone
  panel beside it is the check. A beamformed panel is needed (V6).
- The out and back legs of a monostatic sonar coincide, so the dashed back legs hide under the solid ones.
- With 22 rays a leg the section is busy; the strongest 8 are coloured, the rest grey.
- The plan view's beam fan is analytic (section 0), and with the 4-element `sst_echo_level` array its beams are
  wider than the fan's spacing.
- The 256-colour PNG can blur thin coloured lines into grey; legend lines are drawn thicker for that reason.
- In the 3-D pages the semi-transparent surface and bed blend into a grey-brown when seen through each other; the
  hydrophones (3 m and 0.4 m arrays at 1 km scale) are too small to see the lighting at that zoom; the bed is drawn
  as a plane whatever the tier, labelled in the info panel when it is only the site's depth. Not verified visually:
  the play animation over time (only two slider positions were captured), resizing, and the CDN pages (the
  screenshots are of the embedded pages; the CDN pages differ only in where three.js comes from).
- Targets: one point per patch object (the strongest patch this ping), up to 6 a scenario; a hull's other echoes
  are not drawn as rays.

## 12. Packages V1 and V2 as done (2026-11-27)

What failed or is missing first:

- **Four scenarios have no picture on this CPU**: `flank_patrol_doppler`, `flank_patrol_doppler_tone`,
  `flank_patrol_doppler_underway`, `popoto_12km_doppler` (a 1 s tone and a sweep as one sampled pulse train,
  synthesised with its reverberation in NumPy) gave no page within 11 min (15 min for the first; ours). Their pages
  say so and name the command for the GPU job (`tools/scenario_view.py NAME --out DIR`, CUDA synthesis). Not thinned:
  the cost is the pulse and the reverberation, not a scene density. `popoto_far_mover_doppler` (538 s) made it.
- **`eigenray3d` is a projection.** Its section draws 2-D Bellhop rays on the 3-D bed's vertical section along each
  target's bearing, matched to Bellhop3D's eigenrays by bounce counts and labelled "PROJECTION" on the page; an
  eigenray with no 2-D ray of its bounce counts is not drawn and counted on the page (`flank_ridge_3d` 4 of 34,
  `flank_seamount_3d` 4 of 9; ours). Bellhop3D's own 3-D rays are V4.
- **The palette.** Pages are PNG with a FIXED 256-colour palette (128 greys and 13 tints of each hue the page uses,
  no dithering), not true colour: a thin coloured line keeps its hue (the prototype's adaptive palette turned it grey)
  at a third of the size (`flank_target_1km` 144 kB against 486 kB true colour; ours). `--truecolour` writes true
  colour. SVG was not smaller (the record pictures are rasters).
- **Not drawn yet**: the time-spread taps' own spread (V3); reverberation split by item (V3); the roughness-scattered
  field of `penetration: rough` (a sum over interface cells, not a ray; no scenario uses it); `{axis, position}`
  walls (no scenario uses them); a page's waterfall does not mark one-way arrivals (their emission times are the
  source's); the plan view's steered beams are 5 of the configured bearings.
- **Models no scenario uses** (counted 2026-11-27 by grep over `configs/scenarios` and `configs/environments`):
  `reverberation: site` (`boundary_scattering`), the Williams-Jackson and Mourad-Jackson opt-ins, `bubble_model` and
  the `bubble_layer_surface`/`kraus_surface` items, `penetration: rough`, layered and elastic beds. The models list
  names them when a configuration uses them (`test_models_no_scenario_uses_are_listed_when_configured`); no scenario
  was invented for them.

What a page shows now (`viz/page2d.py`, `viz/trace.py`, `viz/beams.py`, `viz/catalogue.py`):

- the ping it shows and why (`catalogue.view_plan`): the first; for a synthetic-aperture run with more than one ping
  the middle of the track (or of leg 0, or of the raster), simulated alone at its position; with platform `motion`
  the sway is a one-ping draw; a multiplexed array's ping holds one half. The plan view draws the whole track.
- rays per tier: straight (free field), folded images (flat), specular points among general planes with wall
  bounces (`fls_120k_quay_multipath`, `pool_fls_p4_2v_walls`), the refracted ray with its interface crossing and the
  evanescent region hatched below the critical angle (sediment), Bellhop's rays over a range-dependent bed (fjord,
  `_nx2d`), the 2-D projection of `eigenray3d`; one-way legs from each source (`sst_asiaex`, `asiaex_dahl_2004`); the
  projector's direct sound (`popoto_testauv_direct_overload*`).
- monostatic sonars: the back legs are drawn dashed and offset by 0.6 % of the depth span, said in the title.
  The 6 strongest eigenrays of each leg in colour, the rest thin and grey (`--all-rays`: all).
- boundaries and medium: the bed flat, tilted, by range or as a 3-D section; the surface's rms-height band, wind and
  bubble loss where `time_spread.surface` sets them; profiles by range (small multiples); losses per bounce against
  grazing for every model in force (fluid or elastic Rayleigh, constant, wall coefficients, the coherent surface
  loss exp(-Γ²/2) (J&R (13.14), confirmed, `acoustics.rough_coherent_reflection`), bubble loss, transmission into
  the sediment V_wp/sqrt(a_rho)) with the critical angle and the drawn bounces' angles; absorption in dB/km at the
  band centre; noise; the parts of the signal against time (echoes, direct sound, reverberation or basin tail or
  one-way sources, noise).
- beams: the processing's bearings (a deployment file naming the sonar, else the live chain's defaults; the gallery's
  fan for FLS, else the sonar's field of view; none for SAS) with 5 steered beams' -3 dB edges measured on the array
  factor of the real element positions with the chain's window times the element pattern at the band centre; the
  projector's pattern, or "kind point, omnidirectional".
- the list of models in force: area, model, key, value, source, status; and "not shown:" in the header.

`--all` on this CPU (4 cores shared with other agents; NumPy engine; Bellhop and Bellhop3D as separate processes;
ours, 2026-11-27): 92 of 96 pages; simulate + trace + page summed over the scenarios 10,758 s (3.0 h, of which the
four time-outs 2,880 s and the six `eigenray3d` scenes about 3,200 s). Pages 61 to 172 kB (mean about 110 kB); traces
12 kB to 48 MB (the 12 km scenes: 640 pairs x up to 96 channel delays); the directory 271 MB. Committed: one page a
family under `docs/figures/scenes/` with its index (section 6.3); the rest stay in the scratch directory.

## 13. Package V4 as done (2026-11-27): the 3-D page for every scenario

What failed, is missing or is uncertain first:

- **Four scenarios have 3-D pages without eigenrays**: `flank_patrol_doppler`, `flank_patrol_doppler_tone`,
  `flank_patrol_doppler_underway`, `popoto_12km_doppler` (no trace: their one-ping run timed out on this CPU, section
  12). Their pages are built from `trace.static_trace` (geometry, boundaries, beams, models; no simulation) and say
  "NO EIGENRAYS" with the GPU-job command: `PYTHONPATH=python python3 tools/scenario_view.py NAME --out DIR --3d`.
- **Bellhop3D's rays are found by shooting, not read from the provider's run.** Bellhop3D writes a ray file only for
  run type 'R' (`bellhop3D.f90` calls `WriteRay3D` for 'R' only; 'E' writes nothing in 3-D), and the provider runs
  'a' and 'I'. `viz/scene3d.py` runs Bellhop3D again ('R', a separate process, the provider's water, half-space, bed
  grid step and source key; nothing in `sim/bellhop3d.py` changed) on a small fan about each eigenray's launch angles,
  refined twice, and keeps the ray whose bounce counts before its closest approach to the target equal the eigenray's.
  The drawn ray is therefore Bellhop3D's own ray with the eigenray's bounces, passing near (not through) the target:
  over the six scenes 153 of 153 drawn eigenrays matched, misses 0.0 to 13.7 m at 1 to 10 km (ours). The launch
  azimuth was not in the cached traces (now added to the trace, `launch_azimuth_deg`): for them the fan is a band of
  +-6 deg about the target's bearing. Time: 9 to 80 s a scene (ours).
- **Bounces in the ray file**: a reflection is the same point written twice (`WriteRay3D`); counting those equals
  Bellhop3D's own counts on 697 of 697 rays over the seamount (ours). The loss per bounce is the same Rayleigh law
  evaluated again at the drawn grazing angle (section 0), not Bellhop3D's number.
- **The `eigenray` tier stays planar**: its pages say so (Bellhop 2-D rays placed on each target's bearing).
- **Not done**: sound (V5), the live chain's taps (V6; `chain` stays null in the page data), a link back from the 2-D
  PNG to the 3-D page (a PNG cannot link; the index links both), per-item reverberation and the time-spread taps'
  own spread (V3). Targets move by first-order motion (the label's mean velocity) over the pings; rays exist only for
  the traced ping, and the page says so when another ping is chosen.
- **The vendored three.js came from the npm registry, not a CDN**: jsdelivr and cdnjs refused this machine's request
  (HTTP 403 from the proxy); the registry's tarball was used, its SHA-512 equal to the registry's published integrity
  (third_party/three/README.md). If that route is not acceptable, delete `third_party/three/`: pages then load
  three.js from the CDN (`page3d.three_source`).

What a page shows (`viz/page3d.py`, its template `viz/page3d_template.html`, `viz/scene3d.py`):

- the sea surface (blue, labelled with its model; an rms-height band where time spreading sets one; faint and
  labelled "not in the propagation" for free field and sediment) and the bed (sand, shaded by depth, labelled with
  its half-space or constant coefficient and critical angle): flat; tilted and walls as planes (general planes);
  by range from the sonar (range-dependent `eigenray` beds); Bellhop3D's 3-D bed on a grid (`eigenray3d`); the
  sediment interface with a slab below; the site's depth dashed where the tier has no bottom;
- the projector, the hydrophones as screen-sized dots coloured by row, the array's axis and look as arrows, "zoom to
  array" (from behind the array along its look), the vertical exaggeration (auto, 1 to 60) with "true scale" one
  click away;
- the processing's receive bearings (shaded sector) and 5 steered beams with their -3 dB edges (`viz/beams.py`), the
  projector's -3 dB cone where it has a pattern;
- targets, their tracks and position at a chosen ping (ping slider), the platform's track;
- eigenrays (the 8 strongest drawable a leg), coloured by bounce class, out solid and back dashed, bounce dots;
  one-way sources and the projector's direct sound; per class, per leg, per target and per layer toggles;
- a click on a ray: delay, level, launch angle, bounce list (boundary, grazing, loss, position), the drawn ray's travel
  time and miss, the pairs that use it; a hydrophone selector (or a click on a hydrophone): its arrivals in time
  order, a click on one shows its two rays; a time slider and play that move a dot along every ray and light the
  hydrophones as the strongest pairs arrive (the hook for V5's sound);
- the sound-speed profile (profiles by range where the water changes with range), absorption, the models in force with
  their status words, and "not shown" (the trace's own list, the page's thinning, V5 and V6).

### 13.1 Sizes, checks, how to run

- `tools/scenario_view.py --all --3d --traces DIR --out DIR3` builds every 3-D page from a previous run's traces (no
  simulation): 96 pages in 3.0 min on this CPU (ours, 2026-11-27; the six Bellhop3D shootings 8 to 78 s each). Pages
  with three.js embedded (the default when `third_party/three/` holds it): 0.74 to 0.89 MB; loading it from the CDN
  (`--three cdn`): 44 to 200 kB (the same pages less the 691,648-byte library; ours). No page needed thinning beyond
  the 8 strongest rays a leg and 40 pairs (page data 3 to 159 kB against the 330 kB budget). Limits held by
  `test_page_sizes_stay_under_their_limits`: 400 kB (CDN) and 1.2 MB (embedded), on a trace far larger than any
  scenario's. The directory with traces' pages, screenshots and index: about 100 MB.
- `tools/page3d_check.js` (Node, Playwright's Chromium with SwiftShader WebGL) loaded all 96 embedded pages with no
  console error, page error or failed request, and screenshotted them (deviceScaleFactor 1). The CDN pages were not
  loaded here (the CDN refuses this machine); they differ from the embedded ones only in the loader line.
- Tests (`tests/python/test_scene_3d.py`, 108): the page data equal the trace at each tier (ray identity and order,
  delays, bounce counts and points, end points; the `eigenray` tier's rays planar on their bearing; general planes
  carried with their wall bounces); the Bellhop3D ray reader on a hand-made file in the documented format; Bellhop3D's
  rays shot about the image tier's eigenrays on a level bed bounce on the surface and the bed at the image's points
  (within 4 m), with the image's grazing angles (0.5 deg) and travel times (1 ms); an `eigenray3d` page draws the 3-D
  ray and says so; the vendored file is the stated one and only `viz/page3d.py` names it; every scenario yields a
  3-D page in quick form (static trace, no Bellhop3D); target motion over the pings; the pages load in Chromium
  (skipped when Node or Playwright is absent, as in the GPU job's image).
- Committed: `third_party/three/` (0.69 MB), and under `docs/figures/scenes/` one CDN page per family plus
  `flank_seamount_3d` (Bellhop3D's rays), each with a 128-colour screenshot `NAME_3d.png`, and the index with a 3-D
  column (1.9 MB in all). Everything else stays in the session scratch directory.
