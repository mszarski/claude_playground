# Realistic scenarios, mode by mode

Every mode of the suite has at least one scenario with an extended scene, run end to end:
simulate, condition, (estimate motion,) image, (detect and track). `tools/gallery.py`
does this for all of them and writes a figure and numbers.

- Full size, on a GPU: `GPUSONAR_JOB=gallery GPUSONAR_ORACLES=1 GPUSONAR_CERES=1 tools/gpu_job.sh`,
  or `tools/gallery.py --backend cuda` on a provisioned machine. The figures and numbers
  below are from an NVIDIA A10G (`docs/figures/gallery.json`).
- Thinned, on a CPU: `tools/gallery.py --quick` (same geometry and processing, fewer
  scatterers and pings). `tests/python/test_realistic_scenes.py` runs this for every
  scenario and checks each object shows where it should.

All of it is simulation. Scenes are clouds of point scatterers on object surfaces
(`DECISIONS.md` D-41): they give highlights, shadows, layover and speckle, not specular
glints, diffraction or resonances.

| Mode | Scenario | Sonar | Scene | Physics in the run | Run time (A10G) |
| --- | --- | --- | --- | --- | --- |
| FLS, pool, Phase 1 | `pool_fls_l12_5d` | L12-5, 5 MHz, both mux halves | far wall, side wall, cinder block, pipe, rope, two wires | free field, shadows, 12-bit ADC, noise | 4 s |
| FLS, pool, Phase 2 | `pool_fls_p4_2v` | P4-2v, 2.7 MHz | far wall, three steps, ladder, three lane floats | same | 3 s |
| FLS, sea, Phase 3 | `harbour_fls_750k` | 64 elements, 750 kHz | sandy floor, quay wall, six pilings, mooring block, rocks | free field, shadows, absorption | 3 s |
| FLS, sea, Phase 4 | `seafloor_objects_fls` | 128 elements, 300 kHz | seafloor, sphere, box, rocks | free field, shadows | 1 s |
| SAS, AUV | `wreck_sas` | 16 elements, 200 kHz | seafloor, 3 m box | free field, shadows, unknown sway and heave, motion estimation | 5 s |
| SAS, AUV | `minefield_sas` | same | rippled sand, two 2 m cylinders, sphere, rocks | same | 8 s |
| SAS, rail | `rail_objects` | P4-2v on the rail | pool floor, ball, pipe, brick, pebbles | free field, shadows, stop-and-go | 40 s |
| 3D, buried | `buried_objects` | sub-bottom, 20 kHz, raster | buried cylinder and sphere, rough interface, sediment clutter | refraction at the interface | 30 s |
| Flank array | `flank_patrol` | 32 hydrophones, 6–10 kHz | moving 56 m target, small vehicle, wreck | Bellhop multipath, bottom and surface reverberation, ambient noise, six pings, detection and tracking | 239 s |
| Popoto array | `popoto_trial_1000m` (also 100 m, 2000 m) | 8 hydrophones, 27.5 kHz | one +10 dB target | Bellhop multipath, bottom reverberation, ambient noise | 7 s |

Point-target scenarios for measuring resolution (`point_grid_*`, `stripmap_points`,
`rail_points`, `pool_wires_l12_5d`) and the long-range link-budget cases
(`flank_target_*km`) are separate; see `RESULTS.md`.

## Forward-looking sonar

FLS figures use Hann aperture weighting, a 30 log r range gain and the paper's dynamic
range compression for display. "Over median" is the mean image power over an object's
scatterers against the image median, after the range gain.

### Phase 1 pool: `pool_fls_l12_5d`

![pool_fls_l12_5d](figures/pool_fls_l12_5d.png)

| Object | Over median |
| --- | --- |
| Wire at 1 m / 2 m | 77.5 / 66.8 dB |
| Pipe (5 cm, vertical) | 48.0 dB |
| Cinder block | 42.6 dB |
| Far wall at 2.8 m | 31.0 dB |
| Side wall (grazing) | 13.1 dB |
| Rope | 3.8 dB |

- The far wall shows only from about −0.1 to +0.65 m across. Left of that it is in the
  cinder block's shadow (−0.74 to −0.08 m by geometry); towards both edges of the fan its
  return falls away.
- The mottled patch at left between 1.3 and 1.75 m has no object in it: it is the side
  wall seen through a grating lobe (the probe's pitch is one wavelength at 5 MHz).
- Only a 16 cm slab around the probe's depth is modelled (its elevation beam is about 4
  degrees), so there is no floor or surface multipath, which the plan expects to be
  strong in a real pool. Wall speckle is under-developed (150,000 scatterers per square
  metre against about 250,000 for one per resolution cell).

### Phase 2 pool: `pool_fls_p4_2v`

![pool_fls_p4_2v](figures/pool_fls_p4_2v.png)

Steps 34 to 41 dB, ladder 52 to 56 dB, floats 46 to 49 dB, far wall 48 dB over median.
The three floats, 0.6 m apart at 7.5 m, merge into one arc: the 19 mm aperture resolves
about 0.3 m there and the display is saturated by objects 50 dB above the background.
The steps stand against the wall and hide it on the left.

### Phase 3 harbour: `harbour_fls_750k`

![harbour_fls_750k](figures/harbour_fls_750k.png)

Quay wall 55.8 dB, pilings 56 to 58 dB, mooring block 38 dB, seafloor 35.7 dB over
median. The mooring block and the rocks cast long shadows at this 2 m altitude. The floor
is 60 scatterers per square metre, far below one per resolution cell, so it is a field of
separate points. The sonar config itself is a placeholder (`fls_750k`).

### Phase 4 seafloor: `seafloor_objects_fls`

![seafloor_objects_fls](figures/seafloor_objects_fls.png)

Sphere 43.4 dB, box 44.2 dB, rocks 35.3 dB, seafloor 28.2 dB over median; the floor behind
the sphere is more than 10 dB darker than beside it (`test_scene_objects.py`). One ping,
300 kHz, receive chain taken as already at baseband.

## Synthetic aperture sonar

### `wreck_sas` and `minefield_sas`

![wreck_sas](figures/wreck_sas.png)

![minefield_sas](figures/minefield_sas.png)

| Quantity | `wreck_sas` | `minefield_sas` |
| --- | --- | --- |
| Sway error after motion estimation, RMS | 0.089 λ | 0.080 λ |
| Heave error, RMS | 0.098 λ | 0.087 λ |
| Sharpness: navigation only / estimated / true motion | 69 / 272 / 279 | 220 / 321 / 319 |

- Navigation is not told about 10 mm RMS of sway (more than a wavelength) and 4 mm of
  heave; displaced-phase-centre estimation recovers both and the image matches the
  true-motion one.
- In `minefield_sas` the cylinder lying across track, the one along track, the sphere and
  the rocks each show a highlight towards the sonar and a shadow away from it; the sand
  ripples show as stripes because their slopes change the Lambert return.
- Tall objects are laid over towards the sonar and smeared along track (they are above
  the image plane). The seafloor is 25 to 30 scatterers per square metre: separate points,
  not developed speckle.

### Rail SAS: `rail_objects`

![rail_objects](figures/rail_objects.png)

Ball 29.7 dB, pipe 31.7 dB, brick 30.1 dB, pebbles 27.7 dB over median, each with a shadow
on the floor behind it. Rail positions are exact here (`rail_position_errors` has the
perturbed case, not yet imaged with correction).

## 3D imaging of buried objects: `buried_objects`

![buried_objects](figures/buried_objects.png)

| Quantity | True | Refraction on | Refraction off |
| --- | --- | --- | --- |
| Depth of the cylinder's brightest return | top at 0.25 m, axis at 0.30 m | 0.26 m | 0.23 m |
| Depth of the sphere's brightest return | top at 0.14 m | 0.14 m | 0.13 m |
| Cylinder over sediment clutter | — | 21.1 dB | — |

The sediment model is one refracted path with unit transmission: no sediment attenuation,
no angle-dependent transmission loss, so the clutter level relative to the objects is an
assumption (the scenario's `rms_amplitude`), not a prediction.

## Flank array: `flank_patrol`

![flank_patrol](figures/flank_patrol.png)

| Target | Result over six pings |
| --- | --- |
| Submarine-like target, 56 m, +16 dB in all, 3.0 m/s along x | tracked; velocity 3.05, −0.47 m/s; 30 m from its centre at the last ping |
| Wreck on the seabed, +12 dB, not moving | tracked; speed 0.03 m/s; 6 m from truth |
| Small vehicle, −3 dB, 2 m/s | **not detected** |

- 5 to 7 detections per ping gave 5 confirmed tracks for two objects: multipath splits
  the wreck's echo into several clusters, each of which starts a track. Clustering that
  merges them is not written (`STATUS.md`).
- Detection runs on power summed into 4 m range cells, with CFAR windows sized in metres,
  because multipath spreads each echo over tens of metres.
- The outcome depends on the assumed seabed (−27 dB Lambert), wind (5 m/s) and noise
  (45 dB); none is measured. Highlights have no aspect dependence, reverberation has no
  Doppler spread, and the platform stands still.

## Popoto array: `popoto_trial_1000m`

![popoto_trial_1000m](figures/popoto_trial_1000m.png)

Peak at 1000.1 m on the target's bearing, 30 dB over the map median, in multipath, bottom
reverberation and ambient noise. `RESULTS.md` has the 100 m and 2000 m cases and the
comparison with the spherical-spreading estimate.

## What no scenario has yet

- Surface and floor multipath in the pool scenes.
- A moving platform in the forward-looking scenes, or more than one ping of them.
- A flank or Popoto scene with a sound-speed profile other than isovelocity.
- Fully developed speckle on large surfaces (scatterer counts are kept affordable).
- Specular echoes, which dominate real returns from smooth pipes, plates and hulls.
