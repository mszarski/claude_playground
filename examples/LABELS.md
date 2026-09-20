# Generating labelled sonar images

Every picture the sequence examples render is labelled in the forward pass:
a bounding box, an instance mask, a centroid, a peak, a contrast and a class
for every target and every emission in it, read off the fields rather than
found by a detector (`hydropt/labels.py`; the README's section "Labels from
the forward pass" says how and why).  This is how to make them.

## The two examples that write labels

```bash
cd examples
python 28_boat_sequence.py                 # a boat under way past a fixed sonar
python 29_ownship_sequence.py              # the sonar under way past still things
```

Each writes, per scenario, into `figures/` (gitignored: regenerate, do not
commit):

| file | what |
| --- | --- |
| `28_sequence_<quiet|emission>.gif`, `.png` | the frames with the boxes drawn: solid = signal box, dashed = geometry box |
| `28_labels_<quiet|emission>.json` | one record per frame: `t`, the boat's `pose`, and its `labels` |
| `29_ownship_<boat|buoy|kelp>.gif`, `.png` | the same for the ownship sequences |
| `29_labels_<boat|buoy|kelp>.json` | one record per ping: `t`, the `ownship_pose`, the `labels` |

Switches: `HYDROPT_SONAR=330` renders the 330 kHz, 1.4 x 2.8 degree head
(files tagged `_330k`); `HYDROPT_SCENARIO` picks one scenario;
`HYDROPT_FRAMES` the number of pings (16 for 28, 12 for 29); `HYDROPT_FAR`,
`HYDROPT_NEAR` the swath.  The images the labels refer to are the frames'
displayed pictures in metres (300 x 300 cells, `x` forward, `y` to port, the
sonar at the origin), 21's median-TVG display at the first frame's gain.

## A label record

```json
{"name": "boat hull", "kind": "target", "visible": true, "contrast_db": 41.3,
 "n_cells": 212,
 "polar_box": [63, 71, 402, 438],
 "box_m": [231.4, -114.9, 253.0, -96.2],
 "centroid_m": [243.1, -105.6], "peak_m": [244.5, -104.0],
 "geometry_box_m": [227.0, -118.4, 257.9, -93.1]}
```

* `polar_box` is `[beam0, beam1, bin0, bin1]`, inclusive, on the sonar's own
  rectangle (181 beams across +/-60 degrees, 0.5 m bins from `NEAR` to
  `FAR` at 120 kHz; 361 beams at 330 kHz): the form to use for a network
  that works on the beamformed image rather than the resampled one, and the
  tight one for a spoke (one beam wide, every bin long).
* `box_m` is `[x0, y0, x1, y1]` in metres in the Cartesian picture, the box
  round the masked cells' own corners.
* `geometry_box_m` is the object's physical extent (every vertex of its
  mesh, every highlight of a point cloud) dilated by one beam width and one
  range cell; `null` for an emission, which has no body.
* `visible` is a mask of at least two cells with the peak over the 3 dB
  margin.  A shadowed or faint target keeps its record, its class and its
  geometry box, with `visible: false` and no signal box: the negative
  examples come out of the same run.
* `kind` is `"target"` or `"emission"`; the class is `name`.

Masks are not written to the JSON (they are the size of the picture); they
are on the `Label` objects a run holds -- `polar_mask` `[beams, bins]` and
`mask` on the Cartesian grid -- and a dataset that wants them writes them
from there.

## Labelling a scene of your own

```python
from hydropt import PictureRenderer, Trajectory, draw_labels

renderer = PictureRenderer(scene, ...)            # 28 and 29 show the arguments, all 21's
target = mesh_target(verts, faces, position=(x, y, 0.0), yaw=h, ...)
target.label = "boat hull"                        # the class; targets sharing one are one label
picture, labels = renderer.picture([target], labels=True)
for lab in labels:
    print(lab.name, lab.visible, lab.box_m, lab.polar_box, lab.contrast_db)
records = [lab.to_dict() for lab in labels]       # JSON-ready

for t, pose, picture, labels in renderer.sequence(builder, trajectory, times,
                                                  emitters=[propeller], labels=True):
    ...                                           # the moving-target case (28)
for t, pose, picture, labels in renderer.ownship_sequence(world, trajectory, times,
                                                          scene_at=scene_at, labels=True):
    ...                                           # the moving-sonar case (29)
```

An emitter is a callable of the pose returning arrivals
(`emission_arrivals`), with a `label` attribute for its class (`"noise
spoke"`).  `draw_labels(ax, labels)` puts the boxes on a matplotlib axis in
metres.  A target with no `label` attribute is labelled by its class name.

## What decides a label, and the knobs

A cell belongs to a target's mask when the target's own beamformed power
stands over everything else in the cell -- the reverberation, the other
targets, the receiver noise -- by `label_margin_db` (3 dB, a
`PictureRenderer` argument), is within 35 dB of the target's own brightest
cell (`dynamic_range_db` in `label_from_beams`: a strong echo's sidelobe
ring is 43 dB down and is not the target), and lies within the target's
geometry plus two beams and a few range cells (`polar_gate`).  The
resolution the geometry box is dilated by comes from the renderer
(`beam_deg`, by default `101.5 / n` times 1.3 under a shading window, and
the pulse's half-power width in metres).  Change the margin to change what
"seen" means; change nothing else to relabel a scene.
