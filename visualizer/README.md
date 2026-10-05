# Browser viewer

Plays Reachy Mini moves in 3D, in the browser: a gallery of this recreation's results, plus any move JSON you drop
on the page (the SDK's recorded-move format, e.g. a pipeline output or a clip from Pollen's libraries).

```bash
python scripts/build_gallery.py --ckpt hf://mszarski/reachy-motion-generator/generator.pt   # examples/examples.json
python -m http.server -d visualizer 8000                                                     # open localhost:8000
```

Hosted copy: the private Space [mszarski/reachy-mini-motions](https://huggingface.co/spaces/mszarski/reachy-mini-motions)
(a static Space holding this folder plus a README with the Space header; re-upload the folder to update it).

`python -m rmr.viewer runs/one="my prompts"` builds the gallery from any pipeline output folders instead.
`?prompt=sneezing` opens the first entry whose prompt starts with that text.

The gallery shipped here holds, for the 16 probe prompts and the 12 held-out emotions, one motion from the
fine-tuned Qwen3.5-4B planner and one from the zero-shot Kimi-K3 planner (both through our generator), and Pollen's
real clip of each held-out emotion.

How it works: each frame's head pose, antennas and body yaw become the six Stewart-platform motor angles through a
JavaScript port of the SDK's IK (`src/StewartIK.js`; `tests/test_viewer.py` checks it against the Rust IK to 1e-9 rad),
and the passive joints follow from the 3D model's kinematics. three.js, urdf-loader and the Draco decoder load from
cdn.jsdelivr.net.

Adapted from the reference's `visualizer/` (pham-tuan-binh/reachy-motion-generator); 3D model and passive-joint
kinematics from [8bitkick/reachy_mini_3d_web_viz](https://huggingface.co/spaces/8bitkick/reachy_mini_3d_web_viz).
All Apache-2.0: see `LICENSE-reachy_mini_3d_web_viz.txt` and the repository's NOTICE.
