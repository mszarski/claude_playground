"""Real reference motion for the generator: ``[(name, move)]`` lists in the move format.

Sources:

- ``local(dir)``: a folder of move JSON files, e.g. a snapshot of Pollen's emotions library
  (https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library, 85 clips).
- ``hub(repo)``: the same, downloaded from the Hugging Face Hub (``hf`` extra; needs network access to huggingface.co).
- ``dances()``: Pollen's dance moves rendered procedurally from the ``reachy-mini-dances-library`` package
  (the source of the Hub's dances library). Install it with ``pip install --no-deps reachy-mini-dances-library``:
  the collection needs only numpy, and the package's full dependency (the robot SDK) is not needed here.

Reference: ``common/data.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import copy
import glob
import importlib
import importlib.util
import json
import os
import re
import sys
import types

import numpy as np

from .motion import FPS, rpy_to_mat

EMOTIONS = "pollen-robotics/reachy-mini-emotions-library"
DANCES = "pollen-robotics/reachy-mini-dances-library"

# Emotions kept out of generator training and used for evaluation; they span sad/low, angry/high-energy,
# positive and calm, so the held-out score means something.
HELD_OUT_EMOTIONS = ["disgusted1", "downcast1", "electric1", "exhausted1", "frustrated1", "lonely1",
                     "rage1", "relief1", "surprised1", "thoughtful1", "welcoming1", "impatient1"]
# Used when training on dances only (no emotions available): a nod, a sway and a transient gesture.
HELD_OUT_DANCES = ["yeah_nod@114", "groovy_sway_and_roll@114", "side_glance_flick@114"]


def _project(moves, project):
    if not project:
        return moves
    from .reach import Reach   # ~5% of emotion-library frames are not reachable as recorded

    R = Reach()
    return [(n, R.project(copy.deepcopy(m))[0]) for n, m in moves]


def local(root, project=True):
    """Every move JSON in ``root`` -> ``[(name, move)]``, optionally projected onto the reachable set."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*.json"))):
        with open(p) as f:
            m = json.load(f)
        if "set_target_data" in m:
            out.append((os.path.basename(p)[:-5], m))
    return _project(out, project)


def hub(repo, project=True, cache_dir=None):
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo, repo_type="dataset", allow_patterns=["*.json"], cache_dir=cache_dir)
    return local(root, project)


def _dance_collection():
    """``AVAILABLE_MOVES`` without importing the package's ``__init__`` (which pulls in the robot SDK)."""
    name = "reachy_mini_dances_library"
    try:
        return importlib.import_module(f"{name}.collection.dance").AVAILABLE_MOVES
    except ImportError:
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise ImportError("pip install --no-deps reachy-mini-dances-library") from None
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[mod]
        pkg = types.ModuleType(name)
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules[name] = pkg
        return importlib.import_module(f"{name}.collection.dance").AVAILABLE_MOVES


def dances(bpms=(100, 114, 128), beats=16, project=True):
    """Each dance move rendered at 25 Hz for ``beats`` beats at each tempo in ``bpms``.
    Names are ``"<move>@<bpm>"``. Antennas follow the SDK's ``DanceMove`` (offsets used as positions)."""
    out = []
    for move_name, (fn, params, _) in sorted(_dance_collection().items()):
        for bpm in bpms:
            t = np.arange(int(round(beats * 60 / bpm * FPS))) / FPS
            frames = []
            for ts in t:
                o = fn(ts * bpm / 60, **params)
                H = np.eye(4)
                roll, pitch, yaw = o.orientation_offset
                H[:3, :3] = rpy_to_mat(np.array([[roll, pitch, yaw]]))[0]
                H[:3, 3] = o.position_offset
                frames.append({"head": H.tolist(), "antennas": [float(a) for a in o.antennas_offset], "body_yaw": 0.0})
            out.append((f"{move_name}@{bpm}", {"description": move_name.replace("_", " "), "time": t.tolist(),
                                                "set_target_data": frames}))
    return _project(out, project)


def caption(name, move):
    """Library clip -> prompt, e.g. ``("downcast1", move) -> "downcast. <the clip's description>"``."""
    base = re.sub(r"\d+$", "", name.split("@")[0]).replace("_", " ")
    return f"{base}. {move.get('description', '')}".strip()
