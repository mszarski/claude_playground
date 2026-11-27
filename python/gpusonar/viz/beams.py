"""The receive beams a page draws: the processing's own bearings and the array's real pattern at the band centre, and
the projector's pattern where the sonar gives it one (docs/tasks/VISUALIZATION.md section 0 and V1).

Steered receive beams are not in the simulator: they belong to the processing. Where the bearings come from, per mode:

- `flank` (the flank array and every Popoto build): the live chain. A deployment file under configs/deploy whose
  source names the scenario's sonar gives `chain.bearings_deg`, `look` and `window`; otherwise the live chain's
  defaults (`live.DEFAULTS`: -40 to 40 deg from broadside by 0.5, Hann, starboard).
- `fls`: the fan the gallery images the scene with (tools/gallery.py, its `FLS` table, or -60 to 60 deg by 0.4 for
  the `FLS_TRANSIT` scenes; read from the file as text, not imported); otherwise +- half the sonar's
  `fov.horizontal_deg`; otherwise none.
- `sas`: none. A synthetic-aperture image is backprojected along the track (recon.sas_image); the page shows the
  element and projector footprints instead.

The pattern: the array factor of the sonar's own element positions projected on the array's axis, with the chain's
aperture window (Hann or none), steered to a bearing, at the band centre, times the element's power pattern (the
separable sinc the simulator applies, sim/reverb.py `element_power_pattern`), in the horizontal plane. The -3 dB width
of each steered beam is measured on that pattern (ours).
"""

from __future__ import annotations

import ast
import glob
import os

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _gallery_fls_bearings() -> dict:
    """{scenario: (first, last, step) deg} of tools/gallery.py, read from the file (literal dicts), never imported."""
    path = os.path.join(_ROOT, "tools", "gallery.py")
    out = {}
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except OSError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in ("FLS", "FLS_TRANSIT"):
            name = node.targets[0].id
            for k, v in zip(node.value.keys, node.value.values):
                key = ast.literal_eval(k)
                if name == "FLS":
                    for kw in v.keywords:
                        if kw.arg == "bearings":
                            out[key] = tuple(float(x) for x in ast.literal_eval(kw.value))
                else:
                    out[key] = (-60.0, 60.0, 0.4)           # run_fls_transit: np.arange(-60.0, 60.01, 0.4)
    return out


def _deploy_chain(sonar_name: str):
    """(path, chain block) of the deployment file whose source is `sonar_name`, or (None, None)."""
    from ..config import load_yaml
    for path in sorted(glob.glob(os.path.join(_ROOT, "configs", "deploy", "*.yaml"))):
        try:
            cfg = load_yaml(path)
        except Exception:                                    # noqa: BLE001  (a picture never stops on a file)
            continue
        if (cfg.get("source") or {}).get("sonar") == sonar_name:
            return os.path.relpath(path, _ROOT), cfg.get("chain") or {}
    return None, None


def chain_bearings(scenario: str, sonar) -> dict:
    """{bearings_deg: [first, last, step] or None, look, window, source: where they come from}."""
    mode = sonar.config.get("mode")
    if mode == "flank":
        from .. import live
        path, chain = _deploy_chain(sonar.config.get("name"))
        base = dict(live.DEFAULTS)
        if chain:
            b = chain.get("bearings_deg", base["bearings_deg"])
            return {"bearings_deg": [float(x) for x in b], "look": chain.get("look", base["look"]),
                    "window": chain.get("window", base["window"]), "source": f"live chain, {path}"}
        return {"bearings_deg": [float(x) for x in base["bearings_deg"]], "look": base["look"], "window": base["window"],
                "source": "live chain defaults (live.DEFAULTS); no deployment file names this sonar"}
    if mode == "fls":
        g = _gallery_fls_bearings()
        if scenario in g:
            return {"bearings_deg": list(g[scenario]), "look": "ahead", "window": "hann",
                    "source": "the fan tools/gallery.py images this scene with"}
        fov = (sonar.config.get("fov") or {}).get("horizontal_deg")
        if fov:
            h = 0.5 * float(fov)
            return {"bearings_deg": [-h, h, 0.5], "look": "ahead", "window": "hann",
                    "source": "the sonar's fov.horizontal_deg (no chain configured for this scene; step ours)"}
        return {"bearings_deg": None, "look": "ahead", "window": "hann",
                "source": "no chain configured for this scene and no fov in the sonar"}
    return {"bearings_deg": None, "look": None, "window": None,
            "source": "synthetic aperture: no steered receive beams; the image is backprojected along the track"}


def _sinc_power(u, bw):
    return 1.0 if not np.isfinite(bw) else np.sinc(0.886 * u / bw) ** 2


def beam_part(scenario: str, sonar, record, sound_speed: float, n_steer: int = 5) -> dict:
    """What the plan view and the beam-pattern panel draw (see the module docstring). Angles are from the array's
    broadside in the horizontal plane, positive towards the array's +axis."""
    from .. import _gpusonar as core
    fc = float(sonar.center_frequency)
    lam = float(sound_speed) / fc
    k = 2.0 * np.pi / lam
    chain = chain_bearings(scenario, sonar)
    rot = np.asarray(core.rotation_ned(*record.geometry.rx_pose.rpy), float)
    pos = np.asarray(sonar.positions, float)
    group = list(record.metadata.channel_element) or list(range(len(pos)))
    pos = pos[group]
    bore = rot @ np.asarray(sonar.boresight, float)
    bore_h = np.array([bore[0], bore[1], 0.0])
    if np.linalg.norm(bore_h) < 1e-9:                         # looking straight down: the array axis decides
        bore_h = np.array([1.0, 0.0, 0.0])
    bore_h /= np.linalg.norm(bore_h)
    if chain.get("look") == "port":
        bore_h = -bore_h
    axis_h = np.array([-bore_h[1], bore_h[0], 0.0])           # 90 deg clockwise from the look (NED: towards +y of it)
    x = (rot @ pos.T).T @ axis_h
    x = x - x.mean()
    n = len(x)
    w = np.hanning(n + 2)[1:-1] if (chain.get("window") == "hann" and n > 2) else np.ones(n)
    order = np.argsort(x)
    w_sorted = np.empty(n)
    w_sorted[order] = w
    th = np.radians(np.linspace(-90.0, 90.0, 1441))
    elem = np.asarray(_sinc_power(np.sin(th), float(sonar.beamwidth_az)), float) * np.ones_like(th)
    steer = []
    b = chain.get("bearings_deg")
    if b is not None and n > 1:
        first, last = float(b[0]), float(b[1])
        for s in np.linspace(first, last, n_steer):
            af = np.abs(np.exp(1j * k * np.outer(np.sin(th) - np.sin(np.radians(s)), x)) @ w_sorted) ** 2
            p = af / af.max() * elem
            db = 10.0 * np.log10(np.maximum(p / p.max(), 1e-12))
            i0 = int(np.argmax(db))
            lo = i0
            while lo > 0 and db[lo] > -3.0:
                lo -= 1
            hi = i0
            while hi < len(th) - 1 and db[hi] > -3.0:
                hi += 1
            steer.append({"steer_deg": float(s), "peak_deg": float(np.degrees(th[i0])),
                          "edges_deg": [float(np.degrees(th[lo])), float(np.degrees(th[hi]))],
                          "pattern_db": np.round(db[::4], 2).tolist()})
    tx = record.geometry.tx
    txc = sonar.config["transmit"]
    kind = txc.get("kind", "point")
    has_tx = np.isfinite(tx.beamwidth_az) or np.isfinite(tx.beamwidth_el)
    tx_bore = rot @ np.asarray(tx.boresight, float)
    proj = None
    if has_tx:
        # The projector's power pattern in the horizontal plane through its boresight's azimuth (the same sinc).
        tb = np.array([tx_bore[0], tx_bore[1], 0.0])
        tb = tb / np.linalg.norm(tb) if np.linalg.norm(tb) > 1e-9 else bore_h
        rel = float(np.degrees(np.arctan2(np.cross(bore_h, tb)[2], bore_h @ tb)))
        proj = {"pattern_db": np.round(10.0 * np.log10(np.maximum(
            np.asarray(_sinc_power(np.sin(th - np.radians(rel)), float(tx.beamwidth_az)), float) * np.ones_like(th), 1e-12))[::4], 2).tolist(),
            "boresight_rel_deg": rel, "beamwidth_az_deg": float(np.degrees(tx.beamwidth_az)) if np.isfinite(tx.beamwidth_az) else None,
            "beamwidth_el_deg": float(np.degrees(tx.beamwidth_el)) if np.isfinite(tx.beamwidth_el) else None,
            "boresight": tx_bore.tolist()}
    if kind == "point" and not has_tx:
        proj_text = "projector: kind point, omnidirectional (the sonar gives it no beamwidth or aperture)"
    elif kind == "point":
        proj_text = ("projector: kind point with a beam pattern (separable sinc, -3 dB "
                     + ", ".join(f"{a} {v:.1f} deg" for a, v in (("az", proj["beamwidth_az_deg"]), ("el", proj["beamwidth_el_deg"])) if v)
                     + ")")
    else:
        proj_text = f"projector: kind {kind}"
    return {"center_frequency": fc, "wavelength": lam, "chain": chain, "angles_deg": np.round(np.degrees(th[::4]), 3).tolist(),
            "element_db": np.round(10.0 * np.log10(np.maximum(elem[::4], 1e-12)), 2).tolist(),
            "element_beamwidth_az_deg": float(np.degrees(sonar.beamwidth_az)) if np.isfinite(sonar.beamwidth_az) else None,
            "steered": steer, "projector": proj, "projector_text": proj_text, "projector_kind": kind,
            "broadside": bore_h.tolist(), "axis": axis_h.tolist(), "n_elements": int(n),
            "aperture_m": float(x.max() - x.min()) if n > 1 else 0.0}
