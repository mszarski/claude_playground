"""Target classes for range and depth planning, and a model figure for the target strength of a cylinder.

A target class is what a study is run against: a name, a target strength (one value or a [low, high] range), the
depths it may be at, the speeds it may move at. A build spec carries them under `targets:` (a list, or the name of a
file under configs/builds/targets/); `FIELDS` is the schema, `resolve` loads and completes them, `problems` checks them.

No measured target strength is built in. A class either states its figure (and whether anyone vouches for it:
`status: assumed | confirmed`), or gives a `cylinder: {length_m, diameter_m}` and takes the geometric-acoustics
figure for a rigid cylinder of that size (`status: model`):

    side, broadside, far field (range > L^2 / lambda):     TS = 10 log10(a L^2 / (2 lambda))
    side, at an angle t from broadside:                     x cos(t) sinc^2(k L sin(t))        (deep nulls; averaged over the band here)
    side, inside L^2 / lambda (a long hull at sonar range): the phase curvature along the hull limits the length that adds
                                                           in phase to about sqrt(lambda R / 2): TS -> 10 log10(a R / 4)
    ends:                                                   flat discs (`caps`), seen only near end-on

computed with the simulator's own shapes (`sim.targets.CylinderGlint` resolved along its length, `DiscGlint`), so the
figure is the one a scenario with that shape returns. It is the hull as a rigid, smooth, bare cylinder: a real hull
with a coating, a sail, fins and a tapered stern can be far lower at some aspects and higher at others, and is not
this. The figure is for sizing a study, not for predicting a vehicle.
"""

from __future__ import annotations

import warnings

import numpy as np

from .config import config_dir, load_yaml

# field -> (kind, unit, required, meaning, where a spec sheet has it, what to do if unknown)
FIELDS = {
    "name": ("str", "", True, "short name of the class (letters, digits, underscores): tables and file keys use it", "the requirement", "must be given"),
    "title": ("str", "", False, "one line for headings", "", "the name"),
    "target_strength_db": ("num or [low, high]", "dB re 1 m", False, "target strength in the sonar's band: one value, or the range over aspect. The planner works at the LOW end unless told otherwise",
                           "a measurement or a requirement document", "give `cylinder` and take the model figure, or a placeholder with status: assumed"),
    "cylinder": ("map", "", False, "{length_m, diameter_m, caps}: a rigid cylinder of that size stands in for the hull and the target strength is the geometric-acoustics figure for it "
                 "(gpusonar.target_classes.cylinder_ts_db), over `aspects_deg`", "the vehicle's outline", "give target_strength_db"),
    "aspects_deg": ("list", "deg", False, "aspects from broadside (0) to end-on (90) over which the cylinder's figure is taken; the range is its lowest and highest band-averaged value", "", "[0, 15, 30, 45, 60, 75]"),
    "depth_m": ("[low, high]", "m", True, "depths the target may be at (a surface ship: its hull's draught range)", "the requirement", "must be given"),
    "speed_m_s": ("[low, high]", "m/s", False, "speeds it may move at: the scenarios use the top of the range for a mover", "the requirement", "leave it out: no mover scenario is derived"),
    "status": (("assumed", "confirmed", "model"), "", False, "who vouches for the target strength: assumed (a placeholder for the owner to replace), confirmed (measured or stated by the owner), model (the cylinder figure)",
               "", "assumed"),
    "note": ("str", "", False, "where the figures come from, in words", "", "leave it out"),
}
DEFAULT_ASPECTS = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]


_CACHE: dict = {}


def cylinder_ts_db(length_m: float, diameter_m: float, aspect_deg, f_lo: float, f_hi: float, range_m: float | None = None, caps: bool = True,
                   sound_speed: float = 1500.0, n_f: int = 21) -> np.ndarray:
    key = (float(length_m), float(diameter_m), tuple(np.atleast_1d(np.asarray(aspect_deg, dtype=float)).tolist()), float(f_lo), float(f_hi), range_m, bool(caps), float(sound_speed), int(n_f))
    if key not in _CACHE:
        _CACHE[key] = _cylinder_ts_db(*key[:2], np.array(key[2]), *key[3:])
    return _CACHE[key].copy()


def _cylinder_ts_db(length_m, diameter_m, aspect_deg, f_lo, f_hi, range_m, caps, sound_speed, n_f) -> np.ndarray:
    """Band-averaged monostatic target strength (dB re 1 m) of a rigid cylinder at aspects from broadside (deg), seen
    from `range_m` (None: the far field, taken as 50 L^2 / lambda), with the simulator's glint shapes. Averaged in power
    over `n_f` frequencies of the band: single-frequency nulls off broadside are filled, as a wideband pulse fills them."""
    from .sim import targets as shapes
    length, a = float(length_m), 0.5 * float(diameter_m)
    aspects = np.atleast_1d(np.asarray(aspect_deg, dtype=float))
    freqs = np.linspace(float(f_lo), float(f_hi), int(n_f))
    lam_min = sound_speed / float(f_hi)
    rng = float(range_m) if range_m is not None else 50.0 * length**2 / lam_min
    item = {"shape": "cylinder", "start": [-0.5 * length, 0.0, 0.0], "end": [0.5 * length, 0.0, 0.0], "radius": a, "caps": bool(caps), "resolved": True, "spacing": 0.25 * lam_min}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parts = shapes.shapes_from_config(item)
    out = np.zeros(aspects.shape)
    for i, asp in enumerate(aspects):
        t = np.radians(asp)
        sonar = rng * np.array([np.sin(t), np.cos(t), 0.0])                  # broadside: across the axis
        power = 0.0
        for f in freqs:
            k = 2.0 * np.pi * f / sound_speed
            total = 0.0 + 0.0j
            for shape in parts:
                points, amp = shape.evaluate(sonar, sonar, k)
                r = np.linalg.norm(points - sonar, axis=1)
                total += np.sum(amp * np.exp(-2j * k * (r - rng)) * (rng / r) ** 2)
            power += abs(total) ** 2
        out[i] = 10.0 * np.log10(max(power / len(freqs), 1e-30))
    return out


def cylinder_closed_forms(length_m: float, diameter_m: float, frequency: float, range_m: float, sound_speed: float = 1500.0) -> dict:
    """The textbook limits the numbers above are held to: broadside in the far field, broadside inside L^2 / lambda, the flat end."""
    lam, a, length = sound_speed / frequency, 0.5 * diameter_m, float(length_m)
    return {"broadside_far_field_db": float(10.0 * np.log10(a * length**2 / (2.0 * lam))), "broadside_near_field_db": float(10.0 * np.log10(a * range_m / 4.0)),
            "far_field_beyond_m": float(length**2 / lam), "end_on_disc_db": float(20.0 * np.log10(np.pi * a**2 / lam)),
            "broadside_lobe_deg": float(np.degrees(lam / (2.0 * length)))}


def load(spec_value) -> list:
    """The `targets` value of a build spec as a list of mappings: a list as written, or the classes of the named file
    under configs/builds/targets/ (its `targets:` list)."""
    if spec_value is None:
        return []
    if isinstance(spec_value, str):
        path = config_dir() / "builds" / "targets" / f"{spec_value}.yaml"
        if not path.exists():
            known = sorted(q.stem for q in (config_dir() / "builds" / "targets").glob("*.yaml"))
            raise FileNotFoundError(f"no target-class file {spec_value!r}; known: {', '.join(known)}")
        return list(load_yaml(path).get("targets") or [])
    return list(spec_value)


def problems(classes) -> list:
    """(field, message) for everything wrong with a list of target classes, in the spec validator's words."""
    out = []
    if not isinstance(classes, list):
        return [("targets", "must be a list of classes ({name, target_strength_db or cylinder, depth_m, ...}) or the name of a file under configs/builds/targets/")]
    seen = set()
    for i, t in enumerate(classes):
        where = f"targets[{i}]"
        if not isinstance(t, dict):
            out.append((where, "must be a mapping"))
            continue
        name = t.get("name")
        where = f"targets.{name}" if isinstance(name, str) else where
        for key in t:
            if key not in FIELDS:
                out.append((f"{where}.{key}", f"unknown field; fields are {', '.join(FIELDS)}"))
        if not isinstance(name, str) or not name.replace("_", "").isalnum():
            out.append((f"{where}.name", "give the class a name of letters, digits and underscores"))
        elif name in seen:
            out.append((f"{where}.name", "two classes have this name"))
        seen.add(name)
        ts, cyl = t.get("target_strength_db"), t.get("cylinder")
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
        pair = lambda v: isinstance(v, list) and len(v) == 2 and all(num(x) for x in v) and v[0] <= v[1]  # noqa: E731
        if ts is None and cyl is None:
            out.append((f"{where}.target_strength_db", "missing: give a target strength (dB re 1 m; one value or [low, high]) or a `cylinder: {length_m, diameter_m}` for the model figure"))
        if ts is not None and not (num(ts) or pair(ts)):
            out.append((f"{where}.target_strength_db", f"is {ts!r}; it must be a number or [low, high] with low <= high (dB re 1 m)"))
        elif ts is not None and not (-60.0 <= float(np.min(ts)) and float(np.max(ts)) <= 80.0):
            out.append((f"{where}.target_strength_db", f"{ts!r} dB is outside anything usual (-60 to 80 dB re 1 m)"))
        if cyl is not None and not (isinstance(cyl, dict) and num(cyl.get("length_m")) and num(cyl.get("diameter_m")) and cyl["length_m"] > 0 and 0 < cyl["diameter_m"] <= cyl["length_m"]):
            out.append((f"{where}.cylinder", "give {length_m, diameter_m} in metres, the diameter no more than the length"))
        if not pair(t.get("depth_m")) or t["depth_m"][0] < 0:
            out.append((f"{where}.depth_m", f"is {t.get('depth_m')!r}; give [shallowest, deepest] in metres below the surface"))
        if t.get("speed_m_s") is not None and not (pair(t["speed_m_s"]) and t["speed_m_s"][0] >= 0):
            out.append((f"{where}.speed_m_s", f"is {t['speed_m_s']!r}; give [slowest, fastest] in m/s"))
        if t.get("status") is not None and t["status"] not in FIELDS["status"][0]:
            out.append((f"{where}.status", f"is {t['status']!r}; one of {', '.join(FIELDS['status'][0])}"))
        if t.get("status") == "confirmed" and ts is None:
            out.append((f"{where}.status", "confirmed, but no target strength is given: a cylinder's model figure is not a confirmed one"))
        if t.get("aspects_deg") is not None and not (isinstance(t["aspects_deg"], list) and t["aspects_deg"] and all(num(x) and 0 <= x <= 90 for x in t["aspects_deg"])):
            out.append((f"{where}.aspects_deg", "give a list of aspects from broadside, 0 to 90 degrees"))
    return out


def resolve(classes, f_lo: float, f_hi: float, range_m: float = 12000.0, sound_speed: float = 1500.0) -> list:
    """Each class completed: `ts_db` [low, high] (the stated figure, or the cylinder's over its aspects at `range_m`),
    `status`, and for a cylinder the table `by_aspect` and the closed-form limits."""
    out = []
    for t in classes:
        c = dict(t)
        ts, cyl = t.get("target_strength_db"), t.get("cylinder")
        if cyl is not None:
            aspects = [float(x) for x in (t.get("aspects_deg") or DEFAULT_ASPECTS)]
            near = cylinder_ts_db(cyl["length_m"], cyl["diameter_m"], aspects, f_lo, f_hi, range_m, bool(cyl.get("caps", True)), sound_speed)
            far = cylinder_ts_db(cyl["length_m"], cyl["diameter_m"], aspects, f_lo, f_hi, None, bool(cyl.get("caps", True)), sound_speed)
            c["by_aspect"] = [{"aspect_deg": a, "ts_db": float(v), "ts_far_field_db": float(w)} for a, v, w in zip(aspects, near, far)]
            c["closed_forms"] = cylinder_closed_forms(cyl["length_m"], cyl["diameter_m"], 0.5 * (f_lo + f_hi), range_m, sound_speed)
            c["model_range_m"] = float(range_m)
        if ts is not None:
            lo, hi = (float(ts), float(ts)) if not isinstance(ts, list) else (float(ts[0]), float(ts[1]))
            c["status"] = t.get("status", "assumed")
        else:
            lo, hi = float(np.min(near)), float(np.max(near))
            c["status"] = "model"
        c["ts_db"] = [lo, hi]
        c["depth_m"] = [float(x) for x in t["depth_m"]]
        out.append(c)
    return out
