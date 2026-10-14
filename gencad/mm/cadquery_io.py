"""CadQuery side of the CAD pipeline: DeepCAD programs -> CadQuery code (CAD-Recode style), running code, IoU.

CAD-Recode (filapro/cad-recode-v1.5) writes parts as short CadQuery programs with integer coordinates, e.g.

  import cadquery as cq
  w0=cq.Workplane('XY',origin=(0,0,11))
  r=w0.sketch().segment((16,-20),(19,-17)).segment((16,-14)).close().assemble().finalize().extrude(-111)

`deepcad_to_cadquery` writes our DeepCAD programs in the same style, so one model can learn from both datasets:
each part is centred and scaled so its bounding box spans 200 units on its longest side (like CAD-Recode's
[-100, 100]), sketches are drawn on the named plane that shares their normal ('XY', 'YZ', 'ZX' or a reversed one)
when there is one, and coordinates are rounded to integers.

`install_occ_shim` lets upstream cadlib (written for pythonocc, `OCC.Core.*`) build its solids with CadQuery's OCP
bindings, so reference shapes and CadQuery shapes live in the same OpenCascade.
"""
from __future__ import annotations

import sys
import types

import numpy as np

NAMED_PLANES = {   # CadQuery named planes: (xDir, normal)
    "XY": ((1, 0, 0), (0, 0, 1)), "YZ": ((0, 1, 0), (1, 0, 0)), "ZX": ((0, 0, 1), (0, 1, 0)),
    "XZ": ((1, 0, 0), (0, -1, 0)), "YX": ((0, 1, 0), (0, 0, -1)), "ZY": ((0, 0, 1), (-1, 0, 0)),
}
SPAN = 200.0


def install_occ_shim():
    """Make `import OCC.Core.<mod>` resolve to OCP.<mod> (only if pythonocc is not installed)."""
    try:
        import OCC.Core.gp  # noqa: F401
        return
    except ImportError:
        pass
    import importlib
    occ, core, extend = types.ModuleType("OCC"), types.ModuleType("OCC.Core"), types.ModuleType("OCC.Extend")
    occ.Core, occ.Extend = core, extend
    sys.modules.update({"OCC": occ, "OCC.Core": core, "OCC.Extend": extend})
    for name in ["gp", "BRepBuilderAPI", "BRepPrimAPI", "BRepAlgoAPI", "GC", "Bnd", "TopoDS", "TopAbs", "TopExp",
                 "BRepGProp", "GProp", "BRep", "BRepMesh", "TopLoc"]:
        mod = importlib.import_module(f"OCP.{name}")
        sys.modules[f"OCC.Core.{name}"] = mod
        setattr(core, name, mod)
    # OCP's Shape.Reversed() returns a TopoDS_Shape where pythonocc returns the subtype: cast for MakeFace.Add
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.TopoDS import TopoDS

    class _MakeFace(BRepBuilderAPI_MakeFace):
        def Add(self, wire):
            return super().Add(TopoDS.Wire_s(wire))
    bba = types.ModuleType("OCC.Core.BRepBuilderAPI")
    bba.__dict__.update(importlib.import_module("OCP.BRepBuilderAPI").__dict__)
    bba.BRepBuilderAPI_MakeFace = _MakeFace
    sys.modules["OCC.Core.BRepBuilderAPI"] = bba
    bbl = types.ModuleType("OCC.Core.BRepBndLib")
    from OCP.BRepBndLib import BRepBndLib
    bbl.brepbndlib_Add = lambda shape, box, *a: BRepBndLib.Add_s(shape, box, *a)
    bbl.brepbndlib = types.SimpleNamespace(Add=bbl.brepbndlib_Add)
    sys.modules["OCC.Core.BRepBndLib"] = bbl
    de = types.ModuleType("OCC.Extend.DataExchange")
    de.write_stl_file = de.write_step_file = lambda *a, **k: None
    sys.modules["OCC.Extend.DataExchange"] = de


def _np_cross_2d():
    """cadlib calls np.cross on 2-D vectors, which NumPy >= 2.x no longer accepts: restore the old behaviour
    (2-D x 2-D -> z component, 2-D x 3-D -> 3-D with z = 0 padding)."""
    if getattr(np.cross, "_cadlib_compat", False):
        return
    cross = np.cross

    def compat(a, b, *args, **kw):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape[-1] == 2 or b.shape[-1] == 2:
            both = a.shape[-1] == 2 and b.shape[-1] == 2
            pad = lambda v: np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], -1) if v.shape[-1] == 2 else v  # noqa: E731
            r = cross(pad(a), pad(b), *args, **kw)
            return r[..., 2] if both else r
        return cross(a, b, *args, **kw)
    compat._cadlib_compat = True
    np.cross = compat


def _cadlib():
    install_occ_shim()
    _np_cross_2d()
    import matplotlib
    use = matplotlib.use
    matplotlib.use = lambda *a, **k: None   # upstream cadlib/sketch.py forces TkAgg
    try:
        from cadlib.extrude import CADSequence
        from cadlib.visualize import create_CAD
    finally:
        matplotlib.use = use
    return CADSequence, create_CAD


def deepcad_reference(vec):
    """DeepCAD vector -> cadquery Shape built by upstream cadlib (the ground-truth geometry), or None."""
    import cadquery as cq
    CADSequence, create_CAD = _cadlib()
    try:
        seq = CADSequence.from_vector(np.asarray(vec).astype(float), is_numerical=True, n=256)
        return cq.Shape.cast(create_CAD(seq))
    except Exception:  # noqa: BLE001
        return None


DECIMALS = 1   # coordinate precision; module-level so the output style is one switch


def _q(x):
    return np.round(np.asarray(x, float), DECIMALS)


def _fmt(x):
    v = round(float(x), DECIMALS)
    return str(int(v)) if DECIMALS == 0 or v == int(v) else f"{v:.{DECIMALS}f}"


def _pt(p):
    return f"({_fmt(p[0])},{_fmt(p[1])})"


def deepcad_to_cadquery(vec):
    """DeepCAD (L, 17) program -> CadQuery source (str), or None if the program cannot be parsed."""
    from copy import copy
    CADSequence, _ = _cadlib()
    from cadlib.curves import Arc, Circle, Line
    try:
        seq = CADSequence.from_vector(np.asarray(vec).astype(float), is_numerical=True, n=256)
    except Exception:  # noqa: BLE001
        return None

    # first pass: every extrude's sketch in global coordinates, to get the part's bounding box
    ops = []
    pts_all = []
    for ext in seq.seq:
        prof = copy(ext.profile)
        prof.denormalize(ext.sketch_size)
        o = np.asarray(ext.sketch_pos, float)
        x, y, n = (np.asarray(v, float) for v in (ext.sketch_plane.x_axis, ext.sketch_plane.y_axis,
                                                   ext.sketch_plane.normal))
        g = lambda p: o + p[0] * x + p[1] * y  # noqa: E731
        loops = []
        for loop in prof.children:
            curves = []
            for c in loop.children:
                if isinstance(c, Line):
                    curves.append(("L", g(c.start_point), g(c.end_point)))
                elif isinstance(c, Arc):
                    curves.append(("A", g(c.start_point), g(c.mid_point), g(c.end_point)))
                elif isinstance(c, Circle):
                    curves.append(("C", g(c.center), float(abs(c.radius))))
            loops.append(curves)
        e1, e2 = float(ext.extent_one), float(ext.extent_two)
        ops.append(dict(o=o, x=x, y=y, n=n, loops=loops, e1=e1, e2=e2, type=int(ext.extent_type),
                        op=int(ext.operation)))
        for curves in loops:
            for c in curves:
                if c[0] == "C":
                    pts_all += [c[1] + c[2] * s * v for s in (-1, 1) for v in (x, y)]
                else:
                    pts_all += list(c[1:])
                    pts_all += [p + e * n for p in c[1:] for e in (e1, -e2 if int(ext.extent_type) == 2 else
                                                                    (-e1 if int(ext.extent_type) == 1 else 0))]
    if not pts_all:
        return None
    P = np.array(pts_all)
    center = (P.max(0) + P.min(0)) / 2
    scale = SPAN / max((P.max(0) - P.min(0)).max(), 1e-9)

    lines = ["import cadquery as cq"]
    planes = {}
    body = None
    for k, op in enumerate(ops):
        n = op["n"]
        # DeepCAD quantises plane angles in 1.4 deg steps, so axis-aligned sketches can come back tilted: snap within ~3 deg
        name = next((nm for nm, (xd, nd) in NAMED_PLANES.items() if np.allclose(n, nd, atol=0.05)), None)
        o = (op["o"] - center) * scale
        if name is not None:            # named plane through the sketch origin; 2D coords in its own basis
            xd, nd = (np.array(v, float) for v in NAMED_PLANES[name])
            yd = np.cross(nd, xd)
            org = o
            key = (name, *_q(org))
            decl = f"cq.Workplane('{name}',origin=({','.join(_fmt(v) for v in org)}))"
        else:                           # general plane: explicit axes (rare in DeepCAD)
            xd, nd = op["x"], n
            yd = np.cross(nd, xd)
            org = o
            key = ("P", *_q(org), *np.round(xd, 4), *np.round(nd, 4))
            f3 = lambda v: "(" + ",".join(f"{float(t):.4g}" for t in v) + ")"  # noqa: E731
            decl = f"cq.Workplane(cq.Plane(origin={f3(org)},xDir={f3(xd)},normal={f3(nd)}))"
        if key not in planes:
            planes[key] = f"w{len(planes)}"
            lines.append(f"{planes[key]}={decl}")
        w = planes[key]
        to2d = lambda p: (((p - center) * scale - org) @ xd, ((p - center) * scale - org) @ yd)  # noqa: E731

        # A CadQuery Sketch keeps every edge drawn so far, so all polygon loops of the profile are drawn first and
        # assembled once (the face gets the inner loops as holes); circles are separate faces, added ('a') when
        # they are the outer loop and subtracted ('s') when they are holes.
        circles, edges = [], ""
        for li, curves in enumerate(op["loops"]):
            if len(curves) == 1 and curves[0][0] == "C":
                c = to2d(curves[0][1])
                circles.append((li, f".push([{_pt(c)}]).circle({_fmt(curves[0][2] * scale)}" +
                                ("" if li == 0 else ",mode='s'") + ")"))
                continue
            edges += _loop_edges(curves, to2d, first=not edges)
        outer_circle = bool(circles) and circles[0][0] == 0
        sk = ".sketch()"
        if outer_circle:
            sk += circles[0][1]
        if edges:
            sk += edges + (".assemble(mode='s')" if outer_circle else ".assemble()")
        sk += "".join(c for li, c in circles if li != 0)
        e1, e2 = op["e1"] * scale, op["e2"] * scale
        if op["type"] == 1:      # symmetric
            ext = f".finalize().extrude({_fmt(e1)},both=True)"
        elif op["type"] == 2:    # two sides: start the sketch e2 behind the plane and extrude e1 + e2
            sk = sk.replace(".sketch()", f".workplane(offset={_fmt(-e2)}).sketch()", 1)
            ext = f".finalize().extrude({_fmt(e1 + e2)})"
        else:
            ext = f".finalize().extrude({_fmt(e1)})"
        solid = f"{w}{sk}{ext}"
        if body is None:
            body = solid
        else:
            verb = {0: "union", 1: "union", 2: "cut", 3: "intersect"}[op["op"]]
            body = f"{body}.{verb}({solid})"
    lines.append(f"r={body}")
    return "\n".join(lines)


def _loop_edges(curves, to2d, first=True):
    """One closed loop as Sketch edge calls. CadQuery's Sketch.close() adds a segment back to the first point of the
    whole sketch, so only a profile's first edge loop uses it (leaving the closing line to it, or skipping it when the
    loop already ends on its start with an arc); later loops close with an explicit segment."""
    items, start, last = [], None, None
    for c in curves:
        ends = [_q(to2d(p)) for p in c[1:]]
        collinear = c[0] == "A" and abs(np.cross(ends[1] - ends[0], ends[2] - ends[0])) < 10 ** -DECIMALS
        if c[0] == "L" or collinear or np.array_equal(ends[1], ends[0]) or np.array_equal(ends[1], ends[2]):
            a, b = ends[0], ends[-1]
            kind, pts = "L", [b]
        else:
            a, b = ends[0], ends[2]
            kind, pts = "A", [ends[1], b]
        if start is None:
            start, last = a, a
        if np.array_equal(b, last):
            continue
        items.append((kind, pts))
        last = b
    ends_on_start = last is not None and np.array_equal(last, start)
    if first and items and items[-1][0] == "L" and np.array_equal(items[-1][1][-1], start):
        items.pop()
        closer = ".close()"
    elif first:
        closer = "" if ends_on_start else ".close()"
    else:
        closer = "" if ends_on_start else f".segment({_pt(start)})"
    if len(items) < 2 and not (items and items[0][0] == "A"):
        return ""                    # degenerate loop (a point or a single back-and-forth line)
    out = ""
    for k, (kind, pts) in enumerate(items):
        args = ([start] if k == 0 else []) + pts
        out += f".{'segment' if kind == 'L' else 'arc'}({','.join(_pt(p) for p in args)})"
    return out + closer


def run_cadquery(code):
    """Execute CadQuery source that assigns `r`; -> cadquery Shape (non-empty solid) or None."""
    import cadquery as cq
    ns = {"cq": cq}
    try:
        exec(code, ns)
        r = ns.get("r")
        shape = r.val() if hasattr(r, "val") else r
        if shape is None or not shape.isValid() or shape.Volume() <= 1e-9:
            return None
        return shape
    except Exception:  # noqa: BLE001
        return None


def normalise(shape):
    """Centre the bounding box at 0 and scale its longest side to 1 (the CAD-Recode / cadrille protocol)."""
    bb = shape.BoundingBox()
    c = np.array([(bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2])
    s = 1.0 / max(bb.xlen, bb.ylen, bb.zlen)
    return shape.translate(tuple(-c)).scale(s)


def iou(a, b):
    """Volume IoU of two solids already in a common frame."""
    va, vb = a.Volume(), b.Volume()
    vi = a.intersect(b).Volume()
    return vi / max(va + vb - vi, 1e-12)
