"""Compositional random parts for breadth (not a named part type).

1-4 primitive bodies built in the XY frame (extruded rects / rounded rects / circles / rings / polygons / outlines /
slots, boxes, horizontal revolved shafts) are combined with union/cut at positions that keep one connected solid; then
0-4 finishing operations are applied where they are geometrically valid: fillets/chamfers of real edges picked with
near(...) (or exact string selectors), holes / cboreHole / cskHole (single, rarray, polarArray, pushPoints), blind
pockets, slots, and shells.

The generator builds the program incrementally: every body and every feature is evaluated on the real CadQuery
object built from the code emitted so far, and kept only if the result is one valid solid and the feature sits
cleanly (holes and pockets clear of every edge, blind pockets with a floor, fillets smaller than the faces they
blend). Positions are computed from the geometry actually built (faces sampled with clearance from their boundary,
points on real edges), so every number in the emitted program is a literal.

`generate_mix(rng, hard)` is shared with `random_mix_hard`.
"""
import math

import cadquery as cq

from datagen.families.common import f, orient

_NS = None


def _ns():
    global _NS
    if _NS is None:
        from mm.cad_helpers import namespace
        _NS = namespace()
    return _NS


def ev(obj, seg):
    ns = _ns()
    ns["_r"] = obj
    try:
        return eval("_r" + seg, ns)
    finally:
        ns.pop("_r", None)


def ev_expr(expr):
    return eval(expr, _ns())


def solid_of(w):
    objs = [o for o in w.vals() if isinstance(o, cq.Shape)]
    if len(objs) != 1:
        return None
    return objs[0]


def good(w):
    """The workplane holds exactly one valid solid."""
    try:
        s = solid_of(w)
        if s is None:
            return None
        if len(s.Solids()) != 1 or not s.isValid() or s.Volume() <= 0:
            return None
        return s
    except Exception:  # noqa: BLE001
        return None


def try_seg(w, seg):
    try:
        w2 = ev(w, seg)
    except Exception:  # noqa: BLE001
        return None, None
    s = good(w2)
    if s is None:
        return None, None
    return w2, s


# ----------------------------------------------------------------------------------------------- geometry queries
class Classifier:
    def __init__(self, shape):
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        self.c = BRepClass3d_SolidClassifier(shape.wrapped)

    def inside(self, p):
        from OCP.gp import gp_Pnt
        from OCP.TopAbs import TopAbs_OUT
        self.c.Perform(gp_Pnt(*p), 1e-6)
        return self.c.State() != TopAbs_OUT


class _BB:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax = xmin, ymin, zmin, xmax, ymax, zmax
        self.xlen, self.ylen, self.zlen = xmax - xmin, ymax - ymin, zmax - zmin


def fbb(shape):
    """Fast (non-optimal, tolerance-free) bounding box; Shape.BoundingBox() uses the slow optimal algorithm."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    b.SetGap(0.0)
    BRepBndLib.Add_s(shape.wrapped, b, True)
    return _BB(*b.Get())


def vtx(p):
    return cq.Vertex.makeVertex(*map(float, p))


def face_normal(face):
    n = face.normalAt()
    return (n.x, n.y, n.z)


def axis_of(n):
    """Index and sign of the coordinate axis a unit normal points along, or None."""
    for i in range(3):
        if abs(abs(n[i]) - 1) < 1e-6:
            return i, (1 if n[i] > 0 else -1)
    return None


class FaceInfo:
    def __init__(self, face):
        self.face = face
        self.n = face_normal(face)
        self.ax, self.sg = axis_of(self.n)
        self.bnd = cq.Compound.makeCompound(face.Wires())
        self.bb = fbb(face)
        self.area = face.Area()
        self.level = [self.bb.xmin, self.bb.ymin, self.bb.zmin][self.ax]

    def clearance(self, p):
        """Distance from p (a point in the face plane) to the face boundary, or -1 if p is outside the face."""
        v = vtx(p)
        if self.face.distance(v) > 1e-3:
            return -1.0
        return self.bnd.distance(v)

    def centre(self):
        c = self.face.Center()
        return (c.x, c.y, c.z)

    def sample(self, rng, clear, tries=25, centre_first=True):
        if centre_first:
            p = self.centre()
            if self.clearance(p) >= clear:
                return p
        lo = (self.bb.xmin, self.bb.ymin, self.bb.zmin)
        hi = (self.bb.xmax, self.bb.ymax, self.bb.zmax)
        for _ in range(tries):
            p = tuple(rng.uniform(lo[i], hi[i]) if i != self.ax else self.level for i in range(3))
            if self.clearance(p) >= clear:
                return p
        return None

    def min_width(self):
        a = self.area
        per = sum(e.Length() for e in self.face.Edges())
        return 2 * a / max(per, 1e-9)


def planar_faces(shape, axes=(2,), signs=(1, -1), min_area=100.0):
    out = []
    for fc in shape.Faces():
        if fc.geomType() != "PLANE":
            continue
        n = face_normal(fc)
        a = axis_of(n)
        if a is None or a[0] not in axes or a[1] not in signs:
            continue
        fi = FaceInfo(fc)
        if fi.area < min_area:
            continue
        out.append(fi)
    return out


def face_selector(w, shape, fi, rng):
    """Code selecting exactly face fi: a string selector when it is the unique extreme face, else near(point)."""
    name = "XYZ"[fi.ax]
    sel = (">" if fi.sg > 0 else "<") + name
    try:
        got = ev(w, f".faces('{sel}')").vals()
        if len(got) == 1 and got[0].isSame(fi.face):
            return f".faces('{sel}')"
    except Exception:  # noqa: BLE001
        pass
    p = fi.sample(rng, 2.0)
    if p is None:
        return None
    s = f".faces(near(({f(p[0])},{f(p[1])},{f(p[2])})))"
    try:
        got = ev(w, s).vals()
        if len(got) == 1 and got[0].isSame(fi.face):
            return s
    except Exception:  # noqa: BLE001
        pass
    return None


def extent_below(shape, fi):
    """How far the solid extends behind face fi (along -normal)."""
    bb = fbb(shape)
    lo = [bb.xmin, bb.ymin, bb.zmin][fi.ax]
    hi = [bb.xmax, bb.ymax, bb.zmax][fi.ax]
    return (fi.level - lo) if fi.sg > 0 else (hi - fi.level)


def clean_cut(cl, pts_world, fi, depth, blind, floor=1.5, step=None):
    """A cut of the given outline (world points on the face, already grown by a margin) going `depth` behind the
    face: at every sampled depth all outline points share one inside/outside state (the cut never grazes a wall)
    and, if blind, the solid continues `floor` below its bottom (the pocket has a floor)."""
    n = fi.n
    nd = max(4, min(40, int(depth / (step or 2.0))))
    ts = [0.4 + (depth - 0.8) * k / (nd - 1) for k in range(nd)] if depth > 1 else [depth / 2]
    for t in ts:
        st = None
        for p in pts_world:
            q = (p[0] - n[0] * t, p[1] - n[1] * t, p[2] - n[2] * t)
            s = cl.inside(q)
            if st is None:
                st = s
            elif s != st:
                return False
    if blind:
        t = depth + floor
        for p in pts_world:
            if not cl.inside((p[0] - n[0] * t, p[1] - n[1] * t, p[2] - n[2] * t)):
                return False
    return True


def material_depth(cl, fi, p, maxd):
    """Depth of material directly behind point p on face fi."""
    n = fi.n
    t = 0.3
    while t < maxd:
        if not cl.inside((p[0] - n[0] * t, p[1] - n[1] * t, p[2] - n[2] * t)):
            return t
        t += 0.7
    return maxd


def ring(c, r, k=12, plane=None):
    return [(c[0] + r * math.cos(2 * math.pi * i / k + 0.3), c[1] + r * math.sin(2 * math.pi * i / k + 0.3))
            for i in range(k)]


# ----------------------------------------------------------------------------------------------- body emitters
def wp(x, y, z):
    if f(x) == "0" and f(y) == "0" and f(z) == "0":
        return "cq.Workplane('XY')"
    return f"cq.Workplane('XY',origin=({f(x)},{f(y)},{f(z)}))"


def r1(x):
    return round(float(x), 1)


def outline_pts(rng, w, d):
    """A plausible plate outline inside a w x d box, centred: L, T, U, plus, trapezoid, chamfered rect, hexagon."""
    a, b = w / 2, d / 2
    k = rng.choice(["L", "T", "U", "plus", "trap", "chamf", "hex"])
    if k == "L":
        cx, cy = rng.uniform(0.3, 0.6) * w, rng.uniform(0.3, 0.6) * d
        pts = [(-a, -b), (a, -b), (a, -b + cy), (-a + cx, -b + cy), (-a + cx, b), (-a, b)]
    elif k == "T":
        sw, sh = rng.uniform(0.3, 0.5) * w, rng.uniform(0.3, 0.6) * d
        pts = [(-a, b), (-a, b - sh), (-sw / 2, b - sh), (-sw / 2, -b), (sw / 2, -b), (sw / 2, b - sh), (a, b - sh),
               (a, b)]
    elif k == "U":
        gw, gd = rng.uniform(0.3, 0.5) * w, rng.uniform(0.35, 0.65) * d
        pts = [(-a, -b), (a, -b), (a, b), (gw / 2, b), (gw / 2, b - gd), (-gw / 2, b - gd), (-gw / 2, b), (-a, b)]
    elif k == "plus":
        sx, sy = rng.uniform(0.3, 0.5) * w / 2, rng.uniform(0.3, 0.5) * d / 2
        pts = [(-a, -sy), (-sx, -sy), (-sx, -b), (sx, -b), (sx, -sy), (a, -sy), (a, sy), (sx, sy), (sx, b), (-sx, b),
               (-sx, sy), (-a, sy)]
    elif k == "trap":
        t = rng.uniform(0.45, 0.8) * a
        pts = [(-a, -b), (a, -b), (t, b), (-t, b)]
    elif k == "chamf":
        c = rng.uniform(0.1, 0.3) * min(w, d)
        pts = [(-a + c, -b), (a - c, -b), (a, -b + c), (a, b - c), (a - c, b), (-a + c, b), (-a, b - c), (-a, -b + c)]
    else:
        c = rng.uniform(0.2, 0.45) * w
        pts = [(-a + c, -b), (a - c, -b), (a, 0), (a - c, b), (-a + c, b), (-a, 0)]
    return [(r1(x), r1(y)) for x, y in pts]


def pts_code(pts):
    return "[" + ",".join(f"({f(x)},{f(y)})" for x, y in pts) + "]"


def make_base(rng, T, hard):
    """Base body: returns (kind, emit(dx,dy,dz)->expr). Footprint centred at the origin, bottom at z=0."""
    kinds = ["box", "rect", "rrect", "rrect", "circle", "ring", "ngon", "outline", "outline", "slot", "revolve"]
    k = rng.choice(kinds)
    w = r1(T)
    d = r1(T * rng.choice([rng.uniform(0.35, 0.7), rng.uniform(0.6, 1.0)]))
    plate = rng.random() < 0.6
    h = r1(T * (rng.uniform(0.05, 0.16) if plate else rng.uniform(0.16, 0.45)))
    if k == "box":
        return k, (w, d, h), lambda dx, dy, dz: f"{wp(dx, dy, h / 2 + dz)}.box({f(w)},{f(d)},{f(h)})"
    if k in ("rect", "rrect"):
        cut = ""
        if rng.random() < 0.3:
            if rng.random() < 0.6:
                cut = f".push([(0,0)]).circle({f(min(w, d) * rng.uniform(0.12, 0.3))},mode='s')"
            else:
                cut = f".push([(0,0)]).rect({f(w * rng.uniform(0.2, 0.45))},{f(d * rng.uniform(0.2, 0.45))},mode='s')"
        if k == "rect" and not cut:
            return k, (w, d, h), lambda dx, dy, dz: f"{wp(dx, dy, dz)}.rect({f(w)},{f(d)}).extrude({f(h)})"
        fil = f".vertices().fillet({f(min(w, d) * rng.uniform(0.06, 0.25))})" if k == "rrect" else ""
        return k, (w, d, h), lambda dx, dy, dz: (f"{wp(dx, dy, dz)}.sketch().rect({f(w)},{f(d)}){fil}{cut}"
                                                  f".finalize().extrude({f(h)})")
    if k in ("circle", "ring"):
        R = r1(T / 2)
        if k == "ring":
            ri = r1(R * rng.uniform(0.25, 0.6))
            return k, (T, T, h), lambda dx, dy, dz: (f"{wp(dx, dy, dz)}.circle({f(R)}).circle({f(ri)})"
                                                      f".extrude({f(h)})")
        return k, (T, T, h), lambda dx, dy, dz: f"{wp(dx, dy, dz)}.circle({f(R)}).extrude({f(h)})"
    if k == "ngon":
        n = rng.choice([3, 5, 6, 6, 8])
        cut = f".circle({f(T * rng.uniform(0.1, 0.25))})" if rng.random() < 0.4 else ""
        return k, (T, T, h), lambda dx, dy, dz: f"{wp(dx, dy, dz)}.polygon({n},{f(T)}){cut}.extrude({f(h)})"
    if k == "outline":
        pts = outline_pts(rng, w, d)
        return k, (w, d, h), lambda dx, dy, dz: f"{wp(dx, dy, dz)}.polyline({pts_code(pts)}).close().extrude({f(h)})"
    if k == "slot":
        dd = r1(T * rng.uniform(0.25, 0.5))
        return k, (w, dd, h), lambda dx, dy, dz: f"{wp(dx, dy, dz)}.slot2D({f(w)},{f(dd)}).extrude({f(h)})"
    # revolve: stepped shaft / pulley about a horizontal axis (local X of the XY workplane), resting at z=0..2R
    n = rng.randint(2, 4 if not hard else 5)
    cuts = sorted(rng.uniform(0.1, 0.9) for _ in range(n - 1))
    xs = [-T / 2] + [-T / 2 + c * T for c in cuts] + [T / 2]
    rmax = T * rng.uniform(0.15, 0.35)
    rads = [r1(rmax * rng.uniform(0.35, 1.0)) for _ in range(n)]
    rads[rng.randrange(n)] = r1(rmax)
    pts = [(r1(xs[0]), 0)]
    for i in range(n):
        x0, x1 = r1(xs[i]), r1(xs[i + 1])
        if i > 0 and rng.random() < 0.25 and abs(rads[i] - rads[i - 1]) > 2:   # occasional cone step
            pts.append((r1(x0 + min(0.4 * (x1 - x0), abs(rads[i] - rads[i - 1]))), rads[i]))
        else:
            pts.append((x0, rads[i]))
        pts.append((x1, rads[i]))
    pts.append((r1(xs[-1]), 0))
    clean = [pts[0]]
    for p in pts[1:]:
        if p != clean[-1]:
            clean.append(p)
    R = max(rads)
    return k, (T, 2 * R, 2 * R), lambda dx, dy, dz: (f"{wp(dx, dy, R + dz)}.polyline({pts_code(clean)}).close()"
                                                      f".revolve(360,(0,0,0),(1,0,0))")


# ----------------------------------------------------------------------------------------------- composition
class Part:
    def __init__(self, base_emit):
        self.base_emit = base_emit
        self.adds = []            # (op, emit)
        self.w = None
        self.s = None

    def code(self, dx=0.0, dy=0.0, dz=0.0):
        return self.base_emit(dx, dy, dz) + "".join(f".{op}({em(dx, dy, dz)})" for op, em in self.adds)

    def build(self, dx=0.0, dy=0.0, dz=0.0):
        try:
            w = ev_expr(self.code(dx, dy, dz))
        except Exception:  # noqa: BLE001
            return False
        s = good(w)
        if s is None:
            return False
        self.w, self.s = w, s
        return True

    def try_add(self, op, em):
        try:
            w = ev(self.w, f".{op}({em(0, 0, 0)})")
        except Exception:  # noqa: BLE001
            return False
        s = good(w)
        if s is None:
            return False
        if op == "cut" and s.Volume() > self.s.Volume() - 1:
            return False
        if op == "union" and s.Volume() < self.s.Volume() + 1:
            return False
        self.adds.append((op, em))
        self.w, self.s = w, s
        return True


def add_ears(rng, part, kind, dims):
    """Mounting ears/tabs at the bottom of the base on one or both sides."""
    w, d, h = dims
    ax = rng.choice([0, 1]) if kind not in ("slot", "circle", "ring", "ngon") else 0
    span = w if ax == 0 else d
    side = d if ax == 0 else w
    te = r1(h * rng.uniform(0.35, 1.0)) if h > 8 else r1(h)
    te = max(te, 4.0)
    wt = r1(min(side * rng.uniform(0.3, 0.6), 0.25 * span + 20))
    le = r1(span * rng.uniform(0.12, 0.25))
    both = rng.random() < 0.6
    rounded = rng.random() < 0.6
    sides = [1, -1] if both else [rng.choice([1, -1])]
    # inset the ear root so it overlaps the base wall even on round/rounded outlines
    edge = span / 2 - (0.0 if kind in ("box", "rect", "outline") else min(0.25 * span, wt))
    for sg in sides:
        if rounded:
            L = r1(2 * (span / 2 - edge + le))
            c = sg * edge
            ang = 0 if ax == 0 else 90
            em = (lambda c=c, L=L, ang=ang: lambda dx, dy, dz: (
                f"{wp((c if ax == 0 else 0) + dx, (c if ax == 1 else 0) + dy, dz)}"
                f".slot2D({f(L)},{f(wt)}{',' + str(ang) if ang else ''}).extrude({f(te)})"))()
        else:
            ln = r1(span / 2 - edge + le + 2)
            c = sg * (span / 2 + le - ln / 2)
            bw, bd = (ln, wt) if ax == 0 else (wt, ln)
            em = (lambda c=c, bw=bw, bd=bd: lambda dx, dy, dz: (
                f"{wp((c if ax == 0 else 0) + dx, (c if ax == 1 else 0) + dy, te / 2 + dz)}"
                f".box({f(bw)},{f(bd)},{f(te)})"))()
        part.try_add("union", em)


def add_wall(rng, part, dims, T):
    """Upright wall flush with one edge of a rectangular base (L-bracket)."""
    w, d, h = dims
    ax = rng.choice([0, 1])
    sg = rng.choice([1, -1])
    tw = r1(max(4.0, T * rng.uniform(0.05, 0.12)))
    hw = r1(T * rng.uniform(0.3, 0.7))
    span = d if ax == 0 else w
    ln = r1(span * rng.choice([1.0, 1.0, rng.uniform(0.5, 0.9)]))
    c = sg * ((w if ax == 0 else d) / 2 - tw / 2)
    bw, bd = (tw, ln) if ax == 0 else (ln, tw)
    em = lambda dx, dy, dz: (f"{wp((c if ax == 0 else 0) + dx, (c if ax == 1 else 0) + dy, h + hw / 2 + dz)}"
                             f".box({f(bw)},{f(bd)},{f(hw)})")
    return part.try_add("union", em)


def add_bodycut(rng, part, dims):
    """Step, corner notch, U-notch or groove cut from a rectangular base."""
    w, d, h = dims
    k = rng.choice(["step", "notch", "unotch", "groove"])
    e = 2.0
    if k == "step":
        ax, sg = rng.choice([0, 1]), rng.choice([1, -1])
        span = w if ax == 0 else d
        a = r1(span * rng.uniform(0.2, 0.45))
        dep = r1(h * rng.uniform(0.3, 0.6))
        c = sg * (span / 2 - a / 2 + e / 2)
        other = (d if ax == 0 else w) + 2 * e
        bw, bd = (a + e, other) if ax == 0 else (other, a + e)
        z = h - dep / 2 + e / 2
        em = lambda dx, dy, dz: (f"{wp((c if ax == 0 else 0) + dx, (c if ax == 1 else 0) + dy, z + dz)}"
                                 f".box({f(bw)},{f(bd)},{f(dep + e)})")
    elif k == "notch":
        sx, sy = rng.choice([1, -1]), rng.choice([1, -1])
        a, b = r1(w * rng.uniform(0.15, 0.35)), r1(d * rng.uniform(0.15, 0.35))
        x, y = sx * (w / 2 - a / 2 + e / 2), sy * (d / 2 - b / 2 + e / 2)
        em = lambda dx, dy, dz: f"{wp(x + dx, y + dy, h / 2 + dz)}.box({f(a + e)},{f(b + e)},{f(h + 2 * e)})"
    elif k == "unotch":
        ax, sg = rng.choice([0, 1]), rng.choice([1, -1])
        span, other = (w, d) if ax == 0 else (d, w)
        rr = r1(min(other * rng.uniform(0.1, 0.25), span * 0.2))
        c = sg * span / 2
        em = lambda dx, dy, dz: (f"{wp((c if ax == 0 else 0) + dx, (c if ax == 1 else 0) + dy, -e + dz)}"
                                 f".circle({f(rr)}).extrude({f(h + 2 * e)})")
    else:
        ax = rng.choice([0, 1])
        span, other = (w, d) if ax == 0 else (d, w)
        g = r1(span * rng.uniform(0.08, 0.2))
        dep = r1(h * rng.uniform(0.25, 0.5))
        c = r1(span * rng.uniform(-0.25, 0.25))
        bw, bd = (g, other + 2 * e) if ax == 0 else (other + 2 * e, g)
        z = h - dep / 2 + e / 2
        em = lambda dx, dy, dz: (f"{wp((c if ax == 0 else 0) + dx, (c if ax == 1 else 0) + dy, z + dz)}"
                                 f".box({f(bw)},{f(bd)},{f(dep + e)})")
    return part.try_add("cut", em)


def add_boss(rng, part, T, hard):
    """A body standing on an upward planar face (boss, pad, tube, rib, polygon)."""
    faces = [fi for fi in planar_faces(part.s, axes=(2,), signs=(1,), min_area=400.0)]
    if not faces:
        return False
    faces.sort(key=lambda fi: -fi.area)
    fi = faces[0] if rng.random() < 0.7 else rng.choice(faces)
    k = rng.choice(["cyl", "cyl", "tube", "box", "rrect", "ngon", "rib", "slot"])
    mw = fi.min_width()
    for shrink in (1.0, 0.7, 0.5):
        size = mw * rng.uniform(0.4, 1.1) * shrink
        if size < 8:
            return False
        if k in ("cyl", "tube", "ngon"):
            rad = r1(size / 2)
            circ = rad
        elif k in ("box", "rrect"):
            a, b = r1(size * rng.uniform(0.6, 1.0)), r1(size * rng.uniform(0.4, 0.9))
            circ = math.hypot(a, b) / 2
        elif k == "rib":
            a, b = r1(size * rng.uniform(1.2, 2.5)), r1(max(3.0, size * rng.uniform(0.08, 0.15)))
            if rng.random() < 0.5:
                a, b = b, a
            circ = math.hypot(a, b) / 2
        else:
            a, b = r1(size * rng.uniform(1.2, 2.0)), r1(size * rng.uniform(0.35, 0.6))
            circ = a / 2
        p = fi.sample(rng, circ + 1.5, centre_first=rng.random() < 0.5)
        if p is None:
            continue
        x, y, z = r1(p[0]), r1(p[1]), fi.level
        hb = r1(T * rng.uniform(0.08, 0.35 if not hard else 0.45))
        if k == "rib":
            hb = r1(T * rng.uniform(0.06, 0.2))
        if k == "cyl":
            body = f".circle({f(rad)}).extrude({f(hb)})"
        elif k == "tube":
            body = f".circle({f(rad)}).circle({f(rad * rng.uniform(0.4, 0.7))}).extrude({f(hb)})"
        elif k == "ngon":
            body = f".polygon({rng.choice([5, 6, 6, 8])},{f(2 * rad)}).extrude({f(hb)})"
        elif k == "box":
            body = f".rect({f(a)},{f(b)}).extrude({f(hb)})"
        elif k == "rrect":
            body = f".sketch().rect({f(a)},{f(b)}).vertices().fillet({f(min(a, b) * rng.uniform(0.1, 0.35))}).finalize().extrude({f(hb)})"
        elif k == "rib":
            body = f".rect({f(a)},{f(b)}).extrude({f(hb)})"
        else:
            ang = rng.choice([0, 0, 90, 45, 30])
            body = f".slot2D({f(a)},{f(b)}{',' + str(ang) if ang else ''}).extrude({f(hb)})"
        em = (lambda x=x, y=y, z=z, body=body: lambda dx, dy, dz: f"{wp(x + dx, y + dy, z + dz)}{body}")()
        if part.try_add("union", em):
            return True
    return False


def add_pad(rng, part, dims):
    """A rectangular foot under a horizontal shaft (pillow-block style)."""
    T, D, _ = dims
    R = D / 2
    pw = r1(T * rng.uniform(0.3, 0.8))
    pd = r1(D * rng.uniform(1.4, 2.4))
    ph = r1(R * rng.uniform(0.25, 0.6))
    x = r1(T * rng.uniform(-0.2, 0.2) * (1 - pw / T))
    top = R * rng.uniform(0.6, 1.0)
    hh = r1(top + ph)
    em = lambda dx, dy, dz: f"{wp(x + dx, dy, -ph + hh / 2 + dz)}.box({f(pw)},{f(pd)},{f(hh)})"
    return part.try_add("union", em)


def build_bodies(rng, hard):
    T = rng.uniform(105, 190)
    kind, dims, emit = make_base(rng, T, hard)
    part = Part(emit)
    if not part.build():
        return None
    nb = rng.choice([1, 1, 2, 2, 3, 4]) if not hard else rng.choice([2, 2, 3, 3, 4, 4])
    rectish = kind in ("box", "rect", "rrect")
    used = 1
    if kind == "revolve":
        if used < nb and rng.random() < 0.6:
            used += add_pad(rng, part, dims)
    else:
        if used < nb and rng.random() < 0.35:
            n0 = len(part.adds)
            add_ears(rng, part, kind, dims)
            used += len(part.adds) - n0
        walled = False
        if used < nb and kind in ("box", "rect") and rng.random() < 0.3:
            walled = add_wall(rng, part, dims, T)
            used += walled
        if used < nb and rectish and not walled and rng.random() < 0.4:
            used += add_bodycut(rng, part, dims)
        tries = 0
        while used < nb and tries < 4:
            tries += 1
            used += add_boss(rng, part, T, hard)
    # recentre on the bounding-box centre and rebuild
    bb = part.s.BoundingBox()
    dx, dy, dz = -round((bb.xmin + bb.xmax) / 2), -round((bb.ymin + bb.ymax) / 2), -round((bb.zmin + bb.zmax) / 2)
    if not part.build(dx, dy, dz):
        return None
    bb = part.s.BoundingBox()
    side = max(bb.xlen, bb.ylen, bb.zlen)
    if not 100 <= side <= 205:
        return None
    return part, part.code(dx, dy, dz), kind, len(part.adds) == 0


# ----------------------------------------------------------------------------------------------- finishing ops
def local(plane, p):
    v = plane.toLocalCoords(cq.Vector(*p))
    return (v.x, v.y)


def world(plane, q):
    v = plane.toWorldCoords((q[0], q[1]))
    return (v.x, v.y, v.z)


def pick_face(rng, s, side_p=0.2, min_area=300.0):
    axes = (2,) if rng.random() > side_p else (0, 1)
    faces = planar_faces(s, axes=axes, min_area=min_area)
    if not faces:
        faces = planar_faces(s, axes=(0, 1, 2), min_area=min_area)
    if not faces:
        return None
    wts = [fi.area ** 0.5 for fi in faces]
    return rng.choices(faces, weights=wts)[0]


def pt_code(q):
    return f"({f(q[0])},{f(q[1])})"


def op_holes(rng, st, T):
    w, s = st["w"], st["s"]
    fi = pick_face(rng, s)
    if fi is None:
        return None
    sel = face_selector(w, s, fi, rng)
    if sel is None:
        return None
    wpl = ev(w, sel + ".workplane()")
    plane = wpl.plane
    cl = Classifier(s)
    below = extent_below(s, fi)
    mw = fi.min_width()
    d = round(min(T * rng.uniform(0.03, 0.09), mw * rng.uniform(0.2, 0.5)) * 2) / 2
    if d < 3:
        return None
    htype = rng.choice(["hole", "hole", "cbore", "csk"])
    outer = d * (1.8 if htype != "hole" else 1.0)
    mrg = max(2.0, 0.35 * d)
    thick_c = material_depth(cl, fi, fi.centre(), below + 1)
    through = rng.random() < 0.65
    # placement pattern
    pat = rng.choice(["single", "rarray", "rarray", "polar", "push"])
    c = fi.sample(rng, 0, centre_first=True) if pat != "push" else None
    pts = []
    head = ""
    if pat == "single":
        p = fi.sample(rng, outer / 2 + mrg, centre_first=rng.random() < 0.6)
        if p is None:
            return None
        q = local(plane, p)
        pts = [p]
        head = f".center({f(q[0])},{f(q[1])})" if (f(q[0]), f(q[1])) != ("0", "0") else ""
        if head:
            pts = [world(plane, (r1(q[0]), r1(q[1])))]
    elif pat == "rarray":
        if c is None:
            return None
        bb = fi.bb
        ext = sorted([(bb.xlen, 0), (bb.ylen, 1), (bb.zlen, 2)])[1:]
        # local extents along the workplane x/y
        lx = abs(plane.xDir.x) * bb.xlen + abs(plane.xDir.y) * bb.ylen + abs(plane.xDir.z) * bb.zlen
        ly = abs(plane.yDir.x) * bb.xlen + abs(plane.yDir.y) * bb.ylen + abs(plane.yDir.z) * bb.zlen
        nx, ny = rng.choice([(2, 2), (2, 2), (2, 1), (1, 2), (3, 2), (2, 3), (3, 1), (4, 1), (1, 3)])
        for _ in range(4):
            sx = r1(lx * rng.uniform(0.4, 0.8) / max(nx - 1, 1)) if nx > 1 else 1
            sy = r1(ly * rng.uniform(0.4, 0.8) / max(ny - 1, 1)) if ny > 1 else 1
            q0 = local(plane, c)
            q0 = (r1(q0[0]), r1(q0[1]))
            loc = [(q0[0] + (i - (nx - 1) / 2) * sx, q0[1] + (j - (ny - 1) / 2) * sy) for i in range(nx)
                   for j in range(ny)]
            pts = [world(plane, q) for q in loc]
            if all(fi.clearance(p) >= outer / 2 + mrg for p in pts):
                break
            pts = []
        if not pts:
            return None
        del ext
        head = (f".center({f(q0[0])},{f(q0[1])})" if (f(q0[0]), f(q0[1])) != ("0", "0") else "") + \
            f".rarray({f(sx)},{f(sy)},{nx},{ny})"
    elif pat == "polar":
        if c is None:
            return None
        q0 = local(plane, c)
        q0 = (r1(q0[0]), r1(q0[1]))
        n = rng.choice([3, 4, 4, 5, 6, 6, 8])
        a0 = rng.choice([0, 0, 45, 90, 30])
        pts = []
        for _ in range(4):
            R = r1(mw * rng.uniform(0.4, 1.0))
            if R < outer:
                continue
            loc = [(q0[0] + R * math.cos(math.radians(a0 + 360 * i / n)),
                    q0[1] + R * math.sin(math.radians(a0 + 360 * i / n))) for i in range(n)]
            pts = [world(plane, q) for q in loc]
            if all(fi.clearance(p) >= outer / 2 + mrg for p in pts) and 2 * R * math.sin(math.pi / n) > outer + mrg:
                break
            pts = []
        if not pts:
            return None
        head = (f".center({f(q0[0])},{f(q0[1])})" if (f(q0[0]), f(q0[1])) != ("0", "0") else "") + \
            f".polarArray({f(R)},{a0},360,{n})"
    else:
        k = rng.randint(2, 4)
        loc = []
        for _ in range(k * 4):
            p = fi.sample(rng, outer / 2 + mrg, centre_first=False, tries=6)
            if p is None:
                continue
            q = local(plane, p)
            q = (r1(q[0]), r1(q[1]))
            if all(math.hypot(q[0] - a[0], q[1] - a[1]) > outer + mrg for a in loc):
                loc.append(q)
            if len(loc) == k:
                break
        if len(loc) < 2:
            return None
        pts = [world(plane, q) for q in loc]
        head = ".pushPoints(" + pts_code_list(loc) + ")"
    # depth
    if through:
        depth = below
        dtxt = ""
    else:
        thick = min(material_depth(cl, fi, p, below + 1) for p in pts)
        if thick < 4:
            return None
        depth = r1(thick * rng.uniform(0.4, 0.75))
        dtxt = f",{f(depth)}"
    if htype == "hole":
        call = f".hole({f(d)}{dtxt})"
    elif htype == "cbore":
        cd = r1(min(thick_c * rng.uniform(0.2, 0.45), (depth if not through else below) * 0.5))
        if cd < 1:
            call = f".hole({f(d)}{dtxt})"
        else:
            call = f".cboreHole({f(d)},{f(d * rng.uniform(1.5, 1.9))},{f(cd)}{dtxt})"
    else:
        call = f".cskHole({f(d)},{f(d * rng.uniform(1.6, 2.0))},{rng.choice([82, 90, 90, 100])}{dtxt})"
    # every hole cuts cleanly
    for p in pts:
        if not clean_cut(cl, [world(plane, (local(plane, p)[0] + u[0], local(plane, p)[1] + u[1]))
                              for u in ring((0, 0), outer / 2 + 1.2, 10)] + [p], fi, depth, not through):
            return None
    return sel + ".workplane()" + head + call


def pts_code_list(loc):
    return "[" + ",".join(pt_code(q) for q in loc) + "]"


def op_pocket(rng, st, T):
    """Blind pocket or slot (rect, rounded rect, circle, slot2D), occasionally cut through."""
    w, s = st["w"], st["s"]
    fi = pick_face(rng, s, side_p=0.1, min_area=600.0)
    if fi is None:
        return None
    sel = face_selector(w, s, fi, rng)
    if sel is None:
        return None
    plane = ev(w, sel + ".workplane()").plane
    cl = Classifier(s)
    below = extent_below(s, fi)
    mw = fi.min_width()
    k = rng.choice(["rect", "rrect", "circle", "slot", "slot"])
    mrg = 2.5
    for shrink in (1.0, 0.7, 0.5):
        size = mw * rng.uniform(0.5, 1.2) * shrink
        if size < 6:
            return None
        ang = rng.choice([0, 0, 0, 90, 45, 30, -30])
        if k in ("rect", "rrect"):
            a, b = r1(size * rng.uniform(0.7, 1.4)), r1(size * rng.uniform(0.4, 0.8))
            ol = [(sx * (a / 2 + mrg) * t, sy * (b / 2 + mrg)) for sx in (1, -1) for sy in (1, -1)
                  for t in (1, 0.5, 0)] + [(sx * (a / 2 + mrg), sy * (b / 2 + mrg) * t) for sx in (1, -1)
                                           for sy in (1, -1) for t in (0.5, 0)]
            circ = math.hypot(a, b) / 2
        elif k == "circle":
            rad = r1(size / 2)
            ol = ring((0, 0), rad + mrg, 12)
            circ = rad
        else:
            a, b = r1(size * rng.uniform(1.2, 2.2)), r1(max(4.0, size * rng.uniform(0.25, 0.45)))
            L2 = (a - b) / 2
            ol = [(x, y) for x in (-L2, -L2 / 2, 0, L2 / 2, L2) for y in (b / 2 + mrg, -b / 2 - mrg)] + \
                 [(sx * (L2 + (b / 2 + mrg) * math.cos(t)), (b / 2 + mrg) * math.sin(t))
                  for sx in (1, -1) for t in (-1.0, 0, 1.0)]
            circ = a / 2
        if ang and k != "circle":
            ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            ol = [(x * ca - y * sa, x * sa + y * ca) for x, y in ol]
        p = fi.sample(rng, min(circ, mw / 2) * 0.5, centre_first=rng.random() < 0.6)
        if p is None:
            continue
        q = local(plane, p)
        q = (r1(q[0]), r1(q[1]))
        olw = [world(plane, (q[0] + x, q[1] + y)) for x, y in ol] + [world(plane, q)]
        if not all(fi.clearance(pp) > 0 for pp in olw):
            continue
        thick = material_depth(cl, fi, world(plane, q), below + 1)
        thru = rng.random() < 0.2 and k != "rrect"
        if thru:
            depth = below
        else:
            if thick < 5:
                continue
            depth = r1(thick * rng.uniform(0.25, 0.6))
            if depth < 1.5:
                continue
        if not clean_cut(cl, olw, fi, depth, not thru):
            continue
        cen = f".center({f(q[0])},{f(q[1])})" if (f(q[0]), f(q[1])) != ("0", "0") else ""
        angs = f",{ang}" if ang else ""
        if k == "rect":
            prof = f"{cen}.rect({f(a)},{f(b)})" if not ang else \
                f".sketch().push([{pt_code(q)}]).rect({f(a)},{f(b)},{ang}).finalize()"
        elif k == "rrect":
            prof = (f".sketch().push([{pt_code(q)}]).rect({f(a)},{f(b)}{angs}).vertices()"
                    f".fillet({f(min(a, b) * rng.uniform(0.15, 0.45))}).finalize()")
        elif k == "circle":
            prof = f"{cen}.circle({f(rad)})"
        else:
            prof = f"{cen}.slot2D({f(a)},{f(b)}{angs})"
        tail = ".cutThruAll()" if thru else f".cutBlind({f(-depth)})"
        return sel + ".workplane()" + prof + tail
    return None


def adjacent_faces(s, e):
    try:
        return list(s.ancestors(e, "Face"))
    except Exception:  # noqa: BLE001
        return []


def edge_point(e):
    t = 0.37 if e.geomType() == "CIRCLE" and e.IsClosed() else 0.5
    p = e.positionAt(t)
    return (p.x, p.y, p.z)


def near_ok(w, pts, edges):
    try:
        got = ev(w, ".edges(near(" + ",".join(f"({f(p[0])},{f(p[1])},{f(p[2])})" for p in pts) + "))").vals()
    except Exception:  # noqa: BLE001
        return False
    if len(got) != len(edges):
        return False
    return all(any(g.isSame(e) for g in got) for e in edges)


def face_size(fc):
    per = sum(e.Length() for e in fc.Edges())
    return 2 * fc.Area() / max(per, 1e-9)


def op_edges(rng, st, T):
    """Fillet or chamfer a group of real edges."""
    w, s = st["w"], st["s"]
    edges = [e for e in s.Edges() if e.geomType() in ("LINE", "CIRCLE") and e.Length() > 4]
    if not edges:
        return None
    mode = rng.random()
    sel = None
    group = []
    if mode < 0.15 and st["simple"]:
        cand = ev(w, ".edges('|Z')").vals()
        if 2 <= len(cand) <= 12 and all(e.geomType() == "LINE" for e in cand):
            sel, group = ".edges('|Z')", cand
    elif mode < 0.3:
        fi = pick_face(rng, s, side_p=0.15)
        if fi is not None:
            fs = face_selector(w, s, fi, rng)
            if fs is not None and fs.startswith(".faces('") and len(fi.face.Edges()) <= 16:
                sel, group = fs, fi.face.Edges()
    if sel is None:
        seed = rng.choice(edges)
        if seed.geomType() == "CIRCLE":
            same = [e for e in edges if e.geomType() == "CIRCLE" and abs(e.radius() - seed.radius()) < 0.05
                    and abs(edge_point(e)[2] - edge_point(seed)[2]) < 0.05]
        else:
            dv = seed.tangentAt(0.5)
            same = [e for e in edges if e.geomType() == "LINE" and abs(abs(e.tangentAt(0.5).dot(dv)) - 1) < 1e-6
                    and abs(e.Length() - seed.Length()) < 0.05]
        rng.shuffle(same)
        k = rng.choice([1, 2, 4, 4, 6])
        group = [seed] + [e for e in same if not e.isSame(seed)][:k - 1]
        pts = [edge_point(e) for e in group]
        if not near_ok(w, pts, group):
            group = [seed]
            pts = [edge_point(seed)]
            if not near_ok(w, pts, group):
                return None
        sel = ".edges(near(" + ",".join(f"({f(p[0])},{f(p[1])},{f(p[2])})" for p in pts) + "))"
    lim = 1e9
    for e in group:
        for fc in adjacent_faces(s, e):
            lim = min(lim, face_size(fc))
        if e.geomType() == "LINE":
            lim = min(lim, e.Length())
    if lim > 1e8 or lim < 3:
        return None
    nf = len(s.Faces())
    fillet = rng.random() < 0.55
    for frac in (rng.uniform(0.2, 0.45), 0.15):
        r = r1(max(1.0, lim * frac))
        if r >= lim * 0.5:
            continue
        seg = sel + (f".fillet({f(r)})" if fillet else f".chamfer({f(r)})")
        w2, s2 = try_seg(w, seg)
        if w2 is not None and len(s2.Faces()) > nf:
            return seg, w2, s2
    return None


def op_shell(rng, st, T):
    w, s = st["w"], st["s"]
    bb = fbb(s)
    t = r1(max(2.0, min(bb.xlen, bb.ylen, bb.zlen) * rng.uniform(0.06, 0.12)))
    if bb.zlen < 3 * t + 6:
        return None
    sel = rng.choice([".faces('>Z')", ".faces('>Z')", ".faces('<Z')"])
    seg = f"{sel}.shell({f(-t)})"
    w2, s2 = try_seg(w, seg)
    if w2 is None or s2.Volume() > 0.85 * s.Volume():
        return None
    return seg, w2, s2


def generate_mix(rng, hard=False):
    for _attempt in range(6):
        res = build_bodies(rng, hard)
        if res is None:
            continue
        part, body_code, kind, single = res
        bb0 = fbb(part.s)
        T = max(bb0.xlen, bb0.ylen)
        st = {"w": part.w, "s": part.s, "simple": single and kind in ("box", "rect", "ngon", "outline")}
        nops = rng.choice([0, 1, 2, 2, 3, 3, 4, 4]) if not hard else rng.randint(3, 7)
        segs = []
        if single and kind in ("box", "rect", "rrect", "circle", "ngon", "outline") and rng.random() < 0.3:
            out = op_shell(rng, st, T)
            if out is not None:
                segs.append(out[0])
                st["w"], st["s"] = out[1], out[2]
                st["simple"] = False
        tries = 0
        while len(segs) < nops and tries < nops * 3:
            tries += 1
            k = rng.choices(["holes", "pocket", "edges"], weights=[4, 2.5, 4])[0]
            try:
                if k == "edges":
                    out = op_edges(rng, st, T)
                    if out is None:
                        continue
                    seg, w2, s2 = out
                else:
                    seg = (op_holes if k == "holes" else op_pocket)(rng, st, T)
                    if seg is None:
                        continue
                    w2, s2 = try_seg(st["w"], seg)
                    if w2 is None or s2.Volume() > st["s"].Volume() - 1:
                        continue
            except Exception:  # noqa: BLE001
                continue
            segs.append(seg)
            st["w"], st["s"] = w2, s2
        if hard and len(segs) < 3:
            continue
        bb = st["s"].BoundingBox()
        cen = ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
        if max(abs(c) for c in cen) > 10:
            continue
        return "import cadquery as cq\nr=" + body_code + "".join(segs) + orient(rng)
    return "import cadquery as cq\nr=cq.Workplane('XY').box(120,80,20)" + orient(rng)


def generate(rng):
    return generate_mix(rng, hard=False)
