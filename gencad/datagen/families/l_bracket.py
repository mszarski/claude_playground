"""L / angle brackets: two flanges at 90 degrees (equal or unequal legs), mounting holes or slots in each leg
(plain, counterbored or countersunk), inner bend radius (sheet-metal style adds the matching outer radius), rounded
or chamfered free corners, and optional triangular gussets.

Built in the XY frame: the L cross-section is sketched on XY and extruded symmetrically along Z (the bracket width).
The base leg runs along +X with its outer face at y = -H/2; the upright leg rises along +Y at x = -L/2. Gussets are
triangles sketched on XY workplanes at their Z station and unioned in. Holes are drilled from the inner faces.
"""
import math

from datagen.families.common import f, orient

C45 = math.sqrt(0.5)


def _pts(pts):
    return "[" + ",".join(f"({f(a)},{f(b)})" for a, b in pts) + "]"


def _p3(pts):
    return ",".join(f"({f(a)},{f(b)},{f(c)})" for a, b, c in pts)


def generate(rng):
    S = rng.uniform(100, 200)                                   # longest side
    shape = rng.random()
    if shape < 0.35:                                            # equal legs
        L, H, W = 1.0, rng.uniform(0.85, 1.0), rng.uniform(0.3, 0.8)
    elif shape < 0.75:                                          # unequal legs
        L, H, W = 1.0, rng.uniform(0.35, 0.8), rng.uniform(0.3, 0.9)
    else:                                                       # wide angle (long extrusion)
        W, L, H = 1.0, rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    if rng.random() < 0.5:
        L, H = H, L
    k = S / max(L, H, W)
    L, H, W = L * k, H * k, W * k
    t = max(3.0, rng.uniform(0.05, 0.12) * min(L, H))
    t = round(min(t, 0.25 * W, 0.2 * min(L, H)), 1)
    style = rng.choices(["sheet", "sketch_fillet", "fillet", "sharp"], [0.35, 0.2, 0.35, 0.1])[0]
    ri = rng.uniform(0.5, 1.5) * t if style == "sheet" else rng.uniform(0.4, 1.3) * t
    if style == "sharp":
        ri = 0.0
    ro = ri + t if style == "sheet" else 0.0
    ri, ro = round(ri, 1), round(ro, 1)
    x0, y0 = -L / 2, -H / 2                                     # outer corner
    xi, yi = x0 + t, y0 + t                                     # inner corner
    yt = y0 + H

    # ---------------- profile
    if style in ("sheet", "sketch_fillet"):
        s = f"cq.Workplane('XY').moveTo({f(L / 2)},{f(y0)})"
        if ro > 0:
            cx, cy = x0 + ro, y0 + ro
            s += (f".lineTo({f(cx)},{f(y0)}).threePointArc(({f(cx - ro * C45)},{f(cy - ro * C45)}),"
                  f"({f(x0)},{f(cy)}))")
        else:
            s += f".lineTo({f(x0)},{f(y0)})"
        cx, cy = xi + ri, yi + ri
        s += (f".lineTo({f(x0)},{f(yt)}).lineTo({f(xi)},{f(yt)}).lineTo({f(xi)},{f(cy)})"
              f".threePointArc(({f(cx - ri * C45)},{f(cy - ri * C45)}),({f(cx)},{f(yi)}))"
              f".lineTo({f(L / 2)},{f(yi)}).close().extrude({f(W / 2)},both=True)")
    else:
        prof = [(L / 2, y0), (x0, y0), (x0, yt), (xi, yt), (xi, yi), (L / 2, yi)]
        s = f"cq.Workplane('XY').polyline({_pts(prof)}).close().extrude({f(W / 2)},both=True)"
        if style == "fillet":
            s += f".edges(near(({f(xi)},{f(yi)},{f(W / 4)}))).fillet({f(ri)})"
        if rng.random() < 0.35:
            s += f".edges(near(({f(x0)},{f(y0)},{f(W / 4)}))).chamfer({f(rng.uniform(0.2, 0.4) * t)})"

    # ---------------- hole sizing / gussets
    d = max(4.0, min(rng.uniform(0.1, 0.2) * W, 2.2 * t + 4, 18.0))
    kind = rng.choices(["hole", "csk", "cbore", "slot"], [0.4, 0.25, 0.15, 0.2])[0]
    if kind == "cbore" and t < 0.45 * d + 2.5:
        kind = "hole"
    if kind == "csk" and t < 0.3 * d + 1.5:
        kind = "hole"
    head = {"hole": d / 2, "csk": d, "cbore": 0.85 * d, "slot": d / 2}[kind]
    e = max(0.8 * d + 1.5, head + 1.5)                          # edge distance (centre to edge)

    gus = rng.choices([0, 1, 2], [0.5, 0.2, 0.3])[0]
    gt = round(max(t, rng.uniform(0.8, 1.5) * t), 1)
    ws = None
    if gus == 1:
        gzs = [0.0]
        p_lo, p_hi = gt / 2 + head + 2, W / 2 - e
        if p_lo <= p_hi:
            ws = [-1, 1]
    elif gus == 2:
        gz = W / 2 - gt / 2 - (0 if rng.random() < 0.6 else rng.uniform(0, 0.1) * W)
        gzs = [-gz, gz]
        lim = gz - gt / 2 - head - 2
        p_lo, p_hi = max(0.3 * lim, d + 2), lim
        if lim > 0:
            ws = [0] if (p_hi < p_lo or rng.random() < 0.5) else [-1, 1]
    if ws is None:
        gus, gzs = 0, []
        p_lo, p_hi = d + 2, W / 2 - e
        opts = [[0]]
        if p_hi > p_lo:
            opts += [[-1, 1], [-1, 1]]
            if p_hi > 1.6 * d + 2 and W > 70:
                opts += [[-1, 0, 1]]
        ws = rng.choice(opts)
    p = rng.uniform(p_lo, p_hi) if len(ws) > 1 else 0.0
    wv = [sgn * p for sgn in ws]
    wedge = W / 2 - max(abs(w) for w in wv)

    def stations(lo, hi):
        """Hole stations along one leg between lo (near the bend) and hi (free-end limit)."""
        if hi < lo:
            return None
        if hi - lo > 2.5 * d + 6 and rng.random() < 0.55:
            return [rng.uniform(lo, lo + 0.3 * (hi - lo)), hi]
        u = rng.random()
        return [hi if u < 0.3 else (lo + hi) / 2 if u < 0.7 else rng.uniform(lo, hi)]

    bend_clear = max(ri, ro - t) + head + 1.5
    xs = stations(xi + bend_clear, L / 2 - e)
    ys = stations(yi + bend_clear, yt - e)

    # ---------------- free corners
    corner = rng.choices(["none", "fillet", "chamfer"], [0.3, 0.5, 0.2])[0]
    if corner != "none":
        rmax = min(wedge, e, W / 4)
        if xs:
            rmax = min(rmax, L / 2 - max(xs))
        if ys:
            rmax = min(rmax, yt - max(ys))
        rc = rng.uniform(0.4, 1.0) * rmax
        if corner == "chamfer":
            rc = min(rc, 0.6 * rmax)
        if rc >= 1.0:
            which = rng.choice(["both", "both", "base", "leg"])
            pts = []
            if which in ("both", "base"):
                pts += [(L / 2, y0 + t / 2, -W / 2), (L / 2, y0 + t / 2, W / 2)]
            if which in ("both", "leg"):
                pts += [(x0 + t / 2, yt, -W / 2), (x0 + t / 2, yt, W / 2)]
            s += f".edges(near({_p3(pts)})).{'fillet' if corner == 'fillet' else 'chamfer'}({f(rc)})"

    # ---------------- gussets
    pre = ""
    if gus:
        ga = rng.uniform(0.4, 0.75) * (L - t)
        gb = rng.uniform(0.4, 0.75) * (H - t)
        if rng.random() < 0.5:
            tri = [(xi - 0.5, yi - 0.5), (xi + ga, yi - 0.5), (xi - 0.5, yi + gb)]
        else:                                                    # trimmed tips
            c = min(ga, gb) * rng.uniform(0.08, 0.15)
            tri = [(xi - 0.5, yi - 0.5), (xi + ga, yi - 0.5), (xi + ga, yi + c),
                   (xi + c, yi + gb), (xi - 0.5, yi + gb)]
        for j, gz in enumerate(gzs):
            pre += (f"g{j}=cq.Workplane('XY',origin=(0,0,{f(gz)})).polyline({_pts(tri)}).close()"
                    f".extrude({f(gt / 2)},both=True)\n")
            s += f".union(g{j})"

    # ---------------- holes, drilled from the inner faces of both legs
    base_wp = f".faces('<Y').workplane(centerOption='ProjectedOrigin',origin=(0,0,0),invert=True,offset={f(t)})"
    leg_wp = f".faces('<X').workplane(centerOption='ProjectedOrigin',origin=(0,0,0),invert=True,offset={f(t)})"
    sl = rng.uniform(1.4, 2.6) * d                              # slot overall length

    def cut(wp, pts, slot_ang, room):
        nonlocal s
        pp = ".pushPoints(" + _pts(pts) + ")"
        if kind == "hole":
            s += wp + pp + f".hole({f(d)})"
        elif kind == "csk":
            s += wp + pp + f".cskHole({f(d)},{f(2 * d)},{rng.choice([82, 90, 90])})"
        elif kind == "cbore":
            s += wp + pp + f".cboreHole({f(d)},{f(1.7 * d)},{f(min(0.45 * d + 1, 0.55 * t))})"
        else:
            ln = min(sl, room)
            if ln < d * 1.2:
                s += wp + pp + f".hole({f(d)})"
            else:
                s += wp + pp + f".slot2D({f(ln)},{f(d)},{slot_ang}).cutThruAll()"

    if xs:                                                       # base: u = x, v = -z
        pts = [(x, -w) for x in xs for w in wv]
        room = 2 * min(xs[0] - (xi + bend_clear), L / 2 - e - xs[0]) + d if len(xs) == 1 else d
        cut(base_wp, pts, 0, room)
    if ys and rng.random() < 0.92:                               # upright: u = -y, v = -z
        pts = [(-y, -w) for y in ys for w in wv]
        room = 2 * min(ys[0] - (yi + bend_clear), yt - e - ys[0]) + d if len(ys) == 1 else d
        cut(leg_wp, pts, 0, room)
    return "import cadquery as cq\n" + pre + "r=" + s + orient(rng)
