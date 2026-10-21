"""Structural / aluminium extrusion pieces: a 2D section (angle, channel, Z, I-beam, T, rectangular tube, T-slot) drawn
in the XY plane with polyline (or nested rects), extruded along Z to length, with concave root fillets / concentric
heel rounds / small tip rounds on the long edges, cross holes, counterbores, countersinks or slots drilled through the
flanges, and chamfered ends.

Side-face workplanes use the default ProjectedOrigin: on a face with outward normal n (horizontal), local x is
Z x n = (-n_y, n_x, 0) dotted with the global point and local y is the global z.
"""
from datagen.families.common import f, orient


def _r(p):
    return (round(p[0], 1), round(p[1], 1))


def _pt3(p, z=0.0):
    return f"({f(p[0])},{f(p[1])},{f(z)})"


def _pts(pts):
    return "[" + ",".join(f"({f(x)},{f(y)})" for x, y in pts) + "]"


def _angle(rng, S):
    a = S * rng.uniform(0.6, 1.0)
    b = a * rng.choice([1.0, 1.0, rng.uniform(0.5, 1.0)])
    t = max(2.5, a * rng.uniform(0.07, 0.15))
    rin = t * rng.uniform(0.4, 1.0)
    pts = [(0, 0), (a, 0), (a, t), (t, t), (t, b), (0, b)]
    return dict(pts=pts, concave=[((t, t), rin)], heel=[((0, 0), rin + t)],
                tips=[(a, 0), (a, t), (t, b), (0, b)], tmin=t,
                walls=[((max(t + rin, rin + t), 0), (a, 0), (0, -1), t),
                       ((0, b), (0, max(t + rin, rin + t)), (-1, 0), t)])


def _channel(rng, S):
    H = S * rng.uniform(0.6, 1.0)
    B = H * rng.uniform(0.3, 0.7)
    tw = max(2.5, H * rng.uniform(0.05, 0.1))
    tf = max(2.5, tw * rng.uniform(1.0, 1.5))
    rin = min(tw, tf) * rng.uniform(0.4, 1.0)
    rh = rin + min(tw, tf)
    pts = [(0, 0), (B, 0), (B, tf), (tw, tf), (tw, H - tf), (B, H - tf), (B, H), (0, H)]
    return dict(pts=pts, concave=[((tw, tf), rin), ((tw, H - tf), rin)], heel=[((0, 0), rh), ((0, H), rh)],
                tips=[(B, 0), (B, tf), (B, H - tf), (B, H)], tmin=min(tw, tf),
                walls=[((0, max(tf + rin, rh)), (0, H - max(tf + rin, rh)), (-1, 0), tw),
                       ((max(tw + rin, rh), 0), (B, 0), (0, -1), tf),
                       ((B, H), (max(tw + rin, rh), H), (0, 1), tf)])


def _zed(rng, S):
    H = S * rng.uniform(0.6, 1.0)
    B = H * rng.uniform(0.3, 0.6)
    t = max(2.5, H * rng.uniform(0.05, 0.1))
    rin = t * rng.uniform(0.4, 1.0)
    rh = rin + t
    pts = [(-(B - t), 0), (t, 0), (t, H - t), (B, H - t), (B, H), (0, H), (0, t), (-(B - t), t)]
    return dict(pts=pts, concave=[((t, H - t), rin), ((0, t), rin)], heel=[((t, 0), rh), ((0, H), rh)],
                tips=[(-(B - t), 0), (-(B - t), t), (B, H - t), (B, H)], tmin=t,
                walls=[((-(B - t), 0), (t - rh, 0), (0, -1), t),
                       ((B, H), (rh, H), (0, 1), t),
                       ((t, max(t + rin, rh)), (t, H - max(t + rin, rh)), (1, 0), t)])


def _ibeam(rng, S):
    H = S * rng.uniform(0.6, 1.0)
    B = H * rng.uniform(0.45, 1.0)
    tw = max(2.5, H * rng.uniform(0.04, 0.08))
    tf = max(2.5, tw * rng.uniform(1.2, 1.8))
    rin = tw * rng.uniform(0.5, 1.5)
    x0, x1 = (B - tw) / 2, (B + tw) / 2
    pts = [(0, 0), (B, 0), (B, tf), (x1, tf), (x1, H - tf), (B, H - tf), (B, H), (0, H), (0, H - tf), (x0, H - tf),
           (x0, tf), (0, tf)]
    return dict(pts=pts, concave=[((x1, tf), rin), ((x1, H - tf), rin), ((x0, H - tf), rin), ((x0, tf), rin)],
                heel=[], tips=[(0, 0), (B, 0), (B, tf), (B, H - tf), (B, H), (0, H), (0, H - tf), (0, tf)],
                tmin=min(tw, tf),
                walls=[((x1 + rin, 0), (B, 0), (0, -1), tf), ((B, H), (x1 + rin, H), (0, 1), tf),
                       ((x1, tf + rin), (x1, H - tf - rin), (1, 0), tw)])


def _tee(rng, S):
    B = S * rng.uniform(0.6, 1.0)
    H = B * rng.uniform(0.5, 1.0)
    tf = max(2.5, B * rng.uniform(0.06, 0.12))
    tw = max(2.5, tf * rng.uniform(0.8, 1.2))
    rin = tw * rng.uniform(0.4, 1.0)
    x0, x1 = (B - tw) / 2, (B + tw) / 2
    pts = [(x0, 0), (x1, 0), (x1, H - tf), (B, H - tf), (B, H), (0, H), (0, H - tf), (x0, H - tf)]
    return dict(pts=pts, concave=[((x1, H - tf), rin), ((x0, H - tf), rin)], heel=[],
                tips=[(x0, 0), (x1, 0), (B, H - tf), (B, H), (0, H), (0, H - tf)], tmin=min(tw, tf),
                walls=[((B, H), (x1 + rin, H), (0, 1), tf), ((x0 - rin, H), (0, H), (0, 1), tf),
                       ((x1, 0), (x1, H - tf - rin), (1, 0), tw)])


def _wall_code(rng, walls, L, shift, d_max_global):
    """Holes / slots through one or two of the profile's walls."""
    out = ""
    rng.shuffle(walls)
    nw = 1 if rng.random() < 0.6 else 2
    for (A, B, n, t) in walls[:nw]:
        A = (A[0] - shift[0], A[1] - shift[1])
        B = (B[0] - shift[0], B[1] - shift[1])
        xd = (-n[1], n[0])
        sa, sb = sorted((A[0] * xd[0] + A[1] * xd[1], B[0] * xd[0] + B[1] * xd[1]))
        width = sb - sa
        if width < 6:
            continue
        d = min(width * rng.uniform(0.3, 0.6), d_max_global, rng.uniform(5, 16))
        if d < 3:
            continue
        m = d / 2 + max(1.0, 0.2 * d)
        lo, hi = sa + m, sb - m
        if hi < lo:
            continue
        # face pick point: on the outer face between the end and the first hole, mid width. workplane(origin=(0,0,0))
        # puts the local origin at the projection of the global origin: local y = global z
        e = max(1.5 * d, rng.uniform(0.08, 0.2) * L)
        mid = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
        zp = -L / 2 + 0.4 * e - d * 0.1
        face = f".faces(near({_pt3(mid, zp)})).workplane(origin=(0,0,0))"
        two_rows = (hi - lo) > 2.6 * d and rng.random() < 0.35
        if two_rows:
            g = rng.uniform(0.5, 1.0) * (hi - lo)
            xs = [(lo + hi) / 2 - g / 2, (lo + hi) / 2 + g / 2]
        else:
            xs = [rng.uniform(lo, hi) if rng.random() < 0.3 else (lo + hi) / 2]
        span = L - 2 * e
        k = rng.random()
        if k < 0.2:                                              # slots along the length
            sl = min(rng.uniform(2, 4) * d, span / 2)
            ns = rng.choice([1, 2, 2, 3])
            if ns > 1 and (span - sl) / (ns - 1) < sl + d:
                ns = 1
            zs = [0.0] if ns == 1 else [-(span - sl) / 2 + i * (span - sl) / (ns - 1) for i in range(ns)]
            pts = [(x, z) for x in xs for z in zs]
            out += face + f".pushPoints({_pts(pts)}).slot2D({f(sl + d)},{f(d)},90).cutThruAll()"
            continue
        nh = rng.choice([1, 2, 2, 3, 3, 4, 5, 6])
        pitch = span / max(nh - 1, 1)
        while nh > 1 and pitch < 2.2 * d:
            nh -= 1
            pitch = span / max(nh - 1, 1)
        if nh == 1:
            zs = [0.0]
        else:
            zs = [-span / 2 + i * pitch for i in range(nh)]
        if len(xs) == 1 and nh > 1 and rng.random() < 0.5:
            pat = f".center({f(xs[0])},0).rarray(1,{f(pitch)},1,{nh})"
        else:
            pat = f".pushPoints({_pts([(x, z) for x in xs for z in zs])})"
        k2 = rng.random()
        if k2 < 0.6 or t < 4:
            depth = "" if rng.random() < 0.7 else f",{f(t * 1.5)}"
            out += face + pat + f".hole({f(d)}{depth})"
        elif k2 < 0.8 and (hi - lo) / len(xs) > 0:
            cb = min(d * 1.6, width - 2)
            if cb <= d + 1:
                out += face + pat + f".hole({f(d)})"
            else:
                out += face + pat + f".cboreHole({f(d)},{f(cb)},{f(t * rng.uniform(0.3, 0.5))})"
        else:
            cs = min(d * 1.8, width - 2, d + 1.2 * t)
            if cs <= d + 1:
                out += face + pat + f".hole({f(d)})"
            else:
                out += face + pat + f".cskHole({f(d)},{f(cs)},90)"
    return out


def _tslot(rng, L):
    a = rng.uniform(30, 70)
    nx = rng.choice([1, 1, 1, 2])                    # 1x1 or 1x2 module
    W, H = a * nx, a
    wo = a * rng.uniform(0.2, 0.26)                   # slot opening
    lt = a * rng.uniform(0.07, 0.09)                  # lip thickness
    wi = a * rng.uniform(0.38, 0.44)                  # undercut width
    d1 = a * rng.uniform(0.06, 0.08)
    d2 = a * rng.uniform(0.04, 0.07)                  # tapered slot bottom
    # undercut corners stay clear of the corner diagonals: wi/2 + lt + d1 <= 0.39 a
    prof = [(-wo / 2, 0), (-wo / 2, lt), (-wi / 2, lt), (-wi / 2, lt + d1), (-wo / 2, lt + d1 + d2),
            (wo / 2, lt + d1 + d2), (wi / 2, lt + d1), (wi / 2, lt), (wo / 2, lt), (wo / 2, 0)]
    corners = [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)]
    pts = []
    for i in range(4):
        P, Q = corners[i], corners[(i + 1) % 4]
        ln = abs(Q[0] - P[0]) + abs(Q[1] - P[1])
        u = ((Q[0] - P[0]) / ln, (Q[1] - P[1]) / ln)
        v = (-u[1], u[0])
        pts.append(P)
        nslot = round(ln / a)
        for k in range(nslot):
            c = (k + 0.5) * a
            for s, dpt in prof:
                pts.append((P[0] + (c + s) * u[0] + dpt * v[0], P[1] + (c + s) * u[1] + dpt * v[1]))
    pts = [_r(p) for p in pts]
    s = f"cq.Workplane('XY').workplane(offset={f(-L / 2)}).polyline({_pts(pts)}).close()"
    bore = a * rng.uniform(0.14, 0.24)
    centres = [((i + 0.5) * a - W / 2, 0) for i in range(nx)]
    if rng.random() < 0.8:
        if nx == 1:
            s += f".circle({f(bore / 2)})"
        else:
            s += f".pushPoints({_pts(centres)}).circle({f(bore / 2)})"
    s += f".extrude({f(L)})"
    if rng.random() < 0.7:
        rc = a * rng.uniform(0.02, 0.05)
        s += f".edges(near({','.join(_pt3(c) for c in corners)})).fillet({f(rc)})"
    if rng.random() < 0.7:                            # access / connector holes across the profile
        d = min(wo * rng.uniform(0.6, 0.95), bore * 1.1) if rng.random() < 0.5 else bore * rng.uniform(0.9, 1.3)
        e = rng.uniform(0.05, 0.2) * L + d
        side = rng.choice(["x", "y"])
        if side == "y" or nx == 1:
            n = (0, -1) if rng.random() < 0.5 else (0, 1)
            P = (centres[0][0] + (wo / 2 + a / 2) / 2, n[1] * H / 2)
            xs = [c[0] * -n[1] for c in centres]                 # local x = global x * (-n_y)
            z_s = [-L / 2 + e, L / 2 - e] if rng.random() < 0.6 else [-L / 2 + e]
            pts = [(x, z) for x in xs for z in z_s]
        else:
            n = (1, 0) if rng.random() < 0.5 else (-1, 0)
            P = (n[0] * W / 2, (wo / 2 + a / 2) / 2)
            z_s = [-L / 2 + e, L / 2 - e] if rng.random() < 0.6 else [-L / 2 + e]
            pts = [(0, z) for z in z_s]
        s += f".faces(near({_pt3(P, 0)})).workplane(origin=(0,0,0)).pushPoints({_pts(pts)}).hole({f(d)})"
    if rng.random() < 0.3 and nx == 1:
        s += f".faces('>Z').workplane(origin=(0,0,0)).hole({f(bore * rng.uniform(1.0, 1.15))},{f(rng.uniform(0.1, 0.25) * L)})"
    return s


def generate(rng):
    L = round(rng.uniform(100, 200) / 2, 1) * 2         # even tenths: offset -L/2 and length L stay exact
    kind = rng.choice(["angle", "angle", "channel", "channel", "zed", "ibeam", "ibeam", "tee", "tube", "tube",
                       "tslot", "tslot"])
    if kind == "tslot":
        return "import cadquery as cq\nr=" + _tslot(rng, L) + orient(rng)
    S = min(rng.uniform(35, 95), 0.75 * L)
    z0 = -L / 2
    if kind == "tube":
        W = S * rng.uniform(0.6, 1.0)
        H = W * rng.choice([1.0, rng.uniform(0.4, 1.0)])
        t = max(2.0, min(W, H) * rng.uniform(0.05, 0.12))
        s = (f"cq.Workplane('XY').workplane(offset={f(z0)}).rect({f(W)},{f(H)}).rect({f(W - 2 * t)},{f(H - 2 * t)})"
             f".extrude({f(L)})")
        ro = t * rng.uniform(1.2, 2.5) if rng.random() < 0.8 else 0
        if ro:
            outer = [(sx * W / 2, sy * H / 2) for sx in (-1, 1) for sy in (-1, 1)]
            inner = [(sx * (W / 2 - t), sy * (H / 2 - t)) for sx in (-1, 1) for sy in (-1, 1)]
            s += f".edges(near({','.join(_pt3(_r(p)) for p in outer)})).fillet({f(ro)})"
            s += f".edges(near({','.join(_pt3(_r(p)) for p in inner)})).fillet({f(max(ro - t, 0.5))})"
        walls = [((-W / 2 + ro + t, -H / 2), (W / 2 - ro - t, -H / 2), (0, -1), t),
                 ((W / 2, -H / 2 + ro + t), (W / 2, H / 2 - ro - t), (1, 0), t),
                 ((-W / 2 + ro + t, H / 2), (W / 2 - ro - t, H / 2), (0, 1), t)]
        if rng.random() < 0.85:
            s += _wall_code(rng, walls, L, (0, 0), 0.5 * min(W, H))
        if rng.random() < 0.5:
            s += f".faces('>Z or <Z').chamfer({f(t * rng.uniform(0.2, 0.4))})"
        return "import cadquery as cq\nr=" + s + orient(rng)

    prof = {"angle": _angle, "channel": _channel, "zed": _zed, "ibeam": _ibeam, "tee": _tee}[kind](rng, S)
    xs = [p[0] for p in prof["pts"]]
    ys = [p[1] for p in prof["pts"]]
    shift = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)
    if rng.random() < 0.5:                                     # mirror the section for variety
        mx = -1
    else:
        mx = 1

    def tr(p):
        return _r((mx * (p[0] - shift[0]), p[1] - shift[1]))

    pts = [tr(p) for p in prof["pts"]]
    if mx == -1:
        pts = pts[::-1]
    s = f"cq.Workplane('XY').workplane(offset={f(z0)}).polyline({_pts(pts)}).close().extrude({f(L)})"
    tmin = prof["tmin"]
    if rng.random() < 0.8 and prof["concave"]:
        rin = prof["concave"][0][1]
        s += f".edges(near({','.join(_pt3(tr(p)) for p, _ in prof['concave'])})).fillet({f(rin)})"
        if prof["heel"] and rng.random() < 0.7:
            s += f".edges(near({','.join(_pt3(tr(p)) for p, _ in prof['heel'])})).fillet({f(prof['heel'][0][1])})"
    rtip = 0
    if rng.random() < 0.4:
        rtip = round(tmin * rng.uniform(0.15, 0.35), 1)
        s += f".edges(near({','.join(_pt3(tr(p)) for p in prof['tips'])})).fillet({f(rtip)})"
    walls = []
    for A, B, n, t in prof["walls"]:
        A2, B2 = tr((A[0], A[1])), tr((B[0], B[1]))
        walls.append((A2, B2, (mx * n[0], n[1]), t))
    if rng.random() < 0.85:
        s += _wall_code(rng, walls, L, (0, 0), 0.5 * S)
    ch = tmin * rng.uniform(0.15, 0.3) if not rtip else rtip * rng.uniform(0.3, 0.5)
    if rng.random() < 0.4 and ch >= 0.5:
        s += f".faces('>Z or <Z').chamfer({f(ch)})"
    return "import cadquery as cq\nr=" + s + orient(rng)
