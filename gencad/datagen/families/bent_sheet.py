"""Sheet-metal-looking parts made as solids: a constant-thickness bent profile (L, Z/joggle, U, hat, lipped channel,
multi-bend bracket) drawn in the XY plane as the offset outline of a centreline, extruded along Z (the sheet width),
with the bends rounded (inner radius r on the inside corner, r + t on the outside corner: concentric), holes / slots
punched through the flat faces, notches and corner rounds at the free ends.

Side-face workplanes use workplane(origin=(0,0,0)): on a face with (horizontal) outward normal n, local x is
(Z x n) . p = -n_y*x + n_x*y and local y is the global z.
"""
import math

from datagen.families.common import f, orient


def _pts(pts):
    return "[" + ",".join(f"({f(x)},{f(y)})" for x, y in pts) + "]"


def _p3(p, z):
    return f"({f(p[0])},{f(p[1])},{f(z)})"


def _pt_seg(x, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    ll = dx * dx + dy * dy
    u = 0 if ll == 0 else max(0, min(1, ((x[0] - a[0]) * dx + (x[1] - a[1]) * dy) / ll))
    return math.hypot(x[0] - a[0] - u * dx, x[1] - a[1] - u * dy)


def _seg_dist(p, q, a, b):
    """Exact distance between 2D segments pq and ab."""
    def cr(o, x, y):
        return (x[0] - o[0]) * (y[1] - o[1]) - (x[1] - o[1]) * (y[0] - o[0])
    d1, d2, d3, d4 = cr(a, b, p), cr(a, b, q), cr(p, q, a), cr(p, q, b)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(_pt_seg(p, a, b), _pt_seg(q, a, b), _pt_seg(a, p, q), _pt_seg(b, p, q))


def _shape(rng):
    """Turn angles (deg, + = left) and relative segment lengths."""
    k = rng.choice(["L", "L", "L", "Z", "Z", "U", "U", "hat", "lipped", "multi", "multi"])
    if k == "L":
        th = 90 if rng.random() < 0.7 else rng.choice([rng.uniform(45, 80), rng.uniform(100, 135)])
        return k, [th], [1.0, rng.uniform(0.3, 1.0)]
    if k == "Z":
        th = 90 if rng.random() < 0.6 else rng.uniform(30, 70)          # Z or joggle/offset
        return k, [th, -th], [rng.uniform(0.4, 1.0), rng.uniform(0.3, 1.0), rng.uniform(0.4, 1.0)]
    if k == "U":
        a = rng.uniform(0.3, 1.0)
        b = a if rng.random() < 0.6 else rng.uniform(0.3, 1.0)
        return k, [90, 90], [a, rng.uniform(0.4, 1.0), b]
    if k == "hat":
        brim, wall, top = rng.uniform(0.2, 0.4), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.8)
        return k, [90, -90, -90, 90], [brim, wall, top, wall, brim]
    if k == "lipped":
        lip, fl, web = rng.uniform(0.12, 0.25), rng.uniform(0.35, 0.6), 1.0
        return k, [90, 90, 90, 90], [lip, fl, web, fl, lip]
    n = rng.choice([2, 3, 3, 4])
    turns = [rng.choice([90, -90, 90, -90, rng.uniform(40, 135), -rng.uniform(40, 135)]) for _ in range(n)]
    return k, turns, [rng.uniform(0.25, 1.0) for _ in range(n + 1)]


def _try(rng):
    kind, turns, lens = _shape(rng)
    S = rng.uniform(100, 200)                        # longest side
    # centreline in relative units
    pts, ang = [(0.0, 0.0)], 0.0
    for i, ln in enumerate(lens):
        pts.append((pts[-1][0] + ln * math.cos(ang), pts[-1][1] + ln * math.sin(ang)))
        if i < len(turns):
            ang += math.radians(turns[i])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ext = max(max(xs) - min(xs), max(ys) - min(ys))
    Wr = rng.uniform(0.3, 1.1) * ext                 # sheet width (along Z), relative
    sc = S / max(ext, Wr)
    pts = [(x * sc, y * sc) for x, y in pts]
    lens = [ln * sc for ln in lens]
    W = round(Wr * sc / 2, 1) * 2
    if W < 25:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    t = max(2.0, S * rng.uniform(0.018, 0.04))
    t_fill = 0.035 * (bw + 3) * (bh + 3) / sum(lens)            # keep volume / bbox above the sliver limit
    if t_fill > 0.065 * S:
        return None
    t = round(max(t, t_fill), 1)
    ri = round(t * rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]), 1)
    ro = round(ri + t, 1)
    tans = [math.tan(math.radians(abs(a)) / 2) for a in turns]
    # every segment must hold its bend allowances plus a flat
    flats = []
    for i, ln in enumerate(lens):
        ta = tans[i - 1] if i > 0 else 0.0
        tb = tans[i] if i < len(turns) else 0.0
        need = (ro + t) * (ta + tb) + 2 * t
        if ln < need:
            return None
        flats.append(((ri + t / 2) * ta, ln - (ri + t / 2) * tb))
    # no self intersection / near contact between non-adjacent segments
    segs = list(zip(pts[:-1], pts[1:]))
    for i in range(len(segs)):
        for j in range(i + 2, len(segs)):
            if _seg_dist(*segs[i], *segs[j]) < 2.5 * t + ri:
                return None
    # centre: shift so the outline's bbox is centred
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    pts = [(x - cx, y - cy) for x, y in pts]
    dirs = [((q[0] - p[0]) / ln, (q[1] - p[1]) / ln) for (p, q), ln in zip(segs, lens)]
    nls = [(-d[1], d[0]) for d in dirs]              # left normals
    left, right = [], []
    for i, p in enumerate(pts):
        if i == 0 or i == len(pts) - 1:
            n = nls[0] if i == 0 else nls[-1]
            m = (n[0] * t / 2, n[1] * t / 2)
        else:
            na, nb = nls[i - 1], nls[i]
            c = 1 + na[0] * nb[0] + na[1] * nb[1]
            m = ((na[0] + nb[0]) / c * t / 2, (na[1] + nb[1]) / c * t / 2)
        left.append((round(p[0] + m[0], 1), round(p[1] + m[1], 1)))
        right.append((round(p[0] - m[0], 1), round(p[1] - m[1], 1)))
    outline = left + right[::-1]
    return dict(kind=kind, turns=turns, lens=lens, pts=pts, dirs=dirs, nls=nls, left=left, right=right,
                outline=outline, W=W, t=t, ri=ri, ro=ro, flats=flats, segs=list(zip(pts[:-1], pts[1:])))


def _clear(g, i, h, n, reach):
    """Is the ray from point h (on segment i's centreline) in direction n clear of other segments for `reach`?"""
    q = (h[0] + n[0] * 400, h[1] + n[1] * 400)
    for j, (a, b) in enumerate(g["segs"]):
        if j == i:
            continue
        if _seg_dist(h, q, a, b) < reach:
            return False
    return True


def generate(rng):
    g = None
    while g is None:
        g = _try(rng)
    W, t, ri, ro = g["W"], g["t"], g["ri"], g["ro"]
    s = f"cq.Workplane('XY').workplane(offset={f(-W / 2)}).polyline({_pts(g['outline'])}).close().extrude({f(W)})"
    inner, outer = [], []
    for i, a in enumerate(g["turns"]):
        L, R = g["left"][i + 1], g["right"][i + 1]
        (inner if a > 0 else outer).append(L)
        (outer if a > 0 else inner).append(R)
    if rng.random() < 0.93:
        s += f".edges(near({','.join(_p3(p, 0) for p in inner)})).fillet({f(ri)})"
        s += f".edges(near({','.join(_p3(p, 0) for p in outer)})).fillet({f(ro)})"
    nseg = len(g["lens"])
    ends = [0, nseg - 1]
    # free-end treatment decided first so holes keep clear of it
    end_round = {e: 0.0 for e in ends}
    end_notch = {e: None for e in ends}
    for e in ends:
        fl = g["flats"][e]
        flat_len = fl[1] - fl[0]
        r = rng.random()
        if r < 0.35:
            end_round[e] = round(min(W * rng.uniform(0.08, 0.25), 0.4 * flat_len), 1)
        elif r < 0.45 and W < 90:
            end_round[e] = round(min(W * 0.42, 0.6 * flat_len), 1)        # rounded tab / lug end
        elif r < 0.65 and W > 40:
            wn = W * rng.uniform(0.15, 0.35)
            dn = min(flat_len * rng.uniform(0.1, 0.25), wn * 1.2)
            if dn > 2 * t:
                end_notch[e] = (wn, dn, rng.random() < 0.5)
    code_holes = ""
    order = list(range(nseg))
    rng.shuffle(order)
    nholes = 0
    for i in order:
        if rng.random() < 0.45 and not (nholes == 0 and i == order[-1]):
            continue
        u0, u1 = g["flats"][i]
        if i == 0:
            u0 = max(u0, end_round[0] + (end_notch[0][1] if end_notch[0] else 0))
        if i == nseg - 1:
            u1 = min(u1, g["lens"][i] - end_round[nseg - 1] - (end_notch[nseg - 1][1] if end_notch[nseg - 1] else 0))
        span = u1 - u0
        d = round(min(rng.uniform(0.03, 0.08) * max(W, sum(g['lens']) / 2), 0.3 * W, 0.45 * span), 1)
        if d < 3:
            continue
        m_u = d / 2 + max(t, 0.4 * d)
        m_z = d / 2 + max(t, 0.5 * d)
        lug = i in ends and end_round[i] >= 0.41 * W and rng.random() < 0.8
        p, dvec = g["pts"][i], g["dirs"][i]
        side = rng.choice([1, -1])
        n = (g["nls"][i][0] * side, g["nls"][i][1] * side)
        xdir = (-n[1], n[0])

        def at(u):
            return (p[0] + dvec[0] * u, p[1] + dvec[1] * u)

        if lug:
            u_end = g["lens"][i] if i == nseg - 1 else 0.0
            uc = u_end - W / 2 if i == nseg - 1 else W / 2
            dd = round(W * rng.uniform(0.25, 0.45), 1)
            us, zs, d, kind = [uc], [0.0], dd, "hole"
            m_z = dd / 2 + 2
        else:
            if span < 2 * m_u:
                continue
            k = rng.random()
            if k < 0.22 and span > 3 * d + 2 * m_u:
                kind = "slot"
            else:
                kind = rng.choice(["hole", "hole", "hole", "csk"]) if t >= 3 else "hole"
            nu = rng.choice([1, 1, 2, 2, 3]) if kind != "slot" else rng.choice([1, 2])
            nz = 1 if W - 2 * m_z < 2 * d else rng.choice([1, 2, 2])
            if kind == "slot":
                sl = min(span - 2 * m_u, d * rng.uniform(1.5, 3.5))
                m_u += sl / 2
                if span < 2 * m_u:
                    continue
            if nu > 1 and (span - 2 * m_u) / (nu - 1) < 2.5 * d:
                nu = 1
            lo_u, hi_u = u0 + m_u, u1 - m_u
            us = [(lo_u + hi_u) / 2] if nu == 1 else [lo_u + j * (hi_u - lo_u) / (nu - 1) for j in range(nu)]
            if nu > 1 and rng.random() < 0.5:                       # pull rows in from the margins
                c, sh = (lo_u + hi_u) / 2, rng.uniform(0.6, 1.0)
                us = [c + (u - c) * sh for u in us]
            zr = W / 2 - m_z
            zs = [0.0] if nz == 1 else [-zr * rng.uniform(0.6, 1.0), 0.0]
            if nz == 2:
                zs[1] = -zs[0]
        reach = t / 2 + d / 2 + 1.0
        thru = all(_clear(g, i, at(u), (-n[0], -n[1]), reach) and _clear(g, i, at(u), n, reach) for u in us)
        um = (u0 + u1) / 2
        fp = at(um)
        fp = (fp[0] + n[0] * t / 2, fp[1] + n[1] * t / 2)
        zp = W / 2 - 0.5 * min(max(t, 0.4 * d), W / 2 - max(abs(z) for z in zs) - d / 2)
        face = f".faces(near({_p3(fp, zp)})).workplane(origin=(0,0,0))"
        pl = [(at(u)[0] * xdir[0] + at(u)[1] * xdir[1], z) for u in us for z in zs]
        if len(us) > 1 and len(zs) == 1 and rng.random() < 0.4 and abs(zs[0]) < 1e-9:
            xc = (pl[0][0] + pl[-1][0]) / 2
            pat = f".center({f(xc)},0).rarray({f(abs(pl[-1][0] - pl[0][0]) / (len(us) - 1))},1,{len(us)},1)"
        else:
            pat = f".pushPoints({_pts(pl)})"
        if kind == "slot":
            depth = ".cutThruAll()" if thru else f".cutBlind({f(-1.5 * t)})"
            code_holes += face + pat + f".slot2D({f(sl + d)},{f(d)},{0}){depth}"
        elif kind == "csk":
            code_holes += face + pat + f".cskHole({f(d)},{f(d + 1.2 * t)},90)"
        else:
            code_holes += face + pat + (f".hole({f(d)})" if thru else f".hole({f(d)},{f(1.5 * t)})")
        nholes += 1
    s += code_holes
    for e in ends:                                                   # notches at the free ends
        if not end_notch[e]:
            continue
        wn, dn, round_notch = end_notch[e]
        p, dvec = g["pts"][e], g["dirs"][e]
        u_end = 0.0 if e == 0 else g["lens"][e]
        sgn = 1 if e == 0 else -1
        side = rng.choice([1, -1])
        n = (g["nls"][e][0] * side, g["nls"][e][1] * side)
        xdir = (-n[1], n[0])
        endp = (p[0] + dvec[0] * u_end, p[1] + dvec[1] * u_end)
        u_f = u_end + sgn * dn * 0.5
        fp = (p[0] + dvec[0] * u_f + n[0] * t / 2, p[1] + dvec[1] * u_f + n[1] * t / 2)
        zp = W / 2 - 0.5 * (W / 2 - wn / 2)
        lx = endp[0] * xdir[0] + endp[1] * xdir[1]
        thru = _clear(g, e, endp, n, t / 2 + wn / 2 + 1) and _clear(g, e, endp, (-n[0], -n[1]), t / 2 + wn / 2 + 1)
        cut = ".cutThruAll()" if thru else f".cutBlind({f(-1.5 * t)})"
        face = f".faces(near({_p3(fp, zp)})).workplane(origin=(0,0,0))"
        if round_notch:
            s += face + f".pushPoints([({f(lx)},0)]).slot2D({f(2 * dn + wn)},{f(wn)},0){cut}"
        else:
            s += face + f".pushPoints([({f(lx)},0)]).rect({f(2 * dn)},{f(wn)}){cut}"
    pts_r, rr = [], []
    for e in ends:                                                   # corner rounds at the free ends
        if end_round[e] >= 1:
            c = g["pts"][0] if e == 0 else g["pts"][-1]
            pts_r.append([_p3(c, -W / 2), _p3(c, W / 2)])
            rr.append(end_round[e])
    if pts_r:
        if len(pts_r) == 2 and abs(rr[0] - rr[1]) < 0.05 * max(rr):
            s += f".edges(near({','.join(pts_r[0] + pts_r[1])})).fillet({f(min(rr))})"
        else:
            for pr, r in zip(pts_r, rr):
                s += f".edges(near({','.join(pr)})).fillet({f(r)})"
    return "import cadquery as cq\nr=" + s + orient(rng)
