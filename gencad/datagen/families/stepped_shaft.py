"""Stepped shafts and pins: a revolved (r, z) profile with several diameters, chamfered ends, snap-ring grooves and
undercuts, then optional keyways, cross holes, wrench flats and centre / tapped holes in the ends.

The shaft axis is Z (the profile is drawn in the XZ plane and revolved about its local y axis = global Z); the part is
centred at the origin and its length (the longest side) is 100-200.
"""
from datagen.families.common import f, on_circle, orient


def _q(x):
    return round(float(x), 1)


def _poly(pts):
    """Clean an (r, z) point list: round, drop repeated and collinear points; format as a polyline literal."""
    pts = [(_q(a), _q(b)) for a, b in pts]
    out = []
    for p in pts:
        if out and p == out[-1]:
            continue
        out.append(p)
    if out[0] == out[-1]:
        out.pop()
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-9:
                out.pop(i)
                changed = True
                break
    return "[" + ",".join(f"({f(a)},{f(b)})" for a, b in out) + "]"


def generate(rng):
    L = _q(rng.uniform(100, 200))
    kind = rng.random()
    n = rng.choice([2, 3, 3, 4, 4, 5, 5, 6, 7])
    dmax = L * rng.uniform(0.08, 0.3)
    # diameters along the shaft: a "central collar" shape (rise then fall), monotonic, or pin-like
    if kind < 0.45:                                      # bearing shaft: rises to a collar then falls
        peak = rng.randrange(1, n - 1) if n > 2 else 1
        ds = []
        for i in range(n):
            k = abs(i - peak)
            ds.append(dmax * (1 - 0.18 * k * rng.uniform(0.6, 1.2)))
    elif kind < 0.8:                                     # monotonic stepped shaft
        ds = sorted([dmax * rng.uniform(0.45, 1.0) for _ in range(n)], reverse=True)
        ds[0] = dmax
    else:                                                # pin / bolt-like: long body, short head or neck
        n = rng.choice([2, 3])
        ds = [dmax * rng.uniform(1.25, 1.6)] + [dmax * rng.uniform(0.7, 1.0) for _ in range(n - 1)]
        ds[1] = dmax * 0.85
    ds = [max(_q(d), 6.0) for d in ds]
    for i in range(1, n):                               # distinct neighbouring steps
        if abs(ds[i] - ds[i - 1]) < 0.06 * max(ds[i], ds[i - 1]):
            ds[i] = _q(ds[i - 1] * (0.85 if ds[i] <= ds[i - 1] else 1.15))
    # step lengths
    if kind >= 0.8:
        w = [rng.uniform(0.08, 0.18)] + [rng.uniform(0.6, 1.0) for _ in range(n - 1)]
    else:
        w = [rng.uniform(0.5, 1.5) for _ in range(n)]
    tw = sum(w)
    ls = [L * x / tw for x in w]
    z = [-L / 2]
    for l in ls:
        z.append(z[-1] + l)
    z = [_q(v) for v in z]
    z[-1] = _q(L / 2)
    z[0] = -z[-1]
    rs = [d / 2 for d in ds]

    # end chamfers (in the profile) or later chamfer op
    cb = _q(min(rs[0], ls[0]) * rng.uniform(0.08, 0.2)) if rng.random() < 0.7 else 0
    ct = _q(min(rs[-1], ls[-1]) * rng.uniform(0.08, 0.2)) if rng.random() < 0.7 else 0
    chamfer_op_ends = rng.random() < 0.5                 # use .chamfer() instead of profile points
    cb = max(cb, 0.5) if cb else 0
    ct = max(ct, 0.5) if ct else 0

    # grooves: on step i near its end, away from shoulders
    grooves = {}
    for i in range(n):
        if ls[i] > 0.25 * rs[i] + 6 and rs[i] > 4 and rng.random() < 0.35:
            gw = _q(max(1.0, rs[i] * rng.uniform(0.08, 0.16)))
            gd = _q(max(0.6, rs[i] * rng.uniform(0.06, 0.12)))
            if ls[i] < 4 * gw + 4:
                continue
            side = rng.choice([0, 1])
            off = max(gw * rng.uniform(1.0, 2.5), (cb if i == 0 else 0) + (ct if i == n - 1 else 0) + gw)
            if side == 0:
                g0 = z[i] + off
            else:
                g0 = z[i + 1] - off - gw
            g0 = _q(g0)
            if g0 - z[i] < gw * 0.8 or z[i + 1] - (g0 + gw) < gw * 0.8:
                continue
            grooves[i] = (g0, _q(g0 + gw), _q(rs[i] - gd))

    # profile: up the axis at r=0
    pts = [(0, z[0])]
    if cb and not chamfer_op_ends:
        pts += [(rs[0] - cb, z[0]), (rs[0], z[0] + cb)]
    else:
        pts += [(rs[0], z[0])]
    for i in range(n):
        if i in grooves:
            g0, g1, gr = grooves[i]
            pts += [(rs[i], g0), (gr, g0), (gr, g1), (rs[i], g1)]
        if i < n - 1:
            pts += [(rs[i], z[i + 1]), (rs[i + 1], z[i + 1])]
    if ct and not chamfer_op_ends:
        pts += [(rs[-1], z[-1] - ct), (rs[-1] - ct, z[-1])]
    else:
        pts += [(rs[-1], z[-1])]
    pts += [(0, z[-1])]
    s = f"cq.Workplane('XZ').polyline({_poly(pts)}).close().revolve(360,(0,0,0),(0,1,0))"

    if chamfer_op_ends:
        ends = []
        if cb:
            ends.append((rs[0], z[0], cb))
        if ct:
            ends.append((rs[-1], z[-1], ct))
        if len(ends) == 2 and ends[0][2] == ends[1][2]:
            s += f".edges(near({on_circle(ends[0][0], ends[0][1])},{on_circle(ends[1][0], ends[1][1])})).chamfer({f(cb)})"
        else:
            for r_, z_, c_ in ends:
                s += f".edges(near({on_circle(r_, z_)})).chamfer({f(c_)})"

    # fillet at one or two shoulders (step-down corners)
    if rng.random() < 0.4:
        cand = [i for i in range(n - 1) if i not in grooves and i + 1 not in grooves]
        rng.shuffle(cand)
        for i in cand[:rng.choice([1, 2])]:
            small = min(rs[i], rs[i + 1])
            big = max(rs[i], rs[i + 1])
            fr = _q(min(big - small, ls[i], ls[i + 1]) * rng.uniform(0.2, 0.4))
            if fr >= 0.5:
                s += f".edges(near({on_circle(small, z[i + 1])})).fillet({f(fr)})"

    tail = []
    # keyway on a step: slot cut into the cylinder from +X
    kcand = [i for i in range(n) if i not in grooves and ls[i] > 2.2 * rs[i] and rs[i] >= 5]
    if kcand and rng.random() < 0.55:
        i = rng.choice(kcand)
        kw = _q(max(2.0, rs[i] * rng.uniform(0.45, 0.6)))
        kd = _q(max(1.0, rs[i] * rng.uniform(0.2, 0.3)))
        margin = max(kw * 0.5 + 1.5, (cb if i == 0 else 0) + (ct if i == n - 1 else 0) + kw * 0.5 + 1)
        kl = _q(min(ls[i] - 2 * margin, ls[i] * rng.uniform(0.5, 0.8)))
        if kl > kw * 1.5:
            zc = (z[i] + z[i + 1]) / 2
            if rng.random() < 0.5:   # end-mill slot (rounded ends)
                s += (f".copyWorkplane(cq.Workplane('YZ',origin=({f(rs[i])},0,0))).center(0,{f(zc)})"
                      f".slot2D({f(kl)},{f(kw)},90).cutBlind({f(-kd)})")
            else:                     # square-ended keyway
                s += (f".copyWorkplane(cq.Workplane('YZ',origin=({f(rs[i])},0,0))).center(0,{f(zc)})"
                      f".rect({f(kw)},{f(kl)}).cutBlind({f(-kd)})")
            kcand.remove(i)
    # wrench flats on a step (two parallel flats), or a single D-flat
    fcand = [i for i in kcand if i not in grooves and rs[i] >= 5]
    if fcand and rng.random() < 0.3:
        i = rng.choice(fcand)
        across = rs[i] * rng.uniform(0.7, 0.88)
        fl = _q(ls[i] * rng.uniform(0.4, 0.8))
        zc = _q((z[i] + z[i + 1]) / 2 if (0 < i < n - 1) else (z[i + 1] - fl / 2 if i == 0 else z[i] + fl / 2))
        if i == 0:
            zc = _q(z[0] + fl / 2)
        elif i == n - 1:
            zc = _q(z[-1] - fl / 2)
        t = _q(rs[i] - across + 1)
        xc = _q(across + t / 2)
        if rng.random() < 0.6:
            s += (f".cut(cq.Workplane('XY').box({f(t)},{f(2 * rs[i] + 2)},{f(fl)})"
                  f".translate(({f(xc)},0,{f(zc)})))")
            s += (f".cut(cq.Workplane('XY').box({f(t)},{f(2 * rs[i] + 2)},{f(fl)})"
                  f".translate(({f(-xc)},0,{f(zc)})))")
        else:
            s += (f".cut(cq.Workplane('XY').box({f(2 * rs[i] + 2)},{f(t)},{f(fl)})"
                  f".translate((0,{f(-xc)},{f(zc)})))")
        kcand.remove(i)
    # cross hole through a step (along Y)
    hcand = [i for i in range(n) if i not in grooves and rs[i] >= 4 and ls[i] > rs[i]]
    if hcand and rng.random() < 0.35:
        i = rng.choice(hcand)
        hd = _q(max(1.5, 2 * rs[i] * rng.uniform(0.15, 0.35)))
        lo = z[i] + hd / 2 + 1.5 + (cb if i == 0 else 0)
        hi = z[i + 1] - hd / 2 - 1.5 - (ct if i == n - 1 else 0)
        if hi > lo:
            if i in (0, n - 1):
                zc = _q(lo + (hi - lo) * 0.3) if i == 0 else _q(hi - (hi - lo) * 0.3)
            else:
                zc = _q((lo + hi) / 2)
            s += (f".cut(cq.Workplane('XZ').center(0,{f(zc)}).circle({f(hd / 2)})"
                  f".extrude({f(rs[i] + 1)},both=True))")
    # end holes: centre drill (countersunk) or tapped hole
    for end, sel, ri in ((0, "<Z", rs[0]), (1, ">Z", rs[-1])):
        u = rng.random()
        if ri < 4:
            continue
        if u < 0.25:
            cd = _q(max(1.0, ri * rng.uniform(0.15, 0.25)))
            s += f".faces('{sel}').workplane().cskHole({f(cd)},{f(cd * 2.2)},60,{f(cd * 2.5)})"
        elif u < 0.4:
            hd = _q(ri * rng.uniform(0.35, 0.6))
            s += f".faces('{sel}').workplane().hole({f(hd)},{f(min(hd * rng.uniform(1.5, 2.5), ls[0 if end == 0 else -1] * 0.8))})"
    s += "".join(tail)
    return "import cadquery as cq\nr=" + s + orient(rng)
