"""Pulleys, V-belt sheaves, flat-belt / timing pulleys, wheels and knobs: a revolved (r, z) section (hub + web + rim,
with V-grooves, a flat, flanged or crowned face, or a rounded tyre), then a bore with optional keyway or set-screw
hole, fillets at the web roots, chamfers, and lightening holes / slots in the web (or grip flutes on a knob).

Axis = Z: the section is drawn in the XZ plane and revolved about its local y axis (= global Z). Diameter 100-200.
"""
import math

from datagen.families.common import f, on_circle, orient


def _q(x):
    return round(float(x), 1)


def _path_code(items):
    """items: list of ('L', (r, z)) or ('A', (rm, zm), (r, z)); the first must be 'L'. Emits a closed wire."""
    pts = []
    # round and drop repeated points
    norm = []
    for it in items:
        if it[0] == "L":
            p = (_q(it[1][0]), _q(it[1][1]))
            if norm and norm[-1][0] == "L" and norm[-1][1] == p:
                continue
            if norm and norm[-1][0] == "A" and norm[-1][2] == p:
                continue
            norm.append(("L", p))
        else:
            norm.append(("A", (_q(it[1][0]), _q(it[1][1])), (_q(it[2][0]), _q(it[2][1]))))
    if norm[-1][0] == "L" and norm[-1][1] == norm[0][1]:
        norm.pop()
    # drop collinear line points (only between two line points)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(norm)):
            if norm[i][0] != "L" or norm[i - 1][0] != "L":
                continue
            nxt = norm[(i + 1) % len(norm)]
            if nxt[0] != "L":
                continue
            a, b, c = norm[i - 1][1], norm[i][1], nxt[1]
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-9:
                norm.pop(i)
                changed = True
                break
    code = "cq.Workplane('XZ')"
    run, first = [], True
    for it in norm:
        if it[0] == "L":
            run.append(it[1])
        else:
            if run:
                code += f".polyline([{','.join(f'({f(a)},{f(b)})' for a, b in run)}]{'' if first else ',includeCurrent=True'})"
                first = False
                run = []
            code += f".threePointArc(({f(it[1][0])},{f(it[1][1])}),({f(it[2][0])},{f(it[2][1])}))"
    if run:
        code += f".polyline([{','.join(f'({f(a)},{f(b)})' for a, b in run)}]{'' if first else ',includeCurrent=True'})"
    return code + ".close()"


def generate(rng):
    D = _q(rng.uniform(100, 200))
    R = D / 2
    kind = rng.choices(["vbelt", "flat", "wheel", "knob", "timing"], [0.38, 0.18, 0.16, 0.13, 0.15])[0]
    gdepth = 0
    if kind == "vbelt":
        ng = rng.choice([1, 1, 2, 2, 3, 4])
        pitch = R * rng.uniform(0.13, 0.2) * (1 if ng < 3 else 0.8)
        gdepth = pitch * rng.uniform(0.7, 0.95)
        W = ng * pitch + pitch * rng.uniform(0.3, 0.7)
    elif kind == "flat":
        W = D * rng.uniform(0.2, 0.5)
    elif kind == "wheel":
        W = D * rng.uniform(0.18, 0.35)
    elif kind == "knob":
        W = D * rng.uniform(0.25, 0.45)
    else:
        W = D * rng.uniform(0.15, 0.35)
    W = _q(W)
    hub_r = _q(R * rng.uniform(0.24, 0.4))
    bore_r = _q(hub_r * rng.uniform(0.35, 0.6))
    hub_w = _q(W * rng.uniform(1.0, 1.6)) if kind != "knob" else W
    side = rng.choice([0, 0, 0.5, 1, 1])                       # where the extra hub length goes
    hz0, hz1 = -W / 2 - (hub_w - W) * side, -W / 2 - (hub_w - W) * side + hub_w
    rim_t = max(R * 0.08, gdepth + R * rng.uniform(0.05, 0.09))
    ri = R - rim_t
    web_t = W * rng.uniform(0.25, 0.45)
    zw = rng.choice([0, 0, 0, 1, -1]) * (W - web_t) / 2 * 0.5
    solid_web = kind == "knob" or rng.random() < 0.25

    # outer surface items from (., -W/2) up to (., W/2)
    if kind == "vbelt":
        half = math.radians(rng.choice([17, 18, 19, 20]))
        top_w = pitch * rng.uniform(0.8, 0.9)
        bot_w = max(top_w - 2 * gdepth * math.tan(half), pitch * 0.15)
        land = (W - ng * pitch) / 2
        outer = [("L", (R, -W / 2))]
        for k in range(ng):
            zc = -W / 2 + land + pitch * (k + 0.5)
            outer += [("L", (R, zc - top_w / 2)), ("L", (R - gdepth, zc - bot_w / 2)),
                      ("L", (R - gdepth, zc + bot_w / 2)), ("L", (R, zc + top_w / 2))]
        outer.append(("L", (R, W / 2)))
    elif kind in ("flat", "timing"):
        u = rng.random()
        if u < 0.45:                                            # flanged
            fh = R * rng.uniform(0.04, 0.08)
            ft = W * rng.uniform(0.07, 0.12)
            outer = [("L", (R, -W / 2)), ("L", (R, -W / 2 + ft)), ("L", (R - fh, -W / 2 + ft)),
                     ("L", (R - fh, W / 2 - ft)), ("L", (R, W / 2 - ft)), ("L", (R, W / 2))]
            ri = min(ri, R - fh - R * 0.06)
        elif u < 0.7 and kind == "flat":                        # crowned
            cr = R * rng.uniform(0.02, 0.04)
            outer = [("L", (R - cr, -W / 2)), ("A", (R, 0), (R - cr, W / 2))]
            ri = min(ri, R - cr - R * 0.06)
        else:
            outer = [("L", (R, -W / 2)), ("L", (R, W / 2))]
    elif kind == "wheel":
        tyre = R * rng.uniform(0.1, 0.2)
        outer = [("L", (R - tyre, -W / 2)), ("A", (R, 0), (R - tyre, W / 2))]
        ri = min(ri, R - tyre - R * 0.06)
    else:  # knob
        h1 = W * rng.uniform(0.35, 0.6)
        rt = R * rng.uniform(0.5, 0.75)
        outer = [("L", (R, -W / 2)), ("L", (R, -W / 2 + h1))]
        if rng.random() < 0.6:
            mid_a = math.radians(45)
            outer.append(("A", (rt + (R - rt) * math.cos(mid_a), -W / 2 + h1 + (W - h1) * math.sin(mid_a)),
                          (rt, W / 2)))
        else:
            outer.append(("L", (rt, W / 2)))

    if kind == "knob":
        zb = -W / 2 + W * rng.uniform(0.4, 0.65)
        items = [("L", (bore_r, -W / 2))] + outer + [("L", (0, W / 2)), ("L", (0, zb)), ("L", (bore_r, zb))]
        hz0, hz1 = -W / 2, zb
    else:
        items = [("L", (bore_r, hz0)), ("L", (hub_r, hz0))]
        if solid_web:
            if hz0 < -W / 2:
                items.append(("L", (hub_r, -W / 2)))
            items += outer
            if hz1 > W / 2:
                items.append(("L", (hub_r, W / 2)))
            items += [("L", (hub_r, hz1))]
        else:
            items += [("L", (hub_r, zw - web_t / 2)), ("L", (ri, zw - web_t / 2)), ("L", (ri, -W / 2))]
            items += outer
            items += [("L", (ri, W / 2)), ("L", (ri, zw + web_t / 2)), ("L", (hub_r, zw + web_t / 2)),
                      ("L", (hub_r, hz1))]
        items.append(("L", (bore_r, hz1)))
    # centre along z
    zlo, zhi = min(hz0, -W / 2), max(hz1 if kind != "knob" else W / 2, W / 2)
    sh = (zlo + zhi) / 2
    items = [(it[0],) + tuple((p[0], p[1] - sh) for p in it[1:]) for it in items]
    W2 = W / 2
    zb0, zb1, zt0, zt1 = _q(-W2 - sh), _q(W2 - sh), _q(hz0 - sh), _q(hz1 - sh)
    zw_s = zw - sh
    hub_r, bore_r, ri = _q(hub_r), _q(bore_r), _q(ri)
    web_lo, web_hi = _q(zw_s - web_t / 2), _q(zw_s + web_t / 2)
    if not solid_web and kind != "knob":
        items = [(it[0],) + tuple((p[0], web_lo if abs(p[1] - (zw_s - web_t / 2)) < 1e-6 else
                                   (web_hi if abs(p[1] - (zw_s + web_t / 2)) < 1e-6 else p[1])) for p in it[1:])
                 for it in items]
    s = _path_code(items) + ".revolve(360,(0,0,0),(0,1,0))"

    # fillets where the web meets hub and rim
    if not solid_web and kind != "knob" and rng.random() < 0.6:
        fr = _q(min(web_t, ri - hub_r, rim_t, (W - web_t) / 2 - abs(zw), hub_w - web_t) * rng.uniform(0.2, 0.35))
        if fr >= 0.5:
            ps = [on_circle(hub_r, web_lo), on_circle(hub_r, web_hi), on_circle(ri, web_lo), on_circle(ri, web_hi)]
            s += f".edges(near({','.join(ps)})).fillet({f(fr)})"
    # chamfers on the bore ends / knob base
    if kind != "knob" and rng.random() < 0.5:
        c = _q(max(0.5, min(bore_r, hub_r - bore_r) * rng.uniform(0.08, 0.15)))
        s += f".edges(near({on_circle(bore_r, zt0)},{on_circle(bore_r, zt1)})).chamfer({f(c)})"
    if kind == "knob" and rng.random() < 0.6:
        c = _q(W * rng.uniform(0.04, 0.08))
        s += f".edges(near({on_circle(R, zb0)})).chamfer({f(c)})"
    if kind in ("flat", "timing") and solid_web and rng.random() < 0.4:
        c = _q(W * rng.uniform(0.03, 0.06))
        s += f".edges(near({on_circle(R, zb0)},{on_circle(R, zb1)})).chamfer({f(c)})"

    # keyway in the bore, or radial set-screw hole
    if kind != "knob" and rng.random() < 0.55:
        kw = _q(bore_r * rng.uniform(0.45, 0.6))
        kd = _q(bore_r * rng.uniform(0.2, 0.3))
        s += (f".cut(cq.Workplane('XY').box({f(2 * kd)},{f(kw)},{f(zt1 - zt0 + 2)})"
              f".translate(({f(bore_r)},0,{f((zt0 + zt1) / 2)})))")
    elif rng.random() < 0.6 and hub_r - bore_r > 3:
        hd = _q(max(2.0, min(bore_r * rng.uniform(0.3, 0.5), (hub_r - bore_r) * 0.8)))
        if kind == "knob":
            zc, ln = _q(zb0 + (zt1 - zb0) * 0.45), R + 1
            hd = min(hd, _q((zt1 - zb0) * 0.4))
        elif solid_web:
            zc, ln = (_q((zt0 + zt1) / 2), R + 1) if kind in ("flat", "timing") else (None, 0)
            if zt1 > zb1 + hd + 3:                              # put it in the protruding hub when there is one
                zc, ln = _q((zb1 + zt1) / 2), hub_r + 1
        else:
            if zt1 - web_hi > hd + 3:
                zc, ln = _q((web_hi + zt1) / 2), hub_r + 1
            elif web_lo - zt0 > hd + 3:
                zc, ln = _q((zt0 + web_lo) / 2), hub_r + 1
            else:
                zc, ln = None, 0
        if zc is not None:
            s += f".cut(cq.Workplane('XZ').center(0,{f(zc)}).circle({f(hd / 2)}).extrude({f(-ln)}))"

    # lightening holes / slots in the web, or grip flutes on a knob
    span = ri - hub_r
    if not solid_web and kind != "knob" and span > R * 0.2 and rng.random() < 0.65:
        n = rng.choice([3, 4, 5, 6, 6, 8])
        rc = _q((ri + hub_r) / 2)
        top = "'>Z'" if zt1 >= zb1 else "'<Z'"
        wp = f".faces({top}).workplane(centerOption='ProjectedOrigin')"
        if rng.random() < 0.6:
            hd = _q(min(span * rng.uniform(0.45, 0.65), 2 * math.pi * rc / n * 0.55))
            s += wp + f".polarArray({f(rc)},0,360,{n}).hole({f(hd)})"
        else:
            w_ = _q(span * rng.uniform(0.3, 0.45))
            ln = _q(max(min(2 * math.pi * rc / n * 0.6, span * 1.5), w_ * 1.3))
            if ln > 2 * math.pi * rc / n * 0.7:
                n = max(3, int(2 * math.pi * rc / ln * 0.65))
            s += wp + f".polarArray({f(rc)},0,360,{n}).slot2D({f(ln)},{f(w_)},90).cutThruAll()"
    elif kind == "knob" and rng.random() < 0.65:
        n = rng.choice([6, 8, 10, 12, 16])
        fr_ = _q(min(R * rng.uniform(0.05, 0.09), math.pi * R / n * 0.6))
        s += (f".faces('<Z').workplane(centerOption='ProjectedOrigin').polarArray({f(R + fr_ * 0.3)},0,360,{n})"
              f".circle({f(fr_)}).cutBlind({f(-h1 * rng.uniform(0.75, 1.0))})")
    return "import cadquery as cq\nr=" + s + orient(rng)
