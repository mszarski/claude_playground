"""Spur gears, sprockets, racks and gear shafts, modelled at canonical scale.

Tooth profiles are drawn with the Sketch API: a root circle fused with a polar array (parray) of trapezoids whose
flanks follow a ~20-degree pressure angle (an approximation of an involute), optionally with 2D vertex fillets;
sprockets subtract roller seats between teeth. Gears get hubs, recessed webs, lightening holes, bores with keyways
and set-screw holes; racks get a toothed bar with counterbored mounting holes; gear shafts a pinion on a stepped
shaft with a keyway.
"""
import math

from datagen.families.common import f, on_circle, orient


def rd(x):
    return round(x, 1)


def _tooth_sketch(n, m, rng, fillet=True):
    """Sketch code for an n-tooth spur outline of module m: root circle + parray of trapezoid teeth."""
    Rp = m * n / 2
    Rr = Rp - 1.25 * m
    Rt = Rp + m
    e = 0.3 * m                                            # teeth start inside the root circle
    root_space = 2 * math.pi * Rr / n
    wr = min(1.5708 * m + 0.728 * 1.25 * m, 0.62 * root_space)   # tooth width at the root circle
    wt = min(0.84 * m, 0.5 * wr) * rng.uniform(0.85, 1.1)          # width at the tip
    hh = Rt - Rr
    a1 = math.degrees(math.atan2(hh, (wr - wt) / 2))
    wb = wr + 2 * e / math.tan(math.radians(a1))
    h = hh + e
    rc = Rr - e + h / 2
    sk = (f"cq.Sketch().circle({f(Rr)}).parray({f(rc)},0,360,{n})"
          f".trapezoid({f(wb)},{f(h)},{f(a1)},angle=-90)")
    if fillet:
        sk += f".clean().reset().vertices().fillet({f(m * 0.2)})"
    else:
        sk += ".clean()"
    return sk, Rr, Rt


def _bore(s, bore, rng):
    """Append a bore cut through the hub from the top face, with an optional keyway."""
    s += f".faces('>Z').workplane().hole({f(bore)})"
    if rng.random() < 0.55:
        kw = bore * 0.25
        s += (f".faces('>Z').workplane().pushPoints([(0,{f(bore / 2)})]).rect({f(kw)},{f(kw)})"
              f".cutThruAll()")
    return s


def _spur(rng, sprocket=False):
    OD = rng.uniform(100, 200)
    n = rng.randint(9, 40) if sprocket else rng.randint(10, 40)
    if sprocket:
        Rp = OD / 2 / (1 + 0.6 * math.sin(math.pi / n))      # tips ~0.3 pitch above the pitch circle
        p = 2 * Rp * math.sin(math.pi / n)                     # chain pitch
        dr = 0.62 * p
        Rr = Rp - dr / 2
        Rt = OD / 2
        hh = Rt - Rr
        tip_w = p * rng.uniform(0.15, 0.25)
        base_w = p - dr * 0.75
        e = 0.2 * dr
        a1 = math.degrees(math.atan2(hh, max((base_w - tip_w) / 2, 0.05 * p)))
        wb = base_w + 2 * e / math.tan(math.radians(a1))
        h = hh + e
        rc = Rr - e + h / 2
        sk = (f"cq.Sketch().circle({f(Rr + 0.05 * dr)}).parray({f(rc)},0,360,{n})"
              f".trapezoid({f(wb)},{f(h)},{f(a1)},angle=-90)"
              f".reset().parray({f(Rp)},{f(180 / n)},360,{n}).circle({f(dr / 2)},mode='s').clean()")
        b = rd(OD * rng.uniform(0.04, 0.09))
        m = p / math.pi
    else:
        m = OD / (n + 2)
        sk, Rr, Rt = _tooth_sketch(n, m, rng, fillet=n <= 16 and rng.random() < 0.6)
        b = rd(OD * rng.uniform(0.08, 0.3))
    web_in = max(Rr - 2.2 * m, 0.6 * Rr) if not sprocket else Rr - 0.6 * dr      # inside of the toothed rim
    hub_r = OD / 2 * rng.uniform(0.18, 0.32)
    hub_r = min(hub_r, web_in * 0.75)
    bore = hub_r * rng.uniform(0.75, 1.1)
    hub = rng.random() < (0.85 if sprocket else 0.7)
    hub_len = rd(b * rng.uniform(1.6, 3.0)) if hub else b
    if sprocket:
        hub_len = rd(max(hub_len, OD * 0.12))
    if not hub:
        bore = Rr * rng.uniform(0.35, 0.7)
    sym = rng.random() < 0.5 and not sprocket
    # z layout: gear from zg to zg + b; hub from zh to zh + hub_len
    if not hub:
        zg = rd(-b / 2)
        zh = zg
    elif sym:
        zh = rd(-hub_len / 2)
        zg = rd(-b / 2)
    else:
        zh = rd(-hub_len / 2)
        zg = zh
    s = f"cq.Workplane('XY').workplane(offset={f(zg)}).placeSketch({sk}).extrude({f(b)})"
    # lightening holes / recessed web, only when there is room between hub and rim
    span = web_in - (hub_r if hub else bore / 2 * 1.6)
    rm = web_in - span / 2
    if span > 0.18 * OD:
        k = rng.random()
        if k < 0.45:
            nl = rng.choice([4, 5, 6, 6, 8])
            dl = min(span * rng.uniform(0.5, 0.75), 2 * rm * math.sin(math.pi / nl) * 0.7)
            s += f".faces('>Z').workplane().polarArray({f(rm)},0,360,{nl}).hole({f(dl)})"
        elif k < 0.75 and not sprocket and b > 0.12 * OD:
            depth = rd(b * rng.uniform(0.2, 0.32))
            r_in = (hub_r if hub else bore / 2 * 1.6) + 0.04 * OD
            ann = f".circle({f(web_in)}).circle({f(r_in)}).cutBlind({f(-depth)})"
            s += f".faces('>Z').workplane()" + ann
            if sym or not hub:
                s += f".faces('<Z').workplane()" + ann
            if rng.random() < 0.5:
                nl = rng.choice([4, 5, 6])
                dl = min((web_in - r_in) * rng.uniform(0.45, 0.65), 2 * rm * math.sin(math.pi / nl) * 0.7)
                s += (f".faces('>Z').workplane(offset={f(-depth)}).polarArray({f((web_in + r_in) / 2)},0,360,{nl})"
                      f".hole({f(dl)})")
    lines = []
    if hub:
        hs = f"cq.Workplane('XY').workplane(offset={f(zh)}).circle({f(hub_r)}).extrude({f(hub_len)})"
        if rng.random() < 0.5:
            hs += f".faces('>Z or <Z').chamfer({f(hub_r * 0.06)})"
        s += f".union({hs})"
    s = _bore(s, bore, rng)
    if hub and not sym and rng.random() < 0.4:   # radial set screw through the hub
        zs = rd(zh + b + (hub_len - b) / 2)
        s += (f".cut(cq.Solid.makeCylinder({f(min(hub_len - b, hub_r) * 0.18)},{f(hub_r + 2)},"
              f"cq.Vector(0,0,{f(zs)}),cq.Vector(0,1,0)))")
    lines.append("r=" + s)
    return lines


def _rack(rng):
    L = rng.uniform(110, 200)
    n = rng.randint(8, 30)
    p = L / (n + rng.uniform(0.3, 1.5))
    m = p / math.pi
    hb = m * rng.uniform(3.0, 6.0)                         # bar height under the teeth
    bw = rd(m * rng.uniform(3.0, 8.0))                     # face width (extrude)
    wt, wr = 0.84 * m, 0.84 * m + 2 * 2.25 * m * math.tan(math.radians(20))
    e = 0.3 * m
    hh = 2.25 * m
    a1 = math.degrees(math.atan2(hh, (wr - wt) / 2))
    wb = wr + 2 * e / math.tan(math.radians(a1))
    H = hb + hh
    y0 = -H / 2
    yc = y0 + hb - e + (hh + e) / 2
    sk = (f"cq.Sketch().push([(0,{f(y0 + hb / 2)})]).rect({f(L)},{f(hb)})"
          f".push([(0,{f(yc)})]).rarray({f(p)},1,{n},1).trapezoid({f(wb)},{f(hh + e)},{f(a1)}).clean()")
    if rng.random() < 0.5:
        sk = sk[:-len(".clean()")] + f".clean().reset().vertices('>Y').fillet({f(m * 0.25)})"
    s = f"cq.Workplane('XY').workplane(offset={f(-bw / 2)}).placeSketch({sk}).extrude({f(bw)})"
    nh = rng.choice([2, 3, 4])
    if bw > 2.4 * m and hb > 2.5 * m:
        dh = min(bw * 0.35, hb * 0.5)
        sp = L * rng.uniform(0.6, 0.8) / (nh - 1)
        s += f".faces('<Y').workplane(centerOption='CenterOfBoundBox').rarray({f(sp)},1,{nh},1)"
        if rng.random() < 0.5:
            s += f".cboreHole({f(dh)},{f(dh * 1.6)},{f(hb * 0.35)},{f(hb * 0.9)})"
        else:
            s += f".hole({f(dh)},{f(hb * 0.8)})"
    return ["r=" + s]


def _shaft(rng):
    """Pinion gear cut on a stepped shaft."""
    L = rng.uniform(110, 200)
    n = rng.randint(10, 24)
    OD = L * rng.uniform(0.3, 0.55)
    m = OD / (n + 2)
    sk, Rr, Rt = _tooth_sketch(n, m, rng, fillet=n <= 16 and rng.random() < 0.5)
    b = rd(L * rng.uniform(0.15, 0.3))
    ds1 = min(2 * Rr * rng.uniform(0.45, 0.7), OD * 0.4)
    ds2 = ds1 * rng.uniform(0.65, 0.85)
    l_left = rd((L - b) * rng.uniform(0.3, 0.55))
    l_right = rd(L - b - l_left)
    step = rng.random() < 0.6
    ls = rd(l_right * rng.uniform(0.3, 0.5)) if step else 0.0
    z0 = rd(-L / 2)
    zg = rd(z0 + l_left)
    s = f"cq.Workplane('XY').workplane(offset={f(zg)}).placeSketch({sk}).extrude({f(b)})"
    s += f".faces('>Z').workplane().circle({f(ds1 / 2)}).extrude({f(l_right - ls)})"
    if step:   # second, smaller journal on the right end
        s += f".faces('>Z').workplane().circle({f(ds2 / 2)}).extrude({f(ls)})"
    s += f".faces('<Z').workplane().circle({f(ds1 / 2)}).extrude({f(l_left)})"
    s += f".faces('>Z or <Z').chamfer({f(ds2 * 0.08)})"
    if rng.random() < 0.6:     # keyway on the long end of the shaft
        kl = l_left * rng.uniform(0.4, 0.7)
        kw = ds1 * 0.28
        zc = z0 + l_left * 0.1 + kl / 2
        s += (f".cut(cq.Workplane('XY').box({f(kw)},{f(kw)},{f(kl)})"
              f".translate((0,{f(ds1 / 2)},{f(zc)})))")
    if rng.random() < 0.4:     # centre holes
        s += f".faces('<Z').workplane().cskHole({f(ds2 * 0.15)},{f(ds2 * 0.35)},60,{f(ds2 * 0.4)})"
    return ["r=" + s]


def _cluster(rng):
    """Compound (cluster) gear: two spur gears of different size on one hub."""
    OD = rng.uniform(110, 200)
    n1 = rng.randint(24, 40)
    m = OD / (n1 + 2)
    n2 = rng.randint(8, max(9, int(n1 * 0.6)))
    sk1, Rr1, _ = _tooth_sketch(n1, m, rng, fillet=False)
    m2 = m * rng.uniform(0.9, 1.3)
    sk2, Rr2, Rt2 = _tooth_sketch(n2, m2, rng, fillet=n2 <= 14)
    if Rt2 > 0.8 * OD / 2:
        m2 = 0.8 * OD / 2 / (n2 / 2 + 1)
        sk2, Rr2, Rt2 = _tooth_sketch(n2, m2, rng, fillet=False)
    b1 = rd(OD * rng.uniform(0.08, 0.18))
    b2 = rd(OD * rng.uniform(0.12, 0.3))
    H = b1 + b2
    z0 = rd(-H / 2)
    lines = [f"g=cq.Workplane('XY').workplane(offset={f(z0)}).placeSketch({sk1}).extrude({f(b1)})"]
    s = f"g.faces('>Z').workplane().placeSketch({sk2}).extrude({f(b2)})"
    bore = min(Rr2 * rng.uniform(0.6, 1.0), Rr2 - 1.5 * m2)
    s = _bore(s, bore, rng)
    if rng.random() < 0.5:
        span = Rr1 - 2 * m - Rr2
        if span > 0.1 * OD:
            nl = rng.choice([4, 5, 6])
            rm = Rr2 + span / 2
            dl = min(span * 0.6, 2 * rm * math.sin(math.pi / nl) * 0.7)
            s += f".faces('<Z').workplane().polarArray({f(rm)},0,360,{nl}).hole({f(dl)})"
    lines.append("r=" + s)
    return lines


def generate(rng):
    k = rng.random()
    if k < 0.45:
        lines = _spur(rng)
    elif k < 0.65:
        lines = _spur(rng, sprocket=True)
    elif k < 0.75:
        lines = _rack(rng)
    elif k < 0.88:
        lines = _shaft(rng)
    else:
        lines = _cluster(rng)
    lines[-1] += orient(rng)
    return "import cadquery as cq\n" + "\n".join(lines)
