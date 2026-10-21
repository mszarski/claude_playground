"""Links, levers and connecting bars: a flat bar with rounded ends (stadium via slot2D, tapered outline of lines and
tangent arcs with unequal ends, or a dog-bone: narrow shank between round ends with filleted junctions), a hole at
each end (often different sizes), optional end bosses (one or both sides), an optional middle pivot, lightening
slots / holes / blind pockets in the shank, and filleted or chamfered outline edges. Also perforated straps.
"""
import math

from datagen.families.common import f, on_circle, orient


def _pts(pts):
    return "[" + ",".join(f"({f(x)},{f(y)})" for x, y in pts) + "]"


def generate(rng):
    Lt = rng.uniform(100, 200)                                    # overall length
    style = rng.choices(["stadium", "taper", "dogbone", "strap"], [0.3, 0.3, 0.25, 0.15])[0]
    R1 = rng.uniform(0.07, 0.17) * Lt
    R2 = R1 * rng.uniform(0.5, 0.8) if style == "taper" else R1
    if style == "strap":
        R1 = R2 = rng.uniform(0.05, 0.1) * Lt
    x1, x2 = -Lt / 2 + R1, Lt / 2 - R2                            # end centres
    c = x2 - x1
    t = max(rng.uniform(0.3, 0.8) * min(R1, R2), 4) if style != "strap" else max(rng.uniform(0.15, 0.35) * R1, 3)
    wsh = 2 * min(R1, R2) * rng.uniform(0.45, 0.7)                # dog-bone shank width

    boss = style != "strap" and rng.random() < 0.45
    two_sided = boss and rng.random() < 0.4
    rb1 = R1 * rng.uniform(0.65, 0.85)
    rb2 = rb1 if style != "taper" else R2 * rng.uniform(0.65, 0.85)
    hb = rng.uniform(0.3, 0.9) * t if boss else 0.0
    mid = style in ("stadium", "dogbone") and rng.random() < 0.25    # middle pivot (three-hole lever)
    rbm = min(R1 * rng.uniform(0.75, 1.0), 0.32 * c) if style == "dogbone" else min(R1, 0.32 * c)
    z0 = -t / 2 if (two_sided or not boss) else -(t + hb) / 2
    zt = z0 + t

    s = f"cq.Workplane('XY').workplane(offset={f(z0)})"
    if style in ("stadium", "strap"):
        s += f".slot2D({f(Lt)},{f(2 * R1)}).extrude({f(t)})"
    elif style == "taper":
        sn = (R1 - R2) / c
        cs = math.sqrt(1 - sn * sn)
        p1u, p2u = (x1 + R1 * sn, R1 * cs), (x2 + R2 * sn, R2 * cs)
        p1l, p2l = (p1u[0], -p1u[1]), (p2u[0], -p2u[1])
        P = lambda p: f"({f(p[0])},{f(p[1])})"  # noqa: E731
        s += (f".moveTo{P(p1u)}.lineTo{P(p2u)}.threePointArc(({f(x2 + R2)},0),{P(p2l)})"
              f".lineTo{P(p1l)}.threePointArc(({f(x1 - R1)},0),{P(p1u)}).close().extrude({f(t)})")
    else:
        s += f".sketch().rect({f(c)},{f(wsh)}).push([({f(x1)},0),({f(x2)},0)]).circle({f(R1)})"
        if mid:
            s += f".push([(0,0)]).circle({f(rbm)})"
        s += f".finalize().extrude({f(t)})"
        # fillet the concave junctions between the shank and the round ends
        jx = math.sqrt(R1 * R1 - wsh * wsh / 4)
        pts = [(x1 + jx, y) for y in (wsh / 2, -wsh / 2)] + [(x2 - jx, y) for y in (wsh / 2, -wsh / 2)]
        if mid:
            jm = math.sqrt(rbm * rbm - wsh * wsh / 4)
            pts += [(a * jm, y) for a in (-1, 1) for y in (wsh / 2, -wsh / 2)]
        rj = rng.uniform(0.15, 0.4) * min(R1, (c - 2 * jx) / 2 if not mid else (c / 2 - jx - jm) / 2)
        s += f".edges(near({','.join(f'({f(x)},{f(y)},{f(z0 + t / 2)})' for x, y in pts)})).fillet({f(rj)})"

    # outline edge treatment (before bosses and holes, so only the outline is touched)
    rf = 0.0
    k = rng.random()
    if k < 0.35 and style != "dogbone":                          # (fillets on the fused dog-bone outline are slow)
        rf = rng.uniform(0.15, 0.35) * t
        s += f".edges('not |Z').fillet({f(rf)})" if not boss or two_sided else f".edges('>Z').fillet({f(rf)})"
    elif k < 0.6 or (k < 0.7 and style == "dogbone"):
        rf = min(rng.uniform(0.1, 0.2) * t, 2)
        s += f".edges('not |Z').chamfer({f(rf)})" if rng.random() < 0.5 else f".edges('>Z').chamfer({f(rf)})"

    # bosses at the ends
    if boss:
        same = abs(rb1 - rb2) < 0.5
        for zf in ([zt, z0] if two_sided else [zt]):
            face = "'>Z'" if zf == zt else "'<Z'"
            if same:
                s += f".faces({face}).workplane().pushPoints([({f(x1)},0),({f(x2)},0)]).circle({f(rb1)}).extrude({f(hb)})"
            else:                                                # second boss starts from the bar face, not boss 1
                s += f".faces({face}).workplane().pushPoints([({f(x1)},0)]).circle({f(rb1)}).extrude({f(hb)})"
                s += f".faces(near(({f(x2)},0,{f(zf)}))).workplane().pushPoints([({f(x2)},0)]).circle({f(rb2)}).extrude({f(hb)})"
        if rng.random() < 0.35 and not two_sided:               # fillet the boss roots
            rr = min(0.3 * hb, 0.35 * (min(R1 - rb1, R2 - rb2) - rf))
            if rr > 0.6:
                s += f".edges(near({on_circle(rb1, zt, x1)},{on_circle(rb2, zt, x2)})).fillet({f(rr)})"
        ra1, ra2 = rb1, rb2
    else:
        ra1, ra2 = R1, R2

    # holes at the ends (and the pivot)
    d1 = 2 * ra1 * rng.uniform(0.35, 0.6)
    d2 = d1 if (abs(R1 - R2) < 0.5 and rng.random() < 0.6) else 2 * ra2 * rng.uniform(0.35, 0.6)
    top = ".faces('>Z').workplane()"
    if style == "strap":
        n = rng.choice([3, 4, 5, 6, 7])
        p = c / (n - 1)
        d1 = min(d1, p - max(0.25 * p, 4))
        if rng.random() < 0.6:
            s += top + f".rarray({f(p)},1,{n},1).hole({f(d1)})"
        else:
            s += top + f".rarray({f(p)},1,{n},1).cskHole({f(d1)},{f(min(d1 * 1.8, d1 + 1.6 * t))},90)"
        return "import cadquery as cq\nr=" + s + orient(rng)
    if abs(d1 - d2) < 0.05:
        s += top + f".pushPoints([({f(x1)},0),({f(x2)},0)]).hole({f(d1)})"
    else:
        s += top + f".pushPoints([({f(x1)},0)]).hole({f(d1)})"
        s += top + f".pushPoints([({f(x2)},0)]).hole({f(d2)})"
    if mid:
        dm = 2 * rbm * rng.uniform(0.3, 0.5) if style == "dogbone" else min(d1, d2) * rng.uniform(0.6, 1.0)
        s += top + f".hole({f(dm)})"

    # lightening in the shank
    lo = x1 + max(ra1 if boss else d1 / 2 + 0.25 * R1, d1 / 2 + 0.3 * R1) + 0.06 * c
    hi = x2 - max(ra2 if boss else d2 / 2 + 0.25 * R2, d2 / 2 + 0.3 * R2) - 0.06 * c
    if style == "taper":
        hw = R2 - 0.3 * R2                                       # narrowest half-width minus a rim
    elif style == "dogbone":
        hw = wsh / 2 - max(0.25 * wsh / 2, 3)
    else:
        hw = R1 - max(0.3 * R1, 3)
    if style == "dogbone" and rng.random() < 0.5:
        hw = 0                                                   # plain shank is common on dog-bones
    span = hi - lo
    lk = rng.random()
    if hw > 2.5 and span > 15 and not mid:
        sw = 2 * hw * rng.uniform(0.6, 1.0)
        if lk < 0.4 and span > 1.6 * sw:                         # through lightening slot
            s += top + f".pushPoints([({f((lo + hi) / 2)},0)]).slot2D({f(span)},{f(sw)}).cutThruAll()"
        elif lk < 0.65:                                          # row of lightening holes
            n = max(1, min(int(span / (sw * 1.4)), 4))
            hd = min(sw, span / n * 0.75)
            if n == 1:
                s += top + f".pushPoints([({f((lo + hi) / 2)},0)]).hole({f(hd)})"
            else:
                pitch = (span - hd) / (n - 1)
                xs = [(lo + hi) / 2 + (i - (n - 1) / 2) * pitch for i in range(n)]
                s += top + f".pushPoints({_pts([(x, 0) for x in xs])}).hole({f(hd)})"
        elif lk < 0.85 and span > 1.6 * sw and t >= 6:            # blind pocket(s): I-beam web
            dp = t * rng.uniform(0.2, 0.35)
            xm = (lo + hi) / 2
            s += f".faces(near(({f(xm)},0,{f(zt)}))).workplane()"
            s += f".pushPoints([({f((lo + hi) / 2)},0)]).slot2D({f(span)},{f(sw)}).cutBlind({f(-dp)})"
            if not boss or two_sided:
                s += f".faces(near(({f(xm)},0,{f(z0)}))).workplane()"
                s += f".pushPoints([({f((lo + hi) / 2)},0)]).slot2D({f(span)},{f(sw)}).cutBlind({f(-dp)})"
    return "import cadquery as cq\nr=" + s + orient(rng)
