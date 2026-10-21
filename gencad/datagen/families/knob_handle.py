"""Control knobs, star/lobed clamping knobs, handwheels and knurled grips, modelled at canonical scale.

  * round knob: plain or skirted, optional taper, fluted/knurled side (polarArray of circle or slot cuts), fillet or
    chamfer on top, pointer line or cap recess on top, hollow underside with a central boss, round bore with a radial
    set-screw hole or a D-shaped bore;
  * star / scalloped knob: hub unioned with a polarArray of lobes (or a disc with scallops cut), concave vertical
    fillets, rounded top/bottom, bottom boss, through hole plus hex nut pocket or counterbore;
  * handwheel: rounded rim (filleted annulus), polar spokes or a web with lightening holes, hub with bore and
    keyway, optional revolving handle peg;
  * grip: long cylinder with collar, domed or chamfered end, axial flutes or ring grooves, blind bore (round, D or
    hex), optional lanyard cross hole.
"""
import math

from datagen.families.common import f, on_circle, orient


def rd(x):
    return round(x, 1)


def _dbore(rb, depth, flat_frac):
    """D-shaped bore cut from the current (bottom) workplane: circle radius rb with a flat at y = flat."""
    fl = rb * flat_frac
    a = math.sqrt(rb * rb - fl * fl)
    return (f".faces('<Z').workplane().moveTo({f(-a)},{f(fl)}).lineTo({f(a)},{f(fl)})"
            f".threePointArc((0,{f(-rb)}),({f(-a)},{f(fl)})).close().cutBlind({f(-depth)})")


def _round_knob(rng):
    Dmax = rng.uniform(100, 200)
    skirt = rng.random() < 0.45
    if skirt:
        Ds = Dmax
        D = Ds * rng.uniform(0.7, 0.85)
        ts = rd(Ds * rng.uniform(0.05, 0.1))
    else:
        D, Ds, ts = Dmax, Dmax, 0.0
    hb = rd(D * rng.uniform(0.35, 0.75))
    H = ts + hb
    z0 = rd(-H / 2)
    ztop = z0 + ts + hb
    taper = rng.uniform(3, 8) if rng.random() < 0.3 else 0.0
    top_r = D / 2 - hb * math.tan(math.radians(taper))
    if top_r < 0.3 * D:
        taper, top_r = 0.0, D / 2
    flutes = (not taper) and rng.random() < 0.7
    groove_r = 0.0
    if flutes:   # fluted outline drawn as a sketch: circle minus a polar array of circles or slots
        n = rng.choice([12, 15, 18, 20, 24, 30, 36])
        pitch = math.pi * D / n
        if rng.random() < 0.6:
            groove_r = pitch * rng.uniform(0.18, 0.3)
            tool = f".circle({f(groove_r)},mode='s')"
        else:
            groove_r = D * rng.uniform(0.02, 0.035)
            tool = f".rect({f(2 * groove_r)},{f(pitch * rng.uniform(0.25, 0.4))},mode='s')"
        body = (f".placeSketch(cq.Sketch().circle({f(D / 2)}).parray({f(D / 2)},0,360,{n}){tool})"
                f".extrude({f(hb)})")
    elif taper:
        body = f".circle({f(D / 2)}).extrude({f(hb)},taper={f(taper)})"
    else:
        body = f".circle({f(D / 2)}).extrude({f(hb)})"
    if skirt:
        s = f"cq.Workplane('XY').workplane(offset={f(z0)}).circle({f(Ds / 2)}).extrude({f(ts)})"
        if rng.random() < 0.5:
            s += f".edges(near({on_circle(Ds / 2, z0 + ts)})).chamfer({f(ts * 0.4)})"
        s += ".faces('>Z').workplane()" + body
    else:
        s = f"cq.Workplane('XY').workplane(offset={f(z0)})" + body
        if not taper and not flutes and rng.random() < 0.5:
            s += f".edges(near({on_circle(D / 2, z0)})).chamfer({f(min(hb * 0.06, D * 0.03))})"
    top_flat = top_r
    if flutes:
        c = min(groove_r * 0.6, hb * 0.08)
        s += f".faces('>Z').chamfer({f(c)})"
        top_flat -= c + groove_r
    else:
        fr = min(hb * 0.4, top_r * 0.35) * rng.uniform(0.3, 1.0)
        s += f".edges(near({on_circle(top_r, ztop)})).fillet({f(fr)})"
        top_flat -= fr
    k = rng.random()
    if k < 0.35:      # pointer line engraved on top
        x0, x1 = top_flat * 0.25, top_flat * 0.85
        s += (f".faces('>Z').workplane().pushPoints([({f((x0 + x1) / 2)},0)]).rect({f(x1 - x0)},{f(D * 0.035)})"
              f".cutBlind({f(-min(hb * 0.08, 4))})")
    elif k < 0.6:     # cap recess
        s += f".faces('>Z').workplane().hole({f(top_flat * rng.uniform(1.0, 1.5))},{f(hb * rng.uniform(0.05, 0.12))})"
    # underside and bore
    rb = D * rng.uniform(0.06, 0.12)
    depth = H * rng.uniform(0.45, 0.7)
    if rng.random() < 0.4 and H > 0.3 * D:     # hollow underside with a central boss
        wall = D * rng.uniform(0.06, 0.1)
        ro = D / 2 - groove_r - wall
        rboss = rb + wall
        hd = H * rng.uniform(0.4, 0.65)
        if ro > rboss + wall:
            s += f".faces('<Z').workplane().circle({f(ro)}).circle({f(rboss)}).cutBlind({f(-hd)})"
            depth = max(depth, hd + 0.15 * H)
    if rng.random() < 0.5:
        s += _dbore(rb, depth, rng.uniform(0.45, 0.7))
    else:
        s += f".faces('<Z').workplane().hole({f(2 * rb)},{f(depth)})"
        if rng.random() < 0.5:   # radial set-screw hole through the wall into the bore
            zs = z0 + min(depth * 0.5, H * 0.3)
            s += (f".cut(cq.Solid.makeCylinder({f(rb * 0.4)},{f(Ds / 2 + 5)},cq.Vector(0,0,{f(zs)}),"
                  f"cq.Vector(0,1,0)))")
    return s


def _star_knob(rng):
    OD = rng.uniform(100, 200)
    R = OD / 2
    n = rng.choice([3, 4, 5, 5, 6, 6, 7, 8])
    t = rd(OD * rng.uniform(0.16, 0.32))
    boss = rng.random() < 0.65
    rbz = R * rng.uniform(0.3, 0.42)
    hbz = rd(t * rng.uniform(0.4, 1.0)) if boss else 0.0
    H = t + hbz
    z0 = rd(-H / 2 + hbz)              # bottom of the star body
    if rng.random() < 0.6:    # lobes on a hub
        smax = math.sin(math.pi / n) / 1.15
        q = rng.uniform(0.25, min(0.6, smax))
        rl = R / (1 + q)
        lr = q * rl
        Rh = rng.uniform(rl - lr + 0.3 * lr, min(rl * 0.92, rl - lr + 1.2 * lr))
        sk = f"cq.Sketch().circle({f(Rh)}).parray({f(rl)},0,360,{n}).circle({f(lr)})"
        fr = min(lr, Rh) * rng.uniform(0.15, 0.35)
        inner = rl - lr
    else:                     # scalloped disc
        rs = R * math.sin(math.pi / n) * rng.uniform(0.55, 0.85)
        rc = R + rs * rng.uniform(0.1, 0.5)
        sk = (f"cq.Sketch().circle({f(R)}).parray({f(rc)},{f(180 / n)},360,{n})"
              f".circle({f(rs)},mode='s')")
        fr = rs * rng.uniform(0.1, 0.3)
        inner = rc - rs
    sk += f".clean().reset().vertices().fillet({f(fr)})"
    s = f"cq.Workplane('XY').workplane(offset={f(z0)}).placeSketch({sk}).extrude({f(t)})"
    c = min(t * rng.uniform(0.08, 0.15), fr * 0.8)
    if rng.random() < 0.6:
        s += f".faces('>Z or <Z').edges().fillet({f(c)})"
    else:
        s += f".faces('>Z or <Z').chamfer({f(min(c, fr * 0.5))})"
    rbz = min(rbz, inner * 0.9)
    if boss:
        s += f".faces('<Z').workplane().circle({f(rbz)}).extrude({f(hbz)})"
        if rng.random() < 0.5:
            s += f".faces('<Z').chamfer({f(hbz * 0.15)})"
    hole = rbz * rng.uniform(0.45, 0.65) if boss else inner * rng.uniform(0.3, 0.45)
    k = rng.random()
    thru = f".faces('>Z').workplane().circle({f(hole / 2)}).cutThruAll()"
    if k < 0.4:   # hex nut pocket from the top + through hole
        s += thru
        s += f".faces('>Z').workplane().polygon(6,{f(hole * 1.9)}).cutBlind({f(-t * rng.uniform(0.35, 0.6))})"
    elif k < 0.65:  # counterbored through hole
        s += thru + f".faces('>Z').workplane().hole({f(hole * 1.6)},{f(t * 0.3)})"
    elif k < 0.85:
        s += thru
    else:         # blind threaded insert from below
        s += f".faces('<Z').workplane().hole({f(hole)},{f(H * 0.7)})"
    return s


def _handwheel(rng):
    OD = rng.uniform(110, 200)
    R = OD / 2
    w = OD * rng.uniform(0.06, 0.1)                  # rim radial width
    Ri = R - w
    tr = rd(w * rng.uniform(0.9, 1.4))               # rim thickness
    dh = OD * rng.uniform(0.18, 0.28)
    th = rd(tr * rng.uniform(1.4, 2.6))
    up = rng.uniform(0.2, 0.8)                      # share of the hub protrusion above the rim
    zr0 = rd(-tr / 2)
    zh0 = rd(-tr / 2 - (th - tr) * (1 - up))
    peg = rng.random() < 0.35
    peg_h = rd(OD * rng.uniform(0.15, 0.3)) if peg else 0.0
    zmin = min(zr0, zh0)
    zmax = max(zr0 + tr + peg_h, zh0 + th)
    dz = rd((zmin + zmax) / 2)
    zr0, zh0 = rd(zr0 - dz), rd(zh0 - dz)
    rtop, rbot = zr0 + tr, zr0
    lines = []
    ts = rd(tr * rng.uniform(0.45, 0.7))
    zs0 = rd(zr0 + (tr - ts) / 2)
    fr = min(min(w, tr) * rng.uniform(0.25, 0.45), (tr - ts) / 2 * 0.9)   # rim rounds stay clear of the spokes
    rim = (f"cq.Workplane('XY').workplane(offset={f(zr0)}).circle({f(R)}).circle({f(Ri)}).extrude({f(tr)})"
           f".edges(near({on_circle(R, rtop)},{on_circle(R, rbot)},{on_circle(Ri, rtop)},{on_circle(Ri, rbot)}))"
           f".fillet({f(fr)})")
    lines.append("rim=" + rim)
    hub = f"cq.Workplane('XY').workplane(offset={f(zh0)}).circle({f(dh / 2)}).extrude({f(th)})"
    if rng.random() < 0.7:    # spokes
        n = rng.choice([3, 3, 4, 5, 5, 6])
        r0 = dh / 2 * 0.7
        r1 = Ri + w * 0.4
        sw = min(OD * rng.uniform(0.04, 0.07), 2 * r0 * math.sin(math.pi / n) * 0.85)
        web = (f"cq.Workplane('XY').workplane(offset={f(zs0)}).polarArray({f((r0 + r1) / 2)},0,360,{n})"
               f".rect({f(r1 - r0)},{f(sw)}).extrude({f(ts)})")
    else:                     # web with lightening holes
        n = rng.choice([3, 4, 5, 6, 8])
        rm = (dh / 2 + Ri) / 2
        span = Ri - dh / 2
        hd = min(span * rng.uniform(0.45, 0.7), 2 * rm * math.sin(math.pi / n) * 0.7)
        web = (f"cq.Workplane('XY').workplane(offset={f(zs0)}).circle({f(Ri + w * 0.4)}).extrude({f(ts)})"
               f".faces('>Z').workplane().polarArray({f(rm)},0,360,{n}).hole({f(hd)})")
    lines.append("web=" + web)
    body = "rim.union(" + hub + ").union(web)"
    if peg:
        pr = w * rng.uniform(0.3, 0.42)
        pz = rd(zr0 + tr * 0.5)
        ph = rd(peg_h + tr * 0.5)
        peg_s = (f"cq.Workplane('XY').workplane(offset={f(pz)}).center({f(Ri + w / 2)},0).circle({f(pr)})"
                 f".extrude({f(ph)}).faces('>Z').edges().fillet({f(pr * 0.6)})")
        body += ".union(" + peg_s + ")"
    bore = dh * rng.uniform(0.35, 0.55)
    lines.append("r=" + body)
    # bore through the hub (+ keyway) cut by a cylinder solid so it does not depend on which face is highest
    zb = rd(zh0 - 1)
    cut = f".cut(cq.Workplane('XY').workplane(offset={f(zb)}).circle({f(bore / 2)}).extrude({f(th + 2)}))"
    if rng.random() < 0.5:
        kw = bore * 0.25
        cut += (f".cut(cq.Workplane('XY').workplane(offset={f(zb)}).center(0,{f(bore / 2)})"
                f".rect({f(kw)},{f(kw)}).extrude({f(th + 2)}))")
    lines[-1] += cut
    return lines


def _grip(rng):
    L = rng.uniform(100, 200)
    d = L * rng.uniform(0.2, 0.38)
    R = d / 2
    z0 = rd(-L / 2)
    collar = rng.random() < 0.6
    hc = rd(L * rng.uniform(0.05, 0.1)) if collar else 0.0
    Rc = R * rng.uniform(1.15, 1.4)
    lb = rd(L - hc)
    lines = []
    if collar:
        s = f"cq.Workplane('XY').workplane(offset={f(z0)}).circle({f(Rc)}).extrude({f(hc)})"
        s += f".faces('>Z').workplane().circle({f(R)}).extrude({f(lb)})"
        if rng.random() < 0.5:
            s += f".edges(near({on_circle(Rc, z0)},{on_circle(Rc, z0 + hc)})).chamfer({f(hc * 0.25)})"
    else:
        s = f"cq.Workplane('XY').workplane(offset={f(z0)}).circle({f(R)}).extrude({f(lb)})"
        s += f".edges(near({on_circle(R, z0)})).chamfer({f(R * 0.1)})"
    zt = z0 + hc + lb
    if rng.random() < 0.6:
        s += f".edges(near({on_circle(R, zt)})).fillet({f(R * rng.uniform(0.3, 0.8))})"
    else:
        s += f".edges(near({on_circle(R, zt)})).chamfer({f(R * rng.uniform(0.1, 0.25))})"
    # grooves in a band away from both ends
    zg0 = z0 + hc + lb * rng.uniform(0.1, 0.2)
    zg1 = zt - R * 0.9 - lb * rng.uniform(0.02, 0.1)
    lg = zg1 - zg0
    if lg > 0.3 * L:
        k = rng.random()
        if k < 0.65:
            n = rng.choice([6, 8, 10, 12, 16, 20])
            pitch = math.pi * d / n
            if rng.random() < 0.5:
                tool = f".circle({f(pitch * rng.uniform(0.15, 0.28))})"
            else:
                tool = f".rect({f(d * rng.uniform(0.05, 0.1))},{f(pitch * rng.uniform(0.2, 0.35))})"
            s += (f".cut(cq.Workplane('XY').workplane(offset={f(zg0)}).polarArray({f(R)},0,360,{n})"
                  f"{tool}.extrude({f(lg)}))")
        else:
            m = rng.choice([2, 3, 4, 5])
            gw = min(lg / (2 * m), L * 0.04)
            gd = R * rng.uniform(0.06, 0.12)
            for i in range(m):   # generator-side loop: emits one literal ring cut per groove
                zc = zg0 + lg * (i + 0.5) / m - gw / 2
                s += (f".cut(cq.Workplane('XY').workplane(offset={f(zc)}).circle({f(R + 2)}).circle({f(R - gd)})"
                      f".extrude({f(gw)}))")
    # bore from the collar end
    depth = L * rng.uniform(0.3, 0.6)
    k = rng.random()
    rb = R * rng.uniform(0.3, 0.5)
    if k < 0.4:
        s += f".faces('<Z').workplane().hole({f(2 * rb)},{f(depth)})"
    elif k < 0.7:
        s += _dbore(rb, depth, rng.uniform(0.5, 0.7))
    else:
        s += f".faces('<Z').workplane().polygon(6,{f(2 * rb * 1.1)}).cutBlind({f(-depth)})"
    if rng.random() < 0.35:    # lanyard / tommy-bar cross hole near the free end
        zc = zt - R * 0.9 - L * 0.06
        s += (f".cut(cq.Solid.makeCylinder({f(R * 0.18)},{f(2 * Rc + 4)},cq.Vector(0,{f(-Rc - 2)},{f(zc)}),"
              f"cq.Vector(0,1,0)))")
    return ["r=" + s]


def generate(rng):
    k = rng.random()
    if k < 0.35:
        lines = ["r=" + _round_knob(rng)]
    elif k < 0.6:
        lines = ["r=" + _star_knob(rng)]
    elif k < 0.8:
        lines = _handwheel(rng)
    else:
        lines = _grip(rng)
    lines[-1] += orient(rng)
    return "import cadquery as cq\n" + "\n".join(lines)
