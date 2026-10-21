"""Electronics enclosures and lids: a box with filleted vertical edges shelled to an open box (or a shallow lid
open at the bottom), internal screw bosses in the corners, a rebate or raised lip on the rim, connector cut-outs
through the side walls (rectangular, slotted or round), optional mounting ears / base flange with holes, vent slots
on lids.
"""
from datagen.families.common import f, on_circle, orient


def _rrect(a, b, r):
    """Rounded rectangle via the Sketch API, as code (call on a workplane)."""
    return f".sketch().rect({f(a)},{f(b)}).vertices().fillet({f(r)}).finalize()"


def _boss_inset(rng, rb, rc, t):
    """Distance from a corner boss centre to the inner wall faces: merged into the wall (overlapping it by <= 0.4 t,
    never poking through the outside, also around the rounded corner) or free-standing."""
    if rng.random() < 0.5:
        return rb + rng.uniform(1.0, 3.0)
    c = max(rb * rng.uniform(0.55, 0.8), rb - 0.4 * t)
    ri = rc - t
    while rb + 1.4143 * max(0.0, ri - c) > rc - 0.4 * t:
        c += 0.5
    return c


def _side_cuts(rng, L, W, H, t, z_lo, z_hi, keep_x):
    """Connector cut-outs through side walls. keep_x: half-length (along the wall) available for cut-outs."""
    s = ""
    walls = rng.sample(["<Y", ">Y", "<X", ">X"], rng.choice([1, 1, 2, 2, 3]))
    for wall in walls:
        span = keep_x[0] if wall in ("<Y", ">Y") else keep_x[1]     # usable half-span along this wall
        zh = z_hi - z_lo
        if span < 12 or zh < 10:
            continue
        zc = (z_lo + z_hi) / 2
        k = rng.random()
        wp = f".faces('{wall}').workplane()"
        if k < 0.4:                                    # one or two rectangular connector openings
            n = rng.choice([1, 1, 2, 3])
            w = rng.uniform(0.15, 0.5) * 2 * span / n
            h = rng.uniform(0.25, 0.6) * zh
            pitch = 2 * span / n
            xs = [(-span + pitch * (i + 0.5)) * rng.uniform(0.9, 1.0) for i in range(n)]
            pts = ",".join(f"({f(x)},{f(zc)})" for x in xs)
            s += wp + f".pushPoints([{pts}]).rect({f(w)},{f(h)}).cutBlind({f(-2 * t)})"
        elif k < 0.75:                                 # round holes (glands, jacks)
            n = rng.choice([1, 2, 2, 3, 4])
            pitch = 2 * span / n
            d = min(rng.uniform(0.35, 0.8) * pitch, rng.uniform(0.3, 0.7) * zh)
            if n == 1:
                s += wp + f".center({f(rng.uniform(-0.3, 0.3) * span)},{f(zc)}).circle({f(d / 2)}).cutBlind({f(-2 * t)})"
            else:
                s += wp + (f".center(0,{f(zc)}).rarray({f(pitch)},1,{n},1).circle({f(d / 2)})"
                           f".cutBlind({f(-2 * t)})")
        else:                                          # slot (e.g. USB / cable entry)
            h = rng.uniform(0.15, 0.35) * zh
            w = min(rng.uniform(0.3, 0.8) * 2 * span, 2 * span - 4)
            if w <= h + 2:
                continue
            s += wp + f".center({f(rng.uniform(-0.2, 0.2) * (2 * span - w))},{f(zc)}).slot2D({f(w)},{f(h)}).cutBlind({f(-2 * t)})"
    return s


def _box(rng):
    ears = rng.random()
    ear_kind = "none" if ears < 0.55 else ("ears" if ears < 0.8 else "flange")
    L = rng.uniform(100, {"none": 195, "ears": 150, "flange": 160}[ear_kind])
    W = L * rng.uniform(0.45, 0.95)
    H = L * rng.uniform(0.2, 0.6)
    t = rng.uniform(1.8, 4.5) * L / 140
    rc = rng.uniform(max(1.6 * t, t + 2), max(1.6 * t, t + 2, 0.15 * W))
    z0 = -H / 2
    bottom_r = 0.0
    if ear_kind == "none" and rng.random() < 0.4:
        hi = min(rc * 0.8, H * 0.2)
        lo = 1.4 * t
        if hi > lo:
            bottom_r = rng.uniform(lo, hi)
    s = f"cq.Workplane('XY').box({f(L)},{f(W)},{f(H)}).edges('|Z').fillet({f(rc)})"
    if bottom_r:
        s += f".edges('<Z').fillet({f(bottom_r)})"
    lip = rng.random()
    lip_kind = "none" if lip < 0.4 else ("rebate" if lip < 0.7 else "lip")
    lh = rng.uniform(1.0, 2.5) * t
    if lip_kind == "lip":                      # raised spigot lip, made before shelling so the shell follows it
        s += (f".faces('>Z').workplane(centerOption='CenterOfBoundBox')" + _rrect(L - t, W - t, rc - t / 2)
              + f".extrude({f(lh)})")
    s += f".faces('>Z').shell({f(-t)})"
    pre = ""
    ztop = H / 2
    zsurf = ztop + (lh if lip_kind == "lip" else 0)
    # corner screw bosses
    has_boss = rng.random() < 0.8
    rb = rng.uniform(2.2, 3.5) * t
    rb = min(rb, 0.12 * W)
    c = _boss_inset(rng, rb, rc, t)
    bx, by = L / 2 - t - c, W / 2 - t - c
    boss_top = ztop - rng.uniform(0.1, 0.3) * H
    hole_d = rb * rng.uniform(0.6, 0.9)
    if has_boss:
        zb = z0 + t / 2
        pre += (f"b=cq.Workplane('XY').workplane(offset={f(zb)}).rect({f(2 * bx)},{f(2 * by)},forConstruction=True)"
                f".vertices().circle({f(rb)}).extrude({f(boss_top - zb)})\n")
        s += ".union(b)"
        depth = (zsurf - boss_top) + (boss_top - z0 - t) * rng.uniform(0.6, 0.85)
        s += (f".faces('>Z').workplane(centerOption='CenterOfBoundBox').rect({f(2 * bx)},{f(2 * by)},forConstruction=True)"
              f".vertices().hole({f(hole_d)},{f(depth)})")
    if lip_kind == "rebate":                   # step cut into the inner half of the rim (lid seat)
        s += (f".faces('>Z').workplane(centerOption='CenterOfBoundBox')" + _rrect(L - t, W - t, rc - t / 2)
              + f".cutBlind({f(-lh)})")
    # connector cut-outs in the walls, clear of the corners / bosses and the floor / rim
    keep = (L / 2 - max(rc, t + c + rb) - 2, W / 2 - max(rc, t + c + rb) - 2)
    z_lo = z0 + t + max(bottom_r, 2) + 1
    z_hi = ztop - (lh if lip_kind == "rebate" else 0) - 3
    if rng.random() < 0.85:
        s += _side_cuts(rng, L, W, H, t, z_lo, z_hi, keep)
    # mounting ears / base flange (thickness <= floor so nothing pokes into the cavity)
    te = t * rng.uniform(0.85, 1.0)
    if ear_kind == "ears":
        e = rng.uniform(0.08, 0.14) * L
        we = rng.uniform(0.3, 0.6) * W
        xe = L / 2 + e / 2 - rc / 2
        er = min(we, e) * rng.uniform(0.2, 0.45)
        hd = min(e, we) * rng.uniform(0.3, 0.45)
        pre += (f"e=cq.Workplane('XY').workplane(offset={f(z0)}).pushPoints([({f(xe)},0),({f(-xe)},0)])"
                f".rect({f(e + rc / 1)},{f(we)}).extrude({f(te)}).edges('|Z').fillet({f(er)})\n")
        s += ".union(e)"
        if rng.random() < 0.5:
            s += (f".faces('<Z').workplane(centerOption='CenterOfBoundBox')"
                  f".pushPoints([({f(L / 2 + e / 2)},0),({f(-L / 2 - e / 2)},0)]).hole({f(hd)})")
        else:
            s += (f".faces('<Z').workplane(centerOption='CenterOfBoundBox')"
                  f".pushPoints([({f(L / 2 + e / 2)},0),({f(-L / 2 - e / 2)},0)]).slot2D({f(hd * 2)},{f(hd)},90)"
                  f".cutThruAll()")
    elif ear_kind == "flange":
        e = rng.uniform(0.06, 0.11) * L
        hd = e * rng.uniform(0.35, 0.5)
        pre += (f"e=cq.Workplane('XY').workplane(offset={f(z0)})" + _rrect(L + 2 * e, W + 2 * e, rc + e * 0.8)
                + f".extrude({f(te)})\n")
        s += ".union(e)"
        hx, hy = L / 2 + e / 2, W / 2 + e / 2
        if rng.random() < 0.5:
            s += (f".faces('<Z').workplane(centerOption='CenterOfBoundBox').rect({f(2 * hx)},{f(2 * hy)},forConstruction=True)"
                  f".vertices().hole({f(hd)})")
        else:
            s += (f".faces('<Z').workplane(centerOption='CenterOfBoundBox').rarray({f(2 * hx)},{f(W * 0.5)},2,2)"
                  f".hole({f(hd)})")
    return pre + "r=" + s


def _lid(rng):
    L = rng.uniform(100, 195)
    W = L * rng.uniform(0.45, 0.95)
    t = rng.uniform(1.8, 4.0) * L / 140
    H = max(rng.uniform(0.06, 0.16) * L, 3.5 * t)
    rc = rng.uniform(max(1.6 * t, t + 2), max(1.6 * t, t + 2, 0.15 * W))
    z0 = -H / 2
    s = f"cq.Workplane('XY').box({f(L)},{f(W)},{f(H)}).edges('|Z').fillet({f(rc)})"
    if rng.random() < 0.5:
        tr = min(rng.uniform(1.2, 2.5) * t, 0.5 * H, 0.8 * rc)
        if tr > 1.3 * t:
            s += f".edges('>Z').fillet({f(tr)})"
    s += f".faces('<Z').shell({f(-t)})"
    pre = ""
    rb = min(rng.uniform(2.2, 3.2) * t, 0.12 * W)
    c = _boss_inset(rng, rb, rc, t)
    bx, by = L / 2 - t - c, W / 2 - t - c
    hole_d = rb * rng.uniform(0.6, 0.85)
    if rng.random() < 0.6:                     # bosses hanging from the inside of the top plate
        zb = H / 2 - t / 2
        hb = rng.uniform(0.5, 1.0) * (H - t)
        pre += (f"b=cq.Workplane('XY').workplane(offset={f(zb - hb)}).rect({f(2 * bx)},{f(2 * by)},forConstruction=True)"
                f".vertices().circle({f(rb)}).extrude({f(hb)})\n")
        s += ".union(b)"
    pts = f".faces('>Z').workplane(centerOption='CenterOfBoundBox').rect({f(2 * bx)},{f(2 * by)},forConstruction=True).vertices()"
    k = rng.random()
    if k < 0.45:
        s += pts + f".cskHole({f(hole_d)},{f(min(hole_d * 2, 2 * rb + 2 * t))},90)"
    elif k < 0.75:
        s += pts + f".cboreHole({f(hole_d)},{f(min(hole_d * 1.7, 2 * rb))},{f(t * 0.5)})"
    else:
        s += pts + f".hole({f(hole_d)})"
    # locating lip under the lid
    if rng.random() < 0.5:
        lh = rng.uniform(1.0, 2.5) * t
        s += (f".faces('<Z').workplane(centerOption='CenterOfBoundBox')" + _rrect(L - t, W - t, rc - t / 2)
              + f".cutBlind({f(-lh)})")
    # vents / window on the top
    k = rng.random()
    ix, iy = bx - rb - 3, by - rb - 3
    if k < 0.35 and ix > 15 and iy > 15:
        n = rng.randint(3, 8)
        sw = min(rng.uniform(0.25, 0.5) * 2 * iy / n, 8)
        sl = rng.uniform(0.4, 0.9) * 2 * ix
        if sl > sw + 2:
            s += (f".faces('>Z').workplane(centerOption='CenterOfBoundBox').rarray(1,{f(2 * iy / n)},1,{n})"
                  f".slot2D({f(sl)},{f(sw)}).cutBlind({f(-2 * t)})")
    elif k < 0.55 and ix > 15 and iy > 15:
        w, h = rng.uniform(0.4, 0.9) * 2 * ix, rng.uniform(0.4, 0.9) * 2 * iy
        rw = min(w, h) * rng.uniform(0.05, 0.2)
        s += f".faces('>Z').workplane(centerOption='CenterOfBoundBox')" + _rrect(w, h, rw) + f".cutBlind({f(-t * 0.4)})"
    elif k < 0.7 and ix > 15 and iy > 15:
        nx, ny = rng.randint(3, 8), rng.randint(2, 5)
        p = min(2 * ix / nx, 2 * iy / ny) * rng.uniform(0.5, 0.9)
        d = p * rng.uniform(0.4, 0.7)
        s += (f".faces('>Z').workplane(centerOption='CenterOfBoundBox').rarray({f(p)},{f(p)},{nx},{ny})"
              f".hole({f(d)},{f(2 * t)})")
    # side cut-out on a lid skirt (cable notch)
    if rng.random() < 0.3 and H - t > 8:
        wn = rng.uniform(0.1, 0.25) * W
        hn = rng.uniform(0.3, 0.6) * (H - t)
        s += f".faces('<X').workplane().center(0,{f(z0)}).rect({f(wn)},{f(2 * hn)}).cutBlind({f(-2 * t)})"
    return pre + "r=" + s


def generate(rng):
    body = _lid(rng) if rng.random() < 0.3 else _box(rng)
    return "import cadquery as cq\n" + body + orient(rng)
