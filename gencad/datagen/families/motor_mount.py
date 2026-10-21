"""Motor mounts and clamp blocks:
  * plate: square/rectangular NEMA-style plate with a central pilot bore (plain or recessed) and a square 4-hole
    pattern (plain, counterbored, countersunk or slotted for belt tension), optional corner mounting holes;
  * bracket: the same motor face standing on a perpendicular base (L-bracket) with optional side gussets, an inner
    corner fillet and base slots;
  * clamp: split clamp block with a bore, a slit cut from the bore to one side, a cross clamping hole with a deep
    counterbore across the slit and counterbored mounting holes;
  * ring: motor clamp ring on a base: bore, slit through a top lug, cross clamping hole through the lug.
Built in the XY frame; side features are sketched on face workplanes (origin=(0,0,0) projects the world origin).
"""
import math

from datagen.families.common import f, orient


def _motor_pattern(rng, p, hd, pilot, t, wp, cz=None):
    """Pilot bore + 4-hole square pattern on the workplane chain `wp` (centred, or at local (0, cz))."""
    s = wp + (f".center(0,{f(cz)})" if cz is not None else "")
    if rng.random() < 0.35:              # recessed pilot (motor spigot seat) with a smaller shaft clearance hole
        s += f".cboreHole({f(pilot * rng.uniform(0.35, 0.6))},{f(pilot)},{f(t * rng.uniform(0.25, 0.5))})"
    else:
        s += f".hole({f(pilot)})"
    s += wp + (f".center(0,{f(cz)})" if cz is not None else "")     # fresh workplane: the stack now holds the solid
    s += f".rect({f(p)},{f(p)},forConstruction=True).vertices()"
    k = rng.random()
    if k < 0.4:
        s += f".hole({f(hd)})"
    elif k < 0.6:
        s += f".cboreHole({f(hd)},{f(hd * 1.8)},{f(t * rng.uniform(0.3, 0.5))})"
    elif k < 0.75:
        s += f".cskHole({f(hd)},{f(hd * 1.9)},90)"
    else:                                # tensioning slots (along local x)
        s += f".slot2D({f(hd * rng.uniform(1.8, 2.8))},{f(hd)},{rng.choice([0, 90])}).cutThruAll()"
    return s


def _plate(rng):
    S = rng.uniform(100, 200)
    S2 = S * (1.0 if rng.random() < 0.6 else rng.uniform(0.7, 0.95))
    t = S * rng.uniform(0.05, 0.12)
    p = S2 * rng.uniform(0.45, 0.65)
    pilot = p * rng.uniform(0.6, 1.0)
    hd = p * rng.uniform(0.08, 0.11)
    s = f"cq.Workplane('XY').box({f(S)},{f(S2)},{f(t)})"
    if rng.random() < 0.7:
        s += f".edges('|Z').fillet({f(min(S, S2) * rng.uniform(0.04, 0.12))})"
    s += _motor_pattern(rng, p, hd, pilot, t, ".faces('>Z').workplane(origin=(0,0,0))")
    # corner mounting holes, clear of the motor pattern
    m = S2 * rng.uniform(0.08, 0.12)
    hd2 = m * rng.uniform(0.7, 0.9)
    if rng.random() < 0.6 and S2 / 2 - m - hd2 / 2 > p / 2 + hd + 3:
        s += f".faces('<Z').workplane(origin=(0,0,0)).rect({f(S - 2 * m)},{f(S2 - 2 * m)},forConstruction=True).vertices().hole({f(hd2)})"
    elif rng.random() < 0.4 and S > S2 * 1.15:     # long plate: mounting slots at the ends
        xs = (S / 2 + max(p / 2, pilot / 2) + hd) / 2
        sl = min(S2 * 0.4, S / 2 - xs - 3)
        if sl > hd2 + 4:
            s += f".faces('<Z').workplane(origin=(0,0,0)).pushPoints([({f(-xs)},0),({f(xs)},0)]).slot2D({f(sl)},{f(hd2)},90).cutThruAll()"
    return s


def _bracket(rng):
    S = rng.uniform(90, 170)                       # face width (X)
    D = S * rng.uniform(0.5, 0.9)                  # base depth (Y)
    Hv = S * rng.uniform(0.85, 1.15)               # face height (Z)
    tb = S * rng.uniform(0.06, 0.1)
    tv = S * rng.uniform(0.06, 0.1)
    z0 = -Hv / 2
    hd0 = S * rng.uniform(0.04, 0.06)
    zfree = z0 + tb + min(tb, tv)                  # bottom of the free motor face (above the corner fillet)
    p = min(S * rng.uniform(0.4, 0.55), (Hv / 2 - zfree) - 3.5 * hd0 - 6)
    hd = min(hd0, p * 0.11)
    pilot = p * rng.uniform(0.6, 1.0)
    zc = (zfree + Hv / 2) / 2                      # middle of the free face
    s = f"cq.Workplane('XY').workplane(offset={f(z0)}).rect({f(S)},{f(D)}).extrude({f(tb)})"
    s += f".faces('<Y').workplane(origin=(0,0,0)).rect({f(S)},{f(Hv)}).extrude({f(-tv)})"
    yi = -D / 2 + tv
    if rng.random() < 0.5:                         # top corners of the face
        rr = S * rng.uniform(0.05, 0.15)
        if rng.random() < 0.5:
            s += f".edges(near(({f(-S / 2)},{f(-D / 2 + tv / 2)},{f(Hv / 2)}),({f(S / 2)},{f(-D / 2 + tv / 2)},{f(Hv / 2)}))).fillet({f(rr)})"
        else:
            s += f".edges(near(({f(-S / 2)},{f(-D / 2 + tv / 2)},{f(Hv / 2)}),({f(S / 2)},{f(-D / 2 + tv / 2)},{f(Hv / 2)}))).chamfer({f(rr)})"
    if rng.random() < 0.6:                         # inner corner fillet
        s += f".edges(near((0,{f(yi)},{f(z0 + tb)}))).fillet({f(min(tb, tv) * rng.uniform(0.5, 1.0))})"
    tg = 0.0
    if rng.random() < 0.55:                        # side gussets
        tg = min(tv * rng.uniform(0.7, 1.0), S / 2 - p / 2 - hd - 4)
        if tg > 2:
            gd = (D - tv) * rng.uniform(0.5, 0.85)
            gh = (Hv - tb) * rng.uniform(0.4, 0.8)
            v0 = z0 + tb
            u0 = D / 2 - tv                         # on '<X': local x = -Y
            tri = [(u0 + 1, v0 - 1), (u0 - gd, v0 - 1), (u0 + 1, v0 + gh), (u0 + 1, v0 - 1)]
            s += (f".faces('<X').workplane(origin=(0,0,0)).polyline([{','.join(f'({f(a)},{f(b)})' for a, b in tri)}])"
                  f".close().extrude({f(-tg)})")
            tri = [(-u0 - 1, v0 - 1), (-u0 + gd, v0 - 1), (-u0 - 1, v0 + gh), (-u0 - 1, v0 - 1)]
            s += (f".faces('>X').workplane(origin=(0,0,0)).polyline([{','.join(f'({f(a)},{f(b)})' for a, b in tri)}])"
                  f".close().extrude({f(-tg)})")
        else:
            tg = 0.0
    s += _motor_pattern(rng, p, hd, pilot, tv, ".faces('<Y').workplane(origin=(0,0,0))", zc)
    # base mounting holes / slots (from the underside: local y = -Y)
    free0, free1 = yi + 3, D / 2 - 3
    hb = S * rng.uniform(0.04, 0.065)
    xs = S / 2 - tg - hb * 1.5 - 3
    if free1 - free0 > 3 * hb and xs > hb * 2:
        yc = (free0 + free1) / 2
        if rng.random() < 0.5:
            sl = min((free1 - free0) * 0.6, hb * 3)
            s += f".faces('<Z').workplane(origin=(0,0,0)).pushPoints([({f(-xs)},{f(-yc)}),({f(xs)},{f(-yc)})]).slot2D({f(sl)},{f(hb)},90).cutThruAll()"
        else:
            s += f".faces('<Z').workplane(origin=(0,0,0)).pushPoints([({f(-xs)},{f(-yc)}),({f(xs)},{f(-yc)})]).hole({f(hb)})"
    return s


def _clamp(rng):
    L = rng.uniform(100, 180)                      # X
    W = L * rng.uniform(0.45, 0.75)                # Y
    H = L * rng.uniform(0.2, 0.45)                 # Z (bore axis)
    rb = W * rng.uniform(0.13, 0.22)
    yb = W * rng.uniform(0.0, 0.12)
    sw = max(1.0, rb * rng.uniform(0.08, 0.18))     # slit width
    s = f"cq.Workplane('XY').box({f(L)},{f(W)},{f(H)})"
    if rng.random() < 0.6:
        s += f".edges('|Z').fillet({f(min(W, L) * rng.uniform(0.05, 0.15))})"
    s += f".faces('>Z').workplane(origin=(0,0,0)).center(0,{f(yb)}).hole({f(2 * rb)})"
    # slit from the bore out through the +Y side
    s += f".faces('>Z').workplane(origin=(0,0,0)).center(0,{f((yb + W / 2) / 2)}).rect({f(sw)},{f(W / 2 - yb)}).cutThruAll()"
    # cross clamping hole along X across the slit, deep counterbore on one side
    ycr = (yb + rb + W / 2) / 2
    dcr = min((W / 2 - yb - rb) * rng.uniform(0.35, 0.5), H * 0.4)
    wall = max(dcr * 1.5, sw * 3, L * 0.08)
    cbd = L / 2 - sw / 2 - wall
    k = rng.random()
    side = ".faces('>X').workplane(origin=(0,0,0))"   # local x = +Y, y = Z
    if k < 0.75 and cbd > 2:
        s += side + f".center({f(ycr)},0).cboreHole({f(dcr)},{f(dcr * 1.7)},{f(cbd)})"
    else:
        s += side + f".center({f(ycr)},0).hole({f(dcr)})"
    # mounting holes through Z in the solid half, clear of the bore and the cross hole
    hm = min(L * rng.uniform(0.05, 0.08), (W / 2 + yb - rb) * 0.5)
    xm = (rb + L / 2) / 2 + rng.uniform(-0.05, 0.1) * L
    xm = min(xm, L / 2 - hm - 4)
    ym = min(-W * rng.uniform(0.0, 0.15), ycr - 0.85 * dcr - hm * 0.9 - 3)
    if xm - hm * 0.9 > rb + 3 and hm > 3:
        pts = f".pushPoints([({f(-xm)},{f(ym)}),({f(xm)},{f(ym)})])"
        if rng.random() < 0.6:
            s += f".faces('>Z').workplane(origin=(0,0,0))" + pts + f".cboreHole({f(hm)},{f(hm * 1.7)},{f(H * 0.25)})"
        else:
            s += f".faces('>Z').workplane(origin=(0,0,0))" + pts + f".hole({f(hm)})"
    if rng.random() < 0.6:                         # chamfer the bore mouths
        a = math.radians(-120)
        c = min(rb * 0.12, 2.0)
        p1 = f"({f(rb * math.cos(a))},{f(yb + rb * math.sin(a))},{f(H / 2)})"
        p2 = f"({f(rb * math.cos(a))},{f(yb + rb * math.sin(a))},{f(-H / 2)})"
        s += f".edges(near({p1},{p2})).chamfer({f(c)})"
    return s


def _ring(rng):
    Lb = rng.uniform(110, 190)
    Wb = Lb * rng.uniform(0.2, 0.4)
    tb = Lb * rng.uniform(0.06, 0.1)
    Ro = Lb * rng.uniform(0.2, 0.28)
    Ri = Ro * rng.uniform(0.7, 0.82)
    wl = Ro * rng.uniform(0.5, 0.8)                 # lug width (X)
    hl = Ro * rng.uniform(0.25, 0.45)               # lug height above the ring
    hc = Ro + tb * rng.uniform(0.6, 1.2)
    Lh = Wb * rng.uniform(0.8, 1.0)
    Htot = hc + Ro + hl
    z0 = -Htot / 2
    zc = z0 + hc
    ztop = zc + Ro + hl
    wp_ = 2 * Ro * rng.uniform(0.7, 0.85)
    s = f"cq.Workplane('XY').workplane(offset={f(z0)}).rect({f(Lb)},{f(Wb)}).extrude({f(tb)})"
    if rng.random() < 0.6:
        s += f".edges('|Z').fillet({f(Wb * rng.uniform(0.1, 0.25))})"
    hd = Lb * rng.uniform(0.04, 0.065)
    xb = (wp_ / 2 + Lb / 2) / 2 + hd * 0.3
    xb = min(xb, Lb / 2 - hd - 4)
    if rng.random() < 0.6:
        s += f".faces('>Z').workplane(origin=(0,0,0)).pushPoints([({f(-xb)},0),({f(xb)},0)]).hole({f(hd)})"
    else:
        s += f".faces('>Z').workplane(origin=(0,0,0)).pushPoints([({f(-xb)},0),({f(xb)},0)]).slot2D({f(hd * 2.2)},{f(hd)},0).cutThruAll()"
    zl = zc + (Ro + hl) / 2 + Ri / 2               # lug rect centre (from inside the ring to its top)
    lug_h = ztop - (zc + Ri * 0.5)
    s += (f".faces('<Y').workplane(origin=(0,0,0),offset={f((Lh - Wb) / 2)})"
          f".sketch().push([(0,{f(zc)})]).circle({f(Ro)})"
          f".push([(0,{f((z0 + zc) / 2)})]).rect({f(wp_)},{f(zc - z0)})"
          f".push([(0,{f(ztop - lug_h / 2)})]).rect({f(wl)},{f(lug_h)}).clean().finalize().extrude({f(-Lh)})")
    side_top = zc - math.sqrt(Ro * Ro - (wp_ / 2) ** 2)
    if side_top - (z0 + tb) > 2 and rng.random() < 0.6:
        fr = min(tb, side_top - z0 - tb, xb - wp_ / 2 - hd) * rng.uniform(0.3, 0.5)
        if fr > 0.8:
            s += f".edges(near(({f(-wp_ / 2)},0,{f(z0 + tb)}),({f(wp_ / 2)},0,{f(z0 + tb)}))).fillet({f(fr)})"
    # bore, slit through the lug, cross clamping hole through the lug
    s += f".faces(near((0,{f(-Lh / 2)},{f(zc)}))).workplane(origin=(0,0,0)).center(0,{f(zc)}).hole({f(2 * Ri)})"
    sw = max(1.0, Ro * rng.uniform(0.05, 0.1))
    s += f".faces('>Z').workplane(origin=(0,0,0)).rect({f(sw)},{f(Lh + 2)}).cutBlind({f(-(ztop - zc - Ri) - 1)})"
    zcr = zc + Ro + hl * 0.5 if hl > 8 else zc + (Ri + Ro + hl) / 2
    dcr = min(hl * 0.6, Lh * 0.35, (Ro + hl - Ri) * 0.45)
    cbd = wl / 2 - sw / 2 - max(dcr, 3)
    if cbd > 2 and rng.random() < 0.6:
        s += f".faces(near(({f(wl / 2)},0,{f(ztop - 1)}))).workplane(origin=(0,0,0)).center(0,{f(zcr)}).cboreHole({f(dcr)},{f(dcr * 1.7)},{f(cbd)})"
    else:
        s += f".faces(near(({f(wl / 2)},0,{f(ztop - 1)}))).workplane(origin=(0,0,0)).center(0,{f(zcr)}).hole({f(dcr)})"
    return s


def generate(rng):
    kind = rng.choices(["plate", "bracket", "clamp", "ring"], [0.3, 0.3, 0.2, 0.2])[0]
    body = {"plate": _plate, "bracket": _bracket, "clamp": _clamp, "ring": _ring}[kind](rng)
    return "import cadquery as cq\nr=" + body + orient(rng)
