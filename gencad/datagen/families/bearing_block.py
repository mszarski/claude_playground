"""Pillow blocks, bearing housings and shaft supports: a base plate with two bolt holes or slots (plain or
counterbored), a raised housing (round on a straight or tapered pedestal, or a square block) carrying a through bore
with a bearing-seat counterbore on one or both ends, fillets where the pedestal meets the base, chamfered bore
edges, optional grease-nipple boss with a lubrication hole.

Frame: base on XY (bottom at z0), shaft axis along Y at height zc. The housing profile is sketched on the base's
-Y end face and extruded along +Y.
"""
import math

from datagen.families.common import f, orient


def _on_xz_circle(r, y, zc, deg=57.0):
    """Point on a circle of radius r around the Y-parallel axis (x=0, z=zc) at plane y, away from seams."""
    a = math.radians(deg)
    return f"({f(r * math.cos(a))},{f(y)},{f(zc + r * math.sin(a))})"


def generate(rng):
    Lb = rng.uniform(110, 200)                         # base length (X)
    kind = rng.choices(["round", "taper", "square"], [0.4, 0.3, 0.3])[0]
    Wb = Lb * rng.uniform(0.22, 0.42)                  # base width (Y)
    tb = Lb * rng.uniform(0.07, 0.13)                  # base thickness
    Ro = Lb * rng.uniform(0.16, 0.26)                  # housing outer radius / half width
    rbr = Ro * rng.uniform(0.38, 0.55)                 # shaft through-bore radius
    rcb = min(rbr + (Ro - rbr) * rng.uniform(0.35, 0.55), Ro - 0.25 * (Ro - rbr) - 2)   # bearing seat radius
    zgap = max(tb + rcb + max(2.5, 0.35 * (Ro - rcb)), Ro * rng.uniform(0.85, 1.15) + 0.2 * tb)
    hc = zgap * rng.uniform(1.0, 1.25)                 # shaft height above base bottom
    Lh = Wb * rng.choice([1.0, 1.0, rng.uniform(0.7, 0.95), rng.uniform(1.05, 1.3)])   # housing length (Y)
    H = hc + Ro
    nip = kind != "square" and rng.random() < 0.35
    bh = Ro * rng.uniform(0.15, 0.3) if nip else 0.0   # grease boss height above housing
    z0 = -(H + bh) / 2
    zc = z0 + hc
    ztb = z0 + tb
    pre = ""
    s = f"cq.Workplane('XY').workplane(offset={f(z0)}).rect({f(Lb)},{f(Wb)}).extrude({f(tb)})"
    if rng.random() < 0.7:
        s += f".edges('|Z').fillet({f(min(Wb, Lb) * rng.uniform(0.08, 0.2))})"
    # pedestal width where it meets the base / the housing
    if kind == "round":
        w_top = w_bot = 2 * Ro * rng.uniform(0.7, 0.9)
    elif kind == "taper":
        w_top = 2 * Ro * rng.uniform(0.6, 0.85)
        w_bot = min(2 * Ro * rng.uniform(1.15, 1.5), Lb * 0.62)
        while True:                                    # sloped sides must stay clear of the bearing seat
            dx, dz = (w_bot - w_top) / 2, hc
            if w_top / 2 * dz / math.hypot(dx, dz) >= rcb + max(2.0, 0.3 * (Ro - rcb)):
                break
            w_top += 2.0
            if w_top > 1.9 * Ro:
                kind, w_top, w_bot = "round", 2 * Ro * 0.85, 2 * Ro * 0.85
                break
    else:
        w_top = w_bot = 2 * Ro
    # base bolt holes / slots, done on the plain base before the housing goes on
    hd = Lb * rng.uniform(0.045, 0.075)
    style = rng.choices(["hole", "cbore", "slot"], [0.4, 0.3, 0.3])[0]
    outer = hd * (1.8 if style == "cbore" else 1.0) / 2
    slot_l = hd * rng.uniform(1.8, 2.6) if style == "slot" else hd
    x_lo = w_bot / 2 + tb * 0.5 + max(outer, slot_l / 2) + 2
    x_hi = Lb / 2 - max(outer, slot_l / 2) - max(3, 0.6 * hd)
    if x_hi < x_lo:                                    # not enough room: smaller holes
        hd *= 0.7
        style = "hole"
        x_lo = w_bot / 2 + tb * 0.5 + hd / 2 + 2
        x_hi = Lb / 2 - hd / 2 - 3
    xb = (x_lo + x_hi) / 2 if x_hi >= x_lo else Lb / 2 - hd
    yb = Wb / 2 - max(outer, hd / 2) - max(3, 0.6 * hd)
    if style != "slot" and yb > max(outer, hd / 2) + 3 and rng.random() < 0.3:     # four-bolt base
        pts = f".rect({f(2 * xb)},{f(2 * yb)},forConstruction=True).vertices()"
    else:
        pts = f".pushPoints([({f(-xb)},0),({f(xb)},0)])"
    wp = ".faces('>Z').workplane(origin=(0,0,0))"
    if style == "hole":
        s += wp + pts + f".hole({f(hd)})"
    elif style == "cbore":
        s += wp + pts + f".cboreHole({f(hd)},{f(hd * 1.8)},{f(tb * rng.uniform(0.25, 0.45))})"
    else:
        s += wp + pts + f".slot2D({f(slot_l)},{f(hd)},0).cutThruAll()"
    # housing profile on the -Y end face
    s += f".faces('<Y').workplane(origin=(0,0,0),offset={f((Lh - Wb) / 2)})"
    if kind == "square":
        top_z = zc + Ro
        s += (f".sketch().push([(0,{f((z0 + top_z) / 2)})]).rect({f(2 * Ro)},{f(top_z - z0)}).finalize()"
              f".extrude({f(-Lh)})")
        if rng.random() < 0.7:                         # round the top long edges of the block
            fr = Ro * rng.uniform(0.2, 0.4)
            s += f".edges(near(({f(-Ro)},0,{f(top_z)}),({f(Ro)},0,{f(top_z)}))).fillet({f(fr)})"
    else:
        poly = [(-w_bot / 2, z0), (w_bot / 2, z0), (w_top / 2, zc), (-w_top / 2, zc), (-w_bot / 2, z0)]
        ps = ",".join(f"({f(x)},{f(z)})" for x, z in poly)
        s += (f".sketch().push([(0,{f(zc)})]).circle({f(Ro)}).reset().polygon([{ps}]).clean().finalize()"
              f".extrude({f(-Lh)})")
    # fillet where the pedestal sides meet the base top (only where straight side lines exist)
    if kind == "square":
        side_top = zc + Ro
    else:
        hw = w_top / 2
        side_top = zc - math.sqrt(max(Ro * Ro - hw * hw, 0)) if kind == "round" else zc - Ro * 0.5
    if side_top - ztb > 0.25 * tb and rng.random() < 0.75:
        xs = w_bot / 2 + (w_top - w_bot) / 2 * (tb / hc)   # side line x at the base top
        fr = min(tb, side_top - ztb, xb - xs - outer) * rng.uniform(0.25, 0.5)
        if fr > 0.8:
            s += f".edges(near(({f(-xs)},0,{f(ztb)}),({f(xs)},0,{f(ztb)}))).fillet({f(fr)})"
    # grease-nipple boss on top (before the bore so the bore clears its inside)
    if nip:
        rn = Ro * rng.uniform(0.18, 0.28)
        pre += f"n=cq.Workplane('XY').workplane(offset={f(zc)}).circle({f(rn)}).extrude({f(Ro + bh)})\n"
        s += ".union(n)"
    # through bore with bearing seat(s)
    yA, yB = -Lh / 2, Lh / 2
    rmid = (rcb + Ro) / 2 if kind != "square" else (rcb + Ro) / 2
    cbd = Lh * rng.uniform(0.2, 0.35)
    both = rng.random() < 0.6
    s += (f".faces(near((0,{f(yA)},{f(zc)}))).workplane(origin=(0,0,0)).center(0,{f(zc)})"
          f".cboreHole({f(2 * rbr)},{f(2 * rcb)},{f(cbd)})")
    if both:
        s += (f".faces(near({_on_xz_circle(rmid, yB, zc)})).workplane(origin=(0,0,0)).center(0,{f(zc)})"
              f".circle({f(rcb)}).cutBlind({f(-cbd)})")
    if kind == "square" and rng.random() < 0.5:     # set screw / oil hole from the top into the bore
        s += f".faces('>Z').workplane(origin=(0,0,0)).hole({f(Ro * rng.uniform(0.12, 0.2) * 2)},{f(Ro - rbr + 1)})"
    if rng.random() < 0.3:                          # relief channel across the underside of the base
        wr = min(w_bot * rng.uniform(0.5, 0.8), 2 * xb - 2 * outer - 6)
        if wr > 10:
            s += f".faces('<Z').workplane(origin=(0,0,0)).rect({f(wr)},{f(Wb + 2)}).cutBlind({f(-tb * rng.uniform(0.15, 0.3))})"
    if nip:
        s += f".faces('>Z').workplane(origin=(0,0,0)).hole({f(Ro * rng.uniform(0.06, 0.1) * 2)},{f(Ro + bh - rbr + 1)})"
    # chamfer the bearing-seat mouths
    if rng.random() < 0.7:
        c = min((Ro - rcb) * 0.3, rng.uniform(0.6, 2.0))
        p = [_on_xz_circle(rcb, yA, zc)]
        if both:
            p.append(_on_xz_circle(rcb, yB, zc))
        else:
            p.append(_on_xz_circle(rbr, yB, zc))
        s += f".edges(near({','.join(p)})).chamfer({f(c)})"
    return "import cadquery as cq\n" + pre + "r=" + s + orient(rng)
