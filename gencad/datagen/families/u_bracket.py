"""U-channels, clevis brackets and saddle mounts: a base plate with two upright ears, an aligned cross hole through
both ears (optional second pin hole, ear bosses, chamfered hole edges), base mounting holes or slots, fillets at the
ear roots, arc-ended (tangent-line) or flat ears, rounded or chamfered base corners.

Built in the XY frame: the ear profile is sketched on XY workplanes and extruded along Z (the pin axis), so the
cross holes are drilled from the '>Z' face. The base is a plate under the ears (normal -Y), drilled from its top.
"""
import math

from datagen.families.common import f, on_circle, orient


def _tangent(px, py, cx, cy, R, left):
    """Tangent point on the circle (cx, cy, R) of the line from (px, py), on the outer side."""
    vx, vy = px - cx, py - cy
    d = math.hypot(vx, vy)
    a = math.acos(min(1.0, R / d))
    ux, uy = vx / d, vy / d
    cands = []
    for s in (a, -a):
        c, sn = math.cos(s), math.sin(s)
        cands.append((cx + R * (ux * c - uy * sn), cy + R * (ux * sn + uy * c)))
    return min(cands) if left else max(cands)


def generate(rng):
    kind = rng.choices(["clevis", "channel", "saddle"], [0.4, 0.3, 0.3])[0]
    Lb = 1.0
    if kind == "clevis":
        wb = rng.uniform(0.35, 0.65)
        G = rng.uniform(0.25, 0.8)
        te = rng.uniform(0.07, 0.15)
        ext = 0.0 if rng.random() < 0.5 else rng.uniform(0.05, 0.3)
        hc = rng.uniform(0.35, 0.9)
    elif kind == "channel":
        wb = Lb
        G = rng.uniform(0.35, 1.0)
        te = rng.uniform(0.05, 0.12)
        ext = 0.0 if rng.random() < 0.8 else rng.uniform(0.05, 0.2)
        hc = rng.uniform(0.25, 0.8)
    else:
        wb = rng.uniform(0.25, 0.45)
        G = rng.uniform(0.2, 0.6)
        te = rng.uniform(0.07, 0.15)
        ext = rng.uniform(0.12, 0.4)
        hc = rng.uniform(0.2, 0.5)
    top = rng.choices(["arc", "flat"], [0.75, 0.25])[0]
    R = wb / 2 * (rng.uniform(0.55, 1.0) if top == "arc" and rng.random() < 0.6 else 1.0)
    if top == "flat":
        R = 0.0
    tb = rng.uniform(0.06, 0.13)
    Zb = G + 2 * te + 2 * ext
    Hy = tb + hc + (R if top == "arc" else wb * 0.35)
    S = rng.uniform(100, 200)
    k = S / max(Lb, Zb, Hy)
    Lb, wb, G, te, ext, hc, R, tb, Zb, Hy = (v * k for v in (Lb, wb, G, te, ext, hc, R, tb, Zb, Hy))
    te = max(te, 3.0)
    tb = max(tb, 3.0)
    Zb = G + 2 * te + 2 * ext
    ytop_flat = hc * rng.uniform(0.3, 0.5) if top == "flat" else 0.0
    Hy = tb + hc + (R if top == "arc" else ytop_flat)
    y0 = -Hy / 2
    yb = y0 + tb                                  # base top
    yc = yb + hc                                  # pin centre
    zi, zo = G / 2, G / 2 + te                    # ear inner / outer faces
    rf = round(rng.uniform(0.4, 0.9) * min(te, tb), 1) if rng.random() < 0.75 else 0.0

    # ---------------- ear profile
    if top == "arc":
        tl = _tangent(-wb / 2, yb, 0.0, yc, R, True)
        tr = (-tl[0], tl[1])
        ear = (f".moveTo({f(-wb / 2)},{f(yb)}).lineTo({f(tl[0])},{f(tl[1])})"
               f".threePointArc((0,{f(yc + R)}),({f(tr[0])},{f(tr[1])})).lineTo({f(wb / 2)},{f(yb)}).close()")
        ear_r = R
    else:
        yt = yc + ytop_flat
        ear = f".center(0,{f((yb + yt) / 2)}).rect({f(wb)},{f(yt - yb)})"
        ear_r = min(wb / 2, ytop_flat)
    boss = rng.random() < 0.25 and top == "arc"
    pre = f"e=cq.Workplane('XY',origin=(0,0,{f(zi)})){ear}.extrude({f(te)})"
    if top == "flat":
        cc = rng.uniform(0.3, 0.8) * min(wb / 2, ytop_flat)
        if cc > 1.5:
            yt = yc + ytop_flat
            pts = f"near(({f(-wb / 2)},{f(yt)},{f(zi + te / 2)}),({f(wb / 2)},{f(yt)},{f(zi + te / 2)}))"
            pre += f".edges({pts}).{rng.choice(['fillet', 'fillet', 'chamfer'])}({f(cc)})"
    dp = max(4.0, rng.uniform(0.35, 0.65) * 2 * ear_r)
    dp = min(dp, 2 * ear_r - 2 * max(3.0, 0.3 * ear_r))
    if dp < 4:
        dp = 0.0
    hb = 0.0
    if boss and dp:
        rbo = min(ear_r * rng.uniform(0.6, 0.9), dp / 2 + max(3.0, 0.45 * dp))
        hb = rng.uniform(0.3, 0.8) * te
        if yc - rbo < yb + rf + 2 or rbo < dp / 2 + 2.5:
            boss = False
        else:
            pre += (f".faces('>Z').workplane(centerOption='ProjectedOrigin',origin=(0,0,0))"
                    f".center(0,{f(yc)}).circle({f(rbo)}).extrude({f(hb)})")
    else:
        boss = False
    pre += "\n"

    s = f"cq.Workplane('XY').center(0,{f(y0 + tb / 2)}).rect({f(Lb)},{f(tb)}).extrude({f(Zb / 2)},both=True)"
    s += ".union(e).union(e.mirror('XY'))"

    # ---------------- root fillets
    if rf > 0:
        pts = [(0, yb, zi), (0, yb, -zi)]
        if ext > rf + 1:
            pts += [(0, yb, zo), (0, yb, -zo)]
        if Lb / 2 - wb / 2 > rf + 1:
            zm = (zi + zo) / 2
            pts += [(wb / 2, yb, zm), (-wb / 2, yb, zm), (wb / 2, yb, -zm), (-wb / 2, yb, -zm)]
        if len(pts) == 2 or rng.random() < 0.7:
            ps = ",".join(f"({f(a)},{f(b)},{f(c)})" for a, b, c in pts)
            s += f".edges(near({ps})).fillet({f(rf)})"
        else:
            rf_used = rf
            ps = ",".join(f"({f(a)},{f(b)},{f(c)})" for a, b, c in pts[:2])
            s += f".edges(near({ps})).fillet({f(rf_used)})"

    # ---------------- base mounting holes
    d = max(4.0, min(rng.uniform(0.07, 0.13) * max(Lb, Zb), 2.5 * tb + 3, 16.0))
    hk = rng.choices(["hole", "csk", "cbore", "slot"], [0.45, 0.2, 0.15, 0.2])[0]
    if hk == "cbore" and tb < 0.45 * d + 2.5:
        hk = "hole"
    if hk == "csk" and tb < 0.3 * d + 1.5:
        hk = "hole"
    head = {"hole": d / 2, "csk": d, "cbore": 0.85 * d, "slot": d / 2}[hk]
    e = max(0.8 * d + 1.5, head + 1.5)
    clr = rf + head + 2.0
    regions = []
    xin, xout = wb / 2 + clr, Lb / 2 - e                        # beside the ears (in X)
    if xout >= xin:
        regions.append("side")
    if zi - clr >= 0 and Lb / 2 - e > 0:                          # between the ears
        regions.append("between")
    zin, zout = zo + clr, Zb / 2 - e                            # outside the ears (in Z)
    if zout >= zin:
        regions.append("outside")
    pts = []
    slot_ang = 0
    if regions:
        reg = rng.choice(regions)
        if reg == "side":
            xh = rng.uniform(xin, xout) if rng.random() < 0.4 else xout
            zmax = Zb / 2 - e
            if zmax > d + 2 and rng.random() < 0.6:
                zh = rng.uniform(max(d + 2, 0.4 * zmax), zmax)
                pts = [(sx * xh, sz * zh) for sx in (-1, 1) for sz in (-1, 1)]
            else:
                pts = [(-xh, 0), (xh, 0)]
            slot_ang = 0 if rng.random() < 0.6 else 90
            room = 2 * min(xh - xin, xout - xh) + d if slot_ang == 0 else 2 * (Zb / 2 - e - max(abs(z) for _, z in pts)) + d
        elif reg == "between":
            xm = Lb / 2 - e
            n = rng.choice([1, 2, 2, 3]) if xm > d + 2 else 1
            if n == 1:
                pts = [(0, 0)]
            elif n == 2:
                xh = rng.uniform(max(d + 2, 0.4 * xm), xm)
                pts = [(-xh, 0), (xh, 0)]
            else:
                pts = [(-xm, 0), (0, 0), (xm, 0)]
            slot_ang = 90 if rng.random() < 0.5 else 0
            room = 2 * (zi - clr) + d if slot_ang == 90 else (d if n > 1 else 2 * xm + d)
        else:
            zh = rng.uniform(zin, zout) if rng.random() < 0.4 else (zin + zout) / 2
            xm = Lb / 2 - e
            if xm > d + 2 and rng.random() < 0.7:
                xh = rng.uniform(max(d + 2, 0.4 * xm), xm)
                pts = [(sx * xh, sz * zh) for sx in (-1, 1) for sz in (-1, 1)]
            else:
                pts = [(0, -zh), (0, zh)]
            slot_ang = 0
            room = d if len(pts) == 4 else 2 * xm + d
        # base workplane at the base top: local u = x, v = -z
        ptxt = "[" + ",".join(f"({f(x)},{f(-z)})" for x, z in pts) + "]"
        s += (f".faces('<Y').workplane(centerOption='ProjectedOrigin',origin=(0,0,0),invert=True,offset={f(tb)})"
              f".pushPoints({ptxt})")
        if hk == "hole":
            s += f".hole({f(d)})"
        elif hk == "csk":
            s += f".cskHole({f(d)},{f(2 * d)},{rng.choice([82, 90, 90])})"
        elif hk == "cbore":
            s += f".cboreHole({f(d)},{f(1.7 * d)},{f(min(0.45 * d + 1, 0.55 * tb))})"
        else:
            ln = min(rng.uniform(1.5, 2.6) * d, room)
            s += f".slot2D({f(ln)},{f(d)},{slot_ang}).cutThruAll()" if ln >= 1.25 * d else f".hole({f(d)})"

    # ---------------- base corners
    if pts:
        hx = max(abs(x) for x, _ in pts)
        hz = max(abs(z) for _, z in pts)
        cmax = min(Lb / 2 - hx, Zb / 2 - hz)
    else:
        cmax = min(Lb, Zb) / 4
    cmax = min(cmax, max(Lb / 2 - wb / 2, Zb / 2 - zo) - rf - 1)
    if cmax > 2 and rng.random() < 0.6:
        rc = rng.uniform(0.4, 0.95) * cmax
        cp = ",".join(f"({f(sx * Lb / 2)},{f(y0 + tb / 2)},{f(sz * Zb / 2)})" for sx in (-1, 1) for sz in (-1, 1))
        s += f".edges(near({cp})).{rng.choice(['fillet', 'fillet', 'chamfer'])}({f(rc)})"

    # ---------------- cross holes
    if dp:
        s += f".faces('>Z').workplane(centerOption='ProjectedOrigin',origin=(0,0,0)).pushPoints([(0,{f(yc)})])"
        u = rng.random()
        zface = zo + hb
        if u < 0.15 and boss:
            s += f".cboreHole({f(dp)},{f(dp * 1.35)},{f(hb * 0.8)})"
        else:
            s += f".hole({f(dp)})"
            if u > 0.7:
                cm = round(max(0.5, min(0.12 * dp, 0.25 * te)), 1)
                ps = ",".join([on_circle(dp / 2, zface, 0, yc), on_circle(dp / 2, -zface, 0, yc)])
                s += f".edges(near({ps})).chamfer({f(cm)})"
        # second, smaller pin hole lower on the ear (locking pin)
        d2 = rng.uniform(0.3, 0.5) * dp
        y2 = yc - dp / 2 - max(3.0, 0.5 * d2) - d2 / 2 - rng.uniform(0, 0.3 * hc)
        half_w = wb / 2 if top == "flat" else wb / 2 - (wb / 2 - abs(tl[0])) * (y2 - yb) / max(tl[1] - yb, 1e-6)
        if (rng.random() < 0.25 and d2 >= 3 and y2 - d2 / 2 > yb + rf + 3
                and half_w > d2 / 2 + 3 and (not boss or y2 + d2 / 2 < yc - rbo - 1)):
            s += (f".faces('>Z').workplane(centerOption='ProjectedOrigin',origin=(0,0,0))"
                  f".pushPoints([(0,{f(y2)})]).hole({f(d2)})")
    return "import cadquery as cq\n" + pre + "r=" + s + orient(rng)
