"""Bushings, spacers, sleeves, nozzles and caps: a revolved (r, z) section of a tube with optional shoulders /
flanges, tapered (nozzle) outer, internal steps (counterbores, a tip orifice), entry chamfers, a closed end (cap),
then optional fillet under the shoulder, radial cross holes, wrench flats, flange bolt holes, or end-face holes.

Axis = Z: the section is drawn in the XZ plane and revolved about its local y axis (= global Z). Longest side 100-200.
"""
import math

from datagen.families.common import f, on_circle, orient


def _q(x):
    return round(float(x), 1)


def _clean(pts):
    pts = [(_q(a), _q(b)) for a, b in pts]
    out = []
    for p in pts:
        if not out or p != out[-1]:
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
    return out


def _r_at(prof, z):
    """Min radius of a z-monotonic (r, z) polyline over a tiny window at z (vertical steps take the smaller)."""
    best = None
    for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
        lo, hi = min(z0, z1), max(z0, z1)
        if lo - 1e-6 <= z <= hi + 1e-6:
            r = r0 if abs(z1 - z0) < 1e-9 else r0 + (r1 - r0) * (z - z0) / (z1 - z0)
            if abs(z1 - z0) < 1e-9:
                r = min(r0, r1)
            best = r if best is None else min(best, r)
    return best


def _rmax_over(prof, za, zb):
    """Max radius of an (r, z) polyline over segments overlapping the z-interval [za, zb]."""
    out = 0.0
    for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
        if max(z0, z1) >= za - 1e-6 and min(z0, z1) <= zb + 1e-6:
            out = max(out, r0, r1)
    return out


def generate(rng):
    kind = rng.choices(["sleeve", "flanged", "spacer", "nozzle", "cap"], [0.17, 0.3, 0.18, 0.17, 0.18])[0]
    S = rng.uniform(100, 200)                                        # longest side
    # body radius ro and length L (one of them sets the size)
    if kind in ("sleeve", "nozzle"):
        L = S
        ro = L * rng.uniform(0.12, 0.35)
    elif kind == "spacer":
        if rng.random() < 0.5:
            L = S
            ro = L * rng.uniform(0.15, 0.4)
        else:
            ro = S / 2
            L = S * rng.uniform(0.15, 0.6)
    elif kind == "flanged":
        if rng.random() < 0.5:
            L = S
            ro = L * rng.uniform(0.12, 0.25)
        else:
            L = 0
            ro = 0
    else:  # cap
        if rng.random() < 0.5:
            L = S
            ro = L * rng.uniform(0.2, 0.4)
        else:
            ro = S / 2
            L = S * rng.uniform(0.3, 0.8)
    if kind == "flanged" and L == 0:
        rf = S / 2
        ro = rf * rng.uniform(0.45, 0.7)
        L = S * rng.uniform(0.3, 0.8)
    else:
        rf = ro * rng.uniform(1.3, 1.8)
    if kind == "flanged" and 2 * rf > 200:
        rf = 99.0
    L, ro, rf = _q(L), _q(ro), _q(rf)
    z1 = _q(L / 2)
    z0, L = -z1, 2 * z1
    tw = max(ro * rng.uniform(0.2, 0.45), 2.5)                       # wall thickness
    ri = ro - tw

    # outer profile, bottom -> top, as (r, z)
    outer = []
    ce = _q(min(tw, L * 0.1) * rng.uniform(0.15, 0.3)) if rng.random() < 0.6 else 0
    flange_z = None
    if kind == "flanged":
        tf = _q(L * rng.uniform(0.12, 0.25))
        outer = [(rf, z0), (rf, z0 + tf), (ro, z0 + tf), (ro, z1)]
        flange_z = z0 + tf
        if rng.random() < 0.35:                                     # a second, smaller step at the top
            zs = _q(z1 - L * rng.uniform(0.15, 0.35))
            r2 = _q(ro - tw * rng.uniform(0.25, 0.45))
            outer = outer[:3] + [(ro, zs), (r2, zs), (r2, z1)]
    elif kind == "spacer":
        if rng.random() < 0.6:                                      # shoulder in the middle / both ends
            if rng.random() < 0.5:
                a, b = _q(-L * rng.uniform(0.05, 0.2)), _q(L * rng.uniform(0.05, 0.2))
                rs = _q(min(ro * rng.uniform(1.2, 1.5), 99))
                outer = [(ro, z0), (ro, a), (rs, a), (rs, b), (ro, b), (ro, z1)]
                flange_z = a
            else:
                rs = ro
                ro = _q(ro * rng.uniform(0.7, 0.85))
                ri = min(ri, ro - 2.5)
                tf = _q(L * rng.uniform(0.1, 0.2))
                outer = [(rs, z0), (rs, z0 + tf), (ro, z0 + tf), (ro, z1 - tf), (rs, z1 - tf), (rs, z1)]
                flange_z = z0 + tf
        else:
            outer = [(ro, z0), (ro, z1)]
    elif kind == "nozzle":
        rt = _q(ro * rng.uniform(0.35, 0.6))
        zc = _q(z0 + L * rng.uniform(0.3, 0.6))
        if rng.random() < 0.5:
            rb = _q(min(ro * rng.uniform(1.15, 1.35), 99.0))                    # hex-ish base collar
            zb = _q(z0 + L * rng.uniform(0.1, 0.2))
            outer = [(rb, z0), (rb, zb), (ro, zb), (ro, zc), (rt, z1 - L * 0.1), (rt, z1)]
        else:
            outer = [(ro, z0), (ro, zc), (rt, z1 - L * 0.08), (rt, z1)]
        ri = min(ri, rt * rng.uniform(0.4, 0.7))
    elif kind == "cap":
        if rng.random() < 0.5:
            tf = _q(L * rng.uniform(0.08, 0.18))
            outer = [(ro, z0), (ro, z1 - tf), (_q(min(ro * rng.uniform(1.08, 1.2), 99.0)), z1 - tf),
                     (_q(ro * rng.uniform(1.08, 1.2)), z1)]
            outer[3] = (outer[2][0], z1)
        else:
            outer = [(ro, z0), (ro, z1)]
    else:  # sleeve
        if rng.random() < 0.4:                                      # slight step down at one end
            zs = _q(z0 + L * rng.uniform(0.2, 0.4))
            r2 = _q(ro - tw * rng.uniform(0.2, 0.4))
            outer = [(r2, z0), (r2, zs), (ro, zs), (ro, z1)]
        else:
            outer = [(ro, z0), (ro, z1)]
    outer = [(_q(r), _q(z)) for r, z in outer]
    tmin = max(1.5, tw * 0.35)

    # inner profile: bore + optional counterbore steps at one or both ends; cap closes at top
    ztop = z1
    if kind == "cap":
        tcap = _q(max(2.5, L * rng.uniform(0.08, 0.2)))
        ztop = _q(z1 - tcap)
    rmin_outer = min(_r_at(outer, z) for z in [p[1] for p in outer] + [z0, ztop])
    ri = _q(max(1.5, min(ri, rmin_outer - tmin)))
    inner = [(ri, z0), (ri, ztop)]                                   # bottom -> top
    if rng.random() < 0.55 and kind != "nozzle":                     # counterbore from the bottom
        dz = _q(L * rng.uniform(0.15, 0.35))
        rc = min(_r_at(outer, z) for z in [z0, z0 + dz] + [p[1] for p in outer if z0 <= p[1] <= z0 + dz]) - tmin
        rc = _q(min(rc, ri + tw * 0.6))
        if rc > ri + 1:
            inner = [(rc, z0), (rc, z0 + dz), (ri, z0 + dz)] + inner[1:]
    if kind != "cap" and rng.random() < 0.4:                         # counterbore / step at the top
        dz = _q(L * rng.uniform(0.15, 0.3))
        zz = z1 - dz
        if zz > inner[-2][1] + 2:
            rc = min(_r_at(outer, z) for z in [zz, z1] + [p[1] for p in outer if zz <= p[1] <= z1]) - tmin
            rc = _q(min(rc, ri + tw * 0.6))
            if rc > ri + 1:
                inner = inner[:-1] + [(ri, zz), (rc, zz), (rc, z1)]
    if kind == "nozzle" and rng.random() < 0.7:                      # small orifice at the tip
        dz = _q(L * rng.uniform(0.08, 0.2))
        ro_ = _q(max(1.0, ri * rng.uniform(0.3, 0.6)))
        inner = inner[:-1] + [(ri, z1 - dz), (ro_, z1 - dz), (ro_, z1)]
    # internal groove (oil / O-ring groove)
    if rng.random() < 0.25 and kind in ("sleeve", "flanged", "spacer"):
        rlo = inner[0][1] if len(inner) == 2 else None
        segs = [(a, b) for a, b in zip(inner, inner[1:]) if a[0] == b[0] and b[1] - a[1] > 16]
        if segs:
            a, b = max(segs, key=lambda s: s[1][1] - s[0][1])
            zm = (a[1] + b[1]) / 2
            gw = _q(min(6, (b[1] - a[1]) * 0.15) + 1)
            rg = _q(a[0] + min(tw * 0.3, _r_at(outer, zm) - a[0] - tmin))
            if rg > a[0] + 0.6:
                gz0, gz1 = _q(zm - gw / 2), _q(zm + gw / 2)
                i = inner.index(b)
                inner = inner[:i] + [(a[0], gz0), (rg, gz0), (rg, gz1), (a[0], gz1)] + inner[i:]
    inner = [(_q(r), _q(z)) for r, z in inner]

    # entry chamfers in the profile (bore entries) and outer end chamfers
    ci = _q(min(inner[0][0], tw) * rng.uniform(0.1, 0.2)) if rng.random() < 0.5 else 0
    if ci >= 0.5 and inner[1][1] - inner[0][1] > 2 * ci and _r_at(outer, z0) - inner[0][0] > ci + tmin:
        inner = [(inner[0][0] + ci, z0), (inner[0][0], z0 + ci)] + inner[1:]
    if ce >= 0.5 and outer[1][1] - outer[0][1] > 2 * ce and outer[0][0] - inner[0][0] > ce + tmin + 1:
        outer = [(outer[0][0] - ce, z0), (outer[0][0], z0 + ce)] + outer[1:]
    if kind == "cap":
        pts = outer + [(0, z1), (0, ztop)] + inner[::-1][:-1] + [inner[0]]
        pts = outer + [(0, z1), (0, ztop)] + [p for p in inner[::-1] if p[1] <= ztop]
    else:
        pts = outer + inner[::-1]
    pts = _clean(pts)
    s = (f"cq.Workplane('XZ').polyline([{','.join(f'({f(a)},{f(b)})' for a, b in pts)}]).close()"
         f".revolve(360,(0,0,0),(0,1,0))")

    # fillet at the shoulder root
    if flange_z is not None and rng.random() < 0.55:
        rr = _r_at(outer, flange_z + 0.5)
        step_h = max(r for r, z in outer if abs(z - flange_z) < 1e-6) - rr
        seg = min(abs(z - flange_z) for r, z in outer if abs(z - flange_z) > 1e-6)
        fr = _q(min(step_h, seg, tw) * rng.uniform(0.15, 0.3))
        if fr >= 0.5:
            s += f".edges(near({on_circle(rr, flange_z)})).fillet({f(fr)})"
    # top outer edge chamfer by op
    if rng.random() < 0.35 and kind != "nozzle":
        rtop = outer[-1][0]
        rin_top = 0 if kind == "cap" else inner[-1][0]
        c = _q(min(rtop - rin_top - tmin, L * 0.08) * rng.uniform(0.15, 0.3))
        if c >= 0.5:
            s += f".edges(near({on_circle(rtop, z1)})).chamfer({f(c)})"

    rbig = max(r for r, z in outer)
    bolts = False
    # flange bolt holes (flanged kind with a big enough flange)
    if kind == "flanged" and rf - ro > 10 and rng.random() < 0.6:
        n = rng.choice([3, 4, 4, 6, 6, 8])
        bc = _q((rf + ro) / 2)
        hd = _q(min((rf - ro) * rng.uniform(0.35, 0.5), 2 * math.pi * bc / n * 0.4))
        wp = ".faces('<Z').workplane(centerOption='ProjectedOrigin')"
        bolts = True
        u = rng.random()
        if u < 0.5:
            s += wp + f".polarArray({f(bc)},0,360,{n}).hole({f(hd)})"
        elif u < 0.75:
            s += wp + f".polarArray({f(bc)},0,360,{n}).cskHole({f(hd)},{f(hd * 1.8)},90)"
        else:
            tf_ = outer[1][1] - z0 if outer[1][0] == rf else L * 0.1
            s += wp + f".polarArray({f(bc)},0,360,{n}).cboreHole({f(hd)},{f(hd * 1.6)},{f(max(0.5, tf_ * 0.3))})"
    # wrench flats on the body
    elif kind in ("sleeve", "spacer", "nozzle", "cap") and rng.random() < 0.3:
        # flats on the widest outer cylinder segment
        segs = [(a, b) for a, b in zip(outer, outer[1:]) if a[0] == b[0] and b[1] - a[1] > 6]
        if segs:
            a, b = max(segs, key=lambda sg: (sg[0][0], sg[1][1] - sg[0][1]))
            r_ = a[0]
            rin = _rmax_over(inner, a[1], b[1])
            across = max(r_ * rng.uniform(0.78, 0.9), rin + tmin + 1)
            if across < r_ - 0.8:
                t = _q(r_ - across + 1)
                ln = _q((b[1] - a[1]) + 2) if rng.random() < 0.5 else _q((b[1] - a[1]) * rng.uniform(0.4, 0.7))
                zc = _q((a[1] + b[1]) / 2)
                xc = _q(across + t / 2)
                box = f"cq.Workplane('XY').box({f(t)},{f(2 * r_ + 2)},{f(ln)})"
                s += f".cut({box}.translate(({f(xc)},0,{f(zc)})))"
                s += f".cut({box}.translate(({f(-xc)},0,{f(zc)})))"
    # radial cross hole(s)
    if rng.random() < 0.45:
        segs = [(a, b) for a, b in zip(outer, outer[1:]) if a[0] == b[0] and b[1] - a[1] > 8
                and not (bolts and a[0] >= rf - 0.05)]
        if segs:
            a, b = rng.choice(segs)
            hd = _q(max(2.0, min((b[1] - a[1]) * 0.4, a[0] * rng.uniform(0.2, 0.4))))
            zc = _q((a[1] + b[1]) / 2)
            if rng.random() < 0.5:
                s += f".cut(cq.Workplane('XZ').center(0,{f(zc)}).circle({f(hd / 2)}).extrude({f(rbig + 1)},both=True))"
            else:
                s += f".cut(cq.Workplane('YZ').center(0,{f(zc)}).circle({f(hd / 2)}).extrude({f(rbig + 1)}))"
    # cap end face: holes pattern or a centre hole
    if kind == "cap" and rng.random() < 0.5:
        rtop = outer[-1][0]
        if rng.random() < 0.5:
            s += f".faces('>Z').workplane().hole({f(_q(ri * rng.uniform(0.3, 0.7)))})"
        else:
            n = rng.choice([4, 6, 8])
            pc = _q(ri * rng.uniform(0.45, 0.6))
            hd = _q(min(ri * 0.3, 2 * math.pi * pc / n * 0.45))
            s += f".faces('>Z').workplane().polarArray({f(pc)},0,360,{n}).hole({f(hd)})"
    return "import cadquery as cq\nr=" + s + orient(rng)
