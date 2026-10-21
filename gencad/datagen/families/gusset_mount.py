"""Ribbed boss mounts: a T-, cross- or bar-shaped base plate (arms with rounded or chamfered ends, filleted inner
corners, optional round hub pad), a vertical boss with a bore (through or counterbored, optional chamfer and a
bolt circle of blind holes on its top), triangular ribs running from the boss along the arms, and bolt holes or
slots near the arm ends.

Built in the XY frame: the base outline is a polyline on XY extruded up (+Z), the boss is a circle extruded from the
base top. The ribs are triangles sketched on XZ / YZ workplanes through the boss axis (extruded symmetrically) and
unioned in; the finished part is centred and optionally oriented with one rotate.
"""
import math

from datagen.families.common import f, on_circle, orient


def _clean(pts):
    """Drop repeated and collinear vertices of a closed axis-aligned outline."""
    out = []
    for p in pts:
        if not out or abs(out[-1][0] - p[0]) > 1e-6 or abs(out[-1][1] - p[1]) > 1e-6:
            out.append(p)
    if abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) < 1e-6:
                out.pop(i)
                changed = True
                break
    return out


def generate(rng):
    shape = rng.choices(["cross", "tee", "bar"], [0.4, 0.35, 0.25])[0]
    hx = rng.uniform(0.07, 0.15)                       # half width of the X arms
    hy = hx if rng.random() < 0.6 else rng.uniform(0.07, 0.15)
    ax = rng.uniform(0.3, 0.5)
    arms = {"+x": ax, "-x": ax if rng.random() < 0.7 else rng.uniform(0.3, 0.5)}
    if shape in ("cross", "tee"):
        arms["+y"] = rng.uniform(0.25, 0.5)
    if shape == "cross":
        arms["-y"] = arms["+y"] if rng.random() < 0.7 else rng.uniform(0.25, 0.5)
    hub = rng.random() < 0.3
    hmin = min(hx, hy) if shape != "bar" else hx
    rb = rng.uniform(0.6, 0.95) * hmin if not hub else rng.uniform(0.8, 1.3) * hmin
    rh = rb * rng.uniform(1.15, 1.4) if hub else 0.0
    tb = rng.uniform(0.035, 0.07)
    hbs = rng.uniform(0.8, 2.6) * rb
    # scale so the longest side is S
    xs_ = arms["+x"] + arms["-x"]
    ys_ = arms.get("+y", hx) + arms.get("-y", hx)
    if hub:
        ys_ = max(ys_, 2 * rh)
    S = rng.uniform(100, 200)
    k = S / max(xs_, ys_, tb + hbs)
    hx, hy, rb, rh, tb, hbs = (round(v * k, 1) for v in (hx, hy, rb, rh, tb, hbs))
    for a in arms:
        arms[a] = round(arms[a] * k, 1)
    tb = max(tb, 4.0)
    if shape == "bar":
        hy = hx
    # outline (local coordinates, boss at 0,0)
    apx, anx = arms["+x"], arms["-x"]
    apy, any_ = arms.get("+y", hx), arms.get("-y", hx)
    outline = _clean([(apx, -hx), (apx, hx), (hy, hx), (hy, apy), (-hy, apy), (-hy, hx), (-anx, hx), (-anx, -hx),
                      (-hy, -hx), (-hy, -any_), (hy, -any_), (hy, -hx)])
    ymin, ymax = min(-any_, -rh), max(apy, rh)
    ox = round(-(apx - anx) / 2, 1)
    oy = round(-(ymax + ymin) / 2, 1)
    H = tb + hbs
    z0 = round(-H / 2, 1)
    zt = z0 + tb                                       # base top
    ztop = zt + hbs

    def P(x, y):
        return f"({f(x + ox)},{f(y + oy)})"

    def P3(x, y, z):
        return f"({f(x + ox)},{f(y + oy)},{f(z)})"

    s = (f"cq.Workplane('XY',origin=(0,0,{f(z0)})).polyline([" + ",".join(P(x, y) for x, y in outline)
         + f"]).close().extrude({f(tb)})")

    # ---------------- bolt holes on the arms
    d = max(4.0, min(rng.uniform(0.45, 0.8) * min(hx, hy), 2.5 * tb + 3, 18.0))
    kind = rng.choices(["hole", "cbore", "csk", "slot"], [0.4, 0.25, 0.15, 0.2])[0]
    if kind == "cbore" and tb < 0.45 * d + 2.5:
        kind = "hole"
    if kind == "csk" and tb < 0.3 * d + 1.5:
        kind = "hole"
    head = {"hole": d / 2, "csk": d, "cbore": 0.85 * d, "slot": d / 2}[kind]
    e = max(0.75 * d + 1.5, head + 1.5)
    two_across = rng.random() < 0.3
    tr = round(max(3.0, rng.uniform(0.3, 0.6) * min(hx, hy)), 1)
    tr = min(tr, 2 * min(hx, hy) - 2)
    holes, rib_end = [], {}
    slot_len = rng.uniform(1.5, 2.4) * d
    for a, L in arms.items():
        h = hx if a[1] == "x" else hy
        inner = (hy if a[1] == "x" else hx)            # arm starts beyond the centre square
        inner = max(inner, rh, rb) + head + 2
        if kind == "slot":
            pos = L - e - (slot_len - d) / 2
            lo = inner + (slot_len - d) / 2
        else:
            pos = L - e
            lo = inner
        if pos < lo:
            rib_end[a] = L - 2
            continue
        across = two_across and h - e >= head + 1.5 and h >= 2 * head + 3
        q = [(pos, -(h - e)), (pos, h - e)] if across else [(pos, 0.0)]
        for u, v in q:
            if a == "+x":
                holes.append((u, v))
            elif a == "-x":
                holes.append((-u, v))
            elif a == "+y":
                holes.append((v, u))
            else:
                holes.append((v, -u))
        clear_between = across and h - e - (slot_len / 2 if kind == "slot" else head) >= tr / 2 + 1.5
        rib_end[a] = L - 1.5 if clear_between else pos - (slot_len / 2 if kind == "slot" else head) - 2

    # ---------------- arm ends and inner corners (vertical edges of the plate)
    end_r = []
    for a, L in arms.items():
        h = hx if a[1] == "x" else hy
        for sgn in (-1, 1):
            if a == "+x":
                end_r.append((apx, sgn * hx))
            elif a == "-x":
                end_r.append((-anx, sgn * hx))
            elif a == "+y":
                end_r.append((sgn * hy, apy))
            else:
                end_r.append((sgn * hy, -any_))
    # corner radius: as round as the arm allows while every hole keeps its edge distance
    def corner_ok(rc):
        for a, L in arms.items():
            h = hx if a[1] == "x" else hy
            c = (L - rc, h - rc)
            for x, y in holes:
                u, v = {"+x": (x, y), "-x": (-x, y), "+y": (y, x), "-y": (-y, x)}[a]
                if u > L - rc - 1e-6 and abs(v) > h - rc - 1e-6 and abs(v) <= h + 1e-6 and u <= L:
                    if rc - math.hypot(u - c[0], abs(v) - c[1]) < head + 1.5:
                        return False
                elif u > L - rc and u <= L and abs(v) <= h:
                    if L - u < head + 1.5 or h - abs(v) < head + 1.5:
                        return False
        return True

    if rng.random() < 0.8:
        rc = rng.uniform(0.4, 0.97) * min(hx, hy)
        while rc > 2 and not corner_ok(rc):
            rc *= 0.85
        if rc > 2:
            op = "fillet" if rng.random() < 0.75 else "chamfer"
            if op == "chamfer":
                rc = min(rc, 0.7 * min(hx, hy))
            s += f".edges(near(" + ",".join(P3(x, y, zt - tb / 2) for x, y in end_r) + f")).{op}({f(rc)})"
    inner_c = []
    if shape != "bar":
        for sx in (-1, 1):
            for sy in (-1, 1):
                if (sy > 0 and "+y" in arms) or (sy < 0 and "-y" in arms):
                    inner_c.append((sx * hy, sy * hx))
    if inner_c and rng.random() < 0.7:
        ri = rng.uniform(0.3, 0.8) * min(hx, hy, 0.5 * min(arms.values()) - 0.5 * max(hx, hy))
        if ri > 1.5:
            s += f".edges(near(" + ",".join(P3(x, y, zt - tb / 2) for x, y in inner_c) + f")).fillet({f(ri)})"

    # ---------------- hub pad and boss
    if hub:                                            # round pad under the boss, same thickness as the plate
        s += (f".faces('<Z').workplane(centerOption='ProjectedOrigin',origin=({f(ox)},{f(oy)},0),invert=True)"
              f".circle({f(rh)}).extrude({f(tb)})")
    s += (f".faces('>Z').workplane(centerOption='ProjectedOrigin',origin=({f(ox)},{f(oy)},0))"
          f".circle({f(rb)}).extrude({f(hbs)})")
    rib_dirs = [a for a in arms if rng.random() < 0.85]
    if not rib_dirs and rng.random() < 0.7:
        rib_dirs = list(arms)
    if rng.random() < 0.6:
        rr = round(rng.uniform(0.2, 0.5) * min(tb, rb * 0.5), 1)
        if rr >= 1:
            s += f".edges(near({on_circle(rb, zt, ox, oy)})).fillet({f(rr)})"

    # ---------------- ribs
    hr = rng.uniform(0.6, 0.95) * hbs
    flat = rng.uniform(0.0, 0.25) * hbs if rng.random() < 0.5 else 0.0
    pre = ""
    j = 0
    for a in rib_dirs:
        L = arms[a]
        uend = min(rib_end.get(a, L - 2), L - 1.5)
        u0 = rb * 0.5
        if uend - rb < max(6.0, 0.6 * hr) or tr < 3:
            continue
        sgn = 1 if a[0] == "+" else -1
        prof = [(u0, zt - 0.5), (uend, zt - 0.5), (uend, zt + flat), (u0, zt + hr)]
        if a[1] == "x":                                 # XZ plane: local (u, v) = (X, Z), extrude along -Y
            pts = [(sgn * u + ox, v) for u, v in prof]
            wp = f"cq.Workplane('XZ',origin=(0,{f(oy)},0))"
        else:                                           # YZ plane: local (u, v) = (Y, Z), extrude along +X
            pts = [(sgn * u + oy, v) for u, v in prof]
            wp = f"cq.Workplane('YZ',origin=({f(ox)},0,0))"
        pre += (f"rib{j}={wp}.polyline([" + ",".join(f"({f(u)},{f(v)})" for u, v in pts)
                + f"]).close().extrude({f(tr / 2)},both=True)\n")
        s += f".union(rib{j})"
        j += 1

    # ---------------- bolt holes / slots in the arms (from the base top: local = global x, y)
    if holes:
        wp = f".faces('<Z').workplane(centerOption='ProjectedOrigin',origin=(0,0,0),invert=True,offset={f(tb)})"
        hp = ".pushPoints([" + ",".join(P(x, y) for x, y in holes) + "])"
        if kind == "hole":
            s += wp + hp + f".hole({f(d)})"
        elif kind == "cbore":
            s += wp + hp + f".cboreHole({f(d)},{f(1.7 * d)},{f(min(0.45 * d + 1, 0.55 * tb))})"
        elif kind == "csk":
            s += wp + hp + f".cskHole({f(d)},{f(2 * d)},{rng.choice([82, 90, 90])})"
        else:
            xh = [p for p in holes if abs(p[1]) < abs(p[0]) or (abs(p[1]) == abs(p[0]) and p[1] == 0)]
            yh = [p for p in holes if p not in xh]
            if xh:
                s += wp + ".pushPoints([" + ",".join(P(x, y) for x, y in xh) + "])" + \
                     f".slot2D({f(slot_len)},{f(d)},0).cutThruAll()"
            if yh:
                s += wp + ".pushPoints([" + ",".join(P(x, y) for x, y in yh) + "])" + \
                     f".slot2D({f(slot_len)},{f(d)},90).cutThruAll()"

    # ---------------- bore and boss-top features
    bore = rng.uniform(0.35, 0.65) * 2 * rb
    top_wp = f".faces('>Z').workplane(centerOption='ProjectedOrigin',origin=({f(ox)},{f(oy)},0))"
    u = rng.random()
    if u < 0.5:
        s += top_wp + f".hole({f(bore)})"
    elif u < 0.7 and rb - bore / 2 > 4:
        cb = min(bore + rng.uniform(0.3, 0.6) * (2 * rb - bore), 2 * rb - 5)
        s += top_wp + f".cboreHole({f(bore)},{f(cb)},{f(rng.uniform(0.15, 0.35) * hbs)})"
    elif u < 0.85:
        s += top_wp + f".hole({f(bore)},{f(rng.uniform(0.5, 0.9) * hbs)})"
    else:
        s += top_wp + f".hole({f(bore)})"
        cm = round(max(0.6, min(0.12 * bore, 0.3 * (rb - bore / 2))), 1)
        s += f".edges(near({on_circle(bore / 2, ztop, ox, oy)})).chamfer({f(cm)})"
    # bolt circle of blind holes on the boss top
    wall = rb - bore / 2
    dh = round(max(3.0, min(0.35 * wall, 10.0)), 1)
    if u < 0.5 and wall > dh + 5 and rng.random() < 0.45:
        bc = round(bore / 2 + wall / 2, 1)
        n = rng.choice([3, 4, 4, 6, 6, 8])
        while n > 3 and 2 * bc * math.sin(math.pi / n) < dh + 3:
            n -= 1
        s += top_wp + f".polarArray({f(bc)},{rng.choice([0, 0, 45 if n == 4 else 30])},360,{n}).hole({f(dh)},{f(min(2 * dh, 0.8 * hbs))})"
    elif u >= 0.5 and rng.random() < 0.4 and wall > 3:
        rt = round(rng.uniform(0.2, 0.5) * min(wall, 6.0), 1)
        if rt >= 0.8:
            s += f".edges(near({on_circle(rb, ztop, ox, oy)})).chamfer({f(rt)})"
    return "import cadquery as cq\n" + pre + "r=" + s + orient(rng)
