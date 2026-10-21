"""Hex hardware, modelled at canonical scale: hex nuts (plain, jam, coupling, flanged, acorn/cap, slotted/castle),
hex standoffs/spacers (female-female, male-female), hex bar stock pieces and hex-head bolts.

Typical construction: polygon(6, across_corners) prism; the 30-degree corner chamfer of a real nut is made by
intersecting the prism with a cylinder whose top/bottom circular edges are chamfered; through/blind holes, optional
countersunk hole entry, washer face, flange or collar, shank with chamfered tip.
"""
import math

from datagen.families.common import f, on_circle, orient

SQ3 = math.sqrt(3.0)


def _crown(R, s, z0, h, rng, top=True, bot=True):
    """Code for a chamfered cylinder used to trim the hex corners (intersect). R across-corners radius, s across flats."""
    rc = R * rng.uniform(1.0, 1.02)
    face_r = s / 2 * rng.uniform(0.88, 0.97)        # radius of the flat bearing circle left on the face
    c = min(rc - face_r, 0.45 * h)
    sel = []
    if top:
        sel.append(on_circle(rc, z0 + h))
    if bot:
        sel.append(on_circle(rc, z0))
    return (f"cq.Workplane('XY').workplane(offset={f(z0)}).circle({f(rc)}).extrude({f(h)})"
            f".edges(near({','.join(sel)})).chamfer({f(c)})")


def _nut(rng):
    kind = rng.choice(["plain", "plain", "jam", "coupling", "flanged", "flanged", "acorn", "castle"])
    hr = {"plain": rng.uniform(0.7, 0.95), "jam": rng.uniform(0.45, 0.6), "coupling": rng.uniform(2.2, 3.4),
          "flanged": rng.uniform(0.65, 0.85), "acorn": rng.uniform(0.6, 0.8), "castle": rng.uniform(0.7, 0.85)}[kind]
    if kind == "coupling":
        D = rng.uniform(100, 200) / (hr * SQ3 / 2)
    elif kind == "flanged":
        D = rng.uniform(85, 175)
    else:
        D = rng.uniform(100, 200)                    # across corners
    D = round(D, 1)
    R = D / 2
    s = R * SQ3                                      # across flats
    hole = s * rng.uniform(0.48, 0.62)
    h = round(hr * s, 1)
    lines = []
    if kind == "flanged":
        fd = min(D * rng.uniform(1.12, 1.3), 200) / 2       # flange radius
        ft = round(h * rng.uniform(0.18, 0.28), 1)
        H = h + ft
        z0 = round(-H / 2, 1)
        fl = (f"cq.Workplane('XY').workplane(offset={f(z0)}).circle({f(fd)}).extrude({f(ft)})")
        if rng.random() < 0.7:  # tapered flange top
            fl += f".edges(near({on_circle(fd, z0 + ft)})).chamfer({f(ft * 0.45)},{f((fd - R) * 0.6)})"
        lines.append("f=" + fl)
        hexs = f"cq.Workplane('XY').workplane(offset={f(z0 + ft)}).polygon(6,{f(D)}).extrude({f(h)})"
        lines.append("c=" + _crown(R, s, z0 + ft, h, rng, top=True, bot=False))
        body = f"{hexs}.intersect(c).union(f)"
        s_top = z0 + H
        b_bot = z0
    elif kind == "acorn":
        dome_r = s / 2 * rng.uniform(0.8, 0.92)
        dome_h = round(dome_r * rng.uniform(1.05, 1.3), 1)
        H = h + dome_h
        z0 = round(-H / 2, 1)
        lines.append("c=" + _crown(R, s, z0, h, rng, top=True, bot=rng.random() < 0.7))
        dome = (f"cq.Workplane('XY').workplane(offset={f(z0 + h - 1)}).circle({f(dome_r)}).extrude({f(dome_h + 1)})"
                f".faces('>Z').edges().fillet({f(dome_r * 0.95)})")
        lines.append("d=" + dome)
        body = f"cq.Workplane('XY').workplane(offset={f(z0)}).polygon(6,{f(D)}).extrude({f(h)}).intersect(c).union(d)"
        s_top = None
        b_bot = z0
    else:
        wf = kind in ("plain", "jam") and rng.random() < 0.35
        tw = round(h * 0.08, 1) if wf else 0.0
        H = h + tw
        z0 = round(-H / 2 + tw, 1)
        both = rng.random() < 0.75
        lines.append("c=" + _crown(R, s, z0, h, rng, top=True, bot=both and not wf))
        body = f"cq.Workplane('XY').workplane(offset={f(z0)}).polygon(6,{f(D)}).extrude({f(h)}).intersect(c)"
        if wf:   # washer face: thin bearing boss under the nut
            body += f".faces('<Z').workplane().circle({f(s * 0.47)}).extrude({f(tw)})"
        s_top = z0 + h
        b_bot = -H / 2
    s_code = body
    if kind == "acorn":   # blind tapped hole from below
        s_code += f".faces('<Z').workplane().hole({f(hole)},{f(h * rng.uniform(0.75, 1.0))})"
    else:
        s_code += f".faces('>Z').workplane().hole({f(hole)})"
        if kind == "castle":  # radial slots across the top
            sw = hole * rng.uniform(0.3, 0.45)
            sd = h * rng.uniform(0.25, 0.4)
            n_sl = 3
            s_code += (f".faces('>Z').workplane().polarArray(0,0,180,{n_sl}).rect({f(D * 1.1)},{f(sw)})"
                       f".cutBlind({f(-sd)})")
        elif rng.random() < 0.45:    # countersunk thread entry on top
            s_code += f".edges(near({on_circle(hole / 2, s_top)})).chamfer({f(hole * 0.08)})"
    lines.append("r=" + s_code)
    return lines


def _thumb(rng):
    """Knurled thumb nut: round fluted head, hex or round shoulder below, through hole."""
    Dh = rng.uniform(100, 200)
    th = round(Dh * rng.uniform(0.18, 0.32), 1)
    shoulder = rng.random() < 0.8
    hex_sh = rng.random() < 0.5
    sh_d = Dh * rng.uniform(0.45, 0.62)
    sh_h = round(th * rng.uniform(0.6, 1.4), 1) if shoulder else 0.0
    H = th + sh_h
    z0 = round(-H / 2 + sh_h, 1)
    n = rng.choice([12, 16, 18, 20, 24, 30])
    g = Dh * rng.uniform(0.012, 0.025) if n > 20 else Dh * rng.uniform(0.02, 0.04)
    b = f"cq.Workplane('XY').workplane(offset={f(z0)}).circle({f(Dh / 2)}).extrude({f(th)})"
    b += f".faces('>Z').workplane().polarArray({f(Dh / 2)},0,360,{n}).circle({f(g)}).cutBlind({f(-th)})"
    if shoulder:
        if hex_sh:
            b += f".faces('<Z').workplane().polygon(6,{f(sh_d)}).extrude({f(sh_h)})"
        else:
            b += f".faces('<Z').workplane().circle({f(sh_d / 2)}).extrude({f(sh_h)})"
            b += f".faces('<Z').chamfer({f(sh_h * 0.15)})"
    hole = sh_d * rng.uniform(0.45, 0.6) if shoulder else Dh * rng.uniform(0.2, 0.3)
    b += f".faces('>Z').workplane().hole({f(hole)})"
    if rng.random() < 0.5:
        b += f".edges(near({on_circle(hole / 2, z0 + th)})).chamfer({f(hole * 0.08)})"
    return ["r=" + b]


def _standoff(rng):
    L = rng.uniform(100, 200)
    s = L * rng.uniform(0.14, 0.32)                 # across flats
    D = 2 * s / SQ3
    z0 = -L / 2
    male = rng.random() < 0.4
    stud_l = L * rng.uniform(0.15, 0.25) if male else 0.0
    body_l = L - stud_l
    hole = s * rng.uniform(0.4, 0.55)
    c = s * rng.uniform(0.04, 0.08)
    lines = []
    b = f"cq.Workplane('XY').workplane(offset={f(z0)}).polygon(6,{f(D)}).extrude({f(body_l)})"
    if rng.random() < 0.6:
        b += f".faces('>Z or <Z').chamfer({f(c)})"
    else:   # turned (cylinder-trimmed) ends like bar stock cut on a lathe
        lines.append("c=" + _crown(D / 2, s, z0, body_l, rng))
        b += ".intersect(c)"
    if male:
        b += f".faces('>Z').workplane().circle({f(hole / 2)}).extrude({f(stud_l)})"
        b += f".faces('>Z').chamfer({f(hole * 0.12)})"
        b += f".faces('<Z').workplane().hole({f(hole)},{f(body_l * rng.uniform(0.3, 0.45))})"
    else:
        if rng.random() < 0.5:
            b += f".faces('>Z').workplane().hole({f(hole)})"
        else:
            dep = body_l * rng.uniform(0.25, 0.4)
            b += f".faces('>Z').workplane().hole({f(hole)},{f(dep)})"
            b += f".faces('<Z').workplane().hole({f(hole)},{f(dep)})"
    lines.append("r=" + b)
    return lines


def _bar(rng):
    """Hex bar stock piece with chamfered ends, optional cross hole and centre drill."""
    L = rng.uniform(100, 200)
    s = L * rng.uniform(0.2, 0.45)
    D = 2 * s / SQ3
    z0 = -L / 2
    b = f"cq.Workplane('XY').workplane(offset={f(z0)}).polygon(6,{f(D)}).extrude({f(L)})"
    b += f".faces('>Z or <Z').chamfer({f(s * rng.uniform(0.03, 0.08))})"
    k = rng.random()
    if k < 0.35:
        b += f".faces('>Z').workplane().hole({f(s * rng.uniform(0.3, 0.6))})"
    elif k < 0.6:
        b += f".faces('>Z').workplane().cskHole({f(s * 0.15)},{f(s * 0.35)},60,{f(L * 0.1)})"
        b += f".faces('<Z').workplane().cskHole({f(s * 0.15)},{f(s * 0.35)},60,{f(L * 0.1)})"
    if rng.random() < 0.5:  # cross holes through two opposite flats (flats face +-Y)
        n = rng.choice([1, 2])
        zs = [0.0] if n == 1 else [-L * 0.25, L * 0.25]
        dh = s * rng.uniform(0.15, 0.3)
        pts = ",".join(f"(0,{f(z)})" for z in zs)
        b += f".faces('>Y').workplane(centerOption='CenterOfBoundBox').pushPoints([{pts}]).hole({f(dh)})"
    return ["r=" + b]


def _bolt(rng):
    """Hex head bolt / cap screw blank: crowned hex head, optional washer face, plain shank, chamfered tip."""
    L = rng.uniform(100, 200)
    d = L * rng.uniform(0.12, 0.28)                 # shank diameter
    s = d * rng.uniform(1.45, 1.65)
    D = 2 * s / SQ3
    k = d * rng.uniform(0.6, 0.75)                  # head height
    z0 = -L / 2
    lines = ["c=" + _crown(D / 2, s, z0, k, rng, top=False, bot=True)]
    b = f"cq.Workplane('XY').workplane(offset={f(z0)}).polygon(6,{f(D)}).extrude({f(k)}).intersect(c)"
    b += f".faces('>Z').workplane().circle({f(d / 2)}).extrude({f(L - k)})"
    b += f".faces('>Z').chamfer({f(d * rng.uniform(0.08, 0.15))})"
    if rng.random() < 0.4:  # fillet under the head
        b += f".edges(near({on_circle(d / 2, z0 + k)})).fillet({f(d * 0.08)})"
    if rng.random() < 0.3:  # drilled shank (cotter-pin hole) through the flats direction
        b += (f".faces('>Y').workplane(centerOption='CenterOfBoundBox').pushPoints([(0,{f(L / 2 - d * 0.6)})])"
              f".hole({f(d * 0.22)})")
    lines.append("r=" + b)
    return lines


def generate(rng):
    k = rng.random()
    if k < 0.4:
        lines = _nut(rng)
    elif k < 0.5:
        lines = _thumb(rng)
    elif k < 0.68:
        lines = _standoff(rng)
    elif k < 0.82:
        lines = _bar(rng)
    else:
        lines = _bolt(rng)
    lines[-1] += orient(rng)
    return "import cadquery as cq\n" + "\n".join(lines)
