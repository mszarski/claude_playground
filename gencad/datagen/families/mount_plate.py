"""Mounting / adapter plates: rectangular plate (optionally rounded vertical corners, chamfered top edges), a mounting
hole pattern (corner holes, edge rows, corner + mid-edge holes or adjustment slots; plain, counterbored or
countersunk), and an optional central feature (bore, counterbored bore, bore with a bolt circle, rounded rectangular
cutout, slot(s), raised spigot with a bore).
"""
from datagen.families.common import f, on_circle, orient


def _hole_op(rng, d, t):
    """Hole op for diameter d in a plate of thickness t -> (code, outer radius of the feature)."""
    k = rng.random()
    if k < 0.45:
        return f".hole({f(d)})", d / 2
    if k < 0.75:
        cb = d * rng.uniform(1.5, 1.9)
        return f".cboreHole({f(d)},{f(cb)},{f(t * rng.uniform(0.25, 0.5))})", cb / 2
    cs = min(d * rng.uniform(1.7, 2.0), d + 1.6 * t)
    return f".cskHole({f(d)},{f(cs)},{rng.choice([82, 90, 90])})", cs / 2


def _pts(pts):
    return "[" + ",".join(f"({f(x)},{f(y)})" for x, y in pts) + "]"


def generate(rng):
    L = rng.uniform(100, 200)
    W = L * rng.choice([rng.uniform(0.4, 0.7), rng.uniform(0.6, 1.0)])
    t = max(min(rng.uniform(0.04, 0.13) * L, 0.3 * W), 5)
    rc = rng.choice([0, 0, rng.uniform(0.04, 0.14) * W])          # vertical corner radius

    # ---- mounting pattern (computed first: it bounds the central feature)
    d = max(rng.uniform(0.04, 0.09) * min(L, W * 1.6), 4)
    hop, ro = _hole_op(rng, d, t)
    m = max(ro + rng.uniform(0.4, 1.0) * d, rc * 0.8 + 0.6 * ro, 0.07 * W)  # edge distance of hole centres
    top = ".faces('>Z').workplane()"
    pat = rng.random()
    if pat < 0.25:
        mount = top + f".rarray({f(L - 2 * m)},{f(W - 2 * m)},2,2)" + hop
    elif pat < 0.45:
        mount = top + f".rect({f(L - 2 * m)},{f(W - 2 * m)},forConstruction=True).vertices()" + hop
    elif pat < 0.65 and L > 1.3 * W:                               # rows of holes along both long edges
        n = rng.choice([3, 4, 5]) if L > 140 else 3
        mount = top + f".rarray({f((L - 2 * m) / (n - 1))},{f(W - 2 * m)},{n},2)" + hop
    elif pat < 0.82:                                               # four adjustment slots along X
        sl = rng.uniform(1.8, 3.0) * d                             # overall slot length
        ro = d / 2
        cx, cy = L / 2 - m - (sl - d) / 2, W / 2 - m
        mount = top + f".pushPoints({_pts([(a * cx, b * cy) for a in (-1, 1) for b in (-1, 1)])})"
        mount += f".slot2D({f(sl)},{f(d)},0).cutThruAll()"
    else:                                                          # corners plus mid-edge holes
        pts = [(a * (L / 2 - m), b * (W / 2 - m)) for a in (-1, 1) for b in (-1, 1)]
        pts += [(0, b * (W / 2 - m)) for b in (-1, 1)]
        if W > 0.75 * L:
            pts += [(a * (L / 2 - m), 0) for a in (-1, 1)]
        mount = top + f".pushPoints({_pts(pts)})" + hop
    # half-sizes left for a central feature, keeping a wall to the mounting holes
    wall = max(0.05 * W, 4)
    hx, hy = L / 2 - m - ro - wall, W / 2 - m - ro - wall
    R = min(hx, hy)

    c = rng.random()
    boss = c < 0.2 and R > 15
    bh = rng.uniform(0.4, 1.0) * t if boss else 0.0
    z0 = -(t + bh) / 2

    if rng.random() < 0.5:
        s = f"cq.Workplane('XY').box({f(L)},{f(W)},{f(t)})"
        if boss:
            s += f".translate((0,0,{f(z0 + t / 2)}))"
    else:
        s = f"cq.Workplane('XY').workplane(offset={f(z0)}).rect({f(L)},{f(W)}).extrude({f(t)})"
    if rc > 2:
        s += f".edges('|Z').fillet({f(rc)})"
    if rng.random() < 0.45:
        s += f".edges('>Z').chamfer({f(min(rng.uniform(0.1, 0.3) * t, 3))})"
    s += mount

    if boss:                                                       # raised spigot with a bore (motor / bearing adapter)
        Db = rng.uniform(0.5, 0.95) * 2 * R
        ztop = z0 + t
        s += f".faces('>Z').workplane().circle({f(Db / 2)}).extrude({f(bh)})"
        if rng.random() < 0.5:
            s += f".edges(near({on_circle(Db / 2, ztop + bh)})).chamfer({f(min(0.3 * bh, 2))})"
        s += f".faces('>Z').workplane().hole({f(Db * rng.uniform(0.45, 0.75))})"
    elif R < 8:
        pass
    elif c < 0.4:                                                  # plain or counterbored bore
        bd = rng.uniform(0.3, 0.8) * 2 * R
        if rng.random() < 0.5:
            s += f".faces('>Z').workplane().hole({f(bd)})"
        else:
            cb = min(bd * rng.uniform(1.15, 1.4), 2 * R)
            s += f".faces('>Z').workplane().cboreHole({f(bd)},{f(cb)},{f(t * rng.uniform(0.25, 0.5))})"
    elif c < 0.6:                                                  # bore with a bolt circle around it (adapter plate)
        n = rng.choice([3, 4, 4, 6, 6, 8])
        hd = max(min(rng.uniform(0.14, 0.22) * R, d), 3)
        bc = R - hd / 2 - rng.uniform(0.0, 0.1) * R
        bd = 2 * (bc - hd / 2 - max(0.12 * R, 2.5))
        if bd > 6:
            s += f".faces('>Z').workplane().hole({f(bd)})"
            a0 = rng.choice([0, 45]) if n in (4, 8) else rng.choice([0, 90, 30])
            s += f".faces('>Z').workplane().polarArray({f(bc)},{a0},360,{n}).hole({f(hd)})"
    elif c < 0.8:                                                  # rounded rectangular cutout
        a = rng.uniform(0.4, 0.95) * 2 * hx
        b = rng.uniform(0.4, 0.95) * 2 * hy
        if min(a, b) > 8:
            rr = rng.uniform(0.1, 0.35) * min(a, b)
            if rng.random() < 0.5:
                s += (f".faces('>Z').workplane().sketch().rect({f(a)},{f(b)}).vertices().fillet({f(rr)})"
                      f".finalize().cutThruAll()")
            else:
                zm = z0 + t / 2
                s += f".faces('>Z').workplane().rect({f(a)},{f(b)}).cutThruAll()"
                pts = ",".join(f"({f(x * a / 2)},{f(y * b / 2)},{f(zm)})" for x in (-1, 1) for y in (-1, 1))
                s += f".edges(near({pts})).fillet({f(rr)})"
    elif c < 0.92:                                                 # central slot(s)
        sw = rng.uniform(0.15, 0.4) * 2 * hy
        sl = rng.uniform(0.5, 0.95) * 2 * hx
        if sw > 4 and sl > sw * 1.5:
            if rng.random() < 0.6 or 2 * hy < 3.2 * sw:
                s += f".faces('>Z').workplane().slot2D({f(sl)},{f(sw)}).cutThruAll()"
            else:
                p = min(rng.uniform(1.6, 2.2) * sw, 2 * hy - sw)
                s += f".faces('>Z').workplane().rarray(1,{f(p)},1,2).slot2D({f(sl)},{f(sw)}).cutThruAll()"
    return "import cadquery as cq\nr=" + s + orient(rng)
