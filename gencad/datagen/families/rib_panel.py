"""Ribbed panels and cast-looking bases: a base plate with a perimeter wall (optionally inset, leaving a bolting
flange), a grid or a parallel set of thin ribs (rarray of rects extruded from the floor, optionally drafted), bosses at
rib crossings or in the wall corners with through / counterbored / blind holes, fillets at the rib roots, corner rounds,
mounting holes and a chamfered underside.

All the walls, ribs and bosses are extruded from the floor workplane (the plate's top face): extrude() keeps the
workplane, so each rarray/pushPoints sketch starts on the floor. Horizontal-face selectors: CadQuery sorts '>Z[i]'
from the bottom, so '>Z[1]' is the second level from the bottom, i.e. the floor (and the flange top).
"""
from datagen.families.common import f, orient


def _pts(pts):
    return "[" + ",".join(f"({f(x)},{f(y)})" for x, y in pts) + "]"


def generate(rng):
    Lx = rng.uniform(100, 200)
    Ly = Lx * rng.uniform(0.45, 1.0)
    T = max(3.0, Lx * rng.uniform(0.025, 0.06))
    h = Lx * rng.uniform(0.07, 0.22)                         # wall height above the floor
    rt = max(2.5, Lx * rng.uniform(0.015, 0.03))             # rib thickness
    wall = rng.random() < 0.75
    w = rt * rng.uniform(1.0, 1.6) if wall else 0.0
    m = Ly * rng.uniform(0.07, 0.13) if wall and rng.random() < 0.4 else 0.0     # bolting flange
    A, B = Lx - 2 * m, Ly - 2 * m                            # wall outer size
    hr = h if rng.random() < 0.55 else h * rng.uniform(0.55, 0.9)                 # rib height
    taper = rng.choice([0, 0, 0, 1, 2])
    tan_t = {0: 0.0, 1: 0.01746, 2: 0.03492}[taper]
    if 2 * hr * tan_t > 0.5 * rt or (wall and hr * tan_t > 0.4 * w):
        taper, tan_t = 0, 0.0                                 # draft would make the ribs too thin at the top
    z0 = -(T + h) / 2
    s = f"cq.Workplane('XY').workplane(offset={f(z0)}).box({f(Lx)},{f(Ly)},{f(T)},centered=(True,True,False))"
    rc = 0.0
    if not wall or m > 0:
        if rng.random() < 0.7:
            rc = (m * rng.uniform(0.3, 0.6)) if m > 0 else min(Lx, Ly) * rng.uniform(0.03, 0.1)
            s += f".edges('|Z').fillet({f(rc)})"
    s += ".faces('>Z').workplane()"
    if wall:
        s += f".rect({f(A)},{f(B)}).rect({f(A - 2 * w)},{f(B - 2 * w)}).extrude({f(h)})"
        if m == 0 and rng.random() < 0.6:                    # flush wall: round the outer vertical corners
            pc = [(sx * A / 2, sy * B / 2) for sx in (-1, 1) for sy in (-1, 1)]
            zc = z0 + (T + h) / 2
            s += f".edges(near({','.join(f'({f(x)},{f(y)},{f(zc)})' for x, y in pc)})).fillet({f(w * rng.uniform(0.5, 0.9))})"
        ix, iy = A - 2 * w, B - 2 * w                        # inner pocket
        lx, ly = A - w + 2 * hr * tan_t, B - w + 2 * hr * tan_t   # ribs run into the wall (drafted ends too)
    else:
        e = 0.0 if rng.random() < 0.5 else rt * rng.uniform(1.5, 3)
        ix, iy = Lx - 2 * e, Ly - 2 * e
        lx, ly = ix, iy
        if e == 0 and rc:
            lx, ly = Lx - 2 * rc - rt, Ly - 2 * rc - rt     # keep rib ends off the rounded corners
            ix, iy = lx, ly
    # rib pattern
    pat = rng.choice(["grid", "grid", "grid", "x", "y"]) if wall else rng.choice(["grid", "grid", "x", "y"])
    cell_min = max(4 * rt, 14)
    nx = rng.randint(1, 5) if pat in ("grid", "x") else 0     # ribs parallel to Y, spread along X
    ny = rng.randint(1, 4) if pat in ("grid", "y") else 0
    while nx and ix / (nx + 1) < cell_min:
        nx -= 1
    while ny and iy / (ny + 1) < cell_min:
        ny -= 1
    if not wall and nx + ny == 0:
        nx = 1
    sx, sy = ix / (nx + 1), iy / (ny + 1)
    tp = f",taper={taper}" if taper else ""
    if nx:
        arr = f".rarray({f(sx)},1,{nx},1)" if nx > 1 else ""
        s += f"{arr}.rect({f(rt)},{f(ly)}).extrude({f(hr)}{tp})"
    if ny:
        arr = f".rarray(1,{f(sy)},1,{ny})" if ny > 1 else ""
        s += f"{arr}.rect({f(lx)},{f(rt)}).extrude({f(hr)}{tp})"
    xs = [-ix / 2 + sx * (i + 1) for i in range(nx)]
    ys = [-iy / 2 + sy * (j + 1) for j in range(ny)]
    # bosses
    bosses, db, hb = [], 0.0, h
    cell = min(sx if nx else ix, sy if ny else iy)
    if nx and ny and rng.random() < 0.6:
        nodes = [(x, y) for x in xs for y in ys]
        if len(nodes) > 6 and rng.random() < 0.5:           # only the outer nodes
            nodes = [(x, y) for x in (xs[0], xs[-1]) for y in (ys[0], ys[-1])]
            nodes = list(dict.fromkeys(nodes))
        db = min(rt * rng.uniform(2.5, 4.5), 0.55 * cell)
        bosses = nodes
    elif wall and rng.random() < 0.5:                         # corner bosses merged into the wall
        db = min(rt * rng.uniform(2.5, 4.5), 0.45 * min(ix, iy) / 2)
        cx, cy = ix / 2 - db * 0.35, iy / 2 - db * 0.35      # boss centre just inside the wall corner
        bosses = [(-cx, -cy), (cx, -cy), (cx, cy), (-cx, cy)]
    if bosses and db < 2.2 * rt:
        bosses = []
    if bosses:
        hb = h if rng.random() < 0.7 or not wall else h * rng.uniform(1.0, 1.15)
        if not wall:
            hb = hr * rng.uniform(1.0, 1.3)
        s += f".pushPoints({_pts(bosses)}).circle({f(db / 2)}).extrude({f(hb)})"
    # fillets at the rib / wall / boss roots (floor faces = second horizontal level from the bottom)
    top_z = max(h if wall else 0, hr, hb if bosses else 0)
    if (nx or ny or wall) and rng.random() < 0.55:
        rf = rt * rng.uniform(0.3, 0.7)
        if m > 0 or not wall:
            rf = min(rf, 0.6 * T)                             # plate's outer top edge is in that set too
        s += f".faces('>Z[1]').fillet({f(rf)})"
    if rng.random() < 0.4:
        s += f".faces('<Z').chamfer({f(T * rng.uniform(0.15, 0.35))})"
    # holes through the bosses
    if bosses:
        dh = db * rng.uniform(0.35, 0.55)
        k = rng.random()
        bot = [(x, -y) for x, y in bosses]                    # '<Z' workplane: local y = -global y
        if k < 0.4:
            s += f".faces('<Z').workplane(origin=(0,0,0)).pushPoints({_pts(bot)}).hole({f(dh)})"
        elif k < 0.65:
            cb = min(db * 0.9, dh * 1.7)
            s += (f".faces('<Z').workplane(origin=(0,0,0)).pushPoints({_pts(bot)})"
                  f".cboreHole({f(dh)},{f(cb)},{f(T * rng.uniform(0.4, 0.7))})")
        elif hb >= top_z - 1e-6 and (hb > h + 0.5 or not wall or hb > hr + 0.5):
            if hb > max(hr, h if wall else 0) + 0.5:
                s += (f".faces('>Z').workplane(origin=(0,0,0)).pushPoints({_pts(bosses)})"
                      f".hole({f(dh)},{f(hb * rng.uniform(0.5, 0.8))})")
            else:
                s += f".faces('<Z').workplane(origin=(0,0,0)).pushPoints({_pts(bot)}).hole({f(dh)})"
        else:
            s += f".faces('<Z').workplane(origin=(0,0,0)).pushPoints({_pts(bot)}).hole({f(dh)})"
    # mounting holes in the flange
    if m > 0 and rng.random() < 0.85:
        dm = m * rng.uniform(0.3, 0.45)
        q = rng.random()
        if q < 0.5 or Lx - m < 2.5 * m * 3:
            s += (f".faces('<Z').workplane(origin=(0,0,0)).rect({f(Lx - m)},{f(Ly - m)},forConstruction=True)"
                  f".vertices().hole({f(dm)})")
        else:                                                 # 3 per long side
            px = (Lx - m) / 2
            mp = [(x, y) for y in (-(Ly - m) / 2, (Ly - m) / 2) for x in (-px, 0, px)]
            if q < 0.75:
                s += f".faces('<Z').workplane(origin=(0,0,0)).pushPoints({_pts(mp)}).hole({f(dm)})"
            else:
                s += (f".faces('<Z').workplane(origin=(0,0,0)).pushPoints({_pts(mp)})"
                      f".cskHole({f(dm)},{f(min(dm * 1.8, m * 0.8))},90)")
    elif not wall and rng.random() < 0.5:                     # plain ribbed plate: corner holes
        gx = Lx / 2 - (xs[-1] + rt / 2 if nx else -1e9)       # edge -> outermost rib face
        gy = Ly / 2 - (ys[-1] + rt / 2 if ny else -1e9)
        gap = min(gx, gy)
        dm = min(gap * 0.4, Lx * 0.05)
        e = max(dm / 2 + 3, rc * 0.6 + dm / 2)
        if dm >= 3 and e + dm / 2 + 2 < gap:
            s += (f".faces('<Z').workplane(origin=(0,0,0)).rect({f(Lx - 2 * e)},{f(Ly - 2 * e)},forConstruction=True)"
                  f".vertices().hole({f(dm)})")
    return "import cadquery as cq\nr=" + s + orient(rng)
