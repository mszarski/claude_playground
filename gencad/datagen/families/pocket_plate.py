"""Pocketed plates and blocks: a thick plate (optionally rounded vertical corners, chamfered top edges) with blind
pockets cut from the top: one large pocket (rectangle with filleted corners, rounded rectangle, obround or circle),
optionally stepped (a deeper inner pocket), with a through window or holes in its floor and small tapped-hole-like
blind holes in the surrounding wall; or an array of pockets / bearing seats separated by ribs.
"""
from datagen.families.common import f, on_circle, orient


def _pts(pts):
    return "[" + ",".join(f"({f(x)},{f(y)})" for x, y in pts) + "]"


def _pocket(rng, a, b, depth, zt, kind, rc, d0=0.0):
    """Code cutting a pocket of size a x b, depth `depth` from the top face at z=zt; kind picks the outline."""
    top = ".faces('>Z').workplane()"
    if kind == "circle":
        return top + f".circle({f(min(a, b) / 2)}).cutBlind({f(-depth)})"
    if kind == "slot":
        return top + f".slot2D({f(max(a, b))},{f(min(a, b))},{0 if a >= b else 90}).cutBlind({f(-depth)})"
    if kind == "sketch":
        return top + f".sketch().rect({f(a)},{f(b)}).vertices().fillet({f(rc)}).finalize().cutBlind({f(-depth)})"
    s = top + f".rect({f(a)},{f(b)}).cutBlind({f(-depth)})"
    if rc > 0.8:
        zm = zt - (depth + d0) / 2                           # mid-height of the new walls
        pts = ",".join(f"({f(x * a / 2)},{f(y * b / 2)},{f(zm)})" for x in (-1, 1) for y in (-1, 1))
        s += f".edges(near({pts})).fillet({f(rc)})"
    return s


def generate(rng):
    L = rng.uniform(100, 200)
    W = L * rng.uniform(0.45, 1.0)
    T = min(max(rng.uniform(0.12, 0.35) * W, 12), 0.35 * L)
    zt = T / 2
    w = max(rng.uniform(0.07, 0.16) * W, 6)                      # outer wall
    rc_out = rng.choice([0, 0, rng.uniform(0.3, 0.5) * w])

    if rng.random() < 0.5:
        s = f"cq.Workplane('XY').box({f(L)},{f(W)},{f(T)})"
    else:
        s = f"cq.Workplane('XY').workplane(offset={f(-T / 2)}).rect({f(L)},{f(W)}).extrude({f(T)})"
    if rc_out > 1.5:
        s += f".edges('|Z').fillet({f(rc_out)})"
    if rng.random() < 0.4:
        s += f".edges('>Z').chamfer({f(min(rng.uniform(0.05, 0.12) * T, 0.25 * w, 2.5))})"

    if rng.random() < 0.6:                                        # ---- one large pocket
        a, b = L - 2 * w, W - 2 * w
        if rng.random() < 0.4:                                    # smaller pocket, thicker walls
            a *= rng.uniform(0.55, 0.9)
            b *= rng.uniform(0.6, 0.95)
        kind = rng.choice(["rect", "rect", "sketch", "sketch", "slot", "circle"])
        if kind == "circle":
            a = b = min(a, b)
        dp = rng.uniform(0.3, 0.65) * T
        rc = rng.uniform(0.08, 0.3) * min(a, b)
        s += _pocket(rng, a, b, dp, zt, kind, rc)
        if kind == "circle" and rng.random() < 0.4:              # fillet the pocket floor
            s += f".edges(near({on_circle(a / 2, zt - dp)})).fillet({f(min(0.3 * dp, 0.1 * a))})"
        # inner feature in the floor
        k = rng.random()
        ia, ib = a * rng.uniform(0.4, 0.7), b * rng.uniform(0.4, 0.7)
        if k < 0.35:                                              # stepped (deeper) inner pocket
            kind2 = "circle" if kind == "circle" else rng.choice(["rect", "sketch", "circle"])
            dp2 = dp + rng.uniform(0.3, 0.6) * (T - dp)
            s += _pocket(rng, ia, ib, dp2, zt, kind2, rng.uniform(0.1, 0.3) * min(ia, ib), dp)
        elif k < 0.6:                                             # through window
            kind2 = "circle" if kind == "circle" or rng.random() < 0.3 else "rect"
            if kind2 == "circle":
                s += f".faces('>Z').workplane().hole({f(min(ia, ib))})"
            else:
                s += f".faces('>Z').workplane().rect({f(ia)},{f(ib)}).cutThruAll()"
        elif k < 0.8 and kind in ("rect", "sketch"):                 # four holes in the floor
            hd = max(min(0.12 * min(a, b), 12), 4)
            s += (f".faces('>Z').workplane().rarray({f(a * 0.6)},{f(b * 0.5)},2,2)"
                  f".hole({f(hd)})")
        # tapped-hole-like small holes in the wall
        wx, wy = (L - a) / 2, (W - b) / 2
        if min(wx, wy) >= 9 and rng.random() < 0.75:
            hd = min(rng.uniform(0.2, 0.4) * min(wx, wy), 10)
            hd = max(hd, 3)
            px, py = L / 2 - wx / 2, W / 2 - wy / 2
            pts = [(x * px, y * py) for x in (-1, 1) for y in (-1, 1)]
            if rng.random() < 0.5:
                pts += [(0, y * py) for y in (-1, 1)]
                if W > 0.7 * L:
                    pts += [(x * px, 0) for x in (-1, 1)]
            hop = f".hole({f(hd)},{f(min(rng.uniform(1.5, 3) * hd, 0.8 * T))})" if rng.random() < 0.7 \
                else f".hole({f(hd)})"
            s += f".faces('>Z').workplane().pushPoints({_pts(pts)})" + hop
    else:                                                         # ---- array of pockets separated by ribs
        rib = max(rng.uniform(0.04, 0.1) * W, 5)
        if L > 1.4 * W:
            nx, ny = rng.choice([2, 3, 3, 4]), rng.choice([1, 1, 2])
        else:
            nx, ny = rng.choice([2, 2, 3]), rng.choice([1, 2, 2])
        px = (L - 2 * w - (nx - 1) * rib) / nx
        py = (W - 2 * w - (ny - 1) * rib) / ny
        dp = rng.uniform(0.35, 0.7) * T
        top = ".faces('>Z').workplane()"
        arr = f".rarray({f(px + rib)},{f(py + rib)},{nx},{ny})"
        kind = rng.choice(["rect", "sketch", "circle", "circle"])
        if kind == "circle":
            dcir = min(px, py)
            s += top + arr + f".circle({f(dcir / 2)}).cutBlind({f(-dp)})"
            if rng.random() < 0.6:                                # through hole in each seat
                s += top + arr + f".hole({f(dcir * rng.uniform(0.3, 0.6))})"
        elif kind == "sketch":
            rr = rng.uniform(0.1, 0.3) * min(px, py)
            cen = [((i - (nx - 1) / 2) * (px + rib), (j - (ny - 1) / 2) * (py + rib))
                   for i in range(nx) for j in range(ny)]
            s += (top + f".sketch().push({_pts(cen)}).rect({f(px)},{f(py)}).reset().vertices().fillet({f(rr)})"
                  f".finalize().cutBlind({f(-dp)})")
        else:
            s += top + arr + f".rect({f(px)},{f(py)}).cutBlind({f(-dp)})"
            if rng.random() < 0.5 and min(px, py) > 10:          # through slot in each pocket floor
                sw = min(px, py) * rng.uniform(0.2, 0.4)
                sl = max(px, py) * rng.uniform(0.5, 0.8)
                s += top + arr + f".slot2D({f(sl)},{f(sw)},{0 if px >= py else 90}).cutThruAll()"
        if w >= 10 and rng.random() < 0.6:                       # mounting holes in the corners
            hd = max(min(0.4 * w, 9), 3.5)
            s += top + f".rect({f(L - w)},{f(W - w)},forConstruction=True).vertices().hole({f(hd)})"
    return "import cadquery as cq\nr=" + s + orient(rng, 0.55)
