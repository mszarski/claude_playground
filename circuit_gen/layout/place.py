"""
Greedy connectivity-driven placement.

Parts are placed one at a time, most-connected first. Each new part goes to
the free grid position that minimises the estimated wire length to the parts
already placed. Connectors are then put on the edge of the cluster nearest
the parts they connect to. This is deliberately simple: it produces compact,
routable boards for small designs, and it is the obvious stage to replace
with a learned placer later.

Coordinates are in millimetres; (x, y) is a part's centre.
"""

import math
from dataclasses import dataclass, field


@dataclass
class Item:
    ref: str
    width: float
    height: float
    nets: set = field(default_factory=set)
    is_connector: bool = False
    x: float = 0.0
    y: float = 0.0
    placed: bool = False

    def box(self, margin=0.0):
        hw, hh = self.width / 2 + margin, self.height / 2 + margin
        return (self.x - hw, self.y - hh, self.x + hw, self.y + hh)


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def net_weights(items, ignore_nets=()):
    """Weight of each net for wirelength: 1/(pins-1); big nets such as GND count little."""
    counts = {}
    for it in items:
        for n in it.nets:
            counts[n] = counts.get(n, 0) + 1
    return {n: (0.0 if n in ignore_nets else 1.0 / (c - 1)) for n, c in counts.items() if c > 1}


def wirelength(items, weights):
    """Weighted half-perimeter wire length over placed items."""
    total = 0.0
    boxes = {}
    for it in items:
        if not it.placed:
            continue
        for n in it.nets:
            if n not in weights:
                continue
            b = boxes.get(n)
            boxes[n] = (it.x, it.y, it.x, it.y) if b is None else (
                min(b[0], it.x), min(b[1], it.y), max(b[2], it.x), max(b[3], it.y))
    for n, b in boxes.items():
        total += weights[n] * ((b[2] - b[0]) + (b[3] - b[1]))
    return total


def _connection(a, b, weights):
    return sum(weights.get(n, 0.0) for n in a.nets & b.nets)


def _fits(item, placed, clearance):
    box = item.box(clearance / 2)
    return not any(_overlaps(box, p.box(clearance / 2)) for p in placed)


def _candidate_positions(cx, cy, step, max_radius):
    """Grid points in rings of increasing distance from (cx, cy)."""
    yield cx, cy
    r = 1
    while r * step <= max_radius:
        for i in range(-r, r + 1):
            for j in (-r, r):
                yield cx + i * step, cy + j * step
                if abs(i) != r:
                    yield cx + j * step, cy + i * step
        r += 1


def _place_near(item, placed, weights, clearance, step, max_radius, n_candidates=40):
    conn = [(p, _connection(item, p, weights)) for p in placed]
    total = sum(w for _, w in conn)
    if total > 0:
        cx = sum(p.x * w for p, w in conn) / total
        cy = sum(p.y * w for p, w in conn) / total
    elif placed:
        cx = sum(p.x for p in placed) / len(placed)
        cy = sum(p.y for p in placed) / len(placed)
    else:
        cx = cy = 0.0
    cx, cy = round(cx / step) * step, round(cy / step) * step

    best, found = None, 0
    others = placed + [item]
    for x, y in _candidate_positions(cx, cy, step, max_radius):
        item.x, item.y = x, y
        if not _fits(item, placed, clearance):
            continue
        item.placed = True
        cost = wirelength(others, weights) + 0.01 * math.hypot(x - cx, y - cy)
        item.placed = False
        if best is None or cost < best[0]:
            best = (cost, x, y)
        found += 1
        if found >= n_candidates:
            break
    if best is None:
        raise RuntimeError(f"no room to place {item.ref}")
    item.x, item.y = best[1], best[2]
    item.placed = True


def _place_connector(item, core, placed, weights, clearance, step):
    """Put a connector just outside the core cluster on the side nearest its nets."""
    x0 = min(p.box()[0] for p in core)
    y0 = min(p.box()[1] for p in core)
    x1 = max(p.box()[2] for p in core)
    y1 = max(p.box()[3] for p in core)
    conn = [(p, _connection(item, p, weights)) for p in core]
    total = sum(w for _, w in conn) or 1.0
    tx = sum(p.x * w for p, w in conn) / total if any(w for _, w in conn) else (x0 + x1) / 2
    ty = sum(p.y * w for p, w in conn) / total if any(w for _, w in conn) else (y0 + y1) / 2

    gap = clearance + 1.0
    sides = {
        "left": (x0 - gap - item.width / 2, ty, (0, 1)),
        "right": (x1 + gap + item.width / 2, ty, (0, 1)),
        "top": (tx, y0 - gap - item.height / 2, (1, 0)),
        "bottom": (tx, y1 + gap + item.height / 2, (1, 0)),
    }
    dist = {"left": tx - x0, "right": x1 - tx, "top": ty - y0, "bottom": y1 - ty}
    for side in sorted(sides, key=lambda s: dist[s]):
        sx, sy, (dx, dy) = sides[side]
        for k in range(0, 200):
            offset = (k + 1) // 2 * step * (1 if k % 2 else -1)
            item.x, item.y = round((sx + dx * offset) / step) * step, round((sy + dy * offset) / step) * step
            if _fits(item, placed, clearance):
                item.placed = True
                return
    raise RuntimeError(f"no room to place connector {item.ref}")


def place(items, clearance=1.0, step=0.5, ignore_nets=(), max_radius=60.0):
    """Assign x, y to every item in place. Returns the items."""
    weights = net_weights(items, ignore_nets)
    core = [it for it in items if not it.is_connector]
    connectors = [it for it in items if it.is_connector]
    if not core:
        core, connectors = connectors, []

    # Start from the part with the most connections (usually the main IC).
    def degree(it):
        return sum(weights.get(n, 0.0) for n in it.nets) + 0.01 * it.width * it.height

    remaining = sorted(core, key=degree, reverse=True)
    placed = []
    while remaining:
        if placed:
            remaining.sort(key=lambda it: (sum(_connection(it, p, weights) for p in placed),
                                           degree(it)), reverse=True)
        item = remaining.pop(0)
        _place_near(item, placed, weights, clearance, step, max_radius)
        placed.append(item)

    for item in sorted(connectors, key=degree, reverse=True):
        _place_connector(item, core, placed, weights, clearance, step)
        placed.append(item)
    return items


def bounding_box(items, margin=0.0):
    return (min(it.box()[0] for it in items) - margin, min(it.box()[1] for it in items) - margin,
            max(it.box()[2] for it in items) + margin, max(it.box()[3] for it in items) + margin)
