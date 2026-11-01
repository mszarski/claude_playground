"""Skeleton strokes: named nodes joined by line or curve segments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import hobby_cubic, line_cubic
from .params import val


@dataclass
class Node:
    name: str
    pos: np.ndarray                 # resolved coordinates
    dir: float | None = None        # direction of travel in degrees, None = infer
    tension: float = 1.0            # curve tension at this node (both sides); >1 flatter
    terminal: Any = None            # truthy for stroke ends that take terminals
    fullname: str = ""
    dw: float = 0.0                 # width delta at this node
    da: float = 0.0                 # direction delta (degrees)
    dpa: float = 0.0                # pen-angle delta at this node (degrees, interpolated along segments)

    @property
    def tension_eff(self) -> float:
        return max(0.4, self.tension)


@dataclass
class Stroke:
    name: str
    nodes: list[Node]
    kinds: list[str]                # segment kinds, len = len(nodes)-1 (or len(nodes) if closed)
    closed: bool = False
    pen: Any = None
    fullname: str = ""
    inside: Any = None              # closed strokes: share of contrast on the inner side
    profile: Any = None             # width profile along the stroke (ductus.profiles)
    bias: Any = None                # open strokes: share of modulation on the left of travel
    smooth: bool = False            # open strokes: solve free node directions for continuous
                                    # (mock) curvature, as METAFONT does; False = chord rule

    def is_ccw(self) -> bool:
        P = np.array([n.pos for n in self.nodes])
        x, y = P[:, 0], P[:, 1]
        return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) > 0

    def segment_ends(self, i: int) -> tuple[int, int]:
        return i, (i + 1) % len(self.nodes)

    def directions(self) -> list[float]:
        """Direction of travel (radians) at every node."""
        out = self._chord_directions()
        if self.smooth and not self.closed:
            out = self._hobby_directions(out)
        return out

    def _hobby_directions(self, chord_dirs: list[float]) -> list[float]:
        """Free node directions (``dir=None``, not next to a line) chosen so the
        mock curvature arriving at each node equals the one leaving it --
        Hobby's spline with unit tensions (the tensions still shape the
        handles). Free ends use curl 1. A direction imposed on an inflection
        makes both sides curl to meet it; with high contrast that dip turns
        into a kink in the width. Fixed directions stay as given."""
        n = len(self.nodes)
        if n < 3:
            return chord_dirs
        P = np.array([nd.pos for nd in self.nodes], float)
        ch = np.diff(P, axis=0)
        d = np.maximum(np.hypot(ch[:, 0], ch[:, 1]), 1e-9)
        c = np.unwrap(np.arctan2(ch[:, 1], ch[:, 0]))
        free = []
        for i, nd in enumerate(self.nodes):
            near_line = (i > 0 and self.kinds[i - 1] == "line") or (i < n - 1 and self.kinds[i] == "line")
            free.append(nd.dir is None and not near_line)
        if not any(free):
            return chord_dirs
        # reference angle per node, near its chords (for unwrapping fixed dirs)
        ref = np.array([c[0]] + [(c[i - 1] + c[i]) / 2 for i in range(1, n - 1)] + [c[-1]])
        known = np.array(chord_dirs, float)
        known = ref + (known - ref + math.pi) % (2 * math.pi) - math.pi
        A = np.zeros((n, n)); b = np.zeros(n)
        for k in range(n):
            if not free[k]:
                A[k, k] = 1.0; b[k] = known[k]
            elif k == 0:                                    # curl 1: theta_0 = phi_1
                A[k, 0] = 1.0; A[k, 1] = 1.0; b[k] = 2 * c[0]
            elif k == n - 1:                                # curl 1: phi_n = theta_(n-1)
                A[k, n - 2] = 1.0; A[k, n - 1] = 1.0; b[k] = 2 * c[n - 2]
            else:
                # METAFONT's mock-curvature equation for unit tensions,
                # A th[k-1] + (B + C) th[k] + D th[k+1] = -B psi[k] - D psi[k+1],
                # rewritten in absolute directions (th = dir - chord angle)
                A[k, k - 1] = 1 / d[k - 1]
                A[k, k] = 2 * (1 / d[k - 1] + 1 / d[k])
                A[k, k + 1] = 1 / d[k]
                b[k] = 3 * c[k - 1] / d[k - 1] + 3 * c[k] / d[k]
        sol = np.linalg.solve(A, b)
        return [float(sol[i]) + (math.radians(self.nodes[i].da) if free[i] else 0.0) if free[i] else chord_dirs[i]
                for i in range(n)]

    def _chord_directions(self) -> list[float]:
        n = len(self.nodes)
        nseg = len(self.kinds)
        out = []
        for i, node in enumerate(self.nodes):
            if node.dir is not None:
                out.append(math.radians(node.dir + node.da))
                continue
            before = (i - 1) % nseg if (self.closed or i > 0) else None
            after = i if (self.closed or i < nseg) else None
            prev_p = self.nodes[(i - 1) % n].pos if before is not None else None
            next_p = self.nodes[(i + 1) % n].pos if after is not None else None
            if after is not None and self.kinds[after] == "line":
                v = next_p - node.pos
            elif before is not None and self.kinds[before] == "line":
                v = node.pos - prev_p
            elif before is not None and after is not None:
                v = next_p - prev_p
            elif after is not None:
                v = next_p - node.pos
            else:
                v = node.pos - prev_p
            out.append(math.atan2(v[1], v[0]) + math.radians(node.da))
        return out

    def cubics(self) -> list[tuple[np.ndarray, str]]:
        dirs = self.directions()
        out = []
        for i, kind in enumerate(self.kinds):
            a, b = self.segment_ends(i)
            p0, p3 = self.nodes[a].pos, self.nodes[b].pos
            if kind == "line":
                out.append((line_cubic(p0, p3), kind))
            else:
                out.append((hobby_cubic(p0, dirs[a], p3, dirs[b],
                                         self.nodes[a].tension_eff, self.nodes[b].tension_eff), kind))
        return out

    def node_dirs(self) -> list[float]:
        """Actual tangent direction (radians) at each node, from the cubics."""
        cub = self.cubics()
        out = []
        for i in range(len(self.nodes)):
            if i < len(cub):
                P = cub[i][0]; v = P[1] - P[0] if np.linalg.norm(P[1] - P[0]) > 1e-9 else P[3] - P[0]
            else:
                P = cub[i - 1][0]; v = P[3] - P[2] if np.linalg.norm(P[3] - P[2]) > 1e-9 else P[3] - P[0]
            out.append(math.atan2(v[1], v[0]))
        return out

    def to_json(self, shift: float = 0.0, pen=None) -> dict:
        def s(p):
            return [round(float(p[0]) + shift, 2), round(float(p[1]), 2)]
        pen = self.pen or pen
        dirs = self.node_dirs()
        nodes = []
        for n, d in zip(self.nodes, dirs):
            w = (float(pen.widths(np.array([d]), np.array([math.radians(n.dpa)]))[0]) + n.dw
                 if pen is not None else None)
            nodes.append({"name": n.fullname, "x": s(n.pos)[0], "y": s(n.pos)[1],
                          "dir": round(math.degrees(d), 2), "width": None if w is None else round(w, 2),
                          "tension": round(n.tension_eff, 3), "terminal": bool(n.terminal),
                          "pen_angle": None if pen is None else round(val(pen.angle) + n.dpa, 2)})
        return {
            "name": self.fullname,
            "closed": self.closed,
            "segments": [[s(p) for p in P] for P, _ in self.cubics()],
            "kinds": self.kinds,
            "nodes": nodes,
        }
