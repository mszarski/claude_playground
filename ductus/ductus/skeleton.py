"""Skeleton strokes: named nodes joined by line or curve segments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import hobby_cubic, line_cubic


@dataclass
class Node:
    name: str
    pos: np.ndarray                 # resolved coordinates
    dir: float | None = None        # direction of travel in degrees, None = infer
    tension: float = 1.0            # tension of the segment arriving at this node
    terminal: Any = None            # truthy for stroke ends that take terminals
    fullname: str = ""


@dataclass
class Stroke:
    name: str
    nodes: list[Node]
    kinds: list[str]                # segment kinds, len = len(nodes)-1 (or len(nodes) if closed)
    closed: bool = False
    pen: Any = None
    fullname: str = ""

    def segment_ends(self, i: int) -> tuple[int, int]:
        return i, (i + 1) % len(self.nodes)

    def directions(self) -> list[float]:
        """Direction of travel (radians) at every node."""
        n = len(self.nodes)
        nseg = len(self.kinds)
        out = []
        for i, node in enumerate(self.nodes):
            if node.dir is not None:
                out.append(math.radians(node.dir))
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
            out.append(math.atan2(v[1], v[0]))
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
                out.append((hobby_cubic(p0, dirs[a], p3, dirs[b], self.nodes[b].tension), kind))
        return out

    def to_json(self, shift: float = 0.0) -> dict:
        def s(p):
            return [round(float(p[0]) + shift, 2), round(float(p[1]), 2)]
        return {
            "name": self.fullname,
            "closed": self.closed,
            "segments": [[s(p) for p in P] for P, _ in self.cubics()],
            "kinds": self.kinds,
            "nodes": [{"name": n.fullname, "x": s(n.pos)[0], "y": s(n.pos)[1]} for n in self.nodes],
        }
