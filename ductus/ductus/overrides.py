"""Hand adjustments stored as deltas on semantic names (spec section 5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

STAGES = ("skeleton", "outline")


@dataclass
class Override:
    name: str
    dx: float = 0.0
    dy: float = 0.0
    stage: str = "skeleton"


def parse_overrides(data: dict | None) -> dict[str, Override]:
    out = {}
    for name, spec in (data or {}).items():
        spec = spec or {}
        stage = spec.get("stage", "skeleton")
        if stage not in STAGES:
            raise ValueError(f"override {name}: unknown stage {stage!r}")
        out[name] = Override(name, float(spec.get("dx", 0)), float(spec.get("dy", 0)), stage)
    return out


def load_overrides(path) -> dict[str, Override]:
    p = Path(path)
    if not p.exists():
        return {}
    return parse_overrides(yaml.safe_load(p.read_text()))


def dump_overrides(ovs: dict[str, Override]) -> str:
    lines = []
    width = max((len(n) for n in ovs), default=0) + 1
    for name in sorted(ovs):
        o = ovs[name]
        parts = [f"dx: {o.dx:g}", f"dy: {o.dy:g}"]
        if o.stage != "skeleton":
            parts.append(f"stage: {o.stage}")
        lines.append(f"{(name + ':').ljust(width)} {{ {', '.join(parts)} }}")
    return "\n".join(lines) + ("\n" if lines else "")


def save_overrides(path, ovs: dict[str, Override]):
    Path(path).write_text(dump_overrides(ovs))


def add_delta(path, name: str, dx: float, dy: float, stage: str = "skeleton") -> dict[str, Override]:
    """Accumulate a drag delta into the overrides file (used by the canvas)."""
    ovs = load_overrides(path)
    o = ovs.get(name) or Override(name, stage=stage)
    o.dx += dx
    o.dy += dy
    ovs[name] = o
    save_overrides(path, ovs)
    return ovs
