"""
Build a KiCad board from a circuit: load footprints, assign nets, place parts
and draw the board outline. Uses KiCad's `pcbnew` Python module.
"""

from dataclasses import dataclass
from pathlib import Path

import pcbnew

from .. import config
from ..checks import is_ground
from . import place as placer


@dataclass
class BoardRules:
    track_width: float = 0.25    # mm
    clearance: float = 0.2       # mm
    via_diameter: float = 0.6    # mm
    via_drill: float = 0.3       # mm
    hole_clearance: float = 0.15 # mm; some KiCad library footprints (USB-C) need < 0.25
    part_spacing: float = 1.0    # mm between part bounding boxes
    edge_margin: float = 2.0     # mm from parts to board edge


def _mm(v):
    return pcbnew.FromMM(v)


def load_footprint(fp_id):
    lib, name = fp_id.split(":", 1)
    path = Path(config.KICAD_FOOTPRINT_DIR) / f"{lib}.pretty"
    fp = pcbnew.FootprintLoad(str(path), name)
    if fp is None:
        raise KeyError(f"footprint {fp_id} not found")
    fp.SetFPID(pcbnew.LIB_ID(lib, name))
    return fp


def _apply_rules(board, rules):
    ds = board.GetDesignSettings()
    nc = ds.m_NetSettings.m_DefaultNetClass
    nc.SetTrackWidth(_mm(rules.track_width))
    nc.SetClearance(_mm(rules.clearance))
    nc.SetViaDiameter(_mm(rules.via_diameter))
    nc.SetViaDrill(_mm(rules.via_drill))
    ds.m_TrackMinWidth = _mm(min(rules.track_width, 0.2))
    ds.m_MinClearance = _mm(min(rules.clearance, 0.2))
    ds.m_HoleClearance = _mm(rules.hole_clearance)


def _draw_outline(board, x0, y0, x1, y1):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetStart(pcbnew.VECTOR2I(_mm(ax), _mm(ay)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(bx), _mm(by)))
        seg.SetWidth(_mm(0.1))
        board.Add(seg)


def build_board(circuit, rules=None, origin=(100.0, 100.0)):
    """Return (board, placement items) for a circuit dict from the runner."""
    rules = rules or BoardRules()
    board = pcbnew.BOARD()
    _apply_rules(board, rules)

    nets = {}
    for part in circuit["parts"]:
        for pin in part["pins"]:
            if pin["net"] and pin["net"] not in nets:
                net = pcbnew.NETINFO_ITEM(board, pin["net"])
                board.Add(net)
                nets[pin["net"]] = net

    footprints, items = {}, []
    for part in circuit["parts"]:
        fp = load_footprint(part["footprint"])
        fp.SetReference(part["ref"])
        fp.SetValue(part["value"])
        pin_nets = {p["num"]: p["net"] for p in part["pins"]}
        for pad in fp.Pads():
            net = pin_nets.get(pad.GetNumber())
            if net:
                pad.SetNet(nets[net])
        board.Add(fp)
        footprints[part["ref"]] = fp

        # Include visible text (the reference) so silkscreen labels don't overlap.
        bb = fp.GetBoundingBox(True, False)
        items.append(placer.Item(
            ref=part["ref"],
            width=pcbnew.ToMM(bb.GetWidth()),
            height=pcbnew.ToMM(bb.GetHeight()),
            nets={n for n in pin_nets.values() if n},
            is_connector=part["ref"].startswith("J"),
        ))

    ground = {n for n in nets if is_ground(n)}
    placer.place(items, clearance=rules.part_spacing, ignore_nets=ground)

    ox, oy = origin
    for it in items:
        fp = footprints[it.ref]
        # Footprint origins are often not their centre (e.g. pin 1 of a header).
        bb = fp.GetBoundingBox(True, False)
        off = bb.GetCenter() - fp.GetPosition()
        fp.SetPosition(pcbnew.VECTOR2I(_mm(ox + it.x) - off.x, _mm(oy + it.y) - off.y))

    x0, y0, x1, y1 = placer.bounding_box(items, margin=rules.edge_margin)
    _draw_outline(board, ox + x0, oy + y0, ox + x1, oy + y1)
    return board, items


def board_size_mm(items, rules=None):
    rules = rules or BoardRules()
    x0, y0, x1, y1 = placer.bounding_box(items, margin=rules.edge_margin)
    return x1 - x0, y1 - y0
