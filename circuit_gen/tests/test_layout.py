"""
Tests for placement, session import and the full build pipeline.
"""

from pathlib import Path

import pytest

from circuit_gen.layout.place import Item, place, bounding_box, wirelength, net_weights

from conftest import needs_pcbnew, needs_toolchain

EXAMPLES = Path(__file__).parent.parent / "examples"


def _overlap(a, b):
    a, b = a.box(), b.box()
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class TestPlacement:
    def _items(self):
        return [
            Item("U1", 6, 5, {"VCC", "GND", "A", "B", "C"}),
            Item("R1", 3, 1.5, {"VCC", "A"}),
            Item("R2", 3, 1.5, {"A", "B"}),
            Item("C1", 3, 1.5, {"B", "GND"}),
            Item("C2", 3, 1.5, {"VCC", "GND"}),
            Item("D1", 3, 1.5, {"C", "GND"}),
            Item("J1", 3, 6, {"VCC", "GND"}, is_connector=True),
        ]

    def test_no_overlaps(self):
        items = place(self._items(), clearance=1.0, ignore_nets={"GND"})
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                assert not _overlap(a, b), f"{a.ref} overlaps {b.ref}"

    def test_connector_outside_core(self):
        items = place(self._items(), clearance=1.0, ignore_nets={"GND"})
        core = [it for it in items if not it.is_connector]
        x0, y0, x1, y1 = bounding_box(core)
        j = next(it for it in items if it.ref == "J1")
        jx0, jy0, jx1, jy1 = j.box()
        assert jx1 <= x0 or jx0 >= x1 or jy1 <= y0 or jy0 >= y1

    def test_beats_naive_row(self):
        items = place(self._items(), clearance=1.0)
        weights = net_weights(items)
        placed = wirelength(items, weights)

        row = self._items()
        x = 0.0
        for it in row:
            it.x, it.y, it.placed = x + it.width / 2, 0.0, True
            x += it.width + 1.0
        assert placed < wirelength(row, weights)

    def test_single_part(self):
        items = place([Item("J1", 3, 6, {"A"}, is_connector=True)])
        assert items[0].placed


SES_TEMPLATE = """(session board.ses
  (routes
    (resolution um 10)
    (library_out
      (padstack "Via[0-1]_600:300_um"
        (shape (circle F.Cu 6000 0 0))
        (shape (circle B.Cu 6000 0 0))
      )
    )
    (network_out
      (net SIG
        (wire (path F.Cu 2500 1000000 -1000000 1010000 -1000000 1010000 -1005000))
        (via "Via[0-1]_600:300_um" 1010000 -1005000)
        (wire (path B.Cu 2500 1010000 -1005000 1020000 -1005000))
      )
    )
  )
)
"""


@needs_pcbnew
class TestSessionImport:
    def test_tracks_and_vias(self, tmp_path):
        import pcbnew
        from circuit_gen.layout.route import apply_session

        board = pcbnew.BOARD()
        net = pcbnew.NETINFO_ITEM(board, "SIG")
        board.Add(net)
        ses = tmp_path / "board.ses"
        ses.write_text(SES_TEMPLATE)

        tracks, vias = apply_session(board, ses)
        assert (tracks, vias) == (3, 1)

        items = list(board.GetTracks())
        via = next(t for t in items if isinstance(t, pcbnew.PCB_VIA))
        assert pcbnew.ToMM(via.GetPosition().x) == pytest.approx(101.0)
        assert pcbnew.ToMM(via.GetPosition().y) == pytest.approx(100.5)   # Specctra y is flipped
        assert pcbnew.ToMM(via.GetWidth()) == pytest.approx(0.6)
        assert pcbnew.ToMM(via.GetDrillValue()) == pytest.approx(0.3)
        first = next(t for t in items if not isinstance(t, pcbnew.PCB_VIA))
        assert pcbnew.ToMM(first.GetWidth()) == pytest.approx(0.25)
        assert first.GetNetname() == "SIG"
        assert {board.GetLayerName(t.GetLayer()) for t in items if not isinstance(t, pcbnew.PCB_VIA)} == {"F.Cu", "B.Cu"}


@needs_toolchain
@pytest.mark.parametrize("example", ["ldo_3v3.py", "blinker_555.py"])
def test_examples_build_end_to_end(example, tmp_path):
    from circuit_gen.pipeline import build

    report = build((EXAMPLES / example).read_text(), tmp_path)
    assert report["success"], report["stages"]
    assert report["stages"]["routing"]["unrouted"] == 0
    assert report["stages"]["drc"]["errors"] == 0
    assert (tmp_path / "gerbers" / "board-F_Cu.gtl").exists()
