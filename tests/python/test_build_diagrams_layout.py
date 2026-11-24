"""tools/build_diagrams.py: the hardware figure's heading stays on the page and the rows' labels do not overlap, for a
build whose face is small beside its projector's offset (the example of docs/examples/reference_design) and for the
shipped stages."""

import importlib.util
import pathlib

import pytest

from gpusonar import buildspec as bs

pytest.importorskip("matplotlib")
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs" / "examples" / "reference_design" / "example_static_2x24.yaml"


def tool():
    spec = importlib.util.spec_from_file_location("build_diagrams", ROOT / "tools" / "build_diagrams.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def drawn(diagrams, d, path, title, monkeypatch):
    """The hardware figure as drawn: (figure, renderer), kept open."""
    kept = []
    monkeypatch.setattr(diagrams.plt, "close", lambda fig=None: kept.append(fig))
    diagrams.draw_hardware(d, path, title)
    fig = kept[0]
    fig.canvas.draw()
    return fig, fig.canvas.get_renderer()


def test_spread_moves_only_labels_that_would_overlap():
    spread = tool().spread
    assert spread([-0.75, -0.2, 0.75], 0.1) == [-0.75, -0.2, 0.75]              # far enough apart: where they are
    assert spread([0.06, -0.06], 0.5) == pytest.approx([-0.25, 0.25])           # about their middle, lowest first
    out = spread([0.0, 0.01, 0.02, 1.0], 0.2)
    assert all(b - a >= 0.2 - 1e-12 for a, b in zip(out, out[1:])) and (out[0] + out[-1]) / 2 == pytest.approx(0.5)


def cases():
    diagrams = tool()
    yield "example", lambda: bs.derive(bs.load(EXAMPLE)), bs.load(EXAMPLE).get("title")
    for label, sonar, deploy, stem in diagrams.STAGES:
        yield stem, (lambda sonar=sonar, deploy=deploy: bs.describe_shipped(sonar, deploy)), label


@pytest.mark.parametrize("name,describe,title", list(cases()), ids=[c[0] for c in cases()])
def test_heading_is_on_the_page_and_row_labels_are_apart(name, describe, title, tmp_path, monkeypatch):
    diagrams = tool()
    d = describe()
    fig, renderer = drawn(diagrams, d, tmp_path / "hardware.png", title, monkeypatch)
    page = fig.bbox
    texts = list(fig.texts) + [ax.title for ax in fig.axes]
    heading = [t for t in texts if t.get_text().strip()]
    assert heading and (title or d["name"]) in heading[0].get_text().replace("\n", " ")
    for t in heading:
        box = t.get_window_extent(renderer)
        assert box.x0 >= page.x0 and box.x1 <= page.x1 and box.y1 <= page.y1 + 1.0, (name, t.get_text()[:40], box)
    face = fig.axes[0]
    assert heading[0].get_window_extent(renderer).y0 > face.get_window_extent(renderer).y1      # above the face, not on it
    labels = [t for t in face.texts if t.get_text().endswith(tuple(f"\n{n}" for n in d["per_row"])) and " m\n" in t.get_text()]
    assert len(labels) == d["rows"]
    assert sorted(t.get_text() for t in labels) == sorted(f"{h:+.2f} m\n{n}" for h, n in zip(d["heights_m"], d["per_row"]))   # no number changed
    boxes = [t.get_window_extent(renderer) for t in labels]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), name
    (tmp_path / "hardware.png").unlink()
