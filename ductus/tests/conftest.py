from pathlib import Path

import pytest

from ductus.loader import load_font

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "test_font" / "font.py"


@pytest.fixture
def font():
    return load_font(EXAMPLE)


@pytest.fixture
def write_font(tmp_path):
    """Write a small font module and load it."""
    def _w(src: str):
        p = tmp_path / "font.py"
        p.write_text("from ductus import *\n" + src)
        return load_font(p)
    return _w
