"""Load a font module (``font.py``) and collect its parameters."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

from .params import Param
from .parts import DuctusError, Font
from .pen import Pen

_counter = itertools.count()


def load_font(path) -> Font:
    path = Path(path).resolve()
    if not path.exists():
        raise DuctusError(f"font source not found: {path}")
    modname = f"_ductus_font_{next(_counter)}"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = vars(mod)
    font = g.get("font")
    if not isinstance(font, Font):
        fonts = [v for v in g.values() if isinstance(v, Font)]
        if len(fonts) != 1:
            raise DuctusError(f"{path}: define exactly one Font (preferably named `font`)")
        font = fonts[0]
    params: dict[str, Param] = {}
    for k, v in g.items():
        if isinstance(v, Param):
            if v.name is None:
                v.name = k
            params.setdefault(v.name, v)
    # params declared inline on pens (``Pen(angle=param(...))``)
    pens = [v for v in g.values() if isinstance(v, Pen)]
    if font.pen is None:
        if len(pens) != 1:
            raise DuctusError(f"{path}: set font.pen")
        font.pen = pens[0]
    for pen in {id(p): p for p in pens + [font.pen]}.values():
        for attr in ("angle", "width", "contrast"):
            v = getattr(pen, attr)
            if isinstance(v, Param):
                if v.name is None:
                    v.name = f"{pen.name or 'pen'}_{attr}"
                params.setdefault(v.name, v)
    font.params = params
    font.source_path = path
    return font
