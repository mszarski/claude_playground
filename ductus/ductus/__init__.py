"""Ductus: a parametric font design language and compiler."""
from .params import param, val, lazy_min as min_, lazy_max as max_
from .parts import Font, part, glyph, DuctusError
from .pen import Pen
from .terminals import serifs

__all__ = ["Font", "param", "part", "glyph", "Pen", "serifs", "val", "min_", "max_", "DuctusError"]
__version__ = "0.1.0"
