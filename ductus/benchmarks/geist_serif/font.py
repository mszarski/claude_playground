"""INTERNAL BENCHMARK ONLY -- never build or release this font.

The serif kit with parameter values tuned by `ductus fit` (saved in
fitted.yaml) to measure how well the language represents Geist Serif.
Fitted values derive from the reference, so this stays in benchmarks/.
"""
from pathlib import Path

import yaml

from ductus import Font
from ductus.kits import serif as kit

font = Font("Ductus Bench Serif", upm=1000)

fitted = yaml.safe_load((Path(__file__).parent / "fitted.yaml").read_text()) or {}
P = kit.params(**fitted)
kit.define(font, P, glyphs="onHeasgv", terminals=kit.bracketed_set(P))
