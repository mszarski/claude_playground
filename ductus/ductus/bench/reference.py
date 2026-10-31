"""Reference fonts, read with fontTools -- used for measurement only."""
from __future__ import annotations

from pathlib import Path

from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont

from ..geometry import Contour


class Reference:
    def __init__(self, path):
        self.path = Path(path)
        self.font = TTFont(str(self.path))
        self.glyphset = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm
        os2 = self.font["OS/2"]
        self.metrics = {"xheight": os2.sxHeight, "capheight": os2.sCapHeight,
                        "ascender": os2.sTypoAscender, "descender": os2.sTypoDescender}
        self._cache: dict[str, tuple] = {}

    def name_for(self, unicode: int | None, fallback: str) -> str | None:
        if unicode is not None and unicode in self.cmap:
            return self.cmap[unicode]
        return fallback if fallback in self.glyphset else None

    def glyph(self, name: str) -> tuple[list[Contour], int]:
        """(contours as cubics, advance) of a reference glyph."""
        if name not in self._cache:
            pen = DecomposingRecordingPen(self.glyphset)
            self.glyphset[name].draw(pen)
            self._cache[name] = (Contour.from_ops(pen.value), self.glyphset[name].width)
        return self._cache[name]
