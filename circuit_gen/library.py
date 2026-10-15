"""
Search the KiCad symbol and footprint libraries.

These lookups back the model's tools: the model may only use parts that exist
in the installed libraries, so it has to find them here rather than recall
part names and pinouts from memory.
"""

import fnmatch
import json
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

from . import config, sexp

_SYMBOL_LINE = re.compile(r'^  \(symbol "([^"]+)"(?: \(extends "([^"]+)"\))?')
_PROPERTY_LINE = re.compile(r'^    \(property "([^"]+)" "((?:[^"\\]|\\.)*)"')
_INDEXED_PROPERTIES = {"Footprint": "footprint", "ki_keywords": "keywords",
                       "ki_description": "description", "ki_fp_filters": "fp_filters",
                       "Description": "description"}


@dataclass
class SymbolEntry:
    lib: str
    name: str
    description: str = ""
    keywords: str = ""
    footprint: str = ""
    fp_filters: str = ""
    extends: str = ""


def _index_library(path):
    entries = []
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _SYMBOL_LINE.match(line)
            if m:
                current = SymbolEntry(lib=path.stem, name=m.group(1), extends=m.group(2) or "")
                entries.append(current)
                continue
            if current is not None:
                p = _PROPERTY_LINE.match(line)
                if p and p.group(1) in _INDEXED_PROPERTIES:
                    setattr(current, _INDEXED_PROPERTIES[p.group(1)], p.group(2))
    return entries


@lru_cache(maxsize=1)
def symbol_index(symbol_dir=None):
    """All symbols in the library directory. Cached in memory and on disk."""
    symbol_dir = Path(symbol_dir or config.KICAD_SYMBOL_DIR)
    cache = Path.home() / ".cache" / "circuit_gen" / "symbol_index.json"
    files = sorted(symbol_dir.glob("*.kicad_sym"))
    stamp = [str(symbol_dir), len(files), max((f.stat().st_mtime for f in files), default=0)]
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("stamp") == stamp:
            return [SymbolEntry(**e) for e in data["entries"]]
    entries = []
    for f in files:
        entries.extend(_index_library(f))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"stamp": stamp, "entries": [asdict(e) for e in entries]}))
    return entries


def _score(text_fields, terms):
    score = 0
    for term in terms:
        hit = False
        for weight, text in text_fields:
            if term in text:
                score += weight
                hit = True
        if not hit:
            return 0
    return score


def search_symbols(query, limit=20):
    """Symbols whose name, description or keywords contain every query term."""
    terms = query.lower().split()
    scored = []
    for e in symbol_index():
        fields = [(5, e.name.lower()), (2, e.keywords.lower()), (1, e.description.lower()),
                  (1, e.lib.lower())]
        s = _score(fields, terms)
        if s:
            # Prefer exact name matches, then shorter (more generic) names.
            if e.name.lower() == query.lower():
                s += 100
            scored.append((-s, len(e.name), e))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in scored[:limit]]


@lru_cache(maxsize=32)
def _parsed_library(lib):
    path = Path(config.KICAD_SYMBOL_DIR) / f"{lib}.kicad_sym"
    return sexp.parse(path.read_text(encoding="utf-8"))[0]


def _find_symbol(lib, name):
    for s in sexp.children(_parsed_library(lib), "symbol"):
        if s[1] == name:
            return s
    return None


def _symbol_pins(lib, name):
    sym = _find_symbol(lib, name)
    if sym is None:
        raise KeyError(f"symbol {lib}:{name} not found")
    ext = sexp.child(sym, "extends")
    if ext:
        return _symbol_pins(lib, ext[1])
    pins = {}
    for unit in sexp.children(sym, "symbol"):
        # Sub-symbols are named <name>_<unit>_<style>; unit 0 is shared by all units.
        unit_no = int(unit[1].rsplit("_", 2)[-2])
        for pin in sexp.children(unit, "pin"):
            num = sexp.child(pin, "number")[1]
            pname = sexp.child(pin, "name")[1]
            pins[num] = {"num": num, "name": pname, "type": pin[1], "unit": unit_no}
    return sorted(pins.values(), key=lambda p: (len(p["num"]), p["num"]))


def get_symbol(lib, name):
    """Full description of one symbol: properties and pins."""
    entry = next((e for e in symbol_index() if e.lib == lib and e.name == name), None)
    if entry is None:
        raise KeyError(f"symbol {lib}:{name} not found")
    info = asdict(entry)
    info["pins"] = _symbol_pins(lib, name)
    return info


@lru_cache(maxsize=1)
def footprint_index():
    """All footprints as 'Library:Name' strings."""
    root = Path(config.KICAD_FOOTPRINT_DIR)
    names = []
    for lib in sorted(root.glob("*.pretty")):
        for mod in sorted(lib.glob("*.kicad_mod")):
            names.append(f"{lib.stem}:{mod.stem}")
    return names


def footprint_exists(fp):
    if ":" not in fp:
        return False
    lib, name = fp.split(":", 1)
    return (Path(config.KICAD_FOOTPRINT_DIR) / f"{lib}.pretty" / f"{name}.kicad_mod").exists()


def search_footprints(query, limit=25):
    """Footprints matching a query. Words are substrings; words with * or ? are globs."""
    terms = query.lower().split()
    out = []
    for fp in footprint_index():
        low = fp.lower()
        name = low.split(":", 1)[1]
        if all(fnmatch.fnmatch(name, t) if any(c in t for c in "*?") else t in low for t in terms):
            out.append(fp)
            if len(out) >= limit:
                break
    return out


def footprint_pads(fp):
    """Pad numbers defined in a footprint file (used to check pin/pad agreement)."""
    lib, name = fp.split(":", 1)
    path = Path(config.KICAD_FOOTPRINT_DIR) / f"{lib}.pretty" / f"{name}.kicad_mod"
    pads = re.findall(r'\(pad (?:"([^"]*)"|([^\s()"]+))', path.read_text(encoding="utf-8"))
    return {quoted or bare for quoted, bare in pads} - {""}
