"""
KiCad schematic (.kicad_sch) export through circuit-synth (optional).

The build goes from netlist straight to PCB, so there is no schematic for a
person to review. circuit-synth (MIT, https://github.com/circuit-synth/circuit-synth)
generates and places one from Python code close to SKiDL's. It is not a
dependency of circuit_gen: install it in its own environment and point
CIRCUIT_GEN_CS_PYTHON at that interpreter.

    uv venv ~/.cache/circuit_gen/cs-env
    VIRTUAL_ENV=~/.cache/circuit_gen/cs-env uv pip install circuit-synth==0.12.1
    export CIRCUIT_GEN_CS_PYTHON=~/.cache/circuit_gen/cs-env/bin/python
    python -m circuit_gen schematic runs/x

The schematic is read back (labels, power symbols, wires) and its netlist is
compared with the circuit's, so an export that loses or merges a connection
is reported instead of trusted. circuit-synth writes the KiCad 9 file format;
open it with KiCad 9.
"""

import json
import math
import os
import re
import subprocess
from pathlib import Path

from .sexp import child, children, head, parse

CS_PYTHON = os.environ.get("CIRCUIT_GEN_CS_PYTHON", "")


def to_circuit_synth(circuit, name):
    """circuit-synth source that rebuilds `circuit` (circuit.json) and writes a KiCad project."""
    func = re.sub(r"\W", "_", name)
    lines = ["from circuit_synth import Component, Net, circuit", "",
             f"@circuit(name={name!r})", f"def {func}():", "    nets = {}"]
    for net in sorted({p["net"] for part in circuit["parts"] for p in part["pins"] if p.get("net")}):
        lines.append(f"    nets[{net!r}] = Net({net!r})")
    for i, part in enumerate(circuit["parts"]):
        args = [f"symbol={part['symbol']!r}", f"ref={part['ref']!r}"]
        if part.get("value"):
            args.append(f"value={part['value']!r}")
        if part.get("footprint"):
            args.append(f"footprint={part['footprint']!r}")
        lines.append(f"    p{i} = Component({', '.join(args)})")
        for pin in part["pins"]:
            if pin.get("net"):
                lines.append(f"    p{i}[{pin['num']!r}] += nets[{pin['net']!r}]")
    lines += ["", "", "if __name__ == '__main__':",
              f"    {func}().generate_kicad_project(project_name={name!r}, generate_pcb=False)", ""]
    return "\n".join(lines)


def expected_nets(circuit):
    """{net: {(ref, pin)}} from circuit.json, for comparison with the schematic."""
    nets = {}
    for part in circuit["parts"]:
        for pin in part["pins"]:
            if pin.get("net"):
                nets.setdefault(pin["net"], set()).add((part["ref"], pin["num"]))
    return nets


def _load(path):
    return _parse_sheet(Path(path).read_text())


def _parse_sheet(text):
    tree = parse(text)
    if head(tree) != "kicad_sch":
        tree = next(x for x in tree if isinstance(x, list) and head(x) == "kicad_sch")
    return tree


def _lib_pins(tree):
    """{lib_id: {unit: {pin number: (x, y, angle)}}}; unit 0 holds pins shared by every unit."""
    libs = {}
    for sym in children(child(tree, "lib_symbols"), "symbol"):
        units = {}
        for sub in children(sym, "symbol"):  # "<name>_<unit>_<style>"
            unit = int(sub[1].rsplit("_", 2)[-2])
            for pin in children(sub, "pin"):
                at = child(pin, "at")
                angle = float(at[3]) if len(at) > 3 else 0.0
                units.setdefault(unit, {})[child(pin, "number")[1]] = (
                    float(at[1]), float(at[2]), angle)
        libs[sym[1]] = units
    return libs


def _pins(tree):
    """[(reference, pin number, (x, y), direction)] for every placed pin, power symbols included.

    Handles units (each placed unit has only its own pins plus shared ones),
    rotation and mirroring. `direction` is the angle the pin points away from
    its symbol, in sheet coordinates.
    """
    libs = _lib_pins(tree)
    out = []
    for sym in children(tree, "symbol"):
        at = child(sym, "at")
        x, y = float(at[1]), float(at[2])
        rot = float(at[3]) if len(at) > 3 else 0.0
        mirror = child(sym, "mirror")
        ref = next(p[2] for p in children(sym, "property") if p[1] == "Reference")
        unit = int(child(sym, "unit")[1]) if child(sym, "unit") else 1
        units = libs[child(sym, "lib_id")[1]]
        for num, (px, py, angle) in {**units.get(0, {}), **units.get(unit, {})}.items():
            if mirror and mirror[1] == "y":
                px, angle = -px, 180 - angle
            elif mirror and mirror[1] == "x":
                py, angle = -py, -angle
            # Symbol coordinates are y-up; the sheet is y-down.
            r = math.radians(rot)
            rx = px * math.cos(r) - py * math.sin(r)
            ry = px * math.sin(r) + py * math.cos(r)
            out.append((ref, num, (round(x + rx, 2), round(y - ry, 2)), (angle + rot) % 360))
    return out


def schematic_nets(path):
    """{net: {(ref, pin)}} read back from a flat .kicad_sch.

    Labels and power symbols name nets; pins, labels and wire ends at the same
    point are connected. Pins on no named net come back as "unnamed@(x, y)".
    """
    tree = _load(path)
    point = lambda x, y: (round(x, 2), round(y, 2))
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    names = {}
    for node in tree[1:]:
        if isinstance(node, list) and head(node) in ("label", "global_label", "hierarchical_label"):
            at = child(node, "at")
            names[point(float(at[1]), float(at[2]))] = node[1]
    for wire in children(tree, "wire"):
        pts = [point(float(p[1]), float(p[2])) for p in children(child(wire, "pts"), "xy")]
        for a, b in zip(pts, pts[1:]):
            parent[find(a)] = find(b)
    values = {next(p[2] for p in children(s, "property") if p[1] == "Reference"):
              next((p[2] for p in children(s, "property") if p[1] == "Value"), "")
              for s in children(tree, "symbol")}
    pins = []
    for ref, num, pt, _ in _pins(tree):
        find(pt)
        if ref.startswith("#PWR"):
            names[pt] = values[ref]
        else:
            pins.append((ref, num, pt))
    labelled = {}
    for pt, name in names.items():
        labelled.setdefault(find(pt), set()).add(name)
    nets = {}
    for ref, num, pt in pins:
        label = labelled.get(find(pt))
        net = " / ".join(sorted(label)) if label else f"unnamed@{pt}"
        nets.setdefault(net, set()).add((ref, num))
    return nets


def _top_level_forms(text):
    """(start, end) character spans of the forms directly inside the file's root form."""
    spans, depth, start, i, quoted = [], 0, None, 0, False
    while i < len(text):
        c = text[i]
        if quoted:
            if c == "\\":
                i += 1
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == "(":
            depth += 1
            if depth == 2:
                start = i
        elif c == ")":
            if depth == 2:
                spans.append((start, i + 1))
            depth -= 1
        i += 1
    return spans


LABEL = """\t(label {name}
\t\t(at {x} {y} {angle})
\t\t(effects
\t\t\t(font
\t\t\t\t(size 1.27 1.27)
\t\t\t)
\t\t\t(justify {justify} bottom)
\t\t)
\t\t(uuid "{uuid}")
\t)
"""


def _fix_instances(text, root_uuid):
    """Make each placed symbol's instance record match the symbol.

    circuit-synth 0.12 records every unit of a multi-unit part as unit 1 for the
    root sheet (and mixes "/" with "/<root uuid>" paths), so KiCad sees the power
    unit as a second copy of unit 1: its pins drop out of the netlist.
    """
    spans = _top_level_forms(text)
    for start, end in reversed(spans):
        form = text[start:end]
        if not form.startswith("(symbol") or "(lib_id" not in form or "(instances" not in form:
            continue
        own = re.search(r"^\t\t\(unit (\d+)\)", form, re.M)
        unit = own.group(1) if own else "1"
        cut = form.index("(instances")
        tail = re.sub(r'\(path "[^"]*"', f'(path "/{root_uuid}"', form[cut:])
        tail = re.sub(r"\(unit \d+\)", f"(unit {unit})", tail)
        text = text[:start] + form[:cut] + tail + text[end:]
    return text


GRID = 2.54
ORDER = "JUQDRCLY"  # connectors, ICs, transistors, diodes, resistors, capacitors, ...


def _layout(text):
    """Re-place every symbol on a grid so none overlap: connectors first, then ICs
    (all units of a part together), transistors, diodes and passives. Returns text.

    circuit-synth 0.12 stacks a part's units 12.7 mm apart, which overlaps for
    symbols taller than that, and can place other parts on top of them.
    """
    tree = _parse_sheet(text)
    boxes = {}  # (ref, unit) -> [min x, min y, max x, max y] relative to the symbol
    origin = {}
    for sym in children(tree, "symbol"):
        at = child(sym, "at")
        ref = next(p[2] for p in children(sym, "property") if p[1] == "Reference")
        unit = int(child(sym, "unit")[1]) if child(sym, "unit") else 1
        origin[(ref, unit)] = (float(at[1]), float(at[2]))
        boxes[(ref, unit)] = [0.0, 0.0, 0.0, 0.0]
    # Extent of each placed unit's own pins, relative to the symbol's origin.
    for sym in children(tree, "symbol"):
        ref = next(p[2] for p in children(sym, "property") if p[1] == "Reference")
        unit = int(child(sym, "unit")[1]) if child(sym, "unit") else 1
        ox, oy = origin[(ref, unit)]
        sub = ["kicad_sch", child(tree, "lib_symbols"), sym]
        for _, _, (x, y), _ in _pins(sub):
            box = boxes[(ref, unit)]
            box[0], box[1] = min(box[0], x - ox), min(box[1], y - oy)
            box[2], box[3] = max(box[2], x - ox), max(box[3], y - oy)

    def sort_key(key):
        ref, unit = key
        prefix = re.match(r"#?[A-Z]*", ref).group(0)
        number = int(re.sub(r"\D", "", ref) or 0)
        rank = ORDER.find(prefix[:1]) if prefix[:1] in ORDER else len(ORDER)
        return rank, prefix, number, unit

    pad_x, pad_y = 15.24, 10.16  # room for labels beside pins and fields above/below
    width_limit = 260.0
    x0 = y0 = 25.4
    x, y, row_h = x0, y0, 0.0
    moves = {}
    for key in sorted(boxes, key=sort_key):
        if key[0].startswith("#"):
            continue
        bx0, by0, bx1, by1 = boxes[key]
        w, h = (bx1 - bx0) + 2 * pad_x, (by1 - by0) + 2 * pad_y
        if x + w > width_limit and x > x0:
            x, y, row_h = x0, y + row_h, 0.0
        snap = lambda v: round(v / GRID) * GRID
        moves[key] = (snap(x + pad_x - bx0), snap(y + pad_y - by0))
        x += w
        row_h = max(row_h, h)
    if y + row_h > 190:  # does not fit on A4
        text = text.replace('(paper "A4")', '(paper "A3")', 1)

    out, last = [], 0
    for start, end in _top_level_forms(text):
        form = text[start:end]
        if not form.startswith("(symbol") or "(lib_id" not in form:
            continue
        sym = _parse_sheet(f"(kicad_sch {form})")
        sym = children(sym, "symbol")[0]
        ref = next(p[2] for p in children(sym, "property") if p[1] == "Reference")
        unit = int(child(sym, "unit")[1]) if child(sym, "unit") else 1
        if (ref, unit) not in moves:
            continue
        ox, oy = origin[(ref, unit)]
        nx, ny = moves[(ref, unit)]
        dx, dy = nx - ox, ny - oy

        def shift(m):
            return f"(at {round(float(m.group(1)) + dx, 2)} {round(float(m.group(2)) + dy, 2)}"
        out.append(text[last:start])
        out.append(re.sub(r"\(at (-?[\d.]+) (-?[\d.]+)", shift, form))
        last = end
    out.append(text[last:])
    return "".join(out)


def relabel(path, circuit):
    """Replace the schematic's net labels and power symbols with one local label per pin.

    circuit-synth 0.12 places the labels of a multi-unit part's pins relative
    to its first unit, which shorts nets on op-amps and logic ICs. Its symbol
    placement is kept; only the connections are redrawn here.
    """
    import uuid
    path = Path(path)
    text = path.read_text()
    tree = _load(path)
    drop = []
    for start, end in _top_level_forms(text):
        form = parse(text[start:end])
        if form and head(form) is None:  # parse wraps a single form in a list
            form = form[0]
        kind = head(form)
        if kind in ("label", "global_label", "hierarchical_label"):
            drop.append((start, end))
        elif kind == "symbol" and child(form, "lib_id") and any(
                p[1] == "Reference" and p[2].startswith("#PWR") for p in children(form, "property")):
            drop.append((start, end))
    nets = {(part["ref"], pin["num"]): pin["net"]
            for part in circuit["parts"] for pin in part["pins"] if pin.get("net")}
    for start, end in sorted(drop, reverse=True):
        # Also remove the whitespace up to the end of the line.
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end] == "\n":
            end += 1
        while start > 0 and text[start - 1] in " \t":
            start -= 1
        text = text[:start] + text[end:]
    text = _fix_instances(text, child(tree, "uuid")[1])
    text = _layout(text)
    labels = []
    for ref, num, (x, y), direction in _pins(_parse_sheet(text)):
        net = nets.get((ref, num))
        if not net:
            continue
        # Text runs away from the symbol, in the direction the pin points.
        angle = {0: 0, 90: 90, 180: 0, 270: 90}.get(round(direction) % 360, 0)
        justify = "right" if round(direction) % 360 in (180, 90) else "left"
        labels.append(LABEL.format(name=json.dumps(net), x=x, y=y, angle=angle,
                                   justify=justify, uuid=uuid.uuid4()))
    anchor = text.find("\t(sheet_instances")
    anchor = anchor if anchor >= 0 else text.rstrip().rfind(")")
    path.write_text(text[:anchor] + "".join(labels) + text[anchor:])


def compare(expected, actual):
    """Differences between two netlists, as readable lines (empty when they match)."""
    problems = []
    by_pin = {pin: net for net, pins in actual.items() for pin in pins}
    for net, pins in sorted(expected.items()):
        for ref, num in sorted(pins):
            got = by_pin.get((ref, num))
            if got != net:
                problems.append(f"{ref} pin {num}: expected {net}, schematic has {got or 'nothing'}")
    return problems


def kicad_nets(path):
    """{net: {(ref, pin)}} from KiCad's own netlist export of the schematic, or None.

    An independent check of schematic_nets. KiCad 9 reads the file as is; for
    KiCad 7 a copy is written in its older syntax (same content) first. Returns
    None when kicad-cli is missing or cannot load the file.
    """
    import shutil
    import tempfile
    from . import config
    if not shutil.which(config.KICAD_CLI):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        text = Path(path).read_text()
        old = text.replace("(version 20250114)", "(version 20230121)")
        for pattern in (r'\n\t*\(generator_version "[^"]*"\)', r"\n\t*\(embedded_fonts (yes|no)\)",
                        r"\n\t*\(exclude_from_sim (yes|no)\)", r"\n\t*\(fields_autoplaced no\)"):
            old = re.sub(pattern, "", old)
        old = old.replace("(hide yes)", "hide").replace("(fields_autoplaced yes)", "(fields_autoplaced)")
        for i, version in enumerate((text, old)):
            sch, net = tmp / f"b{i}.kicad_sch", tmp / f"b{i}.net"
            sch.write_text(version)
            subprocess.run([config.KICAD_CLI, "sch", "export", "netlist", "--format", "kicadsexpr",
                            str(sch), "-o", str(net)], capture_output=True, text=True, timeout=120)
            if net.exists():
                tree = parse(net.read_text())
                tree = tree if head(tree) == "export" else tree[0]
                return {child(n, "name")[1].lstrip("/"):
                        {(child(x, "ref")[1], child(x, "pin")[1]) for x in children(n, "node")}
                        for n in children(child(tree, "nets"), "net")}
    return None


def export_schematic(circuit, out_dir, name="board", python=None):
    """Write <out_dir>/<name>/<name>.kicad_sch from circuit.json data.

    Returns (path, problems): problems lists every pin whose net in the
    schematic differs from the circuit, read back here and, where kicad-cli
    can load the file, by KiCad itself (empty when the export is faithful).
    """
    python = python or CS_PYTHON
    if not python:
        raise RuntimeError("set CIRCUIT_GEN_CS_PYTHON to a Python with circuit-synth installed "
                           "(see circuit_gen/schematic.py)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / f"{name}_schematic.py"
    script.write_text(to_circuit_synth(circuit, name))
    run = subprocess.run([python, script.name], cwd=out_dir, capture_output=True, text=True,
                         timeout=600)
    found = sorted((out_dir / name).glob("*.kicad_sch"))
    if run.returncode != 0 or not found:
        tail = "\n".join((run.stderr or run.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"circuit-synth failed (exit {run.returncode}):\n{tail}")
    path = found[0]
    relabel(path, circuit)
    expected = expected_nets(circuit)
    problems = compare(expected, schematic_nets(path))
    from_kicad = kicad_nets(path)
    if from_kicad is not None:
        problems += [f"KiCad netlist: {p}" for p in compare(expected, from_kicad)]
    return path, problems


def export_run(run_dir, python=None):
    """Export the schematic for a build or eval sample directory (has circuit.json)."""
    run_dir = Path(run_dir)
    circuit = json.loads((run_dir / "circuit.json").read_text())
    # Eval samples live in <board>/sNN; name them after the board.
    name = (f"{run_dir.parent.name}_{run_dir.name}" if re.fullmatch(r"s\d+", run_dir.name)
            else run_dir.name)
    return export_schematic(circuit, run_dir / "schematic", name=re.sub(r"\W", "_", name),
                            python=python)
