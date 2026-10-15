"""
Functional checks: simulate a circuit in ngspice against a board's test spec.

The circuit (from the runner) is converted to a SPICE netlist using generic
models chosen from each part's KiCad library and symbol: resistors,
capacitors, inductors, diodes and LEDs, op-amps, fixed-output linear
regulators, MOSFETs and BJTs. Parts without a model (MCUs, logic, timers, ...)
are left out of the netlist and listed as unmodeled, so tests that depend on
them can fail; the report says which parts were missing.

A test spec looks like:

    {"name": "LED current",
     "analysis": "op",                      # "op" (DC operating point) or "ac"
     "freq": 1000,                          # ac only, Hz
     "sources": {"VCC": 5, "IN": "ac"},     # voltage sources from net to ground
     "loads": {"OUT": 1000, "LOAD": {"to": "V12", "r": 12}},
     "measure": "i(VCC)",                   # v(NET), i(NET) or gain(OUT,IN)
     "expect": [0.008, 0.012]}              # inclusive range

i(NET) is the current the source on NET delivers into the circuit.
Net names are the ones the description asked the circuit to use.
"""

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import library
from .checks import is_ground

NGSPICE = "ngspice"

_SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3,
       "K": 1e3, "M": 1e6, "G": 1e9, "R": 1.0, "r": 1.0}


def parse_value(text):
    """Parse component values such as '4.7k', '4k7', '100n', '10uF', '1R5', '330'."""
    t = str(text).strip().replace("Ω", "").replace("ohm", "").replace("Ohm", "")
    t = re.sub(r"(?<=[\d.fpnuµmkKMG])\s*[FH]$", "", t)       # drop unit letters F, H
    m = re.fullmatch(r"(\d+)([fpnuµmkKMGRr])(\d+)", t)       # 4k7, 1R5
    if m:
        return float(f"{m.group(1)}.{m.group(3)}") * _SI[m.group(2)]
    m = re.fullmatch(r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([fpnuµmkKMGRr]?)", t)
    if not m:
        raise ValueError(f"cannot parse value {text!r}")
    return float(m.group(1)) * (_SI[m.group(2)] if m.group(2) else 1.0)


MODELS = """
.subckt CG_OPAMP inp inn out vp vn
* Single-pole op-amp: 100 dB DC gain, 1 MHz GBW, output limited to the rails.
Gin 0 int inp inn 1e-3
Rint int 0 1e8
Cint int 0 1.59e-10
Bout outi 0 V = max(min(V(int), V(vp)-0.05), V(vn)+0.05)
Rout outi out 10
Iq vp vn 0.5m
.ends
.subckt CG_REG vi vo gnd vout=3.3 drop=1.0
* Fixed-output linear regulator: regulates to vout until the input drops.
Bo voi gnd V = max(min(vout, V(vi,gnd)-drop), 0)
Vsense voi vos 0
Ro vos vo 0.05
* The load current is drawn from the input, plus a small quiescent current.
Fin vi gnd Vsense 1
Gq vi gnd vi gnd 1e-5
.ends
.model CG_DIODE D(IS=2.5n N=1.75 RS=0.5 BV=100)
.model CG_SCHOTTKY D(IS=1u N=1.05 RS=0.05 BV=40)
.model CG_NMOS NMOS(LEVEL=1 VTO=1.5 KP=20 LAMBDA=0.01)
.model CG_PMOS PMOS(LEVEL=1 VTO=-1.5 KP=10 LAMBDA=0.01)
.model CG_NPN NPN(IS=1e-14 BF=200 VAF=100)
.model CG_PNP PNP(IS=1e-14 BF=150 VAF=100)
"""

# Tolerance used for a passive when the circuit does not give one (part.tolerance).
DEFAULT_TOLERANCE = {"R": 0.05, "C": 0.10, "L": 0.20}
# Tolerance corners: each sample puts every R/C/L at one end of its tolerance band.
CORNER_SAMPLES = 24

# LED forward voltage at 10 mA by colour.
_LED_VF = [("red", 1.9), ("orange", 2.0), ("yellow", 2.0), ("amber", 2.0), ("green", 2.2),
           ("blue", 3.0), ("white", 3.0), ("uv", 3.3), ("ir", 1.3)]


def _zener_voltage(*texts):
    """Breakdown voltage from a value or part name such as '5.1V', '5V1', 'BZV55C5V1'."""
    for text in texts:
        m = re.search(r"(\d+)V(\d+)", text or "")
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
        m = re.search(r"(\d+(?:\.\d+)?)\s*V\b", text or "")
        if m:
            return float(m.group(1))
    return None


def _led_model(name, value):
    vf = next((v for colour, v in _LED_VF if colour in value.lower()), 2.0)
    n, vt = 2.0, 0.02585
    is_ = 0.01 / math.exp(vf / (n * vt))
    return f".model {name} D(IS={is_:.3e} N={n} RS=2)"


@dataclass
class Netlist:
    lines: list = field(default_factory=list)
    models: list = field(default_factory=list)
    node: dict = field(default_factory=dict)       # net name -> SPICE node
    unmodeled: list = field(default_factory=list)
    tolerances: dict = field(default_factory=dict)  # SPICE element -> (nominal, relative tolerance)

    def node_for(self, net):
        if net is None:
            return None
        if is_ground(net):
            return "0"
        if net not in self.node:
            self.node[net] = f"n{len(self.node) + 1}"
        return self.node[net]


def _symbol_info(part):
    lib, name = part["symbol"].split(":", 1)
    try:
        return lib, name, library.get_symbol(lib, name)
    except (KeyError, FileNotFoundError):
        return lib, name, None


def _pins_by_name(part):
    return {p["name"]: p for p in part["pins"]}


def _add_part(nl, part):
    ref, value = part["ref"], part["value"]
    lib, name, info = _symbol_info(part)
    pins = part["pins"]
    by_name = _pins_by_name(part)
    nodes = [nl.node_for(p["net"]) for p in pins]
    desc = (info or {}).get("description", "") + " " + name

    if ref.startswith("J") or ref.startswith(("H", "TP", "MH")):
        return                                     # connectors, mounting holes, test points
    if any(n is None for n in nodes) and lib != "Amplifier_Operational":
        # Unconnected pins: model what we can with a floating node.
        nodes = [n or f"nc_{ref}_{i}" for i, n in enumerate(nodes)]

    if lib == "Device" and name.startswith(("R", "C", "L")) and len(pins) == 2 \
            and name[0] == ref[0]:
        nominal = parse_value(value)
        nl.lines.append(f"{name[0]}{ref} {nodes[0]} {nodes[1]} {nominal:.6g}")
        tol = part.get("tolerance")
        nl.tolerances[f"{name[0]}{ref}".lower()] = (
            nominal, DEFAULT_TOLERANCE[name[0]] if tol is None else float(tol))
        return
    if lib == "Device" and name.startswith("LED") and {"A", "K"} <= by_name.keys():
        model = f"CG_LED_{ref}"
        nl.models.append(_led_model(model, value))
        nl.lines.append(f"D{ref} {nl.node_for(by_name['A']['net'])} {nl.node_for(by_name['K']['net'])} {model}")
        return
    if {"A", "K"} <= by_name.keys() and len(pins) == 2 and (lib in ("Diode", "Device")):
        low = desc.lower()
        model = "CG_SCHOTTKY" if "schottky" in low else "CG_DIODE"
        if "zener" in low:
            bv = _zener_voltage(value, name, (info or {}).get("description", ""))
            if bv is None:
                nl.unmodeled.append(f"{ref} (zener voltage unknown: put it in the value, e.g. '5.1V')")
                return
            model = f"CG_Z_{ref}"
            nl.models.append(f".model {model} D(IS=1n N=1.5 RS=1 BV={bv:g} IBV=1m)")
        nl.lines.append(f"D{ref} {nl.node_for(by_name['A']['net'])} {nl.node_for(by_name['K']['net'])} {model}")
        return
    if lib == "Amplifier_Operational" and info:
        lib_pins = {p["num"]: p for p in info["pins"]}
        power = {p["name"]: p for p in pins if lib_pins.get(p["num"], {}).get("type") == "power_in"}
        vp = nl.node_for((power.get("V+") or {}).get("net"))
        vn = nl.node_for((power.get("V-") or {}).get("net"))
        if vp is None or vn is None:
            nl.unmodeled.append(f"{ref} (op-amp supply pins unconnected)")
            return
        units = {}
        for p in pins:
            u = lib_pins.get(p["num"], {}).get("unit", 0)
            if u and lib_pins[p["num"]]["type"] != "power_in":
                units.setdefault(u, {})[lib_pins[p["num"]]["type"] + p["name"]] = p["net"]
        for u, unit in sorted(units.items()):
            inp, inn = unit.get("input+"), unit.get("input-")
            out = next((v for k, v in unit.items() if k.startswith("output")), None)
            if inp is None and inn is None and out is None:
                continue                            # unit left entirely unconnected
            nodes = [nl.node_for(n) or f"nc_{ref}_{u}_{i}" for i, n in enumerate((inp, inn, out))]
            nl.lines.append(f"X{ref}_{u} {nodes[0]} {nodes[1]} {nodes[2]} {vp} {vn} CG_OPAMP")
        return
    if lib == "Regulator_Linear" and info:
        vin = by_name.get("VI") or by_name.get("IN") or by_name.get("VIN")
        vout = by_name.get("VO") or by_name.get("OUT") or by_name.get("VOUT")
        gnd = by_name.get("GND")
        m = re.search(r"Fixed Output (\d+(?:\.\d+)?)V", info.get("description", ""), re.I) \
            or re.search(r"-(\d+(?:\.\d+)?)$", name)
        if vin and vout and gnd and m:
            drop = 2.0 if re.match(r"L?M?78|7805|LM78", name) else 1.1 if "1117" in name else 0.3
            nl.lines.append(f"X{ref} {nl.node_for(vin['net'])} {nl.node_for(vout['net'])} "
                            f"{nl.node_for(gnd['net'])} CG_REG vout={m.group(1)} drop={drop}")
            return
    if lib.startswith("Transistor_FET") and {"G", "D", "S"} <= by_name.keys():
        model = "CG_PMOS" if "p-channel" in desc.lower() or "pmos" in desc.lower() else "CG_NMOS"
        d, g, s = (nl.node_for(by_name[k]["net"]) or f"nc_{ref}_{k}" for k in "DGS")
        nl.lines.append(f"M{ref} {d} {g} {s} {s} {model}")
        # Body diode: source to drain for N-channel, drain to source for P-channel.
        anode, cathode = (d, s) if model == "CG_PMOS" else (s, d)
        nl.lines.append(f"DB{ref} {anode} {cathode} CG_DIODE")
        return
    if lib.startswith("Transistor_BJT") and {"B", "C", "E"} <= by_name.keys():
        model = "CG_PNP" if "pnp" in desc.lower() else "CG_NPN"
        c, b, e = (nl.node_for(by_name[k]["net"]) or f"nc_{ref}_{k}" for k in "CBE")
        nl.lines.append(f"Q{ref} {c} {b} {e} {model}")
        return
    nl.unmodeled.append(f"{ref} ({part['symbol']})")


def to_netlist(circuit):
    nl = Netlist()
    for part in circuit["parts"]:
        try:
            _add_part(nl, part)
        except ValueError as e:
            nl.unmodeled.append(f"{part['ref']} ({e})")
    return nl


@dataclass
class TestResult:
    name: str
    passed: bool
    value: float | None = None
    expect: list | None = None
    error: str | None = None
    spread: list | None = None      # [min, max] over tolerance corners, when run

    def describe(self):
        if self.error:
            return f"{self.name}: ERROR {self.error}"
        lo, hi = self.expect
        text = f"{self.name}: {'pass' if self.passed else 'FAIL'} measured {self.value:.4g}"
        if self.spread:
            text += f" ({self.spread[0]:.4g}..{self.spread[1]:.4g} across tolerances)"
        return text + f", required {lo:g}..{hi:g}"


def _testbench(nl, test):
    """SPICE lines for one test, plus the expression to print. Raises KeyError for unknown nets."""
    def node(net):
        if is_ground(net):
            return "0"
        if net not in nl.node:
            raise KeyError(f"net '{net}' not found in the circuit")
        return nl.node[net]

    lines = []
    for net, spec in test.get("sources", {}).items():
        n = node(net)
        if spec == "ac":
            lines.append(f"V_{n} {n} 0 DC 0 AC 1")
        else:
            lines.append(f"V_{n} {n} 0 DC {float(spec)}")
    for i, (net, spec) in enumerate(test.get("loads", {}).items()):
        if isinstance(spec, dict):
            lines.append(f"R_load{i} {node(net)} {node(spec['to'])} {float(spec['r'])}")
        else:
            lines.append(f"R_load{i} {node(net)} 0 {float(spec)}")

    m = re.fullmatch(r"\s*(v|i|gain)\(\s*([^,)]+?)\s*(?:,\s*([^)]+?)\s*)?\)\s*", test["measure"])
    if not m:
        raise ValueError(f"bad measure expression {test['measure']!r}")
    kind, a, b = m.group(1), m.group(2), m.group(3)
    ac = test.get("analysis", "op") == "ac"
    if kind == "v":
        # In AC analysis voltages are complex; compare magnitudes.
        expr = f"mag(v({node(a)}))" if ac else f"v({node(a)})"
    elif kind == "i":
        n = node(a)
        if a not in test.get("sources", {}):
            raise ValueError(f"i({a}) needs a source on {a}")
        expr = f"mag(i(v_{n}))" if ac else f"-i(v_{n})"
    else:
        if b is None:
            raise ValueError("gain() needs two nets: gain(OUT,IN)")
        expr = f"mag(v({node(a)}))/mag(v({node(b)}))"

    if test.get("analysis", "op") == "ac":
        f = float(test.get("freq", 1000))
        analysis = f"ac lin 1 {f} {f}"
    else:
        analysis = "op"
    return lines, analysis, expr


def corner_variants(nl, samples=CORNER_SAMPLES, seed=0):
    """Component values for tolerance-corner runs: all-high, all-low, then random corners."""
    import random
    elems = sorted(e for e, (_, tol) in nl.tolerances.items() if tol > 0)
    if not elems:
        return []
    rng = random.Random(seed)
    signs = [{e: 1 for e in elems}, {e: -1 for e in elems}]
    signs += [{e: rng.choice((-1, 1)) for e in elems} for _ in range(max(0, samples - 2))]
    return [{e: nl.tolerances[e][0] * (1 + sign[e] * nl.tolerances[e][1]) for e in elems}
            for sign in signs]


def run_test(nl, test, timeout=60, corners=False):
    """Run one test at nominal values, and at tolerance corners if `corners` is set.

    The reported value is the nominal one; with corners, the test passes only if
    every corner is inside the expected range, and `spread` gives the min/max.
    """
    name = test.get("name", test["measure"])
    try:
        tb, analysis, expr = _testbench(nl, test)
    except (KeyError, ValueError) as e:
        return TestResult(name=name, passed=False, expect=test["expect"], error=str(e).strip("'\""))

    variants = [{}] + (corner_variants(nl) if corners else [])
    control = [".control"]
    for k, values in enumerate(variants):
        control += [f"alter {elem} = {value:.6g}" for elem, value in values.items()]
        control += [analysis, f"let cg_r{k} = {expr}", f"print cg_r{k}"]
    control += [".endc"]
    deck = "\n".join([f"* {name}", MODELS, *nl.models, *nl.lines, *tb,
                      ".option rshunt=1e10",           # tie nodes left floating by unmodeled parts
                      *control, ".end", ""])
    with tempfile.TemporaryDirectory() as tmp:
        cir = Path(tmp) / "test.cir"
        cir.write_text(deck)
        try:
            proc = subprocess.run([NGSPICE, "-b", str(cir)], capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return TestResult(name=name, passed=False, expect=test["expect"], error="ngspice timed out")
    out = proc.stdout + proc.stderr
    found = {int(k): float(v) for k, v in re.findall(r"cg_r(\d+)\s*=\s*([-+0-9.eE]+)", out)}
    if 0 not in found or len(found) < len(variants):
        err = next((l.strip() for l in out.splitlines() if "rror" in l), "no result from ngspice")
        return TestResult(name=name, passed=False, expect=test["expect"], error=err)
    lo, hi = test["expect"]
    values = [found[k] for k in range(len(variants))]
    spread = [min(values), max(values)] if corners else None
    passed = all(lo <= v <= hi for v in values)
    return TestResult(name=name, passed=passed, value=found[0], expect=[lo, hi], spread=spread)


@dataclass
class SimReport:
    results: list
    unmodeled: list

    @property
    def ok(self):
        return all(r.passed for r in self.results)

    def summary(self):
        lines = [r.describe() for r in self.results]
        if self.unmodeled:
            lines.append("Not simulated (no model): " + ", ".join(self.unmodeled))
        return "\n".join(lines)


def simulate(circuit, tests, corners=False):
    nl = to_netlist(circuit)
    return SimReport(results=[run_test(nl, t, corners=corners) for t in tests],
                     unmodeled=nl.unmodeled)


def requirement_tests(circuit):
    """The circuit's own require(...) declarations as test specs."""
    return [{"name": r.get("name") or r["measure"], "analysis": r.get("analysis", "op"),
             "freq": r.get("freq", 1000), "sources": r.get("sources") or {},
             "loads": r.get("loads") or {}, "measure": r["measure"],
             "expect": [r["min"], r["max"]]}
            for r in circuit.get("requirements", [])]
