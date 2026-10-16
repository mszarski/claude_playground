"""
Spec-first design: write the requirements before the circuit.

A separate model call sees only the description and writes require(...)
lines. They are parsed without executing anything (only literal require
calls are accepted) and frozen: the design phase must meet them unchanged,
so the designer cannot loosen a requirement its circuit fails.
"""

import ast
import json
import re

SPEC_PROMPT = """\
You write acceptance tests for a circuit board before anyone designs it. \
Given the description, write require(...) lines that a correct board must \
pass and an incorrect one would fail. You do not design the circuit.

Each line is checked in ngspice at nominal values and across component \
tolerances (resistors 5%, capacitors 10%), so set ranges the description \
implies, not tighter: "about 100" allows roughly +/-25%, while "3.3 V +/-5%" \
means exactly that.

    require(measure, low, high, sources={...}, loads={...}, analysis="op", freq=1000, name="...")

- measure: v(NET), i(NET) (magnitude of the current through the voltage \
source on NET, always >= 0), gain(OUT,IN) with \
analysis="ac" and an "ac" source on IN, or freq(NET), an oscillation \
frequency in Hz, with analysis="tran" and duration= long enough for at least \
four cycles. In AC analysis every measure is a \
magnitude: gain() is never negative, so an inverting gain of -10 is 9 to 11.
- sources: voltage sources from nets to ground, e.g. {"VCC": 5, "IN": "ac"}. \
A net without a source is left floating (useful for "input left open" cases).
- loads: resistors from a net to ground, e.g. {"OUT": 10000}, or between \
nets, e.g. {"LOAD_NEG": {"to": "V12", "r": 12}}.
- Use only the net names the description gives; internal nets do not exist yet.
- Simulation uses generic models of passives, diodes, LEDs, op-amps, fixed \
linear regulators, MOSFETs, BJTs, 555 timers and CMOS logic gates. \
Behaviour that depends on an unmodelled part (e.g. an MCU) cannot be \
checked; skip it.
- Boards built around an IC without a model (MCUs, interface transceivers \
such as RS-485, CAN or USB PHYs, sensors, memories): specify only the \
circuitry around it (pull-ups, termination, dividers, LED currents, supply \
filtering), never the IC's own function, or the designer will be pushed to \
replace the IC with discrete parts.
- All values are in SI base units: amperes, volts, ohms, hertz, seconds \
(10 mA is 0.01, 4.7 kHz is 4700).

Make the set consistent:
- Every requirement must be satisfiable together with all the others by one \
circuit; two requirements with identical sources and loads cannot expect \
different values.
- A source fixes its net's voltage, so do not also require v() of that net, \
and remember that a source overrides whatever the circuit would do there.
- A DC (op) analysis shows one static state. For oscillators and blinkers, \
require freq(NET) with analysis="tran", not high and low v() levels.
- Loads are resistances in ohms: a 10 mA load at 5 V is 500.

Cover:
- every number the description gives, at each end of any stated range of \
conditions (e.g. both supply voltages of "9 V to 15 V");
- what must not happen: blocking, off states, an input left open;
- qualitative claims turned into numbers: "second-order", "small drop", \
"good rejection", "able to drive a 10k load";
- outputs under the load the description implies.

Reply with only the require(...) lines, one per call, in a ```python block."""

FIELDS = ("measure", "low", "high")
OPTIONAL = {"sources": {}, "loads": {}, "analysis": "op", "freq": 1000, "name": None,
            "duration": 5.0}


class SpecError(ValueError):
    pass


_UNQUOTED = re.compile(r"require\(\s*(v|i|gain|freq)\(\s*([^)]*?)\s*\)")


def _quote_measures(code):
    """Quote measures models write bare, e.g. require(v(+5V), ...) or require(freq('OUT'), ...)."""
    def fix(m):
        inner = re.sub(r"[\s'\"]", "", m.group(2))
        return f'require("{m.group(1)}({inner})"'
    return _UNQUOTED.sub(fix, code)


def _nets_in(req):
    nets = set(req.get("sources") or {})
    for net, load in (req.get("loads") or {}).items():
        nets.add(net)
        if isinstance(load, dict) and load.get("to"):
            nets.add(load["to"])
    m = re.match(r"\s*\w+\((.*)\)\s*$", str(req["measure"]))
    if m:
        nets.update(n.strip() for n in m.group(1).split(",") if n.strip())
    return nets


def named_nets(description):
    """Nets listed in the description's "Name the ... nets A, B and C." sentences, or None."""
    sentences = re.findall(r"[Nn]ame the[^.]*?(?:\.(?=\s|$)|$)", description)
    if not sentences:
        return None
    nets = set()
    for sentence in sentences:
        for token in re.findall(r"(?<![A-Za-z0-9_+-])[-+]?[A-Z0-9_][A-Za-z0-9_+/-]*", sentence):
            if re.search(r"[A-Z+]", token):
                nets.add(token.rstrip(",.;"))
    return nets


def check_nets(reqs, description):
    """Requirements may only use nets the description names (plus ground)."""
    named = named_nets(description)
    for r in reqs:
        for net in _nets_in(r):
            if net.upper() in ("GND", "0"):
                continue
            if named is not None:
                ok = net in named
            else:
                ok = re.search(rf"(?<![A-Za-z0-9_+-]){re.escape(net)}(?![A-Za-z0-9_])", description)
            if not ok:
                raise SpecError(f"net {net!r} is not named in the description; requirements may "
                                f"only use the nets the description names: {r.get('name') or r['measure']}")


def parse_requirements(text):
    """Parse require(...) calls from model output. Raises SpecError on anything else."""
    if "```" in text:
        blocks = text.split("```")
        code = max((b for b in blocks[1::2]), key=lambda b: b.count("require("), default="")
        code = code.split("\n", 1)[1] if code.startswith(("python", "py")) else code
    else:
        code = text
    code = _quote_measures(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SpecError(f"requirements are not valid Python: {e}") from e
    reqs = []
    for node in tree.body:
        call = node.value if isinstance(node, ast.Expr) else None
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "require"):
            raise SpecError(f"only require(...) calls are allowed, got: {ast.unparse(node)[:80]}")
        try:
            args = [ast.literal_eval(a) for a in call.args]
            kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        except ValueError as e:
            raise SpecError(f"require arguments must be literals: {ast.unparse(call)[:80]}") from e
        req = dict(OPTIONAL)
        req.update(zip(FIELDS, args))
        req.update(kwargs)
        missing = [f for f in FIELDS if f not in req]
        if missing:
            raise SpecError(f"require(...) missing {', '.join(missing)}: {ast.unparse(call)[:80]}")
        if float(req["low"]) > float(req["high"]):
            raise SpecError(f"low above high in: {ast.unparse(call)[:80]}")
        for net, load in (req.get("loads") or {}).items():
            ohms = load.get("r") if isinstance(load, dict) else load
            if not isinstance(ohms, (int, float)) or ohms < 1:
                raise SpecError(f"loads are resistances in ohms, got {load!r} on {net}; for a "
                                f"10 mA load at 5 V use 500: {ast.unparse(call)[:80]}")
        if str(req["measure"]).startswith("i(") and float(req["high"]) < 0:
            raise SpecError("i() is a magnitude and never negative: "
                            f"{ast.unparse(call)[:80]}")
        if str(req["measure"]).startswith("gain(") and float(req["high"]) < 0:
            raise SpecError("gain() is a magnitude and never negative; for an inverting "
                            f"gain of -10 require 9 to 11: {ast.unparse(call)[:80]}")
        reqs.append(req)
    if not reqs:
        raise SpecError("no require(...) lines found")
    return reqs


def to_code(reqs):
    """Render requirements as require(...) lines to append to circuit code."""
    lines = ["", "# ---- frozen requirements (written before the design) ----"]
    for r in reqs:
        parts = [repr(r["measure"]), repr(float(r["low"])), repr(float(r["high"]))]
        for key in ("sources", "loads"):
            if r.get(key):
                parts.append(f"{key}={r[key]!r}")
        if r.get("analysis", "op") == "ac":
            parts += [f"analysis='ac'", f"freq={float(r.get('freq', 1000))!r}"]
        elif r.get("analysis") == "tran":
            parts += [f"analysis='tran'", f"duration={float(r.get('duration', 5.0))!r}"]
        if r.get("name"):
            parts.append(f"name={r['name']!r}")
        lines.append(f"require({', '.join(parts)})")
    return "\n".join(lines) + "\n"


def write_spec(description, backend, workdir, attempts=3):
    """Ask the model for requirements only. Returns (requirements, code) or raises SpecError."""
    backend.start(SPEC_PROMPT, f"Board description:\n\n{description}", [])
    last_error = None
    for _ in range(attempts):
        turn = backend.step()
        with open(workdir / "spec_transcript.jsonl", "a") as f:
            f.write(json.dumps({"stop": turn.stop, **turn.record}, default=str) + "\n")
        if turn.stop in ("refusal", "max_tokens"):
            last_error = SpecError(f"spec turn ended with {turn.stop}")
            continue
        try:
            reqs = parse_requirements(turn.text)
            check_nets(reqs, description)
        except SpecError as e:
            last_error = e
            backend.add_user_message(f"{e}. Reply with only require(...) lines.")
            continue
        code = to_code(reqs)
        (workdir / "spec.py").write_text(code)
        return reqs, code
    raise last_error


REVISE_NOTE = """The designer could not meet your requirements and objected:

Requirement: {requirement}
Objection: {reason}

Current requirements:
{spec}

If the objection is right, reply with the complete corrected list of \
require(...) lines. If it is wrong, reply with the same list unchanged. Do \
not weaken a requirement the description actually demands."""


def revise_spec(description, backend, workdir, spec_code, requirement, reason, attempts=3):
    """Let the spec writer answer a designer's objection. Returns (requirements, code)."""
    backend.start(SPEC_PROMPT, f"Board description:\n\n{description}\n\n" + REVISE_NOTE.format(
        requirement=requirement, reason=reason, spec=spec_code.strip()), [])
    last_error = None
    for _ in range(attempts):
        turn = backend.step()
        with open(workdir / "spec_transcript.jsonl", "a") as f:
            f.write(json.dumps({"stop": turn.stop, "revision": True, **turn.record}, default=str) + "\n")
        if turn.stop in ("refusal", "max_tokens"):
            last_error = SpecError(f"spec revision ended with {turn.stop}")
            continue
        try:
            reqs = parse_requirements(turn.text)
            check_nets(reqs, description)
        except SpecError as e:
            last_error = e
            backend.add_user_message(f"{e}. Reply with only require(...) lines.")
            continue
        code = to_code(reqs)
        (workdir / "spec_revised.py").write_text(code)
        return reqs, code
    raise last_error
