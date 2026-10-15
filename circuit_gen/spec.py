"""
Spec-first design: write the requirements before the circuit.

A separate model call sees only the description and writes require(...)
lines. They are parsed without executing anything (only literal require
calls are accepted) and frozen: the design phase must meet them unchanged,
so the designer cannot loosen a requirement its circuit fails.
"""

import ast
import json

SPEC_PROMPT = """\
You write acceptance tests for a circuit board before anyone designs it. \
Given the description, write require(...) lines that a correct board must \
pass and an incorrect one would fail. You do not design the circuit.

Each line is checked in ngspice at nominal values and across component \
tolerances (resistors 5%, capacitors 10%), so set ranges the description \
implies, not tighter: "about 100" allows roughly +/-25%, while "3.3 V +/-5%" \
means exactly that.

    require(measure, low, high, sources={...}, loads={...}, analysis="op", freq=1000, name="...")

- measure: v(NET), i(NET) (current the voltage source on NET delivers into \
the circuit; negative if the circuit sinks it), gain(OUT,IN) with \
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


def parse_requirements(text):
    """Parse require(...) calls from model output. Raises SpecError on anything else."""
    if "```" in text:
        blocks = text.split("```")
        code = max((b for b in blocks[1::2]), key=lambda b: b.count("require("), default="")
        code = code.split("\n", 1)[1] if code.startswith(("python", "py")) else code
    else:
        code = text
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
        # Accept an unquoted measure, e.g. require(v(OUT), ...), as models often write it.
        if call.args and isinstance(call.args[0], ast.Call) and \
                isinstance(call.args[0].func, ast.Name) and call.args[0].func.id in ("v", "i", "gain"):
            call.args[0] = ast.Constant(ast.unparse(call.args[0]).replace(" ", "").replace("'", ""))
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
        except SpecError as e:
            last_error = e
            backend.add_user_message(f"{e}. Reply with only require(...) lines.")
            continue
        code = to_code(reqs)
        (workdir / "spec.py").write_text(code)
        return reqs, code
    raise last_error
