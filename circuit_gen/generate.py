"""
Description -> SKiDL circuit code, using a language model with library and check tools.

The model looks up real symbols and footprints, writes SKiDL code, calls
`check_circuit` to run it through SKiDL's ERC and our design checks, and can
`simulate` it in ngspice to verify voltages, currents and gains. It repairs
the code until the checks pass. Every attempt is saved in the run directory;
over time those logs become training data for a fine-tuned generator.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import library
from .backends import ToolResult, make_backend
from .checks import check_circuit
from .runner import run_circuit_code

EXAMPLES_DIR = Path(__file__).parent / "examples"
FEW_SHOT_EXAMPLES = ["ldo_3v3.py", "blinker_555.py"]
MAX_TURNS = 40

SYSTEM_PROMPT = """\
You are an electronics design engineer. You turn a plain-language description \
of a circuit board into SKiDL (Python) code that defines its parts and \
connections. A later automatic stage places and routes the board, so your job \
is only the circuit: correct parts, correct footprints, correct connections.

How to work:
1. Work out the circuit: the blocks needed and their component values. Write \
   the key calculations as comments in the code.
2. Find every part in the installed KiCad libraries with search_symbols and \
   get_symbol. Use only symbols and pin names that these tools return; never \
   rely on memory for part names or pinouts.
3. Choose footprints with search_footprints. The footprint's pad numbers must \
   match the symbol's pin numbers. Prefer 0805 passives \
   (Resistor_SMD:R_0805_2012Metric, Capacitor_SMD:C_0805_2012Metric, \
   LED_SMD:LED_0805_2012Metric) and 2.54 mm pin headers for connectors, \
   unless the description asks for something else.
4. Call check_circuit with the complete code. Fix every error it reports and \
   check again.
5. Verify the behaviour that matters with simulate (DC voltages and currents, \
   AC gain): for example the LED current, a regulator's output under load, or \
   an amplifier's gain. Fix the design if the numbers are wrong. Simple generic \
   models are used, and parts without a model (MCUs, logic, timers) are left \
   out of the simulation.
6. You are done when check_circuit passes and simulation agrees with the \
   description; then reply with a short summary of the design and any \
   assumptions you made.

SKiDL conventions for this environment:
- `from skidl import *` and the library paths are already set up; do not \
  import skidl, call ERC(), or generate a netlist yourself.
- Create parts with Part("<Library>", "<Symbol>", value="...", footprint="<Lib>:<Footprint>").
- Name every net: vcc = Net("VCC"). Use GND for ground. When the description \
  names nets, use exactly those names: the board is tested through them.
- Connect with `+=` (net += part["PIN"]), pin names or numbers \
  (part[1, 2] += a, b), or chains for series parts (vcc & r & led["A K"] & gnd).
- Power and I/O enter and leave the board through connectors (Connector \
  library, e.g. Conn_01x02_Pin). Mark nets supplied through a connector with \
  `net.drive = POWER` so ERC knows they are powered.
- Every IC power pin needs a decoupling capacitor (typically 100n) to ground \
  on the same net; regulators need input and output capacitors per their \
  datasheet.
- Connect or deliberately leave unused pins; tie unused inputs to a defined level.
- Keep the design as simple as the description allows.

Here are complete examples of the expected style:
"""


def _system_prompt():
    parts = [SYSTEM_PROMPT]
    for name in FEW_SHOT_EXAMPLES:
        parts.append(f"\n# ---- {name} ----\n{(EXAMPLES_DIR / name).read_text()}")
    return "".join(parts)


TOOLS = [
    {
        "name": "search_symbols",
        "description": "Search the installed KiCad symbol libraries. Every query word must "
                       "appear in the symbol's name, keywords, description or library. Returns "
                       "library, name, description and default footprint for up to 20 matches.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "e.g. 'LDO 3.3V', 'NE555', 'USB C receptacle'"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_symbol",
        "description": "Pins (number, name, electrical type), default footprint and footprint "
                       "filters for one symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"lib": {"type": "string"}, "name": {"type": "string"}},
            "required": ["lib", "name"],
        },
    },
    {
        "name": "search_footprints",
        "description": "Search installed KiCad footprints. Words match as substrings of "
                       "'Library:Name'; words containing * or ? are globs against the name "
                       "(e.g. a symbol's footprint filter). Returns up to 25 'Library:Name' ids.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "check_circuit",
        "description": "Run complete SKiDL circuit code and report errors from Python, SKiDL's "
                       "electrical rules check, and the design checks (footprints exist and match "
                       "pins, no dangling nets, ICs decoupled). Always pass the full code.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Complete SKiDL code"}},
            "required": ["code"],
        },
    },
    {
        "name": "simulate",
        "description": "Run complete SKiDL code in ngspice and return one measurement. Voltage "
                       "sources are applied from named nets to ground; loads are resistors from a "
                       "net to ground, or to another net with {\"to\": NET, \"r\": OHMS}. measure "
                       "is v(NET), i(NET) (current the source on NET delivers into the circuit) "
                       "or gain(OUT_NET,IN_NET) (AC analysis, needs an \"ac\" source on IN_NET).",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Complete SKiDL code"},
                "analysis": {"type": "string", "enum": ["op", "ac"],
                             "description": "op = DC operating point (default), ac = small-signal"},
                "freq": {"type": "number", "description": "AC frequency in Hz (ac only)"},
                "sources": {"type": "object",
                            "description": "net name -> volts, or \"ac\" for the AC input, "
                                           "e.g. {\"VCC\": 5, \"IN\": \"ac\"}"},
                "loads": {"type": "object",
                          "description": "net name -> ohms to ground, or {\"to\": NET, \"r\": OHMS}"},
                "measure": {"type": "string", "description": "v(NET), i(NET) or gain(OUT,IN)"},
            },
            "required": ["code", "measure"],
        },
    },
]


@dataclass
class Attempt:
    code: str
    passed: bool
    feedback: str


@dataclass
class GenerationResult:
    description: str
    code: str | None
    passed: bool
    attempts: list = field(default_factory=list)
    summary: str = ""
    turns: int = 0
    error: str | None = None


def _format_symbols(entries):
    return json.dumps([{"lib": e.lib, "name": e.name, "description": e.description,
                        "footprint": e.footprint} for e in entries], indent=1)


class Toolbox:
    """Executes the model's tool calls and records circuit attempts."""

    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self.attempts = []
        self.sim_runs = 0

    def check_circuit(self, code):
        n = len(self.attempts) + 1
        run = run_circuit_code(code, self.workdir / "attempts" / f"{n:02d}")
        feedback = run.summary()
        passed = False
        if run.ok:
            report = check_circuit(run.circuit)
            feedback += "\n" + report.summary()
            passed = report.ok and not run.erc_errors
        feedback = ("PASSED. " if passed else "FAILED. ") + feedback
        self.attempts.append(Attempt(code=code, passed=passed, feedback=feedback))
        return feedback

    def simulate(self, args):
        """Run one model-requested measurement. Returns (content, is_error)."""
        from .spice import simulate
        self.sim_runs += 1
        run = run_circuit_code(args["code"], self.workdir / "sims" / f"{self.sim_runs:02d}")
        if not run.ok:
            return run.summary(), True
        test = {"name": args["measure"], "analysis": args.get("analysis", "op"),
                "freq": args.get("freq", 1000), "sources": args.get("sources") or {},
                "loads": args.get("loads") or {}, "measure": args["measure"],
                "expect": [-1e30, 1e30]}
        report = simulate(run.circuit, [test])
        result = report.results[0]
        if result.error:
            text = f"Simulation error: {result.error}"
        else:
            text = f"{args['measure']} = {result.value:.6g}"
        if report.unmodeled:
            text += "\nNot simulated (no model): " + ", ".join(report.unmodeled)
        return text, bool(result.error)

    def execute(self, name, args):
        """Run one tool call. Returns (content, is_error)."""
        try:
            if name == "search_symbols":
                hits = library.search_symbols(args["query"])
                return (_format_symbols(hits) if hits else "No symbols found."), False
            if name == "get_symbol":
                return json.dumps(library.get_symbol(args["lib"], args["name"]), indent=1), False
            if name == "search_footprints":
                hits = library.search_footprints(args["query"])
                return ("\n".join(hits) if hits else "No footprints found."), False
            if name == "check_circuit":
                return self.check_circuit(args["code"]), False
            if name == "simulate":
                return self.simulate(args)
            return f"Unknown tool {name}", True
        except KeyError as e:
            return f"Not found: {e}", True


def _validate_input(tool_name, data):
    """Check a tool call's parsed input against its schema's required string fields."""
    schema = next((t["input_schema"] for t in TOOLS if t["name"] == tool_name), None)
    if schema is None or not isinstance(data, dict):
        return False
    return all(isinstance(data.get(k), str) and data.get(k) for k in schema["required"])


NUDGE = ("check_circuit has not passed yet. Fix the problems it reported (or call it if you "
         "have not yet) and call check_circuit again with the complete code.")


def generate_circuit(description, workdir, backend=None, max_turns=MAX_TURNS, max_nudges=2):
    """Ask the model for a circuit matching `description`. Returns a GenerationResult."""
    backend = backend or make_backend()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tools = Toolbox(workdir)
    backend.start(_system_prompt(), f"Design this board:\n\n{description}", TOOLS)
    result = GenerationResult(description=description, code=None, passed=False)
    transcript = workdir / "transcript.jsonl"
    nudges = 0

    for turn_no in range(1, max_turns + 1):
        result.turns = turn_no
        turn = backend.step()
        with transcript.open("a") as f:
            f.write(json.dumps({"turn": turn_no, "backend": backend.name, "stop": turn.stop,
                                **turn.record}, default=str) + "\n")

        if turn.stop == "refusal":
            result.error = "model declined the request"
            break
        if turn.stop == "max_tokens":
            result.error = "response hit max_tokens"
            break

        if not turn.tool_calls:
            # Open models sometimes stop early or paste code as text; push them back once or twice.
            if not any(a.passed for a in tools.attempts) and nudges < max_nudges:
                nudges += 1
                backend.add_user_message(NUDGE)
                continue
            result.summary = turn.text
            break

        results = []
        for call in turn.tool_calls:
            if not _validate_input(call.name, call.input):
                content, is_error = json.dumps({"INVALID_JSON": call.raw[:2000]}), True
            else:
                content, is_error = tools.execute(call.name, call.input)
            results.append(ToolResult(id=call.id, content=content, is_error=is_error))
        backend.add_tool_results(results)
    else:
        result.error = f"stopped after {max_turns} turns"

    result.attempts = tools.attempts
    passing = [a for a in tools.attempts if a.passed]
    if passing:
        result.code, result.passed = passing[-1].code, True
    elif tools.attempts:
        result.code = tools.attempts[-1].code
    return result
