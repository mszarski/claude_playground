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
MAX_TOKEN_RETRIES = 2

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
4. State what the board must do as require(...) lines in the code: every \
   number the description gives (currents, voltages, gains, cutoffs) and the \
   behaviour that matters (a regulator's output under load, a switch's on and \
   off states). check_circuit simulates each requirement in ngspice at nominal \
   values and across component tolerances (resistors 5%, capacitors 10% \
   unless you set tolerance= on the part), and fails if any value falls \
   outside its range.
5. Call check_circuit with the complete code. Fix every error it reports and \
   check again. A failed requirement means the circuit is wrong: change the \
   circuit (or tighten part tolerances), never widen a requirement beyond what \
   the description asks for. Use simulate to investigate failures.
   Make each range as tight as the description implies, not tighter: \
   "about 100" allows roughly ±25%, while "3.3 V ±5%" means exactly that. \
   Check outputs under a realistic load (e.g. 10k on an audio or signal \
   output, the rated current on a supply rail), not only unloaded.
6. You are done only when check_circuit passes, with requirements that cover \
   the description; then reply with a short summary of the design and any \
   assumptions you made.

Requirements:
    require(measure, low, high, sources={...}, loads={...}, analysis="op", freq=1000, name="...")
- measure: v(NET), i(NET) (current the source on NET delivers into the \
  circuit; negative if the circuit sinks it), or gain(OUT,IN) with \
  analysis="ac" and an "ac" source on IN. In AC analysis every measure is \
  a magnitude: gain() is never negative. freq(NET) measures an oscillation \
  frequency with analysis="tran" and duration= at least four cycles long.
- sources: voltage sources from nets to ground, e.g. {"VCC": 5, "IN": "ac"}.
- loads: resistors from a net to ground, e.g. {"OUT": 10000}, or to another \
  net, e.g. {"LOAD_NEG": {"to": "V12", "r": 12}}.
- Generic models are used: R/C/L, diodes and LEDs, op-amps, fixed linear \
  regulators, MOSFETs, BJTs, 555 timers and CMOS logic gates. Parts without \
  a model (e.g. MCUs) are left out, so only require what the modelled parts \
  determine.

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
        "name": "report_spec_problem",
        "description": "Report that a fixed requirement is wrong (e.g. wrong units, physically "
                       "impossible, or it tests an IC that has no simulation model) and stop. "
                       "Use this instead of distorting the circuit to satisfy a requirement you "
                       "believe is incorrect. The session ends as a failure with your reason.",
        "input_schema": {
            "type": "object",
            "properties": {
                "requirement": {"type": "string", "description": "the requirement's name"},
                "reason": {"type": "string", "description": "why it is wrong"},
            },
            "required": ["requirement", "reason"],
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

    def __init__(self, workdir, suffix=""):
        self.workdir = Path(workdir)
        self.suffix = suffix        # frozen requirements appended to every circuit
        self.attempts = []
        self.sim_runs = 0

    def check_circuit(self, code):
        n = len(self.attempts) + 1
        run = run_circuit_code(code + self.suffix, self.workdir / "attempts" / f"{n:02d}")
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
        run = run_circuit_code(args["code"] + self.suffix,
                               self.workdir / "sims" / f"{self.sim_runs:02d}")
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


CUT_OFF_NOTE = ("Your last reply ran out of space before it finished. Think briefly, then make "
                "the next tool call.")

NUDGE = ("check_circuit has not passed yet. Fix the problems it reported (or call it if you "
         "have not yet) and call check_circuit again with the complete code.")


SPEC_NOTE = """

These requirements were written before the design and are fixed. They are \
appended to your code automatically every time it is checked or simulated, so \
do not copy them; check_circuit fails until all of them pass. You may add \
more require(...) lines of your own. If a requirement is itself wrong (wrong \
units, physically impossible, or it tests an IC that has no model), call \
report_spec_problem; never bend the circuit away from the description to \
satisfy a requirement you believe is wrong.

{spec}"""


def generate_circuit(description, workdir, backend=None, max_turns=MAX_TURNS, max_nudges=2,
                     spec_first=False):
    """Ask the model for a circuit matching `description`. Returns a GenerationResult.

    With `spec_first`, a separate model call first writes the requirements from
    the description alone (spec.py); they are frozen and appended to every
    circuit the designer submits.
    """
    from .spec import SpecError, write_spec

    backend = backend or make_backend()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    user = f"Design this board:\n\n{description}"
    suffix = ""
    if spec_first:
        try:
            _, suffix = write_spec(description, backend, workdir)
        except SpecError as e:
            return GenerationResult(description=description, code=None, passed=False,
                                    error=f"spec phase failed: {e}")
        user += SPEC_NOTE.format(spec=suffix.strip())
    tools = Toolbox(workdir, suffix=suffix)
    backend.start(_system_prompt(), user, TOOLS)
    result = GenerationResult(description=description, code=None, passed=False)
    transcript = workdir / "transcript.jsonl"
    nudges = 0
    max_token_retries = 0

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
            # Usually runaway reasoning; a fresh sample of the same turn tends to finish.
            if max_token_retries < MAX_TOKEN_RETRIES:
                max_token_retries += 1
                backend.add_user_message(CUT_OFF_NOTE)
                continue
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
        disputed = next((c for c in turn.tool_calls if c.name == "report_spec_problem"
                         and _validate_input(c.name, c.input)), None)
        if disputed:
            result.error = (f"spec disputed: {disputed.input['requirement']}: "
                            f"{disputed.input['reason']}")
            break
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
    if hasattr(backend, "export"):
        # The full design conversation, tool results included: training data.
        (workdir / "conversation.json").write_text(json.dumps(
            {"backend": backend.name, "model": backend.model, **backend.export()}, default=str))
    passing = [a for a in tools.attempts if a.passed]
    if passing:
        result.code, result.passed = passing[-1].code + suffix, True
    elif tools.attempts:
        result.code = tools.attempts[-1].code + suffix
    return result
