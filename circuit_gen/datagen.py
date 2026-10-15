"""
Training data for a circuit-design model.

1. generate_descriptions(): an LLM writes new board descriptions, spread over
   topics, that are not in the benchmark (the benchmark stays held out).
2. Those boards are designed with the normal pipeline (`eval --spec-first`);
   only samples that pass every check, including the frozen requirements, and
   route to DRC-clean Gerbers are kept: rejection sampling with verification.
3. export_dataset() turns the kept runs into chat-format JSONL for SFT:
   - direct: description -> final circuit code (with its requirements)
   - agentic: the full tool-using conversation (Hugging Face backend runs)
"""

import json
import re
from pathlib import Path

from .backends import make_backend

BENCHMARK = Path(__file__).parent / "benchmarks" / "boards.json"

TOPICS = [
    "passive RC/LC filters (high-pass, band-pass, notch)",
    "op-amp amplifiers and buffers (summing, non-inverting, transimpedance, single and dual supply)",
    "op-amp active filters and integrators",
    "comparators and threshold detectors made from op-amps, with hysteresis",
    "BJT circuits (emitter follower, current mirror, LED switch, Darlington switch)",
    "MOSFET switches and load drivers (low-side, high-side with PNP or P-MOSFET)",
    "linear power (fixed regulators, zener references, voltage dividers for ADC inputs)",
    "555 timer oscillators and PWM generators",
    "CMOS logic circuits (Schmitt-trigger RC oscillators, gate logic, inverters)",
    "LED indicator boards and current-limited LED arrays",
    "sensor and interface breakouts with pull-ups, level setting and decoupling",
    "protection circuits (reverse polarity, clamping with zeners or diodes)",
]

GEN_PROMPT = """\
You write short design briefs for small circuit boards, like an engineer \
handing work to a colleague. Each brief:
- describes one board in one to three sentences, with concrete numbers \
(supply voltage, gain, frequency, current, cutoff, load);
- says how power and signals connect (pin headers or terminals);
- ends by naming the external nets, e.g. "Name the nets VCC, GND, IN and OUT.";
- can be built from common parts: resistors, capacitors, diodes, LEDs, \
op-amps, 555 timers, 74HC logic, BJTs, MOSFETs, linear regulators, connectors.
Vary difficulty from simple to moderately hard. Do not repeat any of the \
existing boards listed by the user.

Reply with only a JSON array of objects {"id": "snake_case_id", "description": "..."}."""


def _existing():
    return json.loads(BENCHMARK.read_text())


def generate_descriptions(n_per_topic=4, backend=None, topics=TOPICS):
    """New board descriptions, n_per_topic for each topic. Benchmark boards are excluded."""
    backend = backend or make_backend()
    existing = _existing()
    taken = {c["id"] for c in existing}
    listing = "\n".join(f"- {c['id']}: {c['description']}" for c in existing)
    boards = []
    for topic in topics:
        backend.start(GEN_PROMPT, f"Topic: {topic}\nWrite {n_per_topic} briefs.\n\n"
                                  f"Existing boards (do not repeat):\n{listing}", [])
        turn = backend.step()
        for item in _parse_json_array(turn.text):
            bid = re.sub(r"[^a-z0-9_]", "_", str(item.get("id", "")).lower()).strip("_")
            desc = str(item.get("description", "")).strip()
            if not bid or not desc or bid in taken:
                continue
            taken.add(bid)
            boards.append({"id": bid, "description": desc, "topic": topic})
    return boards


def _parse_json_array(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


DIRECT_SYSTEM = (
    "You are an electronics design engineer. Given a board description, write "
    "SKiDL (Python) code for the circuit: parts from the KiCad libraries with "
    "footprints, named nets, and require(...) lines stating the behaviour the "
    "board must have.")


def export_dataset(run_dirs, out_dir, include_benchmark=False):
    """Write sft_direct.jsonl and sft_agentic.jsonl from successful runs. Returns stats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    held_out = set() if include_benchmark else {c["id"] for c in _existing()}
    direct, agentic, skipped = [], [], {"failed": 0, "held_out": 0}
    for run_dir in run_dirs:
        for result_file in sorted(Path(run_dir).glob("*/s*/result.json")):
            sample = result_file.parent
            result = json.loads(result_file.read_text())
            if result["id"] in held_out:
                skipped["held_out"] += 1
                continue
            if not result["success"]:
                skipped["failed"] += 1
                continue
            gen = json.loads((sample / "generation.json").read_text())
            code = (sample / "circuit.py").read_text()
            meta = {"board": result["id"], "sample": result["sample"],
                    "model": gen.get("model"), "run": str(run_dir)}
            direct.append({"messages": [
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": gen["description"]},
                {"role": "assistant", "content": f"```python\n{code.strip()}\n```"},
            ], **meta})
            conv_file = sample / "conversation.json"
            if conv_file.exists():
                conv = json.loads(conv_file.read_text())
                if conv.get("backend") == "hf":
                    agentic.append({"messages": conv["messages"], "tools": conv["tools"], **meta})
    for name, rows in (("sft_direct.jsonl", direct), ("sft_agentic.jsonl", agentic)):
        with open(out_dir / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    stats = {"direct": len(direct), "agentic": len(agentic),
             "boards": len({r["board"] for r in direct}), **skipped}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
