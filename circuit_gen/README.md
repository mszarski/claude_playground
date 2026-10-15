# circuit_gen

Description → circuit → routed PCB. The circuit design step uses a language
model: by default an open-weight model through Hugging Face Inference Providers
(`Qwen/Qwen3.8-27B`), or Claude with `--backend claude`. The layout and routing
steps use conventional tools (KiCad, FreeRouting).

```
description
   │  LLM (backends.py: hf | claude) + tools: search_symbols, get_symbol, search_footprints,
   │                                          check_circuit, simulate
   ▼
SKiDL circuit code  ◄── repair loop on Python / ERC / design-check errors
   │  runner.py (subprocess)
   ▼
circuit.json + netlist ──► checks.py (footprints exist & match pins, no dangling nets, ICs decoupled,
   │                                     inputs have a DC path, outputs not tied to rails)
   │                      ──► spice.py (ngspice with generic models vs. the board's test spec)
   ▼
layout/board.py: KiCad board, footprints, nets ──► layout/place.py: greedy placement
   │
   ▼
layout/route.py: DSN ──► FreeRouting (headless) ──► SES ──► tracks/vias ──► KiCad DRC ──► Gerbers + SVG
```

## Setup

```bash
circuit_gen/setup.sh          # KiCad 7, ngspice, FreeRouting 2.1, SKiDL (Ubuntu 24.04)
export HF_TOKEN=...           # default backend (open models via Hugging Face)
export ANTHROPIC_API_KEY=...  # only for --backend claude
```

## Usage

```bash
# Generate a circuit from a description, then lay it out and route it
python -m circuit_gen design "An LED that blinks at about 2 Hz using a 555 timer, powered from 9 V" -o runs/blinker

# Pick another open model, or Claude
python -m circuit_gen design "..." --model zai-org/GLM-5.3 -o runs/glm
python -m circuit_gen design "..." --backend claude -o runs/claude

# Lay out and route a hand-written SKiDL file (no model involved)
python -m circuit_gen build circuit_gen/examples/blinker_555.py -o runs/blinker

# Benchmark: how many descriptions make it all the way to DRC-clean Gerbers
# and pass their SPICE tests
python -m circuit_gen eval circuit_gen/benchmarks/boards.json -o runs/eval
python -m circuit_gen eval --examples -o runs/examples   # layout half only, no API calls
```

Defaults can also be set with `CIRCUIT_GEN_BACKEND`, `CIRCUIT_GEN_HF_MODEL`,
`CIRCUIT_GEN_HF_PROVIDER` and `CIRCUIT_GEN_CLAUDE_MODEL`. Any Hub model whose
providers support tool calling works with the `hf` backend.

Each run directory holds `circuit.py`, `circuit.json`, `circuit.net`,
`board.kicad_pcb` (open it in KiCad), `drc.rpt`, `gerbers/`, `board.svg` and
`report.json` (per-stage results). `design` runs also keep `transcript.jsonl`
and every attempted circuit under `attempts/`, which is the raw material for
a future fine-tuning dataset.

## How designs are checked

1. **Structural rules** (`checks.py`), on every attempt: footprints exist and
   match the symbol's pins; no net with a single pin; every IC power pin is
   decoupled; every net with an input pin has a DC path to a supply, ground,
   output or connector (a capacitor alone leaves it floating); no push-pull
   output is wired straight to a supply or ground.
2. **Simulation** (`spice.py`): benchmark boards carry test specs in
   `benchmarks/boards.json`, run in ngspice against the nets named in the
   description. For example, "the LED draws 8–12 mA at 5 V", "the gain at 1 kHz
   is 70–140", "the output stays at 4.75–5.25 V with a 50 Ω load". The model can
   also call `simulate` on its own design. Parts are mapped to generic models
   by library and symbol: R/C/L, diodes and LEDs (Vf by colour), op-amps
   (1 MHz, rail-limited), fixed linear regulators, MOSFETs and BJTs. MCUs,
   logic and timers have no model yet, so boards built around them have no
   functional tests.

## Status and known limits

- Both hand-written examples build end to end: routed with 0 unrouted nets and
  0 DRC errors (only silkscreen/library warnings).
- The `design` loop runs live with `Qwen/Qwen3.8-27B` via Hugging Face. The
  Claude backend is tested only with a scripted client.
- Placement is a simple greedy heuristic: parts are not rotated, decoupling
  caps are not forced next to their IC pins, and there is no ground pour. All
  nets are routed as 0.25 mm tracks, so it is not suitable for high-current or
  high-speed boards.
- Design checks are structural. They do not verify that the circuit does what
  the description asked; SPICE simulation (ngspice is installed) is the next step.
- Generated code runs in a subprocess on this machine. That is isolation of
  SKiDL state, not a security sandbox.

## Tests

```bash
pytest circuit_gen/tests/ -v
```

Tests that need SKiDL, KiCad or FreeRouting skip automatically when those are missing.
