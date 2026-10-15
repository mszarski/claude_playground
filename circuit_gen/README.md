# circuit_gen

Description → circuit → routed PCB. The circuit design step uses Claude; the
layout and routing steps use conventional tools (KiCad, FreeRouting).

```
description
   │  Claude + tools: search_symbols, get_symbol, search_footprints, check_circuit
   ▼
SKiDL circuit code  ◄── repair loop on Python / ERC / design-check errors
   │  runner.py (subprocess)
   ▼
circuit.json + netlist ──► checks.py (footprints exist & match pins, no dangling nets, ICs decoupled)
   │
   ▼
layout/board.py: KiCad board, footprints, nets ──► layout/place.py: greedy placement
   │
   ▼
layout/route.py: DSN ──► FreeRouting (headless) ──► SES ──► tracks/vias ──► KiCad DRC ──► Gerbers + SVG
```

## Setup

```bash
circuit_gen/setup.sh          # KiCad 7, ngspice, FreeRouting 2.1, SKiDL (Ubuntu 24.04)
export ANTHROPIC_API_KEY=...  # only needed for `design` and benchmark `eval`
```

## Usage

```bash
# Generate a circuit from a description, then lay it out and route it
python -m circuit_gen design "An LED that blinks at about 2 Hz using a 555 timer, powered from 9 V" -o runs/blinker

# Lay out and route a hand-written SKiDL file (no model involved)
python -m circuit_gen build circuit_gen/examples/blinker_555.py -o runs/blinker

# Benchmark: how many descriptions make it all the way to DRC-clean Gerbers
python -m circuit_gen eval circuit_gen/benchmarks/boards.json -o runs/eval
python -m circuit_gen eval --examples -o runs/examples   # layout half only, no API calls
```

Each run directory holds `circuit.py`, `circuit.json`, `circuit.net`,
`board.kicad_pcb` (open it in KiCad), `drc.rpt`, `gerbers/`, `board.svg` and
`report.json` (per-stage results). `design` runs also keep `transcript.jsonl`
and every attempted circuit under `attempts/`, which is the raw material for
a future fine-tuning dataset.

## Status and known limits

- Both hand-written examples build end to end: routed with 0 unrouted nets and
  0 DRC errors (only silkscreen/library warnings).
- The `design` loop has been tested with a scripted client, not yet against the
  live API; the benchmark has not been run yet.
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
