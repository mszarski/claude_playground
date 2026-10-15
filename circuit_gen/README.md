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

# Spec-first: requirements are written from the description alone, then frozen
python -m circuit_gen design "..." --spec-first -o runs/x

# Benchmark: how many descriptions make it all the way to DRC-clean Gerbers
# and pass their SPICE tests (--resume reuses finished samples)
python -m circuit_gen eval circuit_gen/benchmarks/boards.json -o runs/eval --samples 5 --jobs 6
python -m circuit_gen eval --examples -o runs/examples   # layout half only, no API calls
python -m circuit_gen compare runs/eval_a runs/eval_b    # side-by-side pass rates
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
2. **Requirements in the design file**: the model states what the board must
   do next to the circuit, in the spirit of atopile's `assert`:

   ```python
   require("gain(AUDIO_OUT,MIC)", 75, 125, sources={"VCC": 5, "MIC": "ac"},
           loads={"AUDIO_OUT": 10000}, analysis="ac", freq=1000, name="gain ~100")
   r_f = Part("Device", "R", value="100k", footprint="...", tolerance=0.01)
   ```

   `check_circuit` simulates every requirement in ngspice at nominal values
   and at 24 tolerance corners (each R/C/L at one end of its band; defaults
   R 5%, C 10%, L 20%, or the part's `tolerance=`), and fails unless every
   corner is in range. So "passes" means "passes across component tolerances",
   not "passed once".
3. **Hidden benchmark tests** (`spice.py`): benchmark boards carry their own
   test specs in `benchmarks/boards.json`, run at nominal values against the
   nets named in the description. The model never sees these, so they check
   whether its own requirements were the right ones.

Parts are mapped to generic SPICE models by library and symbol: R/C/L,
diodes and LEDs (Vf by colour), op-amps (1 MHz, rail-limited), fixed linear
regulators (load current drawn from the input), MOSFETs and BJTs (with body
diodes), zeners, 555 timers and CMOS logic gates. MCUs and other complex ICs
have no model, so their behaviour is not simulated.

## Benchmark results (October 2026)

19 boards in `benchmarks/boards.json`: 12 simple ones and 7 harder analog ones
(Sallen-Key filter, inverting and difference amplifiers, common-emitter stage,
discrete current source, zener regulator, reverse-polarity protection). Each
harder board has a reference design in `benchmarks/reference/` that passes its
hidden tests, so the specs are achievable. All runs use the `hf` backend.

Open models, 2 samples per board (designer writes its own requirements):

| model | passed |
|---|---|
| deepseek-ai/DeepSeek-V4.1-Flash | 38/38 |
| zai-org/GLM-5.3 | 38/38 (but ~30x slower per board) |
| Qwen/Qwen3.8-27B | 30/38 |
| openai/gpt-oss-120b | 25/38 (3 false accepts) |

Qwen3.8-27B on the 7 harder boards, 5 samples each:

| mode | passed | false accepts |
|---|---|---|
| designer writes its own requirements | 23/35 | 8 |
| `--spec-first` (frozen requirements) | 31/35 | 0 |

Qwen3.8-27B with `--spec-first` on all 19 boards, 3 samples each: 88% of
samples pass, 0 false accepts (11 samples excluded for API errors).

DeepSeek-V4.1-Flash with `--spec-first`, 19 boards x 3 samples:

| spec-first version | passed | false accepts |
|---|---|---|
| frozen spec, designer can only fail | 72% | 2 (spec errors the designer gamed) |
| + designer can dispute a requirement | 40% | 0 (most sessions end in accurate disputes) |
| + spec fixes, one revision round | 79% | 1 (spec invented a net; now rejected) |

The spec writer's common mistakes were current sign conventions, loads
written as currents, DC checks of oscillators, unmodelled ICs, invented nets
and contradictory requirements. `i()` is now a magnitude, specs are
sanity-checked (load units, named nets), and a disputed requirement goes
back to the spec writer once.

A false accept is a design the model declared finished that fails the hidden
tests. Without a separate spec, the designer loosened or omitted the
requirements it was failing. The remaining spec-first failures are the model
running out of its 64k-token reply budget while reasoning.

## Training data

Two sources, both verified the same way (structure, ERC, and requirements
simulated across component tolerances) and exported as chat-format JSONL:
`sft_direct.jsonl` (description -> circuit code) and `sft_agentic.jsonl`
(a tool-using design session). Benchmark boards are never used.

- **Programmatic, no LLM** (`procgen.py`): 12 parametric families (LED
  resistor, divider, RC filters, linear regulators, inverting and
  non-inverting amps, Sallen-Key, 555 astable, zener regulator, MOSFET and
  BJT switches, buffered reference). Values come from the design equations
  rounded to E-series; sessions are built from real tool calls. 1,500
  generated, 1,491 verified, in about 15 minutes on 4 CPU cores:

  ```bash
  python -m circuit_gen procgen -n 1500 -o runs/procgen
  ```

- **LLM-designed** (`datagen.py`): `datagen` writes new descriptions (60 in
  `benchmarks/train_boards.json`), `eval --spec-first` designs them, and
  `export` keeps the passing sessions. More varied than the templates, but
  each session costs inference; the first run stopped at the account's
  spending limit after 22 verified sessions.

`train/sft.py` is a LoRA SFT script for Qwen3-8B on Hugging Face Jobs.

## Status and known limits

- Both hand-written examples build end to end: routed with 0 unrouted nets and
  0 DRC errors (only silkscreen/library warnings).
- The `design` loop runs live with `Qwen/Qwen3.8-27B` via Hugging Face. The
  Claude backend is tested only with a scripted client.
- Placement is a simple greedy heuristic: parts are not rotated, decoupling
  caps are not forced next to their IC pins, and there is no ground pour. All
  nets are routed as 0.25 mm tracks, so it is not suitable for high-current or
  high-speed boards.
- Simulation uses generic models, so it checks the circuit topology and
  values, not a specific part's datasheet limits (e.g. a MOSFET's real
  threshold voltage).
- Generated code runs in a subprocess on this machine. That is isolation of
  SKiDL state, not a security sandbox.

## Tests

```bash
pytest circuit_gen/tests/ -v
```

Tests that need SKiDL, KiCad or FreeRouting skip automatically when those are missing.
