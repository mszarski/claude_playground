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

# Requirements are written from the description alone, then frozen (the default);
# --no-spec-first lets the designer write its own
python -m circuit_gen design "..." --no-spec-first -o runs/x

# The spec writer can be another model, e.g. on Inference Providers while the
# designer runs on a local server (CIRCUIT_GEN_HF_BASE_URL)
python -m circuit_gen design "..." --spec-model deepseek-ai/DeepSeek-V4.1-Flash \
    --spec-provider deepinfra -o runs/x

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

## Schematics (optional)

The build goes from netlist straight to PCB. `schematic` adds a KiCad
schematic for review, using [circuit-synth](https://github.com/circuit-synth/circuit-synth)
(MIT) in its own environment so its dependencies stay out of this package:

```bash
uv venv ~/.cache/circuit_gen/cs-env
VIRTUAL_ENV=~/.cache/circuit_gen/cs-env uv pip install circuit-synth==0.12.1
export CIRCUIT_GEN_CS_PYTHON=~/.cache/circuit_gen/cs-env/bin/python
python -m circuit_gen schematic runs/x runs/eval/led_resistor/s01
```

circuit-synth supplies the symbols and the KiCad 9 file; `schematic.py` then
re-places the parts on a grid and draws one net label per pin, because
circuit-synth 0.12.1 shorts multi-unit parts (it places every unit's labels
relative to the first unit and records every unit as unit 1) and overlaps
their units. Each export is checked against the circuit twice, by reading the
labels back and by KiCad's own netlist export, and mismatches are reported.
All 19 benchmark references and the 32 passing Qwen3.8-27B benchmark designs
export with matching connections. Open the result with KiCad 9.

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

27 boards in `benchmarks/boards.json`: 12 simple ones, 7 harder analog ones
(Sallen-Key filter, inverting and difference amplifiers, common-emitter stage,
discrete current source, zener regulator, reverse-polarity protection), and 8
added on October 31 (three-op-amp instrumentation amplifier, MFB band-pass,
window comparator, two-stage 60 dB amplifier, XOR from NAND gates, current
mirror, high-side P-MOSFET switch, three-LED bar graph). All runs use the
`hf` backend.

Every board has a reference design in `benchmarks/reference/` and two or three
or more hidden tests that the reference passes, so every spec is achievable.
The tests are also checked from the other side: 18 plausible wrong designs (a
reference with one mistake, such as an LED without its resistor, a missing
mic bias resistor, an undersized coupling capacitor, comparator inputs
swapped or a MOSFET fitted backwards) must each fail them
(`tests/test_benchmark.py`).

Results before October 17 used a weaker version: 9 boards had references and
most boards had a single hidden test, so they may overstate pass rates.

**Current result.** Qwen3.8-27B, spec-first, all 27 boards, 2 samples each,
after the reasoning-budget fixes (November 1): **51/54 (94%)**, 26/27 boards
solved at least once, 25/27 every time, **0 false accepts**. No spec-writer
turn hit the token cap (median 2.4k tokens, previously 35-41k); 44 of 572
design turns did and were retried. Median sample time 48 s. Failures:
`mfb_bandpass` twice (the model keeps re-deriving the MFB topology: out of
turns, then out of tokens) and one `sallen_key_lowpass` out of turns.

On November 1 the `ce_amplifier`, `inverting_amp` and `mic_preamp`
descriptions were changed to state the band their hidden tests check
(100 Hz-10 kHz, 300 Hz-3 kHz); they said only 1 kHz, so a design that met
the description could fail. Earlier results used the old wording.

Earlier runs, before those fixes:

Qwen3.8-27B on the 8 boards added October 31, spec-first, 2 samples each:
13/16 (81%), 7/8 boards solved at least once, 0 false accepts. Both
`mfb_bandpass` samples failed when the spec writer ran out of its reply
budget; one `nand_xor` sample ran out of turns.

Qwen3.8-27B on the first 19 boards, spec-first (the default; the same model
writes the spec), 2 samples per board: **32/38 (84%)**, 18/19 boards solved at
least once, 14/19 every time, **0 false accepts**. The six failures: three ran
out of the 64k-token reply budget while designing (both Sallen-Key samples
and one blinker), two spec writers reasoned past the budget and wrote
nothing, and one spec used a net the description doesn't name (rejected).

Older results:

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
running out of its reply budget while reasoning (the provider caps replies at
40,960 tokens); see the current result above for the fix.

## Training data

Two sources, both verified the same way (structure, ERC, and requirements
simulated across component tolerances) and exported as chat-format JSONL:
`sft_direct.jsonl` (description -> circuit code) and `sft_agentic.jsonl`
(a tool-using design session). Benchmark boards are never used.

- **Programmatic, no LLM** (`procgen.py`): 20 parametric families (LED
  resistor, divider, RC filters, linear regulators, inverting and
  non-inverting amps, Sallen-Key, 555 astable, zener regulator, MOSFET and
  BJT switches, buffered reference, and since October 31 instrumentation
  amp, MFB band-pass, window comparator, two-stage amp, NAND-gate logic,
  current mirror, high-side switch, LED bar graph). Values come from the design equations
  rounded to E-series; sessions are built from real tool calls. With
  `--repair-fraction`, some sessions start from a planted bug (one R or C off
  by ten) whose real check output is followed by a diagnosis and the fixed
  code. 1,500 generated, 1,491 verified (618 repair sessions), about 1.5
  hours on 4 CPU cores:

  ```bash
  python -m circuit_gen procgen -n 1500 -o runs/procgen --repair-fraction 0.5
  ```

  `generate()` skips any board whose brief matches a benchmark board (up to
  the varied opener). Before October 31 it did not: 3 of the 1,491 sessions
  above duplicated `rc_lowpass` (twice) and `zener_regulator`, and were in the
  data the 8B was fine-tuned on. They are removed from `procgen_v3` in the
  dataset: 2,288 sessions over all 20 families, 941 of them repair sessions.

- **LLM-designed** (`datagen.py`): `datagen` writes new descriptions (167 in
  `benchmarks/train_boards.json`), `eval` designs them, and `export` keeps
  the passing sessions. DeepSeek-V4.1-Flash, 2 samples each: 144 verified
  sessions covering 96 boards.

`train/sft.py` is a LoRA SFT script for Hugging Face Jobs, and
`train/eval_job.sh` benchmarks an adapter against its base model in one job
(vLLM serves both on the job's GPU next to the normal eval).

### Fine-tuning Qwen3-8B: no gain

LoRA (r=32) on Qwen3-8B with the sets above, one epoch on an H200, did not
beat the base model. With DeepSeek writing the specs for every designer
(1 sample per board):

| designer | passed |
|---|---|
| Qwen3-8B, thinking on | 3/15 |
| Qwen3-8B, thinking off | 0/19 |
| fine-tuned, thinking off | 0/19 |
| fine-tuned, thinking on | 1/19 |

The fine-tuned model learned the session format but not debugging: it
narrated fixes and stopped without calling the tool, or repeated the same
call. The likely cause is the mix: about 1,340 short templated sessions
against about 120 varied ones. Things to know before trying again:

- Qwen3's thinking-off prompt adds an empty `<think></think>` that training
  sessions (no reasoning) never contain; the thinking-on prompt matches them.
- Specs should come from a strong model (`--spec-model`): the 8B wrote
  unsatisfiable ones (zero-width ranges, unquoted measures), now rejected.
- Much more varied LLM-designed data is the first thing to add; Qwen3.8-27B
  without fine-tuning is the better open designer for now.

## Status and known limits

- Both hand-written examples build end to end: routed with 0 unrouted nets and
  0 DRC errors (only silkscreen/library warnings).
- The `design` loop runs live with `Qwen/Qwen3.8-27B` via Hugging Face. The
  Claude backend is tested only with a scripted client.
- A model that repeats an identical tool call is told so in the result from
  the second time on; small models otherwise loop on one lookup until they
  run out of turns.
- Placement is a simple greedy heuristic: parts are not rotated, decoupling
  caps are not forced next to their IC pins, and there is no ground pour. All
  nets are routed as 0.25 mm tracks, so it is not suitable for high-current or
  high-speed boards.
- Simulation uses generic models, so it checks the circuit topology and
  values, not a specific part's datasheet limits (e.g. a MOSFET's real
  threshold voltage). The op-amp model's output current is not drawn from its
  supply pins, so `i(VCC)` misses current an op-amp delivers to a load.
- Generated code runs in a subprocess on this machine. That is isolation of
  SKiDL state, not a security sandbox.

## Tests

```bash
pytest circuit_gen/tests/ -v
```

Tests that need SKiDL, KiCad or FreeRouting skip automatically when those are missing.
