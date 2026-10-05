# Reachy Mini text-to-motion: a recreation

The goal is to recreate the system from Binh Pham's post
[*The best expressive harness for robots*](https://garden.binhph.am/articles/the-best-expressive-harness-for-robots).
You type what [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) should express (*"sneezing"*,
*"a cat stalking prey"*, *"heartbroken"*) and get a motion the robot can play: head, antennas and body at 25 Hz,
fitted to what the robot can physically reach.

Reference implementation: [pham-tuan-binh/reachy-motion-generator](https://github.com/pham-tuan-binh/reachy-motion-generator)
(Apache-2.0). We rebuild it piece by piece, with tests, rather than copying it wholesale.

## The idea

Reachy Mini has only about 9 minutes of real expressive motion: Pollen's 85 emotion clips plus 19 dances. That is
far too little to learn what *heartbroken* means. So the system splits the problem in two:

```
text ──planner (fine-tuned LLM)──► recipe ──expand──► plan (8 ch @ 2 Hz) ──generator (flow matching)──► motion (9 DoF @ 25 Hz) ──IK projection──► move
```

- **What to express** is world knowledge. An LLM *planner* writes a short motion **recipe**:
  `go .6 e=40 p=-6 z=4 E=1 | go 1 p=-16 z=10 e=70 E=3 | hold .6 E=6 | go .12 p=18 z=-8 e=100 E=10 | ...`
- **How Reachy Mini moves** (organic timing, overshoot, antenna flicks) is learned by a small *generator* from
  real clips. It never sees text, so every clip can be used for training, including the dances, through its
  automatically extracted plan.
- **Reachability.** Every frame is checked with the SDK's inverse kinematics. Unreachable poses are pulled back
  along a line search, so the robot never freezes.

## Status

| step | module | status |
|---|---|---|
| Move format ↔ 9-DoF arrays, mirror, time-stretch | `rmr/motion.py` | ✅ done, tested |
| Plans: extract from any clip, interpolate to frames | `rmr/plan.py` | ✅ done, tested |
| Recipe language: parse, validate, expand, randomised variants | `rmr/recipe.py` | ✅ done, tested on all teacher recipes |
| Reachability projection (Stewart-platform IK) | `rmr/reach.py` | ✅ done, tested |
| Generator: 21.8M flow-matching transformer | `rmr/generator/` | ✅ trained on Hub emotions + dances (69 min on 4 CPU cores): 91.7% top-1 on the 12 held-out emotions (reference 89%). See [results](docs/results/generator.md) |
| Planner, zero-shot stand-in: open-weight LLM via HF Inference Providers, probe suite | `rmr/planner/` | ✅ Kimi-K3 passes 16/16 probes; real clips 20% top-1 (reference zero-shot 22%). See [results](docs/results/planner_zeroshot.md) |
| Planner: LoRA fine-tune of Qwen3.5-4B on the teacher data (HF Jobs, $5.79) | `rmr/planner/` | ✅ probes 0.88 / 0.81, real clips 28% top-1 (reference 4B: 0.91 / 0.875, 32%). See [results](docs/results/planner_finetune.md) |
| Offline pipeline (text → reachable moves) | `rmr/pipeline.py` | ✅ done |
| MuJoCo renderer, browser visualizer | | ⏳ |
| Serving (FP8, MTP speculative decoding, CUDA graphs) | | optional |

See [docs/ROADMAP.md](docs/ROADMAP.md) for the plan of each step.

## Setup

```bash
pip install -e ".[ik,dev]"   # core + Stewart-platform IK + pytest
pytest
```

Extras for later steps: `generator` (torch), `hf` (Hugging Face Hub and datasets), `planner` (transformers, PEFT, TRL).

Text to motion, end to end (zero-shot planner through Hugging Face, so `HF_TOKEN` is needed; the trained
generator is in the private Hub repo `mszarski/reachy-motion-generator`):

```bash
python -m rmr.pipeline --prompt "sneezing. You build up and then sneeze." --out runs/one \
    --ckpt hf://mszarski/reachy-motion-generator/generator.pt
```

```python
from rmr.recipe import variants
plans = variants("go 1 e=150 p=22 z=-16 E=5 | osc 3 z 4 .9 E=6 | hold 1 E=4", n=4, seed=0)   # sobbing
```

## Data

`data/teacher/` holds the reference project's teacher dataset: 10,872 prompt → idea + recipe rows written by
frontier LLMs, 39 held-out validation prompts, 287 seed recipes and the evaluation prompts. See
[data/teacher/README.md](data/teacher/README.md).

## License

Apache-2.0. See [NOTICE](NOTICE) for what comes from the reference implementation and from Pollen Robotics.
