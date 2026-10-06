# Roadmap

The order follows the data flow and what can run without a GPU or Hugging Face access. Numbers in brackets are
the reference implementation's results, which are our targets.

## 1. Foundations ✅

`rmr/motion.py`, `rmr/plan.py`, `rmr/recipe.py`, `rmr/reach.py`, plus tests. CPU only, no network.

## 2. Generator (plan → 25 Hz motion) ✅

**Status.** Trained on the Hub libraries (85 emotions + 19 dances, 12 emotions held out) for 5,000 steps on CPU
(0.82 s/step); the best held-out checkpoint is from step 2,750. It identifies **91.7%** of held-out emotions [89%],
and head and antenna speeds match real motion. Sustained high-energy recipes (sobbing, shivering) come out too
calm (fixed since; see below). Details and the demo-recipe check: [results/generator.md](results/generator.md).

```bash
python -m rmr.generator train --hub --steps 5000 --ckpt checkpoints/generator.pt
python -m rmr.generator evaluate --hub --ckpt checkpoints/generator.pt
python -m rmr.generator sample --ckpt checkpoints/generator.pt --recipes examples/demo_recipes.json --out runs/samples
```

`--hub` downloads the libraries through `huggingface_hub` (`pip install -e ".[hf]"`). Checkpoints are gitignored:
publish them as a release or to the Hub rather than committing them.

The energy undershoot is fixed by gated energy completion after sampling (`fill_energy`, on in the pipeline and
server): held energy 0.15× → 0.91× of the request with identification and speeds unchanged. Two training-side
attempts (`--simplify`, `--tremor`) are documented in [results/generator.md](results/generator.md).

Needs Pollen's [emotions](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library) and
[dances](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library) libraries from Hugging Face
(network access to `huggingface.co`) and torch. It trains in minutes on one GPU, or more slowly on CPU or Apple silicon.

- **Data.** Load each move, resample to 25 Hz (`move_to_traj`), and hold out 12 emotions (listed in
  `data/teacher/eval_prompts.txt`). Augment by mirroring and time-stretching ×0.8/1/1.25. Pair each clip with its
  own `plan.extract`. Cap clips at 28.8 s.
- **Model.** One token per frame: noisy motion (9) + plan interpolated to the frame (8, `plan.frames`) + a
  has-plan flag, projected to 384 dimensions. 8 transformer blocks with full attention over frames, and the flow
  time injected through AdaLN in every block. About 21.8M parameters.
- **Training.** Flow-matching loss plus a velocity term on the implied clean motion (without it, motion comes out
  half as fast as real motion). 10% plan dropout enables classifier-free guidance. 5,000 steps; keep the best
  held-out checkpoint.
- **Sampling.** 8 Euler steps from duration × 25 frames of noise, guidance 1.5, then a 4 Hz low-pass. Finish with
  `Reach.project`.
- **Eval.** Generate from the true plan of each held-out emotion and check whether it is closer to its own real
  clip than to the other 11 [89%]. Compare head and antenna speeds against the real clips.

## 3. Planner (text → recipe) ✅

**Status.** The zero-shot stand-in works: `rmr/planner/` holds the reference's prompts, a client for any
OpenAI-compatible endpoint (default: Hugging Face Inference Providers with `HF_TOKEN`, model `moonshotai/Kimi-K3`),
the validate-and-fix writer, the 16 physical-check probes and an evaluation. Kimi-K3 passes 16/16 probes and gets
20% real-clip top-1 [zero-shot 22%; fine-tuned 27B 27%]. See [results/planner_zeroshot.md](results/planner_zeroshot.md).

**Fine-tuned.** Qwen3.5-4B, 2 epochs on one A100 ($5.79 in total): probes 0.88 OOD-core / 0.81 skill, plan
agreement 0.65, real clips 28% top-1 [4B: 0.91 / 0.875 / 0.69 / 32%]. Weakest probe: yawning (2/12). See
[results/planner_finetune.md](results/planner_finetune.md). Weights: private repo `mszarski/reachy-mini-planner-4b`.
Possible follow-ups: the 27B (better on multi-phase probes in the reference), or LoRA on the DeltaNet projections too.

How to reproduce it on Hugging Face Jobs (one A100, with a timeout derived from a dollar cap):

```bash
python -m rmr.planner.sft --out runs/sft                                  # 17,364 rows; leak filter drops 45 prompts
python -m rmr.planner.hfjob submit --data runs/sft --max-usd 9            # train + merge + generate on an A100
python -m rmr.planner.hfjob fetch --out runs/4b
python -m rmr.planner.evaluate --generations runs/4b/generations.json      # probes, agreement, real clips (CPU)
```

Differences from the reference: plain TRL + PEFT instead of Unsloth (same LoRA, lr, schedule and best-by-val-loss
selection), and prompts rendered with `enable_thinking=False` during training, as at inference. Qwen3.5's template
otherwise ends the training prompt inside an open `<think>` block and misaligns the answer-only loss mask. The job
installs `flash-linear-attention`; without it the Gated DeltaNet layers fall back to PyTorch and training is 1.5×
slower (7.5 s instead of 11.3 s per step of 32 rows on an A100).


Use `data/teacher/dataset.jsonl` with all sources except `astra`. Each row becomes a chat example: a compact system
prompt (~330 words: units, the recipe grammar, four motion rules, three examples), the user prompt, and a JSON answer
`{"idea", "recipe"}`. Also train each prompt as its bare word and as its sentence alone. Loss on the answer only.

- **Leak filter.** Drop training prompts with cosine > 0.72 to any eval prompt (Qwen3-Embedding-0.6B), plus any
  that mention sneeze, startle, drunk, toddler, stalking, heartbroken or ecstatic.
- **Fine-tune.** LoRA rank 32 on the attention and MLP projections, AdamW 8-bit, lr 1e-4, batch 32, 1.5–2 epochs.
  The smallest model, Qwen3.5-0.8B, took 24 min on an RTX 5090 in the reference.
- **Serving loop.** Sample, validate with `recipe.check`, and resample at temperature 0.7 until the answer is valid.
- **Eval.** 16 out-of-distribution probes, each with a physical check (does the sneeze release move the head
  *down*?), 12 samples each [27B 0.96 / 4B 0.91 / 0.8B 0.66].

Before training, a frontier LLM stands in as a zero-shot planner (`rmr.planner.write`, with the teacher prompt
`SYSTEM`) to exercise the whole pipeline.

## 4. Pipeline, rendering, visualizer ✅

**Status.** `python -m rmr.pipeline` runs prompts (or recipes) → plans → motions → reachable move JSON, and with
`--render` → MuJoCo videos and a grid. `rmr/renderer/` (`python -m rmr.renderer video|sheet|grid`) plays a move on
Pollen's official MJCF model like the SDK's MuJoCo backend: SDK IK per frame, 500 Hz physics, settle-and-ramp reset.
Checked against the commands: on real emotion clips the simulated head follows the commanded rotation to a 1.9°
median error once its ~120 ms servo lag is allowed for, and +20° pitch / +15 mm height come out as 20° / 15 mm in
the tests.

The browser viewer (`visualizer/`, see its README) plays a gallery built from pipeline outputs
(`python -m rmr.viewer`, `scripts/build_gallery.py`) or any dropped move file, with the SDK IK ported to JavaScript.

Two environment notes. Headless on CPU needs `MUJOCO_GL=osmesa` (about 0.6 s per 520×420 frame). With OSMesa,
`rmr.renderer` must be imported before `mujoco` when torch is also used in the process: it preloads Triton,
whose statically linked LLVM otherwise collides with Mesa's and segfaults torch training.

`prompts → recipes → plans → motions → videos`, rendered in MuJoCo (`reachy-mini` model files, `MUJOCO_GL=egl` when
headless), plus a three.js viewer that plays move JSON.

## 5. Serving (optional)

A REST API (`/generate-dense`, `/generate-sparse`), FP8 weights, speculative decoding with a fine-tuned MTP head,
and batched generator passes replayed as CUDA graphs.
