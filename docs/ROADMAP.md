# Roadmap

The order follows the data flow and what can run without a GPU or Hugging Face access. Numbers in brackets are
the reference implementation's results, which are our targets.

## 1. Foundations ✅

`rmr/motion.py`, `rmr/plan.py`, `rmr/recipe.py`, `rmr/reach.py`, plus tests. CPU only, no network.

## 2. Generator (plan → 25 Hz motion)

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

## 3. Planner (text → recipe)

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

Before training, a frontier LLM can stand in as a zero-shot planner (with the system prompt above) to exercise the
whole pipeline.

## 4. Pipeline, rendering, visualizer

`prompts → recipes → plans → motions → videos`, rendered in MuJoCo (`reachy-mini` model files, `MUJOCO_GL=egl` when
headless), plus a three.js viewer that plays move JSON.

## 5. Serving (optional)

A REST API (`/generate-dense`, `/generate-sparse`), FP8 weights, speculative decoding with a fine-tuned MTP head,
and batched generator passes replayed as CUDA graphs.
