# Fine-tuned planner results

Qwen3.5-4B, LoRA-fine-tuned on the teacher data on Hugging Face Jobs (2026-10-05). Weights: the private repo
`mszarski/reachy-mini-planner-4b` (`merged/` for inference, `adapter/` for the LoRA alone).

```bash
python -m rmr.planner.sft --out runs/sft
python -m rmr.planner.hfjob submit --data runs/sft --max-usd 9
python -m rmr.planner.hfjob fetch --out runs/4b
python -m rmr.planner.evaluate --generations runs/4b/generations.json --out runs/4b/eval.json
python -m rmr.pipeline --prompt "a squirrel spotting a nut" --planner mszarski/reachy-mini-planner-4b --out runs/ft
```

## Data and training

- 17,364 training rows from the `claude`, `claude_events` (×3), `seed` and `astra_lively` sources, each
  "word. sentence." prompt also trained as the word alone and as the sentence alone. The blocklist dropped 51 rows
  and the embedding leak filter (Qwen3-Embedding-0.6B, cosine > 0.72 to an eval or probe prompt) dropped 45
  prompts, e.g. *"Crossing check. You watch both lanes then face the crossing."* (close to the "checking both
  ways" probe). The 39 val prompts are never trained on.
- LoRA rank 32 (alpha 32) on q/k/v/o and the MLPs. Qwen3.5's Gated DeltaNet projections stay frozen, as in the
  reference. AdamW 8-bit, lr 1e-4 cosine, batch 16 × 2 accumulation, 2 epochs (1,086 steps). Loss on the answer
  only, with prompts rendered without thinking, as at inference.
- One A100 80 GB: 6.5 s/step, 119 min of training. Val loss fell to 0.712 at epoch 0.96 (step 520), then held
  flat (0.717 at epoch 1.84). Step 520 is kept. The reference 4B reached 0.690.

| epoch | 0.07 | 0.37 | 0.66 | **0.96** | 1.25 | 1.55 | 1.84 |
|---|---|---|---|---|---|---|---|
| val loss | 1.333 | 0.835 | 0.751 | **0.712** | 0.722 | 0.719 | 0.717 |

- Cost: $5.79 of HF Jobs in total (2 h 19 min of A100 at $2.50/h): two smoke runs (6 and 8 min) and the full run
  (125 min, including install, merge, generation and upload).

## Evaluation

Probes: 12 samples per probe at temperature 0.7. Agreement and real clips: greedy answers. Real clips go through
our generator (5 plan variants per emotion, 100 flow steps).

| | OOD-core | skill | plan agreement | real clips top-1 / rank | valid | sneeze correct |
|---|---|---|---|---|---|---|
| **Qwen3.5-4B (ours)** | 0.88 | 0.81 | 0.65 | **28% / 4.03** | 100% | 19/24 |
| *reference Qwen3.5-4B* | *0.91* | *0.875* | *0.69* | *32% / 3.82* | *100%* | *21/24* |
| *reference Qwen3.8-27B* | *0.96* | *0.97* | *0.73* | *27% / 4.10* | *100%* | *24/24* |
| zero-shot Kimi-K3 (4 samples) | 1.00 | 1.00 | | 20% / 4.65 | 100% | 8/8 |

- Teacher-vs-itself agreement (the ceiling) is 0.92. Per descriptor, the planner agrees best on energy (0.87–0.92)
  and ear droop (0.82–0.85) and worst on yaw range (0.30).
- Within noise of the reference 4B (±0.1 on probes, ±7 points on real clips), slightly below on each metric.
- Weakest probe: *yawning widely* (2/12). The model rarely puts the head back before the droop, a multi-phase
  pattern the reference also names as hard for small models. *Bowing* (8/12), *sleepy toddler* and *drunk* (9/12
  each) follow. The sneeze release sometimes goes up instead of down (19/24 correct).
- The zero-shot frontier LLM beats it on the probes; the fine-tuned 4B is better on real clips (within noise). It
  runs locally in 0.4 s per answer, batched, on an A100 (plain transformers), and needs no API.
