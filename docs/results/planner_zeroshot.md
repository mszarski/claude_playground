# Zero-shot planner results

A frontier open-weight LLM, prompted with the reference's teacher system prompt (`rmr.planner.prompt.SYSTEM`,
14 worked examples), stands in for the fine-tuned planner. It runs through Hugging Face Inference Providers with
`HF_TOKEN` (2026-10-05).

```bash
python -m rmr.planner.evaluate --model moonshotai/Kimi-K3 --samples 4 --out runs/planner_eval.json
python -m rmr.pipeline --prompt "startled. A sudden loud noise just made you jump." --out runs/one
```

## Scores

- **Probes:** the 16 out-of-distribution probes (`rmr.planner.probes`), 4 samples each at temperature 0.7.
- **Real clips:** the model's recipe for each of the 12 held-out emotions, 5 plan variants each, run through our
  generator (`checkpoints/generator.pt`, 100 flow steps). Each motion is identified among the 12 real clips
  (chance 8.3% top-1, mean rank 6.5).

| model | OOD-core probes | skill probes | real clips top-1 / mean rank | valid |
|---|---|---|---|---|
| zai-org/GLM-5.3 | 0.97 | 1.00 | 15% / 4.47 | 100% |
| deepseek-ai/DeepSeek-V4-Pro | 0.97 | 0.97 | 25% / 4.20 | 100% |
| **moonshotai/Kimi-K3** (default) | **1.00** | **1.00** | 20% / 4.65 | 100% |
| *reference: LLM-written plans, zero-shot* | | | *22% / 4.36* | |
| *reference: fine-tuned Qwen 27B / 4B / 0.8B (12 samples)* | *0.96 / 0.91 / 0.66* | *0.97 / 0.875 / 0.57* | *27% / 32% / 17%* | |

- All three models write valid, physically sensible recipes. The only probe misses were one of four startles
  (GLM, DeepSeek) and yawns (DeepSeek). Every sneeze uses anticipation correctly: up and back on the build-up,
  then a forward-down snap on the release.
- Real-clip scores match the reference's zero-shot number. With 12 emotions they move by ±7 points from noise,
  so the three models are not separable on them. As the reference notes, the remaining gap is mostly *taste*:
  the LLM's idea of how "disgusted" moves versus how Pollen's animator made it move. Given the true plan, the
  generator scores 91.7% ([generator results](generator.md)).
- Kimi-K3 is the default (`PLANNER_MODEL` overrides it): it is the only model that passed every probe.
- Cost: one evaluation (about 10 batched requests with ~3k-token prompts) costs a few cents of HF credits.
