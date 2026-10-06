# Generator results

First full training run on the real libraries (2026-10-05).

```bash
python -m rmr.generator train --hub --steps 5000 --ckpt checkpoints/generator.pt
python -m rmr.generator evaluate --hub --ckpt checkpoints/generator.pt
python -m rmr.generator sample --ckpt checkpoints/generator.pt --recipes examples/demo_recipes.json --out runs/samples
```

## Training

- Data: Pollen's emotions (85) and dances (19) from the Hub. 12 emotions held out, leaving 73 + 19 = 92 training
  clips, 552 after mirroring and time-stretching.
- 21.8M parameters, batch 16, 5,000 steps on 4 CPU cores: 0.82 s/step, 69 min in total (not the 3.3 h estimated).
- Held-out flow loss was lowest at step 2,750 (0.142) and rose slightly after that (0.152 at step 5,000, while
  training loss kept falling from 0.51 to 0.39). That is the expected overfitting on 9 minutes of motion. The
  step-2,750 checkpoint is kept.

| step | 250 | 500 | 1000 | 1500 | 2000 | 2500 | **2750** | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 12.96 | 1.27 | 0.99 | 0.83 | 0.68 | 0.60 | 0.55 | 0.51 | 0.44 | 0.39 |
| held-out | 0.352 | 0.256 | 0.195 | 0.164 | 0.158 | 0.151 | **0.142** | 0.147 | 0.148 | 0.152 |

## Held-out evaluation (12 emotions × 3 seeds)

A motion is generated from the true plan of each held-out clip and counts as identified when it is closer to its
own real clip than to the other 11 (chance 8.3%).

| sampler | top-1 | mean rank | pitch speed p95 / peak (°/s) | antenna speed p95 / peak (°/s) |
|---|---|---|---|---|
| real clips | | | 31 / 54 | 230 / 254 |
| **8 steps, cfg 1.5 (default sampler)** | **91.7%** | 1.33 | 30 / 49 | 228 / 334 |
| 100 steps, cfg 1.5 (`evaluate` default) | 91.7% | 1.36 | 30 / 50 | 234 / 353 |
| 8 steps, cfg 3.0 | 94.4% | 1.33 | 57 / 101 | 449 / 764 |
| 100 steps, cfg 3.0 | 88.9% | 1.42 | 55 / 97 | 457 / 769 |

The reference reports 89%. At the default guidance (1.5), typical speeds match real motion. Peak antenna speed is
about 30% higher than real: the occasional antenna flick comes out sharper. Guidance 3.0 doubles every speed, so
the motion becomes jittery.

## Demo recipes

`examples/demo_recipes.json`, 3 randomised variants each. Each generated motion's plan is re-extracted and compared
with the plan it was generated from (correlation / RMS error over the clip):

| prompt | antennas | pitch | height z | yaw | energy (asked → got, mean °) | frames projected |
|---|---|---|---|---|---|---|
| sneezing | 0.97–0.99 / 8–11° | 0.97–1.00 / 2–4° | 0.94–1.00 / 1–3 mm | | 3.3 → 2.8 | 1–22% |
| sobbing | 1.00 / 7–13° | 1.00 / 3–4° | 0.99–1.00 / 2–3 mm | | 4.8 → 0.9 | 0% |
| brave | 1.00 / 4–16° | 1.00 / 2° | 1.00 / 2–3 mm | | 1.8 → 1.1 | 0% |
| shivering | 0.98–0.99 / 8–13° | 0.99–1.00 / 1° | 0.99 / 1 mm | | 6.1 → 1.2 | 0% |
| conducting | 0.62–0.78 / 5–14° | 0.97–0.98 / 3–8° | 0.74–0.93 / 1–2 mm | 0.93–0.97 / 3–5° | 4.3 → 1.9 | 0–40% |

- **Posture follows the plan closely.** Pitch, height, yaw and antennas reach the targets within a few degrees or
  millimetres, including the sneeze's fast release.
- **Sustained high energy comes out too calm.** Sobbing (E=5–6 for 4 s) and shivering (E=6–7 for 3 s) ask for 5–6°
  of fast detail and get about 1°. Short bursts like the sneeze are fine. Such levels are rare in the training data
  (median energy in the emotion library is 1.3°, 90th percentile 4.8°). The 4 Hz low-pass and the number of Euler
  steps are not the cause (lifting either changes energy by at most 0.2°). Guidance 3.0 brings sobbing and
  conducting to the requested level, but doubles speeds on the held-out set (above), so the default stays at 1.5,
  as in the reference.
- Conducting's sweeping yaw (±20° at 1.4 s) runs into the head's reach, which is why up to 40% of its frames are
  projected.

## Fixing the held-energy undershoot (2026-10-06)

`energy_hold` (now part of `python -m rmr.generator evaluate`) asks for a held pose at constant energy, as recipes
write it, and reports generated / requested energy. The reference-trained generator scores 0.15-0.21.

What the undershoot is:

- On real plans the generator's energy is calibrated (held-out median ratio 1.07), so the problem is specific to
  recipe plans.
- Shortcut: freezing a real plan's posture curves (energy kept) drops its generated energy to 0.24×, while zeroing
  its energy (posture kept) only drops it to 0.45×. The generator reads liveliness mostly from traces of the fast
  motion left in the 1 Hz posture curves, which recipe plans do not have.
- Level vs change: even with posture frozen, setting a plan's energy to its own peak for the whole clip drops the
  response from 0.53× to 0.19×. In training, energy is always relative to calmer parts of a clip (bursts at sharp
  transitions; only 6% of high-energy frames have a still posture), so energy held over most of a clip reads as
  baseline.

What was tried (energy_hold at E = 2 / 4 / 6 / 8, held-out top-1, antenna speed p95; real clips 230 °/s):

| generator | energy_hold | top-1 | antenna p95 | notes |
|---|---|---|---|---|
| reference training | 0.21 / 0.15 / 0.16 / 0.17 | 91.7% | 228 | |
| guidance 3.0 (sampler) | | 94.4% | 449 | all speeds double |
| energy guidance (sampler, vs the plan with low energy) | ≈ 1 at E ≥ 8 | | | overshoots real plans by 35-58% |
| `--simplify 0.5` (smoother, sparser plan posture in training) | 0.25 / 0.17 / 0.28 / 0.31 | 91.7% | 263 | real plans 0.77× → 0.99×; shortcut halved |
| `--simplify 0.5 --tremor 0.5` (+ synthetic trembling copies) | 0.17 / 0.16 / 0.25 / 0.36 | 86.1% | 237 | |
| **reference training + gated energy completion** | **0.90 / 0.91 / 0.91 / 0.92** | **91.7%** | **239** | pitch p95 34 (real 31) |

Energy completion (`generate_batch(..., fill_energy=True)`, on by default in `rmr.pipeline` and `rmr.server`):
after sampling, wherever the generated detail is below half the requested energy, add 1.5-4 Hz noise on the head
rotations and antennas with RMS √(requested² − generated²), measured the way `plan.extract` measures energy.
Frames where the generator already delivers are untouched, which keeps real plans and held-out identification
unchanged. Demo recipes, energy asked → got: sobbing 4.8 → 0.9 before, 4.2 after; shivering 6.1 → 1.2 / 5.3;
conducting 4.3 → 1.9 / 3.5; sneezing 3.3 → 2.8 / 2.9.

The training options stay available (opt-in, default off) but are not used: with completion on top they add
nothing and raise antenna speeds.
