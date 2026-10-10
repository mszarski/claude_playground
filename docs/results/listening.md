# Listening, learned from real listeners

The listening behaviour in `rmr/listen.py` is hand-written: lean in, nod at pauses, perk the antennas when the voice
lifts. This is the data-driven alternative for the head: a small model that learned from 25 hours of real people
listening to each other.

## Data

[Seamless Interaction](https://huggingface.co/datasets/facebook/seamless-interaction) (Meta, **CC-BY-NC 4.0**):
4,000+ hours of two-person conversations, one item per participant with their own denoised microphone, voice
activity, and 30 Hz head rotation from the dataset's face tracker. Partners sit in different 1 GB tar shards, so
`scripts/extract_listening.py` picks 100 shards greedily to complete as many conversations as possible and reduces
them in an HF Job (8 vCPU, about $0.05) to 25 Hz loudness, voice activity and head rotation, discarding the video:

| | conversations | listeners (both directions) | listening (listener silent, tracked) |
|---|---|---|---|
| train | 399 | 790 | 25.5 h |
| dev | 54 | 103 | 3.3 h |

Axis 0 of the dataset's head rotation is pitch with + = head down (checked against facial keypoints: the
nose-to-shoulder distance shrinks as it grows, r = -0.55 to -0.70), 1 is yaw, 2 is roll.

## Model

`rmr/listen_model.py`: a causal GRU (96 units, 41k parameters). Each frame (25 Hz) it reads the speaker's loudness
relative to their own running speech level, their voice activity, onsets and offsets, and time since the last
change, plus its own previous motion. It outputs, for each axis, a distribution over the next change of the head
angle (31 bins over ±3° per frame) and samples it. Sampling, rather than predicting the average, keeps the variety
of real motion; a regression would average nods away. The target is the fast part of the motion (the angle minus its
0.3 Hz low-pass: nods, tilts, quick glances), so the model doesn't learn where someone happened to be looking. The
same causal 0.3 Hz high-pass is applied to what it samples, so small biases cannot add up to a drift.
Training: 12 epochs on CPU, 12 minutes; dev negative log-likelihood 3.48 → 0.84 nats per axis-frame (uniform: 3.43).

## Results (dev conversations, 198 minutes of listening)

`scripts/eval_listener.py`, sampling temperature 0.8:

| | RMS pitch / yaw / roll | pitch speed | nods per minute | nods within 1 s after a pause | motion after a pause vs during speech |
|---|---|---|---|---|---|
| real listeners | 1.83 / 2.29 / 1.53° | 6.5°/s | 31.6 | 17% (1.0× chance) | 1.05× |
| rules (`rmr/listen.py`) | 1.25 / 0.00 / 0.27° | 3.7°/s | 18.5 | 54% (3.3× chance) | 3.41× |
| **learned** | 1.44 / 1.59 / 1.26° | 5.3°/s | 29.1 | 17% (1.0× chance) | 1.02× |

(A nod: a downward pitch peak at least 1.5° prominent. Temperature 1.0 overshoots, at 2.69° RMS pitch and 49 nods a
minute, and 0.6 undershoots; 0.8 is the default.)

- The learned head moves like the people in the data: about the same amount, speed and number of small nods, on all
  three axes. The rules hardly turn or tilt.
- The surprise is timing. Real listeners' head motion is barely tied to the speaker's pauses (1.0× chance), and the
  learned model copies that. The rules nod at pauses 3.3× more often than chance. This measure only sees voice
  activity gaps, so it may miss backchannels at phrase boundaries inside speech. Still, it means the rules' "I heard
  you" nods are clearer and more regular than what people do. The learned head looks more natural; the rules signal
  more clearly. Which is better on a robot is a question for the rating page, not for this metric.

## Using it

The weights are CC-BY-NC (trained on non-commercial data), so they are optional and not in this repo: the
rule-based listener stays the default. They are in the private `mszarski/reachy-listening` repo
(`listener/v1/listener.json`).

```bash
python -m rmr.listen talk.wav --video learned.mp4 --listener-model listener.json          # render
python -m rmr.server ... --listener-model hf://mszarski/reachy-listening/listener/v1/listener.json   # viewer
python -m rmr.robot --server ... --listener-model listener.json                           # robot
```

With `--listener-model` the server serves the weights at `/api/listener`, and the viewer uses them for the head
while you talk. The antennas, the lean-in and the breathing still come from the rules, since humans don't have
antennas. `visualizer/src/ListenModel.js` is the browser port; `tests/test_listen.py` checks that it matches Python to
1e-6 with the same random draws.

## Tuning the style by rating, without the licensed data

The learned head above depends on non-commercial data. The alternative keeps the open, rule-based listener and lets a
person's ratings choose its style (`rmr.listen.STYLE`, the same in `Listen.js`, parity-tested): nod size, the pause
and amount of speech that earn a nod, double nods, how far it leans in, antenna perks, a gentle sway (three slow sines
per axis) and occasional glances aside (from a seeded generator that runs identically in Python and JS).

- **Speech**: four 16 s clips of people holding the floor in the AMI meeting corpus (CC BY 4.0), each speaker's own
  headset channel placed on the meeting's timeline, so their real pauses are kept.
- **Candidates**: 24 styles: the current one, a still baseline, seven hand-made variants, 14 spread evenly over the
  settings (Latin hypercube), and the learned model as a yardstick only.
- **Adaptive page** (`deploy/listen_duel_page.html`): two robots side by side over the same speech. The page fits a
  Bradley-Terry model to the votes so far and picks each next duel by double Thompson sampling (the likely best
  against its strongest challenger); every 20th duel repeats an earlier one with the sides swapped, to measure
  consistency.
- **Simulated rater** (`tests/duel_sim.mjs`, 24 styles, 40 votes, 300 runs): adaptive duels find the true best style
  27% of the time vs 15% for random pairs, a true top-3 style 58% vs 38%, and cut the average gap to the best style
  from 0.80 to 0.58 (log-strength). 40 votes cannot reliably single out the best of 24; they reliably narrow it down.
- **Fit** (`scripts/listen_rating.py fit`): Bradley-Terry strengths with bootstrap intervals, the consistency of the
  repeats, and a suggested next style: a quadratic fit of strength over the settings, searched only near the three
  strongest styles, to test in a short second round.
