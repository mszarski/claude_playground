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

## Architecture, top to bottom

Four stages, each with one name used throughout:

| name | what it is | in code |
|---|---|---|
| **listener** | rules (no model) that move the head while you are speaking; its **style** is a set of numbers (nod size, timing, lean, sway) | `rmr/listen.py`, `STYLE` |
| **hearing** | Whisper for the words, plus two **tone models** that read the feeling in your voice | `rmr/voice.py` |
| **responder** | the fine-tuned 4B LLM that decides how Reachy answers and writes the recipe. It is *distilled* from a large teacher LLM, so the code and results call it the *student*; versions v3 … v8-dpo | `rmr/respond.py`, `Student` |
| **generator** | the motion model that turns a recipe's plan into 25 Hz motion | `rmr/generator/` |

```
 you speak ─────────────────────────────────────────────────────────────────────────────┐
   │ microphone (browser viewer or the robot), 25 Hz loudness                           │
   ▼                                                                                    │
 LISTENER (rules, no model, in its chosen style) ── leans in, nods at your pauses, perks antennas ──► head pose while you talk
   │ when you stop: the recording goes to the server
   ▼
 HEARING     Whisper-small ─► words          tone models (WavLM) ─► tone: feeling + arousal/valence/dominance
   │                                            └─ per-session calibration (mic and room levels)
   ▼
 RESPONDER   the student (Qwen3-4B fine-tune, 8-bit GGUF on llama.cpp), one call, streamed:
   │           {"feeling", "response", "recipe", "reading"}  + the last two lines of the conversation
   │           (typed prompts skip hearing: a zero-shot LLM planner writes the recipe)
   ▼
 RECIPE      go .3 e=-15 p=-8 z=12 E=6 | osc 1.2 z 5 .5 E=7 | hold 1.2 E=1.5 | ...
   │ expand + randomise (amplitude, timing)
   ▼
 PLAN        8 channels at 4 Hz: ears L/R, pitch, roll, yaw, height, body yaw, energy
   │
   ▼
 GENERATOR   21.8M flow-matching transformer: plan ─► 9-DoF motion at 25 Hz (8 Euler steps, ~50 ms)
   │
   ▼
 REACH       Stewart-platform IK on every frame; a pose with no solution, or needing a motor past
   │           its servo range, is pulled back toward the last good pose
   ▼
 MOVE        Reachy Mini recorded-move JSON ─► the 3D viewer, or the robot (rmr.robot)
```

**The listener** (`rmr/listen.py`, ported line for line to `visualizer/src/Listen.js`) is a small controller on the
loudness of your voice, so it reacts within a frame with no network. It tracks the room's noise floor, nods after a
pause that follows enough speech, leans in while you talk, perks the antennas when your voice lifts, and adds a slow
sway and an occasional glance. Its style (nod size, timing, lean, sway) is the one you picked in the duel ratings
("listener style" below).

**Hearing** (`rmr/voice.py`). Whisper turns speech into words. Two tone models (WavLM, trained on MSP-Podcast) read
the tone: a feeling (neutral, happy, sad, angry, surprised, ...) and three attributes, arousal, valence and dominance.
The same voice reads differently on different microphones, so each session keeps running means and shifts its
readings toward the levels the responder was trained on (`Calibration`).

**The responder** (`rmr/respond.py`). One model call turns what was heard into an answer: the feeling it reads, a
one-sentence response ("Calm attention. You hold steady, ears relaxed, one slow nod"), the motion recipe, and a
short reading of the person. The server streams the answer, and as soon as the recipe is complete the motion is
generated and sent; the reading arrives after. The responder is a fine-tune of Qwen3-4B-Instruct (Apache-2.0), served
on the live Space as an 8-bit GGUF by llama.cpp (about 2 s to the recipe on an L4 GPU). For typed prompts, a
zero-shot open LLM (`rmr/planner/`) writes the recipe instead.

**Recipes and plans** (`rmr/recipe.py`, `rmr/plan.py`). A recipe is a tiny language an LLM can write reliably:
`go` (ease to a pose), `hold` and `osc` (a sinusoid on one channel), over ears, pitch, roll, yaw, height, body yaw
and an energy level. It expands into a plan: keyframes every 0.25 s of the posture (below 1 Hz) and the energy,
meaning how much fast organic detail to add. Each expansion is randomised a little, so the same recipe never plays
twice the same way.

**The generator** (`rmr/generator/`). A transformer with one token per 25 Hz frame (d=384, 8 layers, 6 heads). Each
token carries the noisy motion and the plan at that frame; the flow time enters every block through AdaLN. It is
trained with flow matching and samples in 8 steps with classifier-free guidance 1.5. It never sees text: it only
knows how Reachy Mini moves, which is why it can learn from every clip.

**Reach** (`rmr/reach.py`). The head sits on a 6-motor Stewart platform, so its reachable poses are coupled: a tilt
that works at one height fails at another. Every frame goes through the SDK's analytical IK, and any frame with no
solution, or that would need a motor beyond its servo range, is line-searched back toward the last good frame. The
browser viewer has the same IK (checked against the SDK's to 1e-9 rad) and holds the last good frame as a safety net.

**Around it.** `rmr/server.py` serves the viewer and the API (`/api/respond` for speech, `/api/generate` for
typed prompts, `/api/idle` for small idle moves when the room is quiet). `rmr/robot.py` runs the same loop on a
Reachy Mini: the robot's microphone, the listener on board, the models on the server. `deploy/live_space.py`
deploys everything to a private Hugging Face Space on an L4 GPU that sleeps when idle.

## How it was trained

Each model learns one thing, from data that suits it.

| model | learns | data | method | cost |
|---|---|---|---|---|
| generator v2 | how Reachy Mini moves | Pollen's 85 emotion clips + 19 dances (~9 min), 12 emotions held out | flow matching on the clips' own plans, simplified like recipes | 25-run sweep, $1.68 (A10G) |
| planner (optional) | text → recipe | 17,364 prompt → recipe rows written by frontier LLMs (the reference's teacher data) | LoRA on Qwen3.5-4B, 2 epochs | $5.79 (A100) |
| responder v8 | heard → feeling, response, recipe | 4,450 examples: LLM-written lines + real CREMA-D tone readings, answered by Kimi-K3 | LoRA SFT on Qwen3-4B-Instruct-2507 | under $1 (A10G) |
| responder v8-dpo (default) | your taste | 160 preference pairs from your ratings | DPO, LoRA r16, 3 epochs | under $1 |
| listener style | your taste | 25 side-by-side votes over 24 styles | Bradley-Terry fit | free |

**1. The generator: motion without text.** There are only about 9 minutes of real expressive motion for this robot,
far too little to learn language from, but enough to learn how it moves. Every clip is turned into its own plan
automatically (`rmr.plan.extract`: posture below 1 Hz plus the energy of the faster detail), mirrored and
time-stretched (552 training clips), and the generator learns to rebuild the full 25 Hz motion from the plan.
Held out, it rebuilds a clip that is identified among the 12 unseen emotions 91.7% of the time (the reference: 89%).
Version 2 fixed a deployment gap: real plans still carry traces of the fast motion, recipe plans are smooth, so the
first generator came out too calm on recipes. Training on recipe-like simplifications of each clip brought the
generated liveliness to real levels (0.96× vs 0.63×). [Results](docs/results/generator.md).

**2. The planner: language without motion.** Knowing what *heartbroken* looks like is world knowledge, so it comes
from LLMs. Frontier models wrote prompt → idea + recipe examples (the reference's teacher data), and a 4B model was
LoRA-tuned on them; held-out prompts and probe prompts are filtered out by embedding similarity so they stay a real
test. The fine-tuned 4B matches more real clips (28% top-1 vs 20%); a strong open model used zero-shot passes all
16 probe prompts and needs no hosting, so the live system uses it for typed prompts.
Grounding the planner in transcriptions of Pollen's real clips did not help (documented as a negative result).
[Zero-shot](docs/results/planner_zeroshot.md), [fine-tune](docs/results/planner_finetune.md).

**3. The responder: reading people, licence-clean.** The first versions learned from MELD (TV dialogue); the
default one uses no MELD at all:

- *Words*: about 4,200 lines written by an open LLM, six feelings, with random topics and speakers and styles that
  fit each feeling.
- *Tone*: each line is paired with the emotion model's reading of a real CREMA-D recording of the same feeling
  (actors reading set sentences; only the readings are used). 800 extra examples use CREMA-D's flat sentences as
  the words ("It's eleven o'clock."), so the feeling can only come from the tone: this is where the responder learns
  to use the voice. 18 of the 91 actors are held out for testing.
- *Answers*: a frontier open model (Kimi-K3, chosen in a bake-off of four) answers every example: feeling, response,
  recipe and reading. Answers whose feeling disagrees with the known label get one hinted retry, and recipes must
  pass a physical check (for sad or angry people the head moves slowly with little energy; for happy ones the ears
  come up and the motion is lively; otherwise it stays moderate).
- *Fine-tune*: LoRA on Qwen3-4B-Instruct-2507, so the 4B does in one call what took a hearing model plus a large
  LLM.

On the held-out tone test (flat sentences, so words alone score chance) it reads the feeling 42% of the time
(chance 20%), and 90% of its responses on fresh lines pass the physical check. [Results](docs/results/voice.md).

**4. Your preferences.** Automatic scores turned out not to predict what you prefer: an LLM judge and two video
judges (vision models watching the rendered motion) all sat at chance against your ratings. So your ratings decide.
On the rating page you picked the best and worst of four responses; those choices became 160 preference pairs, and
DPO tuned the responder on them (v8-dpo, the default). The automatic scores held while the motions shifted toward what
you preferred. The listener style was picked the same way: an adaptive page showed two styles side by side over
real meeting speech (AMI), and a Bradley-Terry fit of your votes chose fewer, bigger nods.

**5. Serving.** The responder is exported to GGUF and served by llama.cpp: the 8-bit build scores the same as full
precision and is about twice as fast; the 4-bit build was faster still but its motions fit less often on the tone
test, so it is not used.

## Status

| step | module | status |
|---|---|---|
| Move format ↔ 9-DoF arrays, mirror, time-stretch | `rmr/motion.py` | ✅ done, tested |
| Plans: extract from any clip, interpolate to frames | `rmr/plan.py` | ✅ done, tested |
| Recipe language: parse, validate, expand, randomised variants | `rmr/recipe.py` | ✅ done, tested on all teacher recipes |
| Reachability projection (Stewart-platform IK) | `rmr/reach.py` | ✅ done, tested |
| Generator: 21.8M flow-matching transformer | `rmr/generator/` | ✅ 91.7% top-1 on the 12 held-out emotions (reference 89%); v2 trained on recipe-like plans matches real motion's liveliness in deployment. See [results](docs/results/generator.md) |
| Planner, zero-shot stand-in: open-weight LLM via HF Inference Providers, probe suite | `rmr/planner/` | ✅ Kimi-K3 passes 16/16 probes; real clips 20% top-1 (reference zero-shot 22%). See [results](docs/results/planner_zeroshot.md) |
| Planner: LoRA fine-tune of Qwen3.5-4B on the teacher data (HF Jobs, $5.79) | `rmr/planner/` | ✅ probes 0.88 / 0.81, real clips 28% top-1 (reference 4B: 0.91 / 0.875, 32%). See [results](docs/results/planner_finetune.md) |
| Offline pipeline (text → reachable moves) | `rmr/pipeline.py` | ✅ done |
| Live server: type a prompt, watch it in the viewer (zero-shot planner + CPU generator) | `rmr/server.py`, `deploy/` | ✅ private Docker Space `mszarski/reachy-mini-motions-live`: L4 GPU, responder on llama.cpp (first motion ~2.5 s after you stop), sleeps when idle |
| Voice: hold to talk; Reachy responds to what you say and how you sound (open models, runs locally) | `rmr/voice.py`, `rmr/respond.py`, `rmr/converse.py` | ✅ reads people right 48% (4 classes, real dialogue), responds appropriately 66%; a distilled local 4B does it in one call (50%, 73%); a 1.7B starts moving in under 6 s on a CPU (streaming, GGUF for llama.cpp / Ollama); hands-free turn-taking. Default student v8-dpo is licence-clean (no MELD: written lines + real CREMA-D voice readings, a frontier open teacher, then preference-tuned on human ratings): uses tone (42% on a tone-only test, chance 25%), physically fitting responses 90% of the time, 72% appropriate on MELD with per-session calibration (on by default). Automatic judges, LLM or video, don't agree with human raters, so ratings decide. See [results](docs/results/voice.md) |
| Listening: while you talk, Reachy leans in, nods at your pauses and perks its antennas when your voice lifts (live in the viewer, no model) | `rmr/listen.py`, `visualizer/src/Listen.js` | ✅ done; the JS port matches Python exactly (tested). `python -m rmr.listen talk.wav --video out.mp4` |
| Learned listening head: a GRU trained on 25 h of real listeners (Seamless Interaction, CC-BY-NC; optional) | `rmr/listen_model.py`, `scripts/train_listener.py` | ✅ moves like real listeners (RMS, speed, nod rate); real listeners' nods are not pause-timed, the rules' are. See [results](docs/results/listening.md) |
| Listener style tuned by your own ratings: 10 styles on AMI meeting speech (CC-BY 4.0), side by side; Bradley-Terry fit picks the default | `rmr/listen.py` (`STYLE`), `scripts/listen_rating.py` | ✅ 25 votes over 24 styles: fewer, bigger nods (x06) won and is the default; the learned head came 15th. See [results](docs/results/listening.md) |
| Robot: listen, nod along and answer on a Reachy Mini; models stay on the server | `rmr/robot.py` | ✅ tested on the SDK's MuJoCo simulator; not yet on hardware |
| Human ratings: blind A/B page for responses; ratings on training clips become DPO pairs | `scripts/rating_set.py`, `deploy/rating_page.html`, `scripts/dpo_from_ratings.py` | ✅ two rounds rated; 160 pairs DPO-tuned the default responder (v8-dpo). Automatic judges sat at chance against the ratings. See [results](docs/results/voice.md) |
| MuJoCo renderer: videos, contact sheets, grids (official Reachy Mini model, SDK IK, 500 Hz physics) | `rmr/renderer/` | ✅ done, tested |
| Browser viewer (three.js): gallery of results, drop any move JSON | `visualizer/`, `rmr/viewer.py` | ✅ done; browser IK checked against the SDK's to 1e-9 rad |
| Serving (FP8, MTP speculative decoding, CUDA graphs) | | optional |

See [docs/ROADMAP.md](docs/ROADMAP.md) for the plan of each step.

## Setup

```bash
pip install -e ".[ik,dev]"   # core + Stewart-platform IK + pytest
pytest
```

Extras for later steps: `generator` (torch), `hf` (Hugging Face Hub and datasets), `planner` (transformers, PEFT, TRL),
`render` (MuJoCo, imageio, Pillow; plus `pip install --no-deps reachy-mini` for the robot's model files and, headless
on CPU, `apt install libosmesa6` with `MUJOCO_GL=osmesa`).

Text to motion, end to end (zero-shot planner through Hugging Face, so `HF_TOKEN` is needed; the trained
generator is in the private Hub repo `mszarski/reachy-motion-generator`):

```bash
python -m rmr.pipeline --prompt "sneezing. You build up and then sneeze." --out runs/one \
    --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt --render      # + videos/ and grid.mp4
python -m rmr.renderer sheet runs/one/motions/*.json --out runs/one/sheet.png  # contact sheet
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
