# Responding to how people sound

Hold to talk in the viewer (or `python -m rmr.converse clip.wav`): speech → transcript and tone → an LLM decides how
Reachy responds → the motion pipeline. Every model is openly licensed and runs locally.

| stage | model | licence |
|---|---|---|
| speech to text | `openai/whisper-small` | Apache-2.0 |
| emotion from the voice | `3loi/SER-Odyssey-Baseline-WavLM-Categorical` + `...-Multi-Attributes` (Goncalves et al., Odyssey 2024; trained on MSP-Podcast) | MIT |
| deciding the response, and the zero-shot planner | Qwen3 instruct models, e.g. `Qwen/Qwen3-4B-Instruct-2507` (local), `Qwen/Qwen3-Next-80B-A3B-Instruct` (live Space) | Apache-2.0 |
| fine-tuned planner (optional) | `mszarski/reachy-mini-planner-4b` (Qwen3.5-4B + LoRA) | Apache-2.0 base |
| motion | `generator_v2.pt` (this repo, trained on Pollen's Apache-2.0 clips) | |

The MSP-Podcast models are rebuilt in `rmr/voice.py` and load their weights strictly, so no remote code runs.

## How often the voice reading is right

`scripts/eval_voice_emotion.py`: four emotions every candidate can name, 100 clips each (chance 25%), unweighted
accuracy. MELD is conversation from *Friends* (closest to what Reachy hears); CREMA-D is acted.

| model | licence | MELD (conversation) | CREMA-D (acted) |
|---|---|---|---|
| **MSP-Podcast WavLM** (used) | MIT | **46%** | 59% |
| SpeechBrain wav2vec2 IEMOCAP | Apache-2.0 | 43% | 53% |
| wav2vec2 trained on acted sets (Dpngtm) | MIT | 30% | 90% |

- The model trained on acted speech is near chance on conversation (it most likely saw CREMA-D in training); acted
  benchmarks say little about real use.
- Even the best model is wrong more often than right on natural speech. MSP's main confusion is neutral heard as happy;
  it catches sadness 30% of the time. So the responder is told that the voice reading is often wrong and that the
  words win when they disagree, and responses stay gentle.

## End to end (MELD test clips)

With `Qwen/Qwen3-Next-80B-A3B-Instruct` (HF Inference) as responder and zero-shot planner:

| clip (MELD label) | heard | voice reading | Reachy's response |
|---|---|---|---|
| sadness | "…It's really bad. The only thing there that isn't burned is an axe…" | happy 69% (wrong) | notices "devastating news with an oddly cheerful tone"; tilts slowly, ears drooping a little |
| anger | "Look, I fell asleep before I could take a shower. Now I don't have time." | angry 79% | a slow, calm forward tilt and one gentle nod |
| joy | "Well, that makes me feel so good." | happy 87% | perks up, antennas up, light bounce |
| neutral | "Yeah, I really like his glasses." | happy 81% (wrong) | a soft perk-up |

About 6-7 s per response on a 4-core CPU (listening 3.5-4.5 s, planning 1-2 s, generation < 1 s).

Fully offline (`HF_HUB_OFFLINE=1`, `PLANNER_BASE_URL=local`, `Qwen/Qwen3-4B-Instruct-2507` in-process for both the
responder and the planner) works on the same clips, at 60-75 s per response on that CPU (mostly the 4B model writing
the recipe in plain transformers, unquantised). The small model judges worse: it handled anger well but read the
sad clip as cheerful and shared the joy. For local use, a quantised model in Ollama or llama.cpp is much faster, and a
larger instruct model (or the fine-tuned planner for the recipe) responds better.

## End-to-end evaluation (200 MELD test clips)

`scripts/eval_respond.py`: 50 clips each of neutral, joy, sadness and anger from MELD's test split (real dialogue).
Reading = the responder's `feeling` against MELD's human label (4-class unweighted accuracy, chance 25%). Response =
physical checks on the motion recipe against how the person actually felt (sad → gentle, angry → calm, happy →
joyful, neutral → attentive), with a missing recipe counted as a failure. With the two previous lines of the dialogue
as context.

| responder | reads tone only | words only | words + tone | appropriate response | valid recipe | LLM calls |
|---|---|---|---|---|---|---|
| voice model alone | 33% | | | | | |
| Qwen3-Next-80B-A3B-Instruct (hosted) | | 45% | 48% | 66% | 97% | 2 |
| Qwen3-4B-Instruct-2507 (local, off the shelf) | | 48% | 37% | 58% | 78% | 2 |
| student v1: Qwen3-4B distilled (local) | | | 46% | 64% | **100%** | **1** |
| student v2: + hinted relabelling (local) | | | 49% | 74% | **100%** | **1** |
| **student v3**: + per-feeling cap, recipe first (local) | | | **50%** | **73%** | **100%** | **1** |
| student v3 small: same data on Qwen3-1.7B (local) | | | 45% | 71% | **100%** | **1** |

- The voice reading adds little: nothing for the 80B (45% → 48% is within noise for 200 clips), and it misleads the
  off-the-shelf 4B (48% → 37%), which over-trusts it.
- Context (the previous two lines) changed little for either model.
- The most common error for the 80B and student v1 is anger read as happiness (16-18 of 50 angry clips): MELD's
  anger is often sarcastic, and the voice model hears it as happy. Student v2 fixes most of it (3 of 50).

## The distilled student

One local model does the responder's and the planner's job in one call: heard → `{feeling, reading, response,
recipe}` (`rmr.respond.Student`, `--voice-model mszarski/reachy-voice:student/v3`).

- Data: 4,898 MELD *training* clips, listened to on a GPU job (`scripts/listen_meld.py`, $0.57); the 80B teacher
  labelled each (`scripts/label_respond.py`); kept only answers whose `feeling` matches MELD's human label and whose
  recipe passes the physical check: 1,870 examples (teacher agreement by label: happy 62%, sad 47%, angry 35%,
  surprised 30%, neutral 26%, anxious 24%).
- Training: LoRA on Qwen3-4B-Instruct-2507 (Apache-2.0), 2 epochs, one A10G, 26 min ($0.55); val loss 1.37 → 0.44.
- Result: reads people about as well as the 80B (46% vs 48%), always writes a valid recipe (100% vs 78% for the base
  4B), responds appropriately 64% of the time (base 4B 58%, 80B 66%), in one call instead of two. On the "bad news
  told as a joke" clip the voice model hears *happy*; the base 4B shared the joy, the student read "overwhelmed by
  loss, despite a happy tone in their voice" and gave three slow, comforting nods.
- Speed: 0.65 s per answer batched on an A10G. On a 4-core CPU, unquantised in transformers, a cold process takes
  about 70 s (model loading included).

### Student v2: learning from the teacher's mistakes

v1 only learned from clips the teacher read right, so it never saw the hard ones: the teacher agreed with the human
label on just 35% of angry clips, and the angry clips it got right were the obvious ones. For v2 the teacher answered
each of the 2,971 clips it had misread again, this time told the human label and asked to find the cues for it in the
words, the conversation and the voice, and to write its reading as if it had noticed them itself (rationalisation, as
in STaR). The student is trained on that answer without the hint, so at test time it has to find the cues itself.
4,609 training examples (v1: 1,770; angry 1,155 vs about 400), same training recipe, 69 min of training on an A10G (about $1.25 for the job).

Reading per true feeling, out of 50 clips each (rows: MELD label; columns: the model's reading):

| | v1 → neutral | happy | sad | angry | other | v2 → neutral | happy | sad | angry | other |
|---|---|---|---|---|---|---|---|---|---|---|
| neutral | 16 | 15 | 8 | 2 | 9 | **24** | 5 | 6 | 5 | 10 |
| happy | 7 | **34** | 2 | 1 | 6 | 8 | 27 | 1 | 8 | 6 |
| sad | 2 | 10 | **25** | 4 | 9 | 7 | 5 | 20 | 12 | 6 |
| angry | 3 | 16 | 4 | 18 | 9 | 6 | 3 | 3 | **27** | 11 |

- Anger read as happiness: 16 → 3; angry recall 36% → 54%; neutral read as happy 15 → 5.
- The cost: v2 now reads some sad (12) and happy (8) clips as angry; it over-learned anger a little. Hinting
  every misread clip shifted the label balance toward the classes the teacher found hardest; rebalancing the
  hinted set is the obvious next tweak.
- Responses improve more than readings (64% → 74%, above the 80B teacher's 66%): reading sadness as anger still
  gets a calm, slow response, which suits both, while reading anger as joy got a bouncy one.

### Student v3: balanced, and the recipe first

v2 over-learned anger, because hinting every misread clip gave it 1,155 angry examples to 541 sad ones. For v3,
`scripts/label_respond.py --cap 700` keeps at most 700 examples per feeling, preferring answers the teacher got right
unaided (3,460 examples). The answer also changed order to feeling, response, recipe, reading, so a streaming client
can start the motion before the reading is written (`rmr.respond.ANSWER_ORDER`).

| reading, of 50 clips each | v2: sad → angry | happy → angry | angry → angry | angry → happy |
|---|---|---|---|---|
| v2 | 12 | 8 | 27 | 3 |
| v3 | **5** | **4** | 23 | 6 |
| v3 small (1.7B) | 5 | 6 | 21 | 7 |

The cap removes most of v2's false anger at a small cost in angry recall, with overall reading (50%) and responses
(73%) unchanged. Training cost about $1.20 (4B) and $0.50 (1.7B) on an A10G.

**Speed with streaming** (Q4_K_M behind `llama-server`, 4-vCPU cloud VM, CPU only, median over 10 clips):

| | first motion | full answer |
|---|---|---|
| v3 (4B, 2.5 GB) | 15.0 s | 19.4 s |
| v3 small (1.7B, 1.1 GB) | **5.8 s** | 7.5 s |

`/api/respond?stream=1` sends the motion as soon as the recipe is complete; on this CPU the 1.7B student starts
moving in under 6 s. On a GPU both are well under a second. Pick v3 for the best reading, v3 small for a CPU-only
laptop. The small model is a hybrid thinking model: start `llama-server` with
`--chat-template-kwargs '{"enable_thinking":false}'`, as it was trained.

### Quantised for laptops (GGUF)

`scripts/gguf_student.sh` (a CPU job, $0.02) converts the merged student with llama.cpp to
`mszarski/reachy-voice/gguf/{v1,v2,v3}` and `gguf/v3-1.7b` (1.1 GB; same file name): `reachy-voice-4b-Q4_K_M.gguf` (2.5 GB) and `Q8_0` (4.3 GB). On a balanced 60-clip
subset of the evaluation, Q4_K_M behind `llama-server` vs the full-precision student:

| student v1 | reads people (UA) | appropriate response | valid recipe | same feeling as bf16 | seconds per answer |
|---|---|---|---|---|---|
| bf16, transformers | 40% | 57% | 100% | | 0.65 batched (A10G GPU) |
| Q4_K_M, llama.cpp | 35% | 63% | 100% | 55 / 60 | 15.5 median, 24 max (4 vCPU, CPU only) |

The differences are within noise for 60 clips (3-4 clips each way). 15 s is on a small 4-vCPU cloud VM (8 tokens/s);
a laptop with Metal or any GPU is several times faster, and a recent 8-core CPU about twice as fast.

## Running it locally

```bash
pip install -e ".[ik,generator,hf,serve,voice]"
# any OpenAI-compatible local server, e.g. Ollama or llama.cpp serving Qwen3-4B-Instruct-2507 (GGUF)
PLANNER_BASE_URL=http://localhost:11434/v1 PLANNER_API_KEY=local \
  python -m rmr.server --model <served model name> --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt
# or the distilled student, one local call, no server and no network after the first download
python -m rmr.converse clip.wav --voice-model mszarski/reachy-voice:student/v3 --out runs/converse
python -m rmr.server --voice-model mszarski/reachy-voice:student/v3 --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt
# or the quantised student behind llama.cpp (2.5 GB, CPU is fine) ...
hf download mszarski/reachy-voice gguf/v3/reachy-voice-4b-Q4_K_M.gguf --local-dir .
llama-server -m gguf/v3/reachy-voice-4b-Q4_K_M.gguf --port 8080 -c 4096
# (the 1.7B: gguf/v3-1.7b/..., and add --chat-template-kwargs '{"enable_thinking":false}')
python -m rmr.server --voice-model http://localhost:8080/v1 --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt
# ... or Ollama (deploy/ollama/Modelfile)
ollama create reachy-voice -f deploy/ollama/Modelfile
python -m rmr.server --voice-model "http://localhost:11434/v1#reachy-voice" --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt
```

The llama.cpp route is tested here; the Ollama Modelfile is not (Ollama wasn't installable in this environment).

Open `http://localhost:7860` and hold the mic button, or press **Hands-free** and just talk: the viewer sends each
turn when you pause for 0.9 s (after at least 0.6 s of speech), and doesn't take a new turn while Reachy answers.
Browsers allow the microphone on localhost and HTTPS.

## On the robot

`rmr/robot.py` runs the same loop on a Reachy Mini: the robot's microphone feeds the listening controller at 25 Hz
(`set_target` on every frame), the end of a turn is posted to the server (`/api/respond?stream=1`), and the answer is
played frame by frame as soon as it arrives. The models stay on the server, a laptop or the Space, so the robot side
needs only the `reachy-mini` SDK.

```bash
python -m rmr.robot --server http://<laptop>:7860                 # robot mic, hands-free
python -m rmr.robot --server http://localhost:7860 --wav talk.wav  # a recording instead of the mic
```

Tested against the SDK's MuJoCo simulator (`reachy-mini-daemon --sim --headless --no-media`), not yet on hardware:
a 3.7 s MELD clip → heard after 5.7 s (Whisper and the voice models on CPU) → moving after 24.5 s with the v2 4B
student on CPU (with v3 small, about 12 s: 5.7 s to hear plus 5.8 s to the first motion). The simulated head follows the streamed frames with a median error of 2.3°
(about two frames of lag). Untested on hardware: the robot's own microphone path (`mini.media.get_audio_sample`),
and how loud its motors are in its own mic while it moves.

## Listening while you talk

Responding only after you finish feels like talking to a voicemail. While you hold the mic button, the viewer runs a
small controller on your voice's loudness, 25 times a second (`visualizer/src/Listen.js`, a line-for-line port of
`rmr/listen.py`; `tests/test_listen.py` checks they agree to 1e-9):

- **attentive pose** while you speak: leans in a little, tilts its head (alternating sides each time you start
  again), antennas slightly up; relaxes to neutral 4 s after you stop.
- **nods** at your pauses: 0.3 s of silence after at least 1 s of speech earns a nod, after 3 s a double nod (at most
  one every 1.2 s).
- **antenna perks** when your voice lifts (the fast loudness envelope 8 dB above the slow one, at most every 3 s).

The noise floor is tracked continuously (it drops at once and creeps up slowly), so the thresholds follow the room. No
model and no network: it reacts within a frame. After you let go it keeps listening to the silence (the closing nod)
until the response move arrives. The constants were set by ear on read speech and a few MELD clips, not fitted to
data. A learned alternative for the head is in [listening.md](listening.md).

```bash
python -m rmr.listen talk.wav --video listening.mp4 --json listening.json   # render it, with the speech as soundtrack
```

## Conversation memory and idle behaviour

- **Memory**: the server keeps each session's last two lines (for five minutes) and passes them to the responder as
  "earlier in the conversation", the same format the student was trained on (MELD's previous two lines). The viewer
  and the robot send a session id; sessions are separate, and at most 1,000 are kept.
- **Idle** (`rmr/idle.py`, `/api/idle`): after 10 s of quiet, every 12 to 25 s, a small move (look around, stretch,
  curious tilt, antenna twitch, settle, glance up; drowsy after a minute), written as recipes and generated, so no
  two repeat. In the viewer's hands-free mode and in `rmr.robot` (`--no-idle` to turn it off); speaking cuts an idle
  short at once.

## Licence-clean voice data (no MELD)

MELD is cut from a TV show, so students trained on it inherit that. Student v5 is trained without it:

- **Words**: about 4,200 lines written by the open teacher (`scripts/synth_lines.py`): six feelings, a random topic
  and speaker per batch, styles that fit the feeling (sarcasm only for anger, tiredness for sad / angry / anxious).
  Neutral lines first came out sarcastic or weary ("Beautiful overcast perfect day to feel emotionally drained"),
  and the teacher read 56% of them as sad, so styles are now per feeling.
- **Tone**: every line gets the voice reading of a *real* recording of the same feeling: the voice model run on all
  7,442 CREMA-D clips (`scripts/tone_bank.py`; ODbL, 91 actors). Only the readings are used, not the audio or the
  words. TTS was tried first (`scripts/synth_speech.py`, Zonos, Apache-2.0, blended voices), but its voices read as sad
  whatever the emotion setting (sad 100%, angry 7%, happy 27%), so the tone would have been made up.
- **Tone-only examples**: 800 CREMA-D clips with their flat sentence as the words ("It's eleven o'clock."), so only
  the reading tells the feeling. This is where the student learns what this voice model's readings mean on real
  voices: it hardly ever says "happy" for a happy actor (they read as neutral or surprise), and it hears fear as sad.
- Teacher labels with the hinted relabelling and a per-feeling cap, as in v3 (`scripts/label_respond.py`, which now
  takes a written feeling as the label). 18 of the 91 actors are held out of everything for evaluation.

**Evaluation without MELD** (`scripts/eval_clean.py`): a *tone test* (300 clips from the held-out actors, five
feelings, flat sentences: words alone score chance) and 255 *fresh lines* (another writer seed, none seen in
training, readings from held-out actors). UA = unweighted accuracy of the feeling; physical = the response's
recipe passes the physical check (as on MELD).

| | tone test UA (chance 20%) | fresh lines UA (chance 17%) | MELD test UA | MELD appropriate |
|---|---|---|---|---|
| teacher, words only | 20% | 42% | | |
| teacher (80B), words + tone | 40% | 43% | 48% | 66% |
| v3 4B (MELD-trained) | 32% | 52% | **50%** | **74%** |
| v4 4B (lines only) | 28% | **64%** | 34% | 64% |
| **v5 4B (lines + tone clips)** | **47%** | 63% | 37% / **42%** calibrated | 60% / **72%** calibrated |
| v5 1.7B | 41% | 59% | 40% / 40% calibrated | 69% / 72% calibrated |

- **v5 uses tone.** On the tone test it beats the teacher and every earlier student. A logistic regression on the
  same readings (held-out actors) reaches 51% (54% with all class probabilities), so v5 is close to what the
  readings allow. v4, without the tone-only clips, ignored tone: it read the words.
- The fresh lines favour v4 / v5 (same writer as their training data); take that column as in-distribution.
- **Calibration** (`rmr.voice.Calibration`, `python -m rmr.server --calibrate`). The readings depend on the
  microphone and room. Mean valence is 0.38 on CREMA-D, 0.43 on meeting talk (AMI, `scripts/calibrate_tone.py`) and
  0.56 on MELD. The voice model says "happy" for 1% of CREMA-D clips, 16% of AMI and 40% of MELD (laugh tracks, loud
  delivery). Uncalibrated, v5 read MELD as happy (24 of 50 neutral clips). Calibration keeps a running mean per
  session and shifts the attributes so that mean sits at the tone bank's level. It also rescales the category
  probabilities by the tone bank's prior over the session's (a prior-shift correction). On MELD (corrected with
  MELD test's own means) v5 goes from 37% to 42% UA and from 60% to 72% appropriate. That is within two points of
  v3, which was trained on MELD itself.

Data: `mszarski/reachy-voice` `synth/v4/` (lines, readings, tone bank), `student/v5`, `student/v5-1.7b`.

## Human ratings

`deploy/rating_page.html` shows a clip someone said (transcript, two earlier lines, how their voice sounded) and k of
Reachy's answers as robot videos, with Reachy's reading under each. Raters mark the best and the worst
(best-worst scaling; with two answers, the better one). Two pools:

- **test items** (the held-out tone test and fresh lines of `scripts/eval_clean.py`, never trained on; 42 items,
  seven per feeling): one answer from each of four models (student v5, v5 1.7B, the MELD-trained v3, the 80B teacher).
  Each gets a human score: (times best - times worst) / times shown.
- **training lines** (synthetic, from v5's training set): four sampled answers of student v5. A best-worst judgement
  orders five of the six pairs, so each becomes five DPO pairs (`scripts/dpo_from_ratings.py`), where an a/b pick
  gives one.

The page was rebuilt on these licence-clean items before any votes came in. The first version used MELD clips.

`scripts/rating_agreement.py` then asks whether the automatic checks agree with people: when exactly one of the two
ends passes the physical check (or reads the feeling as labelled), how often is it the one people preferred?
If the checks agree, they can steer future experiments without asking anyone; if not, they need fixing first.

## Caveats

These models read expressed emotion, not what someone feels, and they vary across speakers, accents and cultures.
Audio is processed per request and not stored. The EU AI Act prohibits emotion recognition in workplaces and
schools (since February 2025); a home companion is outside that, but deployment context matters.
