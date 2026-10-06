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

## Running it locally

```bash
pip install -e ".[ik,generator,hf,serve,voice]"
# any OpenAI-compatible local server, e.g. Ollama or llama.cpp serving Qwen3-4B-Instruct-2507 (GGUF)
PLANNER_BASE_URL=http://localhost:11434/v1 PLANNER_API_KEY=local \
  python -m rmr.server --model <served model name> --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt
# or without any server: the LLM in-process with transformers (slow on CPU)
PLANNER_BASE_URL=local python -m rmr.converse clip.wav --model Qwen/Qwen3-4B-Instruct-2507 --out runs/converse
```

Open `http://localhost:7860` and hold the mic button (browsers allow the microphone on localhost and HTTPS).

## Caveats

These models read expressed emotion, not what someone feels, and they vary across speakers, accents and cultures.
Audio is processed per request and not stored. The EU AI Act prohibits emotion recognition in workplaces and
schools (since February 2025); a home companion is outside that, but deployment context matters.
