# Session handoff

Notes for a fresh Claude session picking up this project, written at the end of the long session that built it
(branch `claude/great-tesla-ln84m6` of `mszarski/claude_playground`, moving to its own repo). Read this, then
`CLAUDE.md`, `README.md`, `docs/ROADMAP.md` and `docs/results/voice.md`. Most of the reasoning below exists nowhere else.

## 1. Goal

- **Started as:** a recreation of the Reachy Mini text-to-motion system from
  [pham-tuan-binh/reachy-motion-generator](https://github.com/pham-tuan-binh/reachy-motion-generator) (Binh Pham's blog post
  *The best expressive harness for robots*). The pipeline is text → LLM planner → recipe → plan → flow-matching generator →
  25 Hz 9-DoF motion → IK projection → reachable move JSON.
- **Became:** an emotionally responsive companion. It hears your words *and* tone of voice, reacts within a second,
  answers with expressive, physically reachable motion, listens while you talk, keeps a mood, and learns from your ratings.
- **What counts as success:** the user's own judgement. The user rates clips (duels, thumbs) and those ratings decide.
  Automatic judges (LLM and VLM) were tested and agreed with the user at chance level, so they don't decide anything.
- **Constraints:**
  - Open licences everywhere: Apache/MIT models, licence-clean training data. MELD was removed from training for this
    reason.
  - It must be able to run locally on open-weight models.
  - Models, data and Spaces live in private Hugging Face repos under `mszarski`.
  - GPU spend goes through HF Jobs with a dollar cap.
  - The user has **no robot yet**. Everything is checked in the 3D viewer and the MuJoCo simulator.

## 2. Current state

### Live: Space `mszarski/reachy-mini-motions-live` (private, Docker, `l4x1`, sleeps after 1800 s idle)

**Serving stack:**
- `deploy/Dockerfile.vllm` (base `vllm/vllm-openai:v0.30.0`) runs `deploy/start.sh`. `start.sh` starts
  `vllm serve` in FP8 for the audio responder, then `python3 -m rmr.server`.
- **Responder:** audio-native, Qwen2-Audio-7B-Instruct fine-tune **v2** (`mszarski/reachy-voice:audio/student/v2`). It
  hears the audio directly and also gets the Whisper transcript and the tone line.
- **Hearing:** Whisper-large-v3-turbo fp16 on GPU (`rmr/voice.py` `ASR_GPU`; Whisper-small is the CPU fallback), plus two
  WavLM tone models (MSP-Podcast) with per-session `Calibration`.
- **Generator:** `hf://mszarski/reachy-motion-generator/generator_v2.pt` (21.8M-param flow-matching transformer).
- **Reach:** `rmr/reach.py` Stewart IK on every frame, plus `MOTOR_LIMITS` from the URDF, so no pose needs a servo
  past its range. This fixed "neck pulling off the platform". The viewer has the same check (`StewartIK.feasible()`).
- **Listener:** rules-based, style **x06** (`rmr/listen.py` `STYLE`, with ears_off and pitch_off), biased by mood
  (`mood_style()`). The turn ends after `END_OF_TURN = 0.7` s of silence.

**Live features:**
- An instant reaction (`rmr/react.py`) fires the moment the feeling word is decoded.
- Mood (`rmr/mood.py`): valence/arousal with a 90 s half-life. It drives idles (`rmr/idle.py` `MOOD_IDLES`) and the
  listening style.
- Thumbs up/down (`rmr/feedback.py`) push through `CommitScheduler` to the private dataset `mszarski/reachy-feedback`
  under `live/`.
- Smooth 0.4 s slerp blending between moves (`visualizer/src/Player.js`).
- A front-on camera with no auto-rotate.
- A puppeteering page (`/puppet.html`): MediaPipe Face Landmarker → head and antennas. Saved clips go to the same
  feedback dataset.
- iPhone Safari fixes (commit `1781734`).

**Headline numbers (eval_clean, held-out CREMA-D actors).** "Physical" is the share of answers whose motion passes the
physical check.

| responder | tone test (300 real clips): feeling right / physical | cloned test lines (255): right / physical | + unrelated context |
|---|---|---|---|
| **audio v2 (live)** | **91.0%** / 97.5% | **68.3%** / 88.0% | **67.9%** / 89.1% |
| text pipeline (Whisper + tone line → Qwen3-4B student v8-dpo) | 42.3% | 57.8% | |

**Latency on the live L4 (server clock):**
- Hearing: 180–455 ms.
- Reaction motion: 0.5–0.9 s.
- Full response motion: 3.1–3.9 s.
- Add the 0.7 s end-of-turn wait to each. Reachy starts moving about 1.3–1.6 s after you stop talking.

### Done but waiting on the user

- **Thumbs ratings:** the plumbing works end to end. They are meant to feed a reward model and GRPO. Not enough ratings
  have been collected yet.
- **Puppet clips:** they save to the dataset, and `python -m rmr.generator ... --clips` can train on them. No clips
  recorded yet.
- **Puppet page:**
  - Not yet tried on a real face. The mirror-mode sign (left vs right) is untested.
  - The retarget gains in `visualizer/src/Puppet.js` (`Retarget`) are first guesses.
- **iPhone Safari fixes:** tested only in Playwright/WebKit emulation, not on the user's real iPhone.

### Stale or untested

- **The old gallery Space `mszarski/reachy-mini-motions`** shows examples from earlier models. Regenerate them or retire
  the Space.
- **The README architecture diagram** still describes the text path: Whisper-small and the GGUF student on llama.cpp.
  The text path remains the default of `deploy/live_space.py` without `--audio-model`, but the live Space runs the audio
  path (described further down in README.md and `docs/results/voice.md`). Worth tidying.
- **`rmr/robot.py`** (the real-robot loop) is tested against the simulator only.

## 3. Decisions and reasons

- **Qwen2-Audio-7B over Voxtral.** A Voxtral-Mini-3B fine-tune on the same data got **38%** on the tone test, against
  91% for Qwen2-Audio. Faster isn't worth that.
- **vLLM FP8 on an L4.**
  - Over bf16 transformers: the bf16 model didn't fit an L4 comfortably.
  - Over an A10G or L40S: FP8 kept accuracy equal. The L4 is cheap, and A10G and L40S capacity was often unavailable.
- **v2 over v3 (v3 = no transcript in the input).** v3 could start before Whisper finishes, but its accuracy on cloned
  lines fell to 62.8% against 68.3% (sign test p=0.035). The user said "keep it", meaning v2. The parallel path
  (`--audio-no-transcript`, `Engine._respond_audio_parallel`) stays in the code.
- **Whisper-large-v3-turbo over Whisper-small and over Voxtral-Mini-4B-Realtime.**
  - Turbo against small: 7.3% vs 8.4% WER and 0.14 vs 0.23 s per clip on the L4.
  - Voxtral-Realtime was not more accurate. Streaming only finishes about 0.5 s after speech ends by design, so it isn't
    faster than turbo here.
- **Q8 over Q4 GGUF for the text student.** Q4's physical fit on the tone test was 70% against 78% (p=0.01).
- **END_OF_TURN 0.7 s, not 0.6 s.** The test speech has natural gaps of 0.6 s, so 0.6 cut people off.
- **Listener style x06** was chosen by the user's listening-duel votes (`docs/results/listening.md`). The learned
  listener (trained on licensed conversational data) is non-commercial, so it's opt-in only.
- **Servo limits enforced in reach, not only IK solvability.** IK alone accepted poses whose motors were past range,
  which caused the visual "neck off the platform".
- **Real fixes, not synthetic post-processing.** An added tremor/trembling effect was removed at the user's request:
  "We should want the model to be better, not add synthetic changes after". The generator v2 was retrained on
  recipe-like plans (`--simplify 1.0`) instead.
- **Kimi-K3 as the teacher LLM** writes the responder labels (relabel v8). Then DPO on 160 of the user's own preference
  pairs gave **v8-dpo**, which the user preferred over v8 ("why would we not use the dpo one?").
- **Training audio:**
  - Written lines are voiced by **Chatterbox** zero-shot cloning of the paired CREMA-D clip (training actors only), so the
    line carries a real actor's emotional delivery.
  - 676 tone-only examples use real CREMA-D audio.
  - Lines with Whisper WER > 0.35 are dropped, leaving **4,238** examples.
  - Held-out actors are CREMA-D `actor % 5 == 0`.
- **Ratings decide, judges don't.** LLM and VLM judges were at chance against the user, even with dense frames (4+ per
  second).

## 4. Dead ends (don't retry without a new idea)

| tried | result | where |
|---|---|---|
| Voxtral-Mini-3B audio responder fine-tune (+ encoder LoRA) | 38% tone, 61% cloned lines | `audio/student/voxtral-v1`, job `6acea370c662881452231b4a` |
| v3 transcript-free audio responder | 62.8% vs 68.3% cloned lines (p=0.035) | `audio/student/v3`, job `6aced73c5ccb47ba0f78f4e7` |
| Voxtral-Mini-4B-Realtime as the transcriber | no more accurate than turbo, not faster offline | job `6ad9bd6b0a35baab53a272ed`, `scripts/asr_bench.py` |
| Q4_K_M GGUF of the text student | physical fit 70% vs 78% | `docs/results/voice.md` |
| Plain TTS (e.g. Zonos) to voice training lines | every voice sounds sad whatever was asked | `scripts/synth_speech.py` |
| MELD in training | licence problem; removed | `scripts/listen_meld.py` is evaluation and history only |
| Automatic judges (LLM text judge, VLM video judge) | chance agreement with the user's ratings | `scripts/judge_ratings.py`, `scripts/vlm_judge.py` |
| Planner grounding on real clips | negative result | `scripts/ground_planner.py`, `docs/results/planner_finetune.md` |
| Synthetic tremor/trembling on the generator output | user rejected it as unrealistic; removed | — |

## 5. Gotchas

**Hugging Face and infrastructure:**
- **GPU capacity:** HF L4, L40S and A10G capacity is often short.
  - For jobs, launch on two flavors and cancel the loser.
  - A Space can loop on restarts until capacity frees up.
- **BUILD_ERROR exit 128:** a Space build failing this way with no logs is transient. Push a new commit to retrigger.
- **Paid hardware:** switching a Space to paid hardware needs the user's explicit OK (the auto-mode classifier blocks
  it otherwise).
- **Sandbox network:** the cloud sandbox may block `*.hf.space`. The user had to allow the domain. Until then, test
  the Space from inside an HF job.

**Images and Python packages:**
- **`vllm/vllm-openai` image:** has no `python` binary. Use `python3` (in `start.sh` and the job scripts).
- **llama.cpp server image:** needs `LD_LIBRARY_PATH=/app`. `start.sh` exits if llama-server dies, so the Space restarts
  instead of hanging.
- **torch wheels:** use the cu128 index. The default wheel targeted CUDA 13, which the job drivers didn't support.
- **transformers 5:**
  - There is no `warmup_ratio`; use `warmup_steps`.
  - `use_cpu` and bf16 must depend on whether a GPU is present.
  - Smoke-test training code on the tiny `trl-internal-testing` models before paying for a GPU.
- **Voxtral:**
  - Needs `transformers==5.18.0` and `mistral-common==1.12.0`.
  - It allows no system message alongside audio, so the instructions go in the user turn (`rmr/audio_responder.py`
    `voxtral_inputs`).
- **Chatterbox install:** needs `pyarrow<17` (newer pyarrow needs numpy 2) and `setuptools<81` (otherwise the perth
  watermarker loads as `None`). Run it in its own venv: it pins its own torch and transformers.

**Code conventions and test traps:**
- **Units and channel order** are in CLAUDE.md:
  - Trajectories are SI; plans and recipes are deg and mm.
  - The right antenna droops with negative angles, the left with positive.
  - 25 Hz everywhere; library clips are ~100 Hz and must be resampled on their `time` array.
- **Listening-style tests:** tests that count listener nods must monkeypatch `listen.STYLE = listen.CLASSIC`, because
  x06 turns some gestures off.
- **`Mood.update()`** must return `state(self.t)`, not the state decayed to wall-clock time. That bug was fixed; keep the
  test.

**Browser and viewer:**
- **iOS Safari:**
  - Create the `AudioContext` synchronously inside the tap handler, before any `await`, then `resume()`.
  - Wrap `setPointerCapture` in try/catch.
  - Inputs must be ≥16 px or the page zooms.
  - Use `dvh` units and safe-area insets.
- **Playwright testing in the sandbox:**
  - jsdelivr is blocked, so route three, urdf-loader and mediapipe to local npm copies.
  - The proxy doesn't bypass loopback.

**Session mechanics:**
- **Long-running watches:** the Monitor tool expires after about 5 minutes. Use `run_in_background` bash loops.
- **Restarts:** container restarts kill background watchers. Re-create them after a restart.
- **Evaluation-only data:** `data/teacher/val.jsonl`, the 12 held-out emotions, and the held-out CREMA-D actors are
  evaluation only. Never train on them.

## 6. The user's preferences

**How they work:**
- They want plain answers to direct questions ("does the server take the microphone", "what branch are we on?"). Answer
  first, then offer next steps.
- Asked "what's next", they like a short numbered list of options with a recommendation. They often answer "do all of
  them" or pick numbers.
- They are happy for work to run long and autonomously (they were away for two days at one point). Keep cooking, but
  report status when asked.
- Commit and push as you go. A stop hook complains about uncommitted or unpushed work.

**What they value:**
- Real model improvements over hacks or post-processing.
- Open licences, and runnable locally.
- Their own ratings over any automatic score. When they're asked to rate, the rating tool must not repeat clips. They
  noticed the duel page repeating once.

**Devices and access:**
- They're often on an **iPhone**. Earlier they couldn't load Spaces or artifacts in iOS Safari; that's why the viewer was
  made Safari-friendly, and why rendered MP4s were sent as files.
- Give them direct links: they asked "Link to space?"

**Money:**
- Spending on HF is fine within stated caps: "cap at $15", "up to 15 dollars" for training. Their HF account has
  auto-recharge on.
- Always use `--max-usd` on jobs.
- Ask before switching a Space to bigger paid hardware. They approved the L4 and said "Do both" for the bigger-GPU and FP8
  options.

## 7. Next steps

1. **Collect ratings and clips.** Ask the user to use the thumbs on the live Space and record a few puppet clips.
   Everything lands in `mszarski/reachy-feedback` (`live/ratings.jsonl`, `live/audio/`, `live/clips/`).
2. **Reward model and GRPO.** Once there are a few hundred thumbs, train a reward model on (heard, answer, rating) and
   run GRPO or DPO on the audio responder (TRL; see `scripts/dpo_from_ratings.py` for the text-student version).
   Re-run eval_clean and the sign tests against v2 before switching the Space.
3. **Retrain the generator with puppet clips:** `python -m rmr.generator train --hub --clips <dir> ...`. Compare against
   `generator_v2.pt` on the 12 held-out emotions (91.7% identification for v1) and on the user's ratings.
4. **Check the puppet page and Safari on the user's real iPhone:** mirror sign, retarget gains, mic and hold-to-talk.
5. **Streaming transcription,** only if latency becomes the complaint. Turbo is already 180–455 ms; the end-of-turn
   wait and the responder dominate.
6. **Regenerate the gallery Space's examples** (`scripts/build_gallery.py`) with the current models, or retire that
   Space.
7. **A v8 vs v8-dpo rating round** on the text path, to confirm the DPO gain with fresh clips.
8. **Better audio training data:** real conversational speech (licence-clean) instead of acted CREMA-D plus cloning.
   The cloned-lines score (68%) is far below the tone score (91%).
9. **Robot hardware test** of `rmr/robot.py` once the user has a Reachy Mini.

## 8. Pointers

**URLs:**
- Space: https://huggingface.co/spaces/mszarski/reachy-mini-motions-live
- App: https://mszarski-reachy-mini-motions-live.hf.space
- Puppet page: https://mszarski-reachy-mini-motions-live.hf.space/puppet.html
- Reference implementation: https://github.com/pham-tuan-binh/reachy-motion-generator

**Hugging Face repos (all private, owner `mszarski`):**
- `reachy-voice` (dataset):
  - `student/v8-dpo`: the text responder.
  - `gguf/v8-dpo/reachy-voice-4b-Q8_0.gguf`.
  - `audio/student/v1`, `audio/student/v2` (live), `audio/student/v3`, `audio/student/voxtral-v1`.
  - `audio/asr_bench_*`, `audio/vllm_bench*`.
  - `code/`: code uploaded by `deploy/hf_job.py`.
  - Job outputs.
- `reachy-motion-generator`: `generator.pt` and `generator_v2.pt` (live).
- `reachy-mini-planner-4b`: the fine-tuned typed-prompt planner.
- `reachy-planner-sft`: the planner's SFT data.
- `reachy-listening`: the learned-listener weights (`listener/v1/listener.json`, non-commercial, opt-in).
- `reachy-feedback` (dataset): live thumbs and puppet clips.
- Spaces: `reachy-mini-motions-live` (live) and `reachy-mini-motions` (old gallery).

**HF Job IDs:**

| job | ID |
|---|---|
| audio v1 | `6acd5407de32f9b0f4325dd9` |
| audio v2 | `6acd80cdb0ad3e9da5e9f9b5` |
| audio v3 | `6aced73c5ccb47ba0f78f4e7` |
| Voxtral fine-tune | `6acea370c662881452231b4a` |
| ASR bench | `6ad9bd6b0a35baab53a272ed` |
| vLLM FP8 bench | `6acd7991b0ad3e9da5e9ea69` |

**Commands:**

```bash
pip install -e ".[ik,dev]" && pytest        # all tests (CPU, no network)

# redeploy the live Space as it is now (needs HF_TOKEN in the env; it becomes a Space secret)
python deploy/live_space.py --repo mszarski/reachy-mini-motions-live \
    --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt \
    --audio-model mszarski/reachy-voice:audio/student/v2 --vllm --hardware l4x1 --sleep 1800

# run anything on an HF GPU with a cost cap; /work/out is uploaded to --out-repo/--out-path
python deploy/hf_job.py --max-usd 2 --flavor l4x1 --out-repo mszarski/reachy-voice --out-path audio/<name> \
    --name <name> --packages "transformers>=5.5 soundfile" -- python scripts/audio_llm.py <subcommand> ...

# local server (text path); the audio path adds --audio-model <repo:path or http://host:port/v1#name>
python -m rmr.server --voice-model mszarski/reachy-voice:student/v8-dpo \
    --ckpt hf://mszarski/reachy-motion-generator/generator_v2.pt
```

**Key files:**
- Server and engine: `rmr/server.py`.
- Responders: `rmr/audio_responder.py` and `rmr/respond.py`.
- Hearing: `rmr/voice.py`.
- Live behaviour: `rmr/listen.py`, `rmr/react.py`, `rmr/mood.py`, `rmr/idle.py`, `rmr/feedback.py`.
- Motion: `rmr/reach.py`, `rmr/generator/`.
- Audio responder pipeline: `scripts/audio_llm.py` (bakeoff/build/train/answer/bench) and `scripts/clone_speech.py`
  (voice/check).
- Evaluation: `scripts/eval_clean.py` and `scripts/eval_respond.py`.
- Deployment: `deploy/start.sh`, `deploy/Dockerfile.vllm`.
- Viewer: `visualizer/src/app.js` and `visualizer/src/puppet_app.js`.
- Results write-ups: `docs/results/voice.md`, `docs/results/listening.md`, `docs/results/generator.md`.
- Issue tracking: the `bd` CLI (see CLAUDE.md).
