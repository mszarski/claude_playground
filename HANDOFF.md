# HANDOFF: marine VHF radio transcription (`marine_vhf_asr`)

Session handoff notes, written 2026-11-02 at the end of a long Claude Code
session on branch `claude/modest-sagan-r554gj` of `mszarski/claude_playground`.
The project is moving to its own GitHub repo. `marine_vhf_asr/README.md` has
the full per-run results tables; this file is the context that lived only in
the conversation.

Everything below is pushed. No HF Jobs are running and nothing is billing.

---

## 1. Goal

Build a marine VHF radio speech-to-text model: port control / VTS, pilots,
coastguard, bridge-to-bridge traffic. It is the marine equivalent of
[`pphilip/voxtral-3B-atc-transcribe`](https://huggingface.co/pphilip/voxtral-3B-atc-transcribe),
which LoRA-fine-tunes `mistralai/Voxtral-Mini-3B-2507` on air traffic control
radio (8.0% WER on real ATC).

The user's framing, in order of importance:
- **General, not port-specific.** "needs to be general and not port specific -
  like what if theres a vessel out there". One model for any port, including
  vessels it has never seen and that are on no AIS list.
- **Usable accuracy on real radio.** "a 70% WER is unusable". Vessel names
  matter most: a transcript with the wrong ship is useless or dangerous.
- **Honest numbers.** The user challenges claims that can't be verified
  ("how do you know if it got it wrong if you dont trust the transcript?").

What success looks like (not formally agreed, my reading):
- Measured on **real, human-verified** marine VHF recordings (none exist yet):
  WER in the ATC model's range (~10%) and high name accuracy when an AIS
  candidate list contains the speaker.
- Low rate of confidently wrong names (a list name written when the speaker
  isn't on the list), currently ~4% on VHF-8K.
- Honest handling of unknown vessels: at minimum a good phonetic spelling,
  ideally a low-confidence flag.

## 2. Current state

### Works
- Full pipeline in `marine_vhf_asr/`: SMCP dialogue generator, VHF channel
  simulator, 5 TTS backends, data loaders for 4 public datasets plus our own
  manifests, LoRA training on HF Jobs, evaluation with WER / slot accuracy /
  wrong-name rate / name CER, vessel-name context prompting, runaway guards,
  two-pass name verification, real AIS candidate lists for 9 regions.
- Tests: `pytest marine_vhf_asr/tests/`: 74 pass in a full venv; with system
  Python 56 pass and 2 skip (optional deps).
- **Recommended model: v6**, adapter `mszarski/voxtral-3B-marine-vhf-v6`
  (private HF model repo).

### Model lineage (all adapters private on HF, LoRA r=32/alpha=64 on LM + audio encoder, 2 epochs, 1x A100)

| Version | HF repo | What changed | Key result |
|---|---|---|---|
| v1 | `mszarski/voxtral-3B-marine-vhf` | VHF-8K + MARTTS + ATC + ATCOSIM, ~11k clips | VHF-8K 28.1%→12.6%, MARTTS 15.3%→7.5%, ATC 69.1%→9.8% WER |
| v2 | `mszarski/voxtral-3B-marine-vhf-context` | candidate vessel list in prompt (`#mix`) | speaker on 5-name list: 97% names, 1.9% WER |
| v3 | `mszarski/voxtral-3B-marine-vhf-v3` | lists of 5-100, speaker missing 30% (`#mixlong`) | speaker not on 25/100 list: wrong name 16%→3.8% / 21.5%→4.8% |
| v4 | `mszarski/voxtral-3B-marine-vhf-v4` | + real-AIS dialogues from 7 regions | timed out; only step 1500 evaluated; wrong-name regression to 13.5% |
| v5 | `mszarski/voxtral-3B-marine-vhf-v5` | fixes v4: 120-name cap, speaker dropped 50%, Kokoro train voices only | NY/Øresund Chatterbox no list 11.9% / 14.0% WER |
| v6 | `mszarski/voxtral-3B-marine-vhf-v6` | AIS dialogues voiced by Kokoro + Piper + Chatterbox (1/3 each) | 13-25% relative WER gain on unseen engine (Parler) |

### Latest numbers

v6 on Parler-TTS test audio (an engine no training data uses), only clips
Whisper large-v3 judges intelligible (57% of NY clips, 43% of Øresund):

| WER / names right | v5 | v6 | Nemotron 30B (zero-shot) |
|---|---|---|---|
| New York, no list | 24.0% / 16% | 20.6% / 16% | 45.3% / 15% |
| New York, moving class A list | 18.0% / 49% | 13.5% / 56% | 28.5% / 30% |
| Øresund, no list | 27.3% / 8% | 24.1% / 9% | 54.5% / 6% |
| Øresund, moving class A list | 19.6% / 46% | 16.2% / 51% | 32.7% / 36% |
| Unknown vessels, names right NY / Øresund | 12% / 20% | 11% / 25% | |

v6 regression checks: VHF-8K no list 12.5% WER; speaker on 25-name list 96%
names; speaker not on list 3.8% wrong name; MARTTS 8.6%; ATC (real radio) 9.6%.

Nemotron-3.5-Transcribe-30B-A4B (zero-shot, just finished): worse than v6
everywhere. VHF-8K no list 59.8% (43% of clips came back empty), MARTTS
13.6%, ATC 52.4%. With a context list it rarely returns empty and improves a
lot (NY Chatterbox 30.1%→14.8%). Full table in the README section
"Comparison: NVIDIA Nemotron-3.5-Transcribe-30B-A4B".

### Half-finished / open
- **No real marine test set exists.** Every marine number above is synthetic
  speech. This is the single biggest gap.
- **Kaggle real data explored, not used yet.** "Marine Radio Chatter" (18
  Broadcastify feeds, see section 8). A sample is in
  `mszarski/marine-radio-chatter-sample`. Nothing trained on it.
- **The 5 real Kaggle clips Nemotron returned empty for** were sent to the
  user to listen to (it's unknown whether they contain speech). No answer yet.
  Clips: `22612/22612-20230711-1604`, `22851/22851-20230311-0127`,
  `22851/22851-20230804-1958`, `22851/22851-20240104-2333`,
  `26383/26383-20230731-2323` (in the sample dataset under `audio/`).
- **Pending user decision** on the next data step (see section 7).
- Unknown-vessel names have not improved since v5 (~10-25% exactly right on
  Parler, name CER ~0.4). Nothing tried so far moves them.

## 3. Decisions and reasons

- **Base model Voxtral-Mini-3B + LoRA, same recipe as the ATC model.** Same
  kind of audio (narrowband FM, squelch, accented English, formulaic
  phraseology); fits one 24 GB GPU. Not revisited; Nemotron 30B zero-shot
  confirmed a big general model doesn't beat the fine-tuned 3B.
- **Train on HF Jobs** (user authorized spending: "lets go", "yes do it").
  The dev container has no GPU. A100 `a100-large` at $2.50/h for training and
  eval; `a10g-small` ($1/h) for TTS/cheap evals; `cpu-upgrade` for data pulls.
- **Mix in real ATC radio** (`jacktol/ATC-ASR-Dataset`) and ATCOSIM: the only
  real VHF acoustics available; its transcript conventions match ours. Side
  effect: the model sometimes writes spoken letters as NATO words
  ("L Y N N" → "LIMA YANKEE ...").
- **MARTTS is fine to use**: the user decided its CC-BY licence is OK.
- **Score after normalizing both sides** (uppercase, numbers as words). That's
  why our ATC baseline is 69% not the card's 105.6%; numbers are not directly
  comparable with the ATC model's card.
- **Vessel-name context in the prompt** (from AIS in real use), spliced into
  Voxtral's transcription prompt before `[/INST]`; candidates shuffled
  (VHF-8K's own prompt always puts the true name last, which would leak).
- **Train with the speaker missing from many lists** (30% in v3, 50% in
  v5/v6 AIS data) so the model learns "match by sound" not "the answer is
  always on the list". Accepted ~3 points fewer names on short lists for 4-5x
  fewer confidently wrong names: the right trade for a radio log.
- **Two-pass verification is optional**, off by default: with v3+ it removes
  a third to half of the remaining wrong names but costs 1-3 points of names
  and 2x inference. Not worth it on real AIS lists in NY (wrong names <1%).
- **Generalization design (after the user's "a vessel out there" push):**
  real AIS from 9 regions; **New York and Øresund held out entirely**, every
  vessel name seen in their AIS excluded from training; ~17% of test speakers
  are "unknown" (on no list, named from the other test region). Rejected:
  training on NY's own traffic (it would have memorized NY's fleet).
- **Candidate lists from AIS snapshots**: vessels within 20 nm of the
  receiver in the last 10 min. Two variants evaluated: `moving_a` (class A
  transponder, moved >1 kn; median ~85 names in NY) and `all` (median ~350).
- **Honest test engine**: train and test on different TTS engines. Kokoro
  held-out voices were not enough (v4/v5 looked 3x better on Kokoro than on
  another engine). Chatterbox was used next, then Parler because MARTTS (in
  training) is itself Chatterbox audio.
- **Whisper large-v3 as an intelligibility judge for Parler** (Parler garbles
  a lot: Whisper gets 49% WER on Parler vs 19-27% on other engines). A clip
  counts if Whisper recovers >=80% of the non-name words. Names are excluded
  from the judgement because Whisper fails on unseen names.
- **v6 is recommended** over v5: consistent gain on an unseen engine, no
  meaningful regressions.
- **Nemotron rejected as a replacement**; possible use as a second opinion
  for pseudo-labelling real audio.
- **Kaggle dataset: internal research only.** Uploader says MIT, but it's
  scraped Broadcastify archives whose terms restrict redistribution. Don't
  republish audio or build a product on it without checking.

## 4. Dead ends

- **Kokoro-only training data (v4/v5)**: huge gains on Kokoro test audio
  (NY no list 16.6%→5.2% WER) that mostly vanished on another engine
  (11.9%). The model learns how one engine pronounces spellings. Don't judge
  a model on audio from an engine that is in its training data.
- **The Chatterbox "unseen engine" test** was contaminated: MARTTS is
  Chatterbox. Use Parler (or real audio) as the held-out engine.
- **Lists of up to ~350 names in training (v4)**: training ran ~2x slower
  (7-10 s/step), the job timed out at step 2449/2756, and the step-2000
  checkpoint upload failed silently. Lists are now capped at 120
  (`MAX_TRAIN_LIST` in `sources.py`).
- **Short lists only in training (v2)**: the model learned "the answer is on
  the list" and wrote random list names when the speaker was absent (21.5% on
  100-name lists).
- **Multi-engine training for unknown vessels**: v6 didn't move unknown-vessel
  names at all. More synthetic variety doesn't fix names nobody has heard.
- **Parler-TTS as a raw test set**: too garbled to use without the Whisper
  filter, and slow (~6-7 transmissions/min on A10G; had to halve the test set
  to `dialogue_id % 2 == 0` and split into 2 shards).
- **Nemotron model-card config on A100**: its replayssm/FlashInfer Mamba
  kernels need Hopper (sm_90: `griddepcontrol` PTX). Use the plain config on
  A100 or rent H100/H200.
- **Downloading the whole Kaggle audio dataset** (~100 GB) via
  `kagglehub.dataset_download` failed. Pull the transcripts plus selected
  files instead (`jobs/kaggle_sample_job.py`).

## 5. Gotchas

Code and libraries:
- transformers 5.x: no `warmup_ratio`; use `warmup_steps=0.03` (float =
  fraction). `bf16` must be `torch.cuda.is_available()` or CPU runs crash.
- Manifest `slots` dicts became sparse structs in `datasets`; columns are
  rebuilt in `sources.py`.
- Voxtral transcription prompts are **left-padded**; context is spliced
  before `[/INST]` (`prompting.build_inputs`). Loss only on transcript tokens
  + EOS.
- Runaway outputs: base Voxtral emitted 512 tokens of "37000..." on one ATC
  clip (16 points of WER); v2 got stuck on "VTS VTS VTS". Guards in
  `prompting.py`: `max_new_tokens_for` (15 tok/s + 24, cap 1024),
  `trim_runaway` (40 chars/s + 40), `collapse_loops` (1-4-gram repeated >=8x
  → 3 copies). They change no reference transcript.
- Verifier needs letter-insensitive matching ("H T M S" vs "HTMS"):
  normalize with `expand_letters=False` and try windows of 1..k+1 words.
- `ais.snapshot` needs the dataframe sorted by `t` (binary search);
  `ais.load` sorts it. `harbor.py --min-listed 5` is needed for Danish waters
  (short moving lists) or Great Belt yields 0 dialogues.
- `ais.spoken_name`: vowel-less 2-4 letter tokens are read as letters, with
  exceptions in `LETTER_ACRONYMS` (NYPD, FDNY, ...). An old rule spelled
  "LYNN" as letters; the original `test_ny` set was built before the fix.
- VHF-8K writes numbers as digits; ambiguous ones (3+ digit runs, times,
  coordinates; ~6%) are dropped. MARTTS spaced letters are joined
  (`acronyms=True`). VHF-8K single letters in names stay letters.
- About 2.5 WER points on NY come from letter spacing conventions ("M S C" vs
  "MSC"); Nemotron also loses on US spellings (DRAFT/METERS vs
  DRAUGHT/METRES) and "G II" → "GOLF TWO".
- Chatterbox needs numpy 1.26, `pyarrow<18`, `datasets<4`, `setuptools<81`,
  and a fallback `perth.PerthImplicitWatermarker = perth.DummyWatermarker`
  (already in `tts.py`).

Infrastructure:
- The dev container's network blocks `coast.noaa.gov` and `kaggle.com`: pull
  that data in HF Jobs (`jobs/ais_extract_job.py`, `jobs/kaggle_sample_job.py`).
- HF Jobs containers have no repo checkout: **upload code first** with
  `hf upload mszarski/marine-vhf-asr-code marine_vhf_asr marine_vhf_asr`
  (exclude tests/pycache), and job scripts fetch it via `CODE_REPO`.
  Forgetting this runs stale code.
- `hf jobs` timeout format: `90m`, `5h`; "1h30m" is invalid.
- `hf jobs wait` returns early; poll `hf jobs inspect` in a loop instead.
  Background shell tasks in Claude Code are limited to 2 h, so re-arm them.
- Hub downloads of 13k small files took 25 min; use `max_workers=32`.
- `datasets` cache filled the dev disk (15 GB of `cache-*.arrow`): delete
  them and use `datasets.disable_caching()` locally.
- `hf upload-large-folder` puts files at the repo root unless you set the
  path; check before uploading. Old `huggingface_hub` lacks
  `HfApi.upload_large_folder`.
- Training: use `hub_strategy="checkpoint"`, push the adapter BEFORE the
  long eval, and `--resume` from the Hub's last checkpoint if a job times out.
- Scheduler stalls on `a10g-large` and `l4` were seen; `a10g-small` worked.
- Whisper large-v3 judge: batch size 4 (OOM at 16), `chunk_length_s=30` for
  clips over 30 s.
- Heredocs in launch commands turned "\n" into real newlines once (syntax
  error in a paid job): `ast.parse` job scripts before launching.
- `pkill -f <pattern>` once killed its own shell; kill by PID.
- Nemotron (`jobs/nemotron/run.sh`): pip CUDA compiler packages must all be
  one version (nvcc, crt **and nvvm** pinned to the runtime's 13.0); a newer
  nvvm emits PTX 9.2 that ptxas 13.0 rejects. Set
  `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid another JIT compile. Each attempt
  spends ~15-22 min installing before vLLM starts (~$0.90 per failed try).
- Nemotron returns **empty output** on many VHF clips when no context terms
  are given (43% on VHF-8K); this, not mishearing, is most of its no-list WER.

Data:
- Kaggle WhisperX transcripts are machine-made: "thank you" is the most
  common segment (Whisper hallucination on squelch/noise); 19% of words have
  low alignment confidence. Real traffic is casual ("roger", "copy", "yes
  sir", "break"), not textbook SMCP.
- ATC test-set baseline on a 16-clip sample (35.7%) was wildly optimistic vs
  300 clips (69.1%). Don't trust tiny samples.

## 6. User preferences (how to work with mszarski)

- Messages are short and casual; reply with a clear verdict first, then a
  compact results table. Be explicit about **which model** a number belongs
  to: the user once read the untrained baseline's 69% as our model's result
  ("how did this get worse?", "a 70% WER is unusable").
- Wants general solutions; pushed back on a port-specific (NY-trained) plan.
- Values honesty over flattering numbers. Say when a test is weak, and
  correct earlier claims proactively (I had to retract that Chatterbox was an
  unseen engine, and that Nemotron's real-clip output was "plausible": I had
  no reference to check it against). Don't present judgment as measurement.
- Spending: authorized HF Jobs spending for this project. Practice that
  worked: give a cost estimate before each round, report actual cost after,
  and **ask before launching a new training run** (e.g. "Want me to go ahead
  with v5?"). Cheap evals/data jobs (<$5) were launched without asking.
- Answers "Both" / "lets do 1 and 2" to option lists: give numbered options
  with a recommendation.
- Asked to be told when long runs finish ("let me know when v6 results are
  in"): keep watching jobs and report without being asked again.
- Git: always commit and push (a stop hook flags uncommitted/untracked
  files). Work on the designated branch. **No PR unless asked.** Commit
  messages end with the Co-Authored-By / Claude-Session lines; no model names
  in commits or repo files.
- Don't edit `CLAUDE.md` for handoff purposes (updated separately).

## 7. Next steps (priority order)

The user was offered these three Kaggle-data options and hadn't chosen yet;
after the "how do you know it got it wrong" question I recommended **1 first**:

1. **Real test set via a hand-correction page.** ~200-300 transmissions
   from feeds held out from any training (e.g. New York 17329, Puget Sound),
   segmented with the WhisperX timestamps, pre-transcribed by v6 (and
   optionally Nemotron with context / Whisper). A web page (Claude Artifact
   with shared storage, or a small Gradio Space) shows audio + drafts + an
   editable box; ~1-2 h of the user's listening. This gives the first true
   real-world WER and name accuracy. Needs the user's time.
2. **v7: consensus pseudo-label training.** Segment Kaggle recordings, keep
   segments where v6 and Whisper (and/or Nemotron) agree closely, train on
   ~20k real segments + the synthetic mix, test feeds excluded. ~$15-20.
   Agreement reduces but doesn't remove shared errors, so it still needs (1)
   to measure.
3. **Real phrasing as TTS text**: use real transcripts as the text for
   multi-engine synthesis so synthetic data sounds like casual real traffic.
4. **Low-confidence flag for names not on the AIS list** (no GPU): unknown
   vessels can't be looked up; flag them instead of presenting a guess.
5. Optional: Nemotron on H100/H200 with the card's config (speed only), or
   with context terms as a pseudo-label second opinion.

Open questions:
- Do the 5 "empty" Kaggle clips contain speech? (sent to the user)
- Which feeds to hold out for the real test set; who does the listening.
- Licence position on the Broadcastify-derived Kaggle data for anything
  beyond internal research.
- Can the user get legally recorded audio from a receiver near a port?
  (Earlier answer: "Not yet".)

## 8. Pointers

Code (`marine_vhf_asr/`):
- `normalize.py` transcript conventions; `metrics.py` WER, slot accuracy,
  `decoy_copy_rate`, `name_cer`.
- `smcp.py` dialogue generator; `gazetteer.py` ports + `AIS_REGIONS`;
  `vhf_channel.py` radio simulation; `tts.py` Kokoro / Chatterbox / Piper /
  Parler / Sine backends; `synthesize.py` CLI.
- `sources.py` dataset specs: `vhf8k:split#none|supported|decoys|mix|mixlong|supported:N|decoys:N@limit`,
  `martts:test`, `atc:test@300`, `atcosim`, `path/manifest.jsonl#none|moving_a|all|mix`.
- `prompting.py` context prompts, guards, `verify_names`;
  `evaluate.py` (`--verify`); `train.py` (`--resume`).
- `ais.py` spoken names, snapshots, candidate lists; `harbor.py` AIS-driven
  dialogues (`--exclude-names`, `--unknown-names`, `--min-listed`).
- `jobs/`: `hf_train_job.py`, `hf_eval_job.py`, `ais_extract_job.py`,
  `harbor_tts_job.py` (+ `_chatterbox`, `_parler`), `whisper_judge_job.py`,
  `kaggle_sample_job.py`, `compare_parler_judged.py`,
  `nemotron/run.sh` + `nemotron/client.py`.
- `README.md`: every run's results tables and caveats.

HF repos (all under `mszarski`, private):
- Code mirror for jobs: `marine-vhf-asr-code` (model repo).
- Adapters: `voxtral-3B-marine-vhf` (v1), `-context` (v2), `-v3`, `-v4`,
  `-v5`, `-v6`, `-smoke`. Each has eval results, per-clip predictions under
  `diagnostics/`, and a model card.
- Datasets: `marine-vhf-ais` (multi-region AIS parquet extracts),
  `marine-vhf-ais-nyc` (first NY pull), `marine-vhf-nyc-synth` (first NY
  test audio, v3 era), `marine-vhf-ais-synth` (current: `train*`
  Kokoro/Piper(`_pp`)/Chatterbox(`_cb1/4/7`) training audio,
  `test_{ny,oresund}{,_kv,_cb,_pa0,_pa2}`, `exclude_test_names.txt`,
  `unknown_for_{ny,oresund,train}.txt`, `diagnostics/` incl. Whisper judge
  and Nemotron outputs), `marine-radio-chatter-sample` (Kaggle: feed
  listing, WhisperX transcripts tar.gz, 3 audio files per feed).
- Trackio Space `mszarski/trackio`, project `marine-vhf-asr`.

Commands:
```bash
pip install -r marine_vhf_asr/requirements.txt          # + requirements-train.txt for GPU
pytest marine_vhf_asr/tests/
hf upload mszarski/marine-vhf-asr-code marine_vhf_asr marine_vhf_asr --exclude "tests/*" --exclude "__pycache__/*"
```
v6 training launch (pattern; `{DATA}` resolves to a snapshot of `DATA_REPO`):
```bash
hf jobs uv run --flavor a100-large --timeout 8h --secrets HF_TOKEN \
  --env CODE_REPO=mszarski/marine-vhf-asr-code --env DATA_REPO=mszarski/marine-vhf-ais-synth \
  --env "DATA_PATTERNS=train_v5/manifest_k.jsonl,train_dk_v5/manifest_k.jsonl,train_v5/wav/*,train_dk_v5/wav/*,train_pp/*,train_dk_pp/*,train_cb1/*,train_dk_cb1/*,train_cb4/*,train_dk_cb4/*,train_cb7/*,train_dk_cb7/*" \
  --detach marine_vhf_asr/jobs/hf_train_job.py \
  --train vhf8k:train#mixlong vhf8k:discovery#mixlong martts:train+clean atc:train@4000 atcosim:train@2000 \
    {DATA}/train_v5/manifest_k.jsonl#mix {DATA}/train_dk_v5/manifest_k.jsonl#mix \
    {DATA}/train_pp/manifest.jsonl#mix {DATA}/train_dk_pp/manifest.jsonl#mix \
    {DATA}/train_cb{1,4,7}/manifest.jsonl#mix {DATA}/train_dk_cb{1,4,7}/manifest.jsonl#mix \
  --eval vhf8k:confirmation#mixlong@300 --lora-audio --epochs 2 --save-steps 500 \
  --output runs/voxtral-3B-marine-vhf-v6 --hub-model-id mszarski/voxtral-3B-marine-vhf-v6 \
  --trackio-project marine-vhf-asr --trackio-space mszarski/trackio
```
(expand the `{1,4,7}` lists explicitly; the original used a shell loop.)
Eval: `jobs/hf_eval_job.py` with `--env ADAPTER=... --env NAME=... --env SPECS="..."`
(see `marine_vhf_asr/jobs/hf_eval_job.py` header). Nemotron eval (Docker
image, not `uv run`; upload `jobs/nemotron` to the code repo first):
```bash
hf jobs run --flavor a100-large --timeout 3h --secrets HF_TOKEN \
  --env CODE_REPO=mszarski/marine-vhf-asr-code --env DATA_REPO=mszarski/marine-vhf-ais-synth \
  --env "DATA_PATTERNS=test_ny_cb/*,..." --env "SPECS=vhf8k:test#none@400 ... {DATA}/test_ny_cb/manifest.jsonl#moving_a ..." \
  --env REAL_REPO=mszarski/marine-radio-chatter-sample --env REAL_N=12 \
  --env OUT_REPO=mszarski/marine-vhf-ais-synth --env NAME=nemotron_eval --detach \
  ghcr.io/astral-sh/uv:python3.13-bookworm \
  bash -c 'curl -sSL -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/$CODE_REPO/resolve/main/marine_vhf_asr/jobs/nemotron/run.sh" | bash'
```
Outputs go to `OUT_REPO` under `diagnostics/<NAME>.json` and `<NAME>_predictions.jsonl`.

HF Job IDs (view at `https://huggingface.co/jobs/mszarski/<id>`):
v1 train `6ad9c5535275124786e324f2`; v2 `6ad9e76d925af0e27d4f28de`;
longer-lists eval `6ada6beac191903149a75565`; v3 `6ada7b5ec191903149a75f1e`;
NY AIS eval `6adadcd1925af0e27d4fa4fb`; v4 `6adb107dc191903149a7c413`;
v5 `6adbc31f046eb82b1ecd6e3a`, v5 eval `6adc0780046eb82b1ecd924a`;
v6 train `6adc45bf046eb82b1ecdaa35`; Kaggle sample `6adef74e046eb82b1ecec46e`;
Nemotron (successful) `6ae267ed41e90535df77cd5e`.

Costs so far (HF Jobs): v1 ~$4.50; v2 ~$5.40; v3 ~$6.40; NY AIS ~$1.70;
v4 round ~$25; v5 round $26.29; v6 round $24.59; Nemotron ~$5 incl. 4 failed
attempts. Roughly $100 total.

External resources:
- ATC reference model: https://huggingface.co/pphilip/voxtral-3B-atc-transcribe
- VHF-8K: https://huggingface.co/datasets/PassbyGrocer/VHF-8K (licence "other")
- MARTTS: https://huggingface.co/datasets/bonalor/synthetic_maritime_radio_communication (CC-BY 4.0; Chatterbox audio)
- ATC: https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset ; ATCOSIM: https://huggingface.co/datasets/jlvdoorn/atcosim
- AIS: NOAA MarineCadastre (US, public domain) and Danish Maritime Authority (web.ais.dk)
- Kaggle real radio: https://www.kaggle.com/datasets/linogova/marine-radio-chatter-bridge-2-bridge-communication
  and transcripts https://www.kaggle.com/datasets/linogova/marine-radio-chatter-bridge-to-bridge-transcript
  (collector: https://github.com/latent-variable/MaritimeCommCollector).
  Feeds include New York Sector 17329, Puget Sound, SF, Mobile, Jacksonville,
  Great Lakes / St. Lawrence. ~96k MP3s, ~3,000 h speech, ~1,300 h with WhisperX transcripts.
- Nemotron: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (OpenMDW-1.1; vLLM 0.32 + NeMo Speech plugin)
- SMCP: https://www.imo.org/en/OurWork/Safety/Pages/StandardMarineCommunicationPhrases.aspx
