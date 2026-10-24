# marine_vhf_asr

Tooling to build a marine VHF radio transcription model: port control, VTS,
pilots, coastguard and ship-to-ship traffic. It follows the same approach as
[`pphilip/voxtral-3B-atc-transcribe`](https://huggingface.co/pphilip/voxtral-3B-atc-transcribe),
which does this for air traffic control (ATC).

## The reference model

| | ATC model |
|---|---|
| Base | `mistralai/Voxtral-Mini-3B-2507` |
| Data | `jacktol/ATC-ASR-Dataset` (~8k clips: UWB ATC + ATCO2, real VHF) + `jlvdoorn/atcosim` (clean, simulated) |
| Compute | 1× RTX 3090 Ti 24 GB, ~7 epochs, ~23 h |
| Result | 8.0% WER on noisy real ATC (base model: 105.6%); 1.3% on ATCOSIM |
| License | Apache 2.0 |

The base model gets over 100% WER, so it is essentially unusable on raw
radio. Most of the gain comes from in-domain data: radio channel noise, a
fixed phraseology, spelled-out numbers and the phonetic alphabet.

## Can we do the same for marine radio?

**Yes. The model side carries over directly; the data is the hard part.**

What carries over:
- Marine VHF is the same kind of audio as ATC: narrowband FM voice in the
  300–3000 Hz band, squelch, clipping, push-to-talk transmissions of a few
  seconds, and heavily accented English.
- The phraseology is just as formulaic. The IMO
  [Standard Marine Communication Phrases (SMCP)](https://www.imo.org/en/OurWork/Safety/Pages/StandardMarineCommunicationPhrases.aspx)
  play the role of ICAO phraseology: digits spoken one by one, the phonetic
  alphabet, message markers (INSTRUCTION, ADVICE, QUESTION, ...), and
  MAYDAY / PAN PAN / SECURITE.
- The same base model and LoRA fine-tune fit on one 24 GB GPU.

What doesn't carry over:
- **There is no public corpus of real, transcribed marine radio.** The real
  recordings people have used (Fraunhofer marFM, about 56 h; the Hamburg
  port study; the Irish Coast Guard work) are proprietary. The open marine
  data is synthetic (see *Datasets* below), so we combine it with real VHF
  acoustics from the ATC corpora.
- **Vocabulary is local.** Vessel names, port and berth names, and VTS
  sector names vary by region. The model can only learn names it has heard,
  so `gazetteer.py` has to be filled in for the ports you care about.
- **Legal and privacy.** In many jurisdictions, recording and redistributing
  VHF traffic falls under secrecy-of-correspondence rules (for example the
  UK Wireless Telegraphy Act), and in the EU voices and names are subject to
  GDPR. Check this before you collect real recordings.

## Datasets

| Source spec | Dataset | What it is | License |
|---|---|---|---|
| `vhf8k` | [PassbyGrocer/VHF-8K](https://huggingface.co/datasets/PassbyGrocer/VHF-8K) | 8k synthetic VTS transmissions (13.5 h, one TTS voice, 4 VHF profiles), vessel-name annotations | other (check upstream terms) |
| `martts` | [bonalor/synthetic_maritime_radio_communication](https://huggingface.co/datasets/bonalor/synthetic_maritime_radio_communication) (MARTTS) | 240 synthetic distress / SAR dialogues, VCTK-cloned accents, real marine noise; one 30–125 s clip per dialogue | CC-BY 4.0 |
| `atc` | [jacktol/ATC-ASR-Dataset](https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset) | 8k **real** accented ATC VHF clips | not stated |
| `atcosim` | [jlvdoorn/atcosim](https://huggingface.co/datasets/jlvdoorn/atcosim) | 10k clean simulated ATC clips | ATCOSIM terms |
| `path.jsonl` | `synthesize.py` output or your own recordings | SMCP dialogues for your ports | yours |

Specs take a split and an optional cap, e.g. `vhf8k:train@2000`,
`martts:train+clean`. The loaders convert each source to our transcript
conventions:
- VHF-8K writes numbers as digits ("8 knots", "MERLION 31"). They are read as
  cardinals, as a TTS voice speaks them. Transcripts with ambiguous numbers
  (3+ digit runs, coordinates, times; about 6%) are dropped.
- MARTTS ships the spoken-form text ("M M S I - number two six ..."). Spaced
  letters are joined into acronyms (`MMSI`).
- MARTTS scenarios 9–10 of each topic are held out as `martts:test`. Its a/b/c
  variants share a script, so splitting by scenario avoids leaking text.

## Plan

1. **Bootstrap (done here).** Fine-tune on VHF-8K + MARTTS + ATC.
2. **Local vocabulary.** Fill in `gazetteer.py` for your ports and generate
   SMCP dialogues with `synthesize.py`. TTS with many voices, then
   `vhf_channel.py` simulates the radio: mic background, band-pass, limiter,
   receiver hiss at 3–30 dB SNR, squelch bursts, fades.
3. **Real marine data (needed for a usable model).** Record a few hours from
   a receiver near your target port, with permission. Have the step-1 model
   pre-transcribe it, then correct the output by hand. Keep a held-out real
   test set; synthetic WER will be optimistic.
4. **Evaluate on slot accuracy, not just WER.** One wrong digit in a channel
   or position is a safety problem, even though it barely moves WER.

## Usage

```bash
pip install -r marine_vhf_asr/requirements.txt          # generation + metrics
pip install -r marine_vhf_asr/requirements-train.txt    # GPU training / eval
pip install kokoro soundfile                            # optional TTS (needs espeak-ng)

# Look at the generated phraseology
python -m marine_vhf_asr.synthesize --out data/text --dialogues 50 --text-only

# Train on a local GPU (24 GB: use --batch-size 2 --grad-accum 8)
python -m marine_vhf_asr.train \
    --train vhf8k:train vhf8k:discovery martts:train+clean atc:train@4000 atcosim:train@2000 \
    --eval vhf8k:confirmation@300 \
    --wer-eval vhf8k:test@400 martts:test atc:test@300 --baseline \
    --lora-audio --output runs/voxtral-marine

python -m marine_vhf_asr.evaluate --data vhf8k:test@400 martts:test \
    --adapter runs/voxtral-marine/adapter

pytest marine_vhf_asr/tests/
```

### On Hugging Face Jobs (no local GPU)

The job container has no checkout of this repository, so upload the package to
a private Hub repo, then point the job entry script at it:

```bash
hf upload <user>/marine-vhf-asr-code marine_vhf_asr marine_vhf_asr --private \
    --exclude "tests/*" --exclude "__pycache__/*"
hf jobs uv run --flavor a100-large --timeout 5h --secrets HF_TOKEN \
    --env CODE_REPO=<user>/marine-vhf-asr-code --detach \
    marine_vhf_asr/jobs/hf_train_job.py \
    --train vhf8k:train vhf8k:discovery martts:train+clean atc:train@4000 atcosim:train@2000 \
    --eval vhf8k:confirmation@300 --wer-eval vhf8k:test@400 martts:test atc:test@300 --baseline \
    --lora-audio --hub-model-id <user>/voxtral-3B-marine-vhf \
    --trackio-project marine-vhf-asr --trackio-space <user>/trackio
```

The job pushes checkpoints, the final adapter, `eval_results.json`, per-example
predictions and a model card to `--hub-model-id` (private).

Real recordings use the same JSONL format: `{"audio": "relative/path.wav",
"text": "..."}`. `slots` is optional.

## Transcript conventions

These match the ATC datasets so the two can be mixed:
- Uppercase, with no punctuation.
- Numbers are written as spoken. On the radio that usually means one digit
  at a time (`16` → `ONE SIX`, the default). Use `numbers="cardinal"` for
  sources where numbers were read as words (`16` → `SIXTEEN`). Digits after a
  decimal point are always spelled one by one.
- Isolated letters become phonetic words (`P` → `PAPA`), except `A` and `I`.
  With `acronyms=True`, spaced letters are joined instead (`M M S I` → `MMSI`).
- `PAN-PAN` → `PAN PAN`, `SÉCURITÉ`/`SECURITAY` → `SECURITE`,
  `NINER` → `NINE`.

## Results (first run)

LoRA r=32 on the LM and audio encoder, 10,958 clips
(`vhf8k:train` + `vhf8k:discovery` + `martts:train+clean` + `atc:train@4000` +
`atcosim:train@2000`), 2 epochs, 1× A100, 1.8 h (~$4.50 on HF Jobs).
The adapter is at `mszarski/voxtral-3B-marine-vhf` (private).

| Held-out set | Base Voxtral WER | Fine-tuned WER | Vessel names correct |
|---|---|---|---|
| VHF-8K test (400) | 28.1% | **12.6%** | 23.3% → 34.6% |
| MARTTS test (48 dialogues) | 15.3% | **7.5%** | – |
| ATC test, real radio (300) | 69.1% | **9.8%** | – |

WER here is scored after normalizing both sides to the same format, so it is
not directly comparable with the 105.6% baseline on the ATC model's card. For
reference, the ATC model reports 8.0% on the full ATC test set.

Unseen vessel names are the main weakness without context. The VHF-8K test set
uses vessel names that never appear in training (by design), and the model
writes them phonetically ("CHARIKLIA JUNIOR" → "TREKLIER JUNIOR").

## Results: vessel-name context (second run)

Same data and settings, but VHF-8K clips are trained with a candidate list in
the prompt (`#mix`). Adapter: `mszarski/voxtral-3B-marine-vhf-context` (private).
Both models are scored with the current evaluation (output-length cap and
repetition-loop guard, VHF-8K letters kept as letters), on the same clips.

| VHF-8K test (400 clips) | v1 WER | v1 names | v2 WER | v2 names | v2 decoy copied |
|---|---|---|---|---|---|
| No list | 12.6% | 34.8% | 13.2% | 33.6% | – |
| Spoken name + 4 decoys | 10.9% | 65.7% | **1.9%** | **97.3%** | 0.5% |
| Decoys only (name absent) | 19.0% | 34.1% | 13.6% | 32.8% | 3.8% |

| Other held-out sets | v1 WER | v2 WER |
|---|---|---|
| MARTTS test (48 dialogues) | 7.8% | 8.7% |
| ATC test, real radio (300) | 9.6% | 9.9% |

- With the right vessel in the list, v2 gets 97% of names right and the overall
  WER drops to 1.9%. It almost never copies a decoy (0.5% of clips).
- When the list does not contain the spoken name, v2 copes much better than v1
  (13.6% vs 19.0% WER). It still writes a decoy name in 3.8% of clips; a
  downstream check against AIS position would catch most of these.
- Without a list, v2 is on par with v1 (13.2% vs 12.6%: 54 clips worse, 44 better).
  MARTTS and ATC moved by under a point, which is within noise at these sample sizes.
- The first v2 evaluation hit one clip stuck repeating "VTS VTS VTS ...", which
  added 94 errors. That is why `evaluate.transcribe` now cuts repetition loops
  (a 1-4 word phrase repeated 8+ times) and caps output length by clip duration.
  Neither guard alters any reference transcript in the train or test sets.

In real use, the list would come from AIS: vessels within radio range of the
receiver, typically more than the 5 used here. Testing with longer lists is a
sensible next step.

## Results: realistic candidate lists (third run)

v2's lists had at most ~10 names and the speaker was missing only 15% of the
time. With longer lists that lack the speaker, v2 often wrote some list name
anyway, usually an unrelated one ("CHARIKLIA JUNIOR" → "TURTLE BUDDY II").
v3 trains on lists of 5-100 names with the speaker missing 30% of the time
(`#mixlong`). Adapter: `mszarski/voxtral-3B-marine-vhf-v3` (private).

"Wrong name" = an unspoken list name appears in the transcript.
"Verified" = `evaluate.transcribe_verified`: a second pass without the list,
keeping a list name only if something similar was heard (2x inference).

| VHF-8K test (400 clips) | v2 names / wrong | v3 names / wrong | v3 verified names / wrong |
|---|---|---|---|
| No list | 34% / – | 33% / – | – |
| Speaker on 10-name list | 97% / 0.8% | 94% / 0.5% | 92% / 0.0% |
| Speaker on 25-name list | 97% / 1.5% | 93% / 0.2% | 91% / 0.0% |
| Speaker on 100-name list | 91% / 4.0% | 90% / 1.5% | 88% / 1.2% |
| Speaker **not** on 25-name list | – / 16.0% | – / **3.8%** | – / **2.0%** |
| Speaker **not** on 100-name list | – / 21.5% | – / **4.8%** | – / **3.5%** |

| WER | v2 | v3 |
|---|---|---|
| VHF-8K, no list | 13.2% | 12.9% |
| VHF-8K, speaker on 100-name list | 3.4% | 3.2% |
| VHF-8K, speaker not on 100-name list | 15.4% | 13.2% |
| MARTTS test | 8.7% | 7.6% |
| ATC test, real radio | 9.9% | 9.7% |

v3 trades about 3 points of name accuracy on short lists for 4-5x fewer wrong
names when the speaker is not listed, and is as good or better everywhere
else. It is the recommended adapter. Verification removes about a third to a
half of the remaining wrong names at 2x inference cost; worth it where a
wrong vessel name matters more than latency.

## Results: New York Harbor with real AIS (v3, synthetic audio)

Test set: 300 dialogues / 1,143 transmissions built by `harbor.py` from real
NOAA AIS for NY Harbor on three held-out days (2024-07-16, 2024-10-08,
2024-12-03), receiver at VTS New York (Staten Island), Kokoro TTS + VHF channel
simulation. Real vessel names; about a quarter of speakers are small craft.

What a receiver hears (10-minute window, 20 nm): median 353 vessels; class A
transponders that moved: median 85. Those are the two candidate lists tested.

| Condition | WER | Vessel names correct (listed) | (not listed) | Wrong list name |
|---|---|---|---|---|
| No list | 17.1% | – | 40% | – |
| Moving class A list (~100 names) | **7.5%** | **86%** | 40% | 0.9% |
| Everything in range (~380 names) | **7.6%** | **80%** | – | 1.0% |
| Moving class A + verification | 8.1% | 84% | 40% | 0.7% |
| Everything + verification | 8.1% | 78% | – | 0.7% |

- Real AIS lists work, and a ~380-name list (4x longer than any in training)
  is almost as good as the filtered ~100-name list.
- Wrong names stay under 1% even with 380 names, so verification is not
  worth its cost here (it loses ~2 points of names and 0.5 points of WER).
- No-list WER is higher than on VHF-8K (17% vs 13%): US vessel names are new
  to the model, and about 2.5 points come from letter-spacing conventions
  ("M S C" vs "MSC").
- Failure modes seen: spoken letters written as NATO words ("L Y N N" →
  "LIMA YANKEE NINE", learned from ATC data where letters are spoken that
  way); concatenated AIS names ("OCEANQUEENROCKSTAR"); occasional ATC phrasing
  on short calls. The test set also used a since-fixed rule that spelled
  "LYNN" as letters.
- 85% of the vessel names mentioned on the test days also appear in AIS on
  other days (71% for small craft): a harbor's fleet is fairly stable, so
  training on the port's own AIS-driven dialogues should help, including for
  vessels that are not on the candidate list.

## Results: general model across regions (v4, partial)

Goal: one model for any port and any vessel, including vessels on no list.
Real AIS from 9 regions (NOAA: New York, Houston, LA/Long Beach, San Francisco,
Puget Sound, New Orleans, Port Arthur; Danish Maritime Authority: Great Belt,
Oresund). v4 trains on the general mix plus `harbor.py` dialogues from 7
regions (11k transmissions). **New York and Oresund are held out**, and every
vessel name seen in their AIS is excluded from training (6% overlap removed).
About 17% of test speakers are "unknown" vessels: on no candidate list and
never seen in training (NY test uses Oresund names, and vice versa).

The v4 run timed out (long candidate lists slowed training ~2x) and its
step-2000 checkpoint upload failed, so these numbers are from **step 1500 of
2756** (1.1 epochs). Fixes are in `train.py` (list cap, adapter pushed before
eval, resumable checkpoints).

| Held-out test, no list | v3 WER | v4@1500 WER | v3 names / CER | v4@1500 names / CER |
|---|---|---|---|---|
| New York, all speakers | 16.1% | **4.7%** | 45% / 0.16 | **69% / 0.05** |
| New York, unknown vessels | 19.7% | 6.7% | 23% / 0.26 | **56% / 0.09** |
| Oresund, all speakers | 22.4% | **5.6%** | 26% / 0.26 | **61% / 0.07** |
| Oresund, unknown vessels | 17.7% | 3.4% | 43% / 0.20 | **73% / 0.06** |

| With candidate list | v3 WER / names | v4@1500 WER / names |
|---|---|---|
| New York, moving class A (~100) | 7.7% / 76% | 1.8% / 88% |
| New York, all in range (~380) | 8.5% / 76% | 1.8% / 89% |
| Oresund, moving class A (~25) | 10.6% / 76% | 2.2% / 89% |
| Oresund, all in range (~260) | 10.4% / 73% | 1.4% / 94% |

Regressions vs v3 on the general sets:

| | v3 | v4@1500 |
|---|---|---|
| VHF-8K, speaker not on 25-name list: wrong name used | 3.8% | **13.5%** |
| VHF-8K, speaker on 25-name list: names | 93% | 96% |
| VHF-8K no list WER | 12.9% | 13.2% |
| MARTTS WER | 7.6% | 9.8% |
| ATC (real radio) WER | 9.7% | 10.8% |

Caveats:
- Train and test dialogues come from the same generator, Kokoro voices and
  channel simulator. Much of the drop in overall WER is the model learning
  those templates and voices, and will not carry over to real radio. The name
  metrics are more meaningful (the names are genuinely unseen), but even they
  benefit from the TTS pronouncing spellings consistently.
- The VHF-8K wrong-name regression matters for safety; v4 saw fewer
  "speaker not on the list" examples relative to list examples than v3.
- Step 1500 is mid-schedule; MARTTS/ATC may recover with full training.

## Results: v5 and an honest test (different TTS engine)

v5 fixes v4's problems: candidate lists capped at 120 names in training,
the speaker removed from 50% of AIS lists (v4: 30%), the adapter pushed before
evaluation, resumable checkpoints. AIS training audio uses Kokoro *training*
voices only. Two versions of the held-out New York / Oresund test audio:
- `_kv`: Kokoro, five voices never used in training
- `_cb`: **Chatterbox**, a different TTS engine cloning LibriSpeech speakers,
  never used in training

Adapter: `mszarski/voxtral-3B-marine-vhf-v5` (private). 2 epochs, 22k clips,
4.9 h on one A100; final eval loss 0.160 (v3: 0.169).

| Held-out test: WER / names right | v3 | v5 |
|---|---|---|
| New York, Kokoro held-out voices, no list | 16.6% / 43% | 5.2% / 68% |
| Oresund, Kokoro held-out voices, no list | 22.3% / 27% | 6.1% / 60% |
| **New York, Chatterbox, no list** | 16.3% / 33% | **11.9% / 39%** |
| **Oresund, Chatterbox, no list** | 21.3% / 22% | **14.0% / 29%** |
| New York, Chatterbox, moving class A list | 8.8% / 74% | 6.3% / 71% |
| Oresund, Chatterbox, moving class A list | 11.5% / 72% | 6.3% / 74% |

Unknown vessels (on no list, never seen), Chatterbox audio:

| names right / name CER | v3 | v5 |
|---|---|---|
| New York, no list | 19% / 0.29 | 20% / 0.25 |
| Oresund, no list | 30% / 0.20 | 39% / 0.17 |

Regression checks (v4 regressed on the first row):

| | v3 | v4@1500 | v5 |
|---|---|---|---|
| VHF-8K, speaker not on 25-name list: wrong name used | 3.8% | 13.5% | **4.0%** |
| VHF-8K, speaker on 25-name list: names right | 93% | 96% | 95% |
| VHF-8K no list WER | 12.9% | 13.2% | 12.7% |
| MARTTS WER | 7.6% | 9.8% | 7.5% |
| ATC (real radio) WER | 9.7% | 10.8% | 9.4% |

What this shows:
- **Most of the dramatic v4/v5 gain was Kokoro-specific.** Holding out Kokoro
  voices is not enough; the model learns how Kokoro pronounces spellings. On
  Chatterbox the gain shrinks to ~30% relative WER and +6-9 points of names
  without a list, and names with a list are flat.
- v5 is still the best model overall: better than v3 on every held-out
  Chatterbox WER, no regressions on VHF-8K, MARTTS or the real ATC radio, and
  v4's wrong-name regression is gone.
- **Unknown vessels remain the open problem**: 20-39% of names exactly right
  on Chatterbox audio (CER 0.17-0.25). Audio alone gives roughly a phonetic
  spelling; getting names right needs a list, and a vessel that is not on any
  list cannot be looked up.
- Every number here is still synthetic speech. Real recordings are the only
  way to know real-world accuracy; the Chatterbox rows are the better guide.

## Status

The text generator, channel simulator, normalizer, metrics, data loaders and
synthesis pipeline are tested on CPU. The full train → eval → push path has
been run on a tiny random model on CPU, and for real on HF Jobs (above).
The Kokoro TTS backend has not been run (it needs `espeak-ng`).
