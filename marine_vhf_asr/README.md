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

Unseen vessel names are the main weakness. The VHF-8K test set uses vessel
names that never appear in training (by design), and the model writes them
phonetically ("CHARIKLIA JUNIOR" → "TREKLIER JUNIOR", "LYNX SATU" → "LYNK SATU").
This is expected from audio alone. The next step is contextual biasing:
put the vessel names currently in the area (from AIS) into the prompt.
VHF-8K provides candidate lists (`ship_names`) for exactly this.

## Status

The text generator, channel simulator, normalizer, metrics, data loaders and
synthesis pipeline are tested on CPU. The full train → eval → push path has
been run on a tiny random model on CPU, and for real on HF Jobs (above).
The Kokoro TTS backend has not been run (it needs `espeak-ng`).
