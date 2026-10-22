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
- **There is no public marine equivalent of the ATC corpora.** The real
  recordings people have used (Fraunhofer marFM, about 56 h; the Hamburg
  port study; the Irish Coast Guard work) are proprietary. The closest open
  resource is DLR's synthetic **MARTTS** corpus (240 dialogues, Zenodo,
  March 2026). So this package generates synthetic data and borrows real VHF
  acoustics from the ATC dataset.
- **Vocabulary is local.** Vessel names, port and berth names, and VTS
  sector names vary by region. The model can only learn names it has heard,
  so `gazetteer.py` has to be filled in for the ports you care about.
- **Legal and privacy.** In many jurisdictions, recording and redistributing
  VHF traffic falls under secrecy-of-correspondence rules (for example the
  UK Wireless Telegraphy Act), and in the EU voices and names are subject to
  GDPR. Check this before you collect real recordings.

## Plan

1. **Text.** `smcp.py` generates SMCP-style dialogues for 19 real ports,
   covering VTS entry reports, pilot requests, berthing, MAYDAY, PAN PAN,
   SECURITE, ship-to-ship passing and radio checks. Each one carries labelled
   *slots*: vessel, channel, position, MMSI, call sign, ETA, berth.
2. **Audio.** `synthesize.py` runs TTS with multiple voices, then
   `vhf_channel.py` simulates the radio: mic background, band-pass,
   limiter, receiver hiss at 3–30 dB SNR, squelch bursts, fades, and an
   optional 8 kHz recorder path.
3. **Real VHF acoustics.** `train.py --atc-fraction 0.3` mixes in
   `jacktol/ATC-ASR-Dataset`. Its transcripts already use our conventions
   (uppercase, digits spelled out, phonetic letters).
4. **Real marine data (needed for a usable model).** Record a few hours from
   a scanner or AIS-linked receiver near your target port, with permission,
   and transcribe it. Bootstrapping with the stage-3 model and correcting its
   output by hand is much faster than transcribing from scratch. Keep a
   held-out real test set; synthetic WER will be optimistic.
5. **Train.** LoRA on Voxtral Mini 3B. Add `--lora-audio` to also adapt the
   audio encoder to the channel.
6. **Evaluate.** `evaluate.py` reports WER, WER per scenario, and **slot
   accuracy**. On the radio, one wrong digit in a channel or position is a
   safety problem, even though it barely moves WER.

## Usage

```bash
pip install -r marine_vhf_asr/requirements.txt          # generation + metrics
pip install kokoro soundfile                            # TTS (needs espeak-ng)
pip install -r marine_vhf_asr/requirements-train.txt    # GPU training / eval

# Look at the generated phraseology
python -m marine_vhf_asr.synthesize --out data/text --dialogues 50 --text-only

# Synthetic train / test sets (different seeds)
python -m marine_vhf_asr.synthesize --out data/synth_train --dialogues 4000 --tts kokoro --seed 0
python -m marine_vhf_asr.synthesize --out data/synth_test  --dialogues 300  --tts kokoro --seed 1

# Baseline: untrained Voxtral
python -m marine_vhf_asr.evaluate --manifest data/synth_test/manifest.jsonl --limit 200

# Fine-tune (single 24 GB GPU)
python -m marine_vhf_asr.train --train data/synth_train/manifest.jsonl \
    --eval data/synth_test/manifest.jsonl --atc-fraction 0.3 --lora-audio \
    --output runs/voxtral-marine

python -m marine_vhf_asr.evaluate --manifest data/synth_test/manifest.jsonl \
    --adapter runs/voxtral-marine/adapter

pytest marine_vhf_asr/tests/
```

Real recordings use the same JSONL format: `{"audio": "relative/path.wav",
"text": "..."}`. `scenario` and `slots` are optional.

## Transcript conventions

These match the ATC datasets so the two can be mixed:
- Uppercase, with no punctuation.
- Digits are spelled one at a time (`16` → `ONE SIX`). Number words as
  spoken are kept (`SIXTEEN`).
- Isolated letters become phonetic words (`P` → `PAPA`), except `A` and `I`.
- `PAN-PAN` → `PAN PAN`, `SÉCURITÉ`/`SECURITAY` → `SECURITE`,
  `NINER` → `NINE`.

## Status

The text generator, channel simulator, normalizer, metrics and synthesis
pipeline are tested on CPU. The training collator and LoRA target patterns
have been checked against the real Voxtral processor and model layout.
These steps have not been run yet:
- a full training run (needs a GPU)
- the Kokoro backend (needs `espeak-ng`)
