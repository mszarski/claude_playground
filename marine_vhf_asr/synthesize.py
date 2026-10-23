"""
Generate a synthetic marine VHF ASR dataset.

Writes 16 kHz mono WAVs plus a JSONL manifest with one line per transmission:
    {"audio": "wav/000001_02.wav", "text": "...", "scenario": "...", "port": "...",
     "dialogue_id": 1, "turn": 2, "speaker": "shore", "voice": "...",
     "slots": {...}, "channel": {...}}

Usage:
    python -m marine_vhf_asr.synthesize --out data/synth --dialogues 2000 --tts kokoro
    python -m marine_vhf_asr.synthesize --out data/text --dialogues 100 --text-only
    python -m marine_vhf_asr.synthesize --out data/nyc_test --from-text data/nyc_test/text.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .smcp import DialogueGenerator
from .tts import BACKENDS
from .vhf_channel import ChannelConfig, simulate_vhf

TARGET_SR = 16000


def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio
    g = np.gcd(sr, TARGET_SR)
    return signal.resample_poly(audio, TARGET_SR // g, sr // g)


def _generated_records(args):
    """Records for --dialogues freshly generated SMCP dialogues."""
    gen = DialogueGenerator(seed=args.seed)
    for d_id in range(args.dialogues):
        dialogue = gen.generate()
        for t_idx, turn in enumerate(dialogue.turns):
            yield {
                "text": turn.target, "tts_text": turn.text, "scenario": dialogue.scenario,
                "port": dialogue.port, "dialogue_id": d_id, "turn": t_idx,
                "speaker": turn.speaker, "slots": turn.slots,
            }


def _text_records(path):
    """Records from a text-only manifest (e.g. harbor.py output); needs tts_text and text."""
    for line in open(path):
        rec = json.loads(line)
        rec.setdefault("speaker", rec.get("turn_speaker", "unknown"))
        yield rec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dialogues", type=int, default=100)
    parser.add_argument("--from-text", type=Path,
                        help="Synthesize a text-only manifest (e.g. from harbor.py) instead of generating dialogues")
    parser.add_argument("--tts", choices=sorted(BACKENDS), default="kokoro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-copies", action="store_true",
                        help="Also write an undegraded copy of each clip (for curriculum / debugging)")
    parser.add_argument("--text-only", action="store_true", help="Write transcripts only, no audio")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    cfg = ChannelConfig()
    tts = None if args.text_only else BACKENDS[args.tts]()
    records = _text_records(args.from_text) if args.from_text else _generated_records(args)

    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"

    n, dialogues = 0, set()
    voices = {}  # one voice per speaker role within a dialogue
    with open(manifest_path, "w") as manifest:
        for record in records:
            d_id, t_idx = record["dialogue_id"], record["turn"]
            if d_id not in dialogues:
                dialogues.add(d_id)
                voices = {}
            if tts is not None:
                voice = voices.setdefault(record["speaker"], tts.voices[rng.integers(len(tts.voices))])
                audio, sr = tts.synthesize(record["tts_text"], voice)
                audio = _to_16k(np.asarray(audio, dtype=np.float64), sr)
                degraded, params = simulate_vhf(audio, TARGET_SR, rng, cfg)
                name = f"{d_id:06d}_{t_idx:02d}.wav"
                wavfile.write(wav_dir / name, TARGET_SR, degraded)
                if args.clean_copies:
                    clean = (0.9 * audio / (np.max(np.abs(audio)) + 1e-12)).astype(np.float32)
                    wavfile.write(wav_dir / f"clean_{name}", TARGET_SR, clean)
                record.update(audio=f"wav/{name}", voice=voice, channel=params)
            manifest.write(json.dumps(record) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"  {n} transmissions", flush=True)

    print(f"Wrote {n} transmissions from {len(dialogues)} dialogues to {manifest_path}")


if __name__ == "__main__":
    main()
