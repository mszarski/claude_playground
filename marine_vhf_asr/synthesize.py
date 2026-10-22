"""
Generate a synthetic marine VHF ASR dataset.

Writes 16 kHz mono WAVs plus a JSONL manifest with one line per transmission:
    {"audio": "wav/000001_02.wav", "text": "...", "scenario": "...", "port": "...",
     "dialogue_id": 1, "turn": 2, "speaker": "shore", "voice": "...",
     "slots": {...}, "channel": {...}}

Usage:
    python -m marine_vhf_asr.synthesize --out data/synth --dialogues 2000 --tts kokoro
    python -m marine_vhf_asr.synthesize --out data/text --dialogues 100 --text-only
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dialogues", type=int, default=100)
    parser.add_argument("--tts", choices=sorted(BACKENDS), default="kokoro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-copies", action="store_true",
                        help="Also write an undegraded copy of each clip (for curriculum / debugging)")
    parser.add_argument("--text-only", action="store_true", help="Write transcripts only, no audio")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    gen = DialogueGenerator(seed=args.seed)
    cfg = ChannelConfig()
    tts = None if args.text_only else BACKENDS[args.tts]()

    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"

    n = 0
    with open(manifest_path, "w") as manifest:
        for d_id in range(args.dialogues):
            dialogue = gen.generate()
            # One voice per speaker role within a dialogue
            voices = {}
            for t_idx, turn in enumerate(dialogue.turns):
                record = {
                    "text": turn.target,
                    "tts_text": turn.text,
                    "scenario": dialogue.scenario,
                    "port": dialogue.port,
                    "dialogue_id": d_id,
                    "turn": t_idx,
                    "speaker": turn.speaker,
                    "slots": turn.slots,
                }
                if tts is not None:
                    voice = voices.setdefault(turn.speaker, tts.voices[rng.integers(len(tts.voices))])
                    audio, sr = tts.synthesize(turn.text, voice)
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

    print(f"Wrote {n} transmissions from {args.dialogues} dialogues to {manifest_path}")


if __name__ == "__main__":
    main()
