"""
Pluggable text-to-speech backends for synthetic data generation.

Accent diversity matters more than voice quality here: marine VHF traffic is
heavily multinational, so use as many voices/accents as the backend offers
and mix in real recordings wherever possible.
"""

import io
import random
from typing import List, Protocol, Tuple

import numpy as np


class TTSBackend(Protocol):
    voices: List[str]

    def synthesize(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        """Return (mono float waveform, sample rate)."""
        ...


class KokoroBackend:
    """
    Kokoro-82M (Apache 2.0), American and British English voices.

    pip install kokoro soundfile  (Python <= 3.12)

    Args:
        voice_set: "all", "train" or "test". TEST_VOICES are never used for
            training data, so test audio can come from unseen speakers.
    """

    VOICES = {
        "a": ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael",
              "am_eric", "am_liam", "am_onyx"],
        "b": ["bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bm_george", "bm_lewis",
              "bm_daniel", "bm_fable"],
    }
    TEST_VOICES = {"af_nicole", "am_onyx", "am_liam", "bf_lily", "bm_fable"}

    def __init__(self, voice_set: str = "all"):
        from kokoro import KPipeline
        self._pipelines = {lang: KPipeline(lang_code=lang) for lang in self.VOICES}
        voices = [v for vs in self.VOICES.values() for v in vs]
        if voice_set == "train":
            voices = [v for v in voices if v not in self.TEST_VOICES]
        elif voice_set == "test":
            voices = [v for v in voices if v in self.TEST_VOICES]
        self.voices = voices

    def synthesize(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        pipeline = self._pipelines[voice[0]]
        chunks = [np.asarray(audio) for _, _, audio in pipeline(text, voice=voice)]
        return np.concatenate(chunks), 24000


class ChatterboxBackend:
    """
    Chatterbox TTS (MIT), cloning real speakers from short reference clips.

    A different engine from Kokoro, for test audio: a model trained on Kokoro
    output should not get credit for learning Kokoro's pronunciations.
    Voices are LibriSpeech reference clips (one per speaker, >= 5 s).

    pip install chatterbox-tts datasets soundfile
    """

    def __init__(self, voice_set: str = "all", n_speakers: int = 12):
        import soundfile as sf
        from chatterbox.tts import ChatterboxTTS
        from datasets import Audio, load_dataset

        import perth
        if perth.PerthImplicitWatermarker is None:
            # The neural watermarker's native backend failed to import; the
            # watermark is irrelevant here, so use the library's no-op one
            perth.PerthImplicitWatermarker = perth.DummyWatermarker
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(decode=False))
        refs, seen = [], set()
        for row in ds:
            if row["speaker_id"] in seen:
                continue
            data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            if len(data) / sr < 5:
                continue
            path = f"/tmp/chatterbox_ref_{row['speaker_id']}.wav"
            sf.write(path, data, sr)
            refs.append(path)
            seen.add(row["speaker_id"])
            if len(refs) == n_speakers:
                break
        self.voices = refs

    def synthesize(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        wav = self.model.generate(text, audio_prompt_path=voice,
                                  exaggeration=random.uniform(0.4, 0.7), cfg_weight=random.uniform(0.35, 0.6))
        return wav.squeeze(0).cpu().numpy(), self.model.sr


class SineBackend:
    """
    Dependency-free placeholder that emits tones instead of speech.

    Only for exercising the pipeline end to end (e.g. in tests or CI);
    never train on its output.
    """

    voices = ["tone_low", "tone_high"]

    def __init__(self, voice_set: str = "all"):
        pass

    def synthesize(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        sr = 16000
        f0 = 140.0 if voice == "tone_low" else 220.0
        t = np.arange(int(sr * 0.06 * max(1, len(text.split())))) / sr
        return 0.5 * np.sin(2 * np.pi * f0 * t * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))), sr


BACKENDS = {"kokoro": KokoroBackend, "chatterbox": ChatterboxBackend, "sine": SineBackend}
