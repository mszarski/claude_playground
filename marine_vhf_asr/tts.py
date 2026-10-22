"""
Pluggable text-to-speech backends for synthetic data generation.

Accent diversity matters more than voice quality here: marine VHF traffic is
heavily multinational, so use as many voices/accents as the backend offers
and mix in real recordings wherever possible.
"""

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

    pip install kokoro soundfile  (also needs espeak-ng on the system)
    """

    VOICES = {
        "a": ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael",
              "am_eric", "am_liam", "am_onyx"],
        "b": ["bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bm_george", "bm_lewis",
              "bm_daniel", "bm_fable"],
    }

    def __init__(self):
        from kokoro import KPipeline
        self._pipelines = {lang: KPipeline(lang_code=lang) for lang in self.VOICES}
        self.voices = [v for vs in self.VOICES.values() for v in vs]

    def synthesize(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        pipeline = self._pipelines[voice[0]]
        chunks = [np.asarray(audio) for _, _, audio in pipeline(text, voice=voice)]
        return np.concatenate(chunks), 24000


class SineBackend:
    """
    Dependency-free placeholder that emits tones instead of speech.

    Only for exercising the pipeline end to end (e.g. in tests or CI);
    never train on its output.
    """

    voices = ["tone_low", "tone_high"]

    def synthesize(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        sr = 16000
        f0 = 140.0 if voice == "tone_low" else 220.0
        t = np.arange(int(sr * 0.06 * max(1, len(text.split())))) / sr
        return 0.5 * np.sin(2 * np.pi * f0 * t * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))), sr


BACKENDS = {"kokoro": KokoroBackend, "sine": SineBackend}
