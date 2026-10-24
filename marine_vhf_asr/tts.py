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


class PiperBackend:
    """
    Piper (MIT; voices under their own licenses, see rhasspy/piper-voices):
    fast ONNX TTS, many English voices, including multi-speaker models
    (VCTK, LibriTTS, L2-ARCTIC non-native English speakers).

    pip install piper-tts huggingface_hub
    """

    MODELS = {  # model -> speaker ids to use (None: single-speaker)
        "en_US/l2arctic/medium/en_US-l2arctic-medium": range(24),
        "en_GB/vctk/medium/en_GB-vctk-medium": range(0, 109, 6),
        "en_US/libritts/high/en_US-libritts-high": range(0, 900, 60),
        "en_US/arctic/medium/en_US-arctic-medium": range(18),
        "en_GB/alan/medium/en_GB-alan-medium": None,
        "en_GB/northern_english_male/medium/en_GB-northern_english_male-medium": None,
        "en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium": None,
        "en_US/joe/medium/en_US-joe-medium": None,
        "en_US/kristin/medium/en_US-kristin-medium": None,
        "en_US/kusal/medium/en_US-kusal-medium": None,
        "en_US/hfc_female/medium/en_US-hfc_female-medium": None,
    }

    def __init__(self, voice_set: str = "all"):
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        self._models, self.voices = {}, []
        for model, speakers in self.MODELS.items():
            path = hf_hub_download("rhasspy/piper-voices", f"en/{model}.onnx")
            hf_hub_download("rhasspy/piper-voices", f"en/{model}.onnx.json")
            self._models[model] = PiperVoice.load(path)
            self.voices += [f"{model}#{s}" for s in speakers] if speakers is not None else [model]

    def synthesize(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        from piper import SynthesisConfig

        model, _, speaker = voice.partition("#")
        v = self._models[model]
        cfg = SynthesisConfig(speaker_id=int(speaker) if speaker else None,
                              length_scale=random.uniform(0.85, 1.15))
        audio = np.concatenate([c.audio_float_array for c in v.synthesize(text, syn_config=cfg)])
        return audio, v.config.sample_rate


class ParlerBackend:
    """
    Parler-TTS Mini v1.1 (Apache 2.0): voices from text descriptions plus 34
    named speakers. Used for held-out test audio: no training data uses it.

    pip install git+https://github.com/huggingface/parler-tts.git
    """

    SPEAKERS = ["Jon", "Lea", "Gary", "Jenna", "Mike", "Laura", "Lauren", "Eileen", "Alisa", "Karen",
                "Barbara", "Carol", "Emily", "Rose", "Will", "Patrick", "Eric", "Rick", "Anna", "Tina",
                "Jordan", "Brenda", "David", "Jerry"]

    def __init__(self, voice_set: str = "all"):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        repo = "parler-tts/parler-tts-mini-v1.1"
        self.device = "cuda"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float16).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.voices = self.SPEAKERS

    def synthesize(self, text: str, voice: str) -> Tuple[np.ndarray, int]:
        import torch

        pace = random.choice(["slightly fast", "moderate", "slightly slow", "fast"])
        desc = (f"{voice} speaks at a {pace} pace with a {random.choice(['low', 'moderate', 'high'])} "
                f"pitch, in a very clear recording with almost no background noise.")
        d = self.desc_tok(desc, return_tensors="pt").to(self.device)
        p = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                      prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        return out.to(torch.float32).cpu().numpy().squeeze(), self.model.config.sampling_rate


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


BACKENDS = {"kokoro": KokoroBackend, "chatterbox": ChatterboxBackend, "piper": PiperBackend,
            "parler": ParlerBackend, "sine": SineBackend}
