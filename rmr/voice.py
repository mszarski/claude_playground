"""Voice in: audio -> transcript + how the speaker sounds, with openly licensed models that run locally on CPU.

  from rmr.voice import Listener
  heard = Listener().hear("clip.wav")     # {"text", "emotion", "probs", "arousal", "valence", "dominance", ...}

- Speech to text: Whisper (``openai/whisper-small``, Apache-2.0).
- Emotion: ``3loi/SER-Odyssey-Baseline-WavLM-Categorical`` (MIT) and ``...-Multi-Attributes`` (MIT), the Odyssey 2024
  challenge baselines, trained on MSP-Podcast (natural podcast speech) by the lab that built it. Other candidates are
  wired in for ``scripts/eval_voice_emotion.py``.

These read how someone *sounds* (expressed emotion), not what they feel, and are often wrong on natural speech
(see docs/results/voice.md): treat them as a soft cue next to the words.
"""
import io

import numpy as np

SR = 16000
CANON = ["neutral", "happy", "sad", "angry"]          # the four every candidate model can say

# model id -> (labels in model order, mapping to CANON or None)
MODELS = {
    "msp": "3loi/SER-Odyssey-Baseline-WavLM-Categorical",
    "msp-dims": "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
    "speechbrain": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    "acted": "Dpngtm/wav2vec2-emotion-recognition",
}
TO_CANON = {"neutral": "neutral", "neu": "neutral", "calm": "neutral", "happy": "happy", "hap": "happy", "joy": "happy",
            "sad": "sad", "sadness": "sad", "angry": "angry", "ang": "angry", "anger": "angry"}


def load_audio(src):
    """Path, bytes or (array, rate) -> float32 mono at 16 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    if isinstance(src, tuple):
        x, sr = src
    else:
        x, sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, dtype="float32", always_2d=True)
        x = x.mean(1)
    x = np.asarray(x, np.float32)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def _msp_model(repo):
    """The Odyssey 2024 baseline (WavLM-large + attentive statistics pooling + MLP head), rebuilt here so that
    loading it needs no remote code; the weights load strictly from the repo's safetensors."""
    import json

    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    d = cfg["hidden_size"]

    class Pool(nn.Module):           # attentive statistics pooling (Okabe et al. 2018), as in the repo's code
        def __init__(self):
            super().__init__()
            self.sap_linear = nn.Linear(d, d)
            self.attention = nn.Parameter(torch.zeros(d, 1))

        def forward(self, x):
            w = torch.softmax((torch.tanh(self.sap_linear(x)) @ self.attention).squeeze(-1), dim=1).unsqueeze(-1)
            mu = (x * w).sum(1)
            return torch.cat([mu, ((x ** 2 * w).sum(1) - mu ** 2).clamp(min=1e-5).sqrt()], -1)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(0.5))])
            self.out = nn.Sequential(nn.Linear(d, cfg["num_classes"]))

        def forward(self, x):
            for f in self.fc:
                x = f(x)
            return self.out(x)

    class SER(nn.Module):
        def __init__(self):
            super().__init__()
            self.ssl_model = AutoModel.from_config(AutoConfig.from_pretrained(cfg["ssl_type"]))
            self.pool_model, self.ser_model = Pool(), Head()

        def forward(self, wav):
            return self.ser_model(self.pool_model(self.ssl_model(wav).last_hidden_state))

    m = SER()
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "masked_spec_embed" not in k]
    if missing or unexpected:
        raise RuntimeError(f"{repo}: weights do not fit (missing {missing[:3]}, unexpected {unexpected[:3]})")
    m.eval()
    m.mean, m.std = cfg["mean"], cfg["std"]
    m.labels = [cfg["id2label"][str(i)].lower() for i in range(cfg["num_classes"])]
    return m


class EmotionModel:
    """``predict(wav16k)`` -> ``{label: probability}`` in the model's own labels (lower-case)."""

    def __init__(self, name="msp", device="cpu"):
        import torch

        self.name, self.torch, self.device = name, torch, device
        if name in ("msp", "msp-dims"):
            self.m = _msp_model(MODELS[name]).to(device)
            self.labels = self.m.labels
        elif name == "speechbrain":
            from speechbrain.inference.interfaces import foreign_class

            self.m = foreign_class(source=MODELS[name], pymodule_file="custom_interface.py",
                                   classname="CustomEncoderWav2vec2Classifier")
            self.labels = [self.m.hparams.label_encoder.decode_ndim(i) for i in range(4)]
        elif name == "acted":
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self.fe = AutoFeatureExtractor.from_pretrained(MODELS[name])
            self.m = AutoModelForAudioClassification.from_pretrained(MODELS[name]).eval()
            self.labels = [self.m.config.id2label[i].lower() for i in range(len(self.m.config.id2label))]
        else:
            raise ValueError(f"unknown emotion model {name!r}: {list(MODELS)}")

    def predict(self, wav):
        torch = self.torch
        with torch.no_grad():
            if self.name in ("msp", "msp-dims"):
                x = torch.tensor((wav - self.m.mean) / (self.m.std + 1e-6)).unsqueeze(0).to(self.device)
                out = self.m(x)[0].cpu()
                if self.name == "msp-dims":       # attributes, roughly 0..1
                    return {k: float(v) for k, v in zip(self.labels, out)}
                p = torch.softmax(out, -1)
            elif self.name == "speechbrain":
                out_prob = self.m.classify_batch(torch.tensor(wav).unsqueeze(0))[0][0]
                p = out_prob.exp() if out_prob.max() <= 0 else out_prob      # log-probabilities or probabilities
                p = p / p.sum()
            else:
                inp = self.fe(wav, sampling_rate=SR, return_tensors="pt")
                p = torch.softmax(self.m(**inp).logits[0], -1)
        return {k: float(v) for k, v in zip(self.labels, p)}

    def canon(self, probs):
        """Probabilities restricted to ``CANON`` and renormalised."""
        q = {c: 0.0 for c in CANON}
        for k, v in probs.items():
            if k in TO_CANON:
                q[TO_CANON[k]] += v
        s = sum(q.values()) or 1.0
        return {k: v / s for k, v in q.items()}


class Listener:
    """Transcript + categorical emotion + arousal / valence / dominance for one utterance."""

    def __init__(self, asr="openai/whisper-small", emotion="msp", dims=True, device="cpu"):
        from transformers import pipeline

        self.asr = pipeline("automatic-speech-recognition", model=asr, device=device)
        self.emo = EmotionModel(emotion, device=device)
        self.dims = EmotionModel("msp-dims", device=device) if dims else None

    def hear(self, src):
        wav = load_audio(src)
        text = self.asr({"raw": wav, "sampling_rate": SR}, generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()
        probs = self.emo.predict(wav)
        top = max(probs, key=probs.get)
        out = {"text": text, "emotion": top, "confidence": probs[top], "probs": probs, "seconds": len(wav) / SR}
        if self.dims:
            out.update(self.dims.predict(wav))
        return out
