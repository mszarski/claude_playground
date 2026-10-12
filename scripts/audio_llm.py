"""Audio-native responder: open audio LLMs that hear the voice itself instead of the tone models' one-line reading.

  # zero-shot: how well does each model read feeling from the voice alone? (GPU job)
  python scripts/audio_llm.py bakeoff --heard /work/eval/tone.jsonl --out /work/out/bakeoff.json \
      --models Qwen/Qwen2-Audio-7B-Instruct mistralai/Voxtral-Mini-3B-2507 microsoft/Phi-4-multimodal-instruct

The responder (rmr.respond.Student) gets the tone as text from two small tone models (rmr.voice), and on the tone test
(eval_clean: CREMA-D actors held out of training, flat sentences, so the words say nothing) it reads the feeling 42%
of the time: the tone models' readings are the ceiling. An audio LLM hears pitch, energy, hesitation and voice quality
directly. ``bakeoff`` asks each candidate (open licences only) for the feeling of each held-out clip, with the real
audio, and reports unweighted accuracy (UA) and confusions.

Audio comes from CREMA-D (mteb/crema-d, Open Database License) by file name.
"""
import argparse
import json
import os
import re
import tempfile
import time
from collections import Counter

FEELINGS5 = ["neutral", "happy", "sad", "angry", "anxious"]
ASK = ("Listen to how this person sounds: their tone of voice, not the words. Which one word best describes how "
       "they feel: neutral, happy, sad, angry, or anxious? Answer with that one word.")
SYNONYMS = {"fear": "anxious", "fearful": "anxious", "scared": "anxious", "afraid": "anxious", "nervous": "anxious",
            "worried": "anxious", "anxiety": "anxious", "anger": "angry", "annoyed": "angry", "frustrated": "angry",
            "disgust": "angry", "disgusted": "angry", "joy": "happy", "excited": "happy", "cheerful": "happy",
            "sadness": "sad", "upset": "sad", "calm": "neutral", "neutrality": "neutral"}


def crema_audio(names):
    """CREMA-D clip name (e.g. 1001_IEO_ANG_HI.wav) -> 16 kHz float32 waveform, for the requested names."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from rmr.voice import load_audio

    want, out = set(names), {}
    for i in range(2):
        pf = pq.ParquetFile(hf_hub_download("mteb/crema-d", f"data/train-0000{i}-of-00002.parquet", repo_type="dataset"))
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=["audio"]).to_pylist():
                if r["audio"]["path"] in want:
                    out[r["audio"]["path"]] = load_audio(r["audio"]["bytes"])
    return out


def parse_feeling(text):
    words = re.findall(r"[a-z]+", text.lower())
    for w in words:
        if w in FEELINGS5:
            return w
        if w in SYNONYMS:
            return SYNONYMS[w]
    return "neutral"


class AudioLM:
    """One audio LLM behind transformers' chat template: ``ask(wav, prompt) -> text``."""

    def __init__(self, repo, device="cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoModelForSeq2SeqLM, AutoProcessor  # noqa: F401
        import transformers

        self.repo, self.torch, self.device = repo, torch, device
        self.proc = AutoProcessor.from_pretrained(repo)
        cfg = transformers.AutoConfig.from_pretrained(repo)
        arch = (cfg.architectures or [""])[0]
        cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
        self.m = cls.from_pretrained(repo, dtype=torch.bfloat16).to(device).eval()

    def inputs(self, wav, prompt):
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wav, 16000)
            path = f.name
        conv = [{"role": "user", "content": [{"type": "audio", "path": path}, {"type": "text", "text": prompt}]}]
        try:
            enc = self.proc.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="pt",
                                                add_generation_prompt=True)
        finally:
            os.unlink(path)
        return enc

    def ask(self, wav, prompt, max_new_tokens=8):
        enc = self.inputs(wav, prompt).to(self.device)
        for k, v in enc.items():
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                enc[k] = v.to(self.torch.bfloat16)
        with self.torch.no_grad():
            out = self.m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def bakeoff(a):
    rows = [json.loads(x) for x in open(a.heard)]
    audio = crema_audio([r["tone"] for r in rows])
    print(f"{len(audio)} of {len(rows)} clips found", flush=True)
    results = {}
    for repo in a.models:
        t0 = time.time()
        try:
            lm = AudioLM(repo)
        except Exception as e:
            print(f"{repo}: could not load: {type(e).__name__}: {e}", flush=True)
            results[repo] = {"error": f"{type(e).__name__}: {e}"}
            continue
        got, raw = [], []
        for i, r in enumerate(rows):
            try:
                text = lm.ask(audio[r["tone"]], ASK)
            except Exception as e:
                print(f"{repo} {r['tone']}: {type(e).__name__}: {e}", flush=True)
                text = ""
                if i == 0:
                    break
            raw.append(text)
            got.append((r["label"], parse_feeling(text)))
            if i % 50 == 0:
                print(f"{repo} {i}: {r['label']} -> {text!r}", flush=True)
        acc = {c: [p == c for l, p in got if l == c] for c in FEELINGS5}
        ua = sum(sum(v) / len(v) for v in acc.values() if v) / max(1, sum(1 for v in acc.values() if v))
        res = {"ua": ua, "per_class": {c: sum(v) / len(v) for c, v in acc.items() if v}, "n": len(got),
               "confusions": {c: dict(Counter(p for l, p in got if l == c)) for c in FEELINGS5},
               "raw_examples": raw[:20], "seconds": round(time.time() - t0)}
        results[repo] = res
        print(f"== {repo}: UA {ua:.1%} " + " ".join(f"{c[:3]} {v:.0%}" for c, v in res["per_class"].items()), flush=True)
        del lm
        import torch
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bakeoff")
    b.add_argument("--heard", required=True, help="eval_clean tone.jsonl (CREMA-D file names in 'tone')")
    b.add_argument("--models", nargs="+", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"bakeoff": bakeoff}[a.cmd](a)


if __name__ == "__main__":
    main()
