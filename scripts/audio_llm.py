"""Audio-native responder: open audio LLMs that hear the voice itself instead of the tone models' one-line reading.

  # zero-shot: how well does each model read feeling from the voice alone? (GPU job)
  python scripts/audio_llm.py bakeoff --heard /work/eval/tone.jsonl --out /work/out/bakeoff.json \
      --models Qwen/Qwen2-Audio-7B-Instruct mistralai/Voxtral-Mini-3B-2507 microsoft/Phi-4-multimodal-instruct

The responder (rmr.respond.Student) gets the tone as text from two small tone models (rmr.voice), and on the tone test
(eval_clean: CREMA-D actors held out of training, flat sentences, so the words say nothing) it reads the feeling 42%
of the time: the tone models' readings are the ceiling. An audio LLM hears pitch, energy, hesitation and voice quality
directly. ``bakeoff`` asks each candidate (open licences only) for the feeling of each held-out clip, with the real
audio, and reports unweighted accuracy (UA) and confusions.

Training the audio-native responder (Qwen2-Audio-7B-Instruct, Apache-2.0, won the bake-off):

  python scripts/audio_llm.py build --sft runs/respond_sft_v8 --heard heard_v5.jsonl --voiced voiced/ --out manifest/
  python scripts/audio_llm.py train --data manifest/ --voiced voiced/ --out /work/out          # GPU job (A100)
  python scripts/audio_llm.py answer --model /work/out/merged --heard eval/tone.jsonl --out ans_tone.jsonl

``build`` pairs every training example of the text responder (the same lines, the same teacher answers) with audio:
the real CREMA-D clip for the tone-only examples, the cloned voicing (scripts/clone_speech.py) for the written lines;
voicings that Whisper could not understand (word error rate above ``--max-wer``) are dropped. ``train`` LoRA-tunes
the language model on (audio + transcript + earlier lines) -> the same JSON answer, the audio encoder frozen.
``answer`` writes answers in the shape scripts/eval_clean.py scores.

Audio comes from CREMA-D (mteb/crema-d, Open Database License) by file name.
"""
import argparse
import json
import os
import re
import sys
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
    if a.voiced:                       # cloned voicings (scripts/clone_speech.py): <voiced>/<clip>.wav
        from rmr.voice import load_audio

        rows = [{**r, "label": r.get("label") or r["feeling"], "tone": r["clip"]} for r in rows
                if r.get("label", r.get("feeling")) in FEELINGS5 and os.path.exists(os.path.join(a.voiced, r["clip"] + ".wav"))]
        audio = {r["clip"]: load_audio(open(os.path.join(a.voiced, r["clip"] + ".wav"), "rb").read()) for r in rows}
    else:
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


AUDIO_SYSTEM = """You are Reachy Mini, a small expressive robot (head that tilts, nods and turns, two antenna ears,
a rotating body). Someone just spoke to you: you hear their voice, with a transcript that can be wrong. From how they
sound and what they say, judge how they feel and respond with kind body language: gentle with sadness, calm with
anger, reassuring with fear, joyful with joy, attentive otherwise. Write the motion as a recipe:
  go D k=v ... | hold D [E=v] | osc D ch amp per   channels e (ears: 0 up, 15 relaxed, 150 drooped), p (pitch, + = down),
  r (roll), y (yaw), z (height mm), b (body), E (energy: 0 still, 1 calm, 3 lively, 6+ shaking)
Reply with JSON only: {"feeling": "...", "response": "...", "recipe": "...", "reading": "..."}"""


def user_text(ex):
    parts = []
    if ex.get("context"):
        parts.append("Earlier in the conversation:\n" + "\n".join(f'- "{c}"' for c in ex["context"]))
    parts.append(f'Transcript: "{ex.get("text") or "(no words)"}"\nHow do they feel, and how do you respond?')
    return "\n".join(parts)


def conversation(ex):
    return [{"role": "system", "content": AUDIO_SYSTEM},
            {"role": "user", "content": [{"type": "audio", "audio_url": "speech.wav"},
                                         {"type": "text", "text": user_text(ex)}]}]


def build(a):
    heard = {json.loads(x)["clip"]: json.loads(x) for x in open(a.heard)}
    bad = set()
    check = os.path.join(a.voiced, "check.jsonl")
    if os.path.exists(check):
        bad = {r["clip"] for r in map(json.loads, open(check)) if r["wer"] > a.max_wer}
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "val"):
        rows, skipped = [], Counter()
        for r in map(json.loads, open(os.path.join(a.sft, f"{split}.jsonl"))):
            h = heard[r["clip"]]
            if r["clip"].startswith("t_"):
                audio = "crema:" + h["tone"]
            elif os.path.exists(os.path.join(a.voiced, r["clip"] + ".wav")) and r["clip"] not in bad:
                audio = r["clip"] + ".wav"
            else:
                skipped["unvoiced" if r["clip"] not in bad else "garbled"] += 1
                continue
            rows.append({"clip": r["clip"], "label": r["label"], "audio": audio, "text": h["text"],
                         "context": h.get("context") or [], "answer": r["messages"][-1]["content"]})
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        print(f"{split}: {len(rows)} examples ({Counter(x['audio'].startswith('crema:') for x in rows)[True]} real "
              f"CREMA-D), skipped {dict(skipped)}")


class AudioRows:
    """Manifest rows with their waveforms (CREMA-D by name, voicings from ``voiced``), resampled to 16 kHz."""

    def __init__(self, path, voiced):
        from rmr.voice import load_audio

        self.rows = [json.loads(x) for x in open(path)]
        crema = crema_audio([r["audio"][6:] for r in self.rows if r["audio"].startswith("crema:")])
        self.wav = [crema[r["audio"][6:]] if r["audio"].startswith("crema:") else
                    load_audio(open(os.path.join(voiced, r["audio"]), "rb").read()) for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {**self.rows[i], "wav": self.wav[i][:16000 * 30]}


def collate_fn(proc):
    import torch

    def collate(batch):
        prompts = [proc.apply_chat_template(conversation(ex), add_generation_prompt=True, tokenize=False) for ex in batch]
        fulls = [p + ex["answer"] + "<|im_end|>\n" for p, ex in zip(prompts, batch)]
        wavs = [ex["wav"] for ex in batch]
        enc = proc(text=fulls, audio=wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        n_prompt = [int(proc(text=[p], audio=[w], sampling_rate=16000, return_tensors="pt")["attention_mask"].sum())
                    for p, w in zip(prompts, wavs)]
        labels = torch.full_like(enc["input_ids"], -100)
        for i, n in enumerate(n_prompt):
            pos = enc["attention_mask"][i].nonzero().flatten()
            ans = pos[n:]                                       # the answer: every real token after the prompt
            labels[i, ans] = enc["input_ids"][i, ans]
        enc["labels"] = labels
        return enc
    return collate


def train(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Trainer, TrainingArguments

    proc = AutoProcessor.from_pretrained(a.model)
    gpu = torch.cuda.is_available()
    model = Qwen2AudioForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16 if gpu else torch.float32)
    model.config.use_cache = False
    targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.r, lora_dropout=0.0, target_modules=targets,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    tr, va = AudioRows(os.path.join(a.data, "train.jsonl"), a.voiced), AudioRows(os.path.join(a.data, "val.jsonl"), a.voiced)
    if a.max_rows:
        tr.rows, tr.wav = tr.rows[:a.max_rows], tr.wav[:a.max_rows]
    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
                             gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, max_steps=a.max_steps,
                             learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=8, bf16=gpu, use_cpu=not gpu,
                             logging_steps=5, eval_strategy="steps", eval_steps=a.eval_steps, save_strategy="steps",
                             save_steps=a.eval_steps, save_total_limit=2, load_best_model_at_end=True,
                             metric_for_best_model="eval_loss", greater_is_better=False, remove_unused_columns=False,
                             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                             dataloader_num_workers=0, report_to=[])
    t0 = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, data_collator=collate_fn(proc))
    trainer.train()
    print(f"best checkpoint {trainer.state.best_model_checkpoint} (eval_loss {trainer.state.best_metric:.4f}) in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(os.path.join(a.out, "merged"))
    proc.save_pretrained(os.path.join(a.out, "merged"))


def answer(a):
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from rmr.planner.llm import parse_json
    from rmr.voice import load_audio

    rows = [json.loads(x) for x in open(a.heard)]
    if a.voiced:
        wavs = [load_audio(open(os.path.join(a.voiced, r["clip"] + ".wav"), "rb").read()) for r in rows]
    else:
        crema = crema_audio([r["tone"] for r in rows])
        wavs = [crema[r["tone"]] for r in rows]
    proc = AutoProcessor.from_pretrained(a.model)
    gpu = torch.cuda.is_available()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    model = Qwen2AudioForConditionalGeneration.from_pretrained(a.model, dtype=dt).to(dev).eval()
    out, t0 = [], time.time()
    for i in range(0, len(rows), a.batch):
        batch = rows[i:i + a.batch]
        texts = [proc.apply_chat_template(conversation(r), add_generation_prompt=True, tokenize=False) for r in batch]
        proc.tokenizer.padding_side = "left"
        enc = proc(text=texts, audio=[w[:16000 * 30] for w in wavs[i:i + a.batch]], sampling_rate=16000,
                   return_tensors="pt", padding=True).to(dev)
        enc["input_features"] = enc["input_features"].to(dt)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=400, do_sample=False)
        for r, g in zip(batch, proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            try:
                d = parse_json(g)
            except Exception:
                d = {}
            out.append({"clip": r["clip"], "label": r["label"], "raw": g, "recipe": d.get("recipe"),
                        "both": {k: d.get(k, "") for k in ("feeling", "reading", "response")}})
        print(f"{len(out)}/{len(rows)} answered ({(time.time() - t0) / len(out):.2f} s each, batched)", flush=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in out)


def text_baseline(a):
    """The current pipeline on the same audio: Whisper + the tone models (rmr.voice) -> the text responder."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from rmr.respond import Student
    from rmr.voice import Listener

    rows = [json.loads(x) for x in open(a.heard)]
    lis, st = Listener(device="cuda"), Student(a.model)
    crema = {} if a.voiced else crema_audio([r["tone"] for r in rows])
    out = []
    for i, r in enumerate(rows):
        path = os.path.join(a.voiced, r["clip"] + ".wav") if a.voiced else None
        if path and not os.path.exists(path):
            continue
        h = lis.hear(path if path else (crema[r["tone"]], 16000))
        h["context"] = r.get("context") or []
        try:
            d = st(h)
        except Exception as e:
            d = {"feeling": "", "reading": "", "response": "", "recipe": None, "error": str(e)}
        out.append({"clip": r["clip"], "label": r["label"], "heard_text": h["text"], "emotion": h["emotion"],
                    "recipe": d.get("recipe"), "both": {k: d.get(k, "") for k in ("feeling", "reading", "response")}})
        if i % 25 == 0:
            print(f"{i}/{len(rows)} {r['label']} -> {d.get('feeling')}", flush=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bakeoff")
    b.add_argument("--heard", required=True, help="eval_clean tone.jsonl (CREMA-D file names in 'tone')")
    b.add_argument("--models", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--voiced", default="", help="score cloned voicings (<dir>/<clip>.wav) instead of CREMA-D clips")
    bu = sub.add_parser("build")
    bu.add_argument("--sft", required=True, help="the text responder's SFT folder (train.jsonl, val.jsonl)")
    bu.add_argument("--heard", required=True)
    bu.add_argument("--voiced", required=True)
    bu.add_argument("--out", required=True)
    bu.add_argument("--max-wer", type=float, default=0.35)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--voiced", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--max-rows", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--r", type=int, default=32)
    t.add_argument("--bs", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=8)
    t.add_argument("--eval-steps", type=int, default=25)
    an = sub.add_parser("answer")
    an.add_argument("--model", required=True)
    an.add_argument("--heard", required=True, help="eval rows; audio from CREMA-D ('tone') unless --voiced")
    an.add_argument("--voiced", default="", help="folder of <clip>.wav (cloned voicings of written lines)")
    an.add_argument("--out", required=True)
    an.add_argument("--batch", type=int, default=8)
    tb = sub.add_parser("text_baseline")
    tb.add_argument("--model", default="mszarski/reachy-voice:student/v8-dpo")
    tb.add_argument("--heard", required=True)
    tb.add_argument("--voiced", default="")
    tb.add_argument("--out", required=True)
    a = ap.parse_args()
    {"bakeoff": bakeoff, "build": build, "train": train, "answer": answer, "text_baseline": text_baseline}[a.cmd](a)


if __name__ == "__main__":
    main()
