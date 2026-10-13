"""The audio-native responder: an audio LLM that hears the voice itself (scripts/audio_llm.py trains it).

  st = AudioStudent("mszarski/reachy-voice:audio/student/v1")
  for early, answer in st.answers(heard, wav):   # heard: rmr.voice.Listener.hear (transcript, context); wav: 16 kHz
      ...

Same interface and answer as rmr.respond.Student ({"feeling", "response", "recipe", "reading"}, streamed: the early
answer comes as soon as the recipe is complete), but the model gets the audio, the transcript and the earlier lines
instead of the tone models' one-line reading. Qwen2-Audio-7B-Instruct (Apache-2.0) fine-tuned on the text responder's
examples, voiced (docs/results/voice.md, "Audio-native responder").

``path``: a merged model (dir, Hub repo, or ``repo:sub/dir`` as rmr.planner.finetune.resolve), run in-process with
transformers; or the URL of an OpenAI-compatible server (vLLM) serving it, ``http://host:port/v1#model``.
"""
import json

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



class AudioStudent:
    def __init__(self, path, max_tokens=400):
        self.max_tokens = max_tokens
        if path.startswith(("http://", "https://")):
            self.url, _, self.model = path.partition("#")
            return
        self.url = None
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        from .planner.finetune import resolve

        path = resolve(path)
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = torch.bfloat16 if self.dev == "cuda" else torch.float32
        self.proc = AutoProcessor.from_pretrained(path)
        self.m = Qwen2AudioForConditionalGeneration.from_pretrained(path, dtype=self.dt).to(self.dev).eval()

    def stream(self, heard, wav):
        """Yield the answer text as it grows."""
        if self.url:
            yield from self._stream_http(heard, wav)
            return
        from threading import Thread

        from transformers import TextIteratorStreamer

        text = self.proc.apply_chat_template(conversation(heard), add_generation_prompt=True, tokenize=False)
        enc = self.proc(text=[text], audio=[wav[:16000 * 30]], sampling_rate=16000, return_tensors="pt").to(self.dev)
        enc["input_features"] = enc["input_features"].to(self.dt)
        streamer = TextIteratorStreamer(self.proc.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kw = dict(**enc, max_new_tokens=self.max_tokens, do_sample=False, streamer=streamer)
        th = Thread(target=lambda: self.torch.no_grad()(self.m.generate)(**kw), daemon=True)
        th.start()
        yield from streamer
        th.join()

    def _stream_http(self, heard, wav):
        """vLLM's OpenAI-compatible chat API, the audio inline as base64 WAV (``input_audio``)."""
        import base64
        import io
        import urllib.request

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, wav[:16000 * 30], 16000, format="WAV")
        audio = {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}
        msgs = [{"role": "system", "content": AUDIO_SYSTEM},
                {"role": "user", "content": [audio, {"type": "text", "text": user_text(heard)}]}]
        body = {"model": self.model or "reachy-audio", "messages": msgs, "temperature": 0.0,
                "max_tokens": self.max_tokens, "stream": True}
        req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def stages(self, heard, wav):
        """Yield ``("feeling", word)`` as soon as the feeling is written, then ``("early", answer)`` once the recipe
        is complete (if it comes before the end), then ``("final", answer)``."""
        from .planner.llm import parse_json
        from .respond import FEELINGS, _clean, _valid_recipe, partial_fields

        text, felt, sent = "", False, False
        for chunk in self.stream(heard, wav):
            text += chunk
            if not sent:
                got = partial_fields(text)
                if not felt and "feeling" in got:
                    felt = True
                    f = str(got["feeling"]).lower()
                    yield "feeling", f if f in FEELINGS else "neutral"
                if {"feeling", "recipe"} <= got.keys() and "reading" not in got and _valid_recipe(got["recipe"]):
                    sent = True
                    yield "early", _clean(got)
        d = parse_json(text)
        if not _valid_recipe(d.get("recipe")):
            raise ValueError("the audio responder did not produce a valid recipe")
        yield "final", _clean(d)

    def answers(self, heard, wav):
        """Yield ``(early, answer)`` like rmr.respond.Student.answers."""
        for kind, x in self.stages(heard, wav):
            if kind != "feeling":
                yield kind == "early", x
