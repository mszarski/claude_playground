"""What should Reachy do when it hears someone? An open LLM turns (transcript + how they sound) into a motion prompt.

  from rmr.respond import respond
  respond({"text": "I didn't get the job", "emotion": "sad", "confidence": 0.6, "arousal": 0.3, "valence": 0.2})
  -> {"reading": "...", "response": "consoling. You lean in slowly and stay close, ears softly lowered."}

The response is a prompt in the planner's format ("word. one sentence."), so it goes straight into the motion
pipeline. Any OpenAI-compatible endpoint works (``PLANNER_BASE_URL``: Ollama, llama.cpp, vLLM, Hugging Face), or
``PLANNER_BASE_URL=local`` runs a model in-process with transformers (see ``rmr.planner.llm``).
"""
from .planner import llm

SYSTEM = """You decide how Reachy Mini, a small expressive desktop robot (a head that tilts, nods and turns, two
antennas like ears, a rotating body, no arms, no face), responds with body language to what a person just said.

You get the person's words and an automatic reading of how their voice sounds: an emotion with a confidence, and
arousal (calm 0 .. activated 1) and valence (negative 0 .. positive 1). The voice reading is often wrong; when it
disagrees with the words, trust the words. When both are unclear, respond with warm, attentive interest.

Respond like a kind, emotionally intelligent companion, not a mirror:
- sad, hurt or tired -> gentle, slow, close: lean in, soft lowered ears, small comforting nods
- angry or frustrated -> calm and steady attention, never anger back, no sudden moves
- anxious or afraid -> reassuring, slow and grounded
- happy, excited or proud -> share the joy: perk up, bounce, ears up
- neutral or a question -> attentive listening: a curious tilt, a small nod
Keep it to one short beat (3-6 seconds) that fits the moment.

Reply with JSON only: {"feeling": "<one of: neutral, happy, sad, angry, anxious, surprised>", "reading": "<one
sentence: how they seem>", "response": "<attitude>. <one sentence: what Reachy does, starting with 'You'>"}, where
<attitude> is one or two words naming Reachy's stance, e.g. "comforting", "sharing the joy", "calm attention",
"curious" (not an interjection like "Hmm" or "Wow")."""

FEELINGS = ["neutral", "happy", "sad", "angry", "anxious", "surprised"]

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["feeling", "reading", "response"],
          "properties": {"feeling": {"type": "string", "enum": FEELINGS}, "reading": {"type": "string"},
                         "response": {"type": "string"}}}


def user_message(heard, tone=True):
    """``tone=False`` leaves out the voice reading (words only)."""
    parts = [f'They said: "{heard.get("text") or "(no words)"}"']
    if heard.get("context"):             # the last lines of the conversation, oldest first
        parts.insert(0, "Earlier in the conversation:\n" + "\n".join(f'- "{c}"' for c in heard["context"]))
    if not tone:
        return "\n".join(parts) + "\n(No reading of their voice is available.)"
    if heard.get("emotion"):
        parts.append(f'Their voice sounds {heard["emotion"]} (confidence {heard.get("confidence", 0):.0%}).')
    if "arousal" in heard and "valence" in heard:
        parts.append(f'Arousal {heard["arousal"]:.2f}, valence {heard["valence"]:.2f}.')
    return "\n".join(parts)


def respond(heard, model=None, chat=None, tone=True, temperature=0.4, hint=None):
    """``heard`` (from ``rmr.voice.Listener.hear``) -> ``{"feeling", "reading", "response"}``.

    ``hint`` (training data only): a human annotator's label. The teacher is asked to find the cues that support
    it and respond accordingly; the student trained on the answer never sees the hint."""
    chat = chat or llm.chat_json
    msg = user_message(heard, tone)
    if hint:
        msg += (f"\n\n(For training: a human annotator who saw the whole scene says this person is {hint}. Find the "
                f"cues in their words, the conversation and their voice that show it, write your reading from those "
                f"cues as if you noticed them yourself, set feeling to {hint}, and respond accordingly. Do not mention "
                f"the annotator.)")
    out = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
               SCHEMA, "reachy_response", model=model, temperature=temperature)
    resp = (out.get("response") or "").strip()
    if not resp:
        raise ValueError("the responder returned no response")
    head = resp.split(".", 1)[0]
    if "." not in resp or len(head.split()) > 3:      # the planner expects "word. sentence."
        resp = "responding. " + resp
    feeling = str(out.get("feeling") or "").strip().lower()
    return {"feeling": feeling if feeling in FEELINGS else "neutral", "reading": (out.get("reading") or "").strip(),
            "response": resp}


# A distilled student does the responder's and the planner's job in one call: heard -> feeling, reading, response
# and the motion recipe (see scripts/label_respond.py and docs/results/voice.md).
STUDENT_SYSTEM = """You are Reachy Mini, a small expressive robot (head that tilts, nods and turns, two antenna ears,
a rotating body). Someone just spoke to you. From their words and an automatic, often wrong, reading of their voice,
judge how they feel (trust the words when they disagree) and respond with kind body language: gentle with sadness,
calm with anger, reassuring with fear, joyful with joy, attentive otherwise. Write the motion as a recipe:
  go D k=v ... | hold D [E=v] | osc D ch amp per   channels e (ears: 0 up, 15 relaxed, 150 drooped), p (pitch, + = down),
  r (roll), y (yaw), z (height mm), b (body), E (energy: 0 still, 1 calm, 3 lively, 6+ shaking)
Reply with JSON only: {"feeling": "...", "reading": "...", "response": "...", "recipe": "..."}"""


def student_messages(heard, answer=None):
    """Chat messages for the distilled responder; with ``answer`` (a dict), the training target is appended."""
    import json

    m = [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": user_message(heard)}]
    if answer is not None:
        m.append({"role": "assistant", "content": json.dumps({k: answer[k] for k in ("feeling", "reading", "response",
                                                                                         "recipe")})})
    return m


class Student:
    """The distilled responder: heard -> {"feeling", "reading", "response", "recipe"} in one call.

    ``path``: merged model dir, Hub repo id, or ``repo:sub/dir`` (e.g. ``mszarski/reachy-voice:student/v1``), run
    in-process with transformers; or the URL of an OpenAI-compatible server running the GGUF build, e.g.
    ``http://localhost:8080/v1`` (llama-server) or ``http://localhost:11434/v1#reachy-voice`` (Ollama, ``#model``)."""

    def __init__(self, path, max_tokens=400):
        if path.startswith(("http://", "https://")):
            self.url, _, self.model = path.partition("#")
            self.max_tokens = max_tokens
            return
        self.url = None
        import torch
        from transformers import AutoTokenizer

        from .planner.finetune import lm_class, resolve

        path = resolve(path)
        self.torch, self.max_tokens = torch, max_tokens
        self.tok = AutoTokenizer.from_pretrained(path)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = lm_class(path).from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()

    def _generate(self, heard):
        if self.url:
            import json
            import urllib.request

            body = {"model": self.model or "reachy-voice", "messages": student_messages(heard), "temperature": 0,
                    "max_tokens": self.max_tokens}
            req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                         {"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        text = self.tok.apply_chat_template(student_messages(heard), tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        enc = self.tok(text, return_tensors="pt").to(self.m.device)
        with self.torch.no_grad():
            g = self.m.generate(**enc, max_new_tokens=self.max_tokens, do_sample=False,
                                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def __call__(self, heard):
        from .planner.evaluate import parse_answer
        from .planner.llm import parse_json

        d = parse_json(self._generate(heard))
        recipe = d.get("recipe") if isinstance(d.get("recipe"), str) else None
        if not recipe or not parse_answer(__import__("json").dumps({"recipe": recipe})):
            raise ValueError("the voice model did not produce a valid recipe")
        feeling = str(d.get("feeling", "")).lower()
        return {"feeling": feeling if feeling in FEELINGS else "neutral", "reading": str(d.get("reading", "")),
                "response": str(d.get("response", "")), "recipe": recipe}
