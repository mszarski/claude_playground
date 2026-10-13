"""Live text-to-motion server: the browser viewer plus ``POST /api/generate``.

  pip install -e ".[ik,generator,hf,serve]"
  HF_TOKEN=... python -m rmr.server --ckpt hf://mszarski/reachy-motion-generator/generator.pt   # localhost:7860

``POST /api/respond`` takes a WAV recording (request body) and answers like ``/api/generate`` plus ``heard``
(transcript, voice emotion, arousal / valence / dominance) and ``reading``: speech -> ``rmr.voice`` -> ``rmr.respond``
-> the same motion pipeline. With ``?stream=1`` it answers in stages as newline-delimited JSON (heard, motion,
done), so the robot can start moving before the reading is written.

``POST /api/generate`` takes ``{"prompt", "n", "seed"}`` and returns ``{"prompt", "idea", "recipe", "moves",
"timing_ms"}``: the zero-shot planner (``rmr.planner.write``, Hugging Face Inference Providers) writes a recipe,
it expands into ``n`` randomised plans as served (``fc=2``, ``kdt=0.25``), the generator turns them into 25 Hz
motion on CPU, and each motion is projected onto the reachable set. ``GET /api/health`` tells the viewer to show
its prompt box. Everything else is the viewer in ``visualizer/``.

Reference: ``inference/server.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0), which serves the
fine-tuned planners on a GPU.
"""
import argparse
import os
import time

from .motion import traj_to_move
from .planner.write import _batch
from .reach import Reach
from .recipe import variants

def voice_device():
    """Whisper and the speech-emotion models run on the GPU when there is one."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


VIEWER = os.path.join(os.path.dirname(__file__), "..", "visualizer")
MAX_PROMPT = 300


class Engine:
    """``planner``: None = the zero-shot LLM (``model`` on ``PLANNER_BASE_URL``), or a fine-tuned planner
    (merged model dir or Hub repo id, run in-process). ``responder``: the LLM that turns what it heard into a
    response prompt (``rmr.respond``); voice models load on the first ``/api/respond``."""

    def __init__(self, ckpt, model=None, planner=None, responder=None, voice_model=None, calibrate=False,
                 audio_model=None):
        from .generator.sample import load

        self.net, self.stats, self.dev = load(ckpt)
        self.reach, self.model, self.ckpt = Reach(), model, ckpt
        self.responder, self.listener = responder or model, None
        self.finetuned, self.voice_model = None, voice_model
        self.student = None
        self.audio_model, self.audio_student = audio_model, None   # rmr.audio_responder: hears the voice itself
        self.listener_weights = None        # path of the learned listener's JSON weights, served to the viewer
        self.history = {}                   # session -> [(time, what they said)], the conversation so far
        self.calibrate = calibrate          # per-session level correction of the voice attributes (rmr.voice)
        self.calibrations = {}              # session -> rmr.voice.Calibration
        from .feedback import Turns
        self.turns = Turns()                # recent voice turns by id, for ratings (rmr.feedback)
        self.feedback = None                # rmr.feedback.FeedbackLog, set by main()
        self.moods = {}                     # session -> rmr.mood.Mood
        if planner:
            from .planner.finetune import Planner
            self.finetuned = Planner(planner)

    def plan(self, prompt):
        if self.finetuned is not None:
            return "", self.finetuned.plan(prompt)
        got, errors = _batch([prompt], self.model)
        if prompt not in got:
            raise ValueError(f"the planner did not produce a valid recipe: {errors.get(prompt, 'no answer')}")
        return got[prompt]["idea"], got[prompt]["recipe"]

    HISTORY_TURNS, HISTORY_S, MAX_SESSIONS = 2, 300.0, 1000

    def remember(self, session, heard):
        """Give ``heard`` the person's last few lines in this session as context (the student was trained with the
        two previous lines of the dialogue, oldest first), then remember this one."""
        if not session:
            return
        now = time.time()
        past = [(t, x) for t, x in self.history.get(session, []) if now - t < self.HISTORY_S]
        if past:
            heard["context"] = [x for _, x in past[-self.HISTORY_TURNS:]]
        if heard.get("text"):
            past.append((now, heard["text"]))
        self.history[session] = past[-self.HISTORY_TURNS:]
        if len(self.history) > self.MAX_SESSIONS:                # forget the stalest sessions
            for k in sorted(self.history, key=lambda k: self.history[k][-1][0] if self.history[k] else 0)[
                    :len(self.history) - self.MAX_SESSIONS]:
                del self.history[k]

    def respond(self, audio, n=1, seed=0, session=None):
        """Recorded speech -> what Reachy heard, how it reads the person, and its response motion."""
        out = {}
        for part in self.respond_stream(audio, n, seed, session):
            timing = {**out.get("timing_ms", {}), **part.get("timing_ms", {})}
            out.update(part)
            out["timing_ms"] = timing
        out.pop("stage", None)
        return out

    def mood(self, session):
        from .mood import Mood
        if len(self.moods) > self.MAX_SESSIONS:
            self.moods.clear()
        return self.moods.setdefault(session or "", Mood())

    def respond_stream(self, audio, n=1, seed=0, session=None):
        """``_respond_stream`` with a turn id on every stage (for ratings, rmr.feedback) and Reachy's mood
        (rmr.mood), updated from what it read, on the last."""
        tid = self.turns.new(session, audio)
        for part in self._respond_stream(audio, n, seed, session):
            if part["stage"] == "heard":
                self.turns.update(tid, heard=part["heard"])
            elif part["stage"] == "motion":
                self.turns.update(tid, recipe=part.get("recipe"), response=part.get("idea"),
                                  feeling=part.get("feeling"), seed=seed,
                                  responder=self.audio_model or self.voice_model or self.responder or "default")
            elif part["stage"] == "done":
                h = (self.turns.get(tid) or {}).get("heard", {})
                part["mood"] = self.mood(session).update(part.get("feeling"), h.get("valence"), h.get("arousal"))
                self.turns.update(tid, reading=part.get("reading"), mood=part["mood"])
            yield {**part, "turn": tid}

    def _respond_stream(self, audio, n=1, seed=0, session=None):
        """Like ``respond``, in stages, so a client can act on each as soon as it exists:
        ``{"stage": "heard", "heard"}``, then ``{"stage": "motion", "moves", "recipe", "feeling", "idea", ...}`` as
        soon as the recipe is written (with a v3 student, before the reading), then ``{"stage": "done", "reading"}``."""
        from .respond import respond
        from .voice import Listener

        t0 = time.time()
        if self.listener is None:
            self.listener = Listener(device=voice_device())
        heard = self.listener.hear(audio)
        if self.calibrate:
            from .voice import Calibration
            if len(self.calibrations) > self.MAX_SESSIONS:
                self.calibrations.clear()
            heard = self.calibrations.setdefault(session or "", Calibration(probs=True))(heard)
        self.remember(session, heard)
        t1 = time.time()
        yield {"stage": "heard", "heard": {k: v for k, v in heard.items() if k != "probs"},
               "timing_ms": {"listen": int(1000 * (t1 - t0))}}
        if self.audio_model:              # the audio-native responder: hears the recording itself
            from .audio_responder import AudioStudent
            from .voice import load_audio
            if self.audio_student is None:
                self.audio_student = AudioStudent(self.audio_model)
            wav, moved = load_audio(audio), False
            for early, r in self.audio_student.answers(heard, wav):
                if not moved:
                    out = self.generate(r["response"], n=n, seed=seed, recipe=r["recipe"])
                    out["timing_ms"]["planner"] = int(1000 * (time.time() - t1)) - out["timing_ms"]["generator"]
                    out["timing_ms"]["first_motion"] = int(1000 * (time.time() - t0))
                    yield {"stage": "motion", **out, "idea": r["response"], "feeling": r["feeling"]}
                    moved = True
                if not early:
                    yield {"stage": "done", "reading": r["reading"], "feeling": r["feeling"],
                           "timing_ms": {"total": int(1000 * (time.time() - t0))}}
        elif self.voice_model:            # the distilled student: response and recipe in one call
            from .respond import Student
            if self.student is None:
                self.student = Student(self.voice_model)
            moved = False
            for early, r in self.student.answers(heard):
                if not moved:
                    out = self.generate(r["response"], n=n, seed=seed, recipe=r["recipe"])
                    out["timing_ms"]["planner"] = int(1000 * (time.time() - t1)) - out["timing_ms"]["generator"]
                    out["timing_ms"]["first_motion"] = int(1000 * (time.time() - t0))
                    yield {"stage": "motion", **out, "idea": r["response"], "feeling": r["feeling"]}
                    moved = True
                if not early:
                    yield {"stage": "done", "reading": r["reading"], "feeling": r["feeling"],
                           "timing_ms": {"total": int(1000 * (time.time() - t0))}}
        else:
            r = respond(heard, model=self.responder)
            out = self.generate(r["response"], n=n, seed=seed)
            out["timing_ms"]["first_motion"] = int(1000 * (time.time() - t0))
            yield {"stage": "motion", **out, "feeling": r.get("feeling")}
            yield {"stage": "done", "reading": r["reading"], "feeling": r.get("feeling"),
                   "timing_ms": {"total": int(1000 * (time.time() - t0))}}

    def generate(self, prompt, n=1, seed=0, recipe=None):
        from .generator.sample import generate_batch

        t0 = time.time()
        idea, recipe = ("", recipe) if recipe else self.plan(prompt)
        t1 = time.time()
        plans = variants(recipe, n, seed=seed, fc=2.0, kdt=0.25)
        trajs = generate_batch(self.net, self.stats, plans, self.dev, seeds=[seed * 7919 + k for k in range(n)])
        moves = [self.reach.project(traj_to_move(A, prompt))[0] for A in trajs]
        t2 = time.time()
        return {"prompt": prompt, "idea": idea, "recipe": recipe, "moves": moves,
                "timing_ms": {"planner": int(1000 * (t1 - t0)), "generator": int(1000 * (t2 - t1)),
                              "total": int(1000 * (t2 - t0))}}


def create_app(engine):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    class Req(BaseModel):
        prompt: str = Field(min_length=1, max_length=MAX_PROMPT)
        n: int = Field(1, ge=1, le=4)
        seed: int = Field(0, ge=0, le=2 ** 31 - 1)

    app = FastAPI(title="Reachy Mini text-to-motion")
    app.state.engine = engine

    @app.get("/api/health")
    def health():
        return {"ok": True, "planner": engine.model or "default", "generator": os.path.basename(engine.ckpt)}

    @app.post("/api/generate")
    async def generate(r: Req):
        try:
            return await run_in_threadpool(engine.generate, r.prompt.strip(), r.n, r.seed)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:          # planner endpoint down, quota, ...
            raise HTTPException(502, f"{type(e).__name__}: {e}")

    @app.post("/api/respond")
    async def respond(request: Request, n: int = 1, seed: int = 0, stream: bool = False, session: str = ""):
        audio = await request.body()
        if not 1000 < len(audio) < 8_000_000:
            raise HTTPException(422, "send a WAV recording between 0.1 s and about 4 minutes")
        if stream:                      # newline-delimited JSON, one line per stage (Engine.respond_stream)
            import json

            from fastapi.responses import StreamingResponse

            def lines():
                try:
                    for part in engine.respond_stream(audio, max(1, min(4, n)), seed, session[:64] or None):
                        yield json.dumps(part) + "\n"
                except Exception as e:
                    yield json.dumps({"stage": "error", "detail": f"{type(e).__name__}: {e}"}) + "\n"
            return StreamingResponse(lines(), media_type="application/x-ndjson")
        try:
            return await run_in_threadpool(engine.respond, audio, max(1, min(4, n)), seed, session[:64] or None)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}")

    class Feedback(BaseModel):
        turn: str = Field(min_length=1, max_length=40)
        rating: int = Field(ge=-1, le=1)
        save_audio: bool = False
        note: str = Field("", max_length=500)

    @app.post("/api/feedback")
    def feedback(f: Feedback):
        """A thumbs up (1) or down (-1) on a voice turn (rmr.feedback); the recording only with ``save_audio``."""
        if engine.feedback is None:
            raise HTTPException(503, "feedback logging is off on this server (--feedback-dir)")
        turn = engine.turns.get(f.turn)
        if turn is None:
            raise HTTPException(404, "unknown or expired turn")
        engine.feedback.add(turn, f.rating, f.save_audio, f.note)
        return {"ok": True}

    class Clip(BaseModel):
        label: str = Field(min_length=2, max_length=40, pattern=r"^[a-z][a-z -]+$")
        description: str = Field("", max_length=200)
        mirror: bool = True
        move: dict

    @app.post("/api/clips")
    def clips(c: Clip):
        """A puppeteered motion (visualizer/puppet.html) with its feeling word, saved as training data."""
        if engine.feedback is None:
            raise HTTPException(503, "saving is off on this server (--feedback-dir)")
        fr, t = c.move.get("set_target_data"), c.move.get("time")
        ok = (isinstance(fr, list) and isinstance(t, list) and 25 <= len(fr) == len(t) <= 16 * 25
              and all(len(f.get("head", [])) == 4 and len(f.get("antennas", [])) == 2 for f in fr))
        if not ok:
            raise HTTPException(422, "a move of 1 to 16 s at 25 Hz is expected")
        return {"ok": True, "id": engine.feedback.add_clip(c.label, c.description, c.move, c.mirror)}

    @app.get("/api/mood")
    def mood(session: str = ""):
        """Reachy's current mood in this conversation (rmr.mood), decayed to now."""
        from .mood import label
        st = engine.mood(session[:64] or None).state()
        return {**st, "label": label(st)}

    @app.get("/api/idle")
    async def idle(silence: float = 0.0, seed: int = 0, session: str = ""):
        """An idle move (rmr.idle) for after ``silence`` seconds of quiet: looking around, stretching, drowsing."""
        import random

        from .idle import pick
        mood = engine.mood(session[:64] or None).state() if session else None
        name, recipe = pick(silence, random.Random(seed), mood)
        out = await run_in_threadpool(engine.generate, name, 1, seed, recipe)
        return {**out, "idea": name}

    @app.get("/api/listener")
    def listener_weights():
        """The learned listener's weights (rmr.listen_model), when the server was given them (CC-BY-NC)."""
        if not engine.listener_weights:
            raise HTTPException(404, "no learned listener configured (--listener-model)")
        from fastapi.responses import FileResponse
        return FileResponse(engine.listener_weights, media_type="application/json")

    app.mount("/", StaticFiles(directory=VIEWER, html=True), name="viewer")
    return app


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.environ.get("GENERATOR_CKPT", "checkpoints/generator.pt"))
    ap.add_argument("--model", default=os.environ.get("PLANNER_MODEL"), help="zero-shot planner LLM")
    ap.add_argument("--planner", default=os.environ.get("FINETUNED_PLANNER"),
                    help="fine-tuned planner (dir or Hub repo id) instead of the zero-shot LLM")
    ap.add_argument("--responder", default=os.environ.get("RESPONDER_MODEL"),
                    help="LLM that decides how to respond to speech (default: --model)")
    ap.add_argument("--voice-model", default=os.environ.get("VOICE_MODEL"),
                    help="distilled responder (one local call: response + recipe), e.g. mszarski/reachy-voice:student/v8-dpo")
    ap.add_argument("--audio-model", default=os.environ.get("AUDIO_MODEL"),
                    help="audio-native responder (rmr.audio_responder; hears the voice itself), e.g. "
                         "mszarski/reachy-voice:audio/student/v1, or a vLLM URL http://host:port/v1#model; "
                         "takes precedence over --voice-model")
    ap.add_argument("--preload-voice", action="store_true", default=bool(os.environ.get("PRELOAD_VOICE")),
                    help="load the voice models at startup (in the background) instead of on the first /api/respond")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=None,
                    help="correct the voice readings per session (attribute levels and category priors) toward what "
                         "the licence-clean students were trained on (rmr.voice.Calibration); on by default with "
                         "--voice-model, off for the zero-shot responder (env CALIBRATE_VOICE=1/0)")
    ap.add_argument("--feedback-dir", default=os.environ.get("FEEDBACK_DIR", "runs/feedback"),
                    help="where thumbs up / down on voice turns are logged (rmr.feedback); '' to turn off")
    ap.add_argument("--feedback-repo", default=os.environ.get("FEEDBACK_REPO"),
                    help="private HF dataset the feedback folder is pushed to every 5 minutes")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--listener-model", default=os.environ.get("LISTENER_MODEL"),
                    help="learned listener weights (rmr.listen_model; CC-BY-NC) for the viewer's head motion while "
                         "you talk, a path or hf://<owner>/<repo>/<path>, e.g. "
                         "hf://mszarski/reachy-listening/listener/v1/listener.json")
    a = ap.parse_args()
    import uvicorn

    if a.calibrate is None:
        env = os.environ.get("CALIBRATE_VOICE")
        a.calibrate = env not in ("0", "false", "") if env is not None else bool(a.voice_model or a.audio_model)
    engine = Engine(a.ckpt, a.model, a.planner, a.responder, a.voice_model, calibrate=a.calibrate,
                    audio_model=a.audio_model)
    if a.feedback_dir:
        from .feedback import FeedbackLog
        engine.feedback = FeedbackLog(a.feedback_dir, a.feedback_repo)
    if a.listener_model:
        path = a.listener_model
        if path.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            owner, repo, fname = path[5:].split("/", 2)
            path = hf_hub_download(f"{owner}/{repo}", fname)
        engine.listener_weights = path
    if a.preload_voice:
        import threading

        from .voice import Listener

        def _load():
            engine.listener = Listener(device=voice_device())
            if engine.audio_model and engine.audio_student is None:
                from .audio_responder import AudioStudent
                engine.audio_student = AudioStudent(engine.audio_model)
            elif engine.voice_model and engine.student is None:
                from .respond import Student
                engine.student = Student(engine.voice_model)
            print(f"voice models loaded on {voice_device()}", flush=True)
        threading.Thread(target=_load, daemon=True).start()
    uvicorn.run(create_app(engine), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
