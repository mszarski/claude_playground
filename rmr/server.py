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

VIEWER = os.path.join(os.path.dirname(__file__), "..", "visualizer")
MAX_PROMPT = 300


class Engine:
    """``planner``: None = the zero-shot LLM (``model`` on ``PLANNER_BASE_URL``), or a fine-tuned planner
    (merged model dir or Hub repo id, run in-process). ``responder``: the LLM that turns what it heard into a
    response prompt (``rmr.respond``); voice models load on the first ``/api/respond``."""

    def __init__(self, ckpt, model=None, planner=None, responder=None, voice_model=None):
        from .generator.sample import load

        self.net, self.stats, self.dev = load(ckpt)
        self.reach, self.model, self.ckpt = Reach(), model, ckpt
        self.responder, self.listener = responder or model, None
        self.finetuned, self.voice_model = None, voice_model
        self.student = None
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

    def respond(self, audio, n=1, seed=0):
        """Recorded speech -> what Reachy heard, how it reads the person, and its response motion."""
        out = {}
        for part in self.respond_stream(audio, n, seed):
            timing = {**out.get("timing_ms", {}), **part.get("timing_ms", {})}
            out.update(part)
            out["timing_ms"] = timing
        out.pop("stage", None)
        return out

    def respond_stream(self, audio, n=1, seed=0):
        """Like ``respond``, in stages, so a client can act on each as soon as it exists:
        ``{"stage": "heard", "heard"}``, then ``{"stage": "motion", "moves", "recipe", "feeling", "idea", ...}`` as
        soon as the recipe is written (with a v3 student, before the reading), then ``{"stage": "done", "reading"}``."""
        from .respond import respond
        from .voice import Listener

        t0 = time.time()
        if self.listener is None:
            self.listener = Listener()
        heard = self.listener.hear(audio)
        t1 = time.time()
        yield {"stage": "heard", "heard": {k: v for k, v in heard.items() if k != "probs"},
               "timing_ms": {"listen": int(1000 * (t1 - t0))}}
        if self.voice_model:              # the distilled student: response and recipe in one call
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
    async def respond(request: Request, n: int = 1, seed: int = 0, stream: bool = False):
        audio = await request.body()
        if not 1000 < len(audio) < 8_000_000:
            raise HTTPException(422, "send a WAV recording between 0.1 s and about 4 minutes")
        if stream:                      # newline-delimited JSON, one line per stage (Engine.respond_stream)
            import json

            from fastapi.responses import StreamingResponse

            def lines():
                try:
                    for part in engine.respond_stream(audio, max(1, min(4, n)), seed):
                        yield json.dumps(part) + "\n"
                except Exception as e:
                    yield json.dumps({"stage": "error", "detail": f"{type(e).__name__}: {e}"}) + "\n"
            return StreamingResponse(lines(), media_type="application/x-ndjson")
        try:
            return await run_in_threadpool(engine.respond, audio, max(1, min(4, n)), seed)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}")

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
                    help="distilled responder (one local call: response + recipe), e.g. mszarski/reachy-voice:student/v2")
    ap.add_argument("--preload-voice", action="store_true", default=bool(os.environ.get("PRELOAD_VOICE")),
                    help="load the voice models at startup (in the background) instead of on the first /api/respond")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    a = ap.parse_args()
    import uvicorn

    engine = Engine(a.ckpt, a.model, a.planner, a.responder, a.voice_model)
    if a.preload_voice:
        import threading

        from .voice import Listener

        def _load():
            engine.listener = Listener()
            print("voice models loaded", flush=True)
        threading.Thread(target=_load, daemon=True).start()
    uvicorn.run(create_app(engine), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
