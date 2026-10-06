"""Live text-to-motion server: the browser viewer plus ``POST /api/generate``.

  pip install -e ".[ik,generator,hf,serve]"
  HF_TOKEN=... python -m rmr.server --ckpt hf://mszarski/reachy-motion-generator/generator.pt   # localhost:7860

``POST /api/generate`` takes ``{"prompt", "n", "seed"}`` and returns ``{"prompt", "idea", "recipe", "moves",
"timing_ms"}``: the zero-shot planner (``rmr.planner.write``, Hugging Face Inference Providers) writes a recipe,
it expands into ``n`` randomised plans as served (``fc=2``, ``kdt=0.25``), the generator turns them into 25 Hz
motion on CPU (with ``fill_energy``, so held energy is honoured), and each motion is projected onto the reachable set. ``GET /api/health`` tells the viewer to show
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
    def __init__(self, ckpt, model=None):
        from .generator.sample import load

        self.net, self.stats, self.dev = load(ckpt)
        self.reach, self.model, self.ckpt = Reach(), model, ckpt

    def plan(self, prompt):
        got, errors = _batch([prompt], self.model)
        if prompt not in got:
            raise ValueError(f"the planner did not produce a valid recipe: {errors.get(prompt, 'no answer')}")
        return got[prompt]["idea"], got[prompt]["recipe"]

    def generate(self, prompt, n=1, seed=0):
        from .generator.sample import generate_batch

        t0 = time.time()
        idea, recipe = self.plan(prompt)
        t1 = time.time()
        plans = variants(recipe, n, seed=seed, fc=2.0, kdt=0.25)
        trajs = generate_batch(self.net, self.stats, plans, self.dev, seeds=[seed * 7919 + k for k in range(n)],
                               fill_energy=True)
        moves = [self.reach.project(traj_to_move(A, prompt))[0] for A in trajs]
        t2 = time.time()
        return {"prompt": prompt, "idea": idea, "recipe": recipe, "moves": moves,
                "timing_ms": {"planner": int(1000 * (t1 - t0)), "generator": int(1000 * (t2 - t1)),
                              "total": int(1000 * (t2 - t0))}}


def create_app(engine):
    from fastapi import FastAPI, HTTPException
    from fastapi.concurrency import run_in_threadpool
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    class Req(BaseModel):
        prompt: str = Field(min_length=1, max_length=MAX_PROMPT)
        n: int = Field(1, ge=1, le=4)
        seed: int = Field(0, ge=0, le=2 ** 31 - 1)

    app = FastAPI(title="Reachy Mini text-to-motion")

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

    app.mount("/", StaticFiles(directory=VIEWER, html=True), name="viewer")
    return app


def main():
    ap = argparse.ArgumentParser(prog="python -m rmr.server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.environ.get("GENERATOR_CKPT", "checkpoints/generator.pt"))
    ap.add_argument("--model", default=os.environ.get("PLANNER_MODEL"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    a = ap.parse_args()
    import uvicorn

    uvicorn.run(create_app(Engine(a.ckpt, a.model)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
