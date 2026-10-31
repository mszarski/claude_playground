"""``ductus serve``: live preview and canvas editing.

The browser is a thin client. It sends intents (parameter values, override
edits, undo...) over a websocket; the server owns all state, writes canvas
edits to ``overrides.yaml`` (never to Python), recompiles and pushes the
``CompiledFont`` JSON plus the compiled .otf back to every client.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import time
import traceback
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..compile import CompiledFont, compile_font
from ..emit import build_otf
from ..loader import load_font
from ..overrides import DELTAS, STAGES, Override, dump_overrides, load_overrides
from ..terminals import serifs

STATIC = Path(__file__).parent / "static"
HISTORY = 200


class Session:
    def __init__(self, font_path: Path, overrides_path: Path, reference=None):
        self.reference = None
        if reference:
            from ..bench.reference import Reference
            self.reference = Reference(reference)
        self.font_path = Path(font_path).resolve()
        self.ov_path = Path(overrides_path).resolve()
        self.font = load_font(self.font_path)
        self.overrides = load_overrides(self.ov_path)
        self.params: dict[str, float] = {}
        self.terminal_name: str | None = None
        self.undo: list[dict] = []
        self.redo: list[dict] = []
        self.cf: CompiledFont | None = None
        self.otf: bytes = b""
        self.error: str | None = None
        self.version = 0
        self.timing: dict[str, float] = {}
        self._written: str | None = None
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.compile()

    # -- compile ---------------------------------------------------------
    def terminal_styles(self) -> dict:
        return getattr(self.font, "terminal_styles", None) or {"none": serifs.none, "slab": serifs.slab()}

    def _params(self) -> dict[str, float]:
        return {k: v for k, v in self.params.items() if k in self.font.params}

    def compile(self, glyph: str | None = None):
        t0 = time.perf_counter()
        style = self.terminal_styles().get(self.terminal_name) if self.terminal_name else None
        try:
            if glyph and self.cf is not None and glyph in self.cf.glyphs:
                part = compile_font(self.font, self._params(), self.overrides, style, glyphs=[glyph])
                self.cf.glyphs[glyph] = part.glyphs[glyph]
                self.cf.warnings = self._stale_warnings()
                self.cf.compile_ms = part.compile_ms
            else:
                self.cf = compile_font(self.font, self._params(), self.overrides, style)
            t1 = time.perf_counter()
            buf = io.BytesIO()
            build_otf(self.cf, buf)
            self.otf = buf.getvalue()
            self.error = None
            self.timing = {"compile": (t1 - t0) * 1000, "emit": (time.perf_counter() - t1) * 1000}
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        self.version += 1

    def _stale_warnings(self) -> list[str]:
        known = set()
        for g in self.cf.glyphs.values():
            known |= set(g.points) | set(g.outline_points) | {f"{g.name}.lsb", f"{g.name}.rsb"}
        out = []
        for name, o in self.overrides.items():
            hit = name in known or (o.stage == "outline" and
                                    any(k in known for k in (name + ".l", name + ".r")))
            if not hit:
                out.append(f"override {name!r} does not match any point; ignored")
        return out

    def reference_data(self) -> dict:
        """Per glyph: the reference outline aligned to ours, scores, and a
        distance-coloured sampling of our outline (error heat map)."""
        from ..bench.metrics import distance_field, score_glyph
        from ..geometry import svg_path
        out = {}
        t0 = time.perf_counter()
        cache = self.__dict__.setdefault("_ref_cache", {})
        for name, g in self.cf.glyphs.items():
            rname = self.reference.name_for(g.unicode, name)
            if rname is None or not g.contours:
                continue
            key = (name, g.advance, hash(tuple(tuple(p) for c in g.contours for p in c.pts)))
            if key in cache:
                out[name] = cache[key]
                continue
            rc, radv = self.reference.glyph(rname)
            sc = score_glyph(name, g.contours, rc, g.advance, radv, self.cf.upm)
            out[name] = {"svg": svg_path(sc.ref), "iou": round(sc.iou, 4), "hausdorff": round(sc.hausdorff, 1),
                         "mean": round(sc.mean_dist, 2), "advance_delta": sc.advance_delta,
                         "heat": distance_field(g.contours, sc.ref)}
            cache[key] = out[name]
        if len(cache) > 512:
            cache.clear()
        self.timing["reference"] = (time.perf_counter() - t0) * 1000
        return out

    def payload(self) -> dict:
        data = self.cf.to_json() if self.cf else None
        if data is not None:
            if self.reference is not None:
                data["reference"] = self.reference_data()
            data["timing"] = {k: round(v, 1) for k, v in self.timing.items()}
        return {
            "type": "font", "version": self.version, "error": self.error, "data": data,
            "otf": base64.b64encode(self.otf).decode() if self.otf else None,
            "overrides": {n: {**{k: getattr(o, k) for k in DELTAS}, "stage": o.stage}
                          for n, o in self.overrides.items()},
            "terminals": self.terminal_name, "terminal_styles": list(self.terminal_styles()),
            "can_undo": bool(self.undo), "can_redo": bool(self.redo),
            "source": str(self.font_path), "overrides_file": str(self.ov_path),
        }

    # -- overrides -------------------------------------------------------
    def _glyph_of(self, name: str) -> str | None:
        best = None
        for g in self.font.glyphs:
            if name.startswith(g + ".") and (best is None or len(g) > len(best)):
                best = g
        return best

    def checkpoint(self):
        self.undo.append(copy.deepcopy(self.overrides))
        del self.undo[:-HISTORY]
        self.redo.clear()

    def save(self):
        text = dump_overrides(self.overrides)
        self._written = text
        self.ov_path.write_text(text)

    def set_override(self, name: str, values: dict, stage: str = "skeleton", checkpoint: bool = True):
        if stage not in STAGES:
            raise ValueError(f"bad stage {stage!r}")
        if checkpoint:
            self.checkpoint()
        o = Override(name, stage=stage, **{k: float(values.get(k, 0) or 0) for k in DELTAS})
        if o.is_zero():
            self.overrides.pop(name, None)
        else:
            self.overrides[name] = o
        self.save()
        self.compile(self._glyph_of(name))

    def remove(self, names: list[str]):
        self.checkpoint()
        glyphs = set()
        for n in names:
            self.overrides.pop(n, None)
            glyphs.add(self._glyph_of(n))
        self.save()
        self.compile(glyphs.pop() if len(glyphs) == 1 else None)

    def step(self, back: bool):
        src, dst = (self.undo, self.redo) if back else (self.redo, self.undo)
        if not src:
            return
        dst.append(copy.deepcopy(self.overrides))
        self.overrides = src.pop()
        self.save()
        self.compile()

    # -- file watching ---------------------------------------------------
    def on_files_changed(self, paths: set[Path]) -> bool:
        changed = False
        if self.ov_path in paths:
            text = self.ov_path.read_text() if self.ov_path.exists() else ""
            if text != self._written:
                try:
                    self.overrides = load_overrides(self.ov_path)
                    self._written = text
                    changed = True
                except Exception as e:
                    self.error = f"overrides.yaml: {e}"
                    self.version += 1
                    return True
        if any(p.suffix == ".py" for p in paths):
            try:
                self.font = load_font(self.font_path)
                changed = True
            except Exception as e:
                self.error = f"{self.font_path.name}: {type(e).__name__}: {e}"
                self.version += 1
                return True
        if changed:
            self.compile()
        return changed

    # -- messages --------------------------------------------------------
    def handle(self, msg: dict):
        t = msg.get("type")
        if t == "params":
            for k, v in (msg.get("values") or {}).items():
                if k in self.font.params:
                    self.params[k] = float(v)
            self.compile()
        elif t == "reset_params":
            names = msg.get("names")
            if names:
                for n in names:
                    self.params.pop(n, None)
            else:
                self.params.clear()
            self.compile()
        elif t == "terminals":
            self.terminal_name = msg.get("name") or None
            self.compile()
        elif t == "set":
            self.set_override(msg["name"], msg.get("values") or {}, msg.get("stage", "skeleton"),
                              bool(msg.get("checkpoint", True)))
        elif t == "remove":
            self.remove(msg.get("names") or [])
        elif t == "undo":
            self.step(True)
        elif t == "redo":
            self.step(False)
        elif t == "hello":
            pass
        else:
            raise ValueError(f"unknown message type {t!r}")


def create_app(font_path, overrides_path=None, watch: bool = True, reference=None) -> Starlette:
    font_path = Path(font_path).resolve()
    session = Session(font_path, overrides_path or font_path.parent / "overrides.yaml", reference)

    async def broadcast(payload: str):
        for ws in list(session.clients):
            try:
                await ws.send_text(payload)
            except Exception:
                session.clients.discard(ws)

    async def index(request):
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def state(request):
        return JSONResponse(session.payload())

    async def otf(request):
        from starlette.responses import Response
        return Response(session.otf, media_type="font/otf")

    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        session.clients.add(ws)
        try:
            await ws.send_text(json.dumps(session.payload()))
            while True:
                msg = json.loads(await ws.receive_text())
                async with session.lock:
                    try:
                        await asyncio.get_running_loop().run_in_executor(None, session.handle, msg)
                        payload = session.payload()
                    except Exception as e:
                        traceback.print_exc()
                        await ws.send_text(json.dumps({"type": "problem", "message": f"{type(e).__name__}: {e}",
                                                       "seq": msg.get("seq")}))
                        continue
                payload["seq"] = msg.get("seq")
                text = json.dumps(payload)
                await ws.send_text(text)
                others = [c for c in session.clients if c is not ws]
                for c in others:
                    try:
                        await c.send_text(text)
                    except Exception:
                        session.clients.discard(c)
        except WebSocketDisconnect:
            pass
        finally:
            session.clients.discard(ws)

    async def watcher():
        from watchfiles import awatch
        async for changes in awatch(font_path.parent, session.ov_path.parent):
            paths = {Path(p).resolve() for _, p in changes}
            relevant = {p for p in paths if p.suffix == ".py" or p == session.ov_path}
            if not relevant:
                continue
            async with session.lock:
                changed = await asyncio.get_running_loop().run_in_executor(None, session.on_files_changed, relevant)
                if changed:
                    await broadcast(json.dumps(session.payload()))

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(watcher()) if watch else None
        try:
            yield
        finally:
            if task:
                task.cancel()

    app = Starlette(routes=[Route("/", index), Route("/state", state), Route("/font.otf", otf),
                            WebSocketRoute("/ws", ws_endpoint)],
                    lifespan=lifespan)
    app.state.session = session
    return app


def serve(font_path, host="127.0.0.1", port=8765, overrides_path=None, reference=None):
    import uvicorn
    app = create_app(font_path, overrides_path, reference=reference)
    print(f"Ductus preview: http://{host}:{port}/  (watching {Path(font_path).resolve().parent})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
