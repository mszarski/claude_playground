"""``ductus build | test | serve | proof``."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

from .compile import compile_font
from .emit import build_otf, ps_name, validate_otf
from .lint import check_compatibility, parse_sweep, run_lints
from .loader import load_font
from .overrides import load_overrides

app = typer.Typer(add_completion=False, help="Ductus: a parametric font compiler.")


def _overrides_for(font_py: Path, overrides: Optional[Path]):
    path = overrides or font_py.parent / "overrides.yaml"
    return load_overrides(path)


def _parse_set(items: list[str]) -> dict[str, float]:
    out = {}
    for s in items:
        k, _, v = s.partition("=")
        out[k.strip()] = float(v)
    return out


@app.command()
def build(font_py: Path,
          out: Path = typer.Option(Path("out"), "-o", "--out", help="output directory"),
          set_: list[str] = typer.Option([], "--set", help="override a parameter, e.g. --set stem=120"),
          sweep: list[str] = typer.Option([], "--sweep", help="check compatibility, e.g. --sweep stem=40,160"),
          overrides: Optional[Path] = typer.Option(None, help="overrides.yaml (default: next to font.py)"),
          debug_json: bool = typer.Option(False, "--json", help="also write CompiledFont JSON")):
    """Compile FONT_PY to a static .otf."""
    font = load_font(font_py)
    ovs = _overrides_for(font_py, overrides)
    cf = compile_font(font, _parse_set(set_), ovs)
    for w in cf.warnings:
        typer.secho(f"warning: {w}", fg="yellow")
    path = build_otf(cf, out / f"{ps_name(cf)}.otf")
    problems = validate_otf(path)
    typer.echo(f"wrote {path}  ({len(cf.glyphs)} glyphs, compiled in {cf.compile_ms:.0f} ms)")
    if debug_json:
        jp = out / f"{ps_name(cf)}.json"
        jp.write_text(json.dumps(cf.to_json()))
        typer.echo(f"wrote {jp}")
    for p in problems:
        typer.secho(f"invalid: {p}", fg="red")
    failed = bool(problems)
    if sweep:
        settings = parse_sweep(sweep)
        findings = check_compatibility(font, ovs, settings)
        for f in findings:
            typer.secho(str(f), fg="red" if f.level == "error" else None)
        errors = [f for f in findings if f.level == "error"]
        typer.echo(f"sweep: {len(settings)} settings, "
                   f"{'compatible' if not errors else f'{len(errors)} incompatible'}")
        failed |= bool(errors)
    raise typer.Exit(1 if failed else 0)


@app.command()
def test(font_py: Path,
         sweep: list[str] = typer.Option([], "--sweep", help="custom sweep (default: every ranged param at both ends)"),
         overrides: Optional[Path] = typer.Option(None)):
    """Run the lints: compile, overlaps, direction, stems, compatibility, overrides."""
    t0 = time.perf_counter()
    font = load_font(font_py)
    findings, cf = run_lints(font, _overrides_for(font_py, overrides), parse_sweep(sweep) if sweep else None)
    if cf is not None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for p in validate_otf(build_otf(cf, Path(d) / "t.otf")):
                from .lint import Finding
                findings.append(Finding("otf", "error", p))
    colors = {"error": "red", "warning": "yellow", "info": None}
    for f in findings:
        typer.secho(str(f), fg=colors[f.level])
    errors = sum(f.level == "error" for f in findings)
    typer.secho(f"{'FAIL' if errors else 'PASS'}: {errors} error(s), "
                f"{sum(f.level == 'warning' for f in findings)} warning(s) "
                f"in {time.perf_counter() - t0:.1f}s", fg="red" if errors else "green", bold=True)
    raise typer.Exit(1 if errors else 0)


@app.command()
def serve(font_py: Path,
          port: int = typer.Option(8765, "--port", "-p"),
          host: str = typer.Option("127.0.0.1", "--host"),
          reference: Optional[Path] = typer.Option(None, help="reference font for onion skin + error heat map"),
          overrides: Optional[Path] = typer.Option(None)):
    """Live editor: sliders, draggable points, text in the compiled font."""
    from .serve import serve as run
    run(font_py, host=host, port=port, overrides_path=overrides, reference=reference)


def _default_reference(font_py: Path) -> Optional[Path]:
    for cand in (font_py.parent / "reference" / "GeistSerif-Regular.woff2",
                 Path("benchmarks/geist_serif/reference/GeistSerif-Regular.woff2")):
        if cand.exists():
            return cand
    return None


@app.command()
def bench(font_py: Path,
          reference: Optional[Path] = typer.Option(None, help="reference font (default: benchmarks/geist_serif/reference)"),
          glyphs: str = typer.Option("", help="comma-separated glyph names (default: all non-empty glyphs)"),
          docs: Path = typer.Option(Path("docs"), help="where representability.md and iterations/ live"),
          note: str = typer.Option("", help="notes column for the report"),
          iteration: Optional[int] = typer.Option(None, help="iteration number (default: next)"),
          report: bool = typer.Option(True, help="append rows + overlays to docs/"),
          overrides: Optional[Path] = typer.Option(None)):
    """Score glyphs against a reference: IoU, Hausdorff, override count."""
    from .bench.reference import Reference
    from .bench.report import append_rows, next_iteration, overlay_image
    from .bench.run import override_stats, score_font
    ref_path = reference or _default_reference(font_py)
    if ref_path is None:
        raise typer.BadParameter("no reference font; run benchmarks/geist_serif/fetch_reference.py or pass --reference")
    font = load_font(font_py)
    ovs = _overrides_for(font_py, overrides)
    cf = compile_font(font, overrides=ovs)
    names = [g.strip() for g in glyphs.split(",") if g.strip()] or [n for n, g in cf.glyphs.items() if g.contours]
    scores = score_font(cf, Reference(ref_path), names)
    it = iteration or next_iteration(docs)
    rows = []
    for g, s in scores.items():
        n, mag = override_stats(g, ovs)
        prims = ", ".join(p.split(":", 1)[1] for p in cf.glyphs[g].primitives if p.startswith(("contour:", "terminal:")))
        prims = prims or ", ".join(p.split(":", 1)[-1] for p in cf.glyphs[g].primitives)
        typer.echo(f"{g:8} IoU {s.iou:.4f}  Hausdorff {s.hausdorff:6.1f}  mean {s.mean_dist:5.1f}  "
                   f"Δadv {s.advance_delta:+4d}  overrides {n} (Σ|Δ| {mag:.0f})")
        rows.append({"glyph": g, "iou": s.iou, "hausdorff": s.hausdorff, "mean_dist": s.mean_dist,
                     "advance_delta": s.advance_delta, "overrides": n, "magnitude": mag,
                     "primitives": prims, "notes": note})
        if report:
            overlay_image(s, docs / "iterations" / f"{it:02d}" / f"{g}.png", title=f"#{it:02d}  ")
    if report:
        typer.echo(f"report: {append_rows(docs, it, rows)}  overlays: {docs / 'iterations' / f'{it:02d}'}")


@app.command()
def fit(font_py: Path,
        glyph: str = typer.Option(..., "--glyph", help="glyph, or comma-separated glyphs fitted jointly"),
        reference: Optional[Path] = typer.Option(None),
        param: list[str] = typer.Option([], "--param", help="params or globs (n_*); default: all ranged params that affect the glyph"),
        maxiter: int = typer.Option(400),
        samples: int = typer.Option(256),
        restarts: int = typer.Option(3),
        exclude: str = typer.Option("xheight,capheight,ascender,descender",
                                    help="params never fitted (default: the fixed vertical metrics)"),
        save: Optional[Path] = typer.Option(None, help="merge fitted values into this YAML file"),
        method: str = typer.Option("nelder-mead", help="nelder-mead (default) or powell (polishing)"),
        only_improve: bool = typer.Option(False, "--only-improve",
                                          help="save only if every fitted glyph improves its IoU and stays within target"),
        objective: str = typer.Option("distance", help="distance (smooth; default), iou (polishing), or target (joint fits: no glyph may drop below target)"),
        overrides: Optional[Path] = typer.Option(None)):
    """Tune parameters (not overrides) to minimise error vs the reference."""
    import yaml
    from .bench.reference import Reference
    from .bench.run import fit_glyph
    ref_path = reference or _default_reference(font_py)
    if ref_path is None:
        raise typer.BadParameter("no reference font")
    font = load_font(font_py)
    glyphs = [g.strip() for g in glyph.split(",") if g.strip()]
    r = fit_glyph(font, Reference(ref_path), glyphs, _overrides_for(font_py, overrides), param or None, maxiter,
                  samples=samples, restarts=restarts, method=method, objective_kind=objective,
                  exclude={e.strip() for e in exclude.split(",") if e.strip()})
    for g, s in r.scores.items():
        typer.echo(f"{g}: IoU {s.iou:.4f}  Hausdorff {s.hausdorff:.1f}  mean {s.mean_dist:.2f}  dadv {s.advance_delta:+d}")
    typer.echo(f"({r.evaluations} evaluations; first glyph before: IoU {r.before.iou:.4f})" if r.before else "")
    for k, v in r.params.items():
        typer.echo(f"  {k} = {v:g}")
    if save and only_improve:
        from .bench.run import score_font
        before = score_font(compile_font(font, overrides=_overrides_for(font_py, overrides), glyphs=glyphs),
                            Reference(ref_path), glyphs)
        worse = [g for g, sc in r.scores.items()
                 if sc.iou <= before[g].iou or sc.iou < 0.93 or sc.hausdorff > 20]
        if worse:
            typer.secho(f"not saved: no improvement (or below target) for {', '.join(worse)}", fg="yellow")
            save = None
    if save:
        data = yaml.safe_load(save.read_text()) if save.exists() else {}
        data = data or {}
        data.update({k: float(v) for k, v in r.params.items()})
        save.write_text(yaml.safe_dump(dict(sorted(data.items())), sort_keys=False))
        typer.echo(f"saved -> {save}")


@app.command()
def proof(font_py: Path,
          out: Path = typer.Option(Path("docs/contact_sheet.png"), "-o", "--out"),
          setting: list[str] = typer.Option([], "--setting",
                                            help='row as "Label:stem=40,contrast=0.2"; repeatable'),
          overrides: Optional[Path] = typer.Option(None)):
    """Render a contact sheet of every glyph at one or more parameter settings."""
    from .proof import contact_sheet
    font = load_font(font_py)
    ovs = _overrides_for(font_py, overrides)
    rows = []
    for s in setting or ["defaults:"]:
        label, _, kv = s.partition(":")
        params = _parse_set([x for x in kv.split(",") if x.strip()])
        rows.append((label.replace("\\n", "\n"), compile_font(font, params, ovs)))
    contact_sheet(rows, out)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
