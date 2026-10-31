"""``ductus build | test | proof`` (``serve`` lands next)."""
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
