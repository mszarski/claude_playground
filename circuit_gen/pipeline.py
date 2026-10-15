"""
End-to-end pipeline: circuit code -> checks -> simulation -> board -> routing -> DRC -> Gerbers.

Each stage writes its outputs into the run directory and records its result in
report.json, so a failed run shows how far it got.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from .checks import check_circuit
from .runner import run_circuit_code

STAGES = ["circuit", "checks", "placement", "routing", "drc", "export"]
SIM_STAGE = "simulation"   # only when the board has functional tests


def _layout(circuit, out_dir, report, max_passes):
    # pcbnew is imported here so the circuit stages work without KiCad's Python module.
    import pcbnew
    from .layout.board import build_board, board_size_mm
    from .layout.route import route_board, run_drc, export_outputs

    board, items = build_board(circuit)
    w, h = board_size_mm(items)
    report["stages"]["placement"] = {"ok": True, "board_mm": [round(w, 1), round(h, 1)]}
    pcbnew.SaveBoard(str(out_dir / "placed.kicad_pcb"), board)

    routed = route_board(board, out_dir, max_passes=max_passes)
    report["stages"]["routing"] = asdict(routed)
    if not routed.ok:
        return
    board_path = out_dir / "board.kicad_pcb"
    pcbnew.SaveBoard(str(board_path), board)

    drc = run_drc(board_path, out_dir / "drc.rpt")
    report["stages"]["drc"] = {"ok": drc.ok, "errors": drc.errors, "warnings": drc.warnings,
                               "unconnected": drc.unconnected}

    exported = export_outputs(board_path, out_dir)
    report["stages"]["export"] = {"ok": not exported["errors"], **exported}


def build(code, out_dir, max_passes=30, require_checks=True, tests=None):
    """Run the whole pipeline on SKiDL `code`. Returns the report dict.

    `tests` are SPICE test specs (see spice.py). When given, the board only
    counts as a success if they pass too.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.py").write_text(code)
    start = time.time()
    report = {"stages": {}}

    run = run_circuit_code(code, out_dir)
    report["stages"]["circuit"] = {"ok": run.ok, "error": run.error,
                                   "erc_errors": run.erc_errors, "erc_warnings": run.erc_warnings}
    if run.ok:
        checks = check_circuit(run.circuit)
        report["stages"]["checks"] = {"ok": checks.ok, "errors": checks.errors,
                                      "warnings": checks.warnings}
        if tests:
            from .spice import simulate
            sim = simulate(run.circuit, tests)
            report["stages"][SIM_STAGE] = {
                "ok": sim.ok, "unmodeled": sim.unmodeled,
                "results": [{"name": r.name, "passed": r.passed, "value": r.value,
                             "expect": r.expect, "error": r.error} for r in sim.results]}
        if checks.ok or not require_checks:
            try:
                _layout(run.circuit, out_dir, report, max_passes)
            except Exception as e:  # report layout failures rather than crash an eval run
                report["stages"]["layout_error"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    required = STAGES + ([SIM_STAGE] if tests else [])
    report["completed"] = [s for s in required if report["stages"].get(s, {}).get("ok")]
    report["success"] = report["completed"] == required
    report["functionally_tested"] = bool(tests)
    report["seconds"] = round(time.time() - start, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
