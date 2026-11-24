#!/usr/bin/env python3
"""The backend table of docs/handoff/SOFTWARE.md, drawn from the source tree.

    tools/software_diagrams.py        # writes docs/handoff/img/backends_head.png

One row per processing stage: the host source file, the CUDA translation unit when the stage has one, and how
the backend is chosen. Nothing here is typed from memory: every file named in ROWS must exist, and every `.cu`
file under libs/ must be named by a row, or the tool stops and says which. So a stage that gains a device build
cannot be left out, and a row cannot claim one that is not in the tree. It imports nothing of gpusonar and needs
no build. (docs/architecture/backends.png, drawn by tools/architecture_diagrams.py from a typed table, was corrected to
the same content on 2026-11-24; this tool's check is what keeps the two from drifting.)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (group, stage, host file, CUDA file or None, how the backend is chosen)
ROWS = [
    ("Simulator", "Echo synthesis: every scatterer, path and channel", "libs/sim/src/synth.cpp", "libs/sim/src/synth_cuda.cu",
     "run_scenario(backend=)"),
    ("Simulator", "Boundary reverberation: time series from the power stage", "libs/sim/src/reverb.cpp", "libs/sim/src/reverb_cuda.cu",
     "sim.reverb add_reverberation(backend=)"),
    ("Simulator", "Path geometry: free field, images, sediment", "libs/sim/src/paths.cpp", None, "host only"),
    ("Conditioning", "Baseband conversion, pulse compression", "libs/dsp/src/pulse_compression.cpp", "libs/dsp/src/dsp_cuda.cu",
     "recon.prepare(backend=)"),
    ("Conditioning", "Whitening, TVG, median TVG, DRC", "libs/dsp/src/conditioning.cpp", None, "host only"),
    ("Conditioning", "Decimate first, board synchronisation", "libs/live/src/decimate.cpp", None, "host only"),
    ("Imaging", "Backprojection: fan, stripmap, volume, range x bearing", "libs/recon/src/backproject.cpp",
     "libs/recon/src/backproject_cuda.cu", "core.backproject(backend=)"),
    ("Imaging", "Far-field and MVDR beamformers", "libs/live/src/beamform.cpp", "libs/live/src/beamform_cuda.cu",
     "chain beamformer: farfield | mvdr"),
    ("Imaging", "Motion estimation: delays between pings", "libs/motion/src/delay_estimation.cpp", "libs/motion/src/delay_cuda.cu",
     "MotionOptions.delay_backend (default numpy)"),
    ("Imaging", "Motion estimation: the Ceres solve (ASASIN Eq. 8)", "libs/motion/src/dpc_solver.cpp", None, "host only"),
    ("Detection", "Cell-averaging CFAR", "libs/detect/src/cfar.cpp", "libs/detect/src/cfar_cuda.cu", "detect.cfar(backend=)"),
    ("Detection", "Multiscale detector: the cell statistic", "libs/live/src/detector.cpp", "libs/live/src/detector_cuda.cu",
     "chain backend (flagging, grouping on the host)"),
    ("Detection", "Sparse Bayesian bearings", "libs/live/src/doa.cpp", "libs/live/src/doa_cuda.cu", "chain doa: sbl"),
    ("Detection", "Clustering", "libs/live/src/cluster.cpp", None, "host only"),
    ("Detection", "Tracking", "libs/live/src/tracker.cpp", None, "host only"),
    ("Detection", "Classification, echo grouping", "libs/live/src/classify.cpp", None, "host only"),
    ("Detection", "Depth: fold, ray trace, sigma", "libs/live/src/elevation.cpp", None, "host only (steering: backprojection)"),
    ("Program", "Board blocks to pings, navigation", "libs/live/src/udp_source.cpp", None, "host only"),
]


def check() -> list[str]:
    """What is wrong with ROWS against the tree: a named file that is missing, a .cu file no row names."""
    wrong = []
    for _, stage, host, cuda, _ in ROWS:
        for f in (host, cuda):
            if f is not None and not (ROOT / f).is_file():
                wrong.append(f"{stage}: {f} is not in the tree")
    named = {cuda for *_, cuda, _ in ROWS if cuda}
    for f in sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "libs").rglob("*.cu")):
        if f not in named:
            wrong.append(f"{f}: a CUDA translation unit no row names")
    return wrong


def draw(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    blue, aqua, ink, muted, surface, grid = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
    plt.rcParams.update({"figure.facecolor": surface, "savefig.facecolor": surface, "font.size": 9, "text.color": ink})
    h = 0.46 * len(ROWS) + 1.5
    fig, ax = plt.subplots(figsize=(15.5, h))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, h)
    ax.axis("off")
    top = h - 0.45
    for x, t in ((0.2, "Stage"), (6.9, "Host (C++, OpenMP)"), (9.9, "CUDA translation unit"), (12.5, "Chosen by")):
        ax.text(x, top, t, fontsize=10, color=muted)
    group = None
    for i, (g, stage, host, cuda, how) in enumerate(ROWS):
        y = top - 0.5 - i * 0.46
        if g != group:
            ax.plot([0.2, 15.3], [y + 0.23, y + 0.23], color=grid, lw=0.8)
            ax.text(0.2, y, g, fontsize=8.5, color=muted, va="center")
            group = g
        ax.text(1.55, y, stage, fontsize=8.6, va="center")
        for x, w, name, colour in ((6.9, 2.8, host, blue), (9.9, 2.4, cuda, aqua)):
            if name:
                ax.add_patch(FancyBboxPatch((x, y - 0.16), w, 0.32, boxstyle="round,pad=0.02,rounding_size=0.08", fc=colour, ec="none"))
                ax.text(x + w / 2, y, name.replace("libs/", "").replace("src/", ""), ha="center", va="center", color="white", fontsize=7.6)
            else:
                ax.text(x + w / 2, y, "none", ha="center", va="center", color=muted, fontsize=8)
        ax.text(12.5, y, how, fontsize=7.6, color=muted, va="center")
    n = sum(1 for *_, cuda, _ in ROWS if cuda)
    ax.text(0.2, 0.25, f"{n} CUDA translation units under libs/ (every *.cu file in the tree is on this table; tools/software_diagrams.py checks it). "
            "Paths are under libs/, without src/.", fontsize=8, color=muted)
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check the table against the tree; write nothing")
    args = ap.parse_args(argv)
    wrong = check()
    if wrong:
        print("\n".join(wrong), file=sys.stderr)
        return 1
    if args.check:
        print(f"{len(ROWS)} rows agree with the tree")
        return 0
    out = ROOT / "docs" / "handoff" / "img"
    out.mkdir(parents=True, exist_ok=True)
    draw(out / "backends_head.png")
    print("wrote", (out / "backends_head.png").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
