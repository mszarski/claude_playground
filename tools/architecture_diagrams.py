#!/usr/bin/env python3
"""Diagrams for docs/ARCHITECTURE.md, docs/DEPLOYMENT.md and docs/SIM_CALIBRATION.md.

    tools/architecture_diagrams.py        # writes docs/architecture/*.png

The backend table (which stage runs where) is data in this file, STAGES, and is checked
against the code by tests/python/test_architecture.py: a stage listed as having a CUDA
implementation must expose one.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5fd6"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9, "text.color": INK})

# (group, stage, host implementation, CUDA implementation or None, how it is reached, note)
STAGES = [
    ("Simulator", "Echo synthesis (every scatterer, path and channel)", "C++ kernel, OpenMP", "same kernel, CUDA",
     "core.synthesize_into(backend=)", "the bulk of a simulation"),
    ("Simulator", "Paths: free field, images, sediment refraction", "NumPy", None, "sim.propagation", "small next to synthesis"),
    ("Simulator", "Paths: Bellhop eigenrays", "external program", None, "sim.bellhop", "GPL, separate process"),
    ("Simulator", "Boundary reverberation (scattering function)", "NumPy FFT", None, "sim.reverb", "per ping, per azimuth"),
    ("Simulator", "Ambient noise, hardware gain, ADC", "NumPy", None, "sim.noise, sim.receiver", ""),
    ("Conditioning", "Baseband conversion, pulse compression", "C++ (pocketfft)", "cuFFT", "recon.prepare(backend=)", ""),
    ("Conditioning", "Calibration, TVG, whitening, median TVG, DRC", "C++", None, "core.whiten, core.drc ...", "light"),
    ("Conditioning", "Board synchronisation", "C++ (libs/live) and NumPy", None, "live chain; sync", "once per ping"),
    ("Conditioning", "Array calibration", "NumPy", None, "calibrate", "once per trial"),
    ("Imaging", "Backprojection: fan, stripmap, volume, range x bearing", "C++ kernel, OpenMP", "same kernel, CUDA",
     "core.backproject(backend=), BpSession", "with ray culling; the bulk of processing"),
    ("Imaging", "Far-field FFT beamformer, MVDR", "NumPy", None, "flank", "the live chain uses the kernel instead"),
    ("Imaging", "Motion estimation (SAS)", "C++ and Ceres", None, "motion", "BSD build of Ceres"),
    ("Detection", "CFAR", "C++ kernel", "same kernel, CUDA", "detect.cfar(backend=)", ""),
    ("Detection", "Sparse Bayesian bearings (active gates, passive)", "C++ (Eigen, OpenMP)", None,
     "core.doa_gate, core.doa_passive, doa: sbl", "32 x 32 systems per frequency"),
    ("Detection", "Clustering, tracking, classification", "C++ (libs/live) and NumPy", None, "live chain; detect, classify",
     "tens of objects per ping: nothing to accelerate"),
    ("Live chain", "Board stream to pings, navigation, report", "C++", None, "build/bin/sonar_live",
     "the program that goes on the vehicle"),
]


def box(ax, xy, w, h, text, color=BLUE, text_color="white", fs=8.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=text_color, fontsize=fs)


def arrow(ax, a, b, color=MUTED, style="-|>", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10, lw=lw, color=color, shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def draw_dataflow(path):
    fig, ax = canvas(15.5, 7.6)
    ax.text(0.2, 7.3, "Sources: each yields ping records", color=MUTED, fontsize=10)
    srcs = [("Simulator\nscenario YAML", AQUA), ("Container file\n(.h5) / a tap", AQUA), ("WAV + sidecar\n(file or directory)", AQUA),
            ("Board stream\n(UDP blocks)", AQUA), ("Ultrasound rig\n(NVIDIA reference)", AQUA)]
    for i, (t, c) in enumerate(srcs):
        box(ax, (0.2, 6.0 - i * 1.15), 2.3, 0.85, t, color=c)
        arrow(ax, (2.5, 6.42 - i * 1.15), (3.2, 3.9))
    box(ax, (3.2, 3.3), 1.6, 1.2, "Ping record\nsamples + geometry\n+ waveform + time", color=INK, fs=8)
    chain = [("prepare\nbaseband,\ncompress", 5.3), ("sync,\ncalibrate", 6.9), ("whiten,\nTVG", 8.5), ("motion\n(SAS)", 10.1)]
    arrow(ax, (4.8, 3.9), (5.3, 3.9))
    for (t, x), nxt in zip(chain, [6.9, 8.5, 10.1, None]):
        box(ax, (x, 3.3), 1.3, 1.2, t, color=BLUE, fs=8)
        if nxt:
            arrow(ax, (x + 1.3, 3.9), (nxt, 3.9))
    box(ax, (11.9, 4.9), 1.9, 1.1, "backprojection\nimage / volume /\nrange x bearing", color=ORANGE, fs=8)
    box(ax, (11.9, 3.35), 1.9, 1.1, "detect (CFAR)\ncluster", color=ORANGE, fs=8)
    box(ax, (11.9, 1.8), 1.9, 1.1, "track, classify", color=ORANGE, fs=8)
    arrow(ax, (11.4, 4.1), (11.9, 5.3))
    arrow(ax, (12.85, 4.9), (12.85, 4.45))
    arrow(ax, (12.85, 3.35), (12.85, 2.9))
    for y, t in ((5.45, "GeoTIFF,\ndisplay"), (2.35, "report\n(JSON lines,\nUDP)")):
        box(ax, (14.2, y - 0.45), 1.15, 0.9, t, color=YELLOW, text_color=INK, fs=8)
        arrow(ax, (13.8, y), (14.2, y))
    # Taps under every arrow.
    for x in (5.05, 6.65, 8.25, 9.85, 11.55):
        arrow(ax, (x, 3.75), (x, 2.55), color=VIOLET, lw=1.0)
    arrow(ax, (12.85, 1.8), (12.85, 1.25), color=VIOLET, lw=1.0)
    box(ax, (4.6, 1.75), 7.2, 0.8, "taps (gpusonar/taps.py): the records after any stage as a container file;\nimages, beam maps and reports beside them",
        color=VIOLET, fs=8.5)
    arrow(ax, (4.6, 2.15), (2.5, 5.05), color=VIOLET, rad=-0.2, lw=1.0)
    ax.text(4.6, 1.35, "a tap file is a source: restart the chain from any stage, with simulated or recorded data", color=VIOLET, fontsize=8.5)
    # Simulator's own stages.
    ax.text(0.2, 1.05, "Inside the simulator (sim_capture saves the samples after each):", color=MUTED, fontsize=9)
    sim = ["scene at\nping time", "paths", "echoes", "reverber-\nation", "reference,\nclocks", "noise", "hardware\ngain", "ADC"]
    for i, t in enumerate(sim):
        box(ax, (0.2 + i * 1.3, 0.1), 1.1, 0.75, t, color=AQUA if i > 1 else GRID, text_color="white" if i > 1 else INK, fs=7.5)
        if i:
            arrow(ax, (0.2 + i * 1.3 - 0.2, 0.47), (0.2 + i * 1.3, 0.47))
    ax.text(5.3, 4.75, "Stages: library functions; the same ones are Holoscan operators and the live chain", color=MUTED, fontsize=10)
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def draw_backends(path):
    fig, ax = canvas(15.5, 0.46 * len(STAGES) + 1.4)
    cols = [(0.2, "Stage"), (6.3, "CPU"), (9.0, "GPU"), (11.3, "Where")]
    top = 0.46 * len(STAGES) + 0.9
    for x, t in cols:
        ax.text(x, top, t, fontsize=10, color=MUTED)
    group = None
    for i, (g, stage, host, cuda, where, note) in enumerate(STAGES):
        y = top - 0.5 - i * 0.46
        if g != group:
            ax.plot([0.2, 15.3], [y + 0.23, y + 0.23], color=GRID, lw=0.8)
            group = g
        ax.text(0.2, y, f"{g}: ", fontsize=8.5, color=MUTED, va="center")
        ax.text(1.55, y, stage, fontsize=9, va="center")
        box(ax, (6.3, y - 0.16), 2.5, 0.32, host, color=BLUE, fs=8)
        if cuda:
            box(ax, (9.0, y - 0.16), 2.1, 0.32, cuda, color=AQUA, fs=8)
        else:
            ax.text(10.05, y, "CPU only", fontsize=8, color=MUTED, ha="center", va="center")
        ax.text(11.3, y, where + (f"  ({note})" if note else ""), fontsize=7.5, color=MUTED, va="center")
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def draw_deployment(path):
    fig, ax = canvas(15.5, 6.6)
    ax.text(0.2, 6.3, "Wet end", color=MUTED, fontsize=10)
    box(ax, (0.2, 4.9), 2.2, 0.9, "Projector + amplifier\n4-6 kHz, 197 dB", color=GRID, text_color=INK)
    box(ax, (0.2, 3.5), 2.2, 0.9, "32 hydrophones\n15 cm pitch", color=GRID, text_color=INK)
    box(ax, (0.2, 2.1), 2.2, 0.9, "shared reference\n(sync bursts)", color=GRID, text_color=INK)
    for i in range(4):
        box(ax, (3.2, 5.0 - i * 1.0), 1.9, 0.75, f"Popoto board {i + 1}\n8 inputs, own clock", color=YELLOW, text_color=INK, fs=8)
        arrow(ax, (2.4, 3.95), (3.2, 5.37 - i * 1.0))
        arrow(ax, (2.4, 2.55), (3.2, 5.2 - i * 1.0), color=VIOLET, lw=0.8)
    ax.text(3.2, 6.3, "Capture", color=MUTED, fontsize=10)
    box(ax, (6.0, 4.6), 2.2, 0.9, "files: WAV + sidecar\n(wav_watch source)", color=AQUA, fs=8)
    box(ax, (6.0, 3.1), 2.2, 0.9, "or blocks over UDP\n(flank_udp source)", color=AQUA, fs=8)
    for i in range(4):
        arrow(ax, (5.1, 5.37 - i * 1.0), (6.0, 4.9 if i < 2 else 3.5))
    ax.text(8.9, 6.3, "Vehicle computer: tools/sonar_live.py, one YAML file", color=MUTED, fontsize=10)
    steps = ["record raw\npings (.h5)", "sync boards", "prepare", "beamform", "CFAR,\ncluster", "track,\nclassify"]
    for i, t in enumerate(steps):
        col = INK if i == 0 else BLUE if i < 3 else ORANGE
        box(ax, (8.9 + (i % 3) * 2.15, 4.6 - (i // 3) * 1.5), 1.9, 0.9, t, color=col, fs=8.5)
    for i in range(5):
        a = (8.9 + (i % 3) * 2.15 + 1.9, 5.05 - (i // 3) * 1.5)
        b = (8.9 + ((i + 1) % 3) * 2.15, 5.05 - ((i + 1) // 3) * 1.5)
        if i == 2:
            arrow(ax, (14.15, 4.6), (9.85, 4.0), rad=0.0)
        else:
            arrow(ax, a, b)
    arrow(ax, (8.2, 5.05), (8.9, 5.05))
    arrow(ax, (8.2, 3.55), (8.9, 4.8))
    box(ax, (8.9, 1.2), 1.9, 0.9, "navigation in\n(UDP JSON)", color=GRID, text_color=INK, fs=8.5)
    arrow(ax, (9.85, 2.1), (9.85, 3.1))
    box(ax, (11.05, 1.2), 1.9, 0.9, "report out\n(JSON lines, UDP)", color=YELLOW, text_color=INK, fs=8.5)
    arrow(ax, (14.15, 3.1), (12.6, 2.1))
    box(ax, (13.2, 1.2), 1.9, 0.9, "taps: every stage\nof the first pings", color=VIOLET, fs=8.5)
    ax.text(0.2, 0.75, "The same sonar_live.py runs on a simulated scenario (--scenario), on a recording (--replay) and on an emulated board\n"
            "(--emulate): what is tested ashore is what runs at sea. Grey boxes are hardware or vehicle interfaces not yet confirmed.",
            color=MUTED, fontsize=9)
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def draw_simcal(path):
    fig, ax = canvas(15.5, 5.2)
    box(ax, (0.2, 3.6), 2.6, 1.0, "real FLS log\nping records (absolute)\nor beamformed frames (relative)", color=INK, fs=8)
    box(ax, (0.2, 1.4), 2.6, 1.0, "scenario of the dive\nsonar, depth, water depth,\nwhich boundaries scatter", color=AQUA, fs=8)
    box(ax, (3.6, 3.6), 2.2, 1.0, "image to\nrange x bearing;\nmedian over bearing", color=BLUE, fs=8)
    for i, t in enumerate(("seabed alone", "surface alone", "noise alone")):
        box(ax, (3.6, 2.35 - i * 0.75), 2.2, 0.6, f"simulate {t}", color=AQUA, fs=8)
        arrow(ax, (2.8, 1.9), (3.6, 2.65 - i * 0.75))
        arrow(ax, (5.8, 2.65 - i * 0.75), (6.6, 2.65 - i * 0.75))
        box(ax, (6.6, 2.35 - i * 0.75), 1.9, 0.6, "same imaging", color=BLUE, fs=8)
        arrow(ax, (8.5, 2.65 - i * 0.75), (9.4, 2.9))
    arrow(ax, (2.8, 4.1), (3.6, 4.1))
    arrow(ax, (5.8, 4.1), (9.4, 3.3))
    box(ax, (9.4, 2.5), 2.6, 1.2, "non-negative fit\nlog(r) = sum a_k part_k(r)\nper range bin", color=ORANGE, fs=8.5)
    arrow(ax, (12.0, 3.1), (12.7, 3.1))
    box(ax, (12.7, 2.5), 2.6, 1.2, "scenario overrides\noffset_db per boundary,\nnoise level; what the log\ncould not constrain", color=YELLOW, text_color=INK, fs=8)
    ax.text(0.2, 4.9, "tools/calibrate_sim.py fls: the simulator is linear in each contribution, so each is simulated once and scaled", color=MUTED, fontsize=10)
    box(ax, (0.2, 0.05), 2.6, 0.75, "SAS image (relative)", color=INK, fs=8.5)
    box(ax, (3.6, 0.05), 4.9, 0.75, "bare-seabed blocks: scintillation index, speckle correlation area;\ndarkest tiles", color=BLUE, fs=8)
    box(ax, (9.4, 0.05), 5.9, 0.75, "scatterers per resolution cell -> seafloor density; shadow depth below the seabed", color=YELLOW, text_color=INK, fs=8)
    arrow(ax, (2.8, 0.42), (3.6, 0.42))
    arrow(ax, (8.5, 0.42), (9.4, 0.42))
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def main():
    out = ROOT / "docs" / "architecture"
    out.mkdir(parents=True, exist_ok=True)
    draw_dataflow(out / "dataflow.png")
    draw_backends(out / "backends.png")
    draw_deployment(out / "deployment.png")
    draw_simcal(out / "sim_calibration.png")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
