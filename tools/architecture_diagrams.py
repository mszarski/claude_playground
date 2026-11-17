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
    ("Simulator", "Paths: free field, images, sediment refraction", "C++ (NumPy kept)", None, "sim.propagation(engine=)", "small next to synthesis"),
    ("Simulator", "Paths: Bellhop eigenrays", "external program", None, "sim.bellhop", "GPL, separate process"),
    ("Simulator", "Boundary reverberation: sum over sectors into channels", "C++ kernel (NumPy kept)", "same kernel, CUDA",
     "sim.reverb(backend=, engine=)", "its FFTs and the power stage stay on the CPU"),
    ("Simulator", "Ambient noise, hardware gain, ADC", "NumPy", None, "sim.noise, sim.receiver", ""),
    ("Conditioning", "Baseband conversion, pulse compression", "C++ (pocketfft)", "cuFFT", "recon.prepare(backend=)", ""),
    ("Conditioning", "Calibration, TVG, whitening, median TVG, DRC", "C++", None, "core.whiten, core.drc ...", "light"),
    ("Conditioning", "Board synchronisation", "C++ (libs/live) and NumPy", None, "live chain; sync", "once per ping"),
    ("Conditioning", "Array calibration", "C++ (libs/live) and NumPy", None, "calibrate; core.calibrate_*_cpp", "once per trial"),
    ("Imaging", "Backprojection: fan, stripmap, volume, range x bearing", "C++ kernel, OpenMP", "same kernel, CUDA",
     "core.backproject(backend=), BpSession", "with ray culling; the bulk of processing"),
    ("Imaging", "Far-field beamformer, MVDR", "C++ (libs/live) and NumPy", None, "beamformer: farfield | mvdr; flank",
     "the live chain's default is the kernel"),
    ("Imaging", "Motion estimation (SAS)", "C++ and Ceres", None, "motion", "BSD build of Ceres"),
    ("Detection", "CFAR", "C++ kernel", "same kernel, CUDA", "detect.cfar(backend=)", ""),
    ("Detection", "Sparse Bayesian bearings (active gates, passive)", "C++ kernels, OpenMP", "same kernels, CUDA",
     "core.doa_gate(backend=), core.doa_passive, doa: sbl", "2.5x on an L4"),
    ("Detection", "Stacked rows: per-row maps, elevation at each detection", "C++ (libs/live) and NumPy on the backprojection kernel", "same kernel, CUDA",
     "chain: {elevation: {...}}", "no kernel of its own; depth, fold and ray trace are CPU"),
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


# The live chain, stage by stage: (key, label, where it runs, CUDA-capable?, note). CUDA-capable is what the code shows
# (the stage takes the chain's `backend`) and what docs/benchmarks/gpu_benchmark_all_l4.md timed; test_architecture.py checks it.
LIVE_STAGES = [
    ("record", "record raw ping\n(.h5, with pose)", "both", False, "before any processing"),
    ("sync", "synchronise boards\n(reference bursts)", "both", False, "per-board offset, drift"),
    ("prepare", "baseband,\npulse compression", "both", True, "cuFFT on the device"),
    ("rows", "split into rows\n(from positions)", "both", False, "one row: skipped"),
    ("map", "range x bearing map\nper row, powers added", "both", True, "backprojection kernel"),
    ("cfar", "CFAR on the\nsummed map", "both", True, "kernel, host or CUDA"),
    ("cluster", "cluster to\ndetections", "both", False, "tens of objects"),
    ("sbl", "doa: sbl  bearings\nat each range gate", "cpp", True, "<= 64 channels a row"),
    ("elevation", "elevation: steer all rows\nat each detection;\ncandidates if repeats", "both", True, "backprojection kernel"),
    ("depth", "depth: attitude at arrival,\nfold, ray trace; sigma from\nnoise, roll, refraction", "both", False, "sound speed, water depth"),
    ("track", "tracker\n(range, bearing, depth)", "both", False, "ownship motion removed"),
    ("classify", "classify: mover, static,\nmultipath", "both", False, ""),
    ("report", "report: JSON line\n(file, UDP)", "both", False, ""),
]


def draw_live_chain(path):
    """The long-range array's chain with stacked rows: both engines, active and passive, GPU-capable stages marked."""
    from gpusonar import live
    fig, ax = canvas(15.5, 8.6)
    ax.text(0.2, 8.3, "Active: one ping at a time (gpusonar/live.py LiveChain; libs/live Chain; build/bin/sonar_live)", color=MUTED, fontsize=10)
    box(ax, (0.2, 7.0), 1.9, 0.9, "board blocks (UDP)\nor recording / simulator", color=AQUA, fs=7.8)
    box(ax, (0.2, 5.75), 1.9, 0.9, "navigation: pose, attitude\nsamples, sound speed,\nwater depth (UDP JSON)", color=GRID, text_color=INK, fs=7.2)
    decimate = "decimate" in live.DEFAULTS or "decimate_first" in live.DEFAULTS
    x0, w, h, gap = 2.5, 1.72, 0.95, 0.14
    pos = {}
    for i, (key, label, engines, cuda, note) in enumerate(LIVE_STAGES):
        row, col = divmod(i, 7)
        x, y = x0 + col * (w + gap), 7.0 - row * 2.05
        pos[key] = (x, y)
        colour = AQUA if cuda else (BLUE if key not in ("record", "report") else INK)
        box(ax, (x, y), w, h, label, color=colour, fs=7.3)
        ax.text(x + w / 2, y - 0.16, note + (" | C++ engine only" if engines == "cpp" else ""), ha="center", fontsize=6.6, color=MUTED)
        if col:
            arrow(ax, (x - gap, y + h / 2), (x, y + h / 2))
    arrow(ax, (2.1, 7.45), (2.5, 7.45))
    ax.text(pos["cluster"][0] + w / 2, 6.45, "continues below", ha="center", fontsize=7, color=MUTED)
    ax.text(pos["sbl"][0] - 0.12, pos["sbl"][1] + h / 2, ">", ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(0.2, 5.6, "navigation is written into each\nrecord (pose, attitude samples,\nenvironment); the depth stage\nreads it from the record", fontsize=6.8, color=MUTED, va="top")
    state = "in this tree (chain key present)" if decimate else "not in this tree: being added separately (stage 2)"
    ax.text(pos["prepare"][0], 8.02, f"decimate first (to twice the band before the chain): {state}", fontsize=7, color=VIOLET)
    # legend and engines
    box(ax, (0.2, 3.45), 1.2, 0.4, "CPU stage", color=BLUE, fs=7.5)
    box(ax, (1.5, 3.45), 1.7, 0.4, "host or CUDA kernel", color=AQUA, fs=7.5)
    ax.text(3.4, 3.65, "The same stages exist twice and are held together by tests: `engine: python` (NumPy on the C++ kernels, a tap after every stage) and `engine: cpp` (libs/live, what build/bin/sonar_live runs).\n"
            "GPU-capable = the stage takes the chain's `backend` and was timed host against CUDA in docs/benchmarks/gpu_benchmark_all_l4.md (live chain, 1 / 2 / 3 rows).", fontsize=7.6, color=MUTED, va="center")
    ax.plot([0.2, 15.3], [3.2, 3.2], color=GRID, lw=0.8)
    ax.text(0.2, 2.9, "Passive: no transmission (build/bin/sonar_live --listen / --passive; core.doa_passive)", color=MUTED, fontsize=10)
    passive = [("board stream cut into\nstretches of one frame", AQUA, False), ("record the stretch", INK, False), ("spectra per channel,\naveraged over snapshots", BLUE, False),
               ("bearings: sparse Bayesian\nkernels (or conventional)", AQUA, True), ("elevation of each bearing\n(stacked rows)", BLUE, False), ("frame: bearings, powers,\nnoise, elevations (JSON)", INK, False)]
    for i, (label, colour, _) in enumerate(passive):
        x = 0.2 + i * 2.55
        box(ax, (x, 1.6), 2.3, 0.95, label, color=colour, fs=7.5)
        if i:
            arrow(ax, (x - 0.25, 2.07), (x, 2.07))
    ax.text(0.2, 1.25, "Passive runs in the C++ library only (the Python chain has no listen mode; `gpusonar.elevation.passive_elevation` is the NumPy reference of the elevation step).\n"
            "The sparse Bayesian kernels take at most 64 channels in one row and refuse more with an error; with stacked rows the solver works on the first row.", fontsize=7.6, color=MUTED, va="top")
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def draw_build_deployment(path, sonar="popoto_long_range_3row", deploy="popoto_vehicle_elevation"):
    """Deployment view of one build, from its files: programs, ports, files, service, configs."""
    from gpusonar import buildspec
    d = buildspec.describe_shipped(sonar, deploy)
    dep = d["deploy"]
    src, out, nav = dep["source"], dep["output"], dep.get("nav") or {}
    fig, ax = canvas(15.5, 6.4)
    ax.text(0.2, 6.1, f"Deployment view: {sonar} with configs/deploy/{deploy}.yaml", color=MUTED, fontsize=10)
    box(ax, (0.2, 4.4), 2.6, 1.1, f"{d['n_boards']} capture boards\n{d['channels']} channels, {d['sample_rate_hz'] / 1e3:g} kHz\n{d['rate_total_bit_s'] / 1e6:.0f} Mbit/s", color=YELLOW, text_color=INK, fs=8)
    box(ax, (0.2, 2.9), 2.6, 1.0, "navigation unit\n" + ("typed-in pose (nav.static)" if "static" in nav else f"JSON datagrams, UDP {nav.get('port')}"), color=GRID, text_color=INK, fs=8)
    box(ax, (0.2, 1.4), 2.6, 1.0, "vehicle computer / operator\n" + (f"report lines, UDP {out['report_udp'][0]}:{out['report_udp'][1]}" if out.get("report_udp") else "report file"), color=GRID, text_color=INK, fs=8)
    box(ax, (4.0, 3.5), 4.2, 2.0, "tools/sonar_live.py DEPLOY.yaml   (Python, either engine)\nor\nbuild/bin/sonar_live --udp PORT --template T.h5 --options chain.options\n(C++, no Python at run time; systemd unit configs/deploy/sonar_live.service)", color=BLUE, fs=7.6)
    arrow(ax, (2.8, 4.95), (4.0, 4.8))
    ax.text(3.4, 5.1, f"UDP {src['port']}", fontsize=7.5, color=MUTED, ha="center")
    arrow(ax, (2.8, 3.4), (4.0, 4.0))
    arrow(ax, (4.0, 3.7), (2.8, 1.9))
    box(ax, (4.0, 1.4), 4.2, 1.5, "also the same program:\n--scenario NAME (simulator)  --replay FILE (a recording)\n--emulate NAME (simulated pings as board blocks over UDP)\n--listen / --passive (C++ program: passive bearings)", color=VIOLET, fs=7.6)
    files = [("configs/sonars/" + src["sonar"] + ".yaml", "array, transmit, sampling, window, board clocks"), ("configs/deploy/" + deploy + ".yaml", "source, nav, chain, output"),
             ("template.h5 + chain.options", "the same two files for the C++ program (--export-template)"), (out.get("recording", "-"), "every raw ping with its pose"),
             (out.get("report", "-"), "one JSON line a ping"), ((out.get("taps") or {}).get("dir", "-"), f"every stage of the first {(out.get('taps') or {}).get('pings', 0)} pings"),
             ("configs/builds/<build>.yaml", "the spec all of the above is generated from (tools/build_from_spec.py)")]
    ax.text(9.0, 5.55, "Files", color=MUTED, fontsize=10)
    for i, (name, what) in enumerate(files):
        y = 5.0 - i * 0.62
        box(ax, (9.0, y), 3.9, 0.48, name, color=INK if i in (3, 4, 5) else AQUA, fs=7.2)
        ax.text(13.0, y + 0.24, what, fontsize=7.3, color=MUTED, va="center")
    ax.text(0.2, 0.75, "Yellow: capture hardware. Grey: platform interfaces. Black: written at run time. Ports and paths are read from the deployment file; nothing here is typed in.", color=MUTED, fontsize=8.5)
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
    draw_live_chain(out / "live_chain.png")
    draw_build_deployment(out / "deployment_build.png")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
