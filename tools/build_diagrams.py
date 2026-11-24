#!/usr/bin/env python3
"""Hardware and timing diagrams of a build, drawn from its config files.

    tools/build_diagrams.py                      # the six stages of the Popoto plan -> docs/design/build_*.png
    tools/build_diagrams.py SONAR [DEPLOY] --out DIR

Both figures are functions of `gpusonar.buildspec.describe(...)`, i.e. of a sonar file (with its `platform` and
`hardware` blocks), its `_clocks` variant and its deployment file, so tools/build_from_spec.py draws them for any
build. Every number on them is computed from those files; a value the files do not hold is drawn as "not chosen"
or "to be supplied", never filled in.

  hardware   the mounting face to scale (rows, hydrophones coloured by capture board, usable zone and margins,
             projector), then the system: hydrophones -> boards row by row, the synchronisation reference to every
             board, projector and amplifier, network to the compute unit, navigation / attitude / sound speed /
             water depth, storage, the link to the vehicle computer (or the surface cable of a static array), power.
  timing     one ping on a time axis: transmit, the projector's direct sound at the hydrophones and the blind
             range, the receive window, the synchronisation bursts; data rates and storage.
"""
import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from gpusonar import buildspec  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5fd6"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
BOARD_COLOURS = [BLUE, ORANGE, AQUA, YELLOW, VIOLET, "#c2417a"]
RC = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9, "text.color": INK,
      "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED}

STAGES = [   # (label, sonar file, deployment file, file stem): the staged Popoto plan
    ("1: test AUV, 1 x 32 (short row)", "popoto_testauv_1row", "popoto_testauv_1row", "testauv_1x32"),
    ("2: static array, 2 x 32", "popoto_static_2row", "popoto_static_2row", "static_2x32"),
    ("3: static array, 3 x 32", "popoto_static_3row", "popoto_static_3row", "static_3x32"),
    ("4: full face, 1 x 32", "popoto_face_1row", "popoto_face_1row", "face_1x32"),
    ("5: full face, 2 x 32", "popoto_long_range_2row", "popoto_face_2row", "face_2x32"),
    ("6: full face, 3 x 32", "popoto_long_range_3row", "popoto_vehicle_elevation", "face_3x32"),
]


def box(ax, xy, w, h, text, color=BLUE, text_color="white", fs=8):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=text_color, fontsize=fs)


def arrow(ax, a, b, color=MUTED, lw=1.1, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, lw=lw, color=color, shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}"))


def mbit(v):
    return f"{v / 1e6:.1f} Mbit/s"


def size(v):
    return f"{v / 1e9:.2f} GB" if v >= 1e9 else f"{v / 1e6:.0f} MB"


def named(value, missing="not chosen"):
    return str(value) if value else missing


TITLE_CHARS = 185      # a heading line's length: 185 characters of this kind of text at 9 pt are 10.8 in of the panel's 13.95 in


def spread(values, need: float) -> list[float]:
    """The values in ascending order, moved apart until neighbours are at least `need` apart, about their own middle:
    where to write labels that would overlap at the values themselves. Unchanged when they are far enough apart."""
    v = sorted(float(x) for x in values)
    if all(b - a >= need for a, b in zip(v, v[1:])):
        return v
    out = [v[0]]
    for x in v[1:]:
        out.append(max(x, out[-1] + need))
    shift = (v[0] + v[-1]) / 2 - (out[0] + out[-1]) / 2
    return [x + shift for x in out]


def draw_hardware(d: dict, path, title: str | None = None) -> None:
    """The face to scale and the system block diagram of one build."""
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(15.5, 10.4))
        plat, hw = d["platform"], d["hardware"]
        span, height = float(plat.get("span_m", d["row_length_m"] + 1.0)), float(plat.get("height_m", 1.0))
        usable = float(plat.get("usable_span_m", span))
        usable_h = float(plat.get("usable_height_m", height))
        pos, tx = d["positions"], np.asarray(d["projector_m"])
        # --- the face, to scale ---
        ax = fig.add_axes([0.05, 0.60, 0.90, 0.33])
        ax.add_patch(Rectangle((-span / 2, -height / 2), span, height, fc="#eef0ee", ec=MUTED, lw=1.0))
        ax.add_patch(Rectangle((-usable / 2, -usable_h / 2), usable, usable_h, fc="#dfe9f6", ec=BLUE, lw=0.8, ls="--"))
        for i, p in enumerate(pos):
            ax.plot(p[0], -p[2], "o", ms=3.2 if d["channels"] > 40 else 4.5, color=BOARD_COLOURS[int(d["channel_board"][i]) % len(BOARD_COLOURS)])
        ax.plot(tx[0], -tx[2], marker="s", ms=11, color=INK)
        ax.annotate("projector\n" + f"[{tx[0]:g}, {tx[1]:g}, {tx[2]:g}] m", (tx[0], -tx[2]), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
        bp = d["cables"]["assumed_board_position_m"]
        ax.plot(bp[0], -bp[2], marker="P", ms=10, color=VIOLET)
        ax.annotate("boards (assumed)", (bp[0], -bp[2]), textcoords="offset points", xytext=(8, -12), fontsize=8, color=VIOLET)
        lim_x = max(span / 2, abs(tx[0]) + 0.2) * 1.08
        lim_y = max(height / 2, abs(tx[2]) + 0.1) * 1.25
        # A face much taller than wide (a projector hung far below a short frame) would leave a narrow box with the
        # labels outside it: the box keeps the panel's shape instead, the scale being set by the height either way.
        box_w, box_h = (fig.get_size_inches() * [0.90, 0.33]).tolist()
        lim_x = max(lim_x, lim_y * box_w / box_h)
        ax.set_xlim(-lim_x, lim_x)
        ax.set_ylim(-lim_y, lim_y)
        ax.set_aspect("equal", adjustable="box")
        # Row labels: at their row, or, where rows are closer on the page than a label is tall, left of the face, spread
        # about the rows' middle, with a line to each row.
        label_x, row_x = -d["row_length_m"] / 2 - 0.05 * span, -d["row_length_m"] / 2
        need = 2.5 * 8.0 / (box_h * 72.0 / (2.0 * lim_y))                    # two lines of 8 pt, in metres on the face
        rows_low_first = sorted(zip(d["heights_m"], d["per_row"]))
        label_y = spread(d["heights_m"], need)
        moved = any(abs(y - h) > 1e-9 for (h, _), y in zip(rows_low_first, label_y))
        text_w = 0.55 * 8.0 * 7 / (box_h * 72.0 / (2.0 * lim_y))             # "+0.00 m" at 8 pt, in metres on the face
        if moved or label_x - text_w < -span / 2 < label_x:                  # ... or the label would lie across the face's edge
            label_x = min(label_x, -span / 2 - 0.015 * lim_x)
        if moved:
            for (h, _), y in zip(rows_low_first, label_y):
                ax.plot([label_x + 0.004 * lim_x, row_x - 0.004 * lim_x], [y, h], color=MUTED, lw=0.6)
        for (h, n), y in zip(rows_low_first, label_y):
            ax.text(label_x, y, f"{h:+.2f} m\n{n}", ha="right", va="center", fontsize=8, color=MUTED)
        ax.set_xlabel("along the platform (m), forward to the right")
        ax.set_ylabel("up (m)")
        end = (usable - d["row_length_m"]) / 2
        row_word = "row" if d["rows"] == 1 else "rows"
        # The title belongs to the panel, not to the box (which is as narrow as the face is): from the panel's left edge,
        # each sentence wrapped to the panel's width.
        heading = [f"{title or d['name']}: {named(plat.get('mounting'), 'mounting not described')}, {span:g} m x {height:g} m (grey); usable zone {usable:g} m x {usable_h:g} m (dashed)",
                   f"{d['rows']} {row_word} of {d['per_row'][0]} at {d['pitch_m'] * 100:.1f} cm ({d['pitch_wavelengths_top']:.2f} wavelengths at {d['band_hz'][1] / 1e3:g} kHz), "
                   f"{d['row_length_m']:.2f} m long, {end:.2f} m of the usable zone free at each end; colours: capture boards ({d['inputs_per_board']} inputs each)"
                   + (f"; wired {d['interleave']} of each row per board (interleave)" if d["interleave"] else "; wired row by row")]
        fig.text(0.05, 0.995, "\n".join(textwrap.fill(line, TITLE_CHARS) for line in heading), fontsize=9, ha="left", va="top")
        # --- the system ---
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.53])
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 6.4)
        ax.axis("off")
        static = bool(plat.get("static"))
        rows = d["rows"]
        row_h = min(1.0, 3.6 / rows)
        ax.text(0.1, 6.15, "Wet end", color=MUTED, fontsize=10)
        ax.text(4.0, 6.15, "Capture", color=MUTED, fontsize=10)
        ax.text(8.3, 6.15, "Compute", color=MUTED, fontsize=10)
        ax.text(12.6, 6.15, "Platform" if not static else "Surface", color=MUTED, fontsize=10)
        hyd = named((hw.get("hydrophone") or {}).get("model"), "hydrophone not chosen")
        for r in range(rows):
            y = 5.0 - r * (row_h + 0.12)
            boards = [b for b in d["boards"] if r in b["rows"]]
            first, last = min(b["first"] for b in boards), max(b["first"] + b["count"] for b in boards) - 1
            box(ax, (0.1, y), 3.2, row_h * 0.86, f"row at {d['heights_m'][r]:+.2f} m: {d['per_row'][r]} hydrophones\n{hyd}, {d['sensitivity_db']:g} dB re 1 V/uPa", color=GRID, text_color=INK, fs=7.5)
            spare = sum(b["spare_inputs"] for b in boards)
            box(ax, (4.0, y), 3.4, row_h * 0.86, f"boards {boards[0]['index'] + 1}-{boards[-1]['index'] + 1} ({len(boards)} x {d['inputs_per_board']} inputs"
                + (f", {spare} spare" if spare else "") + f")\nchannels {first}-{last}; {mbit(len(boards) * d['rate_board_bit_s'])}", color=YELLOW, text_color=INK, fs=7.5)
            arrow(ax, (3.3, y + row_h * 0.43), (4.0, y + row_h * 0.43))
            ax.text(3.65, y + row_h * 0.43 + 0.08, f"{d['per_row'][r]}", ha="center", fontsize=7, color=MUTED)
            arrow(ax, (7.4, y + row_h * 0.43), (8.3, 4.4), color=AQUA)
        board_model = named((hw.get("boards") or {}).get("model"), "board not chosen")
        ax.text(4.0, 5.0 + row_h * 0.86 + 0.08, f"{board_model}: {d['sample_rate_hz'] / 1e3:g} kHz, {d['adc_bits']} bit, {d['gain_db']:g} dB, full scale {d['adc_full_scale_v']:.2f} V at the ADC", fontsize=7.5, color=MUTED)
        y_low = 5.0 - (rows - 1) * (row_h + 0.12)
        cab = d["cables"]
        ax.text(0.1, y_low - 0.32, f"cables: {d['channels']} runs, {cab['shortest_m']:.1f} to {cab['longest_m']:.1f} m, {cab['total_m']:.0f} m in all\n"
                f"(boards assumed at [{cab['assumed_board_position_m'][0]:g}, {cab['assumed_board_position_m'][1]:g}, {cab['assumed_board_position_m'][2]:g}] m, {cab['slack_m']:g} m slack each)",
                fontsize=7.5, color=MUTED, va="top")
        # sync reference
        if d["sync_scheme"] == "shared_reference":
            ref = d["reference"] or {}
            need = f"\nrows agree to {d['elevation']['clock_us_for_error']:.2f} us for {d['elevation']['clock_error_deg']:g} deg" if d["elevation"] else f"\nboards agree to {d['alignment_tolerance_us']:.1f} us (clean beams)"
            box(ax, (4.0, 0.25), 3.4, 0.95, "sync reference generator: one burst to every board,\nsummed into its first input"
                + (f" ({ref['f_start'] / 1e3:g}-{ref['f_stop'] / 1e3:g} kHz, {ref['duration'] * 1e3:g} ms)" if ref else "") + need, color=VIOLET, fs=7.2)
            arrow(ax, (5.7, 1.2), (5.7, y_low), color=VIOLET)
            ax.text(5.8, (1.2 + y_low) / 2, f"{d['n_boards']} equal lines", fontsize=7, color=VIOLET, va="center")
        else:
            box(ax, (4.0, 0.25), 3.4, 0.8, "one sample clock for every board:\nno synchronisation stage", color=VIOLET, fs=7.5)
        # projector
        pj = hw.get("projector") or {}
        over = d["direct"]["over_full_scale_db"]
        box(ax, (0.1, 0.25), 3.2, 0.95, f"projector ({named(pj.get('model'))})\n{d['source_level_db']:g} dB re 1 uPa at 1 m, {d['band_hz'][0] / 1e3:g}-{d['band_hz'][1] / 1e3:g} kHz, {d['pulse_s']:g} s\n"
            f"amplifier: {named(pj.get('amplifier'))}", color=GRID, text_color=INK, fs=7.5)
        ax.text(0.1, 1.28, f"direct sound at the nearest hydrophone ({d['direct']['nearest_m']:.2f} m): {d['direct']['level_db']:.0f} dB re 1 uPa,\n"
                + (f"{over:.0f} dB OVER the inputs' full scale: protection / recovery open" if over > 0 else f"{-over:.0f} dB below the inputs' full scale"),
                fontsize=7.5, color=ORANGE if over > 0 else MUTED, va="bottom")
        # compute
        comp = hw.get("compute") or {}
        dep = d.get("deploy") or {}
        src, out = dep.get("source") or {}, dep.get("output") or {}
        box(ax, (8.3, 3.9), 3.6, 1.0, f"network: {d['n_boards']} board links -> compute\n{mbit(d['rate_total_bit_s'])} while receiving"
            + (f"; UDP port {src['port']}" if src.get("port") else ""), color=AQUA, fs=7.8)
        box(ax, (8.3, 2.45), 3.6, 1.15, f"compute unit ({named(comp.get('unit'))})\nsonar_live: record, synchronise, beamform,\ndetect, " + ("elevation and depth, " if rows > 1 else "")
            + f"track, classify\noption: {named(comp.get('option'), 'not set').replace('_', ' ')}", color=BLUE, fs=7.8)
        arrow(ax, (10.1, 3.9), (10.1, 3.6))
        box(ax, (8.3, 1.1), 3.6, 1.05, f"storage: every raw ping before processing\n{size(d['record_bytes_per_ping'])} a ping, {size(d['record_bytes_per_hour'])} an hour\n"
            f"(pings {d['ping_interval_s']:.1f} s apart; window {d['window_m'][0]:g}-{d['window_m'][1]:g} m)", color=INK, fs=7.8)
        arrow(ax, (10.1, 2.45), (10.1, 2.15))
        box(ax, (8.3, 0.25), 3.6, 0.6, "power: boards, reference, amplifier, compute\n(no draw is known: to be supplied)", color=GRID, text_color=INK, fs=7.5)
        # platform side
        nav = hw.get("navigation") or {}
        navd = dep.get("nav") or {}
        if "static" in navd:
            nav_text = "pose: typed in (no sensor)\n" + ", ".join(f"{k} {v:g}" for k, v in navd["static"].items() if k in ("z", "roll", "pitch", "yaw"))
        else:
            nav_text = f"navigation and attitude: {named(nav.get('unit'), 'unit not named')}\nUDP port {navd.get('port', '?')}; rate {named(nav.get('rate_hz'), 'to be supplied')}, time stamp {named(nav.get('timestamp_sigma_s'), 'to be supplied')}"
        box(ax, (12.6, 4.6), 3.3, 0.95, nav_text, color=GRID, text_color=INK, fs=7.5)
        arrow(ax, (12.6, 5.0), (11.9, 3.3))
        ss = (hw.get("sound_speed") or {}).get("rung", "gradient_prior")
        wd = (hw.get("water_depth") or {}).get("source", "unknown")
        box(ax, (12.6, 3.45), 3.3, 0.9, f"sound speed: {ss.replace('_', ' ')}\nwater depth: {wd}", color=GRID, text_color=INK, fs=7.5)
        arrow(ax, (12.6, 3.9), (11.9, 3.1))
        rep = out.get("report_udp")
        box(ax, (12.6, 2.2), 3.3, 0.95, ("surface cable: report lines and control" if static else "vehicle computer: report lines")
            + (f"\nUDP {rep[0]}:{rep[1]}" if rep else "") + "\n(one JSON line a ping: detections, tracks)", color=YELLOW, text_color=INK, fs=7.5)
        arrow(ax, (11.9, 2.9), (12.6, 2.7))
        box(ax, (12.6, 1.1), 3.3, 0.8, "ping trigger -> amplifier -> projector; the transmit\ninstant is in the board stream (tx_sample of a block)", color=GRID, text_color=INK, fs=7.5)
        ax.text(12.6, 0.3, "grey: hardware or platform interface, not confirmed\nunless the build's spec marks it so", fontsize=7.5, color=MUTED)
        fig.savefig(path, dpi=110)
        plt.close(fig)


def draw_timing(d: dict, path, title: str | None = None) -> None:
    """One ping on a time axis, with the rates and storage that follow."""
    with plt.rc_context(RC):
        fig, (ax, tx) = plt.subplots(1, 2, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [2.0, 1.1]})
        c, pulse = d["sound_speed"], d["pulse_s"]
        t0, t1 = d["window_s"]
        dr = d["direct"]
        lanes = [("transmit pulse", 0.0, pulse, ORANGE, f"{pulse:g} s, {d['band_hz'][0] / 1e3:g}-{d['band_hz'][1] / 1e3:g} kHz"),
                 ("direct sound at the hydrophones", dr["starts_s"], dr["ends_s"], "#c2417a",
                  f"{dr['level_db']:.0f} dB re 1 uPa at {dr['nearest_m']:.2f} m: " + (f"{dr['over_full_scale_db']:.0f} dB over full scale" if dr["over_full_scale_db"] > 0 else "inside full scale")),
                 ("blind: echoes arrive during the pulse", 0.0, pulse, GRID, f"nothing inside {d['blind_range_m']:g} m"),
                 ("receive window", t0, t1, BLUE, f"{d['window_m'][0]:g} to {d['window_m'][1]:g} m: {d['samples_per_ping']:,} samples a channel")]
        if d.get("reference_times_s"):
            ref = d["reference"]
            lanes.append(("sync bursts (every board's first input)", d["reference_times_s"][0], d["reference_times_s"][0] + ref["duration"], VIOLET,
                          f"{ref['duration'] * 1e3:g} ms at {ref['lead'] * 1e3:g} ms after the window opens and {ref['tail'] * 1e3:g} ms before it closes"))
        span = max(t1, d["ping_interval_s"]) * 1.02
        for i, (label, a, b, colour, note) in enumerate(lanes):
            y = len(lanes) - i
            ax.barh(y, max(b - a, span * 0.004), left=a, height=0.5, color=colour)
            if label.startswith("sync") and d.get("reference_times_s"):
                ax.barh(y, span * 0.004, left=d["reference_times_s"][1], height=0.5, color=colour)
            ax.text(min(b + span * 0.01, span * 0.42), y, f"{label}: {note}", va="center", fontsize=8)
        ax.axvline(d["ping_interval_s"], color=MUTED, lw=0.8, ls="--")
        ax.text(d["ping_interval_s"], len(lanes) + 0.55, f"next ping no sooner than {d['ping_interval_s']:.2f} s", ha="right", fontsize=8, color=MUTED)
        ax.set_xlim(0, span)
        ax.set_ylim(0.3, len(lanes) + 0.9)
        ax.set_yticks([])
        ax.set_xlabel("time after the ping trigger (s)")
        top = ax.secondary_xaxis("top", functions=(lambda t: 0.5 * c * np.maximum(t - pulse, 0.0) / 1e3, lambda r: 2e3 * r / c + pulse))
        top.set_xlabel("range of an echo whose END arrives then (km)")
        ax.set_title(f"{title or d['name']}: one ping", fontsize=10, loc="left", pad=28)
        tx.axis("off")
        el = d["elevation"]
        lines = [("Sampling", f"{d['sample_rate_hz'] / 1e3:g} kHz x {d['adc_bits']} bit x {d['channels']} channels"),
                 ("One channel", mbit(d["rate_channel_bit_s"])), (f"One board ({d['inputs_per_board']} inputs)", mbit(d["rate_board_bit_s"])),
                 (f"All {d['n_boards']} boards", mbit(d["rate_total_bit_s"])),
                 ("Stream, one ping's window", size(d["stream_bytes_per_ping"])), ("Recording, one ping", size(d["record_bytes_per_ping"]) + " (4 bytes a sample)"),
                 ("Recording, one hour", size(d["record_bytes_per_hour"]) + f" at {d['pings_per_hour']:.0f} pings"),
                 ("Stream, one hour of listening", size(d["stream_bytes_per_hour_continuous"])),
                 ("Boards agree for clean beams", f"{d['alignment_tolerance_us']:.1f} us (1/20 period, band top)")]
        if el:
            lines.append((f"Rows agree for {el['clock_error_deg']:g} deg elevation", f"{el['clock_us_for_error']:.2f} us ({el['deg_per_us']:.3f} deg per us)"))
        for i, (k, v) in enumerate(lines):
            tx.text(0.0, 0.95 - i * 0.092, k, fontsize=8.5, color=MUTED, transform=tx.transAxes)
            tx.text(0.62, 0.95 - i * 0.092, v, fontsize=8.5, transform=tx.transAxes)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)


def draw_build(d: dict, out_dir, stem: str, title: str | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"build_{stem}_hardware.png", out_dir / f"build_{stem}_timing.png"]
    draw_hardware(d, paths[0], title)
    draw_timing(d, paths[1], title)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sonar", nargs="?", help="a sonar file's name (default: the six stages of the Popoto plan)")
    ap.add_argument("deploy", nargs="?", help="its deployment file's name")
    ap.add_argument("--out", default=str(ROOT / "docs" / "design"))
    args = ap.parse_args(argv)
    todo = [(args.sonar, args.sonar, args.deploy, args.sonar)] if args.sonar else STAGES
    for label, sonar, deploy, stem in todo:
        for p in draw_build(buildspec.describe_shipped(sonar, deploy), args.out, stem, label):
            print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
