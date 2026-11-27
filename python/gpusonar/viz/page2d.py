"""One static 2-D page per scenario from a scene trace and the run's first record (docs/tasks/VISUALIZATION.md, V1).

Panels: the sound-speed profile; the range-depth section with every drawn eigenray coloured by its bounce class
and each bounce marked with its grazing angle and the loss the model applied there; the bottom's reflection loss
against grazing angle with the angles the eigenrays use marked; the scattering strength of each reverberation item;
the two-way arrivals as stems; the channel-by-time picture of the matched-filtered record with the arrivals marked;
and the list of models in force. Matplotlib only (already a dependency).
"""

from __future__ import annotations

import numpy as np

# Bounce classes: four hues, checked with the dataviz validator (CVD and normal-vision separation pass; the green is
# below 3:1 on white, so every class also has its own marker and a legend entry).
CLASS_STYLE = {
    "direct": dict(color="#2a78d6", marker="o", label="direct"),
    "surface": dict(color="#1baf7a", marker="^", label="surface bounces only"),
    "bottom": dict(color="#eb6834", marker="v", label="bottom bounces only"),
    "both": dict(color="#8e5cc4", marker="D", label="surface and bottom"),
}


def bounce_class(n_surface: int, n_bottom: int) -> str:
    if n_surface == 0 and n_bottom == 0:
        return "direct"
    if n_bottom == 0:
        return "surface"
    if n_surface == 0:
        return "bottom"
    return "both"


def compressed(record):
    """(times s after transmit [N], |matched filter output| [C, N]) of a copy of `record` (baseband first when it
    is real): what the processing chain's first stage makes of it (core.to_baseband, core.pulse_compress)."""
    from .. import core
    r = record.copy()
    if not r.samples.is_complex():
        core.to_baseband(r)
    core.pulse_compress(r)
    s = r.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    return t, np.abs(np.asarray(s.array))


def _with_samples(record, samples):
    """A copy of `record` holding `samples` (same shape and kind), for drawing a simulator stage like a record."""
    r = record.copy()
    smp = r.samples
    smp.array = np.asarray(samples, dtype=np.asarray(smp.array).dtype)
    r.samples = smp
    return r


def _plan_view(ax, trace, n_beams: int = 9, span_deg: float = 40.0):
    """Plan view: hydrophones, projector, targets and their bearings, and a fan of receive beams steered from
    broadside of a line array (half-power width 0.886 lambda / (N d cos theta), the uniform line array's; the
    live chain's own steering and window are not read here)."""
    hyd = np.asarray(trace["sonar"]["hydrophones"], float)
    tx = np.asarray(trace["sonar"]["projector"], float)
    c = trace["environment"]["sound_speed"]
    lam = c / trace["sonar"]["center_frequency"]
    ref = hyd.mean(axis=0)
    rng = max([np.linalg.norm(np.asarray(t["position"][:2]) - ref[:2]) for t in trace["targets"]] + [1.0])
    ax.plot(hyd[:, 1], hyd[:, 0], "s", color="0.1", ms=3)
    ax.plot(tx[1], tx[0], "*", color="0.1", ms=9)
    axis = hyd[-1, :2] - hyd[0, :2]
    n = len(hyd)
    if n > 1 and np.linalg.norm(axis) > 0:
        axis = axis / np.linalg.norm(axis)
        normal = np.array([-axis[1], axis[0]])
        if trace["targets"] and np.dot(np.asarray(trace["targets"][0]["position"][:2]) - ref[:2], normal) < 0:
            normal = -normal
        d = np.linalg.norm(hyd[-1, :2] - hyd[0, :2]) / (n - 1)
        for th in np.radians(np.linspace(-span_deg, span_deg, n_beams)):
            half = 0.5 * min(0.886 * lam / (n * d * max(np.cos(th), 1e-3)), np.pi / 2)
            for edge, ls in ((th - half, ":"), (th + half, ":"), (th, "-")):
                v = np.cos(edge) * normal + np.sin(edge) * axis
                ax.plot([ref[1], ref[1] + rng * v[1]], [ref[0], ref[0] + rng * v[0]], ls, color="#2a78d6",
                        lw=0.8 if ls == "-" else 0.5, alpha=0.8)
    for t in trace["targets"]:
        p = np.asarray(t["position"])
        ax.plot([ref[1], p[1]], [ref[0], p[0]], "--", color="#c0392b", lw=0.8)
        ax.plot(p[1], p[0], "X", color="#c0392b", ms=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("east y (m)")
    ax.set_ylabel("north x (m)")
    ax.set_title(f"plan: {n_beams} receive beams (-3 dB, dotted) over +-{span_deg:g} deg", fontsize=9)


def _section_axes(trace):
    """(origin xy, unit horizontal direction) of the vertical plane through the projector and the first target."""
    tx = np.asarray(trace["sonar"]["projector"], float)
    if trace["targets"]:
        d = np.asarray(trace["targets"][0]["position"][:2]) - tx[:2]
        if np.linalg.norm(d) > 0:
            return tx[:2], d / np.linalg.norm(d)
    return tx[:2], np.array([0.0, 1.0])


def draw_page(trace: dict, record, path: str, stages: dict | None = None, n_marked: int = 12, max_rays: int = 8,
              dpi: int = 80) -> dict:
    """Write the page to `path` (PNG or SVG by its extension). `stages`: the simulator's own taps of this ping
    ({stage: samples}, taps.read_sim_taps), for the echoes-alone picture and the level of each part against time.
    The `max_rays` strongest eigenrays of each leg are drawn in colour with their bounces, the rest thin and grey.
    Returns what was drawn (counts), for the tests."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = trace["environment"]
    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(4, 4, width_ratios=[0.7, 3, 3, 3], height_ratios=[1.25, 1, 1.1, 1], hspace=0.42, wspace=0.32)
    ax_ssp = fig.add_subplot(gs[0, 0])
    ax_sec = fig.add_subplot(gs[0, 1:], sharey=ax_ssp)
    ax_rl = fig.add_subplot(gs[1, 0:2])
    ax_ss = fig.add_subplot(gs[1, 2])
    ax_plan = fig.add_subplot(gs[1, 3])
    ax_stem = fig.add_subplot(gs[2, 0:2])
    ax_wf = fig.add_subplot(gs[2, 2])
    ax_wf2 = fig.add_subplot(gs[2, 3], sharey=ax_wf)
    ax_lvl = fig.add_subplot(gs[3, 0:2])
    ax_txt = fig.add_subplot(gs[3, 2:])
    drawn = {"rays": 0, "bounces": 0, "pairs": len(trace["pairs"]), "marked": 0}

    # Sound-speed profile.
    bottom_z = env.get("bottom_z") or env.get("bottom_z_site")
    if env.get("ssp"):
        ax_ssp.plot(env["ssp"]["c"], env["ssp"]["z"], "-", color="0.2", lw=1.5)
    else:
        ax_ssp.plot([env["sound_speed"]] * 2, [0.0, bottom_z or 1.0], "-", color="0.2", lw=1.5)
    ax_ssp.set_xlabel("sound speed (m/s)")
    ax_ssp.set_ylabel("depth (m)")
    ax_ssp.set_title("profile", fontsize=10)
    ax_ssp.tick_params(axis="x", labelrotation=45)

    # Range-depth section.
    # Horizontal range from the sonar, whatever the bearing (each 2-D eigenray lies in the vertical plane of its
    # own bearing; targets on different bearings share this one range axis).
    o = np.asarray(trace["sonar"]["projector"], float)[:2]

    def to_rz(xyz):
        a = np.asarray(xyz, float).reshape(-1, 3)
        return np.linalg.norm(a[:, :2] - o, axis=1), a[:, 2]

    seen = set()
    for leg in trace["legs"]:
        for which, ls in (("out", "-"), ("back", "--")):
            keep = {r_["k"] for r_ in sorted(leg[which], key=lambda r_: -r_["amplitude"])[:max_rays]}
            for ray in leg[which]:
                if ray["polyline"] is None:
                    continue
                if ray["k"] not in keep:
                    r, z = to_rz(ray["polyline"])
                    ax_sec.plot(r, z, "-", color="0.82", lw=0.5, zorder=0)
                    drawn["faint"] = drawn.get("faint", 0) + 1
                    continue
                cls = bounce_class(ray["n_surface"], ray["n_bottom"])
                st = CLASS_STYLE[cls]
                r, z = to_rz(ray["polyline"])
                ax_sec.plot(r, z, ls, color=st["color"], lw=1.0, alpha=0.8,
                            label=st["label"] if cls not in seen else None)
                seen.add(cls)
                drawn["rays"] += 1
                for b in ray["bounces"]:
                    br, bz = to_rz(b["xyz"])
                    ax_sec.plot(br, bz, st["marker"], color=st["color"], ms=5, mec="white", mew=0.6)
                    drawn["bounces"] += 1
    hyd = np.asarray(trace["sonar"]["hydrophones"])
    hr, hz = to_rz(hyd)
    ax_sec.plot(hr, hz, "s", color="0.1", ms=6, label="hydrophones")
    pr, pz = to_rz(trace["sonar"]["projector"])
    ax_sec.plot(pr, pz, "*", color="0.1", ms=12, label="projector")
    for t in trace["targets"]:
        tr_, tz = to_rz(t["position"])
        ax_sec.plot(tr_, tz, "X", color="#c0392b", ms=11)
        ax_sec.annotate(t["label"], (tr_[0], tz[0]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    rmax = max([float(np.max(to_rz(t["position"])[0])) for t in trace["targets"]] + [1.0]) * 1.05
    ax_sec.axhline(env.get("surface_z") or 0.0, color="#5dade2", lw=2)
    if bottom_z is not None:
        ax_sec.axhspan(bottom_z, bottom_z * 1.08 + 1.0, color="#c8a165", alpha=0.6, lw=0)
        b = env.get("bottom") or {}
        btxt = ", ".join(f"{k} {v:g}" for k, v in b.items() if isinstance(v, (int, float)) and v) if isinstance(b, dict) else str(b)
        ax_sec.text(0.01 * rmax, bottom_z * 1.04 + 0.5, f"bottom: {btxt or 'see models'}"
                    + ("" if env.get("bottom_z") is not None else " (site depth; this tier has no bottom)"),
                    fontsize=8, va="center")
        ax_sec.set_ylim(bottom_z * 1.08 + 1.0, -0.04 * bottom_z - 1.0)
    else:
        ax_sec.invert_yaxis()
    wind = env.get("wind_speed")
    surface_txt = ("free field: no surface or bottom in the propagation" if trace["tier"] == "free_field"
                   else "surface: flat, pressure release")
    ax_sec.text(0.01 * rmax, -0.02 * (bottom_z or 1.0), surface_txt
                + (f"; wind {wind:g} m/s (in the reverberation and noise models)" if wind else ""), fontsize=8, va="bottom")
    ax_sec.set_xlim(min(0.0, float(np.min(hr)) - 0.02 * rmax), rmax)
    ax_sec.set_xlabel("horizontal range from the sonar, on each ray's own bearing (m)")
    ax_sec.set_title(f"{trace['scenario']}: tier {trace['tier']}; eigenrays of each target, the {max_rays} strongest a leg in "
                     "colour (solid: projector to target, dashed: target to array), the rest grey", fontsize=10)
    for h in ax_sec.legend(loc="lower right", fontsize=8, framealpha=0.9).legend_handles:
        h.set_linewidth(2.5)
    plt.setp(ax_sec.get_yticklabels(), visible=False)

    # Reflection loss against grazing angle, with the angles used marked.
    from ..sim import propagation as prop
    g = np.radians(np.linspace(0.1, 90.0, 300))
    b = env.get("bottom") or {}
    c_w = env["sound_speed"]
    used = [(bb["grazing_deg"], bb["loss_db"]) for leg in trace["legs"] for w in ("out", "back") for ray in leg[w]
            for bb in ray["bounces"] if bb["boundary"] == "bottom" and bb["loss_db"] is not None]
    if isinstance(b, dict) and b.get("speed") and b.get("density"):
        rl = -20.0 * np.log10(np.abs(prop.rayleigh_reflection(g, c_w, float(b["speed"]), float(b["density"]),
                                                             float(b.get("attenuation") or 0.0),
                                                             shear_speed=float(b.get("shear_speed") or 0.0))))
        ax_rl.plot(np.degrees(g), rl, "-", color="#eb6834", lw=1.5, label="bottom, Rayleigh (this scenario's half-space)")
    elif isinstance(b, dict) and b.get("coefficient") is not None:
        ax_rl.plot([0, 90], [-20.0 * np.log10(max(float(b["coefficient"]), 1e-30))] * 2, "-", color="#eb6834", lw=1.5,
                   label=f"bottom, constant |R| = {float(b['coefficient']):g} (this scenario)")
    if used:
        ga, la = np.array(used).T
        ax_rl.plot(ga, la, "v", color="#eb6834", ms=7, mec="0.2", mew=0.6, ls="none", label="bounces of the drawn eigenrays")
    if trace["tier"] != "free_field":
        ax_rl.plot([0, 90], [0, 0], "-", color="#1baf7a", lw=1.5, label="surface, -1 (flat, pressure release)")
    else:
        ax_rl.text(0.5, 0.5, "free field: no boundary reflects", ha="center", transform=ax_rl.transAxes)
    ax_rl.set_xlabel("grazing angle (deg)")
    ax_rl.set_ylabel("loss per bounce (dB)")
    ax_rl.set_title("reflection loss the propagation model applies", fontsize=10)
    ax_rl.legend(fontsize=8)
    ax_rl.set_xlim(0, 90)

    # Scattering strength of each reverberation item (monostatic: incident = scattered angle).
    items = [m for m in trace["models"] if m["key"].startswith("reverberation[")]
    if items:
        ax_ss.text(0.02, 0.98, "\n".join(f"{m['key']}: {m['model']} {m['value']}" for m in items), transform=ax_ss.transAxes,
                   fontsize=8, va="top")
        styles = ["-", "--", ":", "-."]
        for i, sc in enumerate(trace.get("scattering") or []):
            ax_ss.plot(sc["grazing_deg"], sc["strength_db"], styles[i % 4], color="0.2", lw=1.5,
                       label=f"{sc['key']}: {sc['model']}")
        if trace.get("scattering"):
            ax_ss.legend(fontsize=8, loc="lower right")
        ax_ss.set_xlim(0, 90)
        ax_ss.set_xlabel("grazing angle (deg)")
        ax_ss.set_ylabel("backscattering strength (dB)")
        ax_ss.set_ylim(-80.0, None)
    else:
        ax_ss.text(0.5, 0.5, "no reverberation in this scenario", ha="center", va="center", transform=ax_ss.transAxes)
        ax_ss.set_axis_off()
    ax_ss.set_title("scattering strength in force", fontsize=10)

    # Models in force.
    ax_txt.set_axis_off()
    lines = [f"{m['area']}: {m['model']} [{m['status']}]" + (" (default)" if m["chosen"] == "default" else "")
             for m in trace["models"]]
    ax_txt.text(0.0, 1.0, "what is modelled here\n" + "\n".join(lines), fontsize=8, va="top", family="monospace",
                transform=ax_txt.transAxes)

    # Arrival stems.
    pairs = trace["pairs"]
    if pairs:
        ref = max(p["level_db"] for p in pairs)
        seen = set()
        for p in pairs:
            cls = bounce_class(p["n_surface"], p["n_bottom"])
            st = CLASS_STYLE[cls]
            lv = p["level_db"] - ref
            if lv < -40.0:
                continue
            ax_stem.plot([p["delay_s"]] * 2, [-42.0, lv], "-", color=st["color"], lw=1.0)
            ax_stem.plot(p["delay_s"], lv, st["marker"], color=st["color"], ms=6, mec="white", mew=0.5,
                         label=st["label"] if cls not in seen else None)
            seen.add(cls)
        ax_stem.set_ylim(-42.0, 3.0)
        ax_stem.legend(fontsize=8, loc="upper right")
    ax_stem.set_xlabel("two-way delay (s)")
    ax_stem.set_ylabel("pair level re strongest (dB)")
    ax_stem.set_title("two-way arrivals within 40 dB of the strongest\n(spreading and reflection; no absorption, no beams)",
                      fontsize=9)

    # Channel by time, matched-filtered, with the strongest arrivals marked at each hydrophone: the record as the
    # chain receives it, and the echoes alone (the simulator's own tap after its echo stage) when given.
    strongest = sorted(pairs, key=lambda p: -p["level_db"])[:n_marked]
    window = None
    if strongest:
        lo = min(p["delay_s"] for p in strongest)
        hi = max(p["delay_s"] for p in strongest)
        pad = max(0.15 * (hi - lo), 20.0 / trace["sonar"]["sample_rate"], 0.002)
        window = (lo - pad, hi + pad)
    parts = {"record": record}
    if stages and "echoes" in stages:
        parts["echoes"] = _with_samples(record, stages["echoes"])
    for ax, (name, rec) in zip((ax_wf, ax_wf2), parts.items()):
        t, env_ = compressed(rec)
        db = 20.0 * np.log10(np.maximum(env_ / max(env_.max(), 1e-30), 1e-6))
        nc = db.shape[0]
        ax.imshow(db, aspect="auto", origin="lower", cmap="gray", vmin=-40, vmax=0,
                  extent=[t[0], t[-1], -0.5, nc - 0.5], interpolation="nearest")
        for p in strongest:
            st = CLASS_STYLE[bounce_class(p["n_surface"], p["n_bottom"])]
            ax.plot(p["channel_delay_s"], np.arange(nc), "|", color=st["color"], ms=4, mew=1.0)
        if window:
            ax.set_xlim(max(t[0], window[0]), min(t[-1], window[1]))
        ax.set_xlabel("time after transmit (s)")
        ax.set_title(("the record" if name == "record" else "echoes alone (simulator tap)")
                     + f",\nmatched-filtered, dB re its peak; ticks: {len(strongest)} strongest arrivals", fontsize=9)
    ax_wf.set_ylabel("hydrophone")
    drawn["marked"] = len(strongest)
    if len(parts) == 1:
        ax_wf2.set_axis_off()

    # Level of each simulated part against time (mean over the channels, smoothed), from the simulator's taps.
    if stages:
        order = [k for k in ("echoes", "reverberation", "noise") if k in stages]
        prev = None
        colours = {"echoes": "#2a78d6", "reverberation": "#eb6834", "noise": "0.45"}
        names = {"echoes": "echoes", "reverberation": "reverberation (all items)", "noise": "ambient noise"}
        fs = trace["sonar"]["sample_rate"]
        t0 = trace["sonar"]["start_time_s"]
        for k in order:
            x = np.asarray(stages[k])
            part = x if prev is None else x - prev
            prev = x
            pw = np.mean(np.abs(part) ** 2, axis=0)
            n = max(int(fs * 0.005), 1)
            pw = np.convolve(pw, np.ones(n) / n, mode="same")
            if not np.any(pw > 0.0):
                continue                                   # a part this scenario does not have (no reverberation, no noise)
            tt = t0 + np.arange(len(pw)) / fs
            ax_lvl.plot(tt, 10.0 * np.log10(np.maximum(pw, 1e-30)), "-", color=colours[k], lw=1.2, label=names[k])
        ax_lvl.legend(fontsize=8)
        ax_lvl.set_xlabel("time after transmit (s)")
        ax_lvl.set_ylabel("power, channel mean (dB)")
        top = ax_lvl.get_ylim()[1]
        ax_lvl.set_ylim(top - 80.0, top + 3.0)
    else:
        ax_lvl.text(0.5, 0.5, "no simulator taps given", ha="center", transform=ax_lvl.transAxes)
    ax_lvl.set_title("parts of the received signal, before matched filtering\n(reverberation items not split)", fontsize=9)

    _plan_view(ax_plan, trace)

    if str(path).lower().endswith(".png"):
        # A 256-colour palette keeps a page near 150-300 kB (the repository is small); the picture is unchanged
        # but for the shading of the grey maps.
        import io

        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, bbox_inches="tight", format="png")
        Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return drawn
