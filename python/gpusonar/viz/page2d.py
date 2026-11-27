"""One static 2-D page per scenario from a scene trace and the run's record (docs/tasks/VISUALIZATION.md, V1 and V2).

Panels: the sound-speed profile (one line per range where the water changes with range); the range-depth section
with the drawn eigenrays coloured by bounce class (the strongest few of each leg, the rest thin and light), the bed
as the tier has it (flat, tilted, by range, a sediment interface with the refracted leg and the evanescent region),
the surface (with its rms-height band where one is set), one-way sources and the projector's direct sound; the loss
per bounce against grazing angle of every boundary model in force, with the angles the drawn rays used; the
scattering strength of each reverberation item; the plan view (rays, the processing's bearings with steered -3 dB
beams, the track of a moving platform, walls); the receive and projector patterns at the band centre; arrival
stems; channel-by-time pictures of the record and of the echoes alone; the parts of the signal against time; and
the list of models in force with key, value, source and status. Matplotlib only (already a dependency).
"""

from __future__ import annotations

import textwrap

import numpy as np

# Bounce classes: four hues, checked with the dataviz validator (CVD and normal-vision separation pass; the green is
# below 3:1 on white, so every class also has its own marker and a legend entry).
CLASS_STYLE = {
    "direct": dict(color="#2a78d6", marker="o", label="direct"),
    "surface": dict(color="#1baf7a", marker="^", label="surface bounces only"),
    "bottom": dict(color="#eb6834", marker="v", label="bottom bounces only"),
    "both": dict(color="#8e5cc4", marker="D", label="surface and bottom"),
}
FAINT = "#c9c9c9"
ONEWAY = "#b8860b"            # one-way sources and the projector's direct sound (a fifth hue, with its own marker)
TARGET = "#c0392b"


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


def _short(v, n: int = 46) -> str:
    s = str(v).replace("'", "")
    return s if len(s) <= n else s[:n - 1] + "~"


# --- the range-depth section ----------------------------------------------------------------------------------------

def _all_rays(trace):
    """[(group, which, ray)] of every ray of the trace: two-way legs, one-way sources, the direct sound."""
    out = []
    for leg in trace["legs"]:
        for which in ("out", "back"):
            out += [("legs", which, r) for r in leg[which]]
    for src in trace.get("oneway") or []:
        out += [("oneway", "src", r) for r in src["rays"]]
    if trace.get("direct"):
        out += [("direct", "direct", r) for r in trace["direct"]["rays"]]
    return out


def _monostatic(trace) -> float | None:
    """The projector's distance from the array's centre when it is under 1 % of the nearest target's range (the out
    and back legs then lie on top of each other), else None."""
    tx = np.asarray(trace["sonar"]["projector"], float)
    ref = np.asarray(trace["sonar"]["hydrophones"], float).mean(axis=0)
    if not trace["targets"]:
        return None
    rng = min(np.linalg.norm(np.asarray(t["position"]) - ref) for t in trace["targets"])
    sep = float(np.linalg.norm(tx - ref))
    return sep if sep < 0.01 * max(rng, 1e-9) else None


def _draw_section(ax, trace, max_rays: int, all_rays: bool, drawn: dict):
    env = trace["environment"]
    o = np.asarray(trace["sonar"]["projector"], float)[:2]

    def to_rz(xyz):
        a = np.asarray(xyz, float).reshape(-1, 3)
        return np.linalg.norm(a[:, :2] - o, axis=1), a[:, 2]

    pts = [trace["sonar"]["projector"]] + trace["sonar"]["hydrophones"] + [t["position"] for t in trace["targets"]]
    pts += [s["position"] for s in trace.get("oneway") or []]
    for _, _, r in _all_rays(trace):
        if r["polyline"] is not None:
            pts += r["polyline"]
    P = np.asarray(pts, float)
    rr, zz = to_rz(P)
    rmax = max(float(rr.max()) * 1.04, 1e-3)
    zmin = min(float(zz.min()), env.get("surface_z") or 0.0)
    bottom_z = env.get("bottom_z")
    if bottom_z is None:
        bottom_z = env.get("bottom_z_site")
    bed = env.get("bathymetry")
    zmax = max(float(zz.max()), bottom_z if bottom_z is not None else float(zz.max()))
    if bed:
        zmax = max(zmax, float(np.max(np.asarray(bed, float)[:, 1])))
    span = max(zmax - zmin, 1e-3)
    mono = _monostatic(trace)
    offset = 0.006 * span if mono is not None else 0.0

    seen = set()
    for leg in trace["legs"]:
        for which in ("out", "back"):
            rays = [r for r in leg[which] if r["polyline"] is not None]
            keep = {r["k"] for r in sorted(rays, key=lambda r: -r["amplitude"])[:max_rays]} if not all_rays else {r["k"] for r in rays}
            for ray in rays:
                r, z = to_rz(ray["polyline"])
                if which == "back":
                    z = z + offset
                if ray["k"] not in keep:
                    ax.plot(r, z, "-", color=FAINT, lw=0.4, zorder=1)
                    drawn["faint"] = drawn.get("faint", 0) + 1
                    continue
                cls = bounce_class(ray["n_surface"], ray["n_bottom"])
                st = CLASS_STYLE[cls]
                label = st["label"] if not (cls == "direct" and trace["tier"] == "sediment") else "no boundary bounce (refracted into the sediment where buried)"
                ax.plot(r, z, ls="-" if which == "out" else (0, (4, 2)), color=st["color"], lw=1.1 if which == "out" else 0.9,
                        alpha=0.9, zorder=3, label=label if cls not in seen else None)
                seen.add(cls)
                drawn["rays"] += 1
                for b in ray["bounces"]:
                    br, bz = to_rz(b["xyz"])
                    mk = {"interface": "o", "wall": "s"}.get(b["boundary"], st["marker"])
                    ax.plot(br, bz + (offset if which == "back" else 0.0), mk, color=st["color"], ms=5, mec="white",
                            mew=0.6, zorder=4)
                    drawn["bounces"] += 1
            # Grazing angle and loss at the bounces of the strongest ray of each leg (one leg when monostatic).
            if rays and not (which == "back" and mono is not None):
                best = max(rays, key=lambda r: r["amplitude"] * (1.0 if r["bounces"] else 1e-9))
                for b in best["bounces"][:6]:
                    br, bz = to_rz(b["xyz"])
                    if b["loss_db"] is not None:
                        ax.annotate(f"{b['grazing_deg']:.0f}°, {b['loss_db']:.1f} dB", (br[0], bz[0]), xytext=(3, -9 if b["boundary"] == "surface" else 4),
                                    textcoords="offset points", fontsize=6.5, color="0.25", zorder=5)
    # One-way sources and the projector's direct sound.
    for group, which, ray in _all_rays(trace):
        if group == "legs" or ray["polyline"] is None:
            continue
        r, z = to_rz(ray["polyline"])
        ax.plot(r, z, ":", color=ONEWAY, lw=1.1, zorder=2,
                label=("one-way: source to array" if group == "oneway" else "direct sound: projector to array")
                if group not in seen else None)
        seen.add(group)
        drawn["oneway_rays"] = drawn.get("oneway_rays", 0) + 1
        for b in ray["bounces"]:
            br, bz = to_rz(b["xyz"])
            ax.plot(br, bz, "o", color=ONEWAY, ms=3.5, mec="white", mew=0.5, zorder=4)
    for src in trace.get("oneway") or []:
        sr, sz = to_rz(src["position"])
        ax.plot(sr, sz, "P", color=ONEWAY, ms=10, mec="0.1", mew=0.6, zorder=6)
        ax.annotate(f"source {src['label']}", (sr[0], sz[0]), xytext=(6, 6), textcoords="offset points", fontsize=8)

    # Boundaries.
    surface_z = env.get("surface_z") or 0.0
    sfc = env.get("surface")
    tier = trace["tier"]
    has_surface = tier not in ("free_field", "sediment") and env.get("surface_z", 0.0) is not None
    if has_surface:                                          # free field: no surface (z = 0 may be the seabed's frame)
        ax.axhline(surface_z, color="#5dade2", lw=2)
    if sfc and sfc.get("rms_height"):
        h = max(sfc["rms_height"], 0.004 * span)
        ax.axhspan(surface_z - h, surface_z + h, color="#5dade2", alpha=0.25, lw=0)
    planes = env.get("planes")
    if bed:
        b = np.asarray(bed, float)
        ax.fill_between(b[:, 0], b[:, 1], zmax + 0.06 * span, color="#c8a165", alpha=0.6, lw=0)
        ax.plot(b[:, 0], b[:, 1], "-", color="#8a6d3b", lw=1.2)
    elif planes:
        u = None
        if trace["targets"]:
            d = np.asarray(trace["targets"][0]["position"][:2]) - o
            u = d / max(np.linalg.norm(d), 1e-12)
        rs = np.linspace(0.0, rmax, 50)
        for pl in planes:
            n, p0 = np.asarray(pl["normal"]), np.asarray(pl["point"])
            if pl["kind"] == "bottom" and u is not None and abs(n[2]) > 1e-9:
                xy = o[None, :] + rs[:, None] * u[None, :]
                zb = p0[2] - ((xy[:, 0] - p0[0]) * n[0] + (xy[:, 1] - p0[1]) * n[1]) / n[2]
                ax.fill_between(rs, zb, zmax + 0.06 * span, color="#c8a165", alpha=0.6, lw=0)
                ax.plot(rs, zb, "-", color="#8a6d3b", lw=1.2)
            elif pl["kind"] == "wall" and u is not None:
                # Where the section's vertical plane meets the wall (at the projector's depth), if it does.
                den = n[0] * u[0] + n[1] * u[1]
                if abs(den) > 1e-9:
                    rw = ((p0[0] - o[0]) * n[0] + (p0[1] - o[1]) * n[1]) / den
                    if 0.0 < rw < rmax:
                        ax.axvline(rw, color="0.35", lw=2.5, alpha=0.7)
                        ax.text(rw, zmin + 0.05 * span, " wall", fontsize=7.5, va="top")
    elif bottom_z is not None:
        ax.axhspan(bottom_z, zmax + 0.06 * span, color="#c8a165", alpha=0.6 if env.get("bottom_z") is not None else 0.25, lw=0)
    itf = env.get("interface")
    if itf and itf.get("critical_deg") and trace["targets"]:
        # The evanescent region: points under the interface whose straight line from the projector meets it below
        # the critical angle (lossless, (8.56)).
        zs = float(trace["sonar"]["projector"][2])
        tanc = np.tan(np.radians(itf["critical_deg"]))
        zi = float(itf["z"])
        z2 = zmax + 0.06 * span
        r0 = (zi - zs) / tanc
        if r0 < rmax:
            poly = np.array([[r0, zi], [rmax * 2, zi], [rmax * 2, z2], [(z2 - zs) / tanc, z2]])
            ax.fill(poly[:, 0], poly[:, 1], facecolor="none", edgecolor="0.3", hatch="///", lw=0, zorder=2)
            ax.text(r0, zi + 0.03 * span, f"  below the critical angle {itf['critical_deg']:.1f}° (hatched): "
                    f"evanescent field ({itf['subcritical'] or 'no field model'})", fontsize=7, va="top", color="0.25")
        else:
            ax.text(0.99, 0.12, f"critical angle {itf['critical_deg']:.1f}°: every point drawn is reached above it "
                    f"(the evanescent region starts {r0:.3g} m out)", transform=ax.transAxes, ha="right", fontsize=7, color="0.25")

    # Sonar and targets.
    hyd = np.asarray(trace["sonar"]["hydrophones"])
    hr, hz = to_rz(hyd)
    ax.plot(hr, hz, "s", color="0.1", ms=4, label="hydrophones", zorder=6)
    pr, pz = to_rz(trace["sonar"]["projector"])
    ax.plot(pr, pz, "*", color="0.1", ms=12, label="projector", zorder=6)
    for t in trace["targets"]:
        tr_, tz = to_rz(t["position"])
        ax.plot(tr_, tz, "X", color=TARGET, ms=10, zorder=6)
        ax.annotate(t["label"], (tr_[0], tz[0]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_ylim(zmax + 0.06 * span, zmin - 0.08 * span)
    ax.set_xlim(min(0.0, float(np.min(hr))) - 0.02 * rmax, rmax)

    # Words on the boundaries.
    if tier == "free_field":
        stxt = "free field: no surface or bottom in the propagation"
    elif tier == "sediment":
        stxt = "sediment tier: no surface in the propagation; one refracted ray into the sediment"
    elif not has_surface:
        stxt = "no surface in the propagation (the scenario's planes are walls only)"
    else:
        stxt = "surface: flat, pressure release (-1)"
        if sfc:
            stxt += f"; rms height {sfc['rms_height']:.3f} m" + (f" (wind {sfc['wind_speed']:g} m/s)" if sfc.get("wind_speed") else "")
            stxt += " in time spreading, band not to scale"
            if sfc.get("bubble_loss"):
                stxt += f"; bubble loss {np.max(sfc['bubble_loss']['loss_db']):.2g} dB a bounce"
        if env.get("wind_speed_propagation"):
            stxt += f"; coherent loss at wind {env['wind_speed_propagation']:g} m/s"
    wind = env.get("wind_speed")
    if wind and not env.get("wind_speed_propagation"):
        stxt += f"; site wind {wind:g} m/s (reverberation, noise)"
    ax.text(0.005, 0.995, stxt, transform=ax.transAxes, fontsize=7.5, va="top")
    b = env.get("bottom") or {}
    if tier == "sediment" and itf:
        bt = (f"interface at z = {itf['z']:g} m; sediment {itf['sediment_speed']:.0f} m/s; penetration {itf['penetration']}"
              + (f", density {itf['bottom']['density']:.3g}, attenuation {itf['bottom']['attenuation']:.3g} dB/λ" if itf.get("bottom") else "")
              + (f"; transmission {itf['transmission']:g}" if itf.get("transmission") is not None else ""))
    elif isinstance(b, dict) and b:
        bt = "bottom: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, (int, float)) else f"{k} {v}"
                                     for k, v in b.items() if v not in (None, 0, 0.0) and k != "water_density")
        if env.get("bottom_z") is None:
            bt += " (the site's depth; this tier has no bottom)"
        if env.get("bathymetry_note"):
            bt += "; " + env["bathymetry_note"]
        if env.get("bottom_slope"):
            bt += f"; tilted, slope {env['bottom_slope']}"
    elif env.get("bottom_z_site") is not None:
        bt = f"site depth {env['bottom_z_site']:g} m (the tier has no bottom)"
    else:
        bt = ""
    if bt:
        ax.text(0.005, 0.01, bt, transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1.5))
    leg = ax.legend(loc="best", fontsize=7.5, framealpha=0.9)
    for h in leg.legend_handles:
        h.set_linewidth(2.5)
    ax.set_xlabel("horizontal range from the projector, on each ray's own bearing (m)")
    what = (f"the {max_rays} strongest eigenrays of each leg in colour, the rest thin grey" if not all_rays
            else "every eigenray in colour")
    where = "at the array centre" if mono is not None and mono < 1e-3 else f"{mono:.2g} m from the array centre" if mono is not None else ""
    mono_txt = (f"; monostatic (projector {where}): the dashed back legs are drawn "
                f"{offset:.2g} m deeper so as not to hide under the out legs" if mono is not None else "; dashed: target to array")
    ax.set_title(f"{trace['scenario']}: tier {tier}; {what}\nsolid: projector to target{mono_txt}", fontsize=9)
    return to_rz


# --- loss per bounce against grazing ----------------------------------------------------------------------------------

def _draw_losses(ax, trace):
    from .. import acoustics
    from ..sim import propagation as prop
    env = trace["environment"]
    g = np.radians(np.linspace(0.1, 90.0, 300))
    gd = np.degrees(g)
    c_w = env["sound_speed"]
    fc = trace["sonar"]["center_frequency"]
    b = env.get("bottom") or {}
    itf = env.get("interface")
    used = {}
    for _, _, ray in _all_rays(trace):
        for bb in ray["bounces"]:
            if bb["loss_db"] is not None:
                used.setdefault(bb["boundary"], []).append((bb["grazing_deg"], bb["loss_db"]))
    if trace["tier"] == "sediment" and itf and itf.get("bottom"):
        bt = dict(itf["bottom"])
        v = np.abs(prop.fluid_transmission(g, c_w, **bt)) / np.sqrt(bt["density"] / bt.get("water_density", 1.0))
        ax.plot(gd, -20.0 * np.log10(np.maximum(v, 1e-30)), "-", color="#eb6834", lw=1.5,
                label="into the sediment, per leg: -20 log |V_wp| / sqrt(a_rho) (J&R 8.47)")
    elif trace["tier"] == "sediment" and itf:
        ax.plot([0, 90], [-20.0 * np.log10(max(itf.get("transmission") or 1.0, 1e-30))] * 2, "-", color="#eb6834", lw=1.5,
                label="into the sediment: the constant transmission")
    elif isinstance(b, dict) and b.get("speed") and b.get("density"):
        rl = -20.0 * np.log10(np.abs(prop.rayleigh_reflection(g, c_w, float(b["speed"]), float(b["density"]),
                                                             float(b.get("attenuation") or 0.0),
                                                             shear_speed=float(b.get("shear_speed") or 0.0))))
        el = float(b.get("shear_speed") or 0.0) > 0
        ax.plot(gd, rl, "-", color="#eb6834", lw=1.5,
                label=f"bottom, {'elastic' if el else 'fluid'} Rayleigh half-space (this scenario's)")
    elif isinstance(b, dict) and b.get("coefficient") is not None:
        ax.plot([0, 90], [-20.0 * np.log10(max(float(b["coefficient"]), 1e-30))] * 2, "-", color="#eb6834", lw=1.5,
                label=f"bottom, constant |R| = {float(b['coefficient']):g} (this scenario's)")
    crit = env.get("critical_angle_deg") or (itf or {}).get("critical_deg")
    if crit:
        ax.axvline(crit, color="#eb6834", lw=0.8, ls="--")
        ax.text(crit, 0.97, f" critical {crit:.1f}°", transform=ax.get_xaxis_transform(), fontsize=7.5, va="top")
    if trace["tier"] not in ("free_field", "sediment") and env.get("surface_z", 0.0) is not None:
        sfc = env.get("surface")
        wprop = env.get("wind_speed_propagation") or 0.0
        if sfc and sfc.get("rms_height"):
            coh = acoustics.rough_coherent_reflection(g, sfc["rms_height"], fc, c_w)
            ax.plot(gd, -20.0 * np.log10(np.maximum(coh, 1e-30)), "-", color="#1baf7a", lw=1.5,
                    label=f"surface, coherent part exp(-Γ²/2) at rms height {sfc['rms_height']:.3f} m (time spreading keeps the rest as taps)")
            if sfc.get("bubble_loss"):
                bl = sfc["bubble_loss"]
                ax.plot(bl["grazing_deg"], bl["loss_db"], "--", color="#1baf7a", lw=1.2, label="surface, bubble loss per bounce")
        elif wprop > 0:
            coh = acoustics.surface_coherent_reflection(g, wprop, fc, c_w)
            ax.plot(gd, -20.0 * np.log10(np.maximum(coh, 1e-30)), "-", color="#1baf7a", lw=1.5,
                    label=f"surface, coherent loss at wind {wprop:g} m/s (propagation.wind_speed)")
        else:
            ax.plot([0, 90], [0, 0], "-", color="#1baf7a", lw=1.5, label="surface, -1 (flat, pressure release): 0 dB")
    for pl in env.get("planes") or []:
        if pl["kind"] == "wall" and pl.get("coefficient") is not None:
            ax.plot([0, 90], [-20.0 * np.log10(max(pl["coefficient"], 1e-30))] * 2, ":", color="0.35", lw=1.2,
                    label=f"wall, constant |R| = {pl['coefficient']:g}")
    style = {"bottom": ("v", "#eb6834"), "surface": ("^", "#1baf7a"), "wall": ("s", "0.35"), "interface": ("o", "#eb6834")}
    for name, vals in used.items():
        mk, col = style.get(name, ("o", "0.3"))
        ga, la = np.array(vals).T
        extra = (" (flat mirror; time spreading then splits it)" if name == "surface" and (env.get("surface") or {}).get("rms_height")
                 else "")
        ax.plot(ga, la, mk, color=col, ms=6, mec="0.15", mew=0.5, ls="none", label=f"{name}: the drawn rays' bounces{extra}")
    if trace["tier"] == "free_field":
        ax.text(0.5, 0.5, "free field: no boundary reflects", ha="center", transform=ax.transAxes)
    ax.set_xlabel("grazing angle (deg)")
    ax.set_ylabel("loss per bounce (dB)")
    ax.set_title("loss per bounce the propagation applies, and the angles the rays used", fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6.5, loc="upper left")
    ax.set_xlim(0, 90)
    lo, hi = ax.get_ylim()
    if hi > 40.0:
        ax.text(0.99, 0.02, "curves clipped at 40 dB", transform=ax.transAxes, ha="right", fontsize=7, color="0.3")
    ax.set_ylim(min(lo, -0.5), min(max(hi * 1.35, 1.0), 40.0))


# --- plan view and patterns -----------------------------------------------------------------------------------------

def _draw_plan(ax, trace, max_rays: int):
    hyd = np.asarray(trace["sonar"]["hydrophones"], float)
    tx = np.asarray(trace["sonar"]["projector"], float)
    ref = hyd.mean(axis=0)
    view = trace.get("view") or {}
    track = np.asarray(view.get("track") or [], float)
    pts = [ref[:2]] + [np.asarray(t["position"][:2]) for t in trace["targets"]] + [np.asarray(s["position"][:2]) for s in trace.get("oneway") or []]
    reach = max([np.linalg.norm(p - ref[:2]) for p in pts] + [1.0])
    if len(track) > 1:
        ax.plot(track[:, 1], track[:, 0], ".", color="0.6", ms=2, label=f"platform at each of {len(track)} pings")
        k = view.get("ping_of_run") or 0
        if 0 <= k < len(track):
            ax.plot(track[k, 1], track[k, 0], "o", mfc="none", mec="0.1", ms=9, label=f"ping {k} (this page)")
    bm = trace.get("beams") or {}
    chain = bm.get("chain") or {}
    if chain.get("bearings_deg") and bm.get("broadside"):
        bs, ax_h = np.asarray(bm["broadside"])[:2], np.asarray(bm["axis"])[:2]
        first, last = np.radians(chain["bearings_deg"][:2])
        th = np.linspace(first, last, 60)
        fan = np.array([ref[:2]] + [ref[:2] + reach * (np.cos(t) * bs + np.sin(t) * ax_h) for t in th])
        ax.fill(fan[:, 1], fan[:, 0], color="#2a78d6", alpha=0.07, lw=0)
        for st in bm.get("steered") or []:
            for e, ls, lw in ((st["edges_deg"][0], ":", 0.7), (st["peak_deg"], "-", 0.8), (st["edges_deg"][1], ":", 0.7)):
                t = np.radians(e)
                v = np.cos(t) * bs + np.sin(t) * ax_h
                ax.plot([ref[1], ref[1] + reach * v[1]], [ref[0], ref[0] + reach * v[0]], ls, color="#2a78d6", lw=lw)
    for leg in trace["legs"]:
        rays = sorted([r for r in leg["out"] if r["polyline"] is not None], key=lambda r: -r["amplitude"])[:max_rays]
        for r in rays:
            p = np.asarray(r["polyline"])
            ax.plot(p[:, 1], p[:, 0], "-", color=CLASS_STYLE[bounce_class(r["n_surface"], r["n_bottom"])]["color"], lw=0.8, alpha=0.8)
            for b in r["bounces"]:
                if b["boundary"] == "wall":
                    ax.plot(b["xyz"][1], b["xyz"][0], "s", color="0.3", ms=4)
    for pl in trace["environment"].get("planes") or []:
        if pl["kind"] != "wall":
            continue
        n, p0 = np.asarray(pl["normal"]), np.asarray(pl["point"])
        d = np.array([-n[1], n[0]])
        if np.linalg.norm(d) < 1e-9:
            continue
        d = d / np.linalg.norm(d)
        span = np.linspace(-2 * reach, 2 * reach, 2)
        bounds = {ax_i: (lo, hi) for ax_i, lo, hi in (pl.get("bounds") or [])}
        seg = p0[None, :2] + span[:, None] * d[None, :]
        if 0 in bounds:
            seg[:, 0] = np.clip(seg[:, 0], *bounds[0])
        if 1 in bounds:
            seg[:, 1] = np.clip(seg[:, 1], *bounds[1])
        ax.plot(seg[:, 1], seg[:, 0], "-", color="0.35", lw=2.5, alpha=0.7)
    for src in trace.get("oneway") or []:
        p = np.asarray(src["position"])
        ax.plot([ref[1], p[1]], [ref[0], p[0]], ":", color=ONEWAY, lw=1.0)
        ax.plot(p[1], p[0], "P", color=ONEWAY, ms=9, mec="0.1", mew=0.5)
    ax.plot(hyd[:, 1], hyd[:, 0], "s", color="0.1", ms=2.5)
    ax.plot(tx[1], tx[0], "*", color="0.1", ms=9)
    for t in trace["targets"]:
        p = np.asarray(t["position"])
        ax.plot(p[1], p[0], "X", color=TARGET, ms=7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("east y (m)")
    ax.set_ylabel("north x (m)")
    if len(track) > 1:
        ax.legend(fontsize=6.5, loc="best")
    if chain.get("bearings_deg"):
        b = chain["bearings_deg"]
        title = (f"plan: bearings {b[0]:g} to {b[1]:g} deg by {b[2]:g} (shaded); {len(bm.get('steered') or [])} of them "
                 f"steered, -3 dB dotted\n({_short(chain.get('source'), 60)})")
    else:
        title = "plan\n(" + _short(chain.get("source", "no receive beams"), 70) + ")"
    ax.set_title(title, fontsize=8)


def _draw_patterns(ax, trace):
    bm = trace.get("beams")
    if not bm:
        ax.text(0.5, 0.5, "receive beams not drawn", ha="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ang = np.asarray(bm["angles_deg"])
    for i, st in enumerate(bm.get("steered") or []):
        ax.plot(ang, st["pattern_db"], "-", color="#2a78d6", lw=1.2 if abs(st["steer_deg"]) < 1e-9 else 0.7,
                alpha=0.9, label="receive beams, steered (array factor x element)" if i == 0 else None)
    ax.plot(ang, bm["element_db"], "--", color="0.2", lw=1.0,
            label="element" + (f" (-3 dB {bm['element_beamwidth_az_deg']:.0f}°)" if bm.get("element_beamwidth_az_deg") else ", omnidirectional"))
    if bm.get("projector"):
        ax.plot(ang, bm["projector"]["pattern_db"], ":", color=ONEWAY, lw=1.5, label="projector")
    ax.set_ylim(-40, 3)
    ax.set_xlim(-90, 90)
    ax.set_xlabel("angle from broadside, horizontal (deg)")
    ax.set_ylabel("power (dB)")
    ax.legend(fontsize=6.5, loc="lower center")
    ax.set_title(f"patterns at {bm['center_frequency'] / 1e3:.4g} kHz, {bm['n_elements']} elements over "
                 f"{bm['aperture_m']:.3g} m, {(bm['chain'].get('window') or 'no')} window\n" + _short(bm["projector_text"], 75), fontsize=8)


# --- the page -------------------------------------------------------------------------------------------------------

def _palette():
    """A fixed 256-colour palette: 128 greys, and each hue the page uses with 13 tints towards white. A thin coloured
    line, antialiased against white, maps to a tint of its own hue, never to a grey (what the adaptive 256-colour
    quantiser of the prototype did); a page is a third of its true-colour size (ours: flank_target_1km 144 kB
    against 486 kB)."""
    pal = []
    for g in range(0, 256, 2):
        pal += [g, g, g]
    for h in [c["color"] for c in CLASS_STYLE.values()] + [ONEWAY, TARGET, "#5dade2", "#c8a165", "#8a6d3b"]:
        c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)
        for t in np.linspace(0.0, 0.9, 13):
            pal += [int(v) for v in np.round(c + (255.0 - c) * t)]
    pal = pal[:768]
    return pal + [255] * (768 - len(pal))


def save_figure(fig, path: str, dpi: int, truecolour: bool = False) -> None:
    """PNG with the fixed palette of `_palette` (no dithering), or true colour with `truecolour`; SVG by extension."""
    if not str(path).lower().endswith(".png") or truecolour:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return
    import io

    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight", format="png")
    pal = Image.new("P", (1, 1))
    pal.putpalette(_palette())
    Image.open(buf).convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE).save(path, optimize=True)


def _draw_spectrogram(ax, x, s, trace, ch: int, what: str) -> None:
    """Spectrogram of one channel in the water's frequencies (a baseband record about its carrier), the transmit band
    as two lines and the strongest arrivals at this hydrophone as ticks on top."""
    fs, t0 = s["sample_rate"], s["start_time_s"]
    x = np.asarray(x)
    n = len(x)
    lo, hi = sorted(s["band_hz"])
    # at least 16 frequency bins across the band and its margin (a real record at many times the band's rate)
    width = (hi - lo) * 2.0 + 0.2 * hi if not np.iscomplexobj(x) else fs
    nfft = int(min(max(64, 2 ** int(np.ceil(np.log2(max(n / 150.0, 16.0 * fs / max(width, 1e-9), 1.0))))), 16384))
    hop = max(nfft // 2, int(np.ceil((n - nfft) / 300.0)), 1)
    if n < nfft:
        x = np.pad(x, (0, nfft - n))
    starts = np.arange(0, len(x) - nfft + 1, hop)
    w = np.hanning(nfft)
    seg = np.stack([x[i:i + nfft] * w for i in starts])
    if np.iscomplexobj(x):
        S = np.fft.fftshift(np.abs(np.fft.fft(seg, axis=1)) ** 2, axes=1).T
        f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs)) + s["center_frequency"]
    else:
        S = (np.abs(np.fft.rfft(seg, axis=1)) ** 2).T
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        pad = max(0.5 * (hi - lo), 0.1 * hi)
        sel = (f >= max(lo - pad, 0.0)) & (f <= hi + pad)
        S, f = S[sel], f[sel]
    db = 10.0 * np.log10(np.maximum(S / max(float(S.max()), 1e-30), 1e-6))
    tt = t0 + (starts + nfft / 2) / fs
    ax.imshow(db, aspect="auto", origin="lower", cmap="gray", vmin=-60, vmax=0, interpolation="nearest",
              extent=[tt[0], tt[-1] if len(tt) > 1 else tt[0] + nfft / fs, f[0] / 1e3, f[-1] / 1e3])
    for b in (lo, hi):
        ax.axhline(b / 1e3, color="#2a78d6", lw=0.8, ls="--")
    top = f[-1] / 1e3
    for p in sorted(trace["pairs"], key=lambda q: -q["level_db"])[:12]:
        ax.plot([p["channel_delay_s"][ch]] * 2, [top, top - 0.06 * (top - f[0] / 1e3)], "-",
                color=CLASS_STYLE[bounce_class(p["n_surface"], p["n_bottom"])]["color"], lw=1.5)
    ax.set_xlabel("time after transmit (s)")
    ax.set_ylabel("kHz (in the water)")
    ax.set_title(f"spectrogram of {what}, hydrophone {ch}, dB re its peak (60 dB)\ndashed: the transmit band; ticks: "
                 "the 12 strongest arrivals here", fontsize=8)


def _draw_items(ax, stages, s) -> int:
    """Level against time of each reverberation item (the opt-in per-item taps), with the echoes and the noise."""
    from . import audio
    meta = stages.get("__meta__")
    parts = audio.decompose({k: v for k, v in stages.items() if k != "__meta__"}, meta)
    fs, t0 = s["sample_rate"], s["start_time_s"]
    styles = ["-", "--", ":", "-."]
    n_items = 0
    for i, (k, (v, what)) in enumerate(parts.items()):
        if k == "reference_and_errors":
            continue
        pw = np.mean(np.abs(v) ** 2, axis=0)
        n = max(int(fs * 0.005), 1)
        pw = np.convolve(pw, np.ones(n) / n, mode="same")
        if not np.any(pw > 0.0):
            continue
        tt = t0 + np.arange(len(pw)) / fs
        item = k.startswith("reverberation_")
        n_items += item
        colour = "#eb6834" if item else {"echoes": "#2a78d6", "noise": "0.45"}.get(k, ONEWAY)
        ax.plot(tt, 10.0 * np.log10(np.maximum(pw, 1e-30)), styles[n_items % 4] if item else "-", color=colour, lw=1.0,
                label=k.replace("reverberation_", "item ").replace("_", " "))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=6.5)
        top = ax.get_ylim()[1]
        ax.set_ylim(max(ax.get_ylim()[0], top - 120.0), top + 3.0)
    ax.set_xlabel("time after transmit (s)")
    ax.set_ylabel("power, channel mean (dB)")
    ax.set_title("reverberation by item (opt-in per-item taps), with the echoes and the noise", fontsize=8)
    return n_items


def draw_page(trace: dict, record, path: str, stages: dict | None = None, n_marked: int = 12, max_rays: int = 6,
              dpi: int = 72, all_rays: bool = False, truecolour: bool = False) -> dict:
    """Write the page to `path` (PNG with a fixed hue-keeping palette, or true colour with `truecolour`, or SVG by its
    extension). `stages`: the simulator's own taps of this
    ping ({stage: samples}, taps.read_sim_taps), for the echoes-alone picture and the level of each part against time.
    The `max_rays` strongest eigenrays of each leg are drawn in colour with their bounces, the rest thin and grey
    (`all_rays`: every one in colour). Returns what was drawn (counts), for the tests."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = trace["environment"]
    fig = plt.figure(figsize=(16, 24.5))
    gs = fig.add_gridspec(7, 4, width_ratios=[0.75, 3, 3, 3], height_ratios=[0.12, 1.35, 1, 1, 1, 0.85, 0.95],
                          hspace=0.5, wspace=0.34)
    ax_head = fig.add_subplot(gs[0, :])
    ax_ssp = fig.add_subplot(gs[1, 0])
    ax_sec = fig.add_subplot(gs[1, 1:], sharey=ax_ssp)
    ax_rl = fig.add_subplot(gs[2, 0:2])
    ax_ss = fig.add_subplot(gs[2, 2])
    ax_pat = fig.add_subplot(gs[2, 3])
    ax_plan = fig.add_subplot(gs[3, 0:2])
    ax_stem = fig.add_subplot(gs[3, 2:])
    ax_wf = fig.add_subplot(gs[4, 0:2])
    ax_wf2 = fig.add_subplot(gs[4, 2], sharey=ax_wf)
    ax_lvl = fig.add_subplot(gs[4, 3])
    ax_sg = fig.add_subplot(gs[5, 0:2])
    ax_sg2 = fig.add_subplot(gs[5, 2])
    ax_items = fig.add_subplot(gs[5, 3])
    ax_txt = fig.add_subplot(gs[6, :])
    drawn = {"rays": 0, "bounces": 0, "pairs": len(trace["pairs"]), "marked": 0}

    # Header: what this page is, which ping, what it cannot show.
    ax_head.set_axis_off()
    s = trace["sonar"]
    view = trace.get("view") or {}
    head = (f"{trace['scenario']}   tier {trace['tier']}   sonar {s['name']} ({s['mode']}), "
            f"{s['band_hz'][0] / 1e3:.4g}-{s['band_hz'][1] / 1e3:.4g} kHz, {s['n_channels']} channels "
            f"({s['sample_kind']}, {s['sample_rate'] / 1e3:.6g} kHz)\n"
            f"shown: {view.get('note') or 'ping ' + str(trace['ping'])}.  Everything drawn is the simulator's own model, "
            "not a measurement (docs/tasks/VISUALIZATION.md section 0).")
    miss = trace.get("missing") or []
    if miss:
        head += "\nnot shown: " + "; ".join(miss)
    ax_head.text(0.0, 1.0, "\n".join(textwrap.fill(line, 205) for line in head.split("\n")), fontsize=9, va="top",
                 transform=ax_head.transAxes)

    # Sound-speed profile, one line per range where the water changes with range.
    sec = env.get("ssp_section")
    bottom_z = env.get("bottom_z") or env.get("bottom_z_site")
    if sec:
        greys = np.linspace(0.1, 0.65, len(sec["r"]))
        for r, row, gr in zip(sec["r"], sec["c"], greys):
            ax_ssp.plot(row, sec["z"], "-", color=str(gr), lw=1.2, label=f"{r / 1e3:g} km")
        ax_ssp.legend(fontsize=6.5, title="range", title_fontsize=6.5, loc="lower left")
        ax_ssp.set_title("profile by range", fontsize=9)
    elif env.get("ssp"):
        ax_ssp.plot(env["ssp"]["c"], env["ssp"]["z"], "-", color="0.2", lw=1.5)
        ax_ssp.set_title("profile", fontsize=9)
    else:
        ax_ssp.plot([env["sound_speed"]] * 2, [0.0, bottom_z or 1.0], "-", color="0.2", lw=1.5)
        ax_ssp.set_title("sound speed", fontsize=9)
    ax_ssp.set_xlabel("m/s")
    ax_ssp.set_ylabel("depth (m)")
    ax_ssp.tick_params(axis="x", labelrotation=45, labelsize=7)
    _draw_section(ax_sec, trace, max_rays, all_rays, drawn)
    plt.setp(ax_sec.get_yticklabels(), visible=False)

    _draw_losses(ax_rl, trace)

    # Scattering strength of each reverberation item (monostatic: incident = scattered angle).
    items = [m for m in trace["models"] if m["key"].startswith("reverberation[")]
    if items:
        styles = ["-", "--", ":", "-."]
        for i, sc in enumerate(trace.get("scattering") or []):
            ax_ss.plot(sc["grazing_deg"], sc["strength_db"], styles[i % 4], color="0.2", lw=1.5, label=f"{sc['key']}: {sc['model']}")
        if trace.get("scattering"):
            ax_ss.legend(fontsize=7, loc="lower right")
        ax_ss.set_xlim(0, 90)
        ax_ss.set_xlabel("grazing angle (deg)")
        ax_ss.set_ylabel("backscattering strength (dB)")
        ax_ss.set_ylim(-80.0, None)
    else:
        ax_ss.text(0.5, 0.5, "no reverberation items in this scenario\n(scene seafloor or surface patches,\nif any, are listed below)",
                   ha="center", va="center", transform=ax_ss.transAxes, fontsize=8)
        ax_ss.set_axis_off()
    ax_ss.set_title("scattering strength in force (monostatic)", fontsize=9)

    _draw_patterns(ax_pat, trace)
    _draw_plan(ax_plan, trace, max_rays)

    # Arrival stems: two-way pairs, one-way arrivals of the sources and the direct sound.
    pairs = trace["pairs"]
    levels = [p["level_db"] for p in pairs]
    if pairs:
        ref = max(levels)
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
    ow = [a for src in trace.get("oneway") or [] for a in src["arrivals"]]
    if ow:
        ref1 = max(a["level_db"] for a in ow)
        for i, a in enumerate(ow):
            lv = a["level_db"] - ref1
            if lv >= -40:
                ax_stem.plot([a["delay_s"]] * 2, [-42.0, lv], ":", color=ONEWAY, lw=1.0)
                ax_stem.plot(a["delay_s"], lv, "P", color=ONEWAY, ms=6, mec="white", mew=0.5,
                             label="one-way from the source (re its strongest)" if i == 0 else None)
    if trace.get("direct"):
        for i, a in enumerate(trace["direct"]["arrivals"]):
            ax_stem.axvline(a["delay_s"], color=ONEWAY, lw=1.0, ls="--", label="direct sound (one way)" if i == 0 else None)
    ax_stem.set_ylim(-42.0, 3.0)
    if ax_stem.get_legend_handles_labels()[0]:
        ax_stem.legend(fontsize=7, loc="upper right")
    else:
        ax_stem.text(0.5, 0.5, "no arrivals: no target and no source", ha="center", transform=ax_stem.transAxes)
    ax_stem.set_xlabel("delay (s): two-way for echoes, one-way for sources and the direct sound")
    ax_stem.set_ylabel("level re strongest (dB)")
    ax_stem.set_title("arrivals within 40 dB of the strongest (spreading and reflection; no absorption, no beams)", fontsize=9)

    # Channel by time, matched-filtered, with the strongest arrivals marked at each hydrophone.
    strongest = sorted(pairs, key=lambda p: -p["level_db"])[:n_marked]
    marks = [(p["channel_delay_s"], CLASS_STYLE[bounce_class(p["n_surface"], p["n_bottom"])]["color"]) for p in strongest]
    window = None
    if marks:
        lo = min(min(m[0]) for m in marks)
        hi = max(max(m[0]) for m in marks)
        pad = max(0.15 * (hi - lo), 20.0 / s["sample_rate"], 0.002)
        window = (lo - pad, hi + pad)
    if trace.get("direct"):
        # The direct sound is marked where it falls in the picture (it often arrives before the receive window opens).
        t_rec = (s["start_time_s"], s["start_time_s"] + s["n_samples"] / s["sample_rate"])
        w = window or t_rec
        marks += [(a["channel_delay_s"], ONEWAY) for a in trace["direct"]["arrivals"]
                  if w[0] <= np.mean(a["channel_delay_s"]) <= w[1]]
    parts = {"record": record}
    if stages and "echoes" in stages and pairs:
        parts["echoes"] = _with_samples(record, stages["echoes"])
    for ax, (name, rec) in zip((ax_wf, ax_wf2), parts.items()):
        t, env_ = compressed(rec)
        db = 20.0 * np.log10(np.maximum(env_ / max(env_.max(), 1e-30), 1e-6))
        nc = db.shape[0]
        ax.imshow(db, aspect="auto", origin="lower", cmap="gray", vmin=-40, vmax=0,
                  extent=[t[0], t[-1], -0.5, nc - 0.5], interpolation="nearest")
        for d, col in marks:
            ax.plot(d, np.arange(nc), "|", color=col, ms=4, mew=1.0)
        if window:
            ax.set_xlim(max(t[0], window[0]), min(t[-1], window[1]))
        ax.set_xlabel("time after transmit (s)")
        ax.set_title(("the record" if name == "record" else "echoes alone (simulator tap)")
                     + f", matched-filtered, dB re its peak\nticks: {len(strongest)} strongest two-way arrivals"
                     + (", the direct sound" if trace.get("direct") else "")
                     + ("; one-way arrivals not marked (their emission times are the source's)" if ow and name == "record" else ""),
                     fontsize=8)
    ax_wf.set_ylabel("hydrophone")
    drawn["marked"] = len(strongest)
    if len(parts) == 1:
        ax_wf2.set_axis_off()

    # Level of each simulated part against time (mean over the channels, smoothed), from the simulator's taps.
    if stages:
        prev = None
        colours = {"echoes": "#2a78d6", "direct_arrival": ONEWAY, "reverberation": "#eb6834", "noise": "0.45"}
        kinds = [w for w, on in (("reverberation (all items)", any(m["key"].startswith("reverberation[") for m in trace["models"])),
                                 ("the basin's late tail", any(m["model"] == "basin" for m in trace["models"])),
                                 ("one-way sources", bool(trace.get("oneway")))) if on]
        rest = " and ".join(kinds) or "reverberation"
        names = {"echoes": "echoes", "direct_arrival": "direct sound", "reverberation": rest, "noise": "ambient noise"}
        fs, t0 = s["sample_rate"], s["start_time_s"]
        direct = np.asarray(stages["direct_arrival"]) if "direct_arrival" in stages else None
        for k in [k for k in ("echoes", "direct_arrival", "reverberation", "noise") if k in stages]:
            x = np.asarray(stages[k])
            if k == "direct_arrival":
                part = x                                      # the direct sound alone (sim/simulator.py taps it alone)
            else:
                part = x if prev is None else x - prev
                if k == "reverberation" and direct is not None:
                    part = part - direct
                prev = x
            pw = np.mean(np.abs(part) ** 2, axis=0)
            n = max(int(fs * 0.005), 1)
            pw = np.convolve(pw, np.ones(n) / n, mode="same")
            if not np.any(pw > 0.0):
                continue                                   # a part this scenario does not have
            tt = t0 + np.arange(len(pw)) / fs
            ax_lvl.plot(tt, 10.0 * np.log10(np.maximum(pw, 1e-30)), "-", color=colours[k], lw=1.0, label=names[k])
        if ax_lvl.get_legend_handles_labels()[0]:
            ax_lvl.legend(fontsize=7)
            top = ax_lvl.get_ylim()[1]
            ax_lvl.set_ylim(max(ax_lvl.get_ylim()[0], top - 120.0), top + 3.0)
        ax_lvl.set_xlabel("time after transmit (s)")
        ax_lvl.set_ylabel("power, channel mean (dB)")
    else:
        ax_lvl.text(0.5, 0.5, "no simulator taps given", ha="center", transform=ax_lvl.transAxes)
    split = bool(stages) and any(k.startswith("reverberation_item_") for k in stages)
    ax_lvl.set_title("parts of the received signal against time,\nbefore matched filtering"
                     + (" (items: next row)" if split else " (items not split)"), fontsize=8)

    # Spectrogram of one hydrophone: the record and the echoes alone, in the water's frequencies, arrivals marked.
    ch = int((trace.get("audio") or {}).get("spectrogram_channel", 0))
    _draw_spectrogram(ax_sg, np.asarray(record.samples.array)[ch], s, trace, ch, "the record")
    if stages and "echoes" in stages and pairs:
        _draw_spectrogram(ax_sg2, np.asarray(stages["echoes"])[ch], s, trace, ch, "echoes alone")
    else:
        ax_sg2.set_axis_off()
    drawn["spectrogram_channel"] = ch
    # Reverberation by item (sim_capture per_item: surface, bottom, volume), from the simulator's own taps.
    drawn["items"] = _draw_items(ax_items, stages, s) if split else 0
    if not split:
        ax_items.text(0.5, 0.5, "reverberation by item: not tapped for this page\n(tools/scenario_view.py --audio taps it)"
                      if any(m["key"].startswith("reverberation[") for m in trace["models"]) else
                      "no reverberation items in this scenario", ha="center", va="center", fontsize=8,
                      transform=ax_items.transAxes)
        ax_items.set_axis_off()

    # What is modelled here: key, value, source, status.
    ax_txt.set_axis_off()
    rows = [("area", "model", "key", "value", "source", "status")]
    for m in trace["models"]:
        rows.append((m["area"], m["model"] + (" (default)" if m["chosen"] == "default" else ""), _short(m["key"], 34),
                     _short(m["value"], 44), _short(m["source"], 62), m["status"]))
    extra = []
    if env.get("absorption_db_per_km") is not None:
        extra.append(f"absorption {env['absorption']}: {env['absorption_db_per_km']:.3g} dB/km at {s['center_frequency'] / 1e3:.4g} kHz")
    if env.get("noise"):
        extra.append(f"noise: {_short(env['noise'], 80)}")
    widths = [max(len(str(r[i])) for r in rows) for i in range(6)]
    text = "\n".join("  ".join(str(r[i]).ljust(widths[i]) for i in range(6)) for r in rows)
    ax_txt.text(0.0, 1.0, "what is modelled here\n" + text + ("\n" + "; ".join(extra) if extra else ""), fontsize=7,
                va="top", family="monospace", transform=ax_txt.transAxes)

    save_figure(fig, path, dpi, truecolour)
    plt.close(fig)
    return drawn
