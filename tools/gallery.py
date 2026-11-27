#!/usr/bin/env python3
"""Run the realistic scenario of every mode end to end and save a picture and a few
numbers for each: simulate, condition, (estimate motion,) image, (detect, track).

  FLS    pool_fls_l12_5d, pool_fls_p4_2v, seafloor_objects_fls, harbour_fls_750k,
         fls_120k_transit, fls_170k_transit, fls_330k_transit (moving ownship, tracking)
  SAS    wreck_sas, minefield_sas (with motion estimation), sas_minsas_survey (two legs),
         rail_objects
  3D     buried_objects
  flank  flank_patrol, popoto_far_mover, popoto_trial_1000m (all need Bellhop)

Usage: tools/gallery.py [--backend cuda|host] [--only name,name] [--out DIR]
                        [--quick]   thinned scenes and fewer pings, for a CPU
                        [--evidence]  each track's classifier evidence in gallery.json (opt-in)
                        [--reverb doppler=centre] [--reverb bistatic_rule=mean_angle]  an old reverberation law (opt-in)

Writes <out>/<scenario>.png and <out>/gallery.json. --out defaults to docs/figures (the committed
full-size gallery) for a full run and to build/gallery_quick under the repository (not committed)
with --quick, so a quick run never overwrites the committed gallery; pass --out to put it elsewhere. The object-over-background figures
are the mean image power over each labelled object's scatterers against the image median.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import gpusonar as gs  # noqa: E402
from gpusonar import classify, detect, flank, motion, recon  # noqa: E402
from gpusonar.config import load_scenario_config  # noqa: E402
from gpusonar.sim import find_bellhop, run_scenario  # noqa: E402

core = gs.core


def db(power, ref):
    return 10.0 * np.log10(np.maximum(power / ref, 1e-12))


def display(power, target_median=0.2):
    """The paper's display mapping (dynamic range compression, Eq. 11-12) of an intensity image."""
    return core.drc(np.sqrt(power).astype(np.float32), target_median=target_median)[0]


def write_animation(path, anim, fps=2.0):
    """Write a GIF from `anim`: frames (2-D arrays), extent, labels, per-frame titles and
    optional per-frame overlays [(xs, ys, style dict), ...]."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    frames = anim["frames"]
    fig, ax = plt.subplots(figsize=anim.get("figsize", (6.4, 5.0)), constrained_layout=True)
    im = ax.imshow(frames[0], extent=anim["extent"], cmap="gray", vmin=anim["vmin"], vmax=anim["vmax"],
                   aspect=anim.get("aspect", "equal"), origin=anim.get("origin", "lower"), interpolation="nearest")
    ax.set_xlabel(anim["xlabel"])
    ax.set_ylabel(anim["ylabel"])
    title = ax.set_title(anim["titles"][0], fontsize=9)
    artists = []

    def draw(i):
        im.set_data(frames[i])
        title.set_text(anim["titles"][i])
        while artists:
            artists.pop().remove()
        for xs, ys, style in (anim.get("overlays") or [[]] * len(frames))[i]:
            artists.extend(ax.plot(xs, ys, **style))
        return [im, title]

    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(path, writer=PillowWriter(fps=fps), dpi=anim.get("dpi", 80))
    plt.close(fig)


def shrink(a, max_side=420):
    """Decimate a 2-D frame for an animation by block averaging."""
    k = max(int(np.ceil(max(a.shape) / max_side)), 1)
    if k == 1:
        return a
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def polar_to_grid(values, ranges, bearings, origin_xy, heading, gx, gy):
    """Resample a [range, bearing] map onto world points (gx ahead, gy to starboard)."""
    dx, dy = gx - origin_xy[0], gy - origin_xy[1]
    r = np.hypot(dx, dy)
    b = np.arctan2(dy, dx) - heading
    ir = np.rint((r - ranges[0]) / (ranges[1] - ranges[0])).astype(int)
    ib = np.rint((b - bearings[0]) / (bearings[1] - bearings[0])).astype(int)
    ok = (ir >= 0) & (ir < len(ranges)) & (ib >= 0) & (ib < len(bearings))
    out = np.zeros(gx.shape, dtype=np.float32)
    out[ok] = values[ir[ok], ib[ok]]
    return out


def sweep_frames(image, radius, n=10):
    """A ping travelling out: the image revealed ring by ring (`radius` per pixel)."""
    edges = np.linspace(radius[radius > 0].min(), radius.max(), n + 1)[1:]
    return [np.where(radius <= e, image, 0.0) for e in edges], [f"echoes back from within {e:.4g} m" for e in edges]


def object_levels(scene, power, to_pixel, median=None):
    """Mean image power over each labelled object's scatterers, in dB over the image median."""
    median = float(np.median(power)) if median is None else float(median)
    out = {}
    for label, mask in scene.label_masks().items():
        idx = [i for i in to_pixel(scene.positions[mask]) if i is not None]
        if len(idx) < 3:
            continue
        out[label] = float(db(np.mean([power[i] for i in idx]), median))
    return out


# --- forward-looking fan ---------------------------------------------------------------------

# Range is imaged at about the range resolution and averaged in power over `avg` samples
# for display; sampling range more coarsely than the resolution aliases the speckle.
FLS = {
    "pool_fls_l12_5d": dict(ranges=(0.8, 2.9, 0.00025), avg=4, bearings=(-24.0, 24.0, 0.1)),
    "pool_fls_p4_2v": dict(ranges=(5.2, 9.9, 0.0003), avg=6, bearings=(-32.0, 32.0, 0.2)),
    "seafloor_objects_fls": dict(ranges=(3.6, 10.0, 0.0025), avg=4, bearings=(-22.0, 22.0, 0.1)),
    "harbour_fls_750k": dict(ranges=(4.5, 31.0, 0.003), avg=4, bearings=(-50.0, 50.0, 0.2)),
}


def run_fls(name, cfg, backend, quick):
    p = FLS[name]
    result = run_scenario(cfg, backend=backend)
    records = result.records
    ranges = np.arange(*p["ranges"])
    bearings = np.radians(np.arange(*p["bearings"]))
    avg = p["avg"]
    if quick:
        bearings = bearings[::2]
    ranges = ranges[:len(ranges) // avg * avg]
    # Hann aperture weighting (strong objects would otherwise ring across the whole fan) and
    # a range gain of 30 log r for display: spreading both ways, less the growth of the
    # resolution cell with range.
    # Sinc interpolation: linear interpolation shows as a stripe at broadside, where every
    # channel shares one fractional delay.
    img = recon.fan_image(records, ranges, bearings, interp="sinc", window="hann", backend=backend)
    power = np.abs(img.image) ** 2 * ranges[:, None] ** 3
    power = power.reshape(-1, avg, power.shape[1]).mean(axis=1)
    ranges = ranges.reshape(-1, avg).mean(axis=1)
    origin = np.vstack([r.rx_positions() for r in records]).mean(axis=0)

    def to_pixel(points):
        d = points - origin
        r = np.linalg.norm(d, axis=1)
        b = np.arctan2(d[:, 1], d[:, 0])
        ir = np.rint((r - ranges[0]) / (ranges[1] - ranges[0])).astype(int)
        ib = np.rint((b - bearings[0]) / (bearings[1] - bearings[0])).astype(int)
        return [(i, j) if 0 <= i < len(ranges) and 0 <= j < len(bearings) else None for i, j in zip(ir, ib)]

    seen = result.scene.visible(origin)
    scene = result.scene
    levels = object_levels(scene, power, to_pixel)
    R, B = np.meshgrid(ranges, bearings, indexing="ij")
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    m = ax.pcolormesh(R * np.sin(B), R * np.cos(B), display(power, p.get("median", 0.2)), cmap="gray",
                      vmin=0.0, vmax=1.0, shading="auto", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("across (m)")
    ax.set_ylabel("forward (m)")
    ax.set_title(f"{name}: {cfg['sonar']}, {len(records)} ping(s), {len(scene)} scatterers (simulated)")
    fig.colorbar(m, ax=ax, label="display level (range gain, then dynamic range compression)", shrink=0.8)
    half = ranges[-1] * np.sin(bearings[-1])
    gy, gx = np.meshgrid(np.linspace(-half, half, 420), np.linspace(ranges[0] * 0.9, ranges[-1], 380))
    cart = polar_to_grid(display(power, p.get("median", 0.2)), ranges, bearings, (0.0, 0.0), 0.0, gx, gy)
    frames, titles = sweep_frames(cart, np.hypot(gx, gy))
    anim = dict(frames=frames, titles=[f"{name}: {t}" for t in titles], extent=[-half, half, gx[0, 0], gx[-1, 0]],
                xlabel="across (m)", ylabel="forward (m)", vmin=0.0, vmax=1.0)
    return fig, {"scatterers": len(scene), "hidden_fraction": float(1.0 - seen.mean()),
                 "object_over_median_db": levels,
                 "dynamic_range_db": float(db(power.max(), np.median(power)))}, anim


# --- stripmap SAS ----------------------------------------------------------------------------

SAS = {
    "wreck_sas": dict(x=(6.0, 13.0), y=(39.0, 46.5), pixel=0.025),
    "minefield_sas": dict(x=(4.0, 16.0), y=(38.5, 47.0), pixel=0.025),
}


def sharpness(im):
    return float((im**2).sum() / im.sum() ** 2 * im.size)


def run_sas(name, cfg, backend, quick):
    p = SAS[name]
    result = run_scenario(cfg, backend=backend)
    records, truth = result.records, result.true_positions
    recon.prepare(records, backend=backend)
    pixel = p["pixel"] * (2 if quick else 1)
    x, y = np.arange(*p["x"], pixel), np.arange(*p["y"], pixel)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    def image(recs):
        return np.abs(recon.sas_image(recs, points, interp="linear", backend=backend)) ** 2

    panels = {"Navigation only": image(records)}
    metrics = {"scatterers": len(result.scene), "pings": len(records)}
    if core.HAVE_MOTION:
        estimate = motion.estimate_motion(records)
        corrected = [r.copy() for r in records]
        motion.apply_motion(corrected, estimate)
        panels["Estimated motion"] = image(corrected)
        wavelength = records[0].environment.sound_speed / records[0].samples.center_frequency
        metrics["sway_error_rms_wavelengths"] = float(np.sqrt(np.mean((estimate.y - truth[:, 1]) ** 2)) / wavelength)
        metrics["heave_error_rms_wavelengths"] = float(np.sqrt(np.mean((estimate.z - truth[:, 2]) ** 2)) / wavelength)
    exact = [r.copy() for r in records]
    interval = float(cfg["pings"]["interval"])
    for r, pos, vel in zip(exact, truth, np.gradient(truth, interval, axis=0)):
        geo = r.geometry
        pose = geo.rx_pose
        pose.position, pose.velocity = tuple(pos), tuple(vel)
        geo.rx_pose = geo.tx_pose = pose
        r.geometry = geo
    panels["True motion"] = image(exact)
    metrics["sharpness"] = {k: sharpness(v) for k, v in panels.items()}
    best = panels.get("Estimated motion", panels["True motion"])

    def to_pixel(pts):
        i = np.rint((pts[:, 0] - x[0]) / pixel).astype(int)
        j = np.rint((pts[:, 1] - y[0]) / pixel).astype(int)
        return [(a, b) if 0 <= a < len(x) and 0 <= b < len(y) else None for a, b in zip(i, j)]

    metrics["object_over_median_db"] = object_levels(result.scene, best, to_pixel)
    ref = np.median(panels["True motion"])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.2), sharey=True, constrained_layout=True)
    for ax, (title, im) in zip(np.atleast_1d(axes), panels.items()):
        m = ax.imshow(db(im, ref), extent=[y[0], y[-1], x[-1], x[0]], cmap="gray", vmin=-15, vmax=30, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("ground range y (m)")
    np.atleast_1d(axes)[0].set_ylabel("along track x (m)")
    fig.colorbar(m, ax=axes, label="dB re median seafloor", shrink=0.85)
    fig.suptitle(f"{name}: {len(records)} pings, {len(result.scene)} scatterers, motion unknown to navigation (simulated)")
    order = [k for k in ("Navigation only", "Estimated motion", "True motion") if k in panels]
    anim = dict(frames=[shrink(np.clip(db(panels[k], ref), -15, 30)) for k in order],
                titles=[f"{name}: {k.lower()}" for k in order], extent=[y[0], y[-1], x[-1], x[0]], origin="upper",
                xlabel="ground range y (m)", ylabel="along track x (m)", vmin=-15, vmax=30)
    return fig, metrics, anim


def run_rail(name, cfg, backend, quick):
    result = run_scenario(cfg, backend=backend)
    records = result.records
    pixel = 0.002 if quick else 0.001
    x, y = np.arange(-0.02, 0.32, pixel), np.arange(1.78, 1.985, pixel)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    power = np.abs(recon.sas_image(records, points, interp="linear", backend=backend)) ** 2

    def to_pixel(pts):
        i = np.rint((pts[:, 0] - x[0]) / pixel).astype(int)
        j = np.rint((pts[:, 1] - y[0]) / pixel).astype(int)
        return [(a, b) if 0 <= a < len(x) and 0 <= b < len(y) else None for a, b in zip(i, j)]

    levels = object_levels(result.scene, power, to_pixel)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    m = ax.imshow(db(power, np.median(power)).T, extent=[x[0] * 100, x[-1] * 100, y[-1], y[0]], cmap="gray",
                  vmin=-10, vmax=40, aspect="auto")
    ax.invert_yaxis()
    ax.set_xlabel("along the rail (cm)")
    ax.set_ylabel("ground range (m)")
    ax.set_title(f"{name}: {cfg['sonar']}, {len(records)} pings, {len(result.scene)} scatterers (simulated)")
    fig.colorbar(m, ax=ax, label="dB re image median", shrink=0.8)
    coarse = points[::2, ::2]
    frames, titles = [], []
    for k in np.linspace(len(records) / 6, len(records), 6).astype(int):
        part = np.abs(recon.sas_image(records[:k], coarse, interp="linear", backend=backend)) ** 2
        frames.append(np.clip(db(part, np.median(power)), -10, 40).T)
        titles.append(f"{name}: first {k} of {len(records)} pings ({k * 0.2:.1f} cm of rail)")
    anim = dict(frames=frames, titles=titles, extent=[x[0] * 100, x[-1] * 100, y[0], y[-1]], aspect="auto",
                xlabel="along the rail (cm)", ylabel="ground range (m)", vmin=-10, vmax=40)
    return fig, {"scatterers": len(result.scene), "pings": len(records), "object_over_median_db": levels,
                 "sharpness": sharpness(power)}, anim


# --- buried objects --------------------------------------------------------------------------

def run_buried(name, cfg, backend, quick):
    result = run_scenario(cfg, backend=backend)
    records = result.records
    recon.prepare(records, backend=backend)
    step = 0.04 if quick else 0.02
    x, y = np.arange(-0.6, 0.7001, step), np.arange(-0.5, 0.7001, step)
    depths = np.arange(0.02, 0.5001, 0.01)
    prop = cfg["propagation"]
    vol = recon.image_volume(records, x, y, depths, interface_z=float(prop["interface_z"]),
                             sediment_speed=float(prop["sediment_speed"]), backend=backend)
    plain = recon.image_volume(records, x, y, depths, backend=backend)
    mag = np.abs(vol.data)

    def top_of(mag_, cx, cy):
        i, j = int(np.argmin(np.abs(x - cx))), int(np.argmin(np.abs(y - cy)))
        return float(depths[int(np.argmax(mag_[:, i, j]))])

    metrics = {"scatterers": len(result.scene), "pings": len(records),
               "cylinder_top_depth_m": {"true": 0.25, "refraction_on": top_of(mag, 0.0, 0.0),
                                        "refraction_off": top_of(np.abs(plain.data), 0.0, 0.0)},
               "sphere_top_depth_m": {"true": 0.14, "refraction_on": top_of(mag, 0.35, 0.40),
                                      "refraction_off": top_of(np.abs(plain.data), 0.35, 0.40)}}
    clutter = np.median(mag[(depths > 0.08) & (depths < 0.45)])
    metrics["cylinder_over_clutter_db"] = float(20 * np.log10(mag[:, int(np.argmin(np.abs(x))), int(np.argmin(np.abs(y)))].max() / clutter))
    ref = clutter
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, depth, title in ((axes[0], 0.25, "Depth slice at 0.25 m (cylinder top)"),
                             (axes[1], 0.14, "Depth slice at 0.14 m (sphere top)")):
        m = ax.imshow(20 * np.log10(np.maximum(vol.depth_slice(depth) / ref, 1e-6)),
                      extent=[y[0], y[-1], x[-1], x[0]], cmap="gray", vmin=-6, vmax=30, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("y (m)")
        ax.set_ylabel("x (m)")
    side = mag.max(axis=2)                                   # brightest voxel along y: x against depth
    axes[2].imshow(20 * np.log10(np.maximum(side / ref, 1e-6)), extent=[x[0], x[-1], depths[-1], depths[0]],
                   cmap="gray", vmin=-6, vmax=30, aspect="equal")
    axes[2].set_title("Brightest along y: x against depth")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("depth below the interface (m)")
    fig.colorbar(m, ax=axes, label="dB re sediment clutter", shrink=0.85)
    fig.suptitle(f"{name}: {len(records)} pings, refraction at the interface on (simulated)")
    sel = range(0, len(depths), 2)
    anim = dict(frames=[np.clip(20 * np.log10(np.maximum(mag[i] / ref, 1e-6)), -6, 30) for i in sel],
                titles=[f"{name}: slice {depths[i] * 100:.0f} cm below the interface" for i in sel],
                extent=[y[0], y[-1], x[-1], x[0]], origin="upper", xlabel="y (m)", ylabel="x (m)", vmin=-6, vmax=30,
                figsize=(5.4, 5.4))
    return fig, metrics, anim


# --- flank array -----------------------------------------------------------------------------

# The range of a detection in the scenes that run the energy detector over multipath (run_flank, run_popoto_far):
# "first" (the default since the owner's choice on sonar-ca3 / sonar-a98: the nearest of its cells,
# Detection.first_range, the leading edge) or "centroid" (`--range-estimate centroid`: detect.cluster's power-weighted
# centroid of the detection's cells, the gallery before that choice). In these scenes the echo of one object is several
# arrivals by the surface and bottom, the later ones the stronger, which merge into one detection whose centroid lies
# 28-38 m beyond the object; the leading edge is the earliest arrival's. It reads short where the earliest arrival is
# the near end of a long hull at the cluster's mean bearing (popoto_env_gravel_30m: 10 m short of the hull) or a strong
# echo's leading sidelobe. The live chain reads the leading edge by default (D-160); its detector is another one
# (multiscale), so the two are not the same numbers.
RANGE_ESTIMATES = ("centroid", "first")
RANGE_ESTIMATE = "first"


def set_range_estimate(hits, estimate=None):
    """`hits` (detect.cluster's) with their range the estimate `estimate` (default RANGE_ESTIMATE) asks for:
    unchanged for "centroid", Detection.first_range for "first". Changes the detections in place and returns them."""
    estimate = RANGE_ESTIMATE if estimate is None else estimate
    if estimate not in RANGE_ESTIMATES:
        raise ValueError(f"range estimate {estimate!r}: one of {', '.join(RANGE_ESTIMATES)}")
    if estimate == "first":
        for h in hits:
            h.range = float(h.first_range)
    return hits


# The LFM range-Doppler coupling correction (sonar-fyj; the live chain's `range_doppler`, opt-in) in the tracking scenes
# (run_flank, run_popoto_far, run_fls_transit): "off" (the default: not corrected, the gallery as before), "ownship"
# (each detection's range + f0 T / B x the ownship's closing speed along its line of sight, detect.lfm_range_doppler_factor
# and the record's transmit and receive velocities) or "full" (that, and each classified track's position moved along
# the line of sight from the ownship by the factor x its own closing speed, classify.shift_for_target_doppler, through
# classify_and_score's `range_doppler`). `--range-doppler`.
RANGE_DOPPLERS = detect.RANGE_DOPPLER
RANGE_DOPPLER = "off"


def doppler_range(record, rng, look):
    """Range `rng` (m) of a detection seen along the world unit vector `look` (x, y) with the ownship's part of the LFM
    coupling taken out when RANGE_DOPPLER is not "off" (as the live chain does: range + factor x (v_tx + v_rx) / 2 . look);
    unchanged otherwise."""
    if RANGE_DOPPLER not in RANGE_DOPPLERS:
        raise ValueError(f"RANGE_DOPPLER = {RANGE_DOPPLER!r}: one of {RANGE_DOPPLERS}")
    if RANGE_DOPPLER == "off":
        return float(rng)
    g = record.geometry
    v = 0.5 * (np.asarray(g.tx_pose.velocity, dtype=float) + np.asarray(g.rx_pose.velocity, dtype=float))
    return float(rng) + detect.lfm_range_doppler_factor(record.transmit.waveform) * float(v[:2] @ np.asarray(look, dtype=float)[:2])


def target_doppler(record, origin_yx):
    """classify_and_score's `range_doppler` for a run (the target's part, "full" only): {"factor": s, "origin": the
    ownship in the tracker's (y, x) order}; None otherwise."""
    if RANGE_DOPPLER != "full":
        return None
    return {"factor": detect.lfm_range_doppler_factor(record.transmit.waveform), "origin": [float(v) for v in origin_yx]}


def run_flank(name, cfg, backend, quick):
    result = run_scenario(cfg, backend=backend)
    interval = float(cfg["pings"]["interval"])
    bearings = np.radians(np.arange(-40.0, 40.01, 0.5))
    tracker = detect.Tracker(range_sigma=5.0, bearing_sigma=np.radians(0.7), confirm_hits=3, max_misses=2,
                             initial_speed_sigma=5.0)
    first = None
    counts, maps = [], []
    for i, rec in enumerate(result.records):
        bm = flank.farfield_beamform(rec, bearings, window="hann").valid()
        # Multipath spreads each echo over tens of metres, far more than a range sample, so
        # power is summed into 4 m range cells before detection (an energy detector); the
        # CFAR guard (60 m) and training (240 m) windows are set in those cells.
        fine = np.abs(bm.data) ** 2
        k = max(int(round(4.0 / (bm.ranges[1] - bm.ranges[0]))), 1)
        n = fine.shape[1] // k * k
        power = fine[:, :n].reshape(len(bearings), -1, k).mean(axis=2)
        ranges = bm.ranges[:n].reshape(-1, k).mean(axis=1)
        hits = detect.cluster(detect.cfar(power, pfa=1e-6, guard=15, train=60), power, ranges, bearings,
                              guard=15, train=60)
        hits = set_range_estimate([h for h in hits if h.snr_db > 12.0])
        for h in hits:
            h.range = doppler_range(rec, h.range, (np.sin(h.bearing), np.cos(h.bearing)))     # sonar-fyj, opt-in
            h.time = i * interval + echo_delay(rec, h.range)      # where the echo shows the target (D-156)
        counts.append(len(hits))
        tracks = tracker.update(i * interval, hits)
        if first is None:
            first = (ranges, power, hits)
        maps.append(shrink(np.clip(db(power, np.median(power)), 0, 40)))
    t_last = (len(result.records) - 1) * interval
    truth = {}
    start = 0
    for item in cfg["scene"]:
        n = len(item["positions"])
        v = np.asarray(item.get("velocity", (0.0, 0.0, 0.0)), dtype=float)
        truth[item["label"]] = TrueObject(np.asarray(item["positions"], dtype=float).mean(axis=0), v, item_footprint(item))
        start += n
    metrics = {"pings": len(result.records), "detections_per_ping": counts, "confirmed_tracks": len(tracks), "targets": {}}
    # Classified as the other world-frame scenes are, through classify_and_score in the tracker order the scoring reads
    # (flank_tracks_yx: this tracker's state is (x, y)); the classify options are classify_tracks' defaults, the gate
    # this scene's own 60 m, the array (the origin) the sonar for the outlier test (sonar-a98). Added; the targets
    # scored below are unchanged.
    _, metrics["classification"] = classify_and_score(flank_tracks_yx(tracks), truth, t_last, 60.0, origin=(0.0, 0.0),
                                                      range_doppler=target_doppler(result.records[-1], (0.0, 0.0)))
    for label, obj in truth.items():
        pos, vel = obj

        def miss(t, centre=False):                     # against the truth where the track's state holds (D-156)
            # This tracker's state is (x, y) in the world frame: from the object's outline (D-157), or its centre.
            if centre:
                return float(np.linalg.norm(t.state[:2] - (pos[:2] + vel[:2] * state_time(t))))
            return outline_distance(t.state[:2], object_outline(obj, state_time(t)))[0]
        best = min(tracks, key=lambda t: (miss(t), miss(t, True)), default=None)
        if best is None:
            metrics["targets"][label] = {"tracked": False}
            continue
        metrics["targets"][label] = {
            "tracked": bool(miss(best) < 60.0),
            "position_error_m": miss(best), "centroid_error_m": miss(best, True),
            "true_velocity": [float(vel[0]), float(vel[1])],
            "track_velocity": [float(best.state[2]), float(best.state[3])]}
    ranges, power, hits = first
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    floor = np.median(power)
    for h in hits:
        axes[0].plot(h.range, np.degrees(h.bearing), "x", color="C3", ms=6)
    m = axes[0].imshow(db(power, floor), extent=[ranges[0], ranges[-1], np.degrees(bearings[-1]), np.degrees(bearings[0])],
                       cmap="gray", vmin=0, vmax=40, aspect="auto")
    axes[0].invert_yaxis()
    for label, (pos, _) in truth.items():
        axes[0].plot(np.hypot(pos[0], pos[1]), np.degrees(np.arctan2(pos[0], pos[1])), "o", mfc="none", ms=14, label=label)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_xlabel("range (m)")
    axes[0].set_ylabel("bearing (deg)")
    axes[0].set_title("First ping: range x bearing map (circles: truth, crosses: detections)")
    fig.colorbar(m, ax=axes[0], label="dB re map median", shrink=0.8)
    for t in tracks:
        h = np.array(t.history)
        axes[1].plot(h[:, 1], h[:, 2], "-", lw=2)
        axes[1].plot(h[-1, 1], h[-1, 2], "k.", ms=6)
    for label, (pos, vel) in truth.items():
        end = pos[:2] + vel[:2] * t_last
        axes[1].plot([pos[0], end[0]], [pos[1], end[1]], "--", lw=1, color="0.4")
        axes[1].annotate(label, end, textcoords="offset points", xytext=(6, 6), fontsize=8)
    axes[1].set_xlabel("along the array x (m)")
    axes[1].set_ylabel("broadside y (m)")
    axes[1].set_title(f"Confirmed tracks over {len(result.records)} pings (dashed: truth)")
    axes[1].set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"{name}: {cfg['sonar']}, multipath, bottom and surface reverberation, ambient noise (simulated)")
    overlays = []
    for i in range(len(maps)):
        marks = []
        for label, (pos, vel) in truth.items():
            q = pos[:2] + vel[:2] * i * interval
            marks.append(([np.hypot(q[0], q[1])], [np.degrees(np.arctan2(q[0], q[1]))],
                          dict(marker="o", mfc="none", mec="#eb6834", ms=13, ls="none")))
        overlays.append(marks)
    anim = dict(frames=maps, titles=[f"{name}: ping {i + 1} of {len(maps)}, t = {i * interval:.0f} s (circles: truth)" for i in range(len(maps))],
                extent=[ranges[0], ranges[-1], np.degrees(bearings[0]), np.degrees(bearings[-1])], aspect="auto",
                xlabel="range (m)", ylabel="bearing (deg)", vmin=0, vmax=40, overlays=overlays, figsize=(7.0, 4.6))
    return fig, metrics, anim


def flank_tracks_yx(tracks):
    """run_flank's tracks seen in the order the gallery's scoring and classify_and_score read (world_tracks: state
    [y, x, vy, vx], history (t, y, x), measured (y, x)). run_flank feeds its tracker bearings from broadside (+y)
    towards +x, so its states are (x, y, vx, vy); scored as they are, object_track compared a track's (y, x) with the
    truth (sonar-a98). A view with the axes swapped; everything else (times, pings, snr, cells, counts) as tracked."""
    from types import SimpleNamespace
    out = []
    for t in tracks:
        s = np.asarray(t.state, dtype=float)
        out.append(SimpleNamespace(
            id=t.id, state=np.array([s[1], s[0], s[3], s[2]]), hits=t.hits, misses=getattr(t, "misses", 0),
            history=[(h[0], h[2], h[1]) for h in t.history], measured=[(m[1], m[0]) for m in getattr(t, "measured", [])],
            pings=list(getattr(t, "pings", [])), snr=list(t.snr), cells=list(t.cells)))
    return out


def run_popoto(name, cfg, backend, quick):
    result = run_scenario(cfg, backend=backend)
    rec = result.records[0]
    target = result.scene.positions[0]
    bearings = np.radians(np.arange(-30.0, 30.01, 1.0))
    bm = flank.farfield_beamform(rec, bearings).valid()
    power = np.abs(bm.data) ** 2
    floor = np.median(power)
    ib, ir = np.unravel_index(np.argmax(power), power.shape)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    m = ax.imshow(db(power, floor), extent=[bm.ranges[0], bm.ranges[-1], np.degrees(bearings[-1]), np.degrees(bearings[0])],
                  cmap="gray", vmin=0, vmax=35, aspect="auto")
    ax.invert_yaxis()
    ax.plot(np.linalg.norm(target[:2]), 0.0, "o", mfc="none", ms=14, color="C1")
    ax.set_xlabel("range (m)")
    ax.set_ylabel("bearing (deg)")
    ax.set_title(f"{name}: 8 hydrophones, multipath, bottom reverberation, noise (simulated)")
    fig.colorbar(m, ax=ax, label="dB re map median", shrink=0.8)
    # The echo arriving: the map revealed range by range, as it is received.
    full = shrink(np.clip(db(power, floor), 0, 35))
    r_axis = np.linspace(bm.ranges[0], bm.ranges[-1], full.shape[1])
    edges = np.linspace(r_axis[0], r_axis[-1], 11)[1:]
    anim = dict(frames=[np.where(r_axis[None, :] <= e, full, 0.0) for e in edges],
                titles=[f"{name}: echoes received from within {e:.0f} m" for e in edges],
                extent=[bm.ranges[0], bm.ranges[-1], np.degrees(bearings[0]), np.degrees(bearings[-1])], aspect="auto",
                xlabel="range (m)", ylabel="bearing (deg)", vmin=0, vmax=35, figsize=(7.0, 4.2))
    metrics = {"peak_range_m": float(bm.ranges[ir]), "peak_bearing_deg": float(np.degrees(bearings[ib])),
                 "true_range_m": float(np.linalg.norm(target[:2])),
                 "peak_over_median_db": float(db(power.max(), floor))}
    return fig, metrics, anim


# --- moving ownship: detection and tracking in the world frame ---------------------------------

def world_tracks(detections_per_ping, times, **tracker_kw):
    """Track detections given as world (x, y, snr) or (x, y, snr, delay) per ping, `delay` (s) after the ping's
    transmit time being when the detection shows the target (range / c for a monostatic echo: `echo_delay`; D-156).
    Returns the tracker's confirmed tracks; each state is [y, x, vy, vx] (the tracker's across, forward order) and
    holds at the time of the track's last history point (`state_time`)."""
    tracker = detect.Tracker(**tracker_kw)
    tracks = []
    for t, hits in zip(times, detections_per_ping):
        tracks = tracker.update(float(t), [detect.Detection(float(np.hypot(h[0], h[1])), float(np.arctan2(h[1], h[0])), h[2], 1,
                                                            time=float(t) + float(h[3]) if len(h) > 3 else None)
                                           for h in hits])
    return tracks


def echo_delay(record, rng):
    """When, after the ping's transmit time, a monostatic echo from range `rng` (m) shows its target: rng / c, with the
    record's sound speed c, the one the beamformers turn two-way time into range with (flank.farfield_beamform,
    recon.fan_image). Every sonar the gallery runs transmits from within a few metres of its array (D-156)."""
    return float(rng) / float(record.environment.sound_speed)


def echo_position(origin, velocity, rng, bearing, c):
    """World (x, y) of a monostatic echo at two-way range `rng` (m, c tau / 2) and far-field bearing `bearing` (rad,
    from +y towards +x, the array along x) from an array at `origin` at transmit, moving at `velocity` (world, m/s, the
    record's rx pose velocity), with sound speed `c`.

    The simulator receives with the array where it is when the echo arrives (its elements carry the pose's velocity),
    so the bearing is the direction u from there, p_rx = origin + d, d = v 2 rng / c, to the target q, and the two-way
    path is |q - origin| + |q - p_rx| = 2 rng. To first order in |d| / rng, |q - origin| = |q - p_rx| + u . d, so
    q = p_rx + (rng - u . d / 2) u (ours; the live chain follows the same motion in its backprojection, sonar-8is).
    Placed from `origin` instead, as before sonar-960, the echo lies d behind: 24 m along the track at 12 km and 1.5 m/s
    (popoto_12km). Left out: the order |d|^2 / rng (0.05 m there), and the array's motion during a long pulse, which
    moves the bearing further by up to the distance it travels in half the pulse (a 1 s pulse at 1.5 m/s: 0.75 m;
    measured with the simulator at 25 m/s and 300 m: 0.2 m left for a 10 ms pulse, 1.5 m for 0.1 s, 4 m for 1 s). An
    array at rest (stop and go) gives the old placement exactly."""
    d = np.asarray(velocity, dtype=float)[:2] * (2.0 * float(rng) / float(c))
    u = np.array([np.sin(bearing), np.cos(bearing)])
    p = np.asarray(origin, dtype=float)[:2] + d + (float(rng) - 0.5 * float(u @ d)) * u
    return float(p[0]), float(p[1])


def state_time(tr):
    """When a track's state holds: the time of its last history point (its last detection, or the last ping after a
    miss; a capture saved before D-156 holds at the last ping)."""
    return float(tr.history[-1][0])


def score_tracks(tracks, truth, t_last, gate, kinds=None):
    """For each true object (position at t = 0, velocity), its track at the last ping: the one `object_track` picks
    with the classes `kinds` (track id -> kind; D-155), so that the position, the velocity and the class reported for
    an object come from the same track. Without `kinds`, the nearest track within `gate`. No track within `gate`: the
    nearest track's error, not tracked. Each track is compared with the object where it was when the track's state
    holds (`state_time`; D-156): `position_error_m` (and the gate) from its outline for an extended object (`_miss`,
    D-157), `centroid_error_m` from its centre (the score before D-157; the same for a point object). `t_last` is kept
    for the callers. The states do not all hold at it: a track that took a detection on the last ping holds at
    t_last + R/c, its reflection time, and one that missed it at t_last or, with a ping interval shorter than R/c, at
    its last detection's reflection time if that is later."""
    kinds = kinds if kinds is not None else {tr.id: "uncertain" for tr in tracks}
    out = {}
    for label, obj in truth.items():
        pos, vel = obj
        best = object_track(tracks, kinds, obj, gate)
        if best is None:
            best = min(tracks, key=lambda tr: (_miss(tr, obj), _centroid_miss(tr, obj)), default=None)
        if best is None:
            out[label] = {"tracked": False}
            continue
        err = _miss(best, obj)
        out[label] = {"tracked": bool(err < gate), "position_error_m": err, "centroid_error_m": _centroid_miss(best, obj),
                      "true_velocity_xy": [float(vel[0]), float(vel[1])],
                      "track_velocity_xy": [float(best.state[3]), float(best.state[2])]}
    return out


class TrueObject(tuple):
    """A true object: (centre at t = 0, velocity), as a pair, with `footprint`, its horizontal outline at t = 0 (world
    x, y: the corners of the convex hull of what the scene puts there, an (N, 2) array), or None for a point object.
    An extended object (a ship of several points, a hull, a boat) is scored by the distance from a track to that
    outline, moved by the velocity (`outline_distance`); a point object by the distance to its centre, as before.
    Ours (D-157): the popoto_12km ship is five points over 120 m, and a track on its bow or stern was scored 60 m off."""

    def __new__(cls, centre, velocity, footprint=None):
        obj = super().__new__(cls, (np.asarray(centre, dtype=float), np.asarray(velocity, dtype=float)))
        obj.footprint = None if footprint is None else convex_hull(np.asarray(footprint, dtype=float).reshape(-1, 2))
        return obj

    def __reduce__(self):
        return (TrueObject, (self[0], self[1], self.footprint))


def convex_hull(points) -> np.ndarray:
    """The corners of the convex hull of 2-D points, counter-clockwise (Andrew's monotone chain); collinear points give
    the segment's two ends, one point itself."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=float).reshape(-1, 2)})
    if len(pts) <= 2:
        return np.array(pts, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def outline_distance(p, outline) -> tuple[float, np.ndarray]:
    """How far 2-D point `p` is from a convex outline (`convex_hull`'s corners: one point, a segment or a polygon),
    and the outline's nearest point to it: 0 and `p` itself inside a polygon."""
    p = np.asarray(p, dtype=float)[:2]
    q = np.asarray(outline, dtype=float).reshape(-1, 2)
    if len(q) >= 3:
        edges = np.roll(q, -1, axis=0) - q
        if np.all(edges[:, 0] * (p[1] - q[:, 1]) - edges[:, 1] * (p[0] - q[:, 0]) >= 0.0):
            return 0.0, p.copy()
    best, near = np.inf, q[0]
    for a, b in zip(q, np.roll(q, -1, axis=0) if len(q) > 1 else q):
        ab = b - a
        u = 0.0 if not ab @ ab > 0.0 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        c = a + u * ab
        d = float(np.hypot(*(p - c)))
        if d < best:
            best, near = d, c
    return best, near


def object_outline(obj, t: float) -> np.ndarray:
    """The object's outline at time `t` (world x, y; (N, 2)): its footprint, or its centre for a point object, moved by
    its velocity."""
    pos, vel = obj
    shift = np.asarray(vel, dtype=float)[:2] * float(t)
    fp = getattr(obj, "footprint", None)
    base = np.asarray(pos, dtype=float)[:2].reshape(1, 2) if fp is None else fp
    return base + shift


def item_footprint(item):
    """The horizontal outline at t = 0 of a scene item that is an extended target (world x, y), or None: `points` with
    more than one position (their convex hull), a `hull` or `boat` (the plan of the simulator's hull, scene.hull_points:
    `length` along `heading_deg` about `origin`, half-breadth beam / 2 sqrt(1 - |2u - 1|^3)), a box target given by
    `x_range`, `y_range` (its rectangle). Anything else (a sphere, one point) is a point object."""
    kind = item.get("type")
    if "positions" in item and len(item["positions"]) > 1:
        return convex_hull(np.asarray(item["positions"], dtype=float)[:, :2])
    if kind in ("hull", "boat") and "origin" in item:
        length, beam = float(item["length"]), float(item["beam"])
        u = np.linspace(0.0, 1.0, 41)
        half = 0.5 * beam * np.sqrt(np.maximum(1.0 - np.abs(2.0 * u - 1.0) ** 3, 0.0))
        along, across = (u - 0.5) * length, half
        local = np.concatenate([np.stack([along, across], 1), np.stack([along, -across], 1)])
        h = np.radians(float(item.get("heading_deg", 0.0)))
        rot = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
        return convex_hull(local @ rot.T + np.asarray(item["origin"], dtype=float)[:2])
    if "x_range" in item:
        (x0, x1), (y0, y1) = item["x_range"], item["y_range"]
        return convex_hull([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return None


def scene_truth(cfg):
    """{label: TrueObject (centre at t = 0, velocity; with the item's footprint, `item_footprint`)} of a scenario's
    scene items."""
    truth = {}
    for item in cfg["scene"]:
        if "positions" in item:
            centre = np.asarray(item["positions"], dtype=float).mean(axis=0)
        elif "origin" in item:
            centre = np.asarray(item["origin"], dtype=float)
        elif "centre" in item:
            centre = np.asarray(item["centre"], dtype=float)
        elif "x_range" in item:
            if max(np.ptp(item["x_range"]), np.ptp(item["y_range"])) > 30.0:
                continue                                        # an area (seabed, waves, a boulder field), not a target
            centre = np.array([np.mean(item["x_range"]), np.mean(item["y_range"]), float(item.get("depth", 0.0))])
        else:
            continue
        truth[item["label"]] = TrueObject(centre, np.asarray(item.get("velocity", (0.0, 0.0, 0.0)), dtype=float),
                                          item_footprint(item))
    return truth


KIND_STYLE = {"mover": dict(color="#eb6834", lw=3.0, zorder=4), "static": dict(color="#2a78d6", lw=2.0, zorder=3),
              "line": dict(color="#1baf7a", lw=1.5, zorder=2),
              "multipath": dict(color="#eda100", lw=1.5, zorder=2), "uncertain": dict(color="0.72", lw=1.0, zorder=1)}


ESTABLISHED = ("mover", "static", "line")


def _miss(tr, obj):
    """How far (m) a track's state is from the true object `obj` (TrueObject, or a (world position at t = 0, velocity)
    pair: a point object) where the object was when the state holds (`state_time`): from its outline (`object_outline`;
    0 inside it) for an extended object, from its centre for a point object (D-157)."""
    return outline_distance((tr.state[1], tr.state[0]), object_outline(obj, state_time(tr)))[0]


def _centroid_miss(tr, obj):
    """`_miss` from the object's centre, whatever its extent (the score before D-157)."""
    pos, vel = obj
    at = np.asarray(pos, dtype=float)[:2] + np.asarray(vel, dtype=float)[:2] * state_time(tr)
    return float(np.hypot(tr.state[1] - at[0], tr.state[0] - at[1]))


def object_track(tracks, kinds, obj, gate):
    """The track scored for a true object `obj` (TrueObject; a (world position at t = 0, velocity) pair is a point
    object): the nearest track of an established kind (mover, static, line) within `gate`; if there is none, the
    nearest track of any kind within `gate`, so that an object seen only as uncertain or multipath is still reported;
    None if no track is that near. Each track is compared with the object where it was when the track's state holds
    (`state_time`: its last detection's time, D-156), by `_miss`: the distance from the object's outline for an
    extended object (D-157), ties (several tracks on it) broken by the distance from its centre. `kinds` maps track id
    to kind. Ours (D-155): with an object seen as a dozen arrivals (popoto_12km), the nearest track of any kind is a coin
    toss among them, often a four-point uncertain track or a multipath child."""
    def dist(tr):
        return (_miss(tr, obj), _centroid_miss(tr, obj))
    near = [tr for tr in tracks if dist(tr)[0] < gate]
    est = [tr for tr in near if kinds[tr.id] in ESTABLISHED]
    return min(est or near, key=dist, default=None)


def count_false_movers(classes, moving_ends, gate):
    """Tracks classed `mover` no closer than `gate` to every moving object at the last ping: `moving_ends` holds each
    moving object's outline there (`object_outline` at the last ping; a point object's is its position, one (x, y))."""
    def far(c, e):
        return outline_distance((c.position[1], c.position[0]), e)[0] >= gate
    return int(sum(c.kind == "mover" and all(far(c, e) for e in moving_ends) for c in classes))


# `--reverb KEY=VALUE` (opt-in, repeatable, sonar-4pon): a law put on the scenario's reverberation items before the run,
# to rerun a gallery scene under a law before a default moved: `doppler=centre` (every item: the reverberation Doppler
# before D-251) and `bistatic_rule=mean_angle` (the items whose law reads it, sim/simulator.py: the bistatic surface and
# bottom laws before D-216 A3). Nothing else is accepted.
REVERB_OPTIONS = {"doppler": ("centre", "per_frequency"), "bistatic_rule": ("geometric_mean", "mean_angle")}
BISTATIC_MODELS = ("chapman_harris", "kraus_surface", "searay_bottom", "bubble_layer_surface")


def apply_reverb_options(cfg, options):
    """Put each `KEY=VALUE` of `options` (REVERB_OPTIONS) on the scenario `cfg`'s reverberation items, in place:
    `doppler` on every item, `bistatic_rule` on those of BISTATIC_MODELS. Returns `cfg`; refuses anything else."""
    for opt in options:
        key, _, value = str(opt).partition("=")
        if value not in REVERB_OPTIONS.get(key, ()):
            raise ValueError(f"--reverb {opt!r}: one of " + ", ".join(f"{k}={v}" for k, vs in REVERB_OPTIONS.items() for v in vs))
        for item in cfg.get("reverberation") or []:
            if key == "doppler" or item.get("model", "lambert") in BISTATIC_MODELS:
                item[key] = value
    return cfg


# `--evidence` (opt-in, sonar-4pon): classify_and_score adds `classification.evidence`, one row per track: the classifier
# tap's numbers (classify.CLASS_EVIDENCE: the rule that gave the kind and the numbers it compared), the track's position
# and velocity in world (x, y) at the last ping, whether it is one of the false movers counted, and its nearest true
# object (label, distance from its outline at the last ping). Read only; off: the output as before.
EVIDENCE = False


def class_evidence(classes, evidence, truth, t_last, moving_ends, gate):
    """`classification.evidence` rows (EVIDENCE): `evidence` (classify_tracks' tap, one dict per class, in order) with
    each class's world position and velocity, its false-mover flag (count_false_movers' test) and nearest true object."""
    rows = []
    for c, e in zip(classes, evidence):
        xy = (float(c.position[1]), float(c.position[0]))
        near = min(((outline_distance(xy, object_outline(o, t_last))[0], label) for label, o in truth.items()), default=None)
        rows.append(dict(e, position_xy=list(xy), velocity_xy=[float(c.velocity[1]), float(c.velocity[0])],
                         false_mover=bool(c.kind == "mover" and all(outline_distance(xy, m)[0] >= gate for m in moving_ends)),
                         nearest=None if near is None else {"label": near[1], "distance_m": float(near[0])}))
    return rows


def classify_and_score(tracks, truth, t_last, gate, **kw):
    """Classify the tracks (gpusonar.classify) and score the result against the scene:
    the kind given to each true object's track (object_track) and how many tracks called
    movers are not on an object that moves (count_false_movers, against each moving object's outline at the last
    ping: D-157). The classes' positions are taken at the last ping (`t_last`), as the live report gives them (D-156).
    `range_doppler` ({"factor", "origin"}, target_doppler; None: off): the target's part of the LFM coupling correction
    (sonar-fyj), applied to the classes' positions before they are scored. With EVIDENCE, also `evidence`
    (class_evidence), after the shift."""
    rd = kw.pop("range_doppler", None)
    evidence = [] if EVIDENCE else None
    classes = classify.classify_tracks(tracks, time=t_last, evidence=evidence, **kw)
    if rd is not None:
        classify.shift_for_target_doppler(classes, float(rd["factor"]), tuple(rd["origin"]))
    by_id = {c.id: c for c in classes}
    out = {"counts": classify.summary(classes), "targets": {}}
    moving_ends = []
    for label, obj in truth.items():
        pos, vel = obj
        if np.hypot(*vel[:2]) > 0.0:
            moving_ends.append(object_outline(obj, t_last))
        best = object_track(tracks, {c.id: c.kind for c in classes}, obj, gate)
        if best is not None:
            c = by_id[best.id]
            out["targets"][label] = {"kind": c.kind, "speed_m_s": c.speed, "speed_sigma_m_s": c.speed_sigma,
                                     "truly_moving": bool(np.hypot(*vel[:2]) > 0.0)}
    out["movers_not_on_a_moving_object"] = count_false_movers(classes, moving_ends, gate)
    if evidence is not None:
        out["evidence"] = class_evidence(classes, evidence, truth, t_last, moving_ends, gate)
    return classes, out


def plot_tracks(ax, tracks, truth, own, t_last, title, classes=None):
    kinds = {c.id: c.kind for c in classes} if classes else {}
    for tr in tracks:
        h = np.array(tr.history)
        style = KIND_STYLE[kinds[tr.id]] if kinds else dict(lw=2.5)
        ax.plot(h[:, 1], h[:, 2], "-", **style)
        if not kinds or kinds[tr.id] in ("mover", "static"):
            ax.plot(h[-1, 1], h[-1, 2], "k.", ms=6)
    if kinds:
        from matplotlib.lines import Line2D
        counts = classify.summary(classes)
        ax.legend(handles=[Line2D([], [], label=f"{k} ({counts[k]})", **{a: v for a, v in KIND_STYLE[k].items() if a != "zorder"})
                           for k in KIND_STYLE if counts[k]], loc="lower left", fontsize=8, title="track class")
    for label, (pos, vel) in truth.items():
        end = pos[:2] + vel[:2] * t_last
        ax.plot([pos[1], end[1]], [pos[0], end[0]], "--", lw=1, color="0.35")
        ax.plot(pos[1], pos[0], "o", mfc="none", color="0.35", ms=9)
        ax.annotate(label, (end[1], end[0]), textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.plot(own[:, 1], own[:, 0], "-", color="C3", lw=2)
    ax.annotate("ownship", (own[-1, 1], own[-1, 0]), textcoords="offset points", xytext=(7, 5), fontsize=8, color="C3")
    ax.set_xlabel("y, to starboard (m)")
    ax.set_ylabel("x, ahead (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")


FLS_TRANSIT = {
    "fls_120k_transit": dict(r=(20.0, 298.0, 0.06), cell=8, guard_m=6.0, train_m=30.0),
    "fls_170k_transit": dict(r=(30.0, 446.0, 0.06), cell=8, guard_m=6.0, train_m=30.0),
    "fls_330k_transit": dict(r=(20.0, 298.0, 0.04), cell=12, guard_m=6.0, train_m=30.0),
    "fls_120k_cluttered_seabed": dict(r=(20.0, 298.0, 0.06), cell=8, guard_m=6.0, train_m=30.0),
    "fls_120k_rough_surface": dict(r=(20.0, 298.0, 0.06), cell=8, guard_m=6.0, train_m=30.0),
    "fls_170k_traffic": dict(r=(30.0, 446.0, 0.06), cell=8, guard_m=6.0, train_m=30.0),
}


# The transit scenes' tracker (`world_tracks`), and what its detections are merged by before it sees them.
FLS_TRACKER = dict(range_sigma=1.5, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0, process_noise=0.3)
# Merging is OFF. It was the fix proposed for the anchored boat's track swap (D-121) and it does cure that: the thinned
# `fls_170k_traffic` reads 0.29 m/s for the boat with it, 1.92 m/s without. But measured on the six full transit scenes
# (CPU, one simulation each, detections and tracks both ways) it is not a gain overall: flagged cells within 1.9 m in
# range and 0.73 degrees chain together through anything dense. `fls_120k_rough_surface`: the boat under way joins its
# wake and the waves, its speed reads 8.1 m/s (4.7 without; true 5.0) and it is no longer classed a mover.
# `fls_120k_cluttered_seabed`: 60 detections a ping become 32, the pipeline's 16 `line` tracks become none, movers that
# are not on a moving object go from 4 to 7. The thinned `fls_120k_transit` reads 1.6 m/s for the pinnacle (limit 1.2).
# The long-range chain merges because its targets are smaller than what its tracker resolves; these scenes image
# objects larger than that, and their highlights are what the classes are told apart by. Set True to repeat the comparison.
FLS_MERGE = False


def fls_merge(record, n_cells, samples_per_cell, step, bearings, guard, train):
    """(merge_cells, merge_u) for the transit scenes' detections: the live chain's own rule (DECISIONS D-95) from its
    own plan (`core.live_detector_plan`, libs/live detector.hpp) for this array and map: flagged cells within the
    tracker's range sigma in range and half the beam's half-power width in bearing are one detection. A hull whose
    highlights make two clusters a few metres apart is then one detection on every ping, not two on some and one on
    others: two tracks on one hull swap clusters when a ping joins them, and read a speed (the anchored boat of
    `fls_170k_traffic`: 4.2 m in 2.25 s, 1.9 m/s)."""
    plan = gs.core.live_detector_plan(record, int(n_cells), int(samples_per_cell), float(step), [float(b) for b in bearings], 1, int(guard), int(train), True,
                                      {"merge_m": float(FLS_TRACKER["range_sigma"])})
    return int(plan["merge_cells"]), float(plan["merge_u"])


def run_fls_transit(name, cfg, backend, quick):
    p = dict(FLS_TRANSIT[name])
    if quick:                                             # half the range sampling, same cell size
        p["r"] = (p["r"][0], p["r"][1], 2 * p["r"][2])
        p["cell"] = p["cell"] // 2
    result = run_scenario(cfg, backend=backend)
    ranges = np.arange(*p["r"])
    ranges = ranges[:len(ranges) // p["cell"] * p["cell"]]
    bearings = np.radians(np.arange(-60.0, 60.01, 0.4))
    cell_m = p["cell"] * p["r"][2]
    guard, train = int(round(p["guard_m"] / cell_m)), int(round(p["train_m"] / cell_m))
    per_ping, first, polar = [], None, []
    merge = fls_merge(result.records[0], len(ranges) // p["cell"], p["cell"], p["r"][2], bearings, guard, train) if FLS_MERGE else None
    for rec in result.records:
        origin = rec.rx_positions().mean(axis=0)
        img = recon.fan_image([rec], ranges, bearings, interp="linear", window="hann", backend=backend)
        fine = np.abs(img.image) ** 2
        # Power in range cells of about half a metre; rows are bearings for the detector.
        power = fine.reshape(-1, p["cell"], fine.shape[1]).mean(axis=1).T
        r_cells = ranges.reshape(-1, p["cell"]).mean(axis=1)
        hits = detect.cluster(detect.cfar(power, pfa=1e-7, guard=guard, train=train), power, r_cells, bearings,
                              guard=guard, train=train, merge=merge)
        hits = [h for h in hits if h.snr_db > 13.0 and h.cells >= 2]
        for h in hits:
            h.range = doppler_range(rec, h.range, (np.cos(h.bearing), np.sin(h.bearing)))     # sonar-fyj, opt-in
        per_ping.append([(origin[0] + h.range * np.cos(h.bearing), origin[1] + h.range * np.sin(h.bearing), h.snr_db,
                          echo_delay(rec, h.range)) for h in hits])
        if first is None:
            first = (r_cells, power, origin)
        polar.append((display(power.T * r_cells[:, None] ** 3), origin))
    tracks = world_tracks(per_ping, result.times, **FLS_TRACKER)
    truth = scene_truth(cfg)
    t_last = float(result.times[-1])
    metrics = {"pings": len(result.records), "scatterers": len(result.scene),
               "detections_per_ping": [len(h) for h in per_ping], "confirmed_tracks": len(tracks)}
    # A mover's fitted speed must be five times its uncertainty; positions must fit a line to 2.5 m.
    # A track counts in a mover's row only if its mean SNR is at most 6 dB below the mover's (D-154, owner's
    # ruling after the ten-seed evaluation: no target lost, the cluttered boat a mover on every seed).
    own_last = result.records[-1].rx_positions().mean(axis=0)
    classes, metrics["classification"] = classify_and_score(
        tracks, truth, t_last, 15.0, min_speed=0.5, max_fit_rms=2.5,
        linear={"length": 40.0, "width": 2.0}, detections={"row_snr_db": 6.0},
        range_doppler=target_doppler(result.records[-1], (own_last[1], own_last[0])))
    metrics["targets"] = score_tracks(tracks, truth, t_last, 15.0, kinds={c.id: c.kind for c in classes})
    for label, m in metrics["targets"].items():
        if m.get("tracked"):
            m["track_speed_m_s"] = float(np.hypot(*m["track_velocity_xy"]))
    r_cells, power, origin = first
    R, B = np.meshgrid(r_cells, bearings, indexing="ij")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    m = axes[0].pcolormesh(origin[1] + R * np.sin(B), origin[0] + R * np.cos(B), display(power.T * r_cells[:, None] ** 3),
                           cmap="gray", vmin=0.0, vmax=1.0, shading="auto", rasterized=True)
    for label, (pos, _) in truth.items():
        axes[0].plot(pos[1], pos[0], "o", mfc="none", ms=13, label=label)
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("y, to starboard (m)")
    axes[0].set_ylabel("x, ahead (m)")
    axes[0].set_title("First ping (circles: true positions)")
    fig.colorbar(m, ax=axes[0], label="display level (range gain, dynamic range compression)", shrink=0.75)
    plot_tracks(axes[1], tracks, truth, result.true_positions, t_last,
                f"World-frame tracks over {len(result.records)} pings (dashed: truth)", classes)
    fig.suptitle(f"{name}: {cfg['sonar']}, ownship under way; seabed and surface reverberation, noise (simulated)")
    # World-fixed frames: ownship moves up the picture, static things stay put.
    r_far = r_cells[-1]
    gy, gx = np.meshgrid(np.linspace(-0.87 * r_far, 0.87 * r_far, 440), np.linspace(0.0, r_far + 25.0, 300))
    frames, overlays = [], []
    for i, (disp, own) in enumerate(polar):
        frames.append(polar_to_grid(disp, r_cells, bearings, own[:2], 0.0, gx, gy))
        marks = [([own[1]], [own[0]], dict(marker="^", color="#eb6834", ms=9, ls="none"))]
        for label, (pos, vel) in truth.items():
            q = pos[:2] + vel[:2] * float(result.times[i])
            marks.append(([q[1]], [q[0]], dict(marker="o", mfc="none", mec="#2a78d6", ms=12, ls="none")))
        overlays.append(marks)
    anim = dict(frames=frames, titles=[f"{name}: ping {i + 1} of {len(frames)}, t = {result.times[i]:.2f} s (circles: truth, triangle: ownship)" for i in range(len(frames))],
                extent=[gy[0, 0], gy[0, -1], gx[0, 0], gx[-1, 0]], xlabel="y, to starboard (m)", ylabel="x, ahead (m)",
                vmin=0.0, vmax=1.0, overlays=overlays, figsize=(7.4, 5.4))
    return fig, metrics, anim


SAS_SURVEY = {
    # `zoom`: (label, side in metres) of the full-resolution crops under the leg images.
    "sas_minsas_survey": dict(x=(0.0, 40.0), y=(38.5, 62.0), what="two survey legs over a wreck, cable, rocks and calibration cube", wreck=True,
                              zoom=[("calibration_cube", 3.0), ("rocks", 8.0), ("wreck_deckhouse", 8.0)]),
    "sas_minsas_boulder_field": dict(x=(1.0, 31.0), y=(39.0, 64.0), what="a boulder field on sand", wreck=False,
                                     zoom=[("boulders", 10.0), ("boulders", 4.0), ("boulders", 1.5)]),
    "sas_minsas_rocky_ripples": dict(x=(1.0, 31.0), y=(39.0, 64.0), what="rippled sand, a rock outcrop, stones and a cable", wreck=False,
                                     zoom=[("outcrop", 9.0), ("stones", 4.0), ("cable", 3.0)]),
    # `speckle`: a box of bare seabed (x0, x1, y0, y1) for the speckle statistics; `ghost`: objects whose
    # along-track ambiguity copy (offset in metres) falls on bare seabed inside the image.
    "sas_minsas_speckle": dict(x=(0.0, 13.0), y=(47.0, 53.0), what="a seabed with developed speckle, cube, cable, rocks and a pipe",
                               wreck=False, zoom=[("calibration_cube", 2.0), ("rocks", 3.0), ("pipe", 3.0)],
                               speckle=(2.8, 4.2, 47.3, 52.7), ghost=(["calibration_cube", "cable"], 10.3)),
}


def run_sas_survey(name, cfg, backend, quick):
    p = SAS_SURVEY[name]
    result = run_scenario(cfg, backend=backend)
    # Hamming-weighted pulse compression: range sidelobes drop from -15 dB to -41 dB beyond 10 cm, so
    # bright points stop smearing across the picture; the ground-range width goes from 1.8 cm to 2.6 cm
    # (tests/python/test_minsas_resolution.py measures both).
    recon.prepare(result.records, window="hamming", backend=backend)
    pixel = 0.08 if quick else 0.02                        # 2 cm: the image resolves 2.0 cm x 2.6 cm
    x, y = np.arange(*p["x"], pixel), np.arange(*p["y"], pixel)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    def to_pixel(pts):
        i = np.rint((pts[:, 0] - x[0]) / pixel).astype(int)
        j = np.rint((pts[:, 1] - y[0]) / pixel).astype(int)
        return [(a, b) if 0 <= a < len(x) and 0 <= b < len(y) else None for a, b in zip(i, j)]

    legs = sorted(set(result.legs))
    metrics = {"pings": len(result.records), "scatterers": len(result.scene), "pixel_m": pixel, "legs": {}}
    nz = len(p["zoom"])
    fig = plt.figure(figsize=(max(8.0 * len(legs), 13.0), 13.0), constrained_layout=True)
    gs_ = fig.add_gridspec(2, len(legs) * nz, height_ratios=[1.75, 1.0])
    axes = [fig.add_subplot(gs_[0, k * nz:(k + 1) * nz]) for k in range(len(legs))]
    zoom_axes = [fig.add_subplot(gs_[1, k * len(legs):(k + 1) * len(legs)]) for k in range(nz)]
    full_res = None
    # Strips of seabed just north and just south of the wreck: each leg shadows one of them.
    north = (X > 17.0) & (X < 24.0) & (Y > 53.0) & (Y < 55.5)
    south = (X > 17.0) & (X < 24.0) & (Y > 44.5) & (Y < 47.0)
    anim = None
    for ax, leg in zip(axes, legs):
        recs = [r for r, k in zip(result.records, result.legs) if k == leg]
        # 1.6 beamwidths of aperture weighted by the beam patterns: the same 2.0 cm along track as the
        # default, with the nearest along-track sidelobes at -36 dB instead of -23 dB (measured on a point).
        power = np.abs(recon.sas_image(recs, points, interp="linear", fov_factor=1.6, directivity_weight=True, replica_window="hamming",
                                       backend=backend)) ** 2
        covered = power > 0                                   # pixels this leg's beams reached
        median = float(np.median(power[covered]))
        side = float(np.sign(np.mean([r.rx_positions().mean(axis=0)[1] for r in recs]) - 50.0))
        m_leg = {"sonar_side": "y below" if side < 0 else "y above", "covered_fraction": float(covered.mean()),
                 "object_over_median_db": object_levels(result.scene, power, to_pixel, median)}
        if "speckle" in p:
            x0, x1, y0, y1 = p["speckle"]
            bare = power[(X > x0) & (X < x1) & (Y > y0) & (Y < y1)]
            # Developed speckle: intensity is exponential, so its standard deviation equals its mean.
            m_leg["speckle_contrast"] = float(bare.std() / bare.mean())
            m_leg["bare_seabed_mean_over_median_db"] = float(db(bare.mean(), median))
            masks_ = result.scene.label_masks()
            labels, offset = p["ghost"]
            for label in labels:
                # Where the copy of this object falls: the same pixels moved along the track.
                pts_g = result.scene.positions[masks_[label]] + np.array([offset, 0.0, 0.0])
                pix = [q for q in to_pixel(pts_g) if q is not None]
                here = np.mean([power[q] for q in pix])
                obj = np.mean([power[q] for q in to_pixel(result.scene.positions[masks_[label]]) if q is not None])
                m_leg.setdefault("ghost", {})[label] = {
                    "object_over_bare_seabed_db": float(db(obj, bare.mean())),
                    "at_copy_over_bare_seabed_db": float(db(here, bare.mean()))}
        if p["wreck"]:
            m_leg["far_strip_over_near_strip_db"] = float(db(power[north & covered].mean(), power[south & covered].mean())
                                                          * (1.0 if side < 0 else -1.0))
        metrics["legs"][f"leg_{leg + 1}"] = m_leg
        m = ax.imshow(db(power, median), extent=[y[0], y[-1], x[-1], x[0]], cmap="gray", vmin=-12, vmax=30,
                      aspect="equal", interpolation="nearest")
        ax.invert_yaxis()
        heading = cfg["platform"]["legs"][leg]["heading_deg"]
        ax.set_title(f"Leg {leg + 1}: heading {heading:.0f} deg, sonar at y = {cfg['platform']['legs'][leg]['start'][1]:.0f} m")
        ax.set_xlabel("y (m)")
        if full_res is None:
            # Crops of leg 1 at the pixel the image was formed on: nothing is resampled for display.
            full_res = np.clip(db(power, median), -12, 30)
            masks = result.scene.label_masks()
            for za, (label, side_m) in zip(zoom_axes, p["zoom"]):
                pts_l = result.scene.positions[masks[label]]
                # Centre on the object nearest the label's centroid (one rock of a field, not the gap between rocks).
                c = pts_l[np.argmin(np.linalg.norm(pts_l[:, :2] - pts_l[:, :2].mean(axis=0), axis=1)), :2]
                i0, j0 = int((c[0] - side_m / 2 - x[0]) / pixel), int((c[1] - side_m / 2 - y[0]) / pixel)
                n_px = int(side_m / pixel)
                i0, j0 = int(np.clip(i0, 0, len(x) - n_px)), int(np.clip(j0, 0, len(y) - n_px))
                crop = full_res[i0:i0 + n_px, j0:j0 + n_px]
                za.imshow(crop, extent=[y[j0], y[j0 + n_px - 1], x[i0 + n_px - 1], x[i0]], cmap="gray", vmin=-12, vmax=30,
                          aspect="equal", interpolation="nearest")
                za.invert_yaxis()
                za.set_title(f"Leg 1, {label}: {side_m:g} m x {side_m:g} m at {pixel * 100:g} cm pixels", fontsize=9)
                za.set_xlabel("y (m)")
                ax.add_patch(plt.Rectangle((y[j0], x[i0]), side_m, side_m, fill=False, ec="#eb6834", lw=1.2))
            zoom_axes[0].set_ylabel("x (m)")
        if anim is None:
            # The synthetic aperture building up: the image from the first pings of the leg.
            step = max(int(round(0.08 / pixel)), 1)
            coarse = points[::step, ::step]
            frames, titles = [], []
            for k in np.linspace(len(recs) / 8, len(recs), 8).astype(int):
                part = np.abs(recon.sas_image(recs[:k], coarse, interp="linear", fov_factor=1.6, directivity_weight=True, replica_window="hamming",
                                              backend=backend)) ** 2
                frames.append(shrink(np.clip(db(part, median), -12, 30)))
                titles.append(f"{name}: leg 1, first {k} of {len(recs)} pings")
            anim = dict(frames=frames, titles=titles, extent=[y[0], y[-1], x[0], x[-1]], xlabel="y (m)", ylabel="x (m)",
                        vmin=-12, vmax=30, figsize=(6.0, 6.4))
    axes[0].set_ylabel("x (m)")
    fig.colorbar(m, ax=axes, label="dB re image median", shrink=0.8)
    fig.suptitle(f"{name}: {cfg['sonar']}, {p['what']} (simulated); boxes mark the crops below")
    anim["full_res"] = (full_res, f"{name}_leg1_full.png")      # the whole of leg 1, one image pixel per file pixel
    return fig, metrics, anim


# The Popoto runner's bearing estimate (sonar-4s2): "centroid" (detect.cluster's power-weighted mean over the flagged
# cells; the default, as before) or "parabola" (`subgrid_bearings`). Opt-in.
POPOTO_BEARING = "centroid"
POPOTO_BEARINGS = ("centroid", "parabola")


def parabola_vertex(left, mid, right):
    """Offset (in grid steps, -0.5 to 0.5) of the vertex of the parabola through (-1, left), (0, mid), (1, right), where
    `mid` is a maximum of the three; 0 when they do not curve down. For y = a (u - p)^2 + b sampled at u = -1, 0, 1:
    left - right = 4 a p and left - 2 mid + right = 2 a, so p = (left - right) / (2 (left - 2 mid + right)) (ours,
    algebra). Exact when the samples lie on a parabola; applied to the beam's power in dB, it is exact for a Gaussian
    main lobe, near enough for the Hann beam's top when the grid step is a tenth of the beam (the clip at +-0.5 keeps
    the estimate within the strongest row's cell)."""
    den = left - 2.0 * mid + right
    if not den < 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / den, -0.5, 0.5))


def subgrid_bearings(flags, power, bearings, hits):
    """Each hit of `detect.cluster(flags, power, ranges, bearings)` (no merge) at a sub-grid bearing (sonar-4s2, ours):
    the bearing profile is the power summed over the hit's range cells at every bearing, flagged or not; its strongest
    flagged row and the rows either side give the vertex of a parabola through the profile in dB (`parabola_vertex`).
    At a map edge the hit keeps its centroid. Why: a hit of few cells holds one or two bearing rows, and the
    power-weighted mean over its flagged cells is then the grid's own bearing or the midpoint of two (0.5 deg grid,
    up to 0.25 deg = 52 m at 12 km); at low SNR the flagged cells are an irregular, noise-cut part of the beam whose
    mean is pulled further. Measured on the six Popoto scenes (tools/popoto_bearing_study.py, 10 seeds each, hits
    near the ship): RMS from its centre at 9-11.5 dB 0.45 -> 0.30 deg at 12 km, 1.02 -> 0.57 deg in winter_sand_200m;
    above 17 dB 0.13 -> 0.12 and 0.22 -> 0.26 deg (gravel_30m). The unflagged neighbours still carry the beam's shape."""
    from scipy import ndimage
    labels, n = ndimage.label(np.asarray(flags), structure=np.ones((3, 3), dtype=int))
    if n != len(hits):
        raise ValueError(f"subgrid_bearings: {n} groups of flagged cells for {len(hits)} hits")
    power = np.asarray(power, dtype=float)
    b = np.asarray(bearings, dtype=float)
    out = []
    for k, h in enumerate(hits, start=1):
        ib, ir = np.nonzero(labels == k)
        prof = power[:, np.unique(ir)].sum(axis=1)
        rows = np.unique(ib)
        j = int(rows[np.argmax(prof[rows])])
        if j == 0 or j == len(b) - 1 or not np.all(prof[j - 1:j + 2] > 0.0):
            out.append(float(h.bearing))
            continue
        y = 10.0 * np.log10(prof[j - 1:j + 2])
        out.append(float(b[j] + parabola_vertex(*y) * (b[j + 1] - b[j])))
    return out


def run_popoto_far(name, cfg, backend, quick):
    from gpusonar import sync
    result = run_scenario(cfg, backend=backend)
    bearings = np.radians(np.arange(-40.0, 40.01, 0.5))
    per_ping, first = [], None
    maps, owns = [], []
    n_total = result.records[0].samples.n_channels
    scaling = {}
    for i, rec in enumerate(result.records):
        origin = rec.rx_positions().mean(axis=0)
        if i == 0:
            # The same ping through arrays of fewer hydrophones (the middle ones).
            target = result.scene.positions.mean(axis=0)
            for n in sorted({8, 16, 32, n_total} & set(range(1, n_total + 1))):
                first_row = (n_total - n) // 2
                sub = sync.select_channels(rec, range(first_row, first_row + n))
                bm = flank.farfield_beamform(sub, bearings, window="hann").valid()
                pw = np.abs(bm.data) ** 2
                rng_t = float(np.hypot(target[0] - origin[0], target[1] - origin[1]))
                near = np.abs(bm.ranges - rng_t) < 120.0
                scaling[str(n)] = float(db(pw[:, near].max(), np.median(pw)))
        bm = flank.farfield_beamform(rec, bearings, window="hann").valid()
        fine = np.abs(bm.data) ** 2
        k = max(int(round(2.0 / (bm.ranges[1] - bm.ranges[0]))), 1)        # 2 m range cells
        n = fine.shape[1] // k * k
        power = fine[:, :n].reshape(len(bearings), -1, k).mean(axis=2)
        ranges = bm.ranges[:n].reshape(-1, k).mean(axis=1)
        flags = detect.cfar(power, pfa=1e-5, guard=50, train=150)
        hits = detect.cluster(flags, power, ranges, bearings, guard=50, train=150)
        if POPOTO_BEARING == "parabola":
            for h, b in zip(hits, subgrid_bearings(flags, power, bearings, hits)):
                h.bearing = b
        elif POPOTO_BEARING != "centroid":
            raise ValueError(f"POPOTO_BEARING = {POPOTO_BEARING!r}: one of {POPOTO_BEARINGS}")
        hits = set_range_estimate([h for h in hits if h.snr_db > 9.0])
        for h in hits:
            h.range = doppler_range(rec, h.range, (np.sin(h.bearing), np.cos(h.bearing)))     # sonar-fyj, opt-in
        # Bearing is from broadside (+y) towards +x, the array's own axis; placed from where the array is when the echo
        # arrives (echo_position; sonar-960: from the array at transmit the ship's echoes lay 24 m behind it).
        v_rx, c = rec.geometry.rx_pose.velocity, rec.environment.sound_speed
        per_ping.append([(*echo_position(origin, v_rx, h.range, h.bearing, c), h.snr_db, echo_delay(rec, h.range))
                         for h in hits])
        if first is None:
            first = (ranges, power)
        maps.append(shrink(np.clip(db(power, np.median(power)), 0, 30)))
        owns.append(origin)
    tracks = world_tracks(per_ping, result.times, range_sigma=15.0, bearing_sigma=np.radians(1.0), confirm_hits=3,
                          max_misses=2, initial_speed_sigma=8.0, process_noise=0.05)
    truth = scene_truth(cfg)
    t_last = float(result.times[-1])
    metrics = {"pings": len(result.records), "hydrophones": int(n_total),
               "detections_per_ping": [len(h) for h in per_ping], "confirmed_tracks": len(tracks),
               "peak_over_median_db_by_hydrophone_count": scaling}
    own_last = result.true_positions[-1]
    classes, metrics["classification"] = classify_and_score(
        tracks, truth, t_last, 200.0, min_speed=1.0, static_speed=1.0, max_fit_rms=60.0,
        multipath={"max_excess_range": 800.0, "bearing": np.radians(3.0), "velocity": 2.5, "origin": (own_last[1], own_last[0])},
        range_doppler=target_doppler(result.records[-1], (own_last[1], own_last[0])))
    metrics["targets"] = score_tracks(tracks, truth, t_last, 200.0, kinds={c.id: c.kind for c in classes})
    ranges, power = first
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    m = axes[0].imshow(db(power, np.median(power)), extent=[ranges[0], ranges[-1], np.degrees(bearings[-1]), np.degrees(bearings[0])],
                       cmap="gray", vmin=0, vmax=30, aspect="auto")
    axes[0].invert_yaxis()
    for label, (pos, _) in truth.items():
        axes[0].plot(np.hypot(pos[0], pos[1]), np.degrees(np.arctan2(pos[0], pos[1])), "o", mfc="none", ms=14, label=label)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_xlabel("range (m)")
    axes[0].set_ylabel("bearing from broadside (deg)")
    axes[0].set_title(f"First ping, {n_total} hydrophones")
    fig.colorbar(m, ax=axes[0], label="dB re map median", shrink=0.8)
    plot_tracks(axes[1], tracks, truth, result.true_positions, t_last,
                f"World-frame tracks over {len(result.records)} pings (dashed: truth)", classes)
    rng_km = np.hypot(*next(iter(truth.values()))[0][:2]) / 1e3
    fig.suptitle(f"{name}: Popoto-board array, ownship under way, ship-sized mover at {rng_km:.0f} km; multipath, reverberation, noise (simulated)")
    overlays = []
    for i, own in enumerate(owns):
        marks = []
        for label, (pos, vel) in truth.items():
            q = pos[:2] + vel[:2] * float(result.times[i]) - own[:2]
            marks.append(([np.hypot(q[0], q[1])], [np.degrees(np.arctan2(q[0], q[1]))],
                          dict(marker="o", mfc="none", mec="#eb6834", ms=13, ls="none")))
        overlays.append(marks)
    anim = dict(frames=maps, titles=[f"{name}: ping {i + 1} of {len(maps)}, t = {result.times[i]:.0f} s (circle: truth)" for i in range(len(maps))],
                extent=[ranges[0], ranges[-1], np.degrees(bearings[0]), np.degrees(bearings[-1])], aspect="auto",
                xlabel="range (m)", ylabel="bearing from broadside (deg)", vmin=0, vmax=30, overlays=overlays,
                figsize=(7.0, 4.6))
    return fig, metrics, anim


RUNNERS = {
    "pool_fls_l12_5d": run_fls, "pool_fls_p4_2v": run_fls, "seafloor_objects_fls": run_fls,
    "harbour_fls_750k": run_fls, "fls_120k_transit": run_fls_transit, "fls_170k_transit": run_fls_transit,
    "fls_330k_transit": run_fls_transit, "fls_120k_cluttered_seabed": run_fls_transit,
    "fls_120k_rough_surface": run_fls_transit, "fls_170k_traffic": run_fls_transit,
    "wreck_sas": run_sas, "minefield_sas": run_sas,
    "sas_minsas_survey": run_sas_survey, "sas_minsas_boulder_field": run_sas_survey,
    "sas_minsas_rocky_ripples": run_sas_survey, "sas_minsas_speckle": run_sas_survey, "rail_objects": run_rail, "buried_objects": run_buried,
    "flank_patrol": run_flank, "popoto_far_mover": run_popoto_far, "popoto_12km": run_popoto_far,
    "popoto_env_summer_mud_60m": run_popoto_far, "popoto_env_winter_sand_200m": run_popoto_far,
    "popoto_env_gravel_30m": run_popoto_far, "popoto_env_deep_silt_1000m": run_popoto_far,
    "popoto_trial_1000m": run_popoto,
}
NEEDS_BELLHOP = {n for n in RUNNERS if n.startswith(("flank_", "popoto_"))}

# Thinned versions for a CPU: fewer scatterers and pings, same geometry.
QUICK = {
    "pool_fls_l12_5d": lambda c: _scale_density(c, 0.1),
    "pool_fls_p4_2v": lambda c: _scale_density(c, 0.1),
    "harbour_fls_750k": lambda c: _scale_density(c, 0.25),
    "wreck_sas": lambda c: (_scale_density(c, 0.3), c["pings"].update(count=40)),
    "minefield_sas": lambda c: (_scale_density(c, 0.3), c["pings"].update(count=40)),
    "rail_objects": lambda c: (_scale_density(c, 0.1), c["pings"].update(count=51)),
    "buried_objects": lambda c: c["platform"].update(raster={"x": [-0.9, 0.9001, 0.15], "y": [-0.9, 0.9001, 0.15]}),
    "fls_120k_transit": lambda c: _thin_transit(c),
    "fls_170k_transit": lambda c: _thin_transit(c),
    "fls_330k_transit": lambda c: _thin_transit(c),
    "sas_minsas_survey": lambda c: (_scale_density(c, 0.5), c["platform"]["legs"][0].update(pings=40, start=[9.0, 0.0, -12.0]),
                                    c["platform"]["legs"][1].update(pings=40, start=[31.0, 100.0, -12.0])),
    "popoto_far_mover": lambda c: (c["pings"].update(count=4), c["propagation"].update(max_arrivals=12)),
    "popoto_12km": lambda c: (c["pings"].update(count=4), c["propagation"].update(max_arrivals=16)),
    "popoto_env_summer_mud_60m": lambda c: (c["pings"].update(count=4), c["propagation"].update(max_arrivals=16)),
    "popoto_env_winter_sand_200m": lambda c: (c["pings"].update(count=4), c["propagation"].update(max_arrivals=16)),
    "popoto_env_gravel_30m": lambda c: (c["pings"].update(count=4), c["propagation"].update(max_arrivals=16)),
    "popoto_env_deep_silt_1000m": lambda c: (c["pings"].update(count=4), c["propagation"].update(max_arrivals=16)),
    "fls_120k_cluttered_seabed": lambda c: _thin_transit(c),
    "fls_120k_rough_surface": lambda c: _thin_transit(c),
    "fls_170k_traffic": lambda c: _thin_transit(c),
    "sas_minsas_boulder_field": lambda c: (_scale_density(c, 0.5), c["platform"]["legs"][0].update(pings=40, start=[4.0, 0.0, -12.0])),
    "sas_minsas_speckle": lambda c: _scale_density(c, 0.02),
    "sas_minsas_rocky_ripples": lambda c: (_scale_density(c, 0.5), c["platform"]["legs"][0].update(pings=40, start=[4.0, 0.0, -12.0])),
    "flank_patrol": lambda c: (c["pings"].update(count=4), c["propagation"].update(max_arrivals=8),
                               c["sonar_overrides"].update(transmit={"waveform": {"duration": 0.2}})),
}


def _thin_transit(cfg):
    cfg["pings"].update(count=4)
    for item in cfg["scene"]:
        if "count" in item and item["type"] in ("hull", "boat"):
            item["count"] = int(item["count"] * 0.4)
    _scale_density(cfg, 0.5)


def _scale_density(cfg, factor):
    for item in cfg["scene"]:
        if "density" in item:
            item["density"] = float(item["density"]) * factor


QUICK_OUT = Path(__file__).resolve().parents[1] / "build" / "gallery_quick"


def output_dir(out: str | None, quick: bool) -> Path:
    """--out as given; else docs/figures for a full run and QUICK_OUT (not committed) for --quick."""
    return Path(out) if out is not None else (QUICK_OUT if quick else Path("docs/figures"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="cuda" if gs.cuda_available() else "host")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=None, help=f"output directory (default: docs/figures; with --quick: {QUICK_OUT})")
    ap.add_argument("--quick", action="store_true", help="thinned scenes and fewer pings, for a CPU")
    ap.add_argument("--no-animation", action="store_true", help="figures and numbers only")
    ap.add_argument("--range-estimate", choices=RANGE_ESTIMATES, default="first",
                    help="detection range in flank_patrol and the Popoto tracking scenes (RANGE_ESTIMATE; default %(default)s)")
    ap.add_argument("--range-doppler", choices=RANGE_DOPPLERS, default="off",
                    help="LFM range-Doppler coupling correction in the tracking scenes (RANGE_DOPPLER, sonar-fyj; default %(default)s)")
    ap.add_argument("--evidence", action="store_true",
                    help="add each track's classifier evidence to the tracking scenes' classification (EVIDENCE, sonar-4pon)")
    ap.add_argument("--reverb", action="append", default=[], metavar="KEY=VALUE",
                    help="a law on every reverberation item that reads it: doppler=centre (before D-251), "
                         "bistatic_rule=mean_angle (before D-216 A3); repeatable (REVERB_OPTIONS, sonar-4pon)")
    args = ap.parse_args()
    global RANGE_ESTIMATE, RANGE_DOPPLER, EVIDENCE
    RANGE_ESTIMATE = args.range_estimate
    RANGE_DOPPLER = args.range_doppler
    EVIDENCE = args.evidence
    backend = core.Backend.Cuda if args.backend == "cuda" else core.Backend.Host
    out = output_dir(args.out, args.quick)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.only.split(",") if n] or list(RUNNERS)
    summary_path = out / "gallery.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in names:
        if name in NEEDS_BELLHOP and find_bellhop() is None:
            print(f"[{name}] skipped: Bellhop not installed (GPUSONAR_BELLHOP)")
            continue
        cfg = load_scenario_config(name)
        if args.quick and name in QUICK:
            QUICK[name](cfg)
        apply_reverb_options(cfg, args.reverb)
        t = time.perf_counter()
        out_ = RUNNERS[name](name, cfg, backend, args.quick)
        fig, metrics = out_[0], out_[1]
        metrics["seconds"] = round(time.perf_counter() - t, 1)
        if len(out_) > 2 and out_[2] is not None and "full_res" in out_[2]:
            image, fname = out_[2].pop("full_res")
            if not args.quick:
                from PIL import Image                          # 8-bit greyscale: a third the size of an RGBA file
                Image.fromarray(np.uint8(np.clip((image[::-1] + 12.0) / 42.0, 0.0, 1.0) * 255.0), mode="L").save(
                    out / fname, optimize=True)
                metrics["full_resolution_image"] = fname
        if len(out_) > 2 and out_[2] is not None and not args.no_animation:
            write_animation(out / f"{name}.gif", out_[2])
            metrics["animation"] = f"{name}.gif"
        metrics["backend"] = args.backend
        metrics["quick"] = bool(args.quick)
        if RUNNERS[name] in (run_flank, run_popoto_far):
            metrics["range_estimate"] = args.range_estimate                # a gallery.json without it: centroid
        if args.range_doppler != "off" and RUNNERS[name] in (run_flank, run_popoto_far, run_fls_transit):
            metrics["range_doppler"] = args.range_doppler                  # opt-in: said only when on
        if args.reverb:
            metrics["reverb"] = list(args.reverb)                          # opt-in: said only when given
        fig.savefig(out / f"{name}.png", dpi=110)
        plt.close(fig)
        summary[name] = metrics
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[{name}] {json.dumps(metrics)}")
    print("GALLERY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
