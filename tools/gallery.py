#!/usr/bin/env python3
"""Run the realistic scenario of every mode end to end and save a picture and a few
numbers for each: simulate, condition, (estimate motion,) image, (detect, track).

  FLS    pool_fls_l12_5d, pool_fls_p4_2v, seafloor_objects_fls, harbour_fls_750k
  SAS    wreck_sas, minefield_sas (with motion estimation), rail_objects
  3D     buried_objects
  flank  flank_patrol (needs Bellhop), popoto_trial_1000m (needs Bellhop)

Usage: tools/gallery.py [--backend cuda|host] [--only name,name] [--out docs/figures]
                        [--quick]   thinned scenes and fewer pings, for a CPU

Writes <out>/<scenario>.png and <out>/gallery.json. The object-over-background figures
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
from gpusonar import detect, flank, motion, recon  # noqa: E402
from gpusonar.config import load_scenario_config  # noqa: E402
from gpusonar.sim import find_bellhop, run_scenario  # noqa: E402

core = gs.core


def db(power, ref):
    return 10.0 * np.log10(np.maximum(power / ref, 1e-12))


def display(power, target_median=0.2):
    """The paper's display mapping (dynamic range compression, Eq. 11-12) of an intensity image."""
    return core.drc(np.sqrt(power).astype(np.float32), target_median=target_median)[0]


def object_levels(scene, power, to_pixel):
    """Mean image power over each labelled object's scatterers, in dB over the image median."""
    median = float(np.median(power))
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
    return fig, {"scatterers": len(scene), "hidden_fraction": float(1.0 - seen.mean()),
                 "object_over_median_db": levels,
                 "dynamic_range_db": float(db(power.max(), np.median(power)))}


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
    return fig, metrics


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
    return fig, {"scatterers": len(result.scene), "pings": len(records), "object_over_median_db": levels,
                 "sharpness": sharpness(power)}


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
    return fig, metrics


# --- flank array -----------------------------------------------------------------------------

def run_flank(name, cfg, backend, quick):
    result = run_scenario(cfg, backend=backend)
    interval = float(cfg["pings"]["interval"])
    bearings = np.radians(np.arange(-40.0, 40.01, 0.5))
    tracker = detect.Tracker(range_sigma=5.0, bearing_sigma=np.radians(0.7), confirm_hits=3, max_misses=2,
                             initial_speed_sigma=5.0)
    first = None
    counts = []
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
        hits = [h for h in hits if h.snr_db > 12.0]
        counts.append(len(hits))
        tracks = tracker.update(i * interval, hits)
        if first is None:
            first = (ranges, power, hits)
    t_last = (len(result.records) - 1) * interval
    truth = {}
    start = 0
    for item in cfg["scene"]:
        n = len(item["positions"])
        v = np.asarray(item.get("velocity", (0.0, 0.0, 0.0)), dtype=float)
        truth[item["label"]] = (np.asarray(item["positions"], dtype=float).mean(axis=0), v)
        start += n
    metrics = {"pings": len(result.records), "detections_per_ping": counts, "confirmed_tracks": len(tracks), "targets": {}}
    for label, (pos, vel) in truth.items():
        end = pos[:2] + vel[:2] * t_last
        best = min(tracks, key=lambda t: np.linalg.norm(t.state[:2] - end), default=None)
        if best is None:
            metrics["targets"][label] = {"tracked": False}
            continue
        metrics["targets"][label] = {
            "tracked": bool(np.linalg.norm(best.state[:2] - end) < 60.0),
            "position_error_m": float(np.linalg.norm(best.state[:2] - end)),
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
    return fig, metrics


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
    return fig, {"peak_range_m": float(bm.ranges[ir]), "peak_bearing_deg": float(np.degrees(bearings[ib])),
                 "true_range_m": float(np.linalg.norm(target[:2])),
                 "peak_over_median_db": float(db(power.max(), floor))}


RUNNERS = {
    "pool_fls_l12_5d": run_fls, "pool_fls_p4_2v": run_fls, "seafloor_objects_fls": run_fls,
    "harbour_fls_750k": run_fls, "wreck_sas": run_sas, "minefield_sas": run_sas, "rail_objects": run_rail,
    "buried_objects": run_buried, "flank_patrol": run_flank, "popoto_trial_1000m": run_popoto,
}
NEEDS_BELLHOP = {"flank_patrol", "popoto_trial_1000m"}

# Thinned versions for a CPU: fewer scatterers and pings, same geometry.
QUICK = {
    "pool_fls_l12_5d": lambda c: _scale_density(c, 0.1),
    "pool_fls_p4_2v": lambda c: _scale_density(c, 0.1),
    "harbour_fls_750k": lambda c: _scale_density(c, 0.25),
    "wreck_sas": lambda c: (_scale_density(c, 0.3), c["pings"].update(count=40)),
    "minefield_sas": lambda c: (_scale_density(c, 0.3), c["pings"].update(count=40)),
    "rail_objects": lambda c: (_scale_density(c, 0.1), c["pings"].update(count=51)),
    "buried_objects": lambda c: c["platform"].update(raster={"x": [-0.9, 0.9001, 0.15], "y": [-0.9, 0.9001, 0.15]}),
    "flank_patrol": lambda c: (c["pings"].update(count=4), c["propagation"].update(max_arrivals=8),
                               c["sonar_overrides"].update(transmit={"waveform": {"duration": 0.2}})),
}


def _scale_density(cfg, factor):
    for item in cfg["scene"]:
        if "density" in item:
            item["density"] = float(item["density"]) * factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda" if gs.cuda_available() else "host")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    backend = core.Backend.Cuda if args.backend == "cuda" else core.Backend.Host
    out = Path(args.out)
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
        t = time.perf_counter()
        fig, metrics = RUNNERS[name](name, cfg, backend, args.quick)
        metrics["seconds"] = round(time.perf_counter() - t, 1)
        metrics["backend"] = args.backend
        metrics["quick"] = bool(args.quick)
        fig.savefig(out / f"{name}.png", dpi=110)
        plt.close(fig)
        summary[name] = metrics
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[{name}] {json.dumps(metrics)}")
    print("GALLERY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
