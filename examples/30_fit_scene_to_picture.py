"""Fitting the scene to a picture: the sea, the seabed, the tilt and the gain from a bare ping.

Every fit so far (``examples/16``, ``19``, ``22``) put a boat into a picture
whose background was the model's own.  A real picture's background is not:
its seabed's strength, its surface's, the head's tilt, its gain and its
noise floor are all somewhat wrong in the model, and a boat fitted on top
of a wrong background is fitted to the wrong contrast.  So the first thing
to fit to a real picture is the scene, and this example does it the way it
would be done on a real ping: the picture comes in from a file as an image
in dB with its axes (``hydropt.load_picture``), nothing else about it is
assumed, and the fit descends the swath's level against range.

**What is fitted, and through what.**  Five parameters
(``hydropt.SceneFit``): the seabed's Lambert strength, the surface's
strength as a gain on the surface patches, the head's tilt (the transmit
array factor and the receive envelope both steer with it, and the fan's
directions do not, so no retrace), a gain offset and the ambient level.
The observable is ``range_profile``: the median over the beams of three
bearing sectors at each range, smoothed over 5 m.  A median because it is
the level a display's AGC reads and a target cannot move it; the sector's
60 beams because one beam's speckle scatters by 5.6 dB and sixty's median
by under one.  The loss is the squared difference of profiles in dB, and
it is smooth in all five parameters: they scale amplitudes or steer
weights, none moves a scatterer, so the coherent picture's gradient serves
(where for a pose it did not, ``examples/22``).

**What separates them.**  In 21's geometry the surface is in the lobe from
54 m and the seabed from 191 m; the noise floor takes over past the swath
at 120 kHz (at 330 kHz it is inside it).  So the near profile is the
surface's strength, the far profile the seabed's, the roll-off between
them the tilt, the far end the noise, and the gain is the offset common to
all.  Where two of them are not separable by this picture the fit says so
by moving them together: at 120 kHz over 300 m the noise floor is 12 dB
under the reverberation, so ``noise_db`` is weakly determined and is
reported, not checked.

**The "real" picture** is rendered with hidden values -- the seabed 5 dB
stronger, the surface 3 dB stronger, the head tilted 2 degrees further
up, 4 dB of extra gain, the ambient 2 dB up -- and, so that nothing but
the physics is shared, with independent realisations of everything random:
the sea surface and the seabed relief (a different seed), the transmit
fan's jitter, the patch draw and the noise.  It is saved to
``figures/30_real_picture.npz`` as ``image`` (dB), ``bearings`` and
``ranges``, and loaded back as a real picture would be.  The model starts
from 21's defaults.

Acceptance criteria:
  * the profile misfit falls by more than 80 %;
  * the seabed strength is recovered within 1.5 dB and the gain within
    1.5 dB (the speckle's median over a sector scatters by about a
    decibel, and the two are separated by the seabed's onset at 191 m);
  * the tilt within 1 degree and the surface strength within 2 dB;
  * the fit's gradient agrees with a finite difference (cosine over 0.9)
    at the start, so the descent is descent.

**Construction and assumptions.**  21's sonar and scene (see 21's
docstring; ``HYDROPT_SONAR=330`` runs the other head); ``HIDDEN`` holds
the offsets the real picture is rendered with, ``REAL_SEED`` its seeds;
``SceneFit`` from 21's fan and beamformer settings; ``fit_scene`` with
Adam at ``LR`` for ``STEPS`` steps.  Assumptions: altitude and water depth
known (they move the trace); the wind enters as the surface's strength;
the real picture's display is linear in dB with an unknown offset (the
gain).  To vary: ``HIDDEN``, ``STEPS``, the sectors in ``fit_scene``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import FIGURE_DIR, banner, check, save, setup, timed
from hydropt import (
    RealPicture, SceneFit, azimuth_steering, beam_noise_power, fit_scene,
    line_array_directivity_db, load_picture, make_time_grid, range_profile, shading_window,
)

HIDDEN = dict(seabed_db=-22.0, surface_db=3.0, tilt_deg=-7.0, gain_db=4.0, noise_db=2.0)
# absolute values; 21's defaults are -27, 0, TILT_DEG (-5), 0, 0
REAL_SEED = 101                 # the real picture's sea, seabed, fan, patches and noise
STEPS = int(os.environ.get("HYDROPT_STEPS", 40))
LR = 0.6
SECTORS = ((-60.0, -20.0), (-20.0, 20.0), (20.0, 60.0))


def _ex21():
    path = Path(__file__).resolve().parent / "21_long_range_300m.py"
    spec = importlib.util.spec_from_file_location("_ex21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    setup(double=False)
    banner("30 -- fitting the scene to a picture: seabed, surface, tilt, gain, noise")
    ex = _ex21()
    C = ex.C
    rx = ex.horizontal_array()
    steer, bearings = azimuth_steering(ex.N_BEAMS, ex.SECTOR_DEG)
    grid = make_time_grid(2.0 * ex.NEAR / C, 2.0 * ex.FAR / C, ex.N_BINS)
    ranges = grid * C / 2.0
    shading = shading_window(ex.N_RX, "hamming")
    print(f"  {ex.FREQ_KHZ:.0f} kHz, {ex.N_RX} x {ex.N_TX}, {2 * ex.SECTOR_DEG:.0f} deg swath to "
          f"{ex.FAR:.0f} m; the seabed enters the lobe at {ex.AUV_DEPTH + 0.0:.0f} m altitude "
          f"past about {(ex.WATER_DEPTH - ex.AUV_DEPTH) / 0.094:.0f} m")
    elev_beam = ex.beam_3db_deg(ex.N_RX_ELEV)

    def model_for(scene, dirs, seed, **values):
        solid = (math.radians(2 * ex.SECTOR_DEG)
                 * math.radians(ex.ELEV_DEG[1] - ex.ELEV_DEG[0]) / dirs.shape[0])
        noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=1.0 / ex.PULSE_S,
                                       directivity_db=line_array_directivity_db(ex.N_RX),
                                       wind_speed=ex.WIND))
        return SceneFit(scene, dirs, elements=rx, time_grid=grid, steer=steer, sigma_t=ex.PULSE_S,
                        shading=shading, source_level_db=ex.SOURCE_LEVEL_DB, noise_power=noise,
                        solid_angle_per_ray=solid, n_tx=ex.N_TX, n_rx_elev=ex.N_RX_ELEV,
                        n_elev_beams=ex.N_ELEV_BEAMS, elev_beam_deg=elev_beam,
                        max_arrivals=ex.PATCHES, seed=seed, **values)

    # ---- the "real" picture: hidden values, independent realisations ------- #
    banner("the real picture, rendered with hidden values and its own seeds")
    real_scene, *_ = ex.build_scene(rx, seed=REAL_SEED, learnable=False)
    real_dirs, _ = ex.transmit_fan(seed=REAL_SEED)
    hidden = dict(HIDDEN)         # absolute values: 21's are -27, 0, TILT_DEG, 0, 0
    print("  hidden: " + ", ".join(f"{k} {v:+.1f}" for k, v in hidden.items()))
    with torch.no_grad(), timed("  trace + picture"):
        truth = model_for(real_scene, real_dirs, REAL_SEED, fit=(), **hidden)
        real_db = 10.0 * torch.log10(truth.picture().clamp_min(1e-30))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    real_path = FIGURE_DIR / f"30_real_picture{ex.TAG}.npz"
    np.savez(real_path, image=real_db[:, 0].numpy(), bearings=bearings.numpy(), ranges=ranges.numpy())
    print(f"  wrote {real_path.relative_to(FIGURE_DIR.parent)} (dB, [beams, bins] with its axes)")
    del truth, real_scene

    # ---- the fit, from 21's defaults ------------------------------------- #
    banner("the fit")
    real = load_picture(real_path)                    # as a real picture would come in
    scene, *_ = ex.build_scene(rx, learnable=False)   # 21's sea and seabed: a different realisation
    dirs, _ = ex.transmit_fan(seed=ex.SEED)
    with timed("  trace"):
        model = model_for(scene, dirs, ex.SEED, seabed_db=-27.0, surface_db=0.0,
                          tilt_deg=ex.TILT_DEG, gain_db=0.0, noise_db=0.0)
    start = model.values()
    target = range_profile(real.image_db, bearings, ranges, sectors_deg=SECTORS)

    # the gradient against a finite difference on the two levels, at the start
    def loss_at(m):
        prof = range_profile(10.0 * torch.log10(m.picture().clamp_min(1e-30)), bearings, ranges,
                             sectors_deg=SECTORS)
        return ((prof - target) ** 2).mean()
    with timed("  gradient check"):
        L = loss_at(model); L.backward()
        g = torch.tensor([float(model.seabed_db.grad), float(model.gain_db.grad)])
        fd = []
        for name in ("seabed_db", "gain_db"):
            with torch.no_grad():
                p = getattr(model, name)
                p += 0.25; lp = float(loss_at(model)); p -= 0.5; lm = float(loss_at(model)); p += 0.25
            fd.append((lp - lm) / 0.5)
        fd = torch.tensor(fd)
        cos = float(g @ fd / (g.norm() * fd.norm() + 1e-30))
        model.zero_grad(set_to_none=True)
    print(f"  d loss / d(seabed, gain): analytic ({g[0]:+.3f}, {g[1]:+.3f}), "
          f"finite difference ({fd[0]:+.3f}, {fd[1]:+.3f}), cosine {cos:+.3f}")

    t0 = time.perf_counter()
    history = fit_scene(model, real, steps=STEPS, lr=LR, sectors_deg=SECTORS, log=print)
    per_step = (time.perf_counter() - t0) / STEPS
    final = model.values()
    print(f"\n  {STEPS} steps, {per_step:.1f} s each; loss {history[0]['loss']:.2f} -> "
          f"{history[-1]['loss']:.2f} dB^2")
    print(f"  {'parameter':>11s} {'start':>8s} {'fitted':>8s} {'hidden':>8s} {'error':>7s}")
    for k in hidden:
        print(f"  {k:>11s} {start[k]:8.2f} {final[k]:8.2f} {hidden[k]:8.2f} {final[k] - hidden[k]:+7.2f}")

    # ---- the figure --------------------------------------------------------- #
    with torch.no_grad():
        fitted_db = 10.0 * torch.log10(model.picture().clamp_min(1e-30))
        model0 = model_for(scene, dirs, ex.SEED, fit=(), **start)
        start_db = 10.0 * torch.log10(model0.picture().clamp_min(1e-30))
    profs = {"real (hidden values, own seeds)": target,
             "model at the start": range_profile(start_db, bearings, ranges, sectors_deg=SECTORS),
             "model fitted": range_profile(fitted_db, bearings, ranges, sectors_deg=SECTORS)}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    vmin, vmax = float(target.min()) - 5.0, float(target.max()) + 15.0
    for ax, (title, img) in zip(axes[0], (("the real picture", real.image_db),
                                          ("the model at the start", start_db[:, 0]),
                                          ("the model fitted", fitted_db[:, 0]))):
        m = ax.imshow(img.numpy(), origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="inferno",
                      extent=[float(ranges[0]), float(ranges[-1]), float(bearings[0]), float(bearings[-1])])
        ax.set_title(title); ax.set_xlabel("range (m)"); ax.set_ylabel("bearing (deg)")
    fig.colorbar(m, ax=axes[0].tolist(), fraction=0.02, label="dB re 1 uPa^2")
    for i, (lo, hi) in enumerate(SECTORS):
        ax = axes[1][i]
        for label, prof in profs.items():
            ax.plot(ranges.numpy(), prof[i].numpy(), lw=1.2, label=label)
        ax.set_title(f"sector {lo:+.0f}..{hi:+.0f} deg: the level against range")
        ax.set_xlabel("range (m)"); ax.set_ylabel("dB"); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{ex.FREQ_KHZ:.0f} kHz FLS to {ex.FAR:.0f} m: the scene fitted to a picture -- "
                 + ", ".join(f"{k} {final[k]:+.1f} ({hidden[k]:+.1f})" for k in hidden))
    save(fig, f"30_fit_scene{ex.TAG}.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in hidden:
        ax.plot([h[k] - hidden[k] for h in history], lw=1.2, label=k)
    ax.axhline(0.0, color="k", lw=0.6); ax.set_xlabel("step"); ax.set_ylabel("fitted - hidden")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("the descent")
    save(fig, f"30_fit_scene_history{ex.TAG}.png")

    banner("acceptance")
    ok = check("the profile misfit falls by more than 80 %",
               history[-1]["loss"] < 0.2 * history[0]["loss"],
               f"{history[0]['loss']:.2f} -> {history[-1]['loss']:.2f} dB^2")
    ok &= check("the seabed strength and the gain are recovered within 1.5 dB",
                abs(final["seabed_db"] - hidden["seabed_db"]) < 1.5
                and abs(final["gain_db"] - hidden["gain_db"]) < 1.5,
                f"seabed {final['seabed_db'] - hidden['seabed_db']:+.2f}, gain "
                f"{final['gain_db'] - hidden['gain_db']:+.2f} dB")
    ok &= check("the tilt within 1 degree and the surface strength within 2 dB",
                abs(final["tilt_deg"] - hidden["tilt_deg"]) < 1.0
                and abs(final["surface_db"] - hidden["surface_db"]) < 2.0,
                f"tilt {final['tilt_deg'] - hidden['tilt_deg']:+.2f} deg, surface "
                f"{final['surface_db'] - hidden['surface_db']:+.2f} dB")
    ok &= check("the fit's gradient agrees with the finite difference", cos > 0.9, f"cosine {cos:+.3f}")
    print(f"  (noise level: fitted {final['noise_db']:+.2f} against {hidden['noise_db']:+.2f} dB, "
          f"reported, not checked)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
