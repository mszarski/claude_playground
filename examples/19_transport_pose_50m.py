"""Recovering a pose from 50 m out, where descent on the image cannot.

``examples/16`` fits a boat's pose by descending on an image loss and is honest
about its limit: it is a refiner, capturing from about 1-2 m.  This is why, and
what to do instead.

**Why a squared-error loss cannot reach.**  It compares the two images cell by
cell.  Put the model 25 m from the truth and the two share no lit cell at all,
so the mismatch is the same whichever way the model moves.  Measured here, the
cosine between the descent direction and the true direction home is 0.00 at
25 m -- literally zero, because with the usual 45-75 m window a boat that far
out of place renders nothing into the image -- and widening the window until it
does render makes it worse rather than better: -0.89, pointing away.

**What transport does instead.**  It asks what it would cost to *move* the
model's energy onto the measurement's rather than whether the two coincide, and
that cost falls as the two approach from any separation.  On these images the
Sinkhorn divergence grows as ``d^2`` from 5 m to 50 m while the squared error
peaks at 10 m and then falls back, having saturated.

**Two stages, because each is good at one thing.**  Transport has a floor: the
measurement and the model are independent speckle realisations, so even at zero
offset the divergence is not zero, and the position it can resolve is about
``sqrt`` of that -- a few metres.  Descent on the image has no reach but,
inside a metre or two, resolves to a fraction of a bearing cell.  So transport
walks the boat in from 50 m and the image loss finishes the job, which is the
same detect-then-refine shape a real system has, with a differentiable detector.

Acceptance criteria:
  * the transport loss rises monotonically out to 50 m, where the image loss is
    flat -- the reason one can be descended on and the other cannot;
  * a fit from 50 m out, on transport alone, lands within a few metres;
  * handing that to the image-loss refiner recovers the pose to inside a
    bearing cell;
  * the image loss started from the same 50 m does not converge at all, which
    is the comparison that makes the point;
  * both stages carry gradients to the boat's pose.

Position only.  Heading is left out because ``examples/16`` measured it to be
degenerate at broadside -- yaw barely moves the image there -- and mixing a
degenerate parameter into a search would confuse what the search is being
judged on.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    azimuth_steering, beamform, make_time_grid, shading_window,
    sinkhorn_divergence, target_arrivals,
)

C = 1500.0
N_ELEMENTS = 64
N_BEARINGS = 61
TX_RAYS, RX_RAYS = 300, 400
SECTOR_DEG = 30.0
R0 = 60.0                         # nominal range, for the across-track metric

# The search stage: a wide window and a coarse cell, because the boat may be
# anywhere in it.  The refine stage is examples/16's own resolution.
SEARCH = dict(near=20.0, far=130.0, n_bins=160, sigma_t=2.0e-3)
REFINE = dict(near=45.0, far=75.0, n_bins=233, sigma_t=1.2e-4)

# Blur is the distance mass moves for free, so it sets the reach AND the floor,
# and the floor is what decides where this can hand over.  Measured, the error
# each stage settles at is about four times its blur: 16 m at blur 8, 13 m at
# blur 4, under 8 m at blur 2 -- the entropic term biases the minimum away from
# zero offset, and shrinking the blur walks that minimum home.  A schedule that
# stops at 2 m therefore stops at ~8 m of error, which is outside the refiner's
# reach and leaves the two stages unable to meet.  It has to go down far enough
# that the floor is inside the capture range of what comes next.
BLUR_SCHEDULE = [(8.0, 15), (4.0, 10), (2.0, 20), (1.0, 20), (0.5, 25)]
REFINE_STEPS = 30
START = (50.0, 0.0, 0.0)          # dx, dy, dyaw -- 50 m out in range


def _fls():
    path = Path(__file__).resolve().parent / "12_fls_boat_learnable.py"
    spec = importlib.util.spec_from_file_location("_fls12", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    setup()
    banner("19 -- 50 m out: transport gets there, descent cannot")
    fls = _fls()
    elements = fls.array(N_ELEMENTS)
    scene, *_ = fls.build_scene(elements, learnable=False)
    scene.step_size, scene.n_steps = 3.0, 60
    steer, bearings = azimuth_steering(N_BEARINGS, SECTOR_DEG)
    tx_dirs, tx_w = fls.transmit(TX_RAYS)
    shade = shading_window(elements.shape[0], "hamming")

    beam_deg = 2.0 * math.degrees(math.asin(2.0 / elements.shape[0]))
    beam_m = R0 * math.radians(beam_deg)
    print(f"  {elements.shape[0]} elements -> {beam_deg:.2f} deg beam = "
          f"{beam_m:.2f} m at {R0:.0f} m")
    print(f"  search: {SEARCH['near']:.0f}-{SEARCH['far']:.0f} m window, "
          f"{SEARCH['sigma_t'] * C / 2:.2f} m cell")
    print(f"  refine: {REFINE['near']:.0f}-{REFINE['far']:.0f} m window, "
          f"{REFINE['sigma_t'] * C / 2:.2f} m cell")
    print(f"  starting {START[0]:.0f} m out in range")

    def make(stage):
        grid = make_time_grid(2 * stage["near"] / C, 2 * stage["far"] / C,
                              stage["n_bins"])

        def render(boat, seed):
            a = target_arrivals(scene, boat, tx_dirs, n_rx_rays=RX_RAYS,
                                rx_half_angle_deg=40.0, rx_jitter=1.0,
                                tx_weights=tx_w, max_arrivals_per_leg=8,
                                generator=torch.Generator().manual_seed(seed))
            return beamform(a, elements, scene.freqs_khz, grid, steer,
                            sigma_t=stage["sigma_t"], shading=shade,
                            steer_chunk=8)[:, 0]

        return render, grid * C / 2.0

    render_s, range_s = make(SEARCH)
    render_r, range_r = make(REFINE)
    # Metres on both axes.  Range is already metres; bearing becomes arc length
    # at the nominal range, which keeps the cost separable -- a flat-earth
    # approximation of the wedge, good to a few percent over this sector, and
    # the alternative is a cost matrix that does not factorise and does not fit.
    across = torch.deg2rad(bearings) * R0

    truth = fls.build_boat(learnable=False)
    true_pos = truth.position.detach().clone()
    with torch.no_grad():
        meas_s = render_s(truth, 101)
        meas_r = render_r(truth, 101)

    def offset_boat(dx, dy, dyaw, *, learnable):
        b = fls.build_boat(heading_deg=90.0, learnable=learnable)
        with torch.no_grad():
            b.position += torch.tensor([dx, dy, 0.0])
            b.orientation[0] += math.radians(dyaw)
        return b

    def image_loss(img, ref):
        eps = float(ref.max()) * 1e-4
        return ((torch.log10(img + eps) - torch.log10(ref + eps)) ** 2).mean()

    def transport_loss(img, ref, blur):
        return sinkhorn_divergence(img, ref, x=across, y=range_s, blur=blur,
                                   n_iter=200)

    def error(boat):
        return float((boat.position.detach() - true_pos)[:2].norm())

    # ---- 1. the landscape --------------------------------------------------
    banner("the two losses, as the boat is walked away")
    offs = [0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
    ot, se = [], []
    with torch.no_grad():
        for off in offs:
            img = render_s(offset_boat(off, 0.0, 0.0, learnable=False), 202)
            ot.append(float(transport_loss(img, meas_s, 4.0)))
            se.append(float(image_loss(img, meas_s)))
    print(f"  {'offset':>7} {'transport':>11} {'image loss':>12}")
    for off, o, s in zip(offs, ot, se):
        print(f"  {off:7.1f} {o:11.2f} {s:12.4f}")
    far = [i for i, off in enumerate(offs) if off >= 5.0]
    rises = all(ot[i + 1] > ot[i] for i in far[:-1])
    flat = max(se[i] for i in far) / min(se[i] for i in far)
    print(f"\n  transport rises monotonically past 5 m: {rises}")
    print(f"  the image loss varies by {flat:.2f}x over the same span -- "
          f"saturated, and slightly falling")

    # ---- 2. the fit, on transport ------------------------------------------
    banner(f"stage 1: transport, from {START[0]:.0f} m out")
    boat = offset_boat(*START, learnable=True)
    print(f"  starting error {error(boat):.2f} m")
    with timed("  search"):
        for blur, n_steps in BLUR_SCHEDULE:
            opt = torch.optim.Adam([{"params": [boat.position], "lr": 0.5 * blur}])
            for _ in range(n_steps):
                opt.zero_grad()
                L = transport_loss(render_s(boat, 202), meas_s, blur)
                L.backward()
                with torch.no_grad():
                    boat.position.grad[2] = 0.0
                opt.step()
            print(f"    blur {blur:4.1f} m -> error {error(boat):6.2f} m")
    search_err = error(boat)

    # ---- 3. the handoff ----------------------------------------------------
    banner("stage 2: the image loss finishes it")
    cell = REFINE["sigma_t"] * C / 2.0
    with timed("  refine"):
        opt = torch.optim.Adam([{"params": [boat.position], "lr": 0.35 * cell}])
        for _ in range(REFINE_STEPS):
            opt.zero_grad()
            image_loss(render_r(boat, 202), meas_r).backward()
            with torch.no_grad():
                boat.position.grad[2] = 0.0
            opt.step()
    final_err = error(boat)
    print(f"  {search_err:.2f} m -> {final_err:.2f} m "
          f"({beam_m / max(final_err, 1e-9):.1f}x finer than the bearing cell)")

    # ---- 4. the control: the image loss alone, from the same start ---------
    banner("the control: the image loss from 50 m, on its own")
    alone = offset_boat(*START, learnable=True)
    with timed("  descent"):
        for blur, n_steps in BLUR_SCHEDULE:            # same budget
            opt = torch.optim.Adam([{"params": [alone.position],
                                     "lr": 0.5 * blur}])
            for _ in range(n_steps):
                opt.zero_grad()
                image_loss(render_s(alone, 202), meas_s).backward()
                with torch.no_grad():
                    alone.position.grad[2] = 0.0
                opt.step()
    control_err = error(alone)
    print(f"  {START[0]:.0f} m -> {control_err:.2f} m: "
          f"{'moved' if abs(control_err - START[0]) > 2.0 else 'went nowhere'}")

    banner("gradients")
    probe = offset_boat(20.0, 0.0, 0.0, learnable=True)
    transport_loss(render_s(probe, 202), meas_s, 4.0).backward()
    g = float(probe.position.grad[:2].norm())
    print(f"  |d(transport)/d(position)| at 20 m out: {g:.3e}")

    banner("figure")
    with timed("  draw"):
        fig = draw(offs, ot, se, search_err, final_err, control_err, beam_m)
        save(fig, "19_transport_pose_50m.png")

    banner("acceptance")
    ok = True
    ok &= check("transport rises out to 50 m where the image loss is flat",
                rises and flat < 1.5,
                f"transport {ot[-1] / ot[far[0]]:.0f}x over 5-50 m, "
                f"image loss {flat:.2f}x")
    ok &= check("transport alone walks it in from 50 m",
                search_err < 6.0, f"{START[0]:.0f} m -> {search_err:.2f} m")
    ok &= check("and the image loss then beats the bearing cell",
                final_err < beam_m, f"{final_err:.2f} m against {beam_m:.2f} m")
    ok &= check("while the image loss alone gets nowhere from there",
                control_err > 3.0 * final_err,
                f"{control_err:.2f} m against {final_err:.2f} m for the pair")
    ok &= check("the transport loss carries a gradient at 20 m out",
                g > 0.0, f"|grad| = {g:.2e}")
    return 0 if ok else 1


def draw(offs, ot, se, search_err, final_err, control_err, beam_m):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    ax.plot(offs, np.array(ot) / max(ot), "o-", color="#1b3a5c",
            label="transport (Sinkhorn)")
    ax.plot(offs, np.array(se) / max(se), "s-", color="#a2512f",
            label="image loss (log, squared)")
    ax.set_xlabel("how far the model is from the truth (m)")
    ax.set_ylabel("loss, scaled to its own maximum")
    ax.set_title("only one of these can be descended on")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=9)

    names = ["transport\nalone", "then the\nimage loss", "image loss\nalone"]
    vals = [search_err, final_err, control_err]
    colours = ["#1b3a5c", "#2a9d5c", "#b23b3b"]
    bx.bar(range(3), vals, color=colours)
    bx.axhline(beam_m, ls="--", c="#666", lw=0.9,
               label=f"a bearing cell ({beam_m:.1f} m)")
    for i, v in enumerate(vals):
        bx.text(i, v, f" {v:.2f} m", ha="center", va="bottom", fontsize=9)
    bx.set_xticks(range(3))
    bx.set_xticklabels(names, fontsize=9)
    bx.set_ylabel("final error (m)")
    bx.set_title(f"starting {START[0]:.0f} m out")
    bx.grid(alpha=0.3, lw=0.4, axis="y")
    bx.legend(fontsize=9)
    fig.suptitle("A boat 50 m out of place.  Transport has reach and a floor; "
                 "the image loss has precision and no reach.", fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
