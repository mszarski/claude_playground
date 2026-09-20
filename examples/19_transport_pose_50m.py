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

**Where transport stops, and why it is not a bias.**  It walks in from 50 m and
settles about 5 m out, which looks like a systematic offset and is not: scanned
directly, the divergence has its minimum at *exactly* zero offset, for every
blur and against the same speckle realisation or an independent one.  What it
has is LOCAL minima.  A 12 m hull is a row of highlights, so sliding the model
along range aligns its highlights with the wrong ones -- a cycle slip -- and
the landscape reads 2.9 at the truth, 39 at 3 m, 25 at 4 m, 32 at 5 m.  That
dip at 4 m is where the search lands.  The global basin is only about +/-2 m
wide, and the coarse stages never get closer than 13 m, so the anneal delivers
the boat outside it.

**So the last step is a search, not more descent.**  Nine evaluations on a grid
of range shifts spanning the local minima find the global basin; the fine image
loss then polishes.  Annealing harder does not help and coarsening does not
either -- measured, an image loss at a 3 m cell drives the fit AWAY from the
truth from every start, and at 6 m it is worse still, so "a coarser stage has a
wider capture range" is false here.

**What is left at the end is not an optimisation failure.**  The residual is
almost entirely across-track, and an across-track error inside one 3.75 m
beamwidth barely changes the image -- there is no gradient to descend and no
search that helps.  That is the array's angular resolution, and it takes
another ping from another position to do better.

**Construction and assumptions.**

* *Sonar and scene*: 16's -- 12's 100 kHz FLS with 64 elements, its
  channel non-learnable, 300 transmit rays and 400 return rays, 61
  Hamming beams across +/-30 deg -- at two resolutions: a SEARCH stage
  (20-170 m in 218 bins, 2 ms pulse) and a REFINE stage (16's own,
  45-75 m in 233 bins, 0.12 ms).
* *Target*: 12's boat beam-on at 60 m; the truth rendered with one seed,
  the model with another; the start ``START`` 50 m out in range.
* *The losses*: the mean squared log10 image loss of 16 on the refine
  images; on the search images a Sinkhorn divergence
  (``sinkhorn_divergence``) in metres on both axes (bearing as arc
  length at ``R0``), at blurs from 8 m down to 0.5 m
  (``BLUR_SCHEDULE``, steps per blur), then a grid search of range
  shifts (``GRID_SHIFTS``) at a 2 m blur, then ``REFINE_STEPS`` of the
  image loss; Adam on position only at 0.5 blur (search) or 0.35 cell
  (refine) per step.
* *Assumptions*: position only (yaw is degenerate beam-on, per 16); a
  scene otherwise known; the transport plan is between the images as
  distributions of energy, so a global change of level is invisible to it.
* *To vary*: ``START`` maps the reach; ``BLUR_SCHEDULE`` and
  ``GRID_SHIFTS`` are set by the hull's highlight spacing (the local
  minima the docstring describes), so a different target needs different
  shifts.

Acceptance criteria:
  * the transport loss rises monotonically out to 50 m, where the image loss is
    flat -- the reason one can be descended on and the other cannot;
  * a fit from 50 m out, on transport alone, lands within a few metres -- in a
    local minimum, not on the truth;
  * a grid search over range shifts escapes it, and the fine image loss then
    finishes inside a bearing cell;
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
# The window reaches 60 m past the start: solved, the echo is compact and its
# ghosts trail it by up to 20 m, and a start at the window's edge lost that
# tail off the end -- which moved the mass's centroid inward and turned the
# transport's pull outward.
SEARCH = dict(near=20.0, far=170.0, n_bins=218, sigma_t=2.0e-3)
REFINE = dict(near=45.0, far=75.0, n_bins=233, sigma_t=1.2e-4)

# Blur is the distance mass moves for free, so it sets the reach AND the floor,
# and the floor is what decides where this can hand over.  Measured, the error
# each stage settles at is about four times its blur: 16 m at blur 8, 13 m at
# blur 4, under 8 m at blur 2 -- the entropic term biases the minimum away from
# zero offset, and shrinking the blur walks that minimum home.  A schedule that
# stops at 2 m therefore stops at ~8 m of error, which is outside the refiner's
# reach and leaves the two stages unable to meet.  It has to go down far enough
# that the floor is inside the capture range of what comes next.
# More steps per stage than the splatted echo needed: solved, the echo is a
# glint a cell wide rather than a smear, and the transport's pull on it is
# weaker for the same offset.
BLUR_SCHEDULE = [(8.0, 40), (4.0, 30), (2.0, 30), (1.0, 25), (0.5, 25)]
REFINE_STEPS = 40
# Range shifts to try after the anneal, spanning the local minima the hull's
# repeated highlights create.  Nine renders, no gradients: descent cannot leave
# a local minimum, and this is the cheapest thing that can.
# Wide enough for the ghosts as well as the hull: with every path solved the
# boat trails its seabed images 10-14 m behind it, and the anneal can lock the
# guess's own echo onto the measurement's ghost.
GRID_SHIFTS = [float(v) for v in range(-14, 3, 2)]
# And across-track as well: the anneal can settle a hull length to one side,
# where the guess's hull overlaps the measurement's, and a shift in range
# alone cannot leave that minimum either.
ACROSS_SHIFTS = [float(v) for v in range(-14, 15, 2)]
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
                                return_leg="eigenray", tx_pattern=fls.transmit_pattern,
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
            reached = True
            for _ in range(n_steps):
                opt.zero_grad()
                try:
                    L = transport_loss(render_s(boat, 202), meas_s, blur)
                except ValueError as stuck:
                    # The kernel no longer spans the two images: the anneal
                    # has settled in a local minimum further out than this
                    # blur can see across.  Smaller blurs cannot help; the
                    # range shifts below are what is for that.
                    print(f"    blur {blur:4.1f} m: cannot reach -- "
                          f"{str(stuck).split(' -- ')[0]}")
                    reached = False
                    break
                L.backward()
                with torch.no_grad():
                    boat.position.grad[2] = 0.0
                opt.step()
            if not reached:
                break
            print(f"    blur {blur:4.1f} m -> error {error(boat):6.2f} m")
    search_err = error(boat)

    # ---- 3. the escape -----------------------------------------------------
    banner("stage 2: a grid search, because descent cannot leave a local minimum")
    here = boat.position.detach().clone()
    best_shift, best_loss = (0.0, 0.0), float("inf")
    n_renders = 0
    with torch.no_grad():
        trial = offset_boat(0.0, 0.0, 0.0, learnable=False)
        for shift in GRID_SHIFTS:
            row = []
            for dy in ACROSS_SHIFTS:          # not `across`: that is the metric
                trial.position.copy_(here + torch.tensor([shift, dy, 0.0]))
                v = float(transport_loss(render_s(trial, 202), meas_s, 2.0))   # a blur that still reaches
                n_renders += 1
                row.append(v)
                if v < best_loss:
                    best_loss, best_shift = v, (shift, dy)
            print(f"    range {shift:+5.1f} m: best across {ACROSS_SHIFTS[row.index(min(row))]:+5.1f} m, "
                  f"loss {min(row):9.3f}")
    with torch.no_grad():
        boat.position.copy_(here + torch.tensor([best_shift[0], best_shift[1], 0.0]))
    grid_err = error(boat)
    off = (boat.position.detach() - true_pos)
    print(f"  picked range {best_shift[0]:+.1f} m, across {best_shift[1]:+.1f} m: "
          f"{search_err:.2f} m -> {grid_err:.2f} m "
          f"(range {float(off[0]):+.2f} m, across {float(off[1]):+.2f} m)")

    banner("stage 3: the image loss polishes it")
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
    residual = (boat.position.detach() - true_pos)[:2]
    print(f"  at the {cell:.2f} m cell: {grid_err:.2f} m -> {final_err:.2f} m")
    print(f"  residual: {float(residual[0]):+.2f} m in range, "
          f"{float(residual[1]):+.2f} m across -- against a {beam_m:.2f} m beam")
    print(f"  end to end: {START[0]:.0f} m -> {final_err:.2f} m "
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
    # Within the grid's reach, not within the refiner's: with the echo solved
    # the anneal settles a hull length to one side (12-13 m), and the 2-D
    # grid below is what takes it from there.
    ok &= check("transport alone walks it in from 50 m to within the grid's reach",
                search_err < max(abs(v) for v in ACROSS_SHIFTS),
                f"{START[0]:.0f} m -> {search_err:.2f} m")
    ok &= check("a grid search escapes the local minimum descent settled in",
                grid_err < search_err - 1.0,
                f"{search_err:.2f} m -> {grid_err:.2f} m on {len(GRID_SHIFTS)} "
                f"renders")
    ok &= check("and the image loss then beats the bearing cell",
                final_err < beam_m, f"{final_err:.2f} m against {beam_m:.2f} m")
    ok &= check("what is left is across-track, inside a beamwidth",
                abs(float(residual[1])) > abs(float(residual[0])) and
                abs(float(residual[1])) < beam_m,
                f"{float(residual[0]):+.2f} m range vs "
                f"{float(residual[1]):+.2f} m across")
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
