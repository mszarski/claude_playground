"""A boat as a **triangle mesh**, scattering by Kirchhoff physical optics.

Everything so far built a target by hand: put a cylinder here, a plate there,
a curved patch for the hull.  That works, and for a hull it is even the right
physics, but it means deciding in advance which primitive each part of the boat
is.  A mesh does not ask: it integrates the Kirchhoff surface integral over the
actual geometry and the aspect dependence falls out.

**A mesh needs no new propagation machinery.**  A `ScatteringPattern` answers
"given an incident and a scattered direction, what is sigma?", and a mesh can
answer that directly.  So `MeshScattering` drops into the same `ExtendedTarget`
the other examples use, traces the same two legs, and costs no extra rays.

The one thing that has to be right is the **per-facet phase integral**.  The
obvious shortcut -- treat each facet as a point of amplitude `A exp(i q.c)` --
is valid only when `|q| d << 1`, and at 100 kHz with centimetre facets `|q| d`
is about 25 radians.  That shortcut came out 16x too high even at 82,000
facets, converging only as fast as facet area.  With the exact integral the
same sphere is right to 0.1 dB at 20,000.

Acceptance criteria:
  * a faceted sphere reproduces the analytic `sigma = a^2/4` at every aspect,
    and the point-facet shortcut visibly does not;
  * a hull mesh reproduces the analytic curved-surface hull it replaces, to
    within the facet resolution;
  * an ellipsoid reproduces `sigma = A^2 C^2 / 4 B^2`, which separates `R1` from
    `R2` in a way a sphere cannot;
  * at a forward-looking sonar's depression angle the hull's return depends
    strongly on aspect, and from beneath the flat run aft returns like a plate
    -- bright near normal, through a very narrow lobe;
  * a loss on the beamformed image reaches the mesh *vertices*, so the shape
    itself is learnable;
  * the cost of a mesh forward pass is measured, not asserted.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch

from _common import banner, check, save, setup, timed
from hydropt import CurvedSurfaceScattering, PlateScattering, azimuth_steering, beamform
from hydropt import make_time_grid, shading_window, target_arrivals
from hydropt.mesh import (
    MeshScattering, boat_hull_mesh, facet_geometry, icosphere, mesh_target,
)

C = 1500.0
FREQ_KHZ = 100.0
LAM = C / (FREQ_KHZ * 1e3)
HULL_LENGTH, HULL_BEAM, HULL_DRAFT = 12.0, 3.0, 1.0
N_LONG, N_AROUND = 160, 48          # ~9 cm facets; see the resolution table


def _fls():
    """Reuse the 100 kHz FLS scene from examples/12 rather than rebuild it."""
    path = Path(__file__).resolve().parent / "12_fls_boat_learnable.py"
    spec = importlib.util.spec_from_file_location("_fls12", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mono(pattern, look, freqs):
    d = torch.as_tensor(look, dtype=torch.get_default_dtype()).reshape(1, 3)
    d = d / d.norm()
    return pattern.cross_section(d, -d, freqs)


def main() -> int:
    setup()
    banner("14 -- a boat as a triangle mesh, by Kirchhoff physical optics")
    freqs = torch.tensor([FREQ_KHZ])
    print(f"  {FREQ_KHZ:.0f} kHz, lambda = {LAM * 1e3:.1f} mm")
    print(f"  facets must resolve curvature, about sqrt(lambda R)/3:")
    for r in (0.5, 0.75, 2.0):
        print(f"    R = {r:4.2f} m -> {math.sqrt(LAM * r) / 3 * 100:5.2f} cm")

    # ---- 1. the reference body: a sphere, whose answer is known exactly ----- #
    banner("a sphere is the one body with an exact answer")
    print("  sigma = a^2/4 at every aspect and every frequency.\n")
    print(f"  {'a':>5s} {'facets':>7s} {'facet':>8s} {'sigma':>9s} {'exact':>9s} "
          f"{'err':>7s}")
    sphere_err = []
    for a, sub in ((1.0, 4), (1.0, 5), (0.5, 5)):
        v, f = icosphere(sub, a)
        pat = MeshScattering(v, f, sound_speed=C, facet_chunk=4096)
        size = float((v[f][:, 1] - v[f][:, 0]).norm(dim=-1).mean())
        got = float(_mono(pat, [0.0, 0.0, 1.0], freqs))
        err = 10 * math.log10(got / (a * a / 4))
        sphere_err.append(abs(err))
        print(f"  {a:5.2f} {f.shape[0]:7d} {size * 100:6.2f}cm {got:9.5f} "
              f"{a * a / 4:9.5f} {err:+6.2f}dB")

    v, f = icosphere(5, 1.0)
    pat = MeshScattering(v, f, sound_speed=C, facet_chunk=4096)
    aspects = [[0, 0, 1.0], [1, 0, 0.0], [0, 1, 0.0], [1, 1, 1.0], [-1, 2, 0.5]]
    levels = [float(_mono(pat, look, freqs)) for look in aspects]
    spread = 10 * math.log10(max(levels) / min(levels))
    print(f"\n  across 5 aspects: spread {spread:.2f} dB (exact: 0.00)")

    # the shortcut, for contrast
    centroid, normal, area = facet_geometry(v, f)
    look = torch.tensor([[0.0, 0.0, 1.0]])
    q = (2 * math.pi / LAM) * (-look - look)
    lit = (-(look @ normal.T)).clamp_min(0.0).reshape(-1)
    ph = (centroid * q).sum(-1)
    shortcut = float(((area * lit * torch.complex(ph.cos(), ph.sin())).sum()
                      / LAM).abs() ** 2)
    shortcut_err = 10 * math.log10(shortcut / 0.25)
    print(f"  point-facet shortcut on the same mesh: {shortcut:.3f} "
          f"({shortcut_err:+.1f} dB) -- this is the trap")

    # ---- 2. the hull ------------------------------------------------------- #
    banner("the hull mesh, against the analytic hull it replaces")
    verts, faces = boat_hull_mesh(HULL_LENGTH, HULL_BEAM, HULL_DRAFT,
                                  n_long=N_LONG, n_around=N_AROUND)
    _, _, areas = facet_geometry(verts, faces)
    size = float((verts[faces][:, 1] - verts[faces][:, 0]).norm(dim=-1).mean())
    print(f"  {HULL_LENGTH:.0f} x {HULL_BEAM:.0f} x {HULL_DRAFT:.0f} m wetted hull: "
          f"{verts.shape[0]} vertices, {faces.shape[0]} facets")
    print(f"  mean facet {size * 100:.1f} cm, wetted area {float(areas.sum()):.1f} m^2")

    hull = MeshScattering(verts, faces, sound_speed=C, facet_chunk=4096)
    analytic = CurvedSurfaceScattering(HULL_DRAFT, 30.0, learnable=False)
    print(f"\n  beam-on, the mesh against the analytic curved patch it stands in for:")
    beam_on = float(_mono(hull, [0.0, 1.0, 0.0], freqs))
    print(f"    mesh     TS {10 * math.log10(beam_on):+6.2f} dB")
    print(f"    analytic TS {10 * math.log10(float(_mono(analytic, [0, 1.0, 0], freqs))):+6.2f} dB"
          f"   (one patch, R1={HULL_DRAFT}, R2=30)")

    banner("an ellipsoid separates R1 from R2, where a sphere cannot")
    print("  looking along -y at the y=B specular point, R1 = A^2/B and R2 = C^2/B,")
    print("  so sigma = A^2 C^2 / (4 B^2) -- a closed form in all three axes.\n")
    print(f"  {'A,B,C':>16s} {'facets':>7s} {'mesh TS':>9s} {'exact':>9s} {'err':>7s}")
    ellip_err = []
    for (a, b, c) in [(2.0, 1.0, 1.0), (1.0, 1.0, 2.0), (3.0, 1.0, 0.5)]:
        sv, sf = icosphere(5, 1.0)
        sv = sv * torch.tensor([a, b, c], dtype=sv.dtype)
        pat_e = MeshScattering(sv, sf, sound_speed=C, facet_chunk=8192)
        got = 10 * math.log10(float(_mono(pat_e, [0.0, 1.0, 0.0], freqs)))
        exact = 10 * math.log10(a * a * c * c / (4 * b * b))
        ellip_err.append(abs(got - exact))
        print(f"  {a:4.1f},{b:4.1f},{c:4.1f} {sf.shape[0]:7d} {got:+8.2f} "
              f"{exact:+8.2f} {got - exact:+6.2f}")

    banner("aspect and depression angle -- what the mesh actually says")
    def hull_ts(az_deg, el_deg):
        az, el = math.radians(az_deg), math.radians(el_deg)
        d = [-math.cos(az) * math.cos(el), math.sin(az) * math.cos(el),
             -math.sin(el)]
        return 10 * math.log10(max(float(_mono(hull, d, freqs)), 1e-30))

    fls_el = math.degrees(math.atan2(12.0 - 1.0, 60.0))
    els = [0.0, fls_el, 25.0, 60.0, 89.0]
    print(f"  aspect 0 = bow-on, 90 = beam-on; el = how far the sonar looks UP")
    print(f"  at it.  The FLS in examples/12 sits at {fls_el:.1f} deg.\n")
    print("  aspect |" + "".join(f"  el={e:4.1f}" for e in els))
    print("  -------+" + "-" * (9 * len(els)))
    table = {}
    for az in (0, 30, 45, 60, 75, 90):
        table[az] = [hull_ts(az, e) for e in els]
        print(f"  {az:5d}d |" + "".join(f"{x:+9.2f}" for x in table[az]))
    beneath = max(table[az][-1] for az in table)
    shallow = max(table[az][1] for az in table)
    worst_beneath = min(table[az][-1] for az in table)
    mesh_by_aspect = [hull_ts(a, fls_el) for a in
                      (0, 15, 30, 45, 60, 75, 90, 120, 150, 180)]

    bow_shallow = table[0][1]
    beam_shallow = max(table[75][1], table[90][1])
    beneath_spread = max(table[a][-1] for a in table) - min(table[a][-1] for a in table)
    print(f"\n  Two things fall out of this that the analytic patch could not say.")
    print(f"\n  1. At the depression angle a forward-looking sonar actually has,")
    print(f"     **aspect dominates**.  On the beam the hull is a strong target")
    print(f"     ({beam_shallow:+.1f} dB); head-on it is {bow_shallow:+.1f} dB, "
          f"{beam_shallow - bow_shallow:.0f} dB weaker.")
    print(f"     The reason is geometric: below the waterline a hull's outward")
    print(f"     normal tilts downward, and toward the bow it also swings")
    print(f"     forward, so there is no aspect where a shallow look finds the")
    print(f"     bow's specular point.")
    steep = [hull_ts(0.0, e) for e in (70.0, 80.0, 85.0, 88.0, 89.0)]
    print(f"\n  2. From underneath it is a **plate**, not a curved surface.")
    print(f"     A real hull has a flat run aft, and a flat surface seen near")
    print(f"     normal returns (A/lambda)^2 -- enormous, but through a lobe")
    print(f"     whose first null is at lambda/2L, "
          f"{math.degrees(LAM / (2 * HULL_LENGTH)) * 60:.1f} arcmin for a")
    print(f"     {HULL_LENGTH:.0f} m bottom.  Measured bow-on, TS climbs")
    print(f"     {steep[0]:+.0f} -> {steep[1]:+.0f} -> {steep[2]:+.0f} -> "
          f"{steep[3]:+.0f} -> {steep[4]:+.0f} dB from 70 to 89 deg,")
    print(f"     then rings violently inside the last degree.  So a")
    print(f"     downward-looking sonar gets a spectacular return off a hull")
    print(f"     -- and loses it for a degree of vehicle attitude.")
    print(f"\n     This is the opposite of what a round-bilged spindle would")
    print(f"     say, and getting it right needed the hull to *be* a hull:")
    print(f"     a transom aft, and deadrise that varies from nearly flat")
    print(f"     aft to a sharp V forward.")

    print(f"\n  Note this is the *bare faired hull*.  A real one also carries a")
    print(f"  chine, keel, skeg, shafts, propeller and rudder -- hard features")
    print(f"  returning over a wide angle, which is why examples/12 keeps them")
    print(f"  as separate highlights.  They are what fills in the bow aspects.")

    # ---- 3. through the sonar ---------------------------------------------- #
    banner("through the 100 kHz FLS, as a target like any other")
    fls = _fls()
    elements = fls.array(4)
    scene, bottom, surface, sediment = fls.build_scene(elements)
    boat = mesh_target(verts, faces, position=(fls.TARGET_RANGE, 0.0, fls.HULL_DEPTH),
                       yaw=90.0, n_patches=4, sound_speed=C,
                       learnable=True, learnable_shape=True, facet_chunk=4096)
    print(f"  mesh split into {boat.n_highlights} patches along the hull, "
          f"{boat.pattern_for(0).n_facets} facets each")
    tx_dirs, tx_w = fls.transmit(fls.TX_RAYS)
    t0 = time.perf_counter()
    # The return leg is SOLVED (method of images where the sound speed is
    # constant, traced rays otherwise), not splatted: the splat summed
    # acceptance weights over every ray passing a point without dividing by
    # their sum, +31 dB in the image.  The projector's pattern is then needed
    # as a function of direction, since a solved path has no ray to index.
    arrivals = target_arrivals(scene, boat, tx_dirs, n_rx_rays=fls.RX_RAYS,
                               rx_half_angle_deg=40.0, tx_weights=tx_w,
                               max_arrivals_per_leg=24,
                               return_leg="eigenray", tx_pattern=fls.transmit_pattern,
                               generator=torch.Generator().manual_seed(3))
    steer, bearings = azimuth_steering(121, fls.SECTOR_DEG)
    grid = make_time_grid(2 * (fls.TARGET_RANGE - 25.0) / C,
                          2 * (fls.TARGET_RANGE + 25.0) / C, 600)
    image = beamform(arrivals, elements, scene.freqs_khz, grid, steer,
                     sigma_t=3e-5, shading=shading_window(4, "hamming"),
                     steer_chunk=24)
    forward = time.perf_counter() - t0
    flat = int(image[:, 0].reshape(-1).detach().argmax())
    b_hat = float(bearings[flat // grid.shape[0]])
    r_hat = float(grid[flat % grid.shape[0]]) * C / 2.0
    print(f"  {arrivals.n_arrivals} arrivals, forward {forward:.2f} s")
    print(f"  detection at bearing {b_hat:+.2f} deg, range {r_hat:.2f} m "
          f"(true 0.00 deg, {fls.TARGET_RANGE:.1f} m)")

    banner("invertibility: the gradient reaches the geometry itself")
    t0 = time.perf_counter()
    image.sum().backward()
    backward = time.perf_counter() - t0
    vgrad = boat.pattern_for(0).vertices.grad
    shape_live = (vgrad is not None and torch.isfinite(vgrad).all()
                  and float(vgrad.abs().max()) > 0.0)
    pos_live = (boat.position.grad is not None
                and float(boat.position.grad.abs().sum()) > 0.0)
    print(f"  d(image)/d(mesh vertices): {'OK' if shape_live else 'ZERO'}"
          f"   max |grad| {float(vgrad.abs().max()):.3e}")
    print(f"  d(image)/d(boat position): {'OK' if pos_live else 'ZERO'}")
    print(f"  d(image)/d(seabed):        "
          f"{'OK' if float(bottom.heights.grad.abs().sum()) > 0 else 'ZERO'}")
    print(f"\n  forward {forward:.2f} s + backward {backward:.2f} s")
    print(f"  So the hull's *shape* is a free parameter: you can fit the geometry")
    print(f"  of an unknown body to its echo, not just its position and pose.")

    save(_plot(verts, faces, mesh_by_aspect, image, bearings, grid, fls),
         "14_mesh_boat.png")

    banner("acceptance")
    ok = check("a faceted sphere reproduces sigma = a^2/4",
               max(sphere_err) < 2.0,
               f"worst error {max(sphere_err):.2f} dB over three meshes")
    ok &= check("and it is aspect-independent, as the exact answer is",
                spread < 1.0, f"spread {spread:.2f} dB across 5 aspects")
    ok &= check("an ellipsoid reproduces sigma = A^2 C^2 / 4 B^2, separating R1 from R2",
                max(ellip_err) < 1.0, f"worst error {max(ellip_err):.2f} dB")
    ok &= check("the point-facet shortcut is visibly wrong on the same mesh",
                shortcut_err > 6.0, f"{shortcut_err:+.1f} dB against the exact answer")
    ok &= check("the hull is a strong target on the beam at the FLS's own angle",
                shallow > -6.0, f"{shallow:+.1f} dB at {fls_el:.0f} deg depression")
    ok &= check("from beneath, the flat run aft returns like a plate",
                max(steep) > 20.0,
                f"{max(steep):+.1f} dB near normal, against "
                f"{steep[0]:+.1f} dB at 70 deg")
    ok &= check("and that plate's lobe is narrow, as (A/lambda)^2 requires",
                max(steep) - steep[0] > 25.0,
                f"{max(steep) - steep[0]:.0f} dB fall from 89 to 70 deg")
    ok &= check("but at a shallow angle aspect dominates: bow-on is far weaker",
                beam_shallow - bow_shallow > 15.0,
                f"beam {beam_shallow:+.1f} dB against bow {bow_shallow:+.1f} dB")
    ok &= check("the mesh renders through the FLS and is detected",
                abs(r_hat - fls.TARGET_RANGE) < 5.0,
                f"{r_hat:.2f} m against {fls.TARGET_RANGE:.1f} m")
    ok &= check("a loss on the image reaches the mesh vertices", shape_live)
    ok &= check("and still reaches the scene around it", pos_live)
    return 0 if ok else 1


def _plot(verts, faces, aspect_ts, image, bearings, grid, fls):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 4.4))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    v = verts.detach().numpy()
    tri = v[faces.numpy()]
    ax.add_collection3d(Poly3DCollection(tri[::7], facecolor="#5566aa",
                                         edgecolor="none", alpha=0.85))
    ax.set_xlim(-7, 7); ax.set_ylim(-3.5, 3.5); ax.set_zlim(2, -2)
    ax.set_box_aspect((14, 7, 4))
    ax.set_title(f"wetted hull, {faces.shape[0]} facets", fontsize=10)
    ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    deg = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    ax2.plot(deg, aspect_ts, "o-", lw=1.4, ms=4)
    ax2.set_xlabel("aspect (deg; 0 = bow-on, 90 = beam-on)")
    ax2.set_ylabel("TS (dB)")
    ax2.set_title("aspect pattern, straight off the mesh", fontsize=10)
    ax2.grid(alpha=0.3, lw=0.4)

    ax3 = fig.add_subplot(1, 3, 3)
    rng = grid.detach().numpy() * C / 2.0
    a = image[:, 0].detach().numpy()
    db = 10 * np.log10(np.maximum(a, a.max() * 1e-4) / a.max())
    m = ax3.pcolormesh(rng, bearings.numpy(), db, cmap="inferno", vmin=-25,
                       vmax=0, shading="auto")
    ax3.set_xlabel("range (m)"); ax3.set_ylabel("bearing (deg)")
    ax3.set_title("the mesh through the 100 kHz FLS", fontsize=10)
    fig.colorbar(m, ax=ax3, label="dB re peak")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
