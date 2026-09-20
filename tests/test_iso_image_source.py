"""Isovelocity Pekeris waveguide: arrival times and bounce counts must match the
3-D image-source solution exactly.

With constant ``c``, a flat surface at ``z = 0`` and a flat bottom at ``z = H``,
unfolding the waveguide replaces every reflected path by a straight line to an
*image* of the receiver at depth ``2 m H +/- z_r``.  A ray launched straight at
such an image must arrive at the real receiver after ``L / c`` seconds, having
bounced once for every multiple of ``H`` its unfolded vertical coordinate
crossed: even multiples are surface bounces, odd ones bottom bounces.
"""

import math

import pytest
import torch

from hydropt import ConstantLoss, FlatHeight, IsoProfile, Scene

C, H, Z_SRC, Z_RCV, RANGE = 1500.0, 200.0, 50.0, 120.0, 3000.0
# Incommensurate losses so (n_surface, n_bottom) is uniquely recoverable from
# the accumulated dB -- this is how the bounce counts are asserted.
L_SURFACE, L_BOTTOM = 1.0, 10.0


def _images(m_max: int = 2) -> list[float]:
    return sorted({2 * m * H + s * Z_RCV for m in range(-m_max, m_max + 1) for s in (1, -1)})


def _expected_bounces(z_image: float) -> tuple[int, int]:
    """Surface/bottom bounce counts for the unfolded path ``Z_SRC -> z_image``."""
    lo, hi = sorted((Z_SRC, z_image))
    crossings = [k for k in range(-20, 21) if lo < k * H < hi]
    n_surface = sum(1 for k in crossings if k % 2 == 0)
    return n_surface, len(crossings) - n_surface


def _scene(n_steps: int = 900, step: float = 5.0) -> Scene:
    # step * n_steps must exceed the longest unfolded image path (~3.2 km here),
    # otherwise a ray is still short of the receiver when it runs out of steps.
    return Scene(
        field=IsoProfile(C), bottom=FlatHeight(H), surface=FlatHeight(0.0),
        source=(0.0, 0.0, Z_SRC), receivers=torch.tensor([[RANGE, 0.0, Z_RCV]]),
        surface_loss=ConstantLoss(L_SURFACE, learnable=False),
        bottom_loss=ConstantLoss(L_BOTTOM, learnable=False),
        step_size=step, n_steps=n_steps,
    )


def _aim_at_images() -> tuple[torch.Tensor, list[tuple[float, float, int, int]]]:
    dirs, meta = [], []
    for z_image in _images():
        v = torch.tensor([RANGE, 0.0, z_image - Z_SRC])
        length = float(v.norm())
        n_surface, n_bottom = _expected_bounces(z_image)
        dirs.append(v / v.norm())
        meta.append((length, length / C, n_surface, n_bottom))
    return torch.stack(dirs), meta


def _closest_approach(result, receiver):
    """Interpolated (miss distance, arrival time, accumulated dB) per ray."""
    p0, p1 = result.pos[:, :-1], result.pos[:, 1:]
    seg = p1 - p0
    t = (((receiver - p0) * seg).sum(-1) / (seg * seg).sum(-1).clamp_min(1e-30)).clamp(0, 1)
    dist = (p0 + t.unsqueeze(-1) * seg - receiver).norm(dim=-1)
    k = dist.argmin(1)
    r = torch.arange(dist.shape[0])
    tk = t[r, k]
    tau = result.tau[r, k] + tk * (result.tau[r, k + 1] - result.tau[r, k])
    return dist[r, k], tau, result.refl_db[r, k + 1]


def test_rays_aimed_at_images_hit_the_receiver():
    scene = _scene()
    dirs, meta = _aim_at_images()
    miss, _, _ = _closest_approach(scene.trace(dirs), scene.receivers[0])
    # Sub-millimetre: the only error is the O(h^2) chord used at each bounce.
    assert miss.max().item() < 1e-3


def test_arrival_times_match_the_image_source_solution():
    scene = _scene()
    dirs, meta = _aim_at_images()
    _, tau, _ = _closest_approach(scene.trace(dirs), scene.receivers[0])
    for i, (_, tau_theory, _, _) in enumerate(meta):
        assert tau[i].item() == pytest.approx(tau_theory, abs=1e-9)


def test_bounce_counts_match_the_image_source_solution():
    """Decoded from the accumulated reflection loss, which is what actually
    scales the energy -- a bounce that is counted but not charged for would be
    a silent physics bug."""
    scene = _scene()
    dirs, meta = _aim_at_images()
    _, _, refl_db = _closest_approach(scene.trace(dirs), scene.receivers[0])
    for i, (_, _, n_surface, n_bottom) in enumerate(meta):
        expected = n_surface * L_SURFACE + n_bottom * L_BOTTOM
        assert refl_db[i].item() == pytest.approx(expected, abs=1e-9)
        # Unique decode because L_BOTTOM / L_SURFACE = 10 > max bounce count.
        decoded_bottom = round(refl_db[i].item() // L_BOTTOM)
        assert decoded_bottom == n_bottom


def test_total_bounce_counts_are_reported():
    """The counters on TraceResult must agree with the charged loss."""
    scene = _scene()
    dirs, _ = _aim_at_images()
    res = scene.trace(dirs)
    charged = res.n_surface * L_SURFACE + res.n_bottom * L_BOTTOM
    assert torch.allclose(charged, res.refl_db[:, -1], atol=1e-9)
    assert torch.equal(res.bounces(), (res.n_surface + res.n_bottom).long())


def test_etc_peaks_land_on_image_source_arrival_times():
    """The rendered energy-time curve, not just the raw geometry, must put its
    peaks at the image-source arrivals."""
    from hydropt import make_time_grid, spherical_fan

    scene = _scene(n_steps=400, step=10.0)
    scene.freqs_khz = torch.tensor([0.2])
    grid = make_time_grid(1.99, 2.02, 1500)
    etc = scene.render(spherical_fan(600, 1, (-35, 35), (0, 0)), grid,
                       sigma_d=30.0, sigma_t=3e-4)[0, 0]

    peaks = [grid[i].item() for i in range(1, len(etc) - 1)
             if etc[i] > etc[i - 1] and etc[i] > etc[i + 1] and etc[i] > etc.max() * 0.03]
    theory = sorted(math.hypot(RANGE, z - Z_SRC) / C for z in _images(1))
    theory = [t for t in theory if grid[0] < t < grid[-1]]

    assert len(peaks) >= len(theory)
    for t in theory:
        # A fan is discrete, so the nearest launch angle is not exactly the
        # eigenray; it passes at a small miss distance and arrives marginally
        # early.  0.2 ms at 3 km is well inside that discretisation error.
        assert min(abs(p - t) for p in peaks) < 2e-4
