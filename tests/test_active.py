"""Two-way active sonar: echo timing, target strength, and the convolution.

The separable two-leg composition in :mod:`hydropt.active` is only worth having
if it reproduces what an explicit two-way trace would give, so these tests pin
it to closed-form geometry: an echo must arrive at ``(d_in + d_out) / c``, every
in/out multipath pairing must appear at its own predicted time, and the target
strength must scale the result by exactly ``10 ** (TS / 10)``.
"""

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, Scene, make_time_grid,
    PiecewiseLinearProfile,
)
from hydropt.active import PointTarget, render_echo, return_fan
from hydropt.launch import fibonacci_cone

C = 1500.0
WATER = 30.0
SOURCE = (0.0, 0.0, 10.0)
ARRAY = (0.5, 0.0, 10.0)
TARGET = (40.0, 0.0, 14.0)


def _scene(*, bounded: bool = False, **overrides) -> Scene:
    kw = dict(
        field=IsoProfile(C),
        bottom=FlatHeight(WATER if bounded else 1e5),
        surface=FlatHeight(0.0 if bounded else -1e5),
        source=SOURCE,
        receivers=torch.tensor([list(ARRAY)]),
        surface_loss=ConstantLoss(0.3, learnable=False),
        bottom_loss=ConstantLoss(4.0, learnable=False),
        freqs_khz=torch.tensor([50.0]),
        step_size=0.25,
        n_steps=600,
        max_bounces=4,
    )
    kw.update(overrides)
    return Scene(**kw)


def _fans(target: PointTarget, receivers: torch.Tensor, n: int = 6000,
          half_angle: float = 25.0):
    tx = fibonacci_cone(n, torch.tensor([1.0, 0.0, 0.0]), half_angle)
    rx = return_fan(target, receivers, n, half_angle_deg=half_angle)
    return tx, rx


def _echo(scene, target, grid, *, n: int = 6000, half_angle: float = 25.0,
          sigma_d: float = 0.6, sigma_t: float = 2e-5, **kw):
    tx, rx = _fans(target, scene.receivers, n, half_angle)
    return render_echo(scene, target, tx, rx, grid, sigma_d=sigma_d, sigma_t=sigma_t,
                       ray_chunk=2000, **kw)


def test_echo_arrives_at_the_two_way_travel_time():
    scene = _scene()
    target = PointTarget(TARGET, 0.0, learnable=False)
    d_in = math.dist(SOURCE, TARGET)
    d_out = math.dist(TARGET, ARRAY)
    expected = (d_in + d_out) / C

    grid = make_time_grid(expected - 0.004, expected + 0.004, 1600)
    etc = _echo(scene, target, grid).etc[0, 0]
    peak = grid[etc.argmax()].item()
    # The residual is ray-fan discretisation: the nearest launch direction is
    # not exactly the eigenray, so it passes at a small miss distance.
    assert peak == pytest.approx(expected, abs=5e-5)


def test_monostatic_echo_arrives_at_twice_the_one_way_time():
    scene = _scene(receivers=torch.tensor([list(SOURCE)]))
    target = PointTarget(TARGET, 0.0, learnable=False)
    expected = 2.0 * math.dist(SOURCE, TARGET) / C
    grid = make_time_grid(expected - 0.004, expected + 0.004, 1600)
    etc = _echo(scene, target, grid).etc[0, 0]
    assert grid[etc.argmax()].item() == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("ts_db", [-20.0, -10.0, 0.0, 10.0])
def test_target_strength_scales_the_echo_exactly(ts_db):
    scene = _scene()
    d = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C
    grid = make_time_grid(d - 0.004, d + 0.004, 1200)

    reference = _echo(scene, PointTarget(TARGET, 0.0, learnable=False), grid).etc
    scaled = _echo(scene, PointTarget(TARGET, ts_db, learnable=False), grid).etc
    assert torch.allclose(scaled, reference * 10.0 ** (ts_db / 10.0), rtol=1e-12)


def test_every_multipath_pairing_appears_at_its_predicted_time():
    """With a surface and a seabed, each inbound path pairs with each outbound
    path, and the echo carries the full outer product of their delays."""
    scene = _scene(bounded=True)
    target = PointTarget(TARGET, 0.0, learnable=False)

    def one_way(a, b):
        # Direct and surface-reflected images only: a bottom bounce here needs a
        # 42 deg launch angle, outside the transmit cone used below.
        return [math.dist(a, (b[0], b[1], z)) for z in (b[2], -b[2])]

    pairings = sorted({(p + q) / C for p in one_way(SOURCE, TARGET)
                       for q in one_way(TARGET, ARRAY)})
    grid = make_time_grid(0.050, 0.060, 2000)
    etc = _echo(scene, target, grid, n=8000, half_angle=35.0).etc[0, 0]

    peaks = [grid[i].item() for i in range(1, len(etc) - 1)
             if etc[i] > etc[i - 1] and etc[i] > etc[i + 1] and etc[i] > etc.max() * 0.02]
    for t in pairings:
        if not (grid[0] < t < grid[-1]):
            continue
        assert min(abs(p - t) for p in peaks) < 5e-5, f"no echo near {t * 1000:.3f} ms"


def test_time_kernel_composes_to_the_requested_width():
    """Each leg is rendered at sigma_t/sqrt(2) precisely so that convolving the
    two lands on sigma_t -- not sigma_t*sqrt(2)."""
    scene = _scene()
    target = PointTarget(TARGET, 0.0, learnable=False)
    sigma_t = 4e-5
    d = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C
    grid = make_time_grid(d - 0.002, d + 0.002, 4000)

    tx, rx = _fans(target, scene.receivers)
    etc = render_echo(scene, target, tx, rx, grid, sigma_d=0.6, sigma_t=sigma_t,
                      ray_chunk=2000).etc[0, 0]

    # Second moment of a Gaussian of width sigma is sigma^2.
    w = etc / etc.sum()
    mean = (w * grid).sum()
    width = ((w * (grid - mean) ** 2).sum()).sqrt().item()
    assert width == pytest.approx(sigma_t, rel=0.15)


def test_convolution_is_linear_not_circular():
    """A late echo must not wrap onto an early bin.  Zero-padding the FFT to at
    least 2n-1 is what prevents it; without that the two-way response of a
    distant target reappears at short range as a phantom."""
    scene = _scene()
    target = PointTarget(TARGET, 0.0, learnable=False)
    earliest = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C

    # Window opens well before any physically possible echo.
    grid = make_time_grid(0.0, earliest + 0.004, 3000)
    etc = _echo(scene, target, grid).etc[0, 0]
    before = etc[grid < earliest - 0.001]
    assert before.max() < etc.max() * 1e-9, "energy appeared before the earliest echo"


def test_gradients_reach_the_target_and_the_scene_through_both_legs():
    # Deliberately small.  The claim under test is that gradients *reach* every
    # parameter through both legs, which a few hundred rays settle.  A backward
    # pass at the fan sizes used elsewhere in this file is tens of millions of
    # field evaluations -- minutes of runtime for no extra assurance.
    scene = _scene(bounded=True,
                   field=PiecewiseLinearProfile([0.0, 15.0, WATER],
                                                [1512.0, 1505.0, 1503.0]),
                   bottom_loss=ConstantLoss(4.0),
                   step_size=1.0, n_steps=140)
    target = PointTarget(TARGET, 0.0)
    d = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C
    grid = make_time_grid(d - 0.004, d + 0.006, 400)

    _echo(scene, target, grid, n=500, half_angle=30.0,
          sigma_d=3.0, sigma_t=2e-4).etc.sum().backward()

    for name, p in (("target.position", target.position),
                    ("target.target_strength_db", target.target_strength_db),
                    ("field.values", scene.field.values),
                    ("bottom_loss.loss_db", scene.bottom_loss.loss_db)):
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is non-finite"
        assert p.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_relocated_scene_view_shares_every_parameter():
    """Leg 2 traces from a proxy whose source is the target.  If that proxy
    copied sub-modules instead of sharing them, gradients from the outbound leg
    would silently vanish."""
    from hydropt.active import _RelocatedScene

    scene = _scene(bounded=True, bottom_loss=ConstantLoss(4.0))
    moved = _RelocatedScene(scene, torch.tensor(list(TARGET)))
    assert moved.field is scene.field
    assert moved.bottom is scene.bottom
    assert moved.bottom_loss is scene.bottom_loss
    assert moved.step_size == scene.step_size
    assert torch.equal(moved.source_position(), torch.tensor(list(TARGET)))


def test_echo_result_exposes_both_legs():
    scene = _scene()
    target = PointTarget(TARGET, 0.0, learnable=False)
    d = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C
    grid = make_time_grid(d - 0.004, d + 0.004, 1200)
    result = _echo(scene, target, grid)

    assert result.inbound.shape[0] == 1
    assert result.outbound.shape[0] == scene.receivers.shape[0]
    assert result.inbound.max() > 0 and result.outbound.max() > 0
    # Each leg peaks at its own one-way time.
    t_in = result.leg_time_grid[result.inbound[0, 0].argmax()].item()
    assert t_in == pytest.approx(math.dist(SOURCE, TARGET) / C, abs=5e-5)
