"""The extraction must measure the field, not how densely the fan was sampled.

:func:`hydropt.beamform.extract_arrivals` weights a ray by
``exp(-0.5 (d / sigma_d)^2)`` on its miss distance ``d``, with no normalisation
for the fan's density.  A fixed scalar ``sigma_d`` is therefore only correct for
one fan density at one range: make the fan sparser than ``sigma_d`` and the
amplitude stops measuring the field and starts measuring whether a ray happened
to pass close by.

This was not hypothetical.  With the settings ``examples/12`` and ``13`` shipped
with -- 400 return rays over a 40 degree cone, ``sigma_d = 0.4`` m, target at 60 m
-- the rays are 3.5 m apart where the splat is 0.4 m wide, and eight *identical*
highlights returned energies spanning **30 dB**.

The fix is :func:`hydropt.launch.fan_sigma_d`: size the splat to the fan's own
ray spacing, so densifying the fan narrows the splat in step and the answer
converges.  These tests pin both halves -- the geometry helper, and the
invariance it buys end to end.
"""

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, ExtendedTarget, FlatHeight, IsoProfile, IsotropicScattering,
    Scene, target_arrivals,
)
from hydropt.launch import fan_angular_spacing, fan_sigma_d, fibonacci_cone

C = 1500.0
RANGE = 60.0


# --------------------------------------------------------------------------- #
# fan_angular_spacing
# --------------------------------------------------------------------------- #
def test_spacing_of_two_rays_is_proportional_to_their_separation():
    """With only a pair, the neighbourhood is one ray and the density estimate
    is ``r_1 sqrt(pi)`` -- crude, but it must still scale with the angle."""
    for a in (math.radians(10.0), math.radians(30.0)):
        dirs = torch.tensor([[1.0, 0.0, 0.0], [math.cos(a), math.sin(a), 0.0]])
        got = fan_angular_spacing(dirs)
        assert torch.allclose(got, torch.full((2,), a * math.sqrt(math.pi)),
                              atol=1e-12)


def test_spacing_measures_density_not_regularity():
    """The defect this estimator exists to avoid.

    A nearest-neighbour distance reads an irregular fan as about twice as dense
    as a regular fan of the same size, because in an irregular set some pair is
    always closer than average.  A sigma_d built on that comes out half as wide
    and throws away more than half the energy.  The density estimate has to give
    the same answer for both.
    """
    axis = torch.tensor([1.0, 0.0, 0.0])
    for n in (400, 1600):
        regular = fibonacci_cone(n, axis, 40.0)
        jittered = fibonacci_cone(n, axis, 40.0, jitter=1.0,
                                  generator=torch.Generator().manual_seed(1))
        a = float(fan_angular_spacing(regular).median())
        b = float(fan_angular_spacing(jittered).median())
        assert b / a == pytest.approx(1.0, abs=0.12), (n, a, b)


def test_spacing_is_insensitive_to_the_neighbour_count():
    dirs = fibonacci_cone(900, torch.tensor([1.0, 0.0, 0.0]), 30.0)
    a = float(fan_angular_spacing(dirs, neighbours=4).median())
    b = float(fan_angular_spacing(dirs, neighbours=16).median())
    assert b / a == pytest.approx(1.0, abs=0.15)


def test_spacing_ignores_a_rays_own_entry():
    # Three coincident rays would read spacing 0; distinct ones must not read 0.
    dirs = fibonacci_cone(64, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    assert float(fan_angular_spacing(dirs).min()) > 0.0


def test_spacing_is_chunk_invariant():
    dirs = fibonacci_cone(300, torch.tensor([1.0, 0.0, 0.0]), 25.0)
    a = fan_angular_spacing(dirs, chunk=7)
    b = fan_angular_spacing(dirs, chunk=4096)
    assert torch.equal(a, b)


def test_spacing_is_normalisation_invariant():
    dirs = fibonacci_cone(200, torch.tensor([1.0, 0.0, 0.0]), 25.0)
    a = fan_angular_spacing(dirs)
    b = fan_angular_spacing(dirs * 7.5)  # direction, not length, is what counts
    assert torch.allclose(a, b, atol=1e-12)


def test_spacing_matches_the_solid_angle_estimate():
    # N rays spread over Omega steradians sit about sqrt(Omega / N) apart.
    half = 20.0
    omega = 2.0 * math.pi * (1.0 - math.cos(math.radians(half)))
    for n in (500, 2000):
        dirs = fibonacci_cone(n, torch.tensor([1.0, 0.0, 0.0]), half)
        got = float(fan_angular_spacing(dirs).median())
        assert 0.5 < got / math.sqrt(omega / n) < 1.5


def test_spacing_halves_when_the_fan_quadruples():
    axis = torch.tensor([1.0, 0.0, 0.0])
    coarse = float(fan_angular_spacing(fibonacci_cone(500, axis, 20.0)).median())
    fine = float(fan_angular_spacing(fibonacci_cone(2000, axis, 20.0)).median())
    assert coarse / fine == pytest.approx(2.0, rel=0.1)


def test_spacing_of_a_single_ray_is_finite():
    got = fan_angular_spacing(torch.tensor([[1.0, 0.0, 0.0]]))
    assert got.shape == (1,) and math.isfinite(float(got))


def test_spacing_carries_no_gradient():
    dirs = fibonacci_cone(32, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    dirs.requires_grad_(True)
    assert not fan_angular_spacing(dirs).requires_grad


# --------------------------------------------------------------------------- #
# fan_sigma_d
# --------------------------------------------------------------------------- #
def test_sigma_d_is_spacing_times_range():
    dirs = fibonacci_cone(200, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    arclen = torch.linspace(0.0, 100.0, 5).expand(200, 5).contiguous()
    got = fan_sigma_d(dirs, arclen)
    want = fan_angular_spacing(dirs).reshape(-1, 1) * arclen
    assert torch.allclose(got, want, atol=1e-12)
    assert got.shape == (200, 5)


def test_sigma_d_scales_with_factor_and_respects_floor():
    dirs = fibonacci_cone(100, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    arclen = torch.full((100, 3), 50.0)
    one = fan_sigma_d(dirs, arclen)
    two = fan_sigma_d(dirs, arclen, factor=2.0)
    assert torch.allclose(two, 2.0 * one, atol=1e-12)
    floored = fan_sigma_d(dirs, torch.zeros(100, 3), floor=0.25)
    assert torch.allclose(floored, torch.full((100, 3), 0.25), atol=1e-12)


def test_sigma_d_rejects_a_mismatched_fan():
    dirs = fibonacci_cone(10, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    with pytest.raises(ValueError, match="rays"):
        fan_sigma_d(dirs, torch.zeros(11, 4))


def test_sigma_d_narrows_as_the_fan_densifies():
    axis = torch.tensor([1.0, 0.0, 0.0])
    arclen = torch.full((1, 2), RANGE)
    coarse = float(fan_sigma_d(fibonacci_cone(400, axis, 40.0),
                               arclen.expand(400, 2)).median())
    fine = float(fan_sigma_d(fibonacci_cone(1600, axis, 40.0),
                             arclen.expand(1600, 2)).median())
    assert fine < coarse
    assert coarse / fine == pytest.approx(2.0, rel=0.15)


def test_the_settings_that_broke_are_visibly_mismatched():
    """The defect is diagnosable from the fan alone, before any tracing."""
    axis = torch.tensor([1.0, 0.0, 0.0])
    rx = fan_sigma_d(fibonacci_cone(400, axis, 40.0),
                     torch.full((400, 2), RANGE))
    assert float(rx.median()) > 8.0 * 0.4   # what examples 12/13 used
    tx = fan_sigma_d(fibonacci_cone(3000, axis, 14.0),
                     torch.full((3000, 2), RANGE))
    assert 0.25 < float(tx.median()) < 0.8  # the transmit leg was fine


# --------------------------------------------------------------------------- #
# end to end: identical highlights must return identical energies
# --------------------------------------------------------------------------- #
def _scene() -> Scene:
    return Scene(
        field=IsoProfile(C),
        bottom=FlatHeight(1e5),
        surface=FlatHeight(-1e5),
        source=(0.0, 0.0, 12.0),
        receivers=torch.tensor([[0.0, -0.01, 12.0], [0.0, 0.01, 12.0]]),
        surface_loss=ConstantLoss(0.3, learnable=False),
        bottom_loss=ConstantLoss(4.0, learnable=False),
        freqs_khz=torch.tensor([100.0]),
        step_size=2.0,
        n_steps=60,
        max_bounces=0,
    )


def _row_of_identical_highlights(n: int = 5) -> ExtendedTarget:
    xs = torch.linspace(-4.0, 4.0, n)
    hl = torch.stack([xs, torch.zeros(n), torch.zeros(n)], dim=-1)
    return ExtendedTarget(hl, IsotropicScattering(0.0, learnable=False),
                          position=(RANGE, 0.0, 12.0), yaw=90.0, learnable=False)


def _energies(sigma_d, n_rx: int, n: int = 5):
    tx = fibonacci_cone(2000, torch.tensor([1.0, 0.0, 0.0]), 14.0)
    scene, target = _scene(), _row_of_identical_highlights(n)
    out = []
    for k in range(n):
        one = ExtendedTarget(target.highlights[k:k + 1],
                             IsotropicScattering(0.0, learnable=False),
                             position=(RANGE, 0.0, 12.0), yaw=90.0,
                             learnable=False)
        a = target_arrivals(scene, one, tx, sigma_d=sigma_d, n_rx_rays=n_rx,
                            rx_half_angle_deg=40.0, max_arrivals_per_leg=4,
                            generator=torch.Generator().manual_seed(0))
        out.append(float((a.amplitude.detach() ** 2).sum()))
    return torch.tensor(out)


def _spread_db(e: torch.Tensor) -> float:
    e = e.clamp_min(1e-300)
    return float(10.0 * torch.log10(e.max() / e.min()))


@pytest.mark.parametrize("n_rx", [400, 900])
def test_identical_highlights_return_equal_energies(n_rx):
    """In a homogeneous half-space at equal range, the only honest answer."""
    e = _energies(None, n_rx)
    assert (e > 0).all()
    assert _spread_db(e) < 1.0


def test_a_fixed_sigma_d_throws_the_energy_away():
    """A splat narrower than the ray spacing catches almost nothing.

    This is the mechanism behind the 30 dB spread, and it is the robust half:
    *how much* a given highlight loses depends on where its nearest ray happened
    to fall, but that nearly all of it is lost does not.  Here a 0.4 m splat on a
    fan whose rays are 3.5 m apart keeps under a thousandth of the energy the
    matched width finds.
    """
    auto = float(_energies(None, 400).sum())
    fixed = float(_energies(0.4, 400).sum())
    assert auto / fixed > 100.0


def test_result_is_invariant_to_return_fan_density():
    """Densify the fan 4x; the answer must not move."""
    coarse = _energies(None, 400).sum()
    fine = _energies(None, 1600).sum()
    assert float(fine / coarse) == pytest.approx(1.0, rel=0.10)


def test_a_fixed_sigma_d_is_not_invariant_to_fan_density():
    coarse = float(_energies(0.4, 400).sum())
    fine = float(_energies(0.4, 1600).sum())
    assert max(coarse, fine) / min(coarse, fine) > 2.0


def test_automatic_width_keeps_the_gradient():
    """The default must not cost differentiability -- the whole point of it."""
    scene = _scene()
    target = ExtendedTarget(torch.zeros(1, 3), IsotropicScattering(0.0),
                            position=(RANGE, 0.0, 12.0), learnable=True)
    tx = fibonacci_cone(1500, torch.tensor([1.0, 0.0, 0.0]), 14.0)
    a = target_arrivals(scene, target, tx, n_rx_rays=400, rx_half_angle_deg=40.0,
                        max_arrivals_per_leg=4,
                        generator=torch.Generator().manual_seed(0))
    assert a.n_arrivals > 0
    (a.amplitude ** 2).sum().backward()
    g = target.position.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0.0


def test_sigma_d_factor_widens_the_splat():
    narrow = _energies(None, 400).sum()
    scene, tx = _scene(), fibonacci_cone(2000, torch.tensor([1.0, 0.0, 0.0]), 14.0)
    one = ExtendedTarget(torch.zeros(1, 3), IsotropicScattering(0.0, learnable=False),
                         position=(RANGE, 0.0, 12.0), learnable=False)
    a = target_arrivals(scene, one, tx, n_rx_rays=400, rx_half_angle_deg=40.0,
                        sigma_d_factor=2.0, max_arrivals_per_leg=4,
                        generator=torch.Generator().manual_seed(0))
    b = target_arrivals(scene, one, tx, n_rx_rays=400, rx_half_angle_deg=40.0,
                        sigma_d_factor=1.0, max_arrivals_per_leg=4,
                        generator=torch.Generator().manual_seed(0))
    assert float((a.amplitude ** 2).sum()) > float((b.amplitude ** 2).sum())
    assert narrow > 0


# --------------------------------------------------------------------------- #
# rx_jitter: making `generator` mean something
# --------------------------------------------------------------------------- #
"""A Fibonacci cone is deterministic.  ``target_arrivals`` took a ``generator``
and documented it as the RNG for the return fans, but never asked for any
jitter, so every seed produced a bit-identical fan and a bit-identical answer.
That is worse than a no-op: it makes an inversion look like it avoided the
inverse crime when the model and the synthetic measurement in fact shared their
sampling exactly.
"""


def _one_target_energy(seed, jitter, n_rx=200):
    scene = _scene()
    target = ExtendedTarget(torch.zeros(1, 3),
                            IsotropicScattering(0.0, learnable=False),
                            position=(RANGE, 0.0, 12.0), learnable=False)
    tx = fibonacci_cone(600, torch.tensor([1.0, 0.0, 0.0]), 14.0)
    a = target_arrivals(scene, target, tx, n_rx_rays=n_rx,
                        rx_half_angle_deg=40.0, rx_jitter=jitter,
                        max_arrivals_per_leg=6,
                        generator=torch.Generator().manual_seed(seed))
    return float((a.amplitude.detach() ** 2).sum())


def test_without_jitter_the_generator_does_nothing():
    """Pinned as the documented behaviour, not as an accident."""
    assert _one_target_energy(1, 0.0) == _one_target_energy(2, 0.0)


def test_with_jitter_different_seeds_are_different_realisations():
    a, b = _one_target_energy(1, 1.0), _one_target_energy(2, 1.0)
    assert a != b
    assert a > 0 and b > 0


def test_jitter_is_repeatable_for_a_given_seed():
    assert _one_target_energy(5, 1.0) == _one_target_energy(5, 1.0)


def test_jitter_does_not_bias_the_answer():
    """It re-samples the fan; it must not change what the fan is estimating.

    Averaged over realisations the jittered result has to agree with the
    unjittered one, or 'avoiding the inverse crime' would mean fitting a
    different scene.
    """
    ref = _one_target_energy(0, 0.0, n_rx=900)
    draws = [_one_target_energy(s, 1.0, n_rx=900) for s in range(8)]
    mean = sum(draws) / len(draws)
    assert mean == pytest.approx(ref, rel=0.25)


def test_realisation_noise_does_not_fall_with_fan_density():
    """A property worth knowing before spending rays on it.

    Because ``sigma_d`` is sized to the fan, densifying the fan narrows the
    splat in step and the number of rays *effectively* contributing to an
    arrival stays about the same.  So the spread between independent
    realisations is set by that effective count, not by the fan size, and
    throwing rays at the problem does not make the answer quieter -- it only
    makes it converge, which is a different thing and is pinned above.

    This is why pose recovered from these images stops improving once the fan
    is adequate: the residual is realisation noise, not ray count.
    """
    def spread(n_rx):
        draws = [_one_target_energy(s, 1.0, n_rx=n_rx) for s in range(6)]
        m = sum(draws) / len(draws)
        return (sum((x - m) ** 2 for x in draws) / len(draws)) ** 0.5 / m

    coarse, fine = spread(300), spread(2400)
    assert coarse > 0.0 and fine > 0.0
    assert fine / coarse == pytest.approx(1.0, abs=0.6), (coarse, fine)


# --------------------------------------------------------------------------- #
# The transmit fan has to contain the target, not merely point at it
# --------------------------------------------------------------------------- #
def _lattice_fan(half_elev_deg: float, half_azim_deg: float, n_elev: int,
                 n_azim: int, seed: int = 0) -> torch.Tensor:
    """A jittered lattice about +x, the shape a real fan is sampled on."""
    g = torch.Generator().manual_seed(seed)
    el = torch.linspace(-math.radians(half_elev_deg), math.radians(half_elev_deg),
                        n_elev)
    az = torch.linspace(-math.radians(half_azim_deg), math.radians(half_azim_deg),
                        n_azim)
    e, a = torch.meshgrid(el, az, indexing="ij")
    e, a = e.reshape(-1), a.reshape(-1)
    if n_elev > 1:
        e = e + (torch.rand(e.shape, generator=g, dtype=e.dtype) - 0.5) * (el[1] - el[0])
    if n_azim > 1:
        a = a + (torch.rand(a.shape, generator=g, dtype=a.dtype) - 0.5) * (az[1] - az[0])
    return torch.stack([e.cos() * a.cos(), e.cos() * a.sin(), e.sin()], dim=-1)


def _transmit_energy(half_azim_deg: float, n_elev: int, n_azim: int) -> float:
    """Echo energy off a body 8 m long across the line of sight, at 40 m."""
    scene = _scene()
    target = ExtendedTarget(torch.tensor([[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]),
                            IsotropicScattering(0.0, learnable=False),
                            position=(RANGE, 0.0, 12.0), yaw=90.0,
                            learnable=False)
    tx = _lattice_fan(6.0, half_azim_deg, n_elev, n_azim)
    a = target_arrivals(scene, target, tx, n_rx_rays=400,
                        rx_half_angle_deg=40.0, max_arrivals_per_leg=8,
                        generator=torch.Generator().manual_seed(0))
    return float((a.amplitude.detach() ** 2).sum())


def test_result_is_invariant_to_transmit_fan_density():
    """Densify the transmit fan 16x; the level must not move.

    The return fan's invariance is pinned above; this is the outbound half,
    and it is the one a caller controls directly when they choose how finely to
    sample their projector.
    """
    coarse = _transmit_energy(12.0, 24, 16)
    fine = _transmit_energy(12.0, 96, 64)
    assert 10.0 * abs(math.log10(fine / coarse)) < 1.5


def test_a_target_outside_the_fan_window_collapses_as_the_fan_densifies():
    """The trap: a fan aimed at a body but narrower than it.

    Each highlight is a point, and its echo is a Gaussian in the miss distance
    of the rays passing it, with a width matched to the fan's spacing.  A
    highlight outside the sampled window is reached only by the rays at the
    edge, at an offset that does not shrink -- so refining the fan narrows the
    splat while the offset stays put, and the level falls exponentially with
    the very ray count that should have improved it.

    An 8 m body at 40 m spans +/- 5.7 degrees.  Sampled over +/- 1.5 it does
    not merely lose level under the same 16x refinement that moves it by under
    1.5 dB with the window open -- the echo goes to **zero**, because at the
    finer spacing the splat is narrow enough that the gate rejects every ray
    the body has.  Nothing warns you: the fan is pointed straight at the
    target and every ray in it is legitimate.
    """
    coarse = _transmit_energy(1.5, 24, 16)
    fine = _transmit_energy(1.5, 96, 64)
    assert coarse > 0.0
    assert fine < coarse / 10.0
