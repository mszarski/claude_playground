"""Seabed reverberation, pinned to the classical flat-bottom result.

Lambert scattering off a flat bottom under a monostatic sonar has a closed-form
answer: the reverberation energy density decays as ``r^-5``, with absolute level
``pi c mu H^2 / r^5``.  Matching the *slope* only would show the geometry is
self-consistent; matching the level as well is what checks the solid-angle
bookkeeping and the ``1/sin(theta)`` grazing projection of each ray's footprint.
"""

import math

import numpy as np
import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, Scene, make_time_grid,
)
from hydropt.beamform import azimuth_steering, beamform, ArrivalSet
from hydropt.launch import fibonacci_sphere
from hydropt.reverb import (
    LambertScattering, cone_solid_angle, render_reverberation,
    reverberation_arrivals,
)

C = 1500.0
DEPTH = 50.0


def _downgoing(n: int = 120_000):
    dirs = fibonacci_sphere(n)
    return dirs[dirs[:, 2] > 0.02], 2.0 * math.pi / (n / 2)


def _scene(**overrides) -> Scene:
    kw = dict(
        field=IsoProfile(C), bottom=FlatHeight(DEPTH), surface=FlatHeight(-1e5),
        source=(0.0, 0.0, 0.0), freqs_khz=torch.tensor([1.0]),
        bottom_loss=ConstantLoss(0.0, learnable=False),
        step_size=2.0, n_steps=400, max_bounces=1,
    )
    kw.update(overrides)
    return Scene(**kw)


def _reverb(strength_db: float = -27.0, *, learnable: bool = False, **kw):
    dirs, omega = _downgoing()
    scene = _scene(**kw)
    scattering = LambertScattering(strength_db, learnable=learnable)
    result = scene.trace(dirs)
    arrivals = reverberation_arrivals(result, dirs, scene.freqs_khz,
                                      scattering=scattering,
                                      solid_angle_per_ray=omega, boundary="bottom",
                                      surface=scene.surface, bottom=scene.bottom)
    return scene, scattering, arrivals


def test_cone_solid_angle_matches_the_closed_form():
    assert cone_solid_angle(180.0) == pytest.approx(4 * math.pi)
    assert cone_solid_angle(90.0) == pytest.approx(2 * math.pi)
    assert cone_solid_angle(30.0) == pytest.approx(
        2 * math.pi * (1 - math.cos(math.radians(30.0))))


def test_reverberation_decays_as_range_to_the_minus_five():
    with torch.no_grad():
        _, _, arrivals = _reverb()
        grid = make_time_grid(2 * DEPTH / C * 1.02, 0.7, 4000)
        etc = render_reverberation(arrivals, grid, sigma_t=2e-3)[0, 0]
    rng = grid.numpy() * C / 2.0
    energy = etc.numpy()
    keep = (energy > 0) & (rng > 3 * DEPTH) & (rng < 400.0)
    slope = np.polyfit(np.log10(rng[keep]), np.log10(energy[keep]), 1)[0]
    assert slope == pytest.approx(-5.0, abs=0.1)


@pytest.mark.parametrize("range_m", [100.0, 200.0, 400.0])
def test_reverberation_level_matches_the_analytic_value(range_m):
    """pi c mu H^2 / r^5 for Lambert scattering on a flat bottom."""
    with torch.no_grad():
        _, scattering, arrivals = _reverb()
        grid = make_time_grid(2 * DEPTH / C * 1.02, 0.7, 4000)
        etc = render_reverberation(arrivals, grid, sigma_t=2e-3)[0, 0]
    rng = grid.numpy() * C / 2.0
    i = int(np.argmin(np.abs(rng - range_m)))
    analytic = math.pi * C * float(scattering.mu()) * DEPTH**2 / range_m**5
    assert etc[i].item() == pytest.approx(analytic, rel=0.06)


def test_first_reverberation_arrives_at_twice_the_normal_incidence_time():
    with torch.no_grad():
        _, _, arrivals = _reverb()
    assert arrivals.time.min().item() == pytest.approx(2 * DEPTH / C, abs=2e-3)


@pytest.mark.parametrize("delta_db", [-10.0, -6.0, 6.0])
def test_scattering_strength_scales_reverberation_exactly(delta_db):
    with torch.no_grad():
        _, _, base = _reverb(-27.0)
        _, _, louder = _reverb(-27.0 + delta_db)
    ratio = (louder.amplitude**2).sum() / (base.amplitude**2).sum()
    assert ratio.item() == pytest.approx(10.0 ** (delta_db / 10.0), rel=1e-10)


def test_arrivals_come_back_along_the_reverse_of_their_launch_direction():
    """Monostatic reciprocity is what gives each patch a bearing, and so what
    makes reverberation beamformable at all."""
    dirs, omega = _downgoing(20_000)
    scene = _scene()
    with torch.no_grad():
        result = scene.trace(dirs)
        arrivals = reverberation_arrivals(
            result, dirs, scene.freqs_khz,
            scattering=LambertScattering(-27.0, learnable=False),
            solid_angle_per_ray=omega, boundary="bottom",
            surface=scene.surface, bottom=scene.bottom)
    assert torch.allclose(arrivals.direction.norm(dim=-1),
                          torch.ones(arrivals.n_arrivals), atol=1e-12)
    # Every arrival must be the exact negation of some launch direction, so the
    # *most negative* dot product against the fan has to be essentially -1.
    dots = (arrivals.direction[:64] @ dirs.T).min(dim=1).values
    assert (dots < -0.999).all()


def test_surface_and_bottom_can_be_selected_separately():
    dirs, omega = _downgoing(20_000)
    scene = _scene(surface=FlatHeight(0.0), source=(0.0, 0.0, 25.0), max_bounces=3)
    with torch.no_grad():
        result = scene.trace(dirs)
        kw = dict(scattering=LambertScattering(-27.0, learnable=False),
                  solid_angle_per_ray=omega,
                  surface=scene.surface, bottom=scene.bottom)
        bottom = reverberation_arrivals(result, dirs, scene.freqs_khz,
                                        boundary="bottom", **kw)
        surface = reverberation_arrivals(result, dirs, scene.freqs_khz,
                                         boundary="surface", **kw)
        both = reverberation_arrivals(result, dirs, scene.freqs_khz,
                                      boundary="both", **kw)
    assert bottom.n_arrivals > 0 and surface.n_arrivals > 0
    assert both.n_arrivals == bottom.n_arrivals + surface.n_arrivals


def test_scattering_phase_is_uniformly_random():
    """Rough-boundary scattering randomises phase.  Without this, reverberation
    would add coherently and beamform into a false target."""
    with torch.no_grad():
        _, _, arrivals = _reverb()
    phase = arrivals.phase
    assert phase.min() >= 0.0 and phase.max() < 2 * math.pi
    # A uniform phase has a vanishing circular mean; a constant one does not.
    circular_mean = torch.polar(torch.ones_like(phase), phase).mean().abs().item()
    assert circular_mean < 0.02


def test_reverberation_fills_the_range_window_while_a_target_does_not():
    """The detection problem in one number.

    A target is confined to a couple of range cells; reverberation is returned
    from every patch the pulse has reached and so occupies the whole window.
    That contrast, not the beamformed peak level, is what decides whether a
    target is visible -- and it is the property that survives the fact that
    carrier phase is already effectively random within a range cell.
    """
    freq_hz = 100e3
    lam = C / freq_hz
    n_el = 32
    y = (torch.arange(n_el, dtype=torch.float64) - (n_el - 1) / 2) * (lam / 2)
    elements = torch.stack((torch.zeros(n_el), y, torch.zeros(n_el)), dim=-1)
    freqs = torch.tensor([freq_hz / 1e3])
    grid = make_time_grid(0.06, 0.12, 1200)
    steer, _ = azimuth_steering(61, 60.0)

    dirs, omega = _downgoing(60_000)
    scene = _scene(freqs_khz=freqs)
    with torch.no_grad():
        result = scene.trace(dirs)
        reverb = reverberation_arrivals(
            result, dirs, freqs,
            scattering=LambertScattering(-27.0, learnable=False),
            solid_angle_per_ray=omega, boundary="bottom", max_arrivals=300,
            surface=scene.surface, bottom=scene.bottom)
        assert reverb.n_arrivals > 50
        reverb_power = beamform(reverb, elements, freqs, grid, steer,
                                sigma_t=2e-4, steer_chunk=30)

        look = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        target = ArrivalSet(torch.tensor([0.09]), torch.ones(1, 1), -look,
                            torch.zeros(1), torch.zeros(1), torch.ones(1))
        target_power = beamform(target, elements, freqs, grid, steer,
                                sigma_t=2e-4, steer_chunk=30)

    def occupied_bins(power):
        """Range bins holding half the energy, summed over beams."""
        per_bin = power[:, 0].sum(dim=0)
        ordered = per_bin.sort(descending=True).values
        return int((ordered.cumsum(0) < 0.5 * ordered.sum()).sum()) + 1

    target_bins = occupied_bins(target_power)
    reverb_bins = occupied_bins(reverb_power)
    assert target_bins < 30, f"a target should be range-compact, got {target_bins} bins"
    assert reverb_bins > 5 * target_bins, (
        f"reverberation filled only {reverb_bins} bins against the target's "
        f"{target_bins}; it is not spread over the window")


def test_subsampling_preserves_energy_and_range_spread():
    """Regression: capping reverberation by keeping the *strongest* patches
    collapses the sample onto the first few range cells, because r^-5 makes the
    strongest patches the nearest ones.  Random subsampling with a compensating
    energy scale keeps both the level and the spread."""
    dirs, omega = _downgoing(60_000)
    scene = _scene(freqs_khz=torch.tensor([100.0]))
    with torch.no_grad():
        result = scene.trace(dirs)
        kw = dict(scattering=LambertScattering(-27.0, learnable=False),
                  solid_angle_per_ray=omega, boundary="bottom",
                  surface=scene.surface, bottom=scene.bottom)
        full = reverberation_arrivals(result, dirs, scene.freqs_khz, **kw)
        subset = reverberation_arrivals(result, dirs, scene.freqs_khz,
                                        max_arrivals=300, **kw)
        grid = make_time_grid(0.06, 0.12, 1200)
        etc_full = render_reverberation(full, grid, sigma_t=2e-4)[0, 0]
        etc_sub = render_reverberation(subset, grid, sigma_t=2e-4)[0, 0]

    assert subset.n_arrivals == 300 and full.n_arrivals > 10_000
    assert etc_sub.sum().item() == pytest.approx(etc_full.sum().item(), rel=0.25)

    def half_energy_bins(e):
        ordered = e.sort(descending=True).values
        return int((ordered.cumsum(0) < 0.5 * ordered.sum()).sum()) + 1

    assert half_energy_bins(etc_sub) > 0.5 * half_energy_bins(etc_full)


def test_gradients_reach_the_scattering_strength():
    dirs, omega = _downgoing(20_000)
    # More than one bounce, so that later patches carry the specular loss of the
    # earlier ones -- with max_bounces=1 the bottom loss correctly has no effect
    # on reverberation at all, because the scattered energy was never reflected.
    scene = _scene(bottom_loss=ConstantLoss(2.0), max_bounces=3,
                   surface=FlatHeight(0.0), source=(0.0, 0.0, 25.0))
    scattering = LambertScattering(-27.0, learnable=True)
    result = scene.trace(dirs)
    arrivals = reverberation_arrivals(result, dirs, scene.freqs_khz,
                                      scattering=scattering,
                                      solid_angle_per_ray=omega, boundary="bottom",
                                      surface=scene.surface, bottom=scene.bottom)
    grid = make_time_grid(2 * DEPTH / C, 0.4, 500)
    render_reverberation(arrivals, grid, sigma_t=2e-3).sum().backward()

    for name, p in (("scattering.strength_db", scattering.strength_db),
                    ("bottom_loss.loss_db", scene.bottom_loss.loss_db)):
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is non-finite"
        assert p.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_no_bounces_gives_an_empty_arrival_set():
    dirs, omega = _downgoing(2_000)
    scene = _scene(bottom=FlatHeight(1e6))  # nothing can reach the seabed
    with torch.no_grad():
        result = scene.trace(dirs)
        arrivals = reverberation_arrivals(
            result, dirs, scene.freqs_khz,
            scattering=LambertScattering(-27.0, learnable=False),
            solid_angle_per_ray=omega, boundary="bottom",
            surface=scene.surface, bottom=scene.bottom)
    assert arrivals.n_arrivals == 0
    grid = make_time_grid(0.0, 0.4, 200)
    etc = render_reverberation(arrivals, grid, sigma_t=2e-3)
    assert torch.equal(etc, torch.zeros_like(etc))
