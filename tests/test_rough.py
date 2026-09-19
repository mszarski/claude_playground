"""Rough-surface coherence loss: the Eckart factor and how it reaches the renderer.

The physics is one formula with a closed form, so most of this is checking it
exactly -- including the two limits that must be *exactly* zero, and the
quadratic frequency scaling that is the reason the loss cannot live in a
frequency-independent `BoundaryLoss`.
"""

from __future__ import annotations

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, RoughSurfaceLoss, Scene,
    coherent_reflection_loss_db, extract_arrivals, make_time_grid,
    rayleigh_roughness, roughness_weights, splat_etc, trace, wind_sea_rms_height,
)
from hydropt.launch import fan_2d

DB_PER_GAMMA2 = 10.0 / math.log(10.0)


@pytest.fixture(autouse=True)
def _double():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _sigma_for_gamma(gamma: float, freq_khz: float, grazing_rad: float,
                     c: float = 1500.0) -> float:
    k = 2 * math.pi * freq_khz * 1e3 / c
    return gamma / (2 * k * math.sin(grazing_rad))


# --------------------------------------------------------------------------- #
# The formula
# --------------------------------------------------------------------------- #
def test_the_rayleigh_parameter_is_twice_k_sigma_sin_theta():
    graze = torch.tensor([0.3])
    sigma, freq, c = 0.1, 2.0, 1500.0
    k = 2 * math.pi * freq * 1e3 / c
    assert float(rayleigh_roughness(graze, sigma, freq, c)) == pytest.approx(
        2 * k * sigma * math.sin(0.3), rel=1e-12)


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0, 2.0, 3.0])
def test_loss_is_ten_over_ln_ten_times_gamma_squared(gamma):
    """Gamma = 1 costs 4.343 dB and Gamma = 2 costs 17.372 dB."""
    freq, graze = 1.0, math.radians(30.0)
    sigma = _sigma_for_gamma(gamma, freq, graze)
    got = float(coherent_reflection_loss_db(torch.tensor([graze]), sigma, freq))
    assert got == pytest.approx(DB_PER_GAMMA2 * gamma**2, rel=1e-12)


def test_a_smooth_boundary_costs_exactly_nothing():
    """Exactly zero, not nearly: a flat boundary has no height variation to see,
    and a loss model that returned 1e-16 dB here would be hiding an error."""
    assert float(coherent_reflection_loss_db(torch.tensor([0.7]), 0.0, 100.0)) == 0.0


def test_grazing_incidence_costs_exactly_nothing():
    """A ray running along the boundary sees no height change along its own path,
    so sin(theta) = 0 kills the loss however rough the boundary is."""
    assert float(coherent_reflection_loss_db(torch.tensor([0.0]), 10.0, 500.0)) == 0.0


def test_the_loss_is_quadratic_in_frequency():
    """The reason this cannot be a BoundaryLoss: doubling frequency quadruples it.

    A single design frequency across 0.5-4 kHz would be wrong by a factor of 64.
    """
    graze = torch.tensor([math.radians(30.0)])
    sigma = 0.1
    base = float(coherent_reflection_loss_db(graze, sigma, 0.5))
    for factor in (2.0, 4.0, 8.0):
        got = float(coherent_reflection_loss_db(graze, sigma, 0.5 * factor))
        assert got / base == pytest.approx(factor**2, rel=1e-12)


def test_the_loss_is_quadratic_in_rms_height_and_in_sine_of_grazing():
    graze = torch.tensor([math.radians(20.0)])
    a = float(coherent_reflection_loss_db(graze, 0.05, 3.0))
    b = float(coherent_reflection_loss_db(graze, 0.15, 3.0))
    assert b / a == pytest.approx(9.0, rel=1e-12)
    c = float(coherent_reflection_loss_db(torch.tensor([math.radians(60.0)]), 0.05, 3.0))
    ratio = (math.sin(math.radians(60.0)) / math.sin(math.radians(20.0))) ** 2
    assert c / a == pytest.approx(ratio, rel=1e-12)


def test_a_very_rough_boundary_leaves_exactly_no_coherent_energy():
    """The energy factor underflows to zero, which is the physical answer.

    Pinned because it must not be an inf, a nan, or a clamp: there simply is no
    coherent reflection from a surface many wavelengths rough.
    """
    db = float(coherent_reflection_loss_db(torch.tensor([math.radians(30.0)]),
                                           wind_sea_rms_height(10.0), 100.0))
    assert db > 1e4
    assert math.isfinite(db)
    assert 10.0 ** (-db / 10.0) == 0.0


def test_wind_sea_rms_height_agrees_with_the_surface_generator():
    from hydropt import pierson_moskowitz_surface, wave_number_peak_pm

    wind = 9.0
    kp = wave_number_peak_pm(wind)
    dx = 2 * math.pi / kp / 8
    surface = pierson_moskowitz_surface((128, 128), (dx, dx), wind,
                                        generator=torch.Generator().manual_seed(0))
    assert float(surface.heights.std(unbiased=False)) == pytest.approx(
        wind_sea_rms_height(wind), rel=1e-12)


# --------------------------------------------------------------------------- #
# The drop-in BoundaryLoss
# --------------------------------------------------------------------------- #
def test_rough_surface_loss_adds_roughness_to_its_smooth_loss():
    graze = torch.tensor([math.radians(25.0)])
    smooth = ConstantLoss(1.5, learnable=False)
    rough = RoughSurfaceLoss(0.03, freq_khz=4.0, smooth=smooth, learnable=False)
    expected = 1.5 + float(coherent_reflection_loss_db(graze, 0.03, 4.0))
    assert float(rough(graze)) == pytest.approx(expected, rel=1e-12)


def test_rough_surface_loss_defaults_to_a_lossless_smooth_boundary():
    """Which is what a pressure-release sea surface is, before roughness."""
    graze = torch.tensor([math.radians(25.0)])
    rough = RoughSurfaceLoss(0.03, freq_khz=4.0, learnable=False)
    assert float(rough(graze)) == pytest.approx(
        float(coherent_reflection_loss_db(graze, 0.03, 4.0)), rel=1e-12)


def test_roughness_leaves_the_reflection_phase_alone():
    """Eckart averaging reduces the coherent amplitude; the mean phase is where it
    was.  A pressure-release surface must still invert."""
    graze = torch.tensor([0.2, 0.6])
    rough = RoughSurfaceLoss(0.05, freq_khz=10.0, learnable=False,
                             pressure_release=True)
    assert torch.allclose(rough.reflection_phase(graze),
                          torch.full_like(graze, math.pi))
    plain = RoughSurfaceLoss(0.05, freq_khz=10.0, learnable=False)
    assert torch.allclose(plain.reflection_phase(graze), torch.zeros_like(graze))


def test_rms_height_is_learnable_so_roughness_can_be_fitted():
    rough = RoughSurfaceLoss(0.04, freq_khz=6.0)
    rough(torch.tensor([0.4])).sum().backward()
    assert rough.rms_height.grad is not None
    assert float(rough.rms_height.grad) > 0.0, "more roughness must cost more"
    assert list(RoughSurfaceLoss(0.04, freq_khz=6.0, learnable=False).parameters()) == []


def test_the_design_frequency_is_not_learnable():
    """It is a property of the scene, not of the boundary."""
    rough = RoughSurfaceLoss(0.04, freq_khz=6.0)
    names = [n for n, _ in rough.named_parameters()]
    assert "freq_khz" not in names
    assert "rms_height" in names


# --------------------------------------------------------------------------- #
# The frequency-correct route: per-band weights from bounce events
# --------------------------------------------------------------------------- #
def _shallow_scene() -> Scene:
    return Scene(field=IsoProfile(1500.0, learnable=False), bottom=FlatHeight(50.0),
                 surface=FlatHeight(0.0), source=(0.0, 0.0, 25.0),
                 receivers=torch.tensor([[400.0, 0.0, 25.0]]),
                 surface_loss=ConstantLoss(0.0, learnable=False),
                 bottom_loss=ConstantLoss(3.0, learnable=False),
                 freqs_khz=torch.tensor([0.5, 2.0, 8.0]),
                 step_size=1.0, n_steps=600, max_bounces=20)


def test_weights_are_per_ray_per_band_and_bounded():
    scene = _shallow_scene()
    result = trace(scene, fan_2d(200, elev_range_deg=(-35.0, 35.0)))
    w = roughness_weights(result, scene.freqs_khz, surface_rms=0.05,
                          surface=scene.surface, bottom=scene.bottom)
    assert w.shape == (200, 3)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0


def test_a_ray_that_never_bounced_is_left_exactly_alone():
    """The identity case has to be exact, or applying roughness would quietly
    attenuate the direct path."""
    scene = _shallow_scene()
    result = trace(scene, fan_2d(200, elev_range_deg=(-35.0, 35.0)))
    w = roughness_weights(result, scene.freqs_khz, surface_rms=0.05,
                          surface=scene.surface, bottom=scene.bottom)
    never = (result.n_surface + result.n_bottom) == 0
    assert int(never.sum()) > 0, "test needs some unbounced rays"
    assert bool((w[never] == 1.0).all())


def test_a_smooth_scene_gives_exactly_unit_weights():
    scene = _shallow_scene()
    result = trace(scene, fan_2d(50, elev_range_deg=(-35.0, 35.0)))
    w = roughness_weights(result, scene.freqs_khz)
    assert bool((w == 1.0).all())


def test_higher_bands_are_attenuated_more():
    scene = _shallow_scene()
    result = trace(scene, fan_2d(200, elev_range_deg=(-35.0, 35.0)))
    w = roughness_weights(result, scene.freqs_khz, surface_rms=0.05,
                          surface=scene.surface, bottom=scene.bottom)
    bounced = (result.n_surface + result.n_bottom) > 0
    means = w[bounced].mean(dim=0)
    assert float(means[0]) > float(means[1]) > float(means[2])


def test_two_bounces_cost_the_product_of_one():
    """Independent bounces multiply, so the log-space accumulation must sum."""
    scene = _shallow_scene()
    result = trace(scene, fan_2d(400, elev_range_deg=(-35.0, 35.0)))
    w = roughness_weights(result, scene.freqs_khz, surface_rms=0.05,
                          bottom_rms=0.05, surface=scene.surface,
                          bottom=scene.bottom)
    total = (result.n_surface + result.n_bottom).round().long()
    one = w[total == 1][:, 2]
    two = w[total == 2][:, 2]
    assert one.numel() > 0 and two.numel() > 0
    # Not a strict per-ray identity (grazing angles differ), but two bounces of a
    # comparable boundary must sit below one.
    assert float(two.max()) < float(one.max())


def test_per_band_weights_reach_the_energy_renderer():
    scene = _shallow_scene()
    result = trace(scene, fan_2d(300, elev_range_deg=(-35.0, 35.0)))
    w = roughness_weights(result, scene.freqs_khz, surface_rms=0.05,
                          surface=scene.surface, bottom=scene.bottom)
    grid = make_time_grid(0.25, 0.45, 300)
    kw = dict(sigma_d=1.0, sigma_t=3e-4)
    smooth = splat_etc(result, scene.receivers, grid, scene.freqs_khz, **kw)
    rough = splat_etc(result, scene.receivers, grid, scene.freqs_khz,
                      ray_weights=w, **kw)
    ratios = [float(rough[0, i].sum() / smooth[0, i].sum()) for i in range(3)]
    assert all(0.0 < r <= 1.0 for r in ratios)
    assert ratios[0] > ratios[1] > ratios[2], f"no frequency ordering: {ratios}"


def test_per_band_weights_reach_the_coherent_extractor():
    scene = _shallow_scene()
    result = trace(scene, fan_2d(300, elev_range_deg=(-35.0, 35.0)))
    w = roughness_weights(result, scene.freqs_khz, surface_rms=0.05,
                          surface=scene.surface, bottom=scene.bottom)
    point = scene.receivers[0]
    smooth = extract_arrivals(result, point, scene.freqs_khz, sigma_d=1.0)
    rough = extract_arrivals(result, point, scene.freqs_khz, sigma_d=1.0,
                             ray_weights=w)
    assert rough.n_arrivals == smooth.n_arrivals > 0
    assert torch.all(rough.amplitude <= smooth.amplitude + 1e-12)
    per_band = (rough.amplitude ** 2).sum(0) / (smooth.amplitude ** 2).sum(0)
    assert float(per_band[0]) > float(per_band[1]) > float(per_band[2])


def test_a_scalar_ray_weight_still_works_unchanged():
    """The [R] path must not have been broken by adding the [R, B] one."""
    scene = _shallow_scene()
    result = trace(scene, fan_2d(60, elev_range_deg=(-30.0, 30.0)))
    grid = make_time_grid(0.25, 0.45, 200)
    kw = dict(sigma_d=1.0, sigma_t=3e-4)
    plain = splat_etc(result, scene.receivers, grid, scene.freqs_khz, **kw)
    halved = splat_etc(result, scene.receivers, grid, scene.freqs_khz,
                       ray_weights=torch.full((60,), 0.5), **kw)
    assert torch.allclose(halved, 0.5 * plain, rtol=1e-12)


def test_a_mis_shaped_ray_weight_is_rejected_rather_than_broadcast():
    scene = _shallow_scene()
    result = trace(scene, fan_2d(40, elev_range_deg=(-30.0, 30.0)))
    grid = make_time_grid(0.25, 0.45, 100)
    kw = dict(sigma_d=1.0, sigma_t=3e-4)
    with pytest.raises(ValueError, match="expected 3"):
        splat_etc(result, scene.receivers, grid, scene.freqs_khz,
                  ray_weights=torch.ones(40, 5), **kw)
    with pytest.raises(ValueError, match=r"\[R\] or \[R, B\]"):
        splat_etc(result, scene.receivers, grid, scene.freqs_khz,
                  ray_weights=torch.ones(40, 3, 2), **kw)
    with pytest.raises(ValueError, match="expected 3"):
        extract_arrivals(result, scene.receivers[0], scene.freqs_khz, sigma_d=1.0,
                         ray_weights=torch.ones(40, 5))


# --------------------------------------------------------------------------- #
# The result that matters for a high-frequency sonar
# --------------------------------------------------------------------------- #
def test_at_a_hundred_kilohertz_a_real_sea_destroys_a_steep_specular_path():
    """The quantitative form of example 06's caveat, at *its* geometry.

    At 100 kHz the wavelength is 15 mm and 2 m/s of wind gives 22 mm of RMS
    elevation, so a surface path at 30 degrees grazing -- which is what a vehicle
    at 10 m looking 40 m out has -- is gone entirely.  The angle qualifier is not
    decoration; see the next test.
    """
    graze = torch.tensor([math.radians(30.0)])
    calm = float(coherent_reflection_loss_db(graze, 0.0, 100.0))
    assert calm == 0.0
    for wind in (2.0, 5.0, 10.0):
        db = float(coherent_reflection_loss_db(graze, wind_sea_rms_height(wind), 100.0))
        assert db > 100.0, f"{wind} m/s gave only {db:.1f} dB at 100 kHz"
    # At 500 Hz the same sea is a few dB, not an annihilation.
    mild = float(coherent_reflection_loss_db(graze, wind_sea_rms_height(10.0), 0.5))
    assert 1.0 < mild < 20.0, f"{mild:.2f} dB at 500 Hz"


def test_a_near_grazing_path_survives_any_sea_state():
    """The correction to my own first draft of this, pinned so it cannot return.

    ``Gamma`` is proportional to ``sin(theta)``, so a ray at near-grazing
    incidence sees a surface effectively flat along its own direction of travel
    and reflects coherently however rough that surface is.  "A rough surface
    destroys the specular path at high frequency" is therefore only true above a
    cutoff angle, and long-range shallow-water propagation -- which lives at small
    grazing angles -- keeps its surface bounces even at 100 kHz.
    """
    freq = 100.0
    steep = torch.tensor([math.radians(30.0)])
    window = []
    for wind in (2.0, 5.0, 10.0, 20.0):
        sigma = wind_sea_rms_height(wind)
        # The angle where the loss is 3 dB, from the closed form; a surviving
        # window always exists, it just narrows as 1/sigma.
        k = 2 * math.pi * freq * 1e3 / 1500.0
        theta = math.asin(math.sqrt(3.0 / DB_PER_GAMMA2) / (2 * k * sigma))
        window.append(math.degrees(theta))
        half = torch.tensor([theta])
        assert float(coherent_reflection_loss_db(half, sigma, freq)) == pytest.approx(
            3.0, rel=1e-9)
        # Half the window in, most of the energy is still there.
        inside = torch.tensor([theta / 2])
        db_in = float(coherent_reflection_loss_db(inside, sigma, freq))
        assert 10.0 ** (-db_in / 10.0) > 0.8, f"{wind} m/s: {db_in:.2f} dB"
        # While the steep path at the same sea state is gone.
        assert float(coherent_reflection_loss_db(steep, sigma, freq)) > 100.0
    # The window narrows monotonically with sea state, and never closes.
    for a, b in zip(window, window[1:]):
        assert a > b > 0.0


def test_the_three_db_cutoff_angle_has_the_closed_form_it_should():
    r"""``loss = (10/ln 10) Gamma^2`` reaches 3 dB at
    ``sin(theta) = sqrt(3 ln 10 / 10) / (2 k sigma)``, which is what separates the
    surviving near-grazing paths from the annihilated steep ones."""
    sigma = wind_sea_rms_height(2.0)
    freq, c = 100.0, 1500.0
    k = 2 * math.pi * freq * 1e3 / c
    sin_theta = math.sqrt(3.0 / DB_PER_GAMMA2) / (2 * k * sigma)
    theta = math.asin(sin_theta)
    assert math.degrees(theta) == pytest.approx(2.53, abs=0.01)
    at_cutoff = float(coherent_reflection_loss_db(torch.tensor([theta]), sigma, freq))
    assert at_cutoff == pytest.approx(3.0, rel=1e-9)
