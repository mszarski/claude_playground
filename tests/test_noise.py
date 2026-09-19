"""Ambient noise levels, and the statistics of adding them to an image.

The levels are checked against the Wenz components they are built from, and
the sampler against the moments of a Rice power distribution -- because the
whole point of the cross term is that it changes the variance, and a sampler
that gets the mean right and the variance wrong will quietly misstate every
false-alarm rate computed from it.
"""

import math

import pytest
import torch

from hydropt.noise import (
    add_receiver_noise, ambient_noise_db, beam_noise_power, calibrate,
    line_array_directivity_db,
)


def test_thermal_noise_dominates_the_top_of_the_band():
    """At 100 kHz with no wind and no shipping, the level is thermal.

    Not exactly: the surface term is ``50 + 7.5 sqrt(w)``, which does not
    vanish at zero wind -- a flat calm still has some surface agitation, and
    the fit carries a floor.  It contributes 9.9 dB at 100 kHz against
    thermal's 25, so it moves the total by 0.13 dB and no more.
    """
    f = torch.tensor([100.0])
    got = float(ambient_noise_db(f, wind_speed=0.0, shipping=0.0,
                                 turbulence=False))
    thermal = -15.0 + 20.0 * math.log10(100.0)
    assert thermal < got < thermal + 0.2


def test_thermal_noise_rises_20_db_per_decade():
    """Measured where thermal really is the only term left.

    At 50 kHz the calm-surface floor is still within 3 dB of thermal and drags
    the slope down to 18 dB/decade; by 200 kHz it is 20 dB below and the slope
    is thermal's own.
    """
    f = torch.tensor([200.0, 2000.0])
    q = ambient_noise_db(f, wind_speed=0.0, shipping=0.0, turbulence=False)
    assert float(q[1] - q[0]) == pytest.approx(20.0, abs=0.1)


def test_wind_raises_the_level_everywhere_and_monotonically():
    f = torch.tensor([0.5, 5.0, 50.0])
    quiet = ambient_noise_db(f, wind_speed=0.0)
    for w in (2.0, 10.0, 20.0):
        louder = ambient_noise_db(f, wind_speed=w)
        assert bool((louder > quiet).all())
        quiet = louder


def test_a_seaway_beats_thermal_noise_at_100_khz():
    """Which term dominates decides whether more frequency buys anything."""
    f = torch.tensor([100.0])
    thermal = float(ambient_noise_db(f, wind_speed=0.0, shipping=0.0,
                                     turbulence=False))
    seaway = float(ambient_noise_db(f, wind_speed=10.0, shipping=0.0,
                                    turbulence=False))
    assert seaway - thermal > 5.0


def test_components_sum_as_powers_not_levels():
    f = torch.tensor([1.0])
    both = 10.0 ** (float(ambient_noise_db(f, wind_speed=8.0, shipping=0.5)) / 10)
    no_ship = 10.0 ** (float(ambient_noise_db(f, wind_speed=8.0,
                                              shipping=0.0)) / 10)
    assert both > no_ship                      # adding a source adds power
    assert both < 2.0 * no_ship                # but shipping is not dominant here


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        ambient_noise_db(torch.tensor([0.0]))
    with pytest.raises(ValueError, match="wind"):
        ambient_noise_db(torch.tensor([1.0]), wind_speed=-1.0)
    with pytest.raises(ValueError, match="shipping"):
        ambient_noise_db(torch.tensor([1.0]), shipping=1.5)
    with pytest.raises(ValueError, match="bandwidth"):
        beam_noise_power(torch.tensor([1.0]), bandwidth_hz=0.0)
    with pytest.raises(ValueError, match="element"):
        line_array_directivity_db(0)


def test_directivity_is_ten_log_n_at_half_wavelength():
    assert line_array_directivity_db(64) == pytest.approx(18.06, abs=0.01)
    assert line_array_directivity_db(1) == pytest.approx(0.0)
    # Spreading the same elements wider does not keep buying directivity.
    assert line_array_directivity_db(64, 2.0) < line_array_directivity_db(64, 0.5)


def test_bandwidth_and_directivity_move_the_beam_noise_the_right_way():
    f = torch.tensor([100.0])
    base = float(beam_noise_power(f, bandwidth_hz=8000.0))
    wider = float(beam_noise_power(f, bandwidth_hz=16000.0))
    aimed = float(beam_noise_power(f, bandwidth_hz=8000.0, directivity_db=18.0))
    assert 10 * math.log10(wider / base) == pytest.approx(10 * math.log10(2.0),
                                                          abs=0.01)
    assert 10 * math.log10(base / aimed) == pytest.approx(18.0, abs=0.01)


def test_calibrate_puts_the_image_on_the_noise_scale():
    image = torch.tensor([1.0, 0.25])
    out = calibrate(image, 210.0)
    assert float(10 * torch.log10(out[0])) == pytest.approx(210.0)
    assert float(10 * torch.log10(out[1])) == pytest.approx(210.0 - 6.02, abs=0.01)


def test_noise_only_cells_are_exponential_with_the_right_mean():
    g = torch.Generator().manual_seed(0)
    n = 4.0
    out = add_receiver_noise(torch.zeros(200_000), n, generator=g)
    assert float(out.mean()) == pytest.approx(n, rel=0.02)
    # Exponential: variance = mean^2, and P(x > mean) = 1/e.
    assert float(out.var()) == pytest.approx(n * n, rel=0.05)
    assert float((out > n).to(out.dtype).mean()) == pytest.approx(1 / math.e,
                                                                 rel=0.03)


def test_the_cross_term_is_there():
    """|s+n|^2 has variance N^2 + 2 S N -- the second term IS the cross term.

    Dropping it (adding noise power to signal power) leaves the mean right and
    the variance far too small, and every threshold set from that variance is
    optimistic.
    """
    g = torch.Generator().manual_seed(1)
    s, n = 9.0, 1.0
    out = add_receiver_noise(torch.full((400_000,), s), n, generator=g)
    assert float(out.mean()) == pytest.approx(s + n, rel=0.01)
    assert float(out.var()) == pytest.approx(n * n + 2 * s * n, rel=0.05)


def test_noise_broadcasts_per_band():
    g = torch.Generator().manual_seed(2)
    power = torch.zeros(3, 2, 50_000)
    n = torch.tensor([1.0, 100.0]).reshape(1, 2, 1)
    out = add_receiver_noise(power, n, generator=g)
    assert out.shape == power.shape
    assert float(out[:, 0].mean()) == pytest.approx(1.0, rel=0.05)
    assert float(out[:, 1].mean()) == pytest.approx(100.0, rel=0.05)


def test_the_sampler_stays_differentiable_in_the_signal():
    g = torch.Generator().manual_seed(3)
    power = torch.full((2048,), 4.0, requires_grad=True)
    add_receiver_noise(power, 1.0, generator=g).sum().backward()
    assert power.grad is not None and bool(torch.isfinite(power.grad).all())
    assert float(power.grad.abs().max()) > 0.0


def test_the_beamformer_scale_is_what_the_beamformer_actually_does():
    """Measured against beamform itself, not asserted from the algebra.

    A single arrival of unit amplitude, on the beam that looks straight at it,
    comes out at exactly (sum w)^2 / (2 pi sigma_t^2).  It is 70 dB for a
    64-element array and a 0.12 ms pulse, so an image compared with a noise
    level without dividing it out is not slightly wrong.
    """
    from hydropt import beam_power_scale, beamform, make_time_grid, shading_window
    from hydropt.beamform import ArrivalSet

    n, sigma_t = 64, 1.2e-4
    elements = torch.stack([torch.zeros(n), (torch.arange(n) - (n - 1) / 2) * 0.0075,
                            torch.zeros(n)], dim=-1)
    look = torch.tensor([[1.0, 0.0, 0.0]])
    arrival = ArrivalSet(time=torch.tensor([0.04]), amplitude=torch.ones(1, 1),
                         direction=-look, phase=torch.zeros(1),
                         distance=torch.zeros(1), path_length=torch.ones(1))
    grid = make_time_grid(0.039, 0.041, 801)
    for window in ("uniform", "hamming"):
        w = shading_window(n, window)
        image = beamform(arrival, elements, torch.tensor([100.0]), grid, look,
                         sigma_t=sigma_t, shading=w)
        assert float(image.max()) == pytest.approx(beam_power_scale(w, sigma_t),
                                                   rel=1e-9)


def test_calibrate_undoes_the_beamformer_and_applies_the_source_level():
    image = torch.tensor([4.0])
    out = calibrate(image, 210.0, beam_scale=4.0)
    assert float(10 * torch.log10(out[0])) == pytest.approx(210.0)
    with pytest.raises(ValueError, match="beam_scale"):
        calibrate(image, 210.0, beam_scale=0.0)


def test_an_exactly_zero_cell_does_not_poison_the_gradient():
    """sqrt'(0) is infinite; a zero cell must contribute nothing, not NaN.

    A beamformed image can hold cells that are exactly zero -- bins no arrival
    reached, or values that underflowed -- and the Rice noise model takes the
    square root of every cell.  Its derivative at zero is infinite, and
    ``0 * inf`` is NaN, which the image's sum spreads to every parameter's
    gradient.  Seen on a 90 m image with two such cells.
    """
    import torch
    from hydropt.noise import add_receiver_noise

    scale = torch.tensor(1.0, requires_grad=True)
    power = torch.tensor([0.0, 1e-30, 1.0, 4.0]) * scale
    noisy = add_receiver_noise(power, 0.5, generator=torch.Generator().manual_seed(1))
    noisy.sum().backward()
    assert torch.isfinite(scale.grad).all()
    assert float(scale.grad) > 0
