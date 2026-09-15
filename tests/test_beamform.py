"""Coherent arrivals and beamforming, checked against classical array theory.

A uniform line array has closed-form answers -- a known beamwidth, a -13.26 dB
first sidelobe, nulls at known angles, an array gain of exactly ``N`` -- so the
beamformer is pinned to those rather than to its own output.  If the phase
handling were wrong in any of the ways it easily could be (sign of the steering
delay, a missing surface flip, narrowband steering where time delay is needed),
these numbers move immediately.
"""

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, Scene, make_time_grid,
)
from hydropt.beamform import (
    ArrivalSet, azimuth_steering, beamform, element_field, extract_arrivals,
    shading_window,
)
from hydropt.launch import fibonacci_cone

C = 1500.0
FREQ_HZ = 100e3
LAMBDA = C / FREQ_HZ
N_EL = 32


def _ula(n: int = N_EL, spacing: float = LAMBDA / 2, depth: float = 0.0) -> torch.Tensor:
    """Uniform line array along y, centred on the origin."""
    y = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2) * spacing
    return torch.stack((torch.zeros(n), y, torch.full((n,), depth)), dim=-1)


def _plane_wave(bearing_deg: float, *, tau: float = 0.010,
                phase: float = 0.0) -> ArrivalSet:
    """One arrival from a target at ``bearing_deg``, propagating towards the array."""
    b = math.radians(bearing_deg)
    look = torch.tensor([[math.cos(b), math.sin(b), 0.0]], dtype=torch.float64)
    return ArrivalSet(time=torch.tensor([tau]), amplitude=torch.ones(1, 1),
                      direction=-look, phase=torch.tensor([phase]),
                      distance=torch.zeros(1), path_length=torch.ones(1))


def _sweep(arrivals, elements, *, sigma_t=2e-4, shading=None, half_sector=60.0,
           n=1801, tau=0.010):
    grid = make_time_grid(tau - 0.002, tau + 0.002, 801)
    freqs = torch.tensor([FREQ_HZ / 1e3])
    steer, ang = azimuth_steering(n, half_sector)
    power = beamform(arrivals, elements, freqs, grid, steer, sigma_t=sigma_t,
                     shading=shading, steer_chunk=200)
    return ang, power.max(dim=-1).values[:, 0]


def _pattern_db(pattern):
    return 10.0 * torch.log10(pattern / pattern.max())


def _sidelobe_peak_db(ang, pattern):
    """Highest sidelobe, excluding the mainlobe (the -3 dB region around the peak)."""
    db = _pattern_db(pattern)
    main = set((db > -3.0).nonzero().flatten().tolist())
    lobes = [db[i].item() for i in range(1, len(ang) - 1)
             if db[i] > db[i - 1] and db[i] > db[i + 1] and i not in main]
    return max(lobes)


# --------------------------------------------------------------------------- #
# Classical uniform-line-array results
# --------------------------------------------------------------------------- #
def test_broadside_beam_points_broadside():
    ang, pat = _sweep(_plane_wave(0.0), _ula())
    assert ang[pat.argmax()].item() == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("bearing", [-40.0, -25.0, 0.0, 10.0, 33.0])
def test_beam_reports_the_true_target_bearing(bearing):
    """steer_directions are *look* directions, so the peak lands on the target's
    bearing rather than on the direction the wave is travelling."""
    ang, pat = _sweep(_plane_wave(bearing), _ula())
    assert ang[pat.argmax()].item() == pytest.approx(bearing, abs=0.1)


def test_mainlobe_width_matches_theory():
    """For a uniform line array the -3 dB width is about 101.5/N degrees at
    broadside with half-wave spacing."""
    ang, pat = _sweep(_plane_wave(0.0), _ula())
    inside = (_pattern_db(pat) > -3.0).nonzero().flatten()
    width = (ang[inside[-1]] - ang[inside[0]]).item()
    assert width == pytest.approx(101.5 / N_EL, rel=0.08)


def test_first_sidelobe_is_minus_13_dB():
    """The classical uniform-shading result, -13.26 dB.  This is the single
    most diagnostic number here: get the aperture phase wrong in almost any way
    and it moves."""
    ang, pat = _sweep(_plane_wave(0.0), _ula())
    assert _sidelobe_peak_db(ang, pat) == pytest.approx(-13.26, abs=0.3)


def test_nulls_fall_where_theory_puts_them():
    """Nulls at sin(theta) = 2m/N for half-wave spacing."""
    ang, pat = _sweep(_plane_wave(0.0), _ula())
    db = _pattern_db(pat)
    nulls = sorted((ang[i].item() for i in range(1, len(ang) - 1)
                    if db[i] < db[i - 1] and db[i] < db[i + 1] and db[i] < -25.0),
                   key=abs)
    for m in (1, 2):
        expected = math.degrees(math.asin(2 * m / N_EL))
        assert min(abs(abs(n) - expected) for n in nulls) < 0.2


def test_array_gain_equals_the_element_count():
    """Coherent summation beats incoherent by exactly N for a plane wave."""
    elements = _ula()
    grid = make_time_grid(0.008, 0.012, 801)
    freqs = torch.tensor([FREQ_HZ / 1e3])
    arrivals = _plane_wave(0.0)

    _, pat = _sweep(arrivals, elements)
    field = element_field(arrivals, elements, freqs, grid, sigma_t=2e-4)
    incoherent = (field.real**2 + field.imag**2).sum(0).max().item()
    # shading_window normalises to unit sum, so undo the 1/N to compare powers.
    coherent = pat.max().item() * N_EL**2
    assert coherent / incoherent == pytest.approx(N_EL, rel=1e-6)


@pytest.mark.parametrize("kind,expected_db", [
    ("uniform", -13.26), ("hamming", -42.0), ("blackman", -58.1),
])
def test_shading_trades_mainlobe_width_for_sidelobes(kind, expected_db):
    elements = _ula()
    w = shading_window(N_EL, kind)
    ang, pat = _sweep(_plane_wave(0.0), elements, shading=w)
    assert _sidelobe_peak_db(ang, pat) == pytest.approx(expected_db, abs=1.0)


def test_tapering_widens_the_mainlobe():
    elements = _ula()

    def width(kind):
        ang, pat = _sweep(_plane_wave(0.0), elements, shading=shading_window(N_EL, kind))
        inside = (_pattern_db(pat) > -3.0).nonzero().flatten()
        return (ang[inside[-1]] - ang[inside[0]]).item()

    assert width("uniform") < width("hamming") < width("blackman")


def test_full_wavelength_spacing_produces_grating_lobes():
    """Spacing beyond half a wavelength aliases: a broadside beam reappears at
    endfire.  This is the check that the element phase really is geometric."""
    elements = _ula(n=8, spacing=LAMBDA)
    ang, pat = _sweep(_plane_wave(0.0), elements, half_sector=90.0, n=3601)
    peaks = [ang[i].item() for i in range(1, len(ang) - 1)
             if pat[i] > pat[i - 1] and pat[i] > pat[i + 1] and pat[i] > pat.max() * 0.5]
    assert any(abs(p) > 85.0 for p in peaks), "no grating lobe near endfire"
    assert any(abs(p) < 1.0 for p in peaks), "mainlobe missing"


def test_two_targets_resolve_as_two_beams():
    elements = _ula()
    bearings = (-15.0, 12.0)
    dirs = torch.cat([_plane_wave(b).direction for b in bearings], dim=0)
    arrivals = ArrivalSet(time=torch.full((2,), 0.010), amplitude=torch.ones(2, 1),
                          direction=dirs, phase=torch.zeros(2),
                          distance=torch.zeros(2), path_length=torch.ones(2))
    ang, pat = _sweep(arrivals, elements)
    peaks = sorted(ang[i].item() for i in range(1, len(ang) - 1)
                   if pat[i] > pat[i - 1] and pat[i] > pat[i + 1] and pat[i] > pat.max() * 0.4)
    assert len(peaks) == 2
    for got, want in zip(peaks, bearings):
        assert got == pytest.approx(want, abs=0.2)


# --------------------------------------------------------------------------- #
# Arrivals pulled out of a real traced bundle
# --------------------------------------------------------------------------- #
def _traced_scene(bearing_deg: float, rng: float = 40.0, depth: float = 13.0):
    elements = _ula(depth=10.0)
    centre = elements.mean(0)
    b = math.radians(bearing_deg)
    source = (rng * math.cos(b), rng * math.sin(b), depth)
    scene = Scene(
        field=IsoProfile(C), bottom=FlatHeight(30.0), surface=FlatHeight(0.0),
        source=source, receivers=centre.reshape(1, 3),
        surface_loss=ConstantLoss(0.0, learnable=False, pressure_release=True),
        bottom_loss=ConstantLoss(6.0, learnable=False),
        freqs_khz=torch.tensor([FREQ_HZ / 1e3]),
        step_size=0.2, n_steps=400, max_bounces=3,
    )
    dirs = fibonacci_cone(12000, centre - torch.tensor(source), 40.0)
    return scene, elements, centre, scene.trace(dirs), source


def test_extracted_arrivals_carry_the_right_time_and_direction():
    scene, elements, centre, result, source = _traced_scene(12.0)
    arrivals = extract_arrivals(result, centre, scene.freqs_khz, sigma_d=0.25,
                                max_arrivals=40)
    assert arrivals.n_arrivals > 0

    direct = math.dist(source, centre.tolist())
    first = arrivals.time.min().item()
    assert first == pytest.approx(direct / C, abs=5e-5)
    # Arrival directions are unit vectors pointing the way the wave travels.
    assert torch.allclose(arrivals.direction.norm(dim=-1), torch.ones(arrivals.n_arrivals),
                          atol=1e-9)


def test_surface_bounce_arrivals_carry_a_pi_phase_flip():
    """A pressure-release surface inverts the pressure, so surface-reflected
    arrivals must come back with pi of phase -- the thing an energy-only model
    discards and a beamformer needs."""
    scene, elements, centre, result, source = _traced_scene(0.0)
    arrivals = extract_arrivals(result, centre, scene.freqs_khz, sigma_d=0.4,
                                max_arrivals=200)
    wrapped = torch.remainder(arrivals.phase, 2 * math.pi)
    assert torch.isclose(wrapped, torch.zeros_like(wrapped), atol=1e-9).any()
    assert torch.isclose(wrapped, torch.full_like(wrapped, math.pi), atol=1e-9).any()


def test_beamforming_traced_arrivals_recovers_the_true_bearing():
    """End to end: rays, arrivals, beams.  Individual rays scatter about a
    degree in direction because the nearest launch angle is not the eigenray;
    the coherent sum still lands on the true bearing."""
    bearing = 12.0
    scene, elements, centre, result, source = _traced_scene(bearing)
    arrivals = extract_arrivals(result, centre, scene.freqs_khz, sigma_d=0.25,
                                max_arrivals=40)
    grid = make_time_grid(0.024, 0.032, 900)
    steer, ang = azimuth_steering(721, 45.0)
    power = beamform(arrivals, elements, scene.freqs_khz, grid, steer,
                     sigma_t=3e-5, steer_chunk=120)
    peak = ang[power.max(dim=-1).values[:, 0].argmax()].item()
    assert peak == pytest.approx(bearing, abs=0.3)


def test_beam_power_is_differentiable_in_the_array_and_the_shading():
    elements = _ula().requires_grad_(True)
    shading = shading_window(N_EL, "hamming").requires_grad_(True)
    grid = make_time_grid(0.008, 0.012, 401)
    freqs = torch.tensor([FREQ_HZ / 1e3])
    steer, _ = azimuth_steering(61, 30.0)
    power = beamform(_plane_wave(8.0), elements, freqs, grid, steer,
                     sigma_t=2e-4, shading=shading)
    power.sum().backward()
    for name, p in (("elements", elements), ("shading", shading)):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_empty_arrival_set_is_handled():
    scene, elements, centre, result, _ = _traced_scene(12.0)
    far = centre + torch.tensor([0.0, 5000.0, 0.0])
    arrivals = extract_arrivals(result, far, scene.freqs_khz, sigma_d=0.25)
    assert arrivals.n_arrivals == 0
    grid = make_time_grid(0.024, 0.032, 200)
    steer, _ = azimuth_steering(21, 30.0)
    power = beamform(arrivals, elements, scene.freqs_khz, grid, steer, sigma_t=3e-5)
    assert torch.equal(power, torch.zeros_like(power))


# --------------------------------------------------------------------------- #
# Transmit array factor (the other half of a Mills cross)
# --------------------------------------------------------------------------- #
def test_line_array_factor_peaks_at_one_on_the_mainlobe():
    """Both numerator and denominator vanish there, so the limit has to be taken
    rather than divided."""
    from hydropt import line_array_factor

    for n in (1, 2, 8, 64):
        assert float(line_array_factor(torch.zeros(1), n)) == pytest.approx(1.0)
    steered = line_array_factor(torch.tensor([0.5]), 16, sin_steer=0.5)
    assert float(steered) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [4, 8, 16, 64])
def test_the_nulls_land_where_the_aperture_puts_them(n):
    """A half-wave array of N elements has its first null at sin(theta) = 2/N."""
    from hydropt import line_array_factor

    null = float(line_array_factor(torch.tensor([2.0 / n]), n))
    near = float(line_array_factor(torch.tensor([1.0 / n]), n))
    assert null < 1e-20, f"N={n}: {null:.2e}"
    assert near > 1e-3


def test_beamwidth_follows_one_over_n():
    """The -3 dB width should track 101.5/N degrees, measured off the factor."""
    from hydropt import line_array_factor

    angles = torch.linspace(-40.0, 40.0, 40001)
    for n, expected in ((8, 101.5 / 8), (32, 101.5 / 32)):
        response = line_array_factor(torch.sin(angles * math.pi / 180.0), n)
        inside = angles[response > 0.5]
        width = float(inside.max() - inside.min())
        assert width == pytest.approx(expected, rel=0.06), f"N={n}: {width:.2f}"


def test_wide_spacing_grows_a_grating_lobe():
    """At one-wavelength spacing the array repeats its mainlobe at sin = 1, which
    is why half-wave spacing is the default and not an arbitrary habit."""
    from hydropt import line_array_factor

    sin_angle = torch.linspace(-1.0, 1.0, 20001)
    half = line_array_factor(sin_angle, 8, spacing_wavelengths=0.5)
    full = line_array_factor(sin_angle, 8, spacing_wavelengths=1.0)
    away = sin_angle.abs() > 0.5
    assert float(half[away].max()) < 0.1
    assert float(full[away].max()) > 0.9, "a full-wavelength array should repeat"


def test_steering_moves_the_mainlobe_and_broadens_it():
    """The 1/cos broadening that costs a flat array its edge beams."""
    from hydropt import line_array_factor

    angles = torch.linspace(-90.0, 90.0, 90001)
    sin_a = torch.sin(angles * math.pi / 180.0)
    widths = {}
    for steer_deg in (0.0, 45.0, 60.0):
        response = line_array_factor(sin_a, 32,
                                     sin_steer=math.sin(math.radians(steer_deg)))
        inside = angles[response > 0.5]
        widths[steer_deg] = float(inside.max() - inside.min())
        assert float(angles[response.argmax()]) == pytest.approx(steer_deg, abs=0.1)
    for steer_deg in (45.0, 60.0):
        predicted = widths[0.0] / math.cos(math.radians(steer_deg))
        assert widths[steer_deg] == pytest.approx(predicted, rel=0.08), widths


def test_an_empty_array_is_rejected():
    from hydropt import line_array_factor

    with pytest.raises(ValueError, match="at least one element"):
        line_array_factor(torch.zeros(1), 0)
