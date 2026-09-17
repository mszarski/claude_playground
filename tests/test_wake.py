"""A vessel's surface wake, against the geometry gravity fixes.

The Kelvin wake has exact results to be held to: the envelope closes at
``asin(1/3) = 19.4712`` degrees whatever the speed, and the transverse waves
have wavelength ``2 pi V^2 / g``.  Neither is put into the model -- both come
out of summing wave groups whose phase speed matches the source's component in
their direction -- so they are real tests of the construction rather than
restatements of it.

The curved case has no closed form, which is the whole reason for building it
this way, so it is tested by the asymmetry a turn must produce.
"""

import math

import pytest
import torch

from hydropt.wake import (
    bubble_wake_gain, froude_number, kelvin_wake_surface, wake_elevation,
    wake_packets,
)

G = 9.80665
KELVIN = math.degrees(math.asin(1.0 / 3.0))


def _straight(speed, duration=100.0, n=201):
    t = torch.linspace(0.0, duration, n)
    return torch.stack([speed * t, torch.zeros_like(t)], dim=-1), t


def _turning(speed, radius, duration=100.0, n=201, sign=1.0):
    """A constant-rate turn: heading sweeps at speed / radius, either way.

    The vessel always sets off along +x; ``sign`` only mirrors which way it
    turns.  (Flipping the along-track term instead sends it backwards, which
    is a different track entirely and not the mirror of anything.)
    """
    t = torch.linspace(0.0, duration, n)
    psi = speed * t / radius
    return torch.stack([radius * torch.sin(psi),
                        sign * radius * (1.0 - torch.cos(psi))], dim=-1), t


def _envelope_deg(centre, vessel, min_astern=20.0):
    """Half-angle of the packet centres behind the vessel, in degrees."""
    rel = centre - vessel
    astern = -rel[:, 0]
    keep = astern > min_astern
    return float(torch.rad2deg(torch.atan2(rel[keep, 1].abs(),
                                           astern[keep])).max())


@pytest.mark.parametrize("speed", [3.0, 5.0, 9.0])
def test_the_envelope_closes_at_the_kelvin_angle(speed):
    """19.47 degrees, and the same at every speed -- which is the surprise.

    A faster vessel makes longer waves, not a wider wake.  That the angle is
    speed-independent is the signature of the deep-water dispersion, and it
    falls out of the group construction rather than being imposed.
    """
    track, times = _straight(speed)
    centre, _, _, _ = wake_packets(track, times, n_directions=400,
                                   decay_time=1e6)
    assert _envelope_deg(centre, track[-1]) == pytest.approx(KELVIN, abs=0.02)


@pytest.mark.parametrize("speed", [3.0, 5.0, 9.0])
def test_the_transverse_wavelength_is_two_pi_v_squared_over_g(speed):
    track, times = _straight(speed)
    _, wavevector, _, _ = wake_packets(track, times, n_directions=401,
                                       decay_time=1e6)
    # k = g / (V cos theta)^2 is smallest along the track, so the longest wave
    # present IS the transverse one.
    k_axial = float(wavevector.norm(dim=-1).min())
    assert 2 * math.pi / k_axial == pytest.approx(2 * math.pi * speed ** 2 / G,
                                                  rel=1e-3)


def test_wavelength_goes_as_speed_squared():
    lam = []
    for speed in (4.0, 8.0):
        _, wavevector, _, _ = wake_packets(*_straight(speed), n_directions=401,
                                           decay_time=1e6)
        k = wavevector.norm(dim=-1)
        lam.append(2 * math.pi / float(k.min()))
    assert lam[1] / lam[0] == pytest.approx(4.0, rel=0.02)


def test_a_turn_throws_the_wake_to_one_side():
    """The reason for groups rather than the textbook integral.

    A straight track gives a wake symmetric about the vessel's heading.  A
    turning one cannot: the groups shed early left from a heading the vessel no
    longer holds, so the pattern lies off to one side of where the bow now
    points, and it lies to opposite sides for opposite turns.  There is no
    closed form for this, which is why the model is built out of the emissions
    rather than out of the answer.
    """
    def offset(sign):
        track, times = _turning(6.0, 150.0, sign=sign)
        centre, _, amplitude, _ = wake_packets(track, times, n_directions=64,
                                               decay_time=1e6)
        heading = track[-1] - track[-2]
        heading = heading / heading.norm()
        port = torch.stack([-heading[1], heading[0]])
        rel = centre - track[-1]
        # amplitude-weighted mean offset to port, over the groups astern
        astern = -(rel * heading).sum(-1)
        keep = astern > 20.0
        w = amplitude[keep]
        return float(((rel[keep] * port).sum(-1) * w).sum() / w.sum())

    straight = wake_packets(*_straight(6.0), n_directions=64, decay_time=1e6)
    rel = straight[0] - _straight(6.0)[0][-1]
    keep = -rel[:, 0] > 20.0
    symmetric = float((rel[keep, 1] * straight[2][keep]).sum()
                      / straight[2][keep].sum())
    assert abs(symmetric) < 1.0                     # straight: no preference
    assert offset(+1.0) > 5.0                       # turning one way
    assert offset(-1.0) < -5.0                      # and the other
    assert offset(+1.0) == pytest.approx(-offset(-1.0), rel=0.05)


def test_the_groups_decay_astern():
    near, far = [], []
    for decay in (20.0, 200.0):
        centre, _, amplitude, _ = wake_packets(*_straight(5.0),
                                               n_directions=64,
                                               decay_time=decay)
        rel = centre - _straight(5.0)[0][-1]
        astern = -rel[:, 0]
        far.append(float(amplitude[astern > 150.0].max()))
        near.append(float(amplitude[astern < 50.0].max()))
    # A short decay time costs the far field far more than the near.
    assert far[0] / far[1] < 0.2 * (near[0] / near[1])


def test_culling_groups_does_not_change_the_surface():
    """The cull is an optimisation: 3 envelope widths must equal 8."""
    track, times = _straight(5.0, duration=40.0, n=81)
    packets = wake_packets(track, times, n_directions=48, decay_time=1e6)
    xs = torch.linspace(float(track[-1, 0]) - 120.0, float(track[-1, 0]), 40)
    ys = torch.linspace(-40.0, 40.0, 30)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    tight = wake_elevation(xy, *packets, reach=3.0)
    loose = wake_elevation(xy, *packets, reach=8.0)
    scale = loose.abs().max().clamp_min(1e-12)
    assert float((tight - loose).abs().max() / scale) < 0.02


def test_shallow_water_is_refused_rather_than_extrapolated():
    """Past Froude 0.7 the wake widens and this model does not know it."""
    track, times = _straight(14.0)
    with pytest.raises(ValueError, match="Froude"):
        kelvin_wake_surface(track, times, water_depth=30.0, spacing=4.0,
                            extent=((0.0, 100.0), (-50.0, 50.0)),
                            n_directions=16)
    assert froude_number(14.0, 30.0) > 0.7
    assert froude_number(5.0, 30.0) < 0.7


def test_the_surface_is_a_height_field_that_adds_to_a_sea():
    """It has to compose with the wind sea, not replace it."""
    track, times = _straight(5.0, duration=40.0, n=81)
    extent, spacing = ((100.0, 260.0), (-60.0, 60.0)), 3.0
    nx = int(round((extent[0][1] - extent[0][0]) / spacing)) + 1
    ny = int(round((extent[1][1] - extent[1][0]) / spacing)) + 1
    sea = torch.full((ny, nx), 30.0)
    field = kelvin_wake_surface(track, times, amplitude=0.4, extent=extent,
                                spacing=spacing, base=sea, n_directions=24)
    h = field.heights if hasattr(field, "heights") else None
    assert h is not None and h.shape == (ny, nx)
    assert float((h - 30.0).abs().max()) == pytest.approx(0.4, rel=1e-6)
    probe = torch.tensor([[180.0, 0.0]])
    assert torch.isfinite(field.height(probe)).all()


def test_the_wake_carries_a_gradient_to_the_vessel_track():
    """Differentiable in the course -- which is what makes it invertible.

    Speed reads off the wavelength as V = sqrt(g lambda / 2 pi), so a wake in
    an image is a measurement of the vessel that made it.
    """
    track, times = _straight(5.0, duration=40.0, n=41)
    track = track.clone().requires_grad_(True)
    packets = wake_packets(track, times, n_directions=32, decay_time=1e6)
    xy = torch.tensor([[150.0, 10.0], [170.0, -20.0]])
    wake_elevation(xy, *packets).sum().backward()
    assert track.grad is not None and bool(torch.isfinite(track.grad).all())
    assert float(track.grad.abs().max()) > 0.0


def test_bad_inputs_are_rejected():
    track, times = _straight(5.0, n=21)
    with pytest.raises(ValueError, match="times"):
        wake_packets(track, times[:-1])
    with pytest.raises(ValueError, match="two points"):
        wake_packets(track[:1], times[:1])
    with pytest.raises(ValueError, match="decay_time"):
        wake_packets(track, times, decay_time=0.0)
    with pytest.raises(ValueError, match="water depth"):
        froude_number(5.0, 0.0)


def _elevation_near(track, times, distance, *, half=6.0, n=41, **kw):
    """Peak |elevation| in a square patch ``distance`` metres astern."""
    packets = wake_packets(track, times, **kw)
    heading = track[-1] - track[-2]
    heading = heading / heading.norm()
    centre = track[-1] - distance * heading
    g = torch.linspace(-half, half, n)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    xy = centre + torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    return float(wake_elevation(xy, *packets).abs().max())


def test_the_peak_is_not_at_the_vessel_and_the_trail_survives_astern():
    """Where the source singularity of the Kelvin integral is, and is not.

    At the emission point every direction's group is on top of every other one,
    and the amplitude of a group whose front has not yet lengthened is
    unbounded.  Fading each group in over the wavelength it has to travel to
    separate from the hull takes the divergence out and moves the maximum
    astern, where linear theory applies.

    What it does NOT do is remove the pile-up entirely: the groups still carry
    no intrinsic wave phase, so emission times within a few wavelengths remain
    coherent and the field within about 30 m of a 6 m/s vessel is several times
    the trail behind it.  That is claude_playground-62h, and until it is fixed
    the near field of this model is not to be read as a bow wave.
    """
    track, times = _straight(6.0, duration=90.0, n=181)
    at = {d: _elevation_near(track, times, d, n_directions=96,
                             max_angle_deg=62.0, decay_time=1e6)
          for d in (0.0, 30.0, 180.0)}
    assert at[0.0] < at[30.0]                   # no spike where the hull is
    assert at[180.0] > 0.03 * at[30.0]          # and a wake, not just a bow


def test_the_bubble_wake_is_a_band_on_the_track_that_widens_with_age():
    """Narrow and strong where it was just laid down, broad and faint later."""
    track, times = _straight(6.0, duration=100.0, n=201)
    fresh, old = track[-1], track[-1] - torch.tensor([400.0, 0.0])

    def gain_db(point, across):
        xy = (point + torch.tensor([0.0, across])).reshape(1, 2)
        return 10 * math.log10(float(bubble_wake_gain(xy, track, times,
                                                      gain_db=15.0)))

    assert gain_db(fresh, 0.0) == pytest.approx(15.0, abs=0.1)
    # Across the band: the fresh wake is a line, the old one a stripe.
    assert gain_db(fresh, 12.0) < 3.0
    assert gain_db(old, 12.0) > 8.0
    # ...and away from it there is no wake at all.
    assert gain_db(fresh, 120.0) == pytest.approx(0.0, abs=0.01)


def test_the_bubble_wake_decays_and_follows_a_turn():
    track, times = _turning(6.0, 120.0)
    # A point on the track 30 s ago is lit; the mirror of it across the
    # vessel's present heading -- where a straight wake would have been -- is
    # not.  This is the same asymmetry the height field has, and the reason a
    # curved track needs the track and not a formula.
    on = track[len(track) // 2].reshape(1, 2)
    chord = (2.0 * track[-1] - on).reshape(1, 2)
    live = bubble_wake_gain(on, track, times, gain_db=15.0)
    off = bubble_wake_gain(chord, track, times, gain_db=15.0)
    assert float(live) > 10.0 and float(off) < 2.0
    # Decay: the same point, remembered for five minutes or for five seconds.
    slow = bubble_wake_gain(on, track, times, gain_db=15.0, decay_time=300.0)
    fast = bubble_wake_gain(on, track, times, gain_db=15.0, decay_time=5.0)
    assert float(fast) < 1.1 < float(slow)


def test_the_bubble_wake_carries_a_gradient_to_the_track():
    track, times = _straight(6.0, duration=60.0, n=121)
    track = track.clone().requires_grad_(True)
    xy = torch.tensor([[200.0, 8.0], [120.0, -14.0]])
    bubble_wake_gain(xy, track, times).sum().backward()
    assert track.grad is not None and bool(torch.isfinite(track.grad).all())
    assert float(track.grad.abs().max()) > 0.0


def test_the_bubble_wake_rejects_the_same_bad_inputs():
    track, times = _straight(5.0, n=21)
    xy = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="times"):
        bubble_wake_gain(xy, track, times[:-1])
    with pytest.raises(ValueError, match="two points"):
        bubble_wake_gain(xy, track[:1], times[:1])
    with pytest.raises(ValueError, match="decay_time"):
        bubble_wake_gain(xy, track, times, decay_time=0.0)
