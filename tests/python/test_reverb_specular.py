"""Removal of the near-specular part of a scattering strength when time spreading carries
it (sim/reverb_specular.py; scenario key `exclude_specular` on a bistatic reverberation item).

Energy bookkeeping, for a source and a receiver over a rough boundary, unit source:
  coherent bounce     exp(-G^2) |R|^2 / (r1 + r2)^2            (time spreading, coherent arrival)
  tail                (1 - exp(-G^2)) |R|^2 / (r1 + r2)^2      (time spreading, taps)
  reverberation       integral of strength / (r1^2 r2^2) over the boundary
The removed part of the strength must integrate to the tail.
"""
import warnings

import numpy as np
import pytest

from gpusonar.sim import bottom as bottom_model
from gpusonar.sim import reverb_specular as rs
from gpusonar.sim import run_scenario

C, F = 1500.0, 8000.0


def hemisphere_integral(fn, grazing_in, n=1201):
    """Integral of fn(g_in, g_out, azimuth) over scattered directions (both azimuth signs)."""
    gs = (np.arange(n) + 0.5) * (0.5 * np.pi / n)
    az = (np.arange(2 * n) + 0.5) * (np.pi / (2 * n))
    g, a = np.meshgrid(gs, az, indexing="ij")
    return 2.0 * np.sum(fn(grazing_in, g, a) * np.cos(g)) * (0.5 * np.pi / n) * (np.pi / (2 * n))


def boundary_integral(fn, source, receiver, half_width=None, n=1601):
    """Reverberation energy at the receiver for a unit source: the strength over the plane
    z = 0 with spherical spreading on both legs. Positions are (x, height above the plane)."""
    (xs, hs), (xr, hr) = source, receiver
    x_spec = xs + (xr - xs) * hs / (hs + hr)
    w = half_width or 0.5 * abs(xr - xs)
    x = x_spec + np.linspace(-w, w, n)
    y = np.linspace(-w, w, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r1, r2 = np.sqrt((xx - xs) ** 2 + yy**2 + hs**2), np.sqrt((xx - xr) ** 2 + yy**2 + hr**2)
    g1, g2 = np.arcsin(hs / r1), np.arcsin(hr / r2)
    # Azimuth between the horizontal directions of travel in and out (0 forward).
    din, dout = np.arctan2(yy, xx - xs), np.arctan2(-yy, xr - xx)
    az = np.abs(np.angle(np.exp(1j * (dout - din))))
    return float(np.sum(fn(g1, g2, az) / (r1 * r1 * r2 * r2)) * (x[1] - x[0]) * (y[1] - y[0]))


@pytest.mark.parametrize("s2", [0.005, 0.02, 0.045])
def test_facets_redirect_all_the_energy_that_falls_on_them(s2):
    """Integral of the facet lobe over scattered directions = sin(g_in): energy conservation
    (to the order of the slopes: no shadowing or multiple reflection in geometric optics)."""
    for deg in (20.0, 40.0, 70.0):
        g = np.radians(deg)
        total = hemisphere_integral(lambda gi, gs, az: rs.facet_lobe(gi, gs, az, s2)[0], g)
        assert total / np.sin(g) == pytest.approx(1.0, abs=0.01 + 3.0 * s2 / np.sin(g) ** 2 * 0.25)


STEEP = ((0.0, 150.0), (300.0, 200.0))          # source and receiver: specular grazing angle 49 degrees
SHALLOW = ((0.0, 30.0), (500.0, 50.0))          # 9 degrees (the ASIAEX geometry)


def tail_energy(geometry, s2, h, reflect):
    (xs, hs), (xr, hr) = geometry
    path = np.hypot(xr - xs, hs + hr)
    gamma2 = (2 * 2 * np.pi * F / C * h * (hs + hr) / path) ** 2
    return np.exp(-gamma2) * reflect**2 / path**2, (1.0 - np.exp(-gamma2)) * reflect**2 / path**2, path


@pytest.mark.parametrize("s2,h", [(0.005, 0.05), (0.02, 0.012), (0.045, 0.012)])
def test_removed_reverberation_is_the_tail_of_the_bounce_eigenray(s2, h):
    """The part taken out of the strength, integrated over the boundary with both legs'
    spreading, is the energy time spreading puts in the tail of the bounce path: to 2.5 % when
    the grazing angle is well above the rms slope. At 9 degrees, where the lobe is as wide
    as the grazing angle, it is larger (9 to 65 %): the stated tolerance of the partition."""
    part = rs.specular_part(s2, h, F, C, 0.6)
    _, tail, path = tail_energy(STEEP, s2, h, 0.6)
    assert boundary_integral(part, *STEEP, half_width=1.5 * path) == pytest.approx(tail, rel=0.025)
    _, tail, path = tail_energy(SHALLOW, s2, h, 0.6)
    ratio = boundary_integral(part, *SHALLOW, half_width=1.5 * path) / tail
    assert 1.05 < ratio < 1.7 and ratio == pytest.approx({0.005: 1.09, 0.02: 1.33, 0.045: 1.65}[s2], abs=0.04)


def test_bookkeeping_with_and_without_the_removal():
    """A strength with a facet lobe of the boundary's own slopes on a Lambert floor. With time
    spreading on: (coherent + tail) + reverberation with the lobe removed = coherent + all
    the scattering, to 1 %. Without the removal the tail is counted twice: the excess is the
    tail energy, here 13 dB above the Lambert reverberation of the whole boundary."""
    s2, h, reflect = 0.02, 0.012, 0.6
    source, receiver = STEEP
    k = 2 * np.pi * F / C
    mu = 10.0 ** (-27.0 / 10.0)

    def strength(gi, gs, az):
        lobe, qz, _ = rs.facet_lobe(gi, gs, az, s2)
        return mu * np.sin(gi) * np.sin(gs) - np.expm1(-(k * h * qz) ** 2) * reflect**2 * lobe

    part = rs.specular_part(s2, h, F, C, reflect)
    coherent, tail, path = tail_energy(STEEP, s2, h, reflect)
    w = 1.5 * path
    scattered = boundary_integral(strength, source, receiver, half_width=w)
    kept = boundary_integral(rs.exclude(strength, part), source, receiver, half_width=w)
    lambert = boundary_integral(lambda gi, gs, az: mu * np.sin(gi) * np.sin(gs), source, receiver, half_width=w)
    truth = coherent + scattered
    assert coherent + tail + kept == pytest.approx(truth, rel=0.01)                 # with the removal
    assert kept == pytest.approx(lambert, rel=1e-6)                                # exactly the lobe went
    double = coherent + tail + scattered
    assert double - truth == pytest.approx(tail, rel=1e-9)                          # without: the tail twice
    assert scattered - lambert == pytest.approx(tail, rel=0.01)                     # the lobe of the strength is that tail
    assert 10 * np.log10(double / truth) == pytest.approx(10 * np.log10(1 + tail / truth), abs=1e-9)
    assert 10 * np.log10(double / truth) > 1.0                                     # predicted and found: over 1 dB too much here


def test_a_strength_without_a_lobe_loses_almost_nothing_and_backscatter_nothing():
    s2, h = 0.02, 0.2                                     # rough: nearly all of the bounce is tail
    part = rs.specular_part(s2, h, F, C, 1.0)
    mu = 10.0 ** (-27.0 / 10.0)
    lambert = lambda gi, gs, az=np.pi: mu * np.sin(gi) * np.sin(gs)       # noqa: E731
    source, receiver = (0.0, 30.0), (500.0, 50.0)
    before = boundary_integral(lambert, source, receiver)
    after = boundary_integral(rs.exclude(lambert, part), source, receiver)
    removed = boundary_integral(part, source, receiver)
    # Lambert has no specular peak: clipping removes at most what Lambert had under the lobe,
    # a small part of its own energy and 200 times less than the tail that was "removed".
    assert 0.0 < (before - after) / before < 0.12
    assert (before - after) < 0.01 * removed
    # Backscatter (azimuth pi), grazing 5 to 60 degrees: the lobe is exp(-cot^2 g / s2) down.
    g = np.radians(np.linspace(5.0, 60.0, 56))
    assert np.max(part(g, g, np.pi) / lambert(g, g)) < 1e-3                         # 2.7e-4 of Lambert at 60 degrees
    assert np.max(part(g[:46], g[:46], np.pi) / lambert(g[:46], g[:46])) < 1e-10   # up to 50 degrees
    # At normal incidence the lobe is the backscatter: |R|^2 / (4 pi s2).
    assert float(part(np.pi / 2, np.pi / 2, np.pi)) == pytest.approx(1.0 / (4 * np.pi * s2), rel=1e-6)


def test_kirchhoff_bottom_model_against_the_tail_it_would_double_count():
    """The fluid-sediment model with `roughness_model: kirchhoff` has a forward lobe of its own
    (from its roughness spectrum, not from the time-spread slopes). How much of it the
    removal takes, against the tail energy: reported, and bounded by the tail."""
    par = dict(rho=1.9, nu=1.1, delta=0.0147, w2=2.0e-5, gamma2=3.25, sigma2=0.002)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lambda gi, gs, az=np.pi: bottom_model.strength(gi, gs, F, C, azimuth=az, roughness_model="kirchhoff", **par)   # noqa: E731
        s2, h, reflect = 0.02, 0.05, 0.3
        part = rs.specular_part(s2, h, F, C, reflect)
        source, receiver = (0.0, 30.0), (500.0, 50.0)
        before = boundary_integral(model, source, receiver, n=401)
        after = boundary_integral(rs.exclude(model, part), source, receiver, n=401)
        tail = boundary_integral(part, source, receiver, n=401)
    assert 0.0 <= before - after <= tail * 1.0001
    assert after <= before


SCENARIO = {"sonar": "flank_32", "fidelity": "image_source", "environment": {"sound_speed": C}, "seed": 9,
            "sonar_overrides": {"receive": {"window": {"range": [60.0, 500.0]}}},
            "propagation": {"surface_z": 0.0, "bottom_z": 100.0, "max_order": 1, "bottom_coefficient": 0.5},
            "time_spread": {"bottom": {"rms_height": 0.05, "mean_square_slope": 0.02}, "taps": 4},
            "platform": {"position": [0.0, 0.0, 40.0]}, "pings": {"count": 1}, "receiver": {"adc": False}, "scene": []}


def test_scenario_key_is_opt_in_and_monostatic_items_are_left_alone():
    item = {"boundary_z": 100.0, "mu_db": -27.0}
    plain = run_scenario(dict(SCENARIO, reverberation=[dict(item)])).records[0].samples.array
    with pytest.warns(UserWarning, match="bistatic"):
        mono = run_scenario(dict(SCENARIO, reverberation=[dict(item, exclude_specular=True)])).records[0].samples.array
    assert np.array_equal(plain, mono)                                   # monostatic: refused, nothing removed
    bistatic = dict(item, geometry="bistatic")
    a = run_scenario(dict(SCENARIO, reverberation=[dict(bistatic)])).records[0].samples.array
    b = run_scenario(dict(SCENARIO, reverberation=[dict(bistatic, exclude_specular=True)])).records[0].samples.array
    c = run_scenario(dict(SCENARIO, reverberation=[dict(bistatic, exclude_specular={"mean_square_slope": 0.02, "rms_height": 0.05, "reflection": 0.5})])).records[0].samples.array
    assert np.array_equal(b, c)                                          # `true` reads the same numbers from time_spread and the provider
    # The projector is beside the array and the window holds the nadir return (60 m below):
    # the bistatic item does what it is told and takes the nadir lobe out. Only right when
    # that return is simulated as a one-way source (module docstring).
    assert np.sum(b.astype(float) ** 2) < 0.9 * np.sum(a.astype(float) ** 2)
    # Past nadir (slant range over 150 m, grazing under 24 degrees) backscatter is not touched: under 0.001 dB.
    # (No surface here: a leg that goes by the surface meets the bottom steeply at longer ranges too.)
    short = {"type": "cw", "f_start": 8000.0, "f_stop": 8000.0, "duration": 0.005}      # so that the nadir return is over
    far = dict(SCENARIO, sonar_overrides={"receive": {"window": {"range": [150.0, 500.0]}}, "transmit": {"waveform": short}},
               propagation={"surface_z": None, "bottom_z": 100.0, "max_order": 1, "bottom_coefficient": 0.5})
    # (Compared with the same code path removing nothing: a boundary of no height. A strength given as a
    # function takes other random numbers than the built-in Lambert law.)
    nothing = {"mean_square_slope": 0.02, "rms_height": 1e-9, "reflection": 0.5}
    a = run_scenario(dict(far, reverberation=[dict(bistatic, exclude_specular=nothing)])).records[0].samples.array
    b = run_scenario(dict(far, reverberation=[dict(bistatic, exclude_specular=True)])).records[0].samples.array
    assert abs(10 * np.log10(np.sum(b.astype(float) ** 2) / np.sum(a.astype(float) ** 2))) < 1e-3
    with pytest.raises(ValueError, match="time spreading is not on"):
        run_scenario(dict(SCENARIO, reverberation=[{"boundary_z": 0.0, "model": "chapman_harris", "wind_speed": 5.0, "geometry": "bistatic", "exclude_specular": True}]))
    no_spread = {k: v for k, v in SCENARIO.items() if k != "time_spread"}
    with pytest.raises(ValueError, match="time_spread"):
        run_scenario(dict(no_spread, reverberation=[dict(bistatic, exclude_specular=True)]))
