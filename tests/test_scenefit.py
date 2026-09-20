"""Fitting the scene to a picture: the resampling, the profile, and the fit itself.

Pinned: the polar-Cartesian resampling round-trips a smooth picture;
the range profile is the sector median smoothed in range; the fit,
started wrong on a picture rendered with hidden values and INDEPENDENT
seeds (a different speckle, a different noise draw), recovers the seabed
strength and the gain to within the speckle's own scatter, and its loss
falls.
"""

import math

import pytest
import torch

from hydropt import (
    RealPicture, SceneFit, cartesian_to_polar, fit_scene, range_profile, load_picture,
)
from test_sequence import _renderer, C


def _to_cartesian(image, bearings, grid_ranges, n=121, x_range=(0.0, 90.0), y_range=(-45.0, 45.0)):
    """examples/15's resampling, polar -> metres, for the round trip."""
    x = torch.linspace(*x_range, n); y = torch.linspace(*y_range, n)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    R = torch.hypot(X, Y); B = torch.rad2deg(torch.atan2(Y, X))
    b0, b1 = float(bearings[0]), float(bearings[-1]); r0, r1 = float(grid_ranges[0]), float(grid_ranges[-1])
    gb = 2.0 * (B - b0) / (b1 - b0) - 1.0; gr = 2.0 * (R - r0) / (r1 - r0) - 1.0
    samples = torch.stack([gr, gb], dim=-1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(image.reshape(1, 1, *image.shape), samples, mode="bilinear",
                                          padding_mode="zeros", align_corners=True)
    return out.reshape(X.shape), x, y


def test_cartesian_to_polar_round_trips_a_smooth_picture():
    bearings = torch.linspace(-30.0, 30.0, 61)
    ranges = torch.linspace(20.0, 80.0, 121)
    B, R = torch.meshgrid(bearings, ranges, indexing="ij")
    polar = 40.0 - 0.2 * R + 3.0 * torch.cos(torch.deg2rad(3.0 * B))
    cart, gx, gy = _to_cartesian(polar, bearings, ranges, n=241)
    back = cartesian_to_polar(cart, gx, gy, bearings, ranges)
    inner = (R > 25.0) & (R < 75.0) & (B.abs() < 25.0)
    assert float((back - polar)[inner].abs().max()) < 0.3


def test_range_profile_is_the_sector_median_smoothed():
    bearings = torch.linspace(-60.0, 60.0, 13)
    ranges = 20.0 + 0.5 * torch.arange(40)
    img = torch.zeros(13, 40)
    img[:, :] = ranges.reshape(1, -1) * 0.0 + 10.0
    img[6, :] = 100.0                              # one bright beam does not move a median
    img[:, 20] = 30.0                              # a bright bin is smoothed over 5 m (10 bins)
    prof = range_profile(img, bearings, ranges, smooth_m=5.0)
    assert prof.shape == (3, 40)
    assert float(prof[1, 5]) == pytest.approx(10.0)
    assert 10.0 < float(prof[1, 20]) < 30.0 and float(prof[1, 20]) == pytest.approx(12.0, abs=0.5)
    assert float(prof[1, 0]) == pytest.approx(10.0)  # replicate padding, no edge droop


def _model(r, **kw):
    n = int(r.elements.shape[0])
    args = dict(elements=r.elements, time_grid=r.time_grid, steer=r.steer, sigma_t=r.sigma_t,
                shading=None, source_level_db=200.0, noise_power=1e-2,
                solid_angle_per_ray=r.solid_angle_per_ray, n_tx=3, n_rx_elev=4, n_elev_beams=2,
                elev_beam_deg=15.0, max_arrivals=2000)
    args.update(kw)
    return SceneFit(r.scene, r.directions, **args)


def test_fit_recovers_seabed_and_gain_from_an_independent_realisation():
    r, grid, bearings = _renderer()
    ranges = grid * C / 2.0
    with torch.no_grad():
        truth = _model(r, seabed_db=-20.0, gain_db=3.0, tilt_deg=0.0, fit=(), seed=11)
        real = RealPicture(10.0 * torch.log10(truth.picture().clamp_min(1e-30)), bearings, ranges)
    model = _model(r, seabed_db=-27.0, gain_db=0.0, tilt_deg=0.0,
                   fit=("seabed_db", "gain_db"), seed=5)
    hist = fit_scene(model, real, steps=25, lr=0.8,
                     sectors_deg=((-20.0, -7.0), (-7.0, 7.0), (7.0, 20.0)), smooth_m=3.0)
    assert hist[-1]["loss"] < 0.5 * hist[0]["loss"]
    v = model.values()
    # seabed and gain are separable here only through the noise floor at the far
    # end; what is pinned is their SUM (the level of the seabed's return) to 1.5 dB
    assert abs((v["seabed_db"] + v["gain_db"]) - (-20.0 + 3.0)) < 1.5


def test_load_picture_npz_polar_and_cartesian(tmp_path):
    import numpy as np
    bearings = torch.linspace(-30.0, 30.0, 61); ranges = torch.linspace(20.0, 80.0, 121)
    B, R = torch.meshgrid(bearings, ranges, indexing="ij")
    polar = 40.0 - 0.2 * R + 3.0 * torch.cos(torch.deg2rad(3.0 * B))
    np.savez(tmp_path / "p.npz", image=polar.numpy(), bearings=bearings.numpy(), ranges=ranges.numpy())
    p = load_picture(tmp_path / "p.npz")
    assert torch.allclose(p.image_db, polar) and p.name == "p"
    cart, gx, gy = _to_cartesian(polar, bearings, ranges, n=241)
    np.savez(tmp_path / "c.npz", image=cart.numpy(), gx=gx.numpy(), gy=gy.numpy())
    q = load_picture(tmp_path / "c.npz", bearings_deg=bearings, ranges_m=ranges)
    inner = (R > 25.0) & (R < 75.0) & (B.abs() < 25.0)
    assert float((q.image_db - polar)[inner].abs().max()) < 0.3
    with pytest.raises(ValueError):
        load_picture(tmp_path / "c.npz")


def test_renderer_reproduces_the_fitted_picture():
    r, grid, bearings = _renderer()
    m = _model(r, seabed_db=-24.0, surface_db=2.0, gain_db=3.0, noise_db=1.0, tilt_deg=-3.0,
               fit=(), seed=9)
    with torch.no_grad():
        mine = m.picture()
        one = lambda d: torch.ones(d.shape[:-1], dtype=d.dtype)
        theirs = m.renderer(tx_pattern=one).picture([], frame=0)
    assert torch.allclose(mine, theirs, rtol=1e-4, atol=1e-6 * float(mine.max()))
