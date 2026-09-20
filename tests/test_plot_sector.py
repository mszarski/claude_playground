"""The forward-looking sector display.

A sonar's own picture is a wedge, and drawing it from the beamformer's grid
rather than resampling onto a raster is not only cheaper -- it is the only way
the picture contains nothing the sonar did not measure.  A rectangular grid has
to pad the corners outside the swath, and padding a sonar image with zeros
invents dark water.  These tests pin the geometry that makes the wedge come out
right, and the argument checking that keeps a transposed image from silently
rendering.
"""

import math

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from hydropt.plot import plot_fls_sector


def _power(n_b=37, n_r=60):
    return torch.rand(n_b, n_r) ** 2


def _bearings(n=37, half=60.0):
    return torch.linspace(-half, half, n)


def _ranges(n=60, lo=8.0, hi=90.0):
    return torch.linspace(lo, hi, n)


def test_returns_a_figure_with_one_axis_and_a_colourbar():
    fig = plot_fls_sector(_power(), _bearings(), _ranges())
    assert fig is not None
    # the wedge plus its colourbar
    assert len(fig.axes) == 2


def test_draws_into_a_supplied_axis():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    out = plot_fls_sector(_power(), _bearings(), _ranges(), ax=ax)
    assert out is fig
    assert ax.collections, "nothing was drawn"


def test_the_wedge_spans_the_sector_not_a_rectangle():
    """x = R cos B, y = R sin B: the far corners must be empty, not padded."""
    half, r_hi = 45.0, 90.0
    fig = plot_fls_sector(_power(), _bearings(half=half), _ranges(hi=r_hi))
    ax = fig.axes[0]
    # Across track cannot reach the range limit: a rectangle would, a wedge
    # only reaches r sin(half).  Along track can, straight ahead.
    assert ax.get_xlim()[1] < r_hi * math.sin(math.radians(half)) * 1.2
    assert ax.get_xlim()[1] < r_hi * 0.95
    assert ax.get_ylim()[1] >= r_hi
    # and the drawn cells themselves stay inside the sector
    verts = np.concatenate([p.vertices for p in ax.collections[0].get_paths()])
    brg = np.degrees(np.arctan2(verts[:, 0], verts[:, 1]))
    assert brg.min() >= -half - 2.0 and brg.max() <= half + 2.0


def test_a_time_grid_is_converted_to_range():
    """Two-way time in seconds is recognised by magnitude and scaled by c/2."""
    c, r_hi = 1500.0, 90.0
    t = _ranges(hi=r_hi) * 2.0 / c
    a = plot_fls_sector(_power(), _bearings(), _ranges(hi=r_hi))
    b = plot_fls_sector(_power(), _bearings(), t, sound_speed=c)
    assert a.axes[0].get_ylim() == pytest.approx(b.axes[0].get_ylim(), rel=1e-9)


def test_dynamic_range_sets_the_colour_floor():
    fig = plot_fls_sector(_power(), _bearings(), _ranges(), dynamic_range=18.0)
    mesh = fig.axes[0].collections[0]
    assert mesh.get_clim() == pytest.approx((-18.0, 0.0))


def test_peak_is_zero_db_whatever_the_absolute_level():
    """The display is normalised, so scaling the input must not move anything."""
    p = _power()
    a = plot_fls_sector(p, _bearings(), _ranges())
    b = plot_fls_sector(p * 1e7, _bearings(), _ranges())
    assert (a.axes[0].collections[0].get_array().max()
            == pytest.approx(b.axes[0].collections[0].get_array().max()))


def test_cells_are_drawn_as_edges_not_centres():
    """With a curved mesh matplotlib cannot infer edges, and a beam cell should
    cover the extent it actually spans -- which grows with range."""
    n_b, n_r = 37, 60
    fig = plot_fls_sector(_power(n_b, n_r), _bearings(n_b), _ranges(n_r))
    mesh = fig.axes[0].collections[0]
    # flat shading keeps one value per cell, so the array is the data itself
    assert mesh.get_array().size == n_b * n_r


def test_range_rings_and_spokes_are_drawn():
    fig = plot_fls_sector(_power(), _bearings(), _ranges(), ring_step=25.0)
    ax = fig.axes[0]
    assert len(ax.lines) > 4          # rings, spokes, swath edges, vehicle


def test_overlays_are_drawn_in_metres():
    fig = plot_fls_sector(_power(), _bearings(), _ranges(),
                          overlays=[([5.0, 9.0], [50.0, 58.0], "-", "boat")])
    ax = fig.axes[0]
    assert any(ln.get_label() == "boat" for ln in ax.lines)


def test_rejects_a_non_2d_power_array():
    with pytest.raises(ValueError, match=r"\[bearings, ranges\]"):
        plot_fls_sector(torch.rand(37, 1, 60), _bearings(), _ranges())


def test_rejects_a_transposed_image():
    """The commonest way to get this wrong, and silent if unchecked."""
    with pytest.raises(ValueError, match="bearings"):
        plot_fls_sector(_power(37, 60).T, _bearings(37), _ranges(60))


def test_asymmetric_sectors_work():
    fig = plot_fls_sector(_power(), torch.linspace(-20.0, 70.0, 37), _ranges())
    assert fig.axes[0].get_xlim()[1] > fig.axes[0].get_xlim()[0]


def test_a_shared_reference_makes_two_panels_comparable():
    """Without it, a panel that is uniformly weaker looks identical to a strong
    one, because each is normalised to its own peak -- which defeats the
    purpose of putting them side by side."""
    p = _power()
    strong = plot_fls_sector(p, _bearings(), _ranges())
    weak_own = plot_fls_sector(p * 0.25, _bearings(), _ranges())
    ref = float(p.max())
    weak_ref = plot_fls_sector(p * 0.25, _bearings(), _ranges(), reference=ref)

    top = lambda fig: float(fig.axes[0].collections[0].get_array().max())
    assert top(weak_own) == pytest.approx(top(strong), abs=1e-9)   # the problem
    assert top(weak_ref) == pytest.approx(top(strong) - 6.0206, abs=0.01)


def test_reference_does_not_change_the_geometry():
    p = _power()
    a = plot_fls_sector(p, _bearings(), _ranges())
    b = plot_fls_sector(p, _bearings(), _ranges(), reference=float(p.max()) * 10)
    assert a.axes[0].get_xlim() == pytest.approx(b.axes[0].get_xlim())


def test_the_colourbar_says_what_the_scale_is_relative_to():
    """"dB re peak" stops being true as soon as a reference is passed."""
    power = torch.rand(9, 30) + 1e-6
    bearings = torch.linspace(-30.0, 30.0, 9)
    ranges = torch.linspace(10.0, 60.0, 30)
    import matplotlib.pyplot as plt
    labels = []
    for kw in ({}, {"reference": 4.0}, {"reference": 4.0,
                                        "colorbar_label": "dB re the seabed"}):
        fig = plot_fls_sector(power, bearings, ranges, **kw)
        labels.append(fig.axes[-1].get_ylabel())
        plt.close(fig)
    assert labels == ["dB re peak", "dB re reference", "dB re the seabed"]
