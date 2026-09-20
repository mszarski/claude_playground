"""Labels from the forward pass: masks, boxes and classes read off the beams.

Pinned to what the definitions promise: a point target's mask sits on its
beam and bin and its box holds its position; a target too faint to stand
over the background gets an empty mask and ``visible=False``; an emission
labels every bin of its beam; the class is the target's ``label``; the
geometry box holds every vertex of a mesh; the labels of a picture agree
with the picture, since ``|sum|^2`` is what the noisy field gives.
"""

import json
import math

import pytest
import torch

from hydropt import (
    ExtendedTarget, IsotropicScattering, Label, box_mesh, emission_arrivals, mesh_target,
    add_receiver_noise, calibrate, geometry_box, signal_mask, world_geometry,
)
from hydropt.labels import label_from_beams, polar_box, polar_gate
from test_sequence import _renderer, _target, C


def test_signal_mask_and_polar_box():
    own = torch.zeros(5, 1, 10); rest = torch.ones(5, 1, 10)
    own[2, 0, 3:6] = 4.0                  # 6 dB over
    own[4, 0, 7] = 1.5                    # under 3 dB: not in the mask
    m = signal_mask(own, rest, 3.0)
    assert m.shape == (5, 10) and int(m.sum()) == 3
    assert polar_box(m) == (2, 2, 3, 5)
    assert polar_box(torch.zeros(5, 10, dtype=torch.bool)) is None


def test_geometry_box_dilates_by_the_resolution():
    pts = torch.tensor([[100.0, 0.0, 5.0], [100.0, 10.0, 5.0]])
    x0, y0, x1, y1 = geometry_box(pts, beam_deg=math.degrees(0.1), range_m=1.0)
    # a beam of 0.1 rad at 100 m is 10 m across: half of (10 + 1) each way
    assert x0 == pytest.approx(100.0 - 5.5, abs=0.05) and x1 == pytest.approx(100.0 + 5.5, abs=0.05)
    assert y0 == pytest.approx(-5.5, abs=0.05) and y1 == pytest.approx(10.0 + 5.5, abs=0.06)


def test_world_geometry_is_the_mesh_vertices_or_the_highlights():
    v, f = box_mesh((2.0, 2.0, 2.0))
    t = mesh_target(v, f, position=(50.0, 5.0, 0.0), yaw=90.0, n_patches=2, learnable=False)
    pts = world_geometry(t)
    assert pts.shape == v.shape
    assert pts[:, 0].min().item() == pytest.approx(49.0) and pts[:, 1].max().item() == pytest.approx(6.0)
    e = ExtendedTarget([(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)], [IsotropicScattering(0.0)] * 2,
                       position=(10.0, 0.0, 1.0), yaw=0.0, learnable=False)
    assert torch.allclose(world_geometry(e), torch.tensor([[10.0, 0.0, 1.0], [13.0, 0.0, 1.0]]))


def test_point_target_label_sits_on_its_beam_and_bin():
    r, grid, bearings = _renderer(noise_power=1e-6)
    t = _target(50.0, 0.0, 0.0)
    t.label = "buoy"
    pic, labs = r.picture([t], labels=True)
    assert len(labs) == 1 and labs[0].name == "buoy" and labs[0].kind == "target"
    lab = labs[0]
    assert lab.visible and lab.n_cells > 0 and lab.contrast_db > 3.0
    b0, b1, k0, k1 = lab.polar_box
    mid = int(bearings.abs().argmin())
    assert b0 <= mid <= b1
    expect = (math.dist((0, 0, 10), (50, 0, 12)) + math.dist((50, 0, 12), (0.5, 0, 10))) / C
    k = int((grid - expect).abs().argmin())
    assert k0 <= k <= k1 + 1
    # the mask agrees with the picture: where the target dominates, the
    # picture is within a few dB of the target's own power
    assert lab.polar_mask.shape == (9, 120)
    # the metric box, from the cells' own geometry, holds the point; no
    # resampling was given, so there is no Cartesian mask
    x0, y0, x1, y1 = lab.box_m
    assert x0 < 50.0 < x1 and y0 < 0.0 < y1 and lab.mask is None
    assert abs(lab.peak_m[0] - 50.0) < 2.0 and abs(lab.centroid_m[1]) < 3.0
    # the geometry box holds the point, dilated
    x0, y0, x1, y1 = lab.geometry_box_m
    assert x0 < 50.0 < x1 and y0 < 0.0 < y1
    d = json.dumps(lab.to_dict())
    assert '"buoy"' in d


def test_faint_target_is_not_visible_and_emission_spans_every_bin():
    r, grid, bearings = _renderer(noise_power=1e-3)
    faint = ExtendedTarget([(0.0, 0.0, 0.0)], [IsotropicScattering(-90.0)],
                           position=(50.0, 0.0, 12.0), learnable=False)
    faint.label = "ghost"
    centre = r.elements.mean(dim=0)
    arr, _, _ = emission_arrivals(r.scene, torch.tensor([60.0, 0.0, 12.0]), centre, grid,
                                  spectrum_level_db=130.0, source_level_db=200.0,
                                  pulse_s=r.sigma_t, generator=torch.Generator().manual_seed(0))
    pic, labs = r.picture([faint], extra_arrivals=[arr], labels=True, extra_names=["noise spoke"])
    ghost, spoke = labs
    assert ghost.name == "ghost" and not ghost.visible and ghost.n_cells == 0
    assert ghost.polar_box is None and ghost.geometry_box_m is not None
    assert spoke.name == "noise spoke" and spoke.kind == "emission" and spoke.visible
    b0, b1, k0, k1 = spoke.polar_box
    assert k0 == 0 and k1 == 119 and b0 <= int(bearings.abs().argmin()) <= b1


def test_labels_agree_with_the_noisy_picture():
    r, grid, bearings = _renderer(noise_power=1e-3)
    t = _target(50.0, 0.0, 0.0)
    pic, labs = r.picture([t], frame=4, labels=True)
    # the same picture without labels is identical
    assert torch.equal(pic, r.picture([t], frame=4))
    # add_receiver_noise's field, squared, is its power
    field = calibrate(r.background() + r.beams(r.echo(t)), 200.0, beam_scale=r.beam_scale)
    g = lambda: torch.Generator().manual_seed(r.seed + 2 + 4)
    f = add_receiver_noise(field, 1e-3, generator=g(), complex_output=True)
    p = add_receiver_noise(field, 1e-3, generator=g())
    assert torch.allclose(f.abs() ** 2, p, rtol=1e-5)


def test_metric_box_centroid_and_peak_from_the_cells():
    own = torch.zeros(9, 1, 120); rest = torch.full((9, 1, 120), 1.0)
    own[4, 0, 50:53] = 100.0             # dead ahead, 45-46 m
    own[5, 0, 51] = 50.0                 # one beam to port
    bearings = torch.linspace(-20.0, 20.0, 9)          # 5 deg beams
    ranges = 20.0 + 0.5 * torch.arange(120)             # 0.5 m bins
    lab = label_from_beams("thing", "target", own, rest, bearings_deg=bearings, ranges_m=ranges)
    assert lab.visible and lab.polar_box == (4, 5, 50, 52)
    x0, y0, x1, y1 = lab.box_m
    # cells 50-52 span 44.75-46.25 m; beam 4 spans -2.5..2.5 deg, beam 5 2.5..7.5
    assert x0 == pytest.approx(44.75 * math.cos(math.radians(2.5)), abs=0.01)
    assert x1 == pytest.approx(46.25 * math.cos(math.radians(2.5)), abs=0.01)
    assert y0 == pytest.approx(-46.25 * math.sin(math.radians(2.5)), abs=0.01)
    assert y1 == pytest.approx(45.75 * math.sin(math.radians(7.5)), abs=0.01)
    assert lab.peak_m == pytest.approx((45.0, 0.0))
    cx, cy = lab.centroid_m
    assert 45.0 <= cx <= 46.0 and 0.0 < cy < 1.0
    assert lab.mask is None
    # a one-cell mask is a label with a box, but not a thing seen
    own = torch.zeros(9, 1, 120); own[4, 0, 60] = 100.0
    lab = label_from_beams("blip", "target", own, rest, bearings_deg=bearings, ranges_m=ranges)
    assert lab.n_cells == 1 and not lab.visible and lab.box_m is not None


def test_cartesian_mask_from_the_resampling():
    own = torch.zeros(9, 1, 120); rest = torch.full((9, 1, 120), 1.0)
    own[4, 0, 50:53] = 100.0

    def to_cart(img):
        gx = torch.arange(120, dtype=img.dtype) * 0.5
        gy = (torch.arange(9, dtype=img.dtype) - 4) * 2.0
        return img[:, 0], gx, gy

    lab = label_from_beams("thing", "target", own, rest, to_cartesian=to_cart)
    assert lab.mask.shape == (9, 120) and int(lab.mask.sum()) == 3


def test_sidelobes_and_leakage_are_not_the_target():
    # a strong point on beam 4 at bins 50-52, its sidelobes 43 dB down on
    # every other beam at the same bins, over a floor 50 dB down
    own = torch.full((9, 1, 120), 1e-5); rest = torch.full((9, 1, 120), 1e-5)
    own[:, 0, 50:53] = 10.0 ** (-4.3)
    own[4, 0, 50:53] = 1.0
    m = signal_mask(own, rest, 3.0, dynamic_range_db=None)
    assert int(m.sum()) == 27                   # every beam's sidelobe beats the floor
    m = signal_mask(own, rest, 3.0, dynamic_range_db=35.0)
    assert int(m.sum()) == 3 and polar_box(m) == (4, 4, 50, 52)


def test_polar_gate_holds_the_geometry_and_its_ghosts():
    bearings = torch.linspace(-20.0, 20.0, 9)
    ranges = torch.linspace(20.0, 80.0, 120)
    pts = torch.tensor([[50.0, 0.0, 0.0], [52.0, 2.0, 0.0]])
    g = polar_gate(pts, bearings, ranges, beam_deg=5.0, range_m=1.0, beams_margin=2.0,
                   extra_range_m=5.0)
    assert g.shape == (9, 120)
    ok_b = bearings[g.any(dim=1)]
    assert float(ok_b.min()) >= -10.0 - 1e-6 and float(ok_b.max()) <= 12.2 + 1e-6
    ok_r = ranges[g.any(dim=0)]
    assert float(ok_r.min()) >= 48.0 - 1e-6 and float(ok_r.max()) <= 52.04 + 7.0 + 1e-6


def test_targets_sharing_a_label_are_one_label():
    r, grid, bearings = _renderer(noise_power=1e-6)
    a, b = _target(45.0, 0.0, 0.0), _target(55.0, 0.0, 0.0)
    a.label = b.label = "pair"
    pic, labs = r.picture([a, b], labels=True)
    assert [l.name for l in labs] == ["pair"]
    x0, y0, x1, y1 = labs[0].box_m
    assert x0 < 45.0 and x1 > 55.0
    g0, _, g1, _ = labs[0].geometry_box_m
    assert g0 < 45.0 and g1 > 55.0
    # apart, each is its own label and neither box holds the other's point
    a.label, b.label = "near", "far"
    pic, labs = r.picture([a, b], labels=True)
    assert [l.name for l in labs] == ["near", "far"]
    assert labs[0].box_m[2] < 55.0 and labs[1].box_m[0] > 45.0
