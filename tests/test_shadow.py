"""Segments through a body, and the shadow that leaves on the seabed.

The shadow is not decoration.  An object lying on the bottom is often easier to
classify by the hole it punches in the reverberation than by its own echo, and
the length of that hole is what gives its height.  These tests check the
geometry against the closed form, because "there is a dark patch" is not a
statement anyone can act on.
"""

import math

import pytest
import torch

from hydropt.mesh import icosphere, segment_mesh_transmission


def _parallel_rays(offsets, *, x0=-20.0, x1=20.0):
    """Rays along +x, offset across it -- so the impact parameter is the offset."""
    z = torch.zeros_like(offsets)
    starts = torch.stack([torch.full_like(offsets, x0), offsets, z], dim=-1)
    ends = torch.stack([torch.full_like(offsets, x1), offsets, z], dim=-1)
    return starts, ends


def _plate(height, distance, half_width=20.0, bottom=30.0):
    """A thin vertical plate standing on the seabed, facing the source."""
    z_top = bottom - height
    v = torch.tensor([[distance, -half_width, bottom], [distance, half_width, bottom],
                      [distance, half_width, z_top], [distance, -half_width, z_top]])
    f = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return v, f


def test_a_sphere_blocks_exactly_its_silhouette():
    """The switch has to land on the radius, not near it."""
    radius = 2.0
    v, f = icosphere(4, radius)
    inscribed = float(v.norm(dim=-1).min())      # the tessellation is inside
    offsets = torch.linspace(0.0, 3.0, 301)
    t = segment_mesh_transmission(*_parallel_rays(offsets), v, f)
    edge = float(offsets[(t == 0.0).nonzero().max()])
    assert inscribed <= edge <= radius
    assert float(t[offsets < inscribed].max()) == 0.0
    assert float(t[offsets > radius].min()) == 1.0


def test_the_segment_ends_are_respected():
    """Occlusion is a property of the path, not of the infinite line."""
    v, f = icosphere(3, 2.0)
    far = torch.tensor([[-20.0, 0.0, 0.0]])
    assert float(segment_mesh_transmission(far, torch.tensor([[-5.0, 0.0, 0.0]]),
                                           v, f)) == 1.0      # stops short
    assert float(segment_mesh_transmission(torch.tensor([[5.0, 0.0, 0.0]]),
                                           torch.tensor([[20.0, 0.0, 0.0]]),
                                           v, f)) == 1.0      # starts past it
    assert float(segment_mesh_transmission(far, torch.tensor([[0.0, 0.0, 0.0]]),
                                           v, f)) == 0.0      # ends inside


def test_back_faces_are_not_culled():
    """A path that leaves a body is blocked by it just as much as one entering.

    Culling by winding is right for scattering -- you cannot see the far side --
    and wrong here: a patch under an overhang, or a source inside a hull, meets
    the inward-facing side first.
    """
    v, f = icosphere(3, 2.0)
    inside = torch.tensor([[0.0, 0.0, 0.0]])
    outside = torch.tensor([[20.0, 0.0, 0.0]])
    assert float(segment_mesh_transmission(inside, outside, v, f)) == 0.0


def test_chunking_and_pruning_do_not_change_the_answer():
    v, f = icosphere(3, 2.0)
    offsets = torch.linspace(0.0, 4.0, 97)
    args = (*_parallel_rays(offsets), v, f)
    one = segment_mesh_transmission(*args, facet_chunk=1)
    many = segment_mesh_transmission(*args, facet_chunk=10_000)
    assert torch.equal(one, many)
    # A bundle nowhere near the body is all clear, and never reaches the test.
    away = torch.linspace(50.0, 60.0, 16)
    assert float(segment_mesh_transmission(*_parallel_rays(away), v, f).min()) == 1.0


@pytest.mark.parametrize("height", [0.5, 1.0, 2.0])
def test_shadow_length_matches_the_grazing_geometry(height):
    """The observable the whole example turns on.

    A body of height ``h`` at horizontal distance ``D``, lit from depth
    ``z_s`` over a bottom at ``z_b``, shadows the bottom out to
    ``X* = D (z_b - z_s) / (z_b - h - z_s)``.  Everything an operator does with
    a shadow -- height, and from height, identity -- rests on this line.
    """
    z_s, z_b, D = 18.0, 30.0, 60.0
    v, f = _plate(height, D)
    source = torch.tensor([[0.0, 0.0, z_s]])
    x = torch.linspace(D + 0.05, D + 40.0, 4000)
    patches = torch.stack([x, torch.zeros_like(x), torch.full_like(x, z_b)], dim=-1)
    t = segment_mesh_transmission(source.expand_as(patches), patches, v, f)
    far_edge = float(x[(t == 0.0).nonzero().max()])
    expected = D * (z_b - z_s) / (z_b - height - z_s)
    assert far_edge == pytest.approx(expected, abs=float(x[1] - x[0]))
    # And the shadow is continuous from the body out to that edge.
    assert float(t[x <= far_edge].max()) == 0.0
    assert float(t[x > far_edge + 1e-6].min()) == 1.0


# --------------------------------------------------------------------------- #
# The shadow in the reverberation itself
# --------------------------------------------------------------------------- #
SRC_Z, BOTTOM_Z, PLATE_X = 18.0, 30.0, 60.0


def _vertical_fan(n=400, lo_deg=4.0, hi_deg=20.0):
    """Rays in the x-z plane, spanning depression angles down onto the bottom."""
    a = torch.linspace(math.radians(lo_deg), math.radians(hi_deg), n)
    return torch.stack([a.cos(), torch.zeros_like(a), a.sin()], dim=-1)


def _bottom_patches(occluders=None, height=2.0):
    from hydropt import ConstantLoss, FlatHeight, IsoProfile, Scene
    from hydropt.reverb import LambertScattering, reverberation_arrivals

    scene = Scene(field=IsoProfile(1500.0), bottom=FlatHeight(BOTTOM_Z),
                  surface=FlatHeight(-1e5), source=(0.0, 0.0, SRC_Z),
                  freqs_khz=torch.tensor([100.0]),
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  step_size=1.0, n_steps=400, max_bounces=1)
    dirs = _vertical_fan()
    arrivals = reverberation_arrivals(
        scene.trace(dirs), dirs, scene.freqs_khz,
        scattering=LambertScattering(-27.0, learnable=False),
        solid_angle_per_ray=1e-4, boundary="bottom",
        surface=scene.surface, bottom=scene.bottom, occluders=occluders,
        generator=torch.Generator().manual_seed(0))
    # Two-way time back to the patch's horizontal distance from the sonar.
    r = arrivals.time.detach() * 1500.0 / 2.0
    return (r ** 2 - (BOTTOM_Z - SRC_Z) ** 2).clamp_min(0.0).sqrt()


@pytest.mark.parametrize("height", [1.0, 3.0])
def test_a_body_on_the_bottom_removes_exactly_the_patches_it_hides(height):
    """The dark band has to start at the body and end where the geometry says.

    Too short and the body looks lower than it is; too long and it looks
    taller.  Since height is read straight off that length, the edge is the
    measurement.
    """
    plate = _plate(height, PLATE_X, bottom=BOTTOM_Z)
    lit = _bottom_patches()
    shadowed = _bottom_patches(occluders=[plate], height=height)
    far_edge = PLATE_X * (BOTTOM_Z - SRC_Z) / (BOTTOM_Z - height - SRC_Z)

    assert lit.numel() > shadowed.numel() > 0
    inside = (lit >= PLATE_X) & (lit <= far_edge)
    assert int(lit.numel() - shadowed.numel()) == int(inside.sum())
    # Nothing survives inside the band, and nothing outside it was touched.
    assert not bool(((shadowed >= PLATE_X) & (shadowed <= far_edge)).any())
    assert torch.equal(shadowed, lit[~inside])


def test_no_occluders_changes_nothing():
    """The feature has to be free when it is not used."""
    assert torch.equal(_bottom_patches(), _bottom_patches(occluders=[]))
