"""Triangle-mesh targets, pinned to the closed forms they must reproduce.

A mesh is only worth having if it agrees with the analytic patterns where both
apply, so that is what these tests check rather than shapes: a faceted sphere
against ``sigma = a^2/4``, and a flat facet against ``(A/lambda)^2`` -- the
latter across the whole sinc pattern, nulls included, where it holds to machine
precision because the facet integral is exact rather than approximated.

The one real subtlety is the per-facet phase integral.  Treating a facet as a
point of amplitude ``A e^{i q.c}`` is valid only for ``|q| d << 1``; at 100 kHz
with centimetre facets ``|q| d`` is about 25 radians, and that approximation
came out 16x high even at 82,000 facets.  ``triangle_phase_integral`` is exact,
and the tests check it against quadrature where the phase is wild.
"""

import math

import pytest
import torch

from hydropt import CurvedSurfaceScattering, PlateScattering
from hydropt.mesh import (
    MeshScattering, boat_hull_mesh, facet_geometry, icosphere, load_obj,
    mesh_target, triangle_phase_integral, visible_facets,
)

C = 1500.0
LAM = C / 100e3


def _freq():
    """Built at call time: the float64 fixture is not active at import."""
    return torch.tensor([100.0])


def _mono(pattern, look, freqs=None):
    """Monostatic cross-section looking along `look` (the travel direction)."""
    freqs = _freq() if freqs is None else freqs
    d = torch.as_tensor(look, dtype=torch.get_default_dtype()).reshape(1, 3)
    d = d / d.norm()
    return pattern.cross_section(d, -d, freqs)


# --------------------------------------------------------------------------- #
# the facet integral
# --------------------------------------------------------------------------- #
def _quad(tri, q, n=600):
    """Brute-force midpoint quadrature of exp(i q.r) over one triangle."""
    p = tri.reshape(3, 3)
    e1, e2 = p[1] - p[0], p[2] - p[0]
    area = 0.5 * torch.linalg.cross(e1, e2, dim=-1).norm()
    u = (torch.arange(n, dtype=p.dtype) + 0.5) / n
    U, V = torch.meshgrid(u, u, indexing="ij")
    m = (U + V) <= 1.0
    r = p[0] + U[m].unsqueeze(-1) * e1 + V[m].unsqueeze(-1) * e2
    ph = (r * q).sum(-1)
    return 2 * area * torch.complex(ph.cos(), ph.sin()).sum() / (n * n)


def test_quadrature_converges_to_the_closed_form():
    """The closed form is exact; the midpoint rule is the approximation.

    Asserting a fixed tolerance against quadrature would really be asserting
    the quadrature's own first-order error, so assert the thing that actually
    validates the closed form instead: refine the quadrature and its
    disagreement must fall toward zero, roughly halving per doubling.
    """
    torch.manual_seed(0)
    tri = torch.randn(1, 3, 3, dtype=torch.float64)
    q = torch.randn(3, dtype=torch.float64) * 3.0
    exact = complex(triangle_phase_integral(tri, q.reshape(1, 3)).reshape(()))
    errs = []
    for n in (150, 300, 600, 1200):
        errs.append(abs(complex(_quad(tri, q, n=n)) - exact) / abs(exact))
    assert all(b < a for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < errs[0] / 5.0, errs
    assert errs[-1] < 5e-3, errs


def test_facet_integral_agrees_with_quadrature_to_its_own_accuracy():
    torch.manual_seed(1)
    for _ in range(5):
        tri = torch.randn(1, 3, 3, dtype=torch.float64)
        q = torch.randn(3, dtype=torch.float64) * 2.0
        got = triangle_phase_integral(tri, q.reshape(1, 3)).reshape(())
        want = _quad(tri, q, n=1200)
        assert abs(complex(got) - complex(want)) / abs(complex(want)) < 1e-2


def test_facet_integral_at_zero_phase_is_the_area():
    """q = 0: the integrand is 1, so the integral is the facet's area."""
    tri = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    got = triangle_phase_integral(tri, torch.zeros(1, 3))
    assert float(got.real) == pytest.approx(1.0, abs=1e-12)
    assert float(got.imag) == pytest.approx(0.0, abs=1e-12)


def test_facet_integral_is_stable_in_a_phase_front():
    """The near-degenerate branch: all three vertex phases equal.

    This is where the textbook form divides by zero, and it is exactly where
    the specular return comes from, so it has to be the well-behaved case.
    """
    tri = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    for scale in (0.0, 1e-13, 1e-9, 1e-6):
        q = torch.tensor([[0.0, 0.0, scale]])  # perpendicular: no phase in-plane
        got = triangle_phase_integral(tri, q)
        assert torch.isfinite(got).all()
        assert float(got.abs()) == pytest.approx(0.5, rel=1e-6)


def test_facet_integral_is_symmetric_in_vertex_order():
    tri = torch.tensor([[[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.1, 0.9, 0.0]]])
    q = torch.tensor([[7.0, -3.0, 2.0]])
    a = triangle_phase_integral(tri, q)
    rolled = tri[:, [1, 2, 0]]
    b = triangle_phase_integral(rolled, q)
    assert abs(complex(a) - complex(b)) < 1e-12


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_facet_geometry_on_a_known_triangle():
    v = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    f = torch.tensor([[0, 1, 2]])
    centroid, normal, area = facet_geometry(v, f)
    assert float(area) == pytest.approx(2.0, abs=1e-12)
    assert torch.allclose(normal, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-12)
    assert torch.allclose(centroid, torch.tensor([[2 / 3, 2 / 3, 0.0]]), atol=1e-12)


def test_icosphere_is_a_sphere():
    for sub in (0, 2):
        v, f = icosphere(sub, 2.5)
        assert torch.allclose(v.norm(dim=-1), torch.full((v.shape[0],), 2.5),
                              atol=1e-12)
        assert f.shape[0] == 20 * 4 ** sub


def test_icosphere_normals_point_outward():
    v, f = icosphere(2, 1.0)
    centroid, normal, _ = facet_geometry(v, f)
    assert float((centroid * normal).sum(-1).min()) > 0.0


def test_boat_hull_mesh_dimensions_and_winding():
    v, f = boat_hull_mesh(length=12.0, beam=3.0, draft=1.0)
    assert float(v[:, 0].max() - v[:, 0].min()) == pytest.approx(12.0, rel=1e-9)
    assert float(v[:, 1].abs().max()) <= 1.5 + 1e-9
    assert float(v[:, 2].max()) <= 1.0 + 1e-9
    assert float(v[:, 2].min()) >= -1e-12   # z is down; hull is below the line
    assert f.shape[1] == 3 and f.shape[0] > 0


def test_load_obj_round_trip(tmp_path):
    p = tmp_path / "t.obj"
    p.write_text("# a quad\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    v, f = load_obj(p)
    assert v.shape == (4, 3)
    assert f.shape == (2, 3)          # fan-triangulated
    assert torch.equal(f, torch.tensor([[0, 1, 2], [0, 2, 3]]))


def test_load_obj_accepts_negative_indices_and_slashes(tmp_path):
    p = tmp_path / "t.obj"
    p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf -3/1/1 -2/2/1 -1/3/1\n")
    v, f = load_obj(p)
    assert torch.equal(f, torch.tensor([[0, 1, 2]]))


def test_load_obj_rejects_an_empty_file(tmp_path):
    p = tmp_path / "e.obj"
    p.write_text("# nothing\n")
    with pytest.raises(ValueError, match="vertices"):
        load_obj(p)


# --------------------------------------------------------------------------- #
# the pattern, against closed forms
# --------------------------------------------------------------------------- #
def test_flat_facet_reproduces_the_plate_exactly():
    """Not approximately: the facet integral is exact, so this is machine zero."""
    w, h = 2.0, 0.5
    verts = torch.tensor([[-w / 2, -h / 2, 0.0], [w / 2, -h / 2, 0.0],
                          [w / 2, h / 2, 0.0], [-w / 2, h / 2, 0.0]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    mesh = MeshScattering(verts, faces, sound_speed=C)
    plate = PlateScattering(w, h, normal=(0.0, 0.0, 1.0), sound_speed=C,
                            learnable=False)
    for ang in (0.0, 0.2, 0.5, 1.0, 2.0):     # through the first nulls
        t = math.radians(ang)
        look = [math.sin(t), 0.0, -math.cos(t)]
        got, want = float(_mono(mesh, look)), float(_mono(plate, look))
        assert got == pytest.approx(want, rel=1e-10), f"at {ang} deg"


def test_normal_incidence_is_the_textbook_plate_cross_section():
    w, h = 0.2, 0.3
    verts = torch.tensor([[-w / 2, -h / 2, 0.0], [w / 2, -h / 2, 0.0],
                          [w / 2, h / 2, 0.0], [-w / 2, h / 2, 0.0]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    got = float(_mono(MeshScattering(verts, faces, sound_speed=C), [0, 0, -1.0]))
    assert got == pytest.approx((w * h / LAM) ** 2, rel=1e-10)


def test_a_back_face_is_culled():
    verts = torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0],
                          [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    mesh = MeshScattering(verts, faces, sound_speed=C)
    assert float(_mono(mesh, [0.0, 0.0, 1.0])) == pytest.approx(0.0, abs=1e-20)
    assert float(_mono(mesh, [0.0, 0.0, -1.0])) > 0.0


@pytest.mark.parametrize("radius", [0.5, 1.0])
def test_faceted_sphere_reproduces_the_analytic_cross_section(radius):
    """sigma = a^2/4, which is CurvedSurfaceScattering's whole content."""
    v, f = icosphere(5, radius)
    mesh = MeshScattering(v, f, sound_speed=C, facet_chunk=4096)
    got = float(_mono(mesh, [0.0, 0.0, 1.0]))
    assert abs(10 * math.log10(got / (radius ** 2 / 4))) < 1.0


def test_faceted_sphere_agrees_with_curved_surface_scattering():
    v, f = icosphere(5, 1.0)
    mesh = MeshScattering(v, f, sound_speed=C, facet_chunk=4096)
    analytic = CurvedSurfaceScattering(1.0, 1.0, learnable=False)
    for look in ([0, 0, 1.0], [1, 0, 0.0], [1, 1, 1.0]):
        ratio = float(_mono(mesh, look)) / float(_mono(analytic, look))
        assert abs(10 * math.log10(ratio)) < 1.0, f"at {look}"


def test_the_sphere_is_aspect_independent():
    v, f = icosphere(5, 1.0)
    mesh = MeshScattering(v, f, sound_speed=C, facet_chunk=4096)
    levels = [float(_mono(mesh, look))
              for look in ([0, 0, 1.0], [1, 0, 0.0], [0, 1, 0.0], [1, 1, 1.0])]
    assert 10 * math.log10(max(levels) / min(levels)) < 1.0


def test_point_facet_approximation_would_have_been_wrong():
    """Pins *why* the exact integral is needed, not just that it is used.

    The shortcut amplitude ``A e^{i q.c}`` is what a quick implementation
    reaches for.  At this facet size it is wrong by more than 10 dB, so a test
    that only checked 'a sphere returns something' would not have caught it.
    """
    a, sub = 1.0, 4
    v, f = icosphere(sub, a)
    look = torch.tensor([[0.0, 0.0, 1.0]])
    tri = v[f]
    centroid, normal, area = facet_geometry(v, f)
    k = 2 * math.pi / LAM
    q = k * (-look - look)
    lit = (-(look @ normal.T)).clamp_min(0.0).reshape(-1)
    ph = (centroid * q).sum(-1)
    shortcut = (area * lit * torch.complex(ph.cos(), ph.sin())).sum() / LAM
    shortcut = float(shortcut.abs() ** 2)
    exact = float(_mono(MeshScattering(v, f, sound_speed=C, facet_chunk=4096),
                        [0.0, 0.0, 1.0]))
    assert abs(10 * math.log10(exact / (a * a / 4))) < 2.0
    assert 10 * math.log10(shortcut / exact) > 10.0


def test_vertices_carry_gradient_when_learnable():
    v, f = icosphere(3, 1.0)
    mesh = MeshScattering(v, f, sound_speed=C, learnable=True)
    _mono(mesh, [0.0, 0.0, 1.0]).sum().backward()
    g = mesh.vertices.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0


def test_not_learnable_registers_no_parameters():
    v, f = icosphere(2, 1.0)
    assert list(MeshScattering(v, f).parameters()) == []


def test_facet_chunking_does_not_change_the_answer():
    v, f = icosphere(3, 1.0)
    a = float(_mono(MeshScattering(v, f, sound_speed=C, facet_chunk=64),
                    [0.0, 0.0, 1.0]))
    b = float(_mono(MeshScattering(v, f, sound_speed=C, facet_chunk=100000),
                    [0.0, 0.0, 1.0]))
    assert a == pytest.approx(b, rel=1e-12)


def test_broadcasts_over_a_fan_of_directions_and_bands():
    v, f = icosphere(2, 1.0)
    mesh = MeshScattering(v, f, sound_speed=C)
    inc = torch.nn.functional.normalize(torch.randn(11, 3, dtype=v.dtype), dim=-1)
    got = mesh.cross_section(inc, -inc, torch.tensor([50.0, 100.0]))
    assert got.shape == (11, 2)
    assert torch.isfinite(got).all() and (got >= 0).all()


def test_rejects_out_of_range_faces():
    v = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="out of range"):
        MeshScattering(v, torch.tensor([[0, 1, 9]]))


# --------------------------------------------------------------------------- #
# mesh_target
# --------------------------------------------------------------------------- #
def test_mesh_target_single_patch():
    v, f = boat_hull_mesh(n_long=8, n_around=6)
    t = mesh_target(v, f, position=(60.0, 0.0, 1.0), yaw=90.0)
    assert t.n_highlights == 1
    assert isinstance(t.pattern_for(0), MeshScattering)
    assert t.pattern_for(0).n_facets == f.shape[0]


def test_mesh_target_splits_along_the_longest_axis():
    v, f = boat_hull_mesh(length=12.0, beam=3.0, n_long=12, n_around=6)
    t = mesh_target(v, f, n_patches=4)
    assert t.n_highlights == 4
    # highlights should spread along x (the hull's long axis), not y or z
    h = t.highlights
    assert float(h[:, 0].max() - h[:, 0].min()) > 4.0
    assert float(h[:, 1].max() - h[:, 1].min()) < 1.0
    # every facet is used exactly once
    assert sum(t.pattern_for(i).n_facets for i in range(4)) == f.shape[0]


def test_mesh_target_patches_are_ordered_along_the_body():
    v, f = boat_hull_mesh(n_long=12, n_around=6)
    t = mesh_target(v, f, n_patches=5)
    xs = t.highlights[:, 0]
    assert torch.all(xs[1:] > xs[:-1])


def test_mesh_target_caps_patches_at_the_facet_count():
    v, f = boat_hull_mesh(n_long=3, n_around=3)
    t = mesh_target(v, f, n_patches=10_000)
    assert t.n_highlights == int(f.shape[0])


def test_mesh_target_rejects_zero_patches():
    v, f = boat_hull_mesh(n_long=4, n_around=4)
    with pytest.raises(ValueError, match="at least 1"):
        mesh_target(v, f, n_patches=0)


def test_mesh_target_position_and_shape_gradients():
    v, f = boat_hull_mesh(n_long=6, n_around=5)
    t = mesh_target(v, f, position=(60.0, 0.0, 1.0), yaw=90.0,
                    n_patches=2, learnable=True, learnable_shape=True)
    world = t.world_positions()
    inc = torch.tensor([[1.0, 0.0, 0.0]])
    total = sum(t.cross_section(i, inc, -inc, _freq()).sum()
                for i in range(t.n_highlights))
    (total + world.sum()).backward()
    assert t.position.grad is not None and torch.isfinite(t.position.grad).all()
    g = t.pattern_for(0).vertices.grad
    assert g is not None and float(g.abs().max()) > 0.0


def test_splitting_conserves_the_body_at_low_frequency():
    """Patches are a bookkeeping split, not a physical one.

    Summed coherently with a common phase reference, N patches must give back
    what one patch gives.  Checked where the body is small against a
    wavelength, so the patches' own positions contribute no extra phase.
    """
    v, f = boat_hull_mesh(length=1.0, beam=0.3, draft=0.1, n_long=10, n_around=8)
    look = torch.tensor([[0.0, 0.0, 1.0]])
    freqs = torch.tensor([0.005])         # 300 m wavelength: the body is a point
    whole = MeshScattering(v, f, sound_speed=C, facet_chunk=4096)
    amp_whole = float(_mono(whole, [0.0, 0.0, 1.0], freqs)) ** 0.5
    t = mesh_target(v, f, n_patches=6, learnable=False)
    amp_parts = sum(float(t.pattern_for(i).cross_section(look, -look, freqs)) ** 0.5
                    for i in range(t.n_highlights))
    assert amp_parts == pytest.approx(amp_whole, rel=1e-6)


# --------------------------------------------------------------------------- #
# ellipsoids: the general R1 != R2 case
# --------------------------------------------------------------------------- #
"""A sphere only exercises ``R1 = R2``.  An ellipsoid separates them: looking
along ``-y`` at the ``y = B`` specular point, the principal radii are ``A^2/B``
and ``C^2/B``, so ``sigma = R1 R2 / 4 = A^2 C^2 / (4 B^2)`` -- a closed form
that depends on all three axes and pins the integrator where the sphere cannot.
"""


def _ellipsoid(sub, a, b, c):
    v, f = icosphere(sub, 1.0)
    return v * torch.tensor([a, b, c], dtype=v.dtype), f


@pytest.mark.parametrize("axes", [(2.0, 1.0, 1.0), (1.0, 1.0, 2.0),
                                  (3.0, 1.0, 0.5)])
def test_ellipsoid_reproduces_r1_r2_over_four(axes):
    a, b, c = axes
    v, f = _ellipsoid(5, a, b, c)
    mesh = MeshScattering(v, f, sound_speed=C, facet_chunk=8192)
    got = float(_mono(mesh, [0.0, 1.0, 0.0]))
    exact = a * a * c * c / (4.0 * b * b)
    assert abs(10 * math.log10(got / exact)) < 1.0


def test_ellipsoid_converges_with_refinement():
    a, b, c = 3.0, 1.0, 0.5
    exact = a * a * c * c / (4.0 * b * b)
    errs = []
    for sub in (4, 5):
        v, f = _ellipsoid(sub, a, b, c)
        mesh = MeshScattering(v, f, sound_speed=C, facet_chunk=8192)
        errs.append(abs(10 * math.log10(float(_mono(mesh, [0.0, 1.0, 0.0])) / exact)))
    assert errs[1] < errs[0]


def test_an_elongated_body_is_brighter_broadside_than_a_sphere():
    """sigma scales as A^2 at fixed B, C -- the elongation is the gain."""
    short = MeshScattering(*_ellipsoid(5, 1.0, 1.0, 1.0), sound_speed=C,
                           facet_chunk=8192)
    long_ = MeshScattering(*_ellipsoid(5, 3.0, 1.0, 1.0), sound_speed=C,
                           facet_chunk=8192)
    ratio = float(_mono(long_, [0.0, 1.0, 0.0])) / float(_mono(short, [0.0, 1.0, 0.0]))
    assert 10 * math.log10(ratio) == pytest.approx(20 * math.log10(3.0), abs=0.5)


def _hull_and_transom(v, f):
    """Split the shell facets from the flat transom cap that closes the stern."""
    hub = v.shape[0] - 1
    cap = (f == hub).any(dim=1)
    return ~cap, cap


def test_hull_normals_point_outward():
    """Culling is by the facet's own normal, so a hull wound inward would be
    invisible from outside and would return its far side instead.  This is
    exactly the bug the dimensions-only check above did not catch.

    The transom is excluded: it is an end cap, so "radially outward" is not the
    right question for it -- :func:`test_transom_faces_aft` asks the right one.
    """
    v, f = boat_hull_mesh(12.0, 3.0, 1.0, n_long=40, n_around=16)
    centroid, normal, area = facet_geometry(v, f)
    shell, _ = _hull_and_transom(v, f)
    radial = torch.stack([torch.zeros_like(centroid[:, 1]),
                          centroid[:, 1], centroid[:, 2]], dim=-1)
    radial = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    outward = (normal * radial).sum(-1)[shell]
    assert float(outward.min()) > 0.0
    a = area[shell]
    assert float((a * outward).sum() / a.sum()) > 0.5


def test_transom_faces_aft():
    """A boat is not double-ended.  The stern is a flat plate carrying most of
    the beam and draft, and it is one of the strongest features on the body
    from astern -- so it has to exist, be closed, and point the right way."""
    v, f = boat_hull_mesh(12.0, 3.0, 1.0, n_long=40, n_around=16)
    centroid, normal, area = facet_geometry(v, f)
    _, cap = _hull_and_transom(v, f)
    assert int(cap.sum()) > 0
    assert torch.allclose(normal[cap][:, 0],
                          -torch.ones(int(cap.sum()), dtype=normal.dtype),
                          atol=1e-9)
    assert float(centroid[cap][:, 0].max()) == pytest.approx(-6.0, abs=1e-9)
    assert float(area[cap].sum()) > 0.2


def test_the_hull_is_not_double_ended():
    """The stern carries real beam and draft; only the bow tapers to a point."""
    v, f = boat_hull_mesh(12.0, 3.0, 1.0, n_long=41, n_around=9, transom=0.62)
    stern = v[v[:, 0] < -5.99]
    bow = v[v[:, 0] > 5.99]
    assert float(stern[:, 1].abs().max()) > 0.4 * 1.5
    assert float(stern[:, 2].max()) > 0.4 * 1.0
    assert float(bow[:, 1].abs().max()) < 1e-9
    # widest section is forward of the transom, not at it
    ys = [float(v[v[:, 0].isclose(x, atol=1e-9)][:, 1].abs().max())
          for x in v[:, 0].unique()]
    assert max(ys) > float(stern[:, 1].abs().max()) * 1.2


def test_deadrise_varies_from_boxy_aft_to_a_vee_forward():
    """Sections go from flat-bottomed at the stern to a sharp V at the bow.

    Measured as how much of the section's bounding box it fills: a boxy
    section fills more of it than a V does.
    """
    v, f = boat_hull_mesh(12.0, 3.0, 1.0, n_long=21, n_around=21)
    xs = v[:, 0].unique().sort().values

    def fullness(x):
        sec = v[v[:, 0].isclose(x, atol=1e-9)]
        y, z = sec[:, 1].abs(), sec[:, 2]
        if float(y.max()) < 1e-9 or float(z.max()) < 1e-9:
            return float("nan")
        # mean normalised depth across the section: 1.0 = rectangular, ~0.5 = V
        return float((z / z.max()).mean())

    aft, fwd = fullness(xs[1]), fullness(xs[-4])
    assert aft > fwd, (aft, fwd)


def test_hull_keel_faces_downward():
    """The deepest facets are the bottom; their outward normal is +z (down)."""
    v, f = boat_hull_mesh(12.0, 3.0, 1.0, n_long=40, n_around=16)
    centroid, normal, _ = facet_geometry(v, f)
    deep = centroid[:, 2] > 0.9 * float(centroid[:, 2].max())
    assert float(normal[deep][:, 2].mean()) > 0.9


def test_hull_has_no_degenerate_facets():
    v, f = boat_hull_mesh(12.0, 3.0, 1.0, n_long=40, n_around=16)
    _, normal, area = facet_geometry(v, f)
    assert float(area.min()) > 0.0
    assert torch.isfinite(normal).all()
    assert torch.allclose(normal.norm(dim=-1),
                          torch.ones(f.shape[0], dtype=normal.dtype), atol=1e-10)


def test_hull_is_brightest_from_beneath():
    """A shallow-draft hull's bottom faces down, so the strong specular is
    from underneath -- not from the shallow depression angle a forward-looking
    sonar has.  Pinned because it is the mesh's main physical statement."""
    v, f = boat_hull_mesh(12.0, 3.0, 1.0, n_long=160, n_around=48)
    mesh = MeshScattering(v, f, sound_speed=C, facet_chunk=8192)

    def look(az_deg, el_deg):
        az, el = math.radians(az_deg), math.radians(el_deg)
        return [-math.cos(az) * math.cos(el), math.sin(az) * math.cos(el),
                -math.sin(el)]

    beneath = float(_mono(mesh, look(90.0, 89.0)))
    shallow = float(_mono(mesh, look(90.0, 10.4)))
    assert beneath > shallow * 2.0
    # and it is still a perfectly detectable target on the beam at that angle
    assert 10 * math.log10(shallow) > -12.0


# --------------------------------------------------------------------------- #
# self-occlusion
# --------------------------------------------------------------------------- #
"""Facets are culled by their own normal, which handles the far side of a convex
body but not a facet hidden behind another one: a superstructure over a deck, a
propeller behind a skeg, the far wall of anything concave.

Ray-casting every facet against every other is O(F^2) per direction -- 225
million tests for a 15,000-facet hull, and ``compose_arrivals`` asks for
hundreds of directions.  A depth buffer is O(F), and measured 10% overhead on
that hull.  What these tests pin is that it culls what is genuinely hidden,
leaves everything else exactly alone, and in particular is a *no-op on a convex
body*, which is the property a binning scheme is most likely to break.
"""


def _panel(cx, cz, w=1.0, n=20):
    """A ``w`` x ``w`` panel at ``z = cz``, normal +z, split into n x n cells.

    Asserts its own winding: getting this backwards silently culls the whole
    panel as a back face and makes an occlusion test pass for the wrong reason.
    """
    g = torch.linspace(-w / 2, w / 2, n + 1)
    X, Y = torch.meshgrid(g + cx, g, indexing="ij")
    v = torch.stack([X.reshape(-1), Y.reshape(-1),
                     torch.full(((n + 1) ** 2,), float(cz))], dim=-1)
    f = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b, c, d = a + 1, a + n + 1, a + n + 2
            f += [[a, c, b], [b, c, d]]
    f = torch.tensor(f)
    _, normal, _ = facet_geometry(v, f)
    assert float(normal[:, 2].min()) > 0.99, "panel must face +z"
    return v, f


def _join(*meshes):
    verts, faces, off = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + off)
        off += v.shape[0]
    return torch.cat(verts), torch.cat(faces)


DOWN = [0.0, 0.0, -1.0]      # travelling -z: the observer is at +z, so a
                             # panel at larger z is NEARER and hides one behind.


def test_visible_facets_picks_the_nearest():
    c = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [3.0, 0.0, 5.0]])
    seen = visible_facets(c, torch.tensor([[0.0, 0.0, 1.0]]), 0.5, tolerance=0.2)
    assert seen.tolist() == [[True, False, True]]


def test_visible_facets_follows_the_view_direction():
    c = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    fwd = visible_facets(c, torch.tensor([[0.0, 0.0, 1.0]]), 0.5, tolerance=0.2)
    back = visible_facets(c, torch.tensor([[0.0, 0.0, -1.0]]), 0.5, tolerance=0.2)
    assert fwd.tolist() == [[True, False]]
    assert back.tolist() == [[False, True]]


def test_visible_facets_keeps_what_is_within_tolerance():
    """A continuous surface must not shadow itself just by landing in one bin."""
    c = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]])
    view = torch.tensor([[0.0, 0.0, 1.0]])
    assert visible_facets(c, view, 0.5, tolerance=1.0).all()
    assert not visible_facets(c, view, 0.5, tolerance=0.1).all()


def test_visible_facets_shape_and_detachment():
    c = torch.randn(17, 3, requires_grad=True)
    seen = visible_facets(c, torch.randn(5, 3), 0.4, tolerance=0.5)
    assert seen.shape == (5, 17) and seen.dtype == torch.bool
    assert not seen.requires_grad


@pytest.mark.parametrize("radius", [0.5, 1.0])
def test_occlusion_is_a_no_op_on_a_convex_body(radius):
    """The property a centroid depth buffer most easily breaks.

    Near the limb a sphere runs almost along the line of sight, so one bin
    spans a large depth range and a naive nearest-per-bin rule culls facets
    that nothing is in front of -- it cost 0.26% of the return before the
    margin was widened by obliquity.  A convex body is owed exactly zero.
    """
    v, f = icosphere(4, radius)
    on = MeshScattering(v, f, sound_speed=C, facet_chunk=8192, occlusion=True)
    off = MeshScattering(v, f, sound_speed=C, facet_chunk=8192, occlusion=False)
    for look in ([0, 0, 1.0], [1, 1, 1.0], [-1, 2, 0.5]):
        assert float(_mono(on, look)) == pytest.approx(float(_mono(off, look)),
                                                       rel=1e-9), look


def test_a_hidden_panel_stops_contributing():
    near, far = _panel(0.0, 3.0), _panel(0.0, 0.0)
    alone = MeshScattering(*near, sound_speed=C, occlusion=False)
    both_v, both_f = _join(far, near)
    truth = float(_mono(alone, DOWN))

    off = float(_mono(MeshScattering(both_v, both_f, sound_speed=C,
                                     occlusion=False), DOWN))
    on = float(_mono(MeshScattering(both_v, both_f, sound_speed=C,
                                    occlusion=True), DOWN))
    # Two coherent copies give four times the power; occlusion must give one.
    assert off / truth == pytest.approx(4.0, rel=0.02)
    assert on / truth == pytest.approx(1.0, rel=0.02)


def test_a_panel_that_is_not_in_the_way_is_left_alone():
    far, aside = _panel(0.0, 0.0), _panel(4.0, 3.0)
    v, f = _join(far, aside)
    on = float(_mono(MeshScattering(v, f, sound_speed=C, occlusion=True), DOWN))
    off = float(_mono(MeshScattering(v, f, sound_speed=C, occlusion=False), DOWN))
    assert on == pytest.approx(off, rel=1e-9)


def test_occlusion_follows_the_look_direction():
    """Turn the body over and the other panel is the hidden one."""
    v, f = _join(_panel(0.0, 0.0), _panel(0.0, 3.0))
    mesh = MeshScattering(v, f, sound_speed=C, occlusion=True)
    down = float(_mono(mesh, DOWN))
    up = float(_mono(mesh, [0.0, 0.0, 1.0]))
    # Each panel faces +z, so only one look direction lights anything at all.
    assert down > 0.0
    assert up == pytest.approx(0.0, abs=1e-12)


def test_bistatic_needs_both_ends():
    """Visible from the source is not the same as visible to the receiver."""
    v, f = _join(_panel(0.0, 0.0), _panel(0.0, 2.0))
    inc = torch.tensor([[0.3, 0.0, -1.0]], dtype=torch.get_default_dtype())
    inc = inc / inc.norm()
    sca = torch.tensor([[0.0, 0.0, 1.0]], dtype=inc.dtype)
    freqs = _freq()
    on = float(MeshScattering(v, f, sound_speed=C, occlusion=True)
               .cross_section(inc, sca, freqs))
    off = float(MeshScattering(v, f, sound_speed=C, occlusion=False)
                .cross_section(inc, sca, freqs))
    assert on < off * 0.5


def test_occlusion_does_not_break_the_vertex_gradient():
    v, f = _join(_panel(0.0, 0.0, n=6), _panel(0.0, 3.0, n=6))
    mesh = MeshScattering(v, f, sound_speed=C, occlusion=True, learnable=True)
    _mono(mesh, DOWN).sum().backward()
    g = mesh.vertices.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0


def test_occlusion_can_be_turned_off_for_the_old_behaviour():
    v, f = _join(_panel(0.0, 0.0, n=6), _panel(0.0, 3.0, n=6))
    mesh = MeshScattering(v, f, sound_speed=C, occlusion=False)
    assert "occlusion=off" in repr(mesh)


def test_mesh_target_forwards_the_occlusion_flag():
    v, f = boat_hull_mesh(n_long=8, n_around=6)
    on = mesh_target(v, f, n_patches=2, occlusion=True)
    off = mesh_target(v, f, n_patches=2, occlusion=False)
    assert on.pattern_for(0).occlusion is True
    assert off.pattern_for(0).occlusion is False


def test_splitting_confines_occlusion_to_a_patch():
    """A documented limitation, pinned so it cannot change silently.

    Each patch is its own MeshScattering, so one patch cannot hide another.
    Two panels that fully occlude each other as one mesh stop doing so once
    they are split apart.
    """
    v, f = _join(_panel(0.0, 0.0, n=8), _panel(0.0, 3.0, n=8))
    whole = MeshScattering(v, f, sound_speed=C, occlusion=True)
    split_far = MeshScattering(*_panel(0.0, 0.0, n=8), sound_speed=C,
                               occlusion=True)
    split_near = MeshScattering(*_panel(0.0, 3.0, n=8), sound_speed=C,
                                occlusion=True)
    d = torch.tensor([DOWN], dtype=torch.get_default_dtype())
    freqs = _freq()
    together = float(whole.cross_section(d, -d, freqs))
    apart = (float(split_far.cross_section(d, -d, freqs))
             + float(split_near.cross_section(d, -d, freqs)))
    assert apart > together * 1.5


def test_split_axis_puts_each_hull_of_a_catamaran_in_its_own_patches():
    """The default split axis is wrong for a mesh of separate bodies.

    A catamaran's longest extent is still its length, so the automatic choice
    makes every patch straddle both hulls and puts its highlight midway between
    them -- at a lateral offset that belongs to neither.  Splitting across the
    separation puts every highlight on a hull.
    """
    v, f = boat_hull_mesh(11.0, 2.4, 1.0, n_long=16, n_around=10)
    sep = 5.0
    both = torch.cat([v + torch.tensor([0.0, sep / 2, 0.0]),
                      v - torch.tensor([0.0, sep / 2, 0.0])])
    faces = torch.cat([f, f + v.shape[0]])

    along = mesh_target(both, faces, n_patches=4).highlights
    across = mesh_target(both, faces, n_patches=4, split_axis=1).highlights

    # Splitting along the length: every highlight lands in the gap.
    assert float(along[:, 1].abs().max()) < 0.1 * sep
    # Splitting across it: every highlight sits on one hull or the other.
    assert float(across[:, 1].abs().min()) > 0.3 * sep
    assert (across[:, 1] > 0).any() and (across[:, 1] < 0).any()


def test_split_axis_rejects_an_axis_that_is_not_one_of_three():
    v, f = boat_hull_mesh(n_long=8, n_around=6)
    with pytest.raises(ValueError, match="split_axis"):
        mesh_target(v, f, n_patches=2, split_axis=3)
