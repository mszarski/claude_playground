"""The 2-D limit: a depth-only profile must keep an in-plane ray in plane, and a
constant sound-speed gradient must produce exact circular arcs."""

import math

import pytest
import torch

from hydropt import FlatHeight, IsoProfile, Scene, directions_from_angles
from hydropt.fields import LinearGradientProfile, MunkProfile

C0, G, Z_SRC = 1500.0, 0.016, 3000.0


def _unbounded_scene(field, **kw):
    """Boundaries pushed far away so nothing reflects."""
    return Scene(field=field, bottom=FlatHeight(1e7), surface=FlatHeight(-1e7),
                 source=(0.0, 0.0, Z_SRC), step_size=10.0, n_steps=800, **kw)


def _plane_fan(elev_deg, azimuth_deg=0.0):
    elev = torch.tensor(elev_deg) * math.pi / 180.0
    azim = torch.full_like(elev, math.radians(azimuth_deg))
    return directions_from_angles(elev, azim)


@pytest.mark.parametrize("azimuth_deg", [0.0, 37.0, 90.0])
def test_depth_only_profile_keeps_rays_in_the_launch_plane(azimuth_deg):
    """With dc/dx = dc/dy = 0 there is no out-of-plane force, so the ray stays
    on the vertical plane containing its launch azimuth."""
    scene = _unbounded_scene(MunkProfile(learnable=False))
    res = scene.trace(_plane_fan([-10.0, -3.0, 0.0, 4.0, 12.0], azimuth_deg))

    # Signed distance from the launch plane, whose horizontal normal is
    # (-sin(azim), cos(azim)).
    a = math.radians(azimuth_deg)
    off_plane = -math.sin(a) * res.pos[..., 0] + math.cos(a) * res.pos[..., 1]
    horiz_extent = res.pos[..., :2].norm(dim=-1).max()
    assert off_plane.abs().max() < 1e-9 * max(float(horiz_extent), 1.0)


def test_iso_profile_gives_straight_lines():
    scene = _unbounded_scene(IsoProfile(C0))
    dirs = _plane_fan([-20.0, 0.0, 20.0])
    res = scene.trace(dirs)
    s = res.arclen.unsqueeze(-1)
    expected = scene.source_position() + s * dirs.unsqueeze(1)
    assert torch.allclose(res.pos, expected, atol=1e-8)
    assert torch.allclose(res.tau, res.arclen / C0, atol=1e-10)


@pytest.mark.parametrize("elev_deg", [2.0, 5.0, 10.0])
def test_linear_gradient_gives_exact_circular_arcs(elev_deg):
    """For c(z) = c0 + g z the ray is a circle of radius R = c(z_src) / (g cos(theta0))
    centred at the depth where c extrapolates to zero, z = -c0 / g."""
    scene = _unbounded_scene(LinearGradientProfile(C0, G))
    res = scene.trace(_plane_fan([elev_deg]))
    pos = res.pos[0]

    def circle_through(p, q, r):
        (ax, az), (bx, bz), (cx, cz) = p, q, r
        d = 2 * (ax * (bz - cz) + bx * (cz - az) + cx * (az - bz))
        ux = ((ax**2 + az**2) * (bz - cz) + (bx**2 + bz**2) * (cz - az)
              + (cx**2 + cz**2) * (az - bz)) / d
        uz = ((ax**2 + az**2) * (cx - bx) + (bx**2 + bz**2) * (ax - cx)
              + (cx**2 + cz**2) * (bx - ax)) / d
        return math.hypot(ax - ux, az - uz), uz

    pts = [(pos[k, 0].item(), pos[k, 2].item()) for k in (0, 400, 800)]
    radius, centre_z = circle_through(*pts)

    c_src = C0 + G * Z_SRC
    radius_theory = c_src / (G * math.cos(math.radians(elev_deg)))
    assert radius == pytest.approx(radius_theory, rel=1e-10)
    assert centre_z == pytest.approx(-C0 / G, abs=1e-4)


def test_snell_invariant_is_conserved():
    """cos(theta) / c is the ray invariant; renormalising the slowness vector each
    step is what keeps it from drifting."""
    scene = _unbounded_scene(LinearGradientProfile(C0, G))
    res = scene.trace(_plane_fan([2.0, 6.0, 11.0]))
    step = res.pos[:, 1:] - res.pos[:, :-1]
    tangent = step / step.norm(dim=-1, keepdim=True)
    c = C0 + G * res.pos[:, :-1, 2]
    invariant = tangent[..., :2].norm(dim=-1) / c
    drift = (invariant.max(1).values - invariant.min(1).values) / invariant.mean(1)
    assert drift.max() < 1e-5


@pytest.mark.parametrize("elev_deg", [3.0, 7.0])
def test_turning_depth_matches_snell(elev_deg):
    """A downgoing ray turns where c(z_t) = c(z_src) / cos(theta_0)."""
    scene = _unbounded_scene(LinearGradientProfile(C0, G))
    scene.n_steps = 4000  # long enough to reach the vertex
    res = scene.trace(_plane_fan([elev_deg]))
    c_src = C0 + G * Z_SRC
    z_turn_theory = (c_src / math.cos(math.radians(elev_deg)) - C0) / G
    assert res.pos[0, :, 2].max().item() == pytest.approx(z_turn_theory, abs=0.05)


def test_travel_time_matches_path_quadrature():
    scene = _unbounded_scene(LinearGradientProfile(C0, G))
    res = scene.trace(_plane_fan([4.0]))
    s, z = res.arclen[0], res.pos[0, :, 2]
    integrand = 1.0 / (C0 + G * z)
    tau_quad = torch.trapezoid(integrand, s)
    assert res.tau[0, -1].item() == pytest.approx(tau_quad.item(), rel=1e-8)


# --------------------------------------------------------------------------- #
# The 3-D case: a range-dependent field must bend rays *out* of the launch plane
# --------------------------------------------------------------------------- #
def _front(dcdy: float, c0: float = 1500.0) -> "GriddedField":
    """Uniform horizontal sound-speed gradient dc/dy, as a 3-D grid."""
    from hydropt import GriddedField

    ny, nx, nz = 5, 5, 5
    dy, y0 = 5000.0, -10_000.0
    # Node values keyed to *world* y, so that c(x, y, z) = c0 + dcdy * y exactly
    # and the analytic curvature below applies without an offset.
    y = y0 + dy * torch.arange(ny, dtype=torch.get_default_dtype())
    values = (c0 + dcdy * y).view(1, ny, 1).expand(nz, ny, nx).contiguous()
    return GriddedField(values, origin=(-20_000.0, y0, -5000.0),
                        spacing=(20_000.0, dy, 5000.0), learnable=False)


@pytest.mark.parametrize("dcdy", [0.004, -0.004])
def test_horizontal_gradient_bends_rays_towards_lower_sound_speed(dcdy):
    """Snell's law in the horizontal plane: ``d(eta)/ds = -(1/c^2) dc/dy``, so a
    ray launched along +x must curve towards whichever side is slower.

    This is the check that the tracer is genuinely three-dimensional -- with the
    horizontal components of grad c dropped it would pass every depth-only test
    in this file and still be wrong here."""
    scene = Scene(field=_front(dcdy), bottom=FlatHeight(1e7), surface=FlatHeight(-1e7),
                  source=(0.0, 0.0, 0.0), step_size=25.0, n_steps=400)
    res = scene.trace(torch.tensor([[1.0, 0.0, 0.0]]))
    final_y = res.pos[0, -1, 1].item()

    assert abs(final_y) > 1.0, "ray did not bend at all"
    # Curves towards lower c: positive dc/dy means slower at -y.
    assert math.copysign(1.0, final_y) == math.copysign(1.0, -dcdy)

    # Radius of curvature in the horizontal plane is c / |dc/dy| for a ray
    # travelling perpendicular to the gradient, so y ~ x^2 / (2 R).
    x = res.pos[0, -1, 0].item()
    radius = C0 / abs(dcdy)
    assert abs(final_y) == pytest.approx(x**2 / (2 * radius), rel=0.02)


def test_horizontal_gradient_leaves_depth_untouched():
    """The same field has no vertical gradient, so a level ray must stay level."""
    scene = Scene(field=_front(0.004), bottom=FlatHeight(1e7), surface=FlatHeight(-1e7),
                  source=(0.0, 0.0, 0.0), step_size=25.0, n_steps=400)
    res = scene.trace(torch.tensor([[1.0, 0.0, 0.0]]))
    assert res.pos[..., 2].abs().max() < 1e-9


def test_gridded_field_matches_its_analytic_values_and_gradient():
    field = _front(0.004)
    pts = torch.tensor([[0.0, 2500.0, 0.0], [1000.0, 7300.0, 100.0]])
    c, grad = field.c_and_grad(pts)
    assert torch.allclose(c, 1500.0 + 0.004 * pts[:, 1], atol=1e-9)
    assert torch.allclose(grad[:, 1], torch.full((2,), 0.004), atol=1e-12)
    assert grad[:, 0].abs().max() < 1e-12 and grad[:, 2].abs().max() < 1e-12
