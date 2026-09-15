"""Extended targets: aspect-dependent scattering and multi-highlight geometry.

Every pattern here has a textbook closed form at its strongest aspect and a
predictable null spacing, so the tests check numbers rather than shapes.  The
composition test is the interesting one: N coherent cylinder sections of length
L/N must reproduce exactly the cross-section of one cylinder of length L, which
ties the multi-highlight machinery to the single-body formula.
"""

from __future__ import annotations

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, CurvedSurfaceScattering, CylinderScattering, ExtendedTarget,
    FlatHeight, IsotropicScattering,
    IsoProfile, PiecewiseLinearProfile, PlateScattering, PointTarget, Scene,
    compose_arrivals, extract_arrivals, make_time_grid, render_echo,
    render_extended_echo, rotation_matrix, target_arrivals, trace,
)
from hydropt.launch import fibonacci_cone
from hydropt.targets import sinc

C = 1500.0
FREQ_KHZ = 100.0
LAM = C / (FREQ_KHZ * 1e3)


@pytest.fixture(autouse=True)
def _double():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _f() -> torch.Tensor:
    return torch.tensor([FREQ_KHZ])


# --------------------------------------------------------------------------- #
# sinc convention
# --------------------------------------------------------------------------- #
def test_sinc_is_the_unnormalised_one():
    """torch.sinc is sin(pi x)/(pi x); the acoustics literature is not.

    Mixing the two misplaces every null by a factor of pi, so pin it.
    """
    assert sinc(torch.zeros(1)).item() == pytest.approx(1.0)
    assert sinc(torch.tensor([math.pi])).item() == pytest.approx(0.0, abs=1e-12)
    assert sinc(torch.tensor([1.0])).item() == pytest.approx(math.sin(1.0), rel=1e-12)


# --------------------------------------------------------------------------- #
# Plate
# --------------------------------------------------------------------------- #
def test_plate_broadside_matches_the_textbook_cross_section():
    """sigma = (A/lambda)^2 at normal-incidence backscatter."""
    a, b = 2.0, 1.0
    plate = PlateScattering(a, b, sound_speed=C, learnable=False)
    n = plate.normal
    sigma = plate.cross_section(n, -n, _f())
    assert float(sigma) == pytest.approx((a * b / LAM) ** 2, rel=1e-12)
    ts = 10 * math.log10(float(sigma))
    assert ts == pytest.approx(20 * math.log10(a * b / LAM), rel=1e-12)


@pytest.mark.parametrize("side", ["length", "width"])
def test_plate_nulls_land_where_the_aperture_says(side):
    """Backscatter nulls at sin(theta) = lambda / (2 a), a the side in question."""
    a, b = 2.0, 1.0
    plate = PlateScattering(a, b, sound_speed=C, learnable=False)
    n = plate.normal
    size, axis = ((a, plate.length_axis) if side == "length"
                  else (b, plate.width_axis))
    theta = math.asin(LAM / (2 * size))
    at_null = math.cos(theta) * n + math.sin(theta) * axis
    just_off = math.cos(theta * 0.99) * n + math.sin(theta * 0.99) * axis
    s_null = float(plate.cross_section(at_null, -at_null, _f()))
    s_off = float(plate.cross_section(just_off, -just_off, _f()))
    assert s_null < 1e-20 * s_off, f"null not at {math.degrees(theta):.4f} deg"
    assert s_off > 0.0


def test_plate_length_axis_is_the_axis_the_caller_asked_for():
    """An implicitly-chosen in-plane axis makes length and width meaningless."""
    plate = PlateScattering(2.0, 1.0, normal=(0, 0, 1), length_axis=(1, 0, 0),
                            sound_speed=C, learnable=False)
    assert plate.length_axis.tolist() == [1.0, 0.0, 0.0]
    assert plate.width_axis.tolist() == [0.0, 1.0, 0.0]
    # A length axis parallel to the normal cannot be honoured; fall back cleanly.
    degenerate = PlateScattering(1.0, 1.0, normal=(1, 0, 0), length_axis=(1, 0, 0),
                                 sound_speed=C, learnable=False)
    assert float(degenerate.length_axis @ degenerate.normal) == pytest.approx(0.0)
    assert float(degenerate.width_axis @ degenerate.normal) == pytest.approx(0.0)
    assert float(degenerate.length_axis.norm()) == pytest.approx(1.0)


def test_an_edge_on_plate_returns_exactly_nothing():
    """Documented artefact, not physics: physical optics has no edge diffraction.

    Pinned because it also means no gradient, so anyone fitting a plate's
    orientation needs to know the aspect is a dead end rather than a slow one.
    """
    plate = PlateScattering(2.0, 1.0, sound_speed=C, learnable=False)
    edge = plate.length_axis
    assert float(plate.cross_section(edge, -edge, _f())) == 0.0


# --------------------------------------------------------------------------- #
# Cylinder
# --------------------------------------------------------------------------- #
def test_cylinder_broadside_matches_uricks_formula():
    """sigma = r L^2 / (2 lambda) broadside."""
    length, radius = 3.0, 0.25
    cyl = CylinderScattering(length, radius, sound_speed=C, learnable=False)
    broad = torch.tensor([0.0, 1.0, 0.0])
    sigma = float(cyl.cross_section(broad, -broad, _f()))
    assert sigma == pytest.approx(radius * length**2 / (2 * LAM), rel=1e-12)


def test_cylinder_nulls_land_at_lambda_over_twice_the_length():
    length, radius = 3.0, 0.25
    cyl = CylinderScattering(length, radius, sound_speed=C, learnable=False)
    axis = torch.tensor([1.0, 0.0, 0.0])
    broad = torch.tensor([0.0, 1.0, 0.0])
    theta = math.asin(LAM / (2 * length))
    at = math.sin(theta) * axis + math.cos(theta) * broad
    off = math.sin(theta / 2) * axis + math.cos(theta / 2) * broad
    assert float(cyl.cross_section(at, -at, _f())) < 1e-20 * float(
        cyl.cross_section(off, -off, _f()))


def test_an_end_on_cylinder_returns_exactly_nothing():
    """Also a documented artefact: the model is the broadside specular one.

    A real cylinder end-on returns its end cap, which is why a target meant to
    stay visible at all aspects wants the cap as a separate highlight.
    """
    cyl = CylinderScattering(3.0, 0.25, sound_speed=C, learnable=False)
    axis = torch.tensor([1.0, 0.0, 0.0])
    assert float(cyl.cross_section(axis, -axis, _f())) == 0.0


def test_cylinder_cross_section_scales_with_frequency_as_one_over_lambda():
    cyl = CylinderScattering(3.0, 0.25, sound_speed=C, learnable=False)
    broad = torch.tensor([0.0, 1.0, 0.0])
    sigma = cyl.cross_section(broad, -broad, torch.tensor([50.0, 100.0, 200.0]))
    assert float(sigma[1] / sigma[0]) == pytest.approx(2.0, rel=1e-12)
    assert float(sigma[2] / sigma[1]) == pytest.approx(2.0, rel=1e-12)


def test_n_coherent_sections_reproduce_the_whole_cylinder():
    """The check that ties multi-highlight geometry to the single-body formula.

    N sections of length L/N, added *in phase*, must give exactly the
    cross-section of one cylinder of length L: amplitude per section is
    (L/N) sqrt(r/2 lambda), so N of them sum to L sqrt(r/2 lambda), whose square
    is r L^2 / (2 lambda).  If the per-section level were wrong by any factor
    this would not close.
    """
    length, radius, n = 3.0, 0.25, 5
    broad = torch.tensor([0.0, 1.0, 0.0])
    whole = CylinderScattering(length, radius, sound_speed=C, learnable=False)
    section = CylinderScattering(length / n, radius, sound_speed=C, learnable=False)
    amp_total = n * float(section.cross_section(broad, -broad, _f())).__abs__() ** 0.5
    assert amp_total**2 == pytest.approx(
        float(whole.cross_section(broad, -broad, _f())), rel=1e-12)


# --------------------------------------------------------------------------- #
# Isotropic pattern and PointTarget agreement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ts_db", [-20.0, 0.0, 6.0])
def test_isotropic_pattern_agrees_with_point_target(ts_db):
    iso = IsotropicScattering(ts_db, learnable=False)
    point = PointTarget((0.0, 0.0, 0.0), ts_db, learnable=False)
    d = torch.tensor([0.3, -0.5, 0.8])
    d = d / d.norm()
    sigma = iso.cross_section(d, -d, _f())
    assert float(sigma) == pytest.approx(float(point.cross_section()), rel=1e-12)


def test_isotropic_scattering_ignores_both_directions():
    iso = IsotropicScattering(-3.0, learnable=False)
    a = torch.randn(7, 3)
    a = a / a.norm(dim=-1, keepdim=True)
    b = torch.randn(7, 3)
    b = b / b.norm(dim=-1, keepdim=True)
    sigma = iso.cross_section(a, b, _f())
    assert sigma.shape == (7, 1)
    assert float(sigma.std()) == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# Rotation and geometry
# --------------------------------------------------------------------------- #
def test_rotation_matrix_is_a_rotation():
    r = rotation_matrix(torch.tensor(0.7), torch.tensor(-0.2), torch.tensor(0.35))
    assert float((r @ r.T - torch.eye(3)).abs().max()) < 1e-14
    assert float(torch.det(r)) == pytest.approx(1.0, rel=1e-14)


def test_yaw_ninety_degrees_turns_the_body_x_axis_into_world_y():
    body = torch.tensor([[1.0, 0.0, 0.0]])
    t = ExtendedTarget(body, 0.0, position=(10.0, 0.0, 5.0), yaw=90.0, learnable=False)
    w = t.world_positions()
    assert w[0].tolist() == pytest.approx([10.0, 1.0, 5.0], abs=1e-12)


def test_cross_section_is_invariant_under_rotating_the_whole_problem():
    """A pattern depending only on the geometry between the directions must not
    care how the body is oriented, provided both directions rotate with it."""
    cyl = CylinderScattering(3.0, 0.25, sound_speed=C, learnable=False)
    offs = torch.zeros(1, 3)
    d = torch.tensor([0.3, 0.8, -0.5])
    d = d / d.norm()
    flat = ExtendedTarget(offs, cyl, yaw=0.0, learnable=False)
    turned = ExtendedTarget(offs, cyl, yaw=37.0, pitch=-11.0, roll=5.0, learnable=False)
    r = turned.rotation()
    s_flat = flat.cross_section(0, d, -d, _f())
    s_turned = turned.cross_section(0, d @ r.T, (-d) @ r.T, _f())
    assert float(s_turned) == pytest.approx(float(s_flat), rel=1e-12)


def test_a_shared_pattern_is_registered_once():
    """Five highlights sharing one pattern must not present five copies of its
    parameters to an optimiser, which would scale their effective learning rate."""
    cyl = CylinderScattering(0.6, 0.25, sound_speed=C)
    offs = torch.zeros(5, 3)
    t = ExtendedTarget(offs, cyl, learnable=False)
    assert len(t.patterns) == 1
    assert all(t.pattern_for(i) is cyl for i in range(5))
    # length + radius, once.
    assert sum(p.numel() for p in t.parameters()) == 2


def test_distinct_patterns_are_kept_distinct():
    a = IsotropicScattering(-10.0, learnable=False)
    b = CylinderScattering(1.0, 0.2, sound_speed=C, learnable=False)
    t = ExtendedTarget(torch.zeros(3, 3), [a, b, a], learnable=False)
    assert len(t.patterns) == 2
    assert t.pattern_for(0) is a and t.pattern_for(1) is b and t.pattern_for(2) is a


def test_scalar_and_list_shorthands_build_isotropic_patterns():
    t = ExtendedTarget(torch.zeros(2, 3), -12.0, learnable=False)
    assert isinstance(t.pattern_for(0), IsotropicScattering)
    t2 = ExtendedTarget(torch.zeros(2, 3), [-12.0, -6.0], learnable=False)
    assert float(t2.pattern_for(0).target_strength_db) == pytest.approx(-12.0)
    assert float(t2.pattern_for(1).target_strength_db) == pytest.approx(-6.0)


def test_mismatched_pattern_count_is_rejected():
    with pytest.raises(ValueError, match="3 patterns for 2 highlights"):
        ExtendedTarget(torch.zeros(2, 3), [-1.0, -2.0, -3.0], learnable=False)


def test_an_empty_target_is_rejected():
    with pytest.raises(ValueError, match="at least one highlight"):
        ExtendedTarget(torch.zeros(0, 3), 0.0)


# --------------------------------------------------------------------------- #
# Propagation: launch direction, and the aspect it makes exact
# --------------------------------------------------------------------------- #
def _shelf_scene(receivers: torch.Tensor, *, iso: bool = False) -> Scene:
    field = (IsoProfile(C, learnable=False) if iso else
             PiecewiseLinearProfile([0.0, 10.0, 30.0], [1520.0, 1500.0, 1490.0],
                                    learnable=False))
    return Scene(field=field, bottom=FlatHeight(30.0), surface=FlatHeight(0.0),
                 source=(0.0, 0.0, 10.0), receivers=receivers,
                 surface_loss=ConstantLoss(2.0, learnable=False),
                 bottom_loss=ConstantLoss(8.0, learnable=False),
                 freqs_khz=torch.tensor([FREQ_KHZ]),
                 step_size=0.25, n_steps=400, max_bounces=4)


def test_launch_direction_is_the_launch_direction_not_the_arrival_one():
    """The whole point of the field: in a refracting ocean a ray does not arrive
    travelling the way it left, and an aspect-dependent scatterer needs the way
    it left."""
    rx = torch.tensor([[60.0, 0.0, 12.0]])
    scene = _shelf_scene(rx)
    dirs = fibonacci_cone(2000, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    result = trace(scene, dirs)
    arr = extract_arrivals(result, rx[0], scene.freqs_khz, sigma_d=0.5)
    assert arr.n_arrivals > 0
    assert arr.launch_direction is not None
    assert float(arr.launch_direction.norm(dim=-1).sub(1.0).abs().max()) < 1e-12
    # Each arrival's launch direction is its own ray's first step.
    first = result.pos[:, 1] - result.pos[:, 0]
    first = first / first.norm(dim=-1, keepdim=True)
    # Every launch direction must be one of the fan's, to machine precision.
    closest = (arr.launch_direction @ first.T).max(dim=1).values
    assert float(closest.min()) > 1.0 - 1e-12
    # And in a refracting medium it must actually differ from the arrival one.
    turned = (arr.launch_direction * arr.direction).sum(-1)
    assert float(turned.min()) < 1.0 - 1e-6, "no ray turned; test cannot discriminate"


def test_launch_direction_equals_arrival_direction_on_a_straight_ray():
    rx = torch.tensor([[60.0, 0.0, 10.0]])
    scene = _shelf_scene(rx, iso=True)
    dirs = fibonacci_cone(400, torch.tensor([1.0, 0.0, 0.0]), 5.0)
    arr = extract_arrivals(trace(scene, dirs), rx[0], scene.freqs_khz, sigma_d=0.5)
    assert arr.n_arrivals > 0
    dot = (arr.launch_direction * arr.direction).sum(-1)
    assert float(dot.min()) > 1.0 - 1e-9


def test_aspect_dependent_composition_needs_a_launch_direction():
    """Reverberation arrivals have none, and must fail loudly rather than be
    silently scattered off the wrong aspect."""
    z, z3 = torch.zeros(1), torch.zeros(1, 3)
    d = torch.tensor([[1.0, 0.0, 0.0]])
    leg = torch.ones(1, 1)
    inbound = compose_arrivals.__globals__["ArrivalSet"](
        z, leg, d, z, z, z, d)
    no_launch = compose_arrivals.__globals__["ArrivalSet"](z, leg, d, z, z, z, None)
    t = ExtendedTarget(torch.zeros(1, 3), CylinderScattering(1.0, 0.2, learnable=False),
                       learnable=False)
    with pytest.raises(ValueError, match="no\n?\\s*launch_direction|launch_direction"):
        compose_arrivals(inbound, no_launch, t, freqs_khz=_f())
    # With one present it composes.
    out = compose_arrivals(inbound, inbound, t, freqs_khz=_f())
    assert out.n_arrivals == 1


def test_extended_target_composition_requires_frequencies():
    z, z3 = torch.zeros(1), torch.zeros(1, 3)
    d = torch.tensor([[1.0, 0.0, 0.0]])
    arr = compose_arrivals.__globals__["ArrivalSet"](z, torch.ones(1, 1), d, z, z, z, d)
    t = ExtendedTarget(torch.zeros(1, 3), 0.0, learnable=False)
    with pytest.raises(ValueError, match="frequency"):
        compose_arrivals(arr, arr, t)


def test_composition_weights_each_pair_by_its_own_aspect():
    """Two outbound arrivals leaving in different directions must be scaled by
    different cross-sections -- that is what 'exact' means here."""
    cyl = CylinderScattering(2.0, 0.25, sound_speed=C, learnable=False)
    t = ExtendedTarget(torch.zeros(1, 3), cyl, learnable=False)
    broad = torch.tensor([0.0, 1.0, 0.0])
    inbound = compose_arrivals.__globals__["ArrivalSet"](
        torch.zeros(1), torch.ones(1, 1), broad.reshape(1, 3),
        torch.zeros(1), torch.zeros(1), torch.zeros(1), broad.reshape(1, 3))
    # One scatters straight back (broadside, strong), one along the axis (weak).
    launch = torch.stack([-broad, torch.tensor([1.0, 0.0, 0.0])])
    outbound = compose_arrivals.__globals__["ArrivalSet"](
        torch.zeros(2), torch.ones(2, 1), launch,
        torch.zeros(2), torch.zeros(2), torch.zeros(2), launch)
    echo = compose_arrivals(inbound, outbound, t, freqs_khz=_f())
    assert echo.n_arrivals == 2
    strong, weak = float(echo.amplitude[0, 0]), float(echo.amplitude[1, 0])

    # Both against the closed form worked out by hand, not against the class.
    length, radius = 2.0, 0.25
    k = 2 * math.pi / LAM
    sigma_broadside = radius * length**2 / (2 * LAM)
    assert strong == pytest.approx(math.sqrt(sigma_broadside), rel=1e-12)
    # The forward-bistatic pair is a sidelobe, not a null: q.axis = k for
    # s = +x against i = +y, so the aperture factor is sinc(k L / 2).
    aperture = math.sin(k * length / 2) / (k * length / 2)
    assert weak == pytest.approx(math.sqrt(sigma_broadside) * abs(aperture), rel=1e-10)
    assert weak < strong / 100.0


# --------------------------------------------------------------------------- #
# Equivalence with the point-target path
# --------------------------------------------------------------------------- #
def test_one_isotropic_highlight_reproduces_the_point_target_echo():
    """render_extended_echo must agree with render_echo where they model the same
    thing, or the extra machinery has changed the physics.

    The fan is passed in explicitly: left to itself each function aims its own
    cone, and two differently-aimed Fibonacci cones are different bundles even
    on the same seed, so the comparison would be of two valid but unequal
    answers.
    """
    torch.manual_seed(0)
    rx = torch.tensor([[0.0, -0.1, 10.0], [0.0, 0.1, 10.0]])
    scene = _shelf_scene(rx)
    # 1200 bins over 50 ms: sigma_t spans several bins, so the kernel is actually
    # sampled.  At 200 bins the bin is 251 us against a 141 us kernel and the
    # splat falls between samples.
    grid = make_time_grid(0.04, 0.09, 1200)
    tx = fibonacci_cone(600, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    pos = (40.0, 0.0, 14.0)
    ts = -6.0
    rx_dirs = fibonacci_cone(800, torch.tensor([-1.0, 0.0, -0.1]), 40.0,
                             generator=torch.Generator().manual_seed(7))

    point = PointTarget(pos, ts, learnable=False)
    want = render_echo(scene, point, tx, rx_dirs, grid, sigma_d=0.5, sigma_t=5e-4)

    ext = ExtendedTarget(torch.zeros(1, 3), IsotropicScattering(ts, learnable=False),
                         position=pos, learnable=False)
    got = render_extended_echo(scene, ext, tx, grid, sigma_d=0.5, sigma_t=5e-4,
                               rx_directions=rx_dirs)
    assert want.etc.shape == got.etc.shape
    peak = float(want.etc.max())
    assert peak > 0.0, "no echo to compare"
    # atol relative to the peak: both go through an FFT, whose round-off is set
    # by the largest term and shows up as +/-1e-22 noise in the empty bins.
    assert torch.allclose(got.etc, want.etc, rtol=1e-9, atol=1e-12 * peak)
    assert float(got.etc.sum()) == pytest.approx(float(want.etc.sum()), rel=1e-9)


def test_multiple_highlights_sum_more_energy_than_one():
    torch.manual_seed(0)
    rx = torch.tensor([[0.0, 0.0, 10.0]])
    scene = _shelf_scene(rx)
    grid = make_time_grid(0.04, 0.09, 150)
    tx = fibonacci_cone(500, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    offs = torch.tensor([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    one = ExtendedTarget(torch.zeros(1, 3), -6.0, position=(40.0, 0.0, 14.0),
                         learnable=False)
    three = ExtendedTarget(offs, -6.0, position=(40.0, 0.0, 14.0), learnable=False)
    kw = dict(sigma_d=0.5, sigma_t=2e-4, n_rx_rays=500, rx_half_angle_deg=40.0)
    e1 = render_extended_echo(scene, one, tx, grid, **kw,
                              generator=torch.Generator().manual_seed(1))
    e3 = render_extended_echo(scene, three, tx, grid, **kw,
                              generator=torch.Generator().manual_seed(1))
    assert float(e3.etc.sum()) > 2.0 * float(e1.etc.sum())


# --------------------------------------------------------------------------- #
# The point of the whole thing: aspect changes the echo
# --------------------------------------------------------------------------- #
def test_a_hull_is_far_louder_broadside_than_end_on_end_to_end():
    """Not a pattern unit test -- the full two-way coherent pipeline."""
    torch.manual_seed(0)
    rx = torch.stack([torch.zeros(8),
                      (torch.arange(8.0) - 3.5) * (LAM / 2),
                      torch.full((8,), 10.0)], dim=-1)
    scene = _shelf_scene(rx)
    tx = fibonacci_cone(1200, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    offs = torch.stack([torch.linspace(-1.0, 1.0, 3), torch.zeros(3),
                        torch.zeros(3)], dim=-1)
    hull = CylinderScattering(2.0 / 3, 0.25, sound_speed=C, learnable=False)

    def energy(yaw: float) -> float:
        t = ExtendedTarget(offs, hull, position=(40.0, 0.0, 14.0), yaw=yaw,
                           learnable=False)
        arr = target_arrivals(scene, t, tx, sigma_d=0.5, n_rx_rays=800,
                              rx_half_angle_deg=40.0, max_arrivals_per_leg=6,
                              generator=torch.Generator().manual_seed(3))
        return float((arr.amplitude ** 2).sum())

    end_on, broadside = energy(0.0), energy(90.0)
    assert broadside > 0.0 and end_on >= 0.0
    assert broadside > 1e4 * end_on, (
        f"aspect barely mattered: broadside {broadside:.3e}, end-on {end_on:.3e}")


def test_target_arrivals_carries_the_range_spread_of_the_body():
    """A 3 m body must produce returns spread over metres, not one arrival --
    this is what a point target cannot do."""
    torch.manual_seed(0)
    rx = torch.tensor([[0.0, 0.0, 10.0]])
    scene = _shelf_scene(rx)
    tx = fibonacci_cone(1500, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    offs = torch.stack([torch.zeros(3), torch.linspace(-1.5, 1.5, 3),
                        torch.zeros(3)], dim=-1)
    t = ExtendedTarget(offs, -6.0, position=(40.0, 0.0, 14.0), learnable=False)
    arr = target_arrivals(scene, t, tx, sigma_d=0.5, n_rx_rays=800,
                          rx_half_angle_deg=40.0, max_arrivals_per_leg=4,
                          generator=torch.Generator().manual_seed(5))
    assert arr.n_arrivals >= 3
    assert float(arr.time.max() - arr.time.min()) > 1e-4
    assert torch.all(arr.time[1:] >= arr.time[:-1]), "arrivals must come out sorted"


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
def test_gradients_reach_position_orientation_and_pattern():
    torch.manual_seed(0)
    rx = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.3, 10.0]])
    scene = _shelf_scene(rx)
    tx = fibonacci_cone(500, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    offs = torch.stack([torch.linspace(-0.8, 0.8, 3), torch.zeros(3),
                        torch.zeros(3)], dim=-1)
    hull = CylinderScattering(0.7, 0.25, sound_speed=C, learnable=True)
    t = ExtendedTarget(offs, hull, position=(40.0, 0.0, 14.0), yaw=70.0,
                       learnable=True)
    arr = target_arrivals(scene, t, tx, sigma_d=0.6, n_rx_rays=500,
                          rx_half_angle_deg=40.0, max_arrivals_per_leg=4,
                          generator=torch.Generator().manual_seed(11))
    assert arr.n_arrivals > 0
    (arr.amplitude.sum() + arr.time.sum()).backward()
    for name, p in [("position", t.position), ("orientation", t.orientation),
                    ("length", hull.length), ("radius", hull.radius)]:
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is not finite"
        assert float(p.grad.abs().sum()) > 0.0, f"{name} gradient is identically zero"


# --------------------------------------------------------------------------- #
# The glint: why a smooth body does not resolve into highlights
# --------------------------------------------------------------------------- #
def test_a_specular_section_off_its_own_broadside_is_suppressed():
    """The reason a smooth hull glints instead of resolving.

    A section of length a has a beamwidth of about lambda / 2a.  A hull long
    enough to subtend several of those puts its end sections that far off their
    own broadside, and they return almost nothing -- so the echo comes from the
    one place where the specular condition holds, not from the whole body.
    `examples/09` measures 83% of the energy arriving from a single section of a
    4 m hull; this is the same statement at the level of the pattern alone.
    """
    a = 0.8
    cyl = CylinderScattering(a, 0.25, sound_speed=C, learnable=False)
    broad = torch.tensor([0.0, 1.0, 0.0])
    axis = torch.tensor([1.0, 0.0, 0.0])
    beamwidth = LAM / (2 * a)  # in radians, as sin(theta)
    on = float(cyl.cross_section(broad, -broad, _f()))

    # Three beamwidths off broadside: deep in the sidelobes.
    theta = math.asin(3 * beamwidth)
    d = math.sin(theta) * axis + math.cos(theta) * broad
    off = float(cyl.cross_section(d, -d, _f()))
    assert off < on / 100.0, (
        f"only {10 * math.log10(on / max(off, 1e-300)):.1f} dB down at three "
        f"beamwidths off broadside")


def test_isotropic_highlights_are_not_suppressed_off_axis():
    """The contrast that makes discrete scatterers spread where a hull glints."""
    iso = IsotropicScattering(-12.0, learnable=False)
    broad = torch.tensor([0.0, 1.0, 0.0])
    axis = torch.tensor([1.0, 0.0, 0.0])
    theta = math.asin(3 * LAM / (2 * 0.8))
    d = math.sin(theta) * axis + math.cos(theta) * broad
    assert float(iso.cross_section(d, -d, _f())) == pytest.approx(
        float(iso.cross_section(broad, -broad, _f())), rel=1e-12)


def test_batched_return_traces_agree_with_tracing_each_highlight_separately():
    """`target_arrivals` traces every highlight's return fan in one pass, by
    handing the tracer a per-ray source position -- which works because `trace`
    broadcasts the source against the directions, so an ``[R, 3]`` source is one
    row per ray.

    The speed-up is large (eight highlights went from 7.6 s of tracing to 1.6 s,
    because it is one Python loop over steps instead of eight) and the arithmetic
    is unchanged, so this pins bit-identical agreement rather than closeness.
    """
    from hydropt.active import _RelocatedScene
    from hydropt.tracer import TraceResult

    scene = _shelf_scene(torch.tensor([[60.0, 0.0, 10.0]]))
    offsets = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.5]])
    target = ExtendedTarget(offsets, -6.0, position=(40.0, 0.0, 14.0),
                            learnable=False)
    world = target.world_positions()
    centre = scene.receivers.reshape(-1, 3).mean(0)
    n = 200

    fans = [fibonacci_cone(n, centre - world[i], 40.0,
                           generator=torch.Generator().manual_seed(4))
            for i in range(3)]
    separate = [trace(_RelocatedScene(scene, world[i]), fans[i]) for i in range(3)]

    sources = torch.cat([world[i].reshape(1, 3).expand(n, 3) for i in range(3)], 0)
    batched = trace(_RelocatedScene(scene, sources), torch.cat(fans, 0))

    for i in range(3):
        sliced = TraceResult(*(t[i * n:(i + 1) * n] for t in batched))
        assert torch.equal(sliced.pos, separate[i].pos), f"highlight {i} positions"
        assert torch.equal(sliced.tau, separate[i].tau), f"highlight {i} times"
        assert torch.equal(sliced.refl_db, separate[i].refl_db), f"highlight {i} loss"


def test_a_per_ray_source_is_what_makes_that_possible():
    """Pinned separately because it is a property of `trace` the batching relies
    on: hand it one source row per ray and each ray starts from its own place."""
    scene = _shelf_scene(torch.tensor([[60.0, 0.0, 10.0]]))
    dirs = fibonacci_cone(12, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    starts = torch.stack([torch.linspace(10.0, 21.0, 12), torch.zeros(12),
                          torch.full((12,), 12.0)], dim=-1)
    from hydropt.active import _RelocatedScene
    result = trace(_RelocatedScene(scene, starts), dirs)
    assert torch.allclose(result.pos[:, 0], starts, atol=1e-12)


# --------------------------------------------------------------------------- #
# CurvedSurfaceScattering
# --------------------------------------------------------------------------- #
"""A convex surface faired in two directions has a specular point at *every*
aspect, so physical optics gives ``sigma = R1 R2 / 4`` independent of both
aspect and frequency.  That independence is the whole reason a boat hull is easy
to see and a straight pipe is not, so it is what these tests pin.
"""


def _mono(pattern, aspect_deg, freqs=None):
    a = math.radians(aspect_deg)
    inc = torch.tensor([[math.sin(a), math.cos(a), 0.0]])
    f = torch.tensor([FREQ_KHZ]) if freqs is None else freqs
    return pattern.cross_section(inc, -inc, f)


def test_sphere_limit_matches_urick():
    """R1 = R2 = a gives TS = 10 log10(a^2 / 4), the textbook rigid sphere."""
    for a in (0.5, 1.0, 2.0):
        pat = CurvedSurfaceScattering(a, a, learnable=False)
        got = 10.0 * math.log10(float(_mono(pat, 0.0).squeeze()))
        assert got == pytest.approx(10.0 * math.log10(a * a / 4.0), abs=1e-12)


def test_sphere_of_radius_two_metres_is_zero_db():
    pat = CurvedSurfaceScattering(2.0, 2.0, learnable=False)
    assert float(_mono(pat, 0.0).squeeze()) == pytest.approx(1.0, abs=1e-12)


def test_cross_section_is_aspect_independent():
    """The property that distinguishes it from a straight cylinder."""
    pat = CurvedSurfaceScattering(0.75, 30.0, learnable=False)
    levels = [float(_mono(pat, a).squeeze())
              for a in (0.0, 1.0, 5.0, 17.0, 45.0, 90.0, 137.0, 180.0)]
    assert max(levels) / min(levels) == pytest.approx(1.0, abs=1e-12)
    assert levels[0] == pytest.approx(0.75 * 30.0 / 4.0, abs=1e-12)


def test_a_straight_cylinder_is_not_aspect_independent():
    """The contrast, measured: 2.4 m of straight cylinder collapses off broadside."""
    straight = CylinderScattering(2.4, 0.75, sound_speed=C, learnable=False)
    curved = CurvedSurfaceScattering(0.75, 30.0, learnable=False)
    at_5deg = float(_mono(straight, 5.0).squeeze()) / float(_mono(straight, 0.0).squeeze())
    assert at_5deg < 1e-4          # 40 dB down 5 degrees off broadside
    assert float(_mono(curved, 5.0).squeeze()) / float(_mono(curved, 0.0).squeeze()) \
        == pytest.approx(1.0, abs=1e-12)


def test_cross_section_is_frequency_independent():
    freqs = torch.tensor([10.0, 100.0, 400.0])
    pat = CurvedSurfaceScattering(0.75, 30.0, learnable=False)
    got = _mono(pat, 23.0, freqs).reshape(-1)
    assert got.shape == (3,)
    assert torch.allclose(got, got[:1].expand(3), atol=1e-12)


def test_radii_are_symmetric_and_sign_insensitive():
    a = CurvedSurfaceScattering(0.75, 30.0, learnable=False)
    b = CurvedSurfaceScattering(30.0, 0.75, learnable=False)
    c = CurvedSurfaceScattering(-0.75, 30.0, learnable=False)
    for other in (b, c):
        assert float(_mono(other, 0.0).squeeze()) == pytest.approx(
            float(_mono(a, 0.0).squeeze()), abs=1e-12)


def test_a_normal_makes_it_one_sided():
    """With a normal, a patch returns only where it is both lit and seen."""
    pat = CurvedSurfaceScattering(1.0, 1.0, normal=(0.0, 1.0, 0.0), learnable=False)
    f = torch.tensor([FREQ_KHZ])
    face = torch.tensor([[0.0, -1.0, 0.0]])          # travelling onto the face
    assert float(pat.cross_section(face, -face, f).squeeze()) > 0.0
    behind = torch.tensor([[0.0, 1.0, 0.0]])         # arriving from behind it
    assert float(pat.cross_section(behind, -behind, f).squeeze()) == pytest.approx(
        0.0, abs=1e-14)


def test_obliquity_follows_cos_squared():
    pat = CurvedSurfaceScattering(1.0, 1.0, normal=(0.0, 1.0, 0.0), learnable=False)
    f = torch.tensor([FREQ_KHZ])
    full = float(pat.cross_section(torch.tensor([[0.0, -1.0, 0.0]]),
                                   torch.tensor([[0.0, 1.0, 0.0]]), f).squeeze())
    t = math.radians(40.0)
    inc = torch.tensor([[math.sin(t), -math.cos(t), 0.0]])
    got = float(pat.cross_section(inc, -inc, f).squeeze())
    assert got / full == pytest.approx(math.cos(t) ** 2, abs=1e-12)


def test_broadcasts_over_a_fan():
    pat = CurvedSurfaceScattering(0.75, 30.0, learnable=False)
    inc = fibonacci_cone(17, torch.tensor([1.0, 0.0, 0.0]), 30.0)
    freqs = torch.tensor([50.0, 100.0])
    got = pat.cross_section(inc, -inc, freqs)
    assert got.shape == (17, 2)
    assert torch.allclose(got, torch.full((17, 2), 0.75 * 30.0 / 4.0), atol=1e-12)


def test_radii_are_learnable_and_carry_gradient():
    pat = CurvedSurfaceScattering(0.75, 30.0, learnable=True)
    assert len(list(pat.parameters())) == 2
    _mono(pat, 31.0).sum().backward()
    for p in pat.parameters():
        assert p.grad is not None and float(p.grad.abs()) > 0.0


def test_not_learnable_registers_no_parameters():
    pat = CurvedSurfaceScattering(0.75, 30.0, learnable=False)
    assert list(pat.parameters()) == []
