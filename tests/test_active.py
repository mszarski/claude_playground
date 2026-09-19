"""Two-way active sonar: echo timing, target strength, and the convolution.

The separable two-leg composition in :mod:`hydropt.active` is only worth having
if it reproduces what an explicit two-way trace would give, so these tests pin
it to closed-form geometry: an echo must arrive at ``(d_in + d_out) / c``, every
in/out multipath pairing must appear at its own predicted time, and the target
strength must scale the result by exactly ``10 ** (TS / 10)``.
"""

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, Scene, make_time_grid,
    PiecewiseLinearProfile,
)
from hydropt.active import (
    PointTarget, render_echo, return_fan, target_arrivals,
)
from hydropt.launch import fibonacci_cone

C = 1500.0
WATER = 30.0
SOURCE = (0.0, 0.0, 10.0)
ARRAY = (0.5, 0.0, 10.0)
TARGET = (40.0, 0.0, 14.0)


def _scene(*, bounded: bool = False, **overrides) -> Scene:
    kw = dict(
        field=IsoProfile(C),
        bottom=FlatHeight(WATER if bounded else 1e5),
        surface=FlatHeight(0.0 if bounded else -1e5),
        source=SOURCE,
        receivers=torch.tensor([list(ARRAY)]),
        surface_loss=ConstantLoss(0.3, learnable=False),
        bottom_loss=ConstantLoss(4.0, learnable=False),
        freqs_khz=torch.tensor([50.0]),
        step_size=0.25,
        n_steps=600,
        max_bounces=4,
    )
    kw.update(overrides)
    return Scene(**kw)


def _fans(target: PointTarget, receivers: torch.Tensor, n: int = 6000,
          half_angle: float = 25.0):
    tx = fibonacci_cone(n, torch.tensor([1.0, 0.0, 0.0]), half_angle)
    rx = return_fan(target, receivers, n, half_angle_deg=half_angle)
    return tx, rx


def _echo(scene, target, grid, *, n: int = 6000, half_angle: float = 25.0,
          sigma_d: float = 0.6, sigma_t: float = 2e-5, **kw):
    tx, rx = _fans(target, scene.receivers, n, half_angle)
    return render_echo(scene, target, tx, rx, grid, sigma_d=sigma_d, sigma_t=sigma_t,
                       ray_chunk=2000, **kw)


def test_echo_arrives_at_the_two_way_travel_time():
    scene = _scene()
    target = PointTarget(TARGET, 0.0, learnable=False)
    d_in = math.dist(SOURCE, TARGET)
    d_out = math.dist(TARGET, ARRAY)
    expected = (d_in + d_out) / C

    grid = make_time_grid(expected - 0.004, expected + 0.004, 1600)
    etc = _echo(scene, target, grid).etc[0, 0]
    peak = grid[etc.argmax()].item()
    # The residual is ray-fan discretisation: the nearest launch direction is
    # not exactly the eigenray, so it passes at a small miss distance.
    assert peak == pytest.approx(expected, abs=5e-5)


def test_monostatic_echo_arrives_at_twice_the_one_way_time():
    scene = _scene(receivers=torch.tensor([list(SOURCE)]))
    target = PointTarget(TARGET, 0.0, learnable=False)
    expected = 2.0 * math.dist(SOURCE, TARGET) / C
    grid = make_time_grid(expected - 0.004, expected + 0.004, 1600)
    etc = _echo(scene, target, grid).etc[0, 0]
    assert grid[etc.argmax()].item() == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("ts_db", [-20.0, -10.0, 0.0, 10.0])
def test_target_strength_scales_the_echo_exactly(ts_db):
    scene = _scene()
    d = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C
    grid = make_time_grid(d - 0.004, d + 0.004, 1200)

    reference = _echo(scene, PointTarget(TARGET, 0.0, learnable=False), grid).etc
    scaled = _echo(scene, PointTarget(TARGET, ts_db, learnable=False), grid).etc
    assert torch.allclose(scaled, reference * 10.0 ** (ts_db / 10.0), rtol=1e-12)


def test_every_multipath_pairing_appears_at_its_predicted_time():
    """With a surface and a seabed, each inbound path pairs with each outbound
    path, and the echo carries the full outer product of their delays."""
    scene = _scene(bounded=True)
    target = PointTarget(TARGET, 0.0, learnable=False)

    def one_way(a, b):
        # Direct and surface-reflected images only: a bottom bounce here needs a
        # 42 deg launch angle, outside the transmit cone used below.
        return [math.dist(a, (b[0], b[1], z)) for z in (b[2], -b[2])]

    pairings = sorted({(p + q) / C for p in one_way(SOURCE, TARGET)
                       for q in one_way(TARGET, ARRAY)})
    grid = make_time_grid(0.050, 0.060, 2000)
    etc = _echo(scene, target, grid, n=8000, half_angle=35.0).etc[0, 0]

    peaks = [grid[i].item() for i in range(1, len(etc) - 1)
             if etc[i] > etc[i - 1] and etc[i] > etc[i + 1] and etc[i] > etc.max() * 0.02]
    for t in pairings:
        if not (grid[0] < t < grid[-1]):
            continue
        assert min(abs(p - t) for p in peaks) < 5e-5, f"no echo near {t * 1000:.3f} ms"


def test_time_kernel_composes_to_the_requested_width():
    """Each leg is rendered at sigma_t/sqrt(2) precisely so that convolving the
    two lands on sigma_t -- not sigma_t*sqrt(2)."""
    scene = _scene()
    target = PointTarget(TARGET, 0.0, learnable=False)
    sigma_t = 4e-5
    d = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C
    grid = make_time_grid(d - 0.002, d + 0.002, 4000)

    tx, rx = _fans(target, scene.receivers)
    etc = render_echo(scene, target, tx, rx, grid, sigma_d=0.6, sigma_t=sigma_t,
                      ray_chunk=2000).etc[0, 0]

    # Second moment of a Gaussian of width sigma is sigma^2.
    w = etc / etc.sum()
    mean = (w * grid).sum()
    width = ((w * (grid - mean) ** 2).sum()).sqrt().item()
    assert width == pytest.approx(sigma_t, rel=0.15)


def test_convolution_is_linear_not_circular():
    """A late echo must not wrap onto an early bin.  Zero-padding the FFT to at
    least 2n-1 is what prevents it; without that the two-way response of a
    distant target reappears at short range as a phantom."""
    scene = _scene()
    target = PointTarget(TARGET, 0.0, learnable=False)
    earliest = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C

    # Window opens well before any physically possible echo.
    grid = make_time_grid(0.0, earliest + 0.004, 3000)
    etc = _echo(scene, target, grid).etc[0, 0]
    before = etc[grid < earliest - 0.001]
    assert before.max() < etc.max() * 1e-9, "energy appeared before the earliest echo"


def test_gradients_reach_the_target_and_the_scene_through_both_legs():
    # Deliberately small.  The claim under test is that gradients *reach* every
    # parameter through both legs, which a few hundred rays settle.  A backward
    # pass at the fan sizes used elsewhere in this file is tens of millions of
    # field evaluations -- minutes of runtime for no extra assurance.
    scene = _scene(bounded=True,
                   field=PiecewiseLinearProfile([0.0, 15.0, WATER],
                                                [1512.0, 1505.0, 1503.0]),
                   bottom_loss=ConstantLoss(4.0),
                   step_size=1.0, n_steps=140)
    target = PointTarget(TARGET, 0.0)
    d = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C
    grid = make_time_grid(d - 0.004, d + 0.006, 400)

    _echo(scene, target, grid, n=500, half_angle=30.0,
          sigma_d=3.0, sigma_t=2e-4).etc.sum().backward()

    for name, p in (("target.position", target.position),
                    ("target.target_strength_db", target.target_strength_db),
                    ("field.values", scene.field.values),
                    ("bottom_loss.loss_db", scene.bottom_loss.loss_db)):
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is non-finite"
        assert p.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_relocated_scene_view_shares_every_parameter():
    """Leg 2 traces from a proxy whose source is the target.  If that proxy
    copied sub-modules instead of sharing them, gradients from the outbound leg
    would silently vanish."""
    from hydropt.active import _RelocatedScene

    scene = _scene(bounded=True, bottom_loss=ConstantLoss(4.0))
    moved = _RelocatedScene(scene, torch.tensor(list(TARGET)))
    assert moved.field is scene.field
    assert moved.bottom is scene.bottom
    assert moved.bottom_loss is scene.bottom_loss
    assert moved.step_size == scene.step_size
    assert torch.equal(moved.source_position(), torch.tensor(list(TARGET)))


def test_echo_result_exposes_both_legs():
    scene = _scene()
    target = PointTarget(TARGET, 0.0, learnable=False)
    d = (math.dist(SOURCE, TARGET) + math.dist(TARGET, ARRAY)) / C
    grid = make_time_grid(d - 0.004, d + 0.004, 1200)
    result = _echo(scene, target, grid)

    assert result.inbound.shape[0] == 1
    assert result.outbound.shape[0] == scene.receivers.shape[0]
    assert result.inbound.max() > 0 and result.outbound.max() > 0
    # Each leg peaks at its own one-way time.
    t_in = result.leg_time_grid[result.inbound[0, 0].argmax()].item()
    assert t_in == pytest.approx(math.dist(SOURCE, TARGET) / C, abs=5e-5)


def test_the_echo_from_a_known_scatterer_is_the_sonar_equation_s():
    """Both legs, in absolute terms, against a closed form with no free parameter.

    A unit point scatterer in a free field with source and receiver together
    returns ``sigma / R^4`` in energy: two-way spherical spreading and nothing
    else.  Every factor is known, so this is the calibration and not a
    regression baseline.

    It is here because nothing pinned it for a long time.  ``extract_arrivals``
    sums ``exp(-d^2 / 2 sigma_d^2)`` over every ray that passes within its
    acceptance and never divides by the sum of those weights, and on a lattice
    that sum is about ``2 pi`` -- so the splat reports roughly eight decibels
    per leg more energy than the path carries.  Measured here: 17.10 dB with
    the splat on both legs, and it was invisible because it is invariant to how
    the fan was sampled, so no convergence test could see it.  A target
    seventeen decibels too bright reads as a confident detection when it is
    none, which is exactly what it did.
    """
    import warnings as _w

    from hydropt.absorption import thorp_db_per_km
    from hydropt.launch import fibonacci_cone
    from hydropt.targets import ExtendedTarget, IsotropicScattering

    elements = torch.stack([torch.zeros(16), (torch.arange(16.0) - 7.5) * 0.02,
                            torch.full((16,), 50.0)], dim=-1)
    scene = Scene(field=IsoProfile(1500.0, learnable=False),
                  bottom=FlatHeight(1e5), surface=FlatHeight(-1e5),
                  source=(0.0, 0.0, 50.0), receivers=elements,
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  freqs_khz=torch.tensor([10.0]),
                  step_size=2.0, n_steps=400, max_bounces=0)
    tx = fibonacci_cone(4000, torch.tensor([1.0, 0.0, 0.0]), 6.0)
    alpha = float(thorp_db_per_km(torch.tensor([10.0])))

    for rng in (100.0, 250.0):
        target = ExtendedTarget(torch.tensor([[0.0, 0.0, 0.0]]),
                                [IsotropicScattering(0.0, learnable=False)],
                                position=(rng, 0.0, 50.0), learnable=False)
        with _w.catch_warnings(), torch.no_grad():
            _w.simplefilter("ignore")
            echo = target_arrivals(scene, target, tx, return_leg="eigenray",
                                   n_rx_rays=1500, rx_half_angle_deg=20.0,
                                   max_arrivals_per_leg=400)
        # sigma = 1 m^2 (TS = 0 dB), two-way spreading, two-way Thorp over 2R.
        want = (1.0 / rng ** 2) ** 2 * 10.0 ** (-alpha * 2 * rng / 10000.0)
        assert float((echo.amplitude ** 2).sum()) == pytest.approx(want, rel=1e-3)


def test_solving_one_leg_and_splatting_the_other_is_not_half_right():
    """Why both legs had to change together.

    The splat's over-read is per leg, so a mixed pair leaves half of it -- and
    half an error is the harder one to notice, because the number moves in the
    right direction.  This pins that both legs are solved whenever the return
    leg is, rather than leaving the inbound one to a splat that cannot be
    normalised without grouping its rays by path.
    """
    import warnings as _w

    from hydropt.launch import fibonacci_cone
    from hydropt.targets import ExtendedTarget, IsotropicScattering

    elements = torch.stack([torch.zeros(16), (torch.arange(16.0) - 7.5) * 0.02,
                            torch.full((16,), 50.0)], dim=-1)
    scene = Scene(field=IsoProfile(1500.0, learnable=False),
                  bottom=FlatHeight(1e5), surface=FlatHeight(-1e5),
                  source=(0.0, 0.0, 50.0), receivers=elements,
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  freqs_khz=torch.tensor([10.0]),
                  step_size=2.0, n_steps=400, max_bounces=0)
    target = ExtendedTarget(torch.tensor([[0.0, 0.0, 0.0]]),
                            [IsotropicScattering(0.0, learnable=False)],
                            position=(250.0, 0.0, 50.0), learnable=False)
    tx = fibonacci_cone(4000, torch.tensor([1.0, 0.0, 0.0]), 6.0)

    got = {}
    for leg in ("splat", "eigenray"):
        with _w.catch_warnings(), torch.no_grad():
            _w.simplefilter("ignore")
            echo = target_arrivals(scene, target, tx, return_leg=leg,
                                   n_rx_rays=1500, rx_half_angle_deg=20.0,
                                   max_arrivals_per_leg=400)
        got[leg] = float((echo.amplitude ** 2).sum())
    # Two legs of about 2 pi each, in energy: roughly 17 dB, not 8.5.
    excess_db = 10.0 * math.log10(got["splat"] / got["eigenray"])
    assert excess_db > 14.0, (f"the splat is only {excess_db:.2f} dB over the "
                              "solved pair -- one leg is not being solved")


def test_the_splat_over_reads_an_image_by_twice_what_it_over_reads_energy():
    """The check that ``sum |a|^2`` does not make, and why it matters.

    A splat returns N replicas of ONE path.  Summed in energy they add as
    ``sum(w^2)``; the beamformer sums them COHERENTLY, as ``(sum w)^2``.  For
    Gaussian acceptance weights on a lattice ``sum(w) = 2 sum(w^2)``, so an
    image over-reads by about twice the decibels an energy check sees.

    Measured: +17.10 dB of energy and +31.10 dB of image, from 14,641 arrivals
    standing in for one path.  The energy figure was the one quoted while the
    images were being judged, and it understated the error in the pictures by
    fourteen decibels -- enough that a target which is not a detection reads as
    a confident one.
    """
    import warnings as _w

    from hydropt import azimuth_steering, beamform, shading_window
    from hydropt.launch import fibonacci_cone
    from hydropt.targets import ExtendedTarget, IsotropicScattering

    elements = torch.stack([torch.zeros(16), (torch.arange(16.0) - 7.5) * 0.02,
                            torch.full((16,), 50.0)], dim=-1)
    scene = Scene(field=IsoProfile(1500.0, learnable=False),
                  bottom=FlatHeight(1e5), surface=FlatHeight(-1e5),
                  source=(0.0, 0.0, 50.0), receivers=elements,
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  freqs_khz=torch.tensor([10.0]),
                  step_size=2.0, n_steps=400, max_bounces=0)
    rng = 250.0
    target = ExtendedTarget(torch.tensor([[0.0, 0.0, 0.0]]),
                            [IsotropicScattering(0.0, learnable=False)],
                            position=(rng, 0.0, 50.0), learnable=False)
    tx = fibonacci_cone(4000, torch.tensor([1.0, 0.0, 0.0]), 6.0)
    steer, _ = azimuth_steering(21, 15.0)
    grid = make_time_grid(2 * (rng - 8) / 1500.0, 2 * (rng + 8) / 1500.0, 160)

    energy, image = {}, {}
    for leg in ("splat", "eigenray"):
        with _w.catch_warnings(), torch.no_grad():
            _w.simplefilter("ignore")
            echo = target_arrivals(scene, target, tx, return_leg=leg,
                                   n_rx_rays=1500, rx_half_angle_deg=20.0,
                                   max_arrivals_per_leg=400)
            img = beamform(echo, elements, scene.freqs_khz, grid, steer,
                           sigma_t=1e-4, shading=shading_window(16, "uniform"),
                           steer_chunk=8)
        energy[leg] = float((echo.amplitude ** 2).sum())
        image[leg] = float(img.max())

    energy_db = 10.0 * math.log10(energy["splat"] / energy["eigenray"])
    image_db = 10.0 * math.log10(image["splat"] / image["eigenray"])
    assert image_db > energy_db + 10.0, (
        f"image {image_db:.2f} dB against energy {energy_db:.2f} dB -- the "
        "coherent penalty has gone, so one of the two sums has changed")


def test_a_receive_pattern_weights_each_solved_return_path_by_its_arrival():
    """What a stave's height does, pinned on the one path a free field has.

    With source and receiver together and no boundaries there is exactly one
    outbound path, arriving along the line from target to array.  A receive
    pattern returning 0.25 for every direction must quarter the echo energy,
    and one returning 1 must leave it alone -- the weight is a power weight
    applied once, on the return leg only.
    """
    import warnings as _w

    from hydropt.launch import fibonacci_cone
    from hydropt.targets import ExtendedTarget, IsotropicScattering

    elements = torch.stack([torch.zeros(16), (torch.arange(16.0) - 7.5) * 0.02,
                            torch.full((16,), 50.0)], dim=-1)
    scene = Scene(field=IsoProfile(1500.0, learnable=False),
                  bottom=FlatHeight(1e5), surface=FlatHeight(-1e5),
                  source=(0.0, 0.0, 50.0), receivers=elements,
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  freqs_khz=torch.tensor([10.0]),
                  step_size=2.0, n_steps=400, max_bounces=0)
    target = ExtendedTarget(torch.tensor([[0.0, 0.0, 0.0]]),
                            [IsotropicScattering(0.0, learnable=False)],
                            position=(250.0, 0.0, 50.0), learnable=False)
    tx = fibonacci_cone(4000, torch.tensor([1.0, 0.0, 0.0]), 6.0)

    got = {}
    for name, pat in (("none", None), ("unity", lambda d: torch.ones(d.shape[0])),
                      ("quarter", lambda d: torch.full((d.shape[0],), 0.25))):
        with _w.catch_warnings(), torch.no_grad():
            _w.simplefilter("ignore")
            echo = target_arrivals(scene, target, tx, return_leg="eigenray",
                                   n_rx_rays=1500, rx_half_angle_deg=20.0,
                                   max_arrivals_per_leg=400, rx_pattern=pat)
        got[name] = float((echo.amplitude ** 2).sum())
    assert got["unity"] == pytest.approx(got["none"], rel=1e-9)
    assert got["quarter"] == pytest.approx(0.25 * got["none"], rel=1e-9)


@pytest.mark.parametrize("solver", ["trace", "images"])
def test_reciprocal_return_leg_is_the_solved_one(solver):
    """A monostatic sonar's return paths are its outbound paths reversed.

    Solving the return leg separately can only reproduce the inbound solve
    to within its tolerance, so the default reuses it.  Pinned against the
    two-solve answer in a scene with a bottom bounce, so the reversal has a
    reflected path to get right: same count, same energies, and the return
    leg's arrival direction is the inbound launch direction reversed.
    """
    import warnings as _w

    from hydropt.launch import fibonacci_cone
    from hydropt.targets import ExtendedTarget, IsotropicScattering

    elements = torch.stack([torch.zeros(16), (torch.arange(16.0) - 7.5) * 0.02,
                            torch.full((16,), 50.0)], dim=-1)
    scene = Scene(field=IsoProfile(1500.0, learnable=False),
                  bottom=FlatHeight(120.0), surface=FlatHeight(-1e5),
                  source=(0.0, 0.0, 50.0), receivers=elements,
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  freqs_khz=torch.tensor([10.0]),
                  step_size=2.0, n_steps=400, max_bounces=2)
    tx = fibonacci_cone(4000, torch.tensor([1.0, 0.0, 0.0]), 30.0)
    target = ExtendedTarget(torch.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 1.0]]),
                            [IsotropicScattering(0.0, learnable=False)] * 2,
                            position=(200.0, 0.0, 60.0), learnable=False)
    got = {}
    with _w.catch_warnings(), torch.no_grad():
        _w.simplefilter("ignore")
        for name, flag in (("reciprocal", None), ("two solves", False)):
            got[name] = target_arrivals(scene, target, tx, return_leg="eigenray",
                                        n_rx_rays=1500, rx_half_angle_deg=40.0,
                                        max_arrivals_per_leg=400, reciprocal=flag,
                                        eigenray_method=solver)
    a, b = got["reciprocal"], got["two solves"]
    assert a.n_arrivals == b.n_arrivals >= 4      # direct+bounce, squared, per highlight

    # The two mixed pairs (direct in, bounce out; bounce in, direct out) land
    # at the same time -- exactly so under reciprocity, a hair apart from two
    # solves -- so a sort by time puts them either way round.  Order by the
    # arrival direction as well before comparing.
    def ordered(e):
        key = e.time.double() * 1e3 + e.direction[:, 2].double() * 1e-3
        return type(e)(*(None if f is None else f[key.argsort()] for f in e))
    a, b = ordered(a), ordered(b)
    assert torch.allclose(a.time, b.time, rtol=0, atol=1e-6)
    assert torch.allclose(a.amplitude, b.amplitude, rtol=1e-2, atol=0)   # 0.04 dB
    assert torch.allclose(a.direction, b.direction, atol=2e-3)
    assert torch.allclose(a.launch_direction, b.launch_direction, atol=2e-3)

    # And it refuses when the receiver is somewhere else.
    bistatic = Scene(field=IsoProfile(1500.0, learnable=False),
                     bottom=FlatHeight(120.0), surface=FlatHeight(-1e5),
                     source=(0.0, 0.0, 50.0), receivers=elements + torch.tensor([0.0, 20.0, 0.0]),
                     bottom_loss=ConstantLoss(0.0, learnable=False),
                     surface_loss=ConstantLoss(0.0, learnable=False),
                     freqs_khz=torch.tensor([10.0]),
                     step_size=2.0, n_steps=400, max_bounces=2)
    with pytest.raises(ValueError), torch.no_grad():
        target_arrivals(bistatic, target, tx, return_leg="eigenray",
                        n_rx_rays=1500, rx_half_angle_deg=40.0, reciprocal=True,
                        eigenray_method=solver)


def test_a_zero_cross_section_pair_does_not_poison_the_gradient():
    """sqrt'(0) is infinite; a pair the target cannot scatter contributes 0, not NaN.

    A hull patch whose facets all face away from a steep bounce path has a
    bistatic cross-section of exactly zero.  compose_arrivals takes the square
    root of every pair's cross-section, and at zero its derivative is
    infinite, which 0 * inf turns into NaN for every parameter the image
    reaches.  Seen on example 15's boat once the image solver supplied the
    steep paths.
    """
    from hydropt.active import compose_arrivals
    from hydropt.beamform import ArrivalSet
    from hydropt.targets import ExtendedTarget, ScatteringPattern

    class HalfBlind(ScatteringPattern):
        """Scatters only what arrives from above, with a learnable level."""

        def __init__(self):
            super().__init__()
            self.level = torch.nn.Parameter(torch.tensor(1.0))

        def cross_section(self, ki, ks, freqs_khz):
            shape = torch.broadcast_shapes(ki.shape[:-1], ks.shape[:-1])
            from_above = (ki[..., 2] > 0).to(ki.dtype)     # z is depth-down
            return (self.level * from_above).expand(shape).unsqueeze(-1).expand(
                *shape, int(freqs_khz.shape[0]))

    pattern = HalfBlind()
    target = ExtendedTarget(torch.zeros(1, 3), [pattern], position=(50.0, 0.0, 10.0),
                            learnable=True)
    down = torch.tensor([[0.8, 0.0, 0.6], [0.8, 0.0, -0.6]])       # one from above, one from below
    leg = ArrivalSet(time=torch.tensor([0.05, 0.06]), amplitude=torch.ones(2, 1),
                     direction=down, phase=torch.zeros(2), distance=torch.zeros(2),
                     path_length=torch.full((2,), 75.0), launch_direction=-down)
    echo = compose_arrivals(leg, leg, target, highlight=0, freqs_khz=torch.tensor([10.0]))
    (echo.amplitude ** 2).sum().backward()
    assert pattern.level.grad is not None
    assert torch.isfinite(pattern.level.grad).all()
    assert float(pattern.level.grad) > 0


def test_a_fish_school_is_n_fish_in_an_ellipsoid():
    from hydropt.targets import fish_school
    school = fish_school(200, (100.0, 20.0, 15.0), radii=(15.0, 8.0, 3.0),
                         target_strength_db=-40.0, yaw=30.0, learnable=False,
                         generator=torch.Generator().manual_seed(3))
    assert school.n_highlights == 200
    body = school.highlights
    inside = ((body / torch.tensor([15.0, 8.0, 3.0])) ** 2).sum(-1)
    assert float(inside.max()) <= 1.0 + 1e-9
    assert float(inside.mean()) > 0.3                     # filled, not hollow or clumped
    # every fish is one -40 dB scatterer, whichever way it is looked at
    ki = torch.tensor([[[1.0, 0.0, 0.0]]])
    sigma = school.cross_section(7, ki, -ki, torch.tensor([120.0]))
    assert float(10 * torch.log10(sigma.reshape(-1)[0])) == pytest.approx(-40.0, abs=1e-6)
    # the school's centre is where it was put, and its world layout is rotated
    world = school.world_positions()
    assert torch.allclose(world.mean(0), torch.tensor([100.0, 20.0, 15.0]), atol=1.5)
