"""What a sonar image has to look like, at a size a test can afford.

The examples already assert these things, and take minutes to do it, so the
checks that matter most only ever ran by hand.  This is the same scene shrunk
until it runs in seconds: an AUV at 8 m in 20 m of water, a 120 kHz fan tilted
up, a rough seabed and a wind sea out to 85 m, and a 12 m boat at 65 m viewed
40 degrees off its own heading -- which is 58 degrees off the line of sight,
the aspect a vessel is usually seen at rather than the one that flatters it.

**The pair is the point.**  Every claim here is made about two images that
differ in one thing, so a criterion that cannot fail shows up immediately as
one that passes both.  Three criteria written for ``examples/21`` passed on a
boat that was invisible in the picture, each because its comparison set
flattered the target, and each would have been caught in six seconds by
running it against a case whose answer was already known.

**What this cannot do.**  1,590 background cells against the full-size scene's
43,629, so the clutter's extremes are weaker here and detection is
correspondingly easier; and the fan is dense enough for the background to be
speckle-like but not converged -- ``std/mean`` comes out near 1.95 where fully
developed one-look speckle is 1.00.  So this catches regressions and pins
directions.  Whether a target is *really* detectable is still the full-size
run's answer.
"""

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, IsoProfile, Scene, azimuth_steering, beamform,
    fractal_bathymetry, line_array_factor, make_time_grid,
    pierson_moskowitz_surface, sediment_loss, shading_window, target_arrivals,
)
from hydropt.beamform import ArrivalSet
from hydropt.mesh import boat_hull_mesh, mesh_target
from hydropt.reverb import LambertScattering, reverberation_arrivals
from hydropt.tracer import trace

C, FREQ_KHZ = 1500.0, 120.0
LAMBDA = C / (FREQ_KHZ * 1e3)
AUV_DEPTH, WATER_DEPTH = 8.0, 20.0
NEAR, FAR, SECTOR_DEG = 25.0, 85.0, 30.0
N_RX, N_TX = 32, 5
ELEV_DEG, TILT_DEG = (-30.0, 14.0), -5.0
PULSE_S, RANGE_CELL_M, N_BEAMS = 3e-4, 0.8, 31
N_ELEV, N_AZIM = 40, 256
BOAT_RANGE, BOAT_BEARING_DEG, BOAT_HEADING_DEG = 65.0, -10.0, 40.0
HULL_LENGTH, HULL_BEAM, HULL_DRAUGHT = 12.0, 3.2, 1.6
# Set by matching the 10 to 15 dB beam-to-off-beam swing measured on real
# vessels (Urick; DTIC AD0039542, AD0531451): on this hull that is mu between
# -15 and -10 dB.  The sand seabed's -27 dB, used earlier as a stand-in, gives
# a 26 dB swing and a hull twice as dim off beam aspect as any vessel measured.
# A mirror-smooth hull (43.7 dB of swing) is the comparison, not the model.
DIFFUSE_DB = -12.0


def _array():
    y = (torch.arange(N_RX, dtype=torch.get_default_dtype()) - (N_RX - 1) / 2) * LAMBDA / 2
    return torch.stack((torch.zeros_like(y), y, torch.full_like(y, AUV_DEPTH)), dim=-1)


def _scene(elements, seed=3):
    bottom = fractal_bathymetry((20, 20), (10.0, 10.0), base_depth=WATER_DEPTH,
                                rms=0.6, exponent=3.0, origin=(-20.0, -100.0),
                                learnable=True,
                                generator=torch.Generator().manual_seed(seed))
    surface = pierson_moskowitz_surface((40, 40), (5.0, 5.0), 4.0,
                                        origin=(-20.0, -100.0), learnable=True,
                                        generator=torch.Generator().manual_seed(seed + 1))
    # Energy-conserving boundaries: a rough sea randomises the phase of what
    # bounces off it and spreads it over a few degrees, but reflects all of
    # the energy, and reverberation is an energy quantity.  The multipath a
    # continuing ray produces is real shallow-water physics, not double
    # counting.  Eckart belongs on the target's coherent bounce paths only.
    scene = Scene(
        field=IsoProfile(C, learnable=False), bottom=bottom, surface=surface,
        source=(0.0, 0.0, AUV_DEPTH), receivers=elements,
        surface_loss=ConstantLoss(0.0, learnable=False, pressure_release=True),
        bottom_loss=sediment_loss("sand", learnable=True),
        freqs_khz=torch.tensor([FREQ_KHZ]),
        step_size=2.0, n_steps=60, max_bounces=3)
    return scene, bottom, surface


def _hull(diffuse_db):
    b = math.radians(BOAT_BEARING_DEG)
    verts, faces = boat_hull_mesh(HULL_LENGTH, HULL_BEAM, HULL_DRAUGHT,
                                  n_long=40, n_around=14)
    # z=0: boat_hull_mesh returns the WETTED surface, waterline already at z=0.
    return mesh_target(verts, faces,
                       position=(BOAT_RANGE * math.cos(b), BOAT_RANGE * math.sin(b), 0.0),
                       yaw=BOAT_HEADING_DEG, n_patches=3, sound_speed=C,
                       diffuse_db=diffuse_db, learnable=True,
                       learnable_shape=False, facet_chunk=256)


def _fan(seed=0):
    g = torch.Generator().manual_seed(seed)
    e0, e1 = (math.radians(v) for v in ELEV_DEG)
    a0, a1 = -math.radians(SECTOR_DEG), math.radians(SECTOR_DEG)
    E, A = torch.meshgrid(torch.linspace(e0, e1, N_ELEV),
                          torch.linspace(a0, a1, N_AZIM), indexing="ij")
    E, A = E.reshape(-1), A.reshape(-1)
    E = E + (torch.rand(E.shape, generator=g, dtype=E.dtype) - .5) * (e1 - e0) / (N_ELEV - 1)
    A = A + (torch.rand(A.shape, generator=g, dtype=A.dtype) - .5) * (a1 - a0) / (N_AZIM - 1)
    dirs = torch.stack([E.cos() * A.cos(), E.cos() * A.sin(), E.sin()], dim=-1)
    weights = line_array_factor(torch.sin(E), N_TX,
                                sin_steer=math.sin(math.radians(TILT_DEG)))
    return dirs, weights


def _pattern(directions):
    """The projector's directivity as a function of direction, not of ray.

    A solved inbound path has no ray index to look ``tx_weights`` up by.  This
    is the same function, exactly: ``_fan`` builds directions as
    ``[cos E cos A, cos E sin A, sin E]``, so ``z`` IS ``sin E``.
    """
    return line_array_factor(directions[..., 2], N_TX,
                             sin_steer=math.sin(math.radians(TILT_DEG)))


@pytest.fixture(scope="module")
def ping():
    """One reverberation, three target channels.

    The reverberation does not depend on the target, so it is traced once and
    combined with each -- which is what makes three images affordable at all.
    """
    torch.manual_seed(0)
    rx = _array()
    scene, bottom, surface = _scene(rx)
    steer, bearings = azimuth_steering(N_BEAMS, SECTOR_DEG)
    grid = make_time_grid(2 * NEAR / C, 2 * FAR / C, int((FAR - NEAR) / RANGE_CELL_M))
    dirs, weights = _fan()
    solid = (math.radians(2 * SECTOR_DEG)
             * math.radians(ELEV_DEG[1] - ELEV_DEG[0]) / dirs.shape[0])
    rev = reverberation_arrivals(
        trace(scene, dirs), dirs, scene.freqs_khz,
        scattering=LambertScattering(-27.0, learnable=True),
        solid_angle_per_ray=solid, ray_weights=weights, boundary="both",
        surface=scene.surface, bottom=scene.bottom,
        generator=torch.Generator().manual_seed(1))

    def image(target):
        arrivals = rev
        if target is not None:
            echo = target_arrivals(
                scene, target, dirs, return_leg="eigenray", n_rx_rays=400,
                rx_half_angle_deg=45.0, tx_weights=weights,
                tx_pattern=_pattern, max_arrivals_per_leg=16,
                generator=torch.Generator().manual_seed(0))
            if echo.n_arrivals:
                arrivals = ArrivalSet(*(
                    None if rev[i] is None or echo[i] is None
                    else torch.cat([rev[i], echo[i]], dim=0)
                    for i in range(len(rev))))
        return beamform(arrivals, rx, scene.freqs_khz, grid, steer,
                        sigma_t=PULSE_S, shading=shading_window(N_RX, "hann"),
                        steer_chunk=8)[:, 0]

    rough, smooth = _hull(DIFFUSE_DB), _hull(None)
    return dict(bare=image(None), mirror=image(smooth), diffuse=image(rough),
                bearings=torch.as_tensor(bearings), ranges=grid * C / 2,
                boat=rough, bottom=bottom, surface=surface, scene=scene)


def _masks(ping):
    rng = ping["ranges"].reshape(1, -1).expand_as(ping["bare"])
    brg = ping["bearings"].reshape(-1, 1).expand_as(ping["bare"])
    on_boat = ((rng - BOAT_RANGE).abs() < 10.0) & ((brg - BOAT_BEARING_DEG).abs() < 8.0)
    clutter = ~(((rng - BOAT_RANGE).abs() < 20.0) & ((brg - BOAT_BEARING_DEG).abs() < 16.0))
    return on_boat, clutter, rng, brg


def _beaten_by_clutter(image, on_boat, clutter):
    """How many background cells are brighter than the target's peak."""
    with torch.no_grad():
        peak = float(image[on_boat].max())
        return int((image[clutter] > peak).sum()), int(clutter.sum())


def test_a_real_hull_is_detectable_and_a_mirror_one_is_not(ping):
    """The criterion, and the pair that proves it can fail.

    Physical optics on a smooth mesh returns where a facet points back at you
    and cancels everywhere else, so an obliquely-viewed hull has almost no
    echo -- 34 dB below beam aspect for this shape.  Give it the diffuse
    channel a real vessel's plating and structure provide and it comes back.

    Asserting only the first half would pass a criterion that cannot fail,
    which is exactly what happened three times in ``examples/21``.
    """
    on_boat, clutter, _, _ = _masks(ping)
    rough, n = _beaten_by_clutter(ping["diffuse"], on_boat, clutter)
    mirror, _ = _beaten_by_clutter(ping["mirror"], on_boat, clutter)
    assert rough == 0, f"{rough} of {n} clutter cells beat a hull that should be detectable"
    assert mirror > 10, ("a mirror-smooth hull at 58 degrees off should NOT be "
                         f"detectable, and only {mirror} of {n} cells beat it -- "
                         "the test has stopped being able to fail")


def test_with_no_target_at_all_the_clutter_wins(ping):
    """The control: the same measurement on an image with no boat in it.

    Without this, a bug that made ``on_boat`` pick up the brightest cell in the
    picture would pass the detection test forever.
    """
    on_boat, clutter, _, _ = _masks(ping)
    empty, n = _beaten_by_clutter(ping["bare"], on_boat, clutter)
    assert empty > 100, (f"only {empty} of {n} cells beat an empty patch of sea "
                         "-- the detection measurement is not measuring the boat")


def test_the_background_is_speckle_and_not_a_handful_of_scatterers(ping):
    """Reverberation has to be a field, not a scatter of individual returns.

    One-look speckle is exponential, so ``std/mean`` is exactly 1.  A fan too
    sparse for the image gives a background of isolated bright patches instead,
    whose extremes invent false alarms and make any detection claim wrong in
    both directions: at 0.18 patches per cell this fixture measured 2.92.

    This fan reaches about 1.95, which is speckle-like and NOT converged.  The
    bound is set where it catches the sparse regime, not where it certifies
    Rayleigh statistics -- that would need a fan this test cannot afford.
    """
    with torch.no_grad():
        cell = ping["bare"].reshape(-1)
        cell = cell[cell > 0]
        ratio = float(cell.std() / cell.mean())
    assert ratio < 2.5, f"std/mean {ratio:.2f}: the background is not a speckle field"


def test_the_echo_has_the_extent_of_a_body_and_is_not_smeared_across_the_swath(ping):
    """Bounds on the echo's width, and an honest note on what they do not catch.

    The motivation was the splat return leg reporting a 25 m hull as 54 m
    across, because it placed every accepted ray's arrival at the fan's angular
    spacing rather than where the path actually arrived from.  **This fixture
    does not catch that**, and the claim was checked rather than assumed: with
    the return leg swapped to the splat, the echo measures 9.1 m against the
    eigenray's 11.3 m, and both sit far inside the 24.4 m bound below.  At this
    size the hull spans 1.5 beamwidths and the return fan is fine enough that
    the smearing is small, so the defect simply does not show.  Where it does
    show is the full-size scene, which measured 9.60 degrees of echo against
    4.82 of hull before the eigenray leg and 3.94 after.

    What survives is a pair of sanity bounds -- the echo is a body with extent
    rather than a point, and it is not smeared over several beamwidths -- which
    is worth having and is not the same test.
    """
    on_boat, _, rng, brg = _masks(ping)
    with torch.no_grad():
        target_only = (ping["diffuse"] - ping["bare"]).clamp_min(0.0)
        peak = float(target_only[on_boat].max())
        lit = on_boat & (target_only > peak * 0.1)
        across = (torch.deg2rad(brg[lit] - BOAT_BEARING_DEG) * BOAT_RANGE)
        measured = float(across.max() - across.min())
    beam_m = 2.0 * math.degrees(math.asin(2.0 / N_RX)) * math.pi / 180.0 * BOAT_RANGE
    assert measured < 3.0 * beam_m, (f"{measured:.1f} m of echo against a "
                                     f"{beam_m:.1f} m beam -- smeared across the swath")
    assert measured > 0.2 * HULL_LENGTH, f"{measured:.1f} m: the hull has no extent at all"


def test_the_image_carries_gradients_to_the_boat_and_the_sea(ping):
    """Differentiable end to end, which is the point of the whole library."""
    live = {"boat position": ping["boat"].position,
            "boat heading": ping["boat"].orientation,
            "seabed": ping["bottom"].heights,
            "waves": ping["surface"].heights}
    for p in live.values():
        p.grad = None
    ping["diffuse"].sum().backward(retain_graph=True)
    dead = [k for k, p in live.items()
            if p.grad is None or not bool(torch.isfinite(p.grad).all())
            or float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"no gradient reaches {dead}"
