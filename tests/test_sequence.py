"""Sequences: a trajectory's interpolation and a renderer's frames.

The trajectory is pinned to closed-form geometry (a constant-speed track,
headings along it, a turn through north interpolated the short way), and the
renderer to the linearity it relies on: the picture of a target rendered
through it equals the picture assembled by hand from the same background and
echo beams, the background is formed once, and a moved target's echo moves
in time by what the geometry says.
"""

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, IsotropicScattering, LambertScattering,
    PictureRenderer, Scene, Trajectory, ExtendedTarget, azimuth_steering,
    beamform, box_mesh, calibrate, emission_arrivals, make_time_grid,
    horizontal_line_array, propeller_directivity, reframe_height_field, relative_pose,
    BilinearHeightField,
)
from hydropt.launch import fibonacci_cone

C = 1500.0


def test_trajectory_interpolates_and_clamps():
    tr = Trajectory([0.0, 10.0, 20.0], [[0, 0], [100, 0], [100, 50]], [0.0, 0.0, 90.0])
    assert tr.duration == pytest.approx(20.0)
    x, y, h = tr.at(5.0)
    assert (x, y, h) == pytest.approx((50.0, 0.0, 0.0))
    x, y, h = tr.at(15.0)
    assert (x, y, h) == pytest.approx((100.0, 25.0, 45.0))
    assert tr.at(-3.0) == pytest.approx((0.0, 0.0, 0.0))
    assert tr.at(99.0) == pytest.approx((100.0, 50.0, 90.0))
    assert len(tr.poses([0.0, 1.0, 2.0])) == 3


def test_trajectory_unwraps_headings_through_north():
    tr = Trajectory([0.0, 1.0], [[0, 0], [1, 0]], [350.0, 10.0])
    _, _, h = tr.at(0.5)
    assert h % 360.0 == pytest.approx(0.0, abs=1e-5)     # 20 deg the short way, not 340
    tr = Trajectory([0.0, 1.0], [[0, 0], [1, 0]], [-170.0, 170.0])
    _, _, h = tr.at(0.5)
    assert h % 360.0 == pytest.approx(180.0, abs=1e-5)


def test_trajectory_from_waypoints_has_headings_along_the_track():
    tr = Trajectory.from_waypoints([[0, 0], [30, 0], [30, 40]], speed=5.0, start_time=2.0)
    assert tr.times.tolist() == pytest.approx([2.0, 8.0, 16.0])
    assert tr.headings_deg.tolist() == pytest.approx([0.0, 0.0, 90.0])
    x, y, h = tr.at(12.0)
    assert (x, y) == pytest.approx((30.0, 20.0))
    with pytest.raises(ValueError):
        Trajectory.from_waypoints([[0, 0]], speed=1.0)
    with pytest.raises(ValueError):
        Trajectory([0.0, 0.0], [[0, 0], [1, 0]], [0.0, 0.0])


def _renderer(**kw):
    rx = horizontal_line_array(0.5, -0.05, 0.5, 0.05, 10.0, 8)
    scene = Scene(
        field=IsoProfile(C), bottom=FlatHeight(30.0), surface=FlatHeight(0.0),
        source=(0.0, 0.0, 10.0), receivers=rx,
        surface_loss=ConstantLoss(0.3, learnable=False),
        bottom_loss=ConstantLoss(4.0, learnable=False),
        freqs_khz=torch.tensor([50.0]), step_size=0.5, n_steps=300, max_bounces=2)
    dirs = fibonacci_cone(400, torch.tensor([1.0, 0.0, 0.0]), 20.0)
    steer, bearings = azimuth_steering(9, 20.0)
    grid = make_time_grid(2.0 * 20.0 / C, 2.0 * 80.0 / C, 120)
    one = lambda d: torch.ones(d.shape[:-1], dtype=d.dtype)
    args = dict(elements=rx, directions=dirs, tx_weights=one(dirs), tx_pattern=one,
                rx_pattern=one, time_grid=grid, steer=steer, sigma_t=1e-3,
                shading=None, source_level_db=200.0, noise_power=1e-3,
                scattering=LambertScattering(-27.0, learnable=False),
                solid_angle_per_ray=1e-3, max_arrivals=2000,
                target_kwargs=dict(n_rx_rays=300, rx_half_angle_deg=30.0))
    args.update(kw)
    return PictureRenderer(scene, **args), grid, bearings


def _target(x, y, h):
    return ExtendedTarget([(0.0, 0.0, 0.0)], [IsotropicScattering(-5.0)],
                          position=(x, y, 12.0), yaw=h, learnable=False)


def test_renderer_picture_is_background_plus_echo_and_background_is_cached():
    r, grid, _ = _renderer()
    assert r._background is None
    pic = r.picture([_target(50.0, 0.0, 0.0)], frame=3)
    assert r.n_reverberation > 0
    b_rev = r.background()
    assert b_rev is r.background()                      # cached, not re-traced
    # by hand: the same complex beams, calibrated, the same noise draw
    field = calibrate(b_rev + r.beams(r.echo(_target(50.0, 0.0, 0.0))),
                      200.0, beam_scale=r.beam_scale)
    from hydropt import add_receiver_noise
    noisy = add_receiver_noise(field, 1e-3, generator=torch.Generator().manual_seed(r.seed + 5))
    assert torch.allclose(pic, noisy, rtol=1e-5, atol=1e-6 * float(noisy.detach().abs().max()))
    assert pic.shape == (9, 1, 120)


def test_renderer_sequence_moves_the_echo_along_the_track():
    r, grid, bearings = _renderer(noise_power=0.0)
    tr = Trajectory.from_waypoints([[40.0, 0.0], [70.0, 0.0]], speed=10.0)
    frames = list(r.sequence(_target, tr, [0.0, 3.0]))
    assert [t for t, _, _ in frames] == [0.0, 3.0]
    assert frames[0][1] == pytest.approx((40.0, 0.0, 0.0))
    assert frames[1][1] == pytest.approx((70.0, 0.0, 0.0))
    back = calibrate(r.background(), 200.0, beam_scale=r.beam_scale).abs() ** 2
    mid = bearings.abs().argmin()
    for (_, (x, _, _), pic) in frames:
        excess = (pic - back)[mid, 0]
        # the echo's onset is the direct path both ways; the bounce paths
        # follow it (and with 400 rays one of them may sample better)
        onset = int((excess > 0.1 * excess.max()).nonzero()[0])
        expect = (math.dist((0, 0, 10), (x, 0, 12)) + math.dist((x, 0, 12), (0.5, 0, 10))) / C
        assert float(grid[onset]) == pytest.approx(expect, abs=2.0 * float(grid[1] - grid[0]))


def test_renderer_incoherent_picture_adds_the_echo_in_power():
    r, grid, _ = _renderer(noise_power=0.0)
    t = _target(50.0, 0.0, 0.0)
    inc = r.picture([t], coherent=False)
    back = calibrate(r.background(), 200.0, beam_scale=r.beam_scale).abs() ** 2
    power = beamform(r.echo(t), r.elements, r.scene.freqs_khz, grid, r.steer,
                     sigma_t=r.sigma_t, shading=None, coherent=False, checkpoint=False)
    assert torch.allclose(inc, back + calibrate(power, 200.0, beam_scale=r.beam_scale),
                          rtol=1e-5, atol=1e-6 * float(inc.detach().max()))


def test_renderer_applies_display_and_cartesian_callables():
    r, grid, _ = _renderer(display=lambda img: 10.0 * torch.log10(img + 1e-30),
                           to_cartesian=lambda shown: (shown[:, 0], "gx", "gy"))
    cart, gx, gy = r.picture([_target(50.0, 0.0, 0.0)])
    assert cart.shape == (9, 120) and gx == "gx"


def test_emission_is_a_spoke_at_its_level_and_the_hull_can_shadow_it():
    r, grid, bearings = _renderer(noise_power=0.0)
    centre = r.elements.mean(dim=0)
    emitter = torch.tensor([60.0, 0.0, 12.0])
    arr, n_clear, n_paths = emission_arrivals(
        r.scene, emitter, centre, grid, spectrum_level_db=120.0, source_level_db=200.0,
        pulse_s=r.sigma_t, generator=torch.Generator().manual_seed(0))
    assert n_clear == n_paths > 1
    pic = r.picture([], extra_arrivals=[arr])
    b_emit = r.beams(arr)
    # added to the FIELD: the picture is |b_rev + b_emit|^2, not a sum of powers
    both = calibrate(r.background() + b_emit, 200.0, beam_scale=r.beam_scale).abs() ** 2
    assert torch.allclose(pic, both, rtol=1e-5, atol=1e-6 * float(both.max()))
    emit = calibrate(b_emit, 200.0, beam_scale=r.beam_scale).abs() ** 2
    mid = bearings.abs().argmin()
    # every range cell of the emitter's bearing carries it, well above the
    # bearings either side (a spoke), and its level is the emission's band
    # level less the direct path's spreading, to within the multipath and
    # the random phases of the train
    on = emit[mid, 0, 10:-10]
    off = emit[[0, -1], 0, 10:-10].mean()
    assert float(on.min()) > 0.0
    assert float(on.mean()) > 10.0 * float(off)
    d = float((emitter - centre).norm())
    expect_db = 120.0 + 10.0 * math.log10(1.0 / r.sigma_t) - 20.0 * math.log10(d)
    got_db = 10.0 * math.log10(float(on.mean()))
    assert abs(got_db - expect_db) < 4.0
    # a box around the emitter shadows every path
    verts, faces = box_mesh((4.0, 4.0, 4.0))
    shadowed, n_clear, _ = emission_arrivals(
        r.scene, emitter, centre, grid, spectrum_level_db=120.0, source_level_db=200.0,
        pulse_s=r.sigma_t, occluder=(verts + emitter, faces))
    assert shadowed is None and n_clear == 0
    # and picture() skips a None
    back = calibrate(r.background(), 200.0, beam_scale=r.beam_scale).abs() ** 2
    assert torch.allclose(r.picture([], extra_arrivals=[shadowed]), back)


def test_sequence_emitters_follow_the_pose():
    r, grid, bearings = _renderer(noise_power=0.0)
    seen = []

    def emitter(x, y, h, frame):
        seen.append((x, y, h, frame))
        return None

    tr = Trajectory.from_waypoints([[40.0, 0.0], [70.0, 0.0]], speed=10.0)
    list(r.sequence(_target, tr, [0.0, 3.0], emitters=[emitter]))
    assert seen == [pytest.approx((40.0, 0.0, 0.0, 0)), pytest.approx((70.0, 0.0, 0.0, 1))]


def test_propeller_directivity_is_loud_astern_and_quiet_ahead():
    pat = propeller_directivity(0.0, bow_db=20.0, wake_db=6.0, wake_half_deg=15.0)
    d = torch.tensor([[-1.0, 0.0, 0.0],      # straight astern: the wake's notch
                      [-1.0, -1.0, 0.0],     # the starboard quarter
                      [0.0, 1.0, 0.0],       # abeam
                      [1.0, 0.0, 0.0],       # ahead, through the hull
                      [-0.5, 0.0, 0.8]])     # astern, steeply up: horizontal aspect only
    db = 20.0 * torch.log10(pat(d))
    assert db[0].item() == pytest.approx(-6.0, abs=1e-4)
    assert db[1].item() == pytest.approx(-20.0 * (1 - math.cos(math.radians(45))) / 2, abs=0.05)
    assert db[2].item() == pytest.approx(-10.0, abs=1e-4)
    assert db[3].item() == pytest.approx(-20.0, abs=1e-4)
    assert db[4].item() == pytest.approx(-6.0, abs=1e-4)
    # rotated with the heading: a boat heading +90 has its stern at -y
    pat90 = propeller_directivity(90.0, wake_db=0.0)
    assert float(pat90(torch.tensor([[0.0, -1.0, 0.0]]))) == pytest.approx(1.0)
    assert 20.0 * math.log10(float(pat90(torch.tensor([[0.0, 1.0, 0.0]])))) == pytest.approx(-20.0, abs=1e-4)


def test_emission_pattern_scales_the_paths():
    r, grid, _ = _renderer(noise_power=0.0)
    centre = r.elements.mean(dim=0)
    emitter = torch.tensor([60.0, 0.0, 12.0])
    kw = dict(spectrum_level_db=120.0, source_level_db=200.0, pulse_s=r.sigma_t)
    omni, _, _ = emission_arrivals(r.scene, emitter, centre, grid, **kw,
                                   generator=torch.Generator().manual_seed(0))
    # heading +x: the array is astern of the emitter, the paths leave near the wake's notch
    away, _, _ = emission_arrivals(r.scene, emitter, centre, grid, **kw,
                                   pattern=propeller_directivity(0.0, wake_db=0.0),
                                   generator=torch.Generator().manual_seed(0))
    # heading -x: the array is ahead, every path is 20 dB down (less its climb)
    toward, _, _ = emission_arrivals(r.scene, emitter, centre, grid, **kw,
                                     pattern=propeller_directivity(180.0, wake_db=0.0),
                                     generator=torch.Generator().manual_seed(0))
    assert torch.allclose(away.amplitude, omni.amplitude, rtol=1e-3)
    ratio = 20.0 * torch.log10(toward.amplitude / omni.amplitude)
    assert float(ratio.max()) < -19.0 and float(ratio.min()) > -20.01


def test_relative_pose_turns_the_world_into_the_ownship_frame():
    # ownship at (10, 5) heading 90 (looking along +y): a thing 20 m along +y
    # is 20 m ahead, a thing at +x is to starboard (-y)
    assert relative_pose((10.0, 25.0, 90.0), (10.0, 5.0, 90.0)) == pytest.approx((20.0, 0.0, 0.0))
    assert relative_pose((30.0, 5.0, 0.0), (10.0, 5.0, 90.0)) == pytest.approx((0.0, -20.0, -90.0))
    assert relative_pose((3.0, 4.0, 30.0), (0.0, 0.0, 0.0)) == pytest.approx((3.0, 4.0, 30.0))


def test_reframe_height_field_reads_the_world_at_the_pose():
    # a world plane z = 30 + 0.1 x + 0.02 y on a wide grid
    xs = torch.arange(-200.0, 201.0, 10.0)
    Y, X = torch.meshgrid(xs, xs, indexing="ij")
    world = BilinearHeightField(30.0 + 0.1 * X + 0.02 * Y, origin=(-200.0, -200.0),
                                spacing=(10.0, 10.0), learnable=False)
    local = reframe_height_field(world, shape=(5, 7), spacing=(4.0, 4.0), origin=(-8.0, -8.0),
                                 x=50.0, y=-20.0, heading_deg=90.0)
    assert not any(p.requires_grad for p in local.parameters())
    assert local.shape == (5, 7)
    # the sonar-frame node (8, 4) heading 90 is the world point (50 - 4, -20 + 8)
    expect = 30.0 + 0.1 * (50.0 - 4.0) + 0.02 * (-20.0 + 8.0)
    assert float(local.height(torch.tensor([[8.0, 4.0]]))) == pytest.approx(expect, abs=1e-4)
    assert float(local.height(torch.tensor([[0.0, 0.0]]))) == pytest.approx(30.0 + 5.0 - 0.4, abs=1e-4)


def test_ownship_sequence_rebuilds_the_scene_and_places_targets_relative():
    r, grid, bearings = _renderer(noise_power=0.0)
    built, scenes = [], []

    def builder(x, y, h):
        built.append((x, y, h))
        return _target(x, y, h)

    def scene_at(x, y, h):
        scenes.append((x, y, h))
        return r.scene

    tr = Trajectory([0.0, 10.0], [[0.0, 0.0], [20.0, 0.0]], [0.0, 0.0])
    world = [((60.0, 0.0, 0.0), builder)]
    frames = list(r.ownship_sequence(world, tr, [0.0, 10.0], scene_at=scene_at))
    assert scenes == [pytest.approx((0.0, 0.0, 0.0)), pytest.approx((20.0, 0.0, 0.0))]
    assert built == [pytest.approx((60.0, 0.0, 0.0)), pytest.approx((40.0, 0.0, 0.0))]
    # the background was formed afresh for the second pose (the scene was reset)
    assert r._background is not None
    back = calibrate(r.background(), 200.0, beam_scale=r.beam_scale).abs() ** 2
    mid = bearings.abs().argmin()
    # the echo of the second frame is at 40 m two-way, of the first at 60 m
    for (t, pose, pic), x in zip(frames, (60.0, 40.0)):
        excess = (pic - back)[mid, 0]
        onset = int((excess > 0.1 * excess.max()).nonzero()[0])
        expect = (math.dist((0, 0, 10), (x, 0, 12)) + math.dist((x, 0, 12), (0.5, 0, 10))) / C
        assert float(grid[onset]) == pytest.approx(expect, abs=2.0 * float(grid[1] - grid[0]))


def test_ownship_sequence_takes_a_kept_background():
    r, grid, _ = _renderer(noise_power=0.0)
    traces = []
    kept = {}

    def scene_at(x, y, h):
        if x in kept:
            return r.scene, kept[x]
        traces.append(x)
        return r.scene

    tr = Trajectory([0.0, 10.0], [[0.0, 0.0], [20.0, 0.0]], [0.0, 0.0])
    world = [((60.0, 0.0, 0.0), _target)]
    first = []
    for t, pose, pic in r.ownship_sequence(world, tr, [0.0, 10.0], scene_at=scene_at):
        kept[pose[0]] = r.background()
        first.append(pic)
    again = [pic for _, _, pic in r.ownship_sequence(world, tr, [0.0, 10.0], scene_at=scene_at)]
    assert traces == [0.0, 20.0]                       # the second pass traced nothing
    for a, b in zip(first, again):
        assert torch.equal(a, b)
