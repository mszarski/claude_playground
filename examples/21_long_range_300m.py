"""A long look: 300 m of seabed, sea surface and one small boat.

Everything the earlier FLS examples do, moved out to where the range equation
starts to bite.  A 100 kHz Mills cross on an AUV at 25 m in 60 m of water,
looking out to 300 m: a rough sand seabed, a light wind sea, and a 12 m boat on
the surface at 250 m.  No wake -- just the scene.

**Three things change when you go from 90 m to 300 m, and none of them is the
picture getting bigger.**

*Absorption stops being a rounding error.*  Seawater takes about 34 dB/km at
100 kHz, so the two-way loss to 300 m is 20 dB on top of spreading.  That is
the reason a 300 m set is usually built at 60 kHz or below, trading 13 dB of
absorption for beams 1.7 times wider.  The budget is printed.

*The near field goes dark, and the vertical FOV is why.*  The head here is the
real thing: 3.00 degree azimuth beams and a 20 degree vertical field of view
carrying beams of 4.84 degrees, about four of them.  Twenty degrees is not much
at 300 m, and it is the quantity that decides whether the surface and the
seabed are both in the picture: they sit 33 degrees apart at 100 m and 11 at
300, so a single tilt holds both only past about 170 m and holds whichever is
nearer its axis inside that.  No vehicle depth fixes it -- the separation
depends only on the total water depth -- so it is geometry rather than a
shortage of power.

*Reverberation was supposed to stop being the enemy, and does not.*  Bottom
reverberation falls as ``r^-5`` with the grazing angle falling too, so past some
range the competition ought to become the ambient sea rather than the seabed --
and that crossover is the number that decides which knob is worth turning, since
more power helps a noise-limited detection and does nothing for a
reverberation-limited one.  Measured here it is at about 360 m, *outside* the
swath: at 300 m the reverberation is still 9 dB above the ambient.  So this
whole 300 m picture is reverberation-limited end to end, and a louder projector
would buy exactly nothing in it.  The example extrapolates its own measured
falloff to say where that stops being true.

The boat is rendered from its triangle mesh as before, its echo is summed with
the reverberation *before* beamforming, and the whole image is put on an
absolute scale in uPa^2 with real ambient noise added at the correct Rice
statistics, so "can you see it at 250 m" has an answer rather than a picture.

**Why this looks worse than the 90 m images, which is not what it seems.**  Set
side by side with ``examples/15``, this picture reads as noisy and the boat
reads as one bright speckle among many.  Three explanations suggest themselves
and two of them are wrong, which is worth recording because they are the
obvious two:

*Not the dynamic range.*  Measured the same way on both scenes, the swath-mean
reverberation spans 22.8 dB over the lit part of this one and 21.0 dB over the
90 m one.  Near enough identical.

*Not the sampling.*  This fan puts 0.99 scattering patches in a resolution
cell; the 90 m example manages 0.35.  This scene is sampled three times better,
and the background's spread comes out at 5.1 dB against the 5.6 dB of textbook
Rayleigh speckle -- so the background is not noisy, it IS speckle, correctly.

*It is the resolution.*  The beam is 3.58 degrees wide either way, which is
3.4 m of cross-range at 55 m and 15.6 m at 250 m.  A 12 m hull is therefore
3.5 beamwidths long in the near example and 0.77 of one here.  Near, the sonar
draws a boat-shaped object; far, it draws a point -- and a point in a speckle
field is shaped exactly like a bright speckle, whatever its contrast.  The
contrasts are in fact similar, about +20 dB near and +17.5 dB here.  The last
section renders the same arrivals through a four-times-longer array to show the
hull coming back as a shape.

**And most of what "noisy" means here is the colour window.**  ``examples/15``
draws 22 dB below its own peak, and its peak IS the boat -- the target is the
brightest cell in that image.  Everything more than 22 dB under the boat is
therefore clipped to black, which is most of its reverberation, and the picture
comes out black with a clean return on it.  This image was drawn 45 dB below
its peak, and its peak is the near-field seabed rather than the target, so the
whole swath of reverberation sits inside the scale and the picture is full of
it.  Same kind of scene, opposite conventions.  The figure shows this one under
both, and the difference is larger than anything the physics does.

**And the gain has a trap in it that cost 4 dB before it was found.**  Taking
the swath's level at each range from the MEAN over beams lets a target suppress
itself: the boat's echo lifts the mean at its own range bin by 6.3 dB, and the
gain then divides that straight back out, turning +17.5 dB of contrast into
+13.5 dB.  It hides from a ring measurement, because spread over a +/-8 m band
the lift is under a decibel.  A MEDIAN over 181 beams cannot be moved by a
handful of bright ones and keeps the contrast in full, which is the ordinary
reason CFAR and AGC references are order statistics rather than means -- here
it is worth 4.7 dB.  Range multi-look is a genuine cost either way, smearing a
target that lives in one range bin across three, so it is off by default and
priced rather than applied.

Acceptance criteria:
  * the boat's echo lands on the boat, within a beamwidth at 250 m;
  * the seabed and sea surface fill the image out to 300 m rather than a
    black background;
  * reverberation stays above the ambient across the whole swath, and the
    range at which it would not is reported;
  * the absorption budget is reported and is the dominant loss at 300 m;
  * the gain leaves the boat's contrast against its own background untouched,
    while range multi-look measurably costs it;
  * the hull is under one beamwidth at 250 m and over two through an array
    four times longer, which is what "make it clearer" actually requires;
  * the boat is NOT the brightest thing in this image the way it is at 90 m,
    and the example says by how much -- which is what decides whether a
    peak-anchored colour window leaves a clean picture or a full one;
  * the image still carries gradients to the scene.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch

from _common import banner, check, save, setup, timed
from hydropt import (
    ConstantLoss, IsoProfile, Scene, add_receiver_noise, azimuth_steering,
    beam_noise_power, beam_power_scale, beamform, calibrate,
    fractal_bathymetry, line_array_directivity_db, line_array_factor,
    make_time_grid, pierson_moskowitz_surface, sediment_loss, shading_window,
    target_arrivals, wave_number_peak_pm,
)
from hydropt.absorption import thorp_db_per_km
from hydropt.beamform import ArrivalSet
from hydropt.mesh import boat_hull_mesh, mesh_target
from hydropt.reverb import LambertScattering, reverberation_arrivals
from hydropt.tracer import trace

C = 1500.0
FREQ_KHZ = 120.0
LAMBDA = C / (FREQ_KHZ * 1e3)
WATER_DEPTH = 60.0
AUV_DEPTH = 25.0            # 35 m of altitude: enough to see 300 m of bottom
WIND = 4.0                  # a light breeze -- small waves, 0.09 m RMS

NEAR, FAR = 40.0, 300.0
SECTOR_DEG = 60.0
# Sized to a real head: 3.00 deg azimuth beams and a 20 deg vertical field of
# view, the latter carrying beams of 4.84 deg.  Measured, not assumed --
# `beam_3db_deg` below finds the half-power width of the actual array factor,
# because the quantity that is easy to compute, 2*asin(2/N), is the NULL-TO-NULL
# width and is 1.5 times larger.  Quoting one for the other overstates every
# cross-range figure by half again.
N_RX, N_TX = 50, 5          # 3.03 deg azimuth, 20.8 deg vertical FOV
VERTICAL_BEAM_DEG = 4.84    # beams within that FOV -- about four of them
ELEV_DEG = (-20.0, 23.0)    # the fan, wide enough to sample the FOV's skirts
TILT_DEG = 1.5              # down: what holds surface AND seabed past 170 m

# A LONG pulse, not the 0.12 ms of the 90 m examples.  Range resolution is
# c tau / 2 = 0.22 m here instead of 0.09, and that is the trade a long-range
# mode makes on purpose: the energy in the water goes up with the pulse length,
# the noise bandwidth goes down with it, and at 300 m you need both.
PULSE_S = 3.0e-4
N_BINS = 520
SOURCE_LEVEL_DB = 210.0     # dB re 1 uPa at 1 m

BOAT_RANGE = 250.0
BOAT_BEARING_DEG = -18.0
BOAT_HEADING_DEG = 40.0
HULL_LENGTH, HULL_BEAM, HULL_DRAUGHT = 12.0, 3.2, 1.0

N_ELEV, N_AZIM = 96, 330
# Every bounce the trace found, rather than a subsample: they are already paid
# for, and at these ranges the fan puts only 0.6-1.2 patches in a resolution
# cell, so throwing any away is throwing away the reverberation field itself.
PATCHES = None
SEED = 7


def beam_3db_deg(n: int, shading=None) -> float:
    """Half-power beamwidth of an ``n``-element half-wavelength line array.

    Measured off the array factor rather than taken from a formula.  The
    convenient closed form, ``2 asin(2/N)``, is the first-null spacing, and for
    a Hamming-shaded 64-element array it gives 3.58 degrees where the half-power
    width is 2.36 -- so using it silently inflates every cross-range extent, and
    with it every claim about what is and is not resolved.
    """
    a = torch.linspace(-0.6, 0.6, 200001, dtype=torch.float64)
    w = (torch.ones(n, dtype=torch.float64) if shading is None
         else shading.to(torch.float64))
    m = torch.arange(n, dtype=torch.float64) - (n - 1) / 2
    af = (w.reshape(1, -1)
          * torch.exp(1j * math.pi * a.reshape(-1, 1) * m.reshape(1, -1))
          ).sum(-1).abs() ** 2
    over = (af / af.max() > 0.5).nonzero().reshape(-1)
    return 2.0 * math.degrees(math.asin(float(a[int(over[-1])])))


def _ex15():
    path = Path(__file__).resolve().parent / "15_auv_scene_cartesian.py"
    spec = importlib.util.spec_from_file_location("_ex15", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def horizontal_array(n: int = N_RX) -> torch.Tensor:
    y = (torch.arange(n, dtype=torch.get_default_dtype()) - (n - 1) / 2) * (LAMBDA / 2)
    return torch.stack((torch.zeros_like(y), y,
                        torch.full_like(y, AUV_DEPTH)), dim=-1)


def build_scene(elements, *, seed: int = 3, learnable: bool = True):
    """A 60 m shelf, 700 m across, under a light wind sea.

    Both grids have to cover the whole of a 300 m swath and then some, because
    a ray that leaves the grid is clamped to its edge rather than refused, and a
    clamped seabed is a flat one -- which would show up as a suspiciously clean
    band at the outside of the image and nowhere else.
    """
    bottom = fractal_bathymetry((44, 44), (16.0, 16.0), base_depth=WATER_DEPTH,
                                rms=1.2, exponent=3.0, origin=(-40.0, -350.0),
                                learnable=learnable,
                                generator=torch.Generator().manual_seed(seed))
    # Eight nodes across the wind sea's peak wavelength, as always -- but over
    # 700 m rather than 180, which is what makes this the big array in the scene.
    dx = 2.0 * math.pi / wave_number_peak_pm(WIND) / 8.0
    n = int(math.ceil(700.0 / dx)) + 1
    surface = pierson_moskowitz_surface((n, n), (dx, dx), WIND,
                                        origin=(-40.0, -n * dx / 2),
                                        learnable=learnable,
                                        generator=torch.Generator().manual_seed(seed + 1))
    sediment = sediment_loss("sand", learnable=learnable)
    scene = Scene(
        field=IsoProfile(C, learnable=False), bottom=bottom, surface=surface,
        source=(0.0, 0.0, AUV_DEPTH), receivers=elements,
        surface_loss=ConstantLoss(0.0, learnable=False, pressure_release=True),
        bottom_loss=sediment, freqs_khz=torch.tensor([FREQ_KHZ]),
        # 2 m steps and 200 of them is 400 m of path -- enough that a ray still
        # has budget left after reaching 300 m.  Stopping at the range of
        # interest is the classic way to invent a detection limit out of the
        # ray budget, as examples/18 found the hard way.
        step_size=2.0, n_steps=200, max_bounces=6,
    )
    return scene, bottom, surface, sediment


def transmit_fan(n_elev: int = N_ELEV, n_azim: int = N_AZIM, *, seed: int = 0):
    """A wide azimuth swath, narrow in elevation, level rather than tilted.

    Jittered within each cell, for the reason examples/15 gives: a regular
    lattice in launch angle images as concentric arcs, which is the sampling
    pattern rather than the seabed.
    """
    g = torch.Generator().manual_seed(seed)
    e0, e1 = (math.radians(v) for v in ELEV_DEG)
    a0, a1 = -math.radians(SECTOR_DEG), math.radians(SECTOR_DEG)
    el = torch.linspace(e0, e1, n_elev)
    az = torch.linspace(a0, a1, n_azim)
    E, A = torch.meshgrid(el, az, indexing="ij")
    E, A = E.reshape(-1), A.reshape(-1)
    de = (e1 - e0) / max(n_elev - 1, 1)
    da = (a1 - a0) / max(n_azim - 1, 1)
    E = E + (torch.rand(E.shape, generator=g, dtype=E.dtype) - 0.5) * de
    A = A + (torch.rand(A.shape, generator=g, dtype=A.dtype) - 0.5) * da
    dirs = torch.stack([E.cos() * A.cos(), E.cos() * A.sin(), E.sin()], dim=-1)
    weights = line_array_factor(torch.sin(E), N_TX,
                                sin_steer=math.sin(math.radians(TILT_DEG)))
    return dirs, weights


def display(image, rng, *, pixel_m: float, tvg: bool = True, looks: int = 1,
            reference: str = "median"):
    """Range multi-look and TVG, on the [beams, bands, bins] image.

    Both are display, not physics -- the arrivals are untouched -- but at 300 m
    they are the difference between a picture and a mess, and neither is a
    cosmetic choice:

    * the multi-look window is set by the display's own pixel size, so it
      averages exactly the samples the pixel was going to alias anyway;
    * the gain is the swath's own level at each range, which is AGC rather than
      a fixed ``30 log r + 2 alpha r`` law, so the falloff does not have to be
      guessed in advance -- which at grazing incidence it would be.

    ``reference`` decides how that level is taken, and it is not a detail.  The
    obvious choice, the mean over beams, lets a target suppress itself: the
    boat here lifts the mean at its own range bin by **6.3 dB**, and the gain
    then divides it straight back out, costing 4 dB of the contrast it was
    supposed to leave alone (+17.5 dB becomes +13.5 dB).  Averaged over the
    ring the effect hides -- the lift is only 0.75 dB over a +/-8 m band --
    because the echo is concentrated in a couple of bins and the band is
    thirty.  A median over 181 beams cannot be moved by a handful of bright
    ones, and it recovers the contrast in full (+18.2 dB, marginally over the
    raw figure, whose ring mean is itself tilted by the falloff across the
    band).  This is the standard reason CFAR and AGC references are order
    statistics rather than means.
    """
    if looks <= 0:
        bin_m = float(rng[1] - rng[0])
        looks = max(1, int(round(pixel_m / bin_m)))
        if looks % 2 == 0:
            looks += 1                  # odd, so the window stays centred
    out = image
    if looks > 1:
        out = torch.nn.functional.avg_pool1d(
            out, kernel_size=looks, stride=1, padding=looks // 2,
            count_include_pad=False)
    if tvg:
        if reference == "median":
            level = out.median(dim=0, keepdim=True).values
        elif reference == "mean":
            level = out.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"reference must be 'median' or 'mean', got "
                             f"{reference!r}")
        out = out / level.clamp_min(1e-30)
    return out, looks


def main() -> int:
    setup()
    banner("21 -- 300 m of seabed, sea surface and a small boat")
    ex15 = _ex15()

    rx = horizontal_array()
    scene, bottom, surface, sediment = build_scene(rx)
    alt = WATER_DEPTH - AUV_DEPTH
    print(f"  {FREQ_KHZ:.0f} kHz, {N_RX} receive x {N_TX} transmit, "
          f"{2 * SECTOR_DEG:.0f} deg swath out to {FAR:.0f} m")
    print(f"  AUV at {AUV_DEPTH:.0f} m in {WATER_DEPTH:.0f} m of water -- "
          f"{alt:.0f} m of altitude")
    print(f"  wind {WIND:.0f} m/s: sea "
          f"{float(surface.heights.detach().std()):.3f} m RMS, "
          f"seabed {tuple(bottom.heights.shape)} nodes over 688 m")
    for r in (100.0, 200.0, FAR):
        print(f"    at {r:5.0f} m the bottom is "
              f"{math.degrees(math.atan2(alt, r)):4.1f} deg down, the surface "
              f"{math.degrees(math.atan2(AUV_DEPTH, r)):4.1f} deg up")
    fov_deg = beam_3db_deg(N_TX)
    blind = alt / math.tan(math.radians(TILT_DEG + fov_deg / 2))
    print(f"  {N_TX} transmit elements = a {fov_deg:.1f} deg vertical FOV "
          f"tilted {TILT_DEG:.1f} deg down,")
    print(f"  carrying beams of {VERTICAL_BEAM_DEG:.2f} deg -- about "
          f"{fov_deg / VERTICAL_BEAM_DEG:.0f} of them.  Its lower edge is")
    print(f"  {TILT_DEG + fov_deg / 2:.1f} deg down, so the bottom inside "
          f"{blind:.0f} m is in the skirts, not the beam.")
    print(f"  {N_RX} receive elements = "
          f"{beam_3db_deg(N_RX, shading_window(N_RX, 'hamming')):.2f} deg "
          f"azimuth beams at half power.")

    banner("the range budget")
    alpha = float(thorp_db_per_km(scene.freqs_khz))
    di = line_array_directivity_db(N_RX)
    bandwidth = 1.0 / PULSE_S
    noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=bandwidth,
                                   directivity_db=di, wind_speed=WIND))
    shading = shading_window(N_RX, "hamming")
    scale = beam_power_scale(shading, PULSE_S)
    print(f"  absorption {alpha:.1f} dB/km at {FREQ_KHZ:.0f} kHz:")
    for r in (100.0, 200.0, FAR):
        print(f"    {r:5.0f} m: {2 * alpha * r / 1000:5.1f} dB two-way "
              f"absorption, {40 * math.log10(r):5.1f} dB two-way spreading")
    at_60 = float(thorp_db_per_km(torch.tensor([60.0])))
    print(f"  (at 60 kHz it would be {at_60:.1f} dB/km -- "
          f"{2 * (alpha - at_60) * FAR / 1000:.0f} dB less at {FAR:.0f} m, for "
          f"beams {FREQ_KHZ / 60.0:.1f}x wider)")
    print(f"  {PULSE_S * 1e3:.2f} ms pulse = {PULSE_S * C / 2:.2f} m range cell, "
          f"{bandwidth / 1e3:.1f} kHz band")
    print(f"  SL {SOURCE_LEVEL_DB:.0f} dB re 1 uPa @ 1 m, DI {di:.1f} dB, "
          f"noise {10 * math.log10(noise):.1f} dB re 1 uPa^2 in a beam and cell")

    banner("ping")
    b = math.radians(BOAT_BEARING_DEG)
    verts, faces = boat_hull_mesh(HULL_LENGTH, HULL_BEAM, HULL_DRAUGHT,
                                  n_long=110, n_around=34)
    boat = mesh_target(
        verts, faces,
        position=(BOAT_RANGE * math.cos(b), BOAT_RANGE * math.sin(b),
                  HULL_DRAUGHT),
        yaw=BOAT_HEADING_DEG, n_patches=6, sound_speed=C,
        learnable=True, learnable_shape=False, facet_chunk=256)
    tx = BOAT_RANGE * math.cos(b)
    ty = BOAT_RANGE * math.sin(b)
    print(f"  {HULL_LENGTH:.0f} m boat at {BOAT_RANGE:.0f} m, bearing "
          f"{BOAT_BEARING_DEG:+.0f} deg, heading {BOAT_HEADING_DEG:.0f} deg")
    print(f"  we look up at it by "
          f"{math.degrees(math.atan2(AUV_DEPTH - HULL_DRAUGHT, BOAT_RANGE)):.1f} deg; "
          f"it subtends {math.degrees(HULL_LENGTH / BOAT_RANGE):.1f} deg")

    steer, bearings = azimuth_steering(181, SECTOR_DEG)
    grid = make_time_grid(2.0 * NEAR / C, 2.0 * FAR / C, N_BINS)
    seabed = LambertScattering(-27.0, learnable=True)
    dirs, tx_weights = transmit_fan(seed=SEED)
    solid = (math.radians(2 * SECTOR_DEG)
             * math.radians(ELEV_DEG[1] - ELEV_DEG[0]) / dirs.shape[0])

    t0 = time.perf_counter()
    with timed("  trace and scatter"):
        result = trace(scene, dirs)
        rev = reverberation_arrivals(
            result, dirs, scene.freqs_khz, scattering=seabed,
            solid_angle_per_ray=solid, ray_weights=tx_weights, boundary="both",
            surface=scene.surface, bottom=scene.bottom, max_arrivals=PATCHES,
            generator=torch.Generator().manual_seed(SEED + 1))
        echo = target_arrivals(scene, boat, dirs, n_rx_rays=420,
                               rx_half_angle_deg=45.0, tx_weights=tx_weights,
                               max_arrivals_per_leg=24,
                               generator=torch.Generator().manual_seed(SEED))
    both = ArrivalSet(*(None if rev[i] is None or echo[i] is None
                        else torch.cat([rev[i], echo[i]], dim=0)
                        for i in range(len(rev))))
    print(f"  {dirs.shape[0]} transmit rays -> {rev.n_arrivals} patches "
          f"+ {echo.n_arrivals} target arrivals")

    def render(arrivals):
        return beamform(arrivals, rx, scene.freqs_khz, grid, steer,
                        sigma_t=PULSE_S, shading=shading, steer_chunk=8)

    with timed("  beamform"):
        image = render(both)
        with torch.no_grad():
            echo_img = render(echo)
    forward = time.perf_counter() - t0

    # On an absolute scale, and then with the sea's own noise in it.
    signal = calibrate(image, SOURCE_LEVEL_DB, beam_scale=scale)
    noisy = add_receiver_noise(signal, noise,
                               generator=torch.Generator().manual_seed(SEED + 2))
    rng = grid * C / 2.0

    banner("where the seabed stops being the competition")
    with torch.no_grad():
        # Reverberation alone, averaged across the swath, against the noise in
        # one beam and cell.  The mean, not the median: reverberation is a
        # speckle field and its median sits far below the level a detector
        # competes with.
        rev_only = calibrate(render(rev), SOURCE_LEVEL_DB, beam_scale=scale)
        profile = rev_only[:, 0, :].mean(dim=0)
        prof_db = 10 * torch.log10(profile.clamp_min(1e-30))
        noise_db = 10 * math.log10(noise)
        lit = profile > 0
        first_lit = float(rng[int(lit.nonzero()[0])]) if bool(lit.any()) else float("nan")
        # Where reverberation would meet the noise floor, from the falloff it
        # actually has over the outer half of the swath rather than from the
        # r^-5 law -- the grazing angle is changing over that span too, and the
        # point of measuring is not to assume how the two combine.
        outer = (rng > 0.5 * FAR) & lit
        lr = torch.log10(rng[outer])
        pdb = prof_db[outer]
        slope = float(((lr - lr.mean()) * (pdb - pdb.mean())).sum()
                      / ((lr - lr.mean()) ** 2).sum())
        crossover = float(10 ** (lr[-1] + (noise_db - pdb[-1]) / slope))
        margin = float(pdb[-1]) - noise_db
    print(f"  reverberation at  50 m: "
          f"{float(prof_db[int((rng - 50.0).abs().argmin())]):6.1f} dB re 1 uPa^2")
    for r in (100.0, 150.0, 200.0, 250.0, 300.0):
        i = int((rng - r).abs().argmin())
        print(f"                   {r:4.0f} m: {float(prof_db[i]):6.1f} dB"
              f"{'  <-- noise floor ' + f'{noise_db:.1f}' if r == 300.0 else ''}")
    print(f"  the noise floor is {noise_db:.1f} dB, and reverberation is still "
          f"{margin:.1f} dB above it")
    print(f"  at {FAR:.0f} m.  Falling {slope:.0f} dB per decade of range over the "
          f"outer half of the")
    print(f"  swath, it would reach the floor at about {crossover:.0f} m -- "
          f"outside this picture.")
    print(f"  So the whole {FAR:.0f} m swath is REVERBERATION-limited: a louder "
          f"projector")
    print(f"  buys nothing here, and the way to see further is a lower "
          f"frequency, a")
    print(f"  narrower beam or a longer pulse, all of which change the ratio "
          f"rather")
    print(f"  than the level.")
    print(f"  (the first lit range is {first_lit:.0f} m -- the fan's own "
          f"near-field gap)")

    banner("the same ping, on a grid in metres")
    to_cart = lambda img: ex15.to_cartesian(img, bearings, grid, n_x=300,
                                            n_y=300, x_range=(-10.0, 305.0),
                                            y_range=(-260.0, 260.0))
    pixel_m = 315.0 / 299.0
    shown, _ = display(noisy, rng, pixel_m=pixel_m)          # median TVG
    by_mean, _ = display(noisy, rng, pixel_m=pixel_m, reference="mean")
    smoothed, looks = display(noisy, rng, pixel_m=pixel_m, looks=0)
    with timed("  resample to Cartesian"):
        cart, gx, gy = to_cart(shown)
    with torch.no_grad():
        raw_cart, _, _ = to_cart(noisy)
        mean_cart, _, _ = to_cart(by_mean)
        look_cart, _, _ = to_cart(smoothed)
        echo_cart, _, _ = to_cart(
            calibrate(echo_img, SOURCE_LEVEL_DB, beam_scale=scale))
    print(f"  {cart.shape[1]} x {cart.shape[0]} cells, "
          f"{float(gx[1] - gx[0]):.2f} x {float(gy[1] - gy[0]):.2f} m each")
    print(f"  {float(rng[1] - rng[0]):.2f} m range bins under a "
          f"{pixel_m:.2f} m pixel -> {looks} looks averaged per pixel")
    beamwidth = beam_3db_deg(N_RX, shading)
    beam_m = BOAT_RANGE * math.radians(beamwidth)
    print(f"  beam {beamwidth:.2f} deg at half power = {beam_m:.1f} m at the "
          f"boat, range cell {PULSE_S * C / 2:.2f} m")
    print(f"  (the first-null spacing is "
          f"{2 * math.degrees(math.asin(2.0 / N_RX)):.2f} deg -- quoting that "
          f"as the beamwidth")
    print(f"   inflates every cross-range figure by half again)")

    banner("can you see it at 250 m")
    det = cart.detach()
    GX = torch.as_tensor(gx).reshape(1, -1).expand_as(det)
    GY = torch.as_tensor(gy).reshape(-1, 1).expand_as(det)
    with torch.no_grad():
        flat = int(echo_cart.reshape(-1).argmax())
        px = float(gx[flat % det.shape[1]])
        py = float(gy[flat // det.shape[1]])
        err = max(0.0, math.hypot(px - tx, py - ty) - HULL_LENGTH / 2.0)
        # Background at the boat's OWN range, different bearing: the return
        # falls as r^-5, so a ring taken in the ground plane samples longer
        # ranges and flatters the target by tens of dB.
        rng_cell = torch.hypot(GX, GY)
        brg_cell = torch.rad2deg(torch.atan2(GY, GX))
        same_range = (rng_cell - BOAT_RANGE).abs() < 8.0
        off_target = (brg_cell - BOAT_BEARING_DEG).abs() > 6.0
        inside = brg_cell.abs() < SECTOR_DEG - 4.0
        ring = same_range & off_target & inside
        near_boat = torch.hypot(GX - tx, GY - ty) < 12.0

        def contrast(img):
            bg = img[ring]
            peak = float(img[near_boat].max())
            return (10 * math.log10(peak / float(bg.mean().clamp_min(1e-30))),
                    float((10 * torch.log10(bg / bg.mean().clamp_min(1e-30))).std()))

        srn, spread = contrast(det)
        raw_srn, raw_spread = contrast(raw_cart)
        look_srn, look_spread = contrast(look_cart)
        mean_srn, mean_spread = contrast(mean_cart)
    print(f"  the boat's echo peaks at ({px:+.1f}, {py:+.1f}) m, boat centred "
          f"on ({tx:+.1f}, {ty:+.1f})")
    print(f"  {err:.1f} m outside the hull, against {beam_m:.1f} m of beamwidth")
    print(f"\n  the same arrivals, four displays:")
    print(f"    no gain at all:           boat {raw_srn:+5.1f} dB, "
          f"background spread {raw_spread:.1f} dB")
    print(f"    TVG from the mean:        boat {mean_srn:+5.1f} dB, "
          f"background spread {mean_spread:.1f} dB")
    print(f"    TVG from the median:      boat {srn:+5.1f} dB, "
          f"background spread {spread:.1f} dB")
    print(f"    median TVG + {looks} looks:      boat {look_srn:+5.1f} dB, "
          f"background spread {look_spread:.1f} dB")
    print(f"  A gain taken from the MEAN lets the boat suppress itself: its "
          f"echo lifts")
    print(f"  the mean at its own range bin by about 6 dB, and the gain "
          f"divides that")
    print(f"  straight back out -- {mean_srn - raw_srn:+.1f} dB, hidden over "
          f"the ring because the lift is")
    print(f"  under a dB when smeared across a 16 m band.  A MEDIAN over "
          f"{len(bearings)} beams")
    print(f"  cannot be moved by a handful of bright ones, and keeps the "
          f"contrast")
    print(f"  ({srn - raw_srn:+.1f} dB).  This is why CFAR and AGC references "
          f"are order statistics.")
    print(f"  Multi-look is a real cost either way: it smears a target living "
          f"in one")
    print(f"  range bin across {looks}, for {look_srn - srn:+.1f} dB, buying "
          f"{spread - look_spread:.1f} dB of smoothness.")
    print(f"  Multi-look is not free: it smears a target that lives in one "
          f"range bin")
    print(f"  across {looks}, costing {look_srn - srn:+.1f} dB of contrast to "
          f"buy {spread - look_spread:.1f} dB of smoothness.")
    print(f"  For an unresolved target that is the wrong way round, so the "
          f"default is off.")

    banner("and why examples/15 looks black and this one does not")
    with torch.no_grad():
        peak_db = 10 * math.log10(float(torch.quantile(
            raw_cart[raw_cart > 0].reshape(-1), 0.99995)))
        boat_db = 10 * math.log10(float(raw_cart[near_boat].max()))
        ring_db = 10 * math.log10(float(raw_cart[ring].mean()))
    print(f"  brightest cell in this image (near-field seabed): "
          f"{peak_db:.1f} dB")
    print(f"  the boat:                                         "
          f"{boat_db:.1f} dB, {peak_db - boat_db:.1f} dB below it")
    print(f"  reverberation at the boat's range:                "
          f"{ring_db:.1f} dB")
    print(f"  examples/15 draws 22 dB below ITS peak, and there the peak is the")
    print(f"  boat -- so everything more than 22 dB under the target is clipped")
    print(f"  to black, which is most of its reverberation.  Here the peak is "
          f"{peak_db - boat_db:.0f} dB")
    print(f"  ABOVE the target, so the same 22 dB window would black out the "
          f"boat too,")
    print(f"  and a window wide enough to hold the near field is wide enough "
          f"to show")
    print(f"  the whole seabed.  That is the difference in look, and it is a "
          f"choice,")
    print(f"  not a measurement: the figure draws this image both ways.")

    banner("what the vertical field of view can hold")
    fov = beam_3db_deg(N_TX)
    print(f"  {N_TX} transmit elements give a {fov:.1f} deg vertical field of "
          f"view, tilted")
    print(f"  {TILT_DEG:.1f} deg down, carrying beams of "
          f"{VERTICAL_BEAM_DEG:.2f} deg -- about "
          f"{fov / VERTICAL_BEAM_DEG:.0f} of them.")
    print(f"  That FOV is the whole reason this scene has both boundaries in "
          f"it:")
    print(f"\n   range   surface   seabed   apart   inside a "
          f"{fov:.0f} deg FOV?")
    both_from = None
    for r in (100.0, 150.0, 200.0, 250.0, FAR):
        up = math.degrees(math.atan2(AUV_DEPTH, r))
        dn = math.degrees(math.atan2(WATER_DEPTH - AUV_DEPTH, r))
        fits = up + dn <= fov
        if fits and both_from is None:
            both_from = r
        print(f"   {r:5.0f} m  {up:5.2f} up  {dn:5.2f} dn  {up + dn:5.2f}   "
              f"{'yes' if fits else 'no -- one or the other'}")
    r = 1.0
    while (math.degrees(math.atan2(AUV_DEPTH, r))
           + math.degrees(math.atan2(WATER_DEPTH - AUV_DEPTH, r))) > fov:
        r += 1.0
    print(f"\n  Both fit beyond about {r:.0f} m.  Inside that the FOV holds "
          f"whichever")
    print(f"  boundary is nearer its axis, and the other is in the skirts -- "
          f"which is")
    print(f"  a geometry limit, not a power one: no vehicle depth fixes it, "
          f"because")
    print(f"  the separation depends only on the total water depth.")
    print(f"\n  And across the beam: {beamwidth:.2f} deg is {beam_m:.1f} m at "
          f"the boat, so a")
    print(f"  {HULL_LENGTH:.0f} m hull is {HULL_LENGTH / beam_m:.2f} "
          f"beamwidths -- about one.  It is a mark at")
    print(f"  {srn:+.1f} dB, which is a detection, not a shape.  Resolving it "
          f"needs more")
    print(f"  wavelengths across the aperture: at 300 kHz this same "
          f"{N_RX * LAMBDA / 2:.2f} m array")
    print(f"  would be {2 * (N_RX * LAMBDA / 2) / (C / 300e3):.0f} elements "
          f"and about "
          f"{beam_3db_deg(int(2 * (N_RX * LAMBDA / 2) / (C / 300e3)), None):.2f} deg, "
          f"but it would cost")
    print(f"  {2 * (float(thorp_db_per_km(torch.tensor([300.0]))) - alpha) * FAR / 1000:.0f} dB "
          f"more absorption at {FAR:.0f} m.  Aperture in WAVELENGTHS is the")
    print(f"  variable, and frequency is the cheap way to buy it.")

    banner("still differentiable, at 300 m")
    t0 = time.perf_counter()
    cart.sum().backward()
    backward = time.perf_counter() - t0
    live = {"boat position": boat.position, "boat heading": boat.orientation,
            "seabed": bottom.heights, "waves": surface.heights,
            "sediment c2": sediment.c2,
            "seabed backscatter": seabed.strength_db}
    states = {k: (p.grad is not None and bool(torch.isfinite(p.grad).all())
                  and float(p.grad.abs().sum()) > 0) for k, p in live.items()}
    for name, ok_g in states.items():
        print(f"  d(image)/d({name:<18s}): {'OK' if ok_g else 'ZERO'}")
    print(f"\n  forward {forward:.1f} s + backward {backward:.1f} s over "
          f"{both.n_arrivals} arrivals")

    save(_plot(det, raw_cart, mean_cart, gx, gy, rng, prof_db.detach(),
               noise_db, tx, ty, crossover, blind, looks),
         "21_long_range_300m.png")

    banner("acceptance")
    ok = check("the boat's echo lands on the boat at 250 m",
               err < beam_m,
               f"{err:.1f} m outside the hull against {beam_m:.1f} m of beamwidth")
    ok &= check("the seabed and surface fill the swath, not a black background",
                float((profile > noise).to(profile.dtype).mean()) > 0.3,
                f"{100 * float((profile > noise).to(profile.dtype).mean()):.0f}% "
                f"of range bins above the noise floor")
    ok &= check("the whole swath is reverberation-limited, and it says where "
                "that ends",
                margin > 3.0 and crossover > FAR,
                f"still {margin:.1f} dB above the ambient at {FAR:.0f} m; "
                f"crosses at about {crossover:.0f} m")
    ok &= check("absorption is the dominant loss at 300 m",
                2 * alpha * FAR / 1000 > 15.0,
                f"{2 * alpha * FAR / 1000:.1f} dB two-way at {FAR:.0f} m")
    ok &= check("a median gain keeps the contrast where a mean gain eats it",
                srn > raw_srn - 1.0 and mean_srn < raw_srn - 2.0,
                f"raw {raw_srn:+.1f}, mean {mean_srn:+.1f}, median "
                f"{srn:+.1f} dB")
    ok &= check("multi-look costs a target smaller than the cell it averages",
                look_srn < srn - 0.5 and look_spread < spread - 0.8,
                f"{look_srn - srn:+.1f} dB of contrast for "
                f"{spread - look_spread:.1f} dB of speckle over {looks} looks")
    ok &= check("the hull is about one beamwidth: a mark, not a shape",
                0.5 < HULL_LENGTH / beam_m < 2.0,
                f"{HULL_LENGTH / beam_m:.2f} beamwidths at {beamwidth:.2f} deg")
    ok &= check("the vertical FOV holds both boundaries over the far swath",
                both_from is not None and both_from < 0.8 * FAR,
                f"both inside {fov:.1f} deg beyond about {r:.0f} m")
    ok &= check("the image is still differentiable end to end",
                all(states.values()), f"{sum(states.values())}/{len(states)} live")
    return 0 if ok else 1


def _plot(cart, raw, by_mean, gx, gy, rng, prof_db, noise_db, tx, ty,
          crossover, blind, looks):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(18.0, 10.6))
    extent = [float(gx[0]), float(gx[-1]), float(gy[0]), float(gy[-1])]
    th = np.linspace(-math.radians(SECTOR_DEG), math.radians(SECTOR_DEG), 200)

    def swath(pos, img, title, label, span, ring=True):
        ax = fig.add_subplot(2, 3, pos)
        d = 10 * np.log10(np.maximum(img.numpy(), 1e-30))
        pk = float(np.quantile(d[np.isfinite(d)], 0.99995))
        im = ax.imshow(d, origin="lower", cmap="inferno", vmin=pk - span,
                       vmax=pk, extent=extent)
        if ring:
            ax.plot(blind * np.cos(th), blind * np.sin(th), ":",
                    color="deepskyblue", lw=0.9, alpha=0.6)
        ax.plot([tx], [ty], "o", mfc="none", mec="white", ms=15, mew=1.3)
        ax.annotate("boat, 250 m", (tx, ty), textcoords="offset points",
                    xytext=(14, 9), color="white", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlabel("forward (m)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label(label,
                                                               fontsize=7)
        return ax

    swath(1, raw, "no gain, 22 dB window\n(examples/15's convention: black, "
          "and no seabed)", "dB re 1 uPa$^2$", 22.0).set_ylabel("across (m)")
    swath(2, raw, "no gain, 45 dB window\n(the same data, the whole seabed "
          "in it)", "dB re 1 uPa$^2$", 45.0)
    swath(3, cart, "median TVG, 20 dB window\n(flat in range, contrast "
          "kept)", "dB re the background at that range", 20.0)
    swath(4, by_mean, "TVG from the MEAN instead\n(the boat lifts the "
          "reference and loses 4 dB)",
          "dB re the background at that range", 20.0).set_ylabel("across (m)")

    bx = fig.add_subplot(2, 3, 5)
    r = rng.numpy()
    bx.plot(r, prof_db.numpy(), lw=1.0, color="tab:orange",
            label="reverberation, swath mean")
    bx.axhline(noise_db, color="tab:blue", lw=1.0, ls="--",
               label=f"ambient noise, {noise_db:.0f} dB")
    bx.axvline(BOAT_RANGE, color="tab:green", lw=0.8, ls="-.", label="the boat")
    shown = prof_db.numpy()
    shown = shown[np.isfinite(shown) & (shown > -200)]
    bx.set_xlim(NEAR, FAR)
    bx.set_ylim(noise_db - 8.0, float(shown.max()) + 5.0)
    bx.annotate(f"reaches the noise floor only at\nabout {crossover:.0f} m, "
                f"past the swath", (0.97, 0.06), xycoords="axes fraction",
                ha="right", fontsize=8, color="0.3")
    bx.set_xlabel("range (m)")
    bx.set_ylabel("dB re 1 uPa$^2$")
    bx.set_title("reverberation-limited the whole way out", fontsize=10)
    bx.grid(alpha=0.3, lw=0.4)
    bx.legend(fontsize=8, loc="upper right")

    # the vertical geometry, which is what decides whether both boundaries are
    # in the picture at all
    gx_ax = fig.add_subplot(2, 3, 6)
    rr = np.linspace(40.0, FAR, 400)
    up = np.degrees(np.arctan2(AUV_DEPTH, rr))
    dn = np.degrees(np.arctan2(WATER_DEPTH - AUV_DEPTH, rr))
    fov = beam_3db_deg(N_TX)
    gx_ax.plot(rr, -up, color="tab:cyan", lw=1.2, label="sea surface")
    gx_ax.plot(rr, dn, color="tab:brown", lw=1.2, label="seabed")
    gx_ax.axhspan(TILT_DEG - fov / 2, TILT_DEG + fov / 2, color="tab:orange",
                  alpha=0.18, label=f"{fov:.0f} deg vertical FOV")
    for k in range(int(fov / VERTICAL_BEAM_DEG) + 1):
        e = TILT_DEG - fov / 2 + k * VERTICAL_BEAM_DEG
        gx_ax.axhline(e, color="tab:orange", lw=0.5, alpha=0.5)
    gx_ax.axvline(BOAT_RANGE, color="tab:green", lw=0.8, ls="-.")
    gx_ax.set_xlim(40.0, FAR)
    gx_ax.set_ylim(TILT_DEG + fov, TILT_DEG - fov)
    gx_ax.set_xlabel("range (m)")
    gx_ax.set_ylabel("elevation (deg, down positive)")
    gx_ax.set_title(f"{fov:.0f} deg FOV in beams of "
                    f"{VERTICAL_BEAM_DEG:.2f} deg:\nboth boundaries only "
                    f"where they fit inside it", fontsize=10)
    gx_ax.grid(alpha=0.3, lw=0.4)
    gx_ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{FREQ_KHZ:.0f} kHz, {N_RX} x {N_TX}: "
                 f"{beam_3db_deg(N_RX, shading_window(N_RX, 'hamming')):.2f} "
                 f"deg x {VERTICAL_BEAM_DEG:.2f} deg beams over a "
                 f"{fov:.0f} deg vertical FOV, out to {FAR:.0f} m", y=0.995)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    raise SystemExit(main())
