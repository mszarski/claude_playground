"""A long look: 300 m of seabed, sea surface and one small boat.

Everything the earlier FLS examples do, moved out to where the range equation
starts to bite.  A 120 kHz Mills cross on an AUV at 12 m in 30 m of water,
looking out to 300 m: a rough sand seabed, a light wind sea, and a 30 m boat on
the surface at 250 m, 58 degrees off the line of sight.  No wake -- just the
scene.

The head is modelled as it is built: a vertical transmit array forming four
4.84 degree elevation beams across a 20 degree field of view tilted 5 degrees
up, a horizontal receive array forming 3 degree bearing beams, and receive
staves about 20 degrees tall in elevation.  One trace serves all four beams;
each is scattered, solved and beamformed with its own transmit weights, and
the screen shows them summed -- or the maximum, or one alone, since which a
given display does is not always documented and the example draws all three.

**Three things change when you go from 90 m to 300 m, and none of them is the
picture getting bigger.**

*Absorption stops being a rounding error.*  Seawater takes about 34 dB/km at
100 kHz, so the two-way loss to 300 m is 20 dB on top of spreading.  That is
the reason a 300 m set is usually built at 60 kHz or below, trading 13 dB of
absorption for beams 1.7 times wider.  The budget is printed.

*The near field goes dark, and the vertical FOV is why.*  The head here is the
real thing: 3.00 degree azimuth beams and a 20 degree vertical field of view
carrying beams of 4.84 degrees, about four of them, flown tilted 5 degrees UP.

Twenty degrees is not much at 300 m, and it is the quantity that decides what
is in the picture at all.  Tilted up, the lobe runs from 15.4 degrees above the
horizontal to 5.4 below, so the sea surface is in it from 54 m out -- which is
the point of flying that way, since a surface contact at a few hundred metres
then sits within a fraction of a beam of the axis rather than on its skirt.
The seabed only enters where ``altitude <= 0.094 x range``: under 24 m of
altitude at 250 m.  In 30 m of water flown mid-column that is satisfied past
about 160 m, so the near field of this image is sea surface alone and the
seabed joins it halfway out.  In 60 m of water, with the same attitude, the
bottom would not be in the lobe anywhere inside 300 m.  Geometry, not power:
tilt, altitude and field of view between them decide what a ping can contain.

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
itself: the echo lifts the mean at its own range bin, and the gain then divides
that lift straight back out.  How much depends on how bright the target is --
in a deeper-water version of this scene a boat standing +17.5 dB over its
background lost 4 dB that way, and the example measures the lift rather than
quoting it.  It hides from a ring measurement, because spread over a +/-8 m
band the lift is under a decibel.  A MEDIAN over 181 beams cannot be moved by a
handful of bright ones and never does worse, which is the ordinary reason CFAR
and AGC references are order statistics rather than means.  Range multi-look is a genuine cost either way, smearing a
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
import os
import time
from pathlib import Path

import torch

from _common import FIGURE_DIR, banner, check, save, setup, timed
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
# 12 m down in 30 m of water, as flown.  With the fan tilted UP the seabed only
# enters the main lobe where `altitude <= 0.094 x range`: 18 m of altitude puts
# it at 191 m, so the inner two thirds of this swath is sea surface alone.  The
# grazing angles that result are the other consequence -- 3.4 degrees on the
# bottom at 300 m and 2.3 on the surface -- and Lambert scattering goes as
# sin(theta), so a tilted-up geometry is a quiet one.
WATER_DEPTH = 30.0
AUV_DEPTH = 12.0            # 18 m of altitude: the bottom is in the lobe past 191 m
WIND = 4.0                  # a light breeze -- small waves, 0.09 m RMS

# The swath.  Overridable, because the same scene at a different range is the
# comparison that shows what is geometry and what is display:
#   HYDROPT_FAR=90 HYDROPT_NEAR=8 HYDROPT_BOAT=75 python 21_long_range_300m.py
FAR = float(os.environ.get("HYDROPT_FAR", 300.0))
NEAR = float(os.environ.get("HYDROPT_NEAR", 40.0))
SECTOR_DEG = 60.0
# Sized to a real head: 3.00 deg azimuth beams and a 20 deg vertical field of
# view, the latter carrying beams of 4.84 deg.  Measured, not assumed --
# `beam_3db_deg` below finds the half-power width of the actual array factor,
# because the quantity that is easy to compute, 2*asin(2/N), is the NULL-TO-NULL
# width and is 1.5 times larger.  Quoting one for the other overstates every
# cross-range figure by half again.
# The head as described: 3 x 4.84 degree beams over a 20 degree vertical FOV.
# Five vertical elements make ONE 20.8 degree lobe -- which is the FOV, not a
# beam, and for a while this example treated it as the beam.  A 4.84 degree
# elevation beam takes about 21 elements at half-wavelength spacing, and four
# of them, steered, cover the 20 degrees.  The difference is not cosmetic: the
# seabed-image paths to a surface target arrive 8 to 10 degrees below the
# direct one, inside a 20 degree lobe and 20 dB down in a 4.84 degree beam,
# and the reverberation in a beam is only what lies within 2.4 degrees of it.
# Which leg carries them is the design question, and the answer for a head
# that classifies seafloor, water column and surface is RECEIVE: receive
# beams sort each return by the elevation it ARRIVES from, which is what
# attribution needs -- the seabed's return from below, the surface's from
# above, a hull's seabed-image ghost in the lower beam where it belongs.
# Transmit-steered beams sort by where the sound went and tag that ghost as
# surface.  And receive-formed beams come from one flooded ping, where
# transmit-steered ones cost four.  The SUMMED picture is the same either way,
# to first order: it depends only on the two-way pattern.
N_RX = 50                                          # 3.03 deg azimuth
N_TX = int(os.environ.get("HYDROPT_N_TX", 5))      # floods the 20.8 deg FOV
VERTICAL_BEAM_DEG = 4.84    # beams within that FOV -- about four of them
ELEV_DEG = (-27.0, 17.0)    # the fan, wide enough to sample the FOV's skirts
# Four 4.84 degree beams over a FOV tilted 5 degrees up run from 15.4 degrees
# up to 5.4 down, centred at 12.3, 7.4, 2.6 and -2.3 degrees (up positive);
# the boat at 250 m sits 2.3 degrees up, in the third.  HYDROPT_TILT moves
# the FOV centre.
TILT_DEG = float(os.environ.get("HYDROPT_TILT", -5.0))   # FOV centre; negative is up
# Two ways to run the elevation, one switch:
#
#   HYDROPT_ELEVATION=envelope   (the default)  One pass, with the receive
#       weight the SUM of the four beam patterns.  The beamformer is linear
#       in the arrivals, so every arrival's own energy lands exactly where
#       the four-beam sum would put it; what differs is the cross-terms.
#       Two arrivals in different beams add in power under the four-beam
#       sum and interfere under the envelope (Cauchy-Schwarz: the envelope's
#       cross weight is never smaller).  Measured against the four beams
#       summed, at 300 m: contrast +36.0 against +35.8 dB, clutter 63.4
#       against 63.6, the +1.0 m ghost -6.9 against -8.3 dB, and the hull's
#       width across bearing 31.1 against 27.4 m (28.7 predicted) -- a
#       quarter of the arrivals, and no per-elevation attribution.
#
#   HYDROPT_ELEVATION=beams      Four 4.84 degree receive beams across the
#       FOV, each scattered, solved and beamformed on its own, and combined
#       per HYDROPT_DISPLAY (sum, max, beam:K).  Four times the cost; the
#       per-beam pictures for seafloor / water column / surface attribution,
#       the three-convention figure, and the saved stack that
#       21_redraw_conventions.py reads.
#
# The per-beam element count and beam count can still be set directly.
ELEVATION = os.environ.get("HYDROPT_ELEVATION", "envelope")
if ELEVATION not in ("envelope", "beams"):
    raise SystemExit(f"HYDROPT_ELEVATION must be 'envelope' or 'beams', got {ELEVATION!r}")
N_RX_ELEV = int(os.environ.get("HYDROPT_N_RX_ELEV", 21))   # the head's beams, either way
N_ELEV_BEAMS = int(os.environ.get("HYDROPT_N_ELEV_BEAMS", 4))
# The envelope is the SUM of the four beam patterns, not a beam as wide as the
# four.  Measured: a 5-element receive beam reproduced the summed display's
# contrast to 0.2 dB and its clutter to 0.1, and put the hull's width across
# bearing back to 34.8 m from 27.4 -- because its -13 dB sidelobes admit the
# seabed-image paths ten degrees down that four tiled 21-element beams reject
# at their sharp edge.  Same shortcut, the right envelope.
# HYDROPT_RX_ELEV=point makes the receive side accept every elevation
# equally, which is what a horizontal line of point elements does and what
# made a seabed image ten degrees below boresight count at full strength.
RX_ELEV_BEAMS = os.environ.get("HYDROPT_RX_ELEV", "beams") != "point"
# How the four elevation beams reach the screen: "sum" (added in power),
# "max" (the brightest beam per cell), or "beam:K" for one of them alone.
# Which one a given head does is not always documented, so the example draws
# all three side by side and the operator says which is theirs.
DISPLAY = os.environ.get("HYDROPT_DISPLAY", "sum")


def beam_tilts_deg() -> list[float]:
    """Centres of the head's elevation beams, spaced one beamwidth about the tilt."""
    bw = beam_3db_deg(N_RX_ELEV)
    return [TILT_DEG + (k - (N_ELEV_BEAMS - 1) / 2.0) * bw for k in range(N_ELEV_BEAMS)]


def passes_deg() -> list[float]:
    """What the example actually loops over: every beam, or one envelope pass."""
    return beam_tilts_deg() if ELEVATION == "beams" else [TILT_DEG]

# A LONG pulse, not the 0.12 ms of the 90 m examples.  Range resolution is
# c tau / 2 = 0.22 m here instead of 0.09, and that is the trade a long-range
# mode makes on purpose: the energy in the water goes up with the pulse length,
# the noise bandwidth goes down with it, and at 300 m you need both.
PULSE_S = 3.0e-4
# Range bins at about 0.5 m, roughly twice the 0.22 m cell the pulse gives, so
# the image samples the pulse rather than the sampling.
N_BINS = int(round((FAR - NEAR) / 0.5))
SOURCE_LEVEL_DB = 210.0     # dB re 1 uPa at 1 m
# Display floor for the target picture, in dB over the local background after
# the gain has flattened it.  At +6 dB under 2 percent of a Rayleigh background
# survives, so the picture goes black and what is left is worth looking at --
# at every range, which is the thing a fixed window cannot do.
THRESHOLD_DB = 6.0

BOAT_RANGE = float(os.environ.get("HYDROPT_BOAT", 0.833 * FAR))
BOAT_BEARING_DEG = -18.0
BOAT_HEADING_DEG = float(os.environ.get("HYDROPT_HEADING", 40.0))
# Lambert strength of the hull's own surface and structure: the parameter that
# decides whether the vessel is visible anywhere but beam-on.  It is set by
# matching a MEASURED property of real vessels rather than by making the boat
# appear: surface ships and submarines are reported at about 25 dB of target
# strength at beam aspect and 10 to 12 dB bow-on and stern-on, a swing of 10
# to 15 dB (Urick; DTIC AD0039542 and AD0531451).  On this hull, measured at
# the scene's elevation with the corrected mesh:
#
#     mu      TS beam   TS 40 deg   swing
#     mirror   +15.4     -28.2      43.7 dB
#     -27      +15.5     -10.9      26.4      (sand, the earlier stand-in)
#     -15      +15.7      +1.0      14.8      <- the literature's range
#     -10      +16.3      +6.0      10.3      <-
#
# -12 dB is the middle of that range.  Beam aspect is unmoved by any of these
# -- the diffuse channel fills the nulls the mirror leaves and inflates
# nothing.  It is a learnable parameter, so an image of a real vessel at a
# known aspect can fit it instead of assuming it.
_diffuse = os.environ.get("HYDROPT_DIFFUSE", "-12")
DIFFUSE_DB = None if _diffuse == "off" else float(_diffuse)
# A 30 m hull drawing 4 m is a trawler or a small coaster, and those carry 7 to
# 9 m of beam: 8.0 gives a length-to-beam of 3.75 and about 490 tonnes, which
# are the proportions of a real vessel rather than of a rowing shell.
HULL_LENGTH, HULL_BEAM, HULL_DRAUGHT = 30.0, 8.0, 4.0

N_ELEV, N_AZIM = 96, 330
# Every bounce the trace found, rather than a subsample: they are already paid
# for, and at these ranges the fan puts only 0.6-1.2 patches in a resolution
# cell, so throwing any away is throwing away the reverberation field itself.
PATCHES = None
SEED = 7


def _range_marks(n: int = 3):
    """Ranges to tabulate, spread across whatever swath is configured.

    Hard-coding 100, 200, 300 quietly tabulates ranges outside a 90 m swath,
    and a table that reports what happens at 200 m in a 90 m picture is worse
    than no table -- it decided, once, that both boundaries were in the lobe
    when neither was inside the swath at all.
    """
    lo = max(NEAR, 0.2 * FAR)
    return [lo + (FAR - lo) * i / (n - 1) for i in range(n)]


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
                                rms=0.9, exponent=3.0, origin=(-40.0, -350.0),
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
    # The boundaries conserve energy on a bounce, deliberately.  A rough sea
    # at 120 kHz destroys the COHERENT reflection -- the Eckart loss is
    # thousands of decibels -- but a pressure-release surface reflects all of
    # the energy; roughness randomises its phase and spreads it over a few
    # degrees, it removes nothing.  Reverberation is an energy quantity and a
    # Lambert patch does not care about the phase of what reaches it, so a ray
    # continuing down the waveguide after a surface bounce at full energy is
    # right, and the surface-bottom multipath it produces is real shallow-water
    # physics: 12.5 dB of it at 250 m here.  Charging Eckart on the bounce
    # (which this scene briefly did) removed that, and the "sonar equation"
    # level it then matched is the direct-path-only textbook form.  Eckart
    # belongs on the target's coherent bounce paths, where the eigenray legs
    # already pay it.
    scene = Scene(
        field=IsoProfile(C, learnable=False), bottom=bottom, surface=surface,
        source=(0.0, 0.0, AUV_DEPTH), receivers=elements,
        surface_loss=ConstantLoss(0.0, learnable=False, pressure_release=True),
        bottom_loss=sediment,
        freqs_khz=torch.tensor([FREQ_KHZ]),
        # 2 m steps and 200 of them is 400 m of path -- enough that a ray still
        # has budget left after reaching 300 m.  Stopping at the range of
        # interest is the classic way to invent a detection limit out of the
        # ray budget, as examples/18 found the hard way.
        step_size=2.0, n_steps=200, max_bounces=6,
    )
    return scene, bottom, surface, sediment


def transmit_fan(n_elev: int = N_ELEV, n_azim: int = N_AZIM, *, seed: int = 0,
                 tilt_deg: float | None = None):
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


def receive_beam(directions: torch.Tensor, tilt_deg: float) -> torch.Tensor:
    """One receive elevation beam, at unit ARRIVAL directions.

    Power weight, one on the beam's axis.  Sound arriving from ``tilt_deg``
    up is propagating downward, so its ``z`` (depth-down) is ``+sin(tilt)``;
    that is the steer.  Point elements (``HYDROPT_RX_ELEV=point``) return one
    everywhere.
    """
    if not RX_ELEV_BEAMS:
        return torch.ones(directions.shape[:-1], dtype=directions.dtype,
                          device=directions.device)
    if ELEVATION == "envelope":
        # the sum of the head's beams, in power -- see the note at ELEVATION
        return sum(line_array_factor(directions[..., 2], N_RX_ELEV,
                                     sin_steer=-math.sin(math.radians(t)))
                   for t in beam_tilts_deg())
    return line_array_factor(directions[..., 2], N_RX_ELEV,
                             sin_steer=-math.sin(math.radians(tilt_deg)))


def transmit_pattern(directions: torch.Tensor, tilt_deg: float | None = None
                     ) -> torch.Tensor:
    """The same projector directivity, as a function of direction.

    ``transmit_fan`` returns it as one weight per ray, which is what the splat
    inbound leg indexes.  An eigenray inbound leg solves for paths and has no
    ray to index, so it needs the pattern evaluated at the launch direction the
    path actually left in.

    Exactly the same function, not an approximation of it: the fan builds its
    directions as ``[cos E cos A, cos E sin A, sin E]``, so the ``z`` component
    IS ``sin E``, which is the only thing the array factor depends on.
    """
    return line_array_factor(directions[..., 2], N_TX,
                             sin_steer=math.sin(math.radians(TILT_DEG)))


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
    echo lifts the mean at its own range bin, and the gain divides that lift
    straight back out of the very thing it was supposed to leave alone.  The
    size of it scales with the target -- in a deeper-water version of this
    scene a boat at +17.5 dB lost 4 dB, while a fainter one loses little -- and
    it hides from a ring measurement, because the echo sits in a couple of bins
    while the ring spans thirty.  A median over 181 beams cannot be moved by a
    handful of bright ones and never does worse than the mean, which is the
    standard reason CFAR and AGC references are order statistics.  The example
    measures the lift each run rather than quoting a number from another scene.
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
            # A range bin lit in fewer than half its beams has a median of
            # ZERO, and dividing by it amplifies the few lit cells without
            # bound -- measured, 1e27 on a bin lit in 30 beams of 181, which
            # saturates the display and erases everything else in it.  It does
            # not show up wherever ambient noise fills every bin, which is why
            # it can sit unnoticed in a scene that has noise and appear in one
            # that does not.  Fall back to the mean there, which is nonzero
            # whenever anything at all is lit.
            level = torch.where(level > 0.0, level, out.mean(dim=0,
                                                             keepdim=True))
        elif reference == "mean":
            level = out.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"reference must be 'median' or 'mean', got "
                             f"{reference!r}")
        # ...and a bin lit in a handful of beams still has a tiny reference, so
        # floor it against the swath as a whole rather than against zero.
        floor = 1e-6 * float(level.max())
        out = out / level.clamp_min(floor)
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
    for r in _range_marks():
        print(f"    at {r:5.0f} m the bottom is "
              f"{math.degrees(math.atan2(alt, r)):4.1f} deg down, the surface "
              f"{math.degrees(math.atan2(AUV_DEPTH, r)):4.1f} deg up")
    fov_deg = beam_3db_deg(N_TX)   # the flooded transmit lobe IS the FOV
    edge_dn = TILT_DEG + fov_deg / 2
    edge_up = TILT_DEG - fov_deg / 2
    blind = alt / math.tan(math.radians(edge_dn)) if edge_dn > 0 else float("inf")
    surf_from = AUV_DEPTH / math.tan(math.radians(-edge_up))
    print(f"  {N_TX} transmit elements flood a {fov_deg:.1f} deg vertical FOV; {N_RX_ELEV} vertical receive elements form {N_ELEV_BEAMS} beams of {beam_3db_deg(N_RX_ELEV):.2f} deg "
          f"tilted {abs(TILT_DEG):.1f} deg "
          f"{'up' if TILT_DEG < 0 else 'down'},")
    print(f"  carrying beams of {VERTICAL_BEAM_DEG:.2f} deg -- about "
          f"{fov_deg / VERTICAL_BEAM_DEG:.0f} of them.  The lobe runs "
          f"{-edge_up:.1f} deg up to {edge_dn:.1f} deg down,")
    print(f"  so the sea surface is in it from {surf_from:.0f} m and the "
          f"seabed from {blind:.0f} m:")
    print(f"  the near field of this picture is surface alone.")
    print(f"  {N_RX} receive elements = "
          f"{beam_3db_deg(N_RX, shading_window(N_RX, 'hamming')):.2f} deg "
          f"azimuth beams at half power.")
    boat_el = -math.degrees(math.atan2(AUV_DEPTH - HULL_DRAUGHT, BOAT_RANGE))
    print(f"  the boat at {BOAT_RANGE:.0f} m sits {-boat_el:.2f} deg up, "
          f"{abs(boat_el - TILT_DEG):.2f} deg off the vertical axis")
    print(f"  = {abs(boat_el - TILT_DEG) / VERTICAL_BEAM_DEG:.2f} of a beam, "
          f"which is what the up-tilt is for.")

    banner("the range budget")
    alpha = float(thorp_db_per_km(scene.freqs_khz))
    di = line_array_directivity_db(N_RX)
    bandwidth = 1.0 / PULSE_S
    noise = float(beam_noise_power(scene.freqs_khz, bandwidth_hz=bandwidth,
                                   directivity_db=di, wind_speed=WIND))
    shading = shading_window(N_RX, "hamming")
    scale = beam_power_scale(shading, PULSE_S)
    print(f"  absorption {alpha:.1f} dB/km at {FREQ_KHZ:.0f} kHz:")
    for r in _range_marks():
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
    # Facets scaled with the hull -- along it with the length, around it with
    # the girth -- so they stay the size they were rather than coarsening as
    # the boat grows.  A facet has to resolve the curvature it stands in, so
    # letting them grow with the vessel would quietly change the scattering.
    girth = HULL_BEAM + 2.0 * HULL_DRAUGHT
    verts, faces = boat_hull_mesh(
        HULL_LENGTH, HULL_BEAM, HULL_DRAUGHT,
        n_long=int(round(110 * HULL_LENGTH / 12.0)),
        n_around=int(round(34 * girth / (3.2 + 2.0 * 4.0))))
    boat = mesh_target(
        verts, faces,
        # z=0, not the draught: boat_hull_mesh returns the WETTED surface with
        # its waterline at z=0, so placing the body at z=draught sinks the boat
        # by its own draught.  At 1 m that hid; at 4 m it put the hull at 5.0 to
        # 6.25 m depth, entirely submerged, which is a different target.
        position=(BOAT_RANGE * math.cos(b), BOAT_RANGE * math.sin(b), 0.0),
        yaw=BOAT_HEADING_DEG, n_patches=6, sound_speed=C,
        # Without this the hull is a mirror and exists only at beam aspect:
        # +10.1 dB of target strength broadside against -23.9 dB at the 58
        # degrees off the line of sight this scene views it at.  A real vessel
        # swings 10 to 20 dB across that, not 34, because of everything a
        # faired analytic surface does not have -- ribs, seams, a rudder, a
        # prop.  HYDROPT_DIFFUSE=off restores the mirror, which is the
        # comparison that shows what it is worth.
        diffuse_db=DIFFUSE_DB,
        learnable=True, learnable_shape=False, facet_chunk=256)
    tx = BOAT_RANGE * math.cos(b)
    ty = BOAT_RANGE * math.sin(b)
    print(f"  {HULL_LENGTH:.0f} m boat at {BOAT_RANGE:.0f} m, bearing "
          f"{BOAT_BEARING_DEG:+.0f} deg, heading {BOAT_HEADING_DEG:.0f} deg")
    print(f"  draught {HULL_DRAUGHT:.1f} m, so the wetted hull hangs from the "
          f"surface to {HULL_DRAUGHT:.1f} m")
    print(f"  we look up at its keel by "
          f"{math.degrees(math.atan2(AUV_DEPTH - HULL_DRAUGHT, BOAT_RANGE)):.2f} deg "
          f"and at its waterline by "
          f"{math.degrees(math.atan2(AUV_DEPTH, BOAT_RANGE)):.2f} deg;")
    print(f"  it subtends {math.degrees(HULL_LENGTH / BOAT_RANGE):.2f} deg in "
          f"bearing and {math.degrees(HULL_DRAUGHT / BOAT_RANGE):.2f} deg in "
          f"elevation")
    # Which axis the hull's length lands on decides whether it is resolved:
    # range resolution is the pulse and is the same everywhere, bearing
    # resolution is the beam and grows linearly with range.
    aspect = math.radians(BOAT_HEADING_DEG - BOAT_BEARING_DEG)
    across = abs(HULL_LENGTH * math.sin(aspect))
    along = abs(HULL_LENGTH * math.cos(aspect))
    print(f"  it lies {math.degrees(aspect):.0f} deg off the line of sight, so "
          f"its {HULL_LENGTH:.0f} m splits into")
    print(f"  {across:.1f} m across bearing and {along:.1f} m along range")

    steer, bearings = azimuth_steering(181, SECTOR_DEG)
    grid = make_time_grid(2.0 * NEAR / C, 2.0 * FAR / C, N_BINS)
    seabed = LambertScattering(-27.0, learnable=True)
    dirs, tx_weights = transmit_fan(seed=SEED)
    solid = (math.radians(2 * SECTOR_DEG)
             * math.radians(ELEV_DEG[1] - ELEV_DEG[0]) / dirs.shape[0])
    # One flooded transmit serves every beam.  Each receive beam's weight on
    # the reverberation's return leg: that leg is the outbound ray reversed,
    # so its arrival direction is the launch direction negated, and the weight
    # is a power weight that rides with the transmit one, applied once.
    w_tx = transmit_pattern(dirs)

    tilts = passes_deg()
    if DISPLAY.startswith("beam:"):
        tilts = [tilts[int(DISPLAY.split(":")[1])]]
    print(f"  {len(tilts)} elevation beam(s) at "
          + ", ".join(f"{-t:+.1f}" for t in tilts) + " deg (up positive), "
          f"display '{DISPLAY}', formed on receive"
          f"{'' if RX_ELEV_BEAMS else ' -- DISABLED, point elements'}")

    def render(arrivals):
        return beamform(arrivals, rx, scene.freqs_khz, grid, steer,
                        sigma_t=PULSE_S, shading=shading,
                        steer_chunk=int(os.environ.get("HYDROPT_STEER_CHUNK", 8)))

    t0 = time.perf_counter()
    with timed("  trace"):
        result = trace(scene, dirs)          # once; every beam reweights it
    # Four beams under autograd held 12.5 GB against a ceiling near 14, and
    # the backward pass took 330 s.  The gradient goes through the beam the
    # boat is in; the other three are evaluated without a graph.  The summed
    # IMAGE is identical either way -- only which beams carry the gradient.
    boat_el = -math.degrees(math.atan2(AUV_DEPTH - HULL_DRAUGHT / 2.0, BOAT_RANGE))
    k_boat = int(min(range(len(tilts)), key=lambda k: abs(tilts[k] - boat_el)))
    if len(tilts) > 1:
        print(f"  gradients flow through the boat's beam ({-tilts[k_boat]:+.1f} deg); "
              f"the others are evaluated without a graph")
    per_beam, per_beam_echo, n_patch, n_echo = [], [], 0, 0
    for k, tilt in enumerate(tilts):
        rx_beam = lambda d, t=tilt: receive_beam(d, t)
        with timed(f"  beam at {-tilt:+.1f} deg: scatter, target, beamform"), \
             torch.set_grad_enabled(k == k_boat):
            rev = reverberation_arrivals(
                result, dirs, scene.freqs_khz, scattering=seabed,
                solid_angle_per_ray=solid, ray_weights=w_tx * rx_beam(-dirs),
                boundary="both", surface=scene.surface, bottom=scene.bottom,
                max_arrivals=PATCHES,
                generator=torch.Generator().manual_seed(SEED + 1))
            # The return leg SOLVES for its paths rather than sampling them;
            # see the note on return_leg in an earlier revision.  The transmit
            # pattern is evaluated at each solved inbound launch direction and
            # the receive stave at each solved outbound arrival direction.
            echo = target_arrivals(
                scene, boat, dirs, return_leg="eigenray",
                n_rx_rays=2000, rx_half_angle_deg=45.0, tx_weights=w_tx,
                tx_pattern=transmit_pattern,
                rx_pattern=rx_beam, max_arrivals_per_leg=24,
                generator=torch.Generator().manual_seed(SEED))
            both = ArrivalSet(*(None if rev[i] is None or echo[i] is None
                                else torch.cat([rev[i], echo[i]], dim=0)
                                for i in range(len(rev))))
            per_beam.append(render(both))
            with torch.no_grad():
                per_beam_echo.append(render(echo))
        n_patch += rev.n_arrivals
        n_echo += echo.n_arrivals
    print(f"  {dirs.shape[0]} transmit rays -> {n_patch} patches "
          f"+ {n_echo} target arrivals over {len(tilts)} beam(s)")

    stack = torch.stack(per_beam)                 # [beams, bearings, bands, bins]
    stack_echo = torch.stack(per_beam_echo)
    if DISPLAY == "max":
        image, echo_img = stack.max(dim=0).values, stack_echo.max(dim=0).values
    else:                                        # "sum", or a single beam
        image, echo_img = stack.sum(dim=0), stack_echo.sum(dim=0)
    class _Both:                                 # what later prints read
        n_arrivals = n_patch + n_echo
    both = _Both()
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
    span_y = FAR * math.sin(math.radians(SECTOR_DEG)) * 1.02
    x_range = (-0.03 * FAR, 1.02 * FAR)
    to_cart = lambda img: ex15.to_cartesian(img, bearings, grid, n_x=300,
                                            n_y=300, x_range=x_range,
                                            y_range=(-span_y, span_y))
    pixel_m = (x_range[1] - x_range[0]) / 299.0
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

        # Everything in the picture the boat has to beat, which is not the
        # same as everything at its own range.  A first attempt used the ring
        # -- a few thousand cells within 8 m of the boat's range -- and the
        # oblique boat beat all of them while being invisible in the image.
        # It was not a contradiction: reverberation falls as r^-5, so at 250 m
        # the boat IS the brightest thing in its own range band while tens of
        # thousands of cells further in are brighter still.  A range-normalised
        # display is exactly what removes that excuse, so the comparison has to
        # be made on the gain-corrected image, against the whole lit swath.
        lit_cell = (rng_cell > NEAR) & (rng_cell < FAR) & inside
        clutter = lit_cell & (torch.hypot(GX - tx, GY - ty) > 3.0 * beam_m)

        def false_alarms(img):
            """What fraction of the picture is brighter than the target.

            This, and not the contrast in decibels, is whether you can see it.
            Contrast compares the target's PEAK against the background's MEAN,
            and one-look speckle is exponential: its own peaks run about 10 dB
            over its mean routinely, so a target at "+10 dB" is simply one more
            bright cell among thousands.  A detector set at the target's level
            would raise this fraction of the background with it, which is the
            false-alarm rate, and it is the number an operator lives with.
            """
            bg = img[clutter]
            peak = float(img[near_boat].max())
            return int((bg > peak).sum()), int(bg.numel())

        srn, spread = contrast(det)
        n_over, n_bg = false_alarms(det)
        raw_srn, raw_spread = contrast(raw_cart)
        look_srn, look_spread = contrast(look_cart)
        mean_srn, mean_spread = contrast(mean_cart)
    print(f"  the boat's echo peaks at ({px:+.1f}, {py:+.1f}) m, boat centred "
          f"on ({tx:+.1f}, {ty:+.1f})")
    # The ghost.  The array's seabed image is 2 * WATER_DEPTH - AUV_DEPTH deep,
    # so a leg via the seabed to the hull is longer than the direct one by a
    # known amount, and the hull is imaged again beyond itself at the same
    # bearing: once bounced at half that extra length in displayed range,
    # twice bounced at the full amount.  Measured off the target channel's
    # range profile along the boat's bearing, so the geometry is checked and
    # not eyeballed.
    with torch.no_grad():
        seabed_image = 2.0 * WATER_DEPTH - AUV_DEPTH
        keel = HULL_DRAUGHT / 2.0
        direct = math.hypot(BOAT_RANGE, AUV_DEPTH - keel)
        via_bed = math.hypot(BOAT_RANGE, seabed_image - keel)
        extra = via_bed - direct
        beam_ix = int((torch.as_tensor(bearings) - BOAT_BEARING_DEG).abs().argmin())
        prof = echo_img[beam_ix, 0].detach()
    print(f"  seabed image {seabed_image:.0f} m deep: a bounced leg is {extra:.2f} m "
          f"longer, so the ghosts sit {extra / 2:.1f} and {extra:.1f} m beyond the hull")
    if prof is not None:
        r = rng.detach()
        hull_far = BOAT_RANGE + along / 2.0
        beyond = (r > hull_far + 0.5) & (r < hull_far + 3.0 * extra)
        if bool(beyond.any()):
            seg = prof.clone(); seg[~beyond] = 0.0
            pk = float(prof[(r - BOAT_RANGE).abs() < along].max())
            tops = []
            for _ in range(2):
                i = int(seg.argmax())
                if float(seg[i]) <= 0.0:
                    break
                tops.append((float(r[i]) - hull_far, 10 * math.log10(float(seg[i]) / pk)))
                seg[(r - r[i]).abs() < 1.0] = 0.0
            print("  target channel beyond the hull's far edge: "
                  + ", ".join(f"{d:+.1f} m at {db:+.1f} dB" for d, db in tops))
    print(f"  {err:.1f} m outside the hull, against {beam_m:.1f} m of beamwidth")
    print(f"  {n_over} of the {n_bg} gain-corrected cells in the swath are "
          f"brighter than it")
    print(f"  ({'a detection' if n_over == 0 else 'NOT a detection'} -- the "
          f"count, not the contrast, is what decides that)")
    print(f"\n  the same arrivals, four displays:")
    print(f"    no gain at all:           boat {raw_srn:+5.1f} dB, "
          f"background spread {raw_spread:.1f} dB")
    print(f"    TVG from the mean:        boat {mean_srn:+5.1f} dB, "
          f"background spread {mean_spread:.1f} dB")
    print(f"    TVG from the median:      boat {srn:+5.1f} dB, "
          f"background spread {spread:.1f} dB")
    print(f"    median TVG + {looks} looks:      boat {look_srn:+5.1f} dB, "
          f"background spread {look_spread:.1f} dB")
    # How much the target lifts the reference it is about to be divided by --
    # measured at the bin its own peak lands in, not quoted from another scene.
    with torch.no_grad():
        bin_of_peak = int((rng - math.hypot(px, py)).abs().argmin())
        off_boat = ((torch.as_tensor(bearings) - BOAT_BEARING_DEG).abs()
                    > 3.0 * beamwidth)
        lift = 10 * math.log10(
            float(noisy[:, 0, bin_of_peak].mean()
                  / noisy[off_boat, 0, bin_of_peak].mean().clamp_min(1e-30)))
    print(f"  The boat lifts the MEAN at the bin its own peak sits in by "
          f"{lift:+.1f} dB, and a")
    print(f"  gain taken from that mean divides the lift straight back out of "
          f"the target")
    print(f"  ({mean_srn - srn:+.1f} dB against the median here).  A MEDIAN "
          f"over {len(bearings)} beams cannot be")
    print(f"  moved by a handful of bright ones, which is why CFAR and AGC "
          f"references")
    print(f"  are order statistics -- and the penalty scales with the target: "
          f"in a")
    print(f"  scene where this boat stood {raw_srn + 8:.0f} dB over its "
          f"background the mean gain cost 4 dB.")
    with torch.no_grad():
        bg = det[ring]
        over = lambda t: 100.0 * float((bg > 10 ** (t / 10.0)
                                        * bg.median()).to(bg.dtype).mean())
    print(f"\n  Flattened, the background is what a THRESHOLD can be set "
          f"against -- and that")
    print(f"  is what makes a picture black with a return on it.  Of the "
          f"background here:")
    for t in (3.0, 6.0, 8.0):
        print(f"    above +{t:.0f} dB over its own median: {over(t):5.2f}% of "
              f"cells survive")
    print(f"  and the boat stands {srn:+.1f} dB, so a +"
          f"{THRESHOLD_DB:.0f} dB floor leaves it on near-black")
    print(f"  at EVERY range.  A fixed window anchored on the image peak can "
          f"only do")
    print(f"  that at one range, which is why the 90 m examples look empty "
          f"and this")
    print(f"  does not: their swath spans {94 / 27:.1f}x in range and this one "
          f"{FAR / 44:.1f}x, so the")
    print(f"  reverberation falls {50 * math.log10(FAR / 44) - 50 * math.log10(94 / 27):.0f} dB "
          f"further across it.")

    print(f"  Multi-look is a cost either way: it smears a target living in "
          f"one range")
    print(f"  bin across {looks}, for {look_srn - srn:+.1f} dB, buying "
          f"{spread - look_spread:.1f} dB of smoothness.  For a target under")
    print(f"  a beamwidth that is the wrong way round, so the default is off.")

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
    up_edge, dn_edge = TILT_DEG - fov / 2, TILT_DEG + fov / 2
    print(f"  {N_TX} transmit elements flood a {fov:.1f} deg vertical field of "
          f"view, tilted")
    print(f"  {abs(TILT_DEG):.1f} deg {'up' if TILT_DEG < 0 else 'down'}, "
          f"carrying beams of {VERTICAL_BEAM_DEG:.2f} deg -- about "
          f"{fov / VERTICAL_BEAM_DEG:.0f} of them.  The lobe runs")
    print(f"  {up_edge:+.1f} to {dn_edge:+.1f} deg, taking down as positive, "
          f"and that decides what")
    print(f"  a ping can contain at all:")
    print(f"\n   range   surface   seabed   in the lobe")
    both_from = None
    for r in _range_marks(6):
        up = -math.degrees(math.atan2(AUV_DEPTH, r))
        dn = math.degrees(math.atan2(WATER_DEPTH - AUV_DEPTH, r))
        s_ok, b_ok = up_edge <= up <= dn_edge, up_edge <= dn <= dn_edge
        if s_ok and b_ok and both_from is None:
            both_from = r
        print(f"   {r:5.0f} m  {up:+6.2f}   {dn:+6.2f}   "
              f"{'surface and seabed' if s_ok and b_ok else ('surface only' if s_ok else 'neither')}")
    enters = ((WATER_DEPTH - AUV_DEPTH)
              / math.tan(math.radians(dn_edge)) if dn_edge > 0
              else float("inf"))
    print(f"\n  The seabed enters where altitude <= "
          f"{math.tan(math.radians(dn_edge)):.3f} x range, which with "
          f"{WATER_DEPTH - AUV_DEPTH:.0f} m of")
    if enters < FAR:
        print(f"  altitude is {enters:.0f} m: the inner part of this swath is "
              f"sea surface alone,")
        print(f"  and the seabed joins it from there out.")
    else:
        print(f"  altitude is {enters:.0f} m -- beyond this {FAR:.0f} m swath "
              f"entirely.  Tilted up,")
        print(f"  a short look sees the sea surface and nothing of the bottom "
              f"at all; the")
        print(f"  seabed is in the lobe's lower skirt, not its main beam.")
    print(f"  The same head at the same attitude over "
          f"{WATER_DEPTH * 2:.0f} m of water would have no")
    print(f"  bottom in the lobe anywhere inside {FAR:.0f} m either.  Tilt, "
          f"altitude and")
    print(f"  field of view between them decide what a ping can contain, and "
          f"none of")
    print(f"  the three is a power setting.")
    print(f"\n  It is also a quiet geometry: the seabed is at "
          f"{math.degrees(math.atan2(WATER_DEPTH - AUV_DEPTH, FAR)):.2f} deg "
          f"of grazing at {FAR:.0f} m and")
    print(f"  the surface at "
          f"{math.degrees(math.atan2(AUV_DEPTH, FAR)):.2f} deg, and Lambert "
          f"scattering goes as sin(theta),")
    print(f"  so both boundaries return far less than they would to a "
          f"downward-looking")
    print(f"  fan.  Whether that leaves the picture reverberation-limited or "
          f"noise-limited")
    print(f"  is measured above, not assumed.")
    print(f"\n  And across the beam: {beamwidth:.2f} deg is {beam_m:.1f} m at "
          f"the boat, so a")
    widths = across / beam_m
    verdict = ("a mark, not a shape" if widths < 1.2
               else f"resolved, {widths:.1f} beams of extent")
    print(f"  {HULL_LENGTH:.0f} m hull lies {across:.1f} m across bearing = "
          f"{widths:.2f} beamwidths -- {verdict}.  It stands at")
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

    # The same ping under the three conventions a multi-beam head might use
    # to put its elevation beams on one screen.  Which one a given display
    # does is not always documented; drawn side by side under one colour
    # scale, an operator can say which is theirs.  The boat's own beam is the
    # one whose tilt is nearest its elevation.
    import matplotlib.pyplot as plt   # _common has already chosen the Agg backend

    if len(tilts) > 1:
      with torch.no_grad():
        conventions = {
            "summed over beams": stack.sum(dim=0),
            "max over beams": stack.max(dim=0).values,
            f"one beam, {-tilts[k_boat]:+.1f} deg (the boat's)": stack[k_boat],
        }
        panels = {}
        for name, img_b in conventions.items():
            sig_b = calibrate(img_b, SOURCE_LEVEL_DB, beam_scale=scale)
            noisy_b = add_receiver_noise(
                sig_b, noise, generator=torch.Generator().manual_seed(SEED + 2))
            shown_b, _ = display(noisy_b, rng, pixel_m=pixel_m)
            cart_b, _, _ = to_cart(shown_b)
            # In decibels over the background at that range, as the main
            # figure's fourth panel is; the first draft plotted linear power
            # against a dB floor and came out black but for the boat's peak.
            panels[name] = 10.0 * torch.log10(cart_b.clamp_min(1e-30))
        # The per-beam stack, so this figure can be redrawn without a run.
        torch.save({"stack": stack.detach().cpu(), "tilts": tilts,
                    "bearings": bearings, "grid": grid.detach().cpu(),
                    "k_boat": k_boat}, FIGURE_DIR / f"21_beams_{FAR:.0f}m.pt")
        top = max(float(c.max()) for c in panels.values())
        fig_c, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
        for ax, (name, cart_b) in zip(axes, panels.items()):
            im = ax.imshow(cart_b.numpy(), origin="lower", cmap="inferno",
                           extent=(float(gx[0]), float(gx[-1]),
                                   float(gy[0]), float(gy[-1])),
                           vmin=6.0, vmax=top, aspect="equal")
            ax.add_patch(plt.Circle((tx, ty), 12.0, fill=False, color="white",
                                    lw=1.2))
            ax.set_title(name, fontsize=11)
            ax.set_xlabel("forward (m)")
        axes[0].set_ylabel("across (m)")
        fig_c.colorbar(im, ax=axes, shrink=0.8,
                       label="dB over the background at that range, floored at +6")
        fig_c.suptitle(f"{len(tilts)} receive elevation beams of {beam_3db_deg(N_RX_ELEV):.2f} deg "
                       f"under one flooded transmit: three ways to put them on one screen",
                       fontsize=12)
        save(fig_c, f"21_display_conventions_{FAR:.0f}m.png")

    save(_plot(det, raw_cart, mean_cart, gx, gy, rng, prof_db.detach(),
               noise_db, tx, ty, crossover, blind, looks),
         f"21_scene_{FAR:.0f}m.png")

    banner("acceptance")
    # Two different questions, and only the second one is "can you see it".
    # The first reads the TARGET CHANNEL ALONE, so its peak is on the boat by
    # construction however faint the boat is -- it checks the geometry of the
    # echo, never its detectability, and for a long time it was the only check
    # here and it passed on a boat that was invisible in the image.
    ok = check(f"the target channel puts its echo on the boat at "
               f"{BOAT_RANGE:.0f} m",
               err < beam_m,
               f"{err:.1f} m outside the hull against {beam_m:.1f} m of beamwidth")
    # The second reads the WHOLE IMAGE: how much of the clutter at the boat's
    # own range is brighter than the boat.  A detection means a threshold set
    # at the target lets almost no background through; 1e-3 over a ring of a
    # few thousand cells is already a handful of false alarms per ping.
    # Zero, not "few".  A threshold set at the target's level should raise
    # nothing else in the picture -- that is what being able to point at it
    # and say "that is the boat" means.  A first attempt allowed one tenth of
    # a percent, which at this swath size is 43 cells, and passed a boat that
    # was one bright speck among twenty-three.  The count is reported either
    # way, so a near miss is legible rather than a bare FAIL.
    ok &= check("the boat stands above every other cell in the picture",
                n_over == 0,
                f"{n_over} of {n_bg} gain-corrected cells in the swath are "
                f"brighter than the boat's peak"
                + (" -- a detection" if n_over == 0 else
                   " -- NOT a detection, whatever the contrast in dB says"))
    # What fraction of the swath is lit is geometry, not a target: with the fan
    # tilted up, the surface only enters at 44 m and the seabed at 191, so a
    # short swath is legitimately part dark.  What has to hold is that the LIT
    # part stands above the ambient.
    lit_frac = float((profile > noise).to(profile.dtype).mean())
    # With a receive stave on the return leg the far swath can legitimately go
    # noise-limited -- the stave rejects the steep multipath that propped up
    # reverberation at long range -- so neither of these asserts the swath is
    # reverberation-limited any more.  What has to hold is that the lit part
    # of the picture is above the noise where the example says it is, and that
    # the crossover is reported rather than assumed.
    ok &= check("everything the fan lights stands above the ambient where it says so",
                lit_frac > 0.25,
                f"{100 * lit_frac:.0f}% of range bins lit and above the noise "
                f"floor, by {margin:.1f} dB at {FAR:.0f} m")
    ok &= check("the example says where reverberation stops being the competition",
                math.isfinite(crossover),
                f"still {margin:.1f} dB above the ambient at {FAR:.0f} m; "
                f"crosses at about {crossover:.0f} m")
    # Absorption dominates at long range and is a minor term at short: 23 dB
    # two-way at 300 m against 6.9 at 90.  Which regime you are in decides
    # whether a lower frequency is worth its wider beams, so the example has
    # to say which, not assert one.
    absorbed = 2 * alpha * FAR / 1000
    if FAR >= 200.0:
        ok &= check("absorption is the dominant loss at this range",
                    absorbed > 15.0,
                    f"{absorbed:.1f} dB two-way at {FAR:.0f} m -- more than "
                    f"the {margin:.1f} dB of margin over the ambient")
    else:
        ok &= check("absorption is a minor term at this range",
                    absorbed < 10.0,
                    f"{absorbed:.1f} dB two-way at {FAR:.0f} m, against "
                    f"{40 * math.log10(FAR):.0f} dB of spreading")
    ok &= check("a median gain never does worse than a mean one",
                srn >= mean_srn - 0.2,
                f"raw {raw_srn:+.1f}, mean {mean_srn:+.1f}, median "
                f"{srn:+.1f} dB; the boat lifts the mean at its own bin by "
                f"{lift:+.1f} dB")
    # What multi-look costs scales with how much SMALLER the target is than the
    # window it is averaged over.  A point target loses several dB; this hull
    # is 72 range cells deep and loses almost nothing.  What is invariant is
    # that it smooths the background and never adds contrast.
    if looks > 1:
        ok &= check("multi-look smooths the background and never adds contrast",
                    look_spread < spread - 0.8 and look_srn <= srn + 0.5,
                    f"{look_srn - srn:+.1f} dB of contrast for "
                    f"{spread - look_spread:.1f} dB of speckle over {looks} "
                    f"looks, on a hull "
                    f"{along / (PULSE_S * C / 2):.0f} range cells deep")
    else:
        ok &= check("no multi-look to do: the display pixel is one range bin",
                    look_srn == srn and look_spread == spread,
                    f"{pixel_m:.2f} m pixel against "
                    f"{float(rng[1] - rng[0]):.2f} m bins -- nothing to average")
    # The echo's width across bearing should be the hull's own across-track
    # extent convolved with the beam, which is the check that the image is
    # showing the target's geometry and not just the array's.
    with torch.no_grad():
        near_echo = torch.hypot(GX - tx, GY - ty) < 60.0
        lit_echo = near_echo & (echo_cart > echo_cart[near_echo].max() * 0.1)
        los = torch.tensor([tx, ty]) / math.hypot(tx, ty)
        perp = torch.stack([-los[1], los[0]])
        off = (GX[lit_echo] - tx) * perp[0] + (GY[lit_echo] - ty) * perp[1]
        measured = float(off.max() - off.min())
    predicted = math.hypot(across, beam_m)
    print(f"\n  the echo spans {measured:.1f} m across bearing at -10 dB; the "
          f"hull's own")
    print(f"  {across:.1f} m convolved with a {beam_m:.1f} m beam predicts "
          f"{predicted:.1f} m")
    ok &= check("the echo's width across bearing is the hull, not just the beam",
                abs(measured - predicted) < 0.6 * predicted,
                f"{measured:.1f} m measured against {predicted:.1f} m predicted "
                f"({across / beam_m:.2f} beamwidths of hull)")
    # Whether both boundaries are in the lobe depends on the swath: at 300 m
    # they are, past 200; in a 90 m swath the seabed never enters at all, which
    # is the same geometry reported honestly rather than a failure.
    seabed_from = ((WATER_DEPTH - AUV_DEPTH)
                   / math.tan(math.radians(TILT_DEG + fov / 2)))
    if seabed_from < FAR:
        ok &= check("the vertical FOV holds both boundaries over the far swath",
                    both_from is not None and both_from < FAR,
                    f"both inside the {fov:.1f} deg lobe from about "
                    f"{both_from:.0f} m")
    else:
        ok &= check("the swath is surface-only, and the example says why",
                    both_from is None,
                    f"the seabed enters the lobe at {seabed_from:.0f} m, past "
                    f"the {FAR:.0f} m swath -- tilted up, a short look sees "
                    f"only the surface")
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
        if ring and blind < FAR:
            ax.plot(blind * np.cos(th), blind * np.sin(th), ":",
                    color="deepskyblue", lw=0.9, alpha=0.6)
        ax.plot([tx], [ty], "o", mfc="none", mec="white", ms=15, mew=1.3)
        ax.annotate("boat, 250 m", (tx, ty), textcoords="offset points",
                    xytext=(14, 9), color="white", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
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
    # The target display: flattened, then floored just above the background.
    ax4 = fig.add_subplot(2, 3, 4)
    d4 = 10 * np.log10(np.maximum(cart.numpy(), 1e-30))
    im4 = ax4.imshow(d4, origin="lower", cmap="inferno", vmin=THRESHOLD_DB,
                     vmax=THRESHOLD_DB + 12.0, extent=extent)
    ax4.plot([tx], [ty], "o", mfc="none", mec="white", ms=15, mew=1.3)
    ax4.annotate("boat, 250 m", (tx, ty), textcoords="offset points",
                 xytext=(14, 9), color="white", fontsize=8)
    ax4.set_aspect("equal")
    ax4.set_xlim(extent[0], extent[1])
    ax4.set_ylim(extent[2], extent[3])
    ax4.set_xlabel("forward (m)")
    ax4.set_ylabel("across (m)")
    ax4.set_title(f"median TVG, floored at +{THRESHOLD_DB:.0f} dB\n"
                  f"(black with a return on it -- at every range)", fontsize=10)
    fig.colorbar(im4, ax=ax4, shrink=0.7, pad=0.02).set_label(
        "dB over the background at that range", fontsize=7)

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
    where = ("inside the swath" if crossover < rng[-1] else "past the swath")
    bx.annotate(f"reaches the noise floor at\nabout {crossover:.0f} m, {where}",
                (0.97, 0.06), xycoords="axes fraction",
                ha="right", fontsize=8, color="0.3")
    bx.set_xlabel("range (m)")
    bx.set_ylabel("dB re 1 uPa$^2$")
    bx.set_title("reverberation-limited to about %.0f m" % crossover
                 if crossover < rng[-1] else
                 "reverberation-limited the whole way out", fontsize=10)
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
