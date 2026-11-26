"""Time spreading of boundary-bounce paths, after SST (APL-UW TR 0702, Sec. 12.4).

A path that reflects from a rough boundary does not arrive as one copy of the pulse. Part of
its energy arrives coherently at the specular delay; the rest arrives over the following
milliseconds with random amplitude and phase that change slowly from ping to ping. Without
this, a bounce path is either fully coherent (flat boundary) or simply weaker (coherent loss
only, `acoustics.surface_coherent_reflection`), and processing gain on bounce paths is too good.

`TimeSpreadProvider` wraps any eigenray provider. For each ray with bounces it

1. splits the total (flat-boundary) intensity into a coherent part, the product of
   exp(-Gamma^2), Gamma = 2 k h sin(grazing), over the bounces, and an incoherent rest;
2. keeps the ray with the coherent amplitude;
3. adds `taps` rays later than the specular delay, whose mean intensities follow an
   exponential intensity impulse response and sum to the incoherent part, with complex
   Gaussian amplitudes;
4. lets those amplitudes change in time with an exponential correlation (what SST's AR(1)
   filter gives), at a rate set by the Doppler spread of the bounce.

What comes from TR 0702 and what does not:

- The structure (1-4), the three boundary parameters (rms height, mean-square slope, rms
  vertical speed) and the decay time are TR 0702's. Its rule of thumb for the decay time is
  "mean-square slope x half the one-way travel time for a bounce in mid-path; mean-square
  slope x twice the travel time of the shorter leg for a bounce near one end". Both are
  the two ends of  T = 2 mss t1 t2 / (t1 + t2)  (t1, t2 the travel times before and after
  the bounce), which is what is used.
- TR 0702 takes the coherent/incoherent split, the shape of the intensity impulse response
  and the frequency spread from APL-UW TR 9407, which was not available. Substituted here:
  the Rayleigh-parameter coherent reflection; an exponential shape; and a Doppler width
  2 f v sin(grazing) / c for a boundary moving vertically at rms speed v (the Doppler of a
  mirror), with v the root-sum-square of the surface's own speed and (speed of the
  reflection point) x sqrt(mss) as TR 0702 states.
- Several bounces on one path: by default the bounces' responses are convolved (4 below),
  which is what a cascade of independent bounces gives. `multi_bounce="sum"` is the earlier
  rule: the mean excess delays add and one exponential with that mean is used. TR 0702 says
  its own multi-bounce shape is wrong ("too high immediately after the pulse and too low
  later"); `sum` is wrong in that way for two rough bounces, and the other way round when
  one bounce is mostly coherent with the longer decay (tools/sst_validation.py, ASIAEX).
- The same realisation is used for scatterers within `coherence_length` of each other, so
  the highlights of one target fade together and distant targets fade independently.

The random amplitudes are a deterministic function of (seed, place, ray, tap, time): no
state is kept, a scenario's seed reproduces a ping, and pings can be simulated in any order.

Against the forward-scatter papers read since (Dahl, JASA 100, 748-758, 1996; 105, 2155-2169, 1999;
115, 589-599, 2004; docs/references/surface_forward_scatter_dahl_1996_2004.md and
surface_bistatic_dahl_1999_mcdaniel_1986.md): the geometry of the facet model below is theirs. Its
across-path angular spread sqrt(2 mss) sin(g) r1 / (r1 + r2) is Dahl 1996 eq. (15) with q = sqrt(2)
against the fitted 1.348, and its coherence lengths scale as Dahl 2004 eqs. (7)-(9). So is the
slope since docs/references/CONFORMANCE.md A4: `wind_speed` gives the mean-square slope of the long
waves, Dahl 1996 eq. (A7), 0.0046 ln(2.1 U^2) (`acoustics.large_scale_mean_square_slope`), which
the papers use. Before A4 it gave the Cox-Munk TOTAL slope, 1.4 to 2.2 times larger at 1.5 to
10 m/s; with it the angular spread was 1.2 to 1.5 times too wide and the facet tail 1.6 to 1.9 times
too long at the papers' conditions. That law stays selectable as `surface.slope_law:
cox_munk_total`. A slope given as `mean_square_slope` replaces either. What still does not fit: the
SPACE'08 tails steeper than 12 degrees decay 1.9 times as slowly as the model's with the
long-wave slope (1.3 times with the total; PHYSICS_NOTES Sec. 21.4 item 5), and Dahl 1999's angle spreads imply a slope half
of (A7)'s at 4 m/s (recorded, not followed). The exact Kirchhoff tail (option 9, `kirchhoff_go`) fits Dahl's two
measured decays better than the facet tail but does not mend SPACE'08: 2.1 times too fast above 12 degrees.

Options beyond SST (all off by default; with them off the output is what it was)
---------------------------------------------------------------------------------
The model behind all of them is the one behind the decay time: a boundary of tilted mirrors
(facets) with Gaussian slopes. Nothing here is taken from a paper; it is derived below and
tested against exact geometry. Notation for one bounce: r1, r2 the path lengths before and
after it, g its grazing angle, q = 1/r1 + 1/r2, and (x, y) the position of a facet on the
boundary relative to the specular point, x along the horizontal direction of travel.

Geometry. To second order the path through the facet at (x, y) is longer than the specular
one by  c tau = (x^2 sin^2 g + y^2) q / 2 : the curves of equal delay are ellipses on the
boundary, a factor 1/sin g longer along the path than across it. Seen along the ray the
ellipse is a circle: with (x sin g, y) = rho (cos psi, sin psi),  rho^2 = 2 c tau / q.
The facet reflects from the source to the receiver only if its slopes are
s_x = x sin(g) q / 2  and  s_y = y q / (2 sin g)  (the mirror law; for a far source they are
"tilt the mirror by s, turn the ray by 2 s" in the plane of incidence and "by 2 s sin g"
across it).

1. Angle spreading (`angle_spread`). The tap of excess delay tau comes from the circle of
   radius rho: it leaves the source at an angle  rho / r1 = sqrt(2 c tau r2 / (r1 (r1 + r2)))
   from the specular ray and reaches the far end at  rho / r2 . Where on the circle (psi) is
   drawn once per tap from the density that Gaussian slopes of variance mss/2 per axis give
   on it,  exp(kappa cos 2 psi),  kappa = (c tau q / 4 mss) (1 / sin^2 g - 1): at low grazing
   angles the late facets lie up and down the path rather than beside it. Summed over the
   tail (mean excess delay T) the mean-square angle at the source is 2 c T r2 / (r1 (r1 + r2));
   with the facet decay time below that is  2 mss (1 + sin^2 g) (r2 / (r1 + r2))^2, of which
   the part in the vertical plane is 2 mss (r2 / (r1 + r2))^2 and the part across it is
   sin^2 g times that. The coherent arrival keeps the specular direction. The directions are
   carried in the eigenrays' `launch` and `arrival`, so both engines use them, and a moving
   sonar then sees each tap at its own Doppler shift (which is what TR 0702's "speed of the
   reflection point x sqrt(mss)" stands for).
   On a path with several rough bounces the direction at the source comes from the first of
   them and that at the far end from the last, each with its expected share of the tap's
   excess delay; the cross terms between bounces are neglected.

2. Decay time and shape from the same facets (`decay_time="facet"`, the default; TR 0702's
   rule is `"rule_of_thumb"`; PHYSICS_NOTES Sec. 21.4 has the measurements). The slopes above put
   tau = 2 (s_x^2 + s_y^2 sin^2 g) / (c q), so the mean excess delay is
   T = mss (1 + sin^2 g) t1 t2 / (t1 + t2): TR 0702's rule of thumb at normal incidence and
   half of it at low grazing angles (there only the slopes along the path delay the sound).
   The shape is that of a sum of two squared Gaussians with means a = mss t1 t2 / (t1 + t2)
   and b = a sin^2 g:  exp(-tau (a + b) / (4 a b)) I0(tau (a - b) / (4 a b)) / (2 sqrt(a b)).
   With this option the angles of (1) are Gaussian in each plane, with the variances given.

3. Frequency spreading within a ping (`doppler`). The process of step 4 above is a sum of
   `doppler_lines` sinusoids in time. By default it is sampled once per ping: the tail fades
   from ping to ping but is frozen during a ping, so a tone comes back as a tone. With
   `doppler` each sinusoid is its own ray with a rate of change of path length
   (`EigenraySet.range_rate`) that gives its frequency, and the same process then runs on
   through the ping. Width: a boundary displaced by zeta(t) changes the path by
   -2 zeta sin g, so the reflected field is exp(i chi zeta), chi = 2 k sin g; for Gaussian
   zeta with rms speed v the second moment of the spectrum of the scattered (non-coherent)
   part is exactly
       sigma_nu^2 = (chi v / 2 pi)^2 / (1 - exp(-Gamma^2)),   chi v / 2 pi = 2 f v sin(g) / c.
   For a rough boundary (Gamma >> 1) the spectrum is Gaussian with that rms width; for a
   nearly flat one it is the wave spectrum itself and the width tends to v / (2 pi h), the
   rms wave frequency. Over several bounces the numerators add and the denominator is one
   minus the product of the coherent factors. `doppler_spectrum="gaussian"` (the default)
   draws the sinusoids' frequencies from a Gaussian of that rms width; "lorentzian" is the
   AR(1) spectrum SST uses (half-width 2 f v sin(g) / c; it has no rms width). Measured
   channels support the Gaussian (PHYSICS_NOTES Sec. 21: 15 of 18 surface arrivals at one
   site against 7 of 18 for the Lorentzian), which is why it is the default. A bottom has
   v = 0: its only spread is that of the sonar's motion. With `angle_spread` the sonar's
   motion is left to the taps' directions and only the boundary's own speed sets the
   frequencies within a ping.

4. Several bounces (`multi_bounce="convolve"`). Each bounce's intensity response is
   c_i delta(tau) + (1 - c_i) h_i(tau)  (c_i its coherent fraction, h_i its tail of mean
   T_i), and a path's response is their convolution. The term with no scattering is the
   coherent arrival, as before; the rest is the tail, of energy 1 - prod c_i (as before) and
   mean  sum (1 - c_i) T_i / (1 - prod c_i)  (before: sum T_i). Two fully rough bounces give
   tau exp(-tau / T): nothing right after the specular delay, most at T, a longer tail. The
   convolution is done on the tap grid.

5. Level fluctuations stronger than Rayleigh (`tap_shape`). The taps above are complex Gaussian:
   the intensity of a fully scattered arrival has a scintillation index of 1 and no more.
   Measured steep surface bounces show 1.1 to 4.8 (tools/channel_validation.py, SPACE'08 at
   60 m, 21 to 32 degrees grazing; docs/PHYSICS_NOTES.md Sec. 21.4), which a few strong
   focusing facets would give. With `tap_shape` = nu the scattered part of each ray is
   multiplied by sqrt(tau), tau a gamma variate of shape nu and mean 1 that is common to the
   ray's taps and changes in time at `texture_rate` times the tail's Doppler width. A tap's
   intensity is then K distributed: mean unchanged, second moment (1 + 1/nu) times the
   Gaussian one, scintillation index 1 + 2/nu (tested against these). `tap_shape` may be a
   table {grazing_deg: [...], shape: [...]} (interpolated in the launch grazing angle; large
   values mean Gaussian). Not given, or infinite: no texture, and the output is bit-identical
   to what it was. The coherent arrival is not touched. With `doppler` the texture is that of
   the ping time and does not change within the ping. This is a description of data, not a
   derivation: the shape is a free parameter.

6. Surface height apart from the wind (`surface.rms_height`, `surface.height_scale`). The
   wind law gives a fully developed sea. A real one need not be: the same measurements want
   1.4 times the height for the vertical speed at 4 m/s and a third of it at 9 m/s. An
   explicit `rms_height` (m) replaces the wind law's (the slope and the vertical speed stay
   the wind's unless given too); `height_scale` multiplies whichever height is in force, for
   when only a ratio is known. Default 1.

7. How far the taps reach (`tail_quantile`). By default the taps span three mean excess delays,
   whatever the shape. The facet shape of (2) is long-tailed at low grazing angles: it falls as
   exp(-tau / 2a) / sqrt(tau), an e-fold of twice its mean, and 8.3 % of its energy lies beyond
   three means (5 % for an exponential). That energy is not lost (the taps are normalised) but
   it is put too early. With `tail_quantile` = q the taps span the delay within which a share q
   of the response's energy arrives: the q-quantile of the gamma distribution with the tail's
   mean and variance (for one facet bounce mean a + b and variance 2 (a^2 + b^2): exact for
   b -> 0, where the shape is a chi-square of one degree of freedom, and for a = b, the
   exponential; in between the covered share is within 0.006 of q, tested). For one rough
   bounce each tap then carries the exact energy of its interval of delay (the shape
   integrated over it, the last one with everything later) at that interval's own mean delay,
   so energy and mean excess delay are the shape's own with no rescaling. Several bounces:
   the same grid rule as before on the longer span. Not given: three means, the earlier taps
   bit for bit. Cost: the taps are that much farther apart, so the same spacing needs
   q-quantile / 3 means times as many (2.2 times at q = 0.99 for a low-angle facet bounce).

8. Slow wander of the coherent arrival (`coherent_phase`; class `CoherentPhase`). Above, the
   coherent part of a surface bounce is perfectly steady. Measured at Gamma^2 = 0.03 it is not:
   its phase wanders slowly (PHYSICS_NOTES Sec. 21.6: variance 0.0084 rad^2, 0.36 Hz rms, a line
   near 0.2 Hz) while the tail behind it fluctuates faster. The long waves of the surface move
   the whole reflecting patch up and down: a change of path length -2 zeta sin g, zeta their
   displacement (towards the water) at the specular point. With `coherent_phase` the surface's
   height is split: a large-scale part of rms height h_L (`rms_height`, or `share` of the
   surface's mean-square height) made of `waves` sinusoids zeta = sum A_n cos(psi_n),
   psi_n = 2 pi f_n t - K_n . x + theta_n, with frequencies from the stated spectrum
   (`period_s`: a line at 1 / period of relative rms width `relative_width`; or
   `spectrum: {frequency_hz, density}`), deep-water wavenumbers K = (2 pi f)^2 / 9.81 and the
   direction of travel `direction_deg` (+- `spread_deg`); and the rest, which scatters as
   before (its coherent factor, its tail and the tail's Doppler width now use the rest's
   height sqrt(h^2 - h_L^2) and the surface's `rms_vertical_speed`, which is thereby the
   tail's own rate). First-order theory of reflection from a slightly sinusoidal mirror (the
   Fresnel integral over the delay ellipses of this docstring's geometry; derived for this
   module, not from a paper) gives each wave's effect on the reflection as the factor
       1 + i z_n cos(psi_n) exp(i Theta_n),   z_n = 2 k A_n sin g,
       Theta_n = rho K_n^2 (cos^2(a_n) / sin^2 g + sin^2(a_n)) / (2 k),   rho = r1 r2 / (r1 + r2),
   a_n the angle between the wave's direction and the path. Theta is the phase by which the
   wave's first diffraction orders lag the specular reflection. Small Theta (the Fresnel zone
   is smaller than the wave): a pure change of path length, which is applied AS ONE: the ray's
   delay moves by -(2 sin g / c) sum A_n cos(Theta_n) cos(psi_n) and it gets the matching rate
   of change of path length, so the phase is proportional to frequency across the band,
   continuous through a ping (linear in time within it) and from ping to ping. The part
   z_n sin(Theta_n) cos(psi_n) is a change of amplitude (focusing by the wave's curvature;
   `focusing: false` leaves it out), applied at the centre frequency with the mean energy kept.
   Waves with Theta_n above `fresnel_limit` (default 1 rad) are left with the scattering part:
   their diffraction orders are separate, later arrivals, which is what the tail is. At long
   range and low grazing angles that is every wave: the Fresnel zone spans many crests, the
   coherent arrival does not wander, and this option changes nothing there. That limit is ours.
   The source's own heave is indistinguishable from this in one hydrophone's data and is
   not modelled here (a moving source or platform is: give it a velocity). `legs: 1` for
   one-way sources (the time within a ping at which the surface is met differs).

9. The exact Kirchhoff (geometric-optics) tail of a surface bounce (`decay_time="kirchhoff_go"`;
   docs/references/surface_forward_scatter_dahl_1996_2004.md WP3; PHYSICS_NOTES Sec. 16.6). The facet
   model of (1)-(2) is the second-order expansion of geometric optics about the specular point. This option
   drops the expansion for surface bounces: in the bounce's own straight-ray frame (source r1 and receiver r2
   from the specular point at grazing g, the other boundaries unfolded as flat mirrors) every surface point
   carries the intensity sigma_geo / (R_i^2 R_s^2) dA re the flat mirror's 1 / (r1 + r2)^2 (Dahl 1996 eq. (4)),
   sigma_geo = sec^4(gb) exp(-tan^2(gb) / mss) / (4 pi mss) with gb the tilt of the facet that reflects the
   point's rays (eqs. (7), (9), (10)), and the delay (R_i + R_s - r1 - r2) / c. Integrated around each curve of
   equal delay (an ellipse on the surface; `kirchhoff_go_density`) this gives the tail's intensity against delay.
   - Shape and mean: tabulated as the quantile function in (grazing, f = r1 / (r1 + r2)), the delay in units of
     r1 + r2, at the boundary's own slope (the slope is not interpolated: each slope's nodes are computed on
     first use, a few ms each; `kirchhoff_go_tail`). Interpolation is linear in grazing and in log f, within 2 %
     of the direct quadrature (tested). Steep or nearly flat: it is the facet tail (mean within 1 % at 45-89
     degrees). At 7-12 degrees with Dahl's slopes the mean is 1.4-1.6 times shorter. Near one end of a long
     low-grazing path (f of a few thousandths, grazing a few degrees) it is LONGER than the facet tail and
     heavy-tailed: surface points far from the specular one, seen at nearly the source's own grazing angle,
     need a tilt of only about g / 2. Shadowing, which would remove most of them, is not modelled.
   - Energy: the incoherent part keeps the flat mirror's energy, as for the other options (Dahl 1996 eq. (13)
     says the GO sum is the flat mirror within 0.25 dB; our quadrature gives -0.35 to -0.62 dB at Dahl's 7-12
     degree geometries, within 0.25 dB from 20 degrees up; `kirchhoff_go_closure`). Only the shape is GO's.
   - Taps: one rough bounce: each tap is an interval of delay, its exact energy at that interval's mean delay
     (the last tap everything later), spanning three GO means or, with `tail_quantile` q, the GO tail's own
     q-quantile. Several rough bounces: as for `facet`, the bounces' responses are convolved on the tap grid
     (`multi_bounce: convolve`; GO bounces enter as their energy in each tap's cell) or their means summed
     into one exponential (`sum`); the cross terms are neglected in both, as before.
   - Directions (`angle_spread`): the tap's surface point is drawn on the exact curve of its delay with the
     curve's own intensity weights (`kirchhoff_go_iso_delay`, `kirchhoff_go_draw`); the tap leaves the source
     towards that point and arrives from it, so launch and arrival are jointly those of one facet (exact in an
     iso-velocity water; with refraction the change from the specular ray is applied to the eigenray's own
     direction). On a path with several rough bounces the first sets the launch, the last the arrival, as in (1).
   - Bottom bounces keep the facet tail: the bottom's slope statistics are not the sea surface's.
   - Not from a paper: the quadrature, the table and the sampler are ours; the physics is Dahl 1996 eqs. (4),
     (7)-(10). Python only, like all of this module: the taps reach both simulator engines as eigenrays.
"""
from __future__ import annotations

import functools

import numpy as np

from .. import acoustics
from .propagation import EigenrayProvider, EigenraySet

_M1, _M2, _M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser: a well-mixed 64-bit hash of each element."""
    with np.errstate(over="ignore"):
        x = (x + _M1).astype(np.uint64)
        x = (x ^ (x >> np.uint64(30))) * _M2
        x = (x ^ (x >> np.uint64(27))) * _M3
        return x ^ (x >> np.uint64(31))


def _uniform(key: np.ndarray, stream: int) -> np.ndarray:
    """Uniform numbers in (0, 1), one per key, independent between streams."""
    with np.errstate(over="ignore"):
        h = _mix(key ^ (np.uint64(stream) * _M2))
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


# How the sea surface's mean-square slope follows from `wind_speed` (`surface.slope_law`):
#   large_scale     (default) the long-wave slope of Dahl 1996 eq. (A7), 0.0046 ln(2.1 U^2), the slope the
#                   forward-scatter papers use (acoustics.large_scale_mean_square_slope; CONFORMANCE.md A4);
#   cox_munk_total  the total Cox-Munk slope 0.003 + 0.00512 U (acoustics.mean_square_slope), the default
#                   until CONFORMANCE.md A4: 1.4 to 2.2 times larger at 1.5 to 10 m/s.
SLOPE_LAWS = ("large_scale", "cox_munk_total")


def surface_slope_from_wind(wind_speed: float, slope_law: str = "large_scale") -> float:
    """The surface's mean-square slope for time spreading from the wind (`SLOPE_LAWS`)."""
    if slope_law == "large_scale":
        u = float(wind_speed)
        # No wind: no rough surface (the height is zero too); the law's hold below 1.5 m/s warns only for a real wind.
        return acoustics.large_scale_mean_square_slope(u, warn=u > 0.0)
    if slope_law == "cox_munk_total":
        return acoustics.mean_square_slope(wind_speed)
    import difflib
    near = difflib.get_close_matches(str(slope_law), SLOPE_LAWS, n=1, cutoff=0.0)
    raise ValueError(f"time_spread.surface.slope_law: unknown {slope_law!r}; known: {', '.join(SLOPE_LAWS)}"
                     + (f" (did you mean {near[0]!r}?)" if near else ""))


def boundary_parameters(cfg: dict | None, *, surface: bool, frequency: float | None = None) -> dict | None:
    """{rms_height, mean_square_slope, rms_vertical_speed} from a config section: given
    directly, or for the sea surface from `wind_speed` (Pierson-Moskowitz height and vertical
    speed; the slope by `slope_law`, by default the long-wave slope of Dahl 1996 eq. (A7), with
    `slope_law: cox_munk_total` the Cox-Munk total used before). A value given explicitly
    (`mean_square_slope`, `rms_height`, `rms_vertical_speed`) replaces the wind law's. None or all
    zero: no spreading at that boundary."""
    if not cfg:
        return None
    out = {"rms_height": 0.0, "mean_square_slope": 0.0, "rms_vertical_speed": 0.0}
    law = str(cfg.get("slope_law", "large_scale"))
    if "slope_law" in cfg and not surface:
        raise ValueError("time_spread: slope_law describes the sea surface's wind law, not the bottom")
    if "wind_speed" in cfg:
        if not surface:
            raise ValueError("time_spread: wind_speed describes the surface, not the bottom")
        u = float(cfg["wind_speed"])
        out = {"rms_height": acoustics.rms_wave_height(u), "mean_square_slope": surface_slope_from_wind(u, law),
               "rms_vertical_speed": acoustics.rms_surface_vertical_speed(u)}
    elif "slope_law" in cfg:
        surface_slope_from_wind(1.5, law)                # an unknown name is refused even without a wind
    for key in out:
        if key in cfg:
            out[key] = float(cfg[key])
    if "height_scale" in cfg:                 # the height relative to the wind law (or to the given one)
        if float(cfg["height_scale"]) < 0.0:
            raise ValueError("time_spread: height_scale must not be negative")
        out["rms_height"] *= float(cfg["height_scale"])
    # Energy absorbed by near-surface bubbles at each bounce: a number in dB, a table
    # {grazing_deg: [...], loss_db: [...]}, or a named wind-speed law {model: ...} of
    # sim/surface.py (BUBBLE_LOSS_MODELS): `apl_1994`, the recommended one (D-123; the APL-UW
    # formula as arlpy and UnderwaterAcoustics.jl implement it, via secondary: TR 9407 itself
    # was not read), or `hall_novarini`, `hall_novarini_layer`, `hall_novarini_screened`
    # (D-68, from open sources).
    loss = cfg.get("bubble_loss_db", 0.0)
    if isinstance(loss, dict) and "model" in loss:
        # {model: apl_1994 | hall_novarini | ..., wind_speed}: a wind-speed law (sim/surface.py).
        # The wind defaults to the section's `wind_speed`; the frequency is the wrapper's.
        from . import surface as surface_model
        if not surface or frequency is None or "wind_speed" not in {**cfg, **loss}:
            raise ValueError("time_spread: a bubble loss model needs the surface section, a wind_speed and the frequency")
        out["bubble_loss"] = surface_model.bubble_loss_table(str(loss["model"]), float(loss.get("wind_speed", cfg.get("wind_speed"))),
                                                             float(loss.get("frequency", frequency)))
    elif isinstance(loss, dict):
        out["bubble_loss"] = (np.radians(np.asarray(loss["grazing_deg"], dtype=float)), np.asarray(loss["loss_db"], dtype=float))
    elif float(loss) != 0.0:
        out["bubble_loss"] = (np.array([0.0, np.pi / 2]), np.array([float(loss), float(loss)]))
    return out if out["rms_height"] > 0.0 or "bubble_loss" in out else None


def _i0e(x) -> np.ndarray:
    """exp(-x) I0(x) for x >= 0, without overflow (large x: the asymptotic series)."""
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, 500.0)
    big = np.maximum(x, 500.0)
    series = (1.0 + 1.0 / (8.0 * big) + 9.0 / (128.0 * big**2)) / np.sqrt(2.0 * np.pi * big)
    return np.where(x < 500.0, np.i0(small) * np.exp(-small), series)


def facet_delay_density(tau, a, b):
    """Density of the excess delay tau of one bounce from Gaussian facets: tau is a sum of two
    squared Gaussians with means a (slopes along the path) and b <= a (across it),
    exp(-tau (a + b) / (4 a b)) I0(tau (a - b) / (4 a b)) / (2 sqrt(a b)). For a = b it is
    the exponential of mean 2 a."""
    tau = np.asarray(tau, dtype=float)
    a = np.maximum(np.asarray(a, dtype=float), 1e-300)
    b = np.maximum(np.asarray(b, dtype=float), 1e-300)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.exp(-tau / (2.0 * a)) * _i0e(tau * (a - b) / (4.0 * a * b)) / (2.0 * np.sqrt(a * b))


def tail_span(mean, variance, quantile: float):
    """Delay within which a share `quantile` of a tail's energy arrives: the quantile of the
    gamma distribution of that mean and variance (shape mean^2 / variance)."""
    from scipy.special import gammaincinv
    mean, variance = np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
    ok = (mean > 0.0) & (variance > 0.0)
    shape = np.where(ok, mean**2 / np.where(ok, variance, 1.0), 1.0)
    return np.where(ok, gammaincinv(shape, float(quantile)) * np.where(ok, variance, 0.0) / np.where(ok, mean, 1.0), 0.0)


_GAUSS = np.polynomial.legendre.leggauss(12)


def facet_intervals(edges, a, b):
    """Energy and mean delay of the facet response in each interval of delay between consecutive
    `edges` [..., J + 1] (the first 0), the last interval taking everything later: (weight
    [..., J] summing to 1, delay [..., J]). Gauss-Legendre in sqrt(tau), which removes the
    shape's 1 / sqrt(tau) at the origin."""
    edges = np.asarray(edges, dtype=float)
    a, b = np.asarray(a, dtype=float)[..., None, None], np.asarray(b, dtype=float)[..., None, None]
    lo, hi = np.sqrt(edges[..., :-1, None]), np.sqrt(edges[..., 1:, None])
    x, w = _GAUSS
    u = 0.5 * (hi + lo) + 0.5 * (hi - lo) * x                                          # [..., J, 12]
    f = facet_delay_density(u * u, a, b) * 2.0 * u * (0.5 * (hi - lo) * w)
    weight, moment = f.sum(axis=-1), (f * u * u).sum(axis=-1)
    mean_all = (a + b)[..., 0, 0]
    # The last interval runs to infinity: what is left of the energy and of the first moment.
    rest = np.maximum(1.0 - weight[..., :-1].sum(axis=-1), 1e-300)
    rest_moment = np.maximum(mean_all - moment[..., :-1].sum(axis=-1), rest * edges[..., -2])
    weight = np.concatenate([weight[..., :-1], rest[..., None]], axis=-1)
    moment = np.concatenate([moment[..., :-1], rest_moment[..., None]], axis=-1)
    return weight, moment / np.maximum(weight, 1e-300)


# -- The exact Kirchhoff (geometric-optics) tail of a surface bounce: `decay_time="kirchhoff_go"` (module docstring, 9) --
#
# One bounce in its own frame, lengths in units of the unfolded path r1 + r2 = 1: source S = (-f cos g, 0, -f sin g),
# receiver R = ((1 - f) cos g, 0, -(1 - f) sin g), the specular point at the origin, the surface z = 0, f = r1 / (r1 + r2).
# A surface point P = (x, y, 0) has the excess path delta = |P - S| + |R - P| - 1 and the intensity, re the flat mirror's
# 1 / (r1 + r2)^2, sigma_geo / (R_i^2 R_s^2) dA (Dahl 1996 eq. (4), B = 1), with sigma_geo = sec^4(gb) exp(-tan^2(gb) / mss)
# / (4 pi mss) (eq. (9)) and tan^2(gb) = (alpha^2 + beta^2) / gamma^2, 2 (alpha, beta, gamma) = k_i - k_s (eqs. (7), (10)).

# The table's nodes (ours): grazing in degrees (denser where the tail changes fastest) and f = r1 / (r1 + r2), the
# shorter leg's share (the delay distribution is symmetric in f <-> 1 - f: sigma_geo and R_i^2 R_s^2 are). The slope is
# not a node: each slope gets its own nodes, computed on first use.
_GO_GRAZING_DEG = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.5, 14.0, 16.0, 18.0,
                            20.0, 23.0, 26.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
_GO_FRACTION = np.array([0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
# Energy levels at which each node stores its delay (the quantile function): dense towards 1 for the long tails.
GO_LEVELS = 1.0 - (1.0 - np.linspace(0.0, 1.0, 257)) ** 3


def _go_rho(delta, grazing, fraction, psi):
    """Where the curve of excess path delta crosses the ray from the specular point at angle psi in the stretched
    coordinates (x sin g, y) = rho (cos psi, sin psi): the positive root of the quadratic that the spheroid with foci S
    and the receiver's image (sum of distances 1 + delta) gives on z = 0. Returns (x, y, wx, wy, rho)."""
    s, cg = np.sin(grazing), np.cos(grazing)
    wx, wy = np.cos(psi) / s, np.sin(psi)
    a = wx * cg                                                          # (wx, wy, 0) . axis; |w|^2 - a^2 = 1
    c0 = 0.5 - fraction                                                  # centre of the spheroid along its axis
    a2 = (0.5 * (1.0 + delta)) ** 2                                      # semi-axes^2: along, across
    b2 = 0.25 * delta * (2.0 + delta)
    q = a * a / a2 + 1.0 / b2
    h = a * c0 / a2
    rho = (h + np.sqrt(h * h + q * (1.0 - c0 * c0 / a2))) / q
    return rho * wx, rho * wy, wx, wy, rho


def _go_curve(delta, grazing, fraction, mss, n_psi: int):
    """Points (x, y) of the curve of excess path delta on the half y >= 0 (midpoints of n_psi equal steps of psi) and
    the intensity each carries per unit of delta, both halves counted: [..., n_psi]."""
    delta = np.maximum(np.asarray(delta, dtype=float), 1e-15)[..., None]
    g = np.asarray(grazing, dtype=float)[..., None]
    f = np.asarray(fraction, dtype=float)[..., None]
    psi = (np.arange(n_psi) + 0.5) * (np.pi / n_psi)
    x, y, wx, wy, rho = _go_rho(delta, g, f, psi)
    s, cg = np.sin(g), np.cos(g)
    vix, viy, viz = x + f * cg, y, f * s                                 # source -> point
    vsx, vsy, vsz = (1.0 - f) * cg - x, -y, -(1.0 - f) * s               # point -> receiver
    ri = np.sqrt(vix * vix + viy * viy + viz * viz)
    rs = np.sqrt(vsx * vsx + vsy * vsy + vsz * vsz)
    qx, qy, qz = vix / ri - vsx / rs, viy / ri - vsy / rs, viz / ri - vsz / rs
    tan2 = (qx * qx + qy * qy) / (qz * qz)                               # Dahl 1996 eq. (10)
    sigma = (1.0 + tan2) ** 2 * np.exp(-tan2 / mss) / (4.0 * np.pi * mss)   # eq. (9)
    dl = (wx * vix + wy * viy) / ri - (wx * vsx + wy * vsy) / rs          # d delta / d rho (> 0)
    # dA = rho / sin g drho dpsi, drho = d delta / dl; twice the half circle.
    weight = sigma / (ri * ri * rs * rs) * rho / (s * dl) * (2.0 * np.pi / n_psi)
    return x, y, weight


def kirchhoff_go_density(delta, grazing, fraction, mss, n_psi: int = 128) -> np.ndarray:
    """Intensity per unit excess path of one surface bounce in geometric optics (Dahl 1996 eqs. (4), (9), (10),
    integrated around each curve of equal delay): delta the excess path in units of r1 + r2, f = r1 / (r1 + r2),
    intensity re the flat mirror. Its integral over delta is `kirchhoff_go_closure` (Dahl 1996 eq. (13): 1)."""
    return _go_curve(delta, grazing, fraction, float(mss), n_psi)[2].sum(axis=-1)


def _go_grid(scale: float) -> np.ndarray:
    """Excess paths (units r1 + r2) for one node: 240 steps even in sqrt(delta) up to 30 facet means `scale`, then
    160 steps even in log(delta) up to an excess path of 2 (r1 + r2), which holds the long tails of low grazing
    angles near one end (surface points far from the specular one, reached at nearly the source's own grazing)."""
    near = (np.linspace(0.0, np.sqrt(30.0), 241) ** 2) * scale
    far = np.geomspace(30.0 * scale, max(2.0, 60.0 * scale), 161)[1:]
    return np.concatenate([near, far])


@functools.lru_cache(maxsize=8192)
def _go_node(mss: float, gi: int, fi: int) -> tuple:
    """(delay at GO_LEVELS, closure) of the table node (grazing _GO_GRAZING_DEG[gi], f _GO_FRACTION[fi]) for this
    slope; delay in units of r1 + r2."""
    g, f = np.radians(_GO_GRAZING_DEG[gi]), float(_GO_FRACTION[fi])
    scale = mss * (1.0 + np.sin(g) ** 2) * f * (1.0 - f)                 # the facet model's mean, a natural unit
    delta = _go_grid(scale)
    p = kirchhoff_go_density(delta, g, f, mss)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(delta))])
    closure = float(cdf[-1])
    cdf = cdf / closure
    keep = np.concatenate([[True], np.diff(cdf) > 0.0])                  # strictly increasing for the inversion
    out = np.interp(GO_LEVELS, cdf[keep], delta[keep])
    out.flags.writeable = False
    return out, closure


def kirchhoff_go_closure(grazing, fraction, mss) -> float:
    """Dahl 1996 eq. (13) for one bounce: the GO intensity summed over the surface, re the flat mirror (1 = 0 dB)."""
    g, f = float(grazing), float(fraction)
    scale = mss * (1.0 + np.sin(g) ** 2) * f * (1.0 - f)
    delta = _go_grid(scale)
    p = kirchhoff_go_density(delta, g, f, mss)
    return float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(delta)))


def kirchhoff_go_tail(grazing, t1, t2, mss) -> dict:
    """The GO tail of surface bounces of grazing angle `grazing` (rad) and travel times t1, t2 before and after it
    (arrays of one shape, s), for one mean-square slope `mss`: {"levels": GO_LEVELS, "delay": [..., L] the excess
    delay (s) within which each level's share of the tail's energy arrives (the quantile function), "mean",
    "variance" (s, s^2), "closure" (Dahl 1996 eq. (13), 1 = the flat mirror)}. Interpolated (linearly in grazing,
    in log f) between nodes computed on first use: within 2 % of the direct quadrature (tested)."""
    mss = float(mss)
    t1, t2 = np.broadcast_arrays(np.asarray(t1, dtype=float), np.asarray(t2, dtype=float))
    g = np.broadcast_to(np.degrees(np.asarray(grazing, dtype=float)), t1.shape)
    total = t1 + t2
    f = np.clip(np.minimum(t1, t2) / np.maximum(total, 1e-300), _GO_FRACTION[0], 0.5)
    gd = np.clip(g, _GO_GRAZING_DEG[0], 90.0)
    i = np.clip(np.searchsorted(_GO_GRAZING_DEG, gd, side="right") - 1, 0, len(_GO_GRAZING_DEG) - 2)
    wg = (gd - _GO_GRAZING_DEG[i]) / (_GO_GRAZING_DEG[i + 1] - _GO_GRAZING_DEG[i])
    lf, nodes_f = np.log(f), np.log(_GO_FRACTION)
    k = np.clip(np.searchsorted(nodes_f, lf, side="right") - 1, 0, len(nodes_f) - 2)
    wf = (lf - nodes_f[k]) / (nodes_f[k + 1] - nodes_f[k])
    delay = np.zeros(t1.shape + (len(GO_LEVELS),))
    closure = np.zeros(t1.shape)
    for di, dk, w in ((0, 0, (1 - wg) * (1 - wf)), (1, 0, wg * (1 - wf)), (0, 1, (1 - wg) * wf), (1, 1, wg * wf)):
        gi, fi = i + di, k + dk
        pairs = np.unique(np.stack([gi.ravel(), fi.ravel()], axis=1), axis=0) if gi.size else np.zeros((0, 2), dtype=int)
        table = {(int(a), int(b)): _go_node(mss, int(a), int(b)) for a, b in pairs}
        for (a, b), (quantile, c) in table.items():
            sel = (gi == a) & (fi == b)
            delay[sel] += w[sel][:, None] * quantile
            closure[sel] += w[sel] * c
    delay *= total[..., None]
    dp = np.diff(GO_LEVELS)
    lo, hi = delay[..., :-1], delay[..., 1:]
    mean = np.sum(dp * 0.5 * (lo + hi), axis=-1)                         # the quantile function is linear between levels
    second = np.sum(dp * (lo * lo + lo * hi + hi * hi) / 3.0, axis=-1)
    return {"levels": GO_LEVELS, "delay": delay, "mean": mean, "variance": np.maximum(second - mean * mean, 0.0), "closure": closure}


def _go_moments_at(tau, tail):
    """Share of the tail's energy that arrives by delay tau [..., M] and its first moment (s), from the quantile
    function `tail["delay"]` [..., L] (linear between levels)."""
    d = np.asarray(tail["delay"], dtype=float)
    tau = np.asarray(tau, dtype=float)
    levels = tail["levels"]
    dp = np.diff(levels)
    cum = np.concatenate([np.zeros(d.shape[:-1] + (1,)), np.cumsum(dp * 0.5 * (d[..., :-1] + d[..., 1:]), axis=-1)], axis=-1)
    n = d.shape[-1]
    idx = np.clip((d[..., None, :] <= tau[..., :, None]).sum(axis=-1), 1, n - 1)        # tau in [d[idx - 1], d[idx])
    take = lambda a, j: np.take_along_axis(np.broadcast_to(a, d.shape), j, axis=-1)     # noqa: E731
    d0, d1 = take(d, idx - 1), take(d, idx)
    p0, p1 = levels[idx - 1], levels[idx]
    frac = np.clip((tau - d0) / np.where(d1 > d0, d1 - d0, 1.0), 0.0, 1.0)
    share = p0 + frac * (p1 - p0)
    x = d0 + frac * (d1 - d0)
    moment = take(cum, idx - 1) + (share - p0) * 0.5 * (d0 + x)
    beyond = tau >= d[..., -1:]
    return np.where(beyond, 1.0, share), np.where(beyond, cum[..., -1:], moment)


def kirchhoff_go_cdf(tau, tail) -> np.ndarray:
    """Share of the GO tail's energy arriving within excess delay tau (s) [..., M] of `kirchhoff_go_tail`'s [...]."""
    return _go_moments_at(tau, tail)[0]


def kirchhoff_go_intervals(edges, tail):
    """Energy and mean delay of the GO tail in each interval between consecutive `edges` [..., J + 1] (the first 0),
    the last interval taking everything later: (weight [..., J] summing to 1, delay [..., J]), exact for the tabulated
    quantile function (as `facet_intervals` is for the facet shape)."""
    edges = np.asarray(edges, dtype=float)
    share, moment = _go_moments_at(edges[..., :-1], tail)
    mean = np.asarray(tail["mean"], dtype=float)[..., None]
    share = np.concatenate([share, np.ones(share.shape[:-1] + (1,))], axis=-1)
    moment = np.concatenate([moment, np.broadcast_to(mean, moment.shape[:-1] + (1,))], axis=-1)
    weight, first = np.diff(share, axis=-1), np.diff(moment, axis=-1)
    middle = 0.5 * (edges[..., :-1] + edges[..., 1:])
    middle[..., -1] = edges[..., -2]
    return weight, np.where(weight > 1e-300, first / np.maximum(weight, 1e-300), middle)


def kirchhoff_go_iso_delay(delta, grazing, fraction, mss, n_psi: int = 64):
    """Where on the surface the intensity of excess path delta comes from, for the angle sampler: points [..., 2 n_psi, 2]
    (x along the path from the specular point, y across; units r1 + r2) and their shares [..., 2 n_psi] (sum 1), both
    halves y >= 0 and y < 0 (module docstring, 9)."""
    x, y, w = _go_curve(delta, grazing, fraction, float(mss), n_psi)
    w = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-300)
    pts = np.concatenate([np.stack([x, y], axis=-1), np.stack([x, -y], axis=-1)], axis=-2)
    return pts, 0.5 * np.concatenate([w, w], axis=-1)


def kirchhoff_go_draw(delta, grazing, fraction, mss, u, v, n_psi: int = 64):
    """One surface point per element of delta [...] drawn from `kirchhoff_go_iso_delay`'s shares by the uniform numbers
    u (where on the half curve: the distribution over the psi steps inverted, uniform within a step) and v (< 0.5: y < 0).
    The point is put exactly on the curve of excess path delta. Returns (x, y), units r1 + r2."""
    delta = np.maximum(np.asarray(delta, dtype=float), 1e-15)
    g = np.broadcast_to(np.asarray(grazing, dtype=float), delta.shape)
    f = np.broadcast_to(np.asarray(fraction, dtype=float), delta.shape)
    _, _, w = _go_curve(delta, g, f, float(mss), n_psi)
    cdf = np.cumsum(w, axis=-1)
    target = np.asarray(u, dtype=float)[..., None] * cdf[..., -1:]
    k = np.clip((cdf < target).sum(axis=-1), 0, n_psi - 1)
    hi = np.take_along_axis(cdf, k[..., None], axis=-1)[..., 0]
    lo = hi - np.take_along_axis(w, k[..., None], axis=-1)[..., 0]
    within = np.clip((target[..., 0] - lo) / np.maximum(hi - lo, 1e-300), 0.0, 1.0)
    psi = (k + within) * (np.pi / n_psi)
    x, y, _, _, _ = _go_rho(delta, g, f, psi)
    return x, np.where(np.asarray(v) < 0.5, -y, y)


def _go_full(slot, values, shape) -> np.ndarray:
    """A bounce slot's GO values (one per ray that has the GO tail) spread onto the [S, K] grid, zero elsewhere."""
    out = np.zeros(int(np.prod(shape)))
    out[slot["go"]["index"]] = values
    return out.reshape(shape)


def _go_quantile(tail, q: float) -> np.ndarray:
    """The delay [...] within which a share q of a GO tail's energy arrives (linear between levels)."""
    j = int(np.clip(np.searchsorted(GO_LEVELS, q, side="right"), 1, len(GO_LEVELS) - 1))
    w = (q - GO_LEVELS[j - 1]) / (GO_LEVELS[j] - GO_LEVELS[j - 1])
    return tail["delay"][..., j - 1] + w * (tail["delay"][..., j] - tail["delay"][..., j - 1])


# `decay_time`: the shape and mean of a rough bounce's tail. `facet` (default, D-85; module docstring 2), TR 0702's
# `rule_of_thumb`, or `kirchhoff_go`: surface bounces take the exact geometric-optics tail of Dahl 1996 (9), the bottom
# keeps the facet one.
DECAY_TIMES = ("facet", "rule_of_thumb", "kirchhoff_go")


def iso_delay_angle(kappa, u, v, n_grid: int = 129):
    """Where on its circle of equal delay a tap's facet lies: the angle psi (0: along the path,
    towards the far end; pi/2: across it) drawn from the density exp(kappa cos 2 psi) by
    inverting its distribution at the uniform numbers u; v (uniform) picks the quadrant."""
    kappa = np.asarray(kappa, dtype=float)
    shape = kappa.shape
    kappa = kappa.reshape(-1)
    u, v = np.asarray(u, dtype=float).reshape(-1), np.asarray(v, dtype=float).reshape(-1)
    psi0 = np.empty_like(kappa)
    grid = np.linspace(0.0, 0.5 * np.pi, n_grid)
    sin2 = np.sin(grid) ** 2
    step = 32768                                                       # rows at a time: bounded memory
    for lo in range(0, kappa.size, step):
        k = np.clip(kappa[lo:lo + step, None], 0.0, 1e6)
        pdf = np.exp(-2.0 * k * sin2)                                  # exp(kappa (cos 2 psi - 1))
        cdf = np.concatenate([np.zeros((k.shape[0], 1)), np.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]), axis=1)], axis=1)
        target = u[lo:lo + step] * cdf[:, -1]
        i = np.clip((cdf < target[:, None]).sum(axis=1), 1, n_grid - 1)
        rows = np.arange(k.shape[0])
        c0, c1 = cdf[rows, i - 1], cdf[rows, i]
        psi0[lo:lo + step] = grid[i - 1] + (grid[i] - grid[i - 1]) * (target - c0) / np.maximum(c1 - c0, 1e-300)
    quadrant = np.minimum((4.0 * v).astype(int), 3)
    psi = np.where(quadrant == 0, psi0, np.where(quadrant == 1, np.pi - psi0, np.where(quadrant == 2, np.pi + psi0, -psi0)))
    return psi.reshape(shape)


def _turn(direction, angle, psi, sign: float):
    """Unit vectors `direction` [..., 3] turned by `angle` (rad) towards the facet at angle psi
    on the circle of equal delay. sign +1: the direction leaving the source; -1: the direction
    of travel on arrival. A facet farther along the path is seen at a flatter angle from the
    source and at a steeper one from the far end."""
    hn = np.hypot(direction[..., 0], direction[..., 1])
    safe = np.maximum(hn, 1e-12)
    hx = np.where(hn > 1e-9, direction[..., 0] / safe, 1.0)
    hy = np.where(hn > 1e-9, direction[..., 1] / safe, 0.0)
    dz = direction[..., 2]
    across = np.stack([-hy, hx, np.zeros_like(hx)], axis=-1)           # horizontal, normal to the ray
    in_plane = np.stack([-dz * hx, -dz * hy, hn], axis=-1)             # normal to the ray, in its vertical plane
    steeper = np.where(dz >= 0.0, 1.0, -1.0)
    # The second-order geometry holds for turns well inside the grazing angle. Beyond it a
    # ray would be turned through the horizontal and miss the boundary: the turn towards
    # the horizontal stops at 0.9 of the ray's own elevation.
    flatter = np.minimum(sign * angle * np.cos(psi), 0.9 * np.arcsin(np.clip(np.abs(dz), 0.0, 1.0)))
    out = direction + (sign * angle * np.sin(psi))[..., None] * across - (flatter * steeper)[..., None] * in_plane
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


class CoherentPhase:
    """The large-scale part of the surface and what it does to a coherent reflection (module
    docstring, 8). cfg: {rms_height | share, period_s | spectrum, direction_deg} and optional
    relative_width (0.1), spread_deg (0), waves (16), fresnel_limit (1.0), focusing (true),
    span_s (0: the pulse length, so that the phase is right at mid-pulse), legs (2: echoes; 1:
    one-way sources), seed."""

    GRAVITY = 9.81
    _KEYS = {"rms_height", "share", "period_s", "spectrum", "direction_deg", "relative_width", "spread_deg", "waves", "fresnel_limit",
             "focusing", "span_s", "legs", "seed"}

    def __init__(self, cfg: dict, *, frequency: float, seed: int = 0):
        unknown = set(cfg) - self._KEYS
        if unknown:
            raise ValueError(f"time_spread: coherent_phase: unknown keys {sorted(unknown)}")
        if ("rms_height" in cfg) == ("share" in cfg):
            raise ValueError("time_spread: coherent_phase needs rms_height (m) or share (of the surface's mean-square height), one of them")
        if ("period_s" in cfg) == ("spectrum" in cfg):
            raise ValueError("time_spread: coherent_phase needs period_s or spectrum: {frequency_hz, density}, one of them")
        if "direction_deg" not in cfg:
            raise ValueError("time_spread: coherent_phase needs direction_deg (where the long waves travel to, from +x towards +y)")
        from scipy.special import ndtri
        self.frequency = float(frequency)
        self.rms_height = None if "rms_height" not in cfg else float(cfg["rms_height"])
        self.share = None if "share" not in cfg else float(cfg["share"])
        if self.share is not None and not 0.0 <= self.share <= 1.0:
            raise ValueError("time_spread: coherent_phase.share is between 0 and 1")
        n = int(cfg.get("waves", 16))
        self.fresnel_limit = float(cfg.get("fresnel_limit", 1.0))
        self.focusing = bool(cfg.get("focusing", True))
        self.span, self.legs = float(cfg.get("span_s", 0.0)), int(cfg.get("legs", 2))
        if self.legs not in (1, 2):
            raise ValueError("time_spread: coherent_phase.legs is 2 (echoes) or 1 (one-way sources)")
        with np.errstate(over="ignore"):
            key = _mix(np.uint64(int(cfg.get("seed", seed))) * _M3 ^ np.uint64(0xC0FFEE) ^ (np.arange(n, dtype=np.uint64) + np.uint64(1)) * _M1)
        q = (np.arange(n) + 0.5) / n
        if "period_s" in cfg:
            f0 = 1.0 / float(cfg["period_s"])
            f = f0 * np.maximum(1.0 + float(cfg.get("relative_width", 0.1)) * ndtri(q), 0.05)
        else:
            sp = cfg["spectrum"]
            fx, d = np.asarray(sp["frequency_hz"], dtype=float), np.maximum(np.asarray(sp["density"], dtype=float), 0.0)
            fine = np.linspace(fx[0], fx[-1], 4096)
            cdf = np.cumsum(np.interp(fine, fx, d))
            f = np.interp(q, cdf / cdf[-1], fine)
        self.wave_frequency = f                                                        # Hz, [N]
        self.wavenumber = (2.0 * np.pi * f) ** 2 / self.GRAVITY                        # rad/m, deep water
        heading = np.radians(float(cfg["direction_deg"])) + np.radians(float(cfg.get("spread_deg", 0.0))) * ndtri(np.clip(_uniform(key, 21), 1e-9, 1 - 1e-9))
        self.direction = np.stack([np.cos(heading), np.sin(heading)], axis=1)          # [N, 2]
        self.theta = 2.0 * np.pi * _uniform(key, 22)
        self.weight = np.full(n, 1.0 / n)                                              # share of the large-scale mean-square height per wave

    def height2(self, surface_rms_height: float) -> float:
        """Mean-square height of the large-scale part, m^2 (never more than the surface's)."""
        h2 = self.share * surface_rms_height**2 if self.rms_height is None else self.rms_height**2
        if h2 > surface_rms_height**2 * (1.0 + 1e-9):
            raise ValueError("time_spread: coherent_phase.rms_height is larger than the surface's rms_height")
        return float(min(h2, surface_rms_height**2))

    def modulation(self, t, xy, path_direction, sin_g, rho, sound_speed: float, surface_rms_height: float) -> dict:
        """What the long waves do to a coherent reflection at time t (s) whose specular point is at
        xy [..., 2], on a path of horizontal direction path_direction [..., 2] (unit), grazing
        angle arcsin(sin_g), rho = r1 r2 / (r1 + r2) (m). Returns the change of delay `delay` (s)
        and its rate `delay_rate` (s/s), the relative change of amplitude `amplitude` (zero
        mean) with its variance `amplitude_variance`, and `height2`, the mean-square height of
        the waves that took part (m^2); all of t's shape."""
        t, sin_g, rho = np.asarray(t, dtype=float)[..., None], np.asarray(sin_g, dtype=float)[..., None], np.asarray(rho, dtype=float)[..., None]
        k = 2.0 * np.pi * self.frequency / float(sound_speed)
        amp = np.sqrt(2.0 * self.weight * self.height2(surface_rms_height))             # [N]
        cos2 = np.clip((np.asarray(path_direction, dtype=float)[..., None, :] * self.direction).sum(axis=-1) ** 2, 0.0, 1.0)     # [..., N]
        fresnel = rho * self.wavenumber**2 * (cos2 / np.maximum(sin_g, 1e-6) ** 2 + (1.0 - cos2)) / (2.0 * k)
        keep = fresnel < self.fresnel_limit
        psi = 2.0 * np.pi * self.wave_frequency * t - (np.asarray(xy, dtype=float)[..., None, :] * (self.wavenumber[:, None] * self.direction)).sum(axis=-1) + self.theta
        part = np.where(keep, amp * np.cos(fresnel), 0.0)
        z = 2.0 * k * sin_g * np.where(keep, amp * np.sin(fresnel), 0.0) * (1.0 if self.focusing else 0.0)
        return {"delay": -(2.0 * sin_g[..., 0] / sound_speed) * (part * np.cos(psi)).sum(axis=-1),
                "delay_rate": (2.0 * sin_g[..., 0] / sound_speed) * (part * 2.0 * np.pi * self.wave_frequency * np.sin(psi)).sum(axis=-1),
                "amplitude": -(z * np.cos(psi)).sum(axis=-1), "amplitude_variance": 0.5 * (z**2).sum(axis=-1),
                "height2": np.where(keep, 0.5 * amp**2, 0.0).sum(axis=-1)}


class TimeSpreadProvider(EigenrayProvider):
    """An eigenray provider with time-spread bounce paths (see the module docstring).

    inner: the provider of flat-boundary ("total") eigenrays.
    surface, bottom: boundary parameters (`boundary_parameters`), or None.
    surface_z, bottom_z: boundary depths, m, for the position of each bounce along its path.
    frequency: Hz, for the coherent split and the Doppler width.
    taps: incoherent arrivals added per bounce ray. spacing: s between them (None: so that
    they span three decay times, but never closer than `min_spacing`).
    angle_spread: each tap leaves and arrives in its own direction (module docstring, 1).
    decay_time: "facet" (default; 2), "rule_of_thumb" (TR 0702's) or "kirchhoff_go" (surface bounces take
    the exact geometric-optics tail and directions, 9; the bottom keeps the facet tail).
    doppler: the tail varies during a ping; each tap is then `doppler_lines` rays, each with a
    `range_rate` (3). doppler_spectrum: "gaussian" (default) or "lorentzian".
    multi_bounce: "convolve" (default) or "sum" (4).
    """

    def __init__(self, inner: EigenrayProvider, *, frequency: float, surface: dict | None = None,
                 bottom: dict | None = None, surface_z: float = 0.0, bottom_z: float | None = None,
                 taps: int = 6, min_spacing: float = 0.0, coherence_length: float = 10.0, seed: int = 0,
                 surface_sound_speed: float | None = None, bottom_sound_speed: float | None = None,
                 incoherent: bool = True, angle_spread: bool = False, decay_time: str = "facet",
                 doppler: bool = False, doppler_spectrum: str = "gaussian", doppler_lines: int = 8,
                 multi_bounce: str = "convolve", tap_shape=None, texture_rate: float = 0.25,
                 tail_quantile: float | None = None, coherent_phase: dict | None = None):
        # How far the taps reach (module docstring, 7): None = three mean excess delays, as always.
        self.tail_quantile = None if tail_quantile is None else float(tail_quantile)
        if self.tail_quantile is not None and not 0.5 <= self.tail_quantile < 1.0:
            raise ValueError("time_spread: tail_quantile is a share of the energy, 0.5 to below 1")
        self.coherent_phase = None if not coherent_phase else CoherentPhase(coherent_phase, frequency=float(frequency), seed=int(seed))
        # K-type level fluctuation of the scattered part (module docstring, 5): None = Gaussian taps.
        self.tap_shape = None
        if isinstance(tap_shape, dict):
            self.tap_shape = (np.radians(np.asarray(tap_shape["grazing_deg"], dtype=float)), np.asarray(tap_shape["shape"], dtype=float))
        elif tap_shape is not None and np.isfinite(float(tap_shape)):
            self.tap_shape = (np.array([0.0, np.pi / 2]), np.array([float(tap_shape), float(tap_shape)]))
        if self.tap_shape is not None and np.any(self.tap_shape[1] <= 0.0):
            raise ValueError("time_spread: tap_shape must be positive")
        self.texture_rate = float(texture_rate)
        self.angle_spread, self.doppler = bool(angle_spread), bool(doppler)
        self.decay_time, self.doppler_spectrum, self.multi_bounce = str(decay_time), str(doppler_spectrum), str(multi_bounce)
        self.doppler_lines = int(doppler_lines)
        if self.decay_time not in DECAY_TIMES:
            import difflib
            near = difflib.get_close_matches(self.decay_time, DECAY_TIMES, n=1, cutoff=0.0)
            raise ValueError(f"time_spread: decay_time is {', '.join(repr(d) for d in DECAY_TIMES)}; got {self.decay_time!r}"
                             + (f" (did you mean {near[0]!r}?)" if near else ""))
        if self.doppler_spectrum not in ("lorentzian", "gaussian"):
            raise ValueError("time_spread: doppler_spectrum is 'lorentzian' or 'gaussian'")
        if self.multi_bounce not in ("sum", "convolve"):
            raise ValueError("time_spread: multi_bounce is 'sum' or 'convolve'")
        if self.doppler_lines < 1:
            raise ValueError("time_spread: doppler_lines must be at least 1")
        self.inner = inner
        self.frequency = float(frequency)
        self.surface, self.bottom = surface, bottom
        self.surface_z, self.bottom_z = float(surface_z), (None if bottom_z is None else float(bottom_z))
        self.taps = int(taps)
        self.min_spacing = float(min_spacing)
        self.coherence_length = float(coherence_length)
        self.seed = int(seed)
        self.c_surface, self.c_bottom = surface_sound_speed, bottom_sound_speed
        self.incoherent = bool(incoherent)   # False: coherent loss only (the scattered part is dropped)
        self.time = 0.0                # s: when the current ping is
        self.platform_speed = 0.0      # m/s: horizontal speed of the sonar
        if self.taps < 1:
            raise ValueError("time_spread: taps must be at least 1")

    # Reverberation is already incoherent and uses mean intensities: it takes the total paths.
    @property
    def reverberation_provider(self) -> EigenrayProvider:
        return self.inner

    def set_ping(self, time: float, platform_speed: float = 0.0) -> None:
        self.time, self.platform_speed = float(time), float(platform_speed)

    # -- geometry of the bounces ---------------------------------------------------------------

    def _bounce_fractions(self, source_z, point_z, up_first, n_surface, n_bottom):
        """Fraction of the path (0..1) at which each bounce happens and which boundary it is,
        from unfolding the path vertically: [..., n_max] fractions (NaN where there is no such
        bounce) and a matching array that is True for surface bounces. Straight-ray geometry;
        with refraction it is the same sequence of boundaries with approximate positions."""
        n = (n_surface + n_bottom).astype(int)
        n_max = int(n.max()) if n.size else 0
        frac = np.full(n.shape + (n_max,), np.nan)
        is_surface = np.zeros(n.shape + (n_max,), dtype=bool)
        if n_max == 0:
            return frac, is_surface
        top = self.surface_z
        bot = self.bottom_z if self.bottom_z is not None else np.inf
        depth = bot - top
        # With one kind of boundary only, every bounce is from it, once.
        both = (n_surface > 0) & (n_bottom > 0) | (n > 1)
        first_up = np.where(n_bottom == 0, True, np.where(n_surface == 0, False, up_first))
        d_first = np.where(first_up, source_z - top, bot - source_z)
        last_up = first_up ^ ((n - 1) % 2 == 1)                     # boundary of the last bounce
        d_last = np.where(last_up, point_z - top, bot - point_z)
        d_first = np.abs(np.where(np.isfinite(d_first), d_first, 0.0))
        d_last = np.abs(np.where(np.isfinite(d_last), d_last, 0.0))
        between = np.where(both & np.isfinite(depth), depth, 0.0)
        total = d_first + d_last + np.maximum(n - 1, 0) * between
        total = np.where(total > 0.0, total, 1.0)
        for i in range(n_max):
            here = n > i
            frac[..., i] = np.where(here, (d_first + i * between) / total, np.nan)
            is_surface[..., i] = np.where(here, first_up ^ (i % 2 == 1), False)
        return np.clip(frac, 1e-6, 1.0 - 1e-6), is_surface

    # -- the provider ----------------------------------------------------------------------------

    def eigenrays(self, source, points) -> EigenraySet:
        base = self.inner.eigenrays(source, points)
        if self.surface is None and self.bottom is None:
            return base
        source = np.asarray(source, dtype=float).reshape(3)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        s, k = base.delay.shape
        c = float(base.sound_speed)
        bounces = base.valid & ((base.n_surface + base.n_bottom) > 0)
        if not bounces.any():
            return base
        frac, is_surface = self._bounce_fractions(source[2], points[:, 2][:, None], base.launch[..., 2] < 0.0,
                                                  base.n_surface, base.n_bottom)
        cos_launch = np.clip(np.hypot(base.launch[..., 0], base.launch[..., 1]), 0.0, 1.0)
        coherent2 = np.ones((s, k))            # product of exp(-Gamma^2) over the bounces
        absorbed = np.ones((s, k))             # intensity factor left after absorption (bubbles) at the bounces
        mean_excess = np.zeros((s, k))         # s: mean delay of the incoherent part after the specular one
        speed2 = np.zeros((s, k))              # (m/s)^2 x sin^2(grazing), summed over bounces
        own2 = np.zeros((s, k))                # the same without the sonar's motion: the boundary's own speed
        # The facet shape for every boundary, except that `kirchhoff_go` gives surface bounces the exact GO tail (9).
        facet = self.decay_time in ("facet", "kirchhoff_go")
        go_surface = self.decay_time == "kirchhoff_go"
        slots = []                             # the rough bounces of each ray, in path order
        wander = {"delay": np.zeros((s, k)), "rate": np.zeros((s, k)), "gain": np.ones((s, k))}    # of the coherent arrival (coherent_phase)
        for i in range(frac.shape[-1]):
            for surface, par, c_b in ((True, self.surface, self.c_surface), (False, self.bottom, self.c_bottom)):
                if par is None:
                    continue
                here = np.isfinite(frac[..., i]) & (is_surface[..., i] == surface) & bounces
                if not here.any():
                    continue
                # Grazing angle at the boundary from the launch angle by Snell's law.
                cos_g = np.clip(cos_launch * ((c_b or c) / c), 0.0, 1.0)
                grazing = np.arccos(cos_g)
                height = par["rms_height"]
                if surface and self.coherent_phase is not None:
                    # The long waves heave the coherent reflection (module docstring, 8); the rest of the height scatters.
                    horizontal = points[:, None, :2] - source[None, None, :2]
                    span = np.linalg.norm(horizontal, axis=-1, keepdims=True)
                    f_here = np.where(here, frac[..., i], 0.5)
                    when = self.time + self.coherent_phase.span / 2.0 + (base.delay if self.coherent_phase.legs == 2 else 0.0)
                    mod = self.coherent_phase.modulation(when, source[None, None, :2] + f_here[..., None] * horizontal,
                                                         horizontal / np.maximum(span, 1e-9), np.sin(grazing),
                                                         (c_b or c) * f_here * (1.0 - f_here) * base.delay, c_b or c, par["rms_height"])
                    wander["delay"] += np.where(here, mod["delay"], 0.0)
                    wander["rate"] += np.where(here, mod["delay_rate"], 0.0)
                    wander["gain"] *= np.where(here, (1.0 + mod["amplitude"]) / np.sqrt(1.0 + mod["amplitude_variance"]), 1.0)
                    height = np.sqrt(np.maximum(par["rms_height"] ** 2 - mod["height2"], 0.0))
                if np.ndim(height) == 0:
                    r = acoustics.rough_coherent_reflection(grazing, height, self.frequency, c_b or c)
                else:                                   # the same factor, exp(-Gamma^2 / 2), for a height that differs from ray to ray
                    r = np.exp(-0.5 * (4.0 * np.pi * self.frequency / (c_b or c) * height * np.sin(grazing)) ** 2)
                coherent2 = np.where(here, coherent2 * r**2, coherent2)
                if "bubble_loss" in par:
                    loss_db = np.interp(grazing, *par["bubble_loss"])
                    absorbed = np.where(here, absorbed * 10.0 ** (-loss_db / 10.0), absorbed)
                f = np.where(here, frac[..., i], 0.5)
                t1, t2 = f * base.delay, (1.0 - f) * base.delay
                # Mean excess delays from the slopes along the path and across it (facet model).
                along = par["mean_square_slope"] * t1 * t2 / np.maximum(t1 + t2, 1e-12)
                across = along * np.maximum(np.sin(grazing), 1e-3) ** 2
                if facet:
                    mean_i = np.where(here, along + across, 0.0)
                else:
                    mean_i = np.where(here, 2.0 * par["mean_square_slope"] * t1 * t2 / np.maximum(t1 + t2, 1e-12), 0.0)
                go = None
                if go_surface and surface and par["mean_square_slope"] > 0.0:
                    # The exact GO tail of this bounce (9), for the rays that have it: its mean replaces the facet's.
                    where = np.flatnonzero(here.ravel())
                    go = {"index": where, "grazing": grazing.ravel()[where], "frac": f.ravel()[where], "delay": base.delay.ravel()[where],
                          "mss": float(par["mean_square_slope"]),
                          "tail": kirchhoff_go_tail(grazing.ravel()[where], t1.ravel()[where], t2.ravel()[where], par["mean_square_slope"])}
                    mean_i = mean_i.copy()
                    mean_i.ravel()[where] = go["tail"]["mean"]
                mean_excess += mean_i
                v2 = par["rms_vertical_speed"] ** 2 + self.platform_speed**2 * par["mean_square_slope"]
                speed2 += np.where(here, v2 * np.sin(grazing) ** 2, 0.0)
                own2 += np.where(here, par["rms_vertical_speed"] ** 2 * np.sin(grazing) ** 2, 0.0)
                slots.append({"here": here, "mean": mean_i, "coherent": np.where(here, r**2, 1.0), "frac": f,
                              "along": np.where(here, along, 0.0), "across": np.where(here, across, 0.0), "go": go})
        total = np.where(bounces, base.amplitude * np.sqrt(absorbed), base.amplitude)   # what the boundary does not absorb
        incoherent = np.where(bounces & self.incoherent, np.abs(total) ** 2 * (1.0 - coherent2), 0.0)
        n_rough = np.zeros((s, k), dtype=int)
        for slot in slots:
            n_rough += slot["here"]
        many = (n_rough >= 2) & (self.multi_bounce == "convolve")
        if many.any():
            # Convolved bounces: a bounce delays the tail only when it scatters.
            scattered = sum((1.0 - slot["coherent"]) * slot["mean"] for slot in slots)
            mean_excess = np.where(many, scattered / np.maximum(1.0 - coherent2, 1e-300), mean_excess)
        # Taps: equally spaced from one spacing after the specular delay, exponential mean intensity.
        j = np.arange(1, self.taps + 1)
        if self.tail_quantile is None:
            spacing = np.maximum(3.0 * mean_excess / self.taps, self.min_spacing)         # [S, K]
        else:
            # The span that holds that share of the energy (module docstring, 7): all bounces scattering.
            m_all = sum(slot["mean"] for slot in slots)
            v_all = sum((2.0 * (slot["along"] ** 2 + slot["across"] ** 2)) if facet else slot["mean"] ** 2 for slot in slots)
            for slot in slots:
                if slot["go"] is not None and len(slot["go"]["index"]):
                    # Rays with a GO bounce: its own variance in place of the facet's, and for one rough bounce the
                    # exact quantile in place of the gamma approximation.
                    on = _go_full(slot, np.ones(len(slot["go"]["index"])), (s, k)) > 0.0
                    v_all = np.where(on, v_all - 2.0 * (slot["along"] ** 2 + slot["across"] ** 2)
                                     + _go_full(slot, slot["go"]["tail"]["variance"], (s, k)), v_all)
            if self.multi_bounce == "sum":
                v_all = np.where(n_rough >= 2, m_all**2, v_all)
            span = tail_span(m_all, v_all, self.tail_quantile)
            for slot in slots:
                if slot["go"] is not None and len(slot["go"]["index"]):
                    one_go = _go_full(slot, (n_rough.ravel()[slot["go"]["index"]] == 1).astype(float), (s, k)) > 0.0
                    span = np.where(one_go, _go_full(slot, _go_quantile(slot["go"]["tail"], self.tail_quantile), (s, k)), span)
            spacing = np.maximum(span / self.taps, self.min_spacing)
        tau = spacing[..., None] * j                                                       # [S, K, J]
        decay = np.maximum(mean_excess, 1e-12)[..., None]
        weight = np.exp(-tau / decay)
        if facet:
            # One rough bounce: the facets' own shape (several, unless convolved: the exponential).
            single = (n_rough == 1) & (mean_excess > 0.0)
            a_sum, b_sum = sum(slot["along"] for slot in slots), sum(slot["across"] for slot in slots)
            weight = np.where(single[..., None], facet_delay_density(tau, a_sum[..., None], b_sum[..., None]), weight)
        if many.any():
            weight = np.where(many[..., None], self._convolved(slots, spacing, facet), weight)
        weight = np.where(mean_excess[..., None] > 0.0, weight, 1.0)                       # no slope given: no delay spread
        weight = weight / weight.sum(axis=-1, keepdims=True)
        # A handful of taps cannot follow the exponential exactly: scale their delays so that the
        # mean excess delay of the intensity is the model's (unless the spacing floor forbids it).
        mean_tau = (weight * tau).sum(axis=-1, keepdims=True)
        scale = np.where(spacing[..., None] > self.min_spacing, decay / np.maximum(mean_tau, 1e-30), 1.0)
        tau = np.where(mean_excess[..., None] > 0.0, tau * scale, 0.0)
        go_one = np.zeros((s, k), dtype=bool)
        for slot in slots:
            if slot["go"] is not None and len(slot["go"]["index"]):
                # One rough bounce with the GO tail: each tap is its interval of delay, exactly, with or without
                # tail_quantile (9); the intervals start at the specular delay and the last takes everything later.
                rows = slot["go"]["index"][(n_rough.ravel()[slot["go"]["index"]] == 1) & (slot["go"]["tail"]["mean"] > 0.0)]
                if len(rows):
                    pick = np.isin(slot["go"]["index"], rows)
                    sub = {key: value[pick] for key, value in slot["go"]["tail"].items() if key != "levels"}
                    sub["levels"] = GO_LEVELS
                    edges = spacing.ravel()[rows][:, None] * np.arange(0, self.taps + 1)
                    w_go, t_go = kirchhoff_go_intervals(edges, sub)
                    weight = weight.reshape(s * k, -1).copy()
                    tau = tau.reshape(s * k, -1).copy()
                    weight[rows], tau[rows] = w_go, t_go
                    weight, tau = weight.reshape(s, k, -1), tau.reshape(s, k, -1)
                    go_one.ravel()[rows] = True
        if self.tail_quantile is not None:
            # One rough bounce: each tap is its interval of delay, exactly (the exponential is the facet shape with a = b).
            one = (n_rough == 1) & (mean_excess > 0.0) & (spacing > self.min_spacing) & ~go_one
            if one.any():
                a_one = sum(slot["along"] for slot in slots) if facet else 0.5 * mean_excess
                b_one = sum(slot["across"] for slot in slots) if facet else 0.5 * mean_excess
                edges = np.where(one, spacing, 1.0)[..., None] * np.arange(0, self.taps + 1)
                w_exact, t_exact = facet_intervals(edges, np.where(one, a_one, 1.0), np.where(one, b_one, 1.0))
                weight = np.where(one[..., None], w_exact, weight)
                tau = np.where(one[..., None], t_exact, tau)
        gaussian = self.doppler_spectrum == "gaussian"
        # The rms width of the scattered part's spectrum, as a speed (module docstring, 3).
        speed_sin = np.sqrt(speed2 / np.maximum(1.0 - coherent2, 1e-12)) if gaussian else np.sqrt(speed2)
        g, nu, phase, tap_key = self._process(source, points, base, speed_sin)             # g: [S, K, J] complex, unit variance
        tap_amp = np.sqrt(incoherent[..., None] * weight) * g
        texture = None
        if self.tap_shape is not None:
            width = 2.0 * self.frequency * speed_sin / c
            texture = np.sqrt(self._texture(source, points, base, np.arccos(cos_launch), width))     # [S, K]
            tap_amp = tap_amp * texture[..., None]
        rep = lambda a: np.repeat(a[:, :, None], self.taps, axis=2)                        # noqa: E731
        tap_delay = base.delay[..., None] + tau
        tap_launch, tap_arrival = rep(base.launch), rep(base.arrival)
        tap_valid = rep(bounces) & (incoherent[..., None] > 0.0)
        if self.angle_spread:
            tap_launch, tap_arrival = self._directions(base, slots, tau, tap_key, tap_valid)
        tap_ns, tap_nb = rep(base.n_surface), rep(base.n_bottom)
        range_rate = None
        if self.doppler:
            # Each sinusoid of the process is a ray of its own with the path-length rate of its frequency.
            m = self.doppler_lines
            line = lambda a: np.repeat(a[:, :, :, None], m, axis=3)                        # noqa: E731
            tap_amp = np.sqrt(incoherent[..., None] * weight)[..., None] * np.exp(1j * phase) / np.sqrt(m)
            if texture is not None:
                tap_amp = tap_amp * texture[..., None, None]
            # With angle spreading the sonar's motion acts through each tap's direction instead.
            own = np.sqrt(own2 / np.where(speed2 > 0.0, speed2, 1.0)) if self.angle_spread else np.ones((s, k))
            tap_rate = -c * nu * own[..., None, None] / self.frequency
            tap_delay, tap_launch, tap_arrival = line(tap_delay), line(tap_launch), line(tap_arrival)
            tap_valid, tap_ns, tap_nb = line(tap_valid), line(tap_ns), line(tap_nb)
            range_rate = np.concatenate([np.zeros((s, k)), tap_rate.reshape(s, -1)], axis=1)

        def widen(a, extra):
            return np.concatenate([a, extra.reshape(s, -1, *a.shape[2:])], axis=1)

        coherent_delay, coherent_gain = base.delay, 1.0
        if self.coherent_phase is not None:
            # The delay moves with the surface; within the ping it goes on at its rate. The kernel takes the rate from the
            # time the sound meets the scatterer (echoes) or leaves the source (one-way): the delay is set so that it is
            # right at mid-pulse.
            met = self.coherent_phase.span / 2.0 + (base.delay if self.coherent_phase.legs == 2 else 0.0)
            coherent_delay = base.delay + np.where(bounces, wander["delay"] - wander["rate"] * met, 0.0)
            coherent_gain = np.where(bounces, wander["gain"], 1.0)
            rate = np.where(bounces, c * wander["rate"], 0.0)
            n_extra = tap_delay.reshape(s, -1).shape[1]
            range_rate = np.concatenate([rate, np.zeros((s, n_extra))], axis=1) if range_rate is None else range_rate + np.concatenate([rate, np.zeros((s, n_extra))], axis=1)

        return EigenraySet(
            delay=widen(coherent_delay, tap_delay),
            amplitude=widen(np.where(bounces, total * np.sqrt(coherent2), base.amplitude) * coherent_gain, tap_amp),
            launch=widen(base.launch, tap_launch), arrival=widen(base.arrival, tap_arrival),
            n_surface=widen(base.n_surface, tap_ns), n_bottom=widen(base.n_bottom, tap_nb),
            valid=widen(base.valid, tap_valid), sound_speed=c, range_rate=range_rate)

    def _convolved(self, slots, spacing, facet: bool) -> np.ndarray:
        """Mean tap intensities [S, K, J] (not normalised) of the convolution of the bounces'
        responses c_i delta + (1 - c_i) h_i on the tap grid, without its coherent term."""
        jn = self.taps
        tau = spacing[..., None] * np.arange(1, jn + 1)
        # Index m of a sequence: m spacings of delay. One bounce's tail is sampled at the taps
        # (as for a single bounce) and stored one index down, so that the tail of a path that
        # scattered once sits on the taps and each further scattering adds its delay less one
        # spacing: two equal exponentials then give tau exp(-tau / T) at the taps exactly.
        res = np.zeros(spacing.shape + (jn,))
        res[..., 0] = 1.0
        coherent = np.ones(spacing.shape)
        first = np.zeros(jn, dtype=bool)
        first[0] = True
        for slot in slots:
            here, c_i = slot["here"], slot["coherent"]
            spread = here & (slot["mean"] > 0.0) & (spacing > 0.0)
            if facet:
                shape = facet_delay_density(tau, slot["along"][..., None], slot["across"][..., None])
                if slot["go"] is not None and len(slot["go"]["index"]):
                    # The GO tail (9): its energy in the cell of each tap, (j - 1/2 .. j + 1/2) spacings.
                    idx = slot["go"]["index"]
                    cells = spacing.ravel()[idx][:, None] * (np.arange(0, jn + 1) + 0.5)
                    share = kirchhoff_go_cdf(cells, slot["go"]["tail"])
                    shape = shape.reshape(-1, jn).copy()
                    shape[idx] = np.diff(share, axis=-1)
                    shape = shape.reshape(spacing.shape + (jn,))
            else:
                with np.errstate(under="ignore"):
                    shape = np.exp(-tau / np.maximum(slot["mean"], 1e-300)[..., None])
            shape = np.where(spread[..., None], shape, 0.0)
            norm = shape.sum(axis=-1, keepdims=True)
            # A tail shorter than the grid, or a bounce with no slope given: no delay of its own.
            shape = np.where(norm > 0.0, shape / np.maximum(norm, 1e-300), first)
            seq = np.where(here[..., None], (1.0 - c_i)[..., None] * shape, 0.0)
            seq[..., 0] += np.where(here, c_i, 1.0)
            new = np.zeros_like(res)
            for a in range(jn):
                new[..., a:] += res[..., :jn - a] * seq[..., a:a + 1]
            res = new
            coherent = coherent * np.where(here, c_i, 1.0)
        res[..., 0] -= coherent                                                           # the path that never scattered
        return np.maximum(res, 0.0)

    def _directions(self, base: EigenraySet, slots, tau, tap_key, tap_valid):
        """Launch and arrival directions [S, K, J, 3] of the taps (module docstring, 1)."""
        s, k = base.delay.shape
        convolve = self.multi_bounce == "convolve"
        zero = np.zeros((s, k))
        first = {"set": np.zeros((s, k), dtype=bool), "frac": zero + 0.5, "part": zero, "kappa": zero, "go": None}
        last = {"frac": zero + 0.5, "part": zero, "kappa": zero, "go": None}
        parts = [slot["mean"] * ((1.0 - slot["coherent"]) if convolve else 1.0) for slot in slots]
        whole = sum(parts)
        for slot, part in zip(slots, parts):
            rough = slot["here"] & (slot["mean"] > 0.0)
            share = part / np.where(whole > 0.0, whole, 1.0)
            # kappa / tau = (1/b - 1/a) / 4 with a, b the mean delays from the two slopes.
            a, b = np.where(rough, slot["along"], 1.0), np.where(rough, slot["across"], 1.0)
            kappa = 0.25 * (1.0 / b - 1.0 / a)
            take = rough & ~first["set"]
            for name, value in (("frac", slot["frac"]), ("part", share), ("kappa", kappa)):
                first[name] = np.where(take, value, first[name])
                last[name] = np.where(rough, value, last[name])
            if slot["go"] is not None and len(slot["go"]["index"]):
                # Which end's bounce has the GO tail (9), with that bounce's grazing angle and slope.
                go_here = _go_full(slot, np.ones(len(slot["go"]["index"])), (s, k)) > 0.0
                g_here = _go_full(slot, slot["go"]["grazing"], (s, k))
                for end, sel in ((first, take), (last, rough)):
                    old = end["go"] or {"on": np.zeros((s, k), dtype=bool), "grazing": zero, "mss": zero}
                    end["go"] = {"on": np.where(sel, go_here, old["on"]), "grazing": np.where(sel & go_here, g_here, old["grazing"]),
                                 "mss": np.where(sel & go_here, slot["go"]["mss"], old["mss"])}
            elif first["go"] is not None or last["go"] is not None:
                for end, sel in ((first, take), (last, rough)):
                    if end["go"] is not None:
                        end["go"]["on"] = np.where(sel, False, end["go"]["on"])
            first["set"] |= rough
        u, v = _uniform(tap_key, 4), _uniform(tap_key, 5)
        delay = np.maximum(base.delay, 1e-12)[..., None]
        out = []
        for end, sign, direction in ((first, 1.0, base.launch), (last, -1.0, base.arrival)):
            f = end["frac"][..., None]
            near = f if sign > 0 else 1.0 - f                               # fraction of the path between this end and the bounce
            t = np.where(tap_valid, tau * end["part"][..., None], 0.0)
            angle = np.sqrt(2.0 * t * (1.0 - near) / (near * delay))
            psi = iso_delay_angle(end["kappa"][..., None] * t, u, v)
            turned = _turn(np.repeat(direction[:, :, None], self.taps, axis=2), angle, psi, sign)
            if end["go"] is not None and end["go"]["on"].any():
                turned = self._go_directions(end, direction, t, delay, u, v, sign, tap_valid, turned)
            out.append(turned)
        return out[0], out[1]

    def _go_directions(self, end, direction, t, delay, u, v, sign: float, tap_valid, turned):
        """Directions of the taps whose bounce at this end has the GO tail (module docstring, 9): a surface point is
        drawn on the exact curve of the tap's excess delay (its share at this bounce), and the tap leaves the source
        towards it (sign +1) or arrives from it (sign -1). The change of elevation and azimuth from the specular ray
        in the bounce's own straight-ray frame is applied to the eigenray's direction."""
        rows = end["go"]["on"][..., None] & tap_valid & (t > 0.0)
        if not rows.any():
            return turned
        shape = t.shape
        f = np.broadcast_to(end["frac"][..., None], shape)[rows]
        g = np.broadcast_to(end["go"]["grazing"][..., None], shape)[rows]
        mss = np.broadcast_to(end["go"]["mss"][..., None], shape)[rows]
        delta = (t / delay)[rows]
        x, y = np.zeros(len(delta)), np.zeros(len(delta))
        for m in np.unique(mss):                                         # one slope per boundary
            sel = mss == m
            x[sel], y[sel] = kirchhoff_go_draw(delta[sel], g[sel], f[sel], m, u[rows][sel], v[rows][sel])
        cg, sg = np.cos(g), np.sin(g)
        if sign > 0:                                                     # source -> point
            vx, vy, vz = x + f * cg, y, f * sg
        else:                                                            # point -> receiver (direction of travel)
            vx, vy, vz = (1.0 - f) * cg - x, -y, (1.0 - f) * sg
        d_elevation = np.arctan2(vz, np.hypot(vx, vy)) - g
        d_azimuth = np.arctan2(vy, vx)
        d = np.repeat(direction[:, :, None], self.taps, axis=2)[rows]
        heading = np.arctan2(d[:, 1], d[:, 0]) + d_azimuth
        elevation = np.clip(np.arcsin(np.clip(np.abs(d[:, 2]), 0.0, 1.0)) + d_elevation, 0.0, 0.5 * np.pi)
        up = np.where(d[:, 2] >= 0.0, 1.0, -1.0)
        out = turned.copy()
        out[rows] = np.stack([np.cos(elevation) * np.cos(heading), np.cos(elevation) * np.sin(heading), up * np.sin(elevation)], axis=-1)
        return out

    def _texture(self, source, points, base: EigenraySet, grazing: np.ndarray, width: np.ndarray) -> np.ndarray:
        """Gamma-distributed intensity factor [S, K] of mean 1 and shape `tap_shape` per ray at the
        current ping time (module docstring, 5): a Gaussian process z(t) = sum_m (a_m cos + b_m sin)
        (2 pi f_m t) / sqrt(M), a, b standard normal and f_m normal of rms `texture_rate` x width,
        all hashed from (seed, place, ray), mapped through its distribution function onto the
        gamma quantiles. Exactly gamma over the ensemble of rays at any time."""
        from scipy.special import gammaincinv, ndtr
        n_terms = 4
        cell = np.floor(points / max(self.coherence_length, 1e-6)).astype(np.int64)
        src = np.floor(source / max(self.coherence_length, 1e-6)).astype(np.int64)
        with np.errstate(over="ignore"):
            key = np.uint64(self.seed) * _M3 ^ np.uint64(0x7E57A11CE)
            for axis in range(3):
                key = _mix(key ^ cell[:, axis].astype(np.uint64) * _M1 ^ np.uint64(src[axis]) * _M2)
            ray = (base.n_surface.astype(np.uint64) * np.uint64(64) + base.n_bottom.astype(np.uint64)) * np.uint64(2) \
                + (base.launch[..., 2] < 0.0).astype(np.uint64)
            key = _mix(key[:, None] ^ ray * _M2)                                           # [S, K]
            key = _mix(key[..., None] ^ (np.arange(n_terms, dtype=np.uint64) + np.uint64(1)) * _M1)   # [S, K, M]
        normal = lambda s1, s2: np.sqrt(-2.0 * np.log(_uniform(key, s1))) * np.cos(2.0 * np.pi * _uniform(key, s2))  # noqa: E731
        a, b, f = normal(11, 12), normal(13, 14), normal(15, 16) * self.texture_rate * width[..., None]
        arg = 2.0 * np.pi * f * self.time
        z = (a * np.cos(arg) + b * np.sin(arg)).sum(axis=-1) / np.sqrt(n_terms)
        shape = np.interp(grazing, *self.tap_shape)
        u = np.clip(ndtr(z), 1e-12, 1.0 - 1e-12)
        return gammaincinv(shape, u) / shape

    def _process(self, source, points, base: EigenraySet, speed_sin: np.ndarray):
        """Unit-variance complex amplitudes [S, K, J] at the current ping time, and their parts:
        each is a sum of sinusoids in time of frequencies nu [S, K, J, M] (Hz) and phases
        `phase` (rad, at the ping time); also the taps' hash keys. Lorentzian: Cauchy-distributed
        frequencies of scale B, so the autocorrelation is exp(-2 pi B |t|), the statistics of
        white noise through an AR(1) filter. Gaussian: frequencies of rms B, autocorrelation
        exp(-(2 pi B t)^2 / 2). B = 2 f speed_sin / c."""
        s, k = base.delay.shape
        n_sin = self.doppler_lines
        cell = np.floor(points / max(self.coherence_length, 1e-6)).astype(np.int64)
        src = np.floor(source / max(self.coherence_length, 1e-6)).astype(np.int64)
        with np.errstate(over="ignore"):
            key = np.uint64(self.seed) * _M3
            for axis in range(3):
                key = _mix(key ^ cell[:, axis].astype(np.uint64) * _M1 ^ np.uint64(src[axis]) * _M2)
            # A ray is known by its bounce counts and the sign of its launch: stable from ping to ping.
            ray = (base.n_surface.astype(np.uint64) * np.uint64(64) + base.n_bottom.astype(np.uint64)) * np.uint64(2) \
                + (base.launch[..., 2] < 0.0).astype(np.uint64)
            key = _mix(key[:, None] ^ ray * _M2)                                           # [S, K]
            tap_key = _mix(key[..., None] ^ (np.arange(self.taps, dtype=np.uint64) + np.uint64(1)) * _M3)     # [S, K, J]
            key = _mix(tap_key[..., None] ^ (np.arange(n_sin, dtype=np.uint64) + np.uint64(1)) * _M1)     # [S, K, J, M]
        width = 2.0 * self.frequency * speed_sin / float(base.sound_speed)                 # Hz, [S, K]
        if self.doppler_spectrum == "gaussian":
            unit = np.sqrt(-2.0 * np.log(_uniform(key, 1))) * np.cos(2.0 * np.pi * _uniform(key, 3))   # Box-Muller
            nu = width[..., None, None] * unit
        else:
            nu = width[..., None, None] * np.tan(np.pi * (_uniform(key, 1) - 0.5))
        phase = 2.0 * np.pi * (_uniform(key, 2) + nu * self.time)
        return np.exp(1j * phase).sum(axis=-1) / np.sqrt(n_sin), nu, phase, tap_key


def wrap(provider: EigenrayProvider, cfg: dict, *, frequency: float) -> EigenrayProvider:
    """Wrap a provider according to a scenario's `time_spread` section:
    {surface: {wind_speed | rms_height, mean_square_slope, rms_vertical_speed}, bottom: {...},
     taps, min_spacing, coherence_length, seed, incoherent,
     angle_spread, decay_time, doppler, doppler_spectrum, doppler_lines, multi_bounce, tap_shape, texture_rate,
     tail_quantile, coherent_phase: {...}};
    the surface section also takes `rms_height` beside `wind_speed`, `height_scale`, and `slope_law`
    (`large_scale`, the default, or `cox_munk_total`: `SLOPE_LAWS`). `decay_time`: `DECAY_TIMES`."""
    if getattr(provider, "general", False):
        raise ValueError("time_spread: not with an image_source provider that has bottom_slope or a {point, normal} wall: the "
                         "spreading places each bounce on a flat surface and bottom")
    surface = boundary_parameters(cfg.get("surface"), surface=True, frequency=float(cfg.get("frequency", frequency)))
    bottom = boundary_parameters(cfg.get("bottom"), surface=False)
    surface_z = getattr(provider, "surface_z", 0.0)
    bottom_z = getattr(provider, "bottom_z", None)
    c_surface = c_bottom = None
    env = getattr(provider, "env", None)
    bed = getattr(provider, "bed", None)
    if getattr(provider, "tier", "") == "eigenray3d" and bed is not None and bed.flat_depth is None and cfg.get("bottom"):
        # The 3-D tier over relief (opt-in; docs/tasks/PROPAGATION_3D.md P6): the bottom's spreading below takes each
        # bounce's grazing angle from the ray's launch angle, as over a level bed; a bounce on a sloping facet grazes
        # it at another angle. Not corrected (the arrivals do not say where a ray met the bed): said.
        import warnings
        warnings.warn("time_spread.bottom with fidelity eigenray3d over relief: bottom-bounce grazing angles are taken as over a level "
                      "bed (from the launch angle); on sloping facets they differ", RuntimeWarning, stacklevel=2)
    if env is not None:                       # a ray-traced ocean: its depth and the speeds at its boundaries
        bottom_z = float(env.depth)
        surface_z = 0.0
        c_surface, c_bottom = float(env.speed_at(0.0)), float(env.speed_at(env.depth))
        if getattr(provider, "wind_speed", 0.0) > 0.0:
            # The provider was taking the coherent loss itself; the wrapper does the split instead.
            if surface is None:
                surface = boundary_parameters({"wind_speed": provider.wind_speed}, surface=True)
            provider.wind_speed = 0.0
    return TimeSpreadProvider(provider, frequency=float(cfg.get("frequency", frequency)), surface=surface, bottom=bottom,
                              surface_z=0.0 if surface_z is None else surface_z, bottom_z=bottom_z,
                              taps=int(cfg.get("taps", 6)), min_spacing=float(cfg.get("min_spacing", 0.0)),
                              coherence_length=float(cfg.get("coherence_length", 10.0)), seed=int(cfg.get("seed", 0)),
                              surface_sound_speed=c_surface, bottom_sound_speed=c_bottom,
                              incoherent=bool(cfg.get("incoherent", True)),
                              angle_spread=bool(cfg.get("angle_spread", False)),
                              decay_time=str(cfg.get("decay_time", "facet")),
                              doppler=bool(cfg.get("doppler", False)),
                              doppler_spectrum=str(cfg.get("doppler_spectrum", "gaussian")),
                              doppler_lines=int(cfg.get("doppler_lines", 8)),
                              multi_bounce=str(cfg.get("multi_bounce", "convolve")),
                              tap_shape=cfg.get("tap_shape"), texture_rate=float(cfg.get("texture_rate", 0.25)),
                              tail_quantile=cfg.get("tail_quantile"), coherent_phase=cfg.get("coherent_phase"))
