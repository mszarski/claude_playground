# hydropt

A differentiable 3-D underwater acoustic ray tracer in PyTorch.

`hydropt` traces geometric-acoustic rays through a 3-D ocean -- a sound-speed
field, a sea surface, a seabed -- and renders a time-resolved energy response at
each receiver. Every step of that forward model is differentiable, so the same
code that predicts a measurement can be run backwards to recover the ocean that
produced it: seabed reflection loss, a sound-speed profile, a bathymetry field,
or the position of the source.

It also runs **active sonar**: two-way propagation through a scattering target,
and coherent beamforming on a real array -- see
[Active sonar and beamforming](#active-sonar-and-beamforming).

It takes its overall shape from TU Berlin's *misuka* (Finnendahl et al.,
"Differentiable Geometric Acoustic Path Tracing using Time-Resolved Path Replay
Backpropagation", ACM TOG 2025) -- with ocean physics in place of room
acoustics. See [Relation to misuka](#relation-to-misuka).

> **Note on the 2-D prototype.** This package was specified as a 3-D,
> autograd-native reinterpretation of a `underwater_acoustic_simulator.jsx`
> prototype, but that file is not in this repository, so the physics here is
> written from the standard references (Jensen et al.; Munk; Thorp) rather than
> ported from it. If the prototype makes different modelling choices -- a
> different Munk parameterisation, a different bottom-loss convention, a
> different spreading law -- those differences will show up as disagreement in
> absolute level, and the places to look are `fields.py`, `boundaries.py` and
> the spreading term in `receiver.py`.

```python
import torch
from hydropt import (ConstantLoss, FlatHeight, MunkProfile, Scene,
                     make_time_grid, octave_bands, spherical_fan,
                     vertical_line_array)

scene = Scene(
    field=MunkProfile(c1=1500.0, z1=1300.0, B=1300.0, eps=7.37e-3),
    bottom=FlatHeight(5000.0),
    source=(0.0, 0.0, 1000.0),
    receivers=vertical_line_array(x=50_000.0, y=0.0, z_top=600.0, z_bottom=2000.0, n=6),
    surface_loss=ConstantLoss(0.5),
    bottom_loss=ConstantLoss(6.0),
    freqs_khz=octave_bands(0.5, 4),
    step_size=20.0, n_steps=3000,
)

directions = spherical_fan(100, 20, elev_range_deg=(-20, 20), azim_range_deg=(-10, 10))
etc = scene.render(directions, make_time_grid(32.0, 36.0, 900),
                   sigma_d=120.0, sigma_t=4e-3)   # [receivers, bands, time bins]

etc.sum().backward()                 # gradients land on every scene parameter
print(scene.bottom_loss.loss_db.grad)
```

## Install

```bash
pip install -e .           # torch >= 2.2, numpy, matplotlib
pip install -e '.[dev]'    # + pytest
pip install -e '.[plotly]' # + interactive 3-D ray plots
pytest                     # 131 tests
```

## Coordinates and units

`x`, `y` horizontal (m); `z` **depth**, positive *downward*, zero at the mean sea
surface. Times in s, sound speed in m/s, losses in dB, frequencies in kHz.
Launch elevation is measured from horizontal and is positive downward, so a
direction is `(cos e cos a, cos e sin a, sin e)`.

## The model

### Ray equations

Rays are integrated in arclength `s` with fixed-step RK4 on the eikonal system
(Jensen et al., *Computational Ocean Acoustics*, sec. 3.2):

```
dx/ds = c xi        d(xi)/ds   = -(1/c^2) dc/dx        dtau/ds = 1/c
dy/ds = c eta       d(eta)/ds  = -(1/c^2) dc/dy
dz/ds = c zeta      d(zeta)/ds = -(1/c^2) dc/dz
```

The slowness vector satisfies `|(xi, eta, zeta)| = 1/c`. That constraint is
conserved by the exact flow but drifts under any finite-order integrator, so it
is re-imposed after every step; this is what keeps 50 km runs stable at 20 m
steps. Against the analytic solution for a constant sound-speed gradient, traced
rays match circular-arc theory to 2e-14 relative in radius and 1e-9 in travel
time.

`trace()` returns per-vertex positions, travel times, path lengths, accumulated
reflection loss and liveness, plus per-ray bounce counts.

### Sound-speed fields

`MunkProfile`, `PiecewiseLinearProfile`, `IsoProfile`, `LinearGradientProfile`
and `GriddedField` (3-D trilinear, for fronts and eddies, optionally as a
perturbation on a background profile). Each is an `nn.Module` with learnable
parameters and a batched `c_and_grad(points)`. The built-ins give analytic
gradients -- themselves differentiable in the field parameters -- and the base
class falls back to autograd for anything you write yourself.

### Boundaries

The sea surface and the seabed are both *height fields* `z = h(x, y)`, which is
why a learnable sea state drops in with no change to the tracer. `FlatHeight` is
a plane; `BilinearHeightField` is a learnable grid with analytic normals.

A crossing is located on the chord between step endpoints. The bracket is found
by bisection **under `no_grad`**, then refined by one or more Newton steps
**with gradients on**. This split is not an optimisation -- it is the difference
between working and silently not working. Plain bisection returns a dyadic
rational built from the constants `1/2, 1/4, ...`; its derivative with respect to
the seabed is *identically zero*, so bathymetry inversion would produce no
gradient at all and no error message. One Newton step from the bracketed point
restores `dt/dtheta = -(dg/dtheta)/(dg/dt)`, the implicit-function-theorem
result, just by letting autograd differentiate the update.

Reflection mirrors the slowness vector about the local normal. Losses are
`10^(-L/10)` per bounce, either a learnable constant (`ConstantLoss`) or the
grazing-angle-dependent `RayleighBottomLoss` for a lossy fluid sediment
half-space, with learnable density, sound speed and attenuation. Below the
critical angle the Rayleigh coefficient correctly collapses to the sediment
attenuation alone.

### Absorption and spreading

Thorp absorption (dB/km, kHz) is evaluated for a batch of octave-band centre
frequencies, so geometry is traced once and bands cost almost nothing --
misuka's spectral rendering, transplanted. `francois_garrison_db_per_km` is a
documented hook, not an implementation; see
[Limitations](#limitations-and-what-is-deliberately-absent).

Spreading is `1/s^2` by default, or the true **ray-tube (geometric Jacobian)**
spreading -- see [Spreading and caustics](#spreading-and-caustics).

### Receivers and energy-time curves

There are no hard hit tests. A discrete "did this ray hit the receiver"
predicate has zero gradient almost everywhere, which is useless for
optimisation. Instead the closest approach of each ray segment to each receiver
is computed in closed form, those per-segment values are reduced along the ray,
and each retained arrival is splatted with a Gaussian in miss distance and a
unit-area Gaussian in arrival time onto a fixed time grid. The output is
`[receivers, bands, time_bins]`, and a whole receiver array is one batched call.

How the reduction is done matters more than it looks. hydropt defaults to
`mode="local_min"`: every *local minimum* of the miss distance along the
polyline becomes one arrival, so a ray that swings past a receiver twice -- once
before a bottom bounce and once after -- correctly produces two. The obvious
alternative, splatting every segment weighted by its length, is a clean line
integral and is available as `mode="line_integral"`, but it destroys time
resolution: every segment within `space_gate * sigma_d` contributes, and their
closest-approach times span the whole passage. Measured on the isovelocity
waveguide in `tests/`, that is ~300 ms of arrival smear at 3 km against ~5 ms of
true separation between image-source arrivals. Reducing to local minima
decouples the kernels: `sigma_d` then controls amplitude acceptance only, and
arrival time stays as sharp as the integrator.

#### Choosing the kernel widths

`sigma_d` (m) sets how far from a receiver a ray may pass and still be heard.

* **Large** biases levels high -- rays that physically miss still contribute, and
  geometric shadows fill in -- but widens the basin of attraction enormously.
* **Small** approaches the true geometric answer, but contributions are gated at
  `space_gate * sigma_d` (6 sigma by default, below `exp(-18)`), so the gradient
  is *exactly* zero for rays not currently near a receiver. Too small and an
  imperfect initial guess has no gradient at all.

`sigma_t` (s) is the receiver's impulse response. Set it to a few time-grid
bins; below one bin the splat aliases.

The resolution these imply is a physical statement, not a free choice.
`sigma_t` is the accuracy you are asking of arrival time, and arrival time is
how the ocean encodes almost everything: at 15 km a 1 m/s error in mean sound
speed moves arrivals by 6.7 ms. Asking for a 3 ms kernel is asking for
sub-m/s accuracy, and the misfit becomes correspondingly spiky. Match the final
`sigma_t` to the accuracy you actually expect.

Because of that, inversions **anneal**: start wide enough that the initial guess
still explains part of the measurement, then tighten geometrically.
`hydropt.inverse.fit` takes `sigma_d_schedule=(start, end)` and
`sigma_t_schedule=(start, end)` for exactly this. Without it, a source 1 km from
its true position produces arrivals ~0.7 s from the measured ones against a 4 ms
kernel -- every Gaussian product is `exp(-15000)`, and the gradient is zero in
floating point.

Two rules make annealing actually work, and both were learned the hard way here.

**Blur the measurement to match.** Annealing the model's kernel while comparing
against a fixed measurement fits a blurred prediction to a sharp
measurement -- the misfit cannot reach zero and its gradient is biased. Pass
`target_sigma_t` (the width the measurement was made at) and `fit` convolves the
target up to the model's current width at every iteration. Because both kernels
are unit-area Gaussians, blurring a measurement made at `sigma_meas` by
`sqrt(sigma^2 - sigma_meas^2)` reproduces *exactly* what the model renders at
`sigma`, so the comparison stays like-for-like throughout. `sigma_d` has no such
counterpart -- widening spatial acceptance is not something you can do to a
measurement -- so prefer to hold it fixed and let `sigma_t` do the annealing.

**Stop annealing where the misfit stops being smooth.** Tightening `sigma_t`
past what the geometry resolves does not sharpen the answer, it destroys the
gradient. Scanning `examples/04`'s misfit against seamount amplitude:

| `sigma_t` | 0% | 25% | 50% | 75% | **100%** | 125% | 150% | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 ms | 0.74 | 0.54 | 0.24 | 0.08 | **0** | 0.15 | 0.69 | smooth bowl |
| 20 ms | 0.89 | 0.66 | 0.40 | 0.19 | **0** | 0.38 | 0.71 | smooth bowl |
| 15 ms | 0.93 | 0.68 | 0.51 | 0.29 | **0** | 0.56 | 0.78 | smooth bowl |
| 10 ms | 1.05 | 0.79 | 0.82 | 0.59 | **0** | 0.90 | 1.03 | non-monotone |

At 10 ms the misfit is a needle: zero exactly at the truth and noise everywhere
else, with no usable descent direction. The smooth regime ends where `sigma_t`
drops below the accuracy the measurement geometry actually carries, so scan
before choosing an end point rather than annealing as far as the arithmetic
allows.

One coupling to the tracer: in `line_integral` mode the step length must be
small compared with `sigma_d` (aim for `sigma_d >= 3 * step_size`). The default
`local_min` mode has no such requirement.

### Inversion

`hydropt.inverse.fit` optimises `scene.parameters()` against a measured ETC. It
defaults to a **log-domain** misfit: an ETC spans decades, so a linear MSE is
dominated by the loudest arrival and says almost nothing about the rest, and the
log domain additionally linearises the `10^(-L/10)` dependence on boundary loss.
It also handles sigma annealing, an optional smoothness `regulariser`, a
`project` callback for physical constraints, and per-iteration `track`ing.

## Spreading and caustics

`1/s^2` is exact only in a homogeneous medium. A sound channel has convergence
zones *because* the ray tube collapses there, so using `1/s^2` gets the arrival
times right and the levels wrong -- and wrong *smoothly*, so a fit will absorb
the error into whatever parameter is nearest. In the 50 km Munk channel of
`examples/01` the correction is **+3 to +18 dB** at the array and up to **+30 dB**
locally.

Energy is conserved along a ray tube, so with `dOmega = cos(e) de da` at the
source the intensity is

```
I = cos(e) / J,     J = |(dr/de x dr/da) . t|
```

The `de da` cancels, leaving something independent of how finely the fan was
sampled. In a homogeneous medium `J = s^2 cos(e)` exactly and this collapses
back to `1/s^2`, which is the anchor the tests use.

**Two ways to get the derivatives, and both work.**

`ray_tube` takes central differences between neighbouring rays of a
`structured_fan`. Its error is not slop but exactly the central-difference
truncation `(de^2 + da^2)/6`, matched to four significant figures at every
resolution tried, converging at exactly 4x per halving:

| fan | `d(theta)` | measured error | `(de^2+da^2)/6` | ratio |
| --- | --- | --- | --- | --- |
| 11 x 11 | 0.0698 | 1.626e-3 | 1.625e-3 | |
| 21 x 21 | 0.0349 | 4.063e-4 | 4.062e-4 | 4.00x |
| 41 x 41 | 0.0175 | 1.015e-4 | 1.015e-4 | 4.00x |
| 81 x 81 | 0.0087 | 2.539e-5 | 2.538e-5 | 4.00x |

`ray_tube_jvp` instead pushes forward-mode derivatives of ray position with
respect to launch angle through the integrator, which is what the brief
originally proposed. **It works** -- `torch.func.jvp` handles the whole tracer,
reflections included, agreeing with a central difference to 1e-9
(`scripts/check_jvp.py` is the evidence, not an assertion). It is exact rather
than second-order (1.8e-15 against the homogeneous answer), needs no neighbour
structure so it works on a Fibonacci fan, and needs no "same bounce history"
mask. It costs three traces instead of one.

Reverse-over-forward also works, which was the real question: gradients of
spreading with respect to scene parameters agree between the two methods to
three significant figures. Both are usable inside an inversion.

A note on a result that looks like a bug and is not: the gradient of spreading
with respect to Munk's `c1` is exactly zero. `c1` scales the whole profile, and
Snell's law is scale-invariant, so a uniform change in `c` leaves ray geometry
untouched. The gradient with respect to `eps`, which changes the channel's
*shape*, is not zero -- and a test pins both.

**Caustics.** The signed Jacobian changes sign wherever the tube turns inside
out, so counting sign changes gives the KMAH index. Each caustic advances the
phase by `-pi/2`, which `extract_arrivals` now applies via its `caustics`
argument -- retiring what was previously a documented gap in the coherent path.
Intensity is unbounded at a caustic, so `min_jacobian` floors the tube area.
That is the crude fix; the principled one is below.

### Gaussian beams: finite at caustics, no floor

`hydropt.beams.gaussian_beams` replaces the infinitely thin ray with a beam of
finite transverse width. The beam parameter is complex, so it cannot pass
through zero, and the amplitude stays finite everywhere -- through caustics,
through the source, through everything. `min_jacobian` is gone.

The textbook route integrates the dynamic ray equations alongside the ray,

```
dq/ds = c p,    dp/ds = -(c_nn / c^2) q
```

which needs `c_nn`, the second derivative of sound speed transverse to the ray.
hydropt cannot honestly supply it: a piecewise-linear profile has `d2c/dz2`
equal to a train of delta functions at its knots, and a trilinear grid has it
identically zero inside every cell. Dynamic ray tracing through either is
ill-defined.

But `q` and `p` describe how a *paraxial perturbation of the initial conditions*
evolves, and the tracer can already be differentiated with respect to its
initial conditions exactly. So the two fundamental solutions are just two
forward-mode tangents:

* **`Q1`** -- perturb the **launch angle**, source fixed. The ray starts at the
  same point, so `Q1(0) = 0`: this is the point-source solution, and `det Q1` is
  exactly the geometric ray tube.
* **`Q2`** -- perturb the **source position** transversally, direction fixed, so
  `Q2(0) = I`.

Both come out of `torch.func.jvp` on the ordinary tracer. No second derivative
of `c` is ever formed, and the result is exact rather than second-order. The
beam is the complex combination

```
Q = Q1 + i beta Q2,    spreading = 1 / |det Q|
```

whose determinant cannot vanish where `det Q1` does, because the imaginary part
is still there. `beta` is a transverse width scale at the source, in metres.

**The homogeneous case is checkable by hand and checked to machine precision.**
There `Q1 = s I` and `Q2 = I`, so `det Q = (s + i beta)^2`:

| quantity | closed form | max relative error |
| --- | --- | --- |
| `1/|det Q|` | `1/(s^2 + beta^2)` | 2.5e-15 |
| beam width `W` | `sqrt(c (s^2 + beta^2) / (omega beta))` | 1.7e-14 |

At `s = 0` the geometric tube is singular (1e30, i.e. the clamp) and the beam is
`1/beta^2 = 1.2346e-2` exactly. Far from the source the two agree: `1/(s^2 +
beta^2) -> 1/s^2` once `s >> beta`. The width behaves as it should too -- 2.07 m
at the source and 172.76 m at 750 m range for `beta = 9` m at 500 Hz.

**And it holds in a channel that does have caustics.** 1,400 x 25 m through the
Munk profile, `beta = 24` m:

| | geometric tube | Gaussian beam |
| --- | --- | --- |
| min `|det|` | 1e-30 (clamped) | 576.0 = `beta^2` exactly |
| max spreading | 1e30 (clamped) | 1.736e-3 = `1/beta^2` exactly |

The bound `|det Q| >= beta^2` is not empirical, it is the structure of the
method, and a test asserts it directly rather than asserting that some sampled
profile happens to look smooth.

**Choosing beta is the honest cost.** The theory does not pin it down, and
Gaussian-beam codes differ on the choice: too small and the beam is a geometric
ray again with the singularity back, too large and arrivals merge.
`suggest_beam_width(freq_khz, wavelengths=3.0)` gives a few wavelengths --
9.00 m at 500 Hz, 0.225 m at 20 kHz -- and the resulting `width` is returned so
the effect of changing it is visible rather than hidden.

**Cost: six traces, but roughly 100x the wall time.** Five of the six are
forward-mode dual traces, which neither fuse nor checkpoint -- on 120 rays x
1,500 steps, 112 s against 1.1 s for a plain traced bundle. `ray_tube` stays the
right default at one trace and second-order accuracy; reach for beams when the
amplitude at a caustic is the thing you actually need.

```python
from hydropt import gaussian_beams, suggest_beam_width, splat_etc

beams = gaussian_beams(scene, elev, azim,
                       beam_width=suggest_beam_width(freq_khz=1.0),
                       freq_khz=1.0)
etc = splat_etc(beams.result, receivers, grid, spreading=beams.spreading)
```

### Seabeds by name

`RayleighBottomLoss` is parameterised the way the physics is -- density, sound
speed, attenuation -- which is correct and unhelpful when what you have is the
word "sand". `hydropt.sediments` maps names onto numbers, from `rock` through
`sand` to `clay`, and `sediment_loss("sand")` returns a learnable loss model
because the usual reason to want a preset is to start an inversion from a
plausible seabed rather than to assert one.

**On provenance, plainly:** these are *representative* values drawn from the
ranges in the marine-sediment literature (Hamilton 1980; Hamilton & Bachman 1982;
Jackson & Richardson 2007; APL-UW 1994). They are **not** a transcription of any
one published table and should not be cited as one -- I could not reach those
tables from this environment to verify them, and inventing the precision would
have been worse than saying so. Each preset carries the range it was drawn from,
and real sediments vary by more than the gap between adjacent entries, so
quantitative work wants measured values for the site rather than a name.

What *is* verified is that the presets are physically coherent, and the ordering
is the interesting part:

| | `c2/c1` | critical angle | `R_0` |
| --- | --- | --- | --- |
| rock | 2.000 | 60.0 deg | +0.660 |
| sand | 1.167 | 31.0 deg | +0.390 |
| sandy silt | 1.040 | 15.9 deg | +0.267 |
| clayey silt | 1.007 | 6.6 deg | +0.208 |
| clay | 0.987 | **none** | +0.166 |

Two things fall out that a single number per sediment would hide.

**The critical angle closes, and then stops existing.** Total internal reflection
needs `cos(theta_c) = c1/c2`, so a sediment only has a critical angle when it is
faster than the water above it. Sand is, and is essentially lossless below about
17 degrees grazing (not 31 -- attenuation bites before `theta_c`). Clay is
*slower* than seawater, so it has no critical angle at all and no lossless regime
anywhere: 0 of 60 sampled angles under 0.5 dB, against sand's 24. That is a
qualitative difference between two rows of the same table, and it dominates how a
shallow-water channel behaves.

**Sound speed and impedance are separate axes.** My first draft of this asserted
that `R_0` goes negative for clay, since clay is the slower medium. It does not:
clay is slower *and denser*, so its impedance is still above the water's and
`R_0` stays at +0.17. Speed sets the angular structure, impedance sets the
strength, and reading one off the other gets it wrong. There is a test pinning
that so the mistake cannot come back.

## Building an environment

hydropt's fields take arrays, which is the right interface and an awkward place
to start from. `hydropt.environment` builds those arrays -- a sea surface from a
wind speed, a seabed from a roughness exponent, a range-dependent ocean from an
internal-wave displacement -- and returns ordinary `BilinearHeightField` and
`GriddedField` objects, so nothing downstream knows a generator was involved and
every field stays learnable.

Each generator is specified by a statistic, so each has something to check:

| generator | specified by | measured off the realisation |
| --- | --- | --- |
| `pierson_moskowitz_surface` | `H_s = 0.22 U^2 / g`, RMS = `H_s/4` | exact to 1e-12 |
| `fractal_bathymetry` | 2-D PSD `~ k^-gamma` | slope within 0.03 of `gamma`, at 2.5, 3.0 and 3.5 |
| `internal_wave_perturbation` | `delta c = -(dc/dz) zeta` | exact to 1e-9, depth by depth |
| `gaussian_seamount` | summit height above the seabed | exact to 1e-9 |

**Normalisation is a choice, so it is one.** `normalise="sample"` (the default)
scales a realisation so its *sample* RMS is exactly what you asked for -- a
surface built for a 1.2 m significant wave height has one. That removes the
variance-of-the-variance, so `normalise="ensemble"` gets the RMS right in
expectation instead and leaves each draw's variance free to fluctuate. Since the
RMS sets the level either way, the spectral prefactors cancel and only the
wavenumber dependence matters -- which is why the wind sea takes its level from
`H_s` rather than from Phillips' `alpha`.

**Two details that are easy to get wrong, so they are tested.** The
Pierson-Moskowitz *wavenumber* spectrum does not peak at `omega_p^2 / g`: the
change of variables carries a Jacobian, so the `S(omega)` peak at `0.877 g/U`
maps to `0.769 g/U^2` while `S(k)` peaks at `0.702 g/U^2`. Conflating them
misplaces the dominant wavelength by 10%. And `gaussian_seamount` snaps its
default centre to a *node*, because a summit landing between samples loses
`exp(-(dr/width)^2)` of its height silently -- at 50 m nodes and a 400 m width
that is 0.8%, and at coarser grids much more.

**Internal waves are built from displacement, not from noise.** A parcel of water
carries its sound speed, so a vertical displacement `zeta` shows up as
`delta c = -(dc/dz) zeta`. Building it that way means the perturbation is
automatically largest where the background gradient is steepest and *vanishes in
an isothermal layer* -- which is what is observed, and is not something a field
of independent noise would reproduce. A test pins the vanishing.

### What makes a 3-D ocean 3-D

A depth-only profile keeps a ray in its launch plane exactly, so every example
before `10` is three-dimensional only in its bookkeeping. Example 10 launches
rays at azimuth zero into a generated ocean and measures how far out of plane
each mechanism takes them over 12 km:

| | max out-of-plane | RMS |
| --- | --- | --- |
| depth-only control | **0** (exactly) | 0 |
| wind sea, `H_s` = 3.2 m | 390 m | 43 m |
| power-law seabed, 40 m RMS | 659 m | 61 m |
| internal waves, 12 m heave | **2.2 m** | 0.7 m |
| all three | 785 m | 77 m |

The refraction row has a closed form to check against -- a horizontal gradient
bends a ray on a radius `R = c/|dc/dy|`, offsetting it by `L^2/2R` over a path
`L`, which predicts 2.32 m against the measured 2.24 m.

It is also *two orders of magnitude weaker* than either boundary. That is worth
knowing before reaching for a 3-D sound-speed field to explain out-of-plane
energy: in shallow water it comes overwhelmingly from rough boundaries. The
control row is the other half of the point -- exactly zero, not nearly zero,
because a depth-only profile keeping a ray in plane is a theorem rather than an
approximation.

## Active sonar and beamforming

A passive scene renders one path, source to receiver. An active sonar renders
two, and a beamformer needs something an energy model does not have.

### Two-way propagation

`hydropt.active.render_echo` composes projector-to-target and target-to-array.
The obvious implementation, relaunching a fan for every incident arrival, is
quadratic and unnecessary: for a target whose scattering does not depend on
which path delivered the energy, the legs are separable, so the echo is the time
convolution of the two one-way responses scaled by the cross-section. **Two
renders, whatever the multipath complexity.** Spreading and absorption compose
correctly on their own, and each leg is rendered at `sigma_t / sqrt(2)` so the
convolution lands on the requested `sigma_t` rather than smearing by another
root two.

Verified against closed-form geometry: echoes arrive at `(d_in + d_out) / c` to
13 us, monostatic echoes at `2d/c`, target strength scales the result by exactly
`10^(TS/10)`, and with a surface present every in/out multipath pairing appears
at its own predicted delay. `PointTarget` position and strength are both
learnable, and gradients reach every scene parameter through both legs.

### Why coherent splatting does not work, and what does

Beamforming sums **complex pressure** across elements. Doing the same on energy
throws away the array gain, the nulls and the sidelobe structure -- everything a
beamformer exists for. But simply adding phase to the energy splatter fails,
and the reason is quantitative. Ray-fan discretisation puts each arrival time
out by microseconds because the nearest launch direction is not exactly the
eigenray -- about 80 us at 3 km for a 600-ray fan, 13 us at 40 m for a 4000-ray
cone. Against a 100 kHz quarter-wave period of 2.5 us:

| frequency | phase error from fan discretisation alone |
| --- | --- |
| 20 kHz | 1.6 cycles |
| 100 kHz | 8 cycles |
| 300 kHz | 24 cycles |

Independently-splatted elements would be mutually decorrelated and beamforming
would return noise.

What beamforming actually consumes is not absolute phase but *relative* phase
across the aperture, and that is far better conditioned: at half-wave spacing
adjacent elements are struck by essentially the same ray. So `hydropt.beamform`
extracts arrivals at a single **phase centre** and propagates each across the
aperture analytically as a plane wave along its own measured arrival direction,
`tau_m = tau_0 + (r_m - r_0) . k / c`. The differential delays are then exact to
the plane-wave approximation regardless of the absolute timing error.

Absolute phase remains unreliable. Do not use these outputs for anything that
compares phase *between* pings without first adding ray-tube interpolation.

Steering is a true time delay folded into each arrival before synthesis, not a
narrowband phase shift. A 32-element half-wave array at 100 kHz is 15.5
wavelengths long, so the delay across it is ~155 us -- many times a short
pulse's envelope width. Phase-only steering would align the carriers and leave
the envelopes scattered, collapsing the beam even when pointed correctly.

The tracer accumulates a per-vertex reflection *phase* alongside the loss: the
sea surface is pressure-release, so every bounce flips the sign, and
`RayleighBottomLoss` exposes `arg(R)`, which sweeps from -172 degrees at grazing
incidence to zero at normal.

### Measured against classical array theory

A 32-element uniform line array at half-wave spacing, 100 kHz:

| quantity | hydropt | theory |
| --- | --- | --- |
| mainlobe peak | 0.000 deg | 0 |
| -3 dB beamwidth | 3.07 deg | 101.5/N = 3.17 |
| first sidelobe (uniform) | -13.24 dB | -13.26 |
| first nulls | +/-3.60, +/-7.20 deg | +/-3.58, +/-7.18 |
| array gain | 32.000 | N = 32 |
| Hamming sidelobe | -41.8 dB | ~-42.7 |
| Blackman sidelobe | -58.2 dB | ~-58.1 |

Steering reports true target bearings to 0.001 deg, two targets 27 deg apart
resolve as two beams, and full-wavelength spacing produces grating lobes at
endfire. End to end from traced rays, individual arrivals scatter about a degree
in direction -- the nearest launch angle is not the eigenray -- yet the coherent
sum still lands on the true bearing exactly.

### Reverberation

For active sonar, reverberation -- not noise -- usually sets the detection
limit. `hydropt.reverb` treats every boundary reflection in a traced bundle as
one scattering patch, weighted by the solid angle its ray carries, so **one
render yields the whole reverberation series** instead of one render per patch.
Monostatic reciprocity closes the return path: the echo retraces its outbound
ray, arriving at `2 tau`, carrying the outbound loss twice, and coming back
along the reverse of its launch direction -- which is what gives each patch a
bearing and lets reverberation be beamformed.

Checked against the classical flat-bottom result. With Lambert scattering the
energy density must decay as `r^-5` with absolute level `pi c mu H^2 / r^5`:

| | hydropt | theory |
| --- | --- | --- |
| decay exponent | **-5.02** | -5 |
| level at 100 m | 2.43e-6 | 2.35e-6 (+3.3%) |
| level at 200 m | 7.42e-8 | 7.35e-8 (+1.1%) |
| level at 400 m | 2.28e-9 | 2.30e-9 (-0.5%) |

Matching the slope only would show the geometry is self-consistent; matching the
*level* is what checks the solid-angle bookkeeping and the `1/sin(theta)`
grazing projection of each ray's footprint onto the seabed.
`LambertScattering.strength_db` is learnable, so a measured series inverts for
the bottom type -- `examples/07` recovers a hidden -15 dB to 0.4 dB.

Two traps worth naming, both of which bit during development:

* **Cap patch counts by random subsampling, not by strength.** Keeping the
  strongest patches is right for target echoes and badly wrong here: `r^-5`
  makes the strongest patches the nearest ones, so the sample collapses onto
  the first few range cells instead of filling the window. Random subsampling
  with a compensating energy scale is unbiased and preserves the range spread.
* **Transmit directivity applies once, not twice.** The outbound leg passes
  through the projector's pattern; the return arrives at the receive array,
  whose directivity is the beamformer's job. Squaring it -- tempting, since the
  path is reciprocal in *geometry* -- suppresses off-axis patches twice over and
  understated reverberation by 8.5 dB.

### Extended targets, and the glint that is not a highlight

`hydropt.targets.ExtendedTarget` is a rigid cloud of highlights with a position
and an orientation; a `ScatteringPattern` gives each highlight's bistatic
cross-section in the body frame. Both plate and cylinder patterns are the
standard physical-optics forms and both are pinned to their textbook values:

| | closed form | hydropt |
| --- | --- | --- |
| plate, broadside backscatter | `sigma = (A/lambda)^2` | exact to 1e-12 |
| plate nulls | `sin(theta) = lambda / 2a`, each side | on the predicted angle |
| cylinder, broadside | `sigma = r L^2 / 2 lambda` (Urick) | exact to 1e-12 |
| cylinder nulls | `sin(theta) = lambda / 2L` | on the predicted angle |
| N sections of `L/N`, coherent | one cylinder of `L` | exact to 1e-12 |

That last row is the one that ties the multi-highlight machinery to the
single-body formula: amplitude per section is `(L/N) sqrt(r/2 lambda)`, so N of
them in phase give `L sqrt(r/2 lambda)`, whose square is the whole cylinder's
cross-section. If the per-section level were wrong by any factor it would not
close.

**Aspect dependence is exact on the coherent path.** `compose_arrivals` already
pairs every inbound arrival with every outbound one, and that pair sum is exactly
what a bistatic cross-section needs -- so each pair is weighted by `sigma` for
*its own* geometry, with no approximation and no extra traces. Making that work
needed one new thing: `ArrivalSet.launch_direction`, the direction a ray was
*launched* in. It is not the same as `direction`, which is where the ray ended up
going; refraction and reflection turn a ray in between, and what a scatterer
needs is the direction the energy left it in. Producers that have no such
direction to report -- reverberation -- leave the field `None` rather than
filling it with something plausible, so a pattern that needs it fails loudly.

**It is not available on the energy path, and that is structural.**
`render_echo` is fast because the two legs are *separable*; a `sigma` depending
on both directions at once is precisely what breaks that separability, and an
energy render has already discarded the pairing a bistatic pattern consumes.
`render_extended_echo` therefore handles multi-highlight targets exactly -- one
inbound render, since the highlights are just several receive points, plus one
outbound trace each -- and evaluates `sigma` at the straight-line aspect,
documented as the approximation it is.

**The finding that changed the example.** The natural expectation is that
chopping a body into sections makes it "resolve into highlights" across its
extent. It does not, and the reason is quantitative: a 0.8 m section at 100 kHz
has a beamwidth of `lambda / 2a` = 0.54 deg, while a 4 m hull at 40 m subtends
5.7 deg -- so its end sections see the sonar 2.9 deg off their own broadside,
five beamwidths out, and return almost nothing. **A smooth hull glints.**
Measured in `examples/09`: 83% of the echo energy from one section of five, with
the ends at 0.5%.

And the glint is not a feature of the hull. Sliding the body along its own axis
leaves the glint at the specular point in the *world* while it travels along the
*body*:

| body centre | glint, world y | glint, along body | share |
| --- | --- | --- | --- |
| -1.60 m | +0.00 m | **+1.60 m** | 65% |
| +0.00 m | +0.00 m | **+0.00 m** | 83% |
| +1.60 m | +0.00 m | **-1.60 m** | 75% |

What *does* spread across a body is a set of **discrete** scatterers -- edges,
corners, fittings, a wreck's structure -- each small enough to be broad in
aspect. Same layout, isotropic patterns: 2.07 deg of weighted bearing spread
against the hull's 0.52 deg and a point target's 0.53 deg, out of 5.7 deg
subtended.

**Multipath fills the aspect nulls.** The measured aspect pattern sits a steady
+7.0 dB above the direct-path envelope `cos^2(theta)/(ka sin theta)^2` at every
aspect off broadside, because a surface- or bottom-bounced ray strikes the hull
at a different aspect than the direct one and so is not in the same null. A
shallow-water FLS does not see a target's nulls as deeply as a free-field
calculation predicts. Broadside over end-on is +68.8 dB.

**Why the end caps exist.** Physical optics takes a cylinder's cross-section to
*exactly zero* end-on, because the projected length vanishes -- a modelling
artefact, not physics, and one that also means no gradient there. A real cylinder
end-on returns its end cap, so a body that must stay visible at every aspect
wants the cap as its own highlight. Adding two puts 60.0 dB back into the end-on
echo, and that composability is the reason a target is built from parts rather
than from one formula.

### What is still missing

**Absorption above ~100 kHz.** Thorp is out of range; the Francois-Garrison
hook needs implementing. Two-way absorption is 34 dB/km at 100 kHz and 67 dB/km
at 300 kHz, so useful ranges are 100-300 m -- a regime where the high-frequency
approximation behind ray theory is *better* justified than in the deep-water
examples.

## Autograd strategy

Reverse mode through the unrolled RK4 integrator. `checkpoint_every` wraps each
chunk of steps in `torch.utils.checkpoint`, so the four field evaluations per
step and their intermediates are recomputed in the backward pass instead of
stored; `tests/test_batching.py` pins checkpointed and non-checkpointed
gradients to agree to 1e-10. Rays are independent, so `render_chunked` and
`ray_chunk` bound peak memory by processing the fan in chunks, and the tests
pin those to reproduce the one-shot result exactly.

`tests/test_grad.py` runs `torch.autograd.gradcheck` in float64 through the
whole forward model -- RK4, reflection, splatting -- for boundary losses,
profile knots, bathymetry node heights, source position and receiver positions.

### Benchmark

`python scripts/benchmark.py` runs the reference workload, 5,000 rays x 4,000
steps, on CPU and on GPU when one is present. Measured on the 4-core, 16 GB CPU
container this was developed on, float32:

| 5,000 rays x 4,000 steps | time | peak RSS |
| --- | --- | --- |
| forward trace (`no_grad`) | 9.2 s | 534 MiB (the stored path) |
| forward + backward, no checkpointing | 68.7 s | 13.2 GiB |
| forward + backward, `checkpoint_every=100` | 54.5 s | 6.5 GiB |

Checkpointing halves peak memory and, here, is also *faster* -- at this size the
un-checkpointed graph is large enough that allocator and cache pressure cost
more than recomputing the RK4 stages. On a machine with less than ~16 GB the
un-checkpointed case is simply not runnable.

Two things bound the remaining 6.5 GiB. The stored path is `O(rays x steps)`
whatever you do, because the vertices are the renderer's input -- 534 MiB here.
The rest is the splatting graph and the per-chunk recomputation. Checkpointing
bounds only the *integrator* intermediates, which is where the 6.7 GiB saving
comes from.

Each case runs in its own process. That is not tidiness: sharing a process
gives a misleading answer and can kill the run, because the allocator does not
return one case's peak to the OS and the next case is then OOM-killed before it
prints anything. Smaller GPU figures are not quoted here because this container
has no CUDA device; run the script to get them.

For reference, 2,000 rays x 3,000 steps forward-only is 4.2 s (float32) or
6.2 s (float64), with 160 / 321 MiB of stored path.

### Relation to misuka

misuka backpropagates through acoustic paths with **path replay
backpropagation**: rather than storing the forward path, the backward pass
*re-traces* it from stored launch parameters and random seeds, so memory is
independent of path length. hydropt instead stores the unrolled trajectory and
differentiates through it, with checkpointing as the memory control. The
trade-off is the usual one -- hydropt is simpler and exactly reproduces its own
forward pass, but memory grows with `rays x steps`, whereas PRB's stays flat.

hydropt shares misuka's other core ideas: spectral rendering over frequency
bands, time-resolved energy responses as the differentiable output, and smooth
Gaussian splatting in place of hard intersection tests.

**TODO (scoped): a PRB-style backward pass.** The pieces already in place are the
seeded, reproducible launch generators in `hydropt/launch.py` (`fibonacci_sphere`
and `fibonacci_cone` take a `torch.Generator`) and the fact that `rk4_step` and
`_step` are pure functions of their inputs. What is missing is a
`torch.autograd.Function` whose `forward` traces under `no_grad`, storing only
launch parameters, seeds and the final state; and whose `backward` re-traces
forward while accumulating the adjoint, or integrates the ray equations in
reverse. The awkward part specific to hydropt is boundary reflection: replay has
to land on the *same* crossing points, which means either storing the crossing
fractions (small: one scalar per bounce, not per step) or re-deriving them
deterministically. Until that exists, `checkpoint_every` is the memory knob.

## Examples

```bash
cd examples && python 01_forward_munk_3d.py     # figures land in examples/figures/
```

| example | what it does | measured result |
| --- | --- | --- |
| `01_forward_munk_3d.py` | Deep Munk channel over 50 km, 2,000 rays x 3,000 steps; `1/s^2` vs ray tube | 4.9 s (target: under 30 s); tube adds +3 to +18 dB at the array |
| `02_inverse_seabed_loss.py` | Recovers hidden surface and seabed losses | 0.045 and 0.055 dB in 90 Adam steps (target: 0.5 dB in under 100) |
| `03_inverse_profile.py` | Recovers `c(z)` from a vertical line array | 5.74 -> 3.15 m/s RMS over the illuminated band (45%) |
| `04_inverse_bathymetry.py` | Recovers a seamount from a horizontal array | 38.2 -> 13.1 m RMS (66%; the brief asked 80%) |
| `05_source_localization.py` | Recovers source `(x, y, z)` on a 10 km shelf | 991 m -> 47.8 m (target: within 50 m) |
| `06_active_beamformed_sonar.py` | Active forward-looking sonar: two-way echoes beamformed into a bearing-range image | both targets to 0.00 deg in bearing, 0.05 m in range |
| `07_reverberation_limited_detection.py` | A small target on a rock seabed: is it detectable? | -0.9 dB at one element, +10.9 dB in the beam; bottom type recovered to 0.4 dB |
| `08_gaussian_beams_caustic.py` | Gaussian beams through the Munk channel's caustics | 81 of 120 rays cross one; tube pinned at its floor, beam at `|det Q| = beta^2` exactly |
| `09_extended_target_fls.py` | FLS against a 4 m hull and a wreck-like body of discrete scatterers | hull glints (83% from one section, travelling along the body); discrete scatterers spread 2.07 deg vs a point's 0.53 |
| `10_synthetic_environment.py` | A generated ocean: wind sea, power-law seabed, internal waves | out-of-plane deflection 0 m (control), 390 / 659 / 2.2 m by mechanism; refraction matches `L^2/2R` to 3% |

Each prints explicit `[PASS]`/`[FAIL]` lines for its acceptance criteria and
exits non-zero on failure. Runtimes on a 4-core CPU are seconds for 01-02 and
15-25 minutes for the annealed inversions 03-05.

**Where the inversions stop, and why.** 04 recovers two thirds of the seamount
but its relief comes out ~13 m short of the true 77 m, and 05's residual is
almost entirely in *depth* (9 m horizontal, 47 m vertical). Neither is a tuning
failure -- both were chased:

* Lightening the smoothness/Tikhonov prior, on the theory that it was
  suppressing the seamount, made 04 substantially *worse* (error climbing past
  40 m); the same change hurt 03 (3.15 -> 3.96 m/s). What reads as
  regularisation bias is mostly the null space, and the prior is what keeps the
  fit out of it.
* Decaying the learning rate from iteration zero starved the exploration that
  finds the seamount at all (8% recovered, against 56% at a constant rate),
  which is why 04 runs an explore phase then a refine phase.

The limit is information, not optimisation: 11 receivers on one horizontal
array constrain 20 seabed node heights only where rays actually touch bottom,
and source depth is encoded in surface/bottom multipath differentials whose
misfit valley is real but five times shallower than the range direction. A
second array, a second source position, or a second range is what moves these
numbers.

## Limitations and what is deliberately absent

This is **geometric acoustics**. Rays are a high-frequency approximation, and
everything below follows from that or from choices made for differentiability.

* **No diffraction.** Nothing bends into a geometric shadow; energy behind a
  seamount is zero where the real field is merely quiet. The wavelength never
  enters the geometry.
* **Caustic amplitudes need Gaussian beams, and `beta` is a choice.**
  `hydropt.spreading` counts caustics and applies the `-pi/2` KMAH phase but
  floors the tube area; `hydropt.beams` removes that floor properly, at about
  100x the wall time. Two things neither fixes: the beam width `beta` is not
  determined by the theory, so the field near a caustic depends on a parameter
  you pick; and a single beam is very wide at long range (4.7 km at 38 km for
  `beta` = 24 m at 500 Hz), where a real Gaussian-beam field is a *sum* over
  many narrow beams rather than one wide one. See
  [Gaussian beams](#gaussian-beams-finite-at-caustics-no-floor).
* **Energy, not pressure** on the passive path. Arrivals are summed
  incoherently, with no phase, so there is no interference, no modal structure
  and no Lloyd-mirror pattern. The coherent path in `hydropt.beamform` does
  carry phase, but only differentially across an aperture -- see
  [Active sonar and beamforming](#active-sonar-and-beamforming).
* **Generated environments are samples, not measurements.** `hydropt.environment`
  gives a field the right RMS and the right spectral slope. It does not give it
  crests, breaking, sandwaves or outcrops, and a Gaussian random field has no
  skewness where a real wave field does. There is still no data import -- no
  CTD, GEBCO or netCDF reader -- so measured environments come in as arrays you
  build yourself.
* **Spreading** defaults to `1/s^2`, which is exact only for a homogeneous
  medium. `hydropt.spreading` (geometric tube) and `hydropt.beams` (Gaussian
  beams) both implement better laws, but you have to ask for either -- pass
  `spreading=` to the renderer. See
  [Spreading and caustics](#spreading-and-caustics).
* **Targets are specular or isotropic, and elastic.** `hydropt.targets` has
  physical-optics plate and cylinder patterns, which is the high-frequency
  specular limit: good near broadside, understating grazing aspects where edge
  diffraction dominates, and exactly zero edge-on or end-on. There is no
  circumferential (Lamb) wave structure, so a real elastic shell's mid-frequency
  response is missing, and no shadowing between highlights of the same body.
* **No volume scattering and no rough-surface *reflection*.** Boundary
  reflection is specular. `hydropt.reverb` adds boundary *backscatter* on top
  of that, but the specular path itself is never roughened, so surface
  multipath at high frequency is an optimistic bound -- at 100 kHz a real sea
  surface is very rough against a 15 mm wavelength.
* **Reflection loss is frequency-independent.** hydropt accumulates one scalar
  reflection loss per ray, keeping path memory at `O(rays x steps)` rather than
  `O(rays x steps x bands)`; absorption carries all the spectral dependence. A
  frequency-dependent reflection coefficient needs a per-band accumulator.
* **Thorp only.** Francois-Garrison is a hook. It is not a drop-in: its
  coefficients depend on temperature, salinity and depth, so `alpha` would have
  to be integrated *along the path* rather than multiplied by total path length,
  which changes the ray state.
* **Fixed-step RK4.** Adaptive stepping is a hook; per-ray step sizes would
  desynchronise rays in arclength, which the checkpointed chunking assumes they
  do not.
* **Absolute levels are uncalibrated.** Each ray carries unit energy scaled by
  `1/s^2`, and the acceptance kernel acts as an aperture whose effective area
  depends on `sigma_d` and fan density. Pass `ray_weights` and `source_energy`,
  or fit a scale factor.
* **Inverse problems are non-convex and often underdetermined.** Rays only
  constrain the ocean they pass through; anything else is in the null space.
  Worse, Adam's per-parameter normalisation gives a null-space parameter the
  *same* step size as a well-constrained one, so weakly-constrained parameters
  wander. `examples/03` shows the practical answer: hold the unconstrained
  parameters fixed, anneal the kernels, and check the result against the region
  the rays actually illuminate.

## Layout

```
hydropt/
  fields.py      sound-speed fields (Munk, piecewise-linear, iso, gridded 3-D)
  boundaries.py  height fields, intersection, reflection, loss models
  absorption.py  Thorp; Francois-Garrison hook
  launch.py      spherical fans, Fibonacci sampling, receiver-cone importance
  tracer.py      RK4 integration -> paths, times, losses, bounce counts
  receiver.py    differentiable ETC splatting, receiver arrays
  spreading.py   ray-tube (geometric Jacobian) spreading, caustics, KMAH
  beams.py       Gaussian beams: complex beam parameter, finite at caustics
  active.py      two-way echoes through a scattering target
  targets.py     extended multi-highlight targets, aspect-dependent patterns
  environment.py synthesised surfaces, bathymetry and sound-speed fields
  sediments.py   named seabed presets -> RayleighBottomLoss
  beamform.py    coherent arrivals, aperture synthesis, delay-and-sum beams
  reverb.py      seabed and surface reverberation from bounce events
  scene.py       Scene container
  inverse.py     fit() with annealing, regularisation and logging
  plot.py        matplotlib views; optional plotly
examples/        01-10, each with acceptance checks
scripts/         benchmark.py, check_jvp.py
tests/           207 tests
```

## References

* Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics*, 2nd ed.
  -- ray equations, Rayleigh reflection, sediment attenuation.
* Munk (1974), "Sound channel in an exponentially stratified ocean".
* Thorp (1967), "Analytic description of the low-frequency attenuation
  coefficient".
* Finnendahl, Schwaerzler, et al., "Differentiable Geometric Acoustic Path
  Tracing using Time-Resolved Path Replay Backpropagation", ACM TOG 2025.
