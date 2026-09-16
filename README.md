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

### Rough boundaries: what a wind sea does to a specular reflection

hydropt reflects specularly, so a height field bends the specular direction but
says nothing about the energy a rough boundary scatters *out* of it. Above a few
kHz that is nearly all of it. `hydropt.rough` adds the Eckart correction: a
Gaussian height distribution of RMS `sigma` spreads the reflected phase by
`Gamma = 2 k sigma sin(theta)` radians, costing `(10/ln 10) Gamma^2` dB of
coherent energy -- 4.34 dB at `Gamma = 1`, 17.4 dB at `Gamma = 2`, matched to
1e-12, with both limits (`sigma = 0`, grazing incidence) *exactly* zero.

**It is a weight, not a `BoundaryLoss`, and that is a design consequence.**
`BoundaryLoss` is deliberately frequency-independent -- the tracer accumulates one
scalar per ray, keeping path memory at `O(rays x steps)`. Eckart loss is quadratic
in frequency (verified: x4, x16, x64 for f x2, x4, x8), so putting it there would
mean giving that up. `roughness_weights` instead computes it *after* the trace
from the bounces already recorded, as an `[R, B]` factor passed to the renderer as
`ray_weights` -- which now accepts a per-band weight for exactly this. Path memory
is untouched and the frequency dependence is exact. `RoughSurfaceLoss` remains as
a single-design-frequency drop-in for one-band scenes.

**The energy goes somewhere hydropt does not put it.** This removes energy from
the specular path and does not re-radiate it. Correct for a coherent calculation
-- the beamformer should not see a ghost that is not there -- and one-sided for an
energy budget. `reverb.py` models boundary backscatter separately and the two are
not coupled, so a scene with roughness loss is *missing* that energy rather than
redistributing it.

**Example 06's flagged caveat, resolved -- and it needed an angle qualifier.** That
example renders a 100 kHz FLS over a flat sea and calls its surface multipath "an
optimistic bound". At 100 kHz the wavelength is 15 mm and 2 m/s of wind raises
22 mm of RMS elevation, so the median surface-bounced path loses **43 orders of
magnitude** and 91% of them lose over 20 dB. At that example's geometry -- vehicle
at 10 m, target 40 m out, surface path near 25 degrees -- there is no coherent
surface return at all.

But my first draft of this said flatly that a rough sea destroys the specular path,
and that is wrong. `Gamma` goes as `sin(theta)`, so a **near-grazing** ray sees a
surface effectively flat along its own direction of travel and reflects coherently
however rough it is. There is a cutoff, and it has a closed form: the loss reaches
3 dB at `sin(theta) = sqrt(3 ln 10 / 10) / (2 k sigma)`.

| grazing angle | loss at 100 kHz, 2 m/s sea | energy left |
| --- | --- | --- |
| 0.5 deg | 0.12 dB | 97% |
| 2.0 deg | 1.87 dB | 65% |
| **2.53 deg** | **3.00 dB** (the cutoff) | 50% |
| 5 deg | 11.7 dB | 6.8% |
| 30 deg | 383 dB | 0 |

Measured against that: every ray in example 11 keeping over half its energy
bounces at 2.21 degrees or shallower, against the 2.53 degree cutoff the formula
predicts. So long-range shallow-water propagation, which lives at small grazing
angles, keeps its surface bounces even at high frequency -- the window just
narrows as `1/sigma`, and never closes.

## Using it as a learnable forward model

`examples/12` is the template: a 100 kHz forward-looking sonar with a
**four-element** array, a boat at 60 m, a Pierson-Moskowitz wind sea and a
power-law sand seabed. The output is a beamformed bearing-range image, and a loss
on that image reaches **every** parameter in the scene:

| | gradient |
| --- | --- |
| boat position and heading | live |
| hull section length, hull radius | live |
| propeller / skeg target strength, transom size | live |
| seabed heights (576 nodes) | live |
| wave surface (4,225 nodes) | live |
| sediment sound speed, density, attenuation | live |

11 of 11 parameter classes, checked rather than asserted.

**Four elements is the binding constraint.** At 100 kHz the array is 22.5 mm
long -- 1.5 wavelengths -- so the mainlobe is 47 deg wide against 4.7 deg for 32
elements in the same scene. That is a detector with coarse bearing, not an
imager; a real imaging FLS carries 128-256 elements. Range is unaffected
(61.0 m against a true 60.0 m), because range comes from timing rather than from
the aperture.

### A Mills cross: 120 deg x 20 deg, 2 deg beams

`examples/13` is the arrangement almost every real FLS and multibeam uses -- two
perpendicular line arrays, each doing one axis:

| | elements | aperture | measured |
| --- | --- | --- | --- |
| receive, horizontal | 64 | 47.3 cm | **2.33 deg** azimuth beams |
| transmit, vertical | 6 | 4.5 cm | **17.2 deg** elevation fan |

Neither could do it alone; their product is a 2 deg x 20 deg pencil sweeping 60
beams across 120 deg. `line_array_factor` is the transmit half -- the textbook
array factor rather than a Gaussian stand-in, so a target in a sidelobe still
returns an echo instead of quietly vanishing.

**The edge beams really are worse, by exactly the predicted amount.** A flat array
steered off broadside sees a foreshortened aperture, so the beam widens as
`1/cos(theta)`:

| steered | measured | `1/cos` | ratio |
| --- | --- | --- | --- |
| 0 deg | 2.33 deg | 2.33 | 1.000 |
| 20 deg | 2.43 deg | 2.48 | 0.978 |
| 40 deg | 3.00 deg | 3.05 | 0.985 |
| 55 deg | 4.09 deg | 4.07 | 1.004 |

**And 2 deg beams do not resolve a 12 m boat**, which is the interesting part. The
hull subtends 11.5 deg -- five beams -- so the naive expectation is a target five
beams wide. It is one:

| | beams | width |
| --- | --- | --- |
| above -3 dB | 1 | 0 deg |
| above -10 dB | 3 | 4 deg |
| above -20 dB | 4 | 6 deg |

This is the smooth-hull glint of `examples/09` seen through a real aperture: the
echo is dominated by the one section whose broadside faces the sonar, so the
-3 dB extent is a single beamwidth however finely you resolve bearing. The body's
extent appears only 20 dB down. **For imaging a smooth hull the binding
constraint is dynamic range, not beamwidth** -- and that is what a 2 deg system
has over a 47 deg one, where the weak returns sit inside the mainlobe of the
strong one rather than beside it.

The bearing of the glint itself is measured to +0.00 deg against a true 0.00.

Full sector, still fully differentiable: **6.76 s forward, 1.84 s backward**,
8.61 s per step, for 18,480 transmit rays. The sector is what costs -- a 30 deg
cone at the target is 1,200 rays and 3.6 s.

### Making it fast enough to train

The first working version took **67 s per forward-plus-backward step**. It now
takes **3.6 s**, from two changes worth knowing about:

**Step size was free.** The water is isovelocity, so rays are exactly straight
and RK4 is exact at any step; `step_size` only brackets boundary crossings, and
`find_crossing` bisects *within* a step regardless. 0.3 m and 1.5 m give an
identical answer to five significant figures, and 1.5 m is 4x cheaper. **In a
refracting profile this freedom is gone** -- there the step really is integrating
something.

**Return traces now batch.** `target_arrivals` traced one return fan per
highlight, which for eight highlights is eight Python loops over steps. It now
traces them all in one pass by handing the tracer a **per-ray source position** --
which needs no change to `trace`, because it already broadcasts the source
against the directions, so an `[R, 3]` source is simply one row per ray. Eight
highlights went from 7.6 s of tracing to 1.6 s, bit-identical (a test asserts
`torch.equal`, not closeness).

Measured, on four CPU cores:

| configuration | forward | backward | step |
| --- | --- | --- | --- |
| 1,200 tx / 400 rx rays | 2.95 s | 0.66 s | **3.61 s** |
| 4,000 tx / 1,200 rx rays | 4.87 s | 1.08 s | 5.95 s |

At 800 tx / 250 rx the bearing estimate breaks (-2.7 deg instead of 0.0), so the
cheaper row is the floor rather than a free choice.

So a gradient-based inversion of ~100 steps is a few minutes, and a network
trained with this in the loop is hours per thousand steps. That is workable for
small studies and slow for anything larger; the obvious next lever is a GPU,
which **hydropt has never been run on** -- there are likely device assumptions to
fix before it would work at all.

## Independent validation, and the defect it found

Every check described so far compares ray theory against a closed form derived
*within* the ray picture -- circular arcs, Snell's law, image sources, the ray
tube's own truncation term. Those are sharp, and they share an assumption. A
convention error consistent across the whole package would pass all of them.

`hydropt.pekeris` is the outside check: a **normal-mode** solution of the Pekeris
waveguide (isovelocity water over a fluid half-space), in numpy, with no torch in
it and no code shared with the tracer. A different formulation of the same
physics.

**The reference is validated before it validates anything.** Eigenvalues satisfy
the characteristic equation `tan(gamma H) = -rho2 gamma / rho1 beta` to 1e-8
absolute; modes are orthonormal under `int Z_m Z_n / rho dz` to 2e-5, integrated
over the evanescent tail as well as the water. The mode count comes out as
`floor(2 H sin(theta_c) / lambda + 1/2)` -- which is the **ray critical angle**
setting the number of trapped modes, the two pictures saying the same thing.

The field's prefactor is the easiest thing in a mode sum to get wrong and the
hardest to notice, so it is not taken on trust: `ideal_image_field` solves the
ideal waveguide by the method of images instead, and the two expansions of the
same Green's function agree to **3e-10** in the complex field across 75-300 Hz
and every source/receiver depth tried.

### What it found

**`splat_etc` applies geometric spreading twice.** Each ray carries `1/s^2`, and
the number of rays landing inside the fixed `sigma_d` acceptance *also* falls as
`1/s^2`. They multiply, so ETC energy falls as `1/R^4`:

| `spreading` | measured energy exponent vs range |
| --- | --- |
| default (`1/s^2`) | **4.00** |
| `ray_tube(...).spreading` | **4.00** |
| unit -- ray count supplies it | **2.00** |

Converged, not a sampling artefact: 4.00 at fan densities from 100^2 to 400^2
rays, with ray spacing from 1.76 m down to 0.44 m against `sigma_d` = 5 m. The
old note in `receiver.py` claiming levels were "calibrated only up to a scale
factor" was wrong -- the mis-calibration is a factor of `s^2`.

With unit spreading it is **exactly** right. Calibrating the arbitrary scale once
in free space and transferring it unchanged, each resolved eigenray in an ideal
waveguide matches the exact image-source energy `n/R^2`:

| eigenray | grazing | exact | hydropt / exact |
| --- | --- | --- | --- |
| direct | 0.0 deg | 1.00000e-06 | **1.0000** |
| 1st | 5.7 deg | 1.98020e-06 | **1.0000** |
| 4th | 21.8 deg | 1.72414e-06 | **1.0000** |
| 8th | 38.7 deg | 1.21951e-06 | **1.0000** |

and the free-space calibration constant is itself constant to 1.0000 across
elevations 0-40 degrees and ranges 600-2000 m.

**What this does and does not affect.** Ratios of two renders made the same way
are untouched -- an inversion's prediction against its synthetic measurement, or
one spreading law against another -- which is why it went unnoticed. Examples
02-05 fit through the same renderer on both sides, so the error largely cancels;
it does bias any parameter that trades against range, such as absorption against
boundary loss per bounce. Absolute levels and level-versus-range are wrong by
`20 log10 R` unless rendered as above. Reverberation is unaffected: `reverb.py`
computes patch energy analytically rather than by ray counting, which is why its
`r^-5` law validated correctly. Beamforming is within one range, so unaffected.

A corollary I have reasoned but not measured in a refracting channel: since
focusing is already carried by where rays land, example 01's "+3 to +18 dB" ray
tube correction is a difference between two estimators rather than the physical
correction it is described as. Verified only that the tube and `1/s^2` coincide
in an isovelocity medium and that both give exponent 4.

### Against the modes

With the corrected estimator, hydropt against the incoherent mode sum, 200 Hz,
15 trapped modes, source and receiver at 50 m:

| range | ray TL | mode TL | diff |
| --- | --- | --- | --- |
| 1000 m | 49.307 | 47.684 | +1.623 |
| 2000 m | 52.259 | 50.694 | +1.565 |
| 3000 m | 54.116 | 52.455 | +1.661 |

The **spread is 0.095 dB across a threefold change in range** -- that is the
range dependence, and two independent formulations agree on it to a tenth of a dB.
The constant offset turned out to be my choice of receiver depth. Sweeping depth
at 2 km:

| | ray vs mode |
| --- | --- |
| 17 of 18 depths | within +/-0.4 dB |
| z = 50 m (= H/2 = source depth) | **+1.57 dB**, the outlier |
| point-by-point mean | **+0.056 dB** |
| depth-averaged intensity | **+0.077 dB** |

`z = H/2` is both the source depth and the guide's symmetry plane, where half the
modes have a node, so the incoherent mode sum is anomalous at exactly that depth
and nowhere else. Away from it the two agree to a few tenths of a dB, and
depth-averaged to **0.077 dB**.

`scripts/validate_pekeris.py` prints all of this; `tests/test_pekeris.py` pins the
reference and a fast version of the eigenray comparison.

### The fix: Gaussian beam summation

The defect is not the amplitude, it is the **kernel width**. A splat sums
`amplitude * exp(-n^2/W^2)` over rays, and for a dense fan the number of rays
landing within `W` of a point falls as `1/s^2`, so the sum comes out as
`amplitude * W^2 / s^2`. Reproducing free-field spreading therefore needs

```
amplitude * W^2 = constant
```

A fixed `sigma_d` breaks exactly that. **Gaussian beams satisfy it identically**:
in a homogeneous medium `amplitude = 1/(s^2 + beta^2)` and
`W^2 = c(s^2+beta^2)/(omega beta)`, whose product is `c/(omega beta)` at every
range -- checked on the real beams, constant to 1e-9.

So `sigma_d` now accepts a `[R]` or `[R, S+1]` tensor, interpolated at the arrival
exactly as `spreading` is, and `beam_sum_kwargs(beams)` hands `splat_etc` the
right pair:

```python
beams = gaussian_beams(scene, elev, azim, beam_width=beta, freq_khz=f)
etc = splat_etc(beams.result, receivers, grid, freqs,
                ray_weights=solid_angle, sigma_t=..., **beam_sum_kwargs(beams))
```

Scalar `sigma_d` is untouched and bit-identical, so nothing existing moves.

**It is absolutely normalised, with nothing fitted.** `E s^2 = pi c / (omega beta)`,
so `E s^2 omega beta / (pi c) = 1`:

| `beta` | 600 m | 1200 m | 2400 m |
| --- | --- | --- | --- |
| 200 m | 1.001 | 1.000 | 1.000 |
| 600 m | 1.002 | 1.001 | 1.001 |

`E` is the ETC summed over bins **times `dt`** -- an ETC is an energy density in
time, and forgetting that is a factor of `1/dt`, which is how this constant was
first mis-measured as 531.

**And the answer does not depend on `beta`**, from 200 m to 1200 m within 3%.
That is the strongest evidence the sum reconstructs the field rather than
depending on the decomposition: `beta` parameterises the beam family, not the
physics. The fixed-width control, same beams and same amplitudes, gives an
exponent near 4 instead of 2.

### Where beam summation stops working

Against the Pekeris modes, depth-averaged, 200 Hz:

| range | 100 m channel | 1000 m channel |
| --- | --- | --- |
| 1500 m | +2.06 dB | **+0.22 dB** |
| 2000 m | +2.27 dB | **+0.001 dB** |
| 3000 m | +2.95 dB | **+0.081 dB** |

Same beams, same frequency, same everything but the water depth. The Fresnel
scale `sqrt(2 c s / omega)` is 69 m at 2 km, which fits comfortably inside 1000 m
of water and not at all inside 100 m. A beam wider than the channel has its
transverse Gaussian extending through both boundaries, and the sum here treats
that profile as if it were in free space -- there is no folding of the beam at
the surface and seabed. So:

**beam summation is the right estimator when the beam fits in the waveguide, and
is wrong by a couple of dB with a range trend when it does not.** In shallow
water at low frequency the beams cannot be made narrower -- the minimum width
over `beta` is the Fresnel scale, which is physics -- so that regime needs
image beams, which is not implemented. The counted estimator, whose kernel is
small compared to the channel, handles it: 0.095 dB of range trend against the
same reference.

Two estimators, each with a stated domain, is an honest answer; one estimator
silently wrong by `20 log10 R` was not.

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
| doubly curved convex surface | `sigma = R1 R2 / 4` | exact to 1e-12, all aspects |
| sphere limit `R1 = R2 = a` | `TS = 10 log10(a^2/4)` (Urick) | exact to 1e-12 |
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

### Why a hull was invisible, and it was two separate bugs

A user reported the model disagreeing with the sea: *"in reality I have no
problems seeing boat hulls with my sonar."* They were right, and finding out why
turned up two independent errors that had been masking each other.

**1. A boat hull is not a straight cylinder.** Modelled as straight cylinder
sections, a 2.4 m section at 100 kHz returns only within `lambda/2L` = **0.18
degrees** of its own broadside and collapses 40 dB by 5 degrees off. A real hull
is faired in two directions -- the waterline is a curve -- which makes it a
doubly curved convex surface with a specular point at *every* aspect. Physical
optics gives `sigma = R1 R2 / 4`, independent of aspect *and* of frequency;
`CurvedSurfaceScattering` implements it and reduces to the textbook rigid sphere
`TS = 10 log10(a^2/4)` when `R1 = R2 = a`, exact to 1e-12.

| aspect off broadside | straight 2.4 m section | curved patch (R1=0.75, R2=30) |
| --- | --- | --- |
| 0 deg | +21.6 dB | +7.5 dB |
| 5 deg | -26.7 dB | +7.5 dB |
| 90 deg | -300 dB | +7.5 dB |
| **averaged over aspect** | **-6.5 dB** | **+7.5 dB** |

Within 20 dB of the curved level on **2.3%** of aspects, against 100%. That is
the difference between a target you find on a random pass and one you do not.

**2. The return fan's splat was sized by hand, and the hand was wrong.**
`extract_arrivals` weights a ray by `exp(-0.5 (d/sigma_d)^2)` on its miss
distance, with no normalisation for fan density -- so if the fan's ray spacing at
the target exceeds `sigma_d`, the amplitude stops measuring the field and starts
measuring whether a ray happened to pass close by. `examples/12` and `13` used
400 return rays over a 40 degree cone with `sigma_d = 0.4` m; at 60 m those rays
are **3.5 m apart**. Eight *identical* highlights, whose true spread is 0.08 dB,
came back spanning **30 dB**:

| return rays | `sigma_d` | spread across 8 identical highlights |
| --- | --- | --- |
| 400 | 0.4 m | **30.5 dB** |
| 1600 | 0.4 m | 7.9 dB |
| 6400 | 0.4 m | 3.3 dB |
| 400 | **fan-matched** | **1.7 dB** |
| 6400 | **fan-matched** | **1.3 dB** |

`hydropt.launch.fan_sigma_d` sizes the splat to the fan's own measured ray
spacing (`fan_angular_spacing`) carried out to each vertex's range, and
`target_arrivals` now uses it per leg by default. The result is
**sampling-invariant** -- mean energy 6.09e-7 / 6.07e-7 / 6.04e-7 across a 16x
densification, where a fixed width drifted without bound. It is the cheap
cousin of the Gaussian-beam width above: one nearest-neighbour search rather
than six traces, geometric rather than physical, but it satisfies the same
`amplitude * W^2 = constant` requirement well enough to converge.

One consequence worth stating: with a correctly sized splat, many rays
legitimately pass within it along much the same path, so an aggressive
`max_arrivals_per_leg` now spends its budget on direct-path near-duplicates.
At a cap of 6 the sediment parameters in `examples/12` came back with *exactly
zero* gradient, because every bottom-bounced path had been discarded. The
library default of 24 restores them for about 10% more time.

**Together.** With both fixed, the 12 m hull at 60 m reads through the Mills
cross as a resolved body 6 degrees wide at -3 dB against the 11.5 degrees it
subtends, instead of a single 3 degree glint. The residual off-centre bias of
the energy centroid is *not* a third bug: over a flat surface and flat seabed
the same hull reads -1.4 degrees with its full extent, and swapping the curved
patches for plain isotropic ones changes neither number. It is the wave and
seabed realisation lighting one end more than the other, which is what a single
ping in a real sea does.

### Targets from a triangle mesh

`hydropt.mesh` takes a target as geometry rather than as hand-placed primitives.
It needs **no new propagation machinery**: a `ScatteringPattern` answers "given
an incident and a scattered direction, what is `sigma`?", and a mesh answers it
by integrating the Kirchhoff surface integral over every facet and summing
coherently. So `MeshScattering` drops into the same `ExtendedTarget`, traces the
same two legs, and costs no extra rays.

```python
from hydropt.mesh import load_obj, mesh_target
verts, faces = load_obj("boat.obj")            # or boat_hull_mesh(12, 3, 1)
boat = mesh_target(verts, faces, position=(60.0, 0.0, 1.0), yaw=90.0,
                   n_patches=4, learnable_shape=True)
```

**The facet integral has to be exact.** The tempting shortcut is to treat a
facet as a point of amplitude `A e^{iq.c}`, valid only for `|q| d << 1`. At
100 kHz with centimetre facets `|q| d` is about 25 radians, and that shortcut
came out **16x too high even at 82,000 facets**, converging only as fast as
facet area. The exact integral over a triangle is `2A` times the second divided
difference of `exp` at the three vertex phases; the textbook form of that
divides by vertex-phase differences which vanish precisely where the facet lies
in a phase front, which is where the specular return comes from.
`triangle_phase_integral` evaluates it as a nested divided difference that always
puts the widest-separated pair in the denominator, falling back to a series only
when all three collapse.

Validated against every closed form that applies:

| body | closed form | mesh |
| --- | --- | --- |
| flat facet, any aspect | `PlateScattering`, sincs and nulls included | **exact to 1e-10** |
| sphere | `sigma = a^2/4` | 0.07 dB at 20k facets |
| triaxial ellipsoid | `sigma = A^2 C^2 / 4 B^2` | **0.07 dB**, up to 3:1:0.5 |
| back faces | culled | exactly zero |

The ellipsoid is the one that matters: a sphere only exercises `R1 = R2`, while
`A^2C^2/4B^2` separates the two principal radii and depends on all three axes.

Facets must resolve the surface's **curvature**, not its phase -- about
`sqrt(lambda R)/3`, so 3.5 cm for a 0.75 m radius at 100 kHz. Cost is set by the
facet count times the number of direction pairs, not by ray count: 47,000 facets
against 576 direction pairs is 4.5 s.

**Self-occlusion, by depth buffer.** Culling facets by their own normal handles
the far side of a convex body but not a facet hidden *behind* another one: a
superstructure over a deck, a propeller behind a skeg, the far wall of anything
concave. Ray-casting every facet against every other is `O(F^2)` per direction
-- 225 million tests for a 15,000-facet hull, and `compose_arrivals` asks for
hundreds of directions. `visible_facets` projects the centroids onto the plane
perpendicular to the line of sight, bins them and keeps the nearest per bin,
which is `O(F)` and measured **10% overhead** on that hull.

Two things it has to get right, and both are pinned:

* **A convex body is owed exactly zero change.** Near the limb a sphere runs
  almost along the line of sight, so one bin spans a large depth range and a
  naive nearest-per-bin rule culls facets that nothing is in front of -- it cost
  0.26% of the sphere's return before the depth margin was widened by obliquity
  (`cell / |n.d|`, the depth a bin spans on a surface tilted that far). It is
  now exact to 1e-9.
* **A continuous surface must not shadow itself.** With a strict nearest-per-bin
  rule, two facets of one lit surface landing in the same bin knock each other
  out. So a facet is culled only when something sits more than a tolerance in
  front of it -- on a real body, hidden parts are separated by many facets.

Measured on two identical panels, one directly behind the other: without
occlusion they add coherently to **4x** the power of a single panel; with it,
**1.00x**.  On a scene it is quieter than that, and the geometry says why: a
beam-on catamaran with 5 m between hulls loses **2.8 dB** when the sonar looks
along the separation axis from 4 degrees up, and **nothing at all** from 17
degrees up -- a ray clearing the near keel has risen 1.5 m by the far hull,
which is more than its 0.9 m draught, so the shadow passes over it.  The peak is
unaffected either way, because the peak is the near hull's own specular; what
occlusion removes is spread along the far edge. Slid sideways so nothing is in the way, both give 4x. Bistatic needs
both ends -- visible from the source is not the same as visible to the receiver,
and they coincide only when monostatic.

The visibility mask is binary and detached, so occlusion carries no gradient;
amplitudes of visible facets still do. And `mesh_target(n_patches > 1)` confines
occlusion to *within* a patch, since each patch is its own `MeshScattering` --
so one part of the body can no longer hide another. Use `n_patches=1` when that
matters more than the target's extent in the image.

**What it still does not model.** Physical optics has no edge diffraction, so
grazing returns are understated, and the occlusion is a centroid test, so a
facet much larger than a bin is treated as its centre point.

**`boat_hull_mesh` is a hull, not a spindle.** The first version tapered to a
point at *both* ends -- a canoe -- which showed up as an aspect pattern exactly
symmetric fore and aft. A boat carries most of its beam and draft to a flat
**transom**, which is a large near-vertical plate and one of the strongest
features on the body from astern, so the generator now closes the stern with
facets and only the bow tapers to a stem. Section shape varies too, from boxy
and nearly flat-bottomed aft to a sharp V forward, which is what decides whether
the bottom throws a specular return straight down. Two orientation bugs came
with it -- the shell wound inward, then the transom cap wound forward -- both
caught only after adding tests that check normal *orientation* rather than
dimensions.

**What the mesh then said about hulls**, which the analytic patch could not. At a
forward-looking sonar's shallow depression angle (10.4 deg for the
`examples/12` geometry) **aspect dominates**: the hull is a strong target on the
beam (-5.9 dB) and 28 dB weaker bow-on, because below the waterline a hull's
outward normal tilts downward and toward the bow also swings forward, so a
shallow look never finds the bow's specular point.

From underneath it is a **plate**, not a curved surface: a real hull has a flat
run aft, and a flat surface seen near normal returns `(A/lambda)^2` -- enormous,
but through a lobe whose first null is at `lambda/2L`, 2.2 arcmin for a 12 m
bottom. Measured bow-on, TS climbs -16 -> +1 -> +15 -> +26 -> +31 dB from 70 to
89 degrees and then rings violently inside the last degree. A downward-looking
sonar gets a spectacular return off a hull and loses it for a degree of vehicle
attitude.

That second conclusion is the opposite of what the first version of this
generator said, because that version was a round-bilged spindle with no flat
bottom anywhere. Getting it right needed the hull to be a hull.

### The sector display, and what fills the picture

`examples/15` puts the pieces together as a vehicle would carry them: a 100 kHz
Mills cross on an AUV at 18 m in 30 m of water, a 12 m hull as a mesh on the
surface at 55 m, a wind sea above and a sand seabed below, resampled onto a grid
in **metres** rather than left as bearing x range. The resampling is bilinear
and differentiable, so a loss written in metres -- where a target is, how long
it is -- reaches the scene exactly as one on the beamformed rectangle does.

Three things this surfaced that the earlier examples did not.

**Reverberation is the picture.** `target_arrivals` returns target echoes only,
so an image built from it alone shows a boat on a black background -- which is
neither what a sonar displays nor what sets detection. Every bottom and surface
bounce in the transmit fan is a scattering patch with its own bearing and range,
so `reverberation_arrivals` sums with the echo *before* beamforming. The boat
then stands +17 dB above the reverberation at its own range, which is the number
that actually matters; against the *median* it would read 18 dB better, because
reverberation is a speckle field and its median sits far below the level a
detector competes with.

**A regular fan images its own sampling.** A lattice in (elevation, azimuth)
puts every bounce on a lattice too, and the seabed then renders as a set of
clean concentric arcs. Jittering each ray within its own cell keeps the density
and removes the artefact. Relatedly, a display much finer than the sonar's own
resolution -- 0.2 m cells against 1.5 m of beamwidth -- shows the patch sampling
rather than the seabed.

Arc-shaped *speckle* survives both fixes, and that one is real: the range ripple
is unchanged from 32 to 128 elevation samples, so it is not sampling. At 50 m
the beam is 1.4 m wide across but the pulse is 0.09 m deep, so a resolution cell
is fifteen times longer tangentially than radially and speckle grains come out
as arc segments. That is what sonar speckle looks like.

**Draw the wedge, not a raster.** `hydropt.plot.plot_fls_sector` renders the
sector display straight from the beamformer's own grid: the bearing-range mesh
maps to `x = R cos B`, `y = R sin B` exactly, so nothing is interpolated and,
more to the point, no cell falls outside the swath. A rectangular grid has to
pad its corners with something, and padding a sonar image with zeros invents
dark water the sonar never looked at. Cells grow with range the way the beams
do, which is also the honest thing to show.

**Autograd used to set the image size, and no longer does.** `beamform`'s
kernel works on `[steer, elements, bands, arrivals, gate_width]` -- note the
*element* axis, a 64x multiplier for a 64-element array, and the gate width,
which grows with `sigma_t / bin_width`. Materialising that for every arrival at
once is what made a full-size image cost gigabytes, and it forced `examples/15`
to render its dense ping under `no_grad` and demonstrate gradients on a smaller
stand-in.

It now blocks the arrival axis and recomputes each block in the backward pass
rather than keeping it, summing the element axis inside the loop so the
accumulator is only `[steer, bands, time]`. With gradients on, 64 elements, 181
beams and 500 bins:

| arrivals | before | after |
| --- | --- | --- |
| 400 | 3.7 GB | 1.1 GB |
| 800 | 5.7 GB | 1.2 GB |
| 1600 | 10.7 GB | 1.5 GB |
| 3000 | **exhausted a 14 GB machine** | 1.9 GB |
| 20,000 | -- | 1.9 GB |

The point is the last two rows: the cost stops growing with the arrival count.
`examples/15` now carries gradients through the *same* dense 20,000-arrival ping
it displays, and the split is gone.

Blocks are not free -- each is a checkpoint region -- so *too small* costs more
than none at all: at 3000 arrivals, blocks of 256 came out worse (4.2 GB) than a
single block (2.2 GB), while blocks of 2048 were better (1.9 GB). Hence the
large default. With gradients off there is no graph to trade and blocking alone
bounds the forward peak (1.74 -> 0.59 GB on the same render).

### Is it invertible, or only differentiable?

Every example before `16` checked that gradients were finite and non-zero. That
is not the same as invertible, so `examples/16` starts the boat at the wrong
position and heading and asks whether descending on an image loss recovers the
right ones.

**It works as a refiner, not as a search.** From within about 1-2 m it recovers
position to a fraction of the 3.75 m bearing cell -- 0.25 to 1.3 m across
realisations, three to fifteen times finer -- which is the point of fitting a
model rather than reading a peak. From 3.6 m out it does not converge at all.
So the pipeline is detection first (`examples/15` localises to ~2.5 m) and this
as a refinement stage.

| start error | final position | final heading |
| --- | --- | --- |
| 0.6 m, 2 deg | **1.29 m** (2.9x finer than the cell) | 14.6 deg |
| 1.8 m, 6 deg | 0.57 m | 8.7 deg |
| 3.6 m, 12 deg | 3.12 m | 15.5 deg |
| 7.2 m, 25 deg | 7.11 m | 26.7 deg |

**Heading looks broken in that table, and is not.** The boat is beam-on, and at
broadside a hull's projected length is *stationary* in yaw: rotating it barely
changes the image, so there is nothing for the gradient to hold. Turn it off
broadside and heading comes back as well as position does -- 14.6 deg beam-on,
**2.1 deg** at 70 degrees, **2.8 deg** at 45. A degeneracy of the geometry, not
of the tracer, and one worth knowing before trying to fit heading on a beam
aspect.

Four things had to be right, and each was measured:

* **Log compression.** On the raw image the loss is non-monotone in every
  direction -- speckle -- and has nothing to descend. In log it climbs cleanly
  to about 2 m and then *saturates*: past that the two images are uncorrelated
  and no longer know which way the boat is. That saturation is the capture range.
* **A finite-difference step scaled to the resolution cell.** Checking the
  gradient with a 0.25 m step against a 0.09 m range cell measures a secant
  across three cells and reports the gradient as **wrong** (cos = -0.98). At a
  fifth of a cell it agrees (cos = +0.99). The check was broken, not the tracer
  -- worth knowing before concluding otherwise from a failed gradient test.
* **A learning rate annealed with the pulse.** The image decorrelates over about
  one resolution cell, so a step much larger than a cell lands somewhere
  uncorrelated and descent becomes a random walk. A fixed 0.45 m step against a
  0.09 m cell drove the fit *away* from truth, 7.8 m to 15.1 m.
* **An honest measurement.** Sharing the receive fan between the synthetic
  measurement and the model is an inverse crime, and it was unavoidable until
  now: `target_arrivals` documented a `generator` that did nothing, because a
  Fibonacci cone is deterministic unless jittered. `rx_jitter` fixes that.

**And that turned up a real defect in `fan_angular_spacing`.** It measured the
*nearest-neighbour* distance, which reads the point pattern's regularity as much
as its density: on a lattice the nearest neighbour sits one spacing away, on an
irregular set of the same density about half that. So a jittered fan read as
twice as dense, its `sigma_d` came out half as wide, and it captured **0.457x**
the energy of the identical-density regular fan. It now estimates the density
over a neighbourhood instead -- a cap of radius `r_k` holds about `k` rays, so
the spacing is `r_k sqrt(pi/k)` -- which agrees between regular and jittered
fans to a few percent and leaves the answer invariant to fan density.

One consequence worth planning around: **realisation noise does not fall with
fan density.** Because `sigma_d` is sized to the fan, densifying it narrows the
splat in step and the effective number of contributing rays stays the same. The
spread between independent realisations was 2.47% at 200, 800 and 3200 return
rays alike. That is why recovered pose stops improving once the fan is adequate
-- the residual is realisation noise, not ray count -- and why spending rays on
it is the wrong lever.

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
| `11_rough_surface_coherence.py` | Eckart coherence loss, and example 06's surface ghost | median surface path at 100 kHz loses 43 orders of magnitude; survivors all within the 2.53 deg cutoff |
| `12_fls_boat_learnable.py` | 100 kHz FLS, 4 hydrophones, boat over a rough seabed, wind sea | 11/11 parameter classes carry gradients; 4.5 s per forward+backward step |
| `13_mills_cross_fls.py` | Mills cross: 120 x 20 deg, 2 deg beams, 64 + 6 elements | beams 2.33 deg, broadening matches `1/cos` to 2.2%; hull resolved 6.0 deg at -3 dB against the 11.5 deg it subtends |
| `14_mesh_boat.py` | a boat as 15k triangles, Kirchhoff facet scattering | ellipsoid matches `A^2C^2/4B^2` to 0.07 dB; hull is a plate from beneath (47 dB fall from 89 to 70 deg); gradient reaches the mesh vertices |
| `15_auv_scene_cartesian.py` | AUV FLS: boat, wind sea and seabed, imaged in metres | boat lands on the hull against 3.4 m of beamwidth; return spans 15 m for a 12 m boat; +21 dB over reverberation; gradients through the full 20k-arrival image |
| `16_invert_pose_from_image.py` | recovering boat pose from the image by gradient descent | position 2.9x finer than the bearing cell from a 0.6 m start; heading 2.8 deg off broadside, degenerate on it; no convergence from 3.6 m |

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
* **Absolute levels need an estimator chosen on purpose.** A fixed `sigma_d`
  with the default `spreading` double-counts geometric spreading and gives
  `1/R^4`. Use `beam_sum_kwargs` where the beam fits inside the waveguide, or the
  counted estimator where it does not; see
  [The fix](#the-fix-gaussian-beam-summation). The *default* is still the
  double-counting one, because changing it would silently move every existing
  scene's levels -- callers opt in.
* **Never run on a GPU.** Every timing here is four CPU cores. There are likely
  device assumptions (generators, `.cpu()` calls, dtype defaults) to fix before a
  GPU run would work, and that is the obvious lever if the 3.6 s training step
  is the thing standing in your way.
* **No image beams.** A Gaussian beam wider than the water column is not folded
  at the boundaries, which is what costs 2-3 dB in a 100 m channel at 200 Hz.
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
* **Rough boundaries lose coherent energy but do not scatter it.**
  `hydropt.rough` applies the Eckart coherent-reflection loss, which is the
  correct thing for a specular/coherent calculation and one-sided for an energy
  budget: the energy removed from the specular path is not re-radiated anywhere,
  and is not coupled to `reverb.py`'s boundary backscatter.
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
  mesh.py        triangle-mesh targets by exact Kirchhoff facet integration
  environment.py synthesised surfaces, bathymetry and sound-speed fields
  sediments.py   named seabed presets -> RayleighBottomLoss
  rough.py       Eckart coherent-reflection loss for rough boundaries
  pekeris.py     independent normal-mode reference (numpy; no torch)
  beamform.py    coherent arrivals, aperture synthesis, delay-and-sum beams
  reverb.py      seabed and surface reverberation from bounce events
  scene.py       Scene container
  inverse.py     fit() with annealing, regularisation and logging
  plot.py        matplotlib views, FLS sector display; optional plotly
examples/        01-16, each with acceptance checks
scripts/         benchmark.py, check_jvp.py, validate_pekeris.py, validate_beamsum.py
tests/           424 tests
```

## References

* Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics*, 2nd ed.
  -- ray equations, Rayleigh reflection, sediment attenuation.
* Munk (1974), "Sound channel in an exponentially stratified ocean".
* Thorp (1967), "Analytic description of the low-frequency attenuation
  coefficient".
* Finnendahl, Schwaerzler, et al., "Differentiable Geometric Acoustic Path
  Tracing using Time-Resolved Path Replay Backpropagation", ACM TOG 2025.
