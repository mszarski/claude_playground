# hydropt

A differentiable 3-D underwater acoustic ray tracer in PyTorch.

`hydropt` traces geometric-acoustic rays through a 3-D ocean -- a sound-speed
field, a sea surface, a seabed -- and renders a time-resolved energy response at
each receiver. Every step of that forward model is differentiable, so the same
code that predicts a measurement can be run backwards to recover the ocean that
produced it: seabed reflection loss, a sound-speed profile, a bathymetry field,
or the position of the source.

It is a 3-D, autograd-native reinterpretation of the forward model in the
`underwater_acoustic_simulator.jsx` 2-D prototype, and it takes its overall shape
from TU Berlin's *misuka* (Finnendahl et al., "Differentiable Geometric Acoustic
Path Tracing using Time-Resolved Path Replay Backpropagation", ACM TOG 2025) --
with ocean physics in place of room acoustics. See
[Relation to misuka](#relation-to-misuka).

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
pytest                     # 58 tests
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

Spreading is `1/s^2` in path length. Ray-tube (geometric Jacobian) spreading is
a hook -- see [Limitations](#limitations-and-what-is-deliberately-absent).

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

`python scripts/benchmark.py` reports the reference workload (5,000 rays x
4,000 steps) on CPU and, when present, GPU. On the 4-core CPU container used to
develop this, float32:

| workload | time | memory |
| --- | --- | --- |
| 2,000 rays x 3,000 steps, forward only | 4.2 s | 160 MiB stored path |
| 2,000 rays x 3,000 steps, float64 | 6.2 s | 321 MiB stored path |

The stored path dominates: it is `O(rays x steps)` regardless of
checkpointing, because the vertices are the renderer's input. Checkpointing
bounds the *integrator* intermediates, which are what would otherwise make the
backward pass 4-5x larger. Run the script for the full 5,000 x 4,000 figures on
your own machine -- they depend strongly on core count and memory bandwidth.

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

| example | what it does |
| --- | --- |
| `01_forward_munk_3d.py` | Deep Munk channel over 50 km; 2,000 rays x 3,000 steps; ray plots + ETCs |
| `02_inverse_seabed_loss.py` | Recovers hidden surface and seabed losses with Adam |
| `03_inverse_profile.py` | Recovers a sound-speed profile from a vertical line array |
| `04_inverse_bathymetry.py` | Recovers a seamount height field from a horizontal array |
| `05_source_localization.py` | Recovers source `(x, y, z)` on a 10 km shelf |

Each prints explicit `[PASS]`/`[FAIL]` lines for its acceptance criteria and
exits non-zero on failure.

## Limitations and what is deliberately absent

This is **geometric acoustics**. Rays are a high-frequency approximation, and
everything below follows from that or from choices made for differentiability.

* **No diffraction.** Nothing bends into a geometric shadow; energy behind a
  seamount is zero where the real field is merely quiet. The wavelength never
  enters the geometry.
* **No caustic correction.** Where neighbouring rays cross, ray theory predicts
  infinite intensity. hydropt does not detect caustics or apply the usual `pi/2`
  phase advance, so levels near a convergence zone are wrong -- and it is a
  *smooth* wrongness, so a fit will happily absorb it into other parameters.
* **Energy, not pressure.** Arrivals are summed incoherently, with no phase.
  There is no interference, so no modal structure and no Lloyd-mirror pattern.
* **Spreading is approximated.** `1/s^2` is exact only for a homogeneous medium.
  Real focusing and defocusing needs the ray-tube Jacobian -- the derivative of
  ray position with respect to launch angle, cheapest via forward-mode AD over
  the launch parameters. That is a hook, not an implementation, and it is the
  single largest source of level error in a strongly refracting channel.
* **No volume scattering, no bubbles, no rough-surface scattering.** Boundary
  reflection is specular; `sigma_d` blurs the geometry but does not model
  scattering physics.
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
  scene.py       Scene container
  inverse.py     fit() with annealing, regularisation and logging
  plot.py        matplotlib views; optional plotly
examples/        01-05, each with acceptance checks
scripts/         benchmark.py
tests/           58 tests
```

## References

* Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics*, 2nd ed.
  -- ray equations, Rayleigh reflection, sediment attenuation.
* Munk (1974), "Sound channel in an exponentially stratified ocean".
* Thorp (1967), "Analytic description of the low-frequency attenuation
  coefficient".
* Finnendahl, Schwaerzler, et al., "Differentiable Geometric Acoustic Path
  Tracing using Time-Resolved Path Replay Backpropagation", ACM TOG 2025.
