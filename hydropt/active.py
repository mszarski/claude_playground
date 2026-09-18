"""Active sonar: two-way propagation through a scattering target.

A passive scene renders one path, source to receiver.  An active sonar renders
two: projector to target, then target to receive array.  For a monostatic
system the array sits beside the projector, so the one-way response is just the
outgoing pulse -- the target is what makes the problem interesting.

Composition without a render per arrival
----------------------------------------
The obvious implementation, relaunching a fan for every incident arrival, is
quadratic and unnecessary.  For a target whose scattering does not depend on
which incident path delivered the energy, the two legs are *separable*: the
echo is the time convolution of the inbound channel response with the outbound
one, scaled by the scattering cross-section.

.. math::
    E_{\\text{echo}}(t) = \\sigma \\int E_{\\text{in}}(u)\\,E_{\\text{out}}(t-u)\\,du

So the whole two-way problem costs exactly **two** renders, whatever the
multipath complexity.  Spreading and absorption compose correctly on their own:
the inbound response already carries ``1/s_1^2`` and ``10^(-alpha s_1/10)``, the
outbound ``1/s_2^2`` and ``10^(-alpha s_2/10)``, and their product is the
two-way law over the total path.

The kernel widths compose too.  Each leg is rendered with ``sigma_t / sqrt(2)``
so that convolving the two unit-area Gaussians reproduces exactly the requested
``sigma_t`` in the echo, rather than smearing it by another factor of root two.

What this assumes
-----------------
Scattering is isotropic -- one cross-section, no aspect dependence.  An
aspect-dependent target breaks the separability that makes the convolution
valid, because the outbound amplitude would then depend on the inbound
direction; the honest treatment is a render per incident arrival, or a
factorised pattern applied to the outbound fan for a dominant incident
direction.  See :class:`PointTarget` for where that would attach.

Echoes here are *energy*, summed incoherently.  Coherent echoes -- the ones a
beamformer consumes -- are built from arrival lists instead; see
:mod:`hydropt.beamform`.
"""

from __future__ import annotations

import warnings

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .beamform import ArrivalSet, extract_arrivals
from .launch import (fan_angular_spacing, fan_sigma_d, fibonacci_cone,
                     fibonacci_sphere)
from .receiver import splat_etc
from .scene import Scene
from .targets import ExtendedTarget
from .tracer import TraceResult, trace

__all__ = ["PointTarget", "EchoResult", "return_fan", "render_echo",
           "compose_arrivals", "render_extended_echo", "target_arrivals"]


class PointTarget(nn.Module):
    """An isotropic point scatterer with a learnable position and strength.

    Args:
        position: ``(x, y, z)`` in metres.
        target_strength_db: target strength ``TS``; the scattering
            cross-section used is ``sigma = 10 ** (TS / 10)``.  Real targets
            run from about -40 dB (a small fish) to +10 dB or more (a
            broadside submarine hull).
        learnable: register both as parameters.

    Aspect dependence would attach here as a ``pattern(incident, scattered)``
    method; note that using it invalidates the convolution shortcut in
    :func:`render_echo` -- see the module docstring.
    """

    def __init__(
        self,
        position: tuple[float, float, float] | Tensor,
        target_strength_db: float = -10.0,
        *,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        pos = torch.as_tensor(position, dtype=torch.get_default_dtype()).reshape(3)
        ts = torch.as_tensor(float(target_strength_db))
        if learnable:
            self.position = nn.Parameter(pos)
            self.target_strength_db = nn.Parameter(ts)
        else:
            self.register_buffer("position", pos)
            self.register_buffer("target_strength_db", ts)

    def cross_section(self) -> Tensor:
        """Scattering cross-section ``sigma = 10 ** (TS / 10)``."""
        return 10.0 ** (self.target_strength_db / 10.0)

    def extra_repr(self) -> str:
        return (f"position={self.position.tolist()}, "
                f"TS={float(self.target_strength_db):.1f} dB")


class _RelocatedScene:
    """A read-only view of a scene with the source moved elsewhere.

    :func:`hydropt.tracer.trace` only ever reads attributes off the scene, so a
    proxy is enough and avoids rebuilding an ``nn.Module`` -- which matters,
    because the target position is usually a :class:`torch.nn.Parameter` and
    rebuilding would either detach it or smuggle a grad-requiring tensor into a
    buffer.  Every sub-module is shared by reference, so gradients reach the
    field, boundaries and losses exactly as they would through the original.
    """

    def __init__(self, scene: Scene, source: Tensor) -> None:
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_source", source)

    def __getattr__(self, name: str):
        return getattr(self._scene, name)

    def source_position(self) -> Tensor:
        return self._source


class EchoResult(NamedTuple):
    """Output of :func:`render_echo`."""

    etc: Tensor  # [receivers, bands, echo_bins] two-way energy response
    inbound: Tensor  # [1, bands, leg_bins] projector -> target
    outbound: Tensor  # [receivers, bands, leg_bins] target -> array
    leg_time_grid: Tensor  # [leg_bins] one-way time base of the two legs


def return_fan(
    target: PointTarget,
    receivers: Tensor,
    n_rays: int = 3000,
    *,
    half_angle_deg: float = 45.0,
    n_background: int = 0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Launch directions from a target towards a receive array.

    The array subtends a tiny solid angle from any useful range, so an
    isotropic fan from the target wastes almost every ray.  The cone must still
    be generous: the surface- and bottom-reflected returns leave the target at
    angles well away from the direct bearing, and a tight cone silently drops
    exactly the multipath that carries the target's depth.
    """
    axis = receivers.reshape(-1, 3).mean(0) - target.position.detach().reshape(3)
    dirs = fibonacci_cone(n_rays, axis, half_angle_deg, generator=generator)
    if n_background > 0:
        dirs = torch.cat((dirs, fibonacci_sphere(
            n_background, generator=generator, dtype=dirs.dtype, device=dirs.device)), 0)
    return dirs


def render_echo(
    scene: Scene,
    target: PointTarget,
    tx_directions: Tensor,
    rx_directions: Tensor,
    echo_time_grid: Tensor,
    *,
    sigma_d: float,
    sigma_t: float,
    tx_weights: Tensor | None = None,
    rx_weights: Tensor | None = None,
    ray_chunk: int = 0,
    trace_kwargs: dict | None = None,
    **splat_kwargs,
) -> EchoResult:
    """Render the two-way echo from ``target`` onto the scene's receive array.

    Args:
        scene: the scene; its ``source`` is the projector and its ``receivers``
            the receive array.
        target: the scatterer.
        tx_directions: projector launch directions, ``[Nt, 3]``.  Apply the
            transmit beam pattern through ``tx_weights``, not by narrowing this
            fan, so that sidelobe illumination is still modelled.
        rx_directions: launch directions from the target, ``[Nr, 3]``; see
            :func:`return_fan`.
        echo_time_grid: uniform two-way time grid (s).
        sigma_d, sigma_t: kernel widths for the echo.  Each leg is rendered at
            ``sigma_t / sqrt(2)`` so the convolution lands on ``sigma_t``.
        tx_weights, rx_weights: per-ray weights, e.g. transmit directivity.
        ray_chunk, trace_kwargs, splat_kwargs: forwarded to the tracer/splatter.

    Returns:
        :class:`EchoResult`.  Differentiable in the target position and
        strength, and in every scene parameter, through both legs.
    """
    grid = echo_time_grid
    n_echo = int(grid.shape[0])
    if n_echo < 2:
        raise ValueError("echo_time_grid needs at least two bins")
    dt = (grid[-1] - grid[0]) / (n_echo - 1)
    dt_f = float(dt)
    if float(grid[0]) < 0.0:
        raise ValueError("echo_time_grid must start at a non-negative time")

    # Each leg is one-way, so it only has to span the full echo window; the
    # convolution then covers everything up to twice that.
    n_leg = int(math.ceil(float(grid[-1]) / dt_f)) + 1
    leg_grid = torch.arange(n_leg, dtype=grid.dtype, device=grid.device) * dt
    sigma_leg = sigma_t / math.sqrt(2.0)
    kw = dict(sigma_d=sigma_d, sigma_t=sigma_leg, ray_chunk=ray_chunk, **splat_kwargs)

    # Leg 1: projector -> target.  The scene is unchanged; the target simply
    # stands in as the receiver, which keeps its position differentiable.
    inbound = splat_etc(
        trace(scene, tx_directions, **(trace_kwargs or {})),
        target.position.reshape(1, 3), leg_grid, scene.freqs_khz,
        ray_weights=tx_weights, **kw,
    )

    # Leg 2: target -> array, traced from a scene view whose source is the target.
    outbound = splat_etc(
        trace(_RelocatedScene(scene, target.position), rx_directions, **(trace_kwargs or {})),
        scene.receivers, leg_grid, scene.freqs_khz,
        ray_weights=rx_weights, **kw,
    )

    # Full linear convolution along time, done by FFT so it stays one op and
    # differentiable.  Zero-padding to >= 2n-1 is what makes it linear rather
    # than circular; getting that wrong wraps late echoes onto early ones.
    n_fft = 1 << int(math.ceil(math.log2(max(2 * n_leg - 1, 2))))
    spec = torch.fft.rfft(inbound, n=n_fft) * torch.fft.rfft(outbound, n=n_fft)
    full = torch.fft.irfft(spec, n=n_fft)[..., : 2 * n_leg - 1]
    # dt turns the discrete sum into the continuous convolution integral.
    full = full * dt * target.cross_section()

    offset = int(round(float(grid[0]) / dt_f))
    if offset + n_echo > full.shape[-1]:
        pad = offset + n_echo - full.shape[-1]
        full = torch.nn.functional.pad(full, (0, pad))
    etc = full[..., offset : offset + n_echo]
    return EchoResult(etc=etc, inbound=inbound, outbound=outbound, leg_time_grid=leg_grid)


def compose_arrivals(
    inbound: ArrivalSet,
    outbound: ArrivalSet,
    target: PointTarget | ExtendedTarget,
    *,
    highlight: int = 0,
    freqs_khz: Tensor | None = None,
    max_arrivals: int | None = None,
) -> ArrivalSet:
    """Combine the two legs coherently into echo arrivals at the array.

    The energy path in :func:`render_echo` convolves two splatted responses;
    this is the same composition done on arrival *lists* instead, which keeps
    phase and arrival direction intact so a beamformer can consume the result.

    Every inbound arrival pairs with every outbound one: delays add, complex
    amplitudes multiply, boundary phases add, and the scattering cross-section
    enters once as an amplitude ``sqrt(sigma)``.  The direction is taken from
    the outbound leg, because that is the one the array actually sees -- the
    inbound leg only determines how much energy the target re-radiated and when.

    The pair count is the product of the two arrival counts, so cap the legs
    with ``extract_arrivals(..., max_arrivals=...)``; a dense fan yields many
    near-duplicate arrivals that add cost without adding structure.

    **Aspect dependence is exact here.** Pass an :class:`hydropt.targets.\
ExtendedTarget` with a ``highlight`` index and ``freqs_khz``, and each pair is
    weighted by the bistatic cross-section for *that pair's own geometry*: the
    incident direction is the inbound arrival's ``direction`` (where the energy
    was going when it reached the target) and the scattered direction is the
    outbound arrival's ``launch_direction`` (the direction it left in).  Nothing
    is approximated and no extra traces are needed -- the pair sum this function
    already performs is exactly the sum an aspect-dependent scatterer requires.
    Contrast :func:`render_echo`, whose separability is what a two-directional
    cross-section breaks.

    Args:
        inbound: arrivals at the target (or highlight).
        outbound: arrivals at the array phase centre, from a trace launched at
            the target.  Must carry ``launch_direction`` when ``target`` is
            aspect-dependent.
        target: a :class:`PointTarget`, or an ``ExtendedTarget``.
        highlight: which highlight of an ``ExtendedTarget`` these legs belong to.
        freqs_khz: band centres; required for an ``ExtendedTarget``, whose
            cross-section is frequency-dependent.
        max_arrivals: keep only this many strongest pairs.
    """
    n_in, n_out = inbound.n_arrivals, outbound.n_arrivals
    if n_in == 0 or n_out == 0:
        z = inbound.time[:0]
        z3 = inbound.direction[:0]
        return ArrivalSet(z, inbound.amplitude[:0], z3, z, z, z, z3)
    if isinstance(target, ExtendedTarget):
        if freqs_khz is None:
            raise ValueError("an ExtendedTarget's cross-section is frequency "
                             "dependent, so freqs_khz is required")
        if outbound.launch_direction is None:
            raise ValueError(
                "aspect-dependent scattering needs the direction each outbound "
                "ray left the target in, but this ArrivalSet has no "
                "launch_direction.  extract_arrivals() populates it; producers "
                "that cannot (reverberation) leave it None.")
        # [n_in, 1, 3] against [1, n_out, 3] -- every pair, its own geometry.
        sigma = target.cross_section(
            highlight,
            inbound.direction.unsqueeze(1),
            outbound.launch_direction.unsqueeze(0),
            freqs_khz,
        )  # [n_in, n_out, B]
        amp_scale = sigma.clamp_min(0.0).sqrt().reshape(n_in * n_out, -1)
    else:
        amp_scale = target.cross_section().sqrt()

    time = (inbound.time.view(-1, 1) + outbound.time.view(1, -1)).reshape(-1)
    phase = (inbound.phase.view(-1, 1) + outbound.phase.view(1, -1)).reshape(-1)
    amplitude = (inbound.amplitude.unsqueeze(1) * outbound.amplitude.unsqueeze(0)
                 ).reshape(n_in * n_out, -1) * amp_scale
    direction = outbound.direction.unsqueeze(0).expand(n_in, n_out, 3).reshape(-1, 3)
    distance = outbound.distance.unsqueeze(0).expand(n_in, n_out).reshape(-1)
    path = (inbound.path_length.view(-1, 1) + outbound.path_length.view(1, -1)).reshape(-1)
    if outbound.launch_direction is None:
        launch = None
    else:
        launch = (outbound.launch_direction.unsqueeze(0)
                  .expand(n_in, n_out, 3).reshape(-1, 3))

    echo = ArrivalSet(time=time, amplitude=amplitude, direction=direction,
                      phase=phase, distance=distance, path_length=path,
                      launch_direction=launch)
    if max_arrivals is not None and echo.n_arrivals > max_arrivals:
        order = echo.amplitude.detach().sum(1).argsort(descending=True)[:max_arrivals]
        order = order[echo.time.detach()[order].argsort()]
        echo = _select(echo, order)
    return echo


def _select(arrivals: ArrivalSet, index: Tensor) -> ArrivalSet:
    """Index every populated field of an :class:`ArrivalSet` at once."""
    return ArrivalSet(*(None if t is None else t[index] for t in arrivals))


def _warn_if_fan_too_coarse(fan: Tensor, arclen: Tensor,
                            sigma_d: float | Tensor) -> None:
    """Warn when the return fan cannot resolve the aperture it is collecting at.

    With a physical ``sigma_d`` the fan has to put rays inside it.  If the rays
    are further apart at the array than the aperture is wide, the extraction is
    sampling the fan rather than the receiver: too few rays land, the level
    depends on how many happened to, and each one's own direction is carried
    into the beamformer as if it were the arrival's.
    """
    with torch.no_grad():
        spacing = float(fan_angular_spacing(fan).median())
        reach = float(arclen[:, -1].median())
        transverse = spacing * reach
        width = float(sigma_d if not torch.is_tensor(sigma_d)
                      else torch.as_tensor(sigma_d).median())
    if transverse > width:
        warnings.warn(
            f"return fan is coarser than the aperture it collects at: rays are "
            f"{transverse:.2f} m apart at {reach:.0f} m but sigma_d is "
            f"{width:.2f} m. The echo's level and its angular extent will both "
            f"be set by the fan rather than by the target -- narrow "
            f"rx_half_angle_deg or raise n_rx_rays until the spacing is at or "
            f"below the aperture.",
            RuntimeWarning, stacklevel=3)


def target_arrivals(
    scene: Scene,
    target: ExtendedTarget,
    tx_directions: Tensor,
    *,
    sigma_d: float | Tensor | None = None,
    sigma_d_factor: float = 1.0,
    phase_centre: Tensor | None = None,
    rx_directions: Tensor | None = None,
    n_rx_rays: int = 3000,
    rx_half_angle_deg: float = 45.0,
    rx_jitter: float = 0.0,
    rx_sigma_d: float | Tensor | None = None,
    return_leg: str = "splat",
    tx_weights: Tensor | None = None,
    max_arrivals_per_leg: int | None = 24,
    max_arrivals: int | None = None,
    trace_kwargs: dict | None = None,
    generator: torch.Generator | None = None,
    **extract_kwargs,
) -> ArrivalSet:
    """Coherent echo arrivals from every highlight of an extended target.

    This is the function an imaging sonar wants: it returns one arrival list at
    the array phase centre covering all highlights, with phase, direction and
    exact aspect-dependent scattering intact, ready for
    :func:`hydropt.beamform.beamform`.

    Cost is **one** transmit trace plus one trace per highlight.  The transmit
    trace is shared because the highlights are just several points to extract
    arrivals at -- the fan that insonifies one insonifies them all.  Each
    highlight then needs its own return trace, because each is a different
    source position.

    Args:
        scene: the scene; its ``source`` is the projector, ``receivers`` the array.
        target: the :class:`hydropt.targets.ExtendedTarget` to render.
        tx_directions: projector launch directions, ``[Nt, 3]``.
        sigma_d: arrival acceptance width (m), as in
            :func:`hydropt.beamform.extract_arrivals`.  **Leave it unset.**  The
            default sizes each leg's splat to that leg's own fan with
            :func:`hydropt.launch.fan_sigma_d`, which is the only way the result
            is independent of how densely you happened to sample.  A scalar is
            correct for one fan density at one range and wrong either side of
            it: the return fan is the trap, because it is spread over a wide
            cone and is therefore far sparser than the transmit fan.  With 400
            rays over a 40 deg cone at 60 m the rays are 3.6 m apart, so a
            ``sigma_d`` of 0.4 m made eight *identical* highlights return
            energies spanning 30 dB -- sampling luck, not physics.
        sigma_d_factor: multiplier on the automatic width.  Larger overlaps
            neighbouring beams more.  Ignored when ``sigma_d`` is given.
        phase_centre: where to extract the returns; defaults to the array centroid.
        rx_directions: return fan, ``[Nr, 3]``, shared by every highlight.  By
            default each highlight aims its own cone at the phase centre.
        n_rx_rays, rx_half_angle_deg: shape of the default per-highlight cone.
        rx_sigma_d: acceptance width for the RETURN leg, in metres.  Leave it
            unset and the leg is sized by the fan's own spacing, like the
            transmit leg -- which is right when a fan samples a field over a
            wide area, and wrong here, because the return leg collects at a
            **fixed physical aperture**: the array.

            Tying the acceptance to the sampling means a coarse fan silently
            enlarges the receiver.  Measured on a 0.31 m array at 250 m: a 45
            degree cone of 420 rays gives a 7.8 degree spacing, so ``sigma_d``
            comes out at 28 m -- ninety times the aperture -- and the leg
            collects every ray that passes within 28 m of the array as though
            it had arrived at it.  Two things follow, and both are artefacts.
            The echo is too strong: narrowing the cone to 2 degrees, where
            ``sigma_d`` falls to 0.30 m and matches the array, drops it by
            6.9 dB.  And the rendered angular size of a target has a floor at
            the fan's angular spacing, because each accepted ray keeps its own
            direction: a 30 m hull subtending 4.8 degrees imaged as 12.6, so
            its 25 m of across-bearing extent read as 54 m.

            Pass the array's aperture, and make the fan fine enough to put rays
            inside it -- the spacing at the target's range must be at or below
            the aperture, which this function checks and warns about.
        return_leg: ``"splat"`` (the default, unchanged) or ``"eigenray"``.

            The splat accepts every return ray passing within ``sigma_d`` of
            the array and gives each its own direction, which makes a target's
            rendered angular size no smaller than the fan's angular spacing --
            a 30 m hull subtending 4.8 degrees at 250 m images as 12.6.
            ``"eigenray"`` solves for the discrete paths instead
            (:mod:`hydropt.eigenray`): one arrival per path, the direction it
            actually arrives from, spreading from the ray tube's own
            divergence, and no ``sigma_d`` at all.

            Not the default yet, because it moves every target level in the
            package and that change wants isolating rather than smuggling.
        rx_jitter: randomise the default return fan by this fraction of a
            sample spacing.  **Needed for ``generator`` to do anything**: a
            Fibonacci cone is deterministic, so without jitter every seed gives
            a bit-identical fan and the same answer.  Leave it at 0 for a
            repeatable render; set it to 1 when you want *independent
            realisations* of the same physical scene -- fitting a model to a
            synthetic measurement is the case that needs it, because sharing the
            fan between the two makes the inversion an inverse crime and hides
            how much of the answer the sampling is setting.
        tx_weights: per-ray transmit weights, e.g. projector directivity.
        max_arrivals_per_leg: cap each leg before pairing.  The pair count is a
            product, so capping the legs is far more effective than capping the
            result -- and a dense fan's extra arrivals are near-duplicates.
        max_arrivals: cap the combined result.
        trace_kwargs: forwarded to the tracer.
        generator: RNG for the return fans.  Only has an effect when
            ``rx_jitter`` is non-zero -- see above.
        extract_kwargs: forwarded to ``extract_arrivals`` for both legs.

    Returns:
        One :class:`hydropt.beamform.ArrivalSet` for the whole target,
        differentiable in the target's position, orientation, highlight layout
        and pattern parameters, and in every scene parameter.
    """
    if phase_centre is None:
        phase_centre = scene.receivers.reshape(-1, 3).mean(0)
    auto_sigma_d = sigma_d is None
    kw = dict(extract_kwargs)
    tkw = trace_kwargs or {}
    freqs = scene.freqs_khz
    world = target.world_positions()

    # One transmit trace for every highlight: they are just several points to
    # ask the same bundle about.
    tx_result = trace(scene, tx_directions, **tkw)
    tx_kw = dict(kw)
    tx_kw["sigma_d"] = (fan_sigma_d(tx_directions, tx_result.arclen,
                                    factor=sigma_d_factor)
                        if auto_sigma_d else sigma_d)

    # Inbound first, so a highlight the projector never reached costs no return
    # trace at all.
    inbound_by_highlight: dict[int, ArrivalSet] = {}
    for i in range(target.n_highlights):
        inbound = extract_arrivals(tx_result, world[i], freqs,
                                   ray_weights=tx_weights,
                                   max_arrivals=max_arrivals_per_leg, **tx_kw)
        if inbound.n_arrivals > 0:
            inbound_by_highlight[i] = inbound

    # One batched return trace for every lit highlight, not one per highlight.
    # `trace` reads the source through `scene.source_position()` and immediately
    # broadcasts it against the directions, so handing it a **[R, 3]** source --
    # one row per ray -- already works: rays launched from different highlights
    # integrate together in a single pass.  That is the same arithmetic (verified
    # bit-identical) with one Python loop over steps instead of N, which is most
    # of the cost at these fan sizes: 8 highlights went from 7.6 s to 1.6 s.
    if return_leg not in ("splat", "eigenray"):
        raise ValueError(f"return_leg must be 'splat' or 'eigenray', got "
                         f"{return_leg!r}")

    lit = list(inbound_by_highlight)
    parts: list[ArrivalSet] = []
    if lit and return_leg == "eigenray":
        from .eigenray import eigenray_arrivals
        for i in lit:
            outbound = eigenray_arrivals(
                scene, world[i], phase_centre, freqs,
                bracket_rays=n_rx_rays,
                bracket_half_angle_deg=rx_half_angle_deg,
                trace_kwargs=tkw)
            if outbound.n_arrivals == 0:
                continue
            parts.append(compose_arrivals(inbound_by_highlight[i], outbound,
                                          target, highlight=i, freqs_khz=freqs))
    elif lit:
        fans = []
        for i in lit:
            if rx_directions is None:
                axis = (phase_centre.detach().reshape(3)
                        - world[i].detach().reshape(3))
                fans.append(fibonacci_cone(n_rx_rays, axis, rx_half_angle_deg,
                                           jitter=rx_jitter,
                                           generator=generator))
            else:
                fans.append(rx_directions)
        per_ray = torch.cat(fans, dim=0)
        sources = torch.cat([world[i].reshape(1, 3).expand(f.shape[0], 3)
                             for i, f in zip(lit, fans)], dim=0)
        batched = trace(_RelocatedScene(scene, sources), per_ray, **tkw)

        spacings: dict[int, Tensor] = {}
        start = 0
        for i, fan in zip(lit, fans):
            stop = start + fan.shape[0]
            leg = TraceResult(*(t[start:stop] for t in batched))
            leg_kw = dict(kw)
            if rx_sigma_d is not None:
                leg_kw["sigma_d"] = rx_sigma_d
                _warn_if_fan_too_coarse(fan, leg.arclen, rx_sigma_d)
            elif auto_sigma_d:
                # Measured on this fan alone.  The fans are concatenated for the
                # trace but every one is a cone aimed at the same phase centre,
                # so across the concatenation each ray has a near-duplicate in
                # every other highlight's fan and the spacing would read as ~0.
                # Cached by identity, because a caller-supplied `rx_directions`
                # is the *same* tensor for every highlight and the search is
                # quadratic in the fan size.
                key = id(fan)
                if key not in spacings:
                    spacings[key] = fan_angular_spacing(fan)
                leg_kw["sigma_d"] = fan_sigma_d(fan, leg.arclen,
                                                factor=sigma_d_factor,
                                                spacing=spacings[key])
            else:
                leg_kw["sigma_d"] = sigma_d
            start = stop
            outbound = extract_arrivals(leg, phase_centre, freqs,
                                        max_arrivals=max_arrivals_per_leg,
                                        **leg_kw)
            if outbound.n_arrivals == 0:
                continue
            parts.append(compose_arrivals(inbound_by_highlight[i], outbound,
                                          target, highlight=i, freqs_khz=freqs))

    if not parts:
        z = torch.zeros(0, dtype=world.dtype, device=world.device)
        z3 = torch.zeros(0, 3, dtype=world.dtype, device=world.device)
        return ArrivalSet(z, torch.zeros(0, int(freqs.shape[0]), dtype=world.dtype,
                                         device=world.device), z3, z, z, z, z3)

    echo = ArrivalSet(*(
        None if parts[0][f] is None else torch.cat([p[f] for p in parts], dim=0)
        for f in range(len(parts[0]))
    ))
    order = echo.time.detach().argsort()
    echo = _select(echo, order)
    if max_arrivals is not None and echo.n_arrivals > max_arrivals:
        pick = echo.amplitude.detach().sum(1).argsort(descending=True)[:max_arrivals]
        echo = _select(echo, pick[echo.time.detach()[pick].argsort()])
    return echo


def render_extended_echo(
    scene: Scene,
    target: ExtendedTarget,
    tx_directions: Tensor,
    echo_time_grid: Tensor,
    *,
    sigma_d: float,
    sigma_t: float,
    rx_directions: Tensor | None = None,
    n_rx_rays: int = 3000,
    rx_half_angle_deg: float = 45.0,
    tx_weights: Tensor | None = None,
    ray_chunk: int = 0,
    trace_kwargs: dict | None = None,
    generator: torch.Generator | None = None,
    **splat_kwargs,
) -> EchoResult:
    """Energy-domain two-way echo from an extended target, summed incoherently.

    Cost is one inbound render -- the highlights are just several receive points
    for the same transmit bundle -- plus one outbound trace per highlight, and
    the legs still compose by convolution, so this stays linear in highlights
    rather than quadratic in arrivals.

    **The aspect used here is the straight-line one.** ``sigma`` is evaluated for
    each highlight at the geometry projector -> highlight -> array centroid,
    ignoring which multipath actually delivered the energy.  That is exact for an
    isotropic pattern and an approximation for any other; the approximation is
    unavoidable on this path, because an energy render has already discarded the
    pairing between inbound and outbound directions that a bistatic
    cross-section needs.  When the aspect matters -- which is whenever you are
    imaging rather than budgeting energy -- use :func:`target_arrivals`, where it
    is exact.

    Highlights are summed in **energy**, so this response has no interference
    between them.  That too is what the coherent path is for.

    Args:
        rx_directions: return fan, ``[Nr, 3]``, shared by every highlight.  By
            default each highlight gets its own Fibonacci cone aimed at the array
            centroid, which is what you want; pass this when you need the fan
            fixed, and then it is on you to cover every highlight.
        n_rx_rays, rx_half_angle_deg: shape of the default per-highlight cone.
    """
    grid = echo_time_grid
    n_echo = int(grid.shape[0])
    if n_echo < 2:
        raise ValueError("echo_time_grid needs at least two bins")
    dt = (grid[-1] - grid[0]) / (n_echo - 1)
    dt_f = float(dt)
    if float(grid[0]) < 0.0:
        raise ValueError("echo_time_grid must start at a non-negative time")

    n_leg = int(math.ceil(float(grid[-1]) / dt_f)) + 1
    leg_grid = torch.arange(n_leg, dtype=grid.dtype, device=grid.device) * dt
    kw = dict(sigma_d=sigma_d, sigma_t=sigma_t / math.sqrt(2.0),
              ray_chunk=ray_chunk, **splat_kwargs)
    tkw = trace_kwargs or {}
    freqs = scene.freqs_khz
    world = target.world_positions()
    centre = scene.receivers.reshape(-1, 3).mean(0)

    # Leg 1, once: all highlights are receive points of the same bundle.
    inbound = splat_etc(trace(scene, tx_directions, **tkw), world, leg_grid,
                        freqs, ray_weights=tx_weights, **kw)  # [N, B, T]

    n_fft = 1 << int(math.ceil(math.log2(max(2 * n_leg - 1, 2))))
    spec_in = torch.fft.rfft(inbound, n=n_fft)

    total = None
    outbound_sum = None
    for i in range(target.n_highlights):
        pos_i = world[i]
        if rx_directions is None:
            axis = centre.detach().reshape(3) - pos_i.detach().reshape(3)
            rx_dirs = fibonacci_cone(n_rx_rays, axis, rx_half_angle_deg,
                                     generator=generator)
        else:
            rx_dirs = rx_directions
        outbound = splat_etc(trace(_RelocatedScene(scene, pos_i), rx_dirs, **tkw),
                             scene.receivers, leg_grid, freqs, **kw)
        # Straight-line aspect: projector -> highlight -> array centroid.
        inc = pos_i - scene.source_position().reshape(3)
        inc = inc / inc.norm().clamp_min(1e-30)
        sca = centre.reshape(3) - pos_i
        sca = sca / sca.norm().clamp_min(1e-30)
        sigma = target.cross_section(i, inc, sca, freqs).reshape(1, -1, 1)

        spec = spec_in[i : i + 1] * torch.fft.rfft(outbound, n=n_fft)
        full = torch.fft.irfft(spec, n=n_fft)[..., : 2 * n_leg - 1] * dt * sigma
        total = full if total is None else total + full
        outbound_sum = outbound if outbound_sum is None else outbound_sum + outbound

    offset = int(round(float(grid[0]) / dt_f))
    if offset + n_echo > total.shape[-1]:
        total = torch.nn.functional.pad(total, (0, offset + n_echo - total.shape[-1]))
    return EchoResult(etc=total[..., offset : offset + n_echo], inbound=inbound,
                      outbound=outbound_sum, leg_time_grid=leg_grid)
