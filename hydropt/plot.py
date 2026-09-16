"""Matplotlib views of a traced scene: ray fans, projections and ETCs.

All functions take ready-made tensors and return a Matplotlib ``Figure``, so they
compose with whatever layout an example wants.  Plotly is optional and only used
by :func:`plotly_rays`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .tracer import TraceResult

__all__ = [
    "plot_rays_3d",
    "plot_ray_projections",
    "plot_etc",
    "plot_profile",
    "plot_bathymetry",
    "plot_fit_history",
    "plot_fls_sector",
    "plotly_rays",
]


def _np(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _live_segments(result: TraceResult, stride: int = 1,
                   max_rays: int | None = None) -> list[np.ndarray]:
    """Per-ray path arrays, truncated at the point each ray was retired.

    When ``max_rays`` is given the subset is drawn with a fixed seed rather than
    by striding: a fan is usually laid out elevation-major, so a stride would
    keep re-picking the same few azimuths.
    """
    pos = _np(result.pos)
    alive = _np(result.alive)
    idx = np.arange(pos.shape[0])
    if max_rays is not None and pos.shape[0] > max_rays:
        idx = np.sort(np.random.default_rng(0).choice(pos.shape[0], max_rays, replace=False))
    out = []
    for i in idx:
        live = np.nonzero(alive[i] > 0)[0]
        end = int(live[-1]) + 1 if live.size else 1
        out.append(pos[i, :end:stride])
    return out


def plot_rays_3d(result: TraceResult, *, receivers: Tensor | None = None,
                 source: Tensor | None = None, stride: int = 4,
                 max_rays: int = 200, figsize=(10, 7), title: str = "Ray fan"):
    """3-D ray fan.  ``z`` is drawn increasing downward, as depth."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for p in _live_segments(result, stride, max_rays):
        ax.plot(p[:, 0] / 1e3, p[:, 1] / 1e3, p[:, 2], lw=0.4, alpha=0.6, color="#1f6f8b")
    if source is not None:
        s = _np(source).reshape(3)
        ax.scatter([s[0] / 1e3], [s[1] / 1e3], [s[2]], c="#d94801", s=55, marker="*",
                   label="source", depthshade=False)
    if receivers is not None:
        r = _np(receivers).reshape(-1, 3)
        ax.scatter(r[:, 0] / 1e3, r[:, 1] / 1e3, r[:, 2], c="#222222", s=18,
                   marker="v", label="receivers", depthshade=False)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("depth (m)")
    ax.invert_zaxis()
    ax.set_title(title)
    if source is not None or receivers is not None:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_ray_projections(result: TraceResult, *, receivers: Tensor | None = None,
                         source: Tensor | None = None, bottom=None,
                         stride: int = 4, max_rays: int = 300, figsize=(11, 7),
                         title: str = "Ray paths"):
    """Side (x-z) and top-down (x-y) projections, stacked."""
    import matplotlib.pyplot as plt

    fig, (ax_side, ax_top) = plt.subplots(2, 1, figsize=figsize)
    paths = _live_segments(result, stride, max_rays)
    for p in paths:
        ax_side.plot(p[:, 0] / 1e3, p[:, 2], lw=0.35, alpha=0.55, color="#1f6f8b")
        ax_top.plot(p[:, 0] / 1e3, p[:, 1] / 1e3, lw=0.35, alpha=0.55, color="#1f6f8b")

    if bottom is not None:
        xs = np.linspace(min(p[:, 0].min() for p in paths),
                         max(p[:, 0].max() for p in paths), 400)
        xy = torch.stack((torch.as_tensor(xs, dtype=torch.get_default_dtype()),
                          torch.zeros(len(xs))), dim=-1)
        with torch.no_grad():
            h = _np(bottom.height(xy))
        ax_side.plot(xs / 1e3, h, color="#7f5539", lw=1.6, label="bathymetry")
        ax_side.fill_between(xs / 1e3, h, h.max() * 1.08 + 1.0, color="#d8c3a5", alpha=0.5)
        ax_side.legend(loc="lower right", fontsize=8)

    for ax, ylab in ((ax_side, "depth (m)"), (ax_top, "y (km)")):
        ax.set_xlabel("x (km)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.4)
    ax_side.invert_yaxis()
    ax_side.set_title(f"{title} -- side view (x-z)")
    ax_top.set_title("top-down view (x-y)")

    if source is not None:
        s = _np(source).reshape(3)
        ax_side.plot([s[0] / 1e3], [s[2]], "*", c="#d94801", ms=13)
        ax_top.plot([s[0] / 1e3], [s[1] / 1e3], "*", c="#d94801", ms=13)
    if receivers is not None:
        r = _np(receivers).reshape(-1, 3)
        ax_side.plot(r[:, 0] / 1e3, r[:, 2], "v", c="#222222", ms=5)
        ax_top.plot(r[:, 0] / 1e3, r[:, 1] / 1e3, "v", c="#222222", ms=5)
    fig.tight_layout()
    return fig


def plot_etc(etc: Tensor, time_grid: Tensor, *, freqs_khz: Tensor | None = None,
             receiver_labels: Sequence[str] | None = None, db: bool = True,
             dynamic_range: float = 60.0, figsize=(11, None),
             title: str = "Energy-time curves", compare: Tensor | None = None,
             compare_label: str = "target"):
    """One panel per receiver, one line per frequency band.

    ``db=True`` plots ``10 log10`` of the energy with a floor ``dynamic_range``
    below the peak, which is the only way the weak late arrivals are visible at
    all next to the direct path.
    """
    import matplotlib.pyplot as plt

    etc_np = _np(etc)
    t = _np(time_grid)
    n_recv, n_band, _ = etc_np.shape
    height = figsize[1] if figsize[1] is not None else max(2.1 * n_recv, 3.0)
    fig, axes = plt.subplots(n_recv, 1, figsize=(figsize[0], height),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    peak = max(etc_np.max(), 1e-300)
    cmap = plt.get_cmap("viridis")

    def to_db(a):
        return 10.0 * np.log10(np.maximum(a, peak * 10 ** (-dynamic_range / 10.0) * 1e-3))

    cmp_np = None if compare is None else _np(compare)
    for i, ax in enumerate(axes):
        for b in range(n_band):
            y = etc_np[i, b]
            label = None
            if freqs_khz is not None:
                label = f"{float(freqs_khz[b]):.3g} kHz"
            ax.plot(t, to_db(y) if db else y, lw=1.0,
                    color=cmap(b / max(n_band - 1, 1)), label=label)
        if cmp_np is not None:
            for b in range(n_band):
                y = cmp_np[i, b]
                ax.plot(t, to_db(y) if db else y, lw=0.9, ls="--", color="#bbbbbb",
                        label=compare_label if b == 0 else None, zorder=0)
        if db:
            ax.set_ylim(10 * np.log10(peak) - dynamic_range, 10 * np.log10(peak) + 3)
        lbl = receiver_labels[i] if receiver_labels else f"receiver {i}"
        ax.set_ylabel("dB re 1" if db else "energy")
        ax.text(0.99, 0.92, lbl, transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_title(title)
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def plot_profile(profiles: dict[str, tuple[Tensor, Tensor]], *, figsize=(5, 6),
                 title: str = "Sound-speed profile"):
    """Overlay ``{label: (c, z)}`` sound-speed profiles, depth increasing downward."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    styles = ["-", "--", ":", "-."]
    for k, (label, (c, z)) in enumerate(profiles.items()):
        ax.plot(_np(c), _np(z), styles[k % len(styles)], lw=1.6, label=label)
    ax.set_xlabel("sound speed (m/s)")
    ax.set_ylabel("depth (m)")
    ax.invert_yaxis()
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_bathymetry(fields: dict[str, Tensor], *, extent=None, figsize=(12, 4),
                    title: str = "Bathymetry", receivers: Tensor | None = None):
    """Side-by-side height-field images sharing one colour scale."""
    import matplotlib.pyplot as plt

    arrays = {k: _np(v) for k, v in fields.items()}
    vmin = min(a.min() for a in arrays.values())
    vmax = max(a.max() for a in arrays.values())
    fig, axes = plt.subplots(1, len(arrays), figsize=figsize, squeeze=False)
    for ax, (label, a) in zip(axes[0], arrays.items()):
        im = ax.imshow(a, origin="lower", extent=extent, aspect="auto",
                       cmap="terrain_r", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        if receivers is not None:
            r = _np(receivers).reshape(-1, 3)
            ax.plot(r[:, 0] / 1e3, r[:, 1] / 1e3, "v", c="black", ms=4)
        fig.colorbar(im, ax=ax, label="depth (m)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_fit_history(history, *, param_names: Sequence[str] | None = None,
                     truth: dict[str, Sequence[float]] | None = None,
                     figsize=(11, 4), title: str = "Inversion"):
    """Loss curve plus parameter tracks, with optional dashed true values."""
    import matplotlib.pyplot as plt

    names = list(param_names or history.params.keys())
    fig, axes = plt.subplots(1, 1 + len(names), figsize=figsize, squeeze=False)
    axes = axes[0]
    axes[0].semilogy(history.loss, lw=1.4, color="#1f6f8b")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_title("objective")
    axes[0].grid(alpha=0.25, lw=0.4)

    for ax, name in zip(axes[1:], names):
        track = np.asarray(history.params[name])
        for j in range(track.shape[1]):
            ax.plot(track[:, j], lw=1.2)
        if truth and name in truth:
            for value in np.atleast_1d(truth[name]):
                ax.axhline(float(value), ls="--", lw=0.9, color="#d94801")
        ax.set_xlabel("iteration")
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plotly_rays(result: TraceResult, *, receivers: Tensor | None = None,
                source: Tensor | None = None, stride: int = 6, max_rays: int = 150):
    """Interactive 3-D ray fan.  Requires the optional ``plotly`` extra."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plotly_rays needs the optional 'plotly' extra") from exc

    traces = [
        go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                     line=dict(width=1.4, color="#1f6f8b"), showlegend=False,
                     hoverinfo="skip")
        for p in _live_segments(result, stride, max_rays)
    ]
    if source is not None:
        s = _np(source).reshape(3)
        traces.append(go.Scatter3d(x=[s[0]], y=[s[1]], z=[s[2]], mode="markers",
                                   marker=dict(size=6, color="#d94801"), name="source"))
    if receivers is not None:
        r = _np(receivers).reshape(-1, 3)
        traces.append(go.Scatter3d(x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="markers",
                                   marker=dict(size=4, color="black"), name="receivers"))
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(xaxis_title="x (m)", yaxis_title="y (m)",
                                 zaxis_title="depth (m)",
                                 zaxis=dict(autorange="reversed")))
    return fig


def plot_fls_sector(power: Tensor, bearings: Tensor, ranges: Tensor, *,
                    dynamic_range: float = 24.0, sound_speed: float = 1500.0,
                    reference: float | None = None,
                    ring_step: float | None = None, cmap: str = "afmhot",
                    figsize=(8.0, 7.2), title: str = "Forward-looking sonar",
                    overlays: Sequence[tuple] = (), ax=None):
    """The sector display a forward-looking sonar actually paints.

    The wedge, apex at the vehicle, out to the range limit -- range rings and
    bearing spokes over a dark ground, which is what an operator reads.

    **Drawn on the beamformer's own grid, not resampled onto a raster.** The
    bearing-range mesh maps to ``x = R cos B``, ``y = R sin B`` exactly, so
    ``pcolormesh`` paints the true wedge with no interpolation and, more to the
    point, no cells outside the swath: a rectangular grid has to pad the corners
    with something, and padding a sonar image with zeros invents dark water the
    sonar never looked at.  Cells here grow with range the way the beams do,
    which is also the honest thing to show -- the far field really is sampled
    more coarsely than the near.

    Args:
        power: ``[bearings, ranges]`` beam power, linear (not dB).  A full
            ``[bearings, bands, time]`` image can be passed after selecting a
            band, e.g. ``image[:, 0]``.
        bearings: ``[bearings]`` in **degrees**, as
            :func:`hydropt.beamform.azimuth_steering` returns.
        ranges: ``[ranges]`` in metres, or the two-way time grid -- times are
            detected by magnitude and converted with ``sound_speed``.
        dynamic_range: dB below the peak to show.  A sonar display is a
            deliberately shallow window; 20-30 dB is typical, and showing 60
            turns the picture into reverberation.
        reference: normalise to this level instead of the image's own peak.
            Two panels each normalised to their own peak cannot be compared --
            a change that lowers the whole image by 3 dB looks identical --
            so pass a common reference when the comparison is the point.
        ring_step: metres between range rings; chosen automatically if omitted.
        overlays: ``(x, y, style, label)`` tuples drawn over the wedge, with
            ``x`` across track and ``y`` along track in metres.
        ax: draw into an existing axis instead of making a figure.

    Returns the ``Figure``.
    """
    import matplotlib.pyplot as plt

    p = _np(power)
    if p.ndim != 2:
        raise ValueError(f"power must be [bearings, ranges], got {p.shape}")
    b = _np(bearings).reshape(-1)
    r = _np(ranges).reshape(-1)
    if r.max() < 1.0:  # a two-way time grid, not metres
        r = r * sound_speed / 2.0
    if p.shape != (b.size, r.size):
        raise ValueError(f"power is {p.shape}, but got {b.size} bearings and "
                         f"{r.size} ranges")

    peak = max(float(p.max()) if reference is None else float(reference), 1e-300)
    db = 10.0 * np.log10(np.maximum(p, peak * 10 ** (-dynamic_range / 10.0)) / peak)

    # Cell EDGES, not centres: with a curved mesh matplotlib cannot infer them,
    # and each beam cell should be drawn at the extent it actually covers --
    # which grows with range, as the beams do.
    def edges(v):
        mid = 0.5 * (v[1:] + v[:-1])
        return np.concatenate(([v[0] - (mid[0] - v[0])], mid,
                               [v[-1] + (v[-1] - mid[-1])]))

    B, R = np.meshgrid(np.radians(edges(b)), edges(r), indexing="ij")
    X, Y = R * np.cos(B), R * np.sin(B)      # x along track, y across track

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_facecolor("#05070c")
    mesh = ax.pcolormesh(Y, X, db, cmap=cmap, vmin=-dynamic_range, vmax=0.0,
                         shading="flat", rasterized=True)

    r_max = float(r.max())
    if ring_step is None:
        ring_step = max(10.0 ** np.floor(np.log10(r_max / 3.0)), 1.0)
        while r_max / ring_step > 6:
            ring_step *= 2.0
    edge = np.radians(np.array([b.min(), b.max()]))
    span = np.radians(np.linspace(b.min(), b.max(), 200))
    ring = ring_step
    while ring <= r_max + 1e-9:
        ax.plot(ring * np.sin(span), ring * np.cos(span), color="#7fa8c8",
                lw=0.6, alpha=0.35, zorder=3)
        ax.annotate(f"{ring:g} m", (ring * np.sin(edge[1]), ring * np.cos(edge[1])),
                    color="#7fa8c8", fontsize=7, alpha=0.8, zorder=4,
                    xytext=(3, 2), textcoords="offset points")
        ring += ring_step
    for spoke in np.arange(np.ceil(b.min() / 15.0) * 15.0, b.max() + 1e-9, 15.0):
        a = np.radians(spoke)
        ax.plot([0, r_max * np.sin(a)], [0, r_max * np.cos(a)], color="#7fa8c8",
                lw=0.5, alpha=0.22, zorder=3)
        ax.annotate(f"{spoke:+.0f}", (r_max * np.sin(a) * 1.03,
                                      r_max * np.cos(a) * 1.03),
                    color="#7fa8c8", fontsize=7, alpha=0.75, ha="center",
                    zorder=4)
    for a in edge:  # the swath boundary
        ax.plot([0, r_max * np.sin(a)], [0, r_max * np.cos(a)], color="#7fa8c8",
                lw=0.9, alpha=0.5, zorder=3)

    for item in overlays:
        x, y, style, label = (list(item) + [None])[:4]
        ax.plot(np.atleast_1d(x), np.atleast_1d(y), style, zorder=5,
                **({} if label is None else {"label": label}))
    ax.plot([0], [0], "^", color="#5ff0c0", ms=10, mec="k", mew=0.6, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlabel("across track (m)")
    ax.set_ylabel("along track (m)")
    ax.set_title(title)
    ax.set_xlim(r_max * np.sin(edge).min() * 1.08, r_max * np.sin(edge).max() * 1.08)
    ax.set_ylim(-0.04 * r_max, r_max * 1.1)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("dB re peak")
    if any(len(i) > 3 and i[3] for i in overlays):
        ax.legend(loc="lower right", fontsize=8, framealpha=0.3)
    return fig
