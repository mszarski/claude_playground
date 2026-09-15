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
