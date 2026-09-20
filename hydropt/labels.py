"""Labels from the forward pass: a box, a mask and a class for every target.

A simulator knows what is in its picture.  The renderer forms every
target's beams on their own before adding them to the reverberation's
(:class:`~hydropt.sequence.PictureRenderer`), and what a detector could
see of a target is exactly where the target's own energy stands over
everything else in the cell.  So a label needs no detector, no threshold
tuned by eye and no hand-drawn box: it is read off the fields.

For a target ``A`` with beams ``b_A`` in a picture whose field is
``b = b_rev + sum_k b_k + noise``:

* its **mask** is the set of cells where ``|b_A|^2 > margin * |b - b_A|^2``
  -- where ``A`` dominates everything else, the other targets and the
  noise included, by ``margin_db`` (3 dB by default: the cell is more than
  half ``A``) -- and within ``dynamic_range_db`` (35) of ``A``'s own
  brightest cell, so that the sidelobe ring a strong echo throws round the
  swath at its range (43 dB down under a Hamming window, and still over
  the reverberation when the echo is 48 dB over it) is not the target;
  and, for a thing with geometry, within its own extent plus two beams
  and a few range cells (``polar_gate``), since energy of ``A`` further
  away than that is leakage, not ``A``.  That is the instance segmentation
  of ``A`` as the picture shows it, in beams and range bins
  (``polar_mask``) and, through the example's own resampling, in metres
  (``mask``);
* its **signal box** is the bounding box of that mask, in beams and bins
  (``polar_box``, tight for a spoke, which is one beam wide and every bin
  long) and in metres (``box_m``, axis-aligned: the box round the masked
  cells' own corners, each cell a beam by a range bin, so a one-cell mask
  still has a box);
* its **centroid** and **peak** are the energy-weighted centre and the
  brightest cell of ``|b_A|^2`` over the mask, in metres from the cells'
  bearings and ranges, and its **contrast** is ``|b_A|^2`` over the rest
  at the peak, in dB;
* its **geometry box** is what the object physically spans -- every vertex
  of a mesh, every highlight of an extended target, carried into the world
  -- dilated by one beam width and one range cell, since that is the least
  the sonar smears a point over.  It does not depend on visibility: a
  shadowed hull or a faint school has a geometry box and an empty mask.

The two boxes answer different questions for a training set.  The signal
box is what the sonar shows and what a detector can be asked to find; the
geometry box is where the object is, and the pair says how much of it the
picture reveals (``visible`` is a non-empty mask with the peak over the
margin).  Both are per frame, so a sequence labels itself as it renders.
An emission (a propeller's spoke, :mod:`hydropt.emission`) is labelled the
same way from its own beams, with no geometry.

Boxes are ``(x0, y0, x1, y1)`` in the picture's metres with ``x`` forward
and ``y`` to port, polar boxes ``(beam0, beam1, bin0, bin1)`` inclusive;
:meth:`Label.to_dict` gives a JSON-ready record and :func:`draw_labels`
puts the boxes on a matplotlib axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
from torch import Tensor

__all__ = ["Label", "signal_mask", "polar_box", "polar_gate", "world_geometry", "geometry_box",
           "label_from_beams", "draw_labels"]


@dataclass
class Label:
    name: str
    kind: str                                   # "target" or "emission"
    visible: bool
    contrast_db: float
    n_cells: int                                # of the polar mask
    polar_box: tuple[int, int, int, int] | None  # beam0, beam1, bin0, bin1 (inclusive)
    box_m: tuple[float, float, float, float] | None      # x0, y0, x1, y1 (metres)
    centroid_m: tuple[float, float] | None
    peak_m: tuple[float, float] | None
    geometry_box_m: tuple[float, float, float, float] | None
    polar_mask: Tensor | None = field(default=None, repr=False)
    mask: Tensor | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """A JSON-ready record (the masks left out)."""
        r = lambda v: None if v is None else [round(float(x), 3) for x in v]
        return dict(name=self.name, kind=self.kind, visible=self.visible,
                    contrast_db=round(self.contrast_db, 2), n_cells=self.n_cells,
                    polar_box=None if self.polar_box is None else [int(v) for v in self.polar_box],
                    box_m=r(self.box_m), centroid_m=r(self.centroid_m), peak_m=r(self.peak_m),
                    geometry_box_m=r(self.geometry_box_m))


def signal_mask(own: Tensor, rest: Tensor, margin_db: float = 3.0,
                dynamic_range_db: float | None = 35.0, gate: Tensor | None = None) -> Tensor:
    """Cells where ``own`` power stands over ``rest`` by ``margin_db``: ``[beams, bins]`` bool.

    Both are ``[beams, bands, bins]`` powers; the bands are summed.  With
    ``dynamic_range_db`` a cell must also be within that of ``own``'s
    brightest cell (the sidelobes of a strong echo are not it); ``gate`` is
    a ``[beams, bins]`` bool the mask is confined to (:func:`polar_gate`).
    """
    o = own.sum(dim=1) if own.ndim == 3 else own
    r = rest.sum(dim=1) if rest.ndim == 3 else rest
    m = o > r * 10.0 ** (float(margin_db) / 10.0)
    if gate is not None:
        m = m & gate
    if dynamic_range_db is not None and bool(m.any()):
        peak = float((o * m).max())
        m = m & (o > peak * 10.0 ** (-float(dynamic_range_db) / 10.0))
    return m


def polar_gate(points: Tensor, bearings_deg: Tensor, ranges_m: Tensor, *, beam_deg: float,
               range_m: float, beams_margin: float = 2.0, extra_range_m: float = 5.0) -> Tensor:
    """``[beams, bins]`` bool: the beams and bins a thing's world ``points`` can light.

    Its bearings widened by ``beams_margin`` beam widths either side, its
    ranges by ``beams_margin`` range cells short and by that plus
    ``extra_range_m`` long -- the surface- and bottom-image paths of every
    point arrive a little later than its direct one, never earlier.
    """
    xy = points[:, :2].detach()
    b = torch.rad2deg(torch.atan2(xy[:, 1], xy[:, 0]))
    r = xy.norm(dim=-1)
    db_ = beams_margin * float(beam_deg)
    dr = beams_margin * float(range_m)
    ok_b = (bearings_deg >= float(b.min()) - db_) & (bearings_deg <= float(b.max()) + db_)
    ok_r = (ranges_m >= float(r.min()) - dr) & (ranges_m <= float(r.max()) + dr + float(extra_range_m))
    return ok_b.reshape(-1, 1) & ok_r.reshape(1, -1)


def polar_box(mask: Tensor) -> tuple[int, int, int, int] | None:
    """``(beam0, beam1, bin0, bin1)`` of a ``[beams, bins]`` mask, or ``None`` if empty."""
    beams = mask.any(dim=1).nonzero().reshape(-1)
    bins = mask.any(dim=0).nonzero().reshape(-1)
    if beams.numel() == 0:
        return None
    return int(beams.min()), int(beams.max()), int(bins.min()), int(bins.max())


def world_geometry(target) -> Tensor:
    """``[N, 3]`` world points spanning the target: mesh vertices, or its highlights."""
    with torch.no_grad():
        for p in getattr(target, "patterns", []):
            verts = getattr(p, "vertices", None)
            if verts is not None:
                return target.position.reshape(1, 3) + verts @ target.rotation().T
        return target.world_positions()


def geometry_box(points: Tensor, *, beam_deg: float, range_m: float
                 ) -> tuple[float, float, float, float]:
    """The axis-aligned box in metres round world ``points``, dilated by the resolution.

    A point at range ``R`` is smeared over ``beam_deg`` across and ``range_m``
    along, so the box grows by half of each about every point's own range.
    """
    xy = points[:, :2].detach()
    r = xy.norm(dim=-1)
    d = 0.5 * (math.radians(float(beam_deg)) * r + float(range_m))
    lo = (xy - d.unsqueeze(-1)).min(dim=0).values
    hi = (xy + d.unsqueeze(-1)).max(dim=0).values
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def label_from_beams(name: str, kind: str, own: Tensor, rest: Tensor, *,
                     margin_db: float = 3.0, dynamic_range_db: float | None = 35.0,
                     to_cartesian: Callable | None = None,
                     geometry_points: Tensor | None = None, beam_deg: float = 0.0,
                     range_m: float = 0.0, bearings_deg: Tensor | None = None,
                     ranges_m: Tensor | None = None, min_cells: int = 2,
                     keep_masks: bool = True) -> Label:
    """The label of one thing from its own power and everything else's.

    ``own`` and ``rest`` are ``[beams, bands, bins]`` powers (calibrated or
    not: only their ratio matters).  With the picture's ``bearings_deg``
    (per beam) and ``ranges_m`` (per bin) the metric box, centroid and
    peak come from the masked cells' own geometry; without them the label
    is polar only.  ``to_cartesian(image) -> (cart, gx, gy)``, the
    example's resampling, gives the Cartesian mask (``mask``) on its grid.
    With ``geometry_points`` the mask is gated to the thing's own extent
    (:func:`polar_gate`).  ``visible`` needs at least ``min_cells`` cells:
    one cell over the margin is a noise fluctuation, not a thing seen.
    """
    gate = None
    if (geometry_points is not None and geometry_points.numel()
            and bearings_deg is not None and ranges_m is not None):
        gate = polar_gate(geometry_points, bearings_deg, ranges_m, beam_deg=beam_deg,
                          range_m=range_m)
    pm = signal_mask(own, rest, margin_db, dynamic_range_db, gate)
    pbox = polar_box(pm)
    n_cells = int(pm.sum())
    o = own.sum(dim=1) if own.ndim == 3 else own
    r = rest.sum(dim=1) if rest.ndim == 3 else rest
    if n_cells:
        k = int((o * pm).argmax())
        contrast = 10.0 * math.log10(float(o.reshape(-1)[k]) / max(float(r.reshape(-1)[k]), 1e-300))
    else:
        k = int(o.argmax())
        contrast = 10.0 * math.log10(max(float(o.reshape(-1)[k]), 1e-300)
                                     / max(float(r.reshape(-1)[k]), 1e-300))
    visible = n_cells >= int(min_cells) and contrast > margin_db
    box_m = centroid = peak = None
    cmask = None
    if n_cells and bearings_deg is not None and ranges_m is not None:
        bb, kk = pm.nonzero(as_tuple=True)
        bd = bearings_deg.to(o.dtype); rg = ranges_m.to(o.dtype)
        db_ = float((bd[1:] - bd[:-1]).abs().median()) if bd.numel() > 1 else 0.0
        dr_ = float((rg[1:] - rg[:-1]).abs().median()) if rg.numel() > 1 else 0.0
        bc, rc = bd[bb], rg[kk]
        # the box round the cells' corners: a cell is a beam by a range bin
        xs, ys = [], []
        for sb in (-0.5, 0.5):
            for sr in (-0.5, 0.5):
                th = torch.deg2rad(bc + sb * db_)
                rr = rc + sr * dr_
                xs.append(rr * torch.cos(th)); ys.append(rr * torch.sin(th))
        xs, ys = torch.cat(xs), torch.cat(ys)
        box_m = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        th = torch.deg2rad(bc)
        cx, cy = rc * torch.cos(th), rc * torch.sin(th)
        w = o[bb, kk].detach()
        ws = w.sum().clamp_min(1e-300)
        centroid = (float((w * cx).sum() / ws), float((w * cy).sum() / ws))
        j = int(w.argmax())
        peak = (float(cx[j]), float(cy[j]))
    if to_cartesian is not None:
        shape = own.shape if own.ndim == 3 else (own.shape[0], 1, own.shape[1])
        m_img = pm.to(o.dtype).reshape(shape[0], 1, shape[2])
        cm, gx, gy = to_cartesian(m_img.contiguous())
        cmask = cm > 0.25            # bilinear in the mask: a one-cell mask never reaches one
    gbox = None
    if geometry_points is not None and geometry_points.numel():
        gbox = geometry_box(geometry_points, beam_deg=beam_deg, range_m=range_m)
    return Label(name=name, kind=kind, visible=visible, contrast_db=contrast, n_cells=n_cells,
                 polar_box=pbox, box_m=box_m, centroid_m=centroid, peak_m=peak,
                 geometry_box_m=gbox,
                 polar_mask=pm if keep_masks else None, mask=cmask if keep_masks else None)


def draw_labels(ax, labels: Sequence[Label], *, color: str = "#7fdfff", geometry: bool = True,
                text: bool = True, lw: float = 1.0) -> None:
    """Boxes on a matplotlib axis in metres: solid the signal box, dashed the geometry's."""
    from matplotlib.patches import Rectangle
    for lab in labels:
        if lab.box_m is not None:
            x0, y0, x1, y1 = lab.box_m
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw))
            if text:
                ax.text(x0, y1, lab.name, color=color, fontsize=7, va="bottom", ha="left")
        if geometry and lab.geometry_box_m is not None:
            x0, y0, x1, y1 = lab.geometry_box_m
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw * 0.8,
                                   ls="--", alpha=0.7))
