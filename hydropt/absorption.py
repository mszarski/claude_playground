"""Volume absorption models.  Frequencies in kHz, attenuation in dB/km."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["thorp_db_per_km", "francois_garrison_db_per_km", "octave_bands"]


def thorp_db_per_km(f_khz: Tensor) -> Tensor:
    r"""Thorp (1967) absorption coefficient in dB/km for ``f`` in kHz.

    .. math::
        \alpha = \frac{0.11 f^2}{1 + f^2} + \frac{44 f^2}{4100 + f^2}
                 + 2.75\times10^{-4} f^2 + 0.003

    The four terms are boric-acid relaxation, magnesium-sulphate relaxation,
    pure-water viscosity, and a low-frequency floor.  Valid roughly 0.1-100 kHz
    at 4 degC; it carries no temperature, salinity or depth dependence -- use
    :func:`francois_garrison_db_per_km` when those matter.

    Differentiable in ``f_khz``, so band centres can themselves be optimised.
    """
    f2 = f_khz**2
    return 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003


def francois_garrison_db_per_km(
    f_khz: Tensor,
    *,
    temperature_c: float | Tensor = 4.0,
    salinity_ppt: float | Tensor = 35.0,
    depth_m: float | Tensor = 0.0,
    ph: float | Tensor = 8.0,
) -> Tensor:
    """Francois-Garrison (1982) absorption -- **hook, not yet implemented**.

    Francois-Garrison splits absorption into boric acid, magnesium sulphate and
    pure-water terms, each with its own relaxation frequency that depends on
    temperature, salinity, depth and pH.  Implementing it means replacing the
    fixed Thorp coefficients with those expressions; because the ray tracer
    already carries the sampled positions along each path, a depth-dependent
    ``alpha`` would be integrated along the path rather than multiplied by the
    total path length as Thorp is here.

    That path integral is the reason this is a hook and not a one-liner: it
    changes the accumulation in :mod:`hydropt.tracer` from ``alpha * s`` to a
    running sum of ``alpha(z) ds``, which needs a per-band accumulator in the
    ray state.  See the README's "Limitations" section.
    """
    raise NotImplementedError(
        "Francois-Garrison is a documented hook; use thorp_db_per_km for now. "
        "Implementing it requires per-band path-integrated absorption in the "
        "ray state -- see hydropt/absorption.py docstring."
    )


def octave_bands(
    f_low_khz: float = 0.1,
    n_bands: int = 4,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Octave-band centre frequencies (kHz), ``[n_bands]``.

    Spectral rendering in hydropt mirrors misuka's: geometry is traced once and
    the per-band energy differs only through the absorption exponent, so extra
    bands cost almost nothing.
    """
    k = torch.arange(n_bands, dtype=dtype or torch.get_default_dtype(), device=device)
    return float(f_low_khz) * 2.0**k
