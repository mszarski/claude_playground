r"""An independent normal-mode reference, for cross-checking the ray model.

Everything else in hydropt is ray theory, and every test of it compares ray
theory against a closed form derived *within* the ray picture -- circular arcs,
Snell's law, image sources, the ray tube's own truncation term.  Those checks are
sharp and they share an assumption: that the ray formulation of the problem is
the right one.  A convention error consistent across the whole package would pass
all of them.

This module is the outside check.  It solves the **Pekeris waveguide**
-- an isovelocity water layer over a homogeneous fluid half-space -- by normal
modes: a completely different formulation of the same physics, with no code
shared with the tracer and no torch in it at all.  Where the two agree, both are
more likely right; where they disagree, one of them is wrong and the disagreement
says where to look.

The eigenvalue problem
----------------------
Water of depth ``H``, sound speed ``c1``, density ``rho1``, with a
pressure-release surface at ``z = 0``; bottom half-space ``c2``, ``rho2``.  Modes
are ``Z(z) = sin(gamma z)`` in the water, continued as
``sin(gamma H) exp(-beta (z - H))`` below, with

``gamma^2 = k1^2 - k_r^2``,  ``beta^2 = k_r^2 - k2^2``,  ``k_i = omega / c_i``.

Continuity of pressure and of ``(1/rho) dp/dz`` at ``z = H`` gives the
characteristic equation

.. math:: \tan(\gamma H) = -\frac{\rho_2 \gamma}{\rho_1 \beta}

Because ``beta > 0`` and ``gamma > 0`` for a trapped mode, the right-hand side is
always negative, so ``gamma_m H`` lies in ``((m - 1/2) pi, m pi)`` -- one root per
interval, which makes bracketing exact rather than a search.  A mode is trapped
when ``k2 < k_r < k1``, i.e. ``gamma < gamma_max = sqrt(k1^2 - k2^2)``, so the
number of trapped modes is

``M = floor(gamma_max H / pi + 1/2)``,  ``gamma_max = k1 sin(theta_c)``

and ``theta_c = arccos(c1 / c2)`` is exactly the ray critical angle.  **That is
the ray picture and the mode picture saying the same thing**: a ray steeper than
``theta_c`` refracts into the bottom and is lost, and a mode with ``k_r < k2`` is
leaky for the same reason.  :func:`PekerisWaveguide.n_modes` and the ray formula
agree by construction, and a test pins it.

The field
---------
Expanding ``p = sum_m Phi_m(r) Z_m(z)`` with ``int Z_m Z_n / rho dz = delta_mn``,
and taking the source normalisation ``p_free = exp(i k R) / R`` (so that
transmission loss is ``20 log10 R`` for spherical spreading, zero at 1 m), the
projected equation is ``(nabla^2_2D + k_r^2) Phi_m = -4 pi delta^2(r) Z_m(z_s) /
rho(z_s)``, whose outgoing solution uses the 2-D Green's function
``(i/4) H_0^{(1)}(k r)``:

.. math:: p(r, z) = \frac{i\pi}{\rho(z_s)} \sum_m Z_m(z_s) Z_m(z)
          H_0^{(1)}(k_{r,m} r)

The prefactor is the part of a mode sum that is easiest to get wrong and hardest
to notice, so it is **not** taken on trust here: :func:`ideal_image_field` gives
an independent exact solution of the ideal (pressure-release-bottom) waveguide by
the method of images, and the tests check the mode sum against it.  Two different
expansions of the same Green's function have to agree, and if the prefactor were
out by ``4 pi`` or a ``sqrt(2)`` they would not.

What this is not
----------------
Only trapped modes.  The continuous spectrum -- the part of the field that
radiates into the bottom -- is omitted, so this is wrong at short range (where
the near field matters) and it does not describe the steep paths a ray model
happily traces.  The comparison with hydropt therefore has to be made where
trapped modes dominate: beyond a few water depths in range, at frequencies with
enough trapped modes to matter.  That is a limitation of the *reference*, not of
the ray model, and it is why the comparison script reports the mode count.

Lossless, too: real ``c2`` and no sediment attenuation, so every ``k_r`` is real
and no mode decays with range.  A lossy bottom makes the eigenvalues complex and
the root-finding a 2-D problem, which is more machinery than a cross-check needs.

References
----------
Pekeris (1948).  Jensen, Kuperman, Porter & Schmidt, *Computational Ocean
Acoustics*, ch. 5.  Brekhovskikh & Lysanov, *Fundamentals of Ocean Acoustics*,
ch. 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import hankel1, kv

__all__ = ["PekerisWaveguide", "ideal_image_field", "ideal_mode_field",
           "hankel1_outgoing"]


@dataclass(frozen=True)
class PekerisWaveguide:
    """Isovelocity water over a fluid half-space, solved by normal modes.

    Args:
        depth: water depth ``H`` (m).
        c_water: sound speed in the water (m/s).
        c_bottom: sound speed in the bottom (m/s).  Must exceed ``c_water`` for
            any mode to be trapped -- a slower bottom has no critical angle and no
            discrete spectrum, which is the same statement
            :mod:`hydropt.sediments` makes about clay.
        rho_water, rho_bottom: densities (kg/m^3).
    """

    depth: float
    c_water: float = 1500.0
    c_bottom: float = 1800.0
    rho_water: float = 1000.0
    rho_bottom: float = 1800.0

    # ---- geometry ---------------------------------------------------------- #
    def critical_angle_deg(self) -> float:
        """Ray critical angle, grazing, ``arccos(c1/c2)`` in degrees."""
        if self.c_bottom <= self.c_water:
            raise ValueError("no critical angle: the bottom is not the faster medium")
        return math.degrees(math.acos(self.c_water / self.c_bottom))

    def _gamma_max(self, freq_hz: float) -> float:
        k1 = 2.0 * math.pi * freq_hz / self.c_water
        k2 = 2.0 * math.pi * freq_hz / self.c_bottom
        return math.sqrt(max(k1 * k1 - k2 * k2, 0.0))

    def n_modes(self, freq_hz: float) -> int:
        """Number of trapped modes, ``floor(gamma_max H / pi + 1/2)``.

        Equivalently ``floor(2 H sin(theta_c) / lambda + 1/2)`` -- the mode count
        is set by the ray critical angle and the water depth in wavelengths.
        """
        return int(math.floor(self._gamma_max(freq_hz) * self.depth / math.pi + 0.5))

    def cutoff_frequency_hz(self, mode: int = 1) -> float:
        """Lowest frequency at which ``mode`` is trapped.

        From ``gamma_max H = (mode - 1/2) pi`` with ``gamma_max = k1 sin(theta_c)``.
        Below the mode-1 cutoff the waveguide traps nothing at all.
        """
        sin_c = math.sqrt(max(1.0 - (self.c_water / self.c_bottom) ** 2, 0.0))
        if sin_c <= 0.0:
            raise ValueError("no trapped modes for any frequency")
        return (mode - 0.5) * self.c_water / (2.0 * self.depth * sin_c)

    # ---- eigenvalues ------------------------------------------------------- #
    def _characteristic(self, gamma: np.ndarray, freq_hz: float) -> np.ndarray:
        """``rho2 gamma cos(gamma H) + rho1 beta sin(gamma H)``, zero at a mode.

        Written in this product form rather than as ``tan(gamma H) + ...`` so it
        stays finite where the tangent blows up; a residual from this form is
        directly comparable between modes.
        """
        gmax = self._gamma_max(freq_hz)
        beta = np.sqrt(np.maximum(gmax * gmax - np.asarray(gamma) ** 2, 0.0))
        gh = np.asarray(gamma) * self.depth
        return (self.rho_bottom * np.asarray(gamma) * np.cos(gh)
                + self.rho_water * beta * np.sin(gh))

    def gammas(self, freq_hz: float, *, tol: float = 1e-15,
               max_iter: int = 200) -> np.ndarray:
        """Vertical wavenumbers of the trapped modes, ascending.

        One root per interval ``gamma H in ((m-1/2) pi, m pi)``, so each is found
        by bisection in a bracket known in advance -- no global search, and no
        chance of missing a mode or finding one twice.
        """
        gmax = self._gamma_max(freq_hz)
        out: list[float] = []
        for m in range(1, self.n_modes(freq_hz) + 1):
            lo = (m - 0.5) * math.pi / self.depth
            hi = min(m * math.pi / self.depth, gmax)
            # Nudge off the bracket ends, where the characteristic function is
            # exactly zero for a degenerate reason (beta = 0 at gamma_max).
            span = hi - lo
            lo_e, hi_e = lo + 1e-12 * span, hi - 1e-12 * span
            f_lo = float(self._characteristic(np.array([lo_e]), freq_hz)[0])
            f_hi = float(self._characteristic(np.array([hi_e]), freq_hz)[0])
            if f_lo * f_hi > 0.0:
                continue  # mode sits exactly at cutoff; not trapped
            for _ in range(max_iter):
                mid = 0.5 * (lo_e + hi_e)
                f_mid = float(self._characteristic(np.array([mid]), freq_hz)[0])
                if f_lo * f_mid <= 0.0:
                    hi_e = mid
                else:
                    lo_e, f_lo = mid, f_mid
                if hi_e - lo_e < tol * max(1.0, abs(mid)):
                    break
            out.append(0.5 * (lo_e + hi_e))
        return np.array(out)

    def wavenumbers(self, freq_hz: float) -> np.ndarray:
        """Horizontal wavenumbers ``k_r`` of the trapped modes, descending."""
        k1 = 2.0 * math.pi * freq_hz / self.c_water
        gamma = self.gammas(freq_hz)
        return np.sqrt(k1 * k1 - gamma * gamma)

    def phase_velocity(self, freq_hz: float) -> np.ndarray:
        """``omega / k_r`` per mode (m/s).  Between ``c_water`` and ``c_bottom``."""
        return 2.0 * math.pi * freq_hz / self.wavenumbers(freq_hz)

    def group_velocity(self, freq_hz: float, *, d_hz: float | None = None
                       ) -> np.ndarray:
        """``d omega / d k_r`` per mode (m/s), by central difference.

        Differenced rather than derived because the closed form needs
        ``d gamma / d omega`` from the implicit characteristic equation, and a
        reference implementation is worth more when it is obviously correct than
        when it is clever.
        """
        step = d_hz if d_hz is not None else max(freq_hz * 1e-5, 1e-6)
        lo, hi = self.wavenumbers(freq_hz - step), self.wavenumbers(freq_hz + step)
        n = min(len(lo), len(hi))
        if n == 0:
            return np.array([])
        d_omega = 2.0 * math.pi * 2.0 * step
        return d_omega / (hi[:n] - lo[:n])

    # ---- mode shapes ------------------------------------------------------- #
    def _amplitudes(self, freq_hz: float) -> np.ndarray:
        """Normalising constants so that ``int Z^2 / rho dz = 1``, analytically."""
        gamma = self.gammas(freq_hz)
        gmax = self._gamma_max(freq_hz)
        beta = np.sqrt(np.maximum(gmax * gmax - gamma * gamma, 0.0))
        gh = gamma * self.depth
        water = (self.depth / 2.0 - np.sin(2.0 * gh) / (4.0 * gamma)) / self.rho_water
        with np.errstate(divide="ignore", invalid="ignore"):
            bottom = np.where(beta > 0.0,
                              np.sin(gh) ** 2 / (2.0 * np.maximum(beta, 1e-300)
                                                 * self.rho_bottom),
                              0.0)
        return 1.0 / np.sqrt(water + bottom)

    def mode_shapes(self, freq_hz: float, z: np.ndarray) -> np.ndarray:
        """Normalised mode functions, ``[n_modes, len(z)]``.

        Valid for ``z`` in the water *and* in the bottom -- the evanescent tail
        below ``H`` carries real energy and omitting it from the normalisation
        would misstate every level by a few dB.
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        gamma = self.gammas(freq_hz)
        if gamma.size == 0:
            return np.zeros((0, z.size))
        gmax = self._gamma_max(freq_hz)
        beta = np.sqrt(np.maximum(gmax * gmax - gamma * gamma, 0.0))
        amp = self._amplitudes(freq_hz)
        g = gamma.reshape(-1, 1)
        b = beta.reshape(-1, 1)
        zz = z.reshape(1, -1)
        in_water = np.sin(g * np.minimum(zz, self.depth))
        tail = np.sin(g * self.depth) * np.exp(-b * np.maximum(zz - self.depth, 0.0))
        return amp.reshape(-1, 1) * np.where(zz <= self.depth, in_water, tail)

    # ---- the field --------------------------------------------------------- #
    def pressure(self, freq_hz: float, ranges: np.ndarray, z_source: float,
                 z_receiver: float) -> np.ndarray:
        """Complex pressure at each range, with ``p_free = exp(ikR)/R``.

        ``p = i pi / rho(z_s) sum_m Z_m(z_s) Z_m(z_r) H_0^(1)(k_rm r)``.
        """
        r = np.asarray(ranges, dtype=float).reshape(-1)
        k_r = self.wavenumbers(freq_hz)
        if k_r.size == 0:
            return np.zeros(r.size, dtype=complex)
        zs = self.mode_shapes(freq_hz, np.array([z_source]))[:, 0]
        zr = self.mode_shapes(freq_hz, np.array([z_receiver]))[:, 0]
        rho_s = self.rho_water if z_source <= self.depth else self.rho_bottom
        h = hankel1(0, np.outer(k_r, np.maximum(r, 1e-12)))  # [M, R]
        return 1j * math.pi / rho_s * ((zs * zr).reshape(-1, 1) * h).sum(axis=0)

    def transmission_loss(self, freq_hz: float, ranges: np.ndarray,
                          z_source: float, z_receiver: float, *,
                          coherent: bool = True) -> np.ndarray:
        """``-20 log10 |p|`` in dB, so spherical spreading gives ``20 log10 r``.

        ``coherent=False`` drops the cross terms between modes, summing
        ``|Z_m Z_m H_0|^2`` instead.  That is the quantity a ray model computing
        *energy* corresponds to: hydropt sums arrivals incoherently, so comparing
        against a coherent mode sum would be comparing against interference
        fringes the ray model never claimed to have.
        """
        r = np.asarray(ranges, dtype=float).reshape(-1)
        if self.wavenumbers(freq_hz).size == 0:
            # No trapped modes: this reference has nothing to say, and saying it
            # as "6000 dB" would look like an answer.  Both branches agree on it.
            return np.full(r.size, np.inf)
        if coherent:
            amp = np.abs(self.pressure(freq_hz, r, z_source, z_receiver))
        else:
            k_r = self.wavenumbers(freq_hz)
            if k_r.size == 0:
                return np.full(r.size, np.inf)
            zs = self.mode_shapes(freq_hz, np.array([z_source]))[:, 0]
            zr = self.mode_shapes(freq_hz, np.array([z_receiver]))[:, 0]
            rho_s = self.rho_water if z_source <= self.depth else self.rho_bottom
            h = hankel1(0, np.outer(k_r, np.maximum(r, 1e-12)))
            terms = (math.pi / rho_s * (zs * zr).reshape(-1, 1) * h)
            amp = np.sqrt((np.abs(terms) ** 2).sum(axis=0))
        with np.errstate(divide="ignore"):
            return -20.0 * np.log10(np.maximum(amp, 1e-300))


def ideal_image_field(depth: float, c_water: float, freq_hz: float,
                      ranges: np.ndarray, z_source: float, z_receiver: float,
                      *, n_images: int = 4000) -> np.ndarray:
    """Exact field in an *ideal* waveguide, by the method of images.

    Pressure-release at both ``z = 0`` and ``z = depth``.  Images sit at
    ``2nH + z_s`` with sign ``+`` and ``2nH - z_s`` with sign ``-`` for every
    integer ``n``; each pair cancels on both boundaries, which is what makes the
    construction exact.

    This exists to pin down the mode sum's prefactor without taking it on trust.
    It is a different expansion of the same Green's function, so the two must
    agree -- and a factor of ``4 pi`` or a stray ``sqrt(2)`` in either would show
    up immediately.  The sum converges only conditionally (terms fall as ``1/R``),
    so it is truncated **symmetrically** in ``n`` and needs a lot of images.
    """
    r = np.asarray(ranges, dtype=float).reshape(-1)
    k = 2.0 * math.pi * freq_hz / c_water
    n = np.arange(-n_images, n_images + 1).reshape(-1, 1)
    total = np.zeros(r.size, dtype=complex)
    for sign, z_img in ((1.0, 2.0 * n * depth + z_source),
                        (-1.0, 2.0 * n * depth - z_source)):
        dz = z_receiver - z_img  # [N, 1]
        dist = np.sqrt(r.reshape(1, -1) ** 2 + dz ** 2)
        total = total + sign * (np.exp(1j * k * dist) / dist).sum(axis=0)
    return total


def hankel1_outgoing(argument: np.ndarray) -> np.ndarray:
    r"""``H_0^{(1)}`` for a real *or* purely imaginary argument, without overflow.

    An evanescent mode has ``k_r = i kappa``, and evaluating
    ``H_0^{(1)}(i x) = J_0(i x) + i Y_0(i x)`` term by term is a numerical trap:
    each part grows like ``e^x`` while the combination decays like ``e^{-x}``, so
    the answer is lost to cancellation long before it overflows to ``nan`` -- which
    is exactly what the first version of this module did.

    The identity ``K_0(x) = (i pi / 2) H_0^{(1)}(i x)`` gives the decaying form
    directly:

    .. math:: H_0^{(1)}(i x) = -\frac{2i}{\pi} K_0(x)

    so this routes real arguments to ``hankel1`` and imaginary ones to ``K_0``.
    """
    arg = np.asarray(argument)
    if not np.iscomplexobj(arg):
        return hankel1(0, arg)
    out = np.zeros(arg.shape, dtype=complex)
    # A mode exactly at cutoff has k_r = 0, and H_0^(1)(0) diverges: the normal-mode
    # expansion has a genuine logarithmic singularity there, resolved physically by
    # the continuous spectrum the discrete sum leaves out.  Such a mode is dropped
    # rather than allowed to poison the sum with an infinity.  This is not
    # hypothetical -- H = 100 m, c = 1500 m/s and f = 150 Hz puts mode 20 exactly
    # at cutoff, which is how it was found.
    at_cutoff = np.abs(arg) < 1e-12
    imaginary = (np.abs(arg.real) < 1e-300) & ~at_cutoff
    real_part = ~imaginary & ~at_cutoff
    if real_part.any():
        out[real_part] = hankel1(0, arg[real_part].real)
    if imaginary.any():
        out[imaginary] = -2j / math.pi * kv(0, arg[imaginary].imag)
    return out


def ideal_mode_field(depth: float, c_water: float, freq_hz: float,
                     ranges: np.ndarray, z_source: float, z_receiver: float,
                     *, n_modes: int = 2000) -> np.ndarray:
    """Ideal waveguide by normal modes, the counterpart to :func:`ideal_image_field`.

    Pressure-release at both boundaries, so the eigenvalues are exactly
    ``gamma_m = m pi / H`` and ``Z_m = sqrt(2 rho / H) sin(gamma_m z)``.  The
    density cancels out of the field, as it must for a waveguide with no
    impedance contrast anywhere.

    Evanescent modes (``gamma_m > k``) are included via :func:`hankel1_outgoing`,
    so this is exact at any range rather than only in the far field.  A mode sitting
    exactly at cutoff is dropped, since the expansion is singular there -- and it is
    the *same* Green's function :func:`ideal_image_field` computes by images.
    Comparing the two is what verifies the ``i pi / rho`` prefactor in
    :meth:`PekerisWaveguide.pressure`, which is otherwise the easiest thing in a
    mode sum to get wrong and the hardest to notice.
    """
    r = np.asarray(ranges, dtype=float).reshape(-1)
    k = 2.0 * math.pi * freq_hz / c_water
    m = np.arange(1, n_modes + 1)
    gamma = m * math.pi / depth
    k_r2 = k * k - gamma * gamma
    k_r = np.where(k_r2 >= 0.0, np.sqrt(np.abs(k_r2)) + 0j,
                   1j * np.sqrt(np.abs(k_r2)))
    h = hankel1_outgoing(np.outer(k_r, np.maximum(r, 1e-12)))
    weight = (np.sin(gamma * z_source) * np.sin(gamma * z_receiver)).reshape(-1, 1)
    # i pi / rho * (2 rho / H) = 2 pi i / H -- the density cancels.
    return (2j * math.pi / depth) * (weight * h).sum(axis=0)
