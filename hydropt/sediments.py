r"""Sediment presets: plausible seabeds by name, and the physics that orders them.

:class:`hydropt.boundaries.RayleighBottomLoss` is parameterised the way the
physics is -- density, sound speed, attenuation -- which is correct and unhelpful
when what you have is the word "sand".  This maps names onto numbers.

**Read this before quoting a number out of here.** These are *representative*
values assembled from the ranges reported in the marine-sediment literature
(Hamilton 1980; Hamilton & Bachman 1982; Jackson & Richardson 2007; the APL-UW
High-Frequency Ocean Environmental Acoustic Models Handbook, 1994).  They are
**not** a verbatim transcription of any one published table, and they should not
be cited as one.  Real sediments vary by more than the difference between
adjacent entries here -- porosity, grain-size distribution, depth in the
sediment column and gas content all move them -- so each preset carries the
range it was drawn from, and for quantitative work you want measured values for
your site rather than a name.

What the presets are good for is starting a scene, and starting an *inversion*:
``sediment_loss("sand", learnable=True)`` puts the optimiser somewhere sane and
lets it find the rest, which is what `examples/02` and `07` do by hand.

The ordering is the physics, and it is testable
----------------------------------------------
The presets are not an arbitrary list.  Going from gravel to clay, sound speed
and density both fall, and two things follow that the tests pin:

**The critical angle closes.** With grazing angles measured from the interface,
total internal reflection needs ``cos(theta_c) = c_1 / c_2``, so a sediment only
*has* a critical angle when it is faster than the water above it.  Sand is
(``c_2`` ~ 1750 m/s against 1500), and reflects almost perfectly below about 31
degrees grazing.  **Clay is not** -- at ~1480 m/s it is slower than seawater, so
there is no critical angle at all and no angular regime of near-total
reflection.  That is a qualitative difference between two entries in the same
table, not a matter of degree, and it dominates how a shallow-water channel
behaves.

**Normal-incidence reflection follows the impedance, which is a separate axis.**
``R_0 = (Z_2 - Z_1) / (Z_2 + Z_1)`` with ``Z = rho c``, and it decreases
monotonically down the list -- but it stays *positive* all the way to clay, +0.17,
because clay is denser than seawater even while being slower than it.  So the two
properties disagree: clay has no critical angle and still reflects a sixth of the
incident pressure at normal incidence.  Sound speed sets the angular structure
and impedance sets the strength, and reading one off the other gets it wrong.

References
----------
Hamilton (1980), "Geoacoustic modeling of the sea floor", JASA 68.  Hamilton &
Bachman (1982), "Sound velocity and related properties of marine sediments",
JASA 72.  Jackson & Richardson (2007), *High-Frequency Seafloor Acoustics*.
Jensen et al., *Computational Ocean Acoustics*, sec. 1.4, for the loss-tangent
form the attenuation enters through.
"""

from __future__ import annotations

from typing import NamedTuple

from .boundaries import RayleighBottomLoss

__all__ = [
    "Sediment",
    "SEDIMENTS",
    "sediment",
    "sediment_loss",
    "sediment_names",
    "critical_angle_deg",
    "impedance_contrast",
]

WATER_DENSITY = 1024.0  # kg/m^3
WATER_SOUND_SPEED = 1500.0  # m/s


class Sediment(NamedTuple):
    """Representative acoustic properties of one seabed type.

    ``density`` in kg/m^3, ``sound_speed`` in m/s, ``alpha_lambda`` in dB per
    wavelength.  The ``*_range`` fields record the spread reported for that
    class, and are there to be read: if the range spans your answer, the name is
    not enough information.
    """

    name: str
    density: float
    sound_speed: float
    alpha_lambda: float
    density_range: tuple[float, float]
    sound_speed_range: tuple[float, float]
    alpha_lambda_range: tuple[float, float]
    note: str = ""

    @property
    def density_ratio(self) -> float:
        """``rho_2 / rho_1`` against seawater."""
        return self.density / WATER_DENSITY

    @property
    def sound_speed_ratio(self) -> float:
        """``c_2 / c_1`` against seawater.  Below 1 means no critical angle."""
        return self.sound_speed / WATER_SOUND_SPEED

    @property
    def impedance(self) -> float:
        """``rho c`` in Rayl (kg m^-2 s^-1)."""
        return self.density * self.sound_speed

    @property
    def has_critical_angle(self) -> bool:
        return self.sound_speed > WATER_SOUND_SPEED

    def critical_angle_deg(self) -> float | None:
        """Grazing angle of total internal reflection, or ``None`` if there is none."""
        return critical_angle_deg(self.sound_speed)

    def loss(self, **kwargs) -> RayleighBottomLoss:
        """A :class:`hydropt.boundaries.RayleighBottomLoss` with these values."""
        kwargs.setdefault("rho1", WATER_DENSITY)
        kwargs.setdefault("c1", WATER_SOUND_SPEED)
        return RayleighBottomLoss(self.density, self.sound_speed,
                                  self.alpha_lambda, **kwargs)


def _s(name, rho, c, al, rho_r, c_r, al_r, note="") -> Sediment:
    return Sediment(name, rho, c, al, rho_r, c_r, al_r, note)


# Ordered coarse to fine, which is also decreasing impedance.
SEDIMENTS: dict[str, Sediment] = {
    s.name: s for s in (
        _s("rock", 2500.0, 3000.0, 0.1, (2200.0, 2700.0), (2500.0, 5000.0),
           (0.05, 0.3),
           "Basalt or consolidated rock.  A wide class: c_2 from 2500 m/s for "
           "weathered or fractured rock to 5000 and above for fresh basalt."),
        _s("gravel", 2300.0, 1850.0, 0.6, (2000.0, 2500.0), (1750.0, 2000.0),
           (0.4, 0.9),
           "Cobble and pebble.  Scattering dominates reflection at high "
           "frequency, which this model does not include."),
        _s("coarse sand", 2100.0, 1836.0, 0.9, (1950.0, 2200.0), (1750.0, 1900.0),
           (0.6, 1.2)),
        _s("sand", 2000.0, 1750.0, 0.8, (1900.0, 2100.0), (1700.0, 1800.0),
           (0.5, 1.0),
           "Medium sand: the default a shallow-water scene usually wants."),
        _s("fine sand", 1900.0, 1700.0, 0.8, (1800.0, 2000.0), (1650.0, 1750.0),
           (0.5, 1.0)),
        _s("silty sand", 1800.0, 1620.0, 0.7, (1700.0, 1900.0), (1570.0, 1670.0),
           (0.4, 0.9)),
        _s("sandy silt", 1700.0, 1560.0, 0.5, (1600.0, 1800.0), (1520.0, 1600.0),
           (0.3, 0.7)),
        _s("silt", 1650.0, 1530.0, 0.4, (1550.0, 1750.0), (1500.0, 1570.0),
           (0.2, 0.6),
           "Marginal: at 1530 m/s the critical angle is only about 11 degrees, "
           "so small changes in c_2 change the channel's character a lot."),
        _s("clayey silt", 1550.0, 1510.0, 0.2, (1450.0, 1650.0), (1490.0, 1530.0),
           (0.1, 0.4)),
        _s("clay", 1450.0, 1480.0, 0.1, (1350.0, 1550.0), (1450.0, 1500.0),
           (0.02, 0.2),
           "Slower than seawater, so there is NO critical angle and no regime "
           "of near-total reflection -- qualitatively unlike sand.  Still denser "
           "than seawater, so R_0 stays positive at +0.17."),
    )
}

_ALIASES = {
    "medium sand": "sand",
    "mud": "clayey silt",
    "basalt": "rock",
    "pebble": "gravel",
    "cobble": "gravel",
}


def sediment_names() -> list[str]:
    """Preset names, coarse to fine."""
    return list(SEDIMENTS)


def sediment(name: str) -> Sediment:
    """Look up a preset by name, case- and separator-insensitively.

    Raises:
        KeyError: naming every preset, rather than leaving you to guess -- a
            silently-defaulted seabed is a wrong answer that looks like a right
            one.
    """
    key = " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())
    key = _ALIASES.get(key, key)
    if key not in SEDIMENTS:
        raise KeyError(
            f"unknown sediment {name!r}; available: "
            + ", ".join(sorted(set(SEDIMENTS) | set(_ALIASES))))
    return SEDIMENTS[key]


def sediment_loss(name: str, *, learnable: bool = True, **kwargs
                  ) -> RayleighBottomLoss:
    """A :class:`hydropt.boundaries.RayleighBottomLoss` for a named seabed.

    Args:
        name: a preset name or alias; see :func:`sediment_names`.
        learnable: on by default, because the usual reason to want a preset is to
            start an inversion from a plausible seabed rather than to assert one.
        kwargs: forwarded to ``RayleighBottomLoss`` (``rho1``, ``c1``).

    Example:
        >>> from hydropt.sediments import sediment_loss
        >>> loss = sediment_loss("sand")            # fit from here
        >>> fixed = sediment_loss("clay", learnable=False)
    """
    return sediment(name).loss(learnable=learnable, **kwargs)


def critical_angle_deg(sound_speed: float,
                       water_sound_speed: float = WATER_SOUND_SPEED) -> float | None:
    """Grazing angle of total internal reflection, in degrees, or ``None``.

    ``cos(theta_c) = c_1 / c_2`` with grazing measured from the interface, so
    there is no critical angle at all when the sediment is the slower medium --
    which this returns as ``None`` rather than as zero, because zero would read
    as "grazing incidence reflects totally" and the truth is the opposite.
    """
    import math

    if sound_speed <= water_sound_speed:
        return None
    return math.degrees(math.acos(water_sound_speed / sound_speed))


def impedance_contrast(name: str, *, water_density: float = WATER_DENSITY,
                       water_sound_speed: float = WATER_SOUND_SPEED) -> float:
    """Normal-incidence pressure reflection coefficient ``R_0``.

    ``(Z_2 - Z_1) / (Z_2 + Z_1)`` with ``Z = rho c``.  Positive for every preset
    here, clay included: clay is slower than seawater but denser, so its
    impedance is still the higher of the two.  A genuinely softer bottom would
    give a negative ``R_0``, which is a phase reversal rather than a small number.
    """
    s = sediment(name)
    z1 = water_density * water_sound_speed
    z2 = s.impedance
    return (z2 - z1) / (z2 + z1)
