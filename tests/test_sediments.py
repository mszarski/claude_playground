"""Sediment presets: the internal physics, not the provenance of the numbers.

The values are representative rather than a transcription of any one published
table (see the module docstring), so there is nothing to assert about their
provenance.  What *is* assertable is that they are physically coherent and that
the relationships between them hold: the critical angle follows the sound speed
ratio, the reflection strength follows the impedance, and the two are separate
axes that the presets are ordered along independently.
"""

from __future__ import annotations

import math

import pytest
import torch

from hydropt import (
    SEDIMENTS, RayleighBottomLoss, critical_angle_deg, impedance_contrast, sediment,
    sediment_loss, sediment_names,
)
from hydropt.sediments import WATER_DENSITY, WATER_SOUND_SPEED


@pytest.fixture(autouse=True)
def _double():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
def test_every_preset_is_reachable_by_its_own_name():
    for name in sediment_names():
        assert sediment(name).name == name


@pytest.mark.parametrize("spelling", ["sand", "SAND", " Sand ", "medium sand",
                                      "medium_sand", "MEDIUM-SAND"])
def test_names_are_case_and_separator_insensitive(spelling):
    assert sediment(spelling).name == "sand"


@pytest.mark.parametrize("alias,target", [("mud", "clayey silt"),
                                          ("basalt", "rock"),
                                          ("pebble", "gravel"),
                                          ("cobble", "gravel")])
def test_aliases_resolve(alias, target):
    assert sediment(alias).name == target


def test_an_unknown_name_lists_the_options_rather_than_defaulting():
    """A silently-defaulted seabed is a wrong answer that looks like a right one."""
    with pytest.raises(KeyError) as exc:
        sediment("granite")
    message = str(exc.value)
    assert "granite" in message
    for expected in ("sand", "clay", "rock"):
        assert expected in message


# --------------------------------------------------------------------------- #
# Physical coherence and ordering
# --------------------------------------------------------------------------- #
def test_the_presets_are_ordered_coarse_to_fine_in_every_property():
    """Not an arbitrary list: density, sound speed and impedance all fall together
    from rock to clay, and a preset out of order would be a typo."""
    entries = [SEDIMENTS[n] for n in sediment_names()]
    for a, b in zip(entries, entries[1:]):
        assert a.density > b.density, f"{a.name} vs {b.name}"
        assert a.sound_speed > b.sound_speed, f"{a.name} vs {b.name}"
        assert a.impedance > b.impedance, f"{a.name} vs {b.name}"


def test_every_preset_sits_inside_the_range_it_reports():
    for name in sediment_names():
        s = SEDIMENTS[name]
        assert s.density_range[0] <= s.density <= s.density_range[1], name
        assert s.sound_speed_range[0] <= s.sound_speed <= s.sound_speed_range[1], name
        lo, hi = s.alpha_lambda_range
        assert lo <= s.alpha_lambda <= hi, name
        assert s.density_range[0] < s.density_range[1], name
        assert s.sound_speed_range[0] < s.sound_speed_range[1], name


def test_ratios_are_against_seawater():
    s = sediment("sand")
    assert s.density_ratio == pytest.approx(s.density / WATER_DENSITY, rel=1e-12)
    assert s.sound_speed_ratio == pytest.approx(
        s.sound_speed / WATER_SOUND_SPEED, rel=1e-12)
    assert s.impedance == pytest.approx(s.density * s.sound_speed, rel=1e-12)


# --------------------------------------------------------------------------- #
# The critical angle, which is the qualitative divide
# --------------------------------------------------------------------------- #
def test_critical_angle_follows_cos_theta_equals_the_speed_ratio():
    for name in sediment_names():
        s = SEDIMENTS[name]
        got = s.critical_angle_deg()
        if s.sound_speed <= WATER_SOUND_SPEED:
            assert got is None, name
        else:
            expected = math.degrees(math.acos(WATER_SOUND_SPEED / s.sound_speed))
            assert got == pytest.approx(expected, rel=1e-12), name


def test_a_slower_sediment_has_no_critical_angle_and_says_so_with_none():
    """None rather than zero: zero would read as 'grazing incidence reflects
    totally', and the truth is the opposite."""
    assert critical_angle_deg(1480.0) is None
    assert critical_angle_deg(1500.0) is None
    assert critical_angle_deg(1750.0) == pytest.approx(31.0027, abs=1e-3)
    assert sediment("clay").has_critical_angle is False
    assert sediment("clay").critical_angle_deg() is None
    assert sediment("sand").has_critical_angle is True


def test_the_critical_angle_closes_monotonically_from_rock_to_clay():
    angles = [SEDIMENTS[n].critical_angle_deg() for n in sediment_names()]
    present = [a for a in angles if a is not None]
    assert len(present) == len(angles) - 1, "only clay should lack one"
    for a, b in zip(present, present[1:]):
        assert a > b


def test_sand_reflects_almost_perfectly_below_its_critical_angle():
    """The consequence that matters for a shallow-water channel: the Rayleigh loss
    must actually collapse below theta_c, not merely have one on paper."""
    loss = sediment_loss("sand", learnable=False)
    theta_c = sediment("sand").critical_angle_deg()
    below = torch.tensor([math.radians(theta_c * 0.5)])
    above = torch.tensor([math.radians(min(theta_c * 2.0, 80.0))])
    db_below = float(loss(below))
    db_above = float(loss(above))
    assert db_below < 0.5, f"{db_below:.3f} dB below the critical angle"
    assert db_above > 2.0 * max(db_below, 1e-6), (
        f"below {db_below:.3f} dB, above {db_above:.3f} dB -- no critical-angle "
        f"behaviour")


def test_clay_has_no_angular_regime_of_near_total_reflection():
    """The qualitative difference from sand, asserted through the loss model."""
    clay = sediment_loss("clay", learnable=False)
    sand = sediment_loss("sand", learnable=False)
    graze = torch.linspace(math.radians(2.0), math.radians(40.0), 60)
    clay_db = clay(graze)
    sand_db = sand(graze)
    # Sand has a substantial near-lossless regime and clay has none at all.  The
    # count is not 60 * theta_c / 40: sediment attenuation starts to bite well
    # below the critical angle, so sand is under 0.5 dB out to about 17 degrees
    # rather than all the way to 31.  The contrast is the claim, not the count.
    assert float((sand_db < 0.5).sum()) >= 20
    assert float((clay_db < 0.5).sum()) == 0
    assert float(clay_db.min()) > 4.0 * float(sand_db.min())


# --------------------------------------------------------------------------- #
# Impedance is a separate axis from sound speed
# --------------------------------------------------------------------------- #
def test_normal_incidence_reflection_follows_the_impedance():
    for name in sediment_names():
        s = SEDIMENTS[name]
        z1 = WATER_DENSITY * WATER_SOUND_SPEED
        expected = (s.impedance - z1) / (s.impedance + z1)
        assert impedance_contrast(name) == pytest.approx(expected, rel=1e-12)


def test_reflection_strength_falls_monotonically_but_never_reverses_phase():
    """The correction to my own first draft, pinned so it cannot come back.

    Clay is *slower* than seawater and so has no critical angle, yet it is
    *denser*, so its impedance is still the higher of the two and R_0 stays
    positive.  Sound speed sets the angular structure and impedance sets the
    strength; reading one off the other gets it wrong.
    """
    values = [impedance_contrast(n) for n in sediment_names()]
    for a, b in zip(values, values[1:]):
        assert a > b
    assert all(v > 0.0 for v in values), "no preset is acoustically softer than water"
    assert impedance_contrast("clay") == pytest.approx(0.1657, abs=1e-3)
    assert sediment("clay").sound_speed < WATER_SOUND_SPEED
    assert sediment("clay").impedance > WATER_DENSITY * WATER_SOUND_SPEED


# --------------------------------------------------------------------------- #
# It produces a working, learnable loss model
# --------------------------------------------------------------------------- #
def test_sediment_loss_builds_the_matching_rayleigh_model():
    s = sediment("silty sand")
    loss = sediment_loss("silty sand", learnable=False)
    assert isinstance(loss, RayleighBottomLoss)
    assert float(loss.rho2) == pytest.approx(s.density, rel=1e-12)
    assert float(loss.c2) == pytest.approx(s.sound_speed, rel=1e-12)
    assert float(loss.alpha_lambda) == pytest.approx(s.alpha_lambda, rel=1e-12)
    assert float(loss.rho1) == pytest.approx(WATER_DENSITY, rel=1e-12)
    assert float(loss.c1) == pytest.approx(WATER_SOUND_SPEED, rel=1e-12)


def test_a_preset_is_learnable_by_default_because_that_is_what_it_is_for():
    """The point of a preset is to start an inversion from a plausible seabed."""
    loss = sediment_loss("sand")
    assert [n for n, _ in loss.named_parameters()] != []
    loss(torch.tensor([0.4])).sum().backward()
    for name in ("rho2", "c2", "alpha_lambda"):
        grad = getattr(loss, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
    assert list(sediment_loss("sand", learnable=False).parameters()) == []


def test_every_preset_gives_a_finite_positive_loss_across_all_grazing_angles():
    graze = torch.linspace(math.radians(0.5), math.radians(89.5), 200)
    for name in sediment_names():
        db = sediment_loss(name, learnable=False)(graze)
        assert torch.isfinite(db).all(), name
        assert float(db.min()) >= -1e-9, f"{name} amplifies the reflection"
