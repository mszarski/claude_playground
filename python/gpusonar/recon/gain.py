"""The time-varying gain law of a record, derived from the record instead of assumed.

`core.apply_tvg` applies

    G(R) = spreading_db_per_decade * log10(R / reference_range) + 2 * absorption_db_per_m * R   dB

at the two-way range R = c t / 2, clamped at `max_gain_db`. Its defaults (40 dB per decade,
no absorption) are an echo's two-way spherical spreading at a frequency low enough for
absorption not to matter. They are silently wrong elsewhere (docs/HARDCODED_AUDIT.md K-4):
at 330 kHz and 12 deg C absorption is 0.083 dB/m, so the default law is 50 dB short at
300 m; a one-way arrival (a passive record, a transponder) spreads by 20 dB per decade.

`tvg_parameters` gives the keyword arguments of `core.apply_tvg` for one record:

- spreading: "two_way" (40), "one_way" (20), "auto" (two-way when the record has a transmit
  pulse, one-way when it has none), or a number of dB per decade;
- absorption: "none" (0, the default of `core.apply_tvg`), "record" (from the record's centre
  frequency and the water it states), or a number of dB/m ONE WAY.

Units and conventions: absorption is dB/m one way (the law doubles it); the range is the
two-way range of a monostatic echo; depth is metres below the surface (the world frame's z,
positive down, surface at z = 0). No source level enters, so the peak-or-rms question of
audit P-5 does not arise here.
"""

from __future__ import annotations

import math

from .. import acoustics

SPREADING = {"two_way": 40.0, "one_way": 20.0}


def record_absorption(record, *, frequency: float | None = None, temperature: float | None = None,
                      salinity: float | None = None, depth: float | None = None, ph: float = 8.0,
                      model: str = "francois_garrison") -> float:
    """Absorption in dB/m (one way) for the water a record states, at its centre frequency.

    Temperature and salinity come from `record.environment`; a record that does not carry
    them raises, because there is no water to assume (pass them to override or to supply).
    `depth` defaults to the receive array's depth (its z, never above the surface); `ph` is
    not in the record and defaults to 8.0, which matters only below about 10 kHz.
    """
    env = record.environment
    f = float(frequency) if frequency is not None else float(record.transmit.waveform.center())
    if not f > 0.0:
        f = float(record.samples.center_frequency)
    if not f > 0.0:
        raise ValueError("record_absorption: the record gives no frequency (no transmit band and no carrier)")
    t = env.temperature if temperature is None else float(temperature)
    s = env.salinity if salinity is None else float(salinity)
    if not (math.isfinite(t) and math.isfinite(s)):
        raise ValueError(
            "record_absorption: the record's environment has no temperature and salinity, so absorption cannot be "
            "derived from it; pass temperature= and salinity=, or give the absorption in dB/m")
    if depth is None:
        depth = max(float(record.geometry.rx_pose.position[2]), 0.0)
    return float(acoustics.absorption(f, t, s, float(depth), float(ph), model))


def tvg_parameters(record, *, spreading="two_way", absorption="none", reference_range: float = 1.0,
                   max_gain_db: float = 120.0, undo_hardware_ramp: bool = True, **water) -> dict:
    """Keyword arguments of `core.apply_tvg` for this record (see the module text).

    `water` (frequency, temperature, salinity, depth, ph, model) goes to `record_absorption`
    when `absorption="record"`.
    """
    if isinstance(spreading, str):
        if spreading == "auto":
            spreading = "two_way" if record.transmit.waveform.duration > 0.0 else "one_way"
        if spreading not in SPREADING:
            raise ValueError(f"tvg: spreading must be a number of dB per decade or one of {sorted(SPREADING)} or 'auto'")
        spreading = SPREADING[spreading]
    if isinstance(absorption, str):
        if absorption == "none":
            absorption = 0.0
        elif absorption == "record":
            absorption = record_absorption(record, **water)
        else:
            raise ValueError("tvg: absorption must be a number of dB/m (one way), 'none' or 'record'")
    elif water:
        raise ValueError(f"tvg: {sorted(water)} apply only with absorption='record'")
    absorption = float(absorption)
    if not (absorption >= 0.0 and math.isfinite(absorption)):
        raise ValueError("tvg: absorption must be a finite number of dB/m >= 0")
    return {"spreading_db_per_decade": float(spreading), "absorption_db_per_m": absorption,
            "reference_range": float(reference_range), "max_gain_db": float(max_gain_db),
            "undo_hardware_ramp": bool(undo_hardware_ramp)}


def gain_law_db(range_m, *, spreading_db_per_decade: float = 40.0, absorption_db_per_m: float = 0.0,
                reference_range: float = 1.0, max_gain_db: float = 120.0, **_ignored):
    """The law `core.apply_tvg` applies, in dB at two-way range(s) `range_m` (for reports and tests)."""
    import numpy as np
    r = np.maximum(np.asarray(range_m, dtype=float), 1e-6)
    db = np.where(r > reference_range, spreading_db_per_decade * np.log10(r / reference_range), 0.0)
    return np.minimum(db + 2.0 * absorption_db_per_m * r, max_gain_db)
