"""Models of hardware settings that shape the data, so the simulator and the processing
chain agree with what a rig was told to do.

TX73L64 receive attenuator ("DTGC") as the NVIDIA ultrasound reference programs it
(docs/HOLOSCAN_NOTES.md section 6, from `python/nvus/tx.py` at commit 24d405e). It is an
attenuator: it starts `start_db` down and steps back to 0 dB in 0.25 dB steps. The limits
below are those of the reference's one-segment encoder, not of the chip.
"""

from __future__ import annotations

DTGC_DB_PER_STEP = 0.25
DTGC_MAX_DB = 12.0
DTGC_START_TGC_MAX = 0x3FF            # start delay, in ticks
DTGC_STEP_PERIOD_MAX = 0xFF           # ticks per attenuation step
FIXED_ATTENUATION_CODE_MAX = 48


def tx73l64_dtgc(start_db: float, start_us: float, ramp_us: float, bf_clock_hz: float = 200e6) -> dict:
    """One-segment attenuation ramp: register values and the gain curve it gives.

    Returns `reg_0x05`, `reg_0x06`, the quantised `start_s` and `ramp_s`, and `tvg` knots
    (`time` in seconds after transmit, `gain_db` relative to the final gain) in the form
    a sonar config's `receive.tvg` takes.
    """
    tick = 2.0 / bf_clock_hz
    steps = round(start_db / DTGC_DB_PER_STEP)
    if not (DTGC_DB_PER_STEP <= start_db <= DTGC_MAX_DB) or abs(steps * DTGC_DB_PER_STEP - start_db) > 1e-9:
        raise ValueError(f"DTGC start must be {DTGC_DB_PER_STEP} to {DTGC_MAX_DB} dB in {DTGC_DB_PER_STEP} dB steps")
    start_ticks = round(start_us * 1e-6 / tick)
    if not 0 <= start_ticks <= DTGC_START_TGC_MAX:
        raise ValueError(f"DTGC start delay must be at most {DTGC_START_TGC_MAX * tick * 1e6:.2f} us")
    period = round(ramp_us * 1e-6 / (steps * tick))
    if not 1 <= period <= DTGC_STEP_PERIOD_MAX:
        raise ValueError("DTGC ramp time cannot be represented by one segment "
                         f"(at most {DTGC_STEP_PERIOD_MAX * steps * tick * 1e6:.1f} us at {start_db} dB)")
    start_s, ramp_s = start_ticks * tick, period * steps * tick
    return {"reg_0x05": (start_ticks << 6) | steps, "reg_0x06": (steps << 8) | period,
            "start_s": start_s, "ramp_s": ramp_s,
            "tvg": {"time": [start_s, start_s + ramp_s], "gain_db": [-float(start_db), 0.0]}}


def tx73l64_fixed_attenuation_db(code: int) -> float:
    """Gain (negative dB) of the fixed attenuation setting, nominal 0.25 dB per code."""
    if not 0 <= int(code) <= FIXED_ATTENUATION_CODE_MAX:
        raise ValueError(f"attenuation code must be 0 to {FIXED_ATTENUATION_CODE_MAX}")
    return -DTGC_DB_PER_STEP * int(code)


def expand_gain_ramp(spec: dict) -> dict:
    """A config's `receive.gain_ramp` mapping to `receive.tvg` knots."""
    model = spec.get("model")
    if model == "tx73l64_dtgc":
        return tx73l64_dtgc(float(spec["start_db"]), float(spec["start_us"]), float(spec["ramp_us"]),
                            float(spec.get("bf_clock_hz", 200e6)))["tvg"]
    if model == "tx73l64_fixed":
        gain = tx73l64_fixed_attenuation_db(int(spec["code"]))
        return {"time": [0.0, 1.0], "gain_db": [gain, gain]}
    raise ValueError(f"unknown gain ramp model {model!r}")
