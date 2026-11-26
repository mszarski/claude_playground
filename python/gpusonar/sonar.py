"""A sonar described by config: array, transmit, sampling. Builds record templates.

Adding a sonar means adding a YAML file under configs/sonars. Nothing here or in
the signal processing is specific to one sonar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _gpusonar as core
from . import acoustics
from .config import load_sonar_config

_MODES = {"fls": core.Mode.Fls, "sas": core.Mode.Sas, "flank": core.Mode.Flank}
_WAVEFORMS = {
    "cw": core.WaveformType.Cw,
    "lfm": core.WaveformType.Lfm,
    "gaussian": core.WaveformType.Gaussian,
    # A sampled pulse (a file, a list of samples, or a built-in built from samples):
    # sim/waveform.py. The record carries it as its real replica.
    "sampled": core.WaveformType.Arbitrary,
}
_KINDS = {
    "real": core.SampleKind.RealPassband,
    "baseband": core.SampleKind.ComplexBaseband,
}


# --- source level: ONE convention ------------------------------------------------------------------------
# A configured source level is dB re 1 uPa at 1 m, RMS: the rms pressure of the pulse while it is on (the usual
# datasheet figure, and what the sonar equation uses against rms noise). The key is `source_level_db_rms`.
# Amplitudes are peak values (a record's `transmit.waveform.amplitude`, an echo's envelope): a tone or a sweep of
# rms pressure p has the peak sqrt(2) p. That factor is applied here and nowhere else. A RECORD holds peak values
# throughout (core/record.hpp): its `source_level_db` is the configured RMS level + 3.01 dB (`make_record`).
SOURCE_LEVEL_KEY = "source_level_db_rms"
SOURCE_LEVEL_ALIAS = "source_level_db"          # deprecated: read as the RMS level, with a warning
PEAK_OVER_RMS = float(np.sqrt(2.0))
PEAK_OVER_RMS_DB = float(20.0 * np.log10(PEAK_OVER_RMS))      # 3.01 dB
# Receive keys that older files carried and nothing reads: refused, with where the model is.
NOT_SIMULATED_RECEIVE_KEYS = {
    "highpass_corner_hz": "the AC coupling is modelled only as receive.overload.highpass_corner_hz "
                          "(a build's corner is kept as hardware.boards.highpass_hz)",
    "bandpass_hz": "no receive band-pass filter is modelled; remove the key",
}


def peak_pressure_at_1m(source_level_db_rms: float) -> float:
    """Peak pressure (uPa) at 1 m of a pulse whose RMS source level is `source_level_db_rms` (dB re 1 uPa at 1 m)."""
    return PEAK_OVER_RMS * 10.0 ** (float(source_level_db_rms) / 20.0)


def source_level_db_rms(waveform: dict, where: str = "transmit.waveform"):
    """The RMS source level of a waveform mapping (dB re 1 uPa at 1 m), or None when it has none. Reads
    `source_level_db_rms`; the older `source_level_db` is the same number under a name that did not say so: it is
    accepted with a FutureWarning. Both with different values is an error."""
    new, old = waveform.get(SOURCE_LEVEL_KEY), waveform.get(SOURCE_LEVEL_ALIAS)
    if old is not None:
        if new is not None and float(new) != float(old):
            raise ValueError(f"{where}: {SOURCE_LEVEL_KEY} ({new}) and the deprecated {SOURCE_LEVEL_ALIAS} ({old}) are both given and differ")
        import warnings
        warnings.warn(f"{where}.{SOURCE_LEVEL_ALIAS} is deprecated: write {SOURCE_LEVEL_KEY}. Source levels are RMS levels (dB re 1 uPa at 1 m); the "
                      "number is taken as one (a peak figure is 3 dB higher: subtract 3 dB from it)", FutureWarning, stacklevel=3)
        return float(old)
    return None if new is None else float(new)


# --- transmitted energy: the accounting, and an opt-in way to state it --------------------------------------------
# `source_level_db_rms` (SL) is the RMS level of a constant-envelope pulse whose peak is the transmitted pulse's peak:
# the record's amplitude is sqrt(2) 10^(SL / 20) and multiplies the envelope b(u), whose peak is 1, BEFORE any taper.
# So the energy at 1 m (integral of p^2 over the pulse, the real pressure p = amplitude Re{b exp(2 pi i F u)}) is
#     E = 10^(SL / 10) * integral |b(u)|^2 du,   i.e.  source energy level SE = SL + 10 log10(integral |b|^2 du / 1 s),
# dB re 1 uPa^2 s at 1 m (narrowband: the carrier's double-frequency term averages out). A Tukey taper of fraction
# alpha on a pulse of length T gives integral |b|^2 = T (1 - 5 alpha / 8): a Hann tone (alpha 1) carries 0.375 T, a
# Tukey 0.05 sweep 0.969 T, so at the same SL the Hann tone has 4.1 dB less energy (docs/tasks/DOPPLER.md 0 item 4).
# This is what SL has always meant; it is unchanged. Opt-in: `source_energy_db` (SE) states the energy instead, and
# SL is derived from it (the record then holds the derived SL as always); giving both with different values is an
# error. A train's segments can state theirs (sim/waveform.py `build_train`).
SOURCE_ENERGY_KEY = "source_energy_db"


def envelope_energy_s(waveform: dict, sampled=None):
    """Integral of |b(u)|^2 du (s) of a transmit waveform mapping's unit-peak envelope, or None where there is no closed
    form (a Gaussian pulse). A sampled pulse (and a train) from its samples; CW and LFM from the Tukey taper."""
    if sampled is not None:
        return sampled.envelope_energy()
    kind = str(waveform.get("type", "lfm")).lower()
    if kind not in ("cw", "lfm"):
        return None
    alpha = min(max(float(waveform.get("window_alpha", 0.0)), 0.0), 1.0)
    return float(waveform["duration"]) * (1.0 - 5.0 * alpha / 8.0)


def _level_from_energy(waveform: dict, sampled) -> dict:
    """The waveform mapping with `source_level_db_rms` derived from the stated energy (opt-in `source_energy_db`, or a
    train's segment energies)."""
    stated = waveform.get(SOURCE_ENERGY_KEY)
    own = getattr(sampled, "source_energy_db", None)
    if stated is not None and own is not None and abs(float(stated) - own) > 1e-9:
        raise ValueError(f"transmit.waveform.{SOURCE_ENERGY_KEY} ({stated}) and the train's segment energies ({own:.6g} dB in all) "
                         "are both given and differ")
    energy_db = float(stated) if stated is not None else float(own)
    e = envelope_energy_s(waveform, sampled)
    if e is None:
        raise ValueError(f"transmit.waveform.{SOURCE_ENERGY_KEY}: only for CW, LFM and sampled pulses (a "
                         f"{waveform.get('type')} pulse has no stated envelope energy here); give {SOURCE_LEVEL_KEY}")
    level = energy_db - 10.0 * np.log10(e)
    given = source_level_db_rms(waveform)
    if given is not None and abs(given - level) > 1e-6:
        raise ValueError(f"transmit.waveform: {SOURCE_LEVEL_KEY} ({given}) and {SOURCE_ENERGY_KEY} ({energy_db:.6g}, which "
                         f"means {SOURCE_LEVEL_KEY} {level:.6g} for this pulse) are both given and differ; give one (set the "
                         f"other to null to replace a base's)")
    return {**waveform, SOURCE_LEVEL_KEY: float(level), SOURCE_ENERGY_KEY: energy_db}


def record_source_level_db_rms(record) -> float:
    """The RMS source level (dB re 1 uPa at 1 m) of a record: its `transmit.waveform.source_level_db` is the PEAK
    level (core/record.hpp), 3.01 dB above. NaN when the record has none."""
    return float(record.transmit.waveform.source_level_db) - PEAK_OVER_RMS_DB


def normalise_source_level(cfg: dict) -> dict:
    """A sonar config (or an override of one) with the deprecated `transmit.waveform.source_level_db` moved to
    `source_level_db_rms`, so that a file or an override written with the old key still replaces a base's value."""
    w = ((cfg or {}).get("transmit") or {}).get("waveform")
    if not isinstance(w, dict) or SOURCE_LEVEL_ALIAS not in w:
        return cfg
    w = dict(w)
    old = w.pop(SOURCE_LEVEL_ALIAS)
    if old is None:
        w[SOURCE_LEVEL_KEY] = None
    else:
        w[SOURCE_LEVEL_KEY] = source_level_db_rms({SOURCE_LEVEL_ALIAS: old, **({SOURCE_LEVEL_KEY: w[SOURCE_LEVEL_KEY]} if w.get(SOURCE_LEVEL_KEY) is not None else {})})
    return {**cfg, "transmit": {**cfg["transmit"], "waveform": w}}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _positions(array_cfg: dict) -> np.ndarray:
    """Element lever arms [N, 3] from a `uniform_line`, `line`, `stacked_lines` or `explicit` layout."""
    layout = array_cfg.get("layout", "uniform_line")
    origin = np.asarray(array_cfg.get("origin", [0.0, 0.0, 0.0]), dtype=float)
    if layout == "uniform_line":
        n = int(array_cfg["count"])
        axis = _unit(array_cfg.get("axis", [1.0, 0.0, 0.0]))
        idx = np.arange(n) - (n - 1) / 2.0
        return origin + idx[:, None] * float(array_cfg["pitch"]) * axis
    if layout == "line":
        # Arbitrary positions along one axis (sparse arrays), metres from the origin.
        axis = _unit(array_cfg.get("axis", [1.0, 0.0, 0.0]))
        along = np.asarray(array_cfg["positions"], dtype=float)
        return origin + along[:, None] * axis
    if layout == "explicit":
        pos = np.asarray(array_cfg["positions"], dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("array.positions must be a list of [x, y, z]")
        return origin + pos
    if layout == "stacked_lines":
        # Two or more parallel lines at different heights on the vehicle (elevation as well as
        # bearing). Channels run row by row. Each row: `up` (m towards the vehicle's top, i.e. -z,
        # made perpendicular to the axis), `out` (m along the boresight, for a curved hull) and
        # `along` (m along the axis), and optionally its own `count`, `pitch` or `positions`.
        # `interleave: n` orders the channels n from each row in turn instead (with boards of
        # eight inputs and two rows, n = 4 puts the same stations of both rows on one board,
        # so the rows of a board share its clock).
        axis = _unit(array_cfg.get("axis", [1.0, 0.0, 0.0]))
        up = np.array([0.0, 0.0, -1.0]) - axis * -axis[2]
        up = _unit(up)
        out_dir = np.asarray(array_cfg.get("boresight", [0.0, 1.0, 0.0]), dtype=float)
        out_dir = _unit(out_dir - axis * (out_dir @ axis) - up * (out_dir @ up))
        rows = array_cfg.get("rows")
        if not rows:
            raise ValueError("array.layout stacked_lines needs array.rows")
        parts = []
        for row in rows:
            if row.get("positions") is not None:
                along = np.asarray(row["positions"], dtype=float)
            else:
                n = int(row.get("count", array_cfg.get("count", 0)))
                if n < 1:
                    raise ValueError("stacked_lines: give array.count or a count for every row")
                along = (np.arange(n) - (n - 1) / 2.0) * float(row.get("pitch", array_cfg.get("pitch")))
            along = along + float(row.get("along", 0.0))
            parts.append(origin + along[:, None] * axis + float(row.get("up", 0.0)) * up + float(row.get("out", 0.0)) * out_dir)
        n_inter = int(array_cfg.get("interleave", 0))
        if n_inter > 0:
            if len({len(p) for p in parts}) != 1:
                raise ValueError("stacked_lines: interleave needs rows of equal count")
            parts = [p[i:i + n_inter] for i in range(0, len(parts[0]), n_inter) for p in parts]
        return np.vstack(parts)
    raise ValueError(f"unknown array layout {layout!r}")


def _beamwidth(cfg: dict, key: str, size_key: str, wavelength: float) -> float:
    """FWHM beamwidth: explicit value, or 0.886 * lambda / aperture, or omnidirectional."""
    if cfg.get(key) is not None:
        return float(cfg[key])
    if cfg.get(size_key) is not None:
        return 0.886 * wavelength / float(cfg[size_key])
    return float("nan")


def apply_fov(cfg: dict) -> dict:
    """Expand a config's `fov` block into boresights and beamwidths.

        fov: {horizontal_deg: 120, vertical_deg: 20, tilt_up_deg: 0}

    sets the element and transmit beamwidths (FWHM) in azimuth and elevation and tilts
    both boresights up from the horizontal `array.boresight` by `tilt_up_deg` (z is down,
    so up is -z). Explicit `beamwidth_*` or `transmit.boresight` keys win over the block.
    """
    fov = cfg.get("fov")
    if not fov or fov.get("applied"):
        return cfg
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["fov"]["applied"] = True
    arr, tx = cfg["array"], cfg["transmit"]
    level = _unit(arr.get("boresight", [1.0, 0.0, 0.0]))
    if abs(level[2]) > 1e-9:
        raise ValueError("fov: array.boresight must be horizontal; set the tilt with fov.tilt_up_deg")
    tilt = np.radians(float(fov.get("tilt_up_deg", 0.0)))
    boresight = [float(v) for v in np.cos(tilt) * level - np.sin(tilt) * np.array([0.0, 0.0, 1.0])]
    arr["boresight"] = boresight
    tx.setdefault("boresight", boresight)
    for block in (arr, tx):
        if fov.get("horizontal_deg") is not None:
            block.setdefault("beamwidth_az", float(np.radians(fov["horizontal_deg"])))
        if fov.get("vertical_deg") is not None:
            block.setdefault("beamwidth_el", float(np.radians(fov["vertical_deg"])))
    return cfg


@dataclass
class SonarModel:
    name: str
    mode: str
    config: dict
    positions: np.ndarray  # [N, 3] lever arms of the full array, body frame
    boresight: np.ndarray
    beamwidth_az: float
    beamwidth_el: float
    mux_groups: list[list[int]]  # element indices captured by each transmit of a scan
    channel_board: np.ndarray  # [N] board index of each element's channel
    boards: list[dict] = field(default_factory=list)
    nominal_sound_speed: float = acoustics.NOMINAL_SOUND_SPEED
    sampled_waveform: object = None  # sim.waveform.SampledWaveform when transmit.waveform.type is `sampled`

    # --- construction -------------------------------------------------------

    @classmethod
    def load(cls, name_or_path, overrides: dict | None = None) -> "SonarModel":
        return cls.from_config(load_sonar_config(name_or_path, overrides))

    @classmethod
    def from_config(cls, cfg: dict) -> "SonarModel":
        cfg = normalise_source_level(apply_fov(cfg))
        tx_wf = cfg["transmit"]["waveform"]
        sampled = None
        if tx_wf.get("type") == "sampled":
            # Build the pulse once; a file's or a list's duration is its length.
            from .sim.waveform import build_waveform
            sampled = build_waveform(tx_wf)
            if tx_wf.get("duration") is None or tx_wf.get("source") in ("samples", "file"):
                cfg = {**cfg, "transmit": {**cfg["transmit"], "waveform": {**tx_wf, "duration": sampled.duration}}}
                tx_wf = cfg["transmit"]["waveform"]
            if str(tx_wf.get("source", "")).lower() == "train":
                # A train's duration is its segments' sum and its band holds them (a base's sweep keys are replaced).
                cfg = {**cfg, "transmit": {**cfg["transmit"], "waveform": {
                    **tx_wf, "duration": sampled.duration, "f_start": float(sampled.f_start), "f_stop": float(sampled.f_stop)}}}
                tx_wf = cfg["transmit"]["waveform"]
        if tx_wf.get(SOURCE_ENERGY_KEY) is not None or getattr(sampled, "source_energy_db", None) is not None:
            cfg = {**cfg, "transmit": {**cfg["transmit"], "waveform": _level_from_energy(tx_wf, sampled)}}
            tx_wf = cfg["transmit"]["waveform"]
        fc = 0.5 * (float(tx_wf["f_start"]) + float(tx_wf.get("f_stop", tx_wf["f_start"])))
        c = float(cfg.get("nominal_sound_speed", acoustics.NOMINAL_SOUND_SPEED))
        wavelength = c / fc
        arr = cfg["array"]
        if "layouts" in arr:
            # Several named element layouts for one board (e.g. dense and sparse).
            choice = arr.get("layout_choice")
            # Keys given beside `layouts` (for example `array.count` in an override) win
            # over the chosen layout's own, so an array scales with one number.
            top = {k: v for k, v in arr.items() if k not in ("layouts", "layout_choice")}
            arr = {**arr["layouts"][choice], **top}
        positions = _positions(arr)
        n = positions.shape[0]

        rx = cfg["receive"]
        for key, where in NOT_SIMULATED_RECEIVE_KEYS.items():
            if key in rx:
                raise ValueError(f"receive.{key} is not simulated (it would look modelled and change nothing): {where}")
        if rx.get("reference"):
            from .sim.receiver import REFERENCE_MODES
            if rx["reference"].get("mode") not in REFERENCE_MODES:
                raise ValueError(f"receive.reference.mode {rx['reference'].get('mode')!r}: the simulated modes are {sorted(REFERENCE_MODES)}")
        mux = rx.get("mux")
        if mux:
            groups = []
            for g in mux["groups"]:
                if isinstance(g, dict):
                    groups.append(list(range(int(g["first"]), int(g["first"]) + int(g["count"]))))
                else:
                    groups.append([int(i) for i in g])
        else:
            groups = [list(range(n))]
        for g in groups:
            if min(g) < 0 or max(g) >= n:
                raise ValueError("mux group refers to an element outside the array")

        boards_cfg = rx.get("boards")
        if not boards_cfg and rx.get("board_size"):
            # Capture boards of a fixed channel count, as many as the array needs, so the
            # array scales by changing `array.count` alone. Ideal clocks unless
            # `receive.boards` spells them out.
            size = int(rx["board_size"])
            boards_cfg = [{"first": first, "count": min(size, n - first)} for first in range(0, n, size)]
        channel_board = np.zeros(n, dtype=np.int32)
        boards: list[dict] = [{}]
        if boards_cfg:
            boards = []
            for b_index, b in enumerate(boards_cfg):
                boards.append(dict(b))
                first, count = int(b["first"]), int(b["count"])
                channel_board[first : first + count] = b_index

        return cls(
            name=str(cfg["name"]),
            mode=str(cfg["mode"]),
            config=cfg,
            positions=positions,
            boresight=_unit(arr.get("boresight", [0.0, 1.0, 0.0])),
            beamwidth_az=_beamwidth(arr, "beamwidth_az", "element_width", wavelength),
            beamwidth_el=_beamwidth(arr, "beamwidth_el", "element_height", wavelength),
            mux_groups=groups,
            channel_board=channel_board,
            boards=boards,
            nominal_sound_speed=c,
            sampled_waveform=sampled,
        )

    # --- derived quantities -------------------------------------------------

    @property
    def n_elements(self) -> int:
        return self.positions.shape[0]

    @property
    def waveform_config(self) -> dict:
        return self.config["transmit"]["waveform"]

    @property
    def center_frequency(self) -> float:
        w = self.waveform_config
        return 0.5 * (float(w["f_start"]) + float(w.get("f_stop", w["f_start"])))

    @property
    def bandwidth(self) -> float:
        w = self.waveform_config
        return abs(float(w.get("f_stop", w["f_start"])) - float(w["f_start"]))

    @property
    def sample_rate(self) -> float:
        return float(self.config["receive"]["sample_rate"])

    @property
    def aperture(self) -> float:
        """Largest distance between two elements, m."""
        p = self.positions
        return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))

    @property
    def source_level_db_rms(self):
        """Configured source level, dB re 1 uPa at 1 m RMS, or None (an uncalibrated sonar)."""
        return source_level_db_rms(self.waveform_config)

    @property
    def source_energy_db(self):
        """Transmitted energy at 1 m, dB re 1 uPa^2 s: SL + 10 log10(integral |b|^2 du / 1 s) (the accounting above
        `envelope_energy_s`). None without a source level or for a pulse with no stated envelope energy."""
        sl = self.source_level_db_rms
        e = envelope_energy_s(self.waveform_config, self.sampled_waveform)
        return None if sl is None or e is None else float(sl + 10.0 * np.log10(e))

    def segment_energies_db(self):
        """A train's energy per segment at 1 m (dB re 1 uPa^2 s; None for a gap), or None for any other pulse."""
        segs = getattr(self.sampled_waveform, "segments", None)
        sl = self.source_level_db_rms
        if not segs or sl is None:
            return None
        return [None if s.kind == "gap" else float(sl + 10.0 * np.log10(s.energy)) for s in segs]

    def amplitude_at_1m(self) -> float:
        """PEAK echo amplitude from a 0 dB target at 1 m range each way, in sample units.

        With a source level and receive sensitivity this is volts at the ADC:
        sqrt(2) 10^((SL + sensitivity + gain) / 20), SL the RMS source level in dB re 1 uPa at 1 m
        (`peak_pressure_at_1m`: the one place the sqrt(2) is applied) and sensitivity in dB re 1 V/uPa.
        Without them the chain is uncalibrated and `amplitude` (a peak value) is used.
        """
        w = self.waveform_config
        rx = self.config["receive"]
        sl = self.source_level_db_rms
        sens = rx.get("sensitivity_db")
        if sl is not None and sens is not None:
            return peak_pressure_at_1m(sl) * 10.0 ** ((float(sens) + float(rx.get("gain_db", 0.0))) / 20.0)
        return float(w.get("amplitude", 1.0))

    # --- record template ----------------------------------------------------

    def window(self, sound_speed: float | None = None) -> tuple[float, int]:
        """(start_time, n_samples) of the receive window."""
        rx = self.config["receive"]
        win = rx["window"]
        if "range" in win:
            c = sound_speed or self.nominal_sound_speed
            r0, r1 = (float(x) for x in win["range"])
            start = 2.0 * r0 / c
            stop = 2.0 * r1 / c + float(self.waveform_config["duration"])
            return start, int(np.ceil((stop - start) * self.sample_rate))
        return float(win["start"]), int(win["samples"])

    def make_record(self, pose: "core.Pose", *, sequence: int = 0, sub_sequence: int = 0,
                    time_ns: int = 0, environment: dict | None = None,
                    tx_pose: "core.Pose | None" = None) -> "core.PingRecord":
        """An empty (all-zero samples) record for one transmit of this sonar."""
        cfg = self.config
        rx = cfg["receive"]
        env = environment or {}
        group = self.mux_groups[sub_sequence % len(self.mux_groups)]
        c = float(env.get("sound_speed", self.nominal_sound_speed))
        start, n_samples = self.window(c)

        r = core.PingRecord()
        s = r.samples
        kind = _KINDS[rx.get("kind", "real")]
        dtype = np.complex64 if kind == core.SampleKind.ComplexBaseband else np.float32
        s.kind = kind
        s.array = np.zeros((len(group), n_samples), dtype=dtype)
        s.kind = kind
        s.sample_rate = self.sample_rate
        s.center_frequency = self.center_frequency
        s.start_time = start
        r.samples = s

        w = r.transmit.waveform
        wc = self.waveform_config
        w.type = _WAVEFORMS[wc["type"]]
        w.f_start = float(wc["f_start"])
        w.f_stop = float(wc.get("f_stop", wc["f_start"]))
        w.duration = float(wc["duration"])
        w.window_alpha = float(wc.get("window_alpha", 0.0))
        if self.sampled_waveform is not None:
            # Existing record fields: type Arbitrary, band, duration, the real replica and its rate.
            self.sampled_waveform.to_record(w)
        # The boundary between the two conventions. Configured levels are RMS; a record's `source_level_db` and
        # `amplitude` are PEAK values (core/record.hpp): the level of the pulse's peak pressure, 3.01 dB above the
        # configured figure, so that inside a record amplitude = 10^((source_level_db + sensitivity + gain) / 20).
        # `record_source_level_db_rms` goes back.
        sl = self.source_level_db_rms
        w.source_level_db = float(sl) + PEAK_OVER_RMS_DB if sl is not None else float("nan")
        w.amplitude = self.amplitude_at_1m()
        t = r.transmit
        t.waveform = w
        t.time_ns = int(time_ns)
        txc = cfg["transmit"]
        if txc.get("kind", "point") == "plane_wave":
            t.kind = core.TransmitKind.PlaneWave
            t.direction = tuple(_unit(txc.get("direction", self.boresight)))
        r.transmit = t

        geo = r.geometry
        geo.rx_pose = pose
        geo.tx_pose = tx_pose if tx_pose is not None else pose
        elements = []
        for i in group:
            e = core.Element()
            e.lever_arm = tuple(self.positions[i])
            e.boresight = tuple(self.boresight)
            e.beamwidth_az = self.beamwidth_az
            e.beamwidth_el = self.beamwidth_el
            elements.append(e)
        geo.rx = elements
        txe = core.Element()
        txe.lever_arm = tuple(float(x) for x in txc.get("lever_arm", [0.0, 0.0, 0.0]))
        txe.boresight = tuple(_unit(txc.get("boresight", self.boresight)))
        wavelength = c / self.center_frequency
        txe.beamwidth_az = _beamwidth(txc, "beamwidth_az", "aperture_width", wavelength)
        txe.beamwidth_el = _beamwidth(txc, "beamwidth_el", "aperture_height", wavelength)
        geo.tx = txe
        if env.get("frame_origin_lla") is not None:
            geo.frame_origin_lla = tuple(env["frame_origin_lla"])
        r.geometry = geo

        e = r.environment
        e.sound_speed = c
        for key in ("temperature", "salinity", "water_depth"):
            if env.get(key) is not None:
                setattr(e, key, float(env[key]))
        r.environment = e

        r.set_default_channel_tables()
        m = r.metadata
        m.array_id = self.name
        m.mode = _MODES[self.mode]
        m.sequence = int(sequence)
        m.sub_sequence = int(sub_sequence % len(self.mux_groups))
        m.channel_element = [int(i) for i in group]
        boards = []
        for b in self.boards:
            board = core.Board()
            if b.get("reference_channel") is not None:
                board.reference_channel = int(b["reference_channel"])
            boards.append(board)
        m.boards = boards
        m.channel_board = [int(self.channel_board[i]) for i in group]
        m.hardware_gain_db = float(rx.get("gain_db", 0.0))
        tvg = rx.get("tvg")
        if rx.get("gain_ramp"):
            # A named hardware model (gpusonar.hardware) instead of explicit knots.
            if tvg:
                raise ValueError("give receive.tvg or receive.gain_ramp, not both")
            from .hardware import expand_gain_ramp
            tvg = expand_gain_ramp(rx["gain_ramp"])
        if tvg:
            m.tvg_time = [float(x) for x in tvg["time"]]
            m.tvg_gain_db = [float(x) for x in tvg["gain_db"]]
        m.adc_bits = int(rx.get("adc_bits", 0))
        m.adc_full_scale = float(rx["adc_full_scale"]) if rx.get("adc_full_scale") is not None else float("nan")
        r.metadata = m
        return r


def resolve_environment(env: dict | None, depth: float = 0.0) -> dict:
    """The environment the simulator works with (gpusonar/environment.py `resolve`): sound speed filled in from
    temperature and salinity, a profile, or `acoustics.NOMINAL_SOUND_SPEED`; `env` may be a mapping, the name
    of an environment, or None."""
    from . import environment
    return environment.resolve(env, depth)
