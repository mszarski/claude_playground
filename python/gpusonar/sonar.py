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
}
_KINDS = {
    "real": core.SampleKind.RealPassband,
    "baseband": core.SampleKind.ComplexBaseband,
}


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
    nominal_sound_speed: float = 1500.0

    # --- construction -------------------------------------------------------

    @classmethod
    def load(cls, name_or_path, overrides: dict | None = None) -> "SonarModel":
        return cls.from_config(load_sonar_config(name_or_path, overrides))

    @classmethod
    def from_config(cls, cfg: dict) -> "SonarModel":
        cfg = apply_fov(cfg)
        tx_wf = cfg["transmit"]["waveform"]
        fc = 0.5 * (float(tx_wf["f_start"]) + float(tx_wf.get("f_stop", tx_wf["f_start"])))
        c = float(cfg.get("nominal_sound_speed", 1500.0))
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

    def amplitude_at_1m(self) -> float:
        """Peak echo amplitude from a 0 dB target at 1 m range each way, in sample units.

        With a source level and receive sensitivity this is volts at the ADC:
        10^((SL + sensitivity + gain) / 20), SL in dB re 1 uPa at 1 m and sensitivity
        in dB re 1 V/uPa. Without them the chain is uncalibrated and `amplitude` is used.
        """
        w = self.waveform_config
        rx = self.config["receive"]
        sl = w.get("source_level_db")
        sens = rx.get("sensitivity_db")
        if sl is not None and sens is not None:
            return 10.0 ** ((float(sl) + float(sens) + float(rx.get("gain_db", 0.0))) / 20.0)
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
        w.source_level_db = float(wc["source_level_db"]) if wc.get("source_level_db") is not None else float("nan")
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
    """Fill in sound speed from temperature and salinity when it is not given."""
    env = dict(env or {})
    if env.get("sound_speed") is None:
        if env.get("temperature") is None:
            env["sound_speed"] = 1500.0
        else:
            env["sound_speed"] = acoustics.sound_speed(
                env["temperature"], env.get("salinity", 35.0), env.get("depth", depth))
    return env
