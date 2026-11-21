"""The long-range array chain, one ping at a time: the part that runs on a vehicle.

    source (boards over UDP, a recording, or the simulator)
      -> record raw pings                      (always: the recording is the flight data)
      -> synchronise the capture boards        (when there are several: shared reference, sync.py)
      -> prepare (baseband, pulse compression)
      -> beamform (range x bearing)
      -> detect (cell-averaging CFAR, clustering)
      -> track in a world-fixed frame (ownship's motion removed)
      -> classify (mover / static / multipath / uncertain)
      [with `elevation` and an array of stacked rows: elevation and depth of each detection,
       bearing corrected for it, depth carried by the tracks (elevation.py)]
      -> report: one JSON object per ping, to a file and optionally a UDP port

The same object runs on simulated pings, on a replayed recording and on the live stream,
so what is tested ashore is what runs at sea. Every intermediate product can be tapped
(gpusonar/taps.py) for the first `taps.pings` pings.
"""

from __future__ import annotations

import copy
import json
import socket
import time

import numpy as np

from . import _gpusonar as core
from . import classify, detect, flank
from .taps import TapWriter

DEFAULTS = {
    "bearings_deg": [-40.0, 40.0, 0.5],        # first, last, step: from broadside
    "window": "hann",                          # aperture weighting
    "sync": None,                              # several boards: {sonar: popoto_4x7} or {reference: {...}}, see sync.py
    "blanking": None,                          # {recovery_s, taper_s, mode: zero | ignore}: the projector's own sound overloads the
                                               # inputs for the pulse and their recovery after it: blank that, report no nearer
    "decimate": None,                          # {} or {output_rate: ..., ...}: band-select and decimate a raw ping to complex
                                               # baseband first, boards synchronised in the same stage (libs/live decimate.hpp)
    "beamformer": "backprojection",            # the kernel; or "farfield" / "mvdr" (flank.py, the C++ beamformers). All three run
                                               # on `backend` below, host or CUDA
    "backend": "auto",                         # "auto": CUDA when built and present, else host; "host"; "cuda"
    "look": "starboard",                       # which side of a line array the bearings are on
    # Sizes that follow from the sonar are None here: derived from the first ping's band, pulse and array by
    # libs/live derive.hpp (the one source: 2 m cells, 100 m guard, 300 m training, 15 m and 1 deg tracker sigmas, 3 deg
    # multipath gate, 60 m fit limit for a kilometre-range array of 2 kHz band or more; more for a narrower band or a
    # wider beam). A number given in the configuration wins. `LiveChain.resolved` says what was used.
    "range_cell_m": None,                      # power is averaged into cells this long before detection
    "cfar": {"pfa": 1e-5, "guard_m": None, "train_m": None,      # pfa: per cell, for cell_statistic mean and peak
             # cell_statistic multiscale (libs/live detector.hpp): the false-alarm budget is per ping, shared between the
             # integration lengths (None: the range cell, beside the single sample; [1, 4, 16] adds 4 and 16 cells) and the
             # map's independent cells; a flagged cell within sidelobe_margin_db of a stronger cell's sidelobe envelope is
             # put down to it; flagged cells of one bearing within merge_m in range (None: tracker.range_sigma) are one detection.
             "false_alarms_per_ping": 1e-3, "integrate_cells": None, "sidelobe_margin_db": 10.0, "sidelobe_floor_db": -40.0,
             # search_factor None: the independent tests follow the threshold and the rows, with measured constants
             # (libs/live detector.hpp, kSearchSample; tools/detector_search_factor.py). A number: that many tests per
             # range resolution and beamwidth whatever the threshold and the rows (2.5: the detector as it was).
             "search_factor": None, "merge_m": None},
    "min_snr_db": 9.0,
    "cell_statistic": "multiscale",            # how a range cell of the finer map is tested (docs/DEPLOYMENT.md, "What snr_db is"):
                                               # "multiscale": on its strongest sample, its mean and the means of several cells, each
                                               # against the threshold of its own statistic; "mean": its mean power against the
                                               # one-sample threshold (the detector until now); "peak": its strongest sample likewise
    "engine": "python",                        # "cpp": libs/live does every stage after recording (what build/bin/sonar_live runs)
    "doa": "map",                              # "sbl": sparse Bayesian bearings at each detected range (engine cpp)
    "sbl": {"grid_step_deg": 0.25, "max_sources": 6, "min_snr_db": 6.0, "dynamic_range_db": 20.0,
            "max_gates": 24, "min_contrast_db": 8.0, "max_iterations": 20, "tolerance": 0.01},
    "tracker": {"range_sigma": None, "bearing_sigma_deg": None, "confirm_hits": 3, "max_misses": 2,
                "initial_speed_sigma": 8.0, "process_noise": 0.05,
                "association": "likelihood"},     # or "in_order", the rule before D-141 (gpusonar/detect.py, Tracker)
    "elevation": None,                         # stacked rows: {} or {sector_deg: ..., ...} switches elevation and depth on (elevation.py)
    "sound_speed": None,                       # m/s at the vehicle, when navigation does not send it: replaces the record's
    "dead_channels": [],                       # channel rows (after synchronisation) left out of every stage
    "classify": {"min_speed": 1.0, "static_speed": 1.0, "max_fit_rms": None,
                 "multipath": {"max_excess_range": 800.0, "bearing_deg": None, "velocity": 2.5}},
}
# The derived settings: (path in the configuration, C++ option name, factor from the configuration's unit).
DERIVED = ((("range_cell_m",), "range_cell_m", 1.0), (("cfar", "guard_m"), "cfar_guard_m", 1.0), (("cfar", "train_m"), "cfar_train_m", 1.0),
           (("tracker", "range_sigma"), "tracker_range_sigma", 1.0), (("tracker", "bearing_sigma_deg"), "tracker_bearing_sigma", np.pi / 180.0),
           (("classify", "max_fit_rms"), "classify_max_fit_rms", 1.0),
           (("classify", "multipath", "bearing_deg"), "multipath_bearing", np.pi / 180.0))
REPORT_UDP_MAX_BYTES = 65000        # a report line longer than this goes out as several datagrams (split_report)


def _given(cfg: dict) -> dict:
    """The derived settings the configuration gave, by C++ option name and in its units."""
    out = {}
    for path, name, factor in DERIVED:
        v = cfg
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
        if v is not None:
            out[name] = float(np.radians(v)) if factor != 1.0 else float(v)
    return out


def finite_or_null(obj):
    """The report with every non-finite number (NaN, infinity) as None, so that its JSON line says `null`, as
    libs/live Report::json writes it (a `snr_db` the detector could not form, for one), and stays valid JSON."""
    if isinstance(obj, dict):
        return {k: finite_or_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_or_null(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def split_report(report: dict, max_bytes: int = REPORT_UDP_MAX_BYTES) -> list[str]:
    """A report as the text of its UDP datagrams, none longer than `max_bytes`: the one line when it fits.
    Otherwise several lines, each a complete report of the same ping holding a share of the detections
    and tracks and a field "part": [k, n] (k from 1). A reader that knows nothing of parts parses each
    as a report with fewer detections; one that does joins the lists of the n parts of a ping
    (`join_report_parts`). The line is never cut. libs/live Report::datagrams does the same. Non-finite
    numbers are written as null (`finite_or_null`)."""
    report = finite_or_null(report)
    line = json.dumps(report)
    if len(line.encode()) <= max_bytes:
        return [line]
    head = dict(report, detections=[], tracks=[], part=[999, 999])
    room = max_bytes - len(json.dumps(head).encode())
    items = [("detections", d) for d in report["detections"]] + [("tracks", t) for t in report["tracks"]]
    shares, now, used = [], [], 0
    for item in items:
        size = len(json.dumps(item[1]).encode()) + 2
        if now and used + size > room:
            shares.append(now)
            now, used = [], 0
        now.append(item)
        used += size
    shares.append(now)
    out = []
    for k, share in enumerate(shares):
        part = dict(report, detections=[v for key, v in share if key == "detections"], tracks=[v for key, v in share if key == "tracks"],
                    part=[k + 1, len(shares)])
        out.append(json.dumps(part))
    return out


# --- what snr_db is, and how it relates to a link budget ---------------------------------------------------------------
#
# A link budget (gpusonar.sonar_equation, buildspec.performance) states the peak signal power over the mean noise power
# at the output of an unweighted beamformer of one row. A detection's `snr_db` is 10 log10 of what the detector saw of
# that over its background (the mean of the cells `cfar.train_m` either side along range, beyond `cfar.guard_m`), on the
# beamformed, pulse-compressed map. What it saw depends on `cell_statistic`:
#   multiscale  the strongest SAMPLE of the detection. Lower than the budget's figure by the aperture window's loss
#               against uncorrelated noise, N sum(w^2) / (sum w)^2 (1.76 dB for Hann), and by where the echo falls
#               between the map's samples (0.5 dB on average at 0.75 of a resolution or finer).
#   mean        the strongest range CELL's mean power: lower by the window and by the averaging of an echo one range
#               resolution long with the other samples of the cell (7.7 dB for a 2 m cell at 2 kHz of band).
#   peak        the strongest sample over the mean of the cells' largest samples, which is the k-th harmonic number
#               above the mean noise power (4.3 dB for k = 8).
# An echo spread over the cell in range (a long target, multipath) loses less under "mean" than this figure, which is
# for a point; and the budget's source level convention (peak or rms) is a separate 3 dB the budget must state.
# Whether a given budget SNR is DETECTED is a probability, not a threshold: `detection_probability`.

def _map_geometry(record, cfg: dict) -> dict:
    """The chain's range map for this record as the backprojection beamformer lays it: cell, samples per cell, their
    step, the number of cells, the CFAR window in cells and the derived settings."""
    given = _given(cfg)
    given["bearing_step_deg"] = float(cfg["bearings_deg"][2])
    d = core.live_derive(record, given)
    w, s = record.transmit.waveform, record.samples
    c = record.environment.sound_speed
    band = abs(w.f_stop - w.f_start) or 1.0 / w.duration
    resolution = c / (2.0 * band)
    cell = float(d["range_cell_m"])
    k = max(int(np.ceil(cell / (0.75 * resolution))), 1)
    step = cell / k
    r0 = 0.5 * c * s.start_time
    r1 = 0.5 * c * (s.start_time + (s.n_samples - 1) / s.sample_rate - w.duration)
    guard, train = core.cfar_window(cell, float(resolution), guard_m=float(d["cfar_guard_m"]), train_m=float(d["cfar_train_m"]))
    return {"cell": cell, "k": k, "step": step, "n_cells": max(int((r1 - r0) / step) // k, 1), "guard": int(guard), "train": int(train),
            "resolution": resolution, "derived": d}


def _rows_of(record, cfg: dict) -> int:
    """Rows whose maps the chain adds in power: the array's rows with `elevation`, else one."""
    if cfg.get("elevation") is None:
        return 1
    from .elevation import DEFAULTS as EL
    e = _merge(EL, cfg["elevation"])
    tolerance = core.live_row_tolerance(record) if e["row_tolerance_m"] is None else float(e["row_tolerance_m"])
    rows = core.live_find_rows(record, tolerance, int(e["min_row_elements"]))
    return len(rows) if len(rows) >= 2 else 1


def detector_plan(record, cfg: dict | None = None, rows: int | None = None) -> dict:
    """The multiscale detector's plan (libs/live detector.hpp) for the map the chain makes of this record: the tests,
    their looks and thresholds for `cfar.false_alarms_per_ping`, the sidelobe envelopes, what is merged. `record` may be
    the sonar's empty template (`SonarModel.make_record`); `rows`: maps added in power (default: the array's rows when
    `elevation` is configured, else 1)."""
    cfg = _merge(DEFAULTS, cfg)
    g = _map_geometry(record, cfg)
    rows = _rows_of(record, cfg) if rows is None else int(rows)
    b = cfg["bearings_deg"]
    bearings = np.radians(np.arange(b[0], b[1] + 0.5 * b[2], b[2]))
    o = detector_options(cfg)
    o.setdefault("merge_m", float(g["derived"]["tracker_range_sigma"]))
    plan = core.live_detector_plan(record, g["n_cells"], g["k"], g["step"], [float(v) for v in bearings], rows, g["guard"], g["train"],
                                   cfg["window"] == "hann" and cfg["beamformer"] != "mvdr", o)
    plan["train_cells"] = 2 * g["train"]
    return plan


def _window_loss(n: int, window: str) -> float:
    """The aperture window's loss against uncorrelated noise, as a power ratio >= 1."""
    if window != "hann":
        return 1.0
    u = (np.arange(n) - 0.5 * (n - 1)) / n                  # as the map lays it: zero half a pitch beyond the ends
    weights = np.cos(np.pi * u) ** 2
    return float(n * np.sum(weights**2) / np.sum(weights) ** 2)


def _cell_response(resolution: float, cell: float, k: int, swept: bool = True):
    """The compressed pulse of a point echo on the map's samples, the echo anywhere within one cell:
    [where the echo is, cell (7 about it), sample of the cell]."""
    step = cell / k
    offsets = (np.arange(400) + 0.5) / 400.0 * cell
    samples = step * np.arange(-3 * k, 4 * k)[None, :] - offsets[:, None]
    response = np.sinc(samples / resolution) ** 2 if swept else np.maximum(1.0 - np.abs(samples) / resolution, 0.0) ** 2
    return response.reshape(len(offsets), 7, k)


def snr_loss_db(record, range_cell_m: float, window: str = "hann", cell_statistic: str = "mean", row_elements: int | None = None) -> dict:
    """What the chain's `snr_db` reads below a link budget's SNR (peak signal over mean noise at an unweighted
    beamformer's output) for a point echo, dB: {"window_db", "cell_db", "total_db", "samples_per_cell"}.
    `record` gives the band and the sound speed; `row_elements` the hydrophones of one row (default: all of them).
    The cell figure is the mean over where the echo falls among the cell's samples (tests/python/test_hardcoded.py and
    test_detector.py hold it to the chain's own numbers within 1 dB)."""
    w = record.transmit.waveform
    c = record.environment.sound_speed
    band = abs(w.f_stop - w.f_start) or 1.0 / w.duration
    resolution = c / (2.0 * band)
    n = int(row_elements or record.samples.n_channels)
    window_db = float(10.0 * np.log10(_window_loss(n, window)))
    k = max(int(np.ceil(range_cell_m / (0.75 * resolution))), 1)
    cells = _cell_response(resolution, range_cell_m, k)
    if cell_statistic in ("peak", "multiscale"):
        signal = cells.max(axis=2).max(axis=1)
        # peak: the background is the mean of the largest of k independent noise powers; multiscale: the mean noise power
        background = float(np.sum(1.0 / np.arange(1, k + 1))) if cell_statistic == "peak" else 1.0
    else:
        signal = cells.mean(axis=2).max(axis=1)
        background = 1.0
    cell_db = float(-10.0 * np.log10(np.mean(signal) / background))
    return {"window_db": window_db, "cell_db": cell_db, "total_db": window_db + cell_db, "samples_per_cell": k}


def detection_threshold_db(cfg: dict | None = None, train_cells: int = 300, record=None, rows: int | None = None) -> float:
    """The `snr_db` a detection needs to be reported: the larger of `min_snr_db` and the detector's own threshold over
    its background. mean, peak: cell-averaging CFAR at `cfar.pfa` with `train_cells` training cells (10.7 dB at 1e-5).
    multiscale: the single-sample test's threshold for the map of `record` at `cfar.false_alarms_per_ping` (13.1 dB
    for one row of the long-range array's 1.4 km window at 1e-3, 9.5 dB for three rows added); a detection that only
    the cell-mean test found reads lower, down to `min_snr_db`."""
    cfg = _merge(DEFAULTS, cfg)
    if cfg["cell_statistic"] == "multiscale":
        if record is None:
            raise ValueError("detection_threshold_db: the multiscale detector's threshold follows from the map: give the record")
        plan = detector_plan(record, cfg, rows)
        s = plan["scales"][0]
        factor = s["factor"] * core.training_correction(s["pfa"], plan["train_cells"] * plan["train_looks_per_cell"])
        return float(max(cfg["min_snr_db"], 10.0 * np.log10(factor)))
    return float(max(cfg["min_snr_db"], 10.0 * np.log10(detect.cfar_threshold_factor(cfg["cfar"]["pfa"], int(train_cells)))))


def detection_probability(snr_db, record, cfg: dict | None = None, rows: int | None = None, row_elements: int | None = None,
                          fluctuation: str = "none"):
    """The probability that the chain as configured detects a POINT echo whose link-budget SNR is `snr_db` (peak signal
    over mean noise at the output of an unweighted beamformer of ONE row; a number or an array), on one ping.

    `record`: a ping or the sonar's empty template (band, window, array). `rows`: maps added in power (default: the
    array's rows when `elevation` is configured, else 1); `row_elements`: hydrophones of one row (default: the channels
    over the rows). `fluctuation`: "none" (a steady echo) or "swerling1" (its power exponentially distributed from ping
    to ping, the same in every row).

    How: the echo's power on the map is the budget's less the aperture window's loss, times the compressed pulse's
    response at the sample (or its mean over the cell) for every place the echo can fall between samples; the statistic
    with that signal is a noncentral chi-square (2 x rows degrees of freedom for a sample; 2 x rows x looks for a mean,
    which is exact for the noise and an approximation for the signal's share); it is compared with the detector's own
    threshold (multiscale: per test, with the training correction; mean and peak: the cell-averaging factor at
    `cfar.pfa`, as the chain applies it) and with `min_snr_db`. multiscale: the better of its sample test and its
    cell-mean test at each place (their union is slightly higher). tests/python/test_detector.py holds this to the
    chain on simulated pings. Sources: docs/REFERENCES.md Sec. 12.1."""
    from scipy import stats
    cfg = _merge(DEFAULTS, cfg)
    statistic = cfg["cell_statistic"]
    g = _map_geometry(record, cfg)
    rows = _rows_of(record, cfg) if rows is None else int(rows)
    n = int(row_elements or record.samples.n_channels // rows)
    w = record.transmit.waveform
    swept = abs(w.f_stop - w.f_start) > 0.0
    cells = _cell_response(g["resolution"], g["cell"], g["k"], swept)         # [place, cell, sample]
    loss = _window_loss(n, cfg["window"])
    floor = 10.0 ** (cfg["min_snr_db"] / 10.0)
    n_train = 2 * g["train"]
    corr = core.live_sample_correlation(record, g["step"], g["k"])
    w_mean = core.mean_power_weights(corr, g["k"])
    looks = core.equivalent_looks(w_mean)
    best_sample = cells.max(axis=2).max(axis=1)                                # [place]
    best_mean = cells.mean(axis=2).max(axis=1)
    tests = []                                                                 # (threshold, looks of the statistic, signal's share [place])
    if statistic == "multiscale":
        plan = detector_plan(record, cfg, rows)
        for s in plan["scales"]:
            factor = s["factor"] * core.training_correction(s["pfa"], plan["train_cells"] * plan["train_looks_per_cell"])
            if s["cells"] == 0:
                tests.append((max(factor, floor), 1.0, best_sample))
            elif s["cells"] == 1:
                tests.append((factor, looks, best_mean))
            # longer means: a point echo adds nothing they see before the two above
    elif statistic == "mean":
        tests.append((max(detect.cfar_threshold_factor(cfg["cfar"]["pfa"], n_train), floor), looks, best_mean))
    else:
        harmonic = float(np.sum(1.0 / np.arange(1, g["k"] + 1)))               # the background: the mean of the largest of k
        tests.append((max(detect.cfar_threshold_factor(cfg["cfar"]["pfa"], n_train), floor) * harmonic, 1.0, best_sample))
    snr = 10.0 ** (np.atleast_1d(np.asarray(snr_db, dtype=float)) / 10.0) / loss

    def steady(s):                                                             # s: [snr]; mean over the places of the best test
        pd = np.zeros((len(s), cells.shape[0]))
        for threshold, m, share in tests:
            dof = 2.0 * rows * m
            pd = np.maximum(pd, stats.ncx2.sf(dof * threshold, dof, np.maximum(dof * s[:, None] * share[None, :], 1e-12)))
        return pd.mean(axis=1)

    if fluctuation == "none":
        out = steady(snr)
    elif fluctuation == "swerling1":
        x, wx = np.polynomial.laguerre.laggauss(48)
        out = sum(wi * steady(snr * xi) for xi, wi in zip(x, wx))
    else:
        raise ValueError(f"fluctuation: {fluctuation!r} (none or swerling1)")
    return float(out[0]) if np.ndim(snr_db) == 0 else out


def budget_threshold_db(record, cfg: dict | None = None, range_cell_m: float | None = None, row_elements: int | None = None,
                        pd: float | None = None, rows: int | None = None, fluctuation: str = "none") -> float:
    """The link-budget SNR (peak signal over mean noise at the output of an unweighted beamformer of one row, source
    level as the record's) from which the chain as configured reports a point echo: what a range prediction should use
    as its detection threshold in place of a typed 13 dB.

    With `pd` (or cell_statistic multiscale, where it defaults to 0.5): the SNR at which `detection_probability` is
    `pd`, for `rows` maps added in power. Without, for mean and peak: detection_threshold_db + snr_loss_db, as before
    (the same thing at a probability near one half). `range_cell_m`: the cell the chain resolved (default: the
    configuration's, else the derivation's for this record)."""
    cfg = _merge(DEFAULTS, cfg)
    if range_cell_m is not None:
        cfg = dict(cfg, range_cell_m=float(range_cell_m))
    if pd is None and cfg["cell_statistic"] != "multiscale":
        cell = cfg["range_cell_m"] or core.live_derive(record, {})["range_cell_m"]
        train = cfg["cfar"]["train_m"] or core.live_derive(record, {})["cfar_train_m"]
        loss = snr_loss_db(record, float(cell), cfg["window"], cfg["cell_statistic"], row_elements)
        return detection_threshold_db(cfg, int(round(2.0 * train / cell))) + loss["total_db"]
    from scipy import optimize
    want = 0.5 if pd is None else float(pd)
    return float(optimize.brentq(lambda s: detection_probability(s, record, cfg, rows, row_elements, fluctuation) - want, -20.0, 80.0, xtol=1e-3))


def join_report_parts(parts: list[dict]) -> dict:
    """The report the parts of one ping were split from (any order; all n must be there)."""
    parts = sorted(parts, key=lambda p: p["part"][0])
    if [p["part"][0] for p in parts] != list(range(1, parts[0]["part"][1] + 1)) or len({p["ping"] for p in parts}) != 1:
        raise ValueError("report parts: not the n parts of one ping")
    out = {k: v for k, v in parts[0].items() if k != "part"}
    out["detections"] = [d for p in parts for d in p["detections"]]
    out["tracks"] = [t for p in parts for t in p["tracks"]]
    return out


def _merge(base: dict, over: dict | None) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


# The words a chain setting may be; libs/live options.cpp refuses anything else in the same way.
CHOICES = {"beamformer": ("backprojection", "farfield", "mvdr"), "look": ("starboard", "port"), "window": ("hann", "rect"),
           "backend": ("auto", "host", "cuda")}


def check_choices(cfg: dict) -> None:
    """Refuse a (merged) chain configuration whose `beamformer`, `look`, `window` or `backend` is not one of its words:
    `look: Starboard` would otherwise look to port and a misspelt window be rectangular."""
    for key, allowed in CHOICES.items():
        if cfg.get(key) not in allowed:
            raise ValueError(f"chain.{key}: {cfg.get(key)!r} ({', '.join(allowed[:-1])} or {allowed[-1]})")


def cpp_chain_options(cfg: dict, reference: dict | None = None) -> dict:
    """The chain section of a deployment file as the options of the C++ chain (`core.LiveChainCpp`,
    and `key=value` arguments of build/bin/sonar_live through `program_options`). `cfg` is merged over the
    defaults first. `backend` is the configuration's word (auto, host or cuda); LiveChain replaces it with
    the backend it resolved."""
    cfg = _merge(DEFAULTS, cfg)
    check_choices(cfg)
    if cfg["cell_statistic"] not in CELL_STATISTICS:
        raise ValueError(f"cell_statistic: {cfg['cell_statistic']!r} (mean, peak or multiscale)")
    b, t, c, s = cfg["bearings_deg"], cfg["tracker"], cfg["classify"], cfg["sbl"]
    o = {"bearing_first_deg": b[0], "bearing_last_deg": b[1], "bearing_step_deg": b[2],
         "cfar_pfa": cfg["cfar"]["pfa"],
         "min_snr_db": cfg["min_snr_db"], "hann": cfg["window"] == "hann", "starboard": cfg["look"] == "starboard",
         "beamformer": cfg["beamformer"], "doa": cfg["doa"], "sbl_grid_step_deg": s["grid_step_deg"], "sbl_max_sources": int(s["max_sources"]),
         "sbl_min_snr_db": s["min_snr_db"], "sbl_dynamic_range_db": s["dynamic_range_db"],
         "sbl_max_gates": int(s["max_gates"]), "sbl_min_contrast_db": s["min_contrast_db"],
         "sbl_max_iterations": int(s["max_iterations"]), "sbl_tolerance": s["tolerance"],
         "tracker_confirm_hits": int(t["confirm_hits"]), "tracker_max_misses": int(t["max_misses"]),
         "tracker_initial_speed_sigma": t["initial_speed_sigma"], "tracker_process_noise": t["process_noise"],
         "tracker_association": t["association"],
         "classify_min_speed": c["min_speed"], "classify_static_speed": c["static_speed"],
         "multipath": bool(c.get("multipath")), "backend": cfg["backend"]}
    if cfg["cell_statistic"] == "peak":
        o["cell_peak"] = True
    o["cell_multiscale"] = cfg["cell_statistic"] == "multiscale"      # always said: the C++ default is not assumed
    if o["cell_multiscale"]:
        o.update({"cfar_" + k: v for k, v in detector_options(cfg).items()})
    # The sizes that follow from the sonar go to the C++ chain only when the configuration gives them: left out,
    # it derives them from the first ping (libs/live derive.hpp).
    given = _given(cfg)
    if not c.get("multipath"):
        given.pop("multipath_bearing", None)
    o.update(given)
    if c.get("multipath"):
        m = c["multipath"]
        o.update(multipath_max_excess_range=m["max_excess_range"], multipath_velocity=m["velocity"])
    if cfg.get("dead_channels"):
        o["dead_channels"] = [int(v) for v in cfg["dead_channels"]]
    if cfg.get("sound_speed"):
        o["sound_speed"] = float(cfg["sound_speed"])
    if decimate_options(cfg.get("decimate")) is not None:
        o["decimate"] = decimate_options(cfg["decimate"])
    if cfg.get("blanking") is not None:
        o["blanking"] = blanking_options(cfg["blanking"])
    if cfg.get("multipath_depth") is not None:          # hook: depth from multipath timing (multipath_depth.py)
        from .multipath_depth import cpp_options as multipath_options
        o["multipath_depth"] = multipath_options(cfg["multipath_depth"])
    if cfg.get("coverage") is not None:                 # hook: the coverage picture (coverage.py)
        from .coverage import chain_options as coverage_options       # refuses a coverage without the site's water depth and bottom
        o["coverage"] = dict(coverage_options(cfg["coverage"]), surface_z=float((cfg.get("elevation") or {}).get("surface_z", 0.0)))
    if cfg.get("elevation") is not None:
        from .elevation import DEFAULTS as EL
        from .elevation import normalise_ssp, profile_options
        e = _merge(EL, cfg["elevation"])
        e["ssp"] = normalise_ssp(e["ssp"])
        o["elevation"] = {"first_deg": float(e["sector_deg"][0]), "last_deg": float(e["sector_deg"][1]), "step_deg": float(e["sector_deg"][2]),
                          "range_window_m": float(e["range_window_m"] or 0.0), "max_detections": int(e["max_detections"]),
                          "first_arrival": e["arrival"] == "first", "first_arrival_db": float(e["first_arrival_db"]),
                          "first_arrival_snr_db": float(e["first_arrival_snr_db"]),
                          "min_quality": float(e["min_quality"]), "ambiguity_db": float(e["ambiguity_db"]),
                          "ambiguity_sigma": float(e["ambiguity_sigma"]), "alternatives_db": float(e["alternatives_db"]),
                          "sector_from_depth": e["sector_from_depth"] is not None,
                          "sector_bounces": float((e["sector_from_depth"] or {}).get("bounces", 1)),
                          "sector_margin": float(np.radians((e["sector_from_depth"] or {}).get("margin_deg", 1.0))),
                          "water_depth_sigma": float(e["water_depth_sigma_m"]),
                          "attitude_arrival": e["attitude_time"] == "arrival",
                          "attitude_rate_sigma": float(np.radians(e["attitude_rate_sigma_deg_s"])),
                          "use_gradient": e["gradient_per_s"] is not None, "gradient": float(e["gradient_per_s"] or 0.0),
                          "ray_arcs": e["ray_method"] == "arcs",
                          "sector_from_rays": e["sector_from_rays"] is not None,
                          "sector_rays_bounces": float((e["sector_from_rays"] or {}).get("bounces", 2)),
                          "sector_rays_margin": float(np.radians((e["sector_from_rays"] or {}).get("margin_deg", 1.0))),
                          "sector_rays_max_depth": float((e["sector_from_rays"] or {}).get("max_depth_m", 500.0)),
                          "multipath_sigma": e["multipath_sigma"] is not None,
                          "multipath_sigma_bounces": float((e["multipath_sigma"] or {}).get("bounces", 2)),
                          "multipath_sigma_max_depth": float((e["multipath_sigma"] or {}).get("max_depth_m", 500.0)),
                          "gradient_sigma": float(e["gradient_sigma_per_s"]),
                          "correct_bearing": bool(e["correct_bearing"]), "depth": bool(e["depth"]), "fold": bool(e["fold"]),
                          "classify": bool(e["classify"]), "classify_sigma": float(e["classify_sigma"]),
                          "water_depth": float(e["water_depth"] or 0.0), "surface_z": float(e["surface_z"]),
                          "ssp_depth": [float(v) for v in (e["ssp"] or {}).get("depth", [])],
                          "ssp_speed": [float(v) for v in (e["ssp"] or {}).get("speed", [])],
                          "roll_sigma": float(np.radians(e["roll_sigma_deg"])), "depth_process_noise": float(e["depth_process_noise"]),
                          "min_row_elements": int(e["min_row_elements"])}
        if e["row_tolerance_m"] is not None:
            o["elevation"]["row_tolerance_m"] = float(e["row_tolerance_m"])
        if e["profile"] is not None:
            o["elevation"]["profile"] = profile_options(e["profile"])
    if reference is not None:
        o["sync"] = {k: (bool(reference[k]) if k == "dedicated" else float(reference[k]))
                     for k in ("f_start", "f_stop", "duration", "lead", "tail", "dedicated", "min_quality", "search_time", "tolerance_us",
                               "clip_level") if reference.get(k) is not None}
    return o


CELL_STATISTICS = ("mean", "peak", "multiscale")


def detector_options(cfg: dict) -> dict:
    """The multiscale detector's options from a (merged) chain configuration, as libs/live DetectorOptions names them."""
    f = cfg["cfar"]
    o = {"false_alarms_per_ping": float(f["false_alarms_per_ping"]), "sidelobe_margin_db": float(f["sidelobe_margin_db"]),
         "sidelobe_floor_db": float(f["sidelobe_floor_db"])}
    if f.get("search_factor") is not None:
        o["search_factor"] = float(f["search_factor"])
    if f.get("integrate_cells") is not None:
        o["integrate_cells"] = [int(v) for v in f["integrate_cells"]]
    if f.get("merge_m") is not None:
        o["merge_m"] = float(f["merge_m"])
    return o


def sync_reference(cfg: dict) -> dict | None:
    """The shared reference signal a chain's `sync` section names, or None."""
    sc = cfg.get("sync")
    if not sc:
        return None
    if "reference" in sc:
        ref = sc["reference"]
    else:
        from .sonar import SonarModel
        ref = SonarModel.load(sc["sonar"], sc.get("sonar_overrides")).config["receive"]["reference"]
    # Beside `reference` or `sonar`: min_quality (a burst below it, or clipped, is not used and its board is flagged),
    # search_time (s either side of where a burst is expected; default 2 ms: boards that start further apart need more)
    # and tolerance_us (what the boards must agree within; default a twentieth of a period at the top of the band).
    extra = {k: float(sc[k]) for k in ("min_quality", "search_time", "tolerance_us") if sc.get(k) is not None}
    return dict(ref, **extra) if extra else ref


def program_stage_options(o: dict, record=None, chain_cfg: dict | None = None) -> dict:
    """The opt-in stages of `cpp_chain_options` (`multipath_depth`, `coverage`: flat mappings) as key=value settings of
    build/bin/sonar_live, in place: `<stage>=1` and `<stage>_<name>=value`, lists comma-separated. With `record` (a ping
    or the sonar's template) and the chain's configuration the coverage picture's detection-probability table goes in
    too (`coverage_pd_snr_db`, `coverage_pd_value`: the program cannot make it); without it the program's report says
    that the table is missing."""
    def text(v):
        if isinstance(v, (list, tuple)):
            return ",".join(f"{float(x):.10g}" for x in v)
        return int(v) if isinstance(v, bool) else v
    for stage in ("multipath_depth", "coverage"):
        flat = o.pop(stage, None)
        if flat is None:
            continue
        o[stage] = 1
        o.update({f"{stage}_{k}": text(v) for k, v in flat.items()})
        if stage == "coverage" and record is not None:
            from .coverage import pd_table
            snr, pd = pd_table(record, {k: v for k, v in (chain_cfg or {}).items() if k != "engine"})
            o.update(coverage_pd_snr_db=text(snr), coverage_pd_value=text(pd))
    return o


def _option_text(key: str, v):
    """One value of a key=value option as build/bin/sonar_live reads it (libs/live option_value.hpp): booleans 1 / 0,
    numbers (floats to 10 significant digits), lists comma-separated, words as they are. Anything else is refused here
    rather than written as Python text that the program would misread."""
    if isinstance(v, (bool, np.bool_)):
        return int(bool(v))
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10g}"
    if isinstance(v, str):
        if not v or any(ch in v for ch in "\n#=") or v.strip() != v:
            raise ValueError(f"{key}: {v!r} cannot be written as a key=value option")
        return v
    if isinstance(v, (list, tuple, np.ndarray)):
        items = [_option_text(key, x) for x in v]
        if any(isinstance(x, str) and "," in x for x in items):
            raise ValueError(f"{key}: {v!r}: a list item holds a comma")
        return ",".join(str(x) for x in items)
    raise ValueError(f"{key}: {v!r} ({type(v).__name__}) is not a value build/bin/sonar_live can read")


def program_options(chain: dict | None, record=None, deploy: dict | None = None) -> dict:
    """A deployment file's chain section as the key=value settings of build/bin/sonar_live: the ONE mapping that
    `tools/sonar_live.py --export-template` (the printed command) and `tools/build_from_spec.py` (chain.options) write.
    `record`: the sonar's template, for the coverage picture's detection-probability table. `deploy`: the whole
    deployment file, for the settings of the program around the chain (`source`: the board stream and decimation in
    the receiver; `nav`; `output.report_udp_max_bytes`). Returns {key: text or int}, every value as the program
    reads it (`_option_text`); nested sections are flattened to the keys libs/live options.cpp knows."""
    chain = chain or {}
    deploy = deploy or {}
    src = deploy.get("source") or {}
    o = cpp_chain_options(chain, sync_reference(chain))
    program_stage_options(o, record, chain)               # multipath_depth, coverage (with its detection-probability table)
    out = {}
    sync = o.pop("sync", None)
    out["look"] = "starboard" if o.pop("starboard") else "port"
    out["window"] = "hann" if o.pop("hann") else "rect"
    # Sizes the deployment file leaves out are not given either: the program derives them (derive.hpp).
    if "tracker_bearing_sigma" in o:
        out["tracker_bearing_sigma_deg"] = float(np.degrees(o.pop("tracker_bearing_sigma")))
    if "multipath_bearing" in o:
        out["multipath_bearing_deg"] = float(np.degrees(o.pop("multipath_bearing")))
    # How a range cell is tested, always said: the program's own default is not assumed.
    multiscale = o.pop("cell_multiscale", False)
    out["cell_statistic"] = "peak" if o.pop("cell_peak", False) else ("multiscale" if multiscale else "mean")
    el = o.pop("elevation", None)
    blanking = o.pop("blanking", None)
    dec = o.pop("decimate", None)
    out.update(o)
    if el is not None:
        # Stacked rows: elevation=1 and one elevation_<key> per setting (docs/CONFIGURATION.md).
        el = dict(el)
        out["elevation"] = 1
        for name in ("roll_sigma", "sector_margin", "sector_rays_margin"):
            el[f"{name}_deg"] = float(np.degrees(el.pop(name)))
        el["attitude_rate_sigma_deg_s"] = float(np.degrees(el.pop("attitude_rate_sigma")))
        for key in ("ssp_depth", "ssp_speed"):
            if not el.get(key):
                el.pop(key, None)
        profile = el.pop("profile", None)
        out.update({f"elevation_{k}": v for k, v in el.items()})
        if profile is not None:
            out["elevation_profile"] = 1
            out.update({f"elevation_profile_{k}": v for k, v in profile.items()})
    if sync:
        out["sync"] = 1
        out.update({f"sync_{k}": v for k, v in sync.items()})
    if blanking is not None:
        out["blanking"] = 1
        out.update({f"blanking_{k}": v for k, v in blanking.items()})
    # Decimate first: in the receiver (source.decimate: the recording holds decimated pings) or in the chain
    # (chain.decimate: the recording holds raw pings).
    streaming = decimate_options(src.get("decimate"))
    if streaming is not None or dec is not None:
        out["decimate"] = 1
        out["decimate_streaming"] = int(streaming is not None)
        out.update({f"decimate_{k}": v for k, v in (streaming if streaming is not None else dec).items()})
    # The program around the chain: the board stream's word, socket and queue; when to give up; where to listen.
    for key, option in (("sample_bytes", "stream_sample_bytes"), ("buffer_bytes", "stream_buffer_bytes"), ("queue_mb", "stream_queue_mb"),
                        ("timeout", "idle_seconds"), ("host", "bind")):
        if src.get(key) is not None:
            out[option] = src[key]
    if (deploy.get("output") or {}).get("report_udp_max_bytes") is not None:
        out["report_udp_max_bytes"] = int(deploy["output"]["report_udp_max_bytes"])
    nav = deploy.get("nav") or {}
    water = nav.get("water_depth") or {}
    if water.get("source") == "altitude":
        out["nav_water_depth_from_altitude"] = 1
        out.update({f"nav_{k}": v for k, v in dict(WATER_DEPTH_DEFAULTS, **water).items() if k != "source"})
    for key in ("keep_s", "surface_z", "profile_step_m", "history_margin_s"):
        if nav.get(key) is not None:
            out[f"nav_{key}"] = nav[key]
    return {k: _option_text(k, v) for k, v in out.items()}


NAV_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "sound_speed", "water_depth", "altitude", "bottom_lock",
              "roll_sigma", "pitch_sigma", "yaw_sigma", "position_sigma", "velocity_sigma", "depth_sigma", "altitude_sigma",
              "water_depth_sigma", "sound_speed_sigma")
_F = {k: i for i, k in enumerate(NAV_FIELDS)}
WATER_DEPTH_DEFAULTS = {            # nav.water_depth in a deployment file
    "source": "datagram",           # "altitude": vehicle depth + the Doppler log's altitude above the bottom
    "dvl_offset_m": 0.0,            # the log's transducer below the navigation reference point, m
    "altitude_sigma_m": 0.0,        # used when a fix carries no "altitude_sigma" (datasheet; unconfirmed)
    "depth_sigma_m": 0.0,           # used when a fix carries no "depth_sigma" (datasheet; unconfirmed)
    "hold_s": 60.0,                 # without bottom lock the last value is held this long, then the depth is unknown
    "hold_sigma_m_per_s": 0.0,      # its sigma grows by this per second held
    "hold_sigma_m_per_m": 0.05,     # and by this per metre travelled (a bottom slope not ruled out; an assumption)
    "max_altitude_m": 0.0,          # an altitude above this is not believed (0: no limit)
}


class NavListener:
    """The vehicle's navigation, received as JSON datagrams on a UDP port, one fix each:
        {"time_ns": ..., "x": north m, "y": east m, "z": down m, "roll": rad, "pitch": rad, "yaw": rad,
         "vx": ..., "vy": ..., "vz": ...}                                   (velocity over ground, m/s)
    and, where the vehicle has them (an inertial unit with a Doppler log does; all optional):
        "sound_speed" (m/s at the vehicle), "altitude" (m above the bottom), "bottom_lock" (0 | 1; absent:
        locked when the altitude is a positive number), "water_depth" (m, when the vehicle works it out itself),
        and one-sigma uncertainties "roll_sigma", "pitch_sigma", "yaw_sigma" (rad), "position_sigma" (m),
        "velocity_sigma" (m/s), "depth_sigma", "altitude_sigma", "water_depth_sigma" (m), "sound_speed_sigma" (m/s).
    tools/nav_adapter.py turns a vendor's stream into these. `at(time_ns)` gives the pose at a ping's
    transmit time, interpolated between the two fixes around it (held from the nearest one outside
    them). A fixed pose can be given instead for a bench or a moored array.

    `water_depth` (WATER_DEPTH_DEFAULTS): with source "altitude" the water depth is the vehicle's depth
    plus the log's altitude, with its uncertainty, held with a growing uncertainty when bottom lock is
    lost. Fixes are kept for `keep_s` seconds: a ping is processed after its receive window has closed,
    and the attitude history of that window must still be there. The C++ counterpart is libs/live
    `Navigation`."""

    def __init__(self, port: int | None = None, host: str = "0.0.0.0", static: dict | None = None, keep: int | None = None,
                 water_depth: dict | None = None, keep_s: float = 120.0, surface_z: float = 0.0, profile_step_m: float = 0.5,
                 history_margin_s: float = 2.0):
        import collections
        import threading
        self._fixes = collections.deque(maxlen=keep or 200000)
        self._samples = collections.deque(maxlen=2000)       # (time_ns, x, y, z, sound speed) for the profile, since the last ping
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.socket = None
        self.keep_ns = int(keep_s * 1e9)
        self.surface_z = float(surface_z)
        self.profile_step_m = float(profile_step_m)
        self.history_margin_ns = int(history_margin_s * 1e9)   # a ping's attitude history reaches this far either side of its window
        self._water = None
        if water_depth and water_depth.get("source", "datagram") == "altitude":
            unknown = set(water_depth) - set(WATER_DEPTH_DEFAULTS)
            if unknown:
                raise ValueError(f"nav.water_depth: unknown key(s) {sorted(unknown)}")
            w = dict(WATER_DEPTH_DEFAULTS, **water_depth)
            w.pop("source")
            self._water = core.WaterDepthTracker(dict(w, from_altitude=True))
        if static is not None:
            self.push(dict(static, time_ns=0))
        if port is not None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((host, port))
            self.socket.settimeout(0.2)
            self.port = self.socket.getsockname()[1]
            threading.Thread(target=self._listen, daemon=True).start()

    @classmethod
    def from_config(cls, nav: dict) -> "NavListener":
        """From the `nav` section of a deployment file."""
        keys = ("port", "host", "static", "water_depth", "keep_s", "surface_z", "profile_step_m", "history_margin_s")
        return cls(**{k: nav[k] for k in keys if nav.get(k) is not None})

    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                self.push(json.loads(self.socket.recv(65535)))     # any datagram, whole
            except socket.timeout:
                continue
            except (OSError, ValueError, KeyError, TypeError):
                continue                       # a bad datagram is dropped, not fatal

    def push(self, fix: dict) -> None:
        def number(key, default):
            v = fix.get(key)
            return default if v is None else float(v)
        row = np.array([number(k, 0.0 if i < 9 else float("nan")) for i, k in enumerate(NAV_FIELDS)] + [float("nan")] * 4)
        time_ns = int(fix["time_ns"])
        if self._water is not None:
            lock = True if np.isnan(row[_F["bottom_lock"]]) else bool(row[_F["bottom_lock"]])
            row[len(NAV_FIELDS):] = self._water.update(time_ns, row[0], row[1], row[2] - self.surface_z, row[_F["altitude"]], lock,
                                                       row[_F["altitude_sigma"]], row[_F["depth_sigma"]])
        with self._lock:
            self._fixes.append((time_ns, row))
            while len(self._fixes) > 1 and self._fixes[0][0] < time_ns - self.keep_ns:
                self._fixes.popleft()
            c = row[_F["sound_speed"]]
            if np.isfinite(c) and c > 0.0 and (not self._samples or abs(row[2] - self._samples[-1][3]) >= self.profile_step_m):
                self._samples.append((time_ns, row[0], row[1], row[2], c))

    def at(self, time_ns: int):
        """(position [3], rpy [3], velocity [3]) at `time_ns`, or None before any fix has arrived."""
        with self._lock:
            fixes = list(self._fixes)
        if not fixes:
            return None
        times = np.array([f[0] for f in fixes], dtype=np.int64)
        i = int(np.searchsorted(times, time_ns))
        if i == 0 or i == len(fixes) or len(fixes) == 1:
            row = fixes[min(i, len(fixes) - 1)][1]
        else:
            w = (time_ns - times[i - 1]) / max(times[i] - times[i - 1], 1)
            row = (1.0 - w) * fixes[i - 1][1] + w * fixes[i][1]
            # Angles go the short way round.
            row[3:6] = fixes[i - 1][1][3:6] + w * np.angle(np.exp(1j * (fixes[i][1][3:6] - fixes[i - 1][1][3:6])))
        return row[0:3], row[3:6], row[6:9]

    def _nearest(self, time_ns: int):
        with self._lock:
            fixes = list(self._fixes)
        return min(fixes, key=lambda f: abs(f[0] - time_ns))[1] if fixes else None

    def extras(self, time_ns: int):
        """(sound speed, water depth) of the fix nearest `time_ns`; NaN where navigation does not send one
        (the water depth is the one sent, else the one from altitude + depth when that is configured)."""
        row = self._nearest(time_ns)
        if row is None:
            return float("nan"), float("nan")
        water = row[_F["water_depth"]]
        if not (np.isfinite(water) and water > 0.0):
            water = row[len(NAV_FIELDS)]
        return float(row[_F["sound_speed"]]), float(water)

    def facts(self, time_ns: int) -> dict:
        """What goes into a ping's record under metadata.extra_json "nav": the uncertainties the nearest fix
        carried, whether the sound speed is a measurement, how the water depth was got, and the
        (seconds after the transmit, x, y, z, sound speed) samples taken since the last ping (they are handed
        out once)."""
        row = self._nearest(time_ns)
        out = {}
        if row is None:
            return out
        if np.isfinite(row[_F["sound_speed"]]) and row[_F["sound_speed"]] > 0.0:
            out["sound_speed_measured"] = 1
        water = row[_F["water_depth"]]
        if np.isfinite(water) and water > 0.0:
            if np.isfinite(row[_F["water_depth_sigma"]]):
                out["water_depth_sigma_m"] = float(row[_F["water_depth_sigma"]])
        elif np.isfinite(row[len(NAV_FIELDS)]):
            out["water_depth_sigma_m"] = float(row[len(NAV_FIELDS) + 1])
            out["water_depth_held_s"] = float(row[len(NAV_FIELDS) + 3]) if row[len(NAV_FIELDS) + 2] else 0.0
        for key in ("roll_sigma", "pitch_sigma", "yaw_sigma", "position_sigma", "velocity_sigma", "depth_sigma", "altitude_sigma",
                    "sound_speed_sigma"):
            if np.isfinite(row[_F[key]]):
                out[key] = float(row[_F[key]])
        with self._lock:
            samples = list(self._samples)
            self._samples.clear()
        if samples:
            out["sound_speed_samples"] = [[round((t - time_ns) * 1e-9, 6), round(x, 3), round(y, 3), round(z, 3), round(c, 4)]
                                          for t, x, y, z, c in samples]
        return out

    def history(self, first_ns: int, last_ns: int, margin_ns: int | None = None):
        """The fixes from `margin_ns` (default: `history_margin_s`) before `first_ns` to as long after `last_ns`: [(time_ns, row)]."""
        margin_ns = self.history_margin_ns if margin_ns is None else margin_ns
        with self._lock:
            return [f for f in self._fixes if first_ns - margin_ns <= f[0] <= last_ns + margin_ns]

    def close(self) -> None:
        self._stop.set()
        if self.socket:
            self.socket.close()


def apply_pose(record, position, rpy, velocity=(0.0, 0.0, 0.0)) -> None:
    """Put the vehicle's pose into a record (both the projector's and the array's)."""
    g = record.geometry
    for name in ("tx_pose", "rx_pose"):
        pose = getattr(g, name)
        pose.position, pose.rpy, pose.velocity = (tuple(float(v) for v in a) for a in (position, rpy, velocity))
        setattr(g, name, pose)
    record.geometry = g


def apply_history(record, fixes, surface_z: float = 0.0) -> None:
    """Put navigation fixes [(time_ns, row)] around a ping into the record as its attitude history
    (navigation.ins, times in seconds after the transmit): what `elevation.attitude_time: arrival`
    reads, and what a replay of the recording then has too. Fixes that carry them also leave the
    vehicle's depth (navigation.depth) and the Doppler log's altitude and velocity over ground in the
    vehicle's axes (navigation.dvl; altitude NaN without bottom lock)."""
    out, depth, dvl = [], [], []
    for time_ns, row in fixes:
        a = core.InsSample()
        a.t = (time_ns - record.transmit.time_ns) * 1e-9
        a.position, a.rpy, a.velocity = (tuple(float(v) for v in row[k:k + 3]) for k in (0, 3, 6))
        out.append(a)
        if len(row) <= 11:
            continue
        d = core.DepthSample()
        d.t, d.depth = a.t, float(row[2] - surface_z)
        depth.append(d)
        altitude, lock = row[_F["altitude"]], row[_F["bottom_lock"]]
        if np.isfinite(altitude) or not np.isnan(lock):
            q = core.DvlSample()
            q.t = a.t
            q.velocity_body = tuple(float(v) for v in np.asarray(core.rotation_ned(*a.rpy)).T @ row[6:9])
            q.altitude = float(altitude) if (np.isnan(lock) or lock != 0.0) and np.isfinite(altitude) and altitude > 0.0 else float("nan")
            dvl.append(q)
    nav = record.navigation
    nav.ins = out
    if depth:
        nav.depth = depth
    if dvl:
        nav.dvl = dvl
    record.navigation = nav


def apply_nav_facts(record, facts: dict) -> None:
    """What navigation said about its own numbers for this ping, into the record (metadata.extra_json "nav")."""
    if not facts:
        return
    m = record.metadata
    extra = json.loads(m.extra_json or "{}")
    extra["nav"] = facts
    m.extra_json = json.dumps(extra)
    record.metadata = m


def nav_facts(record) -> dict:
    try:
        return json.loads(record.metadata.extra_json or "{}").get("nav") or {}
    except ValueError:
        return {}


def decimated_sync(record) -> dict | None:
    """The clock facts a record carries when the decimate-first stage has synchronised its boards."""
    try:
        d = json.loads(record.metadata.extra_json or "{}").get("decimated") or {}
    except ValueError:
        return None
    boards = record.metadata.boards
    if "sync_residual_us" not in d or not boards or not all(b.synchronised for b in boards):
        return None
    out = {"ok": bool(d["sync_ok"]), "worst_residual_us": float(d["sync_residual_us"]),
           "offsets_us": [float(b.clock_offset * 1e6) for b in boards], "drift_ppm": [float(b.clock_drift_ppm) for b in boards]}
    flagged = _flag_list((d.get("sync_flagged") or [[]])[0])
    if flagged:
        out["flagged"] = flagged
    return out


def blanking_options(blanking) -> dict | None:
    """`chain.blanking` checked: {recovery_s, taper_s, mode}."""
    if blanking is None or blanking is False:
        return None
    b = dict({"recovery_s": 0.0, "taper_s": 0.0, "mode": "zero"}, **({} if blanking is True else blanking))
    if set(b) - {"recovery_s", "taper_s", "mode"} or b["mode"] not in ("zero", "ignore"):
        raise ValueError("blanking: {recovery_s, taper_s, mode: zero | ignore}")
    return {"recovery_s": float(b["recovery_s"]), "taper_s": float(b["taper_s"]), "mode": b["mode"]}


def _flag_list(flagged) -> list:
    """Boards timed from one reference burst (their number) or not at all (-number - 1), as the report lists them."""
    return [int(b) if f == 1 else -int(b) - 1 for b, f in enumerate(flagged) if f]


DECIMATE_KEYS = ("f_low", "f_high", "band_margin", "output_rate", "oversample", "ripple_db", "stopband_db", "taps")


def decimate_options(decimate) -> dict | None:
    """`decimate` of a deployment file ({} or True: every default) as the options of the stage (libs/live decimate.hpp)."""
    if decimate is None or decimate is False:
        return None
    d = {} if decimate is True else dict(decimate)
    if d.pop("enabled", True) is False:
        return None
    d.pop("reference", None)
    unknown = set(d) - set(DECIMATE_KEYS) - {"blank_until_s", "blank_taper_s"}
    if unknown:
        raise ValueError(f"decimate: unknown key(s) {sorted(unknown)}")
    return {k: v for k, v in d.items() if v is not None}


def apply_environment(record, sound_speed=float("nan"), water_depth=float("nan")) -> None:
    """The vehicle's own sound speed and the water depth it reports, into the record."""
    e = record.environment
    if np.isfinite(sound_speed) and sound_speed > 0.0:
        e.sound_speed = float(sound_speed)
    if np.isfinite(water_depth) and water_depth > 0.0:
        e.water_depth = float(water_depth)
    record.environment = e


class LiveChain:
    """Process ping records as they arrive. `process(record)` returns the report for that ping."""

    def __init__(self, config: dict | None = None, *, recording: str | None = None, report_path: str | None = None,
                 report_udp: tuple | None = None, taps: dict | None = None, nav: NavListener | None = None,
                 report_udp_max_bytes: int = REPORT_UDP_MAX_BYTES):
        self.cfg = copy.deepcopy(_merge(DEFAULTS, config))
        self.resolved = None                   # what the first ping resolved of the settings left to be derived (derive.hpp)
        self.report_udp_max_bytes = int(report_udp_max_bytes)
        self.reports_split = 0                 # reports that went out as several datagrams
        self.sbl_fallbacks = 0                 # pings whose SBL ran on the host reference (more channels than its kernels take)
        t = self.cfg["tracker"]
        self._elevation = None
        self._ladder = None                    # the sound-speed ladder (elevation.profile), Python engine
        self._decimate = decimate_options(self.cfg.get("decimate"))
        self._blanking = blanking_options(self.cfg.get("blanking"))
        self._blank_range = None               # m: nothing nearer is reported (blanking on), this ping
        if self.cfg.get("elevation") is not None:
            from .elevation import DEFAULTS as EL
            from .elevation import normalise_ssp, profile_options
            self._elevation = _merge(EL, self.cfg["elevation"])
            self._elevation["ssp"] = normalise_ssp(self._elevation["ssp"])
            if self._elevation["profile"] is not None and self.cfg["engine"] != "cpp" and self.cfg["doa"] != "sbl":
                self._ladder = core.SoundSpeedLadder(profile_options(self._elevation["profile"]))
        self.tracker = None                    # built at the first ping, once its sigmas are known (Python engine)
        self._multipath = None                 # hook: depth from multipath timing, Python engine (the C++ engine runs it itself)
        if self.cfg.get("multipath_depth") is not None and self.cfg["engine"] != "cpp" and self.cfg["doa"] != "sbl":
            from .multipath_depth import ChainStage
            self._multipath = ChainStage(self.cfg["multipath_depth"])
        self._coverage = None                  # hook: the coverage picture, Python engine (the C++ engine runs it itself)
        self._coverage_table = self.cfg.get("coverage") is None     # the detection-probability table is made at the first ping
        if self.cfg.get("coverage") is not None and self.cfg["engine"] != "cpp" and self.cfg["doa"] != "sbl":
            from .coverage import chain_options as coverage_options
            self._coverage = core.Coverage(dict(coverage_options(self.cfg["coverage"]), surface_z=float((self.cfg.get("elevation") or {}).get("surface_z", 0.0))))
        if self._elevation is not None and self._elevation["ray_method"] not in ("steps", "arcs"):
            raise ValueError(f"elevation.ray_method: {self._elevation['ray_method']!r} (steps or arcs)")
        b = self.cfg["bearings_deg"]
        self.bearings = np.radians(np.arange(b[0], b[1] + 0.5 * b[2], b[2]))
        self._writer = core.RecordWriter(recording) if recording else None
        self._report = open(report_path, "w", encoding="utf-8") if report_path else None
        self._udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), tuple(report_udp)) if report_udp else None
        self._tap_cfg = taps or {}
        self.nav = nav                         # None: the records already carry the pose (simulation, replay)
        self.pings_without_nav = 0
        self._reference = sync_reference(self.cfg)
        self.boards_out_of_sync = 0
        check_choices(self.cfg)
        want = self.cfg["backend"]
        self.backend = core.Backend.Cuda if want == "cuda" or (want == "auto" and core.cuda_available()) else core.Backend.Host
        if self.cfg["cell_statistic"] not in CELL_STATISTICS:
            raise ValueError(f"cell_statistic: {self.cfg['cell_statistic']!r} (mean, peak or multiscale)")
        self.detector = None                   # the multiscale detector's plan for the last map (libs/live detector.hpp)
        self._plan_key = None
        if self.cfg["doa"] not in ("map", "sbl"):
            raise ValueError(f"doa: {self.cfg['doa']!r} (map or sbl)")
        if self.cfg["engine"] not in ("python", "cpp"):
            raise ValueError(f"engine: {self.cfg['engine']!r} (python or cpp)")
        self._cpp = None
        if self.cfg["engine"] == "cpp" or self.cfg["doa"] == "sbl":
            # One C++ object does synchronisation, conditioning, beamforming, detection, bearings,
            # tracking and classification; this class keeps the navigation, recording, taps and report.
            self._cpp = core.LiveChainCpp(dict(cpp_chain_options(self.cfg, self._reference), backend=self.backend))
        self._origin = None                    # world position of the array at the first ping
        self._t0 = None
        self.pings = 0
        self.last = {}                         # the most recent beam map and detections, for display

    # -- one ping ----------------------------------------------------------------------------

    def process(self, record) -> dict:
        t_wall = time.perf_counter()
        index = self.pings
        self.pings += 1
        tap = None
        if self._tap_cfg.get("dir") and index < int(self._tap_cfg.get("pings", 1)):
            tap = TapWriter(f"{self._tap_cfg['dir']}/ping{index:04d}")
        if self.nav is not None:
            fix = self.nav.at(record.transmit.time_ns)
            if fix is None:
                self.pings_without_nav += 1    # processed where the record says it is; counted and reported
            else:
                apply_pose(record, *fix)
                apply_environment(record, *self.nav.extras(record.transmit.time_ns))
                apply_nav_facts(record, self.nav.facts(record.transmit.time_ns))
                s_ = record.samples
                window_ns = int((s_.start_time + s_.n_samples / s_.sample_rate) * 1e9)
                history = self.nav.history(record.transmit.time_ns, record.transmit.time_ns + window_ns)
                if len(history) > 1:           # a fixed pose has one fix and no history
                    apply_history(record, history, self.nav.surface_z)
        if self.cfg.get("sound_speed"):
            apply_environment(record, sound_speed=float(self.cfg["sound_speed"]))
        if self._writer:
            self._writer.write(record)         # raw samples, with the pose, before anything else touches it
            self._writer.flush()
        if tap:
            tap.records("raw", [record])
        cfg = self.cfg
        if not self._coverage_table:
            # The probability that this chain reports an echo against its link-budget SNR, from the detector's own plan
            # (detection_probability): made once, for the coverage picture of either engine.
            from .coverage import pd_table
            table = pd_table(record, {k: v for k, v in cfg.items() if k != "engine"})
            (self._cpp.set_coverage_pd if self._cpp is not None else self._coverage.set_pd_table)(*table)
            self._coverage_table = True
        if self._cpp is not None:
            report = json.loads(self._cpp.process(record))
            if self.resolved is None:
                self.resolved = json.loads(self._cpp.resolved())
            if cfg["cell_statistic"] == "multiscale":
                self.detector = json.loads(self._cpp.detector())
            self.sbl_fallbacks += bool(report.get("sbl_fallback"))
            report.update(ping=index, pings_without_nav=self.pings_without_nav,
                          processing_seconds=time.perf_counter() - t_wall)
            if report.get("sync") and not report["sync"]["ok"]:
                self.boards_out_of_sync += 1
            power, ranges, _ = self._cpp.last_map()
            if tap:
                tap.records("prepare", [record])               # conditioned in place by the C++ chain
                tap.arrays("beamform", power=power, ranges=ranges, bearings=self.bearings)
                tap.json("report", report)
            self.last = {"power": power, "ranges": ranges, "hits": report["detections"], "classes": report["tracks"]}
            return self._emit(report)
        sync_facts = None
        bl = self._blanking
        blank_until = record.transmit.waveform.duration + bl["recovery_s"] if bl else 0.0
        if self._decimate is not None and not record.samples.is_complex():
            # Decimate first: band, baseband and board clocks in one early stage (the recording above holds the raw ping).
            # Blanking is done inside it: after the reference is read, before the filter.
            blank = {"blank_until_s": blank_until, "blank_taper_s": bl["taper_s"]} if bl and bl["mode"] == "zero" else {}
            core.live_decimate_record(record, dict(self._decimate, **blank), self._reference)
        if self._reference is not None and decimated_sync(record) is not None:
            sync_facts = decimated_sync(record)                 # the decimating stage has synchronised the boards already
            if not sync_facts["ok"]:
                self.boards_out_of_sync += 1
            if tap:
                tap.records("sync", [record])
        elif self._reference is not None:
            # Boards run on their own clocks: line them up on the shared reference and drop its inputs.
            from . import sync
            rep_ = sync.synchronise(record, self._reference, enforce=False)
            sync_facts = {"ok": rep_.ok, "worst_residual_us": float(rep_.residual.max() * 1e6),
                          "offsets_us": [float(v * 1e6) for v in rep_.estimate.offset],
                          "drift_ppm": [float(v) for v in rep_.estimate.drift_ppm]}
            if rep_.estimate.flagged is not None and rep_.estimate.flagged.any():
                sync_facts["flagged"] = _flag_list(rep_.estimate.flagged)
            if not rep_.ok:
                self.boards_out_of_sync += 1   # reported; this ping's detections are still passed on, flagged
            if self._reference.get("dedicated"):
                record = sync.drop_reference_channels(record)   # those inputs carry no hydrophone
            if tap:
                tap.records("sync", [record])
        self._blank_range = None
        if bl:
            # A record the decimating stage blanked says so; its filter spreads the edge by its own delay.
            try:
                note = json.loads(record.metadata.extra_json or "{}").get("decimated") or {}
            except ValueError:
                note = {}
            if bl["mode"] == "zero" and not note.get("blanked_s"):
                core.live_blank_interval(record, -1e300, blank_until, bl["taper_s"])
            ring = float(note.get("delay_s", 0.0)) if record.samples.is_complex() else 0.0
            self._blank_range = 0.5 * record.environment.sound_speed * (blank_until + bl["taper_s"] + ring)
        if cfg["dead_channels"]:
            from . import sync
            dead = {int(v) for v in cfg["dead_channels"]}
            record = sync.select_channels(record, [r for r in range(record.samples.n_channels) if r not in dead])
        if self.resolved is None:
            self._resolve(record)
        from . import recon
        recon.prepare([record], backend=self.backend)        # baseband and pulse compression (cuFFT on a GPU)
        if tap:
            tap.records("prepare", [record])
        rows = subs = None
        if self._elevation is not None:
            from . import elevation
            rows = elevation.find_rows(record, self._elevation["row_tolerance_m"], int(self._elevation["min_row_elements"]))
            if len(rows) < 2:
                rows = None                    # one row: exactly the chain without the option
            else:
                subs = elevation.row_records(record, rows)
        if rows is None:
            fine, fine_ranges, k = self._fine_map(record)
        else:
            # Each row beamformed on its own, the rows' power added: a row's bearing response
            # does not depend on the elevation the arrival comes from.
            fine = None
            for sub in subs:
                f_, fine_ranges, k = self._fine_map(sub)
                fine = f_ if fine is None else fine + f_
        self._rows, self._subs = rows, subs
        return self._after_map(record, fine, fine_ranges, k, index, tap, sync_facts, t_wall)

    def _resolve(self, record) -> None:
        """Every size the configuration left out, from this sonar's band, pulse and array: once, at the first ping
        (libs/live derive.hpp, the same function the C++ chain calls)."""
        cfg = self.cfg
        given = _given(cfg)
        given["bearing_step_deg"] = float(cfg["bearings_deg"][2])
        if self._elevation is not None:
            given["min_row_elements"] = int(self._elevation["min_row_elements"])
            if self._elevation["row_tolerance_m"] is not None:
                given["row_tolerance_m"] = float(self._elevation["row_tolerance_m"])
        d = core.live_derive(record, given)
        for path, name, factor in DERIVED:
            node = cfg
            for key in path[:-1]:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, dict) and node.get(path[-1]) is None:
                node[path[-1]] = float(np.degrees(d[name])) if factor != 1.0 else float(d[name])
        if self._elevation is not None:
            self._elevation["row_tolerance_m"] = float(d["row_tolerance_m"])
        t = cfg["tracker"]
        self._row_tolerance = float(d["row_tolerance_m"])
        self._bearing_sigma = float(d["tracker_bearing_sigma"])      # rad, as derived: no trip through degrees
        self._multipath_bearing = float(d["multipath_bearing"])
        self.tracker = detect.Tracker(range_sigma=t["range_sigma"], bearing_sigma=self._bearing_sigma,
                                      confirm_hits=t["confirm_hits"], max_misses=t["max_misses"],
                                      initial_speed_sigma=t["initial_speed_sigma"], process_noise=t["process_noise"],
                                      depth_process_noise=(self._elevation or {}).get("depth_process_noise", 0.5),
                                      association=t["association"])
        self.resolved = json.loads(d["json"])

    def _fine_map(self, record):
        """Power over bearing x fine range for one line of hydrophones, the fine ranges and
        the number of fine samples per range cell."""
        cfg = self.cfg
        if cfg["beamformer"] == "backprojection":
            # Delay and sum by the backprojection kernel onto range x bearing: exact in the near
            # field too, and the same code on the host and on a GPU.
            c = record.environment.sound_speed
            w = record.transmit.waveform
            band = abs(w.f_stop - w.f_start) or 1.0 / w.duration
            s_ = record.samples
            r0 = 0.5 * c * s_.start_time
            r1 = 0.5 * c * (s_.start_time + (s_.n_samples - 1) / s_.sample_rate - w.duration)
            k = max(int(np.ceil(cfg["range_cell_m"] / (0.75 * c / (2.0 * band)))), 1)   # samples of the image per cell
            step = cfg["range_cell_m"] / k
            fine_r = r0 + step * np.arange(int((r1 - r0) / step) // k * k)
            fan = flank.backprojection_map(record, fine_r, self.bearings, window=cfg["window"], interp="linear",
                                           backend=self.backend)
            return (np.abs(fan.image) ** 2).T, fine_r, k      # [bearing, range]
        else:
            bm = (flank.mvdr_beamform(record, self.bearings, backend=self.backend) if cfg["beamformer"] == "mvdr"
                  else flank.farfield_beamform(record, self.bearings, window=cfg["window"], backend=self.backend)).valid()
            fine, fine_ranges = np.abs(bm.data) ** 2, bm.ranges
            k = max(int(round(cfg["range_cell_m"] / (fine_ranges[1] - fine_ranges[0]))), 1)
            return fine, fine_ranges, k

    def _after_map(self, record, fine, fine_ranges, k, index, tap, sync_facts, t_wall):
        cfg = self.cfg
        rows, subs = self._rows, self._subs
        n = fine.shape[1] // k * k
        peak = cfg["cell_statistic"] == "peak"
        cells = fine[:, :n].reshape(len(self.bearings), -1, k)
        power = cells.max(axis=2) if peak else cells.mean(axis=2)
        looks = float(np.sum(1.0 / np.arange(1, k + 1))) if peak else 1.0     # mean of the largest of k noise samples over their mean
        ranges = fine_ranges[:n].reshape(-1, k).mean(axis=1)
        multi = cfg["cell_statistic"] == "multiscale"
        fine, fine_ranges = fine[:, :n], np.asarray(fine_ranges)[:n]
        if self._blank_range is not None:
            # The map starts where whole echoes do: the detector neither reports nearer nor trains on the blanked cells.
            skip = min(int(np.count_nonzero(ranges < self._blank_range)), max(len(ranges) - 3, 0))
            power, ranges = power[:, skip:], ranges[skip:]
            fine, fine_ranges = fine[:, skip * k:], fine_ranges[skip * k:]
        cell = ranges[1] - ranges[0]
        # The window in cells of this map, by the library's one conversion (libs/detect cfar_window: rounded up, the
        # guard at least one range resolution): the C++ chain calls the same function.
        w_ = record.transmit.waveform
        resolution = record.environment.sound_speed / (2.0 * (abs(w_.f_stop - w_.f_start) or 1.0 / w_.duration))
        guard, train = core.cfar_window(float(cell), float(resolution), guard_m=float(cfg["cfar"]["guard_m"]), train_m=float(cfg["cfar"]["train_m"]))
        background_map = None
        if multi:
            # Each cell on its strongest sample, its mean and the means of several cells, against the thresholds of those
            # statistics; sidelobes of a stronger cell dropped; touching cells one detection (libs/live detector.hpp).
            n_rows = len(rows) if rows is not None else 1
            key = (power.shape[1], k, train, n_rows)
            if self._plan_key != key:
                o = detector_options(cfg)
                o.setdefault("merge_m", float(cfg["tracker"]["range_sigma"]))
                o["row_tolerance_m"] = self._row_tolerance
                if self._elevation is not None:
                    o["min_row_elements"] = int(self._elevation["min_row_elements"])
                self._plan = core.live_detector_plan(record, power.shape[1], k, float(cell) / k, [float(v) for v in self.bearings], n_rows,
                                                     int(guard), int(train), cfg["window"] == "hann" and cfg["beamformer"] != "mvdr", o)
                self._plan_key, self.detector = key, json.loads(self._plan["json"])
            hits, background_map = detect.multiscale(fine, power, fine_ranges, ranges, self.bearings, guard, train, self._plan)
        else:
            hit_map = detect.cfar(power, pfa=cfg["cfar"]["pfa"], guard=guard, train=train, backend=self.backend)
            hits = detect.cluster(hit_map, power, ranges, self.bearings, guard=guard, train=train)
        hits = [h for h in hits if h.snr_db > cfg["min_snr_db"] and (self._blank_range is None or h.range >= self._blank_range)]
        if self._reference is not None and not self._reference.get("dedicated"):
            # A reference summed into hydrophone inputs leaves its two bursts in the data, one just
            # after the window opens and one just before it closes: no detections are taken there.
            c = record.environment.sound_speed
            s_ = record.samples
            t_open, t_close = s_.start_time, s_.start_time + s_.n_samples / s_.sample_rate
            pulse, burst = record.transmit.waveform.duration, float(self._reference["duration"])
            lo = 0.5 * c * (t_open + float(self._reference["lead"]) + burst)
            hi = 0.5 * c * (t_close - float(self._reference["tail"]) - pulse - burst)
            hits = [h for h in hits if lo < h.range < hi]
        if tap:
            tap.arrays("beamform", power=power.astype(np.float32), ranges=ranges, bearings=self.bearings)

        # World position of each detection: the array's own position and heading come with the record.
        rx = record.rx_positions()
        origin = rx.mean(axis=0)
        axis = rx[-1, :2] - rx[0, :2]
        el_cfg = self._elevation if rows is not None else None
        els = [None] * len(hits)
        ladder = None                          # the rung of the sound-speed ladder this ping (elevation.profile)
        limit_water = None                     # the water the multipath term of the depth sigma is worked out in (elevation.multipath_sigma)
        if el_cfg is not None:
            # What navigation said about its own numbers for this ping (absent: the configuration's).
            said = nav_facts(record)
            el_cfg = dict(el_cfg, roll_sigma_deg=max(el_cfg["roll_sigma_deg"], float(np.degrees(said.get("roll_sigma", 0.0)))))
            if not el_cfg["water_depth"]:
                el_cfg["water_depth_sigma_m"] = float(np.hypot(el_cfg["water_depth_sigma_m"], said.get("water_depth_sigma_m", 0.0)))
            # Stacked rows: the elevation of the strongest detections, from the whole array.
            from . import elevation
            frame = elevation.array_frame(record, rows, cfg["look"])
            origin, axis = frame.origin, frame.along[:2]
            mean = background_map if multi else detect._window_mean(np.asarray(power, dtype=float), guard, train)[0]
            order = sorted(range(len(hits)), key=lambda i: -hits[i].snr_db)[:int(el_cfg["max_detections"])]
            background = [float(mean[int(np.argmin(np.abs(self.bearings - hits[i].bearing))),
                                     int(np.argmin(np.abs(ranges - hits[i].range)))]) / looks for i in order]
            window = "rect" if cfg["beamformer"] == "mvdr" else cfg["window"]
            if el_cfg["sector_from_rays"] is not None or (el_cfg["multipath_sigma"] is not None and el_cfg["depth"]):
                # hook (opt-in): the elevations sound can arrive from, and the echoes that share a range cell, by rays through
                # the water held this ping (as the C++ chain)
                from .elevation import profile_options as _po
                own_ = float(origin[2] - el_cfg["surface_z"])
                water_ = el_cfg["water_depth"] or record.environment.water_depth
                water_ = float(water_) if water_ and np.isfinite(water_) and water_ > 0.0 else float("nan")
                ssp_ = el_cfg["ssp"] or {"depth": [], "speed": []}
                if self._ladder is not None and el_cfg["depth"]:
                    measured = self._ladder.absorb(record, el_cfg["surface_z"]) or bool(cfg.get("sound_speed"))
                    ladder = self._ladder.evaluate(int(record.transmit.time_ns), float(origin[0]), float(origin[1]), own_, float(record.environment.sound_speed),
                                                   bool(measured), water_, ssp_["depth"], ssp_["speed"], float(el_cfg["gradient_per_s"] or 0.0),
                                                   float(el_cfg["gradient_sigma_per_s"]))
                through = core.live_water_profile(_po(el_cfg["profile"]) if el_cfg["profile"] is not None else None, ladder, ssp_["depth"], ssp_["speed"],
                                                  el_cfg["gradient_per_s"] is not None, float(el_cfg["gradient_per_s"] or 0.0), float(el_cfg["gradient_sigma_per_s"]),
                                                  own_, float(record.environment.sound_speed), water_)
                limit_water = (through["depth"], through["speed"])
                if el_cfg["sector_from_rays"] is not None:
                    el_cfg = dict(el_cfg, _sector_profile=(through["depth"], through["speed"]))
            got = elevation.estimate(record, rows, subs, [(hits[i].range, hits[i].bearing, hits[i].first_range) for i in order], background, frame,
                                     el_cfg, noise_weight=sum(elevation.weight_power(len(r), window) for r in rows),
                                     range_cell=cfg["range_cell_m"], backend=self.backend)
            for i, e in zip(order, got):
                els[i] = e
            water = el_cfg["water_depth"] or record.environment.water_depth
            water = float(water) if water and np.isfinite(water) and water > 0.0 else None
            sound = record.environment.sound_speed
            profile = None                         # depth along a ray instead of the straight line
            own_depth = float(origin[2] - el_cfg["surface_z"])
            if ladder is not None:
                if ladder["depth"]:                # climbed already, for the rays' sector
                    profile = (ladder["depth"], ladder["speed"])
            elif self._ladder is not None and el_cfg["depth"]:
                # The best rung this ping: fed from the record, so a replay climbs the ladder the same way.
                measured = self._ladder.absorb(record, el_cfg["surface_z"]) or bool(cfg.get("sound_speed"))
                ssp = el_cfg["ssp"] or {"depth": [], "speed": []}
                ladder = self._ladder.evaluate(int(record.transmit.time_ns), float(origin[0]), float(origin[1]), own_depth, float(sound), bool(measured),
                                               water if water is not None else float("nan"), ssp["depth"], ssp["speed"],
                                               float(el_cfg["gradient_per_s"] or 0.0), float(el_cfg["gradient_sigma_per_s"]))
                if ladder["depth"]:
                    profile = (ladder["depth"], ladder["speed"])
            elif el_cfg["ssp"]:
                profile = (el_cfg["ssp"]["depth"], el_cfg["ssp"]["speed"])
            if profile is None and el_cfg["gradient_per_s"] is not None:
                profile = elevation.gradient_profile(origin[2] - el_cfg["surface_z"], sound, float(el_cfg["gradient_per_s"]), water)
        # hook: the water this ping's rays go through (libs/live raytrace.hpp), opt-in: the multipath depth with `rays`, the
        # coverage picture, and the profile ensemble. One row has no elevation, but it has a ladder all the same.
        # Mirrors the C++ chain's hook; the profile itself is assembled by the same C++ function (water_profile).
        from .elevation import DEFAULTS as EL_DEFAULTS
        ecfg = self._elevation if self._elevation is not None else EL_DEFAULTS
        want_rays = (self._multipath is not None and bool(self._multipath.cfg["rays"])) or self._coverage is not None
        ensemble_on = el_cfg is not None and el_cfg["depth"] and bool((ecfg["profile"] or {}).get("ensemble"))
        rays_water, rays_depth = None, float(origin[2] - ecfg["surface_z"])
        rays_bottom = ecfg["water_depth"] or record.environment.water_depth
        rays_bottom = float(rays_bottom) if rays_bottom and np.isfinite(rays_bottom) and rays_bottom > 0.0 else float("nan")
        if want_rays or ensemble_on:
            from .elevation import profile_options as _profile_options
            if ladder is None and self._ladder is not None and want_rays:
                measured = self._ladder.absorb(record, ecfg["surface_z"]) or bool(cfg.get("sound_speed"))
                ssp_ = ecfg["ssp"] or {"depth": [], "speed": []}
                ladder = self._ladder.evaluate(int(record.transmit.time_ns), float(origin[0]), float(origin[1]), rays_depth,
                                               float(record.environment.sound_speed), bool(measured), rays_bottom, ssp_["depth"], ssp_["speed"],
                                               float(ecfg["gradient_per_s"] or 0.0), float(ecfg["gradient_sigma_per_s"]))
            ssp_ = ecfg["ssp"] or {"depth": [], "speed": []}
            rays_water = core.live_water_profile(_profile_options(ecfg["profile"]) if ecfg["profile"] is not None else None, ladder, ssp_["depth"],
                                                 ssp_["speed"], ecfg["gradient_per_s"] is not None, float(ecfg["gradient_per_s"] or 0.0),
                                                 float(ecfg["gradient_sigma_per_s"]), rays_depth, float(record.environment.sound_speed), rays_bottom)
        arcs = el_cfg is not None and el_cfg["ray_method"] == "arcs"

        def ensemble_span(level, rng, depth_m):
            """The same arrival through every member of the profile ensemble: where the depth may lie."""
            if rays_water is None or not rays_water["ensemble"]:
                return {}
            zs = [depth_m] + [ecfg["surface_z"] + core.ray_depth_arcs(rays_depth, level, rng, rays_water["depth"], member, 0.0 if np.isnan(rays_bottom) else rays_bottom)
                              for member in rays_water["ensemble"]]
            return {"depth_ensemble_m": [float(min(zs)), float(max(zs))]}

        axis = axis / max(np.linalg.norm(axis), 1e-12)
        side = 1.0 if cfg["look"] == "starboard" else -1.0
        broadside = side * np.array([-axis[1], axis[0]])
        def bend(level, rng):
            """What the angle cannot know: how the ray bent on the way. The prior alone, or what the profile's rung claims."""
            if ladder is not None and ladder["depth"]:
                if arcs:
                    return core.live_profile_depth_sigma(own_depth, level, rng, ladder["depth"], ladder["speed"], ladder["sigma"],
                                                         float(el_cfg["gradient_sigma_per_s"]), water or 0.0, True)
                return elevation.profile_depth_sigma(own_depth, level, rng, ladder["depth"], ladder["speed"], ladder["sigma"],
                                                     el_cfg["gradient_sigma_per_s"], water)
            return elevation.refraction_sigma(rng, sound, el_cfg["gradient_sigma_per_s"])

        def multipath_term(level, rng, sigma, depth_m):
            """What echoes sharing the range cell add to a depth read at `level` with the angle's own sigma (rad): the
            term in metres and the report's members (elevation.multipath_sigma; the C++ chain's own function)."""
            if el_cfg is None or el_cfg["multipath_sigma"] is None or limit_water is None or not limit_water[0]:
                return 0.0, {}
            wf = record.transmit.waveform
            f_lo, f_hi = min(wf.f_start, wf.f_stop), max(wf.f_start, wf.f_stop)
            band = f_hi - f_lo if f_hi > f_lo else 1.0 / wf.duration
            tall = float(np.max(frame.heights) - np.min(frame.heights))
            lim = core.elevation_multipath_limit(
                limit_water[0], limit_water[1], water or 0.0, own_depth, rng, level, sigma, float(depth_m - el_cfg["surface_z"]), 0.5 * sound / band,
                sound / (0.5 * (f_lo + f_hi)) / tall if tall > 0.0 else 2.0, el_cfg["arrival"] == "first", float(el_cfg["first_arrival_db"]),
                float(el_cfg.get("ambiguity_sigma", 3.0)), int(el_cfg["multipath_sigma"].get("bounces", 2)),
                float(el_cfg["multipath_sigma"].get("max_depth_m", 500.0)))
            return float(lim["extra"]), json.loads("{" + lim["members"] + "}")

        if self._origin is None:
            self._origin, self._t0 = origin.copy(), record.transmit.time_ns
        t = (record.transmit.time_ns - self._t0) * 1e-9
        local = []
        extra = [None] * len(hits)             # what elevation adds to each detection's report
        for i, h in enumerate(hits):
            p = origin[:2] - self._origin[:2] + h.range * (np.sin(h.bearing) * axis + np.cos(h.bearing) * broadside)
            depth = depth_sigma = float("nan")
            alternatives = None
            e = els[i]
            if el_cfg is not None:
                extra[i] = {"elevation_deg": None}
            if e is not None and e.valid:
                u = np.asarray(e.world)
                level = float(np.arcsin(np.clip(-u[2], -1.0, 1.0)))       # elevation above the horizontal, roll and pitch applied
                extra[i] = {"elevation_deg": float(np.degrees(level)), "elevation_sigma_deg": float(np.degrees(e.sigma)),
                            "elevation_quality": e.quality, "elevation_ambiguous": e.ambiguous}
                if el_cfg["correct_bearing"] and not e.ambiguous:
                    # The line array gave the cone angle; with the elevation the horizontal bearing follows.
                    p = origin[:2] - self._origin[:2] + e.range * u[:2]
                    extra[i].update(cone_bearing_deg=float(np.degrees(h.bearing)),
                                    bearing_deg=float(np.degrees(np.arctan2(u[:2] @ axis, u[:2] @ broadside))))
                elif el_cfg["correct_bearing"]:
                    # An ambiguous elevation is one of several candidates, and which is picked changes from ping to ping:
                    # placing the detection by it would move it by the range times the difference between their cosines
                    # (a kilometre at 12 km) and break its track. The row's own bearing and the slant range are kept, and said.
                    extra[i]["bearing_corrected"] = False
                if e.ambiguous and e.alternatives:
                    extra[i]["elevation_alternatives_deg"] = [float(np.degrees(np.arcsin(np.clip(-w[2], -1.0, 1.0)))) for w in e.alternatives]
                if el_cfg["depth"]:
                    image = float(origin[2] + e.range * u[2])             # straight-line depth: outside the water for a bounce
                    sigma_z = float(e.range * np.cos(level) * e.sigma)    # from the angle: noise, roll, attitude timing
                    folded, path = (elevation.fold_depth(image, water, el_cfg["surface_z"]) if el_cfg["fold"]
                                    else (image, "direct"))
                    if profile is not None:
                        # Back along the ray through the profile, not the straight line.
                        folded = el_cfg["surface_z"] + elevation.ray_depth(origin[2] - el_cfg["surface_z"], level, e.range, profile[0], profile[1], water,
                                                                           method=el_cfg["ray_method"])
                    # What the angle cannot know: the gradient along the path, and the water depth a bottom bounce is folded about.
                    bent = bend(level, e.range)
                    fold_sigma = 2.0 * el_cfg["water_depth_sigma_m"] if path in ("bottom", "multiple") else 0.0
                    extra[i].update(depth_m=folded, depth_sigma_m=float(np.sqrt(sigma_z**2 + bent**2 + fold_sigma**2)),
                                    depth_refraction_sigma_m=bent, image_depth_m=image, path=path)
                    if ladder is not None:
                        extra[i]["depth_profile_source"] = ladder["rung"]
                    extra[i].update(ensemble_span(level, e.range, folded))
                    depth, depth_sigma = image, sigma_z
                    more, members = multipath_term(level, float(e.range), float(e.sigma), folded)
                    if members:
                        # What the rows cannot know: which of the echoes sharing the range cell they read. It is the same
                        # boundary ping after ping, so the track's filter is not given it to average (the track's sigma gains it).
                        extra[i].update(members)
                        extra[i]["depth_sigma_m"] = float(np.sqrt(sigma_z**2 + bent**2 + fold_sigma**2 + more**2))
                    if e.ambiguous:
                        # The track, if it has a depth already, takes the candidate that fits it.
                        alternatives = [float(origin[2] + e.range * w[2]) for w in e.alternatives]
            # The tracker's measurement is (range, bearing) about the frame origin, across then forward.
            local.append(detect.Detection(float(np.hypot(*p)), float(np.arctan2(p[1], p[0])), h.snr_db, h.cells, depth, depth_sigma,
                                          depth_alternatives=alternatives))
        tracks = self.tracker.update(t, local)
        c = cfg["classify"]
        own = origin[:2] - self._origin[:2]
        mp = dict(c["multipath"], bearing=self._multipath_bearing, origin=(own[1], own[0])) if c.get("multipath") else None
        classes = classify.classify_tracks(tracks, min_speed=c["min_speed"], static_speed=c["static_speed"],
                                           max_fit_rms=c["max_fit_rms"], multipath=mp)
        track_extra = {}
        if el_cfg is not None and el_cfg["depth"]:
            by_id = {tr.id: tr for tr in tracks}
            for c_ in classes:
                tr = by_id[c_.id]
                if not np.isfinite(tr.depth):
                    track_extra[c_.id] = {"depth_m": None}
                    continue
                slant = float(np.hypot(np.hypot(c_.position[1] - own[0], c_.position[0] - own[1]), tr.depth - origin[2]))
                bent = bend(float(np.arcsin(np.clip((origin[2] - tr.depth) / slant, -1.0, 1.0))), slant)
                folded, path = (elevation.fold_depth(tr.depth, water, el_cfg["surface_z"]) if el_cfg["fold"] else (tr.depth, "direct"))
                fold_sigma = 2.0 * el_cfg["water_depth_sigma_m"] if path in ("bottom", "multiple") else 0.0
                # The filter averages what differs from ping to ping; the bending of the ray does not.
                sigma_z = float(np.sqrt(tr.depth_var + bent**2 + fold_sigma**2))
                if profile is not None:
                    # The track's straight-line depth as an arrival angle at its range, then along the ray.
                    folded = el_cfg["surface_z"] + elevation.ray_depth(origin[2] - el_cfg["surface_z"], float(np.arcsin((origin[2] - tr.depth) / slant)),
                                                                       slant, profile[0], profile[1], water, method=el_cfg["ray_method"])
                track_extra[c_.id] = {"depth_m": folded, "depth_sigma_m": sigma_z, "path": path}
                track_extra[c_.id].update(ensemble_span(float(np.arcsin(np.clip((origin[2] - tr.depth) / slant, -1.0, 1.0))), slant, folded))
                level_ = float(np.arcsin(np.clip((origin[2] - tr.depth) / slant, -1.0, 1.0)))
                more, members = multipath_term(level_, slant, float(np.sqrt(tr.depth_var) / (slant * max(np.cos(level_), 1e-6))), folded)
                if members:
                    # The same boundary echoes ping after ping: what they add is not averaged away by the filter.
                    track_extra[c_.id].update(members)
                    track_extra[c_.id]["depth_sigma_m"] = float(np.hypot(sigma_z, more))
                if ladder is not None:
                    track_extra[c_.id]["depth_profile_source"] = ladder["rung"]
                if el_cfg["classify"] and elevation.outside_water(tr.depth, float(np.hypot(np.sqrt(tr.depth_var), bent)), water, el_cfg["surface_z"],
                                                                  el_cfg["classify_sigma"], el_cfg["water_depth_sigma_m"]):
                    c_.kind = "multipath"      # its straight-line depth is above the surface or below the bottom: a bounce
        report = {
            "ping": index, "sequence": int(record.metadata.sequence), "time_ns": int(record.transmit.time_ns),
            "ownship_xyz": [float(v) for v in origin], "pings_without_nav": self.pings_without_nav,
            "sync": sync_facts,
            "detections": [dict({"range_m": h.range, "bearing_deg": float(np.degrees(h.bearing)), "snr_db": h.snr_db},
                                **({"extent_m": h.extent, "first_range_m": h.first_range} if np.isfinite(h.extent) else {}),
                                **({"arrivals": [list(a) for a in h.arrivals]} if h.arrivals and len(h.arrivals) > 1 else {}), **(x or {}))
                           for h, x in zip(hits, extra)],
            "tracks": [{"id": c_.id, "kind": c_.kind, "parent": c_.parent,
                        # World coordinates: x north, y east, metres, in the frame the records use.
                        "x": float(c_.position[1] + self._origin[0]), "y": float(c_.position[0] + self._origin[1]),
                        "vx": c_.velocity[1], "vy": c_.velocity[0], "speed_m_s": c_.speed,
                        "range_m": float(np.hypot(c_.position[1] - own[0], c_.position[0] - own[1])),
                        "mean_snr_db": c_.mean_snr_db, "seen": c_.seen, **track_extra.get(c_.id, {})} for c_ in classes],
            "processing_seconds": time.perf_counter() - t_wall,
        }
        if self._blank_range is not None:
            report["blanking"] = {"min_range_m": float(self._blank_range), "mode": self._blanking["mode"]}
        if el_cfg is not None:
            report["rows"] = len(rows)
        if ladder is not None:
            report["profile"] = {"rung": ladder["rung"]}
            if ladder["bins"] > 0:
                report["profile"].update(bins=int(ladder["bins"]), span_m=[ladder["span_lo"], ladder["span_hi"]])
        if self._multipath is not None:        # hook: adds report members only
            self._multipath.annotate(record, report, origin, hits, local, self._origin, rows=rows,
                                     surface_z=el_cfg["surface_z"] if el_cfg is not None else 0.0,
                                     water=water if el_cfg is not None else None, backend=self.backend, profile=rays_water)
        if self._coverage is not None:         # hook: one member of the report line, refreshed when something changed
            report.update(json.loads("{" + self._coverage.update(record, tuple(float(v) for v in origin), rays_bottom, rays_water) + "}"))
        report = finite_or_null(report)           # null where libs/live writes null: a non-finite snr_db, for one
        if tap:
            tap.json("report", report)
        self.last = {"power": power, "ranges": ranges, "hits": hits, "classes": classes}
        return self._emit(report)

    def _emit(self, report: dict) -> dict:
        report = finite_or_null(report)
        line = json.dumps(report, allow_nan=False)
        if self._report:
            self._report.write(line + "\n")
            self._report.flush()
        if self._udp:
            # A line too long for a datagram goes out as several complete lines ("part": [k, n]); it is never cut.
            sock, addr = self._udp
            datagrams = [line] if len(line) <= self.report_udp_max_bytes else split_report(report, self.report_udp_max_bytes)
            self.reports_split += len(datagrams) > 1
            for text in datagrams:
                try:
                    sock.sendto(text.encode(), addr)
                except OSError:
                    pass                       # nobody listening is not a reason to stop
        return report

    def run(self, source, max_pings: int | None = None) -> list[dict]:
        """Process every record a source yields."""
        reports = []
        for record in source:
            reports.append(self.process(record))
            if max_pings is not None and len(reports) >= max_pings:
                break
        return reports

    def close(self) -> None:
        if self._writer:
            self._writer.close()
        if self._report:
            self._report.close()
        if self._udp:
            self._udp[0].close()
        if self.nav is not None:
            self.nav.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
