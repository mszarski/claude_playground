"""Taps: capture what flows between stages, look at it, and feed it back in.

Every stage of the suite passes ping records (docs/RECORD_FORMAT.md). A tap writes the
records as they stand after a stage to an ordinary container file, so a tap file is also a
source: any later stage can be run from it, on this machine or another, with simulated or
real data alike. Products that are not records (beam maps, images, detections, tracks) are
written beside them as .npz or .json.

    tap = TapWriter("taps/")                  # or pipeline `output: {taps: {dir: taps/}}`
    tap.records("prepare", records)           # -> taps/01_prepare.h5
    records = read_tap("taps/", "prepare")    # the same records back
    export_wav(records[0], "ping0.wav")       # for any audio or signal tool

The simulator has its own stages before a record is complete (echoes, reverberation,
noise, hardware gain, ADC); `sim_capture(dir)` saves the samples after each of them.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct

import numpy as np

from . import _gpusonar as core


class TapWriter:
    """Writes numbered tap files into a directory and a manifest, taps.json."""

    def __init__(self, directory, stages="all"):
        self.directory = str(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.stages = None if stages in (None, "all") else set(stages)
        self.manifest = []
        self._index = 0

    def wants(self, stage: str) -> bool:
        return self.stages is None or stage in self.stages

    def _path(self, stage: str, ext: str, new: bool) -> str:
        if new:
            self._index += 1
        return os.path.join(self.directory, f"{self._index:02d}_{stage}.{ext}")

    def records(self, stage: str, records) -> str | None:
        """Snapshot of the ping records after `stage`: a container file."""
        if not self.wants(stage):
            return None
        path = self._path(stage, "h5", new=True)
        writer = core.RecordWriter(path)
        for r in records:
            writer.write(r)
        writer.close()
        s = records[0].samples
        self.manifest.append({"stage": stage, "records": os.path.basename(path), "pings": len(records),
                              "channels": s.n_channels, "samples": s.n_samples, "sample_rate": s.sample_rate,
                              "kind": str(s.kind).split(".")[-1]})
        self._save()
        return path

    def arrays(self, stage: str, **arrays) -> str | None:
        """Other products of `stage` (the most recent stage tapped, or a new one)."""
        if not self.wants(stage) or not arrays:
            return None
        new = not self.manifest or self.manifest[-1]["stage"] != stage
        path = self._path(stage, "npz", new=new)
        np.savez_compressed(path, **arrays)
        if new:
            self.manifest.append({"stage": stage})
        self.manifest[-1]["arrays"] = os.path.basename(path)
        self.manifest[-1]["array_names"] = sorted(arrays)
        self._save()
        return path

    def json(self, stage: str, obj) -> str | None:
        if not self.wants(stage):
            return None
        new = not self.manifest or self.manifest[-1]["stage"] != stage
        path = self._path(stage, "json", new=new)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=float)
        if new:
            self.manifest.append({"stage": stage})
        self.manifest[-1]["json"] = os.path.basename(path)
        self._save()
        return path

    def _save(self) -> None:
        with open(os.path.join(self.directory, "taps.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1)


def manifest(directory) -> list:
    with open(os.path.join(str(directory), "taps.json"), encoding="utf-8") as f:
        return json.load(f)


def read_tap(directory, stage) -> list:
    """The ping records tapped after `stage` (its name, or its index in the manifest)."""
    entries = [e for e in manifest(directory) if "records" in e]
    if isinstance(stage, int):
        entry = manifest(directory)[stage]
    else:
        matches = [e for e in entries if e["stage"] == stage]
        if not matches:
            raise KeyError(f"no tap of stage {stage!r}; tapped: {[e['stage'] for e in entries]}")
        entry = matches[-1]
    from .sources.container import ContainerSource
    with ContainerSource(os.path.join(str(directory), entry["records"])) as src:
        return list(src)


def read_arrays(directory, stage) -> dict:
    entry = [e for e in manifest(directory) if e["stage"] == stage and "arrays" in e][-1]
    with np.load(os.path.join(str(directory), entry["arrays"])) as z:
        return {k: z[k] for k in z.files}


# --- looking at a record -------------------------------------------------------------------------

def stats(record) -> list[dict]:
    """Per channel: RMS, peak, mean, the fraction of samples at the ADC rails, and the
    frequency of the strongest spectral line (Hz; for baseband records, about the carrier)."""
    x = record.samples.array
    fs = record.samples.sample_rate
    full_scale = float(record.metadata.adc_full_scale or 0.0)
    out = []
    for ch in x:
        mag = np.abs(ch)
        spec = np.abs(np.fft.fft(ch - ch.mean()))
        f = np.fft.fftfreq(len(ch), 1.0 / fs)
        k = int(np.argmax(spec if np.iscomplexobj(ch) else spec[:len(ch) // 2]))
        out.append({"rms": float(np.sqrt(np.mean(mag**2))), "peak": float(mag.max()), "mean": complex(ch.mean()).real,
                    "clipped_fraction": float(np.mean(mag >= 0.999 * full_scale)) if full_scale > 0 and not np.iscomplexobj(ch) else 0.0,
                    "strongest_line_hz": float(f[k] + (record.samples.center_frequency if np.iscomplexobj(ch) else 0.0))})
    return out


def export_wav(record, path, channels=None, normalise: bool = True) -> dict:
    """Write a record's samples as a 32-bit float WAV file: one WAV channel per sonar
    channel, or two (I then Q) per channel for a baseband record. Sample rates above what
    the header can hold are not a problem for signal tools; audio players may refuse them.
    Returns {"scale": the factor the samples were multiplied by, "layout": ...}."""
    x = record.samples.array
    rows = list(range(x.shape[0])) if channels is None else list(channels)
    x = x[rows]
    if np.iscomplexobj(x):
        x = np.stack([x.real, x.imag], axis=1).reshape(2 * len(rows), -1)
        layout = [f"ch{r}_{p}" for r in rows for p in "IQ"]
    else:
        layout = [f"ch{r}" for r in rows]
    scale = 1.0 / max(float(np.abs(x).max()), 1e-30) if normalise else 1.0
    data = (x * scale).astype("<f4").T.copy()                     # frames x channels
    fs, n_ch = int(round(record.samples.sample_rate)), data.shape[1]
    payload = data.tobytes()
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(payload), b"WAVE", b"fmt ", 16, 3, n_ch, fs,
                         fs * n_ch * 4, n_ch * 4, 32, b"data", len(payload))
    with open(path, "wb") as f:
        f.write(header + payload)
    return {"scale": scale, "layout": layout, "sample_rate": fs}


# --- inside the simulator ------------------------------------------------------------------------

_sim = {"dir": None, "ping": -1, "step": 0, "meta": [], "per_item": False}
SIM_STAGES = ("echoes", "reverberation", "reference_and_clocks", "noise", "hardware_gain", "adc")


@contextlib.contextmanager
def sim_capture(directory, per_item: bool = False):
    """While active, every simulated ping saves its samples after each simulator stage:
    <dir>/ping0000_1_echoes.npy and so on, and sim_taps.json describing them.

    `per_item=True` (opt-in, for the scene pages' sound: python/gpusonar/viz/audio.py) also saves the samples before
    the first reverberation item (stage `reverberation_start`, after the echoes and a basin's late tail) and after
    each item i (`reverberation_item_<i>`, with the item's model and boundary in its entry). The reverberation items
    run in one Python loop in sim/simulator.py whatever the engine, so these taps exist for both engines; like every
    tap they only read the record (it is bit-identical with or without them)."""
    os.makedirs(str(directory), exist_ok=True)
    _sim.update(dir=str(directory), ping=-1, step=0, meta=[], per_item=bool(per_item))
    try:
        yield
    finally:
        with open(os.path.join(_sim["dir"], "sim_taps.json"), "w", encoding="utf-8") as f:
            json.dump(_sim["meta"], f, indent=1)
        _sim.update(dir=None, per_item=False)


def sim_item_tap(record, stage: str, info=None) -> None:
    """A per-item reverberation tap (`sim_capture(..., per_item=True)`); does nothing otherwise. `info`: a dict, or a
    callable returning one (called only while capturing)."""
    if _sim["dir"] is None or not _sim["per_item"]:
        return
    sim_tap(record, stage, info=info() if callable(info) else info)


def sim_tap(record, stage: str, samples=None, info: dict | None = None) -> None:
    """Called by the simulator after each of its stages; does nothing unless capturing."""
    if _sim["dir"] is None:
        return
    if stage == SIM_STAGES[0]:
        _sim["ping"] += 1
        _sim["step"] = 0
    _sim["step"] += 1
    x = np.asarray(record.samples.array if samples is None else samples)
    name = f"ping{_sim['ping']:04d}_{_sim['step']}_{stage}.npy"
    np.save(os.path.join(_sim["dir"], name), x)
    _sim["meta"].append({"ping": _sim["ping"], "stage": stage, "file": name, "sample_rate": record.samples.sample_rate,
                         "center_frequency": record.samples.center_frequency, "start_time": record.samples.start_time,
                         "rms": float(np.sqrt(np.mean(np.abs(x) ** 2))), **({"info": info} if info else {})})


def read_sim_taps(directory, ping: int = 0) -> dict:
    """{stage: samples} of one simulated ping, in the order the simulator produced them."""
    with open(os.path.join(str(directory), "sim_taps.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return {m["stage"]: np.load(os.path.join(str(directory), m["file"])) for m in meta if m["ping"] == ping}
