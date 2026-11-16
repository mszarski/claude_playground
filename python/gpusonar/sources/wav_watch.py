"""Recordings as they appear: a directory the capture boards (or a copy job) write into.

Each recording is a .wav file (or one per channel) with a YAML sidecar (sources/wav.py).
A sidecar is taken once its audio files have stopped growing. This is the capture path
that needs nothing from the boards beyond what they already do: write files.
"""

from __future__ import annotations

import os
import time
from typing import Iterator

import yaml

from .._gpusonar import PingRecord
from .base import Source, register
from .wav import WavSource


@register
class WavWatchSource(Source):
    """Yields the ping records of every sidecar that appears in `directory`, oldest first.
    Ends after `timeout` seconds without a new one (None: never), or after `max_files`."""

    type_name = "wav_watch"

    def __init__(self, directory: str, pattern: str = ".yaml", poll: float = 0.5, settle: float = 1.0,
                 timeout: float | None = 30.0, max_files: int | None = None):
        self.directory, self.pattern = str(directory), pattern
        self.poll, self.settle, self.timeout, self.max_files = float(poll), float(settle), timeout, max_files
        self.done: list[str] = []
        self.failed: dict[str, str] = {}

    def _ready(self, sidecar: str) -> bool:
        try:
            with open(sidecar, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            files = cfg.get("files") or [cfg["file"]]
            newest = max(os.path.getmtime(os.path.join(self.directory, name)) for name in files)
        except (OSError, KeyError, TypeError, yaml.YAMLError):
            return False                      # still being written, or its audio is not there yet
        return time.time() - max(newest, os.path.getmtime(sidecar)) >= self.settle

    def __iter__(self) -> Iterator[PingRecord]:
        last = time.time()
        while True:
            names = sorted((n for n in os.listdir(self.directory) if n.endswith(self.pattern)),
                           key=lambda n: os.path.getmtime(os.path.join(self.directory, n)))
            fresh = [n for n in names if n not in self.done and n not in self.failed]
            progressed = False
            for name in fresh:
                path = os.path.join(self.directory, name)
                if not self._ready(path):
                    continue
                try:
                    with WavSource(path) as src:
                        records = list(src)
                except Exception as exc:       # a bad recording must not stop the mission
                    self.failed[name] = str(exc)
                    continue
                self.done.append(name)
                progressed = True
                last = time.time()
                yield from records
                if self.max_files is not None and len(self.done) >= self.max_files:
                    return
            if not progressed:
                if self.timeout is not None and time.time() - last > self.timeout:
                    return
                time.sleep(self.poll)
