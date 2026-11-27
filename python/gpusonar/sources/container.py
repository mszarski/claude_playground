"""File replay of our own HDF5 container."""

from __future__ import annotations

from typing import Iterator

from .._gpusonar import PingRecord, RecordReader
from .base import Source, register


@register
class ContainerSource(Source):
    type_name = "container"

    def __init__(self, path: str, start: int = 0, stop: int | None = None, sonar: str | None = None):
        # `sonar`: the sonar the recording is of, as a deployment file names it (configs/deploy/fls_*.yaml: a sonar
        # with no stream source yet replays recordings). Read by what picks a deployment file by sonar (the viewer, the
        # tests); not checked against the records, which carry their own geometry.
        self.sonar = sonar
        self._reader = RecordReader(str(path))
        n = len(self._reader)
        self._start = max(0, start)
        self._stop = n if stop is None else min(stop, n)

    @property
    def info(self):
        return self._reader.info

    def __len__(self) -> int:
        return max(0, self._stop - self._start)

    def read(self, index: int) -> PingRecord:
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._reader.read(self._start + index)

    def __iter__(self) -> Iterator[PingRecord]:
        for i in range(len(self)):
            yield self.read(i)

    def close(self) -> None:
        self._reader = None
