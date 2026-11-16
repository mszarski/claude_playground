"""gpusonar: one GPU sonar suite for FLS, SAS and long-range flank arrays."""

from . import _gpusonar as core
from ._gpusonar import (
    Mode,
    PingRecord,
    RecordReader,
    RecordWriter,
    SampleKind,
    StorageType,
    TransmitKind,
    WaveformType,
    WriterOptions,
    cuda_available,
    cuda_compiled,
    validate,
    validate_file,
)

__version__ = "0.1.0"

__all__ = [
    "core",
    "Mode",
    "PingRecord",
    "RecordReader",
    "RecordWriter",
    "SampleKind",
    "StorageType",
    "TransmitKind",
    "WaveformType",
    "WriterOptions",
    "cuda_available",
    "cuda_compiled",
    "validate",
    "validate_file",
]
