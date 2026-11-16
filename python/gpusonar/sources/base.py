"""Source base class. Every data reader derives from it and yields ping records
(ASASIN Sec. II: "all data readers are derived from a base class")."""

from __future__ import annotations

import abc
from typing import Iterator

from .._gpusonar import PingRecord


class Source(abc.ABC):
    """A finite or live stream of ping records.

    Subclasses implement `__iter__`. Sources that know their length also
    implement `__len__` and `read(index)` for random access.
    """

    #: short name used in configs (`source: {type: <name>, ...}`)
    type_name: str = ""

    @abc.abstractmethod
    def __iter__(self) -> Iterator[PingRecord]:
        ...

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} has no known length")

    def read(self, index: int) -> PingRecord:
        raise TypeError(f"{type(self).__name__} does not support random access")

    def close(self) -> None:
        """Release files or devices. Safe to call twice."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_REGISTRY: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    """Class decorator: make a source constructible from config by its `type_name`."""
    if not cls.type_name:
        raise ValueError(f"{cls.__name__} needs a type_name")
    _REGISTRY[cls.type_name] = cls
    return cls


def make_source(config: dict) -> Source:
    """Build a source from a config mapping `{type: name, **kwargs}`."""
    kwargs = dict(config)
    name = kwargs.pop("type", None)
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown source type {name!r}; known types: {known}")
    return _REGISTRY[name](**kwargs)
