"""YAML configs: named sonars and scenarios, with `base:` inheritance."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads `1e3` and `2.5e6` as floats.

    PyYAML follows YAML 1.1, where a float needs a dot and a signed exponent, so
    `270.0e3` would silently load as a string.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^[-+]?(?:[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                   |[0-9][0-9_]*[eE][-+]?[0-9]+
                   |\.(?:inf|Inf|INF)
                   |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def config_dir() -> Path:
    """Root of the shipped configs. Override with GPUSONAR_CONFIG_DIR."""
    env = os.environ.get("GPUSONAR_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Return `base` updated by `override`; nested mappings merge, everything else replaces."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_Loader)  # noqa: S506 (a SafeLoader subclass)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return data


def _resolve(kind: str, name_or_path) -> Path:
    p = Path(str(name_or_path))
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    candidate = config_dir() / kind / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    known = sorted(q.stem for q in (config_dir() / kind).glob("*.yaml"))
    raise FileNotFoundError(f"no {kind[:-1]} config {name_or_path!r}; known: {', '.join(known)}")


def _load_with_base(kind: str, name_or_path, seen=()) -> dict:
    path = _resolve(kind, name_or_path)
    if path in seen:
        raise ValueError(f"config inheritance loop at {path}")
    cfg = load_yaml(path)
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_merge(_load_with_base(kind, base, seen + (path,)), cfg)
    return cfg


def load_sonar_config(name_or_path, overrides: dict | None = None) -> dict:
    """Resolved sonar config: a name under configs/sonars or a path, plus overrides."""
    cfg = _load_with_base("sonars", name_or_path)
    return deep_merge(cfg, overrides) if overrides else cfg


def load_scenario_config(name_or_path, overrides: dict | None = None) -> dict:
    cfg = _load_with_base("scenarios", name_or_path)
    return deep_merge(cfg, overrides) if overrides else cfg


def list_configs(kind: str) -> list[str]:
    return sorted(p.stem for p in (config_dir() / kind).glob("*.yaml"))


def dump_yaml(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False)
