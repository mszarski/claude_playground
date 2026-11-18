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


_EXTRA_DIRS: list[Path] = []


def add_config_dir(path) -> None:
    """Also look for named configs under `path` (a tree with sonars/, scenarios/, ...), after the shipped
    ones: how a generated build (tools/build_from_spec.py) is used without copying it into configs/.
    The environment variable GPUSONAR_CONFIG_PATH (directories separated by the path separator) does
    the same for programs that are not told in code."""
    p = Path(path)
    if p not in _EXTRA_DIRS:
        _EXTRA_DIRS.append(p)


def remove_config_dir(path) -> None:
    if Path(path) in _EXTRA_DIRS:
        _EXTRA_DIRS.remove(Path(path))


def config_dirs() -> list[Path]:
    """The shipped root first, then GPUSONAR_CONFIG_PATH, then `add_config_dir` directories."""
    env = [Path(d) for d in os.environ.get("GPUSONAR_CONFIG_PATH", "").split(os.pathsep) if d]
    return [config_dir()] + env + list(_EXTRA_DIRS)


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
    for root in config_dirs():
        candidate = root / kind / f"{name_or_path}.yaml"
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
        parent = _load_with_base(kind, base, seen + (path,))
        cfg = merge_flags(parent, cfg, deep_merge(parent, cfg))
    return cfg


def _covers(entry: str, flag: str) -> bool:
    """`entry` names `flag` itself or something above it (`receive` covers `receive.boards`)."""
    return flag == entry or flag.startswith(entry + ".")


def merge_flags(parent: dict, child: dict, merged: dict) -> dict:
    """The `unconfirmed` and `confirmed` lists of a config that extends another.

    `unconfirmed` lists dotted keys whose values are assumptions. A list is not a mapping, so a
    plain merge would let a child's list REPLACE its base's and hide the base's flags. Here they
    accumulate: the result holds the base's entries and then the child's. A child takes a flag off
    by naming it under `confirmed` (the value is now measured or stated): that removes the entry
    itself and anything below it (`confirmed: [receive]` clears `receive.boards` too), and the
    `confirmed` lists accumulate the same way, so the record of what was confirmed stays. A key
    named in a child's `unconfirmed` again is unconfirmed again (the later file wins).
    """
    if "unconfirmed" not in parent and "unconfirmed" not in child and "confirmed" not in child:
        return merged
    for key in ("unconfirmed", "confirmed"):
        for cfg in (parent, child):
            if not isinstance(cfg.get(key, []), list):
                raise ValueError(f"`{key}` must be a list of dotted keys")
    newly_confirmed = [str(k) for k in child.get("confirmed", [])]
    again = [str(k) for k in child.get("unconfirmed", [])]
    flags = [str(k) for k in parent.get("unconfirmed", []) if not any(_covers(c, str(k)) for c in newly_confirmed)]
    flags += [k for k in again if k not in flags]
    confirmed = [str(k) for k in parent.get("confirmed", []) if not any(_covers(str(k), a) or _covers(a, str(k)) for a in again)]
    confirmed += [k for k in newly_confirmed if k not in confirmed and k not in again]
    out = dict(merged)
    out["unconfirmed"] = flags
    if confirmed:
        out["confirmed"] = confirmed
    else:
        out.pop("confirmed", None)
    return out


def load_sonar_config(name_or_path, overrides: dict | None = None) -> dict:
    """Resolved sonar config: a name under configs/sonars or a path, plus overrides."""
    cfg = _load_with_base("sonars", name_or_path)
    return deep_merge(cfg, overrides) if overrides else cfg


def load_scenario_config(name_or_path, overrides: dict | None = None) -> dict:
    cfg = _load_with_base("scenarios", name_or_path)
    return deep_merge(cfg, overrides) if overrides else cfg


def list_configs(kind: str) -> list[str]:
    """Names of the shipped configs of a kind (extra directories are not listed)."""
    return sorted(p.stem for p in (config_dir() / kind).glob("*.yaml"))


def dump_yaml(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False)
