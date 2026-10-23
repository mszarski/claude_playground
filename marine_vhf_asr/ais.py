"""
AIS (Automatic Identification System) helpers: real vessel lists for prompts.

Works on NOAA MarineCadastre-style columns (MMSI, BaseDateTime, LAT, LON, SOG,
VesselName, VesselType, TransceiverClass, ...), e.g. the extract written by
jobs/ais_extract_job.py.

A *snapshot* is the set of vessels a receiver could hear at a moment: every
vessel that reported a position within `radius_nm` of the receiver during the
preceding `window_min` minutes. Snapshots supply realistic candidate lists
and real vessel names for synthetic dialogues.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .normalize import cardinal, normalize_transcript, spell_digits

EARTH_RADIUS_NM = 3440.065
VOWELS = set("AEIOUY")
# Vowel-less tokens are read as letters; these contain a Y but are acronyms too
LETTER_ACRONYMS = {"NYPD", "FDNY", "NYFD", "NYSB", "NYS", "NYC", "NY", "NYWW", "NYCDEP", "NYCEDC"}

# MarineCadastre VesselType codes, grouped by how a vessel uses the radio
TYPE_GROUPS = {
    "fishing": range(30, 31), "towing": range(31, 33), "sailing": range(36, 37),
    "pleasure": range(37, 38), "pilot_sar_law": range(50, 52), "tug": range(52, 53),
    "port_service": range(53, 60), "passenger": range(60, 70), "cargo": range(70, 80),
    "tanker": range(80, 90),
}


def type_group(code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "unknown"
    return next((g for g, r in TYPE_GROUPS.items() if code in r), "other")


def spoken_name(ais_name: str) -> Optional[str]:
    """
    Convert an AIS vessel name to how it is said on the radio, in transcript form.

    "SCORPIO 2" -> "SCORPIO TWO", "NYPD353" -> "N Y P D THREE FIVE THREE",
    "BUCHANAN #12" -> "BUCHANAN TWELVE". Vowel-less 2-4 letter tokens are read as
    letters. Returns None for names that are not speakable (empty, MMSI-like).
    """
    if not ais_name:
        return None
    name = re.sub(r"[^A-Z0-9 &'-]", " ", ais_name.upper())
    name = re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", name).replace("&", " AND ")
    words = []
    for tok in re.split(r"[\s-]+", name):
        tok = tok.strip("'")
        if not tok:
            continue
        if tok.isdigit():
            words.append(cardinal(tok) if len(tok) <= 2 else spell_digits(tok))
        elif len(tok) == 1 or tok in LETTER_ACRONYMS or (len(tok) <= 4 and not set(tok) & VOWELS):
            words.extend(tok)
        else:
            words.append(tok)
    # Same normalization as transcripts, so names match references exactly
    spoken = normalize_transcript(" ".join(words), numbers="cardinal", expand_letters=False)
    letters = sum(c.isalpha() for c in ais_name)
    if letters < 2 or spoken in {"UNKNOWN", "NONE", "NA", "N A", "TEST"}:
        return None
    return spoken


def haversine_nm(lat, lon, lat0, lon0):
    la, lo, la0, lo0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(a))


@dataclass
class Vessel:
    mmsi: int
    name: str            # spoken transcript form
    ais_name: str
    group: str           # type_group()
    class_a: bool
    moving: bool         # max SOG in the window > 1 kn


def snapshot(df, t, lat0: float, lon0: float, radius_nm: float = 20.0, window_min: int = 10) -> List[Vessel]:
    """Vessels heard within `radius_nm` of (lat0, lon0) in the `window_min` minutes before `t`."""
    import pandas as pd

    t = pd.Timestamp(t)
    w = df[(df["t"] > t - pd.Timedelta(minutes=window_min)) & (df["t"] <= t)]
    w = w[haversine_nm(w.LAT.values, w.LON.values, lat0, lon0) <= radius_nm]
    out = []
    for mmsi, g in w.groupby("MMSI"):
        names = g.VesselName.dropna()
        name = spoken_name(names.iloc[-1]) if len(names) else None
        if not name:
            continue
        out.append(Vessel(int(mmsi), name, names.iloc[-1], type_group(g.VesselType.dropna().iloc[-1] if g.VesselType.notna().any() else None),
                          bool((g.TransceiverClass == "A").any()), bool((g.SOG > 1).any())))
    return out


def load(paths) -> "pd.DataFrame":
    """Load one or more AIS parquet extracts and parse timestamps."""
    import pandas as pd

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["t"] = pd.to_datetime(df.BaseDateTime)
    return df.sort_values("t")


def candidate_lists(vessels: List[Vessel]) -> Dict[str, List[str]]:
    """The list variants evaluated: what a real system might pass as context."""
    dedup = lambda vs: sorted({v.name for v in vs})  # noqa: E731
    return {
        "moving_a": dedup(v for v in vessels if v.class_a and v.moving),
        "all": dedup(vessels),
    }
