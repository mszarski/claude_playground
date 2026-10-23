"""
Transcripts built from real AIS traffic, for any region with an AIS extract.

For each dialogue: pick a region and a moment, take the AIS snapshot of vessels
within radio range of that region's receiver, pick who is talking the way a
harbor actually talks (ships and tugs to VTS and each other, pleasure craft to
the Coast Guard), and fill an SMCP dialogue with their real names. Each
transmission records the candidate lists a real system could pass as context:
    moving_a  class A transponders that moved in the last 10 minutes (~85 names in NY)
    all       everything heard in range (~350 names in NY)
Speakers are
    commercial   on both lists
    small_craft  on `all` only (pleasure, sailing, fishing)
    unknown      on neither: a vessel the system has never seen ("a vessel out
                 there"), named from --unknown-names
--exclude-names keeps a set of names (e.g. every vessel in the test regions)
out of the dialogues and the lists, so test vessels are unseen in training.

Writes a text-only JSONL manifest; synthesize audio with jobs/harbor_tts_job.py.

Usage:
    python -m marine_vhf_asr.harbor --ais-root data/ais --regions new_york \\
        --dialogues 300 --unknown-names data/names_oresund.txt --out data/test_ny/text.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import ais
from .gazetteer import port_for_region
from .normalize import normalize_transcript
from .smcp import DialogueGenerator

COMMERCIAL = {"cargo", "tanker", "passenger", "tug", "towing", "port_service", "pilot_sar_law", "other"}
SMALL_CRAFT = {"pleasure", "sailing", "fishing"}

# Scenarios that fit each kind of speaker, with weights
SCENARIOS = {
    "commercial": {"vts_entry_report": 3, "berthing": 2, "ship_to_ship": 3, "pilot_request": 1,
                   "securite": 1, "radio_check": 1, "pan_pan": 0.3, "mayday": 0.2},
    "small_craft": {"radio_check": 3, "ship_to_ship": 1, "pan_pan": 1, "mayday": 0.5},
}
SCENARIOS["unknown"] = SCENARIOS["small_craft"]


def _pick(rng, weights):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def load_regions(ais_root, regions, days=None):
    """{region: AIS dataframe} from <ais_root>/data/<region>/<day>.parquet."""
    out = {}
    for region in regions:
        paths = sorted(Path(ais_root, "data", region).glob("*.parquet"))
        if days:
            paths = [p for p in paths if p.stem in days]
        if paths:
            out[region] = ais.load(paths)
    return out


def region_names(dfs):
    """Spoken names of every vessel in the given AIS dataframes."""
    names = set()
    for df in dfs:
        names |= {ais.spoken_name(n) for n in df.VesselName.dropna().unique()}
    return names - {None}


def build_dialogues(regional, n_dialogues, seed=0, shares=(0.6, 0.25, 0.15), unknown_names=(),
                    exclude=frozenset(), min_listed=20):
    """
    Yield one manifest record per transmission.

    Args:
        regional: {region: AIS dataframe}
        shares: fraction of commercial / small_craft / unknown speakers
        unknown_names: pool for unknown speakers (filtered against the snapshot)
        exclude: spoken names never used, in dialogues or lists
    """
    rng = np.random.default_rng(seed)
    gen = DialogueGenerator(seed=seed)
    regions = sorted(regional)
    unknown_names = sorted(set(unknown_names) - set(exclude))
    made = attempts = 0
    while made < n_dialogues:
        attempts += 1
        if attempts > 50 * n_dialogues + 1000:
            raise RuntimeError(f"Only {made} dialogues after {attempts} attempts; relax min_listed or shares")
        region = regions[rng.integers(len(regions))]
        df, port = regional[region], port_for_region(region)
        days = df["t"].dt.normalize().unique()
        # 06:00-20:00 local, roughly, for US (UTC-4..-8) and Denmark (UTC+1..+2)
        local_start = 10 if port.receiver[1] < 0 else 5
        t = days[rng.integers(len(days))] + np.timedelta64(int(rng.integers(local_start * 60, (local_start + 14) * 60)), "m")
        snap = [v for v in ais.snapshot(df, t, *port.receiver) if v.name not in exclude]
        lists = ais.candidate_lists(snap)
        listed = [v for v in snap if v.class_a and v.moving and v.group in COMMERCIAL]
        small = [v for v in snap if v.group in SMALL_CRAFT and v.name not in set(lists["moving_a"])]
        heard = set(lists["all"])
        unknown = [n for n in unknown_names if n not in heard]
        if len(lists["moving_a"]) < min_listed or not listed:
            continue

        kind = ["commercial", "small_craft", "unknown"][rng.choice(3, p=np.array(shares) / sum(shares))]
        if kind == "small_craft" and not small or kind == "unknown" and not unknown:
            continue
        if kind == "unknown":
            speaker_name, group = unknown[rng.integers(len(unknown))], "unknown"
        else:
            pool = small if kind == "small_craft" else listed
            v = pool[rng.integers(len(pool))]
            speaker_name, group = v.name, v.group
        other = listed[rng.integers(len(listed))].name
        scenario = _pick(rng, SCENARIOS[kind])
        names = [speaker_name, other] if other != speaker_name else [speaker_name]
        dialogue = gen.generate(scenario, port, vessels=names)

        for i, turn in enumerate(dialogue.turns):
            text = normalize_transcript(turn.text, numbers="cardinal", expand_letters=False)
            spoken = [nm for nm in names if f" {nm} " in f" {text} "]
            yield {
                "tts_text": turn.text, "text": text,
                "slots": {f"vessel_{k}": nm for k, nm in enumerate(spoken)},
                "contexts": lists, "region": region,
                "speaker_kind": kind, "speaker_group": group, "turn_speaker": turn.speaker,
                "scenario": scenario, "dialogue_id": made, "turn": i, "snapshot_utc": str(t),
            }
        made += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ais-root", required=True, type=Path, help="Folder with data/<region>/<day>.parquet")
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--days", nargs="*", help="Only these days (YYYY-MM-DD)")
    parser.add_argument("--dialogues", type=int, default=300)
    parser.add_argument("--shares", default="0.6,0.25,0.15", help="commercial,small_craft,unknown speaker shares")
    parser.add_argument("--unknown-names", type=Path, help="Text file, one spoken name per line")
    parser.add_argument("--exclude-names", type=Path, help="Text file of spoken names never to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    read = lambda p: [l.strip() for l in open(p) if l.strip()] if p else []  # noqa: E731
    regional = load_regions(args.ais_root, args.regions, args.days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for rec in build_dialogues(regional, args.dialogues, args.seed, tuple(map(float, args.shares.split(","))),
                                   read(args.unknown_names), frozenset(read(args.exclude_names))):
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Wrote {n} transmissions from {args.dialogues} dialogues ({', '.join(sorted(regional))}) to {args.out}")


if __name__ == "__main__":
    main()
