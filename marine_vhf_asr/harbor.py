"""
Harbor-specific test/training transcripts built from real AIS traffic.

For each dialogue: pick a moment, take the AIS snapshot of vessels within radio
range of the receiver, pick who is talking the way the harbor actually talks
(ships and tugs to VTS and each other, pleasure craft to the Coast Guard), and
fill an SMCP dialogue with their real names. Each transmission records the
candidate lists a real system could pass as context:
    moving_a  class A transponders that moved in the last 10 minutes (~85 names in NY)
    all       everything heard in range (~350 names in NY)
About a quarter of speakers are pleasure craft, which are on `all` but not on
`moving_a`, so both "speaker listed" and "speaker not listed" cases occur.

Writes a text-only JSONL manifest; synthesize audio with jobs/harbor_tts_job.py.

Usage:
    python -m marine_vhf_asr.harbor --ais data/ais/*.parquet --port "New York" \\
        --receiver 40.61,-74.06 --dialogues 300 --out data/nyc_test/text.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import ais
from .gazetteer import PORTS
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


def _pick(rng, weights):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def build_dialogues(df, port, receiver, n_dialogues, seed=0, small_craft_share=0.25, min_listed=40):
    """Yield one manifest record per transmission."""
    rng = np.random.default_rng(seed)
    gen = DialogueGenerator(seed=seed)
    t0, t1 = df["t"].min(), df["t"].max()
    days = sorted(df["t"].dt.normalize().unique())
    made = 0
    while made < n_dialogues:
        day = days[rng.integers(len(days))]
        # 10:00-24:00 UTC is roughly 06:00-20:00 in New York
        t = day + np.timedelta64(int(rng.integers(10 * 60, 24 * 60)), "m")
        if not (t0 <= t <= t1):
            continue
        snap = ais.snapshot(df, t, *receiver)
        lists = ais.candidate_lists(snap)
        listed = [v for v in snap if v.class_a and v.moving and v.group in COMMERCIAL]
        small = [v for v in snap if v.group in SMALL_CRAFT and v.name not in set(lists["moving_a"])]
        if len(lists["moving_a"]) < min_listed or not listed or not small:
            continue

        kind = "small_craft" if rng.random() < small_craft_share else "commercial"
        speaker = (small if kind == "small_craft" else listed)[rng.integers(len(small if kind == "small_craft" else listed))]
        other = listed[rng.integers(len(listed))]
        scenario = _pick(rng, SCENARIOS[kind])
        names = [speaker.name, other.name] if other.name != speaker.name else [speaker.name]
        dialogue = gen.generate(scenario, port, vessels=names)

        for i, turn in enumerate(dialogue.turns):
            text = normalize_transcript(turn.text, numbers="cardinal", expand_letters=False)
            spoken = [nm for nm in names if f" {nm} " in f" {text} "]
            yield {
                "tts_text": turn.text, "text": text,
                "slots": {f"vessel_{k}": nm for k, nm in enumerate(spoken)},
                "contexts": lists,
                "speaker_kind": kind, "speaker_group": speaker.group, "turn_speaker": turn.speaker,
                "scenario": scenario, "dialogue_id": made, "turn": i, "snapshot_utc": str(t),
            }
        made += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ais", nargs="+", required=True, help="AIS parquet extracts")
    parser.add_argument("--port", default="New York")
    parser.add_argument("--receiver", default="40.61,-74.06", help="lat,lon of the listening station")
    parser.add_argument("--dialogues", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    port = next(p for p in PORTS if p.name == args.port)
    receiver = tuple(map(float, args.receiver.split(",")))
    df = ais.load(args.ais)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for rec in build_dialogues(df, port, receiver, args.dialogues, args.seed):
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Wrote {n} transmissions from {args.dialogues} dialogues to {args.out}")


if __name__ == "__main__":
    main()
