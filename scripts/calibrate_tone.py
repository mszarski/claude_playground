"""Check the tone bank against conversational speech: what the voice model reads in everyday meeting talk (AMI).

  python scripts/calibrate_tone.py --bank runs/synth_v4/tone_bank.jsonl --out runs/synth_v4/calib.json --n 300

CREMA-D is acted in a studio, on twelve flat sentences, and the voice model's arousal / valence / dominance come out
far lower there than in TV dialogue (neutral: arousal 0.30, valence 0.37, against about 0.57 in MELD), and the
teacher reads most such lines as sad. This measures the same attributes on utterances from the AMI meeting corpus
(edinburghcstr/ami, CC BY 4.0, headset mics; mostly neutral talk) and writes the shift that would move CREMA-D's
neutral clips onto them, with the categorical readings on AMI.

Result (300 AMI utterances): arousal 0.25, valence 0.43, read neutral 46% / sad 37% / happy 16%. Everyday talk reads
as low as CREMA-D (shift at most 0.06), so the readings are used as they are; it is MELD's loud TV dialogue that is
the outlier.
"""
import argparse
import json
import random
from collections import Counter

DIMS = ("arousal", "dominance", "valence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from rmr.voice import EmotionModel, load_audio

    pf = pq.ParquetFile(hf_hub_download("edinburghcstr/ami", "ihm/test-00000-of-00004.parquet", repo_type="dataset"))
    meta = pf.read(columns=["begin_time", "end_time", "text"]).to_pandas()
    ok = [i for i, r in meta.iterrows() if 2.0 <= r.end_time - r.begin_time <= 10.0 and len(r.text.split()) >= 4]
    pick = set(random.Random(0).sample(ok, min(a.n, len(ok))))
    emo, dims = EmotionModel("msp", device=a.device), EmotionModel("msp-dims", device=a.device)
    rows, i0 = [], 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["audio"]).to_pylist()
        for j, r in enumerate(t):
            if i0 + j in pick:
                wav = load_audio(r["audio"]["bytes"])
                p = emo.predict(wav)
                rows.append({"emotion": max(p, key=p.get), **dims.predict(wav)})
        i0 += len(t)
    bank = [json.loads(x) for x in open(a.bank)]
    neutral = [b for b in bank if b["crema"] == "Neutral"]
    ami = {d: float(np.mean([r[d] for r in rows])) for d in DIMS}
    crema = {d: float(np.mean([b[d] for b in neutral])) for d in DIMS}
    out = {"shift": {d: round(ami[d] - crema[d], 4) for d in DIMS}, "ami_mean": ami, "crema_neutral_mean": crema,
           "ami_n": len(rows), "ami_emotions": dict(Counter(r["emotion"] for r in rows).most_common())}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
