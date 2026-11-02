"""
Compare v6 and Nemotron (v5 is loaded too) on the Parler test clips Whisper large-v3 finds intelligible.

A Parler clip counts only if Whisper recovers >= 80% of its non-name words
(judge output of jobs/whisper_judge_job.py). Run from a folder holding:
    judge/test_{ny,oresund}/manifest.jsonl                       (from mszarski/marine-vhf-ais-synth)
    judge/diagnostics/whisper_parler_judge_per_clip.jsonl          (same dataset)
    pa/diagnostics/parler_test_predictions.jsonl                   (v5 eval predictions)
    v6e/diagnostics/v6_eval_predictions.jsonl                      (v6 eval predictions)
    nres/diagnostics/nemotron_eval_predictions.jsonl               (Nemotron predictions)
with PYTHONPATH pointing at the repo root.
"""
import json, sys, collections
from marine_vhf_asr.normalize import normalize_transcript as n
from marine_vhf_asr.metrics import corpus_wer, slot_accuracy, name_cer
kind = {}
for reg in ["test_ny", "test_oresund"]:
    for r in map(json.loads, open(f"judge/{reg}/manifest.jsonl")):
        kind[(reg, r["text"])] = r["speaker_kind"]
judge = collections.defaultdict(list)
for c in map(json.loads, open("judge/diagnostics/whisper_parler_judge_per_clip.jsonl")):
    ref = c["text"]
    judge[(c["split"], ref)].append(c)
def recall(c, names):
    ref = f" {c['text']} "
    for nm in names: ref = ref.replace(f" {nm} ", " ")
    words = ref.split()
    hyp = set(n(c["whisper"], numbers="cardinal", expand_letters=False).split())
    return sum(w in hyp for w in words) / max(1, len(words))
def load(path):
    out = collections.defaultdict(list)
    for p in map(json.loads, open(path)):
        if "_pa" not in p["spec"]: continue
        split, cond = p["spec"].split("/")[0], p["spec"].split("#")[1]
        cs = judge.get((split, p["text"]), [])
        ok = bool(cs) and min(recall(c, p["slots"].values()) for c in cs) >= 0.8
        reg = split.rsplit("_", 1)[0]
        out[(reg, cond)].append(dict(p, ok=ok, kind=kind.get((reg, p["text"]), "?")))
    return out
models = {"v5": load("pa/diagnostics/parler_test_predictions.jsonl"), "v6": load("v6e/diagnostics/v6_eval_predictions.jsonl"), "nem": load("nres/diagnostics/nemotron_eval_predictions.jsonl")}
def fmt(ps):
    sl = [(p["slots"], p["hyp"]) for p in ps if p["slots"]]
    return f"WER {corpus_wer((p['text'], p['hyp']) for p in ps):5.1%}  names {slot_accuracy(sl)['all']:4.0%}  CER {name_cer(sl):.2f}" if sl else "-"
for key in sorted(models["v5"]):
    all5, all6, alln = models["v5"][key], models["v6"][key], models["nem"][key]
    ok5 = [p for p in all5 if p["ok"]]; ok6 = [p for p in all6 if p["ok"]]; okn = [p for p in alln if p["ok"]]
    print(f"{key[0]:13s} {key[1]:9s} intelligible {len(ok6)}/{len(all6)} nem {len(okn)}/{len(alln)} | v6 {fmt(ok6)} | nem {fmt(okn)}")
    for k in ["commercial", "small_craft", "unknown"]:
        b = [p for p in ok6 if p["kind"] == k]; c = [p for p in okn if p["kind"] == k]
        print(f"   {k:12s} n={len(b):4d}  | v6 {fmt(b)} | nem {fmt(c)}")
