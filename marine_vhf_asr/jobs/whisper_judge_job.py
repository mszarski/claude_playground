# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.45", "accelerate", "soundfile", "scipy", "numpy", "huggingface_hub"]
# ///
"""
Independent audio sanity check: Whisper large-v3 on synthetic test splits.

Reports Whisper WER per split (the same transmissions in each, for comparing TTS
engines) and writes per-clip Whisper transcripts, so clips where the TTS engine
garbled the speech can be excluded before scoring our models.

    hf jobs uv run --flavor a10g-large --timeout 1h --secrets HF_TOKEN \\
        --env CODE_REPO=<user>/marine-vhf-asr-code --env SPLITS=test_ny_kv,test_ny_cb \\
        --env LIMIT=200 --env NAME=whisper_engine_check marine_vhf_asr/jobs/whisper_judge_job.py
"""
import json, os, sys
import numpy as np, torch
from huggingface_hub import snapshot_download, upload_file
sys.path.insert(0, snapshot_download(os.environ["CODE_REPO"]))
from marine_vhf_asr.sources import load_audio
from marine_vhf_asr.metrics import corpus_wer, edit_distance
from marine_vhf_asr.normalize import normalize_transcript as n
from transformers import pipeline

SPLITS = os.environ.get("SPLITS", "test_ny_kv,test_ny_cb,test_ny_pa0").split(",")
data = snapshot_download("mszarski/marine-vhf-ais-synth", repo_type="dataset", max_workers=32,
                         allow_patterns=[f"{s}/*" for s in SPLITS])
load = lambda s: [json.loads(l) for l in open(f"{data}/{s}/manifest.jsonl")]
key = lambda r: (r["dialogue_id"], r["turn"])
LIMIT = int(os.environ.get("LIMIT", "200"))
want = {key(r) for r in load(SPLITS[-1])[:LIMIT]} if LIMIT else None
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda",
               chunk_length_s=30)
out = {}
per_clip = []
for split in SPLITS:
    rows = [r for r in load(split) if want is None or key(r) in want]
    audios = [load_audio({"path": f"{data}/{split}/{r['audio']}", "bytes": None}) for r in rows]
    hyps = [h["text"] for h in asr([{"raw": a, "sampling_rate": 16000} for a in audios], batch_size=4,
                                   generate_kwargs={"language": "en"})]
    per = [edit_distance(n(r["text"], numbers="cardinal").split(), n(h, numbers="cardinal").split()) / max(1, len(n(r["text"]).split())) for r, h in zip(rows, hyps)]
    secs_per_word = np.array([len(a) / 16000 / max(1, len(r["text"].split())) for a, r in zip(audios, rows)])
    per_clip += [{"split": split, "dialogue_id": r["dialogue_id"], "turn": r["turn"], "text": r["text"],
                  "whisper": h, "whisper_wer": p} for r, h, p in zip(rows, hyps, per)]
    out[split] = {"n": len(rows), "secs_per_word_median": float(np.median(secs_per_word)),
                  "clips_over_1s_per_word": float(np.mean(secs_per_word > 1.0)), "wer": corpus_wer(((n(r["text"], numbers="cardinal"), n(h, numbers="cardinal")) for r, h in zip(rows, hyps)), normalize=False),
                  "clips_over_50pct": float(np.mean(np.array(per) > 0.5)),
                  "examples": [{"ref": r["text"], "hyp": h, "wer": p} for r, h, p in zip(rows[:40], hyps[:40], per[:40])]}
    print("RESULT", split, out[split]["n"], f"wer {out[split]['wer']:.1%}", f"over50 {out[split]['clips_over_50pct']:.0%}",
          f"s/word {out[split]['secs_per_word_median']:.2f}", f"slow {out[split]['clips_over_1s_per_word']:.0%}", flush=True)
name = os.environ.get("NAME", "whisper_engine_check")
open("/tmp/w.json", "w").write(json.dumps(out, indent=1))
with open("/tmp/c.jsonl", "w") as f:
    for c in per_clip:
        f.write(json.dumps(c) + "\n")
upload_file(path_or_fileobj="/tmp/c.jsonl", path_in_repo=f"diagnostics/{name}_per_clip.jsonl",
            repo_id="mszarski/marine-vhf-ais-synth", repo_type="dataset", commit_message="Whisper per-clip judge")
upload_file(path_or_fileobj="/tmp/w.json", path_in_repo=f"diagnostics/{name}.json",
            repo_id="mszarski/marine-vhf-ais-synth", repo_type="dataset", commit_message="Whisper sanity check per TTS engine")
print("UPLOADED", flush=True)
