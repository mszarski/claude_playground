"""
Score Nemotron-3.5-Transcribe (served by vLLM on localhost) on marine_vhf_asr source specs.
Candidate-vessel lists go in the model's own contextual-ASR prompt ("Context terms: ...").
Outputs get the same runaway guards and normalization as our own models.
"""
import base64, io, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from huggingface_hub import snapshot_download, upload_file
from openai import OpenAI
from scipy.io import wavfile

sys.path.insert(0, snapshot_download(os.environ["CODE_REPO"]))
from marine_vhf_asr.metrics import corpus_wer, decoy_copy_rate, name_cer, slot_accuracy  # noqa: E402
from marine_vhf_asr.prompting import candidate_names, collapse_loops, trim_runaway  # noqa: E402
from marine_vhf_asr.sources import SR, load_audio, load_source  # noqa: E402

BASE = ("Transcribe word for word without punctuation. Write numbers as spoken words "
        "and keep individually spoken letters separate.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=600)


def transcribe(audio, context):
    buf = io.BytesIO()
    wavfile.write(buf, SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    b64 = base64.b64encode(buf.getvalue()).decode()
    names = candidate_names(context)
    text = BASE + (f" Context terms: {', '.join(names)}." if names else "")
    r = client.chat.completions.create(
        model="nemotron", temperature=0, max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}]}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    out = r.choices[0].message.content or ""
    return collapse_loops(trim_runaway(out, len(audio) / SR))


def score(ps):
    m = {"n": len(ps), "wer": corpus_wer((p["text"], p["hyp"]) for p in ps)}
    sl = [(p["slots"], p["hyp"]) for p in ps if p["slots"]]
    if sl:
        m["slot_accuracy"] = slot_accuracy(sl)["all"]
        m["name_cer"] = name_cer(sl)
    if any(p["decoys"] for p in ps):
        m["decoy_copy_rate"] = decoy_copy_rate((p["decoys"], p["text"], p["hyp"]) for p in ps)
    return m


data = snapshot_download(os.environ["DATA_REPO"], repo_type="dataset", max_workers=32,
                         allow_patterns=os.environ["DATA_PATTERNS"].split(","))
results, preds = {}, []
for spec in os.environ["SPECS"].split():
    spec = spec.replace("{DATA}", data)
    key = spec.replace(os.path.join(data, ""), "")
    ds = load_source(spec)
    rows = [ds[i] for i in range(len(ds))]
    with ThreadPoolExecutor(32) as pool:
        hyps = list(pool.map(lambda r: transcribe(load_audio(r["audio"]), r["context"]), rows))
    ps = [{"spec": key, "source": r["source"], "text": r["text"], "hyp": h, "slots": json.loads(r["slots"]),
           "context": r["context"], "decoys": json.loads(r["decoys"])} for r, h in zip(rows, hyps)]
    results[key] = score(ps)
    preds += ps
    print("RESULT", key, json.dumps(results[key]), flush=True)

# A few real recordings (no ground truth): side by side with WhisperX
real = []
if os.environ.get("REAL_REPO"):
    import glob
    rdir = snapshot_download(os.environ["REAL_REPO"], repo_type="dataset", allow_patterns=["audio/*"])
    import soundfile as sf
    from scipy.signal import resample_poly
    for path in sorted(glob.glob(f"{rdir}/audio/*/*.mp3"))[:int(os.environ.get("REAL_N", "12"))]:
        try:
            x, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            print("REAL SKIP", path, e); continue
        x = x.mean(1)[: sr * 120]
        g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        real.append({"file": path.split("/audio/")[1], "nemotron": transcribe(x, "")})
        print("REAL", real[-1]["file"], real[-1]["nemotron"][:300], flush=True)

name = os.environ.get("NAME", "nemotron")
open("/tmp/r.json", "w").write(json.dumps({"results": results, "real": real}, indent=1))
with open("/tmp/p.jsonl", "w") as f:
    for p in preds:
        f.write(json.dumps(p) + "\n")
for local, remote in [("/tmp/r.json", f"diagnostics/{name}.json"), ("/tmp/p.jsonl", f"diagnostics/{name}_predictions.jsonl")]:
    upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=os.environ["OUT_REPO"], repo_type="dataset",
                commit_message=f"Nemotron-3.5-Transcribe eval: {name}")
print("UPLOADED", flush=True)
