"""Speech-to-text candidates for the hearing stage: word error rate and speed on clips with known words.

  python scripts/asr_bench.py --tone eval/tone.jsonl --lines eval/lines.jsonl --voiced audio/voiced/eval_lines \
      --out /work/out/asr.json                                          # GPU job

Today Whisper-small transcribes each turn after it ends (0.3-1.3 s on the live L4). Candidates:
- ``openai/whisper-small`` (MIT): the current one.
- ``openai/whisper-large-v3-turbo`` (MIT): a larger Whisper with a pruned decoder.
- ``mistralai/Voxtral-Mini-4B-Realtime-2602`` (Apache-2.0): a streaming transcriber built to transcribe while the person
  speaks, finishing ~0.5 s after they stop (its default 480 ms delay). Here it runs offline on the whole clip
  (transformers), which measures its accuracy; streaming latency is its design delay, not measured here.

Clips: the 300 held-out CREMA-D actor clips of the tone test (real voices, flat sentences) and the 255 cloned test
lines (scripts/clone_speech.py; written lines in held-out actors' cloned voices). WER is word-level, lower case,
punctuation stripped. Time is per clip, one at a time, after a warm-up.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audio_llm import crema_audio  # noqa: E402
from clone_speech import wer  # noqa: E402


def whisper(repo):
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=repo, device="cuda",
                   dtype=torch.float16 if "large" in repo else torch.float32)
    return lambda wav: asr({"raw": wav, "sampling_rate": 16000},
                           generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()


def voxtral_realtime(repo):
    import torch
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    proc = AutoProcessor.from_pretrained(repo)
    m = VoxtralRealtimeForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()

    def run(wav):
        inputs = proc(wav, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out = m.generate(**inputs)
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    return run


MODELS = {"openai/whisper-small": whisper, "openai/whisper-large-v3-turbo": whisper,
          "mistralai/Voxtral-Mini-4B-Realtime-2602": voxtral_realtime}


def main():
    from rmr.voice import load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--tone", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--voiced", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tone = [json.loads(x) for x in open(a.tone)]
    crema = crema_audio([r["tone"] for r in tone])
    clips = [("tone", r["text"], crema[r["tone"]]) for r in tone]
    for r in map(json.loads, open(a.lines)):
        p = os.path.join(a.voiced, r["clip"] + ".wav")
        if os.path.exists(p):
            clips.append(("lines", r["text"], load_audio(open(p, "rb").read())))
    print(f"{len(clips)} clips", flush=True)
    res = {}
    for repo in a.models:
        try:
            run = MODELS[repo](repo)
            run(clips[0][2])                                   # warm-up
        except Exception as e:
            print(f"{repo}: could not run: {type(e).__name__}: {e}", flush=True)
            res[repo] = {"error": f"{type(e).__name__}: {e}"}
            continue
        per, times, durs, examples = {"tone": [], "lines": []}, [], [], []
        for i, (kind, text, wav) in enumerate(clips):
            t0 = time.time()
            hyp = run(wav)
            times.append(time.time() - t0)
            durs.append(len(wav) / 16000)
            per[kind].append(wer(text, hyp))
            if i % 100 == 0:
                examples.append([text, hyp])
        times, durs = np.array(times), np.array(durs)
        res[repo] = {"wer": {k: float(np.mean(v)) for k, v in per.items()},
                     "seconds_median": float(np.median(times)), "seconds_p90": float(np.percentile(times, 90)),
                     "seconds_per_audio_second": float(np.median(times / durs)),
                     "seconds_long": float(np.median(times[durs > 5])) if (durs > 5).any() else None,
                     "examples": examples}
        print(repo, json.dumps({k: v for k, v in res[repo].items() if k != "examples"}), flush=True)
        del run
        import torch
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
