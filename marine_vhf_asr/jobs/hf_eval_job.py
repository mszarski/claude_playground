# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0",
#     "datasets==5.1.0", "huggingface_hub==2.2.0", "mistral-common[audio]==1.12.0",
#     "soundfile==0.14.0", "scipy==1.18.1", "numpy",
# ]
# ///
"""
Evaluate an adapter on source specs on HF Jobs; upload metrics + predictions to
the adapter repo under diagnostics/<NAME>*.

    hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN \\
        --env CODE_REPO=<user>/marine-vhf-asr-code --env ADAPTER=<user>/voxtral-3B-marine-vhf-v3 \\
        --env NAME=nyc --env VERIFY=1 --env DATA_REPO=<user>/marine-vhf-nyc-synth \\
        --env "SPECS={DATA}/test/manifest.jsonl#none {DATA}/test/manifest.jsonl#moving_a" \\
        marine_vhf_asr/jobs/hf_eval_job.py
"""
import json, os, sys
from huggingface_hub import snapshot_download, upload_file
sys.path.insert(0, snapshot_download(os.environ["CODE_REPO"]))
from marine_vhf_asr import evaluate
from marine_vhf_asr.sources import load_source

adapter, name, specs = os.environ["ADAPTER"], os.environ["NAME"], os.environ["SPECS"].split()
verify = os.environ.get("VERIFY") == "1"
if os.environ.get("DATA_REPO"):
    # Specs may refer to a Hub dataset folder as {DATA}, e.g. {DATA}/test/manifest.jsonl#all
    data = snapshot_download(os.environ["DATA_REPO"], repo_type="dataset",
                             allow_patterns=os.environ.get("DATA_PATTERNS", "*").split(","))
    specs = [s.replace("{DATA}", data) for s in specs]
model, processor = evaluate.load_model(adapter=adapter)
results, preds = {}, []
for spec in specs:
    print("Evaluating", spec, flush=True)
    m, p = evaluate.evaluate_dataset(model, processor, evaluate.BASE_MODEL, load_source(spec), verify=verify)
    key = spec.replace(os.path.join(data, ""), "") if os.environ.get("DATA_REPO") else spec
    results.update({key: v for v in m.values()})
    preds += [dict(x, spec=key) for x in p]
    print("RESULT", key, json.dumps(results[key]), flush=True)
open("/tmp/r.json", "w").write(json.dumps(results, indent=2))
with open("/tmp/p.jsonl", "w") as f:
    for x in preds:
        f.write(json.dumps(x) + "\n")
for local, remote in [("/tmp/r.json", f"diagnostics/{name}.json"), ("/tmp/p.jsonl", f"diagnostics/{name}_predictions.jsonl")]:
    upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=adapter, commit_message=f"Eval: {name}")
print("UPLOADED")
