#!/usr/bin/env python3
"""Turn approved eval-v3 items into dataset rows in the eval-v2 format (grade_eval_v2.py / eval_model.py work unchanged).

    python eval_v3/build_eval_v3.py --work <dir with screens/ approved/> --out eval_v3.jsonl [--push user/design-edits-eval-v3]
Split: whole screens go to dev or test, balanced greedily, so a screen never appears in both.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from design_edit.design_system import render_design_system, sample_design_system, user_message_ds  # noqa: E402
from design_edit.prompts import SYSTEM_PROMPT  # noqa: E402
from design_edit.schema import apply_patch, parse, serialize, strip_fences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default=None)
    a = ap.parse_args()
    W = Path(a.work)
    rows = []
    for f in sorted((W / "approved").glob("*.json")):
        p = json.loads(f.read_text())
        meta = json.loads((W / "screens" / f"{p['screen_id']}.json").read_text())
        doc = parse((W / "screens" / f"{p['screen_id']}.dsl").read_text().strip())
        ds = render_design_system(sample_design_system(int(meta["ds_seed"])))
        patch = strip_fences(p["reference_patch"]).strip()
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": user_message_ds(serialize(doc), ds, p["request"])}],
            "completion": [{"role": "assistant", "content": patch}],
            "before": serialize(doc), "after": serialize(apply_patch(doc, patch)), "kind": f"v3_{p['kind']}",
            "instruction": p["request"], "orig_instruction": p["request"], "relax": "[]", "alts": "[]", "refs": "[]",
            "criteria": json.dumps(p["criteria"]), "design_system": ds, "screen": meta["screen"],
            "screen_id": p["screen_id"], "item_id": f.stem, "checks": json.dumps(p["checks"]),
            "allowed": json.dumps(p["allowed"]), "alternatives": json.dumps(p.get("alternatives", [])),
            "traps": json.dumps(p.get("traps", [])), "notes": p.get("notes", ""), "review": p.get("review", ""),
        })
    cnt = Counter(r["screen_id"] for r in rows)
    side, asg = {"dev": 0, "test": 0}, {}
    for s in sorted(cnt, key=lambda s: (-cnt[s], hashlib.sha1(s.encode()).hexdigest())):
        t = "dev" if side["dev"] < side["test"] else "test"
        asg[s] = t
        side[t] += cnt[s]
    for r in rows:
        r["split"] = asg[r["screen_id"]]
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "items", side, Counter(r["kind"] for r in rows))
    if a.push:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi
        DatasetDict({"test": Dataset.from_list(rows)}).push_to_hub(a.push, private=True)
        root = Path(__file__).resolve().parents[1]
        for d in ("design_edit", "scripts", "eval_v3"):
            HfApi().upload_folder(repo_id=a.push, repo_type="dataset", folder_path=str(root / d), path_in_repo=f"code/{d}",
                                  ignore_patterns=["__pycache__/*", "*.pyc"])
        print("pushed", a.push)


if __name__ == "__main__":
    main()
