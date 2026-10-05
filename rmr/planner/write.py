"""Ask a zero-shot LLM for a recipe per prompt; validate each with ``recipe.check`` and send the invalid ones back
with their errors.

Reference: ``planner/write.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from ..recipe import check
from . import llm
from .prompt import SCHEMA, SYSTEM, user_message


def _batch(prompts, model, max_fix=2, chat=None):
    chat = chat or llm.chat_json
    got, errors = {}, None
    todo = list(prompts)
    for _ in range(max_fix + 1):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(todo, errors)}]
        out = chat(msgs, SCHEMA, "motion_recipes", model=model)
        by_prompt = {m["prompt"].strip(): m for m in out.get("motions", []) if isinstance(m, dict) and "prompt" in m}
        errors = {}
        for p in todo:
            m = by_prompt.get(p.strip())
            if m is None or not isinstance(m.get("recipe"), str):
                errors[p] = "missing from your answer"
                continue
            err = check(m["recipe"])
            if err:
                errors[p] = err
            else:
                got[p] = dict(recipe=m["recipe"], idea=m.get("idea", ""))
        todo = list(errors)
        if not todo:
            break
    return got, errors


def write_recipes(prompts, out_path=None, model=None, batch=8, workers=4, chat=None, log=print):
    """``{prompt: recipe}`` for every prompt the LLM got right within the retries (the format of
    ``examples/demo_recipes.json``, which ``python -m rmr.generator sample --recipes`` reads).

    Resumable: prompts already in ``out_path`` are skipped, and the file is rewritten after each batch.
    """
    have = json.load(open(out_path)) if out_path and os.path.exists(out_path) else {}
    todo = [p for p in prompts if p not in have]
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    failed = {}
    with ThreadPoolExecutor(workers) as ex:
        for got, errs in ex.map(lambda c: _batch(c, model, chat=chat), chunks):
            have.update({p: v["recipe"] for p, v in got.items()})
            failed.update(errs)
            if out_path:
                with open(out_path, "w") as f:
                    json.dump(have, f, indent=1)
            log(f"  {len(have)} recipes written")
    for p, e in failed.items():
        log(f"  FAILED {p[:60]!r}: {e}")
    return have
