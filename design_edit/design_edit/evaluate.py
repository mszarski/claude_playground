"""Model-agnostic evaluation: give it rows and a ``generate(list_of_prompts) -> list_of_texts`` function."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

from .rewards import score

GenerateFn = Callable[[list[list[dict]]], list[str]]

METRICS = ("valid", "exact", "f1", "precision", "recall", "new_lint", "reward")


def evaluate(rows: Iterable[dict], generate: GenerateFn, batch_size: int = 8) -> dict:
    rows = list(rows)
    per_row: list[dict] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        outs = generate([r["prompt"] for r in batch])
        for r, out in zip(batch, outs):
            s = score(r["before"], r["after"], out, r["relax"], r.get("alts"))
            per_row.append({"kind": r["kind"], "output": out, **{m: float(getattr(s, m)) for m in METRICS},
                            "error": s.error})
    return summarize(per_row) | {"rows": per_row}


def summarize(per_row: list[dict]) -> dict:
    def agg(items: list[dict]) -> dict:
        return {m: round(sum(x[m] for x in items) / len(items), 4) for m in METRICS} | {"n": len(items)}

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by_kind[r["kind"]].append(r)
    return {"overall": agg(per_row), "by_kind": {k: agg(v) for k, v in sorted(by_kind.items())}}


def format_report(summary: dict) -> str:
    lines = [f"{'kind':<26}{'n':>5}{'valid':>8}{'exact':>8}{'f1':>8}{'lint+':>8}"]
    for k, v in [("OVERALL", summary["overall"]), *summary["by_kind"].items()]:
        lines.append(f"{k:<26}{v['n']:>5}{v['valid']:>8.3f}{v['exact']:>8.3f}{v['f1']:>8.3f}{v['new_lint']:>8.3f}")
    return "\n".join(lines)
