export const meta = {
  name: 'eval-v22-repair',
  description: 'Opus agents re-review every eval item: fair criteria, recompiled exact checks, keep/fix/drop',
  phases: [
    { title: 'Repair', detail: '8 Opus agents, ~56 items each, blind to model outputs', model: 'opus' },
    { title: 'Consistency', detail: 'one Opus agent checks decisions are consistent across batches', model: 'opus' },
  ],
}
const R = args.dir
const N = 445, B = 8
const size = Math.ceil(N / B)
const batches = Array.from({ length: B }, (_, k) => [k * size, Math.min(N, (k + 1) * size)])
const SUMMARY = {
  type: 'object',
  properties: {
    saved: { type: 'array', items: { type: 'integer' } },
    unsaved: { type: 'array', items: { type: 'integer' } },
    notes: { type: 'string' },
  },
  required: ['saved', 'unsaved', 'notes'],
}
const common = `You are repairing an evaluation set for a model that edits UI designs. Work directory: ${R}

Read first: ${R}/CONVENTIONS.md (the grading rules; follow them exactly), and the check language in
/home/user/claude_playground/design_edit/scripts/compile_checks.py (the SPEC string; it also supports
{"any": [check, ...]} for alternatives). design_edit/schema.py in that repo defines the DSL and patch language.

For each item id assigned to you:
1. Read ${R}/items/<id>.json: request, design_system, design (before), reference_patch, the current criteria
   (criteria_v21, a repaired version of criteria_v2_original), current_checks/allowed (may be stale or missing).
2. Decide keep (current criteria are fair; you still write fresh checks), fix (write fair criteria), or drop
   (CONVENTIONS rule 8). Think as a careful product designer: what answers would you accept for this request?
   Write down at least one *different* valid answer than the reference when one exists (e.g. text= vs name=,
   another acceptable variant), as a patch, under "alternatives"; your checks must accept it.
3. Write ${R}/proposals/<id>.json = {"verdict", "reason" (one or two sentences), "criteria": [...],
   "checks": [...], "allowed": [...], "alternatives": [...]} and run: python3 ${R}/save_item.py <id>
   It validates that the reference and alternatives pass the checks and that doing nothing fails them. If it
   prints REJECTED, fix the checks (or your verdict) and rerun. "python3 ${R}/save_item.py <id> --try '<patch>'"
   grades any patch. Use {"judge": "..."} checks only for genuinely subjective criteria. "allowed" lists every
   existing node the edit may touch (keep it generous enough for every valid answer).
Do NOT look at any model's outputs or grades (anything outside ${R} and the repo source): this review must be
blind to how models did. Do not modify the repo. Every assigned id must end with a results/<id>.json.`

phase('Repair')
const res = await parallel(batches.map(([a, b], k) => () =>
  agent(`${common}\n\nYour items: ids ${a} to ${b - 1} inclusive (${b - a} items). Return the ids saved, any you could not save, and short notes on recurring problems you saw.`,
    { label: `repair ${a}-${b - 1}`, phase: 'Repair', model: 'opus', effort: 'high', schema: SUMMARY })))
const ok = res.filter(Boolean)
const unsaved = ok.flatMap(r => r.unsaved)
log(`saved ${ok.reduce((s, r) => s + r.saved.length, 0)} items; unsaved: ${unsaved.join(',') || 'none'}; failed batches: ${res.length - ok.length}`)

phase('Consistency')
const notes = ok.map((r, k) => `batch ${k}: ${r.notes}`).join('\n')
const crit = await agent(`${common}\n\nEight reviewers each repaired a slice of the 445 items; their decisions are in ${R}/results/<id>.json. Your job is consistency across slices, not re-doing their work:
1. Find any ids 0..444 with no results file and repair them (reviewers reported unsaved: ${unsaved.join(',') || 'none'}).
2. Group similar items (same kind of request, e.g. relabels, tab selection, "next to" inserts, destructive/secondary buttons, vague goals) and look for the same situation judged differently (one kept, another dropped; one accepts an option another rejects). Fix outliers to the conventions by rewriting proposals/<id>.json and rerunning save_item.py.
3. Spot-check at least 30 random "keep"/"fix" items for criteria that still demand arbitrary specifics, and at least all "drop" items for drops that are not justified by rule 8.
Reviewers' notes:\n${notes}\n\nReturn saved = ids you changed or added, unsaved = ids still missing, notes = what you changed and any convention problems.`,
  { label: 'consistency', phase: 'Consistency', model: 'opus', effort: 'high', schema: SUMMARY })
return { batches: ok, critic: crit }