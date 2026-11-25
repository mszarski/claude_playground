# Eval v3: what a hard item looks like

Eval v2.2 is nearly saturated: the best models pass about 96%, mostly because its items are single, local edits on
small screens (15-45 nodes). Eval v3 tests the same skill — editing a design in the design language from a
natural request, following the design system — under realistic difficulty. Grading follows CONVENTIONS.md exactly:
hard must never mean ambiguous, arbitrary or unfair. A good designer reading the request must agree on what counts
as correct.

## Screens
Larger and denser than v2: 45-110 nodes, real hierarchy (sections, repeated components such as cards, rows, list
items, form fields, tabs), several components in different states, and 3-6 deliberate, realistic imperfections
(off-scale sizes or spacing, an inconsistent color, a component that drifted from its siblings). Use only the DSL
properties (name, text, layout, gap, pad, align, justify, size, weight, color, fill, stroke, radius, opacity, w, h),
6-digit hex colors, ids n1..nN in document order, meaningful layer names. Buttons are frames containing a text
label. Follow the item's design system (tool.py ds <seed>).

## Item kinds (each item must need reasoning or several coordinated changes; no one-property edits)
- **goal**: a designer-level goal on a busy area ("this pricing section is hard to scan", "the primary action gets
  lost", "tighten this form") that needs 2+ coordinated changes; several solutions acceptable, token edits fail.
- **structure**: add, remove, regroup, reorder or re-nest elements where layout must stay right (wrap fields into a
  two-column row, add a section that follows the existing section pattern, move an action into a toolbar, split a
  long list into groups with headings).
- **multi_step**: one request with 2-4 distinct changes in different places ("rename X, make Y destructive and
  move Z below W"); every part graded.
- **propagate**: a change whose correct execution must reach every relevant element or its implied consequences:
  all instances of a pattern, every invalid field, exclusive selection, a renamed concept everywhere it appears on
  screen, keeping siblings consistent.
- **system_audit**: bring a named region or the whole screen back to the design system; several deviations to find
  (and in-system values to leave alone).
- **pattern_copy**: create a new element by following an existing complex pattern (a new card, table row, step,
  settings row) with new content, matching structure and styles exactly where the pattern is fixed.

## Each item needs
- request: natural, as a user would type it; no ids; hex only if a real user would say it.
- reference_patch: a correct answer (minimal for what the request needs).
- criteria: 2-7 statements, all must hold, following CONVENTIONS (merged alternatives, no new-node ids, ...),
  last one "Nothing unrelated to the request changes."
- checks + allowed: exact checks in the check language (scripts/compile_checks.py SPEC, plus {"any": [...]});
  {"judge": "..."} only for genuinely subjective parts. "allowed" covers every node a valid answer may touch.
- alternatives: at least one different valid answer (the gate checks it passes).
- traps: 1-3 plausible wrong answers a capable model might give (does only part of it, misses an instance,
  over-edits, breaks the system, copies the pattern with a wrong style), each with why_wrong and caught_by.
- notes: what makes it hard.
