# Eval v2.2 grading conventions

Each item: a design system, a design (DSL, one node per line, `type#id "name" props`), a request, criteria and a
reference patch. A model's answer passes when it applies, stays in the design language and meets every criterion.
Criteria are judged by an LLM; checks are exact, machine-run versions of the criteria. Both must accept EVERY
reasonable answer a good product designer would accept, and reject answers that are wrong.

1. **One reading per criterion, alternatives merged.** Never "One acceptable solution ... / Another acceptable
   solution ..." as separate criteria (they get AND-ed). Write one criterion that names the acceptable options.
2. **No arbitrary specifics.** Never require the id, layer name or child index of a node the edit creates. Exact
   strings only when the request quotes them; exact values only when the request, the design system, or a sibling
   to match pins them down. For vague requests ("a bit wider", "cleaner", "stand out more") require the direction and
   a noticeable amount, not a number, and accept any reasonable means.
3. **Labels stored in the layer name.** Some text nodes have no `text=` prop; the name is their label. Renaming such
   a label is correct via `set nX text="..."` OR `set nX name="..."` (or both). Criteria must accept either.
4. **"Next to X" / "beside X".** Accept both an adjacent sibling in X's container and wrapping X and the new node in
   a horizontal row frame, unless the request or the layout clearly rules one out.
5. **Exclusive selection.** Selecting one tab/option/segment deselects the previously selected one. Its state
   swap covers every property that differs between the selected and unselected siblings in the design (fill, color,
   weight, stroke, ...); the siblings, not the design-system prose, define the states.
6. **Design system vs the design.** When the request asks to follow the system, the system's tokens are required.
   When the request asks to match an existing element, that element's values are correct even if off-system. When a
   variant is needed (e.g. a secondary or destructive button), any variant the design system defines is acceptable.
7. **Minimality.** "Nothing unrelated to the request changes" stays as a criterion. Changes a designer would see as
   part of doing the request (a label's color when a button's fill changes; deselecting the old tab) are related.
8. **Drop the item** if the request rests on a false premise (names an element that doesn't exist, contradicts the
   design), is ambiguous enough that good designers would disagree, or its reference is a poor edit and no fair
   criteria can be written that a good answer would meet. Don't drop just because the item is hard.
9. **The reference must pass** your criteria and checks. If the reference is wrong in a detail but the item is
   good, you may still keep the item only if the reference passes fair criteria; otherwise drop.
