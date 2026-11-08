import random

import pytest

from design_edit import edits, generator, rewards, schema


@pytest.fixture(params=range(25))
def doc(request):
    return generator.generate_design(request.param, max_sections=3)


def test_serialize_roundtrip(doc):
    assert schema.docs_equal(schema.parse(schema.serialize(doc)), doc)


def test_generator_is_deterministic():
    assert schema.serialize(generator.generate_design(7)) == schema.serialize(generator.generate_design(7))


@pytest.mark.parametrize("kind", sorted(edits.ALL_EDITS))
def test_reference_patch_reproduces_target_and_scores_max(doc, kind):
    ex = edits.ALL_EDITS[kind](doc, random.Random(0))
    if ex is None:
        pytest.skip("edit not applicable to this design")
    patch = schema.format_patch(schema.diff(ex.before, ex.after))
    assert schema.docs_equal(schema.apply_patch(ex.before, patch), ex.after)
    s = rewards.score(ex.before, ex.after, patch, ex.relax)
    assert s.valid and s.exact and s.new_lint == 0


def test_diff_handles_moves_inserts_deletes():
    a = schema.parse('frame#n1 "A" layout=col\n  text#n2 "x" text="x" size=14\n  frame#n3 "B" layout=row\n    rect#n4 "r" w=1')
    b = schema.parse('frame#n1 "A" layout=row\n  frame#n3 "B" layout=row\n    text#n2 "x" text="y" size=14\n  rect#n5 "new" w=2')
    patch = schema.format_patch(schema.diff(a, b))
    assert schema.docs_equal(schema.apply_patch(a, patch), b)


def test_quoted_values_with_spaces_and_equals():
    doc = schema.parse('frame#n1 "Root"\n  text#n2 "Label" text="a = b \\"c\\""')
    out = schema.apply_patch(doc, 'set n2 text="hello world = 1"')
    assert schema.find(out, "n2")["props"]["text"] == "hello world = 1"


@pytest.mark.parametrize("bad", ["frobnicate n1", "set n99 fill=#000", "delete n1", "move n2 n2 0",
                                 "insert n2 0 text#n9 text=\"x\"", "insert n1 9 rect#n9", "set n2"])
def test_invalid_patches_raise(bad):
    doc = schema.parse('frame#n1 "Root" layout=col\n  text#n2 "T" text="hi" size=14')
    with pytest.raises(schema.PatchError):
        schema.apply_patch(doc, bad)


def test_reward_ordering():
    before = schema.parse('frame#n1 "R" layout=col gap=8\n  text#n2 "T" text="a" size=14 color=#111827')
    after = schema.parse('frame#n1 "R" layout=col gap=16\n  text#n2 "T" text="a" size=20 color=#111827')
    gold = "set n1 gap=16\nset n2 size=20"
    full = rewards.score(before, after, gold).reward
    partial = rewards.score(before, after, "set n1 gap=16").reward
    wrong = rewards.score(before, after, "set n2 color=#FF0000").reward
    invalid = rewards.score(before, after, "not a patch").reward
    assert full > partial > wrong > invalid == -1.0


def test_relaxed_props_accept_any_value_in_right_direction():
    doc = schema.parse('frame#n1 "R" layout=col\n  text#n2 "Title" text="t" size=20')
    after = schema.parse('frame#n1 "R" layout=col\n  text#n2 "Title" text="t" size=24')
    relax = [{"id": "n2", "prop": "size", "cmp": "gt"}]
    assert rewards.score(doc, after, "set n2 size=30", relax).exact
    assert not rewards.score(doc, after, "set n2 size=16", relax).exact
    assert not rewards.score(doc, after, "set n2 size=30").exact


def test_inserted_nodes_match_regardless_of_chosen_id():
    doc = schema.parse('frame#n1 "R" layout=col\n  text#n2 "T" text="a" size=14')
    after = schema.parse('frame#n1 "R" layout=col\n  text#n2 "T" text="a" size=14\n  text#n3 "S" text="b" size=14')
    assert rewards.score(doc, after, 'insert n1 1 text#n42 "S" text="b" size=14').exact


def test_lint_flags_low_contrast_and_off_scale():
    doc = schema.parse('frame#n1 "R" layout=col gap=7 fill=#FFFFFF\n  text#n2 "T" text="a" size=15 color=#F3F4F6')
    issues = rewards.lint(doc)
    assert {"contrast:n2", "type_scale:n2", "spacing:n1:gap"} <= set(issues)


def test_edits_never_target_ambiguous_labels():
    for seed in range(300):
        doc = generator.generate_design(seed, max_sections=3)
        for n, _ in edits.nodes(doc, lambda n, p: True):
            same = [m for m, _ in schema.walk(doc) if edits.label(m) == edits.label(n)]
            assert len(same) == 1


def test_goal_alternative_answers_score_exact():
    n = 0
    for seed in range(300):
        doc = generator.generate_design(seed, max_sections=3)
        for fn in edits.GOAL.values():
            ex = fn(doc, random.Random(seed))
            if ex is None:
                continue
            for alt, relax in ex.alts:
                patch = schema.format_patch(schema.diff(ex.before, alt))
                alts = [{"after": schema.serialize(alt), "relax": relax}]
                assert rewards.score(ex.before, ex.after, patch, ex.relax, alts).exact
                n += 1
    assert n > 50


def test_disabled_accepts_grey_fill_convention():
    for seed in range(100):
        doc = generator.generate_design(seed)
        ex = edits.disable_button(doc, random.Random(seed))
        if ex and ex.before is not None and schema.find(ex.before, ex.after["id"]):
            btn_id = next(k for k in schema.index_nodes(ex.after)
                          if schema.find(ex.after, k)["props"].get("opacity") == 0.4)
            lab_id = schema.find(ex.before, btn_id)["children"][0]["id"]
            alts = [{"after": schema.serialize(a), "relax": r} for a, r in ex.alts]
            s = rewards.score(ex.before, ex.after, f"set {btn_id} fill=#D1D5DB\nset {lab_id} color=#6B7280",
                              ex.relax, alts)
            assert s.exact
            return
    raise AssertionError("no disable_button example generated")


def test_patch_extraction_from_channel_format():
    doc = schema.parse('frame#n1 "R" layout=col\n  text#n2 "T" text="a" size=14')
    for raw in [" to=user<|message|>set n2 size=20<|eot|>",
                " to=self<|message|>bigger title<|eot|><|start|>assistant to=user<|message|>set n2 size=20",
                " to=user set n2 size=20",
                "```patch\nset n2 size=20\n```"]:
        assert schema.find(schema.apply_patch(doc, raw), "n2")["props"]["size"] == 20, raw
