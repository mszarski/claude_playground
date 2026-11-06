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
    doc = generator.generate_design(3)
    ex = edits.compose(doc, random.Random(1))
    gold = schema.format_patch(schema.diff(ex.before, ex.after))
    assert len(gold.splitlines()) >= 2
    first_line = gold.splitlines()[0]
    full = rewards.score(ex.before, ex.after, gold).reward
    partial = rewards.score(ex.before, ex.after, first_line).reward
    wrong = rewards.score(ex.before, ex.after, "set n1 fill=#000000").reward
    invalid = rewards.score(ex.before, ex.after, "not a patch").reward
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
