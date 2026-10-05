import json
from pathlib import Path

import numpy as np
import pytest

from rmr.motion import FPS
from rmr.plan import CH
from rmr.recipe import NEUTRAL, RecipeError, check, expand, to_plan, variants

TEACHER = Path(__file__).parent.parent / "data" / "teacher"
SNEEZE = "go .6 e=40 p=-6 z=4 E=1 | go 1 p=-16 z=10 e=70 E=3 | hold .6 E=6 | go .12 p=18 z=-8 e=100 E=10 | go 1 p=2 z=2 e=30 E=1 | hold .6"


def test_expand_starts_neutral_and_reaches_targets():
    F = expand("go 1 p=10 z=-5 E=4 | hold .5")
    assert F.shape[1] == 8
    np.testing.assert_allclose(F[0], NEUTRAL)
    assert F[-1, 2] == pytest.approx(10)
    assert F[-1, 5] == pytest.approx(-5)
    assert F[-1, 7] == pytest.approx(4)


def test_expand_duration_close_to_written():
    F = expand("go 1 p=5 | hold 1 | osc 2 r 5 .8")
    assert 4 * 0.85 <= len(F) / FPS <= 4 * 1.15 + 1 / FPS


def test_hold_only_changes_energy():
    F = expand("go .5 p=10 | hold 1 p=-20 E=6")
    assert F[-1, 2] == pytest.approx(10)
    assert F[-1, 7] == pytest.approx(6)


def test_osc_on_both_ears_returns_to_pose():
    F = expand("go .5 e=40 | osc 2 e 20 .8")
    assert np.ptp(F[-40:, 0]) > 10
    assert F[-1, 0] == pytest.approx(40, abs=0.15 * 20 * 1.2)   # fade-out leaves <= 1/(0.3*FPS) of amp


def test_sneeze_release_moves_head_down():
    F = expand(SNEEZE)
    i = int(np.argmax(F[:, 7]))          # energy peaks at the "choo"
    assert F[i, 2] > 10                  # +pitch = head lowered


@pytest.mark.parametrize("bad", [
    "", "jump 1", "go x p=1", "go 20 p=1", "go 1 p=50", "go 1 q=3", "go 1 p", "osc 1 p 5",
    "osc 1 p 5 .1", "osc 1 E 5 .8", "go 10 p=1 | go 10 p=2 | go 10 p=3 | go 10 p=4",
])
def test_invalid_recipes_rejected(bad):
    with pytest.raises(RecipeError):
        expand(bad)
    assert check(bad)


def test_to_plan_keys_and_mirror():
    F = expand("go 1 r=15 y=20 b=10 eR=100 eL=20 | hold 1")
    plan = to_plan(F, np.random.default_rng(1), ear_jitter=0)
    mirrored = to_plan(F, np.random.default_rng(1), ear_jitter=0, mirror=True)
    assert set(plan["keys"][0]) == {"t", *CH}
    assert plan["keys"][1]["t"] == 0.5
    last, mlast = plan["keys"][-1], mirrored["keys"][-1]
    assert mlast["roll"] == pytest.approx(-last["roll"])
    assert mlast["earR"] == pytest.approx(last["earL"])


def test_variants_are_deterministic_and_differ():
    a, b = variants(SNEEZE, 2, seed=3), variants(SNEEZE, 2, seed=3)
    assert a == b
    assert a[0] != a[1]


def _teacher_recipes():
    rows = [json.loads(l) for l in open(TEACHER / "dataset.jsonl")]
    rows += [lab for r in map(json.loads, open(TEACHER / "val.jsonl")) for lab in r["labels"]]
    return [r["recipe"] for r in rows] + list(json.load(open(TEACHER / "recipes.json")).values())


def test_every_teacher_recipe_is_valid():
    bad = [(r, err) for r in _teacher_recipes() if (err := check(r))]
    assert not bad, bad[:5]
