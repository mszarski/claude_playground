import io
import json
import urllib.error

import pytest

from rmr.planner import llm, slug
from rmr.planner.evaluate import held_out_prompts
from rmr.planner.probes import PROBES, score
from rmr.planner.prompt import COMPACT, EXAMPLES, SYSTEM
from rmr.planner.write import _batch, write_recipes
from rmr.recipe import check
from rmr import library


def test_prompt_examples_are_valid_recipes():
    assert all(check(r) is None for r in EXAMPLES.values())
    assert "Respond with JSON only." in SYSTEM and COMPACT.endswith('"recipe": "<recipe>"}')


def test_parse_json_skips_think_and_fences():
    text = '<think>maybe {"no": 1}</think>\n```json\n{"idea": "x", "recipe": "hold 1"}\n```'
    assert llm.parse_json(text) == {"idea": "x", "recipe": "hold 1"}
    with pytest.raises(ValueError):
        llm.parse_json("no json here")


def _reply(content):
    return {"choices": [{"message": {"content": content}}]}


def test_chat_json_falls_back_without_structured_outputs():
    calls = []

    def post(kw):
        calls.append(kw)
        if "response_format" in kw:
            raise urllib.error.HTTPError("u", 400, "unsupported", {}, io.BytesIO(b""))
        return _reply('{"motions": []}')

    assert llm.chat_json([], {}, "x", model="m", post=post) == {"motions": []}
    assert "response_format" in calls[0] and "response_format" not in calls[1]


def test_batch_sends_errors_back_and_keeps_fixed_recipes():
    prompts = ["a. one.", "b. two."]
    answers = [
        {"motions": [{"prompt": "a. one.", "idea": "", "recipe": "go 1 p=99"},     # out of range
                     {"prompt": "b. two.", "idea": "", "recipe": "hold 1"}]},
        {"motions": [{"prompt": "a. one.", "idea": "", "recipe": "go 1 p=20"}]},
    ]
    seen = []

    def chat(msgs, schema, name, model=None):
        seen.append(msgs[1]["content"])
        return answers[len(seen) - 1]

    got, errors = _batch(prompts, None, chat=chat)
    assert {p: v["recipe"] for p, v in got.items()} == {"a. one.": "go 1 p=20", "b. two.": "hold 1"}
    assert errors == {} and len(seen) == 2
    assert "p=99 outside" in seen[1] and "b. two." not in seen[1]


def test_write_recipes_is_resumable(tmp_path):
    out = tmp_path / "r.json"
    out.write_text(json.dumps({"a. one.": "hold 1"}))
    asked = []

    def chat(msgs, schema, name, model=None):
        asked.append(msgs[1]["content"])
        return {"motions": [{"prompt": "b. two.", "idea": "", "recipe": "hold 2"}]}

    r = write_recipes(["a. one.", "b. two."], str(out), chat=chat, log=lambda s: None)
    assert r == {"a. one.": "hold 1", "b. two.": "hold 2"} == json.loads(out.read_text())
    assert len(asked) == 1 and "a. one." not in asked[0]


def test_probes_pass_good_and_fail_bad_recipes():
    sneeze, startle = PROBES[0][0], PROBES[2][0]
    good_sneeze = "go .6 e=40 p=-6 z=4 E=1 | go 1 p=-16 z=10 e=70 E=3 | hold .6 E=6 | go .12 p=18 z=-8 e=100 E=10 | hold .6"
    up_sneeze = "go .6 e=40 p=6 E=1 | go .12 p=-18 z=8 E=10 | hold .6"
    good_startle = "go .5 e=15 E=.5 | go .15 e=-15 p=-10 z=16 E=10 | hold 1 E=1"
    _, per = score({sneeze: [good_sneeze, up_sneeze, None], startle: [good_startle]})
    assert per[sneeze] == pytest.approx(1 / 3) and per[startle] == 1.0


def test_held_out_prompts_cover_every_held_out_emotion():
    d = held_out_prompts()
    assert list(d) == library.HELD_OUT_EMOTIONS
    assert all(p.split(".")[0] == h.rstrip("0123456789") for h, p in d.items())
    assert slug("A cat stalking prey. You crouch.") == "a_cat_stalking_prey"
