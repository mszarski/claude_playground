"""Tests for the synthetic dialogue generator."""

import pytest

from marine_vhf_asr.gazetteer import PORTS
from marine_vhf_asr.normalize import normalize_transcript
from marine_vhf_asr.smcp import DialogueGenerator, digits, number_words


def test_number_words():
    assert number_words(7) == "seven"
    assert number_words(16) == "sixteen"
    assert number_words(40) == "forty"
    assert number_words(73) == "seventy three"


def test_digits():
    assert digits("074") == "zero seven four"


def test_deterministic_with_seed():
    a = [t.text for t in DialogueGenerator(seed=1).generate().turns]
    b = [t.text for t in DialogueGenerator(seed=1).generate().turns]
    assert a == b


@pytest.mark.parametrize("scenario", list(DialogueGenerator().scenarios))
def test_every_scenario_well_formed(scenario):
    gen = DialogueGenerator(seed=0)
    for port in PORTS:
        d = gen.generate(scenario, port)
        assert d.scenario == scenario and d.port == port.name
        assert len(d.turns) >= 2
        for turn in d.turns:
            target = turn.target
            assert target and target == target.upper()
            assert not any(c.isdigit() for c in target)
            # Every slot must be recoverable verbatim from the reference
            for name, value in turn.slots.items():
                assert f" {normalize_transcript(value)} " in f" {target} ", (name, value, target)


def test_distress_calls_carry_markers():
    gen = DialogueGenerator(seed=3)
    assert gen.generate("mayday").turns[0].target.startswith("MAYDAY MAYDAY MAYDAY")
    assert gen.generate("pan_pan").turns[0].target.startswith("PAN PAN PAN PAN PAN PAN")
    assert gen.generate("securite").turns[0].target.startswith("SECURITE SECURITE SECURITE")
