"""Tests for transcript normalization and metrics."""

import pytest

from marine_vhf_asr.metrics import corpus_wer, edit_distance, slot_accuracy
from marine_vhf_asr.normalize import normalize_transcript, spell_digits, spell_letters


class TestNormalize:
    def test_channel_digits(self):
        assert normalize_transcript("Switch to channel 16, over.") == \
            "SWITCH TO CHANNEL ONE SIX OVER"

    def test_attached_digits(self):
        assert normalize_transcript("Ch16") == "CH ONE SIX"

    def test_letters_expanded(self):
        assert normalize_transcript("call sign P D 2") == "CALL SIGN PAPA DELTA TWO"

    def test_a_and_i_kept(self):
        assert normalize_transcript("I have a man overboard") == "I HAVE A MAN OVERBOARD"

    def test_urgency_markers(self):
        assert normalize_transcript("Pan-Pan, Pan-Pan") == "PAN PAN PAN PAN"
        assert normalize_transcript("Sécurité") == "SECURITE"
        assert normalize_transcript("securitay") == "SECURITE"

    def test_spelling_variants(self):
        assert normalize_transcript("Alpha Juliet Whisky niner") == "ALFA JULIETT WHISKEY NINE"

    def test_coordinates(self):
        assert normalize_transcript("50°12.5'N") == \
            "FIVE ZERO DEGREES ONE TWO POINT FIVE MINUTES NORTH"
        assert normalize_transcript("004° 03' W") == \
            "ZERO ZERO FOUR DEGREES ZERO THREE MINUTES WEST"
        # A lone letter that is not part of a coordinate is still spelled
        assert normalize_transcript("berth N") == "BERTH NOVEMBER"

    def test_idempotent(self):
        text = "MAYDAY RED KITE POSITION FIVE ONE DEGREES"
        assert normalize_transcript(normalize_transcript(text)) == text

    def test_contractions(self):
        assert normalize_transcript("we're 'inbound'") == "WE'RE INBOUND"

    def test_spell_helpers(self):
        assert spell_digits("235") == "TWO THREE FIVE"
        assert spell_letters("pd2") == "PAPA DELTA TWO"


class TestMetrics:
    def test_edit_distance(self):
        assert edit_distance("A B C".split(), "A X C".split()) == 1
        assert edit_distance("A B C".split(), "A C".split()) == 1
        assert edit_distance([], "A B".split()) == 2

    def test_corpus_wer(self):
        pairs = [("channel 16", "CHANNEL ONE SEVEN"), ("over", "over")]
        # refs: CHANNEL ONE SIX (3) + OVER (1) = 4 words, 1 error
        assert corpus_wer(pairs) == pytest.approx(0.25)

    def test_slot_accuracy(self):
        examples = [
            ({"channel": "one six", "vessel": "Red Kite"}, "red kite switch to channel one six"),
            ({"channel": "one six"}, "switch to channel one seven"),
        ]
        acc = slot_accuracy(examples)
        assert acc["channel"] == 0.5
        assert acc["vessel"] == 1.0
        assert acc["all"] == pytest.approx(2 / 3)

    def test_slot_requires_whole_words(self):
        # "ON" is a prefix of "ONE" but not a word in the hypothesis
        assert slot_accuracy([({"x": "on"}, "one six")])["x"] == 0.0


class TestNormalizeOptions:
    def test_cardinal_numbers(self):
        assert normalize_transcript("MERLION 31, 8 knots, channel 16", numbers="cardinal") == \
            "MERLION THIRTY ONE EIGHT KNOTS CHANNEL SIXTEEN"

    def test_cardinal_keeps_leading_zero_digitwise(self):
        assert normalize_transcript("heading 045", numbers="cardinal") == "HEADING ZERO FOUR FIVE"

    def test_cardinal_large(self):
        assert normalize_transcript("105 and 2500", numbers="cardinal") == \
            "ONE HUNDRED FIVE AND TWO THOUSAND FIVE HUNDRED"

    def test_decimals_always_digitwise(self):
        assert normalize_transcript("12.75", numbers="cardinal") == "TWELVE POINT SEVEN FIVE"

    def test_bad_numbers_option(self):
        with pytest.raises(ValueError):
            normalize_transcript("1", numbers="roman")

    def test_acronyms(self):
        assert normalize_transcript("M V - AZAMARA, M M S I number, E T A", acronyms=True) == \
            "MV AZAMARA MMSI NUMBER ETA"

    def test_acronyms_leave_words_alone(self):
        assert normalize_transcript("I have a man overboard", acronyms=True) == \
            "I HAVE A MAN OVERBOARD"


class TestDecoyCopyRate:
    def test_counts_unspoken_candidates(self):
        from marine_vhf_asr.metrics import decoy_copy_rate
        examples = [
            (["PAN AMBER"], "VTS SHAN BANG OVER", "VTS PAN AMBER OVER"),     # copied a decoy
            (["PAN AMBER"], "VTS SHAN BANG OVER", "VTS SHAN BANG OVER"),     # clean
            (["PAN AMBER"], "PAN AMBER THIS IS VTS", "PAN AMBER THIS IS VTS"),  # actually spoken
            ([], "X", "Y"),                                                  # no decoys: ignored
        ]
        assert decoy_copy_rate(examples) == pytest.approx(1 / 3)



class TestNameCer:
    def test_exact_spacing_insensitive_and_near_miss(self):
        from marine_vhf_asr.metrics import name_cer
        assert name_cer([({"v": "H T M S TAKSIN"}, "VTS HTMS TAKSIN GO AHEAD")]) == 0.0
        near = name_cer([({"v": "CHARIKLIA JUNIOR"}, "VTS TO TREKLIER JUNIOR REDUCE SPEED")])
        assert 0.2 < near < 0.5
        assert name_cer([({"v": "SEA LION"}, "ROGER OVER")]) > 0.7
        assert name_cer([]) == 0.0
