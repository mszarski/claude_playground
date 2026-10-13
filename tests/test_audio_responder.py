import json

from rmr.audio_responder import AudioStudent


def test_feeling_is_announced_before_the_recipe_is_complete():
    answer = json.dumps({"feeling": "sad", "recipe": "go 1 e=90 p=8 E=1 | hold 1", "response": "gentle. You lower "
                         "your ears.", "reading": "They sound low."})
    st = AudioStudent.__new__(AudioStudent)
    seen = []

    def stream(heard, wav):
        for i in range(0, len(answer), 5):          # token-ish chunks
            seen.append(i)
            yield answer[i:i + 5]
    st.stream = stream
    out = []
    for kind, x in st.stages({"text": "hi"}, None):
        out.append((kind, len(seen), x if kind == "feeling" else x["recipe"]))
    assert [k for k, _, _ in out] == ["feeling", "early", "final"]
    assert out[0][2] == "sad" and out[0][1] < out[1][1] < out[2][1]     # feeling first, while the rest still streams
    assert list(st.answers({"text": "hi"}, None))[-1][1]["reading"] == "They sound low."
