import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from listen_rating import bradley_terry  # noqa: E402


def test_bradley_terry_recovers_the_order():
    true = {"a": 4.0, "b": 2.0, "c": 1.0, "d": 0.5}
    rng = random.Random(0)
    games = []
    for _ in range(30):
        for x, y in itertools.combinations(true, 2):
            win = x if rng.random() < true[x] / (true[x] + true[y]) else y
            games.append((win, y if win == x else x, 1.0))
    s = bradley_terry(list(true), games)
    assert sorted(s, key=lambda n: -s[n]) == ["a", "b", "c", "d"]
    assert 2.5 < s["a"] / s["c"] < 6.5          # true ratio 4


def test_ties_and_unbeaten_styles_stay_finite():
    s = bradley_terry(["a", "b"], [("a", "b", 1.0), ("a", "b", 1.0), ("a", "b", 0.5), ("b", "a", 0.5)])
    assert s["a"] > s["b"] > 0


def test_adaptive_duels_beat_random_pairs():
    import re
    import shutil
    import subprocess

    import pytest
    if not shutil.which("node"):
        pytest.skip("node not installed")
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(["node", str(root / "tests" / "duel_sim.mjs"), str(root / "deploy" / "listen_duel_page.html"),
                          "40", "150"], capture_output=True, text=True, check=True).stdout
    regret = {m.group(1): float(m.group(2)) for m in re.finditer(r"(random|adaptive)\s+40 votes:.*regret ([\d.]+)", out)}
    assert regret["adaptive"] < regret["random"] - 0.1, out
