"""tools/test_shards.py: every test file is in exactly one shard, one shard of one is the whole suite, the split
depends only on the tree and the weights, and it is balanced when the weights are known."""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def tool():
    spec = importlib.util.spec_from_file_location("test_shards_tool", ROOT / "tools" / "test_shards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_every_file_is_in_exactly_one_shard(n):
    t = tool()
    on_disk = sorted(p.name for p in (ROOT / "tests" / "python").glob("test_*.py"))
    shards = t.split(n)
    assert len(shards) == n
    assert sorted(x for s in shards for x in s) == on_disk                       # none lost, none twice
    assert t.split(n) == shards                                                   # the same split again
    if n == 1:
        assert shards[0] == on_disk


def test_the_printed_shard_is_the_split_and_bad_specs_are_refused(capsys):
    t = tool()
    assert t.main(["2/3"]) == 0
    assert capsys.readouterr().out.split() == [f"tests/python/{x}" for x in t.split(3)[1]]
    for bad in ("0/3", "4/3", "x", "1/0"):
        with pytest.raises(ValueError):
            t.parse(bad)
    with pytest.raises(ValueError):
        t.split(0)


def test_known_weights_balance_and_an_unlisted_file_gets_the_median():
    t = tool()
    names = [f"test_{c}.py" for c in "abcdefgh"]
    w = t.weights(names, table={"test_a.py": 100.0, "test_b.py": 60.0, "test_c.py": 40.0, "test_d.py": 10.0})
    assert w["test_e.py"] == 50.0                                                 # the median of 100, 60, 40, 10
    shards = t.split(2, names, w)
    load = [sum(w[x] for x in s) for s in shards]
    assert sum(load) == sum(w.values()) and abs(load[0] - load[1]) <= max(w.values())


def test_seconds_are_read_from_a_durations_log():
    t = tool()
    log = "12.50s call     tests/python/test_x.py::test_a\n0.70s setup    tests/python/test_x.py::test_a\n3.00s call tests/python/test_y.py::test_b[1]\n(5 durations < 0.5s hidden.)\n"
    assert t.seconds_from_log(log) == {"test_x.py": 13.2, "test_y.py": 3.0}
