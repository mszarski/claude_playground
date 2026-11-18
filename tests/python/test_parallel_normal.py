"""gpusonar.sim.parallel_normal: the deviates of `rng.standard_normal`, drawn on several threads,
must be those very numbers, and the generator must be left where the plain call leaves it."""

import numpy as np
import pytest

from gpusonar.sim import parallel_normal as pn


def reference(seed, shape, kind=np.random.PCG64, before=None):
    rng = np.random.Generator(kind(seed))
    if before is not None:
        before(rng)
    return rng.standard_normal(shape), rng.bit_generator.state, rng.standard_normal(5)


@pytest.mark.parametrize("seed,shape,threads", [(0, (3, 2, 1 << 20), 4), (1, ((1 << 22) + 12345,), 8), (2, (1 << 21,), 2),
                                                (3, (7, 2, 400001), 3), (4, ((1 << 23) + 1,), 16)])
def test_the_same_deviates_and_the_same_state(seed, shape, threads):
    want, state, after = reference(seed, shape)
    rng = np.random.default_rng(seed)
    out = np.empty(shape)
    before = dict(pn.stats)
    assert pn.standard_normal(rng, out, threads=threads) is out
    assert pn.stats["parallel"] == before["parallel"] + 1 and pn.stats["not_lined_up"] == before["not_lined_up"]
    assert np.array_equal(out, want)                                  # bit for bit
    assert rng.bit_generator.state == state
    assert np.array_equal(rng.standard_normal(5), after)              # and the sequence goes on as it would have


def test_a_generator_part_way_through_its_stream_with_a_cached_half_word():
    """The state NumPy keeps besides the 128 bits (half of a 64-bit draw, after a 32-bit one)
    is carried over untouched, as the plain call leaves it."""
    def before(rng):
        rng.standard_normal(1001)
        rng.integers(0, 1 << 30, size=3, dtype=np.uint32)
    want, state, after = reference(11, (1 << 22,), before=before)
    rng = np.random.default_rng(11)
    before(rng)
    out = np.empty(1 << 22)
    pn.standard_normal(rng, out, threads=4)
    assert np.array_equal(out, want) and rng.bit_generator.state == state and np.array_equal(rng.standard_normal(5), after)


def test_successive_calls_continue_one_sequence():
    want, state, _ = reference(5, (3, 1 << 21))
    rng = np.random.default_rng(5)
    out = np.empty((3, 1 << 21))
    for row in out:
        pn.standard_normal(rng, row, threads=4)
    assert np.array_equal(out, want) and rng.bit_generator.state == state


@pytest.mark.parametrize("kind", [k for k in (getattr(np.random, "PCG64DXSM", None), np.random.MT19937, np.random.Philox) if k])
def test_other_generators(kind):
    """PCG64DXSM is handled like PCG64 (its multiplier is read off the generator); anything else
    takes the plain call."""
    want, state, _ = reference(9, (1 << 22,), kind=kind)
    rng = np.random.Generator(kind(9))
    out = np.empty(1 << 22)
    before = dict(pn.stats)
    pn.standard_normal(rng, out, threads=4)
    assert np.array_equal(out, want) and repr(rng.bit_generator.state) == repr(state)
    assert pn.stats["not_lined_up"] == before["not_lined_up"]
    assert (pn.stats["parallel"] > before["parallel"]) == (kind.__name__.startswith("PCG64"))


def test_small_or_awkward_requests_take_the_plain_call():
    for shape, threads in (((1000,), 8), ((1 << 22,), 1)):
        want, state, _ = reference(6, shape)
        rng = np.random.default_rng(6)
        out = np.empty(shape)
        before = pn.stats["plain"]
        pn.standard_normal(rng, out, threads=threads)
        assert pn.stats["plain"] == before + 1 and np.array_equal(out, want) and rng.bit_generator.state == state
    # A view that is not contiguous is filled by NumPy itself.
    want, state, _ = reference(6, (1 << 21,))
    rng, big = np.random.default_rng(6), np.zeros(1 << 22)
    pn.standard_normal(rng, big[::2], threads=4)
    assert np.array_equal(big[::2], want) and not big[1::2].any() and rng.bit_generator.state == state


def test_distance_along_the_generator():
    """lcg_distance against PCG64's own jump-ahead; the multiplier comes from the generator."""
    bg = np.random.PCG64(21)
    multiplier, per_deviate = pn._measure(bg)
    assert 1.0 < per_deviate < 1.05                                   # nearly every deviate takes one draw
    start = bg.state
    for steps in (0, 1, 2, 12345, (1 << 70) + 99):
        other = np.random.PCG64(1)
        other.state = start
        other.advance(steps)
        assert pn.lcg_distance(int(start["state"]["state"]), int(other.state["state"]["state"]), multiplier,
                               int(start["state"]["inc"])) == steps
    bg.random_raw(1000)
    assert pn.lcg_distance(int(start["state"]["state"]), int(bg.state["state"]["state"]), multiplier, int(start["state"]["inc"])) == 1000


def test_a_chunk_that_cannot_be_lined_up_falls_back_to_the_plain_call(monkeypatch):
    monkeypatch.setitem(pn._measured, np.random.PCG64, (pn._measure(np.random.PCG64(1))[0], 1.5))   # a wrong mean
    want, state, _ = reference(8, (1 << 22,))
    rng = np.random.default_rng(8)
    out = np.empty(1 << 22)
    before = pn.stats["not_lined_up"]
    pn.standard_normal(rng, out, threads=4)
    assert pn.stats["not_lined_up"] == before + 1
    assert np.array_equal(out, want) and rng.bit_generator.state == state


def test_a_forked_child_draws_too():
    """A child process inherits the pool object without its threads; it must make its own."""
    import multiprocessing as mp
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("no fork on this platform")
    rng = np.random.default_rng(3)
    pn.standard_normal(rng, np.empty(1 << 21), threads=2)             # the parent has a pool now
    want, _, _ = reference(4, (1 << 21,))
    ctx = mp.get_context("fork")
    with ctx.Pool(1) as pool:
        got = pool.apply_async(_child_draw, (4, 1 << 21)).get(timeout=120)
    assert np.array_equal(got, want)


def _child_draw(seed, n):
    out = np.empty(n)
    pn.standard_normal(np.random.default_rng(seed), out, threads=2)
    return out
