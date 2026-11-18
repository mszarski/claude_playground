"""Standard normal deviates of a NumPy generator, drawn on several threads: bit for bit the
numbers `rng.standard_normal` gives, and the same generator state afterwards.

Why this exists. The simulator keeps its random numbers in NumPy so that both engines and
both backends draw the same ones (docs/DECISIONS.md D-61). The reverberation generator needs
two deviates per reduced-rate sample of every component, 10^8 to 10^9 of them per ping, and
`standard_normal` fills them on one thread at about 10 ns each: once the transforms run on
the device, that is most of what is left.

How it stays exact. NumPy's normal sampler (a ziggurat) turns the generator's stream of
64-bit draws into deviates one after another; a deviate takes one draw, occasionally more,
so where the k-th deviate starts in the stream is known only by producing the ones before
it. But the sampler has no memory between deviates: started at a draw where a deviate of
the true sequence starts, it produces the true sequence from there on. And a sampler
started a little before such a place falls into step with the true sequence within a few
deviates (almost every draw is the start of a deviate). So:

1. The stream is cut into one chunk of deviates per thread. Each thread jumps a copy of the
   generator (PCG64 can be advanced in O(log n)) to a little *before* where its chunk is
   expected to start, using the measured mean number of draws per deviate, and draws its
   chunk plus a margin, with NumPy's own `standard_normal` (which releases the GIL).
2. The chunks are joined in order. The deviates that follow chunk c in thread c's output are
   true ones (they continue the true sequence); a run of eight of them is looked for in
   thread c+1's output, which gives the place where thread c+1 is in step, and from there on
   its numbers are the true ones.
3. The generator's state after the last deviate is found the same way: a short draw started
   just before the end, lined up on the last true deviates, repeated to stop exactly there.

Nothing here knows the sampler's tables or the generator's constants (the multiplier is read
off the generator itself), and nothing is approximate: if any chunk cannot be lined up (the
margins are tens of standard deviations wide), or the generator is not a PCG64, the plain
single-threaded call is made instead. The result is checked against `rng.standard_normal`
in tests/python/test_parallel_normal.py.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MIN_PARALLEL = 1 << 21        # deviates below which the plain call is used
MIN_CHUNK = 1 << 19           # deviates per thread, at least
SLACK = 1 << 14               # draws a chunk starts before the place it is expected to start
MARGIN = 1 << 16              # deviates a chunk draws beyond its own
PROBE_AT, PROBE = 64, 8       # the run of true deviates looked for in the next chunk, and where it starts
MASK = (1 << 128) - 1

_measured = {}                # bit generator class -> (multiplier, mean draws per deviate)
_pool, _pool_threads, _pool_pid = None, 0, 0
# How the calls went: on threads, by the plain call because the input did not qualify, or by
# the plain call because a chunk could not be lined up (should not happen).
stats = {"parallel": 0, "plain": 0, "not_lined_up": 0}


def threads_available() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                                       # not on Linux
        return os.cpu_count() or 1


def _state_of(bit_generator) -> int:
    return int(bit_generator.state["state"]["state"])


def lcg_distance(start: int, end: int, multiplier: int, increment: int) -> int:
    """Number of steps x -> multiplier x + increment (mod 2^128) from `start` to `end`: one bit
    of the answer per doubling of the step, as jump-ahead does in reverse."""
    distance, bit = 0, 1
    while start != end:
        if (start ^ end) & bit:
            start = (start * multiplier + increment) & MASK
            distance |= bit
        bit <<= 1
        increment = ((multiplier + 1) * increment) & MASK
        multiplier = (multiplier * multiplier) & MASK
        if bit > MASK:
            raise ValueError("the states are not on one sequence")
    return distance


def _supported(bit_generator) -> bool:
    kinds = tuple(k for k in (getattr(np.random, "PCG64", None), getattr(np.random, "PCG64DXSM", None)) if k is not None)
    return isinstance(bit_generator, kinds) and hasattr(bit_generator, "advance")


def _measure(bit_generator):
    """(multiplier of the generator's recurrence, mean number of draws per normal deviate)."""
    kind = type(bit_generator)
    if kind not in _measured:
        scratch = kind(1)
        state = scratch.state
        increment = int(state["state"]["inc"])
        state["state"]["state"] = 1
        scratch.state = state
        scratch.random_raw(1)
        multiplier = (_state_of(scratch) - increment) & MASK            # 1 * multiplier + increment
        before = _state_of(scratch)
        n = 1 << 21
        np.random.Generator(scratch).standard_normal(n)
        _measured[kind] = (multiplier, lcg_distance(before, _state_of(scratch), multiplier, increment) / n)
    return _measured[kind]


def _jumped(bit_generator, state, draws: int):
    """A generator of the same kind at `state`, advanced by `draws`."""
    copy = type(bit_generator)(1)
    copy.state = state
    if draws > 0:
        copy.advance(draws)
    return np.random.Generator(copy)


def _find(buffer: np.ndarray, probe: np.ndarray) -> int:
    """Index in `buffer` where the run `probe` is, or -1."""
    for j in np.flatnonzero(buffer[:min(len(buffer), 2 * MARGIN) - len(probe) + 1] == probe[0]):
        if np.array_equal(buffer[j:j + len(probe)], probe):
            return int(j)
    return -1


def standard_normal(rng: np.random.Generator, out: np.ndarray, threads: int | None = None) -> np.ndarray:
    """Fill `out` (float64, C-contiguous) with the deviates `rng.standard_normal(out=out)` would
    give, leaving `rng` in the state that call would leave it in; drawn on `threads` threads
    (default: the CPUs this process may use) when there are enough of them to be worth it."""
    global _pool, _pool_threads, _pool_pid
    n = out.size
    threads = min(threads_available() if threads is None else int(threads), 32, n // MIN_CHUNK)
    bit_generator = rng.bit_generator
    flat = out.reshape(-1) if out.flags.c_contiguous and out.dtype == np.float64 else None
    if flat is None or threads < 2 or n < MIN_PARALLEL or not _supported(bit_generator):
        stats["plain"] += 1
        if flat is None:
            out[...] = rng.standard_normal(out.shape)                   # NumPy fills contiguous float64 arrays only
        else:
            rng.standard_normal(out=out)
        return out

    def plain():
        stats["not_lined_up"] += 1
        rng.standard_normal(out=out)
        return out
    multiplier, per_deviate = _measure(bit_generator)
    start = bit_generator.state
    counts = [n // threads] * threads                                 # every chunk at least MIN_CHUNK deviates
    counts[-1] += n - sum(counts)
    firsts = [int(f) for f in np.cumsum([0] + counts[:-1])]

    def draw(c):
        # Chunk 0 is the true sequence from its start; the others start SLACK draws early.
        jump = max(int(firsts[c] * per_deviate) - SLACK, 0) if c else 0
        return _jumped(bit_generator, start, jump).standard_normal(counts[c] + (MARGIN if c else PROBE_AT + PROBE))

    if _pool is None or _pool_threads < threads or _pool_pid != os.getpid():
        # (A forked child has the parent's pool object but none of its threads.)
        _pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="gpusonar-normal")
        _pool_threads, _pool_pid = threads, os.getpid()
    buffers = list(_pool.map(draw, range(len(firsts))))
    # Where each chunk is in step with the true sequence: buffers[c][offsets[c] + i] is the
    # i-th deviate of chunk c for i >= PROBE_AT (chunk 0: for every i).
    offsets = [0]
    for c in range(1, len(firsts)):
        head = buffers[c - 1][offsets[c - 1] + counts[c - 1]:]          # true deviates after chunk c - 1
        j = _find(buffers[c], head[PROBE_AT:PROBE_AT + PROBE]) if len(head) >= PROBE_AT + PROBE else -1
        offset = j - PROBE_AT
        if j < 0 or offset + counts[c] + PROBE_AT + PROBE > len(buffers[c]):
            return plain()
        offsets.append(offset)
    # The generator after the last deviate: a short draw lined up on the last true deviates.
    last = len(firsts) - 1
    back = 1024                                                         # the probe starts this many deviates before the end
    end = offsets[last] + counts[last]
    probe = buffers[last][end - back:end - back + PROBE]
    jump = max(int((n - back) * per_deviate) - SLACK, 0)
    j = _find(_jumped(bit_generator, start, jump).standard_normal(MARGIN), probe)
    if j < 0:
        return plain()
    finder = _jumped(bit_generator, start, jump)
    finder.standard_normal(j + back)

    def copy(c):
        first, count, offset = firsts[c], counts[c], offsets[c]
        if c:
            head = buffers[c - 1][offsets[c - 1] + counts[c - 1]:]
            flat[first:first + PROBE_AT] = head[:PROBE_AT]
            flat[first + PROBE_AT:first + count] = buffers[c][offset + PROBE_AT:offset + count]
        else:
            flat[first:first + count] = buffers[c][:count]

    list(_pool.map(copy, range(len(firsts))))
    # Jump-ahead drops the half word NumPy keeps after a 32-bit draw; the normal sampler neither
    # uses nor changes it, so the plain call would have left it as it was.
    final = finder.bit_generator.state
    for key in ("has_uint32", "uinteger"):
        if key in start:
            final[key] = start[key]
    bit_generator.state = final
    stats["parallel"] += 1
    return out
