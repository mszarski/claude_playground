"""The Sensor Bridge receive path on the CPU, with no network: what FrameReceiver (gpusonar/sources/hsb.py)
and the PingAssembler behind it (sources/flank_stream.py) count when frames go missing, repeat, arrive out of
order, damaged or not of this stream.

Every expected number is derived from the rules the code documents (FrameReceiver's and PingAssembler's docstrings):
  * frame numbers (the receiver operator's `frame_number` metadata): numbers skipped are awaited. One that arrives
    while its place is still open is `frames_late` and used (its ping kept); one that never arrives is `frames_lost`
    once its place is given up (its ping emitted or dropped, or, between pings, the stream `reorder_blocks` = 2 frames
    past it), and the number of the frame that arrived after the gap goes in `trouble_frames` (the first 16); one
    arriving after that is `frames_too_late` (and `frames_late`), not used. A number already received is
    `frames_repeated`, ignored. Without numbers the same is told from the blocks' sample counter.
  * `frame_bytes_received` below the frame's size: `frames_damaged`, the frame's number in `trouble_frames`, the
    frame not used. Not a block of this stream (size, magic, version, channel count): `bad_frames`, not used, not
    in `trouble_frames`.
  * The assembler sees only the frames used: samples never received are `lost_samples` once given up, and the ping
    whose window they cut is discarded (never zero-filled); a ping first named by a header after its window has
    opened, with what came before already received or given up, is discarded too ("joined too late"); a jump of the
    header's ping index counts the pings none of whose frames arrived. All count in `dropped_pings`.
(Changed from inv-hsb ea25f52 by the fixes of the three faults it exposed: see the comments marked "fixed:".)
What the tests cannot reach without Sensor Bridge: the metadata values themselves (`frame_number`,
`frame_bytes_received`, cumulative `packets_dropped` come from LinuxReceiverOp), the receiver's queue dropping
its oldest frames, the data socket's drop count and the one-ping-per-frame emission of HsbFramesToPingsOp.
"""

import numpy as np
import pytest

from gpusonar.sources import hsb
from gpusonar.sources.flank_stream import encode_block
from test_live import N, OVERRIDES, pings  # noqa: F401  (pings is a fixture)

# Brought from the investigation branch (inv-hsb ea25f52) without its two reserve_core tests: reserve_core is an
# unmerged candidate there.

FRAME = 400                        # samples per frame, as test_hsb.py
LEAD = 1.0


@pytest.fixture(scope="module")
def run(pings):
    """The frames of the six test_live pings, the records a clean pass gives, and where each ping lies in frames:
    T[i] the frame its transmit starts, W[i] the frame its receive window opens inside, E[i] the frame holding the
    window's last sample."""
    stream = hsb.FrameStream(pings, FRAME, lead_seconds=LEAD)
    frames = list(stream)
    ws, wl = stream.window_start, stream.window_length
    T = [tx // FRAME for tx in stream.tx]
    W = [(tx + ws) // FRAME for tx in stream.tx]
    E = [(tx + ws + wl - 1) // FRAME for tx in stream.tx]
    # The cases below rely on this geometry: transmits on a frame boundary, windows opening inside a frame, a few
    # frames between a transmit and its window, quiet frames between one window's end and the next transmit.
    assert all(tx % FRAME == 0 and (tx + ws) % FRAME for tx in stream.tx)
    assert all(W[i] - T[i] >= 3 and T[i + 1] - E[i] >= 40 for i in range(N - 1))
    clean, receiver = receive(numbered(frames))
    assert len(clean) == N and receiver["dropped_pings"] == 0 and receiver["frames"] == len(frames)
    return {"frames": frames, "T": T, "W": W, "E": E, "clean": clean}


def numbered(frames, skip=(), offset=0):
    """(frame, metadata) as the receiver operator gives them: every frame complete, numbered from `offset`."""
    skip = set(skip)
    return [(f, {"frame_number": offset + k, "frame_bytes_received": len(f), "packets_dropped": 0})
            for k, f in enumerate(frames) if k not in skip]


def receive(items):
    receiver = hsb.FrameReceiver("flank_32", OVERRIDES, frame_samples=FRAME)
    records = [r for f, m in items for r in receiver.take(f, m)]
    return records, receiver.stats()


def same_as_clean(records, clean):
    """Each record that came out is the clean pass's record of the same ping, sample for sample (nothing zero-filled)."""
    by_seq = {r.metadata.sequence: r for r in clean}
    for r in records:
        assert np.array_equal(r.samples.array, by_seq[r.metadata.sequence].samples.array)
        assert r.transmit.time_ns == by_seq[r.metadata.sequence].transmit.time_ns


def runs_of(skip):
    """The first frame after each contiguous run of skipped frames: where a gap is noticed."""
    s = sorted(skip)
    return [k + 1 for i, k in enumerate(s) if i + 1 == len(s) or s[i + 1] != k + 1]


# Which frames go missing (as a function of the geometry) and which pings that costs, by the window rule.
LOSSES = {
    # A transmit's own frame: the next frame's header names the ping before its window opens, so nothing is lost.
    "transmit_frame": (lambda g: [g["T"][2]], []),
    # Everything from the transmit to the frame before the one the window opens inside: the next frame starts
    # at or before the window's first sample, so the ping is heard of in time and kept.
    "transmit_to_before_window": (lambda g: range(g["T"][2], g["W"][2]), []),
    # ... one frame more: the first frame heard of ping 2 starts after its window opened (joined too late).
    "transmit_into_window": (lambda g: range(g["T"][2], g["W"][2] + 1), [2]),
    # The frame the window opens inside, alone (the ping is open): the gap cuts the window.
    "window_first_frame": (lambda g: [g["W"][2]], [2]),
    # The frame before it: the gap ends before the window opens.
    "frame_before_window": (lambda g: [g["W"][2] - 1], []),
    # The frame holding the window's last sample: cut.
    "window_last_frame": (lambda g: [g["E"][2]], [2]),
    # The frame after it: ping 2 was complete one frame earlier.
    "frame_after_window": (lambda g: [g["E"][2] + 1], []),
    # A run from inside ping 1's window to inside ping 2's: ping 1 is cut, ping 2 is heard of too late.
    "run_across_boundary_into_window": (lambda g: range(g["E"][1] - 10, g["W"][2] + 11), [1, 2]),
    # A run from inside ping 1's window to just before ping 2's opens: ping 1 is cut, ping 2 kept.
    "run_across_boundary_before_window": (lambda g: range(g["E"][1] - 10, g["W"][2]), [1]),
    # A run between the windows, over ping 2's transmit: nothing is cut.
    "run_between_windows_over_transmit": (lambda g: range(g["E"][1] + 1, g["W"][2]), []),
}


@pytest.mark.parametrize("case", sorted(LOSSES))
def test_lost_frames_cost_their_samples_and_exactly_the_pings_they_cut(run, case):
    where, dropped = LOSSES[case]
    skip = list(where(run))
    records, s = receive(numbered(run["frames"], skip))
    kept = [i for i in range(N) if i not in dropped]
    assert [r.metadata.sequence for r in records] == kept
    same_as_clean(records, run["clean"])
    # fixed: the stats name frames_late and frames_too_late (0 here: nothing arrives out of order).
    assert s == {"frames": len(run["frames"]) - len(skip), "frames_lost": len(skip), "frames_damaged": 0,
                 "frames_repeated": 0, "frames_late": 0, "frames_too_late": 0, "bad_frames": 0, "packets_dropped": 0, "first_frame_number": 0,
                 "trouble_frames": runs_of(skip), "pings": len(kept), "dropped_pings": len(dropped),
                 "lost_samples": FRAME * len(skip)}


def test_a_ping_lost_whole_is_counted_as_dropped(run):
    # fixed: was a strict xfail (dropped_pings 0). Counted from the jump of the header's ping index, 1 to 3.
    skip = range(run["T"][2], run["T"][3])               # every frame whose header names ping 2
    records, s = receive(numbered(run["frames"], skip))
    assert [r.metadata.sequence for r in records] == [0, 1, 3, 4, 5]
    assert (s["frames_lost"], s["lost_samples"]) == (len(skip), FRAME * len(skip))
    assert s["dropped_pings"] == 1


def test_repeated_and_out_of_order_frames(run):
    frames, W = run["frames"], run["W"]
    a, b = W[2] + 50, W[2] + 51

    def item(k, **change):
        return frames[k], dict({"frame_number": k, "frame_bytes_received": len(frames[k]), "packets_dropped": 0}, **change)

    # A repeat is ignored before anything else is looked at: a short copy reporting dropped packets changes nothing.
    items = [item(k) for k in range(len(frames))]
    items.insert(a + 1, item(a, frame_bytes_received=100, packets_dropped=9))
    items.insert(a + 2, item(W[2] + 10))                 # an older frame again, later
    records, s = receive(items)
    assert [r.metadata.sequence for r in records] == list(range(N))
    same_as_clean(records, run["clean"])
    assert (s["frames_repeated"], s["frames_damaged"], s["packets_dropped"], s["frames"], s["frames_lost"],
            s["lost_samples"], s["trouble_frames"]) == (2, 0, 0, len(frames), 0, 0, [])
    # Two frames swapped. fixed: was ([0, 1, 3, 4, 5]; frames_lost 1, frames_repeated 1, trouble [b], lost_samples FRAME,
    # dropped_pings 1, frames len - 1): the earlier frame now fills its place, nothing is lost, ping 2 is kept.
    items = [item(k) for k in range(len(frames))]
    items[a], items[b] = items[b], items[a]
    records, s = receive(items)
    assert [r.metadata.sequence for r in records] == list(range(N))
    same_as_clean(records, run["clean"])
    assert (s["frames_lost"], s["frames_repeated"], s["frames_late"], s["frames_too_late"], s["trouble_frames"],
            s["lost_samples"], s["dropped_pings"], s["frames"]) == (0, 0, 1, 0, [], 0, 0, len(frames))
    # The same swap with no metadata (no frame numbers): the sample counter places it. fixed: was ([0, 1, 3, 4, 5];
    # lost_samples 2 FRAME - the jump to b and the jump back counted - and dropped_pings 1).
    order = list(range(len(frames)))
    order[a], order[b] = order[b], order[a]
    records, s = receive([(frames[k], None) for k in order])
    assert [r.metadata.sequence for r in records] == list(range(N))
    same_as_clean(records, run["clean"])
    assert (s["frames"], s["frames_lost"], s["frames_repeated"], s["frames_late"], s["lost_samples"], s["dropped_pings"]) == \
        (len(frames), 0, 0, 1, 0, 0)
    # A repeat with no frame number: its header (same capture start, same transmit, samples already received) tells it
    # from a restart. fixed: was ([0, 1, 3, 4, 5]; frames len + 1, frames_repeated 0, dropped_pings 1).
    order = list(range(len(frames)))
    order.insert(a + 1, a)
    records, s = receive([(frames[k], None) for k in order])
    assert [r.metadata.sequence for r in records] == list(range(N))
    same_as_clean(records, run["clean"])
    assert (s["frames"], s["frames_repeated"], s["frames_late"], s["lost_samples"], s["dropped_pings"]) == (len(frames), 1, 0, 0, 0)


def test_damaged_frames_and_frames_not_of_this_stream_are_told_apart(run):
    frames, W, E = run["frames"], run["W"], run["E"]
    size = len(frames[0])
    quiet_damaged, window_damaged = E[1] + 5, W[3] + 5
    bad_magic, other_channels, bad_version = W[4] + 5, E[4] + 3, E[4] + 7
    items = []
    for k, f in enumerate(frames):
        # The receiver's cumulative count of dropped packets as it reports it; lower again later (a restarted count).
        dropped = 0 if k < quiet_damaged else 1 if k < window_damaged else 4 if k < window_damaged + 2 else 2
        m = {"frame_number": k, "frame_bytes_received": size, "packets_dropped": dropped}
        if k in (quiet_damaged, window_damaged):
            m["frame_bytes_received"] = size - 1408                    # one 1408-byte datagram missing
            f = f[:4096] + b"\xff" * 1408 + f[4096 + 1408:]
        elif k == bad_magic:
            f = b"XXXX" + f[4:]
        elif k == other_channels:                                       # same size: 16 channels of twice the samples
            f = encode_block(np.zeros((2 * FRAME, 16), dtype=np.int32), k * FRAME)
            assert len(f) == size
        elif k == bad_version:
            f = f[:4] + (99).to_bytes(2, "little") + f[6:]
        items.append((f, m))
    records, s = receive(items)
    assert [r.metadata.sequence for r in records] == [0, 1, 2, 5]
    same_as_clean(records, run["clean"])
    # fixed: the stats name frames_late and frames_too_late (0 here).
    assert s == {"frames": len(frames) - 5, "frames_lost": 0, "frames_damaged": 2, "frames_repeated": 0, "frames_late": 0,
                 "frames_too_late": 0, "bad_frames": 3,
                 "packets_dropped": 4, "first_frame_number": 0, "trouble_frames": [quiet_damaged, window_damaged],
                 "pings": 4, "dropped_pings": 2, "lost_samples": 5 * FRAME}


def test_a_stream_numbered_from_elsewhere_and_joined_late(run):
    frames, T, W = run["frames"], run["T"], run["W"]
    offset = 7_000_000
    # Joined at frame 3 of a sender counting from 7,000,000, then one frame of ping 3's window lost: the frames
    # before the first one seen are neither lost frames nor lost samples.
    gap = W[3] + 50
    items = numbered(frames, skip=[0, 1, 2, gap], offset=offset)
    records, s = receive(items)
    assert [r.metadata.sequence for r in records] == [0, 1, 2, 4, 5]
    same_as_clean(records, run["clean"])
    assert (s["first_frame_number"], s["frames_lost"], s["trouble_frames"], s["lost_samples"], s["dropped_pings"]) == \
        (offset + 3, 1, [offset + gap + 1], FRAME, 1)
    # Joined inside ping 1's window: ping 1 is heard of too late and counted dropped; ping 0, never heard of, is not.
    records, s = receive(numbered(frames, offset=offset)[W[1] + 10:])
    assert [r.metadata.sequence for r in records] == [2, 3, 4, 5]
    assert (s["first_frame_number"], s["frames_lost"], s["lost_samples"], s["dropped_pings"], s["pings"]) == \
        (offset + W[1] + 10, 0, 0, 1, 4)
    # Joined between ping 1's transmit and its window: ping 1 is kept.
    records, s = receive(numbered(frames, offset=offset)[T[1] + 1:])
    assert [r.metadata.sequence for r in records] == [1, 2, 3, 4, 5] and s["dropped_pings"] == 0
    same_as_clean(records, run["clean"])


def test_trouble_frames_keep_the_first_sixteen(run):
    E, T = run["E"], run["T"]
    skip = [E[0] + 1 + 2 * j for j in range(20)]                       # 20 single quiet frames, every other one
    assert skip[-1] < T[1]
    records, s = receive(numbered(run["frames"], skip))
    assert len(records) == N and s["dropped_pings"] == 0
    assert (s["frames_lost"], s["lost_samples"], s["trouble_frames"]) == (20, 20 * FRAME, [k + 1 for k in skip[:16]])



# -- added with the fixes: reordering, repeats and whole pings, numbered, unnumbered and in both assemblers -------------

def test_frames_out_of_order_keep_their_ping_and_late_ones_are_counted(run):
    frames, W, E = run["frames"], run["W"], run["E"]

    def items(order):
        return [(frames[k], {"frame_number": k, "frame_bytes_received": len(frames[k]), "packets_dropped": 0}) for k in order]

    # The first frame of ping 2's window ten frames late, and the last one swapped with the frame after the window
    # (reorder_blocks = 2 holds the window that long): every ping kept, nothing lost.
    order = [k for k in range(len(frames)) if k != W[2]]
    order.insert(order.index(W[2] + 10) + 1, W[2])
    i, j = order.index(E[2]), order.index(E[2] + 1)
    order[i], order[j] = order[j], order[i]
    records, s = receive(items(order))
    assert [r.metadata.sequence for r in records] == list(range(N))
    same_as_clean(records, run["clean"])
    assert (s["frames"], s["frames_late"], s["frames_too_late"], s["frames_lost"], s["frames_repeated"], s["trouble_frames"],
            s["lost_samples"], s["dropped_pings"]) == (len(frames), 2, 0, 0, 0, [], 0, 0)
    # A frame of ping 2's window that comes after the stream is 3 frames past the window: ping 2 was given up when the
    # stream got 2 frames past it, the frame counted lost then; on arrival it is counted too late and not used.
    k = W[2] + 5
    order = [x for x in range(len(frames)) if x != k]
    order.insert(order.index(E[2] + 3) + 1, k)
    for numbered_ in (True, False):
        records, s = receive(items(order) if numbered_ else [(frames[x], None) for x in order])
        assert [r.metadata.sequence for r in records] == [0, 1, 3, 4, 5]
        same_as_clean(records, run["clean"])
        assert (s["frames"], s["frames_late"], s["frames_too_late"], s["frames_repeated"], s["lost_samples"], s["dropped_pings"]) == \
            (len(frames) - 1, 1, 1, 0, FRAME, 1)
        assert (s["frames_lost"], s["trouble_frames"]) == ((1, [k + 1]) if numbered_ else (0, []))   # frames_lost needs numbers


def assembler_for(engine, **options):
    from gpusonar import _gpusonar as core
    from gpusonar.sonar import SonarModel
    from gpusonar.sources.flank_stream import PingAssembler
    sonar = SonarModel.load("flank_32", OVERRIDES)
    if engine == "python":
        return PingAssembler(sonar, **options)
    return core.PingAssemblerCpp(sonar.make_record(core.Pose()), **options)


def with_ping_index(frame, index):
    """The frame with its header's ping index replaced."""
    return frame[:28] + int(index & 0xFFFFFFFF).to_bytes(4, "little") + frame[32:]


def ping_of(frame):
    return int.from_bytes(frame[28:32], "little")


def scenario(run, name):
    """(frames in arrival order, expected sequences, expected counters) for one case."""
    frames, T, W, E = run["frames"], run["T"], run["W"], run["E"]
    k = list(range(len(frames)))
    a = W[2] + 50
    if name == "ping_lost_whole":
        return [frames[x] for x in k if not T[2] <= x < T[3]], [0, 1, 3, 4, 5], \
            dict(dropped_pings=1, lost_samples=FRAME * (T[3] - T[2]), late_blocks=0, too_late_blocks=0, repeated_blocks=0, restarts=0)
    if name == "swapped":
        k[a], k[a + 1] = k[a + 1], k[a]
        return [frames[x] for x in k], list(range(N)), \
            dict(dropped_pings=0, lost_samples=0, late_blocks=1, too_late_blocks=0, repeated_blocks=0, restarts=0)
    if name == "repeated":
        k.insert(a + 1, a)
        return [frames[x] for x in k], list(range(N)), \
            dict(dropped_pings=0, lost_samples=0, late_blocks=0, too_late_blocks=0, repeated_blocks=1, restarts=0)
    if name == "too_late":
        k.remove(W[2] + 5)
        k.insert(k.index(E[2] + 3) + 1, W[2] + 5)
        return [frames[x] for x in k], [0, 1, 3, 4, 5], \
            dict(dropped_pings=1, lost_samples=FRAME, late_blocks=1, too_late_blocks=1, repeated_blocks=0, restarts=0)
    if name == "index_wraps":
        # Ping indices 2^32 - 3 .. 2: ping 2 (index 2^32 - 1) lost whole is counted across the wrap.
        out = [with_ping_index(frames[x], ping_of(frames[x]) - 3) for x in k if not T[2] <= x < T[3]]
        return out, [2**32 - 3, 2**32 - 2, 0, 1, 2], \
            dict(dropped_pings=1, lost_samples=FRAME * (T[3] - T[2]), late_blocks=0, too_late_blocks=0, repeated_blocks=0, restarts=0)
    if name == "index_renumbered":
        # From ping 3 on the board numbers from 1003: a jump of 1001 one ping interval after the last is a renumbering.
        out = [with_ping_index(f, ping_of(f) + 1000) if x >= T[3] else f for x, f in enumerate(frames)]
        return out, [0, 1, 2, 1003, 1004, 1005], \
            dict(dropped_pings=0, lost_samples=0, late_blocks=0, too_late_blocks=0, repeated_blocks=0, restarts=0)
    if name == "restarted":
        # The capture starts again from sample 0 (same capture start time: a board without a clock).
        return frames + frames, list(range(N)) * 2, \
            dict(dropped_pings=0, lost_samples=0, late_blocks=0, too_late_blocks=0, repeated_blocks=0, restarts=1)
    raise KeyError(name)


SCENARIOS = ["ping_lost_whole", "swapped", "repeated", "too_late", "index_wraps", "index_renumbered", "restarted"]


@pytest.mark.parametrize("engine", ["python", "cpp"])
@pytest.mark.parametrize("name", SCENARIOS)
def test_both_assemblers_make_the_same_decisions(run, engine, name):
    frames, sequences, counters = scenario(run, name)
    asm = assembler_for(engine)
    records = [r for f in frames for r in asm.feed(f)]
    assert [r.metadata.sequence for r in records] == sequences
    assert {key: getattr(asm, key) for key in counters} == counters
    kept = {"ping_lost_whole": [0, 1, 3, 4, 5], "too_late": [0, 1, 3, 4, 5], "index_wraps": [0, 1, 3, 4, 5],
            "restarted": list(range(N)) * 2}.get(name, list(range(N)))                # which clean ping each record is
    for r, i in zip(records, kept, strict=True):
        assert np.array_equal(r.samples.array, run["clean"][i].samples.array)
        assert r.transmit.time_ns == run["clean"][i].transmit.time_ns


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_both_assemblers_reorder_before_decimating(run, engine):
    """With decimation the samples behind a hole wait for it: swapped frames give the clean stream's records exactly."""
    frames, W = run["frames"], run["W"]
    k = list(range(len(frames)))
    for a in (W[1] + 3, W[3] + 20):
        k[a], k[a + 1] = k[a + 1], k[a]
    k.remove(W[2])
    k.insert(k.index(W[2] + 30), W[2])                    # the first frame of a window 30 frames late
    asm, clean_asm = assembler_for(engine, decimate={}), assembler_for(engine, decimate={})
    got = [r for x in k for r in asm.feed(frames[x])]
    want = [r for f in frames for r in clean_asm.feed(f)]
    assert len(got) == len(want) == N and asm.dropped_pings == 0 and asm.late_blocks == 3
    for g, w in zip(got, want):
        assert g.samples.is_complex() and np.array_equal(np.asarray(g.samples.array), np.asarray(w.samples.array))


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_reordering_adds_no_output_latency(run, engine):
    """Each ping comes out of the frame holding its window's last sample, in order or not (a late frame that completes
    a window releases it at once)."""
    frames, W, E = run["frames"], run["W"], run["E"]
    asm = assembler_for(engine)
    emitted = [k for k, f in enumerate(frames) for _ in asm.feed(f)]
    assert emitted == E
    k = list(range(len(frames)))
    k[E[2]], k[E[2] + 1] = k[E[2] + 1], k[E[2]]          # ping 2's last frame one frame late: out when it arrives
    asm = assembler_for(engine)
    emitted = [k[i] for i in range(len(k)) for _ in asm.feed(frames[k[i]])]
    assert emitted == E and asm.late_blocks == 1


# -- added with the review of fix-hsb: a reduced geometry (window start 256, length 384: W = 640; blocks of 64) -----------

SMALL = {"receive": {"window": {"range": [3.0, 6.0]}}, "transmit": {"waveform": {"duration": 0.002}}}


def small(engine, **options):
    from gpusonar import _gpusonar as core
    from gpusonar.sonar import SonarModel
    from gpusonar.sources.flank_stream import PingAssembler
    sonar = SonarModel.load("flank_32", SMALL)
    if engine == "python":
        return PingAssembler(sonar, **options)
    return core.PingAssemblerCpp(sonar.make_record(core.Pose()), **options)


def small_stream(txs, block, total, index=None, start_ns=1000, seed=0):
    """{first sample: block bytes} of a stream with transmits `txs`, each block naming the latest transmit at or before
    its first sample, and the blocks' first samples in order."""
    index = list(range(len(txs))) if index is None else index
    data = np.random.default_rng(seed).integers(-1000, 1000, size=(total, 32)).astype(np.int32)
    out = {}
    for first in range(0, total, block):
        last = max((i for i, t in enumerate(txs) if t <= first), default=-1)
        out[first] = encode_block(data[first:first + block], first, txs[last] if last >= 0 else -1,
                                  index[last] if last >= 0 else 0, 0, 0, start_ns)
    return out, list(out)


def counters(a):
    return {k: int(getattr(a, k)) for k in ("dropped_pings", "lost_samples", "late_blocks", "too_late_blocks",
                                             "repeated_blocks", "restarts")}


def feed_all(a, blocks, finish=False):
    out = [int(r.metadata.sequence) for b in blocks for r in a.feed(b)]
    return out + ([int(r.metadata.sequence) for r in a.finish()] if finish else [])


P = 640
TXS = [64 + i * P for i in range(5)]                  # back to back: each window ends at the next transmit


def review_case(name):
    """(blocks, expected sequences, expected counters) of the reviewer's cases, as fixed."""
    by, order = small_stream(TXS, 64, TXS[-1] + P)
    t = TXS[2]
    zero = dict(dropped_pings=0, lost_samples=0, late_blocks=0, too_late_blocks=0, repeated_blocks=0, restarts=0)
    if name == "swap_at_ping_boundary":       # was [0, 2, 3, 4], dropped 1, lost 64, too late 1
        i = order.index(t - 64)
        order[i], order[i + 1] = order[i + 1], order[i]
        return [by[f] for f in order], [0, 1, 2, 3, 4], dict(zero, late_blocks=1)
    if name == "hole_across_two_pings":       # was [0, 3, 4], dropped 2, lost 384, too late 5
        late = [f for f in order if t <= f < t + 320]
        o = [f for f in order if f not in late and f != t - 64]
        i = o.index(t + 384)
        o = o[:i + 1] + late + o[i + 1:]
        return [by[f] for f in o], [0, 2, 3, 4], dict(zero, dropped_pings=1, lost_samples=64, late_blocks=5)
    if name == "stale_duplicate":             # was [0, 1, 2, 4], dropped 4, restarts 1
        order.insert(order.index(TXS[3] + 320), TXS[1] + 320)
        return [by[f] for f in order], [0, 1, 2, 3, 4], dict(zero, repeated_blocks=1)
    if name == "two_stale_adjacent":          # two contiguous old blocks of the same capture: not yet a restart
        i = order.index(TXS[4] + 320)
        order[i:i] = [TXS[1] + 320, TXS[1] + 384]
        return [by[f] for f in order], [0, 1, 2, 3, 4], dict(zero, repeated_blocks=2)
    if name == "first_two_swapped":           # was restarts 1, dropped 1, lost 64
        order[0], order[1] = order[1], order[0]
        return [by[f] for f in order], [0, 1, 2, 3, 4], dict(zero, late_blocks=1)
    if name == "restart_with_straggler":      # was restarts 3, dropped 2
        new, new_order = small_stream(TXS, 64, TXS[-1] + P, start_ns=99999)
        old = [by[f] for f in order]
        fresh = [new[f] for f in new_order]
        return old[:30] + [fresh[0], old[30], fresh[1]] + fresh[2:], [0, 1, 2, 0, 1, 2, 3, 4], dict(zero, restarts=1)
    one, one_order = small_stream([i * P for i in range(6)], P, 6 * P)
    if name == "one_block_pings_swapped":     # was emitted [0, 2, 1, 3, 4, 5], dropped 2, restarts 1
        one_order[1], one_order[2] = one_order[2], one_order[1]
        return [one[f] for f in one_order], list(range(6)), dict(zero, late_blocks=1)
    two, two_order = small_stream([i * P for i in range(6)], P // 2, 6 * P)
    if name == "two_block_ping_named_late":   # was [0, 1, 3, 4, 5], dropped 3, restarts 1
        two_order = two_order[:2] + [two_order[4], two_order[2], two_order[3]] + two_order[5:]
        return [two[f] for f in two_order], list(range(6)), dict(zero, late_blocks=2)
    long_txs = [64 + i * 4 * P for i in range(4)]
    renumbered, r_order = small_stream(long_txs, 64, long_txs[-1] + P, index=[0, 1, 4, 5])
    if name == "long_interval_renumbered":    # was dropped 2: the jump 1 -> 4 one interval apart counted
        return [renumbered[f] for f in r_order], [0, 1, 4, 5], zero
    lost, l_order = small_stream(long_txs, 64, long_txs[-1] + P)
    if name == "long_interval_ping_lost":     # ping 2 lost whole, one interval of 4 W: counted from the interval
        return [lost[f] for f in l_order if not long_txs[2] <= f < long_txs[3]], [0, 1, 3], \
            dict(zero, dropped_pings=1, lost_samples=long_txs[3] - long_txs[2])
    if name == "jump_before_an_interval_is_known":   # ping 1 lost whole after ping 0: no interval yet, not counted
        return [lost[f] for f in l_order if not long_txs[1] <= f < long_txs[2]], [0, 2, 3], \
            dict(zero, lost_samples=long_txs[2] - long_txs[1])
    raise KeyError(name)


REVIEW = ["swap_at_ping_boundary", "hole_across_two_pings", "stale_duplicate", "two_stale_adjacent", "first_two_swapped", "restart_with_straggler",
          "one_block_pings_swapped", "two_block_ping_named_late", "long_interval_renumbered", "long_interval_ping_lost",
          "jump_before_an_interval_is_known"]


@pytest.mark.parametrize("engine", ["python", "cpp"])
@pytest.mark.parametrize("name", REVIEW)
def test_review_cases_in_both_assemblers(engine, name):
    blocks, sequences, want = review_case(name)
    a = small(engine)
    assert feed_all(a, blocks) == sequences
    assert counters(a) == want


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_finish_counts_what_is_still_awaited_and_nothing_else(engine):
    by, order = small_stream(TXS, 64, TXS[-1] + P)
    # The stream ends 2 blocks into ping 4's window, one of them lost: not yet given up, then counted by finish().
    end = order.index(TXS[4] + 256 + 128)
    blocks = [by[f] for f in order[:end] if f != TXS[4] + 256]
    a = small(engine)
    assert feed_all(a, blocks) == [0, 1, 2, 3] and (a.dropped_pings, a.lost_samples) == (0, 0)
    assert a.finish() == [] and (a.dropped_pings, a.lost_samples) == (1, 64)
    # Cut short with nothing missing: not a drop.
    a = small(engine)
    assert feed_all(a, [by[f] for f in order[:end]], finish=True) == [0, 1, 2, 3] and counters(a)["dropped_pings"] == 0
    # A complete ping waiting for an earlier one that may still be named comes out of finish(): one block per ping,
    # ping 1's block lost; ping 2 is complete but waits while the hole where ping 1 would be is open.
    one, one_order = small_stream([i * P for i in range(6)], P, 6 * P)
    a = small(engine)
    assert feed_all(a, [one[f] for f in one_order[:3] if f != P]) == [0]
    assert [int(r.metadata.sequence) for r in a.finish()] == [2] and (a.dropped_pings, a.lost_samples) == (0, P)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_sample_indices_out_of_range_are_refused_alike(engine):
    from gpusonar.sources.flank_stream import BlockError
    a = small(engine)
    zeros = np.zeros((64, 32), dtype=np.int32)
    for block in (encode_block(zeros, 1 << 62, -1, 0), encode_block(zeros, (1 << 64) - 64, -1, 0),
                  encode_block(zeros, 0, 1 << 62, 0), encode_block(zeros, 0, -2, 0)):
        if engine == "python":
            with pytest.raises(BlockError, match="out of range"):
                a.feed(block)
        else:
            assert a.feed(block) == []
    assert engine == "python" or a.bad_blocks == 4
    assert a.feed(encode_block(zeros, (1 << 62) - 64, (1 << 62) - 64, 7)) == [] and a.last_block == "in_order"


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_rows_held_behind_a_hole_are_bounded(engine):
    """Decimating, a ping whose first block is late holds the rest raw; past `max_held_bytes` it is given up (counted)."""
    by, order = small_stream(TXS, 64, TXS[-1] + P)
    o = list(order)
    w = TXS[1] + 256                                                   # ping 1's first window block, 5 blocks late
    o.remove(w)
    o.insert(o.index(w + 320) + 1, w)
    blocks = [by[f] for f in o]
    roomy, clean, tight = small(engine, decimate={}), small(engine, decimate={}), small(engine, decimate={}, max_held_bytes=3 * 64 * 32 * 4)
    got = [r for b in blocks for r in roomy.feed(b)]
    want = [r for f in order for r in clean.feed(by[f])]
    assert [int(r.metadata.sequence) for r in got] == [0, 1, 2, 3, 4] and roomy.dropped_pings == 0
    assert all(np.array_equal(np.asarray(g.samples.array), np.asarray(w_.samples.array)) for g, w_ in zip(got, want))
    assert roomy.held_peak_bytes == 5 * 64 * 32 * 4 and clean.held_peak_bytes == 0     # measured: 5 blocks of rows
    assert feed_all(tight, blocks) == [0, 2, 3, 4] and tight.dropped_pings == 1 and tight.lost_samples == 64


def test_frame_numbers_restarting_or_repeating_are_told_apart(run):
    frames = run["frames"]
    # The sender restarts: samples and frame numbers from 0 again (was: the whole second run ignored as repeats).
    items = [(f, {"frame_number": k % len(frames), "frame_bytes_received": len(f), "packets_dropped": 0})
             for k, f in enumerate(frames + frames)]
    records, s = receive(items)
    assert [r.metadata.sequence for r in records] == list(range(N)) * 2
    assert (s["frames"], s["frames_repeated"], s["frames_lost"], s["dropped_pings"]) == (2 * len(frames), 0, 0, 0)
    # A frame lost, then arriving too late, then again: counted once too late, then a repeat (was: too late twice).
    W, E = run["W"], run["E"]
    k = W[2] + 5
    order = [x for x in range(len(frames)) if x != k]
    i = order.index(E[2] + 3) + 1
    order[i:i] = [k, k]
    for numbered_ in (True, False):
        items = [(frames[x], {"frame_number": x, "frame_bytes_received": len(frames[x]), "packets_dropped": 0} if numbered_ else None)
                 for x in order]
        records, s = receive(items)
        assert (s["frames_too_late"], s["frames_late"], s["frames_repeated"]) == (1, 1, 1)
        assert s["frames"] + s["frames_repeated"] + s["frames_too_late"] == len(order)          # every arrival once
        assert s["frames_lost"] >= s["frames_too_late"] if numbered_ else s["frames_lost"] == 0
