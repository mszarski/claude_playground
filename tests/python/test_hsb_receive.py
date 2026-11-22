"""The Sensor Bridge receive path on the CPU, with no network: what FrameReceiver (gpusonar/sources/hsb.py)
and the PingAssembler behind it (sources/flank_stream.py) count when frames go missing, repeat, arrive out of
order, damaged or not of this stream.

Every expected number is derived from the rules the code documents:
  * frame numbers (the receiver operator's `frame_number` metadata): a number above the last one seen counts the
    numbers skipped in `frames_lost` and puts the arriving frame's number in `trouble_frames` (the first 16); a
    number at or below the last one is `frames_repeated` and the frame is ignored entirely (no reordering).
  * `frame_bytes_received` below the frame's size: `frames_damaged`, the frame's number in `trouble_frames`, the
    frame not used. Not a block of this stream (size, magic, version, channel count): `bad_frames`, not used, not
    in `trouble_frames`.
  * The assembler sees only the frames used: a jump in `sample_index` adds the samples skipped to `lost_samples`
    and discards the ping whose window it cuts through (never zero-filled); a ping first named by a header after
    its window has opened is discarded too ("joined too late"). Both count in `dropped_pings`.
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
    assert s == {"frames": len(run["frames"]) - len(skip), "frames_lost": len(skip), "frames_damaged": 0,
                 "frames_repeated": 0, "bad_frames": 0, "packets_dropped": 0, "first_frame_number": 0,
                 "trouble_frames": runs_of(skip), "pings": len(kept), "dropped_pings": len(dropped),
                 "lost_samples": FRAME * len(skip)}


@pytest.mark.xfail(strict=True, reason="BUG (reported, not fixed): a ping none of whose frames arrives - every frame "
                   "from its transmit to the next one lost - is never named by a header, so dropped_pings does not "
                   "count it; only frames_lost and lost_samples show the loss")
def test_a_ping_lost_whole_is_counted_as_dropped(run):
    skip = range(run["T"][2], run["T"][3])               # every frame whose header names ping 2
    records, s = receive(numbered(run["frames"], skip))
    assert [r.metadata.sequence for r in records] == [0, 1, 3, 4, 5]
    assert (s["frames_lost"], s["lost_samples"]) == (len(skip), FRAME * len(skip))
    assert s["dropped_pings"] == 1                       # 0 today: the ping vanishes from the count


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
    # Two frames swapped: there is no reordering. The later one arrives first and opens a gap of one frame (lost,
    # the ping it cuts dropped); the earlier one then has a lower number and is ignored as a repeat.
    items = [item(k) for k in range(len(frames))]
    items[a], items[b] = items[b], items[a]
    records, s = receive(items)
    assert [r.metadata.sequence for r in records] == [0, 1, 3, 4, 5]
    assert (s["frames_lost"], s["frames_repeated"], s["trouble_frames"], s["lost_samples"], s["dropped_pings"],
            s["frames"]) == (1, 1, [b], FRAME, 1, len(frames) - 1)
    # The same swap with no metadata (no frame numbers): both frames are used. The assembler counts the jump to b
    # (one frame) and the jump from b back to b+1 when the stream continues (one more); the step back costs nothing.
    order = list(range(len(frames)))
    order[a], order[b] = order[b], order[a]
    records, s = receive([(frames[k], None) for k in order])
    assert [r.metadata.sequence for r in records] == [0, 1, 3, 4, 5]
    assert (s["frames"], s["frames_lost"], s["frames_repeated"], s["lost_samples"], s["dropped_pings"]) == \
        (len(frames), 0, 0, 2 * FRAME, 1)
    # A repeat with no frame number is not recognised: the step back is a restart to the assembler, which discards the
    # ping it falls in although no sample is missing (the numbered repeat above cost nothing).
    order = list(range(len(frames)))
    order.insert(a + 1, a)
    records, s = receive([(frames[k], None) for k in order])
    assert [r.metadata.sequence for r in records] == [0, 1, 3, 4, 5]
    assert (s["frames"], s["frames_repeated"], s["lost_samples"], s["dropped_pings"]) == (len(frames) + 1, 0, 0, 1)


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
    assert s == {"frames": len(frames) - 5, "frames_lost": 0, "frames_damaged": 2, "frames_repeated": 0, "bad_frames": 3,
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

