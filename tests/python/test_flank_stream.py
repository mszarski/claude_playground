"""Milestone 11: the low-rate flank capture stream (32 channels, 24-bit, 64 kSPS) from an
emulated board, through blocks and UDP, back to ping records."""

import json

import numpy as np
import pytest

from gpusonar import flank
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator, PingAssembler, make_source
from gpusonar.sources import flank_stream as fs

OVERRIDES = {"receive": {"window": {"range": [100.0, 400.0]}}}
INTERVAL = 2.0           # s between transmits (the window is 0.4 s of range plus the 1 s pulse)


@pytest.fixture(scope="module")
def pings():
    sonar = SonarModel.load("flank_32", OVERRIDES)
    scene = Scene().add_points([[40.0, 250.0, 0.0]], target_strength_db=10.0)
    rng = np.random.default_rng(2)
    return sonar, [simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1500.0},
                                 sequence=i, noise_rms=1e-5, rng=rng) for i in range(3)]


def test_block_round_trip():
    rng = np.random.default_rng(0)
    counts = rng.integers(-(1 << 23), 1 << 23, size=(128, 32))
    counts[0, :4] = [-(1 << 23), (1 << 23) - 1, -1, 0]
    data = fs.encode_block(counts, sample_index=123456789012, tx_sample=123456780000, ping_index=77,
                           gain_code=5, flags=fs.FLAG_OVERFLOW, start_ns=1_700_000_000_123_456_789)
    assert len(data) == 128 + 128 * 32 * 3
    b = fs.decode_block(data)
    np.testing.assert_array_equal(b["counts"], counts)
    assert (b["sample_index"], b["tx_sample"], b["ping_index"], b["gain_code"], b["flags"], b["start_ns"]) == \
        (123456789012, 123456780000, 77, 5, 1, 1_700_000_000_123_456_789)
    # A full-size block for the Sensor Bridge data plane is a multiple of 128 bytes.
    assert len(fs.encode_block(np.zeros((6400, 32), dtype=np.int32), 0)) % 128 == 0
    with pytest.raises(fs.BlockError, match="24-bit"):
        fs.encode_block(np.full((1, 32), 1 << 23), 0)
    with pytest.raises(fs.BlockError, match="magic"):
        fs.decode_block(b"XXXX" + data[4:])
    with pytest.raises(fs.BlockError, match="length"):
        fs.decode_block(data[:-3])


@pytest.mark.parametrize("block_samples", [128, 1000, 6400])
def test_assembler_recovers_the_pings(pings, block_samples):
    sonar, records = pings
    emulator = FlankBoardEmulator(records, INTERVAL, block_samples=block_samples, start_ns=10**18, gain_code=3)
    assembler = PingAssembler(sonar)
    out = [r for block in emulator.blocks() for r in assembler.feed(block)]
    assert len(out) == 3 and assembler.dropped_pings == 0 and assembler.lost_samples == 0
    lsb = records[0].metadata.adc_full_scale / 2**23
    for i, (got, want) in enumerate(zip(out, records)):
        assert got.samples.array.shape == want.samples.array.shape
        assert np.abs(got.samples.array - want.samples.array).max() <= 0.51 * lsb + 1e-12
        assert got.samples.start_time == pytest.approx(want.samples.start_time, abs=0.5 / 64000.0)
        assert got.metadata.sequence == i
        assert got.transmit.time_ns == 10**18 + int(round(i * INTERVAL * 1e9))
        assert json.loads(got.metadata.extra_json)["flank_stream"]["gain_code"] == 3
    # The recovered ping goes through the same processing: the target is where it was.
    # (Beamforming converts a record in place, so it gets copies.)
    a = flank.farfield_beamform(out[1].copy(), np.radians([0.0, 9.1])).valid()
    b = flank.farfield_beamform(records[1].copy(), np.radians([0.0, 9.1])).valid()
    assert np.argmax(np.abs(a.data[1])) == np.argmax(np.abs(b.data[1]))


def test_a_lost_block_costs_one_ping_only(pings):
    sonar, records = pings
    emulator = FlankBoardEmulator(records, INTERVAL, block_samples=128)
    blocks = list(emulator.blocks())
    period_blocks = int(INTERVAL * 64000) // 128
    lost = period_blocks + 200                       # inside the second ping's receive window
    assembler = PingAssembler(sonar)
    out = [r for i, block in enumerate(blocks) if i != lost for r in assembler.feed(block)]
    assert [r.metadata.sequence for r in out] == [0, 2]
    assert assembler.dropped_pings == 1 and assembler.lost_samples == 128
    # A block lost between windows costs nothing.
    assembler = PingAssembler(sonar)
    out = [r for i, block in enumerate(blocks) if i != 10 for r in assembler.feed(block)]
    assert len(out) == 3 and assembler.dropped_pings == 0 and assembler.lost_samples == 128
    # Joining a stream in the middle of a window: that ping is skipped, the next is whole.
    assembler = PingAssembler(sonar)
    out = [r for block in blocks[150:] for r in assembler.feed(block)]
    assert [r.metadata.sequence for r in out] == [1, 2] and assembler.dropped_pings == 1


def test_wrong_channel_count_is_refused(pings):
    sonar, _ = pings
    with pytest.raises(fs.BlockError, match="channels"):
        PingAssembler(sonar).feed(fs.encode_block(np.zeros((16, 8), dtype=np.int32), 0))


def test_emulated_board_over_udp(pings):
    sonar, records = pings
    source = make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": OVERRIDES,
                          "max_pings": 3, "timeout": 5.0})
    try:
        emulator = FlankBoardEmulator(records, INTERVAL, block_samples=128)
        sender = emulator.send(source.port, pace=0.1)        # paced: small socket buffers drop bursts
        out = list(source)
        sender.join(timeout=30.0)
    finally:
        source.close()
    assert source.bad_blocks == 0
    if source.assembler.lost_samples:                 # loopback UDP may drop under load: say so
        pytest.skip(f"the loopback dropped {source.assembler.lost_samples} samples")
    assert len(out) == 3
    lsb = records[0].metadata.adc_full_scale / 2**23
    for got, want in zip(out, records):
        assert np.abs(got.samples.array - want.samples.array).max() <= 0.51 * lsb + 1e-12
