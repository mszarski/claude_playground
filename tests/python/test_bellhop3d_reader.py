"""P2 of docs/tasks/PROPAGATION_3D.md: Bellhop3D's arrivals read back, and its beam contributions merged into
eigenrays (sim/bellhop3d.py).

Fixtures (no program needed): a hand-written ASCII arrivals file and the same content as a Fortran unformatted
('a') file, read back exactly; the merge on synthetic hat-beam contributions. Oracle (Bellhop3D,
GPUSONAR_BELLHOP3D): on a level bed, the merged arrivals of a 3-D hat-beam run against the same run's own
incoherent TL, and what Gaussian beams do instead.
"""

import os
import struct
import subprocess
import tempfile

import numpy as np
import pytest

from gpusonar.sim import bellhop3d as b3
from bellhop3d_oracle import read_shade_bearings

needs_bellhop3d = pytest.mark.skipif(b3.find_bellhop3d() is None,
                                     reason="Bellhop3D not installed (GPUSONAR_BELLHOP3D; bellhop3d.exe beside the Bellhop executable is found too)")

# Two bearings x one depth x one range; the first receiver holds two arrivals, the second one.
ROWS = {(0, 0, 0): [[2.0e-3, 180.0, 0.6667, 0.0, -5.0, 0.0, 5.0, 0.5, 1, 0], [1.0e-3, 0.0, 0.6700, -1e-6, 10.0, -1.0, -10.0, -1.5, 0, 1]],
        (1, 0, 0): [[3.0e-3, 90.0, 0.6680, 0.0, 2.0, 30.0, -2.0, 29.0, 2, 2]]}


def _ascii_fixture(path):
    lines = ["'3D'", "  8000.0000", " 1  0.0", " 1  0.0", " 1  50.0", " 1  90.0", " 1  1000.0", " 2  0.0  30.0", " 2"]
    for key in ((0, 0, 0), (1, 0, 0)):
        lines.append(f" {len(ROWS[key])}")
        lines += ["  " + "  ".join(repr(float(v)) if i < 8 else str(int(v)) for i, v in enumerate(r)) for r in ROWS[key]]
    open(path, "w").write("\n".join(lines) + "\n")


def _record(payload: bytes) -> bytes:
    return struct.pack("<i", len(payload)) + payload + struct.pack("<i", len(payload))


def _binary_fixture(path):
    rec = [_record(b"'3D'"), _record(struct.pack("<f", 8000.0))]
    rec += [_record(struct.pack("<id", 1, 0.0)), _record(struct.pack("<id", 1, 0.0)), _record(struct.pack("<if", 1, 50.0))]
    rec += [_record(struct.pack("<if", 1, 90.0)), _record(struct.pack("<id", 1, 1000.0)), _record(struct.pack("<idd", 2, 0.0, 30.0))]
    rec.append(_record(struct.pack("<i", 2)))
    for key in ((0, 0, 0), (1, 0, 0)):
        rec.append(_record(struct.pack("<i", len(ROWS[key]))))
        for r in ROWS[key]:
            rec.append(_record(struct.pack("<df2f4f2f", r[0], r[1], r[2], r[3], *r[4:8], float(r[8]), float(r[9]))))
    open(path, "wb").write(b"".join(rec))


@pytest.mark.parametrize("kind", ["ascii", "binary"])
def test_fixture_is_read_back_exactly(kind, tmp_path):
    path = str(tmp_path / "run.arr")
    (_ascii_fixture if kind == "ascii" else _binary_fixture)(path)
    arr = b3.read_arrivals3d(path)
    assert arr.frequency == 8000.0 and np.allclose(arr.source, [0.0, 0.0, 50.0])
    assert np.allclose(arr.depths, [90.0]) and np.allclose(arr.ranges, [1000.0]) and np.allclose(arr.bearings, [0.0, 30.0])
    assert list(arr.counts()) == [2, 1]
    tol = dict(rtol=1e-6, atol=0.0) if kind == "binary" else dict(rtol=1e-12, atol=0.0)
    for key, rows in ROWS.items():
        rows = np.array(rows, dtype=float)
        # The suite's exp(+i omega t) convention: Bellhop's phase conjugated, as sim/bellhop.py does.
        assert np.allclose(arr.amplitude[key], rows[:, 0] * np.exp(-1j * np.radians(rows[:, 1])), **tol)
        assert np.allclose(arr.delay[key], rows[:, 2], **tol)
        assert np.allclose(np.degrees(arr.source_decl[key]), rows[:, 4], **tol)
        assert np.allclose(np.degrees(arr.source_azim[key]), rows[:, 5], **tol)
        assert np.allclose(np.degrees(arr.receiver_decl[key]), rows[:, 6], **tol)
        assert np.allclose(np.degrees(arr.receiver_azim[key]), rows[:, 7], **tol)
        assert list(arr.n_surface[key]) == list(rows[:, 8].astype(int)) and list(arr.n_bottom[key]) == list(rows[:, 9].astype(int))


def _synthetic(contributions):
    """An Arrivals3D with one receiver holding `contributions` = [(amp, delay, sdecl, sazim, rdecl, razim, ns, nb)]
    (angles in degrees)."""
    arr = b3.Arrivals3D(8000.0, np.zeros(3), np.array([50.0]), np.array([1000.0]), np.array([0.0]))
    c = np.array(contributions, dtype=complex)
    key = (0, 0, 0)
    arr.amplitude[key], arr.delay[key] = c[:, 0], c[:, 1].real
    for name, col in (("source_decl", 2), ("source_azim", 3), ("receiver_decl", 4), ("receiver_azim", 5)):
        setattr(arr, name, {key: np.radians(c[:, col].real)})
    arr.n_surface[key], arr.n_bottom[key] = c[:, 6].real.astype(int), c[:, 7].real.astype(int)
    return arr, key


def test_merge_joins_the_neighbouring_beams_of_one_family_and_keeps_families_apart():
    """A family arrives as four hat contributions (weights 0.42, 0.18, 0.28, 0.12: sum one) from the beams around its
    launch angle, with a small spread of phase and delay; a second family with the SAME bounce counts, delay and
    arrival angle left the source 4 degrees away (over relief that happens); a third has other counts. Merged: three
    eigenrays, each with its family's amplitude (the magnitudes add: 1e-3 and 2e-3), the delay energy-weighted."""
    da, db = 0.25, 0.33
    fam = [(1e-3 * w * np.exp(1j * 0.1 * k), 1.0 + 1e-5 * k, 5.0 + da * (k % 2), 1.0 + db * (k // 2), -5.0, 1.0, 1, 1)
           for k, w in enumerate([0.42, 0.18, 0.28, 0.12])]
    other = [(2e-3 * w, 1.0, 9.0 + da * (k % 2), 1.0 + db * (k // 2), -5.0, 1.0, 1, 1) for k, w in enumerate([0.5, 0.5, 0.0, 0.0])]
    third = [(5e-4, 1.2, -3.0, 0.0, 3.0, 0.0, 2, 1)]
    arr, key = _synthetic(fam + other + third)
    m = b3.merge_arrivals(arr, key, np.radians(da), np.radians(db))
    assert len(m.amplitude) == 3 and sorted(m.contributions.tolist()) == [1, 4, 4]
    assert np.allclose(sorted(np.abs(m.amplitude)), [5e-4, 1e-3, 2e-3])
    first = int(np.argmin(np.abs(np.abs(m.amplitude) - 1e-3)))
    w = np.array([0.42, 0.18, 0.28, 0.12]) ** 2
    assert m.delay[first] == pytest.approx(np.sum(w * (1.0 + 1e-5 * np.arange(4))) / w.sum())
    assert np.degrees(m.receiver_decl[first]) == pytest.approx(-5.0) and m.n_surface[first] == 1
    assert np.angle(m.amplitude[first]) == pytest.approx(np.angle(np.sum([c[0] for c in fam])))
    # One contribution only: unchanged.
    arr1, key1 = _synthetic(third)
    one = b3.merge_arrivals(arr1, key1, np.radians(da), np.radians(db))
    assert one.amplitude[0] == pytest.approx(5e-4) and one.delay[0] == 1.2
    empty = b3.Arrivals3D(8000.0, np.zeros(3), np.array([50.0]), np.array([1.0]), np.array([0.0]))
    empty.amplitude[key], empty.delay[key] = np.zeros(0, complex), np.zeros(0)
    assert len(b3.merge_arrivals(empty, key, 0.01, 0.01).amplitude) == 0


# --- against Bellhop3D's own TL run ----------------------------------------------------------------------------------

D, C, BOTTOM, F, ZS = 200.0, 1500.0, (1700.0, 1.9, 0.5), 8000.0, 50.0
BEARINGS, DEPTHS, RANGES = [0.0, 2.0], [50.0, 90.0], [1000.0, 3000.0, 5000.0]


def _level_run(run_type, beam, n_alpha, n_beta=61, beta=10.0):
    flat = lambda x, y: np.full(np.broadcast(x, y).shape, D)                     # noqa: E731
    with tempfile.TemporaryDirectory(prefix="gpusonar_b3d_") as tmp:
        b3.write_bty3d(os.path.join(tmp, "run.bty"), flat, np.arange(-1000.0, 6001.0, 500.0), np.arange(-2000.0, 2001.0, 500.0))
        b3.write_env3d(os.path.join(tmp, "run.env"), frequency=F, profile=([0.0, D], [C, C]), depth=D, bottom=BOTTOM,
                       source_depth=ZS, receiver_depths=DEPTHS, ranges=RANGES, bearings_deg=BEARINGS, run_type=run_type,
                       mode="3", n_alpha=n_alpha, alpha_deg=(-30.0, 30.0), n_beta=n_beta, beta_deg=(-beta, beta),
                       box_km=(6.0, 2.0), beam=beam)
        subprocess.run([b3.find_bellhop3d(), "run"], cwd=tmp, capture_output=True, check=True, timeout=300)
        if run_type in ("A", "a"):
            path = os.path.join(tmp, "run.arr")
            return b3.read_arrivals3d(path), os.path.getsize(path)
        return -20.0 * np.log10(np.abs(read_shade_bearings(os.path.join(tmp, "run.shd"))[0])), None


@needs_bellhop3d
def test_merged_hat_beam_arrivals_hold_the_tl_run_over_a_level_bed(capsys):
    """3-D mode, hat beams, 241 x 61 over +-30 and +-10 degrees, 12 receivers (2 bearings x 2 depths x 1, 3, 5 km).
    Measured: the ASCII ('A') and binary ('a') files hold the same arrivals (170 and 52 bytes an arrival); raw counts
    up to 129 a receiver merge to at most 30 (none beyond the 64 strongest); the merged arrivals summed in power are
    within 0.05 dB of the run's own incoherent TL, which is within 0.15 dB of the image sources (test_bellhop_nx2d_
    agreement.py) from 1 km. Bounds: 0.1 dB; at most 64 eigenrays with under 1 % of the energy beyond them. 3.5 s a run."""
    tl, _ = _level_run("I", "G", 241)
    ascii_arr, ascii_size = _level_run("A", "G", 241)
    arr, size = _level_run("a", "G", 241)
    diffs, counts, merged_counts = [], [], []
    for key in arr.keys():
        assert np.allclose(arr.amplitude[key], ascii_arr.amplitude[key], rtol=1e-5, atol=1e-12)
        assert np.allclose(arr.delay[key], ascii_arr.delay[key], rtol=1e-6)
        m = b3.merge_arrivals(arr, key, np.radians(60.0 / 240), np.radians(20.0 / 60))
        energy = np.sort(np.abs(m.amplitude) ** 2)[::-1]
        assert energy[64:].sum() < 0.01 * energy.sum()
        diffs.append(-10.0 * np.log10(energy.sum()) - tl[key])
        counts.append(len(arr.amplitude[key]))
        merged_counts.append(len(m.amplitude))
    with capsys.disabled():
        print(f"\n[bellhop3d] level bed, hat beams: merged arrivals minus TL run, max |diff| {np.abs(diffs).max():.3f} dB; "
              f"counts {max(counts)} -> {max(merged_counts)}; {ascii_size / arr.counts().sum():.0f} bytes an arrival ASCII, "
              f"{size / arr.counts().sum():.0f} binary")
    assert np.abs(diffs).max() < 0.1
    assert max(merged_counts) <= 64 and max(counts) > 64


@needs_bellhop3d
def test_gaussian_beam_arrivals_do_not_hold_the_field(capsys):
    """The same receivers with Gaussian beams ('B', 481 x 61): the run's TL is right (within 0.2 dB of the hat run),
    but its arrivals are thousands of overlapping contributions whose weights do not sum to one: summed in power they
    are 8.7 to 11.0 dB below the TL (measured). The provider therefore runs hat beams. Bound: more than 5 dB below."""
    tl_hat, _ = _level_run("I", "G", 241)
    tl, _ = _level_run("I", "B", 481)
    arr, _ = _level_run("a", "B", 481)
    raw = np.array([-10.0 * np.log10(arr.energy(key)) - tl[key] for key in arr.keys()])
    with capsys.disabled():
        print(f"\n[bellhop3d] level bed, Gaussian beams: raw arrivals minus TL run {raw.min():+.1f} to {raw.max():+.1f} dB; "
              f"TL Gaussian minus hat max |diff| {np.abs(tl - tl_hat).max():.2f} dB; up to {arr.counts().max()} arrivals")
    assert np.abs(tl - tl_hat).max() < 0.2
    assert raw.min() > 5.0
