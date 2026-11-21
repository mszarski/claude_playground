"""The multiscale detector's statistic on the device (libs/live detector.hpp, multiscale_cells with backend cuda) held to
the host reference on the maps the C++ chain detects on in the benchmark's live case (tools/gpu_benchmark_all.py, group
live: the stacked-row free-field scenario with popoto_long_range[_2row|_3row] and the deployment files' chains).

The maps are those the Python engine hands its detector (its fine map, cell means, window and plan, captured on the way
in), which the engine-parity tests hold to the C++ chain's. On each:
  - the C++ detector on the host gives the NumPy one's detections (the binding is the chain's detector);
  - on a CUDA build with a device, the statistic on the device gives the host's flags and strongest samples, its
    background within 1e-12 of the host's (relative), and the same detections. Skipped otherwise, with the reason.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import gpu_benchmark_all as gba  # noqa: E402

from gpusonar import core, detect  # noqa: E402
from gpusonar.config import load_yaml  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402

SONARS = {1: "popoto_long_range", 2: "popoto_long_range_2row", 3: "popoto_long_range_3row"}
PINGS = 2


def _maps(rows: int, monkeypatch) -> list[dict]:
    chain = load_yaml(ROOT / "configs" / "deploy" / ("popoto_vehicle.yaml" if rows == 1 else "popoto_vehicle_elevation.yaml"))["chain"]
    chain.pop("sync", None)                       # the simulated pings carry no reference (as the benchmark does)
    cfg = dict(chain, engine="python", doa="map", beamformer="backprojection", backend="host")
    seen = []
    real = detect.multiscale

    def capture(fine, cell_mean, fine_ranges, cell_ranges, bearings, guard, train, plan):
        hits, background = real(fine, cell_mean, fine_ranges, cell_ranges, bearings, guard, train, plan)
        seen.append(dict(fine=np.ascontiguousarray(fine, dtype=np.float32), cell_mean=np.ascontiguousarray(cell_mean, dtype=np.float32),
                         fine_ranges=[float(v) for v in fine_ranges], cell_ranges=[float(v) for v in cell_ranges],
                         bearings=[float(v) for v in bearings], guard=int(guard), train=int(train), plan=plan, numpy=hits))
        return hits, background

    monkeypatch.setattr(detect, "multiscale", capture)
    with LiveChain(cfg) as live:
        for r in gba.live_pings(SONARS[rows], PINGS):
            live.process(r)
    assert len(seen) == PINGS
    return seen


def _run(m: dict, backend: str) -> dict:
    return core.live_multiscale_detect(m["fine"], m["cell_mean"], m["fine_ranges"], m["cell_ranges"], m["bearings"], m["guard"], m["train"],
                                       m["plan"], backend=core.Backend.Cuda if backend == "cuda" else core.Backend.Host)


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(x["range_m"] == y["range_m"] and x["bearing"] == y["bearing"] and x["cells"] == y["cells"]
                                    and x["extent_m"] == y["extent_m"] and abs(x["snr_db"] - y["snr_db"]) < 1e-5 for x, y in zip(a, b))


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_the_device_statistic_gives_the_hosts_detections_on_the_benchmark_pings(rows, monkeypatch):
    maps = _maps(rows, monkeypatch)
    for m in maps:
        host = _run(m, "host")
        # The binding's detector is the chain's: the NumPy engine's detections on the same map.
        numpy = m["numpy"]
        assert len(host["detections"]) == len(numpy)
        for h, p in zip(host["detections"], numpy):
            assert h["range_m"] == pytest.approx(p.range, abs=1e-6)
            assert h["bearing"] == pytest.approx(p.bearing, abs=1e-9)
            assert h["snr_db"] == pytest.approx(p.snr_db, abs=1e-4)
    if not core.cuda_available():
        pytest.skip("the device comparison needs a CUDA build and a device (CUDA " + ("compiled, no device" if core.cuda_compiled() else "not compiled") + ")")
    worst, flagged, found = 0.0, 0, 0
    for m in maps:
        host, dev = _run(m, "host"), _run(m, "cuda")
        assert np.array_equal(host["flag"], dev["flag"])
        assert np.array_equal(host["peak"], dev["peak"])
        level = np.asarray(host["level"])
        worst = max(worst, float(np.max(np.abs(np.asarray(dev["level"]) - level) / np.maximum(np.abs(level), 1e-300))))
        assert _same(host["detections"], dev["detections"]), (host["detections"], dev["detections"])
        flagged += int(np.count_nonzero(host["flag"]))
        found += len(host["detections"])
    print(f"\n{rows} row(s), {len(maps)} pings: {flagged} flagged cells and {found} detections, the same on the device; "
          f"background's largest relative difference {worst:.3g}")
    assert worst < 1e-12
    assert found > 0
