"""Write tests/python/data/centroid_d158.json: the live chain's report lines, both engines, on two short scenarios, made
with the code from before D-158 (where the range estimate was the centroid by default) and no range_estimate given.
test_detector.py::test_range_estimate_centroid_given_is_the_output_from_before_d158 runs `reports` of this file on the
current code with `cfar.range_estimate: centroid` given and wants these lines byte for byte.

    python3 tests/python/data/make_centroid_d158.py <root of the code to run>   # e.g. a `git archive 42d4539` build
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
GOLDEN = HERE / "centroid_d158.json"

# test_live.py's scene: ownship making 1.5 m/s along a flank array, one target crossing at 3 m/s, one standing still.
FLANK_CHAIN = {"bearings_deg": [-40.0, 40.0, 1.0], "range_cell_m": 1.0, "cfar": {"guard_m": 15.0, "train_m": 50.0},
               "min_snr_db": 12.0, "backend": "host",
               "tracker": {"range_sigma": 3.0, "bearing_sigma_deg": 1.0, "confirm_hits": 3, "max_misses": 2},
               "classify": {"min_speed": 1.0, "static_speed": 0.6, "max_fit_rms": 10.0, "multipath": None}}
# test_detector.py's long-range sonar: a weak first arrival and five 6 dB stronger ones 10 to 50 m behind it, and a
# second target, over four pings.
# D-256: the shipped popoto_long_range sends a 2 s hyperbolic sweep (blind inside 1.5 km); the golden lines were made with its
# former 1 s linear sweep, stated here so that they stay what they were (an older tree reads the same override as its default).
POPOTO_WINDOW = {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1700.0]}},
                 "transmit": {"waveform": {"type": "lfm", "source": None, "duration": 1.0}}}
POPOTO_CHAIN = {"backend": "host"}


def _flank_pings():
    from gpusonar.sim import Scene, make_pose, simulate_ping
    from gpusonar.sonar import SonarModel
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [100.0, 400.0]}}})
    rng = np.random.default_rng(5)
    out = []
    for i in range(6):
        t = i * 2.0
        scene = Scene().add_points([[40.0 - 3.0 * t, 250.0, 0.0], [-60.0, 320.0, 0.0]], target_strength_db=10.0)
        out.append(simulate_ping(sonar, make_pose(position=(1.5 * t, 0.0, 0.0), velocity=(1.5, 0.0, 0.0)), scene,
                                 environment={"sound_speed": 1500.0}, sequence=i, time_ns=int(t * 1e9), noise_rms=1e-5, rng=rng))
    return out


def _popoto_pings():
    from gpusonar.sim import Scene, make_pose, simulate_ping
    from gpusonar.sonar import SonarModel
    sonar = SonarModel.load("popoto_long_range", POPOTO_WINDOW)

    def at(r, bearing_deg):
        b = np.radians(bearing_deg)
        return [r * np.sin(b), r * np.cos(b), 0.0]

    out = []
    for i in range(4):
        scene = Scene()
        for dr, ts in zip((0.0, 10.0, 20.0, 30.0, 40.0, 50.0), (0.0, 6.0, 6.0, 6.0, 6.0, 6.0)):
            scene.add_points([at(1300.0 + dr + 4.0 * i, 5.0)], target_strength_db=ts)
        scene.add_points([at(1500.0, -12.0)], target_strength_db=3.0)
        out.append(simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=2e-4, rng=np.random.default_rng(300 + i),
                                 sequence=i, time_ns=i * 2_000_000_000, environment={"sound_speed": 1500.0}, adc=False,
                                 absorption_model="none"))
    return out


def reports(extra_cfar: dict | None = None) -> dict:
    """{scenario: {engine: [report line, ...]}}, each line as json.dumps(sort_keys=True) of the chain's report without
    `processing_seconds`."""
    from gpusonar.live import LiveChain
    out = {}
    for name, pings, chain in (("flank", _flank_pings(), FLANK_CHAIN), ("popoto", _popoto_pings(), POPOTO_CHAIN)):
        out[name] = {}
        for engine in ("cpp", "python"):
            cfg = json.loads(json.dumps(chain))
            cfg["engine"] = engine
            if extra_cfar:
                cfg.setdefault("cfar", {}).update(extra_cfar)
            with LiveChain(cfg) as c:
                lines = c.run([p.copy() for p in pings])
            # Everything but the wall-clock time the ping took.
            out[name][engine] = [json.dumps({k: v for k, v in r.items() if k != "processing_seconds"}, sort_keys=True) for r in lines]
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve() / "python"))
    got = reports()
    GOLDEN.write_text(json.dumps(got, indent=1) + "\n")
    for name, engines in got.items():
        for engine, lines in engines.items():
            print(name, engine, [len(json.loads(line)["detections"]) for line in lines], [len(json.loads(line)["tracks"]) for line in lines])
