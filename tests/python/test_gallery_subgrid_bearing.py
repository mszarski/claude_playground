"""tools/gallery.py subgrid_bearings: the Popoto runner's opt-in sub-grid bearing (sonar-4s2).

A hit of a few cells holds one or two rows of the 0.5 deg bearing grid, and detect.cluster's power-weighted mean over
its flagged cells is then the grid's bearing (or the midpoint of two): up to 0.25 deg, 52 m at 12 km. The parabola
through the bearing profile in dB about its strongest flagged row, the unflagged neighbours included, is not tied to
the grid."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gallery():
    spec = importlib.util.spec_from_file_location("gallery_subgrid_under_test", ROOT / "tools" / "gallery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("p", [-0.5, -0.31, 0.0, 0.07, 0.42])
def test_the_vertex_of_a_parabola_through_three_samples(gallery, p):
    y = [-3.0 * (u - p) ** 2 + 7.0 for u in (-1.0, 0.0, 1.0)]
    assert gallery.parabola_vertex(*y) == pytest.approx(p, abs=1e-12)


def test_the_vertex_is_zero_without_a_peak_and_clipped_to_the_cell(gallery):
    assert gallery.parabola_vertex(1.0, 1.0, 1.0) == 0.0          # flat
    assert gallery.parabola_vertex(0.0, 1.0, 2.0) == 0.0          # a slope: no curvature
    assert gallery.parabola_vertex(1.0, 0.0, 1.0) == 0.0          # a minimum
    assert gallery.parabola_vertex(-10.0, 0.0, 0.5) == 0.5        # vertex beyond the cell: held at its edge


def _map(bearing_deg, step_deg=0.5):
    """A noise-free map of one point echo through the gallery's Popoto beamformer path (farfield_beamform, Hann, power
    in 2 m range cells) on a grid of `step_deg`, the point at `bearing_deg` from broadside."""
    from gpusonar import flank
    from gpusonar.sim import Scene, make_pose, simulate_ping
    from gpusonar.sonar import SonarModel
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [800.0, 1200.0]}},
                                         "transmit": {"waveform": {"type": "lfm", "f_start": 6000.0, "f_stop": 10000.0,
                                                                   "duration": 0.01, "window_alpha": 0.05,
                                                                   "source_level_db_rms": 200.0}}})
    q = 1000.0 * np.array([np.sin(np.radians(bearing_deg)), np.cos(np.radians(bearing_deg)), 0.0])
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), Scene().add_points([q], target_strength_db=10.0),
                        environment={"sound_speed": 1500.0}, sequence=0, time_ns=0, noise_rms=1e-6,
                        rng=np.random.default_rng(1))
    bearings = np.radians(np.arange(-20.0, 20.001, step_deg))
    bm = flank.farfield_beamform(rec, bearings, window="hann").valid()
    fine = np.abs(bm.data) ** 2
    k = max(int(round(2.0 / (bm.ranges[1] - bm.ranges[0]))), 1)
    n = fine.shape[1] // k * k
    power = fine[:, :n].reshape(len(bearings), -1, k).mean(axis=2)
    ranges = bm.ranges[:n].reshape(-1, k).mean(axis=1)
    return power, ranges, bearings


@pytest.mark.parametrize("truth", [2.2, -7.13, 0.37])
def test_a_hit_of_few_cells_is_placed_off_the_grid(gallery, truth):
    """Only the strongest cell flagged (a low-SNR hit of one row): the cluster's centroid is that row's grid bearing,
    up to 0.25 deg off; the parabola lies within 0.02 deg of the point."""
    from gpusonar import detect
    power, ranges, bearings = _map(truth)
    flags = power >= power.max()
    hits = detect.cluster(flags, power, ranges, bearings, guard=4, train=16)
    assert len(hits) == 1 and hits[0].cells == 1
    got = np.degrees(gallery.subgrid_bearings(flags, power, bearings, hits)[0])
    centroid = np.degrees(hits[0].bearing)
    assert abs(got - truth) < 0.02, (got, truth)
    assert abs(centroid - truth) > 0.05, (centroid, truth)       # the grid's error, which the parabola removes


def test_a_whole_main_lobe_agrees_with_the_centroid(gallery):
    """Flagged over the main lobe (a strong hit), both estimates lie on the point: the option changes little there."""
    from gpusonar import detect
    power, ranges, bearings = _map(3.31)
    flags = power > 10 ** (-0.6) * power.max()
    hits = detect.cluster(flags, power, ranges, bearings, guard=4, train=16)
    got = np.degrees(gallery.subgrid_bearings(flags, power, bearings, hits)[0])
    assert abs(got - 3.31) < 0.02 and abs(np.degrees(hits[0].bearing) - 3.31) < 0.05


def test_hits_must_match_the_flagged_groups(gallery):
    from gpusonar import detect
    power, ranges, bearings = _map(1.0)
    flags = power > 0.5 * power.max()
    hits = detect.cluster(flags, power, ranges, bearings, guard=4, train=16)
    with pytest.raises(ValueError, match="groups"):
        gallery.subgrid_bearings(flags, power, bearings, hits + hits)


def test_the_default_is_the_centroid(gallery):
    assert gallery.POPOTO_BEARING == "centroid"


def test_the_study_tool_estimates_from_recorded_profiles_as_the_gallery_does(gallery):
    """tools/popoto_bearing_study.py reads the profiles classify_rules_eval --profiles recorded (in dB to 1e-3) and gives
    the bearing subgrid_bearings gives from the map, within what that rounding allows; `centroid` is cluster's own and
    `grid` the strongest flagged row's."""
    from gpusonar import detect
    spec = importlib.util.spec_from_file_location("cre_for_study", ROOT / "tools" / "classify_rules_eval.py")
    cre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cre)
    spec = importlib.util.spec_from_file_location("popoto_bearing_study", ROOT / "tools" / "popoto_bearing_study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    power, ranges, bearings = _map(-4.62)
    flags = power > 0.2 * power.max()
    hits = detect.cluster(flags, power, ranges, bearings, guard=4, train=16)
    want = gallery.subgrid_bearings(flags, power, bearings, hits)
    rec = cre.cluster_profiles(flags, power, ranges, bearings, hits)
    step = bearings[1] - bearings[0]
    for h, p, w in zip(hits, rec, want):
        assert study.estimate(p, bearings[0], step, "parabola") == pytest.approx(w, abs=np.radians(1e-3))
        assert study.estimate(p, bearings[0], step, "centroid") == h.bearing
        assert study.estimate(p, bearings[0], step, "grid") == pytest.approx(bearings[p["rows"][int(np.argmax([p["sum"][r - p["lo"]] for r in p["rows"]]))]])
    assert np.degrees(study.estimate(rec[0], bearings[0], step, "parabola")) == pytest.approx(-4.62, abs=0.02)
