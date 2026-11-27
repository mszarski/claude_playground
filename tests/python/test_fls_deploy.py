"""sonar-tpex (D-255, the owner: "yes"): the forward-looking sonars' own deployment files, and `look: boresight`.

The live chain's default `look: starboard` is a flank array's: broadside is the array's axis (first element to last)
turned 90 degrees clockwise seen from above. A forward-looking sonar's array is across the bow with its elements
numbered port to starboard, so that broadside is astern: every echo was placed behind the sonar, mirrored fore and aft
(D-253: tracks 150 to 230 m astern). `look: port` happened to give ahead for that numbering. `look: boresight` (opt-in,
both engines) takes the side the record's elements face, every ping. The FLS deployment files (configs/deploy/fls_*.yaml,
l12_5d.yaml, p4_2v.yaml) say it, with the bearing fan, cells, CFAR windows and tracker and classifier settings of an FLS.
Nothing else moves: `starboard` and `port` are as before (the report hashes of two flank / long-range scenes in both
engines, D-256's proposal), and the gallery's FLS settings, now read from the files, are the numbers it typed before.
"""

import copy
import importlib.util
import inspect
import json
import pathlib
import subprocess

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import live
from gpusonar.config import load_scenario_config, load_yaml
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, run_scenario, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources.base import _REGISTRY

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "configs" / "deploy"
EXE = ROOT / "build" / "bin" / "sonar_live"
FLS_FILES = {"fls_120k", "fls_170k", "fls_330k", "l12_5d", "p4_2v"}
GALLERY_FILES = ("fls_120k", "fls_170k", "fls_330k")      # the transit sonars: the gallery's settings
STRIP = ("processing_seconds",)


def strip(r):
    return {k: v for k, v in r.items() if k not in STRIP}


def gallery():
    spec = importlib.util.spec_from_file_location("_gallery_for_fls_deploy", ROOT / "tools" / "gallery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the files ---------------------------------------------------------------------------------------------------------

def fls_deploy_files() -> dict:
    """Every deployment file whose source names a forward-looking sonar (found by the sonar's mode, not by a list)."""
    out = {}
    for p in sorted(DEPLOY.glob("*.yaml")):
        cfg = load_yaml(p)
        if SonarModel.load(cfg["source"]["sonar"]).config.get("mode") == "fls":
            out[p.stem] = cfg
    return out


def test_every_fls_deployment_file_loads_and_holds_only_keys_both_engines_read():
    files = fls_deploy_files()
    assert set(files) == FLS_FILES
    for name, cfg in files.items():
        assert set(cfg) <= {"source", "chain", "output", "nav", "unconfirmed"}, name
        src = dict(cfg["source"])
        kind = src.pop("type")
        assert set(src) <= set(inspect.signature(_REGISTRY[kind].__init__).parameters) - {"self"}, name   # the source takes them
        assert src["sonar"] == name and cfg["unconfirmed"] == ["source"]
        chain = cfg["chain"]
        assert chain["look"] == "boresight", name
        o = live.cpp_chain_options(chain)                 # the Python engine's key and word checks (check_keys, check_choices)
        assert o["look_boresight"] is True
        core.LiveChainCpp(o)                               # the C++ engine refuses a key it does not read
        for engine in ("python", "cpp"):
            with LiveChain(dict(chain, engine=engine, backend="host")):
                pass
        flat = live.program_options(chain, None, cfg)      # build/bin/sonar_live's key=value options read back by its parser
        back = json.loads(core.live_options_from_text("\n".join(f"{k}={v}" for k, v in flat.items())))
        assert flat["look"] == "boresight" and back["look"] == "boresight"
        b = chain["bearings_deg"]
        assert b[0] == -b[1] and abs(round(b[1] / b[2]) * b[2] - b[1]) < 1e-9, name          # symmetric, whole steps


def test_the_gallery_reads_its_fls_settings_from_the_files_and_they_are_the_numbers_it_typed():
    """Bit for bit the values tools/gallery.py typed before sonar-tpex (FLS_TRACKER, FLS_TRANSIT's cell and windows, the
    detector's 1e-7 and 13 dB, the fan, the classifier's speed and fit): its FLS results cannot move by the change."""
    g = gallery()
    before_tracker = dict(range_sigma=1.5, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0,
                          process_noise=0.3)
    for sonar in GALLERY_FILES:
        s = g.fls_settings(sonar)
        assert s["tracker"] == before_tracker
        assert (s["pfa"], s["guard_m"], s["train_m"], s["min_snr_db"]) == (1e-7, 6.0, 30.0, 13.0)
        assert s["classify"] == {"min_speed": 0.5, "static_speed": 0.5, "max_fit_rms": 2.5}   # static_speed: classify_tracks' default
        b0, b1, step = s["bearings_deg"]
        assert np.array_equal(np.radians(np.arange(b0, b1 + 0.01, step)), np.radians(np.arange(-60.0, 60.01, 0.4)))
    cells = {"fls_120k_transit": 8, "fls_170k_transit": 8, "fls_330k_transit": 12, "fls_120k_cluttered_seabed": 8,
             "fls_120k_rough_surface": 8, "fls_170k_traffic": 8}            # samples a cell before (FLS_TRANSIT["cell"])
    for scene, cell in cells.items():
        sonar = load_scenario_config(scene)["sonar"]
        step = g.FLS_TRANSIT[scene]["r"][2]
        rc = g.fls_settings(sonar)["range_cell_m"]
        assert int(round(rc / step)) == cell and int(round(rc / (2 * step))) == cell // 2        # full and --quick
        assert int(round(6.0 / (cell * step))) == int(round(g.fls_settings(sonar)["guard_m"] / (cell * step)))


# --- look: boresight ---------------------------------------------------------------------------------------------------

FLS_WINDOW = {"receive": {"window": {"range": [40.0, 80.0]}}}


@pytest.fixture(scope="module")
def fls_point_pings():
    """fls_120k, ownship still, heading north: a point 60 m ahead and 10 m to starboard, three pings."""
    sonar = SonarModel.load("fls_120k", FLS_WINDOW)
    rng = np.random.default_rng(3)
    scene = Scene().add_points([[60.0, 10.0, 0.0]], target_strength_db=0.0)
    return [simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1500.0}, sequence=i, time_ns=int(i * 0.5e9),
                          noise_rms=1e-6, rng=rng) for i in range(3)]


FLS_SMALL = {"bearings_deg": [-60.0, 60.0, 0.5], "range_cell_m": 0.48, "cell_statistic": "mean",
             "cfar": {"pfa": 1e-7, "guard_m": 6.0, "train_m": 30.0}, "min_snr_db": 13.0, "backend": "host",
             "tracker": {"range_sigma": 1.5, "bearing_sigma_deg": 0.4}, "classify": {"multipath": None}}


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_look_boresight_puts_a_forward_looking_sonars_echo_ahead(fls_point_pings, engine):
    """Fails before sonar-tpex: `boresight` was not a word of `look`. The default puts the point 60 m astern (mirrored
    fore and aft, same side), `port` and `boresight` 60 m ahead, where it is."""
    def first(look):
        cfg = dict(FLS_SMALL, engine=engine, **({"look": look} if look else {}))
        with LiveChain(cfg) as chain:
            reps = [chain.process(r.copy()) for r in fls_point_pings]
        d = max(reps[0]["detections"], key=lambda d: d["snr_db"])
        return reps, d
    reps_b, d_b = first("boresight")
    _, d_def = first(None)
    reps_p, _ = first("port")
    assert d_b["range_m"] == pytest.approx(np.hypot(60.0, 10.0), abs=0.5)
    assert d_b["bearing_deg"] == pytest.approx(np.degrees(np.arctan2(10.0, 60.0)), abs=0.5)     # positive to starboard
    assert d_def["bearing_deg"] == pytest.approx(d_b["bearing_deg"], abs=1e-9)                   # the same bearing, ...
    tr = max(reps_b[-1]["tracks"], key=lambda t: t["mean_snr_db"])
    assert (tr["x"], tr["y"]) == pytest.approx((60.0, 10.0), abs=0.5)                              # ... placed ahead
    assert [strip(r) for r in reps_p] == [strip(r) for r in reps_b]                                  # port is the same side here
    with LiveChain(dict(FLS_SMALL, engine=engine)) as chain:
        astern = [chain.process(r.copy()) for r in fls_point_pings][-1]
    tr = max(astern["tracks"], key=lambda t: t["mean_snr_db"])
    assert (tr["x"], tr["y"]) == pytest.approx((-60.0, 10.0), abs=0.5)                             # the failure D-253 saw


def test_look_side_follows_the_elements_and_refuses_a_boresight_along_the_array(fls_point_pings):
    rec = fls_point_pings[0]
    assert live.look_side(rec, "boresight") == "port"          # elements port to starboard, facing ahead
    assert live.look_side(rec, "starboard") == "starboard" and live.look_side(rec, "port") == "port"
    flipped = rec.copy()                                         # the same array numbered starboard to port: still ahead
    g = flipped.geometry
    rx = list(g.rx)[::-1]
    g.rx = rx
    flipped.geometry = g
    assert live.look_side(flipped, "boresight") == "starboard"
    flank = SonarModel.load("flank_32", {"receive": {"window": {"range": [100.0, 120.0]}}}).make_record(make_pose())
    assert live.look_side(flank, "boresight") == "starboard"     # a flank array's elements face starboard: its default
    along = rec.copy()
    g = along.geometry
    els = list(g.rx)
    for e in els:
        e.boresight = (0.0, 1.0, 0.0)                            # along the array: no side
    g.rx = els
    along.geometry = g
    with pytest.raises(ValueError, match="look: boresight"):
        live.look_side(along, "boresight")
    for engine in ("python", "cpp"):
        with LiveChain(dict(FLS_SMALL, engine=engine, look="boresight")) as chain:
            with pytest.raises(ValueError, match="look: boresight"):
                chain.process(along.copy())


def test_look_boresight_is_opt_in_and_said_only_when_on():
    """Nothing of it reaches the C++ options or the program's command for starboard or port: every other line as before."""
    for look in ("starboard", "port", None):
        cfg = {"look": look} if look else {}
        assert "look_boresight" not in live.cpp_chain_options(cfg)
        assert live.program_options(cfg)["look"] == (look or "starboard")
    assert json.loads(core.live_options_from_text("look=port"))["look"] == "port"
    assert json.loads(core.live_options_from_text("look=boresight"))["look"] == "boresight"
    with pytest.raises(ValueError, match="look"):
        core.live_options_from_text("look=ahead")


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_look_boresight_on_a_flank_array_is_its_starboard_byte_for_byte(engine):
    """flank_32's elements face starboard: `boresight` is its default side, report for report."""
    from test_live import CHAIN, OVERRIDES
    sonar = SonarModel.load("flank_32", OVERRIDES)
    rng = np.random.default_rng(5)
    recs = [simulate_ping(sonar, make_pose(position=(1.5 * t, 0.0, 0.0), velocity=(1.5, 0.0, 0.0)),
                          Scene().add_points([[40.0 - 3.0 * t, 250.0, 0.0]], target_strength_db=10.0),
                          environment={"sound_speed": 1500.0}, sequence=i, time_ns=int(t * 1e9), noise_rms=1e-5, rng=rng)
            for i, t in enumerate((0.0, 2.0, 4.0, 6.0))]
    runs = []
    for look in ("starboard", "boresight"):
        with LiveChain(dict(CHAIN, engine=engine, look=look)) as chain:
            runs.append([strip(chain.process(r.copy())) for r in recs])
    assert runs[0] == runs[1] and runs[0][-1]["tracks"]


# --- a quick FLS scene through the chain with its file -----------------------------------------------------------------

@pytest.fixture(scope="module")
def transit():
    """tools/gallery.py's --quick fls_120k_transit (4 pings, thinned), simulated once; and its truth."""
    g = gallery()
    cfg = load_scenario_config("fls_120k_transit")
    g.QUICK["fls_120k_transit"](cfg)
    res = run_scenario(cfg)
    return res.records, g.scene_truth(cfg), float(res.times[-1])


def _run(records, chain, engine):
    with LiveChain(dict(copy.deepcopy(chain), engine=engine, backend="host")) as lc:
        return [lc.process(r.copy()) for r in records]


def _misses(report, truth, t_last):
    """For each true object, the horizontal distance (m) to its nearest track at the last ping."""
    out = {}
    for label, (pos, vel) in truth.items():
        q = pos[:2] + vel[:2] * t_last
        out[label] = min(float(np.hypot(t["x"] - q[0], t["y"] - q[1])) for t in report["tracks"])
    return out


TRUTH_MISS_M = 2.5      # m: the file's classify.max_fit_rms; measured 0.36 to 1.85 m (both engines alike)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_with_its_deployment_file_an_fls_tracks_ahead_and_near_the_truth(transit, engine):
    """The mirrored-astern failure (D-253) is the defaults' half: every track behind the sonar (measured 20 to 265 m
    astern on these pings, each object's nearest track 219 to 471 m from it). With configs/deploy/fls_120k.yaml every track is ahead and each of the scene's four objects
    (two boats, a buoy, a pinnacle) has one within TRUTH_MISS_M."""
    records, truth, t_last = transit
    own = lambda r: np.asarray(r["ownship_xyz"])  # noqa: E731
    bad = _run(records, {}, engine)[-1]
    assert bad["tracks"] and all(t["x"] - own(bad)[0] < 0.0 for t in bad["tracks"])          # every track astern
    assert min(_misses(bad, truth, t_last).values()) > 150.0                                   # measured 219 to 471 m
    good = _run(records, load_yaml(DEPLOY / "fls_120k.yaml")["chain"], engine)
    last = good[-1]
    assert len(last["tracks"]) >= len(truth) and all(t["x"] - own(last)[0] > 0.0 for t in last["tracks"])
    misses = _misses(last, truth, t_last)
    assert set(misses) == {"boat_under_way", "boat_at_anchor", "mooring_buoy", "pinnacle"}
    assert max(misses.values()) < TRUTH_MISS_M, misses


def test_the_engines_give_the_same_fls_report_with_the_file(transit):
    records, _, _ = transit
    chain = load_yaml(DEPLOY / "fls_120k.yaml")["chain"]
    a, b = _run(records, chain, "python"), _run(records, chain, "cpp")
    _same(a, b)


def _same(want, got):
    assert len(want) == len(got)
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"])
        for x, y in zip(w["detections"], g["detections"]):
            assert x["range_m"] == pytest.approx(y["range_m"], abs=0.05) and x["bearing_deg"] == pytest.approx(y["bearing_deg"], abs=0.05)
        assert [(t["id"], t["kind"]) for t in w["tracks"]] == [(t["id"], t["kind"]) for t in g["tracks"]]
        for x, y in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "range_m"):
                assert x[key] == pytest.approx(y[key], abs=0.05), key


needs_exe = pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")


@needs_exe
def test_sonar_live_with_the_fls_file_on_a_replay_equals_the_python_chain(transit, tmp_path):
    """build/bin/sonar_live with the file's chain as its key=value options (live.program_options, the mapping
    --export-template prints) on the recording of the Python chain's run: the same detections and tracks."""
    records, _, _ = transit
    deploy = load_yaml(DEPLOY / "fls_120k.yaml")
    with LiveChain(dict(deploy["chain"], engine="python", backend="host"), recording=str(tmp_path / "rec.h5")) as chain:
        want = [chain.process(r.copy()) for r in records]
    flat = live.program_options(deploy["chain"], None, deploy)
    assert flat["look"] == "boresight" and flat["cell_statistic"] == "mean"
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), *[f"{k}={v}" for k, v in flat.items()]],
                         capture_output=True, text=True, timeout=1200)
    assert out.returncode == 0, out.stderr
    _same(want, [json.loads(line) for line in out.stdout.splitlines()])


# --- the pool scenes ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scene", ["pool_fls_l12_5d", "pool_fls_p4_2v"])
def test_the_pool_scenes_run_with_their_files_and_the_defaults_refuse_them(scene):
    """With the chain's defaults the C++ engine refuses the pool records (2 m cells: fewer than three in a 2.2 m or 5 m
    window). With the sonar's own file (cells of one range resolution) both engines run them: detections ahead, within
    the window and the fan, the engines alike; no confirmed track (two pings, or one in --quick for p4_2v, under the
    tracker's three). The Python engine on the defaults does not refuse alike (an IndexError on l12_5d, a run on p4_2v):
    a follow-up, not changed here."""
    g = gallery()
    cfg = load_scenario_config(scene)
    g.QUICK[scene](cfg)
    records = run_scenario(cfg).records
    with pytest.raises(ValueError, match="shorter than three range cells"):
        _run(records, {}, "cpp")
    deploy = load_yaml(DEPLOY / f"{cfg['sonar']}.yaml")
    lo, hi = SonarModel.load(cfg["sonar"]).config["receive"]["window"]["range"]
    b = deploy["chain"]["bearings_deg"]
    runs = {e: _run(records, deploy["chain"], e) for e in ("python", "cpp")}
    _same(runs["python"], runs["cpp"])
    for r in runs["cpp"]:
        assert r["detections"] and r["tracks"] == []
        assert all(lo <= d["range_m"] <= hi and b[0] - 1e-9 <= d["bearing_deg"] <= b[1] + 1e-9 for d in r["detections"])


# --- the viewer --------------------------------------------------------------------------------------------------------

def test_the_viewer_takes_the_look_from_the_file_and_says_so_only_without_one(fls_point_pings):
    from gpusonar.viz import chain as vchain
    cfg, source = vchain.chain_config("fls_120k")
    assert cfg["look"] == "boresight" and "configs/deploy/fls_120k.yaml" in source
    same, note = vchain.look_toward_boresight(cfg, "fls_120k", fls_point_pings[0])
    assert same == cfg and note is None                          # the file gives the look: no workaround
    bare, source = vchain.chain_config("fls_300k")               # no deployment file names it
    assert bare == {} and "defaults" in source
    rec = SonarModel.load("fls_300k", {"receive": {"window": {"range": [10.0, 11.0]}, "tvg": None}}).make_record(make_pose())
    cfg, note = vchain.look_toward_boresight(bare, "fls_300k", rec)
    assert cfg == {"look": "boresight"} and "look: boresight" in note and "no deployment file" in note
    flank = SonarModel.load("flank_32", {"receive": {"window": {"range": [100.0, 120.0]}}}).make_record(make_pose())
    assert vchain.look_toward_boresight({}, "flank_32", flank) == ({}, None)          # the default already faces it
