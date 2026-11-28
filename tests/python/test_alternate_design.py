"""The alternate design for sea state 5 (the owner, 2026-11-27, D-255): the shipped full face with only its transmit energy
changed (configs/builds/popoto_face_3x32_ss5.yaml and the files generated from it). The baseline (configs/builds/popoto_face_3x32.yaml)
and every shipped file stay as they are."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from gpusonar import buildspec as bs  # noqa: E402
from gpusonar.config import load_sonar_config, load_yaml  # noqa: E402

ALT, BASE = "popoto_face_3x32_ss5", "popoto_face_3x32"
# The fields the alternate names (and nothing else): what it IS (energy), what it is called, and the simulated trial's range moved
# beyond the 4 s pulse's 3 km blind range.
# D-256: the baseline's waveform became the 2 s hyperbolic sweep; the alternate, a placeholder, keeps its 4 s LINEAR sweep, stated
# in its own file, so the waveform's type is now a field the two differ in too.
SPEC_CHANGES = {("build",), ("title",), ("stage",), ("names", "sonar"), ("names", "deploy"), ("projector", "source_level_db_rms"),
                ("projector", "waveform", "duration_s"), ("projector", "waveform", "type"), ("trial", "ranges_m")}


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def differences(a, b, path=()):
    """Paths (tuples of keys) where two mappings differ, to the leaves."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            out |= differences(a.get(k), b.get(k), path + (k,))
        return out
    return set() if a == b else {path}


def test_the_alternate_build_changes_only_what_it_names():
    alt, base = load_yaml(ROOT / "configs" / "builds" / f"{ALT}.yaml"), load_yaml(ROOT / "configs" / "builds" / f"{BASE}.yaml")
    assert differences(alt, base) == SPEC_CHANGES
    assert alt["projector"]["source_level_db_rms"] == 200.0 and alt["projector"]["waveform"]["duration_s"] == 4.0
    assert alt["projector"]["waveform"]["type"] == "lfm" and base["projector"]["waveform"]["type"] == "hfm"      # D-256 (before: both lfm)
    assert alt["projector"]["model"] is None                                                              # no projector is chosen
    assert not [i for i in bs.validate(bs.load(ALT)) if i.level == "error"]
    text = (ROOT / "configs" / "builds" / f"{ALT}.yaml").read_text()
    for said in ("D-255", "206 dB", "TR-1075D", "29 m", "3.0 km", "19.1 %", "via secondary", "TO BE CHOSEN FROM THE deep12 RUN"):
        assert said in text, said


def test_the_alternate_sonar_and_deployment_differ_from_the_shipped_only_in_name_energy_and_window():
    alt, base = load_sonar_config("popoto_long_range_3row_ss5"), load_sonar_config("popoto_long_range_3row")
    got = differences(alt, base)
    assert got == {("name",), ("build",), ("transmit", "waveform", "duration"), ("transmit", "waveform", "source_level_db_rms"),
                   ("receive", "window", "range"), ("transmit", "waveform", "type"), ("transmit", "waveform", "source")}, got   # D-256: type, source
    assert alt["transmit"]["waveform"]["type"] == "lfm" and alt["transmit"]["waveform"]["source"] is None
    assert alt["transmit"]["waveform"]["duration"] == 4.0 and alt["transmit"]["waveform"]["source_level_db_rms"] == 200.0
    assert alt["receive"]["window"]["range"] == [3000.0, 12700.0]           # opens at the blind range c T / 2
    assert "receive.window" in alt["unconfirmed"] and "transmit" in alt["unconfirmed"]
    a = load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle_elevation_ss5.yaml")
    b = load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle_elevation.yaml")
    # D-256: and the rows' gain the coverage picture states (3.99 dB for the alternate's window from 3 km, 3.98 for the baseline's from
    # 1.5 km; both were 3.99 while the baseline's window opened at 0.75 km: the detector's arithmetic over the window, rounded to 0.01 dB)
    assert differences(a, b) == {("source", "sonar"), ("source", "timeout"), ("chain", "coverage", "budget", "source_level_db"),
                                 ("chain", "coverage", "reverberation", "rows_gain_db")}
    # the shipped (baseline) files are what the shipped spec generates, and so is the alternate's
    gen = tool("build_from_spec")
    for rel, text in gen.shipped_files().items():
        assert (ROOT / "configs" / rel).read_text() == text, rel


def test_the_alternate_build_holds_the_trade_study_figure_at_sea_state_5():
    """The build spec's own plan at the deep site at sea state 5 with the bubble law equals the trade study's figure for 4 s at 200 dB
    (docs/design/popoto_trade_study.json, `--transmit`) within 0.05 dB, both classes; and it holds both (the reason it exists)."""
    T = tool("popoto_trade_study")
    from gpusonar import depthplan, environment
    spec = bs.load(ALT)
    d = bs.derive(spec)
    v = bs.derive(T.variant_spec(3, 32, "2k", 4.0, 200.0))          # the study's in-memory variant of the same option
    assert all(d[k] == v[k] for k in ("source_level_db", "pulse_s", "band_hz", "rows", "per_row", "window_m", "blind_range_m"))
    grid = json.loads((ROOT / "docs" / "design" / "popoto_trade_study.json").read_text())["transmit"]["grid"]["results"]
    envs = T.tx_sites(environment.load_list("illustrative"))
    classes = bs.target_classes_of(spec, range_m=12000.0)
    for key in ("deep_no_bottom_return__ss5__apl_1994", "deep_no_bottom_return_sea_state_5__as_stated"):
        plan = depthplan.plan(d, envs[key], classes)
        for c in ("large_submerged", "ship"):
            got = plan["classes"][c]["recommended"]["margin_db"]
            assert got == pytest.approx(grid[key]["4 s 200 dB"][c]["margin_db"], abs=0.05), (key, c)
            assert got >= 0.0, (key, c)
    # the baseline at the same site does not hold either (why the alternate exists)
    base = depthplan.plan(bs.derive(bs.load(BASE)), envs["deep_no_bottom_return__ss5__apl_1994"], classes)
    assert all(base["classes"][c]["recommended"]["margin_db"] < 0.0 for c in ("large_submerged", "ship"))


def test_the_alternate_is_the_least_energy_option_with_a_single_listed_projector():
    """4 s at 200 dB is taken from the table, not typed: among the options with the least energy that holds both classes at sea state 5
    with the bubble law, the one a single listed projector supports (D-247's table; D-255)."""
    T = json.loads((ROOT / "docs" / "design" / "popoto_trade_study.json").read_text())["transmit"]
    least = T["answers"]["least_energy"]["bubble_law"]["5"]
    assert least["energy_db"] == pytest.approx(206.0, abs=0.1) and "4 s 200 dB" in least["options"]
    single = [o["id"] for o in T["options"] if o["id"] in least["options"] and o["costs"]["projector"]["single_units"]]
    assert single == ["4 s 200 dB"]
    opt = next(o for o in T["options"] if o["id"] == "4 s 200 dB")
    assert opt["costs"]["blind_range_m"] == pytest.approx(3000.0) and opt["costs"]["duty"] == pytest.approx(0.191, abs=1e-3)
    assert opt["costs"]["projector"]["single_units"][0]["model"] == "Massa TR-1075D"
    assert copy.deepcopy(opt)["costs"]["projector"]["single_units"][0]["least_depth_maker_m"] == pytest.approx(29.1, abs=0.1)
