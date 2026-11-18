"""The reference design is usable from a spec sheet: build specs validate, the generator's output is what is
committed, the numbers on the diagrams and in the parts list follow from the files, and a deliberately different
build goes from its spec through the simulator and both engines of the chain to a truth check."""

import copy
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import buildspec as bs
from gpusonar import config
from gpusonar.config import load_scenario_config, load_sonar_config, load_yaml
from gpusonar.sonar import SonarModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
EXAMPLE = ROOT / "docs" / "examples" / "reference_design" / "example_static_2x24.yaml"
STAGES = ["popoto_testauv_1x32", "popoto_static_2x32", "popoto_static_3x32", "popoto_face_1x32", "popoto_face_2x32", "popoto_face_3x32"]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- config flags -------------------------------------------------------------------------------------

def test_unconfirmed_accumulates_through_base_and_confirmed_takes_a_flag_off(tmp_path):
    """A child's `unconfirmed` used to REPLACE its base's (a list is not merged): the base's flags vanished."""
    (tmp_path / "sonars").mkdir()
    (tmp_path / "sonars" / "a.yaml").write_text("name: a\nx: {p: 1, q: 2}\nunconfirmed: [x.p, x.q, y]\n")
    (tmp_path / "sonars" / "b.yaml").write_text("base: a\nname: b\nz: 3\nunconfirmed: [z]\n")
    (tmp_path / "sonars" / "c.yaml").write_text("base: b\nname: c\nconfirmed: [x.p, y]\n")
    (tmp_path / "sonars" / "d.yaml").write_text("base: c\nname: d\nconfirmed: [x]\nunconfirmed: [y]\n")
    config.add_config_dir(tmp_path)
    try:
        assert load_sonar_config("b")["unconfirmed"] == ["x.p", "x.q", "y", "z"]            # the base's flags stay, in order
        c = load_sonar_config("c")
        assert c["unconfirmed"] == ["x.q", "z"] and c["confirmed"] == ["x.p", "y"]          # the explicit way to confirm
        d = load_sonar_config("d")
        assert d["unconfirmed"] == ["z", "y"] and d["confirmed"] == ["x.p", "x"]            # `x` clears what is below it; `y` is unconfirmed again
        assert "unconfirmed" not in load_sonar_config("flank_32") or isinstance(load_sonar_config("flank_32")["unconfirmed"], list)
    finally:
        config.remove_config_dir(tmp_path)
    # The shipped case that hid flags: a `_clocks` file listed two and its base's were lost.
    flags = load_sonar_config("popoto_long_range_3row_clocks")["unconfirmed"]
    assert {"receive.boards", "receive.reference", "receive.sample_rate", "receive.window", "transmit"} <= set(flags)
    assert flags == list(dict.fromkeys(flags))
    with pytest.raises(ValueError, match="list"):
        config.merge_flags({"unconfirmed": "x"}, {"unconfirmed": ["y"]}, {})


def test_every_stage_sonar_states_its_own_receive_window():
    """The window used to be inherited silently from the eight-hydrophone board's file ([50, 2000] m for a 12 km sonar)."""
    for name in STAGES:
        spec = bs.load(name)
        text = (ROOT / "configs" / "sonars" / f"{bs.names(spec)['sonar']}.yaml").read_text()
        assert "window:" in text and "range:" in text, name
        got = load_sonar_config(bs.names(spec)["sonar"])["receive"]["window"]["range"]
        assert got == bs.window_range(spec) and got[0] == pytest.approx(0.5 * 1500.0 * spec["projector"]["waveform"]["duration_s"])   # from the blind range
        assert "receive.window" in load_sonar_config(bs.names(spec)["sonar"])["unconfirmed"]        # a choice, flagged
    assert load_sonar_config("popoto_long_range_3row")["receive"]["window"]["range"] == [750.0, 12700.0]


# --- specs, validator ---------------------------------------------------------------------------------

def test_shipped_specs_are_valid_and_the_committed_files_are_what_they_generate():
    assert bs.shipped() == sorted(STAGES)
    gen = tool("build_from_spec")
    for rel, text in gen.shipped_files().items():
        assert (ROOT / "configs" / rel).read_text() == text, f"{rel}: run tools/build_from_spec.py --write-shipped"
    for name in STAGES:
        spec = bs.load(name)
        assert not [i for i in bs.validate(spec) if i.level == "error"], name
        assert bs.files(spec) == bs.files(bs.load(name))                                    # deterministic
        sonar = SonarModel.load(bs.names(spec)["sonar"])
        assert sonar.n_elements == spec["array"]["rows"] * 32 and np.allclose(sonar.positions, bs.positions(spec))
        clocks = SonarModel.load(bs.names(spec)["sonar"] + "_clocks")
        assert [(b["clock_offset"], b["clock_drift_ppm"]) for b in clocks.boards] == bs.clock_errors(spec)
        deploy = load_yaml(ROOT / "configs" / "deploy" / f"{bs.names(spec)['deploy']}.yaml")
        assert deploy["source"]["sonar"] == clocks.name and ("elevation" in deploy["chain"]) == (spec["array"]["rows"] > 1)


def test_row_modules_one_long_row_for_static_and_face_the_short_one_only_on_the_test_auv():
    mods = bs.modules()
    assert set(mods) == {"long_4p8m", "testauv_short"} and all(m["array"]["count"] == 32 for m in mods.values())
    used = {name: bs.load(name)["array"]["module"] for name in STAGES}
    assert used.pop("popoto_testauv_1x32") == "testauv_short" and set(used.values()) == {"long_4p8m"}
    static, face = SonarModel.load("popoto_static_3row"), SonarModel.load("popoto_long_range_3row")
    assert np.allclose(static.positions, face.positions)                                    # the same rows at the same heights
    assert static.config["platform"]["static"] is True and static.config["platform"]["span_m"] >= 4.8
    # A field given in the build wins over the module; a missing module is said so.
    spec = load_yaml(ROOT / "configs" / "builds" / "popoto_face_1x32.yaml")
    spec["array"]["pitch_m"] = 0.12
    assert bs.resolve_module(spec)["array"]["pitch_m"] == 0.12 and bs.resolve_module(spec)["array"]["count"] == 32
    spec["array"]["module"] = "nope"
    with pytest.raises(FileNotFoundError, match="long_4p8m"):
        bs.resolve_module(spec)


def issues_of(change):
    spec = copy.deepcopy(bs.load(EXAMPLE))
    change(spec)
    return bs.validate(spec)


def has(issues, level, where, text):
    return any(i.level == level and i.where == where and text in i.message for i in issues)


def test_validator_says_what_is_wrong_in_the_spec_sheets_words():
    assert not [i for i in bs.validate(bs.load(EXAMPLE)) if i.level == "error"]
    assert has(issues_of(lambda s: s["array"].pop("count")), "error", "array.count", "missing")
    assert has(issues_of(lambda s: s.pop("projector")), "error", "projector", "section missing")
    assert has(issues_of(lambda s: s["array"].update(pich_m=0.06)), "error", "array.pich_m", "unknown field")
    assert has(issues_of(lambda s: s["boards"].update(sample_rate_hz="64k")), "error", "boards.sample_rate_hz", "a number")
    assert has(issues_of(lambda s: s["boards"].update(sample_rate_hz=24000.0)), "error", "boards.sample_rate_hz", "cannot sample")
    assert has(issues_of(lambda s: s["array"].update(pitch_m=0.2)), "error", "array.pitch_m", "grating lobe")
    assert has(issues_of(lambda s: s["array"].update(pitch_m=0.08)), "warning", "array.pitch_m", "steers to")
    assert has(issues_of(lambda s: s["array"].update(count=40)), "error", "array.count", "usable span")
    assert has(issues_of(lambda s: s["array"].update(heights_m=[-0.3, 0.3])), "error", "array.heights_m", "outside the usable height")
    assert has(issues_of(lambda s: s["array"].update(heights_m=[0.06])), "error", "array.heights_m", "2 rows need 2 heights")
    assert has(issues_of(lambda s: s["array"].update(count=26, pitch_m=0.058)), "warning", "boards.inputs", "input(s) spare")
    assert has(issues_of(lambda s: s["boards"].update(input_full_scale_vpp=1.0)), "error", "boards.adc_full_scale_v", "contradicts")
    assert has(issues_of(lambda s: s["projector"].update(band_hz=[9.0, 13.0])), "warning", "projector.band_hz", "kHz")
    assert has(issues_of(lambda s: s["sync"]["reference"].update(f_stop_hz=40.0e3)), "error", "sync.reference", "half the sample rate")
    assert has(issues_of(lambda s: s["sync"].update(scheme="none")), "error", "sync.scheme", "own clocks")
    assert has(issues_of(lambda s: s["hydrophone"].update(band_hz=[10000.0, 30000.0])), "error", "hydrophone.band_hz", "does not cover")
    assert has(issues_of(lambda s: s["trial"].update(ranges_m=[60.0, 400.0])), "error", "trial.ranges_m", "blind range")
    assert has(issues_of(lambda s: s["trial"].update(own_depth_m=50.0)), "error", "trial.own_depth_m", "below the bottom")
    assert has(issues_of(lambda s: s["boards"].update(clocks={"explicit": [[0, 0]]})), "error", "boards.clocks", "8 boards")
    assert has(issues_of(lambda s: s["hydrophone"].update(confirmed=["noise_db"])), "error", "hydrophone.confirmed", "no value")
    # The projector's direct sound against the inputs: a warning that carries the computed level.
    loud = issues_of(lambda s: s["projector"].update(position_m=[0.0, 0.0, 0.3], source_level_db_rms=197.0))
    msg = next(i.message for i in loud if i.where == "projector.position_m" and "direct sound" in i.message)
    spec = copy.deepcopy(bs.load(EXAMPLE))
    spec["projector"].update(position_m=[0.0, 0.0, 0.3], source_level_db_rms=197.0)
    d = bs.derive(spec)["direct"]
    assert f"{d['level_db']:.0f} dB re 1 uPa" in msg and f"{d['over_full_scale_db']:.0f} dB over full scale" in msg and d["over_full_scale_db"] > 20.0
    assert d["level_db"] == pytest.approx(197.0 - 20.0 * np.log10(d["nearest_m"]))
    assert has(loud, "warning", "sync.reference", "recover")                                # the sync burst right after an overload
    with pytest.raises(bs.SpecError, match="array.count"):
        s = copy.deepcopy(bs.load(EXAMPLE))
        s["array"].pop("count")
        bs.check(s)


def test_status_of_a_field_flows_into_unconfirmed():
    spec = bs.load("popoto_face_3x32")
    assert bs.status(spec, "platform", "span_m") == "confirmed" and bs.status(spec, "platform", "usable_span_m") == "assumed"
    assert bs.status(spec, "hydrophone", "model") == "unknown"
    flags = bs.sonar_config(spec)["unconfirmed"]
    assert "platform.usable_span_m" in flags and "platform.span_m" not in flags and "receive.sample_rate" in flags and "receive.gain_db" not in flags
    spec["boards"]["confirmed"].append("sample_rate_hz")
    spec["projector"]["confirmed"] = ["position_m"]
    flags = bs.sonar_config(spec)["unconfirmed"]
    assert "receive.sample_rate" not in flags and "transmit.lever_arm" not in flags and "transmit.waveform.source_level_db_rms" in flags
    written = bs.against_base(bs.sonar_config(spec), "popoto_long_range")
    assert "receive.sample_rate" in written["confirmed"] and "transmit" in written["confirmed"]   # the base's flags this build settles or narrows


# --- derived numbers, performance, parts, diagrams ----------------------------------------------------

def test_derived_numbers_are_arithmetic_on_the_files():
    d = bs.describe_shipped("popoto_long_range_3row", "popoto_vehicle_elevation")
    assert (d["channels"], d["n_boards"], d["rows"], d["per_row"]) == (96, 12, 3, [32, 32, 32])
    assert d["rate_total_bit_s"] == 96 * 102400 * 24 and d["rate_board_bit_s"] == 8 * 102400 * 24
    assert d["blind_range_m"] == 750.0 and d["ping_interval_s"] == pytest.approx(2 * 12700 / 1500 + 1.0)
    assert d["samples_per_ping"] == int(np.ceil((d["window_s"][1] - d["window_s"][0]) * 102400)) and d["record_bytes_per_ping"] == 96 * d["samples_per_ping"] * 4
    study = json.loads((ROOT / "docs" / "design" / "popoto_elevation_study.json").read_text())
    pj = study["projector"]
    assert d["direct"]["nearest_m"] == pytest.approx(pj["nearest_hydrophone_m"]) and d["direct"]["over_full_scale_db"] == pytest.approx(pj["over_full_scale_db"])
    assert d["elevation"]["clock_us_for_error"] == pytest.approx(study["sensitivity"]["clock_us_for_deg"]["0.05"])     # 0.87 us for 0.05 deg at 1.5 m
    assert d["elevation"]["highest_repeat_db"] == pytest.approx(study["three_rows"]["best_level_db"], abs=0.05)
    # Full scale: the plan's 37 mVpp at the input with 43 dB is the files' 2.61 V; with -208 dB re 1 V/uPa that is 173 dB re 1 uPa.
    assert 0.5 * 0.037 * 10 ** (43 / 20) == pytest.approx(d["adc_full_scale_v"], rel=0.01) and d["input_full_scale_pressure_db"] == pytest.approx(173.3, abs=0.1)
    # Cables: each run along the face, then up or down, to the assumed board position, plus the slack.
    pos = d["positions"]
    assert d["cables"]["total_m"] == pytest.approx(np.abs(pos).sum() + 0.5 * 96) and d["cables"]["longest_m"] == pytest.approx(np.abs(pos).sum(axis=1).max() + 0.5)
    # The simulator's direct-sound study (tools/direct_sound_study.py) is separate code: per-hydrophone overload must agree.
    ds = json.loads((ROOT / "docs" / "design" / "direct_sound_study.json").read_text())["popoto_long_range_3row"]["summary"]
    assert d["direct"]["over_full_scale_db"] == pytest.approx(ds["overload_db_nearest"], abs=1e-6) and d["direct"]["over_full_scale_farthest_db"] == pytest.approx(ds["overload_db_farthest"], abs=1e-6)
    assert d["direct"]["hydrophones_overloaded"] == ds["hydrophones_overloaded"] == 96 and d["direct"]["distance_for_full_scale_m"] == pytest.approx(ds["distance_for_no_overload_m"], rel=1e-6)
    assert d["direct"]["gain_reduction_needed_db"] == pytest.approx(ds["gain_reduction_for_no_overload_db"]) and d["direct"]["min_range_memoryless_m"] == pytest.approx(753.2, abs=0.1)
    one = bs.describe_shipped("popoto_testauv_1row", "popoto_testauv_1row")
    assert one["elevation"] is None and one["n_boards"] == 4 and one["blind_range_m"] == 75.0


def test_expected_performance_is_the_studys_for_every_stage():
    """buildspec.performance is the generator's arithmetic for any geometry; tools/popoto_elevation_study.py is separate
    code for the Popoto stages. They must agree."""
    study = json.loads((ROOT / "docs" / "design" / "popoto_elevation_study.json").read_text())
    assert len(study["stages"]) == 6
    for label, st in study["stages"].items():
        d = bs.describe_shipped(st["sonar"])
        perf = bs.performance(d)
        assert perf["beamwidth_deg"] == pytest.approx(st["beamwidth_deg"], rel=1e-6), label
        assert perf["gain_isotropic_noise_db"] == pytest.approx(st["gain_isotropic_noise_db"], abs=1e-6), label
        assert perf["reach_ship_m"] == pytest.approx(st["reach_ship_m"], rel=1e-3) and perf["reach_small_m"] == pytest.approx(st["reach_small_m"], rel=1e-3), label
        if st["rows"] > 1:
            assert perf["elevation"]["sigma_deg"]["20.0"] == pytest.approx(st["elevation_sigma_deg_at_20_db"], rel=1e-6), label
            assert perf["elevation"]["highest_repeat_db"] == pytest.approx(st["highest_repeat_db"], abs=1e-6), label
            assert perf["rows_detection_gain_db"] == pytest.approx(st["rows_detection_gain_db"], abs=1e-6), label


def test_parts_list_quotes_the_plan_and_invents_nothing():
    parts_list = tool("parts_list")
    d = bs.describe_shipped("popoto_long_range_3row", "popoto_vehicle_elevation")
    rows = {r["key"]: r for r in parts_list.parts(d)}
    assert rows["boards"]["qty"] == 12 and rows["boards"]["status"] == "chosen" and rows["hydrophones"]["qty"] == 96
    assert rows["projector"]["status"] == "specified, not chosen" and rows["input_protection"]["status"] == "unconfirmed"
    assert "22 dB over" in rows["input_protection"]["requirement"] and "0.87 µs" in rows["boards"]["requirement"]
    assert "to be supplied" in rows["navigation"]["requirement"] and "SPRINT-Nav" in rows["navigation"]["requirement"]
    assert "pending" in rows["compute"]["requirement"] or "popoto_stage2.json" in rows["compute"]["requirement"]
    assert {r["status"] for r in rows.values()} <= {"chosen", "specified, not chosen", "unconfirmed"}
    # Every quotation is in the plan's text, word for word (needs pdftotext; the PDF is in the repository).
    import shutil
    import subprocess
    if shutil.which("pdftotext"):
        text = subprocess.run(["pdftotext", str(ROOT / "docs" / "plan" / "GPU-FLS-Phased-Plan.pdf"), "-"], capture_output=True, text=True, check=True).stdout
        flat = " ".join(text.split())
        for key, quotes in parts_list.QUOTES.items():
            for where, quote in quotes:
                if key == "compute" and "table" in where:
                    continue                                                            # table cells: joined here with separators
                assert " ".join(quote.split()) in flat, (key, quote)
    on_disk = (ROOT / "docs" / "design" / "popoto_parts.md").read_text()
    for label, _ in parts_list.stage_columns():
        assert label in on_disk
    # Another build, another table: nothing is Popoto-specific in the function.
    other = {r["key"]: r for r in parts_list.parts(bs.derive(bs.load(EXAMPLE)))}
    assert other["boards"]["qty"] == 8 and other["boards"]["plan"] == [] and "not measured for 2 x 24" in other["compute"]["requirement"]


def test_diagrams_are_drawn_from_the_files_for_every_stage_and_for_another_build(tmp_path):
    pytest.importorskip("matplotlib")
    diagrams = tool("build_diagrams")
    for _, sonar, deploy, stem in diagrams.STAGES:
        for kind in ("hardware", "timing"):
            assert (ROOT / "docs" / "design" / f"build_{stem}_{kind}.png").exists(), stem
        assert (ROOT / "configs" / "sonars" / f"{sonar}.yaml").exists() and (ROOT / "configs" / "deploy" / f"{deploy}.yaml").exists()
    paths = diagrams.draw_build(bs.derive(bs.load(EXAMPLE)), tmp_path, "example")
    assert all(p.exists() and p.stat().st_size > 20000 for p in paths)


def test_design_guide_lists_every_field_of_the_spec():
    text = (ROOT / "docs" / "REFERENCE_DESIGN.md").read_text()
    for section, fields in bs.SCHEMA.items():
        for name in fields:
            assert f"`{section}.{name}`" in text, f"{section}.{name}: run tools/build_from_spec.py --fields"
    gen = tool("build_from_spec")
    assert gen.field_table() in text


# --- a different build, end to end --------------------------------------------------------------------

def test_a_different_build_goes_from_its_spec_to_a_truth_check_on_both_engines(tmp_path):
    """Two rows of 24, 9-13 kHz, six-input boards at 64 kHz, a static frame: validator -> generator -> simulator ->
    the chain on both engines -> truth, and the stream at its channel count. Nothing in it is one of the shipped sonars."""
    gen = tool("build_from_spec")
    spec = bs.load(EXAMPLE)
    assert not [i for i in bs.validate(spec) if i.level == "error"]
    written = gen.generate(spec, tmp_path, diagrams=False)
    assert {"configs/sonars/example_static_2x24.yaml", "configs/sonars/example_static_2x24_clocks.yaml", "configs/deploy/example_static_2x24.yaml",
            "configs/scenarios/example_static_2x24_free.yaml", "configs/scenarios/example_static_2x24_passive.yaml", "chain.options", "BUILD.md", "parts.md"} <= set(written)
    config.add_config_dir(tmp_path / "configs")
    try:
        sonar = SonarModel.load("example_static_2x24_clocks")
        assert (sonar.n_elements, len(sonar.boards), sonar.sample_rate, sonar.center_frequency) == (48, 8, 64000.0, 11000.0)
        assert sonar.channel_board.tolist() == [i // 6 for i in range(48)]
        passive = load_scenario_config("example_static_2x24_passive")
        assert passive["pings"]["transmit"] is False and len(passive["sources"]) == 2
    finally:
        config.remove_config_dir(tmp_path / "configs")
    deploy = load_yaml(tmp_path / "configs" / "deploy" / "example_static_2x24.yaml")
    assert deploy["chain"]["engine"] == "cpp" and deploy["source"]["port"] == 5800 and "static" in deploy["nav"]
    assert "sync=1" in (tmp_path / "chain.options").read_text() and "elevation=1" in (tmp_path / "chain.options").read_text()
    res = gen.self_test(spec, tmp_path, kinds=("free", "rehearsal", "stream"), log=lambda *_: None)
    assert res["ok"], json.dumps(res["tests"], indent=1, default=float)[:3000]
    for engine in ("python", "cpp"):
        free = res["tests"]["free"][engine]
        assert len(free["targets"]) == 3 and all(abs(t["depth_error_m"]) < 3.6 and not t["ambiguous"] for t in free["targets"]), engine
        reh = res["tests"]["rehearsal"][engine]
        assert reh["sync_ok"] and reh["sync_worst_residual_us"] < 1.0, engine
    assert res["tests"]["stream"]["channels"] == 48 and res["tests"]["stream"]["udp_lost_samples"] == 0


# --- the 12 km design point with stacked rows ---------------------------------------------------------

def _twelve_km(rows, bellhop):
    study = tool("popoto_12km_study")
    cfg = load_scenario_config(study.SCENARIO[rows])
    truth = cfg.pop("truth")
    assert truth["ship"]["range_m"] == 12000.0 and cfg["sonar"] == study.SONAR[rows]
    cfg["pings"]["count"] = 4
    if not bellhop:
        # The stand-in checks the files, the geometry and the chain at 12 km, not the channel: straight rays, no reverberation,
        # and the ship's points 12 dB stronger. As shipped (+18 dB a point, 4 m deep) the straight-ray stand-in does NOT hold the
        # ship at 12 km with stacked rows: its direct and surface-reflected paths nearly cancel (docs/DESIGN.md, the 12 km design
        # point; docs/design/popoto_12km.json has the as-shipped rows). Bellhop's ducted field does (the test below).
        cfg = study.image_source(cfg)
        cfg.pop("reverberation", None)
        cfg["scene"][0]["target_strength_db"] += 12.0
    return study, cfg


@pytest.mark.parametrize("rows", [2, 3])
def test_twelve_km_point_with_stacked_rows_detects_and_tracks_the_ship_on_both_engines(rows):
    """popoto_12km_2row / _3row with their deployment files. Image-source paths: always runs."""
    study, cfg = _twelve_km(rows, bellhop=False)
    got = {engine: study.measure(cfg, rows, engine=engine) for engine in ("python", "cpp")}
    for engine, m in got.items():
        assert m["pings_detected"] >= 3 and m["track_confirmed_at_ping"] is not None and m["track_confirmed_at_ping"] <= 4, engine
        hits = [p for p in m["per_ping"] if p["detected"]]
        assert all(abs(p["range_error_m"]) < 80.0 and abs(p["bearing_error_deg"]) < 1.0 for p in hits), engine     # the ship is 120 m long
        assert any(p.get("elevation_deg") is not None or p.get("candidates") for p in hits), engine
    assert got["python"]["pings_detected"] == got["cpp"]["pings_detected"]


@pytest.mark.parametrize("rows", [3])
def test_twelve_km_point_with_bellhop(rows):
    from gpusonar.sim.bellhop import find_bellhop
    if find_bellhop() is None:
        pytest.skip("Bellhop not installed (GPUSONAR_BELLHOP); the image-source test above covers the files and the chain")
    study, cfg = _twelve_km(rows, bellhop=True)
    cfg["propagation"]["max_arrivals"] = 16
    m = study.measure(cfg, rows, engine="cpp")
    assert m["pings_detected"] >= 3 and m["track_confirmed_at_ping"] is not None


def test_twelve_km_section_is_in_the_design_and_reads_its_numbers_from_the_tools():
    plan = tool("reference_plan")
    text = "\n".join(plan.twelve_km())
    assert "### The 12 km design point" in text and "At 12 km you get" in text and "What would close" in text
    _, d, perf = plan.build("popoto_face_3x32")
    assert f"{plan.row_snr(perf, 'ship'):.1f} dB, {plan.margin(perf, 'ship'):+.1f} dB of margin, Pd {plan.pd_of(perf, 'ship'):.2f}" in text and f"{d['ping_interval_s']:.1f} s" in text
    assert "13 dB at the map" not in text and "THRESHOLD IS NOT THE SHIPPED DETECTOR" not in text
    assert f"| {perf['detection']['threshold_db']:.1f} dB | {plan.row_snr(perf, 'ship'):.1f} dB | {plan.margin(perf, 'ship'):+.1f} dB | {plan.pd_of(perf, 'ship'):.2f} |" in text     # needed, SNR, margin, Pd
    # the plan's threshold paragraph: the chain's own figure, and the capitals warning of the typed 13 dB is gone
    depth = "\n".join(plan.depth_and_targets())
    assert f"{perf['detection']['threshold_db']:.1f} dB of each row" in depth and "THE BUDGET'S" not in depth and "13 dB at the map" not in depth
    assert "unexplained" in depth                                            # the budget against the chain, term by term (surface_image_validation.json)
    study = json.loads((ROOT / "docs" / "design" / "popoto_elevation_study.json").read_text())
    assert f"{study['refraction_m']['0.05']['12000.0']:.0f} m" in text
    assert perf["row_snr_db"]["ship"]["12000.0"] == pytest.approx(study["link_snr_db"]["ship, +25 dB"]["12000.0"], abs=0.05)
    assert "### The 12 km design point" in (ROOT / "docs" / "DESIGN.md").read_text()


# --- the link budget's threshold is the chain's own detector's ----------------------------------------------------------

def test_the_budgets_threshold_is_the_deployed_detectors_and_no_gain_is_counted_twice():
    """`buildspec.detection` asks gpusonar.live for the SNR of one row the build's chain needs; the budget's SNR is one
    row's with the row's gain against isotropic noise and nothing for the rows; margin = SNR - threshold. Held against
    independent calls of `live.budget_threshold_db` and `live.detection_probability` on a template made here."""
    from gpusonar import core, live
    from gpusonar.sonar import SonarModel
    from gpusonar.sonar_equation import isotropic_noise_gain_db
    assert not hasattr(bs, "BUDGET_THRESHOLD_DB") and not hasattr(bs, "DETECTOR_THRESHOLD_DB") and not hasattr(bs, "rows_detection_gain_db")
    perf = {}
    for name in bs.shipped():
        spec = bs.load(name)
        d = bs.derive(spec)
        p = perf[name] = bs.performance(d)
        det, rows = p["detection"], int(spec["array"]["rows"])
        assert det["rows"] == rows and det["probability"] == bs.DESIGN_DETECTION_PROBABILITY == 0.5 and det["statistic"] == "multiscale"
        chain = dict(bs.deploy_config(spec)["chain"])
        chain.pop("sync", None)
        template = bs.detector_template(bs.sonar_config(spec))
        assert det["threshold_db"] == pytest.approx(live.budget_threshold_db(template, chain, pd=0.5), abs=1e-6), name     # rows: the chain's own count
        assert det["threshold_high_db"] == pytest.approx(live.budget_threshold_db(template, chain, pd=0.9), abs=1e-6), name
        assert det["threshold_one_row_db"] - det["threshold_db"] == pytest.approx(det["rows_gain_db"]) == pytest.approx(p["rows_detection_gain_db"])
        assert (det["rows_gain_db"] == 0.0) == (rows == 1)
        # the equation holds ONE row's gain and nothing for the rows; the threshold is what the margin and the reach are against
        eq, gains = bs.budget(d, target_strength_db=25.0)
        row = isotropic_noise_gain_db(np.arange(32) * d["pitch_m"], d["centre_hz"], d["sound_speed"])
        assert eq.array_gain() == pytest.approx(row) == pytest.approx(p["gain_isotropic_noise_db"]) and gains["threshold_db"] == det["threshold_db"]
        for r, snr in p["row_snr_db"]["ship"].items():
            assert snr == pytest.approx(float(eq.snr(float(r)))) and p["margin_db"]["ship"][r] == pytest.approx(snr - det["threshold_db"])
            assert p["detection_probability"]["ship"][r] == pytest.approx(live.detection_probability(snr, template, chain), abs=1e-9)
        assert float(eq.snr(p["reach_ship_m"])) == pytest.approx(det["threshold_db"], abs=0.01)
        assert float(eq.snr(p["reach_ship_high_m"])) == pytest.approx(det["threshold_high_db"], abs=0.01) and p["reach_ship_high_m"] < p["reach_ship_m"]
        assert bs.detection_probability(d, det["threshold_db"]) == pytest.approx(0.5, abs=2e-3) and bs.detection_probability(d, det["threshold_high_db"]) == pytest.approx(0.9, abs=2e-3)
    # the face: the same row SNR with one, two and three rows; the rows show in the threshold only
    one, two, three = (perf[f"popoto_face_{k}x32"] for k in (1, 2, 3))
    r = "12000.0"
    assert one["row_snr_db"]["ship"][r] == pytest.approx(two["row_snr_db"]["ship"][r]) == pytest.approx(three["row_snr_db"]["ship"][r])
    assert three["margin_db"]["ship"][r] - one["margin_db"]["ship"][r] == pytest.approx(three["rows_detection_gain_db"]) == pytest.approx(4.0, abs=0.2)
    # the 12 km design point: -1.1 / +1.4 / +2.9 dB and 0.22 / 0.87 / 0.99. It was -1.0 / +1.5 / +3.0 and 0.24 / 0.89 / 0.99 while
    # the detector counted its tests with one flat factor: counted as measured (libs/live detector.hpp, kSearchSample) the
    # thresholds are 0.12 to 0.15 dB higher.
    for p, margin, pd in ((one, -1.1, 0.22), (two, 1.4, 0.87), (three, 2.9, 0.99)):
        assert p["margin_db"]["ship"][r] == pytest.approx(margin, abs=0.15) and p["detection_probability"]["ship"][r] == pytest.approx(pd, abs=0.03)
    # the short row closer than half a wavelength: 12.9 dB against sea noise, not 10 log 32 (D-103's closed form)
    assert perf["popoto_testauv_1x32"]["gain_isotropic_noise_db"] == pytest.approx(12.9, abs=0.1)
    assert SonarModel.load("popoto_testauv_1row") and bs.directivity_db(32, 0.15, 5000.0) == pytest.approx(10 * np.log10(32), abs=1e-9)


def test_the_detectors_template_does_not_need_the_full_rate_window():
    """The template `detection` uses is complex baseband at four times the band; the thresholds are the full-rate record's."""
    from gpusonar import core, live
    from gpusonar.sonar import SonarModel
    spec = bs.load("popoto_testauv_1x32")
    cfg = bs.sonar_config(spec)
    chain = dict(bs.deploy_config(spec)["chain"])
    chain.pop("sync", None)
    full = SonarModel.from_config(copy.deepcopy(cfg)).make_record(core.Pose(), environment={"sound_speed": 1500.0})       # real samples at 102.4 kHz
    small = bs.detector_template(cfg)
    assert small.samples.n_samples * 10 < full.samples.n_samples
    for pd in (0.5, 0.9):
        assert live.budget_threshold_db(small, chain, pd=pd) == pytest.approx(live.budget_threshold_db(full, chain, pd=pd), abs=1e-3)


def test_the_design_detection_probability_is_a_field_of_the_spec():
    spec = bs.load("popoto_face_3x32")
    base = bs.performance(bs.derive(spec))
    high = copy.deepcopy(spec)
    high["operation"]["detection_probability"] = 0.9
    assert not [i for i in bs.validate(high) if i.level == "error"]
    d = bs.derive(high)
    p = bs.performance(d)
    assert d["detection_probability"] == 0.9 and p["detection"]["probability"] == 0.9
    assert p["detection"]["threshold_db"] == pytest.approx(base["detection"]["threshold_high_db"], abs=1e-6)
    assert p["reach_ship_m"] == pytest.approx(base["reach_ship_high_m"], rel=1e-6) and p["margin_db"]["ship"]["12000.0"] < base["margin_db"]["ship"]["12000.0"] - 1.0
    assert bs.files(high) == bs.files(spec)                                # a planning figure: no generated sonar or deployment file moves
    bad = copy.deepcopy(spec)
    bad["operation"]["detection_probability"] = 1.5
    assert any(i.where == "operation.detection_probability" and i.level == "error" for i in bs.validate(bad))
    # the depth plan takes the same threshold and says so
    from gpusonar import depthplan
    env = depthplan.load_environments("illustrative")[0]
    P = depthplan.plan(bs.derive(spec), env, bs.target_classes_of(spec), own_depths=[10.0, 225.0], ranges=[12000.0])
    assert P["threshold_db"] == pytest.approx(base["detection"]["threshold_db"]) and P["detection"]["threshold_is_the_detectors"]
    ship = P["classes"]["ship"]
    assert ship["free_field_margin_db"] == pytest.approx(base["margin_db"]["ship"]["12000.0"] + ship["ts_db"][0] - 25.0, abs=1e-6)
    assert 0.0 <= ship["recommended"]["detection_probability"] <= 1.0


def test_array_gain_against_isotropic_noise_closed_forms():
    """sonar_equation.isotropic_noise_gain_db against the two-hydrophone closed forms, the half-wavelength line and the
    limit of coincident hydrophones; SonarEquation.from_sonar uses it (10 log N only at half a wavelength)."""
    from gpusonar.sonar import SonarModel
    from gpusonar.sonar_equation import SonarEquation, isotropic_noise_gain_db
    c, f = 1500.0, 5000.0
    lam = c / f
    for d in (0.05 * lam, 0.25 * lam, 0.5 * lam, 0.8 * lam):
        kd = 2 * np.pi * d / lam
        s = np.sin(kd) / kd
        assert isotropic_noise_gain_db([0.0, d], f, c) == pytest.approx(10 * np.log10(2.0 / (1.0 + s)), abs=1e-9)                      # broadside
        along = isotropic_noise_gain_db(np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]), f, c, direction=[1.0, 0.0, 0.0])
        assert along == pytest.approx(10 * np.log10(2.0 / (1.0 + s * np.cos(kd))), abs=1e-9)                                           # endfire
    assert isotropic_noise_gain_db(np.arange(32) * 0.5 * lam, f, c) == pytest.approx(10 * np.log10(32), abs=1e-9)
    assert isotropic_noise_gain_db(np.arange(32) * 1e-6, f, c) == pytest.approx(0.0, abs=1e-6)                                         # 32 hydrophones in one place hear one noise
    hann = np.hanning(34)[1:-1]
    assert isotropic_noise_gain_db(np.arange(32) * 0.5 * lam, f, c, weights=hann) == pytest.approx(10 * np.log10(hann.sum() ** 2 / (hann ** 2).sum()), abs=1e-9)
    short, long_ = SonarModel.load("popoto_testauv_1row"), SonarModel.load("popoto_face_1row")
    eq_s, eq_l = SonarEquation.from_sonar(short, 50.0), SonarEquation.from_sonar(long_, 50.0)
    assert eq_l.array_gain() == pytest.approx(10 * np.log10(32), abs=1e-6) and eq_s.array_gain() == pytest.approx(bs.directivity_db(32, 0.09, 5000.0), abs=1e-9)
    assert eq_l.array_gain() - eq_s.array_gain() == pytest.approx(2.2, abs=0.1)
    assert SonarEquation.from_sonar(short, 50.0, array_gain_db=7.0).array_gain() == 7.0                                                # a given gain is kept
