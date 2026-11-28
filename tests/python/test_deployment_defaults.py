"""What the shipped deployment files switch on, and what the coverage picture refuses to run without.

Three decisions of the owner, made at the source (python/gpusonar/buildspec.py: chain_config; coverage.py: of_build,
chain_options; environment.py: coverage_site; libs/live: coverage_site_missing):

1. every stacked-row deployment file searches only the elevations a ray can arrive from (`elevation.sector_from_rays`);
2. wherever depth from multipath timing is on, it goes through rays and reports no flagged depth
   (`multipath_depth: {rays: true, withhold_flagged: true}`); no shipped file has it on;
3. the face and vehicle files carry the coverage picture (`chain.coverage`), and the water depth and the bottom are
   REQUIRED for it, with no default anywhere: the generator and the chain (both engines, and the C++ chain itself)
   refuse to run without them. Before, a chain without them drew the deep-water picture and said that a ship it was
   detecting at 12 km in 100 m over sand could not be seen.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import gpusonar as gs
from gpusonar import buildspec as bs
from gpusonar import coverage, environment, live
from gpusonar.config import load_yaml
from gpusonar.live import LiveChain
from gpusonar.sim import make_pose
from gpusonar.sonar import SonarModel

core = gs.core
ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "configs" / "deploy"
SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}
STACKED = {"popoto_static_2row", "popoto_static_3row", "popoto_face_2row", "popoto_vehicle_elevation", "popoto_elevation_rehearsal",
           "popoto_vehicle_elevation_ss5"}       # the last: the alternate design for sea state 5 (D-255)
WITH_COVERAGE = {"popoto_face_1row", "popoto_face_2row", "popoto_vehicle_elevation", "popoto_vehicle", "popoto_elevation_rehearsal",
                 "popoto_vehicle_elevation_ss5"}


def deployed() -> dict:
    return {p.stem: load_yaml(p) for p in sorted(DEPLOY.glob("*.yaml"))}


def tool(name):
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_stacked_row_deployment_file_searches_only_where_rays_arrive_from():
    """Found by the sonar's rows, not by a list of names: a new stacked-row file without the key fails here."""
    stacked = set()
    for name, cfg in deployed().items():
        if SonarModel.load(cfg["source"]["sonar"]).config.get("mode") == "fls":
            assert "elevation" not in cfg["chain"], name      # one row (sonar-tpex's files); buildspec describes board arrays only
            continue
        if bs.describe_shipped(cfg["source"]["sonar"])["rows"] > 1:
            stacked.add(name)
            assert cfg["chain"]["elevation"]["sector_from_rays"] == {}, name      # {}: two bounces, a degree of margin, to 500 m
            assert cfg["chain"]["elevation"]["multipath_sigma"] == {}, name       # {}: two bounces a leg, target depths to 500 m or the bottom
        else:
            assert "elevation" not in cfg["chain"], name
    assert stacked == STACKED
    for build in bs.shipped():                                                   # and it is the generator that writes it
        spec = bs.load(build)
        el = bs.chain_config(spec).get("elevation")
        assert (el is not None and el["sector_from_rays"] == {}) == (spec["array"]["rows"] > 1), build
        if el is not None:
            o = live.cpp_chain_options(bs.chain_config(spec))["elevation"]
            assert o["sector_from_rays"] is True and o["sector_from_depth"] == (spec["array"]["rows"] == 2)   # two rows keep the water-depth bound beside it
            assert el["multipath_sigma"] == {} and o["multipath_sigma"] is True


def test_multipath_depth_is_deployed_through_rays_and_withholds_flagged_depths():
    """No shipped build switches depth from multipath timing on. A build that does (`operation.chain.multipath_depth`) gets
    both keys from the generator, its own keys laid over them."""
    for name, cfg in deployed().items():
        mp = cfg["chain"].get("multipath_depth")
        assert mp is None or (mp.get("rays") is True and mp.get("withhold_flagged") is True), name
    assert [n for n, cfg in deployed().items() if cfg["chain"].get("multipath_depth") is not None] == []      # all off today: say so if that changes
    spec = bs.load("popoto_testauv_1x32")
    assert "multipath_depth" not in bs.chain_config(spec)                        # not newly enabled
    spec["operation"]["chain"] = {"multipath_depth": {"surface": {"rms_height_m": 0.1}}}
    chain = bs.chain_config(spec)
    assert chain["multipath_depth"] == {"rays": True, "withhold_flagged": True, "surface": {"rms_height_m": 0.1}}
    o = live.cpp_chain_options(chain)["multipath_depth"]
    assert o["rays"] == 1 and o["withhold_flagged"] == 1
    spec["operation"]["chain"] = {"multipath_depth": {"rays": False}}            # a build's own statement stands
    assert bs.chain_config(spec)["multipath_depth"] == {"rays": False, "withhold_flagged": True}


def test_face_and_vehicle_files_carry_the_coverage_picture_with_their_site():
    files = deployed()
    assert {n for n, cfg in files.items() if cfg["chain"].get("coverage") is not None} == WITH_COVERAGE
    for name in WITH_COVERAGE:
        cov = files[name]["chain"]["coverage"]
        assert cov["water_depth_m"] == 100.0 and cov["bottom"] == SAND, name
        assert all(isinstance(cov["budget"][k], float) for k in ("source_level_db", "noise_spectrum_db", "array_gain_db", "absorption_db_per_km")), name
        assert [c["name"] for c in cov["classes"]] == ["ship", "xl_auv", "large_submerged"]
        assert coverage.missing_site(cov) == [] and coverage.chain_options(cov)["water_depth"] == 100.0 and coverage.chain_options(cov)["bottom"] == 1
    # the generated ones: the spec names the site, the levels are buildspec.budget's at the site's wind, and nobody has measured any of it
    for build in ("popoto_face_1x32", "popoto_face_2x32", "popoto_face_3x32"):
        spec = bs.load(build)
        assert spec["operation"]["coverage"] == {"environment": "shelf_100m_sand"}
        cfg = bs.deploy_config(spec)
        cov = cfg["chain"]["coverage"]
        env = next(e for e in bs.environments_of(spec) if e["name"] == "shelf_100m_sand")
        want = coverage.budget_of(bs.describe(bs.sonar_config(spec)), env["wind_speed_m_s"])
        assert cov["budget"] == pytest.approx(want, abs=0.006)                    # written to 0.01 dB
        assert cov["classes"] == coverage.classes_of(bs.target_classes_of(spec, range_m=12700.0))
        assert cov["water_depth_m"] == env["water_depth_m"] and cov["bottom"] == env["bottom"] and cov["bottom_clearance_m"] == env["bottom_clearance_m"]
        assert cov["own_depth_limits_m"] == [5.0, spec["platform"]["max_depth_m"]]
        assert {"chain.coverage.water_depth_m", "chain.coverage.bottom", "chain.coverage.classes", "chain.coverage.budget"} <= set(cfg["unconfirmed"])
    for build in ("popoto_static_2x32", "popoto_static_3x32", "popoto_testauv_1x32"):
        assert "coverage" not in bs.chain_config(bs.load(build)) and not any("coverage" in f for f in bs.deploy_config(bs.load(build))["unconfirmed"])


@pytest.mark.parametrize("lacking, words", [("water_depth_m", "water_depth_m"), ("bottom", "bottom"), ("wind_speed_m_s", "wind_speed_m_s")])
def test_the_generator_refuses_a_coverage_site_without_its_facts(lacking, words):
    site = {"name": "somewhere", "water_depth_m": 80.0, "bottom": dict(SAND), "wind_speed_m_s": 6.0}
    spec = bs.load("popoto_face_1x32")
    spec["operation"]["coverage"] = {"environment": dict(site)}
    assert bs.chain_config(spec)["coverage"]["water_depth_m"] == 80.0             # an inline site is taken as written
    assert bs.chain_config(spec)["coverage"]["budget"]["noise_spectrum_db"] > bs.chain_config(bs.load("popoto_face_1x32"))["coverage"]["budget"]["noise_spectrum_db"]   # its wind, not 5 m/s
    site.pop(lacking)
    spec["operation"]["coverage"] = {"environment": site}
    with pytest.raises(ValueError, match=words) as exc:
        bs.chain_config(spec)
    assert "have no default" in str(exc.value)
    errors = [i for i in bs.validate(spec) if i.level == "error"]
    assert len(errors) == 1 and errors[0].where == "operation.coverage" and words in errors[0].message
    with pytest.raises(bs.SpecError, match=words):
        tool("build_from_spec").generate(spec, Path("/nonexistent/never-written"), diagrams=False)      # stops before anything is written
    with pytest.raises(environment.EnvironmentError, match=words):
        environment.coverage_site(site)


def test_the_generator_refuses_a_named_site_that_is_deep_water_or_unknown():
    spec = bs.load("popoto_face_3x32")
    spec["operation"]["coverage"] = {"environment": "deep_no_bottom_return"}     # one of the build's own environments: no depth, no bottom
    with pytest.raises(ValueError, match="water_depth_m, bottom"):
        bs.chain_config(spec)
    spec["operation"]["coverage"] = {"environment": "not_an_environment"}
    with pytest.raises(ValueError, match="not one of the build's environments"):
        bs.chain_config(spec)
    spec["operation"]["coverage"] = {"environment": "shelf_100m_sand", "water_depth_m": 100.0}
    with pytest.raises(ValueError, match="operation.coverage"):
        bs.chain_config(spec)                                                    # the site is the environment's; nothing is typed beside it
    with pytest.raises(TypeError):
        coverage.budget_of(bs.describe(bs.sonar_config(spec)))                   # the levels have no default wind either


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_chain_refuses_coverage_without_water_depth_or_bottom(engine):
    spec = bs.load("popoto_face_1x32")
    full = bs.chain_config(spec)["coverage"]
    base = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "engine": engine}
    for lacking in (("water_depth_m",), ("bottom",), ("water_depth_m", "bottom")):
        cov = {k: v for k, v in full.items() if k not in lacking}
        with pytest.raises(ValueError) as exc:
            LiveChain(dict(base, coverage=cov))
        text = str(exc.value)
        assert all(f"coverage.{k}" in text.split("Give")[0] for k in lacking) and "neither has a default" in text and "cannot be seen" in text
    with pytest.raises(ValueError, match="positive number"):
        LiveChain(dict(base, coverage=dict(full, water_depth_m=0.0)))
    with LiveChain(dict(base, coverage=full)):                                   # with both it starts
        pass


def test_the_cpp_chain_itself_refuses_too():
    """Not only the Python layer in front of it: build/bin/sonar_live is given the same options as key=value."""
    flat = coverage.cpp_options(bs.chain_config(bs.load("popoto_face_1x32"))["coverage"])
    base = dict(live.cpp_chain_options({"bearings_deg": [-60.0, 60.0, 0.5]}), backend=core.Backend.Host)      # as gpusonar.live hands it over
    core.LiveChainCpp(dict(base, coverage=flat))
    for lacking, words in ((("water_depth",), "coverage_water_depth"), (("bottom",), "coverage_bottom=1"), (("water_depth", "bottom"), "coverage_water_depth")):
        with pytest.raises(ValueError, match=words):
            core.LiveChainCpp(dict(base, coverage={k: v for k, v in flat.items() if k not in lacking}))


def test_the_site_depth_is_used_when_a_ping_carries_none_and_the_ship_at_12_km_is_seen():
    """The popoto_12km situation (sonar at 10 m, 100 m of water over sand, the record carries no water depth): with the
    site's depth and bottom the ship class is seen at 12 km; the picture the chain used to draw there (neither given)
    says it cannot be, which is why both are now required. A depth the ping carries comes first."""
    spec = bs.load("popoto_face_1x32")
    d = bs.derive(spec)
    # The deployment file now carries the site's reverberation (sonar-w2w3, the site rule): the picture then needs the beam's
    # integral, which a chain has from its record; here, numbers only, the row's own (reverb_budget.line_beam_integral).
    from gpusonar import reverb_budget
    sonar = {"f_centre": float(d["centre_hz"]), "bandwidth": float(d["band_hz"][1] - d["band_hz"][0]), "pulse": float(d["pulse_s"]),
             "beam_integral": reverb_budget.line_beam_integral(d["per_row"][0], d["pitch_m"], d["centre_hz"], d["sound_speed"])}
    cfg = bs.chain_config(spec)["coverage"]
    table = coverage.pd_table(SonarModel.from_config(bs.sonar_config(spec)).make_record(make_pose()), {})
    iso = coverage.water([0.0, 5000.0], [1500.0, 1500.0])

    def picture(options, water_depth):
        cov = core.Coverage(options)
        cov.set_pd_table(*table)
        return json.loads("{" + cov.update_numbers(sonar, 750.0, 12700.0, 10.0, 10.0, water_depth, iso) + "}")["coverage"]

    site = picture(coverage.chain_options(cfg), float("nan"))
    assert site["water_depth_m"] == 100.0 and site["water_depth_from"] == "site"
    assert coverage.could_have_seen(site, 12000.0)["ship"] > 0.9
    carried = picture(coverage.chain_options(cfg), 80.0)
    assert carried["water_depth_m"] == 80.0 and "water_depth_from" not in carried
    deep = picture(coverage.cpp_options({k: v for k, v in cfg.items() if k not in ("water_depth_m", "bottom")}), float("nan"))
    assert deep["water_depth_m"] is None and coverage.could_have_seen(deep, 12000.0)["ship"] < 0.1


def test_the_programs_options_carry_the_new_settings():
    """chain.options / --export-template: what the deployment file says, as key=value of build/bin/sonar_live."""
    gen = tool("build_from_spec")
    o = gen._options(bs.chain_config(bs.load("popoto_static_3x32")))
    assert o["elevation_sector_from_rays"] == 1 and float(o["elevation_sector_rays_margin_deg"]) == pytest.approx(1.0) and "elevation_sector_rays_margin" not in o
    assert o["elevation_multipath_sigma"] == 1 and float(o["elevation_multipath_sigma_bounces"]) == 2.0 and float(o["elevation_multipath_sigma_max_depth"]) == 500.0
    assert "coverage" not in o and "multipath_depth" not in o
    spec = bs.load("popoto_face_1x32")
    spec["operation"]["chain"] = {"multipath_depth": {}}
    chain = bs.chain_config(spec)
    record = SonarModel.from_config(copy.deepcopy(bs.sonar_config(spec))).make_record(make_pose())
    o = gen._options(chain, record)
    assert o["coverage"] == 1 and float(o["coverage_water_depth"]) == 100.0 and o["coverage_bottom"] == 1 and float(o["coverage_bottom_speed"]) == 1650.0
    assert o["coverage_classes"].startswith("ship:25.0:2.0:6.0;") and float(o["coverage_source_level_db"]) == 197.0
    snr, pd = coverage.pd_table(record, chain)
    assert [float(v) for v in o["coverage_pd_value"].split(",")] == pytest.approx(pd, abs=1e-9) and len(o["coverage_pd_snr_db"].split(",")) == len(snr)
    assert o["multipath_depth"] == 1 and o["multipath_depth_rays"] == 1 and o["multipath_depth_withhold_flagged"] == 1
    assert not any(isinstance(v, (dict, list)) for k, v in o.items() if k.startswith(("coverage", "multipath_depth", "elevation")))     # values the program can read


def test_the_rehearsal_sync_reference_reaches_the_cpp_receivers():
    """popoto_elevation_rehearsal.yaml's shared reference, read by plain PyYAML as well as by our loader, passes through
    the C++ readers of a reference (PingAssemblerCpp, PingDecimator: flank_udp with decimate and sync) (sonar-iwq).
    Before, it was written `f_start: 38.0e3`: YAML 1.1 reads that as a string, and the readers refused it with a cast
    error."""
    import yaml
    path = DEPLOY / "popoto_elevation_rehearsal.yaml"
    for cfg in (yaml.safe_load(path.read_text()), load_yaml(path)):
        ref = live.sync_reference(cfg["chain"])
        assert ref == {"f_start": 38000.0, "f_stop": 48000.0, "duration": 4.0e-3, "lead": 5.0e-3, "tail": 10.0e-3}
        assert all(type(v) is float for v in ref.values())
        template = SonarModel.load(cfg["source"]["sonar"]).make_record(gs.core.Pose())
        gs.core.PingAssemblerCpp(template, {}, ref)
        gs.core.PingDecimator(template, {}, ref)
