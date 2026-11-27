"""The scene trace and its views (python/gpusonar/viz, tools/scenario_view.py; docs/tasks/VISUALIZATION.md).

What is held: the trace's arrivals are the simulator's own (the same provider, the same pairing, both engines);
the drawn image-source paths are exact and bounce on the boundaries; the marked arrivals sit on the matched-filter
peaks of the simulated record; Bellhop's ray file is matched to the provider's eigenrays by bounces and travel
time; the audio export keeps its rate and duration; the models listed are the ones in force.
"""

import json
import os

import numpy as np
import pytest
from gpusonar.sim import engine as sim_engine
from gpusonar.sim import propagation as prop
from gpusonar.sim.simulator import run_scenario
from gpusonar.viz import audio, page2d, page3d
from gpusonar.viz import trace as tr


def _run(name, engine=None, **overrides):
    try:
        return run_scenario(name, overrides=overrides or None, engine=engine)
    except sim_engine.EngineUnavailable as exc:          # a module without the C++ simulator
        pytest.skip(str(exc))


def test_folded_image_paths_end_on_the_point_and_bounce_on_the_boundaries():
    p = prop.ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=80.0, bottom_coefficient=0.5, max_order=3, engine="numpy")
    src, pt = np.array([0.0, 0.0, 20.0]), np.array([[300.0, 40.0, 55.0]])
    eig = p.eigenrays(src, pt)
    for k in np.flatnonzero(eig.valid[0]):
        line, bounces = tr.folded_path(src, eig.launch[0, k], 1500.0 * eig.delay[0, k], 0.0, 80.0)
        np.testing.assert_allclose(line[-1], pt[0], atol=1e-6)
        assert sum(b == "surface" for _, b in bounces) == eig.n_surface[0, k]
        assert sum(b == "bottom" for _, b in bounces) == eig.n_bottom[0, k]
        for xyz, b in bounces:
            assert xyz[2] == pytest.approx(0.0 if b == "surface" else 80.0, abs=1e-9)
        # The folded path is as long as the image path: straight segments only.
        assert np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1)) == pytest.approx(1500.0 * eig.delay[0, k], rel=1e-12)


@pytest.mark.parametrize("engine", ["numpy", "cpp"])
def test_trace_delays_are_the_simulators_pairs(engine):
    result = _run("sst_echo_level", engine=engine)
    trace = tr.build_trace(result, engine=engine)
    rec = result.records[0]
    tx = np.asarray(rec.tx_position())
    elements = np.asarray(rec.rx_positions())
    ref = elements.mean(axis=0)
    pts = np.array([t["position"] for t in trace["targets"]])
    q, p, _, _, _ = prop.two_way_paths(tr.provider_for(result, engine), tx, ref, pts, np.ones(len(pts)), engine=engine)
    c = result.config["environment"]["sound_speed"]
    sim = np.sort(np.linalg.norm(q - tx, axis=1) / c + np.linalg.norm(p - ref, axis=1) / c)
    ours = np.sort([pp["delay_s"] for pp in trace["pairs"]])
    assert len(ours) == len(sim) == 9                    # 3 x 3 image paths, one bounce at most a leg
    np.testing.assert_allclose(ours, sim, rtol=0, atol=1e-12)
    for pp in trace["pairs"]:
        assert len(pp["channel_delay_s"]) == len(elements)


def test_per_bounce_losses_make_up_the_image_rays_amplitude():
    # The loss drawn at each bounce is the provider's own law at that grazing angle: for an image ray the product of
    # the bounce coefficients over the spherical spreading is the provider's amplitude, exactly.
    result = _run("sst_echo_level", engine="numpy")
    trace = tr.build_trace(result, engine="numpy")
    c = result.config["environment"]["sound_speed"]
    for leg in trace["legs"]:
        for which in ("out", "back"):
            for ray in leg[which]:
                spread_db = 20.0 * np.log10(ray["amplitude"] * c * ray["delay_s"])
                assert spread_db == pytest.approx(-sum(b["loss_db"] for b in ray["bounces"]), abs=1e-9)


def test_marked_arrivals_sit_on_the_matched_filter_peaks():
    result = _run("sst_echo_level", engine="numpy")       # no noise, no reverberation: the echoes alone
    trace = tr.build_trace(result, engine="numpy")
    t, env = page2d.compressed(result.records[0])
    dt = t[1] - t[0]
    for ch in range(env.shape[0]):
        for d in sorted({round(pp["channel_delay_s"][ch], 7) for pp in trace["pairs"]}):
            i = int(np.argmin(np.abs(t - d)))
            lo, hi = max(i - 3, 0), min(i + 4, env.shape[1])
            j = lo + int(np.argmax(env[ch, lo:hi]))
            # A local maximum of the compressed echo within 1.5 samples of the marked delay.
            assert abs(t[j] - d) <= 1.5 * dt, (ch, d, t[j])
            assert env[ch, j] >= env[ch, max(j - 1, 0)] and env[ch, j] >= env[ch, min(j + 1, env.shape[1] - 1)]


def test_bellhop_rays_are_matched_to_the_providers_eigenrays():
    from gpusonar.sim import bellhop
    if bellhop.find_bellhop() is None:
        pytest.skip("Bellhop is not installed (GPUSONAR_BELLHOP)")
    result = _run("flank_target_1km", engine="numpy", reverberation=[], sonar_overrides={"array": {"count": 4}},
                  receiver={"ambient_noise": {"level_db": 0.0}})
    trace = tr.build_trace(result, engine="numpy")
    depth = trace["environment"]["bottom_z"]
    rays = [r for leg in trace["legs"] for w in ("out", "back") for r in leg[w]]
    drawn = [r for r in rays if r["polyline"] is not None]
    assert len(drawn) >= 0.9 * len(rays)
    for r in drawn:
        assert sum(b["boundary"] == "surface" for b in r["bounces"]) == r["n_surface"]
        assert sum(b["boundary"] == "bottom" for b in r["bounces"]) == r["n_bottom"]
        for b in r["bounces"]:
            assert b["xyz"][2] == pytest.approx(0.0 if b["boundary"] == "surface" else depth, abs=1e-2)
            assert 0.0 < b["grazing_deg"] < 90.0 and b["loss_db"] is not None
        # Travel time along Bellhop's ray against the provider's delay (grid lookup plus its first-order
        # correction): within 2 ms at 1 km (ours: 1.2 ms the worst measured, 2026-11-26).
        assert abs(r["polyline_delay_s"] - r["delay_s"]) < 2e-3


def test_audio_export_keeps_rate_and_duration(tmp_path):
    result = _run("sst_echo_level", engine="numpy")
    rec = result.records[0]
    fs, y, how = audio.hydrophone_audio(rec, 0)
    assert how["mode"] == "remodulate" and how["carrier_hz"] == pytest.approx(rec.samples.center_frequency)
    info = audio.write_wav(str(tmp_path / "ch0.wav"), fs, y)
    rate, back = audio.read_wav(str(tmp_path / "ch0.wav"))
    assert rate == info["sample_rate"] and rate >= 2.5 * (rec.samples.center_frequency + 0.5 * rec.samples.sample_rate)
    assert len(back) / rate == pytest.approx(rec.samples.n_samples / rec.samples.sample_rate, abs=1.0 / rate)
    # The remodulated tone's spectrum peaks inside the transmit band (7-9 kHz).
    f = np.fft.rfftfreq(len(back), 1.0 / rate)
    assert 7000.0 <= f[np.argmax(np.abs(np.fft.rfft(back)))] <= 9000.0
    # Shifted down: the kept band (1.8 x the new centre) sits about the new centre, nothing folds through 0 Hz.
    fs2, y2, how2 = audio.hydrophone_audio(rec, 0, mode="shift", audio_center=1000.0)
    spec = np.abs(np.fft.rfft(y2))
    f2 = np.fft.rfftfreq(len(y2), 1.0 / fs2)
    assert how2["shift_hz"] == pytest.approx(7000.0) and np.sum(spec[f2 > 1900.0] ** 2) < 1e-6 * np.sum(spec ** 2)


def test_models_in_force_name_only_what_the_scenario_uses():
    s = tr.build_trace(_run("sst_echo_level", engine="numpy"), engine="numpy")
    models = {m["model"] for m in s["models"]}
    assert "bottom_constant" in models and "rayleigh_fluid" not in models and "lambert" not in models
    assert s["environment"]["bottom"] == {"coefficient": 0.5}       # not the site's sediment row
    for m in s["models"]:
        assert m["status"] and m["source"]


def test_every_scenario_has_a_models_list():
    from gpusonar import acoustics
    from gpusonar import environment as site
    from gpusonar.config import load_scenario_config
    from gpusonar.sonar import SonarModel
    root = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "scenarios")
    names = sorted(f[:-5] for f in os.listdir(root) if f.endswith(".yaml"))
    assert len(names) == 96
    for name in names:
        cfg = acoustics.resolve_sea_state(load_scenario_config(name, None, expand=False))
        sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
        env = site.resolve(cfg.get("environment"), strict=True)
        cfg = dict(site.apply_to_scenario(cfg, env, sonar), environment=env)
        models = tr.models_in_force(cfg)
        assert models[0]["model"] == cfg.get("fidelity", "free_field"), name


def test_pages_hold_what_they_draw(tmp_path):
    result = _run("sst_echo_level", engine="numpy")
    trace = tr.build_trace(result, engine="numpy")
    drawn = page2d.draw_page(trace, result.records[0], str(tmp_path / "p.png"))
    assert drawn["rays"] == 6 and drawn["bounces"] == 4 and drawn["marked"] == 9
    assert os.path.getsize(tmp_path / "p.png") < 300_000
    info = page3d.write_page(trace, str(tmp_path / "p.html"))
    text = (tmp_path / "p.html").read_text(encoding="utf-8")
    blob = text.split('<script type="application/json" id="trace">', 1)[1].split("</script>", 1)[0]
    data = json.loads(blob)
    assert info["rays"] == 6 and sum(len(leg["out"]) + len(leg["back"]) for leg in data["legs"]) == 6
    assert page3d.THREE_CDN in text and info["bytes"] < 100_000
