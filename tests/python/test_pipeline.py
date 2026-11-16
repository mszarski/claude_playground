"""The framework of ASASIN Sec. II: a config-driven chain, a processing log with
performance metrics and data issues, debug products per stage, GeoTIFF output."""

import json

import numpy as np
import pytest
import yaml

import gpusonar as gs
from gpusonar import pipeline
from gpusonar.config import config_dir, load_yaml

rasterio = pytest.importorskip("rasterio")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("chain")
    cfg = load_yaml(config_dir() / "chains" / "stripmap_points.yaml")
    cfg["output"] = {"geotiff": str(out / "image.tif"), "log": str(out / "run.jsonl"),
                     "debug_dir": str(out / "debug")}
    cfg["source"]["overrides"] = {"environment": {"sound_speed": 1500.0, "frame_origin_lla": [-32.05, 115.74, 0.0]}}
    return pipeline.run(cfg, backend=gs.core.Backend.Host), out, cfg


def test_chain_runs_from_a_config_and_produces_the_image(run):
    result, out, cfg = run
    assert result.image.shape == result.display.shape == result.points.shape[:2]
    assert float(np.median(result.display)) == pytest.approx(0.2, abs=0.01)
    # The three targets of the scenario are the brightest things in the image.
    assert (result.display > 0.9).sum() >= 3


def test_processing_log_has_timings_choices_and_the_config(run):
    result, out, cfg = run
    entries = [json.loads(line) for line in open(out / "run.jsonl")]
    events = [e["event"] for e in entries]
    assert events[0] == "start" and events[-1] == "done"
    assert yaml.safe_load(entries[0]["config_yaml"])["steps"] == cfg["steps"]     # reproducible from the log
    stages = [e for e in entries if e["event"] == "stage"]
    assert [s["stage"] for s in stages] == ["prepare", "whiten", "sas_image", "drc"]
    assert all(s["seconds"] > 0 for s in stages)
    image_step = stages[2]
    assert image_step["parameters"]["pixel"] == [0.02, 0.04]
    assert 0 < image_step["ping_blocks_kept"] < image_step["ping_blocks_total"]   # culling reported
    assert entries[-1]["real_time_factor"] > 0
    assert any(e["event"] == "source" and e["pings"] == 90 for e in entries)


def test_debug_products_exist_for_every_stage(run):
    result, out, cfg = run
    files = sorted(p.name for p in (out / "debug").iterdir())
    assert files == ["00_prepare.npz", "01_whiten.npz", "02_sas_image.npz", "03_drc.npz"]
    with np.load(out / "debug" / "02_sas_image.npz") as z:
        assert z["image"].shape == result.image.shape
    with np.load(out / "debug" / "00_prepare.npz") as z:
        assert np.iscomplexobj(z["samples_ping0"])


def test_geotiff_is_georeferenced_where_the_targets_are(run):
    result, out, cfg = run
    from pyproj import Transformer
    with rasterio.open(out / "image.tif") as src:
        assert src.count == 1 and (src.height, src.width) == result.display.shape
        assert src.crs.to_epsg() == 32750                      # UTM zone 50 south, for Fremantle
        band = src.read(1)
        np.testing.assert_allclose(band, result.display, atol=1e-6)
        # The brightest pixel maps back to the first target's position (north 14.4 m,
        # east 41.2 m of the frame origin).
        # (Located on the complex image: the display values saturate across a target's
        # main lobe, so their argmax is not its centre.)
        row, col = np.unravel_index(np.argmax(np.abs(result.image)), band.shape)
        east, north = src.xy(row, col)
        assert src.tags()["gpusonar_frame"] == "EPSG:32750"
    e0, n0 = Transformer.from_crs(4326, 32750, always_xy=True).transform(115.74, -32.05)
    target_ne = min(([14.4, 41.2], [12.0, 38.0], [17.0, 45.0]), key=lambda t: abs(north - n0 - t[0]))
    assert north - n0 == pytest.approx(target_ne[0], abs=0.05)
    assert east - e0 == pytest.approx(target_ne[1], abs=0.08)


def test_multi_layer_geotiff_for_a_volume(tmp_path):
    from gpusonar.output import write_geotiff
    x, y = np.arange(0.0, 1.0, 0.1), np.arange(0.0, 0.6, 0.1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    layers = np.random.default_rng(0).random((4, 10, 6)).astype(np.float32)
    info = write_geotiff(str(tmp_path / "vol.tif"), layers, pts, None, [f"depth {d}" for d in range(4)])
    assert not info["georeferenced"]
    with rasterio.open(tmp_path / "vol.tif") as src:
        assert src.count == 4 and src.descriptions[2] == "depth 2"
        np.testing.assert_allclose(src.read(3), layers[2])
        assert "local NED" in src.tags()["gpusonar_frame"]
        # Local frame: easting = y, northing = x of the pixel centre.
        east, north = src.xy(3, 2)
        assert (north, east) == pytest.approx((0.3, 0.2))


def test_unknown_stage_and_bad_data_are_reported(tmp_path):
    with pytest.raises(ValueError, match="unknown stage"):
        pipeline.run({"source": {"type": "scenario", "name": "point_grid_p4_2v",
                                 "overrides": {"sonar_overrides": {"receive": {"window": {"range": [5.4, 5.6]}}}}},
                      "steps": [{"stage": "nope"}]}, backend=gs.core.Backend.Host)
