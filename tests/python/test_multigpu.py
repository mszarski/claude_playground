"""recon.multigpu.backproject_tiles: tiles of image rows give the same image as one call (audit B8).

The host part runs on every build. The CUDA part needs a CUDA build and a device and skips
otherwise, saying which is missing."""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.recon.multigpu import backproject_tiles, make_session
from gpusonar.sim import run_scenario

core = gs._gpusonar

# sas_image's own defaults, spelled out for the tiled call (it takes core.backproject's).
SAS_DEFAULTS = dict(interp="sinc", fov_factor=1.13, cull_block=16, position_lookup="analytic")


@pytest.fixture(scope="module")
def stripmap():
    records = run_scenario("stripmap_points").records
    recon.prepare(records)
    grid = recon.stripmap_grid(records, (38.0, 44.0), (0.05, 0.1))
    reference = recon.sas_image(records, grid.points)          # one host call over the whole grid
    return records, grid, reference


@pytest.mark.parametrize("n_tiles", [1, 3, 4])
def test_host_tiles_equal_one_sas_image_call(stripmap, n_tiles):
    records, grid, reference = stripmap
    img = backproject_tiles(records, grid.points, devices=list(range(n_tiles)), backend=core.Backend.Host,
                            **SAS_DEFAULTS)
    assert img.shape == reference.shape
    assert np.abs(reference).max() > 0
    # Pixels are independent and each tile sums the same pings in the same order: equal to the bit.
    np.testing.assert_array_equal(img, reference)


def test_host_tiles_reuse_a_session_and_match_without_culling(stripmap):
    records, grid, _ = stripmap
    kw = dict(interp="sinc", fov_factor=1.13, position_lookup="analytic")
    session = make_session(records, dict(kw))
    whole = core.backproject(records, grid.points.reshape(-1, 3), backend=core.Backend.Host, **kw)
    tiled = backproject_tiles(records, grid.points, devices=[0, 0], backend=core.Backend.Host, session=session, **kw)
    np.testing.assert_array_equal(tiled, whole.reshape(grid.shape))


def test_empty_device_list_is_refused(stripmap):
    records, grid, _ = stripmap
    with pytest.raises(ValueError, match="devices is empty"):
        backproject_tiles(records, grid.points, devices=[], backend=core.Backend.Host)


def test_cuda_tiles_match_the_host_image(stripmap):
    if not core.cuda_compiled():
        pytest.skip("CPU-only build (SONAR_ENABLE_CUDA=OFF): the CUDA tiles are not compiled")
    if core.cuda_device_count() < 1:
        pytest.skip("CUDA build but no CUDA device visible")
    records, grid, reference = stripmap
    img = backproject_tiles(records, grid.points, **SAS_DEFAULTS)            # default: one tile per device
    scale = np.abs(reference).max()
    # The device kernel is not bit-equal to the host one (float32 order, fused multiply-adds): a relative rms bound.
    assert np.sqrt(np.mean(np.abs(img - reference) ** 2)) / scale < 1e-4
