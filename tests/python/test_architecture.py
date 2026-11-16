"""The backend table in docs/ARCHITECTURE.md is drawn from tools/architecture_diagrams.py.
A stage listed there with a CUDA implementation must really take a backend argument."""

import importlib.util
import inspect
import pathlib

import gpusonar as gs
from gpusonar import detect, flank, recon
from gpusonar.live import LiveChain

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("arch", ROOT / "tools" / "architecture_diagrams.py")
arch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arch)


def test_stages_listed_with_a_gpu_implementation_take_a_backend():
    with_cuda = {stage for _, stage, _, cuda, _, _ in arch.STAGES if cuda}
    assert len(with_cuda) == 4
    assert "backend" in inspect.signature(recon.prepare).parameters                # conditioning
    assert "backend" in inspect.signature(detect.cfar).parameters                  # CFAR
    assert "backend" in gs.core.backproject.__doc__ and "backend" in gs.core.synthesize_into.__doc__
    assert "backend" in inspect.signature(recon.fan_image).parameters              # what flank.backprojection_map calls
    assert flank.backprojection_map.__doc__


def test_live_chain_defaults_to_the_kernels_that_have_a_gpu_build():
    chain = LiveChain()
    assert chain.cfg["beamformer"] == "backprojection" and chain.cfg["backend"] == "auto"
    assert chain.backend == (gs.core.Backend.Cuda if gs.cuda_available() else gs.core.Backend.Host)
    chain.close()


def test_documents_point_at_files_that_exist():
    for doc in ("ARCHITECTURE.md", "DEPLOYMENT.md", "SIM_CALIBRATION.md"):
        text = (ROOT / "docs" / doc).read_text()
        for name in ("architecture/dataflow.png", "architecture/backends.png", "architecture/deployment.png",
                     "architecture/sim_calibration.png"):
            if name in text:
                assert (ROOT / "docs" / name).exists(), name
    for tool in ("sonar_live.py", "inspect_tap.py", "calibrate_sim.py", "architecture_diagrams.py"):
        assert (ROOT / "tools" / tool).exists()
    assert (ROOT / "configs" / "deploy" / "popoto_vehicle.yaml").exists()
