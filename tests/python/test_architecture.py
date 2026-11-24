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
    # Ten: six kernels' stages, the stacked rows' elevation on the backprojection kernel, and three the table had as CPU
    # only although each has a CUDA translation unit (the expectation was 7 while the diagram was wrong): the far-field and
    # MVDR beamformers (libs/live/src/beamform_cuda.cu, D-122), motion estimation's delays (libs/motion/src/delay_cuda.cu,
    # D-148) and the multiscale detector's cell statistic (libs/live/src/detector_cuda.cu, D-150).
    assert len(with_cuda) == 10
    for unit in ("libs/live/src/beamform_cuda.cu", "libs/motion/src/delay_cuda.cu", "libs/live/src/detector_cuda.cu", "libs/sim/src/reverb_cuda.cu"):
        assert (ROOT / unit).is_file(), unit
    assert "backend" in inspect.signature(flank.farfield_beamform).parameters and "backend" in inspect.signature(flank.mvdr_beamform).parameters
    assert "backend" in gs.core.live_multiscale_detect.__doc__                    # the multiscale statistic
    from gpusonar import motion
    assert "cuda" in motion.DELAY_BACKENDS and motion.MotionOptions().delay_backend == "numpy"   # opt-in device delays
    from gpusonar import elevation
    assert "backend" in inspect.signature(elevation.profile).parameters           # elevation steers with the backprojection kernel, host or CUDA
    assert "backend" in inspect.signature(__import__("gpusonar.sim.reverb", fromlist=["x"]).add_reverberation).parameters   # reverberation
    assert "backend" in gs.core.doa_gate.__doc__                                   # sparse Bayesian bearings
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


def test_live_chain_diagram_marks_gpu_stages_as_the_code_and_the_benchmark_have_them():
    """LIVE_STAGES is what docs/architecture/live_chain.png is drawn from."""
    import json
    cuda = {key for key, _, _, on_gpu, _ in arch.LIVE_STAGES if on_gpu}
    assert cuda == {"prepare", "map", "cfar", "sbl", "elevation"}
    assert "backend" in inspect.signature(recon.prepare).parameters and "backend" in inspect.signature(detect.cfar).parameters
    assert "backend" in gs.core.doa_gate.__doc__ and "backend" in gs.core.backproject.__doc__
    # The benchmark the document quotes timed the chain with 1, 2 and 3 rows on both backends.
    bench = json.loads((ROOT / "docs" / "benchmarks" / "gpu_benchmark_all_l4.json").read_text())
    live = {c["sizes"]["rows"]: c for c in bench["cases"] if c["group"] == "live" and c["case"].endswith("engine cpp, doa map, backprojection")}
    assert set(live) == {1, 2, 3} and all(c["cuda_s"] < c["host_s"] for c in live.values())
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    for rows, digits in ((1, "0.015"), (2, "0.028"), (3, "0.040")):
        assert f"{live[rows]['cuda_s']:.3f}" == digits and digits in text
    for name in ("architecture/live_chain.png", "architecture/deployment_build.png"):
        assert name in text and (ROOT / "docs" / name).exists()
    assert [e for _, _, e, _, _ in arch.LIVE_STAGES].count("cpp") == 1              # doa: sbl runs the C++ engine
