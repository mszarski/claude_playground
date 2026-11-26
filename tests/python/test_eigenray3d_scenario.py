"""P5 of docs/tasks/PROPAGATION_3D.md: one bed for a whole scenario, and the illustrative scenario at the opt-in
`eigenray3d` tier (configs/scenarios/flank_seamount_3d.yaml).

Without Bellhop3D: the top-level `bathymetry` reaches the eigenray3d provider, every bottom reverberation item and
every point-scatterer seafloor (D-73's one description), a different one typed beside it is refused, and no shipped
scenario other than the new one has it; the new scenario runs through the simulator on both engines (host) with
3-D eigenrays (here from a stand-in for the program, so that no oracle is needed) and the records agree. With
Bellhop3D the scenario takes about 4 minutes (four runs of a minute each), so its own run is left to the gallery
tooling and the document's measurement (docs/tasks/PROPAGATION_3D.md P5).
"""

import numpy as np
import pytest

from gpusonar import environment
from gpusonar.config import list_configs, load_scenario_config
from gpusonar.sim import bellhop3d, run_scenario
from gpusonar.sonar import SonarModel

from test_sampled_engine import assert_same

BED = {"seamount": {"depth": 200.0, "height": 180.0, "center": [300.0, 2500.0], "sigma": 500.0}}


def expand(cfg):
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    return environment.apply_to_scenario(cfg, environment.resolve(cfg["environment"], strict=True), sonar)


def test_one_bed_reaches_the_provider_the_reverberation_and_the_seafloor():
    cfg = load_scenario_config("flank_seamount_3d", expand=False)
    cfg = dict(cfg, reverberation=[{"mu_db": -27.0, "geometry": "bistatic"}, {"model": "volume", "sv_db": -80.0},
                                   {"model": "lambert", "boundary": "surface", "mu_db": -40.0}],
               scene=cfg["scene"] + [{"type": "seafloor", "lambert_mu_db": -27.0, "extent": [10.0, 10.0]}])
    out = expand(cfg)
    assert out["propagation"]["bathymetry"] == BED
    assert out["reverberation"][0]["bathymetry"] == BED and "boundary_z" not in out["reverberation"][0]
    assert "bathymetry" not in out["reverberation"][1] and "bathymetry" not in out["reverberation"][2]
    assert out["scene"][1]["bathymetry"] == BED and "bathymetry" not in out["scene"][0]
    # A bed typed beside the shared one must be the same one.
    other = {"plane": {"depth": 200.0}}
    with pytest.raises(ValueError, match=r"propagation\.bathymetry"):
        expand(dict(cfg, propagation=dict(cfg["propagation"], bathymetry=other)))
    with pytest.raises(ValueError, match=r"reverberation\[0\]\.bathymetry"):
        expand(dict(cfg, reverberation=[{"mu_db": -27.0, "geometry": "bistatic", "bathymetry": other}]))
    # Another tier keeps its propagation as it was: the shared bed is for the eigenray3d provider only.
    flat = expand(dict(cfg, fidelity="image_source", propagation={}))
    assert "bathymetry" not in flat["propagation"]


def test_no_shipped_scenario_but_the_new_one_uses_the_new_keys():
    users = [n for n in list_configs("scenarios")
             if load_scenario_config(n, expand=False).get("fidelity") == "eigenray3d"
             or "bathymetry" in load_scenario_config(n, expand=False)]
    # sonar-d7sw added four scenes at the tier and a level-bed control (tests/python/test_3d_scenarios.py)
    assert users == ["flank_bank_3d", "flank_canyon_3d", "flank_level_3d", "flank_ridge_3d", "flank_seamount_3d",
                     "flank_seamount_3d_opening"]


def _stand_in(self, key, it, iz, ir):
    """Bellhop3D's place in the provider: per node a direct path, a surface bounce and a bottom bounce turned 3 degrees
    out of the bearing's plane by the bed, at the delays of the image geometry (so the echoes land where they would)."""
    cache = self._cache.setdefault(key, {})
    c = self.env.speed_at(key[2])
    for a, b, r_i in zip(it, iz, ir):
        bearing, r, z = np.radians(a * self.bearing_step), self.ranges[r_i], self.depths[b]
        paths = []
        for image_z, ns, nb, turn in ((z, 0, 0, 0.0), (-z, 1, 0, 0.0), (2 * self.env.depth - z, 0, 1, np.radians(3.0))):
            dz = image_z - key[2]
            length = np.hypot(r, dz)
            decl = np.arctan2(dz, r)
            paths.append((1.0 / length * (0.5 if nb else -1.0 if ns else 1.0), length / c, decl, bearing + turn,
                          -decl if (ns + nb) % 2 else decl, bearing - turn, ns, nb))
        p = np.array(paths)
        cache[(int(a), int(b), int(r_i))] = bellhop3d.Merged(p[:, 0].astype(complex), p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 5],
                                                             p[:, 6].astype(int), p[:, 7].astype(int), np.ones(3, dtype=int))


def test_the_scenario_runs_on_both_engines_with_3d_eigenrays(monkeypatch):
    """The simulator never sees the lookup: NumPy and C++ pairing and synthesis give the same record for eigenrays
    that leave their bearing's plane (host; the CUDA kernels take the same paths and are not run here)."""
    monkeypatch.setattr(bellhop3d.Bellhop3DProvider, "_trace", _stand_in)
    quiet = {"receiver": {"ambient_noise": {"level_db": -100.0}}}       # the echo alone (measured peak 1.1e-5)
    a = run_scenario(load_scenario_config("flank_seamount_3d", quiet, expand=False), engine="cpp").records[0]
    b = run_scenario(load_scenario_config("flank_seamount_3d", quiet, expand=False), engine="numpy").records[0]
    assert np.abs(a.samples.array).max() > 1e-6
    assert_same(a.samples.array, b.samples.array, "eigenray3d")


def test_time_spread_over_relief_says_what_it_assumes():
    """P6: the bottom's time spreading takes grazing angles as over a level bed; with the 3-D tier over relief it says so
    (a warning only), and stays silent on a level bed or without a bottom section."""
    import warnings
    from gpusonar.sim import timespread
    from gpusonar.sim.bellhop import OceanEnvironment
    env = OceanEnvironment(depth=200.0, sound_speed=1500.0, bottom_speed=1700.0, bottom_density=1.9, bottom_attenuation=0.5)
    spread = {"bottom": {"rms_height": 0.1, "mean_square_slope": 0.01}}
    relief = bellhop3d.Bellhop3DProvider(env, 8000.0, 5000.0, bathymetry=BED)
    with pytest.warns(RuntimeWarning, match="level bed"):
        timespread.wrap(relief, spread, frequency=8000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        timespread.wrap(bellhop3d.Bellhop3DProvider(env, 8000.0, 5000.0), spread, frequency=8000.0)
        timespread.wrap(bellhop3d.Bellhop3DProvider(env, 8000.0, 5000.0, bathymetry=BED), {"surface": {"wind_speed": 5.0}},
                        frequency=8000.0)
