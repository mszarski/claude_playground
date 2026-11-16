"""Milestone 12: the eigenray tier against normal modes (KRAKEN) at 8 kHz.

Both are external programs run as separate processes (GPUSONAR_BELLHOP,
GPUSONAR_KRAKEN_DIR); the test is skipped when either is missing.
"""

import numpy as np
import pytest

from gpusonar.sim import find_bellhop
from gpusonar.sim.bellhop import BellhopProvider, OceanEnvironment, run_bellhop
from kraken_oracle import find_kraken, run_kraken

pytestmark = pytest.mark.skipif(find_bellhop() is None or find_kraken() is None,
                                reason="needs Bellhop and KRAKEN (GPUSONAR_BELLHOP, GPUSONAR_KRAKEN_DIR)")

PEKERIS = dict(depth=100.0, sound_speed=1500.0, bottom_speed=1600.0, bottom_density=1.8, bottom_attenuation=0.5)
FREQUENCY, SOURCE_DEPTH, RECEIVER_DEPTH = 8000.0, 30.0, 50.0
RANGES = np.arange(1000.0, 10001.0, 500.0)


def provider_loss(max_arrivals):
    provider = BellhopProvider(OceanEnvironment(**PEKERIS), FREQUENCY, max_range=10500.0, range_step=25.0,
                               depth_step=2.0, max_arrivals=max_arrivals)
    points = np.column_stack([RANGES, np.zeros_like(RANGES), np.full_like(RANGES, RECEIVER_DEPTH)])
    es = provider.eigenrays(np.array([0.0, 0.0, SOURCE_DEPTH]), points)
    return -10.0 * np.log10(np.sum(np.where(es.valid, np.abs(es.amplitude) ** 2, 0.0), axis=1))


def test_incoherent_loss_of_rays_and_modes_agree(capsys):
    """Ray arrivals summed in power against modes summed in power: two different solutions
    of the same waveguide. They agree to a few tenths of a dB from 1 to 10 km."""
    modes = run_kraken(**PEKERIS, frequency=FREQUENCY, source_depth=SOURCE_DEPTH,
                       receiver_depths=[RECEIVER_DEPTH], ranges=RANGES, incoherent=True)[0]
    rays = run_bellhop(OceanEnvironment(**PEKERIS), FREQUENCY, SOURCE_DEPTH, [RECEIVER_DEPTH],
                       RANGES).transmission_loss(coherent=False)[0]
    diff = rays - modes
    with capsys.disabled():
        print(f"\n[kraken] rays minus modes, incoherent TL at 8 kHz: max |diff| {np.abs(diff).max():.2f} dB, "
              f"at 10 km {diff[-1]:+.2f} dB (TL {modes[-1]:.1f} dB)")
    assert np.abs(diff).max() < 0.5
    assert np.abs(diff[RANGES >= 2000.0]).max() < 0.1
    # Far beyond the water depth the loss approaches cylindrical spreading with bottom loss:
    # well below the spherical 80 dB at 10 km.
    assert modes[-1] < 70.0


def test_arrival_cap_of_the_provider(capsys):
    """The provider keeps the strongest `max_arrivals` rays. Eight is not enough at long
    range; the shipped 10 km scenario uses 80."""
    modes = run_kraken(**PEKERIS, frequency=FREQUENCY, source_depth=SOURCE_DEPTH,
                       receiver_depths=[RECEIVER_DEPTH], ranges=RANGES, incoherent=True)[0]
    few, many = provider_loss(8), provider_loss(80)
    with capsys.disabled():
        print(f"\n[kraken] one-way energy dropped by the arrival cap at 10 km: 8 arrivals "
              f"{few[-1] - modes[-1]:.1f} dB, 80 arrivals {many[-1] - modes[-1]:.2f} dB")
    assert np.abs(many - modes).max() < 0.5
    assert few[-1] - modes[-1] > 3.0
