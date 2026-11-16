"""Milestone 4: FLS fan imaging on the two medical probes.

Acceptance: range resolution within 10% of theory, beam width within 10% of theory,
for `l12_5d` and `p4_2v`. What "theory" means here is spelled out in DECISIONS.md D-26.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.metrics import width_at_level
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

C = 1480.0


def point_response(name, r_target, bearing_deg, n_pings, window_m=0.02, raw=False):
    """Simulate one point target and return (sonar, records, target position)."""
    sonar = SonarModel.load(name, {"receive": {"window": {"range": [r_target - window_m, r_target + window_m]}}})
    b = np.radians(bearing_deg)
    target = r_target * np.array([np.cos(b), np.sin(b), 0.0])
    scene = Scene().add_points([target])
    records = [simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C},
                             absorption_model="none", adc=not raw, sub_sequence=i, sequence=i)
               for i in range(n_pings)]
    return sonar, records, target


def gaussian_range_width(bandwidth, level_db, compressed=True):
    """Width (m) of a Gaussian pulse's range response at `level_db`.

    A Gaussian spectrum of -6 dB width B has a time envelope exp(-t^2 / 2 sigma^2) with
    sigma = sqrt(2 ln 2) / (pi B). Matched filtering squares the spectrum, which
    multiplies sigma by sqrt(2).
    """
    sigma = np.sqrt(2 * np.log(2)) / (np.pi * bandwidth)
    if compressed:
        sigma *= np.sqrt(2)
    half = sigma * np.sqrt(2 * np.log(10 ** (-level_db / 20)))
    return C / 2 * 2 * half


@pytest.mark.parametrize("name, r, pings", [("l12_5d", 1.5, 2), ("p4_2v", 7.0, 1)])
def test_range_resolution_matches_theory(name, r, pings):
    sonar, records, target = point_response(name, r, 0.0, pings)
    B = sonar.bandwidth
    nominal = C / (2 * B)
    ranges = r + np.linspace(-4, 4, 801) * nominal
    img = recon.fan_image(records, ranges, [0.0])
    profile = np.abs(img.image[:, 0])
    assert abs(ranges[np.argmax(profile)] - r) < 0.02 * nominal     # the target is where it should be
    w3 = width_at_level(ranges, profile, -3.0)
    # Theory for this pulse shape, within 10%.
    assert w3 == pytest.approx(gaussian_range_width(B, -3.0), rel=0.10)
    # And against the brief's nominal c/2B: a compressed Gaussian's -3 dB width is
    # 0.88 c/2B, so the nominal figure is met within the same 10% band plus that factor.
    assert 0.80 * nominal < w3 < 1.0 * nominal


@pytest.mark.parametrize("name, r, pings, n_elements", [
    ("l12_5d", 1.5, 2, 128),      # full aperture from both mux halves
    ("l12_5d", 1.5, 1, 64),       # one mux half
    ("p4_2v", 7.0, 1, 64),
])
def test_beam_width_matches_theory(name, r, pings, n_elements):
    sonar, records, target = point_response(name, r, 0.0, pings)
    assert sum(rec.samples.n_channels for rec in records) == n_elements
    wavelength = C / sonar.center_frequency
    aperture = n_elements * 0.3e-3
    theory = 0.886 * wavelength / aperture                 # rad, -3 dB, uniform aperture
    bearings = np.linspace(-4, 4, 801) * theory
    if pings == 1 and name == "l12_5d":
        # One half of the array is centred 9.6 mm off the probe centre; the fan is drawn
        # about the centre of the elements actually used, so the target is slightly off zero.
        pass
    img = recon.fan_image(records, [np.linalg.norm(target - np.vstack([x.rx_positions() for x in records]).mean(axis=0))], bearings)
    profile = np.abs(img.image[0])
    w3 = width_at_level(bearings, profile, -3.0)
    assert w3 == pytest.approx(theory, rel=0.10), (w3, theory)


def test_two_ping_aperture_halves_the_beam_width():
    _, two, target = point_response("l12_5d", 1.5, 0.0, 2)
    _, one, _ = point_response("l12_5d", 1.5, 0.0, 1)
    theory = 0.886 * (C / 5e6) / (128 * 0.3e-3)
    bearings = np.linspace(-6, 6, 1201) * theory

    def width(records):
        centre = np.vstack([x.rx_positions() for x in records]).mean(axis=0)
        img = recon.fan_image(records, [np.linalg.norm(target - centre)], bearings)
        return width_at_level(bearings, np.abs(img.image[0]), -3.0)

    assert width(one) / width(two) == pytest.approx(2.0, rel=0.05)


def test_off_axis_target_is_imaged_at_its_bearing():
    for name, r, bearing in [("p4_2v", 7.0, 25.0), ("l12_5d", 1.5, 12.0)]:
        sonar, records, target = point_response(name, r, bearing, 2 if name == "l12_5d" else 1)
        bearings = np.radians(np.linspace(bearing - 3, bearing + 3, 601))
        ranges = r + np.linspace(-2e-3, 2e-3, 41)
        img = recon.fan_image(records, ranges, bearings)
        ir, ib = np.unravel_index(np.argmax(np.abs(img.image)), img.image.shape)
        assert abs(np.degrees(bearings[ib]) - bearing) < 0.02
        assert abs(ranges[ir] - r) < 1e-4


def test_hann_aperture_window_lowers_sidelobes_and_widens_the_beam():
    from gpusonar.metrics import peak_sidelobe_db
    sonar, records, target = point_response("p4_2v", 7.0, 0.0, 1)
    theory = 0.886 * (C / sonar.center_frequency) / (64 * 0.3e-3)
    bearings = np.linspace(-12, 12, 2401) * theory
    rect = np.abs(recon.fan_image(records, [7.0], bearings, window="rect").image[0])
    hann = np.abs(recon.fan_image(records, [7.0], bearings, window="hann").image[0])
    assert width_at_level(bearings, hann, -3.0) > 1.4 * width_at_level(bearings, rect, -3.0)
    # Wideband pulse: sidelobes sit lower than the narrowband -13.3 dB of a uniform aperture.
    assert peak_sidelobe_db(rect) < -12.0
    assert peak_sidelobe_db(hann) < peak_sidelobe_db(rect) - 10.0


# --- grating lobes of the linear probe -------------------------------------------------------

def test_grating_lobe_prediction_for_the_linear_probe():
    sonar = SonarModel.load("l12_5d")
    wavelength = C / sonar.center_frequency          # 0.296 mm against a 0.3 mm pitch
    pitch = 0.3e-3
    # Pitch is about one wavelength: no steering angle is free of grating lobes.
    assert recon.max_unambiguous_steer(pitch, wavelength) == 0.0
    lobes = recon.grating_lobes(pitch, wavelength, np.radians(20.0), sonar.beamwidth_az)
    assert len(lobes) == 1
    assert lobes[0]["angle"] == pytest.approx(np.arcsin(np.sin(np.radians(20.0)) - wavelength / pitch))
    # The half-wavelength phased array has none until end-fire.
    p4 = SonarModel.load("p4_2v")
    assert recon.grating_lobes(0.3e-3, C / p4.center_frequency, np.radians(40.0), p4.beamwidth_az) == []


def test_grating_lobe_appears_in_the_image_where_predicted():
    """Steer the linear probe to a target at 20 degrees and look for its grating lobe."""
    sonar, records, target = point_response("l12_5d", 1.5, 20.0, 2, window_m=0.05)
    wavelength = C / sonar.center_frequency
    predicted = recon.grating_lobes(0.3e-3, wavelength, np.radians(20.0), sonar.beamwidth_az)[0]
    bearings = np.radians(np.linspace(-60.0, 30.0, 1801))
    ranges = 1.5 + np.linspace(-0.03, 0.03, 61)
    img = recon.fan_image(records, ranges, bearings)
    by_bearing = np.abs(img.image).max(axis=0)
    main = by_bearing.max()
    away = np.degrees(bearings) < 10.0
    lobe_bearing = np.degrees(bearings[away][np.argmax(by_bearing[away])])
    lobe_level = 20 * np.log10(by_bearing[away].max() / main)
    # Wideband pulse: the lobe is smeared in angle (each frequency puts it elsewhere), so
    # it peaks near the centre-frequency prediction and sits below the narrowband level.
    assert abs(lobe_bearing - np.degrees(predicted["angle"])) < 6.0
    assert lobe_level < predicted["level_db"] + 1.0
    assert lobe_level > -40.0          # it is really there: this probe cannot steer cleanly
