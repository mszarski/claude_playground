"""Statistical properties of the reverberation generator (sim/reverb.py, libs/sim/src/reverb.cpp).

1. K-distributed reverberation: a gamma texture of unit mean per resolution cell on top of
   the Gaussian speckle. Checked against the closed-form intensity moments of that compound
   (K) distribution, the Rayleigh limit and the unchanged mean power.
2. The spread-against-coherence results of TR 0702 Sec. 10.6, which any reverberation
   generator should satisfy (docs/PHYSICS_NOTES.md Sec. 5.1 and 9): a Gaussian spread in time
   gives a Gaussian coherence in frequency of width 1 / (2 pi sigma_T) (eq. 89), in Doppler
   a coherence in time of width 1 / (2 pi f sigma_gamma) (eq. 96), in direction a coherence
   along the array of width lambda / (2 pi sigma_theta) (eq. 103).
3. Replica correlation gains 3 dB of signal-to-reverberation ratio per octave of bandwidth
   (TR 0702 p. 70).

The closed forms are checked on both engines where the generator itself is what is tested.
"""

import warnings

import numpy as np
import pytest

from gpusonar import _gpusonar as core
from gpusonar.sim import FreeFieldProvider, Scene, make_pose, reverb, simulate_ping
from gpusonar.sim import engine as sim_engine
from gpusonar.sonar import SonarModel

C = 1500.0
ENGINES = ["numpy"] + (["cpp"] if sim_engine.HAVE_CPP else [])


def generate(engine, rng, density, coarse, *, n_low, fs, n_azimuth, fc=10e3, pulse=(1.0,), offsets=((0.0, 0.0, 0.0),),
             velocity=(0.0, 0.0, 0.0), components=None, texture=None):
    """The generator alone: the series [channels, n_low] of a given power density
    [rows, len(coarse)] (per radian and per second), at the rate `fs`, no decimation."""
    offsets = np.asarray(offsets, dtype=float)
    pulse = np.asarray(pulse, dtype=complex)
    density = np.asarray(density, dtype=float)
    if engine == "cpp":
        return reverb._series_cpp(rng, coarse, density, False, pulse, offsets, fs, n_low, 0.0, 0.0, fc, C, n_azimuth, 1,
                                  n_low, velocity, False, False, core.Backend.Host, components, texture)
    return reverb._series_numpy(rng, coarse, density, pulse, offsets, np.zeros(3), fs, n_low, 0.0,
                                np.array([0.0, 1.0 / fs]), fc, C, n_azimuth, 1, n_low, np.asarray(velocity), False,
                                False, components, texture)


def flat(n_rows, level=1.0):
    return np.array([-1.0, 1e6]), np.full((n_rows, 2), level)


# --- K-distributed reverberation ---------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", [0.5, 2.0, 10.0])
def test_k_distributed_intensity_moments(engine, shape):
    """One cell per sample: intensity = gamma texture (unit mean, shape nu) x exponential
    speckle. Its moments are those of the K distribution:

        E{I^2} / E{I}^2 = 2 (1 + 1/nu),     E{I^3} / E{I}^3 = 6 (1 + 1/nu) (1 + 2/nu)

    (moments of a gamma variate, E{tau^n} = prod_{k<n} (1 + k/nu), times n! for the
    exponential), and the mean power is the Rayleigh one."""
    n, fs = 400_000, 1000.0
    coarse, density = flat(1, level=fs / (2 * np.pi))             # power 1 per sample: sigma^2 = 1/2
    rng = np.random.default_rng(int(10 * shape))
    texture = reverb.CellTexture(rng, shape, n, 1, [0])
    x = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=1, texture=texture)[0]
    inten = np.abs(x) ** 2
    mean = inten.mean()
    assert mean == pytest.approx(1.0, rel=0.02 if shape > 1 else 0.03)       # unit-mean texture: power unchanged
    m2, m3 = (inten**2).mean() / mean**2, (inten**3).mean() / mean**3
    assert m2 == pytest.approx(2.0 * (1.0 + 1.0 / shape), rel=0.06)
    if shape >= 2.0:
        assert m3 == pytest.approx(6.0 * (1.0 + 1.0 / shape) * (1.0 + 2.0 / shape), rel=0.15)
    assert m2 > 2.15                                                         # heavier tailed than Rayleigh


@pytest.mark.parametrize("engine", ENGINES)
def test_large_shape_recovers_rayleigh(engine):
    n, fs = 200_000, 1000.0
    coarse, density = flat(1, level=fs / (2 * np.pi))
    out = {}
    for shape in (1e7, None):
        rng = np.random.default_rng(2)
        texture = None if shape is None else reverb.CellTexture(rng, shape, n, 1, [0])
        if shape is None:
            rng.integers(0, 2**63 - 1)                                       # the draw that seeds a texture
        out[shape] = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=1, texture=texture)[0]
    inten = np.abs(out[1e7]) ** 2
    assert (inten**2).mean() / inten.mean() ** 2 == pytest.approx(2.0, rel=0.02)
    # Sample for sample the Gaussian sequence, to the 1 / sqrt(shape) of the texture.
    assert np.abs(out[1e7] - out[None]).max() < 5e-3 * np.abs(out[None]).max()
    # Rayleigh envelope: mean / rms = sqrt(pi) / 2.
    assert np.abs(out[1e7]).mean() / np.sqrt(inten.mean()) == pytest.approx(np.sqrt(np.pi) / 2, rel=0.01)


@pytest.mark.parametrize("engine", ENGINES)
def test_texture_is_shared_within_a_cell_and_independent_between_cells(engine):
    """Cells in time: samples of one cell share the texture, so their intensities are
    correlated (E{I_a I_b} / E{I}^2 = 1 + 1/nu), and samples of different cells are not.
    Cells in azimuth: a hydrophone that sums N cells of equal power sees
    E{I^2} / E{I}^2 = 2 (1 + 1 / (N nu)), on its way to Rayleigh."""
    shape, n, fs, cell = 1.5, 400_000, 1000.0, 50
    coarse, density = flat(1, level=fs / (2 * np.pi))
    rng = np.random.default_rng(8)
    x = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=1,
                 texture=reverb.CellTexture(rng, shape, n, cell, [0]))[0]
    inten = (np.abs(x) ** 2).reshape(-1, cell)
    mean = inten.mean()
    same = (inten[:, :-1] * inten[:, 1:]).mean() / mean**2          # neighbours within a cell
    other = (inten[:-1] * inten[1:]).mean() / mean**2               # the same place in the next cell
    assert same == pytest.approx(1.0 + 1.0 / shape, rel=0.05)
    assert other == pytest.approx(1.0, rel=0.03)
    # Eight sectors at one hydrophone, grouped into 8, 2 and 1 azimuth cells.
    n = 200_000
    coarse, density = flat(8, level=fs / (2 * np.pi))
    for per_cell, cells in ((1, 8), (4, 2), (8, 1)):
        rng = np.random.default_rng(9)
        texture = reverb.CellTexture(rng, shape, n, 1, np.arange(8) // per_cell)
        y = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=8, texture=texture)[0]
        inten = np.abs(y) ** 2
        assert (inten**2).mean() / inten.mean() ** 2 == pytest.approx(2.0 * (1.0 + 1.0 / (cells * shape)), rel=0.05)


@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
def test_texture_is_the_same_on_both_engines():
    n, fs = 4096, 1000.0
    coarse, density = flat(6, level=3.0)
    out = []
    for engine in ("numpy", "cpp"):
        rng = np.random.default_rng(5)
        texture = reverb.CellTexture(rng, 0.8, n, 16, np.arange(6) // 2)
        out.append((generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=6, texture=texture,
                             offsets=[[0, 0, 0], [0.05, 0, 0], [0.1, 0.02, 0]]), rng.bit_generator.state))
    assert np.abs(out[0][0] - out[1][0]).max() < 5e-6 * np.abs(out[0][0]).max()
    assert out[0][1] == out[1][1]


def empty_ping(ranges=(25.0, 230.0), fs=20e3, waveform=None, name="popoto_8"):
    wf = {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3, "duration": 2e-3, "window_alpha": 0.0,
          "source_level_db_rms": None, "amplitude": 1.0}
    wf.update(waveform or {})
    sonar = SonarModel.load(name, {
        "receive": {"kind": "baseband", "sample_rate": fs, "window": {"range": list(ranges)}, "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": wf}})
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, -20.0)), Scene(), environment={"sound_speed": C},
                         adc=False)


def test_k_statistics_in_a_ping():
    """Through `add_reverberation`: one azimuth cell, cells of 40 ms (twenty pulse lengths).
    The intensity, with the range trend taken out, has the K moment (a little below it:
    a sample whose pulse straddles two cells averages their textures); the default is
    Rayleigh; the mean level is the same. (`band_taper`: the level falls by 59 dB over this
    record, and without it the generator's leakage from the start raises the last samples.)"""
    stats = {"model": "k", "shape": 1.0, "cell_time": 0.04}
    moment, level = {}, {}
    for label, statistics in (("k", stats), ("rayleigh", None)):
        norm = []
        for seed in range(12):
            rec = empty_ping(ranges=(100.0, 1500.0))
            power = reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(seed),
                                             n_azimuth=1, statistics=statistics, band_taper=True)
            norm.append(np.abs(rec.samples.array[0]) ** 2 / power)
        norm = np.concatenate(norm)
        moment[label] = (norm**2).mean() / norm.mean() ** 2
        level[label] = norm.mean()
    assert moment["rayleigh"] == pytest.approx(2.0, rel=0.05)
    assert 3.2 < moment["k"] < 4.4                                   # 2 (1 + 1/nu) = 4
    assert level["k"] == pytest.approx(level["rayleigh"], rel=0.1)


def test_statistics_options():
    def run(statistics, **kw):
        rec = empty_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(1), n_azimuth=12,
                                 statistics=statistics, **kw)
        return rec.samples.array

    plain = run(None)
    assert np.array_equal(run({"model": "rayleigh"}), plain)
    assert np.array_equal(run({"model": "k", "shape": float("inf")}), plain)      # no texture, nothing drawn
    assert not np.array_equal(run({"model": "k", "shape": 1.0}), plain)
    if sim_engine.HAVE_CPP:
        a, b = run({"model": "k", "shape": 1.0}, engine="cpp"), run({"model": "k", "shape": 1.0}, engine="numpy")
        assert np.abs(a - b).max() < 5e-6 * np.abs(b).max()
    with pytest.raises(ValueError, match="shape"):
        run({"model": "k"})
    with pytest.raises(ValueError, match="positive"):
        run({"model": "k", "shape": 0.0})
    with pytest.raises(ValueError, match="statistics"):
        run({"model": "weibull"})


# --- the K shape from a density of scatterers (Abraham and Lyons 2002) --------------------------
#
# D. A. Abraham and A. P. Lyons, "Novel physical interpretations of K-distributed reverberation",
# IEEE J. Oceanic Eng. 27(4), 800-813 (2002): n exponentially sized discrete scatterers in a
# resolution cell give a K-distributed envelope of shape alpha = n / 2 (eq. 15); n patches of
# exponentially distributed area give alpha = n (eq. 19). Its K distribution is the envelope PDF
# of eq. (7) (power E{Y^2} = alpha lambda), with envelope moments eq. (28).

def paper_exceedance(t, alpha):
    """P(I > t) for the paper's K distribution, eq. (7), at unit mean intensity (lambda = 1 / alpha).
    Integrating eq. (7) from y to infinity with d/dx [x^a K_a(x)] = -x^a K_(a-1)(x):
    P(Y > y) = 2 / Gamma(alpha) (y / sqrt(lambda))^alpha K_alpha(2 y / sqrt(lambda)), and I = Y^2."""
    from scipy import special
    x = 2.0 * np.sqrt(alpha * np.asarray(t, dtype=float))
    # In logarithms, with the exponentially scaled Bessel function, so that it holds far into the tail.
    return np.exp(np.log(2.0) - special.gammaln(alpha) + alpha * np.log(0.5 * x) + np.log(special.kve(alpha, x)) - x)


def texture_exceedance(t, shape):
    """P(I > t) for our compound: unit-mean gamma texture tau of shape `shape` times an exponential
    speckle of mean tau, P(I > t) = E_tau{exp(-t / tau)}, by numerical integration over tau."""
    from scipy import integrate, special
    log_pdf = lambda tau: shape * np.log(shape) + (shape - 1.0) * np.log(tau) - shape * tau - special.gammaln(shape)  # noqa: E731
    f = lambda tau: np.exp(-t / tau + log_pdf(tau))                                                                  # noqa: E731
    peak = max(np.sqrt(t / shape), 1e-6)                                    # the integrand's mode, near sqrt(t / shape)
    return sum(integrate.quad(f, a, b, limit=400, epsabs=0.0, epsrel=1e-10)[0]
               for a, b in ((0.0, peak), (peak, 10.0 * peak + 10.0), (10.0 * peak + 10.0, np.inf)))


def threshold_db_above_rayleigh(alpha, pfa=1e-6):
    """The K threshold at false-alarm probability `pfa` over the Rayleigh one of the same power, dB."""
    from scipy import optimize
    rayleigh = np.log(1.0 / pfa)                                             # exp(-t) = pfa at unit power
    t = np.exp(optimize.brentq(lambda lt: np.log(paper_exceedance(np.exp(lt), alpha)) - np.log(pfa), 0.0, 8.0))
    return 10.0 * np.log10(t / rayleigh)


@pytest.mark.parametrize("shape", [0.3, 1.0, 5.0, 10.0, 60.0])
def test_our_shape_is_the_papers_alpha(shape):
    """Our texture (unit-mean gamma of shape nu on the power, times exponential speckle) is the
    paper's K distribution with alpha = nu, lambda = mean power / nu:
    1. moments: eq. (28), E{Y^i} = lambda^(i/2) Gamma(1 + i/2) Gamma(alpha + i/2) / Gamma(alpha), equals
       ours, E{tau^(i/2)} E{|Z|^i} = Gamma(nu + i/2) / (Gamma(nu) nu^(i/2)) x Gamma(1 + i/2), for every i
       (both written out here and compared; the odd i test the envelope, not only the intensity);
    2. the whole distribution: the exceedance of eq. (7) in closed form against the integral over our
       gamma texture, at levels from the median to 1e-8;
    3. a sample: CellTexture's draws times exponential speckle against eq. (28) for i = 2, 4."""
    from scipy import special
    alpha, lam = shape, 1.0 / shape
    for i in range(1, 7):
        paper = lam ** (i / 2) * special.gamma(1 + i / 2) * special.gamma(alpha + i / 2) / special.gamma(alpha)
        ours = special.gamma(shape + i / 2) / (special.gamma(shape) * shape ** (i / 2)) * special.gamma(1 + i / 2)
        assert ours == pytest.approx(paper, rel=1e-12)
    assert paper_exceedance(1.0, 1e-9 + shape) > 0                       # (finite)
    for p in (0.5, 1e-2, 1e-4, 1e-6, 1e-8):
        from scipy import optimize
        t = np.exp(optimize.brentq(lambda lt: np.log(paper_exceedance(np.exp(lt), alpha)) - np.log(p), -12.0, 8.0))
        assert texture_exceedance(t, shape) == pytest.approx(p, rel=1e-5)
    n = 2_000_000
    rng = np.random.default_rng(31)
    tau = reverb.CellTexture(rng, shape, n, 1, [0])(0) ** 2
    inten = tau * rng.exponential(1.0, n)
    for i, rel in ((2, 0.01), (4, 0.05)):
        paper = lam ** (i / 2) * special.gamma(1 + i / 2) * special.gamma(alpha + i / 2) / special.gamma(alpha)
        assert np.mean(inten ** (i / 2)) == pytest.approx(paper, rel=rel if shape >= 1.0 else 4 * rel)


def test_the_papers_worked_example():
    """Sec. IV: 20 scatterers in a range-bearing cell give alpha = 10, and at Pfa = 1e-6 the K
    threshold is 1.7 dB above the Rayleigh one of the same power (Fig. 1, Case 1); with the
    beamwidth halved, alpha = 5 and 2.7 dB (Case 2). Computed from the exceedance of eq. (7):
    1.685 and 2.694 dB, within 0.05 dB of the paper's one-decimal values. Our texture's integral
    gives the same thresholds."""
    assert reverb.k_shape_of_count(20.0, "scatterers_per_m2") == 10.0                           # eq. (15)
    assert reverb.k_shape_of_count(20.0, "patches_per_m2") == 20.0                              # eq. (19)
    for alpha, printed in ((10.0, 1.7), (5.0, 2.7)):
        ours = threshold_db_above_rayleigh(alpha)
        assert ours == pytest.approx(printed, abs=0.05)
        t = np.log(1e6) * 10.0 ** (ours / 10.0)
        assert texture_exceedance(t, alpha) == pytest.approx(1e-6, rel=1e-5)
    assert threshold_db_above_rayleigh(10.0) == pytest.approx(1.685, abs=0.001)
    assert threshold_db_above_rayleigh(5.0) == pytest.approx(2.694, abs=0.001)


def test_the_cell_area():
    """The area of the boundary between the slant ranges of a cell's two edges, r1 = c t1 / 2 and
    r2 = c t2 / 2, inside an azimuth width w, for straight paths from a point a height h above a
    flat boundary: w / 2 (R2^2 - R1^2) with R^2 = r^2 - h^2 the horizontal range, = w / 2 (r2^2 - r1^2)
    once r1 >= h, that is w x r_mid x (r2 - r1): the slant range times the slant-range extent, the
    grazing angle cancelling (the ring is 1 / cos wider than the slant extent and cos nearer)."""
    c, w, h = 1500.0, np.radians(2.0), 25.0
    t1, t2 = np.array([0.0, 0.03, 0.04, 0.2, 0.4]), np.array([0.03, 0.04, 0.05, 0.21, 0.41])
    area = reverb.k_cell_area(t1, t2, c, w, h)
    r1, r2 = c * t1 / 2, c * t2 / 2
    R = lambda r: np.sqrt(np.maximum(r**2 - h**2, 0.0))                     # noqa: E731
    assert area == pytest.approx(0.5 * w * (R(r2) ** 2 - R(r1) ** 2), rel=1e-12)
    assert area[0] == 0.0 and 0.0 < area[1] < area[2]                         # before, across and after the boundary
    assert area[3:] == pytest.approx(w * 0.5 * (r1 + r2)[3:] * (r2 - r1)[3:], rel=1e-12)
    assert area[4] / area[3] == pytest.approx((r1 + r2)[4] / (r1 + r2)[3], rel=1e-12)   # proportional to range
    assert reverb.k_cell_area(t1, t2, c, 2 * w, h) == pytest.approx(2 * area, rel=1e-12)


class Recorded(reverb.CellTexture):
    """CellTexture that keeps every texture made (to read the shapes add_reverberation chose)."""
    made = []

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        Recorded.made.append(self)


def chosen_texture(monkeypatch, statistics, *, waveform=None, ranges=(100.0, 1500.0), **kw):
    monkeypatch.setattr(reverb, "CellTexture", Recorded)
    Recorded.made.clear()
    rec = empty_ping(ranges=ranges, waveform=waveform)
    reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(1), statistics=statistics, **kw)
    (texture,) = Recorded.made
    return texture


def test_the_shape_follows_the_cell(monkeypatch):
    """shape = n / 2 per cell, n = scatterers_per_m2 x cell area (patches: shape = n): proportional to
    the range, to the azimuth width and to the cell's length in time (1 / bandwidth by default)."""
    density = 0.2
    base = chosen_texture(monkeypatch, {"model": "k", "scatterers_per_m2": density, "cell_azimuth_deg": 30.0},
                          n_azimuth=48)
    assert np.ndim(base.shape) == 1 and len(base.shape) == -(-base.n_low // base.cell)
    t1 = base.start_time + np.arange(len(base.shape)) * base.cell / base.rate
    t2 = t1 + base.cell / base.rate
    mid = 0.25 * C * (t1 + t2)
    expected = 0.5 * density * reverb.k_cell_area(t1, t2, C, np.radians(30.0), 20.0)
    on = expected > 0
    assert base.shape[on] == pytest.approx(expected[on], rel=1e-12)
    assert np.all(np.isinf(base.shape[~on]))                                   # no boundary there: no texture
    far = mid > 25.0                                                           # beyond the boundary's depth (20 m)
    ratio = base.shape[far] / mid[far]
    assert ratio == pytest.approx(ratio[0], rel=1e-12)                         # proportional to the range
    i, j = np.flatnonzero(far)[[3, -1]]
    assert base.shape[j] / base.shape[i] == pytest.approx(mid[j] / mid[i], rel=1e-12)
    # Twice the azimuth width: twice the shape. Patches: twice the shape of scatterers.
    wide = chosen_texture(monkeypatch, {"model": "k", "scatterers_per_m2": density, "cell_azimuth_deg": 60.0},
                          n_azimuth=48)
    assert wide.shape[far] == pytest.approx(2.0 * base.shape[far], rel=1e-12)
    patches = chosen_texture(monkeypatch, {"model": "k", "patches_per_m2": density, "cell_azimuth_deg": 30.0},
                             n_azimuth=48)
    assert patches.shape[far] == pytest.approx(2.0 * base.shape[far], rel=1e-12)
    # Twice the bandwidth: half the cell's length (the default 1 / bandwidth), half the shape at a range.
    shapes = {}
    for band in (500.0, 1000.0):
        wf = {"type": "lfm", "f_start": 27.5e3 - band / 2, "f_stop": 27.5e3 + band / 2, "duration": 0.01}
        tx = chosen_texture(monkeypatch, {"model": "k", "scatterers_per_m2": density}, waveform=wf, n_azimuth=48)
        assert tx.cell / tx.rate == pytest.approx(1.0 / band, rel=0.02)
        t1 = tx.start_time + np.arange(len(tx.shape)) * tx.cell / tx.rate
        shapes[band] = np.interp(800.0, 0.25 * C * (2 * t1 + tx.cell / tx.rate), tx.shape)
        shapes[band] /= tx.cell / tx.rate * band                               # the cell as realised in samples
    assert shapes[1000.0] == pytest.approx(0.5 * shapes[500.0], rel=1e-6)


def test_the_default_azimuth_cell_is_the_receive_beamwidth(monkeypatch):
    """Without cell_azimuth_deg the cell is one nominal beamwidth of the receive array, wavelength /
    array extent at the centre frequency, rounded to whole sectors of the generator; the area uses
    the width as realised (the sectors that share one texture)."""
    tx = chosen_texture(monkeypatch, {"model": "k", "scatterers_per_m2": 0.5}, n_azimuth=360)
    rec = empty_ping()
    p = rec.rx_positions()
    width = (C / 27.5e3) / np.linalg.norm(p.max(axis=0) - p.min(axis=0))
    sectors = max(int(round(width / (2 * np.pi / 360))), 1)
    t1 = tx.start_time + np.arange(len(tx.shape)) * tx.cell / tx.rate
    expected = 0.25 * reverb.k_cell_area(t1, t1 + tx.cell / tx.rate, C, sectors * 2 * np.pi / 360, 20.0)
    on = expected > 0
    assert tx.shape[on] == pytest.approx(expected[on], rel=1e-12)
    assert np.array_equal(tx.group_of_row, np.arange(360) // sectors)


def test_density_statistics_in_a_ping():
    """Through add_reverberation, monostatic, one azimuth cell (the whole circle), cells of 20 ms
    (twenty pulse lengths): the shape grows from about 1 at 200 m to 7 at 1400 m. Per range block,
    the intensity's normalised second moment follows 2 (1 + sum_j w_j^2 / nu_j), the K value
    2 (1 + 1 / nu(r)) with the pulse's straddling of two cells taken in (w_j: the share of the
    pulse in cell j); the mean level is the Rayleigh run's. 150 pings (Rayleigh: 40); the tolerance is four
    standard errors of the moment, from its spread over the pings (texture cells are independent
    between pings), and the errors are asserted small enough for the test to mean something."""
    density = 1.0e-4
    stats = {"model": "k", "scatterers_per_m2": density, "cell_time": 0.02}
    wf = {"duration": 1e-3}
    blocks = [(150.0, 400.0), (400.0, 700.0), (700.0, 1050.0), (1050.0, 1450.0)]
    moment, error, level = {}, {}, {}
    for label, statistics, pings in (("k", stats, 150), ("rayleigh", None, 40)):
        b1, b2 = np.zeros((pings, len(blocks))), np.zeros((pings, len(blocks)))
        for seed in range(pings):
            rec = empty_ping(ranges=(100.0, 1500.0), waveform=wf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)          # the first cells hold under one scatterer
                power = reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(seed),
                                                 n_azimuth=1, statistics=statistics, band_taper=True)
            x = np.abs(rec.samples.array[0]) ** 2 / power
            s = rec.samples
            r = 0.5 * C * (s.start_time + np.arange(s.n_samples) / s.sample_rate)
            for k, (a, b) in enumerate(blocks):
                b1[seed, k], b2[seed, k] = x[(r >= a) & (r < b)].mean(), (x[(r >= a) & (r < b)] ** 2).mean()
        level[label] = b1.mean(axis=0)
        moment[label] = b2.mean(axis=0) / level[label] ** 2
        error[label] = b2.std(axis=0, ddof=1) / np.sqrt(pings) / level[label] ** 2
    # The expected moment, from the shapes of the cells (one cell: the whole circle) and a 1 ms pulse.
    t_cell = 0.02
    t_sample = s.start_time + np.arange(s.n_samples) / s.sample_rate              # two-way time of a sample's end
    expected = np.zeros(s.n_samples)
    t0 = s.start_time - 1e-3                                                     # the generator's first time
    for k in range(int((t_sample[-1] - t0) / t_cell) + 2):
        a, b = t0 + k * t_cell, t0 + (k + 1) * t_cell
        w = np.clip(np.minimum(t_sample, b) - np.maximum(t_sample - 1e-3, a), 0.0, None) / 1e-3
        nu = 0.5 * density * reverb.k_cell_area(a, b, C, 2 * np.pi, 20.0)
        if nu > 0:
            expected += w**2 / nu
    expected = 2.0 * (1.0 + expected)
    for k, (a, b) in enumerate(blocks):
        want = expected[(r >= a) & (r < b)].mean()
        assert want - 2.0 > 4.0 * error["k"][k]                                   # resolved from Rayleigh's 2
        assert abs(moment["k"][k] - want) < 4.0 * error["k"][k], (k, moment["k"], want, error["k"])
        assert moment["rayleigh"][k] == pytest.approx(2.0, rel=0.05)
        assert level["k"][k] == pytest.approx(level["rayleigh"][k], rel=0.08)
    assert moment["k"][0] > moment["k"][-1] + 0.6                               # heavier tailed near, Rayleigh-like far


@pytest.mark.parametrize("geometry", ["monostatic", "bistatic"])
@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
def test_density_statistics_are_the_same_on_both_engines(geometry):
    """The texture is drawn in Python (CellTexture) and multiplies the deviates handed to the C++
    generator (host or CUDA backend), so both engines and both generators see one texture."""
    out = []
    for engine in ("cpp", "numpy"):
        rec = empty_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(4), n_azimuth=24,
                                 geometry=geometry, engine=engine,
                                 statistics={"model": "k", "scatterers_per_m2": 0.2, "cell_azimuth_deg": 30.0})
        out.append(rec.samples.array)
    assert np.abs(out[0] - out[1]).max() < 5e-6 * np.abs(out[1]).max()


@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
def test_a_texture_varying_from_cell_to_cell_is_the_same_on_both_engines():
    n, fs = 4096, 1000.0
    coarse, density = flat(6, level=3.0)
    shapes = np.linspace(0.2, 40.0, -(-n // 16))
    shapes[:3] = np.inf                                                          # no boundary there
    out = []
    for engine in ("numpy", "cpp"):
        rng = np.random.default_rng(5)
        texture = reverb.CellTexture(rng, shapes, n, 16, np.arange(6) // 2)
        out.append((generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=6, texture=texture,
                             offsets=[[0, 0, 0], [0.05, 0, 0], [0.1, 0.02, 0]]), rng.bit_generator.state))
    assert np.abs(out[0][0] - out[1][0]).max() < 5e-6 * np.abs(out[0][0]).max()
    assert out[0][1] == out[1][1]


def test_texture_per_cell_shape_large_small_and_none():
    """A shape per cell: unit mean and variance 1 / shape in each cell; shapes in the thousands go
    smoothly to 1 (Rayleigh); shapes far below 1 still give unit mean (gamma draws, no overflow);
    an infinite shape (no boundary in the cell) is a texture of exactly 1."""
    cells, per = 4, 200_000
    shapes = np.array([2e-2, 0.7, 3e3, np.inf])
    texture = reverb.CellTexture(np.random.default_rng(6), np.repeat(shapes, per), cells * per, 1, [0])
    tau = (texture(0) ** 2).reshape(cells, per)
    assert np.all(np.isfinite(tau)) and np.all(tau >= 0.0)
    assert tau[1].mean() == pytest.approx(1.0, rel=0.01) and tau[1].var() == pytest.approx(1 / 0.7, rel=0.03)
    assert tau[2].mean() == pytest.approx(1.0, abs=1e-4) and tau[2].var() == pytest.approx(1 / 3e3, rel=0.02)
    assert np.all(tau[3] == 1.0)
    # 0.02: variance 50, so a sample mean of 2e5 draws is within 0.05 of 1 (3 sigma: 0.047).
    assert tau[0].mean() == pytest.approx(1.0, abs=0.05)
    huge = reverb.CellTexture(np.random.default_rng(6), np.full(1000, 1e12), 1000, 1, [0])(0) ** 2
    assert np.abs(huge - 1.0).max() < 1e-5
    with pytest.raises(ValueError, match="positive"):
        reverb.CellTexture(np.random.default_rng(6), np.array([1.0, 0.0]), 2, 1, [0])
    with pytest.raises(ValueError, match="cells"):
        reverb.CellTexture(np.random.default_rng(6), np.array([1.0, 2.0]), 3, 1, [0])


def test_the_scalar_shape_texture_is_unchanged():
    """Pin of the fixed-shape path (sonar-6fc must not move it): the texture is the square root of
    one gamma draw of unit mean per cell, from a generator seeded by one draw of the caller's, as
    before; and the reverberation with only `shape` given is bit for bit that texture's."""
    rng = np.random.default_rng(12)
    row = reverb.CellTexture(rng, 1.7, 1000, 7, [0, 0, 1])(0)
    own = np.random.default_rng(int(np.random.default_rng(12).integers(0, 2**63 - 1)))
    assert np.array_equal(row, np.sqrt(np.repeat(own.gamma(1.7, 1.0 / 1.7, -(-1000 // 7)), 7)[:1000]))


def test_few_scatterers_per_cell_warns(monkeypatch):
    """Fewer than one scatterer (or patch) expected in a cell with reverberation: the paper's model
    (a fixed number n of scatterers per cell) no longer describes it, though the K distribution
    itself is defined for any positive shape: a warning, and the reverberation is still made."""
    with pytest.warns(RuntimeWarning, match="fewer than one"):
        chosen_texture(monkeypatch, {"model": "k", "scatterers_per_m2": 1e-6}, n_azimuth=48)


def test_the_height_in_the_cell_area(monkeypatch):
    """Flat boundary (and an exactly level bathymetry, which runs the flat code): the height of the
    array centre above it clips the first cells; a seabed with relief: no height (the slant-range form)."""
    stats = {"model": "k", "scatterers_per_m2": 0.5, "cell_azimuth_deg": 15.0}
    expected = {}
    for label, kw in (("flat", {}), ("level", {"bathymetry": {"plane": {"depth": 0.0}}}),
                      ("relief", {"bathymetry": {"plane": {"depth": 0.0, "slope": [0.05, 0.0]}}})):
        tx = chosen_texture(monkeypatch, stats, ranges=(10.0, 120.0), n_azimuth=24, geometry="bistatic", **kw)
        t1 = tx.start_time + np.arange(len(tx.shape)) * tx.cell / tx.rate
        area = reverb.k_cell_area(t1, t1 + tx.cell / tx.rate, C, np.radians(15.0), 0.0 if label == "relief" else 20.0)
        expected[label] = np.where(area > 0, 0.25 * area, np.inf)
        assert tx.shape == pytest.approx(expected[label], rel=1e-12)
    assert np.isinf(expected["flat"]).sum() > np.isinf(expected["relief"]).sum()


def test_a_scenario_item_takes_the_density(monkeypatch):
    """simulate_ping's `reverberation` item hands `statistics` with a density to the generator as it is."""
    monkeypatch.setattr(reverb, "CellTexture", Recorded)
    Recorded.made.clear()
    sonar = SonarModel.load("popoto_8", {
        "receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": [60.0, 200.0]}, "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3,
                                                          "duration": 2e-3, "window_alpha": 0.0,
                                                          "source_level_db_rms": None, "amplitude": 1.0}}})
    simulate_ping(sonar, make_pose(position=(0.0, 0.0, -20.0)), Scene(), environment={"sound_speed": C}, adc=False,
                  reverberation=[{"boundary_z": 0.0, "mu_db": -27.0,
                                  "statistics": {"model": "k", "patches_per_m2": 0.5, "cell_azimuth_deg": 10.0}}],
                  absorption_model="none", rng=np.random.default_rng(5))
    (texture,) = Recorded.made
    assert np.ndim(texture.shape) == 1 and np.nanmax(texture.shape[np.isfinite(texture.shape)]) > 0


def test_density_statistics_options():
    def run(statistics, **kw):
        rec = empty_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(1), n_azimuth=12,
                                 statistics=statistics, **kw)
        return rec.samples.array

    assert not np.array_equal(run({"model": "k", "scatterers_per_m2": 0.2}), run(None))
    with pytest.raises(ValueError, match=r"statistics\.scatterers_per_m: unknown key; did you mean 'scatterers_per_m2'\?"):
        run({"model": "k", "scatterers_per_m": 0.01})
    with pytest.raises(ValueError, match=r"statistics\.cell_azimuth: unknown key; did you mean 'cell_azimuth_deg'\?"):
        run({"model": "k", "shape": 2.0, "cell_azimuth": 3.0})
    with pytest.raises(ValueError, match=r"statistics\.zzzzzzzz: unknown key \(known: "):
        run({"model": "k", "shape": 2.0, "zzzzzzzz": 3.0})
    for two in ({"shape": 2.0, "scatterers_per_m2": 0.01}, {"scatterers_per_m2": 0.01, "patches_per_m2": 0.01}):
        with pytest.raises(ValueError, match="only one of"):
            run(dict(model="k", **two))
    with pytest.raises(ValueError, match="shape"):                         # none of the three: names them all
        run({"model": "k"})
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive and finite"):
            run({"model": "k", "patches_per_m2": bad})
    with pytest.raises(ValueError, match="model: k"):
        run({"model": "rayleigh", "scatterers_per_m2": 0.01})


# --- spread against coherence (TR 0702 Sec. 10.6) ----------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_time_spread_gives_the_coherence_bandwidth(engine):
    """Eq. (89): scattering strength Gaussian in two-way time with RMS width sigma_T gives a
    frequency coherence exp(-df^2 / (2 sigma_f^2)) with sigma_f = 1 / (2 pi sigma_T)."""
    fs, n, sigma_t = 1000.0, 1024, 0.02
    coarse = np.arange(0.0, n / fs, 1e-3)
    density = np.exp(-0.5 * ((coarse - 0.5) / sigma_t) ** 2)[None, :]
    rng = np.random.default_rng(1)
    spectra = np.array([np.fft.fft(generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=1)[0])
                        for _ in range(200)])
    lags = np.arange(0, 21)
    coherence = np.array([np.abs(np.mean(spectra * np.conj(np.roll(spectra, -k, axis=1)))) for k in lags])
    coherence /= coherence[0]
    sigma_f = 1.0 / (2.0 * np.pi * sigma_t)                        # 7.96 Hz
    expected = np.exp(-0.5 * (lags * fs / n / sigma_f) ** 2)
    assert np.abs(coherence - expected).max() < 0.03, coherence - expected
    # The width itself, from the lag where the coherence falls to exp(-1/2).
    width = np.interp(np.exp(-0.5), coherence[::-1], (lags * fs / n)[::-1])
    assert width == pytest.approx(sigma_f, rel=0.05)


@pytest.mark.parametrize("engine", ENGINES)
def test_doppler_spread_gives_the_coherence_time(engine):
    """Eq. (96): scattering strength Gaussian in Doppler with RMS width sigma_gamma gives,
    for a tone of frequency f, a time coherence of RMS width 1 / (2 pi f sigma_gamma).

    Here the Doppler spread is the generator's own: a sonar moving at v along x and
    reverberation arriving in a Gaussian fan of azimuths (RMS sigma_phi) about broadside,
    so gamma = 1 + (2 v / c) cos(phi) has sigma_gamma = 2 v sigma_phi / c. The tone has a
    finite length T, which multiplies the coherence by its own autocorrelation 1 - |s| / T."""
    fs, n, fc, v, sigma_phi, n_az, tone = 400.0, 4096, 10e3, 3.0, 0.1, 360, 0.5
    phi = (np.arange(n_az) + 0.5) * 2 * np.pi / n_az
    weight = np.exp(-0.5 * ((phi - 0.5 * np.pi) / sigma_phi) ** 2)
    weight[weight < 1e-8] = 0.0
    coarse = np.array([-1.0, 1e6])
    density = weight[:, None] * np.ones((1, 2))
    pulse = np.ones(int(tone * fs))
    lags = np.arange(0, 41)
    rng = np.random.default_rng(3)
    cov = 0.0
    for _ in range(12):
        x = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=n_az, fc=fc, pulse=pulse,
                     velocity=(v, 0.0, 0.0))[0]
        cov = cov + np.array([np.mean(x * np.conj(np.roll(x, -k))) for k in lags])
    coherence = cov / cov[0].real
    sigma_gamma = 2.0 * v * sigma_phi / C
    sigma_time = 1.0 / (2.0 * np.pi * fc * sigma_gamma)            # 39.8 ms
    s = lags / fs
    expected = np.exp(-0.5 * (s / sigma_time) ** 2) * (1.0 - s / tone)
    assert np.abs(coherence - expected).max() < 0.06, np.abs(coherence - expected).max()
    measured = np.interp(np.exp(-0.5), (np.abs(coherence) / (1.0 - s / tone))[::-1], s[::-1])
    assert measured == pytest.approx(sigma_time, rel=0.1)


@pytest.mark.parametrize("engine", ENGINES)
def test_direction_spread_gives_the_coherence_length(engine):
    """Eq. (103): arrivals Gaussian in sin(theta) about broadside with RMS width
    sigma_theta give a coherence between hydrophones d apart of exp(-d^2 / (2 sigma_d^2)),
    sigma_d = lambda / (2 pi sigma_theta)."""
    fs, n, fc, n_az, sigma_theta = 1000.0, 4096, 10e3, 720, 0.08
    phi = (np.arange(n_az) + 0.5) * 2 * np.pi / n_az
    weight = np.where(np.sin(phi) > 0.0, np.exp(-0.5 * (np.cos(phi) / sigma_theta) ** 2), 0.0)   # sin(theta) = cos(phi)
    weight[weight < 1e-8] = 0.0
    coarse = np.array([-1.0, 1e6])
    density = weight[:, None] * np.ones((1, 2))
    spacing, channels = 0.05, 12
    offsets = np.column_stack([np.arange(channels) * spacing, np.zeros(channels), np.zeros(channels)])
    rng = np.random.default_rng(4)
    num, den = np.zeros(channels, dtype=complex), 0.0
    for _ in range(4):
        x = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=n_az, fc=fc, offsets=offsets)
        den += np.mean(np.abs(x) ** 2)
        for sep in range(channels):
            num[sep] += np.mean(x[: channels - sep] * np.conj(x[sep:]))
    coherence = num / den
    d = np.arange(channels) * spacing
    sigma_d = (C / fc) / (2.0 * np.pi * sigma_theta)               # 0.298 m
    expected = np.exp(-0.5 * (d / sigma_d) ** 2)
    assert np.abs(coherence - expected).max() < 0.05, np.abs(coherence - expected).max()
    measured = np.interp(np.exp(-0.5), np.abs(coherence)[::-1], d[::-1])
    assert measured == pytest.approx(sigma_d, rel=0.08)


# --- replica correlation -----------------------------------------------------------------------

def test_replica_correlation_gains_3_db_per_octave_of_bandwidth():
    """Pulses of one length and energy, bandwidth doubling: the matched filter's peak on an
    echo is the pulse energy whatever the bandwidth, and its output on reverberation falls
    by half per octave (the reverberation is white over the band, the filter passes a
    resolution cell of 1 / B)."""
    fc, duration = 27.5e3, 0.02
    out_power, energy = {}, {}
    for band in (500.0, 1000.0, 2000.0, 4000.0):
        wf = {"type": "lfm", "f_start": fc - 0.5 * band, "f_stop": fc + 0.5 * band, "duration": duration}
        total = 0.0
        for seed in range(4):
            rec = empty_ping(ranges=(60.0, 400.0), waveform=wf)
            reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(seed))
            s = rec.samples
            replica = reverb.baseband_pulse(rec.transmit.waveform, s.sample_rate, s.center_frequency)
            compressed = np.array([np.convolve(row, np.conj(replica[::-1]), mode="valid") for row in s.array])
            total += np.mean(np.abs(compressed) ** 2)
        out_power[band] = total / 4
        energy[band] = float(np.sum(np.abs(replica) ** 2))
    bands = sorted(out_power)
    assert max(energy.values()) / min(energy.values()) == pytest.approx(1.0, abs=1e-9)
    # Signal-to-reverberation ratio after the filter, relative to the narrowest pulse.
    gain = np.array([10 * np.log10(out_power[bands[0]] / out_power[b]) for b in bands])
    steps = np.diff(gain)
    assert np.abs(steps - 10 * np.log10(2.0)).max() < 0.5, steps
    assert gain[-1] == pytest.approx(30 * np.log10(2.0), abs=0.6)   # three octaves: 9 dB


# --- leakage along a long record ---------------------------------------------------------------

def test_band_taper_keeps_the_far_end_of_a_long_record_at_its_own_level():
    """Over 30 m to 3 km the reverberation falls by 100 dB. With `band_taper` the generated
    level follows the expected power to the end of the record. (Without it the ideal delays
    and interpolation of the generator leak the strong start into the far end: measured
    7 dB too high over 50-75 % of this record and 25 dB over 90-97 %, on both engines.)"""
    excess = {}
    for taper in (True, False):
        ratio = 0.0
        for seed in range(4):
            rec = empty_ping(ranges=(30.0, 3000.0))
            power = reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(seed),
                                             band_taper=taper)
            ratio = ratio + (np.abs(rec.samples.array) ** 2).mean(axis=0) / power / 4
        n = len(ratio)
        assert 10 * np.log10(power[0] / power[-1]) > 95.0
        reference = np.median(ratio[n // 8: n // 4])
        excess[taper] = [10 * np.log10(ratio[int(n * a): int(n * b)].mean() / reference)
                         for a, b in ((0.5, 0.75), (0.75, 0.9), (0.9, 0.97), (0.97, 1.0))]
    assert np.abs(excess[True]).max() < 0.5, excess[True]
    assert min(excess[False]) > 3.0, excess[False]                  # the limit of the default, kept for old results


def test_band_taper_leaves_the_band_alone():
    """The taper touches the pulse's spectrum only above half the reduced rate's Nyquist
    frequency; the rate is at least four times the pulse's band, so the band is inside."""
    pulse = np.exp(2j * np.pi * 0.02 * np.arange(64) ** 2 / 64) * np.hanning(64)      # an LFM within +-0.1 of the rate
    out = reverb.taper_pulse(pulse, 1024)
    f = np.abs(np.fft.fftfreq(1024))
    before, after = np.fft.fft(pulse, 1024), np.fft.fft(out)
    assert np.abs(after - before)[f <= 0.25].max() < 1e-12
    assert np.abs(after[f > 0.499]).max() < 1e-4 * np.abs(before).max()
    assert np.sum(np.abs(out) ** 2) == pytest.approx(np.sum(np.abs(pulse) ** 2), rel=1e-3)


@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
def test_band_taper_is_the_same_on_both_engines():
    out = []
    for engine in ("cpp", "numpy"):
        sonar = SonarModel.load("popoto_8", {
            "receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": [25.0, 230.0]},
                        "sensitivity_db": None},
            "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3,
                                                              "duration": 2e-3, "window_alpha": 0.0,
                                                              "source_level_db_rms": None, "amplitude": 1.0}}})
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, -20.0), velocity=(3.0, -2.0, 0.0)), Scene(),
                            environment={"sound_speed": C}, adc=False)
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(2), n_azimuth=24,
                                 band_taper=True, engine=engine)
        out.append(rec.samples.array)
    assert np.abs(out[0] - out[1]).max() < 5e-6 * np.abs(out[1]).max()
