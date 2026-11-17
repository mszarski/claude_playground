#!/usr/bin/env python3
"""What time spreading does to a bounce-path echo: one target at 1 km in 100 m of isovelocity
water seen by the 32-hydrophone flank array, 60 pings 15 s apart, with the surface (a) flat,
(b) rough with the scattered part dropped (coherent loss only, what the simulator did before),
(c) rough with time spreading (sim/timespread.py). Prints the numbers quoted in docs/RESULTS.md.
Then the options beyond SST: per eigenray the closed-form Doppler width, angular spread and
coherence lengths; the tail of a two-bounce ray summed and convolved; and what at most 16 pings
show with angle spreading and with frequency spreading within the ping (slow: many more paths).

    tools/timespread_study.py [--wind 6] [--pings 60]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from gpusonar import recon  # noqa: E402
from gpusonar.sim import ImageSourceProvider, Scene, make_pose, simulate_ping  # noqa: E402
from gpusonar.sim.timespread import wrap  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

C = 1500.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wind", type=float, default=6.0)
    ap.add_argument("--pings", type=int, default=60)
    args = ap.parse_args(argv)
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [900.0, 1150.0]}}})
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=100.0, max_order=2, bottom=dict(speed=1600.0, density=1.8, attenuation=0.5))
    scene = Scene().add_points([[0.0, 1000.0, 50.0]], target_strength_db=10.0)
    pose = make_pose(position=(0.0, 0.0, 30.0))
    spread = {"surface": {"wind_speed": args.wind}, "taps": 8, "seed": 11}
    cases = {"flat surface": inner,
             "coherent loss only": wrap(inner, dict(spread, incoherent=False), frequency=sonar.center_frequency),
             "time spreading": wrap(inner, spread, frequency=sonar.center_frequency)}
    par = cases["time spreading"].surface
    print(f"wind {args.wind} m/s: rms height {par['rms_height']:.3f} m, mean-square slope {par['mean_square_slope']:.4f}, "
          f"rms vertical speed {par['rms_vertical_speed']:.2f} m/s; centre frequency {sonar.center_frequency:.0f} Hz")
    fs = None
    out = {}
    for name, provider in cases.items():
        env = []
        for i in range(args.pings):
            rec = simulate_ping(sonar, pose, scene, provider=provider, environment={"sound_speed": C}, adc=False, noise_rms=0.0,
                                scenario_time=15.0 * i, sequence=i)
            recon.prepare([rec])
            s = rec.samples
            fs, t0 = s.sample_rate, s.start_time
            env.append(np.abs(s.array[16]) ** 2)                       # one hydrophone, after pulse compression
        out[name] = np.array(env)
    t = t0 + np.arange(out["flat surface"].shape[1]) / fs
    direct = 2.0 * np.hypot(1000.0, 20.0) / C
    res = 2.0 / abs(sonar.waveform_config["f_stop"] - sonar.waveform_config["f_start"])
    first = np.abs(t - direct) < res                                   # the direct echo: no bounce
    rest = t > direct + res                                            # every arrival with at least one bounce
    flat = out["flat surface"]
    print(f"{'case':20s} {'direct echo':>12s} {'bounce energy':>14s} {'strongest bounce peak':>22s} {'its scintillation index':>24s} {'energy later than 5 ms after it':>32s}")
    j = int(np.argmax(np.where(rest, flat.mean(axis=0), 0.0)))
    for name, e in out.items():
        peak = e[:, j - 2:j + 3].max(axis=1)
        late = e[:, (t > t[j] + 0.005) & (t < t[j] + 0.012)].sum(axis=1).mean() / max(flat[:, (t > t[j] + 0.005) & (t < t[j] + 0.012)].sum(axis=1).mean(), 1e-300)
        print(f"{name:20s} {10 * np.log10(e[:, first].sum(axis=1).mean() / flat[:, first].sum(axis=1).mean()):+10.2f} dB "
              f"{10 * np.log10(e[:, rest].sum(axis=1).mean() / flat[:, rest].sum(axis=1).mean()):+11.2f} dB "
              f"{10 * np.log10(peak.mean() / flat[:, j - 2:j + 3].max(axis=1).mean()):+19.2f} dB "
              f"{peak.var() / peak.mean() ** 2:24.2f} {10 * np.log10(late):+29.2f} dB")
    beyond_sst(args, sonar, inner, scene, pose, spread, j, flat)
    return 0


def beyond_sst(args, sonar, inner, scene, pose, spread, j, flat):
    """The options beyond SST (sim/timespread.py: angle and frequency spreading, the facet decay
    time, convolved bounces): the closed forms per eigenray, then what pings show."""
    from gpusonar.sim.timespread import TimeSpreadProvider
    f = sonar.center_frequency
    lam = C / f
    tx = np.asarray(pose.position, dtype=float)
    base = inner.eigenrays(tx, scene.positions)
    probe = wrap(inner, dict(spread, doppler=True, doppler_lines=2), frequency=f)
    par = probe.surface
    frac, is_surface = probe._bounce_fractions(tx[2], scene.positions[:, 2][:, None], base.launch[..., 2] < 0.0, base.n_surface, base.n_bottom)
    print("\nPer eigenray with a surface bounce (one way; closed forms of sim/timespread.py). T: mean excess delay of the tail;"
          "\nDoppler: rms width of the tail's spectrum and the coherence time 1 / (2 pi width) (TR 0702 eq. 96); angles: rms"
          "\nspread of the tail's direction at the sonar, and the coherence lengths lambda / (2 pi sigma) (eq. 103) along a"
          "\nvertical line and a horizontal line across the path.")
    print(f"{'surface,bottom bounces':>22s} {'grazing':>8s} {'coherent':>9s} {'T rule':>9s} {'T facet':>9s} {'Doppler':>9s} {'coh. time':>10s} "
          f"{'in-plane':>9s} {'across':>8s} {'L vert.':>8s} {'L horiz.':>9s}")
    for r in range(base.delay.shape[1]):
        if base.n_surface[0, r] == 0:
            continue
        g = np.arcsin(abs(base.launch[0, r, 2]))
        gamma2 = (4.0 * np.pi * par["rms_height"] * np.sin(g) / lam) ** 2
        t_rule = t_facet = v2 = 0.0
        first = None
        for i in range(frac.shape[-1]):
            if not (np.isfinite(frac[0, r, i]) and is_surface[0, r, i]):
                continue
            fr = frac[0, r, i]
            first = fr if first is None else first
            lever = fr * (1.0 - fr) * base.delay[0, r]
            t_rule += 2.0 * par["mean_square_slope"] * lever
            t_facet += par["mean_square_slope"] * (1.0 + np.sin(g) ** 2) * lever
            v2 += par["rms_vertical_speed"] ** 2 * np.sin(g) ** 2
        n_s = int(base.n_surface[0, r])
        width = 2.0 * f * np.sqrt(v2) / C / np.sqrt(max(1.0 - np.exp(-gamma2 * n_s), 1e-12))
        s_v = np.sqrt(2.0 * par["mean_square_slope"]) * (1.0 - first)      # the first rough bounce sets the direction at the sonar
        s_h = s_v * np.sin(g)
        print(f"{n_s:>11d},{int(base.n_bottom[0, r]):<10d} {np.degrees(g):6.2f} d {np.exp(-gamma2 * n_s):9.2e} {1e3 * t_rule:6.2f} ms {1e3 * t_facet:6.2f} ms "
              f"{width:6.2f} Hz {1.0 / (2.0 * np.pi * width):8.3f} s {np.degrees(s_v):7.2f} d {np.degrees(s_h):6.2f} d "
              f"{lam / (2.0 * np.pi * s_v * np.cos(g)):6.2f} m {lam / (2.0 * np.pi * s_h):7.2f} m")
    print("(TR 0702 p. 98 quotes a surface Doppler spread 'less than 10 Hz' for ASIAEX at 8 kHz in low wind: a bound, not a calibration.)")

    # Several bounces: the tail of the ray with two surface bounces, one exponential against the convolution.
    deeper = ImageSourceProvider(C, surface_z=0.0, bottom_z=100.0, max_order=3, bottom=dict(speed=1600.0, density=1.8, attenuation=0.5))
    base3 = deeper.eigenrays(tx, scene.positions)
    two = np.flatnonzero(base3.n_surface[0] == 2)
    if len(two):
        r, k, taps = int(two[0]), base3.delay.shape[1], 24
        g = np.arcsin(abs(base3.launch[0, r, 2]))
        print(f"\nTail of the ray with two surface bounces and one bottom bounce (grazing {np.degrees(g):.1f} d, coherent fraction per surface bounce "
              f"{np.exp(-(4.0 * np.pi * par['rms_height'] * np.sin(g) / lam) ** 2):.1e}; {taps} taps):")
        for mode in ("sum", "convolve"):
            p = TimeSpreadProvider(deeper, frequency=f, surface=par, surface_z=0.0, bottom_z=100.0, taps=taps, doppler=True, doppler_lines=1, multi_bounce=mode)
            e = p.eigenrays(tx, scene.positions)
            w = abs(e.amplitude[0, k:].reshape(k, taps)[r]) ** 2
            tau = e.delay[0, k:].reshape(k, taps)[r] - base3.delay[0, r]
            mean = np.sum(w * tau) / np.sum(w)
            thirds = [w[(tau > a * mean) & (tau <= (a + 1.0) * mean)].sum() / w.sum() for a in (0.0, 1.0, 2.0)]
            print(f"  multi_bounce: {mode:9s} mean excess delay {1e3 * mean:6.2f} ms, energy in (0,1], (1,2], (2,3] means: " + ", ".join(f"{x:.2f}" for x in thirds)
                  + f", rms spread {1e3 * np.sqrt(np.sum(w * tau**2) / np.sum(w) - mean**2):.2f} ms")

    # Pings. Fewer taps than above: with `doppler` each tap is `doppler_lines` rays.
    n_pings = min(args.pings, 16)
    small = dict(spread, taps=3)
    cases = {"time spreading": small,
             "+ angle_spread": dict(small, angle_spread=True, decay_time="facet"),
             "+ doppler (gaussian)": dict(small, doppler=True, doppler_spectrum="gaussian", doppler_lines=3),
             "+ both": dict(small, angle_spread=True, decay_time="facet", doppler=True, doppler_spectrum="gaussian", doppler_lines=3)}
    print(f"\n{n_pings} pings, 3 taps (3 lines each with doppler), at the strongest bounce arrival after pulse compression (a {sonar.waveform_config['duration']:g} s pulse): mean peak"
          "\nlevel re the flat surface, and the coherence over pings between hydrophones at the two ends and between neighbours of the"
          f"\n{sonar.config['array']['count']}-hydrophone horizontal line (phases of the flat-surface arrival removed).")
    print(f"{'case':22s} {'peak level':>11s} {'coherence end to end':>21s} {'neighbours':>11s} {'paths per ping':>15s}")
    ref = None
    for name, cfg in cases.items():
        provider = wrap(inner, cfg, frequency=f)
        peaks, n_paths = [], 0
        for i in range(n_pings):
            rec = simulate_ping(sonar, pose, scene, provider=provider, environment={"sound_speed": C}, adc=False, noise_rms=0.0,
                                scenario_time=15.0 * i, sequence=i)
            recon.prepare([rec])
            peaks.append(rec.samples.array[:, j])
        if ref is None:
            rec = simulate_ping(sonar, pose, scene, provider=inner, environment={"sound_speed": C}, adc=False, noise_rms=0.0)
            recon.prepare([rec])
            ref = rec.samples.array[:, j]
        e = provider.eigenrays(tx, scene.positions)
        n_paths = int(e.valid.sum()) ** 2
        y = np.array(peaks) * np.conj(ref) / np.abs(ref)                 # [ping, hydrophone]
        coh = lambda a, b: abs(np.mean(a * np.conj(b))) / np.sqrt(np.mean(abs(a) ** 2) * np.mean(abs(b) ** 2))   # noqa: E731
        near = np.mean([coh(y[:, m], y[:, m + 1]) for m in range(y.shape[1] - 1)])
        print(f"{name:22s} {10 * np.log10(np.mean(abs(y) ** 2) / np.mean(abs(ref) ** 2)):+8.2f} dB {coh(y[:, 0], y[:, -1]):21.2f} {near:11.2f} {n_paths:15d}")


if __name__ == "__main__":
    sys.exit(main())
