"""Scenario runner: turns a scenario config into ping records plus ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .. import _gpusonar as core
from .. import acoustics
from .. import environment as site
from ..config import dump_yaml, load_scenario_config
from ..sonar import SonarModel, resolve_environment
from . import receiver
from .propagation import EigenrayProvider, make_provider, two_way_paths
from .scene import Scene, build_scene


def make_pose(position=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
              rates=(0.0, 0.0, 0.0)) -> "core.Pose":
    """A core.Pose from plain sequences, in the record's frames and SI units (record.hpp):
    `position` NED, m (z down, from the frame origin); `rpy` roll, pitch, yaw, rad (ASASIN
    Eq. 2 conventions); `velocity` NED, m/s; `rates` d(roll, pitch, yaw)/dt, rad/s. Each is
    three numbers, converted to float; nothing else is checked."""
    pose = core.Pose()
    pose.position = tuple(float(x) for x in position)
    pose.rpy = tuple(float(x) for x in rpy)
    pose.velocity = tuple(float(x) for x in velocity)
    pose.rates = tuple(float(x) for x in rates)
    return pose


def absorption_table(sonar: SonarModel, env: dict, model: str):
    """(f_min, f_max, table) covering the transmit band, for the synthesis kernel."""
    w = sonar.waveform_config
    f0, f1 = float(w["f_start"]), float(w.get("f_stop", w["f_start"]))
    lo, hi = min(f0, f1), max(f0, f1)
    if hi <= lo:
        hi = lo * 1.0001
    f = np.linspace(lo, hi, core.ABSORPTION_TABLE_SIZE)
    table = acoustics.absorption(f, *site.water(env), model)
    return lo, hi, [float(x) for x in np.broadcast_to(table, f.shape)]


def simulate_ping(sonar: SonarModel, pose: "core.Pose", scene: Scene, *,
                  environment: dict | None = None, provider: EigenrayProvider | None = None,
                  sequence: int = 0, sub_sequence: int = 0, time_ns: int = 0,
                  scenario_time: float = 0.0, absorption_model: str = "francois_garrison",
                  noise_rms: float = 0.0, adc: bool = True, spreading: bool = True,
                  rng: np.random.Generator | None = None, tx_pose: "core.Pose | None" = None,
                  acceleration=(0.0, 0.0, 0.0), ins_rate: float = 0.0,
                  channel_errors: dict | None = None, clock_truth: dict | None = None,
                  noise_spectrum=None, noise_options: dict | None = None, reverberation: list | None = None,
                  beam_floor_db: float | None = None, noise_statistics: dict | None = None, fitted_noise: dict | None = None,
                  sources: list | None = None, transmit: bool = True,
                  backend=core.Backend.Host, engine: str | None = None,
                  front_end_truth=None) -> "core.PingRecord":
    """Simulate one transmit and return its record.

    `engine` ("cpp" or "numpy"; default: sim/engine.py) chooses the implementation of the
    path pairing, of reverberation and of echo synthesis for a sampled pulse (analytic pulses
    have the C++ kernel only); a provider has its own `engine`.

    `sources` (sim/sources.py `build_sources`): radiating sources heard one way, added to
    the same record. `transmit=False`: the sonar does not ping (a passive capture): no
    echoes, no reverberation, no transmit reference; the record keeps its form.

    Two keys of the sonar, both absent by default (the record is then what it always was):
    `transmit.direct_path` adds the pulse as the array hears it straight from the projector
    (sim/sources.py `direct_arrival`); `receive.overload` passes the samples through a
    front end that clips and recovers (sim/receiver.py `apply_front_end`), over the time from
    the transmit to the opening of the window as well (direct arrival only, at full fidelity;
    no echoes, reverberation or noise there), so that the window starts in the state the
    transmit left. `front_end_truth`: a dict, or a list that gets one dict per ping, filled
    with what the model did (per-channel overload figures; with the key "arrays" set true in
    a dict passed in, also the flags, gain and offset of every sample, and "direct", the
    direct arrival alone).
    """
    env = resolve_environment(environment, depth=pose.position[2])
    record = sonar.make_record(pose, sequence=sequence, sub_sequence=sub_sequence,
                               time_ns=time_ns, environment=env, tx_pose=tx_pose)
    c = record.environment.sound_speed
    provider = provider or make_provider("free_field", c)
    if hasattr(provider, "set_ping"):
        # Time-spread bounce paths fade with time and with the sonar's own speed.
        v = pose.velocity
        provider.set_ping(scenario_time, float(np.hypot(v[0], v[1])))
    # The true board clock errors drive synthesis, then are cleared: the record a
    # processor receives does not know them (the sync stage estimates them).
    clocks = _set_board_clocks(record, sonar, truth=True, rng=rng)
    if clock_truth is not None:
        clock_truth.update(clocks)
    direct_cfg = overload_cfg = None
    if sonar.config["transmit"].get("direct_path") or sonar.config["receive"].get("overload") is not None:
        from . import sources as source_models
        direct_cfg = source_models.direct_path_config(sonar) if transmit else None
        overload_cfg = receiver.overload_config(sonar)
        receiver.report_gain_switch(record, overload_cfg)
    if channel_errors is not None and channel_errors.get("position") is not None:
        # True element position errors drive synthesis; the processor is not told.
        cal = record.calibration
        cal.position_offset = [tuple(p) for p in np.asarray(channel_errors["position"], dtype=float)]
        record.calibration = cal
    if transmit and len(scene) > 0:
        tx_position = np.array(record.tx_position())
        rx_reference = record.rx_positions().mean(axis=0)
        # Positions, amplitudes and velocities for this ping: aspect-dependent targets, glints
        # and seabed scattering models (Scene.modifiers) are worked out from where the sonar is.
        points, amplitudes, velocities = scene.evaluate(scenario_time, tx_position, rx_reference,
                                                        frequency=sonar.center_frequency, sound_speed=c)
        if scene.occluders:
            # Shadowing: a scatterer hidden from the projector or from the array returns nothing.
            seen = scene.visible(tx_position, points) & scene.visible(rx_reference, points)
            amplitudes = np.where(seen, amplitudes, 0.0)
        q, p, amp, vq, vp = two_way_paths(provider, tx_position, rx_reference, points, amplitudes,
                                          velocities, scene.lambert,
                                          scene.normals_at(scenario_time) if len(scene.normals) == len(scene) else None,
                                          engine=engine)
        if beam_floor_db is not None and len(amp):
            # Skip paths that leave and return far down both beam patterns: with narrow
            # beams most scatterers contribute nothing measurable to most pings.
            from . import reverb
            keep = np.ones(len(amp), dtype=bool)
            tx_gain = reverb.element_power_pattern(record.geometry.tx,
                                                   np.asarray(core.rotation_ned(*record.geometry.tx_pose.rpy)))
            rx_gain = reverb.element_power_pattern(record.geometry.rx[0],
                                                   np.asarray(core.rotation_ned(*record.geometry.rx_pose.rpy)))
            gain = np.ones(len(amp))
            for fn, origin, target in ((tx_gain, tx_position, q), (rx_gain, rx_reference, p)):
                if fn is not None:
                    d = target - origin
                    gain = gain * fn(d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12))
            keep = gain >= 10.0 ** (beam_floor_db / 10.0)
            q, p, amp = q[keep], p[keep], amp[keep]
            if vq is not None:
                vq, vp = vq[keep], vp[keep]
        if record.transmit.waveform.type == core.WaveformType.Arbitrary:
            # A sampled pulse: the same kernel reading the envelope table of sim/waveform.py
            # (C++, host or CUDA), or the NumPy reference when that engine is selected.
            from . import waveform as sampled
            sampled.synthesize(record, sonar.sampled_waveform, q, p, amp, vq, vp,
                               absorption=absorption_table(sonar, env, absorption_model),
                               spreading=spreading,
                               platform_acceleration=tuple(float(a) for a in acceleration),
                               backend=backend, engine=engine)
        else:
            core.synthesize_into(record, q, p, amp, vq, vp,
                                 absorption=absorption_table(sonar, env, absorption_model),
                                 spreading=spreading, backend=backend,
                                 platform_acceleration=tuple(float(a) for a in acceleration))
    from .. import taps
    taps.sim_tap(record, "echoes")
    if transmit and getattr(provider, "basin", None) is not None:      # `basin_reverberation` (sim/basin.py): the echoes' late tail
        provider.basin.after_echoes(record, sonar, provider, scenario_time, environment=env, absorption_model=absorption_model, backend=backend, engine=engine)
    for item in (reverberation or []) if transmit else []:
        # Boundary reverberation by the scattering-function method (sim/reverb.py).
        from . import reverb
        w = sonar.waveform_config
        fc = 0.5 * (float(w["f_start"]) + float(w.get("f_stop", w["f_start"])))
        alpha = float(acoustics.absorption(fc, *site.water(env), absorption_model))
        strength = None
        if item.get("model", "lambert") in ("chapman_harris", "kraus_surface"):
            # Empirical surface laws: {model: chapman_harris | kraus_surface, wind_speed[, cap:
            # bubble_saturation]}. Without `cap` the law is unchanged and a warning is given once
            # where it lies above the bubble layer's saturation level (sim/surface.py, D-87).
            from . import surface as surface_model
            strength = surface_model.empirical_surface_strength(item["model"], float(item["wind_speed"]), fc, item.get("cap"))
        elif item.get("model") == "searay_bottom":
            # Empirical bottom law by type (sim/surface.py): {model: searay_bottom, bottom_type: mud|sand|gravel|rock}.
            from . import surface as surface_model
            strength = surface_model.searay_bottom_strength(fc, item["bottom_type"])
        elif item.get("model") == "bubble_layer_surface":
            # {model: bubble_layer_surface, wind_speed, extinction: apl_1994 | hall_novarini}: the
            # bubble-layer and facet terms of the published high-frequency surface model (sim/surface.py).
            from . import surface as surface_model
            strength = surface_model.bubble_layer_surface_strength(float(item["wind_speed"]), fc, env["sound_speed"],
                                                                   item.get("extinction", "apl_1994"))
        elif item.get("model") == "fluid_sediment":
            # {model: fluid_sediment, boundary_z, rho, nu, delta, w2, gamma2, sigma2}: interface
            # roughness and sediment volume scattering (sim/bottom.py) in place of Lambert's law.
            # Optional: sediment (a name in configs/sediments.yaml; the item's own values win),
            # roughness_model (small_slope, the default since D-86 | perturbation | kirchhoff |
            # composite | combined),
            # small_scale_kh.
            from . import bottom as bottom_model
            keys = ("rho", "nu", "delta", "w2", "gamma2", "sigma2")
            if "sediment" in item:
                item = dict(bottom_model.sediment(item["sediment"], item.get("sediment_file"), supplied=item), **item)
            missing = [k for k in keys if k not in item]
            if missing:
                raise ValueError(f"fluid_sediment reverberation needs {missing}")
            extra = {k: (item[k] if k == "roughness_model" else float(item[k])) for k in ("roughness_model", "small_scale_kh") if k in item}
            if item.get("geometry", "monostatic") == "bistatic":
                # The model at the true bistatic azimuth of each cell (pi is backscatter).
                strength = lambda g_in, g_out, azimuth=np.pi, _p={**{k: float(item[k]) for k in keys}, **extra}: bottom_model.strength(  # noqa: E731
                    g_in, g_out, fc, env["sound_speed"], azimuth=azimuth, **_p)
            else:
                strength = bottom_model.reverberation_strength(fc, env["sound_speed"], **{k: float(item[k]) for k in keys}, **extra)
        elif item.get("model") == "volume":
            # {model: volume, layer: [top, bottom], sv_db}: a layer of scatterers in the water
            # (fish, plankton, bubbles), as a sheet at its centre depth.
            top, bottom = (float(v) for v in item["layer"])
            strength = reverb.volume_layer_strength(float(item["sv_db"]), abs(bottom - top))
            item = dict(item, boundary_z=0.5 * (top + bottom))
        elif item.get("model", "lambert") != "lambert":
            raise ValueError(f"unknown reverberation model {item['model']!r}")
        elif "sediment" in item and "mu_db" not in item:
            # {boundary_z, sediment: rock | pebbles}: Lambert's law with the measured parameter
            # of a table entry that is not a fluid sediment (configs/sediments.yaml, D-88).
            from . import bottom as bottom_model
            item = dict(item, mu_db=bottom_model.named_lambert_mu_db(item["sediment"], item.get("sediment_file")))
        elif "mu_db" not in item:
            # A Lambert boundary with nothing said about its strength: -29 dB, the low-frequency
            # deep-water value (bottom.DEFAULT_LAMBERT_MU_DB, D-89; -27 dB before); warns once at
            # 10 kHz and above.
            from . import bottom as bottom_model
            item = dict(item, mu_db=bottom_model.default_lambert_mu_db(fc))
        # `offset_db` moves the boundary's strength up or down whatever its model: what
        # calibration against a real log adjusts (gpusonar/simcal.py).
        offset = float(item.get("offset_db", 0.0))
        if strength is not None and offset != 0.0:
            base, scale = strength, 10.0 ** (offset / 10.0)
            if reverb.reverb_bistatic.takes_azimuth(base):
                strength = lambda g_in, g_out, azimuth=np.pi, _b=base, _s=scale: _s * _b(g_in, g_out, azimuth)  # noqa: E731
            else:
                strength = lambda g_in, g_out, _b=base, _s=scale: _s * _b(g_in, g_out)  # noqa: E731
        # `exclude_specular: true` (or {mean_square_slope, rms_height, reflection}), bistatic items:
        # take out of the strength the near-specular part that time spreading already carries
        # as the tail of each bounce eigenray (sim/reverb_specular.py; TR 0702 Sec. 12.4).
        if item.get("exclude_specular"):
            from . import reverb_specular
            part = reverb_specular.boundary_part(item, provider, fc, env["sound_speed"])
            if part is not None:
                if strength is None:
                    mu = 10.0 ** ((float(item["mu_db"]) + offset) / 10.0)        # a Lambert item: mu_db is set above
                    strength = lambda g_in, g_out: mu * np.sin(g_in) * np.sin(g_out)   # noqa: E731
                strength = reverb_specular.exclude(strength, part)
        # New keys, all optional (sim/reverb.py, sim/reverb_bistatic.py): `geometry: bistatic`
        # (the transmitter's and receiver's own positions and velocities, arrival elevation),
        # with it `n_elevation`, `azimuth_subsamples` and `strength_map: {range, offset_db,
        # azimuth_deg, origin}` (a boundary whose strength varies over it); `statistics:
        # {model: k, shape, cell_time, cell_azimuth_deg}` (K-distributed reverberation);
        # `band_taper` and `guard` (against leakage from the strong start of a long record
        # into its weak end).
        extra = {}
        if item.get("geometry") is not None:
            extra["geometry"] = str(item["geometry"])
        if item.get("n_elevation") is not None:
            extra["n_elevation"] = int(item["n_elevation"])
        if item.get("max_doppler_bins") is not None:
            extra["max_doppler_bins"] = int(item["max_doppler_bins"])
        if item.get("azimuth_subsamples") is not None:
            extra["azimuth_subsamples"] = int(item["azimuth_subsamples"])
        if item.get("strength_map") is not None:
            m = item["strength_map"]
            extra["strength_map"] = m if callable(m) else reverb.reverb_bistatic.table_strength_map(
                m["range"], m["offset_db"], m.get("azimuth_deg"), m.get("origin", (0.0, 0.0)))
        if item.get("statistics") is not None:
            extra["statistics"] = dict(item["statistics"])
        if item.get("guard") is not None:
            extra["guard"] = float(item["guard"])
        if item.get("band_taper") is not None:
            extra["band_taper"] = bool(item["band_taper"])
        # `bathymetry` (sim/bathymetry.py: a seabed with relief instead of boundary_z),
        # `shadowing: false`, `scatterer_motion: {velocity, rms_speed, rms_vertical_speed,
        # wind_speed}` (moving scatterers: TR 0702 eq. 72 and a Doppler spread).
        if item.get("bathymetry") is not None:
            extra["bathymetry"] = item["bathymetry"]
            item = dict(item, boundary_z=item.get("boundary_z", 0.0))
        if item.get("shadowing") is not None:
            extra["shadowing"] = bool(item["shadowing"])
        if item.get("bottom_bounces") is not None:
            extra["bottom_bounces"] = str(item["bottom_bounces"])
        if item.get("scatterer_motion") is not None:
            extra["scatterer_motion"] = dict(item["scatterer_motion"])
        reverb.add_reverberation(record, getattr(provider, "reverberation_provider", provider), float(item["boundary_z"]),
                                 float(item.get("mu_db", -29.0)) + offset,     # used for Lambert items only (set above); others pass `strength`
                                 rng or np.random.default_rng(), n_azimuth=item.get("n_azimuth"),
                                 strength=strength,
                                 absorption_db_per_m=alpha, max_range=item.get("max_range"), backend=backend,
                                 engine=engine, **extra)
    if sources:
        # One-way sound of radiating sources (sim/sources.py), with the true clocks and
        # element positions still in the record.
        from . import sources as source_models
        source_models.add_sources(record, sonar, sources, provider, environment=env, absorption_model=absorption_model,
                                  scenario_time=scenario_time, rng=rng, backend=backend, engine=engine,
                                  spreading=spreading, acceleration=acceleration)
    pre_window = None
    truth = None
    if front_end_truth is not None and (direct_cfg is not None or overload_cfg is not None):
        truth = front_end_truth if isinstance(front_end_truth, dict) else {}
        if truth is not front_end_truth:
            front_end_truth.append(truth)
    if direct_cfg is not None:
        # The projector heard directly: a one-way source at its own position, with the true
        # clocks and element positions still in the record.
        direct_kwargs = dict(environment=env, absorption_model=absorption_model, scenario_time=scenario_time,
                             backend=backend, engine=engine, spreading=spreading, acceleration=acceleration)
        direct = source_models.direct_arrival(record, sonar, provider, direct_cfg, **direct_kwargs)
        s = record.samples
        if direct is not None:
            kind = s.kind
            s.array = s.array + direct.astype(s.array.dtype)
            s.kind = kind
            record.samples = s
            taps.sim_tap(record, "direct_arrival", direct)
        if truth is not None and truth.get("arrays"):
            truth["direct"] = direct
        n_pre = int(np.ceil(s.start_time * s.sample_rate - 1e-9))
        if overload_cfg is not None and n_pre > 0:
            # From the transmit to the opening of the window, on the same sample clock.
            pre_window = source_models.direct_arrival(record, sonar, provider, direct_cfg, **direct_kwargs,
                                                      window=(s.start_time - n_pre / s.sample_rate, n_pre))
            if pre_window is None:
                pre_window = np.zeros((s.n_channels, n_pre))
            pre_window = receiver.apply_channel_factors(np.asarray(pre_window, dtype=float), channel_errors)
            ref = sonar.config["receive"].get("reference")
            if ref and ref.get("dedicated"):
                for channel in ref["channels"]:           # a reference input has no hydrophone on it
                    if channel in record.metadata.channel_element:
                        pre_window[record.metadata.channel_element.index(channel)] = 0.0
    taps.sim_tap(record, "reverberation")
    if transmit:
        receiver.add_reference_signal(record, sonar, clocks)
    _set_board_clocks(record, sonar, truth=False)
    if channel_errors is not None:
        receiver.apply_channel_errors(record, channel_errors)
    if ins_rate > 0.0:
        _add_true_ins(record, np.asarray(acceleration, dtype=float), ins_rate)
    taps.sim_tap(record, "reference_and_clocks")
    samples = record.samples.array
    if noise_rms > 0.0:
        samples = receiver.add_noise(samples, noise_rms, rng or np.random.default_rng())
    if noise_spectrum is not None:
        # Ambient noise with a spectrum (dB re 1 uPa^2/Hz against Hz) and the coherence of
        # an isotropic field across the array.
        from . import noise as noise_models
        rx_cfg = sonar.config["receive"]
        sens = float(rx_cfg.get("sensitivity_db") or 0.0) + float(rx_cfg.get("gain_db", 0.0))
        # `receiver.ambient_noise`: `coherence: isotropic` (default) | `surface` | `independent` between hydrophones,
        # `spectrum: shaped` (default: the configured spectrum across the record's band) | `flat` (its level at the
        # centre frequency). Real-passband and baseband records follow the same two choices (D-103).
        opts = noise_options or {}
        coherence = str(opts.get("coherence", noise_models.DEFAULT_COHERENCE))
        shape = str(opts.get("spectrum", "shaped"))
        if coherence not in noise_models.COHERENCE_MODELS or shape not in ("shaped", "flat"):
            raise ValueError(f"receiver.ambient_noise: coherence is one of {', '.join(noise_models.COHERENCE_MODELS)} and spectrum is shaped or flat "
                             f"(got {coherence!r}, {shape!r})")
        stats = noise_models.noise_statistics(noise_statistics)
        # White, sample-by-sample statistics (alpha-stable; K with no correlation time) keep their own generator: a
        # colouring or mixing filter would change the distribution they are asked for.
        white_statistics = stats is not None and (stats["model"] == "alpha_stable" or not stats["correlation_time"] > 0.0)
        if np.iscomplexobj(samples) and not white_statistics and not (coherence == "independent" and shape == "flat"):
            # A baseband record covers a band `sample_rate` wide about the carrier: the configured spectrum across
            # that band, and between hydrophones the coherence the real-passband path gives (sim/noise.py
            # `baseband_gaussian_noise`). Closely spaced hydrophones then share noise, as they do at sea.
            fc_bb = record.samples.center_frequency
            spectrum = noise_models.flat_spectrum(noise_spectrum, fc_bb) if shape == "flat" else noise_spectrum
            g = rng or np.random.default_rng()
            ambient = noise_models.baseband_gaussian_noise(samples.shape[1], record.samples.sample_rate, fc_bb, spectrum, g,
                                                           positions=record.rx_positions(), sound_speed=c, sensitivity_db=sens, coherence=coherence)
            if stats is not None:                       # K with a correlation time: the texture on the Gaussian field
                ambient = ambient * noise_models.gamma_texture(samples.shape[1], record.samples.sample_rate, stats["shape"], stats["correlation_time"],
                                                               g, samples.shape[0], stats["common"]).astype(np.float32)
            samples = samples + ambient.astype(samples.dtype)
        elif np.iscomplexobj(samples):
            # `coherence: independent, spectrum: flat` (what every baseband record had before D-103), and the white
            # statistics: white noise at the spectrum level of the centre frequency, independent between channels
            # (true of an isotropic field at half-wavelength spacing only). The complex envelope carries twice
            # the power of the real signal in that band.
            fs_bb = record.samples.sample_rate
            level = float(np.asarray(noise_spectrum(np.array([record.samples.center_frequency])))[0])
            sigma = np.sqrt(2.0 * 10.0 ** (level / 10.0) * fs_bb) * 10.0 ** (sens / 20.0)
            g = rng or np.random.default_rng()
            ambient = (sigma / np.sqrt(2.0)) * (
                g.standard_normal(samples.shape) + 1j * g.standard_normal(samples.shape)).astype(samples.dtype)
            if stats is not None and stats["model"] == "alpha_stable":
                # Impulsive noise: independent symmetric alpha-stable real and imaginary parts
                # with the scale of the Gaussian parts they replace (identical at alpha = 2).
                ambient = (noise_models.alpha_stable(stats["alpha"], sigma / 2.0, samples.shape, g)
                           + 1j * noise_models.alpha_stable(stats["alpha"], sigma / 2.0, samples.shape, g)).astype(samples.dtype)
            elif stats is not None:
                # K-distributed noise (`ambient_noise.statistics`, sim/noise.py): the gamma
                # texture on the complex envelope; white here, so no filter follows it.
                slow = stats["correlation_time"] > 0.0
                ambient = ambient * noise_models.gamma_texture(
                    samples.shape[1], fs_bb, stats["shape"], stats["correlation_time"], g, samples.shape[0],
                    stats["common"] if slow else False).astype(np.float32)
            samples = samples + ambient
        else:
            spectrum = noise_models.flat_spectrum(noise_spectrum, sonar.center_frequency) if shape == "flat" else noise_spectrum
            samples = samples + noise_models.k_noise(
                samples.shape[1], record.samples.sample_rate, spectrum, rng or np.random.default_rng(),
                noise_statistics, positions=record.rx_positions(), sound_speed=c, sensitivity_db=sens, coherence=coherence)
    if fitted_noise:
        # `receiver.ambient_noise: {model: fitted, file: ...}`: noise with the spectrum, the
        # coherence between hydrophones and the tail of a recording (sim/noise_fit.py), in the
        # recording's sample units (`gain_db` rescales). Added after synthesis on either engine.
        from . import noise_fit
        fc_noise = record.samples.center_frequency if np.iscomplexobj(samples) else 0.0
        samples = samples + noise_fit.scenario_noise(
            fitted_noise, samples.shape[1], record.samples.sample_rate, rng or np.random.default_rng(),
            n_channels=samples.shape[0], center_frequency=fc_noise).astype(samples.dtype)
    taps.sim_tap(record, "noise", samples)
    if overload_cfg is not None:
        # The front end with limits (the record's gain ramp is applied inside it).
        samples = receiver.apply_front_end(record, samples, overload_cfg, pre=pre_window, truth=truth,
                                           arrays=bool(truth is not None and truth.get("arrays")))
    else:
        samples = receiver.apply_hardware_tvg(record, samples)
    taps.sim_tap(record, "hardware_gain", samples)
    if adc:
        samples = receiver.apply_adc(record, samples)
    taps.sim_tap(record, "adc", samples)
    s = record.samples
    kind = s.kind
    s.array = samples
    s.kind = kind
    record.samples = s
    return record


def _noise_spectrum(cfg):
    """Spectrum callable from a scenario's `receiver.ambient_noise` mapping:
    {level_db: 45} for a flat spectrum, or {wind_speed: 5, shipping: 0.5} for the Wenz fit."""
    if not cfg or cfg.get("model") == "fitted":
        return None
    from . import noise as noise_models
    if "level_db" in cfg:
        level = float(cfg["level_db"])
        return lambda f: np.full_like(np.asarray(f, dtype=float), level)
    return lambda f: noise_models.wenz_spectrum_db(f, float(cfg.get("wind_speed", site.DEFAULT_WIND_SPEED_M_S)),
                                                    float(cfg.get("shipping", site.DEFAULT_SHIPPING)))


def _add_true_ins(record, accel: np.ndarray, rate: float) -> None:
    """Attach error-free INS samples covering the receive window (true trajectory)."""
    s = record.samples
    t_end = s.start_time + s.n_samples / s.sample_rate
    pose = record.geometry.rx_pose
    p0, v0 = np.array(pose.position), np.array(pose.velocity)
    out = []
    for t in np.arange(0.0, t_end + 1.5 / rate, 1.0 / rate):
        a = core.InsSample()
        a.t = float(t)
        a.position = tuple(p0 + v0 * t + 0.5 * accel * t * t)
        a.rpy = tuple(np.array(pose.rpy) + np.array(pose.rates) * t)
        a.velocity = tuple(v0 + accel * t)
        a.rates = pose.rates
        out.append(a)
    nav = record.navigation
    nav.ins = out
    record.navigation = nav


def _set_board_clocks(record, sonar: SonarModel, truth: bool, rng=None) -> dict:
    """Write the boards' true clock errors into the record (truth=True) or clear them.
    Returns the truth: {"offset": [...], "drift_ppm": [...]} per board. Each ping draws
    its own jitter on top of the configured offset."""
    meta = record.metadata
    boards = meta.boards
    offsets, drifts = [], []
    for board, cfg in zip(boards, sonar.boards):
        offset = float(cfg.get("clock_offset", 0.0))
        jitter = float(cfg.get("jitter_rms", 0.0))
        if truth and jitter > 0.0 and rng is not None:
            offset += jitter * float(rng.standard_normal())
        drift = float(cfg.get("clock_drift_ppm", 0.0))
        board.clock_offset = offset if truth else 0.0
        board.clock_drift_ppm = drift if truth else 0.0
        offsets.append(offset)
        drifts.append(drift)
    meta.boards = boards
    record.metadata = meta
    return {"offset": offsets, "drift_ppm": drifts}


@dataclass
class Collection:
    """A multi-ping run with a true trajectory and what navigation reported."""

    records: list            # poses as navigation reported them (what a processor gets)
    true_positions: np.ndarray   # [n + 1, 3] true platform position at each transmit (one extra node)
    nav_positions: np.ndarray    # [n, 3]
    times: np.ndarray            # [n] transmit times, s
    scene: Scene


def smooth_disturbance(n: int, rms: float, correlation_pings: float, rng: np.random.Generator) -> np.ndarray:
    """A smooth zero-mean random sequence: white noise through a Gaussian filter, scaled to `rms`."""
    if rms <= 0.0:
        return np.zeros(n)
    pad = int(4 * correlation_pings) + 1
    white = rng.standard_normal(n + 2 * pad)
    k = np.arange(-pad, pad + 1)
    kernel = np.exp(-0.5 * (k / correlation_pings) ** 2)
    out = np.convolve(white, kernel / kernel.sum(), mode="same")[pad:pad + n]
    out -= out[0]                       # start on the nominal track
    return out * rms / max(out.std(), 1e-12)


def simulate_collection(sonar: SonarModel, scene: Scene, *, n_pings: int, interval: float,
                        start=(0.0, 0.0, -15.0), speed: float = 1.2, sway_rms: float = 0.0,
                        heave_rms: float = 0.0, correlation_pings: float = 8.0,
                        nav_knows_disturbance: bool = False, dvl_altitude_noise: float = 0.0,
                        environment: dict | None = None, rng: np.random.Generator | None = None,
                        seafloor_depth: float = 0.0, backend=core.Backend.Host,
                        stop_and_go: bool = False, attitude=None, **ping_kwargs) -> Collection:
    """Fly a straight line along +x with sway (y) and heave (z) disturbances.

    The true trajectory is piecewise linear: during ping i the platform moves at
    v_i = (p_{i+1} - p_i) / interval, the paper's first-order kinematic model. Echoes are
    synthesised from the truth. Each record then carries what navigation would report:
    the nominal straight track (unless `nav_knows_disturbance`), plus a DVL sample with
    the true altitude and surge speed. With `stop_and_go` (a rail) the platform stands
    still during each ping; the disturbances are then position errors of the rail.
    `attitude`, when given, is the true roll, pitch, yaw of each ping (shape (n_pings, 3),
    rad), held constant during the ping; navigation (the INS) reports it exactly. None:
    level. The DVL's velocity_body stays (speed, 0, 0) whatever the attitude.
    """
    rng = rng or np.random.default_rng(0)
    if attitude is None:
        att = np.zeros((n_pings, 3))
    else:
        att = np.asarray(attitude, dtype=float)
        if att.shape != (n_pings, 3):
            raise ValueError(f"simulate_collection: attitude must have shape ({n_pings}, 3), not {att.shape}")
    start = np.asarray(start, dtype=float)
    t = np.arange(n_pings + 1) * interval
    truth = np.tile(start, (n_pings + 1, 1))
    truth[:, 0] += speed * t
    truth[:, 1] += smooth_disturbance(n_pings + 1, sway_rms, correlation_pings, rng)
    truth[:, 2] += smooth_disturbance(n_pings + 1, heave_rms, correlation_pings, rng)
    nominal = np.tile(start, (n_pings, 1))
    nominal[:, 0] += speed * t[:-1]

    records = []
    for i in range(n_pings):
        v_true = np.zeros(3) if stop_and_go else (truth[i + 1] - truth[i]) / interval
        rec = simulate_ping(sonar, make_pose(truth[i], rpy=att[i], velocity=v_true), scene, environment=environment,
                            sequence=i, time_ns=1_700_000_000_000_000_000 + int(round(t[i] * 1e9)),
                            scenario_time=t[i], rng=rng, backend=backend, **ping_kwargs)
        nav_pos = truth[i] if nav_knows_disturbance else nominal[i]
        nav_vel = v_true if (nav_knows_disturbance or stop_and_go) else np.array([speed, 0.0, 0.0])
        geo = rec.geometry
        geo.rx_pose = make_pose(nav_pos, rpy=att[i], velocity=nav_vel)
        geo.tx_pose = geo.rx_pose
        rec.geometry = geo
        dvl = core.DvlSample()
        dvl.t = 0.0
        dvl.velocity_body = (speed, 0.0, 0.0)
        dvl.altitude = float(seafloor_depth - truth[i, 2] + dvl_altitude_noise * rng.standard_normal())
        nav = rec.navigation
        nav.dvl = [dvl]
        rec.navigation = nav
        records.append(rec)
    return Collection(records, truth, nominal if not nav_knows_disturbance else truth[:-1], t[:-1], scene)


@dataclass
class ScenarioResult:
    records: list
    scene: Scene
    config: dict
    true_positions: np.ndarray = None      # [n_pings, 3] platform position at each transmit
    front_end: list = None                 # per ping, what the overload model did (receive.overload / transmit.direct_path), or None
    legs: list = None                      # survey leg index of each record (platform.legs), or None
    times: np.ndarray = None               # [n_pings] scenario time of each transmit, s


def run_scenario(name_or_config, overrides: dict | None = None, output: str | None = None,
                 backend=core.Backend.Host, engine: str | None = None) -> ScenarioResult:
    """Run a scenario (a name under configs/scenarios, a path, or a config dict).

    With `output`, records and ground truth are written to that container file.
    `engine` ("cpp" or "numpy"; default: sim/engine.py) chooses the implementation of the
    eigenray providers, the path pairing and reverberation.
    """
    if isinstance(name_or_config, dict):
        cfg = name_or_config
    else:
        cfg = load_scenario_config(name_or_config, overrides, expand=False)
    typed = cfg                                  # what the scenario says: this is what a written file records
    cfg = acoustics.resolve_sea_state(cfg)     # `sea_state: 0..9` wherever `wind_speed` is accepted
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    # One environment per site (gpusonar/environment.py): inline, a name under configs/environments, or
    # {base: name, ...}. Propagation, time spreading, reverberation, ambient noise and absorption take their
    # site values from it; a typed value that contradicts it is an error unless `environment_overrides` names it.
    env = site.resolve(cfg.get("environment"), strict=True)
    cfg = site.apply_to_scenario(cfg, env, sonar)
    tier = cfg.get("fidelity", "free_field")
    provider = make_provider(tier, env["sound_speed"], cfg.get("propagation"))
    if engine is not None and hasattr(provider, "engine"):
        provider.engine = engine
    if cfg.get("time_spread"):
        from . import timespread
        spread = dict(cfg["time_spread"])
        spread.setdefault("seed", int(cfg.get("seed", 0)))
        provider = timespread.wrap(provider, spread, frequency=sonar.center_frequency)
    if cfg.get("basin_reverberation"):
        from . import basin
        provider = basin.attach(provider, cfg["basin_reverberation"], seed=int(cfg.get("seed", 0)))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    scene = build_scene(cfg.get("scene", []), rng)
    sources = None
    if cfg.get("sources"):
        # Radiating sources (one-way signals): their own generator, so that adding a source
        # does not change the scene or the noise drawn for the rest.
        from . import sources as source_models
        sources = source_models.build_sources(cfg["sources"], np.random.default_rng([int(cfg.get("seed", 0)), 0x50]))

    plat = cfg.get("platform", {})
    position = np.asarray(plat.get("position", [0.0, 0.0, 0.0]), dtype=float)
    velocity = np.asarray(plat.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    rpy = plat.get("rpy", [0.0, 0.0, 0.0])
    pings = cfg.get("pings", {})
    count = int(pings.get("count", 1))
    interval = float(pings.get("interval", 1.0))
    start_ns = int(pings.get("start_ns", 1_700_000_000_000_000_000))
    rx_cfg = cfg.get("receiver", {})

    records, truth_positions = [], []
    n_sub = len(sonar.mux_groups)
    # A rail moves between pings and stands still during each one.
    stop_and_go = bool(plat.get("stop_and_go", False))
    ping_kwargs = dict(
        provider=provider, absorption_model=cfg.get("absorption", site.DEFAULT_ABSORPTION),
        noise_rms=float(rx_cfg.get("noise_rms", 0.0)), adc=bool(rx_cfg.get("adc", True)),
        noise_spectrum=_noise_spectrum(rx_cfg.get("ambient_noise")),
        noise_statistics=(rx_cfg.get("ambient_noise") or {}).get("statistics"),
        noise_options={k: v for k, v in (rx_cfg.get("ambient_noise") or {}).items() if k in ("coherence", "spectrum")},
        fitted_noise=(rx_cfg.get("ambient_noise") if (rx_cfg.get("ambient_noise") or {}).get("model") == "fitted" else None),
        sources=sources, transmit=bool(pings.get("transmit", True)),
        reverberation=cfg.get("reverberation"),
        beam_floor_db=(cfg.get("synthesis") or {}).get("beam_floor_db"), engine=engine)
    front_end = None
    if sonar.config["transmit"].get("direct_path") or sonar.config["receive"].get("overload") is not None:
        front_end = []                     # the truth about overload: beside the records, never in them
        ping_kwargs["front_end_truth"] = front_end
    motion = plat.get("motion")
    raster = plat.get("raster")
    legs_cfg = plat.get("legs")
    leg_of_record, ping_times = None, None
    if legs_cfg is not None:
        # Survey legs: straight runs, each with its own start, heading and length. The
        # platform's yaw follows the heading (0 = +x, 90 = +y), so a side-looking sonar
        # looks to starboard of each leg.
        speed = float(plat.get("speed", np.linalg.norm(velocity)))
        leg_of_record, ping_times, t = [], [], 0.0
        for leg_index, leg in enumerate(legs_cfg):
            heading = np.radians(float(leg["heading_deg"]))
            direction = np.array([np.cos(heading), np.sin(heading), 0.0])
            start = np.asarray(leg["start"], dtype=float)
            n_leg = int(leg["pings"]) if "pings" in leg else int(np.floor(float(leg["length"]) / (speed * interval))) + 1
            for i in range(n_leg):
                pos = start + direction * speed * (i * interval)
                pose = make_pose(pos, (0.0, 0.0, heading), (0.0, 0.0, 0.0) if stop_and_go else speed * direction)
                truth_positions.append(pos)
                seq = len(records)
                records.append(simulate_ping(
                    sonar, pose, scene, environment=env, sequence=seq, sub_sequence=seq % n_sub,
                    time_ns=start_ns + int(round(t * 1e9)), scenario_time=t, rng=rng, backend=backend,
                    **ping_kwargs))
                leg_of_record.append(leg_index)
                ping_times.append(t)
                t += interval
            t += float(leg.get("turn_time", 30.0))           # the turn to the next leg is not simulated
    elif motion is not None:
        # A straight run along +x with sway and heave the navigation does not know about
        # (unless nav_knows_disturbance): the input for motion estimation.
        if np.any(np.abs(velocity[1:]) > 0) or np.any(np.asarray(rpy, dtype=float) != 0):
            raise ValueError("platform.motion needs a level platform moving along +x")
        col = simulate_collection(
            sonar, scene, n_pings=count, interval=interval, start=position, speed=float(velocity[0]),
            sway_rms=float(motion.get("sway_rms", 0.0)), heave_rms=float(motion.get("heave_rms", 0.0)),
            correlation_pings=float(motion.get("correlation_pings", 8.0)),
            nav_knows_disturbance=bool(motion.get("nav_knows_disturbance", False)),
            dvl_altitude_noise=float(motion.get("dvl_altitude_noise", 0.0)), environment=env, rng=rng,
            seafloor_depth=float(motion.get("seafloor_depth", 0.0)), backend=backend,
            stop_and_go=stop_and_go, **ping_kwargs)
        records, truth_positions = col.records, list(col.true_positions[:count])
    else:
        if raster is not None:
            # A stop-and-go raster over x and y at the platform's z: {x: [start, stop, step], y: [...]}.
            xs = np.arange(*[float(v) for v in raster["x"]])
            ys = np.arange(*[float(v) for v in raster["y"]])
            positions = [np.array([px, py, position[2]]) for px in xs for py in ys]
            stop_and_go = True
        else:
            positions = [position + velocity * (i * interval) for i in range(count)]
        for i, pos in enumerate(positions):
            t = i * interval
            # platform.rates (rad/s): the vehicle rolls, pitches and yaws during the ping, from `rpy` at each
            # transmit. The echoes are made with the turning array; the record handed to a processor then
            # holds what navigation would give it: the attitude at the transmit and samples of it over
            # the receive window (navigation.ins, platform.attitude_rate_hz a second), not the rates.
            rates = plat.get("rates")
            pose = make_pose(pos, rpy, (0.0, 0.0, 0.0) if stop_and_go else velocity, rates or (0.0, 0.0, 0.0))
            truth_positions.append(pos)
            records.append(simulate_ping(
                sonar, pose, scene, environment=env, sequence=i, sub_sequence=i % n_sub,
                time_ns=start_ns + int(round(t * 1e9)), scenario_time=t, rng=rng, backend=backend,
                **ping_kwargs))
            if rates and any(float(v) != 0.0 for v in rates):
                _add_true_ins(records[-1], np.zeros(3), float(plat.get("attitude_rate_hz", 10.0)))
                g = records[-1].geometry
                for name in ("tx_pose", "rx_pose"):
                    q = getattr(g, name)
                    q.rates = (0.0, 0.0, 0.0)
                    setattr(g, name, q)
                records[-1].geometry = g

    if output:
        opts = core.WriterOptions()
        opts.config_yaml = dump_yaml(typed)
        opts.compression_level = int(cfg.get("compression_level", 1))
        with core.RecordWriter(str(output), opts) as w:
            for r in records:
                w.write(r)
            w.write_truth("platform_position", np.asarray(truth_positions))
            if len(scene) > 0:
                w.write_truth("scatterer_position", scene.positions)
                w.write_truth("scatterer_velocity", scene.velocities)
                w.write_truth("scatterer_amplitude",
                              np.stack([scene.amplitudes.real, scene.amplitudes.imag], axis=1))
            w.write_truth_text("fidelity", tier)
            w.write_truth_text("labels", json.dumps(scene.labels))
    if ping_times is None:
        ping_times = [i * interval for i in range(len(records))]
    # The result's config: the scenario with its site values filled in and the environment resolved.
    cfg = {**cfg, "environment": env}
    return ScenarioResult(records, scene, cfg, np.asarray(truth_positions, dtype=float), legs=leg_of_record,
                          times=np.asarray(ping_times, dtype=float), front_end=front_end)
