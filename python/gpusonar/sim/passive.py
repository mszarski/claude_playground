"""Records for passive listening: broadband noise sources heard by an array of any layout.

The ping simulator makes echoes of the sonar's own pulse. Listening needs the opposite: no
pulse, and sources that radiate noise of their own. `passive_record` fills an (otherwise
empty) record of a sonar with white noise from point sources at given positions, each element
hearing each source after its own travel time, plus independent noise on every channel. With
`surface_z` every source also has its mirror image in the sea surface (reflection -1): the
direct and surface-reflected arrivals of a source at depth.

It is for testing the passive bearing and elevation stages (libs/live, elevation.py) on
arrays with stacked rows; it knows nothing of propagation loss, bottoms or profiles.
"""

from __future__ import annotations

import numpy as np

from .. import _gpusonar as core


def passive_record(sonar, pose, sources, *, noise_rms: float = 1.0, rng: np.random.Generator | None = None,
                   sound_speed: float = 1500.0, surface_z: float | None = None, time_ns: int = 0):
    """A record of `sonar` at `pose` holding noise only. `sources` is a list of
    {"position": [north, east, down] m, "rms": amplitude at the array's centre}; the record's
    length is the sonar's receive window. Works for raw real and for complex baseband sampling."""
    rng = rng or np.random.default_rng(0)
    rec = sonar.make_record(pose, environment={"sound_speed": sound_speed}, time_ns=time_ns)
    s = rec.samples
    n, fs = s.n_samples, s.sample_rate
    pos = rec.rx_positions()
    centre = pos.mean(axis=0)
    is_complex = s.is_complex()
    f = np.fft.fftfreq(n, 1.0 / fs) + s.center_frequency if is_complex else np.fft.rfftfreq(n, 1.0 / fs)

    def white():
        if is_complex:
            return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
        return rng.standard_normal(n)

    x = noise_rms * np.array([white() for _ in range(len(pos))])
    for src in sources:
        p = np.asarray(src["position"], dtype=float)
        r0 = np.linalg.norm(p - centre)
        spec = (np.fft.fft if is_complex else np.fft.rfft)(float(src["rms"]) * white())
        arrivals = [(p, 1.0)]
        if surface_z is not None:
            arrivals.append((np.array([p[0], p[1], 2.0 * surface_z - p[2]]), -1.0))
        for q, sign in arrivals:
            dist = np.linalg.norm(q - pos, axis=1)                      # [N]
            lead = (r0 - dist) / sound_speed                            # earlier where the element is nearer
            shaped = sign * (r0 / dist)[:, None] * spec[None, :] * np.exp(2j * np.pi * f[None, :] * lead[:, None])
            x = x + (np.fft.ifft(shaped, axis=1) if is_complex else np.fft.irfft(shaped, n=n, axis=1))
    kind = s.kind
    s.array = x.astype(np.complex64 if is_complex else np.float32)
    s.kind = kind
    rec.samples = s
    return rec
