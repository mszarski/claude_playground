"""
sas -- Synthetic Aperture Sonar simulation + beamforming package.

Extends the anchor file `sas_min.py` (which stays runnable and untouched)
into reusable modules:

  waveform.py  : LFM chirp + matched-filter range compression
  forward.py   : point-scatterer forward model (single- or multichannel)
  beamform.py  : Time-Domain Back-Projection (TDBP), exact bistatic delays
  measure.py   : sub-pixel peak / -3 dB resolution measurement helpers

Roadmap rung 1 (multichannel receive array) lives in forward.py/beamform.py;
its correctness check is scripts/rung1_multichannel.py.
"""

from .waveform import chirp, range_compress
from .forward import simulate_raw, rx_offsets_uniform
from .beamform import tdbp
from .measure import peak_position, width_3db, drc

__all__ = [
    "chirp", "range_compress",
    "simulate_raw", "rx_offsets_uniform",
    "tdbp",
    "peak_position", "width_3db", "drc",
]
