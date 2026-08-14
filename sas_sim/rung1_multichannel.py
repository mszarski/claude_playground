"""
Rung 1 -- Multichannel receive array (36 channels).

Simulates the sas_min.py scene twice:

  A. single-channel reference: one monostatic ping at every phase-center
     position (0.03 m along-track sampling, as in sas_min.py)
  B. 36-channel array: 36 receive elements at d_rx = 0.06 m spacing,
     platform advancing N*d_rx/2 = 1.08 m per ping (vernier/DPCA layout),
     so the tx/rx phase centers land on exactly the same 0.03 m grid.

Both are focused with TDBP using exact per-channel bistatic delays.

CHECK (pass/fail, printed):
  1. every focused peak lands on its true scatterer position (<= 2 cm)
     in both images
  2. -3 dB range resolution: multichannel within 5% of single-channel,
     and both within [0.5, 1.5]x the c/(2B) theory
  3. -3 dB along-track resolution: multichannel within 5% of single-channel

PHYSICS ASSUMPTIONS (flagged per project brief -- please confirm):
  * Exact bistatic delays in forward model AND beamformer; no phase-center
    approximation anywhere.
  * Receive elements are ideal omnidirectional point hydrophones (no
    element pattern / baffle / shading).
  * Stop-and-hop platform, no propagation loss, no noise (as in anchor).

Run:  python rung1_multichannel.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sas import (chirp, range_compress, simulate_raw, rx_offsets_uniform,
                 tdbp, peak_position, width_3db, drc)
from sas.forward import fast_time_axis

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# System parameters -- identical to sas_min.py
# ----------------------------------------------------------------------
c  = 1500.0
fc = 100e3
B  = 30e3
Tp = 8e-3
fs = 80e3
K  = B / Tp

# Scene -- identical to sas_min.py
R0 = 30.0
scat = np.array([
    [ 0.0, R0     , 1.0],
    [ 1.0, R0-0.6 , 1.0],
    [-1.0, R0-0.6 , 1.0],
    [ 0.6, R0+0.8 , 1.0],
    [-0.6, R0+0.8 , 1.0],
    [ 0.0, R0+1.4 , 0.8],
])

# ----------------------------------------------------------------------
# Collection geometries
# ----------------------------------------------------------------------
n_rx  = 36
d_rx  = 0.06                          # element spacing -> phase centers at 0.03 m
adv   = n_rx * d_rx / 2               # 1.08 m platform advance per ping (DPCA)
x_tx_multi = (np.arange(4) - 1.5) * adv          # [-1.62 -0.54  0.54  1.62]
rx_off     = rx_offsets_uniform(n_rx, d_rx)

# phase centers of the multichannel collection: (tx + rx)/2
pc = (x_tx_multi[:, None] + rx_off[None, :] / 2).ravel()
pc.sort()
dpc = np.diff(pc)
assert np.allclose(dpc, 0.03), "phase centers must form a uniform 0.03 m grid"

# single-channel reference: one monostatic ping at every phase center
x_tx_single = pc.copy()
rx_off_single = np.array([0.0])

print(f"multichannel: {len(x_tx_multi)} pings x {n_rx} ch = "
      f"{len(x_tx_multi)*n_rx} traces, phase centers {pc[0]:.3f}..{pc[-1]:.3f} m")
print(f"single-channel reference: {len(x_tx_single)} pings on the same grid")

# ----------------------------------------------------------------------
# Simulate -> range-compress -> TDBP (each run gets its own fast-time axis)
# ----------------------------------------------------------------------
img_x = np.arange(-1.8, 1.8, 0.01)
img_y = np.arange(R0-1.8, R0+1.8, 0.01)
GX, GY = np.meshgrid(img_x, img_y)

# fine 1D cuts through the isolated bottom target for resolution measurement
xt, yt = 0.0, R0 + 1.4
cut_y  = np.arange(yt - 0.3, yt + 0.3, 0.001)     # range cut at x = xt
cut_x  = np.arange(xt - 0.3, xt + 0.3, 0.001)     # along-track cut at y = yt


def run(x_tx, rx_off):
    t, tau_min = fast_time_axis(x_tx, rx_off, scat, Tp, fs, c)
    raw = simulate_raw(x_tx, rx_off, scat, t, fc, K, Tp, c)
    rc  = range_compress(raw, t, fs, K, Tp)
    rc[..., t < tau_min - Tp/2] = 0.0             # kill pre-echo ramp
    slc = tdbp(rc, t, x_tx, rx_off, GX, GY, fc, c)
    prof_y = tdbp(rc, t, x_tx, rx_off,
                  np.full_like(cut_y, xt), cut_y, fc, c)
    prof_x = tdbp(rc, t, x_tx, rx_off,
                  cut_x, np.full_like(cut_x, yt), fc, c)
    return slc, prof_y, prof_x


slc_s, prof_y_s, prof_x_s = run(x_tx_single, rx_off_single)
slc_m, prof_y_m, prof_x_m = run(x_tx_multi,  rx_off)

# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------
ok = True

# 1. focused peaks on true scatterer positions, both images
for name, slc in (("single", slc_s), ("multi", slc_m)):
    for x0, y0, _ in scat:
        px, py = peak_position(slc, img_x, img_y, x0, y0)
        err = np.hypot(px - x0, py - y0)
        good = err <= 0.02
        ok &= good
        if not good:
            print(f"  FAIL peak ({x0:+.1f},{y0:.1f}) [{name}]: "
                  f"found ({px:+.3f},{py:.3f}), err {err*100:.1f} cm")
print(f"check 1  peak positions within 2 cm (12 peaks) .......... "
      f"{'PASS' if ok else 'FAIL'}")

# 2. range resolution unchanged, and sane vs theory
res_y_s = width_3db(cut_y, prof_y_s)
res_y_m = width_3db(cut_y, prof_y_m)
theory  = c / (2 * B)
rel_y   = abs(res_y_m - res_y_s) / res_y_s
sane    = (0.5 <= res_y_s / theory <= 1.5) and (0.5 <= res_y_m / theory <= 1.5)
c2 = (rel_y <= 0.05) and sane
ok &= c2
print(f"check 2  range res: single {res_y_s*100:.2f} cm, "
      f"multi {res_y_m*100:.2f} cm (diff {rel_y*100:.2f}%), "
      f"theory c/(2B) {theory*100:.2f} cm ................ {'PASS' if c2 else 'FAIL'}")

# 3. along-track resolution unchanged
res_x_s = width_3db(cut_x, prof_x_s)
res_x_m = width_3db(cut_x, prof_x_m)
rel_x   = abs(res_x_m - res_x_s) / res_x_s
c3 = rel_x <= 0.05
ok &= c3
print(f"check 3  along-track res: single {res_x_s*100:.2f} cm, "
      f"multi {res_x_m*100:.2f} cm (diff {rel_x*100:.2f}%) "
      f"...................... {'PASS' if c3 else 'FAIL'}")

print(f"RUNG 1 {'PASS' if ok else 'FAIL'}")

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, slc, title in ((ax[0], slc_s, "single-channel TDBP (dB)"),
                      (ax[1], slc_m, f"{n_rx}-channel TDBP (dB)")):
    im = a.imshow(drc(slc), aspect='equal', cmap='gray', vmin=-40, vmax=0,
                  extent=[img_x[0], img_x[-1], img_y[-1], img_y[0]])
    a.scatter(scat[:, 0], scat[:, 1], s=80, facecolors='none',
              edgecolors='r', linewidths=0.8)
    a.set(title=title, xlabel="along-track x [m]", ylabel="range y [m]")
    plt.colorbar(im, ax=a, label="dB", fraction=0.046)

ax[2].plot(cut_x, drc(prof_x_s), label="single-channel", lw=1.5)
ax[2].plot(cut_x, drc(prof_x_m), '--', label=f"{n_rx}-channel", lw=1.5)
ax[2].set(title=f"along-track cut through ({xt:.1f}, {yt:.1f})",
          xlabel="along-track x [m]", ylabel="dB", ylim=(-50, 2))
ax[2].axhline(-3.01, color='gray', ls=':', lw=0.8)
ax[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "rung1_multichannel.png"), dpi=110)
print("saved rung1_multichannel.png")

raise SystemExit(0 if ok else 1)
