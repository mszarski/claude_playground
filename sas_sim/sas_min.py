"""
sas_min.py  -- Minimal, physically-correct Synthetic Aperture Sonar
              forward simulator + Time-Domain Back-Projection beamformer.

This is the *real* algorithm, just stripped to its core:
  1. Point-scatterer forward model  -> raw baseband I/Q  (the "ping data")
  2. Range compression (matched filter against the transmit chirp)
  3. TDBP beamforming (delay-and-sum with phase, per output pixel)
  4. SLC (complex) + DRC (log-magnitude) imagery

No GPU, no ray tracing, no shadows, no file formats -- those are the
rungs above this one (see the roadmap). But the focusing physics here
is exactly what ApertureLab does; everything else is realism on top.

Run:  python sas_min.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. System parameters  (HISAS-class-ish, scaled down for a fast demo)
# ----------------------------------------------------------------------
c      = 1500.0        # sound speed in water [m/s]
fc     = 100e3         # center frequency [Hz]
B      = 30e3          # chirp bandwidth [Hz]     -> range res = c/(2B) = 2.5 cm
Tp     = 8e-3          # pulse duration [s]
fs     = 80e3          # baseband complex sample rate [Hz] (> B)
K      = B / Tp        # chirp rate [Hz/s]

# Platform (stripmap): flies along +x at fixed cross-track standoff.
track_len = 4.0        # along-track aperture length [m]
d_ping    = 0.03       # along-track spacing between pings [m]
xp = np.arange(-track_len/2, track_len/2, d_ping)   # ping x-positions
n_pings = len(xp)

# ----------------------------------------------------------------------
# 2. Scene: a handful of point scatterers  (x = along-track, y = range)
#    Arranged as a little constellation so focusing is obvious.
# ----------------------------------------------------------------------
R0 = 30.0                                   # nominal cross-track range [m]
scat = np.array([
    [ 0.0, R0     , 1.0],
    [ 1.0, R0-0.6 , 1.0],
    [-1.0, R0-0.6 , 1.0],
    [ 0.6, R0+0.8 , 1.0],
    [-0.6, R0+0.8 , 1.0],
    [ 0.0, R0+1.4 , 0.8],
])   # columns: x, y, reflectivity
sx, sy, sa = scat[:,0], scat[:,1], scat[:,2]

# ----------------------------------------------------------------------
# 3. Fast-time (range) axis
# ----------------------------------------------------------------------
R_all = np.sqrt((xp[:,None]-sx[None,:])**2 + sy[None,:]**2)
R_near, R_far = R_all.min(), R_all.max()
t0 = 2*R_near/c - 1.5*Tp
t1 = 2*R_far /c + 1.5*Tp
t  = np.arange(t0, t1, 1/fs)
n_fast = len(t)

def chirp(tt):
    """Baseband LFM chirp, nonzero over |tt| <= Tp/2."""
    s = np.exp(1j*np.pi*K*tt**2)
    s[np.abs(tt) > Tp/2] = 0.0
    return s

# ----------------------------------------------------------------------
# 4. Forward model: raw baseband I/Q  [n_pings x n_fast]
#    r_n(t) = sum_k A_k * chirp(t - tau) * exp(-j 2pi fc tau),  tau = 2R/c
# ----------------------------------------------------------------------
raw = np.zeros((n_pings, n_fast), dtype=complex)
for n in range(n_pings):
    R   = np.sqrt((xp[n]-sx)**2 + sy**2)     # range to each scatterer
    tau = 2*R/c
    for k in range(len(sx)):
        raw[n] += sa[k]*chirp(t - tau[k])*np.exp(-1j*2*np.pi*fc*tau[k])

# ----------------------------------------------------------------------
# 5. Range compression: matched filter against the transmit chirp
# ----------------------------------------------------------------------
tref = np.arange(-Tp/2, Tp/2, 1/fs)
h    = np.conj(chirp(tref)[::-1])            # matched filter
h   *= np.hanning(len(h))                    # taper -> lower range sidelobes
nfft = 1 << int(np.ceil(np.log2(n_fast + len(h))))
H    = np.fft.fft(h, nfft)

# --- calibrate the integer lag from a known single echo at t = t[cal] ---
cal   = n_fast // 2
probe = chirp(t - t[cal])
pc    = np.fft.ifft(np.fft.fft(probe, nfft) * H)
lag   = np.argmax(np.abs(pc)) - cal          # shift that maps peak -> true tau

rc   = np.fft.ifft(np.fft.fft(raw, nfft, axis=1) * H[None,:], axis=1)
rc   = rc[:, lag:lag+n_fast]                 # align so peak sits at true tau
rc[:, t < (2*R_near/c - Tp/2)] = 0.0         # kill pre-echo convolution ramp

# ----------------------------------------------------------------------
# 6. TDBP beamforming -> SLC image
#    For each pixel: delay-and-sum across pings with focusing phase.
# ----------------------------------------------------------------------
img_x = np.arange(-1.8, 1.8, 0.01)           # along-track pixels
img_y = np.arange(R0-1.8, R0+1.8, 0.01)      # range pixels
GX, GY = np.meshgrid(img_x, img_y)
slc = np.zeros(GX.shape, dtype=complex)

ti = np.real(rc); tq = np.imag(rc)           # interp real & imag separately
for n in range(n_pings):
    R   = np.sqrt((xp[n]-GX)**2 + GY**2)
    tau = 2*R/c
    smp = (np.interp(tau.ravel(), t, ti[n]) +
           1j*np.interp(tau.ravel(), t, tq[n])).reshape(GX.shape)
    slc += smp*np.exp(1j*2*np.pi*fc*tau)     # focusing phase

# ----------------------------------------------------------------------
# 7. DRC (display) + figures
# ----------------------------------------------------------------------
def drc(z):
    m = np.abs(z); m /= m.max()
    return 20*np.log10(m + 1e-6)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].scatter(sx, sy, c='k', s=40)
ax[0].set(title="1. Scene (point scatterers)", xlabel="along-track x [m]",
          ylabel="range y [m]"); ax[0].invert_yaxis(); ax[0].set_aspect('equal')

ax[1].imshow(np.abs(rc), aspect='auto', cmap='viridis',
             extent=[c*t0/2, c*t1/2, n_pings, 0])
ax[1].set(title="2. Range-compressed raw data\n(hyperbolic 'smiles' = unfocused)",
          xlabel="slant range [m]", ylabel="ping #")

im = ax[2].imshow(drc(slc), aspect='equal', cmap='gray', vmin=-40, vmax=0,
                  extent=[img_x[0], img_x[-1], img_y[-1], img_y[0]])
ax[2].set(title="3. TDBP-focused SLC (dB)\nscatterers collapse to points",
          xlabel="along-track x [m]", ylabel="range y [m]")
plt.colorbar(im, ax=ax[2], label="dB", fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sas_min.png"), dpi=110)

# quick numeric sanity check: range resolution on the isolated bottom target
col  = np.abs(slc[:, np.argmin(np.abs(img_x-0.0))])
win  = (img_y > 31.4-0.25) & (img_y < 31.4+0.25)   # isolate (0, 31.4)
prof = np.where(win, col, 0.0)
peak = prof.max(); half = peak/np.sqrt(2)
above = np.where(prof >= half)[0]
res_meas = (above[-1]-above[0])*0.01 if len(above) else float('nan')
print(f"pings={n_pings}, scene scatterers={len(sx)}")
print(f"theoretical range resolution c/(2B) = {c/(2*B)*100:.2f} cm")
print(f"measured -3dB range width near center target ~ {res_meas*100:.2f} cm")
print("saved sas_min.png")
