# Physics notes: the Sonar Simulation Toolset (SST) as a reference design

Milestone 12 ("realistic propagation and reverberation") reading notes.

Source read in full: R. P. Goddard, "The Sonar Simulation Toolset, Release 4.6:
Science, Mathematics, and Algorithms", APL-UW TR 0702, Oct 2008 (approved for
public release). File: `docs/refs/APLUW_TR_0702.pdf`.

Conventions used in these notes

- "Sec. X, p. N" = TR 0702 section and *printed* page. PDF page = printed page + 10.
- "(n)" = TR 0702 equation number. Equations are transcribed in plain text and
  were checked against the PDF pages where the text extraction was garbled.
- **[TR]** = stated by TR 0702. **[GK]** = general knowledge, *not* from a
  document read for these notes; anything marked "from memory, verify before
  use" must be checked against the primary source before it is coded.
  **[OURS]** = our own inference or design suggestion, not SST's.
- TR 9407 (APL-UW High-Frequency Ocean Environmental Acoustic Models Handbook,
  cited by TR 0702 as [APL Models 1994]) was **not available**. Nothing in
  these notes is taken from it.

---

## 1. How SST is organised

### 1.1 What SST is (Sec. 1.1-1.2, p. 1-3; Sec. 2, p. 6-10)

SST produces sampled multi-channel sound "as heard" by a sonar receiver; it
does not make plots or predict performance. Four signal components (Sec. 2.1,
p. 6): discrete sources (one-way), diffuse noise, discrete scatterers (target
echoes), and reverberation. It is non-real-time, bistatic/multistatic,
broadband, multi-channel (a channel is one element *or* one formed beam), and
ray-based ("useful as low as a few kHz in shallow water", p. 2; its environment
models are best above 10 kHz, p. 7).

### 1.2 Objects (Fig. 1, p. 8; Sec. 2.4, p. 9-10)

| Object | Holds | TR ref |
|---|---|---|
| `Ocean` | depth D(x,y), sound speed c(z), volume attenuation A(f), `surface` and `bottom` `Boundary` objects, list of `ReverbLayer` volume layers | Sec. 6 |
| `EigenrayModel` (+ CASS/GSM subclasses) | maps (rR, rS) to a list of eigenrays; owns the `doTimeSpread` switch | Sec. 5 |
| `Source` | trajectory, list of beams (one per channel), transmitted `Signal` | Sec. 7 |
| `Sonar` (receiver) | trajectory, list of beams (one per channel); output signal | Sec. 7 |
| `Target` | trajectory + highlights (point scatterers moving as a group) | Sec. 9.2 |
| `DirectSignal` / `DirectSpectrum` | one-way source -> receiver sound | Sec. 8 |
| `TargetEcho` | source -> target -> receiver sound | Sec. 9.1 |
| `BBBScatFun`, `BBBDirectionalScat` + `ReverbSignal` | scattering-function reverberation | Sec. 11.3 |
| `DirectionalDopplerDensity` + `DDReverbSignal` | DDD reverberation (new in 4.6) | Sec. 11.2 |
| `BroadbandNoise`, `KNoise`, `HarmonicFamily`, `ModulatedTone` | signal generators | Sec. 4.6 |
| `SumSignal` | adds components into the final ping | Table 1, p. 23 |

An active transmitter is just a `Source`; monostatic = source and sonar share
one trajectory (Sec. 7, p. 49). A target is "a group of receivers and a group
of transmitters back to back" (Sec. 9.1, p. 58).

### 1.3 Data-flow ("pull") architecture (Sec. 4.3-4.4, p. 19-25)

Every time series is a data-flow object in one of four hierarchies: `Signal`
(time), `Spectrum` (frequency, time), `ScatFun` and `DopplerDensity` (Doppler,
frequency, time). Objects are sources/sinks of blocks; a `CopySignal src dst`
command loops `readBlock`/`writeBlock`. Read requests propagate upstream to leaf
objects, results flow downstream. Output is produced **in time order**
("streamable", p. 3), which drives several algorithm choices in Sec. 11.

A ping is assembled as (Sec. 2.4, p. 9-10; Sec. 3, p. 11):

```
ping = SumSignal( DirectSignal(source_i -> sonar)      for each discrete source
                , TargetEcho(source, target_j, sonar)  for each target
                , ReverbSignal(ScatFun, source signal) for each transmitter
                , Gaussian noise with a given PSD      (ambient / self / electronic)
                , any external signal file )
```

### 1.4 Units and frames (Sec. 2.5, p. 10-11)

- Pressure in micro-Pascal; intensity in uPa^2 or dB re uPa^2; angles in
  degrees; everything else MKS.
- Earth frame: (North, East, **Down**), origin at an arbitrary point on the sea
  surface; z = depth, positive down. Earth curvature ignored.
- Platform frame: (Forward, Starboard, Below) about an arbitrary platform
  origin; beam patterns, element offsets and highlights live here.
- Frame mapping (43), p. 50: `r' = M_X(t) (r - r_X(t))`, `S' = M_X(t) S`.
  Trajectories are lists of `Snapshot`s (position, velocity, orientation,
  rotation rate), cubic-interpolated; orientations are quaternions (Sec. 7.1).
- Source signals are pressure "reduced to one meter", units uPa-m; propagation
  loss factor L_p has units 1/m; received signal is uPa (Sec. 5.1, p. 36;
  Sec. 11.2.3, p. 82).
- Signal levels: `ModulatedTone` amplitude `A = 10^(level/20)` (30), p. 33.
- Volume attenuation is entered as a **negative** number in dB/km (example
  `-2.1`, p. 44); in (38) `alpha(f) <= 0` per metre.
- Doppler is a **ratio**: Gamma = received frequency / transmitted frequency
  (Sec. 4.2.2, p. 18). Gamma > 1 is time compression.

### 1.5 Signal representations and transform conventions (Sec. 4.1-4.2, p. 13-18)

Real samples: `x_c(t_n) = x_c(t0 + n h)` (1), with `h < 1/(2 Fmax)` (2).

Complex envelope about centre frequency F, bandwidth W (4)-(6), p. 14-15:

```
x(t)  = sqrt(2) Re{ x~(t) exp(+2 pi i F t) }          (4)
x~(t) = sqrt(2) [ x(t) exp(-2 pi i F t) ]_LP          (5)   LP = ideal low-pass, +-W/2
h <= 1/W                                              (6)
```

sqrt(2) makes `Av{|x~|^2} = Av{x^2}` (power-preserving). Rule of thumb: use
complex envelopes when bandwidth / max frequency < ~25 % (p. 15). Time
convention is exp(+2 pi i f t): a delay tau is the phase factor
exp(-2 pi i f tau) (see (55)).

Windowed frequency domain (short-time Fourier form), update times
`t_u = t0 + u Delta` (8):

```
X_c(f,t_u) = Int w(t - t_u) x_c(t) exp(-2 pi i f (t - t_u)) dt        (7)   units uPa-s
x_c(t)     = Sum_u w'(t - t_u) Int X_c(f,t_u) exp(+2 pi i f (t - t_u)) df   (9)
Sum_u w'(t - t_u) w(t - t_u) = 1                                      (11)
```

Typical: Hann pre-window, rectangular post-window (p. 16).

Time-dependent cross power spectral density (12), p. 17, units uPa^2/Hz:

```
P_cc'(f,t_u) = E{ X_c(f,t_u) X_c'*(f,t_u) } / Int |w(tau)|^2 dtau
```

Scattering function `Z_rr's(T, Gamma, f)`: matrix-valued (receiver channels
r, r') function of two-way time T, Doppler ratio Gamma, frequency f, per source
channel s; a generalised intensity impulse response. A Doppler density
`Q_rs(T, Gamma, f)` is a random realisation whose second moment is Z (13),
p. 18: `Z_rr's = E{Q_rs Q_r's*} / h` (h = time step of the Doppler density).

Window classes (Table 5, p. 34; (31) p. 35), x in [0,1] across the taper:
Cosine `cos(pi x/2)` and Mitchell-McPherson satisfy the squared sum rule (26);
Hann `cos^2(pi x/2)` and Linear `1 - x` satisfy the linear sum rule (11);
also Rectangular and Taylor.

---

## 2. Eigenray models

### 2.1 What an eigenray carries (Sec. 5.1, p. 36)

For end points rS (source) and rR (receiver), eigenray p gives:

| Symbol | Meaning |
|---|---|
| `T_p(rR, rS)` | travel time; **frequency independent** (non-dispersive ocean; small violations go into the phase of L_p, p. 37) |
| `L_p(f, rR, rS)` | complex pressure loss factor, units 1/m: spreading x boundary reflection x absorption; ratio of pressure at rR to source pressure reduced to 1 m |
| `S_Sp = grad_S T_p` | slowness vector at the source; magnitude 1/c(source); direction **opposite** to propagation at the source |
| `S_Rp = grad_R T_p` | slowness vector at the receiver; magnitude 1/c(receiver); direction **along** propagation at the receiver |

No gradients of L_p are provided: loss is assumed constant across an array or
target; big arrays/targets must be split into groups with their own eigenrays
(Sec. 5.2, p. 37).

Local expansion of delay about the end points (32), p. 36, with offsets rho:

```
T_p(rR + rhoR, rS + rhoS) = T_p + S_Rp . rhoR + S_Sp . rhoS + ...        (32)
```

Spherical-wavefront correction replacing each dot product (33), p. 37:

```
tau_p(S, rho) = S . rho + ( |S|^2 |rho|^2 - (S . rho)^2 ) / (2 T_p)      (33)
```

Good for straight-line propagation and horizontal curvature; "less trustworthy
for vertical curvature in the presence of ray bending".

Eigenray as a linear operator (34)-(35), p. 38: the field near rR is the sum
over p of the source field delayed by T_p and convolved with
`l_p(tau) = Int L_p(f) exp(2 pi i f tau) df`, sharply peaked because L_p varies
slowly with f.

### 2.2 Straight-line (isovelocity, flat bottom) model: `EigenrayModel` (Sec. 5.4, p. 38-40)

Valid only for constant c and constant depth D. Method of images. With S_p
surface bounces and B_p bottom bounces, z down, zR receiver depth:

```
R_p = rR - rS - 2 |S_p - B_p| zR z^  +/-  2 B_p D z^                      (36)
      (+ if the bounce nearest the source is the bottom, - if the surface)
T_p = |R_p| / c                                                          (37)
L_p(f) = (1/R_p) exp(alpha(f) R_p) Prod_b beta(s_p, f, r_pb)             (38)
```

- For each nonzero B_p there are four images: S_p = B_p - 1, S_p = B_p + 1,
  and two with S_p = B_p (p. 39). (The pure surface path B_p = 0, S_p = 1 and
  the direct path are the remaining ones; the TR does not say how many orders
  are kept.)
- `s_p` = sine of the grazing angle; `beta` = boundary reflection coefficient
  (Sec. 6.4.1); `alpha(f) <= 0` = absorption per metre.
- Bounce locations (for location-dependent bottoms) (39)-(40), p. 39-40:
  `r_pb = (1 - w_pb) rS + w_pb rR` horizontally, depth 0 or D;
  `w_pb = (zS + b D)/|Z_p|` if bounce 0 is a surface bounce,
  `w_pb = (D - zS + b D)/|Z_p|` if bounce 0 is a bottom bounce, b = 0,1,...
  from the source, `Z_p = R_p . z^`. Bounces alternate.
- Slowness vectors (41): `S_Rp = R_p/(c R_p)`, `S_Sp = -R_p/(c R_p)`, with the
  vertical component of S_Rp sign-reversed when S_p + B_p is odd.

[OURS] (36) as printed needs care: the first image term is written with
`|S_p - B_p| zR`. Re-derive the image depths independently and use (36) only as
a cross-check; the four-image bookkeeping is what matters.

### 2.3 CASS/GRAB (Sec. 5.5, p. 40-41)

GRAB (Gaussian Ray Bundle, Weinberg & Keenan 1996) inside CASS: c may vary in
3-D, bathymetry and bottom type vary with horizontal position, bottom
reflection follows local slope. SST runs CASS as a sub-process
(`CASSEigenrayRun`, recommended) or reads pre-computed files
(`CASSEigenrayModel`, discouraged for consistency reasons). SST passes CASS
the sound speed table and **tables of reflection coefficient in dB versus
grazing angle and frequency, built from `TotalForwardAmp`; phase is dropped**
(Sec. 6.4.1, p. 46; "This needs to be fixed soon"). SST cannot yet specify
horizontally varying sound speed (Sec. 6.2, p. 44).

Heterogeneous bottoms, two options (p. 41): (a) pass each bottom type to CASS
as a separate "environment" (hard limit on count, slow); (b) run CASS with one
location-independent `cassBottom`, then SST multiplies each eigenray amplitude
by (actual reflection coefficient)/(coefficient CASS used) at each bounce. (b)
needs bounce locations and angles, obtained with CASS's "SEPES" variant of
GRAB. CASS prunes weak eigenrays, so choose a hard (low-loss) `cassBottom`.

### 2.4 GSM (Sec. 5.6, p. 41-42)

Generic Sonar Model: range-independent, c(z) only; five eigenray models, of
which MULTIP and FAME are recommended. No longer supported by NUWC.

### 2.5 Interpolation and ray identity (Sec. 5.7, p. 42-43)

Delay, loss and slowness must be **continuous** in the end points; delay
continuity matters most because dT/dt is the Doppler shift. Trivial for the
image model (identity = number and order of bounces). For tabulated eigenrays
SST must match eigenrays at neighbouring grid points before interpolating.
CASS/GSM give a *signature* (numbers of surface bounces, bottom bounces, upper
vertexes, lower vertexes), which is **not unique** (zero or several eigenrays
can share a signature, and the count changes from point to point). SST assigns
ray identities with "a complex set of heuristics that take into account the
closeness of the attributes (especially delay) of one eigenray to a simple
extrapolation of the attributes of an adjacent eigenray". The heuristics are
not given. The interpolation scheme itself (order, variables) is not given.

### 2.6 Frequency dependence

- T_p independent of f. L_p(f) "varies slowly with frequency" (p. 38) and is
  realised as a short FIR filter per eigenray.
- Frequency enters through absorption alpha(f), boundary coefficients
  beta(s, f, r), beam patterns B(f, S'), scattering strength S_l(f, ...).
- Doppler does not shift the frequency at which L_p and B are evaluated
  (Sec. 8.2, p. 55-56, stated for `DirectSpectrum`).

### 2.7 Coherent vs total reflection (Sec. 6.4.1, p. 45-46)

Each `Boundary` provides two complex functions of (sin grazing angle,
frequency, position):

- `CoherentForwardAmp`: used for one-way propagation and target echoes;
  forward-scattered energy is treated as lost.
- `TotalForwardAmp`: used for reverberation legs; forward-scattered energy is
  treated as specularly reflected.

The TR calls this "flawed" (total too large, coherent too small for many
purposes) and offers time spreading (Sec. 12; our 3.6) as the fix. In practice
`CoherentForwardAmp` is only used by the straight-line model; CASS/GSM runs
always get `TotalForwardAmp`.

---

## 3. Time-series synthesis

### 3.1 One-way chain (Sec. 8, p. 53-55)

Combining source (46), eigenray (34), and receiver (44) transformations gives,
per eigenray p, source channel s, receiver channel r (47)-(52), p. 54:

```
x_sp(t) = x_s( t - T_p - tau_p(M_S S_Sp, r_s') )                      (47)  delay
x_p(t)  = Sum_s  FIR[ B_s(f, M_S S_Sp) ] * x_sp                       (48)  source beam filter
y_p(t)  =        FIR[ L_p(f) ]           * x_p                        (49)  eigenray loss filter
x_rp(t) =        FIR[ B_r(f, M_R S_Rp) ] * y_p                        (50)  receiver beam filter
y_rp(t) = x_rp( t - tau_p(M_R S_Rp, r_r') )                           (51)  channel offset delay
y_r(t)  = Sum_p y_rp(t)                                               (52)
```

Three filters and three delays per path, all slowly time-varying because the
platforms move. `DirectSignal` builds, per path, two `VarDelay` objects
sandwiching three `VarFirFilter` objects, summed over paths (Sec. 8.1, p. 54).
Array channels: each channel has a beam pattern `B_c(f, S')`, a phase-centre
offset `r_c'`, and an offset delay from (33) (Sec. 7.2, p. 50-51). Element
level and beam level use the same machinery. Beam classes: Table 7, p. 52
(omni, cone, dipole, line, Dolph-Chebyshev line, piston, rectangular piston,
stick, element-sum, tabulated, product/sum/rotated/weighted).

`DirectSpectrum` (53), p. 55 does the three filters in the windowed frequency
domain; requires the STFT window to be long compared with the filters (54) and
cannot apply receiver channel offsets. "Used heavily in a passive-sonar
application several years ago, and (to my knowledge) has not been used since."

### 3.2 Variable delay: `VarDelay` (Sec. 4.5.1, p. 26-28)

Ideal: `y_c(t) = x_c(t - T_c(t))` (14). Two interpolations:

1. Delay T_c(t), sampled slowly ("a few samples per second or less"), is
   **linearly** interpolated to output sample times.
2. Input signal is interpolated with a **Hann-windowed sinc** of M taps:

```
y_c(t) = Sum_{k=-M/2}^{M/2-1} d( (t - t_n')/h + k ) x_c(t_n' - k h)        (15)
n'     = floor( (t - t0)/h )                                              (16)
d(z)   = cos^2(pi z / M) * sin(pi z)/(pi z),   -M/2 <= z <= M/2           (17)
```

   M - 1 is the "order". For order <= 3 (M <= 4) polynomial interpolation is
   used instead. Coefficients are tabulated on a grid L times finer than the
   sample grid, default L = 512, nearest-entry lookup (18). (As printed, (18)
   reads `d(z) ~ d( floor(z/L + 1/2) L )`, which looks like a typo for
   `floor(z L + 1/2)/L`.)

Accuracy statements: transition width of an M-sample Hann window is about
`2.5/(M h)`; if the signal occupies a fraction a of the Nyquist band, require

```
M >= 2.5 / (1 - a)                                                        (19)
```

e.g. 80 % occupancy needs M >= 13. Default order is 5, "appropriate for no
more than 50 % band coverage". No numerical error figures are given.

Doppler/time dilation is **not a separate step**: it arises from the
time-varying delay. This is why `TargetEcho` works for arbitrarily long signals
(p. 63). For complex envelopes the carrier phase rotation has to accompany the
delay; the TR shows this explicitly only for reverberation, (108)/(114).

### 3.3 Variable FIR filters (Sec. 4.5.2, p. 28-29)

```
y_c(t)     = Int h_c(tau, t) x_c(t - tau) dtau                              (20)
h_c(tau,t) = w(tau) Int_{F-1/(2h)}^{F+1/(2h)} H_c(f,t) exp(2 pi i f tau) df (22)
```

Window method with a Hann window. Requires H_c to change negligibly in t over
the filter length and to be smooth in f on the scale 1/(filter length).
Coefficients are linearly interpolated in t. Direct convolution for short
filters, FFT for longer; break-even 48 taps (`firFourierCutoff`). Filter
lengths are not specified.

### 3.4 First-order ("simple") model and Doppler (Sec. 8.3, p. 56-58)

Not what SST's classes do, but it is the theoretical base of the reverberation
section and is very close to **our** synthesiser. With
`H_rps(f) = exp(-2 pi i f tau_p(M_R S_Rp, r_r')) B_r(f, M_R S_Rp) L_p(f)
B_s(f, M_S S_Sp) exp(-2 pi i f tau_p(M_S S_Sp, r_s'))` (58):

```
emission time:  t' = t - T_p( R_R(t), R_S(t') )           (59)   implicit, no closed form
Gamma_p = 1 - dT_p/dt = (1 - V_R . S_Rp) / (1 + V_S . S_Sp)                (63)
y_r(t)  = Sum_p Sum_s Int dtau h_rps(tau) x_s( Gamma_p [ t - T_p - tau ] ) (64)
```

A spectral line at f_S is received at `f_R = Gamma_p f_S`. The linearisation is
about a base time at the start of a *short* transmit signal.

### 3.5 Two legs and a scatterer (Sec. 9.1, p. 58-59; Sec. 9.3, p. 61-63)

`TargetEcho` is built from `DirectSignal` objects: source -> target highlights,
target transformation, target -> receiver. Cost scales roughly as
`N_r N_p + N_q N_s` (p. 63).

The "simple" pair-sum form (65)-(70), with p = receive-leg eigenray, q =
transmit-leg eigenray, A_T = complex scattering amplitude of highlight T:

```
H_rpTqs(f) = exp(-2 pi i f tau_p(M_R S_Rp, r_r')) B_r(f, M_R S_Rp) L_p(f)
             * A_T(f, S_Tp, S_Tq)
             * L_q(f) Sum_s B_s(f, M_S S_Sq) exp(-2 pi i f tau_q(M_S S_Sq, r_s'))   (70)
y_rT(t)    = Sum_pq Sum_s Int dtau h_rpTqs(tau) x_s( Gamma_pTq [ t - T_p - T_q - tau ] )  (66)
Gamma_pTq  = [ (1 - V_R . S_Rp)/(1 + V_S . S_Sq) ] * [ (1 - V_T . S_Tq)/(1 + V_T . S_Tp) ]  (68)
retarded times:  t''' = t0 - T_p(R_R(t0), R_T(t'''));  t'' = t''' - T_q(R_T(t'''), R_S(t''))  (67)
```

Cost scales as `N_r N_p N_q N_s` filters. The TR notes the linear-Doppler
approximation is "appropriate only for short signals".

### 3.6 Time spreading of boundary-bounce paths (Sec. 12, p. 93-104; Sec. 5.3, p. 38)

Motivation: specular `DirectSignal` paths are fully coherent, reverberation is
fully incoherent; "Both SST's coherent forward reflection model and SST's
reverberation model are inaccurate for near-specular forward scattering"
(p. 99). Processing gain for bounce paths was too optimistic (p. 4).

Algorithm when `doTimeSpread` is true (default) (Sec. 12.4, p. 99-100):

1. For each surface/bottom bounce, split the *total* forward reflection
   coefficient into coherent and incoherent parts "as outlined in Sec. II-6 of
   [TR 9407]". **Formula not in TR 0702.**
2. Coherent part: deterministic, as 3.1, with the split's frequency dependence.
3. Incoherent part: passed through a **stochastic FIR filter**. Tap variances
   follow an "intensity impulse response" computed "as outlined in Sec. II.7a
   of [TR 9407]". **Formula not in TR 0702.** Taps are complex Gaussian.
4. Taps evolve slowly in time: white Gaussian noise through an **AR(1)**
   filter, time scale set by the frequency-spreading width "computed as in
   Sec. II.8 of [TR 9407]". **Formula not in TR 0702.**
5. Stochastic filter is convolved with the deterministic eigenray filter;
   coherent and incoherent filters are summed and used as the path filter.
6. The near-specular part of the reverberation is **removed** from the
   scattering function to avoid double counting (p. 95, 99). How "near
   specular" is delimited is not stated.

Angle spreading is not implemented ("not yet", p. 93).

Control parameters on `Boundary` (p. 100-101); computed from wind speed by
`McDanielSurface` only, user-set (default 0 = no spreading) elsewhere:

| Parameter | Role as stated in TR 0702 |
|---|---|
| `rmsHeight` | coherent/incoherent split. Compare with wavelength / sin(grazing angle): larger -> mostly incoherent, smaller -> mostly coherent. |
| `meanSquareSlope` | decay time of the incoherent tail. Rule of thumb: spreading time ~ mss x (half the total one-way travel time) for a mid-path bounce; ~ mss x (twice the travel time of the shorter leg) for a bounce near one end. |
| `rmsVerticalSpeed` | Doppler spread from surface motion. Second component from motion of the reflection point: (speed of that motion) x sqrt(mss). Total RMS vertical speed = root-sum-square. Doppler spread = f x total RMS vertical speed / (c x sin(grazing angle)). |

[OURS, implemented] `sim/timespread.py` implements this section with open substitutes for
the three TR 9407 items; see `DECISIONS.md` D-62 and `RESULTS.md`.

[OURS] The last formula is transcribed as stated (p. 101); dividing by
sin(grazing angle) is unexpected (a vertically moving reflector normally gives
a shift proportional to sin(grazing)). Treat as unverified until TR 9407 II.8
is in hand.

### 3.7 Sampling / basebanding summary

- Real or complex-envelope sampling at the user's rate; `ResampleSignal`
  converts (windowed finite filters; equations "not yet incorporated", p. 28).
- Signal should occupy <= 50 % of Nyquist at default order 5, <= 80 % at
  order 13 (19).
- Complex-envelope reverberation with different source/receive centre
  frequencies Fx, Fy: (108)-(109), p. 75 (given in 5.1 below).
- Pulse synthesis (`ModulatedTone`) (30), p. 33:
  `x(t) = A e(t) cos(phi(t))`, `phi(t) = phi0 + Int 2 pi fm(tau) dtau`, phase
  integrated numerically (two-point Gauss per sample, exact for cubic FM),
  always phase-continuous.

---

## 4. Target models (Sec. 9.2, p. 59-61)

All SST targets are sets of **highlights**: point scatterers rigidly attached
to the target trajectory. There is **no wave-theory, tabulated target-strength
or aspect-pattern model in TR 0702**.

| Class | Highlight properties | Notes |
|---|---|---|
| `PointTarget` | scattering strength, position and velocity in target frame; echo is a scaled copy of the incident signal | `commonCenter` true: one eigenray set to the target origin, highlight offsets only change arrival **time** (via (33)), not direction. False: separate eigenrays per highlight. `randomPosition` = RMS Gaussian jitter of highlight positions to break up grating effects. |
| `HighlightTarget` | adds a complex, frequency-dependent response and a reception-to-transmission delay per highlight, and a *group number* | highlights in a group share one eigenray set (bow group / stern group) |
| `ExternalTarget` | as `HighlightTarget`, but the highlight list comes from a user program | At the start of each ping SST sends distance and direction to transmitter and receiver and the frequency range over a text pipe; the program returns highlights. **This is the only aspect-dependence mechanism**, evaluated once per ping. |

Equations: the scattering amplitude `A_T(f, S_o, S_i)` in (65)/(70) (complex,
may depend on frequency and on outgoing and incoming slowness) and the Doppler
factor (68). The relation between a highlight's "scattering strength" input and
A_T is not written out; with L in 1/m and the source in uPa-m, A_T has units of
metres, i.e. `|A_T|^2` = differential cross-section [OURS, by dimensional
analysis; `TS = 10 log10 |A_T|^2` re 1 m^2].

---

## 5. Reverberation

SST history (p. 76): point-scatterer method first (REVGEN-4, early SST) but
"no longer used" - removed in the early 1990s "because it was slow" (p. 79);
scattering-function method "standard in SST for a long time"; DDD "new with
SST 4.6 (and, frankly, still immature)".

### 5.1 Common theory (Sec. 10, p. 63-76)

Assumptions (Sec. 10.1): linearity, eigenray propagation, single scattering,
far field of transmitter and receiver, scatterers random and unresolvably
dense, each cell has a distribution of scatterer Doppler.

Reverberation = limit of a sum of target echoes (73), p. 64:

```
y_r(t) = Int d^3R Sum_pqs Int dgamma Int dtau h_rpqs(gamma, R, tau)
                                        x_s( gamma Gamma_pq (t - T_p - T_q - tau) )   (73)
Gamma_pq = (1 - V_R . S_Rp)/(1 + V_S . S_Sq)      platform Doppler             (71)
gamma    = (1 - V_T . S_Tq)/(1 + V_T . S_Tp)      scatterer Doppler            (72)
h_rpqs(gamma,R,tau) = w(tau) Int df H_rpqs(gamma,R,f) exp(i 2 pi f tau)        (74)
H_rpqs(gamma,R,f) = exp(-i 2 pi f S_Rp'.r_r) B_r(f,S_Rp') L_p(f)
                    A_pq(gamma,R,f) L_q(f) B_s(f,S_Sq') exp(-i 2 pi f S_Sq'.r_s)  (75)
```

Separable Poisson assumption (76), p. 65:

```
E{ A_pq(gamma,R,f) A*_p'q'(gamma',R',f') } =
     delta_pp' delta_qq' delta(gamma-gamma') delta(R-R') S(gamma,R,.,.,f) S*(gamma,R,.,.,f')
```

`|S|^2` = differential scattering cross-section per unit volume per unit
scatterer Doppler; integrated over Doppler it is the linear volume scattering
strength. S is taken real = sqrt(scattering strength per unit Doppler).
Boundaries and thin volume sheets are handled by a delta function in depth.
Consequences and caveats stated:

- `delta_pp' delta_qq'`: different eigenray pairs are **uncorrelated**. "A
  high-frequency approximation, and will need to be discarded at lower
  frequencies where Lloyd's Mirror effects in reverberation can be observed as
  striations in the range-Doppler map" (p. 66).
- Gaussian field realisation (77)-(78): `A_pq = S g_pq`, g complex Gaussian,
  delta-correlated. Discretised (79)-(80): integrate g over small cells;
  `E{G G*} = (Delta gamma)_j (Delta R)_i` for the same cell, zero otherwise -
  i.e. **an ordinary complex Gaussian random number with variance equal to the
  cell size**. This is the basis of the point-scatterer method.

Spread/coherence toy results (Sec. 10.6), useful as unit tests:

```
Gaussian time spread sigma_T      -> coherence bandwidth  sigma_f = 1/(2 pi sigma_T)        (89)
Gaussian Doppler spread sigma_g   -> coherence time       sigma_t = 1/(2 pi f sigma_g)      (96)
Gaussian spread sigma_S in S_x    -> coherence length     sigma_d = 1/(2 pi f sigma_S)
                                                                  = lambda/(2 pi sigma_theta)  (103)
```

(sigma_theta = RMS width in sin(theta).) Corollary (p. 73): for a monostatic
sonar the reverberation coherence length at the receiver is about the size of
the transmit aperture. Replica-correlation gain against reverberation grows
3 dB per octave of bandwidth in the ideal single-path case (p. 70).

Complex-envelope form (108)-(109), p. 75 (source centre Fx, output centre Fy):

```
y~_r(t) = exp(-i 2 pi (Fy - Fx) t) Int d^3R Sum_pqs Int dgamma Int dtau h~_rpqs(gamma,R,tau)
          x~_s( gamma Gamma_pq (t - T_p - T_q - tau) )
          exp( -i 2 pi Fx [ T_p + T_q + (1 - gamma Gamma_pq)(t - T_p - T_q - tau) ] )      (108)
h~_rpqs(gamma,R,tau) = w(tau) Int_{-1/2h}^{1/2h} df~ H_rpqs(gamma,R,Fx + f~) exp(i 2 pi f~ tau)  (109)
```

### 5.2 Method A: point scatterers (Sec. 11.1, p. 76-79)

As described, **monostatic and narrow-band** (frequency dependence of
propagation, scattering and beams ignored), except the "noiselet" variant.

Layers: surface, bottom, and one thin sheet at the centre depth z_l of each
volume layer. Choose a cell size (delta_r, delta_theta) such that **each sonar
resolution cell (range, azimuth, elevation, platform Doppler) contains several
cells**. Then, per layer, march outward from r_0 = 0 (110)-(111), p. 77:

```
rho    = delta_r * delta_theta / (2 pi)                 (110)   mean radial step
r_i    = r_{i-1} + rho * ranexp()                       (111)   unit-mean exponential
theta_i= 2 pi * ranun()
R_i    = ( r_i cos theta_i, r_i sin theta_i, z_l )
area_i = 2 pi rho r_i
A_i    = sqrt(area_i) * ranorm()                                zero-mean normal
```

Density and strength: the mean density at horizontal range r is
`1/(2 pi r rho) = 1/(r delta_r delta_theta)` scatterers per m^2, i.e. on
average one scatterer per polar cell [OURS, follows from (110)-(111)]. A_i is
the amplitude "for a uniform scattering strength of unity"; `E{A_i^2} = area_i`
is exactly (80). Each echo is then weighted by

```
A_i * sqrt(scattering strength at the ray angles) * eigenray amplitudes (both legs)
    * transmit and receive beam patterns
```

for **every two-way eigenray pair pq** to that scatterer, Doppler taken from a
bank of pre-computed Doppler-shifted replicas, summed into the output at the
pair's two-way travel time. Because the *same* A_i is used for all pairs, this
version keeps inter-path correlation (Lloyd's mirror striations) - "an
advantage at low frequencies" (p. 78).

Statistics: Gaussian by construction in the dense limit; the method "is
capable of generating non-Gaussian (e.g. K-distributed) reverberation"
(Sec. 11.2.1, p. 79) but details are "deferred to a separate document".

Region selection: none beyond marching outward in range over the full annulus;
beam patterns do the weighting. No culling rule is given.

Variants (p. 78): (i) priority queue of `Echo` records sorted by two-way time
so output can be emitted in time order; (ii) pre-compute and time-sort eigenray
pairs, then place scatterer clusters per pair (time-ordered, but pairs become
independent again); (iii) "noiselets" (Prometheus Inc., for the NUWC WAF
simulator): pre-computed sums of many randomly weighted, slightly delayed
copies of the pulse at one Doppler, used in place of the pulse so far fewer
"scatterers" are needed; broadband; "details will be published separately".

Cost: no formula. Stated scaling: output not in time order; whole ping buffer
in memory; "signals with higher bandwidths required more and more point
scatterers to 'fill in' the distribution" (p. 79).

PSDDD aside (Sec. 11.2.5, p. 84-85): accumulate point-scatterer amplitudes
into the sparse directional Doppler density zeta_s(T, Gamma, f, theta, phi)
(122) and then apply receive beams and the pulse as in 5.3. Suggested as
possibly worthwhile at low frequency where inter-path correlation matters. Not
implemented.

### 5.3 Method B: Directional Doppler Density (DDD) (Sec. 11.2, p. 79-89)

Extra assumptions (Sec. 11.2.1): reverberation from different source channels
is uncorrelated (so transmit elements cannot be "channels"); Gaussian; and the
transmit signal is **short enough to use a single Gamma for the whole pulse**
(`x_s(Gamma t)`). Randomisation is intermediate ("early-ish").

Final synthesis step: sum of delayed, dilated copies of the pulse (113),
(114), (120):

```
y_r(t)  = Sum_s Int dGamma Int dT' u_rs(T',Gamma) x_s( Gamma (t - T') )                 (113)
y~_r(t) = exp(-i 2 pi (Fy-Fx) t) Sum_s Int dGamma Int dT' u_rs(T',Gamma)
          x~_s( Gamma (t - T') ) exp( -i 2 pi Fx (1 - Gamma)(t - T') )                  (114)
y_r(t_i)= Sum_s Sum_j Sum_k [ h dGamma_k u_rs(T'_j,Gamma_k) ] x_s( Gamma_k (t_i - T'_j) )  (120)
```

`u_rs` = reduced Doppler density (units 1/(m s) per unit Doppler; the stored
array includes h dGamma_k and has units 1/m). It is obtained from the Doppler
density Q by a per-(T, Gamma) FIR reduction (116)-(119):

```
q_rs(T,Gamma,tau) = w(tau) Int df Q_rs(T,Gamma,f) exp(2 pi i f tau)                     (117)
u_rs(T',Gamma)    = Int dtau q_rs(T' - tau, Gamma, tau)                    (real)       (116)
u_rs(T',Gamma)    = Int dtau q_rs(T' - tau, Gamma, tau) exp(-i 2 pi Fx (T' - tau))  (complex) (119)
```

If H is frequency independent the reduction is a no-op: `u = Q` (real) or
`u = Q exp(-i 2 pi Fx T)` (complex) (p. 82).

Receiver factored out (121), with (theta, phi) = arrival elevation/azimuth in
vehicle coordinates:

```
Q_rs(T,Gamma,f) = Int dsin(theta) Int dphi zeta_s(T,Gamma,f,theta,phi)
                  B_r(f, S'(theta,phi)) exp(-i 2 pi f S'(theta,phi) . r_r)              (121)
U_pqs(gamma,R,f) = L_p(f) S(gamma,R,.,.,f) L_q(f) B_s(f,S_Sq') exp(-i 2 pi f S_Sq'.r_s)  (123)
```

Random generation from a second-order statistic, the **directional scattering
function** Z_s (124), (127)-(129):

```
E{ zeta_s(T,G,f,th,ph) zeta_s*(T',G',f,th',ph') } = Z_s(T,G,f,th,ph)
        delta(T-T') delta(G-G') delta(sin th - sin th') delta(ph-ph')                   (124)
zeta_s(T,G,f,th,ph) = sqrt( Z_s(T,G,f,th,ph) ) g_s(G,th,ph,T)                            (127)
Z_s(T_nu,G_k,f_m,th_i,ph_j) = 1/( H dG_k dOmega_ij ) Sum_pq Int_{V_ijnu,pq} dR
                              Int_{G_k} dgamma |U_pqs(gamma,R,f_m)|^2                    (129)
```

g_s is complex, unit variance, delta-correlated in all four arguments (128):
**one independent complex Gaussian per (time, Doppler, direction) cell, shared
across frequency** (separability, Sec. 10.7) and across receiver channels -
channel-to-channel coherence comes entirely from summing over directions in
(121). H, dG_k = bin widths in two-way time and Doppler; dOmega_ij = solid
angle of the direction bin. V_ijnu,pq = set of scatterer positions whose
(T_p+T_q, arrival direction) fall in the bin; bins must be "small enough to be
unresolvable by the sonar".

Computation (Sec. 11.2.7, p. 87): for each pair pq lay out the locus of
constant two-way time, sample the integrand along it, and add
(integrand x area represented) to the matching (Gamma, theta, phi) bin. Z_s is
computed on a **coarse "update" time grid**, interpolated to the fine grid h,
square-rooted, multiplied by Gaussians (discretised as in (79)-(80)), then
consumed by (121) (p. 86). Efficiency rests on: Z_s smooth in T; Z_s and zeta
sparse (few directions per T, typically one Doppler cell per direction);
receiver channels enter last so cost is linear in channel count (p. 88-89).

### 5.4 Method C: scattering function (Sec. 11.3, p. 89-93)

"Late randomisation". Bistatic, broadband, multi-channel. Two steps.

**Step 1 - scattering function** (131)-(133), p. 90-91. "A detailed derivation
... [is] beyond the scope of this report" (it is in Goddard 1993, APL-UW 9313,
which we do not have).

```
Z_rr's(T_nu,G_k,f_m) = 1/( H dG_k ) Sum_plq Int_{A_knu,plqs} |U_plqs(f_m,R)|^2
        B_r(f_m, M_R S_Rp(R)) B_r'*(f_m, M_R S_Rp(R)) exp( 2 pi i f_m tau_rr'p(R) ) dR     (131)
U_plqs(f_m,R) = B_s(f_m, M_S S_Sq(R)) L_p(f_m,rR,R) L_q(f_m,R,rS) S_l(f_m, S_Tp(R), S_Tq(R))  (132)
tau_rr'p(R)   = ( M_R S_Rp(R) ) . ( r_r' - r_r'' )        [channel offsets of r and r']      (133)
```

Sum over receive eigenrays p, transmit eigenrays q, layers l (surface, bottom,
volume sheets). `A_knu,plqs` = locus on layer l where two-way time lies in time
bin nu (width H) and platform Doppler lies in Doppler bin k. For straight-line
propagation over a flat or uniformly sloping bottom the iso-time locus is an
**ellipse**; in general SST root-finds an approximately elliptical path and
partitions it by Doppler bin ("details are beyond the scope of this paper").
[OURS: units of Z are 1/(m^2 s) per unit Doppler, from (131) with L in 1/m.]

Notation caveat: (132) writes `S_l` where (123) writes `S` = *square root* of
scattering strength; since (131) squares U, `S_l` in (132) must be the
amplitude (sqrt of the bistatic cross-section per unit area) for the units to
work, even though Sec. 6.4.2 uses `S_l` for the cross-section itself. [OURS]

`BBBScatFun` evaluates (131) directly, "with considerable care for continuity
and accuracy"; beam-pattern evaluations scale as
`N_beams^2 x N_eigenrays^2 x N_layers x N_source_channels` (p. 91).

`BBBDirectionalScat` factors out the receiver (134)-(135), p. 92:

```
Z_s(T_nu,G_k,f_m,th_i,ph_j) = 1/( H dG_k ) Sum_plq Int_{A_kijnu,plqs} U_plqs(f_m,R) dR      (134)
Z_rr's(T_nu,G_k,f_m) = Sum_ij dth_i dph_j Z_s(T_nu,G_k,f_m,th_i,ph_j)
                       B_r(f_m,S_ij) B_r'*(f_m,S_ij) exp( 2 pi i f_m S_ij . (r_r' - r_r'') )  (135)
```

((134) as printed has `U`, not `|U|^2`, and no solid-angle normalisation,
unlike (129); see section 11.) Often faster for many channels and many
eigenrays, but "somewhat jagged"; users are told to compare with `BBBScatFun`
and raise the sample density until they agree (p. 92-93).

**Step 2 - realisation** (Sec. 11.3.1, p. 89-90). Convolve Z with the source
PSD in time and Doppler (130):

```
P_rr's(f,t_u) = Int Int Gamma^-2 Z_rr's(t_u - t_u', Gamma, f) P_s(f/Gamma, t_u') dGamma dt_u'   (130)
```

P_s from (7) and (12) applied to the transmit signal. Then generate a Gaussian
realisation with the Mitchell-McPherson method (Sec. 4.6.1, p. 30-31):

```
P_cc'(f,t_u) = Sum_c'' G_cc''(f,t_u) G*_c'c''(f,t_u)      Cholesky factor (triangular)      (23)
X_c(f,t_u)   = Sum_c' G_cc'(f,t_u) g_c'(f,t_u)            g: iid complex unit Gaussians      (24)
E{ g_c(f,t_u) g*_c'(f',t_u') } = delta_cc' delta_uu' delta(f - f')                          (25)
x_c(t)       = overlap-add of inverse FFTs, eq. (9), with  Sum_u |w'(t - t_u)|^2 = 1         (26)
```

(26) is met by cosine windows at 50 % overlap; the Mitchell-McPherson window
"has somewhat better spectral properties" (its formula is not given). Classes:
`ReverbSpectrum` (130) -> `FactorSpectrum` (23) -> `GaussianSpectrum` (24) ->
`SignalFromSpectrum` (9).

**Stated validity limits**

- Gaussian statistics, and an update interval Delta must exist with P slowly
  varying in time over Delta **and** slowly varying in frequency over 1/Delta
  (p. 30, 90, 98).
- Output has frequency coherence width ~ 1/Delta and coherence time "usually of
  the same order as the inverse of the signal bandwidth but no longer than
  Delta" (p. 90). The algorithms "assume a vanishingly small coherence width"
  (p. 70) and that coherence time < Delta (p. 71).
- Therefore inaccurate "in the neighborhood of transient events like
  near-specular 'startup' of surface or bottom reverberation, where they
  under-estimate the signal coherence" (p. 70); "inaccurate for times when the
  scattering function is changing rapidly" (p. 99).
- Update-interval compromise (p. 98): too short -> generated spectrum too wide;
  too long -> cannot follow transients, gives late/early onsets, slow rise
  times and "regularly spaced lumps" in tails. In the ASIAEX case Delta = half
  the pulse length gave about the right spectral width.
- Processing gain of replica correlation against such reverberation is the
  same as for Gaussian noise with that spectrum - "In fact, that is how it was
  generated" (p. 98).
- Volume layers as thin sheets: reverberation "starts up too suddenly, too
  strongly, and late" early in the ping, especially bistatically (p. 66, 48).
- Eigenray pairs are incoherent (no Lloyd's-mirror striations).
- The cross-power matrix is dense; sparsity is not exploited (p. 73).

---

## 6. Boundary and medium models as used in SST

TR 0702 gives **interfaces and qualitative descriptions only** for all
boundary scattering and loss models. The only medium/boundary formulas in the
report are the image-model loss (38) and the Thorp absorption (42).

### 6.1 Sound speed, depth, absorption (Sec. 6.1-6.3, p. 43-45)

- Depth: `D(x,y)` (x north, y east), constant or bilinear (up to cubic) table.
- Sound speed: `c(z)`, default 1520 m/s; a table is passed to CASS/GSM, which
  interpolate themselves. No sound-speed equation (no Mackenzie/Chen-Millero).
- Volume attenuation `A(f)` in dB/km (negative), constant, table, or Thorp
  (42), from Urick (1983) p. 108 with the last two terms omitted to match CASS:

```
alpha [dB/kyd] = 0.1 f^2/(1 + f^2) + 40 f^2/(4100 + f^2),     f in kHz                 (42)
```

  Stated validity: 4 deg C, depth ~1 km. [OURS] conversions: dB/km =
  dB/kyd / 0.9144; amplitude exponent in (38) is
  `alpha_Np/m = -(dB/km) ln(10) / (20 * 1000)`.

### 6.2 Boundary interface (Sec. 6.4, p. 45-48)

Each `Boundary` supplies `TotalForwardAmp(sinAngle, frequency, pos)`,
`CoherentForwardAmp(sinAngle, frequency, pos)` (complex) and
`BistaticStrength` = `S_l(f, S_p, S_q, r)`, the dimensionless differential
scattering cross-section per unit area for incoming slowness S_q and outgoing
slowness S_p (reduces to two grazing angles and a bistatic angle). "Under most
conditions, the bistatic strength is strongly peaked in the region of the
specular direction" (p. 46).

Monostatic models (`APLSurface`, `APLBottom`) are made bistatic by the
geometric mean of the backscattering strengths at the two grazing angles
(p. 46) - same as CASS; acceptable near backscatter and for bubble-dominated
surface scatter, "particularly bad" near specular (no specular peak):

```
S_bistatic(theta_in, theta_out) = sqrt( S_back(theta_in) * S_back(theta_out) )
```

| Class (Table 6, p. 47) | Use | What TR 0702 says | Source it defers to |
|---|---|---|---|
| `Boundary` | either | user tables of reflection coefficient and monostatic backscatter vs angle and frequency; defaults: coefficient 1, no scattering | - |
| `APLSurface` | surface, monostatic, > 10 kHz | input: wind speed only | TR 9407 |
| `GilbertSurface` | surface, bistatic, ~1-10 kHz (bubble layer) | input: wind speed only | Gilbert, JASA 94 (1993); Kulbago thesis (1994) |
| `McDanielSurface` | surface, bistatic, > 10 kHz | input: wind speed; also computes rmsHeight, meanSquareSlope, vertical speed for time spreading | McDaniel ARL-PSU TM 90-88; Lang & Culver TM 92-342; Donelan et al. (1985) |
| `APLBottom` | bottom, monostatic, > 10 kHz | parameters: roughness, sediment sound speed, absorption, sediment volume scattering; named types | TR 9407; Mourad & Jackson (1989) |
| `JacksonBottom` | bottom, bistatic, mid to high frequency | same parameter set; named types (e.g. `MediumSand`, `CoarseSilt`, `FineSand`) | Williams & Jackson, JASA 103 (1998); TR 9407; others |
| `GriddedBoundary` | location-dependent | nearest-neighbour lookup of `Boundary` objects on an x-y grid | - |

Forward loss statements (p. 47-48):

- Surface models: `TotalForwardAmp` = everything except energy absorbed by
  bubbles. `CoherentForwardAmp` additionally removes scattered intensity
  "using a simple model of surface roughness [TR 9407]"; loss is large when
  wave height >= acoustic wavelength.
- Bottom models: Total = Coherent; reflected intensity = incident minus the
  fraction refracted into the bottom, "estimated using a lossy Rayleigh
  coefficient [Mackenzie 1959]".

### 6.3 Volume scattering (Sec. 6.5, p. 48)

List of `ReverbLayer`s: constant volume scattering strength between an upper
and lower depth, plus a mean horizontal scatterer velocity (current). Each
layer is collapsed onto a sheet at its centre depth. No scattering-strength
model or values are given. [OURS: the sheet's areal strength is then the
volume strength times layer thickness - implied, not stated.]

### 6.4 What is missing, and the open-literature substitutes

[OURS, implemented] Bottom backscatter / bistatic: `sim/bottom.py`, derived there
(`DECISIONS.md` D-63). Volume layers as sheets: `reverb.volume_layer_strength`. Surface
statistics from wind: `acoustics.mean_square_slope`, `rms_surface_vertical_speed`.

All of the following are **[GK]** - cited from general knowledge, not from a
document read for these notes. Any formula below is "from memory, verify
before use".

| Need | TR 0702 gives | Substitute to implement | Notes [GK] |
|---|---|---|---|
| Absorption | Thorp (42) | Francois & Garrison (JASA 72, 1982, two papers) | Thorp is a low-kHz fit; Francois-Garrison covers roughly 0.2 kHz-1 MHz with T, S, depth, pH. Above ~1 MHz (our 2.7 and 5 MHz probes) use the pure-water viscous term (Fisher & Simmons 1977 / F-G third term). Coefficients not reproduced here - take from the papers. |
| Sound speed | table only | Mackenzie (1981) nine-term equation or Chen-Millero/TEOS-10 | - |
| Bottom forward loss | "lossy Rayleigh coefficient" (no formula) | Rayleigh fluid-fluid reflection coefficient with complex sediment sound speed | `V = (m sin th - P)/(m sin th + P)`, `P = sqrt(kappa^2 - cos^2 th)`, `m = rho_sed/rho_w`, `kappa = (1 + i delta)/nu`, `c_sed = c/(1 + i delta)`, th = grazing angle (checked against open sources: `docs/REFERENCES.md` Sec. 12.4; correction 2026-11-20, the row was from memory and wrote the index inverted). Sign of the imaginary part depends on the time convention; SST uses exp(+2 pi i f t). |
| Bottom backscatter / bistatic | pointer to Jackson models | (a) Lambert's law as a floor; (b) Jackson rough-interface + sediment-volume model (Jackson et al. 1986; Mourad & Jackson 1989; Williams & Jackson 1998; Jackson & Richardson, *High-Frequency Seafloor Acoustics*, 2007) | Lambert, from memory, verify before use: `sigma = mu sin(th_i) sin(th_s)`, `10 log10 mu` ~ -27 dB (Mackenzie value) as a classic default. Correction 2026-11-20: the code's default is -29 dB (`bottom.DEFAULT_LAMBERT_MU_DB`, D-89), the one printed value found (Muggleworth, NPS 1994, hdl 10945/28529, p. 34, quoting Mackenzie 1961 for deep water at 530 and 1030 Hz); -27 dB was from memory. Either is a low-frequency value, not one for a sonar band. |
| Surface backscatter | pointer to TR 9407 / McDaniel / Gilbert | Chapman & Harris (JASA 34, 1962) empirical formula for low-mid frequency; a bubble-layer + roughness model for > 10 kHz (Dahl; McDaniel 1993 review) | Chapman-Harris coefficients not reproduced here; take from the paper. |
| Surface forward loss, coherent part | "simple model of surface roughness" | Rayleigh-parameter coherent reflection | From memory, verify before use: `|R_coh| = exp(-2 k^2 h_rms^2 sin^2 th)`. This is consistent with the TR's qualitative rule (rmsHeight vs lambda/sin th) but the TR does not give it. |
| Surface loss, bubbles | "fraction absorbed by bubbles" | APL-UW bubble-layer surface loss (TR 9407 family) or the Hall-Novarini bubble population model (Hall 1989; Novarini et al. 1998) with Ainslie (2005) surface-loss treatment | No formula reproduced. |
| Surface wave statistics | computed inside `McDanielSurface` from wind speed | Pierson-Moskowitz spectrum for rms height; Cox & Munk (1954) for mean-square slope | From memory, verify before use: PM `h_rms ~ 0.0053 U^2` (U at 19.5 m, SI); Cox-Munk `mss ~ 0.003 + 0.00512 U` (U at 12.5 m). |
| Time-spread intensity impulse response | pointer to TR 9407 II.7a | Dahl's forward-scatter model (JASA 100, 1996; JASA 105, 1999; IEEE JOE 26, 2001 - all cited by TR 0702) | Use the TR's own rule of thumb (3.6) for the decay time until Dahl's expressions are checked. |
| Volume scattering strength | user input | tabulated Sv per layer; typical ocean values from Urick ch. 8 | - |
| Ambient noise spectrum | nothing | Wenz (JASA 34, 1962) curves; Knudsen wind noise; thermal noise above ~100 kHz | From memory, verify before use: thermal noise `NL ~ -15 + 20 log10 f[kHz]` dB re uPa^2/Hz (Mellen 1952). |

---

## 7. Noise (Sec. 4.6, p. 29-33; Sec. 2.4, p. 10)

TR 0702 contains **no ambient-noise model** (no Wenz/Knudsen spectra, no
shipping or wind dependence, no spatial-coherence model). "SST has no explicit
model" for distributed noise sources (p. 30). What it provides:

- **Gaussian noise with a user-specified PSD.** Single-channel stationary:
  `BroadbandNoise` (PSD as a table). Multi-channel, possibly non-stationary:
  the Mitchell-McPherson chain (23)-(26) driven by a user-supplied
  cross-spectral density matrix `P_cc'(f, t_u)`. **Spatial coherence across an
  array is whatever the off-diagonal elements of that matrix say**; SST
  factors it by Cholesky per frequency and update, multiplies by iid complex
  Gaussians, and overlap-adds inverse FFTs with squared-sum-rule windows.
  Requirement: P slowly varying over Delta in time and over 1/Delta in
  frequency. Added to the ping with `SumSignal`.
- **K-distributed noise**, `KNoise` (Sec. 4.6.2, p. 31-32): stationary,
  single-channel, **white**, one independent K-distributed draw per sample
  (level in dB + shape parameter; shape > ~10 is essentially Gaussian); colour
  it afterwards with a `VarFirFilter` whose response goes to zero at the
  Nyquist edges. Frequency-domain synthesis is rejected for non-Gaussian noise
  because the inverse FFT re-Gaussianises it (central limit theorem).
  Kurtosis `gamma_2 = m4/m2^2 - 3` (27). No multi-channel K noise.
- **Tonals**: `HarmonicFamily`, `x(t) = Sum_n A_n cos(2 pi n f1 t + phi_n)` (29),
  for machinery lines on a `Source`.
- Radiated noise of a vehicle = `SumSignal` of harmonic families and broadband
  noise, used as a `Source` signal and propagated by `DirectSignal` (so it
  acquires multipath, Doppler and array coherence physically).

[GK] For the coherence matrix we must supply: isotropic 3-D noise
`sinc(2 pi f d / c)` between omni elements; surface-generated noise per Cron &
Sherman (JASA 34, 1962). From memory, verify before use.

---

## 8. Statistics

- Reverberation is **assumed Gaussian** (Sec. 10.4, p. 66-67): "Reverberation
  is Gaussian, and echoes that are not Gaussian are target echoes." False
  targets (rock outcrops) must be modelled explicitly as targets.
- The scattering-function method can only produce Gaussian reverberation (it
  is shaped Gaussian noise).
- Point-scatterer and DDD methods are "capable of generating non-Gaussian
  (e.g. K-distributed) reverberation"; one DDD design objective is "Plan for
  extension to non-Gaussian statistics" (p. 79-80). The method is "described
  in a separate document" - **not in TR 0702**. Reference given: Abraham &
  Lyons, IEEE JOE 27 (2002).
- K-distributed *noise* exists (`KNoise`), white and single-channel only.
- Time-spread FIR taps are complex Gaussian with AR(1) time evolution.
- Poisson (delta-correlated) scatterers in position and scatterer Doppler;
  eigenray pairs uncorrelated; scattering strength separable in frequency.

---

## 9. Worked examples and validation in TR 0702

There are **no fully specified numerical test cases** in TR 0702. It contains
two illustrative figures with partial parameters, a script fragment, and
several closed-form results. No tabulated numbers, no error metrics.

| Item | Where | Parameters given | Stated result | Reproducible? |
|---|---|---|---|---|
| "Pursuit" example | Sec. 3, p. 11-13, Fig. 2 | high-speed vehicle, monostatic forward-looking sonar, 5 receive channels (one shown), narrow-band tone near 10 kHz, three-highlight fleeing target; display 0.5-5 s, 9.6-10.2 kHz | surface reverberation dominates; "reverberation hook" from low early Doppler (sidelobes, overhead) to high late Doppler (main lobe); bottom-bounce streak from ~1.7 s; target echo at ~2.8 s and 10.13 kHz, below reverberation level but Doppler-separated; runs in a few seconds | Qualitatively only. Speeds, depths, beam patterns, pulse length, wind are **not** given (they are in the SST distribution's example script, which we do not have). |
| ASIAEX Run/Set 22 forward scatter | Sec. 12.2-12.6, p. 95-104, Figs. 3-5 | 8 kHz, 3 ms pulse, receiver ~500 m from transmitter, shallow water, low wind (quoted 4 m/s; 3 m/s fits the tail), 20 pings, `McDanielSurface`, bottom `FineSand { rmsHeight = 0.012; meanSquareSlope = 0.045 }` ("fit to the data by eye"), update intervals 1.5 ms and 0.1 ms, time axis 0-35 ms after first arrival | four arrivals (direct, surface, bottom, surface-bottom + bottom-surface); without time spread peaks are crisp and identical ping to ping; with it, tails and ping-to-ping variability resemble the data; multi-bounce peaks "too high immediately after the pulse and too low later"; surface Doppler spread < 10 Hz | Partially. Water depth, source/receiver depths, sound speed, and the McDaniel/TR 9407 formulas are **not** given. Data from Dahl (2002). Only figures, no numbers. The author states parameters were tuned ("I cheated"). |
| `VarDelay` script | Sec. 4.3, p. 20-21 | two-tone signal (220 Hz fundamental; harmonics 1 at -3 dB / 0 deg, 2 at -6 dB / 90 deg), 8 kHz real sampling, t = -2..5 s; delay samples (1.00 1.10 1.25 1.45 1.7 2.0) s at t = 0..5 s | tones "get lower as the rate of change of delay gets larger" | Yes - a good unit test for a variable-delay resampler: expected instantaneous frequency is `f (1 - dT/dt)` with piecewise-linear T. |
| Interpolator design rule | (19), p. 27 | band occupancy a | M >= 2.5/(1 - a); a = 0.8 -> M >= 13; default order 5 for <= 50 % | Yes - measure the passband/stopband response of (17). |
| Spread vs coherence | (89), (96), (103), p. 68-73 | Gaussian spreads | closed-form coherence widths | Yes - statistical unit tests for any reverberation generator. |
| Replica-correlation gain | p. 70 | single path, simple target | signal-to-reverberation gain +3 dB per octave of bandwidth | Yes, as a property test. |
| Doppler factor | (63), (68) | velocities and slowness vectors | closed form | Yes - check our exact solver against it to first order in v/c. |
| Image-model geometry | (36)-(41) | - | closed form | Yes - cross-check our image-source provider. |
| Mitchell-McPherson | (23)-(26) | PSD matrix | ensemble-average PSD converges to the input | Yes, as a property test. |

Other validation mentioned but not contained: Goddard (2000) APL-UW TM 5-00
(North Sea bistatic reverberation, 1998), Eggen & Goddard (2002), Rouseff et
al. (2001). Sec. 13 (p. 105) lists V&V as "a continuing effort".

---

## 10. Mapping to our suite

Sonar keys: P5/P2.7 = 5 and 2.7 MHz probes (1-10 m); FLS = 300 kHz, 300 m;
SAS = 200 kHz, 150 m; HA = 20-35 kHz 8-hydrophone array, 2 km; FA = 6-10 kHz
32-hydrophone flank array, 1 s LFM, 10 km. Tiers: FF = free_field,
IS = image_source, ER = eigenray.

| SST concept | What we have | What to build | Matters for |
|---|---|---|---|
| Eigenray record: T_p, complex L_p(f), slowness at both ends (Sec. 5.1) | `EigenraySet{delay, complex amplitude, launch dir, arrival dir, n_surface, n_bottom, valid}` | Launch/arrival directions + local c are equivalent to slowness vectors. Add: (a) frequency dependence of amplitude - store the frequency-independent geometric part plus per-bounce (grazing angle, position) so `beta(s,f,r)` and absorption are applied at the instantaneous frequency; (b) a stable ray identity. | IS, ER; all sonars |
| Image model (36)-(41) | image-source provider with constant reflection coefficients | Angle- and frequency-dependent complex `beta` per bounce; bounce positions (39)-(40) for patchy bottoms; verify against (36)-(41). | IS; FLS, SAS, HA, FA |
| CASS/GRAB tables + interpolation (Sec. 5.5, 5.7) | Bellhop provider planned, arrivals cached on a range-depth grid | Ray-identity matching between grid nodes before interpolating: group by signature (surface/bottom bounce counts; turning points if available), then match by delay extrapolation as SST does. Interpolate delay so it is continuous (it sets Doppler); never interpolate between different identities. Keep bounce angles so reflection coefficients can be re-applied per frequency (the `cassBottom` ratio trick, p. 41). | ER; HA, FA |
| Spherical-wavefront offset delay (33) | exact per-element arrival-time solve in FF/IS | With cached Bellhop arrivals at the array centre, use (33) for per-hydrophone delays instead of per-element ray traces. Caveat: unreliable for vertical curvature with ray bending. | ER; FA (long array), HA |
| Coherent vs total forward amplitude (Sec. 6.4.1) | one constant coefficient | Two amplitudes per boundary: coherent for target/direct legs, total for reverberation legs. | IS, ER; FLS, HA, FA |
| One-way chain: delay + 3 FIR filters per path (47)-(52) | analytic pulse at solved arrival times, amplitude at instantaneous frequency | Nothing to port. See "Analytic-pulse synthesis vs SST" below. | all |
| `TargetEcho`, highlights (Sec. 9) | point scatterers with complex amplitude and velocity; sum over (outbound, return) eigenray pairs = the "simple model" (66) with exact Doppler | Optional: highlight delay and frequency response; position jitter (`randomPosition`); aspect dependence via a per-ping callback (the `ExternalTarget` pattern). | all |
| Time spreading (Sec. 12) | none | Per bounce path: coherent tap + K incoherent taps at delays tau_k after the specular delay; tap variance from an intensity impulse response (decay time from the mss rule of thumb, shape from Dahl - see 6.4); complex Gaussian taps evolving ping-to-ping and within a ping as AR(1) with the Doppler-spread bandwidth. In our engine each tap is one more analytic-pulse arrival. Tap spacing ~ 1/bandwidth. Remove the near-specular lobe from the reverberation scattering strength when enabled. | IS, ER; HA, FA first; FLS multipath later |
| Point-scatterer reverberation (Sec. 11.1) | seafloor speckle patch generator (random points, complex Gaussian amplitudes) | Make it quantitative: per scatterer representing area dA, complex amplitude variance = sigma(f, angles) x dA, sigma = linear scattering cross-section per unit area from the boundary model at the actual incident/scattered angles; several scatterers per resolution cell; polar marching (110)-(111) or uniform Poisson over the footprint; cull by beam footprint and range gate. Add surface and volume-sheet (or true 3-D volume) layers. Reuse the **same** scatterer amplitude for all eigenray pairs so multipath interference is retained. | FF, IS; **FLS, SAS**, P5/P2.7 (volume) |
| Scattering-function reverberation (Sec. 11.3) | none | Step 1: accumulate `|U|^2 dA` over layer cells into (T, Gamma, theta, phi) bins per eigenray pair and layer - (129)/(134) done as a scatter-add; step 2: either (130)+(23)-(26) or the DDD route below. | ER (also IS); **FA**, HA |
| DDD synthesis (113), (120), (121), (127) | none | Recommended realisation for FA - see below. | ER; **FA**, HA |
| Volume layers as sheets (Sec. 6.5) | none | For P5/P2.7 use true 3-D point scatterers in the beam volume (sheet approximation is wrong at 1-10 m); sheets acceptable for FA. | P5/P2.7; FA |
| Absorption: Thorp (42) | absorption at instantaneous frequency (model unspecified here) | Francois-Garrison for 6 kHz-300 kHz; pure-water term at MHz. Thorp only as a cross-check with CASS-style results. | all |
| Boundary models (Table 6) | constant coefficients; refracting sediment half-space | `Boundary` interface with `coherent_forward_amp`, `total_forward_amp`, `bistatic_strength`, plus `rms_height`, `mean_square_slope`, `rms_vertical_speed`. First implementations from open literature (6.4). | IS, ER |
| Gaussian noise with PSD and channel coherence (Sec. 4.6.1) | white Gaussian only | Mitchell-McPherson generator: PSD matrix (Wenz spectrum x coherence model) -> Cholesky per frequency -> iid complex Gaussians -> overlap-add with 50 %-overlap cosine windows. Stationary case needs one factorisation per frequency. | HA, FA (array coherence); FLS/SAS (coloured, per-channel independent is enough) |
| K-distributed noise (Sec. 4.6.2) | none | Time-domain white K draws + FIR colouring. Low priority. | FLS (snapping shrimp) |
| Complex envelope, sqrt(2) convention (4)-(5) | analytic evaluation | Check our baseband normalisation against (4)-(5) wherever we quote levels in uPa. | all |

### Analytic-pulse synthesis vs SST

SST moves **sampled** signals through interpolating delays (Hann-windowed
sinc, linear interpolation of a slowly sampled delay) and windowed FIR filters.
We evaluate the transmit pulse **analytically** at exactly solved arrival
times.

- *Delay and Doppler.* Ours is exact; SST's is limited by (19) and by linear
  interpolation of the delay track. No reason to adopt `VarDelay`. For the FA
  case (1 s pulse, ~2 m/s) time dilation over the pulse is ~2.7 ms, an order of
  magnitude more than 1/bandwidth (0.25 ms), so true dilation is mandatory;
  SST's first-order formulas (63)/(68) with a fixed Gamma per pulse are a
  "short signal" approximation that we do not need.
- *Frequency-dependent loss, beams, reflection.* SST: FIR filter built from
  H(f). Ours: multiply by H at the instantaneous frequency. For CW, and for
  LFM with large time-bandwidth product (stationary-phase regime), this equals
  SST's result when H(f) is smooth over the band - the same smoothness SST
  requires for its short FIR filters. It differs (a) at pulse edges and for
  short/broadband pulses such as the Gaussian probe pulses, where H should be
  applied to the pulse spectrum (closed form for a Gaussian with locally
  log-linear H), and (b) when H(f) has structure narrower than the local sweep
  resolution. Neither is a concern for single-path boundary and absorption
  terms; **it does matter for anything with intra-path delay spread**, i.e.
  time spreading, which must be represented as extra arrivals (taps), not as a
  multiplicative H.
- *Pair-sum cost.* Our echo is a sum over all (outbound, return) pairs, i.e.
  SST's `N_p N_q` "simple model" scaling rather than `TargetEcho`'s
  `N_p + N_q`. Fine for targets and for FLS/SAS scenes with a handful of
  paths; it is exactly why SST abandons explicit scatterers for long-range
  multipath reverberation.

### Reverberation: which method where

- **FLS and SAS: point scatterers.** Imaging needs a scene that is phase-
  consistent across elements and, for SAS, **across pings**. The
  scattering-function method draws independent Gaussian noise per ping and
  has no persistent scatterers; TR 0702 does not address ping-to-ping
  coherence at all [OURS]. Use Method A with our exact two-way synthesis,
  frequency-dependent amplitudes, bistatic angles, and frozen scatterers
  (optionally with scatterer Doppler for the surface). TR's own warning
  applies: scatterer count grows with bandwidth and resolution; this is the
  GPU's job. Non-Rayleigh statistics come for free by lowering scatterer
  density or modulating strength per patch.
- **FA (and HA at long range): scattering function / DDD.** Explicit scatterers
  out to 10 km over many eigenray pairs are not affordable. Build the
  directional scattering function Z_s(T, Gamma, theta, phi) from the eigenray
  provider (accumulation as in (129)/(134)), then prefer the **DDD realisation**
  over (130) + Mitchell-McPherson [OURS]:
  1. `zeta = sqrt(Z_s) x iid complex Gaussian` per (T, Gamma, direction) cell on
     the fine time grid (127);
  2. per hydrophone, sum over the sparse set of occupied directions with
     element pattern and phase `exp(-i 2 pi f S'.r_r)` (121) - linear in channel
     count, no 32 x 32 Cholesky, spatial coherence emerges physically;
  3. for each Doppler bin, convolve `u_rs(T, Gamma_k)` with the analytically
     dilated pulse `x_s(Gamma_k t)` over T (120) - an FFT convolution per
     Doppler bin, using our analytic pulse (with the complex-envelope phase
     factor of (114)).
  This keeps the transmitted waveform in the reverberation (correct matched-
  filter behaviour and frequency coherence, including onsets), which the
  noise-shaping route (130) does not. Limits to respect: single Gamma per
  pulse (geometry change over 1 s at 2 m/s is 2 m - fine at km ranges, check
  at short range); eigenray pairs incoherent; Gaussian only unless the
  per-cell draws are replaced.
  At 6-10 kHz we are at the bottom of SST's intended range: TR's own caveat on
  `delta_pp' delta_qq'` (Lloyd's mirror striations) applies, and PSDDD
  (shared amplitude across pairs) is the documented escape route.
- **P5/P2.7:** free-field 3-D volume point scatterers; nothing in SST's
  layer/sheet machinery applies at 1-10 m.

---

## 11. Open questions and gaps

Needs TR 9407 (not available) - TR 0702 only points to it:

1. `APLSurface` and `APLBottom` backscattering strength; surface bubble
   absorption loss; coherent-loss "simple model of surface roughness".
2. Coherent/incoherent split of the forward reflection coefficient (TR 9407
   Sec. II-6), the forward-scatter intensity impulse response (II.7a), the
   frequency-spread width (II.8), and "Eq. II-43a" (surface Doppler spread).
3. Named sediment parameter sets (`FineSand`, `MediumSand`, `CoarseSilt`...).

Needs other documents TR 0702 defers to:

4. Derivation of the scattering function and its approximations: Goddard
   (1993), APL-UW 9313.
5. Non-Gaussian (K-distributed) reverberation with the DDD/point-scatterer
   methods: "a separate document", unnamed.
6. `McDanielSurface` (bistatic strength and the wind-speed laws for rmsHeight,
   meanSquareSlope, vertical speed), `GilbertSurface`, `JacksonBottom`: only
   citations.
7. Mitchell-McPherson window function: only the citation (IEEE Trans. AES-17,
   1981).
8. "Noiselet" algorithm: "details will be published separately".

Vague or unspecified in TR 0702:

9. Ray-identity heuristics and the interpolation scheme for tabulated
   eigenrays (Sec. 5.7).
10. Root-finding and partitioning of the iso-time locus into Doppler bins
    (Sec. 11.3.2); sample densities; bin-size selection rules beyond "small
    compared with sonar resolution".
11. How many image orders the straight-line model keeps; any amplitude cutoff.
12. FIR filter lengths; `ResampleSignal` anti-alias equations ("not yet
    incorporated", p. 28).
13. How the near-specular part of the scattering function is identified and
    removed when time spreading is on; AR(1) coefficient; tap spacing; how
    multiple bounces are combined (the author notes the multi-bounce shape is
    wrong, p. 104).
14. How a highlight's scattering strength maps to A_T; how DDD/SF methods treat
    ping-to-ping coherence (not discussed).
15. Aspect dependence of targets exists only via `ExternalTarget`.
16. No ambient-noise spectrum or spatial-coherence model anywhere.

Apparent inconsistencies in the text (all pages were legible in the PDF; no
equation was unrecoverable):

17. (18): printed as `d( floor(z/L + 1/2) L )`; dimensionally it should be
    `floor(z L + 1/2)/L`.
18. (76) and (123) write the scattering strength with arguments
    `(S_Sp, S_Rq)` - slowness at the source and receiver, with p/q roles
    swapped relative to the rest of Sec. 10 - whereas (132) uses the slowness
    vectors at the scatterer `(S_Tp, S_Tq)`. The latter is physically what a
    bistatic scattering strength needs.
19. (131) writes `|U_plqs(f_m, r)|^2` with lower-case r (should be R).
20. (134) has `U`, not `|U|^2`, and no 1/dOmega, while (135) multiplies by
    `dtheta dphi`; (129) has `|U|^2` and 1/dOmega. The text after (129) says
    it is equivalent to "Eq. (122)" where (125) is evidently meant. Use (129)
    with (121) and keep the solid-angle bookkeeping self-consistent.
21. `S_l` denotes a cross-section per unit area in Sec. 6.4.2 but must be its
    square root in (132).
22. (61) writes position updates as `(V . S)(t - t0)` (a scalar times time);
    only its consequence (63) is usable.
23. Doppler-spread rule (p. 101) divides by sin(grazing angle); see 3.6. *Settled by
    derivation: the width goes as sin(g), not 1 / sin(g) (Sec. 16.4). The ASIAEX "less than
    10 Hz" remark does not decide it (Sec. 18.2).*
24. (36): image-vector term `2|S_p - B_p| z_R`; re-derive before relying on it.

---

## 12. Targets with aspect dependence, specular glints, and model-driven seabed scatterers

Added with `python/gpusonar/sim/targets.py`. This is our own section, not notes on TR 0702.

### 12.1 What SST has and what we added

TR 0702's targets (Sec. 4 above) are sets of highlights; aspect dependence exists only through
an external program called once per ping with the distance and direction of the projector and
receiver. `HighlightTarget` is that program built in: at each ping `Scene.evaluate` hands every
target the projector position, the receive reference position, the centre frequency and the
sound speed, and gets back positions, complex amplitudes and velocities for its highlights.
These go into the eigenray pairing as ordinary scatterers, so both engines see them.

| SST | Here |
|---|---|
| highlight: strength, position, velocity in the target frame | `Highlight.position`, `target_strength_db` / `amplitude`; velocity from the target's motion (`velocity`, `turn_rate_deg`, or a `trajectory` function) |
| complex frequency-dependent response | `Highlight.response`, reduced to its value and group delay at the centre frequency (the synthesis kernel has one amplitude and one delay per path) |
| reception-to-transmission delay | `Highlight.delay`: the highlight is moved back along the bisector of the two directions by c tau / |s + r|, with its amplitude scaled so the spreading loss stays that of the true place |
| `commonCenter` | `common_center`: the highlight is put on the bisector through the target's origin at the place with its own two-way path length. (SST does this to save eigenray sets; we still compute one set per highlight, so it only reproduces the behaviour) |
| `randomPosition` | `random_position` |
| group number | not needed: every highlight has its own eigenrays |
| `ExternalTarget` | `AspectTable` and the shapes below; any object with `evaluate(PingContext)` can be put in `Scene.modifiers` |

Not carried: the shape of a frequency response across the band beyond its group delay.

### 12.2 Geometric-acoustics shapes

Source: E. Lavia, J. D. Gonzalez, S. Blanc, "Modelling high-frequency backscattering from a
mesh of curved surfaces using Kirchhoff approximation", arXiv:1805.06945v2 (2018). Taken from
it: the Kirchhoff backscattering integral, Eq. (1); the sphere, Eq. (5); the finite cylinder
without end caps, Eq. (6) (which they cite to Gaunaurd 1985); the flat end cap, Eq. (7);
TS = 20 log10 |f|. Everything else in this subsection is [OURS], derived from Eq. (1) and
checked numerically in `tests/python/test_scene_targets.py`.

With s, r the unit vectors from a surface point to the source and the receiver, B = s + r,
and amplitudes as pressure ratios at 1 m (|amplitude|^2 = differential cross section):

```
sphere, radius a          one highlight at  centre + a n,  n = B / |B| (iterated for finite range)
                          amplitude a / 2                      TS = 10 log10(a^2 / 4)   [limit of Eq. 5]
cylinder side, radius a   glint line at the azimuth of B_perp (B without its part along the axis)
                          per unit length  exp(i pi/4) sqrt(a |B_perp| / (4 lambda))
  whole length L, far     L sqrt(a |B_perp| / (4 lambda)) sinc(k L B_par / 2)
  backscatter, theta_n from broadside:
                          |f|^2 = (a L^2 / 2 lambda) cos(theta_n) sinc^2(k L sin(theta_n))   [large-argument form of Eq. 6]
flat element dA           i (1 / lambda) (n.B / 2) dA,   lit when n.s > 0 and n.r > 0
  rectangle, far          (A / lambda) (n.B / 2) sinc(k B.u / 2) sinc(k B.v / 2)
  disc radius a, far      (pi a^2 / lambda) (n.B / 2) 2 J1(x) / x,  x = k a |B_tangential|    [Eq. 7]
```

Phases are for the simulator's exp(+i w t) signals (the conjugate of Eq. (1)'s convention): a
flat element is a quarter period early and a glint line an eighth, relative to a point
scatterer at the same place. With these a large plate sums to the mirror image of the source.

Remarks:

* The cylinder's obliquity factor is cos(theta_n), which is what Eqs. (1) and (6) give. Tables
  of "simple forms" in the sonar textbooks are often quoted with cos^2; that table could not
  be opened here, so nothing is taken from it.
* Eq. (6) as we read it differs from Eq. (1), integrated numerically, by a quarter turn of
  phase; magnitudes agree to 0.3 % out to 45 degrees from broadside and 1.7 % at 60 degrees
  (k a = 42). Magnitudes are tested against Eq. (6), phases against Eq. (1).
* Bistatic use of the flat element with n.(s + r) / 2 is a symmetrised Kirchhoff form: equal to
  Eq. (1) for backscatter, reciprocal, exact at the specular direction, approximate in the sidelobes.
* `resolved: true` replaces a far-field pattern by a row (cylinder) or grid (plate, disc) of
  coherent points at most half a wavelength apart, each with the directions it sees itself.
  The sum is then right in the near field too (range < L^2 / lambda), which is where imaging
  sonars see man-made objects. One ping sees the Fresnel zone about the foot of the
  perpendicular from the sonar; a synthetic aperture along the cylinder sees the line.
* Validity: k a >> 1. The exact rigid sphere (partial-wave series) is within 0.63 dB of a^2/4
  for k a >= 10, 0.23 dB for k a >= 20, 1.3 dB for k a >= 5 (measured in the tests); a warning
  is given below k a = 10. Not here: creeping waves, elastic resonances, edge diffraction
  beyond what the Kirchhoff edges give, multiple reflections between shapes, the sphere's
  finite-range factor (r - a) / r.
* Directions are straight lines from the sonar. With a multipath provider each bounce pair
  gets the amplitude and glint position of the direct geometry.

### 12.3 Seabed scatterers that follow the bottom model

A `seafloor` item with `model: fluid_sediment` keeps the point-scatterer picture of Sec. 5.2
(one complex Gaussian draw per scatterer, kept from ping to ping) but replaces "sqrt(scattering
strength at the ray angles)" by the model of `sim/bottom.py`: per ping each amplitude is

```
g_i / sqrt(density) * sqrt( sigma(g_in, g_out, azimuth) / (sin g_in sin g_out) )
```

with the angles on the straight lines to the projector and the receive reference point, taken
from the scatterer's own normal where the floor has relief. The scatterers keep the Lambert
flag, so the pairing multiplies each eigenray pair by sqrt(sin g_in sin g_out) of that pair:
the direct pair gets exactly sqrt(sigma); a pair with a boundary bounce gets sigma of the direct
angles times Lambert's ratio for its own angles (an approximation, chosen so that neither
engine had to change). The frequency is the centre frequency: the model's change of strength
across the band is not carried. Mean level: `E|amplitude|^2 x density = sigma`, the quantity
the scattering-function reverberation integrates, so the two agree (tested).

## 13. Boundary models from the open literature (substitutes for TR 9407)

Code: `python/gpusonar/sim/bottom.py`, `python/gpusonar/sim/surface.py`, `configs/sediments.yaml`,
`tools/boundary_models_study.py`. Everything here is opt-in; defaults reproduce earlier results.
TR 9407 was not read. Sources could only be opened from arXiv in this environment (journal
sites, DTIC, IOA and CORE were unreachable), which decided what could be confirmed.

Sources opened, and what was taken from each:

| Source | Taken |
|---|---|
| Olson and Jackson, "Scattering from layered seafloors", arXiv:2009.12420 (2020), eqs. (13)-(16), (20) | Jackson's perturbation-theory factor A for a fluid half-space. Not used in the code: a test checks that our own derivation gives the same number. |
| Olson, "A series approximation to the Kirchhoff integral", arXiv:2106.08287 (2021), eqs. (13), (24) and Sec. II.B | The spectrum/covariance convention and the form of the two-dimensional Kirchhoff integral, with the statement that the Kirchhoff and small-slope approximations both use it with the argument (change of vertical wavenumber)^2. |
| Nennouche, Khalighi, Dowhuszko and Merad, arXiv:2603.23727v2 (2026), eqs. (20)-(23) | The Hall-Novarini bubble population (after Keiffer, Novarini and Norton, JASA 97, 1995): all its constants. |
| Brown, Johnson and Brownstead, arXiv:1809.08459 (2018), Table 1 | Two sediment rows (medium sand, very fine silt), attributed there to TR 9407. |
| Bentrem, Sample, Kalcic and Duncan, arXiv:physics/0607146 (2006), eq. (7), Table II | The centimetre form of the roughness spectrum (for unit conversion) and the definitions of the six parameters. |
| Olson, Lyons and Saebo, arXiv:1601.06078 (2016), Sec. III | Statement that the first-order small-slope cross section is a geoacoustic factor times the Kirchhoff integral (supports the form used; no formula printed). |
| Morency and Stilwell, arXiv:2210.06535 (2022), Sec. II-E and II-F | Empirical surface backscattering law and the SEARAY bottom law with bottom types 1-4 (they quote D. Kraus's lecture notes). No validity range is given there. |

Our own derivations (each tested against an independent calculation):

- Kirchhoff cross section, prefactor |V|^2 / (8 pi) (q^2 / qz)^2 times the Kirchhoff integral.
  Checks: equals perturbation theory for small relief near the specular direction; the
  scattered energy over the hemisphere is 0.88-1.00 of |V|^2 (placeholder sand, 30-300 kHz,
  20-70 degrees incidence; 0.81-1.03 for the "combined" model).
- Structure function constant of a power-law spectrum (checked against the integral).
- The Kirchhoff integral for a power-law structure function, evaluated on a rotated path in
  the complex plane (checked against direct quadrature and the closed form for gamma2 = 3).
- Small-slope form: perturbation factor times the Kirchhoff integral / (2 pi qz^2).
- Composite roughness: the split wavenumber (k x small-scale rms height = 0.5, our choice),
  the large-scale slope variance, the Gaussian slope average with shadowed facets dropped.
- "combined": the small-slope result averaged over the large-scale slopes.
- Resonant-bubble extinction 2 pi^2 a_R^2 n(a_R) / k (checked by integrating damped-bubble
  cross sections over radius) and the per-bounce loss as two slant crossings of the layer.

Not found in any source that could be opened, so not implemented or left as a parameter:

- More named sediments than the two rows now in `configs/sediments.yaml` (medium sand and very
  fine silt, read from Brown, Johnson and Brownstead, arXiv:1809.08459, Table 1, who attribute
  them to TR 9407; the owner allowed arXiv papers that cite it). Olson, Lyons and Saebo,
  arXiv:1601.06078, Table II gives generic rock only as an elastic solid, without roughness or
  volume parameters, so it is not a row. The paper's water values are not printed: 1000 kg/m3
  and 1500 m/s are assumed for the ratios.
- The handbook's formulas for surface bubble loss, surface and bottom backscattering strength,
  surface forward loss and the Kirchhoff/composite interpolation: no arXiv paper found prints
  them (Bentrem et al., arXiv:physics/0607146, print only the six-parameter list and the
  spectrum convention, their eq. (7), used for `bottom.w2_from_cm4`).
- Hamilton's regressions, Mourad and Jackson (1989), Williams and Jackson (1998), Jackson,
  Winebrenner and Ishimaru (1986), Thorsos (1988, 1990), Hall (1989), Ainslie (2005), Dahl's
  papers and McDaniel's review: none could be opened (pubs.aip.org, ieeexplore.ieee.org and
  core.ac.uk answer automated requests with a browser challenge).
- The published interpolation between the Kirchhoff and composite results, and the published
  choice of the composite split: ours are documented in `bottom.py`.
- A physical rough-surface plus bubble-layer backscattering model: only the empirical law is in.

Known weakness: the bubble loss law is 2 to 8 times the dB of the law arlpy implements
(`arlpy.uwa.bubble_surface_loss`, used as an oracle only) between 10 and 100 kHz at 10 m/s,
and the gap grows with frequency. It is an upper estimate above about 10 kHz.

## 14. Reverberation geometry and statistics as implemented

What `sim/reverb.py` and `sim/reverb_bistatic.py` do, against Sec. 5 above. [OURS]
throughout: the derivations here are ours; TR 0702 supplies eqs. (63), (68), (71), (89),
(96), (103) and the structure of Sec. 11.2-11.3, nothing else.

### 14.1 What the default (monostatic) generator models

Established by reading the code and by tests (`tests/python/test_reverb_bistatic.py`):

| Item | Modelled by default | How |
|---|---|---|
| Transmit beam pattern | yes | power pattern at each path's launch direction at the sonar, turned to the sector's azimuth |
| Receive element pattern | yes, the first element's, for all | as above, on the return path (reciprocity) |
| Ray elevation in the patterns | at the platform (launch), not at the boundary | verified with a surface-bounce path that leaves upwards and arrives at the bottom travelling down |
| Scattering strength angles | grazing angles at the boundary of each leg | backscatter only (no bistatic azimuth) |
| Receiver motion | Doppler `2 (v_h . u_az) f / c` per sector | horizontal velocity, as if every path were horizontal: no `cos(elevation)`, no vertical velocity |
| Transmitter position and velocity | no | one provider call from the array centre serves both legs |
| Arrival elevation in the steering | no | plane wave from the sector's azimuth in the horizontal plane |
| Boundary | flat, uniform | |
| Statistics | Gaussian | |

So of the three limits in the old "Simulator limits" line, "stationary platform" and
"omnidirectional projector" were out of date (D-43 lifted them), and "monostatic" was
missing.

Two properties of the default that the work turned up and that are left as they are (old
results must not change):

- `reverberation_power` puts each ring into its nearest time bin, which leaves a ripple
  between neighbouring bins of the expected power (measured -1.4 to +1.7 dB on a 1 ms
  grid). The mean over a few bins is right (0.006 dB).
- The generator's delays and its interpolation to the record's rate are ideal (sinc)
  operations at a reduced rate. Where the pulse's spectrum is not zero at that rate's
  Nyquist frequency (CW: sidelobes about 28 dB down) one sample's response falls only as
  1/distance, so the strong start of a record leaks into its weak far end, forwards and
  (the series being periodic) backwards round the period. Negligible while the power
  falls by less than about 50 dB across the record; measured on a 30 m - 3 km record
  (100 dB): +7 dB over 50-75 % of the record, +25 dB over 90-97 %. `band_taper: true`
  removes it (within 0.2 dB to the end of that record); see 14.5.

### 14.2 Bistatic geometry (`geometry: bistatic`)

The directional form of the scattering function, (129) with the receiver factored out as
in (121): cells of the boundary are binned by two-way time and by arrival direction at
the receiver.

```
cells:      polar grid about the receiver's ground point: azimuth sectors phi_m (width dphi)
            x rings r_i (width dr = c dt / 4)
per cell, receive eigenray p, transmit eigenray q:
  T      = T_q + T_p
  dP     = |a_q|^2 |a_p|^2 S(g_q, g_p, phi_b) B_s(u_Sq) B_r(u_Rp) r_i dr dphi
  phi_b  = angle between the horizontal directions of the incident and scattered waves
           (pi = backscatter, 0 = forward)
  f_d    = f / c (V_S . u_Sq + V_R . u_Rp)
  u_Sq, u_Rp = unit vectors along the rays leaving the transmitter / receiver towards the cell
```

`dP` is shared linearly between the two time bins next to `T`. Binning by `T` is the
integral over SST's iso-time locus `A_knu` of (131): for straight rays the cells of one
bin lie between two ellipses (the boundary plane cut by spheroids of constant
`R_1 + R_2` whose foci are the two platforms; circles when the platforms coincide).
Nothing in the code constructs the ellipse. `g` are the grazing angles at the boundary
(the scatterer-end slowness vectors of (132); cf. inconsistency 18 in Sec. 11).

Doppler: (71) with slowness vectors `S = -u / c` (gradient of the travel time with respect
to the platform's position) is `Gamma = (1 + V_R . u_Rp / c) / (1 - V_S . u_Sq / c)`, and
`f_d = (Gamma - 1) f` to first order in `v / c`. Scatterer Doppler `gamma` (72) is not
modelled (the boundary is fixed in the world).

Stratified-medium shortcut: with a flat boundary in a horizontally stratified medium the
eigenrays of a leg depend only on the horizontal distance from its platform. The receive
leg is one provider call along the rings; the transmit leg is one call along a table of
distances half a ring apart, looked up by nearest entry with its delay corrected by
`(d - d_table) x horizontal slowness` (exact to first order by Snell's law). Cost is
independent of the number of sectors in provider calls and `sectors x rings x K_q x K_p`
in arithmetic.

Components: contributions are collected into (azimuth sector, bin of the receive ray's
elevation at the receiver, bin of total Doppler shift) - see 17.5; the first version had
no Doppler bin, which was a defect. A component has its power against time, the
power-weighted mean of its `u_Rp` (its arrival direction, with elevation) and the
power-weighted mean of its `f_d`. The generator draws one complex Gaussian sequence per
component - the `g_s` of (127)-(128). Within an elevation bin of 20 degrees (the default,
9 bins) the direction is that of the bin's mean ray; raise `n_elevation` close to the
sonar if that matters. Components more than 120 dB below the strongest are dropped.

The generator (NumPy and C++) then delays each component to each hydrophone by
`offset . u / c` with the full 3-D `u`, and shifts it by its `f_d`, pulse included, as
before.

Limits that remain: flat boundary, stratified medium, eigenray pairs incoherent
(`delta_pp' delta_qq'` of (76)), one Doppler shift for the whole pulse of each echo (no
range walk, the "short pulse" assumption of Sec. 11.2.1), the first receive element's
pattern for all, and the sector must be narrow compared with the transmitter's
height over the boundary where the transmitter is far from the receiver
(`azimuth_subsamples`). Chapman-Harris is a backscatter fit and is used at the mean
grazing angle whatever the bistatic azimuth.

### 14.3 Range-varying boundary (`strength_map`)

A factor on the scattering strength as a function of world position, applied per cell.
The boundary itself stays flat (bathymetry would change the eigenrays). From a scenario:
a table of dB against range, optionally azimuth, about a fixed origin.

### 14.4 K-distributed reverberation (`statistics: {model: k, shape}`)

TR 0702 says SST's point-scatterer and DDD methods can make K-distributed reverberation
and gives no method (Sec. 8 above). Ours is the compound representation: each resolution
cell's power is multiplied by a gamma variate `tau` of unit mean and shape `nu`, then the
complex Gaussian speckle is drawn as usual. Intensity `I = tau x E` (E exponential):

```
E{tau^n} = prod_{k<n} (1 + k/nu)         (gamma, unit mean)
E{I}     = mean power (unchanged)
E{I^2} / E{I}^2 = 2 (1 + 1/nu)
E{I^3} / E{I}^3 = 6 (1 + 1/nu)(1 + 2/nu)
```

(elementary moments; `nu -> infinity` is Rayleigh). A cell is `cell_time` long (default
1 / pulse bandwidth) and one azimuth cell wide (default wavelength / array extent: about
two of the generator's sectors); all components of one azimuth cell share the texture.
A hydrophone or beam that sums N cells of equal power sees `2 (1 + 1/(N nu))`, so an
omnidirectional hydrophone is close to Rayleigh and a beam is not: the statistics after
beamforming come out of the array structure, as the levels do. `shape` has no default
(a property of the seabed to be measured or assumed). What is not modelled: correlation
of the texture between pings (each ping draws its own) and between neighbouring cells.

**The shape from a density (`scatterers_per_m2`, `patches_per_m2`; sonar-6fc).** What is
unknown comes first: the density. Abraham and Lyons (2002, read; REFERENCES section 6) give
the rule but no density for any seabed; they estimate the shape from reverberation data. So
the key has no default and a value must come from data (a measured shape divided by the
cell's area, for instance). The rule: n scatterers of exponentially distributed amplitude in
a cell give a K envelope of shape alpha = n / 2 (their eq. 15); n patches of exponentially
distributed area give alpha = n (eq. 19). Our `shape` is their alpha: their eq. (12)-(13) is
the product of a gamma variate of shape alpha (unit scale) and complex Gaussian speckle of
variance lambda, which is ours after dividing the gamma by its mean alpha; and their envelope
moments, eq. (28), E{Y^i} = lambda^(i/2) Gamma(1 + i/2) Gamma(alpha + i/2) / Gamma(alpha), equal ours,
E{tau^(i/2)} E{|Z|^i} with tau of shape nu, at alpha = nu for every i, so
E{I^2} / E{I}^2 = 2 (1 + 1/alpha). The test also matches the exceedance of their eq. (7) with
the integral over our texture to 1e-5 down to 1e-8, and recomputes their Sec. IV example:
alpha = 10 puts the threshold at Pfa 1e-6 1.685 dB above Rayleigh's (printed 1.7), alpha = 5
2.694 dB (printed 2.7).

The cell (ours): the generator lays the texture in cells of `cell_time` (as realised: whole
samples at the reduced rate) from the first two-way time, and over groups of whole azimuth
sectors of width w (`cell_azimuth_deg`, by default wavelength / array extent of the receive
hydrophones at the centre frequency: the receive array's nominal beamwidth, rounded to whole
sectors). Cell k spans two-way times t1..t2, slant ranges r = c t / 2. For straight paths from
a point a height h above a flat boundary, the horizontal range is R with R^2 = r^2 - h^2, so
the boundary between r1 and r2 inside w has area

```
A = w / 2 (R2^2 - R1^2) = w / 2 (r2^2 - r1^2) = w r_mid (r2 - r1)      (r1 >= h)
```

with R clipped to 0 below h. The grazing angle cancels exactly: the ring is 1 / cos(grazing)
wider than the slant extent c dt / 2 and cos(grazing) nearer than r. Each cell's shape is then
density x A / 2 (scatterers) or density x A (patches): proportional to range, to the azimuth
width and to the cell's length (1 / bandwidth by default), and the same in every azimuth
group. Cells the boundary is not yet reached in (A = 0) have no texture (exactly 1). The mean
power is unchanged in every cell (unit-mean gamma), and large shapes (thousands, up to 1e12
tested) go smoothly to 1; small shapes are drawn as they are (unit mean; a shape of 0.02 has
variance 50).

Below one scatterer (shape 0.5) or one patch (shape 1) per cell the K distribution is still a
distribution, but the paper's model, a fixed count n per cell, no longer describes the cell
(most such cells would hold none): the simulator warns when a cell with power beyond the first
one the boundary is reached in holds fewer than one, and makes the reverberation anyway.

Limits: the height h is the array centre's above the flat `boundary_z`; with `bathymetry`
(relief) h = 0 is used, the slant-range form. Refraction and bottom-bounce paths do not enter
the area (a cell is defined by straight-path two-way time). `geometry: bistatic` uses the same
monostatic-equivalent range c t / 2 from the receiver: for separated platforms the true
iso-time locus is an ellipse and the cell is larger by roughly 1 / cos of half the bistatic
angle, which this does not take in. The texture is independent from cell to cell and from
ping to ping: the paper states that the second-order statistics (correlation between samples
at two times) are not established, so no correlation model is adopted. Multipath makes the
envelope more Rayleigh-like (their Sec. III-C); here every path pair arriving in a cell shares
that cell's texture.

### 14.5 Leakage along a long record (`band_taper`, `guard`)

See 14.1. `band_taper` multiplies the pulse's spectrum by 1 below a quarter of the
generation rate and a raised cosine from there to Nyquist (the rate is at least four times
the pulse's band, so the band is untouched), which makes the spectrum continuous round
Nyquist and the delay/interpolation responses fall fast; two pulse lengths of silence
are generated after the record to keep the remaining wrap-round out. Both engines get
the same tapered pulse.

### 14.6 Property tests (Sec. 9 above)

`tests/python/test_reverb_statistics.py`, on both engines:

| Property | Set-up | Closed form | Measured |
|---|---|---|---|
| (89) | power Gaussian in two-way time, sigma_T = 20 ms | sigma_f = 7.958 Hz | 7.971 Hz |
| (96) | sonar at 3 m/s, arrivals Gaussian in azimuth about broadside (0.1 rad), 10 kHz tone of 0.5 s (its own autocorrelation divided out) | sigma_t = 39.79 ms | 40.06 ms |
| (103) | arrivals Gaussian in sin(theta), sigma = 0.08, 10 kHz | sigma_d = 0.2984 m | 0.2973 m |
| replica-correlation gain | LFM, 20 ms, 0.5 / 1 / 2 / 4 kHz | 3.01 dB per octave | 2.73, 3.01, 3.13 dB |
| K moments | nu = 0.5, 2, 10 | m2 = 6, 3, 2.2 | 6.01, 3.00, 2.19 |

## 15. Sampled transmit waveforms and the variable-delay resampler

Code: `python/gpusonar/sim/waveform.py`, `libs/dsp/src/pulse_compression.cpp`
(`build_arbitrary`). Tests: `tests/python/test_sampled_waveform.py`. Source for
everything quoted from SST: TR 0702 Sec. 4.1 (eqs. 3-6), Sec. 4.3 (the script on
p. 20-21), Sec. 4.5.1 (eqs. 14-19), Sec. 4.6.3 (eq. 29), Sec. 8.3 and 9.1 (eqs. 63, 68).

### 15.1 What is carried where

A sampled pulse is a complex envelope b about a carrier F on a uniform grid:
`q(u) = A Re{ b(u) exp(2 pi i F u) }`, `0 <= u <= T` (TR 0702 eq. 4 without the
sqrt(2): our envelopes have unit peak, and A is the record's `amplitude`).

The record format already has what processing needs: `waveform_type = 3`
(arbitrary), `replica` (real samples) and `replica_sample_rate`. We store the real
passband pulse `Re{ b exp(2 pi i F u) }` there, sampled at 4 x the top of the band
unless configured, with `f_start`, `f_stop` = the occupied band (their mean is F)
and `duration` = T. Pulse compression recovers b from it (analytic signal, mixed
down by F), resamples it at the record's rate with the hypothesis' time scale, and
correlates as for the analytic pulses. Nothing was added to the record.

Cost: the replica is stored with every ping (as float64 in the file). A 1 s pulse
at 6-10 kHz is 40 001 samples; a 1 ms pulse at 270-330 kHz is 1 321.

### 15.2 Echo synthesis

Per (path, channel) the kernel's own steps are kept (Sec. 10, "Analytic-pulse
synthesis vs SST"): exact arrival times of the start, middle and end of the pulse
from the moving transmitter, scatterer and receiver; emission time u(t) as the
quadratic through them. The closed-form pulse is replaced by

```
echo(t) = a g(f) b(u(t)) exp(2 pi i F u(t))          [x exp(-2 pi i f_mix t) for a baseband record]
```

with b interpolated between its samples. This is SST's `VarDelay` (14) with
`T_c(t) = t - u(t)`, except that the delay comes from the exact solve at every
output sample instead of linear interpolation of a slow delay track. Doppler is
therefore a time scaling of envelope and carrier together, with range curvature
across the pulse; the first-order factor (63)/(68) is reproduced (15.4) but not
used.

Interpolation is on the envelope, not on the passband signal, so the band
occupancy `alpha` of (19) is `bandwidth / envelope rate` (0.125 by default) whatever
the carrier. For speed it is done in two stages: the windowed sinc once, onto a
table 8 times finer than the envelope's samples (`table_factor`), then a four-point
cubic in that table at each output sample. (SST tabulates the coefficients on a grid
512 times finer and takes the nearest, eq. 18; we tabulate the signal.) The measured
errors in 15.4 are with both stages.

Loss that depends on frequency (absorption, both beam patterns) cannot use an
instantaneous frequency, which a sampled pulse does not have. It is applied in
frequency: the envelope is split by triangular weights about `gain_knots + 1`
frequencies across the band, and each part takes the gain at its frequency, i.e. the
gain is piecewise linear in frequency. This is the first step towards SST's FIR
filter per path (22); the split parts are cut to the pulse's own length, so a gain
that changes fast across the band smears less than it should at the pulse's ends.

*Superseded by Sec. 19.1: the C++ kernel now synthesises sampled pulses on the host (CUDA
written, to be verified on a GPU), and NumPy is the reference engine. The paragraph below
is as first written, and its timings are of the NumPy path.*

Engines: NumPy only. The C++/CUDA kernel (`synth_kernel.hpp`, mirrored in the
`.cu` file) evaluates closed-form pulses in a plain-data struct with no array
argument; giving it a replica means a device buffer and a second kernel. A sampled
pulse therefore always runs in NumPy for the echo step (path pairing and
reverberation keep their C++ engine): the record's `extra_json` says
`{"gpusonar": {"echo_synthesis": "numpy_sampled"}}` and asking for the CUDA backend
or `engine="cpp"` warns once. Measured on 4 cores, 300 scatterers x 128 channels,
1 ms pulse at 1.25 MHz (fls_300k): 18-22 s with beam patterns, 7 s omnidirectional,
against 1.0-1.4 s for the analytic kernel. Scenes of 10^5 scatterers are out of
reach until the kernel is ported.

One difference from the analytic kernel, found by the closed-form Doppler test: the
kernel's fixed-point solve for the arrival time stops after 3 passes on the outbound
leg and 4 on the return leg, each pass gaining a factor v/c. The residual is
`(R/c) (v/c)^3`: 0.14 us for a target closing at 12 m/s at 400 m, which is 0.007 rad
at 8 kHz (measured 6.9e-3 relative error against the closed form with 3 passes,
7e-8 with 8). The NumPy sampled path runs 8 passes. The analytic kernel was left
as it is. (Sec. 19.2: the pass counts are now parameters; the analytic default is still
(3, 4).)

### 15.3 The interpolator against TR 0702 eq. 19

TR 0702 gives the rule `M >= 2.5 / (1 - alpha)` for the Hann-windowed sinc (17) and
no error figure. Measured response of our implementation of (17) as a continuous
filter (worst of: departure from 1 inside `alpha` x Nyquist; leakage of the spectral
copies outside):

| alpha | M at the rule | error at the rule | at half the rule | at twice the rule |
| --- | --- | --- | --- | --- |
| 0.5 | 6 (SST's default order 5) | -39.1 dB | -11.2 dB (M = 2) | -54.3 dB |
| 0.8 | 14 | -33.6 dB | -13.1 dB (M = 6) | -54.6 dB |
| 0.9 | 26 | -29.4 dB | -13.1 dB (M = 12) | -54.6 dB |

So the rule marks where the transition band clears the signal, and buys 2-3 %
amplitude error; the Hann window's error then falls slowly with M (its sidelobes
decay as 1/f^3). We keep (17) as an option (`interp_window: hann`) and use a
16-tap Kaiser window (shape 12, chosen by this measurement, not from a table) by
default: -117 dB for alpha up to 0.5, where 16-tap Hann gives -62 dB.

The `VarDelay` script (two tones at 220 and 440 Hz, 8 kHz real samples, delay
1.00 ... 2.0 s over 5 s) is reproduced: the tones sit at `n f1 (1 - dT/dt)` in each
second within 0.02 Hz, and the output equals `x(t - T(t))` within 5.1e-3 (M = 6,
Hann), 2.3e-4 (M = 16, Hann), 1.5e-6 (M = 16, Kaiser).

### 15.4 Measured accuracy

- Sampled copy of a Tukey-tapered LFM against the analytic LFM (sas_auv,
  omnidirectional, no absorption, moving sonar), relative rms difference of the
  record against envelope samples per cycle of bandwidth: 1.25 -> 1.3e-2,
  2 -> 1.9e-4, 4 -> 1.4e-5, 8 -> 2.4e-6, 16 -> 1.3e-6 (float32 floor). Same for
  real and baseband records.
- The same with rectangular edges: 1.5e-2 (2x), 5.6e-3 (8x), confined to the
  pulse's ends. A pulse that switches on abruptly is not band-limited, so its
  samples do not define it there; taper the pulse or accept the edge error.
- With beam patterns and absorption (frequency-domain gain against gain at the
  instantaneous frequency): 8.5e-4 (fls_300k, BT = 60), 4.8e-4 (flank_32, BT = 80).
- Every tier (free field, image source with and without time spread and
  reverberation, sediment, Bellhop eigenrays), real and baseband: 1.3e-4 to 3.8e-4.
- Doppler: echo of a two-burst wideband pulse from a target (or sonar) closing at
  12 m/s equals `b(Gamma (t - t0)) exp(2 pi i F Gamma (t - t0))` with Gamma from
  (68) within 7e-8; a carrier-shift-only model is wrong by 0.88.
- Phase codes at 7.5 kchip/s in a 15 kHz band: compressed peak at the two-way delay
  within 1/50 chip and equal to the echo amplitude within 2 %; peak sidelobe
  -21.9 dB (Barker 13; -22.3 dB for ideal chips) and -22.3 dB (127-chip
  m-sequence; -23.0 dB for ideal chips). The difference is the band limit.

### 15.5 Limits

- Emission time is a quadratic through three exact arrival times, as in the
  analytic kernel: exact for constant radial speed, second order in acceleration
  and bearing change across the pulse.
- A CW pulse (zero stated bandwidth) cannot be given as `sampled`; use `cw`.
- The live C++ chain and the WAV sidecar reader do not build sampled pulses
  themselves; they process a record that carries one.


## 16. Angle and frequency spreading of bounce paths; several bounces [OURS]

TR 0702 has time spreading and a slow (AR(1)) change of the tail; it has no angle spreading
("not yet", p. 93), takes its frequency-spread width from TR 9407 (not read), and says its
multi-bounce shape is wrong (p. 104). What follows is derived here, not transcribed; it is
implemented in `sim/timespread.py` (options; all were off by default when this was written,
and `multi_bounce: convolve` of 16.5 has since become the default) and tested in
`tests/python/test_timespread.py`. The only things used from TR 0702 are the relations
between a Gaussian spread and a coherence width, eqs. (89), (96), (103), and the remark that
the ASIAEX surface Doppler spread was "less than 10 Hz" (p. 98), as a sanity bound.

Model: the boundary is a set of tilted mirrors (facets) with Gaussian slopes, variance mss/2
per axis; one bounce, r1 and r2 the path lengths before and after it, g the grazing angle,
q = 1/r1 + 1/r2, (x, y) a facet's position from the specular point, x along the path.

**16.1 Geometry.** To second order the path through (x, y) is longer by
`c tau = (x^2 sin^2 g + y^2) q / 2`: ellipses of equal delay, 1/sin g longer along the path.
With `(x sin g, y) = rho (cos psi, sin psi)`, `rho^2 = 2 c tau / q`: seen along the ray the
ellipse is a circle. The facet reflects source to receiver if its slopes are
`s_x = x sin(g) q / 2`, `s_y = y q / (2 sin g)`.

**16.2 Angles.** The tap of excess delay tau leaves the source `rho / r1` from the specular
ray and arrives `rho / r2` from it: `angle^2 = 2 c tau r2 / (r1 (r1 + r2))` at the source.
Its place on the circle has density `exp(kappa cos 2 psi)`,
`kappa = (c tau q / 4 mss)(1 / sin^2 g - 1)`. With Gaussian slopes the angles are Gaussian,
variance `2 mss (r2 / (r1 + r2))^2` in the vertical plane and `sin^2 g` times that across it.
By (103) the tail's coherence across an aperture is `exp(-d^2 / 2 sigma_d^2)`,
`sigma_d = lambda / (2 pi sigma_S)`, sigma_S the rms spread of the direction cosine along
the aperture: `sigma_v cos g` for a vertical line, `sigma_v sin g` for a horizontal line
across the path. Limit: the turn must stay inside the grazing angle (the code stops a turn
towards the horizontal at 0.9 of the ray's elevation); shadowing is not modelled.

**16.3 Decay time.** The same slopes give `tau = 2 (s_x^2 + s_y^2 sin^2 g) / (c q)`, mean
`T = mss (1 + sin^2 g) t1 t2 / (t1 + t2)`. TR 0702's rule of thumb is `2 mss t1 t2 / (t1 + t2)`:
the same at normal incidence, twice as long at low grazing angles. The shape is not an
exponential but `exp(-tau (a + b) / 4ab) I0(tau (a - b) / 4ab) / (2 sqrt(ab))`,
`a = mss t1 t2 / (t1 + t2)`, `b = a sin^2 g`. (`decay_time: facet`, the default since D-85; `rule_of_thumb` stays selectable. Correction 2026-11-20: this note said "the default stays the rule", which was true before D-85.)

**16.4 Frequency.** A boundary displaced by zeta changes the path by `-2 zeta sin g`: the
field is `exp(i chi zeta(t))`, `chi = 2 k sin g`. For Gaussian zeta of rms height h and rms
speed v the scattered part has correlation `exp(-Gamma^2)(exp(chi^2 C(t)) - 1)`, C the
height correlation, so the second moment of its spectrum is exactly
`sigma_nu^2 = (2 f v sin g / c)^2 / (1 - exp(-Gamma^2))`. Rough boundary: Gaussian spectrum
of rms width `2 f v sin(g) / c` (the pdf of the facets' vertical speeds). Nearly flat:
the wave spectrum, width `v / (2 pi h)`. This settles note 23: the width goes as sin g, not
1/sin g. By (96) the coherence time is `1 / (2 pi sigma_nu)`. A bottom has v = 0. The
sonar's own motion: with angle spreading each tap has its own direction, hence its own
Doppler shift from the kernel, `f V . (u_tap - u_specular) / c`, rms about
`2 f V sin(g) sqrt(mss / 2) r2 / ((r1 + r2) c)` for motion along the path - TR 0702's
"speed of the reflection point x sqrt(mss)" within a factor `r2 / (sqrt(2) (r1 + r2))`.
What the default process gives: one gain per tap per ping (ping-to-ping fading only, no
spread within a ping). `doppler: true` makes each sinusoid of the process a ray with its
own shift, so the same process runs through the ping. (Such rays were first paired in NumPy
with a warning; the C++ pairing now carries a range rate per ray, Sec. 19.3.)

**16.5 Several bounces.** Each bounce: `c_i delta(tau) + (1 - c_i) h_i(tau)`. A path: the
convolution. Tail energy `1 - prod c_i`, mean `sum (1 - c_i) T_i / (1 - prod c_i)`; two
rough bounces of equal T give `tau exp(-tau / T) / T^2` (zero at the specular delay, peak
at T), where one exponential of mean 2T is highest at once and has rms spread 2T instead of
sqrt(2) T: the error TR 0702's author describes. (`multi_bounce: convolve`, the default
since the owner's decision recorded in `DECISIONS.md` D-82; `sum` is the earlier rule.)

---

## 17. Reverberation: moving scatterers and a seabed with relief

Additions to Sec. 14 (`sim/reverb_bistatic.py`, `sim/reverb_relief.py`, `sim/bathymetry.py`).
[OURS] except where an equation of TR 0702 is cited.

### 17.1 What TR 0702 says about scatterer Doppler

Only this: the total Doppler of an echo is split into the platform factor `Gamma_pq` (71)
and the scatterer factor `gamma` (72); every cell "is assumed to contain scatterers with a
distribution of Doppler shifts due to scatterer motion" (for the bottom, a delta function at
1; Sec. 10.1); `gamma` is "treated as an independent variable" and the scattering strength
is a density in it (76); the scattering function is accumulated in bins of the total
Doppler `gamma Gamma_pq` (129). No distribution of `gamma` is given for the surface or the
volume (those are in the boundary models SST takes from TR 9407, which we do not have).
Sec. 10.6.2 gives the consequence we test against: a Gaussian spread `sigma_gamma` gives a
coherence time `1 / (2 pi f sigma_gamma)` (96).

### 17.2 Mean velocity

With `a_q`, `a_p` the unit vectors of travel, on arrival at the cell, of the transmit-leg
ray and of the receive-leg ray traced from the receiver, the slowness vectors at the
scatterer are `S_Tq = a_q / c`, `S_Tp = a_p / c` (gradients of each leg's travel time with
respect to the scatterer's position), and (72) is

```
gamma = (1 - V_T . a_q / c) / (1 + V_T . a_p / c)  ~  1 - V_T . k / c,     k = a_q + a_p
f_d(cell) = f / c (V_S . u_Sq + V_R . u_Rp)  -  f / c V_T . k
```

`|k| = 2` for backscatter; `k -> 0` in forward scatter (a scatterer moving along the
bisector plane does not change the path length). `a` comes from the eigenrays' arrival
vectors, so for a bounce path it is the direction at the cell, not at the platform.

### 17.3 Random motion: the Doppler spread

One scatterer with velocity `v` is shifted by `-f/c v . k`. For independent Gaussian
velocity components the shift is Gaussian with

```
sigma_f^2 = (f / c)^2 ( sigma_iso^2 |k|^2 + sigma_w^2 k_z^2 )
```

`sigma_iso` per axis for scatterers moving in every direction (`rms_speed`), `sigma_w` for
a boundary that heaves (`rms_vertical_speed`): for backscatter at grazing angle g,
`2 f sigma_iso / c` and `2 f sigma_w sin(g) / c`. `wind_speed` sets `sigma_w` to the rms
vertical speed of a fully developed sea (`acoustics.rms_surface_vertical_speed`, the second
moment of the Pierson-Moskowitz spectrum; the same quantity `sim/timespread.py` uses for
the forward bounce). Treating the heaving surface's scatterers as moving with the surface's
vertical velocity, Gaussian, ignores the orbital (horizontal) part and the Bragg waves'
own phase speed; the horizontal part can be put into `rms_speed`, a mean drift into
`velocity`. Nothing is derived from the wind beyond `sigma_w`.

Each component (direction) carries the power-weighted mean of `sigma_f^2` over its cells.
The generator then splits the component into Doppler lines: offsets uniform over +-3
`sigma_f`, Gaussian weights summing to 1, rescaled to the exact variance, spaced no wider
than half the pulse's resolution `1 / T` (at most 41 lines; one line when
`sigma_f T < 0.05`). Each line is an independent Gaussian sequence with its own shifted
pulse - the bins in total Doppler of (129). Within a pulse length the lines are not
resolvable, so the echo of a cell decorrelates as `exp(-2 pi^2 sigma_f^2 s^2)` (measured:
39.4 ms against 39.8 ms from (96) at 0.3 m/s, 10 kHz). Mean power is unchanged (the
weights sum to 1). Cost: the number of sequences multiplies by the number of lines.

Ping to ping: this method draws new sequences every ping, so reverberation is already
uncorrelated between pings - right whenever the ping interval is much longer than
`1 / (2 pi sigma_f)` (always, with any surface motion), wrong for a still sonar over a
still bottom (Sec. 14, unchanged).

Platform pitch and heave in the component path: each platform's full velocity vector is
projected on its own ray (elevation included), so heave is exact to first order (tested:
2 f w / c straight down). Attitude rates (the velocity of an element from pitch rate times
its lever arm) are not included: the array is given its centre's velocity.

### 17.4 Relief

`bathymetry` gives the bed's depth D(x, y) (NED, positive down). Per cell of the polar grid
about the receiver's ground point:

```
R        = (x, y, D)
n        = (D_x, D_y, -1) / sqrt(1 + |grad D|^2)          into the water
dA       = r dr dphi sqrt(1 + |grad D|^2)                 the facet's area
sin g_q  = -a_q . n,   sin g_p = -a_p . n                 <= 0: the facet is turned away, no return
phi_b    = angle between a_q - (a_q.n) n  and  -a_p + (a_p.n) n     (pi = backscatter)
T        = T_q(R) + T_p(R),   direction and Doppler from the rays to R itself
```

Eigenrays are asked for at the cell itself, so nothing is interpolated along range (the
flat code's one-line-per-leg shortcut does not apply). Exactness:

- direct and surface-reflected legs: exact for straight rays (the cell is a point in the
  water, and its images in the surface are images);
- legs with a bottom bounce before the cell: not image paths over relief. Left out by
  default. What that costs was measured on a level bed (60 m of water, sonar 40 m above
  the bed, bottom reflection 0.5, surface -1): 0.0 / 0.2 / 2.5 / 4.5 / 5.2 / 5.5 / 5.6 dB
  in successive 40 ms from 60 ms, that is negligible to about two water depths of range
  and 5-6 dB beyond five. `bottom_bounces: flat` keeps them with the earlier bounces off
  the provider's flat bottom and the last incidence on the true facet (exact on a level
  bed; elsewhere an approximation whose error is not bounded here);
- with Bellhop the bathymetry goes to Bellhop: one run per azimuth sector with the depth
  profile of that bearing (N x 2-D), all paths kept, blocking by relief done by the ray
  trace. Transmitter and receiver must be over one ground point. No out-of-plane
  reflection.

Shadowing: a cell is hidden from a path's apparent origin (the platform, or its image in
the surface) when nearer relief stands above the straight line to it. Computed from a
horizon about each origin: along 4 x sectors (at least 720) bearings, the running minimum
of (depth below the origin) / distance; a cell is visible when its own slope is not above
that minimum at two steps nearer, plus 0.2 mrad. Straight rays only.

A level bed given as bathymetry runs the flat code (bit-identical). A uniformly sloping
plane is the flat problem seen from the perpendicular heights: the two codes agree to
0.001 dB.

Point-scatterer scenes take the same description (`seafloor` item): the bed's depth and
normals, amplitudes scaled by `(1 + |grad D|^2)^(1/4)` because the points are drawn per
horizontal area. The scene does not shadow its seabed with its own relief (only listed
occluders shadow), so the two methods differ behind a ridge.


### 17.5 Components are keyed by Doppler (a defect, found by the validation package, fixed)

The first bistatic generator keyed a component by (azimuth sector, elevation bin of the
receive ray) and gave it one shift: the power-weighted mean over every pair of eigenrays
and every two-way time arriving from there. Two things were wrong with that. Pairs with
different transmit legs but the same receive ray shared a shift, where (71) gives each pair
pq its own; and the shift of one direction could not change along the record, so late,
strong returns set the Doppler of early, weak ones (seen in the Pursuit example: the
return by the bottom changed the spectrogram 0.4 s before it could arrive).

The key now has a third part, a bin of the total Doppler shift `(Gamma_pq gamma - 1) f`:
SST's "perception space" of travel time, total Doppler and direction (eq. 129, with its
remark that typically few Doppler cells per direction are occupied). Bins span every shift
the speeds allow (`f v / c` per platform, `2 f v / c` for scatterers, 5 % margin), are half
the pulse's resolution wide (`0.5 / T`: within a pulse nothing finer can be told apart),
and are at most `max_doppler_bins` (128) in number, widening if they would be more. A
contribution goes to the component of its own shift whatever its paths and its time; the
component's shift is the power-weighted mean of what fell into it, so it is within half a
bin of every contribution. Only components that receive power exist (they are found as
the cells are summed), so the count is a few per direction, not `bins x directions`. With
nothing moving there is one bin and the components, and the realisations, are as before;
with motion the realisations of the bistatic mode changed.

The random-motion lines of 17.3 are applied to each component afterwards, as before.

The shift itself is now the factor of (71) and (72), not its first-order form (the
difference is `(v / c)^2 f` in size: 3.6 Hz at 20 m/s and 9.9 kHz).

## 18. Validation against TR 0702 itself [OURS]

`tools/sst_validation.py` runs what TR 0702 lets one check, end to end (scenario -> records ->
processing), and prints one row per statement of the report (`--out`; the last run is in
`docs/figures/sst_validation.md`, its numbers in `sst_validation.json`). The assertions are in
`tests/python/test_sst_validation.py`. Sec. 9 above said what is reproducible; this is what was
done about it. Nothing here was taken from any source but TR 0702's own text.

**18.1 The two worked examples.** `configs/scenarios/sst_pursuit.yaml` (Sec. 3, Fig. 2) and
`sst_asiaex.yaml` (Sec. 12.2-12.6, Figs. 3-5) hold what the report prints and, under
`unconfirmed`, values of our own for everything it does not (speeds, depths, beams, pulse
length, wind and levels for Pursuit; depths, sound speed, ping interval, sediment and the wind
laws for ASIAEX). They are not numerical reproductions of the figures. Each feature the text
describes is checked as a number against a closed form for our geometry:

- Pursuit: the ridge of the surface reverberation against `f (1 + 2 v cos(angle) / c)` at each
  time (the "hook"); the onset of the first return with a bottom-reflected leg at `2 D / c`;
  the echo of the three highlights against eq. (66) with eq. (68), sample by sample; the echo
  below the reverberation in level and apart from it in Doppler.
- ASIAEX: arrival times and levels of the five eigenrays against eqs. (36)-(38); identical,
  crisp pulses without time spreading; tails and ping-to-ping variation with it; decay times
  against the rule of thumb of Sec. 12.4; the surface Doppler spread; the two-bounce tail.

A simulator of echoes has no one-way source (SST's `DirectSignal`). The ASIAEX path is made
with a relay: a point of -20 dB strength 0.1 m from the projector, so that the transmit leg
is 0.1 m long and (1 / 0.1) x 0.1 = 1 leaves the return leg alone.

**18.2 What the examples showed.**

- *One Doppler per component is wrong for pairs of different eigenrays (a defect when this
  was written; fixed since by keying components by Doppler bin, Sec. 17.5, and the two tests
  below are no longer xfail).*
  A reverberation component is (azimuth sector, elevation of the receive ray) with one Doppler
  shift, the power-weighted mean over every transmit path and every time that arrives from
  there (Sec. 14, "typically one Doppler cell per direction" taken literally). With several
  eigenrays per leg, pairs that share a receive ray share a shift: in Pursuit the return with a
  bottom-reflected transmit leg and a direct receive leg is given the Doppler of the direct
  pair, and where the direct pair sits in a null of the beam it is given the bottom pair's.
  The second is visible as a change of the record a second before any bottom path can arrive.
  Two tests are marked xfail for it. Eq. (71) gives each pair pq its own factor; the cure is a
  Doppler bin (or the transmit path) in the component's key.
- *The reverberation's Doppler is first order in v / c.* At 20 m/s and 9.9 kHz it is 3.6 Hz
  below eq. (71), a quarter of the resolution of a 0.1 s pulse; echoes use the exact factor.
  (No longer so in bistatic mode: Sec. 17.5 uses the exact factor.)
- *Eq. (36) is right as printed* (item 24 of Sec. 11): it equals a leg-by-leg unfolding for
  every sequence of up to three bounces, and so do (37), (40), (41) against the provider.
- *The "less than 10 Hz" of Sec. 12.3 does not settle note 23.* For our ASIAEX geometry the
  spread is 0.3 Hz (AR(1) half-width) or 0.7 Hz (Gaussian rms) with sin(g), and 6.5 Hz with
  the rule as printed on p. 101 (1 / sin(g)): all under 10 Hz. Sec. 16 derives sin(g).
- *`multi_bounce: sum` and the remark on p. 104.* TR 0702 does not say how SST combined two
  bounces; it says the result is "too high immediately after the pulse and too low later".
  Our `sum` (one exponential with the summed mean delay) has that error when both bounces
  scatter. With the ASIAEX bottom as printed (rmsHeight 0.012: 93 % coherent on the
  surface-bottom path, with the longer decay) its error has the other sign: all the scattered
  energy is given the bottom's delay although 7 % of it met a rough bottom. `convolve` is the
  one to use where the two boundaries differ.
- *Not in the simulator, so not checked:* a one-way source (the target's radiated noise in
  Pursuit; tonals), K-distributed noise, the removal of near-specular reverberation when time
  spreading is on, a near-specular surface scattering model (Figs. 3-4, panels 3-5).

**18.3 Closed forms through the whole simulator.** `sst_echo_level.yaml`: the compressed echo
of a 7 dB highlight is SL - 2 TL + TS in free field, and each of the six groups of eigenray
pairs of an image medium has the level and delay of eqs. (36)-(38) and (66); a moving highlight
through a sampled pulse and the Doppler bank. `sst_sediment_reverb.yaml`: reverberation of
`medium_sand` against `sigma(grazing) pi c T / R^3`, as scattering-function reverberation and as
point scatterers; signal-to-reverberation after pulse compression against
`TS - 10 log10(sigma pi c R / B)`, 3 dB per octave. Numbers: `docs/figures/sst_validation.md`.

## 19. Sampled pulses and per-ray Doppler in the C++ engine; the arrival-time pass count

This supersedes the "Engines: NumPy only" paragraph of Sec. 15 (echo synthesis) and the
NumPy pairing fallback of Sec. 16 (`doppler`). Nothing here is from a source: it is a port
of our own NumPy code, checked against it and against closed forms.

**19.1 Sampled pulses in the kernel.** `synth_kernel.hpp` has a second pair function,
`pair_sampled`, used when the waveform type is Arbitrary. It is the analytic pair function
with the closed-form pulse replaced by a table read, and follows `sim/waveform.py
synthesize_into` (now the reference engine) step by step:

- the same three arrival times and emission-time quadratic `u(t)`;
- the carrier phase `F u(t)` (less the mixer's phase) is exactly quadratic in the sample
  index, so the kernel's phasor recurrence (re-seeded every 16 samples) has no truncation
  error for it;
- the envelope is read from the fine table of Sec. 15 (windowed sinc onto a grid
  `table_factor` = 8 times finer, built once in NumPy by `SampledWaveform.kernel_table`)
  with the same four-point cubic, clamped the same way;
- loss that depends on frequency: the table holds the `gain_knots + 1` parts of the envelope
  (triangular weights in frequency) and their sum. The gain is evaluated at each knot
  frequency per (path, channel) exactly as before; each sample is
  `sum_i g_i b_i(u)`. NumPy forms `sum_i g_i b_i` over the whole table for each
  (path, channel) and then interpolates; the kernel interpolates and sums per sample. The
  two are the same linear expression, so there is no modelling difference to bound, only
  rounding. When the gain does not vary across the band (to 1e-12) the kernel multiplies the
  summed envelope instead; NumPy decides that for a whole block of paths, the kernel per
  pair, which again changes rounding only.

The table is plain data handed to the kernel as a pointer in its parameter block
(`sim::EnvelopeTable`); the CUDA driver uploads it and points the device copy of the
parameters at the upload. Host and device run the same kernel body.

Not ported: `table_factor: 0` (the windowed sinc at every output sample) and more than 32
`gain_knots`; those waveforms run in NumPy with a warning and the `numpy_sampled` note. The
note is no longer written when C++ ran.

Measured, C++ against NumPy, relative rms of the record
(`tests/python/test_sampled_engine.py -s`, 47 cases: every tier, real and baseband, beams
and absorption, 0/3/8 knots, phase codes, HFM, stepped FM, 12 m/s closing, board clock
errors, platform acceleration): 1.2e-9 to 4.9e-7, worst single sample 3.4e-7 of the peak.
That is float32 rounding: the kernel adds each path into the float32 record as it goes,
NumPy sums in double and rounds once (the largest figures are the cases with the most
paths per sample). Independent of NumPy: a table of ones reproduces the analytic CW pulse
exactly, a ramp envelope is read to 2e-12, and three weighted bands give the weighted sum
of the knot gains (`tests/cpp/test_sim_synth.cpp`, `test_sampled_engine.py`).

Time (4 shared cores under load from other jobs, 300 scatterers x 128 channels, fls_300k,
1 ms pulse; best of 3): with beams and absorption 17.7 s NumPy -> 2.4 s C++ (analytic
kernel 0.8 s); omnidirectional 2.8 s -> 0.54 s (analytic 0.32 s). The remaining factor
of 3 over the analytic kernel with beams is the nine-band sum at every sample
(36 complex table values against one closed-form envelope).

**19.2 The pass count of the arrival-time solve.** The kernel finds when sound emitted at
u meets the moving outbound point, then the moving receiver, by fixed-point passes
`t <- u + r(t) / c`; each pass gains a factor v/c. It has always run 3 passes outbound and
4 on return. The counts are now parameters (`SynthParams::arrival_passes_out/back`,
`arrival_passes=(out, back)` in Python). For a target closing at 12 m/s
from 400 m, 8 kHz CW, against the closed form (echo at `f (c + v)/(c - v)` from
`2 R / (c + v)`), relative rms = phase error in rad:

| passes (out, back) | (3, 4) earlier default | (4, 4) default | (4, 5) | (5, 5) | (8, 8) |
|---|---|---|---|---|---|
| error | 6.9e-3 | 5.5e-5 | 5.5e-5 | 4.4e-7 | 3.7e-8 (float32) |

The default's error is the outbound residual `2 pi f (R/c)(v/c)^3` (6.9e-3 predicted); it
grows with range, frequency and the cube of speed, and is a constant phase offset of the
echo (the Doppler factor itself is right). One more outbound pass removes it to 5.5e-5.
Sampled pulses use (8, 8), as the NumPy reference does. Analytic pulses ran (3, 4) until
the owner chose (4, 4) as the default (D-83): records of moving scenes change at the 1e-3
level or below, static scenes not at all. Cost of the passes alone: 0.33 us (3, 4) to 1.2 us (8, 8) of core time per
(path, channel) pair, against 40 us or more for a pair with a 1 ms pulse in the window:
about 2 % at most, and not resolvable in whole-ping times on the loaded machine.

**19.3 Per-ray range rates in the C++ pairing.** `sim::two_way_paths_doppler` takes a range
rate per eigenray on each leg and returns a velocity for each end of each pair,
`v_q = v_scatterer + rate_out * launch_out`, `v_p = v_scatterer + rate_back * launch_back`
(the same arithmetic as the NumPy branch, in the same order). The kernel already had a
velocity for each end of a path (`SynthPath::q_velocity`, `p_velocity`), on host and
device, so nothing downstream changed. `flank_sst_spread` gives a bit-identical record
through the C++ pairing and through the old NumPy one.

---

## 20. One-way sources, near-specular removal, non-Gaussian noise, bubble loss [OURS]

### 20.1 One-way sources (`sources:`, `sim/sources.py`)

SST's `Source` + `DirectSignal` (TR 0702 Secs. 7-8) and its radiated-noise signals (Sec. 4.6:
Gaussian noise with a spectrum, harmonic families eq. 29, any pulse). A source has a
trajectory (the target description of `sim/targets.py`) and one or more signals. Eigenrays
are found from the array's reference point to the source and travelled backwards, exactly as
the return leg of an echo, so every provider works, the time-spread wrapper with its taps,
angle and Doppler spreading included. Each ray is an apparent source `p` with velocity `w`:
the image's (vertical velocity mirrored once per bounce), corrected along the ray so that its
range rate equals the source's speed along the ray where it leaves the source, plus the ray's
own path-length rate.

Doppler is time scaling, exact: the emission time `u` of the sound an element at
`R + V t` receives at `t` solves `|p + w u - R - V t| = c (t - u)`, a quadratic with the root

```
s = t - u = (-A.w + sqrt((A.w)^2 + (c^2 - w^2) |A|^2)) / (c^2 - w^2),   A = p + w t - R - V t
```

and `du/dt` along a straight ray is TR 0702 eq. 63, `(1 - V_R.n/c) / (1 - V_S.n/c)`.

What runs where. A pulsed signal (`type: waveform`, any waveform of the simulator) runs in
the echo kernel unchanged, host or CUDA: the kernel's path is projector -> q -> p -> element,
and a one-way path is that path with a null first leg. The projector is made omnidirectional
and `q = tx + r0 (e - v/c)` rides with it (r0 = 1 m, e any unit vector, v the platform's
velocity): the first leg is then exactly r0 long whatever the motion, `p` is the apparent
source, the 1 m of delay and the emission time are taken out through the record's start time
(with the mixer phase `2 pi f_c shift` for a baseband record), spreading over 1 m is 1, and
absorption over 1 m is put back at the centre frequency (the residual across a band is
alpha'(f) x 1 m, below 1e-4 dB). Continuous signals (noise, tonals, a file) run in a
streaming synthesiser in NumPy: `s(t)` exactly at knots 4 ms apart per (ray, channel), linear
between (error s'' h^2 / 8, about 1e-12 s); tones evaluated analytically at `u`; sampled
signals read at `u` with the windowed-sinc interpolator of Sec. 15, as real samples or as a
complex envelope about the record's carrier; absorption per frequency at each ray's mean
range. There is no C++ or CUDA streaming synthesiser.

Measured (tests/python/test_oneway_sources.py): kernel pulse against streamed tone with
source and receiver both moving, inside the pulse: 5.5e-5 relative (float32); level against
SL - 20 log r - alpha r: within 0.01 dB; image arrival times within one sample; tonal Doppler
against eq. 63: within 0.02 Hz of about +100 Hz (first order is more than 0.3 Hz off); Lloyd's mirror over 28
depths: within 1 % of `|exp(ikr1)/r1 - exp(ikr2)/r2|`; noise spectrum level at the hydrophone
within 0.25 dB in the mean.

A passive capture is a scenario with `pings.transmit: false`: the same record, of the
length of the receive window, with no echoes, reverberation or transmit reference. The
record schema is unchanged; `sonar_live --passive` and `core.doa_passive` read it.

The ASIAEX example used a -20 dB relay 0.1 m from a projector on a lever arm to fake the
one-way path; it now uses a source. Same 20 pings, relay / source: arrival time error
0.20 / 0.19 us, level error 0.0025 / 0.0026 dB, tail decay 2.946 / 2.945 ms (surface) and
7.693 / 7.694 ms (bottom), ping-to-ping std 1.876 / 1.881 dB (surface), Doppler width
0.3105 / 0.3109 Hz. The relay's outbound multipath (55 dB down) is gone.

### 20.2 Near-specular reverberation when time spreading is on (`exclude_specular`)

TR 0702 (Sec. 12.1, 12.4) removes "the near-specular parts of the reverberation" when
`doTimeSpread` is on and does not say how they are delimited (open question 13). Our
partition comes from the facet model of Sec. 16: a boundary of rms height h with Gaussian
slopes of total mean-square slope s2. Geometric optics on those facets gives the strength

```
lobe = |R|^2 / 4 (|q|^2 / q_z^2)^2 exp(-(q_x^2 + q_y^2) / (q_z^2 s2)) / (pi s2),   q = n_out - n_in
```

whose integral over scattered directions is `|R|^2 sin g_in` (all the incident energy; by
quadrature to 1 %). The tail of a time-spread bounce holds `(1 - exp(-G^2))` of the
flat-boundary energy, so the part of the strength it duplicates is
`(1 - exp(-(k h q_z)^2)) lobe`; the reverberation item uses `max(strength - that, 0)`.

Bookkeeping, source and receiver over the boundary (tests/python/test_reverb_specular.py):
removed reverberation / tail energy = 0.999, 0.996, 0.992 at 49 degrees grazing for
s2 = 0.005, 0.02, 0.045, and 1.09, 1.33, 1.65 at 9 degrees (the lobe is then as wide as the
grazing angle; geometric optics sends more down range than the flat-boundary energy the
tail conserves). With a strength made of that lobe on a Lambert floor: (coherent + tail) +
reverberation with the removal = coherent + all scattering to 1 %; without it the excess is
the tail energy exactly (over 1 dB of the total in the test's geometry). A strength with no
lobe (Lambert) loses under 12 % of its own energy and under 1 % of the tail.

When it applies. Reverberation is the sonar's pulse scattered once by a boundary. A tail
duplicates it only when the same pulse also arrives one way over the bounce: a separate
transmitter given as a source (the ASIAEX set-up) with a bistatic reverberation item.
Tails on the bounce legs of target echoes are boundary-and-target scattering, which no
reverberation item holds. So the key is opt-in, never automatic. Backscatter is untouched
(2.7e-4 of Lambert -27 dB at 60 degrees, under 1e-10 to 50 degrees) except at normal
incidence, where the lobe is the nadir return of a down-looking sonar: nothing else in the
simulator carries that, so a monostatic item refuses the key with a warning.

### 20.3 Non-Gaussian ambient noise

TR 0702 Sec. 4.6.2: `KNoise` is white, single-channel, "a fresh, independent K-distributed
random number for each sample", coloured afterwards by a filter (which, it notes, moves the
result back towards Gaussian). `receiver.ambient_noise.statistics: {model: k, shape}` with
no correlation time is that: kurtosis `3 / shape` for a white spectrum (tested to 6 % for shape 0.5, 2
and 8), about 1/16 of it after a filter 1/16 of the band
wide, and less after the mixing that gives an array its isotropic coherence. With
`correlation_time` it is the compound form used for reverberation (Sec. 14.4): the Gaussian
noise, spectrum and coherence included, times the square root of a unit-mean gamma variate
held over cells that long. The spectrum is unchanged at resolutions coarser than
1 / correlation_time; with `common: true` (default: a burst reaches every hydrophone) the
coherence matrix is exactly the Gaussian one; with `common: false` it falls by
`(Gamma(nu + 1/2) / Gamma(nu))^2 / nu`. An infinite shape is bit-identical to Gaussian.

`{model: alpha_stable, alpha}`: independent symmetric alpha-stable samples by Chambers,
Mallows and Stuck as printed in arlpy's `stable.rnd` (BSD). At alpha = 2 the formula is
`2 c sqrt(w) sin(p)` identically, a Gaussian of variance 2 c^2. For alpha < 2 there is no
variance: the configured spectrum sets the scale c = sigma / sqrt(2), sigma^2 the band power
of the Gaussian noise of that spectrum. White in band, independent channels; no colouring.

### 20.4 Bubble loss per surface bounce

Adopted (owner's ruling): `apl_1994`, `1.26e-3 / sin(g) U^1.57 f_kHz^0.85` dB for U >= 6 m/s
and the 6 m/s value times `exp(1.2 (U - 6))` below, read in arlpy 1.9.3
(`uwa.bubble_surface_loss`, BSD) and in UnderwaterAcoustics.jl (`surface_reflection_coef`,
MIT, which cites TR 9407 II-21). Neither states a range or a cap.

Our own derivation (`hall_novarini`), re-checked. (1) Unit: the source's eq. 24 and Fig. 9
give 4.7e7 bubbles/m^3 at the surface at 13 m/s, which needs n in m^-4; ours integrates to
4.67e7. (2) Air fraction at 2 m: 0.76e-7 at 10.5 m/s against 1e-7 quoted from Walsh and
Mulhearn in arXiv:physics/0006060 (and 0.15e-8 against 1e-8 at 6.2 m/s). (3) Radii: 33 to
330 um for 100 to 10 kHz, inside 10 um to 1 mm. (4) Extinction: integrating the full
damped response of every radius (Commander-Prosperetti as printed in arXiv:2410.14415
eqs. 6-9; damping 0.052 / 0.074 / 0.105 at 10 / 30 / 100 kHz, of which radiation is 0.0136)
gives 1.43 / 1.35 / 1.23 times the resonant approximation: 7 % from the resonance lying 2 to
5 % below Minnaert, the rest from bubbles smaller than resonance absorbing through viscous
and thermal damping (4 to 21 % of the total). So the derivation was not too high; if
anything low. (5) Refraction: the layer is slow (1490.6 m/s at the surface at 10 m/s,
30 kHz); Snell shortens the path by 9 % at 10 degrees and removes the need for the 2 degree
floor (63 dB at 2 degrees instead of 107). (6) Saturation: an opaque layer returns what its
bubbles scatter back, about `(w/2)(1 - m ln(1 + 1/m))` of the incident energy (single
scattering in a half-space; w = 0.11 to 0.19 the scattering share of extinction,
m = sin g): 13 to 15 dB down at 10 degrees. Models `hall_novarini_layer` (4 + 5) and
`hall_novarini_screened` (+ 6, an upper bound: it counts all directions).

Per bounce at 10 m/s, 10 degrees, dB:

| kHz | hall_novarini | layer | screened | apl_1994 |
|---|---|---|---|---|
| 10 | 4.35 | 5.51 | 4.87 | 1.91 |
| 30 | 21.6 | 26.5 | 13.1 | 4.86 |
| 100 | 101.7 | 120.1 | 14.5 | 13.5 |

The two laws cross near 6 to 8 m/s (ratio 0.4 to 2 from 3.5 to 100 kHz at 6 m/s) and part
with wind (ours U^3 times a layer depth that grows with U; adopted U^1.57) and frequency
(f^1.4 against f^0.85). What remains unexplained is a real disagreement between this
population with linear bubble acoustics and a law fitted to sea measurements; the likely
cause is patchiness (the mean of exp(-T) over patchy clouds is far above exp(-mean T)),
which no opened source quantifies.


## 21. Noise fitted to a recording, and time spreading against measured channels [OURS]

Two additions that take their inputs from measurements instead of models. Sources opened for
this section, and what was taken from each:

- APL-UW TR 0702, Sec. 4.6.1 eqs. (23)-(26) and Sec. 4.6.2 (extracted text): the factor of the
  cross-spectral matrix, the Gaussian synthesis, the window condition, and the argument that
  heavy tails must be made in the time domain and filtered.
- NoiseModels.jl (org-arl, MIT), `src/gaussian.jl`, read in full: the "mixing coefficients on
  independent innovations" form. Its estimator (BFGS on lagged covariances) is not used.
- alpha-SGNm (A. Mahmood, BSD 3-clause), `asgn.m` and README, read: the sub-Gaussian
  construction (Gaussian vector times the root of a positive alpha/2-stable variate). Its
  alpha-SGN(m) sampler and tables are not used.
- Underwater Acoustic Channel Repository (UACR): the documentation site's file specification
  and channel pages (MIT), the paper arXiv:2609.03207 (Li, Chitre, Cuji, Preisig, Singer,
  Stojanovic, van Walree, 2026; Secs. II.D, IV), and data files from Zenodo
  doi:10.5281/zenodo.21287414 (CC BY 4.0). Taken: the file layout, the innovations convention
  (alpha-stable of scale 1/sqrt 2), the description of the SPACE'08 geometry and of the
  estimator. The project's replay and noise code was not read.

No coefficient in the code comes from memory of a paper: Kanter's formula for positive stable
variates and the log-moment formulas are tested against their defining properties.

**21.1 Noise model fitted to a recording** (`sim/noise_fit.py`, `tools/fit_noise.py`).
The model is `x = B * eta`: white innovations through mixing filters whose transfer matrix
satisfies `B B^H = S fs`, `S` the two-sided cross-spectral density matrix of the recording
estimated by Welch (Hann, 50 % overlap). `B` is the per-frequency Cholesky factor, TR 0702
eq. (23). For Gaussian noise the generator is TR 0702 eqs. (24)-(26) as printed (independent
complex Gaussians per frequency and update, inverse transform, a window with
`sum |w|^2 = 1`, here a sine window at 50 % overlap). For heavy tails the innovations are
`sqrt(tau) g` with a texture `tau` (gamma: K-type; positive alpha/2-stable: symmetric
alpha-stable), held for a fitted number of samples and common to the channels or not, and the
filters are applied in the time domain: because `g` is white and independent of `tau`, the
spectra and coherences are those of `B` whatever the tail.

*Tail fit.* On the whitened recording `e = B^-1 x` (lines notched out):
`var(log|e|)` is `pi^2/8` for a Gaussian, `pi^2/8 + trigamma(nu)/4` for the K family and
`(pi^2/6)(1/alpha^2 + 1/2)` for the alpha-stable one. Derivation: `log|e| = log(tau)/2 + log|g|`;
`log|g|` has variance `pi^2/8`; `log` of a gamma variate has variance `trigamma(nu)`; a
positive stable variate of index `a` has `E S^p = Gamma(1 - p/a) / Gamma(1 - p)`, so its
logarithm has variance `(pi^2/6)(1/a^2 - 1)`. Both families are fitted from that one number
and the one whose predicted quantile ratio q99.95/q50 is nearer the measured one is kept;
Gaussian when the excess over `pi^2/8` is inside four standard errors. The texture's hold time
is `1 + 2 sum_k rho(k)` with `rho` the autocorrelation of `log e^2` over the texture's variance
(`var(log e^2) - pi^2/2`); the cross-correlation between channels says whether it is common.

*Estimator variance.* With `n_d` independent blocks: the coherence estimate where the truth is
zero has `|gamma^|^2 ~ Beta(1, n_d - 1)`, a floor `E|gamma^| = sqrt(pi / (4 n_d))`; elsewhere
its standard deviation is `(1 - |gamma|^2) / sqrt(2 n_d)`; the spectrum level has
`4.34 / sqrt(n_d)` dB. Half-overlapped Hann blocks count as `K / (1 + 2 rho^2 (1 - 1/K))`,
`rho = 1/6` (computed from the window). So a coherence standard deviation of 0.05 needs 200
independent blocks: 3.4 s at 8 kHz with 256-sample blocks, about 1.1 s at 48 kHz with 512. The
Cholesky factor is refused below `2 x channels` independent blocks. All three numbers are
tested by Monte Carlo.

*What is not fitted as noise.* Reference inputs are dropped (record container). Blocks whose
power is more than 8 robust standard deviations above the channel's median are flagged; runs
of 4 or more and blocks that recur at a fixed interval (pings, sync bursts) are excluded and
listed; isolated ones are kept, since impulsive noise looks like that. Bins more than 3 dB (or
six standard deviations) above the median-smoothed spectrum are taken out as lines, kept
separately, and added back only on request.

*Limits.* Stationarity over the recording is assumed, not tested. The block must be much
longer than the transit time across the array (bias `(1 - transit/block)^2` on the coherence of
a directional source; the report prints it). An alpha-stable variate has no variance, so the
"spectrum level" of such a fit is the level of that record, not an expectation (tested: the
exponent and the scale are recovered, the level is not meaningful). The whitened channels are
mixtures of the physical sources when the channels are strongly coherent, so independent
impulsive sources look more Gaussian after whitening than they are. With the tail families the
generator is the filter form only; the frequency-domain synthesis is Gaussian by construction.

*Another configuration.* Channel count must match, or `channels` picks a subset (exact).
Any sample rate: the matrix is interpolated in Hz and factored again; frequencies the
recording never covered are refused unless asked for (then: no noise there). A complex
baseband output about a centre frequency is the band moved down with twice the power.

*Checked on* synthetic fields with a known cross-spectral matrix (isotropic `sin(kd)/(kd)`
plus a plane-wave interferer at 30 degrees, 8 hydrophones): fitted spectrum within 0.6 dB,
coherence within 0.02 rms of the closed form; regenerated noise the same; beam power against
bearing within 1 dB with the interferer at its bearing; array gain in isotropic noise within
0.35 dB of `C^2 / sum sin(k d_ij)/(k d_ij)`; the Mitchell-McPherson ensemble average converging
to the input as `1/sqrt(R)`. And on noise made, by the documented mixing equation, from two
published site models of the UACR (60 s each): SPACE'08 (12 hydrophones, Gaussian): spectrum 0.05 dB rms
in band, coherence error 0.009 rms against an estimator floor of 0.009; Singapore (3
hydrophones, alpha = 1.7): fitted alpha 1.66.

**21.2 Measured channels** (`sim/channel_replay.py`). A measured channel is a complex-baseband
impulse response `h(t, tau)` per receiver, stored every 15 ms. `replay` passes a transmit
signal through it (linear interpolation of `h` in time between snapshots, the tracked phase
and, for `phi_hat`, its delay drift restored): a one-way propagation tier made of a
measurement. It is a function, not an eigenray provider: a measured `h` has no rays, so it
cannot feed the two-way echo synthesis. Files with a constant resampling factor are refused.

The estimators, each tested on synthetic channels with known values:

| Quantity | Estimator | Recovered to |
| --- | --- | --- |
| delay spread | slope of the log of the incoherent profile after the peak; first moment | 15 % (decay time 0.3 and 0.8 ms at 0.077 ms bins) |
| coherent share | power of the time mean over the arrival, after dividing by the phase of the strongest arrival | 0.05 |
| Doppler width, shape | Welch spectrum over time; rms over +-6 w50, half widths holding 50 % and 90 %; w90/w50 is 2.44 for a Gaussian, about 4 or more for a Lorentzian on this axis | 8 % (rms), 10-15 % (w50) |
| coherence time | 1/e of the autocorrelation of the fluctuating part | 12 % (Gaussian), 20 % (Lorentzian) |
| level statistics | scintillation index of the arrival's energy, floor removed; 1 - c^2 for a Rician arrival | 0.12 |
| scattering function | Welch spectrum over time of every delay bin | marginals within 10-20 % |

A stored channel is the output of an adaptive estimator, not the channel: for the files used
here RLS with forgetting factor 0.995 updated every delay sample (arXiv:2609.03207 Sec. IV),
which to first order is a one-pole low-pass of 10.4 Hz on every tap. The model is passed
through that filter before comparison (`estimator_filter`); this is our approximation.

**21.3 Our time spreading against SPACE'08** (`tools/channel_validation.py`,
figures `docs/figures/channel_validation_*.png`). Data: UACR "Purple", 12.5 kHz, 6.5 kHz
band, one moored source 4 m above the seabed in 15 m of water, moored arrays with the top
element 3.3 m above the seabed at 60 m, 200 m and 1 km; five files of 53 s from two
recordings (`purple_1, 3, 5` and `purple_6, 8`).

*Not given by the dataset, hence assumed:* wind or sea state; sound speed (1500 m/s, no
gradient); seabed (flat; a sand-like half-space, 1700 m/s, 1.9 g/cm^3); omnidirectional
transducers; that the files of one recording are simultaneous; that the sea is described by
one number through Pierson-Moskowitz (height, vertical speed) and Cox-Munk (slope).

*One free parameter per recording*, the wind speed, fitted to one number: the rms Doppler
width of the first surface arrival at 60 m (`2 f v sin(g) / c`). Result: 3.96 +- 0.87 m/s and
3.70 +- 0.60 m/s (two sigma). Everything below is then a prediction; no option was tuned,
both values of each were run. "In" means within twice the larger of the measurement's
standard error (quarters of the record) and the model's record-to-record spread (seeds), plus
one resolution cell. 18 surface-interacting arrivals, grazing angles 1.3 to 32 degrees:

| Quantity | Model A: gaussian, facet, convolve | Model B: lorentzian, rule_of_thumb, sum |
| --- | --- | --- |
| Doppler rms width | 13 in, 3 out; median model/measured 1.06 | 4 in, 12 out; 1.60 |
| Doppler shape w90/w50 (measured 2.0 to 3.1) | 15 in, 3 out; 0.99 | 7 in, 11 out; 1.61 |
| tail decay time | 5 in, 12 out; 0.98 | 4 in, 10 out; 1.63 |
| coherent share (arrival and tail) | 12 in, 6 out; 1.14 | (same) |
| scintillation index | 3 in, 15 out; 0.46 | (same) |
| level (re whole channel) | 4 in, 14 out | (with bubble loss: the same, it is under 0.3 dB at 4 m/s) |
| arrival delay | 9 in, 9 out | (same) |

*What the model gets right.* (1) The Doppler spectrum of a surface-scattered arrival is
Gaussian, not Lorentzian: the measured spectra fall 30 dB by four standard deviations, where
the AR(1) spectrum SST uses (our default) would be at -12 dB. (2) Its width, with the single
fitted wind, across nine other arrivals per recording: one or two surface bounces, 60 m to
1 km, widths from 0.47 to 2.2 Hz; `2 f v sin(g) / c / sqrt(1 - exp(-Gamma^2))`, numerators
adding over bounces. (3) At 60 m (`Gamma^2` 8 to 21) the surface arrivals carry no coherent
part (measured 0.002 to 0.014). (4) The decay time of the first surface arrival at 60 m in the
first recording, 0.267 ms, is the facet value (0.261) and half the rule of thumb (0.554).

*What it gets wrong.* (1) At low grazing angles it is too coherent: at 200 m (6.7 degrees)
the surface arrival's coherent share is 0.14 and 0.21 measured against `exp(-Gamma^2)` =
0.35 and 0.45; at 1 km likewise (the twice-reflected group: 0.43 against 0.77 through the
estimators). Read as a height,
the measured shares want an rms height 1.38 times the Pierson-Moskowitz value for the fitted
wind, in both recordings: a sea with more height per unit of vertical speed than a fully
developed wind sea (swell would do that; not known). A second parameter (`rms_height` given
separately) would absorb it; it was not adjusted. (2) Delay spread: right on average (median
0.98) but not arrival by arrival; in the second recording the same 60 m arrival decays in
0.415 ms (facet 0.220, rule of thumb 0.525), and at 200 m, where `Gamma^2` is about 1, the
facet tail is three times too long in one recording (0.26 measured, 0.79 model) and too short
in the other (1.68 against 0.66). The facet picture is geometric optics; it has no reason to
hold at `Gamma` near 1. (3) Level fluctuations: the measured scintillation index of the
multi-bounce arrivals is 1.1 to 3.1, above the Rayleigh value of 1; the model's complex
Gaussian taps give at most 1 (0.6 after the same windowing). Focusing by wave crests would do
this; the model has nothing of the kind. (4) Levels of paths with a bottom bounce at 26 to 32
degrees are 3 to 6 dB below the assumed half-space; the surface-only arrival is within 1 dB at
60 m and 2 dB at 200 m. That is the assumed seabed, not the surface. (5) Surface arrivals come late: by 0.08 to
0.15 ms at 60 m and 0.23 to 0.27 ms at 200 m, both what a surface about 1 m higher than the
stated 15 m gives. Not corrected. (6) At 1 km the measured response has a tail of 15 ms that
three bounces do not make; only the first arrivals were compared there.

*Not tested by this dataset.* Bubble loss (0.1 to 0.3 dB at this wind); `multi_bounce`
separately from the decay time (model B's `sum` with 24 taps leaves the taps resolved where
the tail is long, an artefact visible in the profiles figure); angle spreading; anything at
other frequencies or sea states; any two-way path.

**21.4 Every site of the repository, and the model's dependencies** (`tools/channel_sites.py`,
`tools/channel_validation.py --sites all`; tables `docs/figures/channel_validation_purple.md`
and `channel_validation_sites.md`, figure `channel_validation_dependencies.png`). Section 21.3
used five Purple files; this is all eight that matter (three recordings) and the other sites.

*What the other sites allow.* The full comparison needs the geometry of each file, arrivals
that straight rays can name, and an estimator fast enough to follow the surface. No site
states wind, sea state, a sound-speed profile or a seabed (the paper's Sec. II and the site
pages were searched), so there is no case with nothing fitted, no Bellhop run with a stated
profile, and no test of a named sediment.

| Site | Carrier | Verdict | Reason |
| --- | --- | --- | --- |
| Purple, SPACE'08 | 12.5 kHz | full protocol | geometry per file, resolved arrivals, RLS tracker 10.4 Hz |
| Green, Norwegian fjord | 6 kHz | nothing | geometry stated, but LMS with 3600 taps and step 1/14400 follows about 0.05 Hz (our estimate; the input power is not given) and snapshots are 4.5 Hz apart: every arrival comes out steady (scintillation 0.00) |
| Black, Mariana Trench | 18 kHz | nothing | 7 ms delay window; the surface reflection above the array is 8 ms after the direct path |
| Red, Singapore | 25 kHz | model-free only | range (100-400 m) and depth (8-20 m) not given per file |
| Yellow, KAM'11 | 13 kHz | model-free only | arrivals are not where straight rays put them (not isovelocity), no profile given |
| Blue, MACE'10 | 13 kHz | model-free only | towed source: range, depth and speed not given per file |
| Pink, Japan | 6 kHz | not run | 14 km, no profile, no bathymetry; not downloaded (1.5 GB a file) |
| Brown, ATOC | 75 Hz | not applicable | 3250 km of deep refracted paths |

*Purple, three recordings.* One number fitted per recording, as before: wind 3.96, 3.70 and
9.14 m/s (two sigma 0.87, 0.60, 1.91). The third recording is a different sea, which is what
makes the following a test. 27 surface arrivals, grazing 1.3 to 41.5 degrees. A: gaussian,
facet, convolve; C: gaussian, rule_of_thumb, convolve (the suite's defaults when this was run, before D-85; the defaults since D-85 are A's: `doppler_spectrum: gaussian`, `decay_time: facet`, `multi_bounce: convolve`, `sim/timespread.py`); B: lorentzian,
rule_of_thumb, sum.

| Quantity | A | C | B |
| --- | --- | --- | --- |
| Doppler rms width (in / out, median model/measured) | 20 / 4, 1.05 | as A | 6 / 18, 1.54 |
| Doppler shape | 23 / 4, 0.97 | as A | 13 / 14, 1.55 |
| tail decay time | 10 / 13, 1.02 | 7 / 12, 2.13 | 8 / 11, 1.93 |
| coherent share (arrival and tail) | 19 / 8 | 19 / 8 | |
| scintillation index | 5 / 22, 0.37 | 6 / 21, 0.38 | |

*Verdict per ingredient* (measured over closed form, across the arrivals not used for a fit):

1. **Doppler width proportional to sin(grazing) and to the square root of the number of surface
   bounces: supported.** Median 0.89, scatter a factor 1.13 (n = 24), from 0.4 to 5 Hz, the
   same below and above 12 degrees (0.88, 0.90) and in the rough recording. Two-bounce
   arrivals 0.82 (n = 4).
2. **Doppler width proportional to frequency: supported, inside one band only.** Upper against
   lower half of 9.2-15.8 kHz: ratio 1.185 +- 0.017 (standard error, 21 arrivals) against 1.216
   for proportionality and 1 for none. At 200 m, where Gamma^2 is about 1, the model's
   own factor 1 / sqrt(1 - exp(-Gamma^2)) lowers the expected ratio to 1.09 in the calm
   recordings; measured 1.05 and 1.03 there (1.20 in the rough one, where Gamma is large). Across carriers it is **not testable**: no second site has both geometry and a
   fast tracker.
3. **Doppler spectrum Gaussian: supported**, and not only at Purple. Shape w90/w50: Purple
   median 2.53 (1.76-3.08, n = 27); Red at 25 kHz 2.3-2.5; Yellow 2.34; Blue 1.77 (Gaussian
   2.44, Lorentzian about 4 on these axes). The other three are unidentified arrivals through
   slower trackers.
4. **Coherent share exp(-Gamma^2): the form is supported, the Pierson-Moskowitz height is
   contradicted.** In the two calm recordings the shares at 200 m want 1.36 times the height of
   the wind law (scatter 1.06, n = 6), and it holds within the band (lower half more coherent
   than upper in 8 of 9 arrivals at 200 m). But the excess is not a constant: read as heights,
   the 200 m surface arrival gives 0.116, 0.102 and 0.144 m in the three recordings where the
   wind law gives 0.084, 0.073 and 0.447 m. The measured height hardly changes while the
   Doppler width (vertical speed) more than doubles: a fetch-limited or swell-carrying sea, not
   a fully developed one. The height has to be an input of its own.
5. **Decay time: neither rule is supported in detail; the rule of thumb is too long.**
   Measured over model: facet median 0.99 with a scatter of a factor 5; rule of thumb 0.47
   with a factor 3. Above 12 degrees (Gamma^2 over 7): facet 1.28, rule of thumb 0.75. Below
   (Gamma^2 of order 1): facet 0.59, rule of thumb 0.28.
6. **Complex Gaussian taps: contradicted at steep angles.** Scintillation index of fully rough
   arrivals: median 1.43 (one bounce, n = 4), 1.31 (two, n = 7), 1.72 (three, n = 6), up to 4.8
   in the rough recording; the model gives 0.6 after the same windowing and cannot exceed 1.
   It grows with grazing angle, not with the number of bounces: at 7-10 degrees it is 0.5-0.9.
   Not general: Red 0.6-0.7, Yellow 0.21, Blue 0.25 (low grazing angles, as far as can be
   told). A K-distributed tap intensity (gamma texture of shape nu on each tap, scintillation
   1 + 2 / nu) would need nu of about 3 to 7 for the medians and 0.5 for the extreme.
7. **Bubble loss: weakly indicated, not testable.** From 4 to 9 m/s the first surface arrival
   at 60 m falls from -9.5 and -11.1 dB to -17.4 dB re the whole channel; the model falls by 2 dB
   without bubbles and 3.3 dB with the Hall-Novarini law (which is outside its stated range at
   12.5 kHz). One arrival in one recording, and levels carry a few dB of normalisation.
8. **Multi-bounce rule, angle spreading, refraction, sediments: not testable** with these data.

*Two things from 21.3 that are now explained.* The 15 ms tail at 1 km is higher-order bounces,
not missing physics: flat images up to 12 bounces put -5.2, -5.6 and -9.0 dB of the channel's
energy in 1-3, 3-8 and 8-18 ms, against -6.3, -7.4 and -10.9 dB measured (three bounces: -13.9
dB and nothing). The late surface arrivals fit a surface 0.3 to 0.6 m (60 m) and 0.8 to 1.5 m
(200 m) above the stated 15 m, growing from the first recording to the third at both ranges;
a tide would do that, but the two ranges do not agree on the amount.

*What the data ask of the model* (nothing was changed; for the owner to decide):
`doppler_spectrum: gaussian` as default (already decided); `decay_time: facet` in place of
`rule_of_thumb` (Purple only, three recordings); the surface's `rms_height` given separately
from `wind_speed` whenever a height is known (`{wind_speed, rms_height}` already works: an
explicit value overrides the wind law); an opt-in non-Gaussian tap amplitude for steep
bounces (one site: not implemented).

*Other datasets.* Watermark (FFI; van Walree et al., IEEE JOE 42, 2017) and the SFI Smart
Ocean dataset (IEEE DataPort, the origin of Green, which per a search result includes
sound-speed profiles) are on hosts this session may not reach (`ffi.no`,
`ieee-dataport.org`): licence and contents not verified. A search of Zenodo found no other
measured underwater channel set, and none above 30 kHz.

**21.5 Watermark, Smart Ocean, and two opt-in additions tried against Purple**
(`tools/channel_watermark.py`, `tools/channel_validation.py --sites all`; tables
`docs/figures/channel_validation_watermark.md`, `channel_validation_purple.md`,
`channel_validation_sites.md`).

*SFI Smart Ocean dataset* (van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh, IEEE DataPort
2025, doi:10.21227/3aa6-4k33, CC BY 4.0; the dataset page was read). Raw hydrophone
recordings of probe and communication signals in three bands (4-8, 9-14, 24-32 kHz), two
hydrophones, an enclosed fjord, overnight; two sound-speed profiles, a photograph of the sea
surface, bathymetry; 15 GB. It is what the question needs (raw data, so the tracking
bandwidth would be ours to choose; three carriers; profiles), but the files are served only
to logged-in users. Not downloaded. The page states no wind or wave height.

*Watermark* (FFI; van Walree, Socheleau, Otnes, Jenserud, IEEE J. Ocean. Eng. 42, 2017, read
as hal-01630274; BCH1 report hal-01404491, read). A 770 MB zip served openly as "free
download"; no licence text in it. Five channels of correlative soundings (no tracking
filter). Stated: range, water depth, deployment (bottom-mounted, suspended, towed), band,
month. Not stated: wind, sea state, sound-speed profile, seabed, the units' heights above the
seabed. (The user manual is on a host that was not reachable.) So no arrival gets a grazing
angle and nothing was fitted; what was tested needs neither:

| Channel | Scattered arrivals: Doppler shape w90/w50 | Fully scattered: scintillation index, median (10-90 %) | Upper / lower band width ratio (frequency ratio) |
| --- | --- | --- | --- |
| NOF1, Oslofjord, 14 kHz, 59 soundings | 2.50 (1.79-3.48) | 0.74 (0.66-0.82), n = 12 | 1.112 +- 0.042 (1.240) |
| NCS1, Norwegian shelf, 14 kHz, 60 soundings | 2.48 (2.14-2.86) | 0.92 (0.72-1.24), n = 718; 8 % above 1.3 | 1.158 +- 0.003 (1.240) |
| BCH1, Brest harbour, 35 kHz, one sounding on 4 hydrophones | 2.94 (2.57-3.53) | 0.13 (0.04-0.48), n = 7 | 1.011 +- 0.004 (1.113) |

A further test without geometry: the model ties an arrival's coherent share c to the rms
width s of its scattered part, s sqrt((1 - c) / ln(1/c)) = v / (2 pi h) for every arrival off
one surface (`channel_replay.wave_frequency`; our own model keeps it, tested). At NOF1 the
width does rise as the share falls (rank correlation -0.43 within a sounding) and the
implied wave frequency is 0.82 Hz, but dividing the widths by the predicted factor does not
reduce their scatter (a factor 1.34 against 1.37). At BCH1 the correlation has the wrong sign
(+0.67; 23 arrivals of one sounding). Soundings of 33 s may be too short for this; on
synthetic soundings five times longer the tool finds the relation.

*What the new data settle.* (1) Gaussian Doppler spectra: confirmed at two more sites with an
estimator that has no tracking filter. (2) Scintillation above 1: not general. 718 fully
scattered arrivals at NCS1 have a median of 0.92; complex Gaussian taps are right there, and
the 1.1 to 4.8 of Purple at 60 m belongs to steep, short bounces. (3) Width growing with
frequency inside a band: confirmed at NCS1 beyond doubt as a trend (1.158 +- 0.003 against 1),
but it is less than proportional (1.240); not seen at 35 kHz in the one BCH1 sounding.
(4) Proportionality across carriers and the wave-height excess: NOT settled. Both need a
known sea state or a known geometry at a second carrier, and no open dataset gives either;
Smart Ocean, behind a login, has the geometry and three carriers but, as far as its page
says, no wind.

*Purple, all nine files, one command.* The 1 km files now use images up to 12 bounces, which
gives the long tail (within about 5 dB of the measured profile to 10 ms, and up to 8 dB too
strong after: the assumed seabed reflects too well). With dozens of merged paths per group the
per-arrival rows at 1 km are not to be relied on; the closed-form comparison is unaffected
(Doppler width, measured over closed form: median 0.96, scatter a factor 1.18, n = 45).

*Two additions, tried as extra option sets* (defaults unchanged; `time_spread.tap_shape`,
`surface.rms_height` / `height_scale`, docstring of sim/timespread.py items 5 and 6). More
knobs fitting better is not validation; what counts is the rows nothing was fitted to.

- D = A with the rms height fitted per recording to the coherent share of the first surface
  arrival at 200 m (two fitted numbers: wind, height; heights 0.116, 0.102, 0.145 m against
  the wind law's 0.084, 0.073, 0.447 m). Predicted, not fitted: the coherent shares of the
  other arrivals. At 200 m, SB+BS: measured 0.042, 0.118, 0.041; D 0.035, 0.061, 0.017; A
  0.206, 0.344, 0.014. Over the 60 m and 200 m files: coherent share in / out 20 / 0 with D
  against 19 / 4 with A; Doppler widths unchanged (16 / 4 against 17 / 3). The gap closes
  with one extra number per recording, and that number is a physical one that could be
  measured.
- E = D with `tap_shape` fitted per recording to the scintillation index of the first
  surface arrival at 60 m (three fitted numbers; shapes 8.5, 1.3, 0.37). Predicted: the
  scintillation of the other arrivals. At 60 m it carries over (in / out 7 / 2 against 1 / 11
  with A; e.g. second recording SB, BS, BSB measured 1.85, 1.31, 1.68, E 1.86, 1.27, 1.70).
  At 200 m in the rough recording it does not: measured 0.89, 0.83, 0.81, E 5.7, 3.4, 2.2. A
  single shape per sea is wrong; it has to depend on grazing angle (the option takes a
  table), and the fitted shapes differ by a factor 20 between recordings. A description of
  the steep arrivals, not a model of them.

**21.6 Raw recordings in the long-range array's band: the SFI Smart Ocean dataset, 4-8 kHz**
(`sim/channel_sounding.py`, `tools/channel_smartocean.py`; table
`docs/figures/channel_validation_smartocean.md`, figures `channel_validation_smartocean_*.png`;
detection runs `tools/decay_time_sensitivity.py`, `docs/figures/decay_time_sensitivity.md`).

*Sources opened.* The dataset page (IEEE DataPort, doi:10.21227/3aa6-4k33, CC BY 4.0) and its
descriptor article (van Walree, Otnes, Tomasi, Henriksen, Vahlin, Danre, Bergh, Socheleau,
Chitre, IEEE Data Descriptions 2 (2025) 218-223, read as hal-05793814). Taken from the
article: positions and depths (its Table II), source level (Table I), 94 Pa per full scale,
file layout, "calm, with little wind and some drizzle". Files used: the 4-8 kHz receptions
(160 recordings), the transmitted waveforms, `CTD.csv`, the depth geodatabase. Not available:
the supplementary waveform description (the transmitted WAV files serve as probes instead),
the sea-surface photograph, the site paper (Bergh et al., Sensors 24 (2024) 6530), the
9-14 and 24-32 kHz receptions.

*The experiment.* A sender at 20 m hung from a floating fish-farm platform; two hydrophones
on bottom tripods, R1 at 43 m depth 274 m away and R2 at 14 m depth 1020 m away; 80 cycles
of probes 10 minutes apart from 17:00 to 06:20 UTC on 14-15 November 2024. Not stated: wind
speed, wave height, tide, seabed type, tripod heights, water depth at the instruments. The
chart (10 m contours, soundings of 2002) gives 47 m at the sender and a seabed that falls
to 61 m and rises to 40 m towards R1, and falls to 86 m, rises to 53 m, falls to 77 m and
climbs a slope to 18 m towards R2. It is not exact: it puts the seabed at R1 above the
stated hydrophone depth, and the casts reach 54.6 m where it says 47 m.

*The two casts* (the file's own sound-speed column; nothing recomputed). Both: 1480-1481 m/s
at the surface, rising to a maximum of 1494.7 and 1495.4 m/s at 19 m, then falling slowly
to 1491-1493 m/s at 54 m. The sender sits on the maximum. The way up reads 1.0 to 1.9 m/s
above the way down at 20 m (the sensor's lag); the mean of the two is used. Assumed: that a
cast at the platform holds along both paths.

*Estimator* (ours; no tracker). Recordings are taken to complex baseband about 6 kHz at
12 kHz. Periodic probes are deconvolved period by period (chirp trains: 4 Hz snapshots with
a 250 ms delay window, 16 s; 40 Hz with a 25 ms window, 10 s; 0.745 s chirps 3 s apart for
the whole response); the 60 s aperiodic sequence by least squares in 0.5 s blocks every
0.25 s over a 50 ms window. Band 5 kHz with a raised-cosine weight: half-power width
0.29 ms, first sidelobe -31 dB. Clock difference between sender and recorders: 1e-6
(removed). Floors, re the direct arrival at R1: -38 dB (4 Hz chirps; this is fluctuating
reverberation, not noise), -33 dB (least squares: reverberation beyond its window), -24 dB
(40 Hz chirps: reverberation folded ten times). Reverberation, not noise, limits every
sounding: 36 % of the energy at R1 and 59 % at R2 arrives later than 10 ms.

*Refraction* (stated: geometry and casts; assumed: a half-space seabed 1600 m/s, 1.8,
0.5 dB per wavelength, a 1.5 m tripod at R1; fitted: nothing).

| | straight rays or Bellhop isovelocity | Bellhop, cast 1 / cast 2 | measured |
| --- | --- | --- | --- |
| R1: surface reflection after the direct arrival | 4.14 ms | 4.53 / 4.45 ms | 4.28 to 4.43 ms through the night |
| R1: grazing angle at the surface | 13.0 degrees | 14.7 degrees | - |
| R1: loss of the direct arrival | 48.8 dB | 48.8 / 48.7 dB | 47.5 dB (source level 183) or 45.5 (181) |
| R2: strong arrivals in the first 6 ms | direct, bottom, surface within 0.37 ms | two 2.15 ms apart, the first stronger / one | two 1.84 ms apart (1.68 to 1.91), the second 1.6 dB stronger (-5.5 to +8) |
| R2: loss of the strongest arrival | 60.2 dB | 58.2 / 57.1 dB | about 71 dB; the whole response 58 dB |

At 274 m the profile moves the surface reflection by 0.3 to 0.4 ms (more than a resolution
cell) and its grazing angle by 1.7 degrees; isovelocity is outside the measured range all
night, the casts are 0.02 to 0.25 ms above it. The measured value drifts smoothly by
0.13 ms, which is 0.4 m of water level: a tide, not stated, of the same size as the
difference left. At 1 km the profile decides everything: with the sender on the sound-speed
maximum there is no direct path, the two casts give qualitatively different arrivals, a
metre of seabed depth at the receiver changes which bounces arrive (tried), and no arrival
there can be named with confidence. Isovelocity cannot make the measured pair 1.84 ms
apart at all; cast 1 makes a pair 2.15 ms apart with the levels the other way round. The
strongest arrival is 13 dB weaker than Bellhop's while the energy of the whole response is
what spherical spreading gives: the sound arrives, spread over a second. The time
difference between the receivers (0.4957 s) is 3 ms short of every prediction (0.4990
isovelocity): 5 m, more than the positions and the recorders' time stamps can be trusted
to, so no sound speed is read from it.

*The surface reflection at R1* is the one arrival that is a single surface bounce, is
named without doubt and stands clear of its neighbours. Against the direct arrival,
snapshot by snapshot:

- Coherent share at its peak 0.987 (0.978 to 0.992 over the night); with its tail to
  0.9 ms, 0.969. The surface was close to a mirror at 6 kHz.
- The fluctuation is two processes. At the peak it is mostly phase (variance 0.0084 rad^2
  against 0.0037 relative amplitude), slow (rms 0.36 Hz, 13 % of its power above 0.5 Hz, a
  line near 0.2 Hz rather than a Gaussian: w90/w50 = 1.4 to 1.5), and it has the shape of
  the steady arrival: a mirror heaving 0.8 cm rms, or the sender heaving 1.3 cm rms on its
  platform (the two cannot be told apart: the first arrivals' phases at R1 and R2, which
  lie in nearly opposite directions, are correlated -0.70, so the sender does sway, by
  millimetres along the paths). Behind the peak there is a tail that the steady arrival
  does not have: 2.2 % of the arrival's energy, faster (rms 0.63 Hz, 74 % above 0.5 Hz),
  absent behind the direct arrival (20 dB lower there). Shorter waves scatter later and
  move faster.
- Phase variance in the upper over the lower half band: 1.57 +- 0.07 where the same
  estimator reads 1.42 for a pure change of path length and 1 for a phase that does not
  depend on frequency (1.26 +- 0.04 against 1.11 for the least-squares soundings). The
  variance grows as frequency squared within the band.
- Scintillation index of the arrival's energy 0.015: far from a fully scattered arrival.

*What this says of the model's ingredients.*

1. **Coherent share exp(-Gamma^2): consistent, but not a test** (the height is the fitted
   quantity: 0.8 cm for the unspread part, 1.1 cm for the tail, 1.4 cm together). No wind
   was stated; the wind whose Pierson-Moskowitz height is the tail's is 1.5 m/s, and the
   wind whose noise curve matches R2's noise level is 1.9 m/s (below): two independent
   readings of "little wind" that agree.
2. **Doppler width: the low-roughness limit of the formula is seen, the wind law is not.**
   For Gamma << 1 the model's width is the rms wave frequency v / (2 pi h) whatever the
   acoustic frequency, and the measured widths are wave frequencies (0.36 and 0.63 Hz).
   The Pierson-Moskowitz sea of that height would have 1.3 Hz. One height and one speed for
   the whole arrival is wrong here: peak and tail are different waves. Proportionality of
   the width to the acoustic frequency belongs to rough surfaces and cannot be tested at
   this site, nor against the 12.5 kHz site (which is rough): the two are in different
   regimes.
3. **Complex Gaussian taps: not what the peak does** (its fluctuation is 70 % phase; a
   Gaussian tap added to a steady arrival gives equal parts), adequate for the tail as far
   as can be seen. The energy involved is 1 %.
4. **Decay time: not settled.** Mean excess delay of the tail, with the pulse and the
   steady arrival's own echoes taken out: 0.50 ms fitted over the first 0.9 ms (four
   quarters of the night: 0.27, 0.63, 0.53, 0.63), 0.96 ms over 3 ms. The rules, with the
   Cox-Munk slope of the 1.5 m/s wind (0.0105; the slope is not measured, and Cox-Munk
   is at the bottom of its range): facet 0.45 ms, rule of thumb 0.85 ms. Our model with the
   surface fitted to the tail (two numbers: height, speed), sampled and fitted like the
   data: facet reads 0.67 and 0.44 ms in the two windows, rule of thumb more than 4 and
   0.96 ms. So the first 0.9 ms decays as the facet option does and not as the rule of
   thumb does; the 3 ms window reads the rule of thumb's value, but there the later echoes
   of the surface arrival fluctuate too and our facet tail is cut at three decay times.
   Read the other way, the measured times are mean-square slopes of 0.012 to 0.022 under
   the facet rule and 0.006 to 0.012 under the rule of thumb: both believable for a light
   wind. The uncertainty of the measurement (a factor 2 between windows, 2.3 between
   quarters) and of the slope is as large as the factor 1.9 between the rules at this
   angle. What the data do show: a tail exists at Gamma^2 = 0.03, and its length is of the
   order mean-square slope x t1 t2 / (t1 + t2), as both rules have it.
5. **Multi-bounce rule, bubble loss, bottom reflection against named sediments: not
   testable.** No arrival with two surface bounces can be named; the seabed is not named
   (the echo 0.5 ms behind the direct arrival at R1 is 12 dB down where a flat half-space
   gives 1 dB: a hilltop under a tripod, not a sediment measurement).
6. **Delay spread: the model has nothing like it.** 90 % of the energy within 165 ms at
   274 m and 275 ms at 1020 m, 99 % within 465 and 636 ms; rms spread 100 and 145 ms.
   Bellhop over the charted section ends after 0.2 s and 98 to 99 % of its energy is in the
   first 50 ms. This is the reverberation of a basin with sides, not of our 2-D ocean.

*Noise* (the 2.5 to 3.5 s before each cycle's first transmission: 222 and 219 s in all).
R2: 43.9 dB re 1 uPa^2/Hz in 4-8 kHz (41.7 to 48.5 through the night), slope -3.5 dB per
octave; the simulator's Wenz-type curve has that level at 1.9 m/s of wind without shipping
(one fitted number) and a slope of -5.2. Tails heavier than Gaussian (q99.95/q50 = 6.0
against 3.3; `noise_fit` chooses the K family, shape 3.5): the clicks the article
attributes to the receiver platform. R1: 62.9 dB, slope +0.8, lines (3.7, 7.3 kHz),
Gaussian: the research station's machinery, 19 dB above the sea. Coherence between the two
hydrophones (1.2 km apart): 0.007, the estimator's floor.

*How much the decay rule matters for the long-range array* (`tools/decay_time_sensitivity.py`,
`docs/figures/decay_time_sensitivity.md`; the long-range scenarios ship without time
spreading, so it was added: the scenario's 5 m/s wind, each rule, two seeds of the taps;
one row of 32 hydrophones through the C++ chain). Detection SNR of the ship, rule of thumb
minus facet (regenerated at `e10b17f`): +0.2 dB (12 km, image-source paths, 6 pings), -0.3 dB
(12 km, Bellhop), -0.4 dB (3 km over gravel, 30 m), -0.9 dB (8 km over sand, 200 m); the two
seeds of one rule differ by 0.1 to 0.8 dB. Time spreading against none: -0.9 to +1.4 dB.
Track confirmation: the same ping under every rule in every case detected (the third). In
the first run (`f68cc4d`, before the detector rework) the image-source 12 km case sat at
threshold (13.5 dB, 3 pings of 6 without spreading): there 0.1 dB decided whether a ping was
detected, and the facet runs lost one detection and the track that the rule-of-thumb runs
kept. That was the margin of the design point, not a property of a rule; now it is detected
on 6 of 6 at 17.7 dB under every rule. Without noise or detector, the echo's strongest sample changes by
-0.08 +- 0.30 dB between the rules at 1 km (flank array, 6 m/s) and by 0.00 dB at 12 km
(grazing angles under a degree: the surface is a mirror for either rule). The Bellhop cases
were run with the 12 strongest arrivals, 3 taps and 3 pings in every variant (with 40
arrivals and 6 taps one run takes over an hour here); 4 km over mud in summer is not
detected in any variant. The rule is worth a few tenths of a dB: less than the seed, far
less than the profile or the seabed.

*What the other two bands would add.* The same surface within minutes at 11.5 and 28 kHz:
Gamma^2 3.7 and 22 times larger (0.1 and 0.7 at R1), so the coherent share against
frequency squared is a test with no free number once the height is fixed here, the Doppler
width leaves its wave-frequency limit, and the tail carries ten to twenty times the energy
with half the pulse width (9 kHz of band): the decay time would be measured, not bracketed.

## 22. The projector's direct sound and a receiver that overloads [OURS]

Not from TR 0702: SST's receiver is linear. This section is our own model, written for the
long-range array, whose projector sits beside the hydrophones. It is behavioural: it says
what a front end of a given description does, and makes no claim about any board. Code:
`sim/sources.py` (`direct_arrival`), `sim/receiver.py` (`apply_front_end`),
`tools/direct_sound_study.py`, `tests/python/test_direct_overload.py`.

### 22.1 Direct arrival (`transmit.direct_path`)

The pulse reaches hydrophone i straight from the projector, a distance `r_i` away:

    p_i(t) = (A_1m / r_i) 10^(-alpha r_i / 20) s(t - r_i / c),     A_1m = 10^((SL + sensitivity + gain) / 20)

`r_i` is each element's own distance (the projector is in the array's near field: on the
three-row face `r_i` runs from 1.69 m to 6.37 m, an 11.5 dB spread of level and 3.1 ms of
delay, nothing like a plane wave). It is the one-way source of Sec. 20 placed at the
projector's true position, moving with the vehicle (so no Doppler), emitting the sonar's own
waveform at the transmit instant; the same kernel as the echoes computes it, so element
patterns, absorption, board clocks, element position errors and sampled waveforms come with
it. The provider's other eigenrays between projector and array are included: with flat
boundaries the images of the projector in the surface and the bottom, at their own
distances, with the boundary's reflection coefficient. `{boundaries: false}` leaves them out.

A projector with a beam pattern: the one-way path treats the source as omnidirectional, so
the pattern is applied afterwards, per ray and per element, at the band centre, in the
direction the ray leaves the projector towards that element (the line from the apparent
source to the element, mirrored in the horizontal once per bounce). Exact for straight rays
and flat horizontal boundaries. A projector on a face with a half-space pattern facing out
has zero response in the plane of the face; real baffles leak and diffract, which is not
modelled: without a measured pattern, omnidirectional (the Popoto files) is the pessimistic
choice.

What runs where: analytic pulses, the C++ kernel (host or CUDA) on both engines; sampled
pulses, C++ or NumPy by engine; the eigenrays by the provider's engine.

Level convention: `A_1m` is the PEAK of the pulse in volts at the ADC, as everywhere in the
simulator. If a projector's source level is quoted rms, its pulse peaks 3 dB higher.

### 22.2 Overload and recovery (`receive.overload`)

Signals are in the record's units: volts at the ADC with all of `receive.gain_db` applied. A
limit `L` volts at the output of an input stage with `G_pre` of the total `G` dB is the
ADC-referred limit `L 10^((G - G_pre) / 20)`. The chain:

    x -> gain switch a(t) -> input stage (limit, dead time, gain recovery)
      -> coupling capacitor (one-pole high-pass) -> receive.tvg ramp -> ADC (limit, dead time, gain recovery)

**A limiting stage.** Input `u`, rails `-L_n`, `+L_p`, dead time `T_d`, recovery constant
`tau_g`; `tau` the time since the stage last clipped (the channel's own, or the latest on
its board when the stage is shared):

    u >= L_p or u <= -L_n :  y = the rail
    0 < tau <= T_d        :  y = the rail last hit (or 0)
    tau > T_d             :  y = g u,   g = 1 - exp(-(tau - T_d) / tau_g)

Whether a sample clips is decided on the input alone, so the stage has no feedback and the
curves are closed forms: the gain is within `x` dB of nominal `T_d + tau_g ln(1 / (1 - 10^(-x/20)))`
after the last clipped sample (2.22 `tau_g` for 1 dB).

Which stage clips first follows from where the gain is: with all the gain ahead of the
input stage's output, that stage clips first if its swing is below the ADC's full scale; with
gain behind it (and behind the capacitor) the ADC clips first.

**The coupling capacitor.** `v_C' = (y_1 - v_C) / tau_c`, `tau_c = 1 / (2 pi f_c)`; output
`y_2 = y_1 - v_C`. The capacitor follows the mean of what the input stage delivers:

- an unclipped or symmetrically clipped burst has zero mean and leaves no charge (only a
  ripple of order `L f_c / f`);
- a burst whose clipped output has mean `m` for a time `T` leaves `v_C = m (1 - exp(-T / tau_c))`.
  For a sine of amplitude `A` between unequal rails,
  `m = [h(L_p) - h(L_n)] / (2 pi)`, `h(L) = 2 A (1 - cos th) + L (pi - 2 th)`, `th = asin(L / A)`,
  which tends to `(L_p - L_n) / 2` when the burst is far over both rails (a square wave);
  a rail held through a dead time is a mean `L` for `T_d`;
- afterwards the output is the signal minus `v_C exp(-t / tau_c)`: an offset decaying with
  the corner's time constant (1.6 ms at 100 Hz, 16 ms at 10 Hz). Multiplied by the gain
  behind the capacitor it can exceed the ADC's full scale `F`; the ADC then sits on its rail for
  `tau_c ln(|v_C| / F)` and nothing is received.

The offset itself is far below the band and a matched filter rejects it; what it costs is
the time on the rail and the headroom until it has decayed. In samples the filter is the
bilinear transform with the corner pre-warped (unit gain in the band to 2e-4 at 5 kHz).

**Gain switching.** `a(t)`: `gain_db` until `until`, then linear in dB to 0 over `ramp`. It
acts ahead of the input stage, so a reduction larger than the overload removes the clipping
and with it every recovery effect; what it costs is the gain (and so the ADC's noise floor
relative to the signal) while it lasts. The record's existing ramp fields report it.

**State before the window.** A receive window usually opens after the pulse, but the front
end's state at that moment was set by the pulse. The model therefore runs from the transmit:
the direct arrival is synthesised for the interval before the window on the same sample
clock (same kernel, full fidelity), passed through the model, and discarded. Echoes,
reverberation and noise in that interval are not simulated: they are tens of dB below any
limit and change no state.

### 22.3 What it gives for the Popoto files (as configured; several inputs unconfirmed)

`tools/direct_sound_study.py` -> `docs/design/direct_sound_study.json`. With SL 197 dB,
sensitivity -208 dB re 1 V/uPa, 43 dB gain, 2.61 V full scale (all flagged unconfirmed in
the sonar files, as is every projector position): the direct sound is at full scale 15.3 m
from the projector. On the three-row face (projector 4 m from the centre) every hydrophone
is overloaded, by 19.1 dB at 1.69 m down to 7.6 dB at 6.37 m; on the static frame by 16.7 to
35.6 dB; on the test AUV by 20.6 to 34.0 dB. A front end that only clips costs nothing
beyond the pulse itself (minimum range `c (T + r / c) / 2`: 78 m and 753 m for 0.1 s and
1 s on the face). Each second of recovery adds 750 m.

### 22.4 Limits

- Behavioural, not a circuit: one dead time and one time constant per stage, a hard limiter
  with no soft knee, no slew limit, no thermal tail, no latch-up, no protection-diode
  conduction at the input (which can charge an input capacitor through a different time
  constant than it discharges: here the charging and the decay share `tau_c`), no
  intermodulation beyond what a hard limiter makes.
- A sigma-delta converter driven past full scale may go unstable and take its own time to
  come back; that is the ADC stage's `dead_time`, a parameter, not a model of the modulator.
- Clipping makes harmonics above half the sample rate; the limiter works on the samples, so
  those alias instead of being removed by an anti-alias filter that in a real board sits
  between the stages. They are out of the processing band either way.
- The decision "clipped" ignores the stage's own recovering gain (a stage at gain 0.3 clips
  at the same input as at gain 1).
- Real (passband) records only: a complex envelope cannot be clipped.
- Before the window only the direct arrival is simulated. A strong echo or boundary return
  before the window that overloads by itself is not seen (none does in the Popoto cases:
  the first surface return at 10 m depth is 26 dB below the direct sound at 1 m).
- Hull-borne (structural) paths from the projector to the hydrophones, and electrical
  crosstalk from the power amplifier, are not modelled; either can exceed the water path.
- No parameter of the Popoto board's overload behaviour is known to us. Dead times, time
  constants, rails and the gain split in the scenarios, tests and study are illustrative.

---

## 23. What the open literature confirms, contradicts or refines (2026-11 search) [OURS]

Register of sources, access and the list of documents still wanted: `docs/REFERENCES.md`.
Code: `sim/bottom.py`, `sim/surface.py`, `acoustics.py`, `sim/noise.py`,
`configs/sediments.yaml`; tests `tests/python/test_sediment_literature.py`,
`test_literature_models.py`. Everything added is opt-in; one table value was corrected
(22.1). TR 9407 was not read; "the published model" below means what open documents print
and attribute to it. `sim/timespread.py` was not changed (findings only, 22.4).

### 23.1 Sediments: the table is complete

Three open documents close the gap of Sec. 13 and of `ORG_ARL_SURVEY.md` Sec. 10:

- names to mean grain size: Cintron (2001), Table 1 (the CASS/GRAB table, 23 rows);
- roughness: `w2 = 0.00207 h^2` cm^4, `h = (2.03846 - 0.26923 Mz) / (1 + 0.076923 Mz)` for
  `-1 <= Mz < 5`, `h = 0.5` for `5 <= Mz <= 9` (Snellen et al. 2013, eq. (5)); exponent 3.25
  for all (Snellen et al. 2011);
- volume: `sigma2 = 0.002` below 5.5 phi, `0.001` from 5.5 phi (both papers).

Conversions, each tested. Spectral strength: the papers' spectrum is `W = (h0 K)^-gamma w2`
with `h0` = 1 cm and `w2` in cm^4, two-dimensional and isotropic, so
`w2 [m^(4-gamma)] = w2 [cm^4] 10^(2 gamma - 8)` (`bottom.w2_from_cm4`); the regression then
lands on the two values that arXiv:1809.08459 prints in metres (1.410e-4 and 1.638e-5) within
0.3 %. Volume parameter: the published volume cross section
`5 delta sigma2 |1 - R^2|^2 sin^2(t) / (nu ln10 |P|^2 Im P)` (Snellen et al. 2013, eq. (3)) is
our `volume_strength` to 1e-10 (the identity is `(1 - R) / (1 + R) = P / (rho sin t)`), so
`sigma2` is the volume cross section per unit volume over the attenuation in dB/m, as we
had it. Loss: `delta` is the ratio of imaginary to real wavenumber in every source.

Rock and pebbles are not fluids, and the literature does not fit the fluid model to them:
the published model uses an "empirical expression" for gravel and rock (not printed anywhere
open), and measurements on rock are summarised by Lambert's law, `mu sin^2(t)`: `mu` = -13 and
-18 dB (arXiv:1601.06078, 100 kHz), -23 dB (Gruber 2019, 200 kHz), -10 dB (from -13 dB at 45
degrees, Fezzani et al. 2025). The table carries -16 dB with that range for `rock` and -9 dB,
flagged unconfirmed, for `pebbles`. The handbook's own rock curves (-8 dB at 20 degrees) are
above the largest Lambert law energy conservation allows (-5 dB): so is our type law
`searay_bottom` for rock (-7.2 dB at 20 degrees).

Family table, default model, dB at 10 / 20 / 45 degrees
(`PYTHONPATH=python python3 tools/boundary_models_study.py --family-only`):

| Sediment (phi) | 6 kHz | 100 kHz | 300 kHz |
|---|---|---|---|
| rock (Lambert -16) | -31.2 / -25.3 / -19.0 | same | same |
| pebbles (Lambert -9) | -24.2 / -18.3 / -12.0 | same | same |
| sandy_gravel (-1) | -34.9 / -27.6 / -22.0 | -26.0 / -18.7 / -14.4 | -22.5 / -15.2 / -10.9 |
| coarse_sand (0.5) | -37.8 / -30.3 / -23.7 | -29.4 / -21.9 / -18.4 | -25.9 / -18.4 / -15.3 |
| medium_sand (1.5) | -41.2 / -33.1 / -26.7 | -33.3 / -25.3 / -22.5 | -29.9 / -21.8 / -19.7 |
| fine_sand (2.5) | -39.9 / -30.4 / -24.4 | -36.7 / -27.7 / -23.6 | -34.3 / -25.5 / -22.7 |
| very_fine_sand (3.5) | -35.7 / -27.1 / -24.6 | -35.1 / -26.8 / -24.5 | -34.4 / -26.4 / -24.2 |
| coarse_silt (4.5) | -32.0 / -27.9 / -25.0 | -32.0 / -27.8 / -25.0 | -31.9 / -27.8 / -24.9 |
| medium_silt (5.5) | -35.7 / -31.7 / -28.3 | -35.6 / -31.7 / -28.3 | -35.6 / -31.7 / -28.3 |
| clay (9) | -36.4 / -32.0 / -28.4 | -36.4 / -32.0 / -28.4 | -36.3 / -32.0 / -28.4 |
| type law sand / mud | -43.7 / -42.9 / -42.3; -44.2 flat | -33.7 / -29.1 / -27.1; -40.3 / -38.7 / -38.0 | -26.4 / -21.6 / -19.5; -31.0 / -28.6 / -27.6 |
| type law gravel | -34.8 / -27.9 / -24.9 | -25.4 / -18.2 / -15.2 | -21.7 / -14.4 / -11.4 |

Reading, honestly:

- At 100 and 300 kHz the family is ordered: the level falls 9 to 10 dB from sandy gravel to
  fine sand (roughness scattering), then the silts and clays sit on a plateau that the volume
  term sets, with a 3 dB step down at 5.5 phi where `sigma2` halves. Sandy gravel agrees with
  the type law for gravel within 1 dB, the sands with the type law for sand within 5 dB.
- At 6 kHz the order is not monotonic: the sands at 10 to 20 degrees are below the silts
  (-41 against -32 dB at 10 degrees), because below their critical angle little sound enters
  a sand and its roughness term has fallen as `f^0.75`, while a silt lets the sound in and
  scatters it from the volume at a level that does not depend on frequency. That is what the
  published defaults say; the published model's stated range is 10 to 100 kHz (Gallaudet
  2001), so 6 kHz is an extrapolation. The type laws for sand and mud are 10 dB or more below
  the sand and silt rows at 6 kHz (the gravel law agrees with sandy gravel there); they state
  no range either. Nothing opened measures bottom backscatter at 4 to 10 kHz.
- `medium_sand` is the one kink: 2 dB below its neighbours at 6 kHz and 45 degrees, because
  its source gives `sigma2` = 0.001 where the rule gives 0.002 (REFERENCES Sec. 11).
- Against measurements at 45 degrees (Fezzani et al. 2025, calibrated, 35 to 440 kHz): medium
  sand about -25 dB and flat above 70 kHz, ours -22.5 (100 kHz) and -19.7 dB (300 kHz), so 2.5
  to 5.5 dB high and rising where the data are flat; sandy mud and muddy very fine sand about
  -22 dB, ours -28.3 (`sandy_mud`) and -24.5 dB (`very_fine_sand`), so 2.5 to 6 dB low.
  Snellen et al. (2013) found at 300 kHz fitted `w2` twice and `sigma2` half the defaults;
  Gallaudet fitted `sigma2` = 0.003 and exponent 3.00 at 68 kHz on a silt-sand bottom
  (78 % fines, Mz 6.36 estimated from porosity; 23.2 has the comparison). Class
  defaults are good to about +-3 to 6 dB; above 100 kHz they overstate sand.

### 23.2 Bottom scattering models

| Ours | Published | Verdict |
|---|---|---|
| perturbation factor | Jackson's form (arXiv:2009.12420) | equal (1e-10), as before |
| volume term | Snellen et al. 2013, eq. (3) | equal (1e-10); the published model adds slope and shadowing corrections to it that we do not have |
| `composite`: tilted perturbation result averaged over large-scale slopes | Olson and Lyons, arXiv:2106.05909, eq. (44), after McDaniel and Gorman (1983) | same structure; their conditions: k x small-scale rms height < 1, grazing above about twice the large-scale rms slope angle |
| split at k x rms height = 0.5 (`small_scale_kh`) | Fonseca et al. 2025, eq. (6), after Novarini and Caruthers (1994): Rayleigh parameter of the small scale = 1, i.e. `4 k^2 zeta^2 = 1` | the same number (tested with both integrals) |
| `small_slope` | lowest-order small-slope approximation; validity in Olson (2026) | form as before; now with limits: k h up to order 1 (pressure release), about 2 to 3 (sand), about 11 (mud), rms slope 10 to 60 degrees by exponent |
| `combined` (small-slope averaged over large-scale slopes) | the published model switches: Kirchhoff from 40 to 90 degrees grazing, composite below 40, "appropriate interpolation" (Siemes et al. 2009) | not comparable formula to formula: the interpolation is not printed anywhere open. Olson (2026) notes the composite model "has largely been superseded by the SSA" |

Recommendation: none needed for the formulas. A default change is the owner's: `small_slope`
is what the current literature uses and it is valid wherever `perturbation` is.

*Against one printed measurement at 68 kHz (Gallaudet 2001; `test_gallaudet_boundaries.py`).*
What it cannot do first: the measurement does not tell the roughness models apart, and its
angular curves (data and the fitted model) are only figures (Fig. 15a), so the comparison is
with band means. The site is a silt-sand bottom (78 % fines, Mz 6.36 from porosity; the
source does not call it silty clay), 190-205 m deep. Printed numbers used: Table VI (p. 60),
the fitted model inputs rho 1.1483, nu 0.9864, delta 0.0033, sigma2 0.003, gamma 3.00,
w2 0.000518 cm^4; Ch. 6 Tables I-II (pp. 196-198), mean backscattering strength -22.6 dB at
48-66 degrees grazing and -28.7 dB at 29-37 degrees, raw values of the sonar equation (the
source puts the transmit-pattern bias at up to -9 dB, and both bands carry reverberation).
- With Table VI as the six inputs, `fluid_sediment` gives -22.8 dB (0.2 dB low) and
  -24.8 dB (3.9 dB high), the same for all five roughness models: for this slow (nu < 1),
  light sediment the roughness term is 20 to 30 dB below the volume term off the specular
  direction. The near-normal band (72-90 degrees, -19.4 dB printed) is not compared: our band
  mean there is -14.5 dB as a mean of intensity and -19.3 dB as a mean of dB, and the source
  does not say which it printed.
- Table VI's rho, nu and delta are what `sediment_from_grain_size(6.36)` gives (within
  4e-5), and w2 is our grain-size rule (0.0005175 cm^4): an independent second print of the
  class regressions. The fit differs from the class in gamma (3.00 against 3.25, no effect
  here) and sigma2 (0.003 against 0.001).
- The class row `fine_silt` (6-7 phi) gives -27.6 and -29.5 dB: 5.0 and 0.8 dB low, 4.8 dB
  below the site fit in both bands (the volume parameter). Inside the +-3 to 6 dB of 23.1.
Nothing here points to a defect in the bottom model; the 3.9 dB at 29-37 degrees is inside the
source's own systematic uncertainty.

### 23.3 Sea surface

*Backscatter.* The published high-frequency model is `10 log10(sigma_r + sigma_b)`, roughness
plus bubbles (Gallaudet 2001, pp. 124-125, summarising the handbook and McDaniel 1993):
`sigma_b = delta_r sin(t) / (8 pi delta) (1 + 8 b e^-2b - e^-4b)`, `b = beta_I / sin(t)`,
`delta = 2.55e-2 f^(1/3)`, `delta_r = 0.0136`, and a facet term near the normal. It says the
model agrees with data for 12 to 70 kHz and 3 to 15 m/s within +-5 dB below 8 m/s. Two things
follow that we did not have:

- a ceiling. Once the layer is opaque the level is `delta_r sin(t) / (8 pi delta)`, whatever
  the wind ("saturation ... verified by several experiments"): -27.9, -29.3 and -30.4 dB at
  10 degrees for 12, 30 and 68 kHz, at most 3.7 dB more on the way there. The empirical
  laws in the tree have no ceiling: at 68 kHz, 15 m/s and 10 degrees Chapman-Harris gives
  -19.1 dB and the Kraus law -23.0 dB, 7 to 11 dB above it. At 30 kHz and 10 m/s the three
  agree within 5 dB (-25.5, -28.3, -30.0 dB at 10 degrees).
- a consistency check of two sources. The module's bubble dynamics (Commander-Prosperetti,
  from arXiv:2410.14415) gives a total damping constant at resonance of 0.0544, 0.0797,
  0.1051, 0.1187 at 10, 30, 68, 100 kHz; the printed `2.55e-2 f^(1/3)` with f in kHz gives
  0.0549, 0.0792, 0.1041, 0.1184. And `delta_r` is `k a` at resonance: 0.01358 from
  Minnaert's radius. So the unit the source does not print is kHz, and neither constant is free.

Implemented as `surface.bubble_layer_surface_db` (scenario `model: bubble_layer_surface`),
with the optical depth `beta_I` taken from one of our loss laws (`extinction: apl_1994` or
`hall_novarini`), because the handbook's law for it is not printed anywhere open; without the
small-scale roughness term (its empirical function is not printed), so between about 30 and
70 degrees it is a lower bound.

*Bubble loss per bounce: is there data to arbitrate between `apl_1994` and our population
model?* Not in anything we could open. The one measurement paper found by title (attenuation
from near-surface bubbles, 1 to 20 kHz, J. Acoust. Soc. Am. 2008) sits behind a challenge
page. The backscatter model cannot arbitrate either: at 10 m/s and above both loss laws put
the layer near saturation, and at 5 m/s they differ by 3.6 dB (30 kHz) and 4.5 dB (68 kHz) in
predicted backscatter at 30 degrees, inside the model's stated +-5 dB. Patchiness: no opened source
quantifies it. The adopted law stays.

*Against one printed measurement at 68 kHz and 3 m/s (Gallaudet 2001, Ch. 6 Tables I-II,
pp. 196-198; `test_gallaudet_boundaries.py`).* The "lower bound between 30 and 70 degrees" is
not supported by it: `bubble_layer_surface_db` is ABOVE the printed band means, -38.1 dB
against -41.5 dB at 44-66 degrees grazing (3.4 dB) and -38.3 against -51.3 dB at 27-38
degrees (13.0 dB); with `extinction: hall_novarini` 12.6 and 22.2 dB above; chapman_harris
15.1 and 19.9 dB above, kraus_surface 10.7 and 15.8. At 3 m/s the bubble term is in its
sparse limit, proportional to the optical depth, which is our loss law's (0.0024 for
`apl_1994`), so this is the part of the model that the source does not print. Whether the
model or the measurement is off cannot be told from this data: the levels are raw sonar-
equation values with a transmit-pattern bias of up to -9 dB (which makes them low), the
averaging is not stated and the statistics are heavy-tailed (scintillation index 3.7 and 14),
the 27-38 degree band carries bottom reverberation, the site was dominated by the towship's
wake, and the one wind reading was 4.5 hours old. The source's own statement (p. 140) is that
its run-1 mean curve (Fig. 10, a figure) is within +-5 dB of the published model at 30-60
degrees of incidence. Bubble clouds from breaking ship waves peak at -23.1 to -25.2 dB
(Table III, p. 158), "within 1dB" of the saturation level (eq. (11)); ours reaches each within
1 dB between 27.5 and 48.4 degrees of grazing, 13 dB above the ambient model at 3 m/s.

*Forward loss, coherent reflection.* `exp(-Gamma^2 / 2)` is the Gaussian-height result;
no opened document gives measured coherent loss against the Rayleigh parameter (22.4).

*Wave statistics.* Pierson-Moskowitz: `S = a g^2 / w^5 exp(-b (g / (U w))^4)`, a = 8.1e-3,
b = 0.74, U at 19.5 m (Ead 2004, eq. (5.1)): our vertical speed is its second moment exactly;
our rms height uses significant height 0.21 U^2 / g where the constants give 0.2092 (0.4 %).
Cox-Munk: upwind 3.16e-3 U, crosswind 3e-3 + 1.92e-3 U at 12.5 m, 1 to 14 m/s
(arXiv:2210.05456, eq. (12)); the sum is 0.8 % below our total 0.003 + 0.00512 U. Added:
`acoustics.mean_square_slope_components`. The sea is steeper along the wind than across it
by 1.2 to 1.5; the time-spreading model takes it isotropic. Chapman-Harris's formula and
range could not be confirmed (no open copy).

### 23.4 Time, angle and Doppler spreading of forward scatter

The papers that measure this (Dahl 1996, 1999, 2001, 2004; Preisig and Deane 2004) could not
be opened: publisher challenge pages, no open copies listed. What was found:

- Doppler width against steepness. Dessalermos (2005) tabulates the Doppler spread of each
  resolved path at ten ranges (8 to 16 kHz probes, 30 m of water): it grows with a path's
  excess delay at every range, e.g. 0.89, 1.10, 2.00, 2.56, 2.85 Hz at 2300 m and 0.66,
  1.30, 2.06 Hz at 4350 m. Later paths are steeper and have more surface bounces. That is
  the behaviour of `2 f v sin(g) / c` with bounces adding, and the opposite of a width that
  goes as `1 / sin(g)` (TR 0702 p. 101), under which the first, most grazing arrivals would
  be the widest. The wind is not stated there, so it is a trend, not a test; the magnitudes
  (0.7 to 2.9 Hz at 12 kHz) are those of Sec. 21 (0.5 to 2.2 Hz at 12.5 kHz in 4 m/s).
- Model too coherent at low grazing (implied rms height 1.38 x Pierson-Moskowitz): the
  Pierson-Moskowitz constants are confirmed, so the factor is not a slip in our numbers. The
  spectrum is for a fully developed wind sea; swell adds height without vertical speed in
  proportion. No opened measurement of coherent reflection against the Rayleigh parameter.
- Scintillation index above 1 on multi-bounce arrivals: not addressed by anything opened
  for forward scatter. (For backscatter the literature does report indices above 1 that grow
  as the resolution cell shrinks and the grazing angle falls, 22.5; it is a different
  mechanism.)
- Facet decay time against the rule of thumb: nothing opened.

Recommendation: none from the literature beyond what Sec. 21 already proposes; the wanted
list names the pages that would settle each point.

### 23.5 Reverberation statistics

No published default for the K shape exists to adopt, and the literature says why. The
shape "is related to the effective number of scatterers within a resolution cell" (Abraham
and Lyons 2002, as quoted in arXiv:1901.08023), and the scintillation index of roughness
backscatter "increases as resolution increases, grazing angle decreases, and spectral
strength increases" (arXiv:2106.05909; 100 and 10 kHz). At 100 kHz and 3 cm resolution every
seabed type measured, rock, gravel, sand and mud, has an index well above 1, falling towards
1 as the resolution is reduced without reaching it (Lehman 2022); on rock a single K
distribution fails and mixtures are needed (arXiv:1901.08023). So `statistics.shape` must
stay a required input, tied to the sonar's cell. Abraham and Lyons (2002) is now read (Sec. 14.4):
it gives the rule from the number of scatterers per cell, which `statistics.scatterers_per_m2`
and `patches_per_m2` implement, but no density for any seabed, so the density is a required
input in the same way.

### 23.6 Targets, noise, medium

- Targets: no open copy of Urick's table of simple forms was found; the cylinder's obliquity
  factor stays `cos` as the Kirchhoff integral of arXiv:1805.06945 gives it. Wanted.
- Ambient noise spectrum: a third open print of the four-term fit (Hurt 2005, eqs.
  (2.7)-(2.9) and listing), with its attributions: wind to Wenz (1962), thermal and shipping
  to Urick (1983). Constants identical to ours. Its eq. (2.7) prints `7.5 w` where its own
  listing has `7.5 w^0.5`. Coates (1990), named in our docstring, was not opened.
- Spatial coherence: isotropic `sin(kd) / kd` confirmed as printed (Shajahan et al. 2022, eq.
  (3)). For surface-generated noise the Cron-Sherman model (their eq. (4), vertical pair:
  `2 [sin x / x + (cos x - 1) / x^2] + 2i [sin x / x^2 - cos x / x]`) is added as
  `noise.surface_noise_coherence` (any geometry, by quadrature; equal to the closed form to
  1e-10). Its real part first vanishes at 0.371 wavelengths against 0.5 for isotropic noise;
  the same paper shows it failing in shallow, range-dependent water (first zero off by up to
  30 %). The noise generator still uses the isotropic matrix.
- Impulsive noise: nothing new opened (alpha = 1.7 for the Singapore site model stays the
  only number, from the UACR).
- Absorption: our Francois-Garrison is within 0.6 % of 21 values tabulated for Arctic
  profiles (Col 2010, Table 1: 0.1, 1 and 10 kHz, surface to 3500 m, pH 7.63 to 7.95), and
  our Ainslie-McColm form is the one printed there (eq. (2.3)) and reproduces its two quoted
  relaxation frequencies (0.769 and 41.372 kHz). Stated validity ranges of both formulas
  were not read.
- Sound speed (Mackenzie, Marczak), Hamilton's regressions: not opened; unchanged.

## 24. The surface image in a link budget; own depth as a lever [OURS]

`python/gpusonar/lloyd.py`, `depthplan.py`, `target_classes.py`; `tools/popoto_depth_planner.py`,
`tools/surface_image_validation.py`. No source beyond the repository's own functions: the image
construction is derived in the module's docstring, the roughness factor is
`acoustics.rough_coherent_reflection`, the bubble law `sim.surface` `apl_1994`, the bottom
`sim.propagation.rayleigh_reflection`, the cylinder `sim.targets`.

*The factor.* One way, re the direct path: `H(f) = 1 + mu (R_d / R_s) exp(-i k (R_s - R_d))`,
`mu = -exp(-Gamma^2 / 2) 10^(-L_b / 20)`, plus a scattered power
`S = (1 - exp(-Gamma^2)) 10^(-L_b / 10) (R_d / R_s)^2` with no fixed phase. Two way, monostatic:
transfer function `H^2` (`H_tx H_rx` when the projector is not at the array's depth), mean
intensity `(|H|^2 + S)^2` because the surface has changed in the two-way travel time. Three
figures are reported: single frequency; energy over the band; peak of the compressed coherent
echo (the one a detection budget uses). With a bottom the sum runs over the layer's images.

*What the band does.* Nulls at `z1 z2 = n lambda R / 2`. The n = 0 null (the surface) is the same
at every frequency: a wide band does not fill it. The n-th null's phase spreads by
`2 pi n B / fc` over the band: at n = 1 with 4 to 6 kHz the energy is -3.7 dB and the peak -6.2 dB
where one frequency gives a null. Beyond `z1 z2 = lambda R fc / (2 B)` the pattern is gone: the
paths are resolved, the direct-surface and surface-direct echoes coincide (+6 dB peak), and the
energy is 1 + 4 + 1 (7.8 dB).

*Roughness at long range.* The grazing angle of the surface path is `(z1 + z2) / R`: 0.07 degrees
for 10 m and 4 m at 12 km, 1.1 degrees for 225 m. `Gamma = 2 k h sin(grazing)` is then small for any
plausible height: the surface is a mirror, and sea state enters through noise and bubbles, not
through roughness. The rms height is an input of its own (Sec. 21: measured heights did not follow
the wind law).

*Bubbles.* The `apl_1994` law goes as `1 / sin(grazing)` (floored at 2 degrees by `sim/surface.py`)
and gives about 5 dB a bounce at 9.8 m/s with that floor (11 dB without it at 1 degree): it weakens the surface path, so the null fills and the
lobe's gain is lost. No validity range is stated by its sources; a target inside the bubble layer
is not modelled. A bound, not a prediction.

*Measured* (`docs/design/surface_image_validation.json`): simulator image-source paths, both
engines, own depth 10 to 600 m, target 4 / 30 / 150 m, 12 and 3 km: within 0.8 dB (peak) and
1.1 dB (energy), largest in the nulls (the sweep's taper; the model's band is flat). Through the
C++ chain with noise: within 0.6 dB from -15 to +12 dB. Bellhop, one way at 12 km: surface image
only, energy within 0.6 dB (0.1 dB except in the 10 m / 4 m null, where the arrivals file's digits
limit Bellhop); 100 m over sand, the image series within 0.1 dB in energy and 0.2 dB in two-way
peak. The projector 1.5 m below the array (popoto_long_range) is 1.2 dB in a null.

*A finding on the way.* For the same target in free field the chain's detector reports 12.2 dB
less than the link budget's SNR. A 0.5 m range cell recovers 3.3 dB, a rectangular aperture
window 1.9 dB, both 5.1 dB; the simulator takes a source level as a peak pressure where the
budget is an energy equation (3 dB, by inspection, HARDCODED_AUDIT P-5); about 4 dB is not
explained.

*Not modelled.* Refraction in the closed form (Bellhop when the environment has a profile), the
scattered energy's spread in delay and angle (so `peak_db` leaves it out), a rough bottom, the
vertical offsets of stacked rows (the array is one depth), reverberation (margins are noise-limited).

## 25. Depth from multipath timing; rays in a layered profile [OURS]

Nothing in this section is taken from a source: the derivations are ours, checked against closed forms, an independent
Runge-Kutta integration, the simulator's image paths, and Bellhop run as a separate program (`tests/python/test_raytrace.py`,
`tests/python/test_multipath_depth.py`). Numbers are from `tools/multipath_depth_study.py` → `docs/design/multipath_depth_study.json`.

### 25.1 The two-way arrival set

Flat surface, straight rays, horizontal range r, target at depth z_t, the array at z_a and the projector at z_p
(a vertical lever arm; 1.5 m on the face). One way, from a point at depth z:

    R_d(z) = sqrt(r² + (z_t − z)²)        direct
    R_s(z) = sqrt(r² + (z_t + z)²)        by the surface (the image of the target at −z_t)
    τ(z)   = (R_s − R_d) / c  ≈  2 z z_t / (r c)            (exact form used; the approximation is good to 0.2% at 12 km)

The echo travels every outbound path with every return path, so there are four arrivals:

| Paths (out, back) | Delay after the first | Amplitude re the first |
| --- | --- | --- |
| direct, direct | 0 | 1 |
| direct, surface | τ(z_a) | μ_a R_d / R_s |
| surface, direct | τ(z_p) | μ_p R_d / R_s |
| surface, surface | τ(z_p) + τ(z_a) | μ_p μ_a (R_d / R_s)² |

μ = −ρ_c 10^(−L/20) is the surface's coefficient: −1 for a flat pressure-release surface (the echo is inverted),
ρ_c = exp(−Γ²/2), Γ = 2 k h sin(grazing) for rms height h, L a bubble loss in dB. With z_p = z_a the middle two
coincide and the echo is the triplet +1, −2, +1 at 0, τ, 2τ: the one-leg arrival is 6 dB above the direct one. At
12 km from 225 m to 150 m, τ = 3.75 ms; the 1.5 m lever arm splits the middle pair by 2 × 1.5 × z_t / (r c) = 25 µs,
an eighth of a carrier cycle, which the pattern carries. In shallow water each leg also has the bottom's images
(reflection coefficient: Rayleigh, at each image's grazing angle), and the two-way set is the product of the two legs'.

In the frequency domain the echo's transfer function is H(f) = Σ a_k exp(−i 2π f τ_k) = (1 + μ e^(−i2πfτ))² for the
triplet, |H|² = 16 sin⁴(π f τ): the Lloyd-mirror ripple of the echo's spectrum, period 1/τ = c r / (2 z_own z_t).

### 25.2 The estimator

The compressed output of the beam at the detection, y(t) (baseband, band B about f_c, noise power σ² per sample), is
A Σ_k a_k e^(−i2πf_cτ_k) χ(t − t_0 − τ_k) + noise, χ(τ) = sinc(Bτ) for a flat band. For a candidate depth z the
generalised likelihood ratio over the unknown complex amplitude A and the time t_0 of the direct arrival is

    T(z, t_0) = |Σ_k conj(a_k) e^(+i2πf_cτ_k) y(t_0 + τ_k)|² / Σ_jk a_j conj(a_k) e^(−i2πf_c(τ_j − τ_k)) sinc(B(τ_j − τ_k))
    L(z) = max over t_0 of T / σ²,            L_1 = |y|²_peak / σ²   (one arrival)

a rake over the predicted arrivals, normalised by the pattern's energy. It is one statistic for resolved arrivals and
for overlapping ones (where it matches the spectral ripple). σ² is the median of |y|² over 128 cells ahead of the
earliest possible arrival, over ln 2. The search over t_0 is anchored: the echo's strongest sample is one of the
pattern's strong arrivals, so only the alignments that put one there are tried (13 values of t_0 per strong arrival).

*Fringes.* L(z) oscillates with the carrier phase of the surface-surface arrival: one fringe per
Δz = r c / (2 f_c (z_a + z_p)) (4 m at 12 km from 225 m, 0.33 m at 1 km). The envelope of the fringes has the width of
one delay cell, r c / (2 B (z_a + z_p)) = f_c / B fringes (10 m at 12 km). Within a ping the fringe is real and gives
the fine depth; the Cramér-Rao bound for the pattern (Fisher information of s(f; z, t_0) with A projected out) is
centimetres at 25 dB, and the measured scatter is within the depth grid of it. Along a track the fringe envelope is
summed over pings (target depth and surface move by more than a fringe between pings). Sigma is the spread of exp(L)
over depth, so neighbouring fringes and envelope sidelobes widen it: near the threshold the error is at times a whole
delay cell (9 m at 12 km) and the sigma covers 94% within two sigma there.

*Decision.* A depth needs L(z) − L_1 ≥ 12. For the resolved triplet the pattern holds 6 units of energy against 4 in its
strongest arrival, so the gain is a third of the echo's total SNR: single-ping depth needs the strongest arrival at
about 15 dB (22% of pings at 11.9 dB, 82% at 14.7 dB, all at 17.6 dB, 12 km); a track of 8 pings at 11 dB gets it.

### 25.3 When it cannot work

- **Too shallow (surface cancellation).** For (τ_p + τ_a) B < about 3 the pattern is a smooth weighting of the band,
  and so is an absorption tilt (6 dB across 4-6 kHz over 24 km), a tapered replica or a target's own response: a Hann-
  weighted single echo is fitted by the pattern 33% of its energy better than by one flat arrival for τB up to 2, 24% at
  2.5, and worse from 3 on (computed;
  with τB ≥ 1 a 31 dB single echo passed as multipath in a third of pings). The search therefore starts at
  z_t = 3 r c / (4 B z_own): 30 m at 12 km from 225 m, 135 m from 50 m, 2.5 m at 1 km from 225 m. A ship at 2 to 6 m is
  never measurable at long range, and a sonar at 10 m measures nothing at 12 km (`merged`).
- **No surface path.** Bubbles at a high sea state (a loss per bounce), or roughness at steep grazing: at 1 km from
  225 m (21°) a 0.3 m rms surface leaves exp(−Γ²/2) ≈ 0, at 12 km (1.8°) it is a mirror. Told the roughness, the stage
  says `no_surface_path`; taken for flat, one of two pings at 3 km gave 76 m for 150 m with a sigma of 0.3 m.
- **Shallow water without the bottom in the pattern.** The surface-only pattern locked onto bottom echoes (99.7 m for
  70 m in 100 m of water); with the bottom's images, 70.1 m. Flagged when the water depth is known.
- **Extended targets.** Five highlights of one sign over 20 to 100 m of range, no surface: never taken for multipath
  (gain −14 to −193); under a surface the depth came out right. Highlights that happen to lie at 0, τ, 2τ with signs
  +, −, + would pass; a rigid body does not make them.
- **Refraction** (next).

### 25.4 Refraction: the delay is not profile-free

A constant gradient g makes every ray an arc of radius c / (g cos θ). To first order this is straight rays under a
surface that curves away by d_1 d_2 g / (2c) between points d_1 and d_2 from the reflection point, which raises both
effective depths; with d_1 d_2 = r² z_own z_t / (z_own + z_t)²,

    τ_refracted / τ_flat − 1 ≈ r² g / (2 c (z_own + z_t)).

Against the ray code in isothermal water (g = 0.017 /s), vehicle at 50 m, target at 120 m: 3.4% at 1 km (rule 3.3%),
32% at 3 km (30%), 143% at 6 km (120%); target at 20 m: 8.3%, 82%, 374%. The flat formula's depth is wrong by that
fraction: the multipath depth is less sensitive to the gradient than the elevation depth (r² g / 2c in metres) only by
the factor z_t / (z_own + z_t). The stage takes an assumed gradient out by this rule (120.1 m and 122.1 m for 120 m at
1 and 3 km, 141 m at 6 km where the first order no longer holds) and puts what is not known of it,
z_t r² σ_g / (2 c (z_own + z_t)), into the sigma. Through the stage-2 thermoclines the question is often moot: from
50 m, in the three profiles with a thermocline, there is no direct path to either target at 6 and 12 km, and at 3 km
in five of the six cases; and in the deep profile
(vehicle at 225 m) the flat depth of a 150 m target is −0.2%, −2.4%, −13.5% off at 1, 3, 6 km and there is no
surface-reflected path at 12 km. The right pattern in such water is the ray code's eigenrays through the ladder's
profile; that is not built into the stage.

### 25.5 Rays in a piecewise-linear profile

`python/gpusonar/raytrace.py`. In a layer with dc/dz = g, with s the sine of the grazing angle (positive going down)
and p = cos θ / c the Snell constant: from d(cos θ) = p g dz and dz = tan θ dr,

    ds/dr = −p g          (s is linear in range, through a turning point as well)
    c = sqrt(1 − s²) / p  (so z = z_1 + (c − c_1) / g)
    dt = dr / (c cos θ) = −ds / (g (1 − s²)),        t_b − t_a = (atanh s_a − atanh s_b) / g

and a ray is stepped from layer boundary to layer boundary in closed form, mirrored at the surface and bottom.
Eigenrays: a fan of launch angles, brackets of the receiver depth between neighbours with the same history of
reflections and turns (intervals whose ends differ in history are halved until they do not), bisection. Amplitude:
the power in cos θ_0 dθ_0 dφ crosses the area r dφ |∂z/∂θ_0| dθ_0 cos θ_r, so I/I_1m = cos θ_0 / (r |∂z/∂θ_0| cos θ_r)
(1/R² in isovelocity water); at a caustic the amplitude is clipped at ten times spherical and marked, and no caustic
phase is applied. Checked: isovelocity image times to 0.2 µs and amplitudes to 0.2%; the Lloyd pattern to 0.05 dB;
the circle of a constant-gradient ray to 1e-10 and its time to 1e-12; a thermocline with turns and bounces against
Runge-Kutta to 2 cm and 2 µs; Bellhop's arrival times to 20 µs and amplitudes to 1.5 dB away from caustics;
`elevation.ray_depth` (small Snell steps) to 2 m.


## 26. A basin's late reverberation, a trial in fjord water, the reach of the tail, a wandering coherent arrival [OURS]

Three additions that follow from the first comparison with measured data in the long-range
array's band (Sec. 21.6). Sources: the same dataset (SFI Smart Ocean, van Walree et al., IEEE
DataPort 2025, doi:10.21227/3aa6-4k33, CC BY 4.0; its descriptor article, read as
hal-05793814) through the soundings of Sec. 21.6, and for 26.3 the "Purple" channels of the
Underwater Acoustic Channel Repository (doi:10.5281/zenodo.21287414, CC BY 4.0), two files
fetched again. No formula or coefficient here is from a paper: the decay law is fitted, the
rest is derived below and tested against closed forms. Everything is opt-in; without the new
keys the eigenrays and records are bit for bit what they were (tested; the time-spread module
compared with its committed version over 128 combinations of its earlier options).

**26.1 Late reverberation of a basin** (`sim/basin.py`, key `basin_reverberation`;
`tools/basin_reverberation_fit.py`, `docs/figures/basin_reverberation_fit.md` and `.png`).

*The measured curves.* For each hydrophone the share of the received energy still to come,
against delay after the first arrival (median of 80 soundings with 0.745 s chirps), is a
straight line in dB from 10 ms down to where the noise correction stops being safe (-40 dB):

| Form, numbers fitted | R1, 274 m: rms / largest misfit, dB | R2, 1020 m |
| --- | --- | --- |
| exponential, 2 (energy, decay) | 0.59 / 1.33 to -30 dB; 0.81 / 2.14 to -40 dB; T = 0.145, 0.158 s | 0.55 / 1.57; 0.49 / 1.59; T = 0.160, 0.162 s |
| power law, 3 | 0.18 / 0.47; 0.33 / 0.79; exponent 11 to 14, scale 1.1 to 1.5 s | as the exponential: it converges to its exponential limit (exponent above 400) |
| two exponentials, 4 | 0.17 / 0.44; 0.24 / 0.51; T = 0.10 and 0.20 s, 0.83 : 0.17 | as the exponential |

An exponential describes both to about half a dB with the same decay at both ranges
(0.145 to 0.162 s; 27 to 30 dB per second; a "reverberation time" to -60 dB of 2.0 to 2.2 s):
the decay belongs to the basin, as a room's does. A power law with an exponent of 11 to 14 is
an exponential with a slight curvature; the gain of 0.4 dB at one range for one more number
does not justify it, and the law implemented is the exponential alone. Fitted from 10 ms to
-30 dB and read back as the simulator would produce it (the modelled arrivals as one arrival
at delay 0, the tail with `energy_db` re them):

| | R1, 274 m | R2, 1020 m |
| --- | --- | --- |
| `energy_db`, `onset_s`, `decay_s` | -4.2, 0.010, 0.145 | +0.2, 0.010, 0.160 |
| 90 % / 99 % / 99.9 % of the energy within, ms: model (measured) | 156 (165) / 489 (465) / 823 (866) | 272 (275) / 642 (636) / 1011 (1012) |
| rms delay spread, ms: model (measured) | 102 (100) | 143 (145) |
| tail's energy re the source's at 1 m (source level 183 dB) | -43.1 dB | -61.1 dB |
| share of the late energy that repeats over five chirps 3 s apart | 0.5 to 0.7 | 0.35 to 0.65 |

The tail holds 27 % of the energy at 274 m and 51 % at 1 km. Relative to the source it is
18 dB weaker at 1 km than at 274 m, more than spreading (11 dB): the late field is not that of a
uniformly filled room, and `energy_db` is given relative to the modelled arrivals at the
range in question, not as one level for the basin. The hydrophones were 1.2 km apart: nothing
was measured about coherence across an array.

*The model* (docstring of `sim/basin.py` for the full statement). A one-way response gets a
tail `x * g`: the modelled signal convolved with a kernel of independent Gaussian samples of
variance `E p(tau - onset)`, `p(u) = exp(-u / T) / T`. Closed forms: share later than t,
`exp(-(t - onset) / T)`; mean delay `onset + T`; rms spread `T`. Two-way: an echo becomes
`echo * (delta + g_out) * (delta + g_back)`. The term with `g_out` is the sound that reached
the target late and came straight back: it has the echo's own wavefront (the full array gain,
smeared in range), energy E. The terms with `g_back` arrive by way of the basin: diffuse,
energy `E + E^2` with the variance envelope `E p + E^2 (p * p)`. An echo's tail therefore
holds `2 E + E^2` of its energy (-0.4 dB at 274 m, +5.0 dB at 1 km). Diffuse: a sum of plane
waves (about 1.3 k D + 16 of them for an array of extent D), each with its own kernel, from
azimuths evenly round the horizon (`coherence_m: isotropic`: coherence `J0(k d)`) or from a
Gaussian sector of rms width `sqrt(2) / (k L)` about `bearing_deg` (coherence `exp(-(d / L)^2)`
across it), elevations Gaussian about the horizontal (10 degrees rms by default: an
assumption). `repeatable` is the share of the energy that is the same on every ping; the
frozen part belongs to places in the water, so an array that has moved sees it decorrelated
as `J0(k x)`. `own_ping_db` adds the basin's answer to the ping itself, the transmitted pulse
convolved with a diffuse kernel of the same law. THE DATA DO NOT GIVE THAT LEVEL: they were
taken 274 m and 1 km from a source, not at it. It exists as a what-if.

*Assumed, not measured:* the coherence across an array; the elevation spread; that the tail
of an echo is that of two one-way legs with the measured one-way numbers; that source and
target sit in the basin as the dataset's source and hydrophones did; no Doppler in the tail.

*Tests* (`tests/python/test_basin.py`): energy, energy shares later than four delays, mean
delay and rms spread of the generated tail against the law (complex baseband and real
records); `J0(k d)` to the first zero and the negative lobe; `exp(-(d / L)^2)` for a sector;
the elevation spread's vertical coherence; `2 E + E^2` with E at the target's bearing (full
array gain) and the rest diffuse; nothing before the onset; ping-to-ping correlation equal to
`repeatable`; `J0` for a moved array; the own-ping energy in a window against the pulse at
1 m and the law; a radiating source's tail; both engines; raw records with board clocks;
scenario without the key bit-identical.

**26.2 A trial in fjord or harbour water against the open-water rehearsal**
(`configs/scenarios/fjord_*_trial_*.yaml`, `tools/fjord_rehearsal.py`,
`docs/figures/fjord_rehearsal.md`; chain as on the branch at commit 9757c94, unchanged).

The three builds trialled first, each in its shipped open-water rehearsal and in the same
geometry in fjord-type water: a sound-speed maximum at 19 m (`basin.stratified_profile`:
five numbers of the shape of the site's casts, not the casts), basins and sills along the
line of sight (`basin.basin_section`: of the kind charted there) through Bellhop, the late
tail with the measured set for the range, and a calm night's heavy-tailed noise. Four pings
each through the C++ chain of the build's deployment file. All of the water is illustrative.

| Build, target | open water: SNR, dB | profile and section | + late tail | fjord as shipped (tail, calm K noise) | + own ping at -43 dB (what-if) |
| --- | --- | --- | --- | --- | --- |
| test AUV, one row; 10 dB at 500 m | 47.5; no other detections; track at ping 3 | 39.4; strongest arrival 19 m late; 2.5 / 1.8 other detections per ping (behind / elsewhere); 2 tracks off the target | 26.3; 3.8 / 20.2; 16 tracks off | 26.5; 3.0 / 46.5; 32 tracks off | not detected; 54 detections per ping, 46 tracks |
| the same at 300 m | 55.8 | 52.8 | 29.9; 5.5 / 31.8; 21 off | | not detected |
| at 1000 m | 32.3 | 31.0; 21 m late | 22.5; 6.0 / 0.2; 5 off | | not detected |
| at 2000 m | 16.3 | 28.6 | 24.4; 0.5 / 0; none off | | 24.2: no effect left |
| static, two rows; 15 dB at 2.6 km, 1 s pulse | 32.3 | 34.9; 41 m late: no direct path (a sill stands in it) | 15.0 | 15.1 | 15.0 |
| static, three rows; 10 dB at 150 m | 68.6 | 64.0 | 24.6 | 24.6 | 14.0 |

What the numbers say:

1. *The late tail caps the SNR the detector reports, whatever the echo's strength.* The cells
   the CFAR trains on behind the echo are filled by the echo's own tail. Per range cell the
   tail is `(2 E + E^2) / (B T)` of the echo at its onset (B the band, T the decay): 25 dB
   down for 2 kHz and the 274 m set, 20 dB for the 1 km set. Measured through the chain: 26,
   30, 22, 24 and 25 dB with the 274 m set (500 m, 300 m, 1 km, 2 km; three rows at 150 m)
   and 15 dB with the 1 km set (two rows, 1 s pulse), where the same water without the tail
   gave 39, 53, 31, 29, 64 and 35. A strong echo loses 5 to 40 dB of reported SNR and no
   detection; one below the cap loses nothing.
2. *The tail is clutter.* It starts at the echo and the CFAR's window ahead of it is quiet:
   20 to 47 detections per ping at other bearings behind a 26 to 30 dB echo, and 16 to 32
   confirmed tracks that are not the target. None is above 15 dB. `min_snr_db: 15` (9 as
   deployed) leaves 0.5 a ping and no track, with the target at its SNR; a short CFAR window
   (guard 30 m, training 100 m) does the same but costs 3 to 6 dB of the target's reported
   SNR. At 2.6 km with the 1 s pulse the target itself is at 15 dB: there `min_snr_db: 15`
   loses it on two pings of four, and the short window (12.4 dB, all four pings, no clutter)
   is the setting to use. Under machinery noise (63 dB re 1 uPa^2/Hz, flat, as by the
   research station) the clutter is hidden and the target keeps 24 dB.
3. *The profile and the seabed move the strongest arrival.* At 500 m and 1 km it is a
   bottom-reflected path 19 to 21 m late, 4 dB above the direct one (a concave basin over a
   reflecting half-space: that is the assumed seabed's doing); at 2.6 km with the array at
   30 m and the target at 60 m there is no direct path at all (the 53 m sill is in the line
   of sight) and the reported range is 28 to 41 m long. Expect the first trial's range to be
   off by tens of metres in such water, and several detections behind the target on its
   bearing (2 to 6 per ping here; the chain's multipath classifier is there for them).
4. *Depth needs the profile.* Three rows at 150 m: the echo arrives 1.5 degrees steeper than
   the straight line (-4.5 against -3.0). With the deployment file's sound-speed assumption
   the depth is 21.3 to 21.8 m for a target at 18.0 m, stated +-2.6 m: 1.3 to 1.5 sigma off,
   every ping the same way. With the profile given to the chain (`chain.elevation.ssp`):
   18.0 to 18.6 m. Two rows at 2.6 km give no depth in either water (one sigma 120 m, and in
   the fjord the arrival is a bounce, +0.9 to +2.4 degrees against -0.7).
5. *The own-ping what-if decides short ranges, if it is anything like that.* With the late
   field at the level it had 274 m from a source, a 10 dB target is not detected out to
   1 km with one row (whatever the settings), is 14 dB instead of 25 dB at 150 m with three
   rows, and is untouched at 2 km and beyond: that reverberation falls 27 to 30 dB a second.
   Nothing measured it at a source; the first ping of a trial will (listen with the
   projector on and no target: the level against time after the pulse is this number).

*Own depth against the maximum* (`tools/fjord_depth_plan.py`, `docs/figures/fjord_depth_plan.md`:
the planner of Sec. 24 with the illustrative profile over a FLAT 70 m seabed, a target of the
trial's strength anywhere between 5 and 40 m; Bellhop). Two-way factor re free field, worst
over the target's depths: one row at 500 m, +2.7 to +5.1 dB at every own depth from 5 to 59 m
(the same water at one sound speed: +3.2 to +5.0); three rows at 150 m, +2.3 to +5.1 (-1.5 to
+0.1 at one sound speed); two rows at 2.6 km, +7.3 to +9.6 from 5 to 29 m (best at 20 m, on
the maximum) and +5.8 to +7.7 below 30 m (one sound speed: +6.7 to +8.5 everywhere). At these
ranges over a reflecting bottom the depth relative to the maximum is worth 3 to 4 dB at
most, against margins of 30 to 75 dB: it is not what decides a trial. What does is the
seabed's shape, which the planner does not have: in the rehearsal the array hung at 30 m
loses the direct path to a target at 60 m behind a 53 m sill. Hang the array above the
shallowest ground between it and the target, and on or above the maximum rather than far
below it.

*What to expect in such water, and how to configure for it.*

- The reported range of a target is that of its strongest arrival, which can be a bounce 20
  to 40 m behind it; several detections follow it on its bearing. Judge range against a
  transponder or a surveyed target, not against the first ping.
- A strong echo reads 22 to 30 dB of SNR, not 40 to 70: its own reverberation fills the
  CFAR's window. That is not a fault of the array.
- Clutter at other bearings for half a second behind every strong echo, up to 15 dB:
  `chain.min_snr_db: 15` for short ranges with the 0.1 s pulse; for long pulses and weak
  targets `chain.cfar: {guard_m: 30, train_m: 100}` instead.
- The ping's own reverberation is not known. Measure it first: ping with no target and plot
  the level against time; it should fall 27 to 30 dB a second if the water is like the
  measured fjord. Until it has fallen below the expected echo, nothing is detectable:
  `chain.blanking` to that time, and plan the first targets beyond it (in the what-if, beyond
  1 km for a 10 dB target and one row).
- Depth from stacked rows needs the sound-speed profile: cast one and give it as
  `chain.elevation.ssp`. Without it expect 3 to 4 m of bias at 150 m (1.5 times the stated
  sigma). With two rows at kilometres, no depth.
- Noise near a quay or a vessel can be 20 dB above the open water's (63 against 44 dB re
  1 uPa^2/Hz measured at the two hydrophones): record a minute of it and fit it
  (`tools/fit_noise.py`, `receiver.ambient_noise: {model: fitted}`) rather than use the wind
  curve. Its clicks are heavy-tailed (K, shape 3.5): they did not raise false alarms here.
- Rehearse with `configs/scenarios/fjord_*_trial_*.yaml`, replacing the profile, the section
  and the tail's three numbers by the site's as soon as a cast, a chart and one sounding of
  a pinger exist (`tools/basin_reverberation_fit.py` shows the fit).

*What to expect and how to configure* is in the report of this package and belongs in
docs/DEPLOYMENT.md when the coordinator merges it.

**26.3 How far the taps of a bounce's tail reach** (`time_spread.tail_quantile`,
`sim/timespread.py` item 7; `tools/channel_smartocean.py tail`,
`docs/figures/facet_tail_smartocean.md`).

The facet shape `exp(-tau (a + b) / 4ab) I0(tau (a - b) / 4ab)` falls as
`exp(-tau / 2a) / sqrt(tau)`: at low grazing angles (b << a) it is a chi-square variate of one
degree of freedom scaled by a, with 8.3 % of its energy beyond three means (an exponential:
5.0 %). The taps spanned three means whatever the shape. With `tail_quantile: q` they span
the q-quantile of the gamma distribution with the tail's mean and variance (`a + b` and
`2 (a^2 + b^2)` for one facet bounce: exact for b -> 0 and for a = b; in between the share
actually covered is within 0.006 of q). For one rough bounce each tap is then one interval
of delay: its weight the shape integrated over the interval (Gauss-Legendre in `sqrt(tau)`,
which removes the singularity at the origin), its delay the interval's own mean, the last
interval taking everything later. Energy and mean excess delay are the shape's own to 1e-9
with no rescaling. Several bounces keep the earlier grid rule on the longer span. Cost: the
reach grows from 3 mean delays to 6.3-6.6 (2.1 to 2.2 times; 1.54 times for an exponential), so
the same tap spacing needs that many more taps (13 for today's 6), and the ray pairs of a
two-way echo between two bounce rays grow as (1 + taps)^2: 196 against 49. With `taps`
unchanged the cost is nil and each tap stands for 2.2 times the delay.

*Against the measured tail of R1's surface reflection* (4-8 kHz; the model sampled like the
soundings and passed through the same fits; two seeds; T in ms):

| | exponential fit, first 0.9 ms | facet-shape fit, 0.9 ms | exponential fit, 3 ms |
| --- | --- | --- | --- |
| measured | 0.50 | 1.08 | 0.96 |
| facet, three means (as shipped), 24 taps | 0.67 / 0.63 | 2.21 / 1.96 | 0.44 / 0.44 |
| facet, `tail_quantile: 0.9`, 24 taps | 0.39 / 0.39 | 0.53 / 0.50 | 0.29 / 0.27 |
| facet, `tail_quantile: 0.99`, 24 taps | 0.37 / 0.37 | 0.56 / 0.53 | 0.56 / 0.60 |
| facet, 0.99, 48 taps | 0.39 / 0.37 | 0.53 / 0.47 | 0.56 / 0.60 |
| facet, 0.999, 48 taps | 0.39 / 0.37 | 0.53 / 0.50 | 0.60 / 0.63 |
| rule of thumb, three means | above 4 | above 4 | 0.96 / 1.02 |
| rule of thumb, 0.99 | 1.54 / 1.45 | above 4 | 0.96 / 0.96 |

Both windows are not matched: the long window moves from 0.44 to 0.56-0.63 against 0.96, the
short one from 0.67 to 0.37-0.39 against 0.50. What changes is that the model is now wrong
the same way everywhere: model over measured was 1.34, 2.05 and 0.46 in the three fits (too
long early, too short late: a shape error), and is 0.74, 0.52 and 0.60 with the taps to
99 % (a scale error). One factor then brings all three within 20 %: a mean-square slope 1.4
to 1.9 times the Cox-Munk value assumed for a wind nobody measured (0.0105; the 1.5 m/s is at
the bottom of that law's range). The decay time put in (0.45 ms) was never fitted. The 0.9
point is worse than three means in the long window; 0.999 adds little to 0.99.

*Against the 12.5 kHz site* (Purple, first recording, the 60 m and 200 m files; 8 surface
arrivals other than the fitted one; two seeds; model A of Sec. 21.3; in / out and median
model over measured): tail decay time 3 / 5 and 1.06 as shipped, 3 / 5 and 1.14 with the
taps to 99 %, 4 / 4 and 1.06 with 48 taps; mean excess delay 5 / 3 and 1.13, 6 / 2 and 1.03,
7 / 1 and 1.04. With the rule of thumb (model C): decay 4 / 3 and 2.30, then 3 / 5 and 1.92;
mean excess 2 / 6 and 1.45, then 4 / 4 and 1.32. A small gain where the tail was already
right on average; no harm.

*Proposed default (not applied):* `tail_quantile: 0.99` with `taps: 12`. Before / after: the
three Smart Ocean fits 1.34 / 2.05 / 0.46 of measured, then 0.74 / 0.52 / 0.60; Purple's
mean excess delay inside the tolerance for 5, then 7, of 8 arrivals. Cost: four times the
ray pairs between two bounce rays. For the long-range scenarios Sec. 21.6 found the decay
rule worth a few tenths of a dB; the reach will be worth less. It matters where a tail is
looked at: channel studies, communications, anything that resolves a millisecond.

**26.4 Slow wander of the coherent surface arrival** (`time_spread.coherent_phase`,
`sim/timespread.py` item 8, class `CoherentPhase`; `tools/coherent_phase_study.py`,
`docs/figures/coherent_phase_study.md`).

*The physics.* The coherent part of a surface bounce is the mean field, `exp(-Gamma^2 / 2)`,
and was perfectly steady. Split the surface's displacement into long waves and the rest. The
long waves move the reflecting patch bodily: the path changes by `-2 zeta sin g`, `zeta`
their displacement at the specular point (towards the water), a phase `2 k zeta sin g`
proportional to frequency. Averaged over the rest only, the reflection is
`exp(-Gamma_S^2 / 2) exp(2 i k zeta_L sin g)`, `Gamma_S` from the rest's height
`sqrt(h^2 - h_L^2)`; its long-time mean is the old coherent factor, and the energy of the
arrival with its tail is unchanged. Whether a wave of wavenumber K moves the patch "bodily"
is a matter of the Fresnel zone. First-order reflection from a sinusoidal mirror (the
stationary-phase integral over the delay ellipses of Sec. 16's geometry) gives the factor
`1 + i z cos(psi) exp(i Theta)`, `z = 2 k A sin g`,
`Theta = rho K^2 (cos^2 a / sin^2 g + sin^2 a) / (2 k)`, `rho = r1 r2 / (r1 + r2)`, a the angle
between the wave and the path. `Theta` is the phase by which the wave's first diffraction
orders lag the specular reflection. For small `Theta` the wave is a change of path length
(`z cos Theta`) with a little change of amplitude in step with it (`z sin Theta`: focusing
by the wave's curvature); for large `Theta` the orders are separate later arrivals, which is
the tail. The option applies the path-length part as a change of the ray's delay with its
rate (so the phase is proportional to frequency, continuous through a ping and from ping to
ping), the amplitude part at the centre frequency with the mean energy kept, and leaves waves
with `Theta` above `fresnel_limit` (1 rad: ours) to the scattering part. The tail keeps
`rms_vertical_speed` as its own rate; the peak has the waves' period.

*Against the measurement.* With the measured unspread height (0.8 cm) and a line at 0.2 Hz:
phase variance 0.010 rad^2 (measured 0.0084; the height was read from the same arrival's
energy, so this is consistency, not a test). The half-band estimator of Sec. 21.6 reads
1.42 on the model as it does for any pure change of path length (measured 1.57 +- 0.07).
Not fitted: for a 0.2 Hz swell running along the path `Theta` is 0.63 rad at R1, an
amplitude variance `tan^2 Theta` = 0.52 of the phase variance; measured 0.0037 / 0.0084 =
0.44. For crests along the path it would be 0.001. So the measured mix of phase and
amplitude is what a swell along that path gives; a source heaving on its platform gives phase
only. The two still cannot be told apart from one hydrophone, but the amplitude share is
now a prediction that distinguishes them, and it sides with the surface.

*Tests* (`tests/python/test_coherent_phase.py`): phase variance `(2 k h sin g)^2`; the line's
frequency and width, and a two-band table spectrum (3 : 1) recovered from the phase; the
Fresnel split (`cos^2`, `sin^2`, in step, waves beyond the limit absent); the coherent ray's
energy `exp(-Gamma_S^2)` and the total unchanged to 1e-9; the tail's Doppler width from the
rest's height; the half-band ratio 1.42 through the dataset's own estimator; a 12 km path
unchanged bit for bit; and through both engines: a 1 s tone by way of the surface both ways
against a 5 s swell of 5 cm, received phase equal to twice `2 k zeta sin g` at the time the
sound was at the target, within the pulse to the curvature the linear law leaves (at most
0.3 rad of a swing of +-1.3 rad) and from ping to ping to 8 % of the swing.

*What it does to the long-range chain* (5 kHz; the echo out and back by the surface, twice the
one-way phase; `docs/figures/coherent_phase_study.md`; computed from the phase law over an
hour of surface, not through the kernel):

| Path | sea | rms phase, rad | 1 s pulse: loss with no Doppler in the replica / after the best shift, dB | 0.1 s pulse, dB | four pings 5 s apart added coherently: loss, dB |
| --- | --- | --- | --- | --- | --- |
| 12 km, 10 m to 4 m, grazing 0.1 degrees | any of those tried, to sea state 4 | 0.00 to 0.02 | 0.00 / 0.00 | 0.00 | 0.00 |
| 2.6 km, 30 m to 60 m, 2.0 degrees | swell 5 cm, 8 s, crests along the path | 0.14 | 0.00 / 0.00 | 0.00 | 0.08 |
| | swell 25 cm, 10 s, crests along | 0.71 | 0.07 / 0.00 | 0.00 | 1.9 |
| | sea state 4, all 51 cm taken as large-scale, crests along | 1.39 | 0.88 / 0.07 | 0.01 | 2.1 |
| 500 m, 10 m to 15 m, 2.9 degrees | swell 5 cm, crests along | 0.21 | 0.01 / 0.00 | 0.00 | 0.18 |
| | swell 25 cm, crests along | 1.03 | 0.15 / 0.00 | 0.00 | 3.0 |
| | sea state 4, crests along | 2.14 | 1.75 / 0.07 | 0.03 | 3.8 |
| 274 m, 20 m to 43 m, 12.9 degrees | fjord, calm night (0.8 cm, 5 s) | 0.10 to 0.15 | 0.01 / 0.00 | 0.00 | 0.01 to 0.03 |
| | swell 25 cm | 4.6 | 2.0 / 0.02 | 0.03 | 5.7 (incoherent: 6.0) |
| | sea state 4 | 7.1 to 9.6 | 5.7 to 7.2 / 1.1 to 1.6 | 0.25 to 0.47 | 5.9 to 6.4 |
| 500 m and 2.6 km | any sea, waves running ALONG the path | 0 (0.18 for the 10 s swell at 500 m, 6 % of which takes part) | 0.00 / 0.00 | 0.00 | 0.00 (0.17) (Fresnel angle 1.9 to 7000 rad: they scatter) |

So: (1) at 12 km the surface path's coherent arrival does not wander in any sea tried. Its
Fresnel zone is kilometres long; only a swell with its crests along the path and longer
than about 8 s moves it as one, and then by at most 0.02 rad because the grazing angle is
0.1 degrees. A 1 s pulse integrates fully, and ping-to-ping coherence over the surface path is
not limited by the surface's heave. (2) At trial ranges with grazing angles of 2 to 3
degrees a 1 s chirp loses under 0.1 dB (the wander within a second is a Doppler shift of
0.02 to 0.4 Hz rms, which a chirp turns into a range offset); a 1 s tone against a replica
with no Doppler loses up to 1.8 dB in sea state 4. (3) Coherent addition of pings 5 s apart
loses 2 to 4 dB at 500 m to 2.6 km in a 25 cm swell or sea state 4, and everything at steep
angles: do not plan on multi-ping coherent gain over surface paths at short range. (4) The
wind-sea rows take all of the wind law's height as large-scale: an upper bound. (5) The
simulator carries the delay and its rate at mid-pulse: the "best shift" column is what its
samples leave out, at most 0.1 dB except on the steep path in a wind sea (0.7 to 1.6 dB).

*Limits.* First order in the amplitude part. Deep-water dispersion. One time per ping for
both legs (the surface is met within the travel time to the target of that instant, which
is taken as small against the wave period: at 12 km it is not, and there the option does
nothing anyway). Linear phase within a ping. The source's own heave is not this option.

## 27. Sound speed that changes with range [OURS]

The eigenray tier had one sound-speed profile for the whole section. A live chain has one too: the
cast taken at the vehicle. To test that chain against water it has not measured, the simulated
water can now change along the path (`environment.profiles_by_range`, docs/CONFIGURATION.md
"Water that changes with range"; `sim/bellhop.py`).

### 27.1 What is modelled

Profiles c_j(z) on one depth grid at ranges r_0 = 0 < r_1 < ... The field is Bellhop's own
2-D option (SSP type `Q`): on each quadrilateral of the range-depth grid the speed is linear in
depth along each of the two profiles and linear in range between them,

    c(r, z) = (1 - s) c_j(z) + s c_j+1(z),     s = (r - r_j) / (r_j+1 - r_j),

and Bellhop traces its rays through that with the range derivative in the ray equations. Nothing
is approximated on our side: the provider writes the `.ssp` file (the number of profiles, their
ranges in km, one row of speeds per depth of the `.env` profile; Acoustics Toolbox
`doc/RangeDepSSPFile.htm`, reader `Bellhop/sspMod.f90` subroutine `Quad`) and reads arrivals as
before. Bellhop stops where a ray leaves the ranges given and traces to its box on both sides of
the source, so the first profile is repeated before the source and the last one beyond the box:
the end profiles hold outside what was given. With one profile the `.env` is written exactly as
before and no `.ssp` exists (the files and the first-ping records of four shipped Bellhop
scenarios were compared before and after: identical).

*What range is.* The provider reduces geometry to (horizontal range from the source of the call,
depth) and uses one section on every bearing: that is how a bathymetry was already defined, and
the water follows it. So `range_m` is horizontal distance from the sonar; the water is rings
about the sonar. For one target on one bearing this is the vertical section through sonar and
target, which is the honest 2-D statement. It is not a front lying across a map: two targets on
different bearings see the same water at the same range, and the water moves with a moving
vehicle (9 m over the four pings of the test-AUV scenario, against a front 100 m wide). The
return leg is traced from the array's reference point and travelled backwards; rays in still
water are reversible, so this stays exact, with the section centred on the array instead of the
projector (about 1 m apart in the static builds).

*The local profile* is the first one, at range 0. The record carries one sound speed, as it
always has (no schema change): the local profile at the sonar's depth, which is what the vehicle
would measure. Everything in the simulator that needs "the" sound speed without a path uses the
local profile (the table in docs/CONFIGURATION.md lists each user): the conversion of an
eigenray's delay to an apparent position, the boundary speeds of time spreading, the
reverberation strength laws, noise coherence. Two of these are approximations worth knowing.
Absorption is one water (temperature, salinity, pH) over a path length of delay times the local
speed: the profiles carry speeds, not temperatures, and a speed 1.5 % off along half the path
changes 1.7 dB of two-way absorption (2.6 km, 5 kHz) by 0.01 dB. The provider's rough-surface
loss finds the surface grazing angle by Snell's law from the launch angle through the local
profile, which does not hold where the speed changes along the ray; at grazing angles of a few
degrees a 1 % change in speed is several degrees. The shipped scenarios therefore reflect the
eigenrays from a flat surface. Tiers without a profile (image source, free field) use the one
local speed everywhere and say, in a warning, that the change with range is not simulated.

### 27.2 Checks against closed forms

`tests/python/test_ssp_range.py`, with Bellhop as a separate program.

*The same water given twice.* A thermocline given as one profile, and as two identical profiles
1.5 km apart: the six strongest arrivals at nine receivers (20 to 90 m, 0.3 to 2.6 km) have the
same reflections, delays within 0.6 microseconds and amplitudes within 0.3 %.

*A constant gradient in range.* Two isovelocity profiles, 1480 m/s at the source and 1520 m/s at
2 km: c = c_0 + g r with g = 0.02 /s, the same at every depth. In a medium with a constant
gradient every ray is an arc of a circle whose centre lies where the speed would be zero (here
the vertical line r = -c_0 / g), and the travel time between two points a distance d apart, with
speeds c_a and c_b, is

    t = acosh(1 + g^2 d^2 / (2 c_a c_b)) / g          (along the gradient: t = ln(c_b / c_a) / g).

Bellhop's direct arrival agrees to 0.05 to 0.23 microseconds along the gradient and to 2 to 10
microseconds off it (receivers 150 to 200 m above and below the source, 0.4 to 1.9 km), and its
launch angle is that of the circle through the two points to 0.05 degrees. The formula was not
taken on trust: the project's own layered ray tracer (`raytrace.py`, Sec. 25), given the same
problem turned by 90 degrees (the gradient in depth, the two points one above the other and
offset sideways), returns the same times to 0.1 microsecond. A straight line at the local speed,
which is all a processor without this knowledge has, is 0.7 ms (0.4 km) to 16 ms (1.9 km) late.

*A front.* 1500 m/s to 900 m, 1470 m/s from 1100 m, linear between; source and receiver at one
depth, so the ray meets the front at right angles and does not bend:
t = 900 / c_1 + 200 ln(c_1 / c_2) / (c_1 - c_2) + (r - 1100) / c_2. Bellhop agrees within 2
microseconds at 0.6, 1.0 (inside the front), 2 and 3 km; the local speed alone would be 27 ms
(41 m) early at 3 km. The provider's own correction from a grid node to a field point uses the
speed at the point's range (1470 m/s there, not the local 1500).

*A limit of Bellhop found on the way.* Bellhop interpolates the delay linearly between ray steps
(a tenth of the water depth unless set). Where the speed changes along the ray the true delay is
curved in range and linear interpolation is out by up to h^2 |dc/dr| / (8 c^2): with 60 m steps
in the front above (0.15 /s) that is 30 microseconds, and 27.6 were seen (which is why the step
is taken to be the cause; Bellhop's interpolation code was not read for this); with the 10 m steps of 100 m water
and the shipped front (0.11 /s) it is 0.6 microseconds. Deep water with a sharp front would need
the step set.

### 27.3 A chain that knows only the local profile

`tools/ssp_range_study.py`, `docs/figures/ssp_range_study.md` (commit 1b313d0). Three
ILLUSTRATIVE scenarios (`configs/scenarios/ssp_*`), each simulated with its water as written and
again with its local profile at every range (same seed, target, noise), four pings each, through
the C++ chain of the build's deployment file as shipped and again with the local profile given
to the elevation stage and multipath depth switched on. Four pings and one seed: these are
observations, not rates.

| Scenario | Quantity | water changes with range | local profile everywhere |
| --- | --- | --- | --- |
| Two-row static, front 1.2 to 1.4 km out (22 m/s slower at the surface, 6.4 m/s below 35 m), target 2.6 km, 60 m | detected; SNR | 4/4; 45.7 dB | 4/4; 45.5 dB |
| | range of the first arrival, minus the straight-line distance | +21.5 m | +10.2 m |
| | bearing error; track confirmed | -0.07 degrees; ping 3 | -0.08 degrees; ping 3 |
| | elevation; depth as deployed; depth with the local profile (true 60 m; stated sigma 124 to 125 m) | -4.2 degrees; 12.8 m; 83.1 m | +3.7 degrees; 75.1 m; 80.4 m |
| Three-row static, mixed layer 15 m at the array, 45 m at 3 km, target 2.6 km, 60 m | detected; SNR | 4/4; 42.6 dB | 4/4; 43.7 dB |
| | range of the first arrival | +14.2 m | +10.2 m |
| | bearing error; track on the target | +1.02 degrees on 3 pings of 4; none in 4 pings (a track 360 m short instead) | -0.06 degrees; ping 3 |
| | elevation; depth as deployed; with the local profile (true 60 m; sigma 122 to 124 m) | -31.1 degrees; 19.3 m; 19.0 m | -5.4 degrees; 67.7 m; 56.9 m |
| Test AUV, one row, front 200 to 300 m out, target 500 m, 15 m, water 30 m | detected; SNR | 4/4; 57.1 dB | 4/4; 51.5 dB |
| | range of the first arrival; bearing error; track | +7.0 m; -0.18 degrees; ping 3 | +7.8 m; -0.32 degrees; ping 3 |
| | depth from multipath timing (true 15 m) | 30.3 m on all 4 pings, sigma 3.5 m, note `bottom_unmodelled` | none (`single_arrival`) |

What can be said from this:

1. *Detection, and bearing where no elevation correction was applied, did not care.* Every ping
   detected in both waters; SNR within 1.1 dB at 2.6 km (5.6 dB higher behind the front at 500 m,
   where the arrivals that interfere are different ones); bearing changed by 0.01 degrees (two
   rows, 2.6 km) and 0.14 degrees (one row, 500 m).
2. *Range moved by what the sound did.* The first arrival came 11.3 m later behind the front and
   4.0 m later under the deepening layer (0.43 % and 0.15 % of 2.6 km). Bellhop's own arrivals,
   turned into range at the local speed, say +21.6 to +22.1 m against +10.8 m and +14.7 against
   +10.7 m: the chain reports the delay it is given. Even in the local water the first arrival is
   10 m long, because in this water (the array inside a downward-refracting thermocline) no ray
   reaches the target without the bottom: the earliest paths have one bottom reflection. Part of
   the front's 11 m is slower water and part is a different path (one surface and one bottom
   reflection in place of one bottom reflection); the two were not separated.
3. *Depth from stacked rows was already governed by which arrivals reach the array, and the change
   of water reshuffled them.* With a correct local profile the two-row depth was 21 m out in the
   local water and 23 m out behind the front; without it, 15 m and 47 m. All inside the stated one
   sigma of 122 to 125 m, which is the gradient prior at this range and says the chain does not
   claim to know. The strongest arrivals come in at 1.4 to 4.2 degrees above and below horizontal
   in near-equal pairs, and their mix differs between the two waters.
4. *The three-row chain took an elevation of -31 degrees under the deepening layer*, on all four
   pings (one flagged ambiguous with an alternative at +2.0 degrees). The simulated sound has no
   arrival steeper than 20.5 degrees (the twelve kept). The horizontal-bearing correction then moved the
   bearing by +1.1 degrees on the three pings not flagged (the cone bearing itself was 0.08 degrees
   out) and the track was placed at 2239 m, labelled multipath, so no track on the target was confirmed. In the local water the
   same chain gave -5.4 degrees and a track at ping 3. Why -31 was preferred (a second lobe of
   three rows is the obvious candidate) was NOT investigated here: the depth estimators are being
   changed elsewhere, and this is one seed.
5. *Multipath depth gave nothing at 2.6 km* in either water (`single_arrival`: with no direct
   path the pattern it looks for is not there), and on the test AUV gave 30.3 m, the water depth,
   for a target at 15 m, with a sigma of 3.5 m and its own note that a bottom path is not in its
   model. The front changed whether it answered, not whether it was right: in this water it has
   no direct arrival to stand on.

*Limits.* Illustrative profiles; flat seabed; flat surface for the eigenrays; twelve arrivals
kept per point, the weakest of them only 6 to 11 dB below the strongest, so later arrivals are
left out in both waters; one target, one seed, four pings. The 2-D section is the same on every bearing
and moves with the vehicle. No horizontal refraction (a ray crossing a front obliquely on a map
bends sideways: bearing errors from that are not in an N x 2-D model at all). The water does not
change in time. Absorption, the boundary models and the rough-surface loss use the local water.

## 28. One ray calculation for both depths, the coverage picture and the ensemble [OURS]

Continues Sec. 25 (depth from multipath timing; rays in a layered profile).

**The fan.** The chain's ray code (`libs/live` `raytrace.hpp`, the C++ counterpart of Sec. 25.5's reference, equal to it
to rounding) shoots one fan of rays from the sonar's depth and reads it at the ranges wanted. At a fixed range the rays
with one reflection history end on one sheet of the wavefront, on which the travel time's gradient is the slowness
vector:

    dt/dz = s / c        (s: sine of the ray's angle at its end, down positive)
    dt/dr = p            (p = cos(theta) / c, the ray's Snell constant)

So an arrival at any depth between two neighbouring rays follows from a cubic Hermite polynomial in z through the two
rays' times with these slopes, and an arrival at a range between two readings of the fan from the same in r. In
isovelocity water the times are the image solution's to nanoseconds and the amplitudes 1/R to 0.2% (test); in the
study's refracting waters they are the bisected eigenrays' to 0.1 microsecond. Where neighbours differ in history the
launch angle between them is halved until each part has one history, so a sheet reaches the boundary it ends on (a
shallow target far away sits between the last direct ray and the first reflected one). Amplitude is the ray tube's,
I = cos(theta_0) / (r |dz/dtheta_0| cos(theta_r)), clipped at a caustic.

**The multipath pattern through rays.** For a candidate depth z and the detection's two-way time, the horizontal range
is the one at which the direct path's two-way time is the detection's (Newton, dt/dr = p_tx + p_rx); the legs' arrivals
there give the pattern: delay (t_tx,j - t_tx,0) + (t_rx,k - t_rx,0), amplitude a_tx,j a_rx,k / (a_tx,0 a_rx,0), with
the surface's coefficient, roughness and bubble loss at each ray's own grazing angle at the surface (Snell:
cos(g) = p c(0)) and the bottom's Rayleigh coefficient at its angle there. A change of the two-way time within the
search moves each delay by (p_j - p_0) dr, exactly to first order. A depth with no direct or no once-surface-reflected
path on either leg has no pattern and is not searched.

**What the profile does not know.** The rung's claim is one tilt about the vehicle (Sec. 22, environment.hpp):
c(z) +- min(sigma(z), sigma_g |z - z_own|). The depth sigma is the sensitivity to it: the depth at which the tilted
profiles give the same both-ways surface delay, a twentieth of the tilt each way, scaled back. With nothing known this
reproduces the first-order rule of Sec. 25.4, z r^2 sigma_g / (2 c (z_own + z)), within 12% from 3 km (test).

**The ensemble.** A single tilt cannot stand for a thermocline the vehicle has not seen. The ensemble continues the
unsampled water with each configured gradient over each configured extent, every pairing above and below, and reports
the span of the answers. It is a set of hypotheses, not a distribution: the span has no probability attached and covers
the truth only as far as the hypotheses do.

**The coverage statistic.** At a target position the two-way arrivals (all pairs of the legs' arrivals) go through the
compressed pulse, a sinc of the band B:

    P(t) = | sum_pairs' a_j a_k exp(-i 2 pi fc tau_jk) sinc(B (t - tau_jk)) |^2  +  sum_rest |a_j a_k|^2 sinc^2(B (t - tau_jk))

The first sum is over pairs whose two legs are each the direct or the once-surface-reflected path: their interference
is the Lloyd mirror and is deterministic. Everything that touched the bottom, or the surface twice, and the part of a
surface reflection that roughness removes from the coherent path (1 - exp(-Gamma^2)), has no phase one could predict
at a wavelength of 0.3 m and adds as power. The factor on the free-field echo is max_t P(t) (R_tx R_rx)^2. In isovelocity
water with a flat surface this is `lloyd.py`'s `peak_db` (within 0.13 dB over 4 to 300 m targets and 1 to 13 km; test);
with a rough surface it lies between `peak_db`, which leaves the scattered energy out, and `energy_db`, which counts all
of it whenever it arrives. In a duct it is far below `energy_db` (the many bottom paths are resolved from one another
by the pulse and a peak detector sees the strongest group, not their sum). SNR = SL - 2 TL + TS - (N0 + 10 log B)
+ 10 log(B T) + AG + factor with spherical TL and absorption on the straight path, and the detection probability is the
chain's own (`live.detection_probability`) at that SNR.

Not in it: range dependence of the water or the bottom, a rough bottom, the caustic phase, leakage into shadows
(diffraction), reverberation as a limit on detection (the table is for noise), the target's aspect.

## 29. Echoes that share a range cell, false alarms counted against rows, and other repairs [OURS]

### 29.1 What stacked rows read when boundary echoes share the range cell

**The observation.** The two-row face (rows 1.5 m apart, 4 to 6 kHz, range resolution c / 2B = 0.375 m) 30 m deep in
100 m of water, a target at 3 km. For the target at 60 m the reported depth was 98, 97 and 11 m on three pings with a
stated sigma of 11 to 15 m. Moving the target through the column (`popoto_face_2x32` rehearsal scenario, three pings
each, C++ chain) shows what the rows read:

| target depth, m | elevation read, degrees | depth reported, m |
| --- | --- | --- |
| 10 | +0.49 to +0.57 | 0 to 4 |
| 25 | +0.36 to +0.42 | 8 to 11 |
| 45 | +0.37 to +0.53 | 2 to 11 |
| 60 | -1.43 to -1.38 | 95 to 98 |
| 80 | -1.33 to -1.21 | 93 to 100 |
| 95 | -1.49 to -1.41 | 92 to 96 |

The reading is the sea surface (atan(30 / 3000) = +0.57 degrees) or the sea bed (-atan(70 / 3000) = -1.34 degrees),
and says only in which part of the column the target is.

**Why.** The echo returns by every pair of one-way paths: out by path i, back by path k. With a direct path D, a
surface path S and a bottom path B there are nine. Echo (i, k) and its mirror (k, i) have the same length exactly and
the same reflection coefficients, so they add as pressures whatever the wavelength: twice the pressure of either. One
of the two arrives at the array from path i's elevation and the other from path k's; at height h on the array their
sum is

    exp(i k h sin(e_i)) + exp(i k h sin(e_k)) = 2 cos(k h (sin(e_i) - sin(e_k)) / 2) exp(i k h (sin(e_i) + sin(e_k)) / 2)

so rows whose aperture cannot separate the two read the MEAN of the two elevations. For D with S that mean is
((z_own - z) + (z_own + z)) / 2r = z_own / r, the elevation of the sea surface above the target: it does not
contain the target's depth z at all. For D with B it is -(H - z_own) / r, the sea bed. These two
pairs are 6 dB above the direct echo each, and where they fall in one resolution cell with each other (their ranges
after the direct echo are z_own z / r and (H - z_own)(H - z) / r) their clump is up to 12 dB above it, so the stage's
rule "the earliest arrival within 8 dB of the strongest" passes the direct echo by and lands on a boundary.

**What the chain now says** (`elevation.multipath_sigma`, `libs/live` `multipath_depth_limit`). For every target depth
z the water allows, the chain's ray fan (Sec. 28) through the profile it holds gives the one-way paths and from them
the echoes: range, pressure (2 a_i a_k for a pair, a_i^2 for i = k), elevation (the mean). Echoes closer than c / 2B
are one clump; the clump read is the earliest whose expected power is within `first_arrival_db` of the strongest's.
At an echo g of that clump the rows read g's elevation pulled by every other echo h:

    s_g^2 = sum_h (a_h / a_g)^2  E2(x_gh)  d_gh^2 / 2

E2(x) = min(1, 1 / (2 pi^2 x^2)) is the compressed pulse's power x cells away (a flat band's sinc^2, its lobes
averaged); d is the difference of the elevations, at most lambda / (2 pi D), which is the most a second wave can turn
the phase across a height D; the half is the mean over the relative phase. How well depth z can give the reading e is

    W(z) = max over the clump's echoes g of exp(-(e - e_g)^2 / 2 (sigma_e^2 + s_g^2))

Every echo of the clump counts fully (which one wins turns on phases nobody knows) and every depth counts by W alone
(nothing is known of the target). The depth's variance is the W-weighted mean of (z - z_reported)^2; the same mean
with the first path alone and resolved is what the angle's sigma already holds, and the difference is the new term,
`depth_unresolved_sigma_m`. Limits: one resolved echo gives W a Gaussian of the angle's own width and the term is
range x s_g, the pull of the far echoes (checked against the closed form to 5 %: 0.7 m at 3 km for a target 60 m below
a sonar at 200 m, the surface pair 46 cells away); a reading at a boundary gives W = 1 over every depth whose clump
holds that pair and the term is the spread of that part of the column. `depth_multipath_limited` is true where the
term is the larger part of the sigma. No depth with W above exp(-9 / 2): the reading is not an echo of this picture
(a far lobe, a path of more bounces) and nothing is claimed.

**How honest it is.** Over the 18 pings of the table, sigma with the term is 19 to 48 m and the largest error is
1.53 sigma (before, on the rehearsal's three pings: 2.5 to 4.5 sigma). At 1 km with two rows (targets at 10, 30, 60, 90 m, 12 pings) the errors are
0.4 to 6.3 m against sigmas of 3.9 to 38 m, at most 0.59 sigma: the term is too wide there by a factor of three to
five for the targets at 10 and 30 m, because depths whose clump holds a mixed surface-and-bottom pair can also give
those readings in the picture. With three rows at 1 km and the target at 60 m the term is 1.8 m and the depth sigma
3.9 m (error 1.1 to 1.6 m). The term is in a track's sigma as well and is not given to the track's filter: the
boundary read is the same one ping after ping.

**Not in it.** Boundary losses (both are perfect mirrors: a soft bottom or a rough surface weakens the pairs and the
term is then too wide); the projector's offset from the array; echoes of more than `bounces` reflections a leg; any
correction of the depth itself (a reading at a boundary carries no depth to correct). Without a water depth there is
no bottom in the picture and the report says so (`depth_unresolved_assumed`).

### 29.2 Depth from multipath timing in water too shallow for it

The surface echo of a target at depth z is tau = 2 z (z_array + z_projector) / (r c) behind its direct echo (both ways,
Sec. 25.1) and the stage searches only depths where tau B is at least `min_delay_bandwidth` (3): from

    z_min = 3 r c / (2 B (z_array + z_projector))

down. The test AUV's trial (`ssp_front_testauv_trial_1row`: array and projector 10 m deep, the detection centred 523 m
away, 2 kHz of band): z_min = 29.3 m, and the sea bed is at 30 m. Through rays the grid of depths is laid in whole
carrier fringes and ran past the sea bed; the last depth has no path, its surface delay is NaN, and that delay sized the
beam window: the stage answered `window` for every detection. With the window sized by the longest finite delay the
stage then fitted the 0.6 m that are left and gave 29.8 m for a target at 15 m (the bottom's echoes fitted as a surface
echo, as the flat pattern's 30.3 m). The depth change that moves the surface echo by one range resolution cell is
1 / (B dtau/dz) = z_min / 3 = 9.8 m: a band thinner
than that cannot tell one depth from another by delay, and the carrier fringes alone repeat every
c r / (2 fc (z_array + z_projector)). So through rays a band thinner than one such cell is answered `unresolvable`, with
`multipath_searched_m` the band clipped to where the paths exist. For this geometry no depth can be measured by timing
whatever the bottom's model: the limit is the range and the two depths, not the estimator.

### 29.3 How many false detections a map of noise gives

**Looks in a cell mean.** White noise through the matched filter has the replica's power spectrum, so the amplitude
correlation between two samples of the compressed noise is the replica's autocorrelation at their two-way time apart.
The plan took sinc(2 B x / c), a flat band; the long-range pulse is a sweep under a 10 % Tukey taper, whose band is
narrower, and the mean of a 2 m cell (8 samples 0.25 m apart) holds 5.48 looks, not 5.80 (5.42 measured on the map).
`sample_correlation` now autocorrelates the replica the chain compresses with. The cell-mean test's threshold rises by
0.12 to 0.17 dB; the training correction uses the same looks.

**Tests on an oversampled map.** A budget of N false detections a ping is met by a threshold x on each test with
P(> x) x tests = its share of N, so the number of independent tests must be known. For a statistic that is the mean of
a = rows x looks unit exponentials, u = a x:

*The single sample* is a field smooth in range and in bearing: X = (1/2) sum of 2a squared unit Gaussian fields g_i with
roughness L (the covariance of their gradients). Near a maximum of height h (in the units of u) the fields are aligned
along one direction e, and the component along e, a Gaussian field with the high value sqrt(2h), falls off as
sqrt(2h) (1 - s'Ls / 2): X(s) = h (1 - s'Ls) to leading order. The region above u about that maximum is the ellipse
s'Ls < (h - u) / h of area pi (h - u) / (u sqrt|L|). The excess h - u of a maximum over a high threshold is exponential
with mean 1 (the tail is e^-u times a power), so a region has mean area pi / (u sqrt|L|), and since the expected area
above u is the map's area times P(> u),

    separate regions = area x P(> u) x u sqrt|L| / pi          tests = regions / P = (area sqrt|L| / pi) u

growing in proportion to u. *A cell's mean* is independent from cell to cell in range (the cells do not overlap) and
smooth along bearing only: its upcrossings of u along bearing are Rice's, f_X(u) E[dX/db > 0 | X = u], and with
dX/db = sum g_i dg_i/db, Gaussian of variance 2 u L_b given X = u, tests = upcrossings / P = sqrt(u L_b / pi) x the
hazard f_X(u) / P(> u) of the statistic.

The areas and roughnesses are not taken from theory: the map is sampled 1.5 times per range resolution, the largest
sample is searched for, a stronger cell's sidelobes are dropped and flagged cells are grouped. One constant per test
and number of rows is measured (`tools/detector_search_factor.py`; 400 noise-only pings for each of one, two and three
rows of 32, 0.8 km window, each test alone at 10, 3, 1, 0.3 and 0.1 false detections a ping, counted on the chain's own
map with the detector's own code):

| | the detector as it was (one factor, 2.5): found / asked at 10, 3, 1, 0.3, 0.1 a ping | constant with the growth law, same budgets | pooled, 3 a ping and fewer |
| --- | --- | --- | --- |
| sample, one row | 0.83, 0.86, 0.85, 0.93, 1.05 | 0.215, 0.197, 0.179, 0.176, 0.184 | 0.191 +- 0.005 |
| sample, two rows | 1.05, 1.14, 1.19, 1.28, 1.38 | 0.215, 0.211, 0.203, 0.201, 0.201 | 0.208 +- 0.005 |
| sample, three rows | 1.08, 1.19, 1.35, 1.43, 1.60 | 0.188, 0.189, 0.198, 0.194, 0.204 | 0.192 +- 0.004 |
| cell mean, one row | 0.63, 0.69, 0.67, 0.64, 0.70 | 0.524, 0.534, 0.488, 0.445, 0.465 | 0.515 +- 0.015 |
| cell mean, two rows | 0.67, 0.73, 0.75, 0.75, 0.82 | 0.528, 0.539, 0.524, 0.499, 0.525 | 0.532 +- 0.015 |
| cell mean, three rows | 0.71, 0.79, 0.92, 1.08, 1.18 | 0.550, 0.571, 0.627, 0.704, 0.730 | 0.598 +- 0.016 |

(Counts 28 to 4327 an entry; the cell-mean rows were measured with the looks corrected.) What the first column shows is
that the flat factor was wrong in two ways at once: more rows gave more false detections at the same budget (1.4 times
for three rows against one), and every case ran further over as the budget tightened. Both are the growth with u: at one
false detection a ping u is 11.9, 14.7 and 17.0 for one, two and three rows. With tests proportional to u the sample
test's constant is the same for one, two and three rows within 8 % and does not depend on the budget within its errors
(one row falls by 15 % from 10 to 1 a ping and is level below). The cell mean's constant is level for one and two rows
and RISES for three rows as the budget tightens (0.55 to 0.73): the law does not hold it, and the cause is not found.

**With the constants in the code**, the whole detector (both tests) on 300 other noise-only pings for each number of
rows gave, at budgets of 3, 1 and 0.3 false detections a ping: 0.96, 0.92, 0.89 of the budget with one row; 0.91,
0.93, 0.90 with two; 0.87, 0.86, 0.79 with three (860 to 71 detections an entry: 0.03, 0.06 and 0.10 of statistical
error). All under one: a place both tests flag is one detection, and each test was calibrated alone.

**What the thresholds moved by.** For the six builds at their own windows and the default budget, the budget SNR of
one row for a detection probability of one half rose by 0.11 to 0.15 dB (0.02 dB of it the looks, the rest the count of
tests, which at that threshold is 1.7 to 2.2 times what the flat factor counted for the sample):

| build | rows | Pd 0.5, dB: before, after | Pd 0.9, dB: before, after | a detection's `snr_db` at least, dB: before, after |
| --- | --- | --- | --- | --- |
| `popoto_face_1x32` | 1 | 15.65, 15.77 | 17.25, 17.38 | 13.50, 13.61 |
| `popoto_face_2x32` | 2 | 13.09, 13.24 | 14.70, 14.84 | 11.07, 11.21 |
| `popoto_face_3x32` | 3 | 11.64, 11.78 | 13.24, 13.37 | 9.76, 9.89 |
| `popoto_static_2x32` | 2 | 12.82, 12.96 | 14.47, 14.60 | 10.82, 10.95 |
| `popoto_static_3x32` | 3 | 11.37, 11.52 | 13.02, 13.14 | 9.52, 9.65 |
| `popoto_testauv_1x32` | 1 | 15.16, 15.27 | 16.85, 16.96 | 13.02, 13.12 |

**What is not measured.** The default budget is one false detection in a thousand pings; counting there would take a
hundred thousand pings. The thresholds at that budget rest on the growth law beyond the measured range (u of 22 to 28
against 10 to 20 measured). If the sample test's count grew only as sqrt(u) beyond it (range sampled too coarsely to be
smooth), the detector would be about 1.3 times under its budget; the three-row cell mean, by its measured trend, could
be 1.5 times over its share. Neither is corrected. The constants belong to this map (bearings every 0.5 degrees, 8
samples per 2 m cell, Hann aperture, 15 m merge): another map needs the tool run again.
