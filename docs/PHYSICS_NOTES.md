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
| Bottom forward loss | "lossy Rayleigh coefficient" (no formula) | Rayleigh fluid-fluid reflection coefficient with complex sediment sound speed | From memory, verify before use: `V = (m sin th - P)/(m sin th + P)`, `P = sqrt(kappa^2 - cos^2 th)`, `m = rho_sed/rho_w`, `kappa = k_sed/k_w = (1/nu)(1 + i delta)^-1`-type complex index, th = grazing angle. Sign of the imaginary part depends on the time convention; SST uses exp(+2 pi i f t). |
| Bottom backscatter / bistatic | pointer to Jackson models | (a) Lambert's law as a floor; (b) Jackson rough-interface + sediment-volume model (Jackson et al. 1986; Mourad & Jackson 1989; Williams & Jackson 1998; Jackson & Richardson, *High-Frequency Seafloor Acoustics*, 2007) | Lambert, from memory, verify before use: `sigma = mu sin(th_i) sin(th_s)`, `10 log10 mu` ~ -27 dB (Mackenzie value) as a classic default. |
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
`a = mss t1 t2 / (t1 + t2)`, `b = a sin^2 g`. (`decay_time: facet`; the default stays the rule.)

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

