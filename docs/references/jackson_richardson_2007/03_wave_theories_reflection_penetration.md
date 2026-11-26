# Jackson and Richardson (2007): wave theories, reflection, penetration, transducers, measurements

Chapters 8 (fluid), 9 (elastic), 10 (poroelastic), 11 (reflection), 15 (penetration), Appendices F
(transducers) and G (measurements), plus eq. (13.14) and Appendix L.3 (coherent and incoherent
reflection), which the reflection material depends on. Issue sonar-gm8e. Written 2026-11-26.

Source: D. R. Jackson and M. D. Richardson, *High-Frequency Seafloor Acoustics*, Springer, 2007.
"p." is the printed page; the PDF page is given once per chapter. All formulas below are re-typed
from the page images in the book's notation; our symbol is given beside it. Status words as in
docs/REFERENCES.md: **confirmed** = read on the page image of this book. Where the book attributes
something to APL-UW TR 9407, that is said; TR 9407 itself was not read.

What is wrong, risky or missing, first:

- **The book does not give a layered-seafloor reflection recursion** in Chs. 8-11 (Ch. 8 has no
  layered section; Sect. 11.1 and 11.3 only state what transition layers do). The recursion in
  Sec. 2.2.9 below is the standard result, **ours** (checked numerically), not this book's.
  Sonar-3k1's layered part must cite Olson and Jackson arXiv:2009.12420 (already `confirmed` in
  REFERENCES.md) or Brekhovskikh, not this file.
- **Our `sediment` tier is badly wrong for buried targets below the critical angle** (Sec. 3.5):
  for a target at 0.3 m depth, 40 m range, 10 m height (direct-line grazing 14 deg, medium sand,
  critical angle 31.9 deg) `SedimentProvider` returns a ray that enters at the critical angle and
  runs 24 m through the sediment with no attenuation, at -31.9 dB (spherical spreading). The flat
  interface theory (eq. 15.1) gives an evanescent field more than 100 dB lower at 20 kHz; the
  book's measured and modelled sub-critical field (roughness scattering, Figs. 15.8, 15.11) is
  about 10-30 dB below the incident field. Above the critical angle the tier misses sediment
  attenuation (6 dB two-way at 20 kHz and 0.3 m in medium sand) and angle-dependent transmission.
- **Rock reflection with the fluid model is wrong by up to about 19 dB** at 10-45 deg grazing
  (Sec. 3.3): the fluid model gives total reflection below 66 deg for our `rock` row; with shear
  waves (speed 40-60 % of the compressional speed, book p. 169) the loss is 1.4-3.1 dB at 5 deg,
  2-7 dB at 10 deg and peaks at 9-19 dB near the shear critical angle. Over a rock floor every
  bounce of our image-source and Bellhop paths is too strong.
- **Sand: the simple fluid model underestimates reflection loss by 1.2-1.6 dB at normal incidence
  and 1.0-1.2 dB at 45 deg** against Biot/EDFM with the same measured speed and attenuation
  (SAX99 parameters, 8-330 kHz, Sec. 3.4), and overestimates roughness backscatter by 1-3 dB.
  The book's own lab comparisons (Fig. 11.6) show the fluid model 2-4 dB low at normal incidence.
- **Four printed values in my chapters are misprints or inconsistent** (Sec. 6): (11.2) intercept
  1.0091, (11.3) first constant 0.4566, (10.39) gives k^2 not k, and the text value
  rho_eff/rho = 0.657 + i 0.0046 at 250 kHz (p. 319) disagrees with eq. (10.75), Table 10.1 and
  Fig. 10.6, which give about 0.86.
- **What the book does NOT cover here**: nothing on the sea surface (sonar-9le) beyond the Eckart
  form (13.14) used for the seafloor; no K-distribution (Ch. 16, not mine); no Kirchhoff/composite
  interpolation or large-roughness law (Ch. 13, not mine); no bistatic model (Ch. 13/J, not mine).

---

## 1. Scope

| Part | PDF pages | Printed pages | Read |
|---|---|---|---|
| Ch. 8 Fluid theories | 256-276 | 245-265 | in full (page images) |
| Ch. 9 Elastic theories | 277-293 | 267-283 | in full |
| Ch. 10 Poroelastic theories | 294-316 | 285-307 | in full |
| Ch. 11 Reflection | 317-328 | 309-320 | in full |
| Ch. 15 Penetration | 415-428 | 407-420 | in full |
| App. F Transducers | 492-496 | 487-491 | in full |
| App. G Measurements | 497-509 | 493-508 | in full |
| Eq. (13.14) and App. L.3 | 343, 547-549 | 335, 543-545 | those pages only |
| Ch. 5.6 rock properties | 181 | 169 | extracted text only (not the page image) |
| Ch. 12 first pages, Ch. 16 first pages | 329-331, 429-431 | 321-323, 421-423 | glanced (other agents' parts) |

Chapter 8.8 and 9.9 defer their research issues to Ch. 10.5 (p. 265, 283); summarised there.

Checks run (scripts in the session scratchpad, `fix/jr3/models.py` and one-off `python3 -c`, with
`PYTHONPATH=<worktree>/python`; the built module was not needed): the book's fluid, elastic,
Biot and EDFM formulas re-typed and compared with the book's printed numbers and with our
`bottom.reflection`, `propagation.rayleigh_reflection`, `bottom.parameters_from_physical`,
`acoustics.rough_coherent_reflection`, `propagation.SedimentProvider`. Results are quoted where used.

---

## 2. What the book gives

### 2.1 Conventions (all confirmed)

| Item | Book | Ours | Agree? |
|---|---|---|---|
| Time dependence | exp(-i w t) throughout (p. 247) | exp(-i w t) (bottom.py, propagation.py) | yes |
| Angle | grazing angle theta_w from the horizontal; sin(theta_w) = beta_w(K) (8.42) | grazing | yes |
| Wavenumbers | k_w = w / c_w in water, k_p = w / c_p complex in sediment; K horizontal (8.13) | k, k2 = k (1 + i delta) / nu | yes |
| Vertical sine | beta_q(K) = sqrt(1 - K^2 / k_q^2), root with positive imaginary part (p. 250) | `_vertical`: root with Im >= 0 | yes |
| Density ratio | a_rho = rho / rho_w (8.54) | `rho` | yes |
| Complex speed ratio | a_p = c_p / c_w (8.23) | 1 / kappa, kappa = (1 + i delta) / nu | yes |
| Phase speed ratio | nu_p = c_pphase / c_w (8.24) (= V_pR in Ch. 5) | `nu` | yes |
| Loss parameter | delta_p = k_p'' / k_p' (8.18) | `delta` | yes |
| Bottom loss | L = -20 log10 |V_ww| (2.11) | same | yes |
| Pressure reflection / transmission | V_ww, V_wp (pressure ratios; displacement-potential versions differ for V_wp, p. 257) | `reflection`, T = 1 + V | yes |
| Source level | SL = 20 log10(s0 / (p_ref r_ref)), |P_i| = sqrt(2) s0 |b_x| / r: s0 / r is the RMS pressure, |P| the peak (F.5, F.7) | - | note the sqrt(2) |

### 2.2 Chapter 8: fluid theory

**2.2.1 Losses and attenuation (confirmed, pp. 250-252).**
- Complex modulus K_b = K_b' - i K_b'' (8.15); complex wavenumber k_p = k_p' + i k_p'' (8.16);
  complex density rho = rho' + i rho'' (8.17), all with positive primed parts under exp(-i w t).
- delta_p = k_p'' / k_p' (8.18). If c_p is frequency independent, delta_p is too and attenuation
  is linear in f; the book says this "can at best be an approximation" (causality, App. I; Biot
  predicts non-linear dependence) (p. 251).
- Attenuation in dB/m: alpha_p = 20 k_p'' / ln 10 (8.20), from exp(-2 k_p'' r) = 10^(-0.1 alpha_p r) (8.19).
- Phase speed: 1 / c_pphase = Re{1 / c_p} (8.22), not Re{c_p}.
- a_p = nu_p / (1 + i delta_p) (8.25).
- alpha_p = 40 pi f delta_p / (nu_p c_w ln 10) dB/m (8.26).
- Per wavelength in the sediment (c_pphase / f): alpha_p_lambda = 40 pi delta_p / ln 10 = 54.58 delta_p dB (8.27).
- Q_p = 1 / (2 delta_p) (8.28); logarithmic decrement Delta_p = 2 pi delta_p (8.29).
- dB/m/kHz (Ch. 11's k): from (8.26), delta_p = k nu_p c_w ln 10 / (40 pi x 1000) (derived; the
  book gives k and (8.26), not this line).
- Intensity I = (1/2) Re{P* v} (8.30); for a plane wave I = (|P|^2 / 2) Re{1 / (rho c_p)} (8.31).

**2.2.2 Boundary conditions (confirmed, pp. 253-254).** Pressure continuous (8.33); normal
displacement continuous (8.34), N = e_z - grad f(R) (8.35); on a flat interface
(1/rho_w) dP/dz|+ = (1/rho) dP/dz|- (8.37).

**2.2.3 Flat-interface reflection and transmission (confirmed, pp. 255-257).**
- Snell: K = k_w cos theta_w = k_p cos theta_p (8.43); sin theta_p = sqrt(1 - a_p^2 cos^2 theta_w) (8.46), complex.
- 1 + V_ww = V_wp (8.47); (k_w beta_w / rho_w)(1 - V_ww) = (k_p beta_p / rho) V_wp (8.48).
- V_ww = (z_wp - 1) / (z_wp + 1) (8.49), z_wp = z_p / z_w (8.50), z_w = rho_w c_w / sin theta_w
  (8.51), z_p = rho c_p / sin theta_p (8.52), z_wp = a_rho a_p sin theta_w / sin theta_p (8.53).
- V_wp = 2 z_wp / (z_wp + 1) (8.55).
- Defined for all K (evanescent included) by z_wp(K) = a_rho a_p beta_w(K) / beta_p(K) (8.64).
- Critical angle (lossless): theta_crit = arccos(1 / nu_p) (8.56). In the lossy case |V| is a
  little below 1 under it and falls quickly above; (8.56) remains "nonetheless useful" (p. 258).
- Intromission angle (slow, lossless): theta_int = arccos( sqrt( (a_rho^2 - 1/nu_p^2) / (a_rho^2 - 1) ) ) (8.57).
- Normal incidence z_wp = a_rho a_p (8.58). A complex SPEED gives a_p a negative imaginary part, a
  complex DENSITY gives a_rho a positive one: they move the phase of V oppositely (p. 258), the
  basis of the reflection-phase test in Sect. 11.2.2.
- Fig. 8.3 parameters: sand nu_p = 1.1, a_rho = 2.0; mud nu_p = 0.98, a_rho = 1.5; delta_p = 0 and
  0.01. Printed: theta_crit = 24.6 deg (sand), theta_int about 10.5 deg (mud). Recomputed:
  24.62 deg and 10.46 deg; |V| = 0.0009 at 10.5 deg for lossless mud.
- Transmission from sediment to water: V_pw(K) = V_wp(K) k_p beta_p(K) / (a_rho k_w beta_w(K)) (8.67).
- Energy (derived from (8.31), checked numerically, lossless, 40-90 deg, to 1e-15):
  1 - |V_ww|^2 = |V_wp|^2 Re{sin theta_p / (a_rho a_p)} / sin theta_w.

**2.2.4 Point sources over a flat seafloor, the evanescent and lateral waves (confirmed, pp. 259-264).**
- Green's function with source term -4 pi delta (8.59); free field G0 = exp(i k_w |r - r'|) / |r - r'|
  (8.60), i.e. unit pressure at 1 m ("unit monopole amplitude", Pierce p. 160).
- Plane-wave (Weyl) expansion G0 = (i / (2 pi k_w)) Int exp(i K.(R - R') + i k_w beta_w |z - z'|) / beta_w d2K
  (8.61); Sommerfeld form with J0 (8.62).
- Reflected field (source and receiver in water): G_ww = G0 + (i / k_w) Int_0^inf V_ww(K) J0(K |R - R'|)
  exp(i k_w beta_w (z + z')) K dK / beta_w (8.63).
- Field in the sediment: G_wp = (i / k_w) Int V_wp(K) J0(K |R - R'|) exp(i k_w beta_w z' - i k_p beta_p z) K dK / beta_w (8.65), z' > 0, z < 0.
- Reciprocity: G_pw(r', r) = G_wp(r, r') / a_rho (8.68); in general G(r', r) / rho(r') = G(r, r') / rho(r) (8.69).
- Figs. 8.4-8.6: 20 kHz Gaussian pulse (5 kHz 3-dB full bandwidth), source 5 m above sand
  (nu_p 1.16, a_rho 2.0, delta_p 0.01). Below the critical angle the evanescent wave is visible
  close under the interface (Fig. 8.5, nu_p raised to 1.2); the book: "may be important in sonars
  designed to detect buried objects". The lateral wave (Fig. 8.6) needed a source 0.05 m above
  the floor and unrealistic nu_p = 1.5, delta_p = 0.0005: "Generally speaking, the lateral wave is
  unimportant at high frequencies, but it may be an issue in close-range measurements of
  reflection" (p. 262).

**2.2.5 When the plane-wave coefficient is wrong (confirmed, Sect. 11.1, p. 311, Fig. 11.1).**
Valid "if the propagation distances are many wavelengths and if the lateral wave can be ignored".
Fig. 11.1 (wavenumber integration of (8.63)): c_w = 1530 m/s, a_rho 2.0, nu_p 1.16, delta_p 0.01,
source and receiver 10 cm above the floor: at 200 kHz (heights about 13 wavelengths) the bottom
loss departs from the plane-wave curve near and below the critical angle (30.5 deg), with
oscillations from interference of reflected and lateral waves; at 500 kHz (about 33 wavelengths)
it follows the plane-wave curve. The curves scale with k h: the same holds at 20 and 50 kHz with
the transducers 1 m above the floor.

**2.2.6 Wood's equations (confirmed, p. 264).** rho = beta rho_w + (1 - beta) rho_g (8.70);
1 / K_b = beta / K_w + (1 - beta) / K_g (8.71); c0 = 1 / sqrt( [beta/K_w + (1-beta)/K_g] [beta rho_w + (1-beta) rho_g] ) (8.72),
beta the fractional porosity. Wood "usually gives a rather poor prediction of sediment sound
speed" (ignores grain contacts, inertial and viscous forces) but is the base of Gassmann, Biot
(EDFM) and Buckingham.

**2.2.7 Gradients.** Only the general equation (8.5), rho div((1/rho) grad P) + (w^2/c_p^2) P = 0,
"capable of describing ... layered or otherwise heterogeneous sediments" (p. 248). No closed form
for a gradient profile in Ch. 8.

**2.2.8 What the book says layers and gradients do (confirmed, Sect. 11.1, p. 311).**
- Density and speed usually decrease towards the interface (transition layer).
- Mourad and Jackson (1989), quoted: a 3-cm density transition layer gives a 6-dB increase in
  bottom loss at normal incidence at 20 kHz; by scaling, 3 mm does the same at 200 kHz.
- Lyons and Orsi (1998): CT-measured density profiles fitted by the analytic function (7.36) with
  a = 10-200 1/m (a = 200 1/m is a 5-mm layer); reflection reductions of 10 dB and more
  against the gradient-free case are predicted in some cases.
- A positive speed gradient (increasing downward) refracts energy back up near the critical
  angle and makes the reflection coefficient oscillate with angle.

**2.2.9 Layered fluid seafloor: recursion (NOT in this book; standard result, ours).**
For layers j = 1..N between water (0) and a half-space (N+1), each fluid with (a_rho_j, nu_j, delta_j),
thickness d_j, with the book's quantities: impedance Z_j = a_rho_j a_j / sin theta_j (water: 1 / sin theta_w,
i.e. z normalised by rho_w c_w), vertical wavenumber kz_j = k_j sin theta_j = (k_w / a_j) sin theta_j,
sin theta_j = sqrt(1 - a_j^2 cos^2 theta_w) (Im >= 0), a_j = nu_j / (1 + i delta_j):

    V_{N} = (Z_{N+1} - Z_N) / (Z_{N+1} + Z_N)
    V_{j-1} = (r_j + V_j e^{2 i kz_j d_j}) / (1 + r_j V_j e^{2 i kz_j d_j}),  r_j = (Z_j - Z_{j-1}) / (Z_j + Z_{j-1})
    V_ww = V_0

(exp(-i w t); the phase factor is e^{+2 i kz d}). Checked: thickness 1e-9 m gives the half-space
coefficient to 1.6e-8; a layer identical to the half-space gives it exactly; a lossless
quarter-wave matching layer (Z_1 = sqrt(Z_2)) gives |V| = 2e-17 at normal incidence; a 5 m layer
with delta 0.05 gives the layer's own half-space coefficient (6e-17). Illustration (ours, a step
layer, not a gradient): a homogeneous 3-cm layer of a_rho 1.3, nu 1.0 on medium sand raises the
normal-incidence loss from 8.6 to 10.2 dB at 20 kHz; the book's 6 dB is for a graded profile not
printed in these chapters, so a staircase of sublayers with a measured profile would be needed to
reproduce it.

### 2.3 Chapter 9: elastic theory (confirmed unless noted)

- Isotropic: sigma_ij = lambda delta_ij div u + mu (d_i u_j + d_j u_i) (9.2); c_p = sqrt((lambda + 2 mu) / rho)
  (9.9), c_t = sqrt(mu / rho) (9.10) (t = shear, "transverse").
- Losses: a_q = c_q / c_w = nu_q / (1 + i delta_q), q = p, t (9.23); alpha_q = 40 pi f delta_q / (nu_q c_w ln 10) (9.24).
- Flat-interface reflection (p. 274-275): V_ww = (z_e - 1) / (z_e + 1) (9.34),
  z_e = z_wp cos^2(2 theta_t) + z_wt sin^2(2 theta_t) (9.35), z_wp = a_rho a_p sin theta_w / sin theta_p (9.36),
  z_wt = a_rho a_t sin theta_w / sin theta_t (9.37), cos theta_q = a_q cos theta_w (9.38),
  cos 2 theta_t = 2 a_t^2 cos^2 theta_w - 1 (9.39), sin^2 2 theta_t = 1 - cos^2 2 theta_t (9.40).
  Only the vertically polarised shear wave is excited on a flat interface.
- Transmission (displacement potentials, after Brekhovskikh and Godin 1990):
  V_wp = -(a_p cos 2 theta_t sin theta_w / sin theta_p)(1 - V_ww) (9.41),
  V_wt = -(a_t sin 2 theta_t sin theta_w / sin theta_t)(1 - V_ww) (9.42).
- When shear matters (p. 267, 275-276): in clay, silt and sand shear speeds are much lower than
  water's and elasticity gives "rather small corrections"; the fluid model "fails entirely" for
  sandstone, granite, basalt. Fig. 9.3: sand, c_w 1500, nu_p 1.1, a_rho 2.0, delta_p 0.01,
  nu_t 0.167 (250 m/s, "rather large"), delta_t 0.01: small difference; at 100 m/s "negligible".
  Fig. 9.4: rock nu_p 2.5, a_rho 2.5, delta_p 0.02, nu_t 1.3, delta_t 0.1: compressional
  critical angle 66.4 deg, shear critical angle 39.7 deg; the elastic model has large loss near
  the shear critical angle (energy into shear waves). Recomputed: minimum |V| = 0.052 at 35.7 deg.
- Interface waves (p. 276-277, Fig. 9.5): the reflection coefficient against K has a large peak
  just inside the evanescent domain at the Scholte (Stoneley) wavenumber (a true surface wave,
  decaying up and down, |V| > 1 allowed there), and nearly vanishes near the leaky-Rayleigh
  wavenumber, which makes the interface "nearly transparent" for angles just below the shear
  critical angle, "important consequences in scattering".
- Gassmann (p. 278): K_b = K_g (K_f + Q') / (K_g + Q') (9.43), Q' = K_w (K_g - K_f) / (beta (K_g - K_w)) (9.44);
  the saturated shear modulus equals the frame's; lambda + 2 mu = K_b + 4 mu / 3. Valid when
  fluid-frame relative motion can be neglected (low permeability: mud, rock).
- Rock properties (Ch. 5.6, p. 169, **extracted text only, page image not checked**): shear speeds
  "roughly 40-60%" of compressional; Poisson's ratio 0.2-0.4; igneous/metamorphic 5-6 km/s,
  sandstones and mudstones 2-6 km/s; densities 2100-2950 kg/m3 by type.

**Buckingham's grain-shearing theory (pp. 278-283, confirmed).**
- Time-domain equation (9.45) with stress-relaxation coefficients lambda_p, eta_t and "material
  impulse response functions" h_p(t) = t_p^-1 (1 + t/t_p)^-n (9.49), h_t(t) = t_t^-1 (1 + t/t_t)^-m (9.50);
  rho and K_b from Wood (8.70)-(8.71).
- Frequency domain: lambda + 2 mu = K_b - i w [lambda_p H_p(w) + (4/3) eta_t H_t(w)] (9.54),
  mu = -i w eta_t H_t(w) (9.55); H_p = (i / (w t_p))^(1-n) Gamma(1 - n) (9.58), similarly H_t with m (9.59).
- With m = n: c_p = c0 sqrt(1 + (3 gamma_p + 4 gamma_t) / (3 rho c0^2) (-i w T)^n) (9.60),
  c_t = c0 sqrt(gamma_t / (rho c0^2)) (-i w T)^(n/2) (9.61),
  c0 = sqrt(K_b / rho) (Wood) (9.62); gamma_p, gamma_t compressional and shear rigidity
  coefficients (9.63)-(9.64); T any convenient time (T = 1 s), compensated by gamma.
- delta_t = tan(n pi / 4) (9.65): shear attenuation fixes n; "Typically n << 1".
  gamma_t = rho c_tphase^2 cos^2(n pi / 4) / (w T)^n (9.66); gamma_p from one measured c_pphase via (9.60).
- Parameters: n (strain-hardening index), gamma_p, gamma_t; "cannot be derived from microscopic
  properties", taken from a few measurements; Buckingham (2005) treats them as universal.
- Fig. 9.6 example (Williams et al. 2002a inputs: cp at 38 kHz, shear at 1 kHz):
  c0 = 1653.4 m/s, rho = 2074 kg/m3, gamma_p = 2.48e8 Pa, gamma_t = 1.517e7 Pa, T = 1 s
  (n for this figure not printed on the pages read; Fig. 10.9 names n = 0.085 for its lower curve).
  Predicted behaviour: speeds rise slowly with frequency, attenuation roughly linear in f.
  Recomputed with n = 0.085: cp_phase 1748 / 1761 / 1772 / 1782 m/s and attenuation
  1.8 / 9.4 / 32 / 94 dB/m at 8 / 38 / 120 / 330 kHz; shear 136-159 m/s, delta_t = 0.067.
- Validity (p. 279): the stick-slip mechanism "may only be applicable to sands"; doubtful for
  muds (electrostatic forces) and cemented carbonates. At one frequency Buckingham's wave theory
  is equivalent to elastic (or fluid) theory, so reflection/scattering tests cannot separate it
  from them at fixed frequency; Biot "tends to predict somewhat lower levels of reflection ... and
  scattering" for sand (p. 282).

### 2.4 Chapter 10: Biot theory and the effective density fluid model (confirmed)

**Parameter set (13 parameters, p. 285; Table 10.1, p. 303: SAX99, Williams et al. 2002a; most
measured, a few adjusted to fit Figs. 10.9-10.10; a from (10.16)):**

| Parameter | Symbol | Unit | SAX99 value (Table 10.1) |
|---|---|---|---|
| Bulk modulus of grains | K_g | Pa | 3.2e10 |
| Permeability | kappa | m^2 | 2.5e-11 |
| Tortuosity | alpha | - | 1.35 |
| Porosity | beta | - | 0.385 |
| Viscosity of water | eta | kg m^-1 s^-1 | 0.00105 |
| Grain density | rho_g | kg m^-3 | 2690 |
| Bulk modulus of water | K_w | Pa | 2.395e9 |
| Water density | rho_w | kg m^-3 | 1023 |
| Frame shear modulus | mu | Pa | (2.92 - i 0.18) e7 |
| Frame bulk modulus | K_f | Pa | (4.36 - i 0.208) e7 |
| Pore size | a | m | 2.65e-5 |

No other per-sediment Biot tables are in Ch. 10 (Chotiros 1995a compiles others; not printed).
Five parameters (fluid and grain properties) can take handbook values; 8 must be measured or
inferred (p. 303).

**Equations.**
- Moduli (Stoll 1974): H = (K_g - K_f)^2 / (D - K_f) + K_f + 4 mu / 3 (10.4), C = K_g (K_g - K_f) / (D - K_f) (10.5),
  M = K_g^2 / (D - K_f) (10.6), D = K_g [1 + beta (K_g / K_w - 1)] (10.7). (10.4) is Gassmann.
- Viscous correction (oscillatory flow in tubes): F(eps) = (eps/4) T(eps) / (1 - (2i/eps) T(eps)) (10.13),
  T = -sqrt(i) J1(eps sqrt(i)) / J0(eps sqrt(i)) (10.14), eps = a sqrt(w rho_w / eta) (10.15);
  pore size from kappa = a^2 beta / (8 alpha) (10.16). Checked: F(10 Hz) = 1.0000 - 0.0018i;
  |F| = 1.03 / 1.93 / 5.46 at 1 / 10 / 100 kHz with Table 10.1 (Fig. 10.2 shows about 1 below
  1 kHz, rising to about 5 at 100 kHz).
- Dispersion: (H k^2 - rho w^2)(m w^2 - M k^2 + i F eta w / kappa) + (C k^2 - rho_w w^2)^2 = 0 (10.31),
  m = alpha rho_w / beta; solved by (10.39)-(10.46): a = C_H^2 C_M^2 - C_C^4, b = -C_H^2 w Omega + 2 C_C^2 w^2 - (rho/rho_w) C_M^2 w^2,
  c = (rho/rho_w) Omega w^3 - w^4, C_X^2 = X / rho_w, Omega = w alpha / beta + i F eta / (rho_w kappa).
  The root (-b +- sqrt(b^2 - 4ac)) / 2a is k_q^2, printed as k_q (Sec. 6). Shear:
  k_t^2 = w^2 (rho - w rho_w / Omega) / mu ((10.36); printed as k_t = w c_t with c_t = sqrt((rho - w rho_w / Omega) / mu),
  (10.47)-(10.48), where "c_t" is a slowness).
- Biot characteristic frequency f_c = beta^2 eta / (2 pi rho_w kappa) (10.52): attenuation ~ f^2 below,
  ~ f^0.5 above. Table 10.1: f_c = 969 Hz.
- Printed at 20 kHz (p. 304): fast wave 1763 m/s, 7.0 dB/m; slow wave 141 m/s, 852 dB/m;
  fluid/frame displacement ratio about 1.8 (fast) and -1.6 (slow). **Reproduced**: 1762.7 m/s,
  6.96 dB/m; 141.0 m/s, 852.1 dB/m (shear 127.5 m/s, 361 dB/m).
- Boundary conditions water/porous medium (open pore): traction, fluid pressure, fluid flow (10.57)-(10.60);
  plane-wave reflection is the 4x4 system A V = D (10.66)-(10.69) for (V_ww, V_w1, V_w2, V_wt).
  Fig. 10.5 (20 kHz): Biot bottom loss is about 2 dB above the fluid model (complex fast-wave
  speed, true density) near normal incidence; |V_w2 / V_w1| = 0.0056 at normal incidence and
  about 0.0001 at small grazing: the slow wave hardly matters; the difference comes from the
  differential fluid-frame motion of the fast wave.

**Effective density fluid model (EDFM, Williams 2001a; Sect. 10.2, pp. 298-302).**
- Frame moduli set to zero: K_eff = H = C = M = ((1 - beta) / K_g + beta / K_w)^-1 (10.70) (= Wood).
- rho_eff = (rho rho~ - rho_w^2) / (rho~ + rho - 2 rho_w) (10.75), rho~ = alpha rho_w / beta + i F eta / (kappa w) (10.76),
  rho from (10.9) = (8.70).
- c_1 = sqrt(K_eff / rho_eff) (10.81): complex, frequency dependent; the fluid equation (10.79)
  and fluid boundary conditions hold, so every fluid reflection, roughness (Ch. 13) and volume
  (Ch. 14) model takes EDFM "by the simple replacement of density by effective density and use of
  the measured sound speed as the compressional wave speed" (p. 299).
- rho_eff has a positive imaginary part (exp(-i w t)); equals rho at low frequency; real part below
  rho at high frequency (Fig. 10.6: about 2050 at 100 Hz falling to about 1750 above 10 kHz).
- Simple EDFM (p. 300-301): take rho_eff real, pore water = overlying water, measured phase speed:
  rho_eff / rho_w = 1 / (nu_p^2 [beta + (1 - beta) K_w / K_g]) (10.82). Inputs: measured nu_p,
  porosity, an assumed grain bulk modulus. Book: K_g / K_w = 13.4 (Table 10.1). Fig. 10.8
  (nu_p 1.16, beta 0.40, K_w 2.395 GPa, K_g 32-75 GPa): +-0.5 dB at normal incidence for a 2:1
  range of K_g (recomputed rho_eff / rho_w = 1.670 to 1.773). In muds (Wood adequate) (10.82)
  gives rho_eff = rho (p. 302).
- Fig. 10.7: EDFM and Biot bottom loss agree closely at 20 kHz with Table 10.1.
- Table 11.1 (p. 319), effective density ratio rho_eff / rho from (10.82) for three lab sands:
  Nolle et al. 1963: Mz 3.0, beta 0.48, a_rho 2.07, nu_p 1.16, delta_p 0.014, rho_eff/rho 0.69;
  Drevet et al. 1999: 2.0, 0.41, 1.973, 1.155, 0.0114, 0.84; Worley 2004: 2.0, 0.403, 2.00, 1.173, 0.01, 0.81.

**Tests of Biot theory (Sect. 10.3).** SAX99 speed dispersion (Fig. 10.9, sound speed ratio about
1.08 below 1 kHz to 1.16 above 50 kHz) is matched best by Biot and EDFM; attenuation (Fig.
10.10) is matched better by Buckingham, especially 100-400 kHz, where Biot is low; Briggs et al.
(2005b) measured 92 dB/m at 400 kHz in SAX04 where heterogeneity was nearly absent, "much higher"
than Biot. Simpson et al. (2003), 3-80 kHz: dispersion, attenuation "essentially a linear function
of frequency".

**Research issues (Sect. 10.5, also for Chs. 8-9).** Better field and lab data needed; measure over
about 100 Hz to 500 kHz; reflection measurements are an important adjunct; alternatives to Biot
(Hovem, Yamamoto and Turgut multiple pore sizes; Chotiros squirt flow and "Biot-plus"; Leurer
squirt flow in clays; Hickey and Sabatier fluid-content coefficient) are open; "it may be
unrealistic to expect a single theory to apply to both" sand and mud.

### 2.5 Chapter 11: reflection data against theory (confirmed)

**Measurement issues (Sect. 11.1).** Roughness: average the SQUARED reflected pressure over many
statistically similar patches: this estimates |V_wwc|^2 + |V_wwi|^2, approximately |V_ww|^2 of the
flat interface, provided scattering stays within a narrow cone around specular (spread in radians
<< 1) and the transducers are not so directional as to cut it off; averaging envelopes or dB is
biased. Volume scattering: comparable to or above roughness scattering; at normal incidence it
arrives later (App. G.2.2 separates it). Gradients and layers: Sec. 2.2.8 above. Spherical-wave
and lateral-wave effects: Sec. 2.2.5 above.

**Grain-size relations printed by the book (pp. 313-314; "the TR 9407 relations", given first by
Mourad and Jackson 1989 and expanded in TR 9407; adjustments of Hamilton and Bachman 1982; "should
not be regarded as geoacoustic regressions": empirical, to improve model-data fits):**

    a_rho = 0.007797 Mz^2 - 0.17057 Mz + 2.3139                          -1 <= Mz < 1
          = -0.0165406 Mz^3 + 0.2290201 Mz^2 - 1.1069031 Mz + 3.0455      1 <= Mz < 5.3
          = -0.0012973 Mz + 1.1565                                        5.3 <= Mz <= 9      (11.1)
    nu_p  = 0.002709 Mz^2 - 0.056452 Mz + 1.2778                          -1 <= Mz < 1
          = -0.0014881 Mz^3 + 0.0213937 Mz^2 - 0.1382798 Mz + 1.3425      1 <= Mz < 5.3
          = -0.0024324 Mz + 1.0091   [printed; see Sec. 6: 1.0019 fits]   5.3 <= Mz <= 9      (11.2)
    k (dB m^-1 kHz^-1) = 0.4566 [printed; 0.4556 fits]                   -1 <= Mz < 0
          = 0.0245 Mz + 0.4556                                            0 <= Mz < 2.6
          = 0.1245 Mz + 0.1978                                            2.6 <= Mz < 4.5
          = 0.20098 Mz^2 - 2.5228 Mz + 8.0399                             4.5 <= Mz < 6.0
          = 0.0117 Mz^2 - 0.2041 Mz + 0.9431                              6.0 <= Mz < 9.5
          = 0.0601                                                        9.5 <= Mz           (11.3)

(11.3) is "taken directly from [Hamilton 1972]" and is "a geoacoustic regression". Mz in phi.
Note (11.3) is discontinuous at Mz = 6.0 as printed (0.338 against 0.140); our table follows the
second piece at 6.0.

**Field data (Sect. 11.2.1).**
- Normal incidence against grain size (Fig. 11.2: Mackenzie 1960 at 16 kHz, Parrott et al.
  1980, Mourad and Jackson 1989 at 20-30 kHz) and porosity (Fig. 11.3: McLeroy 1972, 12 kHz):
  loss about 5-8 dB for coarse sand rising to 15-23 dB for fine sediments; two classes of site:
  near the "standard" (Ch. 5 regression) prediction, or anomalously high (matched by TR 9407).
  McLeroy's data follow the standard curve.
- Shear "would have a negligible effect" on these predictions; with no layering, frequency-
  independent speed and linear attenuation the fluid V has no frequency dependence (p. 314).
- Fig. 11.4 (Mourad and Jackson 1989), measured parameters: Arafura Sea 20 kHz a_rho 1.456,
  nu_p 0.988, delta_p 0.024; Puget Sound 30 kHz 1.74, 1.006, 0.01; Quinault 25-30 kHz 1.92, 1.117,
  0.014. Arafura and Puget data lie well above the measured-parameter fluid curve (anomalously
  high loss); at Quinault the simple EDFM (10.82) with beta 0.418, K_g/K_w 13.4 "underestimates
  bottom loss".
- McConnell and DeProspo (1994), 1-5 kHz, Mz 3.5 and 4.5 phi: high loss, matched better by TR 9407.
- Liebermann (1948, 24 kHz, 9 deg, interference method): biased to high loss by roughness
  "washing out" interference nulls.
**Laboratory data (Sect. 11.2.2, Fig. 11.6, Table 11.1).** Sands at 500 kHz (Nolle; Drevet) and
115 kHz (Worley): the fluid model gives bottom loss "significantly lower" than measured (by about
2-4 dB at steep angles, read from the figure); TR 9407 overestimates for Nolle; "overall" EDFM is
closest: "an indication that Biot effects are significant for sandy seafloors". Yargus (2003),
250 kHz Ottawa sand Mz 1.84: effective density consistent with Williams et al. 2002a.
- Reflection phase at normal incidence: positive imaginary part of V if poroelastic (density)
  losses dominate, negative for fluid, elastic or Buckingham (speed) losses; magnitude of the
  phase about delta_p radians, small (p. 320).
**Research issues (Sect. 11.3).** Poroelasticity lowers reflection relative to fluid/elastic
theory (Buckingham behaves as fluid/elastic here); interpretation of present data is ambiguous;
characterise the medium better; measure roughness and average over patches; volume-scattering
contamination of reflection data deserves more attention; transition layers must be accounted
for unless the wavelength is much larger than the layer (the "occasional success" of TR 9407 is
attributed to this); at 100-500 kHz the needed millimetre resolution of speed profiles is not
available; "the effects of a transition layer, of roughness, and those of poroelasticity are
similar": all lower the reflection coefficient; separating them "may only be possible in the
laboratory".

### 2.6 Coherent and incoherent reflection (eq. 13.14, App. L.3; confirmed)

- Kirchhoff and small-slope: V_wwc(theta) = V_ww(theta) exp(-2 k_w^2 h^2 sin^2 theta_i) (13.14)
  (Eckart 1953), h the rms relief. Book's caution (p. 335): used as the total bottom reflection it
  treats the incoherent energy as lost, a good approximation only if scattering is at wide
  angles from specular; otherwise the scattered field is an important part of the down-range field.
- Incoherent reflection coefficient (L.3): <|P_s(r_s)|^2> = 2 s0^2 |V_wwi(theta_i)|^2 / (d_i + d_s)^2 (L.21),
  |V_wwi|^2 = (1/k_w^2) Int sigma d2(DeltaK) (L.22) = (1/sin theta_i) Int sigma d2 Omega_s (L.23);
  |V_wwi|^2 + |V_wwc|^2 < 1 (L.24); near specular (Kirchhoff/small slope)
  |V_wwi|^2 = |V_ww|^2 (1 - exp(-Delta k_z^2 h^2)) (L.26) = |V_ww|^2 - |V_wwc|^2 (L.27), Delta k_z = 2 k_w sin theta.
  That is: if scattering is near specular, the coherent plus incoherent energy equals the flat
  interface value, and the scattered part spreads as an image source (1/(d_i + d_s)^2).

### 2.7 Chapter 15: penetration of the seafloor (confirmed)

**Background (pp. 407-410).** For a fast sediment, penetration is substantial above the critical
angle and slight below it (evanescent wave, depth of order a wavelength, practical below 10 kHz
only). Yet targets buried in sand are detected below the critical angle: Fig. 15.1 (SAS at 20
kHz, Piper et al. 2002): a target with its top 50 cm deep at 3.3-4.3 deg grazing, and one 15 cm
deep at 8-10 deg, ripples of about 70 cm wavelength. Mechanisms proposed (Fig. 15.2): Biot slow
wave refraction (no critical angle), roughness scattering, volume scattering, both. SAX99 and lab
experiments show sub-critical penetration at high frequency "is due to interface scattering" and
found no fast slow wave (about 1000 m/s). Fig. 15.3 (Maguer et al. 2000b, 25 cm depth): at 31.5
deg (above critical) the field is near the interface level at all frequencies; at 16 deg it
falls with frequency at the evanescent rate, then stops falling and stays 15-30 dB down
(anomalous penetration from roughness).

**Single-realisation and averaged perturbation model (Moe 1996, Moe and Jackson 1998; Sect. 15.1).**
- Zeroth order: P_0(R, z) = [1 + V_ww(K_i)] exp(i K_i.R - i k_p beta_p(K_i) z) (15.1): evanescent below
  the critical angle, refracted above.
- First order in the sediment: P_s(R, z) = Int T_wp^(1)(K_s, K_i) exp(i K_s.R - i k_p beta_p(K_s) z) d2K_s (15.2),
  T_wp^(1) = (i k_w / beta_w(K_s)) A_wp(K_s, K_i) F(K_s - K_i) (15.3), F the Fourier transform of the relief,
  A_wp(K_s, K_i) = (1/2)[1 + V_ww(K_i)][1 + V_ww(K_s)] x
  { (1 - 1/a_rho) [ K_s.K_i / k_w^2 + beta_p(K_i) beta_w(K_s) / a_p ] - 1 + 1 / (a_p^2 a_rho) } (15.4)
  (zoomed image checked).
- Sinusoidal ripple (Sect. 15.1.1): f(R) = sqrt(2) h0 cos(K0.R + theta0) (15.7), K0 = (2 pi / lambda0)(cos phi0, sin phi0) (15.8).
  Diffracted wave K_s = K_i - K0: cos theta_ps = nu_p (cos theta_wi - lambda_w / lambda0) (15.5) (lossless,
  propagation normal to the crests), real only if lambda_w > lambda0 (cos theta_wi - 1/nu_p) (15.6):
  "a high-frequency cutoff". P_-(R, z) = (k_w h0 / (sqrt(2) beta_w(K_s))) A_wp(K_s, K_i) exp(-i theta0 + i K_s.R - i k_p beta_p(K_s) z) at K_s = K_i - K0 (15.9);
  P = P_0 + P_- + P_+ (15.10). Fig. 15.6: h0 = 0.01 m, lambda0 = 0.48 m, 10-40 kHz, 17 cm depth:
  one-way loss about 10-25 dB below the critical angle, depending on position along the ripple.
- Formally averaged (Sect. 15.1.2): cross section for scattering into the seafloor
  sigma_wp(K_s, K_i) = (k_w^4 / nu_p^2) |A_wp|^2 W(K_s - K_i) (15.11) (W the 2-D roughness spectrum, App. D conventions);
  K_i = k_w (R - R_i)/|r - r_i| (15.12), K_s = k_w (R_s - R)/|r_s - r| (15.13);
  CW: <|P_s(r_s)|^2> = Int sigma_wp |P_i(r)|^2 |r_s - r|^-2 exp(-2 k_p'' |r_s - r|) d2R (15.14);
  pulsed, omnidirectional source:
  <|P_s(r_s, t)|^2> = s0^2 Int sigma_wp |u(t - t_d)|^2 exp(-2 k_p'' |r_s - r|) / (|r - r_i|^2 |r_s - r|^2) d2R (15.15),
  t_d = (|r - r_i| + |r_s - r| / nu_p) / c_w (15.16); k_p'' at the centre frequency. Far-field
  conditions of Sect. J.2 assumed (interface in the source's far field; receiver a few wavelengths
  from the interface; broad angular dependence).
- Fig. 15.8 example: SAX99 parameters (Table 13.1), spectrum (D.7) with K0 = 10 1/m, c_w 1530 m/s,
  source 4 m above, 20 m horizontal, grazing about 11 deg (critical about 30 deg),
  u(t) = exp(-t^2 / t_s^2), t_s = 100 us, SL 190 dB re 1 uPa at 1 m: mean-square pressure peaks
  about 133-141 dB at 5-40 cm depth; the zeroth-order field is "evanescent and entirely negligible".
- Tests (Sect. 15.1.3): consistent with Chotiros (1995a) data qualitatively (Thorsos et al.
  2000a); single-realisation perturbation consistent with Maguer et al. (2000a) rippled sand; SAX99
  diffraction angles (15.5) match data (Fig. 15.10); SAX99 field levels at 30 kHz, 20 deg nominal
  (critical about 30 deg): loss about 5-25 dB at 6, 17, 31 cm mean depth, within the spread of ten
  Monte Carlo realisations (Fig. 15.11). Lopes et al. (2005): first order works 15-40 kHz for
  ripples 75 cm wavelength, 1.7 cm rms; at 2.5 cm rms second order improves agreement noticeably.
  The deep Fig. 15.1 detections remain unexplained (environment, burial or target physics).
- Accuracy (Sect. 15.2, Fig. 15.12; Thorsos et al. 2000a): exact 2-D Monte Carlo, w1 = 0.0002 m,
  L_c = 0.227 m, c_w 1500, 20 kHz, k_w h = 1.0, nu_p 1.13, a_rho 2.0, delta_p 0.0155, 20 deg: first
  order is within about 1 dB for scattering into water, and better for scattering into the
  sediment (small speed contrast).
- Research issues (Sect. 15.3): unexplained deep detections in SAS exercises; higher-order
  perturbation and small-slope for large ripples; penetration measurements of dispersion,
  attenuation and particle motion can discriminate sediment theories.

### 2.8 Appendix F: transducers (confirmed)

Far field r > d^2 / lambda (F.1), from Delta theta = lambda / d (F.2) and r Delta theta > d (F.3);
when the spread is set by curvature or beamforming, r > lambda / (Delta theta)^2 (F.4) (also
end-fire and parametric arrays). Fig. F.1/F.2: 5 cm radius piston at 100 kHz, far field beyond
about 65 cm. |P_i| = sqrt(2) s0 |b_x(theta, phi)| / r (F.5), theta from the horizontal, directivity
in earth axes (changes with tilt); D_x = 20 log10 |b_x| (F.6), unity on the MRA; SL = 20 log10(s0 / (p_ref r_ref))
(F.7), dB re 1 uPa at 1 m; D_r = 20 log10 |b_r| (F.8); RS = 20 log10 (|V| / |P|) dB re V/uPa (F.9);
reciprocal transducers have identical transmit and receive patterns.

### 2.9 Appendix G: measurements (confirmed)

**Reflection (G.1).** Image source: |P| = sqrt(2) s0 |V_ww(theta)| |b_x(theta, 0)| exp(-k_w''(r1 + r2)) / (r1 + r2) (G.1);
sonar equation 20 log10 |V_r| = SL + RS - PL - L + D_x(theta, 0) + D_r(theta, 0) (G.2),
PL = 20 log10((r1 + r2) / r_ref) + alpha_w (r1 + r2) (G.3). Only |V| is measured (phase needs
sub-wavelength path knowledge). Average squared pressure over many patches (Sect. 11.1); beyond
a few tens of metres replace (G.3) by ray-traced loss.

**Scattering (G.2).** <|V_r(t)|^2> = s_r^2 Int |P_i|^2 sigma exp(-2 k_w'' |r_s - r|) / |r_s - r|^2 |b_r(theta_r, phi_r)|^2 d2R (G.4)
(incoherent sum, justified in Sect. J.2); ensonified region from |r - r_i| + |r_s - r| = c_w t_0(R) (G.5)
and the pulse window (G.6) (printed with a sign misprint, Sec. 6), monostatic r_t - c_w tau / 4 < |r - r_i| < r_t + c_w tau / 4,
r_t = c_w t / 2 (G.7)-(G.8). Moderate to small grazing (c_w tau / 2 << r_t): strip width
w = c_w tau / (2 cos theta), theta = arcsin(z_i / r_t) (G.9); the cosines cancel in the area;

    <|V_r(t)|^2> / 2 = (s_r s0)^2 sigma exp(-4 k_w'' r_t) / r_t^3 (c_w tau / 2) |b_r(theta,0) b_x(theta,0)|^2 Psi   (G.11)
    Psi = Int |b_r b_x|^2 dphi / |b_r(theta,0) b_x(theta,0)|^2                                                   (G.12)
    10 log10(<|V_r|^2>/2) = SL + RS + S_b + D_x(theta,0) + D_r(theta,0) - 2 alpha_p r_t
                            - 30 log10(r_t / r_ref) + 10 log10(c_w tau Psi / (2 r_ref))                        (G.13)

(the "-2 alpha_p r_t" is printed with subscript p; it is the water absorption, consistent with
exp(-4 k_w'' r_t) in (G.11)). With a Gaussian azimuthal pattern and tilt theta_0:
Psi = (cos(theta - theta_0) / (2 cos theta)) sqrt(pi / ln 2) Psi_3 (G.28), Psi_3 the 3-dB full width of the
product pattern; Psi ~ Psi_3 ignores sqrt(pi / ln 2)/2 (0.3 dB) and the angular factor;
Psi_3 = 1 / sqrt(1/Psi_x^2 + 1/Psi_r^2) (G.29). Valid if the horizontal beamwidth is about 20 deg or
less (G.24)-(G.27). Time series (Fig. G.4): fathometer return, sidelobe return, main-lobe return;
only the main lobe is used for S_b.

**Near-vertical (G.2.2, Jackson and Nesbitt 1988).** <|V_r|^2> = <|V_rr|^2> + <|V_rv|^2> (G.30);
Gaussian beam b(theta) = exp(-(pi/2 - theta)^2 / (2 sigma_b^2)) (G.35); near-vertical cross
section parameterised as sigma = |V_ww(90)|^2 exp(-(pi/2 - theta)^2 / (2 sigma_s^2)) / (8 pi sigma_s^2) (G.36)
(a parameterisation, sigma_s an angular width, not an rms slope); roughness return
<|V_rr|^2> = V1^2 |V_ww(90)|^2 g(t - t0, T_sb, tau) / (1 + sigma_s^2 / sigma_b^2) (G.43),
g = 0 (t <= 0), 1 - e^(-t/T) (0 < t <= tau), e^(-(t - tau)/T) - e^(-t/T) (t > tau) (G.44),
T_sb = sigma_sb^2 t0 (G.45), 1/sigma_sb^2 = 1/sigma_s^2 + 1/sigma_b^2 (G.38), t0 = 2H / c_w (G.40),
V1^2 = (s0 s_r)^2 exp(-4 k_w'' H) / (2 H^2) (G.46) (the flat, |V| = 1 return); volume return
<|V_rv|^2> = 8 pi sigma_v V1^2 |V_wp(90)|^4 H / (a_rho^2 (sigma_p^-2 - sigma_b^-2)) [sigma_b^2 g(t - t0, T_b, tau) - sigma_p^2 g(t - t0, T_p, tau)] (G.54),
T_b = sigma_b^2 t0, T_p = sigma_p^2 t0, sigma_p^2 = 1 / (4 k_p'' H) (G.55)-(G.57), V_wp(90) = 1 + V_ww(90) (G.48),
sigma_v the volume scattering strength, |V_wp|^4 / a_rho^2 the two-way interface transmission.
Fig. G.7: 20 kHz, 3 ms, H = 10 m, sigma_b 0.445, c_w 1540, a_rho 2.0, a_p 0.99, delta_p 0.01,
sigma_2 0.002, sigma_s 0.13. Four fitted parameters: |V_ww(90)|, sigma_s, delta_p, sigma_2, with
sigma_b, s0, s_r calibrated and H from the first arrival; the data are averages of the squared
envelope over many pings with leading edges aligned.

**Statistical error (G.2.3).** Assuming Gaussian scattered fields, each intensity sample is
exponentially distributed (standard deviation = mean); with N independent samples the standard
deviation of the average is mean / sqrt(N); estimate it by measured mean / sqrt(N). "if N = 16,
the standard deviation is one-fourth of the mean, or roughly 1 dB". In dB (ours, from that):
+1 sigma = 10 log10(1 + 1/sqrt(N)): 0.97 dB at N = 16, 0.5 dB at N = 67, 0.25 dB at N = 260.

---

## 3. Against our simulator

### 3.1 Conventions and the fluid reflection coefficient

| Item | Ours | Book | Verdict, how checked |
|---|---|---|---|
| Fluid V | `bottom.reflection(g, rho, nu, delta)`; `propagation.rayleigh_reflection`; C++ `libs/sim/src/paths.cpp:140` | (8.49)-(8.53), (8.25) | **agrees**: max difference 7e-16 to 1.8e-15 over 0.5-90 deg for four sediments (medium sand, Fig. 8.3 sand and mud, rock). Root branch Im >= 0 in both |
| dB/wavelength -> delta | `delta = alpha / (40 pi log10 e)`; `bottom.DB_PER_WAVELENGTH = 40 pi / ln 10` | (8.27) alpha_p_lambda = 40 pi delta_p / ln 10, per sediment wavelength | **agrees**; REFERENCES.md 12.4 calls the conversion "ours" / UA.jl "via secondary": the book now **confirms** it |
| dB/m -> delta | `parameters_from_physical`: delta = alpha c_sed / ((20/ln10) 2 pi f) | (8.26) with nu_p the phase-speed ratio | **agrees**: 10 dB/m at 20 kHz, 1767 m/s -> delta 0.016189 -> back to 10.000 dB/m; 0.8835 dB/lambda; k = 0.500 dB/m/kHz |
| Complex speed c/(1 + i delta) | `c_b = speed / (1 + 1j delta)` | (8.25) | **agrees** (REFERENCES 12.4 row Olson-Jackson: now also confirmed in this book) |
| Coherent reflection | `acoustics.rough_coherent_reflection`, `surface_coherent_reflection`: exp(-Gamma^2/2), Gamma = 2 k h sin g | (13.14) exp(-2 k_w^2 h^2 sin^2 theta) | **agrees** (0.24569 both at 30 deg, 40 kHz, h 1 cm). The book's caveat is the one D-38 states. The bottom use in `timespread.py:597` takes the speed `c_b or c`, where `c_b` is the water speed at the bottom depth (`timespread.py:865`, `env.speed_at(env.depth)`): the book's k_w, **agrees** |
| Wood's equation | `acoustics.bubbly_sound_speed` (gas in water, via arlpy/UA.jl) | (8.70)-(8.72) (grains in water) | same form; the book confirms the mixture rule 1/sqrt(mean density x mean compressibility) |
| Energy flux | not used | (8.31); identity in Sec. 2.2.3 | absent; useful as a test of any transmission code |

### 3.2 Grain-size relations: the book prints what our table holds

`configs/sediments.yaml` rows (via UnderwaterAcoustics.jl and Cintron, "via secondary") were
compared with (11.1)-(11.3) evaluated at each row's `mean_grain_size_phi`, with delta from k by
(8.26) and c_w = 1528 m/s (script `fix/jr3_tr9407.py` in the session scratchpad):

| Rows | rho | nu | delta |
|---|---|---|---|
| 18 rows sandy_gravel (-1) ... sandy_silt (5.0) | equal to 4 digits | equal to 4 digits | equal to 4 digits (with 0.4556 for Mz < 0; the printed 0.4566 would give 0.01709 against the table's 0.01705 at -1 phi) |
| medium_sand (1.5, from arXiv:1809.08459) | 1.8450 vs 1.8446 | 1.1780 vs 1.1782 | 0.01619 vs 0.01624 (the row's 10 dB/m is rounded) |
| 7 rows from 5.5 to 9 phi | equal to 3-4 digits | **table 0.7 % below the printed (11.2)**: e.g. clay 0.9800 vs 0.9872 | table about 1 % below (follows nu) |

So: the book **confirms** (prints) the density ratio and the attenuation regression behind all
rows, and the speed ratio below 5.3 phi. Above 5.3 phi the table matches -0.0024324 Mz + 1.0019,
which is continuous with the cubic at 5.3 (0.98901 both); the printed 1.0091 jumps by 0.007 there.
I take the table as right and the book as misprinted (not proven: the cubic's continuity is the
only evidence). Also: `bottom.sediment_from_grain_size` interpolates the half-phi table linearly
(REFERENCES.md Sec. 10 "wanted" row "Regressions of density, speed and loss against grain size"):
that row is now **closed** by this book (formulas (11.1)-(11.3), attributed by the book to TR 9407
and Hamilton 1972). The book also says these relations are not geoacoustic regressions but
ad hoc fits that already lower speed and density to match reflection and backscatter data (p. 314):
relevant to EDFM (Sec. 3.4).

### 3.3 Elastic bottoms (rock, pebbles)

Ours: `rock` and `pebbles` are fluid `reflection_only` rows (UA.jl); every reflection on them
(image-source, Bellhop `bottom_speed ... 0.0 ...` with zero shear in `sim/bellhop.py:220`) is fluid.
Absent: (9.34)-(9.40). Computed (book formulas; delta_t = 0.1 as in the book's Fig. 9.4;
bottom loss in dB, fluid / elastic):

| Case | 5 deg | 10 | 20 | 30 | 40 | 45 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|
| Fig. 9.4 rock (nu_t 1.3) | 0.0 / 1.8 | 0.0 / 3.1 | 0.0 / 5.0 | 0.0 / 10.1 | 0.0 / 11.5 | 0.0 / 6.8 | 0.1 / 4.3 | 2.8 / 2.8 |
| our `rock`, cs = 0.4 cp (1528 m/s) | 0.0 / 3.1 | 0.0 / 6.8 | 0.0 / 18.8 | 0.0 / 11.7 | 0.0 / 7.9 | 0.0 / 7.1 | 0.1 / 3.6 | 2.8 / 2.8 |
| our `rock`, cs = 0.5 cp (1910 m/s) | 0.0 / 1.8 | 0.0 / 3.3 | 0.0 / 6.1 | 0.0 / 15.4 | 0.0 / 9.2 | 0.0 / 6.4 | 0.1 / 4.4 | 2.8 / 2.8 |
| our `rock`, cs = 0.6 cp (2292 m/s) | 0.0 / 1.4 | 0.0 / 2.0 | 0.0 / 2.4 | 0.0 / 3.2 | 0.0 / 8.9 | 0.0 / 17.3 | 0.1 / 3.8 | 2.8 / 2.8 |
| "generic rock" of the sediments.yaml header (cp 3600, cs 1900, rho 2500, delta 0.0018 / 0.085; via secondary) | 0.0 / 1.4 | 0.0 / 2.6 | 0.0 / 5.1 | 0.0 / 13.7 | 0.0 / 9.0 | 0.0 / 6.4 | 0.0 / 4.5 | 3.0 / 3.0 |
| our `pebbles`, cs 200 m/s | 0.0 / 0.1 | 0.0 / 0.2 | 0.1 / 0.2 | 0.1 / 0.2 | 0.1 / 0.1 | 0.1 / 0.1 | 2.0 / 2.0 | 3.9 / 3.9 |
| our `pebbles`, cs 400 m/s | 0.0 / 0.5 | 0.0 / 0.8 | 0.1 / 1.0 | 0.1 / 0.7 | 0.1 / 0.4 | 0.1 / 0.3 | 2.0 / 2.1 | 3.9 / 3.9 |
| our `pebbles`, cs 800 m/s | 0.0 / 2.2 | 0.0 / 4.3 | 0.1 / 7.3 | 0.1 / 6.5 | 0.1 / 3.5 | 0.1 / 2.1 | 2.0 / 2.5 | 3.9 / 3.9 |
| Fig. 9.3 sand, cs 250 m/s | 0.3 / 0.3 | 0.5 / 0.6 | 0.8 / 0.9 | 5.0 / 5.4 | 7.0 / 7.5 | 7.4 / 7.9 | 8.2 / 8.4 | 8.5 / 8.5 |

Shear speed is not in our rows and not confirmed for gravel (unconsolidated: probably low; the
book gives rock only, p. 169 text). Consequence: on rock, long-range multi-bounce propagation
(8 kHz flank array, Popoto scenes over gravel/rock) has 1.4-7 dB per bounce too little loss at
5-10 deg; ten bounces, 14-70 dB. The book confirms REFERENCES.md row "Olson, Lyons and Saebo"
on the approach (elastic for rock) but gives no rock roughness or scattering law in my chapters.

### 3.4 Sand: EDFM against our fluid model

Ours: fluid with real density (`rho`), complex speed, delta frequency independent (linear
attenuation). Absent: Biot, EDFM, Buckingham. Computed with Table 10.1 (c_w = sqrt(K_w / rho_w) =
1530.1 m/s, rho / rho_w = 2.002). "Fluid" = true density with the EDFM complex speed (the book's
Fig. 10.5 convention); "EDFM" = rho_eff with the same speed. Bottom loss (dB):

| f | EDFM speed, attenuation | nu, delta | rho_eff / rho | 10 deg fluid/EDFM | 30 deg | 45 deg | 90 deg |
|---|---|---|---|---|---|---|---|
| 8 kHz | 1744 m/s, 4.2 dB/m | 1.140, 0.0169 | 0.892 + 0.030i | 0.51 / 0.80 | 2.54 / 3.20 | 6.61 / 7.61 | 8.13 / 9.34 |
| 20 kHz | 1757, 7.1 | 1.148, 0.0114 | 0.879 + 0.020i | 0.33 / 0.51 | 1.76 / 2.36 | 6.46 / 7.54 | 8.06 / 9.42 |
| 40 kHz | 1763, 10.4 | 1.152, 0.0084 | 0.872 + 0.015i | 0.23 / 0.36 | 1.25 / 1.79 | 6.39 / 7.51 | 8.03 / 9.47 |
| 120 kHz | 1770, 18.4 | 1.157, 0.0050 | 0.866 + 0.009i | 0.13 / 0.21 | 0.68 / 1.03 | 6.31 / 7.48 | 8.00 / 9.51 |
| 330 kHz | 1774, 31.0 | 1.159, 0.0031 | 0.862 + 0.005i | 0.08 / 0.13 | 0.38 / 0.57 | 6.27 / 7.46 | 7.98 / 9.53 |

The 20 kHz pair 8.06 / 9.42 dB reproduces Fig. 10.5 and Fig. 10.7 (read: about 8.1 and 9.4).
Backscattering strength with our default model (`bottom.backscattering_strength_db`, small-slope +
volume) and `medium_sand`'s w2, gamma2, sigma2 on the SAX99 sediment, EDFM minus fluid:
roughness term -1.1 to -2.9 dB (largest at 10 deg); total -0.0 to -2.9 dB (the volume term rises
because more sound enters): 8 kHz -1.7 / -0.8 / 0.0 / 0.0 dB at 10 / 20 / 30 / 45 deg; 40 kHz
-2.6 / -1.5 / -0.8 / -0.6; 120 kHz -2.8 / -1.7 / -1.1 / -1.1; 330 kHz -2.9 / -1.8 / -1.2 / -1.4.

Frequency dependence of attenuation: EDFM/Biot attenuation grows about as f^0.5 above f_c
(18.4 dB/m at 120 kHz, 31 at 330 kHz), our rows grow as f (medium sand row: 0.5 dB/m/kHz, i.e. 60
and 165 dB/m). The SAX99 data (Fig. 10.10, read approximately) lie between: about 10 dB/m at 38
kHz, 30-60 dB/m at 100-260 kHz, 92 dB/m at 400 kHz (p. 306). This matters for the volume term
(sigma_v is proportional to attenuation by construction) and for buried targets.

**Do not stack EDFM on the TR 9407 rows.** The simple EDFM (10.82) applied to our `medium_sand`
row (beta = 0.50 from its density with rho_g 2690) gives rho_eff / rho_w = 1.341 (0.73 of the
row's 1.845): normal-incidence loss 8.6 -> 13.0 dB and backscatter 0 to -7.4 dB. That double
counts: the TR 9407 relations already lower density and speed to fit reflection data (p. 314),
and for fine sand (row density 1.45, implied porosity 0.72) the result is meaningless. EDFM must
start from measured (or Ch. 5 regression) density, porosity and speed.

Cost of EDFM: simple form = one real number per row (rho_eff), no runtime cost; full form = a
complex, frequency-dependent density from 7 inputs (beta, alpha, kappa or a, eta, rho_w, rho_g,
K_g, K_w) evaluated once per frequency (Bessel functions of complex argument), and complex
density support in `bottom.reflection` (works as written), `_perturbation_factor` and
`volume_strength` (to check), `rayleigh_reflection`, C++ `paths.cpp`, and Bellhop (real density
only: pass Re, or tabulate a reflection coefficient file).

### 3.5 Buried targets: the `sediment` tier

Ours (`propagation.SedimentProvider`, C++ `core.sim_sediment_eigenrays`): one Snell ray per point,
amplitude `transmission` (default 1) / (c t), no sediment attenuation, no angle dependence, no
density. Scenarios `buried_cylinder`, `buried_objects` (subbottom sonar 10-30 kHz, 1.5 m above, ±0.9
m raster: steep angles, above the critical angle 28 deg of 1700/1500 m/s) say so in their header.

What the book says, and what it means for us:
1. Above the critical angle (our two scenarios): field = V_wp(K) exp(-i k_p beta_p z) (15.1),
   with sediment attenuation k_p''. Missing: medium sand at 20 kHz is 10 dB/m, so 6 dB two-way at
   0.30 m depth, 9 dB at 30 kHz; the two-way interface factor at normal incidence is
   |V_wp|^2 / a_rho = 1.017 (+0.07 dB), so transmission matters little near vertical but grows
   toward the critical angle (|1 + V| = 1.78 at 25 deg, 1.53 at 40 deg, one way).
2. Below the critical angle: no geometric ray exists. Our provider still finds one, entering at
   the critical angle and running horizontally through the sediment (checked: source 10 m up,
   point 0.3 m deep at 40 m: launch 31.9 deg, amplitude -31.9 dB = 1/slant range; direct-line
   grazing 14 deg). The flat-interface field is evanescent: decay
   (20 / ln 10) k_w Im sqrt(1/a_p^2 - cos^2 theta_w) dB/m: medium sand, 10 deg, 145 / 364 / 728 / 2183 dB/m at
   8 / 20 / 40 / 120 kHz (|P0| at 0.3 m: -43, -109, -218, -654 dB). The book's measured and
   modelled sub-critical field comes from roughness: 15-30 dB below the incident level at 25 cm
   (Fig. 15.3), 5-25 dB (Fig. 15.11, 30 kHz, 20 deg), about 23-31 dB (Fig. 15.8, 20 kHz, 11 deg,
   SAX99 roughness). Our tier is therefore wrong in level by order +20 dB to +30 dB (too loud, no
   attenuation over metres of sediment path) or, once attenuation is added, by -100 dB (too
   quiet, no roughness penetration), and wrong in arrival structure (the real field arrives from
   the patch above the target, delayed by (15.16)).
3. The lateral wave our ray resembles is "unimportant at high frequencies" (p. 262).
4. First-order fix (WP1 and WP2 below): add attenuation and V_wp to the ray; mark points with no
   real ray (or a ray whose sediment leg is mostly horizontal) as sub-critical; give them the
   evanescent zeroth-order term and an incoherent roughness-penetration term (15.11)-(15.16)
   using our existing first-order perturbation solution, whose sediment-side amplitude B
   (bottom.py docstring, lines 36-39) is the same physics as A_wp (15.4).

### 3.6 Near-field validity of plane-wave coefficients (image-source and Bellhop tiers)

Book criterion (Sect. 2.2.5): fine when source and receiver are many wavelengths from the floor;
at about 13 wavelengths (Fig. 11.1, 200 kHz at 10 cm) errors appear near and below the critical
angle; at 33 wavelengths none visible. Our cases: FLS 120-330 kHz at metres (hundreds of
wavelengths): fine. Flank array 8 kHz (lambda 19 cm) at tens of metres: fine. Subbottom 20 kHz at
1.5 m = 20 wavelengths, near-vertical: fine at steep angles, marginal near the critical angle.
SAS at a few metres altitude: fine. Not a gap for the shipped scenarios.

### 3.7 Measurements and statistical error (real data, sonar-sul, `tools/fit_sediment.py`)

`tools/fit_sediment.py` takes `sigma_db` per value or `--sigma-db`; nothing in the repository says
how to set it. G.2.3 gives it: sigma_db = 10 log10(1 + 1/sqrt(N)) for N independent intensity
samples (Gaussian field). `sim/reverb.k_cell_area` states the cosines cancel in the ensonified
area: **agrees** with Sect. G.2.1. The book's monostatic sonar equation (G.13) with Psi from
(G.28)-(G.29) is the estimator for S_b from a calibrated sonar; the near-vertical fit (G.43),
(G.54) gives |V(90)|, sigma_s, delta_p, sigma_2 from an echosounder time series.

---

## 4. What this closes or changes in open issues

| Issue / row | Effect |
|---|---|
| sonar-3k1 (bottom as published; layered) | Elastic half-space reflection: formulas now available (9.34)-(9.42) -> WP4. Layered: **not in this book's Chs. 8-11**; the recursion of Sec. 2.2.9 is standard (ours), cite Olson-Jackson arXiv:2009.12420 -> WP7. The book's statements on transition layers (6 dB at 20 kHz from 3 cm) give a target number but not the profile. Regime interpolation, large-roughness law, volume slope/shadow corrections, bistatic: Ch. 13/14/J, not covered here. |
| REFERENCES.md Sec. 10 "Regressions of density, speed and loss against grain size" | **Closed**: (11.1)-(11.3) printed (two misprints, Sec. 6). The 20 table rows agree. -> WP5 |
| REFERENCES.md 12.4 "dB per wavelength to delta ... ours" | **Confirmed** by (8.27); delta definition (8.18); complex speed (8.25) |
| D-38 / surface coherent loss | Form exp(-2 k^2 h^2 sin^2) confirmed (13.14) with the same caveat; the book is about the seafloor and says nothing about bubbles or surface wave statistics (sonar-9le untouched) |
| sonar-sul (real data), `fit_sediment.py` | Estimators (G.2), (G.13), (G.28)-(G.29); statistical error G.2.3; reflection by averaging squared pressure over patches (Sect. 11.1) -> WP6 |
| Buried-object scenes (SIMULATION.md "Refraction into sediment") | Tier wrong below the critical angle and missing attenuation above it -> WP1, WP2 |
| Elastic bottoms (STATUS, 3k1) | Quantified: rock 2-19 dB per bounce at 5-45 deg -> WP4 |
| "medium_sand volume parameter 0.001 vs 0.002" (D-90), Lambert parameters, K statistics (sonar-6fc), ping-to-ping coherence (sonar-6t5), relief reverberation (sonar-7iy), Kirchhoff/composite cut-off, bistatic | Not in my chapters. (G.2.2's Fig. G.7 uses sigma_2 = 0.002 for a sand-like example: an illustration, not a sediment table value) |
| Sediment fit tool still uses perturbation (STATUS item 30) | Not addressed by these chapters |

---

## 5. Work packages (ordered by value to realism per effort)

### WP1. Honest sediment tier above the critical angle (S)
- Goal: `SedimentProvider` amplitude = V_wp(theta_w) (8.55) x exp(-k_p'' L_sed) x spreading, with
  the reciprocity factor 1/a_rho on the return leg (8.68), and the sediment named from
  `configs/sediments.yaml` (density, nu, delta) instead of a bare `sediment_speed`.
- Formulas: (8.46), (8.49)-(8.55), (8.20)/(8.26) for k_p'' = k_w delta / nu (per metre of
  sediment path), (8.68).
- Files: `python/gpusonar/sim/propagation.py` (SedimentProvider), C++ `core.sim_sediment_eigenrays`
  (libs/sim) with the engine parity test; scenarios `buried_cylinder.yaml`, `buried_objects.yaml`.
- Tests first: energy identity 1 - |V|^2 = |V_wp|^2 Re{sin theta_p/(a_rho a_p)}/sin theta_w (lossless, to 1e-12);
  V_wp = 1 + V; a_rho = 1, nu = 1 gives unit transmission; attenuation over a vertical path of
  0.3 m at 20 kHz in medium sand = 3.0 dB one way; Python = C++.
- Moves default results: yes (both buried scenes: about -6 to -9 dB on the targets at 20-30 kHz,
  sediment clutter likewise): owner first; ship as opt-in `transmission: model` and prepare
  before/after target peak levels for the two scenes.
- Must not: silently change `transmission` default; claim sub-critical correctness (WP2).

### WP2. Sub-critical penetration: evanescent term plus roughness scattering (M-L)
- Goal: for points in the sediment reached at grazing below the critical angle, replace the
  critical-angle ray by (a) the evanescent zeroth-order field (15.1) and (b) the incoherent
  roughness-scattered field (15.14)/(15.15) with sigma_wp (15.11), A_wp (15.4), the scene's roughness
  spectrum W (the bottom's w2, gamma2, our conventions: 2-D, rad/m), delay (15.16).
- Implementation hint: our perturbation solution already has the sediment amplitude B
  (bottom.py docstring); prove B equals A_wp of (15.4) up to the book's normalisation first. For
  the simulator, (15.15) is a patch sum over the interface above the target: the same machinery as
  the relief reverberation (`reverb_relief.py`), with one leg in the sediment.
- Tests first: (i) sinusoidal ripple: diffraction angle (15.5) and its cutoff (15.6) as a closed
  form; (ii) reproduce Fig. 15.6 qualitatively (h0 0.01 m, lambda0 0.48 m, 10-40 kHz, 17 cm: 10-25
  dB one-way loss below critical; position dependence); (iii) reproduce Fig. 15.8 (SAX99 Table 13.1
  parameters: from the Ch. 13 agent's file; t_s = 100 us, SL 190 dB, source 4 m up, 20 m off:
  peak 133-141 dB at 5-40 cm); (iv) h -> 0 gives zero scattered field; (v) above critical the
  zeroth order dominates.
- Moves defaults: new behaviour, opt-in; buried scenes at shallow grazing become possible (SAS
  buried-target detection, Fig. 15.1 style). Owner decides whether shipped scenes use it.
- Depends on: WP1. Must not: use the coherent flat transmission below the critical angle.

### WP3. EDFM option for sand (S for simple, M for full)
- Goal: opt-in `effective_density` for fluid sediments: simple (10.82) (inputs: porosity, K_g/K_w,
  default 13.4) and full (10.70), (10.75)-(10.76), (10.81) with (10.13)-(10.16) (Table 10.1 as the
  documented default set "sax99_biot").
- Files: `python/gpusonar/sim/bottom.py` (a function `effective_density(f, ...)` and use in
  `reflection`, perturbation, volume), `configs/sediments.yaml` (new row `sax99_sand` with measured
  density 2048 kg/m3, porosity 0.385, Table 10.1 Biot block), C++ only if the image-source tier
  uses it (`paths.cpp`).
- Tests first: F(eps) -> 1 at low frequency; rho_eff -> rho at 10 Hz; high-frequency limit
  rho_eff -> (rho alpha rho_w/beta - rho_w^2)/(alpha rho_w/beta + rho - 2 rho_w) (1755.5 kg/m3);
  Biot fast wave at 20 kHz 1763 m/s, 7.0 dB/m (p. 304; our check 1762.7, 6.96); bottom loss at 20
  kHz, normal incidence: EDFM 9.42 dB, fluid 8.06 dB (Figs. 10.5, 10.7); simple EDFM equals rho in a
  mud whose speed obeys Wood (p. 302); Fig. 10.8 range +-0.5 dB for K_g 32-75 GPa.
- Moves defaults: no if opt-in. Prepare for the owner: on the SAX99 sand, reflection loss +1.2 to
  +1.6 dB at normal incidence and +1.0 to +1.2 dB at 45 deg; backscatter -0 to -2.9 dB (Sec. 3.4).
- Must not: apply EDFM to TR 9407 rows (double count, Sec. 3.4); change the attenuation law of
  existing rows silently (EDFM speed/attenuation dispersion is a separate opt-in).

### WP4. Elastic flat-interface reflection for rock (S-M)
- Goal: `reflection` / `rayleigh_reflection` / `paths.cpp` take optional shear speed and shear
  loss and use (9.34)-(9.40); Bellhop gets the shear column (`sim/bellhop.py:220` writes 0.0 now)
  and shear attenuation.
- Tests first: mu -> 0 (nu_t -> 0) recovers the fluid coefficient; Fig. 9.4: minimum |V| 0.052 at
  35.7 deg, critical angles arccos(1/2.5) = 66.4 deg and arccos(1/1.3) = 39.7 deg; Fig. 9.3:
  differences below 0.5 dB; Bellhop with shear against our coefficient on a flat bottom.
- Config: rock shear speed is not in any row: book p. 169 (text) 40-60 % of cp; the header's
  "generic rock" 1900 m/s (Olson et al., via secondary, attributed to TR 9407 Table IV.2) is the
  best sourced number; mark `unconfirmed_fields: [shear]`.
- Moves defaults: yes for any scene with a `rock` bottom (and `pebbles` if a shear speed is
  given): owner first; prepare per-bounce loss before/after (Sec. 3.3) and the transmission-loss
  change of `popoto_env_gravel_30m` if its bottom is rock/pebbles.
- Must not: give pebbles a shear speed without a source.

### WP5. Grain-size relations as formulas (S)
- Goal: `bottom.sediment_from_grain_size(phi)` evaluates (11.1)-(11.3) (+ (8.26) with c_w 1528)
  instead of interpolating the half-phi table; record the two misprints (Sec. 6) and use the
  continuous constants 1.0019 and 0.4556 that the tabulated rows follow.
- Tests first: all 20 rows reproduced (rho, nu to 4 digits; delta to 3 significant); continuity at
  1, 5.3 (and the printed discontinuity of (11.3) at 6.0 kept as printed, documented).
- Moves defaults: only between tabulated phi values (interpolation vs polynomial): measure and
  show the largest change before asking the owner.

### WP6. Measurement estimators and error bars for real data (S)
- Goal: a small module (or `tools/fit_sediment.py` option `--independent-samples N`) giving
  sigma_db = 10 log10(1 + 1/sqrt(N)) (G.2.3); a helper for S_b from calibrated data by (G.13) with
  Psi (G.28)-(G.29); reflection by averaging squared envelopes over N patches (Sect. 11.1), never dB.
  Add to docs/tasks/REAL_DATA.md.
- Tests first: N = 16 -> 0.97 dB; simulated Rayleigh speckle with N independent cells gives the
  stated spread; (G.13) inverts a synthetic reverberation level from our own simulator to S_b
  within 0.3 dB (the Psi approximation).
- Moves defaults: no.

### WP7. Layered fluid seafloor reflection (M; source: Olson-Jackson arXiv:2009.12420, not this book)
- Goal: opt-in layered stack in `bottom.reflection` / `rayleigh_reflection` (Sec. 2.2.9), used by
  image-source and the volume/roughness terms' transmission; a staircase for a density gradient
  (Lyons-Orsi type (7.36) profile from the Ch. 7 agent's file).
- Tests first: the four checks of Sec. 2.2.9; reproduce the book's quoted 6 dB at 20 kHz only if
  the Mourad-Jackson profile is found.
- Moves defaults: no (opt-in).

### WP8. Biot reflection as a test oracle (M, low priority)
- Goal: (10.66)-(10.69) solved numerically, used only in tests to show EDFM tracks Biot (Fig. 10.7)
  at our frequencies. Not for the runtime.

Order: WP1 (S, large error in buried scenes), WP4 (S-M, large error over rock), WP3 (S, 1-3 dB
everywhere on sand, opt-in), WP6 (S, needed by sonar-sul), WP5 (S, closes a wanted row), WP2
(M-L, needed for sub-critical buried targets), WP7 (M), WP8.

---

## 6. Not verified / unclear on the page

- (11.2), p. 314: last piece printed "-0.0024324 Mz + 1.0091" (zoomed image). Continuity with the
  cubic at Mz = 5.3 and our 7 tabulated rows (5.5-9 phi) need 1.0019. Treated as a misprint;
  not proven.
- (11.3), p. 314: first piece printed "k = 0.4566" (zoomed image); the next piece starts at 0.4556
  and our tabulated rows at -1 and -0.5 phi follow 0.4556. Treated as a misprint. (11.3) is also
  discontinuous at Mz = 6.0 as printed; our rows follow it.
- (10.39), p. 292: printed as k_q; dimensionally and by re-deriving from (10.31) the root is k_q^2.
  With k_q^2 the book's 20 kHz numbers are reproduced exactly.
- (10.47)-(10.48), p. 293: "k_t = w c_t", c_t = sqrt((rho - w rho_w / Omega) / mu) is a slowness, not
  a speed; consistent with (10.36).
- p. 319: "For a frequency of 250 kHz, rho_eff / rho = 0.657 + i 0.0046" from Table 10.1 in
  (10.75). Our evaluation gives 0.863 + 0.006i, consistent with Fig. 10.6 (about 1750/2050) and
  with the high-frequency limit 0.857. Either the text is a misprint or it used different inputs;
  unresolved.
- (G.6), (G.7), p. 497: both bounds printed with "t - tau/2" (and "r_t - c_w tau / 4") on both
  sides; the upper bound must be "+". (G.13) writes "-2 alpha_p r_t" where water absorption is meant.
- (9.61), p. 281: read as c0 sqrt(gamma_t / (rho c0^2)) (-i w T)^(n/2); consistent with (9.65)
  (delta_t = tan(n pi / 4) follows from the factor (-i)^(n/2)). n for Fig. 9.6 not printed on the
  pages read.
- Rock properties (p. 169) read from the extracted text only.
- Values read off figures (Fig. 10.6, 10.10, 11.6, 15.3, 15.8, 15.11) are approximate and marked "read".
- Not read: Appendices I (causality), J (bistatic/far field), K (perturbation T-matrix), L.1-L.2,
  L.4 beyond the pages listed; Table 13.1 (SAX99 roughness) needed by WP2; Ch. 7's (7.36) needed by WP7.
