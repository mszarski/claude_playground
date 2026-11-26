# Sea-surface forward scatter: Dahl (1996) and Dahl (2004), extracted for the simulator

Two papers the owner supplied. Both are all rights reserved: formulas are re-typed here, constants carry
their page, equation, table or figure, the text is in our own words, and the PDFs are not in the repository.
Status words are those of `docs/REFERENCES.md`: **confirmed** = read on the page image of the paper;
**read from figure** = a value read off a printed curve (about +-1 dB on the level plots, +-0.03 on the
coherence plots, never a printed number); **ours** = our derivation, assumption or computation.

- [D96] P. H. Dahl, "On the spatial coherence and angular spreading of sound forward scattered from the sea
  surface: Measurements and interpretive model", J. Acoust. Soc. Am. 100 (2), Pt. 1, 748-758 (August 1996).
  11 pages.
- [D04] P. H. Dahl, "Forward scattering from the sea surface and the van Cittert-Zernike theorem",
  J. Acoust. Soc. Am. 115 (2), 589-599 (February 2004). 11 pages. DOI 10.1121/1.1635417.

**Correction to `docs/REFERENCES.md`:** the "wanted" row (Sec. 10) and the Sec. 3 row give the 1996 paper as
"J. Acoust. Soc. Am. 100, 2742 (1996)". The paper read here is 100 (2), Pt. 1, **748-758**. Its title and
subject are the ones the row wants.

This file continues `docs/references/surface_papers_mcdaniel_dahl.md`, which covers backscatter (McDaniel 1993;
Dahl and Plant 1997; Dahl et al. 1997). That file found no forward-scatter, coherent-loss or time-spread data in
its three papers (its Sec. 4 rows "Coherent fraction ..." and "Doppler spread ..."). This file answers part of
those rows. It also finds one place where that file needs a note: the bubble attenuation of a forward bounce in
[D96] eq. (17) is twice, in dB, what [D97] Sec. VI and our code use for the same beta_I (Sec. 2.1 here).

Numbers in Secs. 3-4 come from a scratch script outside the repository,
`/tmp/claude-0/-home-user-claude-playground/32e65dcd-82ae-5f75-a363-7372d2e7f847/scratchpad/jr/dahl9604.py`.
It imports a copy of this worktree's `gpusonar`, with the built module linked in, and changes nothing in the
worktree. It contains three things. (a) The closed forms of both papers. (b) A numerical geometric-optics
model of [D96]: eqs. (4), (7)-(10) on a 1 m x 0.25-0.5 m grid of the surface, exact bistatic geometry, the
vCZT phase of [D04] eq. (1) for displaced receivers, and the delay histogram of the same intensities. In the
tables this model is called "GO". (c) Our `TimeSpreadProvider` (`angle_spread: true`,
`decay_time: facet`, `tail_quantile: 0.995`, 40 taps, 60 seeds) run on the papers' geometries with an
`ImageSourceProvider`. From its taps the script computes coherence along x, y and z as
sum |a|^2 exp(i k d u_arrival . axis) / sum |a|^2, the estimator of
`test_array_coherence_of_the_tail_drops_as_the_angular_spread_predicts`.

## 1. Scope

Risky first. Neither paper prints a Doppler spread, a measured coherent fraction or a scintillation index.
Time spread appears only in two figures of measured intensity against time ([D96] Fig. 2; [D04] Fig. 3c).
The decay numbers in Sec. 3.3 are read from those figures. The [D04] model curves (Figs. 5, 6) rest on a
small-slope model whose formulas are in another paper (Dahl 1999, J. Acoust. Soc. Am. 105, 2155), which was
not supplied.

| Paper | Pages | How |
|---|---|---|
| [D96] | 748-758, all 11 (incl. appendices and references) | every page image; eqs. (4)-(10), (15)-(18) and Fig. 2 re-rendered at 220-400 dpi |
| [D04] | 589-599, all 11 (incl. appendix and references) | every page image; eqs. (1)-(8) and Figs. 3c, 5, 6 re-rendered at 220-400 dpi |

## 2. What the papers give

### 2.1 Conventions

| Item | [D96] | [D04] | Our code |
|---|---|---|---|
| Angle | grazing: theta_i, theta_s; nominal specular grazing theta_g; bistatic (azimuth) angle phi_b (Fig. 3, p. 750) | grazing theta_g at the nominal specular point (Fig. 1, p. 590) | grazing |
| Rayleigh parameter | chi = k H (sin theta_i + sin theta_s) (eq. 6), H the rms wave height | kH >> 1 needed for an incoherent source (p. 589) | Gamma = 2 k h sin g: **the same** as chi for theta_i = theta_s |
| Cross section | sigma_kr, sigma_geo: bistatic cross section per unit area (eqs. 4-5); intensity per patch <I> = B^2 A sigma / (R_i^2 R_s^2) (eq. 4) | sigma(xi, eta) / r_inc^2 used as the source intensity distribution (p. 590) | facet model, no explicit sigma |
| Slope | s_L^2: mean-square large-scale slope, **both axes together** (eq. 9 is isotropic, exponent tan^2 / s_L^2) | S_L the same (p. 591, 599) | `mean_square_slope`: both axes together (Cox-Munk clean, 0.003 + 0.00512 U) |
| Coherence length | Gaussian fit Gamma(kd) = exp(-(kd sigma_theta_h)^2 / 2) (eq. 14); kd* where Gamma = e^(-1/2); sigma_theta_h = 1/(kd*) | d* where Gamma = e^(-1/2) (p. 591) | test: exp(-d^2 / 2 sigma_d^2), sigma_d = 1/(k sigma_S): **the same definition** (d* = sigma_d) |
| Bubbles | beta_I "the depth-integrated extinction cross section per unit volume", dimensionless (p. 754) | beta_1, "area-normalized, depth-integrated extinction cross section" (p. 594) | `bubble_layer_optical_depth`: "nepers of intensity lost in one vertical crossing" |
| Time dependence | not stated (complex envelopes, magnitudes only) | phase of Gamma_V printed as exp(+i k dz sin theta_g) (eq. 5) | - |

**Bubble attenuation, a factor 2 (confirmed on the page; the consistency check is ours).** [D96] eq. (17)
attenuates the specular (facet) term by alpha_b = exp(-2 (beta_I / sin theta_i + beta_I / sin theta_s)). For a
symmetric bounce that is exp(-4 beta_I / sin g), a loss of 17.37 beta_I / sin g dB. The same paper's bubble
scattering term, eq. (18), can be rebuilt path by path: the direct path plus the doubly reflected path give
(1 - e^(-2(b_i + b_s))) / (b_i + b_s), and the two singly reflected paths give the second fraction, where
b = beta_I / sin theta. That reconstruction holds only if one crossing of the layer costs exp(-beta_I / sin theta).
This is also what [D97] eq. (17) and Sec. VI (two-way 20 log10(e) beta_I / sin theta dB) and our
`bubble_loss_db` / `bubble_layer_optical_depth` use: (10/ln 10) x 2 beta_I / sin g = 8.69 beta_I / sin g dB per
bounce. So eq. (17) is either a misprint or a different convention, and the paper does not say which. With
beta_I = 0.05 and g = 12 degrees: eq. (17) gives 4.18 dB per bounce, our convention 2.09 dB.

### 2.2 [D96] experiment (Sec. I, Table I, pp. 748-752; Appendix A) - confirmed

FLIP, off southern California (32 N, 125 W), January 1992. Deep water (about 4000 m, Fig. 1). The water column
was mixed to about 30 m; c = 1500 m/s was used. Sources were ITC-1032 omnidirectional transducers hung from a
spar buoy at 57 or 147 m. The receiver was a line array on FLIP's hull at 66 m, oriented across the path:
8 elements over the middle half of a 1.3 m array, separations 1.59-57.15 cm, baffled (+3 dB directivity). The
signals were CW pulses of 4-12 ms at 20, 30 or 40 kHz, 30-50 pulses 1 s apart, heterodyned to 5 kHz, sampled at
20 kHz and band-passed 4-6 kHz. The direct path was removed by time gating. Winds were 0-8 m/s with a large swell;
H about 0.5 m ("characterized by H ~ 0.5 m", p. 750), so chi = O(10).

Table I (p. 752), all printed:

| Set | f (kHz) | SD (m) | Range (m) | theta_g (deg) | U (m/s) | Wind dir (deg) | Swell dir (deg) | Buoy bearing (deg) | s_L | s_LA | kd*/(2 pi) | sigma_theta_h (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20, 30, 40 | 147 | 1000 | 12 | 7.6 | 285 | 075 | 120 | 0.149 | 0.155 | 5.0 | 1.8 |
| 2 | 30 | 57 | 1000 | 7 | 5.3 | 000 | 330 | 075 | 0.137 | 0.120 | 14.5 | 0.6 |
| 3 | 30, 40 | 147 | 576 | 20 | 3.2 | 330 | 150 | 135 | 0.119 | 0.110 | 4.9 | 1.9 |
| 4 | 40 | 147 | 724 | 16 | 1.5 | 025 | 240 | 150 | 0.085 | 0.040 | 17.7 | 0.5 |
| 5A | 30, 40 | 147 | 714 | 17 | 5.0 | 190 | 070 | 000 | 0.135 | 0.114 | 5.0 | 1.8 |
| 5B | 30, 40 | 57 | 714 | 10 | 5.0 | 190 | 070 | 000 | 0.135 | 0.114 | 12.5 | 0.7 |

RD = 66 m throughout. s_L is the value from eq. (A7) at the measured wind. s_LA is the value adjusted to fit
the coherence, a change equivalent to <= 2 m/s of wind. Wire-gauge slopes: set 1, 0.160; set 3, 0.130 (p. 752).

- Intensity of the forward-scattered arrival, set 1 run 1: "approximately exponentially distributed" (p. 749),
  i.e. a scintillation index of about 1. "Certain surface conditions may produce a focusing effect that leads to
  stronger fluctuations" (refs. 8, 9). **No number is given**; the paper says this was under study.
- Calibrated pair (p. 751): with all intensity incoherent (chi >> 1), the Kirchhoff model's energy conservation
  means 0 dB forward-scatter loss after absorption. 0-dB losses were measured "under sufficiently mild conditions
  to preclude bubble formation" (ref. 23, APL-UW TR 9307, not read). Losses beyond absorption and spreading are
  attributed to bubbles.
- Refraction: equivalent sound-speed gradients O(1e-2) s^-1. Ray-traced theta_g differs from the geometric value
  by <= 0.5 degrees (p. 754).

### 2.3 [D96] model (Sec. II, pp. 750-751) - confirmed

| Eq. | Formula | Notes |
|---|---|---|
| (1)-(3) | C_ij(tau) = <E_i(t) E_j*(t + tau)> / (<abs E_i> <abs E_j>); abs Gamma_ij = abs C_ij(0); Var(abs Gamma) = (1 - abs Gamma^2)^2 / (2N), N pings | estimator |
| (4) | <I> = B^2 A sigma_kr / (abs R_i^2 abs R_s^2) | intensity from patch A |
| (5) | sigma_kr = (k^2 F^2 / 2 pi) int_0^inf rho J0(2 kappa rho) exp(-chi^2 [1 - C_eta(rho)]) d rho | Kirchhoff; C_eta the normalised height covariance; the text says kappa "equals k sqrt(alpha^2 + beta^2)" and calls it the transverse component of (k_i - k_s), which is twice that; with J0(2 kappa rho) the formula is consistent |
| (6) | chi = k H (sin theta_i + sin theta_s) | |
| (7) | 2 alpha = cos theta_i - cos theta_s cos phi_b; 2 beta = -cos theta_s sin phi_b; 2 gamma = -(sin theta_i + sin theta_s) | 2(alpha, beta, gamma) = (k_i - k_s)/k |
| (8) | F = (alpha^2 + beta^2 + gamma^2) / gamma | |
| (9) | sigma_geo = sec^4(gamma_bar) / (4 pi s_L^2) exp(-tan^2(gamma_bar) / s_L^2) | geometric-optics limit; independent of frequency |
| (10) | tan^2(gamma_bar) = (alpha^2 + beta^2) / gamma^2 | the facet tilt that reflects specularly toward the receiver |
| (11) | u = (sin theta_a cos theta_e, cos theta_a cos theta_e, sin theta_e) | array coordinates; sin theta_h = sin theta_a cos theta_e |
| (12) | Gamma(kd) = int_{-pi/2}^{pi/2} cos theta_h P_h(theta_h) exp(i kd sin theta_h) d theta_h | narrow P_h: a Fourier pair |
| (13) | sum_n sigma_geo_n A / (R_i_n^2 R_s_n^2) = A_0 / (R_i_SP + R_s_SP)^2 | within 0.25 dB numerically: the Kirchhoff sum reproduces a flat mirror (energy conservation) |
| (14) | Gamma(kd) = exp[-(kd sigma_theta_h)^2 / 2] | a one-parameter summary, good for kd/(2 pi) <= 7 |
| (15) | **sigma_theta_h = q s_L sin theta_g / (1 + RD/SD)** | **q = 1.348**, a least-squares fit to the six (sigma_theta_h, s_LA) of Table I |
| (16) | sigma' = sigma_geo alpha_b + sigma_b | bubbles |
| (17) | alpha_b = exp[-2 (beta_I / sin theta_i + beta_I / sin theta_s)] | see the factor-2 note in 2.1 |
| (18) | sigma_b = (beta_I / 4 pi)(delta_r / delta) [ (1 - e^(-2(b_i + b_s))) / (b_i + b_s) + (e^(-2 b_i) - e^(-2 b_s)) / (b_s - b_i) ], b = beta_I / sin theta | modified Crowther (APL-UW, mid-1980s); delta_r = 0.0136; delta "approximately 0.0792 at 30 kHz" |

- Grid: 2 m x 2 m patches, half of the surface (by symmetry). P_h is summed in bins of theta_h and transformed by
  FFT. Absorption and refraction are ignored in the coherence model (p. 751).
- Expanding eq. (10) about the specular point gives tan gamma_bar about 0.5 phi_b / tan theta_g: the width of
  eq. (9) goes as s_L tan theta_g (p. 754). This is the origin of eq. (15).
- Bubbles (Fig. 11, set 1, s_LA = 0.155, beta_I = 0.01, 0.03, 0.05, 0.10): the effect on coherence is
  "marginal". It is in the right direction for beta_I = 0.05 to 0.10, and smaller at 750 m / 16 degrees.
  Bubble measurements put **beta_I at 0.05-0.10 for 7-10 m/s** (p. 754; frequency not stated with the range;
  the delta quoted is at 30 kHz). Langmuir plumes would need a Monte Carlo model; not done.
- Two qualitative statements (p. 754): a larger beta_I lowers kd* (out-of-plane bubble scatter). Bubbles matter
  less as sin theta_g / (1 + RD/SD) grows.

### 2.4 [D96] slope law and patch size (Appendices A-B, pp. 755-757) - confirmed

| Eq. | Formula | Notes |
|---|---|---|
| (A1) | s_L^2 = (1/g^2) int_0^{omega_c} omega^4 S(omega) d omega | from the height frequency spectrum |
| (A2) | s_L^2 = int_0^{K_c} K^2 [K F(K)] dK | F(K) directionally averaged, K F(K) dK = S(omega) d omega |
| (A3) | 4 k^2 h_s^2 sin^2 theta_g << 1 | small-scale variance criterion (McDaniel and Gorman) |
| (A4) | h_s^2 = int_{K_c}^inf K F(K) dK = 0.0025 / K_c^2 | Plant's equilibrium range F(K) about 0.005 K^-4 |
| (A5) | **K_c > 0.1 k sin theta_g** | lower bound; numerical studies suggest K_c about 0.25 k |
| (A6) | 2 / (k R sin^3 theta_g) < 1 | Kirchhoff radius-of-curvature condition |
| (A7) | **s_L^2 = 0.0046 log_e(2.1 U^2)**, U at 10 m in m/s | Phillips; "a fit to the Cox and Munk oil-slick data" with a K^-4 equilibrium range; 0.0046 is "a more precise value of the 0.005" in (A4) |

- The cut-off used for the radar-altimeter slopes is K_c = 2 pi / 0.3 rad/m, i.e. omega_c / 2 pi = 2.28 Hz,
  chosen to match Cox and Munk's slick surface (wavelengths shorter than about 30 cm removed). It satisfies (A5)
  for 20-40 kHz. The fitted frequency exponent of the spectrum's tail was p = -4.5 +- 0.7 (p. 756).
- Fig. A2 (read from figure): the altimeter's s_L^2 scatter between about 0.002 and 0.03 at 2-8 m/s. Eq. (A7)
  runs through the middle. Swell alone gives a background s_L of about 0.03 (p. 753). Slopes at about 1 m/s vary
  by a factor of 10 (p. 753).
- Set 4 (1.5 m/s) is the outlier: s_L and s_LA differ by a factor of 2. Slicks, internal waves and swell
  modulation are offered as reasons (p. 753). Under eq. (A7) the slope fluctuations go as 1/(U sqrt(log U)), so
  coherence is most sensitive to wind at low wind.
- Patch (App. B): chi^2 rho^2 / l_H^2 = 0.69 gives the 3 dB point of the integrand of eq. (5) (B1). With
  s_L^2 = 4 H^2 / l_H^2 (B2), H about 0.5 m and 5 m/s, l_H is about 7 m. At 30 kHz and 15 degrees the field
  correlation l_A is about 0.4 m, so sqrt(A) = 2 m.
- No directional effect of wind or swell on coherence was seen (Table I), except that set 5A at kd/(2 pi) >= 7 is
  "puzzling" (higher than the model).

### 2.5 [D04] the van Cittert-Zernike formulation (Sec. II, pp. 590-592) - confirmed

Source S at depth SD, central receiver R_0 at depth RD and range R. The surface intensity distribution
I(xi, eta) is centred at the specular point, X = R RD / (SD + RD) from the receiver. rho is the slant range from
the specular point to R_0.

| Eq. | Formula | Notes |
|---|---|---|
| (1) | Gamma_01 = (1 / sqrt(I_0 I_1)) int I(xi, eta) exp(i k (r_0 - r_1)) / (r_0 r_1) d xi d eta | vCZT, no Fourier approximation |
| (2) | I_i = int I(xi, eta) d xi d eta / r_i^2 | |
| (3) | r_1 about rho - X eta / (2 rho) + (y_1^2 - 2 xi y_1 + xi^2 + eta^2) / (2 rho) | binomial (Fresnel-region) approximation |
| (4) | Gamma_H(dy) = exp(i Omega) exp[-(k L_T dy / (2 rho))^2], Omega = -k dy^2 / (2 rho) | for I = exp[-(xi/L_T)^2 - (eta/L_L)^2]; Omega negligible if k dy^2 << 2 rho; **d* = 2 rho / (k L_T)** |
| (5) | Gamma_V(dz) = exp(i Omega + i k dz sin theta_g) exp[-(k L_L sin theta_g cos theta_g dz / (2 rho))^2], Omega = -k (dz cos theta_g)^2 / (2 rho) | vertical, with an intrinsic phase k dz sin theta_g |
| (6) | Gamma_HL(dx) = exp(i Omega - i k dx cos theta_g) exp[-(k L_L sin^2 theta_g dx / (2 rho))^2], Omega = -k (dx sin theta_g)^2 / (2 rho) | horizontal along the path |
| (7) | **kd*_H ~ (1 + RD/SD) / (S_L sin theta_g)** | from the e^(-1/2) width phi_b about 2 S_L tan theta_g, and theta_H / phi_b = cos theta_g / (1 + RD/SD); [D96] eq. (15) with an O(1) factor |
| (8) | **kd*_V ~ (1 + RD/SD) / (S_L cos theta_g)** | taking L_T / L_L ~ sin^2 theta_g (McDaniel 1991) |
| (9) | **kd*_HL ~ (1 + RD/SD) / (S_L sin theta_g)** | = kd*_H |

- Validity (printed): the bandwidth condition is ds / c << 1 / df. For these geometries ds/c <= 0.1 ms, so the
  pulse length sets the bandwidth (p. 590). The observation region is delta / rho < 0.1 (p. 590). The nonlinear
  phase terms are negligible only when **cos^2 theta_g / (sin^4 theta_g (kd*_H)^2) << 1**, which "often requires
  theta_g >= 10 deg" (p. 592).
- Coherence volume about (2 d*_V)(2 d*_H)(2 d*_HL). The vCZT's "d* grows linearly with range" does not hold for
  surface forward scatter, because the source region changes with the geometry. The scaling parameter for
  horizontal coherence is g* = (1 + RD/SD) / sin theta_g (p. 592).

### 2.6 [D04] measurements and comparisons (Secs. III-IV, Appendix) - confirmed unless marked

**Horizontal, 30 kHz (Sec. III, Fig. 2).** Two data sets.
- Pelagic: [D96] set 5A/5B, 714 m, U10 = 5 m/s (10-min average), H 0.5 m, SD 147/57 m, RD 66 m,
  theta_g 17/10 degrees, g* = 5.0/12.4.
- Littoral: 72 m range, U10 = 4 m/s, H 0.07 m, SD 17.2/7.7 m, RD 9.5 m (array from a ship), theta_g 20/14
  degrees, g* = 4.5/9.2.

The grazing criterion is 0.1-0.2 for these data. Plotted against kd/(2 pi), the 72 m data roughly match the
714 m data, a factor of 10 greater in range. Plotted against kd/(2 pi)/g*, all four collapse to one cluster.

**Vertical, ASIAEX East China Sea (Sec. IV.A, pp. 593-594).**
- Site and dates: R/V Melville, 29.65 N, 126.82 E, water 105 m, May-June 2001.
- Source: ITC 1007 at 25 m. Range about 500 m (measured acoustically). Fig. 4 is computed for 467 m, the text
  of Sec. IV.C says 497 m.
- Receivers: two VLAs of 4 ITC 1042 elements each, top-to-bottom separations 13, 30 and 60 cm, centred at
  26 m and 52 m. 50 kHz sampling.
- Signals: 3 ms CW at 8 and 20 kHz (from a family of seven waveforms at 2-20 kHz cycled every 5 s). Sets 6-8,
  20 receptions each, 60 receptions averaged, about 6000 s in all, 0630-0800 UTC 31 May 2001.
- Sea: U = **7 +- 0.5 m/s**, **H = 0.3 +- 0.1 m**, wind and waves from the north, path north-south. TRIAXYS
  buoy 0.5 km away.
- Estimator (eq. 10): Gamma_pq = <p_i q_i*> / sqrt(<p p*><q q*>). Standard deviation of abs Gamma:
  (1 - abs Gamma^2) / sqrt(n). Real and imaginary parts: that times sqrt 2 (bootstrap).
- Refraction (sound speed about 1521-1528 m/s over the depth, Fig. 3a, read from figure): specular arrival
  angles 6.1 and 10.9 degrees with refraction, against 6.1 and 9.4 iso-speed. Applied as a phase,
  Gamma'_V = Gamma_V exp(-i k dz (sin theta_V0 - sin theta_V)).
- Bubbles: beta_1 = **0.009 at 20 kHz and 0.004 at 8 kHz** (from "Eq. (5) of Ref. 18"; see Sec. 6) were
  included and have "little or no measurable impact" on coherence at 7 m/s.

The model is the full vCZT, eq. (1), with sigma from the small-slope approximation (Dahl 1999 eqs. 1, 5, 6, not
supplied) and the measured 2-D height autocovariance. The printed numbers, Fig. 5, give the e^(-1/2) points of
the model curves in wavelengths: upper VLA 3.4 (20 kHz), 3.9 (8 kHz); lower VLA 4.0 (20 kHz), 4.5 (8 kHz). In
wavelengths, coherence at 20 kHz is about 10 % shorter than at 8 kHz. The model compares "reasonably well" with
the data.

Fig. 5 data, abs Gamma_V at 0.13 / 0.30 / 0.43 / 0.60 / 0.89 / 1.02 m (read from figure):

| VLA | 8 kHz | 20 kHz |
|---|---|---|
| upper (26 m) | 0.98 / 0.86 / 0.75 / 0.76 / 0.62 / 0.57 | 0.85 / 0.57 / 0.46 / 0.33 / 0.30 / 0.24 |
| lower (52 m) | 1.00 / 0.95 / 0.91 / 0.78 / 0.63 / 0.59 | 0.93 / 0.69 / 0.59 / 0.32 / 0.22 / 0.28 |

**Coherence volume (Sec. IV.C, Fig. 6, 20 kHz).** At the 52 m receiver: 2d*_V about 0.5 m, 2d*_H about 2.5 m,
2d*_HL about 3.5 m (printed). Rounded for 25-50 m receivers at 20 kHz and 0.5 km: a layer 0.5 m thick by 3 m in
each horizontal direction, "or 4-5 m^3". At 8 kHz each dimension is 2.5 times larger. The model curves in Fig. 6
cross e^(-1/2) at about the following separations (read from figure):

| Receiver | d*_V | d*_H | d*_HL |
|---|---|---|---|
| 26 m | 0.27 m | 1.33 m | > 2 m |
| 52 m | 0.29 m | 1.2 m | 1.75 m |

The grazing criterion is 0.11 at 52 m and 0.75 at 26 m, so the scaling relations are less reliable at 26 m.
Fig. 6 triangles show the scaling relations with S_L about 0.15 (read from figure): d*_V about 0.17 m (26 m) and
0.25 m (52 m); d*_H = d*_HL about 1.5 m.

**Appendix (pp. 597-599).**
- C~(u, v) = int d phi int dK K F(K, phi) cos(K [u cos phi + v sin phi]) (A1). Buoy data (low-pass) are used
  for K <= 1.1 rad/m, and Plant's (2002) D-spectrum at 7 m/s, fully developed, for the high-pass part up to
  K = 500 rad/m.
- Slopes from the autocovariance: S_u^2 = 0.0157 up/down wind, S_v^2 = 0.0132 crosswind, total **0.0289**,
  ratio 0.84.
- "Length scale support by wave numbers up to about k/4 is essentially all that is needed" for forward-scatter
  coherence at 10-100 kHz. At 20 kHz, k/4 is about 21 rad/m, and a low-pass C~ with K <= 21 gives "essentially
  the same results". The slope of that low-pass surface is S_L = 0.152. "A simple formula" of Dahl (2001,
  IEEE JOE 26, 141; not supplied) gives S_L = 0.146 at 7 m/s and 20 kHz. Ours: [D96] eq. (A7) gives 0.146 at
  7 m/s too.
- A directionally averaged C~ gives coherence curves only slightly above the directional ones: wave
  directivity is not resolvable in these data.

**Not in [D04]:** time spread (except Fig. 3c), coherent fraction, scintillation, Doppler.

### 2.7 Time-spread evidence (figures only) - read from figure

- [D96] Fig. 2 (p. 749): ensemble average of set 1 run 1, 4 ms pulse, after leading-edge alignment. Set 1:
  SD 147, RD 66, 1000 m, 12 degrees, 7.6 m/s, frequency not stated (20, 30 or 40 kHz). The pulse ends at about
  5.6 ms. Level in dB re the plateau at 6.0 / 6.5 / 7 / 7.5 / 8 / 9 / 10 / 11 / 12 / 13 / 14 / 15 / 16 / 17 /
  18 / 19 / 20 ms: -1.5 / -4.5 / -7 / -8 / -9 / -10.2 / -11.5 / -12.5 / -13.8 / -14.6 / -15.5 / -17 / -17.8 /
  -18.5 / -18.3 / -19.5 / -21.2. The pre-arrival floor is about -33 dB. The rise to within 1 dB of the plateau
  takes about 1.2 ms. Shape: a fast drop of about 6 dB in the first ms after the pulse, then about 1 dB/ms.
- [D04] Fig. 3c (p. 593): one element, average of 20 pings, set 7, 20 kHz, 3 ms pulse. Lower VLA (52 m): the
  surface arrival peaks at about -1.5 dB at 6.5 ms, then -5 / -7 / -11 / -13.5 / -15 / -18 / -21 dB at
  7.5 / 8 / 9 / 10 / 11 / 12 / 13 ms. The bottom arrival starts at 13.7 ms. Upper VLA (26 m): the surface
  arrival overlaps the direct path, so its tail readings (-5 at 6 ms down to -31 at 15 ms) are less reliable.

## 3. Against our simulator

Wrong or doubtful first.

1. **Our time-spreading slope is too steep for forward scatter.** `boundary_parameters` takes the clean-surface
   Cox-Munk total, 0.003 + 0.00512 U. [D96] and [D04] use the large-scale slope: waves longer than about 30 cm,
   or K <= k/4, from eq. (A7) or a low-pass spectrum. That slope reproduces the measured horizontal coherence
   (q = 1.348). Ours is 1.4-2.2 times larger in mean-square slope at 1.5-10 m/s, which makes the angular spread
   1.2-1.5 times too wide. Our coherence lengths across the path come out **25-45 % short** at five of the six
   [D96] sets and **3 times short** at 1.5 m/s. Our facet decay time is 1.6-1.9 times what the same model gives
   with Dahl's slopes.
2. **The geometry of our facet model is right; the constant is right within 5 %.** In closed form our across-path
   spread is sqrt(2 mss) sin g / (1 + RD/SD): [D96] eq. (15) with q = sqrt 2 = 1.414, against the fitted 1.348.
3. **Tail length.** Measured decays fall faster than our facet rule by 4-10 dB at 5-14 ms after the pulse
   ([D96] Fig. 2 and [D04] Fig. 3c), and faster than the rule of thumb by 8-16 dB. The exact-geometry
   Kirchhoff model of [D96], with Dahl's slopes, is within 1-4 dB. So the **facet rule is supported over the
   rule of thumb**, and both of our rules are too long. Two causes: the slope, and our second-order geometry,
   which overstates the mean delay by a factor 1.4-1.6 at 9-12 degrees.
4. **Vertical coherence at low grazing is about 2 times too short in our model, and in the geometric-optics
   model.** At ASIAEX, the measured abs Gamma_V at 20 kHz and 0.30 m is 0.57-0.69. Ours gives 0.03-0.2, and GO
   with S_L = 0.15 gives 0.44-0.48. Dahl's small-slope model (not in hand) fits.
5. **Bubbles.** For beta_I, our `apl_1994` matches [D96]'s 0.05-0.10 at 7-10 m/s (30 kHz). It is 4.4 times
   [D04]'s ECS values at 7 m/s. The factor 2 in [D96] eq. (17) is unresolved (2.1).

### 3.1 Item by item

| Item | Papers | Our code (file, key, decision) | Agreement (how checked) |
|---|---|---|---|
| Rayleigh parameter, coherent split | chi = kH(sin theta_i + sin theta_s) ([D96] eq. 6); "coherent intensity negligible" at >= 20 kHz, H 0.5 m | `TimeSpreadProvider` exp(-Gamma^2); `acoustics.surface_coherent_reflection` (D-38) | **identical** definition. Ours at the papers' conditions: coherent fraction < 1e-9 everywhere except ASIAEX 8 kHz at the upper VLA, 0.012 (H 0.3 m) or 0.034 (PM height); [D96] set 4 with PM height 0.012 m gives 0.48, but the measured swell H about 0.5 m gives 0 (computed). **No measured coherent fraction is in either paper**: PHYSICS_NOTES 21's "too coherent at low grazing" is neither supported nor contradicted |
| Sea height | H about 0.5 m at 0-8 m/s (swell, [D96]); 0.3 m at 7 m/s ([D04] ECS); 0.07 m at 4 m/s (littoral) | `acoustics.rms_wave_height` (PM: 0.31, 0.26, 0.086 m at 7.6, 7, 4 m/s) | swell makes H independent of wind (3.7 times PM at 5 m/s in [D96]); [D04] ECS 1.15 times PM. Supports PHYSICS_NOTES 21's "1.38 times PM" reading and the separate `rms_height` key (option 6) |
| Kirchhoff facet cross section | [D96] eq. (9) with s_L^2 both axes | `facet_backscatter`, `reverb_specular` lobe, the time-spread facets | **identical** form (the companion file showed eq. (7) of McDaniel equals ours; eq. (9) is its bistatic form) |
| Slope law | eq. (A7) 0.0046 ln(2.1 U^2), fit to Cox-Munk **slick**; cut-off K_c = 2 pi/0.3 rad/m; K <= k/4 ([D04]) | `mean_square_slope`: Cox-Munk **clean** 0.003 + 0.00512 U ("unconfirmed") | **differs**: ours/A7 = 1.50, 1.37, 1.57, 1.82, 1.90, 2.20 at 1.5, 3.2, 5.0, 7.0, 7.6, 10 m/s (computed). [D04] directional total 0.0289 at 7 m/s (K up to 500 rad/m, buoy + Plant), ours 0.0388 |
| Across-path angular spread | eq. (15), q = 1.348 | angle_spread: sqrt(2 mss) sin g r1/(r1 + r2) (test `test_rms_angular_spread_matches_the_closed_forms`) | form identical, q = sqrt 2 (+4.9 %). Re-fit of q from the printed Table I: 1.342 (sigma column), 1.348 (kd* column) (computed): the printed q reproduces |
| Coherence lengths | [D04] eqs. (7)-(9); d*_HL = d*_H | our facet: across sigma_v sin g, along sigma_v sin g, vertical sigma_v cos g | **same scaling** as eqs. (7)-(9), including d*_HL = d*_H; the O(1) factor is sqrt 2 |
| Decay time | none printed; Figs. 2 / 3c | `decay_time: facet` default (D-85), `rule_of_thumb` (TR 0702) | Sec. 3.3 |
| Bubble attenuation of the bounce | [D96] eq. (17): exp(-2 beta (1/sin theta_i + 1/sin theta_s)) | `bubble_loss_db`, `bubble_layer_optical_depth`: exp(-2 beta / sin g) per bounce | factor 2 in the exponent (2.1); our convention agrees with [D96] eq. (18) and [D97] |
| beta_I | 0.05-0.10 at 7-10 m/s ([D96], 30 kHz implied); 0.009 at 20 kHz, 0.004 at 8 kHz, 7 m/s ([D04], ECS) | `apl_1994` 0.063 (7.6 m/s, 30 kHz), 0.056 (7), 0.097 (10); 0.039 (7 m/s, 20 kHz), 0.018 (8 kHz). `hall_novarini` 0.118, 0.090, 0.43; 0.051, 0.014 | FLIP: `apl_1994` **within the range**, `hall_novarini` 4 times high at 10 m/s. ECS: `apl_1994` **4.4-4.5 times** [D04] (6.4 dB of beta). A site factor of about 5 at the same wind (cf. McDaniel's coastal/open 10 dB in the companion file) |
| delta at 30 kHz | 0.0792 ([D96] p. 754) | `bubble_damping_constant(30 kHz)` = 0.0792 | **identical** (computed) |
| delta_r | 0.0136 | `BUBBLE_RERADIATION_DAMPING` 0.0136 | identical |
| Bistatic bubble scatter | [D96] eq. (18) | not modelled in forward scatter (only `bubble_backscatter`, monostatic) | absent; [D96] and [D04] find its effect on coherence marginal at 7-8 m/s |
| Scintillation of the forward arrival | "approximately exponentially distributed" (set 1; SI about 1); focusing mentioned, no numbers | complex Gaussian taps (SI 1); `tap_shape` K texture opt-in | consistent with SI about 1 for this set; the focusing explanation **is not in these papers** |
| Doppler | not given | D-84 Gaussian, 2 f v sin g / c | nothing to compare |
| Refraction | [D04]: phase correction of Gamma_V; arrival angles 6.1 / 10.9 vs 6.1 / 9.4 degrees | eigenray provider (Bellhop or image) | not compared |

### 3.2 Angular spread and coherence against the papers (computed)

**[D96] Table I, horizontal coherence across the path, kd*/(2 pi)** (computed at 30 kHz; the facet and GO
results scale with k, so kd* is frequency-free):

| Set | U | Measured | [D96] eq. (15), s_LA | Ours, closed form, Cox-Munk | Ours, provider Monte Carlo, wind law | Ours, closed form with s_LA | GO, s_LA | GO, Cox-Munk | Ours (MC) vs measured |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.6 | 5.0 | 5.3 | 3.8 | 3.7 | 5.0 | 4.7 | 3.4 | -26 % |
| 2 | 5.3 | 14.5 | 17.4 | 11.5 | 11.4 | 16.6 | 13.6 | 8.6 | -21 % |
| 3 | 3.2 | 4.9 | 4.5 | 3.4 | 3.4 | 4.3 | 4.2 | 3.3 | -31 % |
| 4 | 1.5 | 17.7 | 15.5 | 5.6 | 5.8 | 14.4 | 14.4 | 5.5 | **-67 %** |
| 5A | 5.0 | 5.0 | 5.1 | 3.4 | 3.5 | 5.0 | 4.9 | 3.2 | -30 % |
| 5B | 5.0 | 12.5 | 12.9 | 8.5 | 7.6 | 12.5 | 11.0 | 6.9 | -39 % |

As rms angles: measured sigma_theta_h 1.8 / 0.6 / 1.9 / 0.5 / 1.8 / 0.7 degrees; ours (provider) 2.44 / 0.80 /
2.69 / 1.56 / 2.65 / 1.17 degrees. Reading the table:
- Our provider agrees with our closed form within 3 % at sets 1-5A. At 5B it is 11 % below (the turn toward the
  horizontal is clamped at 0.9 of the elevation in `_turn`).
- Our closed form with Dahl's slope matches the measurements within 0-12 % at five sets and 19 % at set 4.
- The whole difference from the measurements is the slope law, not the geometry.

**ASIAEX ([D04]), SD 25 m, range 467 m, 7 m/s; d* in m (and wavelengths), c = 1525 m/s** (computed):

| RD, f | [D04] model (printed / read) | GO S_L 0.15 | GO eq. (A7) | GO Cox-Munk | Ours, wind law (Cox-Munk) | Ours, mss 0.0225 | [D04] eqs. (7)/(8), S_L 0.15 |
|---|---|---|---|---|---|---|---|
| 26 m, 20 kHz: V | 3.4 lambda (0.26 m) | 0.146 (1.9) | 0.149 (2.0) | 0.114 (1.5) | 0.116 (1.5) | 0.140 (1.8) | 0.166 |
| H | 1.33 (read) | 0.82 | 0.85 | 0.58 | 0.75 | 0.98 | 1.52 |
| HL | > 2 (read) | 1.32 | 1.35 | 1.01 | 1.02 | 1.27 | = H |
| 26 m, 8 kHz: V | 3.9 lambda (0.73 m) | 0.363 (1.9) | 0.372 | 0.281 | 0.290 (1.5) | 0.349 | 0.415 |
| 52 m, 20 kHz: V | 4.0 lambda (0.30 m); 2d*_V about 0.5 m | 0.184 (2.4) | 0.189 | 0.141 | 0.137 (1.8) | 0.176 (2.3) | 0.253 |
| H | 2d*_H about 2.5 m (1.2 read) | 0.84 | 0.87 | 0.59 | 0.84 | 1.11 | 1.53 |
| HL | 2d*_HL about 3.5 m (1.75 read) | 1.10 | 1.13 | 0.84 | 0.84 | 1.08 | = H |
| 52 m, 8 kHz: V | 4.5 lambda (0.84 m) | 0.457 (2.4) | 0.469 | 0.348 | 0.343 (1.8) | 0.439 | 0.631 |

The rows give d*_V, d*_H and d*_HL for each receiver and frequency.

abs Gamma_V at 0.13 / 0.30 / 0.43 / 0.60 / 0.89 / 1.02 m against the data of 2.6 (computed; our values carry
Monte Carlo noise of about +-0.1 beyond 0.3 m):

| | upper 20 kHz | lower 20 kHz |
|---|---|---|
| measured | 0.85 / 0.57 / 0.46 / 0.33 / 0.30 / 0.24 | 0.93 / 0.69 / 0.59 / 0.32 / 0.22 / 0.28 |
| GO, S_L 0.15 | 0.65 / 0.44 / 0.38 / 0.32 / 0.26 / 0.24 | 0.72 / 0.48 / 0.39 / 0.32 / 0.23 / 0.21 |
| ours, wind law | 0.54 / 0.20 / ... (noise) | 0.64 / 0.03 / ... (noise) |

Reading:
- In the vertical, our facet model and GO agree with each other for the same slope (1.5 against 1.5
  wavelengths, 1.8 against 1.8). Both lie about 2 times below [D04]'s model and the data at low grazing
  (6-11 degrees). GO also has a long tail beyond the e^(-1/2) point, as the data do.
- Ours falls off faster than GO beyond d*. The facet angles are Gaussian, while GO's angular distribution is
  heavy-tailed: an asymmetric footprint that runs from near the source to near the receiver ([D04] Fig. 4).
- Our across-path d*_H at 52 m happens to match GO with S_L 0.15 (0.84 m). Ours with S_L 0.15 is 32 % above GO,
  and the [D04] model gives about 1.2 m.
- So at ASIAEX grazing angles neither the slope fix nor the second-order geometry is the whole story. The
  missing factor of about 2 in vertical coherence lies between GO and Dahl's small-slope model.

### 3.3 Time spread (computed against the readings of 2.7)

Model intensity = the impulse response convolved with the rectangular pulse, normalised to its peak, aligned at
the pulse end. "Model minus read" in dB at the reading times after the pulse:

| Case (lever t1 t2/(t1+t2)) | Model | Mean excess delay (ms) | Late e-fold (ms) | Model - read, dB (first ... last) | rms |
|---|---|---|---|---|---|
| [D96] set 1, 12 deg, 4 ms (145.8 ms) | facet, Cox-Munk 0.0419 (ours as shipped) | 6.37 | 12.2 | 0.5, 2.6, 4.4 ... 8.9, 10.0 | 6.9 |
| | rule of thumb, Cox-Munk | 12.2 | 12.2 | 1.4 ... 16.1 | 11.3 |
| | facet, s_LA^2 0.024 | 3.65 | 7.0 | 0.2 ... 6.0 | 4.7 |
| | facet, eq. (A7) 0.0221 | 3.36 | 6.4 | 0.1 ... 5.1 | 4.2 |
| | rule of thumb, s_LA^2 | 7.0 | 7.0 | 1.2 ... 12.3 | 9.3 |
| | **GO ([D96] eqs. 4, 9, exact geometry), s_LA^2** | **2.65** | - | 0.2, 2.0, 3.6, 3.7 ... 1.1, 1.3, 1.9 | **2.9** |
| [D04] 52 m, 9.4 deg, 3 ms (68.1 ms) | facet, Cox-Munk 0.0388 | 2.71 | 5.3 | 1.7 ... 10.3 | 6.5 |
| | rule of thumb, Cox-Munk | 5.29 | 5.3 | 4.2 ... 15.6 | 10.7 |
| | facet, S_L^2 0.0225 | 1.57 | 3.1 | 0.8 ... 6.1 | 3.8 |
| | **GO, S_L^2 0.0225** | **1.00** | - | 0.1, 0.3, 0.9, 0.3, -1.2, -1.4, -1.7 | **1.0** |
| [D04] 26 m, 6.2 deg, 3 ms (77.0 ms) | facet, Cox-Munk | 3.03 | 6.0 | 1.7 ... 17.5 | 13.6 |
| | facet, S_L^2 | 1.75 | 3.5 | 1.0 ... 12.0 | 10.7 |
| | GO, S_L^2 | 0.78 | - | grid-limited beyond 10 ms (the integration window of +-55 m across ends near 8 ms of delay): use the first three points only, -0.4 / 1.9 / 3.2 | - |

Reading, for STATUS item 27 ("the decay-time rule is not settled by any data set"):
- Two measured decays at stated geometry and sea now exist (from figures, +-1 dB). Both reject the rule of thumb
  as shipped, by 8-16 dB at 5-14 ms. Both say our default facet tail is too long, by 4-10 dB.
- Our tail becomes right only with (a) the large-scale slope, giving a facet mean 1.6-1.9 times shorter, and
  (b) the exact Kirchhoff geometry, giving a further 1.4-1.6 times. GO with Dahl's own slope fits 52 m within
  1.7 dB, and [D96] set 1 within 4 dB (the data still fall slightly faster).
- Caveats (ours): one figure each; [D96] Fig. 2's frequency is not stated, which does not matter for GO; the
  leading-edge alignment and the omitted bubble term are not modelled; the 26 m case is contaminated by the
  direct path.

Interpretation (ours): TR 0702's ASIAEX example ran at "3 m/s, which fits the tail better" than the quoted
4 m/s (`configs/scenarios/sst_asiaex.yaml`). Cox-Munk at 3.0-3.9 m/s equals [D96]'s large-scale slope at 7 m/s
(0.018-0.023), the wind [D04] measured. Shortening the slope is what made that fit work.

## 4. What this closes or changes in open issues

Unanswered first: **Doppler spread against grazing angle** is not in either paper. **A measured coherent
fraction at low grazing** is not in either paper (both work at chi >> 1). **The focusing explanation of
scintillation above 1** is only named in [D96] p. 749, with refs. 8-9; no numbers.

### docs/REFERENCES.md Sec. 10 "wanted" rows

| Row | Result |
|---|---|
| Coherent fraction, delay spread, angle spread and scintillation of surface forward scatter | **Angle spread: answered** ([D96] Table I, eq. 15, q = 1.348; [D04] eqs. 7-9, ASIAEX vertical data). **Delay spread: partly**, from two figures (Sec. 3.3), not a printed law; Dahl 2001 (IEEE JOE 26, 141, "characteristic scales of time and angle spreading", ref. 2 of [D04]) is where the printed law would be: still wanted. **Coherent reflection loss against the Rayleigh parameter: not answered** (only "negligible" at chi = O(10)). **Scintillation: SI about 1** for one set, focusing named, no numbers |
| Doppler spread against grazing angle | not answered; wanted from Dahl 2001 or TR 9407 as quoted |
| Bubble loss per surface bounce | partly: beta_I 0.05-0.10 at 7-10 m/s (FLIP, Pacific); 0.009 / 0.004 at 20 / 8 kHz, 7 m/s (ECS); forward-bounce loss as printed in eq. (17), with the factor-2 caveat (2.1). A 0-dB forward-scatter loss was measured in mild conditions (ref. 23, TR 9307, not read) |
| Dahl papers row (Sec. 3, "wanted") | [D96] and [D04]: **confirmed** (with the citation correction above). Still wanted: Dahl 1999 (JASA 105, 2155: the small-slope bistatic model and its bubble eq. 13); Dahl 2001 (IEEE JOE 26, 141); Preisig and Deane 2004 |
| Cox-Munk total regression ("unconfirmed") | not printed here; [D96] prints Phillips' slick-surface law (A7) instead, which these papers use for acoustics |

### docs/STATUS.md and PHYSICS_NOTES

- **Item 27 (decay rule):** evidence now exists (Sec. 3.3). It rejects the rule of thumb and shows the facet
  default too long by 4-10 dB at the papers' conditions. The remedy the papers support is the large-scale slope
  plus exact geometry. Not yet a printed law (that is Dahl 2001).
- **PHYSICS_NOTES 21, "too coherent at low grazing":** not tested by these papers. Their seas (H 0.3-0.5 m)
  make the coherent part vanish, as ours does. Their measured H, and its departure from the PM law (swell),
  support the 1.38-times-PM reading and the `rms_height` key.
- **PHYSICS_NOTES 21 / D-62 "angle spreading not testable":** now testable against [D96] Table I and [D04] Fig. 5.
  The tests in WP1 would make this permanent.
- **D-107 (facet tail does not fit both windows of one clean bounce):** consistent with this file. The
  second-order facet tail is long-tailed (e-fold 2a) and our slope is large.

### Issues (named from the companion file; `.beads/issues.jsonl` was not re-read for this file)

- sonar-9le (high-frequency surface): forward scatter 8-40 kHz at 6-20 degrees: answered as above. Nothing above
  40 kHz. The papers say s_L depends "mildly" on frequency through K_c (K_c > 0.1 k sin theta_g; k/4).
- sonar-6t5 (ping-to-ping coherence): the [D96] pings 1 s apart give an exponential intensity distribution for
  set 1, i.e. independent pings at 1 s. No decorrelation time is printed.

## 5. Work packages

Ordered by value per effort. None changes a default without the owner.

**Exposure** (grep of `configs/scenarios/*.yaml`): three shipped scenarios use `time_spread`.
- `flank_sst_shallow`: wind 6 m/s from the environment, `surface: {bubble_loss_db: 0.5}`, `taps: 4`, defaults
  otherwise (facet decay, convolve, gaussian, no angle spread).
- `flank_sst_spread`: wind 6 m/s, `angle_spread: true`, `decay_time: facet`, `doppler: true`,
  `doppler_spectrum: gaussian`, `doppler_lines: 3`, `multi_bounce: convolve`, `taps: 3`, bubble 0.5 dB.
- `sst_asiaex`: wind 3 m/s, `surface: true`, bottom {0.012 m, 0.045}, `taps: 12`, 8 kHz, 3 ms.

All three take their slope from `acoustics.mean_square_slope`. The `reverb_specular` facet lobe uses the same
slopes. The Doppler width uses sqrt(mss) for a moving platform.

**WP1 (S): characterisation tests, no behaviour change.** New `tests/python/test_forward_scatter_dahl.py`:
- (a) Our closed-form across-path spread equals [D96] eq. (15) with q = sqrt 2, and with s_LA reproduces the
  Table I kd*/(2 pi) within 20 % (all six sets; computed 0-19 %).
- (b) The provider's Monte Carlo agrees with the closed form within 12 % at the six Table I geometries.
- (c) An **xfail** recording that the wind law gives kd*/(2 pi) 21-67 % below Table I.
- (d) The [D96] Fig. 2 readings (2.7) against `facet_delay_density` convolved with 4 ms at the set 1 geometry:
  the facet rule beats the rule of thumb by >= 4 dB rms (computed 6.9 vs 11.3 with Cox-Munk; 4.2 vs 8.8 with
  eq. A7).
- (e) An **xfail** for ASIAEX vertical coherence at 52 m, 20 kHz: abs Gamma_V(0.30 m) >= 0.5 (measured 0.69;
  ours about 0.03-0.2).
- (f) `bubble_damping_constant(30e3) == 0.0792` within 1e-4.

Proposed REFERENCES text: [D96] and [D04] rows confirmed, the citation corrected (100, 748-758), the wanted row
updated as in Sec. 4. Files: the test file only.

**WP2 (S-M): a named large-scale slope law, opt-in.** `time_spread.surface.slope_law: phillips_slick`
uses s_L^2 = 0.0046 ln(2.1 U^2) ([D96] eq. A7). It is clamped below at the 1.5 m/s value with a warning, since
it is negative below 0.69 m/s. Ours: [D96]'s Fig. A2 scatter goes down to about 0.002. `mean_square_slope` can
already be given explicitly today, so this is a convenience plus a test.
- Files: `sim/timespread.py` (`boundary_parameters`), `acoustics.py` (new `large_scale_mean_square_slope`).
- Tests first: values at 1.5/3.2/5/7.6 m/s (0.0071, 0.0141, 0.0182, 0.0221); Table I kd* with it within 20 %
  at sets 1-3, 5A, 5B.
- A default change needs the owner. Before -> after at 6 m/s (flank scenarios): mss 0.0337 -> 0.0199;
  facet decay time x0.59; angular spread x0.77; coherence length x1.30. At 3 m/s (`sst_asiaex`):
  0.0184 -> 0.0135, x0.74. Record numbers of the three scenarios would move. Prepare the numbers, ask, do not
  switch.
- Do not change `acoustics.mean_square_slope` itself: backscatter and other users depend on it, and the facet
  backscatter question is separate.

**WP3 (M-L): exact Kirchhoff (GO) tail and angles, opt-in `decay_time: kirchhoff_go`.** Integrate [D96]
eqs. (4), (9), (10) over the surface for each bounce: intensity sigma_geo / (R_i^2 R_s^2), delay
R_i + R_s - specular. Take the tail's shape and mean, and the joint distribution of arrival and launch directions,
from it. Tabulate in (grazing, r1/(r1+r2), mss, lever) to keep it cheap.
- Files: `sim/timespread.py` (new shape and angle sampler), docs.
- Tests first: [D96] eq. (13) energy closure (sum = flat mirror within 0.25 dB); the GO mean delay against the
  closed facet form at steep grazing (they converge); the [D04] Fig. 3c 52 m readings within 2 dB rms (computed
  1.0); [D96] Fig. 2 within 3.5 dB rms (computed 2.9); Table I kd* within 20 %.
- This is a model change on top of WP2. Default facet stays; the owner decides.
- Not to do: tune the slope per figure.

**WP4 (S): bubble attenuation convention, documentation and a test.** A test that our per-bounce loss equals
exp(-2 beta_I / sin g), and that `bubble_backscatter`'s bracket equals the path-by-path reconstruction of [D96]
eq. (18) in its monostatic limit (ours, 2.1). Note in `surface.py` that [D96] eq. (17) prints twice that
exponent, and is not followed. Wanted: Dahl 1999 eq. (13) to see which convention Dahl kept. No code change.

**WP5 (S, owner decision): the ASIAEX scenario's facts.** [D04] prints the 2001 ECS geometry: source 25 m,
VLAs at 26 and 52 m, about 467-500 m, water 105 m, 8 and 20 kHz, 3 ms CW, U 7 +- 0.5 m/s, H 0.3 +- 0.1 m.
`sst_asiaex.yaml` uses depths 30/50 m and 3 m/s, as ours and SST's choices. Whether TR 0702's example is this
data set is not stated in either source. Ask the owner before changing the scenario. Its records and the
`sst_validation` rows would move.

**WP6 (M, blocked on Dahl 1999): vertical coherence at low grazing.** Our facet model and GO both give d*_V about
2 times short of ASIAEX at 6-11 degrees. Dahl's small-slope model fits. Needs Dahl 1999 eqs. (1), (5), (6), (13),
and the vCZT eq. (1) of [D04] as the test harness (re-typed in 2.5). Until then the xfail of WP1(e) records the
gap.

Must not be done: change `mean_square_slope`, `decay_time` or the bubble convention by default; fit a slope to one
figure; use the figure readings at a tolerance tighter than +-1 dB.

## 6. Not verified / unclear on the page

- [D96] eq. (17): exponent factor 2 against eq. (18)'s own construction and [D97] (2.1). The paper gives no
  reason.
- [D96] p. 750: "kappa ... equals k sqrt(alpha^2 + beta^2)" while calling it the transverse component of
  (k_i - k_s), which is 2 k sqrt(alpha^2 + beta^2). With J0(2 kappa rho) in eq. (5) the formula itself is
  consistent.
- [D96] Fig. 2: the frequency of set 1 run 1 is not stated; the time origin is after leading-edge alignment.
- [D96] the beta_I range 0.05-0.10 is "for wind speeds between 7 and 10 m/s" with no frequency stated.
- [D04] p. 594: beta_1 is estimated with "Equation (5) of Ref. 18". Ref. 18 is Plant (2002), a microwave model.
  Probably Ref. 17 (Dahl 2003, bubbles) was meant; the convention of beta_1 (per crossing?) is not stated.
- [D04] range: Fig. 4 caption 467 m; Sec. IV.C 497 m; Sec. IV.A "nominal 500 m". 467 m reproduces the printed
  9.4-degree iso-speed angle (computed 9.36 degrees).
- [D04] 2d*_V "about 0.5 m" at 52 m, against 4.0 wavelengths (0.30 m, so 2d* = 0.6 m) from Fig. 5: rounding.
- [D04] Fig. 6: the d*_HL crossing at 26 m lies beyond the 2 m plotted.
- [D04] the small-slope model's formulas are in Dahl 1999 (not supplied); the printed model curves cannot be
  reproduced from these two papers alone.
- All Fig. 2, 3c, 5 and 6 values in this file are read from figures.
