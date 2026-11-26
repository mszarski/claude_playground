# Jackson and Richardson (2007), Chapters 12-14 and Appendices J-M: seafloor scattering models

Source: D. R. Jackson and M. D. Richardson, *High-Frequency Seafloor Acoustics*, Springer, 2007.
These are reference notes written in our own words. Formulas were retyped from the page images. Every
item carries its printed page and its equation, table or figure number. "Confirmed" means the item was
read on the page image of this book (REFERENCES.md status words). The book attributes some items to
APL-UW TR 9407: this file says so where it happens. TR 9407 itself was not read.

**What is missing, wrong or risky. Read this first.**

1. **Not in the book.** Three things that `sonar-3k1` and REFERENCES.md section 10 hoped to find here are not
   printed. (a) The interpolation function between the Kirchhoff result near specular and the
   composite/perturbation result elsewhere. Section 13.2.4 (p. 349) and the caption of Fig. 13.8 (p. 351)
   say only that a "smooth interpolation" was used, with no formula. (b) An empirical "large-roughness"
   cross section for gravel or rock, with constants. Chapters 12-14 give none. The book says no experiments
   have been done on well-characterised gravel or rock seafloors (p. 362). It gives Lambert fits to old data
   (Ch. 12) and calls rough rocky seafloors a research problem (p. 376). (c) The Williams-Jackson bistatic
   interpolation formula. Only the general bistatic cross sections (13.9)-(13.11), (14.6)/(14.10) and a
   figure of model against data (Fig. 13.8) are printed.
2. **What the book confirms in our code.** It CONFIRMS most of what `sim/bottom.py` marks "ours". Retyped
   and evaluated independently, these all agree with our functions to better than 0.001 dB, backscatter
   and bistatic: the perturbation factor (13.21)-(13.25), the Kirchhoff cross section (13.10) with
   (13.12)-(13.17), the small-slope cross section (13.11), the structure constant (D.12), the volume
   term (14.8)/(14.9)/(8.26), the composite cut-off rule k_w h_s = 1/2 (13.72)-(13.75) and the large-scale
   slope (13.76)-(13.77). Our `medium_sand` roughness row is exactly the book's medium-sand example
   (Fig. 13.2: a_rho 1.845, nu_p 1.1782, delta_p 0.01624, gamma2 3.25, w2 0.000141 m^0.75).
3. **Where we differ.** (a) The book's composite model multiplies the slope average by Wagner's shadowing
   factor (13.81)-(13.83). Ours has none. Our `composite`/`combined` are therefore HIGHER than the book's
   composite at small grazing angles: +0.9 dB (medium sand, 30 kHz, 5 deg), +3.0 dB (medium sand, 100 kHz,
   5 deg), +3.4 / +6.5 dB (sandy gravel, 30 / 100 kHz, 5 deg), about +1 to +3.6 dB at 10 deg (Sec. 3.4). Our
   default (`small_slope`) does not use slope averaging, so the default does not move. (b) The book's
   volume model (14.10) depends on bistatic angle (density fluctuations scatter as a dipole). Ours is
   isotropic in sigma_v. Away from backscatter the difference is -4 to +11 dB depending on the
   density-compressibility correlation (Sec. 3.6). (c) Book Lambert fits to its Chapter 12 data: rock
   10 log mu = -5.6 dB and gravel -11.7 dB. Our `rock` (-16 dB) is 10.4 dB below the rock fit. Our `pebbles`
   (-9 dB) is 2.7 dB above the gravel fit.
4. **Risky.** Wagner's Q (13.82) is printed with pi^(+1/2). The usual Smith/Wagner shadow function has
   pi^(-1/2) in the same place. The two differ by 2 to 5 dB at 2 deg for rms slopes 0.05-0.35 (Sec. 2.6). Check
   Wagner (1967) before implementing. Two formulas in Appendix L look misprinted against Chapter 13
   (Sec. 6).
5. **Frequency.** The book's subject is "very roughly" 10 kHz to 1 MHz (p. 1). Its fluid models are said to be
   poor below about 10 kHz, where layering is exposed, and possibly above 100 kHz, where there are
   gradients in the top millimetres (p. 378). The roughness model fails above 100 kHz at two sites
   (p. 351). Our 8 kHz flank array is below the book's range. Our 120-330 kHz FLS is above the range where
   its models were tested against data.

Numbers in Sections 3 and 4 come from the scratch script
`/tmp/claude-0/-home-user-claude-playground/32e65dcd-82ae-5f75-a363-7372d2e7f847/scratchpad/jr/book_models.py`
(output `.../scratchpad/jr/out.txt`). It implements the book's formulas without calling ours, then calls
`gpusonar.sim.bottom` from this worktree. To run it, a temporary symlink of the built `_gpusonar` module was
placed in `python/gpusonar/` and removed afterwards. Command:
`PYTHONPATH=<worktree>/python python3 book_models.py`. The script and its output are not in the repository.

---

## 1. Scope

| Part | Printed pages | PDF pages | Read |
|---|---|---|---|
| Ch. 12 Seafloor scattering experiments | 321-330 | 329-338 | in full (page images) |
| Ch. 13 Roughness scattering models | 331-376 | 339-384 | in full |
| Ch. 14 Sediment volume scattering | 377-406 | 385-414 | in full |
| App. J The scattering cross section and plane waves | 519-528 | 523-532 | in full |
| App. K Rough-interface perturbation theory | 529-540 | 533-544 | in full |
| App. L Kirchhoff and small-slope approximations | 541-546 | 545-550 | in full |
| App. M Volume perturbation theory | 547-557 | 551-561 | in full |
| Supporting definitions: Ch. 1 p. 1 (frequency range); Sect. 2.3.2 pp. 21-27 (cross section, reciprocity, energy, Lambert); Sect. 6.1 pp. 174-175 (eqs. 6.4-6.8); App. D pp. 476-478 (D.4-D.24); Sect. 8.3 p. 252 (8.23-8.29) | | 15, 35-41, 186-187, 481-483, 263 | read on the page images, only these pages |
| Not read | Ch. 7 (heterogeneity spectra 7.14-7.17, used by Table 14.1 sites with finite L_c), Ch. 9-10 (elastic/Biot reflection 9.34, 10.66; Table 10.1 Biot parameters for SAX99), Ch. 15-16, App. G (measurement), Sect. 5.6 (rock geoacoustic parameters) | | other agents' parts |

The page offset between printed and PDF page is 8 in Ch. 12-14, 4 in App. J-M and 14 in Ch. 1-2.

---

## 2. What the book gives

### 2.0 Conventions (all confirmed)

| Item | Book | Ours (`sim/bottom.py`) | Mapping checked |
|---|---|---|---|
| Time dependence | exp(-i w t) (p. 252, Sect. 8.3) | sediment wavenumber k(1 + i delta)/nu, decaying branch Im > 0 (`_vertical`) | the same convention for the sediment. Cross sections are |.|^2, so the record's phase convention does not enter |
| Angles | grazing angles theta_i, theta_s. Incident k_i = k_w(e_x cos th_i cos ph_i + e_y cos th_i sin ph_i - e_z sin th_i), scattered k_s with +e_z sin th_s (13.1)-(13.2), p. 333. Backscatter: theta_s = theta_i, phi_s = phi_i + pi (p. 24) | `grazing_in`, `grazing_out`, `azimuth` = angle between horizontal directions, pi = backscatter | ours `azimuth` = book phi_s with phi_i = 0. Equal to <0.001 dB at 7 bistatic geometries (Sec. 3.3) |
| Wave-vector differences | Delta k = k_s - k_i, Delta K = K_s - K_i (horizontal), Delta k_z = k_sz - k_iz, Delta k^2 = Delta K^2 + Delta k_z^2 (13.5)-(13.8) | `q_h`, `q_z`, q | identical |
| Roughness spectrum | W(K) = (2 pi)^-2 int B(R) exp(-i K.R) d2R (6.5); h^2 = int W d2K over all Kx, Ky (6.7); power law W = w2 / K^gamma2 (6.8), K in rad/m (K = 2 pi / lambda, 6.6); w2 "spectral strength" in m^(4 - gamma2) (Fig. 13.2: "w2 = 0.000141 m^0.75"); gamma2 "spectral exponent" | the same: "W(K) = w2 / K^gamma2, K in rad/m, <zeta^2> = int W d2K" | identical (no 1 cm reference length in the book's w2; our `w2_from_cm4` converts the other convention) |
| 1-D spectrum | W1(Kx) = int W dKy (D.16); von Karman 1-D: gamma2 = gamma1 + 1 (D.18), w2 = w1 Gamma(gamma2/2) / (sqrt(pi) Gamma((gamma2 - 1)/2)) (D.19); per cycles/m: Phi1(F) = 2 pi W1(2 pi F) (D.22), phi1 = w1 / (2 pi)^(gamma1 - 1) (D.24), p. 478 | not used in code | conversion recipe for site data |
| Structure function | S(R) = <[f(R0 + R) - f(R0)]^2> (D.5) = 2 int (1 - cos K.R) W d2K (D.10); power law: S = C_h^2 R^(2 alpha) (D.11), C_h^2 = 2 pi w2 Gamma(2 - alpha) 2^(-2 alpha) / (alpha (1 - alpha) Gamma(1 + alpha)) (D.12), alpha = gamma2/2 - 1 (D.13), p. 477 | `structure_constant`: Ch2 = 2 pi w2 Gamma(1 - a) / (4^a a Gamma(1 + a)) | identical (Gamma(2 - a) = (1 - a) Gamma(1 - a)); numerically 1.5757e-3 both for w2 = 1.41e-4, gamma2 = 3.25 |
| Speed and loss | a_p = c_p / c_w complex (8.23); nu_p = c_phase / c_w real (8.24); a_p = nu_p / (1 + i delta_p) (8.25); attenuation alpha_p = 40 pi f delta_p / (nu_p c_w ln 10) dB/m (8.26); per wavelength 40 pi delta_p / ln 10 (8.27); Q = 1/(2 delta_p) (8.28), p. 252 | `nu`, `delta`; `volume_strength` uses (20/ln 10) k delta / nu = the same dB/m; `DB_PER_WAVELENGTH` = 40 pi / ln 10 | identical |
| Density ratio | a_rho = sediment / water density | `rho` | identical |
| Cross section | sigma, dimensionless, "per unit area per unit solid angle"; scattering strength S_b = 10 log10 sigma, units dB, not "dB per m^2" (2.15)-(2.17), p. 23 | `strength` (linear), `backscattering_strength_db` | identical |
| Volume parameter | sigma_2 = sigma_v / alpha_p, alpha_p in dB/m (14.9), p. 381-382; sigma_v in m^2/m^3 = 1/m (p. 380) | `sigma2` = volume cross section per unit volume / attenuation in dB/m | identical |
| Volume heterogeneity spectra | normalised fluctuations gamma_rho = d rho / <rho>, gamma_kappa = d kappa / <kappa> (14.20)-(14.21); 3-D spectra W_ab(k) = (2 pi)^-3 int B_ab(r) exp(-i k.r) d3r (14.25), dimension length^3; power law W = w3 / k^gamma3 (14.28), w3 in m^(3 - gamma3) (Table 14.1 caption) | none in code | for a future perturbation volume model (WP-4) |
| Grain size | phi units, M_z (Ch. 12: "mud" M_z >= 5 phi, gravel M_z <= 0 phi, p. 323) | `mean_grain_size_phi` | same unit |

### 2.1 General properties and definitions (Sect. 2.3.2, Sect. 13.1, App. J). Confirmed.

- Definition (2.15), p. 23: <|P_s|^2> = |P_i|^2 A sigma / r_s^2 for a patch of area A in the far field of the
  source, with absorption and refraction in the water neglected. This is a statistical (ensemble) average.
  Pings fluctuate, but sigma does not (p. 23). Sigma must not depend on pulse length or transducer
  height. If it does, the experiment is suspect (p. 519).
- Total scattered power (2.16): U_s = A |P_i|^2 / (2 rho_w c_w) int_{2 pi} sigma dOmega_s.
- Reciprocity (2.18), p. 24: sigma(th_i, ph_i, th_s, ph_s) = sigma(th_s, ph_s, th_i, ph_i). Plane-wave form
  (J.5): beta_w(K_s) T(K_s, K_i) = beta_w(K_i) T(-K_i, -K_s). The elastic perturbation result (13.27)-(13.31)
  is written so that reciprocity is evident (p. 343).
- Energy (2.20), p. 24: int_{2 pi} sigma dOmega_s + |V_wwc|^2 sin th_i < sin th_i (V_wwc is the coherent
  reflection coefficient). Weaker form without the coherent term (2.21), p. 25.
- Lambert's law (2.22), p. 26: sigma = mu sin th_s sin th_i (backscatter mu sin^2 th). Energy bound (2.23):
  mu < 1/pi, i.e. 10 log mu < -4.97 dB.
- Small angles, p. 27: the cross section must vanish at least as fast as sin^2 th_i as th_i goes to 0. A
  measured backscattering strength that flattens near grazing indicates multipath or noise.
- Coherent reflection (13.14), p. 335 (Kirchhoff and small-slope alike): V_wwc(th) = V_ww(th) exp(-2 k_w^2 h^2
  sin^2 th_i) (Eckart). The book warns against using it as the total bottom reflection when scattering is
  not concentrated near specular, because the scattered field then contributes down-range.
- Incoherent reflection coefficient (13.18), p. 337 and (L.22)-(L.27), pp. 544-545: near specular,
  |V_wwi|^2 = |V_ww|^2 - |V_wwc|^2 = |V_ww|^2 (1 - exp(-Delta k_z^2 h^2)), with |V_wwi|^2 = (1/sin th_i) int
  sigma d2Omega_s (L.23). The mean-square near-specular field from a point source (L.21) is
  <|P_s|^2> = 2 s_0^2 |V_wwi|^2 / (d_i + d_s)^2: like a flat-bottom reflection with 1/(d_i + d_s)^2 spreading.
  This needs a CW or long enough pulse (p. 338).
- T-matrix formalism (App. J): incident and scattered fields as plane-wave spectra (J.1)-(J.3).
  Cross section from the second moment (J.13)-(J.14): sigma(K_s, K_i) = k_w^2 beta_w^2(K_s) C(K_s, K_i, K_i),
  where <T_s T_s'*> = C delta(K_s - K_i - K_s' + K_i'). Mean T-matrix = V_wwc delta (J.12).
- Sonar-equation integral (J.15), p. 522: <|P_s(r_s)|^2> = int_{z=0} sigma(K_s, K_i) |P_i(r)|^2 |r_s - r|^-2 d2R.
  The angles are taken per patch from the source and receiver positions (J.16)-(J.17).

### 2.2 Far-field criteria for using sigma (App. J.2, pp. 523-528). Confirmed.

- The "conventional" criterion r > d^2/lambda on the ensonified patch (J.29) cannot be met when the patch
  grows with range (d = Psi r gives r < lambda / Psi^2). The book calls this "unduly pessimistic" (p. 381).
- The rigorous criterion uses the correlation length L_f of the effective surface field (J.34):
  k_w L_f^2 / |r - r'| << 1. Here L_f is about lambda / Delta theta, where Delta theta is the angular width of the
  narrowest feature of sigma (J.36 and the text below it, p. 527).
- (J.15) is valid when (p. 527): (1) the boundary is in the far field of the source; (2) source and receiver
  are in the far field with respect to L_f; (3) the ensonified region is much larger than L_f; (4) the
  field point is at least a few wavelengths from the boundary.
- Broad backscatter lobes mean a short L_f. A narrow forward (specular) lobe means a long L_f, so
  forward-scatter measurements need more range (p. 528). Strong reflections from strata deeper than a
  wavelength make the layer depth act as L_f and need larger ranges (p. 528). A sigma that depends on
  sonar height may indicate such structure (p. 519).

### 2.3 Small-roughness perturbation theory (Sect. 13.2, App. K). Confirmed.

General form, all wave theories (13.9) = (K.9), pp. 334, 530:

    sigma = k_w^4 |A_ww(K_s, K_i)|^2 W(Delta K)

This is second order in the relief (the superscript (2) in App. K). First-order T-matrix (K.3):
T(1) = (i k_w / beta_w(K_s)) A_ww F(K_s - K_i), with F the relief transform (K.4). Normalisation of the
relief transform: <F(K1) F*(K2)> = W(K1) delta(K1 - K2) (K.8). General identity (K.5) = (L.9):
A_ww(K_i, K_i) = -2 beta_w^2(K_i) V_ww(K_i) (Voronovich). This is what makes the small-slope and Kirchhoff
results agree in the specular direction, and it is "a useful test of numerical algorithms" (p. 530).
Bragg condition, backscatter (13.20), p. 339: Delta K = 2 k_w cos th_i. Scattered vertical wavenumber
(13.19): k_sz = sqrt(k_w^2 - K_s^2).

**Fluid (13.21)-(13.25), p. 340** (attributed to D. Jackson 1994 and derived in K.1 and M.1.2; lossy
version D. Jackson and Briggs 1992; lossless Kuo 1964):

    A_ww = (1/2) [1 + V_ww(th_i)] [1 + V_ww(th_s)] G
    G = (1 - 1/a_rho) [cos th_i cos th_s cos phi_s - B] - 1 + 1/(a_p^2 a_rho)
    B = sin th_pi sin th_ps / (a_p^2 a_rho)
    sin th_p = sqrt(1 - a_p^2 cos^2 th)     (for th_i and th_s)

Wave-vector form (K.44), p. 536: the same with (K_s.K_i)/k_w^2 and beta_p(K_i) beta_p(K_s)/(a_p^2 a_rho).

**Layered or graded fluid (13.26), p. 340** (Moe and Jackson 1994): replace B by

    B = sin th_i sin th_s [1 - V(th_i)][1 - V(th_s)] / ([1 + V(th_i)][1 + V(th_s)]) a_rho

with V the reflection coefficient of the stratified seafloor, and a_p, a_rho evaluated just below the
interface (p. 341). Conditions: parameters roughly constant over the vertical extent of the relief; no
horizontal variation; fluid-like just below the rough interface; all buried interfaces smooth. An
elastic or poroelastic upper layer cannot be obtained by letting a fluid layer vanish (p. 341). Rough buried
interfaces: McDaniel 1992, Lyons et al. 1994, Ivakin 1994b, Tang 1996a (not printed). Check: for a
homogeneous half-space, (1 - V)/(1 + V) = P/(a_rho sin th), so (13.26) reduces to (13.23) (our algebra).

**Elastic (13.27)-(13.34), pp. 342-343** (derivation K.2; equivalent to Dacol and Berman 1988, D. Jackson and
Ivakin 1998). With S = cos th_s cos th_i cos phi_s (13.32), cos th_p = a_p cos th (13.33), cos th_t = a_t cos th
(13.34), and a_t the complex shear speed ratio:

    A_ww = (1/2) ( D1 [1+V(th_s)][1+V(th_i)] + D2 [1-V(th_s)][1+V(th_i)]
                 + D3 [1+V(th_s)][1-V(th_i)] + D4 [1-V(th_s)][1-V(th_i)] )                       (13.27)
    D1 = -1 + S + 1/(a_p^2 a_rho cos2th_ts cos2th_ti)
         - a_t^2 [(a_t^-2 - 2cos^2 th_s - 2cos^2 th_i + 2S) S + 2 cos^2 th_s cos^2 th_i] / (a_rho cos2th_ts cos2th_ti)   (13.28)
    D2 = -(4 a_t^3 sin th_s sin th_ts / (cos2th_ts cos2th_ti)) [a_p^-2 sin^2 th_pi cos^2 th_s + (cos^2 th_i - S) S]      (13.29)
    D3 = -(4 a_t^3 sin th_i sin th_ti / (cos2th_ts cos2th_ti)) [a_p^-2 sin^2 th_ps cos^2 th_i + (cos^2 th_s - S) S]      (13.30)
    D4 = (2 a_t^4 a_rho sin th_s sin th_i sin th_ts sin th_ti / (cos2th_ts cos2th_ti))
         x [2(a_t^-2 - 2S) S - 4 cos^2 th_s cos^2 th_i (1 - 2 a_t^2 a_p^-2)] - (a_rho - 1) sin th_s sin th_i            (13.31)

(cos2th_t means cos(2 theta_t). V is the elastic flat-interface reflection coefficient, (9.34).) Statements:
for sand with shear speed about 100 m/s, shear is a slight refinement. For rock (shear speed 2-4 km/s)
it is dominant, with a dip near the Rayleigh angle (Fig. 13.5, p. 344: 30 kHz, w2 0.000141 m^0.75, gamma2 3.25;
sand with shear speed 250 m/s nearly equals fluid; rock with nu_t = 1.3 departs strongly between about 15 and
40 deg, the parameters being those of Figs. 9.3 and 9.4, not read). Valid only for homogeneous seafloors.

**Poroelastic (Biot) (13.35)-(13.60), pp. 344-347** (Williams et al. 2001a; matrix method of Ivakin; K.3).
A_ww is the first element of A_all = Y1(K_s) [P(3)(K_s)]^-1 B(K_s, K_i) V(K_i) (13.36). B =
k_w^-1 [(K_s1 - K_i1) E(1)(K_i) + (K_s2 - K_i2) E(2)(K_i)] - E(3)(K_i) Y2(K_i) (13.38). Y1, Y2 are diagonal
matrices of nu_q = beta_q / a_q (13.39)-(13.41). E(n) is a 5 x 6 matrix of P(n) and Q(n) elements
(13.42)-(13.60). These involve the Biot coefficients gamma_1, gamma_2, H, C, M, mu (Sect. 10.1.3, not read here)
and unit vectors e+-. The element formulas are on pp. 346-347 and should be retyped from the page if this
is implemented. Fig. 13.6, p. 348 (Biot parameters of Table 10.1, w2 0.0000433 m^0.96, gamma2 3.04, at
0.4/2/10/50 kHz): the effective-density fluid matches Biot closely. A pure fluid is higher by an amount that
shrinks at low frequency.

**Accuracy and tests (Sect. 13.2.4, pp. 348-353).**
- Frequency law (13.61), p. 351: if nu_p and delta_p do not depend on frequency, sigma is proportional to
  f^(4 - gamma2). For 3 < gamma2 < 4 this is 0 to 10 dB per decade (gamma2 = 3.25 gives 7.5 dB/decade, the
  slope of the sand line in Fig. 12.7).
- Fig. 13.7 (p. 350), six sandy sites at 35-40 kHz, inputs from Table 13.1 (below). In words: the fit is
  reasonably good. The model overpredicts at SAX99 (c) and Panama City 84 (d), likely because poroelastic
  effects are neglected (Williams et al. 2002b error analysis). A maximum near the critical angle is
  seen in data (c), (d), (e) and predicted. None is predicted or seen at Key West (f, 7 phi).
- Fig. 13.9 (p. 352): frequency dependence at 10 or 20 deg. Slopes match to about 100 kHz, given errors of
  2-3 dB. The model overpredicts at Panama City 98, SAX99 and Panama City 84. Above 100 kHz the
  Jacksonville and Panama City 84 data fall below the model ("the model has failed at these higher
  frequencies", p. 351). Quinault: the roughness was highly directional, so an isotropic spectrum is not
  realistic (p. 353).
- Pouliquen and Lyons 2002: perturbation theory fitted backscatter at 140 kHz on a silty sand ("Tellaro").
  Vertical gradients had a significant effect on roughness scattering (p. 353).
- SAX99, Biot model (Figs. 13.10-13.11, pp. 354-355; Williams et al. 2002b): within uncertainties, Biot fits
  better than fluid at 40 kHz. Agreement is satisfactory to 100 kHz. At 300 kHz there is substantial
  disagreement, "tentatively ascribed to volume scattering of unknown type". The model's fall-off above
  150 kHz comes from a change in slope of the measured roughness spectrum.
- Bistatic: Day and Yamamoto 1999 (3.75, 7.5, 15 kHz, sandy, roughness not measured); Choi et al. 2001
  (240 kHz, assigned parameters); Williams and Jackson 1997/1998, Fig. 13.8 (p. 351). Fig. 13.8 is a fluid
  model with Kirchhoff near specular, perturbation elsewhere and a smooth interpolation (not printed).
  Agreement is within measurement error; input errors were not counted.

**Table 13.1 (p. 349): geoacoustic inputs of Figs. 13.7 and 13.9** (confirmed. w2 units are not printed in
the table. They are m^(4 - gamma2), as the SAX99 row equals Fig. 13.10's "0.0000433 m^0.96". The rows were
reprocessed to the upper 2 cm of the cores for 35-40 kHz, p. 348):

| Site | M_z (phi) | a_rho | nu_p | delta_p | gamma2 | w2 | data source (Fig. 13.7 caption) |
|---|---|---|---|---|---|---|---|
| Jacksonville | 0.51 | 1.956 | 1.087 | 0.049 | 2.47 | 0.00000455 | Stanic et al. 1989 |
| Panama City 98 | 0.62 | 1.98 | 1.113 | 0.0324 | 3.12 | 0.00014 | (Fig. 13.9: Stanic et al. 1998) |
| Panama City 93 | 0.68 | 1.948 | 1.097 | 0.032 | 3.12 | 0.0001475 | D. Jackson et al. 1996a |
| SAX99 | 1.28 | 2.00 | 1.16 | 0.010 | 3.04 | 0.0000433 | Williams et al. 2002b |
| Falmouth 2 | 2.07 | 1.96 | 1.16 | 0.004 | 3.0 | 0.000035 | (Fig. 13.9: D. Jackson et al. 1986b) |
| Panama City 84 | 2.63 | 1.93 | 1.115 | 0.0185 | 1.89 | 0.000000356 | Stanic et al. 1988 |
| Quinault | 2.95 | 1.94 | 1.113 | 0.0115 | 3.92 | 0.0041 | D. Jackson and Briggs 1992 |
| Key West | 7.00 | 1.499 | 1.003 | 0.0153 | 3.29 | 0.000464 | Briggs et al. 2002a |

### 2.4 Kirchhoff approximation (Sect. 13.3, App. L.1, L.4). Confirmed.

(13.10), p. 334, with (13.12)-(13.13):

    sigma = |V_ww(th_is)|^2 / (8 pi) [Delta k^2 / (Delta K Delta k_z)]^2 I_K
    I_K = (Delta K^2 / 2 pi) int exp(-i Delta K.R) [exp(-Delta k_z^2 S(R)/2) - exp(-Delta k_z^2 h^2)] d2R
    th_is = asin(Delta k / (2 k_w))      (local specular angle; Thorsos, private communication, as cited in D. Jackson 1994)

Isotropic (13.15): I_K = int_0^inf J0(u) [exp(-Delta k_z^2 S(u/Delta K)/2) - exp(-Delta k_z^2 h^2)] u du. Power law
(h infinite) (13.16)-(13.17), pp. 335-336: I_K = int_0^inf J0(u) exp(-q u^(2 alpha)) u du, with
q = (1/2) C_h^2 Delta k_z^2 Delta K^(-2 alpha). The integral "requires special consideration numerically"
(Drumheller and Gragg 2001). The second term subtracts the coherent part (p. 335). Assumes Gaussian relief
(p. 356), and uses V at the facet-specular angle (an approximation beyond the tangent plane, p. 542).
Derivation (L.1)-(L.5): the surface field is replaced by the tangent-plane field, giving the coherent
reflection (L.5) V_wwc = V_ww exp(-(1/2) Delta k_z^2 h^2) with Delta k_z = 2 k_w beta_w(K_i).

Behaviour (pp. 336, 356): the Kirchhoff result is too large at moderate and small grazing angles. A rising
trend towards grazing means energy conservation is violated. Near vertical it widens and lowers with
frequency, with larger w2, and with smaller gamma2 (Fig. 13.12, medium sand of Fig. 13.2 at 5-80 kHz).
Near-vertical data can constrain the spectrum (refs. p. 356).

**High-frequency (geometric-optics, facet) limit** (13.62)-(13.71), pp. 357-360; derivation (L.28)-(L.36):

    sigma = (1/4) |V_ww(th_is)|^2 (Delta k^4 / Delta k_z^4) p_s(s)                           (13.62)
    p_s(s) = exp(-(1/2) s^T B_s^-1 s) / (2 pi ||B_s||^(1/2)),  s = Delta K / Delta k_z         (13.63), (13.66)
    isotropic: sigma = |V(th_is)|^2 / (8 pi sigma_s^2) (Delta k^4 / Delta k_z^4) exp(-Delta K^2 / (2 sigma_s^2 Delta k_z^2))   (13.67)
    sigma_s^2 = <(df/dx)^2> = pi int K^3 W(K) dK   (per axis)                                  (13.68), (13.69)
    backscatter: sigma = |V(90 deg)|^2 exp(-cot^2 th / (2 sigma_s^2)) / (8 pi sigma_s^2 sin^4 th)   (13.70)
    B_smn = int K_m K_n W(K) d2K                                                               (13.71)

It is independent of frequency and concave downward, with no inflection (Fig. 13.14: rms slopes 0.02, 0.04,
0.08, 0.16). For power laws the slope integral diverges unless the spectrum falls faster than K^-4, so a
cut-off must be imposed. A frequency-dependent cut-off is "ad hoc" but finds some support in the
composite model (p. 360). Tests (Sect. 13.3.2): near-vertical model-data differences of up to 10 dB
(Fig. 13.15: Falmouth 20 kHz, Puget Sound 35 kHz, D. Jackson et al. 1986a), possibly measurement error at
narrow lobes. Laboratory: good near vertical (Thorne and Pace 1984), "moderate accord" at smaller grazing
angles (Thorne et al. 1988).

### 2.5 Small-slope approximation, lowest order (Sect. 13.4, App. L.2). Confirmed.

(13.11), p. 334:

    sigma = k_w^4 |A_ww|^2 I_K / (2 pi Delta K^2 Delta k_z^2)

The A_ww is the perturbation factor of 2.3 (fluid, elastic or poroelastic). It is valid as written only
for a homogeneous seafloor, with no gradients or layering (p. 334). T-matrix form (L.6)-(L.8): Phi_0(K_s, K_i)
= -2 k_w^2 A_ww / Delta k_z, with Phi_0(K_i, K_i) = 2 k_w beta_w V_ww. Limits (p. 336): small k_w h reduces it to
(13.9). Near specular it equals the Kirchhoff result (via (L.9)). Fig. 13.2 (p. 337, medium sand, 30 kHz):
small-slope and perturbation overlap below about 70 deg; small-slope and Kirchhoff overlap above about 70 deg.
Book's judgement (pp. 336, 361-362): it covers all angles with one expression, "with no loss in accuracy"
compared with the traditional approximations, so composite and Kirchhoff are less needed. Model-data
tests are covered by those of the perturbation and Kirchhoff approximations. Soukup and Gragg 2003 found
that fits to rock data give plausible parameters. The lowest-order small-slope result "does not
incorporate the effects of large-scale tilting and shadowing" (p. 368).

### 2.6 Composite roughness (Sect. 13.5, pp. 362-368). Confirmed.

The heuristic form follows D. Jackson et al. 1986a and Mourad and Jackson 1989. It is derivable from first
principles (Brown 1978/1980, McDaniel and Gorman 1983). Large scales tilt the seafloor and shadow it. Small
scales scatter. Near-vertical incidence is not treated by this form.

    cut-off criterion:      k_w h_s = 1/2                                    (13.72)  [D. Jackson et al. 1986a]
    small-scale variance:   h_s^2 = int_{K > K_c} W d2K = 2 pi w2 / ((gamma2 - 2) K_c^(gamma2 - 2))   (13.73), (13.74)
    cut-off:                K_c = (8 pi w2 k_w^2 / (gamma2 - 2))^(1/(gamma2 - 2))                   (13.75)
    large-scale slope:      sigma_sL^2 = (1/2) int_{K < K_c} K^2 W d2K = pi w2 K_c^(4 - gamma2) / (4 - gamma2)   (13.76), (13.77)
    conditions:             K_c < Bragg wavenumber (sets a maximum grazing angle); sigma_sL < 0.1 (13.78); theta < 70 deg
    local angle:            theta_L = theta + s_x  (in-plane slope only)                            (13.79)
    average (backscatter):  sigma_cr(theta) = R(theta, sigma_sL) / (sqrt(2 pi) sigma_sL) int_{-theta}^{inf} sigma(theta + s_x) exp(-s_x^2 / (2 sigma_sL^2)) ds_x   (13.80)
    Wagner shadowing:       R = (1 - exp(-2Q)) / (2Q),  Q = (1/(4t)) {pi^(1/2) exp(-t^2) - t [1 - erf(t)]},  t = tan theta / (sqrt(2) sigma_sL)   (13.81)-(13.83) [Wagner 1967]

sigma(.) inside the integral is the small-roughness backscatter cross section, "typically" first-order
perturbation theory. The lower limit -theta removes slopes facing away from the source. Shadowing by crests
needs the extra factor R. The pi^(+1/2) in Q is as printed. See Sec. 6 and the table below.

Wagner factor 10 log R as printed (pi^(1/2)) / with pi^(-1/2) (script output):

| sigma_sL | 2 deg | 5 deg | 10 deg | 20 deg | 30 deg |
|---|---|---|---|---|---|
| 0.05 | -2.28 / -0.44 | -0.25 / -0.02 | 0.00 / 0.00 | 0 | 0 |
| 0.1 | -5.01 / -1.45 | -1.60 / -0.26 | -0.24 / -0.02 | 0 | 0 |
| 0.2 | -8.23 / -3.37 | -4.04 / -1.04 | -1.57 / -0.25 | -0.21 / -0.02 | -0.01 / 0 |
| 0.35 | -10.81 / -5.59 | -6.54 / -2.25 | -3.47 / -0.82 | -1.14 / -0.16 | -0.31 / -0.03 |

Statements (pp. 365-368):
- Fig. 13.18 (p. 366): composite corrections to the Fig. 13.7 perturbation curves are small except where
  sigma_sL is large: panel (b) 0.134 and panel (f) about 0.27 (small print, see Sec. 6). There condition
  (13.78) is violated, and the corrections "are suspect and do not improve the fit". It could not be
  applied for Panama City 84 (gamma2 = 1.89 < 2: h_s diverges). An exponent slightly under 4 gives an
  unrealistically large slope (D. Jackson and Briggs 1992). These are artefacts of extrapolating the power law.
- "The cutoff wavenumber is dependent on a rather arbitrary criterion (13.72) ... no unique criterion is
  available" (pp. 365-366). One option is to choose it to fit a more accurate approximation, e.g.
  Kirchhoff near normal incidence (Novarini and Caruthers 1994).
- Near-vertical (pp. 366-367): McDaniel and Gorman 1982 ascribe near-vertical scattering to large scales via
  (13.70). Novarini and Caruthers 1994 replace V by the coherent reflection coefficient (13.14) with
  h = h_s. With (13.72) this divides V by sqrt(e). They use (13.76) WITHOUT the factor 1/2, so their slope
  variance is twice (13.77). Fig. 13.19 (p. 368, medium sand of Fig. 13.2 at 20 kHz): the high-frequency form
  is "consistently lower" than full Kirchhoff. The printed rms slopes for w2 = 0.0000422, 0.0001406 and
  0.0004218 are 0.0227, 0.0595 and 0.1432. **Reproduced exactly** with (13.75) and (13.77): 0.0227, 0.0595,
  0.1432 (our hand computation; `large_scale_slope` uses the same formulas).
- "The composite-roughness approximation may be most useful where its corrections are small" (p. 367). Its
  main motivation was numerical economy. It remains useful when the relief has a clear separation of
  scales (e.g. ripples much longer than the wavelength) (pp. 367-368).
- Roughness-volume interaction (Sect. 14.1.2, p. 383): the volume cross section can be averaged over
  large-scale slopes in the same way, "readily implemented numerically via (13.80)" (Ivakin and Lysanov 1981b;
  D. Jackson et al. 1986a; Mourad and Jackson 1989). This neglects diffraction by smaller scales. Correlation
  between relief and sediment layering needs App. M.1.2.

### 2.7 Interpolation between regimes; empirical rough-bottom law; bistatic model

- **Interpolation Kirchhoff to composite/perturbation: not printed.** Only "smooth interpolation" (p. 349,
  Fig. 13.8 caption p. 351). The book's own recommendation is the small-slope approximation as the single
  expression (p. 336).
- **Large-roughness empirical law for gravel/rock: not printed in Ch. 12-14.** What is printed for coarse and
  hard seafloors is in Sec. 2.11 (Lambert fits). The book also states that no experiments have yet been done
  on well-characterised gravel or rock (p. 362), that "rough, rocky seafloors are an especially challenging
  area for future research" (p. 376), that the extremely high levels of the TR 9407 rock data exceed the
  Lambert limit and suggest backscatter enhancement (p. 324), and that elastic perturbation theory with
  volume terms predicts strong volume scattering in rock (Sect. 14.2.3).
- **Bistatic.** Every roughness formula above is bistatic as written: (13.9)-(13.13) through Delta K and
  Delta k_z; the elastic (13.27) with S = cos th_s cos th_i cos phi_s. Volume: (14.6), (14.8) and (14.10) are
  bistatic. Experimental notes (Sect. 12.2, pp. 329-330): Urick 1960 at 22 kHz (shallow water, 3 km
  separation, multipath) showed little dependence on bistatic angle. Schmidt 1971 (3.2-6.3 kHz) was
  compromised by surface multipath. Stanic et al. 1991/1993 (20-180 kHz) near backward: little variation,
  no backscatter enhancement on sand. Bistatic model against data: Fig. 13.8 (Williams and Jackson 1997;
  grazing bins 5-15, 10-20, 15-25, 20-30 deg; bistatic angle 0-180 deg). Volume bistatic pattern for mud:
  Fig. 14.3 (Sec. 2.9).

### 2.8 Discrete features on the interface (Sect. 13.6, pp. 368-369). Confirmed.

No formulas printed. Stanton 2000 (spheres on a flat interface, geometric scattering, interface
neglected): normal-incidence sigma = (single-scatterer backscatter cross section / its cross-sectional
area, about independent of frequency and size) x (packing factor = fraction of area covered). At small
grazing angles this is multiplied by the Lambert factor sin^2 theta for shadowing and multiple scattering.
Williams et al. 2001b include coherent reflection by the seafloor. The model agreed reasonably with 40 kHz
data from glass marbles of radius 1.75 cm spread on sand. Dorfman and Dyer 1999: rock facets
(low-frequency reverberation). Research issue.

### 2.9 Sediment volume scattering (Ch. 14, App. M). Confirmed.

**Basic model (Stockhausen 1963 extended), (14.1)-(14.8), pp. 379-381.** Assumptions: weak scattering; a
cross section linear in volume; penetration small against ranges; interface effects on scattering
neglected (p. 378). Flat interface. Plane-wave incidence at the interface. Green's functions (8.6) are used,
so subcritical penetration (the evanescent wave) is kept.

    |G_wp|^2 = |V_wp(th_i)|^2 / r_i^2 exp(2 Im[k_p beta_p(K_i)] z)                 (14.2)
    |G_pw|^2 = |V_wp(th_s)|^2 / (|a_rho|^2 r_s^2) exp(2 Im[k_p beta_p(K_s)] z)     (14.3)  (reciprocity, 8.69)
    sigma = |V_wp(th_i)|^2 |V_wp(th_s)|^2 / |a_rho|^2 int_{-inf}^0 sigma_v(z) exp(2 k_w Im[P(th_i) + P(th_s)] z) dz   (14.6)
    P(theta) = sqrt(a_p^-2 - cos^2 theta)                                          (14.7)
    depth-uniform sigma_v:  sigma = |V_wp(th_i)|^2 |V_wp(th_s)|^2 sigma_v / (2 k_w |a_rho|^2 Im[P(th_i) + P(th_s)])   (14.8)
    sigma_2 = sigma_v / alpha_p    (alpha_p in dB/m, (8.26); Mourad and Jackson 1989)  (14.9)

V_wp is the flat-interface pressure transmission coefficient (1 + V_ww for a fluid). Typical sigma_2: 0.0001
to 0.004. Values above 0.004 may indicate multiple scattering (p. 382, D. Jackson et al. 1986a). With sigma_2
independent of frequency and attenuation linear in f, sigma is nearly independent of frequency (p. 382).
Fig. 14.2 (p. 382): 30 kHz, c_w 1500 m/s, a_rho 1.5, delta_p 0.015, sigma_2 0.002, nu_p 0.98/1.05/1.1. Volume
scattering is suppressed below the critical angles (about 18 and 25 deg; reproduced: 17.8 and 24.6 deg).

**Perturbation (Born) volume cross section (14.10)-(14.19), pp. 383-385** (derived in M.1; Ivakin):

    sigma_v = (pi |k_p|^4 / 2) [ W_kk(Delta k_p) + 2 Re{ (k_ps . k_pi) k_p*^2 / |k_p|^4 W_rk(Delta k_p) }
                                 + |k_ps . k_pi|^2 / |k_p|^4 W_rr(Delta k_p) ]                       (14.10)
    k_pi = K_i - e_z k_w P(th_i),  k_ps = K_s + e_z k_w P(th_s)                                      (14.13), (14.14)
    k_pi.k_pi = k_ps.k_ps = k_p^2,  k_p = k_w (1 + i delta_p) / nu_p                                 (14.17), (14.18)
    Delta k_p = Re{k_ps - k_pi}    (real part: derived in App. M)                                  (14.19)

Density fluctuations scatter as a dipole along the refracted incident direction. Compressibility
fluctuations scatter as a monopole (p. 384). Correlation model (14.26)-(14.27), p. 386: W_kk = mu^2 W_rr,
W_rk = rho_rk mu W_rr, mu >= 0, -1 <= rho_rk <= 1. Common practice is rho_rk = -1 (perfect negative
correlation). The book argues that some correlation must exist and that perfect correlation is unproven.
Power law (14.28): W_ab = w3_ab / k^gamma3. gamma3 = 3 is self-similar (w3 dimensionless) and gives a nearly
frequency-independent cross section (p. 387; Ivakin 1981, D. Jackson and Briggs 1992). Fig. 14.3 (p. 388):
bistatic strength map for mud (35 kHz, c_w 1500, a_rho 1.5, nu_p 0.98, delta_p 0.01, gamma3 3.0,
w3_rr 0.0005, mu 1, source and receiver 50 m up, 400 m apart). In words: with rho_rk = -1 forward scattering
is weak and backscatter substantial. With rho_rk = 0 forward scattering is substantial.

Our evaluation of (14.10) for backscatter (k_ps.k_pi = -k_p^2): sigma_v = (pi |k_p|^4 / 2)(mu^2 - 2 rho_rk mu + 1)
W_rr(|Delta k_p|) with |Delta k_p| = 2 k_w sqrt(cos^2 th + (Re P)^2). This is our algebra, checked in the script.

**Tests (Sect. 14.1.5, pp. 387-390).** Lyons et al. 1994 (6.5 kHz, silt-clay over sand): within the +-5 dB
spread for 12-57 deg. Rogers and Yamamoto 1999 (Straits of Florida, 7.5 and 15 kHz; sound-speed spectrum
from tomography with gamma3 2.3 to 2.8): general agreement with differences up to 10 dB. Fig. 14.4 (p. 389):
five soft-sediment sites with the inputs of Table 14.1. Agreement is "satisfactory" but clouded by
heterogeneity uncertainty (small cores, vertical-only analysis, isotropy assumed, rho_rk not measured but
set to -1; Arafura had buried shell).

**Table 14.1 (p. 390): volume inputs** (rho_rk = -1, aspect ratio 1 at all sites; w3 in m^(3 - gamma3)):

| Site | M_z (phi) | a_rho | nu_p | delta_p | gamma3 | w3 | L_c (m) | mu | freq (Fig. 14.4) | data |
|---|---|---|---|---|---|---|---|---|---|---|
| Arafura | 5.6 | 1.39 | 0.988 | 0.0055 | 3.45 | 1.32e-3 | inf | 1.389 | 20 kHz | D. Jackson and Briggs 1992 |
| San Francisco | 6.4 | 1.41 | 1.002 | 0.0104 | 3.63 | 1.37e-2 | inf | 1.200 | 40 kHz | D. Jackson and Briggs 1992 |
| Orcas | 8.1 | 1.41 | 0.977 | 0.0019 | 4.00 | 6.44e-3 | 0.032 | 0.934 | 40 kHz | Jones and Jackson 1997 |
| Porto Venere | 9.1 | 1.46 | 0.993 | 0.0011 | 7.60 | 5.13e9 | 0.0014 | 1.667 | 140 kHz | Pouliquen and Lyons 2002 |
| P. della Mariella | 7.2 | 1.66 | 1.002 | 0.0028 | 8.00 | 3.20e11 | 0.0012 | 1.667 | 140 kHz | Pouliquen and Lyons 2002 |

The finite-L_c rows need the Ch. 7 spectral form with a cut-off length (7.14/7.16/7.17). It was not read
here (Sec. 6). Arafura and San Francisco heterogeneity parameters are from Yamamoto 1996.

**Accuracy (Sect. 14.1.6, pp. 390-391).** Jones and Jackson 2001 compared exact 2-D Monte Carlo results with
perturbation theory. Exponential density spectrum (14.29): W_rr(k) = (sigma_rho^2 / (2 pi L_c)) / (k^2 + L_c^-2)^(3/2).
The others are proportional, with rho_rk = -1. Fig. 14.5: a_rho 2.0, nu_p 1.10, delta_p 0.00183, mu 2,
normalised density std 0.1, L_c = one wavelength. Windowing (half-space) effects are significant near the
specular direction. The ordinary perturbation model is "quite accurate in other directions". With
typical parameters the model is "comfortably within its realm of accuracy".

**Discrete heterogeneity (Sect. 14.1.7, pp. 391-401).**
- sigma_v = int_0^inf N(a) sigma_d(a) da (14.30), isotropic single scattering. It is inserted in (14.6) or (14.8).
- Bubbles (Medwin and Clay 1998, Ch. 8; resonance and losses after Lyons et al. 1996, correcting Anderson and
  Hampton 1980a): sigma_d = a^2 / ([(w0/w)^2 - 1]^2 + delta^2) (14.31).
  w0 = (1/a) sqrt(3 gamma P0 / (A rho) + 4 mu' / rho) (14.32). A and B (14.33)-(14.34). X = a (2 w0 rho0 S_p / C0)^(1/2) (14.35).
  delta = delta_r + delta_h + delta_s (14.36): delta_r = a Re{k_p} (14.37), delta_h = B (w0/w)^2 (14.38),
  delta_s = 4 mu'' / (rho w^2 a^2) (14.39). P0 = D rho_w g (14.40). rho0 = (1 + P0/P_A) rho0|_{D=0}, P_A about
  101.3 kPa (14.41). Solve by iteration starting from A = 1 (p. 394). Shear raises the resonance frequency
  appreciably (Fig. 14.6). Narrow-resonance approximation: sigma_v = pi a^3 N(a) / (2 delta) (14.42), at the
  resonant radius a = (1/w0) sqrt(3 gamma P0 / (A rho) + 4 mu'/rho) (14.43) (Boyle and Chotiros 1995a).
  Table 14.2 (p. 395, methane in very soft sediment): S_p 2256 J/(kg degC), gamma 1.31, C0 0.034 J/(s m degC),
  rho0(D=0) 0.717 kg/m^3, sediment density 1250 kg/m^3, mu' 10^6 Pa, mu'' 10^5 Pa, c_w 1500 m/s, nu_p 0.991;
  hydrostatic pressure 2.5 atm (about 25 m). For a = 1 mm the geometric level 20 log10(a/2) = -66.0 dB re m^2
  is printed (p. 396; reproduced: -66.02). Gas volume fractions of order 1e-5 could explain observed
  backscatter (Boyle and Chotiros 1995a). Resonant sizes above 10 kHz are below present resolution (p. 397).
- Shells as fluid spheres (pp. 398-401): Rayleigh backscatter sigma_r = (k_p')^4 a^6 [(e - 1)/(3e) + (h - 1)/(2h + 1)]^2
  (14.44), with e = K_d / K_b' (14.45), h = rho_g / rho (14.46), and K_d taken as lambda + 2 mu. Full-range
  approximation (Ivakin 2004): sigma_d = sigma_r / (1 + sigma_r / (V_d^2 a^2 / 4)) (14.47),
  V_d = (sqrt(e h) - 1) / (sqrt(e h) + 1) (14.48). The high-frequency limit is |V_d|^2 a^2 / 4. Fig. 14.10 uses
  e = 15.7, h = 1.4, so the limit is 0.420 a^2/4, our arithmetic, consistent with the figure's asymptote.
  SAX99 (Fig. 14.12, 35 deg): the shell model is below the data at low frequency, where roughness accounts
  for it. Roughness plus shells agree over the full range (Ivakin 2005). Size distribution conversion: the
  dimensionless distribution x 3/(4 pi a^4) gives N(a) (p. 401). Lyons 2005 (Panama City): shells
  negligible against roughness at 20 kHz. Comparable above the critical angle at 100 kHz. Multiple
  scattering matters above 1 % shell by volume.

**Elastic volume (Sect. 14.2, pp. 401-406; App. M.2).** Bistatic cross section (14.49):

    sigma = -(pi k_w^4 a_rho^2 / 2) Im( sum_{eta, alpha, eta', beta} d_{eta alpha} d*_{eta' beta}
            W_ab((Delta k_eta + Delta k*_eta')/2) / (Delta k_eta,z - Delta k*_eta',z) )

with Delta k_eta = k+_q(K_s) - k-_q'(K_i) (14.50) for eta = qq' over pp, pt, tp, tt; d = w_eta D_eta,alpha (14.51);
w_qq' = V_wq(K_s) V_wq'(K_i) a_q^-1 a_q'^-1 (14.52); D = (D_rho | D_p | D_t) (14.53) with D_rho = D'_rho + D_t/2,
D'_rho = (1 - b_pp, b_pv, -b_vp, b_vv)^T (14.54), D_p = (2, 0, 0, 0)^T (14.55),
D_t = 2 (2 g^2 b_pp^2 - 2 g^2, -2 g b_pt b_pv, 2 g b_tp b_vp, -b_vt b_tv - b_vv b_tt)^T (14.56), g = a_t / a_p,
b_qq' = e+_q(K_s) . e-_q'(K_i) (14.57). Spectra for density, compressional and shear speed fluctuations
(alpha, beta = rho, p, t) are evaluated at complex arguments. Only vertically polarised shear converts to
water at first order. Statements (p. 404): shear has a negligible effect on volume scattering in sand and
matters in rock. Fig. 14.14 (p. 405, rock at 30 kHz): a_rho 2.5, nu_p 2.5, delta_p 0.02, nu_t 1.3, delta_t 0.1,
w2 0.000141 m^0.75, gamma2 3.25, w3_rr = w3_pp = w3_tt = 6e-6, gamma3 3.0, cross-spectra zero, shear
critical angle 39.7 deg. Appreciable volume scattering is predicted despite the low w3. Fig. 14.15:
"volume scattering is predicted to dominate roughness scattering near the specular direction" for rock.
Untested; laboratory tests suggested (p. 406).

**Layered and graded fluid sediments (App. M.1, pp. 547-554).**
- General first-order T-matrix (M.19) from Ivakin's unified approach: kernels D1 = k_p^2(z) v(K_s, z) v(K_i, z)
  (M.21) and D2 = K_s.K_i v v + dv/dz dv/dz (M.22). v(K, z) is the depth dependence of the field for a unit
  plane wave (v(K, 0) = 1 + V_ww(K), (M.13)), which can be computed e.g. by OASES OASR. Stratified cross
  section (M.23), a four-fold integral, with depth-dependent covariances (M.24)-(M.26).
- Unstratified: (M.27)-(M.35). Neglecting the finite-depth windowing, it equals (14.8) with (14.10) (p. 552).
  Windowing matters near specular (Jones and Jackson 2000/2001). It was unimportant in one bistatic
  comparison away from specular (Briggs et al. 2002a).
- Homogeneous layers (M.36)-(M.47): in layer n, v = sum over alpha = 1, 2 of a_alpha(K) exp(-i nu_alpha (z - z_(n-1))) with
  nu_1 = k_p beta_p, nu_2 = -k_p beta_p (M.38)-(M.40). Layer contribution (M.45):

      sigma_n = (pi / (2 |a_rho_n|^2)) sum_{alpha, beta, alpha', beta'} [(1 - exp(-p d_n)) / p] a_alpha(K_s) a_beta(K_i) a*_alpha'(K_s) a*_beta'(K_i)
                x { |k_p|^4 W_kk(Delta k_p) + (k_ps^a . k_pi^b)(k_ps^a' . k_pi^b')* W_rr(Delta k_p)
                    + [(k_ps^a . k_pi^b) k_p*^2 + (k_ps^a' . k_pi^b')* k_p^2] W_rk(Delta k_p) }
      p = -i (k_psz^(a) - k_piz^(b) - k_psz^(a')* + k_piz^(b')*)                                             (M.46)
      Delta k_p = K_s - K_i + (1/2) e_z (k_psz^(a) - k_piz^(b) + k_psz^(a')* - k_piz^(b')*)  (complex)         (M.47)

  The total is the sum over layers. The deepest layer is semi-infinite with a_2 = 0 (p. 553). The spectra
  are evaluated at complex arguments (simplest: insert complex arguments in the form (7.14)).
- Volume treatment of roughness (M.1.2): gamma_rho = -(rho0'/rho0) f(R) (M.48), gamma_kappa = -(kappa0'/kappa0) f(R)
  (M.49). For a homogeneous sediment, steps give delta functions (M.50)-(M.51), and the result reproduces
  (K.3) with A_ww of (13.21)-(13.25) (M.52). This is the formal basis for cases where roughness and volume
  cannot be separated (relief lifting the strata).

### 2.10 Mathematical accuracy (Sect. 13.7, pp. 369-376). Confirmed. These numbers bound model error.

Definitions: h^2 = B(0); slope covariance (13.84)-(13.85); for a Gaussian covariance B = h^2 exp(-R^2/L^2)
(13.86); general L = sqrt(2) h / sqrt(-d2B/dR2|0) (13.88) = sqrt(2) h / sigma_s (13.89). This is undefined for
2-D power laws with gamma2 < 4. Multiscale: h^2 = int W d2K (13.94); total mean-square slope = int K^2 W d2K
(13.95); 1-D forms (13.96)-(13.97). Pierson-Moskowitz (13.92): W1 = alpha / (4 |K_x|^3) exp(-beta g^2 / (K_x^2 U^4)),
alpha = 8.1e-3, beta = 0.74, g = 9.81 m/s^2, U the wind at 19.5 m. Von Karman 1-D (13.93):
W1 = w1 / [K_x^2 + L_c^-2]^(gamma1/2), with gamma2 = gamma1 + 1.

| Approximation | Accuracy statement (source as cited by the book) | Page |
|---|---|---|
| First-order perturbation, Gaussian covariance, pressure release, 1-D, all bistatic angles | within 1 dB for k_w h < about 0.6. The left side of the region is limited to moderately small slopes, sigma_s = sqrt(2) h / L. Also inaccurate at large k_w L, where the Gaussian spectrum is tiny at the Bragg wavenumber (Thorsos and Jackson 1989; Fig. 13.20 after Kaczkowski and Thorsos 1994) | 371-372 |
| Fourth-order perturbation sigma(4) (K.13) | extends the 1 dB region (Fig. 13.20) | 372 |
| Kirchhoff, nonpenetrable Gaussian | within 1 dB for k_w L > 6 (13.90) AND th_i, th_s > 2 sigma_s (13.91). Below 2 sigma_s, shadowing and multiple scattering spoil it. For k_w L < 6 it can still be accurate near specular. Robust for near-forward scattering, which includes near-vertical backscatter. Accuracy depends on correlation length, not curvature (Chen and Fung 1988, Thorsos 1988) | 372-373 |
| Gap | a region of k_w h - k_w L where neither lowest-order perturbation nor Kirchhoff is accurate. Higher-order small-slope fills it (Broschat and Thorsos 1997). It is "not evident that the lowest-order small-slope approximation ... dramatically outperforms" PT or Kirchhoff in the gap | 373 |
| Thorsos 1990, PM sea surface, pressure release, 200 Hz, U 20 m/s, k_w h = 1.79 | Kirchhoff accurate near specular. Worse at incident grazing <= 15 deg, poor at wide angles including backscatter. PT accurate for bistatic angles not too near specular | 375 |
| Thorsos 1990 composite, sea surface backscatter at 200 kHz, K_c = k_w/2 and k_w, U 10 and 20 m/s (k_w h 0.45 and 1.79; large-scale rms slopes 0.072 to 0.117) | at 15 and 20 deg slope averaging adds 1 to 3 dB and brings PT to within 0.5 dB. At 10 deg and 20 m/s averaging adds 5-6 dB where 3 dB was needed. Shadowing reduced the excess by only 0.5 dB | 375 |
| Berman 1991, PM, pressure release, 150 Hz, k_w h = 1.34 | Kirchhoff good near specular. PT accurate over almost all scattering directions | 375 |
| Berman 1991, elastic (rock), von Karman L_c 50 m, gamma1 4, k_w h = 2.1, incident 57 deg | Kirchhoff within about 2 dB for scattered grazing > 30 deg. Second-order small-slope accurate only within about 20 deg of specular | 375 |
| Thorsos et al. 2000a, fluid sand, von Karman L_c 0.227 m, gamma1 2, 20 kHz, k_w h = 1.0, incident 20 deg | PT within 1 dB over the entire range of bistatic angle (Fig. 15.12) | 375 |
| Lowest-order small-slope, multiscale | "not especially clear". It is reasonable to suppose it is accurate where PT and Kirchhoff are | 376 |
| Measurements | the best careful measurement is unlikely to be better than about 2 dB; larger errors are expected in most cases (p. 326) | 326-327 |

Table 13.2 (p. 374) lists these four test configurations (reference, boundary, frequency, spectrum, gamma1,
k_w h) as above.

### 2.11 Chapter 12: experiments and empirical trends. Confirmed.

- Grazing-angle dependence (Figs. 12.2-12.5, pp. 324-326). Lambert's law was fitted to each broad grouping
  for grazing below 60 deg. **Printed fits 10 log10 mu: mud -19.7 dB, sand -20.2 dB, gravel -11.7 dB, rock
  -5.6 dB.** The grouping is mud M_z >= 5 phi and gravel M_z <= 0 phi. The sand bound is printed as
  "5 phi < M_z <= 0 phi" (inverted as printed; read as 0 < M_z < 5 phi). Mud and sand follow Lambert well
  below about 60 deg and rise steeply towards specular. Spread for mud and sand is about +-5-8 dB, so the
  sediment name is "not a good predictor of scattering strength". Mud and sand average about the same.
  Gravel and rock data are very sparse but strong. The TR 9407 rock data exceed the Lambert limit (p. 324).
- Data sets in those figures (frequencies as labelled). Mud: D. Jackson and Briggs 1992 (20, 40 kHz),
  Jones and Jackson 1997 (40), Gensane 1989 (frequency average), McKinney and Anderson 1964 (100),
  D. Jackson et al. 1996a (40), D. Jackson et al. 1986b PS1 (25) and North Sea (35), Urick 1954 S6/S5 (60),
  Pouliquen and Lyons 2002 (140, two sites). Sand: Pouliquen and Lyons 2002 (140), Stanic et al. 1988 (40, two),
  Stanic et al. 1989 (20), D. Jackson et al. 1996a (40), D. Jackson and Briggs 1992 (40), Williams et al.
  2002b (40), Boehme et al. 1985 (30), Boehme and Chotiros 1988 (30), Gensane 1989, Stanic et al. 1998 (40),
  D. Jackson et al. 1986b (20). Gravel: McKinney and Anderson 1964 (100), D. Jackson et al. 1986b English
  Channel (25), Gensane 1989. Rock: McKinney and Anderson 1964 (100), Urick 1954 (55), APL-UW TR 9407 (30).
- Frequency dependence (Figs. 12.6-12.9, pp. 327-329): mud about +3 dB per decade on average; sand 6-8 dB per
  decade (line drawn 7.5); gravel line 3 dB/decade; rock line 0 dB/decade. Gravel and rock are too sparse to
  conclude anything.
- Low-angle data can be contaminated by multipath and surface scatter (p. 323). Near-vertical
  measurements need care (App. G.2.2) (p. 327).
- No backscattering-strength values are printed as numbers in Ch. 12-14 except the Lambert fits above. All
  measured curves are figures only.

### 2.12 Relative size of roughness and volume scattering. Confirmed.

- p. 331: roughness peaks near vertical and near the critical angle (about 25 deg in Fig. 13.1) and falls
  above it. Volume rises above the critical angle. In soft sediments (mud; nu_p < 1, no critical angle),
  volume "tends to dominate roughness scattering except for angles near vertical". In coarse sand one
  expects roughness to dominate, "although this may not always be the case". For gravel, or for layered
  sediments with undulating strata, the separation may be neither possible nor desirable.
- Fig. 13.1 (p. 332), sand-like at 30 kHz (a_rho 2, nu_p 1.15, delta_p 0.016, gamma2 3, w2 0.0001 m, gamma3 3,
  w3 0.0004): roughness (small-slope) is above volume (perturbation) at all grazing angles shown.
- p. 341: mud roughness scattering is lower because of the reduced contrast. The deficit "is often made up
  by increased volume scattering".
- SAX99 (p. 353): roughness should dominate below 100 kHz. At 300 kHz the excess is attributed to volume
  scattering of unknown type, and shells explain part of it (Fig. 14.12).
- Lyons 2005 (Panama City, p. 401): shell volume negligible at 20 kHz, comparable to roughness above the
  critical angle at 100 kHz.

### 2.13 Frequency range of validity. Confirmed.

- Book subject: "very roughly, frequencies from 10 kHz to 1 MHz", wavelengths 15 cm to 1.5 mm. Only the top
  few metres (low end) to few centimetres (high end) matter (p. 1).
- Fluid volume model with depth-constant properties: "may be a poor approximation at steep grazing angles
  or at frequencies below roughly 10 kHz, where deeper penetration may expose sediment layers". It may also
  be poor above 100 kHz because of strong gradients in the first few millimetres (Lyons and Orsi 1998,
  Pouliquen and Lyons 2002, Tang et al. 2002) (p. 378).
- Roughness model: slopes match data to about 100 kHz. It fails at two sites above 100 kHz (p. 351). At
  SAX99 the Biot roughness model holds to 100 kHz, while 300 kHz needs another mechanism (p. 353).
- Biot comparison, Fig. 13.6: 0.4 and 2 kHz are "most likely below the band of applicability", because
  penetration exposes layering and gradients (p. 347).

### 2.14 Research Issues (summaries)

- Ch. 12 (p. 330): future experiments need detailed geoacoustic characterisation (roughness, discrete
  scatterers, gradients, heterogeneity). Coverage in angle and frequency has been limited. Data sets must be
  concurrent in space and time. There are too few data for mud and rock. Bistatic geometry is uncertain.
  Laboratory experiments are useful, but scaling them is a problem.
- Ch. 13 (p. 376): single-scattering approximations fail for some roughness conditions. Higher-order
  small-slope may help if tractable. Rough rocky seafloors lack theory and characterisation. Discrete
  objects such as shells need study.
- Ch. 14 (p. 406): tests are unconvincing because heterogeneity is poorly characterised. Continuous models
  do not cover discrete scatterers. Work is needed on the "gray area" between interface and volume
  scattering (worm tubes, shell deposits on the interface, thin layers with rough boundaries).

---

## 3. Against our simulator

### 3.1 Item by item

| Book item | Our code | Verdict | How checked |
|---|---|---|---|
| Perturbation fluid (13.9), (13.21)-(13.25) | `perturbation_strength`, `_perturbation_factor` (D-63) | **agrees**. Max difference < 0.001 dB, 5-85 deg, at medium sand, very fine silt and sandy gravel at 30 and 100 kHz, plus 7 bistatic geometries | script: book A_ww retyped, |.|^2 k^4 W |
| Kirchhoff (13.10), (13.12)-(13.17) | `kirchhoff_strength` ("ours" prefactor, D-66) | **agrees** (< 0.001 dB). Algebra: ours (q^2/q_z)^2 I with I = I_K / Delta K^2 equals (Delta k^2 / (Delta K Delta k_z))^2 I_K. V at the angle of (13.13) | script; I_K by independent real-axis quadrature: ours `kirchhoff_integral` equals it to 2e-7 relative for b = 1e-4 to 1, 7e-7 at b = 10 |
| Small-slope (13.11) | `small_slope_strength` ("ours", default since D-86) | **agrees** (< 0.001 dB) | script |
| Structure constant (D.11)-(D.13) | `structure_constant` | **agrees** (identical expression; 1.5757e-3 both) | algebra + number |
| Cut-off (13.72)-(13.75) | `large_scale_slope`, `small_scale_kh = 0.5` | **agrees**. The book attributes the rule to D. Jackson et al. 1986a. Our docstring credits Fonseca 2025 / Novarini-Caruthers 1994 | K_c identical in all six cases (e.g. 20.93 rad/m, medium sand 30 kHz) |
| Large-scale slope (13.76)-(13.77) | `large_scale_slope` (per-axis s^2) | **agrees** (per axis. Novarini-Caruthers use twice this, p. 367). Reproduces the book's printed slopes 0.0227, 0.0595, 0.1432 (Fig. 13.19) | hand computation |
| Composite average (13.80) | `composite_strength`: 2-D Gaussian average over (u, v), local angles with atan, shadowed facets dropped, small-scale spectrum zero below K_c | **differs**: no Wagner factor R (13.81)-(13.83). Ours is 2-D (for backscatter the cross-slope component has no effect) and uses atan for theta_L. Without R the two agree to within 0.3 dB below 30 deg | table 3.4 |
| Composite validity (13.78), theta < 70 deg, K_c < Bragg | not checked in code | **absent**: medium sand at 100 kHz has sigma_sL 0.157 and sandy gravel 0.178 / 0.367 at 30 / 100 kHz, all beyond 0.1, where the book calls corrections "suspect" | script |
| "combined" (small-slope averaged over slopes) | `combined_strength` | **ours, no book counterpart**. The book recommends small-slope as the all-angle expression and notes it lacks tilting/shadowing (p. 368) | - |
| Interpolation Kirchhoff/composite | none (D-66 chose small-slope instead) | **book prints none**. Our D-66 choice is the book's own recommendation (p. 336) | - |
| Volume (14.8), (14.9), (8.26) | `volume_strength` | **agrees** (< 0.001 dB, all cases) | script |
| Volume slope/shadow correction (Sect. 14.1.2) | none | **absent** (the book prints only the recipe: average via (13.80)) | - |
| Perturbation volume from spectra (14.10), (14.26)-(14.28) | none. sigma2 is a constant | **absent**. Gives frequency dependence (gamma3) and a bistatic pattern | Sec. 3.6 |
| Layered / graded (13.26), (M.36)-(M.47) | none (sonar-3k1 plans arXiv:2009.12420) | **absent**. (13.26) needs only a layered V, see WP-5 | - |
| Elastic roughness (13.27)-(13.34); elastic volume (14.49)-(14.57) | none. Rock is Lambert (D-88) | **absent** | - |
| Poroelastic (13.35)-(13.60) | none. Effective-density fluid is the book's simpler alternative (Fig. 13.6) | **absent** | - |
| High-frequency Kirchhoff (13.62)-(13.71) | none (full Kirchhoff instead) | absent; not needed (the book shows it lower than full Kirchhoff, Fig. 13.19) | - |
| Coherent reflection (13.14) | `reverb_specular` / time spreading (other packages) | not compared here | - |
| Lambert bound (2.23) | `LAMBERT_MAX_MU_DB` = -4.97 dB | **agrees** (the book is a third print of mu < 1/pi) | - |
| sin^2 small-angle limit (p. 27) | our models vanish at grazing | agrees qualitatively | - |
| Reciprocity (2.18) | tests in `test_bottom_models.py` | agrees (both A_ww forms are symmetric) | - |
| `medium_sand` row | rho 1.845, nu 1.178, delta 0.016189, w2 1.41e-4, gamma2 3.25 | **confirmed** by Fig. 13.2's "typical of a medium sand" inputs (1.845, 1.1782, 0.01624, 0.000141 m^0.75, 3.25). The book does not say they are TR 9407's. sigma2 for medium sand is not printed (D-90 unsettled) | p. 337 |
| Rock Lambert -16 dB, pebbles -9 dB (D-88) | `configs/sediments.yaml` | **differs from book fits**: rock fit -5.6 dB (ours 10.4 dB lower); gravel fit -11.7 dB (ours pebbles 2.7 dB higher). The book's fits are to sparse, older data. The rock data include TR 9407 levels that exceed the energy bound | Sec. 3.5 |
| Bistatic roughness | `strength(..., azimuth=)` used by `reverb_bistatic` (simulator passes the true bistatic azimuth for `fluid_sediment` items; Lambert items use mu sin sin) | **agrees** with (13.9)-(13.11) at all tested geometries | Sec. 3.3 |
| Bistatic volume | isotropic sigma_v | **differs** from (14.10) | Sec. 3.6 |

### 3.2 Backscatter tables: book against ours (dB; "ours - book"; script output)

Columns: total backscattering strength with the book's small-slope + volume (14.8) model, the same with our
default (`small_slope` + `volume_strength`), and the differences per term. "comp" is the book's composite
(13.80) with the small-scale spectrum truncated at K_c and Wagner's R as printed. "no-sh" is the same
without R. The last column is the book's Kirchhoff roughness term, for orientation.

**Medium sand (Fig. 13.2 inputs, sigma2 0.001 as our row), 30 kHz.** K_c 20.93 rad/m, sigma_sL 0.0760 (both).

| grazing | book SSA+vol | ours default | d pert | d Kirch | d SSA | d vol | book comp | ours composite - book comp | ours composite - book no-sh | ours combined - book comp | book Kirch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | -47.48 | -47.48 | 0.000 | 0.000 | 0.000 | 0.000 | -42.46 | +0.88 | -0.03 | +0.89 | -35.11 |
| 10 | -36.97 | -36.97 | 0.000 | 0.000 | 0.000 | 0.000 | -35.88 | +0.05 | -0.01 | +0.07 | -34.94 |
| 20 | -28.85 | -28.85 | 0.000 | 0.000 | 0.000 | 0.000 | -29.36 | 0.00 | 0.00 | +0.05 | -34.25 |
| 30 | -25.43 | -25.43 | 0.000 | 0.000 | 0.000 | 0.000 | -27.13 | +0.01 | +0.01 | +0.10 | -33.04 |
| 40 | -25.15 | -25.15 | 0.000 | 0.000 | 0.000 | 0.000 | -27.94 | -0.01 | -0.01 | +0.17 | -31.23 |
| 60 | -22.25 | -22.25 | 0.000 | 0.000 | 0.000 | 0.000 | -23.91 | -0.01 | -0.01 | +0.48 | -24.93 |
| 70 | -18.29 | -18.29 | 0.000 | 0.000 | 0.000 | 0.000 | -17.90 | +0.01 | +0.01 | +0.38 | -19.38 |
| 80 | -10.45 | -10.45 | 0.000 | 0.000 | 0.000 | 0.000 | -8.63 | -2.37 | -2.37 | -0.26 | -10.66 |
| 85 | -5.57 | -5.57 | 0.000 | 0.000 | 0.000 | 0.000 | -8.62 | -1.09 | -1.09 | +3.04 | -5.64 |

**Medium sand, 100 kHz.** K_c 143.7 rad/m, sigma_sL 0.157 (> 0.1: the book's composite is outside its
condition (13.78)).

| grazing | book SSA+vol | ours default | d (all terms) | book comp | ours composite - book comp | ours composite - book no-sh | ours combined - book comp | book Kirch |
|---|---|---|---|---|---|---|---|---|
| 5 | -43.84 | -43.84 | < 0.001 | -35.85 | +2.96 | -0.11 | +3.04 | -31.18 |
| 10 | -33.33 | -33.33 | < 0.001 | -30.64 | +0.94 | -0.02 | +1.04 | -31.00 |
| 15 | -28.20 | -28.20 | < 0.001 | -27.56 | +0.25 | -0.02 | +0.38 | -30.70 |
| 20 | -25.19 | -25.19 | < 0.001 | -25.64 | +0.08 | +0.02 | +0.24 | -30.27 |
| 30 | -21.91 | -21.91 | < 0.001 | -23.91 | -0.10 | -0.10 | +0.15 | -29.00 |
| 40 | -22.57 | -22.57 | < 0.001 | -23.48 | -0.07 | -0.07 | +0.33 | -27.10 |
| 60 | -19.12 | -19.12 | < 0.001 | -17.94 | +0.47 | +0.47 | -0.33 | -20.81 |
| 70 | -15.46 | -15.46 | < 0.001 | -14.80 | -1.53 | -1.53 | +0.31 | -16.20 |
| 85 | -9.94 | -9.94 | < 0.001 | -14.50 | -7.02 | -7.02 | +3.86 | -10.07 |

**Very fine silt (our row: rho 1.147, nu 0.984, delta 0.00189, w2 1.638e-5, sigma2 0.000986), 30 kHz** (K_c 3.74,
sigma_sL 0.0136) **and 100 kHz** (K_c 25.7, sigma_sL 0.0280). All terms differ by < 0.001 dB. Ours composite
and book composite (with or without R) agree within 0.01 dB below 80 deg (slopes are tiny). Combined minus
book composite is +0.0 to +0.6 dB at 5-80 deg. Totals (book = ours):

| grazing | 30 kHz total | 30 kHz book Kirch rough | 100 kHz total | 100 kHz book Kirch rough |
|---|---|---|---|---|
| 5 | -42.03 | -60.19 | -42.01 | -56.27 |
| 10 | -36.08 | -60.03 | -36.06 | -56.11 |
| 20 | -31.95 | -59.37 | -31.94 | -55.44 |
| 30 | -30.05 | -58.21 | -30.04 | -54.27 |
| 40 | -28.86 | -56.47 | -28.85 | -52.51 |
| 60 | -27.49 | -50.40 | -27.45 | -46.39 |
| 80 | -26.34 | -35.25 | -25.51 | -31.05 |
| 85 | -22.98 | -25.25 | -20.33 | -21.42 |

The book composite roughness term for this silt is -67.6 dB (30 kHz) and -63.5 dB (100 kHz) at 5 deg. Even
the Kirchhoff term, which overestimates at these angles, is 18 to 28 dB below the total at 5-60 deg. The
volume term therefore dominates by about 20 dB or more away from vertical, and by only about 9 dB at 80 deg. This matches the book's statement for soft
sediments (Sec. 2.12) and our D-193 finding.

**Rough coarse case: sandy gravel (-1 phi; `sediment_from_grain_size(-1)`: rho 2.492, nu 1.337, delta 0.01705,
w2 4.09e-4, sigma2 0.002).** All model terms equal the book's to < 0.001 dB. Composite differences:

| grazing | 30 kHz total (book = ours) | 30 kHz ours comp - book comp | 30 kHz ours comb - book comp | 100 kHz total | 100 kHz ours comp - book comp | 100 kHz ours comb - book comp |
|---|---|---|---|---|---|---|
| 5 | -39.98 | +3.44 | +3.55 | -36.10 | +6.48 | +6.90 |
| 10 | -29.85 | +1.19 | +1.33 | -25.94 | +3.64 | +4.06 |
| 15 | -25.15 | +0.40 | +0.57 | -21.20 | +2.20 | +2.58 |
| 20 | -22.49 | +0.10 | +0.32 | -18.47 | +1.06 | +1.37 |
| 30 | -19.38 | -0.06 | +0.27 | -15.24 | +0.58 | +1.03 |
| 40 | -17.23 | -0.13 | +0.31 | -13.19 | +1.21 | +0.40 |
| 60 | -15.03 | -0.24 | -0.13 | -13.55 | -2.16 | +1.51 |
| 80 | -8.67 | +0.17 | +2.96 | -13.27 | -3.75 | +3.88 |

(sigma_sL is 0.178 at 30 kHz and 0.367 at 100 kHz, both beyond the book's condition 0.1. K_c is below the
Bragg wavenumber only below 78.7 / 66.3 deg.)

### 3.3 Bistatic geometry (medium sand, 30 kHz, theta_i = 20 deg; ours - book in dB)

| theta_s | phi_s | d pert | d SSA | d Kirchhoff |
|---|---|---|---|---|
| 20 | 180 (back) | 0.000 | 0.000 | 0.000 |
| 20 | 90 | 0.000 | 0.000 | 0.000 |
| 20 | 30 | 0.000 | 0.000 | 0.000 |
| 40 | 0 (forward) | 0.000 | 0.000 | 0.000 |
| 10 | 0 | 0.000 | 0.000 | 0.000 |
| 60 | 120 | 0.000 | 0.000 | 0.000 |
| 20 | 1 (near specular) | 0.000 | 0.000 | 0.000 |

So `reverb_bistatic`'s bottom away from backscatter (a `fluid_sediment` item: `bottom.strength` at the true
bistatic azimuth) already follows the book's bistatic roughness formulas. It does not use Lambert's ratio
and it does not reuse the backscatter value. Lambert items use mu sin(ti) sin(ts), which is (2.22). Seabed
point scatterers (`targets.SedimentScatterers`, D-64) take the model at the direct angles and Lambert's ratio
for the others: an approximation the book neither supports nor contradicts.

### 3.4 What changes if ours adopted the book's composite form

Add Wagner's R to `composite_strength` / `combined_strength`. These are opt-in models, not the default, so
no default result moves. Low-angle values fall by the "ours composite - book comp" column: -0.9 dB at 5 deg
for medium sand at 30 kHz, -3.0 dB at 5 deg and -0.9 dB at 10 deg at 100 kHz, -3.4 / -6.5 dB at 5 deg for sandy
gravel at 30 / 100 kHz. These amounts hold if Q is as printed. With pi^(-1/2) the changes are about a third
in dB (Sec. 2.6 table). Changing the 2-D average and atan to the book's 1-D linear form changes backscatter
by < 0.3 dB below 30 deg. The default `small_slope` is unaffected. Above about 60-70 deg the composite is
outside its stated range in both versions.

### 3.5 Lambert laws for hard bottoms (book Ch. 12 fits against ours; dB, same at every frequency)

| grazing | ours rock (-16) - book rock fit (-5.6) | ours pebbles (-9) - book gravel fit (-11.7) | our sandy_gravel fluid model (SSA+vol) 30 kHz - book gravel Lambert | same, 100 kHz |
|---|---|---|---|---|
| 5 | -10.4 | +2.7 | -7.09 | -3.21 |
| 10 | -10.4 | +2.7 | -2.94 | +0.96 |
| 20 | -10.4 | +2.7 | -1.47 | +2.55 |
| 30 | -10.4 | +2.7 | -1.66 | +2.48 |
| 40 | -10.4 | +2.7 | -1.69 | +2.35 |
| 60 | -10.4 | +2.7 | -2.08 | -0.60 |
| 80 | -10.4 | +2.7 | +3.16 | -1.44 |

Both Lambert pairs are frequency-independent. The book's gravel data rise about 3 dB per decade and its
rock data about 0 (Ch. 12 lines), so at 30 and 100 kHz the comparison is the same. Reading: the book's rock
fit (-5.6 dB) is 0.6 dB under the energy bound and is driven by TR 9407 data that the book itself says
exceed it. Our -16 dB (the middle of modern site fits, D-88) is not contradicted by anything measured that
the book prints as numbers. The book's gravel fit lies within about +-3 dB of our fluid model at the
grain-size defaults for -1 phi between 10 and 60 deg.

### 3.6 Volume term away from backscatter (Fig. 14.3 parameters, theta_i = theta_s = 20 deg)

(14.10) sigma_v relative to its backscatter value, against bistatic azimuth (180 = back, 0 = forward). Ours is
0 dB everywhere: sigma_v is isotropic, and the transmission factors are identical in both.

| phi_s | 180 | 150 | 120 | 90 | 60 | 30 | 10 |
|---|---|---|---|---|---|---|---|
| rho_rk = -1, mu = 1 | 0.0 | -0.1 | -0.5 | -1.2 | -2.2 | -3.4 | -4.0 |
| rho_rk = 0, mu = 1 | 0.0 | -0.1 | -0.2 | +0.7 | +3.9 | +8.5 | +10.6 |

So towards forward scatter our isotropic volume term is 4 dB too high (rho_rk = -1, the book's usual choice)
or 10.6 dB too low (uncorrelated). The roughness term usually dominates near forward specular anyway
(p. 387).

Equivalent sigma2 of two Table 14.1 sites, by (14.10) in backscatter with the printed spectra (our
computation; rho_rk = -1). Arafura (5.6 phi, 20 kHz): sigma_v -19.05 dB/m, sigma2 = 0.0031 (flat over
10-60 deg). San Francisco (6.4 phi, 40 kHz): sigma_v -12.5 dB/m, sigma2 = 0.0037. The grain-size rule
(Snellen 2013) gives 0.001 for both grain sizes. The site values are 4.9 and 5.7 dB higher, and agree with
Gallaudet's fitted 0.003 at 6.36 phi (D-193). This is evidence that the class default for muds is low at the
book's sites. It is not a decision: the book stresses the heterogeneity inputs are uncertain.

---

## 4. What this closes or changes in open issues

| Issue / row | Effect of this material |
|---|---|
| **sonar-3k1** (bottom as published, layered) | Partly closed. Confirmed: perturbation, Kirchhoff, small-slope, volume and composite cut-off/slope (Sec. 3.1), so "ours" can become "confirmed (J&R 2007 eq. ...)". Not in the book: the regime interpolation, the large-roughness gravel/rock law and the Williams-Jackson interpolation, which stay wanted (Mourad and Jackson 1989; Williams and Jackson 1998; TR 9407 remains unread). Newly available to implement from the book: Wagner shadowing (13.81)-(13.83); the volume slope average (Sect. 14.1.2 recipe); the layered-roughness B (13.26); the layered volume (M.45); the elastic roughness (13.27)-(13.34) and elastic volume (14.49)-(14.57); poroelastic (13.35)-(13.60). |
| Kirchhoff/composite interpolation and cut-off | Cut-off settled: (13.72)-(13.75) equal ours, and the book attributes them to D. Jackson et al. 1986a. Interpolation not printed. The book recommends the small-slope approximation in its place (p. 336), which supports D-66/D-86. |
| Large-roughness gravel/rock law | Not in the book. Lambert fits are printed instead: gravel -11.7, rock -5.6 dB (Ch. 12). D-88's -16 dB for rock stays defensible (Sec. 3.5). The owner may want the book's fits added to `rock`/`pebbles` `basis` text as a further reference range. |
| Bistatic model | Our bistatic roughness equals the book's formulas (Sec. 3.3). Volume is isotropic in ours. The book's (14.10) gives the pattern (Sec. 3.6). The Williams-Jackson interpolation itself is not printed. |
| Elastic and layered bottoms | All formulas needed are printed (Sec. 2.3, 2.9). |
| Medium sand sigma2 0.001 against 0.002 (D-90) | Not settled: the book prints no sigma2 for medium sand. Its typical range is 0.0001-0.004, and its examples use 0.002 (Fig. 14.2). Its roughness inputs for medium sand equal our row exactly (Fig. 13.2). |
| Sediment fit tool with perturbation (STATUS item 30, D-86) | Supported: perturbation and small-slope overlap below about 70 deg for medium sand (Fig. 13.2). The book's model-data comparisons used perturbation (Figs. 13.7, 13.9). |
| Grain-size regressions | Not in these chapters (Ch. 5 / TR 9407, other agents' parts). Table 13.1 and Table 14.1 give eight + five measured site rows with M_z, usable as checks of the regression (not done here). |
| Lambert parameters | Book fits mud -19.7, sand -20.2, gravel -11.7, rock -5.6 dB (Ch. 12), a further print of the energy bound (2.23), and the sin^2 low-angle argument (p. 27). Our default -29 dB (D-89) is a deep-water low-frequency value. The book's mud/sand fits (-20 dB) say that at 10-140 kHz a typical shelf mud or sand is near -20 dB, i.e. 9 dB above our bare default. That default already warns at 10 kHz and above. |
| **sonar-9le** (high-frequency surface) | The book is about the SEAFLOOR. It gives no sea-surface model, bubble layer or wind law. Relevant only indirectly: the Pierson-Moskowitz spectrum (13.92) with constants, and the accuracy studies of Thorsos 1990 and Berman 1991 on PM surfaces (k_w h 1.79 and 1.34; composite corrections at 200 kHz; Sec. 2.10), which bound a perturbation/composite surface model. |
| **sonar-6fc** (K texture) | Ch. 16 (not my part). Nothing in Ch. 12-14 except the statement that sigma is an ensemble average and ping fluctuations are separate (p. 23). |
| **sonar-sul** (real data) | Test cases a validation could reproduce: Table 13.1 sites (roughness, 35-40 kHz, data sources named), Table 14.1 sites (volume, 20-140 kHz), SAX99 (Figs. 13.10-13.11, 14.12), the near-vertical Falmouth/Puget Sound (Fig. 13.15). The measured values are figure-only in the book: the original papers are needed for numbers. Measurement accuracy is at best about 2 dB (p. 326). |
| **sonar-6t5** (ping-to-ping coherence) | App. J: the correlation length of the surface field L_f is about lambda / (angular width of sigma) (J.36). This is the scale that sets field correlation. No ping-to-ping model. |
| **sonar-7iy** (relief reverberation) | (J.15) with per-patch angles is the book's justification for summing sigma dA over a relief with local angles. Composite slope averaging is the statistical counterpart. Nothing specific to bottom-bounce legs. |
| REFERENCES section 10 rows | "Interpolation ... cut-off ... empirical expression": cut-off settled; the other two stay wanted (the book named in that row does not print them). "Bistatic bottom model": formulas general, interpolation still wanted. "Medium sand volume parameter": unchanged. |
| REFERENCES 3.2 rows marked "ours" | Kirchhoff prefactor, structure constant, small-slope form: now confirmable from J&R 2007 (13.10), (D.12), (13.11). Composite recipe: structure confirmed (13.80), but ours lacks (13.81). "combined" stays ours. |

---

## 5. Work packages for agents (ordered by value to realism per effort)

**WP-1 Pin the book's equalities as tests and relabel provenance (S).**
Goal: what this note checked numerically becomes tests, and "ours" becomes "confirmed" where the book prints it.
Formulas: (13.21)-(13.25), (13.10)/(13.16)-(13.17), (13.11), (D.12), (13.75), (13.77), (14.8)/(14.9)/(8.26).
Tests first (`tests/python/test_bottom_models.py`): (a) retyped book A_ww against `_perturbation_factor` at the
Fig. 13.2 inputs, backscatter 5-85 deg and three bistatic geometries, to 1e-6 relative; (b) `large_scale_slope`
at 20 kHz, gamma2 3.25, w2 = 0.0000422 / 0.0001406 / 0.0004218 gives 0.0227 / 0.0595 / 0.1432 (printed, p. 368);
(c) sigma ~ f^(4 - gamma2) of `perturbation_strength` with frequency-independent nu, delta (13.61); (d) the
critical angles of Fig. 14.2 (17.8 / 24.6 deg) and the intromission angle of Fig. 13.4 (10.5 deg, a_rho 1.5,
nu 0.98) as zero / extremum checks of `reflection`. Files: tests, `sim/bottom.py` docstring,
`docs/REFERENCES.md` 3.2 (proposed text to the orchestrator), `configs/sediments.yaml` `medium_sand`
provenance (add "J&R 2007 Fig. 13.2"). Default results: none move. Must NOT change any formula.

**WP-2 Wagner shadowing and the book's composite form, opt-in (S/M).**
Goal: `composite_strength`/`combined_strength` take `shadowing="none" | "wagner"` (default "none") and optionally
`slope_average="book_1d"` ((13.79)-(13.80)). Formulas (13.80)-(13.83). First settle the pi^(+-1/2) in Q
against Wagner (1967) or another print. Add a validity warning when sigma_sL > 0.1 or theta > 70 deg (13.78).
Tests: R goes to 1 for large t and to 0 as theta goes to 0; R monotonic; with "none" the outputs are
bit-identical to today; the composite with R against a direct implementation of (13.80) at Fig. 13.2 inputs.
Prepare before -> after for the owner (opt-in, so not required). Roughness term of `composite` at 5 deg, from
Sec. 3.2; the "after" is the book's form, which also differs by -0.03 to -0.29 dB through the 1-D average:
medium sand 30 kHz about -41.6 -> -42.5 dB; 100 kHz about -32.9 -> -35.9 dB; sandy gravel 100 kHz about
-21.4 -> -27.8 dB.
Must NOT make "combined" or "composite" the default.

**WP-3 Volume term with large-scale slope averaging and shadowing, opt-in (M).**
Goal: the "slope and shadowing corrections of the volume term" of sonar-3k1. Recipe (Sect. 14.1.2, p. 383):
average `volume_strength` over the large-scale slope pdf with (13.80) (and R from WP-2). Use the K_c of
(13.75) or a user slope. Files: `sim/bottom.py` (`volume_slope_average=False`). Tests: zero slope gives
exactly today's value; the average of a sin-independent constant equals the constant times the shadow
factor; reciprocity in bistatic use. Before -> after numbers to prepare: medium sand and very fine silt at
30/100 kHz, 5-30 deg. Depends on WP-2. Must NOT change defaults without the owner.

**WP-4 Perturbation volume scattering from heterogeneity spectra, opt-in (M).**
Goal: `volume_model="spectral"` with (w3, gamma3, mu, rho_rk) instead of sigma2, giving the frequency
dependence and the bistatic dipole/monopole pattern. Formulas (14.10), (14.13)-(14.19), (14.26)-(14.28),
inserted into (14.8). Tests: backscatter closed form (pi |k_p|^4 / 2)(mu^2 - 2 rho_rk mu + 1) W(|Delta k_p|);
gamma3 = 3 gives near frequency independence (p. 387); the bistatic ratios of Sec. 3.6 (Fig. 14.3 inputs) as
regression; equivalent sigma2 at Arafura 20 kHz = 0.0031 and San Francisco 40 kHz = 0.0037 (our numbers,
labelled as such). Files: `sim/bottom.py`, `configs/sediments.yaml` (optional site rows from Table 14.1 with
`status: confirmed (J&R 2007 Table 14.1)`). Also the elastic volume (14.49)-(14.57) later (L).
Must NOT replace sigma2 as default.

**WP-5 Layered or graded seafloor: roughness via (13.26), volume via (M.45) (M/L).**
Goal: sonar-3k1's layered seafloor, most relevant to the 8 kHz flank array (below the book's range, where
layering is exposed, p. 378). Roughness: replace B by (13.26) with the reflection coefficient of a stack of
fluid layers (any standard recursion, or arXiv:2009.12420). Volume: (M.36)-(M.47) per layer. Tests: one
half-space layer reproduces today's numbers exactly ((13.26) reduces to (13.23), Sec. 2.3); a layer of the
same material as the half-space changes nothing; energy check of (2.20) for the layered V. Files:
`sim/bottom.py` (new `layers=` input), `configs/sediments.yaml` (optional). Depends on a layered reflection
routine (`propagation` may have one). Must NOT treat an elastic top layer as the limit of a vanishing fluid
layer (p. 341).

**WP-6 Elastic rough-interface perturbation for rock, opt-in (M).**
Goal: a physical rock model in place of Lambert where wanted. Formulas (13.27)-(13.34) with the elastic V
(9.34, read from Ch. 9) in the small-slope form (13.11). Tests: shear speed to 0 equals the fluid A_ww;
reciprocity (swap ti, ts); (K.5)/(L.9) identity A(K_i, K_i) = -2 beta^2 V; a dip near the Rayleigh angle
for rock (Fig. 13.5 description). Inputs: rock parameters from Sect. 5.6 or `rock` reflection data plus a
shear speed (owner/agent reading Ch. 5/9). The default for rock (Lambert -16 dB, D-88) moves only with the
owner. Prepare before -> after at 30/100 kHz, 5-60 deg.

**WP-7 Rock and gravel Lambert references (S, documentation).**
Goal: add the book's fits (gravel -11.7, rock -5.6 dB, Ch. 12 pp. 324-326) to the `basis` text and to
REFERENCES as a further print, with its caveats (sparse old data; TR 9407 rock above the bound). Default
unchanged unless the owner decides (before -> after for rock would be +10.4 dB at every angle; for
pebbles -2.7 dB).

**WP-8 Discrete volume scatterers: shells and bubbles, opt-in (M).**
Formulas (14.30)-(14.48), Table 14.2. Tests: fluid sphere high-frequency limit |V_d|^2 a^2 / 4 with e = 15.7,
h = 1.4 gives 0.420 (Fig. 14.10); Rayleigh a^6 k^4 at small ka; bubble resonance iteration converges, and
the shear term raises w0 (Fig. 14.6). Geometric level -66.0 dB re m^2 for a = 1 mm (p. 396). Use in (14.8)
via sigma_v. Not a default.

**WP-9 Bistatic and energy tests of the bottom model (S).**
Goal: hold what Sec. 3.3 found. Tests: reciprocity (2.18) of `strength(ti, ts, phi)`; hemisphere energy
(2.20) for small_slope, with incoherent reflection near specular (L.26) |V|^2 (1 - exp(-Delta k_z^2 h^2)) for a
spectrum with finite h (von Karman D.7/D.8). No code change expected.

**WP-10 Validation set from the book's test sites (M; feeds sonar-sul).**
Encode Table 13.1 (8 sites) and Table 14.1 (5 sites) as site parameter sets in a test-data file. Get the
measured backscatter as numbers from the original papers (Stanic et al. 1988/1989/1998; D. Jackson and Briggs
1992; D. Jackson et al. 1996a; Williams et al. 2002b; Briggs et al. 2002a; Jones and Jackson 1997; Pouliquen
and Lyons 2002), because the book prints them only as figures. Until then, tests may check model features
only: the presence or absence of the critical-angle maximum; the frequency slope (13.61). Never digitise the
book's figures. Expect differences of 2-3 dB from measurement error alone (p. 351).

---

## 6. Not verified / unclear on the page

- (13.82) Q is printed with pi^(+1/2) exp(-t^2). The usual Smith/Wagner shadow function has pi^(-1/2) in this
  place. This is a possible misprint or a different definition. Wagner (1967) was not opened. Sec. 2.6
  gives both.
- (L.34), p. 546, appears printed as (1/4) |V|^2 (Delta k^2 / Delta k_z^2) p_s, and (L.35) with 1/(2 pi ||B_s||). (13.62)
  and (13.63) have Delta k^4 / Delta k_z^4 and ||B_s||^(1/2). (13.67) and (13.70) are consistent with the Chapter 13
  versions (backscatter gives 1/sin^4), so the Appendix L versions look misprinted. Use Chapter 13.
- Fig. 13.18 panel labels: sigma_sL values were read as (a) 0.065, (b) 0.134, (c) 0.056, (e) 0.048 and
  (f) about 0.27. The print is small. Only (b) and (f) are quoted in the text as "large".
- Table 13.1 does not print w2 units. m^(4 - gamma2) is inferred from the SAX99 row matching Fig. 13.10.
- Table 14.1 rows with finite L_c (Orcas, Porto Venere, P. della Mariella) use a spectral form with a
  cut-off from Ch. 7 (7.14/7.16/7.17), which was not read. The equivalent sigma2 was computed only for the
  two L_c = infinity rows.
- The sand bound in Sect. 12.1.1 is printed "5 phi < M_z <= 0 phi" (inverted as printed).
- Fig. 13.1's w3 = 0.0004 is printed without units. gamma3 = 3 makes it dimensionless (p. 387).
- The elastic D1 (13.28) is long. It was retyped from the page and not evaluated numerically here. Check it
  against the page before implementing (WP-6). Likewise the Biot matrices (13.42)-(13.60).
- Fig. 13.2's inputs are "typical of a medium sand seafloor". The book does not say they come from TR 9407.
- The Kirchhoff/composite interpolation of Williams and Jackson (Fig. 13.8) and of the published APL-UW
  model is not printed anywhere in the parts read.
