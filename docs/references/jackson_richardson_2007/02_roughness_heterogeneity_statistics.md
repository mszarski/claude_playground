# Jackson and Richardson (2007): roughness, heterogeneity, backscatter statistics

D. R. Jackson and M. D. Richardson, *High-Frequency Seafloor Acoustics*, Springer (2007). Chapters 6, 7,
16, Appendix D and Appendix E. Issue sonar-gm8e. Everything below is written in our own words. Formulas
were re-typed from the page images. "confirmed" means read on the page image of this book. "ours" marks a
number or derivation computed here (the script is described in Sec. 3.0). Printed page numbers are given
first, then the PDF page.

**Read this first: what is wrong, unknown or risky**

- **Two misprints in Appendix D that matter for unit conversion** (Sec. 2.6). (D.19) prints
  `Gamma((gamma2 - 2)/2)` where the marginal integral gives `Gamma((gamma2 - 1)/2)`; checked numerically to
  1e-10. Read as printed, (D.19) gives w2 1.5 times too small at gamma2 = 3.25 (1.8 dB) and 3.0 times too
  small at 2.5. (D.23) prints `F^(gamma1 - 1)` where (6.11), (D.22) and (D.24) need `F^gamma1`. Anyone who
  converts Briggs' 1-D data (Table 6.1) must use the corrected forms. It is not known which form Briggs et
  al. (2005a) used for Fig. 6.18.
- **The number printed for Fig. 6.19's trend line does not fit the line.** Sec. 6.7 prints
  S(F0) = 7.1e-5 cm^3 at F0 = 0.05 cycle/cm. Read from the figure, the dashed line corresponds to about
  1.4 cm^3, using definition (6.11) at two points read off the figure (Sec. 2.3). Do not use 7.1e-5.
- **The book prints no regression of roughness on grain size** (Sec. 6.6, p. 193-195: "no empirical
  relationships are evident"; the dashed lines of Fig. 6.18 "are not regression fits"). So it neither
  confirms nor contradicts our grain-size rule for w2 (Snellen et al. 2013). It only provides raw data. Our
  own comparison with those data (Sec. 3.2): at the Bragg wavenumber, for sands (Mz < 3, 41 rows), our rule
  sits at the median of Briggs' data within 1.4 dB at 10-300 kHz, with a spread of -13 to +10 dB. For
  silts and clays (Mz >= 5.5, 8 rows) our rule is 1 to 7 dB BELOW the median of the data.
- **K-distribution patches: the book and D-196 disagree by a factor of 2.** The book (p. 433) says K
  reverberation arises from "a number, 2nu" of patches with exponentially distributed area. D-196 read
  Abraham and Lyons (2002) itself and took shape = n for n patches (their eq. 19) and n/2 for n
  scatterers. The book's wording matches the scatterer count, not the patch count. The primary paper
  outranks the book's one-sentence summary. D-196 stays as it is, but the discrepancy should be noted
  (REFERENCES Sec. 11).
- **The book prints no measured K shape values for any seabed.** Chapter 16 shows fits only as figures
  (Fig. 16.3, from Lyons and Abraham 1999, 80 kHz). The only fitted texture parameters printed as numbers
  are McDaniel's (1996) correlation length L = 1 m and normalised cross-section variance gamma = 1.5, fitted
  to Gensane (1989) data (p. 441). These numbers do give a physical model for the open half of sonar-6fc
  (texture correlated between cells; shape set by both cell dimensions): see WP-1.
- **Our isotropic, single-power-law roughness is the book's simplest case, not its finding.** Ripples make
  the spectrum anisotropic, with peaks (Figs. 6.8, 6.9, 6.12). Measured spectra break slope near
  1-2 cycles/cm (Fig. 6.7) and roll over beyond 10 cm wavelength (Fig. 6.16). Roughness changes over hours
  to days, and backscatter with it: up to 10 dB at 100 kHz and 20 deg (Fig. 6.14); -21 to -36 dB in a day
  after raking at 40 kHz (Fig. 6.13).
- **Not covered by this book at all: the sea surface** (sonar-9le), except where a chapter mentions it in
  passing. The book is about the seafloor.

## 1. Scope

| Part | Printed pages | PDF pages | How read |
|---|---|---|---|
| Ch. 6 Seafloor Roughness (6.1-6.9, Table 6.1) | 171-200 | 183-212 | in full, page images |
| Ch. 7 Sediment Heterogeneity (7.1-7.6, Table 7.1) | 201-243 | 213-255 | in full, page images |
| Ch. 16 Backscatter Statistics (16.1-16.4) | 421-442 | 429-450 | in full, page images |
| App. D Roughness and Volume Statistics (D.1-D.2.3) | 475-484 | 480-489 | in full, page images; (D.19), (D.23), (D.35) zoomed at 220 dpi |
| App. E Complex Representation of Signals | 485-486 | 490-491 | in full, page images |
| Ch. 14, Secs. 14.1-14.1.6 (pp. 380-390) | 380-390 | 388-398 | read for the link heterogeneity -> sigma_v -> sigma_2 only (another file covers Ch. 14) |
| App. A List of Symbols, first two pages | 443-444 | 451-452 | glanced at |
| App. C Table C.1 (coefficients of variation), App. G (ensonified area, G.2.1), App. M (volume perturbation details) | - | - | **not read**: only quoted as the chapters quote them |

The page offset is not constant (plates are inserted): PDF = printed + 12 in Ch. 6-7, + 8 in Ch. 14-16,
and + 5 in App. D-E.

## 2. What the book gives

### 2.1 Roughness: definitions and normalisation (Sec. 6.1, App. D.1); all confirmed

Conventions: the relief is z = f(R), with R = (x, y). The wavenumber K = 2 pi / lambda is in rad per unit
length (6.6). The 2-D spectrum is two-sided over (Kx, Ky). The time dependence is exp(-i w t)
(p. 247, Ch. 8, and (E.1)).

| Quantity | Book | Eq., page | Our symbol / code |
|---|---|---|---|
| mean, mean square | <f> = 0; h^2 = <f^2> ("RMS roughness" h) | (6.2), (6.3) p. 173 | - |
| covariance | B(R) = <f(R0 + R) f(R0)> | (6.4) p. 174; (D.1) | - |
| 2-D spectrum | W(K) = (2 pi)^-2 Integral B(R) exp(-i K.R) d^2R | (6.5) p. 174; (D.3) p. 475 | `bottom.py` docstring: same |
| normalisation | h^2 = Integral W(K) d^2K over all Kx, Ky; W has dimension length^4 | (6.7) p. 174; (D.4) p. 476 | same |
| power law | W(K) = w2 / K^gamma2; w2 "spectral strength", gamma2 "spectral exponent" | (6.8) p. 175; (D.9) p. 476 | `w2`, `gamma2`, `_power_law` |
| von Karman | W(K) = w2 / (K^2 + K0^2)^(gamma2/2); K0 is the "cutoff" wavenumber, 1/K0 the outer scale | (D.7) p. 476 | absent |
| its mean square | h^2 = 2 pi w2 / ((gamma2 - 2) K0^(gamma2 - 2)) | (D.8) p. 476 | absent (checked numerically, ours: equal to 1e-14) |
| structure function | S(R) = <[f(R0 + R) - f(R0)]^2> = 2[B(0) - B(R)] = 2 Integral (1 - cos K.R) W d^2K | (D.5), (D.6), (D.10) p. 476 | `structure_constant` docstring: same |
| power-law structure function | S(R) = C_h^2 R^(2 alpha); C_h^2 = 2 pi w2 Gamma(2 - alpha) 2^(-2 alpha) / (alpha (1 - alpha) Gamma(1 + alpha)); alpha = gamma2/2 - 1 (here alpha is NOT the K-distribution alpha) | (D.11)-(D.13) p. 477 | `structure_constant`: `a` = alpha |
| spatial-frequency spectrum | K = 2 pi F (F in cycles per length); Phi(Fx, Fy) = (2 pi)^2 W(2 pi Fx, 2 pi Fy) | (6.9) p. 176 | - |
| 1-D (marginal) spectrum | W1(Kx) = (1/2pi) Integral B(x, 0) exp(-i Kx x) dx = Integral W(Kx, Ky) dKy; W1 has dimension length^3 | (D.14)-(D.16) p. 477 | - |
| 1-D in spatial frequency | Phi1(F) = 2 pi W1(2 pi F); its integral over all F is h^2 | (6.10) p. 176; (D.22) p. 478 | - |
| Briggs' fit | Phi1(F) = phi1 / F^gamma1; phi1 is the "intercept" (value at F = 1), slope = -gamma1 | (6.11) p. 176 | - |
| 1-D von Karman | W1(K) = w1 / (K^2 + K0^2)^(gamma1/2); gamma2 = gamma1 + 1 | (D.17), (D.18) p. 478 | - |
| 1-D -> 2-D strength | printed: w2 = w1 Gamma(gamma2/2) / (sqrt(pi) Gamma((gamma2 - 2)/2)). **Correct (ours, checked by quadrature to 1e-10): Gamma((gamma2 - 1)/2) in the denominator** | (D.19) p. 478 | - |
| isotropic 1-D -> 2-D, general | W(K) = (1/pi) Integral_0^inf [W1(K) - W1(sqrt(K^2 - q^2))] q^-2 dq (Henyey 1991) | (D.20) p. 478 | - |
| intercept <-> 1-D strength | phi1 = w1 / (2 pi)^(gamma1 - 1); printed (D.23) Phi1 = phi1 / F^(gamma1 - 1): **misprint, must be F^gamma1** to agree with (6.11), (D.22) and (D.24) (ours, checked) | (D.23), (D.24) p. 478 | - |
| units | w2 has dimension length^(4 - gamma2). With a reference length h0, W = w2 / (h0 K)^gamma2 and w2 has length^4 (Mourad and Jackson 1993; Williams et al. 2001a). Converting cm -> m: in the h0 form divide by 1e8; in the plain form divide by 100^(4 - gamma2). w1 and phi1 carry the same units as w2 | D.1.3 p. 479 | `w2_from_cm4` (h0 = 1 cm): same |

Stated limits (confirmed). A pure power law gives infinite h^2 (the integral diverges as K -> 0), so the
spectrum must roll over at an outer scale (p. 175). That outer scale is "apparently rather large", larger
than the field of view of relief measurements (p. 175, Sec. 6.5). The RMS roughness therefore depends on
the size of the measured patch ("scale-dependent", p. 175). With gamma2 = 3, the RMS height difference
grows as the square root of the separation (p. 477). Self-similarity holds only for gamma2 = 4; otherwise
the relief is self-affine (p. 175). Gaussian relief is "largely untested" and is assumed by Kirchhoff and
small-slope models (p. 175). Isotropy holds only without directional features such as ripples (p. 175,
p. 475). Slices through a 2-D spectrum are NOT 1-D spectra (p. 183). The usual working range of gamma2 is
2 < gamma2 < 4 (p. 477).

### 2.2 Measurement and measured roughness spectra (Secs. 6.2, 6.3, Table 6.1); confirmed

- Methods (pp. 176-182): diver stereo photography (Briggs 1989; 35 mm camera at 91 cm, about 1 mm
  vertical and 0.5 mm horizontal resolution, profiles of 30-100 cm, prewhitened, 20 % cosine taper,
  periodogram, ensemble-averaged); remote 70 mm cameras at 90-200 cm; digital stereo at NURC since 1997
  (scales about 1 m to 1 mm, 2-D maps and 2-D spectra, Lyons and Pouliquen 2004); resistivity probes (IMP,
  1 cm horizontal resolution; IMP2, 4 m tracks); laser line scan (Moore and Jaffe 2002); manual tracing on
  Mylar; acoustic microprofiler at 5.75 MHz. Agreement between methods: resistivity and stereo give
  "statistically similar" strength and exponent, but resistivity lacks the highest spatial frequencies
  (p. 181). Laser and stereo agree over 0.02-2 cycles/cm (p. 182).
- **Table 6.1** (pp. 199-200; PDF 211-212): Briggs (NRL), 52 rows from 0.85 to 9.88 phi. Each row gives
  the RMS height (cm), the 1-D slope -gamma1 and the intercept phi1. Anisotropic sites have crest-to-crest
  and along-strike rows. The slope and intercept use the units of D.1.3: Phi1 in cm^3, F in cycles/cm, so
  phi1 is in cm^(3 - gamma1). Stated accuracy (p. 184): +2/-3 dB (95 %), so model-data differences up to
  4 dB "do not necessarily imply disagreement". Summary (ours, from the table as typed):
  - gamma2 = gamma1 + 1: median 3.43, quartiles 3.18-3.71, range 2.33-4.35. Sands (Mz < 3): median 3.39.
    Mz >= 5.5: median 3.65.
  - RMS height: sands 0.26-4.10 cm (median 0.63); Mz >= 5.5 0.21-0.65 cm (median 0.42). Fig. 6.17
    (p. 194): finer than 5 phi stays below 0.75 cm; coarse sediments range from 0.2 to 2.3 cm, with one
    point at 4.1 cm.
  - Some benchmark-site rows that a test could quote: SAX99 (c-c) 1.27 phi, 0.83 cm, -2.54, 0.00077;
    SAX04 (c-c, stereo) 1.47 phi, 1.81 cm, -2.99, 0.00014; Eckernfoerde Bay 9.88 phi, 0.37 cm, -2.42,
    0.00030; Panama City 1993 0.98 phi, 0.52 cm, -2.12, 0.00198.
- 2-D spectra: Figs. 6.4, 6.9, 6.10, 6.12. Bioturbated La Spezia fine sand is isotropic; Elba ripples give
  two off-axis spectral peaks; raking at SAX99 left an anisotropic spectrum that returned to isotropic
  within 24 h. These are figures only, with no fitted 2-D anisotropic parameters.
- A 1-D example from IMP2 at SAX04 (Fig. 6.8) shows a spectral peak from ripples near 0.13 cm^-1
  (wavenumber, rad/cm). This is a single track; a figure only.

### 2.3 Relations of roughness to sediment and between parameters (Secs. 6.6, 6.7); confirmed

- No predictive relation between RMS roughness and mean grain size (p. 193). None between gamma2 or w2
  and grain size either (p. 194, Fig. 6.18). The values cluster by type ("separate trends for sand and
  mud") and the dashed lines are a visual delineation only. "Sediment mean grain size alone may never
  yield" predictive relations (p. 195). Pouliquen et al. (2003) report the same for 1-D slope and
  intercept.
- Slope and intercept are correlated (Fig. 6.19, p. 196). The dashed line is the family of 1-D power laws
  with a fixed value S(F0) at F0 = 0.05 cycle/cm (20 cm ripple wavelength). The printed value is
  S(F0) = 7.1e-5 cm^3. **Unclear: the plotted line passes about (phi1 4e-5 cm^3, slope -3.5) and
  (1e-2 cm^3, -1.65). By (6.11) those points give S(F0) = phi1 / F0^gamma1 = 1.43 and 1.40 cm^3 (ours),
  not 7.1e-5.** The slope of the line itself confirms F0 = 0.05. Sites to the left (arrow 1) are muddy
  sites with little biogenic roughness, or sands whose ripples have decayed. Sites to the right (arrow 2)
  are fresh ripples. SAX99 ripple decay runs from far right towards the line (pp. 196-197).

### 2.4 Temporal variability and the power-law assumption (Secs. 6.4, 6.5); confirmed

| Observation | Numbers | Page |
|---|---|---|
| SAX99 storm ripples (50-70 cm wavelength) degraded by biology | to "subtle undulations" within 15 days; both slope and intercept fell; about 2 dB lower modelled backscatter at 40 kHz | p. 186-187, Fig. 6.7 |
| Laser scanner, SAX99 | ripple amplitude 2.5 -> 2.0 cm; smaller features degraded after 20-30 h | p. 181 |
| Raked ripples (1.95 cm tines, the Bragg spacing at 40 kHz) | gone within 24 h; backscatter at 40 kHz fell from about -21 dB to about -36 dB over about 24 h (site background -35 dB; all SAX99 sites -32 dB) | p. 187, Figs. 6.12, 6.13 |
| Elba, every 10 min for 10 days | wave ripples while wind > 30 kt and Hs > 1 m, then isotropic again with RMS roughness falling; diurnal burrowing (60-80 mounds and funnels per m^2) changed modelled backscatter at 20 deg and 100 kHz by up to 10 dB (model: first-order perturbation, nu_p = 1.31, delta_p = 0, a_rho = 1.92). Fitted exponents mostly 2.7-3.3 (figure) | pp. 187-189, Fig. 6.14 |
| Fine-grained, low-energy sites | an isotropic equilibrium of biogenic relief, often within "a few days" (Wheatcroft 1994) | p. 186 |
| Backscatter decorrelation time | about one month where volume scattering dominates; under one day where roughness dominates | p. 203 (Ch. 7) |
| Single power law | Fox and Hayes (1985): one power law from cm to km. SAX99 (Briggs et al. 2005a, Fig. 6.15): two power laws, one below 10 cm (biogenic) and one above 100 cm (morphology), with ripples at 10-100 cm. Fig. 6.16: roll-over beyond 10 cm; a slope break near 1-2 cycles/cm (Fig. 6.7, p. 182; real or a digitisation artefact is not known). Extrapolating to metre scales (a few kHz) "is reasonable" but unverified. Test: if the RMS height obtained by integrating down to the lowest spatial frequency exceeds the known relief, the power law is overextended | pp. 191-193 |

Research issues of Ch. 6 (Sec. 6.9, p. 198), summarised:
- Stationarity and Gaussianity are untested.
- It is not known whether one power law covers km to mm, nor over what band one (w2, gamma2) predicts
  scattering.
- The size of patches that can be treated as stationary is unknown.
- Concurrent roughness and acoustic measurements are needed.
- Rapid remote methods are needed.
- Means of describing patchiness should be developed (this links to Ch. 16).
- Summary statement (p. 197): predicting relief spectra from grain size "is unreliable"; measure them
  during the experiment.

### 2.5 Sediment heterogeneity (Ch. 7, App. D.2); confirmed

Definitions:

| Quantity | Book | Eq., page |
|---|---|---|
| normalised fluctuation | alpha(r) = <alpha>(z) [1 + gamma_alpha(r)] (alpha is any parameter) | (7.3), (7.4) p. 222 |
| covariance, cross-covariance | B_ab(r) = <gamma_a(r1) gamma_b(r1 + r)>; stationary within a layer | (7.5)-(7.7) pp. 222-223 |
| density, speed fluctuations | gamma_rho = delta rho / <rho>; gamma_p = delta Vp / <Vp> | (7.8), (7.9) p. 223 |
| 3-D spectrum | W_ab(k) = (2 pi)^-3 Integral B_ab(r) exp(-i k.r) d^3r; Integral W_aa d^3k = sigma_hat_a^2 (normalised variance); 100 x sigma_hat is the "coefficient of variation" | (7.10)-(7.13) pp. 223-224; (D.28)-(D.30) |
| model spectrum (after Yamamoto 1996) | W_ab(k) = w3ab / [Lambda^2 kx^2 + Lambda^2 ky^2 + kz^2 + Lc^-2]^(gamma3/2); w3 has dimension length^(3 - gamma3); Lc is the (vertical) correlation length or outer scale; Lc / Lambda is the horizontal one | (7.14) p. 224, p. 225 |
| its variance | sigma_hat^2 = pi^(3/2) w3 Lc^(gamma3 - 3) Gamma((gamma3 - 3)/2) / (Lambda^2 Gamma(gamma3/2)) | (7.15) p. 225 |
| pure power law | W = w3 / k^gamma3. gamma3 = 3 is self-similar, w3 is dimensionless, and the volume cross section is nearly frequency independent | (7.16) p. 225; p. 387 |
| exponential covariance | gamma3 = 4, Lambda = 1: W = w3 / (k^2 + Lc^-2)^2, B(r) = pi^2 w3 Lc exp(-r / Lc), sigma_hat^2 = pi^2 w3 Lc | (7.17), (7.18) p. 225 |
| AR(1) from core sections | B(n dz) = B(0) phi^n; Lc = -dz / ln(phi) (Yaglom's [1/2 + phi/(1 - phi)] dz differs negligibly) | (7.19), (7.20) p. 226 |
| 3-D -> 1-D (vertical) | W_z(kz) = 2 pi w3 / ((gamma3 - 2) Lambda^2 (kz^2 + Lc^-2)^((gamma3 - 2)/2)); power law: w1 = 2 pi w3 / (gamma3 - 2), gamma1 = gamma3 - 2 | (7.21)-(7.24) p. 226; (D.35) p. 481 (the subscript of the first gamma in (D.35) is cut off in the image; (7.21) prints gamma3, which the integral requires) |
| isotropic 1-D -> 3-D | W(k) = -(1 / (2 pi k)) dW_z/dk | (D.36) p. 481 |
| density from porosity | gamma_rho = -[(1 - rho_w / <rho>) / (1 - beta)] delta beta (beta is fractional porosity); sigma_hat_rho = [(1 - rho_w/<rho>)/(1 - beta)] sqrt(<delta beta^2>) | (7.25), (7.26) p. 227 |
| density-porosity regressions (Fig. 7.19, p. 228) | rho = 2.695 - 0.01697 beta (siliciclastic, r^2 0.9936); rho = 2.794 - 0.01783 beta (carbonate, r^2 0.9906); beta in %, rho in g/cm^3 | p. 228 |
| compressibility | c_p = 1/sqrt(kappa rho), shear neglected. **gamma_kappa = -2 gamma_p - gamma_rho**; B_kk = 4 B_pp + 4 B_prho + B_rhorho; B_rhok = -2 B_rhop - B_rhorho; the same holds for the spectra | (7.27)-(7.30) p. 229 |
| hypothesised spectra ("mu" relations) | W_kk = mu^2 W_rhorho; W_rhok = rho_rhok mu W_rhorho; mu >= 0, -1 <= rho_rhok <= 1 | (D.37), (D.38) p. 481; (14.26), (14.27) p. 386 |
| from speed spectra | if W_pp = eta^2 W_rhorho and W_rhop = rho_rhop eta W_rhorho, then **mu = sqrt(1 + 4 rho_rhop eta + 4 eta^2)** and rho_rhok = -(2 rho_rhop eta + 1)/mu | (D.39)-(D.42) p. 482 |
| perfect anticorrelation | rho_rhok = -1 requires rho_rhop = +-1. rho_rhop is expected negative only for very fine sediments below 1.4 g/cm^3 and positive otherwise (p. 482) | p. 482 |
| finite-resolution bias | measured covariance = true covariance convolved with the autocorrelation of the averaging function; measured spectrum W_aa(k) V(k) with V = |Integral y(r) exp(i k.r) d^3r|^2; core sections ("hockey puck"): V = b_T^2(K) b_z^2(kz), b_T = (2/(Ka)^2) Integral_0^Ka J0(u) u du, b_z = sinc(kz dz/2); resistivity probe: V = [ka Integral_ka^inf sin x / x^3 dx]^2 | (7.31)-(7.34) pp. 229-230; (D.43)-(D.54) pp. 482-484 |

Measured values (confirmed; individual values as printed):

| Site / study | Values | Page |
|---|---|---|
| Coefficients of variation, mean over sites (App. C Table C.1, not read) | attenuation 32.53 %, grain size 18.74 %, porosity 5.84 %, bulk density 2.95 %, sound speed 1.20 %, grain density 0.99 %; no trend of attenuation CV with grain size | pp. 234-235 |
| Deep-sea cores (Yefimov et al. 1988) | power law gamma3 = 3.5 over 0.2-100 m^-1 | p. 235 |
| Example used for the bias (Fig. 7.21) | gamma3 = 4 (7.17), w = 0.001 m^-1, Lc = 0.04 m; core sections of 2 cm length and 3 or 6 cm diameter bias the spectrum low by 5 dB or more at 10-100 kHz; the variance is halved; the inferred Lc is 9.5 cm against a true 4 cm | pp. 231-233 |
| Briggs and Tang (2002), Monte Carlo | true Lc 3.51 cm, estimated over 15 cm after core averaging | p. 229 |
| CBBL, Dry Tortugas (carbonate sand-silt-clay) | AR(1) Lc about 4 cm fits lags under 5 cm; CV of density 1.4 % (core) and 0.85 % (CT); surface transition in the top 5 cm (density ratio 1.475 -> 1.77; speed ratio 1.0 -> 1.018, Fig. 7.24) | pp. 235-237 |
| CBBL, Panama City (coarse quartz sand with shell, CT) | W_z(k) = a1/[1 + (L1 k)^2]^beta1 + a2/[1 + (L2 k)^2]^beta2 with a1 = 8.59e-7 m, a2 = 1.96e-10 m, L1 = 0.09 cm, L2 = 0.01 cm, beta1 = 2.8, beta2 = 2.1, so gamma3 = 7.6 and 6.2; power behaviour only for k > 1000 m^-1 (above about 100 kHz); below that shells dominate | (7.35) pp. 237-238 |
| Pouliquen and Lyons (2002), CT, Table 7.1 (p. 240), fit of (7.14) with Lambda = 1 | Tellaro 4.3 phi: sigma_hat_rho 0.021, gamma3 6.4, Lc 0.2 cm. Venere Azzurra 6.1 phi: 0.021, 6.4, 0.2 cm. Porto Venere 9.1 phi: 0.023, 7.6, 0.14 cm. P. D. Mariella 7.2 phi: 0.031, 8.0, 0.12 cm. Mean profile rho(z) = rho_s - (rho_s - rho_0)/(1 + a z), with (rho_s, rho_0, a) = (1.97, 1.52, 300), (1.99, 1.37, 800), (1.61, 1.18, 800), (1.81, 1.1, 100) (g/cm^3, g/cm^3, m^-1) | (7.36) pp. 239-240 |
| SAX99 sand, IMP (Tang et al. 2002) | mean rho(z) = 1.98 - 0.4 exp(-3.5 z^0.6); sigma_hat_rho(z) = 0.0152 + 0.096 exp(-3.7 z^0.82) (z in cm, from the figure's depth axis; units not restated by the equation); CV about 1.5 % below a few mm, about 10 % at the interface; 1-D vertical power law w1 = 0.0676 cm^(1 - gamma1), gamma1 = 2.17 (Fig. 7.29); horizontal gamma1 = 1.52-2.35 (smaller deeper, at 5-6 cm). Cores: Lc 2.55 (+0.95/-1.68) cm for sound speed and 3.51 (+2.27/-1.03) cm for density; CV 0.7 % and 0.73 % | (7.37), (7.38) pp. 241-243 |
| Microresistivity, Gulf of Mexico sand (Wheatcroft 2002) | porosity fluctuations 0.05-0.15 about the mean, vertical scales 5-15 mm | p. 214 |
| Eckernfoerde (Briggs et al. 1998) | vertical correlation lengths 2.1 cm (resistivity) and 2.8 cm (gravimetric), about 10 times those from X-ray; horizontal lengths longer than vertical | p. 221 |
| Volume scattering model inputs (Table 14.1, p. 390; rho_rhok = -1, Lambda = 1; w3 is in m^(3 - gamma3)) | Arafura 5.6 phi: a_rho 1.39, nu_p 0.988, delta_p 0.0055, gamma3 3.45, w3 1.32e-3, Lc inf, mu 1.389. San Francisco 6.4: 1.41, 1.002, 0.0104, 3.63, 1.37e-2, inf, 1.200. Orcas 8.1: 1.41, 0.977, 0.0019, 4.00, 6.44e-3, 0.032 m, 0.934. Porto Venere 9.1: 1.46, 0.993, 0.0011, 7.60, 5.13e9, 0.0014 m, 1.667. P. della Mariella 7.2: 1.66, 1.002, 0.0028, 8.00, 3.20e11, 0.0012 m, 1.667. For the first three sites mu came from eta (the speed/density CV ratio) with rho_rhop = 1; for the last two eta = 1/3 was assumed (Yamamoto) with rho_rhop = 1 (p. 482) | p. 390 |

Volume cross section from heterogeneity (Ch. 14, read only for this link; confirmed):
- sigma = |V_wp(ti)|^2 |V_wp(ts)|^2 sigma_v / (2 k_w |a_rho|^2 Im[P(ti) + P(ts)]), with
  P(t) = sqrt(a_p^-2 - cos^2 t), V_wp the pressure transmission coefficient, and sigma_v in m^2/m^3 = 1/m
  (14.8), p. 381. The incident and scattered fields are treated as plane waves (App. J).
- **sigma_2 = sigma_v / alpha_p, where alpha_p is the attenuation in dB/m** (14.9), from Mourad and Jackson
  (1989), pp. 381-382. "Typical values of sigma_2 range from 0.0001 to 0.004". Values above 0.004 "may
  indicate failure of the weak scattering assumption" (multiple scattering; D. Jackson et al. 1986a).
- Small-perturbation (Born) sigma_v (14.10), p. 384:
  `sigma_v = (pi |k_p|^4 / 2) [W_kk(dk) + 2 Re{(k_ps.k_pi) k_p*^2 / |k_p|^4 W_rhok(dk)} + |k_ps.k_pi|^2 / |k_p|^4 W_rhorho(dk)]`.
  Here k_p = k_w (1 + i delta_p)/nu_p (14.18) and dk = Re(k_ps - k_pi) (14.19). For backscatter this
  reduces to (pi |k_p|^4 / 2)(mu^2 - 2 rho_rhok mu + 1) W_rhorho(dk) (ours, from (14.10)-(14.17)).
  Density fluctuations scatter as a dipole and compressibility fluctuations as a monopole. With mu = 1 and
  rho_rhok = -1 the forward scatter cancels (p. 387).
- Roughness-volume interaction (14.1.2, p. 383): Ivakin and Lysanov (1981b) average the volume cross
  section over the distribution of large-scale slopes, numerically via (13.80), as in D. Jackson et al.
  (1986a) and Mourad and Jackson (1989). Correlation between relief and layering is open (App. M.1.2).
- Model-data comparisons at five soft sites (Fig. 14.4, 20-140 kHz) are "satisfactory". The heterogeneity
  inputs are uncertain: core-section bias, and rho_rhok was never measured but set to -1 (p. 389). Jones's
  exact 2-D calculations (14.1.6) find perturbation theory accurate except near the specular direction.

Research issues of Ch. 7 (Sec. 7.6, p. 243), summarised:
- No canonical spectral form exists. CT and conductivity give 3-D exponents of 6-8 at the highest
  wavenumbers.
- Isotropy at small scales is supported by CT.
- Probe disturbance, the thin "transition layer" at the sand surface, and core handling all need study.
- Correlations between acoustic parameters need new high-resolution methods.
- Discrete scatterers (shells, bubbles) lack a usable statistical description.
- Elsewhere (p. 221-222): the assumption that compressibility covaries with density is "widely used" but
  "has not been validated". Extrapolating low-resolution data with one power law "is not supported by
  available data".

### 2.6 Backscatter statistics (Ch. 16); confirmed

Complex representation (App. E, p. 485): p(t) = Re{P(t) exp(-i w_c t)} (E.1). |P| is the PEAK amplitude,
and the rms of a sinusoid is |P|/sqrt(2) (p. 486). A delay t_d gives the baseband P(t - t_d) exp(i w_c t_d),
not P(t - t_d) (E.4).

General (16.1, pp. 424-425): P = U + iV with <U> = <V> = 0, <U^2> = <V^2> = sigma^2, <UV> = 0, so
<P> = 0, <PP*> = 2 sigma^2, <P^2> = 0 (16.1)-(16.7). The envelope is x = |P| (16.8). The false-alarm
probability is P_fa(x) = Integral_x^inf p_x dx' (16.9). The scintillation index is
sigma_I^2 = (<x^4> - <x^2>^2)/<x^2>^2 (16.10). A uniform phase "is not a physical law", and it fails near
specular forward scatter (p. 425).

Gaussian case (16.2, pp. 425-429):
- Intensity y = x^2 is exponential: p_y = exp(-y/<x^2>)/<x^2>. The envelope is Rayleigh:
  p_x = (2x/<x^2>) exp(-x^2/<x^2>), with <x^2> = 2 sigma^2. P_fa(x) = exp(-x^2/<x^2>) (16.11)-(16.14).
- b = ln y: <b> = ln<x^2> - 0.5772157, and the std of b is pi/sqrt(6). **That is 5.57 dB in 10 log10
  intensity** (16.15)-(16.17), so Rayleigh reverberation fluctuates "a good deal" with no patchiness. The
  scintillation index is 1.
- Time covariance: B_PP(tau) = <P(t) P*(t + tau)>; <P(t)P(t + tau)> = 0; <P dP*/dt> = 0
  (16.18)-(16.20). The power spectrum is (1/2pi) Integral B_PP exp(-i w tau) dtau (16.21).
- If the cross section varies by less than about 1 dB across the band, then
  **B_PP(tau) = <|P|^2> rho_s(tau)**, where rho_s is the normalised autocorrelation of the baseband
  transmitted signal s(t) (16.22), (16.23) (Plemons et al. 1972). This does not need Gaussian statistics.
  It fails for a strongly frequency-selective (layered) seabed, which is unlikely at high frequency
  (pp. 428-429).
- Independent samples must be spaced by at least the pulse time resolution: the width of rho_s, about
  1/bandwidth, and the pulse length for a CW pulse (p. 429).
- Intensity covariance: <y(t) y(t + tau)> - <y>^2 = <|P|^2>^2 rho_s^2(tau) (16.26) (McDaniel 1990). It
  follows from the Gaussian fourth-moment rule (16.25).
- Ping-to-ping: fixed-platform experiments get their ensembles from ping-to-ping changes caused by the
  water column. These "may not correspond to those obtained with a sonar moving across the seafloor"
  (p. 422-424). The book gives no formula for ping-to-ping coherence of seafloor reverberation.

Non-Gaussian models (16.3). All are heuristic or empirical, and patchiness inside the ensonified area
(Lc x Ld, Fig. 16.4; App. G.2.1) is their common cause. Departures grow as the area shrinks (p. 430).

| Model | PDF of envelope x; mean square; P_fa | Parameters | Eqs., page |
|---|---|---|---|
| Lognormal | p_d(d) Gaussian in d = ln x with mean d_bar, variance sigma_d^2; p_x = exp(-(ln x - d_bar)^2/(2 sigma_d^2)) / (x sqrt(2 pi sigma_d^2)); <x^2> = exp(2(d_bar + sigma_d^2)) | 2 (1 free once <x^2> = 1) | (16.27)-(16.29) pp. 431-432 |
| Weibull | p_x = alpha beta x^(beta - 1) exp(-alpha x^beta); <x^2> = alpha^(-2/beta) Gamma(2/beta + 1); P_fa = exp(-alpha x^beta); beta = 2 is Rayleigh | alpha, beta | (16.30)-(16.32) p. 432 |
| **K** | **p_x = (4/(sqrt(alpha) Gamma(nu))) (x/sqrt(alpha))^nu K_(nu-1)(2x/sqrt(alpha)); <x^2> = alpha nu; P_fa = (2/Gamma(nu)) (x/sqrt(alpha))^nu K_nu(2x/sqrt(alpha))**. K_nu is the modified Bessel function of the second kind. Tends to Rayleigh for nu >> 1 | alpha (scale), nu (shape; need not be an integer) | (16.33)-(16.35) p. 433 |
| Rayleigh mixture | p_x = Sum P_m (2x/sigma_m^2) exp(-x^2/sigma_m^2), Sum P_m = 1; <x^2> = Sum P_m sigma_m^2; P_fa = Sum P_m exp(-x^2/sigma_m^2). Patches are larger than the ensonified area (one type per ping). It is NOT the PDF of a sum. Lyons and Abraham (1999) fit with two terms | P_m, sigma_m | (16.36)-(16.39) p. 434 |
| Crowther (1980) | p_ys(y, s) = (1/s) exp(-y/s) p_s(s) with <s> = 1; P_fa = Integral exp(-x^2/s) p_s ds (16.40), (16.41). Two patch types on a line of length X: the fraction beta of strong type (cross-section ratio g), <beta> = p, s = (1 - beta + g beta)/q, q = 1 + (g - 1)p, 1/q < s < g/q (16.42)-(16.45). Markov transitions with patch length L; nu = X/L; P0 = (1 - p) exp(-nu p), P1 = p exp(-nu(1 - p)) (16.46)-(16.48); continuous part with Bessel I0, I1, mu = 2 nu sqrt(p(1 - p) beta(1 - beta)) (16.49)-(16.52); closed P_fa (16.53). The book says it corrects inconsistencies in Crowther's expressions | g, p, nu | (16.40)-(16.53) pp. 434-437 |
| McDaniel 1990 | p_x = (2x/alpha) Integral_0^inf p_nd(x^2/(alpha v)) p_nc(v) dv/v, with chi-square p_n(u) = u^(n/2 - 1) exp(-u)/Gamma(n/2). The denominator is printed as Gamma(n/s), presumably a misprint for Gamma(n/2). <x^2> = alpha nc nd / 4; P_fa (16.57) with the incomplete gamma (16.58). With nc or nd = 2 it reduces to K with nu = nd/2 or nc/2. Its physics is "questionable" (McDaniel 1996) | nc, nd (degrees of freedom cross- and down-range) | (16.54)-(16.58) pp. 437-438 |
| **McDaniel 1996** (physical) | sonar sees s = Integral Omega(R) sigma(R) d^2R with Integral Omega = 1 (16.59), (16.60). Exponential intensity given s, so sigma_I^2 = 2<s^2>/<s>^2 - 1 (16.61). <sigma(R1) sigma(R2)> = <sigma>^2 [1 + gamma C_sigma(R1 - R2)], where gamma is the normalised variance of the cross section and C_sigma(0) = 1 (16.64). sigma_I^2 = 1 + 2 gamma Integral Integral Omega Omega C_sigma (16.65). Gaussian Omega = exp(-(x^2/Ld^2 + y^2/Lc^2))/(pi Lc Ld) and C_sigma = exp(-R^2/L^2) give **sigma_I^2 = 1 + 2 gamma / sqrt((1 + Lc^2/L^2)(1 + Ld^2/L^2))** (16.66)-(16.68). For a pulse with Gaussian envelope exp(-t^2/ts^2): **Ld = c_w ts / (2 cos theta)** (16.69), with Lc from range x azimuthal beamwidth. **Fitted to Gensane (1989): L = 1 m, gamma = 1.5** (p. 441). Fig. 16.7: theta = 30 deg, c_w = 1500 m/s, Lc = 1, 2, 4, 8, 16 m, ts 0.1-10 ms; the index runs from about 3.1 (Lc = 1 m, short pulse) to about 1.2 (Lc = 16 m). The statistics are stationary although the model is patchy (p. 442) | L, gamma (plus Lc, Ld from the sonar) | (16.59)-(16.69) pp. 439-441 |

Measured or fitted statistics as printed (no shape values in tables):
- Lyons and Abraham (1999), 80 kHz (Fig. 16.3, p. 426): shell, sand and mud sites; grazing bands 20-40,
  40-60 and 60-80 deg. Rayleigh, Weibull, K and Rayleigh mixture are compared. The departure from Rayleigh
  grows with grazing angle "due to the reduction in area of the ensonified region". Shell is the most
  non-Rayleigh, mud the least. The fitted values are figures only.
- Chotiros et al. (1985) (p. 432): 30 kHz, 0.25 ms pulse, ranges 80-101 m, height 4.5 m, 3-dB full
  beamwidths 2.8 and 21 deg; fine sand Mz 3.5 phi, a_rho 1.9, nu_p 1.10, alpha_p 14.1 dB/m. Lognormal with
  sigma_d fixed at 5.57 dB. The wide beam is Rayleigh. The narrow beam is lognormal in the tail below
  P_fa 0.03 and Rayleigh above it.
- Gensane (1989): lognormal fits best of Rayleigh, lognormal, Weibull, chi-squared and Rice, but the data
  failed stationarity tests (p. 432).
- Crowther (1980): fitted at 1.8, 4.1 and 8.1 kHz at several sites (p. 437).
- Abraham and Lyons (2002a, 2004), 450-700 Hz: **nu increased with bandwidth (resolution) "up to a point"
  and then decreased at the highest resolution**. This is explained by the large aspect ratio Lc/Ld of the
  cell (p. 433).
- Examples of the size of the effect: Weibull with beta = 1.3 (Fig. 16.1): 18 of 4000 samples exceed a
  threshold 10 dB above the rms envelope (P_fa about 0.0045). Rayleigh gives exp(-10) = 4.5e-5 there.
  Ours, from (16.31)-(16.32): 0.0042. Fig. 16.5 curves: lognormal (sigma_d at 5.57 dB), K (nu = 10),
  Weibull (beta = 1.5), mixture (P1 0.75, P2 0.25, sigma_2^2/sigma_1^2 = 2), Crowther (nu 8, g 10, p 0.1;
  Fig. 16.6).

Research issues of Ch. 16 (Sec. 16.4, p. 442), summarised:
- Appropriate environmental measurements are lacking.
- Most models are empirical or heuristic.
- Large echo ensembles are hard to collect without platform and environmental drift.
- Mechanisms should be isolated, partly in the laboratory.
- "Stationary, Gaussian roughness can produce non-Gaussian scattering due to elevated scattering from
  steep slopes."
- Start with simple seafloors; develop and apply measures of patchiness.
- Also (p. 441-442): where first-order perturbation theory fails (ripple height comparable to the
  wavelength), non-Gaussian statistics may arise even on featureless seafloors.

## 3. Against our simulator

### 3.0 How it was checked (ours)

Scripts in the scratchpad (not committed): `jr2/checks.py`, `jr2/checks2.py`, `jr2/t61.py`. They load
`python/gpusonar/sim/bottom.py` of this worktree through a stub package, so no compiled module is needed
and no symlink was made. `reverb.py` needs the compiled module, so `CellTexture`'s two lines (a gamma
variate of shape s and mean 1, its square root times a complex Gaussian) were reproduced and simulated
with 2e6 samples. Table 6.1 was typed from the page image and checked against the text extraction (all 52
rows agree).

### 3.1 Roughness conventions

| Item | Ours | Book | Verdict |
|---|---|---|---|
| Spectrum definition and normalisation | `bottom.py` docstring: W(K) = w2/K^gamma2, K in rad/m, 2-D, <zeta^2> = Integral W d^2K | (6.5)-(6.8), (D.3)-(D.4), (D.9) | **agrees, to the factor** (same definition: 2-D two-sided, rad/m, length^(4 - gamma2)). This confirms the docstring line marked "as in the published models" |
| Structure constant | `structure_constant`: Ch2 = 2 pi w2 Gamma(1 - a)/(4^a a Gamma(1 + a)) | (D.12): 2 pi w2 Gamma(2 - a) 2^(-2a)/(a(1 - a) Gamma(1 + a)) | **identical** (Gamma(2 - a) = (1 - a) Gamma(1 - a)); numerically equal to 2e-16 at gamma2 = 2.5, 3.0, 3.25, 3.9; (D.10) by quadrature matches to 1.5e-5. Ours was marked "ours"; now **confirmed** |
| cm -> m | `w2_from_cm4`: w2 = beta 10^(2 gamma - 8) | D.1.3: divide by 100^(4 - gamma2) | **identical** (numerically equal at 3.0, 3.25, 3.5). Bentrem's form, until now confirmed only from arXiv:physics/0607146, is **confirmed** here too |
| Composite split, slope variance | `large_scale_slope`: s^2 per axis = pi w2 Kc^(4 - gamma2)/(4 - gamma2) | follows from (6.7) with the same normalisation (RMS slope itself is (13.68), Ch. 13, not read here) | consistent with the normalisation; the formula is ours |
| Outer scale | none (pure power law; Kirchhoff and small-slope use the structure function) | von Karman (D.7) with K0; the book says the power law must roll over | absent. It matters only for quantities that need h (Rayleigh parameter, coherent loss); our models avoid h |
| Isotropy | assumed | anisotropic ripples are common in sands (Secs. 6.3, 6.4) | **absent**: no directional spectrum, so no azimuth dependence of bottom backscatter |
| Time variation | none | hours to days, up to 10 dB (Sec. 2.4) | absent (a scenario can only set w2) |

### 3.2 Our sediment roughness parameters against the book's data

REFERENCES.md Sec. 3.1 marks w2 (Snellen 2013 regression) and gamma2 = 3.25 as "via secondary". The book
does not print the handbook's regression (it prints no regression at all), so the rows stay "via
secondary". The data it gives allow an independent check (ours). I compare W(K) at the backscatter Bragg
wavenumber K = 2k cos(30 deg) for each Table 6.1 row (converted to 2-D with the CORRECTED (D.19), in
m^(4 - gamma2)) against `grain_size_scattering(Mz)`:

| Frequency | Sands Mz < 3 (41 rows) | Fine 3 <= Mz < 5.5 (3) | Mz >= 5.5 (8) |
|---|---|---|---|
| 10 kHz (K = 73 rad/m) | median +1.4 dB (quartiles -1.6 / +3.0; range -12.4 to +9.5) | +7.8 | +7.1 (+2.9 to +14.5) |
| 100 kHz (726 rad/m) | -0.0 dB (-4.7 / +2.0; -13.1 to +7.4) | +7.0 | +2.2 (-2.4 to +13.4) |
| 300 kHz (2177 rad/m) | -0.6 dB (-6.3 / +2.2; -15.3 to +9.9) | +7.4 | -0.9 (-7.1 to +13.2) |

(Positive values mean the data have more roughness than our rule.)

- Sands: our rule is at the centre of the measured spread at every frequency. The spread (about 20 dB) is
  the book's point: grain size does not predict roughness.
- Silts and clays: our rule is below the data at low frequency, because the data have steeper exponents
  (median gamma2 3.65 against our 3.25) and a higher level. The eight rows come from four sites, two of
  them glacial till and carbonate, so this is weak evidence. It is not a reason to change a default.
- gamma2 = 3.25 sits inside the quartiles of the data (3.18-3.71), below the median 3.43.
- Gallaudet's fit (gamma 3.00, w2 0.000518 cm^4 at about 6.4 phi; REFERENCES Sec. 3.2) lies within the
  Table 6.1 spread for muds.

### 3.3 Volume term and the volume parameter

| Item | Ours | Book | Verdict |
|---|---|---|---|
| Volume cross section | `volume_strength`: sigma_vol |T_i|^2 |T_s|^2 / (rho^2 2 (Im b2i + Im b2s)) | (14.8) | **identical**: computed both ways at 10, 30 and 60 deg (30 kHz, a_rho 1.5, nu 1.05, delta 0.015, sigma2 0.002): -35.318, -26.293, -24.227 dB, equal to 1e-13 dB. REFERENCES Sec. 3.2 "Our derivation ... volume term" (ours, checked against Snellen 2013) is now **confirmed in the book** |
| sigma2 definition | sigma_vol / attenuation in dB/m (`bottom.py`, `configs/sediments.yaml` header) | (14.9): sigma_v / alpha_p in dB/m (Mourad and Jackson 1989) | **identical**, confirmed |
| Published curve to reproduce | - | Fig. 14.2 (p. 382): 30 kHz, a_rho 1.5, delta 0.015, sigma2 0.002, nu 0.98/1.05/1.1; critical angles about 18 and 25 deg | ours computes -24.1 / -23.6 / -23.3 dB at 90 deg and -27.3 / -26.3 / -25.7 at 30 deg; the figure reads about -24 to -23.5 and -27 to -26 (by eye); critical angles acos(1/nu) = 17.8 and 24.6 deg. A test can hold the curve shape against the figure |
| Typical range of sigma2 | rule 0.002 (< 5.5 phi), 0.001 (>= 5.5 phi); medium_sand 0.001 (D-90) | 0.0001-0.004; > 0.004 suspect (p. 382) | both values for medium sand are inside the book's range. **The book does not settle D-90**: it prints no sigma2 for any named sediment class |
| sigma2 implied by measured heterogeneity (ours: (14.10) backscatter with rho_rhok = -1, Table 14.1 inputs) | rule 0.001 for all five (Mz 5.6-9.1) | Arafura 20 kHz 0.0031; San Francisco 40 kHz 0.0037; Orcas 40 kHz 0.0008; Porto Venere 140 kHz 0.0017; P. della Mariella 140 kHz 0.0016 | four of five muds are 1.6-3.7 times our rule (+2 to +5.7 dB of volume scattering). With gamma3 != 3, sigma2 is frequency dependent, which the rule ignores. This weakly supports Gallaudet's fitted 0.003 for a silt. Not a reason by itself to move the 0.001 step (it would move every mud result); it is a reason to show the owner |
| Heterogeneity spectra, mu, rho_rhok | absent (sigma2 lumps them) | Sec. 2.5 | absent; needed only for the bistatic volume pattern (dipole/monopole) and for frequency dependence |
| Slope averaging of the volume term | absent (REFERENCES Sec. 3.2: "we apply none") | 14.1.2: average over large-scale slopes, via (13.80) | the book confirms the published model does this; the method is (13.80), in Ch. 13 (another file) |

### 3.4 Statistics

| Item | Ours | Book | Verdict |
|---|---|---|---|
| K PDF | `reverb.CellTexture`: intensity = Gamma(shape s, mean 1) x exponential; E{I^2}/E{I}^2 = 2(1 + 1/s) | (16.33)-(16.35): shape nu, scale alpha, <x^2> = alpha nu | **our `shape` = the book's nu** (and Abraham-Lyons alpha, D-196). Checked: the book's PDF integrates to 1, <x^2> = 1, <x^4> = 2.6667 at nu = 3. Our simulated P_fa at x = 1.5 / 2.5 / 3.5 is 0.1141 / 0.00966 / 0.00059 against the book's (16.35) 0.1142 / 0.00964 / 0.00061. Scintillation index 1.663 against 1 + 2/nu = 1.667. D-196's worked example recomputes from (16.35): 1.685 dB (nu 10) and 2.694 dB (nu 5) above Rayleigh at 1e-6. **Notation trap: the book's alpha is the SCALE; Abraham-Lyons' alpha is the SHAPE** |
| shape <-> count | D-196: n scatterers -> n/2; n patches -> n | p. 433: "2 nu" patches of exponentially distributed area | **conflict** (see the top of this file); keep D-196 (primary source) |
| Shape against cell size | `scatterers_per_m2` / `patches_per_m2`: shape proportional to area | McDaniel 1996: sigma_I^2 - 1 proportional to 1/sqrt((1 + Lc^2/L^2)(1 + Ld^2/L^2)); in the small-patch limit this is proportional to L^2/(Lc Ld), i.e. to 1/area, which is our rule; for cells smaller than L it saturates at 1 + 2 gamma. Abraham-Lyons: nu stops rising and falls at the highest resolution | ours is the large-cell limit; it lacks saturation for small cells (SAS, high-frequency FLS) and correlation between cells |
| Texture correlation between cells | none (independent per cell; sonar-6fc) | McDaniel 1996: correlated cross-section field, C_sigma = exp(-R^2/L^2), L = 1 m fitted | absent; the book gives a usable model and one fitted pair (L, gamma) |
| Intensity statistics in Rayleigh | exponential, uniform phase | (16.11)-(16.17): 5.57 dB std of the dB level, index 1 | agrees (the standard result) |
| Sample correlation within a ping | generator: complex Gaussian sequence filtered by the pulse | (16.22), (16.26): B_PP = <|P|^2> rho_s; intensity covariance rho_s^2 | consistent with how our generator builds reverberation (filtered white sequence). The detector constants of D-93/D-168 (8 correlated samples a 2 m cell; "15 samples a cell, 10.8 looks") are the effective-looks count N^2 / Sum_ij rho_s^2(t_i - t_j) that (16.26) implies. That formula is standard; the book supplies (16.26) as a confirmed print |
| Non-Rayleigh thresholds | the detector assumes Gaussian noise (D-93 "Limits") | Fig. 16.1: at 10 dB above rms, Weibull beta 1.3 gives 100 times the Rayleigh P_fa | the book quantifies the risk; nothing to change without non-Rayleigh scenes |
| Ping-to-ping coherence | independent realisations per ping (sonar-6t5) | none for the seafloor; the in-ping time covariance only; the decorrelation time of the seabed itself is about 1 day (roughness) to 1 month (volume) (p. 203) | the book gives no closed form for sonar-6t5. It does say a still sonar's ping ensemble is NOT representative of a moving sonar: with a fixed seafloor realisation, only water-column fluctuation changes the echo |
| Baseband convention | ours: analytic signal with exp(+2 pi i F t), baseband = x exp(-2 pi i f_mix t) (PHYSICS_NOTES Sec. 15) | (E.1): p = Re{P exp(-i w_c t)} | **opposite sign of the carrier**: our baseband is the complex conjugate of the book's P. Envelope statistics are unaffected. When coding a book formula with phases (E.4, the delay factor exp(+i w_c t_d)), use exp(-i w_c t_d) in ours. The book's |P| is the PEAK amplitude; check our level convention (PHYSICS_NOTES Sec. 10 row "Complex envelope, sqrt(2) convention") before quoting a level in uPa against the book |

### 3.5 What the book confirms of rows marked "ours", "via secondary" or "wanted"

- `bottom.py` roughness spectrum convention, `structure_constant`, and `w2_from_cm4`: **confirmed**
  (above).
- The volume term (REFERENCES 3.2 row 1) and the definition of sigma2: **confirmed**.
- REFERENCES 3.1, per-parameter w2 rule and gamma2 = 3.25: still "via secondary". The book prints no
  regression. Its data are consistent for sands (Sec. 3.2).
- REFERENCES 10, "Medium sand: volume parameter 0.001 or 0.002": **not settled** (no class table in these
  chapters).
- REFERENCES 10, "K-distribution shape against cell size": partly answered. There are no shapes per
  seabed, but McDaniel's model and L = 1 m, gamma = 1.5 (fitted) are printed, plus the qualitative
  Abraham-Lyons bandwidth result.
- REFERENCES 10, "Interpolation between Kirchhoff and composite ... rough-bottom empirical expression":
  not in these chapters (Ch. 13 / Ch. 12; another file).

## 4. What this closes or changes in open issues

| Issue / row | Effect |
|---|---|
| sonar-6fc (K texture: correlation across cells and pings, shape tied to cell size) | Gives a physical model with printed parameters (McDaniel 1996; WP-1): a correlated cross-section field yields both the shape against cell size, saturating for small cells (which our area rule lacks), and correlation between neighbouring cells. Correlation between pings follows automatically for a frozen field (seabed decorrelation time of a day or more). Confirms that our `shape` is the book's nu. Raises the factor-2 wording conflict with D-196 (record it, change nothing). Does not provide per-seabed shapes |
| sonar-6t5 (ping-to-ping coherence) | No closed form in these chapters. Supports the issue's premise: the seabed realisation is frozen over seconds to hours (decorrelation about 1 day for roughness and 1 month for volume, p. 203), so a still sonar should repeat. The in-ping covariance (16.22) is the within-ping half. WP-1's frozen field also serves 6t5 for the texture |
| sonar-3k1 (bottom model as published) | Confirms our roughness spectrum convention, structure function and volume term (eqs. (14.8)-(14.9), D.1-D.13) to the factor. Adds: the von Karman form with outer scale (D.7)-(D.8); slope averaging of the volume term (14.1.2, method in Ch. 13); the heterogeneity-based volume model (14.10) as an alternative to sigma2 (WP-4). It does not print the handbook's regressions or interpolation in these chapters |
| D-90 medium_sand sigma2 0.001 vs 0.002 | Not settled. Both lie in the typical 0.0001-0.004 range (p. 382) |
| sigma2 step 0.001 for muds | The heterogeneity-derived values for four of five soft sites are 0.0016-0.0037 (ours via (14.10)); see WP-4 for whether to show the owner |
| Grain-size regressions (w2) | The book prints none and says grain size does not predict roughness. Our rule is at the centre of 41 measured sand rows; muds are 1-7 dB below the data (Sec. 3.2) |
| STATUS 30 (sediment fit with perturbation) | Indirect: the book's measured roughness spread (about 20 dB at fixed grain size) and the temporal changes (up to 10 dB in a day) argue for fitting w2 per site rather than trusting a class value. Nothing in these chapters on the fit's model choice |
| sonar-9le (surface) | **Nothing**: the book is about the seafloor. Ch. 16's statistics models are generic and apply to surface reverberation only by analogy |
| sonar-sul (real data) | Table 6.1, Fig. 14.4 and Fig. 6.13-6.14 are measured seafloor numbers a fit or a check could use; no echo data |
| sonar-7iy, bistatic, elastic, layered, Lambert, large-roughness gravel/rock law | Not in these chapters |
| Detector constants (D-93, D-168) | (16.26) confirms the intensity covariance rho_s^2 behind the effective-looks count; (16.14) and (16.17) confirm the Rayleigh/exponential baseline. Non-Rayleigh risk quantified (Fig. 16.1) |

## 5. Work packages for agents (ordered by value to realism per effort)

**WP-1 Correlated cross-section texture (McDaniel 1996) as an opt-in K-like model (sonar-6fc, sonar-6t5). Size M.**
- Goal: texture correlated between cells and frozen between pings, with a shape that follows both cell
  dimensions and saturates for small cells.
- Model: a stationary positive field sigma(R)/<sigma> = 1 + delta(R) on the boundary plane, with variance
  gamma and correlation exp(-R^2/L^2) (16.64), (16.67). The cell's mean intensity is the
  Omega-weighted average s (16.59), and the speckle is exponential given s (16.61).
- Closed forms for tests:
  - sigma_I^2 = 1 + 2 gamma / sqrt((1 + Lc^2/L^2)(1 + Ld^2/L^2)) (16.68), with Ld = c ts/(2 cos theta)
    (16.69).
  - The K-equivalent shape is nu = 2/(sigma_I^2 - 1).
  - Fig. 16.7 values: theta 30 deg, L = 1 m, gamma = 1.5. Short-pulse plateaux are about 3.12 (Lc 1 m),
    2.3 (2 m), 1.73 (4 m), 1.37 (8 m) and 1.19 (16 m); ours gives 3.121 and 1.187 for the first and last.
- Make the field a gamma (or lognormal) random field so it stays positive. Gaussian delta with gamma = 1.5
  goes negative: choose and document.
- Seed it by geographic position so a still sonar repeats and a moving one decorrelates (sonar-6t5's key
  idea).
- Files: `python/gpusonar/sim/reverb.py` (`CellTexture` or a new class beside it), the `statistics` keys
  (`model: correlated`, `correlation_length_m`, `cross_section_variance`), both engines if it changes
  deviates (today the texture is drawn in Python and multiplies the deviates handed to both engines, so
  parity is kept by construction, D-196), `docs/PHYSICS_NOTES.md` Sec. 14.
- Tests first:
  - (16.68) against Monte Carlo of the cell average for Gaussian Omega and C;
  - the large-cell limit reproduces the area rule of D-196;
  - correlation between adjacent cells against the closed-form overlap integral of two Gaussian windows
    (ours to derive);
  - P_fa of the simulated envelope against the K form at the equivalent shape (approximate, documented
    tolerance);
  - a still sonar gives correlation 1 between pings, a moving one decorrelates over about L.
- Owner first: none if opt-in. No default changes.
- Must not: present L = 1 m, gamma = 1.5 as a default for any seabed. They are one fit to one data set
  (Gensane 1989, via McDaniel 1996, via the book): status "via secondary".
- Depends on: nothing. It also feeds sonar-kal's clutter scene.

**WP-2 Record the conventions and the confirmations in the docs (no code). Size S.**
- In `docs/REFERENCES.md`:
  - Add the book as a source, status "confirmed", for the rows of Sec. 3.5 of this file.
  - Sec. 11: the factor-2 wording conflict between the book (p. 433) and D-196.
  - The (D.19) and (D.23) misprints.
  - The Fig. 6.19 constant.
- In `bottom.py` docstrings (comments only): cite (6.5)-(6.8), (D.11)-(D.13), D.1.3, (14.8)-(14.9).
- PHYSICS_NOTES Sec. 15 / App. E: our baseband is the conjugate of the book's P.
- Tests: none needed (comments); optionally add a test that holds `structure_constant` equal to (D.12) as
  printed, and `w2_from_cm4` to D.1.3.
- Owner: no.

**WP-3 A 1-D -> 2-D roughness conversion helper and Briggs' Table 6.1 as test data. Size S.**
- Goal: let a user (or `tools/fit_sediment.py`) enter measured 1-D spectra the way the literature prints
  them.
- Implement:
  - `roughness_2d_from_1d(phi1, gamma1, units="cm")`: gamma2 = gamma1 + 1 (D.18);
    w1 = phi1 (2 pi)^(gamma1 - 1) (D.24); w2 = w1 Gamma(gamma2/2)/(sqrt(pi) Gamma((gamma2 - 1)/2))
    (D.19 corrected); then divide by 100^(4 - gamma2) for metres (D.1.3).
  - Optionally a von Karman `K0` (D.7) and its h^2 (D.8), used where an RMS height is needed.
- Tests first:
  - the quadrature of the 2-D law over Ky equals W1 (to 1e-8);
  - Phi1 integrates to h^2 for a von Karman pair;
  - one Table 6.1 row by hand (SAX99 c-c: gamma2 3.54, w2 7.54e-3 cm^(4 - gamma2) (ours)).
- Do NOT add the 52 rows as a shipped table (copyright; raw measurements). A test may carry three or four
  rows with page citations.
- Owner: no (new function).

**WP-4 Show the owner the volume-parameter evidence; optional heterogeneity-based volume model. Size S (memo) / M (model).**
- Memo: the numbers of Sec. 3.3 (implied sigma2 0.0008-0.0037 for five soft sites against the rule's
  0.001; the book's range 0.0001-0.004). Before -> after if the mud step were 0.002: +3.0 dB on the
  volume term of every row >= 5.5 phi at all angles. These rows are volume dominated off specular, so
  their backscatter rises by nearly 3 dB. Prepare the table per row at 6, 30 and 100 kHz and 30/45 deg
  with `named_backscattering_strength_db`.
- **Owner decides; no change without it.**
- Optional model: `volume_model: "heterogeneity"` with inputs (gamma3, w3 in m^(3 - gamma3), Lc, Lambda,
  mu, rho_rhok), computing sigma_v by (14.10) with the spectrum (7.14). It gives the frequency dependence
  and bistatic dipole/monopole pattern that sigma2 lacks. Tests:
  - backscatter reduces to (pi|k_p|^4/2)(mu^2 - 2 rho_rhok mu + 1) W(dk);
  - with gamma3 = 3 the cross section is nearly frequency independent (p. 387);
  - the Table 14.1 numbers above;
  - Fig. 14.3's forward null for mu = 1, rho_rhok = -1.
- Fig. 14.4 is a figure only.

**WP-5 Optional anisotropic (rippled) roughness spectrum. Size M. Depends on the bistatic/azimuth plumbing.**
- Goal: azimuth-dependent bottom backscatter over ripples. The book shows peaks in 2-D spectra (Figs. 6.9,
  6.12) but prints no parametric anisotropic form in these chapters. The model would be ours: for
  example, the isotropic power law plus a Gaussian ripple peak at wavenumber K_r and direction psi_r.
  Any parameter values must come from a site.
- Tests: the isotropic limit equals today's result to the bit; the Bragg peak appears at
  2k cos(theta) = K_r when looking across the crests.
- Owner: opt-in, so no. Do not use Fig. 6.8 numbers (figure only).

**WP-6 K-aware detection threshold option (D-93 limits). Size M. Lower priority.**
- Goal: given a K shape nu, set the threshold from (16.35) instead of the exponential.
- Tests: D-196's 1.685 / 2.694 dB at 1e-6; Rayleigh as nu -> inf.
- Owner first: it changes detection results if made default, so keep it opt-in.
- Depends on WP-1 for realistic nu.

Must not, across all WPs:
- change `sediments.yaml` values or the 0.001/0.002 sigma2 step without the owner;
- tabulate Table 6.1 or Table C.1 in the repository;
- use (D.19), (D.23) or the 7.1e-5 constant as printed;
- present McDaniel's L and gamma, or any Table 14.1 site value, as a default for a sediment class.

## 6. Not verified / unclear on the page

- (D.19) denominator Gamma((gamma2 - 2)/2): read clearly at 220 dpi. Judged a misprint by calculation, not
  by an erratum.
- (D.23) exponent F^(gamma1 - 1): read clearly. Judged a misprint against (6.11), (D.22) and (D.24).
- (D.35): the subscript of the first gamma is cut off at the bottom of the zoomed crop. The 100 dpi page
  suggests gamma2; (7.21) prints gamma3. Not re-zoomed.
- Sec. 6.7, S(F0) = 7.1e-5 cm^3: printed clearly. The line in Fig. 6.19 implies about 1.4 cm^3 (my reading
  of two points off the figure, +-20 %).
- (16.55) prints Gamma(n/s); presumably Gamma(n/2).
- (7.38): units of z not restated next to the equation (the figure's depth axis is in cm). Taken as cm,
  unverified.
- Which conversion Briggs et al. (2005a) used for Fig. 6.18 is not known. My corrected values put the
  highest point (Mission Bay coarse) at 0.047 cm^(4 - gamma2) against about 0.03 read off the figure. The
  as-printed (D.19) gives 0.034, which would fit the figure better. **So Fig. 6.18 may have been computed
  with the misprinted formula; unresolved.**
- The Fig. 14.2 comparison is by eye from the figure (about 0.5-1 dB).
- Not read: App. C (Table C.1), App. G (G.2.1 ensonified area), App. M, and Ch. 13's slope formula (13.68)
  and composite averaging (13.80), which this file only points to.
