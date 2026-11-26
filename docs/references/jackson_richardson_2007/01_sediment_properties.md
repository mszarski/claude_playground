# Jackson and Richardson (2007), Chapters 1-5 and Appendices A, B, C, H, I: sediment properties, definitions, seawater

D. R. Jackson and M. D. Richardson, *High-Frequency Seafloor Acoustics*, Springer, 2007. Notes for issue
sonar-gm8e. Formulas are re-typed in the book's notation with our symbol beside them; values carry the printed
page and the equation or table number. "confirmed" means read on the page image of this book. Where the book
attributes a relation to APL-UW TR 9407 (not read, distribution-limited), this file records what the book prints
and says so. Numbers marked "ours" were computed in this session with the scripts named in Sec. 3.0.

Page numbers are the book's printed pages. PDF page = printed + 14 in Chs. 1-5, + 8 in Chs. 11 and 14, + 7 in
the appendices (the PDF and the printed numbering drift; check before quoting a PDF page).

**What matters most, first.**
- The "wanted" piecewise polynomials rho(Mz), nu(Mz), delta(Mz) (REFERENCES.md Sec. 10) ARE printed in the book,
  but in Ch. 11 (p. 313-314, eqs. (11.1)-(11.3)), attributed to Mourad and Jackson (1989) and TR 9407, and the book
  says plainly they are *ad hoc model-fit relations, not geoacoustic regressions* (p. 314, also p. 145-146). Our
  `GRAIN_SIZE_TABLE` reproduces them at every half phi to the printed digits (rho within 4.5e-4, delta within 0.27 %
  with c_w = 1528 m/s) with ONE exception: for Mz >= 5.3 the book prints nu = -0.0024324 Mz + **1.0091**; our table
  (and its three earlier prints) follows **1.0019**, which is the only constant that makes the relation continuous
  at 5.3 phi. The book's constant is very probably a transposed-digit misprint. Our table stands.
- Our linear interpolation between half-phi rows differs from the book's polynomials by at most 0.0106 in rho
  (1.25 phi), 0.0041 in nu (5.3 phi), 7.6 % in delta (5.8 phi) and **0.68 dB in the normal-incidence reflection
  coefficient** (5.3 phi); below 5 phi and above 6 phi it is under 0.1 dB.
- The book's own ("standard") regressions of Ch. 5 (Richardson-Briggs data, 400 kHz cores and 38/58 kHz in situ)
  disagree with the TR 9407 relations by up to **12.5 dB** in normal-incidence bottom loss for silts (4-7 phi): the
  TR 9407 density ratio for fine sediments is far below measured densities. The book reproduces this in its
  Fig. 11.2 and we reproduce the figure's curves (Sec. 3.2). Neither is "wrong"; they answer different questions.
- Medium sand volume parameter (D-90): **the book prints no named-sediment table with sigma2.** It gives only the
  definition sigma2 = sigma_v / alpha_p, a typical range 0.0001 to 0.004, and examples with 0.002. D-90 stays open.
- Seawater: our density (one atmosphere), Francois-Garrison absorption and the loss-parameter conversions are
  printed in the book with the same constants: confirmed. The book does not print Mackenzie (1981); it prints
  Medwin (1975) and Del Grosso (1974); our Mackenzie agrees with them within 0.25 and 0.4 m/s.
- Four misprints found in my chapters (Sec. 6): Table 5.3's carbonate constant and both attenuation regressions,
  eq. (5.8), and the ripple constant (500 D text vs 535 D figure).

---

## 1. Scope

| Part | Printed pages | PDF pages | How read |
|---|---|---|---|
| Ch. 1 Introduction | 1-6 | 15-20 | text in full (no equations) |
| Ch. 2 Basic concepts and definitions | 7-27 | 21-41 | page images in full |
| Ch. 3 Nature of marine sediments | 29-73 | 43-87 | text skimmed; page images of 62-63 (ripple relations) and 72-73 (Research Issues); 3.3 bubbles read in text |
| Ch. 4 Physical properties | 75-122 | 87-136 | page images of 77-79, 84-86, 88, 91-98, 100-102, 112-115, 119-122; text of 116-118; 4.4.2-4.4.4 (electrical, thermal conductance) only skimmed |
| Ch. 5 Geoacoustic properties | 123-170 | 135-184 | page images in full |
| App. A List of symbols | 443-454 | 451-461 | page images in full |
| App. B Seawater | 455-469 | 462-476 | page images in full |
| App. C Coefficient of variation | 471-473 | 477-479 | page images in full |
| App. H Units and decibels | 509-510 | 513-514 | page images in full |
| App. I Causality | 511-518 | 515-522 | page images in full |
| Outside my part, read only to answer the brief | Ch. 8 p. 250-252; Ch. 11 p. 312-314; Ch. 14 p. 381-382 | 261-263, 320-322, 389-390 | page images: the loss-parameter definitions, the TR 9407 grain-size relations, the volume-parameter definition |

Not read: Chs. 6-16 otherwise, Apps. D-G, J onwards (other parts of the extraction).

## 2. What the book gives

### 2.1 Conventions and definitions (Ch. 2, Ch. 8, App. A, App. H, App. I) - all confirmed

| Item | Book | Where | Our code | Agrees? |
|---|---|---|---|---|
| Time dependence | exp(-i w t); plane wave exp(i k_p x) decays if Im k_p > 0 | p. 247 (text), p. 250 eq. (8.16), App. I eq. (I.7) | `bottom.reflection`, `_vertical`: k2 = k (1 + i delta)/nu, branch with Im > 0 | yes (same sign convention) |
| Angles | grazing angles from the horizontal: theta_i, theta_s; bistatic azimuth phi_s; backscatter theta_s = theta_i, phi_s = pi | p. 18, Fig. 2.4 | grazing angles; backscatter azimuth = pi (`strength(..., azimuth=np.pi)`) | yes |
| Grain size | phi = -log2(d), d in mm | eq. (2.1) p. 8; Table 4.1 p. 78 | `mean_grain_size_phi`, `grain_size_phi` | yes |
| Mean grain size | graphic mean Mz = (phi16 + phi50 + phi84)/3 | eq. (4.2) p. 84; used throughout | Mz of Cintron/Snellen tables (assumed the same definition; not stated there) | presumably |
| Density ratio | a_rho = rho / rho_w | eq. (2.4) p. 15 | `rho` | yes |
| Sound-speed ratio | nu_p = V_p / c_w, V_p the real phase speed; also written V_pR | eq. (2.5) p. 15; (8.24) p. 252 | `nu` | yes |
| Complex speed ratio | a_p = c_p / c_w = nu_p / (1 + i delta_p) | (8.23), (8.25) p. 252 | kappa = (1 + i delta)/nu = 1/a_p | yes, exact |
| Loss parameter | delta_p = k_p'' / k_p' | (8.18) p. 251 | "imaginary / real part of the sediment wavenumber" | yes |
| delta from attenuation | delta_p = alpha_p nu_p c_w ln(10) / (40 pi f), alpha_p in dB/m, f in Hz | (2.6) p. 15; inverse (8.26) p. 252 | `parameters_from_physical`: delta = a c_sed / ((20/ln10) 2 pi f) | identical (c_sed = nu c_w) |
| Attenuation and k_p'' | alpha_p = 20 k_p'' / ln(10) | (8.20) p. 251 | `volume_strength`: alpha = (20/ln10) k delta / nu | yes |
| dB per wavelength | alpha_p_lambda = 40 pi delta_p / ln(10) (= 54.58 delta) | (8.27) p. 252 | `paths.cpp:141`, `propagation.rayleigh_reflection` (REFERENCES 12.4) | yes |
| Q, log decrement | Q_p = 1/(2 delta_p); Delta_p = 2 pi delta_p | (8.28), (8.29) p. 252 | not used | - |
| Hamilton's attenuation factor | k = alpha_p / f_kHz, units dB m^-1 kHz^-1; frequency-independent moduli give alpha proportional to f and no dispersion | p. 10, p. 128 | constant delta = the same assumption | yes |
| Shear ratios | nu_t = shear speed / WATER sound speed; delta_t as for compressional | p. 16 | no shear in the code | absent |
| Shear log decrement | Delta_s = alpha_s V_s ln(10) / (20 f) | (5.7) p. 153 | - | absent |
| Scattering cross section | <|P_s|^2> = |P_i|^2 A sigma / r_s^2, dimensionless, "per unit area per unit solid angle"; S_b = 10 log10 sigma | (2.15), (2.17) p. 23 | `strength` returns sigma; `backscattering_strength_db` = 10 log10 | yes |
| Units of S_b | plain dB, not "dB per unit area" or "dB re 1 m" | p. 23 | - | - |
| Scattered power | U_s = A |P_i|^2 / (2 rho_w c_w) integral over 2 pi of sigma dOmega (the 2 from time averaging) | (2.16) p. 23 | - | - |
| Reciprocity | sigma(theta_i, phi_i, theta_s, phi_s) = sigma(theta_s, phi_s, theta_i, phi_i) | (2.18) p. 24 | held by tests in `test_bottom_models.py`, `test_reverb_bistatic.py` | yes |
| Energy bound | integral of sigma dOmega_s + |V_wwc|^2 sin(theta_i) < sin(theta_i); without coherent part, integral < sin(theta_i) | (2.20), (2.21) p. 24-25 | `test_scattered_energy_stays_within_what_the_flat_bottom_reflects` | yes |
| Lambert | sigma = mu sin(theta_s) sin(theta_i); backscatter mu sin^2(theta); mu < 1/pi | (2.22), (2.23) p. 26 (book cites Lurton 2002 p. 91) | `lambert_backscatter_db`, `LAMBERT_MAX_MU_DB` = -4.97 dB | yes |
| Small-angle limit | backscattering cross section must fall at least as sin^2(theta_i) as theta_i -> 0; a level that does not fall "should be viewed with suspicion" (multipath or noise) | p. 27 | fluid models and Lambert do; empirical laws (`searay_bottom`, `kraus_surface`) not checked | partly |
| Bottom loss | L = -20 log10 |V_ww(theta)| | (2.11) p. 19 | `20 log10 |reflection|` | yes |
| Critical angle | theta_crit = arccos(1/nu_p); nu_p = 1.1 gives 24.6 deg | (2.12) p. 20 | - | arccos(1/1.1) = 24.62 deg |
| Decibels | pressure in dB: 20 log10(|P(t)| / (sqrt 2 p_ref)), p_ref = 1 uPa, the sqrt 2 because power is time-averaged | (H.1) p. 509 | - | - |
| Wavenumbers | magnitude K of horizontal wave vector in rad/length; "spatial frequency" (cycles/length) used elsewhere, factor 2 pi | p. 11 | `w2` with K in rad/m | yes (spectrum conversions are App. D, another part) |
| Volume parameter | sigma_2 = sigma_v / alpha_p, alpha_p in dB/m (Mourad and Jackson 1989); typical 0.0001 to 0.004; above 0.004 multiple scattering may set in | (14.9) p. 381-382 (outside my part) | `sigma2`, same definition | yes |
| Volume term | sigma = |V_wp(theta_i)|^2 |V_wp(theta_s)|^2 sigma_v / (2 k_w |a_rho|^2 Im[P(theta_i) + P(theta_s)]), P = sqrt(a_p^-2 - cos^2 theta) | (14.8), (14.7) p. 381 (outside my part) | `volume_strength`: 2 k_w Im P = 2 Im b2 | same expression; our values for the Fig. 14.2 case (30 kHz, a_rho 1.5, delta 0.015, sigma2 0.002) read on the figure within about 1 dB (ours) |

Typographical slip on p. 23: seawater density "about 1025 kg/m^2" (should be kg/m^3).

### 2.2 Sediment description and grain size (Ch. 3, Ch. 4)

| Item | Value | Where | Status |
|---|---|---|---|
| Size classes (Udden-Wentworth, Folk 1980) | boulder -8 to -12 phi; cobble -6 to -8; pebble -2 to -6; granule -1 to -2; very coarse sand -1 to 0; coarse sand 0-1; medium 1-2; fine 2-3; very fine 3-4; coarse silt 4-5; medium silt 5-6; fine silt 6-7; very fine silt 7-8; clay > 8 phi (3.9 um). Geotechnical practice puts the silt-clay break at 9 phi (2 um) | Table 4.1 p. 78-79; p. 77; p. 8 | confirmed |
| Sand vs mud speeds | muds (< 62 um) "slow", 1-5 % below water; sands (> 62 um) "fast", 10-15 % above | p. 15 | confirmed |
| Sorting / skewness / kurtosis (Folk and Ward 1957) | sigma_I = (phi84 - phi16)/4 + (phi95 - phi5)/6.6; Sk_I and K_G formulas; descriptive classes (e.g. well sorted 0.35-0.50) | (4.3)-(4.5) p. 85; Table 4.3 p. 86 | confirmed |
| SAX99 medium sand | Mz 1.27 phi, sorting 0.70, 98 % sand | Table 4.2 p. 84 | confirmed |
| Phase relations | rho = rho_w beta + rho_g (1 - beta); e = beta/(1 - beta); w, e, beta, rho mutual conversions | Table 4.5 p. 93; (4.6)-(4.9) | confirmed |
| Salt correction of porosity | beta = beta0 + 0.036 (S_p/35) - 0.0224 (S_p/35) beta0; or x 1.012 for high porosity at 35 ppt | (4.10) p. 93 | confirmed |
| Grain and water densities for bulk density | 2650 and 1000 kg/m3 "adequate" | p. 97 | confirmed |
| Relative density of sands | D_r = 100 (e_max - e)/(e_max - e_min); natural sands 65-85 %; SAX99 bulk density 2074 kg/m3, D_r 75 % | (4.11) p. 94 | confirmed |
| Free gas | methane in organic muds rarely > 1 % of pore volume, 4-5 % reported; swash-zone saturation < 75 %, subtidal rarely < 95 % | p. 94 | confirmed |
| Permeability ranges | clayey 1e-18 to 1e-15 m^2 (beta 0.5-0.9); silty 1e-16 to 1e-12 (0.35-0.5); sandy 1e-14 to 1e-10 (0.35-0.5); gravelly 1e-10 to 1e-7 (0.25-0.4) | Table 4.6 p. 102 | confirmed |
| Darcy, Kozeny-Carman | V = -(K/(g rho_f)) grad P; K = kappa g rho_f / mu; kappa = e^3 / (k0 T^2 S0^2 (1 + e)) | (4.13)-(4.16) p. 100-102 | confirmed |
| Archie-type porosity from resistivity | F = a beta^-m; examples F = 1.30 beta^-1.45 (Bering Sea), beta^-1.5 sands, beta^-2.0 clays; m 1.2-1.9 by grain shape; "not recommended" as a universal law | p. 96-97 | confirmed |
| Grain density and bulk modulus | quartz 2.65 g/cm3, 36-40 GPa; feldspars 2.50-2.80, 37.5; clay minerals 2.35-3.00, 6-50 GPa; calcite 2.71-2.72, 64-75; aragonite 2.93-2.95, 45; dolomite 2.72-2.87, 70-90. Measured coastal grain densities 2350-2800 kg/m3; quartz sands near 2650; carbonates 2700-2800 | Table 4.7 p. 112; p. 114 | confirmed |
| Grain bulk modulus debate | handbook quartz 36-40 GPa vs proposed 7 GPa; measured 38-47 GPa (95 % limits 30-57) for quartz sands: use handbook values. Wood's equation with 2650 and 1023 kg/m3 and K_w = 2.4 GPa shows up to 150 m/s difference | p. 115-116, Fig. 4.17 | confirmed (text p. 116 read in the text file) |
| Clay bulk modulus | measurements "show little agreement", 6-50 GPa | p. 117 | text read |
| Pore fluid (Biot inputs) | Stoll: structure factor 1.25 sand, 3.0 fine; pore size 1/6-1/7 of mean grain size; viscosity 0.001 kg m^-1 s^-1; pore water density 1000 kg/m3 often assumed | p. 119-120 | confirmed |
| Shear speeds of muds | generally below 50 m/s | p. 118 | text read |
| Ripples (Wiberg and Harris 1994) | orbital ripples (d0/D < 2000): lambda = 0.62 d0, eta/lambda = 0.17; anorbital (d0/D > 5000): lambda = 500 D in the text but 535 D in Fig. 3.24; ripples form for grains below 0.8 mm, wavelengths 10-200 cm | p. 59 (text), p. 62-63 (image) | confirmed (constant inconsistent, Sec. 6) |
| Bioturbation | mixing coefficient D_b 10-100 cm^2/yr, mixing depth L_b 10-30 cm | p. 70 (text) | text read |
| Bubbles in sediment (3.3) | three types (interstitial, reservoir, displacement); Type III methane bubbles most common in experiments; gas volume usually < 1 %, 6-9 % in pockmarks, up to 12 %; CT-resolved radii 0.42-10 mm, acoustic evidence of 0.25 mm; oblate, vertically oriented, "corn flake" shapes; total volume alone is "insufficient" to model acoustics (size, shape and distribution needed) | p. 51-57 | text read; Sect. 14.1.7 (scattering) is another part |

### 2.3 Compressional speed and attenuation regressions (Ch. 5) - all confirmed on the page images

Data behind them:
- **Core data set** (Sects. 5.1.3-5.1.4): about 800 cores from 67 shallow sites (55 siliciclastic, 12 carbonate),
  about 4500 points (3922 siliciclastic, 621 carbonate), upper 30 cm, Richardson-Briggs time-of-flight method at
  **400 kHz** on 5.9 cm cores (eqs. (5.1)-(5.2) p. 128). Ratios are to pore water at the same T, S, P; Hamilton's
  laboratory standard is 23 deg C, 35 ppt, 1 atm = c_w 1529.4 m/s, rho_w 1024 kg/m3 (p. 128). Site means in
  Tables 5.1-5.2 (p. 132-134; not reproduced here: 57 siliciclastic and 12 carbonate sites, Mz 0.08 to 10.94 phi,
  V_pR 0.968 to 1.163, rho 1.17 to 2.07 g/cm3, k 0.097 to 1.74 dB/m/kHz).
- **In situ data set** (Sect. 5.1.6): ISSAMS probes at 87-88 sites, **38 or 58 kHz**, path 30-100 cm, 5-30 cm deep;
  regressions on site averages.

Table 5.3 (p. 136), core data, 400 kHz. Validity printed: 1100 < rho (kg/m3) < 2200, 30 < eta (%) < 95,
0 < Mz < 12. rho in g/cm3 in the formulas, eta = porosity in percent, k in dB m^-1 kHz^-1.

| Type | Regression | N | r^2 |
|---|---|---|---|
| siliciclastic | V_pR = 1.603 - 0.0156 eta + 0.0001 eta^2 | 3905 | 0.95 |
| carbonate | V_pR = 11.760 - 0.0206 eta + 0.0001 eta^2 (as printed; 1.760 must be meant, Sec. 6) | 609 | 0.91 |
| all | V_pR = 1.606 - 0.0158 eta + 0.0001 eta^2 | 4514 | 0.95 |
| siliciclastic | V_pR = 1.585 - 0.8991 rho + 0.3352 rho^2 | 3905 | 0.94 |
| carbonate | V_pR = 1.878 - 1.2289 rho + 0.4232 rho^2 | 609 | 0.90 |
| all | V_pR = 1.649 - 0.9807 rho + 0.3595 rho^2 | 4514 | 0.93 |
| siliciclastic | V_pR = 1.184 - 0.0288 Mz + 0.0008 Mz^2 | 2392 | 0.82 |
| carbonate | V_pR = 1.161 - 0.0308 Mz + 0.0013 Mz^2 | 371 | 0.82 |
| all | V_pR = 1.184 - 0.0307 Mz + 0.0010 Mz^2 | 2763 | 0.82 |
| all | k = 0.74 - 0.07 Mz - 0.02 Mz^2 (as printed: negative beyond 4.5 phi, unusable, Sec. 6) | 2653 | 0.10 |
| all | k = -1.121 + 0.066 eta + 0.0006 eta^2 (as printed: 9.7 at eta 90 against data near 0.1, unusable) | 4391 | 0.19 |

The book's own judgment (p. 136): attenuation at 400 kHz is "very poorly correlated" with grain size or porosity
and the attenuation regressions "have little predictive value" at that frequency; the 400 kHz values include
scattering by shells and heterogeneity.

Table 5.4 (p. 140), in situ, 38/58 kHz. Validity: 1200 < rho (kg/m3) < 2100, 30 < eta < 90, 0 < Mz < 10. Here
rho is in kg/m3 (the caption says g/cm3; the coefficients only make sense in kg/m3 - checked: rho = 2000 gives
V_pR 1.10, matching Fig. 5.7).

| Predictor | Regression | N | r^2 |
|---|---|---|---|
| density | V_pR = 1.705 - 1.035e-3 rho + 3.664e-7 rho^2 | 86 | 0.92 |
| porosity | V_pR = 1.576 - 0.015677 eta + 1.0269e-4 eta^2 | 86 | 0.91 |
| mean grain size | V_pR = 1.190 - 0.03956 Mz + 1.9476e-3 Mz^2 | 86 | 0.92 |
| density | k = 0.00332 e^(0.00241 rho) | 87 | 0.45 |
| porosity | k = 2.153 e^(-0.0401 eta) | 87 | 0.43 |
| mean grain size | k = 0.697 e^(-0.183 Mz) | 87 | 0.52 |

In situ speed ratios run from 0.978 (Eckernfoerde silty clay, 1428 m/s) to 1.170 (North Sea coarse sand,
1767 m/s) (p. 140). Lab against in situ (Fig. 5.8, 78 sites): k(lab) = 1.3276 k(in situ), r^2 = 0.16;
V_p(lab) = -0.0154 + 1.0187 V_p(in situ), r^2 = 0.089 (p. 142). Single-source time-of-flight may read 10-15 m/s
low in sands (p. 140). Attenuation in situ is "often lower" than at 400 kHz and, unlike Hamilton (1972), does not
peak in fine sand-silt; it rises with grain size (p. 140-142).

Comparison with Hamilton and Bachman (1982) (Sect. 5.1.7, Fig. 5.10, Table 5.5): Hamilton-Bachman predict higher
speed for sands; at SAX99 (measured 1766.3 m/s, ratio 1.154, pore water 1536 m/s) Bachman's regressions give
1847/1830/1880 m/s from porosity 37 %/density 2074/Mz 1.25 phi, Richardson-Briggs 1786/1785/1764 (ratios 1.163,
1.162, 1.149). The book recommends the Richardson-Briggs regressions for high-frequency work (p. 145). Check (ours):
the Richardson-Briggs numbers of Table 5.5 are Table 5.3's **siliciclastic** rows (1.1627, 1.1621, 1.1493).

Gradients (Sect. 5.1.5): Hamilton's speed gradients 20 s^-1 (sand), 1 s^-1 (mud) over tens of metres; in the top
20-30 cm sands show 50 s^-1 or more and attenuation may double; do not extrapolate beyond 0-30 cm (p. 137).

Index of impedance (Sect. 5.1.8): IOI = rho (g/cm3) x V_pR, independent of pore-water T, S, P (p. 146-147).
Table 5.6 (p. 150), core data; validity 1.05 < IOI < 2.6 (printed with units "kg m^-1 s^-1", but IOI is in g/cm3 as
defined; Sec. 6). SG = percent sand plus gravel.

| Type | Regression | r^2 |
|---|---|---|
| siliciclastic / carbonate / all | V_pR = 1.149 - 0.2821 I + 0.1203 I^2 / 1.164 - 0.3001 I + 0.1253 I^2 / 1.164 - 0.3001 I + 0.1253 I^2 | 0.97 / 0.96 / 0.97 |
| siliciclastic / carbonate / all | k = -2.61 + 3.41 I - 0.885 I^2 / -5.96 + 6.94 I - 1.174 I^2 / -3.31 + 4.33 I - 1.138 I^2 | 0.16 / 0.43 / 0.22 |
| siliciclastic / carbonate / all | eta = 178.60 - 94.60 I + 14.86 I^2 / 186.18 - 102.20 I + 17.29 I^2 / 174.16 - 89.12 I + 13.37 I^2 | 0.99 each |
| siliciclastic / carbonate / all | rho = 1.01 + 1.22 ln I / -0.52 + 1.81 I - 0.305 I^2 / 1.02 + 1.21 ln I | 0.99 each |
| siliciclastic / carbonate / all | Mz = 17.7 - 6.8 I / 19.3 - 7.6 I / 17.9 - 6.0 I | 0.85 / 0.75 / 0.84 |
| siliciclastic / carbonate | SG = -109.6 + 87.7 I / -143.2 + 101.4 I | 0.82 / 0.73 |

Scatter about these: bands of 14 m/s, 0.001 (ratio), 0.04 g/cm3 and 4 % porosity for carbonate vs siliciclastic
(p. 147); up to 0.6 phi and 17 % SG between the two sets. Dispersion warning (p. 147): at SAX99 speeds were up to
75 m/s higher at 400 kHz than at 3.5 kHz; using 3.5 kHz IOI would change predicted porosity +2.6 %, density
-50 kg/m3, Mz +0.69 phi. Grain size is a weak index (Fig. 5.14): at one Mz, porosity varies about 10 % in sands
(packing) and much more in silts and clays (sorting, compaction), p. 150-151.

Bulk density (Sect. 5.5, p. 167-169), core data, siliciclastic and carbonate: **rho = 2.17 - 0.082 Mz** (g/cm3,
eq. (5.14)); inverse Mz = 22.11 - 9.82 rho (eq. (5.15)); r^2 = 0.80 each; Fig. 5.28 shows the 95 % bounds about
+-0.23 g/cm3 (read from the figure, approximate). "Grain size is not an accurate predictor of bulk density".

### 2.4 TR 9407 grain-size relations as printed in Ch. 11 (outside my chapters; needed for the "wanted" row)

Book p. 313-314, eqs. (11.1)-(11.3), "first given by Mourad and Jackson 1989 and expanded in" TR 9407. Confirmed
on the page images.

    a_rho = 0.007797 Mz^2 - 0.17057 Mz + 2.3139                          -1 <= Mz < 1
          = -0.0165406 Mz^3 + 0.2290201 Mz^2 - 1.1069031 Mz + 3.0455      1 <= Mz < 5.3
          = -0.0012973 Mz + 1.1565                                        5.3 <= Mz <= 9      (11.1)
    nu_p  = 0.002709 Mz^2 - 0.056452 Mz + 1.2778                          -1 <= Mz < 1
          = -0.0014881 Mz^3 + 0.0213937 Mz^2 - 0.1382798 Mz + 1.3425      1 <= Mz < 5.3
          = -0.0024324 Mz + 1.0091   (as printed; 1.0019 is continuous)   5.3 <= Mz <= 9      (11.2)
    k     = 0.4566                                   -1 <= Mz < 0
          = 0.0245 Mz + 0.4556                        0 <= Mz < 2.6
          = 0.1245 Mz + 0.1978                        2.6 <= Mz < 4.5
          = 0.20098 Mz^2 - 2.5228 Mz + 8.0399         4.5 <= Mz < 6.0
          = 0.0117 Mz^2 - 0.2041 Mz + 0.9431          6.0 <= Mz < 9.5
          = 0.0601                                    9.5 <= Mz                               (11.3)

k in dB m^-1 kHz^-1; delta_p follows from (2.6) with alpha_p = k f_kHz:
delta_p = k nu_p c_w ln(10) / (40 000 pi), c_w in m/s. The book says (p. 314): (11.1)-(11.2) are adjustments of
Hamilton and Bachman (1982) on limited data and "should not be regarded as geoacoustic regressions", rather
"empirical expressions intended to improve acoustic model-data fits"; (11.3) is "taken directly from [Hamilton
1972]". They give lower speed and density ratios than the Ch. 5 regressions for fine sediments, possibly because
bioturbation raised the porosity of the top centimetres at the original sites (not supported by the Ch. 5 data).
The relations are frequency independent (linear-in-f attenuation, no dispersion) and fluid; the book notes that
including shear would change the predicted bottom loss negligibly for these sediments (p. 314, citing Sect. 9.5).

### 2.5 Shear waves (Sect. 5.2) - confirmed

| Relation | Formula | Where | Fit and range |
|---|---|---|---|
| Mean effective stress | sigma_0' = (1 + 2 K_0)/3 x P_0; K_0 from 0.4 (dense sand) to 1.0 (very soft mud) | (5.4) p. 153 | - |
| Shear modulus (Bryan and Stoll 1988, laboratory) | mu = 2526 (sigma_0')^0.448 e^(-1.504 e), e void ratio | (5.5) p. 153 | units of sigma_0' and mu not stated on the page (Sec. 6) |
| Shear speed | V_s = sqrt(mu / rho) | (5.6) | - |
| V_s vs porosity, all 87 sites (interparticulate porosity for Florida Keys carbonates) | V_s = 692 e^(-0.0494 eta) | Fig. 5.16 p. 157 | r^2 0.86; siliciclastic alone 770 e^(-0.0517 eta) (89 %), carbonate 656 e^(-0.0405 eta) (67 %), Fig. 5.15 |
| V_s vs bulk density (kg/m3) | V_s = 0.3823 e^(0.00284 rho) | Fig. 5.17 p. 158 | r^2 0.85 |
| V_s vs Mz | V_s = 141 - 16.65 Mz + 0.455 Mz^2 | Fig. 5.18 p. 158 | carbonate and siliciclastic nearly identical |
| V_s vs void ratio | V_s = 59.7 e^(-1.12 e) | Fig. 5.20 p. 160 | r^2 0.86 |
| V_s vs V_p (in situ) | V_s = -481 + 0.348 V_p | Fig. 5.25 p. 165 | r^2 0.85, 87 sites |
| Depth dependence, upper metre, d in m | V_s = 992 e^(-0.0494 eta) d^0.3; 0.549 e^(0.00284 rho) d^0.3; (202.5 - 23.9 Mz + 0.6528 Mz^2) d^0.3; 85.7 e^(-1.12 e) d^0.3 | (5.10)-(5.13) p. 160 | the book prefers these to Hamilton (1976, 1987) and to (5.5); in situ moduli 1.3-2.5 x laboratory (Stoll 1989), p. 161 |
| Older void-ratio forms (Richardson et al. 1991) | V_s = 55.3 e^(-1.1394 [e?]); V_s = 79.5 e^(-1.1394 e) d^0.3 | (5.8), (5.9) p. 159 | (5.8) printed without the void ratio in the exponent (Sec. 6) |
| Shear attenuation factor, 19 sites, k_s in dB m^-1 kHz^-1 | k_s = 42.2 + 2.2 eta; k_s = 20.2 + 9.23 Mz; k_s = 321 - 0.136 rho (kg/m3) | Fig. 5.21 p. 162 | sparse; k_s much larger than k (25-200 vs 0.1-1.7) |
| Hamilton's shear log decrement | 0.3 +- 0.15 (sand), 0.2 +- 0.10 (mud); Richardson's values slightly higher, 0.1-0.6 | p. 161, Fig. 5.22 | - |
| North Sea sand profile (1 kHz) | V_s = 72.3 + 232 d or 186.5 d^0.27 (preferred); alpha_s = 21.7 + 96 d or 67.1 d^0.31 dB/m | Figs. 5.19, 5.23 p. 159, 164 | single site |
| General facts | measured range 7.7 m/s (Eckernfoerde mud) to 150 m/s (shelly coarse sand); depth gradients follow power laws with exponent 0.25-0.33; in situ V_s 15-60 % above laboratory-derived values; shear attenuation 1-2 orders above compressional | p. 155-156 | - |

Elastic parameters as the book frames them: five (rho, V_p, alpha_p, V_s, alpha_s) or nu_p, nu_t, delta_p, delta_t
plus a_rho (p. 16).

### 2.6 Slow wave, dispersion, frequency dependence (Sects. 5.3, 5.4; App. I) - confirmed

- Biot slow wave not observed in field sediments; poroelasticity can lower reflection and scattering in sands by
  1-3 dB, well approximated by the effective density fluid model (Sect. 10.2, another part) (p. 161-165).
- SAX99 (Figs. 5.26-5.27, from Williams et al. 2002a): speed ratio rises from about 1.04-1.06 at 0.1-0.4 kHz to
  1.12 near 1-2 kHz and about 1.16 above 20 kHz, nearly flat to 400 kHz; attenuation rises from about 1.5 dB/m at
  2-3 kHz to about 10 dB/m at 40 kHz, 30-60 dB/m at 100-250 kHz and about 170 dB/m at 400 kHz (values read off
  log-scale figures: approximate). The book judges direct measurements or empirical relations the best estimates;
  theories (Hamilton, Biot-Stoll, Buckingham) need inputs that are hard to get (p. 166-167).
- Causality (App. I): with exp(-i w t), the complex wavenumber is analytic in the upper half w-plane; real and
  imaginary parts of the refractive index n = c0 k / w obey Kramers-Kronig relations (I.22)-(I.23); for a
  non-vanishing index use one subtraction (I.25). With n'' = delta_1 constant (attenuation linear in f), (I.28):

      n'(w) = 1 - (2 delta_1 / pi) ln|w / w1|,    i.e. phase speed c(w) ~= c(w1) [1 + (2 delta_1/pi) ln(w/w1)]

  A constant complex speed (our model) "violates causality", but "leads to negligible error in most cases of
  interest" (p. 517, citing Thorsos et al. 2000a). Wingham (1985) found (I.28) matches sand data at 50-350 kHz;
  Williams et al. (2002a) find it holds above several kHz with more dispersion below (p. 517).
- Research Issues 5.7 (p. 170): no internally consistent set of speed and attenuation measured at one place over a
  few hundred Hz to hundreds of kHz exists; measurement accuracies are rarely reported; attenuation measurements
  mix intrinsic loss, conversion and scattering; the high attenuation above 25 kHz is "sometimes ascribed" to
  scattering "but this has not been shown convincingly"; shear attenuation data sparse; anisotropy of shear
  neglected; the importance of poroelasticity for reflection and scattering is disputed.

### 2.7 Rock (Sect. 5.6, p. 169) - confirmed

Saturated low-porosity (< 1 %) rock: basalt 2870 kg/m3, granite 2660; sandstones and siltstones (10-30 %
porosity) 2100-2500; calcareous reef 2000-2950 (shell material 2700-2950). Compressional speed: igneous and
metamorphic about 5-6 km/s; sedimentary rock 2-6 km/s (lower at higher porosity); low-porosity carbonate up to
5.5-6.0 km/s, much less for porous reef. Shear speed roughly 40-60 % of compressional (Poisson's ratio 0.2-0.4;
unconsolidated sediments near 0.5). Attenuation in rock: frequency dependent, "beyond the scope". No roughness or
scattering parameters for rock in this chapter. Sect. 2.1.3 (p. 13): no high-frequency experiment has
characterised the random heterogeneity of rock or coral, "yet models predict that this could be an important cause
of scattering" (Sect. 14.2.3, another part).

### 2.8 Seawater (App. B) - confirmed

| Quantity | Book | Where | Stated range and accuracy |
|---|---|---|---|
| Density, 1 atm | rho(S,T,0) = rho(0,T,0) + S(0.824493 - 4.0899e-3 T + 7.6438e-5 T^2 - 8.2467e-7 T^3 + 5.3875e-9 T^4) + S^1.5(-5.72466e-3 + 1.0227e-4 T - 1.6546e-6 T^2) + 4.8314e-4 S^2; pure water 999.842594 + 6.793952e-2 T - 9.095290e-3 T^2 + 1.001685e-4 T^3 - 1.120083e-6 T^4 + 6.536332e-9 T^5 | (B.2), (B.3) p. 456 | EOS 80 (UNESCO 1980): -2 to 40 deg C, 0-42 ppt, 0-1000 bar, 0.05 kg/m3 (p. 457) |
| Density with pressure | rho(S,T,P) = rho(S,T,0) / (1 - P/K(S,T,P)), P in bar above 1 atm, K the secant bulk modulus polynomial (B.4) | (B.1), (B.4) p. 456-457 | as above; coastal composition anomalies up to 0.05 kg/m3 (p. 120) |
| Sound speed, Del Grosso (1974), 19 terms | c_w = 1402.392 + 5.01109398873 T - 5.50946843172e-2 T^2 + ... (B.5), T deg C, S ppt, P "absolute pressure (i.e., includes atmospheric pressure) (kg m^-2)" [sic: kg/cm^2 elsewhere on the same page] | (B.5) p. 459-460 | Del Grosso deteriorates for pure water under pressure and outside 33-38 ppt; Del Grosso vs Chen-Millero < 0.3 m/s on shelves, < 0.12 m/s with Millero-Li corrections (p. 459) |
| Sound speed, Medwin (1975) | V_p = 1449.2 + 4.6 T - 0.055 T^2 + 0.00029 T^3 + (1.34 - 0.01 T)(S - 35) + 0.016 D | (4.26) p. 121 | < 1 m/s from Chen-Millero for 0-40 deg C, 0-40 PSU, 0-500 m |
| Absorption, Francois-Garrison (1982b) | alpha = 1e-3 [A1 P1 f1 f^2/(f^2 + f1^2) + A2 P2 f2 f^2/(f^2 + f2^2) + A3 P3 f^2] dB/m, f kHz; A1 = 8.86/c 10^(0.78 pH - 5); A2 = 21.44 S/c (1 + 0.025 T); P1 = 1; P2 = 1 - 1.37e-4 z + 6.2e-9 z^2; P3 = 1 - 3.83e-5 z + 4.9e-10 z^2; f1 = 2.8 (S/35)^0.5 10^(4 - 1245/(273 + T)); f2 = 8.17 10^(8 - 1990/(273 + T)) / (1 + 0.0018 (S - 35)); c = 1412 + 3.21 T + 1.19 S + 0.0167 z; A3 = 4.937e-4 - 2.59e-5 T + 9.11e-7 T^2 - 1.50e-8 T^3 (T <= 20), 3.964e-4 - 1.146e-5 T + 1.45e-7 T^2 - 6.5e-10 T^3 (T >= 20) | (B.6)-(B.12) p. 461 | open ocean S 30-40 ppt; "may not be appropriate" for low-salinity estuaries; suspended particles (0.1-1 kg/m^2 [sic]) can add attenuation comparable to MgSO4 at 1-100 kHz (p. 462) |
| Viscosity | Richards (1998) after Matthaeus: mu (Pa s) polynomial in T, S, P (dbar) | (B.16) p. 465 | distilled 1.80 cP at 0 deg C to 0.72 at 35; seawater 2-7 % higher |
| Conductivity, salinity | Table B.1 (S/m vs T, S), PSS 78 (B.17) | p. 466-468 | not used by us |

The text on p. 460 names f1 and f2 as the MgSO4 and boric acid frequencies "respectively"; the formulas (B.6),
(B.9) make f1 the boric acid one (the 1245 K term). The formulas are what we follow.

### 2.9 Coefficients of variation (App. C, Table C.1, p. 471-473) - confirmed

Within-site (core-to-core and depth) coefficients of variation for V_p, alpha_p, Mz, porosity beta, bulk density
rho and grain density rho_g at 64 siliciclastic and 14 carbonate sites (400 kHz cores; the units are not stated -
the values are evidently percent). Not reproduced; summary (read from the table):

| Parameter | Sands (Mz < 3 phi) | Silts and clays (Mz > 5 phi) |
|---|---|---|
| CV of V_p | mostly 0.3-2.4 %, up to 3.8 % | mostly 0.1-2 %, up to 4.3 % |
| CV of alpha_p | mostly 17-48 %, up to 83 % | mostly 12-42 %, up to 89 % |
| CV of porosity | mostly 2-6.5 %, up to 22 % (TBay/crse) | mostly 3.5-9 %, up to 21 % |
| CV of bulk density | mostly 0.5-3 %, up to 7.5 % | mostly 2-6 %, up to 18 % |
| CV of grain density | mostly 0.2-1.5 %, up to 2.9 % | 0.4-4 % |
| CV of Mz (phi) | meaningless near Mz = 0 (366 % at 0.08 phi) | 2-36 % |

The ratio and percent-porosity CVs equal those of the quantities they come from (caption). These are the input for
continuous-heterogeneity volume scattering (Ch. 7, 14) and an honest statement of within-site spread; between-site
(class) scatter is the regressions' scatter in Sec. 2.3.

### 2.10 Research Issues of Chs. 3-5 (summary in our words)

- Ch. 3 (p. 72): predict roughness (ripple growth by waves, decay by animals) and heterogeneity (bioturbation
  mixing) from hydrodynamic and biological models - the forward problem; then invert acoustics for those processes;
  use microstructure in wave theory (poroelastic problems).
- Ch. 4 (p. 122): whether destructive grain-size methods suit acoustics; effect of grain shape on wave speeds is
  speculative; relative density of sands generally unknown; 1 % porosity changes speed noticeably; bulk moduli of
  clays and carbonates unknown; pore water may differ from overlying seawater.
- Ch. 5 (p. 170): see Sec. 2.6 above.

## 3. Against our simulator

### 3.0 What was run (all in the scratchpad, on a copy of `python/gpusonar` with the built module linked; the
worktree was not touched)

`.../scratchpad/jr01/cmp.py`, `cmp2.py`, `cmp3.py`, `cmp4.py` (grain-size relations), `vol.py` (volume term vs
Fig. 14.2), `sw.py`, `rho.py` (seawater), `elastic.py` (shear effect; our own fluid-elastic reflection coefficient),
`kk.py` (causal dispersion size), `ms.py` (medium sand vs SAX99). Results quoted below.

### 3.1 Conventions: agree

Every definition in Sec. 2.1 that our code uses agrees: grazing angles, exp(-i w t) with Im k > 0, a_rho, nu_p,
delta_p (2.6)/(8.18)/(8.25)-(8.27), sigma and S_b, reciprocity, the energy bound, Lambert and mu < 1/pi, sigma_2
and the volume expression (14.8). Our volume term for the Fig. 14.2 case (nu 0.98/1.05/1.1) gives -24.1/-23.6/
-23.3 dB at 90 deg and -33.4/-35.3/-40.2 dB at 10 deg; the figure reads about -24, -23.5 and -33, -36, -40 dB
(by eye). The Hamilton/TR 9407 attenuation-to-delta step needs c_w = **1528 m/s** to reproduce our table's delta
(0.01624 at 1.5 phi exactly): the table was built with the same reference water as UnderwaterAcoustics.jl.

### 3.2 Grain-size relations (`bottom.GRAIN_SIZE_TABLE`, `sediment_from_grain_size`; REFERENCES Sec. 3.1, Sec. 10)

| Comparison | Result (ours, `cmp.py`, `cmp4.py`) |
|---|---|
| Table rows vs eqs. (11.1)-(11.3) at the 20 half-phi nodes | rho within 4.5e-4; nu within 4.9e-5 when the Mz >= 5.3 constant is 1.0019, but 0.0072 off with the printed 1.0091 (all eight rows 5.5-9 phi); delta within 0.27 % (c_w 1528) |
| Continuity at 5.3 phi | cubic branch gives nu = 0.98902; linear branch gives 0.98901 with 1.0019, 0.99621 with 1.0091. The printed constant breaks continuity; the table's three earlier prints all follow 1.0019 |
| Effect of the misprint if copied | normal-incidence reflection 0.47-0.53 dB too low (more loss) at 5.5-9 phi |
| Our linear interpolation vs the polynomials (fine grid, 1.0019) | -1..1 phi: < 0.004 dB; 1..2: rho off up to 0.0106 (1.25 phi), 0.07 dB; 2..5: 0.10 dB, delta up to 2.4 %; **5..6: 0.68 dB at 5.3 phi**, nu 0.0041, delta 7.6 %; 6..9: 0.02 dB, delta up to 5.7 % |
| Fig. 11.2 (bottom loss at normal incidence vs Mz) | our fluid reflection with (11.1)-(11.3) gives 6.1, 6.9, 10.6, 14.8, 18.5, 22.2 dB at 0..5 phi; the figure's "TR 9407" curve reads about 6, 7, 10.5, 15, 18.5, 22 (by eye). With Table 5.4 nu, eq. (5.14) rho over 1.024 and Table 5.4 k: 7.3 dB at 0 phi to 14.3 at 8 phi, which is the figure's "standard" curve |
| TR 9407 table vs the book's "standard" regressions | normal-incidence loss differs by -1.4 dB (gravel, -1 phi: ours lower loss) to +12.5 dB (5.5 phi: ours higher loss); rho off by up to 0.57 (4 phi), nu by up to 0.088 (0 phi), delta by up to 140 % (4.5 phi) |
| Medium sand row vs SAX99 cores (Table 5.1: Mz 1.27, rho 2.066, ratio 1.154, k 0.444 at 400 kHz) | our medium_sand (1.845, 1.178, 0.0162) vs (2.018, 1.154, 0.0144): reflection loss 0.66 dB higher at 90 deg, 0.18 dB at 45 deg, 0.65 dB lower at 30 deg (`ms.py`) |

Status for REFERENCES: our table is now **confirmed against the book's print of the TR 9407 relations** (p. 313-
314), except the nu constant for Mz >= 5.3, where the book's print is taken as a misprint (evidence: continuity
and three independent prints). It is still the relations' provenance that matters: the book calls them ad hoc
model-fit relations, not property predictions.

### 3.3 Seawater (`acoustics.py`; REFERENCES Sec. 7)

| Ours | Book | Check (ours) | Verdict |
|---|---|---|---|
| `water_density` (one atmosphere, coefficients via arlpy/UnderwaterAcoustics.jl) | (B.2), (B.3) | every coefficient identical | **confirmed**; the book adds the pressure term (B.1)/(B.4): +0.45 kg/m3 (0.044 %) at 100 m, +2.25 at 500 m, +4.6 at 1000 m (10 deg C, 35 PSU). Our transcription of (B.4) reproduces the UNESCO check values K(35, 25, 1000 bar) = 27108.95 and K(0, 5, 0) = 20337.80 bar |
| `absorption_francois_garrison` | (B.6)-(B.12) | every constant identical, incl. 273 + T, the two A3 branches, the boric term | **confirmed**; stated scope S 30-40 ppt (our code applies it down to 5 PSU and drops boric at S = 0: ours) |
| `absorption_ainslie_mccolm` | named only (p. 461) | - | unchanged (via secondary) |
| `sound_speed` (Mackenzie 1981) | not printed; Medwin (4.26) and Del Grosso (B.5) are | max difference 0.25 m/s to Medwin and 0.39 m/s to Del Grosso (P absolute) / 0.22 (P gauge) over 0-30 deg C, 30-40 PSU, 0-500 m | Mackenzie stays "unconfirmed in print"; the REFERENCES Sec. 10 row "Sound speed formula ranges" is not settled by this book. Del Grosso and Medwin can serve as independent check oracles |
| `absorption_thorp` | not in the book | - | - |

### 3.4 Sediment geoacoustics we do not model

| Topic | Book | Ours | Size of the omission (ours) |
|---|---|---|---|
| Shear in sediments | Sect. 5.2 regressions; V_s 8-150 m/s in unconsolidated sediment | absent (fluid only) | fluid-elastic reflection, medium sand with V_s 120 m/s: < 0.1 dB at every angle; sandy gravel with an assumed 250 m/s: < 0.2 dB. Negligible for reflection, as the book also says (p. 314). Elastic effects on roughness scattering are Sect. 13.2.2 (another part) |
| Shear in rock | Sect. 5.6: V_s 40-60 % of V_p | `rock`, `pebbles` reflect as fluids (nu 2.5 and 1.8) | our rock row as a fluid has near-total reflection up to its compressional critical angle (66 deg); with V_s = 0.5 V_p or Olson's generic rock (3600/1900/2500) the loss is **4.8-5.5 dB at 40-60 deg grazing** (energy into the shear wave above the shear critical angle, 37 deg) and equal at normal incidence |
| Rock compressional speed | igneous/metamorphic 5-6 km/s, sedimentary 2-6 | `rock` 3820 m/s, 2557.5 kg/m3 (ratios 2.5/2.5 on 1528/1023) | a sedimentary-rock value; granite (2660, about 5500 m/s) would raise the normal-incidence reflection by about 0.9 dB (|R| 0.72 -> 0.81) and move the critical angle from 66 to 74 deg (ours, hand calculation) |
| Pebbles / gravel | no gravel geoacoustic data in the book (the core set has few coarse samples, p. 131); no gravel Lambert value in my chapters | `pebbles` 2750 m/s, Lambert -9 dB above 150 kHz (unconfirmed) | not settled here |
| Gas in sediment | Sect. 3.3: < 1 % typical, up to 6-12 %; bubble radii 0.25-10 mm; shape matters | absent | not quantifiable from my chapters (scattering is Sect. 14.1.7) |
| Dispersion | (I.28) for constant delta; SAX99 measured | none: nu and delta constant in f | a causal constant-delta sand (delta 0.0162) gains 2.4 % speed per decade: normal-incidence reflection +0.24 dB, critical angle 31.9 -> 34.0 deg per decade (`kk.py`); very fine sand 2.75 %/decade (+0.75 dB); clay 0.2 %. Measured SAX99 dispersion is concentrated below about 20 kHz (Fig. 5.26), so between 20 and 400 kHz our constant ratio is closer to the data than (I.28) |
| Depth gradients | top 30 cm: speed gradients >= 50 s^-1 in sand, attenuation may double | homogeneous half-space | affects penetration and volume term at the higher frequencies; not quantified |
| Frequency dependence of attenuation | Fig. 5.27: SAX99 alpha grows faster than linear above about 100 kHz (to 170 dB/m at 400 kHz); Hamilton's k is linear | delta constant (alpha linear in f) | sigma_2 = sigma_v/alpha_p makes the volume term almost frequency-independent in either case (p. 382); reflection changes little since delta is small |

### 3.5 REFERENCES.md rows touched (proposed status changes, for the orchestrator)

| Row | Now | Proposed |
|---|---|---|
| Sec. 10 "Regressions of density, speed and loss against grain size" (wanted) | wanted | **found**: Jackson and Richardson (2007) eqs. (11.1)-(11.3), p. 313-314 (attributed there to Mourad and Jackson 1989 and TR 9407); `GRAIN_SIZE_TABLE` confirmed against it except the nu constant for Mz >= 5.3 (book 1.0091, ours 1.0019; the book's print breaks continuity) |
| Sec. 3.1 UnderwaterAcoustics.jl / Cintron / Snellen rows (rho, nu, delta) | via secondary | add: confirmed against the relations as printed in J&R (2007) p. 313-314 (still TR 9407's relations, called "ad hoc" by the book) |
| Sec. 10 "Medium sand: volume parameter 0.001 or 0.002" | wanted | unchanged: the book prints no per-class sigma2; it gives the range 0.0001-0.004 (p. 382) |
| Sec. 7 "Francois and Garrison (1982)" | via secondary | **confirmed** (J&R App. B eqs. (B.6)-(B.12), p. 461; open-ocean scope 30-40 ppt, p. 462) |
| Sec. 7 `water_density` (arlpy, UnderwaterAcoustics.jl) | via secondary | **confirmed** (J&R eqs. (B.2)-(B.3), p. 456: EOS 80 at one atmosphere; range -2 to 40 deg C, 0-42 ppt, 0.05 kg/m3, p. 457) |
| Sec. 7 Mackenzie (1981) | unconfirmed in print | unchanged; add Medwin (4.26) and Del Grosso (B.5) as printed check formulas (agree within 0.25/0.4 m/s) |
| Sec. 3.2 Muggleworth row (Lambert bound) | confirmed | add a second print: J&R eq. (2.23), p. 26 |
| Sec. 12.4 delta and dB per wavelength | ours; agrees with an implementation | **confirmed** (J&R eqs. (2.6), (8.18), (8.25)-(8.27), p. 15, 251-252) |
| Sec. 7 Hamilton shear regression (UnderwaterAcoustics.jl) | via secondary | J&R Sect. 5.2 gives the Richardson regressions (newer data), (5.10)-(5.13), confirmed; Hamilton's own not printed |

## 4. What this closes or changes in open issues

| Issue / item | Effect of Chs. 1-5 and Apps. A, B, C, H, I |
|---|---|
| sonar-3k1 (bottom model as published; layered) | Closes the "regressions against grain size" gap (eqs. 11.1-11.3) and confirms our table. Gives rock and shear data for an elastic option (Sects. 5.2, 5.6) and shows where it matters (rock, 37-66 deg, about 5 dB). Gives sediment gradient facts for a layered/graded option (Sect. 5.1.5) but no layered model (Ch. 8-9, Olson-Jackson remain the route). Regime interpolation, large-roughness law, bistatic model: not in my chapters |
| sonar-9le (high-frequency surface) | Nothing: the book is about the seafloor. App. B covers seawater density, speed, absorption and viscosity only - no surface scattering, no bubbles in water |
| sonar-6fc (K texture) | Nothing in my chapters (Ch. 16 is another part) |
| sonar-sul (real data) | Tables 5.1-5.2 (p. 132-134): site means of V_p, V_pR, alpha_p, Mz, porosity, rho, k, IOI for 69 sites at 400 kHz - a validation set for any grain-size -> geoacoustic mapping (e.g. SAX99, Duck, Tirrenia, Eckernfoerde). App. C gives within-site spreads to set tolerances |
| sonar-6t5 (ping-to-ping coherence) | Nothing (Ch. 3 describes changes over hours to days: storms make ripples, animals erase them; not ping-scale) |
| sonar-7iy (relief reverberation) | Nothing |
| Sediment fit tool (STATUS item 30, sonar-5bf2) | Priors and bounds for a fit: class scatter from the regressions (Tables 5.3, 5.4, 5.6, Fig. 5.28: about +-0.23 g/cm3 in density at fixed Mz) and within-site CVs (App. C). Physical bounds: a_rho from 1.1 to about 2.2 for sediments, nu 0.97-1.2 (Table 5.1/5.4 ranges), sigma_2 0.0001-0.004 |
| medium_sand sigma2 0.001 vs 0.002 (D-90) | **Not settled**: no named-sediment sigma2 table in the book |
| Kirchhoff/composite interpolation and cut-off | Not in my chapters (Ch. 13) |
| Large-roughness gravel/rock law; Lambert parameters (D-88) | Only the bound mu < 1/pi (2.23) and the Fig. 2.7 illustration of a "maximum Lambert" curve and of a "strong" curve that exceeds it only by backscatter enhancement (p. 25-26); no measured rock Lambert values here |
| Bistatic model | Definitions only: reciprocity (2.18) and the energy bound (2.20) - our tests already rest on these |
| Grain-size regressions | Done (Secs. 2.3, 2.4, 3.2) |
| Elastic and layered bottoms | Inputs (Sects. 5.2, 5.6), not the theory |
| Volume parameter definition | sigma_2 = sigma_v / alpha_p with alpha_p in dB/m (14.9) = ours |
| REFERENCES Sec. 10 wanted rows | "Regressions ... against grain size": found (Sec. 3.5). "Medium sand volume parameter": not found. "Sound speed formula ranges (Mackenzie)": not found (other formulas given). "Interpolation ... rough-bottom empirical expression": Ch. 13, not mine |

## 5. Work packages (ordered by realism gained per effort)

Each changes no default unless the owner agrees; both live-chain engines must stay equal where a package touches
`libs/sim` (`paths.cpp` reflection).

### WP-A. Grain-size relations as printed (S)
- Goal: replace the ours-labelled linear interpolation in `bottom.sediment_from_grain_size` by eqs. (11.1)-(11.3)
  (Sec. 2.4) with c_w = 1528 m/s in the delta step, the nu constant for Mz >= 5.3 being 1.0019 (documented as a
  correction of the book's 1.0091, with the continuity argument).
- Files: `python/gpusonar/sim/bottom.py` (new `apl_grain_size_relations(phi)`; `sediment_from_grain_size(phi,
  method="table_linear" | "relations")`), `tests/python/test_sediment_literature.py` or a new
  `test_grain_size_relations.py`; docs: REFERENCES Sec. 3.1/10 rows (Sec. 3.5).
- Tests first: (1) the relations reproduce every row of `GRAIN_SIZE_TABLE` (rho 5e-4, nu 1e-4, delta 0.5 %);
  (2) continuity at -1..9 breakpoints (rho, nu to 1e-4 at 1 and 5.3 phi; k jumps of 0.001-0.002 at 0, 2.6, 4.5,
  6.0 are in the print and allowed); (3) Fig. 11.2: normal-incidence loss 6.1 +- 0.5 dB at 0 phi, 14.8 +- 0.7 at 3,
  22.2 +- 0.7 at 5 phi; (4) the printed 1.0091 is NOT used (test that nu(5.3) is continuous).
- Default that could move: `sediment_from_grain_size` results between nodes (scenarios and fits that give a grain
  size, not a name). Before -> after: up to 0.68 dB at normal incidence at 5.3 phi, < 0.1 dB outside 5-6 phi,
  rho up to 0.011 at 1.25 phi. Owner first, or ship as opt-in `method="relations"`.
- Do NOT: change the named rows; copy 1.0091; call the relations "regressions" (the book says they are not).

### WP-B. Ch. 5 "standard" geoacoustic regressions as an alternative input path (S-M)
- Goal: `sediment_from_properties(porosity=..., density=..., grain_size=..., relations="richardson_briggs_insitu"
  | "richardson_briggs_core")` returning rho, nu, delta, preferring porosity or density (r^2 0.91-0.95) over Mz
  (0.82-0.92). Formulas: Table 5.4 (in situ, 38/58 kHz) for nu and k, Table 5.3 siliciclastic/all for the core
  path, eq. (5.14) for density from Mz, Table 4.5 for density from porosity (rho_g 2650, rho_w from
  `water_density`), Table 5.6 when only an impedance is known. Shear (if WP-C is done): (5.10)-(5.13).
- Tests: Table 5.5 Richardson-Briggs column (1.163/1.162/1.149 and 1786/1785/1764 m/s with c_w 1536);
  Fig. 11.2 "standard" curve (7.3 dB at 0 phi, 14.3 at 8 phi with rho_w 1.024); validity limits raise errors
  outside Table 5.3/5.4 ranges; refuse Table 5.3's two k regressions (misprinted).
- Default moves: none (new function). Owner question to prepare: should a site with measured porosity use these
  instead of the TR 9407 relations? Difference up to 12.5 dB in normal-incidence loss for silts (Sec. 3.2).
- Do NOT: mix these with the named TR 9407 rows silently; use the 400 kHz attenuation regressions for prediction
  (the book says they have little predictive value).

### WP-C. Elastic rock (and optional elastic sediment) reflection (M)
- Goal: an elastic half-space reflection coefficient (fluid over elastic solid; Ch. 9 eqs., another part of this
  extraction) for `rock` and optionally `pebbles`, with V_s from Sect. 5.6 (40-60 % of V_p) or Olson's generic rock
  (3600, 1900 m/s, 2500 kg/m3, delta 0.0018 / 0.085, REFERENCES Sec. 3.1). Sediment shear from (5.10)-(5.13) only
  if asked (effect < 0.2 dB).
- Files: `python/gpusonar/sim/bottom.py`, `propagation.rayleigh_reflection`, `libs/sim/src/paths.cpp` (both
  engines), `configs/sediments.yaml` (new optional `shear_speed`, `shear_delta` keys), parity tests.
- Tests first: V_s -> 0 equals the fluid result (1e-12); normal incidence independent of V_s; |R| = 1 below both
  critical angles for lossless rock; shear critical angle arccos(c_w / V_s) where loss appears; energy
  (|R|^2 + transmitted fluxes = 1, lossless); engine parity.
- Default moves: reflection off `rock` (Bellhop, image paths) at 37-66 deg grazing: about 0 -> about 5 dB loss
  (`elastic.py`). Owner first; opt-in by an explicit `shear_speed` key until then.
- Do NOT: change the rock Lambert scattering (D-88) - a separate question; invent rock V_s per site.

### WP-D. Seawater check oracles and the density pressure term (S)
- Goal: add Medwin (4.26) and Del Grosso (B.5) as test oracles for `sound_speed` (0.3 m/s / 0.4 m/s tolerance over
  0-30 deg C, 30-40 PSU, 0-500 m); add opt-in `water_density(..., depth=)` using (B.1)/(B.4).
- Tests: UNESCO check values K(35, 25, 1000) = 27108.95 bar, K(0, 5, 0) = 20337.80 bar; depth 0 equals today's
  function exactly; +0.45 kg/m3 at 100 m (10 deg C, 35 PSU).
- Default moves: none unless the owner wants depth-dependent density (0.04 % at 100 m - negligible for sonar).
- Docs: REFERENCES Sec. 7 rows (Sec. 3.5).

### WP-E. Uncertainty of class defaults and fit priors (S)
- Goal: a documented table (and optional helper) of the spread of rho, nu, delta about the grain-size relations:
  between-site scatter (Fig. 5.28 bounds about +-0.23 g/cm3 in density; Table 5.3/5.4 r^2) and within-site CVs
  (App. C summary, Sec. 2.9); use as priors/bounds in `tools/fit_sediment.py` (STATUS item 30).
- Tests: the fit tool keeps parameters inside the bounds; the stated spreads reproduce the numbers above.
- Default moves: fit results only where a bound binds; tell the owner.

### WP-F. Causal dispersion option (M; lower value)
- Goal: optional frequency-dependent speed ratio from (I.28): nu(f) = nu(f_ref) / (1 - (2 delta/pi) ln(f/f_ref)),
  with f_ref stated per source (TR 9407 relations: none stated; Richardson core data 400 kHz; in situ 38/58 kHz).
- Tests: Kramers-Kronig numerically ((I.25) by quadrature against (I.28)); f = f_ref leaves everything unchanged.
- Default moves: per decade from f_ref, sand reflection +0.24 dB and critical angle +2.1 deg. Owner first; the
  SAX99 data (Fig. 5.26) argue the ratio is nearly constant above 20 kHz, so the value of this option is mainly
  below 20 kHz (flank array).
- Do NOT: switch it on by default.

### WP-G. Gassy sediments and ripples (L; research-level, after the volume-scattering part)
- Goal: a "gassy mud" option needs the bubble-scattering model of Sect. 14.1.7 (another part) and the population
  facts of Sect. 3.3 (gas fraction, radii 0.25-10 mm, shapes); ripple generation for anisotropic roughness from
  Wiberg-Harris (lambda = 0.62 d0, eta/lambda = 0.17; anorbital 500 D or 535 D - resolve first).
- Owner first: neither is a default; both need data we lack.

## 6. Not verified / unclear on the page

| Item | Problem |
|---|---|
| Eq. (11.2), third branch (p. 314) | printed constant 1.0091 is discontinuous with the cubic at 5.3 phi; 1.0019 (our table, three earlier prints) is continuous. Treated as a misprint; the original (TR 9407 / Mourad and Jackson 1989) not read |
| Table 5.3 carbonate porosity regression (p. 136) | printed 11.760; only 1.760 gives ratios near 1 |
| Table 5.3 attenuation regressions (p. 136) | k = 0.74 - 0.07 Mz - 0.02 Mz^2 goes negative beyond 4.5 phi; k = -1.121 + 0.066 eta + 0.0006 eta^2 gives 9.7 at 90 % porosity against data near 0.1 (Fig. 5.3). Signs or coefficients misprinted; do not use |
| Table 5.4 units (p. 140) | caption says rho in g/cm3, the coefficients need kg/m3 (as the figure axis) |
| Table 5.6 units (p. 150) | validity "1.05 < IOI (kg m^-1 s^-1) < 2.6": IOI is defined in g/cm3 (p. 131, 147) |
| Fig. 5.28 / eq. (5.14) | axis labelled kg/m3 but values are g/cm3 |
| Eq. (5.8) (p. 159) | exponent printed as -1.1394 without the void ratio; (5.9) has -1.1394 e |
| Eq. (5.5) (p. 153) | units of sigma_0' and mu not given on the page |
| Ripple constant (p. 63) | anorbital lambda = 500 D in text, 535 D in Fig. 3.24 |
| p. 115 | "handbook values of quartz for both grain density (2560 kg/m3)" - 2650 elsewhere (p. 97, 114, Table 4.7) |
| p. 23 | rho_w "about 1025 kg/m^2" (kg/m^3 meant) |
| Eq. (B.5) pressure (p. 460) | "absolute pressure ... (kg m^-2)"; kg/cm^2 meant (as on p. 458); whether Del Grosso's original used gauge pressure was not checked: our comparison shows 0.22 vs 0.39 m/s for the two readings |
| App. B text (p. 460) | names f1/f2 as MgSO4/boric "respectively", opposite to the formulas |
| App. C (p. 471) | CV units not stated (percent evident) |
| "Standard" curve of Fig. 11.2 | the book says it uses Table 5.4 and "Eq. (5.15)"; (5.15) is Mz(rho), so (5.14) must be meant; rho_w used for the ratio not stated (1.024 reproduces the curve) |
| Fig. 5.26-5.27 values | read off log-scale figures: approximate only |
| Ch. 3 bubbles, Ch. 4 Sects. 4.4.2-4.4.4, p. 116-118, p. 59 and 70 | read in the extracted text, not on page images (no equations used from them) |
