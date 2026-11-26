# Sea-surface backscatter: McDaniel (1993) and Dahl et al. (1997), extracted for the simulator

Three papers the owner supplied (all rights reserved: formulas re-typed, constants with their page,
equation, table or figure; own words; the PDFs are not in the repository). Status words as in
`docs/REFERENCES.md`: **confirmed** = read on the page image of that paper; **via secondary** = a
formula of another author as this paper reprints it; **ours** = our assumption or derivation;
**read from figure** = a value read off a printed curve (about +-1.5 dB; never a printed number).

- [McD93] S. T. McDaniel, "Sea surface reverberation: A review", J. Acoust. Soc. Am. 94 (4), 1905-1922
  (October 1993). A scan, 18 pages; read from the page images.
- [DP97] P. H. Dahl and W. J. Plant, "The variability of high-frequency acoustic backscatter from the
  region near the sea surface", J. Acoust. Soc. Am. 101 (5), Pt. 1, 2596-2602 (May 1997). 7 pages.
- [D97] P. H. Dahl, W. J. Plant, B. Nuetzel, A. Schmidt, H. Herwig and E. A. Terray, "Simultaneous
  acoustic and microwave backscattering from the sea surface", J. Acoust. Soc. Am. 101 (5), Pt. 1,
  2583-2595 (May 1997). 13 pages.

Numbers in sections 3-4 were computed with scratch scripts outside the repository
(`.../scratchpad/jr/surf_papers.py`, `rough.py`, `part3.py`, `part4.py`, `part5.py`), importing this
worktree's `gpusonar` (a copy of the package in the scratchpad with the built module linked in; nothing in
the worktree was changed).

## 1. Scope

Risky first: [McD93] is a scan; its equations were read from 250-dpi renders of the pages, and two
equations look misprinted (marked in Sec. 2). Nothing in any of the three papers is above 70 kHz.

| Paper | Pages read | How |
|---|---|---|
| [McD93] | 1905-1922, all 18 (incl. references) | every page image; equations (22)-(25) and Figs. 10, 16, 18-20 re-rendered at 250 dpi |
| [DP97] | 2596-2602, all 7 | every page image |
| [D97] | 2583-2595, all 13 (incl. appendix and references) | every page image; Figs. 1 and 8 re-rendered at 200 dpi |

## 2. What the papers give

### 2.1 Conventions (all three)

| Item | [McD93] | [D97], [DP97] | Our code |
|---|---|---|---|
| Angle | grazing, theta (Fig. 1, p. 1905) | grazing ("we use the grazing angle", p. 2585) | grazing |
| Scattering strength | SS = 10 log(I_S / I_I), I_S from 1 yd^2 at 1 yd (eq. 1, p. 1905): cross section per unit area per unit solid angle | 10 log10(sigma_0^a), sigma_0^a = sigma^a / area, sigma^a per steradian (eqs. 4-8, p. 2584) | same (per unit area and solid angle) |
| Microwave vs acoustic | - | sigma^a = sigma^m / (4 pi) (eq. 8): microwave normalized radar cross sections are 11 dB above the acoustic number for the same surface (p. 2593) | - |
| Wind | U10 mostly; many old data at unknown height, corrections "less than 1 m/s" (p. 1914); Nuetzel et al. at 47 m (p. 1916) | U10 (sonic anemometers on the platform, corrected to 10 m); Nuetzel-Herwig data at 47 m (p. 2593) | U10 |
| Wave spectrum | W(K) 2-D, normalised so that sigma^2 = integral dKx dKy W(K) / (2 pi) (eq. 5); nondirectional W~(K) = integral_0^{2pi} K dphi W(K) / (2 pi) (eq. 6); K in rad/cm in the printed spectra, W~ in cm^3 (Fig. 10 axis "dB re 1 cm^3") | psi normalised so that its integral over (kx, ky) is the mean-square height (p. 2585): psi = W / (2 pi) | the time-spreading model uses Pierson-Moskowitz frequency spectra, not these |
| Time dependence | not stated (none needed: intensities only) | not stated | - |

Check (ours, algebra): [D97] eq. (10) divided by 4 pi with g_h -> sin^2 theta gives
sigma_0^a = 4 k^4 sin^4 theta psi(0, 2k cos theta); [McD93] eq. (4) with W = 2 pi psi gives
2 k^4 sin^4 theta (2 pi psi) / pi, the same. The two papers' perturbation results agree, and the 2 pi
in eq. (5) is the only place the normalisation enters.

### 2.2 [McD93] rough-surface scattering (Sec. I.A, pp. 1907-1908) - confirmed

| Eq. | Formula (theta grazing, k acoustic wavenumber) | Notes |
|---|---|---|
| (3) | zeta(x,y) = int int dKx dKy W^(1/2)(K) cos(i[Kx x + Ky y + phi(K)]) | Rice's representation, phi random |
| (4) | SS = 10 log[ 2 k^4 sin^4 theta W(2k cos theta, 0) / pi ] | lowest-order perturbation (Bragg, K = 2k cos theta); "expected to be valid provided that the rms surface slope ... as well as the value of kh is small" |
| (7) | SS = 10 log{ exp[-(2 tan^2 theta sigma_x'^2)^-1] / (8 pi sin^4 theta sigma_x'^2) } | Kirchhoff, stationary phase (Beckmann-Spizzichino), isotropic surface, sigma_x' the x component of the rms slope; "independent of frequency"; specular scatter from facets |
| (8) | SS = 10 log[ 2 k^4 <sin^4(theta + eps)> W(2k cos theta, 0) / pi ] | composite roughness (Kur'ynov 1963; Bachmann 1973, "for low grazing angles"); <> over tan eps, the large-scale slope; partition 0 <= K_L <= 2k cos theta |

- Partition used throughout the review: **K_L = k/4** (p. 1908); K_L = 0 gives eq. (4).
- Shadowing: Andreeva et al. (1978) found its effects negligible in composite theory (p. 1907).
- "At high grazing angles, composite-roughness theory yields results that are extremely sensitive to
  the choice of K_L"; the one high-angle comparison uses McDaniel (1986) instead (p. 1908). No
  interpolation formula between facet and small-scale terms is printed.
- Kirchhoff "generally considered valid for forward scatter (Thorsos, 1990)" (p. 1907). That is all the
  review says about forward or near-specular scattering: **it contains no forward-scatter,
  near-specular-reverberation or coherent-reflection-loss model.** Its reference list (p. 1921) has
  McDaniel 1986, "Diffractive corrections to the high-frequency Kirchhoff approximation", JASA 79,
  952-957, with the note that the right-hand side of that paper's eq. (18) "should be multiplied by a
  factor of 2"; and McDaniel 1993, "Sea surface reverberation fluctuations", JASA 94, 1551-1559.
- Fig. 4 (p. 1908): at 10 m/s composite theory is strongly frequency dependent at 0.4-4 kHz below
  30 degrees, perturbation theory "very weakly" so.

### 2.3 [McD93] resonant microbubbles (Sec. I.B, pp. 1908-1909) - confirmed

| Eq. | Formula | Constants (as printed) |
|---|---|---|
| (9) | SS = 10 log{ delta_r sin theta [1 + 8q exp(-2q) - exp(-4q)] / (8 pi delta) } | Crowther (1980), four paths of Fig. 5, (b) and (c) adding coherently |
| (10) | q = 2 C pi^2 a_r^3 N / (delta_r sin theta) | a_r(m) = 3.2 / f(Hz); delta_r = 0.0136; delta = delta_r + 3.8e-4 [f(Hz)]^(1/2); N "the depth-integrated bubble density per unit volume per unit bubble radius"; C = 1 for N proportional to a^-4 (Physics of Sound in the Sea), Medwin (1977) 1.67, numerical integration of Clay and Medwin "close to unity" |
| (11) | SS = 10 log(3 C pi a_r^3 N / delta) | weak scattering, small q: proportional to N, independent of grazing angle |
| (12) | SS = 10 log[ delta_r sin theta / (8 pi delta) ] | saturation, large q: independent of N, "varies as sin theta"; "a unique feature of scattering from resonant bubbles" |
| (13) | SS = 10 log{ (C pi a_r^3 N / delta) [3 - 4/(1 + 4 gamma^2) + 1/(1 + 16 gamma^2)] } | Crowther, thin layer N(z) = N0 exp(-z/D), gamma = k D sin theta; for "layers that are thin on an absorption scale" |

- Overshoot before saturation: 3-4 dB (Fig. 6, p. 1908; 4 and 40 kHz at 10 degrees); far fewer bubbles
  saturate 4 kHz than 40 kHz.
- Coherence condition: the four paths add coherently when 2 k sigma sin theta < 1 (sigma the rms
  roughness) (p. 1909). Eq. (13) goes from sin^4 theta (D << lambda) to eq. (11) (thick layer) (Fig. 7).
- Neglected: refraction in the bubbly layer, lateral inhomogeneity. The sound speed at the surface "may
  be as much as 1% lower than at a depth of 2 m" (Farmer and Vagle 1989); "at low grazing angles,
  refraction ... can thus be important" (p. 1909).

### 2.4 [McD93] bubble plumes (Sec. I.C, pp. 1909-1910) - confirmed (low-frequency models, below 2 kHz)

dsigma/dOmega = F(D) G(a) (14). McDonald: F_M(D) = (D dc_s / c)^2 (4 D^2 k^2 sin^2 theta /
(1 + 4 D^2 k^2 sin^2 theta))^2 (15), G_M(a) = [k b J1(2ka cos theta) / cos theta]^2 (16), with D = 1.5 m,
a = 0.33 m, b = 1 m at 10 m/s, dc_s/c = 0.01. Henyey: F_H(D) = int_0^inf dL f(L) (L dc / c)^2
(1 - sin(2kL sin theta)/(2kL sin theta))^2 (17), G_H(a) = int_{0.75 m}^{2 m} drho g(rho) G_M(rho) (18),
f(L) = (0.9 + 0.1 L m^-1) exp(-L m^-1 / 1.75) / 1.75 m (19), g(rho) = [(2 m - rho)(rho - 0.75 m)]^(1/2) /
(0.1953 m^2) (20); SS = 10 log(N_P dsigma/dOmega) (21), N_P plumes per unit area. Layer depths 1.5 and
1.75 m. Not relevant above a few kHz; listed for completeness.

### 2.5 [McD93] wave-number spectrum (Sec. II.A, pp. 1910-1911) - confirmed, two misprints noted

| Eq. | Formula | Range and constants |
|---|---|---|
| (22) | printed "Omega = g K (1 + tau K^2)" | tau = surface tension / (g x water density). **Misprint (ours):** dimensionally Omega^2. tau's value is not printed |
| (23) | W~_D(K) = alpha exp[-(K_p/K)^2] gamma^Gamma / (K^2.5 K_p^(1/2)) | Donelan et al. (1985), K <= 10 K_p |
| | alpha = 0.006 (U10/C_p)^0.55; gamma = 1.7 (U10/C_p <= 1), 1.7 + 6 ln(U10/C_p) (>= 1) | |
| | printed "Gamma = exp[-(K/K_p)^(1/2) / (2 s^2)]" | **probably misprinted** (the peak-enhancement exponent of the Donelan form needs (sqrt(K/K_p) - 1)^2); irrelevant at Bragg wavenumbers, where Gamma -> 0 either way |
| | s = 0.08 [1 + 4 (C_p/U10)^3]; C_p = U10 x~^(-0.23) / 11.6; x~ = g x / U10^2 | x the fetch |
| | fully developed: U10/C_p = 0.8, K_p = g / (1.2 U10^2) | so W~_D, and the predicted backscatter, rise linearly with wind; with K^-2.5 the backscatter rises as f^(1/2) |
| (24) | W~_P(K) = 0.875 (2 pi)^(P-1) (1 + 3 K^2 tau) / ( g^((P-1)/2) [K + K^3 tau]^((P+1)/2) ), 0.942 cm^-1 <= K <= K_c | Pierson (1976); P = 5 - log(u*), u* the friction velocity; units not printed (cgs from the "cm" in the range; ours: u* in cm/s) |
| (24) | W~_P(K) = 7.04e-3 u*^3 / K^9, K >= K_c | K_c "the onset of the Cox viscous cutoff"; **no value or law for K_c is printed** |
| (25) | W~(K) = W~_P(0.942 cm^-1) (K / 0.942 cm^-1)^Q, 10 K_p <= K < 0.942 cm^-1, Q = log[W~_D(10 K_p) / W~_P(0.942 cm^-1)] / log(10 K_p / 0.942 cm^-1) | straight line in log-log between the two |

- The nondirectional spectrum is used because data at 3 kHz and above show no dependence on heading
  relative to the wind (p. 1911). Fetch 50 km raises the spectrum by 3 dB over fully developed in the
  range of interest (Fig. 10). Unstable atmosphere: u* at most 10 % larger for the same U10 (p. 1911).
- **No u*(U10) relation and no Cox-Munk slope constants are printed** in any of the three papers.
- Fig. 10 (10 m/s, fully developed): read from figure, about +67 dB re cm^3 at the peak (log K = -3.25),
  -17 dB at K = 1 cm^-1, -31 dB at log K = 0.6 and a steep fall beyond log K of about 0.65-0.7
  (K about 4.5-5 cm^-1): the viscous cutoff.

### 2.6 [McD93] bubble populations and whitecaps (Sec. II.B, pp. 1911-1914) - confirmed

- Stages (Fig. 11, Monahan and Lu 1990): Stage A whitecap, about 0.5 s, 0.5 m^2, air to about 0.25 m,
  void fraction 0.05-0.10; Stage B foam, about 4 s, 1 m^2, 0.5 m, void fraction about 1e-4; persistent
  plumes (several minutes, to about 10 m, Langmuir) about 1e-6; a background layer two or three orders
  lower.
- Wu (1981): N(a, z) proportional to a^-n U^m exp(-z/D) (26), 50 <= a <= 300 micrometres,
  3.5 <= n <= 5, m about 4.5, D about 1 m for z <= 3 m, independent of wind over 6-13 m/s. Medwin and
  Breitz: n = 2.7 for a > 50 micrometres; Akulichev and Bulanov: 3.5 <= n <= 3.8. **No absolute
  constant is printed** (a proportionality only).
- Reported populations "are generally higher than those needed to account for measured acoustic
  backscattering levels in the open ocean at frequencies of 20-60 kHz" (McDaniel 1988) (p. 1913).
- Largest reliably counted bubble, 300 micrometres, resonates at 10 kHz (p. 1913). No population for
  radii below about 16-20 micrometres is reported (Fig. 12, 11-13 m/s, radii 20-700 micrometres).
- Whitecap fraction W: Wu, proportional to u*^3; Monahan and Lu, a least-squares line of W^(1/3) against
  U10 (Fig. 14; constants not printed); North Sea (Wei and Wilson) exceeds the open-ocean fit by a
  factor 3.5 at 9 m/s (Fig. 26, p. 1917), which "accounts for 5 dB of the 10-dB disparity" between
  coastal and open-ocean backscatter at 9 m/s if bubbles scale with W (p. 1918).
- Viscosity and gas solubility against temperature (Fig. 13); supersaturation seasonal (Figs. 15, 30).
  Effects of supersaturation on reverberation: "no conclusions can be drawn" (p. 1919).

### 2.7 [McD93] data (Sec. III) - confirmed statements; curves read from figures

Statements (printed):
- 3-25 kHz, grazing above 30 degrees: data agree with rough-surface theory (p. 1906); 3-3.5 kHz and
  20-25 kHz at high grazing verify theory and spectrum (p. 1916).
- 3-60 kHz: at low grazing and high wind the excess is resonant microbubbles, shown by saturation at
  the predicted level (Figs. 16-17, 19, 21, 24).
- **Coastal waters exceed the open ocean by about 10 dB before saturation, 3-60 kHz** (p. 1917);
  open-ocean data at 20-60 kHz from different experiments agree (McDaniel and Gorman 1982), no clear
  dependence on water temperature.
- Lilly and McConnell (1978), 15-60 kHz: "no dependence on frequency" (p. 1915).
- Wildt (1946) 25 kHz data were first misread with damping 0.22; the correct 0.072 gives "acceptable
  agreement" (Fig. 17, p. 1914).
- Reeves et al. (1969), 60 kHz, low grazing: weak dependence on wind direction below 5 m/s, isotropic
  above (p. 1915). [D97] p. 2586 quotes the same work: 3-4 dB upwind/crosswind at 20-40 degrees,
  vanishing above about 5 m/s.
- 3 kHz, 30 degrees, Nuetzel: excursions of 20 dB above the mean at 6.5 m/s, decaying in 2-3 s;
  bubbles resonant at 3 kHz rise at 20 cm/s, so a layer depth D of 0.4-0.6 m; 0.25 <= D/lambda <= 1 in
  the open ocean (p. 1917).
- Below 1 kHz, the volume-reverberation floor is about -60 dB (p. 1918); resonant-bubble response time
  exceeds 40 ms below 1 kHz (p. 1920).

Figure readings used for tests in Sec. 3 (read from figure, +-1.5 dB):

| Fig. | Data | Readings (grazing deg: dB) |
|---|---|---|
| 18 (p. 1915) | Urick and Hoover (1956), 60 kHz, open ocean off Key West, smooth curves | 9.0 m/s: 15: -26, 30: -26, 45: -24.5, 60: -21. 6.4 m/s: 15: -34.3, 30: -33.8, 45: -31.5, 60: -26. 3.9 m/s: 15: -46.3, 30: -44.5, 45: -41, 60: -34. 2.6 m/s: 32: -50, 45: -46.5, 60: -43. All cross near -15 dB at 76 degrees |
| 19 (p. 1915) | Garrison et al. (1960), 60 kHz, Dabob Bay (coastal) | 2.1 m/s: 10: -50, 30: -40.5, 50: -40, 60: -38. 4.1 m/s: 20: -28, 30: -27, 50: -26.5, 60: -25.5. 6.2 m/s: 30: -23, 50: -21.5, 60: -20.5. 7.2-15.4 m/s: 20: -22, 30: -21, 50: -20.5, 60: -19.5. Predicted saturation curve: 10: -30.5, 30: -25.5, 60: -23 |
| 16 (p. 1914) | Wildt (1946), 25 kHz, 3 degrees, curve through scattered points | flat -64.5 below 2 m/s (masked by volume reverberation below 2.5 m/s), -60 at 3, -51 at 4.3, -45 at 5.5, -39 at 6.5, -35 at 7.8, -30 at 9 m/s and flat above |
| 20 left (p. 1916) | Lilly and McConnell (1978), 20-25 kHz, open ocean | about 10 m/s: 3: -32, 10: -28, 17: -25, 28: -23. About 5 m/s: 5: -43, 10: -40, 17: -37, 28: -37 |
| 20 right (p. 1916) | Nuetzel et al. (1987), 20 kHz, 7 m/s (wind at 47 m), North Sea | 42.5: -31.3, 52.5: -25.8, 62.5: -17.4, 72.5: -5.8, 82.5: +4.6, 88: +7.3 |

Low frequency (Sec. III.B, pp. 1918-1920):
- **Chapman and Harris (1962)** as printed (eq. 27, p. 1918; via secondary):
  SS = 3.3 beta log(theta/30) - 42.4 log(beta) + 2.6, beta = 158 (v f^(1/3))^(-0.58), theta grazing in
  degrees, v wind speed in **knots**, f in Hz. Range stated: explosive sources north of Bermuda, March
  1961, **400-6400 Hz, wind up to 15 m/s, "at low grazing angles"**; no upper angle is printed (Fig. 27
  plots 0-30 degrees). At 30 degrees it goes as wind^2.5 and frequency^0.8. (The text twice calls it
  "Eq. (28)", pp. 1918-1919; there is no eq. (28): a misprint.)
- Fig. 27: at 10 m/s, rough-surface theory is 10-20 dB below Chapman-Harris. Most later low-frequency
  data lie below Chapman-Harris (Richter 1964, Fig. 28; Ogden-Erskine and Brown et al., Fig. 29; Figs.
  31-32): "the Chapman and Harris empirical model yields scattering strengths that exceed the remaining
  data" (p. 1920). Chapman-Harris's sea was not fully developed (wind rose 0-15 m/s in 15 h, p. 1918).

### 2.8 [D97] the bubble model and its fit (Sec. II.B, V; pp. 2587-2592) - confirmed

| Eq. | Formula | Constants |
|---|---|---|
| - | single resonant bubble: sigma_bs = a_R^2 / delta^2 | delta = 2.55e-3 f^(1/3), f in Hz (Devin 1959); a_R (m) about 3.25 / f; at 70 kHz (a_R about 46 micrometres) 10 log sigma_bs = -67 dB; four paths, two coherent: factor 6; over 9 m^2: -69 dB (p. 2587) |
| - | weak-limit surface strength 10 log10(6 s_I), s_I the depth integral of s_V (dimensionless) | |
| - | beta_I = 4 pi (delta / delta_r) s_I | beta_I "the depth-integrated extinction cross section per unit volume due to scattering and absorption", dimensionless; delta_r = 0.0136; the "hybrid resonant approximation" |
| (17) | sigma_0^b = (sin theta / (8 pi)) (delta_r / delta) (1 + 8 beta e^(-2 beta) - e^(-4 beta)), beta = beta_I / sin theta | |
| (18) | sigma_0^b = sin theta delta_r / (8 pi delta) | beta_I >~ O(1) and theta <~ 30 degrees: saturation |
| (19) | sigma_0^b = 1.5 beta_I delta_r / (pi delta) | beta_I << 1; reverts to sin theta at extremely small grazing |
| (20) | N_I(a_R) = beta_I delta_r / (2 pi^2 a_R^3) | resonant-bubble depth-integrated density per unit area per unit radius |
| (21) | N(a_R; z) = (N_I(a_R) / L) e^(-z/L), L about 0.1 U10 | assumed e-folding depth |
| Sec. VI | the roughness term is reduced by **20 (log10 e) beta_I / sin theta dB** | two-way travel through the layer |

**Table II (p. 2591): log10 beta_I = a + b log10 U10**, fitted by inverting eq. (17) (Newton-Raphson)
to 1-min averages at 17 degrees grazing, North Sea (SAXON-FPN, Dec 1991 and Mar 1992):

| f (kHz) | resonant radius at the surface (micrometres) | a | b |
|---|---|---|---|
| 26.5 | 121 | -6.16 | 5.50 |
| 50 | 64 | -6.70 | 6.40 |
| 70 | 46 | -5.75 | 5.35 |

- Fitted range: data 0-13 m/s (Fig. 5); the fit is drawn dashed below 2.5 m/s "to emphasize the
  extrapolation" (p. 2591). beta_I grows from O(1e-3) to O(1) over 3-15 m/s, as U10^b with b about 6
  (p. 2592). beta_I rises with frequency as about f^0.8 (clearest between 26.5 and 70 kHz), which
  implies n about 3.8 in N(a) proportional to a^-n (p. 2592).
- Comparisons (Fig. 6): Farmer and Lemon (1984), log10 beta_I = -3.5 + 2.9 log10 U at 25 kHz (U from
  land stations, reliable only above 10 m/s; scaled by f^0.8 to 50 and 70 kHz by [D97]); Crowther,
  Griffiths and Hansla (1993), U^4, constants not printed (extended by a polynomial fit to 26.5-70 kHz).
  "Our results do point to a significantly higher power-law dependence for beta_I than the approximately
  cubic dependence ... suggested by the FL data" (p. 2592).
- Bubble-induced sound-speed anomaly near the surface of order -1 m/s at 10 m/s (Fig. 7, p. 2593);
  bubbles "confined to a layer of order 1 m".
- **Calibration**: an uncertainty was found in the acoustic data, so the data were **forced** to the
  eq. (18) saturation levels using 11 < U10 <= 13 m/s (Table I, p. 2589):

| f (kHz) | Table I saturation, 17 degrees (dB) |
|---|---|
| 26.5 | -26.9 |
| 50 | -27.8 |
| 70 | -28.3 |

- Power law sigma_0 ~ U^n over 2-7 m/s at 17 degrees (p. 2591): acoustic n about 5.3, 6.4, 5.5 at 70,
  50, 26.5 kHz; microwave HH 3.8, 3.3, 2.1 (Ku, X, C); VV 2.8, 2.3, 1.8.
- Transition (pp. 2589, 2593): at 17 degrees bubbles dominate above about 3 m/s ("does not exceed
  3 m/s"); at 45 degrees above about 5 m/s. Thresholds quoted: capillary waves from about 1.5 m/s
  (Stolte); sporadic small-scale breaking from about 2.5 m/s (Wille and Geyer). Sec. E of Fig. 4:
  3 -> 5 m/s raised acoustic scattering 15 dB and microwave 5 dB. Noise floors: about -78 dB acoustic,
  -74 dB microwave (p. 2589).
- Appendix (pp. 2593-2594): a beam- or pulse-limited geometry makes the ensonified volume differ from
  area x layer depth l_b; with rho = V_eff / (A l_b) < 1, beta_I from eq. (17) is biased low by about
  10 log10 rho; rho near 1 for l_b <= 1 m at 17 degrees and 24 m depth (Fig. A2); at 45 degrees and
  50-100 m depth, errors "on the order of 3 dB" (Fig. A3).
- Rough-surface part (Sec. II.A, VI): composite surface theory for HH microwave (Plant 1986) with its
  short-wave growth rate changed to (sin^2 theta + 0.25)/1.25: the HH strength is an **upper limit** to
  acoustic roughness scattering (p. 2586; eqs. 9-16: Bragg condition k_B = 2 k_0 cos theta, eq. (10)
  sigma_0^m = 16 pi k_0^4 |g_p|^2 psi(0, k_B), g_v (11), g_h (12), tilt (13)-(15), average over slopes
  (16)). **Plant's spectrum is not printed.** Composite theory stated for about 20-70 degrees grazing.
  Fig. 1 (Ku band HH, models; read from figure): at 30 degrees about -35 to -42 dB at 5 m/s and -23 to
  -27 dB at 20 m/s; at 60 degrees about -16 to -23 dB at 5 m/s; subtract 11 dB for the acoustic number.
- Sec. VI and Fig. 8 (45 degrees, 26.5 and 50 kHz, against Nuetzel and Herwig 1995 data, wind at
  47 m): the bubble-attenuated surface term (read from figure) peaks at about -34.5 dB near 10 m/s
  (26.5 kHz) and -35 dB near 8 m/s (50 kHz), is about -41 dB at 3 m/s and falls below -50 dB above
  about 13-17 m/s; the total is dominated by bubbles above about 5 m/s.
- Ku-band HH acoustic-equivalent at 70 kHz, 17 degrees (Fig. 5, read from figure, microwave already
  minus 11 dB): about -62 dB at 2 m/s, -57 at 4, -48 at 5, -45 at 6, -42 to -38 at 9-12 m/s.

### 2.9 [DP97] variability and statistics - confirmed

- Data: SAXON-FPN, 17 degrees, 50 and 70 kHz (also 26.5 kHz in the sigma_T law), two-way beam widths
  2.9 and 2 degrees, 4 ms pulses, transducer at 24 m, pings every 0.4 s; FLIP, 30 kHz, vertical
  incidence, 16 degrees, 1 ms, 0.25 s, volume strength 1 m below the moving surface (pp. 2596-2597).
- Fast fluctuation: exponential PDF of intensity (Rayleigh amplitude); its dB form has standard
  deviation **5.57 dB** (= (10 log10 e) pi / sqrt 6). Decorrelation time tau = O(10) ms at 50 kHz
  (phase change pi/2 when |v| tau about 0.75 cm, v the orbital speed of the bubbles), so pings 0.4 s
  apart are independent (p. 2598).
- Slow modulation: lambda = <sigma_0> over T = 10 s is lognormal (eq. 1); E(lambda) = e^(mu1 +
  sigma1^2/2) (2), sigma1 = sigma_T / k, mu1 = mu_T / k, k = 10 log10 e. The PDF of s = 10 log10
  sigma_0 is p_s(s) = int p_{s|lambda}(s|lambda) p_lambda(lambda) dlambda (3), p_{s|lambda} =
  e^(-e^(s/k)/lambda + s/k) / (k lambda) (4). Moments (5): **E(s) = mu_T - C_E k (6)**, C_E Euler's
  constant (C_E k = 2.507 dB), **Var(s) = k^2 pi^2 / 6 + sigma_T^2 (7)**.
- Fitted values (p. 2598, Figs. 4, 6, 7): 5.8 m/s, 50 kHz: mu_T = -30.63 dB, sigma_T = 3.07 dB, chi^2
  probability 0.20, series standard deviation 6.42 dB; FLIP 9.8 m/s volume: mu_T = -50.67 dB, sigma_T =
  3.45 dB (text: -50.7, 3.4), 0.25; SAXON 11.1 m/s: T = 10 s, -29.59 dB, 3.88 dB, < 0.01; T = 3 s,
  -30.06 dB, 4.41 dB, 0.05.
- **sigma_T (T = 10 s) about -0.18 U10 + 4.85 dB for 3.8-11.6 m/s**, 17 degrees, all of Phase II
  including 26.5 kHz; representative 3.7 dB (p. 2602). Less variance at high wind: less patchy, and
  saturation caps the level.
- Spectra of s: below about 0.1 Hz a large low-frequency part (bubble clouds advected through the
  footprint), roll-off between f^-1 and f^-2, still rising below 0.01 Hz to a broad peak near 0.005 Hz;
  20-30 % of the variance (SAXON), 40 % (FLIP). A peak at the dominant-wave frequency (0.14 Hz at
  11.1 m/s; 0.07 Hz FLIP) holding about 10 % of the variance, with coherence < 0.2 with the wave orbital
  velocity: a strongly nonlinear modulation (p. 2599-2601).

## 3. Against our simulator

Wrong or doubtful first:

1. **Low wind, open ocean: our `bubble_layer_surface` (default `extinction: apl_1994`) is 6 to 12 dB
   above every open-ocean data set these papers show at 2.6-6.4 m/s and 5-45 degrees** (0-3 dB at 60 degrees), and Gallaudet's 68 kHz values agree
   with those data (D-193 is a model problem, not only a measurement one). Coastal data (Dabob Bay) are
   1-7 dB above ours at 4 m/s. McDaniel's ~10 dB coastal-open disparity is a parameter our model does
   not have.
2. Our wind dependence at 17 degrees, 2-7 m/s, is U^3.8-4.1; [D97] measured U^5.3-6.4 at 26.5-70 kHz.
3. The omitted small-scale roughness term, computed with [McD93]'s spectrum, would add 5-17 dB at
   50-70 degrees and 3 m/s; but Urick-Hoover's 60 kHz data at 2.6-3.9 m/s and 60 degrees lie 11-18 dB
   below that sum. That spectrum, with our assumed u* and cut-off, over-predicts at steep angles and low
   wind. Neither paper prints the handbook's G(U).
4. Above 70 kHz nothing in these papers holds any model.

### 3.1 Item by item

| Item | Papers | Our code | Agreement (how checked) |
|---|---|---|---|
| Bubble term, eq. form | [D97] (17)-(19); [McD93] (9)-(12) with q = beta_I / sin theta, beta_I = 2 C pi^2 a_r^3 N / delta_r | `surface.bubble_backscatter` | **identical** formula (re-typed and evaluated; our expm1 form equals the bracket). REFERENCES row (Gallaudet) "via secondary" -> the formula is now **confirmed** in [D97] |
| delta | [D97] 2.55e-3 f^(1/3), f in **Hz** | `bubble_damping_constant`: 2.55e-2 (f/kHz)^(1/3) | **identical** (2.55e-2 x 10^-1 = 2.55e-3); the kHz unit marked "ours" in surface.py is confirmed by [D97]'s Hz form |
| delta, McDaniel's | delta_r + 3.8e-4 sqrt(f Hz) | - | differs: saturation level McD minus ours +0.30, +0.24, +0.05, -0.01, -0.21, -0.35, -0.62, -1.19 dB at 8, 12, 25, 30, 50, 68, 120, 330 kHz |
| delta_r | 0.0136 (both) | `BUBBLE_RERADIATION_DAMPING` 0.0136 | identical |
| a_R | [McD93] 3.2/f; [D97] 3.25/f | `resonant_radius` (Minnaert at depth 0) | 47.1 / 47.8 / 47.7 micrometres at 68 kHz: within 1.5 % |
| Saturation level | [D97] Table I: -26.9, -27.8, -28.3 dB at 17 degrees | `bubble_saturation_db` | -26.82, -27.74, -28.22 dB: **within 0.1 dB** |
| Overshoot | [McD93] 3-4 dB (Fig. 6) | bracket maximum 2.36 (+3.7 dB) (surface.py comment) | agrees |
| Optical-depth convention | [D97] Sec. VI: two-way loss 20 log10(e) beta_I / sin theta dB | `bubble_layer_optical_depth` from a loss law: beta_I = loss_dB sin t ln10 / 20 | **identical** convention (one vertical crossing, nepers of intensity) |
| beta_I(U, f) | [D97] Table II (fit, 17 degrees, North Sea) | `apl_1994` (default) or `hall_novarini` | **differs**, table below |
| Facet term | [McD93] eq. (7) with sigma_x'^2 = mss/2 | `facet_backscatter` (Gallaudet's sec^4/(4 pi s^2) exp(-tan^2/s^2)) and `reverb_specular` lobe at back-azimuth | **identical** to < 1e-6 dB for 2-15 m/s, 5-89 degrees (computed) |
| Cox-Munk constants | not printed in any of the three | `acoustics.mean_square_slope` 0.003 + 0.00512 U | nothing to compare |
| Facet near normal | Nuetzel 20 kHz 7 m/s (Fig. 20, read) | `facet_backscatter` | ours **3.3 to 4.3 dB low** at 82.5-88 degrees, 1.3 dB low at 72.5 (wind at 47 m, so the 10 m wind is lower: at 6 m/s ours is +0.6 dB higher, still 3-4 dB low); McDaniel's own curve there is her 1986 diffraction-corrected Kirchhoff, whose formula is not in the review |
| Chapman-Harris | [McD93] eq. (27) | `acoustics.chapman_harris_db` | **identical** formula (difference 0 to 1e-14 dB at four points; exponents at 30 degrees 2.46 in wind and 0.82 in frequency, as printed "2.5" and "0.8"). Range: 400-6400 Hz, wind to 15 m/s, low grazing; our docstring's "below about 40 degrees" is **not** in the review |
| Small-scale roughness | [McD93] eqs. (4), (8), (23)-(25); [D97] composite (Plant) not printed in full | omitted (D-87, D-132) | size computed below |
| Bubble populations | Wu: a^-n U^4.5, n 3.5-5, D about 1 m; [D97]: n about 3.8, L about 0.1 U10 | Hall-Novarini: U^3, a^-4 / a^-4.37, L = 0.4 m below 7.5 m/s | wind exponent of ours lower than Wu's 4.5 and [D97]'s 5-6 for beta_I |
| Statistics | [DP97] lognormal-modulated exponential | surface reverberation: Rayleigh; K texture opt-in (D-70) | absent; see 3.6 |
| Doppler, forward scatter, coherent loss | none quantitative in these papers | D-84, D-85, D-62, D-76 | nothing to compare |

### 3.2 beta_I: [D97] Table II against our two laws (computed)

beta_I, and our law over [D97]'s in dB (positive: ours larger):

| U10 (m/s) | 26.5 kHz D97 | apl_1994 / D97 | hall_novarini / D97 | 70 kHz D97 | apl_1994 / D97 | hall_novarini / D97 |
|---|---|---|---|---|---|---|
| 2 | 3.1e-5 | +10.1 | +17.5 | 7.3e-5 | +10.1 | +19.5 |
| 3 | 2.9e-4 | +5.7 | +13.1 | 6.3e-4 | +5.9 | +15.4 |
| 4 | 1.4e-3 | +4.0 | +10.0 | 3.0e-3 | +4.4 | +12.5 |
| 5 | 4.8e-3 | +3.9 | +7.6 | 9.8e-3 | +4.4 | +10.2 |
| 6 | 1.3e-2 | +4.7 | +5.6 | 2.6e-2 | +5.4 | +8.3 |
| 8 | 6.4e-2 | -0.2 | +3.0 | 0.12 | +0.7 | +6.0 |
| 10 | 0.22 | -4.0 | +2.2 | 0.40 | -3.0 | +5.5 |
| 12 | 0.60 | -7.1 | +1.2 | 1.06 | -6.0 | +5.0 |
| 15 | 2.0 | -10.9 | -0.2 | 3.5 | -9.7 | +4.2 |

(50 kHz: apl_1994 +15.2 dB at 2 m/s, +9.1 at 3, +5.3 at 5, -0.6 at 8, -13.8 at 15.) Farmer-Lemon at
25 kHz scaled by f^0.8 is 14-19 dB above [D97] at 2-3 m/s and within 2 dB at 10-12 m/s: published low-wind
beta_I laws span about 14 dB at 3 m/s (19 dB at 2 m/s). At high wind beta_I barely changes the backscatter
(saturation), so Table II is weakly constrained there.

Backscatter, ours (bubble + facet, `apl_1994`) minus the same with [D97] beta_I, dB, grazing 5 / 10 / 17 /
30 / 45 / 60 / 80 degrees (computed):

| f | 2 m/s | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| 26.5 kHz | +10.1 (all <= 60 deg) | +5.6 | +3.8..+4.0 | +3.2..+3.8 | +2.3..+4.4 | -0.1 | +2.6..-2.5 | +3.6..-1.9 | +1.3..+3.7 |
| 50 kHz | +15.1 | +9.0 | +5.9..+6.3 | +4.0..+5.2 | +1.2..+4.9 | -0.4..+0.4 | +3.1..-1.4 | +0.7..+3.7 | +0.2..+3.7 |
| 70 kHz | +10.0 | +5.8 | +3.9..+4.3 | +2.8..+4.2 | +0.1..+4.6 | -0.5..+0.4 | +2.1..-0.8 | +0.2..+3.2 | 0..+3.7 |

At 80 degrees the facet term dominates and the difference is 0. Above 8 m/s ours is up to +3.7 dB
higher because our smaller beta_I sits on the overshoot while [D97]'s is past it.

Wind exponent n of sigma_0 ~ U^n, 2-7 m/s, 17 degrees: ours 4.13 / 3.94 / 3.80 at 26.5 / 50 / 70 kHz;
eq. (17) with Table II 5.37 / 6.19 / 5.10; [D97] measured 5.5 / 6.4 / 5.3.

### 3.3 Against data read from [McD93]'s figures (computed; readings +-1.5 dB)

| Data (Fig.) | Wind (m/s) | Grazing | Read | Ours | Ours - read | With [D97] beta_I |
|---|---|---|---|---|---|---|
| Urick-Hoover 60 kHz open ocean (18) | 9.0 | 15 / 30 / 45 / 60 | -26 / -26 / -24.5 / -21 | -24.8 / -22.6 / -21.9 / -19.9 | +1.2 / +3.4 / +2.6 / +1.1 | +1.0 / +3.8 / +3.3 / +1.6 |
| | 6.4 | same | -34.3 / -33.8 / -31.5 / -26 | -25.3 / -24.0 / -23.5 / -23.0 | **+9.0 / +9.8 / +8.0 / +3.0** | +6.5 / +6.5 / +4.3 / -0.4 |
| | 3.9 | same | -46.3 / -44.5 / -41 / -34 | -34.1 / -34.0 / -33.9 / -33.9 | **+12.2 / +10.5 / +7.1 / +0.1** | +7.9 / +6.2 / +2.7 / -4.2 |
| | 2.6 | 32 / 45 / 60 | -50 / -46.5 / -43 | -40.7 (all) | **+9.3 / +5.8 / +2.3** | +2.3 / -1.2 / -4.7 |
| Garrison 60 kHz Dabob Bay (19) | 2.1 | 10 / 30 / 50 / 60 | -50 / -40.5 / -40 / -38 | -43.3 (all) | +6.7 / -2.8 / -3.3 / -5.3 | -2.7 / -12.2 / -12.7 / -14.7 |
| | 4.1 | 20 / 30 / 50 / 60 | -28 / -27 / -26.5 / -25.5 | -33.0 / -33.0 / -32.9 / -32.9 | -5.0 / -6.0 / -6.4 / -7.4 | -9.2 .. -11.5 |
| | 6.2 | 30 / 50 / 60 | -23 / -21.5 / -20.5 | -24.1 / -23.6 / -23.3 | -1.1 / -2.1 / -2.8 | -5.0 .. -6.6 |
| | 10 (of 7.2-15.4) | 20 / 30 / 50 / 60 | -22 / -21 / -20.5 / -19.5 | -23.6 / -22.4 / -21.4 / -18.8 | -1.6 / -1.4 / -0.9 / +0.7 | -2.2 .. +1.6 |
| Lilly-McConnell 20-25 kHz open ocean (20), at 22.5 kHz | about 10 | 3 / 10 / 17 / 28 | -32 / -28 / -25 / -23 | -31.9 / -25.3 / -23.8 / -23.0 | +0.1 / +2.7 / +1.2 / 0.0 | - |
| | about 5 | 5 / 10 / 17 / 28 | -43 / -40 / -37 / -37 | -31.3 / -30.8 / -30.6 / -30.5 | **+11.7 / +9.2 / +6.4 / +6.5** | - |
| Wildt 25 kHz, 3 deg (16) | 3 / 4.3 / 5.5 / 6.5 / 7.8 / 9 / 12 | 3 | -60 / -51 / -45 / -39 / -35 / -30 / -30 | -40.7 / -34.5 / -30.7 / -30.7 / -31.2 / -31.8 / -33.2 | +19.3 / +16.5 / +14.3 / +8.3 / +3.8 / -1.8 / -3.2 | +13.8 .. -4.2 |

Saturation at 25 kHz, 3 degrees: ours -34.2 dB, McDaniel's delta = 0.072 gives -34.1; the Wildt plateau is
-30 (the overshoot and refraction are her stated reasons). Garrison's printed saturation curve
(-25.5 dB at 30 degrees, -23 at 60) equals our `bubble_saturation_db` at 60 kHz (-25.7, -23.3).

Reading: at 9-10 m/s every data set agrees with ours within about 3.5 dB. At 3-6.4 m/s ours lies between
open-ocean data (6-12 dB above them) and coastal data (1-7 dB below). [D97]'s beta_I brings the open ocean to
-1 .. +8 dB and the coast to -9 .. -15 dB.

### 3.4 (a) D-193: what these laws give at 68 kHz, 3 m/s (computed; band means of intensity, uniform in grazing, as `test_gallaudet_boundaries.py`)

| Model | 27-38 deg (printed -51.3) | 44-66 deg (printed -41.5) |
|---|---|---|
| ours, `apl_1994` (beta_I 2.4e-3) | -38.3 (+13.0) | -38.1 (+3.4) |
| ours, `hall_novarini` (2.1e-2) | -29.1 | -28.9 |
| eq. (17) with [D97] Table II beta_I (70 kHz fit scaled by (68/70)^0.8: 6.2e-4) + facet | **-44.1 (+7.2)** | **-43.4 (-1.9)** |
| same with McDaniel's delta | -44.5 | -43.7 |
| Farmer-Lemon (scaled f^0.8, 1.7e-2) | -30.0 (+21.3) | -29.9 (+11.6) |
| [D97] beta_I + [McD93] composite roughness (our assumptions, Sec. 3.5) | -43.4 (+7.9) | -25.8 (+15.7) |
| Urick-Hoover 60 kHz open ocean, interpolated to 3 m/s (read from Fig. 18, log-linear in wind) | about -48 | about -39.5 |

Answer: a published law explains part of it. [D97]'s beta_I removes all 3.4 dB at 44-66 degrees (leaving
1.9 dB for a roughness term) and 5.8 of the 13.0 dB at 27-38 degrees. The open-ocean 60 kHz measurements
of 1956 sit within about 2-3.5 dB of Gallaudet's printed means in both bands, so Gallaudet's low values are
not an outlier. Two further items bear on the rest, but neither proves it:
- averaging. Under [DP97]'s statistics the mean of dB values lies below the dB of the mean intensity by
  C_E k + k sigma1^2 / 2 = 3.4 to 4.5 dB (sigma_T 2.8 to 4.2 dB, eq. 6 with E(lambda)). If Gallaudet's
  band means are means of dB, his numbers are 3.4-4.5 dB below what our intensity-mean test compares
  them with;
- the omitted roughness term. Computed from [McD93]'s spectrum it would raise the 44-66 degree band to
  about -26 dB, 15.7 dB above Gallaudet's value and about 14 dB above Urick-Hoover's, so that version of
  the term cannot be right at low wind and steep angles (3.5).

### 3.5 (b) the small-scale roughness term we omit (computed; ours = `bubble_layer_surface` with `apl_1994`)

Implementation (scratch `rough.py`): eq. (8) with K_L = k/4, eqs. (23)-(25) for a fully developed sea,
W(K) = W~(K)/K (isotropic, from eq. 6), Gaussian tilt in the plane of incidence with x-slope variance
(1/2) int_0^{K_L} K^2 W~ dK, the result attenuated by exp(-2 beta_I / sin theta) ([D97] Sec. VI).
**Ours, not printed:** u* = sqrt(1.2e-3) U10; tau = 0.0736 cm^2; log = log10 with u* in cm/s; K_c where the
two branches of eq. (24) meet (gives 2.6, 3.0, 3.6, 4.4, 5.0 cm^-1 at 2, 3, 5, 10, 15 m/s); Gamma as
printed. Check against Fig. 10 (10 m/s): computed +66.4 dB re cm^3 at K_p, -17.6 dB at 1 cm^-1, -32.1 at
log K = 0.6, cut-off 4.4 cm^-1 (log 0.65): **matches the readings within about 1 dB**. Check against
[D97] Fig. 1 (Ku HH - 11 dB, Bragg wavenumber of 68 kHz): 30 degrees, 5 m/s: computed -47.9 against about
-49 (agrees); 60 degrees, 5 m/s: -21.9 against about -30 (**8 dB high**). Against [D97] Fig. 8 (45
degrees, attenuated term): computed -34 / -32 / -31 / -32 dB at 3 / 5 / 8 / 10 m/s (26.5 kHz) against
about -41 / -37 / -35 / -34.5: 3-7 dB high at low wind. The large-scale x-slope variance at K_L = k/4 is
0.042 at 30 kHz and 10 m/s, so the total, 0.084, is 1.5 times our Cox-Munk 0.054: the spectrum is steeper
than the optical regression.

dB added to ours by the term (cut-off included; without the cut-off only the 68 kHz entries change, as noted):

| f | U10 | 30 deg | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|---|
| 12 kHz | 3 | 2.2 | 5.7 | 10.9 | 17.3 | 8.2 |
| | 5 | 0.4 | 1.3 | 3.7 | 8.2 | 2.4 |
| | 8 / 10 / 15 | <= 0.1 | <= 0.4 | 0.8-1.3 | 0.7-3.0 | 0.6-1.0 |
| 30 kHz | 3 | 2.1 | 5.2 | 9.6 | 15.0 | 6.9 |
| | 5 | 0.4 | 1.3 | 3.3 | 7.0 | 2.1 |
| | 8 / 10 / 15 | 0.1 | <= 0.4 | 0.8-1.2 | 0.7-2.7 | 0.5-0.9 |
| 68 kHz | 3 | 0.1 (1.2 no cut-off) | 0.7 (3.7) | 4.5 (8.2) | 14.3 | 8.0 |
| | 5 | 0.0 | 0.3 | 2.5 | 6.9 | 2.8 |
| | 8 / 10 / 15 | 0.0-0.1 | 0.3 | 0.8-1.2 | 0.9-2.9 | 0.7-1.2 |

Against data: at 60 kHz and 60 degrees ours plus this term is +11.2 dB (3.9 m/s), +18.3 dB (2.6 m/s)
above Urick-Hoover, +3.6 dB at 9 m/s; at 20 kHz, 7 m/s, 62.5-82.5 degrees (Nuetzel) it improves ours
from -4.5 / -1.3 / -3.3 to -1.5 / -0.3 / +0.2 dB. Conclusion: at 8 m/s and above the term is at most
about 3 dB and fits the one high-angle data set; at 5 m/s and below its size depends on the capillary
spectrum and cut-off, which neither paper fixes, and the version computed here is contradicted by the
open-ocean data by 10-18 dB at 60 degrees. **The D-87 / D-132 phrase "lower bound at 30-70 degrees" is
supported at high wind only.** Above about 75 degrees the perturbation term is meaningless (+25.6 dB at
88 degrees); the facet term holds there.

### 3.6 (c) the FLS band, 120-330 kHz (computed; nothing in these papers is validated there)

| f | U10 | Grazing 5 / 10 / 20 / 30 / 45 / 60 | Chapman-Harris (shipped) | ours bubble+facet | [D97] beta_I scaled f^0.8 + facet | [McD93] composite (cut-off) |
|---|---|---|---|---|---|---|
| 120 kHz | 5 | | -36.0 / -31.6 / -27.3 / -24.7 / -22.1 / -20.3 | -30.6 / -28.6 / -27.6 / -27.3 / -27.1 / -26.9 | -32.6 .. -31.0 | -82.7 / -78.6 / -70.6 / -62.6 / -49.0 / -30.9 |
| | 10 | | -24.9 .. -14.4 | -34.2 / -29.9 / -25.0 / -23.0 / -21.6 / -18.6 | -34.3 .. -18.5 | -71.3 .. -21.9 |
| 170 kHz | 5 | | -34.0 .. -19.4 | -31.1 .. -26.3 | -32.3 .. -30.4 | -91.4 .. -40.0 |
| 330 kHz | 5 | | -30.4 / -26.8 / -23.2 / -21.1 / -19.0 / -17.5 | -32.9 / -29.0 / -26.8 / -26.0 / -25.5 / -25.2 | -32.3 .. -29.2 | -107.9 .. -57.3 |
| | 10 | | -19.9 .. -11.3 | -35.7 .. -19.4 | -35.7 .. -20.2 | -96.2 .. -48.3 |

Chapman-Harris minus ours at 5 m/s: 120 kHz -5.4 / -3.0 / +0.4 / +2.6 / +5.0 / +6.6 dB; 330 kHz +2.5 / +2.2 /
+3.6 / +4.9 / +6.5 / +7.7 dB; at 10 m/s +4 to +16 dB at every angle and frequency. Against the [D97]-based
bubble term Chapman-Harris is up to 11.7 dB higher (5 m/s) and 15.8 dB higher (10 m/s).

What the papers do say about frequency: Lilly and McConnell found no frequency dependence at 15-60 kHz
([McD93] p. 1915); the saturation level falls only as delta (f^(1/3), or McDaniel's sqrt-f law, which
differs from ours by 0.6 dB at 120 kHz and 1.2 dB at 330 kHz); beta_I rises about as f^0.8 between 26.5
and 70 kHz ([D97]); the resonant radii at 120-330 kHz are 27-10 micrometres, below the smallest radius
counted in any population shown (about 16-20 micrometres, [McD93] Fig. 12; 16-116 micrometres for
Farmer-Vagle, [D97] p. 2593). The Bragg wavenumber at 120 kHz exceeds the spectrum's cut-off (about 3-5
rad/cm) below about 60 degrees, so with [McD93]'s spectrum the roughness term collapses above about
100 kHz except near normal incidence. **Nothing in these papers covers above 70 kHz**; the extrapolations
above are ours.

### 3.7 Statistics, time scales, Doppler

- [DP97] in our terms: the scintillation index of the lognormal-modulated exponential is
  2 exp(sigma1^2) - 1 (ours, from eq. 2): 4.0, 3.6, 2.7, 2.3, 2.0 at 3.8, 5, 8, 10, 11.6 m/s (from the
  sigma_T law), 3.1 at the representative 3.7 dB. A K distribution with the same index has nu = 2/(SI - 1)
  = 0.66 to 2.0 (ours). Total standard deviation of s: 6.2-7.0 dB. Gallaudet's indices 3.7 and 14 (D-193)
  are of the same order at the low end.
- Time scales: independent pings above about O(10) ms at 50 kHz; slow modulation over seconds to
  minutes. No Doppler spectrum is given in any of the three papers ([D97] cites Igarashi and Stern 1971,
  15-60 kHz, for long-wave Doppler, no numbers). D-84 (Gaussian Doppler) and D-85 (facet decay) are
  neither supported nor contradicted.

## 4. What this closes or changes in open issues

Unanswered first: **nothing here covers the surface above 70 kHz** (data end at 60 kHz in [McD93] and
70 kHz in [D97]); the handbook's G(U) of sigma_sc = G tan^4 is in none of the three; no near-specular
forward-scatter model, coherent reflection loss or time/angle spread is in any of them.

### sonar-9le (high-frequency surface)
- Small-scale roughness term: the published composite family is now in hand ([McD93] eqs. 4, 8, 23-25;
  [D97] eqs. 9-16, spectrum not printed) with checks (Fig. 10, Ku band). Its size (3.5): <= 3 dB at
  >= 8 m/s, 30-70 degrees, 12-68 kHz; 2.5-17 dB at 50-60 degrees and 3-5 m/s by [McD93]'s spectrum, which
  the open-ocean 60 kHz data contradict. Still wanted: the handbook's G(U) (or Plant's spectrum), and a
  low-wind capillary cut-off law.
- Model above 70 kHz: **not answered**. Physics stated here (bubble saturation ~ delta; roughness Bragg
  term collapsing beyond the viscous cut-off) supports the structure bubbles + facets, not any numbers.
- Near-specular scattering: [McD93] eq. (7) equals our facet term and our `reverb_specular` lobe at
  backscatter; no forward/near-specular model. TR 0702's "McDaniel" near-specular model is not in this
  review; the candidate in its references is McDaniel 1986 (diffractive corrections to the
  high-frequency Kirchhoff approximation, with the printed factor-2 erratum). Wanted.

### sonar-64sm (the D-193 follow-up)
- The optical-depth law at 3 m/s against another source: done. [D97] Table II gives a beta_I 5.9 dB
  below `apl_1994` at 68 kHz, 3 m/s, and Farmer-Lemon (scaled) one 8.5 dB above: the two published laws differ by 14 dB there.
- A second open low-wind measurement with stated conditions: Urick-Hoover 60 kHz open ocean (via
  [McD93] Fig. 18, curves; averaging not stated) agrees with Gallaudet within about 2-3.5 dB at 3 m/s;
  Lilly-McConnell (20-25 kHz, 5 m/s) is 6-12 dB below ours. So our model, not only the measurement, is
  high at low wind in the open ocean. Coastal waters run ~10 dB higher ([McD93] p. 1917), which our
  single wind law cannot express.
- Averaging: [DP97] eqs. (6)-(7) quantify the dB-versus-intensity bias (3.4-4.5 dB).
- "Lower bound" wording: true of the omitted term only at >= 8 m/s (3.5).

### sonar-6fc (K texture)
- For the surface: [DP97] supplies a measured compound model (exponential within a ping, lognormal over
  ~10 s) with sigma_T(U10), and scintillation indices 2-4 (equivalent K nu 0.66-2). Not for the seabed.

### docs/REFERENCES.md "wanted" rows (Sec. 10)
| Row | Result |
|---|---|
| "The handbook's law for beta_I and its small-scale roughness function G(U)" | beta_I(U, f): a published fit, [D97] Table II (26.5/50/70 kHz, 17 degrees, North Sea, 2.5-13 m/s), confirmed. Whether it is the handbook's law: not stated. G(U): **not in these papers**. Interpolation facet/small-scale: **not printed** (composite with K_L = k/4; McDaniel 1986 at high angles) |
| "Chapman-Harris fit and its range" | **via secondary** ([McD93] eq. 27, p. 1918): formula identical to ours; 400-6400 Hz, to 15 m/s, low grazing angles, explosive sources, March 1961 north of Bermuda; McDaniel: it exceeds most other low-frequency data. Upgrade `acoustics.chapman_harris_db` and the Sec. 4 row from unconfirmed to via secondary |
| "Bubble loss per surface bounce" | partly: [D97] Sec. VI gives the two-way loss through the layer as 20 log10(e) beta_I / sin theta dB, with beta_I from Table II, e.g. 70 kHz, 10 m/s: 3.5 dB / sin theta (`apl_1994`: 1.7 dB / sin theta); a backscatter inversion, not a measured loss |
| "Coherent fraction, delay spread, angle spread and scintillation of forward scatter" (Dahl's other papers) | **not in these three** |
| "Doppler spread against grazing angle" | not in these three |

## 5. Work packages

Ordered by value per effort. None changes a default without the owner. Shipped exposure, read from
`configs/scenarios/*.yaml`: **no shipped scenario uses `bubble_layer_surface` or `kraus_surface`.**
18 use `{model: chapman_harris}`: FLS at 120/170/330 kHz, 5 m/s (`fls_120k_transit`,
`fls_120k_cluttered_seabed`, `fls_170k_transit`, `fls_170k_traffic`, `fls_330k_transit`); flank_32 6-10 kHz
(`flank_patrol` 5 m/s, `flank_sst_shallow` 6, `flank_sst_spread` 6, `sst_pursuit` 8, bistatic);
`popoto_far_mover` 6-10 kHz, 5 m/s; popoto_long_range 4-6 kHz, 5 m/s from the environment (`popoto_12km`,
`_2row`, `_3row`, `_two_ships`, `popoto_env_*` x4). The bubble-loss default is `apl_1994` (builds) and two
scenarios type `bubble_loss_db: 0.5`.

**WP1 (S): tests that hold what is now confirmed; documentation status.** Tests first (characterisations,
no behaviour change), new `tests/python/test_surface_papers_1997.py`: [D97] Table I against
`bubble_saturation_db` at 17 degrees (-26.82/-27.74/-28.22 against -26.9/-27.8/-28.3, tolerance 0.15 dB);
`bubble_damping_constant(f) == 2.55e-3 f_Hz^(1/3)`; `facet_backscatter` equals [McD93] eq. (7) with
sigma_x'^2 = mss/2 (1e-9 relative); `chapman_harris_db` equals eq. (27) (1e-9 dB) and its exponents 2.46,
0.82 at 30 degrees; eq. (19) weak limit and eq. (18) limit of `bubble_backscatter`; the bracket peak
+3.7 dB within McDaniel's 3-4 dB. Docs (proposed text for the orchestrator): REFERENCES Sec. 4
Chapman-Harris row -> via secondary with the range; the Gallaudet row's formula -> confirmed in [D97];
new rows for [McD93], [D97], [DP97] (confirmed); `acoustics.chapman_harris_db` docstring: drop "below
about 40 degrees" (not printed), give McDaniel's range. Files: tests, docstrings only.

**WP2 (M): `extinction: dahl_1997` for `bubble_layer_surface` (opt-in).** Formula: beta_I = 10^(a + b
log10 U10) from [D97] Table II, interpolated in log f between 26.5, 50 and 70 kHz (ours), and scaled as
f^0.8 outside 26.5-70 kHz with a warning (the f^0.8 is [D97]'s own, p. 2592; outside their band it is
ours); a warning below 2.5 m/s (their dashed range). Files: `sim/surface.py`
(`bubble_layer_optical_depth`), `sim/simulator.py` (accept the key), C++ only if surface reverberation
has a C++ twin (check `libs/`; the live-chain parity rule applies if so). Tests first: Table II values at
the three frequencies; eq. (17) inversion round trip; at 68 kHz, 3 m/s the Gallaudet bands -44.1 / -43.4
dB (from -38.3 / -38.1); the Urick-Hoover readings of 3.3 within +8 dB; n between 5.0 and 6.3 over
2-7 m/s at 17 degrees. Default stays `apl_1994`: switching it needs the owner (before -> after at
68 kHz, 3 m/s: -38.3 -> -44.1 dB at 27-38 degrees; grid in 3.2; no shipped scenario affected).

**WP3 (M): a site parameter for coastal versus open ocean?** Owner first. [McD93] documents ~10 dB at
3-60 kHz before saturation and attributes 5 dB of it to whitecap coverage (factor 3.5); no formula is
given. A scalar multiplier on beta_I named for the site is the only form these papers support; do not
add it silently or tune it to one data set.

**WP4 (M-L): small-scale roughness term, opt-in only and as a diagnostic first.** Formulas: [McD93] eqs.
(4), (8), (22)-(25) with the assumptions of 3.5 stated in code (u*, tau, K_c by continuity, Gamma), the
two-way attenuation of [D97] Sec. VI. Files: new functions in `sim/surface.py`, an option on
`bubble_layer_surface` (`roughness: mcdaniel_1993`), default off. Tests first: spectrum against Fig. 10
readings (+66 dB at K_p, -17 dB at 1 cm^-1, cut-off between 4 and 5.5 cm^-1, +-2 dB); the eq. (4) /
[D97] eq. (10)/4 pi identity; K_L = 0 limit equals eq. (4); Ku comparison at 30 degrees (5 m/s, within
3 dB of -49 dB); **an expected-failure test** documenting the 11-18 dB excess over Urick-Hoover at
60 degrees, 2.6-3.9 m/s. Do not make it default: it worsens D-193 at 44-66 degrees (to -25.8 dB) and
contradicts open-ocean data at low wind. Needs the handbook's G(U) or Plant's spectrum before it can be
recommended. Dependency: WP2 (the attenuation uses beta_I).

**WP5 (M): surface reverberation statistics from [DP97] (opt-in).** A per-item texture: lognormal mean
with sigma_T = -0.18 U10 + 4.85 dB (3.8-11.6 m/s; clamp and warn outside) held over a correlation time
of ~10 s (T of [DP97]), exponential intensity within it; pings independent beyond O(10) ms. Files:
`sim/reverb.py` (`CellTexture` or a sibling), docs. Tests first: Var(s) = 5.57^2 + sigma_T^2 and
E(s) = mu_T - 2.507 dB on 1e5 samples (0.1 dB); SI = 2 exp((sigma_T/k)^2) - 1. Spatial correlation of
the modulation: **not given** (state it; tie to sonar-6fc). Default unchanged (Rayleigh).

**WP6 (S): facet term near normal.** Nuetzel 20 kHz (Fig. 20) is 3-4 dB above our facet term at
82-88 degrees; McDaniel's curve uses her 1986 diffraction corrections (not in hand). Action now: a test
recording the gap (characterisation) and the "wanted" row for McDaniel 1986; no code change. Note: the
spectrum's slope variance (3.5) is larger than Cox-Munk, so the fix is not simply "filter the slopes";
read the 1986 paper first.

**WP7 (S, owner decision): FLS scenarios' surface model.** These papers do not validate anything above
70 kHz. If the owner wants the FLS scenarios off Chapman-Harris, the surface strength would change (after minus
before, `bubble_layer_surface` minus Chapman-Harris) at 5 m/s by +5.4 / +3.0 / -0.4 / -2.6 / -5.0 / -6.6 dB
at 5 / 10 / 20 / 30 / 45 / 60 degrees at 120 kHz, and by -2.5 to -7.7 dB at 330 kHz; records of the five FLS scenarios would change. Prepare the numbers, ask, do not
switch.

Must not be done: change `extinction`'s default, add the roughness term by default, or tune any constant
to one figure reading; figure readings are +-1.5 dB and only for tests with that tolerance.

## 6. Not verified / unclear on the page

- [McD93] eq. (22) printed "Omega = gK(1 + tau K^2)": should be Omega^2 (dimensions); tau's value not
  printed.
- [McD93] Gamma in eq. (23) printed exp[-(K/K_p)^(1/2) / (2 s^2)]: probably a misprint of the Donelan
  form; immaterial at acoustic Bragg wavenumbers.
- [McD93] eq. (24): units of u* and of "log" in P = 5 - log(u*) are not printed (cgs assumed); K_c has
  no value or law; the range is printed "0.942 cm <= K", the unit cm^-1 elsewhere.
- [McD93] text refers to "Eq. (28)" for Chapman-Harris; only eq. (27) exists.
- [McD93] eq. (7) assumes isotropy with sigma_x' the x-component slope; our identity used sigma_x'^2 =
  mss/2.
- [D97] whether Table II's 17-degree, North-Sea fit applies at other angles and sites is not tested in
  the paper beyond Fig. 8 (45 degrees, figure only); the data were forced to saturation (calibration
  uncertainty), so the absolute low-wind level carries that forcing.
- [D97] Fig. 1 and Fig. 5 microwave levels, Fig. 8 curves, and every [McD93] curve used here are read
  from figures, +-1.5 dB (+-2 dB on the scanned [McD93] Figs. 18-20 at the curve ends).
- The wind heights of Urick-Hoover, Garrison and Lilly-McConnell are not given ([McD93] p. 1914:
  corrections usually under 1 m/s); Nuetzel's are at 47 m.
- Whether TR 0702's "McDaniel" near-specular model is McDaniel 1986: not determinable from these papers.
