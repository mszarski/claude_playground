# Bistatic and near-specular sea-surface scattering: Dahl (1999) and McDaniel (1986), extracted for the simulator

Continues `surface_papers_mcdaniel_dahl.md` (McDaniel 1993 [McD93], Dahl and Plant 1997 [DP97], Dahl
et al. 1997 [D97]); what that file records is not repeated. Both papers are all rights reserved:
formulas re-typed, constants with page, equation, table or figure number, own words; the PDFs are not
in the repository. Status words as in `docs/REFERENCES.md`: **confirmed** = read on the page image of
that paper; **via secondary** = another author's formula as this paper reprints it; **ours** = our
assumption or derivation; **read from figure** = a value read off a printed curve (about +-1.5 dB,
+-2 dB on the scan; never a test bound tighter than that).

- [D99] P. H. Dahl, "On bistatic sea surface scattering: Field measurements and modeling",
  J. Acoust. Soc. Am. 105 (4), 2155-2169 (April 1999). 15 pages, typeset; received 9 March 1998,
  revised 6 November 1998, accepted 11 December 1998.
- [McD86] S. T. McDaniel, "Diffractive corrections to the high-frequency Kirchhoff approximation",
  J. Acoust. Soc. Am. 79 (4), 952-957 (April 1986). 6 pages, a scan; received 18 March 1985, accepted
  10 December 1985. [McD93] p. 1921 records an erratum: the right-hand side of its eq. (18) is to be
  multiplied by 2.

Numbers in sections 3-5 come from scratch scripts outside the repository
(`/tmp/claude-0/-home-user-claude-playground/32e65dcd-82ae-5f75-a363-7372d2e7f847/scratchpad/jr/dahl99.py`,
`kcheck.py`, reusing `rough.py` of the earlier file), importing a copy of this worktree's `gpusonar` in
the scratchpad with the built module linked in. Nothing in the worktree was changed except this file.

## 1. Scope

Risky first: [McD86] is a scan; its equations were read from 250-dpi renders and two of them look
misprinted or incomplete (Sec. 6). [D99]'s model of the rough-surface part (sigma_r) needs the
measured directional wave spectrum blended with Plant's spectrum, and **that spectrum is not printed**
(only plotted, Fig. A2): every sigma_r number in this file uses a stand-in spectrum of ours (Sec. 3.3).
Nothing in either paper is above 60 kHz ([McD86] says its results are "very similar" up to 60 kHz) and
[D99] is one frequency (30 kHz), winds of 1-4 m/s, grazing 13-20 degrees, forward scatter only.

| Paper | Pages read | How |
|---|---|---|
| [D99] | 2155-2169, all 15 (incl. appendices, references) | every page image; eqs. (1)-(4), (11)-(13), App. B and (A2)-(A3) re-rendered at 250-300 dpi |
| [McD86] | 952-957, all 6 (incl. appendix, references) | every page image; eqs. (8)-(18) and Q_L re-rendered at 250 dpi |

## 2. What the papers give

### 2.1 Conventions

| Item | [D99] | [McD86] | Our code |
|---|---|---|---|
| Angles | grazing, theta_i incident, theta_s scattered; bistatic angle phi_b, **0 = forward in-plane** (eq. 3, Fig. 4) | grazing phi (Fig. 1); k = k r/r and k' = k r'/r' point from the surface patch towards source and receiver | grazing; bistatic azimuth 0 forward, pi back (`reverb_bistatic.bistatic_azimuth`, `reverb_specular.facet_lobe`): **same as [D99]** |
| Cross section | sigma per unit area per unit solid angle, scattering strength 10 log sigma | S = 10 log10[<P P*> (r r')^2 / A] (eq. 19), A the illuminated area (App. eq. A6: A = pi U V for a Gaussian beam) | same |
| Wave-vector difference | (alpha, beta, gamma) = (k_i - k_s)/(2k) by eq. (3) (the text says "/k"; the printed components carry the 1/2) | i(k_perp + k'_perp).r = i(alpha x + beta y): alpha, beta are the full horizontal components | - |
| Height spectrum | F(K) directionally averaged, H^2 = int_0^inf K F(K) dK (eq. 8); so F = 2 pi psi for an isotropic 2-D spectrum psi with int psi d2K = H^2, and F = W~_[McD93] / K | W(K, theta) directional, sigma^2 = (1/2 pi) int int W d2K (eq. 9 at r = 0): W = 2 pi psi, the [McD93] normalisation | bottom: J&R psi; surface: Pierson-Moskowitz frequency spectrum (time spreading) |
| Slope variance | s_L^2 = int_0^{K_sep} K^3 F dK (eq. 9): both axes together, large scales only | sigma_x', sigma_y' rms slopes per axis of the large-scale surface; isotropic: sigma_x' = sigma_y' = 2^(-1/2) sigma' (p. 955) | `mean_square_slope` both axes (Cox-Munk), s2/2 per axis |
| Boundary | pressure release implied (SSA conversion and eq. 7 are the Dirichlet forms; check Sec. 3.1) | "pressure release" (abstract, eq. 1) | surface coefficient -1 |
| Height statistics | Gaussian (Kirchhoff integral, eq. 1) | Gaussian heights (before eq. 7); slopes and heights independent | Gaussian |
| Time dependence | not stated (intensities only) | not stated; Green's function (r)^-1 exp(i k.(r - r'')) | - |
| Wind | 6.9 m above the sea (propeller vane anemometer), 5-min averages every 10 min; **not corrected to 10 m** | "10 kn" (Pierson spectrum, U at 19.5 m in Fig. 2) | U10 |

### 2.2 [D99] the experiment (Sec. I, Table I) - confirmed

13-16 February 1995 near the Dry Tortugas (24 deg 36.7' N, 82 deg 50.7' W), water depth 25.6 m,
carbonate sand-silt-clay bottom, sound speed about 1520 m/s, well mixed. CW pulses of 1 ms at 30 kHz,
10 or 20 pings at 5-s intervals, from one of two omnidirectional transducers (ITC-1032) on a spar buoy at
7.7 and 17.2 m (buoy period 12 s; dominant waves about 6 s), 70-100 m from the receiving ship. Eight-element
horizontal line array transverse to the path, 7-10 m deep, separations 1.59-57.15 cm (two elements
failed: maximum 31.75 cm, 15 spacing combinations). Waves: Datawell directional Waverider about 100 m away,
spectra hourly, resolved to 0.58 Hz. Bottom-interacting paths lost 20-25 dB (bottom critical angle about
10 degrees); the single surface bounce lost "essentially 0 dB".

| Run | Source depth (m) | Receiver depth (m) | Range (m) | theta_g (deg) | Wind (m/s, at 6.9 m) | H rms (cm) | chi | s_L | kd*/(2 pi) | sigma_theta_h (deg) | Decay L (ms, p. 2165) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 17.2 | 6.9 | 107 | 13 | <= 1 | 8 | 4.4 | 0.060 | 20.0 | 0.45 | 0.03 |
| 2 | 7.7 | 9.5 | 72 | 14 | 4 | 16 | 9.5 | 0.130 | 12.0 | 0.76 | 0.21 |
| 3 | 17.2 | 9.5 | 72 | 20 | 4 | 16 | 13.4 | 0.130 | 4.9 | 1.87 | 0.31 |
| 4 | 7.7 | 10.0 | 75 | 13 | 3.8 | 16 | 9.1 | 0.134 | 11.2 | 0.82 | 0.21 |

Checks (ours): chi = 2 k H sin theta_g with c = 1520 m/s gives 4.46 / 9.6 / 13.6 / 9.1; sigma_theta_h =
1/(kd*) in radians gives 0.46 / 0.76 / 1.86 / 0.81 deg: the table is self-consistent. The s_L column is
the rms slope (s_L^2 = 0.0036, 0.0169, 0.0169, 0.018), matching "s_L^2 (0.017)" on p. 2160.

Sea state (p. 2161, App. A): spectra peaked near 0.16 Hz (swell from the Gulf of Mexico, wave age c_p/U
about 2.5 for runs 2-4 and about 10 for run 1; fully developed Pierson-Moskowitz is 1.2), wind over the
previous 12 h about 6 m/s; run 1 had mixed directions (below 0.35 Hz from 225 deg, above from 20-50 deg).
The coherent intensity is "negligible ... in all but mirrorlike sea surface conditions" at 30 kHz (p. 2155).

### 2.3 [D99] the bistatic cross section model (Sec. II) - confirmed

sigma = sigma_r alpha_b + sigma_b (11): roughness term attenuated by the bubble layer, plus bubble scattering,
added incoherently.

**Roughness, Kirchhoff integral (eqs. 1-4):**

    sigma_KA = (k^2 F^2 / (2 pi)^2) int int dxi dvarsigma exp(2 i k (alpha xi + beta varsigma)) exp(-chi^2 [1 - C(xi, varsigma)])   (1)
    chi = k H (sin theta_i + sin theta_s)                                                                      (2)
    alpha = (cos theta_i - cos theta_s cos phi_b)/2,  beta = -(cos theta_s sin phi_b)/2,  gamma = -(sin theta_i + sin theta_s)/2   (3)
    F = (alpha^2 + beta^2 + gamma^2) / gamma
    isotropic:  sigma_KA = (k^2 F^2 / (2 pi)) int_0^inf rho J0(2 kappa rho) exp(-chi^2 [1 - C(rho)]) drho,  kappa = k sqrt(alpha^2 + beta^2)   (4)

C is the normalised height autocorrelation; the lag axis xi is along the source-receiver line, so
sigma is a function of (theta_i, theta_s, phi_b) and a new C(xi, varsigma) is needed for each direction.

**Second-order small-slope approximation, used for sigma_r (eqs. 5-6):**

    g = (gamma sin theta_i - alpha cos theta_i) / (sin theta_i sin theta_s),    sigma_SSA(2) = sigma_KA / g^2

g = 1 at the specular direction (sigma_KA exactly), and for small chi the result is first-order
perturbation theory:

    sigma_PT = 4 k^4 sin^2 theta_i sin^2 theta_s F(K = 2 kappa) / (2 pi)          (7)

(anisotropic: F(2 kappa)/2 pi -> W(2 k alpha, 2 k beta), the 2-D spectrum). The SSA's stated advantage: one
expression across the angles, no matching of Kirchhoff and perturbation results; the cited alternative
"McDaniel" (ref. 10 = [McD86]) still needs a spectral split (p. 2157).

**Tilt correction (eqs. 9-10, App. B).** K_sep = 0.25 k (after McDaniel and McCammon 1987; radar uses
0.20 k; "nearly transparent" difference), s_L^2 = int_0^{K_sep} K^3 F dK (9). Added to sigma_PT:
4 k^4 T1 F(2 kappa)/(2 pi) with

    T1 = s_L^2 [ (1/2) sin^2 theta_i cos^2 theta_s + (1/2) sin^2 theta_s cos^2 theta_i
                 - 2 sin theta_i cos theta_i sin theta_s cos theta_s cos phi_b - sin^2 theta_i sin^2 theta_s ]   (10)

from the expectation of sin^2(theta_i + Delta_i) sin^2(theta_s - Delta_s) (B1), Delta_i = delta cos psi,
Delta_s = delta cos(psi - phi_b), slopes Gaussian with variance s_L^2/2 each, psi uniform. Hydrodynamic
modulation and shadowing are left out (p. 2159). In this experiment the tilt correction is invisible
(small s_L^2, near-forward geometry).

**Bubbles (eqs. 12-13; [D99] ref. 2 = Dahl, JASA 100, 748-758 (1996), whose eq. (12) is "printed with an error"):**

    alpha_b = exp(-(beta_I / sin theta_i + beta_I / sin theta_s))                                           (12)
    sigma_b = (beta_I / 4 pi) (delta_r / delta) [ (1 - exp(-2 (beta_I/sin theta_i + beta_I/sin theta_s))) / (beta_I/sin theta_i + beta_I/sin theta_s)
              + (exp(-2 beta_I/sin theta_i) - exp(-2 beta_I/sin theta_s)) / (beta_I/sin theta_s - beta_I/sin theta_i) ]   (13)

beta_I dimensionless, "the depth-integrated extinction cross section per unit volume"; delta_r = 0.0136;
delta "approximately 0.0792 at 30 kHz" (p. 2159). The monostatic version is [D97]. Used value:
**beta_I = 0.0012**, from surface backscatter at low grazing angle measured during the experiment ([D99] ref. 21: Dahl,
JASA 100, 2840 (1996)), inverted as in [D97], taken as an **upper bound** (p. 2161).

### 2.4 [D99] spectrum and correlation function (App. A) - confirmed; the blended spectrum is not printed

- Buoy spectrum S(omega) to 0.58 Hz, extended to 2.8 Hz (= K_sep = 31 rad/m through omega^2 = g K) with
  S ~ omega^-n(f), **n(f) = 0.4504 f + 3.7390** (A1), f in Hz: -4 at 0.58 Hz to -5 at 2.8 Hz (Banner 1990:
  A K_p < 0.01 for all spectra, so omega^-5 asymptotically).
- Directional spreading, Donelan-Hamilton-Hui with Banner's high-frequency change:
  D(phi; omega) = (b/2) sech^2[b (phi - phi_bar(omega))] (A2); b = 1.24 (omega/omega_p < 0.56),
  2.61 (omega/omega_p)^1.3 (0.56-0.95), 2.28 (omega/omega_p)^-1.3 (0.95-1.6), and for > 1.6 a power of ten
  printed as 10^(-0.4 + 0.8393 exp[log_e[(omega/omega_p)^2]]) (A3) - **misprinted** (Sec. 6). Mean direction
  from the buoy to 0.58 Hz, held at the mean of the last four bins beyond.
- C(xi, varsigma) = (1/H^2) int_{-pi}^{pi} dphi int_0^{5.6 pi} domega S(omega) D(phi; omega) cos[(omega^2/g)(xi cos phi +
  varsigma sin phi)] (A4); isotropic C(rho) = (1/H^2) int K J0(K rho) F(K) dK (A5).
- F(K) for C(rho): buoy segment blended with Plant's (1986) model above K = 10 rad/m (U = 4 m/s, fetch 20 km;
  fetch "immaterial" for K > 10 when fetch > 1 km), to K = 500 rad/m with a Gaussian taper; overlap K 1-10,
  slope about K^-4 there (Fig. A2). Total mean-square slope to K = 500: **0.024**; s_L^2 = 0.017 for both
  versions (eq. 14, p. 2160). Surface tension matters only for wavelengths <= 7 cm (> 4.7 Hz).
- Bragg wavenumbers of runs 2-3 lie roughly K 30-200 rad/m (dashed segment of Fig. A2, read from figure).

### 2.5 [D99] what the model shows (Sec. III) - confirmed statements; curves read from figure

- "Visibility function" exp(-chi^2 [1 - C]): its 0.1 contour (10 dB) defines an "effective correlation
  boundary". Run 2 (chi 9.5): nearly circular, **diameter 1.1 m** (Fig. 5b), so near-specular scattering is
  independent of direction; runs 3-4 similar (chi about 10 or more). Run 1 (chi 4.4): C must fall only to
  0.88 and the boundary is an ellipse with its semi-minor axis about 75 deg from xi (Fig. 8b); rotating
  C by 75 deg changes sigma by at most 2 dB (Fig. 9).
- 2-D low-pass (K <= K_sep = 31) and isotropic full (K <= 500) versions overlap in the near-specular region
  (Fig. 6); away from it the low-pass version falls faster (from about 10 and 60 m in range, 4 m out of
  plane, Fig. 6, run 2). The full SSA(2) merges with first-order perturbation away from specular and departs
  from it near specular, where perturbation theory fails.
- Bubbles (beta_I = 0.0012) matter only at bistatic angles phi_b >~ 30 deg, where they roughly equal the
  Bragg term; neither affects the spatial coherence here.
- Footprint (Fig. 7): the pattern width scales as **sin(theta_g)/(1 + RD/SD)** (SD, RD source and receiver
  depths).
- Read from figure (not for tests): sigma at the specular point about +15 dB (run 2, Fig. 6a) and about
  +21.5 dB (run 1, Fig. 9).

### 2.6 [D99] measurements against the model (Sec. IV) - confirmed

- Horizontal coherence: Var(|Gamma^|) = (1 - |Gamma^|^2)^2 / (2N) (15), N = 10 or 20 pings; summarised by
  Gamma(kd) = exp[-(kd sigma_theta_h)^2 / 2] (16), 1/(kd*) = sigma_theta_h, the rms horizontal arrival angle
  (values in Table I). Model curves lie "fully within the estimated error bars" in all four runs. Moving
  the source from 7.7 to 17.2 m (runs 2 -> 3) more than doubles sigma_theta_h, about as sin theta_g/(1 + RD/SD).
  An array longer than about lambda / (2 sigma_theta_h) gains nothing (run 3: about 75 cm).
- Intensity impulse response by summing the discretised surface (dA = 0.0625 m^2):
  I_imp(t) = I0 sum_n sigma_n 10^(alpha_w (R_in + R_sn)/10) dA / (R_in^2 R_sn^2) (17), alpha_w = 0.005 dB/m,
  I0 = 176 dB re 1 uPa at 1 m, convolved with the 1-ms pulse.
- Run 1: model matches the steady part and the next ~23 dB of decay; K up to 500 or only to 31 makes no
  difference. Run 2: both match the steady part and the next 20 dB; only the version with K up to 500
  (Bragg-scale waves) matches the further 10 dB of decay: **the late tail of forward scatter is set by the
  small-scale (Bragg) waves, the angular spread by the large-scale ones.**
- **Decay: I_imp ~ exp(-t/L) for t > L, L = 0.03 ms (run 1), 0.21 ms (runs 2 and 4), 0.31 ms (run 3)**;
  frequency coherence bandwidth about 1/(2L): about 2.4 kHz (run 2), 1.6 kHz (run 3), 17 kHz (run 1)
  (Kennedy 1969) (p. 2165).
- Model-data agreement "quite satisfactory" for C(xi, varsigma) or C(rho) alike.

### 2.7 [McD86] the diffractive correction (Secs. I-II, Appendix) - confirmed; two print problems (Sec. 6)

Physical optics for a pressure-release surface; far field (k |r - r''| >> 1), narrow beam:

    <P P*> = (A <(n.k')^2> / (2 (2 pi r r')^2)) int dr_perp exp{ i r_perp.(k_perp + k'_perp) - nu^2 [1 - C(r_perp)] }   (8)
    nu = (k_z + k'_z) sigma,   C(r) = (1 / (2 pi sigma^2)) int_0^inf K dK int_0^{2 pi} dtheta W(K, theta) cos(K.r)        (9)

The "1/2" in (8) comes from int dR_perp <[n''.k'][n'''.k']> = (1/2) A <(n.k')^2> (Appendix, Gaussian beam
B = exp(-(x^2/U^2 + y^2/V^2)) (A1), V^2 = (20 log10 e / 3) r'^2 tan^2(phi/2), U^2 = 20 log10 e r'^2
tan^2(phi/2) / (3 sin^2 theta'), phi the half-power beamwidth, A = pi U V (A6)); the paper says it differs
from Eckart (1953) and Beckmann-Spizzichino by this factor 1/2 and that Lynch and Wagner (1970) showed it
correct. [McD93] later asks for eq. (18) to be multiplied by 2 (Sec. 3.4 settles which is right).

Conventional composite (Sec. I.B): C_L(r) the same integral up to K_L (10), C_S = C - C_L (11), expand
exp(nu^2 C_S) (12); with C_L(r) ~ (sigma_L/sigma)^2 - (1/2) x^2 (sigma_x'/sigma)^2 - (1/2) y^2 (sigma_y'/sigma)^2 (13):

    <P P*>_C = Q_L + (A <(n.k')^2> (nu^2/sigma^2) / (4 (2 pi)^2 (r r')^2 mu_x mu_y)) int_0^inf dK_x int_{(K_L^2 - K_x^2)^(1/2)}^inf dK_y W(K_x, K_y)
               x [ exp(-(alpha + K_x)^2/(2 mu_x^2) - (beta + K_y)^2/(2 mu_y^2)) + exp(-(alpha - K_x)^2/(2 mu_x^2) - (beta - K_y)^2/(2 mu_y^2)) ]   (14)
    mu_x / sigma_x' = mu_y / sigma_y' = nu / sigma,   i (k_perp + k'_perp).r_perp = i alpha x + i beta y

Conditions: nu^2 (sigma^2 - sigma_L^2) << 1, i.e. (nu sigma_S / sigma)^2 << 1 (sigma_S rms height of the
small scales), and a large-scale radius of curvature large enough for (13). Small mu: the exponentials act
as delta functions and (14) gives the perturbation result of McDaniel-Gorman 1983 (ref. 10).

Corrected (Sec. II): keep the curvature terms exactly through

    sigma^2 C_L = sigma_L^2 - [(sigma_x' x)^2 + (sigma_y' y)^2]/2 + (1/2 pi) int_0^{K_L} K dK int dtheta W (cos K.r - 1 + (K.r)^2 / 2)   (15)

(16) the full exponent; assume its third (small-scale height) and fourth (curvature) terms small (17); then

    <P P*> = (2 - nu^2 - alpha^2/(2 mu_x^2) - beta^2/(2 mu_y^2)) Q_L
             + (A <(n.k')^2> (nu^2/sigma^2) / (4 (2 pi)^2 (r r')^2 mu_x mu_y)) int_0^inf int_0^inf dK_x dK_y W(K_x, K_y) [same two exponentials]   (18)
    Q_L = (A <(n.k')^2> / (4 pi mu_x mu_y (r r')^2)) exp(-alpha^2/(2 mu_x^2) - beta^2/(2 mu_y^2))

(as printed, the first exponential of (18) lacks its minus sign: Sec. 6). (18) has the form of (14) at no
extra cost; the partition K_L now enters only through mu (the large-scale slopes), the integral runs over
the whole spectrum. Shadowing is not included, so numbers are restricted to high grazing angles.

Numerical results (Sec. III; 10 kHz, wind 10 kn, Pierson-Stacy / Pierson 1976 spectrum, Fig. 2 for U19.5 =
3.5, 7, 12, 21, 34 m/s; isotropic large-scale slope) - printed numbers:
- Conventional composite backscatter (Fig. 4), K_L from 0.025 to 0.5 cm^-1: **15 dB change at 63 deg grazing,
  8 dB at normal incidence**.
- Corrected (Fig. 5), same K_L: "far less sensitive"; **greatest disparity 2 dB, at normal incidence**.
- No K_L brings the conventional result into agreement with the corrected one (unlike low-grazing backscatter,
  ref. 10).
- Forward scatter (Figs. 6, 7; K_L 0.025-1.0 cm^-1): the correction "substantially reduces" the dependence,
  less dramatically than for backscatter. "Very similar" results up to 60 kHz; higher wind (larger slopes)
  reduces the dependence.
- Conclusion: still "slightly sensitive" near normal incidence and in forward scatter: higher-order terms may
  be needed.

Read from figure (not for tests): Fig. 3 (K_L = 0.3 cm^-1): small-scale (Bragg) term from about -50 dB at 22
deg to about -28 dB near 68 deg; large-scale term from about -50 dB at 63 deg to about +3 dB at 90 deg.
Fig. 6 (conventional forward): flat in angle, about +8 / +5 / 0 dB for K_L = 0.025 / 0.05 / 1.0 cm^-1. Fig. 7
(corrected): about +5 / +4.5 / +3.5 dB.

What [McD86] does **not** give: wind laws for rms height, mean-square slope or vertical speed; any time or
angle spread; any coherent/incoherent split; any rule for removing near-specular reverberation; bubbles.
It is a correction to the high-frequency (geometric-optics) Kirchhoff result for backscatter and specular
forward scatter of a Gaussian, pressure-release surface with a directional (Pierson) spectrum.

## 3. Against our simulator

Wrong or doubtful first:

1. **Our horizontal angle spread of a surface bounce is 1.5-2.6 times too wide against [D99]'s measured
   sigma_theta_h, and our facet decay time 1.5 times too long in runs 2 and 4** (Sec. 3.3). The geometry of
   our facet model (spread ~ sin g x r1/(r1 + r2), Sec. 16 of PHYSICS_NOTES) is exactly [D99]'s stated
   scaling sin theta_g/(1 + RD/SD); the size is wrong because we use the Cox-Munk total slope (0.0235 at
   4 m/s), while the slopes that steer near-specular energy are those of waves longer than about the
   1.1 m correlation boundary: mss about 0.008-0.011 from [D99]'s numbers.
2. **Our `bubble_layer_surface` does not attenuate the facet term by the bubble layer**, which [D99] (11)-(12)
   and [D97] Sec. VI do: -0.6 to -3.2 dB at 80-89 deg and 8-15 m/s with our own beta_I (Sec. 3.2).
3. **No surface model of ours takes the bistatic angle**: `chapman_harris_strength` and
   `bubble_layer_surface_strength` evaluate a backscatter law at the arithmetic mean grazing angle. [D99]
   gives a published bistatic form for the bubble part (13) that our mean-angle rule overstates by 1.8 dB
   in the weak limit and by up to 8 dB for very unequal angles at high wind (Sec. 3.2).
4. Taken to backscatter, [D99]'s bistatic bubble term is 1.76 dB below [D97]'s (it drops the coherent
   doubling of the two once-reflected paths): the published monostatic and bistatic forms do not join.
5. `exclude_specular` removes a lobe whose peak is 2.9 dB below the published (SSA) lobe at the specular
   point and over-removes 0.5-3.5 dB a few degrees off it (Sec. 3.5); TR 0702's McDaniel model is not
   [McD86] by TR 0702's own citation as our notes record it (Sec. 4).

### 3.1 Item by item

| Item | Papers | Our code | Agreement (how checked) |
|---|---|---|---|
| Kirchhoff integral | [D99] (1), (4) | `bottom.kirchhoff_strength` (|V| = 1 for the surface) | **identical**: k^2 F^2 = |q|^4/(4 q_z^2), chi^2 (1 - C) = q_z^2 D(R)/2; numerically < 0.001 dB at six backscatter/forward/bistatic geometries, 30 kHz, power-law relief (kcheck.py) |
| SSA(2) | [D99] (5)-(6) | `bottom.small_slope_strength` (J&R 13.11) | algebraically the same for a pressure-release boundary: g = -(alpha^2+beta^2+gamma^2)/(sin theta_i sin theta_s) (max relative error 7e-16 over 1000 random geometries), and its small-chi limit equals (7) to 6e-15 dB. Our function cannot be run at rho -> 0 (it divides by rho), so the surface case was not run through our code |
| Perturbation (7) | [D99]; [D97] eq. 10; [McD93] eq. 4 | `bottom.perturbation_strength` (fluid); none for the surface | same physics; F = 2 pi psi = W~/K (2.1) |
| Geometric optics (GO) facet lobe | [McD86] Q_L x 2 with <(n.k')^2> = |q|^4/(4 q_z^2) | `reverb_specular.facet_lobe`, `surface.facet_backscatter` | **identical** (< 5e-13 dB over 1990 random bistatic geometries, s2 = 0.02). As printed in 1986 (without the erratum): -3.01 dB everywhere |
| Bubble attenuation alpha_b | [D99] (12) | `bubble_layer_surface` applies none to the facet; `time_spread` applies `bubble_loss` per bounce from a loss law | the same convention as [D97] Sec. VI (one crossing = beta_I / sin theta nepers of intensity); missing on our facet term (item 2 above) |
| Bistatic bubble term | [D99] (13) | `bubble_backscatter` at the mean grazing angle | differs (3.2) |
| delta at 30 kHz | 0.0792 (p. 2159) | `bubble_damping_constant(30e3)` = 0.0792 | **identical** |
| delta_r | 0.0136 | 0.0136 | identical |
| beta_I at 30 kHz, about 4 m/s | 0.0012, upper bound (p. 2161) | `apl_1994` 3.95e-3 (+5.2 dB), `hall_novarini` 1.68e-2 (+11.5 dB) | [D97] Table II interpolated in log f: 1.42e-3 at 4 m/s (+0.7 dB), 1.10e-3 at 3.8 m/s (-0.5 dB): **a second, independent point agrees with [D97]'s low-wind law and lies 4-5 dB below our default** |
| Wind height | 6.9 m | U10 | a 6.9 m wind is about 3 % below U10 over a neutral log profile (ours, z0 = 1e-4 m); not corrected in [D99] |
| rms height at 4 m/s | 16 cm measured (swell) | `rms_wave_height(4)` 8.6 cm (fully developed PM) | ours 0.54x; irrelevant at 30 kHz (coherent part e^-chi^2 < 1e-40 either way), relevant to `coherent_phase` and low frequency |
| Total mss at 4 m/s | 0.024 (to K = 500, blended spectrum) | Cox-Munk 0.0235 | **agrees (2 %)** - the total; what the near-specular lobe sees is smaller (3.3) |
| K_sep | 0.25 k | `bottom.large_scale_slope` (k h = 0.5 rule); [McD93] K_L = k/4 | [D99] uses McDaniel's k/4 |

### 3.2 (a) Backscatter: [D99]'s form at theta_i = theta_s, with [D97] beta_I, against ours (computed)

[D99] taken to backscatter: facet x alpha_b + sigma_b (13) with theta_i = theta_s, beta_I from [D97]
Table II (log-f interpolation; 68 kHz between the 50 and 70 kHz fits). Ours: `bubble_layer_surface_db`
(`apl_1994`). Difference [D99] minus ours, dB, grazing 5 / 10 / 20 / 30 / 45 / 60 / 80 deg:

| U10 | 30 kHz | 68 kHz |
|---|---|---|
| 3 | -8.0 -8.0 -8.1 -8.1 -8.1 -8.1 0.0 | -7.7 -7.8 -7.9 -7.9 -7.9 -7.9 0.0 |
| 4 | -6.0 -6.1 -6.1 -6.2 -6.2 -6.0 0.0 | -5.8 -6.1 -6.2 -6.2 -6.2 -6.1 0.0 |
| 6 | -3.9 -5.2 -5.9 -6.1 -6.3 -5.6 -0.1 | -1.9 -4.3 -5.6 -6.1 -6.4 -5.9 -0.2 |
| 8 | -1.5 -1.6 -1.6 -1.6 -1.6 -1.3 -0.6 | -0.6 -1.4 -1.9 -2.0 -2.1 -1.9 -1.1 |
| 10 | -2.8 -2.6 -0.8 0.0 +0.7 -0.3 -2.2 | -0.8 -2.5 -2.2 -1.4 -0.8 -1.6 -3.5 |
| 12 | -2.1 -3.6 -2.9 -1.5 -0.1 -2.4 -6.2 | -0.3 -2.0 -3.5 -3.2 -2.3 -4.7 -9.4 |

Of these, the [D99]-form itself (13) versus [D97] (17) at the same beta_I: -1.76 dB in the weak limit
(beta_I/sin theta <= 0.01), -1.3 dB at 0.3, 0 at saturation (bracket: 1 - e^-4b + 4b e^-2b against
1 - e^-4b + 8b e^-2b). The rest at 3-6 m/s is beta_I ([D97] below `apl_1994`), and at 80 deg and
10-12 m/s the alpha_b on the facet (-2 to -9 dB with [D97]'s larger beta_I).

D-193 (Gallaudet 68 kHz, 3 m/s, band means of intensity, uniform in grazing): ours -38.3 / -38.1 dB against
printed -51.3 / -41.5 at 27-38 / 44-66 deg; [D99] form with [D97] beta_I -46.2 / -45.2 dB (the earlier file
found -44.1 / -43.4 with the [D97] form). **The bubble law plus [D99]'s bistatic form removes 7.9 of the
13.0 dB at 27-38 deg and over-corrects at 44-66 deg (3.7 dB below Gallaudet)**; the remaining low-band
excess (about 5 dB) is still unexplained (averaging of dB values, [DP97]: 3.4-4.5 dB, is the candidate
the earlier file named).

Facet attenuation with OUR beta_I (`apl_1994`), dB added to `bubble_layer_surface_db` by
facet x exp(-2 beta_I / sin theta), 45 / 60 / 80 / 89 deg:

| f | 8 m/s | 10 | 12 | 15 |
|---|---|---|---|---|
| 30 kHz | 0 / -0.16 / -0.60 / -0.59 | 0 / -0.43 / -0.85 / -0.84 | 0 / -0.75 / -1.13 / -1.11 | 0 / -1.27 / -1.59 / -1.57 |
| 68 kHz | 0 / -0.25 / -1.20 / -1.19 | 0 / -0.71 / -1.70 / -1.68 | 0 / -1.31 / -2.25 / -2.23 | 0 / -2.34 / -3.17 / -3.14 |

Bistatic bubble term: our `bubble_backscatter` at the arithmetic mean (and at the geometric mean, TR
0702's rule for monostatic models, Sec. 6.4.2) against (13), 30 kHz, [D97] beta_I (dB re (13)):

| U10 | theta_i / theta_s | (13), dB | mean angle | geometric mean | alpha_b, dB |
|---|---|---|---|---|---|
| 4 | 5/40, 10/30, 5/80, 20/20 | -41.2 | +1.8 | +1.8 | -0.05..-0.08 |
| 8 | 5/40 | -27.6 | +3.8 | +2.9 | -4.1 |
| 8 | 10/30 | -26.3 | +2.3 | +2.0 | -2.4 |
| 8 | 5/80 | -27.5 | +4.3 | +3.4 | -3.9 |
| 12 | 5/40 | -29.2 | +4.7 | +1.7 | -41.0 |
| 12 | 5/80 | -28.6 | +8.2 | +3.2 | -39.3 |
| 12 | 20/20 | -25.8 | +0.5 | +0.5 | -18.4 |

The geometric mean is closer to (13) than the arithmetic mean whenever the angles differ (by 0.3-5 dB);
neither reproduces (13)'s weak-limit level (+1.8 dB, the backscatter enhancement (13) does not have).
Weak limit of (13): sigma_b = beta_I delta_r / (pi delta), independent of both angles (ours).

Does [D99]'s bubble law repair D-193? It brings one more point: beta_I(30 kHz, 4 m/s) <= 0.0012, which agrees
with [D97] Table II within 0.7 dB and is 5.2 dB below `apl_1994`. That supports the earlier file's WP2
(an opt-in [D97] extinction); it does not by itself close D-193.

### 3.3 (b) Near-specular forward scatter: the SSA lobe against our facet lobe and time spreading (computed)

Stand-in spectrum (**ours**; [D99]'s is not printed): F(K) = B K^-4 for K1 <= K <= 31 rad/m and a steeper
power beyond, up to 500 rad/m, with B, K1 and the exponent fixed by [D99]'s three printed integrals for runs
2-3: H = 0.16 m, s_L^2 = 0.017, total mss 0.024 -> K1 = 0.264 rad/m, exponent 4.27 beyond 31 rad/m. Its
peak is at 0.26 rad/m, not [D99]'s 0.10 (0.16 Hz): it matches the height and the slopes, not the shape.
Check: its visibility function reaches 0.1 at rho = 0.48 m (Fig. 5b: radius 0.55 m).

In-plane cut, run 2 nominal (theta_i = 14 deg, phi_b = 0, 30 kHz), dB:

| theta_s - theta_i | -10 | -6 | -4 | -2 | 0 | +2 | +4 | +6 | +10 | +20 | +40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SSA(2), stand-in | 2.0 | 6.6 | 7.7 | 8.1 | 8.2 | 7.9 | 7.4 | 6.6 | 4.4 | -3.7 | -21.3 |
| `facet_lobe`, Cox-Munk 0.0235 | 4.0 | 4.8 | 5.1 | 5.2 | 5.3 | 5.2 | 5.1 | 4.8 | 4.0 | -0.2 | -18.1 |
| `facet_lobe`, mss 0.0086 | 5.9 | 8.3 | 9.1 | 9.5 | 9.7 | 9.5 | 9.1 | 8.3 | 5.9 | -5.8 | -56.2 |

Out-of-plane cut (theta_s = theta_i = 14 deg), phi_b = 0 / 1 / 2 / 4 / 8 / 16 / 31 / 60 deg: SSA 8.2 / 7.7 /
6.1 / 0.7 / -12.8 / -27.5 / -40.2 / -51.4; facet Cox-Munk 5.3 / 5.1 / 4.4 / 1.8 / -8.5 / -50 / -200 / -724;
facet 0.0086: 9.7 / 9.1 / 7.2 / -0.1 / -29 / -145 / ... The SSA lobe has a sharper peak and broad Bragg
wings that geometric optics cannot have; out of plane it is narrower than the Cox-Munk facets within 4 deg.
Peak level: SSA 8.2 dB, Cox-Munk facets 5.3 dB, [D99] Fig. 6a about +15 dB (read from figure; with his
unprinted spectrum). The peak depends on the spectrum's shape, not only on its slope variance.

Energy, int sigma dOmega / sin theta_i (ours; 1 = everything a flat mirror reflects):

| theta_i | Kirchhoff (1), stand-in | SSA(2), stand-in | of which within 10 deg / 30 deg azimuth of specular | `facet_lobe` Cox-Munk | coherent part exp(-chi^2) |
|---|---|---|---|---|---|
| 14 deg | 1.004 | 0.833 | 0.652 | 0.961 | 9e-41 |
| 30 deg | 0.996 | 0.912 | 0.651 | 0.996 | 1e-171 |

So the Kirchhoff integral and our facet lobe conserve energy (the facet lobe loses 4 % below the horizon at
14 deg); lowest-order SSA does not (0.8-0.4 dB short). Energy is conserved with the coherent part in the
Kirchhoff/facet sense that `time_spread` uses (tail = |R|^2 (1 - e^-Gamma^2)); a published SSA strength would
carry 9-17 % less.

Time spread and angle spread against [D99]'s measurements (our facet model: mean excess delay T = mss (1 +
sin^2 g) t1 t2/(t1 + t2), late e-fold 2a = 2 mss t1 t2/(t1 + t2); "fitted" = log-linear fit of
`timespread.facet_delay_density` over 0.21-0.63 ms; horizontal rms angle at the receiver sqrt(2 mss) sin g
r1/(r1 + r2), PHYSICS_NOTES Sec. 16):

| Run | g (deg) | t1t2/(t1+t2) (ms) | measured L (ms) | ours, Cox-Munk: T / 2a / fitted (ms) | ours, mss 0.0086: fitted | measured sigma_theta_h (deg) | ours, Cox-Munk (deg) | mss implied by sigma_theta_h |
|---|---|---|---|---|---|---|---|---|
| 1 (<= 1 m/s; CM at 1 m/s) | 12.7 | 14.74 | 0.03 | 0.126 / 0.24 / 0.18 | 0.19 | 0.45 | 1.15 | 0.0013 |
| 2 | 13.4 | 12.04 | 0.21 | 0.298 / 0.57 / 0.32 | 0.16 | 0.76 | 1.29 | 0.0081 |
| 3 | 20.3 | 11.58 | 0.31 | 0.305 / 0.54 / 0.30 | 0.16 | 1.87 | 2.78 | 0.0106 |
| 4 | 13.3 | 12.46 | 0.21 | 0.295 / 0.56 / 0.32 | 0.17 | 0.82 | 1.21 | 0.0103 |

- The quadratic-visibility reading of Fig. 5b (ours: 1 - C = s^2 rho^2/(2 H^2) at rho = 0.55 m, chi = 9.5)
  gives a per-axis slope variance 0.0043, total 0.0086, the same as the 0.0081-0.0106 implied by the measured
  angle spreads of runs 2-4. The lobe is steered by the slopes of waves longer than about a metre.
- With that mss our facet e-fold is 0.16-0.21 ms (runs 2, 4: measured 0.21) but too short for run 3 (0.31).
  With Cox-Munk it is 0.30-0.32 ms: right for run 3, 1.5 times too long for runs 2 and 4. Run 1 (calm,
  0.03 ms) is 6 times shorter than either. No single slope value fits the four runs: [D99]'s explanation is
  that the late decay belongs to the Bragg-scale waves (run 2 needs K up to 500), which a facet model
  does not contain.
- Our stand-in SSA impulse response for run 2 by [D99]'s eq. (17) (0.25 m cells, 92 x 30 m patch, no pulse,
  no absorption): e-fold 0.13 ms (0.1-0.4 ms), 0.16 (0.2-0.6), 0.20 (0.4-1.0), 0.40 (1.0-2.5); level re its
  peak -19.5 / -29.7 / -37.4 / -42.9 dB at 0.5 / 1.0 / 1.5 / 2.0 ms. The same sum with the Cox-Munk facet
  lobe: e-fold 0.21-0.26 ms, -12.1 / -20.3 / -31.8 / -44.3 dB. Totals differ by -0.64 dB; the SSA puts 66 %
  of the energy in the first 0.1 ms, the facets 45 %.

### 3.4 (c) [McD86] against the high-frequency Kirchhoff forms (13.62/13.63 versus L.34/L.35)

- Q_L of (18), multiplied by 2 as [McD93]'s erratum asks, with <(n.k')^2> read as |q|^4/(4 q_z^2) (the
  unnormalised normal N = (-zeta_x, -zeta_y, 1) at the specular slope, which is what replacing dS by dr_perp
  implies; ours), **equals J&R (13.62)/(13.63) and our `facet_lobe` to 5e-13 dB** over 1990 random bistatic
  geometries: sigma = (1/4)(|q|^4/q_z^4) p_s(s), p_s = exp(-...)/(2 pi ||B_s||^(1/2)).
- Without the erratum (the 1986 print, with its "factor of one-half" claim) Q_L is 3.01 dB low everywhere. The
  erratum is right: with it Q_L conserves energy (our `facet_lobe` integrates to sin theta_i: 0.998 / 1.000 /
  1.000 at 30 / 60 / 89 deg, s2 = 0.02); without it half the reflected energy would vanish.
- J&R (L.34)/(L.35) as the earlier notes read them ((1/4)(Delta k^2/Delta k_z^2) p_s with 1/(2 pi ||B_s||))
  integrate to **97.9 / 98.1 / 98.1 times** sin theta_i at the same angles (s2 = 0.02): ||B_s|| (a determinant,
  slope^4) cannot normalise a 2-D slope density. **Settled (ours, from [McD86] with the erratum, Dahl's (1) in
  the stationary-phase limit, and energy conservation): (13.62)/(13.63) are right; (L.34)/(L.35) are misprints.**
- [D99]'s (1), expanded to quadratic order in the lag, gives the same GO limit (ours, algebra):
  k^2 F^2 = |q|^4/(4 q_z^2), and the Gaussian integral gives |q|^4 / (8 pi q_z^4 s^2) exp(-q_perp^2/(2 q_z^2 s^2)).
- Our `bottom.kirchhoff_strength` is the full Kirchhoff integral, identical to [D99] (4) (3.1); for the
  bottom's power-law spectra (gamma2 < 4) the slope variance diverges and there is no GO limit at all, which
  is where a [McD86]-type partition would be needed. We use none, and need none while the full integral is
  evaluated (as J&R Fig. 13.19 recommends, earlier notes).
- Reading of <(n.k')^2> in the diffractive term (ours): with <(n.k')^2> = k^2 sin^2 theta_s (mean normal), the
  erratum x2, and the integral of (14)/(18) taken over the **whole** K plane, the small-mu limit of the integral
  term is exactly first-order perturbation theory (computed: -51.20 dB against 4 k^4 sin^4 theta psi(2k cos
  theta) -51.25 dB, 10 kHz, 20 deg, 10 kn) and Q_L is exactly GO at normal incidence; off normal Q_L is then
  below (13.62) by sin^4 (-0.27 / -1.08 / -2.50 / -6.02 dB at 80 / 70 / 60 / 45 deg). With the printed
  quadrant limits (0 to infinity in both K_x and K_y) the in-plane integral covers a quarter of the plane
  (Sec. 6).
- Reproduction (ours; [McD93]'s spectrum standing in for Pierson-Stacy, 10 kHz, 5.14 m/s, readings above):
  conventional (14) at normal incidence 15.6 / 8.1 / 1.6 / 1.1 dB for K_L 0.025 / 0.05 / 0.3 / 0.5 cm^-1
  (spread 14.5 dB; paper 8 dB), at 63 deg -19.7 / -17.6 / -17.5 / -17.2 (paper 15 dB; our spectrum lacks the
  dip of Fig. 4); corrected (18) for K_L 0.05-1.0 cm^-1: within 1.6 dB at every angle 50-90 deg (paper: at
  most 2 dB), e.g. 90 deg 0.3 to 1.8 dB, 70 deg -9.5 to -8.7 dB; at K_L = 0.025 cm^-1 our (18) goes negative:
  with this spectrum (nu sigma_S / sigma)^2 is not small there, so the expansion fails. Forward, corrected
  (18): 0.3 / 1.5 dB at 80-90 deg for K_L 0.05 / 1.0 (conventional 8.0 / 0.6). From Fig. 3's large-scale curve
  (-50 dB at 63 deg, +3 dB at 90 deg) the paper's own Q_L is consistent with 1/(16 pi s^2) and s^2 about 0.0085
  per axis, i.e. its figures were computed without the later factor 2 (ours, read from figure).

### 3.5 (d) What `exclude_specular` removes against the published near-specular lobe (computed, run 2, 30 kHz)

`specular_part` with Cox-Munk mss at 4 m/s (0.0235) and h from our wind law (0.086 m) or measured (0.16 m):
the coherent factor 1 - exp(-(k h q_z)^2) is 1 here, so h does not matter. Chapman-Harris at 4 m/s, mean
angle, for scale.

| theta_s, phi_b (deg) | SSA(2) stand-in (dB) | removed by `exclude_specular` (dB) | SSA minus removed, linear, as dB | Chapman-Harris (dB) |
|---|---|---|---|---|
| 14, 0 (specular) | 8.2 | 5.3 | 5.1 left | -39.2 |
| 16, 0 | 7.9 | 5.2 | 4.6 left | -38.5 |
| 19, 0 | 7.0 | 5.0 | 2.6 left | -37.6 |
| 24, 0 | 4.4 | 4.0 | -6.2 left | -36.3 |
| 34, 0 | -3.7 | -0.2 | over-removed (clipped to 0) | -34.1 |
| 14, 5 | -2.8 | -0.1 | over-removed (clipped) | -39.2 |
| 14, 15 | -26.2 | -43.3 | -26.3 left | -39.2 |
| 14, 31 | -40.2 | -200 | all left | -39.2 |
| 90, 180 (nadir back) | -39.2 | -104 | all left | -26.9 |

Reading: if a reverberation item carried a published near-specular (SSA) strength, `exclude_specular` would
leave up to +5 dB of its peak in the reverberation and clip the shoulders a few degrees out; integrated, the
two lobes differ by 0.64 dB (3.3), so the energy bookkeeping (removed ~ tail) holds to about 15 %. With
Chapman-Harris (what shipped scenarios use) there is no lobe: the removal clips the strength to zero inside
the facet lobe (about +5 dB, i.e. some 44 dB above Chapman-Harris's -39 dB, within about 20 deg of specular), i.e. it removes
everything there, which is the intended behaviour (D-123) and is what the published lobe would attribute to
the specular part at 14 deg to within those few dB.

## 4. What this closes or changes in open issues

Unanswered first: **TR 0702's "McDaniel" near-specular model is not [McD86]** as far as can be told. Our own
reading of TR 0702 (PHYSICS_NOTES Sec. 13, Table 6 row, p. 47) records `McDanielSurface` as deferring to
"McDaniel ARL-PSU TM 90-88; Lang & Culver TM 92-342; Donelan et al. (1985)". [McD86] has no wind laws for the
rms height, slope or vertical speed that `McDanielSurface` computes, no bistatic surface model beyond
in-plane backscatter and specular forward scatter, no bubbles. The TM's title, from the reference list of
Dahl 1996 (JASA 100; only that entry was read): "Models for predicting bistatic surface scattering
strength", Tech. Mem. 90-88, ARL Penn State, 1990 - an unpublished report, not in hand. [McD86] may be its
roughness kernel; nothing here shows it. Status: **wanted (TM 90-88)**; [McD86] answers the earlier file's
guess in the negative (finding 7 of `surface_papers_mcdaniel_dahl.md`, its Sec. 4 sonar-9le item and Sec. 6
last line).

### sonar-9le (high-frequency surface)
- Bistatic surface cross section away from backscatter: **now in hand** for 30 kHz, low wind, forward
  geometries ([D99] (1)-(13)); the roughness part needs a wave spectrum we do not have printed (Plant 1986 is
  cited, not reprinted). The bubble part (12)-(13) is closed-form and usable.
- Interpolation between facet and small-scale terms: [D99] replaces it with SSA(2) (one formula); [McD86]
  gives a corrected composite with a weak partition dependence (2 dB at normal incidence). The earlier file's
  "not printed" is now answered twice.
- Facet term near normal incidence ([McD93] Fig. 20 Nuetzel 3-4 dB above ours at 82-88 deg, earlier WP6): with
  the erratum, [McD86]'s large-scale term at normal incidence is exactly our GO, and its correction there is
  the diffractive integral, which with our reading reduces the lobe (factor 2 - nu^2 ... minus the small-scale
  loss) rather than raising it. Not settled; it needs a Pierson-Stacy spectrum and the erratum-corrected
  evaluation.
- Above 70 kHz: nothing new (60 kHz is [McD86]'s stated ceiling, 30 kHz [D99]'s only frequency).

### sonar-64sm (D-193 follow-up)
- One more independent beta_I point (30 kHz, ~4 m/s, upper bound 0.0012) supports [D97] Table II against
  `apl_1994` (5 dB lower). [D99]'s bistatic bubble form at backscatter is 1.76 dB below [D97]'s; with it the
  27-38 deg Gallaudet band is within 5.1 dB and the 44-66 deg band 3.7 dB below (3.2).

### Time spreading and angle spreading (D-62, D-76; PHYSICS_NOTES Sec. 21; REFERENCES row "Coherent fraction,
delay spread, angle spread and scintillation of surface forward scatter")
- **Delay spread**: measured e-folding times (0.03-0.31 ms) and coherence bandwidths at 30 kHz with full
  geometry: confirmed, usable as test data. **Angle spread**: measured sigma_theta_h (0.45-1.87 deg), with the
  geometric scaling sin theta_g/(1 + RD/SD) our facet model already has: confirmed. **Coherent fraction**:
  negligible at 30 kHz (chi 4-13), so no measured coherent loss here. **Scintillation**: not in [D99].
  The row moves from "wanted" to "partly: [D99] delay and angle spread at 30 kHz, 1-4 m/s".
- Our slopes: Cox-Munk total is right as a total (0.0235 vs 0.024) and wrong for the lobe (3.3): the angle
  spread is 1.5-2.6 times too wide. Ping-to-ping and Doppler: not in [D99].

### TR 0702 matrix rows (docs/figures/sst_validation.md)
- Sec. 6.4.2 row (geometric mean vs our arithmetic mean for Chapman-Harris): a published bistatic bubble law
  now exists to compare with; the geometric mean is the closer of the two rules (3.2), neither is right.
- Sec. 12.4 / 12.2-12.5 rows ("TR 0702's is McDaniel's"): unchanged in substance - the model TR 0702 cites
  (TM 90-88) is still not in hand; [McD86] is not it. Suggest the row say so.
- D-123 / `exclude_specular`: the facet partition is within a few dB of a published near-specular lobe at
  low wind and 30 kHz (3.5); this is the first published check of it.

### docs/REFERENCES.md "wanted" rows (Sec. 10)

| Row | Result |
|---|---|
| "The handbook's law for beta_I and its small-scale roughness function G(U)" | beta_I: one more point ([D99] p. 2161), agreeing with [D97]. G(U): **not in these papers**. Interpolation facet/small-scale: [D99] SSA(2); [McD86] corrected composite - both now confirmed formulas |
| "Coherent fraction, delay spread, angle spread and scintillation of surface forward scatter" | partly: [D99] delay e-fold L and angle spread sigma_theta_h (Table I, p. 2165), 30 kHz, 1-4 m/s, 13-20 deg; coherent fraction negligible at 30 kHz; scintillation absent. Dahl 1996 / 2001 / 2004 remain for the rest |
| "Bistatic bottom model" (Williams-Jackson) | not these papers; but [D99] (1)-(6) are the same Kirchhoff and SSA expressions our bottom uses, so they cross-check `kirchhoff_strength` (identical) |
| New: TR 0702's `McDanielSurface` source | McDaniel, ARL Penn State TM 90-88 (1990), "Models for predicting bistatic surface scattering strength": wanted, unpublished |
| New: Plant (1986) spectrum (JGR 91, 10735) | needed to reproduce [D99]'s sigma_r away from specular (and [D97]'s composite) |

## 5. Work packages

Ordered by value per effort. **Shipped exposure** (grep of `configs/scenarios/*.yaml`): `exclude_specular` is
used by **no** scenario; the only `geometry: bistatic` surface item is `sst_pursuit` (`{model: chapman_harris,
geometry: bistatic, ...}`, transmitter and receiver on one vehicle, so the two grazing angles differ only on
paths with a bottom leg); `time_spread` is on in `sst_asiaex` (surface from the wind; bottom mss 0.045),
`flank_sst_shallow` and `flank_sst_spread` (`bubble_loss_db: 0.5`); `bubble_layer_surface` and
`kraus_surface` are used by none (earlier file). So WP1-WP3 move no shipped result; WP4 would move the
three time-spread scenarios if made default.

**WP1 (S): tests that hold what these papers confirm (characterisations, no behaviour change).** New
`tests/python/test_surface_bistatic_papers.py`:
- `bubble_damping_constant(30e3)` = 0.0792 within 5e-5 ([D99] p. 2159).
- `kirchhoff_strength / |V|^2` equals [D99] eq. (4) written with D(R) (six geometries, 0.01 dB; script
  `kcheck.py` above).
- `facet_lobe` equals [McD86] Q_L x 2 with <(n.k')^2> = |q|^4/(4 q_z^2) (1e-9 relative), and its energy
  integral is sin theta_i within 1 % at 30-89 deg (exists in spirit in `test_reverb_specular.py`; add the
  [McD86] identity and a note that the 1986 print is 3.01 dB low without the erratum).
- [D99] eq. (13) helper (test-local) in the limits: theta_i = theta_s gives (1 - e^-4b + 4b e^-2b) sin theta /
  (8 pi) delta_r/delta; weak limit beta_I delta_r/(pi delta); and its ratio to `bubble_backscatter` -1.76 dB
  at beta_I/sin theta = 1e-3.
- Docs (proposed text for the orchestrator): J&R notes Sec. 6 bullet on (L.34)/(L.35): "settled: misprints;
  (13.62)/(13.63) confirmed by McDaniel 1986 with the McDaniel 1993 erratum and by energy conservation";
  `surface_papers_mcdaniel_dahl.md` finding 7: "McDaniel 1986 read: not TR 0702's model (TR 0702 cites TM
  90-88)"; REFERENCES rows of Sec. 4 above.

**WP2 (M): a bistatic bubble term for the surface (opt-in).** `surface.bubble_layer_surface_strength` and a new
`bistatic: true` option (or a new model key `bubble_layer_bistatic`) taking (g_in, g_out, azimuth): sigma =
facet(g_in, g_out, azimuth) x alpha_b (12) + sigma_b (13), the facet being `reverb_specular.facet_lobe` with
Cox-Munk slopes (it equals `facet_backscatter` at azimuth pi; test). Files: `sim/surface.py`,
`sim/simulator.py` (accept the key), the C++ reverberation strength if one exists (check `libs/`; live-chain
parity applies only if the surface strength is in the live chain). Tests first: limits of WP1; the 3.2
table rows at 30 kHz (to 0.05 dB with the same beta_I); `takes_azimuth` true; symmetry in (g_in, g_out);
energy: the facet part integrates to |R|^2 alpha_b sin theta_i. Default: unchanged (monostatic items keep
`bubble_layer_surface_db`). Owner first if it is to become the bistatic default: before -> after for a
5/40 deg pair at 8 m/s, 30 kHz, same beta_I ([D97]): -23.8 -> -27.6 dB for the bubble part (the mean-angle rule is +3.8 dB); with `apl_1994` the before value differs by the beta_I table of the earlier file.

**WP3 (S, owner decision): bubble attenuation of the facet term in `bubble_layer_surface_db`.** [D97] Sec. VI
and [D99] (11)-(12) attenuate the roughness term by exp(-2 beta_I / sin theta) in backscatter. Ours does not.
Effect with `apl_1994`: 0 at 45 deg, -0.6 to -1.6 dB at 80-89 deg (30 kHz, 8-15 m/s), -1.2 to -3.2 dB at
68 kHz. Tests first: the 3.2 facet-attenuation table; no change below 45 deg. It moves a default result of a
model no shipped scenario uses: owner first, then make it the behaviour (a fix, documented in a D-entry).

**WP4 (M): near-specular slopes for time and angle spreading (opt-in), tested against [D99] Table I.**
Add `surface.lobe_slope` (or `mean_square_slope_lobe`) to `time_spread`: the slope variance used for the
decay time, the delay shape and `angle_spread`, separate from the total mss. Two options, owner to choose:
(a) a number in config (e.g. 0.0086 at 4 m/s, [D99]-derived, ours); (b) computed from a wave spectrum as the
slope variance of waves longer than the first-Fresnel/visibility scale (int K^3 F dK up to K such that
chi^2 (1 - C) reaches ln 10 at its inverse, iterated; ours). Files: `sim/timespread.py`, docs Sec. 16. Tests
first, with [D99] Table I geometries (source/receiver depths, range, c = 1520): horizontal rms arrival angle
within 25 % of 0.76 / 1.87 / 0.82 deg for runs 2-4 with the option, and the documented failure with Cox-Munk
(1.29 / 2.78 / 1.21 deg) as an expected-difference test; late e-fold of the mean intensity response 0.21 ms
+-30 % for runs 2 and 4; run 1 (0.03 ms) and run 3 (0.31 ms) recorded as not reproduced by a facet model
(Bragg-scale tail, [D99] p. 2165). Default stays Cox-Munk (moving it would change `sst_asiaex`,
`flank_sst_shallow`, `flank_sst_spread`: prepare before -> after of their fitted decay times first). Must
not: tune the default slope to these four runs.

**WP5 (M-L): SSA(2) surface strength with a spectrum (opt-in, diagnostic).** [D99] (1), (4)-(6), (7), (9)-(10),
(11)-(13), with a wave spectrum F(K): Pierson-Moskowitz plus a K^-4 equilibrium range is ours; Plant (1986) is
[D99]'s and must be opened first. Reuse `bottom.small_slope_strength`'s numerics only if they can be made to
take a tabulated spectrum and a pressure-release boundary (they divide by rho). Tests first: g identity;
small-chi limit = (7); specular limit = (1) exactly; KA energy integral 1.00 +- 0.01 (stand-in spectrum, 14 and
30 deg); SSA energy 0.83-0.91 recorded; [D99] Fig. 6/9 peak levels only as figure readings (+-2 dB) and only
with the paper's spectrum. Not a default: the spectrum is the unknown.

**WP6 (S): sst_validation matrix wording.** Rows Sec. 6.4.2 and 12.2-12.5: "McDaniel 1986 read: not TR 0702's
McDanielSurface (TR 0702 cites ARL-PSU TM 90-88, not available); the published bistatic bubble law (Dahl
1999 eq. 13) is closer to the geometric-mean rule than to our mean-angle rule (3.2)". Docs only.

Must not be done: change `extinction`, the mean-angle rule, the facet attenuation or the time-spread slopes by
default without the owner; tune any slope or beta_I to one figure or one run; quote [D99] Fig. 6/9 levels or
[McD86] figure levels as test bounds tighter than +-2 dB.

## 6. Not verified / unclear on the page

- [McD86] eq. (18): the first exponential is printed exp(+(alpha + K_x)^2/(2 mu_x^2) - ...) without the leading
  minus that eq. (14) has; read as a misprint (the integral would diverge).
- [McD86] eqs. (14), (18): integration limits 0 to infinity in both K_x and K_y with two exponentials cover
  pairs (K, -K) from one quadrant only; for in-plane geometry the full-plane expression the derivation gives
  is four times the printed integral (ours, derivation from (8)-(12)). [McD93]'s erratum ("multiply the
  right-hand side of eq. (18) by 2") restores Q_L but not, by itself, the quadrant factor; whether the
  "symmetric properties of W" were meant to absorb it is not stated.
- [McD86] <(n.k')^2> is never evaluated in the text; the two readings (mean normal, or the unnormalised facet
  normal) give different off-normal Q_L (Sec. 3.4).
- [McD86] eq. (18) "2 - nu^2 - ...": nu = (k_z + k'_z) sigma with the TOTAL rms height; the large nu^2 cancels
  against the low-K part of the integral (ours); evaluated as written it is numerically unusable.
- [McD86] Appendix: V^2 and U^2 constants (20 log10 e / 3) read from a small scan; the factor-1/2 statement is
  clear.
- [McD86] Fig. 2-7 values are scan readings (+-2 dB); the Pierson-Stacy spectrum is not printed.
- [D99] eq. (3) text says (k_i - k_s)/k, the printed components are half of that; the exponent 2ik(...) and
  chi confirm the printed components.
- [D99] App. B: E(Delta_i Delta_s) printed s_L^2 cos(phi_b); eq. (10) and a Monte Carlo of (B1) (ours: -1.205e-5
  against eq. (10) -1.191e-5 and -3.567e-5 with the printed expectation, at 0.3/0.5 rad forward) need
  s_L^2 cos(phi_b)/2: the appendix text is misprinted, eq. (10) is right.
- [D99] eq. (14) H^2 nabla^2 C(origin) = int K^3 F dK lacks a minus sign (nabla^2 C is negative at the origin).
- [D99] (A3) last branch printed 10^(-0.4 + 0.8393 exp[log_e[(omega/omega_p)^2]]): exp(ln x^2) = x^2, which
  jumps from 1.23 to 56 at omega/omega_p = 1.6. Banner's form has a factor -0.567 inside the exponential
  (10^(-0.4 + 0.8393 exp[-0.567 ln((omega/omega_p)^2)]) gives 1.24 there, continuous): the coefficient -0.567 is
  from memory of Banner (1990), **unconfirmed**; the print is wrong either way.
- [D99] the blended spectrum F(K) (Fig. A2), the spectrum peak values and the Fig. 6/9 levels are plotted
  only; every sigma_r number above uses our stand-in spectrum.
- [D99] wind at 6.9 m: our 3 % conversion to U10 is ours; the paper does not convert.
- Whether TR 0702's `McDanielSurface` uses [McD86]'s correction inside TM 90-88: not determinable from either
  paper.
