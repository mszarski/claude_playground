# High-frequency bottom backscatter, roughness against volume: Jackson and Briggs (1992), extracted for the simulator

Status words as in `docs/REFERENCES.md`: **confirmed** = read on the page image of this paper; **via
secondary** = another author's result as this paper reprints it; **ours** = our derivation or a number we
computed (script named); **figure** = read off a printed curve (the README's "read from figure"; about
+-1 dB on these 10-dB grids, never a test bound tighter than +-1 dB); **one site** = a value measured or
fitted at one site. The paper is all rights reserved: formulas re-typed in our notation, constants with
page and equation, table or figure number, own words; the PDF is not in the repository. The companion note
is `bottom_bistatic_williams_jackson_1998.md` ([WJ]).

Scripts (scratchpad, not committed; `PYTHONPATH=<worktree>/python`, main checkout's built module linked in
temporarily): `/tmp/claude-0/-home-user-claude-playground/32e65dcd-82ae-5f75-a363-7372d2e7f847/scratchpad/ref-b/jb.py`
(eq. (4), eq. (2), (A2), (A11), (A8), the site models, the class rules), `.../ref-b/refl.py` (reflection
loss of site against class), `.../ref-b/kh.py` (composite cut-off). Outputs beside them (`*.out`).

## 0. What disagrees, first

1. **The paper prints fitted volume parameters for the two mud sites that our documents cite only as agents'
   derivations, and they differ.** Arafura Sea: printed fit 0.005 (0.004-0.006 "give reasonable fits"),
   p. 974; our notes derived 0.0031 from J&R Table 14.1 by (14.10). San Francisco: printed fit 0.002,
   p. 975; derived 0.0037. The soft rows' `sigma2_range: [0.0008, 0.0037]` (`configs/sediments.yaml:172`)
   therefore lies below a printed fitted site value (0.005), and its `basis` names the two sites with the
   derived numbers only. Against the class rule (0.001 from 5.5 phi): Arafura +7.0 dB, San Francisco
   +3.0 dB on the volume term (ours, `jb.py`); one site each. The paper itself says 0.005 is at the border of
   single scattering and that Arafura's mean grain size is misleading (bimodal: sand and gravel in a silty
   clay).
2. **Three prints, three cut-offs for the composite model's small-scale part.** The text (p. 976) defines
   K_c so that the small-scale mean-square height equals 1/k^2 (k h_s = 1); its own eq. (A8) gives the
   slope of k h_s = 2^(-1/2) (ours: (A8) equals `large_scale_slope(small_scale_kh=0.7071)` to 5 digits in
   three cases); J&R (13.72)-(13.75) print k h_s = 1/2, our default. On our composite model at 5 deg: -1.2
   to -1.4 dB (0.707) and -2.0 to -2.8 dB (1.0) for medium sand and sandy gravel at 30 and 100 kHz; under
   0.2 dB at 20 deg (ours, `kh.py`). `composite` is opt-in, 0 shipped.
3. **Measured geoacoustics at these sites differ strongly from our class relations at the same grain size**
   (one site each): density ratio 1.94 / 1.39 / 1.41 (Quinault / Arafura / San Francisco, Table II) against
   1.345 / 1.149 / 1.148 from `apl_grain_size_relations` at 2.97 / 5.63 / 6.39 phi; normal-incidence
   reflection loss 8.7 / 16.1 / 15.3 dB against 14.6 / 23.9 / 24.1 dB (ours, `refl.py`). J&R call those
   relations ad hoc model fits (p. 314); this is evidence for owner decision 10 of the plan, not a change.
4. **The published monostatic model is not ours at small and steep angles.** It uses composite roughness
   (slope-averaged, shadowed) below about 80 deg and Kirchhoff above, switching where Kirchhoff has fallen
   15 dB below its peak, and it slope-averages and shadows the volume term too (pp. 976-977). Ours: small
   slope, volume not averaged. With the paper's inputs our default reproduces its curves within about 1 dB
   at 20-70 deg at all three sites, but at Arafura ours is 1.2-1.9 dB higher at 10-20 deg and at Quinault
   1.4-2 dB lower at 10 deg (Sec. 3).
5. Print problems (Sec. 6): Table V's San Francisco pre-storm along-strike w2 (0.000846) is 4.6 % (0.19 dB)
   below what eq. (4) gives from Table III (0.000885), while the other six rows agree within 0.11 %;
   Quinault's loss parameter in Table V follows the whole-core attenuation of Table I, although the text
   says Table V uses the surficial values; Fig. 14 is labelled "95 %" and captioned "90 %"; Fig. 10 caption
   "Quinault size".
6. **Confirmed, and settles an earlier doubt:** eq. (4), the 1-D to 2-D spectral strength, prints
   Gamma((gamma - 1)/2) and the intercept factor (2 pi)^(gamma - 2), i.e. the corrected forms of J&R (D.19)
   and (D.23) that [N2] Sec. 2.1 derived by quadrature: (4) reproduces Table V from Table III to 0.1 %
   (six rows) while (D.19) as printed would give 14-37 % less (ours, `jb.py`). The perturbation term (A2)
   and the volume term (A11) equal ours to 6e-14 dB.

## 1. Citation, scope, data, relation to the restricted report (confirmed)

D. R. Jackson and K. B. Briggs, "High-frequency bottom backscattering: Roughness versus sediment volume
scattering", J. Acoust. Soc. Am. 92 (2), Pt. 1, 962-977 (August 1992). Jackson: Applied Physics
Laboratory, University of Washington; Briggs: Naval Research Laboratory, Stennis Space Center. Received
10 January 1992, accepted 2 April 1992. 16 pages, typeset (a scan of good quality). Read: every page image
(pp. 962-977, incl. the appendix and references); Tables I-V, eqs. (1)-(4) and (A1)-(A13) re-rendered at
220 dpi; Figs. 10, 12, 14, 15 at 220 dpi for readings.

What it is: geoacoustic, roughness and backscatter measurements at three shallow-water sites compared with
the Mourad and Jackson (1989) version of the APL-UW backscatter model, to decide whether roughness or
sediment volume scattering is responsible, and whether directional ripples make backscatter anisotropic.

Stated validity of the model (Sec. IV, App.): sediment a fluid with complex wavenumber (lossy); isotropic
Gaussian power-law relief, 2 < gamma < 4 (A4); composite roughness for roughly 0-80 deg, Kirchhoff for
roughly 80-90 deg; volume term a first-order (Stockhausen-type) result, single scattering (sigma2 above
about 0.004 may violate it, after Jackson et al. 1986a); the five geoacoustic and roughness parameters
frequency independent "in most applications"; sigma2 frequency independent over 10-100 kHz. The authors
add that the perturbation and Kirchhoff regimes for fluid-fluid interfaces are not established and their
conclusions would be overturned only by "massive failure" of the approximations, not by 3-6 dB (p. 975).

Data:

| Site | Where, depth | Bottom (paper) | Acoustic data | Error (paper) |
|---|---|---|---|---|
| Quinault | 17 km W of Washington, 47 deg 34' N 124 deg 35' W, 49 m | fine sand, directional ripples (mean wavelength nearly 12 cm, NNW-SSE strike) | towed array, 15-45 kHz band (Fig. 10 at 25 and 35 kHz), 5-90 deg via several depressions, 20-ping averages | +-3 dB absolute, +-1 dB between directions |
| Arafura Sea | 255 km NNW of Cape Arnhem, 10 deg 01' S 137 deg 50' E, 47 m | silt-clay with buried shell; clayey sands and sand-silt-clays, bimodal (55 % sand and gravel by weight) | towed, 15-45 kHz (Fig. 12 at 20 kHz; Fig. 13 35 kHz; Fig. 14 15-45 kHz at 20 deg) | +-2 dB; along a 144-km track at 20 deg, standard deviation 1 dB |
| San Francisco (STRESS mid-shelf) | 180 km NW of San Francisco, 38 deg 39' N 123 deg 29' W, 90 m | silt, moderate roughness, bioturbated (heart urchins, brittle stars); winter storms made ripples (21-201 deg strike, heights 1.9 cm) | tripod 5.2 m up, 40 kHz FM 2 ms 2-kHz sweep, 15 deg depression, 27 deg vertical and 6.5 deg horizontal beam, 4-30 deg, 49 days, averages of 49 scans | +-2 dB |

Geoacoustics from box cores (6.1-cm subcores; speed and attenuation at 400 kHz in 1-cm steps; Hamilton's
k in dB/m/kHz); roughness from stereophotographs (1-D periodograms, six lines per pair).

Relation to the restricted report: the paper predates APL-UW TR 9407 (1994) and cites no APL-UW technical
report. It cites Mourad and Jackson (1989) for the model and corrects a typographical error in it (in
(A11)). Everything below is what this paper prints: **confirmed**; what it reprints of Mourad and Jackson,
Jackson et al. (1986a), Kuo (1964), Stockhausen (1963) is **via secondary** for those originals.

## 2. Formulas and tables, in our notation (confirmed unless marked)

Angles are grazing (theta); log base 10; cross sections per unit area per unit solid angle; backscattering
strength 10 log sigma (Urick).

| Our symbol | Paper | Meaning |
|---|---|---|
| rho, nu, delta | rho, nu, delta | density ratio, speed ratio, imaginary/real part of the sediment wavenumber (Table IV) |
| sigma2 | sigma_2 | volume parameter (3) |
| gamma2, w2 | gamma, w2 | 2-D relief spectral exponent and strength, w2 in cm^4 at K = 2 pi / lambda = 1 cm^-1 (Table IV) |
| - | phi_1 | 1-D intercept (Table III), cm^3 at 1 cycle/cm |

Critical angle (1): theta_c = arccos(1 / nu) (Quinault 26.0 deg; ours: same).

Loss parameter (2): **delta = alpha nu c ln(10) / (40 pi f)**, with alpha/f in dB/m/kHz and c in m/ms
(the units must be consistent, p. 972). Typical delta 0.005-0.02, nearly independent of grain size.
Ours: `parameters_from_physical` (`bottom.py:193`) is the same relation (ours: algebra).

Volume parameter (3): **sigma2 = sigma_v / alpha**, sigma_v the volume backscattering cross section
(length^2 per length^3 per steradian), alpha the absorption coefficient in dB per length; dimensionless
and independent of units; about independent of frequency at 10-100 kHz; fitted values 0.0001-0.005 (Jackson
et al. 1986a, Stanic et al. 1988, 1989, Mourad and Jackson 1989); lower values would fit McKinney and
Anderson (1964), higher ones were reported by Stewart and Chotiros (1992). The alpha is the amplitude
attenuation in dB (20 log): (A11) follows from (20) of [WJ] only with alpha = (20/ln 10) k delta / nu
(ours, algebra); our `volume_strength` (`bottom.py:606`) uses exactly that.

1-D to 2-D strength (4):

    w2 = (2 pi)^(gamma - 2) phi_1 h0 Gamma(gamma / 2) / [pi^(1/2) Gamma((gamma - 1) / 2)]        (4)

gamma (2-D) = 1 - (1-D slope of Table III); phi_1 the 1-D intercept in cm^3 (the regression's value at a
spatial frequency of 1 cycle/cm); h0 = 1 cm. **This is the units trap**: the 1-D intercept is per cycle/cm
(spatial frequency), the 2-D w2 per rad/cm (wavenumber), and the two differ by (2 pi)^(gamma - 2) and the
Gamma ratio; neither is in m. Checked: (4) gives Table V from Table III for six of seven rows to 0.11 %;
the 1-D/2-D relation behind it checked by quadrature to 1e-15 (ours, `jb.py`). It is J&R (D.19) and (D.23)
with the corrections [N2] Sec. 2.1 made; the printed (D.19) would give 0.0052 for Quinault along-strike
(printed 0.00598).

Model (Appendix):

    sigma_b(theta) = sigma_r(theta) + sigma_v(theta)                                                   (A1)
    sigma_pr = 4 k^4 sin^4 theta |Y(theta)|^2 W(2 k cos theta)                                        (A2)
    W(K)     = w2 / (h0 K)^gamma,   h0 = 1 cm,   2 < gamma < 4                                         (A3), (A4)
    Y(theta) = [(rho - 1)^2 cos^2 theta + rho^2 - kappa^2] / [rho sin theta + P(theta)]^2               (A5)
    kappa    = (1 + i delta) / nu,   P(theta) = (kappa^2 - cos^2 theta)^(1/2)                           (A6), (A7)
    s^2      = (2 pi w2 h0^(-gamma))^(1/alpha) (k^2 / alpha)^((1 - alpha)/alpha) / (2 (1 - alpha))        (A8)
    alpha    = gamma / 2 - 1,   0 < alpha < 1                                                           (A9), (A10)
    sigma_pv = 5 delta sigma2 |1 - R^2(theta)|^2 sin^2 theta / (nu ln 10 |P(theta)|^2 Im[P(theta)])      (A11)
    R(theta) = (y - 1)/(y + 1),   y = rho sin theta / P(theta)                                          (A12), (A13)

(A2) is Kuo's first-order perturbation result for a lossy fluid (via secondary for Kuo); the complex
kappa removes the cusp at the critical angle of the lossless form of Jackson et al. (1986a). s^2 is the
large-scale mean-square slope (both axes), the volume under K^2 W(K) for K < K_c; the cross section (A2) is
averaged over a one-dimensional Gaussian slope density of variance s^2 / 2 and multiplied by a shadowing
function (McDaniel and Gorman 1982, Jackson et al. 1986a; Gaussian-quadrature form in Mourad and Jackson
1989; none of these printed here). (A11) is Stockhausen's result generalised to a lossy fluid (the
corrected form of Mourad and Jackson's equation); it is slope-averaged and shadowed in the same way. Steep
angles: Kirchhoff, "bell shaped with a peak at 90 deg"; used from 90 deg down to where it has fallen 15 dB
below its peak, composite below; the interpolation is in Mourad and Jackson (1989), not printed here.

### 2.1 Tables (confirmed; all values as printed)

Table I, whole-core means (+-1 standard deviation) and Table II, surficial 0-2 cm means (p. 964):

| Site | Depth (m) | Porosity % I / II | Mz (phi) I / II | nu I / II | rho I / II | Attenuation dB/m/kHz I / II |
|---|---|---|---|---|---|---|
| Quinault | 49 | 41.2+-2.1 / 40.5+-1.2 | 2.94+-0.11 / 2.97+-0.06 | 1.113+-0.013 / 1.113+-0.009 | 1.93+-0.03 / 1.94+-0.02 | 0.37+-0.12 / 0.30+-0.04 |
| Arafura Sea | 47 | 69.7+-4.0 / 77.8+-5.2 | 5.24+-0.78 / 5.63+-1.11 | 0.986+-0.004 / 0.988+-0.005 | 1.49+-0.07 / 1.39+-0.09 | 0.84+-0.29 / 0.20+-0.09 |
| San Francisco | 90 | 63.4+-3.9 / 72.8+-2.0 | 6.35+-0.45 / 6.39+-0.29 | 1.009+-0.005 / 1.002+-0.003 | 1.58+-0.07 / 1.41+-0.03 | 0.56+-0.08 / 0.37+-0.06 |

Table III, 1-D roughness spectra, regression slope and intercept at 1 cycle/cm (cm^3) (p. 965):

| Spectrum | Slope | Intercept (cm^3) |
|---|---|---|
| Quinault along strike | -2.92 | 0.28e-3 |
| Quinault across strike | -2.67 | 0.33e-3 |
| Arafura Sea | -2.18 | 0.69e-3 |
| San Francisco along, pre-storm | -2.65 | 0.72e-4 |
| San Francisco along, post-storm | -2.73 | 0.57e-4 |
| San Francisco across, pre-storm | -2.38 | 1.30e-4 |
| San Francisco across, post-storm | -2.56 | 1.27e-4 |

Measured over about 0.03-1 cycle/cm (Figs. 3, 5, 8, 9; figure). Slopes along and across strike do not
differ significantly at Quinault (0.05 level) nor at San Francisco pre-storm; they do post-storm (0.01).

Table V, model inputs (p. 973). w2 in cm^4 (h0 = 1 cm); in brackets our conversion to m^(4 - gamma)
(`w2_from_cm4`):

| Site | nu | rho | delta | sigma2 | gamma along / across | w2 along / across (cm^4) [m^(4 - gamma)] |
|---|---|---|---|---|---|---|
| Quinault | 1.113 | 1.94 | 0.0115 | 0.001 | 3.92 / 3.67 | 0.00598 / 0.00422 [4.137e-3 / 9.232e-4] |
| Arafura Sea | 0.988 | 1.39 | 0.00553 | 0.005 | 3.18 / 3.18 | 0.00318 / 0.00318 [7.285e-5] |
| San Francisco pre-storm | 1.002 | 1.41 | 0.0104 | 0.002 | 3.65 / 3.38 | 0.000846 / 0.000912 [1.688e-4 / 5.248e-5] |
| San Francisco post-storm | 1.002 | 1.41 | 0.0104 | 0.002 | 3.73 / 3.56 | 0.000826 / 0.001296 [2.382e-4 / 1.709e-4] |

sigma2 is the only fitted parameter (**one site** each): Quinault 0.001 (the fit is "rather insensitive" to
it), Arafura 0.005 (0.004-0.006 reasonable), San Francisco 0.002.

Cross-check with J&R (ours): J&R Table 13.1's Quinault row (2.95 phi, 1.94, 1.113, 0.0115, gamma2 3.92,
w2 0.0041) is this paper's along-strike row with w2 converted to m^(4 - gamma2) (0.004137), although this
paper used the across-strike values for Fig. 10 and calls the along-strike exponent unreliable. J&R Table
14.1's Arafura and San Francisco geoacoustics equal Table V. J&R Fig. 11.4 (via [N3] Sec. 2.2) gives other
values for the same sites from Mourad and Jackson (1989): Arafura 1.456 / 0.988 / 0.024, Quinault 1.92 /
1.117 / 0.014.

### 2.2 When volume dominates (confirmed statements)

- Arafura and San Francisco (mean grain size below 0.03 mm, i.e. finer than about 5 phi): sediment volume
  scattering "clearly dominant"; the roughness part with measured inputs falls far short (Figs. 12, 15).
- Quinault (fine sand, 0.13 mm, about 2.9 phi): mainly roughness, except 20-70 deg where volume may be
  significant (comparable at 25-60 deg; below the critical angle, 26 deg, and above about 70 deg roughness
  dominates). A "borderline case".
- Medium and coarse sand and gravel: roughness expected to dominate at all angles (and Stanic et al.,
  Jackson et al. 1986a found so).
- Below the critical angle the volume term is suppressed because little sound enters (p. 972).
- Arafura's frequency independence (Fig. 14, 15-45 kHz) fits a frequency-independent sigma2; with
  attenuation proportional to f that means sigma_v proportional to f, i.e. an inverse-third-power 3-D
  heterogeneity spectrum (gamma3 = 3, self-similar); at Arafura the cause may instead be the gravel-size
  particles.
- San Francisco's angular dependence is stronger than the model's; a sigma2 increasing with depth would
  explain it (p. 975).
- Anisotropy: none measurable at Quinault (+-1 dB) and Arafura; at San Francisco along-strike about 2 dB
  below across-strike at 20-30 deg, before and after the storms (possibly inhomogeneity on 5-10 m scales).
  Directional effects of ripples show in the envelope statistics, not the mean (p. 974).
- A spectral exponent near 4 extrapolated below the measured wavenumbers gives an unrealistic large-scale
  slope (Quinault along strike: 33 deg at 35 kHz) and too strong along-strike scattering; past applications
  had exponents "generally less than 3.7"; a 12-deg slope removes the predicted anisotropy (Fig. 17b).

## 3. Figures to test against (figure, +-1 dB)

Readings of the printed model curves (total; roughness in brackets) and ours: the paper's Table V inputs,
c 1500 m/s, our default small-slope plus volume (and composite plus volume where it differs), `jb.py`.

| Fig., site, f | 10 deg | 20 deg | 30 deg | 40 deg | 50 deg | 60 deg | 70 deg |
|---|---|---|---|---|---|---|---|
| 10a Quinault across, 25 kHz: read | -34.5 (-35.5) | -29.8 (-31.0) | -27.8 (-30.5) | -26.5 (-29.5) | -24.7 (-27.5) | -21.5 (-22.5) | - |
| ours SSA+vol (SSA) | -35.9 (-36.5) | -29.0 (-29.8) | -27.1 (-30.2) | -26.3 (-29.9) | -24.6 (-27.2) | -21.1 (-22.3) | -15.1 |
| ours composite+vol (composite) | -33.9 (-34.3) | -30.0 (-31.0) | -27.1 (-30.2) | -26.2 (-29.8) | -24.6 (-27.2) | -20.7 (-21.7) | -13.6 |
| 10b Quinault across, 35 kHz: read | -33.5 | -29.2 (-30.0) | -27.3 (-30.0) | -26.3 (-29.0) | -24.8 (-27.0) | -21.3 | - |
| ours SSA+vol | -35.5 | -28.5 (-29.3) | -26.8 (-29.7) | -26.0 (-29.3) | -24.2 (-26.6) | -20.5 | -14.8 |
| 12 Arafura, 20 kHz: read | -30.5 (-46) | -26.0 (-43.5) | -24.0 (-42) | -22.5 (-40) | -21.5 (-37) | -21.0 (-33) | -20.5 (-28) |
| ours SSA+vol (SSA) | -28.6 (-46.0) | -24.8 (-43.8) | -23.0 (-42.4) | -21.9 (-40.8) | -21.1 (-38.5) | -20.5 (-35.2) | -19.8 (-30.0) |
| 15 San Francisco pre-storm along, 40 kHz: read | -31.5 (-49) | -28.5 | -27.0 (-47) | -25.8 (-45.5) | -25.0 (-42.5) | -24.3 (-38.5) | -23.5 (-32) |
| ours SSA+vol (SSA) | -31.5 (-48.8) | -28.5 (-47.9) | -26.8 (-46.8) | -25.7 (-45.0) | -24.9 (-42.3) | -24.3 (-38.3) | -23.4 (-31.6) |

Peaks at 90 deg: Quinault about -8 dB read, ours -8.3 / -8.9 (25 / 35 kHz); Arafura and San Francisco the
curves rise to about -5 and -7 dB, ours -4.7 and -6.9. Fig. 14 (Arafura, 20 deg): the model line is flat at
-25.1 dB over 15-45 kHz (figure; ours -24.8 at 20 kHz, frequency independent); the data lie 0-3 dB below it.

Agreement: San Francisco within 0.4 dB at 10-70 deg; Quinault within 1 dB at 20-60 deg (at 10 deg ours is
1.4-2 dB lower with small-slope and 0.2-0.6 dB higher with composite: their model is composite there);
Arafura ours 0.4-1.0 dB higher at 30-70 deg and 1.2-1.9 dB at 10-20 deg (their volume term is
slope-averaged and shadowed, ours is not; that this explains it is our reading, not checked). Their roughness
curve at Arafura lies 1.5-2.2 dB above ours at 50-70 deg (the Kirchhoff blend and slope averaging; not checked).

Measured backscatter (figure, data points): Quinault 25 kHz about -29 dB at 20 deg, -25.5 at 40, -24 at 50,
-20 at 60; Arafura 20 kHz about -28 at 20 deg, -24 at 35-50, -21 at 60; Arafura 35 kHz (Fig. 13, four
headings) about -32 at 10 deg, -27.5 at 20, -25 at 30, -23 at 40; San Francisco 40 kHz about -31.5 at 10,
-27 at 20, -23 at 30 deg.

## 4. Against our code (worst first; computed with the repository's functions)

| # | Item | Paper | Ours (file:line) | Verdict, size |
|---|---|---|---|---|
| 1 | Mud volume parameter at the two mud sites | fitted sigma2: Arafura 0.005 (0.004-0.006), San Francisco 0.002 | class rule 0.001 from 5.5 phi (`bottom.py:682-685`; rows `sediments.yaml:153-208`, `:632-800`); `sigma2_range [0.0008, 0.0037]` with derived 0.0031 / 0.0037 for these two sites in `basis` (`sediments.yaml:172-185`) | DIFFERS, one site each: +7.0 dB (Arafura, 5.63 phi surficial) and +3.0 dB (San Francisco, 6.39 phi) above the class value; the printed fits contradict the agents' derived values at the same sites (0.005 vs 0.0031, 0.002 vs 0.0037). Exposure: 0 shipped scenarios use rows of 5.5 phi and finer (plan U1) |
| 2 | Sand volume parameter | Quinault (2.97 phi) fit 0.001, insensitive | rule 0.002 below 5.5 phi; `medium_sand` 0.001 (D-90) | DIFFERS from the rule by -3.0 dB at one fine-sand site whose fit the paper calls insensitive; weak support for D-90's 0.001 (another grain size). No change |
| 3 | Geoacoustics at the sites against the class relations | Table II | `apl_grain_size_relations` (`bottom.py:718-770`) | DIFFERS, one site each: rho 1.94 / 1.39 / 1.41 against 1.345 / 1.149 / 1.148; delta 0.0115 / 0.00553 / 0.0104 against 0.0172 / 0.0057 / 0.0032; normal-incidence loss 8.7 / 16.1 / 15.3 dB against 14.6 / 23.9 / 24.1 dB; at 20 deg 0.7 / 18.6 / 14.9 against 2.6 / 30.4 / 34.3 dB (`refl.py`). Known kind (J&R p. 314 "ad hoc"); plan decision 10 |
| 4 | Roughness of the sites against the w2 rule | Table V | `grain_size_w2_cm4` (`bottom.py:674-679`), gamma2 3.25 | Spectrum at the 30-deg Bragg wavenumber, site / rule: Quinault +2.3 / +1.1 / -0.6 dB at 20 / 40 / 100 kHz; San Francisco +1.5 / +0.3 / -1.3; Arafura +8.0 / +8.2 / +8.5 (volume dominated there, so the total hardly moves). One site each; agrees with [N2] Sec. 3.2 (rule low for muds at low frequency) |
| 5 | Composite cut-off | text: k h_s = 1; (A8): k h_s = 2^(-1/2) | `small_scale_kh=0.5` default (`bottom.py:490-499`), J&R (13.72)-(13.75) | DIFFERS between prints; on our `composite` at 5 deg -1.2 to -1.4 dB (0.707) and -2.0 to -2.8 dB (1.0), at 10 deg -0.5 to -1.4, at 20 deg under 0.2 dB (`kh.py`). 0 shipped |
| 6 | Monostatic regime scheme | composite below about 80 deg, Kirchhoff above, switch at 15 dB below the Kirchhoff peak; volume slope-averaged and shadowed | small-slope (`bottom.py:548`); volume not averaged (`:596-608`); composite has no shadowing ([N4] Sec. 3.4, G-11) | DIFFERS in method; in numbers within about 1 dB at 20-70 deg with the paper's inputs (Sec. 3); at 10 deg -2 to +1.9 dB depending on site (Sec. 3) |
| 7 | Perturbation backscatter | (A2), (A5)-(A7) | `perturbation_strength` (`bottom.py:383-389`) | MATCHES: 5.7e-14 dB, 3 sites x 2 frequencies x 7 angles |
| 8 | Volume backscatter | (A11)-(A13), sigma2 by (3) | `volume_strength` (`bottom.py:596-608`) | MATCHES: 3.6e-14 dB (same cases); confirms sigma2 is relative to amplitude attenuation in dB (20 log) |
| 9 | delta from attenuation | (2) | `parameters_from_physical` (`bottom.py:185-195`) | MATCHES (algebra) |
| 10 | 1-D to 2-D strength | (4) | no function (plan G-18 to add one) | the paper's form is the corrected J&R (D.19)/(D.23): G-18 can use (4) directly, and Table III -> Table V is its test |
| 11 | Spectrum convention and w2 units | (A3), Table IV: cm^4 at 1 rad/cm, h0 = 1 cm | `w2_from_cm4` (`bottom.py:176-182`) | MATCHES; also confirms J&R Table 13.1's m^(4 - gamma2) (Quinault 0.004137 -> printed 0.0041) |
| 12 | Large spectral exponent | gamma near 4 extrapolated: unrealistic slope; past fits below 3.7 | no warning | DIFFERS: with Quinault along-strike (gamma 3.92) our small-slope is 5.6 dB above across-strike at 30 deg and 10.3 dB at 40 deg (35 kHz, `jb.py`), the same artefact the paper describes for its composite model |
| 13 | Anisotropy | mean level isotropic within 1-2 dB at three sites | isotropic spectrum only | MATCHES the data's conclusion (anisotropic spectra, plan G-27, are not needed for mean levels at these sites) |

## 5. Corrections to apply (ordered; each can be executed without further questions)

Standing rule (the owner's): a printed reference wins over ours; a figure reading or an agent's derivation
does not override a printed class value; one site's value is a named site row, not a class default; where
no source exists nothing changes. No C++ or CUDA twin exists for the bottom scattering models or the
sediment table (`libs/` has none; both engines take strengths from Python). No default result moves in
5.1-5.5.

**5.1 Named site rows** (S). `configs/sediments.yaml`, section "Named site rows", `status: confirmed`,
source this paper (Table V; Tables I-II for Mz), one site each:
- `jackson_briggs_1992_quinault`: rho 1.94, nu 1.113, delta 0.0115, `scattering: {w2_cm4: 0.00422, gamma2:
  3.67, sigma2: 0.001}` (across-strike, as Fig. 10), `mean_grain_size_phi: 2.97`; note: along-strike 0.00598
  / 3.92 printed but unreliable (p. 974); sigma2 fit insensitive; delta follows the whole-core attenuation
  (Sec. 6).
- `jackson_briggs_1992_arafura`: 1.39, 0.988, 0.00553, `{w2_cm4: 0.00318, gamma2: 3.18, sigma2: 0.005}`,
  Mz 5.63; note: 0.004-0.006 fit; bimodal; at the border of single scattering.
- `jackson_briggs_1992_san_francisco`: 1.41, 1.002, 0.0104, `{w2_cm4: 0.000846, gamma2: 3.65, sigma2:
  0.002}` (pre-storm along-strike, as Fig. 15), Mz 6.39; note: the w2 misprint question (Sec. 6) and the
  post-storm and across-strike values.
Tests first (`tests/python/test_sediment_table.py`): `SITE_ROWS` gains the three names (the completeness
test fails before); each row equals Table V as printed and loads without a warning; with the
`fluid_sediment` default model, `jackson_briggs_1992_san_francisco` at 40 kHz, c 1500, gives -31.5 / -28.5
/ -26.8 / -25.7 / -24.9 / -24.3 / -23.4 dB at 10-70 deg (ours, 0.05 dB) and is within +-1 dB of the Fig. 15
readings (figure); `jackson_briggs_1992_arafura` at 20 deg within +-1 dB of Fig. 14's -25.1 at 15, 25, 35,
45 kHz (ours -24.8). Moves: nothing (new rows).

**5.2 The soft rows' range and basis** (S; information keys only, no model reads them). In
`configs/sediments.yaml` the seven rows of 5.5 phi and finer: `sigma2_range: [0.0008, 0.005]` (was
`[0.0008, 0.0037]`), and the `basis` anchor (`:173`) gains: "Jackson and Briggs (1992) Table V print fitted
values at two of these sites, Arafura Sea 0.005 (0.004-0.006; 5.63 phi; bimodal, at the border of single
scattering) and San Francisco 0.002 (6.39 phi); the agents' (14.10) values for the same sites, 0.0031 and
0.0037, disagree with the authors' fits." Class values unchanged (one site each; a printed fit at a site is
not a class value). Test change, stated plainly: `test_the_soft_rows_keep_their_class_values_and_carry_the_measured_range`
asserts `[0.0008, 0.0037]` today; it is changed to `[0.0008, 0.005]` and to require "Jackson and Briggs",
"0.005" and "0.002" in `basis` (fails before, passes after); the class-value assertions stay. Moves: no
number (the loader ignores the keys). Also `docs/tasks/SEDIMENT_DEFAULTS_DECISION.md` decision 1's evidence
table should gain the two printed fits (proposed, not edited here).

**5.3 Spectral-exponent warning** (S; extend plan G-11's warning part). In `bottom.py`, `small_slope`,
`composite`, `combined` (and `williams_jackson_1998` of [WJ] 5.1): warn once when gamma2 > 3.7, citing p. 974
(past fits below 3.7; at 3.92 the extrapolated large-scale slope is 33 deg). Test: warns for the Quinault
along-strike inputs (3.92), not for across-strike (3.67); values bit for bit unchanged. Moves nothing.

**5.4 Documents** (no behaviour). `bottom.py` docstring at `:100-129`: the cut-off is printed three ways (J&R
1/2; this paper's (A8) 2^(-1/2); its text 1); `small_scale_kh=0.7071` reproduces (A8). Docstring at
`:126-127`: the published monostatic scheme's switch (Kirchhoff to 15 dB below its peak) is now read here;
the interpolation formula remains Mourad and Jackson (1989). `docs/PHYSICS_NOTES.md` Sec. 23.2 the same.
`docs/references/jackson_richardson_2007/02_*` Sec. 2.1 and `04_*` Table 13.1 note: (D.19)/(D.23)
corrections and Table 13.1 units now confirmed by a primary print (append-only notes; proposed here).

**5.5 Test data for the 1-D to 2-D helper** (with plan G-18): Table III -> Table V by (4) for six rows
(0.1 %), and the seventh (San Francisco pre-storm along) recorded as 0.000885 computed against 0.000846
printed.

Not applied, with the reason:
- Changing the mud class sigma2 (owner decision 1): one paper, two sites, one of them bimodal and at the
  multiple-scattering border; the printed rule stays. The new evidence: printed fits +7.0 and +3.0 dB above
  the class value (mean +5.0 dB), against the agents' derived +4.9 and +5.7 dB.
- Changing the class density/speed/loss relations to measured values (decision 10): the relations are
  printed (J&R (11.1)-(11.3)); three sites do not make a class value.
- Slope averaging and shadowing of the volume term (G-16): blocked on G-11 (Wagner 1967) as before; this
  paper says the published model does it, which is the evidence for G-16, and the Arafura 10-20 deg
  difference (+1.2 to +1.9 dB) is its likely size (ours, not checked).
- The Kirchhoff/composite interpolation formula: Mourad and Jackson (1989), extracted by another agent in
  parallel; this note records only the 15 dB switch rule.

## 6. Not verified, unclear or misprinted

| Item | What | Evidence |
|---|---|---|
| Table V, San Francisco pre-storm along-strike w2 | 0.000846 printed; (4) from Table III gives 0.000885 (+4.6 %, 0.19 dB) | the other six rows agree within 0.11 % (`jb.py`); which number is misprinted is not known |
| Table V, Quinault delta 0.0115 | follows Table I's whole-core 0.37 dB/m/kHz with c 1.524 m/ms; Table II's surficial 0.30 gives 0.0093 | Arafura and San Francisco follow Table II (c 1.527, 1.531 m/ms), and the text says Table V uses surficial values (p. 973) |
| K_c definition | text: small-scale mean-square height = 1/k^2; (A8) implies 1/(2 k^2) | ours, algebra and `jb.py` (A8 = our split at k h_s = 0.7071 to 5 digits) |
| Fig. 14 | label "95 % confidence band", caption "90 %" | print |
| Fig. 10 caption | "Quinault size" for "site" | print |
| (A4) and (A10) | strict 2 < gamma < 4 | [WJ] (6) prints <= |
| The Kirchhoff/composite interpolation and the shadowing function | not printed here | Mourad and Jackson (1989); Jackson et al. (1986a); McDaniel and Gorman (1982) |
| Model-curve readings (Sec. 3) | by eye on 10-dB grids | +-1 dB |
| Our model in Sec. 3 | uses c 1500 m/s (not printed) | about 0.1 dB per 2 % in c |
