# Bistatic bottom scattering: Williams and Jackson (1998), extracted for the simulator

Status words as in `docs/REFERENCES.md`: **confirmed** = read on the page image of this paper; **via
secondary** = another author's result as this paper reprints it; **ours** = our derivation or a number we
computed (script named); **figure** = read off a printed curve (the README's "read from figure"; about
+-0.5 dB on these grids, never a test bound tighter than +-1 dB); **one site** = a value measured or
fitted at one site. The paper is all rights reserved: formulas re-typed in our notation, constants with
page and equation, table or figure number, own words; the PDF is not in the repository.

Scripts (scratchpad, not committed; run with `PYTHONPATH=<worktree>/python` and the main checkout's built
module linked in temporarily): `/tmp/claude-0/-home-user-claude-playground/32e65dcd-82ae-5f75-a363-7372d2e7f847/scratchpad/ref-b/wj.py`
(the model re-typed, identity checks, model tables), `.../ref-b/wj_data.py` (Table AII against the models),
`.../ref-b/wj_gm.py` (geometric-mean rule), `.../ref-b/vectors.py` (test vectors), `.../ref-b/ell.py`
(eqs. (25)-(26)), `.../ref-b/gal.py`, `.../ref-b/kh.py`. Outputs beside them (`*.out`).

## 0. What disagrees, first

1. **Our default roughness model is not the paper's bistatic model.** The paper joins Kirchhoff and
   first-order perturbation theory by a smooth minimum, eq. (10) with eta = -2. Our default is the
   small-slope approximation (`bottom.py:548`, D-86), which J&R (2007) print and recommend. Re-typed at the
   paper's Panama City inputs (Table I, 40 kHz), the paper's model reproduces its own Figs. 7-8 within the
   reading error (Sec. 3) and lies **below ours by up to 6.1 dB in backscatter (20 deg) and 5.3 dB out of
   plane (20/20 deg, phi 90)**; for our `medium_sand` at 30 kHz up to 6.8 dB (30 deg backscatter). At the
   specular direction and at 5 deg the two agree for sands; where the Kirchhoff and perturbation results
   are equal (smooth relief: very fine silt) the paper's rule is 1.5 dB below both by construction (ours,
   `wj.py`).
2. **Against the paper's own data the paper's model is closer than ours.** Table AII (Panama City, 40 kHz,
   142 values) against each model with the paper's volume term: the paper's model mean -0.45 dB / rms 1.31 dB
   on 68 backscatter values and +3.34 / 4.85 dB on 74 bistatic values; our small-slope +1.36 / 1.73 and
   **+6.08 / 7.11 dB**; our `composite` +4.33 / 4.72 and +6.31 / 7.06 dB (ours, `wj_data.py`). One site;
   the fluid model is known to overpredict at Panama City (J&R p. 351, poroelasticity), so this does not
   show which roughness approximation is right.
3. **Our volume term is isotropic in bistatic angle; the paper's is not** (eqs. (21)-(22), the J&R (14.10)
   form with rho_rk = -1). At Eckernfoerde (Table I) the paper's volume term has a null near phi = 39 deg
   for theta_i = theta_s = 10 deg (ours, `vectors.py`; the printed curves of Fig. 8 dip below the axis there,
   figure). No class value exists for its three parameters: two named sites only.
4. **Everything else we compute matches the print exactly**: the perturbation cross section (16)-(19) to
   5.7e-14 dB, the Kirchhoff cross section (11)-(15) to 2.8e-14 dB, the structure constant (14) to 1e-16,
   the transmission form of the volume term (20) to 1.1e-14 dB, the bistatic azimuth convention, and the
   spectrum convention with w2 in cm^4 at K = 1 rad/cm (ours, `wj.py`).
5. Print problems (Sec. 6): the abstract's "5 deg" horizontal beam against 6.5 deg in the text; "delta^2"
   in the list of reflection-coefficient parameters (p. 171); a figure reference "(b) and (f)" for
   "(b) and (d)" (p. 177); in Table AI one strength printed positive (+35.5); in Table AII one entry printed
   twice and one value truncated ("-33.").

## 1. Citation, scope, data, relation to the restricted report (confirmed)

K. L. Williams and D. R. Jackson, "Bistatic bottom scattering: Model, experiments, and model/data
comparison", J. Acoust. Soc. Am. 103 (1), 169-181 (January 1998). Applied Physics Laboratory, University
of Washington. Received 1 December 1996, accepted 1 September 1997. 13 pages, typeset. Read: every page
image (pp. 169-181, incl. the appendix tables and references); eqs. (2)-(4), (10)-(22) and (25)-(28)
re-rendered at 220 dpi; Figs. 7 and 8 at 220 dpi for readings; Table AII also from the PDF text layer
(checked against the page image).

What it is: the bistatic generalisation of the APL-UW high-frequency backscatter model (Jackson et al.
1986, Mourad and Jackson 1989, Jackson and Briggs 1992), with roughness and volume terms, a time-domain
intensity integral for bistatic geometry, and a test against bistatic and backscatter data from two
Coastal Benthic Boundary Layer sites.

Stated validity and assumptions (pp. 169-170): 10-100 kHz; the sediment a lossy fluid (elasticity and
porosity neglected); no gradients of the mean properties; slight penetration, so volume scattering is
treated as an interface cross section; relief an isotropic 2-D Gaussian process with a power-law spectrum,
2 <= gamma2 <= 4 (eq. (6)); volume inhomogeneities weak (Born), power-law spectrum, compressibility
fluctuations proportional to density fluctuations; transverse isotropy, so only the azimuth difference
matters. Kirchhoff near the specular direction, perturbation elsewhere (p. 170).

Data (Sec. II, pp. 173-176): the BAMS tripod (5 m high, 40 kHz, FM 39-41 kHz, 2 ms, source level
217 dB re 1 uPa at 1 m; horizontal full width 6.5 deg at half power, vertical 14 deg transmit and 27 deg
on receive, first side lobes -13 to -15 dB, axis depressed about 12.5 deg; a 360-deg scan in 72 steps of
5 deg) as transmitter, and ship-deployed steerable arrays (quadrants 32 cm long, beam widths 8 and 37 deg)
as receivers 20-100 m away (under 60 m for the data shown). Geometry uncertainties: compass +-2 deg, range
+-1 m (about 2 m in practice), altimeter and pressure +-30 cm, inclinometers +-0.1 deg (receiver array
calibrated to +-0.01 deg). Strengths derived by simulating each received intensity series with eq. (27)
and the measured beam patterns.

| Site | Sediment (paper's words) | What dominates (paper) |
|---|---|---|
| Panama City | coarse sand; Table I from Ref. 30 (D. Jackson, Briggs, Williams, Richardson 1996, IEEE JOE 21) | "dominated by the measured parameters rho, nu, delta, gamma2 and w2" (p. 176): roughness |
| Eckernfoerde Bay | gassy mud: porosity about 85 %, attenuation 0.18 dB/kHz/m, methane bubbles 0.5-2.0 m below the interface; volume strengths from bubble data -10 to -20 dB/m | bubbles (volume), Tang et al. 1994; the volume parameters are phenomenological, and w3 was adjusted (not Ref. 30's value) |

Relation to the restricted report: the paper neither cites nor mentions APL-UW TR 9407. It cites a
different report, D. Jackson, "A model for bistatic bottom scattering in the frequency range 10-100 kHz",
APL-UW TR 9305 (1993) (Ref. 31), for volume scatterers with monopole and dipole parts. That report was not
read or fetched. Everything below is what this paper prints: **confirmed** for the paper.

## 2. Formulas, in our notation (confirmed unless marked)

Angles: grazing; theta_i incident, theta_s scattered; phi_s the bistatic azimuth, the difference of the
horizontal directions of the incident and scattered waves (Fig. 1). From eq. (2) and Table AI
(backscatter tabulated at phi = 180.0): **phi_s = 180 deg is backscatter, 0 forward**, our
`reverb_bistatic.bistatic_azimuth` convention. k is the water wavenumber (rad/m). Time dependence not
stated (intensities only). log is base 10.

| Our symbol | Paper | Definition |
|---|---|---|
| rho | rho | sediment / water mass density ratio |
| nu | nu | sediment / water sound speed ratio |
| delta | delta | imaginary / real part of the sediment wavenumber |
| w2, gamma2 | w2, gamma2 | relief spectral strength and exponent |
| (none; G-15) | w3, gamma3, mu | volume spectrum strength, exponent; compressibility/density fluctuation ratio |
| q_h / (2k) | Delta_t | half the horizontal change of the unit wave vector |
| q_z / (2k) | Delta_z | half the vertical change |
| q / (2k) | Delta | half the magnitude of the change |

Scattering strength (1): S_b = 10 log10(sigma_br + sigma_bv), cross sections per unit area per unit solid
angle (dimensionless).

Geometry (2)-(4):

    Delta_t = (1/2) [cos^2 th_i - 2 cos th_i cos th_s cos phi_s + cos^2 th_s]^(1/2)      (2)
    Delta_z = (1/2) (sin th_i + sin th_s)                                                (3)
    Delta   = (Delta_t^2 + Delta_z^2)^(1/2)                                              (4)

Relief spectrum (5)-(6): W(K) = w2 / (h0 K)^gamma2, 2 <= gamma2 <= 4. K the magnitude of the 2-D wave
vector. Normalised so that the integral over a region of K space (both signs of the components) is the
mean-square height contributed by those components. **w2 has dimension length^4; h0 is a reference length
set numerically to 1** (p. 170; Ref. 4). Table I gives w2 in **cm^4 "at wave number 2 pi / lambda =
1 cm^-1"**, i.e. h0 = 1 cm and K in rad/cm: our `w2_from_cm4` (w2 [m^(4 - gamma2)] = w2 [cm^4]
x 10^(2 gamma2 - 8)) is that convention (ours: it reproduces J&R Table 13.1's Panama City 93 row, 0.0001475,
from this paper's 0.00849 cm^4 at gamma2 3.12, exactly; Sec. 4).

Volume spectra (7)-(9): W_rr(k) = w3 / (h0 k)^gamma3 (3-D, isotropic; its integral over a volume of k
space is the mean-square density fluctuation over the squared mean density); W_kk = mu^2 W_rr; W_rk =
mu W_rr. w3 in **cm^3 at k = 1 cm^-1** in Table I: w3 [m^(3 - gamma3)] = w3 [cm^3] x 10^(2 gamma3 - 6)
(ours, same argument as for w2; 1.3e-4 cm^3 -> 0.013 m^-1 at gamma3 = 4).

Roughness, interpolation (10): **sigma_br = [sigma_kr^eta + sigma_pr^eta]^(1/eta), eta = -2**: "the
smaller cross section takes precedence" (p. 171). Reason printed: with power-law spectra perturbation
theory overpredicts near specular (the spectrum's singularity at K = 0) and Kirchhoff overpredicts
elsewhere (Thorsos 1990).

Kirchhoff (11)-(15):

    sigma_kr = |R(th_is)|^2 / (8 pi) [Delta^2 / (Delta_z Delta_t)]^2  int_0^inf exp(-q u^(2 alpha)) J0(u) u du   (11)
    q        = 2 k^2 Delta_z^2 C_h^2 (2 k Delta_t)^(-2 alpha)                                                   (12)
    alpha    = gamma2 / 2 - 1                                                                                   (13)
    C_h^2    = 2 pi w2 Gamma(2 - alpha) 2^(-2 alpha) / [h0^gamma2 alpha (1 - alpha) Gamma(1 + alpha)]          (14)
    th_is    = arcsin(Delta)                                                                                    (15)

R is the flat-interface (Rayleigh) pressure reflection coefficient at th_is, the grazing angle of a facet
tilted to reflect specularly from transmitter to receiver (Thorsos, private communication, finds it better
than other choices; p. 172).

Perturbation (16)-(19):

    sigma_pr = (1/4) k^4 |1 + R(th_i)|^2 |1 + R(th_s)|^2 |G|^2 W(2 k Delta_t)                         (16)
    G        = (1/rho - 1) [cos th_i cos th_s cos phi_s - P(th_i) P(th_s) / rho] + 1 - kappa^2 / rho  (17)
    kappa    = (1 + i delta) / nu                                                                    (18)
    P(th)    = (kappa^2 - cos^2 th)^(1/2)                                                            (19)

(branch: Im P > 0, decaying into the sediment; not stated, ours as in `bottom._vertical`.)

Volume (20)-(22):

    sigma_bv = |1 + R(th_i)|^2 |1 + R(th_s)|^2 sigma_v / (2 k rho^2 Im[P(th_i) + P(th_s)])                      (20)
    sigma_v  = (pi / 2) k^4 |mu kappa^2 + cos th_i cos th_s cos phi_s - P(th_i) P(th_s)|^2 W_rr(Delta k)        (21)
    Delta k  = k |4 Delta_t^2 + {Re[P(th_i) + P(th_s)]}^2|^(1/2)                                               (22)

sigma_v is the volume cross section per unit volume (1/m); Delta k the Bragg wavenumber of the volume, the
magnitude of the difference of the real parts of the incident and scattered wave vectors in the sediment.
(20) is the bistatic form of Stockhausen's interface equivalent used by Mourad and Jackson (via secondary
for those two). Mapping to J&R (14.10) (ours, algebra): with W_kk = mu'^2 W_rr and W_rk = rho_rk mu' W_rr,
(14.10) is (pi/2)[mu'^2 |k_p|^4 + 2 rho_rk mu' Re{(k_ps . k_pi) k_p*^2} + |k_ps . k_pi|^2] W_rr; for
rho_rk = -1 this is (pi/2) |k_ps . k_pi - mu' k_p^2|^2 W_rr, and with k_ps . k_pi = k^2 (cos th_i cos th_s
cos phi_s - P(th_i) P(th_s)) and k_p^2 = k^2 kappa^2 it is (21) with **mu' = -mu**. The paper's Table I mu
is negative (-0.69, -2.44): density and compressibility fluctuations anti-correlated, J&R's "common
practice" rho_rk = -1 with mu' = 0.69 and 2.44. At backscatter the bracket of (21) is kappa^2 (mu - 1), so
|...|^2 = |kappa|^4 (mu' + 1)^2.

Scattered intensity (23)-(28), for a pulse of infinitesimal length (convolve with the transmitted
envelope):

    I(t)  = I0 int int [sigma b_x b_r / (D1 D2)^2] 10^(-alpha_w (D1 + D2) / 10) dx dy                (23)
    D     = D1 + D2 = c t                                                                            (24)
    (x - x0)^2 / a^2 + y^2 / b^2 = 1                                                                 (25)
    a = Q / eps,  b = Q / (2 D),  x0 = -beta / (2 eps^2),  eps = 2 [D^2 - (x1 - x2)^2]^(1/2)          (26a-d)
    beta = 4 (x2 - x1)(x2^2 - x1^2 + d2^2 - d1^2 - D^2) - 8 x1 D^2                                    (26e)
    Q = D {([D^2 - (x1 - x2)^2 - d1^2 - d2^2]^2 - 4 d1^2 d2^2) / (D^2 - (x1 - x2)^2)}^(1/2)          (26f)
    I(t)  = I0 (Delta D) int_{-b}^{b} [|dx/dD| sigma b_x b_r / (D1 D2)^2] 10^(-alpha_w D / 10) dy      (27)
    x     = x0 +- a (1 - y^2 / b^2)^(1/2)   (both branches summed)                                   (28)

Transmitter at (x1, 0, d1), receiver at (x2, 0, d2) (Fig. 2); sigma the sum of the two cross sections of
(1); b_x, b_r the beam patterns; alpha_w the water attenuation in dB per unit length; I0 the source
intensity at 1 m. (25)-(26) checked: points on the printed ellipse have D1 + D2 = D to 1e-9 m for three
geometries (ours, `ell.py`).

### 2.1 Table I, bottom parameters (confirmed; p. 171)

| Symbol | Meaning, units as printed | Eckernfoerde | Panama City |
|---|---|---|---|
| rho | density ratio | 1.18 | 1.97 |
| nu | sound speed ratio | 0.991 | 1.126 |
| delta | loss parameter | 0.00186 | 0.0166 |
| gamma3 | inhomogeneity exponent | 4 | 4 |
| w3 | inhomogeneity strength, cm^3, at 2 pi / lambda = 1 cm^-1 | 0.00013 (adjusted, not Ref. 30's) | 0.0000161 |
| mu | compressibility / density fluctuation ratio | -0.69 | -2.44 |
| gamma2 | relief exponent | 3.42 | 3.12 |
| w2 | relief strength, cm^4, at 2 pi / lambda = 1 cm^-1 | 0.00231 | 0.00849 |

In SI (ours): w2 = 1.598e-4 m^0.58 (Eckernfoerde), 1.475e-4 m^0.88 (Panama City); w3 = 0.013 and 0.00161
m^-1. Water sound speed is not printed; 1500 m/s is used below (a 2 % change in c moves the roughness
term by about 0.1 dB).

### 2.2 Tables AI and AII, measured strengths (confirmed; pp. 179-180; summarised, not copied)

Each entry is (theta_i, theta_s, phi, 10 log sigma) at 40 kHz. Table AI (Eckernfoerde): about 100
backscatter values at 5-14 deg (repeated passes) and about 125 bistatic values (phi 30-172 deg, theta
5.6-47 deg; counted from the text layer, ours). Table AII (Panama City): 68 backscatter values at 5-15 deg and 74 bistatic values (phi
14-148 deg, theta 5.5-61 deg). Summary of Table AII backscatter (ours, mean of intensity of the repeats):

| theta | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 6 | 6 | 6 | 6 | 6 | 7 | 6 | 6 | 6 | 6 | 7 |
| S_b (dB) | -39.1 | -37.4 | -36.0 | -34.5 | -33.5 | -32.2 | -31.6 | -30.2 | -28.9 | -28.2 | -28.3 |

A few Table AII bistatic entries for tests (confirmed, as printed: theta_i, theta_s, phi, dB): (16.8, 15.3,
138.3, -30.7); (21.5, 17.6, 121.7, -31.2); (25.0, 21.5, 95.0, -29.3); (17.8, 12.7, 93.1, -32.0); (13.8,
22.2, 58.0, -31.4); (23.3, 33.1, 58.9, -21.7); (57.6, 35.6, 63.3, -20.4); (20.5, 16.8, 26.0, -20.5); (13.7,
20.6, 21.0, -20.7); (9.9, 16.7, 14.3, -36.3). The rest: p. 180.

## 3. Figures to test against (figure, +-0.5 dB on the 5-dB grid; abscissa phi in degrees)

Fig. 7 (Panama City) and Fig. 8 (Eckernfoerde), p. 176-177: model curves for theta_i = theta_s at the two
ends of each range, against phi, with the data. Readings and our re-typed model (Table I inputs, 40 kHz,
c 1500 m/s, `wj.py` section 3):

| Fig., curve | phi 180 read / ours | phi 100 read / ours | phi 50 read / ours |
|---|---|---|---|
| 7a, 10 deg | -31.6 / -31.9 | -31.9 / -32.1 | -31.2 / -31.2 |
| 7a, 5 deg | -40.0 / -40.1 | -40.8 / -40.8 | -41.2 / -41.3 |
| 7b, 20 deg | -29.5 / -29.7 | -29.1 / -29.3 | -25.3 / -25.6 |
| 7b, 10 deg | -31.8 / -31.9 | -32.0 / -32.1 | -31.0 / -31.2 |
| 7c, 35 deg | -27.6 / -27.9 | -26.7 / -26.6 | - |
| 7d, 60 deg | -21.9 / -22.1 | -20.1 / -19.6 | - |
| 8a, 10 deg | -34.0 / -34.0 | - | null (curve below -45 between about 35 and 50) / null at 39.3 |
| 8a, 5 deg | -39.0 / -39.1 | - | null between about 35 and 60 / null at 42.8 |
| 8b, 15 deg | -31.7 / -31.8 | - | - |
| 8c, 20 deg | -30.6 / -30.4 | - | null near 20 / 18.4 |
| 8d, 20 / 10 deg | -30.3 / -30.4, -33.9 / -34.0 | - | - |

All within 0.5 dB: the re-typed model with these inputs is the paper's (ours, a check of the re-typing).
Fig. 6 (three views of the model surface over (theta_s, phi) at theta_i = 5, 20, 60 deg with data) is not
readable to better than a few dB; not used. Fig. 5 (two time series near forward and near backscatter with
simulated bottom and surface signals) shows the method; not used.

Text statements (confirmed): the 5-deg data agree in level and angular dependence; the 20-deg data show the
predicted rise toward forward scatter; the plateau away from forward is right in shape, but "the plateau
predicted by the model may be too high" (Fig. 6b, 7b), the model "overpredicts the bistatic scattering by
a few decibels at some angles" (p. 178). At Eckernfoerde the model is "empirical, rather than a physical"
one; a bubble model with multiple scattering may be needed.

## 4. Against our code (worst first; computed with the repository's functions)

Our bottom is evaluated at the true bistatic geometry: `simulator.py:200-203` passes the cell's grazing
angles and bistatic azimuth to `bottom.strength` (`bottom.py:611-614`); `reverb_bistatic.py:373-379` calls
it with three arguments and `:402` computes the azimuth. The surface laws' geometric mean (A3,
`surface.py:316-341`) is **not** applied to `fluid_sediment`; it is applied to the empirical bottom laws
(`searay_bottom_strength`, `surface.py:393`) and Lambert's law is its own geometric mean.

| # | Item | Paper | Ours (file:line) | Verdict, size (ours unless marked) |
|---|---|---|---|---|
| 1 | Roughness regime join | (10), eta = -2, Kirchhoff (11) with perturbation (16) | `DEFAULT_ROUGHNESS_MODEL = "small_slope"` (`bottom.py:548`), `small_slope_strength` (`:477-487`); the docstring rejects a smooth minimum (`:144-152`) | DIFFERS. Paper minus ours, roughness only (`wj.py` section 4): backscatter Panama City 40 kHz -1.5 / -6.1 / -5.1 / -2.8 / -2.0 dB at 10 / 20 / 30 / 45 / 60 deg; medium sand 30 kHz -0.6 / -5.1 / -6.8 / -3.0 / -2.2; sandy gravel 30 kHz -0.5 / -4.1 / -5.7 / -4.3 / -2.5; very fine silt -0.6 to -1.6. Out of plane (20/20 deg, phi 90): -5.3 (Panama), -4.0 (medium sand), -2.6 (sandy gravel). Near forward (20/20, phi 10): -0.2 to -2.6. Within 1 deg of specular and at 5 deg backscatter: under 0.03 and 0.2 dB, except very fine silt. **Where Kirchhoff and perturbation agree (smooth relief, very fine silt) eta = -2 puts the result 1.5 dB below both** (10 log10 2^(1/2)): -1.4 to -1.8 dB at every geometry, also at specular. At 100 kHz the differences are those at 30 kHz within 0.5 dB for backscatter at 10-60 deg and phi >= 30 deg, up to 1.6 dB near forward and at 75 deg. Exposure: **0 shipped** (no `geometry: bistatic` `fluid_sediment` item; the 4 monostatic items would move only if the default changed) |
| 2 | Model against measured bistatic data | Table AII | small-slope + the paper's volume | DIFFERS from the data more than the paper's model: bistatic mean +6.08 / rms 7.11 dB (ours) against +3.34 / 4.85 (paper's model); backscatter +1.36 / 1.73 against -0.45 / 1.31; first-order perturbation alone as small-slope within 0.06 dB; `composite` worse (+4.33 / 4.72 backscatter). **one site** |
| 3 | Perturbation | (16)-(19) | `_perturbation_factor` (`bottom.py:356-370`), `perturbation_strength` (`:383-389`) | MATCHES: worst 5.7e-14 dB, 10 geometries x 2 sites (incl. out-of-plane and theta_i != theta_s) |
| 4 | Kirchhoff | (11)-(15) | `kirchhoff_strength` (`:464-474`), `structure_constant` (`:392-398`), `kirchhoff_integral` (`:430-443`) | MATCHES: 2.8e-14 dB; C_h^2 relative 1.1e-16; the integral against direct quadrature to 7 digits at q = 0.082 and 0.277 |
| 5 | Volume, transmission form | (20) | `volume_strength` (`:596-608`) | MATCHES for the same sigma_v: 1.1e-14 dB |
| 6 | Volume cross section | (21)-(22): spectral, dipole + monopole, depends on phi | sigma_v = sigma2 x attenuation in dB/m, isotropic (`:606`) | DIFFERS in angle (G-15; [N4 3.6]). Eckernfoerde (Table I): null at phi 39.3 deg for 10/10 deg, -1.6 dB at phi 90 relative to backscatter (`vectors.py`). Level: the paper's Panama City volume is sigma_v -28.3 dB/m at 20 deg, an equivalent sigma2 of 0.00007, against the grain-size rule's 0.002 (+14.6 dB on the volume term; the total moves +0.4 dB on the backscatter data because roughness dominates); Eckernfoerde equivalent sigma2 0.00133 (ours, `wj_data.py`) |
| 7 | Bistatic azimuth convention | phi_s 180 back, 0 forward ((2); Table AI) | `reverb_bistatic.bistatic_azimuth` (`reverb_bistatic.py:71-75`) | MATCHES |
| 8 | Spectrum normalisation and units | (5), w2 cm^4 at 1 rad/cm | `w2_from_cm4` (`bottom.py:176-182`), docstring `:35-36` | MATCHES; also confirms J&R Table 13.1's w2 units (m^(4 - gamma2)), which [N4] had inferred: Panama City 93 0.00849 cm^4 -> 0.0001475 exactly |
| 9 | Geometric mean for a bottom | none: the paper computes the bistatic cross section | geometric mean for empirical laws only (`surface.py:322-341`, `:393`) | INFORMATION: at Panama City (20/20 deg) the physical bistatic total exceeds the geometric mean of the two backscatter strengths by +0.7 / +2.7 / +9.2 / +16.6 / +29.6 dB at phi 90 / 60 / 30 / 15 / 5; within 0.7 dB for phi >= 90. On the 74 bistatic data the geometric mean of the paper's backscatter is +1.59 / rms 4.07 dB, the paper's bistatic model +3.34 / 4.85 (the data have few points near forward). No printed bistatic rule for an empirical bottom law exists here; nothing changes. Exposure 0 (no bistatic Lambert or `searay_bottom` item) |
| 10 | Intensity integral over the iso-time ellipse | (23)-(28) | `reverb_bistatic.directional_power` bins cells by two-way time; "nothing in the code constructs the ellipse" (PHYSICS_NOTES Sec. 14.2) | MATCHES in kind (same integrand: strength x both beam patterns x spherical spreading x absorption over the area between iso-time ellipses); not compared numerically |
| 11 | Frequency range | 10-100 kHz | `fluid_sediment` used at 20 kHz (`sst_sediment_reverb`) and with the placeholder sand in three scenes | no conflict for the named sediment; the placeholder scenes carry no source anyway |
| 12 | gamma2 range | 2 <= gamma2 <= 4 (6) | small-slope needs 2 < gamma2 < 4; outside, perturbation with a warning (`bottom.py:571-580`) | MATCHES in effect (the end points give no finite structure function) |

What the paper says about when volume dominates (confirmed): the coarse sand (Panama City) is roughness
dominated; in the gassy mud (Eckernfoerde) bubbles dominate at 40 kHz because of low attenuation in the
overlying sediment and high volume strength. Ours agrees in kind: with the paper's inputs the roughness term
at Eckernfoerde is 17-22 dB below the volume term at 5-35 deg backscatter (`wj.py` section 2).

## 5. Corrections to apply (ordered; each can be executed without further questions)

Standing rule used (the owner's): a printed reference wins over ours; a figure reading or an agent's
derivation does not override a printed class value; one site's value is a named site row; where no source
exists nothing changes. Here the paper's interpolation and our default small-slope approximation are
**both printed** (this paper's (10); J&R (13.11), recommended by the book p. 336, D-86), so neither wins by
that rule: the paper's model is added opt-in and the evidence is put to the owner (5.4). No C++ or CUDA
twin exists for any bottom scattering model (`grep -rl "small_slope\|kirchhoff_integral\|volume_strength"
libs/` finds none; both engines take strengths from Python; [plan] Sec. 1 "Engines").

**5.1 Opt-in roughness model `williams_jackson_1998`** (S). In `python/gpusonar/sim/bottom.py`: a function
`williams_jackson_strength(...)` = (kirchhoff_strength^-2 + perturbation_strength^-2)^(-1/2), with zero
where either is zero (q_z = 0); add the name to `ROUGHNESS_MODELS` and `roughness_strength`; docstring cites
(10)-(19) and states the scope (10-100 kHz, fluid, Kirchhoff near specular, perturbation elsewhere, no slope
averaging). Correct the docstring at `:144-152`: the smooth minimum rejected there was Kirchhoff with
*composite*; the paper's published one is Kirchhoff with *perturbation*, which lowers backscatter at 20-45
deg by up to 6.8 dB for sands (item 1). `simulator.py:189` comment and `docs/CONFIGURATION.md`
`roughness_model` values: add the name. `sediment_fit.py` accepts it through `ROUGHNESS_MODELS` (no change).
Tests first, new `tests/python/test_williams_jackson_1998.py` (fails before: unknown model):
- the Panama City Table I inputs (w2 = `w2_from_cm4(0.00849, 3.12)`, rho 1.97, nu 1.126, delta 0.0166),
  40 kHz, c 1500: roughness_model "williams_jackson_1998" gives -40.157, -31.870, -29.799, -28.154, -22.240
  dB at 5, 10, 20, 35, 60 deg backscatter; -29.141 at (20, 20, phi 90); -20.860 at (20, 20, 30); -0.799 at
  (20, 20, 5); -12.292 at (10, 40, 0); -29.179 at (10, 40, 90) (ours, `vectors.py`; tolerance 0.01 dB);
- equal to an inline re-typing of (10)-(19) to 1e-9 dB at those geometries (independent of `bottom.py`);
- never above min(Kirchhoff, perturbation) and within 1.6 dB below it (eta = -2: at most 10 log10(2) = 1.5 dB);
- `small_slope` default untouched bit for bit (`test_bottom_models.py::test_the_perturbation_model_is_unchanged_and_the_default_is_small_slope` keeps passing);
- the parametrised `test_gallaudet_boundaries.py::test_bottom_model_with_the_site_fit_against_the_printed_band_means`
  gains this model automatically: it passes (roughness 22.8 dB below volume at 29-66 deg, `gal.py`).
Moves: nothing shipped (opt-in). For a user who opts in: the item 1 numbers. Status of the formula: confirmed.

**5.2 The figure test of the full model** (S, after 5.1 and with 5.3). Test: with Table I and the paper's
spectral volume term, backscatter at Panama City 5 / 10 / 20 / 35 / 60 deg within +-1 dB of the Fig. 7
readings -40.0 / -31.7 / -29.5 / -27.6 / -21.9 dB (figure; ours -40.15 / -31.85 / -29.73 / -27.87 / -22.13)
and Eckernfoerde 5 / 10 / 15 / 20 deg within +-1 dB of -39.0 / -34.0 / -31.8 / -30.4 (figure; ours -39.06
/ -34.02 / -31.78 / -30.38); the Eckernfoerde volume null between phi 35 and 45 deg at 10/10 deg (ours
39.3). Fails before (no spectral volume, no model).

**5.3 Spectral volume term, opt-in** (M; merge into plan package G-15, which has the J&R (14.10) form).
`volume_strength(..., volume_model="sigma2" | "spectral", w3, gamma3, mu, rho_rk=-1.0)`: sigma_v by (21)
with Delta k by (22), inserted in (20) (= our existing transmission form). Accept the paper's convention
(mu signed, rho_rk implied -1) or J&R's (mu' >= 0, rho_rk); the two agree with mu' = -mu at rho_rk = -1
(ours, Sec. 2). Tests first: Eckernfoerde volume alone -39.139 / -34.068 / -30.412 dB at 5 / 10 / 20 deg
backscatter, -35.653 at (10, 10, 90), -34.038 at (20, 20, 60) (ours, `vectors.py`, 0.01 dB); the null at
phi 42.8 / 39.3 / 32.4 / 18.4 deg for theta 5 / 10 / 15 / 20 (0.5 deg); equals J&R (14.10) with rho_rk = -1
and mu' = -mu to 1e-12; "sigma2" default bit for bit. Default: unchanged (no class values exist for w3,
gamma3, mu). Engines: Python only.

**5.4 Owner decision prepared, not applied: default roughness model for bistatic geometry.** Evidence for
the paper's model: printed by the model's authors for bistatic use; closer to the only bistatic data set
(Table AII: +3.3 against +6.1 dB mean bistatic bias; -0.45 against +1.36 dB backscatter). Against: one
site; the fluid model is known to overpredict at Panama City (J&R p. 351: poroelastic effects; the same
sign as our bias), so a lower roughness term may be fitting a fluid-model error; J&R recommend small-slope
(p. 336), Olson 2026 finds SSA valid where perturbation is; the smooth minimum has no physical basis beyond
"the smaller takes precedence". Recommendation: keep `small_slope` the default; say in
`docs/PHYSICS_NOTES.md` Sec. 23.2 that the published bistatic model is up to 6-7 dB lower at 20-45 deg for
sands. Exposure if switched: 0 bistatic items; the four monostatic `fluid_sediment` items only if the
monostatic default also switched (not proposed).

**5.5 Named site rows** (S; with the Jackson and Briggs rows of the companion note): add to
`configs/sediments.yaml` under "Named site rows" a row `williams_jackson_1998_panama_city`: rho 1.97, nu
1.126, delta 0.0166, `scattering: {w2_cm4: 0.00849, gamma2: 3.12, sigma2: 0.00007}`, with `unconfirmed_fields:
[sigma2]` and provenance "sigma2: ours, the paper's spectral volume (21) at 20 deg backscatter, 40 kHz, as
an equivalent (the paper prints w3 1.61e-5 cm^3, gamma3 4, mu -2.44 instead)"; a `volume_spectral:` block
with the printed three for 5.3; source this paper, Table I; note "one site, coarse sand, 40 kHz; J&R Table
13.1 prints the same site reprocessed to the upper 2 cm (1.948, 1.097, 0.032, gamma2 3.12, w2 0.0001475)".
Eckernfoerde: **no row** (gassy mud; the paper calls its parameters phenomenological and adjusted w3).
Tests: `tests/python/test_sediment_table.py` `SITE_ROWS` gains the name (the completeness test fails before);
a test that the row equals Table I as printed; `bottom.sediment(...)` warns for the unconfirmed sigma2.
Moves: nothing (new row).

**5.6 Documents** (no behaviour): `bottom.py` docstring (5.1); `docs/PHYSICS_NOTES.md` Sec. 23.2: the
published bistatic model and the size of the difference; `docs/references/CONFORMANCE.md` rows (proposed in
the report); `docs/REFERENCES.md` Sec. 10 "Bistatic bottom model" -> found.

Blocked or not applied:
- Mourad and Jackson (1989)'s monostatic Kirchhoff/composite interpolation: not in this paper (it uses
  perturbation, not composite, away from specular). The companion note records Jackson and Briggs' 15 dB
  switching rule; the formula itself: Mourad and Jackson (1989), extracted in parallel by another agent.
- A bistatic rule for empirical bottom laws (Lambert, `searay_bottom`): none printed; the geometric mean
  stays (item 9).
- TR 9305 (bistatic model report, Ref. 31): not read.

## 6. Not verified, unclear or misprinted

| Item | What | Evidence |
|---|---|---|
| Water sound speed | not printed; 1500 m/s used | readings within 0.5 dB, so not material |
| Branch of P | not stated | Im P > 0 assumed (decay into the sediment); the figure readings agree |
| p. 171 "rho, nu, and delta^2" | R depends on delta, not delta^2 | probably a footnote marker "2" (Ref. 2) set as a superscript, or a misprint |
| Abstract "5 deg horizontal beam" | text: 6.5 deg full width at half power; 5 deg is the scan step | inconsistency in the print |
| p. 177 "Figure 5(b) and (f)" | Fig. 5 has panels (a)-(d); (d) meant | caption |
| Table AI | (11.9, 6.4, 159.8) printed +35.5 dB | a lost minus sign (all other values negative, neighbours near -35) |
| Table AII | (37.7, 13.2, 63.7, -29.8) printed twice in a row; (9, 9, 180) printed "-33." | duplicate; truncated value |
| (6) | 2 <= gamma2 <= 4, inclusive | the structure function needs 2 < gamma2 < 4 (Jackson and Briggs (A4) strict) |
| Fig. 6 | 3-D views | not read for numbers |
| Table AI | page break splits two values | the text layer's order was not used for Table AI numbers |
| Model-data rms of Sec. 4 item 2 | uses the PDF text layer of Table AII (142 values parsed; spot-checked against the page image) | the leading glyph "2" is the minus sign |
