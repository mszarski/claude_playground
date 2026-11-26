# Bottom backscatter and forward loss: Mourad and Jackson (1989), extracted for the simulator

**[MJ89]** P. D. Mourad and D. R. Jackson, "High Frequency Sonar Equation Models for Bottom Backscatter and
Forward Loss", Applied Physics Laboratory, University of Washington; printed pages 1168-1175 of the
proceedings OCEANS '89. The pages print only the page numbers; the conference name and year are those of the
PDF's title field ("OCEANS '89. Proceedings") and agree with the paper's own references (Briggs 1989 "accepted
for publication"; Stanic et al. 1989). Volume and publisher are not printed on the pages and are not given here.
Supplied by the owner; all rights reserved; not in the repository. A scan of 8 pages, every page read from the
page images; every equation re-read from 220-dpi renders, the figures from 300-dpi renders.

Status words as in `docs/REFERENCES.md`: **confirmed** = read on the page image of [MJ89]; **via secondary** =
another author's formula or value as [MJ89] reprints it; **ours** = our derivation or a number we computed
(script named); **figure** = a value read off a printed curve, with its reading error, never a printed number.
Symbols: theta is the **grazing** angle throughout (the paper's convention, eq. (9)).

Scripts (scratch, not committed; `.../scratchpad/ref-a/scripts/`): `mj89.py` (the paper's model retyped,
independent of our code), `figs.py` and `variants.py` (model against the figure readings), `ours.py`,
`decomp.py`, `after.py`, `fwd.py` (against `gpusonar.sim.bottom` of this worktree; the main checkout's built
module was linked into `python/gpusonar/` for the run and removed afterwards; output `ours_out.txt`).

## 0. What is wrong, missing or risky (read first)

1. **Eq. (39), the volume term, is misprinted.** It prints the factor |1 - R(theta)|^2 where the paper's own
   figures, Jackson and Richardson (2007) (14.8) and our derivation have |1 - R^2(theta)|^2. With the printed
   factor the model is 1.5-3 dB below the paper's own solid curves at 20-60 degrees (Figs. 6, 9, 10); with the
   squared R it reproduces all seven printed model curves read (Figs. 6-10, surface and two bulk sets) to
   rms 0.3-0.8 dB, worst 1.65 dB (section 3.2). Our `volume_strength` is the correct form (equal to 1e-14
   dB; section 4). Nothing to change in code; a pin test is proposed (section 5, C3).
2. **The paper prints the Kirchhoff/composite interpolation that `docs/REFERENCES.md` Sec. 10 lists as
   wanted**: eqs. (33)-(37), a logistic weight in cos(theta) centred where its Kirchhoff approximation has
   fallen 15 dB below its normal-incidence peak (confirmed). It also prints a closed-form approximation to the
   Kirchhoff integral (19)-(22), exact for gamma2 = 3 (alpha = 1/2) and within 0.6 dB of our exact integral at
   gamma2 = 3.25, but +3.7 dB at gamma2 = 3.67 and +8.4 dB at gamma2 = 3.9 at 60 degrees (ours; section 4,
   row K2). This is the 1989 model; whether TR 9407 kept these forms is not known.
3. **Against our default** (`small_slope` + volume, D-86): the paper's model is up to +11.5 dB above ours at 5
   degrees (sandy gravel 100 kHz; outside the paper's 1-9 phi range and with a large-scale slope 0.42), +9.3
   dB for medium sand at 100 kHz, +3.8 dB at 20 kHz, +1.6 to +6.1 dB for muddy sand at 20-100 kHz; within
   0.4 dB at 5 degrees for silts and clays; at 10 degrees 0.0 to +4.4 dB; 20-85 degrees within -1.6 to +1.0
   dB. The whole low-angle difference is the paper's averaging of BOTH the roughness and the volume term over
   the large-scale slope (eqs. (23), (27)-(29), (38)), which our default does not do (section 4.3). This is a
   choice between two printed models (Jackson and Richardson 2007 recommend small-slope, p. 336); it is not a
   misprint in ours, so the default stays and the paper's model becomes an opt-in (C4).
4. **Large-scale slope (18) does not use the cut-off k h_s = 1/2.** Solved for its cut-off, (18) is the TOTAL
   (both-axes) rms slope of relief below the K_c at which the small-scale relief has k h_s = 1/sqrt(2)
   (ours, algebra; checked numerically: equal to 4 digits in 9 cases). Jackson and Richardson (2007) (13.72)
   use 1/2 (attributed to D. Jackson et al. 1986a), as does our `large_scale_slope`. At gamma2 = 3.25 the
   paper's s is 0.81 of our total slope (ours, `ours.py` section 4). Not a correction to our composite; the
   opt-in model follows (18) as printed.
5. **Grain-size rules: two differences from ours, both kept.** (a) Volume parameter (4): 0.002 for
   1 <= Mz < 5.0, 0.001 for 5.0 <= Mz <= 9. Ours (Snellen et al. 2011/2013, attributed to TR 9407, via
   secondary) steps at 5.5 phi. Only `sandy_silt` (5.0 phi) is affected: the paper's rule would give 0.001
   (-3.0 dB on its volume term). The paper's own Fig. 8 (Arafura, Mz 5.3) used 0.002, i.e. not its printed
   step; the later prints of the same model (Snellen 2011 Table II: 0.002 at 5 phi) step at 5.5. Kept.
   (b) The relief rule (6) steps to h = 0.5 cm at 5.0 phi, as ours; Fig. 8 instead used the rational form
   at 5.3 phi (w2 0.000391 cm^4 printed, against 0.000518 from (6)): an internal inconsistency of the paper.
6. **What the paper settles in our earlier notes.** (a) The constant 1.0091 printed in J&R (11.2) is a
   misprint: [MJ89] (2) prints 1.0018916 (confirmed); our 1.0019 stands. (b) The water speed 1528 m/s used to
   turn Hamilton's k into delta: [MJ89] prints c1 = 1.528 m/ms (confirmed). (c) The Wagner shadowing factor
   with pi^(-1/2) (J&R (13.82) prints pi^(+1/2)): [MJ89] (25) prints pi^(-1/2), as Wagner (1967) (22)
   (`shadowing_wagner_1967.md`). (d) The relief rule (6)-(7) and gamma2 = 3.25 (5) that we hold via Snellen
   2013 are printed here (confirmed). The paper does NOT print the -1 <= Mz < 1 pieces of (11.1)-(11.2), nor
   Hamilton's k(Mz) (11.3), so the 0.4566/0.4556 question stays open; but the five sites' printed inset values
   are reproduced by J&R's pieces to the printed digits, including the 0.84 phi site, which only J&R's
   quadratic piece reproduces (section 2.5).
7. **Range.** 10-100 kHz (abstract); sands to clays, relations defined for 1 <= Mz <= 9 only (confirmed). Our
   `sandy_gravel` (-1 phi) and coarser rows, rock and pebbles, our 8 kHz flank array and the 120-330 kHz FLS
   are outside it.

## 1. Scope

| Pages | Content | How read |
|---|---|---|
| 1168 | abstract, introduction, geoacoustic modelling | page image |
| 1169 | Figs. 1-2, Table 1, eq. (1) | page image; Figs. 1-2 and (1) at 300 / 220 dpi |
| 1170 | eqs. (2)-(12), forward-loss model | page image; equations at 220 dpi |
| 1171 | Figs. 3-5 (forward loss), start of the backscatter model | page image; figures at 300 dpi |
| 1172 | eqs. (13)-(25) | page image; equations at 220 dpi |
| 1173 | eqs. (26)-(39) | page image; equations at 220 dpi |
| 1174 | Figs. 6-10 (backscatter against data) | page image; figures at 300 dpi |
| 1175 | conclusions, references | page image |

Relation to APL-UW TR 9407: the paper predates it (1994) and does not cite it. Jackson and Richardson (2007)
attribute their (11.1)-(11.2) to this paper and to TR 9407. What this paper prints is `confirmed` for
[MJ89]; any statement about TR 9407 drawn from it is at most `via secondary`. The paper cites D. Jackson,
Winebrenner and Ishimaru (1986a) as the model it extends and simplifies, Kuo (1964) for the perturbation
expression, McDaniel and Gorman (1982) for the slope average, Wagner (1967) for shadowing, Stockhausen (1963)
for the volume term, Hamilton (1972) and Hamilton and Bachmann (1982) for bulk geoacoustics. It does NOT cite
Williams and Jackson (1998) or D. Jackson and Briggs (1992) (later papers; another agent's notes).

## 2. What the paper gives

### 2.1 Conventions and stated validity (confirmed)

| Item | [MJ89] | Ours |
|---|---|---|
| Angle | grazing theta (9) | grazing |
| Subscripts | 1 water, 2 sediment (p. 1170) | same ratios |
| Scattering strength | S_b = 10 log10(sigma_r + sigma_v) (13), cross sections per unit area per unit solid angle, dimensionless; Urick's definition | same |
| Relief spectrum | 2-D, W(K) = w2 (h0 K)^(-gamma) (14), h0 = 1 cm, w2 in cm^4 ("the only model parameter that is not dimensionless", p. 1170), K in rad/cm (the spectral strength is defined "at wavenumber 2 pi / lambda = 1 cm^-1", Table 1) | `w2_from_cm4`: w2 [m^(4-gamma2)] = w2[cm^4] 10^(2 gamma2 - 8), bottom.py:176-182; the same convention (perturbation equality below to 4e-15 dB) |
| Loss parameter | delta = k2i / k2r (3), "loss tangent"; complex k2 = k2r + i k2i | `delta`, same |
| Volume parameter | sigma2 = sediment volume scattering cross section / attenuation coefficient (Table 1); attenuation in dB/m (text after (3)) | `sigma2`, same (bottom.py:38-39) |
| Model | fluid half-space, no layering or gradients, statistically homogeneous; isotropic relief ("directional sand ripples are not important", p. 1172) | same |
| Frequency | 10-100 kHz (abstract) | used from 6 to 330 kHz |
| Sediments | "sands through clays"; relations "only defined for 1 <= Mz <= 9" (p. 1169) | -1 to 9 phi |
| Exponent | 2 < gamma < 4 (15), i.e. 0 < alpha < 1 | same check (bottom.py:395-396, 495-496) |
| Surficial versus bulk | the relations give surficial values, "generally smaller, more water-like" than Hamilton's bulk ones; measured density ratios were discarded and the density fitted to acoustic data, the loss tangent left at Hamilton's (p. 1170) | rows equal the relations (section 2.5) |
| Parameter scatter | sigma2 "sometimes takes on values differing by nearly an order of magnitude" (p. 1170); (5)-(7) are "reasonable defaults but not ... predictors" of the relief spectrum (p. 1170) | rows carry this in notes |

Table 1 of the paper lists the six inputs (rho, nu, delta, sigma2, gamma, w2) with the definitions above.

### 2.2 Grain-size relations (eqs. (1)-(7), pp. 1169-1170) - confirmed

Mz = mean grain size in phi. All are "defined for 1 <= Mz <= 9".

| Quantity | 1 <= Mz <= 5.3 (rho, nu) / 1 <= Mz < 5.0 (h, sigma2) | 5.3 <= Mz <= 9 (rho, nu) / 5.0 <= Mz <= 9 (h, sigma2) | Eq. |
|---|---|---|---|
| rho = rho2/rho1 | -0.0165406 Mz^3 + 0.2290201 Mz^2 - 1.1069031 Mz + 3.0455234 | -0.0012973 Mz + 1.1564755 | (1) |
| nu = c2/c1 | -0.0014881 Mz^3 + 0.0213937 Mz^2 - 0.1382798 Mz + 1.3424740 | -0.0024324 Mz + 1.0018916 | (2) |
| delta | k2i/k2r = alpha2 nu c1 ln(10) / (40 pi f), alpha2/f from Hamilton (1972) in dB/m/kHz with c1 = 1.528 m/ms | same | (3) |
| sigma2 | 0.002 | 0.001 | (4) |
| gamma | 3.25 for all grain sizes | 3.25 | (5) |
| h/h0 (rms relief of a 100-cm track, h0 = 1 cm) | (2.03846 - 0.26923 Mz) / (1.0 + 0.076923 Mz) | 0.5 | (6) |
| w2 (cm^4), for gamma = 3.25 | 0.00207 h^2 h0^2 | same | (7) |

- The many decimals are "for internal consistency to four decimal places" (p. 1170).
- rho and nu agree with Hamilton and Bachmann (1982) at Mz = 1; nu also at Mz = 9, rho lower there (p. 1170).
- The measured exponents (Briggs 1989; Jackson et al. 1986b) average 3.23 with standard deviation 0.44
  (p. 1170, via secondary).
- Continuity (ours, `ours.py`): rho 1.14960 / 1.14960 and nu 0.988996 / 0.988999 at 5.3 phi from the two
  pieces; h = 0.5000 from (6) at 5.0 phi, so (6) is continuous.

### 2.3 Forward loss (eqs. (8)-(12), p. 1170) - confirmed

    r0 = -20 log10 |R(theta)|                                           (8)
    R(theta) = (y - 1)/(y + 1),  y = rho sin(theta) / P(theta)          (9), (10)
    P(theta) = sqrt(kappa^2 - cos^2 theta)                              (11)
    kappa = k2/k1 = (1 + i delta)/nu                                    (12)

A lossy Rayleigh coefficient; the paper traces it to Mackenzie (1959) quoting Morse (via secondary). It is
based on reflection, not scattering, because the data showed little pulse elongation (p. 1170). The branch of
the square root is not printed; ours takes the one that decays into the sediment.

### 2.4 Backscatter (eqs. (13)-(39), pp. 1171-1173) - confirmed

Roughness term: Kirchhoff near 90 degrees, composite roughness elsewhere, joined by (33).

| Eq. | Formula (our notation; theta grazing, k = 2 pi f / c1) | Notes |
|---|---|---|
| (14) | W(K) = w2 (h0 K)^(-gamma) | 2-D; isotropic |
| (16) | alpha = gamma/2 - 1 | |
| (17) | C_h^2 = 2 pi w2 Gamma(2 - alpha) 2^(-2 alpha) / (h0^gamma alpha (1 - alpha) Gamma(1 + alpha)) | structure-function constant (Yaglom 1962) |
| (18) | s^2 = (2 pi w2 h0^-gamma)^(1/alpha) / (2 (1 - alpha)) [k^2 / alpha]^((1 - alpha)/alpha) | "large-scale rms bottom slope" s (total of both axes, section 0 item 4) |
| (19) | sigma_kr = b q_c abs(R(90))^2 / (8 pi [cos^(4 alpha) theta + a q_c^2 sin^4 theta]^((1 + alpha)/(2 alpha))) | Kirchhoff approximation; exact for the pressure-release surface at alpha = 1/2 and at 90 degrees, generalised with two parameters fixed by the 90-degree value and by the omnidirectional (area-integrated) Kirchhoff intensity |
| (20) | q_c = C_h^2 2^(1 - 2 alpha) k^(2 (1 - alpha)) | |
| (21) | a = [8 alpha^2 Gamma(1/(2 alpha) + 1/2) / (Gamma(1/2) Gamma(1/alpha) Gamma(1/(2 alpha)))]^(2 alpha) | a = 1 at alpha = 1/2 |
| (22) | b = a^(1/2 + 1/(2 alpha)) Gamma(1/alpha) / (2 alpha) | b = 1 at alpha = 1/2 |
| (23) | sigma_cr = S(theta, s) F(theta, sigma_pr, s) | composite: shadowing times slope average |
| (24) | S(theta, s) = (1 - exp(-2Q)) / (2Q) | "following Wagner (1967)" |
| (25) | Q = [pi^(-1/2) exp(-t^2) - t (1 - erf t)] / (4 t) | pi to the power **-1/2** |
| (26) | t = tan(theta) / s | set S = 1 for t > 2 "to avoid numerical difficulties" (p. 1173) |
| (27) | F = pi^(-1/2) sum_{n=-1..1} a_n sigma_pr(theta - theta_n) | three-point Gauss-Hermite quadrature of the slope average (McDaniel and Gorman 1982; Carnahan et al. 1969) |
| (28) | a_-1 = a_1 = 0.295410, a_0 = 1.181636 | the Gauss-Hermite weights sqrt(pi)/6 and 2 sqrt(pi)/3 (ours: 0.2954090, 1.1816359) |
| (29) | theta_-1 = -theta_1 = 1.224745 arctan(s), theta_0 = 0 | node sqrt(3/2) = 1.2247449 (ours); in-plane slope only |
| text | an argument theta - theta_n <= 0 is set to 1 degree; above 90 degrees to 90 degrees | "nearly equivalent" to dropping shadowed facets |
| (30) | sigma_pr = 4 k^4 sin^4 theta abs(Y)^2 W(K_theta) | first-order perturbation, "a modified version of Kuo's expression (Kuo, 1964)" |
| (31) | Y = [(rho - 1)^2 cos^2 theta + rho^2 - kappa^2] / [rho sin theta + P(theta)]^2 | P, kappa from (11)-(12) |
| (32) | K_theta = [4 k^2 cos^2 theta + (k/10)^2]^(1/2) | the (k/10)^2 term is a deliberate modification against the power-law singularity at K = 0; "only affects ... angles near 90 degrees" |
| (33) | sigma_r = f(x) sigma_kr + [1 - f(x)] sigma_cr | interpolation |
| (34) | f(x) = 1 / (1 + e^x) | |
| (35) | x = 80 [cos theta - cos theta_kdB] | |
| (36) | cos theta_kdB = (1/C4 + 4)^(-1/4) | |
| (37) | C4 = 1000^(1/(1 + alpha)) (a q_c^2)^(1/alpha) | theta_kdB is where the Kirchhoff result "has fallen 15 dB below its peak at 90 degrees" |
| (38) | sigma_v = S(theta, s) F(theta, sigma_pv, s) | volume term, slope averaged and shadowed "in analogy with" (23) |
| (39) | sigma_pv = 5 delta sigma2 abs(1 - R(theta))^2 sin^2 theta / (nu ln(10) abs(P)^2 Im P) **as printed**; abs(1 - R^2(theta))^2 meant (section 0 item 1) | Stockhausen (1963) generalised for absorption in the transmission coefficient |

Checks (ours, `ours.py`, `figs.py`):
- (30)-(31) with K_theta = 2k cos theta equals our `perturbation_strength` in backscatter to 4e-15 dB (medium
  sand, very fine silt, 50 kHz, 5-85 degrees). The (k/10)^2 term changes it by -0.02 dB at 5-45 degrees,
  -0.26 dB at 75 and -2.0 dB at 85 degrees.
- (19)-(22) at alpha = 1/2: equal to our exact `kirchhoff_strength` at every angle (0.0000 dB); at 90 degrees
  equal for every alpha (algebra, ours). In backscatter the facet is always at normal incidence, so
  abs(R(90))^2 is exact there.
- (36)-(37): with cos theta_kdB small, cos^4 theta_kdB -> C4 is exactly the 15 dB point of (19); the "+4"
  keeps theta_kdB >= 45 degrees. Evaluated: the Kirchhoff result at theta_kdB is -13.2 to -15.4 dB below its
  peak (medium sand, very fine silt, sandy gravel, 20 and 100 kHz) except sandy gravel 100 kHz, where the 45
  degree floor holds and it is -4.4 dB. The 10-90 % width of f is 3.2-4.4 degrees.
- (25)/(26) are Wagner's B and v (`shadowing_wagner_1967.md`, Sec. 2.2); (24) is Wagner's R(theta) of his
  eq. (49), the backscatter shadow function, not his S(theta) of eq. (20).
- (39) with abs(1 - R^2)^2: abs(1 - R^2)^2 sin^2 theta / abs(P)^2 = abs(1 + R)^4 / rho^2 (algebra, ours),
  which is J&R (14.8) and our `volume_strength`; unaveraged, equal to ours to 2e-14 dB.

### 2.5 Site parameters printed in the figure insets (confirmed: printed numbers in the figures)

"surface" = from the grain-size relations; "bulk" = measured, as the paper labels them. Spectral strength in
cm^4. Fig. 3-5 are forward loss, Figs. 6-10 backscatter.

| Site, frequency | Fig. | Bulk Mz | rho surface/bulk | nu surface/bulk | delta surface/bulk | sigma2 surface/bulk | gamma surface/bulk | w2 surface/bulk |
|---|---|---|---|---|---|---|---|---|
| Quinault, 35 kHz (25-30 kHz forward) | 4, 6 | 3.0 (porosity 41.2 %) | 1.339 (1.3394) / 1.93 | 1.080 / 1.080 | 0.01727 / 0.02601 | 0.002 / 0.001 | 3.25 / 3.67 | 0.002070 / 0.004225 |
| Puget Sound 1, 25 kHz (30 kHz forward) | 3, 7 | 4.83 (56 %) | 1.178 (1.1782) / 1.74 | 1.006 / 1.006 | 0.01531 / 0.01007 | 0.002 / 0.002 | 3.25 / 3.5 | 0.000599 / 0.0046 |
| Arafura Sea, 20 kHz | 5, 8 | 5.3 (69.7 %) | 1.150 / 1.49 | 0.989 / 0.989 | 0.00871 / 0.01327 | 0.002 / 0.005 | 3.25 / 3.18 | 0.000391 / 0.002166 |
| Stanic et al. 1988, 40 kHz | 9 | 2.59 | 1.428 / 1.966 | 1.1020 / 1.133 | 0.01601 / 0.01856 | 0.002 / 0.004 | 3.25 / 2.92 | 0.002589 / 0.0061636 |
| Stanic et al. 1989, 20 kHz | 10 | 0.84 | 2.176 / 1.993 | 1.2323 / 1.113 | 0.01643 / 0.04534 | 0.002 / 0.005 | 3.25 / 2.47 | 0.005999 / 0.005874 |

Recomputed (ours, `ours.py` section 9): rho, nu from (1)-(2) and delta from J&R (11.3) with c1 = 1528 m/s
give every surface value to its printed digits (Quinault 1.3394, 1.0800, 0.01728; Puget 1.1782, 1.0060,
0.01531; Arafura 1.1496, 0.9890, 0.00871; Stanic 1988 1.4276, 1.1020, 0.01602), and (6)-(7) give w2 for
Quinault (0.002070), Puget (0.000599), Stanic 1988 (0.002589) and Stanic 1989 (0.005999). Exceptions:
- Stanic 1989 at 0.84 phi, below the paper's stated range: (1)-(2) extrapolated give 2.2675 and 1.2405;
  the printed 2.176 and 1.2323 equal J&R's -1 <= Mz < 1 pieces of (11.1)-(11.2) (2.1761, 1.2323). The pieces
  were in use in 1989 although not printed here (ours, inference).
- Arafura at 5.3 phi: sigma2 0.002 and w2 0.000391 contradict (4) and (6) as printed (0.001 and 0.000518);
  0.000391 is (6)'s rational form evaluated at 5.3 (0.0003906). An internal inconsistency.
- The volume parameters "were not measured"; the bulk values are those the original investigators recommend,
  and 0.005 was fitted at Arafura, 15-45 kHz (p. 1173-1174, confirmed).

### 2.6 Statements (confirmed)

- The upper few centimetres control high-frequency interaction; gradients mainly in density (p. 1168-1169).
  Fig. 1 is a "hypothetical (and upside down)" density profile; Fig. 2 its normal-incidence reflection
  coefficient against frequency for a lossless fluid of constant speed (section 3.1).
- Soft sediments fit "substantially" better with the density ratio reduced below its measured value
  (abstract), suggesting a tenuous upper layer. This is the reason the relations' rho is surficial (2.1).
- Forward loss: Arafura (Fig. 5) is not reproduced by either parameter set; the authors think vertical
  gradients must be modelled there (p. 1171). Forward-loss data scatter about 2 dB (p. 1171).
- Backscatter: at Quinault the measured anisotropic spectrum fits "somewhat better"; the hard bottom makes the
  result insensitive to sigma2; at Puget Sound the fit is sensitive to sigma2 and to delta, and including loss
  removes an artefact (a sharp downturn below the critical angle); at Arafura sigma2 = 0.005 describes
  15-45 kHz; at Stanic 1988 a small bump at 60-70 degrees is from the interpolation; at Stanic 1989 (shell
  bottom) the model is "nearly identical" to Jackson et al. 1986 and the measured roughness fits better
  (pp. 1173-1174).

## 3. Figures

### 3.1 Figs. 1-2 (hypothetical profile; figure)

- Fig. 1: water above 1 cm; density ratio about 1.07 at the interface rising to about 2.0 by 5.5-6 cm, then
  constant (reading +-0.05 in rho, +-0.3 cm). The sound speed used for Fig. 2 is not printed.
- Fig. 2: normal-incidence |R| about 0.27 near 0 kHz, 0.20 at 15 kHz, 0.15 at 20 kHz, a minimum about 0.01
  at 32 kHz, a bump about 0.055 at 43 kHz, then 0.03-0.04 above 60 kHz (reading +-0.01 in |R|, +-2 kHz). The
  drop from 0 to 20 kHz is about -5 dB (ours, from the readings).

### 3.2 Backscatter, Figs. 6-10 (figure; solid = surface parameters, dashed = bulk)

Readings +-1.5 dB (at 90 degrees +-2 dB). "Model" = the paper's model as retyped (`mj89.py`), c1 = 1528 m/s,
with (39) as printed and with abs(1 - R^2)^2.

| Fig., curve | 10 deg | 20 | 30 | 45 | 60 | 70 | 80 | 90 | rms / max (squared R) |
|---|---|---|---|---|---|---|---|---|---|
| 6 Quinault surface: read | -36 | -28.5 | -25.9 | -24.6 | -23.5 | -22.5 | -17 | -5 | |
| model, squared R / as printed | -36.8 / -37.7 | -27.8 / -31.1 | -25.4 / -27.8 | -24.3 / -26.0 | -23.4 / -24.8 | -22.3 / -23.4 | -17.6 / -17.9 | -4.7 | 0.48 / 0.82 |
| 7 Puget surface: read | -32.5 | -27.5 | -26 | -24.8 | -24 | - | -22.5 | -2 | |
| model | -31.6 / -32.2 | -28.2 / -29.0 | -26.6 / -27.4 | -25.1 / -25.8 | -24.2 / -24.9 | - | -23.2 / -23.8 | -1.0 | 0.68 / 0.96 |
| 8 Arafura surface: read (5 deg: -38) | -31 | -28 | -26.5 | -25 | -24 | - | -23.5 | -0.5 | |
| model (5 deg: -38.4 / -35.4) | -32.6 / -31.9 | -28.7 / -28.9 | -26.9 / -27.3 | -25.3 / -25.8 | -24.4 / -24.9 | - | -23.7 / -24.2 | +0.5 | 0.77 / 1.65 |
| 9 Stanic 1988 surface: read | -36 | -28.5 | -25 | -24 | -22.7 | -21 | -14 | -5 | |
| model | -37.2 / -37.9 | -28.7 / -31.3 | -25.2 / -27.9 | -24.1 / -26.0 | -22.9 / -24.4 | -21.3 / -22.3 | -15.4 / -15.6 | -5.2 | 0.68 / 1.39 |
| 10 Stanic 1989 surface: read | -34.5 | -27.5 | -24.5 | -22.5 | -20 | -15.5 | -9 | -1 | |
| model | -34.1 / -34.3 | -27.6 / -27.9 | -24.5 / -25.0 | -22.5 / -24.1 | -20.2 / -21.2 | -16.3 / -16.7 | -9.1 | -1.2 | 0.34 / 0.83 |
| 6 Quinault bulk: read | -33 | -29.5 | -28 | -25.5 | -21 | -15.5 | -10 | -9.5 | |
| model | -33.1 / -34.1 | -29.1 / -30.4 | -27.8 / -29.3 | -25.8 / -27.1 | -21.2 / -21.6 | -16.5 / -16.7 | -10.7 | -9.2 | 0.49 / 1.02 |
| 8 Arafura bulk: read | -28 | -25.5 | -23.5 | -21.5 | -20.5 | - | -18 | 0 | |
| model | -28.7 / -29.0 | -24.9 / -26.1 | -23.1 / -24.5 | -21.5 / -23.0 | -20.6 / -22.0 | - | -17.6 / -18.4 | -0.1 | 0.40 / 0.68 |

These figures cannot tell the large-scale slope (18) from J&R's (13.77) or shadowing from none (rms changes
under 0.1 dB, `variants.py`): the slopes are small (0.010-0.086 for the surface sets) and no reading is below
5 degrees.

### 3.3 Forward loss, Figs. 3-5 (figure) against our `reflection`

Readings +-1 dB; ours = -20 log10 |`bottom.reflection`| with the inset parameters (`fwd.py`).

| Fig., curve | readings / ours (dB) |
|---|---|
| 3 Puget surface | 10 deg 12.5 / 11.4; 20: 19.0 / 18.3; 30: 20.3 / 20.2; 60: 21.2 / 21.3; 90: 21.5 / 21.4 |
| 3 Puget bulk | 10: 9.5 / 8.8; 30: 10.8 / 11.0; 90: 11.3 / 11.3 |
| 4 Quinault surface | 10: 1.2 / 1.0; 20: 3.0 / 2.7; 30: 9.5 / 9.3; 60: 14.3 / 14.1; 90: 14.9 / 14.8 |
| 4 Quinault bulk | 20: 2.8 / 2.7; 30: 5.5 / 6.2; 60: 9.0 / 8.8; 90: 9.2 / 9.1 |
| 5 Arafura surface | 10: 19.0 / 18.7; peak 28.0 at 19 deg / 28.0 at 19.1 deg; 40: 25.0 / 24.8; 90: 24.0 / 23.8 |
| 5 Arafura bulk | peak 18.0 at 11 deg / 18.2 at 10.9 deg; 30: 15.0 / 15.0; 90: 14.5 / 14.4 |

## 4. Against our code

### 4.1 Row by row (worst first)

| # | [MJ89] | Ours today (file:line) | Verdict, with numbers (ours = computed, `ours.py`) |
|---|---|---|---|
| D1 | Volume term slope-averaged and shadowed (38) | `volume_strength` unaveraged, bottom.py:596-608 | DIFFERS at low grazing: [MJ89]'s volume minus ours at 5 / 10 / 20 deg: medium sand 20 kHz +3.9 / +0.9 / +0.2; 100 kHz +9.6 / +3.4 / +2.3; sandy gravel 20 kHz +7.9 / +2.2 / +0.3; 100 kHz +12.2 / +6.0 / +6.4; very fine silt +0.04 to +0.3 at 5 deg; within 0.6 dB from 45 deg (except sandy gravel 100 kHz). J&R Sect. 14.1.2 describes the same correction without a formula (G-16) |
| D2 | Roughness: composite with shadowing (23)-(29) joined to Kirchhoff (33)-(37) | default `small_slope`, bottom.py:477-487, 548 | DIFFERS at low grazing (the model choice of D-86): roughness [MJ89] minus ours at 5 / 10 deg: medium sand 20 kHz +3.8 / +0.8, 100 kHz +9.3 / +2.9; sandy gravel 100 kHz +11.5 / +4.4; muddy sand 100 kHz +5.9 / +1.5. 20-85 deg: within -1.6 / +1.0 dB on the total (section 4.3) |
| D3 | Composite carries Wagner's S (24)-(26) | `composite_strength`, `combined_strength` without it, bottom.py:518-543 | DIFFERS (both [MJ89] and J&R (13.80)-(13.83) print it). Correct factor (pi^(-1/2), section 0): 10 log R on our composite at 5 deg: medium sand -0.05 / -0.28 / -0.68 dB at 20 / 50 / 100 kHz; sandy gravel -0.55 / -1.38 / -2.39; very fine silt 0.00; at 2 deg up to -5.8 dB (sandy gravel 100 kHz). Exposure 0 shipped |
| D4 | Large-scale slope (18): total slope with k h_s = 1/sqrt(2) | `large_scale_slope` per axis with k h_s = 1/2 (J&R (13.72)-(13.77)), bottom.py:490-499 | DIFFERS by convention: [MJ89] s = 0.81 x our total sqrt(2) x per-axis (gamma2 3.25; medium sand 100 kHz 0.180 against 0.221; sandy gravel 100 kHz 0.422 against 0.519). Ours follows J&R; no change |
| D5 | Composite average: 1-D in-plane, 3-point Gauss-Hermite, full spectrum W(K_theta) with the (k/10)^2 term, arguments clamped to 1 deg | 2-D 12 x 12 Gauss-Hermite on (u, v) with atan, facets with a negative local angle dropped, spectrum truncated below K_c, bottom.py:502-530 | DIFFERS in recipe. Composite-plus-volume, ours minus [MJ89] total at 5 deg: medium sand +0.7 / +1.3 / +1.7 dB at 20 / 50 / 100 kHz; sandy gravel +1.7 / +2.5 / +3.2 (shadowing, the smaller s of (18), the averaged volume and the recipe together). Three points are adequate up to s about 0.2 (40-point quadrature within 0.1 dB, `decomp.py`), 0.5-2.4 dB off at s = 0.42 |
| D6 | Kirchhoff closed form (19)-(22) | exact Kirchhoff integral, bottom.py:405-474 | DIFFERS (theirs is the approximation): [MJ89] minus ours at 60 / 70 / 80 deg, 20 kHz: gamma2 3.25 +0.59 / +0.28 / -0.29; 3.5 +2.08 / +1.61 / 0.00; 3.67 +3.73 / +3.18 / +0.52; 3.9 +8.44 / +7.01 / -1.04; gamma2 2.5 -0.48 / -0.59 / -0.42; exactly 0 at gamma2 = 3.0 and at 90 deg. Our exact form stays |
| D7 | Interpolation (33)-(37) | none: `small_slope` is the all-angle expression (D-66, D-86); `combined` is ours | ABSENT. Printed now (was "wanted"). Opt-in model C4 |
| D8 | sigma2 step at 5.0 phi (4) | 5.5 phi (`grain_size_sigma2`, bottom.py:682-685; sediments.yaml:66-67; `sandy_silt` sediments.yaml:607-627 sigma2 0.002) | DIFFERS for `sandy_silt` only: 0.001 by (4), 0.002 ours (-3.0 dB on its volume term). Kept (section 0 item 5) |
| D9 | rho, nu constants to 7 digits (1)-(2) | J&R's rounded digits, bottom.py:752-756 | DIFFERS by +2.3e-5 to -2.8e-5 in rho and -2.6e-5 to -8e-6 in nu (1-9 phi); under 0.001 dB in reflection. Optional (C5) |
| A1 | (11.2) last piece 1.0018916 | 1.0019, bottom.py:756 | MATCHES (the printed J&R 1.0091 is a misprint, now confirmed) |
| A2 | delta from Hamilton's alpha/f with c1 = 1.528 m/ms (3) | `GRAIN_SIZE_WATER_SOUND_SPEED` = 1528, bottom.py:712-770 | MATCHES; the five printed inset deltas reproduced to the printed digits |
| A3 | h rule (6), w2 = 0.00207 h^2 (7), gamma 3.25 (5) | `grain_size_w2_cm4`, `GRAIN_SIZE_GAMMA2`, bottom.py:661-679 (Snellen 2013, via secondary) | MATCHES (now confirmed in the original); range here 1 <= Mz <= 9, ours -1 to 9 |
| A4 | Spectrum convention, h0 = 1 cm, cm^4 (14), Table 1 | `w2_from_cm4`, bottom.py:176-182 | MATCHES (perturbation equality to 4e-15 dB) |
| A5 | Structure constant (17) | `structure_constant`, bottom.py:392-398 | MATCHES (Gamma(2 - alpha) = (1 - alpha) Gamma(1 - alpha); h0^gamma is the cm convention) |
| A6 | Perturbation (30)-(31) without the (k/10)^2 term | `_perturbation_factor`, `perturbation_strength`, bottom.py:356-389 | MATCHES to 4e-15 dB |
| A7 | Flat-interface reflection (8)-(12) | `reflection`, bottom.py:330-353 | MATCHES: Figs. 3-5 reproduced within the readings (section 3.3) |
| A8 | Volume (39) corrected, unaveraged | `volume_strength`, bottom.py:596-608 | MATCHES to 2e-14 dB |
| A9 | Kirchhoff at 90 deg and at alpha = 1/2 | `kirchhoff_strength` | MATCHES exactly (D6 is elsewhere) |
| S1 | Shadowing for surface terms | `surface.facet_backscatter` (surface.py:568-576) has none | not in [MJ89] (seafloor only); see `shadowing_wagner_1967.md` Sec. 4: no effect on our facet term |

### 4.2 Shipped exposure

All `fluid_sediment` items in shipped scenarios use the default (`small_slope`) model (CONFORMANCE section 2.3);
no shipped item names `composite` or `combined`; no scenario names `sandy_silt`; `sediment_from_grain_size` has
no caller in the code (CONFORMANCE section 2.5). Nothing shipped moves under C1-C5.

### 4.3 Full model against ours (computed, c = 1500 m/s, backscatter, dB)

[MJ89] = the paper's model with (39) corrected. "d-def" = [MJ89] minus our default (`small_slope` + volume);
"d-comp" = [MJ89] minus our `composite` + volume. Angles 5 / 10 / 20 / 30 / 45 / 60 / 75 / 85 degrees.

| Sediment, f | [MJ89] | d-def | d-comp |
|---|---|---|---|
| medium_sand 20 kHz | -44.8 -37.3 -30.0 -26.7 -25.5 -23.3 -16.1 -4.9 | +3.8 +0.8 0.0 -0.2 -0.1 -0.1 0.0 -0.2 | -0.7 -0.2 0.0 +0.2 0.0 -0.1 -1.4 +5.0 |
| medium_sand 50 kHz | -39.1 -33.6 -27.3 -24.5 -23.9 -21.0 -14.3 -7.0 | +6.8 +1.8 +0.1 -0.5 -0.1 0.0 -0.5 +0.1 | -1.3 -0.5 +0.1 +0.4 -0.1 -0.4 -4.0 +2.3 |
| medium_sand 100 kHz | -34.5 -30.3 -25.2 -22.7 -22.1 -20.0 -13.8 -9.8 | +9.3 +3.0 -0.1 -0.8 +0.1 -0.8 -0.4 +0.1 | -1.7 -0.7 +0.1 +0.7 -0.3 -2.9 -2.1 +10.5 |
| muddy_sand 20 kHz | -47.0 -37.4 -28.0 -25.5 -24.4 -23.5 -21.7 -11.3 | +1.6 +0.3 0.0 0.0 0.0 0.0 -0.1 -0.3 | +1.2 +0.3 +0.1 0.0 0.0 0.0 -0.1 -3.3 |
| muddy_sand 100 kHz | -41.3 -34.7 -27.4 -25.2 -24.1 -22.9 -19.3 -11.9 | +6.1 +1.8 -0.1 0.0 0.0 0.0 -0.2 +0.1 | +2.1 +1.1 +0.1 -0.1 0.0 -0.1 -1.4 +3.2 |
| sandy_silt 20 kHz | -37.4 -31.8 -28.4 -26.7 -25.2 -24.3 -23.7 -21.9 | 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.9 | as d-def |
| very_fine_silt 20 kHz | -42.0 -36.1 -32.0 -30.0 -28.4 -27.5 -26.9 -24.7 | 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.0 | as d-def |
| very_fine_silt 100 kHz | -41.7 -36.1 -31.9 -30.0 -28.4 -27.5 -26.6 -20.6 | +0.3 0.0 0.0 0.0 0.0 0.0 0.0 -0.3 | +0.3 0.0 ... -1.8 |
| clay 100 kHz | -42.3 -36.4 -32.0 -30.1 -28.4 -27.4 -26.6 -20.9 | +0.4 0.0 0.0 0.0 0.0 0.0 0.0 -0.3 | +0.4 0.0 ... -1.7 |
| sandy_gravel 20 kHz (outside 1-9 phi) | -33.5 -29.0 -23.7 -20.9 -18.9 -17.2 -10.6 -5.8 | +7.8 +2.1 +0.1 -0.2 -0.3 -1.1 -0.6 +0.1 | -1.7 -0.7 -0.1 +0.1 -0.3 -2.3 -1.3 +11.1 |
| sandy_gravel 100 kHz (outside; s = 0.42) | -24.6 -21.5 -18.3 -15.7 -12.9 -15.1 -13.5 -13.3 | +11.5 +4.4 +0.1 -0.5 +1.0 -1.5 -0.2 0.0 | -3.2 -1.7 -0.4 +0.4 +2.5 +1.8 +0.2 +5.5 |
| rock, pebbles | not covered (not fluids; outside 1-9 phi) | - | - |

Sandy silt (5.0 phi) is computed with OUR sigma2 0.002 on both sides; with (4)'s 0.001 the [MJ89] column is
3.0 dB lower wherever volume dominates (all angles off specular).

Decomposition (`decomp.py`): at 5 degrees the roughness and the volume differences are nearly equal (e.g.
medium sand 100 kHz +9.3 and +9.6 dB): the low-angle excess is the slope average acting on two steeply rising
functions, not a difference in either term's flat-interface form (A6, A8).

## 5. Corrections to apply (ordered; an implementation agent can execute these)

Owner's standing instruction: a printed reference wins over ours; a figure reading or an agent's derivation does
not override a printed class value; where no source exists nothing changes. None of C1-C5 moves a shipped
scenario (4.2). There is no C++ twin of the bottom scattering model (grep of `libs/`: `penetration.cpp` has its
own perturbation term for transmission, untouched here); the live chain does not use it.

**C1. Wagner shadowing in `composite` and `combined` (default ON inside those two opt-in models).**
- Change: `composite_strength`, `combined_strength` (bottom.py:518-543) multiply the slope average by
  R(theta) = (1 - exp(-2Q)) / (2Q), Q = [pi^(-1/2) exp(-t^2) - t erfc(t)] / (4t), t = tan(theta) /
  (sqrt(2) sigma_sL), sigma_sL = the per-axis slope `large_scale_slope` returns (J&R (13.81)-(13.83) with
  pi^(-1/2); [MJ89] (24)-(26) with s = sqrt(2) sigma_sL; Wagner (49), (20), (22)). R = 1 for t > 2 (the
  [MJ89] rule; error under 1.3e-4 = 0.0005 dB) or a guarded evaluation. Keyword `shadowing="wagner" | "none"`,
  default "wagner"; "none" reproduces today bit for bit. Bistatic (our composite is bistatic): see
  `shadowing_wagner_1967.md` C-W2 for the symmetric form; backscatter uses R(theta).
- Why default inside the opt-in models: both printed composite models include it ([MJ89] (23), J&R (13.80));
  ours omitted it only because the factor was unsettled (CONFORMANCE A13).
- Tests first (`tests/python/test_bottom_models.py`): R(t) = 0.385059, 0.719744, 0.906503, 0.987541,
  0.998564 at t = 0.1, 0.25, 0.5, 1, 1.5 (tolerance 1e-6); J&R's own table with pi^(-1/2) reproduced:
  10 log R = -0.44 / -1.45 / -3.37 / -5.59 dB at 2 deg for sigma_sL 0.05 / 0.1 / 0.2 / 0.35 and -0.26 dB at 5
  deg for 0.1 (+-0.01 dB); R -> 2 sqrt(pi) t as t -> 0 (ratio within 2 % at t = 0.01); `shadowing="none"` equal
  to today to 0 dB; reciprocity test still passes (symmetric bistatic form). Fails before: the R values (no
  function), and the composite at 5 deg below.
- Moves (opt-in models only; backscatter roughness term): medium sand 30 kHz 5 deg -41.58 -> -41.69 dB;
  100 kHz 5 deg -32.88 -> -33.56, 10 deg -29.70 -> -29.82; sandy gravel 30 kHz 5 deg -29.03 -> -29.89;
  100 kHz 5 / 10 / 20 deg -21.37 -> -23.75 / -19.85 -> -20.73 / -18.00 -> -18.18; `combined` the same
  amounts; very fine silt 0.00. **These replace G-11's "before -> after" (-41.6 -> -42.5, -32.9 -> -35.9,
  -21.4 -> -27.8), which used the misprinted pi^(+1/2).**
- Unblocks: CONFORMANCE A13 first bullet, G-11.

**C2. Status and documentation (no behaviour change).** `bottom.py` docstring lines 21-29 and 121-122 ("which
ours lacks"), `apl_grain_size_relations` docstring 741-746 (1.0091: now confirmed misprint by [MJ89] (2)
1.0018916; 0.4566 still unsettled: [MJ89] prints no k(Mz)), sediments.yaml header lines 54-66 (add the
original print of the relief rule and gamma: [MJ89] (5)-(7), confirmed; the sigma2 step's 5.0 vs 5.5 note).
REFERENCES rows: final report.

**C3. Pin the volume-term form against the corrected (39) (test only; passes before and after).**
`tests/python/test_jackson_2007_bottom.py` or a new `tests/python/test_mourad_jackson_1989.py`:
`volume_strength(theta, theta, ...)` equals 5 delta sigma2 abs(1 - V^2)^2 sin^2 theta / (nu ln10 abs(P)^2
Im P) to 1e-12 relative, at medium sand and very fine silt, 5-85 deg; and the printed abs(1 - V)^2 form is
ours times abs(1 + V)^-2 exactly (unaveraged). After the paper's slope average the printed form is 2.7-4.9 dB
below the corrected one for medium sand at 20 kHz, 20-85 deg (`ours.py` section 7).
Records the misprint so nobody "fixes" ours to the printed form.

**C4. The [MJ89] model as an opt-in (`roughness_model="mourad_jackson_1989"`, `volume_model="mourad_jackson_1989"`).**
- Implements, as printed: (13), (16)-(37) for the roughness and (38) with corrected (39) for the volume, with
  the paper's three-point quadrature (27)-(29), the 1 deg / 90 deg clamps, the (k/10)^2 term (32), S = 1 for
  t > 2, s from (18) (k h_s = 1/sqrt(2), total slope), Kirchhoff by (19)-(22) (NOT our exact integral: the
  model is the paper's; our exact Kirchhoff remains the `kirchhoff` model). Backscatter only (the paper is
  monostatic): for theta_in != theta_out raise, or fall back explicitly to the default with a warning; never
  silently.
- Files: `python/gpusonar/sim/bottom.py` (new functions; `ROUGHNESS_MODELS`; `volume_strength` gains
  `volume_model="stockhausen" | "mourad_jackson_1989"`, default unchanged), `python/gpusonar/sim/simulator.py`
  / `scene.py` / `targets.py` only if they must pass `volume_model` (they pass `**params` today; check),
  `sediment_fit.py` unchanged. Python only (no C++ twin).
- Tests first (`tests/python/test_mourad_jackson_1989.py`): (a) the five figure cases of section 3.2 with the
  inset surface parameters and c = 1528 m/s: within 1.5 dB of every reading in the table (worst today 1.65 at
  Arafura 10 deg: use 2.0 dB there, or drop that point and say so), rms <= 1.0 dB per curve; (b) at
  gamma2 = 3.0 its Kirchhoff part equals `kirchhoff_strength` to 1e-6 dB at 10-90 deg; (c) its unaveraged
  perturbation part with K = 2k cos theta equals `perturbation_strength` to 1e-10 dB; (d) a_n, theta_n equal
  `numpy.polynomial.hermite.hermgauss(3)` to 1e-6; (e) theta_kdB reproduces the 15 dB drop to 0.5 dB where it
  is above 45 deg; (f) s from (18) equals sqrt(2) x `large_scale_slope(..., small_scale_kh=2**-0.5)` to 1e-12.
- Default: unchanged (two printed models; J&R 2007, later, recommend the small-slope one). What it would move
  if ever made default: section 4.3, worst +11.5 dB at 5 deg (sandy gravel 100 kHz), +9.3 dB medium sand
  100 kHz; owner first.
- Unblocks: REFERENCES Sec. 10 "Interpolation between Kirchhoff and composite"; CONFORMANCE rows "Interpolation
  Kirchhoff/composite" and "combined" (a printed alternative now exists); G-16's numbers.

**C5. Grain-size constants to the original's digits (default; negligible).** `apl_grain_size_relations`
(bottom.py:752-756): 3.0455 -> 3.0455234, 1.3425 -> 1.3424740, 1.1565 -> 1.1564755, 1.0019 -> 1.0018916
([MJ89] (1)-(2), confirmed; J&R print rounded digits of the same relations). Test that fails before:
`apl_grain_size_relations(3.0)["rho"] == 1.339399` and `["nu"] == 1.079999` to 1e-6 (today 1.339375,
1.080025). Moves rho by up to 2.8e-5 and nu by up to 2.6e-5 for 1-9 phi; the -1 <= Mz < 1 pieces unchanged
(not in [MJ89]); `test_the_relations_reproduce_every_tabulated_row` (5e-4) and the continuity test (1e-4)
still pass. No named row changes (they come from the tables). Optional: if the owner prefers the J&R digits,
nothing is lost.

**Not changed (with reason):**
- sigma2 step 5.0 phi (D8): the later prints of the same model (Snellen 2011 Table II, 2013; TR 9407 via
  secondary) and [MJ89]'s own Fig. 8 use 0.002 above 5.0 phi; `sandy_silt` keeps 0.002. Record in its
  provenance.
- Large-scale slope convention (D4) for our `composite`: J&R (13.72)-(13.77) are the later print; keep.
- Exact Kirchhoff (D6): ours is exact; the paper's form is an approximation for speed.
- Default model (D2): small-slope (D-86) stays.

**Still blocked:** the large-roughness law for gravel and rock (not in [MJ89]: its range is 1-9 phi); a
bistatic model (not in [MJ89]; Williams and Jackson 1998, another agent); whether TR 9407 changed (19), (33)-(37)
or (4) (never read); the -1 <= Mz < 0 constant of (11.3) (0.4566 / 0.4556; not printed here); gradients and
layering (the paper's own conclusion; Fig. 1 is hypothetical, no profile law).

## 6. What this closes or changes

| Item | Effect |
|---|---|
| REFERENCES Sec. 10 "Interpolation between Kirchhoff and composite results; ... empirical expression" | interpolation: found ([MJ89] (33)-(37), confirmed); empirical gravel/rock law: still wanted (not in [MJ89]) |
| REFERENCES Sec. 10 "Regressions of density, speed and loss against grain size" | [MJ89] (1)-(3) are the first print of (11.1)-(11.2) for 1-9 phi; 1.0091 misprint confirmed |
| REFERENCES Sec. 10 "Shadowing in the composite model" | settled: pi^(-1/2) ([MJ89] (25); Wagner (22)) |
| CONFORMANCE A13 (composite shadowing blocked) | unblocked (C1) |
| CONFORMANCE "(13.82) pi^(+1/2)" (SOURCE MISPRINT, suspected) | SOURCE MISPRINT, settled |
| CONFORMANCE "Interpolation Kirchhoff/composite" (NO SOURCE READ) | printed in [MJ89]; ours (small-slope) differs by design; C4 opt-in |
| CONFORMANCE "`combined`" (NO SOURCE READ) | unchanged (ours); [MJ89] is the printed alternative |
| CONFORMANCE "Volume slope and shadow corrections" (DIFFERS, not quantified) | quantified (D1); printed recipe (38) |
| CONFORMANCE "Gamma2 3.25 for all" (MATCHES the secondary) | MATCHES the original ([MJ89] (5)) |
| CONFORMANCE "(11.2) constant 1.0091" (SOURCE MISPRINT) | confirmed by the original (1.0018916) |
| G-6 | 1.0091 settled; 0.4566 not |
| G-11 | unblocked; before -> after numbers corrected (C1) |
| G-16 | a printed recipe ((38) with the three-point average) and numbers (D1) |
| G-10 (EDFM) | [MJ89] says the relations' density is fitted to acoustic data and lower than bulk (2.1): supports G-10's "must not apply EDFM to the TR 9407 rows" (double count) |
| G-8, G-12 | nothing in [MJ89] |
| `sonar-3k1` (bottom as published) | the published 1989 model is now fully printed and reproducible (C4); slope and shadow corrections of the volume term printed (38) |

## 7. Not verified / unclear on the page

- Year and venue rest on the PDF's title field and the paper's references; the pages do not print them.
- The branch of P (11) is not printed (ours: decaying into the sediment; immaterial for the figures read).
- In (17) the h0 exponent is printed as a superscript gamma beside h0; read as h0^gamma (consistent with (14)).
- Figure readings: +-1.5 dB (backscatter), +-1 dB (forward loss), +-2 dB at 90 degrees; Fig. 1-2 readings
  rough (the sound speed of Fig. 2 is not printed).
- Whether the authors' figure code used c1 = 1528 m/s for k (the text gives c1 = 1.528 m/ms only for delta);
  the model curves were computed here with 1528 m/s, ours in section 4.3 with 1500 m/s.
- Arafura (Fig. 8) parameters contradict (4) and (6) as printed (section 2.5); which version TR 9407 adopted for
  h above 5 phi is known only through Snellen (via secondary: 0.5 cm from 5 phi).
