# Shadowing of randomly rough surfaces: Wagner (1967), extracted for the simulator

**[W67]** R. J. Wagner, "Shadowing of Randomly Rough Surfaces", J. Acoust. Soc. Am. 41 (1), 138-147 (1967);
TRW Systems, Redondo Beach, California; received 13 June 1966 (all printed on the pages). Supplied by the owner;
all rights reserved; not in the repository. 10 pages including Appendix A, read from the page images; the
equations of pp. 140-142, 144 and 147 and Figs. 2-4 and 6 re-read from 220-dpi renders.

**`docs/REFERENCES.md` cites it wrongly** (Sec. 10 row "Shadowing in the composite model": "Wagner, IEEE Trans.
Antennas Propag. (1967)"). It is the J. Acoust. Soc. Am. paper above, which is also how Mourad and Jackson
(1989) cite it (their reference list, p. 1175).

Status words as in `docs/REFERENCES.md`: **confirmed** = read on the page image of [W67]; **via secondary** =
another author's result as [W67] reports it; **ours** = our derivation or computation; **figure** = read off a
printed curve, with its error. Angles: [W67] uses the **grazing** angle theta (Fig. 1; "grazing angle" on every
figure axis), as we do.

Scripts (scratch, not committed): `.../scratchpad/ref-a/scripts/wagner.py` (the functions below, the figure
checks, the limits), `mj89.py` (the composite with shadowing), `after.py`, `surf.py` (against our code, run with
the main checkout's built module linked into this worktree's `python/gpusonar/` for the run, removed after).

## 0. What is wrong, missing or risky (read first)

1. **It settles the sign that blocked G-11 and CONFORMANCE A13.** [W67] (22) prints
   B = [exp(-v^2) - sqrt(pi) v erfc v] / (4 sqrt(pi) v), i.e. pi to the power **-1/2** in front of exp(-v^2)
   once divided through. Mourad and Jackson (1989) (25) print the same. Jackson and Richardson (2007) (13.82)
   print pi^(+1/2): a misprint (confirmed in two sources against one). J&R's own table "with pi^(-1/2)" in our
   note 04 is reproduced exactly by [W67] (ours: -0.44 / -1.45 / -3.37 / -5.59 dB at 2 degrees for per-axis
   slopes 0.05 / 0.1 / 0.2 / 0.35).
2. **Two shadow functions; the composite model needs R, not S.** [W67] gives S(theta), the fraction of the
   surface illuminated (20), and R(theta) = S(theta) / {(1/2)[1 + erf v]} (49), the factor for backscatter
   once slopes facing away from the source are excluded (Brockelman and Hagfors' definition, via secondary).
   Mourad and Jackson (24) and J&R (13.81) use R. A slope average that already drops (or clamps) back-facing
   facets must use R; using S would count those facets twice (S/R = (1 + erf v)/2, up to -3 dB at small v).
3. **Accuracy of the form everyone uses.** The B form (20)/(22) is the "simplest" approximation (independent
   crossings); [W67]'s improved form (A29)-(A32) needs the autocorrelation's 4th and 6th derivatives, which a
   power-law relief spectrum does not have finitely (ours, reasoning), so the B form is the usable one for our
   spectra. Against Brockelman and Hagfors' simulations (via secondary) the B form S (20) is above the
   simulations by 0.0-0.1 at 10-60 degrees (Fig. 2, figure) and (A29) agrees with them within the reading
   (Fig. 3). For R at sigma/T = 1 (rms slope sqrt(2) per axis) the B form gives 0.34 / 0.61 / 0.77 at 10 /
   20 / 30 degrees where the printed theoretical curve reads 0.29 / 0.47 / 0.62 (Fig. 6, figure): R from B
   overstates illumination by up to 1.1 dB at these very large slopes (ours). At the slopes of our sediments
   (per axis 0.01-0.37) R differs from 1 by under 1 dB above 10 degrees.
4. **Geometric optics only.** [W67] considers "the shadowing process alone" with diffraction ignored (p. 145).
   It is one-dimensional in the plane of incidence (surface z = zeta(x), a normal stationary process), and
   the bistatic results are for in-plane geometry only. Nothing is printed out of plane.
5. **Our code has no shadowing anywhere in the scattering models** (`bottom.composite_strength`,
   `combined_strength`, `surface.facet_backscatter`, the coherent reflection). Effect computed: bottom composite
   up to -2.4 dB at 5 degrees and -5.8 dB at 2 degrees (sandy gravel 100 kHz); surface facet term 0.000 dB
   wherever that term is within 30 dB of its peak (wind 3-20 m/s); so only the bottom composite changes (C-W1).

## 1. Scope

| Pages | Content | Read |
|---|---|---|
| 138-140 | introduction; Sec. I conditional shadowing probabilities, eqs. (1)-(24); Fig. 1, Fig. 2 | page images; (16)-(24) at 220 dpi |
| 141-142 | Sec. II bistatic shadowing, eqs. (25)-(36); Figs. 3, 4 | page images; (28)-(36) and both figures at 220 dpi |
| 142-143 | Sec. III heights of the illuminated surface, eqs. (37)-(43); Fig. 5 | page images |
| 143-144 | Sec. IV slopes of the illuminated surface, eqs. (44)-(53); Fig. 6 | page images; (46)-(53), Fig. 6 at 220 dpi |
| 145 | Sec. V conclusions; Appendix A start, (A1)-(A5) | page image |
| 146-147 | Appendix A, (A6)-(A33) | page images; (A26)-(A33) at 220 dpi |

No relation to APL-UW TR 9407 (1967 paper). It cites Beckmann (1965) and Brockelman and Hagfors (1966) (via
secondary for their results).

## 2. What the paper gives (confirmed unless marked)

### 2.1 Definitions

- Surface z = zeta(x, y), zero-mean normal stationary process, rms height sigma; shadowing is decided along the
  plane of incidence (x), so a 1-D process zeta(x) suffices (p. 139).
- psi(tau) = autocorrelation of zeta along x; psi_0 = sigma^2; psi_0'' = its second derivative at 0, so
  abs(psi_0'') is the **mean-square slope in the plane of incidence** (one component). Our notation: s_x^2 =
  abs(psi_0''). For an isotropic 2-D surface with total mean-square slope s^2 (both axes), s_x^2 = s^2 / 2 (ours).
- eta_0 = tan(theta): slope of the incident ray (p. 139).
- Shadowing function S(theta): the probability a surface point is illuminated, averaged over height and slope
  (8); under ergodicity "the fraction of all points x at which zeta(x) is illuminated" (p. 141). Incoherent
  scattered power is multiplied by S, the coherent component by S^2 (p. 141; footnote 14: away from
  backscatter the bistatic function applies).
- Conditional probability of illumination given height zeta_1 and slope zeta_1' (7):
  S(theta given zeta_1, zeta_1') = u(eta_0 - zeta_1') exp(-integral_0^inf q(tau) dtau), u the unit step, q the
  crossing probability density (4), approximated by (11) (an upper bound on q, so S from it is a lower bound,
  until (16) is used, footnote 13).

### 2.2 Monostatic shadowing (eqs. (17)-(24), p. 140)

With independent crossings (16):

| Eq. | Formula | Notes |
|---|---|---|
| (17) | S(theta given zeta_1, zeta_1') = exp[-B erfc(zeta_1 / sqrt(2 psi_0))] u(eta_0 - zeta_1') | |
| (18) | S(theta given zeta_1) = (1/2)(1 + erf v) exp[-B erfc(zeta_1 / sqrt(2 psi_0))] | |
| (19) | S(theta given zeta_1') = u(eta_0 - zeta_1') (1 - e^(-2B)) / (2B) | |
| (20) | **S(theta) = (1 + erf v)(1 - e^(-2B)) / (4B)** | the shadowing function |
| (21) | **v = eta_0 / sqrt(2 abs(psi_0''))** = tan(theta) / (sqrt(2) s_x) = tan(theta) / s for an isotropic surface (ours) | depends only on the ratio of ray slope to rms slope (p. 140) |
| (22) | **B = [exp(-v^2) - sqrt(pi) v erfc v] / (4 sqrt(pi) v)** = [pi^(-1/2) exp(-v^2) - v erfc v] / (4v) | |
| (23) | v >> 1: S ~ 1 - exp(-v^2) / (2 sqrt(pi) v) | |
| (24) | v << 1: S ~ sqrt(pi) v = (2 abs(psi_0'') / pi)^(-1/2) tan(theta) | linear in tan(theta), unlike Beckmann's exp[-(4 pi)^(-1/2) cot(theta)] (via secondary) |

Checked (ours, `wagner.py`): S / (sqrt(pi) v) = 1.003 / 1.029 / 1.21 at v = 0.001 / 0.01 / 0.1; 1 - S at
v = 2 is 0.00246 against (23)'s 0.00258.

Name clash: [W67]'s own Q(eta_0) in (A10) is a different quantity. Mourad and Jackson's Q (their (25)) is
[W67]'s B, and their t (26) is [W67]'s v with s the total rms slope.

### 2.3 The backscatter shadow function R (eqs. (48)-(50), p. 144)

    R(theta_1) = [w_ill(eta = -1/eta_1) / w(eta = -1/eta_1)] S(theta_1)                        (48)
               = S(theta_1) / {(1/2)[1 + erf(eta_1 / sqrt(2 abs(psi_0'')))]}                        (49)
    with (20):  R = (1 - e^(-2B)) / (2B)   (ours, substituting (20) in (49))
    w(eta = -1/eta_1) R(theta_1) = w_ill(eta = -1/eta_1) S(theta_1)                             (50)

- R is the ratio of points on the illuminated surface with slope normal to the incident ray to all such points
  (Brockelman and Hagfors, via secondary). By (50), the high-frequency backscattered power, proportional to
  w(eta = -1/eta_1), is corrected for shadowing "by simply multiplying by R(theta_1)" (p. 144).
- Limits (ours): R -> 2 sqrt(pi) v as v -> 0 (twice S); R -> 1 - B as v -> infinity, faster than S
  (1 - R = 1.2e-4 at v = 2, 2.8e-7 at v = 3). No shadowing at normal incidence: v = infinity, R = S = 1. At
  grazing incidence R -> 0 linearly in tan(theta).
- Values (ours): R = 0.385059, 0.719744, 0.906503, 0.987541, 0.998564, 0.999878 at v = 0.1, 0.25, 0.5, 1, 1.5, 2.

### 2.4 Bistatic shadowing (eqs. (25)-(36), pp. 141-142)

theta_1 = grazing angle of the incident ray measured from +x (towards the source); theta_2 = angle of the
observation ray measured from -x; both rays in the plane of incidence. In grazing terms (ours): 0 <= theta_2 <=
pi/2 is the forward half (observer on the far side), theta_2 = theta_1 the specular direction, theta_2 = pi -
theta_1 backscatter.

| Range | Eq. | S(theta_1, theta_2) |
|---|---|---|
| 0 <= theta_2 <= pi/2 (forward) | (33) | {1 - exp[-2(B_1 + B_2)]} {erf v_1 + erf v_2} / [4 (B_1 + B_2)] |
| pi/2 <= theta_2 <= pi - theta_1 | (35) | S(theta_1) |
| pi - theta_1 <= theta_2 <= pi | (36) | S(theta_2) (the monostatic S at the observation ray) |
| | (30), (31) | B_i = [exp(-v_i^2) - sqrt(pi) v_i erfc v_i] / (4 sqrt(pi) v_i), v_i = abs(eta_i) / sqrt(2 abs(psi_0'')), eta_i = tan theta_i |
| | (A33) | improvement: B_i -> (1/2) F(theta_i) of (A30)/(A32) |

So in the backward half S = S(min of the two grazing angles) (ours, restating (35)-(36)); the limit as theta_2
-> 90 degrees is S(theta_1) (p. 142). Forward specular (theta_2 = theta_1) (ours, from (33)):
S_spec = erf(v)(1 - e^(-4B)) / (4B).

Bistatic R (ours, derived from (46)-(47) and (33), (35)-(36) as (48) is from (47) and (20); not printed):
- forward half: R_b = (1 - exp[-2(B_1 + B_2)]) / [2 (B_1 + B_2)] for the facet slopes both rays see;
- backward half: R_b = R(min(theta_1, theta_2)), the monostatic R of the lower ray.
Both are symmetric in the two angles (reciprocity) and equal R(theta) in backscatter.

### 2.5 Illuminated heights and slopes (eqs. (40)-(47), (51)-(53))

| Eq. | Content |
|---|---|
| (40)-(41) | w_ill(zeta) = 2B exp[-B erfc(zeta / sqrt(2 psi_0))] w(zeta) / (1 - exp(-2B)), with B = B_1 + B_2 (forward), B_1 or B_2 (backward ranges) |
| (42)-(43) | Gaussian approximation about the mode zeta_0; Sigma^2 = psi_0 / (1 + zeta_0^2 / psi_0) the illuminated height variance; Sigma^2 / sigma^2 << 1 at small grazing angles: the surface "will reflect radiation as though it were a much smoother surface", so a Rayleigh roughness parameter should use Sigma (p. 143) |
| (46) | forward: w_ill(eta) = 2 u(eta_1 - eta) u(eta + eta_2) w(eta) / [erf(eta_1 / sqrt(2abs(psi_0''))) + erf(eta_2 / sqrt(2abs(psi_0'')))] |
| (47) | backward: w_ill(eta) = 2 u(abs(eta_i) - eta) w(eta) / [1 + erf(abs(eta_i) / sqrt(2abs(psi_0'')))] (i = 1 for pi/2 <= theta_2 <= pi - theta_1, i = 2 beyond) |
| (51)-(53) | mean slope of the illuminated surface: forward 2 sqrt(abs(psi_0'') / 2 pi) [exp(-v_2^2) - exp(-v_1^2)] / (erf v_1 + erf v_2); backward -2 sqrt(abs(psi_0'') / 2 pi) exp(-v_i^2) / (1 + erf v_i); largest magnitude 2 sqrt(abs(psi_0'') / 2 pi) at theta_2 = pi: a rough surface reflects as if tilted |

The step discontinuity in (46)-(47) is an artefact of the approximations; the area under w_ill is somewhat
overestimated for small ray slopes (p. 144).

### 2.6 Appendix A: the small-separation correction (pp. 145-147)

- (A9)-(A10): for large separations q(tau) -> q_1(tau) = (abs(psi_0'') / psi_0)^(1/2) (2 pi)^-1 exp(-X_4^2 /
  2 psi_0) Q(eta_0), Q(eta_0) = exp(-eta_0^2 / 2abs(psi_0'')) - sqrt(pi) eta_0 (2abs(psi_0''))^(-1/2) erfc(eta_0 /
  (2abs(psi_0''))^(1/2)) - the form assumed in Sec. I.
- (A11)-(A20): the small-tau behaviour needs derivatives of psi through 8th order; (A15)-(A17) define alpha,
  beta, gamma^2 from psi_0, psi_0'', psi_0^(4), psi_0^(6).
- (A28)-(A30): S(theta given zeta_1') = u(eta_0 - zeta_1')(1 - e^-F)/F; **S(theta) = (1/2)(1 + erf v)(1 - e^-F)/F**
  (A29); F (A30) is a term in exp(-2abs(psi_0'') v^2 / gamma^2) with a prefactor from psi_0, abs(psi_0''), alpha,
  beta, plus terms of the form of 2B (the last line of (A30) is partly cut on the scan and is not re-typed
  here; the Gaussian case (A32) is clear and is what Fig. 3 plots).
- (A31)-(A32), Gaussian autocorrelation psi = sigma^2 exp(-tau^2 / T^2): psi_0'' = -2 sigma^2 / T^2,
  psi_0^(4) = 12 sigma^2 / T^4, psi_0^(6) = -120 sigma^2 / T^6; alpha = 3 T^6 / (8 sigma^2), beta = 8 sigma^4 /
  T^4, gamma^2 = 32 sigma^2 / (9 T^2);
  **F = (1/2){ (sqrt(3 pi) v)^-1 exp(-9 v^2 / 8) + (sqrt(pi) v)^-1 exp(-v^2) - erfc v }** (A32),
  v = eta_0 / (2 sigma / T). Note F/2 = B + (sqrt(3 pi) v)^-1 exp(-9v^2/8) / 4 (ours): always more shadowing
  than (20).
- As theta -> 0 the approximations behind (A29) fail and (20) "is to be used in place of Eq. A27" (p. 147).

## 3. Figures (figure readings; error +-0.02 in S unless said)

Computed values are ours (`wagner.py`), with v = tan(theta) / (2 sigma / T) for the Gaussian autocorrelation
(sqrt(2 abs(psi_0'')) = 2 sigma / T).

| Figure | Readings | Computed | Verdict |
|---|---|---|---|
| Fig. 2, (20) solid, sigma/T = 1.0 | 10 / 20 / 30 / 40 / 50 / 60 deg: 0.19, 0.36, 0.50, 0.63, 0.75, 0.87 | (20): 0.19, 0.36, 0.50, 0.63, 0.75, 0.87 | (20), (21), (22) as read are what was plotted |
| Fig. 2, Brockelman-Hagfors dashed, sigma/T = 1.0 | 10 / 30 / 50: about 0.14, 0.41, 0.69 | - | (20) is 0.05-0.09 above the simulation |
| Fig. 3, (A29), sigma/T = 1.0 / 0.5 / 0.25 / 0.1 | 1.0: 10 / 20 / 30 / 40 / 50 / 60 / 70 deg 0.12, 0.27, 0.41, 0.54, 0.68, 0.83, 0.96; 0.5: 0.27, 0.49, 0.67, 0.82, 0.93 (10-50); 0.25: 0.48, 0.77, 0.93 (10-30); 0.1: 0.55, 0.83 (5, 10) | (A29): 0.12, 0.26, 0.40, 0.54, 0.69, 0.84, 0.96; 0.26, 0.49, 0.67, 0.82, 0.94; 0.48, 0.77, 0.93; 0.56, 0.84 | (A29), (A32) as read are what was plotted |
| Fig. 4, (33), rms slope 0.25 (sqrt(2abs(psi_0'')) = 0.25), plateau at theta_2 -> 90 | theta_1 = 1.4 / 5.7 / 9.9 deg: 0.21, 0.62, 0.775 (+-0.01); a second, slightly lower line under each plateau (0.19, 0.60, 0.76) is not explained in the text | (33) -> S(theta_1) of (20): 0.209, 0.612, 0.803; with (A33): 0.134, 0.523, 0.752 | 1.4 and 5.7 deg match (20); 9.9 deg reads 0.03 lower (unexplained). Caption "9.5 deg" against arctan(0.25 / sqrt 2) = 10.0 deg (ours) |
| Fig. 4 rise, theta_1 = 9.9 deg | theta_2 = 1 / 3 / 10 deg: about 0.10, 0.30, 0.60 | (33): 0.101, 0.289, 0.624 | matches within reading |
| Fig. 6, R(theta_1), sigma/T = 1 (theory solid) | 10 / 20 / 30 / 40 / 50 / 60 / 70 deg: 0.29, 0.47, 0.62, 0.74, 0.84, 0.93, 0.99 (+-0.02) | R from (A29) via (49): 0.22, 0.44, 0.61, 0.75, 0.86, 0.94, 0.99; R from (20): 0.34, 0.61, 0.77, 0.87, 0.94, 0.98, 1.00 | the plotted R follows the (A29) form within 0.02 from 30 deg; at 10-20 deg it lies between the two forms. [W67]: agreement with the simulation "excellent" above 30 deg; below, theory falls faster |

## 4. Against our code

| # | [W67] / users | Ours (file:line) | Verdict, numbers (ours) |
|---|---|---|---|
| W1 | Composite backscatter needs R (49) with B (22) (and Mourad-Jackson (24)-(26), J&R (13.81)-(13.83)) | `composite_strength`, `combined_strength`, bottom.py:518-543: no shadowing; `_slope_average` drops facets with local angle <= 0 (bottom.py:502-515) | DIFFERS. 10 log R with t = tan(theta) / (sqrt(2) x per-axis slope of `large_scale_slope`): medium sand 20 / 50 / 100 kHz at 2 deg -0.62 / -1.52 / -2.58 dB, at 5 deg -0.05 / -0.28 / -0.68, at 10 deg 0 / -0.02 / -0.12; sandy gravel at 5 deg -0.55 / -1.38 / -2.39, at 2 deg -2.25 / -4.07 / -5.80; very fine silt 0.00 above 2 deg. Ours drops back-facing facets, so R (not S) is the right factor |
| W2 | Bistatic composite: (33), (35)-(36); R_b (ours, 2.4) | ours is bistatic (ti, ts, phi), no shadowing | DIFFERS; out-of-plane not printed |
| W3 | Facet (geometric-optics) backscatter of the sea surface is the w(eta = -1/eta_1) of (50): multiply by R | `surface.facet_backscatter`, surface.py:568-576 (Cox-Munk total mss, `acoustics.mean_square_slope`) | ABSENT but immaterial: where the facet term is within 30 dB of its peak (grazing above 47-70 deg for 3-20 m/s) 10 log R = 0.000 dB; at 10 deg R is -0.02 to -0.35 dB where the facet term is more than 1000 dB below its peak (`surf.py`). For an anisotropic sea the in-plane component of `mean_square_slope_components` is abs(psi_0'') |
| W4 | Coherent component times S^2 (p. 141), forward specular S_spec = erf(v)(1 - e^-4B)/(4B) | coherent reflection (Rayleigh parameter, `acoustics`) without shadowing | ABSENT. 20 log S_spec (total slope s): s 0.1: -11.7 / -2.5 / -0.1 dB at 2 / 5 / 10 deg; s 0.2: -22.4 / -8.9 / -2.4; s 0.3: -29.7 / -14.2 / -5.7. Ray theory only; no source read applies it to acoustic coherent loss. Not proposed for code now (C-W4) |
| W5 | Illuminated height variance Sigma^2 (42)-(43) for the Rayleigh parameter at low grazing | Rayleigh parameter with the full rms height | ABSENT; no numbers printed beyond Fig. 5 for one slope; record only |
| W6 | Geometric shadowing of cells in the scene (PHYSICS_NOTES lines 1840-1853) | occluder-based | unrelated (deterministic geometry, not a statistical shadow function) |

### 4.1 Which form Mourad-Jackson (1989) and J&R (2007) use

| Item | [W67] | Mourad and Jackson (1989) | J&R (2007) (13.79)-(13.83), as in our note 04 |
|---|---|---|---|
| Function | S (20) and R (49) | R: (1 - e^-2Q)/(2Q) (24) | R: same form (13.81) |
| Inner quantity | B (22), pi^(-1/2) | Q (25) = B, pi^(-1/2) | Q (13.82) printed with pi^(+1/2): misprint |
| Argument | v = tan(theta) / sqrt(2 abs(psi_0'')) (21) | t = tan(theta) / s, s total rms slope from their (18) (26) | t = tan(theta) / (sqrt(2) sigma_sL), sigma_sL per axis (13.83) |
| Equivalence | - | s = sqrt(2 abs(psi_0'')) for an isotropic surface | sqrt(2) sigma_sL = sqrt(2 abs(psi_0'')) |
| Slope magnitude | - | (18): cut-off at k h_s = 1/sqrt(2) (ours, algebra); at gamma2 3.25 this s is 0.81 of J&R's total slope | (13.72)-(13.77): k h_s = 1/2 (our `large_scale_slope`) |
| Back-facing facets | excluded by R | clamped to 1 deg in the 3-point average | integral from -theta (dropped) |
| Large t | - | S = 1 for t > 2 | not stated |

The function is the same in both (after the J&R misprint is corrected); the difference between them is only the
slope they put in (a different cut-off) and the treatment of back-facing facets.

## 5. Corrections to apply (ordered)

**C-W1. Monostatic shadowing in `composite` / `combined`.** As C1 of `bottom_mourad_jackson_1989.md` (one
change, one place). Function to add in `python/gpusonar/sim/bottom.py`:
`wagner_backscatter_shadowing(grazing, total_rms_slope)` = (1 - exp(-2B)) / (2B), B = [pi^(-1/2) exp(-v^2) -
v erfc(v)] / (4v), v = tan(grazing) / total_rms_slope; return 1 where v > 2 (error < 1.3e-4) and 0 at grazing
<= 0; evaluate B with `scipy.special.erfcx` for v > 2 if not short-circuited (cancellation). Our
`large_scale_slope` gives the per-axis slope: total = sqrt(2) x it. Tests (fail before: the function does not
exist): the values of 2.3 (1e-6); the limits R/(2 sqrt(pi) v) -> 1 (2 % at v = 0.01) and R(2) = 0.999878;
monotonic in grazing; Fig. 6 is NOT a test of this form (it plots the (A29) form). Moves: as C1 there (worst
-2.39 dB at 5 deg, sandy gravel 100 kHz; opt-in models only; 0 shipped).

**C-W2. Bistatic form inside the composite average (same change, bistatic arguments).** Backscatter (phi = pi,
ti = ts): R(theta). In-plane backward half (phi = pi, ti != ts): R(min(ti, ts)) ([W67] (35)-(36) with (47),
ours). In-plane forward half (phi = 0): R_b = (1 - exp[-2(B(v_i) + B(v_s))]) / [2(B(v_i) + B(v_s))] ([W67]
(33), (46), ours). Out of plane: not printed; use the backward form for cos(phi) < 0 and the forward form for
cos(phi) >= 0 and label it "ours" in the docstring. Test: symmetric in (ti, ts) (the existing
`test_reciprocity` must still pass for `composite` and `combined`); equals C-W1 in backscatter to 0 dB; R_b <=
min(R(ti), R(ts)) in the forward half. Opt-in models only; 0 shipped.

**C-W3. Surface facet term: no change.** Shadowing changes it by 0.000 dB wherever it matters (W3). Record in
`surface.facet_backscatter`'s docstring: "Wagner's backscatter shadow function R (J. Acoust. Soc. Am. 41,
138-147, 1967, eqs. (49), (20)-(22)) is 1 to 1e-6 dB wherever this term is within 30 dB of its peak".

**C-W4. Coherent forward shadowing: record only (not now).** [W67] p. 141 says the coherent component is
multiplied by S^2 (forward specular S_spec from (33)). It would be a new, large low-angle loss (tens of dB at 2
degrees for slopes 0.2-0.3) from a geometric-optics argument that ignores diffraction ([W67] p. 145); no source
read applies it to acoustic coherent reflection. If the owner wants it: opt-in `coherent_shadowing=True` on the
surface coherent reflection, with the numbers of W4 as the test values.

**C-W5. REFERENCES.md citation fix** (final report): Wagner is J. Acoust. Soc. Am. 41 (1), 138-147 (1967), not
IEEE Trans. Antennas Propag.

**Stays blocked / not available here:** out-of-plane bistatic shadowing (not printed; C-W2's rule is ours); a
shadow function for power-law (non-Gaussian-autocorrelation) relief beyond the B form (A29 needs derivatives
that a power law does not have); diffraction corrections (outside [W67]).

## 6. Not verified / unclear on the page

- (A30)'s last line is partly cut on the scan; (A32), the Gaussian case, is clear and reproduces Fig. 3.
- Fig. 4: the second line under each plateau, and the 9.9-degree plateau 0.03 below (20), are unexplained.
- Fig. 6's theory curve at 10-20 degrees lies between the (20) and (A29) forms of R; which was plotted there is
  not stated (the caption says "theoretical curve").
- Fig. 2's dashed (Brockelman-Hagfors) readings: +-0.03.
- The bistatic R_b of 2.4 and the out-of-plane rule of C-W2 are ours (derived, not printed).
