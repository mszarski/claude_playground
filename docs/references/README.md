# Reference notes

Notes taken from sources the owner supplied, for agents who implement or check models against them. The plan
that turns them into work is `docs/tasks/SST_GAPS_JACKSON.md` (merged work packages, owner decisions, misprints,
status changes). The status of every source the suite rests on is kept in `docs/REFERENCES.md`; these notes
propose changes to it but do not make them.

## Rules for these notes

- Own words. Formulas are re-typed in the source's notation, with our symbol beside them; constants and values
  carry the printed page and the equation, table or figure number. Nothing is copied at length.
- Status words as in `docs/REFERENCES.md`: **confirmed** means read on the page image of that source; **via
  secondary** means a formula of another author as this source reprints it; **ours** means our derivation or a
  number we computed (the script is named); **read from figure** means a value read off a printed curve, never a
  printed number.
- The PDFs are not in the repository. Scripts that produced "ours" numbers live in the session scratchpad and are
  not committed.
- APL-UW TR 9407 is distribution-limited and is never read. Where a source attributes something to it, the note
  records what the source prints and says so.
- Each note leads with what is wrong, missing or risky, then gives the source, compares it with our code, lists
  what it closes in open issues, proposes work packages and ends with what was not verified (probable misprints
  included).

## Files

| File | Source (as the note cites it) | Covers | Summary |
|---|---|---|---|
| `jackson_richardson_2007/01_sediment_properties.md` | D. R. Jackson and M. D. Richardson, *High-Frequency Seafloor Acoustics*, Springer, 2007 | Chs. 1-5 (pp. 1-170), Apps. A, B, C, H, I; also Ch. 8 pp. 250-252, Ch. 11 pp. 312-314, Ch. 14 pp. 381-382 | Conventions, grain size, the Ch. 5 geoacoustic regressions, shear, rock and seawater. Our grain-size table equals the TR 9407 relations as the book prints them (11.1)-(11.3), except a probable misprint (1.0091 for 1.0019); the book calls those relations ad hoc model fits. Seawater density and Francois-Garrison confirmed; rock shear would add about 5 dB of loss per bounce at some angles |
| `jackson_richardson_2007/02_roughness_heterogeneity_statistics.md` | same book | Chs. 6, 7, 16 (pp. 171-243, 421-442), Apps. D, E; Ch. 14 pp. 380-390 for the volume link | Roughness spectra and their units, Briggs' measured spectra (Table 6.1), heterogeneity spectra, backscatter statistics (K, lognormal, McDaniel 1996). Our spectrum convention, structure constant, volume term and K shape are confirmed; (D.19) and (D.23) are misprinted. Mud volume parameters implied by measured heterogeneity are mostly above our class value |
| `jackson_richardson_2007/03_wave_theories_reflection_penetration.md` | same book | Chs. 8-11, 15 (pp. 245-320, 407-420), Apps. F, G; eq. (13.14), App. L.3 | Fluid, elastic, Biot and effective-density reflection; reflection data; penetration into sediment; measurement equations. Our fluid reflection is exact; rock as a fluid is wrong by up to about 19 dB per bounce; our `sediment` tier lacks attenuation and transmission (6-9 dB in the buried scenes) and is wrong below the critical angle. No layered recursion is in the book |
| `jackson_richardson_2007/04_scattering_models.md` | same book | Chs. 12-14 (pp. 321-406), Apps. J-M | Perturbation, Kirchhoff, small-slope, composite, volume (continuous and discrete), elastic and layered models, accuracy and the Ch. 12 data. Every model we have agrees to under 0.001 dB; our composite lacks Wagner's shadowing (whose printed form needs checking). The interpolation, the gravel/rock law and the bistatic interpolation are not printed |
| `bottom_mourad_jackson_1989.md` | P. D. Mourad and D. R. Jackson, "High Frequency Sonar Equation Models for Bottom Backscatter and Forward Loss", Proc. OCEANS '89, pp. 1168-1175 (1989) | all 8 pages | The 1989 APL model in full: grain-size relations (1)-(7), forward loss (8)-(12), backscatter (13)-(39) with the Kirchhoff/composite interpolation (33)-(37) that J&R do not print, five sites' parameters. Eq. (39) is misprinted (abs(1 - R)^2 for abs(1 - R^2)^2; the paper's figures and ours use the latter). Settles J&R's 1.0091 (1.0018916) and the shadowing sign. Its slope-averaged model is +1.6 to +11.5 dB above our default at 5 deg for sands and gravel; corrections C1-C5 |
| `shadowing_wagner_1967.md` | R. J. Wagner, "Shadowing of Randomly Rough Surfaces", J. Acoust. Soc. Am. 41 (1), 138-147 (1967) | all 10 pages, Appendix A | Monostatic S and backscatter R, bistatic S, illuminated heights and slopes. Settles pi^(-1/2) (J&R (13.82) misprinted). The composite model needs R; ours has none (-0.7 to -2.4 dB at 5 deg, sands and gravel at 100 kHz); no effect on the surface facet term. REFERENCES cites the wrong journal |
| `surface_papers_mcdaniel_dahl.md` | S. T. McDaniel, "Sea surface reverberation: A review", J. Acoust. Soc. Am. 94 (4), 1905-1922 (1993); P. H. Dahl and W. J. Plant, J. Acoust. Soc. Am. 101 (5), 2596-2602 (1997); P. H. Dahl, W. J. Plant, B. Nuetzel, A. Schmidt, H. Herwig and E. A. Terray, J. Acoust. Soc. Am. 101 (5), 2583-2595 (1997) | all pages of the three papers | Surface roughness, resonant-bubble and plume models, a wave spectrum, a fitted optical-depth law, fluctuation statistics and data up to 70 kHz. Our bubble and facet terms equal the published forms; Chapman-Harris is now via secondary with its range. Our model is 6-12 dB above open-ocean data at low wind; nothing covers the surface above 70 kHz |
| `bottom_bistatic_williams_jackson_1998.md` | K. L. Williams and D. R. Jackson, "Bistatic bottom scattering: Model, experiments, and model/data comparison", J. Acoust. Soc. Am. 103 (1), 169-181 (1998) | all 13 pages | The published bistatic bottom model: Kirchhoff and perturbation joined by (sigma_k^-2 + sigma_p^-2)^(-1/2), a spectral bistatic volume term, Panama City and Eckernfoerde inputs and 40 kHz data. Our perturbation, Kirchhoff and volume forms equal it to 1e-13 dB; its join lies up to 6-7 dB below our small-slope default at 20-45 deg for sands and fits its own bistatic data better (+3.3 against +6.1 dB mean bias, one site). Opt-in model, spectral volume and a site row proposed |
| `bottom_backscatter_jackson_briggs_1992.md` | D. R. Jackson and K. B. Briggs, "High-frequency bottom backscattering: Roughness versus sediment volume scattering", J. Acoust. Soc. Am. 92 (2), Pt. 1, 962-977 (1992) | all 16 pages | Three sites (Quinault fine sand, Arafura and San Francisco muds) with geoacoustics, roughness spectra and fitted volume parameters. Printed fits 0.005 and 0.002 at the mud sites contradict the agents' derived 0.0031 and 0.0037; eq. (4) confirms the corrected J&R (D.19)/(D.23); our model reproduces its curves within about 1 dB at 20-70 deg; three prints give three composite cut-offs. Three site rows and a range update proposed |
