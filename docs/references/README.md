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
| `surface_papers_mcdaniel_dahl.md` | S. T. McDaniel, "Sea surface reverberation: A review", J. Acoust. Soc. Am. 94 (4), 1905-1922 (1993); P. H. Dahl and W. J. Plant, J. Acoust. Soc. Am. 101 (5), 2596-2602 (1997); P. H. Dahl, W. J. Plant, B. Nuetzel, A. Schmidt, H. Herwig and E. A. Terray, J. Acoust. Soc. Am. 101 (5), 2583-2595 (1997) | all pages of the three papers | Surface roughness, resonant-bubble and plume models, a wave spectrum, a fitted optical-depth law, fluctuation statistics and data up to 70 kHz. Our bubble and facet terms equal the published forms; Chapman-Harris is now via secondary with its range. Our model is 6-12 dB above open-ocean data at low wind; nothing covers the surface above 70 kHz |
| `surface_forward_scatter_dahl_1996_2004.md` | P. H. Dahl, "On the spatial coherence and angular spreading of sound forward scattered from the sea surface: Measurements and interpretive model", J. Acoust. Soc. Am. 100 (2), Pt. 1, 748-758 (1996); P. H. Dahl, "Forward scattering from the sea surface and the van Cittert-Zernike theorem", J. Acoust. Soc. Am. 115 (2), 589-599 (2004) | all pages of both papers | Measured horizontal and vertical coherence and angular spread of forward scatter at 8-40 kHz, the large-scale slope law (A7) and two measured decays. Our facet geometry matches Dahl's scaling, but our slope (the Cox-Munk total) makes spreads too wide, coherence lengths 25-45 % short and tails 4-10 dB too long. Corrects the citation of the 1996 paper in `REFERENCES.md` |
| `surface_bistatic_dahl_1999_mcdaniel_1986.md` | P. H. Dahl, "On bistatic sea surface scattering: Field measurements and modeling", J. Acoust. Soc. Am. 105 (4), 2155-2169 (1999); S. T. McDaniel, "Diffractive corrections to the high-frequency Kirchhoff approximation", J. Acoust. Soc. Am. 79 (4), 952-957 (1986) | all pages of both papers | A bistatic Kirchhoff/small-slope surface model with a bistatic bubble term (30 kHz, 1-4 m/s), measured delay e-folds and angle spreads, and McDaniel's corrected composite. Our facet lobe equals the geometric-optics Kirchhoff term (with McDaniel's 1993 erratum), which settles J&R (L.34)/(L.35) as misprints. No surface model of ours takes the bistatic angle; McDaniel (1986) is not TR 0702's near-specular model |
