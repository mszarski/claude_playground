# SAS Simulator + Beamformer — Project Brief

## Goal
Build a synthetic aperture sonar (SAS) simulation + beamforming workbench,
in the spirit of ApertureLab: a scene becomes physics-accurate raw sonar
I/Q, and a beamformer focuses it back into labeled imagery.

## Anchor file
`sas_min.py` is the correct, working core: point-scatterer forward model →
matched-filter range compression → Time-Domain Back-Projection (TDBP) →
SLC + DRC imagery. It self-verifies (focused peaks land on true target
positions; measured range resolution matches c/(2B)). **Do not break it.**
Every new capability extends this, and must preserve its correctness checks.

## Hard constraints
- **Offline.** No network. Use only numpy, scipy, matplotlib (already installed).
  Do not add dependencies that require downloads.
- **One rung per session.** Implement a single roadmap item, prove it with a
  numeric check, then stop. Do not attempt multiple rungs at once.
- **Every change ships a test.** After each rung, print a pass/fail correctness
  check (e.g. peak positions, resolution vs theory, energy conservation).
  If a change degrades an earlier check, it's a regression — fix or revert.
- **Physics decisions get flagged, not guessed.** When a step needs a modeling
  choice you can't validate from first principles (scattering model, roughness
  spectrum, autofocus criterion), state the assumption explicitly in a comment
  and in your reply so the human can confirm it.

## Roadmap (in order)
1. Multichannel receive array (e.g. 36 channels). Check: focused resolution
   unchanged vs single-channel on the same scene.
2. ω-k / range-Doppler beamformer alongside TDBP. Check: ω-k and TDBP images
   agree to within a small tolerance on the same raw data.
3. Height-field seafloor + line-of-sight shadowing (replaces point scatterers
   with a terrain/facet model). Check: shadows fall behind targets, geometry
   matches grazing angle.
4. Seafloor backscatter texture (speckle + Lambertian/specular, roughness
   spectra). Check: speckle statistics match the chosen distribution.
5. Motion errors + micronavigation/autofocus. Check: focusing recovers after
   injected sway, measured by image sharpness metric.
6. GPU acceleration of the TDBP inner loop (CuPy if available, else stay CPU).
   Check: GPU and CPU images identical within float tolerance.
7. Scene editor GUI, node-graph texture editor, industry file I/O (XTF, HDF5).
   Pure software; no new physics.

## Working style
Small, verifiable commits. Keep sas_min.py runnable at every step. Prefer
extending it into a package (`sas/`) over rewriting it. When in doubt about
physics, ask before proceeding.
