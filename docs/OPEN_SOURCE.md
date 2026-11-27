# Open-source survey for the GPU sonar suite

Later additions: section 3.12 (sources of the SST-equivalence work, D-62 to D-79) and
section 3.13 (everything used since D-82: org-arl formulations, Bellhop as an external
process for new purposes, Bellhop3D as the program of the opt-in `eigenray3d` tier since
2026-11-26, the three measured channel data sets, the literature search).

Status: first-hand verification pass, 2026-11-16, from an x86_64 Ubuntu 24.04 container
(no GPU, Python 3.13.18, gcc 13.3, cmake 3.28.3, restricted network).
Every role below is **PROPOSED** — the project owner finalises.

How things were checked (used as shorthand in the tables):

- **cloned** — anonymous `git clone --depth 1`; licence read from the LICENSE/COPYING file in the tree;
  "last activity" is the committer date of the default-branch HEAD at clone time (`git log -1 --format=%cI`).
- **pip** — installed into a scratch venv on Python 3.13 and imported; **PyPI** — metadata only
  (`pypi.org/pypi/<name>/json`).
- **built** — compiled and run here. **apt** — `apt-cache policy` on Ubuntu 24.04 (noble); nothing was installed system-wide.
- **search** — GitHub repository search API via `gh` (repo metadata only; the plain REST repo endpoint was not available).
- Anything marked **unverified** was not checked first-hand (usually because the host is blocked; see section 5).

No GPU was available, so no CUDA code path of any project was executed.

## 1. Rules

1. Permissive licences preferred for anything linked into, vendored into, or ported into the core
   (Apache-2.0, BSD, MIT, BSL-1.0, zlib; MPL-2.0 header-only use is acceptable as-is, file-level copyleft only).
2. GPL / LGPL code is used only as a **separate-process test oracle** or an optional external tool.
   It is never linked into the core, never vendored, and never read-and-ported line by line.
3. Any copyleft item (including LGPL, and including transitive copyleft such as SuiteSparse under Ceres)
   needs the **owner's explicit sign-off** before it appears in a build, a container image or CI.
4. Datasets are tracked like code: licence/terms recorded, never redistributed in this repo unless the terms allow it.
5. Proprietary runtime dependencies (CUDA toolkit, cuFFT, Holoscan binaries' NVIDIA components) are allowed
   as dynamically linked platform dependencies, not vendored.

## 2. Summary table

Roles: depend / vendor / port / test-oracle only / reference reading only / reject. All roles are PROPOSED.

### 2a. Capture / pipeline

| Project | URL | Licence (as read) | Last activity | Validation evidence | Verified here | PROPOSED role | Notes |
|---|---|---|---|---|---|---|---|
| Holoscan SDK | github.com/nvidia-holoscan/holoscan-sdk | Apache-2.0 (LICENSE.txt) + NOTICE.txt | 2026-10-20 (HEAD 358c772); PyPI `holoscan` 5.0.0 uploaded 2026-10-20 | NVIDIA product; benchmarks in tree (`benchmark/`), not run | cloned; aarch64 wheel downloaded; x86_64 wheel pip-installed, **import fails** here (`libcudart.so.13` missing — no CUDA) | depend (optional pipeline layer, not core) | 5.0 is a breaking redesign, CUDA 13 only, Python >= 3.12; see 3.1 |
| Holoscan Sensor Bridge | github.com/nvidia-holoscan/holoscan-sensor-bridge | Apache-2.0 (LICENSE); `third_party_licenses/` dir present, not audited | 2026-11-09 (HEAD 41cecaa); VERSION 3.0.0 | NVIDIA product | cloned only | depend (capture, optional) | README: needs Holoscan SDK >= 5.0.0, CUDA 13, L4T R38.4+ |
| holoscan-ultrasound-imaging | github.com/nvidia-holoscan/holoscan-ultrasound-imaging | Apache-2.0 (LICENSE) for NVIDIA source; Altera FPGA IP and MimickNet weights under separate terms | HEAD cabbbde 2026-11-13; pinned commit 24d405e exists, dated 2026-10-22 ("fix-pgx-link (#7)"); 8 commits total, first 2026-09-16 | none found in README | cloned (full history) | reference reading only | See 3.2 |
| HoloHub | github.com/nvidia-holoscan/holohub | Apache-2.0 (LICENSE) | 2026-11-13 (HEAD 7c44aa7) | — | cloned only | reference reading only | Per-application licences not audited |

### 2b. Ultrasound beamforming and benchmark

| Project | URL | Licence (as read) | Last activity | Validation evidence | Verified here | PROPOSED role | Notes |
|---|---|---|---|---|---|---|---|
| vbeam | github.com/magnusdk/vbeam | Apache-2.0 (LICENSE) | `main` HEAD 2025-03-20 (7a98a79); PyPI 1.0.10 2024-12-25; unmerged `dev_vbeam2.0` branches exist | README cites IUS 2023 paper, doi 10.1109/IUS51837.2023.10307255; README example uses PICMUS | cloned; pip + import OK on 3.13 after installing `jax` separately (0.12.0.1) | test-oracle only (CPU JAX reference beamformer) | `jax` is not pulled in by `pip install vbeam` |
| ultraspy | gitlab.com/pecarlat/ultraspy | MIT text, but the LICENSE copyright line reads "Copyright (c) 2018 The Python Packaging Authority" (template left in) | 2026-09-09 (HEAD 26b0bc1); PyPI 1.2.7 2026-09-09 | no DOI in README (unverified beyond that) | cloned; pip + import OK on 3.13 (DAS class import) | test-oracle only | Licence holder ambiguity, see 3.3. GPU path needs CuPy; has Capon |
| MUST (Matlab) | biomecardio.com/MUST (blocked) | **unverified**; PyMUST source headers say "MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later" | unverified | PyMUST README cites doi 10.1016/j.cmpb.2022.106726 and 10.1016/j.ultras.2020.106309 | NOT verified — host blocked; no GitHub repo found (`creatis-ULTIM/MUST` does not resolve) | reference reading only | Matlab; LGPL per indirect evidence |
| PyMUST | github.com/creatis-ULTIM/PyMUST | LICENSE file is **LGPL-2.1**; `setup.py` and PyPI metadata say LGPL v3; source headers say LGPL-3.0-or-later | 2026-09-14 (HEAD a96be54); PyPI 0.1.9 2026-01-19 | same papers as MUST; README warns "still under development, might have bugs" | cloned; pip + import OK on 3.13 (`simus`, `pfield`, `dasmtx`, `rf2iq` present) | test-oracle only (copyleft: sign-off) | Licence version inconsistent inside the repo, see 3.3 |
| USTB | github.com/unioslo/USTB | **MIT** (LICENSE, "UltraSound ToolBox contributors") | 2026-11-07 (HEAD b61c953); PyPI `ustb` 0.2.0 2026-10-25 | CITATION.cff; doi 10.1109/ULTSYM.2017.8092389, 10.1016/j.ultras.2026.108289; CI compares Python against MATLAB references | cloned; `pip install ustb` + import OK on 3.13 | test-oracle only; port candidates allowed by licence | Matlab core plus a Python package; PICMUS examples in `examples/picmus` and `python/examples/picmus_*.py` |
| zea | github.com/tue-bmd/zea | Apache-2.0 (LICENSE) | 2026-11-13 (HEAD 91bed66) | not assessed | cloned only | reference reading only | Its converter produced the reachable PICMUS mirror |
| PICMUS data | huggingface.co/datasets/zeahub/picmus | card: `license: other`, name `picmus-free-use`: "free of use", must cite PICMUS and the proceedings paper | HF lastModified 2026-08-03 | Liebgott et al., IUS 2016, doi 10.1109/ULTSYM.2016.7728908 (citation embedded in the files) | **downloaded** one file (25.9 MB) and opened with h5py: `raw_data` shape (1, 75, 382, 128, 2) float32 | depend (test data, fetched at test time, not committed) | Conversion to "zea" HDF5 layout, not the original files; see 3.4 |
| k-Wave | github.com/ucl-bug/k-wave | **LGPL-3.0** (LICENSE.txt; README badge and text agree) | 2026-05-05 (HEAD 94e5583) | README cites doi 10.1117/1.3360308, 10.1121/1.3377056 | cloned only (Matlab) | test-oracle only (copyleft: sign-off) | — |
| k-wave-python | github.com/waltsims/k-wave-python | **LGPL-3.0** (LICENSE) | 2026-08-20 (HEAD 1744cf6); PyPI 0.6.2 2026-05-18 | CITATION.cff; benchmark table in docs (not run) | cloned; pip + import OK on 3.13 (pip reported a dependency conflict warning) | test-oracle only (copyleft: sign-off) | Has a pure NumPy/CuPy backend and a `cpp` backend using k-Wave binaries; no simulation was run here |

### 2c. Underwater acoustics and propagation

| Project | URL | Licence (as read) | Last activity | Validation evidence | Verified here | PROPOSED role | Notes |
|---|---|---|---|---|---|---|---|
| arlpy | github.com/org-arl/arlpy | BSD 3-clause (LICENSE; header says "Simplified BSD") | 2026-09-06 (HEAD 2627a8c); PyPI 1.9.3 2026-07-18 | wide use as Bellhop front-end (GitHub search hits); no paper cited in README | cloned; **`pip install arlpy` fails on Python 3.13**; works with `--no-deps` on numpy 2.2.6 / scipy 1.15.3 for the functions tried | reference reading only; small formulas (absorption, sound speed) may be ported with citation | Maintenance mode; needs external `bellhop.exe`; see 3.5 |
| Acoustics Toolbox (BELLHOP, BELLHOP3D, KRAKEN, KRAKENC, SCOOTER, SPARC, BOUNCE, FIELD) | github.com/oalib-acoustics/Acoustics-Toolbox | **GPL-3.0** (LICENSE) | 2025-10-08 (HEAD 8b4682b) | de-facto community reference codes (OALIB); not independently re-validated | cloned; **built and run here** with a user-space gfortran 13.3 after one Makefile fix | test-oracle only, separate process (copyleft: sign-off); as used: BELLHOP is also the `eigenray` tier's program and, opt-in since 2026-11-26, BELLHOP3D the `eigenray3d` tier's, both separate processes (section 3.13) | See 3.6 for the build and test-case locations. `tools/build_oracles.sh` clones exactly 8b4682b. BELLHOP3D (same GPL-3 tree, separate process, found as `GPUSONAR_BELLHOP3D` or `bellhop3d.exe` beside `bellhop.exe`) is the 3-D oracle of `tests/python/test_seamount_3d.py` (`tests/python/bellhop3d_oracle.py`) and, since 2026-11-26 with the owner's approval, the program behind the opt-in `eigenray3d` tier (`python/gpusonar/sim/bellhop3d.py`: writes its input files, runs it as a separate process, reads its arrivals and TL; nothing linked or copied, as for BELLHOP and the `eigenray` tier) |
| A-New-BellHope/bellhop | github.com/A-New-BellHope/bellhop | GPL-3.0 (LICENSE) | 2023-03-30 (HEAD ae1a477) | — | cloned only | reject (superseded by the two rows around it) | Fortran fork used as the bellhopcuda comparison baseline |
| bellhopcxx / bellhopcuda | github.com/A-New-BellHope/bellhopcuda | **GPL-3.0** (LICENSE; README states GPLv3-or-later) | 2026-04-29 (HEAD d28b217) | repo ships comparison scripts and pass/fail lists against Fortran BELLHOP (`tl_match.txt`, `ray_tests_pass.txt`, `ray_tests_fail.txt`, `doc/accuracy.md`) | cloned; **CPU build (`-DBHC_ENABLE_CUDA=OFF`) built and run here**; CUDA build not attempted (no GPU) | test-oracle only, separate process (copyleft: sign-off) | 2D / 3D / Nx2D executables; README notes CMake >= 3.27 for CUDA detection |
| KRAKEN | (inside Acoustics Toolbox) | GPL-3.0 (same tree) | as above | as above | built here (`kraken.exe`, `krakenc.exe`); `MunkK` run produced a `.mod` file | test-oracle only (copyleft: sign-off) | — |
| RAM (Collins, Fortran) | oalib (blocked) | **unverified** | unverified | — | NOT verified — host blocked; not part of the Acoustics-Toolbox tree | reference reading only | — |
| PyRAM | github.com/marcuskd/pyram | BSD-style, 3 clauses (LICENSE.md; "Copyright 2000 Dr. Michael D. Collins, Copyright 2017 Marcus Donnelly") | 2024-12-10 (HEAD 78249c9); PyPI 1.3.0 | none cited in README | **pinned 1.3.0**; pip + import OK on 3.13; **run here** as the PE oracle of `tests/python/test_ram_crosscheck.py` (another version skips). Its dependencies, from their installed package metadata and LICENSE files: numba 0.68.1 (BSD, 2 clauses, "Copyright (c) 2012, Anaconda, Inc."; vendors code under the licences listed in its LICENSES.third-party, appdirs (MIT) and CPython (PSF) among them; requires numpy <2.6) and llvmlite 0.50.0 (`BSD-2-Clause AND Apache-2.0 WITH LLVM-exception`: its own code BSD-2, the bundled LLVM Apache-2.0 with LLVM exception) | test-oracle only (PE cross-check for the flank array); imported by tests/python only, never by the library (a test checks it) | Python/Numba port of RAM v1.5 (its docstring) |
| pyat | github.com/hunterakins/pyat | GPL-3.0 (license.txt) | 2025-08-02 (HEAD 53aac95) | — | cloned only | reject (env-file I/O is easy to write; GPL) | — |
| acoular | github.com/acoular/acoular | BSD 3-Clause (LICENSE) | 2026-11-11 (HEAD 0521af3); PyPI 26.8 2026-08-17 | CITATION.cff; doi 10.1016/j.apacoust.2016.09.015 | cloned; pip + import OK on 3.13 (`BeamformerCapon`, `BeamformerCleansc`, `BeamformerDamas`, … present) | test-oracle only (Capon/MVDR, CLEAN-SC, DAMAS references) | Requires Python >= 3.11 |
| pyroomacoustics | github.com/LCAV/pyroomacoustics | MIT text (LICENSE, EPFL-LCAV) | 2026-10-30 (HEAD 08d1696); PyPI 0.10.1 2026-05-01 | README cites arXiv:1710.04196 | cloned; pip + import OK on 3.13; DOA algorithms present: MUSIC, NormMUSIC, SRP, CSSM, TOPS, WAVES, FRIDA; `rake_mvdr_filters` present | test-oracle only (MUSIC/MVDR references) | cp313 wheels for x86_64 and aarch64 |

### 2d. SAR / SAS

| Project | URL | Licence (as read) | Last activity | Validation evidence | Verified here | PROPOSED role | Notes |
|---|---|---|---|---|---|---|---|
| ASASIN | arXiv:2101.05888 | no code released | paper 2021-01-14, last revised 2021-05-26 | — | paper downloaded and searched; see section 4 | reference reading only | — |
| RITSAR | github.com/dm6718/RITSAR | MIT (LICENSE, 2015 Douglas Macdonald) | **2016-10-04** (HEAD 0e36d2e) — dormant | none cited; widely forked teaching code | cloned; not on PyPI; imports on 3.13 from the clone (`backprojection`, `polar_format`, `omega_k`, `autoFocus`) — no processing run | port (small routines) / test-oracle only | Ships a small subset of AFRL Gotcha data |
| torchbp | github.com/Ttl/torchbp | MIT text (LICENSE, 2025 Henrik Forstén) | 2026-10-06 (HEAD 65e6c30) | none cited in tree | cloned only (needs PyTorch; not installed) | test-oracle only / reference reading | Differentiable SAR backprojection + autofocus, C++/CUDA PyTorch extension |
| gotcha-back | github.com/tbensonatl/gotcha-back | BSD 3-Clause (LICENSE, 2022-2023 Thomas Benson) | 2024-01-28 (HEAD 06da38a) | README: "hobbyist project" | cloned only (needs CUDA, Boost, FLTK, Gotcha data) | reference reading only; port allowed by licence | CUDA backprojection for Gotcha; author name matches an ASASIN co-author |
| sarpy | PyPI `sarpy` (NGA) | MIT per PyPI metadata (unverified from file) | PyPI 2.1.1 2026-09-18 | — | pip + import OK on 3.13 | reference reading only | SAR file I/O, not image formation |
| AFRL Gotcha data | sdms.afrl.af.mil | terms unverified; gotcha-back README says free download after account creation | — | — | NOT verified — host blocked | test data (manual download by owner) | — |
| synthetic_aperture_sonar_autofocus | github.com/isaacgerg/synthetic_aperture_sonar_autofocus | **no licence file** | 2025-02-03 (HEAD 16b1b70) | readme cites Prater et al. 2023 (Shadow PGA) | cloned only | reference reading only (no licence = no reuse) | PGA vs Shadow PGA built on RITSAR's PGA |
| AirSAS code | github.com/tblanford/airsas | **GPL-3.0** (LICENSE) | 2026-08-14 (HEAD 1e4936f) | README: accompanies a Scientific Data paper (Blanford et al.) | cloned only (Matlab) | reference reading only (copyleft) | Data on figshare, doi 10.6084/m9.figshare.26961892 — host blocked, data licence unverified |
| Neural-Volumetric-Reconstruction-for-Coherent-SAS | github.com/awreed/Neural-Volumetric-Reconstruction-for-Coherent-SAS | **no licence file** | 2023-10-28 (HEAD 0de52a0) | SIGGRAPH paper per repo description | cloned (924 MB), then deleted | reference reading only | Contains a PyTorch backprojection (`beamformer.py`) and AirSAS processing |
| ApertureLab Synthetic SAS v1.1 | huggingface.co/datasets/idg101/Aperture_Lab_Synthetic_Aperture_Sonar_v1_1 | custom "ALSD-1.0" with a named **Excluded Party** list | HF lastModified 2026-09-28 | simulator fidelity page on a blocked host (unverified) | card and LICENSE.md read; no data downloaded (41 GB) | reject pending owner's legal read | Beamformed, DRC'd images + labels only — no raw ping data; see 3.7 |

### 2e. Solvers, tracking, geodesy, messages, simulators

| Project | URL | Licence (as read) | Last activity | Validation evidence | Verified here | PROPOSED role | Notes |
|---|---|---|---|---|---|---|---|
| Ceres Solver | github.com/ceres-solver/ceres-solver | BSD-style 3-clause (LICENSE, Google) | 2026-11-05 (HEAD e831bfe; version.h says 2.3) | used by ASASIN itself (paper ref. [10]); very wide use | cloned; apt `libceres-dev` **2.2.0+dfsg-4.1ubuntu2** already installed; small LM auto-diff problem **built and solved here** | depend | apt build links GPL SuiteSparse parts, see 3.8 |
| GTSAM | github.com/borglab/gtsam | BSD (LICENSE + LICENSE.BSD); bundles CCOLAMD (BSD-3), Eigen (MPL2), METIS (Apache-2.0), Spectra (MPL2) | 2026-11-10 (HEAD 8806b33); PyPI 4.3.0 2026-09-19 | wide use in robotics; Zenodo DOI in README | cloned; pip + import OK on 3.13 (`ISAM2`, IMU preintegration present); apt has 4.2.0 | depend (optional, navigation fusion only) | Wheels for cp311–cp314, x86_64 and aarch64 |
| Stone Soup | github.com/dstl/Stone-Soup | MIT (LICENSE, Crown Copyright Dstl et al.) | 2026-11-13 (HEAD 1db378d); PyPI 1.9.1 2026-06-24 | CITATION.cff; FUSION 2023 paper doi 10.23919/FUSION52260.2023.10224185 | cloned; pip + import OK on 3.13 | test-oracle only (tracker reference) / optional Python dependency | Pure Python |
| GDAL | github.com/OSGeo/gdal | "MIT style" in general, with per-component exceptions listed in LICENSE.TXT (not audited) | 2026-11-13 (HEAD ccd3811) | OSGeo project, universal use | licence file read via blobless clone; apt `libgdal-dev` candidate **3.8.4+dfsg-3ubuntu3** (not installed) | depend (GeoTIFF writer) | — |
| PROJ | github.com/OSGeo/PROJ | MIT style (COPYING) | 2026-11-15 (HEAD 5e7a7e0) | OSGeo project | licence file read via blobless clone; apt `libproj-dev` candidate **9.4.0-1build2** (not installed) | depend | — |
| marine_acoustic_msgs | github.com/apl-ocean-engineering/marine_msgs (the candidate URL `apl-ocean-engineering/marine_acoustic_msgs` does not resolve) | BSD 3-Clause (LICENSE.txt, UW APL) | `main` 2024-03-15 (88e39b6); `ros2` branch 2024-03-15 (14bddb4); search API shows a push on 2025-11-21 (other branches) | used by APL sonar drivers | cloned | reference reading only (mirror field names/semantics in our own schema); depend only in an optional ROS bridge | Messages: `ProjectedSonarImage`, `RawSonarImage`, `SonarImageData`, `PingInfo`, `SonarRanges`, `SonarDetections`, `Dvl`, `DetectionFlag` |
| liboculus | github.com/apl-ocean-engineering/liboculus | BSD-style (LICENSE, University of Washington 2018-2020) | 2026-09-17 (HEAD 9e65e03) | — | cloned only | reference reading only / optional driver dependency | Network protocol for Blueprint Oculus; README: **cannot parse `.oculus` files** saved by vendor software; needs spdlog, fmt, boost::asio |
| HoloOcean | github.com/byu-holoocean/HoloOcean (not reachable anonymously) | PyPI metadata: "MIT License (BYU Client Code), Unreal EULA for Unreal Code" | PyPI 0.5.8 2024-04-03 (old client); docs repo 2026-11-14 with versions up to v2.4.0 | docs only | `pip install holoocean` 0.5.8 imports on 3.13 (client only, no worlds); main repo NOT cloned | test-oracle only at most; **owner sign-off** | Source gated behind Epic Games account linking; see 3.9 |

### 2f. Containers, bindings, build and FFT

| Project | URL | Licence (as read) | Last activity | Validation evidence | Verified here | PROPOSED role | Notes |
|---|---|---|---|---|---|---|---|
| HDF5 C library | github.com/HDFGroup/hdf5 | HDF Group licence (LICENSE; BSD-style, full clause text not audited) | 2026-11-11 (HEAD 0bb09e0) | decades of use | apt `libhdf5-dev` **1.10.10+repack-3.1ubuntu4** installed; used in the HighFive build below | (owner's choice) | aarch64: unverified here (amd64 apt only) |
| HighFive | github.com/highfive-devs/highfive (moved from BlueBrain) | **BSL-1.0** (LICENSE) | 2026-05-09 (HEAD 959ec30) | — | cloned; complex<float> write/read round trip **built and run here** (C++20, against apt HDF5 1.10.10) | (owner's choice) vendor/depend | Header-only; no `libhighfive-dev` candidate in the apt sources here |
| h5py | github.com/h5py/h5py | BSD 3-clause (LICENSE) | 2026-11-15 (HEAD e19c4da); PyPI 3.16.0 2026-03-06 | — | pip + import OK on 3.13; read the PICMUS file | (owner's choice) | manylinux aarch64 wheels on PyPI |
| zarr-python | github.com/zarr-developers/zarr-python | MIT (LICENSE.txt) | 2026-11-14 (HEAD d8b8543); PyPI 3.4.2 2026-10-27 | — | pip + import OK on 3.13; wrote a v3 complex64 array | (owner's choice) | Pure Python; requires Python >= 3.12 |
| z5 / z5py | github.com/constantinpape/z5 | MIT (LICENSE) | 2026-11-15 (HEAD 29d17f6); PyPI z5py 3.0.2 2026-07-02 | — | cloned; `pip install z5py` OK on 3.13; read back a zarr-python v3 float32 array correctly | (owner's choice) | C++ zarr v2/v3 + n5; v3 removed xtensor. **No Linux aarch64 wheel on PyPI** (x86_64, macOS arm64, Windows only); C++ library not built here |
| TensorStore | github.com/google/tensorstore | Apache-2.0 (LICENSE) | 2026-11-13 (HEAD c6a9132); PyPI 0.1.86 2026-10-26 | Google project | pip + import OK on 3.13; opened the zarr-python v3 array with the `zarr3` driver | (owner's choice) | C++ core with `zarr`, `zarr3`, `n5` drivers; aarch64 wheels on PyPI; C++ build (Bazel/CMake) not attempted |
| GoogleTest | github.com/google/googletest | BSD 3-clause (LICENSE) | 2026-11-13 (HEAD e882c36) | — | apt `libgtest-dev` 1.14.0 installed; one test built and run | depend (tests only) | — |
| pybind11 | github.com/pybind/pybind11 | BSD-style 3-clause (LICENSE) | 2026-11-13 (HEAD c68d56a); PyPI 3.1.0 | — | cloned; pip + import OK; apt has 2.11.1 | depend (one of the two) | — |
| nanobind | github.com/wjakob/nanobind | BSD-style 3-clause (LICENSE) | 2026-11-01 (HEAD af676a3); PyPI 3.1.0 | — | cloned; pip + import OK | depend (one of the two) | Requires Python >= 3.10 |
| yaml-cpp | github.com/jbeder/yaml-cpp | MIT (LICENSE) | 2026-11-08 (HEAD f473f3b) | — | cloned; apt candidate 0.8.0 (not installed, not built) | depend (config files) | — |
| Eigen | gitlab.com/libeigen/eigen | **MPL-2.0** primarily (COPYING.README), with BSD/Apache/MINPACK parts | 2026-11-15 (HEAD dc57ada) | — | cloned; apt `libeigen3-dev` 3.4.0 installed and used in builds here | depend (header-only) | MPL-2.0 is file-level copyleft; keep unmodified |
| pocketfft (C++) | github.com/mreineck/pocketfft, branch `cpp` (default) | BSD 3-clause (LICENSE.md, Max-Planck-Society) | 2026-10-18 (HEAD 4d865cb) | — | cloned; single-header c2c FFT **built and run here** | vendor (CPU FFT) | One header, `pocketfft_hdronly.h`; no PyPI package named `pocketfft` |
| KISS FFT | github.com/mborgerding/kissfft | BSD-3-Clause (COPYING) | 2026-11-15 (HEAD 57a7017) | — | cloned only; apt `libkissfft-dev` candidate 131.1.0 | reject (pocketfft preferred) or fallback | — |
| FFTW | github.com/FFTW/fftw3 | **GPL-2.0-or-later** (COPYING) | 2026-10-17 (HEAD a7b679c) | — | cloned only; apt candidate 3.3.10 | reject for core; test-oracle only with sign-off | pyFFTW's own wrapper is BSD-3 per PyPI but it still links FFTW |
| cuFFT | NVIDIA CUDA toolkit | proprietary: PyPI `nvidia-cufft-cu12` metadata says `LicenseRef-NVIDIA-Proprietary`; EULA text unverified | PyPI 11.4.1.4 | — | in use since D-42: the toolkit's `libcufft` (11.x with CUDA 12), run on T4 and A10G | depend (GPU FFT, dynamic link, CUDA build only) | Baseband conversion and pulse compression; pocketfft stays the host FFT |

### 2g. Visualisation (since 2026-11-26)

| Project | URL | Licence (as read) | Last activity | Validation evidence | Verified here | PROPOSED role | Notes |
|---|---|---|---|---|---|---|---|
| three.js | github.com/mrdoob/three.js; npm `three` | MIT (LICENSE in the npm package; SPDX header `MIT` in `build/three.module.min.js`) | npm `three@0.170.0` (release 170) used; newer releases exist (not checked) | very wide use (WebGL scene library) | `npm pack three@0.170.0` into the session scratchpad; `build/three.module.min.js` (691,648 bytes, one self-contained ES module) loaded in headless Chromium (Playwright) and drawn: the 3-D prototype pages of `docs/tasks/viz_prototype/` | optional viewer library, **never** imported by the suite: loaded by the HTML pages `gpusonar.viz.page3d` writes, from a pinned CDN URL (`cdn.jsdelivr.net/npm/three@0.170.0`) or embedded (`--three embed`, the default when the vendored copy exists, or `--three PATH`; offline) | The owner approved a viewer dependency on 2026-11-26 ("dependency is fine"). **Vendored since 2026-11-27** (package V4): `third_party/three/three.module.min.js` unmodified, with its `LICENSE` and a README (version, source, SHA-256 `08fd7545…c39e7`); fetched from the npm registry's tarball (its SHA-512 equals the registry's `dist.integrity`), because jsdelivr and cdnjs refused this machine's request (HTTP 403 from the proxy). No add-on vendored: the page's camera controls are ours (section 3.14) |
| Plotly (Python) | github.com/plotly/plotly.py | MIT (wheel metadata `License-Expression: MIT`) | PyPI 7.2.0 | wide use | wheel downloaded and unpacked in the scratchpad only, not installed: 40.4 MB unpacked, needs `narwhals`; `plotly.min.js` inside it is 4,815,814 bytes | not used (rejected for the 3-D viewer, section 3.14) | An embedded page would carry the 4.8 MB script; the CDN form needs the network like three.js |

## 3. Notes on the non-obvious items

### 3.1 Holoscan SDK 5.0 is a breaking release with a narrow platform matrix

Read from `docs/sdk_installation.mdx`, `docs/python_installation.mdx` and `docs/holoscan_5_highlights.mdx` at HEAD 358c772:

- 5.0 "redesigned the API instead of extending 4.x"; there are `migrating_from_4x` guides. Code or tutorials written for 4.x do not carry over.
- Requires **CUDA 13**, Ubuntu 24.04 / glibc 2.39, **Python >= 3.12**, and CMake >= 4.0 for source builds (Ubuntu 24.04 ships 3.28).
- Supported platforms table lists x86_64 workstations, Jetson AGX Thor, Jetson AGX Orin / Orin Nano on JetPack 7.2.1,
  IGX Thor, Grace Hopper/Blackwell. "IGX Orin (IGX OS 1.x) and JetPack 6 are not supported."
- PyPI: `holoscan` 5.0.0 is a meta-package -> `holoscan-cu13` -> `holoscan-core-cu13`, `-vision-`, `-visualization-`, `-inference-`.
  `holoscan-core-cu13` 5.0.0 ships `cp312-abi3-manylinux_2_39` wheels for **x86_64 and aarch64**. The aarch64 wheel was downloaded here (9.2 MB).
- The last 4.x line: `holoscan-cu12` 4.6.0 (2026-09-01) has cp310–cp313 `manylinux_2_35` wheels for x86_64 and aarch64, and requires `cupy-cuda12x`.
  There is no `holoscan-core-cu12`; a CUDA 12 system stays on 4.6.0.
- Import test here: `import holoscan` fails with `ImportError: libcudart.so.13` — expected without CUDA, so nothing beyond packaging was verified.

### 3.2 holoscan-ultrasound-imaging

Very young repository (8 commits, first on 2026-09-16). README licence section: NVIDIA source is Apache-2.0, but
`fpga/build/ip` is under an Altera licence "including its Altera-product-only use restrictions", and optional MimickNet weights
are CC BY-NC 4.0 (not in the repo). `pyproject.toml`: Python >= 3.10, extras `cupy-cuda12x` or `cupy-cuda13x`. Do not copy anything from `fpga/`.

### 3.3 Licence ambiguities in the ultrasound toolboxes

- **ultraspy**: the LICENSE file is the MIT text but still carries the packaging-tutorial placeholder
  "Copyright (c) 2018 The Python Packaging Authority". PyPI metadata has no licence field. Fine as an oracle; ask upstream before porting code.
- **PyMUST**: LICENSE file = LGPL-2.1, `setup.py`/PyPI = "LGPL v3", function headers = "LGPL-3.0-or-later". Treat as LGPL (strictest reading) either way.
- **MUST** itself could not be reached (biomecardio.com blocked). The only evidence is the header text reproduced in PyMUST.
- **USTB** is now on GitHub under `unioslo/USTB` with an MIT licence file and a pip-installable Python package; the older Bitbucket location is still referenced in its README.

### 3.4 PICMUS: where the data can be fetched, and the reference numbers

- Original host `creatis.insa-lyon.fr` and the USTB data server `ustb.no` are blocked here. USTB's CI fetches its `.uff` copies from `zenodo.org/records/20261898` — also blocked.
- **Reachable mirror: Hugging Face `zeahub/picmus`** (public, not gated, 16 files, 0.65 GB): simulation and experiment
  `resolution_distorsion` and `contrast_speckle`, plus in-vivo carotid cross/long, each as IQ and RF HDF5.
  A second, smaller mirror exists at `Rith-P/picmus` (no licence tag, no in-vivo).
- These are **conversions to the zea HDF5 layout** (`tracks/track_0/data/raw_data`, `scan/...`), not the original PICMUS HDF5 files.
  Bit-equivalence with the originals was not checked.
- Terms, as quoted on the card and inside the file attributes: free of use; cite PICMUS and the proceedings paper
  (Liebgott, Rodriguez-Molares, Cervenansky, D'hooge, Bernard, IEEE IUS 2016, doi 10.1109/ULTSYM.2016.7728908).
- **Published reference numbers (FWHM, contrast/CNR for 1/11/75 plane waves) could not be sourced.** doi.org, IEEE Xplore, HAL,
  Crossref and Semantic Scholar are all blocked, the paper is not on arXiv, and none of the cloned repos contain the table.
  No numbers are quoted in this document. Fallback until the owner supplies the paper: compare against USTB/vbeam DAS output on the same data.

### 3.5 arlpy

- README (HEAD 2627a8c): "This package is now in maintainence mode and not being actively developed"; the author points Bellhop users to the Julia packages.
- `setup.py` pins `numpy<2.0.0` and `scipy<1.14.0`. On Python 3.13 pip therefore tries to build SciPy from source and fails:
  `ERROR: Unknown compiler(s): [['gfortran'], ...]`. With `--no-deps` on numpy 2.2.6 / scipy 1.15.3, `arlpy.uwa`, `arlpy.bf` and `arlpy.uwapm` imported and the calls tried worked.
- Contents confirmed: `uwa.py` (soundspeed, absorption, absorption_filter, density, reflection_coeff, doppler, bubble models, spl),
  `bf.py` (plane-wave steering, delay_and_sum, bartlett, capon, music, broadband), `signal.py`, `comms.py`, `uwapm.py` (Bellhop wrapper).
- The wrapper shells out to a literal `bellhop.exe` on PATH (`uwapm.py` line 887); `pm.models()` was `[]` without it and `['bellhop']` with it.
  Both the Fortran `bellhop.exe` built here and a symlink `bellhop.exe -> bellhopcxx2d` were accepted and returned arrivals for the default environment.
  The two returned different arrival counts (36 vs 68) for that default environment; this was **not investigated**.

### 3.6 Bellhop / Acoustics Toolbox build

- `gfortran` is **not installed** (apt candidate 4:13.2.0-7ubuntu1). To avoid a system install, the gfortran-13 `.deb` packages were
  downloaded with `apt-get download` and unpacked into the scratch directory, with a wrapper script supplying `-B`/`-L` paths. That worked.
- Acoustics Toolbox Makefile at HEAD has `FFLAGS= -mcpu=apple-m2 ...` active, which fails on x86_64:
  `f951: Error: bad value 'apple-m2' for '-mtune=' switch`. Replacing it with `-march=native` built everything:
  `bellhop.exe`, `bellhop3d.exe`, `kraken.exe`, `krakenc.exe`, `bounce.exe`, `field.exe`, `field3d.exe`, `scooter.exe`, `sparc.exe`.
- bellhopcxx (CPU): `cmake -DBHC_ENABLE_CUDA=OFF` + `make` built `bellhopcxx`, `bellhopcxx2d`, `bellhopcxx3d`, `bellhopcxxnx2d` and a shared/static library with the stock cmake 3.28.
- Cross-check run here: `MunkB_Coh` (Munk profile, coherent TL, 50 Hz). Fortran BELLHOP 3.8 s, bellhopcxx2d 0.31 s (multithreaded);
  the two `.shd` files have identical size and the pressure fields agree to a relative RMS difference of 8.4e-5 (max abs diff 4.2e-6 against a max magnitude of 9.5e-3).
- Standard test cases in the Acoustics Toolbox tree: `tests/Munk/` (MunkB_* Bellhop, MunkK Kraken, MunkS Scooter), `tests/MunkLeaky`, `tests/MunkRot`, `tests/MunkTS`,
  `tests/PekerisRD/`, `tests/Noise/Pekeris`, `tests/TLslices/pekeris.env`, `tests/Bellhop3DTests/Seamount/` (a cone) and `tests/Bellhop3DTests/DoubleSeamount/`, plus `tests/Dickins`, `tests/wedge`, `tests/arctic`, `tests/SBCX`, `tests/3DAtlantic` and others.
  bellhopcuda carries its own copies under `test/in/`.
- The bellhopcuda README mentions an open issue with 3D runs in complex environments that is also present in the Fortran original.

### 3.7 Open SAS code and data

GitHub search for "synthetic aperture sonar" returned 13 repositories; none is a complete, licensed SAS processing chain.
The useful ones are in table 2d. Searches for "micronavigation sonar", "displaced phase center sonar", "sonar backprojection" and "ASASIN sonar" returned zero repositories.

Hugging Face has the ApertureLab synthetic SAS datasets (author Isaac Gerg, also first author of the ASASIN paper). They contain beamformed
images, DEMs and COCO labels, **not raw element data**, so they cannot exercise a beamformer. The licence (ALSD-1.0) defines an "Excluded Party" list
(Kitware, NSWC PCD, two US Navy programme offices, and anyone working on their behalf) who receive no rights. Needs a legal read before any use.

No openly reachable raw SAS ping dataset was found from this container. AirSAS (in-air) raw data is described as being on figshare, which is blocked.

### 3.8 Ceres and SuiteSparse

Ceres' own licence is BSD-style. The Ubuntu package `libceres-dev` depends on `libsuitesparse-dev`, and a program linked against it here
pulled in `libcholmod.so.5` and `libspqr.so.4`. The Debian copyright file lists `CHOLMOD/Supernodal/*` and `SPQR/*` as **GPL-2+**.
Upstream Ceres has `WITH_SUITESPARSE` defaulting to OFF and a `WITH_EIGENSPARSE` option. For a permissive core: build Ceres from source without SuiteSparse
(the DPC/motion problems are small and dense or banded), or get sign-off for the distro package.

### 3.9 HoloOcean

The main repository does not resolve anonymously. The project docs (v2.4.0 installation page) state that access needs a "GitHub account that is linked to an
Epic Games account" and that the simulator is "Not distributed publicly online due to the Epic Games End User License Agreement (EULA)".
The PyPI package is an old client (0.5.8, 2024) whose metadata splits the licence into MIT (BYU client code) and the Unreal EULA (engine code).
Whether there are additional non-commercial terms could not be checked (EULA host not reachable). Proposed: never a dependency; at most an external data generator after the owner reads the EULA.

### 3.10 Container library facts (no recommendation)

| | HDF5 route | Zarr route |
|---|---|---|
| Core licence | HDF Group licence (BSD-style); HighFive BSL-1.0; h5py BSD-3 | zarr-python MIT; z5 MIT; TensorStore Apache-2.0 |
| C++ access | Mature C library; HighFive header-only wrapper built and round-tripped here | No reference C++ library from the Zarr project. z5 (v3 release, small project) and TensorStore (large Google codebase, Bazel/CMake) both read a zarr-python v3 array here via their Python wheels; neither C++ API was built here |
| Python | h5py 3.16.0, wheels cp310+ x86_64/aarch64 | zarr 3.4.2 pure Python (>= 3.12) |
| aarch64 | h5py aarch64 wheels on PyPI; `libhdf5-dev` for arm64 unverified here | TensorStore aarch64 wheels on PyPI; **z5py has no Linux aarch64 wheel** (source/conda build needed, unverified) |
| apt (noble) | `libhdf5-dev` 1.10.10 (old 1.10 line); no HighFive package in the sources here | none found |
| Seen in this survey | PICMUS mirror, AirSAS, USTB `.uff` are all HDF5-based | — |

### 3.11 Python 3.13 results in one place

Failed: `arlpy` (dependency pins; 3.5). Not on PyPI: `RITSAR`, `pocketfft`. Needs a manual extra: `vbeam` (install `jax`).
Imported fine on 3.13: acoular, pyroomacoustics, stonesoup, gtsam, h5py, zarr, z5py, tensorstore, pymust, k-wave-python, ultraspy, ustb, pyuff-ustb, sarpy, pybind11, nanobind, pyram, holoocean (client only).

### 3.12 Sources used for the SST-equivalence simulator work (D-62 to D-79)

Added after the survey above, when the simulator's models were extended. Nothing in this
list is linked, vendored or installed as a dependency; the papers are sources of
formulas and table rows, each cited where it is used in the code.

**Papers read (arXiv, opened through the abstract and HTML pages).** The owner's ruling
for this work: "Use anything from arxiv that cites things you need".

| Paper | What was taken | Where |
|---|---|---|
| Lavia, Gonzalez and Blanc, "Modelling High-Frequency Backscattering from a Mesh of Curved Surfaces Using Kirchhoff Approximation", arXiv:1805.06945v2 (2018) | eq. (1) the Kirchhoff backscatter integral; eq. (5) the sphere; eq. (6) the finite cylinder without caps (they cite Gaunaurd 1985); eq. (7) the disc end cap; TS = 20 log10\|f\| | `sim/targets.py` (D-65). Eq. (6)'s phase as read differs from eq. (1) by a quarter turn; not settled |
| Olson and Jackson, arXiv:2009.12420 (2020), eqs. (13)–(16), (20) | Jackson's perturbation factor for a fluid half-space, as a check only: our own derivation matches it to 1e-10 | `tests/python/test_bottom_models.py` (D-66) |
| Olson, arXiv:2106.08287 (2021), eqs. (13), (24), Sec. II.B | spectrum convention and the form of the 2-D Kirchhoff integral; the statement that Kirchhoff and small-slope share it | `sim/bottom.py` (D-66) |
| Olson, Lyons and Sæbø, arXiv:1601.06078, Sec. III and Table II | only the statement that first-order small-slope is a geoacoustic factor times the Kirchhoff integral; the "generic rock" elastic values are recorded in a YAML comment and not used | `sim/bottom.py`, `configs/sediments.yaml` (D-67) |
| Brown, Johnson and Brownstead, arXiv:1809.08459, Table 1 (values they attribute to TR 9407) | two sediment rows, six numbers each: medium sand and very fine silt | `configs/sediments.yaml` (D-67) |
| Bentrem, Sample, Kalcic and Duncan, arXiv:physics/0607146, eq. (7) and Table II | the centimetre form of the roughness spectrum, as the basis of a unit conversion; definitions of the six parameters | `bottom.w2_from_cm4` |
| Nennouche, Khalighi, Dowhuszko and Merad, arXiv:2603.23727v2 (2026), eqs. (20)–(23) (they cite Keiffer, Novarini and Norton, JASA 97, 1995) | the Hall–Novarini bubble population and its constants. The unit of n is not printed there; m⁻⁴ was assumed | `sim/surface.py` (D-68) |
| Morency and Stilwell, arXiv:2210.06535 (2022), Sec. II-E, II-F (they quote D. Kraus's lecture notes) | an empirical surface backscatter law and the SEARAY bottom law with types 1–4; no validity range is given | `sim/surface.py` (D-68) |
| arXiv:1611.00608 Table 1, arXiv:2003.11156 Tables 1–2 | looked at, nothing taken (sound speed, density and attenuation only) | — |
| APL-UW TR 0702 (Goddard 2008), the local extract in `docs/refs/` (not committed) | equations and statements as cited in `PHYSICS_NOTES.md`; the only source for the reverberation, waveform, spreading and validation work | throughout |

**Not read.** APL-UW TR 9407 (limited distribution; still not fetched or read). Jackson,
Winebrenner and Ishimaru 1986; Mourad and Jackson 1989; Williams and Jackson 1998; Hall
1989; Hamilton's regressions; Urick's table of simple target shapes; Dahl's ASIAEX
papers; McDaniel's wind laws. Nothing is taken from any of them; where the code needs
what they contain it uses a derivation of its own or says what is missing
(`STATUS.md`, open question 8).

**arlpy as a comparison oracle only.** `tools/boundary_models_study.py` imports
`arlpy.uwa` (BSD 3-clause, see 3.5) if it is installed and prints
`arlpy.uwa.bubble_surface_loss` (its docstring: "APL model (1994)") beside our bubble
law. That is the only use: a study tool, optional, skipped with a message when arlpy is
absent. The core does not import arlpy, its formula was not read or ported, and no value
from it is used by any code. What the repository holds from it is comparison text only:
the ratios of our law to its output in the `surface.py` docstring, and the printed dB
values in `RESULTS.md`. Whether calling a BSD implementation of a handbook law as an oracle is
acceptable is an owner's question (`STATUS.md`, open question 9).

**Holoscan Sensor Bridge 2.7.0: an optional patch to its board emulator.** The survey row
in 2a was made at the repository's HEAD (VERSION 3.0.0, Apache-2.0 per its LICENSE;
`third_party_licenses/` present, not audited); `tools/build_hsb.sh` builds release 2.7.0
against Holoscan 4.6.0 as a separate tree outside this repository. Nothing of Sensor
Bridge is vendored or linked into the core.
`third_party/patches/hsb-2.7.0-emulator-udp-socket.patch` (D-79) is a diff of our own
change to three host C++ files of the emulator
(`src/hololink/emulation/src/linux/rocev2_data_plane.hpp`, `rocev2_data_plane.cpp`,
`rocev2_transmitter.cpp`; 27 changed lines), carrying the few context lines a diff
needs. It is applied at build time to that separately built third-party tree, and only
when `HSB_UDP_PATCH=1`; the patched emulator behaves as stock unless
`HSB_EMULATOR_UDP_SOCKET=1` is set when it runs. The patch file's header states what it
changes and why. The licence of release 2.7.0 itself was not re-read for this note; the
Apache-2.0 statement is the survey's, at HEAD.

### 3.13 Sources used since decision D-82: code read, data used, documents opened

Added 2026-11-18. As in 3.12, nothing in this list is linked, vendored or installed as a
dependency of the core. Where a licence was not read by us or is not stated in the
repository's own documents and work-package reports, the entry says "licence not recorded
— to check" instead of a guess. Physics sources (papers, theses, the tables and equations
taken from them) are registered in `docs/REFERENCES.md`; the org-arl repositories are
surveyed file by file in `docs/ORG_ARL_SURVEY.md`. This section is about licences and how
each thing is used.

**Open-source code whose formulation was read and re-implemented (no code copied).**

| Project | Licence (as read) | What was taken, and where it lives | How it is used at run time |
|---|---|---|---|
| arlpy (`github.com/org-arl/arlpy`, commit `2627a8c`, installed 1.9.3) | BSD 3-clause (LICENSE; file headers say "Simplified BSD") | `uwa.bubble_surface_loss`: the formula of the `apl_1994` bubble-loss law, adopted on the owner's ruling (`sim/surface.py`; `DECISIONS.md` D-123), with the docstring's own attribution ("computed based on APL model (1994)"). `signal.goertzel`: the second-order recurrence and its output convention (`gpusonar/tonal.py`, `libs/live` tonal; D-130). Sea-water density at one atmosphere and Wood's equation (`REFERENCES.md` Sec. 7) | not imported by the core. An optional oracle in `tools/boundary_models_study.py`, `tests/python/test_tonal.py` and the medium-model tests, skipped when absent |
| UnderwaterAcoustics.jl (org-arl, v0.9.0, commit `7b9c47e`) | MIT (`LICENSE.txt`) | the 22-row seabed table it attributes to TR 9407 (density, sound speed, loss parameter), used for the reflection-only sediment rows with that attribution (`configs/sediments.yaml`; D-132); the same bubble-loss law with its more exact attribution; the sea state to wind speed table; the sea-water density equation; Hamilton's shear regression as data only | none. Julia could not be run here: "their" numbers in the survey are hand evaluations of their printed formulas |
| SignalAnalysis.jl (org-arl, commit `1aa1bd5`) | MIT (`LICENSE.txt`) | `goertzel` and `demon` in `src/dsp.jl`: the same recurrence; the structure of the DEMON estimator (envelope reduced to a per-block figure, a 1 Hz high-pass) (D-130) | none |
| NoiseModels.jl, AlphaStableDistributions.jl (org-arl) | MIT | read for what an alpha-stable and a fitted-covariance noise generator need; our fitted noise uses a closed-form spectral factor instead of their optimiser, and alpha-sub-Gaussian noise with memory is not implemented (D-124) | none |
| AcousticsToolbox.jl, AcousticRayTracers.jl, DataDrivenAcoustics.jl, VirtualAcousticOcean.jl, AdaptiveEstimators.jl, UnetUtils.jl, ARLToolkit.jl, unet-doc, unet-contrib, fjage and related (org-arl) | MIT, except `src/bellhopjl/` and `src/krakenjl/` of AcousticsToolbox.jl, which are GPL-3.0-or-later; unet-contrib BSD 3-clause | surveyed only (`ORG_ARL_SURVEY.md`); nothing taken. The GPL directories were not read | none |
| UACR toolboxes (`github.com/uwa-channels`: Python `uwa_channels`, MATLAB twin) | MIT (LICENSE read; commits in the survey) | README, citation file and user-guide pages read; code not read. Our reader of the channel files is `sim/channel_replay.py`, written from the data format | none |
| `ahmd-mahm/alpha-SGNm` (MATLAB) | BSD 3-clause (LICENSE read) | file list only; nothing taken | none |

The owner's ruling that makes the handbook-attributed values usable: a value printed in an
open (MIT or BSD) file or an open paper, with that file's attribution to APL-UW TR 9407,
may be used with the attribution; the handbook itself stays unread and was not downloaded.
This is a practical reading of the licences, not legal advice (`ORG_ARL_SURVEY.md` Sec. 5).

**The Acoustics Toolbox (Bellhop, Bellhop3D, KRAKEN): a separate process only.**
`github.com/oalib-acoustics/Acoustics-Toolbox`, GPL-3.0 (row in 2c, build in 3.6). Since
D-82 it is used for three more things, all through the same boundary: our code writes
Bellhop's input files, runs `bellhop.exe` as an external program
(`GPUSONAR_BELLHOP`), and reads its output files. Nothing of it is linked, vendored, or
ported.

- Range-dependent sound speed (D-110): Bellhop's SSP option `Q` and its `.ssp` file. The
  **file format** was read from the toolbox's documentation page
  (`doc/RangeDepSSPFile.htm`) and from the reader that parses it (`Bellhop/sspMod.f90`,
  subroutine `Quad`), to write a file it accepts. No code was copied or translated; what
  our provider contains is a writer for that format. Reading GPL source to learn an input
  format is noted here so the owner can object.
- A test oracle for the chain's own ray code (`libs/live` `raytrace.hpp`,
  `python/gpusonar/raytrace.py`), which is our derivation from closed-form arcs in a
  layered profile (`PHYSICS_NOTES.md` Secs. 25, 28); the reports of that work state that
  no Bellhop code was read or copied for it.
- A test oracle for the surface-image factor (`lloyd.py`) and the coverage picture.
- **Bellhop3D as the program of the opt-in `eigenray3d` tier** (2026-11-26; the owner approved it
  as an opt-in provider run as a separate process, `docs/tasks/PROPAGATION_3D.md` question 1).
  `python/gpusonar/sim/bellhop3d.py` writes its `.env` and `.bty` files, runs `bellhop3d.exe`
  (`GPUSONAR_BELLHOP3D`, or beside `GPUSONAR_BELLHOP`) in a temporary directory, and reads its
  arrivals (ASCII and binary) and TL output. The **formats** were read from the toolbox's
  `doc/bellhop3d.htm`, `Bellhop/bdry3DMod.f90`, `Bellhop/ReadEnvironmentBell.f90` and
  `Bellhop/ArrMod.f90` (format only); the merge of its beam contributions into eigenrays and
  their scaling to its TL run are ours. Nothing is linked, vendored or ported. Before this it was
  a test oracle only (D-149, `tests/python/bellhop3d_oracle.py`, which now imports the writers).

A cause of a delay error inside a sharp front (interpolation of delay between ray steps)
is inferred from Bellhop's output, not read from its code (`STATUS.md`, open item 26).

**Measured data: fetched at test time, never committed; only derived tables and figures
are in the repository.**

| Data set | Licence | What is used | What the repository holds |
|---|---|---|---|
| Underwater Acoustic Channel Repository (UACR) v1.0, Zenodo doi:10.5281/zenodo.21287414 (cite arXiv:2609.03207 with it) | CC BY 4.0 | measured time-varying channels: the 12.5 kHz "Purple" site (nine files) and the repository's other sites, for the time-spreading validation (`PHYSICS_NOTES.md` Sec. 21) | `tools/fetch_channels.py` with an MD5 table; `docs/figures/channel_validation_{purple,sites}.md` and four `channel_validation_*.png` figures derived from the data, attributed in the legend. Tests that need the files skip without `GPUSONAR_UACR_DIR` |
| Watermark benchmark channels (FFI), a 770 MB archive served as a free download from `www.ffi.no` | **licence not recorded — to check.** The archive carries no licence text, and its user manual could not be reached. Nothing is redistributed | three of its five channels (NOF1, NCS1, BCH1), for tests that need no geometry: Doppler shape, scintillation, width against frequency inside a band | `tools/fetch_channels.py --watermark` with a recorded MD5; `tools/channel_watermark.py`; `docs/figures/channel_validation_watermark.md` and `.png` (derived). The two papers read for what the set states are hal-01630274 and hal-01404491; their licences are not recorded — to check |
| SFI Smart Ocean dataset (van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh; IEEE DataPort 2025, doi:10.21227/3aa6-4k33) | CC BY 4.0 | the 4 to 8 kHz receptions (160 recordings at two hydrophones), the transmit waveforms, two sound-speed casts and the depth chart, supplied to the session after a login (IEEE DataPort serves the files only to logged-in users): soundings, the basin reverberation fit (D-106), the facet tail and coherent-phase comparisons (D-107, D-108), the noise levels | `tools/channel_smartocean.py`, `tools/basin_reverberation_fit.py`; `docs/figures/channel_validation_smartocean*.{md,png}`, `basin_reverberation_fit.{md,png}`, `facet_tail_smartocean.md` (derived, with the attribution in each file's first lines). Two fitted parameter sets are in `sim/basin.py` (`MEASURED`), labelled "one fjord, one night, 4-8 kHz". The descriptor article was read as the open copy hal-05793814; its licence is not recorded — to check |

The plain-language statement on a derived table: these are our analyses of the authors'
recordings, published under CC BY 4.0 with attribution; the numbers are ours, not figures
the authors report, except where a row says "stated by the dataset".

**Documents opened in the 2026-11 literature search** are registered one by one in
`REFERENCES.md` (access, what was taken, status), with the licence or terms as each host
stated them. In summary: US Government theses and reports from calhoun.nps.edu (public
domain, as the register records them); author copies of journal and conference papers
from repository.tudelft.nl (licence not recorded per paper — to check); arXiv papers (the
arXiv licence chosen by each author; not recorded per paper — to check); Frontiers
articles and the GeoHab backscatter report (CC BY 4.0, as the register records them). The
documents themselves are outside the repository; only cited numbers and formulas are in
it. Hosts that refused access are listed in `REFERENCES.md` Sec. 9; nothing was done to
get around any of them. APL-UW TR 9407 was not downloaded or read.

**Manufacturers' data sheets** for projectors and amplifiers
(`docs/design/projector_selection.md`, `projector_candidates.json`): figures are quoted
from the makers' public data sheets as named in that document. The terms under which each
sheet is published are not recorded — to check before the document is circulated outside
the project.

**No new software dependency** was added since D-82. The ray code, the detector's
statistics, the coverage picture, tonal analysis, noise fitting and the environment loader
are written here. Eigen became a private build dependency of `libs/detect` as well (it was
already required; MPL-2.0, header only, row in 2f).

### 3.14 The 3-D scene viewer: three.js in the page, no Python dependency (2026-11-26)

`docs/tasks/VISUALIZATION.md` section 5. Checked first: none of plotly, pyvista, vispy, trimesh, k3d, pythreejs,
pygltflib or vtk is installed here. Chosen: the page is written by our own code (`python/gpusonar/viz/page3d.py`,
standard library only) and drawn by three.js in the browser, so one mechanism serves the interactive view (open
the page) and the export (the page is the export). Rejected: Plotly (MIT, 40.4 MB installed plus `narwhals`, 4.8 MB
of script in every embedded page, animation by frames only); pyvista/VTK (BSD-3, large binary wheels, a desktop
window, HTML export through a further package; not measured); k3d and pythreejs (MIT/BSD, notebook widgets, not a
standalone page). A page that loads three.js from the CDN is 14 to 65 kB for the two prototype scenes and needs the
network; embedded, 706 and 756 kB, and works offline. Vendoring the one file (0.69 MB, MIT) under a future
`third_party/` with its LICENSE would make offline pages reproducible; not done in this package.

Done in package V4 (2026-11-27): the one file is vendored under `third_party/three/` (README: version 0.170.0, the
source tarball and its integrity hash, SHA-256 of the file, "used only by generated pages"). It is read only by
`gpusonar.viz.page3d.write_page` when a page embeds it (`tools/scenario_view.py --3d --three embed`, the default
when the file exists); no module of the package imports it, and `tests/python/test_scene_3d.py` checks that only
`viz/page3d.py` names it. The orbit, pan and zoom controls are our own (about 20 lines in the page), so no three.js
add-on (`examples/jsm/controls/OrbitControls.js`, also MIT) is needed. Page sizes (ours, the 96 pages of 2026-11-27):
embedded 0.74 to 0.89 MB; loading from the CDN 44 to 200 kB. Limits held by the tests: 1.2 MB and 400 kB.
The optional browser check (`tools/page3d_check.js`) uses the Playwright package and its Chromium where they are
installed (not a dependency of the suite: the test skips without them).

## 4. ASASIN coverage map

**No public ASASIN code or data was found.** How this was checked: (a) the arXiv abstract page for 2101.05888 has no code link and its comment field is
"9 pages, 9 figures, submitted to MTS/IEEE OCEANS. Eq 11 fixed"; (b) the full PDF text was searched for github, open source, available, release, download, dataset and licence —
no release statement, and the test data is only described as "a small SAS dataset representative of typical compute loads"; (c) GitHub repository search for "ASASIN sonar" returned 0 results;
(d) Hugging Face dataset search found only the synthetic image sets in 3.7. Not checked: the authors' institutional pages and IEEE Xplore (blocked).

| ASASIN part (as described in the paper) | Open projects that cover it | Open projects that can cross-check it | Gap |
|---|---|---|---|
| Time-domain backprojection on GPU (2D and 3D) | none for sonar. SAR analogues: gotcha-back (BSD-3, CUDA), torchbp (MIT, CUDA), RITSAR (MIT, NumPy) | vbeam / USTB / ultraspy DAS on PICMUS (same delay-and-sum kernel, different geometry); RITSAR on its bundled Gotcha subset; point-target simulation written in-house | No open raw SAS data reachable; sonar-specific geometry (bistatic, multi-receiver, sound-speed) must be written and tested on simulated data |
| DPC micronavigation (ping-to-ping delay estimation) | none found (zero search hits) | k-Wave / k-wave-python or PyMUST `simus` to synthesise overlapping-phase-centre echoes with known motion (LGPL, separate process) | Entirely in-house; needs a simulator-based truth |
| Motion solve with Ceres (L_DPC + smoothness + DVL terms, Levenberg–Marquardt) | Ceres Solver (BSD; the paper's own choice) | GTSAM (BSD) as an independent solver on the same residuals | SuiteSparse licence in the distro build (3.8) |
| Spectral whitening, TVG, dynamic range compression | none as a library; formulas are in the paper | arlpy `uwa.absorption` / `soundspeed` (BSD) for the physical TVG term; NumPy/SciPy reference implementations written in tests | Small; port from the paper |
| Autofocus (not a headline ASASIN stage, but adjacent) | RITSAR `autoFocus` (MIT) | isaacgerg/synthetic_aperture_sonar_autofocus (no licence — read only), torchbp | — |
| GeoTIFF output (multi-layer for 3D) | GDAL (MIT-style) + PROJ (MIT-style) | `gdalinfo` / rasterio read-back in tests | None |
| Configuration-file framework and pluggable data readers | yaml-cpp (MIT) for config; HDF5+HighFive or Zarr for containers; marine_acoustic_msgs (BSD-3) as a field-naming reference; liboculus (BSD) for one FLS format | — | Reader base-class design is in-house |
| Performance model across GPUs | none | Holoscan `benchmark/` methodology (reading only) | — |
| Long-range propagation (not ASASIN; flank array) | none permissive except PyRAM (BSD-style) | BELLHOP / KRAKEN (GPL-3) and bellhopcxx (GPL-3) as separate processes — both built and cross-checked here on Munk | Copyleft sign-off |
| Adaptive beamforming (not ASASIN; FLS / flank array) | in-house | acoular (BSD-3) Capon/CLEAN-SC/DAMAS; pyroomacoustics (MIT) MUSIC/MVDR; arlpy `bf` (BSD) capon/music | — |
| Tracking (not ASASIN; flank array) | Stone Soup (MIT, Python) | same | C++ tracker is in-house if needed |

## 5. Not verified / blocked

| Item | Host | Result here |
|---|---|---|
| PICMUS original data and challenge terms page | www.creatis.insa-lyon.fr | blocked |
| USTB dataset server (`PICMUS_*.uff`) | www.ustb.no | blocked |
| USTB CI dataset copies | zenodo.org | blocked |
| Liebgott et al. IUS 2016 paper and its reference table | doi.org, ieeexplore.ieee.org, hal.science, api.crossref.org, api.semanticscholar.org | blocked; numbers not sourced |
| MUST toolbox and its licence | www.biomecardio.com, www.mathworks.com | blocked |
| OALIB originals (Acoustics Toolbox tarball, RAM) | oalib-acoustics.org | blocked; GitHub mirror used instead; RAM not verified at all |
| AFRL Gotcha data and terms | www.sdms.afrl.af.mil | blocked |
| AirSAS dataset and its licence | doi.org / figshare.com | blocked |
| ApertureLab simulator fidelity page | gergltd.com | blocked |
| Holoscan online docs, NGC containers, CUDA/cuFFT EULA text | docs.nvidia.com, catalog.ngc.nvidia.com | blocked; facts taken from the cloned repo docs and PyPI metadata |
| k-Wave site, Ceres site | www.k-wave.org, ceres-solver.org | blocked; repos used |
| arXiv API (`export.arxiv.org`) | export.arxiv.org | blocked; arxiv.org abstract, PDF and search pages worked |
| APL-UW TR 9407 and other DTIC reports (Urick's shapes among them) | apps.dtic.mil | HTTP 403; TR 9407 is limited-distribution and was not sought further |
| Classic boundary-scattering papers (Jackson, Winebrenner and Ishimaru 1986; Mourad and Jackson 1989; Williams and Jackson 1998; Hall 1989) | pubs.aip.org | allowed during the SST work, but answered with a browser challenge page (HTTP 403, "Just a moment..."); not circumvented |
| Open copies of the same papers and of Urick's table | core.ac.uk | first outside the allowed hosts; once allowed, the same browser challenge (HTTP 403); the fetch tool could not resolve it |
| IEEE papers (Hamilton's regressions were sought here too) | ieeexplore.ieee.org | once allowed, returned an empty HTTP 202; nothing read |
| Institute of Acoustics proceedings, HLS Research pages, FAO documents | www.ioa.org.uk, www.hlsresearch.com, www.fao.org | outside the allowed hosts; not requested further |
| GitHub REST repo endpoint, GitHub web search/topics | api.github.com (direct), github.com/search | not enabled / 403; `gh api search/repositories` worked and was used |
| HoloOcean main repository and Unreal EULA text | github.com/byu-holoocean/HoloOcean | not accessible anonymously (Epic account linking) |
| `apl-ocean-engineering/marine_acoustic_msgs`, `creatis-ULTIM/MUST`, `ustb/ustb` on GitHub | github.com | do not resolve; correct locations are `apl-ocean-engineering/marine_msgs` and `unioslo/USTB`; MUST has no GitHub repo that was found |
| Any CUDA execution (bellhopcuda, torchbp, gotcha-back, Holoscan, CuPy backends) | — | no GPU in this container |
| aarch64 behaviour of anything | — | x86_64 container; only PyPI wheel tags were checked |
| isce3, other SAR processors, yaml-cpp / kissfft / z5 C++ / TensorStore C++ builds | — | not attempted |
