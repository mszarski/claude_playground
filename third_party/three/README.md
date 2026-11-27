# three.js 0.170.0 (vendored, unmodified)

| | |
| --- | --- |
| Project | three.js, github.com/mrdoob/three.js (npm package `three`) |
| Version | 0.170.0 (release r170) |
| Licence | MIT (`LICENSE`, copied from the npm package; the file's own header says `SPDX-License-Identifier: MIT`) |
| File | `three.module.min.js` from the package's `build/` directory: 691,648 bytes, one self-contained ES module |
| SHA-256 | `08fd7545d13d2c7fb65ab691530a802dafefd638596501854f267d0fb13c39e7` (`three.module.min.js`) |
| Source | `https://registry.npmjs.org/three/-/three-0.170.0.tgz`, fetched 2026-11-27; the tarball's SHA-512 equals the registry's published `dist.integrity` (`sha512-FQK+LEpYc0fBD+J8g6oSEyyNzjp+Q7Ks1C568WWaoMRLW+TkNNWmenWeGgJjV105Gd+p/2ql1ZcjYvNiPZBhuQ==`). The same file is served at `https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js` (the URL a CDN page loads; that CDN refused this machine's request, so the registry's tarball was used) |

Used ONLY by the generated 3-D scene pages (`python/gpusonar/viz/page3d.py`, `tools/scenario_view.py --3d`): it is
embedded in an offline page (`--three embed`, the default when this file exists) and drawn by the browser. It is never
imported by the `gpusonar` package, the simulator, the processing or the live chains, and no test imports it (the
optional browser check `tools/page3d_check.js` loads pages that embed it). No three.js add-on is vendored: the camera
controls are our own code in the page.
