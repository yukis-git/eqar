# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-08

First public release.

### Added
- Rust library `eqar`: ArUco detection directly in equirectangular images
  (`Detector::detect`, `track`, `verify`, `proposals`, `generate_marker`), all 16 OpenCV
  predefined dictionaries, great-circle edge fitting, polar-cap projections, NEON kernels on
  aarch64 (scalar elsewhere).
- Python package `eqar` (abi3, Python 3.9+): `detect`, `Detector`, `Marker`, `Prior`,
  `generate_marker`, type stubs, GIL released during detection.
- Test suite: unit and behaviour tests, cross-checks against fixtures produced by the C++
  implementation it was ported from, synthetic accuracy tests, Python tests.
- `bench/bench.py` comparing eqar with `cv2.aruco` on synthetic frames; results in
  `bench/RESULTS.md`.

### Notes
- Images narrower than 320 pixels are rejected (`EqarError::Size`).
- `FLAG_SEAM` is reserved and never set.
