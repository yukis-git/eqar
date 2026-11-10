# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Markers on a white board in front of a darker background were lost or mislocated: the
  bridge decided holes by containment parity, so when the background touched the patch edge
  its untraced outer border flipped the parity, and the white quiet zone's outline (or the
  board's) was taken for the marker. Holes are now decided by each contour's traced winding
  (`PatchContours::hole`). On synthetic 6528x3264 frames (DICT_4X4_50 on boards over
  60-210 DN backgrounds) detection at 40-100 px rose from 86.4 % to 99.1 % and at 20-40 px
  from 60.5 % to 66.9 % (before the small-pass change below), with no false positives.
- This deliberately departs from the C++ reference on four bridge-fixture scenes and three
  pipeline-fixture scenes (a marker on mid-gray), where the reference misses the marker. The
  parity tests check those cases against ground truth instead; every other case still
  matches the reference exactly. With the small-pass change, two pipeline steps also differ
  in `n_candidates` only (a marker dropped by `expect_id` / `in_use_ids`).

### Changed
- The f=2 small-marker pass (`use_second_scale`) now runs whenever at least 1 ms of the
  time budget is left after the f=4 candidates; it used to run only when f=4 had decoded
  nothing and produced no tier-1 candidate, so one larger marker in view hid every marker
  below ~24 px_eq. Candidates inside an already accepted marker are skipped, so f=4 results
  are never replaced. Same synthetic frames: 20-40 px detection 66.9 % -> 90.1 % (20-25 px
  50 % -> 83 %), 40-100 px 99.1 % -> 99.6 %, no false positives.
- Cost: the f=2 proposal runs before the budget check can stop it. A frame with one marker
  at the 2 ms default budget went from 0.94 to 1.35 ms (3840 wide) and 1.77 to 3.12 ms
  (6528 wide); an 8-marker one-shot detect at 6528 from 6.8 to 8.4 ms. `track` is
  unchanged. Set `use_second_scale=False` for the previous speed without the small markers.

### Added
- `erp_synth.paste_board` / `board_corner_rays`: perspective-exact physical boards (distance,
  tilt, yaw, board size) for the Python tests.

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
