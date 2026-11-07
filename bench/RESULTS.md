# Benchmark results

Reproduce: `pip install eqar numpy opencv-python-headless` then
`python bench/bench.py --frames 100 --seed 1 --width 3840` (and `--width 1920`). The script
is `bench/bench.py`; it renders its frames with the pure-numpy projector in
`tests/py/erp_synth.py`, so no data files are needed. Numbers below were measured once, on
the machine described under each table; they are not a general speed or accuracy claim.

What is compared: `eqar` (default parameters, one `Detector`, single thread) against
`cv2.aruco.ArucoDetector` (OpenCV 5.0.0, one thread, `CORNER_REFINE_SUBPIX`) used
(a) directly on the equirectangular image with parameters tuned for it
(`adaptiveThreshWinSizeMax=63`, step 6, `minMarkerPerimeterRate=0.01`,
`polygonalApproxAccuracyRate=0.05`, `cornerRefinementWinSize=5`) and (b) on 20 pinhole views
(icosahedron face centres, 80 degree field of view, 960 x 960 px each, remapped from the frame;
corners mapped back to the sphere; one detection per id). The time of (b) includes the 20
remaps.

Frames: one DICT_4X4_50 marker per frame, random id, direction uniform on the sphere
(poles and seam included), random roll, half angle of the quiet-zone square 3, 4, 6 or 9
degrees, 3 x 3 supersampled edges, Gaussian blur sd 0.8 px, Gaussian noise sd 3 grey levels,
background 205. A detection counts when it has the right id and every corner is within
3 px_eq of the true corner. "Error" is, per detected marker, the mean over its four corners of
the angle between the detected and the true corner ray in equivalent pixels (2 pi / width rad);
the table gives the median and mean over the detected markers. Time is per frame:
thread CPU time (`time.thread_time`) and wall time, median over the frames. There were no
wrong-id detections in any run.

## 3840 x 1920, 100 frames

```
frames 100, 3840x1920, noise sd 3.0, blur sd 0.8, seed 1, hit = right id and every corner within 3.0 px_eq
machine: macOS-27.2-arm64-arm-64bit-Mach-O arm64, 10 cores, python 3.14.7, numpy 2.5.4, opencv 5.0.0, eqar 0.1.0
load average (1/5/15 min) before 14.3/21.7/21.3, after 12.7/20.0/20.6
| method | detected | wrong id | median err px_eq | mean err px_eq | cpu ms/frame (median) | wall ms/frame (median) |
|---|---|---|---|---|---|---|
| eqar | 99/100 | 0 | 0.0187 | 0.0211 | 1.67 | 1.67 |
| cv2-direct | 87/100 | 0 | 0.2150 | 0.2232 | 196.53 | 199.09 |
| cv2-20views | 99/100 | 0 | 0.2401 | 0.2479 | 107.65 | 110.00 |
```

Misses: `eqar` missed 1 frame (the smallest marker, half angle 3 degrees, at latitude 83);
`cv2-direct` missed 13.

## 1920 x 960, 100 frames

```
frames 100, 1920x960, noise sd 3.0, blur sd 0.8, seed 1, hit = right id and every corner within 3.0 px_eq
machine: macOS-27.2-arm64-arm-64bit-Mach-O arm64, 10 cores, python 3.14.7, numpy 2.5.4, opencv 5.0.0, eqar 0.1.0
load average (1/5/15 min) before 9.9/18.5/20.0, after 9.4/17.7/19.7
| method | detected | wrong id | median err px_eq | mean err px_eq | cpu ms/frame (median) | wall ms/frame (median) |
|---|---|---|---|---|---|---|
| eqar | 96/100 | 0 | 0.0303 | 0.0370 | 0.81 | 0.83 |
| cv2-direct | 85/100 | 0 | 0.2840 | 0.3795 | 43.31 | 43.46 |
| cv2-20views | 99/100 | 0 | 0.2874 | 0.3169 | 92.55 | 93.08 |
```

Misses: `eqar` missed 4 frames (half angle 3 to 6 degrees; none was a time-budget drop: the
same four are missed with `time_budget_ms=None`). At this width the 20 views (960 px each) are
oversampled relative to the frame, which favours that method.

## Notes and caveats

- The machine was shared with other jobs (load averages above), which is why thread CPU time
  is the figure to read; wall time agrees to within a few percent here.
- Synthetic frames only. Markers are flat, rendered by the same projector the tests use,
  without JPEG compression, rolling shutter or stitching. The accuracy numbers say nothing
  about real footage; see the limitations in the README.
- The OpenCV parameter choices are mine and a different tuning could move the cv2 rows.
- `eqar` was not tuned for this benchmark. Markers on a mid-grey background (128) with a
  quiet zone of 220 were not detected in an informal check, while the same markers on a
  light background (205, used here) were; the benchmark uses 205.
