#!/usr/bin/env python3
"""Compare eqar with cv2.aruco on synthetic equirectangular frames.

    pip install eqar numpy opencv-python-headless
    python bench/bench.py --frames 60 --width 3840

Each frame has ONE marker (random id of DICT_4X4_50) pasted by the pure-numpy projector in
tests/py/erp_synth.py at a random direction on the sphere (uniform in area, so poles and
the seam are included), a random roll and anti-aliased edges (3 x 3 supersampling), then Gaussian blur and
noise. Marker half angle (quiet-zone square) is 3, 4, 6 or 9 degrees. Ground truth is the exact tag-corner ray.

Methods
  eqar         eqar.Detector(w, h).detect(frame), default parameters
  cv2-direct   cv2.aruco.ArucoDetector on the equirectangular image, parameters tuned for it
  cv2-20views  20 pinhole views (icosahedron face centres, 80 deg field of view,
               960 x 960 px) remapped from the frame, cv2.aruco on each, corners mapped back
               to the sphere, one detection per id kept (the one nearest a view centre)

Metrics per method: detection rate (the right id with every corner within --tol px_eq of
the truth), wrong-id detections, median / mean corner error over the detected markers (the
angle between the detected and the true corner ray, in equivalent pixels = 2 pi / width rad),
and time per frame as thread CPU time (robust against a loaded machine) and wall time.
cv2 is run with one thread. Only wall time is affected by other processes.
"""

import argparse
import itertools
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "py"))
import erp_synth as S  # noqa: E402

import eqar  # noqa: E402

try:
    import cv2
except ImportError:  # the cv2 rows are skipped
    cv2 = None


def make_cases(n, w, seed):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        lat = np.degrees(np.arcsin(rng.uniform(-1, 1)))
        cases.append(
            dict(
                id=int(rng.integers(0, 50)),
                lon=float(rng.uniform(-180, 180)),
                lat=float(lat),
                half=float(rng.choice([3.0, 4.0, 6.0, 9.0])),
                roll=float(rng.uniform(-180, 180)),
            )
        )
    return cases


def icosahedron_centres():
    p = (1 + 5 ** 0.5) / 2
    v = []
    for a, b in itertools.product((-1, 1), repeat=2):
        v += [(0, a, b * p), (a, b * p, 0), (b * p, 0, a)]
    v = np.array(v, float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    edge = np.min([np.linalg.norm(v[i] - v[j]) for i in range(12) for j in range(i)])
    c = []
    for i, j, k in itertools.combinations(range(12), 3):
        d = lambda a, b: abs(np.linalg.norm(v[a] - v[b]) - edge) < 1e-6
        if d(i, j) and d(j, k) and d(i, k):
            s = v[i] + v[j] + v[k]
            c.append(s / np.linalg.norm(s))
    assert len(c) == 20
    return np.array(c)


class Views:
    def __init__(self, w, size=960, fov_deg=80.0):
        h = w // 2
        self.size = size
        self.f = (size / 2) / np.tan(np.radians(fov_deg / 2))
        self.R = []
        self.maps = []
        uu, vv = np.meshgrid(np.arange(size), np.arange(size))
        cam = np.stack([(uu + 0.5 - size / 2) / self.f, (vv + 0.5 - size / 2) / self.f,
                        np.ones_like(uu, float)], axis=-1)
        for c in icosahedron_centres():
            up = np.array([0.0, 1.0, 0.0]) if abs(c[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            right = np.cross(c, up)
            right /= np.linalg.norm(right)
            down = np.cross(c, right)  # right x down = forward
            R = np.stack([right, down, c])  # rows: camera axes in world coordinates
            self.R.append(R)
            d = cam @ R  # world direction of each view pixel
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            px = S.ray_to_pixel(d, w, h)
            self.maps.append((px[..., 0].astype(np.float32), px[..., 1].astype(np.float32)))

    def to_ray(self, k, corners):
        cam = np.stack([(corners[:, 0] + 0.5 - self.size / 2) / self.f,
                        (corners[:, 1] + 0.5 - self.size / 2) / self.f,
                        np.ones(len(corners))], axis=1)
        r = cam @ self.R[k]
        return r / np.linalg.norm(r, axis=1, keepdims=True)


def cv2_detector(direct):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if direct:
        # tuned for the equirectangular image: small markers, strongly stretched near the poles
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 63
        p.adaptiveThreshWinSizeStep = 6
        p.minMarkerPerimeterRate = 0.01
        p.polygonalApproxAccuracyRate = 0.05
        p.cornerRefinementWinSize = 5
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)


def cpu():
    return time.thread_time()


def run(args):
    w, h = args.width, args.width // 2
    cases = make_cases(args.frames, w, args.seed)
    det = eqar.Detector(w, h)
    methods = ["eqar"]
    if cv2 is not None:
        cv2.setNumThreads(1)
        d_direct, d_view = cv2_detector(True), cv2_detector(False)
        methods += ["cv2-direct"] + (["cv2-20views"] if not args.no_views else [])
        views = Views(w) if not args.no_views else None
    res = {m: dict(hit=0, wrong=0, err=[], cpu=[], wall=[]) for m in methods}
    t_render = 0.0
    for i, c in enumerate(cases):
        t0 = time.perf_counter()
        img = S.make_frame(w, [c], noise=args.noise, blur=args.blur, seed=i)
        t_render += time.perf_counter() - t0
        truth = S.corner_rays(c["lon"], c["lat"], c["half"], c["roll"])

        def score(m, found):  # found: list of (id, rays(4,3))
            r = res[m]
            ok = False
            for fid, rays in found:
                if fid != c["id"]:
                    r["wrong"] += 1
                    continue
                e = S.angular_error_px(rays, truth, w)
                if e.max() <= args.tol:
                    if not ok:
                        r["err"].append(float(e.mean()))
                    ok = True
            r["hit"] += ok

        def timed(m, f):
            w0, c0 = time.perf_counter(), cpu()
            out = f()
            res[m]["cpu"].append(cpu() - c0)
            res[m]["wall"].append(time.perf_counter() - w0)
            return out

        ms = timed("eqar", lambda: det.detect(img))
        score("eqar", [(m.id, m.rays) for m in ms])
        if cv2 is None:
            continue

        def direct():
            return d_direct.detectMarkers(img)

        corners, ids, _ = timed("cv2-direct", direct)
        found = []
        if ids is not None:
            for cs, mid in zip(corners, ids.ravel()):
                found.append((int(mid), S.pixel_rays(cs[0][:, 0], cs[0][:, 1], w, h)))
        score("cv2-direct", found)
        if args.no_views:
            continue

        def expanded():
            best = {}
            for k, (mx, my) in enumerate(views.maps):
                v = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                cs, ids, _ = d_view.detectMarkers(v)
                if ids is None:
                    continue
                for cc, mid in zip(cs, ids.ravel()):
                    ctr = np.abs(cc[0].mean(axis=0) - views.size / 2).max()
                    if int(mid) not in best or ctr < best[int(mid)][0]:
                        best[int(mid)] = (ctr, views.to_ray(k, cc[0]))
            return best

        best = timed("cv2-20views", expanded)
        score("cv2-20views", [(mid, rays) for mid, (_, rays) in best.items()])
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(cases)} frames", file=sys.stderr)
    return methods, res, t_render


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=3.0, help="Gaussian noise sd, grey levels")
    ap.add_argument("--blur", type=float, default=0.8, help="Gaussian PSF sd, pixels")
    ap.add_argument("--tol", type=float, default=3.0, help="max corner error for a hit, px_eq")
    ap.add_argument("--no-views", action="store_true", help="skip the 20-view expansion")
    ap.add_argument("--json", help="also write the raw numbers here")
    args = ap.parse_args()

    load0 = os.getloadavg()
    methods, res, t_render = run(args)
    load1 = os.getloadavg()
    n = args.frames
    print(f"\nframes {n}, {args.width}x{args.width // 2}, noise sd {args.noise}, blur sd {args.blur}, "
          f"seed {args.seed}, hit = right id and every corner within {args.tol} px_eq")
    print(f"machine: {platform.platform()} {platform.machine()}, {os.cpu_count()} cores, "
          f"python {platform.python_version()}, numpy {np.__version__}, "
          f"opencv {cv2.__version__ if cv2 else 'n/a'}, eqar {eqar.__version__}")
    print(f"load average (1/5/15 min) before {load0[0]:.1f}/{load0[1]:.1f}/{load0[2]:.1f}, "
          f"after {load1[0]:.1f}/{load1[1]:.1f}/{load1[2]:.1f}\n")
    print("| method | detected | wrong id | median err px_eq | mean err px_eq | cpu ms/frame (median) | wall ms/frame (median) |")
    print("|---|---|---|---|---|---|---|")
    for m in methods:
        r = res[m]
        e = np.array(r["err"])
        med = f"{np.median(e):.4f}" if len(e) else "-"
        mean = f"{e.mean():.4f}" if len(e) else "-"
        print(f"| {m} | {r['hit']}/{n} | {r['wrong']} | {med} | {mean} | "
              f"{np.median(r['cpu']) * 1e3:.2f} | {np.median(r['wall']) * 1e3:.2f} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "results": res}, f)


if __name__ == "__main__":
    main()
