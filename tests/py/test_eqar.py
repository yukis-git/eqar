import gc
import json
import os
import sys
import threading
import time

import numpy as np
import pytest

import eqar
import erp_synth as S
from conftest import GT, W

H = W // 2


def gt_rays(m=GT):
    return S.corner_rays(m["lon"], m["lat"], m["half"], m["roll"])


def corner_err(marker, truth, w=W):
    return S.angular_error_px(marker.rays, truth, w)


# ---------------------------------------------------------------- round trip

def test_round_trip_gray(frame):
    ms = eqar.detect(frame)
    assert [m.id for m in ms] == [7]
    m = ms[0]
    assert corner_err(m, gt_rays()).max() < 0.3  # px_eq, TL TR BR BL order
    assert m.flags & eqar.FLAG_SCAN
    assert m.reject_code == eqar.ACCEPT
    assert m.corner_valid.all()
    assert m.corners.shape == (4, 2) and m.corners.dtype == np.float32
    assert m.rays.shape == (4, 3) and m.rays.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(m.rays, axis=1), 1.0, atol=1e-5)
    c = S.corner_rays(**{"lon_deg": GT["lon"], "lat_deg": GT["lat"], "half_deg": GT["half"],
                         "roll_deg": GT["roll"]})
    assert S.angular_error_px(m.center_ray, S.frame_at(GT["lon"], GT["lat"], GT["roll"])[0], W) < 0.3
    px = S.ray_to_pixel(c, W, H)
    assert np.abs(m.corners - px).max() < 0.5
    assert np.abs(m.center - S.ray_to_pixel(m.center_ray, W, H)).max() < 0.5


@pytest.mark.parametrize("fmt", ["bgr", "rgb"])
def test_round_trip_colour(frame, fmt):
    # distinct channels so that a channel mix-up changes the grey conversion
    tint = np.array([0.6, 1.0, 0.8])
    img = (frame[:, :, None] * tint).astype(np.uint8)
    if fmt == "rgb":
        img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    ms = eqar.detect(img, format=fmt)
    assert [m.id for m in ms] == [7]
    assert corner_err(ms[0], gt_rays()).max() < 0.5


def test_bgr_and_rgb_agree_on_swapped_channels(frame):
    img = np.stack([frame, (frame * 0.5).astype(np.uint8), 255 - frame // 2], axis=2)
    a = eqar.detect(img, format="bgr")
    b = eqar.detect(np.ascontiguousarray(img[:, :, ::-1]), format="rgb")
    assert [m.id for m in a] == [m.id for m in b]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.corners, y.corners)


@pytest.mark.parametrize(
    "m",
    [
        dict(id=3, lon=0, lat=86, half=10, roll=30),    # north pole cap
        dict(id=4, lon=90, lat=-86, half=10, roll=0),   # south pole cap
        dict(id=5, lon=179.0, lat=15, half=9, roll=20), # straddles the seam
        dict(id=6, lon=-179.5, lat=-25, half=9, roll=-40),
    ],
)
def test_poles_and_seam(m):
    img = S.make_frame(W, [m], noise=2)
    ms = eqar.detect(img)
    assert [x.id for x in ms] == [m["id"]]
    assert corner_err(ms[0], gt_rays(m)).max() < 0.5
    if abs(m["lon"]) > 175:  # across the seam: corners at both ends of the image
        xs = ms[0].corners[:, 0]
        assert xs.min() < 0.1 * W and xs.max() > 0.9 * W


def test_several_markers_and_max_markers_caps_output():
    spec = [dict(id=i, lon=-150 + 60 * i, lat=-20 + 10 * i, half=7, roll=10 * i) for i in range(5)]
    img = S.make_frame(W, spec, noise=2)
    ms = eqar.detect(img, time_budget_ms=None)
    assert sorted(m.id for m in ms) == [0, 1, 2, 3, 4]
    capped = eqar.detect(img, time_budget_ms=None, max_markers=2)
    assert len(capped) == 2
    assert eqar.Detector(W, H, max_markers=2).max_markers == 2


def test_detect_default_returns_all_of_many_markers():
    # one-shot detect has no time budget: a 3840x1920 frame with 12 markers keeps all of them
    spec = [
        dict(id=i, lon=-165 + 30 * i, lat=(-30, 0, 30)[i % 3], half=5, roll=7 * i) for i in range(12)
    ]
    img = S.make_frame(3840, spec, noise=2)
    ms = eqar.detect(img)
    assert sorted(m.id for m in ms) == list(range(12))


def test_no_threads_option():
    with pytest.raises(TypeError):
        eqar.Detector(W, H, threads=2)
    with pytest.raises(TypeError):
        eqar.detect(np.zeros((H, W), np.uint8), threads=2)


def test_in_use_ids_expect_id_lean_and_gamma(frame):
    assert eqar.detect(frame, in_use_ids=[1, 2]) == []
    assert [m.id for m in eqar.detect(frame, in_use_ids=[7])] == [7]
    assert eqar.detect(frame, expect_id=3) == []
    assert [m.id for m in eqar.detect(frame, expect_id=7)] == [7]
    assert [m.id for m in eqar.detect(frame, quality="lean")] == [7]
    lut = np.arange(256, dtype=np.uint8)
    a = eqar.detect(frame)[0]
    b = eqar.detect(frame, gamma_lut=lut)[0]
    np.testing.assert_allclose(a.corners, b.corners, atol=1e-3)
    assert [m.id for m in eqar.detect(frame, time_budget_ms=None)] == [7]


def test_other_dictionary_round_trip():
    ncell = 5 + 2
    k = ncell / (ncell + 3)
    assert 0 < k < 1
    img = np.full((H, W), 205.0)
    S.paste_marker(img, 11, 40, 10, 9, 5, dictionary="5x5_100")
    ms = eqar.detect(img.astype(np.uint8), dictionary="5x5_100")
    assert [m.id for m in ms] == [11]
    assert eqar.detect(img.astype(np.uint8), dictionary="4x4_50") == []


# ---------------------------------------------------------------- strides

def test_strided_and_odd_layouts_match_contiguous(frame):
    ref = eqar.detect(frame)[0]
    padded = np.zeros((H, W + 37), np.uint8)
    padded[:, :W] = frame
    views = {
        "row-padded": padded[:, :W],
        "fortran": np.asfortranarray(frame),
        "transposed-twice": np.ascontiguousarray(frame.T).T,
        "readonly": frame.copy(),
    }
    views["readonly"].flags.writeable = False
    for name, v in views.items():
        got = eqar.detect(v)
        assert [m.id for m in got] == [7], name
        np.testing.assert_array_equal(got[0].corners, ref.corners, err_msg=name)
    rgb = np.stack([frame] * 3, axis=2)
    wide = np.zeros((H, W + 5, 3), np.uint8)
    wide[:, :W] = rgb
    chan = np.stack([frame] * 3, axis=2)[:, :, 0]  # channel slice: 2-D, element stride 3
    for name, v, fmt in [("rgb-padded", wide[:, :W], "rgb"), ("channel-slice", chan, "bgr")]:
        got = eqar.detect(v, format=fmt)
        assert [m.id for m in got] == [7], name
        np.testing.assert_allclose(got[0].corners, ref.corners, atol=0.05, err_msg=name)


def test_negative_and_zero_strides_are_copied_not_misread(frame):
    flipped = frame[::-1].copy()[::-1]  # same pixels, view with negative row stride
    assert flipped.strides[0] < 0
    assert [m.id for m in eqar.detect(flipped)] == [7]
    bcast = np.broadcast_to(np.uint8(205), (H, W))
    assert bcast.strides == (0, 0)
    assert eqar.detect(bcast) == []


# ---------------------------------------------------------------- track / verify

def moving(step, n):
    return [dict(id=7, lon=-100 + step * i, lat=20, half=8, roll=15) for i in range(n)]


def test_track_follows_and_frame_gap():
    spec = moving(2.0, 8)
    frames = [S.make_frame(W, [s], noise=2) for s in spec]
    det = eqar.Detector(W, H)
    ms = det.detect(frames[0])
    assert [m.id for m in ms] == [7]
    for i in (1, 2):
        ms = det.track(frames[i], ms)
        assert [m.id for m in ms] == [7]
        assert ms[0].flags & eqar.FLAG_TRACK
        assert corner_err(ms[0], gt_rays(spec[i])).max() < 0.5
    # skip three frames: frame_gap says how many frames passed
    ms = det.track(frames[5], ms, frame_gap=3)
    assert [m.id for m in ms] == [7]
    assert corner_err(ms[0], gt_rays(spec[5])).max() < 0.5
    ms = det.track(frames[6], ms, 1)
    assert [m.id for m in ms] == [7]
    assert det.last_stats["n_accepted"] == 1


def test_track_errors(frame):
    det = eqar.Detector(W, H)
    with pytest.raises(RuntimeError, match="nothing to track"):
        det.track(frame, [])
    ms = det.detect(frame)
    with pytest.raises(ValueError, match="frame_gap"):
        det.track(frame, ms, frame_gap=0)
    with pytest.raises(ValueError, match="frame_gap"):
        det.track(frame, ms, frame_gap=-2)
    with pytest.raises(TypeError):
        det.track(frame, [object()])
    det.track(frame, ms)
    det.reset()
    with pytest.raises(RuntimeError):
        det.track(frame, [])


def test_verify_known_id(frame):
    det = eqar.Detector(W, H)
    prior = eqar.Prior(7, gt_rays(), 1.0)
    ms = det.verify(frame, [prior])
    assert [m.id for m in ms] == [7]
    assert corner_err(ms[0], gt_rays()).max() < 0.5
    assert det.verify(frame, [eqar.Prior(3, gt_rays(), 1.0)]) == []
    assert det.verify(frame, []) == []
    # a prior built from an earlier detection
    m = det.detect(frame)[0]
    again = det.verify(frame, [eqar.Prior.from_marker(m, sigma_deg=2.0)])
    assert [x.id for x in again] == [7]
    # priors may be perturbed within their sigma
    shifted = gt_rays() + 0.005
    ms = det.verify(frame, [eqar.Prior(7, shifted, 1.0)])
    assert [m.id for m in ms] == [7]


def test_prior_validation(frame):
    r = gt_rays()
    with pytest.raises(ValueError, match="shape"):
        eqar.Prior(7, r[:3], 1.0)
    with pytest.raises(ValueError, match="sigma_deg"):
        eqar.Prior(7, r, 0.0)
    with pytest.raises(ValueError, match="sigma_deg"):
        eqar.Prior(7, r, float("nan"))
    with pytest.raises(ValueError, match="sigma_deg"):
        eqar.Prior(7, r, 11.0)
    with pytest.raises(ValueError, match="norm"):
        eqar.Prior(7, r * 3.0, 1.0)
    with pytest.raises(ValueError, match="id"):
        eqar.Prior(-1, r, 1.0)
    det = eqar.Detector(W, H)
    with pytest.raises(ValueError, match="outside dictionary"):
        det.verify(frame, [eqar.Prior(50, r, 1.0)])
    p = eqar.Prior(7, r.tolist(), 1.5)  # nested lists are accepted
    assert p.id == 7 and p.sigma_deg == 1.5 and p.rays.shape == (4, 3)
    assert "Prior(id=7" in repr(p)


# ---------------------------------------------------------------- errors

def test_bad_input_raises_value_error_not_panic(frame):
    det = eqar.Detector(W, H)
    bad = {
        "float image": np.zeros((H, W), np.float32),
        "int image": np.zeros((H, W), np.int32),
        "1-D": np.zeros(W, np.uint8),
        "4-D": np.zeros((1, H, W, 3), np.uint8),
        "two channels": np.zeros((H, W, 2), np.uint8),
        "alpha": np.zeros((H, W, 4), np.uint8),
        "empty": np.zeros((0, 0), np.uint8),
        "not 2:1": np.zeros((H, W // 2), np.uint8),
    }
    for name, a in bad.items():
        with pytest.raises(ValueError):
            det.detect(a)
        with pytest.raises(ValueError):
            eqar.detect(a)
    with pytest.raises(ValueError, match="uint8"):
        det.detect(np.zeros((H, W), np.float64))
    with pytest.raises(ValueError, match="detector was built for"):
        det.detect(np.zeros((480, 960), np.uint8))
    with pytest.raises(ValueError, match="multiple of 4"):
        eqar.detect(np.zeros((501, 1002), np.uint8))
    with pytest.raises(ValueError, match="between"):
        eqar.Detector(2 * 16388, 16388)
    with pytest.raises(ValueError, match="between"):
        eqar.Detector(296, 148)
    with pytest.raises(ValueError, match="format"):
        det.detect(frame, format="hsv")
    with pytest.raises(ValueError, match="gray"):
        det.detect(np.zeros((H, W, 3), np.uint8), format="gray")
    with pytest.raises(TypeError):
        det.detect([[0] * W] * H)
    with pytest.raises(TypeError):
        det.detect(None)
    with pytest.raises(TypeError):
        det.detect(b"abc")


def test_bad_parameters_raise_value_error():
    with pytest.raises(ValueError, match="unknown dictionary"):
        eqar.Detector(W, H, "nope")
    with pytest.raises(ValueError, match="width must equal"):
        eqar.Detector(100, 100)
    with pytest.raises(ValueError, match="multiple of 4"):
        eqar.Detector(1002, 501)
    with pytest.raises(ValueError, match="max_markers"):
        eqar.Detector(W, H, max_markers=0)
    with pytest.raises(ValueError, match="quality"):
        eqar.Detector(W, H, quality="fast")
    with pytest.raises(ValueError, match="time_budget_ms"):
        eqar.Detector(W, H, time_budget_ms=-1.0)
    with pytest.raises(ValueError, match="in_use_ids"):
        eqar.Detector(W, H, in_use_ids=[0, 99])
    with pytest.raises(ValueError, match="expect_id"):
        eqar.Detector(W, H, expect_id=50)
    with pytest.raises(ValueError, match="gamma_lut"):
        eqar.Detector(W, H, gamma_lut=np.zeros(10, np.uint8))
    with pytest.raises(ValueError):
        eqar.generate_marker(50)
    with pytest.raises(ValueError):
        eqar.generate_marker(0, size=3)
    with pytest.raises(ValueError):
        eqar.generate_marker(0, "6x6_9999")
    with pytest.raises(TypeError):
        eqar.Marker()
    assert eqar.Detector(W, H, "DICT_5X5_100").dictionary == "5x5_100"


def test_dictionaries_and_constants():
    assert len(eqar.DICTIONARIES) == 16 and eqar.DICTIONARIES[0] == "4x4_50"
    assert eqar.MAX_WIDTH == 16384 and eqar.MIN_WIDTH == 320
    assert eqar.FLAG_SCAN == 1 and eqar.FLAG_SEAM == 32 and eqar.ACCEPT == 0
    assert (eqar.PROV_LEGACY, eqar.PROV_LINEFIT, eqar.PROV_RELAXED) == (0, 1, 2)
    assert eqar.R_ID_NOT_EXPECTED == 12
    assert eqar.__version__ == "0.1.0"


# ---------------------------------------------------------------- generate_marker

def test_generate_marker():
    g = eqar.generate_marker(7)
    assert g.shape == (400, 400) and g.dtype == np.uint8
    assert set(np.unique(g)) <= {0, 255}
    n = 6  # 4x4 payload + border
    cell = eqar.generate_marker(7, size=n)
    assert (cell[0] == 0).all() and (cell[-1] == 0).all()
    assert (cell[:, 0] == 0).all() and (cell[:, -1] == 0).all()
    big = eqar.generate_marker(7, size=600)
    np.testing.assert_array_equal(big[::100, ::100], cell)  # cell centres
    assert not np.array_equal(eqar.generate_marker(8, size=n), cell)
    np.testing.assert_array_equal(eqar.generate_marker(7, "4x4_50", 6), cell)


# ---------------------------------------------------------------- reporting

def test_timings_stats_and_proposals(frame):
    det = eqar.Detector(W, H)
    assert det.last_timings["total"] == 0.0
    det.detect(frame)
    t = det.last_timings
    assert list(t) == ["propose", "bridge", "decode", "refine", "gate", "total"]
    assert t["total"] > 0
    s = det.last_stats
    assert s["n_accepted"] == 1 and s["n_candidates"] >= 1 and isinstance(s["ran_small"], bool)
    p = det.proposals(frame)
    assert p.ndim == 2 and p.shape[1] == 4 and p.dtype == np.float32 and len(p) >= 1
    np.testing.assert_allclose(np.linalg.norm(p[:, :3], axis=1), 1.0, atol=1e-4)
    assert "1920x960" in repr(det)


def test_marker_repr_to_dict_and_accessors(frame):
    m = eqar.detect(frame)[0]
    r = repr(m)
    assert r.startswith("Marker(id=7,") and "SCAN" in r
    d = m.to_dict()
    json.dumps(d)  # plain Python types only
    assert d["id"] == 7 and len(d["corners"]) == 4 and len(d["rays"][0]) == 3
    assert d["flags"] == m.flags and d["corner_valid"] == [True] * 4
    np.testing.assert_allclose(np.array(d["corners"], np.float32), m.corners)
    for name in ("quality", "side_px", "edge_resid_px", "contrast_dn", "min_cross"):
        assert isinstance(getattr(m, name), float)
    for name in ("reject_code", "prov", "n_bit_errors", "border_fail", "n_erased", "flags"):
        assert isinstance(getattr(m, name), int)
    assert m.corner_flags.shape == (4,) and m.corner_flags.dtype == np.uint8
    assert m.corner_sigma_scale.shape == (4,) and (m.corner_sigma_scale >= 1).all()
    assert m.corner_valid.dtype == np.bool_
    assert m.center.shape == (2,) and m.center_ray.shape == (3,)
    with pytest.raises(AttributeError):
        m.id = 3


# ---------------------------------------------------------------- lifetime

def test_image_refcount_is_not_leaked_by_calls(frame):
    img = frame.copy()
    det = eqar.Detector(W, H)
    base = sys.getrefcount(img)
    for _ in range(20):
        ms = det.detect(img)
        det.track(img, ms)
        det.verify(img, [eqar.Prior(7, gt_rays(), 1.0)])
        det.proposals(img)
    assert sys.getrefcount(img) == base
    with pytest.raises(ValueError):
        eqar.detect(img[:, :-4])
    with pytest.raises(ValueError):
        det.detect(img, format="bad")
    assert sys.getrefcount(img) == base
    # no stray buffer export keeps the array from being resized
    small = np.zeros((160, 320), np.uint8)
    for _ in range(3):
        eqar.detect(small)
    small.resize((80, 160), refcheck=True)


def test_markers_outlive_image_and_detector(frame):
    img = frame.copy()
    det = eqar.Detector(W, H)
    ms = det.detect(img)
    before = [(m.id, m.corners.copy(), m.to_dict()) for m in ms]
    del img, det
    gc.collect()
    junk = [np.random.default_rng(i).integers(0, 255, (H, W), dtype=np.uint8) for i in range(3)]
    for (i, c, d), m in zip(before, ms):
        assert m.id == i
        np.testing.assert_array_equal(m.corners, c)
        assert m.to_dict() == d
    del junk
    # markers can seed a new detector's tracker
    det2 = eqar.Detector(W, H)
    assert [m.id for m in det2.track(frame, ms)] == [7]


def test_accessor_arrays_are_copies(frame):
    m = eqar.detect(frame)[0]
    a = m.corners
    assert a is not m.corners
    a[:] = 0
    assert m.corners.min() > 0
    r = m.rays
    r[:] = 5
    assert m.rays.max() < 1.5
    m.corner_valid[:] = False
    assert m.corner_valid.all()


def test_marker_refcounts_and_churn(frame):
    det = eqar.Detector(W, H)
    ms = det.detect(frame)
    m = ms[0]
    base = sys.getrefcount(m)
    lst = [m] * 5
    assert sys.getrefcount(m) == base + 5
    det.track(frame, lst)
    assert sys.getrefcount(m) == base + 5  # track does not keep or leak references
    del lst
    assert sys.getrefcount(m) == base
    for _ in range(300):
        ms = det.track(frame, ms)
    assert [x.id for x in ms] == [7]
    many = [det.detect(frame)[0] for _ in range(200)]
    assert len({id(x) for x in many}) == 200
    del many
    gc.collect()


# ---------------------------------------------------------------- threads / GIL

@pytest.mark.skipif(bool(os.environ.get("CI")), reason="wall-clock timing; flaky on shared runners")
def test_gil_is_released_during_detection():
    w = 7680
    h = w // 2
    rng = np.random.default_rng(1)
    img = rng.integers(100, 160, (h, w), dtype=np.uint8)  # noisy, no markers: real work
    det = eqar.Detector(w, h, time_budget_ms=None)
    det.detect(img)  # warm up
    t0 = time.perf_counter()
    det.detect(img)
    one = time.perf_counter() - t0
    assert one > 0.005, "frame too cheap to observe the GIL"

    done = threading.Event()
    started = threading.Event()

    def worker():
        started.set()
        for _ in range(4):
            det.detect(img)
        done.set()

    th = threading.Thread(target=worker)
    ticks = 0
    th.start()
    started.wait()
    t_start = time.perf_counter()
    while not done.is_set():
        time.sleep(0.0005)  # needs the GIL back after every sleep
        ticks += 1
    span = time.perf_counter() - t_start
    th.join()
    # with the GIL held the main thread could not tick until the worker finished
    assert ticks >= 10 and ticks > span / 0.01, (ticks, span)


def test_shared_detector_across_threads(frame):
    det = eqar.Detector(W, H)
    out = []

    def work():
        for _ in range(10):
            out.append([m.id for m in det.detect(frame)])
            det.last_timings, det.last_stats  # readable while another thread detects

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
        assert not t.is_alive(), "deadlock"
    assert out == [[7]] * 40


def test_independent_detectors_run_concurrently(frame):
    dets = [eqar.Detector(W, H) for _ in range(3)]
    res = [None] * 3

    def work(i):
        res[i] = [eqar.detect(frame)[0].id] + [m.id for m in dets[i].detect(frame)]

    ts = [threading.Thread(target=work, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert res == [[7, 7]] * 3
