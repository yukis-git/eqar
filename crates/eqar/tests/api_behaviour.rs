//! Behaviour tests for the public API (numeric expectations are those of the original C++
//! tests, unchanged) plus the "nothing panics on bad caller input" fuzz tests.
#![allow(clippy::needless_range_loop, clippy::field_reassign_with_default)]

use eqar::codes::*;
use eqar::decode::{dict_identify, dict_info};
use eqar::geometry::Vec3;
use eqar::{
    generate_marker, Detector, Dict, EqarError, ImageView, Marker, Params, PixelFormat, Prior,
    Quality, PROV_LEGACY,
};
use eqar_testkit::synth::{render_test_marker_id, test_marker_corner_rays};

const W: usize = 3840;
const H: usize = 1920;

fn canvas() -> Vec<u8> {
    vec![205u8; W * H]
}

fn gray(px: &[u8]) -> ImageView<'_> {
    ImageView::packed(px, W, H, PixelFormat::Gray8)
}

fn marker_scene(id: i32) -> Vec<u8> {
    let mut px = canvas();
    render_test_marker_id(&mut px, W as i32, H as i32, -100.0, -8.0, 5.0, 13.0, id);
    px
}

fn prior_of(id: i32, gt: &[Vec3; 4], sigma_deg: f32) -> Prior {
    let mut ray = [[0f32; 3]; 4];
    for k in 0..4 {
        ray[k] = [gt[k].x as f32, gt[k].y as f32, gt[k].z as f32];
    }
    Prior { id, ray, sigma_deg }
}

#[test]
fn defaults_match_the_spec() {
    let p = Params::default();
    assert_eq!(p.dict, Dict::D4x4_50);
    assert_eq!(p.quality, Quality::Default);
    assert_eq!(p.max_markers, 32);
    assert!((p.time_budget_ms - 2.0).abs() < 1e-6);
    assert!(p.use_second_scale);
    assert!(!p.reject_high_resid);
    assert_eq!(p.expect_id, -1);
    assert!(p.gamma_lut.is_none());
    assert!(p.in_use_ids.is_none()); // every id, no coasting
    assert_eq!(p.coast_max, 0);
}

#[test]
fn dict_names_and_indices() {
    for (i, d) in Dict::ALL.iter().enumerate() {
        assert_eq!(d.index(), i as i32);
        assert_eq!(Dict::from_index(i as i32), Ok(*d));
        assert_eq!(d.name().parse::<Dict>(), Ok(*d));
        assert_eq!(
            format!("DICT_{}", d.name().to_uppercase()).parse::<Dict>(),
            Ok(*d)
        );
        assert_eq!(d.len() as i32, dict_info(i as i32).unwrap().n);
    }
    assert_eq!(Dict::from_index(16), Err(EqarError::Dict));
    assert_eq!(Dict::from_index(-1), Err(EqarError::Dict));
    assert_eq!("4x4_51".parse::<Dict>(), Err(EqarError::Dict));
    assert_eq!(Dict::D6x6_250.cells(), 8);
}

#[test]
fn in_use_ids_are_validated_and_filter_cold_detections() {
    let mut p = Params::default();
    p.in_use_ids = Some(vec![0, 99]);
    assert_eq!(Detector::new(W, H, &p).err(), Some(EqarError::Params));
    p.in_use_ids = Some(vec![-1]);
    assert_eq!(Detector::new(W, H, &p).err(), Some(EqarError::Params));

    let px = marker_scene(3);
    let im = gray(&px);
    let mut all = Detector::new(W, H, &Params::default()).unwrap();
    let out = all.detect(&im).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 3);

    p.in_use_ids = Some(vec![5, 6]); // id 3 is outside the in-use set: dropped
    let mut set = Detector::new(W, H, &p).unwrap();
    assert_eq!(set.detect(&im).unwrap().len(), 0);

    p.in_use_ids = Some(vec![2, 3, 4]);
    let mut set = Detector::new(W, H, &p).unwrap();
    let out = set.detect(&im).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 3);

    p.in_use_ids = Some(vec![]); // none
    let mut none = Detector::new(W, H, &p).unwrap();
    assert_eq!(none.detect(&im).unwrap().len(), 0);
}

#[test]
fn verify_validates_its_arguments_and_fills_the_v3_fields() {
    let mut d = Detector::new(W, H, &Params::default()).unwrap();
    let px = marker_scene(3);
    let im = gray(&px);
    let gt = test_marker_corner_rays(-100.0, -8.0, 5.0, 13.0);
    let pr = prior_of(3, &gt, 0.5);
    assert_eq!(d.verify(&im, &[]).unwrap().len(), 0);
    let mut bad = pr;
    bad.sigma_deg = 0.0;
    assert_eq!(d.verify(&im, &[bad]).err(), Some(EqarError::Params));
    bad.sigma_deg = 10.0; // the upper bound is inclusive
    assert!(d.verify(&im, &[bad]).is_ok());
    for s in [10.01f32, 1e9, f32::NAN, f32::INFINITY, -1.0] {
        bad.sigma_deg = s;
        assert_eq!(d.verify(&im, &[bad]).err(), Some(EqarError::Params), "{s}");
    }
    let mut bad = pr;
    bad.id = 50;
    assert_eq!(d.verify(&im, &[bad]).err(), Some(EqarError::Params));
    bad.id = -1;
    assert_eq!(d.verify(&im, &[bad]).err(), Some(EqarError::Params));
    let mut bad = pr;
    bad.ray[2] = [0.0; 3];
    assert_eq!(d.verify(&im, &[bad]).err(), Some(EqarError::Params));
    bad.ray[2] = [f32::NAN, 0.0, 1.0];
    assert_eq!(d.verify(&im, &[bad]).err(), Some(EqarError::Params));

    let out = d.verify(&im, &[pr]).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 3);
    assert!(out[0].flags & FLAG_TRACK != 0);
    for k in 0..4 {
        assert!(out[0].corner_valid[k]);
        assert_eq!(out[0].corner_flags[k], 0);
        assert!((out[0].corner_sigma_scale[k] - 1.0).abs() < 1e-6);
    }
    assert_eq!(out[0].prov, PROV_LEGACY);
    assert_eq!(out[0].n_bit_errors, 0);
    assert_eq!(out[0].n_erased, 0);
    assert_eq!(out[0].border_fail, 0);
    let mut wrong = pr;
    wrong.id = 4; // a wrong id at the right place verifies to nothing
    assert_eq!(d.verify(&im, &[wrong]).unwrap().len(), 0);
}

#[test]
fn error_codes_are_returned_not_panicked() {
    let p = Params::default();
    assert_eq!(Detector::new(0, 0, &p).err(), Some(EqarError::Size));
    assert_eq!(Detector::new(1002, 501, &p).err(), Some(EqarError::Size)); // w % 4 != 0
    assert_eq!(Detector::new(3840, 1000, &p).err(), Some(EqarError::Size)); // w != 2h
    assert!(Detector::new(1920, 960, &p).is_ok()); // a valid smaller size
    let mut d = Detector::new(W, H, &p).unwrap();
    let px = vec![200u8; W * H];
    let mut im = gray(&px);
    im.stride = 100;
    assert_eq!(d.detect(&im).err(), Some(EqarError::Stride));
    im.stride = W;
    im.width = W / 2;
    assert_eq!(d.detect(&im).err(), Some(EqarError::Size));
    let short = &px[..W * H - 1];
    assert_eq!(d.detect(&gray(short)).err(), Some(EqarError::Stride));
    assert_eq!(d.detect(&gray(&[])).err(), Some(EqarError::Stride));
    let bgr = ImageView::packed(&px, W, H, PixelFormat::Bgr8); // a gray buffer is too short for BGR
    assert_eq!(d.detect(&bgr).err(), Some(EqarError::Stride));
    let mut zero = Params::default();
    zero.max_markers = 0;
    let mut dz = Detector::new(W, H, &zero).unwrap();
    assert_eq!(dz.detect(&gray(&px)).err(), Some(EqarError::Cap));
    assert!(!EqarError::Size.to_string().is_empty());
}

#[test]
fn detect_fills_a_marker() {
    let mut d = Detector::new(W, H, &Params::default()).unwrap();
    let px = marker_scene(0);
    let out = d.detect(&gray(&px)).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 0);
    assert_eq!(out[0].reject_code, ACCEPT);
    assert!(out[0].flags & FLAG_SCAN != 0);
    assert!(out[0].quality > 0.5);
    assert!(d.last_timings()[5] > 0.0);
    assert_eq!(d.last_stats().n_accepted, 1);
}

#[test]
fn colour_and_padded_views_give_the_same_detection() {
    let px = marker_scene(0);
    let mut d = Detector::new(W, H, &Params::default()).unwrap();
    let want = d.detect(&gray(&px)).unwrap();
    assert_eq!(want.len(), 1);

    // padded gray rows
    let stride = W + 37;
    let mut padded = vec![0x55u8; stride * H];
    for y in 0..H {
        padded[y * stride..y * stride + W].copy_from_slice(&px[y * W..(y + 1) * W]);
    }
    let mut iv = gray(&padded);
    iv.stride = stride;
    let got = d.detect(&iv).unwrap();
    assert_eq!(got, want);

    // BGR / RGB of a gray image: the integer weights sum to 256, so the gray value is
    // reproduced exactly
    for fmt in [PixelFormat::Bgr8, PixelFormat::Rgb8] {
        let mut c = vec![0u8; W * H * 3];
        for i in 0..W * H {
            c[3 * i..3 * i + 3].fill(px[i]);
        }
        let got = d.detect(&ImageView::packed(&c, W, H, fmt)).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, 0);
        for k in 0..4 {
            for a in 0..2 {
                assert!((got[0].corner_px[k][a] - want[0].corner_px[k][a]).abs() < 0.01);
            }
        }
    }
}

#[test]
fn an_identity_gamma_lut_changes_nothing() {
    let px = marker_scene(0);
    let mut lut = [0u8; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = i as u8;
    }
    let mut p = Params::default();
    p.gamma_lut = Some(lut);
    let mut d = Detector::new(W, H, &p).unwrap();
    let mut plain = Detector::new(W, H, &Params::default()).unwrap();
    // the LUT view takes the general sampler (not the fused gray path): same detection up
    // to rounding
    let (a, b) = (
        d.detect(&gray(&px)).unwrap(),
        plain.detect(&gray(&px)).unwrap(),
    );
    assert_eq!((a.len(), a[0].id, a[0].flags), (1, b[0].id, b[0].flags));
    for k in 0..4 {
        for c in 0..2 {
            assert!((a[0].corner_px[k][c] - b[0].corner_px[k][c]).abs() < 1e-3);
        }
    }
}

#[test]
fn track_reports_noinit_without_a_prior_and_validates_the_gap() {
    let px = canvas();
    let mut d = Detector::new(W, H, &Params::default()).unwrap();
    let im = gray(&px);
    assert_eq!(d.track(&im, &[], 1).err(), Some(EqarError::NoInit));
    // coast_max == 0: no coasting marker can exist
    assert_eq!(
        d.track(&im, &[Marker::default()], 0).err(),
        Some(EqarError::Params)
    );
}

#[test]
fn track_with_gap_one_and_two_and_a_default_marker() {
    let mut d = Detector::new(W, H, &Params::default()).unwrap();
    let mut px = canvas();
    render_test_marker_id(&mut px, W as i32, H as i32, 20.0, 15.0, 5.0, 0.0, 0);
    let im = gray(&px);
    let prev = d.detect(&im).unwrap();
    assert_eq!(prev.len(), 1);
    assert_eq!(d.track(&im, &prev, 2).unwrap().len(), 1);
    let out = d.track(&im, &prev, 1).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 0);
    d.reset_track();
    // an all-zero marker (id 0, no rays) is skipped, not trusted
    assert_eq!(d.track(&im, &[Marker::default()], 1).unwrap().len(), 0);
}

#[test]
fn proposals_lists_candidates() {
    let mut d = Detector::new(W, H, &Params::default()).unwrap();
    let px = marker_scene(0);
    let c = d.proposals(&gray(&px)).unwrap();
    assert!(!c.is_empty());
    for r in &c {
        let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        assert!((n - 1.0).abs() < 1e-4 && r[3] > 0.0);
    }
}

#[test]
fn generated_markers_read_back_through_the_dictionary() {
    for dict in Dict::ALL {
        let info = dict_info(dict.index()).unwrap();
        let ms = info.marker_size as usize;
        let cells = dict.cells();
        assert_eq!(cells, ms + 2);
        let cell = 5;
        for id in [0, 1, info.n / 2, info.n - 1] {
            let img = generate_marker(dict, id, cell).unwrap();
            let side = cells * cell;
            assert_eq!(img.len(), side * side);
            let at = |r: usize, c: usize| img[(r * cell + cell / 2) * side + c * cell + cell / 2];
            for i in 0..cells {
                for k in [0, cells - 1] {
                    assert_eq!(at(k, i), 0);
                    assert_eq!(at(i, k), 0);
                }
            }
            let mut bits = 0u64;
            for r in 0..ms {
                for c in 0..ms {
                    bits = (bits << 1) | (at(r + 1, c + 1) == 255) as u64;
                }
            }
            assert_eq!(
                dict_identify(dict.index(), bits),
                Some((id, 0)),
                "{dict:?} {id}"
            );
        }
    }
    let d = Dict::D4x4_50;
    assert_eq!(generate_marker(d, 50, 4).err(), Some(EqarError::Params));
    assert_eq!(generate_marker(d, -1, 4).err(), Some(EqarError::Params));
    assert_eq!(generate_marker(d, 0, 0).err(), Some(EqarError::Params));
    assert_eq!(
        generate_marker(d, 0, usize::MAX).err(),
        Some(EqarError::Params)
    );
    assert_eq!(
        generate_marker(d, 0, 100_000).err(),
        Some(EqarError::Params)
    );
}

// ---- nothing panics on bad caller input ----

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 8) as f64 / 16777216.0
    }
    fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[(self.next() >> 8) as usize % v.len()]
    }
}

const WEIRD_F32: [f32; 12] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    1e-30,
    1e30,
    f32::MAX,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MIN_POSITIVE,
];

fn weird_marker(r: &mut Lcg) -> Marker {
    let mut m = Marker::default();
    m.id = r.pick(&[-1, 0, 3, 49, 50, i32::MAX, i32::MIN]);
    m.side_px = r.pick(&[0.0, 1.0, 64.0, f32::NAN, f32::INFINITY, 1e30]);
    for k in 0..4 {
        match r.next() % 4 {
            0 => m.ray[k] = [r.pick(&WEIRD_F32), r.pick(&WEIRD_F32), r.pick(&WEIRD_F32)],
            1 => {
                // unit-ish random direction
                let (a, b) = (r.unit() * std::f64::consts::TAU, r.unit() * 2.0 - 1.0);
                let s = (1.0 - b * b).sqrt();
                m.ray[k] = [(s * a.cos()) as f32, b as f32, (s * a.sin()) as f32];
            }
            2 => m.ray[k] = [0.0, 1.0, 0.0], // all corners the same: zero-size marker
            _ => m.ray[k] = [(k as f32) - 1.5, 0.3, 0.2], // elongated, non-unit
        }
        m.corner_px[k] = [r.pick(&WEIRD_F32), r.pick(&WEIRD_F32)];
    }
    m
}

fn weird_prior(r: &mut Lcg) -> Prior {
    let m = weird_marker(r);
    Prior {
        id: r.pick(&[-1, 0, 3, 49, 50, 1000]),
        ray: m.ray,
        sigma_deg: r.pick(&[0.0, 0.01, 0.5, 10.0, 10.01, f32::NAN, f32::INFINITY, -1.0]),
    }
}

/// Valid-looking priors that are geometrically degenerate (antipodal, collinear, equal).
fn degenerate_priors(id: i32) -> Vec<Prior> {
    let ray = |a: [[f32; 3]; 4]| Prior {
        id,
        ray: a,
        sigma_deg: 1.0,
    };
    let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    let neg = |v: [f32; 3]| [-v[0], -v[1], -v[2]];
    vec![
        ray([x, x, x, x]),
        ray([x, neg(x), x, neg(x)]),
        ray([x, y, neg(x), neg(y)]), // a great circle: zero-area quad
        ray([y, y, y, y]),           // the pole
        ray([neg(y), neg(y), neg(y), neg(y)]),
        ray([x, y, z, neg(z)]),
        ray([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.5],
            [1.5, 0.0, 0.0],
        ]),
    ]
}

#[test]
fn fuzz_wrong_image_arguments_never_panic() {
    let mut r = Lcg(12345);
    let mut det = Detector::new(W, H, &Params::default()).unwrap();
    let data = vec![128u8; W * H * 3];
    let dims = [
        0usize,
        1,
        2,
        3,
        4,
        7,
        960,
        1919,
        1920,
        1921,
        3840,
        3841,
        7680,
        usize::MAX,
        1 << 40,
    ];
    let strides = [
        0usize,
        1,
        100,
        W,
        W * 3,
        W * 3 + 1,
        usize::MAX,
        usize::MAX / 2,
        1 << 62,
    ];
    let fmts = [PixelFormat::Gray8, PixelFormat::Bgr8, PixelFormat::Rgb8];
    for _ in 0..400 {
        let iv = ImageView {
            data: &data[..(r.next() as usize) % (data.len() + 1)],
            width: r.pick(&dims),
            height: r.pick(&dims),
            stride: r.pick(&strides),
            format: r.pick(&fmts),
        };
        // every call returns, whatever it returns
        let _ = det.detect(&iv);
        let _ = det.proposals(&iv);
        let _ = det.track(&iv, &[Marker::default()], 1);
        let _ = det.verify(&iv, &[]);
    }
    // detector construction with wrong sizes and params
    for &w in &dims {
        for &h in &dims {
            let _ = Detector::new(w, h, &Params::default());
        }
    }
    let mut p = Params::default();
    p.max_markers = usize::MAX; // oversize capacity: no pre-allocation
    p.coast_max = u32::MAX;
    p.time_budget_ms = f32::NAN;
    let mut d = Detector::new(W, H, &p).unwrap();
    let px = marker_scene(0);
    let _ = d.detect(&gray(&px));
    p.time_budget_ms = f32::NEG_INFINITY;
    let mut d = Detector::new(W, H, &p).unwrap();
    let _ = d.detect(&gray(&px));
    let _ = d.track(&gray(&px), &[Marker::default()], u32::MAX);
}

#[test]
fn fuzz_weird_priors_and_previous_markers_never_panic() {
    let mut r = Lcg(777);
    let px = marker_scene(3);
    let blank = canvas();
    let mut p = Params::default();
    p.coast_max = 3;
    p.time_budget_ms = 20.0;
    let mut det = Detector::new(W, H, &p).unwrap();
    let mut n_ok = 0;
    for it in 0..250 {
        let n = 1 + (r.next() % 5) as usize;
        let img = if it % 2 == 0 { &px } else { &blank };
        let priors: Vec<Prior> = (0..n).map(|_| weird_prior(&mut r)).collect();
        if det.verify(&gray(img), &priors).is_ok() {
            n_ok += 1;
        }
        let prev: Vec<Marker> = (0..n).map(|_| weird_marker(&mut r)).collect();
        let gap = r.pick(&[1u32, 2, 5, 1000, u32::MAX]);
        let _ = det.track(&gray(img), &prev, gap);
        if it % 50 == 49 {
            det.reset_track();
        }
    }
    assert!(n_ok > 0, "some fuzz priors must be valid");
    // geometrically degenerate but formally valid priors
    for id in [0, 3] {
        for pr in degenerate_priors(id) {
            let _ = det.verify(&gray(&px), &[pr]);
            let _ = det.verify(&gray(&blank), &[pr]);
        }
    }
    // priors on the poles and the seam of a real marker image, with extreme sigmas
    for lat in [89.9, -89.9, 0.0] {
        for lon in [179.99, -179.99, 0.0] {
            let gt = test_marker_corner_rays(lon, lat, 3.0, 0.0);
            for sigma in [0.0001f32, 10.0] {
                let _ = det.verify(&gray(&px), &[prior_of(3, &gt, sigma)]);
            }
        }
    }
    // degenerate previous markers, all at once (>= 4 forces lean)
    let mut prev = Vec::new();
    for pr in degenerate_priors(0) {
        let mut m = Marker::default();
        m.id = prev.len() as i32;
        m.side_px = 10.0;
        m.ray = pr.ray;
        prev.push(m);
    }
    for _ in 0..3 {
        let _ = det.track(&gray(&px), &prev, 1);
        let _ = det.track(&gray(&blank), &[], 1);
    }
}

#[test]
fn fuzz_random_images_never_panic() {
    let mut r = Lcg(99);
    let mut det = Detector::new(960, 480, &Params::default()).unwrap();
    let mut img = vec![0u8; 960 * 480];
    for round in 0..12 {
        match round % 4 {
            0 => img.iter_mut().for_each(|b| *b = (r.next() >> 24) as u8),
            1 => img.fill(if round % 8 == 1 { 0 } else { 255 }),
            2 => {
                // blocks of random brightness
                for y in 0..480 {
                    for x in 0..960 {
                        img[y * 960 + x] = ((x / 12 + y / 9 + round) * 37 % 256) as u8;
                    }
                }
            }
            _ => {
                for (i, b) in img.iter_mut().enumerate() {
                    *b = if (i / 960 + i % 960) % 7 < 3 { 20 } else { 230 };
                }
            }
        }
        let iv = ImageView::packed(&img, 960, 480, PixelFormat::Gray8);
        det.detect(&iv).unwrap();
        let _ = det.track(&iv, &[weird_marker(&mut r)], 1);
    }
}

#[test]
fn fuzz_random_valid_priors_on_a_small_frame_never_panic() {
    // Random well-formed priors anywhere on the sphere (poles and seam included), from a
    // fraction of a degree to a large part of the sky, against a frame holding markers.
    let (w, h) = (960usize, 480usize);
    let mut px = vec![205u8; w * h];
    for (i, (lon, lat)) in [(-100.0, -8.0), (179.0, 0.0), (0.0, 82.0), (30.0, -85.0)]
        .iter()
        .enumerate()
    {
        render_test_marker_id(
            &mut px,
            w as i32,
            h as i32,
            *lon,
            *lat,
            5.0,
            13.0 * i as f64,
            i as i32,
        );
    }
    let mut r = Lcg(2024);
    let mut det = Detector::new(w, h, &Params::default()).unwrap();
    for it in 0..1500 {
        let (lon, lat) = (r.unit() * 360.0 - 180.0, r.unit() * 178.0 - 89.0);
        let half = r.pick(&[0.05, 0.3, 1.0, 3.0, 5.0, 15.0, 40.0, 80.0]);
        let gt = test_marker_corner_rays(lon, lat, half, r.unit() * 90.0);
        let id = (r.next() % 50) as i32;
        let pr = prior_of(id, &gt, r.pick(&[0.01, 0.5, 5.0, 10.0]));
        let iv = ImageView::packed(&px, w, h, PixelFormat::Gray8);
        let _ = det.verify(&iv, &[pr]);
        if it % 3 == 0 {
            let mut m = Marker::default();
            (m.id, m.side_px, m.ray) = (id, 40.0, pr.ray);
            let mut m2 = m;
            m2.ray = prior_of(id, &test_marker_corner_rays(lon + 1.0, lat, half, 0.0), 1.0).ray;
            let _ = det.track(&iv, &[m, m2], 1 + (r.next() % 4));
        }
    }
}

/// Widths that are not a multiple of 32 leave a ragged last tile row / column in the f=4
/// plane; widths below `MIN_WIDTH` are refused. Neither may panic.
#[test]
fn ragged_and_small_sizes_are_handled_without_panicking() {
    let p = Params::default();
    for w in [4usize, 32, 100, 296, 316] {
        assert_eq!(
            Detector::new(w, w / 2, &p).err(),
            Some(EqarError::Size),
            "w={w}"
        );
    }
    let mut r = Lcg(7);
    for w in [320usize, 324, 332, 1000, 1004, 2000, 3000] {
        let h = w / 2;
        let mut det = Detector::new(w, h, &p).unwrap();
        for kind in 0..3 {
            let px: Vec<u8> = (0..w * h)
                .map(|_| match kind {
                    0 => 205,
                    1 => (r.next() >> 8) as u8,
                    _ => {
                        if r.next() & 0x100 != 0 {
                            255
                        } else {
                            0
                        }
                    }
                })
                .collect();
            let im = ImageView::packed(&px, w, h, PixelFormat::Gray8);
            det.detect(&im).unwrap();
            det.proposals(&im).unwrap();
        }
    }
}
