//! Behaviour tests for the decode stage (numeric expectations are those of the original
//! C++ unit tests, unchanged).
#![allow(clippy::needless_range_loop)]

use eqar::codes;
use eqar::decode::*;
use eqar::geometry::{basis_at, cross, dot, normalize, px_to_ray, ray_to_px, slerp, Vec3, K_PI};
use eqar::image::{Fmt, View};
use eqar::params::{cold_decode_config, verify_decode_config};
use eqar::refine::K_MIN_CONTRAST_DN;
use eqar_testkit::synth::{render_test_marker_id, test_marker_corner_rays};
use std::sync::atomic::{AtomicUsize, Ordering};

const DICT_4X4_50: i32 = 0;
const DICT_4X4_1000: i32 = 3;
const DICT_6X6_250: i32 = 10;
const DICT_7X7_50: i32 = 12;
const DICT_7X7_1000: i32 = 15;

const IDS_0_TO_9: [i32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

fn gray_view(px: &[u8], w: i32, h: i32) -> View<'_> {
    View {
        data: px,
        w,
        h,
        stride: w as usize,
        fmt: Fmt::Gray8,
        lut: None,
    }
}

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn sum_norm(g: &[Vec3; 4]) -> Vec3 {
    normalize(&Vec3::new(
        g[0].x + g[1].x + g[2].x + g[3].x,
        g[0].y + g[1].y + g[2].y + g[3].y,
        g[0].z + g[1].z + g[2].z + g[3].z,
    ))
}

#[test]
fn dict_table_shapes() {
    let d = dict_info(DICT_4X4_50).unwrap();
    assert_eq!(d.n, 50);
    assert_eq!(d.marker_size, 4);
    assert_eq!(d.max_correction, 1);
    assert_eq!(dict_info(DICT_7X7_1000).unwrap().n, 1000);
    assert_eq!(dict_info(DICT_7X7_1000).unwrap().marker_size, 7);
    assert_eq!(dict_info(DICT_4X4_1000).unwrap().max_correction, 0);
    assert_eq!(dict_info(DICT_7X7_50).unwrap().max_correction, 9);
    assert!(dict_info(16).is_err() && dict_info(-1).is_err());
}

#[test]
fn id0_bits_match_opencv() {
    // DICT_4X4_50 id 0, rotation 0, verified against cv2 5.0.0:
    //   1 0 1 1 / 0 1 0 1 / 0 0 1 1 / 0 0 1 0
    let rows: [u16; 4] = [0xB, 0x5, 0x3, 0x2];
    let mut bits = 0u64;
    for r in 0..4 {
        for c in 0..4 {
            if (rows[r] >> (3 - c)) & 1 != 0 {
                bits |= 1u64 << (15 - (r * 4 + c));
            }
        }
    }
    assert_eq!(dict_info(DICT_4X4_50).unwrap().codes[0], bits);
    assert_eq!(dict_identify(DICT_4X4_50, bits), Some((0, 0)));
}

#[test]
fn identify_finds_every_rotation() {
    let d = dict_info(DICT_4X4_50).unwrap();
    for i in 0..d.n {
        for j in 0..4 {
            assert_eq!(
                dict_identify(DICT_4X4_50, d.codes[(i * 4 + j) as usize]),
                Some((i, j))
            );
        }
    }
}

#[test]
fn identify_corrects_within_budget_and_rejects_beyond() {
    let d = dict_info(DICT_6X6_250).unwrap(); // max_correction 5
    let b = d.codes[42 * 4];
    assert_eq!(
        dict_identify(DICT_6X6_250, b ^ (1u64 << 7)).map(|x| x.0),
        Some(42)
    );
    let wrecked = b ^ 0x0F0F0F0F0F; // 20 flipped bits
    assert!(dict_identify(DICT_6X6_250, wrecked).is_none());
}

fn oblique_marker() -> ([Vec3; 4], [f64; 4], [f64; 4]) {
    // an arbitrary oblique marker plane
    let c = normalize(&Vec3::new(0.4, -0.55, 0.73));
    let (e1, e2) = basis_at(&c);
    let s = 0.17;
    let sx = [-1.0, 1.0, 1.0, -1.0];
    let sy = [1.0, 1.0, -1.0, -1.0];
    let mut r = [Vec3::default(); 4];
    for i in 0..4 {
        r[i] = normalize(&Vec3::new(
            c.x + e1.x * sx[i] * s + e2.x * sy[i] * s,
            c.y + e1.y * sx[i] * s + e2.y * sy[i] * s,
            c.z + e1.z * sx[i] * s + e2.z * sy[i] * s,
        ));
    }
    (r, sx, sy)
}

#[test]
fn homography_reproduces_the_corner_rays() {
    let (r, sx, sy) = oblique_marker();
    let h = homography_closed(&r).expect("homography");
    for i in 0..4 {
        let q = plane_to_ray(&h, sx[i], sy[i]);
        assert!(near(dot(&q, &r[i]), 1.0, 1e-12)); // exact to 1e-6 px_eq and better
    }
    // the centre maps to the plane centre direction
    let m = plane_to_ray(&h, 0.0, 0.0);
    let cs = sum_norm(&r);
    assert!(dot(&m, &cs).clamp(-1.0, 1.0).acos() < 2e-3);
}

#[test]
fn homography_centre_dot_is_machine_precision() {
    // the same centre-direction property, compared on the raw dot product (an acos of a dot
    // that rounds to 1 + 1 ulp is NaN)
    let (r, _, _) = oblique_marker();
    let h = homography_closed(&r).expect("homography");
    let m = plane_to_ray(&h, 0.0, 0.0);
    let cs = sum_norm(&r);
    assert!(near(dot(&m, &cs), 1.0, 1e-9));
}

#[test]
fn homography_rejects_a_degenerate_quad() {
    let r = [Vec3::new(0.0, 0.0, 1.0); 4];
    assert!(homography_closed(&r).is_none());
}

#[test]
fn reads_a_synthetic_marker_at_every_rotation() {
    let (w, h) = (1920, 960);
    let mut px = vec![200u8; (w * h) as usize];
    // renders DICT_4X4_50 id 0 with its quiet zone, exact gnomonic warp
    render_test_marker_id(&mut px, w, h, 137.0, 41.0, 6.0, 0.0, 0);
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(137.0, 41.0, 6.0, 0.0);
    for roll in 0..4 {
        let mut r = [Vec3::default(); 4];
        for i in 0..4 {
            r[i] = gt[(i + roll) & 3];
        }
        let d = decode_rays(&v, DICT_4X4_50, &r, K_SUB_DEFAULT, None);
        assert_eq!(d.id, 0);
        assert!(d.border_ok);
        assert_eq!(d.reject_code, codes::ACCEPT);
        // whatever the input order, the OUTPUT is TL,TR,BR,BL
        for i in 0..4 {
            assert!(near(dot(&d.corners[i], &gt[i]), 1.0, 1e-9));
        }
    }
}

#[test]
fn output_order_is_right_handed() {
    let (w, h) = (1920, 960);
    let mut px = vec![200u8; (w * h) as usize];
    render_test_marker_id(&mut px, w, h, 0.0, 0.0, 6.0, 0.0, 0);
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(0.0, 0.0, 6.0, 0.0);
    let mirrored = [gt[3], gt[2], gt[1], gt[0]]; // reversed traversal
    let d = decode_rays(&v, DICT_4X4_50, &mirrored, K_SUB_DEFAULT, None);
    assert_eq!(d.id, 0);
    assert!(dot(&cross(&d.corners[0], &d.corners[1]), &d.corners[2]) > 0.0);
}

#[test]
fn rejects_a_marker_whose_border_is_not_black() {
    let (w, h) = (1920, 960);
    let mut px = vec![200u8; (w * h) as usize];
    render_test_marker_id(&mut px, w, h, 0.0, 0.0, 6.0, 0.0, 0);
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(0.0, 0.0, 6.0, 0.0);
    // shrink the quad by 25 %: the sampled "border" now lands on payload cells
    let c = sum_norm(&gt);
    let mut small = [Vec3::default(); 4];
    for i in 0..4 {
        small[i] = slerp(&c, &gt[i], 0.75);
    }
    let d = decode_rays(&v, DICT_4X4_50, &small, K_SUB_DEFAULT, None);
    assert!(d.reject_code == codes::R_BORDER_NOT_BLACK || d.id < 0);
}

#[test]
fn lean_subsampling_agrees_with_default() {
    let (w, h) = (1920, 960);
    let mut px = vec![200u8; (w * h) as usize];
    render_test_marker_id(&mut px, w, h, -63.0, -22.0, 5.0, 17.0, 23);
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(-63.0, -22.0, 5.0, 17.0);
    assert_eq!(
        decode_rays(&v, DICT_4X4_50, &gt, K_SUB_DEFAULT, None).id,
        23
    );
    assert_eq!(decode_rays(&v, DICT_4X4_50, &gt, K_SUB_LEAN, None).id, 23);
}

// ---- the NEON cell sampler must match the scalar one ----------------------------------------

#[test]
fn the_neon_cell_sampler_matches_the_scalar_one() {
    // decode_rays takes its NEON path only for a Gray8 view with no gamma LUT, so an IDENTITY
    // LUT over the same buffer selects the scalar path on byte-identical input (sample()'s LUT
    // branch returns lut[g] == g). Both paths must then produce the same bits, id and
    // contrast: the NEON kernel works in float and skips the normalise, and this is what says
    // that costs nothing the decision layer can see.
    let (w, h) = (3840, 1920);
    let mut lut = [0u8; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = i as u8;
    }
    let poses: [(f64, f64, f64, f64, i32); 5] = [
        (0.0, 0.0, 5.0, 0.0, 0),
        (47.0, 45.0, 4.0, 23.0, 7),
        (179.6, 3.0, 6.0, 61.0, 3),
        (-150.0, -78.0, 7.0, 140.0, 12),
        (90.0, 88.5, 4.5, 200.0, 5),
    ];
    let mut px = vec![0u8; (w * h) as usize];
    for q in poses {
        px.fill(198);
        render_test_marker_id(&mut px, w, h, q.0, q.1, q.2, q.3, q.4);
        let gt = test_marker_corner_rays(q.0, q.1, q.2, q.3);
        let fast = gray_view(&px, w, h);
        let slow = View {
            lut: Some(&lut),
            ..fast
        };
        for n_sub in [K_SUB_DEFAULT, K_SUB_LEAN] {
            let a = decode_rays(&fast, DICT_4X4_50, &gt, n_sub, None);
            let b = decode_rays(&slow, DICT_4X4_50, &gt, n_sub, None);
            assert_eq!(a.reject_code, codes::ACCEPT);
            assert_eq!(a.id, q.4);
            assert_eq!(a.id, b.id);
            assert_eq!(a.rot, b.rot);
            assert_eq!(a.bits, b.bits);
            assert_eq!(a.reject_code, b.reject_code);
            assert!((a.contrast as f64 - b.contrast as f64).abs() < 0.01);
        }
    }
}

// ---- the pre-bridge border probe ------------------------------------------------------------

#[test]
fn the_border_probe_reads_a_marker_ring_and_not_background_clutter() {
    // border_dark_fraction is the screen that stands in front of Bridge::run: it must read
    // ~1.0 on a real marker's black outer ring and clearly below the pre-bridge border
    // fraction on the two shapes background clutter actually makes -- a featureless dark
    // rectangle and a striped one.
    let (w, h) = (3840usize, 1920usize);
    let mut px = vec![205u8; w * h];
    render_test_marker_id(&mut px, w as i32, h as i32, -100.0, -8.0, 5.0, 13.0, 0);
    let v = gray_view(&px, w as i32, h as i32);
    let gt = test_marker_corner_rays(-100.0, -8.0, 5.0, 13.0);

    let mut con = 0.0f32;
    let f_marker = border_dark_fraction(&v, DICT_4X4_50, &gt, 2, Some(&mut con));
    assert!(f_marker > 0.99); // every one of the 20 border cells dark
    assert!(con > 40.0); // black ring vs white data modules

    // A quad a couple of px out still reads the ring: the probe runs off the PROPOSAL quad
    // (the blob's four diagonal extremes), never a polished one.
    let pix = K_PI / h as f64;
    let mut off = [Vec3::default(); 4];
    for i in 0..4 {
        off[i] = normalize(&Vec3::new(
            gt[i].x + 2.5 * pix,
            gt[i].y - 2.5 * pix,
            gt[i].z,
        ));
    }
    assert!(border_dark_fraction(&v, DICT_4X4_50, &off, 2, None) >= 0.75);

    // Two clutter shapes over the same bright background, at the same size, addressed by
    // their own four corner rays.
    let (x0, y0, sq) = (300usize, 800usize, 90usize);
    let (wi, hi) = (w as i32, h as i32);
    let rect = [
        px_to_ray(x0 as f64, y0 as f64, wi, hi),
        px_to_ray((x0 + sq) as f64, y0 as f64, wi, hi),
        px_to_ray((x0 + sq) as f64, (y0 + sq) as f64, wi, hi),
        px_to_ray(x0 as f64, (y0 + sq) as f64, wi, hi),
    ];

    let mut solid = vec![205u8; w * h];
    for y in y0..=y0 + sq {
        for x in x0..=x0 + sq {
            solid[y * w + x] = 40;
        }
    }
    let sv = gray_view(&solid, wi, hi);
    let mut con_solid = 0.0f32;
    let f_solid = border_dark_fraction(&sv, DICT_4X4_50, &rect, 2, Some(&mut con_solid));
    // A featureless patch has NO valid Otsu split, so otsu_exact returns min + 1 and every
    // cell reads "dark" -- the fraction alone says 1.0. The contrast is what actually
    // rejects it, which is why the detector tests both.
    assert!(f_solid > 0.99);
    assert!(con_solid < K_MIN_CONTRAST_DN);

    let mut striped = vec![205u8; w * h];
    for y in y0..=y0 + sq {
        for x in x0..=x0 + sq {
            striped[y * w + x] = if (x / 7) & 1 != 0 { 210 } else { 35 };
        }
    }
    let tv = gray_view(&striped, wi, hi);
    let mut con_striped = 0.0f32;
    let f_striped = border_dark_fraction(&tv, DICT_4X4_50, &rect, 2, Some(&mut con_striped));
    assert!(con_striped > K_MIN_CONTRAST_DN); // plenty of contrast...
    assert!(f_striped < 0.75); // ...but no dark RING

    // A degenerate quad is "unmeasurable", never a rejection.
    let same = [gt[0]; 4];
    assert!(border_dark_fraction(&v, DICT_4X4_50, &same, 2, None) < 0.0);
}

// ---- cold decode (legacy + in-use set) and verify decode (known id) -------------------------

#[test]
fn cold_decode_equals_legacy_restricted_to_the_in_use_set_exhaustive() {
    // cold accepts EXACTLY the patterns legacy accepts whose legacy id is in the set, with
    // legacy's id and rotation. (A restricted-set search loosened the bound, e.g. {0}
    // decoded 0x8d, legacy id 49.)
    let s1 = [0];
    let s2 = [0, 1];
    let s5 = [0, 1, 2, 3, 4];
    let sets: [&[i32]; 4] = [&s1, &s2, &s5, &IDS_0_TO_9];
    for dict in [DICT_4X4_50, DICT_4X4_1000] {
        for st in sets {
            let cfg = cold_decode_config(Some(st));
            let (mut n_acc, mut n_legacy) = (0usize, 0usize);
            for x in 0..65536u64 {
                let legacy = dict_identify(dict, x);
                let inset = legacy.is_some_and(|(id, _)| in_use_contains(st, id));
                let cold = dict_identify_in_use(dict, Some(&cfg), x);
                assert_eq!(cold.is_some(), inset);
                if let Some(c) = cold {
                    assert_eq!(Some(c), legacy);
                    n_acc += 1;
                }
                n_legacy += legacy.is_some() as usize;
            }
            assert!(n_acc > 0 || st.is_empty());
            assert!(n_acc <= n_legacy);
        }
    }
    // the reviewer's counterexample
    let only0 = cold_decode_config(Some(&s1));
    assert!(dict_identify_in_use(DICT_4X4_50, Some(&only0), 0x8d).is_none());
    // empty set (Some, n = 0) accepts nothing; None accepts everything legacy accepts
    let none = cold_decode_config(Some(&[]));
    let code0 = dict_info(DICT_4X4_50).unwrap().codes[0];
    assert!(dict_identify_in_use(DICT_4X4_50, Some(&none), code0).is_none());
    assert!(dict_identify_in_use(DICT_4X4_50, None, code0).is_some());
}

#[test]
fn cold_decode_equals_legacy_on_6x6_250_sample() {
    let d = dict_info(DICT_6X6_250).unwrap();
    let cfg = cold_decode_config(Some(&IDS_0_TO_9));
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut n_acc = 0;
    for k in 0..1_000_000i64 {
        let mut x = next() & ((1u64 << 36) - 1);
        if k & 1 != 0 {
            // half: a codeword with 0..8 random bit flips
            x = d.codes[((next() % 250) * 4 + (next() % 4)) as usize];
            let mut f = next() % 9;
            while f > 0 {
                x ^= 1u64 << (next() % 36);
                f -= 1;
            }
        }
        let legacy = dict_identify(DICT_6X6_250, x);
        let inset = legacy.is_some_and(|(id, _)| id < 10);
        let cold = dict_identify_in_use(DICT_6X6_250, Some(&cfg), x);
        assert_eq!(cold.is_some(), inset);
        if let Some(c) = cold {
            assert_eq!(Some(c), legacy);
            n_acc += 1;
        }
    }
    assert!(n_acc > 0);
}

#[test]
fn cold_decode_image_path_drops_ids_outside_the_set() {
    let (w, h) = (1920, 960);
    let cold = cold_decode_config(Some(&IDS_0_TO_9));
    for id in [3, 23] {
        let mut px = vec![200u8; (w * h) as usize];
        render_test_marker_id(&mut px, w, h, 0.0, 0.0, 6.0, 0.0, id);
        let v = gray_view(&px, w, h);
        let gt = test_marker_corner_rays(0.0, 0.0, 6.0, 0.0);
        let a = decode_rays(&v, DICT_4X4_50, &gt, K_SUB_DEFAULT, None);
        let b = decode_rays(&v, DICT_4X4_50, &gt, K_SUB_DEFAULT, Some(&cold));
        assert_eq!(a.id, id);
        if id < 10 {
            assert_eq!(b.id, id);
            assert_eq!(b.bits, a.bits);
        } else {
            assert!(b.id < 0);
        }
    }
}

fn same_decode(a: &DecodeResult, b: &DecodeResult) -> bool {
    if a.id != b.id
        || a.rot != b.rot
        || a.bits != b.bits
        || a.border_ok != b.border_ok
        || a.reject_code != b.reject_code
        || a.contrast != b.contrast
    {
        return false;
    }
    (0..4).all(|i| a.corners[i] == b.corners[i])
}

#[test]
fn default_config_is_bit_identical_to_the_legacy_decoder() {
    let (w, h) = (1920, 960);
    let dflt = DecodeConfig::default(); // in_use None = every id
    assert!(dflt.in_use.is_none());
    let poses: [(f64, f64, f64, f64, i32); 3] = [
        (137.0, 41.0, 6.0, 0.0, 0),
        (-63.0, -22.0, 5.0, 17.0, 23),
        (0.0, 0.0, 6.0, 90.0, 4),
    ];
    for q in poses {
        let mut px = vec![200u8; (w * h) as usize];
        render_test_marker_id(&mut px, w, h, q.0, q.1, q.2, q.3, q.4);
        let v = gray_view(&px, w, h);
        let gt = test_marker_corner_rays(q.0, q.1, q.2, q.3);
        let c = sum_norm(&gt);
        let mut shrunk = [Vec3::default(); 4];
        for i in 0..4 {
            shrunk[i] = slerp(&c, &gt[i], 0.75);
        }
        for n_sub in [K_SUB_DEFAULT, K_SUB_LEAN] {
            for rays in [&gt, &shrunk] {
                let a = decode_rays(&v, DICT_4X4_50, rays, n_sub, None);
                let b = decode_rays(&v, DICT_4X4_50, rays, n_sub, Some(&dflt));
                assert!(same_decode(&a, &b));
                assert_eq!(a.n_erased, 0);
            }
        }
    }
}

/// Pixel bounding box of the four corner rays.
fn bbox(gt: &[Vec3; 4], w: i32, h: i32) -> (i32, i32, i32, i32) {
    let (mut x0, mut x1, mut y0, mut y1) = (w, 0, h, 0);
    for r in gt {
        let (pxx, pyy) = ray_to_px(r, w, h);
        x0 = x0.min(pxx as i32);
        x1 = x1.max(pxx as i32);
        y0 = y0.min(pyy as i32);
        y1 = y1.max(pyy as i32);
    }
    (x0, x1, y0, y1)
}

#[test]
fn verify_decode_reads_a_known_marker_with_a_bright_patch_on_the_border() {
    let (w, h) = (1920, 960);
    let mut px = vec![200u8; (w * h) as usize];
    render_test_marker_id(&mut px, w, h, 0.0, 0.0, 6.0, 0.0, 3);
    let gt = test_marker_corner_rays(0.0, 0.0, 6.0, 0.0);
    let (x0, x1, y0, y1) = bbox(&gt, w, h);
    let clean = px.clone();
    let vclean = gray_view(&clean, w, h);
    let (cw, ch) = ((x1 - x0) / 6, (y1 - y0) / 6);
    for y in y0 - 1..y0 + ch {
        for x in x0 - 1..x0 + cw {
            px[(y * w + x) as usize] = 255;
        }
    }
    let v = gray_view(&px, w, h);
    let vc = verify_decode_config(Some(&IDS_0_TO_9));
    assert_eq!(vc.border_fail_max, 4);
    assert_eq!(vc.erasure_band, 0.15f32);
    assert_eq!(
        decode_rays(&v, DICT_4X4_50, &gt, K_SUB_DEFAULT, None).reject_code,
        codes::R_BORDER_NOT_BLACK
    ); // legacy
    let cold = cold_decode_config(Some(&IDS_0_TO_9));
    assert_eq!(
        decode_rays(&v, DICT_4X4_50, &gt, K_SUB_DEFAULT, Some(&cold)).reject_code,
        codes::R_BORDER_NOT_BLACK
    );
    let d = decode_verify(&v, DICT_4X4_50, &gt, K_SUB_DEFAULT, 3, -1, &vc);
    assert_eq!(d.reject_code, codes::ACCEPT);
    assert_eq!(d.id, 3);
    assert!(d.border_fail >= 1 && d.border_fail <= 4);
    // verifying against the WRONG id must reject
    assert!(decode_verify(&v, DICT_4X4_50, &gt, K_SUB_DEFAULT, 4, -1, &vc).id < 0);
    let d2 = decode_verify(&vclean, DICT_4X4_50, &gt, K_SUB_DEFAULT, 3, 0, &vc);
    assert_eq!(d2.id, 3);
    assert_eq!(d2.border_fail, 0);
}

#[derive(Default, Clone, Copy)]
struct Counts {
    calls: [i64; 5],
    wrong_id: [i64; 5],
    wrong_rot: [i64; 5],
    true_acc: [i64; 5],
}

#[test]
fn verify_cells_exhaustive_safety() {
    // EXHAUSTIVE over: every erasure set of up to 8 of the 16 interior cells (39203 sets, all
    // that min_known_frac 0.5 lets through), all 40 true (id, rot) of ids 0..9, all 10
    // expected ids, and every error set of up to t corrupted KNOWN cells. wrong = verify
    // accepts a different in-use id or the right id with a wrong rotation. other_margin 3 must
    // give 0 wrong for t <= 3 (asserted); t = 4 is the documented limit (printed), and
    // other_margin 2 (printed) is only safe to t <= 2.
    let d = dict_info(DICT_4X4_50).unwrap();
    let eset: Vec<u64> = (0..65536u64).filter(|m| m.count_ones() <= 8).collect();
    assert_eq!(eset.len(), 39203);
    let run = |om: i32, tid: usize, trot: usize, c: &mut Counts| {
        let mut vc = verify_decode_config(Some(&IDS_0_TO_9));
        vc.other_margin = om;
        let tmax = if om == 3 { 4 } else { 3 }; // the informational other_margin 2 run stops at t = 3
        let code = d.codes[tid * 4 + trot];
        for &er in &eset {
            let mut pos = [0u32; 16];
            let mut nk = 0;
            for b in 0..16u32 {
                if (er >> b) & 1 == 0 {
                    pos[nk] = b;
                    nk += 1;
                }
            }
            // enumerate error sets of size 0..tmax over the known positions
            let mut test = |flips: u64, t: usize| {
                let bits = (code ^ flips) & !er;
                for ex in 0..10i32 {
                    c.calls[t] += 1;
                    let Some(ok) = verify_cells(DICT_4X4_50, &vc, bits, er, ex, -1) else {
                        continue;
                    };
                    if ex as usize != tid {
                        c.wrong_id[t] += 1;
                    } else if ok.rot as usize != trot {
                        c.wrong_rot[t] += 1;
                    } else {
                        c.true_acc[t] += 1;
                    }
                }
            };
            test(0, 0);
            for a in 0..nk {
                test(1u64 << pos[a], 1);
                for b in a + 1..nk {
                    test((1u64 << pos[a]) | (1u64 << pos[b]), 2);
                    for cc in b + 1..nk {
                        test((1u64 << pos[a]) | (1u64 << pos[b]) | (1u64 << pos[cc]), 3);
                        if tmax >= 4 {
                            for dd in cc + 1..nk {
                                test(
                                    (1u64 << pos[a])
                                        | (1u64 << pos[b])
                                        | (1u64 << pos[cc])
                                        | (1u64 << pos[dd]),
                                    4,
                                );
                            }
                        }
                    }
                }
            }
        }
    };
    for om in [3, 2] {
        let mut per = vec![Counts::default(); 40];
        let next_job = AtomicUsize::new(0);
        let nthreads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let results = std::sync::Mutex::new(&mut per);
        std::thread::scope(|s| {
            for _ in 0..nthreads {
                s.spawn(|| loop {
                    let tr = next_job.fetch_add(1, Ordering::Relaxed);
                    if tr >= 40 {
                        break;
                    }
                    let mut c = Counts::default();
                    run(om, tr / 4, tr % 4, &mut c);
                    results.lock().unwrap()[tr] = c;
                });
            }
        });
        let mut tot = Counts::default();
        for c in &per {
            for t in 0..5 {
                tot.calls[t] += c.calls[t];
                tot.wrong_id[t] += c.wrong_id[t];
                tot.wrong_rot[t] += c.wrong_rot[t];
                tot.true_acc[t] += c.true_acc[t];
            }
        }
        for t in 0..(if om == 3 { 5 } else { 4 }) {
            eprintln!(
                "verify other_margin {om}, {t} corrupted known cells: {} calls, wrong id {}, wrong rot {}, true accepts {}",
                tot.calls[t], tot.wrong_id[t], tot.wrong_rot[t], tot.true_acc[t]
            );
        }
        if om == 3 {
            for t in 0..=3 {
                assert_eq!(tot.wrong_id[t], 0);
                assert_eq!(tot.wrong_rot[t], 0);
            }
            assert!(tot.true_acc[0] > 0);
        } else {
            for t in 0..=2 {
                assert_eq!(tot.wrong_id[t], 0);
                assert_eq!(tot.wrong_rot[t], 0);
            }
        }
    }
}

#[test]
fn verify_rejects_every_other_id_on_occluded_renders() {
    // pixel path negative test: occluded renders of id X, verified against every OTHER in-use
    // id, must never accept.
    let (w, h) = (1920usize, 960usize);
    let vc = verify_decode_config(Some(&IDS_0_TO_9));
    let (mut n_other, mut n_true, mut n_true_acc) = (0usize, 0usize, 0usize);
    for id in [3, 7] {
        let mut base = vec![200u8; w * h];
        render_test_marker_id(&mut base, w as i32, h as i32, 0.0, 0.0, 6.0, 0.0, id);
        let gt = test_marker_corner_rays(0.0, 0.0, 6.0, 0.0);
        let (x0, x1, y0, y1) = bbox(&gt, w as i32, h as i32);
        let (cw, ch) = ((x1 - x0) / 6, (y1 - y0) / 6);
        // occluded cell blocks in the 6x6 grid [col0,col1) x [row0,row1): border strip,
        // corner block, interior block
        let blocks: [[i32; 4]; 3] = [[0, 3, 0, 1], [0, 2, 0, 2], [2, 4, 2, 4]];
        for kind in 0..4 {
            for blk in &blocks {
                let mut px = base.clone();
                for y in y0 + blk[2] * ch..y0 + blk[3] * ch {
                    for x in x0 + blk[0] * cw..x0 + blk[1] * cw {
                        let val: u8 = match kind {
                            0 => 0,
                            1 => 255,
                            3 => 120,
                            _ => {
                                if ((x / 3) + (y / 3)) & 1 != 0 {
                                    255
                                } else {
                                    0
                                }
                            }
                        };
                        px[y as usize * w + x as usize] = val;
                    }
                }
                let v = gray_view(&px, w as i32, h as i32);
                for ex in 0..10 {
                    let r = decode_verify(&v, DICT_4X4_50, &gt, K_SUB_DEFAULT, ex, -1, &vc);
                    if ex == id {
                        n_true += 1;
                        n_true_acc += (r.id == id) as usize;
                    } else {
                        n_other += 1;
                        assert!(r.id < 0);
                    }
                }
            }
        }
    }
    assert_eq!(n_other, 2 * 4 * 3 * 9);
    assert!(n_true_acc > 0);
    eprintln!(
        "verify_rejects_every_other_id_on_occluded_renders: {n_other} other-id verifies (all rejected), \
         true id verified in {n_true_acc} of {n_true}"
    );
}

// ---- the per-side border probe the line-fit quads are gated by ------------------------------

#[test]
fn border_min_side_dark_reads_a_clean_ring_and_a_side_that_runs_along_an_occluder() {
    let (w, h) = (3840, 1920);
    let mut px = vec![205u8; (w * h) as usize];
    render_test_marker_id(&mut px, w, h, -100.0, -8.0, 5.0, 13.0, 0);
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(-100.0, -8.0, 5.0, 13.0);
    assert!(border_min_side_dark(&v, DICT_4X4_50, &gt, 2) > 0.99); // every side's ring is black

    // A quad shrunk by one module on one side (its "ring" now lies over the data modules, as
    // when a side is fitted along an occluder's edge) reads low there.
    let mut shrunk = gt;
    for i in 1..=2 {
        // pull corners 1 and 2 towards 0 and 3 by a third
        let o = &gt[if i == 1 { 0 } else { 3 }];
        shrunk[i] = normalize(&Vec3::new(
            gt[i].x + (o.x - gt[i].x) / 3.0,
            gt[i].y + (o.y - gt[i].y) / 3.0,
            gt[i].z + (o.z - gt[i].z) / 3.0,
        ));
    }
    assert!(border_min_side_dark(&v, DICT_4X4_50, &shrunk, 2) < 0.75);

    let degenerate = [gt[0]; 4];
    assert!(border_min_side_dark(&v, DICT_4X4_50, &degenerate, 2) < 0.0); // unmeasurable
}
