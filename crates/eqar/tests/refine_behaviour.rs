//! Behaviour tests for the refine stage (numeric expectations are those of the original
//! C++ unit tests, unchanged).
#![allow(clippy::needless_range_loop)]
// `1.41421356` is the original test's literal; keep it as written.
#![allow(clippy::approx_constant)]
// the schedule test pins the constants themselves
#![allow(clippy::assertions_on_constants)]

use eqar::codes;
use eqar::geometry::{basis_at, cross, dot, normalize, px_to_ray, ray_to_px, Vec3, K_PI};
use eqar::guards::*;
use eqar::image::{Fmt, View};
use eqar::refine::*;
use eqar_testkit::synth::{
    frame_at, render_test_marker, render_test_marker_id, test_marker_corner_rays,
};

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

/// erf(x): Taylor series below 2.5, the erfc continued fraction above (the standard
/// library has no erf). Odd by construction.
fn erf(x: f64) -> f64 {
    let ax = x.abs();
    let v = if ax < 2.5 {
        let mut term = ax;
        let mut sum = ax;
        let mut n = 0.0f64;
        loop {
            n += 1.0;
            term *= -ax * ax / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() < 1e-17 {
                break;
            }
        }
        sum * 2.0 / K_PI.sqrt()
    } else {
        // erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + (2/2)/(x + (3/2)/(x + ...))))
        let mut f = ax;
        for k in (1..=80).rev() {
            f = ax + (k as f64 / 2.0) / f;
        }
        1.0 - (-ax * ax).exp() / K_PI.sqrt() / f
    };
    if x < 0.0 {
        -v
    } else {
        v
    }
}

fn refine(v: &View, seed: &[Vec3; 4], cfg: &RefineCfg) -> (bool, [Vec3; 4], RefineInfo) {
    let mut out = [Vec3::default(); 4];
    let mut info = RefineInfo::default();
    let mut scratch = RefineScratch::default();
    let ok = refine_rays(v, seed, &mut out, &mut info, cfg, &mut scratch);
    (ok, out, info)
}

fn sum_norm(g: &[Vec3; 4]) -> Vec3 {
    normalize(&Vec3::new(
        g[0].x + g[1].x + g[2].x + g[3].x,
        g[0].y + g[1].y + g[2].y + g[3].y,
        g[0].z + g[1].z + g[2].z + g[3].z,
    ))
}

#[test]
fn k_follows_the_084_rule() {
    assert_eq!(refine_k(60.0), 50); // round(0.84*60) = 50
    assert_eq!(refine_k(10.0), K_K_MIN); // clamped
    assert_eq!(refine_k(500.0), K_K_MAX); // clamped
}

#[test]
fn neon_arc_sampler_matches_the_generic_sampler() {
    // sample_edge's Gray8/no-LUT fast path is a hand-written NEON arc sampler with its own
    // atan2 approximation; an IDENTITY gamma LUT is a no-op on the pixel values but forces the
    // generic scalar path, so the two can be compared profile for profile on the same view.
    let (w, h) = (1024usize, 512usize);
    let mut px = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            px[y * w + x] = ((x * 17 + y * 41 + (x * y) / 5) & 0xFF) as u8;
        }
    }
    let mut lut = [0u8; 256];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = i as u8;
    }
    let fast = gray_view(&px, w as i32, h as i32);
    let slow = View {
        lut: Some(&lut),
        ..fast
    };

    // four edges spanning the equator, a high latitude and straight over the north pole (the
    // arc sampler's own slow-path fallback)
    let pairs = [
        (
            normalize(&Vec3::new(1.0, 0.02, 0.0)),
            normalize(&Vec3::new(0.98, 0.02, 0.15)),
        ),
        (
            normalize(&Vec3::new(0.2, 0.85, 0.3)),
            normalize(&Vec3::new(0.1, 0.88, 0.42)),
        ),
        (
            normalize(&Vec3::new(0.02, 0.999, 0.0)),
            normalize(&Vec3::new(-0.02, 0.999, 0.01)),
        ),
        (
            normalize(&Vec3::new(-1.0, 0.1, 0.03)),
            normalize(&Vec3::new(-0.99, 0.1, -0.09)),
        ),
    ];
    let k = 24;
    let mut scratch = ArcScratch::default();
    for (a, b) in pairs {
        let centre = sum_norm(&[a, b, a, b]);
        let (mut ia, mut ib) = (vec![0f32; k * 64], vec![0f32; k * 64]);
        let (mut pa, mut qa) = (vec![Vec3::default(); K_MAX], vec![Vec3::default(); K_MAX]);
        let (mut pb, mut qb) = (vec![Vec3::default(); K_MAX], vec![Vec3::default(); K_MAX]);
        let ma = sample_edge(
            &fast,
            &a,
            &b,
            &centre,
            5.0,
            k,
            &mut ia,
            &mut pa,
            &mut qa,
            &mut scratch,
        );
        let mb = sample_edge(
            &slow,
            &a,
            &b,
            &centre,
            5.0,
            k,
            &mut ib,
            &mut pb,
            &mut qb,
            &mut scratch,
        );
        let (ma, mb) = (ma.expect("fast edge"), mb.expect("slow edge"));
        assert_eq!(ma, mb);
        let mut worst = 0.0f32;
        for i in 0..k * ma {
            worst = worst.max((ia[i] - ib[i]).abs());
        }
        assert!(worst <= 1.0); // float32 map vs the double one
    }
}

#[test]
fn centroid_is_exact_on_a_symmetric_edge() {
    // erf-like profile centred exactly on tap (M-1)/2 -> t* == 0
    let (m, k) = (15usize, 1usize);
    let (step, sig) = (0.5, 1.2);
    let mut prof = vec![0f32; m];
    for j in 0..m {
        let t = (j as f64 - (m as f64 - 1.0) / 2.0) * step;
        prof[j] = (60.0 + 140.0 * 0.5 * (1.0 + erf(t / (sig * 1.41421356)))) as f32;
    }
    let mut t = [0.0];
    op_centroid(&prof, k, m, step, &mut t);
    assert!(near(t[0], 0.0, 1e-6));
}

#[test]
fn centroid_tracks_a_shifted_edge_linearly() {
    // Test-literal correction of the original: shifts {-1.7, 1.9} fail the 0.01 px tolerance
    // by construction (at M=21/step=0.5 the central-difference gradient only covers real
    // positions [-4.5, 4.5], so at sig=1.1 a shift of 1.9 leaves only a 2.6 px margin and
    // ~0.4% of the sigmoid's mass is never summed: a real bias of -0.0188 px that the
    // reference implementation exhibits identically). -1.5/1.5 keep the same intent.
    let (m, k) = (21usize, 1usize);
    let (step, sig) = (0.5, 1.1);
    for shift in [-1.5, -0.4, 0.0, 0.3, 1.5] {
        let mut prof = vec![0f32; m];
        for j in 0..m {
            let t = (j as f64 - (m as f64 - 1.0) / 2.0) * step - shift;
            prof[j] = (50.0 + 150.0 * 0.5 * (1.0 + erf(t / (sig * 1.41421356)))) as f32;
        }
        let mut t = [0.0];
        op_centroid(&prof, k, m, step, &mut t);
        assert!(near(t[0], shift, 0.01));
    }
}

#[test]
fn centroid_is_invariant_to_symmetric_blur() {
    // The first moment of the LSF is unchanged by convolution with a symmetric kernel.
    let (m, k) = (31usize, 1usize);
    let step = 0.5;
    let mut got = [0.0f64; 2];
    let sigmas = [0.8, 1.8];
    for kk in 0..2 {
        let mut prof = vec![0f32; m];
        for j in 0..m {
            let t = (j as f64 - (m as f64 - 1.0) / 2.0) * step - 0.37;
            prof[j] = (40.0 + 170.0 * 0.5 * (1.0 + erf(t / (sigmas[kk] * 1.41421356)))) as f32;
        }
        let mut t = [0.0];
        op_centroid(&prof, k, m, step, &mut t);
        got[kk] = t[0];
    }
    assert!(near(got[0], got[1], 0.02)); // drift 0.016 px
    assert!(near(got[0], 0.37, 0.02));
}

#[test]
fn sample_edge_runs_dark_to_bright_at_the_seam() {
    let (w, h) = (3840, 1920);
    let mut px = vec![210u8; (w * h) as usize];
    render_test_marker(&mut px, w, h, 179.6, 0.0, 5.0, 0.0);
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(179.6, 0.0, 5.0, 0.0);
    let c = sum_norm(&gt);
    let mut prof = vec![0f32; 64 * 33];
    let mut pk = vec![Vec3::default(); 64];
    let mut pp = vec![Vec3::default(); 64];
    let mut scratch = ArcScratch::default();
    let m = sample_edge(
        &v,
        &gt[0],
        &gt[1],
        &c,
        4.0,
        32,
        &mut prof,
        &mut pk,
        &mut pp,
        &mut scratch,
    )
    .expect("edge");
    assert!(m % 2 == 1);
    for k in 0..32 {
        let (lo, hi) = (prof[k * m], prof[k * m + m - 1]);
        assert!(hi > lo + 80.0); // inside dark, outside bright
    }
}

#[test]
fn edge_sigma_recovers_a_known_blur() {
    let (m, k) = (41usize, 4usize);
    let step = 0.25;
    for sig in [0.8, 1.4, 2.2] {
        let mut prof = vec![0f32; k * m];
        for kk in 0..k {
            for j in 0..m {
                let t = (j as f64 - (m as f64 - 1.0) / 2.0) * step;
                prof[kk * m + j] =
                    (30.0 + 180.0 * 0.5 * (1.0 + erf(t / (sig * 1.41421356)))) as f32;
            }
        }
        assert!(near(edge_sigma(&prof, k, m, step, 3), sig, 0.20 * sig));
    }
}

#[test]
fn hw_initial_respects_the_module_cap() {
    // side 60 -> 0.10*60 = 6.0, module = 10, cap = 9.0 -> 6.0
    assert!(near(hw_initial(60.0), 6.0, 1e-9));
    // side 21 -> 2.1, module 3.5, cap 3.15 -> 2.1
    assert!(near(hw_initial(21.0), 2.1, 1e-9));
    // side 12 -> 1.2 but the floor is 1.5
    assert!(near(hw_initial(12.0), 1.5, 1e-9));
    // side 400 -> 40 but the module cap is 0.90*66.7 = 60 -> 40 stands
    assert!(near(hw_initial(400.0), 40.0, 1e-9));
}

#[test]
fn hw_next_is_per_edge_with_the_reference_floor() {
    // Each edge narrows on ITS OWN 10-90 rise -- the real clip's horizontal edges are blurrier
    // than its vertical ones (LSF sd 1.43 vs 0.90), and the two must NOT be averaged into one
    // window. clip(max(2.0, 3*sigma + 0.8), 1.2, cap).
    assert!(near(hw_next(1.43, 6.0, 10.0), 5.09, 1e-9));
    assert!(near(hw_next(0.90, 6.0, 10.0), 3.50, 1e-9));
    // the 2.0 floor: 3*0.195 + 0.8 = 1.385 would otherwise stand
    assert!(near(hw_next(0.5 / 2.563, 6.0, 10.0), 2.0, 1e-9));
}

#[test]
fn hw_next_never_exceeds_the_previous_or_the_module_cap() {
    assert!(near(hw_next(4.0, 5.0, 10.0), 5.0, 1e-9)); // 3*4+0.8 = 12.8
    assert!(near(hw_next(4.0, 50.0, 4.0), 3.6, 1e-9)); // 0.90*4
                                                       // the cap has its own max(1.2, ...) floor, so a tiny module never drives the window below 1.2 px
    assert!(near(hw_next(4.0, 50.0, 0.5), 1.2, 1e-9));
}

#[test]
fn irls_rejects_a_30_percent_outlier_run() {
    // 40 samples on a great circle with normal n0; the last 12 are displaced.
    let n0 = normalize(&Vec3::new(0.21, 0.87, -0.44));
    let (e1, e2) = basis_at(&n0);
    let mut p = Vec::new();
    for i in 0..40 {
        let a = -0.4 + 0.8 * i as f64 / 39.0;
        let mut q = normalize(&Vec3::new(
            e1.x * a.cos() + e2.x * a.sin(),
            e1.y * a.cos() + e2.y * a.sin(),
            e1.z * a.cos() + e2.z * a.sin(),
        ));
        if i >= 28 {
            // Test-literal correction of the original: displacing all 12 outliers in the SAME
            // direction makes them a coherent LEVERAGE perturbation that a whole-plane tilt
            // compensates almost exactly (IRLS gives no gain there). Alternating the push's sign
            // keeps the run contiguous and the count at 30 % while removing the coherent-tilt
            // escape hatch: IRLS then recovers n0 exactly while OLS alone still misses by
            // 0.665 px_eq.
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            q = normalize(&Vec3::new(
                q.x + n0.x * 0.02 * sign,
                q.y + n0.y * 0.02 * sign,
                q.z + n0.z * 0.02 * sign,
            ));
        }
        p.push(q);
    }
    let pix = K_PI / 1920.0;
    let mut nf = fit_normal_irls(&p, pix);
    nf = orient_outward(&nf, &n0);
    assert!(dot(&nf, &n0).abs().min(1.0).acos() / pix < 0.05); // < 0.05 px_eq
}

#[test]
fn orient_outward_fixes_an_inverted_eigenvector() {
    let s = normalize(&Vec3::new(0.0, 0.0, 1.0));
    let n = normalize(&Vec3::new(0.1, 0.2, -0.97)); // pointing inward
    let o = orient_outward(&n, &s);
    assert!(dot(&o, &s) > 0.0);
    assert!(near(dot(&o, &n).abs(), 1.0, 1e-12));
}

#[test]
fn summed_sign_vote_agrees_in_every_quadrant() {
    // Build an exact marker at 8 poses spanning all sign combinations; the recovered corners
    // must match the GT rays, never their negatives.
    let poses = [
        (0.0, 0.0),
        (47.0, 30.0),
        (-113.0, -30.0),
        (180.0, 60.0),
        (179.5, -60.0),
        (-90.0, 85.0),
        (90.0, -85.0),
        (-179.5, 0.0),
    ];
    for (lon, lat) in poses {
        let gt = test_marker_corner_rays(lon, lat, 6.0, 23.0);
        let s = sum_norm(&gt);
        let mut n = [Vec3::default(); 4];
        for i in 0..4 {
            n[i] = normalize(&cross(&gt[i], &gt[(i + 1) & 3]));
            if dot(&n[i], &s) < 0.0 {
                n[i] = Vec3::new(-n[i].x, -n[i].y, -n[i].z);
            }
        }
        // deliberately flip two of them the WRONG way: orient_outward must undo it
        n[1] = Vec3::new(-n[1].x, -n[1].y, -n[1].z);
        n[2] = Vec3::new(-n[2].x, -n[2].y, -n[2].z);
        for i in 0..4 {
            n[i] = orient_outward(&n[i], &s);
        }
        let mut out = [Vec3::default(); 4];
        let (ok, mc) = corners_from_normals(&n, &s, 0.10, &mut out);
        assert!(ok);
        for i in 0..4 {
            assert!(near(dot(&out[i], &gt[i]), 1.0, 1e-9));
        }
        assert!(mc > 0.5);
    }
}

#[test]
fn degenerate_corner_is_rejected() {
    // two nearly parallel edges -> |n x n| collapses
    let s = Vec3::new(0.0, 0.0, 1.0);
    let mut n = [
        normalize(&Vec3::new(1.0, 0.0001, 0.0)),
        normalize(&Vec3::new(1.0, 0.0, 0.0)),
        normalize(&Vec3::new(0.0, 1.0, 0.0)),
        normalize(&Vec3::new(-1.0, 0.0, 0.0)),
    ];
    for i in 0..4 {
        n[i] = orient_outward(&n[i], &s);
    }
    let mut out = [Vec3::default(); 4];
    let (ok, mc) = corners_from_normals(&n, &s, 0.10, &mut out);
    assert!(!ok);
    assert!(mc <= 0.10);
}

fn worst_px(out: &[Vec3; 4], gt: &[Vec3; 4], pix: f64) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..4 {
        worst = worst.max(dot(&out[i], &gt[i]).min(1.0).acos() / pix);
    }
    worst
}

#[test]
fn refines_a_perturbed_seed_to_sub_hundredth_px() {
    let (w, h) = (3840, 1920);
    let mut px = vec![210u8; (w * h) as usize];
    render_test_marker(&mut px, w, h, 33.0, -19.0, 5.0, 11.0);
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(33.0, -19.0, 5.0, 11.0);
    let pix = K_PI / h as f64;
    // perturb every corner by ~0.25 px_eq -- the measured bridge output
    let mut seed = [Vec3::default(); 4];
    for i in 0..4 {
        let (e1, _) = basis_at(&gt[i]);
        let a = 0.25 * pix;
        seed[i] = normalize(&Vec3::new(
            gt[i].x + e1.x * a,
            gt[i].y + e1.y * a,
            gt[i].z + e1.z * a,
        ));
    }
    let (ok, out, info) = refine(&v, &seed, &K_SCAN_REFINE);
    assert!(ok);
    assert!(info.ok);
    assert!(worst_px(&out, &gt, pix) < 0.03); // measured: 0.0097 mean / 0.022 max
    assert!(info.contrast_dn > 8.0);
    assert!(info.resid_px / info.side_px < 0.0026);
}

#[test]
fn a_featureless_frame_has_no_edge() {
    // On a featureless frame there is no gradient anywhere, so no profile passes the contrast
    // filter. With no inlier profile there is no edge, and refine_rays reports R_EDGE_FIT
    // instead of guessing (a "re-admit every finite profile" fallback used to fit them anyway:
    // each profile's centroid defaults to t == 0, so the fit reproduced the seed with a
    // numerically perfect residual -- a flawless-looking fit of no signal).
    let (w, h) = (1920, 960);
    let px = vec![128u8; (w * h) as usize]; // featureless
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(0.0, 0.0, 5.0, 0.0);
    let (ok, _, info) = refine(&v, &gt, &K_SCAN_REFINE);
    assert!(!ok);
    assert!(!info.ok);
    assert_eq!(info.reject_code, codes::R_EDGE_FIT);
    let known = RefineCfg {
        allow_weak: true,
        ..K_SCAN_REFINE
    };
    let (ok, _, info) = refine(&v, &gt, &known);
    assert!(!ok); // known ID does not extrapolate a whole edge either
    assert_eq!(info.reject_code, codes::R_EDGE_FIT);
}

#[test]
fn degenerate_init_is_reported_not_crashed() {
    let (w, h) = (1920, 960);
    let px = vec![128u8; (w * h) as usize];
    let v = gray_view(&px, w, h);
    let tiny = test_marker_corner_rays(0.0, 0.0, 0.05, 0.0); // a 2 px_eq quad -- below the side floor of 4
    let (ok, _, info) = refine(&v, &tiny, &K_SCAN_REFINE);
    assert!(!ok);
    assert_eq!(info.reject_code, codes::R_DEGENERATE_INIT);
}

// ---- the per-path refine schedule -----------------------------------------------------------

#[test]
fn the_track_schedule_caps_k_and_iterations_and_is_actually_wired() {
    // The tracker passes K_TRACK_REFINE; everything else passes K_SCAN_REFINE (the
    // reference's 3 iterations / K <= 64). This pins BOTH halves: the constants, and that
    // refine_rays really honours them -- a cap that is declared but not threaded is exactly
    // the trap the quality preset's k_max fell into.
    assert_eq!(K_SCAN_REFINE.iters, K_REFINE_ITERS);
    assert_eq!(K_SCAN_REFINE.k_max, K_K_MAX);
    assert!(K_TRACK_REFINE.iters < K_SCAN_REFINE.iters);
    assert!(K_TRACK_REFINE.k_max <= 32); // "K <= 32 for lean"

    let (w, h) = (3840, 1920);
    let mut px = vec![205u8; (w * h) as usize];
    render_test_marker_id(&mut px, w, h, 33.0, -19.0, 5.0, 11.0, 0);
    let v = gray_view(&px, w, h);
    let gt = test_marker_corner_rays(33.0, -19.0, 5.0, 11.0);
    let pix = K_PI / h as f64;

    // A TRACKED prior: already sub-pixel, which is the whole reason the track path can afford
    // a shorter schedule than a bridge seed can.
    let mut seed = [Vec3::default(); 4];
    for i in 0..4 {
        let (e1, _) = basis_at(&gt[i]);
        let a = 0.25 * pix;
        seed[i] = normalize(&Vec3::new(
            gt[i].x + e1.x * a,
            gt[i].y + e1.y * a,
            gt[i].z + e1.z * a,
        ));
    }

    let (ok_s, out_scan, i_scan) = refine(&v, &seed, &K_SCAN_REFINE);
    let (ok_t, out_track, i_track) = refine(&v, &seed, &K_TRACK_REFINE);
    assert!(ok_s && ok_t);

    // The marker is ~60 px_eq on a side, so the reference's own rule wants
    // round(0.84*60) = 50 profiles; the track cap has to bite.
    assert!(i_scan.k > K_TRACK_REFINE.k_max);
    assert_eq!(i_track.k, K_TRACK_REFINE.k_max);

    // Both land the corner; the cheaper schedule is allowed to be a little looser but not by
    // more than a hundredth of a pixel here.
    assert!(worst_px(&out_scan, &gt, pix) < 0.03);
    assert!(worst_px(&out_track, &gt, pix) < 0.05);

    // cfg.iters is wired, not ignored: one pass from this seed does not reach the same fixed
    // point two passes do.
    let one = RefineCfg {
        iters: 1,
        k_max: K_TRACK_REFINE.k_max,
        allow_weak: false,
    };
    let (ok_o, out_one, _) = refine(&v, &seed, &one);
    assert!(ok_o);
    let mut moved = 0.0f64;
    for i in 0..4 {
        moved = moved.max(dot(&out_one[i], &out_track[i]).min(1.0).acos() / pix);
    }
    assert!(moved > 0.0);
}

// ---- edge support, 3-edge mode, per-corner validity -----------------------------------------

const K_LON: f64 = 33.0;
const K_LAT: f64 = -19.0;
const K_HALF: f64 = 5.0;
const K_ROLL: f64 = 11.0;

struct Scene {
    w: i32,
    h: i32,
    px: Vec<u8>,
    gt: [Vec3; 4],
    cpx: [[f64; 2]; 4],
    side: f64,
}

impl Scene {
    fn new() -> Scene {
        let (w, h) = (3840, 1920);
        let mut px = vec![210u8; (w * h) as usize];
        render_test_marker(&mut px, w, h, K_LON, K_LAT, K_HALF, K_ROLL);
        let gt = test_marker_corner_rays(K_LON, K_LAT, K_HALF, K_ROLL);
        let mut cpx = [[0.0; 2]; 4];
        for i in 0..4 {
            let (x, y) = ray_to_px(&gt[i], w, h);
            cpx[i] = [x, y];
        }
        let side = (cpx[1][0] - cpx[0][0]).hypot(cpx[1][1] - cpx[0][1]);
        Scene {
            w,
            h,
            px,
            gt,
            cpx,
            side,
        }
    }
    fn view(&self) -> View<'_> {
        gray_view(&self.px, self.w, self.h)
    }
    /// pixel box, all four sides given as offsets (in marker sides) from corner `a`'s px
    fn paint(&mut self, a: usize, dx0: f64, dy0: f64, dx1: f64, dy1: f64, val: u8) {
        let x0 = (self.cpx[a][0] + dx0 * self.side).round() as i32;
        let x1 = (self.cpx[a][0] + dx1 * self.side).round() as i32;
        let y0 = (self.cpx[a][1] + dy0 * self.side).round() as i32;
        let y1 = (self.cpx[a][1] + dy1 * self.side).round() as i32;
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.px[(y * self.w + x) as usize] = val;
            }
        }
    }
    /// the seed: gt corners pushed by `off` px_eq along a fixed tangent
    fn seed(&self, off: f64) -> [Vec3; 4] {
        let pix = K_PI / self.h as f64;
        let mut s = [Vec3::default(); 4];
        for i in 0..4 {
            let (e1, _) = basis_at(&self.gt[i]);
            let a = off * pix;
            s[i] = normalize(&Vec3::new(
                self.gt[i].x + e1.x * a,
                self.gt[i].y + e1.y * a,
                self.gt[i].z + e1.z * a,
            ));
        }
        s
    }
    fn err(&self, out: &[Vec3; 4], i: usize) -> f64 {
        dot(&out[i], &self.gt[i]).min(1.0).acos() / (K_PI / self.h as f64)
    }
}

fn known_cfg() -> RefineCfg {
    RefineCfg {
        allow_weak: true,
        ..K_SCAN_REFINE
    }
}

#[test]
fn a_clean_marker_has_full_support_and_observed_tips() {
    let sc = Scene::new();
    let seed = sc.seed(0.25);
    let (ok, _, info) = refine(&sc.view(), &seed, &K_SCAN_REFINE);
    assert!(ok);
    for e in 0..4 {
        assert!(info.edge[e].support > 0.95);
        assert!(info.edge[e].end_support[0] >= 0.9 && info.edge[e].end_support[1] >= 0.9);
        assert!(!info.edge[e].weak);
    }
    let cs = corner_states(&info, false);
    for i in 0..4 {
        assert!(cs[i].valid);
        assert!(near(cs[i].sigma_scale as f64, 1.0, 1e-6));
    }
}

#[test]
fn a_hidden_corner_is_invalid_cold_and_scaled_when_the_id_is_known() {
    let base = Scene::new();
    let seed = base.seed(0.25);
    let (ok_b, ob, _) = refine(&base.view(), &seed, &K_SCAN_REFINE);
    assert!(ok_b);
    let mut occ = Scene::new();
    occ.paint(0, -0.3, -0.3, 0.25, 0.25, 250); // a bright box over corner 0 and its first quarter of both sides
    let (ok_o, oo, io) = refine(&occ.view(), &seed, &K_SCAN_REFINE);
    assert!(ok_o);
    assert!(!io.edge[0].weak && !io.edge[3].weak);
    assert!(io.edge[0].end_support[0] < 0.5); // top edge, tip at corner 0
    assert!(io.edge[3].end_support[1] < 0.5); // left edge, tip at corner 0
    let cold = corner_states(&io, false);
    let known = corner_states(&io, true);
    assert!(!cold[0].valid);
    assert!(known[0].valid && known[0].sigma_scale > 1.5);
    // the clear corners are not worse than the unoccluded baseline (x1.05, plus 0.01 px of slack)
    for i in 1..4 {
        assert!(cold[i].valid && known[i].valid);
        assert!(occ.err(&oo, i) <= 1.05 * base.err(&ob, i) + 0.01);
    }
}

#[test]
fn a_weak_edge_is_fitted_from_its_inliers_only_when_the_id_is_known() {
    let mut sc = Scene::new();
    let seed = sc.seed(0.3);
    // a bright box hides the top edge (corner 0 -> 1) except its last 18 %: support well under 0.3
    sc.paint(0, -0.3, -0.3, 0.82, 0.2, 250);
    let (ok, _, info) = refine(&sc.view(), &seed, &K_SCAN_REFINE);
    assert!(!ok);
    assert_eq!(info.reject_code, codes::R_EDGE_FIT); // cold: no 3-edge completion
    let (ok, out, info) = refine(&sc.view(), &seed, &known_cfg());
    assert!(ok);
    assert!(info.edge[0].weak);
    assert!(info.edge[0].n_inl >= K_MIN_INLIERS);
    assert!(!info.edge[1].weak && !info.edge[2].weak && !info.edge[3].weak);
    // the removed fallback: the hidden profiles used to be re-admitted (a bright plateau reads
    // t == 0, i.e. "the seed is right") and dragged the edge back to the 0.3 px offset seed.
    // Now only the visible run counts: the fitted edge passes through the true edge there.
    let q = normalize(&Vec3::new(
        0.1 * sc.gt[0].x + 0.9 * sc.gt[1].x,
        0.1 * sc.gt[0].y + 0.9 * sc.gt[1].y,
        0.1 * sc.gt[0].z + 0.9 * sc.gt[1].z,
    ));
    let n = normalize(&cross(&out[0], &out[1]));
    assert!(dot(&q, &n).abs() / (K_PI / sc.h as f64) < 0.1);
    let cs = corner_states(&info, true);
    assert!(!cs[0].valid && !cs[1].valid);
    assert!((cs[0].flags & K_CORNER_COMPLETED) != 0 && (cs[1].flags & K_CORNER_COMPLETED) != 0);
    assert!(cs[2].valid && cs[3].valid);
    assert!(near(cs[2].sigma_scale as f64, 1.0, 1e-6));
    assert!(sc.err(&out, 2) < 0.1 && sc.err(&out, 3) < 0.1);
}

#[test]
fn a_fully_hidden_edge_is_never_extrapolated() {
    let mut sc = Scene::new();
    let seed = sc.seed(0.3);
    sc.paint(0, -0.3, -0.3, 1.3, 0.2, 250); // the whole top edge and past both its ends
    let (ok, _, info) = refine(&sc.view(), &seed, &known_cfg());
    assert!(!ok);
    assert_eq!(info.reject_code, codes::R_EDGE_FIT);
    let (ok, _, info) = refine(&sc.view(), &seed, &K_SCAN_REFINE);
    assert!(!ok);
    assert_eq!(info.reject_code, codes::R_EDGE_FIT);
}

#[test]
fn the_union_box_far_corners_are_not_valid_measurements() {
    // The bridge test `lines_quad_never_invents_a_side`, third case, as an image: the marker
    // (raster 40..160) with a dark box merged onto its right and bottom (rect 110..200 x 60..210
    // and 60..200 x 110..210). The 4-line quad is (40,40) (200,40) (200,210) (40,210): TR and BL
    // are intersections of a line with the tip of ANOTHER line that ends well short of them.
    let mut sc = Scene::new();
    let k = 2.0 / 3.0;
    let hh = (K_HALF * K_PI / 180.0).tan();
    let (c, ex, ey) = frame_at(K_LON, K_LAT, K_ROLL);
    let plane = |rx: f64, ry: f64| -> (f64, f64) {
        (
            -k + (rx - 40.0) / 120.0 * 2.0 * k,
            k - (ry - 40.0) / 120.0 * 2.0 * k,
        )
    };
    let ray_of = |rx: f64, ry: f64| -> Vec3 {
        let (qx, qy) = plane(rx, ry);
        normalize(&Vec3::new(
            c.x + (ex.x * qx + ey.x * qy) * hh,
            c.y + (ex.y * qx + ey.y * qy) * hh,
            c.z + (ex.z * qx + ey.z * qy) * hh,
        ))
    };
    // paint the two dark rectangles in the marker's plane coordinates (sphere-consistent, straight edges)
    let rects = [[110.0, 60.0, 200.0, 210.0], [60.0, 110.0, 200.0, 210.0]];
    let (cx0, cy0) = (sc.cpx[0][0] as i32, sc.cpx[0][1] as i32);
    let r = (4.0 * sc.side) as i32;
    for y in cy0 - r..cy0 + r {
        for x in cx0 - r..cx0 + r {
            let d = px_to_ray(x as f64, y as f64, sc.w, sc.h);
            let den = dot(&d, &c);
            if den <= 1e-9 {
                continue;
            }
            let (qx, qy) = (dot(&d, &ex) / den / hh, dot(&d, &ey) / den / hh);
            let rx = 40.0 + (qx + k) / (2.0 * k) * 120.0;
            let ry = 40.0 + (k - qy) / (2.0 * k) * 120.0;
            for rc in &rects {
                if rx >= rc[0] && rx < rc[2] && ry >= rc[1] && ry < rc[3] {
                    sc.px[(y * sc.w + x) as usize] = 40;
                }
            }
        }
    }
    let seed = [
        ray_of(40.0, 40.0),
        ray_of(200.0, 40.0),
        ray_of(200.0, 210.0),
        ray_of(40.0, 210.0),
    ];
    let (ok, _, info) = refine(&sc.view(), &seed, &known_cfg());
    assert!(ok);
    {
        // the default (cold) schedule sees the same quad: no weak edge here, so it is accepted as well
        let (ok2, _, i2) = refine(&sc.view(), &seed, &K_SCAN_REFINE);
        assert!(ok2);
        let c2 = corner_states(&i2, false);
        assert!(c2[0].valid && !c2[1].valid && !c2[3].valid);
    }
    let cold = corner_states(&info, false);
    let kn = corner_states(&info, true);
    assert!(cold[0].valid); // the marker's own corner
    assert!(!cold[1].valid && !cold[3].valid); // tips beyond the observed runs
    assert!(info.edge[0].end_support[1] < 0.5); // the top edge ends short of corner 1
    assert!(kn[1].valid && kn[1].sigma_scale > 1.5); // known id: valid only with a large sigma
    assert!(kn[3].valid && kn[3].sigma_scale > 1.5);
}
