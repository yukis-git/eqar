//! Behaviour tests for the bridge stage (numeric expectations are those of the original
//! C++ unit tests, unchanged).
#![allow(clippy::needless_range_loop)]
// `0.70710678` is the original test's literal; keep it as written.
#![allow(clippy::approx_constant)]

use eqar::bridge::*;
use eqar::geometry::{dot, norm, normalize, Vec3};
use eqar::image::{Fmt, View};
use eqar_testkit::synth::{
    frame_at, render_test_marker, render_test_marker_id, test_marker_center_ray,
    test_marker_corner_rays,
};
use std::f64::consts::PI;

const DEG: f64 = PI / 180.0;

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn gray_view(px: &[u8], w: usize, h: usize) -> View<'_> {
    View {
        data: px,
        w: w as i32,
        h: h as i32,
        stride: w,
        fmt: Fmt::Gray8,
        lut: None,
    }
}

fn max_patch_len() -> usize {
    (K_BRIDGE_MAX_PATCH * K_BRIDGE_MAX_PATCH) as usize
}

fn new_patch() -> Patch {
    Patch {
        img: vec![0; max_patch_len()],
        ..Patch::default()
    }
}

fn unscreened() -> [BridgePass; 2] {
    [K_SCAN_PLAN[1], K_SCAN_PLAN[2]]
}

#[test]
fn patch_side_matches_the_reference_rule() {
    // half_fov 6 deg at H = 1920: 2*tan(6)*611.15*1.6 = 205.6 -> 206; 2*tan(6 deg)*(1920/pi)
    // = 128.5 native px across the patch, x the angular rate: the scan rate is the
    // reference's 1.6 and the track plan renders its single pass at 0.6, so 128.5*0.6 = 77.1
    // -> 77.
    let m = K_BRIDGE_MAX_PATCH;
    assert_eq!(patch_side(6.0 * DEG, 1920, K_BRIDGE_OVERSAMPLE, m), 206);
    assert_eq!(
        patch_side(6.0 * DEG, 1920, K_BRIDGE_TRACK_OVERSAMPLE, m),
        77
    );
    assert_eq!(
        patch_side(0.01 * DEG, 1920, K_BRIDGE_OVERSAMPLE, m),
        K_BRIDGE_MIN_PATCH
    ); // clamped low
    assert_eq!(patch_side(70.0 * DEG, 1920, K_BRIDGE_OVERSAMPLE, m), m); // clamped high
}

#[test]
fn the_scan_plan_matches_the_reference_on_both_passes() {
    // The scan plan is the one the reference agreement rests on: the refiner's K and hw both
    // come from the seed's own measured side, so any change to what Bridge::run returns is
    // a change to the refinement schedule. Lowering pass 1's rate has been tried and
    // reverted twice; this test makes a third attempt delete it deliberately.
    //
    // The leading SCREEN pass is exempt for one structural reason: its quad is discarded and
    // run() leaves `vc`/`r` at the caller's values, so the two REAL passes render exactly
    // the framing they would have without it. Everything below still pins the reference's
    // rates and boxes.
    assert_eq!(K_SCAN_PLAN.len(), 3);
    assert_eq!(K_SCAN_PLAN_PASSES, 3);
    assert!(K_SCAN_PLAN[0].screen);
    assert!(!K_SCAN_PLAN[1].screen);
    assert!(!K_SCAN_PLAN[2].screen);
    assert_eq!(K_SCAN_PLAN[1].box_, K_BRIDGE_BOX);
    assert_eq!(K_SCAN_PLAN[2].box_, K_BRIDGE_BOX2);
    assert_eq!(K_SCAN_PLAN[1].oversample, K_BRIDGE_OVERSAMPLE);
    assert_eq!(K_SCAN_PLAN[2].oversample, K_BRIDGE_OVERSAMPLE);
    // The caps keep two sweep steps of margin from the fidelity cliff on each pass.
    assert_eq!(K_SCAN_PLAN[1].max_side, K_BRIDGE_SCAN_CAP1);
    assert_eq!(K_SCAN_PLAN[2].max_side, K_BRIDGE_SCAN_CAP2);
    // The track plan is untouched by all of it: one pass, no screen, no cap.
    assert_eq!(K_TRACK_PLAN.len(), 1);
    assert!(!K_TRACK_PLAN[0].screen);
    assert_eq!(K_TRACK_PLAN[0].max_side, K_BRIDGE_MAX_PATCH);
}

#[test]
fn patch_side_cap_is_a_rate_reduction_over_the_same_field_of_view() {
    // Capping the SIDE and lowering the RATE are the same operation: tan_t does not depend
    // on the rate, so a capped render is a render at the effective rate cap/S_native over
    // an IDENTICAL field of view. The floor still wins over the cap.
    let hf = 6.0 * DEG;
    let m = K_BRIDGE_MAX_PATCH;
    let o = K_BRIDGE_OVERSAMPLE;
    assert_eq!(patch_side(hf, 1920, o, m), 206);
    assert_eq!(patch_side(hf, 1920, o, 128), 128);
    assert_eq!(patch_side(hf, 1920, o, 1000), 206); // cap above native
    assert_eq!(patch_side(hf, 1920, o, 8), K_BRIDGE_MIN_PATCH);
    // A capped render is bit-identical to the uncapped render at the rate the cap implies:
    // same S, same tan_t, same grid.
    let s = patch_side(hf, 1920, o, 128);
    assert_eq!(patch_side(hf, 1920, o * s as f64 / 206.0, m), s);
}

#[test]
fn patch_to_ray_round_trips_the_centre_and_corners() {
    let mut g = BridgeGrid::new();
    let px = vec![128u8; 512 * 256];
    let v = gray_view(&px, 512, 256);
    let mut p = new_patch();
    patch_render(
        &v,
        &normalize(&Vec3::new(0.3, 0.6, -0.7)),
        8.0 * DEG,
        &mut g,
        &mut p,
        K_BRIDGE_OVERSAMPLE,
        K_BRIDGE_MAX_PATCH,
    );
    let mid = (p.s - 1) as f64 * 0.5;
    let c = patch_to_ray(&p, mid, mid);
    assert!(near(dot(&c, &p.e3), 1.0, 1e-9));
    let k = patch_to_ray(&p, 0.0, 0.0);
    assert!(near(norm(&k), 1.0, 1e-12));
    assert!(dot(&k, &p.e3).clamp(-1.0, 1.0).acos() > 8.0 * DEG); // corner beyond the edge FOV
}

#[test]
fn patch_is_seam_and_pole_transparent() {
    // A view whose value depends only on latitude: a gnomonic patch centred on the seam and
    // one centred 90 deg away must have the same radial profile.
    let (w, h) = (1024usize, 512usize);
    let mut px = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            px[y * w + x] = (30 + (y * 200) / h) as u8;
        }
    }
    let v = gray_view(&px, w, h);
    let mut g = BridgeGrid::new();
    let (mut p1, mut p2, mut p3) = (new_patch(), new_patch(), new_patch());
    let (o, m) = (K_BRIDGE_OVERSAMPLE, K_BRIDGE_MAX_PATCH);
    patch_render(
        &v,
        &Vec3::new(-1.0, 0.0, 0.0),
        10.0 * DEG,
        &mut g,
        &mut p1,
        o,
        m,
    ); // lon 180
    patch_render(
        &v,
        &Vec3::new(0.0, 0.0, 1.0),
        10.0 * DEG,
        &mut g,
        &mut p2,
        o,
        m,
    ); // lon 90
    assert_eq!(p1.s, p2.s);
    let n = (p1.s * p1.s) as usize;
    let mut worst = 0i32;
    for i in 0..n {
        worst = worst.max((p1.img[i] as i32 - p2.img[i] as i32).abs());
    }
    assert!(worst <= 2); // the seam is not a special case
                         // and a patch centred on the north pole is radially symmetric
    patch_render(
        &v,
        &Vec3::new(0.0, 1.0, 0.0),
        20.0 * DEG,
        &mut g,
        &mut p3,
        o,
        m,
    );
    let s = p3.s as usize;
    let mid = ((p3.s - 1) / 2) as usize;
    let d = s / 5;
    let at = |y: usize, x: usize| p3.img[y * s + x] as i32;
    assert!((at(mid - d, mid) - at(mid + d, mid)).abs() <= 2);
    assert!((at(mid, mid - d) - at(mid, mid + d)).abs() <= 2);
}

#[test]
fn grid_is_cached_not_rebuilt() {
    let mut g = BridgeGrid::new();
    let p1 = g.unit(206).as_ptr();
    let u1 = g.unit(206).to_vec();
    let p2 = g.unit(206).as_ptr();
    assert_eq!(p1, p2); // same storage -> cached
    assert!(near(u1[0] as f64, (0.5 / 206.0) * 2.0 - 1.0, 1e-7));
    assert!(near(u1[205] as f64, (205.5 / 206.0) * 2.0 - 1.0, 1e-7));
}

#[test]
fn otsu_exact_splits_two_clusters() {
    let v = [10.0f32, 11.0, 12.0, 13.0, 200.0, 201.0, 202.0, 203.0];
    let mut sc = [0f32; 8];
    let t = otsu_exact(&v, &mut sc);
    assert!(t > 13.0 && t < 200.0);
    let flat = [77.0f32; 4];
    let mut sc2 = [0f32; 4];
    assert!(otsu_exact(&flat, &mut sc2) > 77.0); // degenerate: never split a tie
}

#[test]
fn flatten_removes_a_linear_illumination_ramp() {
    // A dark square on a bright field, plus a 75 % peak-to-peak left-right ramp.
    let mut p = Patch {
        s: 120,
        img: vec![0; 120 * 120],
        flat: vec![0.0; 120 * 120],
        ..Patch::default()
    };
    for y in 0..120usize {
        for x in 0..120usize {
            let base = if (30..90).contains(&x) && (30..90).contains(&y) {
                40.0
            } else {
                200.0
            };
            let ramp = 0.45 + 0.75 * x as f64 / 119.0; // 0.45 .. 1.20
            p.img[y * 120 + x] = (base * ramp).min(255.0) as u8;
        }
    }
    // BEFORE flattening Otsu splits left/right, not marker/background:
    let raw: Vec<f32> = p.img.iter().map(|&b| b as f32).collect();
    let mut sc = vec![0f32; 120 * 120];
    let t_raw = otsu_exact(&raw, &mut sc);
    // (The original's brief-specified pre-flatten sanity check -- "the marker's own
    // right-hand column is BRIGHTER than the field's left edge" -- is mathematically
    // unsatisfiable by these constants: the marker's brightest reachable value is 40 * 1.2
    // = 48, the field's dimmest 200 * 0.45 = 90. It was deleted there, not weakened.)
    let mut scratch = Scratch::default();
    patch_flatten(&mut p, &mut scratch);
    let t_flat = otsu_exact(&p.flat[..120 * 120], &mut sc);
    // AFTER flattening every marker pixel is below threshold and every field pixel is above
    // it, at BOTH ends of the ramp.
    assert!((p.flat[60 * 120 + 35] as f64) < t_flat);
    assert!((p.flat[60 * 120 + 85] as f64) < t_flat);
    assert!(p.flat[60 * 120 + 5] as f64 > t_flat);
    assert!(p.flat[60 * 120 + 115] as f64 > t_flat);
    assert!(t_raw > 0.0);
}

// The direct FIR convolution at radius ~S/2 was O(S^3) -- measured 767 ms at S = 768, ~250x
// this stage's per-candidate budget. The O(S^2) box-decimate + small (<=64x64) field blur +
// bilinear upsample replacement is expected to take ~1 ms there; 30 ms is a generous 25x
// margin, tight enough to catch an O(S^3) reintroduction. (Debug builds, which the project
// does not run its tests in, get 10x.)
#[test]
fn flatten_is_fast_at_the_max_patch_size() {
    let s = K_BRIDGE_MAX_PATCH as usize;
    let mut p = Patch {
        s: s as i32,
        img: vec![0; s * s],
        flat: vec![0.0; s * s],
        ..Patch::default()
    };
    for y in 0..s {
        for x in 0..s {
            p.img[y * s + x] = ((x * 3 + y * 7) & 0xFF) as u8;
        }
    }
    let mut scratch = Scratch::default();
    let t0 = std::time::Instant::now();
    patch_flatten(&mut p, &mut scratch);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let limit = if cfg!(debug_assertions) { 300.0 } else { 30.0 };
    // wall-clock: too noisy on shared CI runners (GitHub sets CI=true)
    if std::env::var_os("CI").is_none() {
        assert!(ms < limit, "flatten took {ms} ms");
    }
}

// patch_render's Gray8/no-LUT fast path is a hand-written NEON kernel with its own atan2
// approximation and its own bilinear gather; the generic sampler is only reached for a
// BGR/RGB or gamma-LUT view. An IDENTITY LUT is a no-op on the pixel values but forces that
// generic path, so the two can be compared pixel for pixel on the same view. (Without NEON
// both are the scalar path and the comparison is trivially exact.)
#[test]
fn neon_gray_render_matches_the_generic_sampler() {
    let (w, h) = (1024usize, 512usize);
    let mut px = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            px[y * w + x] = ((x * 13 + y * 29 + (x * y) / 7) & 0xFF) as u8;
        }
    }
    let mut lut = [0u8; 256];
    for i in 0..256 {
        lut[i] = i as u8; // identity
    }
    let fast = gray_view(&px, w, h);
    let slow = View {
        lut: Some(&lut),
        ..fast
    };
    let mut g = BridgeGrid::new();
    // equator, high latitude, straight over both poles, and on the seam
    let centres = [
        normalize(&Vec3::new(1.0, 0.0, 0.0)),
        normalize(&Vec3::new(0.3, 0.6, -0.7)),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        normalize(&Vec3::new(-1.0, 0.05, 0.0)),
        normalize(&Vec3::new(-1.0, 0.9, 0.02)),
    ];
    for c in &centres {
        let (mut pa, mut pb) = (new_patch(), new_patch());
        let (o, m) = (K_BRIDGE_OVERSAMPLE, K_BRIDGE_MAX_PATCH);
        patch_render(&fast, c, 9.0 * DEG, &mut g, &mut pa, o, m);
        patch_render(&slow, c, 9.0 * DEG, &mut g, &mut pb, o, m);
        assert_eq!(pa.s, pb.s);
        let n = (pa.s * pa.s) as usize;
        let mut worst = 0i32;
        for i in 0..n {
            worst = worst.max((pa.img[i] as i32 - pb.img[i] as i32).abs());
        }
        assert!(worst <= 1); // float32 map vs the double one: at most 1 DN
    }
}

fn big_view_marker_id(
    lon: f64,
    lat: f64,
    half: f64,
    roll: f64,
    id: i32,
    bg: u8,
) -> (Vec<u8>, i32, i32) {
    let (w, h) = (3840, 1920);
    let mut px = vec![bg; (w * h) as usize];
    render_test_marker_id(&mut px, w, h, lon, lat, half, roll, id);
    (px, w, h)
}

fn sum_centre(out: &[Vec3; 4]) -> Vec3 {
    normalize(&Vec3::new(
        out[0].x + out[1].x + out[2].x + out[3].x,
        out[0].y + out[1].y + out[2].y + out[3].y,
        out[0].z + out[1].z + out[2].z + out[3].z,
    ))
}

// The flattened-patch threshold modes must STILL rescue a marker that plain Otsu cannot
// separate (the illumination-gradient case) even though patch_flatten is lazy and runs only
// after raw Otsu has failed. This renders a real DICT_4X4_50 pattern, not a featureless dark
// square: the scan plan's screen pass applies the border-ring probe, whose documented
// behaviour on a UNIFORM cell grid is contrast == 0 ("a featureless dark rectangle is
// exactly the clutter this screen exists for").
#[test]
fn run_still_recovers_a_marker_under_a_strong_gradient() {
    let (mut px, w, h) = big_view_marker_id(30.0, -12.0, 5.0, 0.0, 0, 200);
    // 70 % peak-to-peak left-right illumination ramp across the whole frame
    for y in 0..h as usize {
        for x in 0..w as usize {
            let k = 0.45 + 0.70 * x as f64 / (w - 1) as f64;
            let i = y * w as usize + x;
            px[i] = (px[i] as f64 * k).min(255.0) as u8;
        }
    }
    let v = gray_view(&px, w as usize, h as usize);
    let mut b = Bridge::new(w, h, 0).unwrap();
    let true_c = normalize(&Vec3::new(
        (-12.0 * DEG).cos() * (30.0 * DEG).cos(),
        (-12.0 * DEG).sin(),
        (-12.0 * DEG).cos() * (30.0 * DEG).sin(),
    ));
    let mut out = [Vec3::default(); 4];
    assert!(b.run_scan(&v, true_c, 7.07 * DEG, &mut out).unwrap());
    let cen = sum_centre(&out);
    assert!(dot(&cen, &true_c).min(1.0).acos() * (1920.0 / PI) < 1.0);
}

// ---- the screen pass can reject, but must never perturb -------------------------------
//
// run() leaves `vc` and `r` at the CALLER's values across a screen pass, so the two real
// passes render exactly the framing they would have rendered without one. If that ever
// stops holding, the screen becomes another pass-1 rate change.
#[test]
fn the_screen_pass_never_moves_a_quad_it_passes() {
    let (px, w, h) = big_view_marker_id(-73.0, 41.0, 4.0, 21.0, 7, 200);
    let v = gray_view(&px, w as usize, h as usize);
    let mut b = Bridge::new(w, h, 0).unwrap();
    let (c, _, _) = frame_at(-73.0, 41.0, 21.0);
    let unscr = unscreened();
    for rmul in [1.0, 1.25, 1.6] {
        let mut with = [Vec3::default(); 4];
        let mut without = [Vec3::default(); 4];
        let r = 4.0 * 1.5 * rmul * DEG;
        assert!(b.run_scan(&v, c, r, &mut with).unwrap()); // kScanPlan
        assert!(b.run(&v, c, r, &mut without, &unscr).unwrap());
        for i in 0..4 {
            // Bit-identical, not merely close: the screen shares scratch with the real
            // passes but contributes nothing to their inputs.
            assert_eq!(with[i], without[i]);
        }
    }
}

// A candidate small enough that pass 1's own patch is no bigger than the screen's gets no
// screen at all: it would be the identical render and the identical quad, so it is pure
// duplicated work. Below the crossover the screen therefore cannot reject anything either,
// which this pins from both sides: the featureless square the screen rejects at a large
// radius survives at a small one, because no screen ran.
#[test]
fn no_screen_runs_when_pass_1_is_no_bigger_than_the_screen() {
    let (w, h) = (3840i32, 1920i32);
    let mut px = vec![200u8; (w * h) as usize];
    render_test_marker(&mut px, w, h, 30.0, -12.0, 1.0, 0.0); // featureless, small
    let v = gray_view(&px, w as usize, h as usize);
    let mut b = Bridge::new(w, h, 0).unwrap();
    let (c, _, _) = frame_at(30.0, -12.0, 0.0);
    let r = 1.0 * 1.5 * DEG;

    // Pass 1's patch at this radius is under the screen's, so the screen is skipped -- and
    // the featureless square (which the screen rejects when it does run, below) comes
    // through.
    let hf1 = (K_BRIDGE_BOX * r.tan()).atan();
    assert!(patch_side(hf1, h, K_BRIDGE_OVERSAMPLE, K_BRIDGE_SCAN_CAP1) <= K_BRIDGE_PROBE_PATCH);
    let mut out = [Vec3::default(); 4];
    assert!(b.run_scan(&v, c, r, &mut out).unwrap());
    let unscr = unscreened();
    let mut reference = [Vec3::default(); 4];
    assert!(b.run(&v, c, r, &mut reference, &unscr).unwrap());
    for i in 0..4 {
        assert_eq!(out[i], reference[i]);
    }
}

// The screen's own job: a featureless dark rectangle has no ArUco border ring (the probe
// reads contrast 0 on it), and it is what costs the scan the most -- most of a real
// frame's bridged candidates are clutter of this kind. The UNSCREENED plan must still find
// its geometry, so this also pins that the rejection is the screen's and not the passes'.
#[test]
fn the_screen_pass_rejects_a_featureless_dark_square() {
    let (w, h) = (3840i32, 1920i32);
    let mut px = vec![200u8; (w * h) as usize];
    render_test_marker(&mut px, w, h, 30.0, -12.0, 5.0, 0.0);
    let v = gray_view(&px, w as usize, h as usize);
    let mut b = Bridge::new(w, h, 0).unwrap();
    let (c, _, _) = frame_at(30.0, -12.0, 0.0);
    let r = 5.0 * 1.5 * DEG;

    let mut out = [Vec3::default(); 4];
    assert!(!b.run_scan(&v, c, r, &mut out).unwrap()); // screened out
    let unscr = unscreened();
    assert!(b.run(&v, c, r, &mut out, &unscr).unwrap()); // geometry still there
}

fn draw_marker_bw(bw: &mut [u8], s: usize) {
    // dark ring 40..160 with a bright interior 70..130 -> outer contour + hole
    bw.fill(0);
    for y in 40..160 {
        for x in 40..160 {
            bw[y * s + x] = 255;
        }
    }
    for y in 70..130 {
        for x in 70..130 {
            bw[y * s + x] = 0;
        }
    }
}

fn trace(bw: &[u8], s: i32, minlen: i32) -> (PatchContours, i32, Scratch) {
    let mut c = PatchContours::default();
    let mut sc = Scratch::default();
    let n = trace_contours(bw, s, &mut c, minlen, &mut sc);
    (c, n, sc)
}

fn quad_centre(q: &[[f64; 2]; 4]) -> (f64, f64) {
    let (mut cx, mut cy) = (0.0, 0.0);
    for p in q {
        cx += p[0] * 0.25;
        cy += p[1] * 0.25;
    }
    (cx, cy)
}

#[test]
fn contours_report_nesting_depth() {
    let s = 200usize;
    let mut bw = vec![0u8; s * s];
    draw_marker_bw(&mut bw, s);
    let (c, n, _) = trace(&bw, s as i32, 12);
    assert!(n >= 2);
    let has_even = c.depth.iter().any(|d| d % 2 == 0);
    let has_odd = c.depth.iter().any(|d| d % 2 != 0);
    assert!(has_even);
    assert!(has_odd); // the ring and its hole
}

#[test]
fn selects_the_centred_marker_contour() {
    let s = 200usize;
    let mut bw = vec![0u8; s * s];
    draw_marker_bw(&mut bw, s);
    for y in 4..26 {
        for x in 4..26 {
            bw[y * s + x] = 255; // decoy in a corner
        }
    }
    let (c, _, mut sc) = trace(&bw, s as i32, 12);
    let idx = select_marker_contour(&c, s as i32, 0.80, &mut sc);
    assert!(idx >= 0);
    let mut q = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c, idx, &mut q, &mut sc));
    order_quad(&mut q);
    let (cx, cy) = quad_centre(&q);
    assert!(near(cx, 99.5, 3.0) && near(cy, 99.5, 3.0)); // the CENTRED one
}

#[test]
fn quad_corners_are_ordered_and_square() {
    let s = 200usize;
    let mut bw = vec![0u8; s * s];
    draw_marker_bw(&mut bw, s);
    let (c, _, mut sc) = trace(&bw, s as i32, 12);
    let idx = select_marker_contour(&c, s as i32, 0.80, &mut sc);
    let mut q = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c, idx, &mut q, &mut sc));
    order_quad(&mut q);
    assert!(q[0][0] < 60.0 && q[0][1] < 60.0); // starts top-left
    for i in 0..4 {
        let dx = q[(i + 1) % 4][0] - q[i][0];
        let dy = q[(i + 1) % 4][1] - q[i][1];
        assert!(near((dx * dx + dy * dy).sqrt(), 119.0, 4.0)); // ~120 px sides
    }
}

#[test]
fn outward_offset_removes_the_dilation_bias() {
    // Ideal 120.0 px square with corners at (40,40)..(159,159): contour pixel centres are at
    // 40..159, i.e. the true 50 % crossing is at 39.5 and 159.5.
    let s = 200usize;
    let mut bw = vec![0u8; s * s];
    draw_marker_bw(&mut bw, s);
    let (c, _, mut sc) = trace(&bw, s as i32, 12);
    let idx = select_marker_contour(&c, s as i32, 0.80, &mut sc);
    let mut q = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c, idx, &mut q, &mut sc));
    order_quad(&mut q);
    assert!(polish_quad(
        &c,
        idx,
        &mut q,
        None,
        &PolishBand::default(),
        &mut sc
    ));
    order_quad(&mut q);
    // half-diagonal must be that of a 120 px square, not a 119 px one
    let (cx, cy) = quad_centre(&q);
    assert!(near(cx, 99.5, 0.10) && near(cy, 99.5, 0.10));
    let hd = ((q[0][0] - cx).powi(2) + (q[0][1] - cy).powi(2)).sqrt();
    assert!(near(hd, 120.0 * 0.70710678, 0.15)); // 84.85 px, not 84.4
}

#[test]
fn a_degenerate_edge_gets_a_measured_offset_not_the_half_pixel_guess() {
    // K_EDGE_OFFSET_PX is the MEAN inward bias of a contour over sub-pixel phases; for one
    // axis-aligned edge the real distance is whatever that edge's own phase is. Here the
    // true 50 % crossings sit at 40.20 and 159.80, so the outermost dark pixel centres are
    // 41 and 159 and the honest offsets are 0.80 and 0.80 -- the blind 0.5 leaves the
    // polished square 0.30 px too small PER SIDE, the residual the capped patch turns into
    // 1.07 px_eq of corner error on an equatorial half_deg=12 pose. With the gray plane in
    // hand the offset is measured instead.
    let s = 200usize;
    let (e0, e1) = (40.20f64, 159.80f64); // true edges, in patch px
    let (dark, bright, thr) = (40.0f64, 220.0f64, 130.0f64);
    let mut img = vec![0u8; s * s];
    let mut bw = vec![0u8; s * s];
    // Exact area coverage of the axis-aligned square, so the anti-aliasing carries the
    // sub-pixel phase the measurement is supposed to read.
    let cover1d = |i: usize| {
        let a = (i as f64 - 0.5).max(e0);
        let b = (i as f64 + 0.5).min(e1);
        if b > a {
            b - a
        } else {
            0.0
        }
    };
    for y in 0..s {
        for x in 0..s {
            let f = cover1d(x) * cover1d(y);
            let v = bright + f * (dark - bright);
            img[y * s + x] = (v + 0.5) as u8;
            bw[y * s + x] = if v < thr { 255 } else { 0 };
        }
    }
    let (c, _, mut sc) = trace(&bw, s as i32, 12);
    let idx = select_marker_contour(&c, s as i32, 0.80, &mut sc);
    assert!(idx >= 0);
    let mut q0 = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c, idx, &mut q0, &mut sc));
    order_quad(&mut q0);

    let mut blind = q0;
    assert!(polish_quad(
        &c,
        idx,
        &mut blind,
        None,
        &PolishBand::default(),
        &mut sc
    )); // no profile: the old guess
    order_quad(&mut blind);
    assert!(near(blind[0][0], 40.50, 0.05)); // contour 41, minus 0.5
    assert!(near(blind[0][1], 40.50, 0.05));

    let prof = EdgeProfile {
        gray: Some(&img),
        flat: None,
        s: s as i32,
        thr,
    };
    let mut meas = q0;
    assert!(polish_quad(
        &c,
        idx,
        &mut meas,
        Some(&prof),
        &PolishBand::default(),
        &mut sc
    ));
    order_quad(&mut meas);
    for i in 0..4 {
        // every corner on the truth
        assert!(near(
            meas[i][0],
            if i == 0 || i == 3 { e0 } else { e1 },
            0.05
        ));
        assert!(near(
            meas[i][1],
            if i == 0 || i == 1 { e0 } else { e1 },
            0.05
        ));
    }
}

#[test]
fn an_oblique_edge_keeps_the_half_pixel_offset_untouched() {
    // The other half of the rule: where the staircase DOES sample the phases,
    // K_EDGE_OFFSET_PX is already unbiased, so an oblique edge must come out bit-identical
    // with and without a profile -- that is what leaves the real-frame agreement with the
    // reference on the reference's own quad.
    let s = 200usize;
    let (dark, bright, thr) = (40.0f64, 220.0f64, 130.0f64);
    let (cx, cy, half, rot) = (99.5f64, 99.5f64, 60.0f64, 11.0 * DEG);
    let (cr, sr) = (rot.cos(), rot.sin());
    let mut img = vec![0u8; s * s];
    let mut bw = vec![0u8; s * s];
    for y in 0..s {
        for x in 0..s {
            let mut hit = 0;
            for sy in 0..4 {
                for sx in 0..4 {
                    // 4x4 supersampled coverage
                    let px = x as f64 + (sx as f64 + 0.5) / 4.0 - 0.5 - cx;
                    let py = y as f64 + (sy as f64 + 0.5) / 4.0 - 0.5 - cy;
                    let u = px * cr + py * sr;
                    let w = -px * sr + py * cr;
                    if u.abs() <= half && w.abs() <= half {
                        hit += 1;
                    }
                }
            }
            let v = bright + (hit as f64 / 16.0) * (dark - bright);
            img[y * s + x] = (v + 0.5) as u8;
            bw[y * s + x] = if v < thr { 255 } else { 0 };
        }
    }
    let (c, _, mut sc) = trace(&bw, s as i32, 12);
    let idx = select_marker_contour(&c, s as i32, 0.80, &mut sc);
    assert!(idx >= 0);
    let mut q0 = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c, idx, &mut q0, &mut sc));
    order_quad(&mut q0);
    let (mut blind, mut meas) = (q0, q0);
    let prof = EdgeProfile {
        gray: Some(&img),
        flat: None,
        s: s as i32,
        thr,
    };
    assert!(polish_quad(
        &c,
        idx,
        &mut blind,
        None,
        &PolishBand::default(),
        &mut sc
    ));
    assert!(polish_quad(
        &c,
        idx,
        &mut meas,
        Some(&prof),
        &PolishBand::default(),
        &mut sc
    ));
    for i in 0..4 {
        assert_eq!(meas[i][0], blind[i][0]);
        assert_eq!(meas[i][1], blind[i][1]);
    }
}

#[test]
fn contours_resolve_four_levels_of_nesting() {
    // 240x240 patch, four levels of nesting from a single geometric parent/depth resolution
    // pass: outer 200x200 filled square (depth 0) with a 160x160 hole (depth 1); inside the
    // hole a 60x60 filled module (depth 2) with its own 20x20 hole (depth 3), plus a second,
    // separate 30x30 filled module (depth 2, no hole) sharing the hole as parent.
    let s = 240usize;
    let mut bw = vec![0u8; s * s];
    let fill = |bw: &mut Vec<u8>, x0: usize, y0: usize, x1: usize, y1: usize, val: u8| {
        for y in y0..=y1 {
            for x in x0..=x1 {
                bw[y * s + x] = val;
            }
        }
    };
    fill(&mut bw, 20, 20, 219, 219, 255); // depth 0: outer 200x200 square
    fill(&mut bw, 40, 40, 199, 199, 0); // depth 1: 160x160 hole
    fill(&mut bw, 70, 70, 129, 129, 255); // depth 2: 60x60 module
    fill(&mut bw, 90, 90, 109, 109, 0); // depth 3: 20x20 hole in that module
    fill(&mut bw, 150, 150, 179, 179, 255); // depth 2: second 30x30 module, no hole

    let (c, n, mut sc) = trace(&bw, s as i32, 12);
    assert!(n >= 5);

    let mut hist = [0i32; 8];
    for &d in &c.depth {
        if (0..8).contains(&d) {
            hist[d as usize] += 1;
        }
    }
    assert_eq!(hist[0], 1);
    assert_eq!(hist[1], 1);
    assert_eq!(hist[2], 2);
    assert_eq!(hist[3], 1);

    let idx = select_marker_contour(&c, s as i32, 0.80, &mut sc);
    assert!(idx >= 0);
    let mut q = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c, idx, &mut q, &mut sc));
    order_quad(&mut q);
    let (cx, cy) = quad_centre(&q);
    assert!(near(cx, 119.5, 2.0) && near(cy, 119.5, 2.0)); // the OUTER square, not an inner module
    let mut side_sum = 0.0;
    for i in 0..4 {
        let dx = q[(i + 1) % 4][0] - q[i][0];
        let dy = q[(i + 1) % 4][1] - q[i][1];
        let side = (dx * dx + dy * dy).sqrt();
        assert!(near(side, 200.0, 4.0));
        side_sum += side;
    }
    let area = (side_sum * 0.25) * (side_sum * 0.25);
    assert!(near(area, 200.0 * 200.0, 0.03 * 200.0 * 200.0));

    // A small decoy filled square in the patch corner -- its own, unrelated depth-0 contour
    // -- must not change the selection.
    fill(&mut bw, 5, 5, 14, 14, 255);
    let (c2, _, mut sc2) = trace(&bw, s as i32, 12);
    let idx2 = select_marker_contour(&c2, s as i32, 0.80, &mut sc2);
    assert!(idx2 >= 0);
    let mut q2 = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c2, idx2, &mut q2, &mut sc2));
    order_quad(&mut q2);
    let (cx2, cy2) = quad_centre(&q2);
    assert!(near(cx2, 119.5, 2.0) && near(cy2, 119.5, 2.0));
}

// ---- regression: a dark region touching the patch border must not flip hole parity ------

#[test]
fn dark_frame_at_patch_border_keeps_marker_selectable() {
    // A large near marker on a white board in front of a darker background: the bridge's
    // pass-2 patch sees the background as a dark frame along the patch border (thick at the
    // top, 2 px elsewhere) around the white board, with the marker in the middle. The frame's
    // own outer border lies on the patch edge and is never traced, only its inner border (the
    // board outline). Containment parity then makes that board outline a depth-0 "dark
    // region" and the marker a depth-1 "hole", and the selection used to return the board
    // (or the marker's white interior) instead of the marker.
    let s = 240usize;
    let mut bw = vec![0u8; s * s];
    for y in 0..s {
        for x in 0..s {
            let frame = y < 25 || x < 2 || x >= s - 2 || y >= s - 2;
            let marker = (70..170).contains(&x) && (70..170).contains(&y);
            if frame || marker {
                bw[y * s + x] = 255;
            }
        }
    }
    for y in 95..145 {
        for x in 95..145 {
            bw[y * s + x] = 0; // the marker's white data modules
        }
    }
    let (c, _, mut sc) = trace(&bw, s as i32, 12);
    let idx = select_marker_contour(&c, s as i32, 0.80, &mut sc);
    assert!(idx >= 0);
    let mut q = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c, idx, &mut q, &mut sc));
    order_quad(&mut q);
    let (cx, cy) = quad_centre(&q);
    assert!(near(cx, 119.5, 2.0) && near(cy, 119.5, 2.0));
    for i in 0..4 {
        let dx = q[(i + 1) % 4][0] - q[i][0];
        let dy = q[(i + 1) % 4][1] - q[i][1];
        assert!(near((dx * dx + dy * dy).sqrt(), 100.0, 4.0)); // the marker, not board / modules
    }
    let mut out = [0i32; 3];
    let n = select_marker_contours_relaxed(&c, s as i32, &mut out, 3);
    assert!(n >= 1);
    assert_eq!(out[0], idx);
}

// ---- regression: the DP split point must not become a quad vertex --------------------
//
// Reproduces a real-clip frame's failure exactly. The traced contour of a marker whose top
// edge is shallow starts at the LEFT end of the topmost boundary run, several px along that
// edge from the true corner. Splitting the closed contour at pts[0] and the point farthest
// from it forces pts[0] into the output as a vertex, so one returned "corner" sat mid-edge,
// the corresponding quad side ran ~10 px inside the marker's true edge, and polish_quad's
// band then selected ZERO contour points for that side and failed outright.
// cv2.approxPolyDP splits a closed curve at its two EXTREME points instead (farthest from
// pts[0], then farthest from that), which for a quadrilateral blob are both near-corners.

/// Appends the 8-connected integer boundary from a (exclusive of b) to `out`.
fn walk_edge(out: &mut Vec<i32>, ax: f64, ay: f64, bx: f64, by: f64) {
    let dx = bx - ax;
    let dy = by - ay;
    let n = dx.abs().max(dy.abs()).round() as i32;
    for i in 0..n {
        let t = i as f64 / n as f64;
        out.push((ax + dx * t).round() as i32);
        out.push((ay + dy * t).round() as i32);
    }
}

#[test]
fn quad_dp_does_not_pin_the_contour_start_as_a_corner() {
    // that frame's own quad, in its own 285 px patch coordinates.
    let a = [97.0, 102.0];
    let b = [189.0, 74.0];
    let c2 = [180.0, 203.0];
    let d = [93.0, 210.0];
    // Start the traced point list 6 px BEFORE corner B, exactly as the raster scan does on
    // the real frame (its pts[0] is (183, 73), 6 px along the A->B edge from B at (189, 74)).
    let s0 = [183.0, 72.5];

    let mut c = PatchContours::default();
    walk_edge(&mut c.pts, s0[0], s0[1], b[0], b[1]);
    walk_edge(&mut c.pts, b[0], b[1], c2[0], c2[1]);
    walk_edge(&mut c.pts, c2[0], c2[1], d[0], d[1]);
    walk_edge(&mut c.pts, d[0], d[1], a[0], a[1]);
    walk_edge(&mut c.pts, a[0], a[1], s0[0], s0[1]);
    c.start.push(0);
    c.start.push((c.pts.len() / 2) as u32);
    c.parent = vec![-1];
    c.depth = vec![0];
    assert_eq!(c.pts[0], 183); // the split point really is mid-edge

    let mut sc = Scratch::default();
    let mut q = [[0.0; 2]; 4];
    assert!(quad_from_contour(&c, 0, &mut q, &mut sc));
    order_quad(&mut q);
    let truth = [a, b, c2, d];
    for i in 0..4 {
        let mut best = 1e300f64;
        for k in 0..4 {
            let dx = q[i][0] - truth[k][0];
            let dy = q[i][1] - truth[k][1];
            best = best.min((dx * dx + dy * dy).sqrt());
        }
        assert!(best < 2.0); // every vertex is a TRUE corner
    }
    // and with real corners the polish now has points on every side
    assert!(polish_quad(
        &c,
        0,
        &mut q,
        None,
        &PolishBand::default(),
        &mut sc
    ));
}

#[test]
fn run_recovers_a_badly_centred_candidate() {
    // 3840x1920 with an exact gnomonic marker; seed the bridge with a candidate offset by
    // 40 % of the radius and 1.8x too wide -- the "worst candidate" case measured at 31.3
    // px_eq mean. A real DICT_4X4_50 pattern, centred at lon 30, lat -12.
    let (px, w, h) = big_view_marker_id(30.0, -12.0, 5.0, 0.0, 0, 200);
    let v = gray_view(&px, w as usize, h as usize);
    let mut b = Bridge::new(w, h, 0).unwrap();
    let true_c = normalize(&Vec3::new(
        (-12.0 * DEG).cos() * (30.0 * DEG).cos(),
        (-12.0 * DEG).sin(),
        (-12.0 * DEG).cos() * (30.0 * DEG).sin(),
    ));
    let seed = normalize(&Vec3::new(true_c.x + 0.03, true_c.y - 0.02, true_c.z));
    let mut out = [Vec3::default(); 4];
    assert!(b.run_scan(&v, seed, 5.0 * 1.8 * DEG, &mut out).unwrap());
    let cen = sum_centre(&out);
    assert!(dot(&cen, &true_c).acos() * (1920.0 / PI) < 1.0); // within 1 px_eq
}

// ---- 4-line fit of the marker quad (bitten corner, notch, merge) -----------------------

struct QuadRaster {
    s: usize,
    bw: Vec<u8>,
}

impl QuadRaster {
    fn new() -> QuadRaster {
        QuadRaster {
            s: 220,
            bw: vec![0; 220 * 220],
        }
    }
    /// [x0,x1) x [y0,y1)
    fn rect(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, v: u8) {
        for y in y0..y1 {
            for x in x0..x1 {
                self.bw[y * self.s + x] = v;
            }
        }
    }
    /// the ring marker of draw_marker_bw: dark ring 40..160, bright interior 70..130
    fn marker(&mut self) {
        self.rect(40, 40, 160, 160, 255);
        self.rect(70, 70, 130, 130, 0);
    }
}

/// Largest traced contour (by point count).
fn biggest_contour(c: &PatchContours) -> i32 {
    let mut best = -1;
    let mut bl = 0u32;
    for i in 0..c.start.len() - 1 {
        if c.start[i + 1] - c.start[i] > bl {
            bl = c.start[i + 1] - c.start[i];
            best = i as i32;
        }
    }
    best
}

/// quad_from_lines -> order -> polish; checks each corner against the truth (any order).
fn lines_quad_matches(r: &QuadRaster, truth: &[[f64; 2]; 4], tol: f64) -> bool {
    let (c, _, mut sc) = trace(&r.bw, r.s as i32, 12);
    let idx = biggest_contour(&c);
    if idx < 0 {
        return false;
    }
    let mut q = [[0.0; 2]; 4];
    if !quad_from_lines(&c, idx, &mut q, &mut sc) {
        return false;
    }
    order_quad(&mut q);
    if !polish_quad(&c, idx, &mut q, None, &K_LINE_FIT_BAND, &mut sc) {
        return false;
    }
    order_quad(&mut q);
    for i in 0..4 {
        let mut best = 1e300f64;
        for k in 0..4 {
            best = best.min((q[i][0] - truth[k][0]).hypot(q[i][1] - truth[k][1]));
        }
        if best > tol {
            return false;
        }
    }
    true
}

const K_TRUTH4: [[f64; 2]; 4] = [[39.5, 39.5], [159.5, 39.5], [159.5, 159.5], [39.5, 159.5]];

#[test]
fn lines_quad_on_a_clean_marker_matches_the_truth() {
    let mut r = QuadRaster::new();
    r.marker();
    assert!(lines_quad_matches(&r, &K_TRUTH4, 0.5));
}

#[test]
fn lines_quad_survives_a_bitten_corner() {
    // a bright occluder removes the bottom-right corner; the right and bottom sides are cut
    // short, the corner itself is gone from the blob
    for bite in [20usize, 35] {
        let mut r = QuadRaster::new();
        r.marker();
        r.rect(160 - bite, 160 - bite, 160, 160, 0);
        assert!(lines_quad_matches(&r, &K_TRUTH4, 0.5));
    }
    // a bigger bite that also opens the ring to the interior (the traced outer contour then
    // follows the ring's inner edge): the 4 outer sides still win
    let mut r = QuadRaster::new();
    r.marker();
    r.rect(115, 115, 160, 160, 0);
    assert!(lines_quad_matches(&r, &K_TRUTH4, 0.5));
}

#[test]
fn lines_quad_survives_bites_on_two_corners_and_a_bright_notch() {
    let mut r = QuadRaster::new();
    r.marker();
    r.rect(40, 40, 62, 62, 0); // top-left
    r.rect(138, 138, 160, 160, 0); // bottom-right
    r.rect(92, 40, 104, 52, 0); // a notch in the top side
    assert!(lines_quad_matches(&r, &K_TRUTH4, 0.5));
}

#[test]
fn lines_quad_gives_the_true_quad_when_a_dark_rectangle_merges_in() {
    let mut r = QuadRaster::new();
    r.marker();
    r.rect(160, 62, 195, 112, 255); // dark occluder on the right side
    assert!(lines_quad_matches(&r, &K_TRUTH4, 0.5));
    let mut r2 = QuadRaster::new();
    r2.marker();
    r2.rect(10, 125, 40, 175, 255); // and one that also overhangs the bottom
    assert!(lines_quad_matches(&r2, &K_TRUTH4, 0.5));
}

#[test]
fn lines_quad_never_invents_a_side() {
    // a triangle (3 sides) and a disc (no straight side) give no quad
    let mut tri = QuadRaster::new();
    for y in 40..170usize {
        for x in 40..40 + (y - 40) {
            tri.bw[y * tri.s + x] = 255;
        }
    }
    let (c, _, mut sc) = trace(&tri.bw, tri.s as i32, 12);
    let mut q = [[0.0; 2]; 4];
    assert!(!quad_from_lines(&c, biggest_contour(&c), &mut q, &mut sc));
    let mut disc = QuadRaster::new();
    for y in 0..disc.s {
        for x in 0..disc.s {
            let (dx, dy) = (x as i32 - 110, y as i32 - 110);
            if dx * dx + dy * dy < 60 * 60 {
                disc.bw[y * disc.s + x] = 255;
            }
        }
    }
    let (c2, _, mut sc2) = trace(&disc.bw, disc.s as i32, 12);
    assert!(!quad_from_lines(
        &c2,
        biggest_contour(&c2),
        &mut q,
        &mut sc2
    ));
    // a marker whose right and bottom sides are BOTH swallowed by a dark blob has only two
    // measured sides: no quad (nothing is extrapolated)
    let mut two = QuadRaster::new();
    two.rect(40, 40, 160, 160, 255);
    two.rect(110, 60, 200, 210, 255);
    two.rect(60, 110, 200, 210, 255);
    let (c3, _, mut sc3) = trace(&two.bw, two.s as i32, 12);
    let i3 = biggest_contour(&c3);
    // Rejecting this case outright cannot be done, at contour level or by the ring gate: the
    // blob IS a box (area fill ~0.95), its far corners are as far from the observed runs as
    // a legitimately bitten corner's, and a dark occluder makes the ring read dark. The
    // strict decode decides. What the fit does guarantee, and what is asserted: nothing
    // outside the observed blob's own bounding box is invented, and the marker's own hidden
    // corner (159.5,159.5) is not reported.
    if quad_from_lines(&c3, i3, &mut q, &mut sc3) {
        for i in 0..4 {
            assert!(q[i][0] >= 39.0 && q[i][0] <= 200.0 && q[i][1] >= 39.0 && q[i][1] <= 210.0);
            assert!((q[i][0] - 159.5).hypot(q[i][1] - 159.5) > 10.0);
        }
    }
}

#[test]
fn relaxed_selection_keeps_a_low_solidity_contour_the_strict_one_rejects() {
    // a marker with a big dark rectangle merged on: solidity < 0.80 so the strict selection
    // finds nothing; the relaxed one still offers the contour
    let mut r = QuadRaster::new();
    r.marker();
    r.rect(160, 120, 218, 218, 255);
    let (c, _, mut sc) = trace(&r.bw, r.s as i32, 12);
    assert!(select_marker_contour(&c, r.s as i32, 0.80, &mut sc) < 0);
    let mut out = [0i32; 3];
    let n = select_marker_contours_relaxed(&c, r.s as i32, &mut out, 3);
    assert!(n >= 1);
    assert_eq!(out[0], biggest_contour(&c));
}

#[test]
fn a_failed_later_pass_keeps_the_earlier_passes_quad_and_alternate() {
    // pass 2 renders a patch far smaller than the marker (box 0.05): it finds no quad, so
    // run() must return pass 1's quad untouched (the gate and the fallbacks live inside
    // bridge_find_quad and never overwrite `out` on failure)
    let (px, w, h) = big_view_marker_id(30.0, -12.0, 5.0, 0.0, 0, 200);
    let v = gray_view(&px, w as usize, h as usize);
    let mut b = Bridge::new(w, h, 0).unwrap();
    let c = test_marker_center_ray(30.0, -12.0, 0.0);
    let plan = [
        K_SCAN_PLAN[1],
        BridgePass {
            box_: 0.05,
            oversample: 1.6,
            max_side: 768,
            screen: false,
        },
    ];
    let mut out = [Vec3::default(); 4];
    assert!(b.run(&v, c, 5.0 * 1.5 * DEG, &mut out, &plan).unwrap());
    let truth = test_marker_corner_rays(30.0, -12.0, 5.0, 0.0);
    let mut worst = 0.0f64;
    for i in 0..4 {
        let mut best = 1e9f64;
        for k in 0..4 {
            best = best.min(dot(&out[i], &truth[k]).min(1.0).acos() * h as f64 / PI);
        }
        worst = worst.max(best);
    }
    assert!(worst < 3.0); // px_eq: pass 1's quad
}

// ---- not in the original suite: input checks of the Rust API ---------------------------

#[test]
fn run_rejects_a_view_of_another_size() {
    let px = vec![128u8; 960 * 480];
    let v = gray_view(&px, 960, 480);
    let mut b = Bridge::new(3840, 1920, 0).unwrap();
    let mut out = [Vec3::default(); 4];
    assert!(b
        .run_scan(&v, Vec3::new(1.0, 0.0, 0.0), 0.1, &mut out)
        .is_err());
    assert!(Bridge::new(3840, 1920, 99).is_err());
}
