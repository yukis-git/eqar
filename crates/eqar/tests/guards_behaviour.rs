//! Behaviour tests for the guards stage (numeric expectations are those of the original
//! C++ unit tests, unchanged).
#![allow(clippy::needless_range_loop)]

use eqar::codes;
use eqar::decode::{decode_rays, homography_closed, plane_to_ray, DecodeResult, K_SUB_DEFAULT};
use eqar::geometry::{basis_at, cross, dot, norm, normalize, slerp, Vec3, K_PI};
use eqar::guards::*;
use eqar::image::{Fmt, View};
use eqar::refine::{RefineInfo, K_END_SUPPORT_COLD};
use eqar_testkit::synth::{render_test_marker_id, test_marker_corner_rays};

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn good_case() -> ([Vec3; 4], [Vec3; 4], RefineInfo, DecodeResult) {
    let init = test_marker_corner_rays(21.0, -33.0, 6.0, 12.0);
    let fin = init;
    let info = RefineInfo {
        ok: true,
        side_px: 128.0,
        hw0: 6.0,
        contrast_dn: 150.0,
        resid_px: 0.10,
        min_cross: 0.99,
        ..RefineInfo::default()
    };
    let dec = DecodeResult {
        id: 0,
        rot: 0,
        n_bit_errors: 0,
        n_erased: 0,
        border_fail: 0,
        border_ok: true,
        contrast: 0.0,
        bits: 0,
        corners: init,
        reject_code: codes::ACCEPT,
    };
    (init, fin, info, dec)
}

fn gates(
    a: &[Vec3; 4],
    b: &[Vec3; 4],
    info: &RefineInfo,
    dec: &DecodeResult,
    reject_high_resid: bool,
) -> GateOutcome {
    apply_gates(&GateInput {
        init_rays: a,
        final_rays: b,
        info,
        decode: dec,
        reject_high_resid,
    })
}

#[test]
fn accepts_a_healthy_detection() {
    let (a, b, info, dec) = good_case();
    let o = gates(&a, &b, &info, &dec, false);
    assert_eq!(o.code, codes::ACCEPT);
    assert!(o.quality > 0.9);
    assert_eq!(o.flags & codes::FLAG_HIGH_RESID, 0);
}

#[test]
fn rejects_on_init_geometry_not_fitted_normals() {
    // The FITTED normals look perfect (min_cross 0.99) while the seed quad is collapsed. The
    // gate must still fire.
    //
    // Test-literal correction of the original: squashing corners 1 and 2 toward the quad's
    // centroid does not collapse init_min_cross below 0.30 at ANY squash fraction (it plateaus
    // around 0.71-0.72, because the connecting great circle converges to a well-defined,
    // stable limiting direction). What DOES reliably collapse it is making three corners
    // collinear on a great circle: placing corner 1 exactly on the great circle spanned by
    // corners 0 and 2 forces edges (0,1) and (1,2) to share that SAME great circle, so their
    // normals become exactly (anti)parallel (min_cross ~1e-15).
    let (mut a, b, info, dec) = good_case();
    a[1] = slerp(&a[0], &a[2], 0.5); // collapse corner 1 onto the great circle a[0]-a[2]
    assert!(init_min_cross(&a) <= K_INIT_MIN_CROSS);
    let o = gates(&a, &b, &info, &dec, false);
    assert_eq!(o.code, codes::R_INIT_GEOMETRY);
}

#[test]
fn rejects_a_corner_that_moved_further_than_hw0() {
    let (a, mut b, info, dec) = good_case();
    let pix = K_PI / 1920.0;
    let (e1, _) = basis_at(&b[2]);
    let d = (info.hw0 + 1.0) * pix;
    b[2] = normalize(&Vec3::new(
        b[2].x + e1.x * d,
        b[2].y + e1.y * d,
        b[2].z + e1.z * d,
    ));
    let o = gates(&a, &b, &info, &dec, false);
    assert_eq!(o.code, codes::R_DISPLACEMENT);
}

#[test]
fn low_contrast_and_border_are_rejections() {
    let (a, b, mut info, dec) = good_case();
    info.contrast_dn = 5.0;
    assert_eq!(
        gates(&a, &b, &info, &dec, false).code,
        codes::R_LOW_CONTRAST
    );
    let (a, b, info, mut dec) = good_case();
    dec.border_ok = false;
    dec.reject_code = codes::R_BORDER_NOT_BLACK;
    assert_eq!(
        gates(&a, &b, &info, &dec, false).code,
        codes::R_BORDER_NOT_BLACK
    );
}

#[test]
fn high_resid_flags_by_default_and_rejects_when_asked() {
    let (a, b, mut info, dec) = good_case();
    info.resid_px = 0.0040 * info.side_px; // ratio 0.0040 > 0.0026
    let o = gates(&a, &b, &info, &dec, false);
    assert_eq!(o.code, codes::ACCEPT);
    assert!(o.flags & codes::FLAG_HIGH_RESID != 0);
    assert!(o.quality < 0.8);
    let o = gates(&a, &b, &info, &dec, true);
    assert_eq!(o.code, codes::R_HIGH_RESID);
}

// ---- per-corner validity, sigma scale, post-refine side consistency -------------------------

fn info_with() -> RefineInfo {
    let mut info = RefineInfo {
        side_px: 100.0,
        ..RefineInfo::default()
    };
    for e in 0..4 {
        info.edge[e].n_prof = 50;
        info.edge[e].n_inl = 45;
        info.edge[e].support = 0.9;
        info.edge[e].end_support = [1.0, 1.0];
    }
    info
}

#[test]
fn sigma_scale_is_one_for_an_observed_tip_and_grows_as_the_tip_hides() {
    assert!(near(corner_sigma_scale(1.0) as f64, 1.0, 1e-6));
    assert!(near(
        corner_sigma_scale(K_END_SUPPORT_COLD as f32) as f64,
        1.0,
        1e-6
    ));
    let mut prev = 1.0f32;
    let mut e = 0.45f32;
    while e >= 0.0 {
        let s = corner_sigma_scale(e);
        assert!(s >= prev);
        prev = s;
        e -= 0.05;
    }
    assert!(corner_sigma_scale(0.0) > 10.0);
}

#[test]
fn an_unobserved_tip_is_invalid_cold_and_scaled_when_the_id_is_known() {
    let mut info = info_with();
    info.edge[0].end_support[1] = 0.1; // edge 0 ends short of corner 1
    let cold = corner_states(&info, false);
    let known = corner_states(&info, true);
    for i in 0..4 {
        if i != 1 {
            assert!(cold[i].valid && known[i].valid);
        }
    }
    assert!(!cold[1].valid);
    assert!(known[1].valid);
    assert!(known[1].sigma_scale > 5.0);
    assert!(near(known[1].end_support as f64, 0.1, 1e-6));
    assert_eq!(cold[1].flags, 0);
}

#[test]
fn the_min_of_the_two_adjacent_tip_windows_decides() {
    let mut info = info_with();
    info.edge[2].end_support[1] = 0.45; // corner 3 (arrives edge 2, leaves edge 3)
    info.edge[3].end_support[0] = 0.2;
    let c = corner_states(&info, true);
    assert!(near(c[3].end_support as f64, 0.2, 1e-6));
    assert!(c[3].sigma_scale > c[2].sigma_scale);
}

#[test]
fn the_corners_of_a_weak_edge_are_completed_and_never_valid() {
    let mut info = info_with();
    info.edge[1].weak = true;
    info.edge[1].support = 0.1;
    let cold = corner_states(&info, false);
    let known = corner_states(&info, true);
    for cs in [&cold, &known] {
        assert!(!cs[1].valid && !cs[2].valid);
        assert!(cs[1].flags & K_CORNER_COMPLETED != 0);
        assert!(cs[2].flags & K_CORNER_COMPLETED != 0);
        assert!(cs[0].valid && cs[3].valid);
        assert_eq!(cs[0].flags, 0);
    }
}

#[test]
fn gate_5_invalidates_the_corner_instead_of_rejecting_the_marker() {
    let mut info = info_with();
    info.edge[0].resid_px = (K_RESID_RATIO * info.side_px * 1.5) as f32; // over the bound
    let c = corner_states(&info, true);
    assert!(!c[0].valid && !c[1].valid); // the two corners of edge 0
    assert!(c[0].flags & K_CORNER_RESID != 0);
    assert!(c[2].valid && c[3].valid);
    info.edge[0].resid_px = (K_RESID_RATIO * info.side_px * 0.5) as f32;
    let c = corner_states(&info, true);
    for i in 0..4 {
        assert!(c[i].valid);
    }
}

fn agrees(init: &[Vec3; 4], refined: &[Vec3; 4], e: usize, module: f64, pix: f64) -> bool {
    side_agrees(
        init,
        refined,
        e,
        module,
        pix,
        K_CONSIST_ANGLE_RAD,
        K_CONSIST_OFFSET_CELL,
    )
}

#[test]
fn a_refined_side_must_agree_with_the_bridge_line() {
    let init = test_marker_corner_rays(21.0, -33.0, 6.0, 12.0);
    let refd = init;
    let (pix, module) = (K_PI / 1920.0, 12.0);
    assert!(agrees(&init, &refd, 0, module, pix));
    // side 0 shifted by 0.3 cell (3.6 px) along a tangent: over the 0.15-cell offset bound
    let mut sh = init;
    for i in [0usize, 1] {
        let (e1, _) = basis_at(&init[i]);
        let a = 0.3 * module * pix;
        sh[i] = normalize(&Vec3::new(
            init[i].x + e1.x * a,
            init[i].y + e1.y * a,
            init[i].z + e1.z * a,
        ));
    }
    // the e1 tangent need not be perpendicular to the side; check what it really is, then assert on that
    let nb = normalize(&cross(&init[0], &init[1]));
    let mid = normalize(&Vec3::new(
        init[0].x + init[1].x,
        init[0].y + init[1].y,
        init[0].z + init[1].z,
    ));
    let off = dot(&mid, &normalize(&cross(&sh[0], &sh[1]))).abs() / pix;
    assert_eq!(agrees(&init, &sh, 0, module, pix), off < 0.15 * module);
    // a 3 degree rotation about the side's midpoint (zero offset there): over the 2 degree bound
    let mut rot = init;
    let th = 3.0 * K_PI / 180.0; // each end moves th * |d| in opposite directions: the line turns by th
    for i in [0usize, 1] {
        let d = Vec3::new(init[i].x - mid.x, init[i].y - mid.y, init[i].z - mid.z);
        let sgn = if i == 0 { 1.0 } else { -1.0 };
        let nd = norm(&d);
        rot[i] = normalize(&Vec3::new(
            init[i].x + nb.x * sgn * th * nd,
            init[i].y + nb.y * sgn * th * nd,
            init[i].z + nb.z * sgn * th * nd,
        ));
    }
    assert!(!agrees(&init, &rot, 0, module, pix));
}

#[test]
fn the_gate_5_bound_has_a_floor_for_small_markers() {
    // 0.0026 * 20 px = 0.052 px: below the floor, so a 0.12 px inlier residual must NOT
    // invalidate a small marker's corners
    let mut small = info_with();
    small.side_px = 20.0;
    for e in 0..4 {
        small.edge[e].resid_px = 0.12;
    }
    let c = corner_states(&small, false);
    for i in 0..4 {
        assert!(c[i].valid);
    }
    for e in 0..4 {
        small.edge[e].resid_px = (K_CORNER_RESID_FLOOR_PX * 1.5) as f32;
    }
    let c = corner_states(&small, false);
    for i in 0..4 {
        assert!(!c[i].valid);
        assert!(c[i].flags & K_CORNER_RESID != 0);
    }
    // a large marker: the ratio (0.0026 * 100 = 0.26 px) is above the floor
    let mut big = info_with();
    for e in 0..4 {
        big.edge[e].resid_px = 0.2;
    }
    let c = corner_states(&big, false);
    for i in 0..4 {
        assert!(c[i].valid);
    }
}

// ---- ring_width -----------------------------------------------------------------------------

#[test]
fn ring_width_reads_one_module_on_every_side_at_every_rotation_and_under_0_7_for_an_inward_shift() {
    // The original drove this through the detector (`Detector::scan`, whose reading of a clean
    // marker is `ring[s]`); this variant uses the ordered corners of a decode of the exact
    // ground-truth quad. The detector-driven original is in detector_behaviour.rs.
    let (w, h) = (3840, 1920);
    let pix = K_PI / h as f64;
    for roll in [13.0, 103.0, 193.0, 283.0] {
        let mut px = vec![205u8; (w * h) as usize];
        render_test_marker_id(&mut px, w, h, -100.0, -8.0, 5.0, roll, 3);
        let v = View {
            data: &px,
            w,
            h,
            stride: w as usize,
            fmt: Fmt::Gray8,
            lut: None,
        };
        let gt = test_marker_corner_rays(-100.0, -8.0, 5.0, roll);
        let dec = decode_rays(&v, 0, &gt, K_SUB_DEFAULT, None);
        assert_eq!(dec.id, 3);
        let ray = dec.corners;
        let hm = homography_closed(&ray).expect("homography");
        let module = 2.0 / 6.0; // 4x4 marker + 1-module border: 6 modules across the [-1, 1] plane
        let shift = 0.6 * module;
        for s in 0..4usize {
            let ring = ring_width(&v, 0, dec.id, &ray, s, pix);
            assert!(ring > 0.9 && ring < 1.1); // a clean marker
                                               // move side s (and its two corners) 0.6 module INWARD: only 0.4 module of the black border stays outside it
            let mut c = [[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]];
            for k in [s, (s + 1) & 3] {
                if s == 0 {
                    c[k][1] -= shift;
                }
                if s == 1 {
                    c[k][0] -= shift;
                }
                if s == 2 {
                    c[k][1] += shift;
                }
                if s == 3 {
                    c[k][0] += shift;
                }
            }
            let mut q = [Vec3::default(); 4];
            for i in 0..4 {
                q[i] = plane_to_ray(&hm, c[i][0], c[i][1]);
            }
            let wd = ring_width(&v, 0, dec.id, &q, s, pix);
            assert!((0.0..0.7).contains(&wd));
            assert!(wd < ring);
        }
    }
}
