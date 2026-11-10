//! Behaviour tests for the detector (numeric expectations are those of the original C++
//! unit tests, unchanged). Several tests use the default 2 ms Stage 2-5 budget exactly as
//! the originals did; they only ever need the first candidate.
#![allow(clippy::needless_range_loop, clippy::field_reassign_with_default)]

use eqar::codes::*;
use eqar::decode::{homography_closed, plane_to_ray};
use eqar::detector::{Detection, Detector, Prior};
use eqar::geometry::{cross, dot, normalize, ray_to_px, Vec3, K_PI};
use eqar::guards::{ring_width, K_CORNER_COMPLETED, K_CORNER_PRIOR, K_CORNER_SIDE};
use eqar::image::{Fmt, View};
use eqar::Params;
use eqar_testkit::synth::{
    render_test_marker, render_test_marker_id, test_marker_center_ray, test_marker_corner_rays,
};

const W: i32 = 3840;
const H: i32 = 1920;

fn defp() -> Params {
    Params::default()
}

fn view(px: &[u8]) -> View<'_> {
    View {
        data: px,
        w: W,
        h: H,
        stride: W as usize,
        fmt: Fmt::Gray8,
        lut: None,
    }
}

fn canvas() -> Vec<u8> {
    vec![205u8; (W * H) as usize]
}

fn ang_px(a: &Vec3, b: &Vec3) -> f64 {
    dot(a, b).min(1.0).acos() / (K_PI / H as f64)
}

#[test]
fn finds_one_marker_and_reports_stats() {
    let mut px = canvas();
    render_test_marker_id(&mut px, W, H, -100.0, -8.0, 5.0, 13.0, 0);
    let mut d = Detector::new(W, H, &defp()).unwrap();
    let out = d.scan(&view(&px), 8).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 0);
    let gt = test_marker_corner_rays(-100.0, -8.0, 5.0, 13.0);
    for i in 0..4 {
        assert!(ang_px(&out[0].ray[i], &gt[i]) < 0.05);
    }
    assert!(d.stats().n_candidates >= 1);
    assert_eq!(d.stats().n_accepted, 1);
}

#[test]
fn marker_center_matches_the_plane_centre() {
    let mut px = canvas();
    render_test_marker_id(&mut px, W, H, -100.0, -8.0, 5.0, 13.0, 0);
    let mut d = Detector::new(W, H, &defp()).unwrap();
    let out = d.scan(&view(&px), 8).unwrap();
    assert_eq!(out.len(), 1);
    let gt = test_marker_center_ray(-100.0, -8.0, 13.0);
    assert!(ang_px(&out[0].center_ray, &gt) < 0.02);
    let (pxg, pyg) = ray_to_px(&out[0].center_ray, W, H);
    assert!((out[0].center_px[0] - pxg).abs() < 1e-6);
    assert!((out[0].center_px[1] - pyg).abs() < 1e-6);
}

#[test]
fn seam_straddling_marker_center_px_is_near_the_edge_not_the_middle() {
    let mut px = canvas();
    render_test_marker_id(&mut px, W, H, 179.8, 0.0, 5.0, 0.0, 0);
    let mut d = Detector::new(W, H, &defp()).unwrap();
    let out = d.scan(&view(&px), 8).unwrap();
    assert_eq!(out.len(), 1);
    let cx = out[0].center_px[0];
    let wf = W as f64;
    assert!(cx > -1.0 && cx < wf);
    let dist = (cx - 0.0).abs().min((cx - wf).abs());
    assert!(dist < wf * 0.05);
    assert!((cx - wf / 2.0).abs() > wf * 0.3);
}

#[test]
fn a_codeless_square_costs_no_bridge_no_decode_and_no_refine() {
    // The pre-bridge border probe turns this tier-2 blob away: a featureless patch has no
    // cell-mean spread at all, so a code-less blob never costs a bridge, decode or refine.
    let mut px = canvas();
    render_test_marker(&mut px, W, H, 40.0, 10.0, 5.0, 0.0);
    let mut d = Detector::new(W, H, &defp()).unwrap();
    assert_eq!(d.scan(&view(&px), 8).unwrap().len(), 0);
    assert!(d.stats().n_candidates >= 1);
    assert!(d.stats().n_prebridge_rejected >= 1);
    assert_eq!(d.stats().n_bridge_ok, 0);
    assert_eq!(d.stats().n_decode_ok, 0);
    assert_eq!(d.timings()[3], 0.0); // refine never ran at all
}

#[test]
fn the_pre_bridge_probe_screens_clutter_and_never_the_marker() {
    let mut px = canvas();
    render_test_marker_id(&mut px, W, H, -100.0, -8.0, 5.0, 13.0, 0);
    for k in 0..6usize {
        let (x0, y0, sq) = (400 + k * 220, 700 + (k & 1) * 160, 80 + 6 * k);
        for y in y0..=y0 + sq {
            for x in x0..=x0 + sq {
                px[y * W as usize + x] = 35;
            }
        }
    }
    let mut p = defp();
    p.time_budget_ms = 200.0; // examine every candidate
    let mut d = Detector::new(W, H, &p).unwrap();
    let out = d.scan(&view(&px), 16).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 0);
    assert!(d.stats().n_prebridge_rejected >= 4);
    assert!(d.stats().n_bridge_ok >= 1);
    assert_eq!(d.stats().n_accepted, 1);
}

#[test]
fn keeps_two_touching_neighbours() {
    // ids 6 and 7 at lat 22, quiet zones touching (gap 1.03).
    let mut px = canvas();
    let (half, lat, gap) = (4.5, 22.0, 1.03);
    let dlon = 2.0 * half * gap / (lat * K_PI / 180.0).cos();
    render_test_marker_id(&mut px, W, H, 10.0, lat, half, 0.0, 6);
    render_test_marker_id(&mut px, W, H, 10.0 + dlon, lat, half, 0.0, 7);
    let mut p = defp();
    p.time_budget_ms = 200.0;
    let mut d = Detector::new(W, H, &p).unwrap();
    let out = d.scan(&view(&px), 8).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out.iter().any(|m| m.id == 6));
    assert!(out.iter().any(|m| m.id == 7));
}

#[test]
fn dedupes_the_same_marker_seen_twice() {
    let mut px = canvas();
    render_test_marker_id(&mut px, W, H, 179.8, 0.0, 5.0, 0.0, 0);
    let mut d = Detector::new(W, H, &defp()).unwrap();
    assert_eq!(d.scan(&view(&px), 8).unwrap().len(), 1); // wrap pad must not double-report it
}

#[test]
fn reports_the_most_advanced_failure() {
    let px = canvas();
    let mut d = Detector::new(W, H, &defp()).unwrap();
    assert_eq!(d.scan(&view(&px), 8).unwrap().len(), 0);
    assert_eq!(d.stats().best_reject_code, R_NO_CANDIDATE);
}

#[test]
fn tags_a_polar_marker_with_pole_cap() {
    let mut px = canvas();
    render_test_marker_id(&mut px, W, H, 0.0, 80.0, 5.0, 0.0, 3);
    let mut p = defp();
    p.time_budget_ms = 200.0;
    let mut d = Detector::new(W, H, &p).unwrap();
    let out = d.scan(&view(&px), 8).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 3);
    assert!(out[0].flags & FLAG_POLE_CAP != 0);
    assert!(out[0].flags & FLAG_SECOND_SCALE == 0);
}

#[test]
fn tags_a_small_marker_with_second_scale() {
    let mut px = canvas();
    render_test_marker_id(&mut px, W, H, 0.0, 0.0, 1.0, 0.0, 3); // too small for f=4
    let mut p = defp();
    p.time_budget_ms = 200.0;
    let mut d = Detector::new(W, H, &p).unwrap();
    let out = d.scan(&view(&px), 8).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 3);
    assert!(out[0].flags & FLAG_SECOND_SCALE != 0);
    assert!(out[0].flags & FLAG_POLE_CAP == 0);
}

#[test]
fn small_pass_runs_whenever_budget_is_left() {
    let blank = canvas();
    let mut marker = canvas();
    // A comfortably large marker: the f=4 pass resolves it at reference-grade shape.
    render_test_marker_id(&mut marker, W, H, -100.0, -8.0, 5.0, 13.0, 0);

    // 1. Featureless frame, ample budget: nothing decoded and no tier-1 candidate.
    let mut p = defp();
    p.time_budget_ms = 50.0;
    let mut d = Detector::new(W, H, &p).unwrap();
    assert_eq!(d.scan(&view(&blank), 8).unwrap().len(), 0);
    assert_eq!(d.stats().n_tier1, 0);
    assert_eq!(d.stats().ran_small, 1);

    // 2. Same frame, but no budget left to finish the pass.
    let mut p = defp();
    p.time_budget_ms = 0.0;
    let mut d = Detector::new(W, H, &p).unwrap();
    assert_eq!(d.scan(&view(&blank), 8).unwrap().len(), 0);
    assert_eq!(d.stats().n_tier1, 0);
    assert_eq!(d.stats().ran_small, 0);

    // 3. A marker the f=4 pass sees at reference-grade shape, with ample budget: the pass
    // still runs (a larger marker must not hide smaller ones), but never replaces or
    // duplicates what f=4 accepted.
    let mut p = defp();
    p.time_budget_ms = 50.0;
    let mut d = Detector::new(W, H, &p).unwrap();
    let got = d.scan(&view(&marker), 8).unwrap();
    assert!(d.stats().n_tier1 >= 1);
    assert_eq!(d.stats().ran_small, 1);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id, 0);
    assert_eq!(got[0].flags & eqar::codes::FLAG_SECOND_SCALE, 0);
}

// ---- per-corner validity through the whole detector ----

struct OccScene {
    px: Vec<u8>,
    gt: [Vec3; 4],
    cpx: [[f64; 2]; 4],
    side: f64,
}

impl OccScene {
    fn new(id: i32, roll: f64) -> OccScene {
        let mut px = canvas();
        render_test_marker_id(&mut px, W, H, -100.0, -8.0, 5.0, roll, id);
        let gt = test_marker_corner_rays(-100.0, -8.0, 5.0, roll);
        let mut cpx = [[0.0; 2]; 4];
        for i in 0..4 {
            let (x, y) = ray_to_px(&gt[i], W, H);
            cpx[i] = [x, y];
        }
        let side = (cpx[1][0] - cpx[0][0]).hypot(cpx[1][1] - cpx[0][1]);
        OccScene { px, gt, cpx, side }
    }
    fn std() -> OccScene {
        OccScene::new(3, 13.0)
    }
    fn boxed(&mut self, a: usize, dx0: f64, dy0: f64, dx1: f64, dy1: f64, val: u8) {
        let r = |v: f64| v.round() as i64;
        let x0 = r(self.cpx[a][0] + dx0 * self.side);
        let x1 = r(self.cpx[a][0] + dx1 * self.side);
        let y0 = r(self.cpx[a][1] + dy0 * self.side);
        let y1 = r(self.cpx[a][1] + dy1 * self.side);
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.px[(y * W as i64 + x) as usize] = val;
            }
        }
    }
    fn view(&self) -> View<'_> {
        view(&self.px)
    }
    fn prior(&self, id: i32, sigma_deg: f64) -> Prior {
        Prior {
            id,
            ray: self.gt,
            sigma_rad: sigma_deg * K_PI / 180.0,
        }
    }
}

/// Rotate `v` by `ang` radians about the unit axis `ax` (Rodrigues).
fn rotate_about(v: &Vec3, ax: &Vec3, ang: f64) -> Vec3 {
    let (c, s) = (ang.cos(), ang.sin());
    let k = cross(ax, v);
    let kd = dot(ax, v) * (1.0 - c);
    Vec3::new(
        v.x * c + k.x * s + ax.x * kd,
        v.y * c + k.y * s + ax.y * kd,
        v.z * c + k.z * s + ax.z * kd,
    )
}

#[test]
fn a_clean_marker_has_four_valid_unit_scale_corners() {
    let sc = OccScene::std();
    let mut d = Detector::new(W, H, &defp()).unwrap();
    let out = d.scan(&sc.view(), 8).unwrap();
    assert_eq!(out.len(), 1);
    for i in 0..4 {
        assert!(out[0].corner_valid[i] != 0);
        assert!((out[0].sigma_scale[i] as f64 - 1.0).abs() <= 1e-6);
        assert_eq!(out[0].corner_flags[i], 0);
        assert!(out[0].end_support[i] >= 0.9);
    }
    assert_eq!(out[0].prov, 0);
}

#[test]
fn a_hidden_corner_tip_is_invalid_cold_and_a_scaled_measurement_when_the_id_is_known() {
    {
        // cold (id unknown), dark bite over corner 0: detected by the strict decode, the
        // corner is not a measurement
        let mut sc = OccScene::std();
        sc.boxed(0, -0.25, -0.25, 0.3, 0.3, 40);
        let mut d = Detector::new(W, H, &defp()).unwrap();
        let out = d.scan(&sc.view(), 8).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 3);
        assert!(out[0].corner_valid[0] == 0);
        assert!(out[0].sigma_scale[0] > 10.0);
        for i in 1..4 {
            assert!(out[0].corner_valid[i] != 0 && out[0].sigma_scale[i] == 1.0);
        }
    }
    {
        // a bright bite breaks the black border: cold rejects it, known id verifies it
        let mut sc = OccScene::std();
        sc.boxed(0, -0.25, -0.25, 0.3, 0.3, 250);
        let mut cold = Detector::new(W, H, &defp()).unwrap();
        assert_eq!(cold.scan(&sc.view(), 8).unwrap().len(), 0);
        let mut known = Detector::new(W, H, &defp()).unwrap();
        let pr = sc.prior(3, 0.5);
        let out = known.verify(&sc.view(), &[pr], 8).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 3);
        assert!(out[0].corner_valid[0] != 0 && out[0].sigma_scale[0] > 10.0);
        assert!(out[0].corner_flags[0] & K_CORNER_COMPLETED == 0);
        for i in 1..4 {
            assert!(out[0].corner_valid[i] != 0 && out[0].sigma_scale[i] == 1.0);
            assert!(ang_px(&out[0].ray[i], &sc.gt[i]) < 0.1);
        }
    }
}

#[test]
fn a_linefit_quad_is_rechecked_and_a_disagreeing_side_is_handled_per_mode() {
    // a dark bite over corner 0 makes the bridge return a 4-line-fit (LINEFIT) quad, which
    // gets the post-refine re-check
    let mut sc = OccScene::std();
    sc.boxed(0, -0.25, -0.25, 0.3, 0.3, 40);
    {
        let mut d = Detector::new(W, H, &defp()).unwrap();
        let out = d.scan(&sc.view(), 8).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].prov, 1);
        assert!(d.stats().n_recheck >= 1);
        assert_eq!(d.stats().n_side_disagree, 0);
    }
    {
        // forced disagreement, cold: the LINEFIT candidate is rejected
        let mut d = Detector::new(W, H, &defp()).unwrap();
        d.set_consistency_bounds(1e-9, 1e-9);
        let out = d.scan(&sc.view(), 8).unwrap();
        assert!(d.stats().n_side_disagree >= 1);
        for m in &out {
            assert!(m.prov != 1);
        }
    }
    {
        // forced disagreement, known id: accepted, but every disagreeing side's corners
        // are invalid with SIDE
        let mut d = Detector::new(W, H, &defp()).unwrap();
        d.set_consistency_bounds(1e-9, 1e-9);
        let pr = sc.prior(3, 0.5);
        let out = d.verify(&sc.view(), &[pr], 8).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].prov, 1);
        assert!(d.stats().n_side_disagree >= 1);
        let mut nside = 0;
        for i in 0..4 {
            if out[0].corner_flags[i] & K_CORNER_SIDE != 0 {
                nside += 1;
                assert!(out[0].corner_valid[i] == 0);
            }
        }
        assert!(nside >= 2);
    }
}

#[test]
fn verify_finds_the_expected_id_near_its_prior_and_never_another_id() {
    let mut sc = OccScene::std();
    sc.boxed(0, -0.25, -0.25, 0.3, 0.3, 250); // a bright bite on the border: cold rejects it
    {
        let mut d = Detector::new(W, H, &defp()).unwrap();
        let out = d.verify(&sc.view(), &[sc.prior(3, 0.5)], 8).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 3);
        assert!(out[0].flags & FLAG_TRACK != 0);
    }
    {
        // the same place with every other id as the expectation: nothing verifies
        let mut d = Detector::new(W, H, &defp()).unwrap();
        for id in 0..50 {
            if id == 3 {
                continue;
            }
            let out = d.verify(&sc.view(), &[sc.prior(id, 0.5)], 8).unwrap();
            assert_eq!(out.len(), 0, "id {id}");
        }
    }
    {
        // priors are per quad: a wrong-id prior and the right one in one call give
        // exactly the right one
        let mut d = Detector::new(W, H, &defp()).unwrap();
        let out = d
            .verify(&sc.view(), &[sc.prior(7, 0.5), sc.prior(3, 0.5)], 8)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, 3);
    }
    {
        // the cold path on the same detector is still strict (nothing was left lenient)
        let mut d = Detector::new(W, H, &defp()).unwrap();
        assert_eq!(
            d.verify(&sc.view(), &[sc.prior(3, 0.5)], 8).unwrap().len(),
            1
        );
        assert_eq!(d.scan(&sc.view(), 8).unwrap().len(), 0);
    }
}

#[test]
fn a_corner_off_its_prior_is_invalid_with_prior_and_a_prior_elsewhere_rejects_the_marker() {
    let sc = OccScene::std();
    let sigma = 0.05; // deg; 3 sigma = 0.15 deg ~ 1.6 px_eq
    let up = normalize(&cross(&sc.gt[0], &sc.gt[1]));
    let five = 5.0 * sigma * K_PI / 180.0;
    {
        // one corner 5 sigma off: that corner is invalid + PRIOR, the other three are fine
        let mut d = Detector::new(W, H, &defp()).unwrap();
        let mut pr = sc.prior(3, sigma);
        pr.ray[2] = rotate_about(&pr.ray[2], &up, five);
        let out = d.verify(&sc.view(), &[pr], 8).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].corner_valid[2] == 0);
        assert!(out[0].corner_flags[2] & K_CORNER_PRIOR != 0);
        for i in [0, 1, 3] {
            assert!(out[0].corner_valid[i] != 0);
            assert_eq!(out[0].corner_flags[i] & K_CORNER_PRIOR, 0);
        }
    }
    {
        // three corners off: only one consistent corner is left, so the marker is rejected
        let mut d = Detector::new(W, H, &defp()).unwrap();
        let mut pr = sc.prior(3, sigma);
        for i in [0, 1, 2] {
            pr.ray[i] = rotate_about(&pr.ray[i], &up, five);
        }
        assert_eq!(d.verify(&sc.view(), &[pr], 8).unwrap().len(), 0);
        assert_eq!(d.stats().best_reject_code, R_ID_NOT_EXPECTED);
    }
    {
        // a rigid 2 sigma error is inside the check: all four corners stay valid
        let mut d = Detector::new(W, H, &defp()).unwrap();
        let mut pr = sc.prior(3, sigma);
        for i in 0..4 {
            pr.ray[i] = rotate_about(&pr.ray[i], &up, 2.0 * sigma * K_PI / 180.0);
        }
        let out = d.verify(&sc.view(), &[pr], 8).unwrap();
        assert_eq!(out.len(), 1);
        for i in 0..4 {
            assert!(out[0].corner_valid[i] != 0);
        }
    }
}

#[test]
fn a_dark_bar_hiding_most_of_the_top_edge_verifies_with_the_top_corners_not_valid() {
    let mut sc = OccScene::std();
    sc.boxed(0, -0.3, -0.12, 0.6, 0.12, 30);
    let mut d = Detector::new(W, H, &defp()).unwrap();
    let out = d.verify(&sc.view(), &[sc.prior(3, 0.5)], 8).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 3);
    assert!(out[0].corner_valid[0] == 0 && out[0].corner_valid[1] == 0);
    assert!(out[0].corner_valid[2] != 0 && out[0].corner_valid[3] != 0);
    for i in 0..4 {
        if out[0].corner_flags[i] & K_CORNER_COMPLETED != 0 {
            assert!(out[0].corner_valid[i] == 0);
        }
    }
    let mut cold = Detector::new(W, H, &defp()).unwrap();
    assert_eq!(cold.scan(&sc.view(), 8).unwrap().len(), 0); // rejected without the id
}

#[test]
fn a_dark_bar_merged_outside_the_border_never_leaves_a_valid_corner_on_the_widened_side() {
    let mut sc = OccScene::new(3, 0.0);
    let module = sc.side / 6.0;
    let s = sc.side;
    sc.boxed(1, 0.0, 0.0, 0.5 * module / s, 1.0, 30); // right of corner 1, down the right side
    let mut d = Detector::new(W, H, &defp()).unwrap();
    let out = d.verify(&sc.view(), &[sc.prior(3, 0.5)], 8).unwrap();
    assert!(out.len() <= 1);
    for m in &out {
        for k in 0..4 {
            if m.corner_valid[k] == 0 {
                continue;
            }
            let (px, py) = ray_to_px(&m.ray[k], W, H);
            let mut best = 1e9f64;
            for g in 0..4 {
                best = best.min((px - sc.cpx[g][0]).hypot(py - sc.cpx[g][1]));
            }
            assert!(best < module / 3.0);
        }
    }
}

#[test]
fn a_track_time_verify_leaves_lenient_handling_off_for_the_scan_fallback_and_later_scans() {
    let clean = OccScene::std();
    let mut bar = OccScene::std();
    bar.boxed(0, -0.3, -0.12, 0.6, 0.12, 30);
    let mut d = Detector::new(W, H, &defp()).unwrap();
    let cur = d.scan(&clean.view(), 4).unwrap();
    assert_eq!(cur.len(), 1);
    d.track(&bar.view(), &cur, 4, 1).unwrap();
    assert_eq!(d.scan(&bar.view(), 4).unwrap().len(), 0);
    let blank = canvas(); // nothing there: the verify against the prediction fails
    d.track(&view(&blank), &cur, 4, 1).unwrap();
    assert_eq!(d.scan(&bar.view(), 4).unwrap().len(), 0);
    assert_eq!(d.track(&bar.view(), &[], 4, 1).unwrap().len(), 0);
}

#[test]
fn a_corner_next_to_an_unjudged_ring_side_gets_sigma_scale_two_only_when_the_id_is_known() {
    // A small marker (module < 3 px) has unjudged ring sides: no ring evidence. Known ID
    // floors those corners at 2; the cold detection of the same frame keeps sigma_scale 1.
    let mut px = canvas();
    render_test_marker_id(&mut px, W, H, -100.0, -8.0, 0.9, 13.0, 3);
    let v = view(&px);
    let gt = test_marker_corner_rays(-100.0, -8.0, 0.9, 13.0);
    let pr = Prior {
        id: 3,
        ray: gt,
        sigma_rad: 0.5 * K_PI / 180.0,
    };
    let mut d = Detector::new(W, H, &defp()).unwrap();
    let out: Vec<Detection> = d.verify(&v, &[pr], 4).unwrap();
    assert_eq!(out.len(), 1);
    let mut n_unjudged = 0;
    for s in 0..4 {
        if out[0].ring[s] >= 0.0 {
            continue;
        }
        n_unjudged += 1;
        assert!(out[0].sigma_scale[s] >= 2.0 && out[0].sigma_scale[(s + 1) & 3] >= 2.0);
    }
    assert!(n_unjudged > 0);
    let mut cold = Detector::new(W, H, &defp()).unwrap();
    let o = cold.scan(&v, 4).unwrap();
    if o.len() == 1 {
        for s in 0..4 {
            assert!(o[0].sigma_scale[s] < 2.0);
        }
    }
}

#[test]
fn ring_width_reads_one_module_on_every_side_at_every_rotation_and_under_0_7_for_an_inward_shift() {
    // The detector-driven original of the guards test: `out.ring[s]` is the detector's own
    // reading of a clean marker, whichever quad corner the bridge listed first.
    let pix = K_PI / H as f64;
    for roll in [13.0, 103.0, 193.0, 283.0] {
        let mut px = canvas();
        render_test_marker_id(&mut px, W, H, -100.0, -8.0, 5.0, roll, 3);
        let v = view(&px);
        let mut d = Detector::new(W, H, &defp()).unwrap();
        let out = d.scan(&v, 8).unwrap();
        assert_eq!(out.len(), 1);
        let hm = homography_closed(&out[0].ray).expect("homography");
        let module = 2.0 / 6.0; // 4x4 marker + 1-module border: 6 modules across [-1, 1]
        let shift = 0.6 * module;
        for s in 0..4usize {
            assert!(out[0].ring[s] > 0.9 && out[0].ring[s] < 1.1);
            // move side s (and its two corners) 0.6 module INWARD: only 0.4 module of the
            // black border stays outside it
            let mut c = [[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]];
            for k in [s, (s + 1) & 3] {
                match s {
                    0 => c[k][1] -= shift,
                    1 => c[k][0] -= shift,
                    2 => c[k][1] += shift,
                    _ => c[k][0] += shift,
                }
            }
            let mut q = [Vec3::default(); 4];
            for i in 0..4 {
                q[i] = plane_to_ray(&hm, c[i][0], c[i][1]);
            }
            let w = ring_width(&v, 0, out[0].id, &q, s, pix);
            assert!((0.0..0.7).contains(&w), "w = {w}");
            assert!(w < out[0].ring[s]);
        }
    }
}
