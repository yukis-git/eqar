//! Parity of the bridge stage against the C++ reference:
//! tests/fixtures/b3_bridge.json (Bridge::run results) and b3_patch.json (patch render,
//! threshold, contour and quad diagnostics).
//!
//! Every scene is rebuilt from its description (synth markers, rectangle occluders, an
//! illumination ramp, LCG noise; checked by sha256). The bridge is seeded with the stored
//! proposal candidates, so this test does not depend on the proposal stage.
//!
//! Gates. On aarch64 with NEON the Rust patch renderer is the same float32 kernel as the
//! C++ one (same polynomial atan2, same gathers, fused multiply-adds in the same places), so
//! the patch bytes are bit-identical (sha256), every integer decision (threshold, contours,
//! selection, ok / provenance / alternate) matches exactly, and the corner rays agree to
//! double rounding (the C++ build can contract `a*b+c` inside the double geometry, hence
//! not bit-exact). Elsewhere (x86_64, or `no-neon`) the patch is rendered by the scalar
//! double-precision path, which differs from the C++ float kernel by up to 1 DN on a few
//! pixels; that can move a contour pixel and so a polished corner by a fraction of a patch
//! pixel, and (rarely) flip a borderline accept/reject. There only the stage-level
//! structure is compared with documented tolerances, and `ok` must match on every run
//! except a small allowance.

use eqar::bridge::*;
use eqar::geometry::Vec3;
use eqar::image::{Fmt, View};
use eqar_testkit::json::Value;
use eqar_testkit::load_fixture;
use eqar_testkit::sha256::sha256_hex;
use eqar_testkit::synth::{test_marker_center_ray, SynthCase};

const STRICT: bool = cfg!(all(target_arch = "aarch64", not(feature = "no-neon")));

/// Max absolute difference of a corner-ray component (rays are unit vectors).
const RAY_TOL: f64 = if STRICT { 1e-10 } else { 3e-4 };
/// Max difference of a patch-pixel coordinate (polished quads, in patch px).
const QUAD_TOL: f64 = if STRICT { 1e-8 } else { 0.5 };
/// Runs whose `ok` may differ from the reference on the loose gate.
const LOOSE_OK_MISMATCHES: usize = 4;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
}

fn build_scene(c: &Value) -> Vec<u8> {
    let (w, h) = (c.get("w").i64() as usize, c.get("h").i64() as usize);
    let mut px = SynthCase::from_json(c).render();
    for o in c.get("occ").arr() {
        let f = o.arr();
        let (x0, y0, x1, y1, val) = (
            f[0].i64() as usize,
            f[1].i64() as usize,
            f[2].i64() as usize,
            f[3].i64() as usize,
            f[4].i64() as u8,
        );
        for y in y0..y1 {
            for x in x0..x1 {
                px[y * w + x] = val;
            }
        }
    }
    let ramp = c.get("ramp").f64();
    if ramp > 0.0 {
        for y in 0..h {
            for x in 0..w {
                let k = 0.45 + ramp * x as f64 / (w - 1) as f64;
                let i = y * w + x;
                px[i] = (px[i] as f64 * k).min(255.0) as u8;
            }
        }
    }
    let amp = c.get("noise_amp").i64() as i32;
    if amp > 0 {
        let mut r = Lcg(c.get("noise_seed").i64() as u32);
        for p in px.iter_mut() {
            let v = *p as i32 + ((r.next() >> 24) as i32) % (2 * amp + 1) - amp;
            *p = v.clamp(0, 255) as u8;
        }
    }
    px
}

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

fn rays(v: &Value) -> Vec<Vec3> {
    let f = v.f64s();
    f.chunks(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect()
}

fn max_ray_diff(a: &[Vec3], b: &[Vec3]) -> f64 {
    let mut m = 0.0f64;
    for (p, q) in a.iter().zip(b) {
        m = m
            .max((p.x - q.x).abs())
            .max((p.y - q.y).abs())
            .max((p.z - q.z).abs());
    }
    m
}

/// Smallest ray difference over the four cyclic starting corners, and the shift that gives it.
/// The bridge rolls its quad so that corner 0 minimises x + y in patch coordinates; for a
/// quad whose two corners tie there (a marker rolled by 45 degrees: exactly, in the
/// symmetric synth scenes) the 1e-16 differences in the double geometry decide which of the
/// two comes first. The corners themselves must still agree.
fn cyclic_ray_diff(a: &[Vec3], b: &[Vec3]) -> (f64, usize) {
    let mut best = (f64::INFINITY, 0);
    for k in 0..4 {
        let rolled: Vec<Vec3> = (0..4).map(|i| b[(i + k) % 4]).collect();
        let d = max_ray_diff(a, &rolled);
        if d < best.0 {
            best = (d, k);
        }
    }
    best
}

#[test]
fn bridge_runs_match_the_cpp_reference() {
    let fx = load_fixture("b3_bridge.json");
    let (mut n_runs, mut n_ok, mut n_alt, mut n_prov1, mut mismatches, mut n_shifted) =
        (0, 0, 0, 0, 0usize, 0usize);
    let mut worst = 0.0f64;
    for case in fx.get("cases").arr() {
        let name = case.get("name").str();
        let px = build_scene(case);
        assert_eq!(
            sha256_hex(&px),
            case.get("sha256").str(),
            "{name}: scene bytes"
        );
        let (w, h) = (case.get("w").i64() as i32, case.get("h").i64() as i32);
        let v = gray_view(&px, w, h);
        let mut b = Bridge::new(w, h, 0).unwrap();
        for (ri, run) in case.get("runs").arr().iter().enumerate() {
            let at = format!("{name} run {ri}");
            let cand = run.get("cand").f64s();
            let centre = Vec3::new(cand[0], cand[1], cand[2]);
            let lenient = run.get("lenient").i64() != 0;
            b.set_lenient(lenient);
            if lenient {
                b.begin_frame();
            }
            let mut out = [Vec3::default(); 4];
            let ok = match run.get("plan").str() {
                "scan" => b.run_scan(&v, centre, cand[3], &mut out),
                "track" => b.run(&v, centre, cand[3], &mut out, &K_TRACK_PLAN),
                "unscreened" => b.run(
                    &v,
                    centre,
                    cand[3],
                    &mut out,
                    &[K_SCAN_PLAN[1], K_SCAN_PLAN[2]],
                ),
                p => panic!("unknown plan {p}"),
            }
            .unwrap();
            n_runs += 1;
            let want_ok = run.get("ok").i64() != 0;
            if ok != want_ok {
                mismatches += 1;
                if STRICT {
                    panic!("{at}: ok {ok}, reference {want_ok}");
                }
                continue;
            }
            if !ok {
                continue;
            }
            n_ok += 1;
            let want_out = rays(run.get("out"));
            let (d, shift) = cyclic_ray_diff(&out, &want_out);
            worst = worst.max(d);
            assert!(d <= RAY_TOL, "{at}: corner rays differ by {d:e}");
            n_shifted += (shift != 0) as usize;
            let (prov, has_alt) = (b.prov() as i64, b.alt_quad());
            let want_prov = run.get("prov").i64();
            let want_alt = run.get("has_alt").i64() != 0;
            if STRICT {
                assert_eq!(prov, want_prov, "{at}: provenance");
                assert_eq!(has_alt.is_some(), want_alt, "{at}: alternate present");
            }
            if prov != 0 {
                n_prov1 += 1;
            }
            if let (Some(alt), true) = (has_alt, want_alt) {
                n_alt += 1;
                let (da, shift) = cyclic_ray_diff(&alt, &rays(run.get("alt")));
                n_shifted += (shift != 0) as usize;
                worst = worst.max(da);
                assert!(da <= RAY_TOL, "{at}: alternate rays differ by {da:e}");
            }
        }
    }
    eprintln!(
        "b3 bridge: {n_runs} runs, {n_ok} ok, {n_alt} alternates compared, {n_prov1} non-legacy, \
         {mismatches} ok mismatches, {n_shifted} tie-rolled quads, worst ray diff {worst:e}"
    );
    assert!(n_shifted <= 12);
    assert!(n_runs >= 200 && n_ok >= 150);
    assert!(mismatches <= LOOSE_OK_MISMATCHES);
}

fn quad_of(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Null => None,
        v => Some(v.f64s()),
    }
}

fn flat_quad(q: &[[f64; 2]; 4]) -> Vec<f64> {
    q.iter().flat_map(|p| [p[0], p[1]]).collect()
}

fn check_quad(at: &str, got: Option<Vec<f64>>, want: Option<Vec<f64>>, worst: &mut f64) {
    match (got, want) {
        (None, None) => {}
        (Some(g), Some(w)) => {
            for (a, b) in g.iter().zip(&w) {
                *worst = worst.max((a - b).abs());
                assert!((a - b).abs() <= QUAD_TOL, "{at}: quad {g:?} vs {w:?}");
            }
        }
        (g, w) => {
            // Only the strict gate demands the same quads exist.
            if STRICT {
                panic!("{at}: got {g:?} want {w:?}");
            }
        }
    }
}

#[test]
fn patch_stages_match_the_cpp_reference() {
    let scenes = load_fixture("b3_bridge.json");
    let fx = load_fixture("b3_patch.json");
    let mut worst = 0.0f64;
    let mut sha_ok = 0;
    for (ci, pc) in fx.get("cases").arr().iter().enumerate() {
        let scene_name = pc.get("scene").str();
        let at0 = format!("patch {ci} ({scene_name})");
        let case = scenes
            .get("cases")
            .arr()
            .iter()
            .find(|c| c.get("name").str() == scene_name)
            .unwrap_or_else(|| panic!("scene {scene_name}"));
        let px = build_scene(case);
        let (w, h) = (case.get("w").i64() as i32, case.get("h").i64() as i32);
        let v = gray_view(&px, w, h);
        let centre = test_marker_center_ray(
            pc.get("lon").f64(),
            pc.get("lat").f64(),
            pc.get("roll").f64(),
        );
        let mut g = BridgeGrid::new();
        let mut p = Patch::new_max();
        patch_render(
            &v,
            &centre,
            pc.get("half_fov").f64(),
            &mut g,
            &mut p,
            pc.get("rate").f64(),
            pc.get("cap").i64() as i32,
        );
        let s = p.s;
        let n = (s * s) as usize;
        assert_eq!(s as i64, pc.get("s").i64(), "{at0}: patch side");
        assert_eq!(p.clipped as i64, pc.get("clipped").i64(), "{at0}: clipped");
        assert!(
            (p.tan_t - pc.get("tan_t").f64()).abs() <= 1e-12,
            "{at0}: tan_t"
        );
        let e3 = pc.get("e3").f64s();
        assert!(
            (p.e3.x - e3[0])
                .abs()
                .max((p.e3.y - e3[1]).abs())
                .max((p.e3.z - e3[2]).abs())
                <= 1e-12,
            "{at0}: e3"
        );
        let min_len = 12.max((0.3544 * s as f64) as i32);
        let mut bw = vec![0u8; n];
        let mut sc = Scratch::default();
        for mode in 0..2 {
            let m = pc.get(if mode == 0 { "mode0" } else { "mode1" });
            let at = format!("{at0} mode {mode}");
            let (thr, hash) = if mode == 0 {
                let thr = otsu_hist8_u8(&p.img[..n]);
                patch_binarise_raw(&p, thr, &mut bw);
                (thr, sha256_hex(&p.img[..n]))
            } else {
                patch_flatten(&mut p, &mut sc);
                let thr = otsu_hist8(&p.flat[..n]);
                patch_binarise(&p, thr, &mut bw);
                let bytes: Vec<u8> = p.flat[..n].iter().flat_map(|f| f.to_le_bytes()).collect();
                (thr, sha256_hex(&bytes))
            };
            if STRICT {
                assert_eq!(hash, m.get("sha").str(), "{at}: patch bytes");
                assert_eq!(thr, m.get("thr").f64(), "{at}: threshold");
                sha_ok += 1;
            } else {
                assert!((thr - m.get("thr").f64()).abs() <= 3.0, "{at}: threshold");
            }
            let mut c = PatchContours::default();
            let nc = trace_contours(&bw, s, &mut c, min_len, &mut sc);
            let odd = c.depth.iter().filter(|d| **d % 2 != 0).count() as i64;
            let idx = select_marker_contour(&c, s, 0.80, &mut sc);
            if STRICT {
                assert_eq!(nc as i64, m.get("n_contours").i64(), "{at}: contours");
                assert_eq!(odd, m.get("n_odd").i64(), "{at}: holes");
                assert_eq!(
                    (c.pts.len() / 2) as i64,
                    m.get("n_pts").i64(),
                    "{at}: points"
                );
                assert_eq!(idx as i64, m.get("idx").i64(), "{at}: selected contour");
            } else if idx as i64 != m.get("idx").i64() {
                continue;
            }
            let (mut leg, mut ord, mut mis, mut pol, mut lines) = (None, None, None, None, None);
            if idx >= 0 {
                let mut q = [[0.0; 2]; 4];
                if quad_from_contour(&c, idx, &mut q, &mut sc) {
                    leg = Some(flat_quad(&q));
                    mis = Some(contour_misfit(&c, idx, &q));
                    order_quad(&mut q);
                    ord = Some(flat_quad(&q));
                    let mut qp = q;
                    if polish_quad(&c, idx, &mut qp, None, &PolishBand::default(), &mut sc) {
                        order_quad(&mut qp);
                        pol = Some(flat_quad(&qp));
                    }
                }
                let mut ql = [[0.0; 2]; 4];
                if quad_from_lines(&c, idx, &mut ql, &mut sc) {
                    order_quad(&mut ql);
                    lines = Some(flat_quad(&ql));
                }
            }
            check_quad(
                &format!("{at} quad"),
                leg,
                quad_of(m.get("quad")),
                &mut worst,
            );
            check_quad(
                &format!("{at} ordered"),
                ord,
                quad_of(m.get("quad_ordered")),
                &mut worst,
            );
            check_quad(
                &format!("{at} polished"),
                pol,
                quad_of(m.get("polished")),
                &mut worst,
            );
            check_quad(
                &format!("{at} lines"),
                lines,
                quad_of(m.get("lines")),
                &mut worst,
            );
            match (mis, m.get("misfit")) {
                (Some(g), Value::Num(w)) => {
                    assert!(
                        (g - w).abs() <= if STRICT { 1e-12 } else { 0.05 },
                        "{at}: misfit"
                    )
                }
                (None, Value::Null) => {}
                (g, w) => {
                    if STRICT {
                        panic!("{at}: misfit {g:?} vs {w:?}");
                    }
                }
            }
        }
    }
    eprintln!(
        "b3 patch: {} cases, {sha_ok} strict patch-byte matches, worst quad diff {worst:e}",
        fx.get("cases").arr().len()
    );
}
