//! Whole-pipeline parity against the C++ reference through its C interface:
//! tests/fixtures/b5_pipeline.json (detect / track / verify steps on synthetic scenes with
//! an effectively unlimited time budget: grids, poles, seam, large and tiny markers, multi
//! id, occlusion, clutter, tracking sequences with frame gaps, coasting and the scan
//! fallback, both quality presets, second scale on and off, expected id, in-use sets).
//!
//! Every scene is rebuilt from its description (checked by sha256). Tracking steps are fed
//! the stored C++ output of the step they follow, so one step cannot move another.
//!
//! Gates. aarch64 with NEON: id, reject code, flags, provenance, corner validity and flags,
//! bit/erasure counts and the stage counters equal; every corner within 0.002 px of the
//! reference. Elsewhere (x86_64, or `no-neon`): same count and ids, corners within 0.02 px
//! (measured as the chord between the unit rays in px_eq, which equals the pixel distance
//! except in the polar singularity, where the longitude of a point is ill-conditioned).

#![allow(clippy::field_reassign_with_default)]

use eqar::geometry::{px_err, Vec3};
use eqar::{Detector, EqarError, ImageView, Marker, Params, PixelFormat, Prior, Quality};
use eqar_testkit::json::Value;
use eqar_testkit::load_fixture;
use eqar_testkit::sha256::sha256_hex;
use eqar_testkit::synth::SynthCase;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const STRICT: bool = cfg!(all(target_arch = "aarch64", not(feature = "no-neon")));
/// Corner position gate, px.
const CORNER_TOL: f64 = if STRICT { 0.002 } else { 0.02 };

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

fn f32s(v: &Value) -> Vec<f32> {
    v.arr()
        .iter()
        .map(|x| f32::from_bits(x.i64() as u32))
        .collect()
}

fn parse_marker(m: &Value) -> Marker {
    let f = f32s(m.get("f"));
    let mut o = Marker::default();
    for i in 0..4 {
        o.corner_px[i] = [f[2 * i], f[2 * i + 1]];
        o.ray[i] = [f[8 + 3 * i], f[9 + 3 * i], f[10 + 3 * i]];
        o.corner_sigma_scale[i] = f[30 + i];
        o.corner_valid[i] = m.get("valid").arr()[i].i64() != 0;
        o.corner_flags[i] = m.get("cflags").arr()[i].i64() as u8;
    }
    o.center_px = [f[20], f[21]];
    o.center_ray = [f[22], f[23], f[24]];
    o.quality = f[25];
    o.side_px = f[26];
    o.edge_resid_px = f[27];
    o.contrast_dn = f[28];
    o.min_cross = f[29];
    o.id = m.get("id").i64() as i32;
    o.flags = m.get("flags").i64() as u32;
    o.prov = m.get("prov").i64() as u8;
    o.n_bit_errors = m.get("nbe").i64() as i16;
    o.border_fail = m.get("bf").i64() as i16;
    o.n_erased = m.get("ner").i64() as i16;
    o.reject_code = m.get("reject").i64() as i32;
    o
}

fn parse_prior(p: &Value) -> Prior {
    let f = f32s(p.get("ray"));
    let mut ray = [[0f32; 3]; 4];
    for i in 0..4 {
        ray[i] = [f[3 * i], f[3 * i + 1], f[3 * i + 2]];
    }
    Prior {
        id: p.get("id").i64() as i32,
        ray,
        sigma_deg: f32::from_bits(p.get("sigma").i64() as u32),
    }
}

fn params_of(p: &Value) -> Params {
    let mut o = Params::default();
    o.quality = if p.get("quality").i64() == 0 {
        Quality::Lean
    } else {
        Quality::Default
    };
    o.use_second_scale = p.get("second").i64() != 0;
    o.reject_high_resid = p.get("high_resid").i64() != 0;
    o.expect_id = p.get("expect_id").i64() as i32;
    o.in_use_ids = match p.get("in_use") {
        Value::Null => None,
        v => Some(v.arr().iter().map(|x| x.i64() as i32).collect()),
    };
    o.coast_max = p.get("coast_max").i64() as u32;
    o.time_budget_ms = p.get("budget_ms").f64() as f32;
    o
}

fn error_of(code: i64) -> EqarError {
    match code {
        -2 => EqarError::Size,
        -4 => EqarError::Stride,
        -5 => EqarError::Alloc,
        -6 => EqarError::Dict,
        -7 => EqarError::Cap,
        -8 => EqarError::NoInit,
        -9 => EqarError::Params,
        c => panic!("unexpected reference error {c}"),
    }
}

fn vec3(r: &[f32; 3]) -> Vec3 {
    Vec3::new(r[0] as f64, r[1] as f64, r[2] as f64)
}

/// Largest deviations seen, for the report (`--nocapture`).
#[derive(Default)]
struct Worst {
    corner: f64,
    center: f64,
    ray_px: f64,
    scalar: f64,
}

fn check_marker(got: &Marker, want: &Marker, w: i32, h: i32, tag: &str, worst: &mut Worst) {
    assert_eq!(got.id, want.id, "{tag}: id");
    assert_eq!(got.reject_code, want.reject_code, "{tag}: reject");
    if STRICT {
        assert_eq!(got.flags, want.flags, "{tag}: flags");
        assert_eq!(got.prov, want.prov, "{tag}: prov");
        assert_eq!(got.corner_valid, want.corner_valid, "{tag}: corner_valid");
        assert_eq!(got.corner_flags, want.corner_flags, "{tag}: corner_flags");
        assert_eq!(
            (got.n_bit_errors, got.border_fail, got.n_erased),
            (want.n_bit_errors, want.border_fail, want.n_erased),
            "{tag}: counts"
        );
        for i in 0..4 {
            let d = (got.corner_sigma_scale[i] - want.corner_sigma_scale[i]).abs() as f64;
            assert!(
                d <= 1e-4 * want.corner_sigma_scale[i] as f64,
                "{tag}: sigma_scale {i}"
            );
        }
    }
    let chord = |a: &[f32; 3], b: &[f32; 3]| {
        let (a, b) = (vec3(a), vec3(b));
        // chord length, not acos of a dot product (which is only good to ~1e-4 rad near 1)
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
            / (std::f64::consts::PI / h as f64)
    };
    for i in 0..4 {
        let r = chord(&got.ray[i], &want.ray[i]);
        worst.ray_px = worst.ray_px.max(r);
        assert!(r <= CORNER_TOL, "{tag}: ray {i} differs by {r} px_eq");
        if STRICT {
            let e = px_err(
                got.corner_px[i][0] as f64,
                got.corner_px[i][1] as f64,
                want.corner_px[i][0] as f64,
                want.corner_px[i][1] as f64,
                w,
            );
            worst.corner = worst.corner.max(e);
            assert!(e <= CORNER_TOL, "{tag}: corner {i} differs by {e} px");
        }
    }
    let ce = chord(&got.center_ray, &want.center_ray);
    worst.center = worst.center.max(ce);
    assert!(ce <= CORNER_TOL, "{tag}: centre differs by {ce} px_eq");
    if STRICT {
        let e = px_err(
            got.center_px[0] as f64,
            got.center_px[1] as f64,
            want.center_px[0] as f64,
            want.center_px[1] as f64,
            w,
        );
        assert!(e <= CORNER_TOL, "{tag}: centre px differs by {e}");
    }
    let k = if STRICT { 1.0 } else { 10.0 }; // the loose gate is 10x the strict one
    let scal = [
        ("quality", got.quality, want.quality, 1e-3 * k),
        ("side_px", got.side_px, want.side_px, 0.01 * k),
        (
            "edge_resid_px",
            got.edge_resid_px,
            want.edge_resid_px,
            0.01 * k,
        ),
        ("contrast_dn", got.contrast_dn, want.contrast_dn, 0.5 * k),
        ("min_cross", got.min_cross, want.min_cross, 1e-3 * k),
    ];
    for (n, a, b, tol) in scal {
        let d = (a - b).abs() as f64;
        worst.scalar = worst.scalar.max(d / tol);
        assert!(d <= tol, "{tag}: {n} {a} vs {b}");
    }
}

fn run_case(case: &Value, scenes: &[Value], worst: &mut Worst) -> usize {
    let name = case.get("name").str();
    let (w, h) = (case.get("w").i64() as usize, case.get("h").i64() as usize);
    let mut det = Detector::new(w, h, &params_of(case.get("params"))).unwrap();
    let steps = case.get("steps").arr();
    let mut cache: Option<(usize, Vec<u8>)> = None;
    let mut checked: HashSet<usize> = HashSet::new();
    let mut n_markers = 0;
    for (si, st) in steps.iter().enumerate() {
        let tag = format!("{name} step {si}");
        let op = st.get("op").str();
        if op == "reset" {
            det.reset_track();
            continue;
        }
        let sidx = st.get("scene").i64() as usize;
        if cache.as_ref().map(|c| c.0) != Some(sidx) {
            let px = build_scene(&scenes[sidx]);
            if checked.insert(sidx) {
                assert_eq!(
                    sha256_hex(&px),
                    scenes[sidx].get("sha256").str(),
                    "{tag}: scene {sidx} differs from the reference"
                );
            }
            cache = Some((sidx, px));
        }
        let img = &cache.as_ref().unwrap().1;
        let iv = ImageView::packed(img, w, h, PixelFormat::Gray8);
        let res = match op {
            "detect" => det.detect(&iv),
            "track" => {
                let pi = st.get("prev").i64();
                let prev: Vec<Marker> = if pi < 0 {
                    vec![]
                } else {
                    steps[pi as usize]
                        .get("markers")
                        .arr()
                        .iter()
                        .map(parse_marker)
                        .collect()
                };
                det.track(&iv, &prev, st.get("gap").i64() as u32)
            }
            "verify" => {
                let pri: Vec<Prior> = st.get("priors").arr().iter().map(parse_prior).collect();
                det.verify(&iv, &pri)
            }
            o => panic!("unknown op {o}"),
        };
        let ret = st.get("ret").i64();
        if ret < 0 {
            assert_eq!(res.err(), Some(error_of(ret)), "{tag}: error");
            continue;
        }
        let got = res.unwrap_or_else(|e| panic!("{tag}: {e}"));
        let want: Vec<Marker> = st.get("markers").arr().iter().map(parse_marker).collect();
        assert_eq!(
            got.len(),
            ret as usize,
            "{tag}: marker count (ids {:?} vs {:?})",
            got.iter().map(|m| m.id).collect::<Vec<_>>(),
            want.iter().map(|m| m.id).collect::<Vec<_>>()
        );
        for (k, (g, e)) in got.iter().zip(&want).enumerate() {
            check_marker(
                g,
                e,
                w as i32,
                h as i32,
                &format!("{tag} marker {k}"),
                worst,
            );
        }
        n_markers += got.len();
        if STRICT {
            let s = det.last_stats();
            let v = st.get("stats").f64s();
            let have = [
                s.n_candidates,
                s.n_bridge_ok,
                s.n_decode_ok,
                s.n_accepted,
                s.n_rejected,
                s.n_budget_dropped,
                s.best_reject_code,
            ];
            for (i, (a, b)) in have.iter().zip(&v).enumerate() {
                assert_eq!(*a as f64, *b, "{tag}: stats[{i}]");
            }
        }
    }
    n_markers
}

#[test]
fn pipeline_matches_the_reference() {
    let fx = load_fixture("b5_pipeline.json");
    let scenes = fx.get("scenes").arr();
    let cases = fx.get("cases").arr();
    let next = AtomicUsize::new(0);
    let total = Mutex::new((0usize, Worst::default()));
    let threads = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .min(8);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut worst = Worst::default();
                let mut n = 0;
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cases.len() {
                        break;
                    }
                    n += run_case(&cases[i], scenes, &mut worst);
                }
                let mut t = total.lock().unwrap();
                t.0 += n;
                t.1.corner = t.1.corner.max(worst.corner);
                t.1.center = t.1.center.max(worst.center);
                t.1.ray_px = t.1.ray_px.max(worst.ray_px);
                t.1.scalar = t.1.scalar.max(worst.scalar);
            });
        }
    });
    let t = total.lock().unwrap();
    println!(
        "b5: {} cases, {} markers compared; worst corner {:.2e} px, centre {:.2e} px, ray {:.2e} px_eq, scalar {:.2} of tolerance",
        cases.len(),
        t.0,
        t.1.corner,
        t.1.center,
        t.1.ray_px,
        t.1.scalar
    );
    assert!(t.0 > 200);
}
