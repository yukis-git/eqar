//! Parity of the decode, refine and guards stages against the C++ reference:
//! tests/fixtures/b4_bits.json (bit-level identify / verify), b4_decode.json (cell reading,
//! cold and verify decode, border probes), b4_refine.json (profile stage, refine_rays,
//! corner states) and b4_guards.json (gates, side consistency, ring width).
//!
//! Every image is rebuilt from its description (synth markers, rectangle occluders, an
//! illumination ramp, LCG noise; checked by sha256). Call inputs (ray quads, seeds) are
//! stored in the fixtures, so the stages do not depend on each other.
//!
//! Gates. On aarch64 with NEON the sampling kernels are the same float32 code as the C++
//! ones (same polynomial atan2, same gathers, fused multiply-adds in the same places): the
//! float profiles of the refiner are sha256-identical, every integer decision (bits, ids,
//! reject codes, inlier counts, flags) matches exactly, and corner rays agree to double
//! rounding (the C++ build can contract `a*b+c` in the double geometry, so not bit-exact).
//! Elsewhere (x86_64, or `no-neon`) the same samples are taken by the scalar
//! double-precision path, which differs from the float kernel by up to ~1 DN on a few
//! samples: a cell mean on the edge of an erasure band, or an inlier at the Tukey bound, can
//! flip. There ids / reject codes must match on all but a small allowance of records and
//! the corners agree to 0.02 px_eq; the profile hashes are not compared.

use eqar::codes;
use eqar::decode::*;
use eqar::geometry::{basis_at, norm, normalize, Vec3, K_PI};
use eqar::guards::*;
use eqar::image::{Fmt, View};
use eqar::params::{cold_decode_config, verify_decode_config};
use eqar::refine::*;
use eqar_testkit::json::Value;
use eqar_testkit::load_fixture;
use eqar_testkit::sha256::sha256_hex;
use eqar_testkit::synth::{test_marker_corner_rays, SynthCase};

const STRICT: bool = cfg!(all(target_arch = "aarch64", not(feature = "no-neon")));

/// Corner agreement, px_eq (angle between the rays / (pi / H)).
const CORNER_TOL_PX: f64 = if STRICT { 1e-4 } else { 0.02 };
/// Records (decode calls, refine runs) whose integer outcome may differ on the loose gate.
const LOOSE_MISMATCHES: usize = 8;
const IDS_0_9: [i32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

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

fn rays4(v: &Value) -> [Vec3; 4] {
    let f = v.f64s();
    assert_eq!(f.len(), 12);
    let mut r = [Vec3::default(); 4];
    for i in 0..4 {
        r[i] = Vec3::new(f[3 * i], f[3 * i + 1], f[3 * i + 2]);
    }
    r
}

/// Largest corner distance in px_eq. The chord length is used (not acos of the dot product,
/// which is only good to ~1e-8 rad near 1).
fn max_corner_px(a: &[Vec3; 4], b: &[Vec3; 4], h: i32) -> f64 {
    let pix = K_PI / h as f64;
    (0..4)
        .map(|i| {
            let d = Vec3::new(a[i].x - b[i].x, a[i].y - b[i].y, a[i].z - b[i].z);
            norm(&d) / pix
        })
        .fold(0.0, f64::max)
}

fn f32_of(v: &Value) -> f32 {
    f32::from_bits(v.i64() as u32)
}

fn scene_markers(c: &Value) -> Vec<[Vec3; 4]> {
    c.get("markers")
        .arr()
        .iter()
        .map(|m| {
            test_marker_corner_rays(
                m.get("lon").f64(),
                m.get("lat").f64(),
                m.get("half").f64(),
                m.get("roll").f64(),
            )
        })
        .collect()
}

// ---- bit level ----------------------------------------------------------------------------

#[test]
fn bit_level_identify_and_verify_match_the_cpp_reference() {
    let fx = load_fixture("b4_bits.json");
    let mut n = 0;
    for d in fx.get("cases").arr() {
        let dict = d.get("dict").i64() as i32;
        for row in d.get("rows").arr() {
            let x = row.get("x").i64() as u64;
            let er = row.get("er").i64() as u64;
            let at = format!("dict {dict} x {x:#x}");
            let want = |v: &Value| -> Option<(i32, i32)> {
                match v {
                    Value::Null => None,
                    v => {
                        let a = v.arr();
                        Some((a[0].i64() as i32, a[1].i64() as i32))
                    }
                }
            };
            assert_eq!(
                dict_identify(dict, x),
                want(row.get("id")),
                "{at}: identify"
            );
            let cold = cold_decode_config(Some(&IDS_0_9));
            assert_eq!(
                dict_identify_in_use(dict, Some(&cold), x),
                want(row.get("in_use")),
                "{at}: in-use"
            );
            let ex = row.get("ex").i64() as i32;
            let erot = row.get("erot").i64() as i32;
            for (vc, want_v) in row.get("v").arr().iter().enumerate() {
                let mut cfg = verify_decode_config(if vc == 2 { None } else { Some(&IDS_0_9) });
                if vc == 1 {
                    cfg.other_margin = 2;
                    cfg.max_mismatch = 0;
                    cfg.min_known_frac = 0.3;
                }
                let got = verify_cells(dict, &cfg, x, er, ex, erot);
                let want = match want_v {
                    Value::Null => None,
                    v => {
                        let a = v.arr();
                        Some(VerifyOk {
                            rot: a[0].i64() as i32,
                            n_mismatch: a[1].i64() as i32,
                            n_erased: a[2].i64() as i32,
                        })
                    }
                };
                assert_eq!(got, want, "{at}: verify cfg {vc}");
                n += 1;
            }
        }
    }
    eprintln!("b4 bits: {n} verify_cells calls");
}

// ---- decode -------------------------------------------------------------------------------

fn dec_fields(d: &DecodeResult) -> [i64; 9] {
    [
        d.id as i64,
        d.rot as i64,
        d.n_bit_errors as i64,
        d.n_erased as i64,
        d.border_fail as i64,
        d.border_ok as i64,
        d.bits as i64,
        d.contrast.to_bits() as i64,
        d.reject_code as i64,
    ]
}

/// Compares a decode result with a stored record; returns false on a (loose-gate) mismatch.
fn check_decode(at: &str, got: &DecodeResult, rec: &Value, h: i32, mism: &mut usize) -> bool {
    let (r, corners) = match rec {
        Value::Arr(_) => (rec, None),
        o => (o.get("r"), Some(o.get("c"))),
    };
    let want: Vec<i64> = r.arr().iter().map(|x| x.i64()).collect();
    let mine = dec_fields(got);
    if STRICT {
        assert_eq!(mine.to_vec(), want, "{at}: decode record");
    } else {
        // reject code and id decide; the rest only when those agree
        if mine[0] != want[0] || mine[8] != want[8] {
            *mism += 1;
            return false;
        }
        let dc = (got.contrast - f32::from_bits(want[7] as u32)).abs();
        assert!(dc <= 1.5, "{at}: contrast differs by {dc}");
    }
    if let Some(c) = corners {
        if got.id >= 0 {
            let d = max_corner_px(&got.corners, &rays4(c), h);
            assert!(d <= CORNER_TOL_PX, "{at}: corners differ by {d} px_eq");
        }
    }
    true
}

#[test]
fn decode_matches_the_cpp_reference() {
    let fx = load_fixture("b4_decode.json");
    let (mut n, mut n_acc, mut n_err, mut n_ver, mut mism) = (0, 0, 0, 0, 0usize);
    let mut worst_corner = 0.0f64;
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
        let cold = cold_decode_config(Some(&IDS_0_9));
        let vcfg = verify_decode_config(Some(&IDS_0_9));
        for (mi, mark) in case.get("marks").arr().iter().enumerate() {
            for var in mark.arr() {
                let at = format!("{name} marker {mi} {}", var.get("v").str());
                let rays = rays4(var.get("rays"));
                let c8 = decode_rays(&v, 0, &rays, 8, None);
                n += 1;
                if check_decode(&format!("{at} c8"), &c8, var.get("c8"), h, &mut mism) {
                    if let Value::Obj(_) = var.get("c8") {
                        worst_corner = worst_corner.max(max_corner_px(
                            &c8.corners,
                            &rays4(var.get("c8").get("c")),
                            h,
                        ));
                    }
                    n_acc += (c8.id >= 0) as i32;
                    n_err += (c8.n_bit_errors > 0) as i32;
                }
                if var.has("c4") {
                    let c4 = decode_rays(&v, 0, &rays, 4, None);
                    check_decode(&format!("{at} c4"), &c4, var.get("c4"), h, &mut mism);
                }
                let cs = decode_rays(&v, 0, &rays, 8, Some(&cold));
                check_decode(&format!("{at} cs"), &cs, var.get("cs"), h, &mut mism);
                // the in-use cold decode is the legacy one with ids outside the set dropped
                if c8.id >= 0 && c8.id < 10 {
                    assert_eq!(cs.id, c8.id);
                } else {
                    assert!(cs.id < 0);
                }
                let tid = var.get("tid").i64() as i32;
                let vt = decode_verify(&v, 0, &rays, 8, tid, -1, &vcfg);
                if check_decode(&format!("{at} vt"), &vt, var.get("vt"), h, &mut mism) {
                    n_ver += (vt.id >= 0) as i32;
                    if vt.id >= 0 {
                        // on success id == expected_id, corners are the winding-fixed rays rolled by rot
                        assert_eq!(vt.id, tid);
                        if c8.id >= 0 && c8.rot == vt.rot {
                            assert!(max_corner_px(&vt.corners, &c8.corners, h) <= 1e-9);
                        }
                    }
                }
                let wid = var.get("wid").i64() as i32;
                let vw = decode_verify(&v, 0, &rays, 8, wid, -1, &vcfg);
                check_decode(&format!("{at} vw"), &vw, var.get("vw"), h, &mut mism);
                // the reference never accepts a wrong id
                if wid != tid {
                    assert!(
                        var.get("vw").arr()[0].i64() < 0,
                        "{at}: fixture accepts a wrong id"
                    );
                }
                let mut con = 0.0f32;
                let bdf = border_dark_fraction(&v, 0, &rays, 2, Some(&mut con));
                let msd = border_min_side_dark(&v, 0, &rays, 2);
                let (wb, wc) = (
                    f32_of(&var.get("bdf").arr()[0]),
                    f32_of(&var.get("bdf").arr()[1]),
                );
                let wm = f32_of(var.get("msd"));
                if STRICT {
                    assert_eq!(bdf.to_bits(), wb.to_bits(), "{at}: border_dark_fraction");
                    assert_eq!(con.to_bits(), wc.to_bits(), "{at}: probe contrast");
                    assert_eq!(msd.to_bits(), wm.to_bits(), "{at}: border_min_side_dark");
                } else {
                    assert!((bdf - wb).abs() <= 0.1, "{at}: bdf {bdf} vs {wb}");
                    assert!((con - wc).abs() <= 1.5, "{at}: probe contrast");
                    assert!((msd - wm).abs() <= 0.2, "{at}: msd {msd} vs {wm}");
                }
            }
        }
    }
    eprintln!(
        "b4 decode: {n} variants, {n_acc} cold accepts, {n_err} with bit errors, {n_ver} verify accepts, \
         {mism} loose mismatches, worst corner diff {worst_corner:e} px_eq"
    );
    assert!(n >= 120 && n_acc >= 60 && n_ver >= 40 && n_err >= 1);
    assert!(mism <= LOOSE_MISMATCHES);
}

// ---- refine -------------------------------------------------------------------------------

fn seed_from(gt: &[Vec3; 4], off: f64, rot: bool, h: i32) -> [Vec3; 4] {
    let pix = K_PI / h as f64;
    let mut s = [Vec3::default(); 4];
    for i in 0..4 {
        let (e1, e2) = basis_at(&gt[i]);
        let ang = if rot {
            i as f64 * K_PI / 2.0 + 0.7
        } else {
            0.0
        };
        let (a, b) = (off * pix * ang.cos(), off * pix * ang.sin());
        s[i] = normalize(&Vec3::new(
            gt[i].x + e1.x * a + e2.x * b,
            gt[i].y + e1.y * a + e2.y * b,
            gt[i].z + e1.z * a + e2.z * b,
        ));
    }
    s
}

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

fn cfg_of(name: &str) -> RefineCfg {
    match name {
        "track" => K_TRACK_REFINE,
        "known" => RefineCfg {
            allow_weak: true,
            ..K_SCAN_REFINE
        },
        _ => K_SCAN_REFINE,
    }
}

fn cs_fields(cs: &[CornerState; 4]) -> Vec<i64> {
    cs.iter()
        .flat_map(|c| {
            [
                c.valid as i64,
                c.flags as i64,
                c.sigma_scale.to_bits() as i64,
            ]
        })
        .collect()
}

#[test]
fn refine_matches_the_cpp_reference() {
    let fx = load_fixture("b4_refine.json");
    let (mut n_runs, mut n_ok, mut n_fail, mut n_stage, mut n_weak, mut mism) =
        (0, 0, 0, 0, 0, 0usize);
    let mut worst = 0.0f64;
    let mut scratch = RefineScratch::default();
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
        let gts = scene_markers(case);
        for (mi, mark) in case.get("marks").arr().iter().enumerate() {
            let gt = &gts[mi];
            let seed0 = rays4(mark.get("seed0"));
            // ---- profile stage
            if let st @ Value::Obj(_) = mark.get("stage") {
                let at = format!("{name} marker {mi} stage");
                let side = st.get("side").f64();
                let k = st.get("k").i64() as usize;
                let hw = st.get("hw").f64();
                assert_eq!(refine_k(side) as usize, k, "{at}: K");
                assert!(rel_close(hw_initial(side), hw, 1e-12), "{at}: hw0");
                let s = {
                    let sum = Vec3::new(
                        seed0.iter().map(|r| r.x).sum(),
                        seed0.iter().map(|r| r.y).sum(),
                        seed0.iter().map(|r| r.z).sum(),
                    );
                    normalize(&sum)
                };
                for (e, rec) in st.get("edges").arr().iter().enumerate() {
                    let r = rec.arr();
                    let mut prof = vec![0f32; k * ((2.0 * hw / K_STEP) as usize + 3)];
                    let (mut pk, mut perp) = (vec![Vec3::default(); k], vec![Vec3::default(); k]);
                    let m = sample_edge(
                        &v,
                        &seed0[e],
                        &seed0[(e + 1) & 3],
                        &s,
                        hw,
                        k,
                        &mut prof,
                        &mut pk,
                        &mut perp,
                        &mut scratch.arc,
                    )
                    .expect("edge");
                    assert_eq!(m as i64, r[0].i64(), "{at} edge {e}: M");
                    let prof = &prof[..k * m];
                    if STRICT {
                        let bytes: Vec<u8> = prof.iter().flat_map(|f| f.to_le_bytes()).collect();
                        assert_eq!(sha256_hex(&bytes), r[1].str(), "{at} edge {e}: profile");
                    }
                    let sig = edge_sigma(prof, k, m, K_STEP, 3);
                    let mut ts = vec![0f64; k];
                    op_centroid(prof, k, m, K_STEP, &mut ts);
                    let sum: f64 = ts.iter().sum();
                    let mn = ts.iter().cloned().fold(f64::INFINITY, f64::min);
                    let mx = ts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let (tol_sig, tol_ts) = if STRICT { (1e-12, 1e-9) } else { (0.2, 0.5) };
                    assert!((sig - r[2].f64()).abs() <= tol_sig, "{at} edge {e}: sigma");
                    assert!(
                        (sum - r[3].f64()).abs() <= tol_ts * k as f64,
                        "{at} edge {e}: sum"
                    );
                    assert!((mn - r[4].f64()).abs() <= tol_ts, "{at} edge {e}: min");
                    assert!((mx - r[5].f64()).abs() <= tol_ts, "{at} edge {e}: max");
                }
                n_stage += 1;
            }
            // ---- refine_rays runs
            for run in mark.get("runs").arr() {
                let cfgname = run.get("cfg").str();
                let off = run.get("off").f64();
                let seed = if off == 0.25 {
                    seed0
                } else {
                    seed_from(gt, off, run.get("rot").i64() != 0, h)
                };
                let at = format!("{name} marker {mi} {cfgname} off {off}");
                let mut out = [Vec3::default(); 4];
                let mut info = RefineInfo::default();
                let ok = refine_rays(
                    &v,
                    &seed,
                    &mut out,
                    &mut info,
                    &cfg_of(cfgname),
                    &mut scratch,
                );
                n_runs += 1;
                let wi = run.get("info");
                let want_ok = run.get("ok").i64() != 0;
                let want_rej = wi.get("reject").i64() as i32;
                assert_eq!(ok, info.ok, "{at}: return value vs info.ok");
                if ok != want_ok || info.reject_code != want_rej {
                    mism += 1;
                    if STRICT {
                        panic!(
                            "{at}: ok {ok} reject {}, reference ok {want_ok} reject {want_rej}",
                            info.reject_code
                        );
                    }
                    continue;
                }
                let d = wi.get("d").f64s();
                // the doubles that exist on failure too
                assert!(rel_close(info.side_px, d[0], 1e-9), "{at}: side_px");
                if !ok {
                    n_fail += 1;
                    continue;
                }
                n_ok += 1;
                if !STRICT && want_rej != codes::ACCEPT {
                    continue;
                }
                assert!(rel_close(info.module_px, d[1], 1e-9), "{at}: module_px");
                assert!(rel_close(info.hw0, d[2], 1e-9), "{at}: hw0");
                let (tc, tr) = if STRICT { (1e-9, 1e-6) } else { (0.5, 0.05) };
                assert!(
                    (info.contrast_dn - d[3]).abs() <= tc,
                    "{at}: contrast {} vs {}",
                    info.contrast_dn,
                    d[3]
                );
                assert!(
                    (info.resid_px - d[4]).abs() <= tr,
                    "{at}: resid {} vs {}",
                    info.resid_px,
                    d[4]
                );
                assert!(
                    (info.min_cross - d[5]).abs() <= if STRICT { 1e-6 } else { 1e-2 },
                    "{at}: min_cross"
                );
                assert_eq!(info.k as i64, wi.get("k").i64(), "{at}: K");
                if STRICT {
                    assert_eq!(info.n_used as i64, wi.get("n_used").i64(), "{at}: n_used");
                }
                for (e, rec) in wi.get("edges").arr().iter().enumerate() {
                    let r = rec.arr();
                    let es = &info.edge[e];
                    if STRICT {
                        assert_eq!(es.n_inl as i64, r[0].i64(), "{at} edge {e}: n_inl");
                        assert_eq!(
                            es.support.to_bits() as i64,
                            r[1].i64(),
                            "{at} edge {e}: support"
                        );
                        assert_eq!(
                            es.end_support[0].to_bits() as i64,
                            r[2].i64(),
                            "{at} edge {e}: end0"
                        );
                        assert_eq!(
                            es.end_support[1].to_bits() as i64,
                            r[3].i64(),
                            "{at} edge {e}: end1"
                        );
                        assert!(
                            (es.resid_px - f32_of(&r[4])).abs() <= 1e-5,
                            "{at} edge {e}: resid"
                        );
                        assert_eq!(es.weak as i64, r[5].i64(), "{at} edge {e}: weak");
                    } else {
                        assert!(
                            (es.support - f32_of(&r[1])).abs() <= 0.1,
                            "{at} edge {e}: support"
                        );
                    }
                    n_weak += es.weak as i32;
                }
                let dcorner = max_corner_px(&out, &rays4(run.get("out")), h);
                worst = worst.max(dcorner);
                assert!(
                    dcorner <= CORNER_TOL_PX,
                    "{at}: corners differ by {dcorner} px_eq"
                );
                if STRICT {
                    assert_eq!(
                        cs_fields(&corner_states(&info, false)),
                        run.get("cs_cold")
                            .arr()
                            .iter()
                            .map(|x| x.i64())
                            .collect::<Vec<_>>(),
                        "{at}: corner states (cold)"
                    );
                    if run.has("cs_known") {
                        assert_eq!(
                            cs_fields(&corner_states(&info, true)),
                            run.get("cs_known")
                                .arr()
                                .iter()
                                .map(|x| x.i64())
                                .collect::<Vec<_>>(),
                            "{at}: corner states (known)"
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "b4 refine: {n_stage} profile stages, {n_runs} runs ({n_ok} ok, {n_fail} failed as in the reference, \
         {n_weak} weak edges), {mism} loose mismatches, worst corner diff {worst:e} px_eq"
    );
    assert!(n_runs >= 120 && n_ok >= 100 && n_fail >= 3 && n_weak >= 1);
    assert!(mism <= LOOSE_MISMATCHES);
}

// ---- guards -------------------------------------------------------------------------------

#[test]
fn guards_match_the_cpp_reference() {
    let fx = load_fixture("b4_guards.json");
    let scenes = load_fixture("b4_refine.json");

    // piecewise-linear sigma scale: bit-exact in STRICT; builds that do not fuse
    // multiply-adds round differently, so allow a few ulps there.
    for (i, w) in fx.get("sigma").arr().iter().enumerate() {
        let e = (i as i32 - 2) as f32 / 40.0;
        let got = corner_sigma_scale(e).to_bits() as i64;
        let ulp = (got - w.i64()).abs();
        assert!(
            ulp <= if STRICT { 0 } else { 4 },
            "sigma scale at {e}: {ulp} ulp"
        );
    }

    let gate =
        |init: &[Vec3; 4], fin: &[Vec3; 4], info: &RefineInfo, border_ok: bool, rej: bool| {
            let dec = DecodeResult {
                id: 0,
                rot: 0,
                n_bit_errors: 0,
                n_erased: 0,
                border_fail: 0,
                border_ok,
                contrast: 0.0,
                bits: 0,
                corners: *init,
                reject_code: if border_ok {
                    codes::ACCEPT
                } else {
                    codes::R_BORDER_NOT_BLACK
                },
            };
            apply_gates(&GateInput {
                init_rays: init,
                final_rays: fin,
                info,
                decode: &dec,
                reject_high_resid: rej,
            })
        };
    let check_gates = |at: &str, o: &[GateOutcome; 2], want: &Value| {
        for (rej, w) in want.arr().iter().enumerate() {
            let w = w.arr();
            assert_eq!(
                o[rej].code as i64,
                w[0].i64(),
                "{at}: gate code (reject_high_resid {rej})"
            );
            assert_eq!(o[rej].flags as i64, w[1].i64(), "{at}: gate flags");
            let wq = f32_of(&w[2]);
            assert!(
                (o[rej].quality - wq).abs() <= 1e-6,
                "{at}: quality {} vs {wq}",
                o[rej].quality
            );
        }
    };

    for h in fx.get("hand").arr() {
        let at = format!("hand {}", h.get("name").str());
        let (init, fin) = (rays4(h.get("init")), rays4(h.get("fin")));
        let iv = h.get("info").f64s();
        let info = RefineInfo {
            ok: true,
            side_px: iv[0],
            hw0: iv[1],
            contrast_dn: iv[2],
            resid_px: iv[3],
            min_cross: iv[4],
            ..RefineInfo::default()
        };
        let bo = h.get("border_ok").i64() != 0;
        assert!(
            (init_min_cross(&init) - h.get("imc").f64()).abs() <= 1e-12,
            "{at}: init_min_cross"
        );
        let o = [
            gate(&init, &fin, &info, bo, false),
            gate(&init, &fin, &info, bo, true),
        ];
        check_gates(&at, &o, h.get("gates"));
    }

    let (mut n_real, mut n_ring, mut n_disagree) = (0, 0, 0);
    let mut worst_ring = 0.0f32;
    let mut cache: Option<(String, Vec<u8>)> = None;
    for g in fx.get("real").arr() {
        let sname = g.get("scene").str();
        let at = format!("real {sname} marker {}", g.get("mark").i64());
        let sc = scenes
            .get("cases")
            .arr()
            .iter()
            .find(|c| c.get("name").str() == sname)
            .expect("scene");
        if cache.as_ref().map(|c| c.0 != sname).unwrap_or(true) {
            cache = Some((sname.to_string(), build_scene(sc)));
        }
        let px = &cache.as_ref().unwrap().1;
        let (w, hh) = (sc.get("w").i64() as i32, sc.get("h").i64() as i32);
        let v = gray_view(px, w, hh);
        let (init, fin) = (rays4(g.get("init")), rays4(g.get("fin")));
        let iv = g.get("info").f64s();
        let info = RefineInfo {
            ok: true,
            side_px: iv[0],
            hw0: iv[1],
            contrast_dn: iv[2],
            resid_px: iv[3],
            min_cross: iv[4],
            module_px: iv[5],
            ..RefineInfo::default()
        };
        assert!(
            (init_min_cross(&init) - g.get("imc").f64()).abs() <= 1e-9,
            "{at}: init_min_cross"
        );
        let bo = g.get("border_ok").i64() != 0;
        let o = [
            gate(&init, &fin, &info, bo, false),
            gate(&init, &fin, &info, bo, true),
        ];
        check_gates(&at, &o, g.get("gates"));
        let pix = K_PI / hh as f64;
        let mut agree = 0;
        for e in 0..4 {
            agree |= (side_agrees(
                &init,
                &fin,
                e,
                info.module_px,
                pix,
                K_CONSIST_ANGLE_RAD,
                K_CONSIST_OFFSET_CELL,
            ) as i64)
                << e;
        }
        assert_eq!(agree, g.get("agree").i64(), "{at}: side_agrees mask");
        // ring width on the decoded corners of the refined quad
        let dr = decode_rays(&v, 0, &fin, 8, None);
        assert_eq!(dr.id as i64, g.get("id").i64(), "{at}: decoded id");
        if let Value::Arr(rw) = g.get("ring") {
            for (s, w) in rw.iter().enumerate() {
                let got = ring_width(&v, 0, dr.id, &dr.corners, s, pix);
                let want = f32_of(w);
                let tol = if STRICT { 1e-4 } else { 0.1 };
                worst_ring = worst_ring.max((got - want).abs());
                assert!(
                    (got - want).abs() <= tol,
                    "{at}: ring width side {s}: {got} vs {want}"
                );
                n_ring += 1;
            }
        }
        n_disagree += (agree != 15) as i32;
        n_real += 1;
    }
    eprintln!(
        "b4 guards: {n_real} real cases ({n_ring} ring widths, worst diff {worst_ring:e}, {n_disagree} with a disagreeing side), {} hand-made",
        fx.get("hand").arr().len()
    );
    assert!(n_real >= 30 && n_ring >= 100);
}
