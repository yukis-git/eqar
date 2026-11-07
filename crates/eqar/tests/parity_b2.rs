//! Parity of the proposal stage against the C++ reference: tests/fixtures/b2_proposal.json.
//!
//! Every scene is rebuilt from its synth arguments (+ an optional LCG noise pass, checked
//! by sha256), run through `Proposal::run` and `run_small`, and compared candidate by
//! candidate: same count, same order, same tier / flags, values within tolerance.
//!
//! Tolerances. The integer pipeline (downsample, tile sums, threshold, CCL, shape
//! filter) is exact, so the stage counters, flags and the f32 shape measurements
//! (areas over a double hull area) must match exactly. The remaining values go through
//! libm trig on doubles (`px_to_ray`, `slerp`, `ray_to_px`) and an f32 bilinear sampler.
//! On aarch64 the Rust build fuses the f32 blends exactly where the C++ build
//! (`-ffp-contract=fast`) does and calls the same libm, so doubles agree to a few ulp
//! (the C++ build can also contract `a*b+c` inside the double geometry, hence not
//! bit-exact) and `ring_dn` to a few f32 ulp. Elsewhere (x86_64, or no fused multiply-add)
//! a blend can round differently, which moves `ring_dn` by up to ~1e-3 DN and a cap tap
//! weight by an ulp; the feature-gated tolerances below are looser for that case.

use eqar::geometry::Vec3;
use eqar::image::{Fmt, View};
use eqar::proposal::*;
use eqar_testkit::json::Value;
use eqar_testkit::load_fixture;
use eqar_testkit::sha256::sha256_hex;
use eqar_testkit::synth::SynthCase;

/// (relative tolerance on doubles, absolute tolerance on ring_dn in DN, f32 ulps on shape fields)
#[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
const TOL: (f64, f64, u32) = (1e-12, 1e-4, 0);
#[cfg(not(all(target_arch = "aarch64", not(feature = "no-neon"))))]
const TOL: (f64, f64, u32) = (1e-9, 2e-2, 2);

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
}

fn build_scene(c: &Value) -> Vec<u8> {
    let mut px = SynthCase::from_json(c).render();
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

fn bits(v: &Value) -> f32 {
    f32::from_bits(v.i64() as u32)
}

struct Row {
    c: Candidate,
}

fn parse_row(r: &Value) -> Row {
    let f = r.arr();
    let g = |i: usize| f[i].f64();
    let v3 = |i: usize| Vec3::new(g(i), g(i + 1), g(i + 2));
    Row {
        c: Candidate {
            v: v3(0),
            r: g(3),
            quad: [v3(4), v3(7), v3(10), v3(13)],
            has_quad: f[16].i64() != 0,
            from_cap: f[17].i64() != 0,
            scale: f[18].i64() as u8,
            tier: f[19].i64() as u8,
            solidity: bits(&f[20]),
            extent: bits(&f[21]),
            raw_extent: bits(&f[22]),
            ring_dn: bits(&f[23]),
        },
    }
}

#[derive(Default)]
struct Worst {
    geom: f64,
    ring: f64,
    n: usize,
}

fn close_rel(name: &str, got: f64, want: f64, worst: &mut f64) {
    let err = (got - want).abs() / want.abs().max(1.0);
    if err > *worst {
        *worst = err;
    }
    assert!(
        err <= TOL.0,
        "{name}: got {got:.17e} want {want:.17e} (scaled err {err:e})"
    );
}

fn ulps(a: f32, b: f32) -> u32 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
}

fn compare(label: &str, got: &[Candidate], want: &[Value], w: &mut Worst) {
    assert_eq!(got.len(), want.len(), "{label}: candidate count");
    for (i, (g, wv)) in got.iter().zip(want).enumerate() {
        let e = parse_row(wv).c;
        let at = format!("{label}[{i}]");
        assert_eq!(g.has_quad, e.has_quad, "{at} has_quad");
        assert_eq!(g.from_cap, e.from_cap, "{at} from_cap");
        assert_eq!(g.scale, e.scale, "{at} scale");
        assert_eq!(g.tier, e.tier, "{at} tier");
        for (nm, a, b) in [
            ("solidity", g.solidity, e.solidity),
            ("extent", g.extent, e.extent),
            ("raw_extent", g.raw_extent, e.raw_extent),
        ] {
            assert!(ulps(a, b) <= TOL.2, "{at} {nm}: got {a} want {b}");
        }
        let dr = (g.ring_dn as f64 - e.ring_dn as f64).abs();
        w.ring = w.ring.max(dr);
        assert!(
            dr <= TOL.1,
            "{at} ring_dn: got {} want {}",
            g.ring_dn,
            e.ring_dn
        );
        close_rel(&format!("{at} r"), g.r, e.r, &mut w.geom);
        for (nm, a, b) in [
            ("v", g.v, e.v),
            ("q0", g.quad[0], e.quad[0]),
            ("q1", g.quad[1], e.quad[1]),
            ("q2", g.quad[2], e.quad[2]),
            ("q3", g.quad[3], e.quad[3]),
        ] {
            close_rel(&format!("{at} {nm}.x"), a.x, b.x, &mut w.geom);
            close_rel(&format!("{at} {nm}.y"), a.y, b.y, &mut w.geom);
            close_rel(&format!("{at} {nm}.z"), a.z, b.z, &mut w.geom);
        }
        w.n += 1;
    }
}

#[test]
fn proposal_candidates_match_cpp() {
    let fx = load_fixture("b2_proposal.json");
    let cases = fx.get("cases").arr();
    assert!(cases.len() >= 40);
    let mut worst = Worst::default();
    let mut total = 0usize;
    for c in cases {
        let name = c.get("name").str().to_string();
        let px = build_scene(c);
        assert_eq!(
            sha256_hex(&px),
            c.get("sha256").str(),
            "{name}: image bytes"
        );
        let (w, h) = (c.get("w").i64() as i32, c.get("h").i64() as i32);
        let v = View {
            data: &px,
            w,
            h,
            stride: w as usize,
            fmt: Fmt::Gray8,
            lut: None,
        };
        let mut p = Proposal::new(w, h).unwrap();
        let (mut run_c, mut small_c) = (Vec::new(), Vec::new());
        p.run(&v, &mut run_c).unwrap();
        let d = *p.dbg();
        p.run_small(&v, &mut small_c).unwrap();
        let ds = *p.dbg();

        let k = c.get("counts");
        let ci = |key: &str| k.get(key).i64();
        assert_eq!(d.n_roots_c as i64, ci("n_roots_c"), "{name} n_roots_c");
        assert_eq!(d.n_roots_2c as i64, ci("n_roots_2c"), "{name} n_roots_2c");
        assert_eq!(
            d.n_roots_cap as i64,
            ci("n_roots_cap"),
            "{name} n_roots_cap"
        );
        assert_eq!(d.n_cand_c as i64, ci("n_cand_c"), "{name} n_cand_c");
        assert_eq!(d.n_cand_2c as i64, ci("n_cand_2c"), "{name} n_cand_2c");
        assert_eq!(d.n_cand_cap as i64, ci("n_cand_cap"), "{name} n_cand_cap");
        assert_eq!(d.n_tier1 as i64, ci("n_tier1"), "{name} n_tier1");
        assert_eq!(d.n_tier2 as i64, ci("n_tier2"), "{name} n_tier2");
        assert_eq!(
            d.n_shape_calls as i64,
            ci("n_shape_calls"),
            "{name} n_shape_calls"
        );
        assert_eq!(
            d.n_shape_floods as i64,
            ci("n_shape_floods"),
            "{name} n_shape_floods"
        );
        assert_eq!(
            d.n_shape_flood_px,
            ci("n_shape_flood_px"),
            "{name} n_shape_flood_px"
        );
        assert_eq!(
            d.n_ring_probes as i64,
            ci("n_ring_probes"),
            "{name} n_ring_probes"
        );
        assert_eq!(d.ran_2c as i64, ci("ran_2c"), "{name} ran_2c");
        assert_eq!(
            ds.n_roots_small as i64,
            ci("n_roots_small"),
            "{name} n_roots_small"
        );
        assert_eq!(
            ds.n_cand_small as i64,
            ci("n_cand_small"),
            "{name} n_cand_small"
        );

        compare(
            &format!("{name}/run"),
            &run_c,
            c.get("run").arr(),
            &mut worst,
        );
        compare(
            &format!("{name}/small"),
            &small_c,
            c.get("small").arr(),
            &mut worst,
        );

        let mut all = run_c.clone();
        all.extend_from_slice(&small_c);
        dedupe_candidates(&mut all);
        order_for_budget(&mut all);
        cap_candidates(&mut all, K_MAX_CANDIDATES_PER_FRAME);
        compare(
            &format!("{name}/ordered"),
            &all,
            c.get("ordered").arr(),
            &mut worst,
        );
        total += 1;
    }
    eprintln!(
        "b2 parity: {total} scenes, {} candidates compared, worst scaled geometry err {:e}, worst ring_dn err {:e} DN",
        worst.n, worst.geom, worst.ring
    );
}
