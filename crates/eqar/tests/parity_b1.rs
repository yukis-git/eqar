//! Parity against the C++ reference: fixtures in tests/fixtures/b1_*.json.

use eqar::decode::dict_info;
use eqar::geometry::*;
use eqar::image::{downsample, sample, Fmt, View};
use eqar_testkit::json::Value;
use eqar_testkit::load_fixture;
use eqar_testkit::sha256::sha256_hex;
use eqar_testkit::synth::{test_marker_center_ray, test_marker_corner_rays, SynthCase};

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 24) as u8
    }
}

/// Rebuild a fixture image: returns (bytes, w, h, stride).
fn build_image(v: &Value) -> (Vec<u8>, i32, i32, usize) {
    let (w, h, stride) = (
        v.get("w").i64() as usize,
        v.get("h").i64() as usize,
        v.get("stride").i64() as usize,
    );
    let b = match v.get("kind").str() {
        "ramp" => (0..w * h)
            .map(|i| (((i % w) * 7 + (i / w) * 13) & 0xFF) as u8)
            .collect(),
        "lcg" => {
            let mut r = Lcg(v.get("seed").i64() as u32);
            (0..stride * h).map(|_| r.byte()).collect()
        }
        k => panic!("unknown image kind {k}"),
    };
    (b, w as i32, h as i32, stride)
}

fn fmt_of(i: i64) -> Fmt {
    match i {
        0 => Fmt::Gray8,
        1 => Fmt::Bgr8,
        _ => Fmt::Rgb8,
    }
}

#[test]
fn dictionaries_match_the_cpp_tables() {
    let fx = load_fixture("b1_dict.json");
    let dicts = fx.get("dicts").arr();
    assert_eq!(dicts.len(), 16);
    let mut total = 0;
    for d in dicts {
        let di = dict_info(d.get("index").i64() as i32).unwrap();
        assert_eq!(di.name, d.get("name").str());
        assert_eq!(di.n as i64, d.get("n").i64());
        assert_eq!(di.marker_size as i64, d.get("marker_size").i64());
        assert_eq!(di.max_correction as i64, d.get("max_correction").i64());
        assert_eq!(di.codes.len(), di.n as usize * 4);
        let le: Vec<u8> = di.codes.iter().flat_map(|c| c.to_le_bytes()).collect();
        assert_eq!(sha256_hex(&le), d.get("sha256_le_u64").str(), "{}", di.name);
        let first: Vec<u64> = d.get("first").f64s().iter().map(|&x| x as u64).collect();
        assert_eq!(&di.codes[..4], &first[..]);
        total += di.codes.len();
    }
    assert_eq!(total, 22400);
    assert!(dict_info(16).is_err() && dict_info(-1).is_err());
}

#[test]
fn synth_images_are_byte_identical_to_cpp() {
    let fx = load_fixture("b1_synth.json");
    let cases = fx.get("cases").arr();
    assert!(cases.len() >= 30);
    for c in cases {
        let case = SynthCase::from_json(c);
        let img = case.render();
        assert_eq!(
            sha256_hex(&img),
            c.get("sha256").str(),
            "synth case {}",
            c.get("name").str()
        );
    }
}

#[test]
fn sampler_matches_cpp_bit_for_bit() {
    let fx = load_fixture("b1_image.json");
    let lut_v = fx.get("lut").f64s();
    let mut lut = [0u8; 256];
    for (l, v) in lut.iter_mut().zip(&lut_v) {
        *l = *v as u8;
    }
    let (mut n, mut max_ulp) = (0usize, 0i64);
    for c in fx.get("sample").arr() {
        let (b, w, h, stride) = build_image(c.get("image"));
        let v = View {
            data: &b,
            w,
            h,
            stride,
            fmt: fmt_of(c.get("fmt").i64()),
            lut: if matches!(c.get("lut"), Value::Bool(true)) {
                Some(&lut)
            } else {
                None
            },
        };
        let (xs, ys, os) = (
            c.get("x_bits").f64s(),
            c.get("y_bits").f64s(),
            c.get("out_bits").f64s(),
        );
        for i in 0..xs.len() {
            let got = sample(
                &v,
                f32::from_bits(xs[i] as u32),
                f32::from_bits(ys[i] as u32),
            );
            let want = f32::from_bits(os[i] as u32);
            let ulp = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            max_ulp = max_ulp.max(ulp);
            n += 1;
            assert!(
                ulp <= SAMPLE_MAX_ULP,
                "{} #{i}: {got} vs {want} ({ulp} ulp)",
                c.get("name").str()
            );
        }
    }
    eprintln!("sampler: {n} values, max {max_ulp} ulp");
}

/// aarch64: the sampler fuses like the C++ build and matches bit for bit. Elsewhere (and
/// with `no-neon`) the blends are not fused and round differently; allow a few f32 ulps.
const SAMPLE_MAX_ULP: i64 = if cfg!(all(target_arch = "aarch64", not(feature = "no-neon"))) {
    0
} else {
    8
};

#[test]
fn downsample_matches_cpp_exactly() {
    let fx = load_fixture("b1_image.json");
    for c in fx.get("downsample").arr() {
        let (b, w, h, stride) = build_image(c.get("image"));
        let f = c.get("f").i64() as usize;
        let v = View {
            data: &b,
            w,
            h,
            stride,
            fmt: fmt_of(c.get("fmt").i64()),
            lut: None,
        };
        let (ow, oh) = (w as usize / f, h as usize / f);
        let mut out = vec![0u8; ow * oh];
        downsample(&v, f, &mut out, ow);
        assert_eq!(
            sha256_hex(&out),
            c.get("sha256").str(),
            "{}",
            c.get("name").str()
        );
    }
}

/// Absolute below 1, relative above (acos near 1 amplifies 1-ulp dot differences).
const GEO_TOL: f64 = 1e-12;

fn close(name: &str, i: usize, got: &[f64], want: &[f64], worst: &mut f64) {
    assert_eq!(got.len(), want.len(), "{name} #{i}");
    for (g, w) in got.iter().zip(want) {
        let d = (g - w).abs();
        *worst = worst.max(d / w.abs().max(1.0));
        assert!(
            d <= GEO_TOL * w.abs().max(1.0),
            "{name} #{i}: {g:.17e} vs {w:.17e} (|d|={d:e})"
        );
    }
}

fn v3(a: &[f64]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}
fn m3(a: &[f64]) -> Mat3 {
    Mat3 {
        m: a[..9].try_into().unwrap(),
    }
}
fn flat(vs: &[Vec3]) -> Vec<f64> {
    vs.iter().flat_map(|v| [v.x, v.y, v.z]).collect()
}

#[test]
fn geometry_matches_cpp() {
    let fx = load_fixture("b1_geometry.json");
    let mut worst = 0.0f64;
    let mut count = 0usize;
    let mut run = |name: &str, f: &dyn Fn(&[f64]) -> Vec<f64>| {
        for (i, c) in fx.get(name).arr().iter().enumerate() {
            close(
                name,
                i,
                &f(&c.get("in").f64s()),
                &c.get("out").f64s(),
                &mut worst,
            );
            count += 1;
        }
    };
    run("px_to_ray", &|a| {
        flat(&[px_to_ray(a[0], a[1], a[2] as i32, a[3] as i32)])
    });
    run("ray_to_px", &|a| {
        let (x, y) = ray_to_px(&v3(a), a[3] as i32, a[4] as i32);
        vec![x, y]
    });
    run("px_err", &|a| {
        vec![px_err(a[0], a[1], a[2], a[3], a[4] as i32)]
    });
    run("ang_err", &|a| {
        vec![ang_err_px_eq(&v3(&a[0..3]), &v3(&a[3..6]), a[6] as i32)]
    });
    run("eigvec3", &|a| {
        flat(&[smallest_eigvec3(a.try_into().unwrap())])
    });
    run("accum_ppt", &|a| {
        let n = a[0] as usize;
        let p: Vec<Vec3> = a[1..1 + 3 * n].chunks(3).map(v3).collect();
        accum_ppt(&p, Some(&a[1 + 3 * n..1 + 4 * n])).to_vec()
    });
    run("slerp", &|a| {
        flat(&[slerp(&v3(&a[0..3]), &v3(&a[3..6]), a[6])])
    });
    run("basis_at", &|a| {
        let (e1, e2) = basis_at(&v3(a));
        flat(&[e1, e2])
    });
    run("kabsch", &|a| {
        let from: Vec<Vec3> = a[0..12].chunks(3).map(v3).collect();
        let to: Vec<Vec3> = a[12..24].chunks(3).map(v3).collect();
        kabsch(&from, &to).m.to_vec()
    });
    run("rot_power", &|a| rot_power(&m3(a), a[9]).m.to_vec());
    run("marker_truth", &|a| {
        let mut o = flat(&test_marker_corner_rays(a[0], a[1], a[2], a[3]));
        o.extend(flat(&[test_marker_center_ray(a[0], a[1], a[3])]));
        o
    });
    eprintln!("geometry: {count} cases, worst scaled |diff| = {worst:e}");
}
