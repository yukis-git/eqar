//! Behaviour tests for the proposal stage (numeric expectations are those of the original
//! C++ unit tests, unchanged).
#![allow(clippy::needless_range_loop)]

use eqar::geometry::{px_to_ray, Vec3};
use eqar::image::{Fmt, View};
use eqar::proposal::*;
use eqar_testkit::synth::render_test_marker_id;
use std::f64::consts::PI;

/// A 64x32 plane with a dark 8x8 square at (28,12) and a bright block at the seam.
fn make_plane(w: usize, h: usize) -> Vec<u8> {
    let mut p = vec![128u8; w * h];
    for y in 12..20 {
        for x in 28..36 {
            p[y * w + x] = 20;
        }
    }
    // bright block straddling the seam: last 3 columns and first 3 columns
    for y in 4..8 {
        for x in w - 3..w {
            p[y * w + x] = 250;
        }
        for x in 0..3 {
            p[y * w + x] = 250;
        }
    }
    p
}

fn grid_for(w: usize, h: usize) -> TileGrid {
    TileGrid::new(w / K_TILE, h / K_TILE)
}

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

#[test]
fn tile_sums_are_exact() {
    let (w, h) = (64, 32);
    let p = make_plane(w, h);
    let mut g = grid_for(w, h);
    tile_sums(&p, w, h, w, &mut g);
    // tile (7,3) covers x 28..31, y 12..15 -> all 20
    assert_eq!(g.sum[3 * g.tw + 7], 20 * 16);
    // tile (0,0) is uniform 128
    assert_eq!(g.sum[0], 128 * 16);
}

#[test]
fn box_mean_wraps_in_x() {
    let (w, h) = (64, 32);
    let mut p = vec![100u8; w * h];
    for y in 0..h {
        p[y * w + (w - 1)] = 200; // last column bright
    }
    let mut g = grid_for(w, h);
    tile_sums(&p, w, h, w, &mut g);
    tile_box(&mut g);
    // tile column 0 is within K_TILE_RADIUS of tile column tw-1, so the bright column
    // must raise BOTH ends identically -- proof the box wrapped.
    assert_eq!(g.box_sum[0], g.box_sum[g.tw - 1]);
    assert!(g.box_sum[0] > 100 * g.boxdiv);
}

#[test]
fn box_mean_clamps_in_y() {
    let (w, h) = (64, 32);
    let mut p = vec![100u8; w * h];
    for x in 0..w {
        p[x] = 200; // first row bright
    }
    let mut g = grid_for(w, h);
    tile_sums(&p, w, h, w, &mut g);
    tile_box(&mut g);
    // bottom tile row must NOT see the top row's brightness (no y wrap)
    assert_eq!(g.box_sum[(g.th - 1) * g.tw + 5], 100 * g.boxdiv);
}

#[test]
fn two_offsets_differ() {
    let (w, h) = (64, 32);
    let mut p = make_plane(w, h);
    // a shallow dark patch: 10 DN below the local mean -- visible at C=7, not at C=14.
    // (Its tile row is vertically CLAMPED, so its local mean is ~123.63; 114 is 10 DN
    // below it.)
    for y in 24..28 {
        for x in 44..52 {
            p[y * w + x] = 114;
        }
    }
    let mut g = grid_for(w, h);
    tile_sums(&p, w, h, w, &mut g);
    tile_box(&mut g);
    let mut a = vec![0u8; w * h];
    let mut c = vec![0u8; w * h];
    threshold_from_tiles(&p, w, h, w, &mut g, K_THRESH_C, &mut a, w);
    threshold_from_tiles(&p, w, h, w, &mut g, K_THRESH_C2, &mut c, w);
    assert_eq!(a[26 * w + 48], 255);
    assert_eq!(c[26 * w + 48], 0);
    // the deep square is foreground at BOTH offsets
    assert_eq!(a[16 * w + 32], 255);
    assert_eq!(c[16 * w + 32], 255);
}

/// The vertical box window is CLAMPED (not wrapped) at the poles, and must not be biased
/// by replicating the edge tile row. Tile row 0's window (R=4) covers only tile rows 0..4
/// (5 rows, not 9) here.
#[test]
fn box_mean_is_unbiased_at_the_top_edge() {
    let (w, h) = (64, 32);
    let mut p = vec![100u8; w * h];
    for y in 0..K_TILE {
        for x in 0..w {
            p[y * w + x] = 200; // tile row 0 (pixel rows 0..3) all 200
        }
    }
    let mut g = grid_for(w, h);
    tile_sums(&p, w, h, w, &mut g);
    tile_box(&mut g);
    // Unbiased: mean over the 5 covered tile rows only == (200 + 4*100)/5 == 120.
    let mean = g.box_sum[0] as f64 / g.boxdiv as f64;
    assert!(near(mean, 120.0, 1e-6), "mean {mean}");
    // NOT the replicated-edge (biased) value (5*200 + 4*100)/9 ~= 155.56.
    let biased = (5.0 * 200.0 + 4.0 * 100.0) / 9.0;
    assert!(mean < biased - 5.0);
}

/// tw = 17 (68/4) is not a multiple of 4, exercising tile_sums' NEON tail.
#[test]
fn tile_sums_neon_tail_matches_scalar() {
    let (w, h) = (68, 32);
    let mut seed = 12345u32;
    let p: Vec<u8> = (0..w * h)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) as u8
        })
        .collect();
    let (tw, th) = (w / K_TILE, h / K_TILE);
    let mut g = TileGrid::new(tw, th);
    tile_sums(&p, w, h, w, &mut g);
    for ty in 0..th {
        for tx in 0..tw {
            let mut expect = 0u32;
            for j in 0..K_TILE {
                for i in 0..K_TILE {
                    expect += p[(ty * K_TILE + j) * w + tx * K_TILE + i] as u32;
                }
            }
            assert_eq!(g.sum[ty * tw + tx], expect);
        }
    }
}

fn ccl_of(bin: &[u8], w: usize, h: usize) -> (RunCcl, i32) {
    let mut r = RunCcl::default();
    let n = run_ccl_wrap(bin, w, h, w, &mut r);
    (r, n)
}

/// The last root label, or -1.
fn single_root(r: &mut RunCcl) -> i32 {
    let mut root = -1;
    for i in 1..r.blobs.len() {
        if r.find(i as i32) == i as i32 {
            root = i as i32;
        }
    }
    root
}

#[test]
fn ccl_merges_one_blob_across_the_seam() {
    let (w, h) = (64, 32);
    let mut bin = vec![0u8; w * h];
    // an 8x6 foreground rectangle straddling the seam: x in {60..63, 0..3}
    for y in 10..16 {
        for x in 60..64 {
            bin[y * w + x] = 255;
        }
        for x in 0..4 {
            bin[y * w + x] = 255;
        }
    }
    let (mut r, n) = ccl_of(&bin, w, h);
    assert_eq!(n, 1); // ONE component, not two
    let root = single_root(&mut r);
    assert!(root > 0);
    let b = r.blobs[root as usize];
    assert_eq!(b.area, 8 * 6);
    assert_eq!(b.touches_seam, 1);
    assert_eq!(b.xmax - b.xmin + 1, 8); // unwrapped width is 8, not 64
    assert_eq!(b.ymax - b.ymin + 1, 6);
}

#[test]
fn ccl_keeps_separate_blobs_separate() {
    let (w, h) = (64, 32);
    let mut bin = vec![0u8; w * h];
    for y in 4..8 {
        for x in 8..12 {
            bin[y * w + x] = 255;
        }
    }
    for y in 20..24 {
        for x in 40..44 {
            bin[y * w + x] = 255;
        }
    }
    assert_eq!(ccl_of(&bin, w, h).1, 2);
}

#[test]
fn ccl_extremes_use_the_1032_traversal() {
    // A 12x12 axis-aligned square: shoelace over {0,1,3,2} gives a positive,
    // non-degenerate area; over {0,1,2,3} it self-intersects and collapses.
    let (w, h) = (64, 32);
    let mut bin = vec![0u8; w * h];
    for y in 8..20 {
        for x in 20..32 {
            bin[y * w + x] = 255;
        }
    }
    let (mut r, _) = ccl_of(&bin, w, h);
    let root = single_root(&mut r);
    let b = r.blobs[root as usize];
    let ord = [0usize, 1, 3, 2];
    let mut ax = 0i64;
    for t in 0..4 {
        let (a, u) = (ord[t], ord[(t + 1) & 3]);
        ax += b.extx[a] as i64 * b.exty[u] as i64 - b.extx[u] as i64 * b.exty[a] as i64;
    }
    let qa = ax.abs() / 2;
    assert!(qa >= 100); // ~144 for a 12x12 square
    assert!(b.area as f64 / qa as f64 > 0.8);
}

/// A 24x24 blob: solid ring, 8x8 hole in the middle.
fn draw_ring(bin: &mut [u8], w: usize, x0: usize, y0: usize) {
    for y in 0..24 {
        for x in 0..24 {
            let hole = (8..16).contains(&x) && (8..16).contains(&y);
            bin[(y0 + y) * w + (x0 + x)] = if hole { 0 } else { 255 };
        }
    }
}

#[test]
fn shape_accepts_a_ring_and_sets_has_hole() {
    let (w, h) = (128, 64);
    let mut bin = vec![0u8; w * h];
    draw_ring(&mut bin, w, 40, 20);
    let (mut r, _) = ccl_of(&bin, w, h);
    let root = single_root(&mut r);
    let cfg = ShapeCfg::default();
    let mut sc = ShapeScratch::default();
    assert!(shape_ok(w, h, &mut r, root, &cfg, &mut sc, None));
    assert_eq!(r.blobs[root as usize].has_hole, 1);
}

/// The same 24x24 ring, but drawn straddling the +/-180 seam (columns 116..127, then
/// wrapping to 0..11) so the mask, hole test and blob_quad all go through the unwrap.
#[test]
fn shape_handles_a_seam_straddling_ring() {
    let (w, h) = (128, 64);
    let mut bin = vec![0u8; w * h];
    let (x0, y0) = (116usize, 20usize);
    for y in 0..24 {
        for x in 0..24 {
            let hole = (8..16).contains(&x) && (8..16).contains(&y);
            bin[(y0 + y) * w + (x0 + x) % w] = if hole { 0 } else { 255 };
        }
    }
    let (mut r, n) = ccl_of(&bin, w, h);
    assert_eq!(n, 1); // ONE component
    let root = single_root(&mut r);
    assert!(root > 0);
    let b = r.blobs[root as usize];
    assert_eq!(b.xmax - b.xmin + 1, 24); // unwrapped, not ~128
    let cfg = ShapeCfg::default();
    let mut sc = ShapeScratch::default();
    assert!(shape_ok(w, h, &mut r, root, &cfg, &mut sc, None));
    let b = r.blobs[root as usize];
    assert_eq!(b.has_hole, 1);

    let q = blob_quad(&b);
    let xmin = q.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let xmax = q.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
    assert!(near((xmax - xmin) as f64, 23.0, 2.0)); // extent ~24, not ~128
}

#[test]
fn shape_rejects_a_thin_line() {
    let (w, h) = (128, 64);
    let mut bin = vec![0u8; w * h];
    for x in 10..90 {
        for y in 30..33 {
            bin[y * w + x] = 255;
        }
    }
    let (mut r, _) = ccl_of(&bin, w, h);
    let root = single_root(&mut r);
    let cfg = ShapeCfg::default();
    let mut sc = ShapeScratch::default();
    assert!(!shape_ok(w, h, &mut r, root, &cfg, &mut sc, None)); // aspect 26:1
}

#[test]
fn shape_accepts_a_solid_square_without_a_hole() {
    // The background-sensitivity path: the ring got opened, so no hole survives, but the
    // blob is still square and solid -> it must remain a candidate.
    let (w, h) = (128, 64);
    let mut bin = vec![0u8; w * h];
    for y in 20..44 {
        for x in 40..64 {
            bin[y * w + x] = 255;
        }
    }
    let (mut r, _) = ccl_of(&bin, w, h);
    let root = single_root(&mut r);
    let cfg = ShapeCfg::default();
    let mut sc = ShapeScratch::default();
    assert_eq!(r.blobs[root as usize].has_hole, 0);
    assert!(shape_ok(w, h, &mut r, root, &cfg, &mut sc, None));
}

/// A large marker at proposal resolution is a THIN black ring around big white
/// data-module holes -- exactly this shape. The RAW (unfilled) pixel/hull ratio is well
/// under solidity_min, so only counting the enclosed holes as filled area recovers it.
fn draw_thick_ring(bin: &mut [u8], w: usize, x0: usize, y0: usize, outer: usize, hole: usize) {
    let pad = (outer - hole) / 2;
    for y in 0..outer {
        for x in 0..outer {
            let is_hole = (pad..pad + hole).contains(&x) && (pad..pad + hole).contains(&y);
            bin[(y0 + y) * w + (x0 + x)] = if is_hole { 0 } else { 255 };
        }
    }
}

#[test]
fn shape_accepts_a_large_thin_ring() {
    let (w, h) = (256, 128);
    let mut bin = vec![0u8; w * h];
    let (outer, hole) = (44usize, 30usize); // ring ~7 px thick
    draw_thick_ring(&mut bin, w, 40, 20, outer, hole);
    let (mut r, _) = ccl_of(&bin, w, h);
    let root = single_root(&mut r);
    assert!(root > 0);
    assert_eq!(
        r.blobs[root as usize].area,
        (outer * outer - hole * hole) as i32
    );

    // The raw (unfilled) pixel/hull ratio the OLD formula used: hull is the full outer
    // square (~outer^2, modulo hull_area's "N pixels covers N-1 points" shortfall).
    let raw_hull_est = ((outer - 1) * (outer - 1)) as f64;
    let raw_ratio = r.blobs[root as usize].area as f64 / raw_hull_est;
    assert!(raw_ratio < 0.80);

    let cfg = ShapeCfg::default();
    let mut sc = ShapeScratch::default();
    assert!(shape_ok(w, h, &mut r, root, &cfg, &mut sc, None));
    assert_eq!(r.blobs[root as usize].has_hole, 1);

    // The FILLED-area solidity shape_ok actually thresholded against must be >= 0.9.
    let filled = blob_filled_solidity(w, &r, root, cfg.hole_frac, &mut sc);
    assert!(filled >= 0.9, "filled solidity {filled}");
}

#[test]
fn shape_relaxed_thresholds() {
    let cfg = ShapeCfg::default();
    assert!(near(cfg.solidity_min as f64, 0.80, 1e-6)); // was 0.85
    assert!(near(cfg.extent_min as f64, 0.05, 1e-6)); // was 0.25
}

/// A bright (220) square with a dark (40) ring drawn 8 px in from its own border, leaving
/// a bright (220) untouched centre; the ring's own four corners stand in for a blob_quad's
/// corner rays, so the probe's edge-midpoint step lands squarely on the ring.
#[test]
fn dark_ring_probe_separates_ring_from_flat() {
    let (ww, hh) = (512usize, 256usize);
    let mut px = vec![220u8; ww * hh]; // bright quiet zone
    let (x0, y0, x1, y1, t) = (232usize, 104usize, 280usize, 152usize, 8usize);
    for y in y0..y1 {
        for x in x0..x1 {
            let on_ring = x < x0 + t || x >= x1 - t || y < y0 + t || y >= y1 - t;
            if on_ring {
                px[y * ww + x] = 40; // dark ring
            }
        }
    }
    let v = gray_view(&px, ww, hh);
    let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let mut ray_quad = [[0f32; 3]; 4];
    for (i, &(cx, cy)) in corners.iter().enumerate() {
        let ray = px_to_ray(cx as f64, cy as f64, ww as i32, hh as i32);
        ray_quad[i] = [ray.x as f32, ray.y as f32, ray.z as f32];
    }
    assert!(dark_ring_ok(&v, &ray_quad, 12.0));

    let flat = vec![128u8; ww * hh];
    let vf = gray_view(&flat, ww, hh);
    assert!(!dark_ring_ok(&vf, &ray_quad, 12.0));
}

#[test]
fn cap_geometry_round_trips() {
    let c = PolarCap::new(1920, 4, 1.0);
    // half-FOV 35 deg: 2*tan(35)*(1920/pi)/4 = 213.9, rounded UP to the next whole tile
    assert_eq!(c.r, 216);
    assert_eq!(c.r % K_TILE, 0);
    // The angular sampling density must still match the f=4 band's (H/pi/f px per
    // radian): never coarser, and at most one tile's worth finer.
    let band_px_per_rad = (1920.0 / PI) / 4.0;
    let cap_px_per_rad = c.r as f64 / (2.0 * c.tan_t);
    assert!(cap_px_per_rad >= band_px_per_rad);
    assert!(cap_px_per_rad <= band_px_per_rad * 1.05);
    // centre pixel of the north cap is the north pole
    let r = cap_to_ray(&c, (c.r - 1) as f64 * 0.5, (c.r - 1) as f64 * 0.5);
    assert!(near(r.y, 1.0, 1e-9));
    // a corner is at most the half-FOV diagonal away, and still in the north half
    let k = cap_to_ray(&c, 0.0, 0.0);
    assert!(k.y > 0.0);
    let ang = k.y.acos() * 180.0 / PI;
    assert!(ang > 35.0 && ang < 48.0); // 35 deg edge -> ~44.9 deg corner
                                       // The handover margin: the band alone carries every pose to |lat| 75 and none at 80,
                                       // so a cap edge at 35 deg from the pole (|lat| 55) leaves 20 deg of overlap.
    let edge = cap_to_ray(&c, (c.r - 1) as f64 * 0.5, 0.0);
    let edge_lat = edge.y.asin() * 180.0 / PI;
    assert!(edge_lat < 80.0 - 20.0);
}

#[test]
fn cap_is_continuous_through_the_pole() {
    // A full-res view whose value depends ONLY on latitude, so a correct cap is radially
    // symmetric. A latitude-wrapping sampler would mirror it.
    let (ww, hh) = (256usize, 128usize);
    let mut px = vec![0u8; ww * hh];
    for y in 0..hh {
        for x in 0..ww {
            px[y * ww + x] = (40 + 2 * y) as u8;
        }
    }
    let v = gray_view(&px, ww, hh);
    let mut c = PolarCap::new(hh as i32, 4, 1.0);
    cap_render(&v, &mut c);
    // R is EVEN here, so the raster has no centre pixel -- the true centre sits at the
    // half-pixel coordinate m + 0.5 with m = R/2 - 1. Compare the actual mirror pixels
    // m-d and m+1+d about that centre.
    let r = c.r;
    let m = r / 2 - 1;
    let d = 3;
    let at = |y: usize, x: usize| c.img[y * r + x] as i32;
    let c0 = at(m, m);
    let n = at(m - d, m);
    let s = at(m + 1 + d, m);
    let e = at(m, m + 1 + d);
    let w2 = at(m, m - d);
    assert!((n - s).abs() <= 2);
    assert!((e - w2).abs() <= 2);
    assert!((n - e).abs() <= 2);
    assert!(c0 <= n + 2);
    // The pole (row 0 of the source) is the darkest value in this latitude-only image,
    // and the centre pixel sits closest to the pole of the five sampled here.
    assert!(c0 <= n && c0 <= s && c0 <= e && c0 <= w2);
}

#[test]
fn destretch_shrinks_a_high_latitude_radius() {
    // Same blob pixel size at lat 0 and lat 80: the radius must agree to ~5 %.
    let _p = Proposal::new(3840, 1920).unwrap();
    assert!(destretch_radius(1920, 4, 40.0, 40.0, 0.0) > 0.0);
    let r_eq = destretch_radius(1920, 4, 40.0, 40.0, 0.0);
    let r_hi = destretch_radius(
        1920,
        4,
        40.0 / (80.0f64 * PI / 180.0).cos(),
        40.0,
        (80.0f64 * PI / 180.0).sin(),
    );
    assert!(near(r_hi, r_eq, 0.05 * r_eq));
}

#[test]
fn finds_a_synthetic_dark_square_at_the_seam() {
    // 3840x1920 gray 160 with a 96x96 dark square straddling x = 0.
    let (ww, hh) = (3840usize, 1920usize);
    let mut px = vec![160u8; ww * hh];
    for y in 900..996 {
        for x in ww - 48..ww {
            px[y * ww + x] = 30;
        }
        for x in 0..48 {
            px[y * ww + x] = 30;
        }
    }
    // punch a white hole so the has-hole test also fires
    for y in 930..966 {
        for x in ww - 18..ww {
            px[y * ww + x] = 220;
        }
        for x in 0..18 {
            px[y * ww + x] = 220;
        }
    }
    let v = gray_view(&px, ww, hh);
    let mut p = Proposal::new(ww as i32, hh as i32).unwrap();
    let mut c = Vec::new();
    p.run(&v, &mut c).unwrap();
    assert!(!c.is_empty());
    // one of the candidates must sit within 1 deg of lon 180, lat ~ 3.7 deg
    let hit = c.iter().any(|k| {
        let lon = k.v.z.atan2(k.v.x) * 180.0 / PI;
        let dl = (lon.abs() - 180.0).abs();
        dl < 1.0 && k.v.y.abs() < 0.1
    });
    assert!(hit);
}

fn mk(lon_deg: f64, lat_deg: f64, r: f64) -> Candidate {
    let (lo, la) = (lon_deg * PI / 180.0, lat_deg * PI / 180.0);
    Candidate {
        v: Vec3::new(la.cos() * lo.cos(), la.sin(), la.cos() * lo.sin()),
        r,
        has_quad: false,
        from_cap: false,
        scale: 4,
        ..Default::default()
    }
}

#[test]
fn dedupe_merges_a_true_duplicate() {
    // the same blob seen through the wrap pad and through the band: identical
    let mut c = vec![mk(10.0, 22.0, 0.08), mk(10.05, 22.0, 0.081)];
    dedupe_candidates(&mut c);
    assert_eq!(c.len(), 1);
}

#[test]
fn dedupe_keeps_touching_neighbours() {
    // ids 6 and 7, half-angle 4.5 deg, lat 22, quiet zones touching.
    let (half, lat) = (4.5f64, 22.0f64);
    let r = half * PI / 180.0;
    for gap in [1.03, 1.10, 1.20, 1.40] {
        let dlon = 2.0 * half * gap / (lat * PI / 180.0).cos();
        let mut c = vec![mk(10.0, lat, r), mk(10.0 + dlon, lat, r)];
        dedupe_candidates(&mut c);
        assert_eq!(c.len(), 2); // BOTH must survive, at every gap
    }
}

#[test]
fn dedupe_keeps_a_nested_ring_and_its_marker() {
    let mut c = vec![mk(0.0, 0.0, 0.05), mk(0.0, 0.0, 0.05 * 1.45)];
    dedupe_candidates(&mut c);
    assert_eq!(c.len(), 2); // co-centred but 45 % larger -> keep both
}

#[test]
fn dedupe_is_linear_in_candidate_count() {
    let mut c = Vec::new();
    for i in 0..3000 {
        c.push(mk(
            -180.0 + 360.0 * (i % 60) as f64 / 60.0,
            -80.0 + 160.0 * (i / 60) as f64 / 50.0,
            0.01,
        ));
    }
    let before = c.len();
    dedupe_candidates(&mut c); // must return quickly and keep the grid
    assert!(c.len() > before * 9 / 10);
}

#[test]
fn budget_order_is_smallest_first() {
    let mut c = vec![mk(0.0, 0.0, 0.30), mk(20.0, 0.0, 0.02), mk(40.0, 0.0, 0.11)];
    order_for_budget(&mut c);
    assert!(c[0].r < c[1].r);
    assert!(c[1].r < c[2].r);
}

/// The span short-circuit must never change a verdict. A "C" (a square open on one side)
/// has middle rows spanning only the left arm while its convex hull is nearly the whole
/// bbox, so span/hull lands well under solidity_min and the hole scan is skipped. This C
/// is hole-less, so the hole-optional path IS what would accept it: solidity is the only
/// thing that may stop it.
#[test]
fn shape_rejects_a_c_shape_the_span_bound_dooms() {
    let (w, h) = (256, 128);
    let (x0, y0, side, th) = (40usize, 20usize, 40usize, 8usize);
    let mut bin = vec![0u8; w * h];
    // A "C": full top and bottom bars, plus a left arm joining them.
    for y in 0..side {
        for x in 0..side {
            let bar = y < th || y >= side - th;
            if bar || x < th {
                bin[(y0 + y) * w + (x0 + x)] = 255;
            }
        }
    }
    let (mut r, _) = ccl_of(&bin, w, h);
    let root = single_root(&mut r);
    assert!(root > 0);
    assert_eq!(
        r.blobs[root as usize].area,
        (2 * th * side + (side - 2 * th) * th) as i32
    );

    let cfg = ShapeCfg::default();
    let mut sc = ShapeScratch::default();
    // Hole-less, and its filled area over the hull is under solidity_min -- so the only
    // correct verdict is reject, short-circuit or not.
    let sol = blob_filled_solidity(w, &r, root, cfg.hole_frac, &mut sc);
    assert!(sol < cfg.solidity_min);
    assert!(!shape_ok(w, h, &mut r, root, &cfg, &mut sc, None));
    assert_eq!(r.blobs[root as usize].has_hole, 0);

    // ...and it really does clear the extent gate that precedes solidity.
    let extent = r.blobs[root as usize].area as f32 / (side * side) as f32;
    assert!(extent >= cfg.extent_min);
}

/// Block-mask run extraction vs a scalar reference: runs that start and end exactly on a
/// 16-px block boundary, runs that straddle one, single-pixel runs at either end of the
/// row, a wholly foreground row, and a ragged final block. Patterns sit on EVEN rows only
/// so that no two are 8-connected, which makes each row's blob areas sum to exactly that
/// row's own foreground count.
#[test]
fn run_extraction_matches_a_scalar_reference() {
    let (w, h) = (61usize, 28usize); // 61 = 3 full blocks + a 13 px tail
    let mut bin = vec![0u8; w * h];
    let mut set = |row: usize, a: usize, e: usize| {
        for x in a..=e {
            bin[2 * row * w + x] = 255;
        }
    };
    set(0, 0, 15); // exactly one block
    set(1, 16, 31); // exactly the second block
    set(2, 15, 16); // straddles the 16 boundary
    set(3, 0, 0); // single pixel at column 0
    set(4, 60, 60); // single pixel at the last column
    set(5, 0, 60); // the whole row
    set(6, 48, 60); // the ragged tail block only
    set(7, 47, 48); // straddles into the tail block
    set(8, 1, 1);
    set(8, 3, 3);
    set(8, 5, 5); // isolated single pixels
    set(9, 31, 32); // straddles the 32 boundary
    set(10, 0, 14);
    set(10, 16, 30); // two runs split by one background px
    set(11, 55, 59); // inside the tail, not touching the end
    set(12, 14, 17);
    set(12, 19, 22); // two runs, both crossing a boundary
                     // row 26 stays empty

    // Scalar reference, a byte at a time: foreground pixels per row AND the number of runs
    // per row. Two runs on the SAME row are never merged, so runs-per-row is exactly
    // blobs-per-row -- which catches an extraction that SPLITS a run at a block boundary.
    let mut ref_row = [0i64; 28];
    let mut ref_runs = [0i32; 28];
    let mut ref_total = 0i64;
    for y in 0..h {
        for x in 0..w {
            if bin[y * w + x] != 0 {
                ref_row[y] += 1;
                ref_total += 1;
                if x == 0 || bin[y * w + x - 1] == 0 {
                    ref_runs[y] += 1;
                }
            }
        }
    }

    let (mut r, _) = ccl_of(&bin, w, h);
    let mut got_row = [0i64; 28];
    let mut got_runs = [0i32; 28];
    let mut got_total = 0i64;
    for i in 1..r.blobs.len() {
        if r.find(i as i32) != i as i32 {
            continue;
        }
        let b = r.blobs[i];
        assert_eq!(b.ymin, b.ymax); // no pattern spans two rows
        got_row[b.ymin as usize] += b.area as i64;
        got_runs[b.ymin as usize] += 1;
        got_total += b.area as i64;
    }
    for y in 0..h {
        assert_eq!(got_row[y], ref_row[y], "row {y}");
        assert_eq!(got_runs[y], ref_runs[y], "row {y}");
    }
    assert_eq!(got_total, ref_total);
}

struct XorShift(u32);
impl XorShift {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// shape_ok must never ACCEPT a blob whose exact filled solidity is below solidity_min,
/// over a broad shape population (blocks, rings, C/E/U shapes, ragged fragments).
/// blob_filled_solidity deliberately bypasses the short-circuit, so it is an independent
/// oracle for the exact ratio.
#[test]
fn shape_never_accepts_below_the_exact_solidity_floor() {
    let (w, h) = (192usize, 160usize);
    let cfg = ShapeCfg::default();
    let mut rng = XorShift(0x9E3779B9);
    let (mut accepted, mut rejected) = (0, 0);
    for _trial in 0..400 {
        let mut bin = vec![0u8; w * h];
        let bw = 24 + (rng.next() % 60) as usize;
        let bh = 24 + (rng.next() % 60) as usize;
        let x0 = 8 + (rng.next() % (w - bw - 16) as u32) as usize;
        let y0 = 8 + (rng.next() % (h - bh - 16) as u32) as usize;
        for y in 0..bh {
            for x in 0..bw {
                bin[(y0 + y) * w + x0 + x] = 255;
            }
        }
        let bites = rng.next() % 4;
        for _ in 0..bites {
            let cw = 3 + (rng.next() % (bw - 2) as u32) as usize;
            let ch = 3 + (rng.next() % (bh - 2) as u32) as usize;
            let cx = (rng.next() % (bw - cw + 1) as u32) as usize;
            let cy = (rng.next() % (bh - ch + 1) as u32) as usize;
            for y in 0..ch {
                for x in 0..cw {
                    bin[(y0 + cy + y) * w + x0 + cx + x] = 0;
                }
            }
        }

        let (mut r, _) = ccl_of(&bin, w, h);
        let mut sc = ShapeScratch::default();
        for i in 1..r.blobs.len() {
            if r.find(i as i32) != i as i32 {
                continue;
            }
            if !shape_ok(w, h, &mut r, i as i32, &cfg, &mut sc, None) {
                rejected += 1;
                continue;
            }
            accepted += 1;
            let exact = blob_filled_solidity(w, &r, i as i32, cfg.hole_frac, &mut sc);
            assert!(exact >= cfg.solidity_min);
        }
    }
    // The population has to actually exercise both verdicts.
    assert!(accepted > 20, "accepted {accepted}");
    assert!(rejected > 20, "rejected {rejected}");
}

/// The retained runs ARE the blob: for every root, the union of its runs is exactly the
/// foreground pixels of its own 8-connected component, and nothing else. Checked over a
/// random population that includes seam-straddling and touching-but-separate shapes.
#[test]
fn retained_runs_are_exactly_each_blobs_own_pixels() {
    let (w, h) = (192usize, 96usize);
    let mut rng = XorShift(0xC0FFEE);
    let (mut seam_blobs, mut total_roots) = (0, 0);

    for _trial in 0..60 {
        let mut bin = vec![0u8; w * h];
        for _ in 0..12 {
            let bw = 3 + (rng.next() % 30) as usize;
            let bh = 3 + (rng.next() % 20) as usize;
            let x0 = (rng.next() % w as u32) as usize; // may straddle the seam
            let y0 = (rng.next() % (h - bh) as u32) as usize;
            for y in 0..bh {
                for x in 0..bw {
                    bin[(y0 + y) * w + (x0 + x) % w] = 255;
                }
            }
        }
        let (mut r, nroots) = ccl_of(&bin, w, h);
        total_roots += nroots;

        let mut owner = vec![-1i32; w * h];
        let mut roots_seen = 0;
        for i in 1..r.blobs.len() {
            if r.find(i as i32) != i as i32 {
                continue;
            }
            roots_seen += 1;
            let b = r.blobs[i];
            if b.touches_seam != 0 {
                seam_blobs += 1;
            }
            let bw = b.xmax - b.xmin + 1;
            let mut px = 0i64;
            for k in r.run_first(i as i32)..r.run_last(i as i32) {
                let rn = r.runs[r.run_index[k as usize] as usize];
                assert!(rn.ce >= rn.cs);
                assert!(rn.y >= b.ymin && rn.y <= b.ymax);
                // the blob-local mapping the shape filter paints the mask with
                let mut lx = (rn.cs as i64 - b.xmin as i64) % w as i64;
                if lx < 0 {
                    lx += w as i64;
                }
                assert!(lx + ((rn.ce - rn.cs) as i64) < bw as i64);
                for c in rn.cs..=rn.ce {
                    let p = rn.y as usize * w + c as usize;
                    assert!(bin[p] != 0); // every run pixel is foreground
                    assert_eq!(owner[p], -1); // and claimed by exactly one root
                    owner[p] = i as i32;
                }
                px += (rn.ce - rn.cs + 1) as i64;
            }
            assert_eq!(px as i32, b.area); // run lengths sum to the CCL's area
        }
        assert_eq!(roots_seen, nroots);
        // ...and no foreground pixel is left unclaimed.
        for p in 0..bin.len() {
            if bin[p] != 0 {
                assert!(owner[p] >= 0);
            }
        }
    }
    assert!(total_roots > 100); // the population is not vacuous
    assert!(seam_blobs > 0); // and it really did exercise the seam
}

/// The second threshold offset's pass is a FALLBACK: it runs only when the primary offset
/// produced NO tier-1 candidate. Both branches are pinned, so "skip it always" would fail.
#[test]
fn second_offset_pass_runs_only_without_a_tier1_candidate() {
    let (ww, hh) = (3840usize, 1920usize);
    let mut p = Proposal::new(ww as i32, hh as i32).unwrap();
    let mut c = Vec::new();

    // 1. A marker the primary offset resolves at reference-grade shape (holed, solid, dark
    //    ring): a tier-1 candidate exists, so the second offset's whole pass is skipped.
    {
        let mut px = vec![205u8; ww * hh];
        render_test_marker_id(&mut px, ww as i32, hh as i32, -100.0, -8.0, 5.0, 13.0, 0);
        let v = gray_view(&px, ww, hh);
        c.clear();
        p.run(&v, &mut c).unwrap();
        assert!(p.dbg().n_tier1 >= 1);
        assert!(!p.dbg().ran_2c);
        assert_eq!(p.dbg().n_roots_2c, 0);
        assert_eq!(p.dbg().n_cand_2c, 0);
    }

    // 2. Foreground the primary offset finds, but nothing reference-strict: a hole-less
    //    dark square is tier 2 at best, so the fallback still runs and labels its blobs.
    {
        let mut px = vec![205u8; ww * hh];
        for y in 900..996 {
            for x in 1800..1896 {
                px[y * ww + x] = 30;
            }
        }
        let v = gray_view(&px, ww, hh);
        c.clear();
        p.run(&v, &mut c).unwrap();
        assert_eq!(p.dbg().n_tier1, 0);
        assert!(p.dbg().ran_2c);
        assert!(p.dbg().n_roots_2c > 0);
    }
}

#[test]
fn proposal_rejects_a_mismatched_view() {
    let mut p = Proposal::new(960, 480).unwrap();
    let px = vec![128u8; 480 * 240];
    let v = gray_view(&px, 480, 240);
    assert!(p.run(&v, &mut Vec::new()).is_err());
    assert!(Proposal::new(8, 4).is_err());
}
