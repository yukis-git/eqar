//! Bridge: per-candidate gnomonic re-seed at full resolution.
//!
//! The proposal stage localises a marker to a few px mean / several px max corner error
//! at 1/4 (or 1/2) resolution, while the great-circle refiner's convergence basin is only
//! ~3 px. The bridge closes that gap: for each surviving candidate it renders a small
//! rectilinear (gnomonic) patch centred on the candidate ray at full resolution,
//! thresholds it, extracts the marker's 4-gon and polishes its sides by total-least-squares
//! line fits. In a gnomonic patch the marker is an exact quadrilateral for ANY pose (pole,
//! seam, high obliquity), so one code path works everywhere.

use crate::decode::{border_dark_fraction, border_min_side_dark};
use crate::error::EqarError;
use crate::geometry::{basis_at, dot, normalize, ray_to_px, Vec3, K_PI};
#[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
use crate::image::Fmt;
use crate::image::{sample, View};
use crate::refine::K_MIN_CONTRAST_DN;

/// Angular sampling rate relative to native (the scan plan's rate on both passes).
pub const K_BRIDGE_OVERSAMPLE: f64 = 1.6;
/// Pass 1 patch inflation.
pub const K_BRIDGE_BOX: f64 = 1.7;
/// Pass 2 patch inflation.
pub const K_BRIDGE_BOX2: f64 = 1.35;

/// Angular rate of the TRACK plan. A tracked prior is already within a pixel or two of the
/// answer and the bridge only has to land the refiner inside its basin, so track runs one
/// pass at a rate low enough to be nearly free (a generous box with a low rate beats a
/// tight box at equal cost: the box absorbs the prior's error, the rate only sets how
/// finely the contour handed to the refiner is sampled).
pub const K_BRIDGE_TRACK_OVERSAMPLE: f64 = 0.6;

// The scan plan keeps the reference's rate on both passes. Pass 1 only re-centres, but
// whatever it does to the centring changes which pixels pass 2's border follower walks,
// and on a knife-edge frame that moves the polished quad by ~0.1 px_eq; the response to
// the pass-1 rate is not monotone, so a lower rate would be a lucky sample, not a basin.

/// One entry per bridge pass: the patch inflation applied to the candidate's angular
/// radius, the angular oversampling rate, the hard cap on the rendered patch side, and
/// whether the pass only SCREENS (renders pass 1's framing small, tests the quad's border
/// ring and throws the quad away).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BridgePass {
    pub box_: f64,
    pub oversample: f64,
    pub max_side: i32,
    pub screen: bool,
}

pub const K_BRIDGE_MIN_PATCH: i32 = 40;
pub const K_BRIDGE_MAX_PATCH: i32 = 768;

// Capping the scan plan's patch side. The patch side is proportional to the candidate's
// ANGULAR size, so a 300 px_eq marker would render 768^2 twice where a 21 px_eq one renders
// 40^2 twice: a 370x spread for a stage whose only job is to land the refiner's seed inside
// a basin that is itself 0.10*side wide. Above side ~55 px_eq the capped cost is flat.
// The cap is set by reproduction fidelity, not accuracy: the refiner is bistable on some
// frames and which fixed point it settles into is decided by the bridge quad it was seeded
// with. 192 sits two sweep steps clear of the cliff on each pass.
pub const K_BRIDGE_SCAN_CAP1: i32 = 192;
pub const K_BRIDGE_SCAN_CAP2: i32 = 192;

// The in-bridge probe pass. A candidate that is not a marker would otherwise pay the full
// two-pass bridge and a refine before anything rejects it. The screen is a miniature of
// pass 1 (same centre, box and rate, capped to `K_BRIDGE_PROBE_PATCH`) whose quad is tested
// for an ArUco marker's black outer ring and then thrown away, so it can only reject, never
// perturb: the real passes re-frame from the caller's own centre and radius.
//  * 96 px, not 48: at high obliquity a marker lands on the patch foreshortened by cos(tilt)
//    and resolving its border ring needs about 83 px.
//  * the bar is the detector's pre-bridge border fraction (0.75): a polished quad of an
//    obliquely viewed marker reads 0.85, not 1.0.
//  * raw Otsu only: the flatten and adaptive modes cost more on the screen's own patch than
//    the passes they save, and a surviving candidate gets all three modes again.
pub const K_BRIDGE_PROBE_PATCH: i32 = 96;
pub const K_BRIDGE_PROBE_BORDER: f64 = 0.75;
/// `border_dark_fraction` sub-samples per cell axis.
pub const K_BRIDGE_PROBE_SUB: i32 = 2;

/// The scan plan: a screen plus the reference's two passes.
pub const K_SCAN_PLAN: [BridgePass; 3] = [
    BridgePass {
        box_: K_BRIDGE_BOX,
        oversample: K_BRIDGE_OVERSAMPLE,
        max_side: K_BRIDGE_PROBE_PATCH,
        screen: true,
    },
    BridgePass {
        box_: K_BRIDGE_BOX,
        oversample: K_BRIDGE_OVERSAMPLE,
        max_side: K_BRIDGE_SCAN_CAP1,
        screen: false,
    },
    BridgePass {
        box_: K_BRIDGE_BOX2,
        oversample: K_BRIDGE_OVERSAMPLE,
        max_side: K_BRIDGE_SCAN_CAP2,
        screen: false,
    },
];
pub const K_SCAN_PLAN_PASSES: usize = 3;

/// The track plan keeps a single cheap pass and no probe: a tracked prior is already known
/// to be a marker, so there is nothing for a screen to reject.
pub const K_TRACK_PLAN: [BridgePass; 1] = [BridgePass {
    box_: K_BRIDGE_BOX,
    oversample: K_BRIDGE_TRACK_OVERSAMPLE,
    max_side: K_BRIDGE_MAX_PATCH,
    screen: false,
}];

/// A rendered gnomonic patch. `img` and `flat` may be longer than `s * s`; only the first
/// `s * s` entries are meaningful.
#[derive(Debug, Clone, Default)]
pub struct Patch {
    /// Side, px.
    pub s: i32,
    /// tan(half_fov).
    pub tan_t: f64,
    /// Orthonormal frame, `e3` = centre ray.
    pub e1: Vec3,
    pub e2: Vec3,
    pub e3: Vec3,
    /// Gray patch.
    pub img: Vec<u8>,
    /// Illumination-flattened copy.
    pub flat: Vec<f32>,
    /// True when a per-pass `max_side` cut `s` BELOW the side this framing would have
    /// rendered at `K_BRIDGE_MAX_PATCH`, i.e. this patch is coarser than anything the
    /// reference produces. Read by the degenerate-edge offset measurement.
    pub clipped: bool,
}

impl Patch {
    /// A patch with its buffers reserved for the largest side.
    pub fn new_max() -> Patch {
        let n = (K_BRIDGE_MAX_PATCH * K_BRIDGE_MAX_PATCH) as usize;
        Patch {
            img: vec![0; n],
            flat: vec![0.0; n],
            ..Patch::default()
        }
    }
}

/// Cached per-S unit ramps for the gnomonic patch grid, built once per distinct S actually
/// used: per patch only e1/e2/e3 and tan_t change; the ramp is reused and each pixel costs
/// one multiply-add per axis plus one normalise, with no meshgrid and no atan2 on the grid.
#[derive(Debug, Default)]
pub struct BridgeGrid {
    cache: Vec<Vec<f32>>,
}

impl BridgeGrid {
    pub fn new() -> BridgeGrid {
        BridgeGrid::default()
    }

    /// `s` values in [-1, 1] at pixel centres: `(i + 0.5) / s * 2 - 1`. The slice is the
    /// same storage on repeated calls for the same `s`.
    pub fn unit(&mut self, s: i32) -> &[f32] {
        let si = s.max(0) as usize;
        if self.cache.len() <= si {
            self.cache.resize(si + 1, Vec::new());
        }
        if self.cache[si].is_empty() {
            self.cache[si] = (0..si)
                .map(|i| (((i as f64 + 0.5) / si as f64) * 2.0 - 1.0) as f32)
                .collect();
        }
        &self.cache[si]
    }
}

/// `s = clip(round(2*tan(half_fov)*(H/pi)*oversample), K_BRIDGE_MIN_PATCH,
/// min(max_side, K_BRIDGE_MAX_PATCH))`. `max_side` is applied BEFORE the minimum, so the
/// 40 px floor wins over a smaller cap and the patch never degenerates. Capping the side
/// and lowering the rate are the same operation: the grid is built from `s` alone and
/// tan_t does not depend on the rate.
pub fn patch_side(half_fov: f64, h: i32, oversample: f64, max_side: i32) -> i32 {
    let mut s = (2.0 * half_fov.tan() * (h as f64 / K_PI) * oversample).round();
    if s > max_side as f64 {
        s = max_side as f64;
    }
    if s < K_BRIDGE_MIN_PATCH as f64 {
        s = K_BRIDGE_MIN_PATCH as f64;
    }
    if s > K_BRIDGE_MAX_PATCH as f64 {
        s = K_BRIDGE_MAX_PATCH as f64;
    }
    s as i32
}

/// Scalar body of `patch_render`'s inner loop: the definition of what the NEON kernel must
/// reproduce, and the path taken for a BGR/RGB or gamma-LUT view and on any non-NEON build.
fn patch_render_scalar(v: &View, u: &[f32], p: &mut Patch) {
    let s = p.s as usize;
    for y in 0..s {
        let gy = u[y] as f64 * p.tan_t;
        let row = &mut p.img[y * s..(y + 1) * s];
        for x in 0..s {
            let gx = u[x] as f64 * p.tan_t;
            let d = normalize(&Vec3::new(
                p.e1.x * gx + p.e2.x * gy + p.e3.x,
                p.e1.y * gx + p.e2.y * gy + p.e3.y,
                p.e1.z * gx + p.e2.z * gy + p.e3.z,
            ));
            let (px, py) = ray_to_px(&d, v.w, v.h);
            row[x] = (sample(v, px as f32, py as f32) + 0.5) as u8;
        }
    }
}

/// Renders a gnomonic patch of side `patch_side(half_fov, v.h, ..)` centred on ray
/// `centre`: grid value `g_i = unit[i] * tan(half_fov)`, `d = e1*gx + e2*gy + e3`
/// (normalised), sampled from the full-res view at `ray_to_px(d)`. Longitude wraps and
/// latitude reflects over the pole (the sampler's own sphere topology), so the seam and
/// the poles are not special cases.
pub fn patch_render(
    v: &View,
    centre: &Vec3,
    half_fov: f64,
    g: &mut BridgeGrid,
    p: &mut Patch,
    oversample: f64,
    max_side: i32,
) {
    p.s = patch_side(half_fov, v.h, oversample, max_side);
    // Did a per-pass cap take this patch below the reference's own framing?
    p.clipped = p.s < patch_side(half_fov, v.h, oversample, K_BRIDGE_MAX_PATCH);
    p.tan_t = half_fov.tan();
    p.e3 = normalize(centre);
    let (e1, e2) = basis_at(&p.e3);
    p.e1 = e1;
    p.e2 = e2;
    let s = p.s as usize;
    if p.img.len() < s * s {
        p.img.resize(s * s, 0);
    }
    let u = g.unit(p.s);
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    if v.fmt == Fmt::Gray8 && v.lut.is_none() {
        crate::simd::patch_render_gray(
            v.data,
            v.w,
            v.h,
            v.stride,
            u,
            s,
            p.tan_t,
            &p.e1,
            &p.e2,
            &p.e3,
            &mut p.img[..s * s],
        );
        return;
    }
    patch_render_scalar(v, u, p);
}

/// The same patch-pixel -> ray mapping `patch_render` samples at, without the sampling.
/// `x`, `y` may be fractional (e.g. a polished quad corner).
pub fn patch_to_ray(p: &Patch, x: f64, y: f64) -> Vec3 {
    let gx = ((x + 0.5) / p.s as f64 * 2.0 - 1.0) * p.tan_t;
    let gy = ((y + 0.5) / p.s as f64 * 2.0 - 1.0) * p.tan_t;
    normalize(&Vec3::new(
        p.e1.x * gx + p.e2.x * gy + p.e3.x,
        p.e1.y * gx + p.e2.y * gy + p.e3.y,
        p.e1.z * gx + p.e2.z * gy + p.e3.z,
    ))
}

/// Reusable scratch for every stage of the bridge (the C++ keeps these as thread-local
/// buffers). Create one and pass it to the functions that take `&mut Scratch`.
#[derive(Debug, Default)]
pub struct Scratch {
    // trace_contours
    visited: Vec<u8>,
    area: Vec<f64>,
    bbox: Vec<i32>,
    // contour_hull_area
    hull_p: Vec<(i32, i32)>,
    hull_h: Vec<(i32, i32)>,
    // Douglas-Peucker and quad_from_contour
    stack: Vec<(i32, i32)>,
    p: Vec<[f64; 2]>,
    keep: Vec<u8>,
    arc1: Vec<[f64; 2]>,
    arc2: Vec<[f64; 2]>,
    keep1: Vec<u8>,
    keep2: Vec<u8>,
    v: Vec<i32>,
    alive: Vec<u8>,
    // polish_quad
    q: Vec<[f64; 2]>,
    // quad_from_lines
    lp: Vec<[f64; 2]>,
    larc: Vec<[f64; 2]>,
    lkeep: Vec<u8>,
    lkp: Vec<u8>,
    lv: Vec<i32>,
    runs: Vec<Run>,
    groups: Vec<Grp>,
    // patch_flatten
    field: Vec<f32>,
    ftmp: Vec<f32>,
    line: Vec<f32>,
    ix0: Vec<i32>,
    wxs: Vec<f32>,
    fr: Vec<f32>,
    brow: Vec<f32>,
    // adaptive_binarise
    integ: Vec<f64>,
    cx0v: Vec<i32>,
    cx1v: Vec<i32>,
}

// ---- flatten by sigma = S/6 before Otsu --------------------------------------------
//
// Under an illumination gradient Otsu splits the patch into "lit" and "shaded" halves and
// the marker contour is never found. Dividing by a wide Gaussian first fixes it at every
// measured gradient magnitude with no regression on a clean frame (background
// normalisation, i.e. subtracting the blur, worsens the noise floor by 25-39 %).

/// One edge-clamped running-sum box pass over the `fw` samples `inp`; sample `i` of the
/// result goes to `out[i * stride_out]`.
#[inline(always)]
fn box_line(inp: &[f32], out: &mut [f32], stride_out: usize, bw_half: i32, binv: f32) {
    let fw = inp.len();
    let bh = bw_half as usize;
    let mut acc = 0.0f64;
    // taps outside [0, fw) repeat the end sample
    for k in 0..=2 * bh {
        let kk = k.saturating_sub(bh).min(fw - 1);
        acc += inp[kk] as f64;
    }
    let binv = binv as f64;
    for i in 0..fw {
        out[i * stride_out] = (acc * binv) as f32;
        let ai = (i + bh + 1).min(fw - 1);
        let si = i.saturating_sub(bh);
        acc += inp[ai] as f64 - inp[si] as f64;
    }
}

/// `flat[i] = 255 * img[i] / max(blur(img)[i], 1)`, sigma = S/6.
///
/// sigma = S/6 is deliberately WIDE (it must capture the whole patch's illumination TREND,
/// not its texture), so a direct FIR convolution is O(S^3). Since the field only needs the
/// trend, it is estimated on a box-decimated copy of the patch (at most 64 px on a side),
/// blurred there at the same S/6-in-source-px smoothness by three running-sum box passes
/// (width w = sqrt(4 sigma^2 + 1) approximates a Gaussian within 3 %), then bilinearly
/// upsampled back to S x S: O(S^2) total. For S < 128 the decimation is the identity.
pub fn patch_flatten(p: &mut Patch, sc: &mut Scratch) {
    let s = p.s;
    let su = s as usize;
    if p.flat.len() < su * su {
        p.flat.resize(su * su, 0.0);
    }
    let sigma = s as f64 / 6.0;

    let d = ((s + 63) / 64).max(1);
    let fw = (s + d - 1) / d; // decimated field side, <= 64 for S <= K_BRIDGE_MAX_PATCH
    let fwu = fw as usize;
    let sigma_d = sigma / d as f64;
    let mut bw_half = (0.5 * ((4.0 * sigma_d * sigma_d + 1.0).sqrt() - 1.0)).round() as i32;
    let maxr = fw / 2;
    if bw_half > maxr {
        bw_half = maxr;
    }
    if bw_half < 0 {
        bw_half = 0;
    }

    // Box-decimate: the last row/column of blocks is partial whenever D does not divide S.
    sc.field.clear();
    sc.field.resize(fwu * fwu, 0.0);
    let img = &p.img[..su * su];
    let du = d as usize;
    for (j, frow) in sc.field.chunks_exact_mut(fwu).enumerate() {
        let y0 = j * du;
        let y1 = (y0 + du).min(su);
        for (i, cell) in frow.iter_mut().enumerate() {
            let x0 = i * du;
            let x1 = (x0 + du).min(su);
            let mut sum = 0u32;
            for y in y0..y1 {
                for &b in &img[y * su + x0..y * su + x1] {
                    sum += b as u32;
                }
            }
            let cnt = (y1 - y0) * (x1 - x0);
            *cell = sum as f32 / cnt as f32;
        }
    }

    // Three separable edge-clamped box passes over the decimated field.
    if bw_half > 0 {
        sc.ftmp.clear();
        sc.ftmp.resize(fwu * fwu, 0.0);
        sc.line.clear();
        sc.line.resize(fwu, 0.0);
        let bw = 2 * bw_half + 1;
        let binv = 1.0f32 / bw as f32;
        let (field, ftmp, line) = (&mut sc.field, &mut sc.ftmp, &mut sc.line);
        for _pass in 0..3 {
            for (src, dst) in field.chunks_exact(fwu).zip(ftmp.chunks_exact_mut(fwu)) {
                box_line(src, dst, 1, bw_half, binv);
            }
            for x in 0..fwu {
                for (l, row) in line.iter_mut().zip(ftmp.chunks_exact(fwu)) {
                    *l = row[x];
                }
                box_line(line, &mut field[x..], fwu, bw_half, binv);
            }
        }
    }

    // Bilinear upsample: field cell (i,j) represents the source block centred at source
    // pixel (i*D + (D-1)/2, j*D + (D-1)/2). D == 1 makes this an exact identity lookup.
    // The x-half of the mapping is the same for every row, so it is built once; a row of
    // the blurred field is collapsed to the fw-wide `fr` line by the y-blend first.
    let half = (d - 1) as f64 * 0.5;
    let fwm1 = (fw - 1) as f64;
    sc.ix0.clear();
    sc.ix0.resize(su, 0);
    sc.wxs.clear();
    sc.wxs.resize(su, 0.0);
    for x in 0..su {
        let mut fx = (x as f64 - half) / d as f64;
        if fx < 0.0 {
            fx = 0.0;
        } else if fx > fwm1 {
            fx = fwm1;
        }
        let x0 = fx as i32;
        sc.ix0[x] = x0;
        sc.wxs[x] = (fx - x0 as f64) as f32;
    }
    sc.fr.clear();
    sc.fr.resize(fwu, 0.0);
    sc.brow.clear();
    sc.brow.resize(su, 0.0);
    let (field, fr, brow) = (
        &sc.field[..fwu * fwu],
        &mut sc.fr[..fwu],
        &mut sc.brow[..su],
    );
    let (ix0, wxs) = (&sc.ix0[..su], &sc.wxs[..su]);
    let (img, flat) = (&p.img[..su * su], &mut p.flat[..su * su]);
    for y in 0..su {
        let mut fy = (y as f64 - half) / d as f64;
        if fy < 0.0 {
            fy = 0.0;
        } else if fy > fwm1 {
            fy = fwm1;
        }
        let y0 = fy as usize;
        let y1 = if y0 + 1 < fwu { y0 + 1 } else { y0 };
        let wy = (fy - y0 as f64) as f32;
        let f0 = &field[y0 * fwu..(y0 + 1) * fwu];
        let f1 = &field[y1 * fwu..(y1 + 1) * fwu];
        for ((o, &a), &b) in fr.iter_mut().zip(f0).zip(f1) {
            *o = crate::image::fmadd(b - a, wy, a);
        }
        let srow = &img[y * su..(y + 1) * su];
        let drow = &mut flat[y * su..(y + 1) * su];
        // ix0 is non-decreasing and constant over runs of D columns, so the field line is
        // walked cell by cell -- no per-pixel gather.
        {
            let mut x = 0usize;
            let mut i = 0usize;
            while i < fwu && x < su {
                let a = fr[i];
                let b = fr[if i + 1 < fwu { i + 1 } else { i }];
                let dba = b - a;
                while x < su && ix0[x] == i as i32 {
                    let bv = crate::image::fmadd(dba, wxs[x], a);
                    brow[x] = if bv > 1.0 { bv } else { 1.0 };
                    x += 1;
                }
                i += 1;
            }
            while x < su {
                brow[x] = 1.0; // defensive, unreachable
                x += 1;
            }
        }
        #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
        let mut x = crate::simd::flat_div_row(srow, brow, drow);
        #[cfg(not(all(target_arch = "aarch64", not(feature = "no-neon"))))]
        let mut x = 0usize;
        while x < su {
            drow[x] = 255.0f32 * srow[x] as f32 / brow[x];
            x += 1;
        }
    }
}

/// Exact 1-D Otsu over `v` (sorted internally in `scratch`, O(n log n)); no histogram
/// binning. Sort, prefix-sum, maximise `w0*(1-w0)*(m0-m1)^2` over every split, forbid
/// splitting inside a tie (adjacent equal values), return the midpoint of the winning
/// pair. `scratch` must be at least `v.len()` long; `v` itself is never modified.
pub fn otsu_exact(v: &[f32], scratch: &mut [f32]) -> f64 {
    let n = v.len();
    if n <= 1 {
        return if n == 1 { v[0] as f64 + 1.0 } else { 1.0 };
    }
    let sc = &mut scratch[..n];
    sc.copy_from_slice(v);
    sc.sort_unstable_by(f32::total_cmp);
    if (sc[n - 1] as f64) - (sc[0] as f64) < 1e-9 {
        return sc[0] as f64 + 1.0;
    }
    let mut total = 0.0f64;
    for &x in sc.iter() {
        total += x as f64;
    }
    let mut best_sb = -1.0f64;
    let mut best_i: i32 = -1;
    let mut cs = 0.0f64; // prefix sum through index k-1
    for k in 1..n {
        cs += sc[k - 1] as f64;
        if sc[k] == sc[k - 1] {
            continue; // forbid splitting inside a tie
        }
        let w0 = k as f64 / n as f64;
        let m0 = cs / k as f64;
        let m1 = (total - cs) / (n - k) as f64;
        let dd = m0 - m1;
        let sb = w0 * (1.0 - w0) * dd * dd;
        if sb > best_sb {
            best_sb = sb;
            best_i = k as i32 - 1;
        }
    }
    if best_i < 0 {
        return sc[0] as f64 + 1.0; // defensive: the spread check above rules this out
    }
    let bi = best_i as usize;
    0.5 * (sc[bi] as f64 + sc[bi + 1] as f64)
}

/// Otsu's split over a finished 256-bin histogram.
fn otsu_from_hist(hist: &[u32; 256]) -> f64 {
    let mut total = 0.0f64;
    let mut sum_all = 0.0f64;
    for b in 0..256 {
        total += hist[b] as f64;
        sum_all += b as f64 * hist[b] as f64;
    }
    if total <= 0.0 {
        return 1.0;
    }
    let (mut w0, mut sum0, mut best_sb) = (0.0f64, 0.0f64, -1.0f64);
    let mut best_t: i32 = -1;
    for t in 0..255usize {
        w0 += hist[t] as f64;
        sum0 += t as f64 * hist[t] as f64;
        if w0 <= 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 <= 0.0 {
            break;
        }
        let m0 = sum0 / w0;
        let m1 = (sum_all - sum0) / w1;
        let dd = m0 - m1;
        let sb = w0 * w1 * dd * dd; // proportional to the exact between-class variance
        if sb > best_sb {
            best_sb = sb;
            best_t = t as i32;
        }
    }
    if best_t < 0 {
        return sum_all / total + 1.0; // degenerate: every value in one bin
    }
    best_t as f64 + 0.5
}

/// Four interleaved histograms break the `++hist[v[i]]` store-to-load dependency that
/// otherwise serialises the histogram at ~3 cycles per pixel.
struct Hist4 {
    h: [[u32; 256]; 4],
}

impl Hist4 {
    fn new() -> Hist4 {
        Hist4 { h: [[0; 256]; 4] }
    }
    fn merge(&self) -> [u32; 256] {
        let mut out = [0u32; 256];
        for b in 0..256 {
            out[b] = self.h[0][b] + self.h[1][b] + self.h[2][b] + self.h[3][b];
        }
        out
    }
}

/// The same 256-bin Otsu over genuinely 8-bit data: EXACT, no binning approximation, and
/// bit-for-bit what OpenCV's `THRESH_OTSU` computes on the same uint8 patch. This is the
/// bridge's primary threshold (mode 0). `thr` is always `best_t + 0.5`.
pub fn otsu_hist8_u8(v: &[u8]) -> f64 {
    if v.is_empty() {
        return 1.0;
    }
    let mut acc = Hist4::new();
    let mut ch = v.chunks_exact(4);
    for c in &mut ch {
        acc.h[0][c[0] as usize] += 1;
        acc.h[1][c[1] as usize] += 1;
        acc.h[2][c[2] as usize] += 1;
        acc.h[3][c[3] as usize] += 1;
    }
    for &x in ch.remainder() {
        acc.h[0][x as usize] += 1;
    }
    otsu_from_hist(&acc.merge())
}

/// The same 256-bin Otsu over float data (round-and-clamp binning): a deliberate,
/// validated approximation for `patch_flatten`'s ratio output, which can exceed 255. The
/// bridge never uses `otsu_exact` on a whole patch (sorting 590k values twice per
/// candidate would blow the per-candidate budget).
pub fn otsu_hist8(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 1.0;
    }
    let bin = |f: f32| -> usize {
        let b = (f + 0.5) as i32;
        b.clamp(0, 255) as usize
    };
    let mut acc = Hist4::new();
    let mut ch = v.chunks_exact(4);
    for c in &mut ch {
        acc.h[0][bin(c[0])] += 1;
        acc.h[1][bin(c[1])] += 1;
        acc.h[2][bin(c[2])] += 1;
        acc.h[3][bin(c[3])] += 1;
    }
    for &x in ch.remainder() {
        acc.h[0][bin(x)] += 1;
    }
    otsu_from_hist(&acc.merge())
}

/// `bw[i] = 255` when `p.flat[i] < thr`, else 0 (THRESH_BINARY_INV: the foreground is the
/// DARK marker interior). `bw` must hold at least `s * s` bytes.
pub fn patch_binarise(p: &Patch, thr: f64, bw: &mut [u8]) {
    let n = (p.s * p.s) as usize;
    let t = thr as f32;
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    let mut i = crate::simd::binarise_lt_f32(&p.flat[..n], t, &mut bw[..n]);
    #[cfg(not(all(target_arch = "aarch64", not(feature = "no-neon"))))]
    let mut i = 0usize;
    while i < n {
        bw[i] = if p.flat[i] < t { 255 } else { 0 };
        i += 1;
    }
}

/// Same, straight off the raw uint8 patch (mode 0; `p.flat` need not exist).
pub fn patch_binarise_raw(p: &Patch, thr: f64, bw: &mut [u8]) {
    let n = (p.s * p.s) as usize;
    let t = thr as f32;
    // thr is always best_t + 0.5 with best_t an integer in [0, 254], so "img < thr" is
    // exactly "img <= (uint8)floor(thr)": an integer compare, no float conversion.
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    let mut i = if t > 0.0 && t < 255.0 {
        crate::simd::binarise_le_u8(&p.img[..n], t as u8, &mut bw[..n])
    } else {
        0
    };
    #[cfg(not(all(target_arch = "aarch64", not(feature = "no-neon"))))]
    let mut i = 0usize;
    while i < n {
        bw[i] = if (p.img[i] as f32) < t { 255 } else { 0 };
        i += 1;
    }
}

// ---- patch contour trace and marker-contour selection -------------------------------

#[derive(Debug, Clone, Default)]
pub struct PatchContours {
    /// Interleaved x, y.
    pub pts: Vec<i32>,
    /// Start index (in points) per contour, plus a terminator.
    pub start: Vec<u32>,
    /// -1 for a top-level contour.
    pub parent: Vec<i32>,
    /// Nesting depth by geometric containment. Odd is USUALLY a hole, but not always: a
    /// dark region touching the patch edge has no traced outer border, so everything it
    /// encloses sits one level shallower than it really is. Use `hole` to tell holes from
    /// dark regions.
    pub depth: Vec<i32>,
    /// True for a hole border (background inside), from the traced winding: the border
    /// follower keeps the foreground on a fixed side, so an outer border has a positive
    /// shoelace area and a hole border a negative one, whatever else was traced.
    pub hole: Vec<bool>,
}

/// Even-odd crossing-number test over a contour's own traced points taken as a closed
/// polyline.
fn point_in_polygon(px: f64, py: f64, pts: &[i32], begin: u32, end: u32) -> bool {
    let mut inside = false;
    let n = (end - begin) as usize;
    let b = begin as usize;
    let mut j = n - 1;
    for i in 0..n {
        let xi = pts[2 * (b + i)] as f64;
        let yi = pts[2 * (b + i) + 1] as f64;
        let xj = pts[2 * (b + j)] as f64;
        let yj = pts[2 * (b + j) + 1] as f64;
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Shoelace area of a contour's traced points (cv2.contourArea).
fn polygon_area(pts: &[i32], begin: u32, end: u32) -> f64 {
    polygon_signed_area(pts, begin, end).abs()
}

/// Signed shoelace area (x right, y down): positive for an outer border as
/// `trace_contours` walks it, negative for a hole border.
fn polygon_signed_area(pts: &[i32], begin: u32, end: u32) -> f64 {
    let n = (end - begin) as usize;
    let b = begin as usize;
    let mut s = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let xi = pts[2 * (b + i)] as f64;
        let yi = pts[2 * (b + i) + 1] as f64;
        let xj = pts[2 * (b + j)] as f64;
        let yj = pts[2 * (b + j) + 1] as f64;
        s += xi * yj - xj * yi;
    }
    s * 0.5
}

/// Area-weighted centroid via Green's theorem (cv2.moments), NOT the mean of the boundary
/// points (which is the centroid of the traced curve, generally different for an
/// asymmetric shape).
fn polygon_centroid(pts: &[i32], begin: u32, end: u32) -> (f64, f64) {
    let n = (end - begin) as usize;
    let b = begin as usize;
    let (mut a, mut cx, mut cy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let j = (i + 1) % n;
        let xi = pts[2 * (b + i)] as f64;
        let yi = pts[2 * (b + i) + 1] as f64;
        let xj = pts[2 * (b + j)] as f64;
        let yj = pts[2 * (b + j) + 1] as f64;
        let cr = xi * yj - xj * yi;
        a += cr;
        cx += (xi + xj) * cr;
        cy += (yi + yj) * cr;
    }
    a *= 0.5;
    if a.abs() < 1e-9 {
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for i in 0..n {
            sx += pts[2 * (b + i)] as f64;
            sy += pts[2 * (b + i) + 1] as f64;
        }
        return (sx / n as f64, sy / n as f64);
    }
    (cx / (6.0 * a), cy / (6.0 * a))
}

#[inline]
fn cpt_cross(o: (i32, i32), a: (i32, i32), b: (i32, i32)) -> i64 {
    (a.0 - o.0) as i64 * (b.1 - o.1) as i64 - (a.1 - o.1) as i64 * (b.0 - o.0) as i64
}

/// Monotone-chain convex hull area of a contour's traced points
/// (cv2.contourArea(cv2.convexHull(c))).
fn contour_hull_area(pts: &[i32], begin: u32, end: u32, sc: &mut Scratch) -> f64 {
    let pp = &mut sc.hull_p;
    pp.clear();
    for i in begin as usize..end as usize {
        pp.push((pts[2 * i], pts[2 * i + 1]));
    }
    pp.sort_unstable();
    pp.dedup();
    let n = pp.len();
    if n < 3 {
        return 0.0;
    }
    let hh = &mut sc.hull_h;
    hh.clear();
    hh.resize(2 * n, (0, 0));
    let mut k = 0usize;
    for i in 0..n {
        while k >= 2 && cpt_cross(hh[k - 2], hh[k - 1], pp[i]) <= 0 {
            k -= 1;
        }
        hh[k] = pp[i];
        k += 1;
    }
    let lower = k + 1;
    for i in (0..n - 1).rev() {
        while k >= lower && cpt_cross(hh[k - 2], hh[k - 1], pp[i]) <= 0 {
            k -= 1;
        }
        hh[k] = pp[i];
        k += 1;
    }
    hh.truncate(k - 1);
    let mut s = 0i64;
    for i in 0..hh.len() {
        let a = hh[i];
        let b = hh[(i + 1) % hh.len()];
        s += a.0 as i64 * b.1 as i64 - b.0 as i64 * a.1 as i64;
    }
    (s as f64).abs() * 0.5
}

const DX8: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const DY8: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];

fn trace_one(
    bw: &[u8],
    s: i32,
    vis: &mut [u8],
    c: &mut PatchContours,
    minlen: i32,
    x0: i32,
    y0: i32,
    dir0: i32,
) {
    let begin = (c.pts.len() / 2) as u32;
    let (mut cx, mut cy, mut dir) = (x0, y0, dir0);
    let (sx, sy) = (x0, y0);
    let mut steps = 0;
    let (mut firstnx, mut firstny) = (-1, -1);
    loop {
        vis[cy as usize * s as usize + cx as usize] = 1;
        c.pts.push(cx);
        c.pts.push(cy);
        let (mut nd, mut nx, mut ny) = (-1, 0, 0);
        for i in 0..8 {
            let dd = (dir + i) & 7;
            let tx = cx + DX8[dd as usize];
            let ty = cy + DY8[dd as usize];
            if tx < 0 || ty < 0 || tx >= s || ty >= s {
                continue;
            }
            if bw[ty as usize * s as usize + tx as usize] != 0 {
                nd = dd;
                nx = tx;
                ny = ty;
                break;
            }
        }
        if nd < 0 {
            break; // isolated pixel
        }
        if c.pts.len() / 2 == begin as usize + 1 {
            firstnx = nx;
            firstny = ny;
        } else if cx == sx && cy == sy && nx == firstnx && ny == firstny {
            c.pts.truncate(c.pts.len() - 2);
            break; // Jacob's stopping criterion
        }
        cx = nx;
        cy = ny;
        dir = (nd + 5) & 7; // back up 3 (i.e. +5 mod 8)
        steps += 1;
        if steps > 4 * (s + s) {
            break;
        }
    }
    let len = (c.pts.len() / 2) as u32 - begin;
    if (len as i32) < minlen {
        c.pts.truncate(begin as usize * 2);
        return;
    }
    c.start.push(begin);
}

/// Moore-neighbour border follower (Jacob's stopping criterion, `dir = (nd + 5) & 7`
/// back-up) extended for INNER borders: a foreground pixel with background to its right
/// (not yet visited) starts a hole contour, exactly as one with background to its left
/// starts an outer one. parent/depth are then resolved geometrically (point-in-polygon of
/// each contour's first point against every other, picking the smallest-area container);
/// `hole` comes from each contour's own winding, not from the depth parity.
/// `bw` is `s * s` bytes; contours shorter than `minlen` points are dropped. Returns the
/// number of contours kept.
pub fn trace_contours(
    bw: &[u8],
    s: i32,
    c: &mut PatchContours,
    minlen: i32,
    sc: &mut Scratch,
) -> i32 {
    c.pts.clear();
    c.start.clear();
    c.parent.clear();
    c.depth.clear();
    c.hole.clear();
    let su = s as usize;
    sc.visited.clear();
    sc.visited.resize(su * su, 0);

    // A patch is overwhelmingly background, so the raster scan skips runs of eight zero
    // bytes at a time. The window only ever SKIPS: s[x-1]/s[x+1] are still read one pixel
    // at a time for the pixels the skip does not eliminate.
    for y in 1..s - 1 {
        let row = y as usize * su;
        let mut x = 1i32;
        while x < s - 1 {
            let xu = x as usize;
            #[allow(clippy::int_plus_one)] // mirrors the C++ bound `x + 8 <= S - 1`
            if x + 8 <= s - 1 {
                let mut w8 = [0u8; 8];
                w8.copy_from_slice(&bw[row + xu..row + xu + 8]);
                if u64::from_ne_bytes(w8) == 0 {
                    x += 8;
                    continue;
                }
            }
            if bw[row + xu] == 0 || sc.visited[row + xu] != 0 {
                x += 1;
                continue;
            }
            if bw[row + xu - 1] == 0 {
                trace_one(bw, s, &mut sc.visited, c, minlen, x, y, 6); // outer: bg left, search from N
                x += 1;
                continue;
            }
            if bw[row + xu + 1] == 0 {
                trace_one(bw, s, &mut sc.visited, c, minlen, x, y, 2); // hole: bg right, search from S
            }
            x += 1;
        }
    }
    c.start.push((c.pts.len() / 2) as u32);
    let n = c.start.len() - 1;

    // parent/depth: geometric containment. A contour's parent is the smallest-area OTHER
    // contour that contains its first traced point; depth counts ancestors (odd = a hole).
    c.parent.resize(n, -1);
    c.depth.resize(n, 0);
    c.hole.resize(n, false);
    sc.area.clear();
    sc.area.resize(n, 0.0);
    sc.bbox.clear();
    sc.bbox.resize(n * 4, 0);
    for i in 0..n {
        let b0 = c.start[i];
        let e0 = c.start[i + 1];
        let sa = polygon_signed_area(&c.pts, b0, e0);
        sc.area[i] = sa.abs();
        c.hole[i] = sa < 0.0;
        let mut x0 = c.pts[2 * b0 as usize];
        let mut x1 = x0;
        let mut y0 = c.pts[2 * b0 as usize + 1];
        let mut y1 = y0;
        for k in b0 + 1..e0 {
            let px = c.pts[2 * k as usize];
            let py = c.pts[2 * k as usize + 1];
            if px < x0 {
                x0 = px;
            } else if px > x1 {
                x1 = px;
            }
            if py < y0 {
                y0 = py;
            } else if py > y1 {
                y1 = py;
            }
        }
        sc.bbox[i * 4] = x0;
        sc.bbox[i * 4 + 1] = y0;
        sc.bbox[i * 4 + 2] = x1;
        sc.bbox[i * 4 + 3] = y1;
    }
    // point_in_polygon is O(len) and this pass runs O(n^2) of them; a containing contour
    // must contain the whole child, so its bounding box must contain the child's. That is
    // an O(1) rejection which can only reject pairs point_in_polygon would have rejected,
    // so the resolved parents are unchanged.
    for i in 0..n {
        let px = c.pts[2 * c.start[i] as usize] as f64;
        let py = c.pts[2 * c.start[i] as usize + 1] as f64;
        let bi = [
            sc.bbox[i * 4],
            sc.bbox[i * 4 + 1],
            sc.bbox[i * 4 + 2],
            sc.bbox[i * 4 + 3],
        ];
        let mut best = -1i32;
        let mut best_area = 1e300f64;
        for j in 0..n {
            if j == i || sc.area[j] <= sc.area[i] {
                continue; // a parent must be strictly larger
            }
            if sc.area[j] >= best_area {
                continue;
            }
            let bj = &sc.bbox[j * 4..j * 4 + 4];
            if bi[0] < bj[0] || bi[1] < bj[1] || bi[2] > bj[2] || bi[3] > bj[3] {
                continue;
            }
            if point_in_polygon(px, py, &c.pts, c.start[j], c.start[j + 1]) {
                best = j as i32;
                best_area = sc.area[j];
            }
        }
        c.parent[i] = best;
    }
    for i in 0..n {
        let mut d = 0;
        let mut p = c.parent[i];
        while p >= 0 {
            d += 1;
            p = c.parent[p as usize]; // parent area strictly increases -> no cycles
        }
        c.depth[i] = d;
    }
    n as i32
}

/// Selection thresholds: hole borders (by winding, see `PatchContours::hole`) are skipped, area in [2 %, 92 %] of S^2 rejects, fewer
/// than 12 points rejects, solidity (area / convex-hull area, both by the shoelace
/// formula) below `solidity_min` rejects, not containing the patch centre (S-1)/2
/// rejects; the surviving contour whose AREA-WEIGHTED centroid is nearest the patch
/// centre wins. Returns -1 when no contour qualifies.
pub fn select_marker_contour(
    c: &PatchContours,
    s: i32,
    solidity_min: f64,
    sc: &mut Scratch,
) -> i32 {
    let n = c.start.len() as i32 - 1;
    let ctr = (s as f64 - 1.0) * 0.5;
    let area_lo = 0.02 * s as f64 * s as f64;
    let area_hi = 0.92 * s as f64 * s as f64;
    let mut best = -1i32;
    let mut best_d2 = 1e300f64;
    for i in 0..n {
        let iu = i as usize;
        if c.hole[iu] {
            continue; // a hole, not a dark region
        }
        let begin = c.start[iu];
        let end = c.start[iu + 1];
        if ((end - begin) as i32) < 12 {
            continue;
        }
        let area = polygon_area(&c.pts, begin, end);
        if area < area_lo || area > area_hi {
            continue;
        }
        // The centre test is O(len) and the hull O(len log len): the cheap one goes
        // first (the accept/reject set is identical either way).
        if !point_in_polygon(ctr, ctr, &c.pts, begin, end) {
            continue;
        }
        let ha = contour_hull_area(&c.pts, begin, end, sc);
        if ha <= 0.0 || area / ha < solidity_min {
            continue;
        }
        let (cx, cy) = polygon_centroid(&c.pts, begin, end);
        let dx = cx - ctr;
        let dy = cy - ctr;
        let d2 = dx * dx + dy * dy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = i;
        }
    }
    best
}

fn perp_dist(ax: f64, ay: f64, bx: f64, by: f64, px: f64, py: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-9 {
        let ex = px - ax;
        let ey = py - ay;
        return (ex * ex + ey * ey).sqrt();
    }
    let cr = dx * (py - ay) - dy * (px - ax);
    cr.abs() / len
}

/// Standard stack-based Douglas-Peucker over an OPEN polyline `pts[0..=m]`; both
/// endpoints are always kept.
fn dp_run(pts: &[[f64; 2]], m: i32, eps: f64, keep: &mut [u8], stack: &mut Vec<(i32, i32)>) {
    keep[0] = 1;
    keep[m as usize] = 1;
    stack.clear();
    stack.push((0, m));
    while let Some((lo, hi)) = stack.pop() {
        if hi <= lo + 1 {
            continue;
        }
        let mut best_d = -1.0f64;
        let mut best_i = -1i32;
        let (l, h) = (lo as usize, hi as usize);
        for i in l + 1..h {
            let d = perp_dist(
                pts[l][0], pts[l][1], pts[h][0], pts[h][1], pts[i][0], pts[i][1],
            );
            if d > best_d {
                best_d = d;
                best_i = i as i32;
            }
        }
        if best_d > eps {
            keep[best_i as usize] = 1;
            stack.push((lo, best_i));
            stack.push((best_i, hi));
        }
    }
}

fn convex4(p: &[[f64; 2]], idx: &[i32; 4]) -> bool {
    let mut sign = 0.0f64;
    for i in 0..4 {
        let a = &p[idx[i] as usize];
        let b = &p[idx[(i + 1) % 4] as usize];
        let d = &p[idx[(i + 2) % 4] as usize];
        let ux = b[0] - a[0];
        let uy = b[1] - a[1];
        let vx = d[0] - b[0];
        let vy = d[1] - b[1];
        let cr = ux * vy - uy * vx;
        if cr.abs() < 1e-9 {
            continue; // near-collinear: not a counterexample either way
        }
        if sign == 0.0 {
            sign = cr;
        } else if (cr > 0.0) != (sign > 0.0) {
            return false;
        }
    }
    true
}

/// approxPolyDP-style 4-gon: Douglas-Peucker at eps in {0.02, 0.03, 0.04, 0.05, 0.07,
/// 0.10} x perimeter (closed-curve variant: split into two arcs at the two EXTREME points,
/// simplify each, merge), accepting the first result with exactly 4 convex vertices;
/// otherwise the farthest-point fallback (furthest point from the centroid, furthest from
/// that, then the extreme points either side of the line between them). False when the
/// contour is too short (< 4 points on the DP path, < 8 on the fallback path) or
/// degenerate.
///
/// The closed contour must be split at its two extreme points exactly as
/// cv2.approxPolyDP does (the point farthest from `pts[0]`, then the farthest from that),
/// NOT at `pts[0]`: that is whatever pixel the raster scan reached first, not a corner, and
/// forcing it into the output as a vertex displaces a side's line enough that the polish
/// finds no points for it and fails.
pub fn quad_from_contour(
    c: &PatchContours,
    idx: i32,
    q: &mut [[f64; 2]; 4],
    sc: &mut Scratch,
) -> bool {
    let begin = c.start[idx as usize];
    let end = c.start[idx as usize + 1];
    let n = (end - begin) as i32;
    if n < 4 {
        return false;
    }
    let nu = n as usize;
    sc.p.clear();
    for i in 0..nu {
        sc.p.push([
            c.pts[2 * (begin as usize + i)] as f64,
            c.pts[2 * (begin as usize + i) + 1] as f64,
        ]);
    }
    let pp = &sc.p;
    let mut peri = 0.0f64;
    for i in 0..nu {
        let j = (i + 1) % nu;
        let dx = pp[j][0] - pp[i][0];
        let dy = pp[j][1] - pp[i][1];
        peri += (dx * dx + dy * dy).sqrt();
    }

    let mut ea = 0i32;
    let mut bd = -1.0f64;
    for i in 1..nu {
        let dx = pp[i][0] - pp[0][0];
        let dy = pp[i][1] - pp[0][1];
        let d = dx * dx + dy * dy;
        if d > bd {
            bd = d;
            ea = i as i32;
        }
    }
    let mut eb = ea;
    bd = -1.0;
    for i in 0..nu {
        let dx = pp[i][0] - pp[ea as usize][0];
        let dy = pp[i][1] - pp[ea as usize][1];
        let d = dx * dx + dy * dy;
        if d > bd {
            bd = d;
            eb = i as i32;
        }
    }
    let len1 = (eb - ea + n) % n; // arc 1: ea -> eb forward
    let len2 = n - len1; // arc 2: eb -> ea forward (wraps)
    if len1 < 1 || len2 < 1 {
        return false; // ea == eb: a degenerate contour
    }

    const EPSS: [f64; 6] = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10];
    sc.arc1.clear();
    for i in 0..=len1 {
        sc.arc1.push(pp[((ea + i) % n) as usize]);
    }
    sc.arc2.clear();
    for i in 0..=len2 {
        sc.arc2.push(pp[((eb + i) % n) as usize]);
    }
    for ef in EPSS {
        let eps = ef * peri;
        sc.keep.clear();
        sc.keep.resize(nu, 0);

        sc.keep1.clear();
        sc.keep1.resize(len1 as usize + 1, 0);
        dp_run(&sc.arc1, len1, eps, &mut sc.keep1, &mut sc.stack);
        for i in 0..len1 {
            if sc.keep1[i as usize] != 0 {
                sc.keep[((ea + i) % n) as usize] = 1;
            }
        }

        sc.keep2.clear();
        sc.keep2.resize(len2 as usize + 1, 0);
        dp_run(&sc.arc2, len2, eps, &mut sc.keep2, &mut sc.stack);
        for i in 0..len2 {
            if sc.keep2[i as usize] != 0 {
                sc.keep[((eb + i) % n) as usize] = 1;
            }
        }

        // cv2.approxPolyDP's final clean-up ("remove extra points on the [almost]
        // straight lines"), walked in cv2's own output order -- cyclically from `eb`.
        // Without it the raw DP keeps a vertex within eps/sqrt(2) of the chord through its
        // surviving neighbours, which on a highly oblique marker splits ONE physical edge
        // into two segments (5 vertices, trial skipped; the coarser eps that finally yields
        // 4 then puts a "corner" ~8 px along an edge). Criterion verbatim from OpenCV
        // (whose eps is already squared there): drop `pt` when
        // |(pt-p0) x (p1-p0)|^2 <= 0.5*eps^2*|p1-p0|^2, AND p1-p0 is off-axis in both
        // components, AND the turn does not double back ((pt-p0).(p1-pt) >= 0). p0 is the
        // last SURVIVING vertex.
        sc.v.clear();
        for i in 0..n {
            let gi = (eb + i) % n;
            if sc.keep[gi as usize] != 0 {
                sc.v.push(gi);
            }
        }
        let mut m = sc.v.len();
        if m > 2 {
            sc.alive.clear();
            sc.alive.resize(m, 1);
            let mut nlive = m as i32;
            let mut prev = m - 1;
            let mut i = 0usize;
            while i < m && nlive > 2 {
                let p0 = &pp[sc.v[prev] as usize];
                let pt = &pp[sc.v[i] as usize];
                let p1 = &pp[sc.v[(i + 1) % m] as usize];
                let dx = p1[0] - p0[0];
                let dy = p1[1] - p0[1];
                let cr = (pt[0] - p0[0]) * dy - (pt[1] - p0[1]) * dx;
                let sip = (pt[0] - p0[0]) * (p1[0] - pt[0]) + (pt[1] - p0[1]) * (p1[1] - pt[1]);
                if cr * cr <= 0.5 * eps * eps * (dx * dx + dy * dy)
                    && dx != 0.0
                    && dy != 0.0
                    && sip >= 0.0
                {
                    sc.alive[i] = 0;
                    nlive -= 1; // p0 stays put
                } else {
                    prev = i;
                }
                i += 1;
            }
            let mut w = 0usize;
            for i in 0..m {
                if sc.alive[i] != 0 {
                    sc.v[w] = sc.v[i];
                    w += 1;
                }
            }
            sc.v.truncate(w);
            m = w;
        }

        let nk = m;
        let mut kept = [0i32; 4];
        kept[..m.min(4)].copy_from_slice(&sc.v[..m.min(4)]);
        if nk == 4 && convex4(pp, &kept) {
            for k in 0..4 {
                q[k][0] = pp[kept[k] as usize][0];
                q[k][1] = pp[kept[k] as usize][1];
            }
            return true;
        }
    }

    // Farthest-point fallback for a convex blob: furthest point `a` from the centroid,
    // furthest `b` from `a`, then the extreme points either side of line ab.
    if n < 8 {
        return false;
    }
    let (mut cx, mut cy) = (0.0f64, 0.0f64);
    for i in 0..nu {
        cx += pp[i][0];
        cy += pp[i][1];
    }
    cx /= n as f64;
    cy /= n as f64;
    let mut ia = 0usize;
    bd = -1.0;
    for i in 0..nu {
        let dx = pp[i][0] - cx;
        let dy = pp[i][1] - cy;
        let d = dx * dx + dy * dy;
        if d > bd {
            bd = d;
            ia = i;
        }
    }
    let mut ib = 0usize;
    bd = -1.0;
    for i in 0..nu {
        let dx = pp[i][0] - pp[ia][0];
        let dy = pp[i][1] - pp[ia][1];
        let d = dx * dx + dy * dy;
        if d > bd {
            bd = d;
            ib = i;
        }
    }
    let dxab = pp[ib][0] - pp[ia][0];
    let dyab = pp[ib][1] - pp[ia][1];
    let abl = (dxab * dxab + dyab * dyab).sqrt();
    if abl < 1e-6 {
        return false;
    }
    let nx = -dyab / abl;
    let ny = dxab / abl;
    let (mut ic1, mut ic2) = (usize::MAX, usize::MAX);
    let (mut smax, mut smin) = (-1e300f64, 1e300f64);
    for i in 0..nu {
        let s = (pp[i][0] - pp[ia][0]) * nx + (pp[i][1] - pp[ia][1]) * ny;
        if s > smax {
            smax = s;
            ic1 = i;
        }
        if s < smin {
            smin = s;
            ic2 = i;
        }
    }
    if smax <= 0.0 || smin >= 0.0 {
        return false;
    }
    q[0] = pp[ia];
    q[1] = pp[ic1];
    q[2] = pp[ib];
    q[3] = pp[ic2];
    order_quad(q);
    true
}

/// Cyclic order the 4 points CCW in patch coords (y down => CW on screen), then roll so
/// index 0 minimises x + y (nearest top-left).
pub fn order_quad(q: &mut [[f64; 2]; 4]) {
    let (mut cx, mut cy) = (0.0f64, 0.0f64);
    for p in q.iter() {
        cx += p[0] * 0.25;
        cy += p[1] * 0.25;
    }
    let mut ang = [0.0f64; 4];
    for i in 0..4 {
        ang[i] = (q[i][1] - cy).atan2(q[i][0] - cx);
    }
    let mut order = [0usize, 1, 2, 3];
    for i in 1..4 {
        let mut j = i;
        while j > 0 && ang[order[j]] < ang[order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut tmp = [[0.0f64; 2]; 4];
    for i in 0..4 {
        tmp[i] = q[order[i]];
    }
    let mut k = 0usize;
    let mut best = tmp[0][0] + tmp[0][1];
    for i in 1..4 {
        let s = tmp[i][0] + tmp[i][1];
        if s < best {
            best = s;
            k = i;
        }
    }
    for i in 0..4 {
        q[i] = tmp[(i + k) % 4];
    }
}

// ---- TLS side lines, 0.5 px outward offset ------------------------------------------

/// Outward push of each fitted line before intersecting, px. A contour is the centres of
/// the outermost FOREGROUND pixels, which sit about half a pixel inside the true 50 %
/// crossing; without this the bridge quad is uniformly 0.63 px too small on the
/// half-diagonal (a pure dilation bias, centre shift < 0.05 px).
pub const K_EDGE_OFFSET_PX: f64 = 0.5;

/// Where a quad came from. The refine stage applies the post-refine consistency check
/// (refined edge vs the bridge's own line) to LINEFIT and RELAXED quads only; a LEGACY quad
/// is the polished 4-gon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BridgeProv {
    #[default]
    Legacy = 0,
    LineFit = 1,
    Relaxed = 2,
}

/// `contour_misfit` above which the legacy 4-gon is abandoned for the line fit (fraction
/// of contour points far from the 4-gon's boundary).
pub const K_BRIDGE_MISFIT_MAX: f64 = 0.04;
/// A line-fit quad is believed only if, on every side, at least this fraction of the
/// non-corner border cells is dark. Measured on an occlusion corpus: every quad whose side
/// ran along an occluder's edge read <= 0.50; every line-fit quad that went on to detect
/// read >= 0.75.
pub const K_LINES_SIDE_DARK_MIN: f64 = 0.75;
/// Relaxed-selection tries per frame.
pub const K_BRIDGE_RELAXED_PER_FRAME: i32 = 8;

/// The plane the mask was thresholded from, so a degenerate edge's outward offset can be
/// MEASURED from the gray patch instead of assumed.
///
/// "About half a pixel" is the MEAN of the contour's inward bias over sub-pixel phases, not
/// a fact about one edge: the real distance from an edge's outermost foreground pixel
/// centre to the 50 % crossing is (1 - phi) px for that edge's own phase phi. An OBLIQUE
/// edge's staircase visits every phase, the TLS fit averages the residual to zero, and
/// `K_EDGE_OFFSET_PX` is unbiased. An edge along a patch AXIS (or its 45 degree diagonal)
/// has a straight run at one phase, so the whole (0.5 - phi) residual, up to +-0.5 px, lands
/// in the quad as a coherent dilation. On a patch the per-pass cap CLIPPED, and only for an
/// edge with no phase to average (`K_EDGE_FLAT_SPREAD`), `polish_quad` therefore measures
/// the offset: the coverage integral of the intensity profile along the fitted normal.
/// An unclipped patch is one the reference itself would render, where the blind offset costs
/// at most 0.31 px_eq; measuring there too is slightly more accurate but moves the quad off
/// the reference's and flips the refiner's bistable branch on some frames.
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeProfile<'a> {
    /// Mode 0's raw patch.
    pub gray: Option<&'a [u8]>,
    /// Mode 1's flattened patch.
    pub flat: Option<&'a [f32]>,
    /// Patch side; 0 = "no profile, use 0.5".
    pub s: i32,
    /// The threshold that produced the mask.
    pub thr: f64,
}

/// Half-window, px, of the intensity profile sampled along the fitted normal. The end
/// samples are taken as the two plateau levels, so the window must clear the transition
/// (the source's ~1 px anti-aliasing scaled by the patch rate, plus the bilinear tap) but
/// stay inside the tag's 1-module black border and 1.5-module quiet zone, which at the
/// 40 px floor is the binding constraint.
pub const K_EDGE_PROBE_SPAN: i32 = 2;
/// Minimum plateau-to-plateau contrast, dn, for a profile to be worth integrating (twice
/// the refiner's contrast floor: this profile spans a printed tag's black border against
/// its white quiet zone).
pub const K_EDGE_PROBE_DN: f64 = 16.0;
/// An edge is DEGENERATE (no sub-pixel phase in its staircase) when its selected contour
/// points all project to the same offset along the fitted normal: exactly 0 for an
/// axis-aligned or 45 degree run; the smallest a REAL staircase gives is 1/sqrt(2) = 0.707.
pub const K_EDGE_FLAT_SPREAD: f64 = 0.5;

/// Which contour points a side's fit takes: normalised along-side coordinate in
/// `(t_lo, t_hi)` and distance < `max(1.6, band_frac * L)` from the pre-polish side. The
/// defaults are the legacy 4-gon's. The line-fit quad starts from MEASURED lines, so it
/// passes `band_frac` 0 (a tight band: a bite's own edge a few px from the line must not
/// pull it) and a wider t range (a side shortened by a bite keeps its visible run).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolishBand {
    pub band_frac: f64,
    pub t_lo: f64,
    pub t_hi: f64,
}

impl Default for PolishBand {
    fn default() -> Self {
        PolishBand {
            band_frac: 0.06,
            t_lo: 0.14,
            t_hi: 0.86,
        }
    }
}

pub const K_LINE_FIT_BAND: PolishBand = PolishBand {
    band_frac: 0.0,
    t_lo: 0.06,
    t_hi: 0.94,
};

/// Smallest-eigenvalue eigenvector of a symmetric 2x2 `[[a,b],[b,d]]` (closed form;
/// eigenvector `(lambda-d, b)` when b is not negligible, else axis-aligned).
///
/// The off-diagonal test must be relative to the matrix's own scale: for an untilted
/// (roll 0) square the TLS scatter is near-exactly axis-aligned and `b`, a sum of
/// mean-centred integer coordinates that is mathematically ~0, comes out ~1e-10 against
/// diagonals of order 1e4. An absolute epsilon lets that float noise into the exact
/// formula, where `lambda_min - d` cancels to rounding error and the eigenvector comes out
/// at an ARBITRARY direction.
fn smallest_eigvec2(a: f64, b: f64, d: f64) -> (f64, f64) {
    let tr = a + d;
    let diff = a - d;
    let disc = (diff * diff + 4.0 * b * b).sqrt();
    let lambda_min = 0.5 * (tr - disc);
    let scale = a.abs().max(d.abs());
    if b.abs() > 1e-9 * scale.max(1.0) {
        let x = lambda_min - d;
        let y = b;
        let n = (x * x + y * y).sqrt();
        if n > 1e-300 {
            return (x / n, y / n);
        }
    }
    if a <= d {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    }
}

fn bridge_half_fov(box_: f64, r: f64) -> f64 {
    let r60 = r.min(60.0 * K_PI / 180.0);
    let mut hf = (box_ * r60.tan()).atan();
    let lo = 0.20 * K_PI / 180.0;
    let hi = 72.0 * K_PI / 180.0;
    if hf < lo {
        hf = lo;
    }
    if hf > hi {
        hf = hi;
    }
    hf
}

/// ADAPTIVE_THRESH_MEAN_C fallback: blk = max(3, (int)(S*0.25)|1), C = 7, via a
/// summed-area table so the window mean is O(1) per pixel. Operates on `p.flat`, the same
/// illumination-corrected input the Otsu path uses. Only the table's first row and column
/// are zeroed, the window's clamped x bounds are per-COLUMN constants hoisted out of the y
/// loop, and the two integral rows the window needs are hoisted per row.
fn adaptive_binarise(p: &Patch, bw: &mut [u8], sc: &mut Scratch) {
    let s = p.s;
    let su = s as usize;
    let blk = 3.max(((s as f64 * 0.25) as i32) | 1);
    let r = blk / 2;
    let w1 = su + 1;
    sc.integ.clear();
    sc.integ.resize(w1 * w1, 0.0);
    for y in 0..su {
        let mut rowsum = 0.0f64;
        let frow = &p.flat[y * su..(y + 1) * su];
        let (prev, cur) = sc.integ.split_at_mut((y + 1) * w1);
        let iprev = &prev[y * w1..];
        let irow = &mut cur[..w1];
        irow[0] = 0.0;
        for x in 0..su {
            rowsum += frow[x] as f64;
            irow[x + 1] = iprev[x + 1] + rowsum;
        }
    }

    sc.cx0v.clear();
    sc.cx1v.clear();
    for x in 0..s {
        sc.cx0v.push(0.max(x - r));
        sc.cx1v.push((s - 1).min(x + r));
    }
    let c = 7.0f64;
    for y in 0..s {
        let cy0 = 0.max(y - r);
        let cy1 = (s - 1).min(y + r);
        let hi = &sc.integ[(cy1 as usize + 1) * w1..(cy1 as usize + 2) * w1];
        let lo = &sc.integ[cy0 as usize * w1..(cy0 as usize + 1) * w1];
        let rows = (cy1 - cy0 + 1) as f64;
        let frow = &p.flat[y as usize * su..(y as usize + 1) * su];
        let brow = &mut bw[y as usize * su..(y as usize + 1) * su];
        for x in 0..su {
            let a = sc.cx0v[x] as usize;
            let b = sc.cx1v[x] as usize + 1;
            let sum = (hi[b] - lo[b]) - (hi[a] - lo[a]);
            let mean = sum / (rows * (b - a) as f64);
            brow[x] = if (frow[x] as f64) < mean - c { 255 } else { 0 };
        }
    }
}

/// Bilinear read of the plane the mask was thresholded from; NaN off-patch.
#[inline]
fn profile_at(p: &EdgeProfile, x: f64, y: f64) -> f64 {
    let sm1 = (p.s - 1) as f64;
    if !(x >= 0.0 && y >= 0.0 && x <= sm1 && y <= sm1) {
        return f64::NAN;
    }
    let x0 = x as i32;
    let y0 = y as i32;
    let x1 = if x0 + 1 < p.s { x0 + 1 } else { x0 };
    let y1 = if y0 + 1 < p.s { y0 + 1 } else { y0 };
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let s = p.s as usize;
    let at = |xi: i32, yi: i32| -> f64 {
        let k = yi as usize * s + xi as usize;
        match (p.gray, p.flat) {
            (Some(g), _) => g[k] as f64,
            (None, Some(f)) => f[k] as f64,
            (None, None) => f64::NAN,
        }
    };
    at(x0, y0) * (1.0 - fx) * (1.0 - fy)
        + at(x1, y0) * fx * (1.0 - fy)
        + at(x0, y1) * (1.0 - fx) * fy
        + at(x1, y1) * fx * fy
}

/// Distance, px, from the contour pixel at (x, y) OUTWARD along the unit normal (nx, ny)
/// to the edge's true 50 % crossing.
///
/// Not a threshold crossing: the threshold is Otsu's global split, which on a patch that is
/// mostly tag sits well off the midpoint of THIS edge's two levels, so interpolating to it
/// biases the answer toward the dark side. Instead the profile is integrated. With lo/hi
/// the plateau levels either side, (hi - f)/(hi - lo) is the dark COVERAGE at each sample,
/// and its integral across a window that contains the whole transition is the distance from
/// the window's start to the crossing: exact for any ramp shape or width, and naming no
/// level. Sampling at the pixel centres k = -K..K is the midpoint rule over
/// [-K-0.5, K+0.5].
///
/// NaN when the window runs off the patch, when the levels do not bracket the mask's own
/// threshold (an adjacent module, or an occluder), when they are within
/// `K_EDGE_PROBE_DN` of each other, or when the answer lands outside (-0.5, 2) px. The
/// caller falls back to `K_EDGE_OFFSET_PX` for that point.
fn edge_crossing(p: &EdgeProfile, x: f64, y: f64, nx: f64, ny: f64) -> f64 {
    const K: i32 = K_EDGE_PROBE_SPAN;
    let mut v = [0.0f64; (2 * K + 1) as usize];
    for k in -K..=K {
        let s = profile_at(p, x + nx * k as f64, y + ny * k as f64);
        if s.is_nan() {
            return f64::NAN;
        }
        v[(k + K) as usize] = s;
    }
    let lo = v[0];
    let hi = v[(2 * K) as usize];
    if !(lo < p.thr) || !(hi >= p.thr) || hi - lo < K_EDGE_PROBE_DN {
        return f64::NAN;
    }
    let mut d = -(K as f64 + 0.5);
    for x in v.iter() {
        let f = ((hi - x) / (hi - lo)).clamp(0.0, 1.0);
        d += f;
    }
    if d > -0.5 && d < 2.0 {
        d
    } else {
        f64::NAN
    }
}

/// Per-side total-least-squares line fit over the contour run between corners, then
/// intersect adjacent lines. Per-side selection: contour points inside `pb`'s t range and
/// band; at least 6 points or the side fails. TLS = the smallest eigenvector of the 2x2
/// scatter of the mean-centred selected points; the normal is oriented outward with
/// `n.(mean - quad_centre) > 0`. Each fitted line is pushed `K_EDGE_OFFSET_PX` OUTWARD along
/// its own normal (or, on a clipped patch and a degenerate edge, by the measured offset)
/// before intersecting. False when any side has < 6 selected points, is shorter than 4 px,
/// or two adjacent fitted lines are (near-)parallel.
pub fn polish_quad(
    c: &PatchContours,
    idx: i32,
    q: &mut [[f64; 2]; 4],
    prof: Option<&EdgeProfile>,
    pb: &PolishBand,
    sc: &mut Scratch,
) -> bool {
    let begin = c.start[idx as usize];
    let end = c.start[idx as usize + 1];
    let (mut ctrx, mut ctry) = (0.0f64, 0.0f64);
    for p in q.iter() {
        ctrx += p[0] * 0.25;
        ctry += p[1] * 0.25;
    }

    #[derive(Clone, Copy, Default)]
    struct Line {
        nx: f64,
        ny: f64,
        c: f64,
    }
    let mut lines = [Line::default(); 4];

    for e in 0..4 {
        let ax = q[e][0];
        let ay = q[e][1];
        let bx = q[(e + 1) % 4][0];
        let by = q[(e + 1) % 4][1];
        let dx = bx - ax;
        let dy = by - ay;
        let l = (dx * dx + dy * dy).sqrt();
        if l < 4.0 {
            return false;
        }
        let ux = dx / l;
        let uy = dy / l;
        let nx = -uy;
        let ny = ux;
        let tol = 1.6f64.max(pb.band_frac * l);

        sc.q.clear();
        for k in begin..end {
            let px = c.pts[2 * k as usize] as f64;
            let py = c.pts[2 * k as usize + 1] as f64;
            let t = ((px - ax) * ux + (py - ay) * uy) / l;
            let dist = (px - ax) * nx + (py - ay) * ny;
            if t > pb.t_lo && t < pb.t_hi && dist.abs() < tol {
                sc.q.push([px, py]);
            }
        }
        if sc.q.len() < 6 {
            return false;
        }

        let (mut mx, mut my) = (0.0f64, 0.0f64);
        for pt in sc.q.iter() {
            mx += pt[0];
            my += pt[1];
        }
        mx /= sc.q.len() as f64;
        my /= sc.q.len() as f64;
        let (mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64);
        for pt in sc.q.iter() {
            let zx = pt[0] - mx;
            let zy = pt[1] - my;
            sxx += zx * zx;
            sxy += zx * zy;
            syy += zy * zy;
        }
        let (mut nnx, mut nny) = smallest_eigvec2(sxx, sxy, syy);
        if nnx * (mx - ctrx) + nny * (my - ctry) < 0.0 {
            nnx = -nnx;
            nny = -nny;
        }
        // How much sub-pixel phase does this edge's staircase carry? The residuals of the
        // selected points about the fitted line are exactly zero for an axis-aligned or
        // 45 degree run and >= 1/sqrt(2) for any other. Only the degenerate ones need --
        // and only they get -- a measured offset in place of K_EDGE_OFFSET_PX.
        let mut off = K_EDGE_OFFSET_PX;
        if let Some(prof) = prof {
            if prof.s > 0 {
                let (mut rlo, mut rhi) = (1e300f64, -1e300f64);
                for pt in sc.q.iter() {
                    let r = (pt[0] - mx) * nnx + (pt[1] - my) * nny;
                    if r < rlo {
                        rlo = r;
                    }
                    if r > rhi {
                        rhi = r;
                    }
                }
                if rhi - rlo < K_EDGE_FLAT_SPREAD {
                    let mut sum = 0.0f64;
                    let mut nsub = 0usize;
                    for pt in sc.q.iter() {
                        let d = edge_crossing(prof, pt[0], pt[1], nnx, nny);
                        if !d.is_nan() {
                            sum += d;
                            nsub += 1;
                        }
                    }
                    // Half the run, and never fewer points than the fit itself needs: a
                    // partly unmeasurable edge keeps the blind offset rather than an
                    // average over whichever points happened to work.
                    if nsub >= 6 && nsub * 2 >= sc.q.len() {
                        off = sum / nsub as f64;
                    }
                }
            }
        }
        lines[e] = Line {
            nx: nnx,
            ny: nny,
            c: nnx * mx + nny * my + off,
        };
    }

    let mut out = [[0.0f64; 2]; 4];
    for i in 0..4 {
        let l1 = &lines[(i + 3) % 4];
        let l2 = &lines[i];
        let det = l1.nx * l2.ny - l2.nx * l1.ny;
        if det.abs() < 1e-3 {
            return false;
        }
        out[i][0] = (l1.c * l2.ny - l2.c * l1.ny) / det;
        out[i][1] = (l1.nx * l2.c - l2.nx * l1.c) / det;
    }
    *q = out;
    true
}

// ---- 4-line fit ---------------------------------------------------------------------
//
// quad_from_contour's Douglas-Peucker 4-gon assumes the contour IS a quadrilateral. A bright
// occluder biting a corner, a notch in a side, or a dark occluder merged into the blob all
// break that. quad_from_lines instead MEASURES straight runs of the contour (split-and-merge
// at a tolerance in patch px, collinear runs grouped into one line), then picks the 4 lines,
// taken in contour order and intersected pairwise, whose quad explains the contour best:
// every side must be supported by contour points along it, >= 3 sides by >= 30 % of their
// length, and contour points outside the quad cost. A corner is therefore ALWAYS the
// intersection of two measured sides; with fewer than 4 such sides the call fails and
// nothing is extrapolated. For CANDIDATES only: the strict decode still decides what the
// marker is.

/// A measured straight run (or group of collinear runs).
#[derive(Debug, Clone, Copy)]
struct FLine {
    /// A point on the line (centroid of its points).
    mx: f64,
    my: f64,
    /// Unit direction.
    dx: f64,
    dy: f64,
    /// Support: summed chord length of its runs.
    len: f64,
    /// Contour index of its longest run's midpoint.
    mid: i32,
    /// That longest run's chord length.
    best: f64,
}

impl Default for FLine {
    fn default() -> Self {
        FLine {
            mx: 0.0,
            my: 0.0,
            dx: 1.0,
            dy: 0.0,
            len: 0.0,
            mid: 0,
            best: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Run {
    a: i32,
    b: i32,
    len: f64,
    ux: f64,
    uy: f64,
}

#[derive(Debug, Clone, Default)]
struct Grp {
    l: FLine,
    rg: Vec<[i32; 2]>,
}

#[inline]
fn fl_dist(l: &FLine, x: f64, y: f64) -> f64 {
    (x - l.mx) * (-l.dy) + (y - l.my) * l.dx
}

/// TLS fit over the index ranges [a, b] (cyclic, trimmed by `trim` at both ends).
fn fl_fit(p: &[[f64; 2]], n: i32, rngs: &[[i32; 2]], trim: i32, l: &mut FLine) {
    let (mut mx, mut my) = (0.0f64, 0.0f64);
    let mut cnt = 0i32;
    for r in rngs {
        for i in r[0] + trim..=r[1] - trim {
            let pt = &p[(i % n) as usize];
            mx += pt[0];
            my += pt[1];
            cnt += 1;
        }
    }
    if cnt < 3 {
        return;
    }
    mx /= cnt as f64;
    my /= cnt as f64;
    let (mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for r in rngs {
        for i in r[0] + trim..=r[1] - trim {
            let pt = &p[(i % n) as usize];
            let zx = pt[0] - mx;
            let zy = pt[1] - my;
            sxx += zx * zx;
            sxy += zx * zy;
            syy += zy * zy;
        }
    }
    let (nx, ny) = smallest_eigvec2(sxx, sxy, syy);
    l.mx = mx;
    l.my = my;
    l.dx = -ny;
    l.dy = nx;
}

/// Distance from p to the segment a-b.
#[inline]
fn seg_dist(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let l2 = dx * dx + dy * dy;
    let mut t = if l2 > 0.0 {
        ((px - ax) * dx + (py - ay) * dy) / l2
    } else {
        0.0
    };
    t = t.clamp(0.0, 1.0);
    (px - (ax + t * dx)).hypot(py - (ay + t * dy))
}

/// How badly quad `q` (any order) explains contour `idx`: the fraction of its points
/// farther than `max(2 px, 3 % of the quad's short side)` from the quad's boundary. ~0 for
/// a clean marker, > 0.04 for a bitten or merged one.
pub fn contour_misfit(c: &PatchContours, idx: i32, q: &[[f64; 2]; 4]) -> f64 {
    let begin = c.start[idx as usize];
    let end = c.start[idx as usize + 1];
    let mut sh = 1e300f64;
    for i in 0..4 {
        sh = sh.min((q[(i + 1) & 3][0] - q[i][0]).hypot(q[(i + 1) & 3][1] - q[i][1]));
    }
    let tol = 2.0f64.max(0.03 * sh);
    let mut far = 0;
    for k in begin..end {
        let px = c.pts[2 * k as usize] as f64;
        let py = c.pts[2 * k as usize + 1] as f64;
        let mut d = 1e300f64;
        for i in 0..4 {
            d = d.min(seg_dist(
                px,
                py,
                q[i][0],
                q[i][1],
                q[(i + 1) & 3][0],
                q[(i + 1) & 3][1],
            ));
        }
        if d > tol {
            far += 1;
        }
    }
    far as f64 / (end - begin).max(1) as f64
}

/// Contours `select_marker_contour` rejects ONLY for solidity or for not containing the
/// patch centre (not a hole border, >= 12 points, area in [2 %, 92 %], filling >= 25 % of their
/// bounding box), nearest-centroid first, at most `cap` indices written to `out`. Returns
/// how many. The caller must demand a 4-line fit before it believes one.
pub fn select_marker_contours_relaxed(c: &PatchContours, s: i32, out: &mut [i32], cap: i32) -> i32 {
    let n = c.start.len() as i32 - 1;
    let ctr = (s as f64 - 1.0) * 0.5;
    let area_lo = 0.02 * s as f64 * s as f64;
    let area_hi = 0.92 * s as f64 * s as f64;
    let mut best = [(0.0f64, 0i32); 8];
    let mut nb = 0usize;
    for i in 0..n {
        let iu = i as usize;
        if c.hole[iu] {
            continue;
        }
        let begin = c.start[iu];
        let end = c.start[iu + 1];
        if ((end - begin) as i32) < 12 {
            continue;
        }
        let area = polygon_area(&c.pts, begin, end);
        if area < area_lo || area > area_hi {
            continue;
        }
        {
            // a marker (even merged with an occluder) is a compact blob: it fills >= 25 % of its bbox
            let (mut bx0, mut by0, mut bx1, mut by1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for k in begin..end {
                let x = c.pts[2 * k as usize];
                let y = c.pts[2 * k as usize + 1];
                bx0 = bx0.min(x);
                bx1 = bx1.max(x);
                by0 = by0.min(y);
                by1 = by1.max(y);
            }
            if area < 0.25 * (bx1 - bx0 + 1) as f64 * (by1 - by0 + 1) as f64 {
                continue;
            }
        }
        let (cx, cy) = polygon_centroid(&c.pts, begin, end);
        let d2 = (cx - ctr) * (cx - ctr) + (cy - ctr) * (cy - ctr);
        if nb < 8 {
            best[nb] = (d2, i);
            nb += 1;
        } else {
            let mut w = 0usize;
            for k in 1..8 {
                if best[k].0 > best[w].0 {
                    w = k;
                }
            }
            if d2 < best[w].0 {
                best[w] = (d2, i);
            }
        }
    }
    best[..nb].sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let m = (nb as i32).min(cap).min(out.len() as i32);
    for k in 0..m as usize {
        out[k] = best[k].1;
    }
    m
}

/// The ring gate: true unless a measured side's border cells read < `K_LINES_SIDE_DARK_MIN`
/// dark. A line-fit quad must have the black ring on EVERY side; a side fitted along an
/// occluder's edge is how a measured line goes wrong. Unmeasurable passes.
pub fn lines_ring_ok(v: &View, dict: i32, p: &Patch, q: &[[f64; 2]; 4]) -> bool {
    let r = [
        patch_to_ray(p, q[0][0], q[0][1]),
        patch_to_ray(p, q[1][0], q[1][1]),
        patch_to_ray(p, q[2][0], q[2][1]),
        patch_to_ray(p, q[3][0], q[3][1]),
    ];
    let m = border_min_side_dark(v, dict, &r, K_BRIDGE_PROBE_SUB);
    m < 0.0 || m as f64 >= K_LINES_SIDE_DARK_MIN
}

/// Fits 4 measured lines to contour `idx` (see the section note above). False when fewer
/// than 4 sides can be measured or no 4-subset explains the contour (score < 0.40).
pub fn quad_from_lines(
    c: &PatchContours,
    idx: i32,
    q: &mut [[f64; 2]; 4],
    sc: &mut Scratch,
) -> bool {
    let begin = c.start[idx as usize];
    let end = c.start[idx as usize + 1];
    let n = (end - begin) as i32;
    if n < 16 {
        return false;
    }
    let nu = n as usize;
    sc.lp.clear();
    let (mut x0, mut y0, mut x1, mut y1) = (1e300f64, 1e300f64, -1e300f64, -1e300f64);
    for i in 0..nu {
        let px = c.pts[2 * (begin as usize + i)] as f64;
        let py = c.pts[2 * (begin as usize + i) + 1] as f64;
        sc.lp.push([px, py]);
        x0 = x0.min(px);
        x1 = x1.max(px);
        y0 = y0.min(py);
        y1 = y1.max(py);
    }
    let pp = &sc.lp;
    let area = polygon_area(&c.pts, begin, end);
    let side_est = area.max(1.0).sqrt();
    let tol = 3.0f64.min(1.0f64.max(0.012 * side_est));

    // ---- 1. split-and-merge: DP at `tol` on the two arcs between the extreme points
    let mut ea = 0i32;
    let mut bd = -1.0f64;
    for i in 1..nu {
        let dx = pp[i][0] - pp[0][0];
        let dy = pp[i][1] - pp[0][1];
        let d = dx * dx + dy * dy;
        if d > bd {
            bd = d;
            ea = i as i32;
        }
    }
    let mut eb = ea;
    bd = -1.0;
    for i in 0..nu {
        let dx = pp[i][0] - pp[ea as usize][0];
        let dy = pp[i][1] - pp[ea as usize][1];
        let d = dx * dx + dy * dy;
        if d > bd {
            bd = d;
            eb = i as i32;
        }
    }
    let len1 = (eb - ea + n) % n;
    let len2 = n - len1;
    if len1 < 1 || len2 < 1 {
        return false;
    }
    sc.lkeep.clear();
    sc.lkeep.resize(nu, 0);
    for a in 0..2 {
        let st = if a == 0 { ea } else { eb };
        let len = if a == 0 { len1 } else { len2 };
        sc.larc.clear();
        for i in 0..=len {
            sc.larc.push(pp[((st + i) % n) as usize]);
        }
        sc.lkp.clear();
        sc.lkp.resize(len as usize + 1, 0);
        dp_run(&sc.larc, len, tol, &mut sc.lkp, &mut sc.stack);
        for i in 0..len {
            if sc.lkp[i as usize] != 0 {
                sc.lkeep[((st + i) % n) as usize] = 1;
            }
        }
    }
    sc.lv.clear();
    for i in 0..n {
        let g = (ea + i) % n;
        if sc.lkeep[g as usize] != 0 {
            sc.lv.push(i); // offsets from ea
        }
    }
    let m = sc.lv.len();
    if m < 4 {
        return false;
    }

    // ---- 2. runs -> lines; collinear runs grouped
    sc.runs.clear();
    for k in 0..m {
        let a = sc.lv[k];
        let b = if k + 1 < m {
            sc.lv[k + 1]
        } else {
            sc.lv[0] + n
        }; // offsets from ea
        let pa = &pp[((ea + a) % n) as usize];
        let pb = &pp[((ea + b) % n) as usize];
        let l = (pb[0] - pa[0]).hypot(pb[1] - pa[1]);
        if l < 5.0 * tol || b - a < 6 {
            continue;
        }
        sc.runs.push(Run {
            a,
            b,
            len: l,
            ux: (pb[0] - pa[0]) / l,
            uy: (pb[1] - pa[1]) / l,
        });
    }
    if sc.runs.len() < 4 {
        return false;
    }
    sc.runs.sort_by(|r, s| s.len.total_cmp(&r.len));
    sc.groups.clear();
    const K_SIN8: f64 = 0.139;
    for ri in 0..sc.runs.len() {
        let r = sc.runs[ri];
        let mut g: i32 = -1;
        let pa = &pp[((ea + r.a) % n) as usize];
        let pb = &pp[((ea + r.b) % n) as usize];
        let mut k = 0usize;
        while k < sc.groups.len() && g < 0 {
            let l = &sc.groups[k].l;
            if (r.ux * l.dy - r.uy * l.dx).abs() < K_SIN8
                && fl_dist(l, pa[0], pa[1]).abs() < 2.0 * tol + 0.5
                && fl_dist(l, pb[0], pb[1]).abs() < 2.0 * tol + 0.5
            {
                g = k as i32;
            }
            k += 1;
        }
        let trim = 3.min((r.b - r.a) / 6);
        if g < 0 {
            sc.groups.push(Grp::default());
            g = sc.groups.len() as i32 - 1;
            let gr = &mut sc.groups[g as usize];
            gr.l.best = r.len;
            gr.l.mid = (ea + r.a + (r.b - r.a) / 2) % n;
        }
        let gr = &mut sc.groups[g as usize];
        gr.rg.push([ea + r.a, ea + r.b]);
        gr.l.len += r.len;
        // fit (indices ea+a .. ea+b may exceed n: fl_fit wraps with % n)
        fl_fit(pp, n, &gr.rg, trim, &mut gr.l);
    }
    // keep the strongest few groups
    sc.groups.sort_by(|a, b| b.l.len.total_cmp(&a.l.len));
    let mut mm = 0usize;
    for g in sc.groups.iter() {
        if g.l.len < 0.12 * side_est {
            break;
        }
        mm += 1;
        if mm == 7 {
            break;
        }
    }
    if mm < 4 {
        return false;
    }
    let gl = |i: usize| &sc.groups[i].l;

    // ---- 3. best 4-subset in contour order
    let mut best_score = -1e300f64;
    let mut bestq = [[0.0f64; 2]; 4];
    let mut have = false;
    let ext = (x1 - x0).max(y1 - y0);
    'subsets: for a in 0..mm {
        for b in a + 1..mm {
            for cc in b + 1..mm {
                for d in cc + 1..mm {
                    let mut ord = [a, b, cc, d];
                    for i in 1..4 {
                        let mut j = i;
                        while j > 0 && gl(ord[j]).mid < gl(ord[j - 1]).mid {
                            ord.swap(j, j - 1);
                            j -= 1;
                        }
                    }
                    let mut qq = [[0.0f64; 2]; 4];
                    let mut ok = true;
                    for i in 0..4 {
                        let l1 = gl(ord[i]);
                        let l2 = gl(ord[(i + 1) & 3]);
                        let det = l1.dx * l2.dy - l1.dy * l2.dx; // sin of the angle between them
                        if det.abs() < 0.42 {
                            ok = false;
                            break;
                        }
                        // intersection of  p1 + s d1 = p2 + t d2
                        let s = ((l2.mx - l1.mx) * l2.dy - (l2.my - l1.my) * l2.dx) / det;
                        let k = (i + 1) & 3;
                        qq[k][0] = l1.mx + s * l1.dx;
                        qq[k][1] = l1.my + s * l1.dy;
                        if qq[k][0] < x0 - ext
                            || qq[k][0] > x1 + ext
                            || qq[k][1] < y0 - ext
                            || qq[k][1] > y1 + ext
                        {
                            ok = false;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    // convex, consistent winding, sides of comparable length
                    let mut sgn = 0.0f64;
                    let mut lmax = 0.0f64;
                    let mut lmin = 1e300f64;
                    let mut ll = [0.0f64; 4];
                    for i in 0..4 {
                        if !ok {
                            break;
                        }
                        let ux = qq[(i + 1) & 3][0] - qq[i][0];
                        let uy = qq[(i + 1) & 3][1] - qq[i][1];
                        let vx = qq[(i + 2) & 3][0] - qq[(i + 1) & 3][0];
                        let vy = qq[(i + 2) & 3][1] - qq[(i + 1) & 3][1];
                        let cr = ux * vy - uy * vx;
                        if sgn == 0.0 {
                            sgn = cr;
                        } else if (cr > 0.0) != (sgn > 0.0) {
                            ok = false;
                        }
                        ll[i] = ux.hypot(uy);
                        lmax = lmax.max(ll[i]);
                        lmin = lmin.min(ll[i]);
                    }
                    if !ok || lmin < 0.15 * lmax || lmin < 0.2 * side_est {
                        continue;
                    }
                    // cheap optimistic bound first: a side's support cannot exceed the
                    // summed chord length of its group's runs (+ slack for runs too short
                    // to keep)
                    {
                        let (mut fs_est, mut fmin_est, mut n30_est) = (0.0f64, 1.0f64, 0);
                        for i in 0..4 {
                            let e = 1.0f64.min((gl(ord[i]).len + 6.0 * tol) / ll[i]);
                            fs_est += e;
                            fmin_est = fmin_est.min(e);
                            n30_est += (e >= 0.30) as i32;
                        }
                        if n30_est < 3 || fmin_est < 0.12 || fs_est / 4.0 <= best_score {
                            continue;
                        }
                    }
                    // support per side + points outside
                    let mut frac = [0.0f64; 4];
                    let mut out = 0i32;
                    for i in 0..4 {
                        let ax = qq[i][0];
                        let ay = qq[i][1];
                        let ux = (qq[(i + 1) & 3][0] - ax) / ll[i];
                        let uy = (qq[(i + 1) & 3][1] - ay) / ll[i];
                        let mut cnt = 0i32;
                        for k in 0..nu {
                            let t = (pp[k][0] - ax) * ux + (pp[k][1] - ay) * uy;
                            let dn = (pp[k][0] - ax) * (-uy) + (pp[k][1] - ay) * ux; // left normal
                            if t >= -tol && t <= ll[i] + tol && dn.abs() <= 2.0 * tol {
                                cnt += 1;
                            }
                        }
                        frac[i] = 1.0f64.min(cnt as f64 / ux.abs().max(uy.abs()) / ll[i]);
                    }
                    for k in 0..nu {
                        // outside the (convex) quad by > 2 tol
                        let mut worst = -1e300f64;
                        for i in 0..4 {
                            let ax = qq[i][0];
                            let ay = qq[i][1];
                            let ux = (qq[(i + 1) & 3][0] - ax) / ll[i];
                            let uy = (qq[(i + 1) & 3][1] - ay) / ll[i];
                            let dn = (pp[k][0] - ax) * (-uy) + (pp[k][1] - ay) * ux;
                            worst = worst.max(if sgn > 0.0 { -dn } else { dn });
                        }
                        if worst > 2.0 * tol {
                            out += 1;
                        }
                    }
                    let (mut n30, mut fmin, mut fsum) = (0, 1.0f64, 0.0f64);
                    for i in 0..4 {
                        n30 += (frac[i] >= 0.30) as i32;
                        fmin = fmin.min(frac[i]);
                        fsum += frac[i];
                    }
                    if n30 < 3 || fmin < 0.12 {
                        continue;
                    }
                    let score = fsum / 4.0 - 0.5 * out as f64 / n as f64;
                    if score > best_score {
                        best_score = score;
                        bestq = qq;
                        have = true;
                    }
                    // Early stop: the FIRST subset reaching 0.9 in enumeration
                    // (support-sorted group) order is taken, not the best of all of them;
                    // a higher-scoring later subset is never looked at.
                    if best_score >= 0.9 {
                        break 'subsets;
                    }
                }
            }
        }
    }
    if !have || best_score < 0.40 {
        return false;
    }
    *q = bestq;
    true
}

/// Which plane (if any) `polish_quad` may measure a degenerate edge's offset from.
#[derive(Clone, Copy)]
enum ProfKind {
    None,
    Gray,
    Flat,
}

#[derive(Clone, Copy)]
struct ProfSpec {
    kind: ProfKind,
    thr: f64,
}

impl ProfSpec {
    fn make<'a>(&self, p: &'a Patch) -> EdgeProfile<'a> {
        let n = (p.s * p.s) as usize;
        match self.kind {
            ProfKind::None => EdgeProfile::default(),
            ProfKind::Gray => EdgeProfile {
                gray: Some(&p.img[..n]),
                flat: None,
                s: p.s,
                thr: self.thr,
            },
            ProfKind::Flat => EdgeProfile {
                gray: None,
                flat: Some(&p.flat[..n]),
                s: p.s,
                thr: self.thr,
            },
        }
    }
}

/// Threshold ladder step: binarise into `bw` and trace into `c`. The plane and threshold
/// `polish_quad` measures a degenerate edge's offset from are returned only on a patch a
/// per-pass cap clipped, and never for mode 2, whose threshold is per-pixel and so has no
/// single level for the profile's plateaus to bracket.
#[allow(clippy::too_many_arguments)]
fn prepare(
    p: &mut Patch,
    bw: &mut [u8],
    c: &mut PatchContours,
    sc: &mut Scratch,
    flat_ready: &mut bool,
    mode: i32,
    min_len: i32,
) -> ProfSpec {
    let s = p.s;
    let n = (s * s) as usize;
    let mut spec = ProfSpec {
        kind: ProfKind::None,
        thr: 0.0,
    };
    if mode == 0 {
        let thr = otsu_hist8_u8(&p.img[..n]);
        patch_binarise_raw(p, thr, bw);
        if p.clipped {
            spec = ProfSpec {
                kind: ProfKind::Gray,
                thr,
            };
        }
    } else {
        if !*flat_ready {
            patch_flatten(p, sc);
            *flat_ready = true;
        }
        if mode == 1 {
            let thr = otsu_hist8(&p.flat[..n]);
            patch_binarise(p, thr, bw);
            if p.clipped {
                spec = ProfSpec {
                    kind: ProfKind::Flat,
                    thr,
                };
            }
        } else {
            adaptive_binarise(p, bw, sc);
        }
    }
    trace_contours(&bw[..n], s, c, min_len, sc);
    spec
}

/// Order, polish and bound-check a quad. A quad more than 2 px outside the patch is
/// rejected.
fn polish_finish(
    c: &PatchContours,
    sc: &mut Scratch,
    qq: &mut [[f64; 2]; 4],
    ci: i32,
    lines: bool,
    prof: &EdgeProfile,
    s: i32,
) -> bool {
    order_quad(qq);
    let mut qp = *qq;
    let band = if lines {
        K_LINE_FIT_BAND
    } else {
        PolishBand::default()
    };
    if polish_quad(
        c,
        ci,
        &mut qp,
        if prof.s > 0 { Some(prof) } else { None },
        &band,
        sc,
    ) {
        order_quad(&mut qp);
        *qq = qp;
    }
    for i in 0..4 {
        for k in 0..2 {
            if qq[i][k] < -2.0 || qq[i][k] > s as f64 + 1.0 {
                return false;
            }
        }
    }
    true
}

/// Otsu -> contour -> select -> quad -> polish, with threshold retries on failure.
///
/// THREE modes, in this order:
///   0  Otsu on the RAW patch                    -- the reference's own primary path
///   1  Otsu on the FLATTENED patch              -- the illumination-gradient fix
///   2  adaptive mean on the flattened patch     -- the reference's own fallback
///
/// Mode 0 is first because patch_flatten is NOT free of side effects on a well-lit patch:
/// sigma = S/6 is wide enough that a marker filling much of the patch biases its own
/// "illumination" estimate, brightening the marker's interior, which moves the Otsu split
/// and hence the traced contour (0.61-2.25 px_eq from the reference's bridge quad on real
/// frames; raw-Otsu-first reproduces it to <= 0.047 px_eq). Nothing is lost on gradient
/// cases: there raw Otsu finds NO acceptable contour at all, so mode 1 runs exactly as
/// before. Modes 1-2 also make patch_flatten LAZY.
///
/// `n_modes` limits the escalation: 3 is the full ladder, 1 is raw Otsu alone (the screen).
///
/// Phase 1 is strict selection over the whole ladder. The legacy 4-gon is the only quad
/// for a contour it explains. A bitten / notched / merged contour gets the 4-line fit as
/// the PRIMARY quad when it passes the ring gate; the legacy 4-gon, when it exists, is
/// returned as the ALTERNATE (an occluder that protrudes past a corner makes the legacy
/// envelope the quad the refiner agrees with), and replaces a line fit the gate rejects.
/// Phase 2 (relaxed selection of a contour the strict one rejects) runs only when
/// `relaxed_left` is given, i.e. when the id is known, after the whole ladder failed.
#[allow(clippy::too_many_arguments)]
fn bridge_find_quad(
    p: &mut Patch,
    bw: &mut [u8],
    c: &mut PatchContours,
    sc: &mut Scratch,
    v: &View,
    dict: i32,
    q_out: &mut [[f64; 2]; 4],
    n_modes: i32,
    mut alt_out: Option<(&mut [[f64; 2]; 4], &mut bool)>,
    relaxed_left: Option<&mut i32>,
    mut prov_out: Option<&mut BridgeProv>,
) -> bool {
    let s = p.s;
    let mut flat_ready = false;

    // Any contour select_marker_contour could possibly accept encloses at least 0.02*S^2,
    // and a traced boundary's steps are at most sqrt(2) long, so its point count obeys
    // n >= sqrt(2*pi*0.02)*S = 0.3545*S. Handing that to trace_contours as `minlen`
    // discards, at zero risk, exactly the contours the selection would have thrown away
    // anyway -- and with them the O(n^2) parent-resolution pass and the per-contour hull.
    // (A dropped contour cannot be an ancestor of a selectable one either: a parent's area
    // is strictly larger and the bound is monotone in area.)
    let min_contour_len = 12.max((0.3544 * s as f64) as i32);

    // ---- phase 1
    for mode in 0..n_modes {
        let spec = prepare(p, bw, c, sc, &mut flat_ready, mode, min_contour_len);
        let prof = spec.make(p);
        let idx0 = select_marker_contour(c, s, 0.80, sc);
        if idx0 < 0 {
            continue;
        }
        let mut q = [[0.0f64; 2]; 4];
        let mut ql = [[0.0f64; 2]; 4];
        let leg = quad_from_contour(c, idx0, &mut q, sc);
        let want = !leg || contour_misfit(c, idx0, &q) > K_BRIDGE_MISFIT_MAX;
        let lin = want
            && quad_from_lines(c, idx0, &mut ql, sc)
            && polish_finish(c, sc, &mut ql, idx0, true, &prof, s)
            && lines_ring_ok(v, dict, p, &ql);
        let mut legok = false;
        if leg {
            legok = polish_finish(c, sc, &mut q, idx0, false, &prof, s);
        }
        if lin {
            if let Some((alt, has)) = alt_out.as_mut() {
                **has = legok;
                if legok {
                    **alt = q;
                }
            }
            *q_out = ql;
            if let Some(pr) = prov_out.as_mut() {
                **pr = BridgeProv::LineFit;
            }
            return true;
        }
        if legok {
            if let Some((_, has)) = alt_out.as_mut() {
                **has = false;
            }
            *q_out = q;
            if let Some(pr) = prov_out.as_mut() {
                **pr = BridgeProv::Legacy;
            }
            return true;
        }
    }

    // ---- phase 2: relaxed selection (known id only), after the whole ladder failed
    if let Some(rl) = relaxed_left {
        if *rl > 0 {
            for mode in 0..n_modes.min(2) {
                let spec = prepare(p, bw, c, sc, &mut flat_ready, mode, min_contour_len);
                let prof = spec.make(p);
                let mut cand = [0i32; 3];
                let nc = select_marker_contours_relaxed(c, s, &mut cand, 3);
                let mut k = 0;
                while k < nc && *rl > 0 {
                    *rl -= 1;
                    let mut ql = [[0.0f64; 2]; 4];
                    if quad_from_lines(c, cand[k as usize], &mut ql, sc)
                        && polish_finish(c, sc, &mut ql, cand[k as usize], true, &prof, s)
                        && lines_ring_ok(v, dict, p, &ql)
                    {
                        if let Some((_, has)) = alt_out.as_mut() {
                            **has = false;
                        }
                        *q_out = ql;
                        if let Some(pr) = prov_out.as_mut() {
                            **pr = BridgeProv::Relaxed;
                        }
                        return true;
                    }
                    k += 1;
                }
            }
        }
    }
    false
}

/// Per-candidate gnomonic re-seed at full resolution. The SCREEN pass renders pass 1's own
/// framing small, tests the quad's border ring and discards it, so it can only reject. Pass
/// 1 proper renders a patch at `K_BRIDGE_BOX` x the candidate's angular size, thresholds it
/// (Otsu, or a local-mean adaptive threshold if Otsu finds no acceptable contour), extracts
/// and polishes the marker quad; pass 2 re-centres on the recovered quad
/// (`v = normalize(sum of corner rays)`, `r = max angle from v`) and repeats at the tighter
/// `K_BRIDGE_BOX2`. If pass 2 fails, pass 1's result is kept.
pub struct Bridge {
    w: i32,
    h: i32,
    dict: i32,
    patch: Patch,
    bw: Vec<u8>,
    grid: BridgeGrid,
    contours: PatchContours,
    scratch: Scratch,
    alt: [Vec3; 4],
    have_alt: bool,
    prov: BridgeProv,
    lenient: bool,
    relaxed_left: i32,
}

impl Bridge {
    /// `w`, `h` must match the view passed to every `run` call (the scratch is sized once
    /// here, like the proposal stage). `dict` is the dictionary index the screen pass lays
    /// its border ring out for (the cell count is the only thing it uses the dictionary
    /// for); it is fixed for a detector's lifetime.
    pub fn new(w: i32, h: i32, dict: i32) -> Result<Bridge, EqarError> {
        if w <= 0 || h <= 0 {
            return Err(EqarError::Size);
        }
        crate::decode::dict_info(dict)?;
        Ok(Bridge {
            w,
            h,
            dict,
            patch: Patch::new_max(),
            bw: vec![0; (K_BRIDGE_MAX_PATCH * K_BRIDGE_MAX_PATCH) as usize],
            grid: BridgeGrid::new(),
            contours: PatchContours::default(),
            scratch: Scratch::default(),
            alt: [Vec3::default(); 4],
            have_alt: false,
            prov: BridgeProv::Legacy,
            lenient: false,
            relaxed_left: 0,
        })
    }

    /// Candidate (centre ray, angular radius) -> 4 corner rays, in patch order (NOT
    /// TL,TR,BR,BL; that is the refiner's job). `Ok(false)` when the screen pass rejects
    /// the candidate, or when even pass 1 finds no acceptable quad (`out` then keeps any
    /// earlier pass's result). `plan` selects the passes: `K_SCAN_PLAN` (a screen plus the
    /// reference's two) or `K_TRACK_PLAN` (one cheap one). `Err` when the view is invalid
    /// or its size differs from the one given to `new`.
    pub fn run(
        &mut self,
        v: &View,
        centre: Vec3,
        radius: f64,
        out: &mut [Vec3; 4],
        plan: &[BridgePass],
    ) -> Result<bool, EqarError> {
        v.check()?;
        if v.w != self.w || v.h != self.h {
            return Err(EqarError::Size);
        }
        let n_pass = plan.len();
        let mut vc = normalize(&centre);
        let mut r = radius;
        let mut have = false;
        self.have_alt = false;
        self.prov = BridgeProv::Legacy;
        for pass in 0..n_pass {
            let hf = bridge_half_fov(plan[pass].box_, r);

            // A screen only pays for itself when its patch is SMALLER than the pass it
            // guards. For a candidate small enough that pass 1's own patch is no bigger,
            // the screen shares pass 1's box and rate, so it would render the identical
            // patch and find the identical quad: pure duplicated work. Skipping it costs
            // nothing (pass 1's own find_quad is strictly stronger: it keeps all three
            // threshold modes).
            if plan[pass].screen && pass + 1 < n_pass {
                let hf_next = bridge_half_fov(plan[pass + 1].box_, r);
                if patch_side(hf, self.h, plan[pass].oversample, plan[pass].max_side)
                    >= patch_side(
                        hf_next,
                        self.h,
                        plan[pass + 1].oversample,
                        plan[pass + 1].max_side,
                    )
                {
                    continue;
                }
            }

            patch_render(
                v,
                &vc,
                hf,
                &mut self.grid,
                &mut self.patch,
                plan[pass].oversample,
                plan[pass].max_side,
            );
            // patch_flatten is deliberately NOT called here: bridge_find_quad runs it
            // lazily, only if raw-Otsu (mode 0) finds no acceptable contour.
            let mut q = [[0.0f64; 2]; 4];

            // The SCREEN pass. Its quad is used only to ask "does an ArUco marker's black
            // outer ring live here?" and is then DISCARDED: `vc` and `r` stay at the
            // caller's own values, so the two real passes render exactly the framing they
            // would have rendered without a screen.
            if plan[pass].screen {
                // Raw Otsu only: the flatten and adaptive modes cost more on the screen's
                // own patch than the passes they save.
                let relaxed = if self.lenient {
                    Some(&mut self.relaxed_left)
                } else {
                    None
                };
                if !bridge_find_quad(
                    &mut self.patch,
                    &mut self.bw,
                    &mut self.contours,
                    &mut self.scratch,
                    v,
                    self.dict,
                    &mut q,
                    1,
                    None,
                    relaxed,
                    None,
                ) {
                    return Ok(false);
                }
                let pr = [
                    patch_to_ray(&self.patch, q[0][0], q[0][1]),
                    patch_to_ray(&self.patch, q[1][0], q[1][1]),
                    patch_to_ray(&self.patch, q[2][0], q[2][1]),
                    patch_to_ray(&self.patch, q[3][0], q[3][1]),
                ];
                let mut contrast = 0.0f32;
                let frac = border_dark_fraction(
                    v,
                    self.dict,
                    &pr,
                    K_BRIDGE_PROBE_SUB,
                    Some(&mut contrast),
                );
                // frac < 0 is "unmeasurable" (the homography itself failed) and must be
                // treated as a PASS.
                if frac >= 0.0
                    && ((frac as f64) < K_BRIDGE_PROBE_BORDER || contrast < K_MIN_CONTRAST_DN)
                {
                    return Ok(false);
                }
                continue;
            }

            let mut qalt = [[0.0f64; 2]; 4];
            let mut has_alt = false;
            // On failure `out`, `alt` and `have_alt` still hold the previous pass's result.
            let mut prov = BridgeProv::Legacy;
            let relaxed = if self.lenient {
                Some(&mut self.relaxed_left)
            } else {
                None
            };
            if !bridge_find_quad(
                &mut self.patch,
                &mut self.bw,
                &mut self.contours,
                &mut self.scratch,
                v,
                self.dict,
                &mut q,
                3,
                Some((&mut qalt, &mut has_alt)),
                relaxed,
                Some(&mut prov),
            ) {
                break;
            }
            for i in 0..4 {
                out[i] = patch_to_ray(&self.patch, q[i][0], q[i][1]);
            }
            self.prov = prov;
            self.have_alt = has_alt;
            if has_alt {
                for i in 0..4 {
                    self.alt[i] = patch_to_ray(&self.patch, qalt[i][0], qalt[i][1]);
                }
            }
            have = true;
            let sum = Vec3::new(
                out[0].x + out[1].x + out[2].x + out[3].x,
                out[0].y + out[1].y + out[2].y + out[3].y,
                out[0].z + out[1].z + out[2].z + out[3].z,
            );
            vc = normalize(&sum);
            let mut maxang = 0.0f64;
            for o in out.iter() {
                let ang = dot(o, &vc).clamp(-1.0, 1.0).acos();
                if ang > maxang {
                    maxang = ang;
                }
            }
            r = maxang;
        }
        Ok(have)
    }

    /// `run` with the scan plan.
    pub fn run_scan(
        &mut self,
        v: &View,
        centre: Vec3,
        radius: f64,
        out: &mut [Vec3; 4],
    ) -> Result<bool, EqarError> {
        self.run(v, centre, radius, out, &K_SCAN_PLAN)
    }

    /// After a `run` that returned `true`, the legacy 4-gon quad of the pass that produced
    /// `out`, whenever `out` is a 4-line-fit quad and a legacy 4-gon exists for the same
    /// contour (no distance condition: it is always kept). `None` otherwise (and after a
    /// failed `run`).
    pub fn alt_quad(&self) -> Option<[Vec3; 4]> {
        if self.have_alt {
            Some(self.alt)
        } else {
            None
        }
    }

    /// Provenance of the quad the last successful `run` returned.
    pub fn prov(&self) -> BridgeProv {
        self.prov
    }

    /// The relaxed contour selection (4-line fit of a contour the strict selection
    /// rejects) is lenient handling, so it runs ONLY when the id is known (verify).
    pub fn set_lenient(&mut self, on: bool) {
        self.lenient = on;
    }

    /// Re-arms the relaxed selection's per-frame cap.
    pub fn begin_frame(&mut self) {
        self.relaxed_left = K_BRIDGE_RELAXED_PER_FRAME;
    }
}

#[cfg(all(test, target_arch = "aarch64", not(feature = "no-neon")))]
mod neon_tests {
    use crate::simd;

    #[test]
    fn flat_div_row_matches_scalar() {
        let src: Vec<u8> = (0..37u32).map(|i| (i * 53 % 256) as u8).collect();
        let den: Vec<f32> = (0..37u32).map(|i| 1.0 + i as f32 * 3.7).collect();
        let mut got = vec![0f32; 37];
        let done = simd::flat_div_row(&src, &den, &mut got);
        assert_eq!(done, 32);
        for i in 0..done {
            assert_eq!(got[i], 255.0f32 * src[i] as f32 / den[i], "i = {i}");
        }
    }

    #[test]
    fn binarise_kernels_match_scalar() {
        let flat: Vec<f32> = (0..50u32).map(|i| (i * 37 % 300) as f32 * 0.9).collect();
        let mut got = vec![7u8; 50];
        let done = simd::binarise_lt_f32(&flat, 120.5, &mut got);
        assert_eq!(done, 48);
        for i in 0..done {
            assert_eq!(got[i], if flat[i] < 120.5 { 255 } else { 0 });
        }
        let img: Vec<u8> = (0..50u32).map(|i| (i * 91 % 256) as u8).collect();
        let done = simd::binarise_le_u8(&img, 120, &mut got);
        assert_eq!(done, 48);
        for i in 0..done {
            // thr = 120.5 in the scalar form
            assert_eq!(got[i], if (img[i] as f32) < 120.5 { 255 } else { 0 });
        }
    }
}
