//! Dictionary lookup and bit decoding: cell reading under the closed-form 4-ray
//! homography, cold decode (ID unknown), erasure-aware verify decode (ID known), and the
//! border-ring probes.

use crate::codes;
use crate::dict_tables;
use crate::error::EqarError;
#[cfg(not(target_arch = "aarch64"))]
use crate::geometry::{cross, dot};
use crate::geometry::{normalize, ray_to_px, Vec3};
use crate::image::{sample, View};

/// 8x8 sub-samples per cell.
pub const K_SUB_DEFAULT: i32 = 8;
pub const K_SUB_LEAN: i32 = 4;

/// One predefined dictionary.
#[derive(Debug, Clone, Copy)]
pub struct DictInfo {
    /// `n * 4` entries: id-major, then rotation 0..3.
    pub codes: &'static [u64],
    /// Markers in the dictionary.
    pub n: i32,
    /// 4, 5, 6 or 7.
    pub marker_size: i32,
    /// OpenCV's maxCorrectionBits.
    pub max_correction: i32,
    pub name: &'static str,
}

/// `dict` is the dictionary index (4X4_50 = 0 ... 7X7_1000 = 15).
pub fn dict_info(dict: i32) -> Result<DictInfo, EqarError> {
    let e = usize::try_from(dict)
        .ok()
        .and_then(|i| dict_tables::ENTRIES.get(i))
        .ok_or(EqarError::Dict)?;
    Ok(DictInfo {
        codes: e.codes,
        n: e.n,
        marker_size: e.marker_size,
        max_correction: e.max_correction,
        name: e.name,
    })
}

/// Central fraction of each cell that is sampled.
pub const K_CELL_FRAC: f32 = 0.6;

#[inline]
#[allow(clippy::too_many_arguments)]
fn det3(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64, i: f64) -> f64 {
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

// The reference build contracts the winding and determinant arithmetic below into fused
// multiply-adds, in specific places. The sign of `canonical_winding`'s determinant and the
// `|D| < 1e-15` test only matter for (near-)degenerate quads (a repeated corner), where
// they are rounding noise, but the noise decides whether such a candidate is screened out
// before the bridge, which shows in the stage counters. On aarch64 the same fusion is
// written out so those decisions match the reference; elsewhere the plain forms are used.

#[cfg(target_arch = "aarch64")]
fn normalize_fused(a: &Vec3, semi: bool) -> Vec3 {
    // rays 0..2: z*z + (x*x + y*y) with the x*x fused; ray 3 sums x*x + y*y unfused
    let q = if semi {
        a.z.mul_add(a.z, a.x * a.x + a.y * a.y)
    } else {
        a.z.mul_add(a.z, a.x.mul_add(a.x, a.y * a.y))
    };
    let n = q.sqrt();
    let s = if n > 1e-300 { 1.0 / n } else { 0.0 };
    Vec3::new(a.x * s, a.y * s, a.z * s)
}

/// The decode's own handedness fix: normalise, and reverse the array when
/// det[c0,c1,c2] < 0.
fn canonical_winding(rays: &[Vec3; 4]) -> [Vec3; 4] {
    #[cfg(target_arch = "aarch64")]
    let (mut r, det) = {
        let r = [
            normalize_fused(&rays[0], false),
            normalize_fused(&rays[1], false),
            normalize_fused(&rays[2], false),
            normalize_fused(&rays[3], true),
        ];
        let (a, b, c) = (&r[0], &r[1], &r[2]);
        let cx = a.y.mul_add(b.z, -(a.z * b.y));
        let cy = a.z.mul_add(b.x, -(b.z * a.x));
        let cz = b.y.mul_add(a.x, -(a.y * b.x));
        (r, cz.mul_add(c.z, cx.mul_add(c.x, cy * c.y)))
    };
    #[cfg(not(target_arch = "aarch64"))]
    let (mut r, det) = {
        let r = [
            normalize(&rays[0]),
            normalize(&rays[1]),
            normalize(&rays[2]),
            normalize(&rays[3]),
        ];
        let det = dot(&cross(&r[0], &r[1]), &r[2]);
        (r, det)
    };
    if det < 0.0 {
        r = [r[3], r[2], r[1], r[0]];
    }
    r
}

/// D = det[r1, -r2, r3], the denominator of the closed-form homography.
fn homography_denominator(r1: &Vec3, r2: &Vec3, r3: &Vec3) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        let t1 = r3.y.mul_add(r2.z, -(r2.y * r3.z));
        let t2 = r1.y.mul_add(r3.z, -(r3.y * r1.z));
        let t3 = r2.y.mul_add(r1.z, -(r1.y * r2.z));
        r3.x.mul_add(t3, r1.x.mul_add(t1, r2.x * t2))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        det3(r1.x, -r2.x, r3.x, r1.y, -r2.y, r3.y, r1.z, -r2.z, r3.z)
    }
}

/// Canonical marker corners m = (-1,1,1),(1,1,1),(1,-1,1),(-1,-1,1) map to rays r0..r3.
/// Closed form: solve [r1 -r2 r3] a = r0 for (a1,a2,a3) (a0 = 1) by Cramer's rule, then
/// combine into H = [h1|h2|h3] (columns), stored row-major so `H[i*3+0..2]` is row i.
/// `None` when the system is singular.
pub fn homography_closed(rays: &[Vec3; 4]) -> Option<[f64; 9]> {
    let (r0, r1, r2, r3) = (&rays[0], &rays[1], &rays[2], &rays[3]);
    let a = [
        [r1.x, -r2.x, r3.x],
        [r1.y, -r2.y, r3.y],
        [r1.z, -r2.z, r3.z],
    ];
    let d = homography_denominator(r1, r2, r3);
    if d.abs() < 1e-15 {
        return None;
    }
    let id = 1.0 / d;
    let r0v = [r0.x, r0.y, r0.z];
    let a1 = det3(
        r0v[0], a[0][1], a[0][2], r0v[1], a[1][1], a[1][2], r0v[2], a[2][1], a[2][2],
    ) * id;
    let a2 = det3(
        a[0][0], r0v[0], a[0][2], a[1][0], r0v[1], a[1][2], a[2][0], r0v[2], a[2][2],
    ) * id;
    let a3 = det3(
        a[0][0], a[0][1], r0v[0], a[1][0], a[1][1], r0v[1], a[2][0], a[2][1], r0v[2],
    ) * id;
    let r1v = [r1.x, r1.y, r1.z];
    let r2v = [r2.x, r2.y, r2.z];
    let r3v = [r3.x, r3.y, r3.z];
    let mut h = [0.0; 9];
    for i in 0..3 {
        h[i * 3] = 0.5 * (a1 * r1v[i] - r0v[i]);
        h[1 + i * 3] = 0.5 * (a1 * r1v[i] - a2 * r2v[i]);
        h[2 + i * 3] = 0.5 * (a1 * r1v[i] + a3 * r3v[i]);
    }
    Some(h)
}

/// Marker-plane point (x, y) -> normalised ray under the homography `h`.
pub fn plane_to_ray(h: &[f64; 9], x: f64, y: f64) -> Vec3 {
    normalize(&Vec3::new(
        h[0] * x + h[1] * y + h[2],
        h[3] * x + h[4] * y + h[5],
        h[6] * x + h[7] * y + h[8],
    ))
}

/// Cell means over an `ncell` x `ncell` grid under the marker-plane homography `h`: cell
/// centres c = (i+0.5)/ncell*2-1 with the y axis flipped (row 0 = top); `n_sub` x `n_sub`
/// sub-samples uniform in marker-plane coords over the central `K_CELL_FRAC` of each cell
/// -> H -> ray -> ray_to_px -> sample(). `n_sub` must be in 1..=16 and `cellmean` hold
/// `ncell * ncell` values.
fn cell_means(v: &View, h: &[f64; 9], ncell: i32, n_sub: i32, cellmean: &mut [f32]) {
    let half_step = K_CELL_FRAC as f64 / ncell as f64;
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    let fast = v.fmt == crate::image::Fmt::Gray8 && v.lut.is_none();
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    let kxf = v.w as f32 / std::f32::consts::TAU;
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    let kyf = v.h as f32 / std::f32::consts::PI;
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    let gxv: [f32; 16] = {
        let mut g = [0f32; 16];
        for sx in 0..n_sub as usize {
            g[sx] = ((((sx as f64 + 0.5) / n_sub as f64) * 2.0 - 1.0) * half_step) as f32;
        }
        g
    };
    for row in 0..ncell {
        let cy = -(((row as f64 + 0.5) / ncell as f64) * 2.0 - 1.0);
        for col in 0..ncell {
            let cx = ((col as f64 + 0.5) / ncell as f64) * 2.0 - 1.0;
            let mut acc = 0.0f64;
            for sy in 0..n_sub {
                let ty = ((sy as f64 + 0.5) / n_sub as f64) * 2.0 - 1.0;
                let gy = ty * half_step;
                #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
                if fast {
                    // d = H * (cx + gx, cy + gy, 1); only the x term varies across the
                    // four lanes, so the other two collapse into a per-row constant.
                    let yv = (cy + gy) as f32;
                    let fm = crate::image::fmadd;
                    let a = [h[0] as f32, h[3] as f32, h[6] as f32];
                    let b = [
                        fm(h[1] as f32, yv, h[2] as f32),
                        fm(h[4] as f32, yv, h[5] as f32),
                        fm(h[7] as f32, yv, h[8] as f32),
                    ];
                    let mut sx = 0usize;
                    while sx + 4 <= n_sub as usize {
                        acc += crate::simd::cell_sum4(
                            v.data,
                            v.w,
                            v.h,
                            v.stride,
                            &a,
                            &b,
                            cx as f32,
                            &gxv[sx..sx + 4],
                            kxf,
                            kyf,
                        ) as f64;
                        sx += 4;
                    }
                    while sx < n_sub as usize {
                        let ray = plane_to_ray(h, cx + gxv[sx] as f64, cy + gy);
                        let (px, py) = ray_to_px(&ray, v.w, v.h);
                        acc += sample(v, px as f32, py as f32) as f64;
                        sx += 1;
                    }
                    continue;
                }
                for sx in 0..n_sub {
                    let tx = ((sx as f64 + 0.5) / n_sub as f64) * 2.0 - 1.0;
                    let gx = tx * half_step;
                    let ray = plane_to_ray(h, cx + gx, cy + gy);
                    let (px, py) = ray_to_px(&ray, v.w, v.h);
                    acc += sample(v, px as f32, py as f32) as f64;
                }
            }
            cellmean[(row * ncell + col) as usize] = (acc / (n_sub * n_sub) as f64) as f32;
        }
    }
}

/// Shared set-up of the two border probes: grid size, cell means and their Otsu split.
/// `None` when the probe is unmeasurable (unknown dictionary, bad `n_sub`, degenerate quad).
fn border_cells(
    v: &View,
    dict: i32,
    rays: &[Vec3; 4],
    n_sub: i32,
) -> Option<(i32, [f32; 81], f64)> {
    let d = dict_info(dict).ok()?;
    if !(1..=16).contains(&n_sub) {
        return None;
    }
    let ncell = d.marker_size + 2;
    let r = canonical_winding(rays);
    let h = homography_closed(&r)?;
    let mut cellmean = [0f32; 81];
    cell_means(v, &h, ncell, n_sub, &mut cellmean);
    let n2 = (ncell * ncell) as usize;
    let mut scratch = [0f32; 81];
    let thr = crate::bridge::otsu_exact(&cellmean[..n2], &mut scratch[..n2]);
    Some((ncell, cellmean, thr))
}

/// Pre-bridge border probe: sample the `ncell`^2 cell means under the quad's own
/// homography, Otsu-split them, and report what FRACTION of the 4*ncell-4 border cells
/// fall on the dark side. A real marker's ring is black by construction (reads ~1.0 even
/// off a quad a couple of px out); a solid blob, texture patch or bright rectangle reads
/// ~0.5. Returns -1.0 when unmeasurable (degenerate quad), which callers treat as a PASS.
/// `contrast_out` receives the cell-mean spread (max - min) in DN (0 when unmeasurable).
pub fn border_dark_fraction(
    v: &View,
    dict: i32,
    rays: &[Vec3; 4],
    n_sub: i32,
    contrast_out: Option<&mut f32>,
) -> f32 {
    let mut contrast = 0.0f32;
    let res = border_cells(v, dict, rays, n_sub);
    let out = match res {
        None => -1.0f32,
        Some((ncell, cellmean, thr)) => {
            let n2 = (ncell * ncell) as usize;
            let (mut cmin, mut cmax) = (cellmean[0], cellmean[0]);
            for i in 1..n2 {
                if cellmean[i] < cmin {
                    cmin = cellmean[i];
                }
                if cellmean[i] > cmax {
                    cmax = cellmean[i];
                }
            }
            contrast = cmax - cmin;
            let (mut n_border, mut n_dark) = (0, 0);
            for row in 0..ncell {
                let border_row = row == 0 || row == ncell - 1;
                for col in 0..ncell {
                    if !(border_row || col == 0 || col == ncell - 1) {
                        continue;
                    }
                    n_border += 1;
                    // decode's own "on" test, negated
                    if !(cellmean[(row * ncell + col) as usize] as f64 > thr) {
                        n_dark += 1;
                    }
                }
            }
            n_dark as f32 / n_border as f32
        }
    };
    if let Some(c) = contrast_out {
        *c = contrast;
    }
    out
}

/// `border_dark_fraction`'s grid and split, reported per SIDE: the MINIMUM over the four
/// sides of that side's dark fraction (corner cells excluded, so a bitten corner does not
/// condemn its two sides). -1 = unmeasurable.
pub fn border_min_side_dark(v: &View, dict: i32, rays: &[Vec3; 4], n_sub: i32) -> f32 {
    let Some((ncell, cellmean, thr)) = border_cells(v, dict, rays, n_sub) else {
        return -1.0;
    };
    let nc = ncell as usize;
    let mut dark = [0i32; 4];
    for i in 1..nc - 1 {
        dark[0] += !(cellmean[i] as f64 > thr) as i32;
        dark[1] += !(cellmean[(nc - 1) * nc + i] as f64 > thr) as i32;
        dark[2] += !(cellmean[i * nc] as f64 > thr) as i32;
        dark[3] += !(cellmean[i * nc + nc - 1] as f64 > thr) as i32;
    }
    let m = dark.iter().copied().min().unwrap_or(0);
    m as f32 / (ncell - 2) as f32
}

/// Row-major bits, MSB = cell (0,0). Returns `(id, rot)` on success. For each id, for
/// each of 4 rotations: popcount(bits ^ code). Accept the best when Hamming <=
/// max_correction; reject a tie between two different ids.
pub fn dict_identify(dict: i32, bits: u64) -> Option<(i32, i32)> {
    let d = dict_info(dict).ok()?;
    let (mut best_id, mut best_rot, mut best_dist) = (-1i32, -1i32, 65i32);
    let mut tie = false;
    for i in 0..d.n {
        for j in 0..4 {
            let dist = (bits ^ d.codes[(i * 4 + j) as usize]).count_ones() as i32;
            if dist < best_dist {
                best_dist = dist;
                best_id = i;
                best_rot = j;
                tie = false;
            } else if dist == best_dist && i != best_id {
                tie = true;
            }
        }
    }
    if best_id < 0 || best_dist > d.max_correction || tie {
        return None;
    }
    Some((best_id, best_rot))
}

/// Cold decode configuration (ID unknown). The decoder is the unchanged legacy one (Otsu,
/// all-black border, `dict_identify` with max_correction and the legacy tie rule over the
/// WHOLE dictionary); `in_use` only drops results whose id is outside the set.
/// `None`: every id (legacy); `Some(&[])`: no id at all.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeConfig<'a> {
    pub in_use: Option<&'a [i32]>,
}

pub fn in_use_contains(in_use: &[i32], id: i32) -> bool {
    in_use.contains(&id)
}

/// Cold identification: `dict_identify` over the whole dictionary (unchanged legacy rule),
/// then reject an id outside the in-use set. `cfg` `None` or `cfg.in_use` `None`: exactly
/// `dict_identify`.
pub fn dict_identify_in_use(
    dict: i32,
    cfg: Option<&DecodeConfig>,
    bits: u64,
) -> Option<(i32, i32)> {
    let (i, r) = dict_identify(dict, bits)?;
    if let Some(set) = cfg.and_then(|c| c.in_use) {
        if !in_use_contains(set, i) {
            return None;
        }
    }
    Some((i, r))
}

/// Verify mode: the marker's ID is ALREADY KNOWN (tracked, or from a known map of marker
/// poses). Partial-occlusion completion is allowed only here; cold decoding stays strict.
/// Guarantee (exhaustive test `verify_cells_exhaustive_safety`, 4X4_50, in-use ids 0..9):
/// with `other_margin >= 3` and `max_mismatch` 1, verify never accepts a different in-use
/// id (nor the right id with a wrong rotation) when the true marker's known cells hold up
/// to 3 corrupted bits (any erasure set of up to 8 of the 16 interior cells). It is NOT
/// safe beyond that. `other_margin` 2 is only safe up to 2 corrupted known cells.
#[derive(Debug, Clone, Copy)]
pub struct VerifyConfig<'a> {
    /// Ids the "other marker" check runs against; `None` = whole dictionary.
    pub in_use: Option<&'a [i32]>,
    /// Tolerated erased/failing border cells.
    pub border_fail_max: i32,
    pub erasure_band: f32,
    /// Threshold at the midpoint of the black and white levels instead of Otsu.
    pub local_threshold: bool,
    /// Known (non-erased) interior cells / all interior cells.
    pub min_known_frac: f32,
    /// Expected codeword's mismatches on the known cells.
    pub max_mismatch: i32,
    /// Every other (id, rot) needs >= `max_mismatch + other_margin` mismatches there.
    pub other_margin: i32,
}

impl Default for VerifyConfig<'_> {
    fn default() -> Self {
        VerifyConfig {
            in_use: None,
            border_fail_max: 4,
            erasure_band: 0.15,
            local_threshold: true,
            min_known_frac: 0.5,
            max_mismatch: 1,
            other_margin: 3,
        }
    }
}

/// Accepted verify: rotation, mismatches of the expected codeword on the known cells, and
/// the number of erased interior cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyOk {
    pub rot: i32,
    pub n_mismatch: i32,
    pub n_erased: i32,
}

/// Bit-level verify core. `bits` are the interior bits, `erased` marks unknown interior
/// bits. `expected_rot` -1 = any (the rotation with the fewest mismatches). Rejects when
/// `expected_id` is not in the in-use set.
pub fn verify_cells(
    dict: i32,
    cfg: &VerifyConfig,
    bits: u64,
    erased: u64,
    expected_id: i32,
    expected_rot: i32,
) -> Option<VerifyOk> {
    let d = dict_info(dict).ok()?;
    let n_int = d.marker_size * d.marker_size;
    if expected_id < 0 || expected_id >= d.n || expected_rot >= 4 {
        return None;
    }
    if let Some(set) = cfg.in_use {
        if !in_use_contains(set, expected_id) {
            return None;
        }
    }
    let mask: u64 = if n_int >= 64 { !0 } else { (1u64 << n_int) - 1 };
    let known = !erased & mask;
    let e = (erased & mask).count_ones() as i32;
    if ((n_int - e) as f64) < cfg.min_known_frac as f64 * n_int as f64 {
        return None;
    }
    let (mut mr, mut mm) = (expected_rot, 65i32);
    for j in 0..4i32 {
        if expected_rot >= 0 && j != expected_rot {
            continue;
        }
        let dist = ((bits ^ d.codes[(expected_id * 4 + j) as usize]) & known).count_ones() as i32;
        if dist < mm {
            mm = dist;
            mr = j;
        }
    }
    if mm > cfg.max_mismatch {
        return None;
    }
    let need = cfg.max_mismatch + cfg.other_margin;
    let n = cfg.in_use.map_or(d.n as usize, |s| s.len());
    for a in 0..n {
        let ia = cfg.in_use.map_or(a as i32, |s| s[a]);
        // ids outside the dictionary cannot be compared; the C++ would read past the table
        if ia < 0 || ia >= d.n {
            continue;
        }
        for j in 0..4i32 {
            if ia == expected_id && j == mr {
                continue;
            }
            if (((bits ^ d.codes[(ia * 4 + j) as usize]) & known).count_ones() as i32) < need {
                return None;
            }
        }
    }
    Some(VerifyOk {
        rot: mr,
        n_mismatch: mm,
        n_erased: e,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeResult {
    /// -1 on failure.
    pub id: i32,
    /// CCW quarter-turns, from `dict_identify`.
    pub rot: i32,
    /// Hamming distance of the accepted code over the known bits.
    pub n_bit_errors: i32,
    /// Erased interior cells.
    pub n_erased: i32,
    /// Erased or failing (not black) border cells.
    pub border_fail: i32,
    pub border_ok: bool,
    /// Max cell mean - min cell mean, DN.
    pub contrast: f32,
    pub bits: u64,
    /// Input rays, re-ordered TL,TR,BR,BL on success.
    pub corners: [Vec3; 4],
    /// `codes::ACCEPT` or `codes::R_*`.
    pub reject_code: i32,
}

impl DecodeResult {
    /// The failed-result starting point of both decoders: corners = the input rays.
    fn failed(rays: &[Vec3; 4]) -> Self {
        DecodeResult {
            id: -1,
            rot: -1,
            n_bit_errors: 0,
            n_erased: 0,
            border_fail: 0,
            border_ok: false,
            contrast: 0.0,
            bits: 0,
            corners: *rays,
            reject_code: codes::R_IDENTIFY_FAILED,
        }
    }
}

/// Scalar cell statistics (mean and standard deviation of the `n_sub`^2 sub-samples). Only
/// the erasure-aware path needs the spread, and it is the rare path, so no NEON twin.
fn cell_stats(v: &View, h: &[f64; 9], ncell: i32, n_sub: i32, mean: &mut [f32], sd: &mut [f32]) {
    let half_step = K_CELL_FRAC as f64 / ncell as f64;
    for row in 0..ncell {
        let cy = -(((row as f64 + 0.5) / ncell as f64) * 2.0 - 1.0);
        for col in 0..ncell {
            let cx = ((col as f64 + 0.5) / ncell as f64) * 2.0 - 1.0;
            let (mut acc, mut acc2) = (0.0f64, 0.0f64);
            for sy in 0..n_sub {
                let gy = (((sy as f64 + 0.5) / n_sub as f64) * 2.0 - 1.0) * half_step;
                for sx in 0..n_sub {
                    let gx = (((sx as f64 + 0.5) / n_sub as f64) * 2.0 - 1.0) * half_step;
                    let ray = plane_to_ray(h, cx + gx, cy + gy);
                    let (px, py) = ray_to_px(&ray, v.w, v.h);
                    let s = sample(v, px as f32, py as f32);
                    acc += s as f64;
                    acc2 += s as f64 * s as f64;
                }
            }
            let k = (row * ncell + col) as usize;
            let nn = (n_sub * n_sub) as f64;
            mean[k] = (acc / nn) as f32;
            let m = acc / nn;
            sd[k] = (0.0f64).max(acc2 / nn - m * m).sqrt() as f32;
        }
    }
}

/// Sorts `a` in place and returns the median (mean of the two middle values on an even count).
fn median_of(a: &mut [f32]) -> f32 {
    a.sort_unstable_by(f32::total_cmp);
    let n = a.len();
    if n & 1 == 1 {
        a[n / 2]
    } else {
        0.5 * (a[n / 2 - 1] + a[n / 2])
    }
}

/// The verify decode body: erasure-aware cell reading (local threshold, erasure band,
/// border tolerance from `cfg`) followed by `verify_cells`. Same cell grid and bit packing
/// as the legacy path.
fn decode_verify_body(
    v: &View,
    dict: i32,
    h: &[f64; 9],
    n_sub: i32,
    cfg: &VerifyConfig,
    expected_id: i32,
    expected_rot: i32,
    r: &[Vec3; 4],
    out: &mut DecodeResult,
) {
    let Ok(d) = dict_info(dict) else { return };
    let msize = d.marker_size;
    let ncell = msize + 2;
    let ncell2 = (ncell * ncell) as usize;
    let mut mean = [0f32; 81];
    let mut sd = [0f32; 81];
    let mut scratch = [0f32; 81];
    let mut is_border = [false; 81];
    let (mut nb, mut ni) = (0usize, 0usize);
    let mut bvals = [0f32; 36];
    let mut ivals = [0f32; 49];
    if cfg.erasure_band > 0.0 {
        cell_stats(v, h, ncell, n_sub, &mut mean, &mut sd);
    } else {
        cell_means(v, h, ncell, n_sub, &mut mean);
    }
    let (mut cmin, mut cmax) = (mean[0], mean[0]);
    for row in 0..ncell {
        for col in 0..ncell {
            let k = (row * ncell + col) as usize;
            is_border[k] = row == 0 || row == ncell - 1 || col == 0 || col == ncell - 1;
            cmin = cmin.min(mean[k]);
            cmax = cmax.max(mean[k]);
            if is_border[k] {
                bvals[nb] = mean[k];
                nb += 1;
            } else {
                ivals[ni] = mean[k];
                ni += 1;
            }
        }
    }
    out.contrast = cmax - cmin;

    let mut gap = 0.0f64;
    let thr: f64 = if cfg.erasure_band > 0.0 || cfg.local_threshold {
        // black/white levels: robust border black (median) and the 80th percentile of the
        // interior; with local_threshold the threshold is their midpoint, else Otsu
        // the C++ sizes this 4*9, which a 7x7 interior (49 cells) overflows
        let mut tmp = [0f32; 49];
        tmp[..nb].copy_from_slice(&bvals[..nb]);
        let black0 = median_of(&mut tmp[..nb]);
        tmp[..ni].copy_from_slice(&ivals[..ni]);
        tmp[..ni].sort_unstable_by(f32::total_cmp);
        let white = tmp[(0.8 * (ni as f64 - 1.0)).ceil() as usize];
        let thr0 = 0.5 * (black0 as f64 + white as f64);
        let gap0 = white as f64 - black0 as f64;
        // refine black from the non-erased border cells that read dark
        let mut nk = 0usize;
        for i in 0..nb {
            if (bvals[i] as f64) < thr0
                && (bvals[i] as f64 - thr0).abs() > cfg.erasure_band as f64 * gap0
            {
                tmp[nk] = bvals[i];
                nk += 1;
            }
        }
        let black = if nk > 0 {
            median_of(&mut tmp[..nk])
        } else {
            black0
        };
        gap = white as f64 - black as f64;
        if !(gap > 1e-6) {
            out.reject_code = codes::R_IDENTIFY_FAILED;
            return;
        }
        if cfg.local_threshold {
            0.5 * (black as f64 + white as f64)
        } else {
            crate::bridge::otsu_exact(&mean[..ncell2], &mut scratch[..ncell2])
        }
    } else {
        crate::bridge::otsu_exact(&mean[..ncell2], &mut scratch[..ncell2])
    };

    let (mut bits, mut erased) = (0u64, 0u64);
    let mut bfail = 0;
    for row in 0..ncell {
        for col in 0..ncell {
            let k = (row * ncell + col) as usize;
            let on = mean[k] as f64 > thr;
            let mut er = false;
            if cfg.erasure_band > 0.0 {
                er = (mean[k] as f64 - thr).abs() <= cfg.erasure_band as f64 * gap
                    || sd[k] as f64 > 0.5 * gap;
            }
            if is_border[k] {
                if on || er {
                    bfail += 1;
                }
            } else {
                let bit = 1u64 << (msize * msize - 1 - ((row - 1) * msize + (col - 1)));
                if er {
                    erased |= bit;
                } else if on {
                    bits |= bit;
                }
            }
        }
    }
    out.bits = bits;
    out.border_fail = bfail;
    out.n_erased = erased.count_ones() as i32;
    out.border_ok = bfail <= cfg.border_fail_max;
    if !out.border_ok {
        out.reject_code = codes::R_BORDER_NOT_BLACK;
        return;
    }

    let Some(ok) = verify_cells(dict, cfg, bits, erased, expected_id, expected_rot) else {
        out.reject_code = codes::R_IDENTIFY_FAILED;
        return;
    };
    out.id = expected_id;
    out.rot = ok.rot;
    out.n_bit_errors = ok.n_mismatch;
    out.n_erased = ok.n_erased;
    for i in 0..4 {
        out.corners[i] = r[(i + 4 - ok.rot as usize) % 4];
    }
    out.reject_code = codes::ACCEPT;
}

/// Cell reading and legacy decode. Sub-sampling is `n_sub` x `n_sub`, uniform in
/// MARKER-PLANE coordinates over the central `K_CELL_FRAC` of each cell -- deliberately
/// NOT area-weighted by the equirect Jacobian (the reference tested the weighted variant
/// and kept the unweighted one). Thresholding is `otsu_exact` over the ncell^2 cell means.
/// ncell tops out at 9 (marker_size 7), so everything fits in fixed-size stack arrays: no
/// allocation on this per-candidate hot path.
///
/// `cfg` `None` or `cfg.in_use` `None`: the legacy decoder.
pub fn decode_rays(
    v: &View,
    dict: i32,
    rays: &[Vec3; 4],
    n_sub: i32,
    cfg: Option<&DecodeConfig>,
) -> DecodeResult {
    let mut out = DecodeResult::failed(rays);
    let Ok(d) = dict_info(dict) else { return out };
    if !(1..=16).contains(&n_sub) {
        return out;
    }
    let msize = d.marker_size;
    let ncell = msize + 2;

    // Force det[c0,c1,c2] > 0 on the rays -- if negative, reverse the array.
    let r = canonical_winding(rays);
    let Some(h) = homography_closed(&r) else {
        out.reject_code = codes::R_DEGENERATE_CORNER;
        return out;
    };

    let mut cellmean = [0f32; 81];
    cell_means(v, &h, ncell, n_sub, &mut cellmean);

    let ncell2 = (ncell * ncell) as usize;
    let mut scratch = [0f32; 81];
    let thr = crate::bridge::otsu_exact(&cellmean[..ncell2], &mut scratch[..ncell2]);

    let (mut cmin, mut cmax) = (cellmean[0], cellmean[0]);
    for i in 1..ncell2 {
        if cellmean[i] < cmin {
            cmin = cellmean[i];
        }
        if cellmean[i] > cmax {
            cmax = cellmean[i];
        }
    }
    out.contrast = cmax - cmin;

    // Bits are mean > threshold. Border check: the 4*ncell-4 outer cells must all be
    // black (bit 0). Inner bits pack row-major, MSB = cell (0,0) of the payload.
    let mut border_ok = true;
    let mut bits = 0u64;
    let mut n_border_on = 0;
    for row in 0..ncell {
        let border_row = row == 0 || row == ncell - 1;
        for col in 0..ncell {
            let on = cellmean[(row * ncell + col) as usize] as f64 > thr;
            if border_row || col == 0 || col == ncell - 1 {
                if on {
                    border_ok = false;
                    n_border_on += 1;
                }
            } else if on {
                let (ir, ic) = (row - 1, col - 1);
                bits |= 1u64 << (msize * msize - 1 - (ir * msize + ic));
            }
        }
    }
    out.border_ok = border_ok;
    out.border_fail = n_border_on;
    out.bits = bits;

    if !border_ok {
        out.reject_code = codes::R_BORDER_NOT_BLACK;
        return out;
    }

    // Cold decode: the unchanged legacy identification over the whole dictionary, then drop
    // ids outside the in-use set (a restricted-set search would loosen the correction bound
    // for small sets).
    let Some((id, rot)) = dict_identify_in_use(dict, cfg, bits) else {
        out.reject_code = codes::R_IDENTIFY_FAILED;
        return out;
    };

    out.id = id;
    out.rot = rot;
    out.n_bit_errors = (bits ^ d.codes[(id * 4 + rot) as usize]).count_ones() as i32;
    // corners = roll(rays, rot) -- cv2.aruco TL,TR,BR,BL (numpy.roll: result[i] = a[(i - shift) mod n]).
    for i in 0..4 {
        out.corners[i] = r[(i + 4 - rot as usize) % 4];
    }
    out.reject_code = codes::ACCEPT;
    out
}

/// Verify decode: same cell reading as the erasure-aware path (local threshold, erasures,
/// border tolerance from `vc`), then `verify_cells`. On success `id == expected_id`.
pub fn decode_verify(
    v: &View,
    dict: i32,
    rays: &[Vec3; 4],
    n_sub: i32,
    expected_id: i32,
    expected_rot: i32,
    vc: &VerifyConfig,
) -> DecodeResult {
    let mut out = DecodeResult::failed(rays);
    if dict_info(dict).is_err() || !(1..=16).contains(&n_sub) {
        return out;
    }
    let r = canonical_winding(rays);
    let Some(h) = homography_closed(&r) else {
        out.reject_code = codes::R_DEGENERATE_CORNER;
        return out;
    };
    decode_verify_body(
        v,
        dict,
        &h,
        n_sub,
        vc,
        expected_id,
        expected_rot,
        &r,
        &mut out,
    );
    out
}
