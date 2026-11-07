//! Great-circle edge refinement of the bridge's corner rays.
//!
//! The bridge closes the gap from the proposal's coarse localisation to a few px_eq; the
//! great-circle refiner closes the rest, to a few hundredths of a px_eq. For each of the 4
//! marker edges it samples K short perpendicular intensity profiles straddling the
//! (approximate) edge great circle, finds each profile's sub-pixel edge location by a
//! rectified gradient centroid (immune to a first-moment-preserving symmetric blur),
//! displaces the sample points onto the refined edge, and fits a plane (a great-circle
//! normal) through them with an IRLS Tukey re-weighting so a partial occluder does not drag
//! the fit.

use crate::codes;
use crate::geometry::{
    accum_ppt, cross, dot, norm, normalize, ray_to_px, slerp, smallest_eigvec3, Vec3, K_PI,
};
use crate::image::{sample, View};

/// Skip 8 % at each end of an edge.
pub const K_INSET: f64 = 0.08;
/// px_eq per profile tap.
pub const K_STEP: f64 = 0.5;
pub const K_K_MIN: i32 = 12;
/// Cap on the number of points per edge profile.
pub const K_MAX: usize = 64;
pub const K_K_MAX: i32 = K_MAX as i32;

/// Minimum plateau-to-plateau contrast (DN) for an edge profile to be trusted.
pub const K_MIN_CONTRAST_DN: f32 = 8.0;

/// clip(round((1 - 2*kInset) * side_px), kKMin, kKMax) = clip(round(0.84*side_px), ...)
pub fn refine_k(side_px: f64) -> i32 {
    let k = ((1.0 - 2.0 * K_INSET) * side_px).round() as i32; // 0.84 * side_px
    k.clamp(K_K_MIN, K_K_MAX)
}

/// Per-edge cos/sin tables of the profile offsets (the M offsets are the same for every
/// one of the K profiles, so each pair is built once per edge).
#[derive(Debug, Default)]
pub struct ArcScratch {
    cs: Vec<f64>,
    sn: Vec<f64>,
}

/// Reusable buffers of `sample_edge` and `refine_rays`.
#[derive(Debug, Default)]
pub struct RefineScratch {
    pub arc: ArcScratch,
    prof: Vec<f32>,
}

/// Samples K profiles of M taps across one great-circle edge a->b. `i_out` must hold at
/// least K*M_max floats (caller sizes for the `hw_px` it passes), `pk` and `perp` at least K
/// entries; M is forced odd and returned. `pk[k]` / `perp[k]` are the profile's origin point
/// and outward unit tangent (perpendicular to the edge, in the edge's own great-circle
/// plane) -- both needed later to displace a sample point found by `op_centroid` back onto
/// the sphere. `None` when the edge is degenerate (a, b (near-)antipodal or coincident) or a
/// buffer is too small.
pub fn sample_edge(
    v: &View,
    a: &Vec3,
    b: &Vec3,
    centre: &Vec3,
    hw_px: f64,
    k_n: usize,
    i_out: &mut [f32],
    pk: &mut [Vec3],
    perp: &mut [Vec3],
    scratch: &mut ArcScratch,
) -> Option<usize> {
    let mut n = cross(a, b);
    let nn = norm(&n);
    if nn < 1e-12 {
        return None;
    }
    n = Vec3::new(n.x / nn, n.y / nn, n.z / nn);

    let hh = v.h;
    let mut m = (2.0 * hw_px / K_STEP).round() as i64 + 1;
    if m % 2 == 0 {
        m += 1;
    }
    let m = usize::try_from(m).ok()?;
    if i_out.len() < k_n * m || pk.len() < k_n || perp.len() < k_n {
        return None;
    }

    // flip perp outward once for the whole edge (n's sign already fixed by the a x b
    // winding, so a single dot-product test suffices per edge).
    let flip = dot(&n, centre) > 0.0;

    scratch.cs.resize(m, 0.0);
    scratch.sn.resize(m, 0.0);
    for j in 0..m {
        let t = (j as f64 - (m as f64 - 1.0) / 2.0) * K_STEP * (K_PI / hh as f64);
        scratch.cs[j] = t.cos();
        scratch.sn[j] = t.sin();
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    let fast = v.fmt == crate::image::Fmt::Gray8 && v.lut.is_none();

    for k in 0..k_n {
        let u = if k_n <= 1 {
            K_INSET
        } else {
            K_INSET + (1.0 - 2.0 * K_INSET) * (k as f64 / (k_n - 1) as f64)
        };
        let p = normalize(&slerp(a, b, u));
        let pdn = dot(&p, &n);
        let mut pp = Vec3::new(n.x - pdn * p.x, n.y - pdn * p.y, n.z - pdn * p.z);
        let ppl = norm(&pp);
        if ppl < 1e-9 {
            return None;
        }
        pp = Vec3::new(pp.x / ppl, pp.y / ppl, pp.z / ppl);
        if flip {
            pp = Vec3::new(-pp.x, -pp.y, -pp.z);
        }
        pk[k] = p;
        perp[k] = pp;
        let row = &mut i_out[k * m..(k + 1) * m];
        #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
        if fast {
            crate::simd::sample_arc_gray(
                v.data,
                v.w,
                v.h,
                v.stride,
                &p,
                &pp,
                &scratch.cs,
                &scratch.sn,
                row,
            );
            continue;
        }
        for j in 0..m {
            let (cs, sn) = (scratch.cs[j], scratch.sn[j]);
            let q = Vec3::new(
                p.x * cs + pp.x * sn,
                p.y * cs + pp.y * sn,
                p.z * cs + pp.z * sn,
            );
            let (px, py) = ray_to_px(&q, v.w, v.h);
            row[j] = sample(v, px as f32, py as f32);
        }
    }
    Some(m)
}

/// Rectified gradient centroid of each of the `k_n` profile rows (`m` taps each) -> sub-tap
/// offset in px_eq, written to `t_out[0..k_n]`.
pub fn op_centroid(i: &[f32], k_n: usize, m: usize, step: f64, t_out: &mut [f64]) {
    let mg = m - 2; // number of central-difference gradient samples
    for k in 0..k_n {
        let row = &i[k * m..(k + 1) * m];
        let (mut sum_g, mut sum_jg) = (0.0f64, 0.0f64);
        for j in 0..mg {
            let g = 0.5 * (row[j + 2] as f64 - row[j] as f64);
            let gp = if g > 0.0 { g } else { 0.0 };
            sum_g += gp;
            sum_jg += gp * j as f64;
        }
        let cen = if sum_g > 1e-9 {
            sum_jg / sum_g
        } else {
            (mg as f64 - 1.0) / 2.0
        };
        t_out[k] = ((cen + 1.0) - (m as f64 - 1.0) / 2.0) * step;
    }
}

/// 10-90 edge sigma and the per-edge search half-width.
pub const K_HW_FRAC: f64 = 0.10;
pub const K_HW_MIN: f64 = 1.5;
pub const K_SIG_K: f64 = 3.0;
pub const K_SIG_PAD: f64 = 0.8;

/// Sorts `v` in place and returns the median (average of the two middle values on an even
/// count, matching numpy.median).
fn median_inplace(v: &mut [f64]) -> f64 {
    // NaN never occurs here (finite profile data); an unordered pair counts as equal
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Effective edge sigma (px_eq) from the 10-90 rise width, NOT the gradient 2nd moment
/// (noise inflates the 2nd moment, widening the window, admitting more noise -- measured
/// ~25 % accuracy loss at noise sd 10). Per profile row: normalise by its own end-plateau
/// means (`npl` taps each end), count the taps whose normalised value lies in (0.1, 0.9),
/// multiply by step and divide by 2.563; returns the MEDIAN of the K per-row values.
/// `k_n` is at most `K_MAX`; the reference passes `npl` = 3.
pub fn edge_sigma(i: &[f32], k_n: usize, m: usize, step: f64, npl: usize) -> f64 {
    let mut sig = [0f64; K_MAX];
    for k in 0..k_n {
        let row = &i[k * m..(k + 1) * m];
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for j in 0..npl {
            lo += row[j] as f64;
        }
        for j in 0..npl {
            hi += row[m - 1 - j] as f64;
        }
        lo /= npl as f64;
        hi /= npl as f64;
        let amp = (hi - lo).max(1e-6);
        let mut n = 0;
        for j in 0..m {
            let f = (row[j] as f64 - lo) / amp;
            if f > 0.1 && f < 0.9 {
                n += 1;
            }
        }
        sig[k] = (n as f64 * step).max(step) / 2.563;
    }
    let s = &mut sig[..k_n];
    s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if k_n % 2 == 1 {
        s[k_n / 2]
    } else {
        0.5 * (s[k_n / 2 - 1] + s[k_n / 2])
    }
}

/// Pass 1 (basin) half-width: clamp(kHwFrac * side_px, kHwMin, hw_cap),
/// hw_cap = max(1.2, 0.90 * module_px), module_px = side_px / 6 (the 4x4 dictionary's
/// module count).
pub fn hw_initial(side_px: f64) -> f64 {
    let module_px = side_px / 6.0;
    let cap = (0.90 * module_px).max(1.2);
    let mut h = K_HW_FRAC * side_px;
    if h < K_HW_MIN {
        h = K_HW_MIN;
    }
    if h > cap {
        h = cap;
    }
    h
}

/// Pass 2+ half-width, PER EDGE: clamp(max(2.0, kSigK*sigma_e + kSigPad), 1.2,
/// min(prev_hw_for_that_edge, max(1.2, 0.90*module_px))).
///
/// A median-of-four rule (one window for all four edges) was measured against this per-edge
/// rule on the 20-pose accuracy grid and is worse on every metric, including the anisotropic
/// one it was introduced for; the per-edge rule is also the one that reproduces the Python
/// reference.
pub fn hw_next(sigma_e: f64, prev_hw: f64, module_px: f64) -> f64 {
    let mut h = K_SIG_K * sigma_e + K_SIG_PAD;
    if h < 2.0 {
        h = 2.0;
    }
    let cap = prev_hw.min((0.90 * module_px).max(1.2));
    if h < 1.2 {
        h = 1.2;
    }
    if h > cap {
        h = cap;
    }
    h
}

pub const K_TUKEY_C: f64 = 4.685;
pub const K_REWEIGHT: i32 = 4;
const K_IRLS_MAX_N: usize = 128; // callers pass at most K_MAX (64) points

/// Smallest-eigenvector great-circle-normal fit (`accum_ppt` + `smallest_eigvec3`) with
/// `K_REWEIGHT` Tukey IRLS reweightings, robust to a contiguous run of occluded/outlier
/// samples up to Tukey's ~50 % breakdown point (past ~55 % the fit latches onto the
/// occluder instead -- a documented limit). Residual r_i = P_i . n, centred on its median;
/// scale = max(kTukeyC * 1.4826 * MAD, 0.05 * pix_rad); w = (1-u^2)^2 for u = |r|/scale
/// < 1, else 0. `p` holds 1..=128 points (at most `K_MAX` per call in practice); outside
/// that range the zero vector is returned.
pub fn fit_normal_irls(p: &[Vec3], pix_rad: f64) -> Vec3 {
    let n = p.len();
    if n == 0 || n > K_IRLS_MAX_N {
        return Vec3::default();
    }
    let mut mm = accum_ppt(p, None);
    let mut nf = smallest_eigvec3(&mm);

    let mut r = [0f64; K_IRLS_MAX_N];
    let mut tmp = [0f64; K_IRLS_MAX_N];
    let mut w = [0f64; K_IRLS_MAX_N];
    for _ in 0..K_REWEIGHT {
        for i in 0..n {
            r[i] = dot(&p[i], &nf);
        }
        tmp[..n].copy_from_slice(&r[..n]);
        let med = median_inplace(&mut tmp[..n]);
        for i in 0..n {
            tmp[i] = (r[i] - med).abs();
        }
        let mad = median_inplace(&mut tmp[..n]);
        let mut scale = K_TUKEY_C * 1.4826 * mad;
        if scale < 0.05 * pix_rad {
            scale = 0.05 * pix_rad;
        }

        let mut wsum = 0.0f64;
        for i in 0..n {
            let u = (r[i] - med).abs() / scale;
            let wi = if u < 1.0 {
                (1.0 - u * u) * (1.0 - u * u)
            } else {
                0.0
            };
            w[i] = wi;
            wsum += wi;
        }
        if wsum < 1e-9 {
            break;
        }
        mm = accum_ppt(p, Some(&w[..n]));
        nf = smallest_eigvec3(&mm);
    }
    nf
}

/// The sign of `smallest_eigvec3` is arbitrary: flip it toward the marker-centre direction
/// `centre` BEFORE any cross product is taken with it -- omitting this costs 957 px of
/// corner error.
pub fn orient_outward(n: &Vec3, centre: &Vec3) -> Vec3 {
    if dot(n, centre) < 0.0 {
        return Vec3::new(-n.x, -n.y, -n.z);
    }
    *n
}

/// Summed sign vote: deciding each corner's sign independently fails in reachable poses.
/// V_i = normalize(n_{i-1} x n_i); the global sign sigma = sign((sum_i V_i) . centre)
/// (sigma = +1 on an exact zero dot); c_i = sigma * V_i for all four. Returns
/// `(false, min_cross)` (leaving `out` untouched) when min_i |n_{i-1} x n_i| <= `min_cross`;
/// the returned minimum is always set (diagnostics, even on failure).
pub fn corners_from_normals(
    n: &[Vec3; 4],
    centre: &Vec3,
    min_cross: f64,
    out: &mut [Vec3; 4],
) -> (bool, f64) {
    let mut raw = [Vec3::default(); 4];
    let mut cn = [0f64; 4];
    for i in 0..4 {
        let im1 = (i + 3) & 3;
        raw[i] = cross(&n[im1], &n[i]);
        cn[i] = norm(&raw[i]);
    }
    let mut mn = cn[0];
    for i in 1..4 {
        if cn[i] < mn {
            mn = cn[i];
        }
    }
    if mn <= min_cross {
        return (false, mn);
    }

    let mut vv = [Vec3::default(); 4];
    for i in 0..4 {
        let s = 1.0 / cn[i];
        vv[i] = Vec3::new(raw[i].x * s, raw[i].y * s, raw[i].z * s);
    }
    let sum_v = Vec3::new(
        vv[0].x + vv[1].x + vv[2].x + vv[3].x,
        vv[0].y + vv[1].y + vv[2].y + vv[3].y,
        vv[0].z + vv[1].z + vv[2].z + vv[3].z,
    );
    let sigma = if dot(&sum_v, centre) < 0.0 { -1.0 } else { 1.0 }; // +1 on an exact zero
    for i in 0..4 {
        out[i] = Vec3::new(sigma * vv[i].x, sigma * vv[i].y, sigma * vv[i].z);
    }
    (true, mn)
}

pub const K_REFINE_ITERS: i32 = 3;
pub const K_MIN_CROSS: f32 = 0.10;

/// The refine schedule, per path. The refiner is the whole of the tracking budget; its cost
/// is iters x 4 edges x K profiles x M taps, so the only levers are `iters` and `k_max`.
/// A SCAN seed comes off the bridge and has to converge from there; a TRACKED seed is already
/// sub-pixel, so the track path uses a shorter schedule.
///
/// `allow_weak` is the known-ID mode: it lets ONE edge whose inlier support is under
/// `K_EDGE_MIN_SUPPORT` be fitted from its (>= `K_MIN_INLIERS`) inlier profiles (3-edge
/// mode); without it any such edge fails the refine (`R_EDGE_FIT`). Two weak edges, or one
/// with < `K_MIN_INLIERS` inliers, fail in both modes: a whole edge is never extrapolated.
/// `k_max` below 1 behaves as 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefineCfg {
    pub iters: i32,
    pub k_max: i32,
    pub allow_weak: bool,
}

pub const K_SCAN_REFINE: RefineCfg = RefineCfg {
    iters: K_REFINE_ITERS,
    k_max: K_K_MAX,
    allow_weak: false,
};
pub const K_TRACK_REFINE: RefineCfg = RefineCfg {
    iters: 2,
    k_max: 32,
    allow_weak: false,
};

/// Below this inlier fraction the edge is weak.
pub const K_EDGE_MIN_SUPPORT: f64 = 0.3;
/// Tip window, fraction of the profiles.
pub const K_END_WINDOW: f64 = 0.20;
/// Cold mode (no ID check behind the quad) asks more than known-ID mode: on the calibration corpus
/// two cold-valid corners with 15 px error had end_support 0.62 / 0.69; at >= 0.75 none do.
///
/// Quantisation: the tip window holds nw = max(1, round(0.2 K)) profiles (K = 12..64, so
/// nw = 2..13) and end_support is a multiple of 1/nw. >= 0.75 therefore means ALL of the
/// window for nw <= 3 (the small markers: 2/3 = 0.67 fails), 3 of 4 for nw = 4, and so on;
/// the threshold is stricter for small markers.
pub const K_END_SUPPORT_COLD: f64 = 0.75;
/// Known ID only: a tip window of fewer than this many profiles (small markers) cannot
/// resolve "all of the window" from "most of it", so such a corner never gets sigma_scale 1
/// -- it is floored at `K_TIP_FEW_SCALE`.
pub const K_MIN_TIP_PROFILES: i32 = 4;
pub const K_TIP_FEW_SCALE: f32 = 2.0;
/// px_eq floor of the inlier tolerance.
pub const K_INLIER_MIN_PX: f64 = 0.4;
pub const K_MIN_INLIERS: i32 = 2;

/// Per-edge support (from the LAST pass). An INLIER profile is one that passed the
/// contrast/window filter AND lies within max(kInlierMinPx, Tukey scale) of the fitted great
/// circle. `support` is the inlier fraction of ALL K profiles; `end_support[0/1]` the inlier
/// fraction of the first / last `K_END_WINDOW` of the profiles, i.e. whether the edge was
/// MEASURED up to its tip at the corner e / e+1 (corner e is the intersection of edge e-1
/// and edge e). An edge without enough inliers is WEAK, never silently fitted through
/// occluder profiles.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EdgeSupport {
    pub n_prof: i32,
    pub n_inl: i32,
    pub support: f32,
    pub end_support: [f32; 2],
    /// Profiles in the tip window (nw); `end_support` is a multiple of 1/n_end.
    pub n_end: i32,
    /// rms residual of the inliers about the fit, px_eq.
    pub resid_px: f32,
    /// support < `K_EDGE_MIN_SUPPORT` (3-edge mode edge, known ID only).
    pub weak: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RefineInfo {
    pub ok: bool,
    pub reject_code: i32,
    pub side_px: f64,
    pub module_px: f64,
    pub hw0: f64,
    pub contrast_dn: f64,
    pub resid_px: f64,
    pub min_cross: f64,
    pub k: i32,
    pub n_used: i32,
    /// Edge e = corner e -> corner e+1 of the seed.
    pub edge: [EdgeSupport; 4],
}

/// Great-circle refinement of the 4 (approximate) corner rays `inp` -> `out`, running
/// `cfg.iters` passes with at most `cfg.k_max` profiles per edge -- `K_SCAN_REFINE` (the
/// reference's 3 / 64) for a scan. Only the detector may call it, and only behind the
/// guards. False (with `info.ok == false` and `info.reject_code` set) on a degenerate seed, a
/// featureless/low-contrast frame, or a degenerate (near-collinear) fitted corner; `out` is
/// left as the best available estimate (the seed rays re-centred, when the loop never gets
/// as far as a single successful edge fit).
pub fn refine_rays(
    v: &View,
    inp: &[Vec3; 4],
    out: &mut [Vec3; 4],
    info: &mut RefineInfo,
    cfg: &RefineCfg,
    scratch: &mut RefineScratch,
) -> bool {
    *info = RefineInfo::default();
    info.ok = true;
    info.reject_code = codes::ACCEPT;

    let pix = K_PI / v.h as f64;

    let mut c = [Vec3::default(); 4];
    for i in 0..4 {
        c[i] = normalize(&inp[i]);
        out[i] = c[i];
    }

    let mut side_sum = 0.0f64;
    for i in 0..4 {
        let mut d = dot(&c[i], &c[(i + 1) & 3]);
        d = d.clamp(-1.0, 1.0);
        side_sum += d.acos();
    }
    let side = (side_sum / 4.0) / pix;
    info.side_px = side;
    if !side.is_finite() || side < 4.0 {
        info.ok = false;
        info.reject_code = codes::R_DEGENERATE_INIT;
        return false;
    }

    let module_px = side / 6.0;
    info.module_px = module_px;
    // cfg.k_max caps the reference's own side-driven rule (never raises it): the scan path
    // passes K_K_MAX, so it is a no-op there, and the track path passes 32.
    let k_i = refine_k(side).min(cfg.k_max).max(1);
    info.k = k_i;
    let k_n = k_i as usize;

    // Per-edge search half-width: every edge starts at the same basin width and then
    // narrows on ITS OWN 10-90 rise, never on the other three's.
    let hw0 = hw_initial(side);
    let mut hw_e = [hw0; 4];
    info.hw0 = hw0;

    let mut ns = [Vec3::default(); 4];
    let mut sigma_pass = [0f64; 4];
    let mut con_pass = [0f64; 4];
    let mut res_pass = [0f64; 4];
    let mut nus_pass = [0i32; 4];

    for it in 0..cfg.iters {
        let s = normalize(&Vec3::new(
            c[0].x + c[1].x + c[2].x + c[3].x,
            c[0].y + c[1].y + c[2].y + c[3].y,
            c[0].z + c[1].z + c[2].z + c[3].z,
        ));
        let mut ok_edge = [false; 4];

        for e in 0..4 {
            let a = c[e];
            let b = c[(e + 1) & 3];
            let hw = hw_e[e];
            let mut m = (2.0 * hw / K_STEP).round() as usize + 1;
            if m % 2 == 0 {
                m += 1;
            }
            scratch.prof.resize(k_n * m, 0.0);
            let mut pk = [Vec3::default(); K_MAX];
            let mut perp = [Vec3::default(); K_MAX];
            let Some(mout) = sample_edge(
                v,
                &a,
                &b,
                &s,
                hw,
                k_n,
                &mut scratch.prof,
                &mut pk,
                &mut perp,
                &mut scratch.arc,
            ) else {
                continue;
            };
            let prof = &scratch.prof[..k_n * mout];

            sigma_pass[e] = edge_sigma(prof, k_n, mout, K_STEP, 3);
            // Narrow THIS edge's window for the next pass, straight away -- before the
            // good/ts gate below, so it happens even on an edge this pass ends up dropping.
            hw_e[e] = hw_next(sigma_pass[e], hw, module_px);

            let mut con = [0f64; K_MAX];
            let mut ts = [0f64; K_MAX];
            op_centroid(prof, k_n, mout, K_STEP, &mut ts);
            for k in 0..k_n {
                let row = &prof[k * mout..(k + 1) * mout];
                let hi = 0.5 * (row[mout - 1] as f64 + row[mout - 2] as f64);
                let lo = 0.5 * (row[0] as f64 + row[1] as f64);
                con[k] = hi - lo;
            }

            let mut good = [false; K_MAX];
            let mut ngood = 0;
            for k in 0..k_n {
                let g = (con[k] > K_MIN_CONTRAST_DN as f64)
                    && ts[k].is_finite()
                    && (ts[k].abs() < hw - 0.5 * K_STEP);
                good[k] = g;
                if g {
                    ngood += 1;
                }
            }
            // No "re-admit every finite profile" fallback. Profiles that failed the
            // contrast/window filter are occluder or background samples and must not steer
            // the fit. With fewer than K_MIN_INLIERS good profiles there is no edge: an
            // earlier pass holds the seed plane (so the corners stay consistent and the
            // basin is kept), the last pass leaves the edge unfitted and fails below.
            let last = it + 1 == cfg.iters;
            let es = &mut info.edge[e];
            *es = EdgeSupport::default();
            es.n_prof = k_i;
            if ngood < K_MIN_INLIERS {
                es.weak = true;
                if last {
                    continue;
                }
                let mut nh = normalize(&cross(&a, &b));
                if dot(&nh, &s) < 0.0 {
                    nh = Vec3::new(-nh.x, -nh.y, -nh.z);
                }
                ns[e] = nh;
                ok_edge[e] = true;
                hw_e[e] = hw; // keep the basin: this edge made no progress
                continue;
            }

            let mut pg = [Vec3::default(); K_MAX];
            let mut kidx = [0usize; K_MAX];
            let mut ng = 0usize;
            for k in 0..k_n {
                if !good[k] {
                    continue;
                }
                let tr = ts[k] * pix;
                let (cs, sn) = (tr.cos(), tr.sin());
                kidx[ng] = k;
                pg[ng] = normalize(&Vec3::new(
                    pk[k].x * cs + perp[k].x * sn,
                    pk[k].y * cs + perp[k].y * sn,
                    pk[k].z * cs + perp[k].z * sn,
                ));
                ng += 1;
            }

            let mut nf = fit_normal_irls(&pg[..ng], pix);

            // Inlier profiles: within max(kInlierMinPx, Tukey scale) of the fit.
            {
                let mut r = [0f64; K_MAX];
                let mut tmp = [0f64; K_MAX];
                for i in 0..ng {
                    r[i] = dot(&pg[i], &nf) / pix;
                    tmp[i] = r[i];
                }
                let med = median_inplace(&mut tmp[..ng]);
                for i in 0..ng {
                    tmp[i] = (r[i] - med).abs();
                }
                let mad = median_inplace(&mut tmp[..ng]);
                let tol = K_INLIER_MIN_PX.max(K_TUKEY_C * 1.4826 * mad);
                let mut inl = [false; K_MAX];
                let mut n_inl = 0;
                for i in 0..ng {
                    if (r[i] - med).abs() <= tol {
                        inl[kidx[i]] = true;
                        n_inl += 1;
                    }
                }
                es.n_inl = n_inl;
                es.support = n_inl as f32 / k_i as f32;
                let nw = ((K_END_WINDOW * k_i as f64).round() as i32).max(1);
                let (mut c0, mut c1) = (0, 0);
                for k in 0..nw as usize {
                    if inl[k] {
                        c0 += 1;
                    }
                    if inl[k_n - 1 - k] {
                        c1 += 1;
                    }
                }
                es.n_end = nw;
                es.end_support[0] = c0 as f32 / nw as f32;
                es.end_support[1] = c1 as f32 / nw as f32;
                es.weak = es.support < K_EDGE_MIN_SUPPORT as f32;
                // 3-edge mode (known ID): a weak edge is fitted from its inliers alone.
                if es.weak && last && cfg.allow_weak && n_inl >= K_MIN_INLIERS {
                    let mut pi = [Vec3::default(); K_MAX];
                    let mut ni = 0usize;
                    for i in 0..ng {
                        if inl[kidx[i]] {
                            pi[ni] = pg[i];
                            ni += 1;
                        }
                    }
                    nf = fit_normal_irls(&pi[..ni], pix);
                }
                let mut sq = 0.0f64;
                let mut nsum = 0;
                let mut mr = 0.0f64;
                for i in 0..ng {
                    if inl[kidx[i]] {
                        mr += dot(&pg[i], &nf);
                        nsum += 1;
                    }
                }
                if nsum > 0 {
                    mr /= nsum as f64;
                    for i in 0..ng {
                        if inl[kidx[i]] {
                            let d2 = dot(&pg[i], &nf) - mr;
                            sq += d2 * d2;
                        }
                    }
                    es.resid_px = ((sq / nsum as f64).sqrt() / pix) as f32;
                }
            }
            if dot(&nf, &s) < 0.0 {
                nf = Vec3::new(-nf.x, -nf.y, -nf.z);
            }
            ns[e] = nf;
            ok_edge[e] = true;

            let mut con_sorted = [0f64; K_MAX];
            con_sorted[..k_n].copy_from_slice(&con[..k_n]);
            con_pass[e] = median_inplace(&mut con_sorted[..k_n]);

            let mut mean_r = 0.0f64;
            for k in 0..ng {
                mean_r += dot(&pg[k], &nf);
            }
            mean_r /= ng as f64;
            let mut var_r = 0.0f64;
            for k in 0..ng {
                let d2 = dot(&pg[k], &nf) - mean_r;
                var_r += d2 * d2;
            }
            var_r /= ng as f64;
            res_pass[e] = var_r.sqrt() / pix;
            nus_pass[e] = ng as i32;
        }

        if it + 1 == cfg.iters {
            // Weak edges (support < K_EDGE_MIN_SUPPORT): known ID may complete exactly one
            // of them from >= K_MIN_INLIERS inliers; otherwise the refine fails.
            let (mut nweak, mut wk) = (0, 0usize);
            for e in 0..4 {
                if info.edge[e].weak {
                    nweak += 1;
                    wk = e;
                }
            }
            if nweak > 0 && (!cfg.allow_weak || nweak > 1 || info.edge[wk].n_inl < K_MIN_INLIERS) {
                info.ok = false;
                info.reject_code = codes::R_EDGE_FIT;
                return false;
            }
        }

        if !(ok_edge[0] && ok_edge[1] && ok_edge[2] && ok_edge[3]) {
            info.ok = false;
            info.reject_code = codes::R_EDGE_FIT;
            return false;
        }

        let mut vraw = [Vec3::default(); 4];
        let mut cn = [0f64; 4];
        for i in 0..4 {
            let im1 = (i + 3) & 3;
            vraw[i] = cross(&ns[im1], &ns[i]);
            cn[i] = norm(&vraw[i]);
        }
        let mut mn = cn[0];
        for i in 1..4 {
            if cn[i] < mn {
                mn = cn[i];
            }
        }
        info.min_cross = mn;
        if mn <= K_MIN_CROSS as f64 {
            info.ok = false;
            info.reject_code = codes::R_DEGENERATE_CORNER;
            return false;
        }

        let mut vv = [Vec3::default(); 4];
        for i in 0..4 {
            let sc = 1.0 / cn[i];
            vv[i] = Vec3::new(vraw[i].x * sc, vraw[i].y * sc, vraw[i].z * sc);
        }
        let sum_v = Vec3::new(
            vv[0].x + vv[1].x + vv[2].x + vv[3].x,
            vv[0].y + vv[1].y + vv[2].y + vv[3].y,
            vv[0].z + vv[1].z + vv[2].z + vv[3].z,
        );
        let sigma = if dot(&sum_v, &s) < 0.0 { -1.0 } else { 1.0 }; // +1 on an exact zero
        for i in 0..4 {
            c[i] = Vec3::new(sigma * vv[i].x, sigma * vv[i].y, sigma * vv[i].z);
        }
        out[..4].copy_from_slice(&c);
    }

    let (mut con_sum, mut res_sum, mut nus_sum) = (0.0f64, 0.0f64, 0);
    for e in 0..4 {
        con_sum += con_pass[e];
        res_sum += res_pass[e];
        nus_sum += nus_pass[e];
    }
    info.contrast_dn = con_sum / 4.0;
    info.resid_px = res_sum / 4.0;
    info.n_used = nus_sum;
    true
}
