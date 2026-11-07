//! The diagnostic gates, per-corner validity and sigma scale, post-refine side
//! consistency, and the ring-width check.
//!
//! Five gates, applied in order, each returning the first `codes::R_*` failure code it finds
//! (or `codes::ACCEPT` if all pass). Gate 1 must run on the INIT/bridge quad, never the
//! post-fit normals.

use crate::codes;
use crate::decode::{dict_info, homography_closed, plane_to_ray, DecodeResult};
use crate::geometry::{cross, dot, norm, normalize, ray_to_px, Vec3};
use crate::image::{fmadd, sample, View};
use crate::refine::{
    RefineInfo, K_END_SUPPORT_COLD, K_MIN_CONTRAST_DN, K_MIN_TIP_PROFILES, K_TIP_FEW_SCALE,
};

/// On the INIT geometry.
pub const K_INIT_MIN_CROSS: f64 = 0.30;
/// edge_resid_px / side_px
pub const K_RESID_RATIO: f64 = 0.0026;

#[derive(Debug, Clone, Copy)]
pub struct GateInput<'a> {
    /// The BRIDGE quad, before refinement.
    pub init_rays: &'a [Vec3; 4],
    pub final_rays: &'a [Vec3; 4],
    pub info: &'a RefineInfo,
    pub decode: &'a DecodeResult,
    /// `reject_high_resid` of the parameters.
    pub reject_high_resid: bool,
}

/// min_i |n_{i-1} x n_i| for the (unfitted) quad `rays`, the SAME quantity
/// `corners_from_normals` computes post-fit -- but evaluated here on the raw input
/// geometry, before any IRLS has had a chance to blow up and look deceptively confident.
pub fn init_min_cross(rays: &[Vec3; 4]) -> f64 {
    // Edge-normal magnitude is sign-invariant (|a x b| == |(-a) x b|), so no outward
    // orientation is needed here: this gate only needs the UNSIGNED angle between adjacent
    // edges.
    let mut n = [Vec3::default(); 4];
    for i in 0..4 {
        let c = cross(&rays[i], &rays[(i + 1) & 3]);
        let nn = norm(&c);
        n[i] = if nn > 1e-300 {
            Vec3::new(c.x / nn, c.y / nn, c.z / nn)
        } else {
            Vec3::new(0.0, 0.0, 0.0)
        };
    }
    let mut mn = 2.0; // any real |cross| of two unit vectors is <= 1
    for i in 0..4 {
        let im1 = (i + 3) & 3;
        let cn = norm(&cross(&n[im1], &n[i]));
        if cn < mn {
            mn = cn;
        }
    }
    mn
}

fn compute_quality(info: &RefineInfo) -> f32 {
    let resid_ratio = if info.side_px > 1e-12 {
        info.resid_px / info.side_px
    } else {
        1e9
    };
    let f1 = (info.contrast_dn / 40.0).min(1.0);
    let f2 = (K_RESID_RATIO / resid_ratio.max(1e-6)).min(1.0);
    let f3 = (info.min_cross / 0.5).min(1.0);
    let q = (f1 * f2 * f3).clamp(0.0, 1.0);
    q as f32
}

/// Outcome of `apply_gates`: `code` is `ACCEPT` or the first failing `R_*`; `flags` holds
/// `FLAG_HIGH_RESID` when the residual ratio exceeds `K_RESID_RATIO` (regardless of whether
/// that also causes a rejection; none of the gates before gate 5 set a flag), `quality` is
/// computed whatever the outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateOutcome {
    pub code: i32,
    pub flags: u32,
    pub quality: f32,
}

#[inline]
fn clamp_unit(d: f64) -> f64 {
    d.clamp(-1.0, 1.0)
}

pub fn apply_gates(inp: &GateInput) -> GateOutcome {
    let info = inp.info;
    let mut f = 0u32;
    let quality = compute_quality(info);
    let ret = |code: i32, flags: u32| GateOutcome {
        code,
        flags,
        quality,
    };

    // Gate 1 -- initial geometry: the post-fit normals look perfect EXACTLY when the fit
    // has blown up (a collapsed seed still "converges" onto some plane), so this must run
    // on the BRIDGE quad (init_rays), never on info.min_cross. Gating on post-fit
    // min_cross alone let 36/36 garbage results through; gating here cuts that to 20/36,
    // and combined with gate 4 to 10/36 with 0/50 good results lost.
    if init_min_cross(inp.init_rays) <= K_INIT_MIN_CROSS {
        return ret(codes::R_INIT_GEOMETRY, f);
    }

    // Gate 2 -- displacement: a corner that moved more than info.hw0 from its initial
    // (bridge) position. hw0 and side_px are both in px_eq, a unit this function has no
    // image height to convert to radians with, so the comparison is made as a FRACTION of
    // the seed's own mean edge arc (hw0 / side_px), computed entirely from geometry already
    // on hand. Displacement alone let 29/36 garbage through (0/50 good lost).
    {
        let mut arc_sum = 0.0f64;
        for i in 0..4 {
            let d = clamp_unit(dot(&inp.init_rays[i], &inp.init_rays[(i + 1) & 3]));
            arc_sum += d.acos();
        }
        let init_arc = arc_sum / 4.0; // radians
        let frac = if info.side_px > 1e-12 {
            info.hw0 / info.side_px
        } else {
            0.0
        };
        let disp_limit = frac * init_arc;
        for i in 0..4 {
            let d = clamp_unit(dot(&inp.init_rays[i], &inp.final_rays[i]));
            if d.acos() > disp_limit {
                return ret(codes::R_DISPLACEMENT, f);
            }
        }
    }

    // Gate 3 -- contrast.
    if info.contrast_dn < K_MIN_CONTRAST_DN as f64 {
        return ret(codes::R_LOW_CONTRAST, f);
    }

    // Gate 4 -- border black, already decided by decode_rays; propagate.
    if !inp.decode.border_ok {
        return ret(codes::R_BORDER_NOT_BLACK, f);
    }

    // Gate 5 -- normalised edge residual: flag by default (a hard reject here costs 4% of
    // good results, incompatible with the mandatory 100% detection gates), hard-reject
    // only when the caller opts in.
    let resid_ratio = if info.side_px > 1e-12 {
        info.resid_px / info.side_px
    } else {
        0.0
    };
    if resid_ratio > K_RESID_RATIO {
        f |= codes::FLAG_HIGH_RESID;
        if inp.reject_high_resid {
            return ret(codes::R_HIGH_RESID, f);
        }
    }

    ret(codes::ACCEPT, f)
}

// ---- per-corner validity and sigma scale ----------------------------------------------
//
// A corner is a MEASUREMENT for the estimator only when its tip was observed: both adjacent
// edges have end_support >= K_END_SUPPORT_COLD (0.75) at that corner (the inliers of the
// refine profiles within the last K_END_WINDOW of the edge).
//   * cold mode (ID unknown): any other corner is invalid;
//   * known-ID mode: it stays valid, as the intersection of two MEASURED edges, with
//     sigma_scale = corner_sigma_scale(end_support) > 1;
//   * a corner of a weak (3-edge mode) edge is COMPLETED: invalid in both modes (it comes
//     from the prior + geometry, so feeding it back would be circular);
//   * gate 5, per corner: an adjacent edge whose inlier residual exceeds
//     K_RESID_RATIO * side marks the corner invalid instead of rejecting the marker.

/// Corner of a completed (weak) edge.
pub const K_CORNER_COMPLETED: u8 = 1;
/// Gate 5: adjacent edge residual above the bound.
pub const K_CORNER_RESID: u8 = 2;
/// Post-refine consistency failed on an adjacent side.
pub const K_CORNER_SIDE: u8 = 4;
/// A known-ID corner farther than `K_PRIOR_K` * sigma from its prior ray.
pub const K_CORNER_PRIOR: u8 = 8;
/// The ring-width check failed on an adjacent side.
pub const K_CORNER_RING: u8 = 16;

/// Prior consistency (known ID only): a valid, non-COMPLETED corner must lie within
/// `K_PRIOR_K` * sigma_deg of its predicted ray, else it goes invalid with `K_CORNER_PRIOR`;
/// fewer than `K_PRIOR_MIN_CONSISTENT` consistent measured corners reject the marker (a
/// look-alike at the wrong place).
pub const K_PRIOR_K: f64 = 3.0;
pub const K_PRIOR_MIN_CONSISTENT: i32 = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerState {
    pub valid: bool,
    pub flags: u8,
    /// >= 1; 1 = a clear corner.
    pub sigma_scale: f32,
    /// Min over the two adjacent edges' tip windows.
    pub end_support: f32,
    /// Max inlier residual of the two adjacent edges, px_eq.
    pub resid_px: f32,
    /// Min over the two adjacent edges' whole-edge inlier fraction.
    pub edge_support: f32,
}

impl Default for CornerState {
    fn default() -> Self {
        CornerState {
            valid: false,
            flags: 0,
            sigma_scale: 1.0,
            end_support: 0.0,
            resid_px: 0.0,
            edge_support: 0.0,
        }
    }
}

/// Gate 5's bound is `K_RESID_RATIO` * side, floored at `K_CORNER_RESID_FLOOR_PX` (px_eq):
/// the bare ratio is 0.036 px at a 14 px marker and invalidated 1140 of 3520 clear corners
/// of the calibration corpus. Full-grid frac = 0 population (3520 corners of detected unoccluded
/// markers, cold): inlier-residual p50 0.051 px, p99 = p99.9 = max = 0.196 px, the worst
/// being the px 10 / 14, psi 65 markers. The floor is 0.15, NOT 0.20, on purpose: at 0.20 no
/// frac = 0 corner is invalid, but 8 cold-valid corners above 3 px come back, so the cost of
/// 0.15 is 160 of 3520 frac = 0 corners flagged invalid.
pub const K_CORNER_RESID_FLOOR_PX: f64 = 0.15;

// Piecewise-linear s(end_support), fitted on the calibration corpus in known-ID (verify) mode: s
// multiplies the BASE corner sigma of a clear corner (end_support >= 0.75). It is the ratio
// of the median corner error in each end-support bin (bins merged to >= 30 corners) to the
// median error of the clear corners (sigma_0 = 0.0875 px over 8589 corners), made
// non-increasing: a ratio of medians, not a Gaussian sigma. Knots are bin centres.
const K_SCALE_E: [f32; 7] = [0.0, 0.05, 0.2, 0.4, 0.55, 0.675, 0.75];
const K_SCALE_S: [f32; 7] = [17.20, 17.20, 11.36, 6.67, 4.47, 4.47, 1.00];
const K_SCALE_N: usize = 7;

/// 1 at `end_support >= K_END_SUPPORT_COLD`, rising towards 0.
pub fn corner_sigma_scale(e: f32) -> f32 {
    if !(e < K_SCALE_E[K_SCALE_N - 1]) {
        return 1.0;
    }
    if !(e > K_SCALE_E[0]) {
        return K_SCALE_S[0];
    }
    for i in 1..K_SCALE_N {
        if e <= K_SCALE_E[i] {
            let t = (e - K_SCALE_E[i - 1]) / (K_SCALE_E[i] - K_SCALE_E[i - 1]);
            // fused like the C++ build's -ffp-contract=fast
            return fmadd(t, K_SCALE_S[i] - K_SCALE_S[i - 1], K_SCALE_S[i - 1]);
        }
    }
    1.0
}

/// Post-refine consistency (LINEFIT / RELAXED quads), per side with support >=
/// kEdgeMinSupport: the refined great circle must agree with the bridge's line within this
/// angle and, at the bridge side's midpoint, this offset (in cells; a cell = module_px).
pub const K_CONSIST_ANGLE_RAD: f64 = 2.0 * std::f64::consts::PI / 180.0;
pub const K_CONSIST_OFFSET_CELL: f64 = 0.15;

/// Does side `e` of `refined` agree with the bridge's side `e` of `init` (same corner
/// indexing)? Angle between the two great circles < `max_angle_rad` and, at the bridge
/// side's midpoint, offset < `max_offset_cell` * `module_px` (px_eq; `pix_rad` = pi / H
/// converts radians). The reference defaults are `K_CONSIST_ANGLE_RAD` and
/// `K_CONSIST_OFFSET_CELL`.
pub fn side_agrees(
    init: &[Vec3; 4],
    refined: &[Vec3; 4],
    e: usize,
    module_px: f64,
    pix_rad: f64,
    max_angle_rad: f64,
    max_offset_cell: f64,
) -> bool {
    let f = (e + 1) & 3;
    let mut nb = cross(&init[e], &init[f]);
    let mut nr = cross(&refined[e], &refined[f]);
    let (lb, lr) = (norm(&nb), norm(&nr));
    if lb < 1e-12 || lr < 1e-12 {
        return true; // a degenerate side has no line to compare
    }
    nb = Vec3::new(nb.x / lb, nb.y / lb, nb.z / lb);
    nr = Vec3::new(nr.x / lr, nr.y / lr, nr.z / lr);
    let ang = dot(&nb, &nr).abs().min(1.0).acos();
    let mb = normalize(&Vec3::new(
        init[e].x + init[f].x,
        init[e].y + init[f].y,
        init[e].z + init[f].z,
    ));
    let off_px = dot(&mb, &nr).abs() / pix_rad;
    ang < max_angle_rad && off_px < max_offset_cell * module_px
}

// ---- Ring-width (dark-area) check -------------------------------------------------------
//
// A refiner that locks onto an occluder bar's inner boundary leaves only a part of the
// 1-module black border inward of the refined edge. For side `side` (corner side -> side+1,
// TL,TR,BR,BL order) of a marker of id `id`, every inner code cell the dictionary says is
// WHITE gives a profile along the plane's inward normal (module measured through the
// quad's homography); I_d = I(0.2 module), I_w = mean I(1.3..1.5 modules), darkness
// d(t) = clamp((I_w - I(t)) / (I_w - I_d), 0, 1), width = integral of d over [0, 1.5]
// modules. Genuine border ~1.0, a 0.6-module bar leaves ~0.4.

/// Perpendicular module below this (px_eq): the side cannot be judged.
pub const K_RING_MIN_MODULE_PX: f64 = 3.0;
/// A side fails below this many modules. Chosen from the calibration corpus (cold, frac 0 and 0.1):
/// genuine border widths are >= 0.93 over all 2700 frac=0 sides (psi 0..75; min 0.93 at
/// psi 75), 0.87 at the 0.1 percentile of the sides of accurate (< 1 px) occluded corners;
/// the 8 cold corners above 3 px sit on sides of width 0.44..0.50. 0.7 is in the gap.
pub const K_RING_MIN: f64 = 0.7;
/// Known ID only: a corner next to a side whose ring is unjudged (-1) gets sigma_scale >=
/// this.
pub const K_RING_UNJUDGED_SCALE: f32 = 2.0;
/// Known ID only: a side above this many modules is outside the true border (a dark
/// occluder merged into it). Genuine sides, frac=0 corpus (135 judged sides): max 1.080,
/// p99.9 1.080. p99.9 <= 1.1, so 1.2 stays.
pub const K_RING_MAX: f64 = 1.2;

/// Median ring width over the usable profiles of `side`, or -1 when the side cannot be
/// judged: no white adjacent cell, contrast < `K_MIN_CONTRAST_DN`, the perpendicular module
/// < `K_RING_MIN_MODULE_PX` px_eq, a degenerate quad, or an unknown dictionary / id.
pub fn ring_width(
    v: &View,
    dict: i32,
    id: i32,
    corners: &[Vec3; 4],
    side: usize,
    pix_rad: f64,
) -> f32 {
    let Some(h) = homography_closed(corners) else {
        return -1.0;
    };
    let Ok(di) = dict_info(dict) else {
        return -1.0;
    };
    if id < 0 || id >= di.n {
        return -1.0;
    }
    let ms = di.marker_size as usize;
    let nc = ms + 2;
    let code = di.codes[id as usize * 4]; // the accepted id in the final (rotation 0) orientation
    let w = 2.0 / nc as f64; // one module in plane units
    const K_TAPS: usize = 16; // t = 0, 0.1 .. 1.5 modules
    let mut widths = [0f32; 16];
    let mut nw = 0usize;
    for kk in 0..ms {
        if nw >= 16 {
            break;
        }
        // inner cell (ir, ic) adjacent to this side; side 0 top row, 1 right col, 2 bottom
        // row, 3 left col
        let ir = if side == 0 {
            0
        } else if side == 2 {
            ms - 1
        } else {
            kk
        };
        let ic = if side == 3 {
            0
        } else if side == 1 {
            ms - 1
        } else {
            kk
        };
        if (code >> (ms * ms - 1 - (ir * ms + ic))) & 1 == 0 {
            continue; // adjacent cell must be white
        }
        let cx = -1.0 + (1.0 + ic as f64 + 0.5) * w;
        let cy = 1.0 - (1.0 + ir as f64 + 0.5) * w;
        let mut ii = [0f64; K_TAPS];
        for i in 0..K_TAPS {
            let t = 0.1 * i as f64 * w;
            let x = if side == 3 {
                -1.0 + t
            } else if side == 1 {
                1.0 - t
            } else {
                cx
            };
            let y = if side == 0 {
                1.0 - t
            } else if side == 2 {
                -1.0 + t
            } else {
                cy
            };
            let (px, py) = ray_to_px(&plane_to_ray(&h, x, y), v.w, v.h);
            ii[i] = sample(v, px as f32, py as f32) as f64;
        }
        // the perpendicular module in px_eq: the angle between the t = 0 and t = 1 module points
        let xa = if side == 3 {
            -1.0
        } else if side == 1 {
            1.0
        } else {
            cx
        };
        let ya = if side == 0 {
            1.0
        } else if side == 2 {
            -1.0
        } else {
            cy
        };
        let xb = if side == 3 {
            -1.0 + w
        } else if side == 1 {
            1.0 - w
        } else {
            cx
        };
        let yb = if side == 0 {
            1.0 - w
        } else if side == 2 {
            -1.0 + w
        } else {
            cy
        };
        let d = dot(&plane_to_ray(&h, xa, ya), &plane_to_ray(&h, xb, yb));
        let mod_px = d.clamp(-1.0, 1.0).acos() / pix_rad;
        if mod_px < K_RING_MIN_MODULE_PX {
            continue;
        }
        let id_ = ii[2];
        let iw = (ii[13] + ii[14] + ii[15]) / 3.0;
        let con = iw - id_;
        if con < K_MIN_CONTRAST_DN as f64 {
            continue;
        }
        let mut sum = 0.0f64;
        for i in 0..K_TAPS {
            let mut dk = (iw - ii[i]) / con;
            dk = dk.clamp(0.0, 1.0);
            sum += dk * (if i == 0 || i == K_TAPS - 1 { 0.5 } else { 1.0 }) * 0.1;
        }
        widths[nw] = sum as f32;
        nw += 1;
    }
    if nw == 0 {
        return -1.0;
    }
    widths[..nw].sort_unstable_by(f32::total_cmp);
    if nw % 2 == 1 {
        widths[nw / 2]
    } else {
        0.5 * (widths[nw / 2 - 1] + widths[nw / 2])
    }
}

/// `out[i]` is the state of corner i of the REFINE seed order (edge e = corner e -> e+1).
pub fn corner_states(info: &RefineInfo, known_id: bool) -> [CornerState; 4] {
    let mut out = [CornerState::default(); 4];
    for i in 0..4 {
        let ea = &info.edge[(i + 3) & 3]; // arrives at corner i: its tip window is end 1
        let eb = &info.edge[i]; // leaves corner i: its tip window is end 0
        let c = &mut out[i];
        *c = CornerState::default();
        c.end_support = ea.end_support[1].min(eb.end_support[0]);
        c.edge_support = ea.support.min(eb.support);
        c.resid_px = ea.resid_px.max(eb.resid_px);
        c.sigma_scale = corner_sigma_scale(c.end_support);
        if known_id && ea.n_end.min(eb.n_end) < K_MIN_TIP_PROFILES {
            c.sigma_scale = c.sigma_scale.max(K_TIP_FEW_SCALE);
        }
        if ea.weak || eb.weak {
            c.flags |= K_CORNER_COMPLETED;
            c.valid = false;
            continue;
        }
        c.valid = known_id || (c.end_support >= K_END_SUPPORT_COLD as f32);
        let bound = (K_RESID_RATIO * info.side_px).max(K_CORNER_RESID_FLOOR_PX);
        if ea.resid_px as f64 > bound || eb.resid_px as f64 > bound {
            c.flags |= K_CORNER_RESID;
            c.valid = false;
        }
    }
    out
}
