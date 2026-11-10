//! The decode-first pipeline, budget and tracking.
//!
//! Six stages, in this order: proposal -> bridge -> decode-first -> refine -> gates ->
//! track. `scan` drives stages 1-5 per candidate (via `finish_candidate`: decode BEFORE
//! refine, so a code-less blob dies cheaply instead of paying for a full refine first);
//! `track` drives stage 6, always re-seeding through the SAME bridge (a raw stale prior is
//! unsafe: near the basin edge it settles on a neighbouring module boundary ~0.6 px away
//! and still decodes).
//!
//! This is the internal engine working on rays and [`View`]s; the public, validated API
//! is in [`crate::api`].

use std::collections::BTreeMap;
use std::time::Instant;

use crate::bridge::{
    Bridge, BridgePass, BridgeProv, K_SCAN_PLAN, K_SCAN_PLAN_PASSES, K_TRACK_PLAN,
};
use crate::codes::*;
use crate::decode::{
    border_dark_fraction, decode_rays, decode_verify, dict_info, homography_closed, plane_to_ray,
    DecodeConfig, DecodeResult,
};
use crate::error::EqarError;
use crate::geometry::{dot, norm, normalize, px_err, ray_to_px, Vec3, K_PI};
use crate::guards::{
    apply_gates, corner_states, ring_width, side_agrees, GateInput, K_CONSIST_ANGLE_RAD,
    K_CONSIST_OFFSET_CELL, K_CORNER_PRIOR, K_CORNER_RING, K_CORNER_SIDE, K_PRIOR_K,
    K_PRIOR_MIN_CONSISTENT, K_RING_MAX, K_RING_MIN, K_RING_UNJUDGED_SCALE,
};
use crate::image::View;
use crate::params::{
    cold_decode_config, resolve_quality, verify_decode_config, Params, Quality, QualityPreset,
};
use crate::proposal::{
    cap_candidates, dedupe_candidates, order_for_budget, Candidate, Proposal,
    K_MAX_CANDIDATES_PER_FRAME,
};
use crate::refine::{
    refine_rays, RefineCfg, RefineInfo, RefineScratch, K_EDGE_MIN_SUPPORT, K_MIN_CONTRAST_DN,
    K_SCAN_REFINE, K_TRACK_REFINE,
};
use crate::track::{predict_next, track_update, TrackState};

/// One accepted (or, for a failed attempt, rejected) detection, all in the final
/// TL,TR,BR,BL order.
#[derive(Debug, Clone, Copy, Default)]
pub struct Detection {
    pub ray: [Vec3; 4],
    pub corner_px: [[f64; 2]; 4],
    /// Projective centre of the final refined quad.
    pub center_ray: Vec3,
    pub center_px: [f64; 2],
    pub id: i32,
    pub reject_code: i32,
    pub flags: u32,
    pub quality: f32,
    pub side_px: f32,
    pub edge_resid_px: f32,
    pub contrast_dn: f32,
    pub min_cross: f32,
    /// The corner is a measurement (see `guards`).
    pub corner_valid: [u8; 4],
    /// `K_CORNER_*` bits.
    pub corner_flags: [u8; 4],
    /// >= 1, the corner's sigma multiplier.
    pub sigma_scale: [f32; 4],
    /// Tip-window inlier fraction.
    pub end_support: [f32; 4],
    /// Min whole-edge inlier fraction of the two edges at the corner.
    pub edge_support: [f32; 4],
    /// Max inlier residual (px_eq) of the two edges at the corner.
    pub corner_resid: [f32; 4],
    /// `ring_width` of side i (corner i -> i+1), -1 = not judged.
    pub ring: [f32; 4],
    /// `BridgeProv` of the quad the detection was refined from.
    pub prov: u8,
    /// The accepted decode's counts.
    pub n_bit_errors: i32,
    pub border_fail: i32,
    pub n_erased: i32,
}

/// A known marker and where it is expected (rays in the marker's own TL,TR,BR,BL order).
#[derive(Debug, Clone, Copy, Default)]
pub struct Prior {
    pub id: i32,
    pub ray: [Vec3; 4],
    /// Angular 1-sigma, rad; > 0.
    pub sigma_rad: f64,
}

/// Tracking search/consistency sigma, degrees per frame of gap (so it grows with the gap).
pub const K_TRACK_SIGMA_DEG: f64 = 0.5;

/// Counters of the last call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub n_candidates: i32,
    pub n_bridge_ok: i32,
    pub n_decode_ok: i32,
    pub n_accepted: i32,
    pub n_rejected: i32,
    pub n_budget_dropped: i32,
    pub best_reject_code: i32,
    /// Post-dedupe count of reference-strict f=4 candidates.
    pub n_tier1: i32,
    /// Whether the f=2 pass actually fired.
    pub ran_small: i32,
    /// Candidates the pre-bridge border probe screened out before `Bridge::run` was ever
    /// called. `n_rejected` keeps its meaning of "reached the bridge and failed
    /// afterwards".
    pub n_prebridge_rejected: i32,
    /// Post-refine re-checks of LINEFIT/RELAXED quads: decode re-checks run.
    pub n_recheck: i32,
    /// Sides whose refined edge left the bridge line (cold: the marker is rejected, known
    /// id: the side's corners go invalid).
    pub n_side_disagree: i32,
}

/// The pre-bridge cheap rejection (`border_dark_fraction`): sub-samples per cell axis for
/// the probe, and the share of the `4 * ncell - 4` border cells that must land on the dark
/// side of the cell-mean Otsu split for the candidate to be worth a bridge.
///
/// The margin is permissive on purpose: the probe reads the PROPOSAL quad (the blob's four
/// diagonal extreme points), not a polished contour, so a couple of border cells routinely
/// sample across a real marker's edge. On real footage and the synthetic grids a true
/// marker reads 0.95-1.00 and background clutter 0.40-0.65, so 0.75 sits in an empty band
/// between the two populations.
pub const K_PRE_BRIDGE_SUB: i32 = 2;
pub const K_PRE_BRIDGE_BORDER_FRAC: f32 = 0.75;

/// The f=2 small-marker pass costs ~4x the f=4 band. It fires whenever `use_second_scale` is
/// set and at least this many ms of the Stage 2-5 budget are left after the f=4 candidates,
/// so the pass is never started only to be abandoned mid-way.
///
/// It used to fire only when nothing decoded at f=4 AND f=4 produced no tier-1 candidate.
/// In a scene with several markers one larger marker satisfies neither, and every marker
/// below the f=4 floor (~24 px_eq) was then lost: on synthetic 6528 px frames with 8 markers
/// of 20-40 px that gate cost 66.9 % -> 90.1 % detection (no false positives either way).
/// Under the 2 ms video budget the f=4 candidates usually leave less than this, so video
/// timing is unchanged; a one-shot detect pays ~3-4 ms.
pub const K_SMALL_PASS_MIN_BUDGET_MS: f64 = 1.0;

/// How far a candidate got before failing, most advanced (most informative) first.
fn reason_rank(code: i32) -> i32 {
    match code {
        R_ID_NOT_EXPECTED => 0,
        R_BORDER_NOT_BLACK | R_IDENTIFY_FAILED => 1,
        R_INIT_GEOMETRY | R_DISPLACEMENT | R_LOW_CONTRAST | R_HIGH_RESID => 2,
        R_DEGENERATE_CORNER => 3,
        R_EDGE_FIT => 4,
        R_DEGENERATE_INIT => 5,
        R_BRIDGE_FAILED => 6,
        R_NO_CANDIDATE => 7,
        _ => 8,
    }
}

fn fold_best(best: &mut i32, candidate: i32) {
    if reason_rank(candidate) < reason_rank(*best) {
        *best = candidate;
    }
}

/// Per-candidate extra flags for the scan path: `FLAG_SCAN` always, plus `FLAG_POLE_CAP` /
/// `FLAG_SECOND_SCALE` when the proposal candidate itself says so.
fn candidate_flags(c: &Candidate) -> u32 {
    let mut f = FLAG_SCAN;
    if c.from_cap {
        f |= FLAG_POLE_CAP;
    }
    if c.scale == 2 {
        f |= FLAG_SECOND_SCALE;
    }
    f
}

#[inline]
/// Whether ray `v` lies inside an accepted detection of `raw`: within the largest
/// centre-to-corner angle of that detection's quad.
fn inside_accepted(raw: &[Detection], v: &Vec3) -> bool {
    raw.iter().any(|d| {
        let ang = |a: &Vec3, b: &Vec3| dot(a, b).clamp(-1.0, 1.0).acos();
        let r = d
            .ray
            .iter()
            .map(|c| ang(c, &d.center_ray))
            .fold(0.0, f64::max);
        ang(v, &d.center_ray) <= r
    })
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

fn sum4(r: &[Vec3; 4]) -> Vec3 {
    Vec3::new(
        r[0].x + r[1].x + r[2].x + r[3].x,
        r[0].y + r[1].y + r[2].y + r[3].y,
        r[0].z + r[1].z + r[2].z + r[3].z,
    )
}

fn valid_in_use(ids: &Option<Vec<i32>>, dict: i32) -> bool {
    let Some(ids) = ids else { return true };
    let Ok(d) = dict_info(dict) else { return false };
    ids.iter().all(|&i| i >= 0 && i < d.n)
}

/// One work item of `track`: the predicted corners of a known marker at this frame.
#[derive(Clone, Copy)]
struct Item {
    id: i32,
    pred: [Vec3; 4],
    r: f64,
    sigma_rad: f64,
}

/// The pipeline engine. One thread per detector; all per-frame scratch lives here.
pub struct Detector {
    w: i32,
    h: i32,
    params: Params,
    /// Cold-decode / verify in-use set (`None` = every id).
    in_use: Option<Vec<i32>>,
    /// Non-null while verifying ONE prior: the known-ID mode of `finish_candidate`.
    vprior: Option<Prior>,
    consist_angle: f64,
    consist_off: f64,
    /// `track`: frame clock, advanced by `frame_gap` per call.
    clock: f64,
    proposal: Proposal,
    bridge: Bridge,
    refine_scratch: RefineScratch,
    candidates: Vec<Candidate>,
    small_candidates: Vec<Candidate>,
    track_states: BTreeMap<i32, TrackState>,
    stats: Stats,
    timings: [f64; 6],
    last_n_sub: i32,
    // Reused per-call buffers (taken out of `self` while a call runs).
    raw_scan: Vec<Detection>,
    raw_track: Vec<Detection>,
    raw_verify: Vec<Detection>,
    kept: Vec<Detection>,
    failed_ids: Vec<i32>,
    seen_ids: Vec<i32>,
    items: Vec<Item>,
}

impl Detector {
    /// `Err` for a size that is not 2:1 with a width divisible by 4, an unknown dictionary
    /// or an invalid in-use set.
    pub fn new(w: i32, h: i32, params: &Params) -> Result<Detector, EqarError> {
        if w <= 0 || h <= 0 || w != 2 * h || w % 4 != 0 {
            return Err(EqarError::Size);
        }
        let dict = params.dict.index();
        if !valid_in_use(&params.in_use_ids, dict) {
            return Err(EqarError::Params);
        }
        Ok(Detector {
            w,
            h,
            params: params.clone(),
            in_use: params.in_use_ids.clone(),
            vprior: None,
            consist_angle: K_CONSIST_ANGLE_RAD,
            consist_off: K_CONSIST_OFFSET_CELL,
            clock: 0.0,
            proposal: Proposal::new(w, h)?,
            // The bridge's screen pass lays a border ring out on the SAME dictionary the
            // decoder uses.
            bridge: Bridge::new(w, h, dict)?,
            refine_scratch: RefineScratch::default(),
            candidates: Vec::new(),
            small_candidates: Vec::new(),
            track_states: BTreeMap::new(),
            stats: Stats::default(),
            timings: [0.0; 6],
            last_n_sub: 0,
            raw_scan: Vec::new(),
            raw_track: Vec::new(),
            raw_verify: Vec::new(),
            kept: Vec::new(),
            failed_ids: Vec::new(),
            seen_ids: Vec::new(),
            items: Vec::new(),
        })
    }

    /// Cold-decode in-use set. `false` (and the previous config kept) when an id is
    /// outside the dictionary. `None` = every id, `Some(&[])` = no id.
    pub fn set_in_use(&mut self, ids: Option<&[i32]>) -> bool {
        let owned = ids.map(|s| s.to_vec());
        if !valid_in_use(&owned, self.params.dict.index()) {
            return false;
        }
        self.in_use = owned;
        true
    }

    /// Tighten/loosen the post-refine side-consistency bounds (default `guards`').
    pub fn set_consistency_bounds(&mut self, angle_rad: f64, offset_cell: f64) {
        self.consist_angle = angle_rad;
        self.consist_off = offset_cell;
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// 6 values, ms: `[0]` propose `[1]` bridge `[2]` decode `[3]` refine `[4]` gate
    /// `[5]` total.
    pub fn timings(&self) -> &[f64; 6] {
        &self.timings
    }

    /// The decode `n_sub` actually used by the most recent `scan` / `scan_impl` (including
    /// the once-per-frame fallback inside `track`): lets tests confirm the fallback ran at
    /// the quality level its `FLAG_LEAN` claims.
    pub fn last_n_sub(&self) -> i32 {
        self.last_n_sub
    }

    pub fn reset_track(&mut self) {
        self.track_states.clear();
        self.clock = 0.0;
    }

    pub fn has_coasting(&self) -> bool {
        self.params.coast_max > 0 && !self.track_states.is_empty()
    }

    fn check_view(&self, v: &View) -> Result<(), EqarError> {
        v.check()?;
        if v.w != self.w || v.h != self.h {
            return Err(EqarError::Size);
        }
        Ok(())
    }

    fn decode_one(&self, v: &View, rays: &[Vec3; 4], n_sub: i32) -> DecodeResult {
        let dict = self.params.dict.index();
        if let Some(p) = &self.vprior {
            let vc = verify_decode_config(self.in_use.as_deref());
            return decode_verify(v, dict, rays, n_sub, p.id, -1, &vc);
        }
        let dc: DecodeConfig = cold_decode_config(self.in_use.as_deref());
        decode_rays(v, dict, rays, n_sub, Some(&dc))
    }

    /// Stages 3-5 for one bridge quad; `true` and a filled `d` on acceptance, `false` with
    /// `d.reject_code` set otherwise.
    ///
    /// `skip_predecode` is the track path: the prior already says a marker is here, so the
    /// pre-refine decode -- whose whole job is to kill a code-less scan candidate before it
    /// pays for a refine -- is pure cost there. Refine is then seeded off the raw bridge
    /// quad and the post-refine decode remains the sole authority on the id. `rcfg_in` is
    /// the refine schedule. `prov` is the seed's provenance: a LINEFIT/RELAXED seed gets
    /// the post-refine consistency check.
    fn finish_candidate(
        &mut self,
        v: &View,
        seed: &[Vec3; 4],
        d: &mut Detection,
        n_sub: i32,
        skip_predecode: bool,
        rcfg_in: &RefineCfg,
        prov: u8,
    ) -> bool {
        // 3-edge completion only when the id is known (verify against a prior).
        let mut rcfg = *rcfg_in;
        rcfg.allow_weak = self.vprior.is_some();
        // Stage 3 (decode-first): decode the BRIDGE quad before paying for a refine that a
        // code-less blob would never survive. The track path skips it.
        let mut dec1 = DecodeResult {
            id: -1,
            rot: -1,
            n_bit_errors: 0,
            n_erased: 0,
            border_fail: 0,
            border_ok: false,
            contrast: 0.0,
            bits: 0,
            corners: *seed,
            reject_code: R_IDENTIFY_FAILED,
        };
        if !skip_predecode {
            let t0 = Instant::now();
            dec1 = self.decode_one(v, seed, n_sub);
            self.timings[2] += ms_since(t0);
        }
        let dec1_ok = dec1.reject_code == ACCEPT;
        if dec1_ok {
            self.stats.n_decode_ok += 1;
        }

        // On real (noisy/blurred/compressed) footage the BRIDGE quad alone is sometimes
        // imprecise enough that the exact-homography decode misreads a genuine marker --
        // a border cell samples "on", or a codeword bit flips -- before any edge
        // refinement ever runs. A real marker's cells are never uniform (dec1.contrast,
        // the raw cell-mean spread, is exactly 0 for a solid/code-less blob), so seed the
        // refine off the RAW bridge quad instead of rejecting outright when the failure is
        // one corner precision could plausibly explain (border_not_black /
        // identify_failed) AND there is real cell contrast to refine against. A genuinely
        // degenerate quad (homography itself failed) or a code-less blob (contrast == 0)
        // is still rejected immediately at zero extra cost.
        let recoverable = skip_predecode
            || (!dec1_ok
                && (dec1.reject_code == R_BORDER_NOT_BLACK
                    || dec1.reject_code == R_IDENTIFY_FAILED)
                && dec1.contrast > K_MIN_CONTRAST_DN);
        if !dec1_ok && !recoverable {
            d.reject_code = dec1.reject_code;
            d.id = -1;
            return false;
        }
        // dec1.corners is only decode-ordered (winding- and rotation-corrected) when dec1
        // itself succeeded; in the recovered case fall back to the raw bridge quad --
        // refine_rays needs a consistently-wound quad, not a canonically-rotated one.
        let init_rays: [Vec3; 4] = if dec1_ok { dec1.corners } else { *seed };

        // Stage 4: great-circle refine, seeded from decode's canonically-ordered corners
        // -- or, in the recovered case above, directly off the bridge quad. Exactly ONE
        // refine call: a second pass re-refining an already-refined seed would measure
        // Gate 2's displacement against an intermediate position instead of the true
        // bridge quad, spuriously rejecting a legitimate multi-pixel correction as drift.
        let mut info = RefineInfo::default();
        let mut refined = [Vec3::default(); 4];
        let t1 = Instant::now();
        let rok = refine_rays(
            v,
            &init_rays,
            &mut refined,
            &mut info,
            &rcfg,
            &mut self.refine_scratch,
        );
        self.timings[3] += ms_since(t1);
        if !rok {
            d.reject_code = info.reject_code;
            d.id = -1;
            return false;
        }

        // Stage 5 "decode confirmation": the REFINED corners are more precise than either
        // the raw bridge quad or dec1's read of it, so THIS decode -- not dec1 -- is
        // authoritative for both the border/contrast gate below and the final id. (Gate 2
        // below already bounds how far the refined quad may move from the bridge quad, so
        // a wild "walked onto a different module" jump is rejected there.)
        let t3 = Instant::now();
        let dec2 = self.decode_one(v, &refined, n_sub);
        self.timings[2] += ms_since(t3);
        if dec2.reject_code != ACCEPT {
            d.reject_code = dec2.reject_code;
            d.id = -1;
            return false;
        }

        // Stage 5: diagnostic gates, on the BRIDGE quad as init_rays (never the post-fit
        // normals), and the post-refine decode for the border gate.
        let gi = GateInput {
            init_rays: &init_rays,
            final_rays: &refined,
            info: &info,
            decode: &dec2,
            reject_high_resid: self.params.reject_high_resid,
        };
        let t2 = Instant::now();
        let go = apply_gates(&gi);
        self.timings[4] += ms_since(t2);
        if go.code != ACCEPT {
            d.reject_code = go.code;
            d.id = -1;
            return false;
        }
        let (flags, quality) = (go.flags, go.quality);

        if self.params.expect_id >= 0 && dec2.id != self.params.expect_id {
            d.reject_code = R_ID_NOT_EXPECTED;
            d.id = -1;
            return false;
        }

        // Per-corner validity (refine seed order), then map to the final order.
        let known = self.vprior.is_some();
        let mut cs = corner_states(&info, known);
        let mut perm = [0usize; 4]; // final corner i = refined corner perm[i]
        for i in 0..4 {
            let (mut bj, mut bd) = (0usize, -2.0f64);
            for j in 0..4 {
                let dd = dot(&dec2.corners[i], &refined[j]);
                if dd > bd {
                    bd = dd;
                    bj = j;
                }
            }
            perm[i] = bj;
        }
        // Post-refine consistency (LINEFIT / RELAXED seeds): the refined edge must agree
        // with the bridge's own line, and the decode must not have moved. Cold: reject.
        // Known id: the side's corners go invalid (a decode disagreement rejects in both
        // modes).
        if prov != BridgeProv::Legacy as u8 {
            let pixr = K_PI / self.h as f64;
            let identity = (0..4).all(|i| perm[i] == i);
            // The decode re-check needs a pre-refine decode to compare with. When it failed
            // (the `recoverable` branch: a bitten quad often does not decode at all before
            // refinement), dec2 is the only decode of this candidate, so there is nothing
            // to disagree with; the quad is judged by its sides alone.
            if dec1_ok {
                self.stats.n_recheck += 1;
            }
            if dec1_ok && (dec2.id != dec1.id || !identity || dec2.n_bit_errors > dec1.n_bit_errors)
            {
                d.reject_code = R_IDENTIFY_FAILED;
                d.id = -1;
                return false;
            }
            for e in 0..4 {
                if info.edge[e].support < K_EDGE_MIN_SUPPORT as f32 {
                    continue;
                }
                if side_agrees(
                    &init_rays,
                    &refined,
                    e,
                    info.module_px,
                    pixr,
                    self.consist_angle,
                    self.consist_off,
                ) {
                    continue;
                }
                self.stats.n_side_disagree += 1;
                if !known {
                    d.reject_code = R_DISPLACEMENT; // counted in stats().n_side_disagree
                    d.id = -1;
                    return false;
                }
                for k in [e, (e + 1) & 3] {
                    cs[k].valid = false;
                    cs[k].flags |= K_CORNER_SIDE;
                }
            }
        }
        for i in 0..4 {
            let c = &cs[perm[i]];
            d.corner_valid[i] = c.valid as u8;
            d.corner_flags[i] = c.flags;
            d.sigma_scale[i] = c.sigma_scale;
            d.end_support[i] = c.end_support;
            d.edge_support[i] = c.edge_support;
            d.corner_resid[i] = c.resid_px;
        }
        // Ring-width check: a side whose inner black band is too narrow was locked onto an
        // occluder's boundary; both of its corners are invalid, in BOTH modes. Never
        // rejects the marker.
        {
            let pixr = K_PI / self.h as f64;
            let dict = self.params.dict.index();
            for s in 0..4 {
                d.ring[s] = ring_width(v, dict, dec2.id, &dec2.corners, s, pixr);
                // Known ID only: a side whose band reads wider than K_RING_MAX sits
                // OUTSIDE the true border (an occluder merged into it), the mirror image
                // of the too-narrow case.
                if d.ring[s] >= 0.0
                    && ((d.ring[s] as f64) < K_RING_MIN
                        || (known && (d.ring[s] as f64) > K_RING_MAX))
                {
                    for k in [s, (s + 1) & 3] {
                        d.corner_valid[k] = 0;
                        d.corner_flags[k] |= K_CORNER_RING;
                    }
                }
            }
            // Known ID only: a side whose ring could not be judged (no evidence the border
            // is intact) never gives its two corners full confidence.
            if known {
                for s in 0..4 {
                    if d.ring[s] < 0.0 {
                        for k in [s, (s + 1) & 3] {
                            d.sigma_scale[k] = d.sigma_scale[k].max(K_RING_UNJUDGED_SCALE);
                        }
                    }
                }
            }
        }
        // Prior consistency (known ID only): a valid, non-COMPLETED corner must lie within
        // K_PRIOR_K * sigma of its predicted ray, else it is invalid with PRIOR; fewer than
        // K_PRIOR_MIN_CONSISTENT consistent measured corners reject the marker (a look-alike
        // at the wrong place, or a prior that points elsewhere).
        if let Some(pr) = &self.vprior {
            let tol = K_PRIOR_K * pr.sigma_rad;
            let mut n_ok = 0;
            for i in 0..4 {
                if d.corner_flags[i] & crate::guards::K_CORNER_COMPLETED != 0 {
                    continue;
                }
                let c = dot(&dec2.corners[i], &pr.ray[i]);
                let c = c.clamp(-1.0, 1.0);
                if c.acos() <= tol {
                    n_ok += 1;
                    continue;
                }
                d.corner_valid[i] = 0;
                d.corner_flags[i] |= K_CORNER_PRIOR;
            }
            if n_ok < K_PRIOR_MIN_CONSISTENT {
                d.reject_code = R_ID_NOT_EXPECTED;
                d.id = -1;
                return false;
            }
        }
        d.prov = prov;
        d.n_bit_errors = dec2.n_bit_errors;
        d.border_fail = dec2.border_fail;
        d.n_erased = dec2.n_erased;

        for i in 0..4 {
            d.ray[i] = dec2.corners[i];
            let (px, py) = ray_to_px(&dec2.corners[i], self.w, self.h);
            d.corner_px[i] = [px, py];
        }

        // The marker centre: the projective image of the marker-plane origin under the
        // closed-form homography of the FINAL (refined, TL,TR,BR,BL-ordered) corner rays,
        // i.e. the exact projective centre of the quad -- NOT the mean of the corners,
        // which is biased under perspective and meaningless across the +-180 seam. The
        // homography's overall scale/sign is fixed by an arbitrary a0 = 1, not by which
        // hemisphere the marker is in, so the sign is flipped to make dot(c, mean) > 0.
        // homography_closed can only fail on a near-degenerate quad, which dec2.corners
        // already is not (the SAME rays, cyclically rolled, just solved a homography
        // inside decode_rays); the mean-ray fallback is defensive only.
        {
            let mean = sum4(&dec2.corners);
            let centre = match homography_closed(&dec2.corners) {
                Some(hm) => {
                    let c = plane_to_ray(&hm, 0.0, 0.0);
                    if dot(&c, &mean) < 0.0 {
                        Vec3::new(-c.x, -c.y, -c.z)
                    } else {
                        c
                    }
                }
                None => normalize(&mean),
            };
            d.center_ray = centre;
            let (cx, cy) = ray_to_px(&centre, self.w, self.h);
            d.center_px = [cx, cy];
        }
        d.id = dec2.id;
        d.reject_code = ACCEPT;
        d.flags = flags;
        d.quality = quality;
        d.side_px = info.side_px as f32;
        d.edge_resid_px = info.resid_px as f32;
        d.contrast_dn = info.contrast_dn as f32;
        d.min_cross = info.min_cross as f32;
        true
    }

    /// `bridge.run()` + `finish_candidate()` for one candidate ray/radius seed; on success
    /// appends to `raw` (with `extra_flags` OR'd in) and returns `true`; on failure folds
    /// the reject code into `best_code` by rank and returns `false`. Always counts one
    /// "attempt".
    fn process_seed(
        &mut self,
        v: &View,
        centre: &Vec3,
        radius: f64,
        n_sub: i32,
        extra_flags: u32,
        raw: &mut Vec<Detection>,
        best_code: &mut i32,
        attempted: &mut i32,
        plan: &[BridgePass],
        skip_predecode: bool,
        rcfg: &RefineCfg,
    ) -> Result<bool, EqarError> {
        *attempted += 1;
        let mut seed = [Vec3::default(); 4];
        let tb0 = Instant::now();
        let bok = self.bridge.run(v, *centre, radius, &mut seed, plan)?;
        self.timings[1] += ms_since(tb0);
        if !bok {
            fold_best(best_code, R_BRIDGE_FAILED);
            return Ok(false);
        }
        self.stats.n_bridge_ok += 1;
        let mut d = Detection::default();
        let prov = self.bridge.prov() as u8;
        if self.finish_candidate(v, &seed, &mut d, n_sub, skip_predecode, rcfg, prov) {
            d.flags |= extra_flags;
            raw.push(d);
            return Ok(true);
        }
        // The bridge's 4-line-fit quad did not survive; try the legacy 4-gon it replaced
        // before giving the candidate up.
        if let Some(alt) = self.bridge.alt_quad() {
            fold_best(best_code, d.reject_code);
            let mut d2 = Detection::default();
            if self.finish_candidate(
                v,
                &alt,
                &mut d2,
                n_sub,
                skip_predecode,
                rcfg,
                BridgeProv::Legacy as u8,
            ) {
                d2.flags |= extra_flags;
                raw.push(d2);
                return Ok(true);
            }
            d = d2;
        }
        fold_best(best_code, d.reject_code);
        Ok(false)
    }

    /// Is this candidate worth a `Bridge::run`? A tier-1 candidate (reference-strict shape
    /// AND a hole AND the dark-ring probe) ALWAYS is -- it is the shape a real marker
    /// makes, it is what `order_for_budget` puts first, and the whole real-clip gate rests
    /// on it never being screened out. A tier-2 candidate must show a predominantly dark
    /// border ring on its own proposal quad first. A candidate with no quad at all, or one
    /// whose quad is too degenerate to solve a homography from, is passed through
    /// unscreened -- the probe never rejects on an unmeasurable reading. A tier-2
    /// candidate whose cell means have no spread at all (a featureless rectangle) is
    /// rejected outright.
    fn worth_bridging(&self, v: &View, c: &Candidate) -> bool {
        if c.tier == 1 {
            return true; // never risk the true marker
        }
        if !c.has_quad {
            return true; // nothing to probe
        }
        let mut contrast = 0.0f32;
        let frac = border_dark_fraction(
            v,
            self.params.dict.index(),
            &c.quad,
            K_PRE_BRIDGE_SUB,
            Some(&mut contrast),
        );
        if frac < 0.0 {
            return true; // degenerate quad: unmeasurable
        }
        // The contrast half is not redundant with the fraction: on a UNIFORM patch
        // otsu_exact has no valid split and returns min + 1, which puts every cell --
        // border included -- on the dark side and reads a perfect 1.0. A featureless dark
        // rectangle is exactly the clutter this screen exists for, and it is the case a
        // border-fraction test alone waves through. A real marker's cell means span the
        // black ring and the white data modules, so its spread is tens of DN, not units.
        if contrast < K_MIN_CONTRAST_DN {
            return false;
        }
        frac >= K_PRE_BRIDGE_BORDER_FRAC
    }

    /// Per-id, seam-aware result dedupe: descending `side_px`, drop a same-id detection
    /// whose mean corner distance to an already-kept one is < 0.5 * max(side_px, 1).
    fn dedupe_by_id(&mut self, v: &mut Vec<Detection>) {
        // stable, like the descending comparator being applied to an already-ordered list
        v.sort_by(|a, b| {
            b.side_px
                .partial_cmp(&a.side_px)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.kept.clear();
        self.kept.reserve(v.len());
        for d in v.iter() {
            let mut dup = false;
            for e in &self.kept {
                if e.id != d.id {
                    continue;
                }
                let mut sum = 0.0;
                for i in 0..4 {
                    sum += px_err(
                        d.corner_px[i][0],
                        d.corner_px[i][1],
                        e.corner_px[i][0],
                        e.corner_px[i][1],
                        self.w,
                    );
                }
                let mean = sum / 4.0;
                if mean < 0.5 * (d.side_px as f64).max(1.0) {
                    dup = true;
                    break;
                }
            }
            if !dup {
                self.kept.push(*d);
            }
        }
        std::mem::swap(v, &mut self.kept);
    }

    /// Full-frame scan: stages 1-5 over the proposal candidates, within the time budget.
    pub fn scan(&mut self, v: &View, cap: usize) -> Result<Vec<Detection>, EqarError> {
        self.scan_impl(v, cap, resolve_quality(self.params.quality))
    }

    /// Real body of `scan`: takes the quality preset explicitly so `track`'s once-per-frame
    /// fallback can force the SAME lean-or-default preset it resolved for the per-marker
    /// re-seed loop above it.
    fn scan_impl(
        &mut self,
        v: &View,
        cap: usize,
        qp: QualityPreset,
    ) -> Result<Vec<Detection>, EqarError> {
        self.check_view(v)?;
        self.stats = Stats::default();
        self.bridge.begin_frame();
        self.timings = [0.0; 6];
        let t_total0 = Instant::now();
        self.last_n_sub = qp.n_sub;

        let mut candidates = std::mem::take(&mut self.candidates);
        candidates.clear();
        let mut raw = std::mem::take(&mut self.raw_scan);
        raw.clear();
        let res = self.scan_body(v, cap, qp, &mut candidates, &mut raw, t_total0);
        self.candidates = candidates;
        let out = res.map(|n_out| raw[..n_out].to_vec());
        self.raw_scan = raw;
        out
    }

    fn scan_body(
        &mut self,
        v: &View,
        cap: usize,
        qp: QualityPreset,
        candidates: &mut Vec<Candidate>,
        raw: &mut Vec<Detection>,
        t_total0: Instant,
    ) -> Result<usize, EqarError> {
        let tp0 = Instant::now();
        self.proposal.run(v, candidates)?;
        self.timings[0] += ms_since(tp0);
        dedupe_candidates(candidates);
        order_for_budget(candidates);
        cap_candidates(candidates, K_MAX_CANDIDATES_PER_FRAME);
        self.stats.n_candidates = candidates.len() as i32;
        self.stats.n_tier1 = candidates.iter().filter(|c| c.tier == 1).count() as i32;

        raw.reserve(candidates.len());
        let mut best_code = R_NO_CANDIDATE;
        let mut attempted = 0;

        // Stage 2-5 candidate budget (Params::time_budget_ms) starts AFTER proposal -- it
        // is documented as the "Stage 2-5" budget, not stage 1.
        let budget = self.params.time_budget_ms as f64;
        let t_budget0 = Instant::now();
        let mut idx = 0;
        while idx < candidates.len() {
            if ms_since(t_budget0) >= budget {
                break;
            }
            let c = candidates[idx];
            idx += 1;
            // The cheap screen in front of the bridge. A candidate it turns away costs
            // ~1 us instead of ~0.9 ms and is NOT counted as an attempt -- it never
            // reached Stage 2, so folding a Stage-3 code (border_not_black) into
            // best_reject_code would overstate how far the frame actually got.
            if !self.worth_bridging(v, &c) {
                self.stats.n_prebridge_rejected += 1;
                continue;
            }
            self.process_seed(
                v,
                &c.v,
                c.r,
                qp.n_sub,
                candidate_flags(&c),
                raw,
                &mut best_code,
                &mut attempted,
                &K_SCAN_PLAN[..K_SCAN_PLAN_PASSES],
                false,
                &K_SCAN_REFINE,
            )?;
        }
        if idx < candidates.len() {
            self.stats.n_budget_dropped += (candidates.len() - idx) as i32;
        }

        // The f=2 small-marker pass: see K_SMALL_PASS_MIN_BUDGET_MS for the rule.
        if self.params.use_second_scale
            && budget - ms_since(t_budget0) >= K_SMALL_PASS_MIN_BUDGET_MS
        {
            self.stats.ran_small = 1;
            let mut small = std::mem::take(&mut self.small_candidates);
            small.clear();
            let r = self.small_pass(
                v,
                qp,
                &mut small,
                raw,
                &mut best_code,
                &mut attempted,
                t_budget0,
                budget,
            );
            self.small_candidates = small;
            r?;
        }

        let mut raw_v = std::mem::take(raw);
        self.dedupe_by_id(&mut raw_v);
        *raw = raw_v;

        let n_out = raw.len().min(cap);
        self.stats.n_accepted = n_out as i32;
        self.stats.n_rejected = attempted - raw.len() as i32;
        self.stats.best_reject_code = best_code;
        self.timings[5] = ms_since(t_total0);
        Ok(n_out)
    }

    fn small_pass(
        &mut self,
        v: &View,
        qp: QualityPreset,
        small: &mut Vec<Candidate>,
        raw: &mut Vec<Detection>,
        best_code: &mut i32,
        attempted: &mut i32,
        t_budget0: Instant,
        budget: f64,
    ) -> Result<(), EqarError> {
        let ts0 = Instant::now();
        self.proposal.run_small(v, small)?;
        self.timings[0] += ms_since(ts0);
        dedupe_candidates(small);
        // The pass only exists for what f=4 could not see: a candidate whose centre lies
        // inside a marker already accepted (its own duplicate, or one of its modules) is
        // dropped before it is counted or bridged, so the f=4 results are never replaced.
        small.retain(|c| !inside_accepted(raw, &c.v));
        order_for_budget(small);
        cap_candidates(small, K_MAX_CANDIDATES_PER_FRAME);
        self.stats.n_candidates += small.len() as i32;

        let mut sidx = 0;
        while sidx < small.len() {
            if ms_since(t_budget0) >= budget {
                break;
            }
            let c = small[sidx];
            sidx += 1;
            if !self.worth_bridging(v, &c) {
                self.stats.n_prebridge_rejected += 1;
                continue;
            }
            self.process_seed(
                v,
                &c.v,
                c.r,
                qp.n_sub,
                candidate_flags(&c),
                raw,
                best_code,
                attempted,
                &K_SCAN_PLAN[..K_SCAN_PLAN_PASSES],
                false,
                &K_SCAN_REFINE,
            )?;
        }
        if sidx < small.len() {
            self.stats.n_budget_dropped += (small.len() - sidx) as i32;
        }
        Ok(())
    }

    /// One prior of `verify`: lenient bridge + refine with the prior's id, appended to
    /// `raw` on success.
    fn verify_one(
        &mut self,
        v: &View,
        pr: &Prior,
        n_sub: i32,
        extra_flags: u32,
        raw: &mut Vec<Detection>,
        best_code: &mut i32,
        attempted: &mut i32,
    ) -> Result<bool, EqarError> {
        let centre = normalize(&sum4(&pr.ray));
        let mut r = 0.0f64;
        for k in 0..4 {
            let c = dot(&centre, &pr.ray[k]);
            let c = c.clamp(-1.0, 1.0);
            r = r.max(c.acos());
        }
        // The search window is the bridge patch around the predicted quad; it is framed on
        // the prior's own size. Lenient handling (relaxed contour selection, 3-edge mode,
        // verify decode) holds for THIS candidate only.
        self.vprior = Some(*pr);
        self.bridge.set_lenient(true);
        self.bridge.begin_frame();
        let ok = self.process_seed(
            v,
            &centre,
            r,
            n_sub,
            extra_flags,
            raw,
            best_code,
            attempted,
            &K_SCAN_PLAN[..K_SCAN_PLAN_PASSES],
            false,
            &K_SCAN_REFINE,
        );
        self.bridge.set_lenient(false);
        self.vprior = None;
        ok
    }

    /// The known-ID path: for each prior, bridge/refine near its predicted quad with
    /// lenient handling and `allow_weak` for THAT candidate only, then
    /// `decode_verify(expected_id = prior.id)` and the prior-consistency check. Output is
    /// in prior order (a prior that does not verify yields nothing).
    pub fn verify(
        &mut self,
        v: &View,
        priors: &[Prior],
        cap: usize,
    ) -> Result<Vec<Detection>, EqarError> {
        self.check_view(v)?;
        self.stats = Stats::default();
        self.bridge.begin_frame();
        self.timings = [0.0; 6];
        let t_total0 = Instant::now();
        self.last_n_sub = resolve_quality(self.params.quality).n_sub;
        if priors.is_empty() || cap < 1 {
            return Ok(Vec::new());
        }

        let qp = resolve_quality(self.params.quality);
        let mut raw = std::mem::take(&mut self.raw_verify);
        raw.clear();
        let mut best_code = R_NO_CANDIDATE;
        let mut attempted = 0;
        let mut res = Ok(());
        for pr in priors {
            if raw.len() >= cap {
                break;
            }
            self.stats.n_candidates += 1;
            if let Err(e) = self.verify_one(
                v,
                pr,
                qp.n_sub,
                FLAG_TRACK,
                &mut raw,
                &mut best_code,
                &mut attempted,
            ) {
                res = Err(e);
                break;
            }
        }
        let n_out = raw.len().min(cap);
        let out = raw[..n_out].to_vec();
        self.stats.n_accepted = n_out as i32;
        self.stats.n_rejected = attempted - raw.len() as i32;
        self.stats.best_reject_code = best_code;
        self.timings[5] = ms_since(t_total0);
        self.raw_verify = raw;
        res.map(|_| out)
    }

    /// One fixed-size gnomonic re-seed per incoming marker (never a raw prior fed straight
    /// to the refiner), falling back to a single whole-frame `scan` when any marker's track
    /// attempt fails, matched back by id. `prev.len() >= 4` forces the lean preset for the
    /// whole call.
    ///
    /// `frame_gap` = frames since `prev` was observed (the predictor scales the angular
    /// velocity by it and the prior sigma grows with it). A marker that fails every attempt
    /// is kept internally as a COASTED predicted prior for `coast_max` frames (never
    /// output), and `prev` may be empty when such a marker exists. Order per marker: the
    /// cheap strict re-seed, then a verify against the predicted prior (lenient), then the
    /// once-per-frame scan fallback.
    pub fn track(
        &mut self,
        v: &View,
        prev: &[Detection],
        cap: usize,
        frame_gap: i32,
    ) -> Result<Vec<Detection>, EqarError> {
        self.check_view(v)?;
        self.stats = Stats::default();
        self.bridge.begin_frame();
        self.timings = [0.0; 6];
        let t_total0 = Instant::now();

        let frame_gap = frame_gap.max(1);
        self.clock += frame_gap as f64;
        let t_obs = self.clock - frame_gap as f64; // the frame `prev` was observed in
        if prev.is_empty() && !self.has_coasting() {
            return Ok(Vec::new());
        }

        // 4 or more markers may auto-downgrade to lean quality.
        let lean = prev.len() >= 4;
        let qp = resolve_quality(if lean {
            Quality::Lean
        } else {
            self.params.quality
        });
        let lean_flag = if lean { FLAG_LEAN } else { 0 };

        let mut raw = std::mem::take(&mut self.raw_track);
        raw.clear();
        let mut failed_ids = std::mem::take(&mut self.failed_ids);
        failed_ids.clear();
        let mut seen_ids = std::mem::take(&mut self.seen_ids);
        seen_ids.clear();
        let mut items = std::mem::take(&mut self.items);
        items.clear();
        let res = self.track_body(
            v,
            prev,
            cap,
            frame_gap,
            t_obs,
            qp,
            lean,
            lean_flag,
            &mut raw,
            &mut failed_ids,
            &mut seen_ids,
            &mut items,
            t_total0,
        );
        let out = res.map(|n_out| raw[..n_out].to_vec());
        self.raw_track = raw;
        self.failed_ids = failed_ids;
        self.seen_ids = seen_ids;
        self.items = items;
        out
    }

    fn track_body(
        &mut self,
        v: &View,
        prev: &[Detection],
        cap: usize,
        _frame_gap: i32,
        t_obs: f64,
        qp: QualityPreset,
        lean: bool,
        lean_flag: u32,
        raw: &mut Vec<Detection>,
        failed_ids: &mut Vec<i32>,
        seen_ids: &mut Vec<i32>,
        items: &mut Vec<Item>,
        t_total0: Instant,
    ) -> Result<usize, EqarError> {
        let mut best_code = R_NO_CANDIDATE;
        let mut attempted = 0;
        let clock = self.clock;

        let make_item = |st: &TrackState| -> Item {
            let step = clock - st.t_last;
            let (pred, r) = predict_next(st, step).unwrap_or((st.last, 0.0));
            Item {
                id: st.id,
                pred,
                r,
                sigma_rad: K_TRACK_SIGMA_DEG * (K_PI / 180.0) * step.max(1.0),
            }
        };

        // One work item per marker: predicted corners at this frame. Markers in `prev`
        // first, then the coasting ones (known from earlier calls, not in `prev`, missing
        // for <= coast_max frames).
        for p in prev {
            if p.id < 0 {
                continue;
            }
            // Skip anything that isn't a genuine unit-ish ray rather than feeding
            // NaN/garbage into the bridge. A per-component finiteness check alone lets
            // through finite-but-extreme garbage (components near f64::MAX individually
            // pass but overflow once squared and summed), and side_px is stored straight
            // into the TrackState: check the ray is close to unit length (which also
            // subsumes the per-component check: a NaN or infinite component makes norm()
            // non-finite too) and that side_px is finite.
            let mut finite_ray = true;
            for k in 0..4 {
                if !finite_ray {
                    break;
                }
                let len = norm(&p.ray[k]);
                finite_ray = len.is_finite() && len > 0.5 && len < 1.5;
            }
            if !finite_ray || !p.side_px.is_finite() {
                continue;
            }

            // Register the caller's prior as the newest confirmed observation (this is
            // what makes predict_next extrapolate correctly on the FOLLOWING call:
            // prev/last become the last two REAL, accepted poses, not this frame's
            // still-unknown one).
            let st = self.track_states.entry(p.id).or_default();
            let mut dt = if st.has_last { t_obs - st.t_last } else { 1.0 };
            if !(dt > 1e-9) {
                dt = 1.0;
            }
            track_update(st, &p.ray, p.id, p.side_px, dt, t_obs);
            seen_ids.push(p.id);
            items.push(make_item(st));
        }
        // Coasting markers: known, not seen in `prev`, missing for <= coast_max frames;
        // older ones are dropped.
        let coast_max = self.params.coast_max as f64;
        let mut drop_ids: Vec<i32> = Vec::new();
        for (id, st) in &self.track_states {
            if seen_ids.contains(id) {
                continue;
            }
            let missed = t_obs - st.t_last;
            if !st.has_last || missed > coast_max {
                drop_ids.push(*id);
                continue;
            }
            items.push(make_item(st));
        }
        for id in drop_ids {
            self.track_states.remove(&id);
        }

        for it in items.iter() {
            self.stats.n_candidates += 1;
            let centre = normalize(&sum4(&it.pred));

            // ALWAYS re-seed through the bridge (never the raw prediction fed straight to
            // the refiner): near the basin edge a stale prior settles on a neighbouring
            // module boundary ~0.6 px away and STILL decodes. (A strict re-seed that
            // succeeds with a DIFFERENT id than the prior's counts as a genuine cold
            // detection.)
            let before = raw.len();
            self.process_seed(
                v,
                &centre,
                it.r,
                qp.n_sub,
                FLAG_TRACK | lean_flag,
                raw,
                &mut best_code,
                &mut attempted,
                &K_TRACK_PLAN,
                true,
                &K_TRACK_REFINE,
            )?;
            if raw.len() == before {
                // The id is known here, so a marker the strict re-seed could not read is
                // verified against its predicted quad (lenient: erasures, a hidden edge,
                // 3-edge mode), prior-consistency checked.
                let pr = Prior {
                    id: it.id,
                    ray: it.pred,
                    sigma_rad: it.sigma_rad,
                };
                self.verify_one(
                    v,
                    &pr,
                    qp.n_sub,
                    FLAG_TRACK | lean_flag,
                    raw,
                    &mut best_code,
                    &mut attempted,
                )?;
            }
            if raw.len() == before {
                failed_ids.push(it.id);
            }
        }

        // Scan fallback ONCE per frame (not once per failed marker), matched back by id.
        if !failed_ids.is_empty() && raw.len() < cap {
            let saved_stats = self.stats;
            let saved_timings = self.timings;

            // Thread through the SAME `qp` resolved above (lean when n >= 4), so a marker
            // recovered by the fallback under the lean downgrade is decoded at the preset
            // its FLAG_LEAN claims.
            let scan_out = self.scan_impl(v, cap, qp)?; // resets stats / timings
            let ns = scan_out.len();

            self.stats.n_candidates += saved_stats.n_candidates;
            self.stats.n_bridge_ok += saved_stats.n_bridge_ok;
            self.stats.n_decode_ok += saved_stats.n_decode_ok;
            self.stats.n_budget_dropped += saved_stats.n_budget_dropped;
            self.stats.n_prebridge_rejected += saved_stats.n_prebridge_rejected;
            for i in 0..6 {
                self.timings[i] += saved_timings[i];
            }
            fold_best(&mut best_code, self.stats.best_reject_code);

            for &fid in failed_ids.iter() {
                if raw.len() >= cap {
                    break;
                }
                for k in 0..ns {
                    if scan_out[k].id != fid {
                        continue;
                    }
                    let mut d = scan_out[k]; // already FLAG_SCAN-tagged by scan
                    if lean {
                        d.flags |= FLAG_LEAN;
                    }
                    raw.push(d);
                    break;
                }
            }
        }

        let mut raw_v = std::mem::take(raw);
        self.dedupe_by_id(&mut raw_v);
        *raw = raw_v;

        let n_out = raw.len().min(cap);
        self.stats.n_accepted = n_out as i32;
        self.stats.n_rejected = attempted - raw.len() as i32;
        self.stats.best_reject_code = best_code;
        self.timings[5] = ms_since(t_total0);
        Ok(n_out)
    }

    /// The raw candidate list from the f=4 band + polar caps alone (no `run_small`, no
    /// bridge/decode/refine/gates): `[rx, ry, rz, angular_radius_rad]` each. Diagnostic /
    /// tuning access to stage 1.
    pub fn proposals(&mut self, v: &View) -> Result<Vec<[f32; 4]>, EqarError> {
        self.check_view(v)?;
        let mut cands = std::mem::take(&mut self.candidates);
        cands.clear();
        let r = self.proposal.run(v, &mut cands);
        let out = r.map(|_| {
            dedupe_candidates(&mut cands);
            order_for_budget(&mut cands);
            cap_candidates(&mut cands, K_MAX_CANDIDATES_PER_FRAME);
            cands
                .iter()
                .map(|c| [c.v.x as f32, c.v.y as f32, c.v.z as f32, c.r as f32])
                .collect()
        });
        self.candidates = cands;
        out
    }
}
