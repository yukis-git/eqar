//! 4-ray Kabsch constant-angular-velocity predictor.
//!
//! Pixel-space constant-velocity prediction survives the seam (the wrap error is always
//! an integer multiple of the frame width, which cancels out of a pixel delta) but breaks
//! down at the poles, by up to 446 px: a fixed pixel offset near a pole is a wildly
//! different angular displacement depending on latitude, and the pole itself is a
//! coordinate singularity of the (lon, lat) pixel grid. Predicting directly in SO(3)
//! sidesteps both: the rotation that best explains corners(t-1) -> corners(t) (Kabsch) is
//! applied once more to corners(t), exactly realising a constant angular velocity, with
//! no notion of "pixel" anywhere in the model -- the seam and the poles are not special
//! cases.

use crate::geometry::{dot, kabsch, mat3_apply, normalize, rot_power, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct TrackState {
    /// corner rays at t-1
    pub prev: [Vec3; 4],
    /// corner rays at t
    pub last: [Vec3; 4],
    pub has_prev: bool,
    pub has_last: bool,
    pub id: i32,
    pub side_px: f32,
    /// Frame clock: frames between `prev` and `last` (the gap the angular velocity is
    /// measured over). The default is the one-frame-per-step model.
    pub dt: f64,
    /// Clock value of `last`.
    pub t_last: f64,
}

impl Default for TrackState {
    fn default() -> Self {
        TrackState {
            prev: [Vec3::default(); 4],
            last: [Vec3::default(); 4],
            has_prev: false,
            has_last: false,
            id: -1,
            side_px: 0.0,
            dt: 1.0,
            t_last: 0.0,
        }
    }
}

fn centroid_ray(c: &[Vec3; 4]) -> Vec3 {
    normalize(&Vec3::new(
        c[0].x + c[1].x + c[2].x + c[3].x,
        c[0].y + c[1].y + c[2].y + c[3].y,
        c[0].z + c[1].z + c[2].z + c[3].z,
    ))
}

fn angle_between(a: &Vec3, b: &Vec3) -> f64 {
    let mut d = dot(a, b);
    d = d.clamp(-1.0, 1.0); // NaN passes through
    d.acos()
}

fn angular_radius(centre: &Vec3, c: &[Vec3; 4]) -> f64 {
    let mut r = 0.0;
    for ci in c {
        let a = angle_between(centre, ci);
        if a > r {
            r = a;
        }
    }
    r
}

/// `R = kabsch(prev -> last)`; predicted = `rot_power(R, step / dt)` applied to `last`,
/// i.e. constant angular velocity. Also extrapolates the angular size. `None` unless
/// `has_last` is set. `step` = frames from `last` to the predicted pose; the angular
/// velocity (the rotation prev -> last over `s.dt` frames) is scaled by `step / s.dt`.
/// Returns the predicted corners and the predicted angular radius.
pub fn predict_next(s: &TrackState, step: f64) -> Option<([Vec3; 4], f64)> {
    if !s.has_last {
        return None;
    }

    if !s.has_prev {
        // Only one observed frame: degrade gracefully to the last pose.
        return Some((s.last, angular_radius(&centroid_ray(&s.last), &s.last)));
    }

    // Constant angular velocity: the rotation that took prev -> last, applied once more
    // to last.
    let r = kabsch(&s.prev, &s.last);
    let ratio = step / if s.dt > 1e-9 { s.dt } else { 1.0 };
    let r1 = rot_power(&r, ratio);
    let mut out = [Vec3::default(); 4];
    for i in 0..4 {
        out[i] = mat3_apply(&r1, &s.last[i]);
    }

    let radius_prev = angular_radius(&centroid_ray(&s.prev), &s.prev);
    let radius_last = angular_radius(&centroid_ray(&s.last), &s.last);
    let mut scale = 1.0;
    if radius_prev > 1e-12 {
        scale = radius_last / radius_prev;
        scale = scale.clamp(0.5, 2.0);
        if ratio != 1.0 {
            scale = scale.powf(ratio);
        }
    }
    let centre_pred = centroid_ray(&out);
    Some((out, angular_radius(&centre_pred, &out) * scale))
}

/// Advance the track by one accepted observation: last -> prev, the new
/// corners/id/side_px become `last`. `dt` = frames since the previous observation, `t` =
/// the clock of the new one (only stored; used by the detector to compute the step to a
/// later frame).
pub fn track_update(
    s: &mut TrackState,
    accepted: &[Vec3; 4],
    id: i32,
    side_px: f32,
    dt: f64,
    t: f64,
) {
    if s.has_last {
        s.prev = s.last;
        s.has_prev = true;
    } else {
        s.has_prev = false;
    }
    s.last = *accepted;
    s.has_last = true;
    s.id = id;
    s.side_px = side_px;
    s.dt = dt;
    s.t_last = t;
}
