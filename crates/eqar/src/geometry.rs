//! Sphere geometry: pixel <-> ray maps, eigen/Kabsch helpers and the SO(3) power.

pub const K_PI: f64 = std::f64::consts::PI;
pub const K_TWO_PI: f64 = 2.0 * std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }
}

#[inline]
fn clamp1(v: f64) -> f64 {
    v.clamp(-1.0, 1.0) // NaN passes through, as in the ternary form
}

pub fn px_to_ray(px: f64, py: f64, w: i32, h: i32) -> Vec3 {
    // The reference build contracts these into fused multiply-adds; the rays it produces
    // (and the degenerate-quad decisions downstream of them) are only reproduced to the
    // last bit by doing the same on aarch64.
    #[cfg(target_arch = "aarch64")]
    let (lon, lat) = (
        ((px + 0.5) / w as f64).mul_add(K_TWO_PI, -K_PI),
        (-((py + 0.5) / h as f64)).mul_add(K_PI, 0.5 * K_PI),
    );
    #[cfg(not(target_arch = "aarch64"))]
    let (lon, lat) = (
        (px + 0.5) / w as f64 * K_TWO_PI - K_PI,
        0.5 * K_PI - (py + 0.5) / h as f64 * K_PI,
    );
    let cl = lat.cos();
    Vec3::new(cl * lon.cos(), lat.sin(), cl * lon.sin())
}

/// Returns `(px, py)`.
pub fn ray_to_px(r: &Vec3, w: i32, h: i32) -> (f64, f64) {
    let n = (r.x * r.x + r.y * r.y + r.z * r.z).sqrt();
    let yy = if n > 1e-300 { r.y / n } else { 0.0 };
    let lon = r.z.atan2(r.x);
    let lat = clamp1(yy).asin();
    let px = (lon + K_PI) / K_TWO_PI * w as f64 - 0.5;
    let py = (0.5 * K_PI - lat) / K_PI * h as f64 - 0.5;
    (px, py)
}

pub fn px_err(ax: f64, ay: f64, bx: f64, by: f64, w: i32) -> f64 {
    let wf = w as f64;
    let dx = (ax - bx + 1.5 * wf) % wf - 0.5 * wf;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

pub fn ang_err_px_eq(a: &Vec3, b: &Vec3, h: i32) -> f64 {
    let d = clamp1(a.x * b.x + a.y * b.y + a.z * b.z);
    d.acos() / (K_PI / h as f64)
}

pub fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

pub fn norm(a: &Vec3) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalize(a: &Vec3) -> Vec3 {
    let n = norm(a);
    let s = if n > 1e-300 { 1.0 / n } else { 0.0 };
    Vec3::new(a.x * s, a.y * s, a.z * s)
}

pub fn slerp(a: &Vec3, b: &Vec3, u: f64) -> Vec3 {
    let c = clamp1(dot(a, b));
    let om = c.acos();
    if om < 1e-9 {
        return *a;
    }
    let so = om.sin();
    let s0 = ((1.0 - u) * om).sin() / so;
    let s1 = (u * om).sin() / so;
    Vec3::new(
        a.x * s0 + b.x * s1,
        a.y * s0 + b.y * s1,
        a.z * s0 + b.z * s1,
    )
}

/// Scatter accumulation of `M = sum w_i * p_i p_i^T` over the unit rays `p`
/// (`w = None` means all 1). Packed as `{m00, m01, m02, m11, m12, m22}`.
pub fn accum_ppt(p: &[Vec3], w: Option<&[f64]>) -> [f64; 6] {
    let (mut a, mut b, mut c, mut d, mut e, mut f) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, pi) in p.iter().enumerate() {
        let (x, y, z) = (pi.x, pi.y, pi.z);
        let wi = w.map_or(1.0, |w| w[i]);
        a += wi * x * x;
        b += wi * x * y;
        c += wi * x * z;
        d += wi * y * y;
        e += wi * y * z;
        f += wi * z * z;
    }
    [a, b, c, d, e, f]
}

fn cr3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn n2(c: &[f64; 3]) -> f64 {
    c[0] * c[0] + c[1] * c[1] + c[2] * c[2]
}

/// Closed-form smallest eigenvector of a symmetric 3x3 (the great-circle normal).
/// `m` is packed as `{m00, m01, m02, m11, m12, m22}`.
pub fn smallest_eigvec3(m: &[f64; 6]) -> Vec3 {
    let (m00, m01, m02, m11, m12, m22) = (m[0], m[1], m[2], m[3], m[4], m[5]);
    let p1 = m01 * m01 + m02 * m02 + m12 * m12;
    let q = (m00 + m11 + m22) / 3.0;
    let e3 = if p1 <= 1e-300 {
        if m00 < m11 {
            if m00 < m22 {
                m00
            } else {
                m22
            }
        } else if m11 < m22 {
            m11
        } else {
            m22
        }
    } else {
        let (d0, d1, d2) = (m00 - q, m11 - q, m22 - q);
        let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        let ip = 1.0 / p;
        // det(B)/2 with B = (M - qI)/p
        let (b00, b11, b22) = (d0 * ip, d1 * ip, d2 * ip);
        let (b01, b02, b12) = (m01 * ip, m02 * ip, m12 * ip);
        let r = 0.5
            * (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
                + b02 * (b01 * b12 - b11 * b02));
        let r = clamp1(r);
        let phi = r.acos() / 3.0;
        q + 2.0 * p * (phi + 2.0943951023931953).cos() // + 2pi/3  -> smallest
    };
    // eigenvector: null space of (M - e3 I) = cross of its two most independent rows
    let (a00, a11, a22) = (m00 - e3, m11 - e3, m22 - e3);
    let r0 = [a00, m01, m02];
    let r1 = [m01, a11, m12];
    let r2 = [m02, m12, a22];
    let c0 = cr3(&r0, &r1);
    let c1 = cr3(&r0, &r2);
    let c2 = cr3(&r1, &r2);
    let (n0, n1, n2v) = (n2(&c0), n2(&c1), n2(&c2));
    let mut best = &c0;
    let mut bn = n0;
    if n1 > bn {
        best = &c1;
        bn = n1;
    }
    if n2v > bn {
        best = &c2;
        bn = n2v;
    }
    let s = 1.0 / (if bn > 1e-300 { bn } else { 1.0 }).sqrt();
    Vec3::new(best[0] * s, best[1] * s, best[2] * s)
}

/// Returns `(e1, e2)`.
pub fn basis_at(v: &Vec3) -> (Vec3, Vec3) {
    let mut up = Vec3::new(0.0, 1.0, 0.0);
    if dot(v, &up).abs() > 0.999 {
        up = Vec3::new(0.0, 0.0, 1.0);
    }
    let e1 = normalize(&cross(&up, v));
    let e2 = cross(v, &e1);
    (e1, e2)
}

// ---- Mat3: row-major 3x3, and the 4-ray Kabsch / SO(3) power predictor ----

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [f64; 9],
}

pub fn mat3_identity() -> Mat3 {
    Mat3 {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
}

pub fn mat3_apply(r: &Mat3, v: &Vec3) -> Vec3 {
    Vec3::new(
        r.m[0] * v.x + r.m[1] * v.y + r.m[2] * v.z,
        r.m[3] * v.x + r.m[4] * v.y + r.m[5] * v.z,
        r.m[6] * v.x + r.m[7] * v.y + r.m[8] * v.z,
    )
}

pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = Mat3 { m: [0.0; 9] };
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a.m[i * 3 + k] * b.m[k * 3 + j];
            }
            r.m[i * 3 + j] = s;
        }
    }
    r
}

pub fn mat3_transpose(a: &Mat3) -> Mat3 {
    Mat3 {
        m: [
            a.m[0], a.m[3], a.m[6], a.m[1], a.m[4], a.m[7], a.m[2], a.m[5], a.m[8],
        ],
    }
}

/// Cyclic Jacobi eigen-decomposition of a symmetric 4x4. 24 sweeps of 6 rotations
/// each; cost is irrelevant, this runs once per tracked marker per frame. On return
/// `k` holds the eigenvalues on its diagonal and `v` the eigenvectors as columns.
pub fn jacobi_eigen_sym4(k: &mut [[f64; 4]; 4], v: &mut [[f64; 4]; 4]) {
    for (i, row) in v.iter_mut().enumerate() {
        for (j, e) in row.iter_mut().enumerate() {
            *e = if i == j { 1.0 } else { 0.0 };
        }
    }
    for _sweep in 0..24 {
        for p in 0..3 {
            for q in (p + 1)..4 {
                let apq = k[p][q];
                if apq == 0.0 {
                    continue;
                }
                let (app, aqq) = (k[p][p], k[q][q]);
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                k[p][p] = app - t * apq;
                k[q][q] = aqq + t * apq;
                k[p][q] = 0.0;
                k[q][p] = 0.0;
                for kk in 0..4 {
                    if kk == p || kk == q {
                        continue;
                    }
                    let (akp, akq) = (k[kk][p], k[kk][q]);
                    k[kk][p] = c * akp - s * akq;
                    k[p][kk] = k[kk][p];
                    k[kk][q] = s * akp + c * akq;
                    k[q][kk] = k[kk][q];
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
}

/// Best-fit rotation taking `from[i]` to `to[i]` (unit vectors), det = +1.
/// Davenport q-method: build the attitude profile `C = sum to_i * from_i^T`, pack the
/// symmetric 4x4 `K` from its symmetric part `S = C + C^T`, antisymmetric vector `Z`
/// and trace sigma; the optimal quaternion (x, y, z, w) is K's eigenvector for the
/// LARGEST eigenvalue (Jacobi on the 4x4). Uses `min(from.len(), to.len())` pairs.
pub fn kabsch(from: &[Vec3], to: &[Vec3]) -> Mat3 {
    let mut c = [[0.0f64; 3]; 3];
    for (f, t) in from.iter().zip(to.iter()) {
        let (fx, fy, fz) = (f.x, f.y, f.z);
        let (tx, ty, tz) = (t.x, t.y, t.z);
        c[0][0] += tx * fx;
        c[0][1] += tx * fy;
        c[0][2] += tx * fz;
        c[1][0] += ty * fx;
        c[1][1] += ty * fy;
        c[1][2] += ty * fz;
        c[2][0] += tz * fx;
        c[2][1] += tz * fy;
        c[2][2] += tz * fz;
    }
    let sigma = c[0][0] + c[1][1] + c[2][2];
    let z = [c[1][2] - c[2][1], c[2][0] - c[0][2], c[0][1] - c[1][0]];
    let mut k = [[0.0f64; 4]; 4];
    for i in 0..3 {
        for j in 0..3 {
            k[i][j] = (c[i][j] + c[j][i]) - if i == j { sigma } else { 0.0 };
        }
    }
    for i in 0..3 {
        k[i][3] = z[i];
        k[3][i] = z[i];
    }
    k[3][3] = sigma;

    let mut v = [[0.0f64; 4]; 4];
    jacobi_eigen_sym4(&mut k, &mut v);
    let mut best = 0;
    let mut bestval = k[0][0];
    for i in 1..4 {
        if k[i][i] > bestval {
            bestval = k[i][i];
            best = i;
        }
    }
    let (mut qx, mut qy, mut qz, mut qw) = (v[0][best], v[1][best], v[2][best], v[3][best]);
    let qn2 = qx * qx + qy * qy + qz * qz + qw * qw;
    let s = if qn2 > 1e-300 { 1.0 / qn2.sqrt() } else { 0.0 };
    // The Davenport/Shuster convention's quaternion parametrizes the passive attitude
    // matrix (A with A*from = to read as a frame transform); the active "rotate a
    // vector" matrix built below is its transpose, i.e. the conjugate quaternion --
    // negate the vector part to compensate.
    qx *= -s;
    qy *= -s;
    qz *= -s;
    qw *= s;

    Mat3 {
        m: [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    }
}

/// Rotation by `t` times the angle of `r`, about the same axis (SO(3) power).
/// Converts R to axis-angle (theta = acos((tr R - 1)/2)) and rebuilds with t*theta via
/// Rodrigues. theta ~ 0 -> identity (any power of "no rotation" is still "no
/// rotation"); theta ~ pi -> the skew part of R vanishes, so the axis is instead
/// recovered from the symmetric part axis*axis^T = (R+I)/2, anchored on its largest
/// diagonal term for stability.
pub fn rot_power(r: &Mat3, t: f64) -> Mat3 {
    let tr = r.m[0] + r.m[4] + r.m[8];
    let c = clamp1((tr - 1.0) * 0.5);
    let theta = c.acos();

    let mut axis;
    if theta < 1e-9 {
        return mat3_identity();
    } else if (theta - K_PI).abs() < 1e-3 {
        let xx = (r.m[0] + 1.0) * 0.5;
        let yy = (r.m[4] + 1.0) * 0.5;
        let zz = (r.m[8] + 1.0) * 0.5;
        if xx >= yy && xx >= zz {
            let ax = (if xx > 0.0 { xx } else { 0.0 }).sqrt();
            axis = Vec3::new(
                ax,
                (r.m[1] + r.m[3]) * 0.25 / ax,
                (r.m[2] + r.m[6]) * 0.25 / ax,
            );
        } else if yy >= xx && yy >= zz {
            let ay = (if yy > 0.0 { yy } else { 0.0 }).sqrt();
            axis = Vec3::new(
                (r.m[1] + r.m[3]) * 0.25 / ay,
                ay,
                (r.m[5] + r.m[7]) * 0.25 / ay,
            );
        } else {
            let az = (if zz > 0.0 { zz } else { 0.0 }).sqrt();
            axis = Vec3::new(
                (r.m[2] + r.m[6]) * 0.25 / az,
                (r.m[5] + r.m[7]) * 0.25 / az,
                az,
            );
        }
        // The (R+I)/2 outer product only pins down each |axis component| -- its overall
        // sign is ambiguous (R(a,theta) and R(-a,theta) share the same symmetric part).
        // Disambiguate with the antisymmetric part: R21-R12 = 2 sin(theta) * axis.x (and
        // cyclic). Its MAGNITUDE is ill-conditioned this close to pi (that's exactly why
        // it isn't used to build the axis directly), but sin(theta) > 0 throughout the
        // open interval (0, pi), so its SIGN still reliably says which way `axis` must
        // point; get it wrong and rot_power(R, 1) drifts from R by O(delta) instead of
        // reproducing it to machine precision.
        let witness =
            axis.x * (r.m[7] - r.m[5]) + axis.y * (r.m[2] - r.m[6]) + axis.z * (r.m[3] - r.m[1]);
        if witness < 0.0 {
            axis = Vec3::new(-axis.x, -axis.y, -axis.z);
        }
        axis = normalize(&axis);
    } else {
        let s = theta.sin();
        axis = Vec3::new(
            (r.m[7] - r.m[5]) / (2.0 * s),
            (r.m[2] - r.m[6]) / (2.0 * s),
            (r.m[3] - r.m[1]) / (2.0 * s),
        );
    }

    let nt = t * theta;
    let cn = nt.cos();
    let sn = nt.sin();
    let k = 1.0 - cn;
    Mat3 {
        m: [
            cn + axis.x * axis.x * k,
            axis.x * axis.y * k - axis.z * sn,
            axis.x * axis.z * k + axis.y * sn,
            axis.y * axis.x * k + axis.z * sn,
            cn + axis.y * axis.y * k,
            axis.y * axis.z * k - axis.x * sn,
            axis.z * axis.x * k - axis.y * sn,
            axis.z * axis.y * k + axis.x * sn,
            cn + axis.z * axis.z * k,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! check_near {
        ($a:expr, $b:expr, $tol:expr) => {{
            let (a, b, t): (f64, f64, f64) = ($a, $b, $tol);
            assert!(
                (a - b).abs() <= t,
                "{} ~= {} ({a} vs {b}, tol {t})",
                stringify!($a),
                stringify!($b)
            );
        }};
    }

    #[test]
    fn px_ray_roundtrip() {
        let (w, h) = (3840, 1920);
        let (wf, hf) = (w as f64, h as f64);
        // corners, seam, both poles, and a dense interior sweep
        let pts = [
            (0.0, 0.0),
            (wf - 1.0, 0.0),
            (0.0, hf - 1.0),
            (wf - 1.0, hf - 1.0),
            (wf / 2.0 - 0.5, hf / 2.0 - 0.5),
            (1919.5, 0.25),
            (3839.75, 1919.25),
            (123.25, 7.5),
            (2048.5, 1000.125),
        ];
        for p in pts {
            let r = px_to_ray(p.0, p.1, w, h);
            check_near!((r.x * r.x + r.y * r.y + r.z * r.z).sqrt(), 1.0, 1e-12);
            let (x, y) = ray_to_px(&r, w, h);
            check_near!(px_err(x, y, p.0, p.1, w), 0.0, 1e-6);
        }
        for i in 0..4000i32 {
            let px = ((i * 7919) % w) as f64 + 0.5;
            let py = ((i as i64 * 104729) % (h as i64 - 1)) as f64 + 0.5;
            let r = px_to_ray(px, py, w, h);
            let (x, y) = ray_to_px(&r, w, h);
            check_near!(px_err(x, y, px, py, w), 0.0, 1e-6);
        }
    }

    #[test]
    fn px_err_is_seam_aware() {
        // 0.5 px left of the seam and 0.5 px right of it are 1 px apart, not W-1
        check_near!(px_err(3839.5, 100.0, 0.5, 100.0, 3840), 1.0, 1e-9);
        check_near!(px_err(10.0, 5.0, 14.0, 8.0, 3840), 5.0, 1e-9);
    }

    #[test]
    fn ang_err_px_eq_scale() {
        // 1 px of latitude at the equator == exactly 1 px_eq
        let (w, h) = (3840, 1920);
        let a = px_to_ray(100.0, 959.5, w, h);
        let b = px_to_ray(100.0, 960.5, w, h);
        check_near!(ang_err_px_eq(&a, &b, h), 1.0, 1e-9);
        // 1 px of longitude at lat 60 is only 0.5 px_eq
        let c = px_to_ray(100.0, 320.0, w, h);
        let d = px_to_ray(101.0, 320.0, w, h);
        check_near!(ang_err_px_eq(&c, &d, h), 0.5, 2e-3);
    }

    #[test]
    fn eigvec_recovers_planted_normal() {
        // Sample a great circle with unit normal n, then recover n from sum p p^T.
        let truths = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.3, -0.5, 0.81240384),
            Vec3::new(-0.6, 0.64, 0.4800),
        ];
        for t0 in &truths {
            let n = normalize(t0);
            let (e1, e2) = basis_at(&n);
            let mut p = [Vec3::new(0.0, 0.0, 0.0); 64];
            for (i, pi) in p.iter_mut().enumerate() {
                let a = 0.31 + i as f64 * 0.019; // a short arc, the hard case
                *pi = normalize(&Vec3::new(
                    e1.x * a.cos() + e2.x * a.sin(),
                    e1.y * a.cos() + e2.y * a.sin(),
                    e1.z * a.cos() + e2.z * a.sin(),
                ));
            }
            let m = accum_ppt(&p, None);
            let f = smallest_eigvec3(&m);
            let c = dot(&f, &n).abs();
            check_near!(c, 1.0, 1e-10);
        }
    }

    #[test]
    fn eigvec_degenerate_diagonal() {
        let m = [5.0, 0.0, 0.0, 1.0, 0.0, 9.0]; // smallest is the y axis
        let v = smallest_eigvec3(&m);
        check_near!(v.y.abs(), 1.0, 1e-12);
    }

    #[test]
    fn slerp_endpoints_and_midpoint() {
        let a = normalize(&Vec3::new(1.0, 0.0, 0.0));
        let b = normalize(&Vec3::new(0.0, 1.0, 0.0));
        check_near!(dot(&slerp(&a, &b, 0.0), &a), 1.0, 1e-12);
        check_near!(dot(&slerp(&a, &b, 1.0), &b), 1.0, 1e-12);
        let m = slerp(&a, &b, 0.5);
        check_near!(norm(&m), 1.0, 1e-12);
        check_near!(dot(&m, &a), dot(&m, &b), 1e-12);
    }

    #[test]
    fn basis_at_is_pole_safe() {
        let vs = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.99999, 0.0044),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        for v0 in &vs {
            let v = normalize(v0);
            let (e1, e2) = basis_at(&v);
            check_near!(norm(&e1), 1.0, 1e-12);
            check_near!(norm(&e2), 1.0, 1e-12);
            check_near!(dot(&e1, &e2), 0.0, 1e-12);
            check_near!(dot(&e1, &v), 0.0, 1e-12);
            check_near!(dot(&e2, &v), 0.0, 1e-12);
        }
    }

    #[test]
    fn weighted_accum_matches_unweighted_when_w_is_one() {
        let mut p = [Vec3::new(0.0, 0.0, 0.0); 7];
        let w = [1.0f64; 7];
        for (i, pi) in p.iter_mut().enumerate() {
            *pi = px_to_ray(100.0 * i as f64, 300.0 + i as f64, 3840, 1920);
        }
        let a = accum_ppt(&p, None);
        let b = accum_ppt(&p, Some(&w));
        for i in 0..6 {
            check_near!(a[i], b[i], 1e-12);
        }
    }

    fn rot_axis_angle(a: Vec3, th: f64) -> Mat3 {
        let a = normalize(&a);
        let (c, s) = (th.cos(), th.sin());
        let k = 1.0 - c;
        Mat3 {
            m: [
                c + a.x * a.x * k,
                a.x * a.y * k - a.z * s,
                a.x * a.z * k + a.y * s,
                a.y * a.x * k + a.z * s,
                c + a.y * a.y * k,
                a.y * a.z * k - a.x * s,
                a.z * a.x * k - a.y * s,
                a.z * a.y * k + a.x * s,
                c + a.z * a.z * k,
            ],
        }
    }

    #[test]
    fn kabsch_recovers_exact_rotation() {
        let angs = [0.001, 0.07, 0.7, 1.9, 3.0];
        let axes = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, 0.5, -0.8),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        for ax in &axes {
            for &th in &angs {
                let r = rot_axis_angle(*ax, th);
                let from = [
                    px_to_ray(1000.0, 700.0, 3840, 1920),
                    px_to_ray(1060.0, 700.0, 3840, 1920),
                    px_to_ray(1060.0, 760.0, 3840, 1920),
                    px_to_ray(1000.0, 760.0, 3840, 1920),
                ];
                let to: Vec<Vec3> = from.iter().map(|f| mat3_apply(&r, f)).collect();
                let e = kabsch(&from, &to);
                for i in 0..4 {
                    let p = mat3_apply(&e, &from[i]);
                    check_near!(dot(&p, &to[i]), 1.0, 1e-10);
                }
            }
        }
    }

    #[test]
    fn rot_power_halves_and_doubles() {
        let r = rot_axis_angle(Vec3::new(0.2, -0.9, 0.35), 0.8);
        let h = rot_power(&r, 0.5);
        let rr = mat3_mul(&h, &h);
        for i in 0..9 {
            check_near!(rr.m[i], r.m[i], 1e-10);
        }
        let i1 = mat3_mul(&rot_power(&r, 1.0), &mat3_transpose(&r));
        let id = mat3_identity();
        for i in 0..9 {
            check_near!(i1.m[i], id.m[i], 1e-10);
        }
    }

    #[test]
    fn rot_power_identity_is_stable() {
        let i = mat3_identity();
        let p = rot_power(&i, 2.0);
        for k in 0..9 {
            check_near!(p.m[k], i.m[k], 1e-12);
        }
    }

    #[test]
    fn rot_power_is_stable_near_pi() {
        // Regression for the theta ~= pi branch crossover: at |theta - pi| just outside
        // the old 1e-6 boundary, the skew-part axis (divide by ~2 sin theta) was
        // ill-conditioned and rot_power(R, 1.0) reproduced R to only ~9.85e-4. The
        // crossover was widened to 1e-3, where the symmetric-part axis extraction
        // (R+I)/2 stays accurate.
        let deltas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3];
        let axis = normalize(&Vec3::new(0.2, -0.9, 0.35));
        for delta in deltas {
            let r = rot_axis_angle(axis, K_PI - delta);
            let p = rot_power(&r, 1.0);
            for i in 0..9 {
                check_near!(p.m[i], r.m[i], 1e-7);
            }
            let h = rot_power(&r, 0.5);
            let rr = mat3_mul(&h, &h);
            for i in 0..9 {
                check_near!(rr.m[i], r.m[i], 1e-7);
            }
        }
    }
}
