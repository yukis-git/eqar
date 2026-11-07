//! aarch64 NEON kernels (the main module with `unsafe`; `image.rs` has two small blocks
//! of its own). Everything here is compiled
//! out on other targets and with the `no-neon` feature, where callers use the scalar
//! code in `image`.
//!
//! Safety convention: every `unsafe` block is a call to a `core::arch` intrinsic whose
//! pointer argument is derived from a slice that was just bounds-checked for the exact
//! number of bytes the intrinsic touches, so no read or write leaves its slice.
#![allow(clippy::excessive_precision)] // polynomial coefficients keep their source digits
#![allow(unused_unsafe)] // intrinsics are safe to call on newer toolchains, unsafe on 1.80

/// True when the NEON paths are compiled in.
pub const USE_NEON: bool = cfg!(all(target_arch = "aarch64", not(feature = "no-neon")));

#[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
pub use neon::*;

#[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
mod neon {
    use crate::geometry::{ray_to_px, Vec3};
    use crate::image::sample_gray_plane;
    use std::arch::aarch64::*;

    /// `a * b + c` in f64, fused on aarch64 exactly where the C++ build's
    /// `-ffp-contract=fast` fuses a lone product-plus-addend.
    #[inline(always)]
    fn fma64(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }

    /// f=4 gray box mean. 32 input px -> 8 output px per vector step, 4 rows
    /// accumulated in u16 (max sum 16*255=4080, well clear of the u16 range); a scalar
    /// tail handles OW % 8.
    pub fn down4_gray_neon(
        src: &[u8],
        w: i32,
        h: i32,
        sstride: usize,
        dst: &mut [u8],
        dstride: usize,
    ) {
        let (ow, oh) = (w as usize / 4, h as usize / 4);
        for oy in 0..oh {
            let rows: [&[u8]; 4] = [
                &src[(4 * oy) * sstride..],
                &src[(4 * oy + 1) * sstride..],
                &src[(4 * oy + 2) * sstride..],
                &src[(4 * oy + 3) * sstride..],
            ];
            let d = &mut dst[oy * dstride..];
            let mut ox = 0;
            while ox + 8 <= ow {
                let i = 4 * ox;
                // SAFETY: each load reads 16 bytes from a 16-byte slice; the store
                // writes 8 bytes into an 8-byte slice.
                unsafe {
                    let mut acc = vdupq_n_u16(0);
                    for row in &rows {
                        let a0 = vld1q_u8(row[i..i + 16].as_ptr());
                        let a1 = vld1q_u8(row[i + 16..i + 32].as_ptr());
                        acc = vaddq_u16(acc, vpaddq_u16(vpaddlq_u8(a0), vpaddlq_u8(a1)));
                    }
                    vst1_u8(d[ox..ox + 8].as_mut_ptr(), vshrn_n_u16::<4>(acc));
                }
                ox += 8;
            }
            while ox < ow {
                let i = 4 * ox;
                let mut s = 0u32;
                for row in &rows {
                    for j in 0..4 {
                        s += row[i + j] as u32;
                    }
                }
                d[ox] = (s >> 4) as u8;
                ox += 1;
            }
        }
    }

    /// f=4 fused BGR -> gray -> box mean, with a scalar tail for OW % 8.
    pub fn down4_bgr_neon(
        src: &[u8],
        w: i32,
        h: i32,
        sstride: usize,
        dst: &mut [u8],
        dstride: usize,
    ) {
        let (ow, oh) = (w as usize / 4, h as usize / 4);
        for oy in 0..oh {
            let rows: [&[u8]; 4] = [
                &src[(4 * oy) * sstride..],
                &src[(4 * oy + 1) * sstride..],
                &src[(4 * oy + 2) * sstride..],
                &src[(4 * oy + 3) * sstride..],
            ];
            let d = &mut dst[oy * dstride..];
            let mut ox = 0;
            while ox + 8 <= ow {
                let i = 4 * ox; // 32 source pixels -> 96 bytes BGR
                                // SAFETY: each vld3q_u8 reads 48 bytes from a 48-byte slice; the store
                                // writes 8 bytes into an 8-byte slice.
                unsafe {
                    let (wb, wg, wr) = (vdup_n_u8(29), vdup_n_u8(150), vdup_n_u8(77));
                    let mut acc = vdupq_n_u16(0);
                    for row in &rows {
                        let p = &row[3 * i..3 * i + 96];
                        let mut g = [vdupq_n_u8(0); 2];
                        for (h2, gh) in g.iter_mut().enumerate() {
                            let v = vld3q_u8(p[48 * h2..48 * h2 + 48].as_ptr());
                            let mut lo = vmull_u8(vget_low_u8(v.0), wb);
                            lo = vmlal_u8(lo, vget_low_u8(v.1), wg);
                            lo = vmlal_u8(lo, vget_low_u8(v.2), wr);
                            let mut hi = vmull_u8(vget_high_u8(v.0), wb);
                            hi = vmlal_u8(hi, vget_high_u8(v.1), wg);
                            hi = vmlal_u8(hi, vget_high_u8(v.2), wr);
                            *gh = vcombine_u8(vshrn_n_u16::<8>(lo), vshrn_n_u16::<8>(hi));
                        }
                        acc = vaddq_u16(acc, vpaddq_u16(vpaddlq_u8(g[0]), vpaddlq_u8(g[1])));
                    }
                    vst1_u8(d[ox..ox + 8].as_mut_ptr(), vshrn_n_u16::<4>(acc));
                }
                ox += 8;
            }
            while ox < ow {
                let i = 4 * ox;
                let mut s = 0u32;
                for row in &rows {
                    let p = &row[3 * i..];
                    for j in 0..4 {
                        let px = &p[j * 3..j * 3 + 3];
                        s += (29 * px[0] as u32 + 150 * px[1] as u32 + 77 * px[2] as u32) >> 8;
                    }
                }
                d[ox] = (s >> 4) as u8;
                ox += 1;
            }
        }
    }

    // ---- shared primitives for the ray -> px -> bilinear loops ----------------
    //
    // Accuracy: the atan is a degree-7 odd minimax polynomial with max error 3.8e-8
    // rad over [0, 1], i.e. 2.3e-5 px_eq at H = 1920 -- 200x tighter than the 8e-6 rad
    // that would move a source pixel by 0.005 px.

    #[inline]
    fn atan_unit(q: float32x4_t) -> float32x4_t {
        // SAFETY: register-only intrinsics.
        unsafe {
            let q2 = vmulq_f32(q, q);
            let mut r = vdupq_n_f32(-4.0553016449e-03);
            r = vfmaq_f32(vdupq_n_f32(2.1865522248e-02), r, q2);
            r = vfmaq_f32(vdupq_n_f32(-5.5915883720e-02), r, q2);
            r = vfmaq_f32(vdupq_n_f32(9.6424464897e-02), r, q2);
            r = vfmaq_f32(vdupq_n_f32(-1.3908722058e-01), r, q2);
            r = vfmaq_f32(vdupq_n_f32(1.9946583052e-01), r, q2);
            r = vfmaq_f32(vdupq_n_f32(-3.3329862203e-01), r, q2);
            r = vfmaq_f32(vdupq_n_f32(9.9999933591e-01), r, q2);
            vmulq_f32(r, q)
        }
    }

    #[inline]
    pub fn atan2v(y: float32x4_t, x: float32x4_t) -> float32x4_t {
        // SAFETY: register-only intrinsics.
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let ay = vabsq_f32(y);
            let ax = vabsq_f32(x);
            let mn = vminq_f32(ay, ax);
            let mx = vmaxq_f32(ay, ax);
            let mut q = vdivq_f32(mn, mx);
            q = vbslq_f32(vcgtq_f32(mx, zero), q, zero); // 0/0 -> 0
            let mut a = atan_unit(q);
            a = vbslq_f32(
                vcgtq_f32(ay, ax),
                vsubq_f32(vdupq_n_f32(std::f32::consts::FRAC_PI_2), a),
                a,
            );
            a = vbslq_f32(
                vcltq_f32(x, zero),
                vsubq_f32(vdupq_n_f32(std::f32::consts::PI), a),
                a,
            );
            vbslq_f32(vcltq_f32(y, zero), vnegq_f32(a), a)
        }
    }

    /// `(dx, dy, dz)` need NOT be normalised: longitude is scale-invariant, and the
    /// colatitude py needs is atan2(hypot(dx, dz), dy) directly -- pi/2 - lat with no
    /// asin and no reciprocal square root. `kx = W/2pi`, `ky = H/pi`.
    #[inline]
    pub fn ray_to_px4(
        dx: float32x4_t,
        dy: float32x4_t,
        dz: float32x4_t,
        kx: f32,
        ky: f32,
    ) -> (float32x4_t, float32x4_t) {
        // SAFETY: register-only intrinsics.
        unsafe {
            let px = vfmaq_n_f32(
                vdupq_n_f32(-0.5),
                vaddq_f32(atan2v(dz, dx), vdupq_n_f32(std::f32::consts::PI)),
                kx,
            );
            let rxz = vsqrtq_f32(vfmaq_f32(vmulq_f32(dx, dx), dz, dz));
            let py = vfmaq_n_f32(vdupq_n_f32(-0.5), atan2v(rxz, dy), ky);
            (px, py)
        }
    }

    /// Four wrapped bilinear samples of a gray plane, exactly what
    /// `sample_gray_plane` computes one at a time.
    ///
    /// The four taps of a lane are the two horizontally adjacent bytes of two adjacent
    /// rows, so the fast path reads them as two 16-bit loads per lane inserted into a
    /// NEON lane. A lane takes the general scalar sampler when it needs the pole
    /// reflection (py outside [0, H-1)) or the longitude wrap (x0 == W-1, where the
    /// pair is not contiguous). py == H-1 exactly is deliberately routed to the scalar
    /// path too: the vector form would read row H -- one row past the plane -- for a
    /// tap whose interpolation weight is zero.
    pub fn sample_gray4(
        data: &[u8],
        w: i32,
        h: i32,
        stride: usize,
        px: float32x4_t,
        py: float32x4_t,
    ) -> float32x4_t {
        // SAFETY: register-only intrinsics, plus loads/stores of 4 f32 / 4 i32 to and
        // from 4-element local arrays.
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let vw = vdupq_n_f32(w as f32);
            let vhm1 = vdupq_n_f32((h - 1) as f32);
            let iwm1 = vdupq_n_s32(w - 1);

            let pxw = vbslq_f32(vcltq_f32(px, zero), vaddq_f32(px, vw), px);
            let fx = vrndmq_f32(pxw);
            let fy = vrndmq_f32(py);
            let ax = vsubq_f32(pxw, fx);
            let ay = vsubq_f32(py, fy);
            let x0 = vcvtq_s32_f32(fx);
            let y0 = vcvtq_s32_f32(fy);

            let slow = vorrq_u32(
                vorrq_u32(vcltq_f32(py, zero), vcgeq_f32(py, vhm1)),
                vcgeq_s32(x0, iwm1),
            );
            if vmaxvq_u32(slow) != 0 {
                let (mut pxs, mut pys, mut out) = ([0f32; 4], [0f32; 4], [0f32; 4]);
                vst1q_f32(pxs.as_mut_ptr(), px);
                vst1q_f32(pys.as_mut_ptr(), py);
                for k in 0..4 {
                    out[k] = sample_gray_plane(data, w, h, stride, pxs[k], pys[k]);
                }
                return vld1q_f32(out.as_ptr());
            }

            let (mut xi, mut yi) = ([0i32; 4], [0i32; 4]);
            vst1q_s32(xi.as_mut_ptr(), x0);
            vst1q_s32(yi.as_mut_ptr(), y0);
            // Every lane is in range here (py in [0, H-1), x0 < W-1), so each lane's two
            // rows of two taps end at most at `o + stride + 2`. Checking the farthest
            // lane once lets the loads below be plain subslice reads.
            let o = [
                yi[0] as usize * stride + xi[0] as usize,
                yi[1] as usize * stride + xi[1] as usize,
                yi[2] as usize * stride + xi[2] as usize,
                yi[3] as usize * stride + xi[3] as usize,
            ];
            let end = o[0].max(o[1]).max(o[2]).max(o[3]) + stride + 2;
            if end > data.len() {
                // Not reachable for a valid view; keep the checked scalar path as the
                // answer rather than reading out of bounds.
                let (mut pxs, mut pys, mut out) = ([0f32; 4], [0f32; 4], [0f32; 4]);
                vst1q_f32(pxs.as_mut_ptr(), px);
                vst1q_f32(pys.as_mut_ptr(), py);
                for k in 0..4 {
                    out[k] = sample_gray_plane(data, w, h, stride, pxs[k], pys[k]);
                }
                return vld1q_f32(out.as_ptr());
            }
            let d = &data[..end];
            let pair = |off: usize| -> u16 {
                let b = &d[off..off + 2];
                u16::from_ne_bytes([b[0], b[1]])
            };
            let mut r0 = vdup_n_u16(0);
            let mut r1 = vdup_n_u16(0);
            r0 = vset_lane_u16::<0>(pair(o[0]), r0);
            r1 = vset_lane_u16::<0>(pair(o[0] + stride), r1);
            r0 = vset_lane_u16::<1>(pair(o[1]), r0);
            r1 = vset_lane_u16::<1>(pair(o[1] + stride), r1);
            r0 = vset_lane_u16::<2>(pair(o[2]), r0);
            r1 = vset_lane_u16::<2>(pair(o[2] + stride), r1);
            r0 = vset_lane_u16::<3>(pair(o[3]), r0);
            r1 = vset_lane_u16::<3>(pair(o[3] + stride), r1);

            let lo = vdup_n_u16(0x00FF);
            let v00 = vcvtq_f32_u32(vmovl_u16(vand_u16(r0, lo)));
            let v01 = vcvtq_f32_u32(vmovl_u16(vshr_n_u16::<8>(r0)));
            let v10 = vcvtq_f32_u32(vmovl_u16(vand_u16(r1, lo)));
            let v11 = vcvtq_f32_u32(vmovl_u16(vshr_n_u16::<8>(r1)));
            let a = vfmaq_f32(v00, vsubq_f32(v01, v00), ax);
            let b = vfmaq_f32(v10, vsubq_f32(v11, v10), ax);
            vfmaq_f32(a, vsubq_f32(b, a), ay)
        }
    }

    /// Tile-row sums for the proposal stage: `row` holds `4 * srow.len()` source bytes;
    /// for every full group of 4 tiles (16 bytes) adds each tile's 4 pixels into `srow`.
    /// Returns the number of tiles handled (a multiple of 4); the caller finishes the
    /// rest. Exact integer arithmetic, so identical to the scalar sum.
    pub fn tile_sums_row(row: &[u8], srow: &mut [u32]) -> usize {
        let mut done = 0;
        for (src, dst) in row.chunks_exact(16).zip(srow.chunks_exact_mut(4)) {
            // SAFETY: `src` is exactly 16 bytes and `dst` exactly 4 u32s, which is what
            // the load / load / store below touch.
            unsafe {
                let p2 = vpaddlq_u16(vpaddlq_u8(vld1q_u8(src.as_ptr())));
                let cur = vld1q_u32(dst.as_ptr());
                vst1q_u32(dst.as_mut_ptr(), vaddq_u32(cur, p2));
            }
            done += 4;
        }
        done
    }

    /// `dst[i] = 255 if src[i] < thr[i] else 0` for the leading multiple of 16 bytes of
    /// the three slices; returns how many bytes were handled.
    pub fn cmp_lt_row(src: &[u8], thr: &[u8], dst: &mut [u8]) -> usize {
        let mut done = 0;
        for ((s, t), d) in src
            .chunks_exact(16)
            .zip(thr.chunks_exact(16))
            .zip(dst.chunks_exact_mut(16))
        {
            // SAFETY: all three chunks are exactly 16 bytes.
            unsafe {
                vst1q_u8(
                    d.as_mut_ptr(),
                    vcltq_u8(vld1q_u8(s.as_ptr()), vld1q_u8(t.as_ptr())),
                );
            }
            done += 16;
        }
        done
    }

    /// 16 bytes -> 64-bit mask with nibble `i` equal to 0xF when `s[i] != 0` (the ARM
    /// stand-in for a movemask).
    pub fn nibble_mask16(s: &[u8; 16]) -> u64 {
        // SAFETY: `s` is exactly 16 bytes.
        unsafe {
            let v = vld1q_u8(s.as_ptr());
            let t = vtstq_u8(v, v);
            let nib = vshrn_n_u16::<4>(vreinterpretq_u16_u8(t));
            vget_lane_u64::<0>(vreinterpret_u64_u8(nib))
        }
    }

    /// Gray gnomonic patch render (the bridge's hottest loop), 4 pixels per step: for
    /// pixel (x, y) the ray is `e1*gx + e2*gy + e3` with `g = u[i] * tan_t`; it is NOT
    /// normalised (longitude is scale-invariant and `ray_to_px4` takes the colatitude
    /// from atan2 directly), sampled with the wrapped bilinear gather and rounded to u8.
    /// `u` holds the `s` unit-ramp values, `img` the `s * s` output. The `s % 4` tail
    /// pixels of each row use the scalar path.
    pub fn patch_render_gray(
        data: &[u8],
        w: i32,
        h: i32,
        stride: usize,
        u: &[f32],
        s: usize,
        tan_t: f64,
        e1: &Vec3,
        e2: &Vec3,
        e3: &Vec3,
        img: &mut [u8],
    ) {
        let tan_f = tan_t as f32;
        let kx = w as f32 / std::f32::consts::TAU;
        let ky = h as f32 / std::f32::consts::PI;
        // The only pointer handed to an intrinsic below is `u[x..x + 4].as_ptr()` (a
        // bounds-checked slice of exactly 4 f32); every store goes through a lane
        // extracted to a register and written with safe slice code.
        for y in 0..s {
            let gy = (u[y] as f64 * tan_t) as f32;
            let gyd = gy as f64;
            let (bx, by, bz) = (
                fma64(e2.x, gyd, e3.x) as f32,
                fma64(e2.y, gyd, e3.y) as f32,
                fma64(e2.z, gyd, e3.z) as f32,
            );
            let row = &mut img[y * s..(y + 1) * s];
            let mut x = 0usize;
            // SAFETY: register-only intrinsics plus the 4-float load described above.
            unsafe {
                let (vbx, vby, vbz) = (vdupq_n_f32(bx), vdupq_n_f32(by), vdupq_n_f32(bz));
                let (vax, vay, vaz) = (
                    vdupq_n_f32(e1.x as f32),
                    vdupq_n_f32(e1.y as f32),
                    vdupq_n_f32(e1.z as f32),
                );
                let vhalf = vdupq_n_f32(0.5);
                while x + 4 <= s {
                    let gx = vmulq_n_f32(vld1q_f32(u[x..x + 4].as_ptr()), tan_f);
                    let dx = vfmaq_f32(vbx, vax, gx);
                    let dy = vfmaq_f32(vby, vay, gx);
                    let dz = vfmaq_f32(vbz, vaz, gx);
                    let (px, py) = ray_to_px4(dx, dy, dz, kx, ky);
                    let out = vaddq_f32(sample_gray4(data, w, h, stride, px, py), vhalf);
                    let h16 = vmovn_u32(vcvtq_u32_f32(out));
                    let b8 = vmovn_u16(vcombine_u16(h16, h16));
                    let lane = vget_lane_u32::<0>(vreinterpret_u32_u8(b8));
                    row[x..x + 4].copy_from_slice(&lane.to_le_bytes());
                    x += 4;
                }
            }
            while x < s {
                let gxs = u[x] as f64 * tan_t;
                let d = Vec3::new(
                    e1.x * gxs + e2.x * gyd + e3.x,
                    e1.y * gxs + e2.y * gyd + e3.y,
                    e1.z * gxs + e2.z * gyd + e3.z,
                );
                let (pxd, pyd) = ray_to_px(&d, w, h);
                row[x] =
                    (sample_gray_plane(data, w, h, stride, pxd as f32, pyd as f32) + 0.5) as u8;
                x += 1;
            }
        }
    }

    /// Sum of four gray samples of a marker-plane row (the border probe's inner step):
    /// ray `d = B + A * X` with `X = cx + gx[k]`, sampled like `patch_render_gray`.
    pub fn cell_sum4(
        data: &[u8],
        w: i32,
        h: i32,
        stride: usize,
        a: &[f32; 3],
        b: &[f32; 3],
        cx: f32,
        gx: &[f32],
        kx: f32,
        ky: f32,
    ) -> f32 {
        // SAFETY: the one pointer load reads `gx[..4]` (slice-checked below); the rest
        // is register-only.
        unsafe {
            let xv = vaddq_f32(vdupq_n_f32(cx), vld1q_f32(gx[..4].as_ptr()));
            let (px, py) = ray_to_px4(
                vfmaq_f32(vdupq_n_f32(b[0]), vdupq_n_f32(a[0]), xv),
                vfmaq_f32(vdupq_n_f32(b[1]), vdupq_n_f32(a[1]), xv),
                vfmaq_f32(vdupq_n_f32(b[2]), vdupq_n_f32(a[2]), xv),
                kx,
                ky,
            );
            vaddvq_f32(sample_gray4(data, w, h, stride, px, py))
        }
    }

    /// `out[j]` = gray at the ray `p * cs[j] + pp * sn[j]`, for `j < out.len()`; four taps per
    /// step. `|q|` is exactly 1 here (`p` and `pp` are orthonormal unit vectors), so the
    /// longitude is atan2(qz, qx) and the colatitude is atan2(hypot(qx, qz), qy) directly,
    /// with no asin and no normalisation (the refiner's edge-profile sampler). The tail
    /// (`len % 4`) uses the scalar double-precision path. `cs` and `sn` hold at least
    /// `out.len()` values.
    pub fn sample_arc_gray(
        data: &[u8],
        w: i32,
        h: i32,
        stride: usize,
        p: &Vec3,
        pp: &Vec3,
        cs: &[f64],
        sn: &[f64],
        out: &mut [f32],
    ) {
        let m = out.len().min(cs.len()).min(sn.len());
        let kx = w as f32 / 6.283_185_5_f32;
        let ky = h as f32 / std::f32::consts::PI;
        let mut j = 0usize;
        // SAFETY: register-only intrinsics, plus loads of two f64 from the two-element
        // subslices `cs[j..j + 2]` / `sn[j..j + 2]` (and `j + 2..j + 4`) and a store of four
        // f32 into `out[j..j + 4]`; every slice is bounds-checked by the indexing and
        // `j + 4 <= m`.
        unsafe {
            let (px_, py_, pz_) = (
                vdupq_n_f32(p.x as f32),
                vdupq_n_f32(p.y as f32),
                vdupq_n_f32(p.z as f32),
            );
            let (qx_, qy_, qz_) = (
                vdupq_n_f32(pp.x as f32),
                vdupq_n_f32(pp.y as f32),
                vdupq_n_f32(pp.z as f32),
            );
            while j + 4 <= m {
                let c = vcombine_f32(
                    vcvt_f32_f64(vld1q_f64(cs[j..j + 2].as_ptr())),
                    vcvt_f32_f64(vld1q_f64(cs[j + 2..j + 4].as_ptr())),
                );
                let s = vcombine_f32(
                    vcvt_f32_f64(vld1q_f64(sn[j..j + 2].as_ptr())),
                    vcvt_f32_f64(vld1q_f64(sn[j + 2..j + 4].as_ptr())),
                );
                let qx = vfmaq_f32(vmulq_f32(px_, c), qx_, s);
                let qy = vfmaq_f32(vmulq_f32(py_, c), qy_, s);
                let qz = vfmaq_f32(vmulq_f32(pz_, c), qz_, s);
                let (fx, fy) = ray_to_px4(qx, qy, qz, kx, ky);
                vst1q_f32(
                    out[j..j + 4].as_mut_ptr(),
                    sample_gray4(data, w, h, stride, fx, fy),
                );
                j += 4;
            }
        }
        while j < m {
            let q = Vec3::new(
                p.x * cs[j] + pp.x * sn[j],
                p.y * cs[j] + pp.y * sn[j],
                p.z * cs[j] + pp.z * sn[j],
            );
            let (qpx, qpy) = ray_to_px(&q, w, h);
            out[j] = sample_gray_plane(data, w, h, stride, qpx as f32, qpy as f32);
            j += 1;
        }
    }

    /// `dst[i] = 255 * src[i] / den[i]` for the leading multiple of 8 elements; returns
    /// how many were handled. Same operation order as the scalar loop (multiply, then
    /// divide), so the results are bit-identical.
    pub fn flat_div_row(src: &[u8], den: &[f32], dst: &mut [f32]) -> usize {
        let mut done = 0;
        for ((s, d), o) in src
            .chunks_exact(8)
            .zip(den.chunks_exact(8))
            .zip(dst.chunks_exact_mut(8))
        {
            // SAFETY: each chunk is exactly 8 elements; loads read 8 bytes / 2x4 f32 and
            // stores write 2x4 f32.
            unsafe {
                let s16 = vmovl_u8(vld1_u8(s.as_ptr()));
                let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(s16)));
                let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(s16)));
                let v255 = vdupq_n_f32(255.0);
                vst1q_f32(
                    o.as_mut_ptr(),
                    vdivq_f32(vmulq_f32(lo, v255), vld1q_f32(d.as_ptr())),
                );
                vst1q_f32(
                    o.as_mut_ptr().add(4),
                    vdivq_f32(vmulq_f32(hi, v255), vld1q_f32(d.as_ptr().add(4))),
                );
            }
            done += 8;
        }
        done
    }

    /// `dst[i] = 255 if flat[i] < t else 0` for the leading multiple of 16 elements;
    /// returns how many were handled.
    pub fn binarise_lt_f32(flat: &[f32], t: f32, dst: &mut [u8]) -> usize {
        let mut done = 0;
        for (f, d) in flat.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
            // SAFETY: `f` is 16 f32 (4 loads of 4) and `d` 16 bytes (one store).
            unsafe {
                let vt = vdupq_n_f32(t);
                let m0 = vcltq_f32(vld1q_f32(f.as_ptr()), vt);
                let m1 = vcltq_f32(vld1q_f32(f.as_ptr().add(4)), vt);
                let m2 = vcltq_f32(vld1q_f32(f.as_ptr().add(8)), vt);
                let m3 = vcltq_f32(vld1q_f32(f.as_ptr().add(12)), vt);
                let a = vcombine_u16(vmovn_u32(m0), vmovn_u32(m1));
                let b = vcombine_u16(vmovn_u32(m2), vmovn_u32(m3));
                vst1q_u8(d.as_mut_ptr(), vcombine_u8(vmovn_u16(a), vmovn_u16(b)));
            }
            done += 16;
        }
        done
    }

    /// `dst[i] = 255 if img[i] <= t else 0` for the leading multiple of 16 bytes;
    /// returns how many were handled.
    pub fn binarise_le_u8(img: &[u8], t: u8, dst: &mut [u8]) -> usize {
        let mut done = 0;
        for (s, d) in img.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
            // SAFETY: both chunks are exactly 16 bytes.
            unsafe {
                vst1q_u8(
                    d.as_mut_ptr(),
                    vcleq_u8(vld1q_u8(s.as_ptr()), vdupq_n_u8(t)),
                );
            }
            done += 16;
        }
        done
    }
}
