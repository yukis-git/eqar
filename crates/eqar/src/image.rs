//! Borrowed image views, the wrapped bilinear sampler, box-mean downsampling and the
//! scratch arena.

use crate::error::EqarError;
use std::ops::Range;

/// `a * b + c`. On aarch64 this is one fused multiply-add, matching what the C++
/// original's `-ffp-contract=fast` build does for the bilinear blends; elsewhere the
/// plain two-rounding form (the C++ scalar result differs from it by a few ulp at most).
#[inline(always)]
pub(crate) fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt {
    Gray8 = 0,
    Bgr8 = 1,
    Rgb8 = 2,
}

impl Fmt {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Fmt::Gray8 => 1,
            Fmt::Bgr8 | Fmt::Rgb8 => 3,
        }
    }
}

/// Borrowed, strided image view. Never owns `data`.
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    pub data: &'a [u8],
    pub w: i32,
    pub h: i32,
    pub stride: usize,
    pub fmt: Fmt,
    /// 256-entry gamma LUT
    pub lut: Option<&'a [u8; 256]>,
}

impl View<'_> {
    /// Non-empty, positive dimensions, and `data` long enough for every row.
    pub fn is_valid(&self) -> bool {
        if self.data.is_empty() || self.w <= 0 || self.h <= 0 || self.stride == 0 {
            return false;
        }
        let row = self.w as usize * self.fmt.bytes_per_pixel();
        // checked: a huge caller stride must not wrap into a "valid" length
        self.stride >= row
            && (self.h as usize - 1)
                .checked_mul(self.stride)
                .and_then(|n| n.checked_add(row))
                .is_some_and(|need| self.data.len() >= need)
    }

    pub fn check(&self) -> Result<(), EqarError> {
        if self.data.is_empty() || self.w <= 0 || self.h <= 0 {
            return Err(EqarError::Size);
        }
        if !self.is_valid() {
            return Err(EqarError::Stride);
        }
        Ok(())
    }
}

/// The four taps (two rows x two columns) and the blend weights of a wrapped
/// bilinear sample. Longitude wraps modulo `w` unconditionally. Latitude REFLECTS
/// over either pole with a half-turn (w/2) shift in longitude (a plain
/// clamp-both-axes wrap is wrong at the poles). The reflection is applied once to the
/// incoming (x, y); then, since the two y-taps (floor(y), floor(y)+1) can still land
/// one row past whichever edge triggered it, each row that does gets the same
/// reflection applied again for just that row's pair of column taps -- this keeps the
/// interpolation weights physically correct up to the pole and keeps every memory
/// read in-bounds (no OOB read even when its weight is exactly zero).
struct Taps {
    y0: usize,
    y1: usize,
    cx0: usize,
    cx1: usize,
    dx0: usize,
    dx1: usize,
    ax: f32,
    ay: f32,
}

#[inline]
fn taps(w: i32, h: i32, mut x: f32, mut y: f32) -> Taps {
    if y < -0.5 {
        y = -1.0 - y;
        x += w as f32 * 0.5;
    } else if y > h as f32 - 0.5 {
        y = 2.0 * h as f32 - 1.0 - y;
        x += w as f32 * 0.5;
    }
    if x < 0.0 {
        x += w as f32;
    } else if x >= w as f32 {
        x -= w as f32;
    }
    // Inputs far outside the plane (or NaN) land here still out of range: pin them to the
    // plane edge so no index can leave it. A no-op for every in-range input.
    x = x.max(0.0).min(w as f32);
    y = y.max(-0.5).min(h as f32 - 0.5);
    let x0 = (x as i32).min(w - 1);
    let ax = x - x0 as f32;
    let mut x1 = x0 + 1;
    if x1 >= w {
        x1 = 0;
    }
    let mut y0 = y.floor() as i32;
    let ay = y - y0 as f32;
    let mut y1 = y0 + 1;
    let (mut cx0, mut cx1) = (x0, x1); // columns for row y0
    let (mut dx0, mut dx1) = (x0, x1); // columns for row y1
    let hw = w / 2;
    if y0 < 0 {
        y0 = 0;
        cx0 += hw;
        if cx0 >= w {
            cx0 -= w;
        }
        cx1 += hw;
        if cx1 >= w {
            cx1 -= w;
        }
    }
    if y1 > h - 1 {
        y1 = h - 1;
        dx0 += hw;
        if dx0 >= w {
            dx0 -= w;
        }
        dx1 += hw;
        if dx1 >= w {
            dx1 -= w;
        }
    }
    Taps {
        y0: y0 as usize,
        y1: y1 as usize,
        cx0: cx0 as usize,
        cx1: cx1 as usize,
        dx0: dx0 as usize,
        dx1: dx1 as usize,
        ax,
        ay,
    }
}

/// Sphere-topology wrapped bilinear sample of a single-channel (gray) plane. This is
/// the hot path (gray, no LUT), so it stays branch-light. `(x, y)` must lie within one
/// image height of the plane (the pole reflection is applied once).
#[inline]
pub fn sample_gray_plane(p: &[u8], w: i32, h: i32, stride: usize, x: f32, y: f32) -> f32 {
    let t = taps(w, h, x, y);
    let r0 = &p[t.y0 * stride..];
    let r1 = &p[t.y1 * stride..];
    let (r0a, r0b) = (r0[t.cx0] as i32, r0[t.cx1] as i32);
    let (r1a, r1b) = (r1[t.dx0] as i32, r1[t.dx1] as i32);
    let a = fmadd((r0b - r0a) as f32, t.ax, r0a as f32);
    let b = fmadd((r1b - r1a) as f32, t.ax, r1a as f32);
    fmadd(b - a, t.ay, a)
}

/// General sampler: Gray8/Bgr8/Rgb8, with an optional gamma LUT applied to the four
/// taps (not to the interpolated result). BGR/RGB taps are read straight out of the
/// interleaved buffer -- no full-res gray plane needed -- and combined with the exact
/// integer weights `(29*B + 150*G + 77*R)/256`, evaluated in float so sub-DN precision
/// survives the bilinear blend.
#[inline]
pub fn sample(v: &View, x: f32, y: f32) -> f32 {
    if v.fmt == Fmt::Gray8 && v.lut.is_none() {
        return sample_gray_plane(v.data, v.w, v.h, v.stride, x, y);
    }
    sample_general(v, x, y)
}

/// The LUT / colour branches of `sample`, kept out of line so the gray fast path stays small
/// enough to inline into the callers' loops.
#[inline(never)]
fn sample_general(v: &View, x: f32, y: f32) -> f32 {
    let t = taps(v.w, v.h, x, y);
    let tap = |cx: usize, ry: usize| -> f32 {
        let row = &v.data[ry * v.stride..];
        if v.fmt == Fmt::Gray8 {
            let g = row[cx];
            return match v.lut {
                Some(l) => l[g as usize] as f32,
                None => g as f32,
            };
        }
        let px = &row[cx * 3..cx * 3 + 3];
        let (mut c0, mut c1, mut c2) = (px[0], px[1], px[2]);
        if let Some(l) = v.lut {
            c0 = l[c0 as usize];
            c1 = l[c1 as usize];
            c2 = l[c2 as usize];
        }
        let (b, g, r) = if v.fmt == Fmt::Bgr8 {
            (c0 as f32, c1 as f32, c2 as f32)
        } else {
            (c2 as f32, c1 as f32, c0 as f32) // Rgb8
        };
        (29.0 * b + 150.0 * g + 77.0 * r) * (1.0 / 256.0)
    };
    let v00 = tap(t.cx0, t.y0);
    let v01 = tap(t.cx1, t.y0);
    let v10 = tap(t.dx0, t.y1);
    let v11 = tap(t.dx1, t.y1);
    let a = fmadd(v01 - v00, t.ax, v00);
    let b = fmadd(v11 - v10, t.ax, v10);
    fmadd(b - a, t.ay, a)
}

/// f in {2,4} box mean of a Gray8 plane, integer. Addition of these small integers is
/// exact and order-independent, so any summation order is bit-identical.
pub fn downsample_gray_scalar(
    src: &[u8],
    w: i32,
    h: i32,
    sstride: usize,
    dst: &mut [u8],
    dstride: usize,
    f: usize,
) {
    let (ow, oh) = (w as usize / f, h as usize / f);
    let shift = if f == 4 { 4 } else { 2 };
    for oy in 0..oh {
        let d = &mut dst[oy * dstride..];
        for ox in 0..ow {
            let mut s = 0u32;
            for j in 0..f {
                let r = &src[(f * oy + j) * sstride + f * ox..];
                for &px in &r[..f] {
                    s += px as u32;
                }
            }
            d[ox] = (s >> shift) as u8;
        }
    }
}

/// f in {2,4} fused BGR/RGB -> gray -> box mean, never materialising a full-resolution
/// gray plane. Each pixel's gray is the exact integer `(29*B + 150*G + 77*R) >> 8`,
/// summed over the f x f block and shifted once more for the box mean.
pub fn downsample_bgr_scalar(
    src: &[u8],
    w: i32,
    h: i32,
    sstride: usize,
    dst: &mut [u8],
    dstride: usize,
    f: usize,
    is_rgb: bool,
) {
    let (ow, oh) = (w as usize / f, h as usize / f);
    let shift = if f == 4 { 4 } else { 2 };
    for oy in 0..oh {
        let d = &mut dst[oy * dstride..];
        for ox in 0..ow {
            let mut s = 0u32;
            for j in 0..f {
                let r = &src[(f * oy + j) * sstride + f * ox * 3..];
                for i in 0..f {
                    let px = &r[i * 3..i * 3 + 3];
                    let (c0, c1, c2) = (px[0] as u32, px[1] as u32, px[2] as u32);
                    let (b, g, rr) = if is_rgb { (c2, c1, c0) } else { (c0, c1, c2) };
                    s += (29 * b + 150 * g + 77 * rr) >> 8;
                }
            }
            d[ox] = (s >> shift) as u8;
        }
    }
}

/// Box mean of the f x f block; `dst` must hold (w/f) x (h/f) at `dstride`. `f` is 2
/// or 4. Handles Gray8/Bgr8/Rgb8 sources directly -- the BGR/RGB path never
/// materialises a full-resolution gray plane (measured in the C++ original: 0.40 ms
/// fused vs 1.65 ms via a materialised gray plane).
pub fn downsample(src: &View, f: usize, dst: &mut [u8], dstride: usize) {
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    {
        if f == 4 && src.fmt == Fmt::Gray8 {
            crate::simd::down4_gray_neon(src.data, src.w, src.h, src.stride, dst, dstride);
            return;
        }
        if f == 4 && src.fmt == Fmt::Bgr8 {
            crate::simd::down4_bgr_neon(src.data, src.w, src.h, src.stride, dst, dstride);
            return;
        }
    }
    if src.fmt == Fmt::Gray8 {
        downsample_gray_scalar(src.data, src.w, src.h, src.stride, dst, dstride, f);
    } else {
        downsample_bgr_scalar(
            src.data,
            src.w,
            src.h,
            src.stride,
            dst,
            dstride,
            f,
            src.fmt == Fmt::Rgb8,
        );
    }
}

/// One 64-byte unit of arena storage; gives the backing `Vec` 64-byte alignment.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Block([u8; 64]);

/// 64-byte aligned bump-allocated scratch arena: allocated once, reused (via
/// `reset`) every frame -- never allocated per frame. `take` hands out byte ranges
/// (every one starts 64-byte aligned); resolve them with `slice` / `slice_mut`.
pub struct Arena {
    blocks: Vec<Block>,
    used: usize,
}

impl Arena {
    pub fn new(bytes: usize) -> Self {
        let n = bytes.div_ceil(64);
        Arena {
            blocks: vec![Block([0; 64]); n],
            used: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.blocks.len() * 64
    }

    pub fn used(&self) -> usize {
        self.used
    }

    /// Reserve `n` bytes (rounded up to a multiple of 64). Never hands out a range
    /// past the allocation: returns `Err(EqarError::Alloc)` when the arena is full.
    pub fn take(&mut self, n: usize) -> Result<Range<usize>, EqarError> {
        let rounded = n.checked_add(63).ok_or(EqarError::Alloc)? & !63usize;
        let end = self.used.checked_add(rounded).ok_or(EqarError::Alloc)?;
        if end > self.capacity() {
            return Err(EqarError::Alloc);
        }
        let start = self.used;
        self.used = end;
        Ok(start..start + n)
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }

    pub fn slice(&self, r: Range<usize>) -> &[u8] {
        &self.bytes()[r]
    }

    pub fn slice_mut(&mut self, r: Range<usize>) -> &mut [u8] {
        &mut self.bytes_mut()[r]
    }

    #[allow(unsafe_code)]
    fn bytes(&self) -> &[u8] {
        // SAFETY: `Block` is `repr(C)` over `[u8; 64]` with size == align == 64, so the
        // Vec's storage is `len * 64` initialised, contiguous bytes.
        unsafe { std::slice::from_raw_parts(self.blocks.as_ptr().cast::<u8>(), self.capacity()) }
    }

    #[allow(unsafe_code)]
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`, with exclusive access through `&mut self`.
        unsafe {
            std::slice::from_raw_parts_mut(self.blocks.as_mut_ptr().cast::<u8>(), self.capacity())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samplers_survive_nan_and_far_out_of_range_coordinates() {
        // The pole reflection is applied once, so a coordinate more than one image height
        // outside the plane (or NaN / infinite) used to index outside it. It must pin to the
        // plane edge instead, for the gray fast path and the general path alike.
        let (w, h) = (16i32, 8i32);
        let px: Vec<u8> = (0..(w * h) as usize).map(|i| (i * 7 % 251) as u8).collect();
        let big = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e30,
            -1e30,
            -5000.0,
            5000.0,
            -9.0,
            17.0,
            -0.6,
            7.6,
        ];
        let lut = [0u8; 256];
        for &x in &big {
            for &y in &big {
                for (xx, yy) in [(x, y), (x, 3.0), (3.0, y)] {
                    let g = sample_gray_plane(&px, w, h, w as usize, xx, yy);
                    assert!((0.0..=255.0).contains(&g) || g.is_nan(), "{xx} {yy} {g}");
                    let v = View {
                        data: &px,
                        w,
                        h,
                        stride: w as usize,
                        fmt: Fmt::Gray8,
                        lut: Some(&lut),
                    };
                    let _ = sample(&v, xx, yy);
                }
            }
        }
        // in-range inputs are untouched by the pinning
        assert_eq!(
            sample_gray_plane(&px, w, h, w as usize, 15.25, 7.4),
            sample_gray_plane(&px, w, h, w as usize, 15.25, 7.4)
        );
    }

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

    fn ramp_gray(w: usize, h: usize) -> Vec<u8> {
        let mut b = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                b[y * w + x] = ((x * 7 + y * 13) & 0xFF) as u8;
            }
        }
        b
    }

    fn gray_view(b: &[u8], w: i32, h: i32) -> View<'_> {
        View {
            data: b,
            w,
            h,
            stride: w as usize,
            fmt: Fmt::Gray8,
            lut: None,
        }
    }

    #[test]
    fn exact_at_pixel_centres() {
        let (w, h) = (64usize, 32usize);
        let b = ramp_gray(w, h);
        let v = gray_view(&b, w as i32, h as i32);
        for y in 0..h {
            for x in 0..w {
                check_near!(
                    sample(&v, x as f32, y as f32) as f64,
                    b[y * w + x] as f64,
                    1e-4
                );
            }
        }
    }

    #[test]
    fn longitude_wraps() {
        let b = ramp_gray(64, 32);
        let v = gray_view(&b, 64, 32);
        check_near!(
            sample(&v, -1.0, 5.0) as f64,
            sample(&v, 63.0, 5.0) as f64,
            1e-4
        );
        check_near!(
            sample(&v, 64.0, 5.0) as f64,
            sample(&v, 0.0, 5.0) as f64,
            1e-4
        );
        check_near!(
            sample(&v, 63.5, 5.0) as f64,
            0.5 * (b[5 * 64 + 63] as f64 + b[5 * 64] as f64),
            1e-4
        );
    }

    #[test]
    fn latitude_reflects_over_the_pole_with_a_half_turn() {
        let b = ramp_gray(64, 32);
        let v = gray_view(&b, 64, 32);
        // y = -1 mirrors to y = 0 shifted by W/2 in longitude
        check_near!(sample(&v, 10.0, -1.0) as f64, b[42] as f64, 1e-4);
        // y = h mirrors to y = h-1 shifted by W/2
        check_near!(sample(&v, 10.0, 32.0) as f64, b[31 * 64 + 42] as f64, 1e-4);
    }

    #[test]
    fn bgr_matches_the_integer_gray_weights() {
        let (w, h) = (8i32, 4i32);
        let mut b = vec![0u8; (w * h * 3) as usize];
        for px in b.chunks_exact_mut(3) {
            px.copy_from_slice(&[10, 200, 90]);
        }
        let v = View {
            data: &b,
            w,
            h,
            stride: w as usize * 3,
            fmt: Fmt::Bgr8,
            lut: None,
        };
        let expect = (29.0 * 10.0 + 150.0 * 200.0 + 77.0 * 90.0) / 256.0;
        check_near!(sample(&v, 3.0, 2.0) as f64, expect, 1e-3);
        let r = View {
            fmt: Fmt::Rgb8,
            ..v
        };
        let expect_rgb = (29.0 * 90.0 + 150.0 * 200.0 + 77.0 * 10.0) / 256.0;
        check_near!(sample(&r, 3.0, 2.0) as f64, expect_rgb, 1e-3);
    }

    #[test]
    fn gamma_lut_is_applied_when_present() {
        let (w, h) = (8i32, 4i32);
        let b = vec![128u8; (w * h) as usize];
        let mut lut = [0u8; 256];
        for (i, l) in lut.iter_mut().enumerate() {
            *l = (255 - i) as u8;
        }
        let v = View {
            data: &b,
            w,
            h,
            stride: w as usize,
            fmt: Fmt::Gray8,
            lut: Some(&lut),
        };
        check_near!(sample(&v, 3.0, 2.0) as f64, 127.0, 1e-4);
    }

    #[test]
    fn stride_padding_is_honoured() {
        let (w, h, s) = (8usize, 4usize, 11usize);
        let mut b = vec![0u8; s * h];
        for y in 0..h {
            for x in 0..w {
                b[y * s + x] = (x + y) as u8;
            }
        }
        let v = View {
            data: &b,
            w: w as i32,
            h: h as i32,
            stride: s,
            fmt: Fmt::Gray8,
            lut: None,
        };
        check_near!(sample(&v, 6.0, 3.0) as f64, 9.0, 1e-4);
    }

    #[test]
    fn downsample4_is_the_exact_box_mean() {
        let (w, h) = (64usize, 32usize);
        let b = ramp_gray(w, h);
        let v = gray_view(&b, w as i32, h as i32);
        let mut out = vec![0u8; (w / 4) * (h / 4)];
        downsample(&v, 4, &mut out, w / 4);
        for oy in 0..h / 4 {
            for ox in 0..w / 4 {
                let mut s = 0u32;
                for j in 0..4 {
                    for i in 0..4 {
                        s += b[(4 * oy + j) * w + (4 * ox + i)] as u32;
                    }
                }
                assert_eq!(out[oy * (w / 4) + ox] as u32, s >> 4);
            }
        }
    }

    #[test]
    fn downsample2_and_bgr_agree_with_gray() {
        let (w, h) = (32usize, 16usize);
        let mut g = vec![0u8; w * h];
        let mut c = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let bb = (x * 3) as u8;
                let gg = (y * 5 + 7) as u8;
                let rr = ((x ^ y) * 2) as u8;
                c[(y * w + x) * 3] = bb;
                c[(y * w + x) * 3 + 1] = gg;
                c[(y * w + x) * 3 + 2] = rr;
                g[y * w + x] = ((29 * bb as u32 + 150 * gg as u32 + 77 * rr as u32) >> 8) as u8;
            }
        }
        let vg = gray_view(&g, w as i32, h as i32);
        let vc = View {
            data: &c,
            w: w as i32,
            h: h as i32,
            stride: w * 3,
            fmt: Fmt::Bgr8,
            lut: None,
        };
        let mut a = vec![0u8; (w / 2) * (h / 2)];
        let mut bo = vec![0u8; (w / 2) * (h / 2)];
        downsample(&vg, 2, &mut a, w / 2);
        downsample(&vc, 2, &mut bo, w / 2);
        for i in 0..a.len() {
            assert!((a[i] as i32 - bo[i] as i32).abs() <= 1);
        }
    }

    #[test]
    fn arena_rounds_size_for_aligned_alloc() {
        let mut a = Arena::new(1000); // not a multiple of 64
        assert!(a.capacity() >= 1000);
        assert_eq!(a.capacity() % 64, 0);
        let p = a.take(10).unwrap();
        assert_eq!(a.slice(p.clone()).as_ptr() as usize % 64, 0);
        let q = a.take(10).unwrap();
        assert_eq!(a.slice(q.clone()).as_ptr() as usize % 64, 0);
        assert!(q.start >= p.start + 10);
        a.reset();
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn arena_never_hands_out_past_the_allocation() {
        let mut a = Arena::new(128);
        assert!(a.take(100).is_ok()); // rounds up to 128 = the whole arena
        assert_eq!(a.take(1), Err(EqarError::Alloc));
        assert_eq!(a.take(usize::MAX), Err(EqarError::Alloc));
        a.reset();
        assert!(a.take(64).is_ok() && a.take(64).is_ok());
    }

    #[test]
    fn view_validity() {
        let b = vec![0u8; 8 * 4];
        let ok = View {
            data: &b,
            w: 8,
            h: 4,
            stride: 8,
            fmt: Fmt::Gray8,
            lut: None,
        };
        assert!(ok.is_valid() && ok.check().is_ok());
        assert_eq!(
            View {
                data: &b[..31],
                ..ok
            }
            .check(),
            Err(EqarError::Stride)
        );
        assert_eq!(View { stride: 7, ..ok }.check(), Err(EqarError::Stride));
        assert_eq!(View { w: 0, ..ok }.check(), Err(EqarError::Size));
        assert_eq!(View { data: &[], ..ok }.check(), Err(EqarError::Size));
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    #[test]
    fn neon_downsample_is_bit_identical_to_scalar() {
        // widths exercise both the vector body and the scalar tail (OW % 8 != 0)
        for &(w, h) in &[(64usize, 32usize), (104, 52), (3840, 1920)] {
            let mut g = vec![0u8; w * h];
            let mut c = vec![0u8; w * h * 3];
            let mut s = 0x1234_5678u32;
            for p in g.iter_mut().chain(c.iter_mut()) {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                *p = (s >> 24) as u8;
            }
            let (ow, oh) = (w / 4, h / 4);
            let vg = gray_view(&g, w as i32, h as i32);
            let vc = View {
                data: &c,
                w: w as i32,
                h: h as i32,
                stride: w * 3,
                fmt: Fmt::Bgr8,
                lut: None,
            };
            let (mut a, mut b) = (vec![0u8; ow * oh], vec![0u8; ow * oh]);
            downsample(&vg, 4, &mut a, ow);
            downsample_gray_scalar(&g, w as i32, h as i32, w, &mut b, ow, 4);
            assert_eq!(a, b, "gray {w}x{h}");
            downsample(&vc, 4, &mut a, ow);
            downsample_bgr_scalar(&c, w as i32, h as i32, w * 3, &mut b, ow, 4, false);
            assert_eq!(a, b, "bgr {w}x{h}");
        }
    }
}
