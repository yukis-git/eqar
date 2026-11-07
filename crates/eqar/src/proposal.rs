//! Proposal stage: where might a marker be?
//!
//! A whole-equirect f=4 band (two threshold offsets) plus two polar caps, and an
//! optional f=2 band for small markers, all built on a tile-grid adaptive threshold,
//! a seam-wrapped run-length connected-component pass and a shape filter. Candidates
//! carry a centre ray, an angular radius and the blob-quad corner rays.
//!
//! Scratch buffers live in the structs ([`Proposal`], [`TileGrid`], [`RunCcl`],
//! [`ShapeScratch`]) and are sized once; a frame allocates nothing once their
//! capacities have grown.
//!
//! Faithful to the C++ original, including its quirks: the second threshold offset is a
//! fallback that runs only when the primary offset produced no tier-1 candidate, and
//! `ProposalDebug::n_roots_small` / `n_cand_small` accumulate across `run_small` calls
//! (only [`Proposal::run`] resets the counters).

use crate::error::EqarError;
use crate::geometry::{dot, normalize, px_to_ray, ray_to_px, slerp, Vec3, K_PI};
use crate::image::{downsample, fmadd, sample, Fmt, View};
use std::time::Instant;

// ---- constants ----------------------------------------------------------------

/// Tile grid over a decimated gray plane: tiles are `K_TILE` x `K_TILE` proposal px.
pub const K_TILE: usize = 4;
/// Box mean radius in tiles; 9x9 tiles = a 36 px window.
pub const K_TILE_RADIUS: usize = 4;
/// Primary threshold offset, DN.
pub const K_THRESH_C: i32 = 7;
/// Second threshold offset, DN.
pub const K_THRESH_C2: i32 = 14;
/// Polar cap half field of view. The band is sound to |lat| 75 and the caps carry
/// everything from |lat| 80, so 35 deg (cap edge at |lat| 55) leaves 20 deg of overlap.
pub const K_CAP_HALF_FOV_DEG: f64 = 35.0;
/// A tier-1 candidate whose enclosed white fraction lies in this band is ordered ahead
/// of one outside it (an ordering key only, never a filter).
pub const K_TIER1_HOLE_FRAC_LO: f32 = 0.10;
pub const K_TIER1_HOLE_FRAC_HI: f32 = 0.60;
/// Hard per-frame candidate cap applied after `order_for_budget`.
pub const K_MAX_CANDIDATES_PER_FRAME: usize = 2048;

// ---- tile grid ----------------------------------------------------------------

/// Tile sums and their box sums over a decimated plane; sized once, reused every frame.
#[derive(Debug, Clone, Default)]
pub struct TileGrid {
    /// tiles across / down
    pub tw: usize,
    pub th: usize,
    /// `tw * th` tile sums
    pub sum: Vec<u32>,
    /// `tw * th` box sums (x wrapped, y clamped; rows clamped at the poles are
    /// rescaled onto `boxdiv`)
    pub box_sum: Vec<u32>,
    /// pixels contributing to a full box
    pub boxdiv: u32,
    hbuf: Vec<u32>,
    acc: Vec<u32>,
    thr: Vec<u8>,
}

impl TileGrid {
    pub fn new(tw: usize, th: usize) -> Self {
        TileGrid {
            tw,
            th,
            sum: vec![0; tw * th],
            box_sum: vec![0; tw * th],
            ..Default::default()
        }
    }
}

/// One raster pass: `sum[ty*tw+tx]` = sum of the tile's `K_TILE` x `K_TILE` pixels.
///
/// The NEON path sums 4 tiles (16 source columns) per step with two pairwise widening
/// adds; integer addition of these small values is exact, so NEON and scalar agree
/// bit for bit.
pub fn tile_sums(p: &[u8], _w: usize, _h: usize, stride: usize, g: &mut TileGrid) {
    let tw = g.tw;
    for ty in 0..g.th {
        let srow = &mut g.sum[ty * tw..(ty + 1) * tw];
        srow.fill(0);
        for j in 0..K_TILE {
            let row = &p[(ty * K_TILE + j) * stride..][..tw * K_TILE];
            #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
            let tx = crate::simd::tile_sums_row(row, srow);
            #[cfg(not(all(target_arch = "aarch64", not(feature = "no-neon"))))]
            let tx = 0;
            for (s, seg) in srow[tx..]
                .iter_mut()
                .zip(row[tx * K_TILE..].chunks_exact(K_TILE))
            {
                *s += seg.iter().map(|&b| b as u32).sum::<u32>();
            }
        }
    }
}

/// Separable box filter over the tile grid: horizontal wraps (the +/-180 seam is
/// continuous in tile space), vertical clamps (the poles are not periodic).
///
/// A clamped vertical window is NOT filled out by replicating the edge row (that biases
/// the mean at the poles): only the rows that exist are summed, and a short window's
/// box sum is rescaled (round to nearest) onto the full-window count `boxdiv`, so the
/// threshold step needs no row-varying divisor. Both passes are running sums.
pub fn tile_box(g: &mut TileGrid) {
    let (tw, th, r) = (g.tw, g.th, K_TILE_RADIUS);
    // every element is written by the horizontal pass before it is read
    g.hbuf.resize(tw * th, 0);

    for ty in 0..th {
        let s = &g.sum[ty * tw..(ty + 1) * tw];
        let h = &mut g.hbuf[ty * tw..(ty + 1) * tw];
        let mut run = 0u32;
        for i in -(r as isize)..=(r as isize) {
            run += s[i.rem_euclid(tw as isize) as usize];
        }
        for tx in 0..tw {
            h[tx] = run;
            let mut add = tx + r + 1;
            if add >= tw {
                add -= tw;
            }
            let sub = if tx >= r { tx - r } else { tx + tw - r };
            run = run.wrapping_add(s[add]).wrapping_sub(s[sub]);
        }
    }

    let full = 2 * r + 1;
    g.boxdiv = (full * full * K_TILE * K_TILE) as u32;

    // The vertical pass is a running sum, carried in a scratch row because the rows of
    // `box_sum` get rescaled in place at the clamped edges.
    g.acc.clear();
    g.acc.resize(tw, 0);
    let hi0 = r.min(th - 1);
    for yy in 0..=hi0 {
        for (a, &h) in g.acc.iter_mut().zip(&g.hbuf[yy * tw..(yy + 1) * tw]) {
            *a += h;
        }
    }
    for ty in 0..th {
        let out = &mut g.box_sum[ty * tw..(ty + 1) * tw];
        let lo = ty.saturating_sub(r);
        let hi = (ty + r).min(th - 1);
        out.copy_from_slice(&g.acc);
        {
            // slide the window to row ty + 1
            let nlo = (ty + 1).saturating_sub(r);
            let nhi = (ty + 1 + r).min(th - 1);
            if nhi > hi {
                for (a, &h) in g.acc.iter_mut().zip(&g.hbuf[nhi * tw..(nhi + 1) * tw]) {
                    *a = a.wrapping_add(h);
                }
            }
            if nlo > lo {
                for (a, &h) in g.acc.iter_mut().zip(&g.hbuf[lo * tw..(lo + 1) * tw]) {
                    *a = a.wrapping_sub(h);
                }
            }
        }
        let covered = hi - lo + 1;
        if covered != full {
            for o in out.iter_mut() {
                let scaled = *o as u64 * full as u64;
                *o = ((scaled + (covered / 2) as u64) / covered as u64) as u32;
            }
        }
    }
}

/// `dst[i] = 255` when `p[i] < boxmean(i) - c`, else 0 (THRESH_BINARY_INV).
///
/// Each pixel uses its own tile's box mean (no interpolation). The mean is
/// `(box * recip) >> 16` with `recip` the NEAREST fixed-point reciprocal (a truncated one
/// is biased low by ~1.1 % and flips pixels at the threshold). The per-tile threshold is
/// expanded once per tile row into a `w`-byte line and applied to its `K_TILE` output
/// rows with an unsigned byte compare; clamping it into [0, 255] is exact.
pub fn threshold_from_tiles(
    p: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    g: &mut TileGrid,
    c: i32,
    dst: &mut [u8],
    dstride: usize,
) {
    let tw = g.tw;
    let denom = g.boxdiv;
    let recip = ((1u32 << 16) + denom / 2) / denom;
    g.thr.clear();
    g.thr.resize(w, 0);
    let clampt = |boxsum: u32| -> u8 {
        let t = (boxsum.wrapping_mul(recip) >> 16) as i32 - c;
        t.clamp(0, 255) as u8
    };

    let mut y = 0;
    while y < h {
        // a ragged bottom (h not a multiple of K_TILE) reuses the last tile row
        let ty = (y / K_TILE).min(g.th - 1);
        let boxrow = &g.box_sum[ty * tw..(ty + 1) * tw];
        let (body, tail) = g.thr.split_at_mut(tw * K_TILE);
        for (seg, &bx) in body.chunks_exact_mut(K_TILE).zip(boxrow) {
            seg.fill(clampt(bx));
        }
        tail.fill(clampt(boxrow[tw - 1])); // ragged tail, if any
        let yend = (y + K_TILE).min(h);
        for yy in y..yend {
            let srow = &p[yy * stride..][..w];
            let drow = &mut dst[yy * dstride..][..w];
            #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
            let x = crate::simd::cmp_lt_row(srow, &g.thr, drow);
            #[cfg(not(all(target_arch = "aarch64", not(feature = "no-neon"))))]
            let x = 0;
            for ((d, &s), &t) in drow[x..].iter_mut().zip(&srow[x..]).zip(&g.thr[x..]) {
                *d = if s < t { 255 } else { 0 };
            }
        }
        y = yend;
    }
}

// ---- run-length CCL, wrapped at the +/-180 seam ----------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Blob {
    pub area: i32,
    /// first column touched; the unwrap origin
    pub xref: i32,
    /// UNWRAPPED, relative to `xref` (may exceed `[0, w)`)
    pub xmin: i32,
    pub xmax: i32,
    pub ymin: i32,
    pub ymax: i32,
    /// best of x+y, x-y, -x+y, -x-y (x unwrapped)
    pub ext: [i32; 4],
    /// and where
    pub extx: [i32; 4],
    pub exty: [i32; 4],
    /// 1 when the blob was merged across column 0 / w-1
    pub touches_seam: u8,
    pub has_hole: u8,
}

/// One row's run of foreground pixels, retained after labelling so the accumulation of
/// blob statistics and the shape filter's mask are sequential walks of the blob's own
/// runs. `root` is the provisional label while labelling, the resolved root afterwards.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlobRun {
    pub y: i32,
    pub cs: i32,
    pub ce: i32,
    pub root: i32,
}

#[derive(Debug, Clone, Default)]
pub struct RunCcl {
    pub parent: Vec<i32>,
    pub blobs: Vec<Blob>,
    /// Runs in raster order, and a CSR grouping of them by ROOT label: root `i`'s runs
    /// are `runs[run_index[k]]` for `k` in `run_at[i]..run_at[i+1]`, in raster order.
    pub runs: Vec<BlobRun>,
    pub run_index: Vec<i32>,
    pub run_at: Vec<i32>,
    xref_of: Vec<i32>,
    cursor: Vec<i32>,
    seam_of: Vec<u8>,
}

impl RunCcl {
    pub fn find(&mut self, mut a: i32) -> i32 {
        while self.parent[a as usize] != a {
            let g = self.parent[self.parent[a as usize] as usize];
            self.parent[a as usize] = g;
            a = g;
        }
        a
    }

    pub fn uni(&mut self, a: i32, b: i32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent[a.max(b) as usize] = a.min(b);
        }
    }

    pub fn clear(&mut self) {
        self.parent.clear();
        self.parent.push(0);
        self.blobs.clear();
        self.blobs.push(Blob::default());
        self.runs.clear();
        self.run_index.clear();
        self.run_at.clear();
        self.run_at.extend_from_slice(&[0, 0]);
    }

    pub fn run_first(&self, root: i32) -> i32 {
        self.run_at[root as usize]
    }

    pub fn run_last(&self, root: i32) -> i32 {
        self.run_at[root as usize + 1]
    }
}

/// Re-express `x` relative to `xref` as the representative nearest it.
fn unwrap_x(xref: i32, x: i32, w: i32) -> i32 {
    let mut d = x - xref;
    let half = w / 2;
    while d > half {
        d -= w;
    }
    while d < -half {
        d += w;
    }
    xref + d
}

/// Fold one run's two endpoints (the only points on a horizontal run that can be a
/// linear extreme) into a blob's running stats. Each endpoint can win only two of the
/// four extremes (x+y and x-y at the rightmost unwrapped end, -x+y and -x-y at the
/// leftmost); a run whose endpoints unwrap to opposite sides of the reference falls back
/// to the general form.
fn accumulate_run(b: &mut Blob, cs: i32, ce: i32, y: i32, w: i32) {
    b.area += ce - cs + 1;
    let xu0 = unwrap_x(b.xref, cs, w);
    let xu1 = unwrap_x(b.xref, ce, w);
    b.xmin = b.xmin.min(xu0).min(xu1);
    b.xmax = b.xmax.max(xu0).max(xu1);
    b.ymin = b.ymin.min(y);
    b.ymax = b.ymax.max(y);

    let fold = |b: &mut Blob, xu: i32, k: usize, val: i32| {
        if val > b.ext[k] {
            b.ext[k] = val;
            b.extx[k] = xu;
            b.exty[k] = y;
        }
    };
    if xu1 >= xu0 {
        fold(b, xu1, 0, xu1 + y);
        fold(b, xu1, 1, xu1 - y);
        fold(b, xu0, 2, -xu0 + y);
        fold(b, xu0, 3, -xu0 - y);
    } else {
        for xu in [xu0, xu1] {
            fold(b, xu, 0, xu + y);
            fold(b, xu, 1, xu - y);
            fold(b, xu, 2, -xu + y);
            fold(b, xu, 3, -xu - y);
        }
    }
}

/// Blob-local column for source column `c` (the inverse of the mask mapping): `xmin` is
/// an unwrapped value and may be negative or exceed `w`.
#[inline]
fn local_col(xmin: i32, c: i32, w: i32) -> usize {
    let mut d = (c as i64 - xmin as i64) % w as i64;
    if d < 0 {
        d += w as i64;
    }
    d as usize
}

/// Nibble `i` of the result is 0xF iff `s[x+i] != 0`, for `n <= 16` bytes from `x`.
#[inline]
fn block_mask(s: &[u8], x: usize, n: usize) -> u64 {
    #[cfg(all(target_arch = "aarch64", not(feature = "no-neon")))]
    if n == 16 {
        if let Ok(a) = <&[u8; 16]>::try_from(&s[x..x + 16]) {
            return crate::simd::nibble_mask16(a);
        }
    }
    block_mask_scalar(s, x, n)
}

#[inline]
fn block_mask_scalar(s: &[u8], x: usize, n: usize) -> u64 {
    let mut m = 0u64;
    for (i, &b) in s[x..x + n].iter().enumerate() {
        if b != 0 {
            m |= 0xFu64 << (4 * i);
        }
    }
    m
}

/// One row's runs of foreground pixels, via a 16-byte nibble mask: a wholly background
/// or wholly foreground block is one test, and inside a mixed block every run boundary
/// is one count-trailing-zeros. `open` carries a run reaching the end of a block into the
/// next, so runs come out sorted by `cs` and disjoint.
fn extract_runs(s: &[u8], w: usize, y: i32, runs: &mut Vec<BlobRun>) {
    extract_runs_with(s, w, y, runs, block_mask);
}

fn extract_runs_with(
    s: &[u8],
    w: usize,
    y: i32,
    runs: &mut Vec<BlobRun>,
    mask: impl Fn(&[u8], usize, usize) -> u64,
) {
    let mut open: i32 = -1;
    let mut x = 0usize;
    while x < w {
        let n = (w - x).min(16);
        let m = mask(s, x, n);
        let full = if n == 16 {
            !0u64
        } else {
            (1u64 << (4 * n)) - 1
        };
        if m == 0 {
            // all background: closes any open run
            if open >= 0 {
                runs.push(BlobRun {
                    y,
                    cs: open,
                    ce: x as i32 - 1,
                    root: 0,
                });
                open = -1;
            }
        } else if m == full {
            // all foreground: opens or extends one
            if open < 0 {
                open = x as i32;
            }
        } else {
            let mut i = 0usize;
            while i < n {
                if open < 0 {
                    // hunt the next foreground lane
                    let o = m >> (4 * i);
                    if o == 0 {
                        break;
                    }
                    i += (o.trailing_zeros() >> 2) as usize;
                    open = (x + i) as i32;
                } else {
                    // hunt the lane that ends this run
                    let z = (!m & full) >> (4 * i);
                    if z == 0 {
                        break;
                    }
                    i += (z.trailing_zeros() >> 2) as usize;
                    runs.push(BlobRun {
                        y,
                        cs: open,
                        ce: (x + i) as i32 - 1,
                        root: 0,
                    });
                    open = -1;
                }
            }
        }
        x += n;
    }
    if open >= 0 {
        runs.push(BlobRun {
            y,
            cs: open,
            ce: w as i32 - 1,
            root: 0,
        });
    }
}

/// 8-connected, single raster pass, x-WRAPPED. Returns the number of roots.
///
/// Labelling records each run with a provisional label; once union-find has settled the
/// runs are grouped by ROOT with a counting sort (stable, so each group stays in raster
/// order, which makes `accumulate_run`'s strict `>` tie-breaking deterministic) and each
/// group is folded into its blob once. The previous row's overlapping runs are found by
/// a merge walk, since both run lists are sorted and disjoint.
pub fn run_ccl_wrap(bin: &[u8], w: usize, h: usize, stride: usize, r: &mut RunCcl) -> i32 {
    r.clear();
    r.xref_of.clear();
    r.xref_of.push(0);
    r.seam_of.clear();
    r.seam_of.push(0);
    let wi = w as i32;
    let (mut prev_lo, mut prev_hi) = (0usize, 0usize);
    for y in 0..h {
        let cur_lo = r.runs.len();
        extract_runs(&bin[y * stride..][..w], w, y as i32, &mut r.runs);
        let cur_hi = r.runs.len();
        let (mut first_lab, mut last_lab) = (0i32, 0i32);
        let mut pi = prev_lo;
        for ri in cur_lo..cur_hi {
            let run = r.runs[ri];
            let mut best = 0i32;
            let mut wrap_only = false;
            macro_rules! take {
                ($v:expr, $via:expr) => {{
                    let v: i32 = $v;
                    if v != 0 {
                        if best == 0 {
                            best = v;
                        } else {
                            r.uni(best, v);
                        }
                        if $via {
                            wrap_only = true;
                        }
                    }
                }};
            }
            while pi < prev_hi && r.runs[pi].ce < run.cs - 1 {
                pi += 1;
            }
            let mut j = pi;
            while j < prev_hi && r.runs[j].cs <= run.ce + 1 {
                take!(r.runs[j].root, false);
                j += 1;
            }
            // The two seam links the merge walk cannot see.
            if prev_lo != prev_hi {
                let pfirst = r.runs[prev_lo];
                let plast = r.runs[prev_hi - 1];
                if run.cs == 0 && plast.ce == wi - 1 {
                    let normal = plast.ce >= run.cs - 1 && plast.cs <= run.ce + 1;
                    take!(plast.root, !normal);
                }
                if run.ce == wi - 1 && pfirst.cs == 0 {
                    let normal = pfirst.ce >= run.cs - 1 && pfirst.cs <= run.ce + 1;
                    take!(pfirst.root, !normal);
                }
            }
            if best == 0 {
                // a label is created by its component's first run in raster order, so
                // this is the column of the blob's first-ever pixel
                best = r.parent.len() as i32;
                r.parent.push(best);
                r.xref_of.push(run.cs);
                r.seam_of.push(0);
            }
            r.runs[ri].root = best;
            if wrap_only {
                r.seam_of[best as usize] = 1;
            }
            if run.cs == 0 {
                first_lab = best;
            }
            if run.ce == wi - 1 {
                last_lab = best;
            }
        }
        if first_lab != 0 && last_lab != 0 && first_lab != last_lab {
            r.uni(first_lab, last_lab);
            r.seam_of[first_lab as usize] = 1;
            r.seam_of[last_lab as usize] = 1;
        }
        prev_lo = cur_lo;
        prev_hi = cur_hi;
    }
    let nlab = r.parent.len();
    let nrun = r.runs.len();

    // ---- resolve, group by root, then fold each group once --------------------
    r.run_at.clear();
    r.run_at.resize(nlab + 1, 0);
    for i in 0..nrun {
        let root = r.find(r.runs[i].root);
        r.runs[i].root = root;
        r.run_at[root as usize + 1] += 1;
    }
    for i in 0..nlab {
        r.run_at[i + 1] += r.run_at[i];
    }
    r.run_index.clear();
    r.run_index.resize(nrun, 0);
    r.cursor.clear();
    r.cursor.extend_from_slice(&r.run_at[..nlab]);
    for i in 0..nrun {
        let root = r.runs[i].root as usize;
        r.run_index[r.cursor[root] as usize] = i as i32;
        r.cursor[root] += 1;
    }

    r.blobs.clear();
    r.blobs.resize(nlab, Blob::default());
    let mut count = 0;
    for i in 1..nlab {
        let (k0, k1) = (r.run_at[i] as usize, r.run_at[i + 1] as usize);
        if k0 == k1 {
            continue; // a non-root: its runs went to its root
        }
        count += 1;
        let b = &mut r.blobs[i];
        b.xref = r.xref_of[i];
        b.xmin = i32::MAX;
        b.ymin = i32::MAX;
        b.xmax = i32::MIN;
        b.ymax = i32::MIN;
        b.ext = [i32::MIN; 4];
        for &ri in &r.run_index[k0..k1] {
            let rn = r.runs[ri as usize];
            accumulate_run(b, rn.cs, rn.ce, rn.y, wi);
        }
    }
    // The seam flag can have been raised on any label of a component.
    for i in 1..nlab {
        if r.seam_of[i] != 0 {
            let root = r.find(i as i32) as usize;
            r.blobs[root].touches_seam = 1;
        }
    }
    count
}

// ---- shape filter, hole test, dark-ring probe, blob quad ---------------------------

#[derive(Debug, Clone, Copy)]
pub struct ShapeCfg {
    pub solidity_min: f32,
    pub extent_min: f32,
    pub ar_max: f32,
    /// proposal px
    pub min_side: i32,
    pub min_minor_side: i32,
    /// of the plane height
    pub max_side_frac: f32,
    pub hole_frac: f32,
    /// dark-ring probe threshold
    pub ring_dn_min: f32,
    /// The reference's own (strict) shape thresholds: a candidate that also clears these
    /// and the dark-ring probe is tier 1 and is offered to the budget first. Nothing is
    /// rejected by them. The reference's `ext_hi` is deliberately not ported as such:
    /// its `ext` is the raw blob area over bbox, which is `tier1_raw_extent_*` below.
    pub tier1_solidity_min: f32,
    pub tier1_extent_min: f32,
    pub tier1_raw_extent_min: f32,
    pub tier1_raw_extent_max: f32,
}

impl Default for ShapeCfg {
    fn default() -> Self {
        ShapeCfg {
            solidity_min: 0.80,
            extent_min: 0.05,
            ar_max: 5.0,
            min_side: 6,
            min_minor_side: 3,
            max_side_frac: 0.5,
            hole_frac: 0.04,
            ring_dn_min: 12.0,
            tier1_solidity_min: 0.85,
            tier1_extent_min: 0.25,
            tier1_raw_extent_min: 0.25,
            tier1_raw_extent_max: 0.98,
        }
    }
}

/// The measurements `shape_ok` computes, handed back so a caller can tier a candidate.
/// `solidity` and `extent` are both on the FILLED area (own pixels plus enclosed holes).
#[derive(Debug, Clone, Copy, Default)]
pub struct ShapeInfo {
    pub solidity: f32,
    /// FILLED area / bbox
    pub extent: f32,
    /// the blob's OWN pixels / bbox
    pub raw_extent: f32,
    /// enclosed background components over `hole_frac`
    pub n_holes: i32,
    pub has_hole: bool,
}

const CELL_BLOB: u8 = 1;
const CELL_SEEN: u8 = 2;

#[derive(Debug, Clone, Copy, Default)]
pub struct Pt {
    x: i32,
    y: i32,
}

/// Reusable scratch for the shape filter (mask, span stack, row extremes, hull points).
#[derive(Debug, Clone, Default)]
pub struct ShapeScratch {
    cell: Vec<u8>,
    stack: Vec<i32>,
    row_lo: Vec<i32>,
    row_hi: Vec<i32>,
    pts: Vec<Pt>,
    hull: Vec<Pt>,
}

// A span is pushed as two int32 entries: the row, then (x1<<16)|x2 (bbox-local
// coordinates, bounded by the plane's dimensions).
#[inline]
fn push_span(s: &mut Vec<i32>, y: usize, x1: usize, x2: usize) {
    s.push(y as i32);
    s.push(((x1 as i32) << 16) | x2 as i32);
}

#[inline]
fn pop_span(s: &mut Vec<i32>) -> (usize, usize, usize) {
    let k = s.pop().unwrap_or(0);
    let y = s.pop().unwrap_or(0);
    (y as usize, (k >> 16) as usize, (k & 0xFFFF) as usize)
}

/// Marks, counts and queues the maximal still-0 (background, unseen) run through
/// `(ny, x)`. Returns its length, or 0 when `(ny, x)` is not such a cell.
fn emit_bg_run(cell: &mut [u8], bw: usize, stack: &mut Vec<i32>, ny: usize, x: usize) -> i64 {
    let row = &mut cell[ny * bw..(ny + 1) * bw];
    if row[x] != 0 {
        return 0;
    }
    let mut a = x;
    while a > 0 && row[a - 1] == 0 {
        a -= 1;
    }
    let mut e = x;
    while e < bw - 1 && row[e + 1] == 0 {
        e += 1;
    }
    row[a..=e].fill(CELL_SEEN);
    push_span(stack, ny, a, e);
    (e - a + 1) as i64
}

/// Floods every 4-connected still-0 cell reachable from the spans already on `stack`,
/// marking each `CELL_SEEN`; returns the count the CALLER has not already counted.
fn flood_background(cell: &mut [u8], bw: usize, bh: usize, stack: &mut Vec<i32>) -> i64 {
    let mut n = 0i64;
    while !stack.is_empty() {
        let (y, x1, x2) = pop_span(stack);
        for dy in [-1isize, 1] {
            let ny = y as isize + dy;
            if ny < 0 || ny as usize >= bh {
                continue;
            }
            let ny = ny as usize;
            // 4-connectivity: only a run overlapping [x1, x2] itself is adjacent
            let row = &mut cell[ny * bw..(ny + 1) * bw];
            let mut x = x1;
            while x <= x2 {
                if row[x] != 0 {
                    x += 1;
                    continue;
                }
                let mut e = x;
                while e < bw - 1 && row[e + 1] == 0 {
                    e += 1;
                }
                let mut a = x;
                while a > 0 && row[a - 1] == 0 {
                    a -= 1;
                }
                row[a..=e].fill(CELL_SEEN);
                n += (e - a + 1) as i64;
                push_span(stack, ny, a, e);
                x = e + 1;
            }
        }
    }
    n
}

/// Hole test: 4-connected background components of the mask that do NOT touch the
/// bbox border. Flooding every border-touching background pixel marks "the outside";
/// any background pixel left unvisited is part of a fully enclosed hole. Returns the SUM
/// of every enclosed component's area (so the caller can build the FILLED area);
/// `has_hole` is true iff at least one enclosed component alone exceeds
/// `hole_frac * blob_area`.
#[allow(clippy::too_many_arguments)]
fn mask_hole_areas(
    cell: &mut [u8],
    bw: usize,
    bh: usize,
    blob_area: i32,
    hole_frac: f32,
    has_hole: &mut bool,
    n_holes: &mut i32,
    row_lo: &[i32],
    row_hi: &[i32],
    stack: &mut Vec<i32>,
) -> i64 {
    stack.clear();
    *has_hole = false;
    *n_holes = 0;
    // Seed "the outside": every background run touching the bbox border. The two
    // horizontal borders are walked run at a time.
    let mut outside = 0i64;
    let mut seed_row = |cell: &mut [u8], stack: &mut Vec<i32>, y: usize| {
        let row = &mut cell[y * bw..(y + 1) * bw];
        let mut x = 0;
        while x < bw {
            if row[x] != 0 {
                x += 1;
                continue;
            }
            let mut e = x;
            while e < bw - 1 && row[e + 1] == 0 {
                e += 1;
            }
            row[x..=e].fill(CELL_SEEN);
            push_span(stack, y, x, e);
            outside += (e - x + 1) as i64;
            x = e + 1;
        }
    };
    seed_row(cell, stack, 0);
    if bh > 1 {
        seed_row(cell, stack, bh - 1);
    }
    for y in 0..bh {
        outside += emit_bg_run(cell, bw, stack, y, 0);
        if bw > 1 {
            outside += emit_bg_run(cell, bw, stack, y, bw - 1);
        }
    }
    outside += flood_background(cell, bw, bh, stack); // consume the outside

    // Every cell is now the blob's own pixel, reached by the outside flood, or
    // enclosed. When the first two already account for the whole bbox there is no hole
    // and the scan below can be skipped.
    if outside + blob_area as i64 >= bw as i64 * bh as i64 {
        return 0;
    }

    let thresh = hole_frac as f64 * blob_area as f64;
    let mut total = 0i64;
    for y in 0..bh {
        // A hole pixel always has one of this blob's own pixels to its left AND right on
        // its own row, so only the strict interior of [row_lo, row_hi] can hold one.
        let (lo, hi) = (row_lo[y], row_hi[y]);
        if hi < 0 || hi - lo < 2 {
            continue;
        }
        for x in (lo + 1) as usize..hi as usize {
            if cell[y * bw + x] != 0 {
                continue;
            }
            // Anything still unmarked is enclosed: flood its whole component.
            let mut area = emit_bg_run(cell, bw, stack, y, x);
            area += flood_background(cell, bw, bh, stack);
            total += area;
            if area as f64 > thresh {
                *has_hole = true;
                *n_holes += 1;
            }
        }
    }
    total
}

#[inline]
fn cross2(o: Pt, a: Pt, b: Pt) -> i64 {
    (a.x - o.x) as i64 * (b.y - o.y) as i64 - (a.y - o.y) as i64 * (b.x - o.x) as i64
}

/// Monotone-chain convex hull area over a small point set (the blob's per-row run
/// endpoints; any pixel that is neither its row's leftmost nor rightmost can never be a
/// hull vertex). `pts` must ALREADY be lexicographically sorted and duplicate-free. The
/// shoelace area is a point-coordinate one, so it reads slightly under a solid shape's
/// pixel-count area, which is why the solidity bounds have headroom above 1.0.
fn hull_area(pts: &[Pt], h: &mut Vec<Pt>) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    h.clear();
    h.resize(2 * n, Pt::default());
    let mut k = 0usize;
    for &p in pts {
        while k >= 2 && cross2(h[k - 2], h[k - 1], p) <= 0 {
            k -= 1;
        }
        h[k] = p;
        k += 1;
    }
    let lower = k + 1;
    for i in (0..n - 1).rev() {
        while k >= lower && cross2(h[k - 2], h[k - 1], pts[i]) <= 0 {
            k -= 1;
        }
        h[k] = pts[i];
        k += 1;
    }
    h.truncate(k - 1); // drop the closing duplicate of h[0]
    let mut s = 0i64;
    for i in 0..h.len() {
        let a = h[i];
        let b = h[(i + 1) % h.len()];
        s += a.x as i64 * b.y as i64 - b.x as i64 * a.y as i64;
    }
    (s as f64).abs() * 0.5
}

/// Hull over the blob's per-row run endpoints, emitted as `Pt{row, column}` (row by row,
/// lo before hi) which is already sorted; a shoelace area is invariant under the swap.
fn blob_hull_area(
    row_lo: &[i32],
    row_hi: &[i32],
    bh: usize,
    pts: &mut Vec<Pt>,
    hull: &mut Vec<Pt>,
) -> f64 {
    pts.clear();
    for y in 0..bh {
        let (lo, hi) = (row_lo[y], row_hi[y]);
        if hi < 0 {
            continue;
        }
        pts.push(Pt { x: y as i32, y: lo });
        if hi != lo {
            pts.push(Pt { x: y as i32, y: hi });
        }
    }
    hull_area(pts, hull)
}

struct FilledStats {
    filled_area: i64,
    solidity: f32,
    has_hole: bool,
    n_holes: i32,
    /// the span bound already failed a threshold
    doomed: bool,
}

/// The blob's mask, hole test and FILLED area (own pixels plus every enclosed hole --
/// what the reference gets from the outer contour's area). Shared by `shape_ok` and the
/// test-only `blob_filled_solidity`.
///
/// `span_area` (sum over rows of `row_hi - row_lo + 1`) is a true upper bound on the
/// filled area, and both the extent and the solidity test are monotone in it, so when the
/// bound already falls below a threshold the hole scan cannot change the verdict and is
/// skipped (`doomed`). The bounds are formed in the SAME arithmetic `shape_ok` uses (f32
/// ratio against f32 threshold) so the shortcut can only reject what `shape_ok` would
/// have, and the verdict travels as an explicit flag: the skip also forces
/// `has_hole = false`, so a re-test of the bounds could land the blob in the hole-optional
/// acceptance path and accept it.
#[allow(clippy::too_many_arguments)]
fn blob_filled_stats(
    w: i32,
    ccl: &RunCcl,
    idx: i32,
    b: &Blob,
    hole_frac: f32,
    bw: usize,
    bh: usize,
    sc: &mut ShapeScratch,
    solidity_min: f32,
    extent_min: f32,
) -> FilledStats {
    let ShapeScratch {
        cell,
        stack,
        row_lo,
        row_hi,
        pts,
        hull,
    } = sc;
    row_lo.clear();
    row_lo.resize(bh, i32::MAX);
    row_hi.clear();
    row_hi.resize(bh, -1);

    // The blob's per-row extremes, straight off its runs. A run never wraps, so its
    // blob-local span is one contiguous [lx, lx + length - 1] inside [0, bw).
    let (k0, k1) = (ccl.run_first(idx) as usize, ccl.run_last(idx) as usize);
    for &ri in &ccl.run_index[k0..k1] {
        let rn = ccl.runs[ri as usize];
        let ly = (rn.y - b.ymin) as usize;
        let lx = local_col(b.xmin, rn.cs, w) as i32;
        let lxe = lx + (rn.ce - rn.cs);
        if lx < row_lo[ly] {
            row_lo[ly] = lx;
        }
        if lxe > row_hi[ly] {
            row_hi[ly] = lxe;
        }
    }

    let mut span = 0i64;
    for y in 0..bh {
        if row_hi[y] >= 0 {
            span += (row_hi[y] - row_lo[y] + 1) as i64;
        }
    }

    let hull_a = blob_hull_area(row_lo, row_hi, bh, pts, hull);

    let bbox_area = (bw as i64 * bh as i64) as f32;
    let sol_ub = if hull_a > 0.0 {
        (span as f64 / hull_a) as f32
    } else {
        0.0
    };
    let ext_ub = span as f32 / bbox_area;
    if (hull_a > 0.0 && sol_ub < solidity_min) || ext_ub < extent_min {
        return FilledStats {
            filled_area: span,
            solidity: sol_ub,
            has_hole: false,
            n_holes: 0,
            doomed: true,
        };
    }

    // Only a blob the bound could not retire pays for the mask: one fill per run.
    cell.clear();
    cell.resize(bw * bh, 0);
    for &ri in &ccl.run_index[k0..k1] {
        let rn = ccl.runs[ri as usize];
        let ly = (rn.y - b.ymin) as usize;
        let lx = local_col(b.xmin, rn.cs, w);
        cell[ly * bw + lx..][..(rn.ce - rn.cs + 1) as usize].fill(CELL_BLOB);
    }

    let mut has_hole = false;
    let mut n_holes = 0;
    let hole_area = mask_hole_areas(
        cell,
        bw,
        bh,
        b.area,
        hole_frac,
        &mut has_hole,
        &mut n_holes,
        row_lo,
        row_hi,
        stack,
    );
    let filled_area = b.area as i64 + hole_area;
    let solidity = if hull_a > 0.0 {
        (filled_area as f64 / hull_a) as f32
    } else {
        0.0
    };
    FilledStats {
        filled_area,
        solidity,
        has_hole,
        n_holes,
        doomed: false,
    }
}

/// Shoelace area of `blob_quad`'s 4 (already {0,1,3,2}-ordered) points.
fn quad_shoelace_area(q: &[[f32; 2]; 4]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..4 {
        let j = (i + 1) & 3;
        s += q[i][0] as f64 * q[j][1] as f64 - q[j][0] as f64 * q[i][1] as f64;
    }
    s.abs() * 0.5
}

/// Extreme-point quad in plane coords, ordered {0,1,3,2} (the shoelace-safe traversal),
/// x UNWRAPPED.
pub fn blob_quad(b: &Blob) -> [[f32; 2]; 4] {
    const ORD: [usize; 4] = [0, 1, 3, 2];
    let mut q = [[0f32; 2]; 4];
    for i in 0..4 {
        q[i] = [b.extx[ORD[i]] as f32, b.exty[ORD[i]] as f32];
    }
    q
}

/// Test-only: `shape_ok`'s own FILLED-area solidity (the blob's pixels plus every hole it
/// encloses, over its convex-hull area). Thresholds of 0 so the span shortcut never
/// fires: this reports the exact ratio, never a bound.
pub fn blob_filled_solidity(
    w: usize,
    ccl: &RunCcl,
    idx: i32,
    hole_frac: f32,
    scratch: &mut ShapeScratch,
) -> f32 {
    let b = ccl.blobs[idx as usize];
    let bw = (b.xmax - b.xmin + 1) as usize;
    let bh = (b.ymax - b.ymin + 1) as usize;
    blob_filled_stats(w as i32, ccl, idx, &b, hole_frac, bw, bh, scratch, 0.0, 0.0).solidity
}

/// Fills `blobs[idx].has_hole`; returns true when the blob is a usable candidate. `info`
/// receives the filled-area solidity/extent this call thresholded against -- written even
/// when the verdict is false, as long as the blob got as far as the mask.
pub fn shape_ok(
    w: usize,
    h: usize,
    ccl: &mut RunCcl,
    idx: i32,
    cfg: &ShapeCfg,
    scratch: &mut ShapeScratch,
    info: Option<&mut ShapeInfo>,
) -> bool {
    let mut local = ShapeInfo::default();
    let ok = shape_ok_inner(w, h, ccl, idx, cfg, scratch, &mut local);
    if let Some(i) = info {
        *i = local;
    }
    ok
}

fn shape_ok_inner(
    w: usize,
    h: usize,
    ccl: &mut RunCcl,
    idx: i32,
    cfg: &ShapeCfg,
    scratch: &mut ShapeScratch,
    info: &mut ShapeInfo,
) -> bool {
    let b = ccl.blobs[idx as usize];
    let bw = b.xmax - b.xmin + 1;
    let bh = b.ymax - b.ymin + 1;
    if bw <= 0 || bh <= 0 || b.area <= 0 {
        return false;
    }

    // ---- cheap prefilter: size, aspect (every blob) -----------------------------
    let big = bw.max(bh);
    let sm = bw.min(bh);
    if (big as f32) < (cfg.min_side as f32) {
        return false;
    }
    if sm < cfg.min_minor_side {
        return false;
    }
    if (big as f32) > cfg.max_side_frac * (h as f32) {
        return false;
    }
    if (big as f32) > cfg.ar_max * ((if sm > 1 { sm } else { 1 }) as f32) {
        return false;
    }

    // A RAW-area extent floor, free from the CCL's own area, ahead of the mask. The
    // filled extent is >= the raw extent, so this only rejects wispy blobs that would
    // have to clear extent_min purely on hole area.
    let raw_extent_pre = b.area as f32 / (bw as i64 * bh as i64) as f32;
    if raw_extent_pre < cfg.extent_min {
        return false;
    }

    // ---- this blob's own mask: hole test + FILLED area --------------------------
    // A large marker at proposal resolution is a THIN black ring around big white
    // data-module holes: ring pixels / hull is well under solidity_min, so the
    // reference's formula (the outer contour's area, which includes the holes) is
    // matched by adding the enclosed-component areas to the pixel count.
    let fs = blob_filled_stats(
        w as i32,
        ccl,
        idx,
        &b,
        cfg.hole_frac,
        bw as usize,
        bh as usize,
        scratch,
        cfg.solidity_min,
        cfg.extent_min,
    );
    ccl.blobs[idx as usize].has_hole = fs.has_hole as u8;
    if fs.doomed {
        return false; // span bound already failed extent or solidity
    }

    // Extent's natural range is (0, 1]; a perfectly solid axis-aligned blob (extent 1) is
    // a real, intended candidate, so only the relaxed floor is enforced.
    let bbox_area = (bw as i64 * bh as i64) as f32;
    let extent = fs.filled_area as f32 / bbox_area;
    *info = ShapeInfo {
        solidity: fs.solidity,
        extent,
        raw_extent: b.area as f32 / bbox_area,
        n_holes: fs.n_holes,
        has_hole: fs.has_hole,
    };
    if extent < cfg.extent_min {
        return false;
    }
    if fs.solidity < cfg.solidity_min {
        return false;
    }
    if fs.has_hole {
        return true; // the reference's original has-hole path
    }

    // ---- hole-optional path ----------------------------------------------------
    let q = blob_quad(&b);
    let qa = quad_shoelace_area(&q);
    if qa <= 0.0 {
        return false;
    }
    let fill = fs.filled_area as f64 / qa;
    (0.40..=1.60).contains(&fill)
}

/// 8-sample probe on the source view: inner border dark, outer quiet zone bright.
pub fn dark_ring_ok(v: &View, ray_quad: &[[f32; 3]; 4], min_dn: f32) -> bool {
    dark_ring_dn(v, ray_quad) >= min_dn as f64
}

/// The same probe's raw measurement: mean(outer 4 samples) - mean(inner 4 samples), DN.
/// The tier-1 ordering score.
pub fn dark_ring_dn(v: &View, ray_quad: &[[f32; 3]; 4]) -> f64 {
    let mut q = [Vec3::default(); 4];
    let mut c = Vec3::new(0.0, 0.0, 0.0);
    for i in 0..4 {
        q[i] = Vec3::new(
            ray_quad[i][0] as f64,
            ray_quad[i][1] as f64,
            ray_quad[i][2] as f64,
        );
        c.x += q[i].x;
        c.y += q[i].y;
        c.z += q[i].z;
    }
    let c = normalize(&c);

    let (mut inner_sum, mut outer_sum) = (0.0f64, 0.0f64);
    for i in 0..4 {
        let m = slerp(&q[i], &q[(i + 1) & 3], 0.5); // edge midpoint
        let inner = slerp(&c, &m, 0.85); // 15% back toward centre
        let outer = slerp(&c, &m, 1.25); // 25% past the midpoint
        let (px, py) = ray_to_px(&inner, v.w, v.h);
        inner_sum += sample(v, px as f32, py as f32) as f64;
        let (px, py) = ray_to_px(&outer, v.w, v.h);
        outer_sum += sample(v, px as f32, py as f32) as f64;
    }
    let mean_inner = inner_sum * 0.25;
    let mean_outer = outer_sum * 0.25;
    mean_outer - mean_inner
}

// ---- polar caps ---------------------------------------------------------------------

/// Precomputed bilinear tap of one cap pixel: the top-row offset into the f=4 plane, the
/// delta to the other row, and the two weights. The right-hand tap is always the left-hand
/// one plus one (the plane carries a duplicate of column 0 at column `w`), so the two taps
/// of a row are adjacent bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tap {
    pub o0: i32,
    pub d1: i32,
    pub ax: f32,
    pub ay: f32,
}

/// A gnomonic polar cap raster. The cap raster -> ray -> source-pixel mapping is a pure
/// function of `(r, tan_t, sgn)` and the source plane's geometry, never of the frame's
/// content, so it is built once (`cap_build_map`). The production path renders from the
/// already-decimated f=4 plane (a 4x4 box mean, so a cap pixel inherits sigma/4 of the
/// sensor noise).
#[derive(Debug, Clone, Default)]
pub struct PolarCap {
    /// cap raster side
    pub r: usize,
    /// tan of the half-FOV
    pub tan_t: f64,
    /// +1 north, -1 south
    pub sgn: f64,
    /// `r * r` gray
    pub img: Vec<u8>,
    /// `r`-entry gnomonic coordinate ramp (the i- and j-axis mapping is the same function
    /// of index)
    pub ramp: Vec<f64>,
    /// `r * r` taps; empty until `cap_build_map`
    pub tap: Vec<Tap>,
    pub map_w: i32,
    pub map_h: i32,
    pub map_stride: usize,
}

impl PolarCap {
    /// `r = 2*tan(half-FOV) * (h/pi) / f` -- the same angular sampling density as the
    /// f-scale band -- rounded UP to a multiple of `K_TILE` so the cap's tile grid covers
    /// it exactly.
    pub fn new(h: i32, f: i32, sgn: f64) -> PolarCap {
        let tan_t = (K_CAP_HALF_FOV_DEG * K_PI / 180.0).tan();
        let raw_r = (2.0 * tan_t * (h as f64 / K_PI) / f as f64).round() as usize;
        let r = raw_r.div_ceil(K_TILE) * K_TILE;
        let ramp = (0..r)
            .map(|k| ((k as f64 + 0.5) / r as f64 * 2.0 - 1.0) * tan_t)
            .collect();
        PolarCap {
            r,
            tan_t,
            sgn,
            img: vec![0; r * r],
            ramp,
            ..Default::default()
        }
    }
}

/// Builds `c.tap` for a source plane of `src_w` x `src_h` pixels at row stride
/// `src_stride`. `src_stride` must be at least `src_w + 1` and the plane's column `src_w`
/// must hold a copy of its column 0. The tap/weight derivation is `sample_gray_plane`'s
/// own, evaluated once: longitude wraps, latitude REFLECTS over the pole with a half-turn
/// shift.
pub fn cap_build_map(c: &mut PolarCap, src_w: i32, src_h: i32, src_stride: usize) {
    c.map_w = src_w;
    c.map_h = src_h;
    c.map_stride = src_stride;
    let r = c.r;
    c.tap.clear();
    c.tap.resize(r * r, Tap::default());
    let (w, h, hw) = (src_w, src_h, src_w / 2);
    let st = src_stride as i64;
    for j in 0..r {
        let gz = c.ramp[j];
        for i in 0..r {
            let gx = c.ramp[i];
            let d = normalize(&Vec3::new(gx, c.sgn, gz));
            let (pxd, pyd) = ray_to_px(&d, w, h);
            let (mut x, mut y) = (pxd as f32, pyd as f32);
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
            let x0 = x as i32;
            let ax = x - x0 as f32;
            let mut y0 = y.floor() as i32;
            let ay = y - y0 as f32;
            let mut y1 = y0 + 1;
            // the companion column of each row pair is the NEXT one: x0 + 1 is valid even
            // at x0 == w - 1 because the plane carries a copy of column 0 at column w
            let (mut cx0, mut dx0) = (x0, x0);
            if y0 < 0 {
                y0 = 0;
                cx0 += hw;
                if cx0 >= w {
                    cx0 -= w;
                }
            }
            if y1 > h - 1 {
                y1 = h - 1;
                dx0 += hw;
                if dx0 >= w {
                    dx0 -= w;
                }
            }
            let o0 = y0 as i64 * st + cx0 as i64;
            let o1 = y1 as i64 * st + dx0 as i64;
            c.tap[j * r + i] = Tap {
                o0: o0 as i32,
                d1: (o1 - o0) as i32,
                ax,
                ay,
            };
        }
    }
}

/// Samples the FULL-RESOLUTION view `v` into `c.img`. Longitude wraps, latitude REFLECTS
/// over the pole (`sample`'s own sphere topology). Generic (any format, any gamma LUT) and
/// map-free; the production path is `cap_render_plane`.
pub fn cap_render(v: &View, c: &mut PolarCap) {
    let r = c.r;
    for j in 0..r {
        let gz = c.ramp[j];
        let row = &mut c.img[j * r..(j + 1) * r];
        for (i, px_out) in row.iter_mut().enumerate() {
            let gx = c.ramp[i];
            let d = normalize(&Vec3::new(gx, c.sgn, gz));
            let (px, py) = ray_to_px(&d, v.w, v.h);
            *px_out = (sample(v, px as f32, py as f32) + 0.5) as u8;
        }
    }
}

/// Production path: render the cap from the already-decimated GRAY plane using the
/// precomputed map. `p` must be the plane `cap_build_map` was called for (a mismatch
/// leaves the image untouched).
pub fn cap_render_plane(p: &[u8], w: i32, h: i32, stride: usize, c: &mut PolarCap) {
    if c.tap.is_empty() || c.map_w != w || c.map_h != h || c.map_stride != stride {
        return;
    }
    for (out, t) in c.img.iter_mut().zip(&c.tap) {
        // each row's two taps are ADJACENT bytes
        let o0 = t.o0 as usize;
        let o1 = (t.o0 as i64 + t.d1 as i64) as usize;
        let q0 = &p[o0..o0 + 2];
        let q1 = &p[o1..o1 + 2];
        let (v00, v01) = (q0[0] as f32, q0[1] as f32);
        let (v10, v11) = (q1[0] as f32, q1[1] as f32);
        let a = fmadd(v01 - v00, t.ax, v00);
        let b = fmadd(v11 - v10, t.ax, v10);
        *out = (fmadd(b - a, t.ay, a) + 0.5) as u8;
    }
}

/// The same gnomonic cap-raster -> ray mapping `cap_render` samples at, without the
/// sampling. `cx`, `cy` may be fractional (e.g. a blob's own centroid).
pub fn cap_to_ray(c: &PolarCap, cx: f64, cy: f64) -> Vec3 {
    let gx = ((cx + 0.5) / c.r as f64 * 2.0 - 1.0) * c.tan_t;
    let gz = ((cy + 0.5) / c.r as f64 * 2.0 - 1.0) * c.tan_t;
    normalize(&Vec3::new(gx, c.sgn, gz))
}

// ---- candidate assembly -----------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct Candidate {
    /// centre ray
    pub v: Vec3,
    /// angular radius, rad
    pub r: f64,
    /// blob-quad corner rays (the bridge's preferred seed)
    pub quad: [Vec3; 4],
    pub has_quad: bool,
    pub from_cap: bool,
    /// 4 or 2
    pub scale: u8,
    /// 1 = reference-strict shape, 2 = relaxed
    pub tier: u8,
    /// the shape measurements the tier decision was made from
    pub solidity: f32,
    pub extent: f32,
    pub raw_extent: f32,
    /// `dark_ring_dn`: quiet-zone minus black-border DN, or 0 when the probe was not run
    pub ring_dn: f32,
}

/// De-stretch: an equirect bbox is 1/cos(lat) too wide in longitude, so its longitude
/// extent is de-stretched by `clat` before taking the larger of the two extents. `bw`,
/// `bh` are PLANE-scale (pre-f) blob sizes; `ry` is the candidate centre ray's y component
/// (== sin(lat)). `clat` is floored at 1e-3 so a pole-adjacent radius stays finite.
pub fn destretch_radius(h: i32, f: i32, bw: f64, bh: f64, ry: f64) -> f64 {
    let clat = (1.0 - ry * ry).max(0.0).sqrt().max(1e-3);
    0.5 * bh.max(bw * clat) * f as f64 / (h as f64 / K_PI)
}

/// Wall-clock ms per proposal sub-stage plus the blob/candidate counts each produced,
/// filled by every `run`/`run_small` call.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProposalDebug {
    pub t_downsample: f64,
    pub t_tiles: f64,
    pub t_thresh: f64,
    pub t_ccl: f64,
    pub t_shape: f64,
    /// The second threshold offset's own share of `t_thresh`/`t_ccl`/`t_shape`.
    pub t_thresh_2c: f64,
    pub t_ccl_2c: f64,
    pub t_shape_2c: f64,
    pub t_cap_render: f64,
    pub t_cap_thresh: f64,
    pub t_cap_ccl: f64,
    pub t_cap_shape: f64,
    pub t_small: f64,
    pub t_total: f64,
    pub n_roots_c: i32,
    pub n_roots_2c: i32,
    pub n_roots_cap: i32,
    pub n_roots_small: i32,
    pub n_cand_c: i32,
    pub n_cand_2c: i32,
    pub n_cand_cap: i32,
    pub n_cand_small: i32,
    pub n_tier1: i32,
    pub n_tier2: i32,
    pub n_shape_calls: i32,
    pub n_shape_floods: i32,
    pub n_ring_probes: i32,
    /// summed bbox area of every flooded blob
    pub n_shape_flood_px: i64,
    pub ran_small: bool,
    /// did the second threshold offset's pass run?
    pub ran_2c: bool,
}

#[inline]
fn pms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

/// f=2 box mean that reads only every other SOURCE row (point-sampled in y, box-averaged
/// in x) -- `run_small`'s row-skip decimation. Gray8 only.
fn downsample_gray_f2_rowskip(
    src: &[u8],
    w: usize,
    h: usize,
    sstride: usize,
    dst: &mut [u8],
    dstride: usize,
) {
    let (ow, oh) = (w / 2, h / 2);
    for oy in 0..oh {
        let row = &src[2 * oy * sstride..][..2 * ow];
        let d = &mut dst[oy * dstride..][..ow];
        for (o, px) in d.iter_mut().zip(row.chunks_exact(2)) {
            *o = ((px[0] as i32 + px[1] as i32 + 1) >> 1) as u8;
        }
    }
}

/// Shared per-blob candidate assembly for every threshold/CCL pass: shape filter, an
/// optional early size gate, the blob-quad + centre -> ray conversion (`to_ray`), the
/// angular radius (`radius_fn`), and the dark-ring probe. Returns `(roots, candidates)`
/// counted in this pass.
#[allow(clippy::too_many_arguments)]
fn collect_candidates(
    pw: usize,
    ph: usize,
    ccl: &mut RunCcl,
    scratch: &mut ShapeScratch,
    cfg: &ShapeCfg,
    scale: u8,
    from_cap: bool,
    max_side_gate: f64,
    full_v: &View,
    to_ray: impl Fn(f64, f64) -> Vec3,
    radius_fn: impl Fn(f64, f64, &Vec3) -> f64,
    out: &mut Vec<Candidate>,
    dbg: &mut ProposalDebug,
) -> (i32, i32) {
    let (mut n_roots, mut n_cand) = (0, 0);
    for i in 1..ccl.blobs.len() {
        if ccl.find(i as i32) != i as i32 {
            continue; // non-root: already merged
        }
        n_roots += 1;
        let b0 = ccl.blobs[i];
        let bw = (b0.xmax - b0.xmin + 1) as f64;
        let bh = (b0.ymax - b0.ymin + 1) as f64;
        if max_side_gate > 0.0 && (bw > max_side_gate || bh > max_side_gate) {
            continue;
        }
        dbg.n_shape_calls += 1;
        let mut si = ShapeInfo::default();
        let ok = shape_ok(pw, ph, ccl, i as i32, cfg, scratch, Some(&mut si));
        // si is only written once shape_ok has actually built the mask, so a non-zero
        // extent is exactly "this blob paid for the mask + hole flood"
        if si.extent > 0.0 {
            dbg.n_shape_floods += 1;
            dbg.n_shape_flood_px += (b0.xmax - b0.xmin + 1) as i64 * (b0.ymax - b0.ymin + 1) as i64;
        }
        if !ok {
            continue;
        }
        let b = ccl.blobs[i]; // has_hole is now set

        let mut cnd = Candidate {
            has_quad: true,
            from_cap,
            scale,
            tier: 2,
            solidity: si.solidity,
            extent: si.extent,
            raw_extent: si.raw_extent,
            ..Default::default()
        };

        let q = blob_quad(&b);
        for k in 0..4 {
            cnd.quad[k] = to_ray(q[k][0] as f64, q[k][1] as f64);
        }

        let cx = (b.xmin as f64 + b.xmax as f64) * 0.5;
        let cy = (b.ymin as f64 + b.ymax as f64) * 0.5;
        cnd.v = to_ray(cx, cy);
        cnd.r = radius_fn(bw, bh, &cnd.v);

        // ---- dark-ring probe + tiering ------------------------------------------
        // The probe is mandatory for a hole-less blob (it keeps a solid dark rectangle
        // from counting against the budget). It also runs on a holed blob that is
        // otherwise reference-strict, because "passes the dark-ring probe" is part of the
        // tier-1 definition; failing it only demotes such a blob to tier 2.
        let strict_shape = si.has_hole
            && si.solidity >= cfg.tier1_solidity_min
            && si.extent >= cfg.tier1_extent_min
            && si.raw_extent >= cfg.tier1_raw_extent_min
            && si.raw_extent <= cfg.tier1_raw_extent_max;
        if b.has_hole == 0 || strict_shape {
            let mut rq = [[0f32; 3]; 4];
            for k in 0..4 {
                rq[k] = [
                    cnd.quad[k].x as f32,
                    cnd.quad[k].y as f32,
                    cnd.quad[k].z as f32,
                ];
            }
            dbg.n_ring_probes += 1;
            let dn = dark_ring_dn(full_v, &rq);
            cnd.ring_dn = dn as f32;
            let ring = dn >= cfg.ring_dn_min as f64;
            if b.has_hole == 0 && !ring {
                continue; // unchanged rejection
            }
            if strict_shape && ring {
                cnd.tier = 1;
            }
        }
        if cnd.tier == 1 {
            dbg.n_tier1 += 1;
        } else {
            dbg.n_tier2 += 1;
        }
        n_cand += 1;
        out.push(cnd);
    }
    (n_roots, n_cand)
}

/// f=4 whole-equirect band (both `K_THRESH_C` and `K_THRESH_C2`) + the two polar caps,
/// plus an optional f=2 pass for small markers. All working buffers are sized once from
/// `W`, `H`; nothing is allocated per frame once capacities have grown.
pub struct Proposal {
    w: i32,
    h: i32,
    wb: usize,
    hb: usize,
    wb2: usize,
    hb2: usize,
    /// `wb + 1`: the seam-duplicate column
    plane4_stride: usize,
    plane4: Vec<u8>,
    plane2: Vec<u8>,
    g4: TileGrid,
    g2: TileGrid,
    gcap: TileGrid,
    bin4: Vec<u8>,
    bin2: Vec<u8>,
    bincap: Vec<u8>,
    cap_n: PolarCap,
    cap_s: PolarCap,
    ccl: RunCcl,
    shape_scratch: ShapeScratch,
    cfg: ShapeCfg,
    dbg: ProposalDebug,
}

impl Proposal {
    /// `w` x `h` is the full equirect size (`w == 2h`, `w` a multiple of 4 -- checked by
    /// the detector); `Err(Size)` when the f=4 plane is too small to hold one tile.
    pub fn new(w: i32, h: i32) -> Result<Proposal, EqarError> {
        if w <= 0 || h <= 0 {
            return Err(EqarError::Size);
        }
        let (wb, hb) = (w as usize / 4, h as usize / 4);
        let (wb2, hb2) = (w as usize / 2, h as usize / 2);
        if wb < K_TILE || hb < K_TILE {
            return Err(EqarError::Size);
        }
        // The f=4 plane carries ONE extra column, holding a copy of column 0, so a cap
        // tap's right-hand neighbour is always the next byte even at the seam.
        let plane4_stride = wb + 1;
        let mut cap_n = PolarCap::new(h, 4, 1.0);
        let mut cap_s = PolarCap::new(h, 4, -1.0);
        // the caps render off the SAME f=4 plane the band uses
        cap_build_map(&mut cap_n, wb as i32, hb as i32, plane4_stride);
        cap_build_map(&mut cap_s, wb as i32, hb as i32, plane4_stride);
        let rcap = cap_n.r;
        Ok(Proposal {
            w,
            h,
            wb,
            hb,
            wb2,
            hb2,
            plane4_stride,
            plane4: vec![0; plane4_stride * hb],
            plane2: vec![0; wb2 * hb2],
            g4: TileGrid::new(wb / K_TILE, hb / K_TILE),
            g2: TileGrid::new(wb2 / K_TILE, hb2 / K_TILE),
            gcap: TileGrid::new(rcap / K_TILE, rcap / K_TILE),
            bin4: vec![0; wb * hb],
            bin2: vec![0; wb2 * hb2],
            bincap: vec![0; rcap * rcap],
            cap_n,
            cap_s,
            ccl: RunCcl::default(),
            shape_scratch: ShapeScratch::default(),
            cfg: ShapeCfg::default(),
            dbg: ProposalDebug::default(),
        })
    }

    pub fn dbg(&self) -> &ProposalDebug {
        &self.dbg
    }

    fn check_view(&self, v: &View) -> Result<(), EqarError> {
        v.check()?;
        if v.w != self.w || v.h != self.h {
            return Err(EqarError::Size);
        }
        Ok(())
    }

    /// f=4 band (two threshold offsets) + the two polar caps. Appends to `out`.
    pub fn run(&mut self, v: &View, out: &mut Vec<Candidate>) -> Result<(), EqarError> {
        self.check_view(v)?;
        self.dbg = ProposalDebug::default();
        let t_all = Instant::now();
        let (wb, hb, st) = (self.wb, self.hb, self.plane4_stride);
        let (w_full, h_full) = (self.w, self.h);

        let mut t0 = Instant::now();
        downsample(v, 4, &mut self.plane4, st);
        // the seam-duplicate column the cap map addresses
        for row in self.plane4.chunks_exact_mut(st) {
            row[wb] = row[0];
        }
        self.dbg.t_downsample += pms(t0);

        t0 = Instant::now();
        tile_sums(&self.plane4, wb, hb, st, &mut self.g4);
        tile_box(&mut self.g4);
        self.dbg.t_tiles += pms(t0);

        let to_ray_band = |px: f64, py: f64| -> Vec3 {
            px_to_ray(px * 4.0 + 1.5, py * 4.0 + 1.5, w_full, h_full)
        };
        let radius_band =
            |bw: f64, bh: f64, cv: &Vec3| -> f64 { destretch_radius(h_full, 4, bw, bh, cv.y) };

        t0 = Instant::now();
        threshold_from_tiles(
            &self.plane4,
            wb,
            hb,
            st,
            &mut self.g4,
            K_THRESH_C,
            &mut self.bin4,
            wb,
        );
        self.dbg.t_thresh += pms(t0);
        t0 = Instant::now();
        run_ccl_wrap(&self.bin4, wb, hb, wb, &mut self.ccl);
        self.dbg.t_ccl += pms(t0);
        t0 = Instant::now();
        let (nr, nc) = collect_candidates(
            wb,
            hb,
            &mut self.ccl,
            &mut self.shape_scratch,
            &self.cfg,
            4,
            false,
            0.0,
            v,
            to_ray_band,
            radius_band,
            out,
            &mut self.dbg,
        );
        self.dbg.n_roots_c += nr;
        self.dbg.n_cand_c += nc;
        self.dbg.t_shape += pms(t0);

        // The second offset is a FALLBACK, not a second opinion: when the primary offset
        // has already produced a candidate at reference-grade shape (tier 1) a second,
        // stricter binarisation has nothing left to contribute that the budget would
        // reach. `n_tier1` is exactly the primary offset's tier-1 count here.
        if self.dbg.n_tier1 == 0 {
            self.dbg.ran_2c = true;
            t0 = Instant::now();
            threshold_from_tiles(
                &self.plane4,
                wb,
                hb,
                st,
                &mut self.g4,
                K_THRESH_C2,
                &mut self.bin4,
                wb,
            );
            self.dbg.t_thresh_2c = pms(t0);
            self.dbg.t_thresh += self.dbg.t_thresh_2c;
            t0 = Instant::now();
            run_ccl_wrap(&self.bin4, wb, hb, wb, &mut self.ccl);
            self.dbg.t_ccl_2c = pms(t0);
            self.dbg.t_ccl += self.dbg.t_ccl_2c;
            t0 = Instant::now();
            let (nr, nc) = collect_candidates(
                wb,
                hb,
                &mut self.ccl,
                &mut self.shape_scratch,
                &self.cfg,
                4,
                false,
                0.0,
                v,
                to_ray_band,
                radius_band,
                out,
                &mut self.dbg,
            );
            self.dbg.n_roots_2c += nr;
            self.dbg.n_cand_2c += nc;
            self.dbg.t_shape_2c = pms(t0);
            self.dbg.t_shape += self.dbg.t_shape_2c;
        }

        for north in [true, false] {
            let cap = if north {
                &mut self.cap_n
            } else {
                &mut self.cap_s
            };
            t0 = Instant::now();
            cap_render_plane(&self.plane4, wb as i32, hb as i32, st, cap);
            self.dbg.t_cap_render += pms(t0);
            t0 = Instant::now();
            let rc = cap.r;
            tile_sums(&cap.img, rc, rc, rc, &mut self.gcap);
            tile_box(&mut self.gcap);
            threshold_from_tiles(
                &cap.img,
                rc,
                rc,
                rc,
                &mut self.gcap,
                K_THRESH_C,
                &mut self.bincap,
                rc,
            );
            self.dbg.t_cap_thresh += pms(t0);
            t0 = Instant::now();
            run_ccl_wrap(&self.bincap, rc, rc, rc, &mut self.ccl);
            self.dbg.t_cap_ccl += pms(t0);

            let cap_ref: &PolarCap = cap;
            let to_ray_cap = |px: f64, py: f64| -> Vec3 { cap_to_ray(cap_ref, px, py) };
            let radius_cap = |bw: f64, bh: f64, _cv: &Vec3| -> f64 {
                (0.5 * bw.max(bh) / cap_ref.r as f64 * 2.0 * cap_ref.tan_t).atan()
            };
            t0 = Instant::now();
            let (nr, nc) = collect_candidates(
                rc,
                rc,
                &mut self.ccl,
                &mut self.shape_scratch,
                &self.cfg,
                4,
                true,
                0.0,
                v,
                to_ray_cap,
                radius_cap,
                out,
                &mut self.dbg,
            );
            self.dbg.n_roots_cap += nr;
            self.dbg.n_cand_cap += nc;
            self.dbg.t_cap_shape += pms(t0);
        }
        self.dbg.t_total += pms(t_all);
        Ok(())
    }

    /// f=2 band only: row-skip decimation + an early size gate (blobs whose larger side
    /// exceeds 24 f=2 px belong to `run`) so this reaches markers down to ~14 px_eq. No
    /// caps. Costs ~4x `run`; the detector budget-gates calling it. Does not reset the
    /// debug counters.
    pub fn run_small(&mut self, v: &View, out: &mut Vec<Candidate>) -> Result<(), EqarError> {
        self.check_view(v)?;
        let t_all = Instant::now();
        self.dbg.ran_small = true;
        let (wb2, hb2) = (self.wb2, self.hb2);
        if v.fmt == Fmt::Gray8 {
            downsample_gray_f2_rowskip(
                v.data,
                v.w as usize,
                v.h as usize,
                v.stride,
                &mut self.plane2,
                wb2,
            );
        } else {
            downsample(v, 2, &mut self.plane2, wb2); // no fused BGR/RGB row-skip path
        }
        tile_sums(&self.plane2, wb2, hb2, wb2, &mut self.g2);
        tile_box(&mut self.g2);
        threshold_from_tiles(
            &self.plane2,
            wb2,
            hb2,
            wb2,
            &mut self.g2,
            K_THRESH_C,
            &mut self.bin2,
            wb2,
        );
        run_ccl_wrap(&self.bin2, wb2, hb2, wb2, &mut self.ccl);

        let (w_full, h_full) = (self.w, self.h);
        let to_ray_small = |px: f64, py: f64| -> Vec3 {
            px_to_ray(px * 2.0 + 0.5, py * 2.0 + 0.5, w_full, h_full)
        };
        let radius_small =
            |bw: f64, bh: f64, cv: &Vec3| -> f64 { destretch_radius(h_full, 2, bw, bh, cv.y) };
        // Early size gate: a blob whose larger side exceeds 24 f=2 px (~48 px_eq)
        // belongs to the f=4 pass, not here.
        let (nr, nc) = collect_candidates(
            wb2,
            hb2,
            &mut self.ccl,
            &mut self.shape_scratch,
            &self.cfg,
            2,
            false,
            24.0,
            v,
            to_ray_small,
            radius_small,
            out,
            &mut self.dbg,
        );
        self.dbg.n_roots_small += nr;
        self.dbg.n_cand_small += nc;
        self.dbg.t_small += pms(t_all);
        self.dbg.t_total += self.dbg.t_small;
        Ok(())
    }
}

// ---- bucketed dedupe and budget order -----------------------------------------------

/// 16x16x16 grid over the unit cube: each axis in [0,15]. A unit-vector component sits in
/// [-1,1]; +1 EXACTLY maps to 16, one past the last bucket, so the top edge is clamped.
#[inline]
fn dedupe_bucket_axis(v: f64) -> i32 {
    ((v + 1.0) * 8.0) as i32
}

fn dsu_find(parent: &mut [i32], mut a: i32) -> i32 {
    while parent[a as usize] != a {
        let g = parent[parent[a as usize] as usize];
        parent[a as usize] = g;
        a = g;
    }
    a
}

fn dsu_union(parent: &mut [i32], a: i32, b: i32) {
    let (a, b) = (dsu_find(parent, a), dsu_find(parent, b));
    if a != b {
        parent[b as usize] = a;
    }
}

/// Merges ONLY true duplicates -- never a neighbour, never a nested ring/marker pair. Two
/// candidates merge iff BOTH `angle(v1, v2) < 0.25 * min(r1, r2)` (essentially co-centred)
/// and `max(r1, r2) <= 1.15 * min(r1, r2)` (essentially the same size). Candidates are
/// indexed into a 16x16x16 grid over the unit cube and compared only against the 27
/// neighbouring cells, so this is O(K). The best member of each merged group survives: the
/// strictest tier first, then the smaller radius.
pub fn dedupe_candidates(c: &mut Vec<Candidate>) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let clampk = |k: i32| k.clamp(0, 15);
    let mut key = vec![0i32; n];
    let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); 16 * 16 * 16];
    for i in 0..n {
        let bx = clampk(dedupe_bucket_axis(c[i].v.x));
        let by = clampk(dedupe_bucket_axis(c[i].v.y));
        let bz = clampk(dedupe_bucket_axis(c[i].v.z));
        key[i] = bx | (by << 4) | (bz << 8);
        buckets[key[i] as usize].push(i as i32);
    }

    let mut parent: Vec<i32> = (0..n as i32).collect();
    for i in 0..n {
        let (bx, by, bz) = (key[i] & 0xF, (key[i] >> 4) & 0xF, (key[i] >> 8) & 0xF);
        for dx in -1..=1 {
            let nx = bx + dx;
            if !(0..=15).contains(&nx) {
                continue;
            }
            for dy in -1..=1 {
                let ny = by + dy;
                if !(0..=15).contains(&ny) {
                    continue;
                }
                for dz in -1..=1 {
                    let nz = bz + dz;
                    if !(0..=15).contains(&nz) {
                        continue;
                    }
                    let nk = nx | (ny << 4) | (nz << 8);
                    for &j in &buckets[nk as usize] {
                        if j <= i as i32 {
                            continue; // each unordered pair once
                        }
                        let (r1, r2) = (c[i].r, c[j as usize].r);
                        let rmin = r1.min(r2);
                        let rmax = r1.max(r2);
                        let d = dot(&c[i].v, &c[j as usize].v);
                        let d = d.clamp(-1.0, 1.0);
                        let ang = d.acos();
                        if ang < 0.25 * rmin && rmax <= 1.15 * rmin {
                            dsu_union(&mut parent, i as i32, j);
                        }
                    }
                }
            }
        }
    }

    // Keep the best member of each merged group: the strictest TIER first (the C and 2C
    // passes routinely produce the same marker twice, once reference-strict and once only
    // after the relaxations), then, among equal tiers, the smaller radius. Note this
    // `better` is not a transitive order across groups; it is only ever applied within one
    // group, in index order, exactly as the original does.
    let better = |a: &Candidate, b: &Candidate| -> bool {
        if a.tier != b.tier {
            return a.tier < b.tier;
        }
        a.r < b.r
    };
    let mut best = vec![-1i32; n];
    for i in 0..n {
        let root = dsu_find(&mut parent, i as i32) as usize;
        if best[root] < 0 || better(&c[i], &c[best[root] as usize]) {
            best[root] = i as i32;
        }
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let root = dsu_find(&mut parent, i as i32) as usize;
        if best[root] == i as i32 {
            out.push(c[i]);
        }
    }
    *c = out;
}

#[inline]
fn candidate_hole_fraction(c: &Candidate) -> f32 {
    if c.extent > 0.0 {
        (c.extent - c.raw_extent) / c.extent
    } else {
        0.0
    }
}

/// Sorts by (tier, then ASCENDING angular radius) so every tier-1 candidate is offered to
/// the budget before any tier-2 one and a small marker's cheap patch is never starved.
/// WITHIN tier 1 the keys are, in order: (1) does the enclosed white fraction look like a
/// coded tag's data modules (`K_TIER1_HOLE_FRAC_LO..=HI`), then (2) DESCENDING dark-ring
/// contrast; both are physical, rotation- and scale-invariant properties of a marker.
/// Stable. Comparisons use `total_cmp`; NaN radii or ring values do not occur.
pub fn order_for_budget(c: &mut [Candidate]) {
    c.sort_by(|a, b| {
        a.tier.cmp(&b.tier).then_with(|| {
            if a.tier == 1 {
                let in_band = |x: &Candidate| {
                    let f = candidate_hole_fraction(x);
                    (K_TIER1_HOLE_FRAC_LO..=K_TIER1_HOLE_FRAC_HI).contains(&f)
                };
                let (ba, bb) = (!in_band(a), !in_band(b));
                if ba != bb {
                    return ba.cmp(&bb);
                }
                if a.ring_dn != b.ring_dn {
                    return b.ring_dn.total_cmp(&a.ring_dn);
                }
            }
            a.r.total_cmp(&b.r)
        })
    });
}

/// Hard per-frame candidate cap, applied AFTER `order_for_budget`: keeps the first `cap`.
pub fn cap_candidates(c: &mut Vec<Candidate>, cap: usize) {
    if c.len() > cap {
        c.truncate(cap);
    }
}

#[cfg(all(test, target_arch = "aarch64", not(feature = "no-neon")))]
mod tests {
    use super::*;

    struct Lcg(u32);
    impl Lcg {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            (self.0 >> 24) as u8
        }
    }

    #[test]
    fn neon_kernels_match_scalar() {
        let mut r = Lcg(2024);
        // sparse bytes so runs are long enough to cross block edges
        let s: Vec<u8> = (0..4096)
            .map(|_| if r.byte() < 40 { r.byte() | 1 } else { 0 })
            .collect();
        for x in (0..s.len() - 16).step_by(7) {
            let a: &[u8; 16] = s[x..x + 16].try_into().unwrap();
            assert_eq!(crate::simd::nibble_mask16(a), block_mask_scalar(&s, x, 16));
        }
        // row extraction: NEON mask vs scalar mask give identical runs
        for w in [16usize, 17, 31, 61, 64, 253] {
            let row = &s[100..100 + w];
            let (mut a, mut b) = (Vec::new(), Vec::new());
            extract_runs_with(row, w, 3, &mut a, block_mask);
            extract_runs_with(row, w, 3, &mut b, block_mask_scalar);
            assert_eq!(a, b, "w={w}");
        }
        // threshold compare and tile sums
        let thr: Vec<u8> = (0..64).map(|_| r.byte()).collect();
        let src: Vec<u8> = (0..64).map(|_| r.byte()).collect();
        let mut d = vec![0u8; 64];
        let done = crate::simd::cmp_lt_row(&src, &thr, &mut d);
        assert_eq!(done, 64);
        for i in 0..64 {
            assert_eq!(d[i], if src[i] < thr[i] { 255 } else { 0 });
        }
        let row: Vec<u8> = (0..40).map(|_| r.byte()).collect();
        let mut sn = vec![5u32; 10];
        let done = crate::simd::tile_sums_row(&row, &mut sn);
        assert_eq!(done, 8);
        for t in 0..8 {
            let e: u32 = 5 + row[4 * t..4 * t + 4].iter().map(|&b| b as u32).sum::<u32>();
            assert_eq!(sn[t], e);
        }
        assert_eq!(sn[8], 5);
    }
}
