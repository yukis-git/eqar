//! The public, validated API: [`Detector`], [`ImageView`], [`Marker`], [`Prior`],
//! [`generate_marker`].
//!
//! Every entry point checks its arguments and returns an [`EqarError`] instead of
//! panicking. Validation order for an image call: size, then stride / buffer length, then
//! output capacity ([`Params::max_markers`]).

use crate::decode::dict_info;
use crate::detector::{self, Detection};
use crate::error::EqarError;
use crate::geometry::{norm, normalize, Vec3};
use crate::image::View;
use crate::params::{Dict, Params};

pub use crate::detector::Stats;
pub use crate::image::Fmt as PixelFormat;

/// Largest supported image width (height is half of it).
pub const MAX_WIDTH: usize = 16384;
/// Smallest supported image width. Below it the polar-cap planes degenerate; such frames
/// are far too coarse to read a marker from anyway.
pub const MIN_WIDTH: usize = 320;

/// A borrowed equirectangular image: `width == 2 * height`, `width` a multiple of 4.
///
/// Row `y` starts at byte `y * stride`; `stride` is at least `width` times the bytes per
/// pixel of `format` (1 for [`PixelFormat::Gray8`], 3 for the colour formats) and `data`
/// must hold every row (`(height - 1) * stride + width * bytes_per_pixel` bytes).
#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a> {
    /// Pixel bytes, borrowed for the duration of the call.
    pub data: &'a [u8],
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Bytes per row.
    pub stride: usize,
    /// Pixel layout.
    pub format: PixelFormat,
}

impl<'a> ImageView<'a> {
    /// A tightly packed image (`stride = width * bytes_per_pixel`).
    pub fn packed(data: &'a [u8], width: usize, height: usize, format: PixelFormat) -> Self {
        ImageView {
            data,
            width,
            height,
            stride: width.saturating_mul(format.bytes_per_pixel()),
            format,
        }
    }
}

/// Provenance of the quad a marker was refined from: the polished 4-gon.
pub const PROV_LEGACY: u8 = 0;
/// Provenance: the 4-line fit.
pub const PROV_LINEFIT: u8 = 1;
/// Provenance: the 4-line fit of a contour the strict selection rejects (known id only).
pub const PROV_RELAXED: u8 = 2;

/// One detected marker. Corner order is TL, TR, BR, BL in the marker's own frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Marker {
    /// Full-resolution equirectangular pixel position of each corner (subpixel).
    pub corner_px: [[f32; 2]; 4],
    /// Unit ray of each corner.
    pub ray: [[f32; 3]; 4],
    /// Pixel position of the marker centre (the projective centre of the quad).
    pub center_px: [f32; 2],
    /// Unit ray of the marker centre.
    pub center_ray: [f32; 3],
    /// Dictionary id.
    pub id: i32,
    /// `codes::FLAG_*` bits.
    pub flags: u32,
    /// 0..1, higher is better.
    pub quality: f32,
    /// Mean edge arc length in equivalent pixels.
    pub side_px: f32,
    /// Mean normalised edge residual, equivalent pixels.
    pub edge_resid_px: f32,
    /// Mean edge-profile amplitude in DN.
    pub contrast_dn: f32,
    /// Min `|n_{i-1} x n_i|` of the fitted edge normals.
    pub min_cross: f32,
    /// Always `codes::ACCEPT` on a returned marker.
    pub reject_code: i32,
    /// `true`: the corner is a measurement. `false`: tip hidden / completed /
    /// inconsistent; do not feed it to an estimator.
    pub corner_valid: [bool; 4],
    /// `guards::K_CORNER_*` bits.
    pub corner_flags: [u8; 4],
    /// >= 1; multiplies the corner's sigma.
    pub corner_sigma_scale: [f32; 4],
    /// `PROV_*`.
    pub prov: u8,
    /// Hamming distance of the accepted code over the known bits.
    pub n_bit_errors: i16,
    /// Erased or not-black border cells.
    pub border_fail: i16,
    /// Erased interior cells.
    pub n_erased: i16,
}

impl Marker {
    fn from_detection(d: &Detection) -> Marker {
        let mut m = Marker::default();
        for i in 0..4 {
            m.corner_px[i] = [d.corner_px[i][0] as f32, d.corner_px[i][1] as f32];
            m.ray[i] = [d.ray[i].x as f32, d.ray[i].y as f32, d.ray[i].z as f32];
            m.corner_valid[i] = d.corner_valid[i] != 0;
            m.corner_flags[i] = d.corner_flags[i];
            m.corner_sigma_scale[i] = d.sigma_scale[i];
        }
        m.center_px = [d.center_px[0] as f32, d.center_px[1] as f32];
        m.center_ray = [
            d.center_ray.x as f32,
            d.center_ray.y as f32,
            d.center_ray.z as f32,
        ];
        m.prov = d.prov;
        m.n_bit_errors = d.n_bit_errors as i16;
        m.border_fail = d.border_fail as i16;
        m.n_erased = d.n_erased as i16;
        m.id = d.id;
        m.flags = d.flags;
        m.quality = d.quality;
        m.side_px = d.side_px;
        m.edge_resid_px = d.edge_resid_px;
        m.contrast_dn = d.contrast_dn;
        m.min_cross = d.min_cross;
        m.reject_code = d.reject_code;
        m
    }
}

/// A known marker and where it is expected, for [`Detector::verify`]: its id and the unit
/// rays of its corners in the marker's own TL, TR, BR, BL order (the order of
/// [`Marker::ray`]) with an angular 1-sigma uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prior {
    /// Dictionary id; must be in the dictionary.
    pub id: i32,
    /// Expected corner rays; each must have a norm in `[0.5, 1.5]` (it is normalised).
    pub ray: [[f32; 3]; 4],
    /// Angular 1-sigma in degrees, in `(0, 10]`.
    pub sigma_deg: f32,
}

/// An equirectangular ArUco detector bound to one image size and one [`Params`].
///
/// One thread per detector. Every method that reads an image takes an [`ImageView`] whose
/// size equals the detector's size.
pub struct Detector {
    core: detector::Detector,
    width: usize,
    height: usize,
    max_markers: usize,
    n_ids: i32,
    lut: Option<Box<[u8; 256]>>,
}

/// Validates an image call (size, then stride / buffer length, then capacity) and builds
/// the internal view.
fn make_view<'a>(
    width: usize,
    height: usize,
    max_markers: usize,
    lut: &'a Option<Box<[u8; 256]>>,
    img: &ImageView<'a>,
) -> Result<View<'a>, EqarError> {
    if img.width != width || img.height != height {
        return Err(EqarError::Size);
    }
    let v = View {
        data: img.data,
        w: width as i32,
        h: height as i32,
        stride: img.stride,
        fmt: img.format,
        lut: lut.as_deref(),
    };
    if !v.is_valid() {
        return Err(EqarError::Stride);
    }
    if max_markers < 1 {
        return Err(EqarError::Cap);
    }
    Ok(v)
}

impl Detector {
    /// Creates a detector for `width` x `height` images.
    ///
    /// Errors: [`EqarError::Size`] unless `width == 2 * height`, `width % 4 == 0` and
    /// `MIN_WIDTH <= width <= MAX_WIDTH`; [`EqarError::Params`] for an in-use id outside the
    /// dictionary.
    pub fn new(width: usize, height: usize, params: &Params) -> Result<Detector, EqarError> {
        if !(MIN_WIDTH..=MAX_WIDTH).contains(&width)
            || height.checked_mul(2) != Some(width)
            || width % 4 != 0
        {
            return Err(EqarError::Size);
        }
        let core = detector::Detector::new(width as i32, height as i32, params)?;
        Ok(Detector {
            core,
            width,
            height,
            max_markers: params.max_markers,
            n_ids: dict_info(params.dict.index())?.n,
            lut: params.gamma_lut.map(Box::new),
        })
    }

    /// Full-frame detection: finds every marker of the dictionary in the image, within
    /// [`Params::time_budget_ms`] (counted after the proposal stage). At most
    /// [`Params::max_markers`] are returned.
    pub fn detect(&mut self, img: &ImageView) -> Result<Vec<Marker>, EqarError> {
        let v = make_view(self.width, self.height, self.max_markers, &self.lut, img)?;
        let dets = self.core.scan(&v, self.max_markers)?;
        Ok(dets.iter().map(Marker::from_detection).collect())
    }

    /// Follows markers found earlier. `prev` are the markers of the previous observation,
    /// `frame_gap` (>= 1) the number of frames since then. Each is re-seeded near its
    /// constant-angular-velocity prediction; markers that cannot be followed fall back to
    /// one full-frame scan per call, and (with [`Params::coast_max`]) are remembered for a
    /// few frames. Four or more markers use the lean preset for the call.
    ///
    /// Errors: [`EqarError::Params`] for `frame_gap == 0`; [`EqarError::NoInit`] when
    /// `prev` is empty and no coasting marker exists. `prev` entries with a non-finite ray
    /// or `side_px`, or a negative id, are skipped.
    pub fn track(
        &mut self,
        img: &ImageView,
        prev: &[Marker],
        frame_gap: u32,
    ) -> Result<Vec<Marker>, EqarError> {
        let v = make_view(self.width, self.height, self.max_markers, &self.lut, img)?;
        if frame_gap < 1 {
            return Err(EqarError::Params);
        }
        // No prior at all is an error, unless coasting markers exist (a marker occluded
        // in every frame since its last detection is re-acquired from its prediction).
        if prev.is_empty() && !self.core.has_coasting() {
            return Err(EqarError::NoInit);
        }
        let prev_d: Vec<Detection> = prev
            .iter()
            .map(|m| {
                let mut d = Detection::default();
                for k in 0..4 {
                    d.ray[k] =
                        Vec3::new(m.ray[k][0] as f64, m.ray[k][1] as f64, m.ray[k][2] as f64);
                    d.corner_px[k] = [m.corner_px[k][0] as f64, m.corner_px[k][1] as f64];
                }
                d.id = m.id;
                d.side_px = m.side_px;
                d
            })
            .collect();
        let gap = frame_gap.min(i32::MAX as u32) as i32;
        let dets = self.core.track(&v, &prev_d, self.max_markers, gap)?;
        Ok(dets.iter().map(Marker::from_detection).collect())
    }

    /// Known-ID detection: for each prior, looks near its predicted quad and verifies
    /// marker `id` there (erasure-aware decode against the id, partly hidden edges,
    /// completion). Returns the verified markers in prior order; a prior that does not
    /// verify yields nothing.
    ///
    /// Errors: [`EqarError::Params`] for an id outside the dictionary, a `sigma_deg`
    /// outside `(0, 10]` (NaN included) or a ray whose norm is not finite and in
    /// `[0.5, 1.5]`.
    pub fn verify(&mut self, img: &ImageView, priors: &[Prior]) -> Result<Vec<Marker>, EqarError> {
        let v = make_view(self.width, self.height, self.max_markers, &self.lut, img)?;
        if priors.is_empty() {
            return Ok(Vec::new());
        }
        let mut pri = Vec::with_capacity(priors.len());
        for p in priors {
            if p.id < 0 || p.id >= self.n_ids {
                return Err(EqarError::Params);
            }
            if !(p.sigma_deg > 0.0) || !(p.sigma_deg <= 10.0) {
                return Err(EqarError::Params);
            }
            let mut q = detector::Prior {
                id: p.id,
                sigma_rad: p.sigma_deg as f64 * (std::f64::consts::PI / 180.0),
                ..Default::default()
            };
            for k in 0..4 {
                let r = Vec3::new(p.ray[k][0] as f64, p.ray[k][1] as f64, p.ray[k][2] as f64);
                let len = norm(&r);
                if !len.is_finite() || len < 0.5 || len > 1.5 {
                    return Err(EqarError::Params);
                }
                q.ray[k] = normalize(&r);
            }
            pri.push(q);
        }
        let dets = self.core.verify(&v, &pri, self.max_markers)?;
        Ok(dets.iter().map(Marker::from_detection).collect())
    }

    /// Forgets all tracked and coasting markers.
    pub fn reset_track(&mut self) {
        self.core.reset_track();
    }

    /// Time spent in each stage of the last call, in ms: `[propose, bridge, decode,
    /// refine, gate, total]`.
    pub fn last_timings(&self) -> [f64; 6] {
        *self.core.timings()
    }

    /// Counters of the last `detect` / `track` / `verify` call.
    pub fn last_stats(&self) -> Stats {
        *self.core.stats()
    }

    /// Stage 1 alone (diagnostic): the candidate list as `[rx, ry, rz, angular_radius]`,
    /// best first.
    pub fn proposals(&mut self, img: &ImageView) -> Result<Vec<[f32; 4]>, EqarError> {
        let v = make_view(self.width, self.height, self.max_markers, &self.lut, img)?;
        self.core.proposals(&v)
    }
}

/// Renders marker `id` of `dict` as a grayscale image: `dict.cells()` cells (the code
/// grid plus the one-cell black border) per side, each `cell_px` pixels, black 0 and
/// white 255, no quiet zone. Returns `dict.cells() * cell_px` squared bytes, row-major.
///
/// Errors: [`EqarError::Params`] for an id outside the dictionary, `cell_px == 0` or a
/// side above [`MAX_WIDTH`].
pub fn generate_marker(dict: Dict, id: i32, cell_px: usize) -> Result<Vec<u8>, EqarError> {
    let info = dict_info(dict.index())?;
    if id < 0 || id >= info.n || cell_px == 0 {
        return Err(EqarError::Params);
    }
    let ms = info.marker_size as usize;
    let ncell = ms + 2;
    let side = ncell
        .checked_mul(cell_px)
        .filter(|&s| s <= MAX_WIDTH)
        .ok_or(EqarError::Params)?;
    let code = info.codes[id as usize * 4];
    let mut out = vec![0u8; side * side];
    for y in 0..side {
        let row = y / cell_px;
        for x in 0..side {
            let col = x / cell_px;
            if row == 0 || row == ncell - 1 || col == 0 || col == ncell - 1 {
                continue;
            }
            let (ir, ic) = (row - 1, col - 1);
            if (code >> (ms * ms - 1 - (ir * ms + ic))) & 1 != 0 {
                out[y * side + x] = 255;
            }
        }
    }
    Ok(out)
}
