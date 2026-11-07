//! Python bindings for the `eqar` equirectangular ArUco detector.
//!
//! The native module is `eqar._eqar`; the `eqar` package re-exports it.
//!
//! Threading: every method that reads an image releases the GIL while the detector runs.
//! The image buffer is held through a shared numpy borrow and a strong reference for the
//! whole call, so it cannot be freed or resized underneath the detector. Another Python
//! thread that *writes* into the same array during the call is a data race on the pixels
//! (the detections are then unspecified); do not do that. A detector serialises its own
//! calls with a mutex that is taken after the GIL is released, so two threads sharing a
//! detector queue up instead of deadlocking.

use std::str::FromStr;
use std::sync::Mutex;

use eqar::codes;
use eqar::params::{Dict, Params, Quality};
use eqar::{EqarError, ImageView, PixelFormat, Stats, MAX_WIDTH, MIN_WIDTH};
use numpy::{
    PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn convert_error(e: EqarError) -> PyErr {
    match e {
        EqarError::NoInit => PyRuntimeError::new_err(
            "nothing to track: pass the markers of the previous frame (call detect() first)",
        ),
        EqarError::Alloc => PyMemoryError::new_err(e.to_string()),
        _ => value_err(e.to_string()),
    }
}

// ---------------------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------------------

/// Raw pointer to the pixel bytes of a call, valid while the owning [`Img`] is alive.
#[derive(Clone, Copy)]
struct RawImg {
    ptr: *const u8,
    len: usize,
    width: usize,
    height: usize,
    stride: usize,
    format: PixelFormat,
}

// SAFETY: the pointer is only dereferenced (read-only) while the `Img` that produced it is
// alive; `Img` keeps the array referenced and borrowed for the whole call.
unsafe impl Send for RawImg {}

impl RawImg {
    /// Takes `self` whole (a by-value method call), so a closure captures the `Send`
    /// wrapper and not its raw-pointer field.
    fn with_view<T>(self, f: impl FnOnce(&ImageView) -> T) -> T {
        // SAFETY: `ptr`/`len` describe memory owned by the live `Img` (see above).
        let data = unsafe { std::slice::from_raw_parts(self.ptr, self.len) };
        f(&ImageView {
            data,
            width: self.width,
            height: self.height,
            stride: self.stride,
            format: self.format,
        })
    }
}

/// An input image validated for a call; keeps its source alive.
struct Img<'py> {
    raw: RawImg,
    _borrow: PyReadonlyArrayDyn<'py, u8>,
    _copy: Vec<u8>,
}

fn parse_format(format: &str) -> PyResult<Option<PixelFormat>> {
    match format.to_ascii_lowercase().as_str() {
        "bgr" => Ok(Some(PixelFormat::Bgr8)),
        "rgb" => Ok(Some(PixelFormat::Rgb8)),
        "gray" | "grey" => Ok(None),
        _ => Err(value_err(format!(
            "format must be \"bgr\", \"rgb\" or \"gray\", got {format:?}"
        ))),
    }
}

fn extract_image<'py>(obj: &Bound<'py, PyAny>, format: &str) -> PyResult<Img<'py>> {
    let fmt3 = parse_format(format)?;
    let arr = match obj.downcast::<PyArrayDyn<u8>>() {
        Ok(a) => a,
        Err(_) => {
            return Err(if obj.hasattr("dtype")? && obj.hasattr("shape")? {
                value_err(format!(
                    "image must have dtype uint8, got {}",
                    obj.getattr("dtype")?
                ))
            } else {
                PyTypeError::new_err(format!(
                    "image must be a numpy.ndarray of dtype uint8, got {}",
                    obj.get_type().name()?
                ))
            });
        }
    };
    let ro = arr.readonly();
    let shape = ro.shape().to_vec();
    let (h, w, fmt) = match shape.as_slice() {
        [h, w] => (*h, *w, PixelFormat::Gray8),
        [h, w, 3] => match fmt3 {
            Some(f) => (*h, *w, f),
            None => {
                return Err(value_err(
                    "format \"gray\" needs a 2-D (H, W) image, got a 3-channel array",
                ))
            }
        },
        [_, _, c] => {
            return Err(value_err(format!(
                "image must have shape (H, W) or (H, W, 3), got (H, W, {c}); \
                 pass gray images as 2-D (H, W)"
            )))
        }
        s => {
            return Err(value_err(format!(
                "image must have shape (H, W) or (H, W, 3), got {s:?}"
            )))
        }
    };
    if h == 0 || w == 0 {
        return Err(value_err("image is empty"));
    }
    let bpp = fmt.bytes_per_pixel();
    let row = w
        .checked_mul(bpp)
        .ok_or_else(|| value_err("image is too large"))?;
    let st = ro.strides().to_vec();
    let direct = if shape.len() == 2 {
        st[1] == 1 && st[0] >= row as isize
    } else {
        st[2] == 1 && st[1] == 3 && st[0] >= row as isize
    };
    if direct {
        let stride = st[0] as usize;
        let len = (h - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row))
            .ok_or_else(|| value_err("image is too large"))?;
        let ptr = ro.as_array().as_ptr();
        let raw = RawImg {
            ptr,
            len,
            width: w,
            height: h,
            stride,
            format: fmt,
        };
        Ok(Img {
            raw,
            _borrow: ro,
            _copy: Vec::new(),
        })
    } else {
        // Transposed, mirrored, channel-sliced or broadcast: take a packed copy.
        let copy: Vec<u8> = ro.as_array().iter().copied().collect();
        let raw = RawImg {
            ptr: copy.as_ptr(),
            len: copy.len(),
            width: w,
            height: h,
            stride: row,
            format: fmt,
        };
        Ok(Img {
            raw,
            _borrow: ro,
            _copy: copy,
        })
    }
}

fn check_size(w: usize, h: usize) -> PyResult<()> {
    if w == 0 || h == 0 || h.checked_mul(2) != Some(w) {
        return Err(value_err(format!(
            "image width must equal 2 x height (equirectangular), got {w}x{h}"
        )));
    }
    if w % 4 != 0 {
        return Err(value_err(format!(
            "image width must be a multiple of 4, got {w}"
        )));
    }
    if !(MIN_WIDTH..=MAX_WIDTH).contains(&w) {
        return Err(value_err(format!(
            "image width must be between {MIN_WIDTH} and {MAX_WIDTH}, got {w}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Marker
// ---------------------------------------------------------------------------------------

fn flag_names(flags: u32) -> String {
    const NAMES: [(u32, &str); 9] = [
        (codes::FLAG_SCAN, "SCAN"),
        (codes::FLAG_TRACK, "TRACK"),
        (codes::FLAG_LOW_CONTRAST, "LOW_CONTRAST"),
        (codes::FLAG_HIGH_RESID, "HIGH_RESID"),
        (codes::FLAG_POLE_CAP, "POLE_CAP"),
        (codes::FLAG_SEAM, "SEAM"),
        (codes::FLAG_SECOND_SCALE, "SECOND_SCALE"),
        (codes::FLAG_LEAN, "LEAN"),
        (codes::FLAG_COASTED, "COASTED"),
    ];
    let v: Vec<&str> = NAMES
        .iter()
        .filter(|(b, _)| flags & b != 0)
        .map(|(_, n)| *n)
        .collect();
    if v.is_empty() {
        "0".to_string()
    } else {
        v.join("|")
    }
}

fn flat<const N: usize, const M: usize>(a: &[[f32; M]; N]) -> Vec<f32> {
    a.iter().flat_map(|r| r.iter().copied()).collect()
}

/// One detected marker (immutable). Array attributes return fresh copies.
#[pyclass(name = "Marker", module = "eqar", frozen)]
struct PyMarker {
    inner: eqar::Marker,
}

#[pymethods]
impl PyMarker {
    /// Dictionary id.
    #[getter]
    fn id(&self) -> i32 {
        self.inner.id
    }
    /// (4, 2) float32: corner pixel positions, TL, TR, BR, BL of the marker's own frame.
    #[getter]
    fn corners<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        PyArray1::from_vec(py, flat(&self.inner.corner_px)).reshape([4, 2])
    }
    /// (4, 3) float32: unit ray of each corner.
    #[getter]
    fn rays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        PyArray1::from_vec(py, flat(&self.inner.ray)).reshape([4, 3])
    }
    /// (2,) float32: pixel position of the marker centre.
    #[getter]
    fn center<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.center_px)
    }
    /// (3,) float32: unit ray of the marker centre.
    #[getter]
    fn center_ray<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.center_ray)
    }
    /// (4,) bool: True where the corner is a measurement.
    #[getter]
    fn corner_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &self.inner.corner_valid)
    }
    /// (4,) uint8: per-corner state bits.
    #[getter]
    fn corner_flags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        PyArray1::from_slice(py, &self.inner.corner_flags)
    }
    /// (4,) float32: factor (>= 1) to multiply each corner's sigma by.
    #[getter]
    fn corner_sigma_scale<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.inner.corner_sigma_scale)
    }
    /// 0..1, higher is better.
    #[getter]
    fn quality(&self) -> f32 {
        self.inner.quality
    }
    /// `FLAG_*` bits.
    #[getter]
    fn flags(&self) -> u32 {
        self.inner.flags
    }
    /// Mean edge length in equivalent pixels.
    #[getter]
    fn side_px(&self) -> f32 {
        self.inner.side_px
    }
    /// Mean normalised edge residual, equivalent pixels.
    #[getter]
    fn edge_resid_px(&self) -> f32 {
        self.inner.edge_resid_px
    }
    /// Mean edge-profile amplitude in grey levels.
    #[getter]
    fn contrast_dn(&self) -> f32 {
        self.inner.contrast_dn
    }
    /// Smallest cross product of adjacent fitted edge normals.
    #[getter]
    fn min_cross(&self) -> f32 {
        self.inner.min_cross
    }
    /// Always `ACCEPT` (0) on a returned marker.
    #[getter]
    fn reject_code(&self) -> i32 {
        self.inner.reject_code
    }
    /// Provenance of the refined quad: `PROV_LEGACY`, `PROV_LINEFIT` or `PROV_RELAXED`.
    #[getter]
    fn prov(&self) -> u8 {
        self.inner.prov
    }
    /// Hamming distance of the accepted code over the known bits.
    #[getter]
    fn n_bit_errors(&self) -> i16 {
        self.inner.n_bit_errors
    }
    /// Erased or not-black border cells.
    #[getter]
    fn border_fail(&self) -> i16 {
        self.inner.border_fail
    }
    /// Erased interior cells.
    #[getter]
    fn n_erased(&self) -> i16 {
        self.inner.n_erased
    }

    /// Plain-Python (JSON-friendly) copy of every field; arrays become nested lists.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = &self.inner;
        let d = PyDict::new(py);
        d.set_item("id", m.id)?;
        d.set_item(
            "corners",
            m.corner_px.iter().map(|c| c.to_vec()).collect::<Vec<_>>(),
        )?;
        d.set_item("rays", m.ray.iter().map(|c| c.to_vec()).collect::<Vec<_>>())?;
        d.set_item("center", m.center_px.to_vec())?;
        d.set_item("center_ray", m.center_ray.to_vec())?;
        d.set_item("corner_valid", m.corner_valid.to_vec())?;
        d.set_item(
            "corner_flags",
            m.corner_flags.iter().map(|&f| f as u32).collect::<Vec<_>>(),
        )?;
        d.set_item("corner_sigma_scale", m.corner_sigma_scale.to_vec())?;
        d.set_item("quality", m.quality)?;
        d.set_item("flags", m.flags)?;
        d.set_item("side_px", m.side_px)?;
        d.set_item("edge_resid_px", m.edge_resid_px)?;
        d.set_item("contrast_dn", m.contrast_dn)?;
        d.set_item("min_cross", m.min_cross)?;
        d.set_item("reject_code", m.reject_code)?;
        d.set_item("prov", m.prov)?;
        d.set_item("n_bit_errors", m.n_bit_errors)?;
        d.set_item("border_fail", m.border_fail)?;
        d.set_item("n_erased", m.n_erased)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let m = &self.inner;
        format!(
            "Marker(id={}, center=({:.2}, {:.2}), side_px={:.1}, quality={:.2}, flags={})",
            m.id,
            m.center_px[0],
            m.center_px[1],
            m.side_px,
            m.quality,
            flag_names(m.flags)
        )
    }
}

// ---------------------------------------------------------------------------------------
// Prior
// ---------------------------------------------------------------------------------------

fn to_array<'py, T: numpy::Element>(
    obj: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Bound<'py, numpy::PyArrayDyn<T>>> {
    let np = obj.py().import("numpy")?;
    let a = np.call_method1("asarray", (obj, dtype))?;
    Ok(a.downcast_into::<numpy::PyArrayDyn<T>>()?)
}

/// A known marker and where it is expected, for `Detector.verify`.
#[pyclass(name = "Prior", module = "eqar", frozen)]
struct PyPrior {
    inner: eqar::Prior,
}

fn make_prior(id: i32, ray: [[f32; 3]; 4], sigma_deg: f32) -> PyResult<PyPrior> {
    if id < 0 {
        return Err(value_err(format!("id must be >= 0, got {id}")));
    }
    if !(sigma_deg > 0.0 && sigma_deg <= 10.0) {
        return Err(value_err(format!(
            "sigma_deg must be in (0, 10], got {sigma_deg}"
        )));
    }
    for r in &ray {
        let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        if !n.is_finite() || !(0.5..=1.5).contains(&n) {
            return Err(value_err(
                "each corner ray must be finite with a norm between 0.5 and 1.5",
            ));
        }
    }
    Ok(PyPrior {
        inner: eqar::Prior { id, ray, sigma_deg },
    })
}

#[pymethods]
impl PyPrior {
    /// `Prior(id, rays, sigma_deg=1.0)`: `rays` is (4, 3), unit rays of the corners in
    /// TL, TR, BR, BL order; `sigma_deg` the angular 1-sigma in (0, 10].
    #[new]
    #[pyo3(signature = (id, rays, sigma_deg=1.0))]
    fn new(id: i32, rays: &Bound<'_, PyAny>, sigma_deg: f32) -> PyResult<Self> {
        let a = to_array::<f32>(rays, "float32")?;
        let ro = a.readonly();
        if ro.shape() != [4, 3] {
            return Err(value_err(format!(
                "rays must have shape (4, 3), got {:?}",
                ro.shape()
            )));
        }
        let v: Vec<f32> = ro.as_array().iter().copied().collect();
        let mut ray = [[0f32; 3]; 4];
        for (i, r) in ray.iter_mut().enumerate() {
            r.copy_from_slice(&v[i * 3..i * 3 + 3]);
        }
        make_prior(id, ray, sigma_deg)
    }

    /// A prior at the position of an earlier detection.
    #[staticmethod]
    #[pyo3(signature = (marker, sigma_deg=1.0))]
    fn from_marker(marker: &PyMarker, sigma_deg: f32) -> PyResult<Self> {
        make_prior(marker.inner.id, marker.inner.ray, sigma_deg)
    }

    #[getter]
    fn id(&self) -> i32 {
        self.inner.id
    }
    /// (4, 3) float32.
    #[getter]
    fn rays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        PyArray1::from_vec(py, flat(&self.inner.ray)).reshape([4, 3])
    }
    #[getter]
    fn sigma_deg(&self) -> f32 {
        self.inner.sigma_deg
    }
    fn __repr__(&self) -> String {
        format!(
            "Prior(id={}, sigma_deg={})",
            self.inner.id, self.inner.sigma_deg
        )
    }
}

// ---------------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------------

fn parse_dict(name: &str) -> PyResult<Dict> {
    Dict::from_str(name).map_err(|_| {
        let names: Vec<&str> = Dict::ALL.iter().map(|d| d.name()).collect();
        value_err(format!(
            "unknown dictionary {name:?}; choose one of {}",
            names.join(", ")
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn build_params(
    dictionary: &str,
    quality: &str,
    max_markers: usize,
    time_budget_ms: Option<f32>,
    use_second_scale: bool,
    reject_high_resid: bool,
    expect_id: Option<i32>,
    gamma_lut: Option<&Bound<'_, PyAny>>,
    in_use_ids: Option<Vec<i32>>,
    coast_max: u32,
) -> PyResult<Params> {
    let dict = parse_dict(dictionary)?;
    let quality = match quality.to_ascii_lowercase().as_str() {
        "default" => Quality::Default,
        "lean" => Quality::Lean,
        _ => {
            return Err(value_err(format!(
                "quality must be \"default\" or \"lean\", got {quality:?}"
            )))
        }
    };
    if max_markers < 1 {
        return Err(value_err("max_markers must be >= 1"));
    }
    let budget = time_budget_ms.unwrap_or(f32::INFINITY);
    if budget.is_nan() || budget < 0.0 {
        return Err(value_err(format!(
            "time_budget_ms must be >= 0 (or None for no limit), got {budget}"
        )));
    }
    let n = dict.len() as i32;
    if let Some(id) = expect_id {
        if id < 0 || id >= n {
            return Err(value_err(format!(
                "expect_id {id} is outside dictionary {dictionary} (0..{n})"
            )));
        }
    }
    if let Some(ids) = &in_use_ids {
        if let Some(bad) = ids.iter().find(|&&i| i < 0 || i >= n) {
            return Err(value_err(format!(
                "in_use_ids contains {bad}, outside dictionary {dictionary} (0..{n})"
            )));
        }
    }
    let lut = match gamma_lut {
        None => None,
        Some(o) => {
            let a = to_array::<u8>(o, "uint8")?;
            let ro = a.readonly();
            if ro.shape() != [256] {
                return Err(value_err(format!(
                    "gamma_lut must have shape (256,), got {:?}",
                    ro.shape()
                )));
            }
            let mut t = [0u8; 256];
            for (d, s) in t.iter_mut().zip(ro.as_array().iter()) {
                *d = *s;
            }
            Some(t)
        }
    };
    Ok(Params {
        dict,
        quality,
        max_markers,
        time_budget_ms: budget,
        use_second_scale,
        reject_high_resid,
        expect_id: expect_id.unwrap_or(-1),
        gamma_lut: lut,
        in_use_ids,
        coast_max,
    })
}

// ---------------------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------------------

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn stats_dict<'py>(py: Python<'py>, s: &Stats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n_candidates", s.n_candidates)?;
    d.set_item("n_bridge_ok", s.n_bridge_ok)?;
    d.set_item("n_decode_ok", s.n_decode_ok)?;
    d.set_item("n_accepted", s.n_accepted)?;
    d.set_item("n_rejected", s.n_rejected)?;
    d.set_item("n_budget_dropped", s.n_budget_dropped)?;
    d.set_item("best_reject_code", s.best_reject_code)?;
    d.set_item("n_tier1", s.n_tier1)?;
    d.set_item("ran_small", s.ran_small != 0)?;
    d.set_item("n_prebridge_rejected", s.n_prebridge_rejected)?;
    d.set_item("n_recheck", s.n_recheck)?;
    d.set_item("n_side_disagree", s.n_side_disagree)?;
    Ok(d)
}

/// A detector bound to one image size and one set of parameters.
#[pyclass(name = "Detector", module = "eqar")]
struct PyDetector {
    core: Mutex<eqar::Detector>,
    /// Timings and counters of the last finished call; separate from `core` so reading
    /// them never waits for a running detection.
    last: Mutex<([f64; 6], Stats)>,
    width: usize,
    height: usize,
    dictionary: &'static str,
    max_markers: usize,
}

impl PyDetector {
    fn make(width: usize, height: usize, params: &Params) -> PyResult<Self> {
        check_size(width, height)?;
        let core = eqar::Detector::new(width, height, params).map_err(convert_error)?;
        Ok(PyDetector {
            core: Mutex::new(core),
            last: Mutex::new(([0.0; 6], Stats::default())),
            width,
            height,
            dictionary: params.dict.name(),
            max_markers: params.max_markers,
        })
    }

    fn check_img(&self, img: &Img<'_>) -> PyResult<()> {
        if img.raw.width != self.width || img.raw.height != self.height {
            return Err(value_err(format!(
                "image is {}x{} but the detector was built for {}x{}",
                img.raw.width, img.raw.height, self.width, self.height
            )));
        }
        Ok(())
    }

    /// Runs `f` on the detector with the GIL released.
    fn run<T: Send>(
        &self,
        py: Python<'_>,
        raw: RawImg,
        f: impl FnOnce(&mut eqar::Detector, &ImageView) -> Result<T, EqarError> + Send,
    ) -> PyResult<T> {
        py.allow_threads(move || {
            let mut core = lock(&self.core);
            let r = raw.with_view(|v| f(&mut core, v));
            *lock(&self.last) = (core.last_timings(), core.last_stats());
            r
        })
        .map_err(convert_error)
    }
}

fn wrap(v: Vec<eqar::Marker>) -> Vec<PyMarker> {
    v.into_iter().map(|inner| PyMarker { inner }).collect()
}

#[pymethods]
impl PyDetector {
    /// `Detector(width, height, dictionary="4x4_50", *, quality="default",
    /// max_markers=32, time_budget_ms=2.0, ...)`.
    #[new]
    #[pyo3(signature = (
        width, height, dictionary="4x4_50", *, quality="default", max_markers=32,
        time_budget_ms=Some(2.0), use_second_scale=true, reject_high_resid=false,
        expect_id=None, gamma_lut=None, in_use_ids=None, coast_max=0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        width: usize,
        height: usize,
        dictionary: &str,
        quality: &str,
        max_markers: usize,
        time_budget_ms: Option<f32>,
        use_second_scale: bool,
        reject_high_resid: bool,
        expect_id: Option<i32>,
        gamma_lut: Option<&Bound<'_, PyAny>>,
        in_use_ids: Option<Vec<i32>>,
        coast_max: u32,
    ) -> PyResult<Self> {
        let p = build_params(
            dictionary,
            quality,
            max_markers,
            time_budget_ms,
            use_second_scale,
            reject_high_resid,
            expect_id,
            gamma_lut,
            in_use_ids,
            coast_max,
        )?;
        Self::make(width, height, &p)
    }

    #[getter]
    fn width(&self) -> usize {
        self.width
    }
    #[getter]
    fn height(&self) -> usize {
        self.height
    }
    #[getter]
    fn dictionary(&self) -> &'static str {
        self.dictionary
    }
    #[getter]
    fn max_markers(&self) -> usize {
        self.max_markers
    }

    /// Full-frame detection. Returns a list of `Marker` (at most `max_markers`).
    #[pyo3(signature = (img, *, format="bgr"))]
    fn detect(
        &self,
        py: Python<'_>,
        img: &Bound<'_, PyAny>,
        format: &str,
    ) -> PyResult<Vec<PyMarker>> {
        let im = extract_image(img, format)?;
        self.check_img(&im)?;
        self.run(py, im.raw, |d, v| d.detect(v)).map(wrap)
    }

    /// Follows `markers` (from the previous observation, `frame_gap` frames ago).
    #[pyo3(signature = (img, markers, frame_gap=1, *, format="bgr"))]
    fn track(
        &self,
        py: Python<'_>,
        img: &Bound<'_, PyAny>,
        markers: Vec<PyRef<'_, PyMarker>>,
        frame_gap: i64,
        format: &str,
    ) -> PyResult<Vec<PyMarker>> {
        let im = extract_image(img, format)?;
        self.check_img(&im)?;
        if frame_gap < 1 || frame_gap > u32::MAX as i64 {
            return Err(value_err(format!(
                "frame_gap must be >= 1, got {frame_gap}"
            )));
        }
        let prev: Vec<eqar::Marker> = markers.iter().map(|m| m.inner).collect();
        self.run(py, im.raw, move |d, v| d.track(v, &prev, frame_gap as u32))
            .map(wrap)
    }

    /// Known-id detection near the expected positions in `priors` (list of `Prior`).
    #[pyo3(signature = (img, priors, *, format="bgr"))]
    fn verify(
        &self,
        py: Python<'_>,
        img: &Bound<'_, PyAny>,
        priors: Vec<PyRef<'_, PyPrior>>,
        format: &str,
    ) -> PyResult<Vec<PyMarker>> {
        let im = extract_image(img, format)?;
        self.check_img(&im)?;
        let n = Dict::from_str(self.dictionary)
            .map(|d| d.len())
            .unwrap_or(0) as i32;
        let pri: Vec<eqar::Prior> = priors.iter().map(|p| p.inner).collect();
        if let Some(p) = pri.iter().find(|p| p.id >= n) {
            return Err(value_err(format!(
                "prior id {} is outside dictionary {} (0..{n})",
                p.id, self.dictionary
            )));
        }
        self.run(py, im.raw, move |d, v| d.verify(v, &pri))
            .map(wrap)
    }

    /// Forgets all tracked and coasting markers.
    fn reset(&self, py: Python<'_>) {
        py.allow_threads(|| lock(&self.core).reset_track());
    }

    /// Stage 1 alone (diagnostic): (n, 4) float32 rows `[rx, ry, rz, angular_radius]`.
    #[pyo3(signature = (img, *, format="bgr"))]
    fn proposals<'py>(
        &self,
        py: Python<'py>,
        img: &Bound<'py, PyAny>,
        format: &str,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let im = extract_image(img, format)?;
        self.check_img(&im)?;
        let rows = self.run(py, im.raw, |d, v| d.proposals(v))?;
        let n = rows.len();
        PyArray1::from_vec(py, rows.into_iter().flatten().collect()).reshape([n, 4])
    }

    /// Time per stage of the last call in ms: propose, bridge, decode, refine, gate, total.
    #[getter]
    fn last_timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let t = lock(&self.last).0;
        let d = PyDict::new(py);
        for (k, v) in ["propose", "bridge", "decode", "refine", "gate", "total"]
            .iter()
            .zip(t)
        {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Counters of the last call.
    #[getter]
    fn last_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = lock(&self.last).1;
        stats_dict(py, &s)
    }

    fn __repr__(&self) -> String {
        format!(
            "Detector({}x{}, dictionary={:?}, max_markers={})",
            self.width, self.height, self.dictionary, self.max_markers
        )
    }
}

// ---------------------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------------------

/// One-shot detection on a single frame (builds a detector for the image's size).
///
/// `time_budget_ms` defaults to `None` (no limit) so a one-off call does not drop markers;
/// `Detector` defaults to 2.0 ms for video.
#[pyfunction]
#[pyo3(signature = (
    img, dictionary="4x4_50", *, format="bgr", quality="default", max_markers=32,
    time_budget_ms=None, use_second_scale=true, reject_high_resid=false,
    expect_id=None, gamma_lut=None, in_use_ids=None
))]
#[allow(clippy::too_many_arguments)]
fn detect(
    py: Python<'_>,
    img: &Bound<'_, PyAny>,
    dictionary: &str,
    format: &str,
    quality: &str,
    max_markers: usize,
    time_budget_ms: Option<f32>,
    use_second_scale: bool,
    reject_high_resid: bool,
    expect_id: Option<i32>,
    gamma_lut: Option<&Bound<'_, PyAny>>,
    in_use_ids: Option<Vec<i32>>,
) -> PyResult<Vec<PyMarker>> {
    let im = extract_image(img, format)?;
    let p = build_params(
        dictionary,
        quality,
        max_markers,
        time_budget_ms,
        use_second_scale,
        reject_high_resid,
        expect_id,
        gamma_lut,
        in_use_ids,
        0,
    )?;
    let det = PyDetector::make(im.raw.width, im.raw.height, &p)?;
    det.run(py, im.raw, |d, v| d.detect(v)).map(wrap)
}

/// Renders marker `id` as a (size, size) uint8 image: black border cell plus code cells,
/// 0 / 255, no quiet zone, nearest-neighbour scaled to `size` pixels.
#[pyfunction]
#[pyo3(signature = (id, dictionary="4x4_50", size=400))]
fn generate_marker<'py>(
    py: Python<'py>,
    id: i32,
    dictionary: &str,
    size: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let dict = parse_dict(dictionary)?;
    let n = dict.len() as i32;
    if id < 0 || id >= n {
        return Err(value_err(format!(
            "id {id} is outside dictionary {dictionary} (0..{n})"
        )));
    }
    let cells = dict.cells();
    if size < cells || size > MAX_WIDTH {
        return Err(value_err(format!(
            "size must be between {cells} (one pixel per cell) and {MAX_WIDTH}, got {size}"
        )));
    }
    let grid = eqar::generate_marker(dict, id, 1).map_err(convert_error)?;
    let mut out = Vec::with_capacity(size * size);
    for y in 0..size {
        let r = y * cells / size;
        for x in 0..size {
            out.push(grid[r * cells + x * cells / size]);
        }
    }
    PyArray1::from_vec(py, out).reshape([size, size])
}

#[pymodule]
fn _eqar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyDetector>()?;
    m.add_class::<PyMarker>()?;
    m.add_class::<PyPrior>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(generate_marker, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("MAX_WIDTH", MAX_WIDTH)?;
    m.add("MIN_WIDTH", MIN_WIDTH)?;
    let names: Vec<&str> = Dict::ALL.iter().map(|d| d.name()).collect();
    m.add("DICTIONARIES", PyTuple::new(py, names)?)?;
    for (n, v) in [
        ("ACCEPT", codes::ACCEPT),
        ("R_NO_CANDIDATE", codes::R_NO_CANDIDATE),
        ("R_BRIDGE_FAILED", codes::R_BRIDGE_FAILED),
        ("R_DEGENERATE_INIT", codes::R_DEGENERATE_INIT),
        ("R_INIT_GEOMETRY", codes::R_INIT_GEOMETRY),
        ("R_EDGE_FIT", codes::R_EDGE_FIT),
        ("R_DISPLACEMENT", codes::R_DISPLACEMENT),
        ("R_DEGENERATE_CORNER", codes::R_DEGENERATE_CORNER),
        ("R_LOW_CONTRAST", codes::R_LOW_CONTRAST),
        ("R_HIGH_RESID", codes::R_HIGH_RESID),
        ("R_BORDER_NOT_BLACK", codes::R_BORDER_NOT_BLACK),
        ("R_IDENTIFY_FAILED", codes::R_IDENTIFY_FAILED),
        ("R_ID_NOT_EXPECTED", codes::R_ID_NOT_EXPECTED),
    ] {
        m.add(n, v)?;
    }
    for (n, v) in [
        ("FLAG_SCAN", codes::FLAG_SCAN),
        ("FLAG_TRACK", codes::FLAG_TRACK),
        ("FLAG_LOW_CONTRAST", codes::FLAG_LOW_CONTRAST),
        ("FLAG_HIGH_RESID", codes::FLAG_HIGH_RESID),
        ("FLAG_POLE_CAP", codes::FLAG_POLE_CAP),
        ("FLAG_SEAM", codes::FLAG_SEAM),
        ("FLAG_SECOND_SCALE", codes::FLAG_SECOND_SCALE),
        ("FLAG_LEAN", codes::FLAG_LEAN),
        ("FLAG_COASTED", codes::FLAG_COASTED),
    ] {
        m.add(n, v)?;
    }
    m.add("PROV_LEGACY", eqar::api::PROV_LEGACY)?;
    m.add("PROV_LINEFIT", eqar::api::PROV_LINEFIT)?;
    m.add("PROV_RELAXED", eqar::api::PROV_RELAXED)?;
    Ok(())
}
