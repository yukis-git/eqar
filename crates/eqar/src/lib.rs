//! Equirectangular (360 degree) ArUco marker detector.
//!
//! Finds ArUco markers directly in equirectangular frames (width = 2 x height), on the
//! sphere: no cube-map or perspective views, so markers at the poles and across the
//! +-180 degree seam are found like any other. [`Detector::detect`] scans a frame,
//! [`Detector::track`] follows markers from frame to frame, and [`Detector::verify`]
//! looks for known ids near known positions.
//!
//! ```no_run
//! use eqar::{Detector, ImageView, Params, PixelFormat};
//!
//! # fn main() -> Result<(), eqar::EqarError> {
//! let (width, height) = (3840, 1920);
//! let gray = vec![0u8; width * height]; // one frame, row-major
//! let mut det = Detector::new(width, height, &Params::default())?;
//! let img = ImageView::packed(&gray, width, height, PixelFormat::Gray8);
//! let mut markers = det.detect(&img)?;
//! for m in &markers {
//!     println!("id {} corners {:?}", m.id, m.corner_px);
//! }
//! // next frame: follow what was found
//! if !markers.is_empty() {
//!     markers = det.track(&img, &markers, 1)?;
//! }
//! # let _ = markers;
//! # Ok(())
//! # }
//! ```
#![deny(unsafe_code)]
#![warn(missing_docs)]
// Numeric kernels keep the C++ index-based loops and argument lists so they can be
// compared line by line with the reference.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
// `!(a > b)` is kept where the C++ relies on it being true for NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod api;
#[doc(hidden)]
pub mod bridge;
pub mod codes;
#[doc(hidden)]
pub mod decode;
#[rustfmt::skip]
#[doc(hidden)]
pub mod dict_tables;
#[doc(hidden)]
pub mod detector;
pub mod error;
#[doc(hidden)]
pub mod geometry;
#[doc(hidden)]
pub mod guards;
#[doc(hidden)]
pub mod image;
pub mod params;
#[doc(hidden)]
pub mod proposal;
#[doc(hidden)]
pub mod refine;
#[allow(unsafe_code)]
#[doc(hidden)]
pub mod simd;
#[doc(hidden)]
pub mod track;

pub use api::{
    generate_marker, Detector, ImageView, Marker, PixelFormat, Prior, Stats, MAX_WIDTH, MIN_WIDTH,
    PROV_LEGACY, PROV_LINEFIT, PROV_RELAXED,
};
pub use error::EqarError;
pub use params::{Dict, Params, Quality};
