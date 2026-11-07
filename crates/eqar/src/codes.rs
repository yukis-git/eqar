//! Reject codes and marker flag bits (the numeric values of the C++ implementation).

/// Accepted.
pub const ACCEPT: i32 = 0;
/// No candidate survived the proposal stage.
pub const R_NO_CANDIDATE: i32 = 1;
/// The bridge found no acceptable quad for the candidate.
pub const R_BRIDGE_FAILED: i32 = 2;
/// The initial quad is degenerate.
pub const R_DEGENERATE_INIT: i32 = 3;
/// The initial quad's geometry is implausible.
pub const R_INIT_GEOMETRY: i32 = 4;
/// An edge could not be fitted (too little support).
pub const R_EDGE_FIT: i32 = 5;
/// The refined quad moved too far from the bridge quad.
pub const R_DISPLACEMENT: i32 = 6;
/// A fitted corner is degenerate.
pub const R_DEGENERATE_CORNER: i32 = 7;
/// The edge contrast is too low.
pub const R_LOW_CONTRAST: i32 = 8;
/// The edge residual is too high (with `Params::reject_high_resid`).
pub const R_HIGH_RESID: i32 = 9;
/// The one-cell border is not black.
pub const R_BORDER_NOT_BLACK: i32 = 10;
/// The code is not in the dictionary.
pub const R_IDENTIFY_FAILED: i32 = 11;
/// A different id than the expected one (or a prior elsewhere).
pub const R_ID_NOT_EXPECTED: i32 = 12;

/// Found by the full-frame scan.
pub const FLAG_SCAN: u32 = 1 << 0;
/// Found by the tracker or the known-id verify path.
pub const FLAG_TRACK: u32 = 1 << 1;
/// Low edge contrast.
pub const FLAG_LOW_CONTRAST: u32 = 1 << 2;
/// The edge residual ratio exceeds the bound.
pub const FLAG_HIGH_RESID: u32 = 1 << 3;
/// Found through the polar-cap projection.
pub const FLAG_POLE_CAP: u32 = 1 << 4;
/// Reserved: never set on an output marker (a marker across the +-180 degree seam is
/// reported like any other, with corners near both image borders).
pub const FLAG_SEAM: u32 = 1 << 5;
/// Found by the f=2 small-marker pass.
pub const FLAG_SECOND_SCALE: u32 = 1 << 6;
/// Detected with the lean preset (four or more tracked markers).
pub const FLAG_LEAN: u32 = 1 << 7;
/// Reserved, never set on an output marker.
pub const FLAG_COASTED: u32 = 1 << 8;
