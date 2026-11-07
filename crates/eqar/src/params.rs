//! Per-quality-level presets (the knobs behind `Quality`).
//!
//! `n_sub` feeds the decoder's sub-samples per cell, per axis: 8 by default, 4 under
//! lean (a 4x cut of the dominant per-candidate decode cost). `k_max` documents the
//! "K <= 64 / K <= 32" edge-profile cap but is not independently threaded into the
//! refiner: its side-length driven clamp already tops out at `K_MAX` whatever the
//! quality. The measured lean speed-up comes from `n_sub`, the one knob wired through.

use crate::decode::{DecodeConfig, VerifyConfig, K_SUB_DEFAULT, K_SUB_LEAN};
use crate::error::EqarError;
use crate::refine::K_MAX;

/// Speed / accuracy preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// 4x4 sub-samples per cell in the decoder.
    Lean = 0,
    /// 8x8 sub-samples per cell in the decoder.
    Default = 1,
}

macro_rules! dicts {
    ($($v:ident = $name:literal),* $(,)?) => {
        /// A predefined ArUco dictionary (the OpenCV `DICT_*` set).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(i32)]
        #[allow(non_camel_case_types)]
        pub enum Dict { $(
            #[doc = concat!("`", $name, "`")]
            $v,
        )* }

        impl Dict {
            /// All dictionaries, in index order.
            pub const ALL: [Dict; 16] = [$(Dict::$v),*];

            /// Lower-case name, e.g. `"4x4_50"`.
            pub fn name(self) -> &'static str {
                match self { $(Dict::$v => $name),* }
            }
        }
    };
}

dicts! {
    D4x4_50 = "4x4_50", D4x4_100 = "4x4_100", D4x4_250 = "4x4_250", D4x4_1000 = "4x4_1000",
    D5x5_50 = "5x5_50", D5x5_100 = "5x5_100", D5x5_250 = "5x5_250", D5x5_1000 = "5x5_1000",
    D6x6_50 = "6x6_50", D6x6_100 = "6x6_100", D6x6_250 = "6x6_250", D6x6_1000 = "6x6_1000",
    D7x7_50 = "7x7_50", D7x7_100 = "7x7_100", D7x7_250 = "7x7_250", D7x7_1000 = "7x7_1000",
}

impl Dict {
    /// The dictionary index (4x4_50 = 0 ... 7x7_1000 = 15), as the table lookups use it.
    pub fn index(self) -> i32 {
        self as i32
    }

    /// `Err(EqarError::Dict)` outside 0..16.
    pub fn from_index(i: i32) -> Result<Dict, EqarError> {
        usize::try_from(i)
            .ok()
            .and_then(|i| Dict::ALL.get(i).copied())
            .ok_or(EqarError::Dict)
    }

    /// Number of markers in the dictionary.
    pub fn len(self) -> usize {
        crate::decode::dict_info(self.index()).map_or(0, |d| d.n as usize)
    }

    /// Never empty; present for the `len` convention.
    pub fn is_empty(self) -> bool {
        false
    }

    /// Cells per marker side including the one-cell black border (marker size + 2).
    pub fn cells(self) -> usize {
        crate::decode::dict_info(self.index()).map_or(0, |d| d.marker_size as usize + 2)
    }
}

impl std::str::FromStr for Dict {
    type Err = EqarError;

    /// Case-insensitive; accepts `"4x4_50"` and `"DICT_4X4_50"`.
    fn from_str(s: &str) -> Result<Dict, EqarError> {
        let l = s.to_ascii_lowercase();
        let l = l.strip_prefix("dict_").unwrap_or(&l);
        Dict::ALL
            .iter()
            .copied()
            .find(|d| d.name() == l)
            .ok_or(EqarError::Dict)
    }
}

/// Detector configuration. `Params::default()` is the reference configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Dictionary the markers come from. Default `4x4_50`.
    pub dict: Dict,
    /// Speed / accuracy preset. Default `Quality::Default`.
    pub quality: Quality,
    /// Capacity of every output list (`detect`, `track`, `verify`); must be at least 1.
    /// Default 32.
    pub max_markers: usize,
    /// Stage 2-5 budget per call in milliseconds, started after the proposal stage.
    /// Candidates not reached within it are dropped (counted in `Stats`). Default 2.0;
    /// `f32::INFINITY` removes the limit.
    pub time_budget_ms: f32,
    /// Run the f=2 small-marker pass when nothing larger decoded. Default true.
    pub use_second_scale: bool,
    /// Reject (instead of only flag) markers whose edge residual is high. Default false.
    pub reject_high_resid: bool,
    /// Accept only this id; a negative value accepts every id of the dictionary.
    /// Default -1.
    pub expect_id: i32,
    /// 256-entry gamma look-up table applied to every sample, or `None` (default).
    pub gamma_lut: Option<[u8; 256]>,
    /// Ids in use. Cold detections of other ids are dropped and `verify` checks against
    /// these. `None` (default) is every id of the dictionary; `Some(vec![])` is none.
    pub in_use_ids: Option<Vec<i32>>,
    /// `track` keeps a marker missing for up to this many frames as an internal predicted
    /// prior (never output). Default 0: no coasting.
    pub coast_max: u32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            dict: Dict::D4x4_50,
            quality: Quality::Default,
            max_markers: 32,
            time_budget_ms: 2.0,
            use_second_scale: true,
            reject_high_resid: false,
            expect_id: -1,
            gamma_lut: None,
            in_use_ids: None,
            coast_max: 0,
        }
    }
}

/// The knobs a [`Quality`] stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualityPreset {
    /// decode sub-samples per cell, per axis
    pub n_sub: i32,
    /// documented edge-profile cap; see the module note
    pub k_max: i32,
}

/// The preset behind `quality`.
pub fn resolve_quality(quality: Quality) -> QualityPreset {
    match quality {
        Quality::Lean => QualityPreset {
            n_sub: K_SUB_LEAN,
            k_max: 32,
        },
        Quality::Default => QualityPreset {
            n_sub: K_SUB_DEFAULT,
            k_max: K_MAX as i32,
        },
    }
}

/// COLD decode (ID unknown), the default when the caller supplies an in-use id set: the
/// unchanged legacy decoder, then ids outside the set are dropped. Lenient decoding of an
/// unknown ID gives wrong ids, so nothing is relaxed here.
pub fn cold_decode_config(in_use: Option<&[i32]>) -> DecodeConfig<'_> {
    DecodeConfig { in_use }
}

/// VERIFY decode (ID known from tracking or a map): erasures 0.15, border_fail_max 4,
/// local threshold, other_margin 3 (safe up to 3 corrupted known cells, see `VerifyConfig`).
pub fn verify_decode_config(in_use: Option<&[i32]>) -> VerifyConfig<'_> {
    VerifyConfig {
        in_use,
        ..VerifyConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets() {
        assert_eq!(
            resolve_quality(Quality::Lean),
            QualityPreset {
                n_sub: 4,
                k_max: 32
            }
        );
        assert_eq!(
            resolve_quality(Quality::Default),
            QualityPreset {
                n_sub: 8,
                k_max: 64
            }
        );
    }
}
