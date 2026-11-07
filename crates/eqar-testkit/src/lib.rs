//! Test support for eqar: deterministic synthetic equirectangular images (a port of
//! the reference renderer), a SHA-256, and a minimal JSON reader for `tests/fixtures`.
//! Not published.

// Numeric kernels keep the C++ index-based loops and argument lists so they can be
// compared line by line with the reference.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod json;
pub mod sha256;
pub mod synth;

use std::path::PathBuf;

/// Directory holding the JSON fixtures written by the C++ reference generator.
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

/// Load and parse `tests/fixtures/<name>`.
pub fn load_fixture(name: &str) -> json::Value {
    let p = fixtures_dir().join(name);
    let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    json::parse(&text).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}
