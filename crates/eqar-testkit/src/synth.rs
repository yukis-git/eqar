//! Test-only synthetic equirect marker renderer: a faithful port of the C++
//! renderer (`tests/synth.hpp`), producing byte-identical images for the same
//! arguments. Images are `w * h` row-major gray bytes; markers are painted over the
//! existing contents.

use eqar::decode::dict_info;
use eqar::geometry::{cross, dot, norm, normalize, px_to_ray, Vec3, K_PI, K_TWO_PI};

/// Frame construction: `c = ray(lon, lat)`; `ex = normalize(cross(c, up))` with
/// `up = (0, 1, 0)` (falls back to a fixed meridian exactly at a pole); `ey =
/// cross(ex, c)`; roll then rotates `(ex, ey)` in their own plane. Returns `(c, ex, ey)`.
pub fn frame_at(lon_deg: f64, lat_deg: f64, roll_deg: f64) -> (Vec3, Vec3, Vec3) {
    let lo = lon_deg * K_PI / 180.0;
    let la = lat_deg * K_PI / 180.0;
    let cl = la.cos();
    let c = Vec3::new(cl * lo.cos(), la.sin(), cl * lo.sin());
    let up = Vec3::new(0.0, 1.0, 0.0);
    let mut e0 = cross(&c, &up);
    let n = norm(&e0);
    if n < 1e-9 {
        e0 = Vec3::new(1.0, 0.0, 0.0);
    } else {
        e0 = Vec3::new(e0.x / n, e0.y / n, e0.z / n);
    }
    let e1 = cross(&e0, &c);
    let r = roll_deg * K_PI / 180.0;
    let (cr, sr) = (r.cos(), r.sin());
    let ex = Vec3::new(
        e0.x * cr + e1.x * sr,
        e0.y * cr + e1.y * sr,
        e0.z * cr + e1.z * sr,
    );
    let ey = Vec3::new(
        -e0.x * sr + e1.x * cr,
        -e0.y * sr + e1.y * cr,
        -e0.z * sr + e1.z * cr,
    );
    (c, ex, ey)
}

/// Shared rasteriser: intersects each (4x supersampled) pixel's ray with the marker's
/// tangent plane at `c` (the plane P.c = 1, so t = 1/(D.c)), asks `module_value` for
/// the intensity at the marker-plane coordinates `(qx, qy)` (each in [-1, 1]) and
/// blends it over the existing pixel by coverage. Only the marker's own (small)
/// bounding box is scanned, found analytically from `half_deg`.
fn render(
    px: &mut [u8],
    w: i32,
    h_px: i32,
    lon_deg: f64,
    lat_deg: f64,
    half_deg: f64,
    roll_deg: f64,
    module_value: &dyn Fn(f64, f64) -> f64,
) {
    let (c, ex, ey) = frame_at(lon_deg, lat_deg, roll_deg);
    let h = (half_deg * K_PI / 180.0).tan();
    let (wf, hf) = (w as f64, h_px as f64);

    // Bounding box: the quiet-zone square's diagonal half-angle (atan(sqrt2*h)), with
    // a safety margin, converted to a lat/lon rectangle in pixels.
    let theta_deg = (2.0f64.sqrt() * h).atan() * 180.0 / K_PI * 1.4 + 1.0;
    let mut lat_lo = lat_deg - theta_deg;
    let mut lat_hi = lat_deg + theta_deg;
    lat_lo = lat_lo.max(-89.999);
    lat_hi = lat_hi.min(89.999);
    let row_of = |lat_d: f64| {
        let lat = lat_d * K_PI / 180.0;
        (0.5 * K_PI - lat) / K_PI * hf - 0.5
    };
    let row_lo = 0.max(row_of(lat_hi).floor() as i32 - 1);
    let row_hi = (h_px - 1).min(row_of(lat_lo).ceil() as i32 + 1);

    let full_cols = lat_hi >= 89.5 || lat_lo <= -89.5;
    let mut col_lo = 0;
    let mut col_span = w;
    if !full_cols {
        let coslat = (lat_lo.abs().min(lat_hi.abs()) * K_PI / 180.0).cos();
        let lon_margin = theta_deg / coslat.max(0.05);
        let col_of = |lon_d: f64| {
            let lon = lon_d * K_PI / 180.0;
            (lon + K_PI) / K_TWO_PI * wf - 0.5
        };
        let c0 = col_of(lon_deg - lon_margin);
        let c1 = col_of(lon_deg + lon_margin);
        col_lo = c0.floor() as i32 - 1;
        col_span = c1.ceil() as i32 - col_lo + 2;
        if col_span > w {
            col_lo = 0;
            col_span = w;
        }
    }

    const SS: i32 = 4;
    for y in row_lo..=row_hi {
        for xi in 0..col_span {
            let x = ((col_lo + xi) % w + w) % w;
            let mut acc = 0.0f64;
            let mut hits = 0;
            for sy in 0..SS {
                for sx in 0..SS {
                    let sxp = x as f64 + ((sx as f64 + 0.5) / SS as f64 - 0.5);
                    let syp = y as f64 + ((sy as f64 + 0.5) / SS as f64 - 0.5);
                    let d = px_to_ray(sxp, syp, w, h_px);
                    let den = dot(&d, &c);
                    if den <= 1e-12 {
                        continue; // behind the plane's tangent point
                    }
                    let t = 1.0 / den;
                    let p = Vec3::new(d.x * t, d.y * t, d.z * t);
                    let q = Vec3::new(p.x - c.x, p.y - c.y, p.z - c.z);
                    let qx = dot(&q, &ex) / h;
                    let qy = dot(&q, &ey) / h;
                    if qx.abs() > 1.0 || qy.abs() > 1.0 {
                        continue;
                    }
                    acc += module_value(qx, qy);
                    hits += 1;
                }
            }
            if hits > 0 {
                let idx = y as usize * w as usize + x as usize;
                let bg = px[idx] as f64;
                let v = (acc + (SS * SS - hits) as f64 * bg) / (SS * SS) as f64;
                px[idx] = (v + 0.5) as u8;
            }
        }
    }
}

/// Renders one flat, un-tilted marker into a full-resolution equirect gray buffer:
/// 40 inside the tag (|x|,|y| <= 2/3) and 220 inside the quiet zone (|x|,|y| <= 1) in
/// the marker's own plane coordinates, leaving the background untouched outside it.
pub fn render_test_marker(
    px: &mut [u8],
    w: i32,
    h: i32,
    lon_deg: f64,
    lat_deg: f64,
    half_deg: f64,
    roll_deg: f64,
) {
    render(px, w, h, lon_deg, lat_deg, half_deg, roll_deg, &|qx, qy| {
        if qx.abs() <= 2.0 / 3.0 && qy.abs() <= 2.0 / 3.0 {
            40.0
        } else {
            220.0
        }
    });
}

/// Exact TL,TR,BR,BL tag-corner rays at |coord| = 2/3 of the 9-module square (the
/// tag's true black/white boundary). Uses the same frame as the renderers.
pub fn test_marker_corner_rays(
    lon_deg: f64,
    lat_deg: f64,
    half_deg: f64,
    roll_deg: f64,
) -> [Vec3; 4] {
    let (c, ex, ey) = frame_at(lon_deg, lat_deg, roll_deg);
    let h = (half_deg * K_PI / 180.0).tan();
    let k = 2.0 / 3.0;
    let sx = [-1.0, 1.0, 1.0, -1.0];
    let sy = [1.0, 1.0, -1.0, -1.0];
    let mut out = [Vec3::new(0.0, 0.0, 0.0); 4];
    for i in 0..4 {
        let p = Vec3::new(
            c.x + ex.x * sx[i] * k * h + ey.x * sy[i] * k * h,
            c.y + ex.y * sx[i] * k * h + ey.y * sy[i] * k * h,
            c.z + ex.z * sx[i] * k * h + ey.z * sy[i] * k * h,
        );
        out[i] = normalize(&p);
    }
    out
}

/// The exact plane-centre ray `c` of `frame_at(lon, lat, roll)`: ground truth for a
/// marker's centre ray, independent of any homography reconstruction from corners.
pub fn test_marker_center_ray(lon_deg: f64, lat_deg: f64, roll_deg: f64) -> Vec3 {
    frame_at(lon_deg, lat_deg, roll_deg).0
}

/// Same warp/frame as `render_test_marker`, but paints the actual DICT_4X4_50 bit
/// pattern of `id` (rotation 0) instead of a plain black square: a 9x9-module square
/// with a 1.5-module quiet zone (dark 40 / bright 220 outside/inside k_tag = 2/3), and
/// inside k_tag a 6x6-module tag = a 1-module black border around the 4x4 payload.
/// Payload module (row, col), row 0 = top, is bright (220) when dict bit
/// `ms*ms-1-(row*ms+col)` of `codes[id*4+0]` is 1, else dark (40).
///
/// Panics when `id` is not in DICT_4X4_50 (test helper).
pub fn render_test_marker_id(
    px: &mut [u8],
    w: i32,
    h: i32,
    lon_deg: f64,
    lat_deg: f64,
    half_deg: f64,
    roll_deg: f64,
    id: i32,
) {
    let d = dict_info(0).expect("DICT_4X4_50");
    assert!((0..d.n).contains(&id), "id {id} not in DICT_4X4_50");
    let msize = d.marker_size;
    let ncell = msize + 2;
    let k_tag = ncell as f64 / (ncell + 3) as f64;
    let modw = 2.0 * k_tag / ncell as f64;
    let code = d.codes[id as usize * 4];

    let module_value = move |qx: f64, qy: f64| -> f64 {
        if qx.abs() > k_tag || qy.abs() > k_tag {
            return 220.0; // quiet zone
        }
        let mut col = ((qx + k_tag) / modw).floor() as i32;
        let mut row = ((k_tag - qy) / modw).floor() as i32; // row 0 = top
        col = col.clamp(0, ncell - 1);
        row = row.clamp(0, ncell - 1);
        if row == 0 || row == ncell - 1 || col == 0 || col == ncell - 1 {
            return 40.0;
        }
        let (ir, ic) = (row - 1, col - 1);
        let on = (code >> (msize * msize - 1 - (ir * msize + ic))) & 1 != 0;
        if on {
            220.0
        } else {
            40.0
        }
    };
    render(
        px,
        w,
        h,
        lon_deg,
        lat_deg,
        half_deg,
        roll_deg,
        &module_value,
    );
}

/// One marker to paint in a [`SynthCase`].
#[derive(Debug, Clone, PartialEq)]
pub struct MarkerSpec {
    /// `Some(id)` paints the DICT_4X4_50 pattern; `None` the plain square.
    pub id: Option<i32>,
    pub lon: f64,
    pub lat: f64,
    pub half: f64,
    pub roll: f64,
}

/// A whole synthetic image: constant background plus markers painted in order. The
/// same description is stored in the fixtures and rendered by both implementations.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthCase {
    pub w: i32,
    pub h: i32,
    pub bg: u8,
    pub markers: Vec<MarkerSpec>,
}

impl SynthCase {
    pub fn render(&self) -> Vec<u8> {
        let mut px = vec![self.bg; self.w as usize * self.h as usize];
        for m in &self.markers {
            match m.id {
                Some(id) => {
                    render_test_marker_id(&mut px, self.w, self.h, m.lon, m.lat, m.half, m.roll, id)
                }
                None => render_test_marker(&mut px, self.w, self.h, m.lon, m.lat, m.half, m.roll),
            }
        }
        px
    }

    /// Parse the fixture form: `{"w":..,"h":..,"bg":..,"markers":[{"id":3|null,"lon":..,...}]}`.
    pub fn from_json(v: &crate::json::Value) -> SynthCase {
        SynthCase {
            w: v.get("w").i64() as i32,
            h: v.get("h").i64() as i32,
            bg: v.get("bg").i64() as u8,
            markers: v
                .get("markers")
                .arr()
                .iter()
                .map(|m| MarkerSpec {
                    id: match m.get("id") {
                        crate::json::Value::Null => None,
                        x => Some(x.i64() as i32),
                    },
                    lon: m.get("lon").f64(),
                    lat: m.get("lat").f64(),
                    half: m.get("half").f64(),
                    roll: m.get("roll").f64(),
                })
                .collect(),
        }
    }
}
