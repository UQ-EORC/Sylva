// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Change between two airborne lidar surveys of one area.
//!
//! The principle is that of [`crate::change`]: every change carries its
//! uncertainty or level of detection, and whatever the data cannot support
//! is labelled rather than reported. Everything runs on catalogues through
//! the chunk engine of [`crate::als`]: each chunk reads the same buffered
//! box from both surveys, and the results do not depend on the chunks or
//! the number of workers.
//!
//! * [`align`]: vertical and horizontal offsets between the surveys on
//!   stable surfaces (ground, and optionally roads or roofs), per block or
//!   as a smooth field, with their uncertainty;
//! * [`surface`]: CHM, DSM and DTM differences with a level of detection per
//!   cell, a comparison of the two sensors, and harmonisation by thinning;
//! * [`gaps`]: canopy gaps of each survey, their formation and closure,
//!   polygons and size distributions;
//! * [`trees`]: airborne trees matched between surveys, height growth,
//!   mortality, damage and recruitment;
//! * [`metrics`]: area-based metrics and plant area index compared.

pub mod align;
pub mod gaps;
pub mod metrics;
pub mod surface;
pub mod trees;

use std::collections::HashMap;

use crate::als::{est_points, in_core, read_chunk, Catalog, Chunk};
use crate::error::{Error, Result};
use crate::{Point, PointCloud};

pub use align::Alignment;

/// ASPRS noise classes, left out of every comparison.
const NOISE: [u8; 2] = [7, 18];

/// Normal quantile of a two-sided confidence level.
pub fn z_of(confidence: f64) -> Result<f64> {
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(Error::invalid(format!("confidence must be between 0 and 1, got {confidence}")));
    }
    Ok(crate::change::trees::normal_quantile(0.5 + confidence / 2.0))
}

/// Tiles of `cat` whose extent meets `outer`, in catalogue order.
fn files_in(cat: &Catalog, outer: &[f64; 4]) -> Vec<usize> {
    (0..cat.tiles.len())
        .filter(|&i| {
            let t = cat.tiles[i].xy();
            cat.tiles[i].n_points > 0 && t[0] <= outer[2] && outer[0] <= t[2] && t[1] <= outer[3] && outer[1] <= t[3]
        })
        .collect()
}

fn grown(b: &[f64; 4], d: f64) -> [f64; 4] {
    [b[0] - d, b[1] - d, b[2] + d, b[3] + d]
}

/// Points of the second survey expected in a chunk's box grown by `grow`.
pub(crate) fn est_other(cat_b: &Catalog, chunk: &Chunk, grow: f64) -> u64 {
    let outer = grown(&chunk.outer, grow);
    est_points(cat_b, &outer, &files_in(cat_b, &outer))
}

/// Read the second survey over a chunk of the first: every point in the
/// chunk's buffered box grown by `grow`, moved into the first survey's frame
/// by `alignment` (a point `p` becomes `p - offset(p)`), then kept if it lies
/// in the buffered box. `buffer[i]` is true outside the core.
pub(crate) fn read_other(cat_b: &Catalog, chunk: &Chunk, grow: f64, alignment: Option<&Alignment>) -> Result<(PointCloud, Vec<bool>)> {
    let outer = grown(&chunk.outer, grow);
    let files = files_in(cat_b, &outer);
    if files.is_empty() {
        return Ok((PointCloud::default(), Vec::new()));
    }
    let ch = Chunk { index: chunk.index, core: chunk.core, outer, own: None, est_points: est_points(cat_b, &outer, &files), files, name: chunk.name.clone() };
    let mut cloud = read_chunk(cat_b, &ch)?.cloud;
    if let Some(al) = alignment {
        for p in cloud.xyz.iter_mut() {
            let o = al.offset_at(p[0], p[1]);
            p[0] -= o[0];
            p[1] -= o[1];
            p[2] -= o[2];
        }
    }
    let o = chunk.outer;
    let keep: Vec<usize> = (0..cloud.len()).filter(|&i| {
        let p = cloud.xyz[i];
        p[0] >= o[0] && p[0] <= o[2] && p[1] >= o[1] && p[1] <= o[3]
    }).collect();
    let cloud = if keep.len() == cloud.len() { cloud } else { cloud.take(&keep) };
    let buffer = cloud.xyz.iter().map(|p| !in_core(&chunk.core, p[0], p[1])).collect();
    Ok((cloud, buffer))
}

/// Class of every point (0 when the cloud has none).
pub(crate) fn classes(cloud: &PointCloud) -> Vec<u8> {
    match cloud.attr("classification") {
        Some(c) => (0..cloud.len()).map(|i| c.get_f64(i) as u8).collect(),
        None => vec![0; cloud.len()],
    }
}

/// The cloud without noise (classes 7 and 18).
pub(crate) fn without_noise(cloud: PointCloud, buffer: Vec<bool>) -> (PointCloud, Vec<bool>) {
    let cls = classes(&cloud);
    if !cls.iter().any(|c| NOISE.contains(c)) {
        return (cloud, buffer);
    }
    let keep: Vec<usize> = (0..cloud.len()).filter(|&i| !NOISE.contains(&cls[i])).collect();
    let b = keep.iter().map(|&i| buffer[i]).collect();
    (cloud.take(&keep), b)
}

/// Whether each point is a first return (all are without return numbers).
pub(crate) fn first_returns(cloud: &PointCloud) -> Vec<bool> {
    match cloud.attr("return_number") {
        Some(r) => (0..cloud.len()).map(|i| r.get_f64(i) <= 1.0).collect(),
        None => vec![true; cloud.len()],
    }
}

// ------------------------------------------------------------------ thinning

/// splitmix64 (Steele, Lea and Flood 2014): a well-mixed 64-bit hash.
pub(crate) fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform number in `[0, 1)` from a hash.
pub(crate) fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// A key identifying each point's pulse: its GPS time and flight line, or
/// without GPS time the point's own coordinates (to the millimetre).
pub(crate) fn pulse_keys(cloud: &PointCloud) -> Vec<u64> {
    let psid = cloud.attr("point_source_id");
    match cloud.attr("gps_time") {
        Some(t) => (0..cloud.len()).map(|i| mix(t.get_f64(i).to_bits() ^ psid.map_or(0, |p| (p.get_f64(i) as u64) << 48))).collect(),
        None => cloud.xyz.iter().map(|p| {
            let q = |v: f64| (v * 1000.0).round() as i64 as u64;
            mix(q(p[0]) ^ mix(q(p[1]) ^ mix(q(p[2]))))
        }).collect(),
    }
}

/// Settings of harmonisation by thinning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Harmonise {
    /// Side of the square cells (m, on a grid through the origin) in which
    /// pulse densities are compared.
    pub cell: f64,
    /// Seed of the pulse selection.
    pub seed: u64,
}

impl Harmonise {
    pub fn check(&self) -> Result<()> {
        if !(self.cell.is_finite() && self.cell > 0.0) {
            return Err(Error::invalid(format!("density_cell must be a positive number of metres, got {}", self.cell)));
        }
        Ok(())
    }
}

fn density_key(x: f64, y: f64, cell: f64) -> (i64, i64) {
    ((x / cell).floor() as i64, (y / cell).floor() as i64)
}

/// First returns (pulses) of each density cell.
fn pulse_counts(cloud: &PointCloud, cell: f64) -> HashMap<(i64, i64), f64> {
    let first = first_returns(cloud);
    let mut m: HashMap<(i64, i64), f64> = HashMap::new();
    for (i, p) in cloud.xyz.iter().enumerate() {
        if first[i] {
            *m.entry(density_key(p[0], p[1], cell)).or_default() += 1.0;
        }
    }
    m
}

/// Which points of `own` survive thinning towards the pulse density of
/// `other`: in each density cell where `own` has more pulses, each pulse is
/// kept with probability `n_other / n_own`, decided by a hash of the pulse
/// and the seed, so that every return of a pulse shares its fate and the
/// choice does not depend on how the area was divided. A pulse belongs to
/// the cell of its first return (of its own position when that is not in
/// the cloud).
pub(crate) fn thin_mask(own: &PointCloud, other: &PointCloud, h: &Harmonise) -> Vec<bool> {
    let (n_own, n_other) = (pulse_counts(own, h.cell), pulse_counts(other, h.cell));
    let prob = |x: f64, y: f64| -> f64 {
        let k = density_key(x, y, h.cell);
        let a = n_own.get(&k).copied().unwrap_or(0.0);
        let b = n_other.get(&k).copied().unwrap_or(0.0);
        if a > b && a > 0.0 { b / a } else { 1.0 }
    };
    let keys = pulse_keys(own);
    let first = first_returns(own);
    let mut pulse_p: HashMap<u64, f64> = HashMap::new();
    for (i, p) in own.xyz.iter().enumerate() {
        if first[i] {
            pulse_p.insert(keys[i], prob(p[0], p[1]));
        }
    }
    let seed = mix(h.seed ^ 0x5EED_5EED_5EED_5EED);
    (0..own.len())
        .map(|i| {
            let p = pulse_p.get(&keys[i]).copied().unwrap_or_else(|| prob(own.xyz[i][0], own.xyz[i][1]));
            p >= 1.0 || unit(mix(keys[i] ^ seed)) < p
        })
        .collect()
}

/// Thin both clouds of a chunk towards each other's pulse density.
pub(crate) fn harmonise_pair(a: (PointCloud, Vec<bool>), b: (PointCloud, Vec<bool>), h: &Harmonise) -> ((PointCloud, Vec<bool>), (PointCloud, Vec<bool>)) {
    let ka = thin_mask(&a.0, &b.0, h);
    let kb = thin_mask(&b.0, &a.0, h);
    let apply = |(c, buf): (PointCloud, Vec<bool>), keep: &[bool]| -> (PointCloud, Vec<bool>) {
        let idx: Vec<usize> = (0..c.len()).filter(|&i| keep[i]).collect();
        let b2 = idx.iter().map(|&i| buf[i]).collect();
        (c.take(&idx), b2)
    };
    (apply(a, &ka), apply(b, &kb))
}

// ------------------------------------------------------------------ neighbours and planes

/// A 2-D bucket grid over points, for radius searches whose results come
/// back in index order (so sums over them do not depend on a tree's shape).
pub(crate) struct Grid2 {
    cell: f64,
    map: HashMap<(i64, i64), Vec<usize>>,
}

impl Grid2 {
    pub(crate) fn new(points: &[Point], cell: f64) -> Grid2 {
        let mut map: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            map.entry(((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64)).or_default().push(i);
        }
        Grid2 { cell, map }
    }

    /// Indices of the points within `r` (in x, y) of `(x, y)`, ascending.
    pub(crate) fn within(&self, points: &[Point], x: f64, y: f64, r: f64) -> Vec<usize> {
        let k = (r / self.cell).ceil() as i64;
        let (cx, cy) = ((x / self.cell).floor() as i64, (y / self.cell).floor() as i64);
        let mut out = Vec::new();
        for dx in -k..=k {
            for dy in -k..=k {
                if let Some(v) = self.map.get(&(cx + dx, cy + dy)) {
                    out.extend(v.iter().copied().filter(|&i| (points[i][0] - x).hypot(points[i][1] - y) <= r));
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Distance (in x, y) to the nearest point, searching out to `max`;
    /// infinity if none is that near.
    pub(crate) fn nearest(&self, points: &[Point], x: f64, y: f64, max: f64) -> f64 {
        let (cx, cy) = ((x / self.cell).floor() as i64, (y / self.cell).floor() as i64);
        let kmax = (max / self.cell).ceil() as i64 + 1;
        let mut best = f64::INFINITY;
        for k in 0..=kmax {
            // Ring k of cells around the centre cell.
            for dx in -k..=k {
                for dy in -k..=k {
                    if dx.abs() != k && dy.abs() != k {
                        continue;
                    }
                    if let Some(v) = self.map.get(&(cx + dx, cy + dy)) {
                        for &i in v {
                            best = best.min((points[i][0] - x).hypot(points[i][1] - y));
                        }
                    }
                }
            }
            // Every point in a farther ring is at least k cells away.
            if best <= k as f64 * self.cell {
                break;
            }
        }
        if best <= max { best } else { f64::INFINITY }
    }
}

/// Sort points canonically (x, then y, then z), so that index order is a
/// property of the points and not of how they were read.
pub(crate) fn canonical(mut pts: Vec<Point>) -> Vec<Point> {
    pts.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])).then(a[2].total_cmp(&b[2])));
    pts
}

/// A plane `z = h + gx (x - x0) + gy (y - y0)` fitted by least squares, with
/// the RMS of its residuals and the covariance of its gradient
/// `[[var gx, cov], [cov, var gy]]` from that RMS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Plane {
    pub h: f64,
    pub gx: f64,
    pub gy: f64,
    pub rms: f64,
    pub n: usize,
    pub cov_g: [[f64; 2]; 2],
    /// Variance of `h` per unit residual variance.
    pub var_h: f64,
}

/// Inverse of a 3 x 3 matrix; None if (numerically) singular.
pub(crate) fn inv3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    let diag = (m[0][0] * m[1][1] * m[2][2]).abs();
    if !(det.is_finite() && diag > 0.0 && det.abs() > 1e-12 * diag) {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = c(j, i) / det;
        }
    }
    Some(out)
}

/// Fit a plane through `idx` of `pts`, centred on `(x0, y0)`.
pub(crate) fn fit_plane(pts: &[Point], idx: &[usize], x0: f64, y0: f64) -> Option<Plane> {
    let n = idx.len();
    if n < 3 {
        return None;
    }
    let mut m = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    for &i in idx {
        let p = pts[i];
        let row = [1.0, p[0] - x0, p[1] - y0];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += row[r] * row[c];
            }
            b[r] += row[r] * p[2];
        }
    }
    let inv = inv3(&m)?;
    let s = [0, 1, 2].map(|i| inv[i][0] * b[0] + inv[i][1] * b[1] + inv[i][2] * b[2]);
    let ss: f64 = idx.iter().map(|&i| {
        let p = pts[i];
        let e = p[2] - (s[0] + s[1] * (p[0] - x0) + s[2] * (p[1] - y0));
        e * e
    }).sum();
    let rms = if n > 3 { (ss / (n - 3) as f64).sqrt() } else { 0.0 };
    let v = rms * rms;
    Some(Plane { h: s[0], gx: s[1], gy: s[2], rms, n, cov_g: [[v * inv[1][1], v * inv[1][2]], [v * inv[2][1], v * inv[2][2]]], var_h: inv[0][0] })
}

/// Median of a slice (NaN when empty); sorts it.
pub(crate) fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointcloud::Attr;

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn planes_and_inverses() {
        let pts: Vec<Point> = (0..25).map(|k| {
            let (x, y) = ((k % 5) as f64, (k / 5) as f64);
            [x, y, 1.0 + 0.2 * x - 0.1 * y]
        }).collect();
        let idx: Vec<usize> = (0..25).collect();
        let p = fit_plane(&pts, &idx, 2.0, 2.0).unwrap();
        assert!((p.h - (1.0 + 0.4 - 0.2)).abs() < 1e-12 && (p.gx - 0.2).abs() < 1e-12 && (p.gy + 0.1).abs() < 1e-12 && p.rms < 1e-12);
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
        let inv = inv3(&m).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| m[i][k] * inv[k][j]).sum();
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        assert!(inv3(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]).is_none());
    }

    #[test]
    fn grid_searches_match_brute_force() {
        let pts: Vec<Point> = (0..400).map(|k| [((k * 37) % 101) as f64 * 0.13, ((k * 53) % 97) as f64 * 0.11, 0.0]).collect();
        let g = Grid2::new(&pts, 0.7);
        for &(x, y) in &[(3.0, 4.0), (0.0, 0.0), (12.9, 10.4)] {
            let got = g.within(&pts, x, y, 1.1);
            let want: Vec<usize> = (0..pts.len()).filter(|&i| (pts[i][0] - x).hypot(pts[i][1] - y) <= 1.1).collect();
            assert_eq!(got, want);
            let near = pts.iter().map(|p| (p[0] - x).hypot(p[1] - y)).fold(f64::INFINITY, f64::min);
            assert_eq!(g.nearest(&pts, x, y, 50.0), near);
        }
        assert_eq!(g.nearest(&pts, 100.0, 100.0, 1.0), f64::INFINITY);
    }

    #[test]
    fn thinning_matches_densities_and_keeps_pulses_whole() {
        // Epoch a: one pulse per 0.1 m along x in [0, 20); epoch b: one per 0.25 m.
        let mk = |step: f64, t0: f64| {
            let n = (20.0 / step) as usize;
            let mut xyz = Vec::new();
            let (mut rn, mut nr, mut t) = (Vec::new(), Vec::new(), Vec::new());
            for k in 0..n {
                let x = k as f64 * step;
                for r in 1..=2u8 {
                    xyz.push([x, 0.5, 10.0 / r as f64]);
                    rn.push(r);
                    nr.push(2u8);
                    t.push(t0 + k as f64);
                }
            }
            let mut c = PointCloud::new(xyz);
            c.attrs.insert("return_number".into(), Attr::U8(rn));
            c.attrs.insert("number_of_returns".into(), Attr::U8(nr));
            c.attrs.insert("gps_time".into(), Attr::F64(t));
            c
        };
        let (a, b) = (mk(0.1, 0.0), mk(0.25, 1e6));
        let h = Harmonise { cell: 10.0, seed: 1 };
        let ka = thin_mask(&a, &b, &h);
        let kb = thin_mask(&b, &a, &h);
        assert!(kb.iter().all(|&k| k), "the sparser epoch is not thinned");
        let kept = ka.iter().filter(|&&k| k).count() as f64 / 2.0;
        assert!((kept - 80.0).abs() < 20.0, "{kept} pulses kept of 200, expected about 80");
        for i in (0..a.len()).step_by(2) {
            assert_eq!(ka[i], ka[i + 1], "both returns of a pulse share its fate");
        }
        assert_eq!(ka, thin_mask(&a, &b, &h));
    }
}
