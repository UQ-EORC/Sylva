// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Canopy structure: voxels, plant area density, gap fraction, LAI.

pub mod profile;

use crate::error::{Error, Result};
use crate::transform::{add, norm, scale, sub};
use crate::{Point, Shots};

/// Dense 3-D grid of per-voxel values. Index is `x + nx * (y + ny * z)`.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelGrid {
    pub origin: Point,
    pub voxel_size: f64,
    pub shape: [usize; 3],
    pub counts: Vec<u32>,
}

impl VoxelGrid {
    #[inline]
    pub fn flat(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.shape[0] * (j + self.shape[1] * k)
    }

    pub fn len(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bottom edge of each vertical layer.
    pub fn z_levels(&self) -> Vec<f64> {
        (0..self.shape[2]).map(|k| self.origin[2] + k as f64 * self.voxel_size).collect()
    }

    /// Fraction of voxels occupied per vertical layer.
    pub fn vertical_profile(&self) -> Vec<f64> {
        let per_layer = (self.shape[0] * self.shape[1]).max(1) as f64;
        (0..self.shape[2])
            .map(|k| {
                let mut occ = 0usize;
                for j in 0..self.shape[1] {
                    for i in 0..self.shape[0] {
                        if self.counts[self.flat(i, j, k)] > 0 {
                            occ += 1;
                        }
                    }
                }
                occ as f64 / per_layer
            })
            .collect()
    }

    /// Centres of occupied voxels.
    pub fn occupied_centers(&self) -> Vec<Point> {
        let mut out = Vec::new();
        for k in 0..self.shape[2] {
            for j in 0..self.shape[1] {
                for i in 0..self.shape[0] {
                    if self.counts[self.flat(i, j, k)] > 0 {
                        out.push([
                            self.origin[0] + (i as f64 + 0.5) * self.voxel_size,
                            self.origin[1] + (j as f64 + 0.5) * self.voxel_size,
                            self.origin[2] + (k as f64 + 0.5) * self.voxel_size,
                        ]);
                    }
                }
            }
        }
        out
    }
}

/// Count points per voxel. `origin`/`shape` default to the point bounds.
pub fn voxelize(points: &[Point], voxel_size: f64, origin: Option<Point>, shape: Option<[usize; 3]>) -> Result<VoxelGrid> {
    if voxel_size <= 0.0 {
        return Err(Error::invalid("voxel_size must be positive"));
    }
    let origin = origin.unwrap_or_else(|| {
        let lo = crate::util::spatial::min_corner(points);
        [(lo[0] / voxel_size).floor() * voxel_size, (lo[1] / voxel_size).floor() * voxel_size, (lo[2] / voxel_size).floor() * voxel_size]
    });
    let shape = shape.unwrap_or_else(|| {
        if points.is_empty() {
            return [1, 1, 1];
        }
        let hi = crate::util::spatial::max_corner(points);
        let mut s = [1usize; 3];
        for k in 0..3 {
            s[k] = (((hi[k] - origin[k]) / voxel_size).floor() as usize) + 1;
        }
        s
    });
    let mut grid = VoxelGrid { origin, voxel_size, shape, counts: vec![0; shape[0] * shape[1] * shape[2]] };
    for p in points {
        let mut idx = [0usize; 3];
        let mut ok = true;
        for k in 0..3 {
            let v = ((p[k] - origin[k]) / voxel_size).floor();
            if v < 0.0 || v >= shape[k] as f64 {
                ok = false;
                break;
            }
            idx[k] = v as usize;
        }
        if ok {
            let f = grid.flat(idx[0], idx[1], idx[2]);
            grid.counts[f] += 1;
        }
    }
    Ok(grid)
}

/// Point-count histogram by height. Returns `(bin_bottoms, counts)`.
pub fn vertical_histogram(heights: &[f64], bin_size: f64, max_height: Option<f64>) -> (Vec<f64>, Vec<u64>) {
    let top = max_height.unwrap_or_else(|| heights.iter().cloned().fold(0.0, f64::max));
    let nbins = ((top / bin_size).ceil() as usize).max(1);
    let mut counts = vec![0u64; nbins];
    for &h in heights {
        if h >= 0.0 {
            let b = (h / bin_size) as usize;
            if b < nbins {
                counts[b] += 1;
            }
        }
    }
    ((0..nbins).map(|b| b as f64 * bin_size).collect(), counts)
}

/// Plant area density profile by the vertical contact-frequency method
/// (simplified Hosoi & Omasa 2006, vertical beams, G = 0.5).
///
/// Each voxel column is traced top-down; the fraction of still-alive columns
/// intercepted in a layer gives `PAD = -ln(1 - p) / (G dz)`.
/// Returns `(layer_bottoms, pad)`.
pub fn pad_profile_voxel(points: &[Point], heights: &[f64], voxel_size: f64, max_height: Option<f64>, clumping: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    let keep: Vec<Point> = points.iter().zip(heights).filter(|(_, &h)| h > 0.0).map(|(p, &h)| [p[0], p[1], h]).collect();
    if keep.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let lo = crate::util::spatial::min_corner(&keep);
    let hi = crate::util::spatial::max_corner(&keep);
    let origin = [lo[0].floor(), lo[1].floor(), 0.0];
    let top = max_height.unwrap_or(hi[2]);
    let shape = [
        ((hi[0] - origin[0]) / voxel_size).floor() as usize + 1,
        ((hi[1] - origin[1]) / voxel_size).floor() as usize + 1,
        ((top / voxel_size).ceil() as usize).max(1),
    ];
    let grid = voxelize(&keep, voxel_size, Some(origin), Some(shape))?;
    let nz = shape[2];
    let mut pad = vec![0.0; nz];
    let mut alive = vec![true; shape[0] * shape[1]];
    for k in (0..nz).rev() {
        let n_alive = alive.iter().filter(|&&a| a).count();
        if n_alive == 0 {
            break;
        }
        let mut n_hit = 0usize;
        for j in 0..shape[1] {
            for i in 0..shape[0] {
                let col = i + shape[0] * j;
                if alive[col] && grid.counts[grid.flat(i, j, k)] > 0 {
                    n_hit += 1;
                    alive[col] = false;
                }
            }
        }
        let p = (n_hit as f64 / n_alive as f64).min(1.0 - 1e-6);
        pad[k] = -(1.0 - p).ln() / (0.5 * voxel_size) * clumping;
    }
    Ok((grid.z_levels(), pad))
}

/// Directional gap fraction by zenith ring from one scan position.
///
/// With `shots`, a shot is a gap when it has no echo above `min_height`
/// (heights given per echo). Returns `(ring_centres_deg, gap_fraction)`;
/// rings with no shots are NaN.
pub fn gap_fraction_zenith(shots: &Shots, echo_heights: &[f64], min_height: f64, zenith_edges_deg: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let nb = zenith_edges_deg.len().saturating_sub(1);
    let mut total = vec![0u64; nb];
    let mut hits = vec![0u64; nb];
    for s in 0..shots.n_shots() {
        let d = shots.direction[s];
        let zen = (d[2] / norm(&d).max(1e-12)).clamp(-1.0, 1.0).acos().to_degrees();
        let Some(b) = zenith_edges_deg.windows(2).position(|w| zen >= w[0] && zen < w[1]) else { continue };
        total[b] += 1;
        let a = shots.echo_start[s];
        let hit = (a..a + shots.echo_count[s] as usize).any(|e| echo_heights[e] > min_height);
        if hit {
            hits[b] += 1;
        }
    }
    let centres = zenith_edges_deg.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();
    let gap = (0..nb).map(|b| if total[b] > 0 { 1.0 - hits[b] as f64 / total[b] as f64 } else { f64::NAN }).collect();
    (centres, gap)
}

/// Gap fractions below this are treated as saturated (cf. canopygrid's `pai_lim`).
pub const GAP_FLOOR: f64 = 1e-5;

/// Effective plant area index from directional gap fraction.
///
/// `hinge`: `PAI = -ln P(57.5°) cos(57.5°) / 0.5` (≈ 1.1 · -ln P). `miller`:
/// `PAI = 2 ∫ -ln P(θ) cos θ sin θ dθ`. NaN rings are skipped; a gap of 0 is
/// floored at [`GAP_FLOOR`] so saturated rings contribute a large finite value.
pub fn lai_from_gap_fraction(zenith_deg: &[f64], gap: &[f64], method: &str) -> Result<f64> {
    let pairs: Vec<(f64, f64)> = zenith_deg.iter().zip(gap).filter(|(_, &g)| g.is_finite()).map(|(&z, &g)| (z.to_radians(), g.max(GAP_FLOOR))).collect();
    if pairs.is_empty() {
        return Ok(f64::NAN);
    }
    match method {
        "hinge" => {
            let target = 57.5f64.to_radians();
            let (_, g) = pairs.iter().cloned().min_by(|a, b| (a.0 - target).abs().partial_cmp(&(b.0 - target).abs()).unwrap()).unwrap();
            Ok(-g.ln() * target.cos() / 0.5)
        }
        "miller" => {
            let mut s = 0.0;
            for w in pairs.windows(2) {
                let f = |(z, g): (f64, f64)| -g.ln() * z.cos() * z.sin();
                s += 0.5 * (f(w[0]) + f(w[1])) * (w[1].0 - w[0].0);
            }
            Ok(2.0 * s)
        }
        other => Err(Error::invalid(format!("unknown LAI method {other:?}"))),
    }
}

/// Fraction of CHM cells at or above `threshold`.
pub fn canopy_cover(chm: &[f64], threshold: f64) -> f64 {
    let valid: Vec<&f64> = chm.iter().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        return f64::NAN;
    }
    valid.iter().filter(|&&&v| v >= threshold).count() as f64 / valid.len() as f64
}

/// Ray-traced density grid (raycloudtools / Lowe et al. 2021 style).
///
/// Every shot is traced through the grid; each traversed voxel accumulates
/// the path length and number of rays, and the voxel holding an echo counts
/// a hit. Unbounded shots (no echo) traverse until they leave the grid.
#[derive(Debug, Clone)]
pub struct DensityGrid {
    pub origin: Point,
    pub voxel_size: f64,
    pub shape: [usize; 3],
    pub n_rays: Vec<u32>,
    pub n_hits: Vec<u32>,
    pub path_length: Vec<f64>,
}

impl DensityGrid {
    pub fn new(origin: Point, voxel_size: f64, shape: [usize; 3]) -> Self {
        let n = shape[0] * shape[1] * shape[2];
        DensityGrid { origin, voxel_size, shape, n_rays: vec![0; n], n_hits: vec![0; n], path_length: vec![0.0; n] }
    }

    #[inline]
    fn flat(&self, i: [i64; 3]) -> Option<usize> {
        if (0..3).all(|k| i[k] >= 0 && (i[k] as usize) < self.shape[k]) {
            Some(i[0] as usize + self.shape[0] * (i[1] as usize + self.shape[1] * i[2] as usize))
        } else {
            None
        }
    }

    /// Trace a segment from `origin` along unit `dir` for `length` (∞ for
    /// unbounded), by the Amanatides & Woo (1987) traversal.
    /// Returns the flat index of the last voxel visited (where an echo lies).
    fn trace(&mut self, origin: &Point, dir: &Point, length: f64) -> Option<usize> {
        // Clip to the grid box (slab test).
        let mut t0 = 0.0f64;
        let mut t1 = length;
        let gmax = [
            self.origin[0] + self.shape[0] as f64 * self.voxel_size,
            self.origin[1] + self.shape[1] as f64 * self.voxel_size,
            self.origin[2] + self.shape[2] as f64 * self.voxel_size,
        ];
        for k in 0..3 {
            if dir[k].abs() < 1e-12 {
                if origin[k] < self.origin[k] || origin[k] >= gmax[k] {
                    return None;
                }
                continue;
            }
            let a = (self.origin[k] - origin[k]) / dir[k];
            let b = (gmax[k] - origin[k]) / dir[k];
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
        if t1 <= t0 {
            return None;
        }
        let start = add(origin, &scale(dir, t0 + 1e-9));
        let mut idx = [0i64; 3];
        let mut t_next = [0.0f64; 3];
        let mut step = [0i64; 3];
        let mut t_delta = [f64::INFINITY; 3];
        for k in 0..3 {
            idx[k] = ((start[k] - self.origin[k]) / self.voxel_size).floor() as i64;
            idx[k] = idx[k].clamp(0, self.shape[k] as i64 - 1);
            if dir[k] > 0.0 {
                step[k] = 1;
                t_delta[k] = self.voxel_size / dir[k];
                let edge = self.origin[k] + (idx[k] + 1) as f64 * self.voxel_size;
                t_next[k] = (edge - origin[k]) / dir[k];
            } else if dir[k] < 0.0 {
                step[k] = -1;
                t_delta[k] = -self.voxel_size / dir[k];
                let edge = self.origin[k] + idx[k] as f64 * self.voxel_size;
                t_next[k] = (edge - origin[k]) / dir[k];
            } else {
                t_next[k] = f64::INFINITY;
            }
        }
        let mut t = t0;
        let mut last = None;
        loop {
            let Some(f) = self.flat(idx) else { break };
            let k = if t_next[0] < t_next[1] && t_next[0] < t_next[2] { 0 } else if t_next[1] < t_next[2] { 1 } else { 2 };
            let t_exit = t_next[k].min(t1);
            let seg = (t_exit - t).max(0.0);
            self.path_length[f] += seg;
            self.n_rays[f] += 1;
            last = Some(f);
            if t_next[k] >= t1 {
                break;
            }
            t = t_next[k];
            t_next[k] += t_delta[k];
            idx[k] += step[k];
        }
        last
    }

    /// Accumulate all shots.
    pub fn add_shots(&mut self, shots: &Shots) {
        for s in 0..shots.n_shots() {
            let o = shots.origin[s];
            let d = shots.direction[s];
            let a = shots.echo_start[s];
            let c = shots.echo_count[s] as usize;
            if c == 0 {
                self.trace(&o, &d, f64::INFINITY);
                continue;
            }
            // Trace to the last echo; each echo marks a hit in its voxel.
            let last_range = shots.echo_range[a + c - 1];
            self.trace(&o, &d, last_range);
            for e in a..a + c {
                let p = add(&o, &scale(&d, shots.echo_range[e]));
                let idx = [
                    ((p[0] - self.origin[0]) / self.voxel_size).floor() as i64,
                    ((p[1] - self.origin[1]) / self.voxel_size).floor() as i64,
                    ((p[2] - self.origin[2]) / self.voxel_size).floor() as i64,
                ];
                if let Some(f) = self.flat(idx) {
                    self.n_hits[f] += 1;
                }
            }
        }
    }

    /// Per-voxel plant area density estimate
    /// `2 (n-1)/n · hits / path_length` (spherical leaf angle distribution),
    /// NaN where fewer than `min_hits` hits or no rays.
    pub fn density(&self, min_hits: u32) -> Vec<f64> {
        (0..self.n_rays.len())
            .map(|i| {
                let n = self.n_rays[i] as f64;
                if self.n_rays[i] == 0 || self.n_hits[i] < min_hits {
                    return f64::NAN;
                }
                2.0 * (n - 1.0) * self.n_hits[i] as f64 / (1e-9 + n * self.path_length[i])
            })
            .collect()
    }

    /// Mean density per vertical layer (ignoring NaN).
    pub fn vertical_profile(&self, min_hits: u32) -> Vec<f64> {
        let d = self.density(min_hits);
        (0..self.shape[2])
            .map(|k| {
                let mut s = 0.0;
                let mut n = 0.0;
                for j in 0..self.shape[1] {
                    for i in 0..self.shape[0] {
                        let v = d[i + self.shape[0] * (j + self.shape[1] * k)];
                        if v.is_finite() {
                            s += v;
                            n += 1.0;
                        }
                    }
                }
                if n > 0.0 { s / n } else { f64::NAN }
            })
            .collect()
    }
}

/// Convenience: build a density grid covering `shots`' echoes.
pub fn density_grid_from_shots(shots: &Shots, voxel_size: f64) -> Result<DensityGrid> {
    let xyz = shots.echo_xyz();
    if xyz.is_empty() {
        return Err(Error::invalid("no echoes"));
    }
    let lo = crate::util::spatial::min_corner(&xyz);
    let hi = crate::util::spatial::max_corner(&xyz);
    let origin = [(lo[0] / voxel_size).floor() * voxel_size, (lo[1] / voxel_size).floor() * voxel_size, (lo[2] / voxel_size).floor() * voxel_size];
    let shape = [
        ((hi[0] - origin[0]) / voxel_size).floor() as usize + 1,
        ((hi[1] - origin[1]) / voxel_size).floor() as usize + 1,
        ((hi[2] - origin[2]) / voxel_size).floor() as usize + 1,
    ];
    let mut g = DensityGrid::new(origin, voxel_size, shape);
    g.add_shots(shots);
    Ok(g)
}

#[allow(dead_code)]
fn _unused(a: &Point, b: &Point) -> Point {
    sub(a, b)
}

/// Returns and fired pulses of one or more scans, binned by zenith ring,
/// azimuth sector and height above ground: the inputs of a Jupp et al. (2009)
/// gap-probability profile.
#[derive(Debug, Clone, PartialEq)]
pub struct PgapHistogram {
    /// Zenith ring edges (deg).
    pub zenith_edges: Vec<f64>,
    pub n_azimuth: usize,
    pub height_bin: f64,
    pub n_heights: usize,
    /// Weighted returns `[ring][sector][height]`: each echo of a pulse with
    /// `n` echoes counts `1 / n` (equal weighting, Armston et al. 2013). Heights at or above the top bin go in the
    /// last bin.
    pub hits: Vec<f64>,
    /// Pulses fired `[ring][sector]`.
    pub shots: Vec<f64>,
}

impl PgapHistogram {
    pub fn new(zenith_edges: Vec<f64>, n_azimuth: usize, height_bin: f64, n_heights: usize) -> Self {
        let n_rings = zenith_edges.len().saturating_sub(1);
        let n_azimuth = n_azimuth.max(1);
        PgapHistogram { zenith_edges, n_azimuth, height_bin, n_heights, hits: vec![0.0; n_rings * n_azimuth * n_heights], shots: vec![0.0; n_rings * n_azimuth] }
    }

    pub fn n_rings(&self) -> usize {
        self.zenith_edges.len().saturating_sub(1)
    }

    fn ring_of(&self, zenith_deg: f64) -> Option<usize> {
        let e = &self.zenith_edges;
        if !(zenith_deg >= e[0] && zenith_deg < e[e.len() - 1]) {
            return None;
        }
        Some(e.partition_point(|&v| v <= zenith_deg) - 1)
    }

    fn sector_of(&self, azimuth_deg: f64) -> usize {
        let a = azimuth_deg.rem_euclid(360.0);
        ((a / 360.0 * self.n_azimuth as f64) as usize).min(self.n_azimuth - 1)
    }

    /// Add pulses. `echo_heights` are heights above ground, one per echo;
    /// echoes below `min_height` (or NaN) are not counted as returns but still
    /// count towards the pulse's echo number. `fired_per_ring`, when given,
    /// replaces the observed pulse counts (for streams without the pulses that
    /// returned nothing), spread evenly over the azimuth sectors.
    pub fn add(&mut self, shots: &Shots, echo_heights: &[f64], min_height: f64, fired_per_ring: Option<&[f64]>) {
        let n_rings = self.n_rings();
        let (na, nh) = (self.n_azimuth, self.n_heights);
        for s in 0..shots.n_shots() {
            let d = shots.direction[s];
            let zen = d[2].clamp(-1.0, 1.0).acos().to_degrees();
            let Some(ring) = self.ring_of(zen) else { continue };
            let sector = self.sector_of(d[0].atan2(d[1]).to_degrees());
            if fired_per_ring.is_none() {
                self.shots[ring * na + sector] += 1.0;
            }
            let (e0, n) = (shots.echo_start[s], shots.echo_count[s] as usize);
            if n == 0 {
                continue;
            }
            let w = 1.0 / n as f64;
            for e in e0..e0 + n {
                let h = echo_heights[e];
                if !(h >= min_height) {
                    continue;
                }
                let k = ((h / self.height_bin).floor().max(0.0) as usize).min(nh - 1);
                self.hits[(ring * na + sector) * nh + k] += w;
            }
        }
        if let Some(f) = fired_per_ring {
            for ring in 0..n_rings.min(f.len()) {
                for sector in 0..na {
                    self.shots[ring * na + sector] += f[ring] / na as f64;
                }
            }
        }
    }

    /// Sum of another histogram with the same bins.
    pub fn merge(&mut self, other: &PgapHistogram) -> Result<()> {
        if other.zenith_edges != self.zenith_edges || other.n_azimuth != self.n_azimuth || other.n_heights != self.n_heights || other.height_bin != self.height_bin {
            return Err(Error::invalid("histograms have different bins"));
        }
        self.hits.iter_mut().zip(&other.hits).for_each(|(a, b)| *a += b);
        self.shots.iter_mut().zip(&other.shots).for_each(|(a, b)| *a += b);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn shots(origin: Vec<Point>, direction: Vec<Point>, counts: &[u32], ranges: Vec<f64>) -> Shots {
        let mut start = Vec::with_capacity(counts.len());
        let mut s = 0;
        for &c in counts {
            start.push(s);
            s += c as usize;
        }
        Shots { origin, direction, echo_start: start, echo_count: counts.to_vec(), echo_range: ranges, echo_attrs: BTreeMap::new() }
    }

    #[test]
    fn voxel_grid_counts_layers_and_centres() {
        let pts = [[0.1, 0.1, 0.1], [0.2, 0.3, 0.4], [1.5, 0.5, 0.5], [0.5, 1.5, 1.5], [9.0, 9.0, 9.0]];
        let g = voxelize(&pts, 1.0, Some([0.0; 3]), Some([2, 2, 2])).unwrap();
        assert_eq!((g.len(), g.is_empty()), (8, false));
        // The point outside the given shape is dropped.
        assert_eq!(g.counts.iter().sum::<u32>(), 4);
        assert_eq!(g.counts[g.flat(0, 0, 0)], 2);
        assert_eq!(g.z_levels(), [0.0, 1.0]);
        assert_eq!(g.vertical_profile(), [0.5, 0.25]);
        assert_eq!(g.occupied_centers(), [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 1.5]]);
        // Default origin: the minimum corner floored to the voxel; shape: up to the maximum.
        let d = voxelize(&[[0.3, -0.7, 2.2], [1.1, 0.2, 2.9]], 0.5, None, None).unwrap();
        assert_eq!((d.origin, d.shape), ([0.0, -1.0, 2.0], [3, 3, 2]));
        assert_eq!(voxelize(&[], 1.0, Some([0.0; 3]), None).unwrap().shape, [1, 1, 1]);
        assert_eq!(voxelize(&pts, 0.0, None, None).unwrap_err().to_string(), "voxel_size must be positive");
    }

    #[test]
    fn vertical_histogram_by_brute_force() {
        let h = [0.0, 0.49, 0.5, 1.2, 2.99, 3.0, -0.1, 7.0];
        let (bottoms, counts) = vertical_histogram(&h, 0.5, Some(3.0));
        assert_eq!(bottoms, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let expect: Vec<u64> = bottoms.iter().map(|&b| h.iter().filter(|&&v| v >= b && v < b + 0.5).count() as u64).collect();
        assert_eq!(counts, expect);
        assert_eq!(counts.iter().sum::<u64>(), 5, "below 0 and at or above the top are left out");
        // Without a top, the highest height sets it.
        assert_eq!(vertical_histogram(&[0.2, 1.7], 1.0, None), (vec![0.0, 1.0], vec![1, 1]));
    }

    #[test]
    fn lai_methods_and_cover() {
        // A uniform gap P: hinge = -ln P cos(57.5) / 0.5; Miller = -ln P (sin^2 b - sin^2 a).
        let p: f64 = 0.3;
        let z = [0.0, 30.0, 57.5, 90.0];
        let hinge = lai_from_gap_fraction(&z, &[p; 4], "hinge").unwrap();
        assert!((hinge + p.ln() * 57.5f64.to_radians().cos() / 0.5).abs() < 1e-12);
        let fine: Vec<f64> = (0..=900).map(|i| i as f64 * 0.1).collect();
        let miller = lai_from_gap_fraction(&fine, &vec![p; fine.len()], "miller").unwrap();
        assert!((miller + p.ln()).abs() < 1e-5, "{miller}");
        assert!(lai_from_gap_fraction(&z, &[f64::NAN; 4], "hinge").unwrap().is_nan());
        // A saturated ring counts as the floor, not as infinity.
        let saturated = lai_from_gap_fraction(&[57.5], &[0.0], "hinge").unwrap();
        assert!((saturated + GAP_FLOOR.ln() * 57.5f64.to_radians().cos() / 0.5).abs() < 1e-9);
        assert_eq!(lai_from_gap_fraction(&z, &[p; 4], "ellipse").unwrap_err().to_string(), "unknown LAI method \"ellipse\"");
        assert_eq!(canopy_cover(&[0.0, 3.0, f64::NAN, 2.0, 1.9], 2.0), 0.5);
        assert!(canopy_cover(&[f64::NAN], 2.0).is_nan());
    }

    #[test]
    fn gap_fraction_by_ring() {
        // Four pulses straight up (zenith 0) and two at 45 degrees; one of each returns above 2 m.
        let up = [0.0, 0.0, 1.0];
        let tilted = [0.5f64.sqrt(), 0.0, 0.5f64.sqrt()];
        let s = shots(vec![[0.0; 3]; 6], vec![up, up, up, up, tilted, tilted], &[1, 1, 0, 0, 1, 1], vec![5.0, 1.0, 5.0, 1.0]);
        let heights = [5.0, 1.0, 3.5, 0.7];
        let (centres, gap) = gap_fraction_zenith(&s, &heights, 2.0, &[0.0, 30.0, 60.0, 90.0]);
        assert_eq!(centres, [15.0, 45.0, 75.0]);
        assert_eq!(gap[..2], [0.75, 0.5]);
        assert!(gap[2].is_nan(), "a ring without pulses");
    }

    #[test]
    fn density_grid_traces_to_the_echo() {
        // Two pulses along x through a row of 4 voxels, one stopped in voxel 2, one a miss.
        let s = shots(vec![[-1.0, 0.5, 0.5]; 2], vec![[1.0, 0.0, 0.0]; 2], &[1, 0], vec![3.5]);
        let mut g = DensityGrid::new([0.0; 3], 1.0, [4, 1, 1]);
        g.add_shots(&s);
        assert_eq!(g.n_rays, [2, 2, 2, 1]);
        assert_eq!(g.n_hits, [0, 0, 1, 0]);
        let expect = [2.0, 2.0, 1.5, 1.0];
        for (got, want) in g.path_length.iter().zip(expect) {
            assert!((got - want).abs() < 1e-6, "{got} {want}");
        }
        // 2 (n - 1) / n * hits / path: 2 * 1/2 * 1 / 1.5 in voxel 2.
        let d = g.density(1);
        assert!((d[2] - 2.0 / 3.0).abs() < 1e-6 && d[0].is_nan() && d[3].is_nan());
        assert!((g.vertical_profile(1)[0] - 2.0 / 3.0).abs() < 1e-6);
        // Fitted to the echoes: one voxel around the single echo.
        let fitted = density_grid_from_shots(&s, 1.0).unwrap();
        assert_eq!((fitted.origin, fitted.shape), ([2.0, 0.0, 0.0], [1, 1, 1]));
        assert_eq!(fitted.n_hits, [1]);
        let none = shots(vec![[0.0; 3]], vec![up()], &[0], vec![]);
        assert_eq!(density_grid_from_shots(&none, 1.0).unwrap_err().to_string(), "no echoes");
    }

    fn up() -> Point {
        [0.0, 0.0, 1.0]
    }

    #[test]
    fn pgap_histograms_merge_bin_by_bin() {
        let s = shots(vec![[0.0; 3]; 2], vec![up(), up()], &[2, 0], vec![3.0, 6.0]);
        let mut a = PgapHistogram::new(vec![0.0, 45.0, 90.0], 4, 1.0, 10);
        a.add(&s, &[3.0, 6.0], 0.0, None);
        let mut b = a.clone();
        b.merge(&a).unwrap();
        assert_eq!(b.shots.iter().sum::<f64>(), 4.0);
        assert_eq!(b.hits.iter().sum::<f64>(), 2.0, "two echoes of one pulse weigh 1/2 each");
        assert!(b.hits.iter().zip(&a.hits).all(|(x, y)| *x == 2.0 * y));
        let other = PgapHistogram::new(vec![0.0, 45.0, 90.0], 8, 1.0, 10);
        assert_eq!(b.merge(&other).unwrap_err().to_string(), "histograms have different bins");
    }
}
