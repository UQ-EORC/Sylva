// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A TLS plot registered onto an ALS survey.
//!
//! The plot may be in a local frame (a scanner or project frame placed
//! roughly by `initial`) or already georeferenced with a GNSS error of a few
//! metres. Two stages:
//!
//! 1. **Search.** The canopy height model (CHM) and terrain model (DTM) of
//!    each cloud are made on one grid spacing. The TLS cells are a template
//!    that is turned by a heading and shifted horizontally over the ALS
//!    rasters; each pose is scored by the correlation (Pearson's r) of the two
//!    CHMs, plus `dtm_weight` times that of the two DTMs, over the cells both
//!    measured. The correlation ignores a vertical offset between the
//!    terrains and a TLS canopy that reads low where the scanner saw less of
//!    the crowns. The whole window (`search_radius`, `heading_range`) is
//!    scored at `coarse_resolution`, the best separated peaks are searched
//!    again at `resolution` around themselves (bilinear sampling, and a
//!    parabola through the best score in each of heading, x and y), and the
//!    vertical offset is the median difference of the two DTMs.
//!    The fine cells are widened for sparse surveys, to hold about
//!    `returns_per_cell` ALS returns each (a CHM of cells with one or two
//!    returns is too ragged to place a plot by).
//! 2. **Refinement.** A robust point-to-plane ICP ([`crate::coreg::icp`]:
//!    Huber weights, trimming) of the TLS points onto the ALS points, on the
//!    ground and the canopy both see, or on the ground alone, from each of
//!    the refined peaks. A run counts only if it stays within
//!    `max_refine_shift` and `max_refine_turn` of its peak; of those, the
//!    one that fits best (the most TLS points within the last ICP cut-off
//!    of an ALS point; the higher peak when two fit within half a percent)
//!    is kept. The ICP's 3-D fit of
//!    ground and crowns tells poses apart that the canopy models cannot:
//!    a TLS canopy model reads the crowns from below and to the side, so its
//!    best correlation can lie a few degrees or metres off.
//!
//! Uncertainty is reported twice: the formal covariance of the ICP
//! (`sigma² H⁻¹` from the point-to-plane information, which treats every
//! residual as independent and so is a lower bound), and a jackknife over the
//! four quadrants of the plot (each left out in turn and the ICP run again),
//! which shows how much the answer depends on which part of the plot is used.
//!
//! Everything is computed about a pivot (the centre of the TLS ground after
//! `initial`), so map coordinates of any size lose no precision; the
//! translations and the covariance refer to that pivot.

use nalgebra::{Matrix4, Matrix6};
use rayon::prelude::*;

use crate::coreg::geometry::{voxel_downsample, CoregTree};
use crate::coreg::icp::{self, IcpConfig, IcpTarget};
use crate::error::{Error, Result};
use crate::ground::make_dtm;
use crate::raster::{Raster, Reducer};
use crate::Point;

/// Which points the ICP refinement uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refine {
    /// Keep the search result.
    None,
    /// Ground points of both clouds.
    Ground,
    /// Every point (ground, stems, crowns).
    All,
}

impl Refine {
    pub fn parse(s: &str) -> Result<Refine> {
        match s {
            "none" => Ok(Refine::None),
            "ground" => Ok(Refine::Ground),
            "all" => Ok(Refine::All),
            _ => Err(Error::invalid(format!("unknown refine {s:?}; expected 'all', 'ground' or 'none'"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Refine::None => "none",
            Refine::Ground => "ground",
            Refine::All => "all",
        }
    }
}

/// Settings of [`register`].
#[derive(Debug, Clone)]
pub struct RegisterParams {
    /// Smallest cell size (m) of the fine search and the residuals.
    pub resolution: f64,
    /// ALS returns a fine cell should hold on average: the fine cells are
    /// widened to `sqrt(returns_per_cell / density)` for sparse surveys.
    pub returns_per_cell: f64,
    /// Cell size (m) of the search over the whole window.
    pub coarse_resolution: f64,
    /// Largest horizontal shift (m) searched, from the initial position.
    pub search_radius: f64,
    /// Headings searched either side of the initial one (degrees); 180 or
    /// more searches the full circle.
    pub heading_range: f64,
    /// Heading step (degrees) of the coarse search.
    pub heading_step: f64,
    /// Weight of the DTM correlation against the CHM's.
    pub dtm_weight: f64,
    /// CHM heights below this (m) are 0: ground, low shrubs.
    pub min_height: f64,
    /// Share of the TLS CHM cells that must fall on ALS data for a pose to count.
    pub min_overlap: f64,
    /// Peaks of the coarse search refined.
    pub n_candidates: usize,
    pub refine: Refine,
    /// ICP voxel pyramid (m), coarse to fine.
    pub icp_voxel_sizes: Vec<f64>,
    /// ICP correspondence cut-off per level (m).
    pub icp_max_distances: Vec<f64>,
    /// Planarity a target point needs to be used (0 to 1).
    pub min_planarity: f64,
    /// Largest horizontal move (m) and turn (degrees) of the ICP from the
    /// search result.
    pub max_refine_shift: f64,
    pub max_refine_turn: f64,
    /// Run the quadrant jackknife.
    pub jackknife: bool,
}

impl Default for RegisterParams {
    fn default() -> Self {
        RegisterParams {
            resolution: 0.5,
            returns_per_cell: 10.0,
            coarse_resolution: 2.0,
            search_radius: 10.0,
            heading_range: 180.0,
            heading_step: 3.0,
            dtm_weight: 0.5,
            min_height: 2.0,
            min_overlap: 0.5,
            n_candidates: 3,
            refine: Refine::All,
            icp_voxel_sizes: vec![1.0, 0.5, 0.25],
            icp_max_distances: vec![2.0, 1.0, 0.5],
            min_planarity: 0.3,
            max_refine_shift: 3.0,
            max_refine_turn: 8.0,
            jackknife: true,
        }
    }
}

impl RegisterParams {
    fn check(&self) -> Result<()> {
        let pos = |v: f64| v.is_finite() && v > 0.0;
        if !pos(self.resolution) || !pos(self.coarse_resolution) || self.coarse_resolution < self.resolution {
            return Err(Error::invalid("resolution and coarse_resolution must be positive, coarse_resolution at least resolution"));
        }
        if !(self.search_radius.is_finite() && self.search_radius >= 0.0) || !(self.heading_range.is_finite() && self.heading_range >= 0.0) || !pos(self.heading_step) {
            return Err(Error::invalid("search_radius and heading_range must be >= 0 and heading_step positive"));
        }
        if !(self.dtm_weight.is_finite() && self.dtm_weight >= 0.0) || !self.min_height.is_finite() || !(self.min_overlap > 0.0 && self.min_overlap <= 1.0) {
            return Err(Error::invalid("dtm_weight must be >= 0, min_height finite and min_overlap in (0, 1]"));
        }
        if self.n_candidates == 0 {
            return Err(Error::invalid("n_candidates must be at least 1"));
        }
        if self.icp_voxel_sizes.is_empty() || self.icp_voxel_sizes.len() != self.icp_max_distances.len() || self.icp_voxel_sizes.iter().chain(&self.icp_max_distances).any(|&v| !pos(v)) {
            return Err(Error::invalid("icp_voxel_sizes and icp_max_distances must be positive and of equal length"));
        }
        if !(0.0..=1.0).contains(&self.min_planarity) || !(self.max_refine_shift.is_finite() && self.max_refine_shift >= 0.0) || !(self.max_refine_turn.is_finite() && self.max_refine_turn >= 0.0) {
            return Err(Error::invalid("min_planarity must be in [0, 1], max_refine_shift and max_refine_turn >= 0"));
        }
        if !(self.returns_per_cell.is_finite() && self.returns_per_cell >= 0.0) {
            return Err(Error::invalid("returns_per_cell must be >= 0"));
        }
        Ok(())
    }
}

/// One pose of the search: heading (degrees) and shift (m) of the pivot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    pub heading: f64,
    pub dx: f64,
    pub dy: f64,
    pub score: f64,
    pub chm_r: f64,
    pub dtm_r: f64,
    /// TLS CHM cells that fell on ALS data.
    pub overlap: usize,
}

/// Summary of the ICP refinement.
#[derive(Debug, Clone)]
pub struct IcpSummary {
    pub fitness: f64,
    pub rmse: f64,
    pub n: usize,
    pub iterations: usize,
    pub converged: bool,
    /// False when the ICP moved too far from the search and was dropped.
    pub accepted: bool,
    /// Horizontal move (m) and heading change (degrees) of the ICP from
    /// the peak it started at.
    pub shift: f64,
    pub turn: f64,
    /// The peak (index into the candidates) the kept ICP started from.
    pub start: usize,
}

/// Residuals of the final registration.
#[derive(Debug, Clone, Default)]
pub struct Residuals {
    /// TLS ground points minus the ALS DTM (m): count, median, normalised
    /// median absolute deviation, RMSE.
    pub ground_n: usize,
    pub ground_median: f64,
    pub ground_nmad: f64,
    pub ground_rmse: f64,
    /// Distance (m) of TLS canopy points to the nearest ALS canopy return
    /// (within 5 m): count, median, 90th percentile.
    pub canopy_n: usize,
    pub canopy_median: f64,
    pub canopy_p90: f64,
}

/// Result of [`register`].
#[derive(Debug, Clone)]
pub struct Registration {
    /// Fine and coarse cell sizes used (m) and the ALS density (returns per m²).
    pub resolution: f64,
    pub coarse_resolution: f64,
    pub als_density: f64,
    /// Maps the TLS input coordinates into the ALS frame.
    pub transform: Matrix4<f64>,
    /// The same after the search, before the ICP: the best peak.
    pub search_transform: Matrix4<f64>,
    pub pivot: Point,
    /// Heading (degrees, counter-clockwise) and shift (m) of the pivot
    /// added to `initial`.
    pub heading: f64,
    pub shift: [f64; 3],
    /// Refined peaks of the search, best first.
    pub candidates: Vec<Candidate>,
    /// Misfit (1 - score) of the best peak over that of the second (NaN with
    /// one peak): 0 for a clear answer, near 1 when two poses fit about as
    /// well.
    pub ambiguity: f64,
    pub icp: Option<IcpSummary>,
    /// Formal covariance of `[rx, ry, rz, tx, ty, tz]` (radians, m) at the pivot.
    pub covariance: Option<Matrix6<f64>>,
    /// Jackknife standard errors of the pivot's x, y, z (m) and the heading
    /// (degrees) over the four quadrants.
    pub jackknife: Option<[f64; 4]>,
    pub residuals: Residuals,
    pub n_tls: usize,
    pub n_als: usize,
}

// ------------------------------------------------------------------ rasters

struct Layers {
    /// Canopy height, NaN where there is no point.
    chm: Raster,
    /// Lowest ground point per cell, NaN where there is none.
    dtm: Raster,
    /// Terrain with its gaps filled.
    dtm_filled: Raster,
}

fn snapped(b: [f64; 4], res: f64) -> (f64, f64, f64, f64) {
    ((b[0] / res).floor() * res, (b[1] / res).floor() * res, b[2], b[3])
}

fn layers(points: &[Point], ground: &[bool], res: f64, bounds: [f64; 4], min_height: f64, what: &str) -> Result<Layers> {
    let b = snapped(bounds, res);
    let g: Vec<Point> = points.iter().zip(ground).filter(|(_, &k)| k).map(|(p, _)| *p).collect();
    if g.len() < 3 {
        return Err(Error::invalid(format!("the {what} cloud has fewer than 3 ground points (classification 2) in the search area")));
    }
    let dtm = Raster::from_points(g.iter().map(|p| (p[0], p[1])), g.iter().map(|p| p[2]), res, Reducer::Min, Some(b), f64::NAN)?;
    let dtm_filled = make_dtm(&g, res, Some(b))?;
    let h: Vec<f64> = points.par_iter().map(|p| {
        let v = p[2] - dtm_filled.sample(p[0], p[1]);
        if v < min_height { 0.0 } else { v }
    }).collect();
    let chm = Raster::from_points(points.iter().map(|p| (p[0], p[1])), h.into_iter(), res, Reducer::Max, Some(b), f64::NAN)?;
    Ok(Layers { chm, dtm, dtm_filled })
}

/// Value at `(x, y)`: the cell's (nearest) or bilinear between cell
/// centres; NaN outside the grid or next to a NaN cell.
fn lookup(r: &Raster, x: f64, y: f64, bilinear: bool) -> f64 {
    let fc = (x - r.xmin) / r.resolution;
    let fr = (y - r.ymin) / r.resolution;
    if !bilinear {
        if fc < 0.0 || fr < 0.0 {
            return f64::NAN;
        }
        let (c, row) = (fc as usize, fr as usize);
        if c >= r.ncols || row >= r.nrows {
            return f64::NAN;
        }
        return r.data[row * r.ncols + c];
    }
    let (fc, fr) = (fc - 0.5, fr - 0.5);
    if fc < 0.0 || fr < 0.0 {
        return f64::NAN;
    }
    let (c0, r0) = (fc as usize, fr as usize);
    if c0 + 1 >= r.ncols || r0 + 1 >= r.nrows {
        return f64::NAN;
    }
    let (tx, ty) = (fc - c0 as f64, fr - r0 as f64);
    let n = r.ncols;
    let v = |row: usize, col: usize| r.data[row * n + col];
    v(r0, c0) * (1.0 - tx) * (1.0 - ty) + v(r0, c0 + 1) * tx * (1.0 - ty) + v(r0 + 1, c0) * (1.0 - tx) * ty + v(r0 + 1, c0 + 1) * tx * ty
}

/// Cell centres and values of the finite cells of `r`.
fn template(r: &Raster) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for row in 0..r.nrows {
        for col in 0..r.ncols {
            let v = r.data[row * r.ncols + col];
            if v.is_finite() {
                let (x, y) = r.cell_center(row, col);
                out.push([x, y, v]);
            }
        }
    }
    out
}

/// Pearson's r between the template and the surface under a pose, and the
/// number of cells compared.
fn correlation(tpl: &[[f64; 3]], surf: &Raster, heading: f64, dx: f64, dy: f64, bilinear: bool) -> (f64, usize) {
    let (s, c) = heading.to_radians().sin_cos();
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0usize, 0.0, 0.0, 0.0, 0.0, 0.0);
    for t in tpl {
        let b = lookup(surf, c * t[0] - s * t[1] + dx, s * t[0] + c * t[1] + dy, bilinear);
        if b.is_finite() {
            let a = t[2];
            n += 1;
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
        }
    }
    if n < 3 {
        return (f64::NAN, n);
    }
    let nf = n as f64;
    let (va, vb) = (saa - sa * sa / nf, sbb - sb * sb / nf);
    if va <= 1e-12 * nf || vb <= 1e-12 * nf {
        return (0.0, n);
    }
    ((sab - sa * sb / nf) / (va * vb).sqrt(), n)
}

struct Level {
    chm_tpl: Vec<[f64; 3]>,
    dtm_tpl: Vec<[f64; 3]>,
    als: Layers,
}

impl Level {
    fn score(&self, heading: f64, dx: f64, dy: f64, w: f64, min_overlap: f64, bilinear: bool) -> Candidate {
        let (rc, nc) = correlation(&self.chm_tpl, &self.als.chm, heading, dx, dy, bilinear);
        let (rd, nd) = if w > 0.0 && !self.dtm_tpl.is_empty() { correlation(&self.dtm_tpl, &self.als.dtm_filled, heading, dx, dy, bilinear) } else { (f64::NAN, 0) };
        let enough = (nc as f64) >= min_overlap * self.chm_tpl.len() as f64 && nc >= 3;
        let score = if !enough {
            f64::NEG_INFINITY
        } else {
            let d = if rd.is_finite() && (nd as f64) >= min_overlap * self.dtm_tpl.len() as f64 { rd } else { 0.0 };
            (if rc.is_finite() { rc } else { 0.0 } + w * d) / (1.0 + w)
        };
        Candidate { heading, dx, dy, score, chm_r: rc, dtm_r: rd, overlap: nc }
    }
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

fn wrap(a: f64) -> f64 {
    let w = (a + 180.0).rem_euclid(360.0) - 180.0;
    if w == -180.0 { 180.0 } else { w }
}

fn better(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    // Higher score first; ties in a fixed order of the pose.
    b.score.total_cmp(&a.score).then(a.heading.total_cmp(&b.heading)).then(a.dx.total_cmp(&b.dx)).then(a.dy.total_cmp(&b.dy))
}

fn headings(p: &RegisterParams) -> Vec<f64> {
    if p.heading_range >= 180.0 {
        let n = (360.0 / p.heading_step).ceil() as usize;
        (0..n).map(|k| wrap(-180.0 + k as f64 * 360.0 / n as f64)).collect()
    } else {
        let n = (p.heading_range / p.heading_step).ceil() as i64;
        let step = if n > 0 { p.heading_range / n as f64 } else { 0.0 };
        (-n..=n).map(|k| k as f64 * step).collect()
    }
}

fn shifts(radius: f64, step: f64) -> Vec<(f64, f64)> {
    let n = (radius / step).ceil() as i64;
    let mut out = Vec::new();
    for j in -n..=n {
        for i in -n..=n {
            let (x, y) = (i as f64 * step, j as f64 * step);
            if x.hypot(y) <= radius + 1e-9 {
                out.push((x, y));
            }
        }
    }
    out
}

/// Search a level over every heading and shift, and return its best
/// separated peaks.
fn coarse_search(level: &Level, p: &RegisterParams) -> Vec<Candidate> {
    let hs = headings(p);
    let ts = shifts(p.search_radius, p.coarse_resolution);
    let mut all: Vec<Candidate> = hs.par_iter().flat_map_iter(|&h| ts.iter().map(move |&(x, y)| level.score(h, x, y, p.dtm_weight, p.min_overlap, false))).filter(|c| c.score.is_finite()).collect();
    all.sort_by(better);
    let mut peaks: Vec<Candidate> = Vec::new();
    for c in all {
        if peaks.len() >= p.n_candidates {
            break;
        }
        if peaks.iter().all(|q| (c.dx - q.dx).hypot(c.dy - q.dy) > 2.0 * p.coarse_resolution || angle_diff(c.heading, q.heading) > 2.0 * p.heading_step) {
            peaks.push(c);
        }
    }
    peaks
}

/// Offset of the vertex of the parabola through three equally spaced
/// scores, in steps, within [-1, 1].
fn parabola(a: f64, b: f64, c: f64) -> f64 {
    let den = a - 2.0 * b + c;
    if !(a.is_finite() && b.is_finite() && c.is_finite()) || den >= 0.0 {
        return 0.0;
    }
    (0.5 * (a - c) / den).clamp(-1.0, 1.0)
}

fn fine_search(level: &Level, start: &Candidate, p: &RegisterParams) -> Candidate {
    let dh = p.heading_step / 6.0;
    let nt = (p.coarse_resolution / p.resolution).ceil() as i64;
    let limited = p.heading_range < 180.0;
    let ks: Vec<i64> = (-6..=6).collect();
    let poses: Vec<(f64, f64, f64)> = ks
        .iter()
        .flat_map(|&k| {
            let h = start.heading + k as f64 * dh;
            (-nt..=nt).flat_map(move |j| (-nt..=nt).map(move |i| (h, start.dx + i as f64 * p.resolution, start.dy + j as f64 * p.resolution)))
        })
        .filter(|&(h, x, y)| (!limited || h.abs() <= p.heading_range + 1e-9) && x.hypot(y) <= p.search_radius + p.resolution)
        .collect();
    let mut scored: Vec<Candidate> = poses.par_iter().map(|&(h, x, y)| level.score(h, x, y, p.dtm_weight, p.min_overlap, true)).filter(|c| c.score.is_finite()).collect();
    if scored.is_empty() {
        return *start;
    }
    scored.sort_by(better);
    let b = scored[0];
    let s = |h: f64, x: f64, y: f64| level.score(h, x, y, p.dtm_weight, p.min_overlap, true).score;
    let r = p.resolution;
    let oh = parabola(s(b.heading - dh, b.dx, b.dy), b.score, s(b.heading + dh, b.dx, b.dy));
    let ox = parabola(s(b.heading, b.dx - r, b.dy), b.score, s(b.heading, b.dx + r, b.dy));
    let oy = parabola(s(b.heading, b.dx, b.dy - r), b.score, s(b.heading, b.dx, b.dy + r));
    let polished = level.score(b.heading + oh * dh, b.dx + ox * r, b.dy + oy * r, p.dtm_weight, p.min_overlap, true);
    let mut out = if polished.score.is_finite() && polished.score >= b.score { polished } else { b };
    out.heading = wrap(out.heading);
    out
}

// ---------------------------------------------------------------- transforms

fn rz(heading_deg: f64, t: [f64; 3]) -> Matrix4<f64> {
    let (s, c) = heading_deg.to_radians().sin_cos();
    let mut m = Matrix4::identity();
    m[(0, 0)] = c;
    m[(0, 1)] = -s;
    m[(1, 0)] = s;
    m[(1, 1)] = c;
    m[(0, 3)] = t[0];
    m[(1, 3)] = t[1];
    m[(2, 3)] = t[2];
    m
}

fn translation(t: [f64; 3]) -> Matrix4<f64> {
    rz(0.0, t)
}

fn heading_of(m: &Matrix4<f64>) -> f64 {
    m[(1, 0)].atan2(m[(0, 0)]).to_degrees()
}

fn apply(m: &Matrix4<f64>, pts: &[Point]) -> Vec<Point> {
    icp::transform_points(m, pts)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

/// Every `k`-th element so that at most `cap` remain.
fn thin<T: Copy>(v: &[T], cap: usize) -> Vec<T> {
    if v.len() <= cap {
        return v.to_vec();
    }
    let k = v.len().div_ceil(cap);
    v.iter().step_by(k).copied().collect()
}

fn bounds_of(pts: &[Point]) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in pts {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

fn residuals(tls: &[Point], tls_ground: &[bool], als: &[Point], als_ground: &[bool], dtm: &Raster, min_height: f64) -> Residuals {
    let g: Vec<Point> = thin(&tls.iter().zip(tls_ground).filter(|(_, &k)| k).map(|(p, _)| *p).collect::<Vec<_>>(), 500_000);
    let mut dz: Vec<f64> = g.par_iter().map(|p| p[2] - lookup(dtm, p[0], p[1], true)).filter(|v| v.is_finite()).collect();
    let mut out = Residuals { ground_n: dz.len(), ..Default::default() };
    if !dz.is_empty() {
        out.ground_rmse = (dz.iter().map(|v| v * v).sum::<f64>() / dz.len() as f64).sqrt();
        let m = median(&mut dz);
        let mut dev: Vec<f64> = dz.iter().map(|v| (v - m).abs()).collect();
        out.ground_median = m;
        out.ground_nmad = 1.4826 * median(&mut dev);
    } else {
        (out.ground_median, out.ground_nmad, out.ground_rmse) = (f64::NAN, f64::NAN, f64::NAN);
    }
    let canopy_als: Vec<Point> = als.iter().zip(als_ground).filter(|(p, &k)| !k && p[2] - lookup(dtm, p[0], p[1], true) >= min_height).map(|(p, _)| *p).collect();
    let canopy_tls: Vec<Point> = thin(&tls.iter().zip(tls_ground).filter(|(p, &k)| !k && p[2] - lookup(dtm, p[0], p[1], true) >= min_height).map(|(p, _)| *p).collect::<Vec<_>>(), 200_000);
    let tree = CoregTree::new(&canopy_als);
    let (d, _) = tree.query(&canopy_tls, 5.0);
    let mut d: Vec<f64> = d.into_iter().filter(|v| v.is_finite()).collect();
    out.canopy_n = d.len();
    out.canopy_median = median(&mut d);
    out.canopy_p90 = quantile(&mut d, 0.9);
    out
}

fn icp_config(p: &RegisterParams) -> IcpConfig {
    IcpConfig {
        voxel_sizes: p.icp_voxel_sizes.clone(),
        max_distances: Some(p.icp_max_distances.clone()),
        max_iterations: 30,
        method: "point_to_plane".into(),
        robust: "huber".into(),
        robust_scale: 0.5 * p.icp_voxel_sizes.last().copied().unwrap_or(0.25),
        trim_fraction: 0.8,
        min_planarity: p.min_planarity,
        fitness_threshold: *p.icp_max_distances.last().unwrap_or(&0.5),
        max_points: 150_000,
        ..Default::default()
    }
}

/// Register a TLS plot onto ALS; see the module documentation.
///
/// `tls` and `als` are points with a ground flag each (ASPRS class 2);
/// `initial` maps the TLS coordinates roughly into the ALS frame.
///
/// # Errors
/// For bad settings, too few ground points in either cloud, or no pose at
/// which enough of the TLS canopy model falls on ALS data.
pub fn register(tls: &[Point], tls_ground: &[bool], als: &[Point], als_ground: &[bool], initial: &Matrix4<f64>, p: &RegisterParams) -> Result<Registration> {
    p.check()?;
    if tls.len() != tls_ground.len() || als.len() != als_ground.len() {
        return Err(Error::invalid("one ground flag per point is needed"));
    }
    if initial.iter().any(|v| !v.is_finite()) {
        return Err(Error::invalid("the initial transform must be finite"));
    }
    if tls.iter().chain(als).any(|q| !q.iter().all(|v| v.is_finite())) {
        return Err(Error::invalid("point coordinates must be finite"));
    }
    // Thin the TLS (ground and the rest apart) to a tenth of a fine cell:
    // nothing below needs more, and a plot of tens of millions of points
    // shrinks to a few million.
    let voxel = (0.2 * p.resolution).min(0.1);
    let (g_in, v_in): (Vec<Point>, Vec<Point>) = {
        let g: Vec<Point> = tls.iter().zip(tls_ground).filter(|(_, &k)| k).map(|(q, _)| *q).collect();
        let v: Vec<Point> = tls.iter().zip(tls_ground).filter(|(_, &k)| !k).map(|(q, _)| *q).collect();
        rayon::join(|| voxel_downsample(&g, voxel), || voxel_downsample(&v, voxel))
    };
    let tls_ground: Vec<bool> = std::iter::repeat_n(true, g_in.len()).chain(std::iter::repeat_n(false, v_in.len())).collect();
    let tls_ground = tls_ground.as_slice();
    let thinned: Vec<Point> = g_in.into_iter().chain(v_in).collect();
    // Pivot: the centre of the TLS ground after `initial`, as a local origin.
    let placed = apply(initial, &thinned);
    drop(thinned);
    let g: Vec<&Point> = placed.iter().zip(tls_ground).filter(|(_, &k)| k).map(|(q, _)| q).collect();
    if g.len() < 3 {
        return Err(Error::invalid("the TLS cloud has fewer than 3 ground points (classification 2)"));
    }
    let n = g.len() as f64;
    let mut zs: Vec<f64> = g.iter().map(|q| q[2]).collect();
    let pivot = [g.iter().map(|q| q[0]).sum::<f64>() / n, g.iter().map(|q| q[1]).sum::<f64>() / n, median(&mut zs)];
    let to_local = translation([-pivot[0], -pivot[1], -pivot[2]]);
    let from_local = translation(pivot);
    let tls_local = apply(&to_local, &placed);
    drop(placed);
    // TLS extent and the ALS points that can matter.
    let tb = bounds_of(&tls_local);
    let reach = tb.iter().map(|v| v.abs()).fold(0.0, f64::max) * std::f64::consts::SQRT_2 + p.search_radius + 2.0 * p.coarse_resolution + 5.0;
    let (mut als_local, mut als_g) = (Vec::new(), Vec::new());
    for (q, &k) in als.iter().zip(als_ground) {
        let l = [q[0] - pivot[0], q[1] - pivot[1], q[2] - pivot[2]];
        if l[0].abs() <= reach && l[1].abs() <= reach {
            als_local.push(l);
            als_g.push(k);
        }
    }
    if als_local.len() < 10 {
        return Err(Error::invalid("fewer than 10 ALS points lie within reach of the TLS plot; check `initial` and `search_radius`"));
    }
    let ab = [-reach, -reach, reach, reach];
    // Density over the 5 m cells that hold returns; sparse surveys get
    // wider fine cells.
    let occupied: std::collections::HashSet<(i64, i64)> = als_local.iter().map(|q| ((q[0] / 5.0).floor() as i64, (q[1] / 5.0).floor() as i64)).collect();
    let density = als_local.len() as f64 / (25.0 * occupied.len().max(1) as f64);
    let mut p = p.clone();
    if p.returns_per_cell > 0.0 {
        p.resolution = p.resolution.max((p.returns_per_cell / density).sqrt());
        p.coarse_resolution = p.coarse_resolution.max(p.resolution);
    }
    let p = &p;
    let level = |res: f64| -> Result<Level> {
        let t = layers(&tls_local, tls_ground, res, tb, p.min_height, "TLS")?;
        let a = layers(&als_local, &als_g, res, ab, p.min_height, "ALS")?;
        Ok(Level { chm_tpl: template(&t.chm), dtm_tpl: template(&t.dtm), als: a })
    };
    let coarse = level(p.coarse_resolution)?;
    let peaks = coarse_search(&coarse, p);
    drop(coarse);
    if peaks.is_empty() {
        return Err(Error::invalid("no pose puts enough of the TLS canopy model on ALS data; check `initial`, `search_radius` and that the clouds overlap"));
    }
    let fine = level(p.resolution)?;
    let mut refined: Vec<Candidate> = peaks.iter().map(|c| fine_search(&fine, c, p)).collect();
    refined.sort_by(better);
    let mut distinct: Vec<Candidate> = Vec::new();
    for c in refined {
        if distinct.iter().all(|q| (c.dx - q.dx).hypot(c.dy - q.dy) > 2.0 * p.resolution || angle_diff(c.heading, q.heading) > p.heading_step / 3.0) {
            distinct.push(c);
        }
    }
    let best = distinct[0];
    let ambiguity = if distinct.len() > 1 { let den = 1.0 - distinct[1].score; if den > 0.0 { ((1.0 - best.score) / den).clamp(0.0, 1.0) } else { 1.0 } } else { f64::NAN };
    // Each peak as a pose, its height from the median difference of the terrains.
    let pose_of = |c: &Candidate| -> Matrix4<f64> {
        let (s, co) = c.heading.to_radians().sin_cos();
        let mut dz: Vec<f64> = fine.dtm_tpl.iter().map(|t| lookup(&fine.als.dtm_filled, co * t[0] - s * t[1] + c.dx, s * t[0] + co * t[1] + c.dy, true) - t[2]).filter(|v| v.is_finite()).collect();
        let dz = if dz.is_empty() { 0.0 } else { median(&mut dz) };
        rz(c.heading, [c.dx, c.dy, dz])
    };
    let search_local = pose_of(&best);
    // Refinement: an ICP from every peak; the best fit among those that
    // stayed near their start wins.
    let mut local = search_local;
    let (mut icp_summary, mut covariance, mut jackknife) = (None, None, None);
    if p.refine != Refine::None {
        let cfg = icp_config(p);
        let finest = *p.icp_voxel_sizes.last().unwrap();
        let pick = |pts: &[Point], gr: &[bool]| -> Vec<Point> {
            let v: Vec<Point> = pts.iter().zip(gr).filter(|(_, &k)| p.refine == Refine::All || k).map(|(q, _)| *q).collect();
            voxel_downsample(&v, 0.5 * finest)
        };
        let src = pick(&tls_local, tls_ground);
        let moved = apply(&search_local, &src);
        let mb = bounds_of(&moved);
        let margin = p.icp_max_distances[0] + 2.0 + p.max_refine_shift;
        let dst_all: Vec<Point> = pick(&als_local, &als_g).into_iter().filter(|q| q[0] >= mb[0] - margin && q[0] <= mb[2] + margin && q[1] >= mb[1] - margin && q[1] <= mb[3] + margin).collect();
        if src.len() >= 10 && dst_all.len() >= 10 {
            let target = IcpTarget::new(&dst_all, &cfg);
            let starts: Vec<Matrix4<f64>> = distinct.iter().map(&pose_of).collect();
            let runs: Vec<Result<(IcpSummary, icp::IcpResult)>> = starts
                .par_iter()
                .zip(&distinct)
                .map(|(start, c)| {
                    let r = icp::icp_prepared(&src, &target, Some(*start), &cfg)?;
                    let moved_by = (r.transform[(0, 3)] - start[(0, 3)]).hypot(r.transform[(1, 3)] - start[(1, 3)]);
                    let turn = angle_diff(heading_of(&r.transform), c.heading);
                    let accepted = r.n_correspondences >= 10 && moved_by <= p.max_refine_shift && turn <= p.max_refine_turn && r.transform.iter().all(|v| v.is_finite());
                    Ok((IcpSummary { fitness: r.fitness, rmse: r.inlier_rmse, n: r.n_correspondences, iterations: r.iterations, converged: r.converged, accepted, shift: moved_by, turn, start: 0 }, r))
                })
                .collect();
            let mut runs: Vec<(IcpSummary, icp::IcpResult)> = runs.into_iter().collect::<Result<_>>()?;
            for (k, r) in runs.iter_mut().enumerate() {
                r.0.start = k;
            }
            // The first peak (best search score first) whose accepted run fits
            // within half a percent of the best accepted fit: runs that reach
            // the same pose differ in fitness only by chance.
            let best_fit = runs.iter().filter(|r| r.0.accepted).map(|r| r.0.fitness).fold(f64::NEG_INFINITY, f64::max);
            let chosen = runs.iter().position(|r| r.0.accepted && r.0.fitness >= best_fit - 0.005).unwrap_or(0);
            let (summary, r) = runs.swap_remove(chosen);
            if summary.accepted {
                local = r.transform;
                if let Some(info) = &r.information {
                    covariance = info.hessian.try_inverse().map(|h| h * info.sigma * info.sigma);
                }
                if p.jackknife {
                    let parts: Vec<Option<[f64; 4]>> = (0..4)
                        .into_par_iter()
                        .map(|q| {
                            let sub: Vec<Point> = src.iter().filter(|s| ((s[0] >= 0.0) as usize) + 2 * ((s[1] >= 0.0) as usize) != q).copied().collect();
                            if sub.len() < 10 {
                                return None;
                            }
                            let j = icp::icp_prepared(&sub, &target, Some(local), &cfg).ok()?;
                            Some([j.transform[(0, 3)], j.transform[(1, 3)], j.transform[(2, 3)], heading_of(&j.transform)])
                        })
                        .collect();
                    let parts: Vec<[f64; 4]> = parts.into_iter().flatten().collect();
                    if parts.len() >= 2 {
                        let k = parts.len() as f64;
                        let h0 = heading_of(&local);
                        let mut se = [0.0; 4];
                        for (d, s) in se.iter_mut().enumerate() {
                            let v: Vec<f64> = parts.iter().map(|q| if d == 3 { wrap(q[3] - h0) } else { q[d] }).collect();
                            let m = v.iter().sum::<f64>() / k;
                            *s = ((k - 1.0) / k * v.iter().map(|x| (x - m) * (x - m)).sum::<f64>()).sqrt();
                        }
                        jackknife = Some(se);
                    }
                }
            }
            icp_summary = Some(summary);
        }
    }
    // Residuals against the ALS terrain at the fine resolution.
    let final_tls = apply(&local, &tls_local);
    let residuals = residuals(&final_tls, tls_ground, &als_local, &als_g, &fine.als.dtm_filled, p.min_height);
    let transform = from_local * local * to_local * initial;
    let search_transform = from_local * search_local * to_local * initial;
    Ok(Registration {
        resolution: p.resolution,
        coarse_resolution: p.coarse_resolution,
        als_density: density,
        transform,
        search_transform,
        pivot,
        heading: heading_of(&local),
        shift: [local[(0, 3)], local[(1, 3)], local[(2, 3)]],
        candidates: distinct,
        ambiguity,
        icp: icp_summary,
        covariance,
        jackknife,
        residuals,
        n_tls: tls.len(),
        n_als: als_local.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn terrain(x: f64, y: f64) -> f64 {
        0.03 * x + 0.4 * (y / 7.0).sin() + 0.3 * (x / 5.0).cos()
    }

    /// Ground and a few dozen crowns (hemispheres of points) of mixed sizes,
    /// with their ground flags.
    fn scene(x0: f64, y0: f64, size: f64, n_ground: usize, seed: u64) -> (Vec<Point>, Vec<bool>) {
        let mut s = seed;
        let mut pts = Vec::new();
        let mut g = Vec::new();
        for _ in 0..n_ground {
            let (x, y) = (x0 + lcg(&mut s) * size, y0 + lcg(&mut s) * size);
            pts.push([x, y, terrain(x, y)]);
            g.push(true);
        }
        // Crowns on a fixed pattern (the same for every call), each a
        // hemisphere of radius r at height h.
        let mut t = 7u64;
        for _ in 0..60 {
            let (cx, cy) = (-40.0 + 80.0 * lcg(&mut t), -40.0 + 80.0 * lcg(&mut t));
            let (r, h) = (1.5 + 3.0 * lcg(&mut t), 8.0 + 15.0 * lcg(&mut t));
            if cx < x0 || cx > x0 + size || cy < y0 || cy > y0 + size {
                continue;
            }
            for _ in 0..(40.0 * r * r) as usize {
                let a = lcg(&mut s) * std::f64::consts::TAU;
                let u = lcg(&mut s);
                let rr = r * u.sqrt();
                let (x, y) = (cx + rr * a.cos(), cy + rr * a.sin());
                let z = terrain(cx, cy) + h - r + (r * r - rr * rr).max(0.0).sqrt();
                pts.push([x, y, z]);
                g.push(false);
            }
        }
        (pts, g)
    }

    #[test]
    fn recovers_a_known_pose() {
        let (als, als_g) = scene(-40.0, -40.0, 80.0, 60_000, 1);
        let (plot, plot_g) = scene(-15.0, -15.0, 30.0, 20_000, 2);
        // The TLS frame: the plot turned by 25 degrees and shifted.
        let truth = rz(25.0, [3.2, -2.1, 0.8]);
        let tls = apply(&truth.try_inverse().unwrap(), &plot);
        let p = RegisterParams { search_radius: 6.0, heading_range: 40.0, heading_step: 2.0, ..Default::default() };
        let r = register(&tls, &plot_g, &als, &als_g, &Matrix4::identity(), &p).unwrap();
        let err = r.transform * truth.try_inverse().unwrap();
        let (dx, dy, dz) = (err[(0, 3)], err[(1, 3)], err[(2, 3)]);
        assert!(dx.hypot(dy) < 0.1 && dz.abs() < 0.05, "{dx} {dy} {dz} {r:?}");
        assert!(heading_of(&err).abs() < 0.2, "{}", heading_of(&err));
        assert!(r.residuals.ground_nmad < 0.1, "{:?}", r.residuals);
        assert!(r.covariance.is_some() && r.jackknife.is_some());
        // Large map coordinates change nothing but the numbers.
        let off = translation([512_345.0, 6_912_345.0, 100.0]);
        let als_map = apply(&off, &als);
        let r2 = register(&tls, &plot_g, &als_map, &als_g, &off, &p).unwrap();
        let err2 = off.try_inverse().unwrap() * r2.transform * truth.try_inverse().unwrap();
        assert!(err2[(0, 3)].hypot(err2[(1, 3)]) < 0.1 && err2[(2, 3)].abs() < 0.05, "{err2}");
    }

    #[test]
    fn a_full_circle_search_finds_a_large_turn() {
        let (als, als_g) = scene(-40.0, -40.0, 80.0, 60_000, 3);
        let (plot, plot_g) = scene(-15.0, -15.0, 30.0, 20_000, 4);
        let truth = rz(-140.0, [1.0, 4.0, -0.5]);
        let tls = apply(&truth.try_inverse().unwrap(), &plot);
        let p = RegisterParams { search_radius: 6.0, refine: Refine::None, ..Default::default() };
        let r = register(&tls, &plot_g, &als, &als_g, &Matrix4::identity(), &p).unwrap();
        let err = r.transform * truth.try_inverse().unwrap();
        assert!(err[(0, 3)].hypot(err[(1, 3)]) < 0.3 && heading_of(&err).abs() < 0.5, "{err}");
        assert!(r.icp.is_none() && r.covariance.is_none());
        assert!(r.candidates[0].score > 0.8);
    }

    #[test]
    fn bad_input_is_refused() {
        let (als, als_g) = scene(-40.0, -40.0, 80.0, 1_000, 5);
        let p = RegisterParams::default();
        let none = vec![false; als.len()];
        assert!(register(&als, &none, &als, &als_g, &Matrix4::identity(), &p).is_err());
        assert!(register(&als, &als_g, &als, &als_g[..10], &Matrix4::identity(), &p).is_err());
        let bad = RegisterParams { resolution: 0.0, ..Default::default() };
        assert!(register(&als, &als_g, &als, &als_g, &Matrix4::identity(), &bad).is_err());
        let far = translation([1e4, 0.0, 0.0]);
        assert!(register(&als, &als_g, &als, &als_g, &far, &p).is_err());
        assert!(Refine::parse("icp").is_err());
    }

    #[test]
    fn helpers() {
        assert_eq!(angle_diff(179.0, -179.0), 2.0);
        assert_eq!(wrap(190.0), -170.0);
        assert!((parabola(0.0, 1.0, 0.0)).abs() < 1e-12);
        assert!((parabola(0.5, 1.0, 0.0) + 1.0 / 6.0).abs() < 1e-12);
        let hs = headings(&RegisterParams { heading_range: 10.0, heading_step: 3.0, ..Default::default() });
        assert_eq!(hs.len(), 9);
        assert!((hs[0] + 10.0).abs() < 1e-12 && (hs[8] - 10.0).abs() < 1e-12);
        let r = Raster { data: vec![1.0, 2.0, 3.0, 4.0], nrows: 2, ncols: 2, xmin: 0.0, ymin: 0.0, resolution: 1.0 };
        assert_eq!(lookup(&r, 1.5, 0.5, false), 2.0);
        assert!((lookup(&r, 1.0, 1.0, true) - 2.5).abs() < 1e-12);
        assert!(lookup(&r, -0.1, 0.5, false).is_nan() && lookup(&r, 0.2, 0.5, true).is_nan());
        let tpl = vec![[0.5, 0.5, 1.0], [1.5, 0.5, 2.0], [0.5, 1.5, 3.0]];
        let (c, n) = correlation(&tpl, &r, 0.0, 0.0, 0.0, false);
        assert_eq!(n, 3);
        assert!((c - 1.0).abs() < 1e-12);
    }
}
