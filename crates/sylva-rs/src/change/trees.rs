// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Trees matched between two epochs, and their increments.
//!
//! Matching is an optimal assignment (Kuhn 1955; Munkres 1957) on stem
//! positions, with DBH and height as tie-breakers, solved separately in
//! each group of trees that could be matched to one another. A pair is
//! allowed only within `max_distance`, and only when the second DBH is
//! at most `dbh_tolerance` larger and `max_shrink` smaller (relative to
//! the larger), so that a felled tree and a young one grown beside it are
//! not taken for one tree. Stems left over are then
//! checked for a merge (two stems of the first epoch found as one in the
//! second) or a split (the reverse) before they are called deaths or
//! recruits.
//!
//! The DBH increment of a survivor is measured on the stem profile: circles
//! are fitted to thin slices between 1 and 3 m in each epoch, and the
//! increment is the weighted mean of the paired differences at the same
//! heights, which cancels the stem's own irregularity. Its standard error
//! combines each slice's circle-fit precision (at least the range noise of
//! the epoch), inflated by the scatter of the differences between slices
//! where that exceeds the expected, and the vertical registration
//! uncertainty times the stem taper. The minimum detectable increment is
//! that standard error times the normal quantile of the confidence level.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::trees::{crown_metrics_all, crown_shape, fit_circle, fit_circle_ransac, RansacCircleParams};
use crate::Point;
use super::{positive, non_negative};

/// A tree as matching sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeRow {
    pub x: f64,
    pub y: f64,
    pub dbh: f64,
    pub height: f64,
}

/// Settings of [`match_trees`].
#[derive(Debug, Clone)]
pub struct MatchParams {
    /// Farthest two stems of one tree can be apart (m).
    pub max_distance: f64,
    /// Largest relative DBH increase `(b - a) / max(a, b)` of one tree.
    pub dbh_tolerance: f64,
    /// Largest relative DBH decrease `(a - b) / max(a, b)` of one tree:
    /// stems do not shrink, so this only absorbs measurement error.
    pub max_shrink: f64,
    /// Weights of the squared relative DBH and height differences in the
    /// cost, next to the squared distance over `max_distance`.
    pub dbh_weight: f64,
    pub height_weight: f64,
    /// A stem explains a neighbour that lost its match (a merge or a split)
    /// when its DBH is at least this fraction of the quadratic sum of the
    /// DBHs it would stand for.
    pub merge_factor: f64,
}

impl Default for MatchParams {
    fn default() -> Self {
        MatchParams { max_distance: 1.0, dbh_tolerance: 0.35, max_shrink: 0.15, dbh_weight: 1.0, height_weight: 0.25, merge_factor: 0.8 }
    }
}

/// Status of a tree of the first epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusA {
    Survivor,
    Dead,
    /// Found in the second epoch only as part of a neighbour's stem.
    Merged,
    /// Found in the second epoch as several stems, none matched to it.
    Split,
}

/// Status of a tree of the second epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusB {
    Survivor,
    Recruit,
    /// Part of a stem that the first epoch saw as one with a neighbour.
    Split,
    /// A stem, not matched itself, that stands for several stems of the
    /// first epoch.
    Merged,
}

impl StatusA {
    pub fn name(self) -> &'static str {
        match self {
            StatusA::Survivor => "survivor",
            StatusA::Dead => "dead",
            StatusA::Merged => "merged",
            StatusA::Split => "split",
        }
    }
}

impl StatusB {
    pub fn name(self) -> &'static str {
        match self {
            StatusB::Survivor => "survivor",
            StatusB::Recruit => "recruit",
            StatusB::Split => "split",
            StatusB::Merged => "merged",
        }
    }
}

/// Result of [`match_trees`]; indices refer to the input slices.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeMatch {
    /// `(a, b)` of every survivor, sorted by `a`.
    pub pairs: Vec<(usize, usize)>,
    /// Horizontal distance of each pair (m).
    pub distance: Vec<f64>,
    pub cost: Vec<f64>,
    pub status_a: Vec<StatusA>,
    pub status_b: Vec<StatusB>,
    /// For a merged tree of the first epoch, the stem of the second it went
    /// into; for a split tree of the second, the stem of the first it came
    /// from; otherwise the matched partner, or `None`.
    pub related_a: Vec<Option<usize>>,
    pub related_b: Vec<Option<usize>>,
}

/// Relative difference `(b - a) / max(a, b)`; 0 when either is unknown.
fn relative(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() && a.max(b) > 0.0 {
        (b - a) / a.max(b)
    } else {
        0.0
    }
}

/// Minimum-cost assignment of the rows of a `n x m` cost matrix (`n <= m`)
/// to distinct columns: the shortest augmenting path form of the Hungarian
/// method, O(n² m). Returns the column of each row.
pub fn assign(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    assert!(n <= m, "assign needs no more rows than columns");
    let inf = f64::INFINITY;
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; m + 1]);
    let (mut p, mut way) = (vec![0usize; m + 1], vec![0usize; m + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut delta, mut j1) = (inf, 0usize);
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![0usize; n];
    for j in 1..=m {
        if p[j] > 0 {
            out[p[j] - 1] = j - 1;
        }
    }
    out
}

fn find(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
        r = parent[r];
    }
    let mut k = i;
    while parent[k] != r {
        let next = parent[k];
        parent[k] = r;
        k = next;
    }
    r
}

/// Match the trees of two epochs given in one frame; see the module
/// documentation.
pub fn match_trees(a: &[TreeRow], b: &[TreeRow], p: &MatchParams) -> Result<TreeMatch> {
    if !positive(p.max_distance) || !positive(p.dbh_tolerance) || !non_negative(p.max_shrink) || !positive(p.merge_factor) || !non_negative(p.dbh_weight) || !non_negative(p.height_weight) {
        return Err(Error::invalid("max_distance, dbh_tolerance and merge_factor must be positive, max_shrink and the weights >= 0"));
    }
    if a.iter().chain(b).any(|t| !t.x.is_finite() || !t.y.is_finite()) {
        return Err(Error::invalid("tree positions must be finite"));
    }
    let (na, nb) = (a.len(), b.len());
    // Allowed pairs, found through a grid of max_distance cells.
    let cell = p.max_distance;
    let key = |t: &TreeRow| ((t.x / cell).floor() as i64, (t.y / cell).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (j, t) in b.iter().enumerate() {
        grid.entry(key(t)).or_default().push(j);
    }
    let mut edges: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (i, t) in a.iter().enumerate() {
        let (kx, ky) = key(t);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &j in grid.get(&(kx + dx, ky + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                    let d = (t.x - b[j].x).hypot(t.y - b[j].y);
                    let rd = relative(t.dbh, b[j].dbh);
                    if d <= p.max_distance && rd <= p.dbh_tolerance && rd >= -p.max_shrink {
                        let rh = relative(t.height, b[j].height);
                        let c = (d / p.max_distance).powi(2) + p.dbh_weight * rd * rd + p.height_weight * rh * rh;
                        edges.push((i, j, d, c));
                    }
                }
            }
        }
    }
    edges.sort_by_key(|e| (e.0, e.1));
    // Groups of trees linked by allowed pairs, each assigned on its own.
    let mut parent: Vec<usize> = (0..na + nb).collect();
    for &(i, j, _, _) in &edges {
        let (ri, rj) = (find(&mut parent, i), find(&mut parent, na + j));
        if ri != rj {
            parent[ri.max(rj)] = ri.min(rj);
        }
    }
    let mut groups: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
    for i in 0..na {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().0.push(i);
    }
    for j in 0..nb {
        let r = find(&mut parent, na + j);
        groups.entry(r).or_default().1.push(j);
    }
    let edge_of: HashMap<(usize, usize), (f64, f64)> = edges.iter().map(|&(i, j, d, c)| ((i, j), (d, c))).collect();
    let mut roots: Vec<usize> = groups.keys().copied().collect();
    roots.sort_unstable();
    // Forbidden pairs cost more than all allowed ones together, so the
    // assignment first matches as many trees as it can.
    const FORBIDDEN: f64 = 1e9;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for r in roots {
        let (ga, gb) = &groups[&r];
        if ga.is_empty() || gb.is_empty() {
            continue;
        }
        let transpose = ga.len() > gb.len();
        let (rows, cols) = if transpose { (gb, ga) } else { (ga, gb) };
        let cost: Vec<Vec<f64>> = rows
            .iter()
            .map(|&ri| cols.iter().map(|&ci| {
                let (i, j) = if transpose { (ci, ri) } else { (ri, ci) };
                edge_of.get(&(i, j)).map(|e| e.1).unwrap_or(FORBIDDEN)
            }).collect())
            .collect();
        for (k, c) in assign(&cost).into_iter().enumerate() {
            if cost[k][c] < FORBIDDEN {
                let (i, j) = if transpose { (cols[c], rows[k]) } else { (rows[k], cols[c]) };
                pairs.push((i, j));
            }
        }
    }
    pairs.sort_unstable();
    let mut status_a = vec![StatusA::Dead; na];
    let mut status_b = vec![StatusB::Recruit; nb];
    let mut related_a = vec![None; na];
    let mut related_b = vec![None; nb];
    let (mut distance, mut cost) = (Vec::with_capacity(pairs.len()), Vec::with_capacity(pairs.len()));
    for &(i, j) in &pairs {
        status_a[i] = StatusA::Survivor;
        status_b[j] = StatusB::Survivor;
        related_a[i] = Some(j);
        related_b[j] = Some(i);
        let (d, c) = edge_of[&(i, j)];
        distance.push(d);
        cost.push(c);
    }
    // Merges: a stem of epoch 2 wide enough to stand for its partner (if
    // any) and every unmatched stem of epoch 1 within reach.
    let near = |t: &TreeRow, u: &TreeRow| (t.x - u.x).hypot(t.y - u.y) <= p.max_distance;
    let quad = |v: &[f64]| v.iter().map(|d| d * d).sum::<f64>().sqrt();
    for j in 0..nb {
        let lost: Vec<usize> = (0..na).filter(|&i| status_a[i] == StatusA::Dead && near(&a[i], &b[j])).collect();
        if lost.is_empty() || !b[j].dbh.is_finite() {
            continue;
        }
        let mut members: Vec<f64> = lost.iter().map(|&i| a[i].dbh).collect();
        if let Some(i) = related_b[j] {
            members.push(a[i].dbh);
        }
        if members.len() >= 2 && members.iter().all(|d| d.is_finite()) && b[j].dbh >= p.merge_factor * quad(&members) {
            for &i in &lost {
                status_a[i] = StatusA::Merged;
                related_a[i] = Some(j);
            }
            if status_b[j] == StatusB::Recruit {
                status_b[j] = StatusB::Merged;
                related_b[j] = Some(lost[0]);
            }
        }
    }
    // Splits, the mirror image.
    for i in 0..na {
        let new: Vec<usize> = (0..nb).filter(|&j| status_b[j] == StatusB::Recruit && near(&a[i], &b[j])).collect();
        if new.is_empty() || !a[i].dbh.is_finite() {
            continue;
        }
        let mut members: Vec<f64> = new.iter().map(|&j| b[j].dbh).collect();
        if let Some(j) = related_a[i].filter(|_| status_a[i] == StatusA::Survivor) {
            members.push(b[j].dbh);
        }
        if members.len() >= 2 && members.iter().all(|d| d.is_finite()) && a[i].dbh >= p.merge_factor * quad(&members) {
            for &j in &new {
                status_b[j] = StatusB::Split;
                related_b[j] = Some(i);
            }
            if status_a[i] == StatusA::Dead {
                status_a[i] = StatusA::Split;
                related_a[i] = Some(new[0]);
            }
        }
    }
    Ok(TreeMatch { pairs, distance, cost, status_a, status_b, related_a, related_b })
}

// ------------------------------------------------------------ measurement

/// Settings of [`measure_trees`] and [`tree_increments`].
#[derive(Debug, Clone)]
pub struct IncrementParams {
    /// Heights (m above ground) of the stem slices.
    pub slice_heights: Vec<f64>,
    pub slice_thickness: f64,
    /// Only points this close to the stem centre (m) enter a slice.
    pub search_radius: f64,
    /// Fewest slices measured in both epochs for a DBH increment.
    pub min_slices: usize,
    /// Two-sided confidence of the minimum detectable increment.
    pub confidence: f64,
    /// DBH increments above this (m) are implausible; NaN for no limit.
    pub max_dbh_increment: f64,
    /// Height points at the top of a tree used for its height uncertainty.
    pub top_points: usize,
    /// Assumed one-sigma error of a measured top height, as a fraction of
    /// the height: the top of a tree can be missed (occluded or too thin
    /// to be hit) in a way no single epoch shows.
    pub height_error: f64,
    /// Unassigned points (label < 0) within this horizontal distance (m) of
    /// the stem that continue the tree upwards, with vertical gaps below
    /// `top_gap`, count towards its top: segmentation often leaves a thin
    /// leader unassigned.
    pub top_radius: f64,
    pub top_gap: f64,
    /// Distance (m) from the RANSAC circle within which slice points are
    /// inliers; NaN for `max(0.01, 3 sigma)`. Give both epochs the same
    /// value, so that their circles are fitted alike.
    pub inlier_threshold: f64,
    /// Correlation of the diameter errors of the slices of one stem in one
    /// epoch: the same scanners see every slice from the same directions,
    /// so edge effects and occlusion repeat up the stem and averaging over
    /// slices removes only part of the error.
    pub slice_correlation: f64,
    /// Absolute one-sigma accuracy (m) of a single-epoch DBH, beyond its
    /// precision: bark roughness and edge points widen every circle of a
    /// stem alike. It cancels in increments and is added only to `dbh_se`.
    pub dbh_accuracy: f64,
}

impl Default for IncrementParams {
    fn default() -> Self {
        IncrementParams { slice_heights: (0..9).map(|k| 1.0 + 0.25 * k as f64).collect(), slice_thickness: 0.1, search_radius: 0.75, min_slices: 3, confidence: 0.95, max_dbh_increment: f64::NAN, top_points: 5, height_error: 0.02, top_radius: 0.3, top_gap: 0.5, inlier_threshold: f64::NAN, slice_correlation: 0.3, dbh_accuracy: 0.002 }
    }
}

/// One epoch's measurement of one tree.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeMeasure {
    /// Diameter and its standard error at each slice height (NaN if none).
    pub diameter: Vec<f64>,
    pub diameter_se: Vec<f64>,
    /// DBH at 1.3 m from a weighted line through the slice diameters.
    pub dbh: f64,
    pub dbh_se: f64,
    /// Change of diameter per metre of height (negative for a tapering stem).
    pub taper: f64,
    /// Highest point above ground and its uncertainty.
    pub height: f64,
    pub height_se: f64,
    pub crown_area: f64,
    pub crown_volume: f64,
    pub n_points: usize,
}

/// The standard normal quantile (Acklam's rational approximation, relative
/// error below 1.2e-9).
#[allow(clippy::excessive_precision)]
pub fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    if p.is_nan() || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    let lo = 0.02425;
    if p < lo {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - lo {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -normal_quantile(1.0 - p)
    }
}

/// Circle fit of one slice: diameter, its standard error, or NaN.
fn slice_diameter(xy: &[[f64; 2]], sigma: f64, threshold: f64, max_radius: f64) -> (f64, f64) {
    if xy.len() < 10 {
        return (f64::NAN, f64::NAN);
    }
    let ransac = RansacCircleParams { threshold, max_radius, ..Default::default() };
    let Ok((_, _, _, inl)) = fit_circle_ransac(xy, &ransac) else { return (f64::NAN, f64::NAN) };
    let pts: Vec<[f64; 2]> = xy.iter().zip(&inl).filter(|(_, &k)| k).map(|(q, _)| *q).collect();
    if pts.len() < 10 {
        return (f64::NAN, f64::NAN);
    }
    let Ok((cx2, cy2, r, rmse)) = fit_circle(&pts) else { return (f64::NAN, f64::NAN) };
    if !r.is_finite() || r > max_radius {
        return (f64::NAN, f64::NAN);
    }
    // Share of the circumference seen, in 10-degree sectors.
    let mut seen = [false; 36];
    for q in &pts {
        let a = (q[1] - cy2).atan2(q[0] - cx2).to_degrees().rem_euclid(360.0);
        seen[((a / 10.0) as usize).min(35)] = true;
    }
    let coverage = seen.iter().filter(|&&s| s).count() as f64 / 36.0;
    let s = rmse.max(sigma);
    // Radius precision of a circle fitted to n points with radial noise s
    // is s / sqrt(n) on a full circle; a partial arc is less well
    // conditioned.
    (2.0 * r, 2.0 * s / (pts.len() as f64).sqrt() / coverage.max(0.1).sqrt())
}

/// Weighted least-squares line `d = c0 + c1 (h - 1.3)`: value at 1.3, its
/// standard error and the slope. Standard errors are scaled by the Birge
/// ratio where the points scatter more than their errors.
fn line_at_breast_height(h: &[f64], d: &[f64], se: &[f64]) -> (f64, f64, f64) {
    let rows: Vec<(f64, f64, f64)> = (0..h.len()).filter(|&k| d[k].is_finite() && se[k] > 0.0).map(|k| (h[k] - 1.3, d[k], 1.0 / (se[k] * se[k]))).collect();
    if rows.len() < 2 {
        return rows.first().map(|r| (r.1, 1.0 / r.2.sqrt(), f64::NAN)).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    }
    let (sw, sx, sy, sxx, sxy) = rows.iter().fold((0.0, 0.0, 0.0, 0.0, 0.0), |a, r| (a.0 + r.2, a.1 + r.2 * r.0, a.2 + r.2 * r.1, a.3 + r.2 * r.0 * r.0, a.4 + r.2 * r.0 * r.1));
    let det = sw * sxx - sx * sx;
    if det.abs() < 1e-300 {
        return (sy / sw, (1.0 / sw).sqrt(), f64::NAN);
    }
    let c1 = (sw * sxy - sx * sy) / det;
    let c0 = (sy - c1 * sx) / sw;
    let var0 = sxx / det;
    let chi2: f64 = rows.iter().map(|r| r.2 * (r.1 - c0 - c1 * r.0).powi(2)).sum::<f64>();
    let dof = rows.len() as f64 - 2.0;
    let birge = if dof > 0.0 { (chi2 / dof).max(1.0) } else { 1.0 };
    (c0, (var0 * birge).sqrt(), c1)
}

/// Harmonic mean of the squared standard errors of the measured slices:
/// the error variance of a mean over fully correlated slices.
fn harmonic_variance(d: &[f64], se: &[f64]) -> f64 {
    let w: Vec<f64> = (0..d.len()).filter(|&k| d[k].is_finite() && se[k] > 0.0).map(|k| 1.0 / (se[k] * se[k])).collect();
    if w.is_empty() {
        f64::NAN
    } else {
        w.len() as f64 / w.iter().sum::<f64>()
    }
}

/// Drop slices that do not fit the stem: a slice cut through a branch
/// junction or a neighbour fits a circle far off the taper line of the
/// others. The worst slice is dropped while it lies more than 4 standard
/// errors and 1 cm from the line through the rest, keeping at least three.
fn reject_off_line(h: &[f64], d: &mut [f64], se: &mut [f64]) {
    loop {
        let valid: Vec<usize> = (0..d.len()).filter(|&k| d[k].is_finite()).collect();
        if valid.len() <= 3 {
            return;
        }
        let mut worst = (0.0, usize::MAX);
        for &k in &valid {
            // The line through the other slices, so an outlier cannot pull
            // the line towards itself.
            let (hh, dd, ss): (Vec<f64>, Vec<f64>, Vec<f64>) = valid.iter().filter(|&&q| q != k).map(|&q| (h[q], d[q], se[q])).fold((vec![], vec![], vec![]), |mut a, x| {
                a.0.push(x.0);
                a.1.push(x.1);
                a.2.push(x.2);
                a
            });
            let (c0, c0_se, c1) = line_at_breast_height(&hh, &dd, &ss);
            let fit = c0 + if c1.is_finite() { c1 * (h[k] - 1.3) } else { 0.0 };
            let r = (d[k] - fit).abs();
            let z = r / (se[k] * se[k] + c0_se * c0_se).sqrt();
            if r > 0.01 && z > 4.0 && z > worst.0 {
                worst = (z, k);
            }
        }
        if worst.1 == usize::MAX {
            return;
        }
        d[worst.1] = f64::NAN;
        se[worst.1] = f64::NAN;
    }
}

/// Measure every tree of one epoch: stem profile, DBH, top height and
/// crown. `trees` are `(tree_id, x, y)`; `labels` give each point's tree.
/// `sigma` is the epoch's range noise (m), a floor on the circle-fit
/// residual.
pub fn measure_trees(points: &[Point], heights: &[f64], labels: &[i64], trees: &[(i64, f64, f64)], sigma: f64, p: &IncrementParams) -> Result<Vec<TreeMeasure>> {
    if points.len() != heights.len() || points.len() != labels.len() {
        return Err(Error::invalid("points, heights and labels differ in length"));
    }
    if !positive(p.slice_thickness) || !positive(p.search_radius) || !non_negative(sigma) {
        return Err(Error::invalid("slice_thickness and search_radius must be positive and sigma >= 0"));
    }
    let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l >= 0 && heights[i].is_finite() {
            groups.entry(l).or_default().push(i);
        }
    }
    let crowns: HashMap<i64, [f64; 4]> = crown_metrics_all(points, heights, labels, 0.1).into_iter().collect();
    let empty = Vec::new();
    let threshold = if p.inlier_threshold.is_finite() { p.inlier_threshold } else { (3.0 * sigma).max(0.01) };
    // Unassigned points on a grid, for the tops segmentation left out.
    let cell = p.top_radius.max(0.1);
    let mut loose: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    if p.top_radius > 0.0 {
        for (i, &l) in labels.iter().enumerate() {
            if l < 0 && heights[i].is_finite() {
                loose.entry(((points[i][0] / cell).floor() as i64, (points[i][1] / cell).floor() as i64)).or_default().push(i);
            }
        }
    }
    Ok(trees
        .par_iter()
        .map(|&(id, cx, cy)| {
            let idx = groups.get(&id).unwrap_or(&empty);
            let half = p.slice_thickness / 2.0;
            let (mut diameter, mut diameter_se) = (Vec::with_capacity(p.slice_heights.len()), Vec::with_capacity(p.slice_heights.len()));
            for &hh in &p.slice_heights {
                let xy: Vec<[f64; 2]> = idx.iter().filter(|&&i| (heights[i] - hh).abs() <= half && (points[i][0] - cx).hypot(points[i][1] - cy) <= p.search_radius).map(|&i| [points[i][0], points[i][1]]).collect();
                let (d, se) = slice_diameter(&xy, sigma, threshold, p.search_radius);
                diameter.push(d);
                diameter_se.push(se);
            }
            reject_off_line(&p.slice_heights, &mut diameter, &mut diameter_se);
            let (dbh, dbh_se, taper) = line_at_breast_height(&p.slice_heights, &diameter, &diameter_se);
            let rho = p.slice_correlation;
            let dbh_se = (dbh_se * dbh_se * (1.0 - rho) + rho * harmonic_variance(&diameter, &diameter_se) + p.dbh_accuracy * p.dbh_accuracy).sqrt();
            // Top height: the highest point, continued upwards through
            // unassigned points above the stem; its uncertainty from how far
            // the next highest points lie below it, and the assumed error of
            // a missed top.
            let mut hs: Vec<f64> = idx.iter().map(|&i| heights[i]).collect();
            hs.sort_by(|x, y| y.total_cmp(x));
            if let Some(&top) = hs.first() {
                let (kx, ky) = ((cx / cell).floor() as i64, (cy / cell).floor() as i64);
                let mut above: Vec<f64> = Vec::new();
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for &i in loose.get(&(kx + dx, ky + dy)).map(|v| v.as_slice()).unwrap_or(&[]) {
                            if heights[i] > top && (points[i][0] - cx).hypot(points[i][1] - cy) <= p.top_radius {
                                above.push(heights[i]);
                            }
                        }
                    }
                }
                above.sort_by(f64::total_cmp);
                let mut reached = top;
                for h in above {
                    if h - reached > p.top_gap {
                        break;
                    }
                    reached = h;
                    hs.insert(0, h);
                }
            }
            let (height, height_se) = if hs.is_empty() {
                (f64::NAN, f64::NAN)
            } else {
                let k = p.top_points.max(2).min(hs.len());
                let spread = hs[0] - hs[k - 1];
                (hs[0], (spread.max(sigma).max(0.01).powi(2) + (p.height_error * hs[0]).powi(2)).sqrt())
            };
            let (crown_area, crown_volume) = match crowns.get(&id) {
                Some(m) => {
                    let pts: Vec<Point> = idx.iter().map(|&i| [points[i][0], points[i][1], heights[i]]).collect();
                    (m[0], crown_shape(&pts, Some([cx, cy]), m[1], 0.5).volume)
                }
                None => (f64::NAN, f64::NAN),
            };
            TreeMeasure { diameter, diameter_se, dbh, dbh_se, taper, height, height_se, crown_area, crown_volume, n_points: idx.len() }
        })
        .collect())
}

/// Classification of an increment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    /// At least the minimum detectable increment.
    Growth,
    /// Smaller in magnitude than the minimum detectable increment.
    BelowDetection,
    /// A decrease of at least the minimum detectable increment.
    Decrease,
    /// Not enough data in one of the epochs.
    Unmeasured,
}

impl Change {
    pub fn name(self) -> &'static str {
        match self {
            Change::Growth => "growth",
            Change::BelowDetection => "below_detection",
            Change::Decrease => "decrease",
            Change::Unmeasured => "unmeasured",
        }
    }
}

/// Increment of one survivor.
#[derive(Debug, Clone, PartialEq)]
pub struct Increment {
    pub d_dbh: f64,
    pub d_dbh_se: f64,
    pub d_dbh_mdi: f64,
    /// Slices measured in both epochs.
    pub n_slices: usize,
    pub dbh_change: Change,
    pub d_height: f64,
    pub d_height_se: f64,
    pub d_height_mdi: f64,
    pub height_change: Change,
    pub d_crown_area: f64,
    pub d_crown_volume: f64,
    /// A decrease in DBH beyond detection, or an increase above
    /// `max_dbh_increment`.
    pub implausible: bool,
}

fn classify(delta: f64, mdi: f64) -> Change {
    if !delta.is_finite() || !mdi.is_finite() {
        Change::Unmeasured
    } else if delta >= mdi {
        Change::Growth
    } else if delta <= -mdi {
        Change::Decrease
    } else {
        Change::BelowDetection
    }
}

/// Increment of one tree from its two measurements; `registration_sigma`
/// is the one-sigma vertical uncertainty (m) of the alignment of the epochs.
pub fn increment(a: &TreeMeasure, b: &TreeMeasure, registration_sigma: f64, p: &IncrementParams) -> Increment {
    let z = normal_quantile(0.5 + p.confidence / 2.0);
    let (mut sw, mut swd, mut rows) = (0.0, 0.0, Vec::<(f64, f64)>::new());
    for k in 0..a.diameter.len().min(b.diameter.len()) {
        let (da, db) = (a.diameter[k], b.diameter[k]);
        let var = a.diameter_se[k].powi(2) + b.diameter_se[k].powi(2);
        if da.is_finite() && db.is_finite() && var > 0.0 {
            sw += 1.0 / var;
            swd += (db - da) / var;
            rows.push((db - da, 1.0 / var));
        }
    }
    // Drop the worst paired difference while it is more than 4 of its own
    // standard errors from the mean of the others.
    while rows.len() > p.min_slices.max(3) {
        let mut worst = (0.0, usize::MAX);
        for (k, &(diff, w)) in rows.iter().enumerate() {
            let (sw_o, swd_o) = (sw - w, swd - diff * w);
            let z = (diff - swd_o / sw_o).abs() / (1.0 / w + 1.0 / sw_o).sqrt();
            if z > 4.0 && z > worst.0 {
                worst = (z, k);
            }
        }
        if worst.1 == usize::MAX {
            break;
        }
        let r = rows.remove(worst.1);
        sw -= r.1;
        swd -= r.0 * r.1;
    }
    let n = rows.len();
    let (d_dbh, d_dbh_se) = if n >= p.min_slices.max(1) {
        let mean = swd / sw;
        let chi2: f64 = rows.iter().map(|r| r.1 * (r.0 - mean).powi(2)).sum();
        let birge = if n > 1 { (chi2 / (n as f64 - 1.0)).max(1.0) } else { 1.0 };
        let taper = if a.taper.is_finite() { a.taper } else { 0.0 };
        let rho = p.slice_correlation;
        (mean, (birge * (1.0 - rho) / sw + rho * n as f64 / sw + (taper * registration_sigma).powi(2)).sqrt())
    } else {
        (f64::NAN, f64::NAN)
    };
    let d_dbh_mdi = z * d_dbh_se;
    let dbh_change = classify(d_dbh, d_dbh_mdi);
    let d_height = b.height - a.height;
    let d_height_se = (a.height_se.powi(2) + b.height_se.powi(2)).sqrt();
    let d_height_mdi = z * d_height_se;
    let too_fast = p.max_dbh_increment.is_finite() && d_dbh > p.max_dbh_increment;
    Increment {
        d_dbh,
        d_dbh_se,
        d_dbh_mdi,
        n_slices: n,
        dbh_change,
        d_height,
        d_height_se,
        d_height_mdi,
        height_change: classify(d_height, d_height_mdi),
        d_crown_area: b.crown_area - a.crown_area,
        d_crown_volume: b.crown_volume - a.crown_volume,
        implausible: dbh_change == Change::Decrease || too_fast,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(x: f64, y: f64, dbh: f64) -> TreeRow {
        TreeRow { x, y, dbh, height: f64::NAN }
    }

    #[test]
    fn assignment_is_optimal() {
        // Greedy would take (0, 0) at cost 1 and pay 10 for the rest.
        let cost = vec![vec![1.0, 2.0], vec![2.0, 10.0]];
        assert_eq!(assign(&cost), vec![1, 0]);
        let cost = vec![vec![4.0, 1.0, 3.0], vec![2.0, 0.0, 5.0], vec![3.0, 2.0, 2.0]];
        let a = assign(&cost);
        assert_eq!(a.iter().enumerate().map(|(i, &j)| cost[i][j]).sum::<f64>(), 5.0);
        assert_eq!(assign(&[vec![3.0, 1.0, 2.0]]), vec![1]);
    }

    #[test]
    fn survivors_deaths_recruits() {
        let a = [row(0.0, 0.0, 0.3), row(5.0, 0.0, 0.2), row(10.0, 0.0, 0.4)];
        let b = [row(0.1, 0.05, 0.31), row(10.2, 0.0, 0.41), row(20.0, 0.0, 0.1)];
        let m = match_trees(&a, &b, &MatchParams::default()).unwrap();
        assert_eq!(m.pairs, vec![(0, 0), (2, 1)]);
        assert_eq!(m.status_a, vec![StatusA::Survivor, StatusA::Dead, StatusA::Survivor]);
        assert_eq!(m.status_b, vec![StatusB::Survivor, StatusB::Survivor, StatusB::Recruit]);
    }

    #[test]
    fn crossing_neighbours_take_the_cheaper_assignment() {
        // Two stems 0.6 m apart; the second epoch saw both slightly moved
        // towards each other. Nearest-first would pair a0 with b1.
        let a = [row(0.0, 0.0, 0.3), row(0.6, 0.0, 0.2)];
        let b = [row(0.35, 0.0, 0.2), row(0.25, 0.0, 0.31)];
        let m = match_trees(&a, &b, &MatchParams::default()).unwrap();
        assert_eq!(m.pairs, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn felled_and_replaced_is_not_a_survivor() {
        let a = [row(0.0, 0.0, 0.4)];
        let b = [row(0.4, 0.0, 0.09)];
        let m = match_trees(&a, &b, &MatchParams::default()).unwrap();
        assert!(m.pairs.is_empty());
        assert_eq!((m.status_a[0], m.status_b[0]), (StatusA::Dead, StatusB::Recruit));
    }

    #[test]
    fn merged_and_split_stems() {
        // Two stems seen as one wide stem in the other epoch.
        let a = [row(0.0, 0.0, 0.2), row(0.4, 0.0, 0.18)];
        let b = [row(0.2, 0.0, 0.3)];
        let m = match_trees(&a, &b, &MatchParams::default()).unwrap();
        assert!(m.status_b[0] == StatusB::Survivor || m.status_b[0] == StatusB::Merged);
        assert!(m.status_a.contains(&StatusA::Merged) && !m.status_a.contains(&StatusA::Dead));
        let m = match_trees(&b, &a, &MatchParams::default()).unwrap();
        assert!(m.status_b.contains(&StatusB::Split) && !m.status_b.contains(&StatusB::Recruit));
        assert!(!m.status_a.contains(&StatusA::Dead));
    }

    #[test]
    fn rejects_bad_input() {
        assert!(match_trees(&[row(f64::NAN, 0.0, 0.2)], &[], &MatchParams::default()).is_err());
        assert!(match_trees(&[], &[], &MatchParams { max_distance: 0.0, ..Default::default() }).is_err());
        let m = match_trees(&[], &[], &MatchParams::default()).unwrap();
        assert!(m.pairs.is_empty());
    }

    #[test]
    fn quantile() {
        assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 1e-8);
        assert!((normal_quantile(0.01) + 2.326347874040841).abs() < 1e-8);
        assert!(normal_quantile(0.5).abs() < 1e-12);
    }

    /// A vertical stem of diameter `d` sampled all round at the slice heights.
    fn stem(d: f64, noise: f64, seed: u64) -> (Vec<Point>, Vec<f64>) {
        let mut rng = crate::util::nprandom::Generator::new(seed);
        let mut pts = Vec::new();
        for _ in 0..6000 {
            let h = rng.uniform(0.8, 3.2);
            let a = rng.uniform(0.0, std::f64::consts::TAU);
            let r = d / 2.0 - 0.01 * (h - 1.3) + rng.normal(0.0, noise);
            pts.push([5.0 + r * a.cos(), 5.0 + r * a.sin(), h]);
        }
        let h = pts.iter().map(|p| p[2]).collect();
        (pts, h)
    }

    #[test]
    fn increment_of_a_growing_stem() {
        let p = IncrementParams::default();
        let (pa, ha) = stem(0.3, 0.003, 1);
        let (pb, hb) = stem(0.31, 0.003, 2);
        let ma = measure_trees(&pa, &ha, &vec![1; pa.len()], &[(1, 5.0, 5.0)], 0.003, &p).unwrap();
        let mb = measure_trees(&pb, &hb, &vec![1; pb.len()], &[(1, 5.0, 5.0)], 0.003, &p).unwrap();
        assert!((ma[0].dbh - 0.3).abs() < 0.002, "{}", ma[0].dbh);
        assert!((ma[0].taper + 0.02).abs() < 0.003);
        let inc = increment(&ma[0], &mb[0], 0.01, &p);
        assert_eq!(inc.n_slices, 9);
        assert!((inc.d_dbh - 0.01).abs() < 3.0 * inc.d_dbh_se, "{} {}", inc.d_dbh, inc.d_dbh_se);
        assert_eq!(inc.dbh_change, Change::Growth);
        // A 0.2 mm increment is below detection.
        let (pc, hc) = stem(0.3002, 0.003, 3);
        let mc = measure_trees(&pc, &hc, &vec![1; pc.len()], &[(1, 5.0, 5.0)], 0.003, &p).unwrap();
        assert_eq!(increment(&ma[0], &mc[0], 0.01, &p).dbh_change, Change::BelowDetection);
        // A tree without points is unmeasured.
        let none = measure_trees(&pa, &ha, &vec![-1; pa.len()], &[(1, 5.0, 5.0)], 0.003, &p).unwrap();
        assert_eq!(increment(&ma[0], &none[0], 0.01, &p).dbh_change, Change::Unmeasured);
    }
}
