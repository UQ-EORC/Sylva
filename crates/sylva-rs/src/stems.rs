// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Multi-slice stem detection.
//!
//! Bottom-up: the stem band (1–5 m above ground)
//! is cut into horizontal layers; each layer is clustered in 2-D and circles
//! are fitted by RANSAC (Fischler & Bolles 1981) with an angular-coverage check; circles are linked
//! across layers into stem chains, which must span at least `min_slices`
//! layers and lean less than `max_lean_deg`. A branch, a shrub or a foliage
//! clump rarely produces consistent circles in three or more layers, which
//! is what makes this robust where a single thick slice is not.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::filters::Rng;
use crate::trees::Tree;
use crate::Point;

#[derive(Debug, Clone)]
pub struct StemParams {
    /// Bottom of the stem search band (m above ground).
    pub slice_min: f64,
    /// Top of the band.
    pub slice_max: f64,
    /// Height of each layer.
    pub slice_thickness: f64,
    /// Vertical spacing between layer centres.
    pub slice_step: f64,
    /// Height at which positions and DBH are reported.
    pub reference_height: f64,
    pub min_radius: f64,
    pub max_radius: f64,
    /// 2-D grid cell for connected-component clustering (about half the smallest stem radius).
    pub cluster_cell: f64,
    pub min_cluster_points: usize,
    /// Clusters wider than this cannot be a single stem.
    pub max_cluster_extent: f64,
    pub ransac_iterations: usize,
    /// Inlier band around a hypothesised circle (m).
    pub ransac_tolerance: f64,
    /// Split merged clusters (stem + neighbour) into up to this many circles.
    pub max_circles_per_cluster: usize,
    pub min_circle_inliers: usize,
    /// Fraction of the circumference that must be observed (a single scan sees ~0.5).
    pub min_coverage: f64,
    /// Longest contiguous arc (degrees) the inliers must cover; 0 disables.
    /// 130-180 deg suits merged multi-scan plots.
    pub min_arc_deg: f64,
    pub max_circle_rmse: f64,
    /// Max lateral offset between consecutive layers.
    pub link_radius: f64,
    /// Max relative radius change between layers.
    pub link_radius_ratio: f64,
    pub min_slices: usize,
    pub max_lean_deg: f64,
    /// Absolute radius tolerance for linking (so 5 cm saplings can still link).
    pub link_radius_abs: f64,
    /// Keep only stem-like band points before slicing: locally planar surface
    /// patches with a near-horizontal normal (bark), which removes foliage and
    /// horizontal branches that bridge neighbouring stems into one cluster.
    pub prefilter: bool,
    pub prefilter_k: usize,
    /// Max |normal_z| for a point to count as stem surface.
    pub prefilter_max_nz: f64,
    /// Max surface variation `l1 / (l1 + l2 + l3)` (0 = perfect plane).
    pub prefilter_max_variation: f64,
    pub seed: u64,
    /// RANSAC adaptive stop granularity: the stop is checked before every
    /// block of this many hypotheses (coregistration mode scores 32 at a time); 0 checks
    /// before every hypothesis.
    pub ransac_block: usize,
    /// Draw all `ransac_iterations` triples up front, keep only the distinct,
    /// non-collinear ones whose circle radius is in range, and count only
    /// those as tried (coregistration mode). Otherwise triples are drawn one at a time
    /// and every distinct triple counts as tried.
    pub ransac_presample: bool,
    /// Re-cluster clusters wider than `max_cluster_extent` with half the cell
    /// and keep the stem-sized pieces; false skips them (coregistration mode).
    pub recluster_wide: bool,
    /// Grid each layer from its own xy minimum and order clusters (and the
    /// points in them) as `scipy.ndimage.label` does: by first cell in
    /// row-major (y, x) order, points by index (coregistration mode). Otherwise an
    /// absolute `floor(p / cell)` grid in (x, y) order.
    pub cluster_grid_at_slice_min: bool,
    /// Include points at exactly `slice_max + thickness / 2` in the band (coregistration mode).
    pub band_top_inclusive: bool,
    /// One random stream shared by all layers in order (coregistration mode, sequential);
    /// otherwise each layer gets `seed + layer` and layers run in parallel.
    pub shared_rng: bool,
    /// The DBH taper is a least-squares line whose squared residuals are
    /// weighted by `n_inliers^power`. 1.0 (the default) weights each slice by
    /// its inlier count, as `np.polyfit(w=sqrt(n))` does (polyfit weights the
    /// unsquared residuals).
    pub taper_weight_power: f64,
    /// Clouds with fewer points find no stems (coregistration mode: 100).
    pub min_total_points: usize,
}

impl Default for StemParams {
    fn default() -> Self {
        StemParams {
            slice_min: 1.0,
            slice_max: 5.0,
            slice_thickness: 0.3,
            slice_step: 0.25,
            reference_height: 1.3,
            min_radius: 0.015,
            max_radius: 0.75,
            cluster_cell: 0.06,
            min_cluster_points: 12,
            max_cluster_extent: 2.0,
            ransac_iterations: 120,
            ransac_tolerance: 0.02,
            max_circles_per_cluster: 3,
            min_circle_inliers: 10,
            min_coverage: 0.12,
            min_arc_deg: 0.0,
            max_circle_rmse: 0.02,
            link_radius: 0.2,
            link_radius_ratio: 0.45,
            min_slices: 3,
            max_lean_deg: 25.0,
            link_radius_abs: 0.02,
            prefilter: true,
            prefilter_k: 16,
            prefilter_max_nz: 0.6,
            prefilter_max_variation: 0.15,
            seed: 0,
            ransac_block: 0,
            ransac_presample: false,
            recluster_wide: true,
            cluster_grid_at_slice_min: false,
            band_top_inclusive: false,
            shared_rng: false,
            taper_weight_power: 1.0,
            min_total_points: 0,
        }
    }
}

impl StemParams {
    /// The detector settings used for coregistration stem maps (the
    /// coregistration mode): 0.4 m slice spacing, radii from 2.5 to 60 cm, no
    /// prefilter, over-wide clusters skipped, RANSAC triples presampled and
    /// scored in blocks of 32, each layer gridded from its own minimum, and
    /// one random stream shared by all layers in order.
    pub fn coreg() -> Self {
        StemParams {
            slice_min: 1.0,
            slice_max: 5.0,
            slice_thickness: 0.3,
            slice_step: 0.4,
            reference_height: 1.3,
            min_radius: 0.025,
            max_radius: 0.60,
            cluster_cell: 0.06,
            min_cluster_points: 12,
            max_cluster_extent: 2.0,
            ransac_iterations: 120,
            ransac_tolerance: 0.02,
            max_circles_per_cluster: 3,
            min_circle_inliers: 10,
            min_coverage: 0.12,
            min_arc_deg: 0.0,
            max_circle_rmse: 0.02,
            link_radius: 0.20,
            link_radius_ratio: 0.45,
            min_slices: 3,
            max_lean_deg: 25.0,
            link_radius_abs: 0.0,
            prefilter: false,
            prefilter_k: 16,
            prefilter_max_nz: 0.6,
            prefilter_max_variation: 0.15,
            seed: 0,
            ransac_block: 32,
            ransac_presample: true,
            recluster_wide: false,
            cluster_grid_at_slice_min: true,
            band_top_inclusive: true,
            shared_rng: true,
            taper_weight_power: 1.0,
            min_total_points: 100,
        }
    }
}

/// A circle fitted to one layer.
#[derive(Debug, Clone, Copy)]
pub struct CircleFit {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub height: f64,
    pub n_inliers: usize,
    pub rmse: f64,
    pub coverage: f64,
}

fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<(f64, f64, f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-12 {
        return None;
    }
    let (s1, s2, s3) = (a[0] * a[0] + a[1] * a[1], b[0] * b[0] + b[1] * b[1], c[0] * c[0] + c[1] * c[1]);
    let cx = (s1 * (b[1] - c[1]) + s2 * (c[1] - a[1]) + s3 * (a[1] - b[1])) / d;
    let cy = (s1 * (c[0] - b[0]) + s2 * (a[0] - c[0]) + s3 * (b[0] - a[0])) / d;
    Some((cx, cy, (a[0] - cx).hypot(a[1] - cy)))
}

/// Kåsa (1976) fit about the centroid followed by Gauss–Newton geometric
/// refinement.
pub(crate) fn fit_circle_refined(xy: &[[f64; 2]]) -> Option<(f64, f64, f64)> {
    let n = xy.len() as f64;
    if xy.len() < 3 {
        return None;
    }
    let mx = xy.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = xy.iter().map(|p| p[1]).sum::<f64>() / n;
    let mut ata = nalgebra::Matrix3::<f64>::zeros();
    let mut atb = nalgebra::Vector3::<f64>::zeros();
    for p in xy {
        let (px, py) = (p[0] - mx, p[1] - my);
        let row = nalgebra::Vector3::new(2.0 * px, 2.0 * py, 1.0);
        ata += row * row.transpose();
        atb += row * (px * px + py * py);
    }
    let s = ata.lu().solve(&atb)?;
    let r2 = s[2] + s[0] * s[0] + s[1] * s[1];
    if !r2.is_finite() || r2 <= 0.0 {
        return None;
    }
    let (mut cx, mut cy, mut r) = (s[0], s[1], r2.sqrt());
    for _ in 0..8 {
        let mut jtj = nalgebra::Matrix3::<f64>::zeros();
        let mut jtr = nalgebra::Vector3::<f64>::zeros();
        for p in xy {
            let dx = p[0] - mx - cx;
            let dy = p[1] - my - cy;
            let d = dx.hypot(dy).max(1e-9);
            let res = d - r;
            let j = nalgebra::Vector3::new(-dx / d, -dy / d, -1.0);
            jtj += j * j.transpose();
            jtr += j * res;
        }
        let Some(step) = jtj.lu().solve(&(-jtr)) else { break };
        cx += step[0];
        cy += step[1];
        r += step[2];
        if r <= 0.0 {
            return None;
        }
        if step.norm() < 1e-9 {
            break;
        }
    }
    Some((cx + mx, cy + my, r))
}

/// `(fraction of 10-degree bins occupied, longest contiguous arc in degrees)`.
pub(crate) fn angular_coverage(xy: &[[f64; 2]], cx: f64, cy: f64) -> (f64, f64) {
    const BINS: usize = 36;
    let mut seen = [false; BINS];
    for p in xy {
        let a = (p[1] - cy).atan2(p[0] - cx);
        let b = (((a + std::f64::consts::PI) / std::f64::consts::TAU) * BINS as f64) as usize % BINS;
        seen[b] = true;
    }
    let occupied = seen.iter().filter(|&&s| s).count();
    // Longest run around the circle (wrapping).
    let mut best = 0usize;
    let mut run = 0usize;
    for i in 0..2 * BINS {
        if seen[i % BINS] {
            run += 1;
            best = best.max(run.min(BINS));
        } else {
            run = 0;
        }
    }
    (occupied as f64 / BINS as f64, best as f64 * 360.0 / BINS as f64)
}

/// RANSAC circle scored by (`inliers * (1 - mean_residual / tol)`)
/// and adaptive stopping. Returns `(cx, cy, r, inlier mask)`.
pub(crate) fn ransac_circle(xy: &[[f64; 2]], p: &StemParams, rng: &mut Rng) -> Option<(f64, f64, f64, Vec<bool>)> {
    let n = xy.len();
    if n < 3 {
        return None;
    }
    let mut best_score = 0.0;
    let mut best_count = 0usize;
    let mut best = (0.0, 0.0, 0.0);
    let mut tried = 0usize;
    // Adaptive stop: once a hypothesis explains a large share of the
    // cluster, more sampling is very unlikely to find a better one.
    let stop = |best_count: usize, tried: usize| {
        if best_count < 3 {
            return false;
        }
        let ratio = best_count as f64 / n as f64;
        ratio > 0.99 || tried as f64 >= (1e-3f64).ln() / (1.0 - ratio.powi(3)).max(1e-12).ln()
    };
    let consider = |cx: f64, cy: f64, r: f64, best_score: &mut f64, best_count: &mut usize, best: &mut (f64, f64, f64)| {
        let mut count = 0usize;
        let mut total = 0.0;
        for q in xy {
            let res = ((q[0] - cx).hypot(q[1] - cy) - r).abs();
            if res < p.ransac_tolerance {
                count += 1;
                total += res;
            }
        }
        if count < 3 {
            return;
        }
        let score = count as f64 * (1.0 - total / count as f64 / p.ransac_tolerance);
        if score > *best_score {
            *best_score = score;
            *best_count = count;
            *best = (cx, cy, r);
        }
    };
    let block = p.ransac_block.max(1);
    if p.ransac_presample {
        // Every triple is drawn first; only valid circles are candidates.
        let triples: Vec<(usize, usize, usize)> = (0..p.ransac_iterations).map(|_| (rng.below(n), rng.below(n), rng.below(n))).collect();
        let candidates: Vec<(f64, f64, f64)> = triples
            .into_iter()
            .filter(|&(i, j, k)| i != j && j != k && i != k)
            .filter_map(|(i, j, k)| circumcircle(xy[i], xy[j], xy[k]))
            .filter(|&(_, _, r)| r >= p.min_radius && r <= p.max_radius)
            .collect();
        for chunk in candidates.chunks(block) {
            if stop(best_count, tried) {
                break;
            }
            tried += chunk.len();
            for &(cx, cy, r) in chunk {
                consider(cx, cy, r, &mut best_score, &mut best_count, &mut best);
            }
        }
    } else {
        for _ in 0..p.ransac_iterations {
            if tried.is_multiple_of(block) && stop(best_count, tried) {
                break;
            }
            let (i, j, k) = (rng.below(n), rng.below(n), rng.below(n));
            if i == j || j == k || i == k {
                continue;
            }
            tried += 1;
            let Some((cx, cy, r)) = circumcircle(xy[i], xy[j], xy[k]) else { continue };
            if r < p.min_radius || r > p.max_radius {
                continue;
            }
            consider(cx, cy, r, &mut best_score, &mut best_count, &mut best);
        }
    }
    if best_score <= 0.0 {
        return None;
    }
    let (mut cx, mut cy, mut r) = best;
    let mut mask: Vec<bool> = xy.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).abs() < p.ransac_tolerance).collect();
    for _ in 0..3 {
        let pts: Vec<[f64; 2]> = xy.iter().zip(&mask).filter(|(_, &m)| m).map(|(q, _)| *q).collect();
        if pts.len() < 3 {
            break;
        }
        let (ncx, ncy, nr) = fit_circle_refined(&pts)?;
        if nr < p.min_radius || nr > p.max_radius {
            return None;
        }
        cx = ncx;
        cy = ncy;
        r = nr;
        let new_mask: Vec<bool> = xy.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).abs() < p.ransac_tolerance).collect();
        let cnt = new_mask.iter().filter(|&&m| m).count();
        if cnt < 3 || new_mask == mask {
            if cnt >= 3 {
                mask = new_mask;
            }
            break;
        }
        mask = new_mask;
    }
    Some((cx, cy, r, mask))
}

/// Connected components on a 2-D occupancy grid (8-connectivity).
fn cluster_2d(xy: &[[f64; 2]], cell: f64, min_points: usize) -> Vec<Vec<usize>> {
    if xy.is_empty() {
        return Vec::new();
    }
    let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in xy.iter().enumerate() {
        cells.entry(((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64)).or_default().push(i);
    }
    let mut label: HashMap<(i64, i64), usize> = HashMap::new();
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut stack = Vec::new();
    // Deterministic traversal order (HashMap iteration is randomised), so the
    // RANSAC sampling sequence -- and therefore the result -- is reproducible.
    let mut keys: Vec<(i64, i64)> = cells.keys().copied().collect();
    keys.sort_unstable();
    for start in keys {
        if label.contains_key(&start) {
            continue;
        }
        let id = clusters.len();
        clusters.push(Vec::new());
        label.insert(start, id);
        stack.push(start);
        while let Some(c) = stack.pop() {
            clusters[id].extend_from_slice(&cells[&c]);
            for di in -1..=1 {
                for dj in -1..=1 {
                    let nb = (c.0 + di, c.1 + dj);
                    if (di != 0 || dj != 0) && cells.contains_key(&nb) && !label.contains_key(&nb) {
                        label.insert(nb, id);
                        stack.push(nb);
                    }
                }
            }
        }
    }
    clusters.retain(|c| c.len() >= min_points);
    clusters
}

/// Connected components (8-connectivity) in a fixed order: the grid starts at the points' xy minimum, components come in the
/// order `scipy.ndimage.label` numbers them (first cell in row-major (y, x)
/// order) and the points of each component in index order.
fn cluster_2d_raster(xy: &[[f64; 2]], cell: f64, min_points: usize) -> Vec<Vec<usize>> {
    if xy.is_empty() {
        return Vec::new();
    }
    let lo = [xy.iter().map(|q| q[0]).fold(f64::INFINITY, f64::min), xy.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min)];
    let mut cells: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, q) in xy.iter().enumerate() {
        // (row, column) = (y, x), truncated as numpy's astype(int64) does.
        cells.entry((((q[1] - lo[1]) / cell) as i64, ((q[0] - lo[0]) / cell) as i64)).or_default().push(i);
    }
    let mut keys: Vec<(i64, i64)> = cells.keys().copied().collect();
    keys.sort_unstable();
    let mut seen: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut stack = Vec::new();
    for start in keys {
        if !seen.insert(start) {
            continue;
        }
        let mut members = Vec::new();
        stack.push(start);
        while let Some(c) = stack.pop() {
            members.extend_from_slice(&cells[&c]);
            for di in -1..=1 {
                for dj in -1..=1 {
                    let nb = (c.0 + di, c.1 + dj);
                    if (di != 0 || dj != 0) && cells.contains_key(&nb) && seen.insert(nb) {
                        stack.push(nb);
                    }
                }
            }
        }
        if members.len() >= min_points {
            members.sort_unstable();
            clusters.push(members);
        }
    }
    clusters
}

/// Fit every plausible stem circle in one layer.
fn fit_layer(xy: &[[f64; 2]], height: f64, p: &StemParams, rng: &mut Rng) -> Vec<CircleFit> {
    let cluster = |xy: &[[f64; 2]], cell: f64| if p.cluster_grid_at_slice_min { cluster_2d_raster(xy, cell, p.min_cluster_points) } else { cluster_2d(xy, cell, p.min_cluster_points) };
    let mut circles = Vec::new();
    let extent = |c: &[[f64; 2]]| {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for q in c {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        (hi[0] - lo[0]).max(hi[1] - lo[1])
    };
    let mut clusters: Vec<Vec<[f64; 2]>> = Vec::new();
    for idx in cluster(xy, p.cluster_cell) {
        let cluster_xy: Vec<[f64; 2]> = idx.iter().map(|&i| xy[i]).collect();
        if extent(&cluster_xy) <= p.max_cluster_extent {
            clusters.push(cluster_xy);
            continue;
        }
        if !p.recluster_wide {
            continue;
        }
        // Too wide to be one stem: re-cluster with a finer cell to break
        // bridges, keeping only the pieces that are stem-sized.
        for sub in cluster(&cluster_xy, p.cluster_cell * 0.5) {
            let c: Vec<[f64; 2]> = sub.iter().map(|&i| cluster_xy[i]).collect();
            if extent(&c) <= p.max_cluster_extent {
                clusters.push(c);
            }
        }
    }
    for cluster in clusters {
        let mut remaining = cluster;
        for _ in 0..p.max_circles_per_cluster {
            if remaining.len() < p.min_circle_inliers {
                break;
            }
            let Some((cx, cy, r, mask)) = ransac_circle(&remaining, p, rng) else { break };
            let inliers: Vec<[f64; 2]> = remaining.iter().zip(&mask).filter(|(_, &m)| m).map(|(q, _)| *q).collect();
            if inliers.len() < p.min_circle_inliers {
                break;
            }
            let rmse = (inliers.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).powi(2)).sum::<f64>() / inliers.len() as f64).sqrt();
            let (coverage, arc) = angular_coverage(&inliers, cx, cy);
            if rmse <= p.max_circle_rmse && coverage >= p.min_coverage && arc >= p.min_arc_deg {
                circles.push(CircleFit { x: cx, y: cy, radius: r, height, n_inliers: inliers.len(), rmse, coverage });
            }
            remaining = remaining.iter().zip(&mask).filter(|(_, &m)| !m).map(|(q, _)| *q).collect();
        }
    }
    circles
}

fn predict_next(chain: &[CircleFit], next_height: f64) -> (f64, f64) {
    let last = chain[chain.len() - 1];
    if chain.len() < 2 {
        return (last.x, last.y);
    }
    let prev = chain[chain.len() - 2];
    let dh = last.height - prev.height;
    if dh.abs() < 1e-9 {
        return (last.x, last.y);
    }
    let step = (next_height - last.height) / dh;
    (last.x + (last.x - prev.x) * step, last.y + (last.y - prev.y) * step)
}

/// Greedily link circles of successive layers into chains, extrapolating each
/// chain's lean so leaning stems stay linked.
fn link_layers(layers: &[Vec<CircleFit>], p: &StemParams) -> Vec<Vec<CircleFit>> {
    let mut chains: Vec<Vec<CircleFit>> = Vec::new();
    let mut open: Vec<Vec<CircleFit>> = Vec::new();
    for layer in layers {
        let mut used = vec![false; layer.len()];
        let mut still_open = Vec::new();
        let layer_height = layer.first().map(|c| c.height);
        for mut chain in open {
            let mut best: Option<(usize, f64)> = None;
            if let Some(h) = layer_height {
                let (px, py) = predict_next(&chain, h);
                let r_ref = chain[chain.len() - 1].radius;
                for (i, c) in layer.iter().enumerate() {
                    if used[i] || (c.radius - r_ref).abs() > (p.link_radius_ratio * r_ref).max(p.link_radius_abs) {
                        continue;
                    }
                    let d = (c.x - px).hypot(c.y - py);
                    if d <= p.link_radius && best.map(|b| d < b.1).unwrap_or(true) {
                        best = Some((i, d));
                    }
                }
            }
            match best {
                Some((i, _)) => {
                    used[i] = true;
                    chain.push(layer[i]);
                    still_open.push(chain);
                }
                None if chain.len() >= p.min_slices => chains.push(chain),
                None => {}
            }
        }
        for (i, c) in layer.iter().enumerate() {
            if !used[i] {
                still_open.push(vec![*c]);
            }
        }
        open = still_open;
    }
    chains.extend(open.into_iter().filter(|c| c.len() >= p.min_slices));
    chains
}

fn diameter_at(chain: &[CircleFit], height: f64, weight_power: f64) -> f64 {
    let hs: Vec<f64> = chain.iter().map(|c| c.height).collect();
    let rs: Vec<f64> = chain.iter().map(|c| c.radius).collect();
    let nearest = rs[hs.iter().enumerate().min_by(|a, b| (a.1 - height).abs().partial_cmp(&(b.1 - height).abs()).unwrap()).unwrap().0];
    let span = hs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - hs.iter().cloned().fold(f64::INFINITY, f64::min);
    if chain.len() >= 3 && span > 0.5 {
        // Weighted linear taper r(h).
        let w: Vec<f64> = chain.iter().map(|c| {
            let n = c.n_inliers as f64;
            if weight_power == 0.5 { n.sqrt() } else if weight_power == 1.0 { n } else { n.powf(weight_power) }
        }).collect();
        let sw: f64 = w.iter().sum();
        let mh = hs.iter().zip(&w).map(|(h, w)| h * w).sum::<f64>() / sw;
        let mr = rs.iter().zip(&w).map(|(r, w)| r * w).sum::<f64>() / sw;
        let sxx: f64 = hs.iter().zip(&w).map(|(h, w)| w * (h - mh).powi(2)).sum();
        let sxy: f64 = hs.iter().zip(&rs).zip(&w).map(|((h, r), w)| w * (h - mh) * (r - mr)).sum();
        if sxx > 0.0 {
            let radius = mr + sxy / sxx * (height - mh);
            let (rmin, rmax) = (rs.iter().cloned().fold(f64::INFINITY, f64::min), rs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            if radius >= 0.5 * rmin && radius <= 1.5 * rmax {
                return 2.0 * radius;
            }
        }
    }
    2.0 * nearest
}

/// A detected stem with everything the detector reports for it.
#[derive(Debug, Clone)]
pub struct StemFit {
    /// Position at `reference_height`, DBH, quality etc.; `inlier_fraction` is the coverage.
    pub tree: Tree,
    /// Height above ground of the reported position (the reference height, up to rounding).
    pub z: f64,
    /// Unit stem direction, pointing up, fitted in height-above-ground space.
    pub axis: [f64; 3],
    /// Inlier-weighted mean angular coverage of the circles.
    pub coverage: f64,
    /// The linked circles, bottom up.
    pub circles: Vec<CircleFit>,
}

fn chain_to_stem(chain: &[CircleFit], p: &StemParams) -> Option<StemFit> {
    if chain.len() < p.min_slices {
        return None;
    }
    let wsum: f64 = chain.iter().map(|c| c.n_inliers as f64).sum();
    let w: Vec<f64> = chain.iter().map(|c| c.n_inliers as f64 / wsum).collect();
    let mut mean = [0.0; 3];
    for (c, w) in chain.iter().zip(&w) {
        mean[0] += w * c.x;
        mean[1] += w * c.y;
        mean[2] += w * c.height;
    }
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    for (c, w) in chain.iter().zip(&w) {
        let d = nalgebra::Vector3::new(c.x - mean[0], c.y - mean[1], c.height - mean[2]);
        cov += d * d.transpose() * *w;
    }
    let eig = cov.symmetric_eigen();
    let imax = (0..3).max_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap()).unwrap();
    let mut axis = [eig.eigenvectors[(0, imax)], eig.eigenvectors[(1, imax)], eig.eigenvectors[(2, imax)]];
    if axis[2] < 0.0 {
        axis = [-axis[0], -axis[1], -axis[2]];
    }
    let lean = axis[2].abs().clamp(0.0, 1.0).acos().to_degrees();
    if lean > p.max_lean_deg || axis[2].abs() < 1e-6 {
        return None;
    }
    let t = (p.reference_height - mean[2]) / axis[2];
    let (x, y, z) = (mean[0] + t * axis[0], mean[1] + t * axis[1], mean[2] + t * axis[2]);
    let dbh = diameter_at(chain, p.reference_height, p.taper_weight_power);
    if dbh < 2.0 * p.min_radius || dbh > 2.0 * p.max_radius {
        return None;
    }
    // np.average: divide by the (normalised, so ~1) weight sum.
    let sw: f64 = w.iter().sum();
    let rmse: f64 = chain.iter().zip(&w).map(|(c, w)| c.rmse * w).sum::<f64>() / sw;
    let coverage: f64 = chain.iter().zip(&w).map(|(c, w)| c.coverage * w).sum::<f64>() / sw;
    let quality = (1.0 / (1.0 + rmse / 0.01)) * coverage * (chain.len() as f64 / 6.0).min(1.0);
    let tree = Tree {
        tree_id: 0,
        x,
        y,
        dbh,
        height: f64::NAN,
        n_points: chain.iter().map(|c| c.n_inliers).sum(),
        inlier_fraction: coverage,
        n_slices: chain.len(),
        rmse,
        lean_deg: lean,
        quality: quality.clamp(0.0, 1.0),
    };
    Some(StemFit { tree, z, axis, coverage, circles: chain.to_vec() })
}

/// Detect stems in a height-normalised cloud. Returns trees sorted by
/// quality descending with `tree_id` 1..n.
pub fn detect_stems(points: &[Point], heights: &[f64], p: &StemParams) -> Vec<Tree> {
    detect_stems_full(points, heights, p).into_iter().map(|s| s.tree).collect()
}

/// [`detect_stems`] with each stem's axis, coverage and circles.
pub fn detect_stems_full(points: &[Point], heights: &[f64], p: &StemParams) -> Vec<StemFit> {
    if points.len() < p.min_total_points {
        return Vec::new();
    }
    let half = 0.5 * p.slice_thickness;
    let (lo, hi) = (p.slice_min - half, p.slice_max + half);
    // Restrict to the band once; the prefilter runs on the band only.
    let band_idx: Vec<usize> = (0..points.len())
        .filter(|&i| heights[i] >= lo && (heights[i] < hi || (p.band_top_inclusive && heights[i] == hi)))
        .collect();
    let mut band: Vec<Point> = band_idx.iter().map(|&i| points[i]).collect();
    let mut band_h: Vec<f64> = band_idx.iter().map(|&i| heights[i]).collect();
    if p.prefilter && band.len() > p.prefilter_k {
        let (normals, vals) = crate::filters::local_pca(&band, p.prefilter_k);
        let keep: Vec<bool> = normals
            .iter()
            .zip(&vals)
            .map(|(n, [l1, l2, l3])| {
                let sum = l1 + l2 + l3;
                n[2].abs() <= p.prefilter_max_nz && sum > 0.0 && l1 / sum <= p.prefilter_max_variation
            })
            .collect();
        band = band.iter().zip(&keep).filter(|(_, &k)| k).map(|(q, _)| *q).collect();
        band_h = band_h.iter().zip(&keep).filter(|(_, &k)| k).map(|(&h, _)| h).collect();
    }
    // Centres as np.arange(slice_min, slice_max + 1e-9, slice_step) makes them.
    let delta = (p.slice_min + p.slice_step) - p.slice_min;
    let n_layers = ((p.slice_max + 1e-9 - p.slice_min) / delta).ceil().max(0.0) as usize;
    let centres: Vec<f64> = (0..n_layers).map(|i| if i == 0 { p.slice_min } else { p.slice_min + i as f64 * delta }).collect();
    let layer_xy = |c: f64| -> Vec<[f64; 2]> {
        band.iter()
            .zip(&band_h)
            .filter(|(_, &h)| h >= c - half && h < c + half)
            .map(|(q, _)| [q[0], q[1]])
            .collect()
    };
    let layers: Vec<Vec<CircleFit>> = if p.shared_rng {
        let mut rng = Rng::new(p.seed);
        centres.iter().map(|&c| fit_layer(&layer_xy(c), c, p, &mut rng)).collect()
    } else {
        centres
            .par_iter()
            .enumerate()
            .map(|(li, &c)| fit_layer(&layer_xy(c), c, p, &mut Rng::new(p.seed.wrapping_add(li as u64))))
            .collect()
    };
    let chains = link_layers(&layers, p);
    let mut out: Vec<StemFit> = chains.iter().filter_map(|c| chain_to_stem(c, p)).collect();
    out.sort_by(|a, b| b.tree.quality.partial_cmp(&a.tree.quality).unwrap());
    for (i, s) in out.iter_mut().enumerate() {
        s.tree.tree_id = i as i64 + 1;
    }
    out
}

/// Bark-point support of each candidate's stem below the search band.
///
/// For every tree and every slab centre in `slab_heights` (m above ground),
/// counts the prefiltered (bark-like) points within `slab_half` of that
/// height and within `max(trunk_scale * radius, trunk_min) + tan(lean) *
/// (reference_height - h)` of the stem axis. A real stem is supported all the
/// way down to the ground; a branch or a circle fitted to a clump is not.
pub fn stem_root_support(points: &[Point], heights: &[f64], trees: &[Tree], p: &StemParams, slab_heights: &[f64], slab_half: f64, trunk_scale: f64, trunk_min: f64) -> Vec<Vec<usize>> {
    let lo = slab_heights.iter().cloned().fold(f64::INFINITY, f64::min) - slab_half;
    let hi = slab_heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + slab_half;
    let idx: Vec<usize> = (0..points.len()).filter(|&i| heights[i] >= lo && heights[i] <= hi).collect();
    let mut band: Vec<Point> = idx.iter().map(|&i| points[i]).collect();
    let mut band_h: Vec<f64> = idx.iter().map(|&i| heights[i]).collect();
    if p.prefilter && band.len() > p.prefilter_k {
        let (normals, vals) = crate::filters::local_pca(&band, p.prefilter_k);
        let keep: Vec<bool> = normals
            .iter()
            .zip(&vals)
            .map(|(n, [l1, l2, l3])| {
                let sum = l1 + l2 + l3;
                n[2].abs() <= p.prefilter_max_nz && sum > 0.0 && l1 / sum <= p.prefilter_max_variation
            })
            .collect();
        band = band.iter().zip(&keep).filter(|(_, &k)| k).map(|(q, _)| *q).collect();
        band_h = band_h.iter().zip(&keep).filter(|(_, &k)| k).map(|(&h, _)| h).collect();
    }
    let flat: Vec<Point> = band.iter().map(|q| [q[0], q[1], 0.0]).collect();
    let kd = crate::spatial::KdTree::new(&flat);
    trees
        .par_iter()
        .map(|t| {
            let tan_lean = t.lean_deg.to_radians().tan().abs();
            slab_heights
                .iter()
                .map(|&h| {
                    let r = (trunk_scale * t.dbh / 2.0).max(trunk_min) + tan_lean * (p.reference_height - h).abs();
                    kd.within(&[t.x, t.y, 0.0], r)
                        .into_iter()
                        .filter(|(i, _)| (band_h[*i] - h).abs() <= slab_half)
                        .count()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Half-visible vertical cylinders, as one scan sees them, plus scattered clutter.
    fn plot() -> (Vec<Point>, Vec<f64>, Vec<(f64, f64, f64)>) {
        let stems = vec![(0.0, 0.0, 0.15), (3.0, 1.0, 0.08), (-2.5, 4.0, 0.3), (5.0, -3.0, 0.2), (-4.0, -4.0, 0.05)];
        let mut rng = Rng::new(7);
        let mut unit = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let mut pts = Vec::new();
        for &(x, y, r) in &stems {
            let mut z = 0.0;
            while z < 6.0 {
                for k in 0..40 {
                    let a = std::f64::consts::PI * (k as f64 / 40.0 + 0.1 * unit());
                    let rr = r + 0.003 * (unit() - 0.5);
                    pts.push([x + rr * a.cos(), y - rr * a.sin(), z]);
                }
                z += 0.02;
            }
        }
        for _ in 0..5000 {
            pts.push([12.0 * unit() - 6.0, 12.0 * unit() - 6.0, 6.0 * unit()]);
        }
        let h = pts.iter().map(|q| q[2]).collect();
        (pts, h, stems)
    }

    #[test]
    fn both_modes_find_every_stem() {
        let (pts, h, truth) = plot();
        for p in [StemParams { prefilter: false, ..Default::default() }, StemParams::coreg()] {
            let found = detect_stems_full(&pts, &h, &p);
            for &(x, y, r) in &truth {
                let s = found.iter().find(|s| (s.tree.x - x).hypot(s.tree.y - y) < 0.03).unwrap_or_else(|| panic!("missed ({x}, {y}) with {p:?}"));
                assert!((s.tree.dbh - 2.0 * r).abs() < 0.02, "dbh {} for r {r}", s.tree.dbh);
                assert!(s.axis[2] > 0.99 && (s.z - p.reference_height).abs() < 1e-9);
            }
            assert!(found.windows(2).all(|w| w[0].tree.quality >= w[1].tree.quality));
        }
    }

    #[test]
    fn coreg_mode_is_deterministic() {
        let (pts, h, _) = plot();
        let a = detect_stems(&pts, &h, &StemParams::coreg());
        let b = detect_stems(&pts, &h, &StemParams::coreg());
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(s, t)| s.x == t.x && s.dbh == t.dbh));
    }

    #[test]
    fn raster_clusters_come_in_ndimage_label_order() {
        // Cells (row = y, col = x): a U whose arms start at row 0, cols 0 and 4,
        // one cell at (0, 2) and one at (1, 6): scipy labels them 1 (U), 2, 3.
        let c = |col: f64, row: f64| [col * 0.06 + 0.03, row * 0.06 + 0.03];
        let mut xy = vec![c(6.0, 1.0)];
        for r in 0..4 {
            xy.push(c(0.0, r as f64));
            xy.push(c(4.0, r as f64));
        }
        for col in 1..4 {
            xy.push(c(col as f64, 3.0));
        }
        xy.push(c(2.0, 0.0));
        let cl = cluster_2d_raster(&xy, 0.06, 1);
        assert_eq!(cl.len(), 3);
        assert_eq!(cl[0], (1..12).collect::<Vec<_>>());
        assert_eq!(cl[1], vec![12]);
        assert_eq!(cl[2], vec![0]);
    }
}
