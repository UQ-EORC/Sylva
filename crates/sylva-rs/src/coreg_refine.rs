// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Joint multi-view refinement: every pose solved at once from raw
//! correspondences.
//!
//! The pose graph sees each accepted pair as one relative transform. This
//! keeps every point-to-plane correspondence (Chen & Medioni 1992) of every
//! accepted pair, adds the matched stems as horizontal constraints, and
//! solves all poses together by a damped Gauss-Newton with Huber (1964)
//! weights, left perturbations `se3_exp(xi) @ pose`, a weak prior tying each
//! pose to its starting value, a capped step that must lower the robust
//! cost, and levels from coarse to fine. The arithmetic, the random subsets
//! (NumPy's generator, [`crate::nprandom`]) and every decision follow the
//! NumPy implementation this replaced (`sylva.coreg.refine`).

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, Vector6};

use crate::coreg::numpy_sum;
use crate::coreg_geometry::{estimate_normals, voxel_centroids, CoregTree};
use crate::coreg_transforms::{invert, se3_exp, se3_log, solve, transform_difference, transform_points, transform_vectors, Mat4};
use crate::error::{Error, Result};
use crate::nprandom::Generator;
use crate::numeric::median;
use crate::Point;

/// Settings of [`refine_joint`] (the Python defaults in [`Default`]).
#[derive(Debug, Clone)]
pub struct RefineParams {
    pub voxel_sizes: Vec<f64>,
    pub max_distances: Vec<f64>,
    pub rounds: usize,
    pub iterations: usize,
    pub points_per_scan: usize,
    pub correspondences_per_pair: usize,
    pub min_planarity: f64,
    pub normal_neighbours: usize,
    pub max_normal_angle_deg: f64,
    pub stem_weight: f64,
    pub stem_radius: f64,
    pub stem_scale: f64,
    pub min_voxel_points: usize,
    pub robust_scale: f64,
    pub prior_translation: f64,
    pub prior_rotation_deg: f64,
    pub max_step_translation: f64,
    pub max_step_rotation_deg: f64,
    pub seed: u64,
}

impl Default for RefineParams {
    fn default() -> Self {
        RefineParams {
            voxel_sizes: vec![0.10, 0.05, 0.03],
            max_distances: vec![0.30, 0.15, 0.08],
            rounds: 3,
            iterations: 3,
            points_per_scan: 400_000,
            correspondences_per_pair: 20_000,
            min_planarity: 0.25,
            normal_neighbours: 20,
            max_normal_angle_deg: 45.0,
            stem_weight: 0.05,
            stem_radius: 0.15,
            stem_scale: 0.03,
            min_voxel_points: 1,
            robust_scale: 0.02,
            prior_translation: 0.03,
            prior_rotation_deg: 0.3,
            max_step_translation: 0.05,
            max_step_rotation_deg: 0.5,
            seed: 0,
        }
    }
}

/// Result of [`refine_joint`].
#[derive(Debug, Clone)]
pub struct JointRefinement {
    /// Refined `world_from_scan`.
    pub poses: Vec<Mat4>,
    /// How far each scan moved (m).
    pub shifts: Vec<f64>,
    /// How far each scan turned (degrees).
    pub rotations: Vec<f64>,
    /// Median absolute point-to-plane residual (m) before and after.
    pub residual_before: f64,
    pub residual_after: f64,
    /// Largest number of correspondences in one association.
    pub correspondences: usize,
}

/// Point-to-plane correspondences of scan `a` onto scan `b`, in their own frames.
struct Corr {
    a: usize,
    b: usize,
    p: Vec<Point>,
    q: Vec<Point>,
    nq: Vec<Point>,
}

/// Matched stems of a pair, in insertion order (a Python dict).
type StemPairs = Vec<((usize, usize), (Vec<Point>, Vec<Point>))>;

/// Robust scales per correspondence block, set by the first estimate.
type Scales = Vec<((usize, usize, usize), f64)>;

fn dot3(a: &Point, b: &Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &Point, b: &Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Left-perturbation twist with `t1 = exp(xi) t0`.
fn twist_between(t0: &Mat4, t1: &Mat4) -> Vector6<f64> {
    se3_log(&(t1 * invert(t0)))
}

/// Mutually nearest stems within `radius` horizontally, per edge, under `poses`.
fn match_stems(poses: &[Mat4], stems: &[Vec<Point>], edges: &[(usize, usize)], radius: f64) -> StemPairs {
    let world: Vec<Vec<Point>> = stems.iter().enumerate().map(|(k, s)| transform_points(&poses[k], s).iter().map(|p| [p[0], p[1], 0.0]).collect()).collect();
    let mut out: StemPairs = Vec::new();
    for &(i, j) in edges {
        if world[i].is_empty() || world[j].is_empty() {
            continue;
        }
        let (dij, mij) = CoregTree::new(&world[j]).query(&world[i], radius);
        let (_, mji) = CoregTree::new(&world[i]).query(&world[j], radius);
        let a: Vec<usize> = (0..world[i].len()).filter(|&k| dij[k].is_finite() && mji[mij[k]] == k).collect();
        if !a.is_empty() {
            let pair = (a.iter().map(|&k| stems[i][k]).collect(), a.iter().map(|&k| stems[j][mij[k]]).collect());
            match out.iter_mut().find(|(key, _)| *key == (i, j)) {
                Some(slot) => slot.1 = pair,
                None => out.push(((i, j), pair)),
            }
        }
    }
    out
}

/// `H[a, b] += J^T diag(w) J` and `g += (J diag(w))^T r` for rows of 12
/// Jacobian entries (6 for scan `a`, 6 for scan `b`).
struct Accumulator {
    hp: [[f64; 12]; 12],
    gp: [f64; 12],
}

impl Accumulator {
    fn new() -> Self {
        Accumulator { hp: [[0.0; 12]; 12], gp: [0.0; 12] }
    }

    fn row(&mut self, j: &[f64; 12], w: f64, r: f64) {
        let jw: [f64; 12] = std::array::from_fn(|q| j[q] * w);
        for (p, hp) in self.hp.iter_mut().enumerate() {
            for (h, v) in hp.iter_mut().zip(&jw) {
                *h += j[p] * v;
            }
            self.gp[p] += jw[p] * r;
        }
    }

    fn add_to(&self, h: &mut DMatrix<f64>, g: &mut DVector<f64>, a: usize, b: usize) {
        for (ra, pa) in [(a, 0), (b, 6)] {
            for (rb, pb) in [(a, 0), (b, 6)] {
                for p in 0..6 {
                    for q in 0..6 {
                        h[(6 * ra + p, 6 * rb + q)] += self.hp[pa + p][pb + q];
                    }
                }
            }
        }
        for p in 0..6 {
            g[6 * a + p] += self.gp[p];
        }
        for p in 0..6 {
            g[6 * b + p] += self.gp[6 + p];
        }
    }
}

struct Assembled {
    h: DMatrix<f64>,
    g: DVector<f64>,
    cost: f64,
    median: f64,
    scales: Scales,
}

/// Robust cost, median residual and (optionally) the normal equations.
/// `scales` fixes each block's Huber scale; when `None` they are estimated
/// and returned, so a trial step is costed at the scales of the step it is
/// compared with.
#[allow(clippy::too_many_arguments)]
fn assemble(poses: &[Mat4], corr: &[Corr], stems_by_pair: &StemPairs, n: usize, robust_scale: f64, stem_weight: f64, need_system: bool, scales: Option<&Scales>, stem_scale: f64) -> Assembled {
    let dim = if need_system { 6 * n } else { 0 };
    let mut h = DMatrix::zeros(dim, dim);
    let mut g = DVector::zeros(dim);
    let mut cost = 0.0;
    let mut res_all: Vec<f64> = Vec::new();
    let mut out_scales: Scales = scales.cloned().unwrap_or_default();
    let mut pair_weight: Vec<((usize, usize), f64)> = Vec::new();
    for c in corr {
        let xa = transform_points(&poses[c.a], &c.p);
        let xb = transform_points(&poses[c.b], &c.q);
        let nw = transform_vectors(&poses[c.b], &c.nq);
        let dvec: Vec<Point> = xa.iter().zip(&xb).map(|(u, v)| [u[0] - v[0], u[1] - v[1], u[2] - v[2]]).collect();
        let r: Vec<f64> = nw.iter().zip(&dvec).map(|(u, v)| dot3(u, v)).collect();
        let absr: Vec<f64> = r.iter().map(|v| v.abs()).collect();
        let key_s = (c.a, c.b, r.len());
        let s = match scales {
            Some(fixed) => fixed.iter().find(|(k, _)| *k == key_s).map(|(_, v)| *v).expect("a scale for every block"),
            None => pymax(pymax(robust_scale, median(&absr)), 1e-9),
        };
        if !out_scales.iter().any(|(k, _)| *k == key_s) {
            out_scales.push((key_s, s));
        }
        let w: Vec<f64> = absr.iter().map(|&x| if x <= s { 1.0 } else { s / pymax(x, 1e-12) }).collect();
        let terms: Vec<f64> = r.iter().zip(&absr).map(|(&v, &x)| if x <= s { 0.5 * v * v } else { s * (x - 0.5 * s) }).collect();
        cost += numpy_sum(&terms);
        res_all.extend_from_slice(&absr);
        let key = if stems_by_pair.iter().any(|(k, _)| *k == (c.a, c.b)) { (c.a, c.b) } else { (c.b, c.a) };
        let wsum = numpy_sum(&w);
        match pair_weight.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 += wsum,
            None => pair_weight.push((key, 0.0 + wsum)),
        }
        if need_system {
            let mut acc = Accumulator::new();
            for k in 0..r.len() {
                let c1 = cross(&xa[k], &nw[k]);
                let c2 = cross(&nw[k], &xb[k]);
                let c3 = cross(&dvec[k], &nw[k]);
                let n3 = nw[k];
                let row = [c1[0], c1[1], c1[2], n3[0], n3[1], n3[2], c2[0] - c3[0], c2[1] - c3[1], c2[2] - c3[2], -n3[0], -n3[1], -n3[2]];
                acc.row(&row, w[k], r[k]);
            }
            acc.add_to(&mut h, &mut g, c.a, c.b);
        }
    }
    for ((i, j), (p, q)) in stems_by_pair {
        let Some(&(_, pw)) = pair_weight.iter().find(|(k, _)| *k == (*i, *j)) else { continue };
        let xi = transform_points(&poses[*i], p);
        let xj = transform_points(&poses[*j], q);
        let r2: Vec<f64> = xi.iter().zip(&xj).flat_map(|(u, v)| [u[0] - v[0], u[1] - v[1]]).collect();
        let s = stem_scale;
        let absr: Vec<f64> = r2.iter().map(|v| v.abs()).collect();
        let w0: Vec<f64> = absr.iter().map(|&x| if x <= s { 1.0 } else { s / pymax(x, 1e-12) }).collect();
        let weight = stem_weight * pw / pymax((1.0 - stem_weight) * numpy_sum(&w0), 1e-9);
        let w: Vec<f64> = w0.iter().map(|v| v * weight).collect();
        let terms: Vec<f64> = r2.iter().zip(&absr).map(|(&v, &x)| weight * if x <= s { 0.5 * v * v } else { s * (x - 0.5 * s) }).collect();
        cost += numpy_sum(&terms);
        if need_system {
            let mut acc = Accumulator::new();
            for (k, (a, b)) in xi.iter().zip(&xj).enumerate() {
                // Rows of -skew(xi) and skew(xj): x then y.
                let rx = [-0.0, a[2], -a[1], 1.0, 0.0, 0.0, 0.0, -b[2], b[1], -1.0, 0.0, 0.0];
                let ry = [-a[2], -0.0, a[0], 0.0, 1.0, 0.0, b[2], 0.0, -b[0], 0.0, -1.0, 0.0];
                acc.row(&rx, w[2 * k], r2[2 * k]);
                acc.row(&ry, w[2 * k + 1], r2[2 * k + 1]);
            }
            acc.add_to(&mut h, &mut g, *i, *j);
        }
    }
    let med = if res_all.is_empty() { f64::NAN } else { median(&res_all) };
    Assembled { h, g, cost, median: med, scales: out_scales }
}

/// Python's `f"{x:.2f}"`.
fn fixed2(x: f64) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        format!("{x:.2}")
    }
}

/// Python's `f"{n:,}"`.
fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (k, c) in s.chars().enumerate() {
        if k > 0 && (s.len() - k).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

struct Level {
    pts: Vec<Point>,
    nrm: Vec<Point>,
    ok: Vec<bool>,
}

/// Solve all poses together from point-to-plane and stem correspondences.
///
/// `points[k]` are scan `k`'s planar ICP points and `stems[k]` its stem
/// positions, both in its own frame; `poses` the current `world_from_scan`;
/// `edges` the accepted pairs; `reference` the scan held fixed. Levels run
/// coarse to fine; at each, correspondences are associated `rounds` times
/// under the current poses, each association followed by `iterations`
/// Gauss-Newton steps. Stems are paired afresh at every association,
/// mutually nearest within `stem_radius` horizontally, and carry
/// `stem_weight` of each pair's weight. `log` receives progress messages.
pub fn refine_joint(points: &[Vec<Point>], poses: &[Mat4], edges: &[(usize, usize)], stems: &[Vec<Point>], reference: usize, p: &RefineParams, log: &mut dyn FnMut(&str)) -> Result<JointRefinement> {
    let n = poses.len();
    if points.len() < n {
        return Err(Error::invalid("refine_joint needs the points of every scan"));
    }
    if p.voxel_sizes.len() != p.max_distances.len() {
        return Err(Error::invalid("voxel_sizes and max_distances must have the same length"));
    }
    if p.voxel_sizes.iter().any(|v| v.is_nan() || *v <= 0.0) {
        return Err(Error::invalid("voxel size must be positive"));
    }
    if reference >= n || edges.iter().any(|&(i, j)| i >= n || j >= n) {
        return Err(Error::invalid("scan index out of range"));
    }
    let mut poses = poses.to_vec();
    let start = poses.clone();
    let mut rng = Generator::new(p.seed);
    let edges: Vec<(usize, usize)> = edges.iter().copied().filter(|&(i, j)| i != j && !points[i].is_empty() && !points[j].is_empty()).collect();
    if stems.iter().skip(n).any(|s| !s.is_empty()) {
        return Err(Error::invalid("more stem lists than poses"));
    }
    let mut stems: Vec<Vec<Point>> = stems.iter().take(n).cloned().collect();
    stems.resize(n, Vec::new());
    let mut total_corr = 0usize;
    let mut first_residual = f64::NAN;
    let mut last_residual = f64::NAN;

    for (&voxel, &max_dist) in p.voxel_sizes.iter().zip(&p.max_distances) {
        let mut levels: Vec<Level> = Vec::with_capacity(n);
        for pts_k in points.iter().take(n) {
            let mut pts = if pts_k.is_empty() {
                Vec::new()
            } else {
                let (pts, counts) = voxel_centroids(pts_k, voxel, true);
                if p.min_voxel_points > 1 {
                    pts.into_iter().zip(counts).filter(|(_, c)| *c >= p.min_voxel_points).map(|(q, _)| q).collect()
                } else {
                    pts
                }
            };
            if pts.len() > p.points_per_scan {
                let mut pick = rng.choice(pts.len(), p.points_per_scan);
                pick.sort_unstable();
                pts = pick.into_iter().map(|i| pts[i]).collect();
            }
            let (nrm, ok) = if pts.len() >= p.normal_neighbours {
                let (nrm, planarity) = estimate_normals(&pts, p.normal_neighbours, Some(3.0 * voxel));
                let ok = planarity.iter().map(|&v| v >= p.min_planarity).collect();
                (nrm, ok)
            } else {
                (vec![[0.0; 3]; pts.len()], vec![false; pts.len()])
            };
            levels.push(Level { pts, nrm, ok });
        }
        let cos_limit = p.max_normal_angle_deg.to_radians().cos();

        for round_no in 0..p.rounds {
            let world: Vec<Vec<Point>> = (0..n).map(|k| transform_points(&poses[k], &levels[k].pts)).collect();
            let wnrm: Vec<Vec<Point>> = (0..n).map(|k| transform_vectors(&poses[k], &levels[k].nrm)).collect();
            let mut trees: BTreeMap<usize, CoregTree> = BTreeMap::new();
            let mut corr: Vec<Corr> = Vec::new();
            for &(i, j) in &edges {
                for (a, b) in [(i, j), (j, i)] {
                    let idx_b: Vec<usize> = (0..levels[b].ok.len()).filter(|&k| levels[b].ok[k]).collect();
                    if idx_b.is_empty() || !levels[a].ok.iter().any(|&v| v) {
                        continue;
                    }
                    let tree = trees.entry(b).or_insert_with(|| CoregTree::new(&idx_b.iter().map(|&k| world[b][k]).collect::<Vec<_>>()));
                    let mut src_idx: Vec<usize> = (0..levels[a].ok.len()).filter(|&k| levels[a].ok[k]).collect();
                    if src_idx.len() > p.correspondences_per_pair {
                        let mut pick = rng.choice(src_idx.len(), p.correspondences_per_pair);
                        pick.sort_unstable();
                        src_idx = pick.into_iter().map(|k| src_idx[k]).collect();
                    }
                    let queries: Vec<Point> = src_idx.iter().map(|&k| world[a][k]).collect();
                    let (d, m) = tree.query(&queries, max_dist);
                    let hits: Vec<(usize, usize)> = (0..src_idx.len()).filter(|&k| d[k].is_finite()).map(|k| (src_idx[k], idx_b[m[k]])).collect();
                    if hits.is_empty() {
                        continue;
                    }
                    let keep: Vec<(usize, usize)> = hits.into_iter().filter(|&(s, t)| dot3(&wnrm[a][s], &wnrm[b][t]).abs() >= cos_limit).collect();
                    if keep.len() < 20 {
                        continue;
                    }
                    corr.push(Corr {
                        a,
                        b,
                        p: keep.iter().map(|&(s, _)| levels[a].pts[s]).collect(),
                        q: keep.iter().map(|&(_, t)| levels[b].pts[t]).collect(),
                        nq: keep.iter().map(|&(_, t)| levels[b].nrm[t]).collect(),
                    });
                }
            }
            let stems_by_pair = match_stems(&poses, &stems, &edges, p.stem_radius);
            let n_corr: usize = corr.iter().map(|c| c.p.len()).sum();
            total_corr = total_corr.max(n_corr);
            if n_corr == 0 {
                log(&format!("  level {} m: no correspondences", fixed2(voxel)));
                break;
            }

            let mut damping = 1e-4;
            let pr = 1.0 / p.prior_rotation_deg.to_radians().powi(2);
            let pt = 1.0 / p.prior_translation.powi(2);
            let prior_w = [pr, pr, pr, pt, pt, pt];
            for _ in 0..p.iterations {
                let Assembled { mut h, mut g, mut cost, median: med, scales } = assemble(&poses, &corr, &stems_by_pair, n, p.robust_scale, p.stem_weight, true, None, p.stem_scale);
                for k in (0..n).filter(|&k| k != reference) {
                    let xi = twist_between(&start[k], &poses[k]);
                    let mut sq = [0.0; 6];
                    for q in 0..6 {
                        h[(6 * k + q, 6 * k + q)] += prior_w[q];
                        g[6 * k + q] += prior_w[q] * xi[q];
                        sq[q] = prior_w[q] * (xi[q] * xi[q]);
                    }
                    cost += 0.5 * numpy_sum(&sq);
                }
                if first_residual.is_nan() {
                    first_residual = med;
                }
                last_residual = med;
                let free: Vec<usize> = (0..6 * n).filter(|&c| c / 6 != reference && !levels[c / 6].pts.is_empty()).collect();
                let hf = DMatrix::from_fn(free.len(), free.len(), |r, c| h[(free[r], free[c])]);
                let gf = DVector::from_iterator(free.len(), free.iter().map(|&r| -g[r]));
                let mut accepted = false;
                let (mut scale, mut rot, mut tr) = (0.0, 0.0, 0.0);
                for _attempt in 0..6 {
                    let mut damped = hf.clone();
                    for k in 0..free.len() {
                        damped[(k, k)] += damping * pymax(hf[(k, k)], 1e-9);
                    }
                    let Some(delta) = solve(&damped, &gf) else {
                        damping *= 10.0;
                        continue;
                    };
                    let mut full = vec![0.0; 6 * n];
                    for (k, &c) in free.iter().enumerate() {
                        full[c] = delta[k];
                    }
                    let norm3 = |v: &[f64]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    rot = (1..n).fold(norm3(&full[0..3]), |m, k| pymax(m, norm3(&full[6 * k..6 * k + 3])));
                    tr = (1..n).fold(norm3(&full[3..6]), |m, k| pymax(m, norm3(&full[6 * k + 3..6 * k + 6])));
                    scale = 1.0;
                    for cand in [p.max_step_rotation_deg.to_radians() / pymax(rot, 1e-12), p.max_step_translation / pymax(tr, 1e-12)] {
                        if cand < scale {
                            scale = cand;
                        }
                    }
                    for v in full.iter_mut() {
                        *v *= scale;
                    }
                    let trial: Vec<Mat4> = (0..n)
                        .map(|k| {
                            let xi = Vector6::from_column_slice(&full[6 * k..6 * k + 6]);
                            if xi.iter().any(|&v| v != 0.0) {
                                se3_exp(&xi) * poses[k]
                            } else {
                                poses[k]
                            }
                        })
                        .collect();
                    let t = assemble(&trial, &corr, &stems_by_pair, n, p.robust_scale, p.stem_weight, false, Some(&scales), p.stem_scale);
                    let mut trial_cost = t.cost;
                    for k in (0..n).filter(|&k| k != reference) {
                        let xi = twist_between(&start[k], &trial[k]);
                        let sq: [f64; 6] = std::array::from_fn(|q| prior_w[q] * (xi[q] * xi[q]));
                        trial_cost += 0.5 * numpy_sum(&sq);
                    }
                    if trial_cost <= cost {
                        poses = trial;
                        last_residual = t.median;
                        damping = pymax(damping / 3.0, 1e-6);
                        accepted = true;
                        break;
                    }
                    damping *= 10.0;
                }
                if !accepted || scale * pymax(rot, tr) < 1e-5 {
                    break;
                }
            }
            log(&format!("  level {} m, association {}/{}: {} correspondences, median |residual| {} cm", fixed2(voxel), round_no + 1, p.rounds, thousands(n_corr), fixed2(last_residual * 100.0)));
        }
    }

    let mut shifts = vec![0.0; n];
    let mut rotations = vec![0.0; n];
    for k in 0..n {
        let (rot, tr) = transform_difference(&start[k], &poses[k]);
        shifts[k] = tr;
        rotations[k] = rot.to_degrees();
    }
    Ok(JointRefinement { poses, shifts, rotations, residual_before: first_residual, residual_after: last_residual, correspondences: total_corr })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    /// Ground and two walls seen from two scans, the second started 5 cm off.
    fn scene() -> (Vec<Vec<Point>>, Vec<Mat4>, Mat4) {
        let mut world = Vec::new();
        for i in 0..60 {
            for j in 0..60 {
                let (x, y) = (-3.0 + 0.1 * i as f64, -3.0 + 0.1 * j as f64);
                world.push([x, y, 0.02 * x]);
                world.push([x, 3.0, 0.05 * (i + j) as f64 % 3.0]);
                world.push([3.0, y, 0.05 * (i * 7 + j) as f64 % 3.0]);
            }
        }
        let truth = se3_exp(&Vector6::new(0.0, 0.0, 0.3, 1.0, 0.5, 0.0));
        let local = transform_points(&invert(&truth), &world);
        let off = se3_exp(&Vector6::new(0.0, 0.0, 0.003, 0.03, -0.04, 0.01)) * truth;
        (vec![world, local], vec![Matrix4::identity(), off], truth)
    }

    #[test]
    fn pulls_a_scan_back_onto_its_neighbour() {
        let (points, poses, truth) = scene();
        let p = RefineParams { voxel_sizes: vec![0.2, 0.1], max_distances: vec![0.3, 0.15], prior_translation: 1.0, prior_rotation_deg: 10.0, ..Default::default() };
        let mut messages = Vec::new();
        let r = refine_joint(&points, &poses, &[(0, 1)], &[], 0, &p, &mut |m| messages.push(m.to_string())).unwrap();
        let (_, before) = transform_difference(&poses[1], &truth);
        let (_, after) = transform_difference(&r.poses[1], &truth);
        assert!(after < 0.3 * before, "{before} -> {after}");
        assert_eq!(r.poses[0], poses[0]);
        assert_eq!(messages.len(), 6);
        assert!(r.residual_after < r.residual_before);
    }

    #[test]
    fn formats_as_python() {
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(thousands(12), "12");
        assert_eq!(fixed2(0.1), "0.10");
        assert_eq!(fixed2(f64::NAN), "nan");
        // Ties round half to even on the exact binary value, as Python does.
        assert_eq!(fixed2(0.125), "0.12");
        assert_eq!(fixed2(0.375), "0.38");
        assert_eq!(fixed2(2.675), "2.67");
    }
}
