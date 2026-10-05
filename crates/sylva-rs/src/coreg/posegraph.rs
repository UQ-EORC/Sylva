// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Multi-scan global optimisation over a pose graph.
//!
//! Nodes are `world_from_scan` poses; edges are measured relative transforms
//! (`j_from_i`) with a 6x6 information matrix, rotation block first. The
//! graph is solved by Levenberg-Marquardt (Levenberg 1944; Marquardt 1963)
//! on SE(3), with right perturbations `pose @ se3_exp(delta)`, a Huber (1964)
//! kernel on each edge's Mahalanobis error and solve-and-reject passes that
//! drop edges far above the median error without cutting a node off from
//! the anchors it reached (Lu & Milios 1997 for the formulation). Edge
//! Jacobians are forward differences. The arithmetic and every decision
//! follow the NumPy implementation this replaced (`sylva.coreg.posegraph`).

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Matrix6, Vector3, Vector6};

use crate::coreg::transforms::{invert, se3_exp, se3_log, skew, solve, Mat4};
use crate::util::numeric::median;

/// A measured relative transform: `transform` maps scan `i` into scan `j`.
#[derive(Debug, Clone)]
pub struct Edge {
    pub i: usize,
    pub j: usize,
    pub transform: Mat4,
    pub information: Matrix6<f64>,
    /// Scalar trust for the spanning-tree initialisation.
    pub weight: f64,
}

/// Python's `max(a, b)`: `a` unless `b` is larger (so NaN in `a` stays).
fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Diagonal information from registration quality: translation precision
/// is the residual over the square root of the correspondence count,
/// rotation precision that spread over the scan's `extent` (m).
pub fn default_information(rmse: f64, fitness: f64, n_correspondences: i64, extent: f64) -> Matrix6<f64> {
    let n = n_correspondences.max(1) as f64;
    let sigma_t = pymax(rmse, 1e-4) / n.sqrt();
    let sigma_r = sigma_t / pymax(extent, 1e-3);
    let scale = pymax(pymin(fitness, 1.0), 1e-3);
    let mut info = Matrix6::identity();
    for k in 0..3 {
        info[(k, k)] *= scale / (sigma_r * sigma_r);
        info[(k + 3, k + 3)] *= scale / (sigma_t * sigma_t);
    }
    info
}

/// The 6x6 adjoint of a transform on `[omega, v]` twists:
/// `T se3_exp(xi) T^-1 = se3_exp(adjoint(T) xi)`.
pub fn adjoint(t: &Mat4) -> Matrix6<f64> {
    let r: Matrix3<f64> = t.fixed_view::<3, 3>(0, 0).into();
    let s = skew(&Vector3::new(t[(0, 3)], t[(1, 3)], t[(2, 3)])) * r;
    let mut a = Matrix6::zeros();
    a.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    a.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    a.fixed_view_mut::<3, 3>(3, 0).copy_from(&s);
    a
}

/// Edge information from the point-to-plane correspondences of an ICP.
///
/// `hessian` (`sum w a a^T` for updates `se3_exp(xi) @ transform`) is scaled
/// to the residual `sigma` (floored at `min_sigma`), with `n`
/// correspondences counting as `n / patch_points` independent ones, and
/// carried from the target frame into the frame of the edge residual
/// `se3_log(transform^-1 @ target_from_source)`. A small ridge keeps a
/// direction no surface constrains from making the normal equations
/// singular.
pub fn plane_edge_information(hessian: &Matrix6<f64>, sigma: f64, n: i64, transform: &Mat4, patch_points: f64, min_sigma: f64) -> Matrix6<f64> {
    let h = hessian / n.max(1) as f64;
    let count = pymax(n as f64 / pymax(patch_points, 1.0), 1.0);
    let s = pymax(sigma, min_sigma);
    let info_target = h * count / (s * s);
    let a = adjoint(transform);
    let info = a.transpose() * info_target * a;
    let info = 0.5 * (info + info.transpose());
    let ridge = 1e-9 * pymax(info.trace(), 1e-12);
    info + Matrix6::identity() * ridge
}

/// 6-vector error of one edge under `poses`.
pub fn residual(edge: &Edge, poses: &[Mat4]) -> Vector6<f64> {
    se3_log(&(invert(&edge.transform) * invert(&poses[edge.j]) * poses[edge.i]))
}

fn chi2(r: &Vector6<f64>, info: &Matrix6<f64>) -> f64 {
    (r.transpose() * info * r)[(0, 0)]
}

/// Sum of squared Mahalanobis edge errors over `edges` (indices).
pub fn total_error(edges: &[Edge], subset: &[usize], poses: &[Mat4]) -> f64 {
    let mut total = 0.0;
    for &k in subset {
        let e = &edges[k];
        total += chi2(&residual(e, poses), &e.information);
    }
    total
}

/// Mahalanobis error `sqrt(max(r^T I r, 0))` of every edge.
pub fn edge_errors(edges: &[Edge], poses: &[Mat4]) -> Vec<f64> {
    edges.iter().map(|e| pymax(chi2(&residual(e, poses), &e.information), 0.0).sqrt()).collect()
}

fn adjacency(edges: &[Edge], subset: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &k in subset {
        adj[edges[k].i].push(edges[k].j);
        adj[edges[k].j].push(edges[k].i);
    }
    adj
}

/// Connected components, as sorted lists of nodes, in order of their
/// smallest node.
pub fn components(n: usize, edges: &[Edge]) -> Vec<Vec<usize>> {
    let all: Vec<usize> = (0..edges.len()).collect();
    let adj = adjacency(edges, &all, n);
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut group = Vec::new();
        seen[start] = true;
        while let Some(node) = stack.pop() {
            group.push(node);
            for &nb in &adj[node] {
                if !seen[nb] {
                    seen[nb] = true;
                    stack.push(nb);
                }
            }
        }
        group.sort_unstable();
        out.push(group);
    }
    out
}

/// Nodes that reach an anchor through `subset`.
fn reachable(edges: &[Edge], subset: &[usize], n: usize, anchors: &BTreeSet<usize>) -> BTreeSet<usize> {
    let adj = adjacency(edges, subset, n);
    let mut seen = anchors.clone();
    let mut stack: Vec<usize> = anchors.iter().copied().collect();
    while let Some(node) = stack.pop() {
        for &nb in &adj[node] {
            if seen.insert(nb) {
                stack.push(nb);
            }
        }
    }
    seen
}

/// Initial poses from a maximum-weight spanning tree grown from the anchors:
/// the strongest edge leaving the visited set is chained first. Nodes with
/// no path to an anchor keep the identity; `fixed` nodes keep their poses.
pub fn initialise(n: usize, edges: &[Edge], reference: usize, fixed: &[(usize, Mat4)]) -> Vec<Mat4> {
    let mut adj: Vec<Vec<(f64, usize)>> = vec![Vec::new(); n];
    for (k, e) in edges.iter().enumerate() {
        adj[e.i].push((e.weight, k));
        adj[e.j].push((e.weight, k));
    }
    let mut poses = vec![Matrix4::identity(); n];
    for (k, p) in fixed {
        poses[*k] = *p;
    }
    let mut visited: BTreeSet<usize> = fixed.iter().map(|(k, _)| *k).collect();
    visited.insert(reference);
    let by_weight = |a: &(f64, usize), b: &(f64, usize)| (-b.0).partial_cmp(&-a.0).map_or(std::cmp::Ordering::Equal, |o| o.reverse());
    let mut frontier: Vec<(f64, usize)> = visited.iter().flat_map(|&a| adj[a].iter().copied()).collect();
    frontier.sort_by(by_weight);
    while !frontier.is_empty() {
        frontier.sort_by(by_weight);
        let (_, k) = frontier.remove(0);
        let e = &edges[k];
        let (vi, vj) = (visited.contains(&e.i), visited.contains(&e.j));
        if vi && vj {
            continue;
        }
        let unknown = if vi {
            poses[e.j] = poses[e.i] * invert(&e.transform);
            e.j
        } else {
            poses[e.i] = poses[e.j] * e.transform;
            e.i
        };
        visited.insert(unknown);
        frontier.extend(adj[unknown].iter().copied());
    }
    poses
}

/// Settings of [`optimise`].
#[derive(Debug, Clone)]
pub struct OptimiseParams {
    /// Cap per rejection pass.
    pub max_iterations: usize,
    /// Stop when an iteration improves the error by less than this fraction of it.
    pub tolerance: f64,
    /// Mahalanobis distance beyond which an edge is down-weighted.
    pub huber_delta: f64,
    pub reject_outliers: bool,
    /// Robust standard deviations above the median that make an outlier.
    pub outlier_sigma: f64,
    pub max_rejection_passes: i64,
}

impl Default for OptimiseParams {
    fn default() -> Self {
        OptimiseParams { max_iterations: 200, tolerance: 1e-6, huber_delta: 3.0, reject_outliers: true, outlier_sigma: 5.0, max_rejection_passes: 2 }
    }
}

/// Diagnostics of [`optimise`].
#[derive(Debug, Clone)]
pub struct Optimisation {
    pub poses: Vec<Mat4>,
    pub iterations: usize,
    pub converged: bool,
    pub initial_error: f64,
    pub final_error: f64,
    /// Indices of the rejected edges, ascending.
    pub rejected_edges: Vec<usize>,
    /// Mahalanobis error of every edge at the solution.
    pub edge_errors: Vec<f64>,
}

struct Graph<'a> {
    n: usize,
    edges: &'a [Edge],
    anchors: BTreeSet<usize>,
    huber_delta: f64,
}

impl Graph<'_> {
    fn huber_error(&self, subset: &[usize], poses: &[Mat4]) -> f64 {
        let d = self.huber_delta;
        let mut total = 0.0;
        for &k in subset {
            let e = &self.edges[k];
            let c = chi2(&residual(e, poses), &e.information);
            total += if c > d * d { 2.0 * d * c.sqrt() - d * d } else { c };
        }
        total
    }

    fn jacobians(&self, edge: &Edge, poses: &[Mat4], base: &Vector6<f64>, steps: &[Mat4; 6]) -> (Matrix6<f64>, Matrix6<f64>) {
        let eps = 1e-5;
        let mut ji = Matrix6::zeros();
        let mut jj = Matrix6::zeros();
        let mut trial = poses.to_vec();
        for (k, step) in steps.iter().enumerate() {
            trial[edge.i] = poses[edge.i] * step;
            let col = (residual(edge, &trial) - base) / eps;
            ji.set_column(k, &col);
            trial[edge.i] = poses[edge.i];
            trial[edge.j] = poses[edge.j] * step;
            let col = (residual(edge, &trial) - base) / eps;
            jj.set_column(k, &col);
            trial[edge.j] = poses[edge.j];
        }
        (ji, jj)
    }

    /// Levenberg-Marquardt over the graph: move every scan a little, each
    /// step, so that the measured transform on each edge is better satisfied.
    ///
    /// Anchored scans are held where they are, so `slot` maps a scan to its
    /// six columns in the system being solved, or `usize::MAX` for one that is
    /// not solved for at all.
    fn run_lm(&self, poses: &mut Vec<Mat4>, subset: &[usize], max_iterations: usize, tolerance: f64) -> (usize, bool) {
        let free: Vec<usize> = (0..self.n).filter(|k| !self.anchors.contains(k)).collect();
        let mut slot = vec![usize::MAX; self.n];
        for (k, &node) in free.iter().enumerate() {
            slot[node] = k;
        }
        let dim = 6 * free.len();
        if dim == 0 {
            return (0, true);
        }
        // Six small nudges, one per degree of freedom, used to work out the
        // derivatives numerically: how much the error on an edge changes when
        // a scan is rotated or shifted a touch.
        let steps: [Mat4; 6] = std::array::from_fn(|k| {
            let mut xi = Vector6::zeros();
            xi[k] = 1e-5;
            se3_exp(&xi)
        });
        let d = self.huber_delta;
        let mut lam = 1e-4;
        let mut error = self.huber_error(subset, poses);
        let mut improvement = 0.0;
        for iteration in 0..max_iterations {
            let mut h = DMatrix::zeros(dim, dim);
            let mut b = DVector::zeros(dim);
            for &k in subset {
                // How far this edge's measured transform is from what the
                // current poses imply, and how much to trust it: Huber
                // weighting leaves ordinary edges alone and pulls the
                // influence of a badly wrong one down towards nothing, so one
                // mismatched pair cannot bend the whole survey.
                let e = &self.edges[k];
                let r = residual(e, poses);
                let c = chi2(&r, &e.information);
                let w = if c <= d * d { 1.0 } else { d / pymax(c, 1e-12).sqrt() };
                let omega = e.information * w;
                let (ji, jj) = self.jacobians(e, poses, &r, &steps);
                let blocks: Vec<(usize, Matrix6<f64>)> = [(e.i, ji), (e.j, jj)].into_iter().filter(|(n, _)| slot[*n] != usize::MAX).map(|(n, j)| (slot[n], j)).collect();
                for (si, ja) in &blocks {
                    let jt_omega = ja.transpose() * omega;
                    let g = jt_omega * r;
                    for q in 0..6 {
                        b[6 * si + q] -= g[q];
                    }
                    for (sj, jb) in &blocks {
                        let blk = jt_omega * jb;
                        for p in 0..6 {
                            for q in 0..6 {
                                h[(6 * si + p, 6 * sj + q)] += blk[(p, q)];
                            }
                        }
                    }
                }
            }
            let diagonal: Vec<f64> = (0..dim).map(|k| pymax(h[(k, k)], 1e-12)).collect();
            let mut accepted = false;
            for _ in 0..12 {
                let mut damped = h.clone();
                for k in 0..dim {
                    damped[(k, k)] += lam * diagonal[k];
                }
                let delta = solve(&damped, &b);
                let Some(delta) = delta.filter(|x| x.iter().all(|v| v.is_finite())) else {
                    lam *= 10.0;
                    continue;
                };
                let mut candidate = poses.clone();
                for (k, &node) in free.iter().enumerate() {
                    let xi = Vector6::from_iterator((0..6).map(|q| delta[6 * k + q]));
                    candidate[node] *= se3_exp(&xi);
                }
                let new_error = self.huber_error(subset, &candidate);
                if new_error <= error {
                    *poses = candidate;
                    improvement = error - new_error;
                    error = new_error;
                    lam = pymax(lam * 0.5, 1e-12);
                    accepted = true;
                    break;
                }
                lam *= 10.0;
            }
            if !accepted {
                return (iteration + 1, false);
            }
            if improvement < tolerance * pymax(error, 1.0) {
                return (iteration + 1, true);
            }
        }
        (max_iterations, false)
    }

    /// Edges of `subset` far above the median error, never disconnecting a
    /// node from the anchors it reached, worst first.
    fn find_outliers(&self, subset: &[usize], poses: &[Mat4], sigma: f64) -> Vec<usize> {
        if subset.len() < 4 {
            return Vec::new();
        }
        let errors: Vec<f64> = subset.iter().map(|&k| pymax(chi2(&residual(&self.edges[k], poses), &self.edges[k].information), 0.0).sqrt()).collect();
        let med = median(&errors);
        let dev: Vec<f64> = errors.iter().map(|e| (e - med).abs()).collect();
        let mad = median(&dev) * 1.4826;
        if mad <= 1e-9 {
            return Vec::new();
        }
        let threshold = med + sigma * mad;
        let mut candidates: Vec<(usize, f64)> = subset.iter().zip(&errors).filter(|(_, &e)| e > threshold).map(|(&k, &e)| (k, e)).collect();
        candidates.sort_by(|a, b| (-a.1).partial_cmp(&-b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept: Vec<usize> = subset.to_vec();
        let mut removed = Vec::new();
        let anchored = reachable(self.edges, &kept, self.n, &self.anchors);
        for (c, _) in candidates {
            let trial: Vec<usize> = kept.iter().copied().filter(|&k| k != c).collect();
            if reachable(self.edges, &trial, self.n, &self.anchors).is_superset(&anchored) {
                kept = trial;
                removed.push(c);
            }
        }
        removed
    }
}

fn allclose_identity(p: &Mat4) -> bool {
    let eye = Matrix4::<f64>::identity();
    p.iter().zip(eye.iter()).all(|(a, b)| (a - b).abs() <= 1e-8 + 1e-5 * b.abs())
}

/// Levenberg-Marquardt over all edges, with outlier rejection.
///
/// `poses` is the starting point; if every free node is still at the
/// identity the poses are first initialised by [`initialise`]. The world
/// frame is the `reference` node's, held fixed with the `fixed` nodes. Each
/// pass solves over the edges not yet rejected; between passes edges far
/// above the median error are rejected, so the poses returned never rest on
/// a rejected edge.
pub fn optimise(n: usize, edges: &[Edge], reference: usize, fixed: &[(usize, Mat4)], poses: Vec<Mat4>, p: &OptimiseParams) -> Optimisation {
    let mut poses = poses;
    if edges.is_empty() {
        return Optimisation { poses, iterations: 0, converged: true, initial_error: 0.0, final_error: 0.0, rejected_edges: Vec::new(), edge_errors: Vec::new() };
    }
    let mut anchors: BTreeSet<usize> = fixed.iter().map(|(k, _)| *k).collect();
    anchors.insert(reference);
    let graph = Graph { n, edges, anchors, huber_delta: p.huber_delta };
    if (0..n).filter(|k| !graph.anchors.contains(k)).all(|k| allclose_identity(&poses[k])) {
        poses = initialise(n, edges, reference, fixed);
    }
    let all: Vec<usize> = (0..edges.len()).collect();
    let initial_error = total_error(edges, &all, &poses);
    let mut rejected: Vec<usize> = Vec::new();
    let mut iterations = 0;
    let mut converged = false;
    let passes = if p.reject_outliers { p.max_rejection_passes.max(0) as usize } else { 0 };
    for pass in 0..=passes {
        let active: Vec<usize> = all.iter().copied().filter(|k| !rejected.contains(k)).collect();
        if active.is_empty() {
            break;
        }
        let (used, conv) = graph.run_lm(&mut poses, &active, p.max_iterations, p.tolerance);
        iterations += used;
        converged = conv;
        if pass == passes {
            break;
        }
        let new = graph.find_outliers(&active, &poses, p.outlier_sigma);
        if new.is_empty() {
            break;
        }
        rejected.extend(new);
    }
    let errors = edge_errors(edges, &poses);
    let dropped: BTreeSet<usize> = rejected.into_iter().collect();
    let kept: Vec<usize> = all.iter().copied().filter(|k| !dropped.contains(k)).collect();
    let final_error = total_error(edges, &kept, &poses);
    Optimisation { poses, iterations, converged, initial_error, final_error, rejected_edges: dropped.into_iter().collect(), edge_errors: errors }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pose `k` of a survey whose scan 0 is the world frame.
    fn pose(k: usize) -> Mat4 {
        let at = |f: f64| se3_exp(&Vector6::new(0.01 * f, -0.02 * f, 0.3 * f, 8.0 * f.cos(), 6.0 * f.sin(), 0.1 * f));
        invert(&at(0.0)) * at(k as f64)
    }

    fn edge(i: usize, j: usize, truth: &[Mat4], noise: f64) -> Edge {
        let n = se3_exp(&Vector6::new(noise, -noise, noise * 0.5, 3.0 * noise, noise, -2.0 * noise));
        Edge { i, j, transform: n * invert(&truth[j]) * truth[i], information: default_information(0.01, 0.5, 1000, 15.0), weight: 500.0 }
    }

    #[test]
    fn recovers_poses_and_rejects_a_wrong_edge() {
        let truth: Vec<Mat4> = (0..5).map(pose).collect();
        let mut edges = Vec::new();
        for i in 0..5 {
            for j in i + 1..5 {
                edges.push(edge(i, j, &truth, 1e-4 * (i + 2 * j) as f64));
            }
        }
        edges[3].transform = se3_exp(&Vector6::new(0.0, 0.0, 0.7, 4.0, -3.0, 0.0)) * edges[3].transform;
        let r = optimise(5, &edges, 0, &[], vec![Matrix4::identity(); 5], &OptimiseParams::default());
        assert_eq!(r.rejected_edges, vec![3]);
        assert!(r.final_error < r.initial_error);
        for k in 0..5 {
            assert!((r.poses[k] - truth[k]).abs().max() < 0.05);
        }
    }

    #[test]
    fn components_and_initialisation() {
        let truth: Vec<Mat4> = (0..4).map(pose).collect();
        let edges = vec![edge(0, 1, &truth, 0.0), edge(2, 1, &truth, 0.0)];
        assert_eq!(components(4, &edges), vec![vec![0, 1, 2], vec![3]]);
        let p = initialise(4, &edges, 0, &[]);
        assert!((p[2] - truth[2]).abs().max() < 1e-9);
        assert_eq!(p[3], Matrix4::identity());
    }

    #[test]
    fn plane_information_is_symmetric_and_regular() {
        let mut h = Matrix6::zeros();
        h[(2, 2)] = 100.0;
        h[(3, 3)] = 50.0;
        let info = plane_edge_information(&h, 0.002, 200, &pose(1), 100.0, 0.005);
        assert!((info - info.transpose()).abs().max() < 1e-9);
        assert!(info.cholesky().is_some());
    }
}
