// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Fine registration by iterative closest point, for forest scans.
//!
//! [`icp`] and [`evaluate_registration`], with the SE(3) helpers they use:
//!
//! * point-to-plane (Chen & Medioni 1992) with a planarity gate on the
//!   *target* point, or weighted Kabsch point-to-point (Besl & McKay 1992;
//!   Kabsch 1976);
//! * Huber (1964) / Tukey biweight (Beaton & Tukey 1974) IRLS weights with an
//!   adaptive scale floored at `robust_scale`, plus distance trimming in the
//!   spirit of trimmed ICP (Chetverikov et al. 2002), phased in over
//!   `trim_ramp` iterations (cutoff = numpy's default linear quantile);
//! * a damped Gauss-Newton step on the SE(3) tangent space, over a
//!   coarse-to-fine voxel pyramid.
//!
//! Random subsets (the `max_points` caps) are drawn with a seeded xorshift
//! ([`crate::filters`]'s RNG): a fresh generator per level for the target,
//! one shared generator for the source, and one shared by source and target
//! in the evaluation, so the subsets are reproducible.  Below the caps no
//! randomness is involved.

use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector6};
use rayon::prelude::*;

use crate::coreg::geometry::{estimate_normals, voxel_downsample, CoregTree};
use crate::error::{Error, Result};
use crate::filters::Rng;
use crate::Point;

/// 4x4 homogeneous transform (column-vector convention, `q = T [p, 1]`).
pub type Mat4 = Matrix4<f64>;

/// Tunables for [`icp`].
#[derive(Debug, Clone)]
pub struct IcpConfig {
    pub voxel_sizes: Vec<f64>,
    /// Correspondence cut-off per level; `None` means `5 x voxel`.
    pub max_distances: Option<Vec<f64>>,
    pub max_iterations: usize,
    /// `"point_to_plane"` or `"point_to_point"`.
    pub method: String,
    /// `"huber"`, `"tukey"` or `"none"`.
    pub robust: String,
    pub robust_scale: f64,
    pub trim_fraction: f64,
    pub trim_ramp: usize,
    pub min_planarity: f64,
    pub normal_neighbours: usize,
    pub translation_tolerance: f64,
    pub rotation_tolerance: f64,
    pub fitness_threshold: f64,
    pub damping: f64,
    pub max_points: usize,
    pub plateau_tolerance: f64,
    pub plateau_patience: usize,
    pub seed: u64,
}

impl Default for IcpConfig {
    fn default() -> Self {
        IcpConfig {
            voxel_sizes: vec![0.30, 0.15, 0.07, 0.05],
            max_distances: Some(vec![0.80, 0.40, 0.20, 0.12]),
            max_iterations: 30,
            method: "point_to_plane".into(),
            robust: "huber".into(),
            robust_scale: 0.05,
            trim_fraction: 0.85,
            trim_ramp: 3,
            min_planarity: 0.25,
            normal_neighbours: 20,
            translation_tolerance: 1e-4,
            rotation_tolerance: 2e-5,
            fitness_threshold: 0.10,
            damping: 1e-6,
            max_points: 120_000,
            plateau_tolerance: 0.0,
            plateau_patience: 3,
            seed: 0,
        }
    }
}

impl IcpConfig {
    pub fn distances(&self) -> Result<Vec<f64>> {
        match &self.max_distances {
            Some(d) => {
                if d.len() != self.voxel_sizes.len() {
                    return Err(Error::invalid("max_distances must have one entry per voxel size"));
                }
                Ok(d.clone())
            }
            None => Ok(self.voxel_sizes.iter().map(|v| 5.0 * v).collect()),
        }
    }

    fn validate(&self) -> Result<()> {
        if !matches!(self.method.as_str(), "point_to_plane" | "point_to_point") {
            return Err(Error::invalid(format!("unknown ICP method {:?}", self.method)));
        }
        if !matches!(self.robust.as_str(), "huber" | "tukey" | "none") {
            return Err(Error::invalid(format!("unknown robust loss {:?}", self.robust)));
        }
        if self.voxel_sizes.iter().any(|&v| v.is_nan() || v <= 0.0) {
            return Err(Error::invalid("voxel size must be positive"));
        }
        Ok(())
    }
}

/// Outcome of [`icp`]; `transform` maps source into target coordinates and
/// already includes the initial guess.
#[derive(Debug, Clone)]
pub struct IcpResult {
    pub transform: Matrix4<f64>,
    pub fitness: f64,
    pub inlier_rmse: f64,
    pub n_correspondences: usize,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<f64>,
    /// Point-to-plane information of the final transform ([`plane_information`]);
    /// `None` for point-to-point or too little overlap.
    pub information: Option<PlaneInformation>,
}

/// How well a set of point-to-plane correspondences pins a transform.
///
/// `hessian` is the robust-weighted Gauss-Newton matrix `sum w a a^T` of the
/// residuals `n . (T p - q)`, with `a = [p x n, n]` for a left-multiplied
/// update `exp(xi) T`, `xi = [omega, v]`: its null directions are the ones
/// the surfaces leave free (a slide along flat ground). `sigma` is the
/// weighted RMS residual and `n` the number of correspondences.
#[derive(Debug, Clone)]
pub struct PlaneInformation {
    pub hessian: Matrix6<f64>,
    pub sigma: f64,
    pub n: usize,
}

// --------------------------------------------------------------- SE(3) maps

/// The skew-symmetric matrix of a 3-vector.
pub fn skew(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0)
}

/// Rotation vector to rotation matrix (`so3_exp`).
pub fn so3_exp(w: &Vector3<f64>) -> Matrix3<f64> {
    let theta = w.norm();
    let k = skew(w);
    if theta < 1e-8 {
        return Matrix3::identity() + k + 0.5 * (k * k);
    }
    Matrix3::identity() + (theta.sin() / theta) * k + ((1.0 - theta.cos()) / (theta * theta)) * (k * k)
}

/// Rotation matrix to rotation vector (`so3_log`).
pub fn so3_log(r: &Matrix3<f64>) -> Vector3<f64> {
    let cos_theta = ((r.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let w = Vector3::new(r[(2, 1)] - r[(1, 2)], r[(0, 2)] - r[(2, 0)], r[(1, 0)] - r[(0, 1)]);
    if theta < 1e-8 {
        return 0.5 * w;
    }
    if std::f64::consts::PI - theta < 1e-6 {
        let a = (r + Matrix3::identity()) * 0.5;
        let mut axis = Vector3::new(a[(0, 0)].max(0.0).sqrt(), a[(1, 1)].max(0.0).sqrt(), a[(2, 2)].max(0.0).sqrt());
        let mut kmax = 0;
        for i in 1..3 {
            if axis[i] > axis[kmax] {
                kmax = i;
            }
        }
        if axis[kmax] > 1e-12 {
            axis = a.column(kmax) / axis[kmax];
        }
        axis /= axis.norm().max(1e-12);
        if w[0] * axis[0] + w[1] * axis[1] + w[2] * axis[2] < 0.0 {
            axis = -axis;
        }
        return axis * theta;
    }
    w * (theta / (2.0 * theta.sin()))
}

/// Left Jacobian of SO(3), which maps a twist's translation part.
pub fn left_jacobian(w: &Vector3<f64>) -> Matrix3<f64> {
    let theta = w.norm();
    let k = skew(w);
    if theta < 1e-8 {
        return Matrix3::identity() + 0.5 * k + (1.0 / 6.0) * (k * k);
    }
    let t2 = theta * theta;
    Matrix3::identity() + ((1.0 - theta.cos()) / t2) * k + ((theta - theta.sin()) / (t2 * theta)) * (k * k)
}

/// Twist `[w, t]` to a 4x4 transform (`se3_exp`).
pub fn se3_exp(xi: &Vector6<f64>) -> Matrix4<f64> {
    let w = Vector3::new(xi[0], xi[1], xi[2]);
    let u = Vector3::new(xi[3], xi[4], xi[5]);
    let mut t = Matrix4::identity();
    t.fixed_view_mut::<3, 3>(0, 0).copy_from(&so3_exp(&w));
    t.fixed_view_mut::<3, 1>(0, 3).copy_from(&(left_jacobian(&w) * u));
    t
}

/// Apply a 4x4 transform to points (`transform_points`), in parallel.
pub fn transform_points(t: &Matrix4<f64>, pts: &[Point]) -> Vec<Point> {
    let m = *t;
    pts.par_iter()
        .map(|p| {
            let mut o = [0.0; 3];
            for (i, oi) in o.iter_mut().enumerate() {
                *oi = p[0] * m[(i, 0)] + p[1] * m[(i, 1)] + p[2] * m[(i, 2)] + m[(i, 3)];
            }
            o
        })
        .collect()
}

/// Weighted Kabsch without scaling.  `None` for mismatched or degenerate
/// input.
pub fn kabsch_weighted(src: &[Point], dst: &[Point], weights: &[f64]) -> Option<Matrix4<f64>> {
    if src.len() != dst.len() || src.len() < 3 || weights.len() != src.len() {
        return None;
    }
    let wsum: f64 = weights.iter().sum();
    if wsum <= 1e-12 {
        return None;
    }
    let mut mu_s = Vector3::zeros();
    let mut mu_d = Vector3::zeros();
    for ((s, d), &w) in src.iter().zip(dst).zip(weights) {
        let w = w / wsum;
        mu_s += w * Vector3::new(s[0], s[1], s[2]);
        mu_d += w * Vector3::new(d[0], d[1], d[2]);
    }
    let mut h = Matrix3::zeros();
    for ((s, d), &w) in src.iter().zip(dst).zip(weights) {
        let w = w / wsum;
        let a = Vector3::new(s[0], s[1], s[2]) - mu_s;
        let b = (Vector3::new(d[0], d[1], d[2]) - mu_d) * w;
        h += a * b.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let det = (vt.transpose() * u.transpose()).determinant();
    let sign = if det > 0.0 { 1.0 } else if det < 0.0 { -1.0 } else { 0.0 };
    let dm = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, sign));
    let r = vt.transpose() * dm * u.transpose();
    let t = mu_d - r * mu_s;
    let mut out = Matrix4::identity();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    out.fixed_view_mut::<3, 1>(0, 3).copy_from(&t);
    if out.iter().all(|v| v.is_finite()) {
        Some(out)
    } else {
        None
    }
}

// ------------------------------------------------------------- statistics

/// `np.quantile(x, q)` with the default `"linear"` method.
pub fn quantile_linear(x: &[f64], q: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut v = x.to_vec();
    let virt = (n as f64 - 1.0) * q;
    if virt >= (n - 1) as f64 {
        return v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }
    if virt < 0.0 {
        return v.iter().cloned().fold(f64::INFINITY, f64::min);
    }
    let lo = virt.floor();
    let li = lo as usize;
    let (_, a, rest) = v.select_nth_unstable_by(li, |a, b| a.total_cmp(b));
    let a = *a;
    let b = rest.iter().cloned().fold(f64::INFINITY, f64::min);
    let gamma = virt - lo;
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// `np.median(x)`.
fn median(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut v = x.to_vec();
    let h = n / 2;
    let (lower, m, _) = v.select_nth_unstable_by(h, |a, b| a.total_cmp(b));
    let m = *m;
    if n % 2 == 1 {
        m
    } else {
        let a = lower.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (a + m) / 2.0
    }
}

/// Correspondence rejection (`_reject`): planarity gate and trimming phased
/// in over `trim_ramp` iterations.
pub fn reject(dist: &[f64], planarity: Option<&[f64]>, cfg: &IcpConfig, iteration: usize) -> Vec<bool> {
    let mut keep = vec![true; dist.len()];
    if let Some(pl) = planarity {
        if cfg.min_planarity > 0.0 {
            for (k, &p) in keep.iter_mut().zip(pl) {
                *k = *k && p >= cfg.min_planarity;
            }
        }
    }
    let ramp = if cfg.trim_ramp > 0 { (iteration as f64 / cfg.trim_ramp.max(1) as f64).min(1.0) } else { 1.0 };
    let fraction = 1.0 - (1.0 - cfg.trim_fraction) * ramp;
    let n_keep = keep.iter().filter(|&&k| k).count();
    if fraction < 1.0 && n_keep > 10 {
        let kept: Vec<f64> = dist.iter().zip(&keep).filter(|(_, &k)| k).map(|(&d, _)| d).collect();
        let cutoff = quantile_linear(&kept, fraction);
        for (k, &d) in keep.iter_mut().zip(dist) {
            *k = *k && d <= cutoff;
        }
    }
    keep
}

/// IRLS weights (`_weights`).
pub fn weights(residual: &[f64], cfg: &IcpConfig) -> Vec<f64> {
    if cfg.robust == "none" {
        return vec![1.0; residual.len()];
    }
    let a: Vec<f64> = residual.iter().map(|r| r.abs()).collect();
    let med = if a.is_empty() { 0.0 } else { median(&a) };
    let s = cfg.robust_scale.max(med).max(1e-9);
    if cfg.robust == "tukey" {
        return a
            .iter()
            .map(|&x| {
                let u = (x / (4.685 * s)).clamp(0.0, 1.0);
                (1.0 - u * u).powi(2)
            })
            .collect();
    }
    a.iter().map(|&x| if x <= s { 1.0 } else { s / x.max(1e-12) }).collect()
}

// ---------------------------------------------------------------- solvers

const CHUNK: usize = 8192;

/// One damped Gauss-Newton point-to-plane step (`_solve_point_to_plane`).
pub fn solve_point_to_plane(p: &[Point], q: &[Point], n: &[Point], cfg: &IcpConfig) -> Option<Matrix4<f64>> {
    let residual: Vec<f64> = p
        .par_iter()
        .zip(q.par_iter())
        .zip(n.par_iter())
        .map(|((p, q), n)| (p[0] - q[0]) * n[0] + (p[1] - q[1]) * n[1] + (p[2] - q[2]) * n[2])
        .collect();
    let w = weights(&residual, cfg);
    // Fixed chunks, summed in order: deterministic regardless of threads.
    let idx: Vec<usize> = (0..p.len()).collect();
    let parts: Vec<(Matrix6<f64>, Vector6<f64>)> = idx
        .par_chunks(CHUNK)
        .map(|c| {
            let mut h = Matrix6::zeros();
            let mut g = Vector6::zeros();
            for &i in c {
                let (pi, ni) = (&p[i], &n[i]);
                let a = Vector6::new(
                    pi[1] * ni[2] - pi[2] * ni[1],
                    pi[2] * ni[0] - pi[0] * ni[2],
                    pi[0] * ni[1] - pi[1] * ni[0],
                    ni[0],
                    ni[1],
                    ni[2],
                );
                let aw = a * w[i];
                h += a * aw.transpose();
                g -= aw * residual[i];
            }
            (h, g)
        })
        .collect();
    let mut h = Matrix6::zeros();
    let mut g = Vector6::zeros();
    for (hc, gc) in parts {
        h += hc;
        g += gc;
    }
    let tr = h.trace();
    let scale = if tr.is_finite() { tr / 6.0 } else { 1.0 };
    h += Matrix6::identity() * (cfg.damping * scale.max(1e-9));
    let mut xi = h.lu().solve(&g)?;
    if !xi.iter().all(|v| v.is_finite()) {
        return None;
    }
    let tn = Vector3::new(xi[3], xi[4], xi[5]).norm();
    if tn > 1.0 {
        xi *= 1.0 / tn;
    }
    Some(se3_exp(&xi))
}

/// Weighted Kabsch point-to-point step (`_solve_point_to_point`).
pub fn solve_point_to_point(p: &[Point], q: &[Point], d: &[f64], cfg: &IcpConfig) -> Option<Matrix4<f64>> {
    let w = weights(d, cfg);
    if w.iter().sum::<f64>() <= 0.0 || p.len() < 3 {
        return None;
    }
    kabsch_weighted(p, q, &w)
}

// -------------------------------------------------------------------- ICP

fn random_cap(points: Vec<Point>, max_points: usize, rng: &mut Rng) -> Vec<Point> {
    let total = points.len();
    if total <= max_points {
        return points;
    }
    let mut idx: Vec<usize> = (0..total).collect();
    for i in 0..max_points {
        let j = i + rng.below(total - i);
        idx.swap(i, j);
    }
    idx.truncate(max_points);
    idx.sort_unstable();
    idx.into_iter().map(|i| points[i]).collect()
}

struct Level {
    points: Vec<Point>,
    tree: CoregTree,
    normals: Option<Vec<Point>>,
    planarity: Option<Vec<f64>>,
}

fn build_level(target: &[Point], voxel: f64, cfg: &IcpConfig) -> Level {
    let mut points = voxel_downsample(target, voxel);
    if points.len() > cfg.max_points {
        // A fresh generator seeded with cfg.seed for every level.
        points = random_cap(points, cfg.max_points, &mut Rng::new(cfg.seed));
    }
    let (mut normals, mut planarity) = (None, None);
    if cfg.method == "point_to_plane" && points.len() >= cfg.normal_neighbours {
        let (n, p) = estimate_normals(&points, cfg.normal_neighbours, Some(3.0 * voxel));
        normals = Some(n);
        planarity = Some(p);
    }
    let tree = CoregTree::new(&points);
    Level { points, tree, normals, planarity }
}

/// A target's ICP pyramid (voxel centroids, normals, planarity and search tree
/// per level), built once and reused for every scan registered against it.
///
/// Each level depends only on the target points and the pyramid settings (the
/// random cap is drawn from a fresh generator per level), so registering against
/// a prepared target gives exactly the result of [`icp`].
pub struct IcpTarget {
    points: Vec<Point>,
    voxel_sizes: Vec<f64>,
    max_points: usize,
    method: String,
    normal_neighbours: usize,
    seed: u64,
    levels: Vec<Level>,
}

impl IcpTarget {
    /// Build the pyramid of `target` for `cfg`.
    pub fn new(target: &[Point], cfg: &IcpConfig) -> Self {
        let levels = if target.len() < 10 { Vec::new() } else { cfg.voxel_sizes.iter().map(|&v| build_level(target, v, cfg)).collect() };
        IcpTarget {
            points: target.to_vec(),
            voxel_sizes: cfg.voxel_sizes.clone(),
            max_points: cfg.max_points,
            method: cfg.method.clone(),
            normal_neighbours: cfg.normal_neighbours,
            seed: cfg.seed,
            levels,
        }
    }

    /// Number of target points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the target holds no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Was this pyramid built with the settings of `cfg`?
    pub fn matches(&self, cfg: &IcpConfig) -> bool {
        self.voxel_sizes == cfg.voxel_sizes && self.max_points == cfg.max_points && self.method == cfg.method && self.normal_neighbours == cfg.normal_neighbours && self.seed == cfg.seed
    }
}

/// Align `source` onto `target`.
pub fn icp(source: &[Point], target: &[Point], initial: Option<Matrix4<f64>>, cfg: &IcpConfig) -> Result<IcpResult> {
    cfg.validate()?;
    cfg.distances()?;
    if source.len() < 10 || target.len() < 10 {
        return icp_prepared(source, &IcpTarget { points: target.to_vec(), voxel_sizes: cfg.voxel_sizes.clone(), max_points: cfg.max_points, method: cfg.method.clone(), normal_neighbours: cfg.normal_neighbours, seed: cfg.seed, levels: Vec::new() }, initial, cfg);
    }
    icp_prepared(source, &IcpTarget::new(target, cfg), initial, cfg)
}

/// [`icp`] against a prepared target.
///
/// # Errors
/// If `cfg` is invalid, or `target` was built with other pyramid settings.
pub fn icp_prepared(source: &[Point], target: &IcpTarget, initial: Option<Matrix4<f64>>, cfg: &IcpConfig) -> Result<IcpResult> {
    cfg.validate()?;
    if !target.matches(cfg) {
        return Err(Error::invalid("the prepared ICP target was built with other pyramid settings"));
    }
    let distances = cfg.distances()?;
    let mut t = initial.unwrap_or_else(Matrix4::identity);
    if source.len() < 10 || target.points.len() < 10 {
        return Ok(IcpResult {
            transform: t,
            fitness: 0.0,
            inlier_rmse: f64::INFINITY,
            n_correspondences: 0,
            iterations: 0,
            converged: false,
            history: Vec::new(),
            information: None,
        });
    }

    let mut rng = Rng::new(cfg.seed);
    let mut total_iterations = 0usize;
    let mut converged = false;
    let mut history = Vec::new();
    let mut last_pairs = 0usize;

    for ((&voxel, &max_distance), level) in cfg.voxel_sizes.iter().zip(&distances).zip(&target.levels) {
        let src = random_cap(voxel_downsample(source, voxel), cfg.max_points, &mut rng);
        if src.len() < 10 || level.points.len() < 10 {
            continue;
        }

        let mut level_rmse = f64::INFINITY;
        let mut stalled = 0usize;
        for level_iteration in 0..cfg.max_iterations {
            total_iterations += 1;
            let moved = transform_points(&t, &src);
            let (dist, idx) = level.tree.query(&moved, max_distance);
            let valid: Vec<usize> = (0..moved.len()).filter(|&i| dist[i].is_finite()).collect();
            if valid.len() < 10 {
                break;
            }
            let p: Vec<Point> = valid.iter().map(|&i| moved[i]).collect();
            let q: Vec<Point> = valid.iter().map(|&i| level.points[idx[i]]).collect();
            let d: Vec<f64> = valid.iter().map(|&i| dist[i]).collect();

            let (keep, delta) = match (&level.normals, &level.planarity) {
                (Some(normals), Some(planarity)) if cfg.method == "point_to_plane" => {
                    let n: Vec<Point> = valid.iter().map(|&i| normals[idx[i]]).collect();
                    let pl: Vec<f64> = valid.iter().map(|&i| planarity[idx[i]]).collect();
                    let keep = reject(&d, Some(&pl), cfg, level_iteration);
                    if keep.iter().filter(|&&k| k).count() < 10 {
                        break;
                    }
                    let sel = |v: &[Point]| -> Vec<Point> { v.iter().zip(&keep).filter(|(_, &k)| k).map(|(x, _)| *x).collect() };
                    let delta = solve_point_to_plane(&sel(&p), &sel(&q), &sel(&n), cfg);
                    (keep, delta)
                }
                _ => {
                    let keep = reject(&d, None, cfg, level_iteration);
                    if keep.iter().filter(|&&k| k).count() < 10 {
                        break;
                    }
                    let sel = |v: &[Point]| -> Vec<Point> { v.iter().zip(&keep).filter(|(_, &k)| k).map(|(x, _)| *x).collect() };
                    let dk: Vec<f64> = d.iter().zip(&keep).filter(|(_, &k)| k).map(|(x, _)| *x).collect();
                    let delta = solve_point_to_point(&sel(&p), &sel(&q), &dk, cfg);
                    (keep, delta)
                }
            };
            let Some(delta) = delta else { break };
            t = delta * t;

            let residual: Vec<f64> = d.iter().zip(&keep).filter(|(_, &k)| k).map(|(x, _)| *x).collect();
            last_pairs = residual.len();
            let last_rmse = (residual.iter().map(|r| r * r).sum::<f64>() / residual.len() as f64).sqrt();
            history.push(last_rmse);

            let rot: Matrix3<f64> = delta.fixed_view::<3, 3>(0, 0).into_owned();
            let step_rotation = so3_log(&rot).norm();
            let step_translation = Vector3::new(delta[(0, 3)], delta[(1, 3)], delta[(2, 3)]).norm();
            if step_translation < cfg.translation_tolerance && step_rotation < cfg.rotation_tolerance {
                converged = true;
                break;
            }
            let step = step_translation + 10.0 * step_rotation;
            stalled = if step < level_rmse * cfg.plateau_tolerance { stalled + 1 } else { 0 };
            level_rmse = last_rmse;
            if stalled >= cfg.plateau_patience {
                break;
            }
        }
    }

    // The final evaluation uses the default seed (0), not cfg.seed.
    let (fitness, rmse, n_pairs) = evaluate_registration(source, &target.points, &t, cfg.fitness_threshold, 200_000, Some(0.05), 0);
    Ok(IcpResult {
        transform: t,
        fitness,
        inlier_rmse: rmse,
        n_correspondences: if n_pairs > 0 { n_pairs } else { last_pairs },
        iterations: total_iterations,
        converged,
        history,
        information: plane_information(source, target, &t, cfg),
    })
}

/// Point-to-plane information of `transform` against the finest level of
/// `target`: the correspondences, planarity gate, trim and robust weights of
/// a last ICP iteration, without the step.
///
/// `None` if the target has no normals (point-to-point) or fewer than 10
/// correspondences survive.
pub fn plane_information(source: &[Point], target: &IcpTarget, transform: &Mat4, cfg: &IcpConfig) -> Option<PlaneInformation> {
    let level = target.levels.last()?;
    let (normals, planarity) = (level.normals.as_ref()?, level.planarity.as_ref()?);
    let voxel = *cfg.voxel_sizes.last()?;
    let max_distance = *cfg.distances().ok()?.last()?;
    let src = random_cap(voxel_downsample(source, voxel), cfg.max_points, &mut Rng::new(cfg.seed));
    let moved = transform_points(transform, &src);
    let (dist, idx) = level.tree.query(&moved, max_distance);
    let valid: Vec<usize> = (0..moved.len()).filter(|&i| dist[i].is_finite()).collect();
    if valid.len() < 10 {
        return None;
    }
    let d: Vec<f64> = valid.iter().map(|&i| dist[i]).collect();
    let pl: Vec<f64> = valid.iter().map(|&i| planarity[idx[i]]).collect();
    let keep = reject(&d, Some(&pl), cfg, cfg.trim_ramp);
    let kept: Vec<usize> = valid.iter().zip(&keep).filter(|(_, &k)| k).map(|(&i, _)| i).collect();
    if kept.len() < 10 {
        return None;
    }
    let residual: Vec<f64> = kept
        .iter()
        .map(|&i| {
            let (p, q, n) = (&moved[i], &level.points[idx[i]], &normals[idx[i]]);
            (p[0] - q[0]) * n[0] + (p[1] - q[1]) * n[1] + (p[2] - q[2]) * n[2]
        })
        .collect();
    let w = weights(&residual, cfg);
    let mut hessian = Matrix6::zeros();
    let (mut wr2, mut wsum) = (0.0, 0.0);
    for (k, &i) in kept.iter().enumerate() {
        let (p, n) = (&moved[i], &normals[idx[i]]);
        let a = Vector6::new(p[1] * n[2] - p[2] * n[1], p[2] * n[0] - p[0] * n[2], p[0] * n[1] - p[1] * n[0], n[0], n[1], n[2]);
        hessian += a * a.transpose() * w[k];
        wr2 += w[k] * residual[k] * residual[k];
        wsum += w[k];
    }
    Some(PlaneInformation { hessian, sigma: (wr2 / wsum.max(1e-12)).sqrt(), n: kept.len() })
}

/// Score a registration without changing it (`evaluate_registration`):
/// `(fitness, inlier_rmse, n_inliers)`.
pub fn evaluate_registration(
    source: &[Point],
    target: &[Point],
    transform: &Matrix4<f64>,
    threshold: f64,
    max_points: usize,
    voxel: Option<f64>,
    seed: u64,
) -> (f64, f64, usize) {
    if source.is_empty() || target.is_empty() {
        return (0.0, f64::INFINITY, 0);
    }
    let (mut src, mut dst) = match voxel {
        Some(v) if v != 0.0 => (voxel_downsample(source, v), voxel_downsample(target, v)),
        _ => (source.to_vec(), target.to_vec()),
    };
    let mut rng = Rng::new(seed);
    src = random_cap(src, max_points, &mut rng);
    dst = random_cap(dst, max_points, &mut rng);
    let moved = transform_points(transform, &src);
    let tree = CoregTree::new(&dst);
    let (dist, _) = tree.query(&moved, threshold);
    let inl: Vec<f64> = dist.into_iter().filter(|d| d.is_finite()).collect();
    let n = inl.len();
    if n == 0 {
        return (0.0, f64::INFINITY, 0);
    }
    let rmse = (inl.iter().map(|d| d * d).sum::<f64>() / n as f64).sqrt();
    (n as f64 / src.len() as f64, rmse, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Ground with gentle relief plus vertical cylinders (stems).
    fn scene() -> Vec<Point> {
        let mut s = 42u64;
        let mut pts = Vec::new();
        for _ in 0..40_000 {
            let x = lcg(&mut s) * 30.0 - 15.0;
            let y = lcg(&mut s) * 30.0 - 15.0;
            pts.push([x, y, 0.3 * (0.2 * x).sin() + 0.2 * (0.15 * y).cos()]);
        }
        let stems = [(-8.0, -5.0, 0.20), (3.0, 7.0, 0.35), (9.0, -9.0, 0.15), (-2.0, 1.0, 0.25), (6.0, 2.0, 0.30), (-10.0, 9.0, 0.22)];
        for &(cx, cy, r) in &stems {
            for _ in 0..6_000 {
                let a = lcg(&mut s) * std::f64::consts::TAU;
                let z = lcg(&mut s) * 8.0;
                pts.push([cx + r * a.cos(), cy + r * a.sin(), z]);
            }
        }
        pts
    }

    fn truth() -> Matrix4<f64> {
        se3_exp(&Vector6::new(0.01, -0.008, 0.03, 0.12, -0.08, 0.05))
    }

    #[test]
    fn coreg_icp_recovers_rigid_transform() {
        let target = scene();
        // source = T^-1 target, so icp should return T.
        let t = truth();
        let inv = t.try_inverse().unwrap();
        let source = transform_points(&inv, &target);
        let err = |m: &Matrix4<f64>| {
            let d = t.try_inverse().unwrap() * m;
            let rot = so3_log(&d.fixed_view::<3, 3>(0, 0).into_owned()).norm();
            (Vector3::new(d[(0, 3)], d[(1, 3)], d[(2, 3)]).norm(), rot)
        };
        for method in ["point_to_plane", "point_to_point"] {
            // Default pyramid: voxel centroids of the two clouds are not the
            // same points, so recovery is close but not exact.
            let cfg = IcpConfig { method: method.into(), ..Default::default() };
            let r = icp(&source, &target, None, &cfg).unwrap();
            let (tr, rot) = err(&r.transform);
            // (point-to-point converges slowly on stems and ground)
            let (tol_t, tol_r) = if method == "point_to_plane" { (1e-4, 1e-5) } else { (2e-3, 2e-4) };
            assert!(tr < tol_t && rot < tol_r, "{method} default: {tr} m {rot} rad");
            assert!(r.fitness > 0.99, "{method}: fitness {}", r.fitness);
            // A final level finer than the point spacing keeps every point as
            // its own centroid: exact correspondences, exact recovery.  (Only
            // point-to-point: at that scale no neighbourhood passes the
            // 3 x voxel radius, so the planarity gate rejects everything and
            // the level stops.)
            if method != "point_to_point" {
                continue;
            }
            let cfg = IcpConfig {
                method: method.into(),
                voxel_sizes: vec![0.30, 0.15, 0.07, 0.05, 1e-5],
                max_distances: Some(vec![0.80, 0.40, 0.20, 0.12, 0.05]),
                ..Default::default()
            };
            let r = icp(&source, &target, None, &cfg).unwrap();
            let (tr, rot) = err(&r.transform);
            assert!(tr < 1e-9 && rot < 1e-10, "{method} exact: {tr} m {rot} rad");
            assert!(r.converged && r.fitness > 0.99, "{r:?}");
        }
    }

    #[test]
    fn coreg_trim_quantile_matches_numpy() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0];
        // np.quantile(x, [0.85, 0.3, 0.5, 1.0, 0.0])
        assert!((quantile_linear(&x, 0.85) - 5.5).abs() < 1e-15);
        assert!((quantile_linear(&x, 0.3) - 3.0).abs() < 1e-15);
        assert_eq!(quantile_linear(&x, 0.5), 4.0);
        assert_eq!(quantile_linear(&x, 1.0), 9.0);
        assert_eq!(quantile_linear(&x, 0.0), 1.0);
        let y: Vec<f64> = (0..20).map(|i| (i * i) as f64 * 0.01).collect();
        // np.quantile(y, 0.95) = 3.4295 ; 0.9 -> 2.9 * ... computed below
        let v: f64 = (20.0 - 1.0) * 0.95;
        let lo = v.floor() as usize;
        let g = v - lo as f64;
        let exp = y[lo + 1] - (y[lo + 1] - y[lo]) * (1.0 - g);
        assert_eq!(quantile_linear(&y, 0.95), exp);
        // trim ramp: iteration 0 keeps everything beyond the planarity gate
        let cfg = IcpConfig::default();
        let d: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(reject(&d, None, &cfg, 0).iter().all(|&k| k));
        let k3 = reject(&d, None, &cfg, 3);
        assert_eq!(k3.iter().filter(|&&k| k).count(), 85); // cutoff 84.15
        assert_eq!(median(&[1.0, 5.0, 2.0, 8.0]), 3.5);
    }

    #[test]
    fn coreg_so3_roundtrip() {
        for w in [Vector3::new(0.1, -0.2, 0.3), Vector3::new(1e-10, 0.0, 0.0), Vector3::new(0.0, 0.0, std::f64::consts::PI - 1e-8)] {
            let r = so3_exp(&w);
            assert!((so3_log(&r) - w).norm() < 1e-7, "{w:?}");
        }
    }
}
