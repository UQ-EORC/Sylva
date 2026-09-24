// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Local geometry for scan co-registration: voxel resampling, local PCA,
//! normals / planarity and the planarity filter.
//!
//! A faithful port of `tlsalign.preprocess` (`voxel_downsample`,
//! `_local_covariance_eigh`, `estimate_normals`, `planar_filter`), including
//! its ordering conventions, so results can be compared row for row:
//!
//! * voxel output is ordered by ascending packed voxel key (numpy's
//!   `np.unique` order) for centroids, and by first member index otherwise;
//! * centroids are sums in original point order divided by the count, the
//!   same arithmetic as `np.bincount(..., weights=...) / counts`;
//! * neighbourhoods include the point itself, and neighbours beyond `radius`
//!   are replaced by the point itself (so the covariance still divides by
//!   `k - 1` over `k` entries).
//!
//! Nearest-neighbour queries follow scipy's `cKDTree.query(k=1)` conventions
//! (see [`CoregTree`]).

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Matrix3, SymmetricEigen};
use rayon::prelude::*;
use std::num::NonZeroUsize;

use crate::Point;

// ------------------------------------------------------------------ kd-tree

/// k-d tree with scipy `cKDTree` query semantics.
///
/// `nearest` accepts a neighbour only if its *squared* distance is strictly
/// below `distance_upper_bound²` (scipy squares the bound for `p = 2` and
/// tests `d < bound`), and reports `(inf, n)` otherwise.
pub struct CoregTree {
    tree: Option<ImmutableKdTree<f64, 3>>,
    n: usize,
}

impl CoregTree {
    pub fn new(points: &[Point]) -> Self {
        let tree = if points.is_empty() {
            None
        } else {
            Some(ImmutableKdTree::new_from_slice(points).expect("kd-tree build"))
        };
        CoregTree { tree, n: points.len() }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Nearest neighbour within `upper` (exclusive): `(distance, index)`, or
    /// `(inf, n)` when there is none.
    #[inline]
    pub fn nearest(&self, p: &Point, upper: f64) -> (f64, usize) {
        let Some(tree) = &self.tree else { return (f64::INFINITY, self.n) };
        let r = tree.query(p).nearest_one::<SquaredEuclidean<f64>>().execute();
        let bound = if upper.is_finite() { upper * upper } else { upper };
        if r.distance.is_finite() && r.distance < bound {
            (r.distance.sqrt(), r.item as usize)
        } else {
            (f64::INFINITY, self.n)
        }
    }

    /// Nearest neighbour of every query, in parallel.
    pub fn query(&self, queries: &[Point], upper: f64) -> (Vec<f64>, Vec<usize>) {
        let r: Vec<(f64, usize)> = queries.par_iter().map(|q| self.nearest(q, upper)).collect();
        r.into_iter().unzip()
    }

    /// `k` nearest neighbours (self included), sorted by distance, as
    /// `(distance, index)`.
    pub fn knn(&self, p: &Point, k: usize) -> Vec<(f64, usize)> {
        let (Some(tree), Some(k)) = (&self.tree, NonZeroUsize::new(k)) else { return Vec::new() };
        tree.query(p)
            .nearest_n::<SquaredEuclidean<f64>>(k)
            .execute()
            .into_iter()
            .map(|r| (r.distance.sqrt(), r.item as usize))
            .collect()
    }
}

// ------------------------------------------------------------ voxel resampling

/// Voxel indices per point and the member lists of every voxel, ordered by
/// ascending packed key (lexicographic `(gx, gy, gz)`), each list in
/// ascending point index.
fn voxel_runs(points: &[Point], voxel: f64) -> Vec<Vec<usize>> {
    assert!(voxel > 0.0, "voxel size must be positive");
    let grid: Vec<[i64; 3]> = points
        .par_iter()
        .map(|p| [(p[0] / voxel).floor() as i64, (p[1] / voxel).floor() as i64, (p[2] / voxel).floor() as i64])
        .collect();
    // Subtracting the per-axis minimum does not change the lexicographic
    // order, which is exactly the order of tlsalign's packed int64 key (and of
    // its `np.unique(grid, axis=0)` fallback).
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.par_sort_unstable_by_key(|&i| (grid[i], i));
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut last: Option<[i64; 3]> = None;
    for i in order {
        if last != Some(grid[i]) {
            runs.push(Vec::new());
            last = Some(grid[i]);
        }
        runs.last_mut().unwrap().push(i);
    }
    runs
}

/// `tlsalign.preprocess.voxel_downsample(..., return_counts=True)`.
///
/// `centroid = true`: mean of each voxel's points, ordered by ascending voxel
/// key.  `centroid = false`: the first point (lowest original index) of each
/// voxel, ordered by that index.  Counts are row for row with the output.
pub fn voxel_centroids(points: &[Point], voxel: f64, centroid: bool) -> (Vec<Point>, Vec<usize>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut runs = voxel_runs(points, voxel);
    if !centroid {
        runs.par_sort_unstable_by_key(|r| r[0]);
        let out = runs.iter().map(|r| points[r[0]]).collect();
        let counts = runs.iter().map(|r| r.len()).collect();
        return (out, counts);
    }
    let out = runs
        .par_iter()
        .map(|r| {
            // Sequential sum in original order, then divide: np.bincount's arithmetic.
            let mut s = [0.0f64; 3];
            for &i in r {
                for k in 0..3 {
                    s[k] += points[i][k];
                }
            }
            let c = r.len() as f64;
            [s[0] / c, s[1] / c, s[2] / c]
        })
        .collect();
    let counts = runs.iter().map(|r| r.len()).collect();
    (out, counts)
}

/// Voxel centroids without counts.
pub fn voxel_downsample(points: &[Point], voxel: f64) -> Vec<Point> {
    voxel_centroids(points, voxel, true).0
}

// ---------------------------------------------------------------- local PCA

/// Local PCA of every point: ascending eigenvalues, eigenvectors (`evecs[i][j]`
/// is the unit eigenvector of `evals[i][j]`) and a validity mask.
pub struct LocalPca {
    pub evals: Vec<[f64; 3]>,
    pub evecs: Vec<[[f64; 3]; 3]>,
    pub valid: Vec<bool>,
}

/// Port of `tlsalign.preprocess._local_covariance_eigh`.
///
/// The `k = min(k, n)` nearest neighbours include the point itself.  With a
/// `radius`, neighbours further than it are replaced by the point itself and
/// `valid` requires at least 3 within it.  The covariance divides by
/// `max(k - 1, 1)`.
pub fn local_pca(points: &[Point], k: usize, radius: Option<f64>) -> LocalPca {
    let n = points.len();
    let k = k.min(n);
    let tree = CoregTree::new(points);
    let rows: Vec<([f64; 3], [[f64; 3]; 3], bool)> = points
        .par_iter()
        .map(|p| {
            let nn = tree.knn(p, k);
            let mut idx: Vec<usize> = nn.iter().map(|&(_, i)| i).collect();
            let mut valid = true;
            if let Some(r) = radius {
                // tlsalign collapses onto idx[:, :1], the nearest neighbour
                // (normally the point itself).
                let first = idx.first().copied().unwrap_or(0);
                let mut within = 0usize;
                for (j, &(d, _)) in nn.iter().enumerate() {
                    if d > r {
                        idx[j] = first;
                    } else {
                        within += 1;
                    }
                }
                valid = within >= 3;
            }
            let kk = idx.len();
            let mut mean = [0.0f64; 3];
            for &i in &idx {
                for a in 0..3 {
                    mean[a] += points[i][a];
                }
            }
            for m in mean.iter_mut() {
                *m /= kk.max(1) as f64;
            }
            let mut c = [[0.0f64; 3]; 3];
            for &i in &idx {
                let d = [points[i][0] - mean[0], points[i][1] - mean[1], points[i][2] - mean[2]];
                for a in 0..3 {
                    for b in 0..3 {
                        c[a][b] += d[a] * d[b];
                    }
                }
            }
            let denom = (k.max(2) - 1) as f64;
            let m = Matrix3::from_fn(|a, b| c[a][b] / denom);
            let (ev, vecs) = eigh3(&m);
            (ev, vecs, valid)
        })
        .collect();
    let mut out = LocalPca { evals: Vec::with_capacity(n), evecs: Vec::with_capacity(n), valid: Vec::with_capacity(n) };
    for (e, v, ok) in rows {
        out.evals.push(e);
        out.evecs.push(v);
        out.valid.push(ok);
    }
    out
}

/// Symmetric 3x3 eigendecomposition with ascending eigenvalues.
fn eigh3(m: &Matrix3<f64>) -> ([f64; 3], [[f64; 3]; 3]) {
    if !m.iter().all(|v| v.is_finite()) {
        return ([f64::NAN; 3], [[f64::NAN; 3]; 3]);
    }
    let e = SymmetricEigen::new(*m);
    let mut ord = [0usize, 1, 2];
    ord.sort_by(|&a, &b| e.eigenvalues[a].total_cmp(&e.eigenvalues[b]));
    let evals = [e.eigenvalues[ord[0]], e.eigenvalues[ord[1]], e.eigenvalues[ord[2]]];
    let mut evecs = [[0.0; 3]; 3];
    for (j, &o) in ord.iter().enumerate() {
        let c = e.eigenvectors.column(o);
        evecs[j] = [c[0], c[1], c[2]];
    }
    (evals, evecs)
}

/// Normals and planarity from a [`LocalPca`], as `estimate_normals` derives
/// them: normal = smallest-eigenvalue eigenvector, planarity
/// `(l1 - l0) / l2` (0 when `l2 <= 0`), both zeroed where invalid, planarity
/// clipped to `[0, 1]`.
pub fn normals_from_pca(pca: &LocalPca) -> (Vec<Point>, Vec<f64>) {
    pca.evals
        .par_iter()
        .zip(pca.evecs.par_iter())
        .zip(pca.valid.par_iter())
        .map(|((e, v), &ok)| {
            if !ok {
                return ([0.0; 3], 0.0);
            }
            let pl = if e[2] > 0.0 { (e[1] - e[0]) / e[2] } else { 0.0 };
            (v[0], pl.clamp(0.0, 1.0))
        })
        .unzip()
}

/// `tlsalign.preprocess.estimate_normals(points, k, radius)` (no viewpoint
/// flip): `(normals, planarity)`.  Normal signs are arbitrary.
pub fn estimate_normals(points: &[Point], k: usize, radius: Option<f64>) -> (Vec<Point>, Vec<f64>) {
    if points.len() < 3 {
        return (vec![[0.0; 3]; points.len()], vec![0.0; points.len()]);
    }
    normals_from_pca(&local_pca(points, k, radius))
}

/// `tlsalign.preprocess.planar_filter`: voxel centroids (when `voxel` is set
/// and non-zero), returned unchanged if fewer than `k`, else the points with
/// planarity `>= min_planarity`.
pub fn planar_filter(points: &[Point], min_planarity: f64, voxel: Option<f64>, k: usize, radius: Option<f64>) -> Vec<Point> {
    let pts = match voxel {
        Some(v) if v != 0.0 => voxel_downsample(points, v),
        _ => points.to_vec(),
    };
    if pts.len() < k {
        return pts;
    }
    let (_, planarity) = estimate_normals(&pts, k, radius);
    pts.into_iter().zip(planarity).filter(|(_, pl)| *pl >= min_planarity).map(|(p, _)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn coreg_voxel_order_and_centroids() {
        let pts = vec![
            [0.05, 0.05, 0.05],
            [-0.95, 0.0, 0.0],
            [0.15, 0.05, 0.05],
            [-0.9, 0.02, 0.0],
            [0.0, -0.5, 0.0],
        ];
        let (c, n) = voxel_centroids(&pts, 1.0, true);
        // grid: (0,0,0),(-1,0,0),(0,0,0),(-1,0,0),(0,-1,0); lexicographic:
        // (-1,0,0) < (0,-1,0) < (0,0,0)
        assert_eq!(n, vec![2, 1, 2]);
        assert!((c[0][0] - (-0.95 + -0.9) / 2.0).abs() < 1e-15);
        assert_eq!(c[1], [0.0, -0.5, 0.0]);
        assert!((c[2][0] - 0.1).abs() < 1e-15);
        let (f, nf) = voxel_centroids(&pts, 1.0, false);
        assert_eq!(f, vec![pts[0], pts[1], pts[4]]);
        assert_eq!(nf, vec![2, 2, 1]);
    }

    #[test]
    fn coreg_pca_plane_and_line() {
        let mut s = 7u64;
        let plane: Vec<Point> = (0..500).map(|_| [lcg(&mut s), lcg(&mut s), 0.3]).collect();
        let (nrm, pl) = estimate_normals(&plane, 20, None);
        for (n, p) in nrm.iter().zip(&pl) {
            assert!((n[2].abs() - 1.0).abs() < 1e-9);
            assert!(*p > 0.0 && *p <= 1.0);
        }
        let line: Vec<Point> = (0..200).map(|i| [i as f64 * 0.01, 0.0, 0.0]).collect();
        let (_, pl) = estimate_normals(&line, 10, None);
        assert!(pl.iter().all(|&p| p.abs() < 1e-9));
        // radius collapse: isolated points are invalid
        let sparse: Vec<Point> = (0..10).map(|i| [i as f64 * 10.0, 0.0, 0.0]).collect();
        let pca = local_pca(&sparse, 5, Some(1.0));
        assert!(pca.valid.iter().all(|v| !v));
        let (nrm, pl) = estimate_normals(&sparse, 5, Some(1.0));
        assert!(nrm.iter().all(|n| *n == [0.0; 3]) && pl.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn coreg_tree_upper_bound_is_exclusive() {
        let t = CoregTree::new(&[[0.0, 0.0, 0.0]]);
        assert_eq!(t.nearest(&[0.5, 0.0, 0.0], 0.5), (f64::INFINITY, 1));
        assert_eq!(t.nearest(&[0.25, 0.0, 0.0], 0.5), (0.25, 0));
        assert_eq!(t.nearest(&[3.0, 0.0, 0.0], f64::INFINITY), (3.0, 0));
    }
}
