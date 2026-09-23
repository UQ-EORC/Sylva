// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Subsampling, cropping and noise filtering.

use rayon::prelude::*;

use crate::spatial::{min_corner, voxel_groups, KdTree};
use crate::{Point, PointCloud};

/// Simple xorshift RNG so results are reproducible without extra deps.
pub(crate) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

}

/// Keep one point per voxel.
///
/// `centroid = false` keeps the first original point of each voxel (attributes
/// preserved); `true` returns voxel centroids with attributes dropped.
pub fn voxel_downsample(cloud: &PointCloud, voxel_size: f64, centroid: bool) -> PointCloud {
    if cloud.is_empty() {
        return cloud.clone();
    }
    let origin = min_corner(&cloud.xyz);
    let (_, groups) = voxel_groups(&cloud.xyz, &origin, voxel_size);
    if centroid {
        let xyz: Vec<Point> = groups
            .iter()
            .map(|g| {
                let n = g.len() as f64;
                let mut c = [0.0; 3];
                for &i in g {
                    for k in 0..3 {
                        c[k] += cloud.xyz[i][k];
                    }
                }
                [c[0] / n, c[1] / n, c[2] / n]
            })
            .collect();
        PointCloud::new(xyz)
    } else {
        let mut idx: Vec<usize> = groups.iter().map(|g| g[0]).collect();
        idx.sort_unstable();
        cloud.take(&idx)
    }
}

/// Indices kept by [`voxel_downsample`] with `centroid = false`.
pub fn voxel_downsample_indices(points: &[Point], voxel_size: f64) -> Vec<usize> {
    let origin = min_corner(points);
    let (_, groups) = voxel_groups(points, &origin, voxel_size);
    let mut idx: Vec<usize> = groups.iter().map(|g| g[0]).collect();
    idx.sort_unstable();
    idx
}

/// Random subsample of `n` points (without replacement), in original order.
pub fn random_subsample(cloud: &PointCloud, n: usize, seed: u64) -> PointCloud {
    cloud.take(&random_indices(cloud.len(), n, seed))
}

/// Indices of an approximate Poisson-disk subsample: no two kept points closer than `distance`.
pub fn min_distance_indices(points: &[Point], distance: f64) -> Vec<usize> {
    let coarse_idx = voxel_downsample_indices(points, distance / 3f64.sqrt());
    let coarse: Vec<Point> = coarse_idx.iter().map(|&i| points[i]).collect();
    let tree = KdTree::new(&coarse);
    let mut keep = vec![true; coarse.len()];
    for i in 0..coarse.len() {
        if !keep[i] {
            continue;
        }
        for (j, _) in tree.within(&coarse[i], distance) {
            if j > i {
                keep[j] = false;
            }
        }
    }
    coarse_idx.into_iter().zip(keep).filter(|(_, k)| *k).map(|(i, _)| i).collect()
}

/// Approximate Poisson-disk subsampling: no two kept points closer than `distance`.
pub fn min_distance_subsample(cloud: &PointCloud, distance: f64) -> PointCloud {
    cloud.take(&min_distance_indices(&cloud.xyz, distance))
}

/// Indices of a random subsample of `n` points (sorted).
pub fn random_indices(total: usize, n: usize, seed: u64) -> Vec<usize> {
    let n = n.min(total);
    let mut idx: Vec<usize> = (0..total).collect();
    let mut rng = Rng::new(seed);
    for i in 0..n {
        let j = i + rng.below(total - i);
        idx.swap(i, j);
    }
    idx.truncate(n);
    idx.sort_unstable();
    idx
}

/// Keep points inside an axis-aligned box; use ±infinity for open bounds.
pub fn crop_box(cloud: &PointCloud, min: Point, max: Point) -> PointCloud {
    let mask: Vec<bool> =
        cloud.xyz.iter().map(|p| (0..3).all(|k| p[k] >= min[k] && p[k] <= max[k])).collect();
    cloud.filter(&mask)
}

/// Keep points inside a vertical cylinder (a circular plot).
pub fn crop_cylinder(cloud: &PointCloud, cx: f64, cy: f64, radius: f64, zmin: f64, zmax: f64) -> PointCloud {
    let r2 = radius * radius;
    let mask: Vec<bool> = cloud
        .xyz
        .iter()
        .map(|p| {
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            dx * dx + dy * dy <= r2 && p[2] >= zmin && p[2] <= zmax
        })
        .collect();
    cloud.filter(&mask)
}

/// Keep points within `[min_range, max_range]` of `origin`.
pub fn range_filter(cloud: &PointCloud, origin: Point, min_range: f64, max_range: f64) -> PointCloud {
    let mask: Vec<bool> = cloud
        .xyz
        .iter()
        .map(|p| {
            let r = crate::transform::norm(&crate::transform::sub(p, &origin));
            r >= min_range && r <= max_range
        })
        .collect();
    cloud.filter(&mask)
}

/// Statistical outlier removal mask: `true` = keep.
///
/// A point is dropped when its mean distance to `k` neighbours exceeds
/// `mean + std_ratio * std` of that statistic over the cloud.
pub fn statistical_outlier_mask(points: &[Point], k: usize, std_ratio: f64) -> Vec<bool> {
    if points.len() <= k {
        return vec![true; points.len()];
    }
    let tree = KdTree::new(points);
    let mean_d: Vec<f64> = points
        .par_iter()
        .map(|p| {
            let nb = tree.knn(p, k + 1);
            let s: f64 = nb.iter().skip(1).map(|(_, d)| d).sum();
            s / (nb.len().saturating_sub(1).max(1)) as f64
        })
        .collect();
    let n = mean_d.len() as f64;
    let mu = mean_d.iter().sum::<f64>() / n;
    let var = mean_d.iter().map(|d| (d - mu) * (d - mu)).sum::<f64>() / n;
    let thr = mu + std_ratio * var.sqrt();
    mean_d.iter().map(|&d| d <= thr).collect()
}

pub fn statistical_outlier_removal(cloud: &PointCloud, k: usize, std_ratio: f64) -> PointCloud {
    cloud.filter(&statistical_outlier_mask(&cloud.xyz, k, std_ratio))
}

/// Radius outlier removal mask: keep points with at least `min_neighbors` others within `radius`.
pub fn radius_outlier_mask(points: &[Point], radius: f64, min_neighbors: usize) -> Vec<bool> {
    let tree = KdTree::new(points);
    points.par_iter().map(|p| tree.count_within(p, radius).saturating_sub(1) >= min_neighbors).collect()
}

pub fn radius_outlier_removal(cloud: &PointCloud, radius: f64, min_neighbors: usize) -> PointCloud {
    cloud.filter(&radius_outlier_mask(&cloud.xyz, radius, min_neighbors))
}

/// Per-point local PCA descriptors from `k` neighbours.
///
/// Returns `(normals, eigenvalues ascending)`; normals are unoriented.
pub fn local_pca(points: &[Point], k: usize) -> (Vec<Point>, Vec<[f64; 3]>) {
    let tree = KdTree::new(points);
    let k = k.max(3);
    points
        .par_iter()
        .map(|p| {
            let nb = tree.knn(p, k);
            let n = nb.len() as f64;
            let mut c = [0.0; 3];
            for (i, _) in &nb {
                for a in 0..3 {
                    c[a] += points[*i][a];
                }
            }
            for a in 0..3 {
                c[a] /= n;
            }
            let mut cov = nalgebra::Matrix3::<f64>::zeros();
            for (i, _) in &nb {
                let d = crate::transform::sub(&points[*i], &c);
                for a in 0..3 {
                    for b in 0..3 {
                        cov[(a, b)] += d[a] * d[b];
                    }
                }
            }
            let eig = cov.symmetric_eigen();
            // Sort eigenpairs ascending.
            let mut order = [0usize, 1, 2];
            order.sort_by(|&i, &j| eig.eigenvalues[i].partial_cmp(&eig.eigenvalues[j]).unwrap());
            let v = eig.eigenvectors.column(order[0]);
            let normal = [v[0], v[1], v[2]];
            let vals = [eig.eigenvalues[order[0]], eig.eigenvalues[order[1]], eig.eigenvalues[order[2]]];
            (normal, vals)
        })
        .unzip()
}

/// Unoriented per-point normals from local PCA.
pub fn estimate_normals(points: &[Point], k: usize) -> Vec<Point> {
    local_pca(points, k).0
}

/// Planarity `(l2 - l1) / l3` and linearity `(l3 - l2) / l3` per point (eigenvalues ascending).
pub fn planarity_linearity(points: &[Point], k: usize) -> (Vec<f64>, Vec<f64>) {
    let (_, vals) = local_pca(points, k);
    vals.iter()
        .map(|[l1, l2, l3]| {
            if *l3 > 0.0 {
                ((l2 - l1) / l3, (l3 - l2) / l3)
            } else {
                (0.0, 0.0)
            }
        })
        .unzip()
}

