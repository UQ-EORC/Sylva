// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Spatial indexing: k-d tree and voxel hashing.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use kiddo::{ImmutableKdTree, SquaredEuclidean};

use crate::Point;

/// 3-D k-d tree over a point slice. Items are point indices.
pub struct KdTree {
    tree: ImmutableKdTree<f64, 3>,
}

impl KdTree {
    pub fn new(points: &[Point]) -> Self {
        // kiddo cannot build an empty tree; keep a sentinel that is never returned.
        let tree = if points.is_empty() {
            ImmutableKdTree::new_from_slice(&[[f64::NAN; 3]]).expect("kd-tree build")
        } else {
            ImmutableKdTree::new_from_slice(points).expect("kd-tree build")
        };
        KdTree { tree }
    }

    /// `k` nearest points to `p`, as `(index, distance)` sorted by distance.
    pub fn knn(&self, p: &Point, k: usize) -> Vec<(usize, f64)> {
        let Some(k) = NonZeroUsize::new(k) else { return Vec::new() };
        self.tree
            .query(p)
            .nearest_n::<SquaredEuclidean<f64>>(k)
            .execute()
            .into_iter()
            .filter(|r| r.distance.is_finite())
            .map(|r| (r.item as usize, r.distance.sqrt()))
            .collect()
    }

    /// [`knn`](Self::knn) with ties broken by index: of the points at the
    /// `k`-th distance, those with the lowest indices are kept, and the
    /// neighbours come sorted by distance, then index. The search tree's own
    /// order among equal distances depends on its layout, so two clouds that
    /// hold the same neighbourhood in the same relative order (a tile and the
    /// whole plot, say) get the same neighbours only this way.
    pub fn knn_by_index(&self, p: &Point, k: usize) -> Vec<(usize, f64)> {
        let mut nb = self.knn(p, k);
        if let Some(&(_, dk)) = nb.last().filter(|_| nb.len() == k) {
            // Every point at the k-th distance, however many there are.
            nb = self.within(p, dk * (1.0 + 1e-9) + f64::MIN_POSITIVE).into_iter().filter(|&(_, d)| d <= dk).collect();
        }
        nb.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        nb.truncate(k);
        nb
    }

    /// Nearest point to `p`.
    pub fn nearest(&self, p: &Point) -> Option<(usize, f64)> {
        let r = self.tree.query(p).nearest_one::<SquaredEuclidean<f64>>().execute();
        r.distance.is_finite().then(|| (r.item as usize, r.distance.sqrt()))
    }

    /// All points within `radius` of `p` (unsorted).
    pub fn within(&self, p: &Point, radius: f64) -> Vec<(usize, f64)> {
        self.tree
            .query(p)
            .within::<SquaredEuclidean<f64>>(radius * radius)
            .unsorted()
            .execute()
            .into_iter()
            .filter(|r| r.distance.is_finite())
            .map(|r| (r.item as usize, r.distance.sqrt()))
            .collect()
    }

    /// Number of points within `radius` of `p`.
    pub fn count_within(&self, p: &Point, radius: f64) -> usize {
        self.within(p, radius).len()
    }
}

/// Integer voxel key.
pub type VoxelKey = [i64; 3];

#[inline]
pub fn voxel_key(p: &Point, origin: &Point, size: f64) -> VoxelKey {
    [
        ((p[0] - origin[0]) / size).floor() as i64,
        ((p[1] - origin[1]) / size).floor() as i64,
        ((p[2] - origin[2]) / size).floor() as i64,
    ]
}

/// Group point indices by voxel. Returns `(keys in first-seen order, index lists)`.
pub fn voxel_groups(points: &[Point], origin: &Point, size: f64) -> (Vec<VoxelKey>, Vec<Vec<usize>>) {
    let mut map: HashMap<VoxelKey, usize> = HashMap::new();
    let mut keys = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let k = voxel_key(p, origin, size);
        let g = *map.entry(k).or_insert_with(|| {
            keys.push(k);
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    (keys, groups)
}

/// Minimum corner of a point set.
pub fn min_corner(points: &[Point]) -> Point {
    let mut lo = [f64::INFINITY; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
        }
    }
    if points.is_empty() {
        [0.0; 3]
    } else {
        lo
    }
}

/// Maximum corner of a point set.
pub fn max_corner(points: &[Point]) -> Point {
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for k in 0..3 {
            hi[k] = hi[k].max(p[k]);
        }
    }
    if points.is_empty() {
        [0.0; 3]
    } else {
        hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_by_index_breaks_ties_by_index_in_any_subset() {
        // A lattice: every point has many neighbours at exactly equal distances.
        let pts: Vec<Point> = (0..1000).map(|i| [(i % 10) as f64, ((i / 10) % 10) as f64, (i / 100) as f64]).collect();
        let whole = KdTree::new(&pts);
        // A subset in the same relative order, as a tile holds a plot's points.
        let keep: Vec<usize> = (0..pts.len()).filter(|&i| pts[i][0] < 7.0).collect();
        let sub_pts: Vec<Point> = keep.iter().map(|&i| pts[i]).collect();
        let sub = KdTree::new(&sub_pts);
        for (j, &i) in keep.iter().enumerate() {
            if pts[i][0] > 4.0 {
                continue;
            }
            let a = whole.knn_by_index(&pts[i], 7);
            let b: Vec<(usize, f64)> = sub.knn_by_index(&sub_pts[j], 7).into_iter().map(|(m, d)| (keep[m], d)).collect();
            assert_eq!(a, b, "point {i}");
            assert!(a.windows(2).all(|w| w[0].1 < w[1].1 || (w[0].1 == w[1].1 && w[0].0 < w[1].0)));
        }
        assert!(whole.knn_by_index(&pts[0], 0).is_empty());
        assert_eq!(KdTree::new(&pts[..3]).knn_by_index(&pts[0], 7).len(), 3);
    }
}
