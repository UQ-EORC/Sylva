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
