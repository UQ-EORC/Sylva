// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Masks from the distance to another point cloud.

use std::borrow::Cow;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;

use super::CHUNK;
use crate::error::{Error, Result};
use crate::Point;

/// Whether each point lies within `distance` of the nearest finite point of
/// `other` (distance `<= distance`), in 3-D or, with `horizontal`, in x, y
/// only.
///
/// A k-d tree is built on `other` and queried in parallel; each answer is
/// the exact nearest distance, so the mask does not depend on the thread
/// count. Points of `other` with a non-finite coordinate are ignored; points
/// of `points` with one are never near anything. With an empty `other`
/// nothing is near.
///
/// # Errors
/// `distance` negative or not finite.
pub fn near_mask(points: &[Point], other: &[Point], distance: f64, horizontal: bool) -> Result<Vec<bool>> {
    if !(distance.is_finite() && distance >= 0.0) {
        return Err(Error::invalid(format!("distance must be finite and non-negative, got {distance}")));
    }
    let d2 = distance * distance;
    if horizontal {
        let xy: Vec<[f64; 2]> = other.iter().filter(|p| p[0].is_finite() && p[1].is_finite()).map(|p| [p[0], p[1]]).collect();
        Ok(query(points, &xy, d2, |p| [p[0], p[1]]))
    } else {
        let finite = if other.iter().all(|p| p.iter().all(|v| v.is_finite())) { Cow::Borrowed(other) } else { Cow::Owned(other.iter().filter(|p| p.iter().all(|v| v.is_finite())).copied().collect()) };
        Ok(query(points, &finite, d2, |p| *p))
    }
}

fn query<const K: usize>(points: &[Point], other: &[[f64; K]], d2: f64, key: impl Fn(&Point) -> [f64; K] + Sync) -> Vec<bool> {
    let mut out = vec![false; points.len()];
    if other.is_empty() {
        return out;
    }
    let tree: ImmutableKdTree<f64, K> = ImmutableKdTree::new_from_slice_parallel(other).expect("kd-tree build");
    out.par_chunks_mut(CHUNK).zip(points.par_chunks(CHUNK)).for_each(|(o, p)| {
        for (o, p) in o.iter_mut().zip(p) {
            let q = key(p);
            if q.iter().all(|v| v.is_finite()) {
                *o = tree.query(&q).nearest_one::<SquaredEuclidean<f64>>().execute().distance <= d2;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(points: &[Point], other: &[Point], d: f64, horizontal: bool) -> Vec<bool> {
        points.iter().map(|p| other.iter().any(|q| {
            let dz = if horizontal { 0.0 } else { p[2] - q[2] };
            (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + dz * dz <= d * d
        })).collect()
    }

    fn cloud(n: usize, seed: u64) -> Vec<Point> {
        let mut s = seed.max(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 1_000_000) as f64 / 100_000.0
        };
        (0..n).map(|_| [next(), next(), next()]).collect()
    }

    #[test]
    fn matches_brute_force() {
        let a = cloud(3000, 7);
        let b = cloud(500, 11);
        for h in [false, true] {
            assert_eq!(near_mask(&a, &b, 0.4, h).unwrap(), brute(&a, &b, 0.4, h));
        }
    }

    #[test]
    fn boundary_empty_and_nonfinite() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0], [0.0, 0.0, 5.0]];
        let b = [[0.0, 0.0, 1.0], [f64::INFINITY, 0.0, 0.0]];
        assert_eq!(near_mask(&a, &b, 1.0, false).unwrap(), vec![true, false, false, false]);
        assert_eq!(near_mask(&a, &b, 1.0, true).unwrap(), vec![true, true, false, true]);
        assert_eq!(near_mask(&a, &[], 1.0, false).unwrap(), vec![false; 4]);
        assert!(near_mask(&a, &b, -1.0, false).is_err());
        assert!(near_mask(&a, &b, f64::NAN, false).is_err());
    }
}
