//! How well each tree was seen: its crown envelope in a ray-traced voxel grid.
//!
//! The envelope of a tree is, layer by layer, the convex hull of its points
//! in that voxel layer (grown by half a voxel), so gaps inside the crown that
//! no pulse reached count against it just as much as the foliage that was
//! hit. Every envelope voxel is then read from the grid: observed (a pulse
//! went through or ended in it), occluded (only reached by pulses already
//! stopped before it) or unobserved (nothing came near).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::trees::{convex_hull, in_convex_polygon};
use crate::Point;

/// Voxel states, as in the grid's `state` field.
pub const UNOBSERVED: u8 = 0;
pub const OCCLUDED: u8 = 1;
pub const EMPTY: u8 = 2;
pub const FILLED: u8 = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct TreeSampling {
    pub tree_id: i64,
    /// Voxels in the crown envelope and their volume (m3).
    pub n_voxels: usize,
    pub volume: f64,
    /// Shares of the envelope voxels that were observed (empty or filled),
    /// occluded, and never reached.
    pub observed_fraction: f64,
    pub occluded_fraction: f64,
    pub unobserved_fraction: f64,
    /// Median and 10th percentile of pulses entering an envelope voxel
    /// (unobserved and occluded voxels count 0).
    pub median_beams: f64,
    pub p10_beams: f64,
    /// Share of observed envelope voxels crossed by at least `min_beams` pulses.
    pub well_sampled_fraction: f64,
    /// Median pulses entering an envelope voxel per quarter of the
    /// envelope's height, bottom first.
    pub beams_by_quarter: [f64; 4],
    /// Share of the voxels up to `above` metres over the highest point,
    /// within the footprint of the top metre of the crown, that were observed.
    /// Low: the top may be hidden.
    pub above_observed_fraction: f64,
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
}

/// Sampling of every labelled tree (labels `< 0` are ignored). `shape` is
/// `[nx, ny, nz]`; `state` and `beams` are flat in `(k * ny + j) * nx + i`
/// order.
#[allow(clippy::too_many_arguments)]
pub fn tree_sampling(points: &[Point], labels: &[i64], origin: Point, voxel_size: f64, shape: [usize; 3], state: &[u8], beams: &[f64], min_beams: f64, above: f64) -> Vec<TreeSampling> {
    let [nx, ny, nz] = shape;
    let mut by_tree: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l >= 0 {
            by_tree.entry(l).or_default().push(i);
        }
    }
    let mut ids: Vec<i64> = by_tree.keys().copied().collect();
    ids.sort_unstable();
    ids.par_iter()
        .map(|&id| {
            let members = &by_tree[&id];
            // Points of the tree by voxel layer.
            let mut layers: HashMap<i64, Vec<[f64; 2]>> = HashMap::new();
            for &i in members {
                let p = points[i];
                let k = ((p[2] - origin[2]) / voxel_size).floor() as i64;
                layers.entry(k).or_default().push([p[0], p[1]]);
            }
            let (kmin, kmax) = (layers.keys().copied().min().unwrap_or(0), layers.keys().copied().max().unwrap_or(0));
            let mut n = 0usize;
            let (mut obs, mut occ, mut unobs, mut well) = (0usize, 0usize, 0usize, 0usize);
            let mut beam_vals: Vec<f64> = Vec::new();
            let mut quarter: [Vec<f64>; 4] = Default::default();
            for (&k, xy) in &layers {
                if k < 0 || k >= nz as i64 {
                    continue;
                }
                // Grow the layer's points by half a voxel so thin layers still
                // cover the voxels they touch.
                let h = 0.5 * voxel_size;
                let grown: Vec<[f64; 2]> = xy.iter().flat_map(|q| [[q[0] - h, q[1] - h], [q[0] + h, q[1] - h], [q[0] - h, q[1] + h], [q[0] + h, q[1] + h]]).collect();
                let hull = convex_hull(&grown);
                if hull.len() < 3 {
                    continue;
                }
                let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
                for v in &hull {
                    x0 = x0.min(v[0]);
                    x1 = x1.max(v[0]);
                    y0 = y0.min(v[1]);
                    y1 = y1.max(v[1]);
                }
                let i0 = (((x0 - origin[0]) / voxel_size).floor().max(0.0) as usize).min(nx);
                let i1 = (((x1 - origin[0]) / voxel_size).ceil().max(0.0) as usize).min(nx);
                let j0 = (((y0 - origin[1]) / voxel_size).floor().max(0.0) as usize).min(ny);
                let j1 = (((y1 - origin[1]) / voxel_size).ceil().max(0.0) as usize).min(ny);
                let q = if kmax > kmin { (((k - kmin) as f64 / (kmax - kmin + 1) as f64) * 4.0) as usize } else { 0 }.min(3);
                for j in j0..j1 {
                    for i in i0..i1 {
                        let c = [origin[0] + (i as f64 + 0.5) * voxel_size, origin[1] + (j as f64 + 0.5) * voxel_size];
                        if !in_convex_polygon(&hull, c) {
                            continue;
                        }
                        let idx = (k as usize * ny + j) * nx + i;
                        n += 1;
                        match state[idx] {
                            EMPTY | FILLED => {
                                obs += 1;
                                quarter[q].push(beams[idx]);
                                if beams[idx] >= min_beams {
                                    well += 1;
                                }
                                beam_vals.push(beams[idx]);
                            }
                            OCCLUDED => {
                                occ += 1;
                                quarter[q].push(0.0);
                                beam_vals.push(0.0);
                            }
                            _ => {
                                unobs += 1;
                                quarter[q].push(0.0);
                                beam_vals.push(0.0);
                            }
                        }
                    }
                }
            }
            // The space above the top, over the footprint of the top metre.
            let top_z = members.iter().map(|&i| points[i][2]).fold(f64::NEG_INFINITY, f64::max);
            let crown_top: Vec<[f64; 2]> = members.iter().filter(|&&i| points[i][2] >= top_z - 1.0).map(|&i| [points[i][0], points[i][1]]).collect();
            let h = 0.5 * voxel_size;
            let grown: Vec<[f64; 2]> = crown_top.iter().flat_map(|q| [[q[0] - h, q[1] - h], [q[0] + h, q[1] - h], [q[0] - h, q[1] + h], [q[0] + h, q[1] + h]]).collect();
            let hull = convex_hull(&grown);
            let k_top = ((top_z - origin[2]) / voxel_size).floor() as i64;
            let k_end = ((top_z + above - origin[2]) / voxel_size).floor() as i64;
            let (mut above_n, mut above_obs) = (0usize, 0usize);
            if hull.len() >= 3 {
                for k in (k_top + 1)..=k_end {
                    if k < 0 || k >= nz as i64 {
                        continue;
                    }
                    for j in 0..ny {
                        for i in 0..nx {
                            let c = [origin[0] + (i as f64 + 0.5) * voxel_size, origin[1] + (j as f64 + 0.5) * voxel_size];
                            if in_convex_polygon(&hull, c) {
                                above_n += 1;
                                if state[(k as usize * ny + j) * nx + i] >= EMPTY {
                                    above_obs += 1;
                                }
                            }
                        }
                    }
                }
            }
            let f = |a: usize| if n > 0 { a as f64 / n as f64 } else { f64::NAN };
            let mut b2 = beam_vals.clone();
            TreeSampling {
                tree_id: id,
                n_voxels: n,
                volume: n as f64 * voxel_size.powi(3),
                observed_fraction: f(obs),
                occluded_fraction: f(occ),
                unobserved_fraction: f(unobs),
                median_beams: percentile(&mut beam_vals, 0.5),
                p10_beams: percentile(&mut b2, 0.1),
                well_sampled_fraction: if obs > 0 { well as f64 / obs as f64 } else { f64::NAN },
                beams_by_quarter: quarter.map(|mut v| percentile(&mut v, 0.5)),
                above_observed_fraction: if above_n > 0 { above_obs as f64 / above_n as f64 } else { f64::NAN },
            }
        })
        .collect()
}
