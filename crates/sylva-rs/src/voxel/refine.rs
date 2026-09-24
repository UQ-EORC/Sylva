// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Portions adapted from rayvoxel (raycloudtools fork, Josh Rivory), Copyright (c)
// 2020 CSIRO, under the CSIRO licence in THIRD_PARTY_NOTICES.md.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Flat-top canopy peaks and neighbour priors.

use rayon::prelude::*;

use super::traverse::Geom;
use super::{RayVoxels, F, I};
use crate::Point;

/// Raise `peaks` (highest echo per column, in voxel units above the grid floor).
pub(crate) fn update_peaks(peaks: &mut [f64], echoes: &[Point], geom: &Geom) {
    for p in echoes {
        let v = geom.to_vox(p);
        let (i, j) = (v[0].floor(), v[1].floor());
        if i >= 0.0 && j >= 0.0 && (i as usize) < geom.shape[0] && (j as usize) < geom.shape[1] {
            let c = i as usize + geom.shape[0] * j as usize;
            peaks[c] = peaks[c].max(v[2]);
        }
    }
}

/// Offsets of the 6 face, 12 edge and 8 corner neighbours.
fn shells() -> [Vec<[i64; 3]>; 3] {
    let mut s = [Vec::new(), Vec::new(), Vec::new()];
    for dz in -1i64..=1 {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let order = (dx.abs() + dy.abs() + dz.abs()) as usize;
                if order > 0 {
                    s[order - 1].push([dx, dy, dz]);
                }
            }
        }
    }
    s
}

/// Top up voxels crossed by fewer than `min_rays` weighted beams with a share
/// of their neighbours' statistics: first the face neighbours, then edges,
/// then corners, each scaled so that the voxel just reaches `min_rays`.
///
/// Unlike rayvoxel, neighbours are always read from the grid as it was before
/// any top-up, so the result does not depend on iteration order, and the grid
/// is not padded: the outermost voxel layer is left as measured.
pub(crate) fn apply_neighbour_priors(vox: &mut RayVoxels, min_rays: f32) {
    let [nx, ny, nz] = vox.shape;
    if nx < 3 || ny < 3 || nz < 3 {
        return;
    }
    let shells = shells();
    let stride = [1i64, nx as i64, (nx * ny) as i64];
    let offsets: Vec<Vec<i64>> = shells.iter().map(|s| s.iter().map(|d| d[0] * stride[0] + d[1] * stride[1] + d[2] * stride[2]).collect()).collect();
    let nbw = &vox.f[F::NumBeamsWeighted as usize];
    let hits = &vox.i[I::NumHits as usize];

    // (voxel, share taken from each shell)
    let plan: Vec<(usize, [f64; 3])> = (1..nz - 1)
        .into_par_iter()
        .flat_map_iter(|k| {
            let offsets = &offsets;
            (1..ny - 1).flat_map(move |j| (1..nx - 1).map(move |i| i + nx * (j + ny * k))).filter_map(move |idx| {
                if hits[idx] == 0 && nbw[idx] == 0.0 {
                    return None;
                }
                let mut needed = (min_rays - nbw[idx]) as f64;
                if needed <= 0.0 {
                    return None;
                }
                let mut share = [0.0; 3];
                for (s, offs) in offsets.iter().enumerate() {
                    let sum: f64 = offs.iter().map(|&o| nbw[(idx as i64 + o) as usize] as f64).sum();
                    if sum > 0.0 {
                        share[s] = (needed / sum).min(1.0);
                        needed -= sum * share[s];
                    }
                    if needed <= 0.0 {
                        break;
                    }
                }
                Some((idx, share))
            })
        })
        .collect();
    if plan.is_empty() {
        return;
    }

    for field in vox.f.iter_mut().filter(|f| !f.is_empty()) {
        let old = field.clone();
        for (idx, share) in &plan {
            for (s, offs) in offsets.iter().enumerate() {
                if share[s] > 0.0 {
                    let sum: f64 = offs.iter().map(|&o| old[(*idx as i64 + o) as usize] as f64).sum();
                    field[*idx] += (sum * share[s]) as f32;
                }
            }
        }
    }
    // Counts are truncated per shell, so sparse neighbours may add nothing.
    for field in vox.i.iter_mut() {
        let old = field.clone();
        for (idx, share) in &plan {
            for (s, offs) in offsets.iter().enumerate() {
                if share[s] > 0.0 {
                    let sum: i64 = offs.iter().map(|&o| old[(*idx as i64 + o) as usize] as i64).sum();
                    field[*idx] += (sum as f64 * share[s]) as i32;
                }
            }
        }
    }
}
