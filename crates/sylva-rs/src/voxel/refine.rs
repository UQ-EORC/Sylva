// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Adapted from rayvoxel (Josh Rivory, unpublished), a port of AMAPVox (UMR AMAP);
// see THIRD_PARTY_NOTICES.md.
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::voxel::VoxelParams;

    /// A grid with every `f` field but the beam-section ones, all zero.
    fn grid(shape: [usize; 3]) -> RayVoxels {
        let n = shape[0] * shape[1] * shape[2];
        RayVoxels {
            origin: [0.0; 3],
            voxel_size: 1.0,
            shape,
            params: VoxelParams::default(),
            has_leaf: false,
            has_wood: false,
            f: F::ALL.iter().map(|f| if *f == F::BsEntering { Vec::new() } else { vec![0.0; n] }).collect(),
            i: vec![vec![0; n]; I::COUNT],
            ppl_lambda: None,
            subvoxel_counts: None,
            ground_height: None,
            wood_volume: None,
            predominant_tree: None,
            tree_iad: BTreeMap::new(),
        }
    }

    fn set_f(v: &mut RayVoxels, field: F, values: impl Fn(usize) -> f32) {
        for (idx, x) in v.f[field as usize].iter_mut().enumerate() {
            *x = values(idx);
        }
    }

    #[test]
    fn shells_are_the_26_neighbours_by_order() {
        let s = shells();
        assert_eq!([s[0].len(), s[1].len(), s[2].len()], [6, 12, 8]);
        for (order, shell) in s.iter().enumerate() {
            assert!(shell.iter().all(|d| d.iter().map(|c| c.abs()).sum::<i64>() == order as i64 + 1));
        }
    }

    #[test]
    fn faces_then_edges_top_up_to_min_rays() {
        // 3 x 3 x 3: only the centre (13) is inside the untouched border.
        let mut v = grid([3, 3, 3]);
        set_f(&mut v, F::NumBeamsWeighted, |_| 1.0);
        set_f(&mut v, F::PathLength, |_| 2.0);
        v.i[I::NumHits as usize] = vec![1; 27];
        // Sparse counts: two face and two edge neighbours of the centre.
        for idx in [12, 14, 9, 11] {
            v.i[I::NumMissRays as usize][idx] = 1;
        }
        let before = v.clone();
        apply_neighbour_priors(&mut v, 10.0);
        // 1 beam, needs 9 more: all 6 faces (share 1), then 3 of the 12 edges' beams (share 1/4).
        assert_eq!(v.get_f(F::NumBeamsWeighted, 13), 1.0 + 6.0 + 12.0 * 0.25);
        assert_eq!(v.get_f(F::PathLength, 13), 2.0 + 6.0 * 2.0 + 12.0 * 2.0 * 0.25);
        assert_eq!(v.get_i(I::NumHits, 13), 1 + 6 + 3);
        // Counts are truncated per shell: two edge counts at 1/4 add nothing.
        assert_eq!(v.get_i(I::NumMissRays, 13), 2);
        assert!(v.f[F::BsEntering as usize].is_empty(), "switched-off groups stay empty");
        for idx in (0..27).filter(|&i| i != 13) {
            for f in F::ALL {
                assert_eq!(v.get_f(f, idx), before.get_f(f, idx), "border voxel {idx} changed");
            }
        }
    }

    #[test]
    fn faces_alone_when_they_suffice_and_only_observed_sparse_voxels() {
        let mut v = grid([3, 3, 3]);
        set_f(&mut v, F::NumBeamsWeighted, |_| 1.0);
        v.i[I::NumHits as usize][13] = 1;
        apply_neighbour_priors(&mut v, 4.0);
        assert_eq!(v.get_f(F::NumBeamsWeighted, 13), 4.0); // 1 + 6 * (3 / 6)

        // Crossed by enough beams: left as measured.
        let mut v = grid([3, 3, 3]);
        set_f(&mut v, F::NumBeamsWeighted, |i| if i == 13 { 5.0 } else { 1.0 });
        apply_neighbour_priors(&mut v, 5.0);
        assert_eq!(v.get_f(F::NumBeamsWeighted, 13), 5.0);

        // Never observed (no beam, no hit): stays unobserved.
        let mut v = grid([3, 3, 3]);
        set_f(&mut v, F::NumBeamsWeighted, |i| if i == 13 { 0.0 } else { 1.0 });
        apply_neighbour_priors(&mut v, 5.0);
        assert_eq!(v.get_f(F::NumBeamsWeighted, 13), 0.0);

        // Too thin for an interior: nothing to do.
        let mut v = grid([2, 5, 5]);
        set_f(&mut v, F::NumBeamsWeighted, |_| 1.0);
        let before = v.f.clone();
        apply_neighbour_priors(&mut v, 5.0);
        assert_eq!(v.f, before);
    }

    #[test]
    fn neighbours_are_read_before_any_top_up() {
        // 4 x 3 x 3: two adjacent interior voxels, 17 and 18, both sparse.
        let mut v = grid([4, 3, 3]);
        set_f(&mut v, F::NumBeamsWeighted, |i| if i == 17 || i == 18 { 1.0 } else { 2.0 });
        set_f(&mut v, F::PathLength, |i| i as f32);
        v.i[I::NumHits as usize][17] = 1;
        v.i[I::NumHits as usize][18] = 1;
        apply_neighbour_priors(&mut v, 5.0);
        // Face beams: five neighbours at 2 and the other sparse voxel at 1 = 11; share 4 / 11.
        let share = 4.0 / 11.0;
        assert!((v.get_f(F::NumBeamsWeighted, 17) - 5.0).abs() < 1e-6);
        assert!((v.get_f(F::NumBeamsWeighted, 18) - 5.0).abs() < 1e-6);
        // Face neighbours of 17 (strides 1, 4, 12) and 18, with their values before the top-up.
        let faces = |i: f64| (i - 1.0) + (i + 1.0) + (i - 4.0) + (i + 4.0) + (i - 12.0) + (i + 12.0);
        assert!((v.get_f(F::PathLength, 17) as f64 - (17.0 + share * faces(17.0))).abs() < 1e-4);
        assert!((v.get_f(F::PathLength, 18) as f64 - (18.0 + share * faces(18.0))).abs() < 1e-4);
    }

    #[test]
    fn peaks_keep_the_highest_echo_per_column() {
        let geom = Geom::new([0.0, 0.0, 0.0], 0.5, [2, 2, 4]);
        let mut peaks = vec![f64::MIN; 4];
        let echoes = [[0.2, 0.2, 1.0], [0.3, 0.1, 1.6], [0.7, 0.2, 0.4], [0.9, 0.9, 1.9], [5.0, 0.2, 9.0], [-0.1, 0.2, 9.0]];
        update_peaks(&mut peaks, &echoes, &geom);
        // Voxel units above the floor; echoes outside the columns are ignored.
        assert_eq!(peaks, vec![3.2, 0.8, f64::MIN, 3.8]);
        update_peaks(&mut peaks, &[[0.2, 0.2, 0.5]], &geom);
        assert_eq!(peaks[0], 3.2, "a lower echo does not lower the peak");
    }
}
