// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Adapted from rayvoxel (Josh Rivory, unpublished), a port of AMAPVox (UMR AMAP);
// see THIRD_PARTY_NOTICES.md.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Woody volume per voxel from QSM cylinders.

use std::f64::consts::PI;

use crate::qsm::Cylinder;
use crate::transform::{cross, normalize};
use crate::Point;

/// Axial samples per voxel edge; radial and angular sampling follow the same spacing.
const SAMPLES_PER_VOXEL: f64 = 3.0;
const MAX_RINGS: usize = 8;
const MAX_ANGULAR: usize = 16;

/// Split every cylinder into equal-volume samples (cell-centred along the
/// axis, equal-area rings, even angles) and add each sample's share of the
/// cylinder volume to the voxel it falls in.
pub(crate) fn rasterise(cylinders: &[Cylinder], origin: &Point, size: f64, shape: &[usize; 3], out: &mut [f32]) {
    let step = size / SAMPLES_PER_VOXEL;
    for c in cylinders {
        let volume = c.volume();
        if !(volume > 0.0) {
            continue;
        }
        let axis = normalize(&c.axis);
        let helper = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = normalize(&cross(&axis, &helper));
        let v = normalize(&cross(&axis, &u));
        let n_axial = ((c.length / step).ceil() as usize).max(1);
        let n_rings = ((c.radius / step).ceil() as usize).clamp(1, MAX_RINGS);
        let n_ang = if n_rings == 1 && c.radius < step { 1 } else { ((2.0 * PI * c.radius / step).ceil() as usize).clamp(1, MAX_ANGULAR) };
        let share = (volume / (n_axial * n_rings * n_ang) as f64) as f32;
        for ia in 0..n_axial {
            let t = (ia as f64 + 0.5) / n_axial as f64 * c.length;
            for ir in 0..n_rings {
                let rad = c.radius * ((ir as f64 + 0.5) / n_rings as f64).sqrt();
                for ig in 0..n_ang {
                    let (s, co) = (2.0 * PI * (ig as f64 + 0.5) / n_ang as f64).sin_cos();
                    let mut cell = [0usize; 3];
                    let mut inside = true;
                    for k in 0..3 {
                        let p = c.start[k] + axis[k] * t + (u[k] * co + v[k] * s) * rad;
                        let q = ((p - origin[k]) / size).floor();
                        if q < 0.0 || q >= shape[k] as f64 {
                            inside = false;
                            break;
                        }
                        cell[k] = q as usize;
                    }
                    if inside {
                        out[cell[0] + shape[0] * (cell[1] + shape[1] * cell[2])] += share;
                    }
                }
            }
        }
    }
}
