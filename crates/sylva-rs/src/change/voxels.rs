// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Occupancy change between two ray-traced voxel grids of one plot.
//!
//! In each epoch a voxel is *occupied* when it holds at least `min_hits`
//! echoes, *empty* when it holds none and at least `min_pulses` pulses
//! entered it before their last echo, and *not observed* otherwise (no
//! pulse reached it, only pulses already stopped did, or too few passed to
//! call it empty). A voxel occupied in one epoch and empty in the other is
//! only called lost (or gained) if the empty epoch's pulses were enough to
//! have found its contents: if the occupied epoch saw a share `p` of the
//! entering pulses stopped, `n` pulses all pass with probability
//! `(1 - p)^n`, which must not exceed `alpha`; sparse contents grazed by a
//! few pulses are otherwise unobserved rather than changed. Pairing the two
//! epochs gives, per voxel:
//!
//! | code | class | epoch a | epoch b |
//! |---|---|---|---|
//! | 0 | unobserved | not observed in one or both epochs | |
//! | 1 | stable empty | empty | empty |
//! | 2 | stable occupied | occupied | occupied |
//! | 3 | gained | empty | occupied |
//! | 4 | lost | occupied | empty |
//!
//! So a crown that the later scan could not see is *unobserved*, never
//! *lost*: absence is only reported where enough pulses went through.
//!
//! Plant area density change is given per voxel and per horizontal layer,
//! over the voxels entered by at least `min_pulses` pulses in both epochs
//! (the common support), so that what one epoch did not see does not bias
//! the layer means.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::voxel::{RayVoxels, I};

pub const UNOBSERVED: u8 = 0;
pub const STABLE_EMPTY: u8 = 1;
pub const STABLE_OCCUPIED: u8 = 2;
pub const GAINED: u8 = 3;
pub const LOST: u8 = 4;
/// Class names by code.
pub const CLASS_NAMES: [&str; 5] = ["unobserved", "stable_empty", "stable_occupied", "gained", "lost"];

/// Evidence criteria of [`occupancy`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OccupancyParams {
    /// Fewest pulses entering a voxel for it to count as empty, and for its
    /// plant area density to enter the comparison.
    pub min_pulses: u32,
    /// Fewest echoes for a voxel to count as occupied.
    pub min_hits: u32,
    /// Largest probability, for gained and lost voxels, that the pulses of
    /// the empty epoch all missed contents as dense as those of the occupied
    /// epoch (see [`miss_probability`]); above it the voxel is unobserved.
    pub alpha: f64,
}

impl OccupancyParams {
    fn check(&self) -> Result<()> {
        if self.min_pulses == 0 {
            return Err(Error::invalid("min_pulses must be at least 1"));
        }
        if self.min_hits == 0 {
            return Err(Error::invalid("min_hits must be at least 1"));
        }
        if !(self.alpha > 0.0 && self.alpha <= 1.0) {
            return Err(Error::invalid(format!("alpha must be in (0, 1], got {}", self.alpha)));
        }
        Ok(())
    }
}

/// Per-layer summary of an [`Occupancy`] (layers bottom first).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerChange {
    /// Layer centre z (m, grid frame).
    pub z: Vec<f64>,
    /// Voxels of each class per layer, `counts[class][layer]`.
    pub counts: [Vec<i64>; 5],
    /// Voxels compared for plant area density (entered by `min_pulses` in
    /// both epochs, finite density in both).
    pub n_compared: Vec<i64>,
    /// Mean plant area density of the compared voxels in each epoch, and
    /// their difference b - a (m² m⁻³); NaN where none is compared.
    pub pad_a: Vec<f64>,
    pub pad_b: Vec<f64>,
    pub pad_change: Vec<f64>,
}

/// Occupancy change between two grids (see the module documentation).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Occupancy {
    /// Class code per voxel, row-major `(nz, ny, nx)`.
    pub class: Vec<u8>,
    /// Plant area density of each epoch per voxel.
    pub pad_a: Vec<f64>,
    pub pad_b: Vec<f64>,
    /// Plant area density b - a per voxel; NaN outside the common support.
    pub pad_change: Vec<f64>,
    pub layers: LayerChange,
}

/// Probability that `n` pulses all pass through a voxel whose contents
/// stopped a share `hits / beams` of the pulses entering it in the other
/// epoch, `(1 - p)^n`.
pub fn miss_probability(hits: i32, beams: i32, n: i32) -> f64 {
    let p = if beams > 0 { (hits as f64 / beams as f64).min(1.0) } else { 1.0 };
    (1.0 - p).powi(n)
}

/// Classify voxels from their echo and pulse counts in two epochs.
pub fn classify(hits_a: &[i32], beams_a: &[i32], hits_b: &[i32], beams_b: &[i32], params: &OccupancyParams) -> Result<Vec<u8>> {
    params.check()?;
    let n = hits_a.len();
    if [beams_a.len(), hits_b.len(), beams_b.len()].iter().any(|&m| m != n) {
        return Err(Error::invalid("the two grids have different numbers of voxels"));
    }
    // 1 occupied, 0 empty, -1 not observed.
    let status = |h: i32, b: i32| -> i8 {
        if h >= params.min_hits as i32 {
            1
        } else if h == 0 && b >= params.min_pulses as i32 {
            0
        } else {
            -1
        }
    };
    Ok((0..n)
        .into_par_iter()
        .map(|i| match (status(hits_a[i], beams_a[i]), status(hits_b[i], beams_b[i])) {
            (0, 0) => STABLE_EMPTY,
            (1, 1) => STABLE_OCCUPIED,
            (0, 1) if miss_probability(hits_b[i], beams_b[i], beams_a[i]) <= params.alpha => GAINED,
            (1, 0) if miss_probability(hits_a[i], beams_a[i], beams_b[i]) <= params.alpha => LOST,
            _ => UNOBSERVED,
        })
        .collect())
}

/// Occupancy change from counts and densities of two grids of `shape`
/// `(nx, ny, nz)` with bottom at `z0` and voxel edge `size`.
#[allow(clippy::too_many_arguments)]
pub fn occupancy_from_arrays(shape: [usize; 3], z0: f64, size: f64, hits_a: &[i32], beams_a: &[i32], pad_a: &[f64], hits_b: &[i32], beams_b: &[i32], pad_b: &[f64], params: &OccupancyParams) -> Result<Occupancy> {
    let n = shape[0] * shape[1] * shape[2];
    for (name, len) in [("hits_a", hits_a.len()), ("beams_a", beams_a.len()), ("pad_a", pad_a.len()), ("hits_b", hits_b.len()), ("beams_b", beams_b.len()), ("pad_b", pad_b.len())] {
        if len != n {
            return Err(Error::invalid(format!("{name} has {len} values for {n} voxels")));
        }
    }
    let class = classify(hits_a, beams_a, hits_b, beams_b, params)?;
    let mp = params.min_pulses as i32;
    let pad_change: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| if beams_a[i] >= mp && beams_b[i] >= mp { pad_b[i] - pad_a[i] } else { f64::NAN })
        .map(|d| if d.is_finite() { d } else { f64::NAN })
        .collect();
    let per = shape[0] * shape[1];
    let nz = shape[2];
    let mut layers = LayerChange {
        z: (0..nz).map(|k| z0 + (k as f64 + 0.5) * size).collect(),
        counts: std::array::from_fn(|_| vec![0; nz]),
        n_compared: vec![0; nz],
        pad_a: vec![0.0; nz],
        pad_b: vec![0.0; nz],
        pad_change: vec![f64::NAN; nz],
    };
    for k in 0..nz {
        let (mut sa, mut sb, mut c) = (0.0, 0.0, 0i64);
        for i in k * per..(k + 1) * per {
            layers.counts[class[i] as usize][k] += 1;
            if pad_change[i].is_finite() {
                sa += pad_a[i];
                sb += pad_b[i];
                c += 1;
            }
        }
        layers.n_compared[k] = c;
        if c > 0 {
            layers.pad_a[k] = sa / c as f64;
            layers.pad_b[k] = sb / c as f64;
            layers.pad_change[k] = layers.pad_b[k] - layers.pad_a[k];
        } else {
            layers.pad_a[k] = f64::NAN;
            layers.pad_b[k] = f64::NAN;
        }
    }
    Ok(Occupancy { class, pad_a: pad_a.to_vec(), pad_b: pad_b.to_vec(), pad_change, layers })
}

/// Occupancy change between grid `a` (earlier) and grid `b` (later), which
/// must share origin, voxel size and shape (trace both epochs, registered to
/// one frame, with the same `bounds`). `pad` names the plant area density
/// metric compared (e.g. `pad_fpl`).
///
/// # Errors
/// Grids of different geometry, an unknown metric or invalid criteria.
pub fn occupancy(a: &RayVoxels, b: &RayVoxels, pad: &str, params: &OccupancyParams) -> Result<Occupancy> {
    params.check()?;
    if a.shape != b.shape {
        return Err(Error::invalid(format!("grids differ in shape: {:?} and {:?} (nx, ny, nz)", a.shape, b.shape)));
    }
    if (a.voxel_size - b.voxel_size).abs() > 1e-9 * a.voxel_size {
        return Err(Error::invalid(format!("grids differ in voxel size: {} and {}", a.voxel_size, b.voxel_size)));
    }
    if (0..3).any(|k| (a.origin[k] - b.origin[k]).abs() > 1e-6 * a.voxel_size.max(1.0)) {
        return Err(Error::invalid(format!("grids differ in origin: {:?} and {:?}; trace both epochs with the same bounds", a.origin, b.origin)));
    }
    let (pa, pb) = (a.metric(pad)?, b.metric(pad)?);
    let (hits, beams) = (I::NumHits as usize, I::NumBeams as usize);
    occupancy_from_arrays(a.shape, a.origin[2], a.voxel_size, &a.i[hits], &a.i[beams], &pa, &b.i[hits], &b.i[beams], &pb, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: OccupancyParams = OccupancyParams { min_pulses: 5, min_hits: 1, alpha: 1.0 };

    #[test]
    fn classes_follow_the_table() {
        // a: occupied, empty, empty (few pulses), not reached, occupied, occupied
        // b: occupied, occupied, empty, occupied, empty (few), empty
        let ha = [3, 0, 0, 0, 2, 1];
        let ba = [10, 10, 2, 0, 9, 9];
        let hb = [1, 4, 0, 5, 0, 0];
        let bb = [10, 10, 10, 10, 3, 6];
        let c = classify(&ha, &ba, &hb, &bb, &P).unwrap();
        assert_eq!(c, vec![STABLE_OCCUPIED, GAINED, UNOBSERVED, UNOBSERVED, UNOBSERVED, LOST]);
        let strict = OccupancyParams { min_pulses: 5, min_hits: 2, alpha: 1.0 };
        let c = classify(&ha, &ba, &hb, &bb, &strict).unwrap();
        assert_eq!(c[0], UNOBSERVED);
        assert_eq!(c[5], UNOBSERVED);
        assert!(classify(&ha, &ba, &hb[..2], &bb, &P).is_err());
        assert!(classify(&ha, &ba, &hb, &bb, &OccupancyParams { min_pulses: 0, min_hits: 1, alpha: 1.0 }).is_err());
        assert!(classify(&ha, &ba, &hb, &bb, &OccupancyParams { min_pulses: 5, min_hits: 1, alpha: 0.0 }).is_err());
    }

    #[test]
    fn sparse_contents_need_enough_pulses() {
        // Occupied by 1 of 50 pulses, then 20 pulses without a hit: (0.98)^20 = 0.67.
        let p = OccupancyParams { min_pulses: 5, min_hits: 1, alpha: 0.05 };
        assert!((miss_probability(1, 50, 20) - 0.98f64.powi(20)).abs() < 1e-15);
        assert_eq!(classify(&[1], &[50], &[0], &[20], &p).unwrap(), vec![UNOBSERVED]);
        // Dense contents (20 of 50) and 20 pulses: 0.6^20, far below alpha.
        assert_eq!(classify(&[20], &[50], &[0], &[20], &p).unwrap(), vec![LOST]);
        assert_eq!(classify(&[0], &[20], &[20], &[50], &p).unwrap(), vec![GAINED]);
        assert_eq!(classify(&[0], &[20], &[1], &[50], &p).unwrap(), vec![UNOBSERVED]);
    }

    #[test]
    fn layers_average_over_the_common_support() {
        // 2 x 1 x 2 grid: layer 0 both compared, layer 1 one voxel unseen in b.
        let shape = [2, 1, 2];
        let ha = [1, 0, 1, 1];
        let ba = [10, 10, 10, 10];
        let pa = [1.0, 0.0, 2.0, 4.0];
        let hb = [1, 1, 0, 0];
        let bb = [10, 10, 10, 0];
        let pb = [0.5, 1.0, 0.0, f64::NAN];
        let o = occupancy_from_arrays(shape, 100.0, 0.5, &ha, &ba, &pa, &hb, &bb, &pb, &P).unwrap();
        assert_eq!(o.class, vec![STABLE_OCCUPIED, GAINED, LOST, UNOBSERVED]);
        assert_eq!(o.layers.z, vec![100.25, 100.75]);
        assert_eq!(o.layers.n_compared, vec![2, 1]);
        assert_eq!(o.layers.pad_a, vec![0.5, 2.0]);
        assert_eq!(o.layers.pad_b, vec![0.75, 0.0]);
        assert_eq!(o.layers.pad_change, vec![0.25, -2.0]);
        assert!(o.pad_change[3].is_nan());
        assert_eq!(o.layers.counts[LOST as usize], vec![0, 1]);
        assert_eq!(o.layers.counts[UNOBSERVED as usize], vec![0, 1]);
        assert!(occupancy_from_arrays(shape, 0.0, 0.5, &ha, &ba, &pa[..3], &hb, &bb, &pb, &P).is_err());
    }
}
