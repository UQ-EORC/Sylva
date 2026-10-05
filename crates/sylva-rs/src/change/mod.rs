// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Change detection between two epochs of one plot.
//!
//! The principle throughout is to separate real change from noise and from
//! not having seen something: every change carries its uncertainty or level
//! of detection, and whatever the data cannot support is labelled as such
//! rather than reported as change.
//!
//! * [`epochs`]: the second epoch aligned onto the first on stable features
//!   (stems and ground), with the uncertainty of that alignment;
//! * [`trees`]: trees matched between epochs (survivors, deaths, recruits)
//!   and their increments with a minimum detectable increment;
//! * [`summary`]: plot-level growth, mortality and recruitment with Monte
//!   Carlo intervals;
//! * [`provenance`]: the settings each epoch was processed with, compared;
//! * [`synthetic`]: two scanned epochs of a synthetic plot with known
//!   changes, the reference every function here is validated against;
//! * [`points`]: cloud-to-cloud and M3C2 distances, rasters of difference;
//! * [`voxels`]: voxel occupancy change from two ray-traced grids;
//! * [`qsm`]: QSMs compared, branch by branch, where both were measured.

pub mod als;
pub mod epochs;
pub mod points;
pub mod provenance;
pub mod qsm;
pub mod summary;
pub mod synthetic;
pub mod trees;
pub mod voxels;

/// `v > 0`, false for NaN.
pub(crate) fn positive(v: f64) -> bool {
    v > 0.0
}

/// `v >= 0`, false for NaN.
pub(crate) fn non_negative(v: f64) -> bool {
    v >= 0.0
}
