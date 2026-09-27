// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Change detection between two epochs of one plot.
//!
//! Real change is separated from noise and from not having seen something:
//! every change carries an uncertainty or a level of detection, and what the
//! data cannot support (space neither epoch observed, model parts filled in
//! by priors) is labelled rather than reported as change.

pub mod points;
pub mod voxels;
pub mod qsm;
