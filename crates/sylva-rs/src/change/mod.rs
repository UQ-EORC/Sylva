// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Change between two epochs of a plot.
//!
//! Every change carries an uncertainty or a level of detection, and anything
//! the data cannot support (space neither epoch observed, model parts filled
//! in by priors) is labelled rather than reported as change.

pub mod qsm;
