// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Quantitative structure models: skeletonisation and cylinder fitting.
//!
//! Pipeline: [`skeletonize`] bins geodesic distance from the base over a kNN
//! graph and splits each bin into connected segments; [`fit_cylinders`] fits a
//! cylinder per segment and links parents to build a [`Qsm`].

pub mod buttress;
pub mod cylinder;
pub mod metrics;
pub mod model;
pub mod wood;

pub use cylinder::{fit_cylinder, fit_cylinder_ransac, CylinderFit};
pub use model::{build_qsm, fit_cylinders, skeletonize, Cylinder, Qsm, QsmParams, Skeleton};
