//! Quantitative structure models: skeletonisation and cylinder fitting.
//!
//! Pipeline: [`skeletonize`] bins geodesic distance from the base over a kNN
//! graph and splits each bin into connected segments; [`fit_cylinders`] fits a
//! cylinder per segment and links parents to build a [`Qsm`].

pub mod cylinder;
pub mod model;
pub mod wood;

pub use cylinder::{fit_cylinder, fit_cylinder_ransac, CylinderFit};
pub use model::{build_qsm, fit_cylinders, skeletonize, Cylinder, Qsm, QsmParams, Skeleton};
