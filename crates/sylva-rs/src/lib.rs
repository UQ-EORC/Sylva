//! sylva-rs: terrestrial laser scanning processing for forest ecology.
//!
//! Copyright (C) 2026 Tim Devereux, The University of Queensland. Free
//! software under the GNU General Public License v3.0 or later; see the
//! LICENSE file. There is no warranty, to the extent permitted by law.
//!
//! The crate is organised around two data models:
//!
//! * [`PointCloud`] — an `(N, 3)` set of coordinates plus named per-point
//!   attributes, the unit of exchange with LAS/LAZ/PLY/text files.
//! * [`Shots`] — pulse-centric data (per-pulse origin and direction, with a
//!   CSR list of echoes), which is what ray-based canopy metrics need.
//!
//! Algorithms are grouped by topic: [`filters`], [`cluster`], [`ground`],
//! [`canopy`], [`voxel`], [`trees`], [`registration`] and [`qsm`].

pub mod als;
pub mod canopy;
pub mod change;
pub mod cluster;
pub mod coreg;
pub mod error;
pub mod filters;
pub mod fusion;
pub mod geo;
pub mod ground;
pub mod interpolate;
pub mod io;
pub mod leaves;
pub mod masks;
pub mod pointcloud;
pub mod qsm;
pub mod quality;
pub mod raster;
pub mod registration;
pub mod riscan;
pub mod shots;
pub mod synthetic;
pub mod transform;
pub mod trees;
pub mod util;
pub mod voxel;
pub mod waveform;

pub use error::{Error, Result};
pub use pointcloud::{Attr, PointCloud};
pub use raster::Raster;
pub use shots::Shots;
pub use transform::Transform;

/// A 3-D point.
pub type Point = [f64; 3];
