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

pub mod buttress_detect;
pub mod canopy;
pub mod canopy_profile;
pub mod cluster;
pub mod coreg;
pub mod coreg_ground;
pub mod coreg_geometry;
pub mod coreg_icp;
pub mod coreg_pipeline;
pub mod coreg_posegraph;
pub mod coreg_reflectors;
pub mod coreg_refine;
pub mod coreg_stemmap;
pub mod coreg_survey;
pub mod coreg_transforms;
pub mod error;
pub mod filters;
pub mod ground;
pub mod io;
pub mod json;
pub mod limits;
pub mod masks;
pub mod leaves;
pub mod nprandom;
pub mod nprandom_dist;
pub mod leaf_model;
pub mod mesh_io;
pub mod numeric;
pub mod optim;
pub mod pointcloud;
pub mod progress;
pub mod pyformat;
pub mod qsm;
pub mod qsm_ops;
pub mod qsm_plot;
pub mod quality;
pub mod quality_summary;
pub mod raster;
pub mod registration;
pub mod relay;
pub mod riscan;
pub mod shots;
pub mod shots_ops;
pub mod spatial;
pub mod stems;
pub mod synthetic;
pub mod transform;
pub mod tree_prune;
pub mod trees;
pub mod voxel;
pub mod voxel_grid;

pub use error::{Error, Result};
pub use pointcloud::{Attr, PointCloud};
pub use raster::Raster;
pub use shots::Shots;
pub use transform::Transform;

/// A 3-D point.
pub type Point = [f64; 3];
