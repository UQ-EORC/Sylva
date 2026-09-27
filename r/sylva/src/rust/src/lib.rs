// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! R bindings for sylva-rs.
//!
//! Point clouds cross the boundary as `list(xyz = <n x 3 double matrix>,
//! attrs = <named list of vectors>)`; see `R/` for the friendly wrappers.
//! Attribute types map to R as: floats and 64-bit or unsigned 32-bit
//! integers to double, smaller integers to integer, booleans to logical.

use extendr_api::prelude::*;
use sylva_rs::{filters, io};

mod canopy;
mod shots;
mod riscan;
mod convert;

use convert::{cloud_from_r, cloud_to_r, err, Result};

// ------------------------------------------------------------------------ I/O

/// Read a point cloud (LAS/LAZ, PLY, ASCII or RIEGL RXP).
/// @noRd
#[extendr]
fn core_read(path: &str) -> Result<List> {
    Ok(cloud_to_r(&io::read(path).map_err(err)?))
}

/// Write a point cloud, the format taken from the extension.
/// @noRd
#[extendr]
fn core_write(cloud: List, path: &str) -> Result<()> {
    io::write(&cloud_from_r(&cloud)?, path).map_err(err)
}

/// Check and normalise a cloud built in R (lengths, types).
/// @noRd
#[extendr]
fn core_cloud(cloud: List) -> Result<List> {
    Ok(cloud_to_r(&cloud_from_r(&cloud)?))
}

// -------------------------------------------------------------------- filters

/// One point per voxel: the first, or the centroid.
/// @noRd
#[extendr]
fn core_voxel_downsample(cloud: List, voxel_size: f64, centroid: bool) -> Result<List> {
    if !(voxel_size > 0.0) {
        return Err(Error::Other("voxel_size must be positive".into()));
    }
    Ok(cloud_to_r(&filters::voxel_downsample(&cloud_from_r(&cloud)?, voxel_size, centroid)))
}

extendr_module! {
    mod sylva;
    use canopy;
    use shots;
    use riscan;
    fn core_read;
    fn core_write;
    fn core_cloud;
    fn core_voxel_downsample;
}
