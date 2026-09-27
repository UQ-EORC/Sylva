// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for reading and writing point clouds and transform files.

use extendr_api::prelude::*;
use sylva_rs::io as sio;
use sylva_rs::Transform;

use crate::convert::{cloud_from_r, cloud_to_r, err, fail, matrix4_to_r, Result};

/// Read a point cloud (LAS/LAZ, PLY, ASCII or RIEGL RXP).
/// @noRd
#[extendr]
fn core_read(path: &str) -> Result<List> {
    Ok(cloud_to_r(&sio::read(path).map_err(err)?))
}

/// Write a point cloud, the format taken from the extension.
/// @noRd
#[extendr]
fn core_write(cloud: List, path: &str, point_format: i32, scale: f64, binary: bool) -> Result<()> {
    if !(0..=10).contains(&point_format) {
        return fail("point_format must be a LAS point data record format (0 to 10)");
    }
    let opts = sio::WriteOptions { point_format: point_format as u8, scale, binary };
    sio::write_with(&cloud_from_r(&cloud)?, path, &opts).map_err(err)
}

/// @noRd
#[extendr]
fn core_read_ascii(path: &str, columns: Robj) -> Result<List> {
    let columns: Option<Vec<String>> = if columns.is_null() {
        None
    } else {
        Some(columns.as_string_vector().ok_or_else(|| Error::Other("columns must be a character vector".into()))?)
    };
    Ok(cloud_to_r(&sio::ascii::read_ascii(path, columns.as_deref()).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_find_rivlib(hint: Robj) -> Result<String> {
    let hint = if hint.is_null() { None } else { Some(std::path::PathBuf::from(hint.as_str().ok_or_else(|| Error::Other("hint must be a path".into()))?)) };
    Ok(sio::riegl::find_rivlib(hint.as_deref()).map_err(err)?.to_string_lossy().into_owned())
}

/// @noRd
#[extendr]
fn core_read_matrix_file(path: &str) -> Result<Robj> {
    Ok(matrix4_to_r(&Transform::read_matrix_file(path).map_err(err)?))
}

/// Check and normalise a cloud built in R (lengths, types).
/// @noRd
#[extendr]
fn core_cloud(cloud: List) -> Result<List> {
    Ok(cloud_to_r(&cloud_from_r(&cloud)?))
}

extendr_module! {
    mod io;
    fn core_read;
    fn core_write;
    fn core_read_ascii;
    fn core_find_rivlib;
    fn core_read_matrix_file;
    fn core_cloud;
}
