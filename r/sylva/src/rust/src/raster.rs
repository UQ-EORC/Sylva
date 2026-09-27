// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for rasters: sampling, hole filling, cell geometry and ASCII grids.

use extendr_api::prelude::*;
use sylva_rs::raster as sr;
use sylva_rs::Raster;

use crate::convert::{err, fail, matrix_from_rows, raster_from_r, raster_to_r, Result};

/// @noRd
#[extendr]
fn core_raster_sample(data: Robj, xmin: f64, ymin: f64, resolution: f64, x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
    if x.len() != y.len() {
        return fail("x and y must have the same length");
    }
    let r = raster_from_r(&data, xmin, ymin, resolution)?;
    Ok(r.sample_many(x.iter().cloned().zip(y.iter().cloned())))
}

/// @noRd
#[extendr]
fn core_raster_fill_nearest(data: Robj) -> Result<Robj> {
    let mut r = raster_from_r(&data, 0.0, 0.0, 1.0)?;
    r.fill_nearest();
    Ok(matrix_from_rows(r.nrows, r.ncols, &r.data))
}

/// 1-based rows and columns (NaN where there is no cell index).
/// @noRd
#[extendr]
fn core_raster_cell_index(xmin: f64, ymin: f64, resolution: f64, x: &[f64], y: &[f64]) -> List {
    let (r, c) = sr::cell_indices(xmin, ymin, resolution, x, y);
    let pos = |v: Vec<i64>| -> Vec<f64> { v.into_iter().map(|i| if i == i64::MIN { f64::NAN } else { i as f64 + 1.0 }).collect() };
    list!(row = pos(r), col = pos(c))
}

/// @noRd
#[extendr]
fn core_raster_cell_centers(nrows: i32, ncols: i32, xmin: f64, ymin: f64, resolution: f64) -> List {
    let (nr, nc) = (nrows.max(0) as usize, ncols.max(0) as usize);
    let (x, y) = sr::cell_centers(nr, nc, xmin, ymin, resolution);
    list!(X = matrix_from_rows(nr, nc, &x), Y = matrix_from_rows(nr, nc, &y))
}

/// @noRd
#[extendr]
fn core_write_ascii_grid(path: &str, data: Robj, xmin: f64, ymin: f64, resolution: f64, nodata: f64) -> Result<()> {
    raster_from_r(&data, xmin, ymin, resolution)?.write_ascii_grid(path, nodata).map_err(err)
}

/// @noRd
#[extendr]
fn core_read_ascii_grid(path: &str) -> Result<List> {
    Ok(raster_to_r(&Raster::read_ascii_grid(path).map_err(err)?))
}

extendr_module! {
    mod raster;
    fn core_raster_sample;
    fn core_raster_fill_nearest;
    fn core_raster_cell_index;
    fn core_raster_cell_centers;
    fn core_write_ascii_grid;
    fn core_read_ascii_grid;
}
