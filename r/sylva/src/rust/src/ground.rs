// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for ground classification, terrain and canopy height models.

use extendr_api::prelude::*;
use sylva_rs::ground as sg;

use crate::convert::{bounds_from_r, err, fail, logicals, raster_from_r, raster_to_r, xyz_from_r, Result};

/// @noRd
#[extendr]
fn core_csf_ground_mask(xyz: Robj, cloth_resolution: f64, rigidness: i32, class_threshold: f64, iterations: i32, time_step: f64) -> Result<Logicals> {
    if !(1..=3).contains(&rigidness) {
        return fail("rigidness must be 1, 2 or 3");
    }
    let p = sg::CsfParams { cloth_resolution, rigidness: rigidness as usize, class_threshold, iterations: iterations.max(0) as usize, time_step };
    Ok(logicals(&sg::csf_ground_mask(&xyz_from_r(&xyz)?, &p)))
}

/// @noRd
#[extendr]
fn core_pmf_ground_mask(xyz: Robj, cell_size: f64, max_window: f64, slope: f64, initial_distance: f64, max_distance: f64) -> Result<Logicals> {
    let p = sg::PmfParams { cell_size, max_window, slope, initial_distance, max_distance };
    Ok(logicals(&sg::pmf_ground_mask(&xyz_from_r(&xyz)?, &p).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_make_dtm(ground_xyz: Robj, resolution: f64, bounds: Robj) -> Result<List> {
    Ok(raster_to_r(&sg::make_dtm(&xyz_from_r(&ground_xyz)?, resolution, bounds_from_r(&bounds)?).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_make_chm(xyz: Robj, heights: &[f64], resolution: f64, bounds: Robj, min_height: f64) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    if heights.len() != p.len() {
        return fail("heights must have one value per point");
    }
    Ok(raster_to_r(&sg::make_chm(&p, heights, resolution, bounds_from_r(&bounds)?, min_height).map_err(err)?))
}

/// z minus the raster sampled at each point's x, y.
/// @noRd
#[extendr]
fn core_heights_above(xyz: Robj, data: Robj, xmin: f64, ymin: f64, resolution: f64) -> Result<Vec<f64>> {
    Ok(sg::heights_above(&xyz_from_r(&xyz)?, &raster_from_r(&data, xmin, ymin, resolution)?))
}

extendr_module! {
    mod ground;
    fn core_csf_ground_mask;
    fn core_pmf_ground_mask;
    fn core_make_dtm;
    fn core_make_chm;
    fn core_heights_above;
}
