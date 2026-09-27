// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for laser pulses (shots) and RIEGL scans.

use extendr_api::prelude::*;
use sylva_rs::io::riegl;
use sylva_rs::{Shots, Transform};

use crate::convert::{cloud_from_r, cloud_to_r, doubles, err, fail, shots_from_r, shots_to_r, xyz_to_r, Result};

fn transform_from_r(m: &Robj) -> Result<Transform> {
    let v = doubles(m, "matrix")?;
    if v.len() != 16 {
        return fail("matrix must be 4 x 4");
    }
    // R matrices are column-major.
    Ok(Transform(nalgebra::Matrix4::from_column_slice(&v)))
}

#[allow(clippy::too_many_arguments)]
fn rxp_options(library: &Robj, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: i32, max_points: &Robj, echoes: &str, shot_stride: i32) -> Result<riegl::RxpOptions> {
    let library = if library.is_null() { None } else { Some(library.as_str().ok_or_else(|| Error::Other("library must be a path".into()))?.into()) };
    let max_points = if max_points.is_null() { None } else { Some(doubles(max_points, "max_points")?[0] as usize) };
    Ok(riegl::RxpOptions { library, drop_pseudo_echoes, min_range, max_range, stride: stride.max(1) as usize, max_points, echoes: echoes.to_string(), shot_stride: shot_stride.max(1) as usize })
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_read_rxp(path: &str, library: Robj, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: i32, max_points: Robj, echoes: &str, shot_stride: i32) -> Result<List> {
    let o = rxp_options(&library, drop_pseudo_echoes, min_range, max_range, stride, &max_points, echoes, shot_stride)?;
    Ok(cloud_to_r(&riegl::read_rxp(path, &o).map_err(err)?))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_read_rxp_shots(path: &str, library: Robj, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: i32, max_points: Robj, echoes: &str, shot_stride: i32) -> Result<List> {
    let o = rxp_options(&library, drop_pseudo_echoes, min_range, max_range, stride, &max_points, echoes, shot_stride)?;
    Ok(shots_to_r(&riegl::read_rxp_shots(path, &o).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_shots_check(shots: List) -> Result<List> {
    Ok(shots_to_r(&shots_from_r(&shots)?))
}

/// @noRd
#[extendr]
fn core_shots_from_cloud(cloud: List, origin: &[f64]) -> Result<List> {
    if origin.len() != 3 {
        return fail("origin must have three values");
    }
    Ok(shots_to_r(&Shots::from_pointcloud(&cloud_from_r(&cloud)?, [origin[0], origin[1], origin[2]])))
}

/// @noRd
#[extendr]
fn core_shots_from_ray_cloud(cloud: List) -> Result<List> {
    Ok(shots_to_r(&Shots::from_ray_cloud(&cloud_from_r(&cloud)?).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_shots_to_cloud(shots: List) -> Result<List> {
    Ok(cloud_to_r(&shots_from_r(&shots)?.to_pointcloud()))
}

/// @noRd
#[extendr]
fn core_shots_echo_xyz(shots: List) -> Result<Robj> {
    Ok(xyz_to_r(&shots_from_r(&shots)?.echo_xyz()))
}

/// @noRd
#[extendr]
fn core_shots_subset(shots: List, keep: Robj) -> Result<List> {
    let s = shots_from_r(&shots)?;
    let keep: Vec<bool> = keep.as_logical_slice().ok_or_else(|| Error::Other("keep must be logical".into()))?.iter().map(|b| b.is_true()).collect();
    if keep.len() != s.n_shots() {
        return fail("keep must have one value per shot");
    }
    Ok(shots_to_r(&s.subset(&keep)))
}

/// @noRd
#[extendr]
fn core_shots_transform(shots: List, matrix: Robj) -> Result<List> {
    Ok(shots_to_r(&shots_from_r(&shots)?.transformed(&transform_from_r(&matrix)?)))
}

/// Zenith (degrees from up) and azimuth (degrees clockwise from +y) per shot.
/// @noRd
#[extendr]
fn core_shots_zenith_azimuth(shots: List) -> Result<List> {
    let s = shots_from_r(&shots)?;
    let zen = sylva_rs::canopy_profile::zenith_deg(&s.direction);
    let az: Vec<f64> = s.direction.iter().map(|d| d[0].atan2(d[1]).to_degrees().rem_euclid(360.0)).collect();
    Ok(list!(zenith = zen, azimuth = az))
}

/// @noRd
#[extendr]
fn core_shots_shot_of_echo(shots: List) -> Result<Vec<f64>> {
    Ok(shots_from_r(&shots)?.shot_of_echo().into_iter().map(|v| v as f64).collect())
}

/// @noRd
#[extendr]
fn core_shots_echo_rank(shots: List) -> Result<Vec<i32>> {
    Ok(shots_from_r(&shots)?.echo_rank().into_iter().map(|v| v as i32).collect())
}

extendr_module! {
    mod shots;
    fn core_read_rxp;
    fn core_read_rxp_shots;
    fn core_shots_check;
    fn core_shots_from_cloud;
    fn core_shots_from_ray_cloud;
    fn core_shots_to_cloud;
    fn core_shots_echo_xyz;
    fn core_shots_subset;
    fn core_shots_transform;
    fn core_shots_zenith_azimuth;
    fn core_shots_shot_of_echo;
    fn core_shots_echo_rank;
}
