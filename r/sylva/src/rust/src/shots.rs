// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for laser pulses (shots) and RIEGL scans.

use extendr_api::prelude::*;
use sylva_rs::io::{riegl, shots as shots_io};
use sylva_rs::{Shots, Transform};

use crate::canopy::pattern_from_r;
use crate::convert::{cloud_from_r, cloud_to_r, doubles, err, fail, optional_f64, shots_from_r, shots_to_r, xyz_to_r, Result};

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
    let (zen, az) = shots_from_r(&shots)?.zenith_azimuth();
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

/// @noRd
#[extendr]
fn core_shots_concatenate(parts: List) -> Result<List> {
    let parts: Vec<Shots> = parts.values().map(|p| shots_from_r(&p.try_into()?)).collect::<Result<_>>()?;
    Ok(shots_to_r(&Shots::concatenate(&parts.iter().collect::<Vec<_>>()).map_err(err)?))
}

/// The shots followed by the misses of the scan pattern, or NULL if none.
/// @noRd
#[extendr]
fn core_shots_fill_missing(shots: List, pattern: List, pulses_per_line: Robj, seed: f64, shot_stride: i32) -> Result<Robj> {
    if !(seed >= 0.0 && seed.fract() == 0.0 && seed < 1.8446744073709552e19) {
        return fail("seed must be a non-negative whole number");
    }
    let per_line = optional_f64(&pulses_per_line, "pulses_per_line")?.map(|v| v as i64);
    let s = shots_from_r(&shots)?;
    Ok(match s.fill_missing(&pattern_from_r(&pattern)?, per_line, seed as u64, shot_stride.max(1) as usize) {
        Some(f) => shots_to_r(&f).into(),
        None => ().into(),
    })
}

/// @noRd
#[extendr]
fn core_write_shots(shots: List, path: &str, double: bool, row_group_size: f64, zstd_level: i32, origin_tolerance: f64) -> Result<()> {
    let opts = shots_io::ShotsWriteOptions { double, row_group_size: row_group_size as usize, zstd_level, origin_tolerance };
    shots_io::write_shots(&shots_from_r(&shots)?, path, &opts).map_err(err)
}

/// Row groups `groups` (0-based), or all when NULL.
/// @noRd
#[extendr]
fn core_read_shots(path: &str, groups: Robj) -> Result<List> {
    let file = shots_io::ShotsFile::open(path).map_err(err)?;
    let s = if groups.is_null() {
        file.read_all()
    } else {
        let g: Vec<usize> = doubles(&groups, "groups")?.iter().map(|&v| v as usize).collect();
        file.read_groups(&g)
    };
    Ok(shots_to_r(&s.map_err(err)?))
}

/// @noRd
#[extendr]
fn core_shots_info(path: &str) -> Result<List> {
    let file = shots_io::ShotsFile::open(path).map_err(err)?;
    let (lo, hi) = file.bounds;
    Ok(list!(
        n_shots = file.n_shots as f64,
        n_echoes = file.n_echoes as f64,
        n_groups = file.n_groups() as f64,
        bounds = list!(lo.to_vec(), hi.to_vec()),
        scans = xyz_to_r(file.scans()),
        echo_attrs = file.attr_names().map(|s| s.to_string()).collect::<Vec<_>>()
    ))
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
    fn core_shots_concatenate;
    fn core_shots_fill_missing;
    fn core_write_shots;
    fn core_read_shots;
    fn core_shots_info;
}
