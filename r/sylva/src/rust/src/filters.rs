// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for subsampling, cropping, outlier removal and local geometry.
//! Selections come back as 1-based positions or logical masks; `R/filters.R`
//! subsets the cloud with them.

use extendr_api::prelude::*;
use sylva_rs::{cluster, filters as sf, PointCloud};

use crate::convert::{fail, logicals, point_from_r, positions, xyz_from_r, xyz_to_r, Result};

fn positive(v: f64, what: &str) -> Result<()> {
    if v.is_nan() || v <= 0.0 {
        return fail(format!("{what} must be positive"));
    }
    Ok(())
}

/// @noRd
#[extendr]
fn core_voxel_downsample_indices(xyz: Robj, voxel_size: f64) -> Result<Vec<f64>> {
    positive(voxel_size, "voxel_size")?;
    Ok(positions(&sf::voxel_downsample_indices(&xyz_from_r(&xyz)?, voxel_size)))
}

/// @noRd
#[extendr]
fn core_voxel_centroids(xyz: Robj, voxel_size: f64) -> Result<Robj> {
    positive(voxel_size, "voxel_size")?;
    let c = PointCloud::new(xyz_from_r(&xyz)?);
    Ok(xyz_to_r(&sf::voxel_downsample(&c, voxel_size, true).xyz))
}

/// @noRd
#[extendr]
fn core_random_indices(total: f64, n: f64, seed: f64) -> Result<Vec<f64>> {
    if [total, n, seed].iter().any(|v| v.is_nan() || *v < 0.0) {
        return fail("n and seed must be non-negative");
    }
    Ok(positions(&sf::random_indices(total as usize, n as usize, seed as u64)))
}

/// @noRd
#[extendr]
fn core_min_distance_indices(xyz: Robj, distance: f64) -> Result<Vec<f64>> {
    Ok(positions(&sf::min_distance_indices(&xyz_from_r(&xyz)?, distance)))
}

/// @noRd
#[extendr]
fn core_crop_box_mask(xyz: Robj, min_xyz: &[f64], max_xyz: &[f64]) -> Result<Logicals> {
    Ok(logicals(&sf::crop_box_mask(&xyz_from_r(&xyz)?, point_from_r(min_xyz, "min_xyz")?, point_from_r(max_xyz, "max_xyz")?)))
}

/// @noRd
#[extendr]
fn core_crop_cylinder_mask(xyz: Robj, cx: f64, cy: f64, radius: f64, zmin: f64, zmax: f64) -> Result<Logicals> {
    Ok(logicals(&sf::crop_cylinder_mask(&xyz_from_r(&xyz)?, cx, cy, radius, zmin, zmax)))
}

/// @noRd
#[extendr]
fn core_range_mask(xyz: Robj, origin: &[f64], min_range: f64, max_range: f64) -> Result<Logicals> {
    Ok(logicals(&sf::range_mask(&xyz_from_r(&xyz)?, point_from_r(origin, "origin")?, min_range, max_range)))
}

/// @noRd
#[extendr]
fn core_statistical_outlier_mask(xyz: Robj, k: i32, std_ratio: f64) -> Result<Logicals> {
    Ok(logicals(&sf::statistical_outlier_mask(&xyz_from_r(&xyz)?, k.max(0) as usize, std_ratio)))
}

/// @noRd
#[extendr]
fn core_radius_outlier_mask(xyz: Robj, radius: f64, min_neighbors: i32) -> Result<Logicals> {
    Ok(logicals(&sf::radius_outlier_mask(&xyz_from_r(&xyz)?, radius, min_neighbors.max(0) as usize)))
}

/// @noRd
#[extendr]
fn core_estimate_normals(xyz: Robj, k: i32) -> Result<Robj> {
    Ok(xyz_to_r(&sf::estimate_normals(&xyz_from_r(&xyz)?, k.max(0) as usize)))
}

/// @noRd
#[extendr]
fn core_planarity_linearity(xyz: Robj, k: i32) -> Result<List> {
    let (planarity, linearity) = sf::planarity_linearity(&xyz_from_r(&xyz)?, k.max(0) as usize);
    Ok(list!(planarity = planarity, linearity = linearity))
}

/// @noRd
#[extendr]
fn core_euclidean_clusters(xyz: Robj, radius: f64, min_points: i32) -> Result<Vec<f64>> {
    Ok(cluster::euclidean_clusters(&xyz_from_r(&xyz)?, radius, min_points.max(0) as usize).into_iter().map(|v| v as f64).collect())
}

/// Distances and 1-based indices, `m x k`; NaN where fewer than `k` points exist.
/// @noRd
#[extendr]
fn core_knn(xyz: Robj, queries: Robj, k: i32) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    let q = xyz_from_r(&queries)?;
    let k = k.max(0) as usize;
    let (d, idx) = sf::knn(&p, &q, k);
    let i: Vec<f64> = idx.iter().map(|&v| if v < 0 { f64::NAN } else { v as f64 + 1.0 }).collect();
    let m = q.len();
    Ok(list!(distances = RMatrix::new_matrix(m, k, |r, c| d[r * k + c]), indices = RMatrix::new_matrix(m, k, |r, c| i[r * k + c])))
}

extendr_module! {
    mod filters;
    fn core_voxel_downsample_indices;
    fn core_voxel_centroids;
    fn core_random_indices;
    fn core_min_distance_indices;
    fn core_crop_box_mask;
    fn core_crop_cylinder_mask;
    fn core_range_mask;
    fn core_statistical_outlier_mask;
    fn core_radius_outlier_mask;
    fn core_estimate_normals;
    fn core_planarity_linearity;
    fn core_euclidean_clusters;
    fn core_knn;
}
