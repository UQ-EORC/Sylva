// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::synthetic: the synthetic tree, forest and scan.

use extendr_api::prelude::*;
use sylva_rs::synthetic;

use crate::convert::{cloud_from_r, cloud_to_r, fail, point_from_r, shots_to_r, Result};

fn seed_from_r(seed: f64) -> Result<u64> {
    if seed >= 0.0 && seed.fract() == 0.0 && seed < 1.8446744073709552e19 {
        Ok(seed as u64)
    } else {
        fail("seed must be a non-negative whole number")
    }
}

fn count_from_r(v: f64, what: &str) -> Result<usize> {
    if v >= 0.0 && v.is_finite() {
        Ok(v as usize)
    } else {
        fail(format!("{what} must be a non-negative number"))
    }
}

/// @noRd
#[extendr]
fn core_synthetic_terrain_height(x: &[f64], y: &[f64], slope: f64) -> Result<Vec<f64>> {
    if x.len() != y.len() {
        return fail("x and y differ in length");
    }
    Ok(x.iter().zip(y).map(|(&a, &b)| synthetic::terrain_height(a, b, slope)).collect())
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_synthetic_tree(x: f64, y: f64, dbh: f64, height: f64, z0: f64, n_branches: f64, leaf_points: f64, seed: f64) -> Result<List> {
    let (nb, lp) = (count_from_r(n_branches, "n_branches")?, count_from_r(leaf_points, "leaf_points")?);
    Ok(cloud_to_r(&synthetic::tree(x, y, dbh, height, z0, nb, lp, seed_from_r(seed)?)))
}

/// @noRd
#[extendr]
fn core_synthetic_leaf_area(classification: &[f64]) -> f64 {
    synthetic::leaf_area(classification)
}

/// @noRd
#[extendr]
fn core_synthetic_forest(x: &[f64], y: &[f64], dbh: &[f64], height: &[f64], size: f64, ground_points: f64, margin: f64, seed: f64) -> Result<List> {
    let n = x.len();
    if y.len() != n || dbh.len() != n || height.len() != n {
        return fail("trees: x, y, dbh and height differ in length");
    }
    let trees: Vec<(f64, f64, f64, f64)> = (0..n).map(|i| (x[i], y[i], dbh[i], height[i])).collect();
    Ok(cloud_to_r(&synthetic::forest(&trees, size, count_from_r(ground_points, "ground_points")?, margin, seed_from_r(seed)?)))
}

/// @noRd
#[extendr]
fn core_synthetic_scan(cloud: List, origin: &[f64], resolution_deg: f64, max_zenith_deg: f64, max_echoes: f64, echo_separation: f64) -> Result<List> {
    let c = cloud_from_r(&cloud)?;
    let o = point_from_r(origin, "origin")?;
    Ok(shots_to_r(&synthetic::scan(&c, o, resolution_deg, max_zenith_deg, count_from_r(max_echoes, "max_echoes")?, echo_separation)))
}

extendr_module! {
    mod synthetic;
    fn core_synthetic_terrain_height;
    fn core_synthetic_tree;
    fn core_synthetic_leaf_area;
    fn core_synthetic_forest;
    fn core_synthetic_scan;
}
