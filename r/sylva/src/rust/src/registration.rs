// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for rigid registration and transform matrices.

use extendr_api::prelude::*;
use sylva_rs::registration as sreg;
use sylva_rs::{PointCloud, Transform};

use crate::convert::{err, matrix4_from_r, matrix4_to_r, xyz_from_r, xyz_to_r, Result};

/// @noRd
#[extendr]
fn core_kabsch(source: Robj, target: Robj) -> Result<Robj> {
    Ok(matrix4_to_r(&sreg::kabsch(&xyz_from_r(&source)?, &xyz_from_r(&target)?).map_err(err)?))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_icp(source: Robj, target: Robj, init: Robj, max_correspondence_distance: f64, max_iterations: i32, tolerance: f64, method: &str, trim: f64, normal_k: i32) -> Result<List> {
    let s = PointCloud::new(xyz_from_r(&source)?);
    let t = PointCloud::new(xyz_from_r(&target)?);
    let init = if init.is_null() { None } else { Some(matrix4_from_r(&init)?) };
    let p = sreg::IcpParams { max_correspondence_distance, max_iterations: max_iterations.max(0) as usize, tolerance, method: method.to_string(), trim, normal_k: normal_k.max(0) as usize };
    let r = sreg::icp(&s, &t, init, &p).map_err(err)?;
    Ok(list!(transform = matrix4_to_r(&r.transform), info = list!(rmse = r.rmse, iterations = r.iterations as f64, n_correspondences = r.n_correspondences as f64)))
}

/// @noRd
#[extendr]
fn core_rotation_z(angle_deg: f64) -> Robj {
    matrix4_to_r(&Transform::rotation_z(angle_deg))
}

/// @noRd
#[extendr]
fn core_translation(dx: f64, dy: f64, dz: f64) -> Robj {
    matrix4_to_r(&Transform::translation(dx, dy, dz))
}

/// Points through the upper 3 x 4 block of a 4 x 4 matrix.
/// @noRd
#[extendr]
fn core_transform_xyz(xyz: Robj, matrix: Robj) -> Result<Robj> {
    let t = matrix4_from_r(&matrix)?;
    let p: Vec<_> = xyz_from_r(&xyz)?.iter().map(|q| t.apply(q)).collect();
    Ok(xyz_to_r(&p))
}

extendr_module! {
    mod registration;
    fn core_kabsch;
    fn core_icp;
    fn core_rotation_z;
    fn core_translation;
    fn core_transform_xyz;
}
