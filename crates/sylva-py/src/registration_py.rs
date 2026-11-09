// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for transform matrices and applying them to points.
//!
//! Backs sylva.registration and the callers that build rigid transforms
//! (sylva.coreg, sylva.riscan, the CLI) with sylva_rs::Transform. Matrices
//! are 4x4 float64 arrays and points `(N, 3)` arrays; each function returns
//! a new array, and a wrong shape is a ValueError from the
//! `crate::matrix_from_py` / `crate::xyz_from_py` helpers.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use sylva_rs::Transform;

use crate::{matrix_from_py, matrix_to_py, xyz_from_py, xyz_to_py};

#[pyfunction]
fn rotation_z<'py>(py: Python<'py>, angle_deg: f64) -> Bound<'py, PyArray2<f64>> {
    matrix_to_py(py, &Transform::rotation_z(angle_deg))
}

#[pyfunction]
fn translation<'py>(py: Python<'py>, dx: f64, dy: f64, dz: f64) -> Bound<'py, PyArray2<f64>> {
    matrix_to_py(py, &Transform::translation(dx, dy, dz))
}

/// Points through the upper 3 x 4 block of a 4 x 4 matrix.
#[pyfunction]
fn transform_xyz<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, matrix: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = xyz_from_py(xyz)?;
    let t = matrix_from_py(Some(matrix))?.expect("a matrix was given");
    let out: Vec<_> = py.detach(|| p.iter().map(|q| t.apply(q)).collect());
    Ok(xyz_to_py(py, &out))
}

/// Adds this file's functions to the `_core` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rotation_z, m)?)?;
    m.add_function(wrap_pyfunction!(translation, m)?)?;
    m.add_function(wrap_pyfunction!(transform_xyz, m)?)?;
    Ok(())
}
