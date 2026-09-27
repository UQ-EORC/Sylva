// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::synthetic: the synthetic tree, forest and scan.
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::synthetic;

use crate::{cloud_from_py, cloud_to_py, shots_to_py};

/// Terrain height at each `(x, y)`.
#[pyfunction]
fn synthetic_terrain_height<'py>(py: Python<'py>, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>, slope: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (x, y) = (x.as_array(), y.as_array());
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y differ in length"));
    }
    Ok(x.iter().zip(y.iter()).map(|(&a, &b)| synthetic::terrain_height(a, b, slope)).collect::<Vec<_>>().into_pyarray(py))
}

#[pyfunction]
fn synthetic_tree<'py>(py: Python<'py>, x: f64, y: f64, dbh: f64, height: f64, z0: f64, n_branches: usize, leaf_points: usize, seed: u64) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    cloud_to_py(py, &py.detach(|| synthetic::tree(x, y, dbh, height, z0, n_branches, leaf_points, seed)))
}

#[pyfunction]
fn synthetic_leaf_area(classification: PyReadonlyArray1<f64>) -> f64 {
    synthetic::leaf_area(&classification.as_array().to_vec())
}

#[pyfunction]
fn synthetic_forest<'py>(py: Python<'py>, trees: Vec<(f64, f64, f64, f64)>, size: f64, ground_points: usize, margin: f64, seed: u64) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    cloud_to_py(py, &py.detach(|| synthetic::forest(&trees, size, ground_points, margin, seed)))
}

#[pyfunction]
fn synthetic_scan<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, origin: (f64, f64, f64), resolution_deg: f64, max_zenith_deg: f64, max_echoes: usize, echo_separation: f64) -> PyResult<Bound<'py, PyDict>> {
    let cloud = cloud_from_py(xyz, attrs)?;
    let s = py.detach(|| synthetic::scan(&cloud, [origin.0, origin.1, origin.2], resolution_deg, max_zenith_deg, max_echoes, echo_separation));
    shots_to_py(py, &s)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(synthetic_terrain_height, m)?,
        wrap_pyfunction!(synthetic_tree, m)?,
        wrap_pyfunction!(synthetic_leaf_area, m)?,
        wrap_pyfunction!(synthetic_forest, m)?,
        wrap_pyfunction!(synthetic_scan, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
