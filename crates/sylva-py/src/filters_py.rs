// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the crop and range masks of sylva_rs::filters.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use sylva_rs::filters;

use crate::xyz_from_py;

#[pyfunction]
fn crop_box_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, min_xyz: [f64; 3], max_xyz: [f64; 3]) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    Ok(filters::crop_box_mask(&p, min_xyz, max_xyz).into_pyarray(py))
}

#[pyfunction]
fn crop_cylinder_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, cx: f64, cy: f64, radius: f64, zmin: f64, zmax: f64) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    Ok(filters::crop_cylinder_mask(&p, cx, cy, radius, zmin, zmax).into_pyarray(py))
}

#[pyfunction]
fn range_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, origin: [f64; 3], min_range: f64, max_range: f64) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    Ok(filters::range_mask(&p, origin, min_range, max_range).into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crop_box_mask, m)?)?;
    m.add_function(wrap_pyfunction!(crop_cylinder_mask, m)?)?;
    m.add_function(wrap_pyfunction!(range_mask, m)?)?;
    Ok(())
}
