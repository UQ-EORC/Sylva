// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for raster geometry and heights above a raster.
#![allow(clippy::type_complexity)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use sylva_rs::{ground, raster};

use crate::{raster_from_py, xyz_from_py};

#[pyfunction]
fn raster_cell_index<'py>(py: Python<'py>, xmin: f64, ymin: f64, resolution: f64, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let (r, c) = raster::cell_indices(xmin, ymin, resolution, x.as_slice()?, y.as_slice()?);
    Ok((r.into_pyarray(py), c.into_pyarray(py)))
}

#[pyfunction]
fn raster_cell_centers<'py>(py: Python<'py>, nrows: usize, ncols: usize, xmin: f64, ymin: f64, resolution: f64) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (x, y) = raster::cell_centers(nrows, ncols, xmin, ymin, resolution);
    Ok((PyArray1::from_vec(py, x).reshape([nrows, ncols])?, PyArray1::from_vec(py, y).reshape([nrows, ncols])?))
}

/// z minus the raster sampled at each point's x, y.
#[pyfunction]
fn raster_heights_above<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, data: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = xyz_from_py(xyz)?;
    let r = raster_from_py(data, xmin, ymin, resolution);
    Ok(py.detach(|| ground::heights_above(&p, &r)).into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(raster_cell_index, m)?)?;
    m.add_function(wrap_pyfunction!(raster_cell_centers, m)?)?;
    m.add_function(wrap_pyfunction!(raster_heights_above, m)?)?;
    Ok(())
}
