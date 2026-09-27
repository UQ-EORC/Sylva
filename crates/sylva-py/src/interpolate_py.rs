// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::interpolate.

use std::borrow::Cow;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::interpolate::{self, GridMethod, GridParams, SampleMethod};
use sylva_rs::Point;

use crate::{attr_from_py, err, raster_from_py, raster_to_py};

/// The rows of an `(N, 3)` float64 array as points, without copying when the
/// array is C-contiguous (as `PointCloud.xyz` always is).
fn points<'a>(xyz: &'a PyReadonlyArray2<'_, f64>) -> PyResult<Cow<'a, [Point]>> {
    if xyz.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!("xyz must have shape (N, 3), got (N, {})", xyz.shape()[1])));
    }
    if let Ok(flat) = xyz.as_slice() {
        // SAFETY: `[f64; 3]` has the size of three f64 and the alignment of
        // one, and the slice holds whole rows of three.
        let rows = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const Point, flat.len() / 3) };
        return Ok(Cow::Borrowed(rows));
    }
    Ok(Cow::Owned(xyz.as_array().rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect()))
}

#[pyfunction]
#[pyo3(signature = (source_xyz, target_xyz, max_distance=None))]
fn interp_nearest_indices<'py>(py: Python<'py>, source_xyz: PyReadonlyArray2<f64>, target_xyz: PyReadonlyArray2<f64>, max_distance: Option<f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (s, t) = (points(&source_xyz)?, points(&target_xyz)?);
    Ok(py.detach(|| interpolate::nearest_indices(&s, &t, max_distance)).map_err(err)?.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (source_xyz, target_xyz, labels, k=8, max_distance=None))]
fn interp_majority_indices<'py>(py: Python<'py>, source_xyz: PyReadonlyArray2<f64>, target_xyz: PyReadonlyArray2<f64>, labels: Vec<Bound<'py, PyAny>>, k: usize, max_distance: Option<f64>) -> PyResult<Vec<Bound<'py, PyArray1<i64>>>> {
    let (s, t) = (points(&source_xyz)?, points(&target_xyz)?);
    let keys = labels.iter().map(|l| attr_from_py(l).map(|a| interpolate::label_keys(&a))).collect::<PyResult<Vec<_>>>()?;
    let out = py.detach(|| interpolate::majority_indices(&s, &t, &keys, k, max_distance)).map_err(err)?;
    Ok(out.into_iter().map(|v| v.into_pyarray(py)).collect())
}

#[pyfunction]
#[pyo3(signature = (source_xyz, target_xyz, values, k=8, power=2.0, max_distance=None))]
fn interp_idw<'py>(py: Python<'py>, source_xyz: PyReadonlyArray2<f64>, target_xyz: PyReadonlyArray2<f64>, values: Vec<PyReadonlyArray1<f64>>, k: usize, power: f64, max_distance: Option<f64>) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
    let (s, t) = (points(&source_xyz)?, points(&target_xyz)?);
    let vals: Vec<Vec<f64>> = values.iter().map(|v| v.as_array().to_vec()).collect();
    let out = py.detach(|| interpolate::idw_values(&s, &t, &vals, k, power, max_distance)).map_err(err)?;
    Ok(out.into_iter().map(|v| v.into_pyarray(py)).collect())
}

#[pyfunction]
#[pyo3(signature = (xyz, values, resolution, bounds=None, method="idw", power=2.0, k=12, max_distance=None))]
#[allow(clippy::too_many_arguments)]
fn interp_grid<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, values: PyReadonlyArray1<f64>, resolution: f64, bounds: Option<(f64, f64, f64, f64)>, method: &str, power: f64, k: usize, max_distance: Option<f64>) -> PyResult<Bound<'py, PyDict>> {
    let p = points(&xyz)?;
    let v = values.as_array().to_vec();
    let params = GridParams { method: GridMethod::parse(method).map_err(err)?, power, k, max_distance };
    let r = py.detach(|| interpolate::grid(&p, &v, resolution, bounds, &params)).map_err(err)?;
    raster_to_py(py, &r)
}

#[pyfunction]
#[pyo3(signature = (data, xmin, ymin, resolution, xyz, method="bilinear"))]
fn interp_sample_raster<'py>(py: Python<'py>, data: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64, xyz: PyReadonlyArray2<f64>, method: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = SampleMethod::parse(method).map_err(err)?;
    let r = raster_from_py(data, xmin, ymin, resolution);
    let p = points(&xyz)?;
    Ok(py.detach(|| interpolate::sample_raster(&r, &p, m)).into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interp_nearest_indices, m)?)?;
    m.add_function(wrap_pyfunction!(interp_majority_indices, m)?)?;
    m.add_function(wrap_pyfunction!(interp_idw, m)?)?;
    m.add_function(wrap_pyfunction!(interp_grid, m)?)?;
    m.add_function(wrap_pyfunction!(interp_sample_raster, m)?)?;
    Ok(())
}
