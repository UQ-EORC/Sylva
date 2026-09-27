// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::change::points and sylva_rs::change::voxels:
//! point distances (C2C, M3C2), rasters of difference and voxel occupancy
//! change.

use std::borrow::Cow;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::change::points::{self, CellValue, M3c2Params, Orientation};
use sylva_rs::change::voxels::{self, OccupancyParams};
use sylva_rs::Point;

use crate::voxels_py::PyRayVoxels;
use crate::{err, raster_from_py, raster_to_py};

/// The rows of an `(N, 3)` float64 array as points, without copying when the
/// array is C-contiguous.
fn points<'a>(name: &str, xyz: &'a PyReadonlyArray2<'_, f64>) -> PyResult<Cow<'a, [Point]>> {
    if xyz.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!("{name} must have shape (N, 3), got (N, {})", xyz.shape()[1])));
    }
    if let Ok(flat) = xyz.as_slice() {
        // SAFETY: `[f64; 3]` has the size of three f64 and the alignment of
        // one, and the slice holds whole rows of three.
        let rows = unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const Point, flat.len() / 3) };
        return Ok(Cow::Borrowed(rows));
    }
    Ok(Cow::Owned(xyz.as_array().rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect()))
}

fn flat_points<'py>(py: Python<'py>, v: Vec<Point>) -> PyResult<Bound<'py, PyAny>> {
    let n = v.len();
    let flat: Vec<f64> = v.into_iter().flatten().collect();
    Ok(PyArray1::from_vec(py, flat).reshape([n, 3])?.into_any())
}

#[pyfunction]
#[pyo3(signature = (reference, compared, max_distance=None))]
fn change_c2c<'py>(py: Python<'py>, reference: PyReadonlyArray2<f64>, compared: PyReadonlyArray2<f64>, max_distance: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (a, b) = (points("reference", &reference)?, points("compared", &compared)?);
    Ok(py.detach(|| points::c2c(&a, &b, max_distance)).map_err(err)?.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (a, b, core, normals, normal_scale, projection_scale, max_depth, registration_sigma, min_points, orient_direction=None, orient_towards=None))]
#[allow(clippy::too_many_arguments)]
fn change_m3c2<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, core: PyReadonlyArray2<f64>, normals: Option<PyReadonlyArray2<f64>>, normal_scale: f64, projection_scale: f64, max_depth: f64, registration_sigma: f64, min_points: usize, orient_direction: Option<[f64; 3]>, orient_towards: Option<[f64; 3]>) -> PyResult<Bound<'py, PyDict>> {
    let (pa, pb, pc) = (points("a", &a)?, points("b", &b)?, points("core_points", &core)?);
    let nv = match &normals {
        Some(n) => Some(points("normals", n)?),
        None => None,
    };
    let orientation = match (orient_direction, orient_towards) {
        (_, Some(t)) => Orientation::Towards(t),
        (Some(d), None) => Orientation::Direction(d),
        (None, None) => Orientation::Direction([0.0, 0.0, 1.0]),
    };
    let params = M3c2Params { normal_scale, projection_scale, max_depth, registration_sigma, min_points, orientation };
    let r = py.detach(|| points::m3c2(&pa, &pb, &pc, nv.as_deref(), &params)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("distance", r.distance.into_pyarray(py))?;
    d.set_item("lod", r.lod.into_pyarray(py))?;
    d.set_item("significant", r.significant.into_pyarray(py))?;
    d.set_item("normal", flat_points(py, r.normal)?)?;
    d.set_item("n_a", r.n_a.into_pyarray(py))?;
    d.set_item("n_b", r.n_b.into_pyarray(py))?;
    d.set_item("spread_a", r.spread_a.into_pyarray(py))?;
    d.set_item("spread_b", r.spread_b.into_pyarray(py))?;
    Ok(d)
}

/// A scalar or a `(data, xmin, ymin, resolution)` raster.
fn cell_value(v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<CellValue>> {
    let Some(v) = v else { return Ok(None) };
    if let Ok(x) = v.extract::<f64>() {
        return Ok(Some(CellValue::Scalar(x)));
    }
    let (data, xmin, ymin, res): (PyReadonlyArray2<f64>, f64, f64, f64) = v.extract()?;
    Ok(Some(CellValue::Grid(raster_from_py(data, xmin, ymin, res))))
}

#[pyfunction]
#[pyo3(signature = (a, b, min_detectable=None, sigma_a=None, sigma_b=None))]
fn change_dod<'py>(py: Python<'py>, a: (PyReadonlyArray2<f64>, f64, f64, f64), b: (PyReadonlyArray2<f64>, f64, f64, f64), min_detectable: Option<Bound<'py, PyAny>>, sigma_a: Option<Bound<'py, PyAny>>, sigma_b: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyDict>> {
    let ra = raster_from_py(a.0, a.1, a.2, a.3);
    let rb = raster_from_py(b.0, b.1, b.2, b.3);
    let (m, sa, sb) = (cell_value(min_detectable.as_ref())?, cell_value(sigma_a.as_ref())?, cell_value(sigma_b.as_ref())?);
    let r = py.detach(|| points::dod(&ra, &rb, m.as_ref(), sa.as_ref(), sb.as_ref())).map_err(err)?;
    let d = PyDict::new(py);
    let (nr, nc) = (r.difference.nrows, r.difference.ncols);
    d.set_item("difference", raster_to_py(py, &r.difference)?)?;
    d.set_item("lod", raster_to_py(py, &r.lod)?)?;
    d.set_item("significant", PyArray1::from_vec(py, r.significant).reshape([nr, nc])?)?;
    d.set_item("volume_gained", r.volume_gained)?;
    d.set_item("volume_lost", r.volume_lost)?;
    d.set_item("net_volume", r.net_volume)?;
    d.set_item("area_changed", r.area_changed)?;
    d.set_item("area_compared", r.area_compared)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (a, b, pad="pad_fpl", min_pulses=10, min_hits=1, alpha=0.05))]
fn change_occupancy<'py>(py: Python<'py>, a: PyRef<'py, PyRayVoxels>, b: PyRef<'py, PyRayVoxels>, pad: &str, min_pulses: u32, min_hits: u32, alpha: f64) -> PyResult<Bound<'py, PyDict>> {
    let params = OccupancyParams { min_pulses, min_hits, alpha };
    let (ga, gb) = (&a.inner, &b.inner);
    let r = py.detach(|| voxels::occupancy(ga, gb, pad, &params)).map_err(err)?;
    let s = ga.shape;
    let shape = [s[2], s[1], s[0]];
    let d = PyDict::new(py);
    d.set_item("class", PyArray1::from_vec(py, r.class).reshape(shape)?)?;
    d.set_item("pad_a", PyArray1::from_vec(py, r.pad_a).reshape(shape)?)?;
    d.set_item("pad_b", PyArray1::from_vec(py, r.pad_b).reshape(shape)?)?;
    d.set_item("pad_change", PyArray1::from_vec(py, r.pad_change).reshape(shape)?)?;
    let l = PyDict::new(py);
    l.set_item("z", r.layers.z.into_pyarray(py))?;
    for (name, c) in voxels::CLASS_NAMES.iter().zip(r.layers.counts) {
        l.set_item(*name, c.into_pyarray(py))?;
    }
    l.set_item("n_compared", r.layers.n_compared.into_pyarray(py))?;
    l.set_item("pad_a", r.layers.pad_a.into_pyarray(py))?;
    l.set_item("pad_b", r.layers.pad_b.into_pyarray(py))?;
    l.set_item("pad_change", r.layers.pad_change.into_pyarray(py))?;
    d.set_item("layers", l)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(change_c2c, m)?)?;
    m.add_function(wrap_pyfunction!(change_m3c2, m)?)?;
    m.add_function(wrap_pyfunction!(change_dod, m)?)?;
    m.add_function(wrap_pyfunction!(change_occupancy, m)?)?;
    Ok(())
}
