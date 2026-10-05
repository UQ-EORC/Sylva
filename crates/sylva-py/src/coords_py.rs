// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for shifting, rotating and reprojecting coordinates.

use std::path::PathBuf;

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::geo::crs::{self, Crs, Plan};
use sylva_rs::geo::coords;
use sylva_rs::Transform;

use crate::{err, matrix_from_py, matrix_to_py, xyz_from_py, xyz_to_py};

#[pyfunction]
#[pyo3(signature = (axis, angle_deg, about=None))]
fn coords_rotation_matrix<'py>(py: Python<'py>, axis: [f64; 3], angle_deg: f64, about: Option<[f64; 3]>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(matrix_to_py(py, &coords::rotation(axis, angle_deg, about).map_err(err)?))
}

#[pyfunction]
fn coords_translation_matrix<'py>(py: Python<'py>, dx: f64, dy: f64, dz: f64) -> Bound<'py, PyArray2<f64>> {
    matrix_to_py(py, &Transform::translation(dx, dy, dz))
}

/// Points through a 4 x 4 transform, in parallel; pure translations exactly.
#[pyfunction]
fn coords_apply<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, matrix: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut p = xyz_from_py(xyz)?;
    let t = matrix_from_py(Some(matrix))?.expect("a matrix was given");
    py.detach(|| coords::apply_in_place(&mut p, &t));
    Ok(xyz_to_py(py, &p))
}

#[pyfunction]
fn coords_recentre_origin(py: Python<'_>, xyz: PyReadonlyArray2<f64>) -> PyResult<[f64; 3]> {
    let p = xyz_from_py(xyz)?;
    Ok(py.detach(|| coords::recentre_origin(&p)))
}

fn plan_to_py<'py>(py: Python<'py>, plan: &Plan) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("kind", plan.kind.as_str())?;
    d.set_item("exact", plan.exact)?;
    d.set_item("changes_z", plan.changes_z)?;
    d.set_item("note", &plan.note)?;
    Ok(d)
}

#[pyfunction]
fn crs_info<'py>(py: Python<'py>, definition: &str) -> PyResult<Bound<'py, PyDict>> {
    let c = Crs::parse(definition).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("definition", &c.definition)?;
    d.set_item("name", &c.name)?;
    d.set_item("label", c.label())?;
    d.set_item("proj4", &c.proj4)?;
    d.set_item("wkt", c.to_wkt())?;
    d.set_item("epsg", c.epsg)?;
    d.set_item("vertical_epsg", c.vertical_epsg)?;
    d.set_item("datum", &c.datum)?;
    d.set_item("geographic", c.is_geographic().ok())?;
    Ok(d)
}

#[pyfunction]
fn crs_plan<'py>(py: Python<'py>, src: &str, dst: &str) -> PyResult<Bound<'py, PyDict>> {
    let (s, d) = (Crs::parse(src).map_err(err)?, Crs::parse(dst).map_err(err)?);
    plan_to_py(py, &crs::plan(&s, &d).map_err(err)?)
}

#[pyfunction]
fn crs_reproject<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, src: &str, dst: &str) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let (s, d) = (Crs::parse(src).map_err(err)?, Crs::parse(dst).map_err(err)?);
    let mut p = xyz_from_py(xyz)?;
    let plan = py.detach(|| crs::reproject(&mut p, &s, &d)).map_err(err)?;
    Ok((xyz_to_py(py, &p), plan_to_py(py, &plan)?))
}

#[pyfunction]
fn read_las_crs(py: Python<'_>, path: PathBuf) -> PyResult<Option<String>> {
    py.detach(|| crs::read_las_crs(&path)).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coords_rotation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(coords_translation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(coords_apply, m)?)?;
    m.add_function(wrap_pyfunction!(coords_recentre_origin, m)?)?;
    m.add_function(wrap_pyfunction!(crs_info, m)?)?;
    m.add_function(wrap_pyfunction!(crs_plan, m)?)?;
    m.add_function(wrap_pyfunction!(crs_reproject, m)?)?;
    m.add_function(wrap_pyfunction!(read_las_crs, m)?)?;
    Ok(())
}
