// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::riscan: RiSCAN PRO projects, export settings and
//! RiSCAN's import filter.
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::riscan;

use crate::{err, xyz_from_py};

/// Core errors as Python raises them: `project.rsp` that is not XML as
/// ElementTree's ParseError, an empty percentile as IndexError.
fn riscan_err(py: Python<'_>, e: sylva_rs::Error) -> PyErr {
    if let sylva_rs::Error::File { msg, .. } = &e {
        if let Some(m) = msg.strip_prefix("XML: ") {
            if let Ok(cls) = py.import("xml.etree.ElementTree").and_then(|m| m.getattr("ParseError")) {
                if let Ok(exc) = cls.call1((m.to_string(),)) {
                    return PyErr::from_value(exc);
                }
            }
        }
    }
    if let sylva_rs::Error::Invalid(m) = &e {
        if m.starts_with("index -1 is out of bounds") {
            return PyIndexError::new_err(m.clone());
        }
    }
    err(e)
}

fn mat<'py>(py: Python<'py>, v: &[f64], n: usize) -> Bound<'py, PyArray2<f64>> {
    PyArray1::from_vec(py, v.to_vec()).reshape([n, n]).expect("square")
}

fn f32_rows(xyz: PyReadonlyArray2<f32>) -> PyResult<Vec<[f32; 3]>> {
    let a = xyz.as_array();
    if a.ncols() != 3 {
        return Err(PyValueError::new_err(format!("xyz must have shape (N, 3), got (N, {})", a.ncols())));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

/// A parsed project as a dict of plain values; `sylva.riscan` builds its
/// dataclasses from it.
#[pyfunction]
fn riscan_read_project<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let p = riscan::read_project(&path).map_err(|e| riscan_err(py, e))?;
    let positions = PyList::empty(py);
    for q in &p.positions {
        let d = PyDict::new(py);
        d.set_item("name", &q.name)?;
        d.set_item("scans", q.scans.iter().map(|s| s.to_string_lossy().into_owned()).collect::<Vec<_>>())?;
        d.set_item("sop", q.sop.map(|m| mat(py, &m, 4)))?;
        d.set_item("instrument", &q.instrument)?;
        let pattern = match q.pattern {
            None => None,
            Some(s) => {
                let d = PyDict::new(py);
                d.set_item("theta_start", s.theta_start)?;
                d.set_item("theta_delta", s.theta_delta)?;
                d.set_item("theta_count", s.theta_count)?;
                d.set_item("phi_start", s.phi_start)?;
                d.set_item("phi_delta", s.phi_delta)?;
                d.set_item("phi_count", s.phi_count)?;
                Some(d)
            }
        };
        d.set_item("pattern", pattern)?;
        d.set_item("tiepoints", q.tiepoints.as_ref().map(|t| t.to_string_lossy().into_owned()))?;
        d.set_item("gnss", q.gnss.map(|g| (g[0], g[1], g[2])))?;
        d.set_item("attitude", q.attitude.map(|a| mat(py, &a, 3)))?;
        positions.append(d)?;
    }
    let d = PyDict::new(py);
    d.set_item("name", &p.name)?;
    d.set_item("pop", p.pop.map(|m| mat(py, &m, 4)))?;
    d.set_item("positions", positions)?;
    Ok(d)
}

#[pyfunction]
fn riscan_rotation_zyx<'py>(py: Python<'py>, roll: f64, pitch: f64, yaw: f64) -> Bound<'py, PyArray2<f64>> {
    mat(py, &riscan::rotation_zyx(roll, pitch, yaw), 4)
}

#[pyfunction]
#[pyo3(signature = (text=None))]
fn riscan_parse_matrix<'py>(py: Python<'py>, text: Option<&str>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    Ok(riscan::parse_matrix(text).map_err(err)?.map(|m| mat(py, &m, 4)))
}

#[pyfunction]
fn riscan_gnss_to_local<'py>(py: Python<'py>, coordinates: Vec<Option<(f64, f64, f64)>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let c: Vec<Option<[f64; 3]>> = coordinates.into_iter().map(|c| c.map(|(a, b, h)| [a, b, h])).collect();
    let out = riscan::gnss_to_local(&c);
    PyArray1::from_vec(py, out.iter().flatten().copied().collect()).reshape([out.len(), 3])
}

#[pyfunction]
fn riscan_read_export_settings(path: PathBuf) -> PyResult<Vec<(String, f64, f64)>> {
    riscan::read_export_settings(&path).map_err(err)
}

#[pyfunction]
fn riscan_export_settings_mask<'py>(py: Python<'py>, settings: Vec<(String, f64, f64)>, xyz: PyReadonlyArray2<f64>, attributes: HashMap<String, PyReadonlyArray1<f64>>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    let attrs: HashMap<String, Vec<f64>> = attributes.into_iter().map(|(k, v)| (k, v.as_array().to_vec())).collect();
    let keep = riscan::export_settings_mask(&settings, &p, |n| attrs.get(n).map(|v| v.as_slice())).map_err(err)?;
    Ok(keep.into_pyarray(py))
}

#[pyfunction]
fn riscan_angular_steps(py: Python<'_>, xyz: PyReadonlyArray2<f32>, sample: usize) -> PyResult<(f64, f64)> {
    let p = f32_rows(xyz)?;
    py.detach(|| riscan::angular_steps(&p, sample)).map_err(|e| riscan_err(py, e))
}

#[pyfunction]
#[pyo3(signature = (xyz, amplitude, mode, min_range, window_steps, window_range, min_neighbours, weak_db, steps=None))]
fn riscan_like_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f32>, amplitude: PyReadonlyArray1<f32>, mode: &str, min_range: f64, window_steps: f64, window_range: f64, min_neighbours: usize, weak_db: f64, steps: Option<(f64, f64)>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = f32_rows(xyz)?;
    let a = amplitude.as_array().to_vec();
    let legacy = riscan::LegacyFilter { window_steps, window_range, min_neighbours, weak_db, steps };
    let keep = py.detach(|| riscan::riscan_like_mask(&p, &a, mode, min_range, &legacy)).map_err(|e| riscan_err(py, e))?;
    Ok(keep.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(riscan_read_project, m)?,
        wrap_pyfunction!(riscan_rotation_zyx, m)?,
        wrap_pyfunction!(riscan_parse_matrix, m)?,
        wrap_pyfunction!(riscan_gnss_to_local, m)?,
        wrap_pyfunction!(riscan_read_export_settings, m)?,
        wrap_pyfunction!(riscan_export_settings_mask, m)?,
        wrap_pyfunction!(riscan_angular_steps, m)?,
        wrap_pyfunction!(riscan_like_mask, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
