// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the pulse operations of sylva_rs::shots and
//! sylva_rs::shots::ops: echo bookkeeping, subsets, stacking, beam angles and
//! the misses added from a scan pattern.
#![allow(clippy::type_complexity)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::shots::ops::azimuth_deg;
use sylva_rs::canopy::profile as canopy_profile;
use sylva_rs::Shots;

use crate::canopy_py::pattern_from_py;
use crate::{err, shots_from_py, shots_to_py, xyz_from_py, xyz_to_py};

fn int64<'py>(py: Python<'py>, v: impl IntoIterator<Item = usize>) -> Bound<'py, PyArray1<i64>> {
    v.into_iter().map(|x| x as i64).collect::<Vec<_>>().into_pyarray(py)
}

/// Index of the owning shot of each echo.
#[pyfunction]
fn shots_shot_of_echo<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    Ok(int64(py, shots_from_py(shots)?.shot_of_echo()))
}

/// Rank of each echo within its shot (0 = first).
#[pyfunction]
fn shots_echo_rank<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    Ok(int64(py, shots_from_py(shots)?.echo_rank().into_iter().map(|r| r as usize)))
}

/// Echo coordinates, `origin + direction * range`.
#[pyfunction]
fn shots_echo_xyz<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(xyz_to_py(py, &shots_from_py(shots)?.echo_xyz()))
}

/// The shots where `keep` is true, with all their echoes.
#[pyfunction]
fn shots_subset<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, keep: PyReadonlyArray1<bool>) -> PyResult<Bound<'py, PyDict>> {
    let s = shots_from_py(shots)?;
    let keep = keep.as_array().to_vec();
    if keep.len() != s.n_shots() {
        return Err(PyValueError::new_err(format!("mask has {} values for {} shots", keep.len(), s.n_shots())));
    }
    shots_to_py(py, &s.subset(&keep))
}

/// Stack pulse sets; echo attributes common to all are kept.
#[pyfunction]
fn shots_concatenate<'py>(py: Python<'py>, parts: &Bound<'_, PyList>) -> PyResult<Bound<'py, PyDict>> {
    let parts: Vec<Shots> = parts.iter().map(|p| shots_from_py(p.cast::<PyDict>()?)).collect::<PyResult<_>>()?;
    shots_to_py(py, &Shots::concatenate(&parts.iter().collect::<Vec<_>>()).map_err(err)?)
}

/// Zenith and azimuth (degrees) of each direction.
#[pyfunction]
fn shots_zenith_azimuth<'py>(py: Python<'py>, direction: PyReadonlyArray2<f64>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let d = xyz_from_py(direction)?;
    Ok((canopy_profile::zenith_deg(&d).into_pyarray(py), azimuth_deg(&d).into_pyarray(py)))
}

/// The shots followed by the misses the scan pattern implies, or None if
/// nothing is missing.
#[pyfunction]
#[pyo3(signature = (shots, pattern, pulses_per_line=None, seed=0, shot_stride=1))]
fn shots_fill_missing<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, pattern: &Bound<'_, PyDict>, pulses_per_line: Option<i64>, seed: u64, shot_stride: usize) -> PyResult<Option<Bound<'py, PyDict>>> {
    let s = shots_from_py(shots)?;
    let pattern = pattern_from_py(pattern)?;
    s.fill_missing(&pattern, pulses_per_line, seed, shot_stride.max(1)).map(|f| shots_to_py(py, &f)).transpose()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(shots_shot_of_echo, m)?,
        wrap_pyfunction!(shots_echo_rank, m)?,
        wrap_pyfunction!(shots_echo_xyz, m)?,
        wrap_pyfunction!(shots_subset, m)?,
        wrap_pyfunction!(shots_concatenate, m)?,
        wrap_pyfunction!(shots_zenith_azimuth, m)?,
        wrap_pyfunction!(shots_fill_missing, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
