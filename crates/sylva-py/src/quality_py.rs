// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the stem-noise summary, weighted median and scan ids of
//! sylva.quality.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::quality::summary as qs;

use crate::err;

#[pyfunction]
fn scan_ids_from_origins<'py>(py: Python<'py>, origins: PyReadonlyArray2<f64>, tolerance: f64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a = origins.as_array();
    let flat: Vec<f64> = a.iter().copied().collect();
    Ok(qs::scan_ids_from_origins(&flat, a.ncols(), tolerance).map_err(err)?.into_pyarray(py))
}

#[pyfunction]
fn weighted_median(x: PyReadonlyArray1<f64>, w: PyReadonlyArray1<f64>) -> PyResult<f64> {
    if x.len() != w.len() {
        return Err(PyValueError::new_err("x and w must have the same length"));
    }
    Ok(qs::weighted_median(x.as_slice()?, w.as_slice()?))
}

#[pyfunction]
fn stem_noise_summary<'py>(py: Python<'py>, slice_count: usize, slices: &Bound<'_, PyDict>, scan_slices: &Bound<'_, PyDict>, scans: &Bound<'_, PyDict>, min_scan_slices: f64) -> PyResult<Bound<'py, PyDict>> {
    let get = |d: &Bound<'_, PyDict>, k: &str| -> PyResult<Vec<f64>> {
        let v = d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("no column {k:?}")))?;
        Ok(v.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec())
    };
    let ids = |d: &Bound<'_, PyDict>, k: &str| -> PyResult<Vec<i64>> {
        let v = d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("no column {k:?}")))?;
        Ok(v.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec())
    };
    let (stem, scan) = (ids(slices, "stem")?, ids(scans, "scan")?);
    let (sl_n, sl_sigma, sl_first, sl_tail) = (get(slices, "n_points")?, get(slices, "sigma")?, get(slices, "sigma_first")?, get(slices, "tail_fraction")?);
    let (ss_n, ss_within, ss_local) = (get(scan_slices, "n_points")?, get(scan_slices, "sigma_within")?, get(scan_slices, "sigma_local")?);
    let (sc_n, sc_slices, tx, ty) = (get(scans, "n_points")?, get(scans, "n_slices")?, get(scans, "tx")?, get(scans, "ty")?);
    let t = qs::NoiseTables {
        slice_count,
        slice_stem: &stem,
        slice_n_points: &sl_n,
        slice_sigma: &sl_sigma,
        slice_sigma_first: &sl_first,
        slice_tail_fraction: &sl_tail,
        scan_slice_n_points: &ss_n,
        scan_slice_sigma_within: &ss_within,
        scan_slice_sigma_local: &ss_local,
        scan: &scan,
        scan_n_points: &sc_n,
        scan_n_slices: &sc_slices,
        scan_tx: &tx,
        scan_ty: &ty,
    };
    let s = qs::noise_summary(&t, min_scan_slices).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("n_stems", s.n_stems)?;
    d.set_item("n_slices", s.n_slices)?;
    d.set_item("n_scans", s.n_scans)?;
    if let Some(m) = s.measured {
        d.set_item("sigma_total", m.sigma_total)?;
        d.set_item("sigma_corrected", m.sigma_corrected)?;
        d.set_item("sigma_within", m.sigma_within)?;
        d.set_item("sigma_local", m.sigma_local)?;
        d.set_item("tail_fraction", m.tail_fraction)?;
        d.set_item("n_scans_registered", m.n_scans_registered)?;
        if let Some((rms, max, worst)) = m.registration {
            d.set_item("registration_rms", rms)?;
            d.set_item("registration_max", max)?;
            d.set_item("worst_scan", worst)?;
        }
    }
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(scan_ids_from_origins, m)?, wrap_pyfunction!(weighted_median, m)?, wrap_pyfunction!(stem_noise_summary, m)?] {
        m.add_function(f)?;
    }
    Ok(())
}
