// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for pruning tree lists, basal area and buttress detection.
#![allow(clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::qsm::buttress_detect as bd;
use sylva_rs::trees::prune as tp;

use crate::{err, tree_to_py, trees_from_py, xyz_from_py};

/// Survivors as `(index of the input tree, tree dict)` and the new labels.
#[pyfunction]
#[pyo3(signature = (trees_list, labels, min_height=3.0, merge_radius=0.2, max_dbh=None, min_quality_short=0.0, short_slices=4, min_slenderness=0.0, slender_min_dbh=0.2))]
#[allow(clippy::too_many_arguments)]
fn prune_trees<'py>(py: Python<'py>, trees_list: &Bound<'_, PyList>, labels: PyReadonlyArray1<i64>, min_height: f64, merge_radius: f64, max_dbh: Option<f64>, min_quality_short: f64, short_slices: i64, min_slenderness: f64, slender_min_dbh: f64) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyArray1<i64>>)> {
    let t = trees_from_py(trees_list)?;
    let p = tp::PruneParams { min_height, merge_radius, max_dbh, min_quality_short, short_slices, min_slenderness, slender_min_dbh };
    let (kept, lab) = tp::prune_trees(&t, &labels.as_array().to_vec(), &p).map_err(err)?;
    let list = PyList::empty(py);
    for (i, tr) in &kept {
        list.append((*i, tree_to_py(py, tr)?))?;
    }
    Ok((list, lab.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (dbh, area, min_dbh=0.0))]
fn basal_area(dbh: PyReadonlyArray1<f64>, area: f64, min_dbh: f64) -> PyResult<f64> {
    tp::basal_area(dbh.as_slice()?, area, min_dbh).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (persistence, level=0.6, dip=0.2))]
fn count_ridges(persistence: PyReadonlyArray1<f64>, level: f64, dip: f64) -> PyResult<usize> {
    Ok(bd::count_ridges(persistence.as_slice()?, level, dip))
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, base_xy=None, max_radius=4.0, slice_height=0.1, max_height=6.0, low=1.0, bins=36, max_circle_fit=0.55, min_ridges=2, bark_only=true, voxel=0.02))]
fn detect_buttress<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, base_xy: Option<(f64, f64)>, max_radius: f64, slice_height: f64, max_height: f64, low: f64, bins: usize, max_circle_fit: f64, min_ridges: usize, bark_only: bool, voxel: f64) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let params = bd::ButtressParams { max_radius, slice_height, max_height, low, bins, max_circle_fit, min_ridges, bark_only, voxel };
    let b = py.detach(|| bd::detect_buttress(&p, &h, base_xy.map(|(x, y)| [x, y]), &params)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("buttressed", b.buttressed)?;
    d.set_item("base_circle_fit", b.base_circle_fit)?;
    d.set_item("stem_circle_fit", b.stem_circle_fit)?;
    d.set_item("stem_radius", b.stem_radius)?;
    d.set_item("ridges", b.ridges)?;
    d.set_item("ridge_share", b.ridge_share)?;
    d.set_item("spread", b.spread)?;
    d.set_item("top", b.top)?;
    d.set_item("centre", b.centre.to_vec().into_pyarray(py))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(prune_trees, m)?, wrap_pyfunction!(basal_area, m)?, wrap_pyfunction!(count_ridges, m)?, wrap_pyfunction!(detect_buttress, m)?] {
        m.add_function(f)?;
    }
    Ok(())
}
