// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva.change.qsm: comparing the QSMs of two epochs. Models
//! cross as `(n, 12)` cylinder arrays, a ray-traced grid as
//! `(origin, voxel_size, state)` with `state` the `(nz, ny, nx)` uint8
//! states, and the settings as a dict keyed by the `CompareParams` fields.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::change::qsm::{self as cq, BranchChange, CompareParams, Fate, PlotRow, PlotTree, QsmChange, StateGrid};
use sylva_rs::qsm::Qsm;

use crate::{err, qsm_from_rows};

type Grid<'py> = (Vec<f64>, f64, PyReadonlyArray3<'py, u8>);

fn grid_from_py(g: Option<Grid<'_>>) -> PyResult<Option<StateGrid>> {
    let Some((origin, voxel_size, state)) = g else { return Ok(None) };
    if origin.len() != 3 {
        return Err(PyValueError::new_err("grid origin must have three values"));
    }
    let a = state.as_array();
    let (nz, ny, nx) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let flat: Vec<u8> = a.iter().copied().collect();
    StateGrid::new([origin[0], origin[1], origin[2]], voxel_size, [nx, ny, nz], flat).map(Some).map_err(err)
}

fn params_from_py(d: &Bound<'_, PyDict>) -> PyResult<CompareParams> {
    let get = |k: &str| -> PyResult<f64> {
        d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("missing setting {k}")))?.extract::<f64>()
    };
    let p = CompareParams {
        height_step: get("height_step")?,
        max_base_distance: get("max_base_distance")?,
        max_angle: get("max_angle")?,
        parent_penalty: get("parent_penalty")?,
        direction_reach: get("direction_reach")?,
        min_measured: get("min_measured")?,
        min_fits: d.get_item("min_fits")?.ok_or_else(|| PyValueError::new_err("missing setting min_fits"))?.extract::<usize>()?,
        radius_sigma: get("radius_sigma")?,
        clip: get("clip")?,
        min_observed: get("min_observed")?,
        max_filled: get("max_filled")?,
        top_band: get("top_band")?,
        crown_branch_length: get("crown_branch_length")?,
        crown_slice: get("crown_slice")?,
    };
    p.check().map_err(err)?;
    Ok(p)
}

fn unmatched_to_py<'py>(py: Python<'py>, v: &[BranchChange]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    macro_rules! col {
        ($name:literal, $f:expr) => {
            d.set_item($name, v.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    col!("id", |x| x.id as i64);
    col!("order", |x| x.order as i64);
    col!("parent", |x| x.parent);
    col!("base_x", |x| x.base[0]);
    col!("base_y", |x| x.base[1]);
    col!("base_z", |x| x.base[2]);
    col!("length", |x| x.length);
    col!("volume", |x| x.volume);
    col!("measured", |x| x.measured);
    col!("observed_share", |x| x.observed_share);
    col!("filled_share", |x| x.filled_share);
    col!("volume_sigma", |x| x.volume_sigma);
    col!("trusted", |x| x.trusted);
    d.set_item("status", PyList::new(py, v.iter().map(|x| x.status.as_str()))?)?;
    Ok(d)
}

fn change_to_py<'py>(py: Python<'py>, c: &QsmChange) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("base_z", c.base_z)?;
    d.set_item("height_a", c.height_a)?;
    d.set_item("height_b", c.height_b)?;
    d.set_item("height_trusted", c.height_trusted)?;
    d.set_item("dbh_a", c.dbh_a)?;
    d.set_item("dbh_b", c.dbh_b)?;
    d.set_item("dbh_trusted", c.dbh_trusted)?;
    d.set_item("crown_area_a", c.crown_area_a)?;
    d.set_item("crown_area_b", c.crown_area_b)?;
    d.set_item("crown_volume_a", c.crown_volume_a)?;
    d.set_item("crown_volume_b", c.crown_volume_b)?;
    d.set_item("crown_trusted", c.crown_trusted)?;
    d.set_item("volume_a", c.volume_a)?;
    d.set_item("volume_b", c.volume_b)?;
    d.set_item("measured_volume_a", c.measured_volume_a)?;
    d.set_item("measured_volume_b", c.measured_volume_b)?;
    d.set_item("trusted_change", c.trusted_change)?;
    d.set_item("untrusted_change", c.untrusted_change)?;
    d.set_item("trusted_sigma", c.trusted_sigma)?;
    d.set_item("taper_increment", c.taper_increment)?;
    d.set_item("taper_sigma", c.taper_sigma)?;
    d.set_item("n_taper_bins", c.n_taper_bins)?;

    let o = PyDict::new(py);
    o.set_item("order", c.orders.iter().map(|x| x.order as i64).collect::<Vec<_>>().into_pyarray(py))?;
    o.set_item("volume_a", c.orders.iter().map(|x| x.volume_a).collect::<Vec<_>>().into_pyarray(py))?;
    o.set_item("volume_b", c.orders.iter().map(|x| x.volume_b).collect::<Vec<_>>().into_pyarray(py))?;
    o.set_item("trusted_change", c.orders.iter().map(|x| x.trusted_change).collect::<Vec<_>>().into_pyarray(py))?;
    o.set_item("untrusted_change", c.orders.iter().map(|x| x.untrusted_change).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("orders", o)?;

    let t = PyDict::new(py);
    macro_rules! tcol {
        ($name:literal, $f:expr) => {
            t.set_item($name, c.taper.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    tcol!("z0", |x| x.z0);
    tcol!("z1", |x| x.z1);
    tcol!("radius_a", |x| x.radius_a);
    tcol!("radius_b", |x| x.radius_b);
    tcol!("increment", |x| x.increment);
    tcol!("sigma", |x| x.sigma);
    tcol!("n_fits_a", |x| x.n_fits_a as i64);
    tcol!("n_fits_b", |x| x.n_fits_b as i64);
    tcol!("measured_a", |x| x.measured_a);
    tcol!("measured_b", |x| x.measured_b);
    tcol!("volume_a", |x| x.volume_a);
    tcol!("volume_b", |x| x.volume_b);
    tcol!("fitted", |x| x.fitted);
    tcol!("trusted", |x| x.trusted);
    d.set_item("taper", t)?;

    let m = PyDict::new(py);
    macro_rules! mcol {
        ($name:literal, $f:expr) => {
            m.set_item($name, c.matched.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    mcol!("id_a", |x| x.id_a as i64);
    mcol!("id_b", |x| x.id_b as i64);
    mcol!("order_a", |x| x.order_a as i64);
    mcol!("order_b", |x| x.order_b as i64);
    mcol!("base_distance", |x| x.base_distance);
    mcol!("angle", |x| x.angle);
    mcol!("parent_consistent", |x| x.parent_consistent);
    mcol!("length_a", |x| x.length_a);
    mcol!("length_b", |x| x.length_b);
    mcol!("volume_a", |x| x.volume_a);
    mcol!("volume_b", |x| x.volume_b);
    mcol!("mean_radius_a", |x| x.mean_radius_a);
    mcol!("mean_radius_b", |x| x.mean_radius_b);
    mcol!("tip_shift", |x| x.tip_shift);
    mcol!("measured_a", |x| x.measured_a);
    mcol!("measured_b", |x| x.measured_b);
    mcol!("volume_sigma", |x| x.volume_sigma);
    mcol!("trusted", |x| x.trusted);
    d.set_item("matched", m)?;
    d.set_item("lost", unmatched_to_py(py, &c.lost)?)?;
    d.set_item("new", unmatched_to_py(py, &c.new)?)?;
    Ok(d)
}

/// Compares two models of one tree.
#[pyfunction]
#[pyo3(signature = (cylinders_a, cylinders_b, params, grid_a=None, grid_b=None))]
fn change_compare_qsms<'py>(py: Python<'py>, cylinders_a: PyReadonlyArray2<f64>, cylinders_b: PyReadonlyArray2<f64>, params: &Bound<'_, PyDict>, grid_a: Option<Grid<'_>>, grid_b: Option<Grid<'_>>) -> PyResult<Bound<'py, PyDict>> {
    let a = qsm_from_rows(cylinders_a)?;
    let b = qsm_from_rows(cylinders_b)?;
    let p = params_from_py(params)?;
    let (ga, gb) = (grid_from_py(grid_a)?, grid_from_py(grid_b)?);
    let c = py.detach(|| cq::compare_qsms(&a, &b, ga.as_ref(), gb.as_ref(), &p)).map_err(err)?;
    change_to_py(py, &c)
}

fn fate_from_str(s: &str) -> PyResult<Fate> {
    match s {
        "survivor" => Ok(Fate::Survivor),
        "death" => Ok(Fate::Death),
        "recruit" => Ok(Fate::Recruit),
        _ => Err(PyValueError::new_err(format!("unknown fate {s:?}; use survivor, death or recruit"))),
    }
}

fn row_to_py<'py>(py: Python<'py>, r: &PlotRow) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("fate", r.fate.as_str())?;
    d.set_item("tree_id_a", r.id_a)?;
    d.set_item("tree_id_b", r.id_b)?;
    d.set_item("volume_a_m3", r.volume_a)?;
    d.set_item("volume_b_m3", r.volume_b)?;
    d.set_item("change_m3", r.change)?;
    d.set_item("trusted_change_m3", r.trusted_change)?;
    d.set_item("untrusted_change_m3", r.untrusted_change)?;
    d.set_item("trusted_sigma_m3", r.trusted_sigma)?;
    d.set_item("stem_change_m3", r.stem_change)?;
    d.set_item("branch_change_m3", r.branch_change)?;
    d.set_item("taper_increment_m", r.taper_increment)?;
    d.set_item("taper_sigma_m", r.taper_sigma)?;
    d.set_item("height_a_m", r.height_a)?;
    d.set_item("height_b_m", r.height_b)?;
    d.set_item("n_matched", r.n_matched)?;
    d.set_item("n_lost", r.n_lost)?;
    d.set_item("n_unobserved", r.n_unobserved)?;
    d.set_item("n_present", r.n_present)?;
    d.set_item("n_new", r.n_new)?;
    d.set_item("note", &r.note)?;
    Ok(d)
}

fn row_from_py(d: &Bound<'_, PyDict>) -> PyResult<PlotRow> {
    let item = |k: &str| -> PyResult<Bound<'_, PyAny>> { d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("row has no {k}"))) };
    let f = |k: &str| -> PyResult<f64> { item(k)?.extract::<f64>() };
    let n = |k: &str| -> PyResult<usize> { item(k)?.extract::<usize>() };
    Ok(PlotRow {
        fate: fate_from_str(&item("fate")?.extract::<String>()?)?,
        id_a: item("tree_id_a")?.extract()?,
        id_b: item("tree_id_b")?.extract()?,
        volume_a: f("volume_a_m3")?,
        volume_b: f("volume_b_m3")?,
        change: f("change_m3")?,
        trusted_change: f("trusted_change_m3")?,
        untrusted_change: f("untrusted_change_m3")?,
        trusted_sigma: f("trusted_sigma_m3")?,
        stem_change: f("stem_change_m3")?,
        branch_change: f("branch_change_m3")?,
        taper_increment: f("taper_increment_m")?,
        taper_sigma: f("taper_sigma_m")?,
        height_a: f("height_a_m")?,
        height_b: f("height_b_m")?,
        n_matched: n("n_matched")?,
        n_lost: n("n_lost")?,
        n_unobserved: n("n_unobserved")?,
        n_present: n("n_present")?,
        n_new: n("n_new")?,
        note: item("note")?.extract()?,
    })
}

type Models = (Fate, Option<i64>, Option<i64>, Option<Qsm>, Option<Qsm>);

type TreeEntry<'py> = (String, Option<i64>, Option<i64>, Option<PyReadonlyArray2<'py, f64>>, Option<PyReadonlyArray2<'py, f64>>);

/// Compares the models of every tree of a plot: `trees` is a list of
/// `(fate, id_a, id_b, cylinders_a, cylinders_b)`. Returns the rows, the
/// full comparison of each compared survivor (None elsewhere) and the totals.
#[pyfunction]
#[pyo3(signature = (trees, params, grid_a=None, grid_b=None))]
fn change_compare_plot<'py>(py: Python<'py>, trees: Vec<TreeEntry<'_>>, params: &Bound<'_, PyDict>, grid_a: Option<Grid<'_>>, grid_b: Option<Grid<'_>>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyDict>)> {
    let p = params_from_py(params)?;
    let (ga, gb) = (grid_from_py(grid_a)?, grid_from_py(grid_b)?);
    let mut models: Vec<Models> = Vec::with_capacity(trees.len());
    for (fate, id_a, id_b, a, b) in trees {
        models.push((fate_from_str(&fate)?, id_a, id_b, a.map(qsm_from_rows).transpose()?, b.map(qsm_from_rows).transpose()?));
    }
    let plot: Vec<PlotTree> = models.iter().map(|(fate, id_a, id_b, a, b)| PlotTree { fate: *fate, id_a: *id_a, id_b: *id_b, a: a.as_ref(), b: b.as_ref() }).collect();
    let r = py.detach(|| cq::compare_plot(&plot, ga.as_ref(), gb.as_ref(), &p)).map_err(err)?;
    let rows = PyList::empty(py);
    for row in &r.rows {
        rows.append(row_to_py(py, row)?)?;
    }
    let changes = PyList::empty(py);
    for c in &r.changes {
        match c {
            Some(c) => changes.append(change_to_py(py, c)?)?,
            None => changes.append(py.None())?,
        }
    }
    let t = &r.totals;
    let d = PyDict::new(py);
    d.set_item("n_survivors", t.n_survivors)?;
    d.set_item("n_deaths", t.n_deaths)?;
    d.set_item("n_recruits", t.n_recruits)?;
    d.set_item("n_unmodelled", t.n_unmodelled)?;
    d.set_item("growth_m3", t.growth)?;
    d.set_item("growth_trusted_m3", t.growth_trusted)?;
    d.set_item("growth_sigma_m3", t.growth_sigma)?;
    d.set_item("mortality_m3", t.mortality)?;
    d.set_item("recruitment_m3", t.recruitment)?;
    d.set_item("net_m3", t.net)?;
    d.set_item("net_trusted_m3", t.net_trusted)?;
    Ok((rows, changes, d))
}

/// Writes plot rows (dicts as `change_compare_plot` returns them) as CSV.
#[pyfunction]
fn change_plot_csv(path: PathBuf, rows: Vec<Bound<'_, PyDict>>) -> PyResult<()> {
    let rows: Vec<PlotRow> = rows.iter().map(row_from_py).collect::<PyResult<_>>()?;
    cq::write_plot_csv(&path, &rows).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(change_compare_qsms, m)?)?;
    m.add_function(wrap_pyfunction!(change_compare_plot, m)?)?;
    m.add_function(wrap_pyfunction!(change_plot_csv, m)?)?;
    Ok(())
}
