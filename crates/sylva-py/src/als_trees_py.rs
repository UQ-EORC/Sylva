// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::als::trees and the tree truth of
//! sylva_rs::synthetic::als.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::als::trees::{self, Dalponte, Hull, Li2012, Method, Shape, TopsFrom, Tree, TreeParams, Window};
use sylva_rs::synthetic::als;

use crate::als_py::{catalog_from_py, heights, run_options};
use crate::{cloud_from_py, err, raster_from_py};

fn window(kind: &str, values: Vec<f64>) -> PyResult<Window> {
    let need = |n: usize| -> PyResult<()> {
        if values.len() != n {
            return Err(PyValueError::new_err(format!("a {kind} window takes {n} values, got {}", values.len())));
        }
        Ok(())
    };
    Ok(match kind {
        "fixed" => {
            need(1)?;
            Window::Fixed(values[0])
        }
        "linear" => {
            need(4)?;
            Window::Linear { intercept: values[0], slope: values[1], min: values[2], max: values[3] }
        }
        "table" => {
            if values.len() < 2 {
                return Err(PyValueError::new_err("a window table needs its step and at least one value"));
            }
            Window::Table { step: values[0], values: values[1..].to_vec() }
        }
        "site" => Window::PerSite(values),
        _ => return Err(PyValueError::new_err(format!("unknown window kind {kind:?}"))),
    })
}

fn xy_from(a: &PyReadonlyArray2<f64>) -> PyResult<Vec<[f64; 2]>> {
    let v = a.as_array();
    if v.ncols() < 2 {
        return Err(PyValueError::new_err(format!("coordinates must have shape (N, 2) or (N, 3), got (N, {})", v.ncols())));
    }
    Ok(v.rows().into_iter().map(|r| [r[0], r[1]]).collect())
}

fn hull_of(kind: &str, length: f64) -> PyResult<Hull> {
    match kind {
        "convex" => Ok(Hull::Convex),
        "concave" => Ok(Hull::Concave(length)),
        _ => Err(PyValueError::new_err(format!("unknown hull {kind:?}; expected 'convex' or 'concave'"))),
    }
}

/// Segmentation settings from the flat arguments the Python layer passes.
fn params(method: &str, s: &Bound<'_, PyDict>) -> PyResult<TreeParams> {
    let f = |k: &str| -> PyResult<f64> { s.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("missing setting {k:?}")))?.extract::<f64>() };
    let m = match method {
        "tops" => Method::Tops,
        "watershed" => Method::Watershed { th_tree: f("th_tree")? },
        "dalponte2016" => Method::Dalponte(Dalponte { th_tree: f("th_tree")?, th_seed: f("th_seed")?, th_cr: f("th_cr")?, max_cr: f("max_cr")? }),
        "li2012" => Method::Li2012(Li2012 { dt1: f("dt1")?, dt2: f("dt2")?, r: f("R")?, zu: f("Zu")?, hmin: f("hmin")?, speed_up: f("speed_up")? }),
        _ => return Err(PyValueError::new_err(format!("unknown method {method:?}"))),
    };
    let kind: String = s.get_item("window_kind")?.ok_or_else(|| PyValueError::new_err("missing window"))?.extract()?;
    let values: Vec<f64> = s.get_item("window_values")?.ok_or_else(|| PyValueError::new_err("missing window"))?.extract()?;
    let shape: String = s.get_item("shape")?.ok_or_else(|| PyValueError::new_err("missing shape"))?.extract()?;
    let tops_from: String = s.get_item("tops_from")?.ok_or_else(|| PyValueError::new_err("missing tops_from"))?.extract()?;
    let hull: String = s.get_item("hull")?.ok_or_else(|| PyValueError::new_err("missing hull"))?.extract()?;
    Ok(TreeParams {
        method: m,
        resolution: f("resolution")?,
        window: window(&kind, values)?,
        hmin: f("hmin")?,
        shape: Shape::parse(&shape).map_err(err)?,
        tops_from: match tops_from.as_str() {
            "chm" => TopsFrom::Chm,
            "points" => TopsFrom::Points,
            other => return Err(PyValueError::new_err(format!("unknown tops_from {other:?}; expected 'chm' or 'points'"))),
        },
        hull: hull_of(&hull, f("concavity")?)?,
        min_point_height: f("min_point_height")?,
        smooth: {
            let v = f("smooth")?;
            if !(v.is_finite() && v >= 0.0 && v.fract() == 0.0) {
                return Err(PyValueError::new_err(format!("smooth must be a whole number of cells, 0 or more, got {v}")));
            }
            v as usize
        },
    })
}

fn polygons_to_py<'py>(py: Python<'py>, polys: impl Iterator<Item = &'py Vec<[f64; 2]>>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for p in polys {
        let flat: Vec<f64> = p.iter().flat_map(|q| q.iter().copied()).collect();
        out.append(PyArray1::from_vec(py, flat).reshape([p.len(), 2])?)?;
    }
    Ok(out)
}

fn trees_to_py<'py>(py: Python<'py>, trees: &[Tree]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", trees.iter().map(|t| t.id as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("x", trees.iter().map(|t| t.x).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("y", trees.iter().map(|t| t.y).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("height", trees.iter().map(|t| t.height).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("crown_area", trees.iter().map(|t| t.crown_area).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("n_points", trees.iter().map(|t| t.n_points as i64).collect::<Vec<_>>().into_pyarray(py))?;
    let out = PyList::empty(py);
    for t in trees {
        let flat: Vec<f64> = t.crown.iter().flat_map(|q| q.iter().copied()).collect();
        out.append(PyArray1::from_vec(py, flat).reshape([t.crown.len(), 2])?)?;
    }
    d.set_item("crowns", out)?;
    Ok(d)
}

/// Tree-top cells (row-major indices) of a CHM.
#[pyfunction]
fn als_trees_lmf_raster<'py>(py: Python<'py>, data: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64, window_kind: &str, window_values: Vec<f64>, hmin: f64, shape: &str) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let r = raster_from_py(data, xmin, ymin, resolution);
    let (w, s) = (window(window_kind, window_values)?, Shape::parse(shape).map_err(err)?);
    let tops = py.detach(|| trees::local_maxima_raster(&r, &w, hmin, s)).map_err(err)?;
    Ok(tops.into_iter().map(|i| i as i64).collect::<Vec<_>>().into_pyarray(py))
}

/// Tree-top points (indices) of a cloud.
#[pyfunction]
fn als_trees_lmf_points<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, h: PyReadonlyArray1<f64>, window_kind: &str, window_values: Vec<f64>, hmin: f64, shape: &str) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let xy = xy_from(&xyz)?;
    let h = h.as_array().to_vec();
    let (w, s) = (window(window_kind, window_values)?, Shape::parse(shape).map_err(err)?);
    let tops = py.detach(|| trees::local_maxima_points(&xy, &h, &w, hmin, s)).map_err(err)?;
    Ok(tops.into_iter().map(|i| i as i64).collect::<Vec<_>>().into_pyarray(py))
}

/// Crowns (an id per cell, 0 for none) on a CHM from tops `(k, 3)`.
#[pyfunction]
fn als_trees_crowns<'py>(py: Python<'py>, data: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64, tops: PyReadonlyArray2<f64>, method: &str, th_tree: f64, th_seed: f64, th_cr: f64, max_cr: f64) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let r = raster_from_py(data, xmin, ymin, resolution);
    let t = tops.as_array();
    if t.ncols() != 3 {
        return Err(PyValueError::new_err(format!("tops must have shape (k, 3), got (k, {})", t.ncols())));
    }
    let tops: Vec<[f64; 3]> = t.rows().into_iter().map(|q| [q[0], q[1], q[2]]).collect();
    let labels = py
        .detach(|| {
            let seeds = trees::seed_raster(&r, &tops);
            match method {
                "watershed" => trees::watershed(&r, &seeds, th_tree),
                "dalponte2016" => trees::dalponte2016(&r, &seeds, &Dalponte { th_tree, th_seed, th_cr, max_cr }),
                other => Err(sylva_rs::Error::invalid(format!("unknown method {other:?}; expected 'dalponte2016' or 'watershed'"))),
            }
        })
        .map_err(err)?;
    PyArray1::from_vec(py, labels.into_iter().map(|v| v as i64).collect()).reshape([r.nrows, r.ncols])
}

/// Li et al. (2012): the tree of each point (0 for none).
#[pyfunction]
fn als_trees_li2012<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, h: PyReadonlyArray1<f64>, dt1: f64, dt2: f64, r: f64, zu: f64, hmin: f64, speed_up: f64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let xy = xy_from(&xyz)?;
    let h = h.as_array().to_vec();
    let p = Li2012 { dt1, dt2, r, zu, hmin, speed_up };
    let l = py.detach(|| trees::li2012(&xy, &h, &p)).map_err(err)?;
    Ok(l.into_iter().map(|v| v as i64).collect::<Vec<_>>().into_pyarray(py))
}

/// Outline of 2-D points.
#[pyfunction]
fn als_trees_hull<'py>(py: Python<'py>, xy: PyReadonlyArray2<f64>, kind: &str, length: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pts = xy_from(&xy)?;
    let h = hull_of(kind, length)?;
    if let Hull::Concave(l) = h {
        if !(l.is_finite() && l > 0.0) {
            return Err(PyValueError::new_err(format!("concavity must be a positive number of metres, got {l}")));
        }
    }
    let out = py.detach(|| trees::hull(&pts, h));
    let n = out.len();
    PyArray1::from_vec(py, out.into_iter().flat_map(|q| q.into_iter()).collect()).reshape([n, 2])
}

/// Trees of one cloud and each point's tree id.
#[pyfunction]
fn als_trees_segment<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, h: PyReadonlyArray1<f64>, method: &str, settings: &Bound<'_, PyDict>) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyArray1<i64>>)> {
    let cloud = cloud_from_py(xyz, None)?;
    let h = h.as_array().to_vec();
    let p = params(method, settings)?;
    let (trees, labels) = py.detach(|| trees::segment_cloud(&cloud.xyz, &h, &p)).map_err(err)?;
    Ok((trees_to_py(py, &trees)?, labels.into_iter().map(|v| v as i64).collect::<Vec<_>>().into_pyarray(py)))
}

/// Trees of a catalogue.
#[pyfunction]
#[pyo3(signature = (catalog, method, settings, mode, dtm, dtm_resolution, out_dir, attribute, format, chunk_size, buffer, workers))]
fn als_trees_catalog<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, method: &str, settings: &Bound<'_, PyDict>, mode: &str, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, dtm_resolution: f64, out_dir: Option<PathBuf>, attribute: &str, format: Option<String>, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let c = catalog_from_py(catalog)?;
    let p = params(method, settings)?;
    let h = heights(mode, dtm, dtm_resolution)?;
    let opts = run_options(chunk_size, buffer, workers);
    let r = py.detach(|| trees::catalog_trees(&c, &p, &h, out_dir.as_deref(), attribute, format.as_deref(), &opts)).map_err(err)?;
    let d = trees_to_py(py, &r.trees)?;
    d.set_item("written", r.written.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>())?;
    d.set_item("at_edge", r.at_edge)?;
    d.set_item("unmatched_points", r.unmatched_points)?;
    Ok(d)
}

/// The trees of a synthetic scene, as a dict of columns.
#[pyfunction]
fn synthetic_scene_trees<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: &Bound<'_, PyDict>, terrain_slope: f64) -> PyResult<Bound<'py, PyDict>> {
    let cloud = cloud_from_py(xyz, Some(attrs))?;
    let t = py.detach(|| als::scene_trees(&cloud, terrain_slope)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("tree_id", t.iter().map(|s| s.id as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("stem_x", t.iter().map(|s| s.stem[0]).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("stem_y", t.iter().map(|s| s.stem[1]).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("top_x", t.iter().map(|s| s.top[0]).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("top_y", t.iter().map(|s| s.top[1]).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("top_z", t.iter().map(|s| s.top[2]).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("height", t.iter().map(|s| s.height).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("crown_area", t.iter().map(|s| s.crown_area).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("crowns", polygons_to_py(py, t.iter().map(|s| &s.crown))?)?;
    Ok(d)
}

/// A scene of trees with solid crowns.
#[pyfunction]
fn synthetic_crown_forest<'py>(py: Python<'py>, trees: Vec<(f64, f64, f64, f64)>, form: &str, crown_radius: f64, crown_length: f64, density: f64, size: f64, ground_points: usize, margin: f64, seed: u64) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let f = als::CrownForm::parse(form).map_err(err)?;
    let c = py.detach(|| als::crown_forest(&trees, f, crown_radius, crown_length, density, size, ground_points, margin, seed)).map_err(err)?;
    crate::cloud_to_py(py, &c)
}

/// Random stems for synthetic.forest.
#[pyfunction]
fn synthetic_stand(n: usize, size: f64, min_spacing: f64, heights: (f64, f64), seed: u64) -> PyResult<Vec<(f64, f64, f64, f64)>> {
    als::stand(n, size, min_spacing, heights, seed).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(als_trees_lmf_raster, m)?,
        wrap_pyfunction!(als_trees_lmf_points, m)?,
        wrap_pyfunction!(als_trees_crowns, m)?,
        wrap_pyfunction!(als_trees_li2012, m)?,
        wrap_pyfunction!(als_trees_hull, m)?,
        wrap_pyfunction!(als_trees_segment, m)?,
        wrap_pyfunction!(als_trees_catalog, m)?,
        wrap_pyfunction!(synthetic_scene_trees, m)?,
        wrap_pyfunction!(synthetic_stand, m)?,
        wrap_pyfunction!(synthetic_crown_forest, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
