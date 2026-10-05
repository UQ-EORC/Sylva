// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::als::metrics.
//!
//! Heights cross as `(mode, dtm, dtm_resolution, attribute)`: mode "z",
//! "auto", "dtm" (with the raster) or "attribute" (with its name). Plots
//! cross as an `(N, 3)` array of circles (x, y, radius) or as flattened
//! polygons (see `masks_py`).
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::als::metrics::{self, HeightSource, MetricParams, Plot};
use sylva_rs::als::ops::RunOptions;

use crate::als_py::{catalog_from_py, chunk_from_py, chunk_to_py, heights, layout};
use crate::masks_py::features_from_flat;
use crate::{cloud_from_py, cloud_to_py, err, raster_to_py};

type Dtm<'py> = Option<(PyReadonlyArray2<'py, f64>, f64, f64, f64)>;
type FlatPolygons<'py> = Option<(PyReadonlyArray2<'py, f64>, PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>)>;

fn source(mode: &str, dtm: Dtm<'_>, dtm_resolution: f64, attribute: Option<String>) -> PyResult<HeightSource> {
    if mode == "attribute" {
        return Ok(HeightSource::Attribute(attribute.ok_or_else(|| PyValueError::new_err("the height attribute needs a name"))?));
    }
    Ok(HeightSource::Heights(heights(mode, dtm, dtm_resolution)?))
}

fn params(threshold: f64, entropy_bin: f64, cover_break: f64, min_height: Option<f64>, drop_noise: bool, clamp_negative: bool) -> MetricParams {
    MetricParams { threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative }
}

fn plots_from_py(circles: Option<PyReadonlyArray2<'_, f64>>, polygons: FlatPolygons<'_>) -> PyResult<Vec<Plot>> {
    match (circles, polygons) {
        (Some(c), None) => {
            if c.shape()[1] != 3 {
                return Err(PyValueError::new_err(format!("circles must have shape (N, 3), got (N, {})", c.shape()[1])));
            }
            Ok(c.as_array().rows().into_iter().map(|r| Plot::Circle { x: r[0], y: r[1], radius: r[2] }).collect())
        }
        (None, Some((coords, rings, parts, feats))) => Ok(features_from_flat(coords, rings, parts, feats)?.into_iter().map(Plot::Polygon).collect()),
        _ => Err(PyValueError::new_err("give either circles or polygons")),
    }
}

/// Metrics of one cloud: names and values.
#[pyfunction]
#[pyo3(signature = (xyz, attrs, heights, threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative))]
fn als_cloud_metrics<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, heights: Option<PyReadonlyArray1<f64>>, threshold: f64, entropy_bin: f64, cover_break: f64, min_height: Option<f64>, drop_noise: bool, clamp_negative: bool) -> PyResult<(Vec<String>, Bound<'py, PyArray1<f64>>)> {
    let cloud = cloud_from_py(xyz, attrs)?;
    let h: Vec<f64> = match heights {
        Some(h) => h.as_array().to_vec(),
        None => cloud.xyz.iter().map(|p| p[2]).collect(),
    };
    let p = params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative);
    let (names, values) = py.detach(|| metrics::cloud_metrics(&cloud, &h, &p)).map_err(err)?;
    Ok((names, values.into_pyarray(py)))
}

/// Names of the metrics a catalogue gives.
#[pyfunction]
fn als_metric_names(threshold: f64) -> Vec<String> {
    metrics::metric_names(metrics::Available::ALL, &MetricParams { threshold, ..Default::default() })
}

#[pyfunction]
#[pyo3(signature = (catalog, resolution, names, mode, dtm, dtm_resolution, attribute, threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative, chunk_size, buffer, workers))]
fn als_grid_metrics<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, resolution: f64, names: Option<Vec<String>>, mode: &str, dtm: Dtm<'_>, dtm_resolution: f64, attribute: Option<String>, threshold: f64, entropy_bin: f64, cover_break: f64, min_height: Option<f64>, drop_noise: bool, clamp_negative: bool, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<(Vec<String>, Bound<'py, PyList>)> {
    let c = catalog_from_py(catalog)?;
    let h = source(mode, dtm, dtm_resolution, attribute)?;
    let p = params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative);
    let opts = RunOptions { layout: layout(chunk_size, None), buffer, workers };
    let r = py.detach(|| metrics::grid_metrics(&c, resolution, &h, &p, names.as_deref(), &opts)).map_err(err)?;
    let out = PyList::empty(py);
    for raster in &r.rasters {
        out.append(raster_to_py(py, raster)?)?;
    }
    Ok((r.names, out))
}

/// The chunks of a grid-metrics run (buffer at least 1.5 cells).
#[pyfunction]
#[pyo3(signature = (catalog, resolution, chunk_size, buffer))]
fn als_metrics_plan<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, resolution: f64, chunk_size: Option<f64>, buffer: f64) -> PyResult<Bound<'py, PyList>> {
    let c = catalog_from_py(catalog)?;
    let chunks = metrics::metrics_plan(&c, layout(chunk_size, None), buffer, resolution).map_err(err)?;
    let out = PyList::empty(py);
    for ch in &chunks {
        out.append(chunk_to_py(py, ch)?)?;
    }
    Ok(out)
}

/// A chunk's retained points grouped by grid cell: the points in group
/// order, their heights, each cell's grid index, group starts (length
/// cells + 1) and whether each cell's centre is in the core. None when the
/// chunk has no points of its own or too little ground.
#[pyfunction]
#[pyo3(signature = (catalog, chunk, resolution, mode, dtm, dtm_resolution, attribute, threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative))]
fn als_metric_cells<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, chunk: &Bound<'_, PyDict>, resolution: f64, mode: &str, dtm: Dtm<'_>, dtm_resolution: f64, attribute: Option<String>, threshold: f64, entropy_bin: f64, cover_break: f64, min_height: Option<f64>, drop_noise: bool, clamp_negative: bool) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<bool>>)>> {
    let (c, ch) = (catalog_from_py(catalog)?, chunk_from_py(chunk)?);
    let h = source(mode, dtm, dtm_resolution, attribute)?;
    let p = params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative);
    p.check().map_err(err)?;
    let grid = sylva_rs::als::catalog_grid(&c, resolution).map_err(err)?;
    let got = py
        .detach(|| -> sylva_rs::Result<_> {
            let data = sylva_rs::als::read_chunk(&c, &ch)?;
            if data.n_core() == 0 {
                return Ok(None);
            }
            Ok(metrics::chunk_cells(&c, &ch, &data.cloud, &grid, &h, &p)?.map(|cells| (data.cloud.take(&cells.order), cells)))
        })
        .map_err(err)?;
    let Some((cloud, cells)) = got else { return Ok(None) };
    let (xyz, attrs) = cloud_to_py(py, &cloud)?;
    let i64s = |v: Vec<usize>| v.into_iter().map(|x| x as i64).collect::<Vec<i64>>().into_pyarray(py);
    Ok(Some((xyz, attrs, cells.heights.into_pyarray(py), i64s(cells.cells), i64s(cells.starts), cells.central.into_pyarray(py))))
}

#[pyfunction]
#[pyo3(signature = (catalog, circles, polygons, names, mode, dtm, dtm_resolution, attribute, threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative, buffer, workers))]
fn als_plot_metrics<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, circles: Option<PyReadonlyArray2<f64>>, polygons: FlatPolygons<'_>, names: Option<Vec<String>>, mode: &str, dtm: Dtm<'_>, dtm_resolution: f64, attribute: Option<String>, threshold: f64, entropy_bin: f64, cover_break: f64, min_height: Option<f64>, drop_noise: bool, clamp_negative: bool, buffer: f64, workers: usize) -> PyResult<(Vec<String>, Bound<'py, PyArray2<f64>>)> {
    let c = catalog_from_py(catalog)?;
    let plots = plots_from_py(circles, polygons)?;
    let h = source(mode, dtm, dtm_resolution, attribute)?;
    let p = params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative);
    let (names, rows) = py.detach(|| metrics::plot_metrics(&c, &plots, &h, &p, names.as_deref(), buffer, workers)).map_err(err)?;
    let k = names.len();
    let flat: Vec<f64> = rows.concat();
    let arr = numpy::ndarray::Array2::from_shape_vec((plots.len(), k), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((names, arr.into_pyarray(py)))
}

/// The retained points of each plot, in canonical order, with their
/// heights; None for plots overlapping no tile (or without enough ground).
#[pyfunction]
#[pyo3(signature = (catalog, circles, polygons, mode, dtm, dtm_resolution, attribute, threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative, buffer, workers))]
fn als_plot_points<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, circles: Option<PyReadonlyArray2<f64>>, polygons: FlatPolygons<'_>, mode: &str, dtm: Dtm<'_>, dtm_resolution: f64, attribute: Option<String>, threshold: f64, entropy_bin: f64, cover_break: f64, min_height: Option<f64>, drop_noise: bool, clamp_negative: bool, buffer: f64, workers: usize) -> PyResult<Bound<'py, PyList>> {
    let c = catalog_from_py(catalog)?;
    let plots = plots_from_py(circles, polygons)?;
    let h = source(mode, dtm, dtm_resolution, attribute)?;
    let p = params(threshold, entropy_bin, cover_break, min_height, drop_noise, clamp_negative);
    let got = py.detach(|| metrics::plot_points(&c, &plots, &h, &p, buffer, workers, |_, cloud, heights, idx| Ok((cloud.take(idx), idx.iter().map(|&i| heights[i]).collect::<Vec<f64>>())))).map_err(err)?;
    let out = PyList::empty(py);
    for g in got {
        match g {
            Some((cloud, hs)) => {
                let (xyz, attrs) = cloud_to_py(py, &cloud)?;
                out.append((xyz, attrs, hs.into_pyarray(py)))?;
            }
            None => out.append(py.None())?,
        }
    }
    Ok(out)
}

/// Write plot metrics as CSV.
#[pyfunction]
#[pyo3(signature = (path, names, ids, values))]
fn als_metrics_csv(path: std::path::PathBuf, names: Vec<String>, ids: Option<Vec<String>>, values: PyReadonlyArray2<f64>) -> PyResult<()> {
    let rows: Vec<Vec<f64>> = values.as_array().rows().into_iter().map(|r| r.to_vec()).collect();
    metrics::write_metrics_csv(&path, &names, ids.as_deref(), &rows).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(als_cloud_metrics, m)?,
        wrap_pyfunction!(als_metric_names, m)?,
        wrap_pyfunction!(als_grid_metrics, m)?,
        wrap_pyfunction!(als_metrics_plan, m)?,
        wrap_pyfunction!(als_metric_cells, m)?,
        wrap_pyfunction!(als_plot_metrics, m)?,
        wrap_pyfunction!(als_plot_points, m)?,
        wrap_pyfunction!(als_metrics_csv, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
