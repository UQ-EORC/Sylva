// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::als::trajectory and sylva_rs::als::canopy.
//!
//! A trajectory crosses the boundary as a dict of equal-length arrays
//! (`time`, `x`, `y`, `z` and optionally `roll`, `pitch`, `heading`), a
//! profile grid as a dict of its fields with `(layers, ny, nx)` arrays.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::als::canopy::{self as ac, Angles, CatalogVoxels, ProfileGrid, ProfileParams, Projection, PulseParams, PulseReport, ReturnWeight, TraceSettings};
use sylva_rs::als::trajectory::{self as at, EstimateParams, Estimated, Sbet, Trajectory};
use sylva_rs::geo::crs::{self, Crs};
use sylva_rs::voxel::{self, EchoLabels};
use sylva_rs::Point;

use crate::als_py::{catalog_from_py, heights, run_options};
use crate::voxels_py::params as voxel_params;
use crate::{cloud_from_py, err, raster_from_py, shots_to_py, xyz_to_py};

fn arr(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Option<Vec<f64>>> {
    match d.get_item(k)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec())),
    }
}

fn traj_from_py(d: &Bound<'_, PyDict>) -> PyResult<Trajectory> {
    let need = |k: &str| arr(d, k)?.ok_or_else(|| PyValueError::new_err(format!("the trajectory has no {k:?}")));
    let (t, x, y, z) = (need("time")?, need("x")?, need("y")?, need("z")?);
    if x.len() != t.len() || y.len() != t.len() || z.len() != t.len() {
        return Err(PyValueError::new_err("trajectory time, x, y and z differ in length"));
    }
    let xyz: Vec<Point> = (0..t.len()).map(|i| [x[i], y[i], z[i]]).collect();
    Trajectory::new(t, xyz, arr(d, "roll")?, arr(d, "pitch")?, arr(d, "heading")?).map_err(err)
}

fn traj_to_py<'py>(py: Python<'py>, t: &Trajectory) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("time", t.time.clone().into_pyarray(py))?;
    for (k, name) in ["x", "y", "z"].iter().enumerate() {
        d.set_item(*name, t.xyz.iter().map(|p| p[k]).collect::<Vec<_>>().into_pyarray(py))?;
    }
    for (name, v) in [("roll", &t.roll), ("pitch", &t.pitch), ("heading", &t.heading)] {
        if let Some(v) = v {
            d.set_item(name, v.clone().into_pyarray(py))?;
        }
    }
    Ok(d)
}

/// Check and order a trajectory (sorted, repeated times dropped).
#[pyfunction]
fn als_trajectory_check<'py>(py: Python<'py>, trajectory: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    traj_to_py(py, &traj_from_py(trajectory)?)
}

#[pyfunction]
#[pyo3(signature = (trajectory, times, max_gap=None))]
fn als_trajectory_positions<'py>(py: Python<'py>, trajectory: &Bound<'_, PyDict>, times: PyReadonlyArray1<f64>, max_gap: Option<f64>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let t = traj_from_py(trajectory)?;
    let times = times.as_array().to_vec();
    let g = max_gap.unwrap_or_else(|| t.default_max_gap());
    Ok(xyz_to_py(py, &py.detach(|| t.positions(&times, g))))
}

#[pyfunction]
#[pyo3(signature = (trajectory, times, max_gap=None))]
fn als_trajectory_attitude<'py>(py: Python<'py>, trajectory: &Bound<'_, PyDict>, times: PyReadonlyArray1<f64>, max_gap: Option<f64>) -> PyResult<Option<Bound<'py, numpy::PyArray2<f64>>>> {
    let t = traj_from_py(trajectory)?;
    let times = times.as_array().to_vec();
    let g = max_gap.unwrap_or_else(|| t.default_max_gap());
    Ok(t.attitude(&times, g).map(|a| xyz_to_py(py, &a)))
}

#[pyfunction]
fn als_read_sbet<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let s = py.detach(|| at::read_sbet(&path)).map_err(err)?;
    let d = PyDict::new(py);
    for (k, v) in [("time", s.time), ("latitude", s.latitude), ("longitude", s.longitude), ("height", s.height), ("roll", s.roll), ("pitch", s.pitch), ("heading", s.heading)] {
        d.set_item(k, v.into_pyarray(py))?;
    }
    Ok(d)
}

#[pyfunction]
fn als_write_sbet(path: PathBuf, sbet: &Bound<'_, PyDict>) -> PyResult<()> {
    let need = |k: &str| arr(sbet, k)?.ok_or_else(|| PyValueError::new_err(format!("SBET records need {k:?}")));
    let s = Sbet { time: need("time")?, latitude: need("latitude")?, longitude: need("longitude")?, height: need("height")?, roll: need("roll")?, pitch: need("pitch")?, heading: need("heading")? };
    at::write_sbet(&path, &s).map_err(err)
}

#[pyfunction]
fn als_read_table<'py>(py: Python<'py>, path: PathBuf) -> PyResult<(Vec<String>, Vec<Bound<'py, PyArray1<f64>>>)> {
    let t = py.detach(|| at::read_table(&path)).map_err(err)?;
    Ok((t.names, t.columns.into_iter().map(|c| c.into_pyarray(py)).collect()))
}

#[pyfunction]
fn als_column_role(name: &str) -> Option<&'static str> {
    at::column_role(name)
}

/// Project longitude / latitude (degrees) and height to `crs`.
#[pyfunction]
fn als_project_geographic<'py>(py: Python<'py>, lon: PyReadonlyArray1<f64>, lat: PyReadonlyArray1<f64>, height: PyReadonlyArray1<f64>, crs: &str) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (lon, lat, h) = (lon.as_array(), lat.as_array(), height.as_array());
    if lat.len() != lon.len() || h.len() != lon.len() {
        return Err(PyValueError::new_err("longitude, latitude and height differ in length"));
    }
    let mut xyz: Vec<Point> = (0..lon.len()).map(|i| [lon[i], lat[i], h[i]]).collect();
    let dst = Crs::parse(crs).map_err(err)?;
    if dst.is_geographic().map_err(err)? {
        return Err(PyValueError::new_err(format!("{crs:?} is a geographic CRS; ray tracing needs a projected one (metres)")));
    }
    py.detach(|| crs::reproject(&mut xyz, &Crs::from_epsg(4326)?, &dst)).map_err(err)?;
    Ok(xyz_to_py(py, &xyz))
}

fn estimated_to_py<'py>(py: Python<'py>, e: &Estimated) -> PyResult<Bound<'py, PyDict>> {
    let d = traj_to_py(py, &e.trajectory)?;
    d.set_item("line", e.line.clone().into_pyarray(py))?;
    d.set_item("n_pulses", e.n_pulses.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("rms", e.rms.clone().into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (xyz, attrs, interval, min_pulses, min_separation, max_pulses, extend))]
fn als_estimate_trajectory<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, interval: f64, min_pulses: usize, min_separation: f64, max_pulses: usize, extend: f64) -> PyResult<Bound<'py, PyDict>> {
    let c = cloud_from_py(xyz, attrs)?;
    let p = EstimateParams { interval, min_pulses, min_separation, max_pulses, extend };
    let e = py.detach(|| at::estimate(&at::thin(at::pulse_lines(&c, min_separation)?, interval, max_pulses), &at::line_extents(&c)?, &p)).map_err(err)?;
    estimated_to_py(py, &e)
}

#[pyfunction]
#[pyo3(signature = (catalog, interval, min_pulses, min_separation, max_pulses, extend, workers))]
fn als_estimate_trajectory_catalog<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, interval: f64, min_pulses: usize, min_separation: f64, max_pulses: usize, extend: f64, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let c = catalog_from_py(catalog)?;
    let p = EstimateParams { interval, min_pulses, min_separation, max_pulses, extend };
    let e = py.detach(|| ac::estimate_catalog(&c, &p, workers)).map_err(err)?;
    estimated_to_py(py, &e)
}

fn report_to_py<'py>(py: Python<'py>, r: &PulseReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in [("n_returns", r.n_returns), ("n_pulses", r.n_pulses), ("n_incomplete", r.n_incomplete), ("n_missing_returns", r.n_missing_returns), ("n_dropped", r.n_dropped), ("n_split", r.n_split), ("n_unpositioned", r.n_unpositioned), ("n_filled", r.n_filled)] {
        d.set_item(k, v)?;
    }
    d.set_item("pulse_interval", r.pulse_interval)?;
    d.set_item("line_offset_median", r.line_offset_median)?;
    d.set_item("line_offset_p95", r.line_offset_p95)?;
    Ok(d)
}

fn pulse_params(max_gap: Option<f64>, time_offset: f64, fill_missing: bool, max_fill: usize, drop_incomplete: bool) -> PulseParams {
    PulseParams { max_gap, time_offset, fill_missing, max_fill, drop_incomplete }
}

#[pyfunction]
#[pyo3(signature = (xyz, attrs, trajectory, max_gap, time_offset, fill_missing, max_fill, drop_incomplete))]
fn als_pulses<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, trajectory: &Bound<'_, PyDict>, max_gap: Option<f64>, time_offset: f64, fill_missing: bool, max_fill: usize, drop_incomplete: bool) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let c = cloud_from_py(xyz, attrs)?;
    let t = traj_from_py(trajectory)?;
    let p = pulse_params(max_gap, time_offset, fill_missing, max_fill, drop_incomplete);
    let (s, r) = py.detach(|| ac::reconstruct(&c, &t, &p)).map_err(err)?;
    Ok((shots_to_py(py, &s)?, report_to_py(py, &r)?))
}

// ---------------------------------------------------------------- profiles

fn profile_params(resolution: f64, min_height: f64, bin_size: f64, max_height: Option<f64>, top_quantile: f64, drop_noise: bool, weighting: &str, lad: &str, lad_params: Vec<f64>, g: Option<f64>, angles: &str, trajectory: Option<&Bound<'_, PyDict>>, max_gap: Option<f64>, time_offset: f64, max_zenith: f64, anchor: &str) -> PyResult<ProfileParams> {
    let projection = match g {
        Some(v) => Projection::Constant(v),
        None => Projection::Lad(voxel::Lad::parse(lad, &lad_params).map_err(err)?),
    };
    let angles = match (angles, trajectory) {
        ("trajectory", Some(t)) => Angles::Trajectory { trajectory: traj_from_py(t)?, max_gap, time_offset },
        ("trajectory", None) => return Err(PyValueError::new_err("angles='trajectory' needs a trajectory")),
        ("scan_angle", _) => Angles::ScanAngle,
        ("none", _) => Angles::Nadir,
        (other, _) => return Err(PyValueError::new_err(format!("unknown angles {other:?}; expected 'trajectory', 'scan_angle' or 'none'"))),
    };
    let anchor_ground = match anchor {
        "ground" => true,
        "return" => false,
        other => return Err(PyValueError::new_err(format!("anchor must be 'ground' or 'return', got {other:?}"))),
    };
    Ok(ProfileParams { resolution, min_height, bin_size, max_height, top_quantile, drop_noise, weighting: ReturnWeight::parse(weighting).map_err(err)?, projection, angles, max_zenith, anchor_ground })
}

fn grid_to_py<'py>(py: Python<'py>, g: &ProfileGrid) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("xmin", g.xmin)?;
    d.set_item("ymin", g.ymin)?;
    d.set_item("resolution", g.resolution)?;
    d.set_item("min_height", g.min_height)?;
    d.set_item("bin_size", g.bin_size)?;
    d.set_item("weight", PyArray1::from_vec(py, g.weight.clone()).reshape([g.nz + 2, g.ny, g.nx])?)?;
    d.set_item("weight_k", PyArray1::from_vec(py, g.weight_k.clone()).reshape([g.nz + 2, g.ny, g.nx])?)?;
    d.set_item("n_skipped", g.n_skipped)?;
    Ok(d)
}

fn grid_from_py(weight: PyReadonlyArray3<f64>, weight_k: PyReadonlyArray3<f64>, bin_size: f64) -> PyResult<ProfileGrid> {
    let s = weight.shape();
    if s != weight_k.shape() || s[0] < 2 {
        return Err(PyValueError::new_err("weight and weight_k must be (layers, ny, nx) arrays of the same shape, with at least 2 layers"));
    }
    Ok(ProfileGrid { nz: s[0] - 2, ny: s[1], nx: s[2], bin_size, resolution: 1.0, weight: weight.as_array().iter().cloned().collect(), weight_k: weight_k.as_array().iter().cloned().collect(), ..Default::default() })
}

#[pyfunction]
#[pyo3(signature = (xyz, attrs, heights, bounds, resolution, min_height, bin_size, max_height, top_quantile, drop_noise, weighting, lad, lad_params, g, angles, trajectory, max_gap, time_offset, max_zenith, anchor))]
fn als_profile_cloud<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, heights: PyReadonlyArray1<f64>, bounds: Option<(f64, f64, f64, f64)>, resolution: f64, min_height: f64, bin_size: f64, max_height: Option<f64>, top_quantile: f64, drop_noise: bool, weighting: &str, lad: &str, lad_params: Vec<f64>, g: Option<f64>, angles: &str, trajectory: Option<&Bound<'_, PyDict>>, max_gap: Option<f64>, time_offset: f64, max_zenith: f64, anchor: &str) -> PyResult<Bound<'py, PyDict>> {
    let c = cloud_from_py(xyz, attrs)?;
    let h = heights.as_array().to_vec();
    let p = profile_params(resolution, min_height, bin_size, max_height, top_quantile, drop_noise, weighting, lad, lad_params, g, angles, trajectory, max_gap, time_offset, max_zenith, anchor)?;
    let b = bounds.map(|b| [b.0, b.1, b.2, b.3]);
    let grid = py.detach(|| ac::profile_cloud(&c, &h, &p, b)).map_err(err)?;
    grid_to_py(py, &grid)
}

#[pyfunction]
#[pyo3(signature = (catalog, mode, dtm, dtm_resolution, resolution, min_height, bin_size, max_height, top_quantile, drop_noise, weighting, lad, lad_params, g, angles, trajectory, max_gap, time_offset, max_zenith, anchor, chunk_size, buffer, workers))]
fn als_profile_catalog<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, mode: &str, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, dtm_resolution: f64, resolution: f64, min_height: f64, bin_size: f64, max_height: Option<f64>, top_quantile: f64, drop_noise: bool, weighting: &str, lad: &str, lad_params: Vec<f64>, g: Option<f64>, angles: &str, trajectory: Option<&Bound<'_, PyDict>>, max_gap: Option<f64>, time_offset: f64, max_zenith: f64, anchor: &str, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let c = catalog_from_py(catalog)?;
    let h = heights(mode, dtm, dtm_resolution)?;
    let p = profile_params(resolution, min_height, bin_size, max_height, top_quantile, drop_noise, weighting, lad, lad_params, g, angles, trajectory, max_gap, time_offset, max_zenith, anchor)?;
    let opts = run_options(chunk_size, buffer, workers);
    let grid = py.detach(|| ac::profile_catalog(&c, &h, &p, &opts)).map_err(err)?;
    grid_to_py(py, &grid)
}

/// Plant area density `(nz, ny, nx)`, plant area index and cover `(ny, nx)` of a profile grid.
#[pyfunction]
fn als_profile_products<'py>(py: Python<'py>, weight: PyReadonlyArray3<f64>, weight_k: PyReadonlyArray3<f64>, bin_size: f64) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let g = grid_from_py(weight, weight_k, bin_size)?;
    let (pad, pai, cover) = py.detach(|| (g.pad(), g.pai(), g.cover()));
    Ok((
        PyArray1::from_vec(py, pad).reshape([g.nz, g.ny, g.nx])?.into_any(),
        PyArray1::from_vec(py, pai).reshape([g.ny, g.nx])?.into_any(),
        PyArray1::from_vec(py, cover).reshape([g.ny, g.nx])?.into_any(),
    ))
}

/// The pooled column of the cells in `mask`: its PAD per layer and gap probability per boundary.
#[pyfunction]
#[pyo3(signature = (weight, weight_k, bin_size, mask=None))]
fn als_profile_pooled<'py>(py: Python<'py>, weight: PyReadonlyArray3<f64>, weight_k: PyReadonlyArray3<f64>, bin_size: f64, mask: Option<PyReadonlyArray2<bool>>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let g = grid_from_py(weight, weight_k, bin_size)?;
    let m: Option<Vec<bool>> = mask.map(|m| m.as_array().iter().cloned().collect());
    let (w, wk) = g.pooled(m.as_deref()).map_err(err)?;
    Ok((ac::column_pad(&w, &wk, bin_size).into_pyarray(py), ac::column_pgap(&w).into_pyarray(py), ac::column_pai(&w, &wk)))
}

/// The profile metrics of every cell: their names and a `(k, ny, nx)` array.
#[pyfunction]
fn als_profile_metrics<'py>(py: Python<'py>, weight: PyReadonlyArray3<f64>, weight_k: PyReadonlyArray3<f64>, min_height: f64, bin_size: f64, strata: f64) -> PyResult<(Vec<String>, Bound<'py, PyAny>)> {
    let mut g = grid_from_py(weight, weight_k, bin_size)?;
    g.min_height = min_height;
    let (names, values) = py.detach(|| g.metrics(strata)).map_err(err)?;
    let k = names.len();
    Ok((names, PyArray1::from_vec(py, values).reshape([k, g.ny, g.nx])?.into_any()))
}

/// The profile metrics of areas, each the pooled column of its cells
/// (row-major indices): their names and an `(areas, k)` array.
#[pyfunction]
fn als_profile_area_metrics<'py>(py: Python<'py>, weight: PyReadonlyArray3<f64>, weight_k: PyReadonlyArray3<f64>, min_height: f64, bin_size: f64, strata: f64, areas: Vec<Vec<usize>>) -> PyResult<(Vec<String>, Bound<'py, PyAny>)> {
    let mut g = grid_from_py(weight, weight_k, bin_size)?;
    g.min_height = min_height;
    let (names, rows) = py.detach(|| g.area_metrics(&areas, strata)).map_err(err)?;
    let (n, k) = (rows.len(), names.len());
    Ok((names, PyArray1::from_vec(py, rows.concat()).reshape([n, k])?.into_any()))
}

// ------------------------------------------------------------ ray tracing

fn voxels_to_py<'py>(py: Python<'py>, v: &CatalogVoxels) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("origin", v.origin.to_vec().into_pyarray(py))?;
    d.set_item("voxel_size", v.voxel_size)?;
    let [nx, ny, nz] = v.shape;
    let f = PyDict::new(py);
    for (name, data) in &v.fields {
        f.set_item(name, PyArray1::from_vec(py, data.clone()).reshape([nz, ny, nx])?)?;
    }
    d.set_item("fields", f)?;
    d.set_item("n_pulses", v.n_pulses)?;
    d.set_item("n_unpositioned", v.n_unpositioned)?;
    d.set_item("reach", v.reach)?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
fn labels(class_attr: &str, ground_class: Option<i64>, ground_distance: f64, leaf_classes: Vec<i64>, wood_classes: Vec<i64>, tree_attr: &str, intensity_attr: &str) -> EchoLabels {
    EchoLabels { class_attr: class_attr.into(), ground_class, ground_distance, leaf_classes, wood_classes, tree_attr: tree_attr.into(), intensity_attr: intensity_attr.into() }
}

#[pyfunction]
#[pyo3(signature = (source, trajectory, fields, bounds, z_range, max_gap, time_offset, fill_missing, max_fill, drop_incomplete, voxel_size, dtm, class_attr, ground_class, ground_distance, leaf_classes, wood_classes, tree_attr, intensity_attr, weighting, occlusion, beam, average_leaf_area, lad, lad_params, attenuation, unbounded_range, chunk_size, buffer, workers))]
fn als_ray_voxelize<'py>(py: Python<'py>, source: &Bound<'_, PyAny>, trajectory: &Bound<'_, PyDict>, fields: Vec<String>, bounds: Option<(f64, f64, f64, f64, f64, f64)>, z_range: Option<(f64, f64)>, max_gap: Option<f64>, time_offset: f64, fill_missing: bool, max_fill: usize, drop_incomplete: bool, voxel_size: f64, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, class_attr: &str, ground_class: Option<i64>, ground_distance: f64, leaf_classes: Vec<i64>, wood_classes: Vec<i64>, tree_attr: &str, intensity_attr: &str, weighting: &str, occlusion: bool, beam: Option<(f64, f64)>, average_leaf_area: f64, lad: &str, lad_params: Vec<f64>, attenuation: Vec<String>, unbounded_range: f64, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let traj = traj_from_py(trajectory)?;
    let vp = voxel_params(voxel_size, None, weighting, occlusion, false, 0, beam, 0, 10, average_leaf_area, lad, lad_params, attenuation, false, 18, 10, 0.05, unbounded_range)?;
    let dtm = dtm.map(|(d, x, y, r)| raster_from_py(d, x, y, r));
    let s = TraceSettings { trajectory: &traj, pulses: pulse_params(max_gap, time_offset, fill_missing, max_fill, drop_incomplete), voxel: vp, labels: labels(class_attr, ground_class, ground_distance, leaf_classes, wood_classes, tree_attr, intensity_attr), dtm: dtm.as_ref(), fields };
    // A catalogue dict, or (xyz, attrs) of one cloud.
    let v = if let Ok(cat) = source.cast::<PyDict>() {
        let c = catalog_from_py(cat)?;
        let opts = run_options(chunk_size, buffer, workers);
        py.detach(|| ac::voxelize_catalog(&c, &s, z_range, &opts)).map_err(err)?
    } else {
        let (xyz, attrs): (PyReadonlyArray2<f64>, Option<Bound<'_, PyDict>>) = source.extract()?;
        let c = cloud_from_py(xyz, attrs.as_ref())?;
        let b = bounds.map(|b| [b.0, b.1, b.2, b.3, b.4, b.5]);
        py.detach(|| ac::voxelize_cloud(&c, &s, b)).map_err(err)?
    };
    voxels_to_py(py, &v)
}

fn catalog_voxels(origin: (f64, f64, f64), voxel_size: f64, shape: (usize, usize, usize)) -> CatalogVoxels {
    CatalogVoxels { origin: [origin.0, origin.1, origin.2], voxel_size, shape: [shape.2, shape.1, shape.0], ..Default::default() }
}

/// Layer means of `values` by height above the DTM; `shape` is `(nz, ny, nx)`.
#[pyfunction]
#[pyo3(signature = (origin, voxel_size, values, beams, dtm, bin_size, min_beams))]
fn als_voxel_height_profile<'py>(py: Python<'py>, origin: (f64, f64, f64), voxel_size: f64, values: PyReadonlyArray3<f64>, beams: PyReadonlyArray3<f64>, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, bin_size: f64, min_beams: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let s = values.shape();
    let v = catalog_voxels(origin, voxel_size, (s[0], s[1], s[2]));
    let dtm = dtm.map(|(d, x, y, r)| raster_from_py(d, x, y, r));
    let (vals, b): (Vec<f64>, Vec<f64>) = (values.as_array().iter().cloned().collect(), beams.as_array().iter().cloned().collect());
    let (h, m) = ac::height_profile(&v, &vals, &b, dtm.as_ref(), bin_size, min_beams).map_err(err)?;
    Ok((h.into_pyarray(py), m.into_pyarray(py)))
}

/// Column sums of `values * voxel_size` above `min_height`; `(ny, nx)`.
#[pyfunction]
#[pyo3(signature = (origin, voxel_size, values, beams, dtm, min_height, min_beams))]
fn als_voxel_column_sums<'py>(py: Python<'py>, origin: (f64, f64, f64), voxel_size: f64, values: PyReadonlyArray3<f64>, beams: PyReadonlyArray3<f64>, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, min_height: f64, min_beams: f64) -> PyResult<Bound<'py, PyAny>> {
    let s = values.shape();
    let v = catalog_voxels(origin, voxel_size, (s[0], s[1], s[2]));
    let dtm = dtm.map(|(d, x, y, r)| raster_from_py(d, x, y, r));
    let (vals, b): (Vec<f64>, Vec<f64>) = (values.as_array().iter().cloned().collect(), beams.as_array().iter().cloned().collect());
    let sums = py.detach(|| ac::column_sums(&v, &vals, &b, dtm.as_ref(), min_height, min_beams)).map_err(err)?;
    Ok(PyArray1::from_vec(py, sums).reshape([s[1], s[2]])?.into_any())
}

#[pyfunction]
fn als_week_seconds(t: f64) -> f64 {
    ac::week_seconds(t)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(als_trajectory_check, m)?,
        wrap_pyfunction!(als_trajectory_positions, m)?,
        wrap_pyfunction!(als_trajectory_attitude, m)?,
        wrap_pyfunction!(als_read_sbet, m)?,
        wrap_pyfunction!(als_write_sbet, m)?,
        wrap_pyfunction!(als_read_table, m)?,
        wrap_pyfunction!(als_column_role, m)?,
        wrap_pyfunction!(als_project_geographic, m)?,
        wrap_pyfunction!(als_estimate_trajectory, m)?,
        wrap_pyfunction!(als_estimate_trajectory_catalog, m)?,
        wrap_pyfunction!(als_pulses, m)?,
        wrap_pyfunction!(als_profile_cloud, m)?,
        wrap_pyfunction!(als_profile_catalog, m)?,
        wrap_pyfunction!(als_profile_products, m)?,
        wrap_pyfunction!(als_profile_pooled, m)?,
        wrap_pyfunction!(als_profile_metrics, m)?,
        wrap_pyfunction!(als_profile_area_metrics, m)?,
        wrap_pyfunction!(als_ray_voxelize, m)?,
        wrap_pyfunction!(als_voxel_height_profile, m)?,
        wrap_pyfunction!(als_voxel_column_sums, m)?,
        wrap_pyfunction!(als_week_seconds, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
