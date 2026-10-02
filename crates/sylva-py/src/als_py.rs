// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::als, sylva_rs::als_ops and sylva_rs::synthetic_als.
//!
//! A catalogue crosses the boundary as a dict of per-tile lists (what
//! `sylva.als.Catalog._core` builds) and a chunk as a dict of its fields.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::als::{self, Catalog, Chunk, Layout, Tile};
use sylva_rs::als_ops::{self, Decimation, DtmMethod, GroundMethod, Heights, NoiseMethod, RunOptions};
use sylva_rs::ground::{CsfParams, PmfParams};
use sylva_rs::interpolate::{GridMethod, GridParams};
use sylva_rs::io::las::LasWriteOptions;
use sylva_rs::synthetic_als::{self, FlightParams, ScanPattern, Trajectory};

use crate::{cloud_from_py, cloud_to_py, err, raster_from_py, raster_to_py, xyz_to_py};

fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("catalogue is missing {key:?}")))?.extract::<T>().map_err(|e| {
        let e: PyErr = e.into();
        PyValueError::new_err(format!("catalogue field {key:?}: {e}"))
    })
}

pub(crate) fn catalog_from_py(d: &Bound<'_, PyDict>) -> PyResult<Catalog> {
    let paths: Vec<PathBuf> = get(d, "paths")?;
    let bounds: Vec<(f64, f64, f64, f64, f64, f64)> = get(d, "bounds")?;
    let n_points: Vec<u64> = get(d, "n_points")?;
    let point_format: Vec<u8> = get(d, "point_format")?;
    let version: Vec<(u8, u8)> = get(d, "version")?;
    let crs: Vec<Option<String>> = get(d, "crs")?;
    let scale: Vec<(f64, f64, f64)> = get(d, "scale")?;
    let offset: Vec<(f64, f64, f64)> = get(d, "offset")?;
    let spatial_index: Vec<bool> = get(d, "spatial_index")?;
    let file_size: Vec<u64> = get(d, "file_size")?;
    let n = paths.len();
    if [bounds.len(), n_points.len(), point_format.len(), version.len(), crs.len(), scale.len(), offset.len(), spatial_index.len(), file_size.len()].iter().any(|&k| k != n) {
        return Err(PyValueError::new_err("catalogue fields differ in length"));
    }
    let tiles = (0..n)
        .map(|i| {
            let b = bounds[i];
            Tile {
                path: paths[i].clone(),
                bounds: [b.0, b.1, b.2, b.3, b.4, b.5],
                n_points: n_points[i],
                point_format: point_format[i],
                version: version[i],
                crs: crs[i].clone(),
                scale: [scale[i].0, scale[i].1, scale[i].2],
                offset: [offset[i].0, offset[i].1, offset[i].2],
                spatial_index: spatial_index[i],
                file_size: file_size[i],
            }
        })
        .collect();
    let missing: Vec<PathBuf> = get(d, "missing")?;
    let unreadable: Vec<(PathBuf, String)> = get(d, "unreadable")?;
    Ok(Catalog { tiles, missing, unreadable })
}

fn catalog_to_py<'py>(py: Python<'py>, c: &Catalog) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let t = &c.tiles;
    d.set_item("paths", t.iter().map(|t| t.path.to_string_lossy().to_string()).collect::<Vec<_>>())?;
    d.set_item("bounds", t.iter().map(|t| (t.bounds[0], t.bounds[1], t.bounds[2], t.bounds[3], t.bounds[4], t.bounds[5])).collect::<Vec<_>>())?;
    d.set_item("n_points", t.iter().map(|t| t.n_points).collect::<Vec<_>>())?;
    d.set_item("point_format", t.iter().map(|t| t.point_format).collect::<Vec<_>>())?;
    d.set_item("version", t.iter().map(|t| t.version).collect::<Vec<_>>())?;
    d.set_item("crs", t.iter().map(|t| t.crs.clone()).collect::<Vec<_>>())?;
    d.set_item("scale", t.iter().map(|t| (t.scale[0], t.scale[1], t.scale[2])).collect::<Vec<_>>())?;
    d.set_item("offset", t.iter().map(|t| (t.offset[0], t.offset[1], t.offset[2])).collect::<Vec<_>>())?;
    d.set_item("spatial_index", t.iter().map(|t| t.spatial_index).collect::<Vec<_>>())?;
    d.set_item("file_size", t.iter().map(|t| t.file_size).collect::<Vec<_>>())?;
    d.set_item("missing", c.missing.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>())?;
    d.set_item("unreadable", c.unreadable.iter().map(|(p, m)| (p.to_string_lossy().to_string(), m.clone())).collect::<Vec<_>>())?;
    Ok(d)
}

pub(crate) fn chunk_to_py<'py>(py: Python<'py>, c: &Chunk) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("index", c.index)?;
    d.set_item("core", (c.core[0], c.core[1], c.core[2], c.core[3]))?;
    d.set_item("outer", (c.outer[0], c.outer[1], c.outer[2], c.outer[3]))?;
    d.set_item("own", c.own)?;
    d.set_item("files", c.files.clone())?;
    d.set_item("est_points", c.est_points)?;
    d.set_item("name", c.name.clone())?;
    Ok(d)
}

pub(crate) fn chunk_from_py(d: &Bound<'_, PyDict>) -> PyResult<Chunk> {
    let b4 = |k: &str| -> PyResult<[f64; 4]> {
        let v: (f64, f64, f64, f64) = get(d, k)?;
        Ok([v.0, v.1, v.2, v.3])
    };
    Ok(Chunk { index: get(d, "index")?, core: b4("core")?, outer: b4("outer")?, own: get(d, "own")?, files: get(d, "files")?, est_points: get(d, "est_points")?, name: get(d, "name")? })
}

pub(crate) fn layout(chunk_size: Option<f64>, origin: Option<(f64, f64)>) -> Layout {
    match chunk_size {
        Some(size) => Layout::Grid { size, origin },
        None => Layout::Tiles,
    }
}

pub(crate) fn run_options(chunk_size: Option<f64>, buffer: f64, workers: usize) -> RunOptions {
    RunOptions { layout: layout(chunk_size, None), buffer, workers }
}

fn paths_to_py(paths: Vec<PathBuf>) -> Vec<String> {
    paths.into_iter().map(|p| p.to_string_lossy().to_string()).collect()
}

/// Read the headers of `paths` into a catalogue dict.
#[pyfunction]
fn als_catalog<'py>(py: Python<'py>, paths: Vec<PathBuf>) -> PyResult<Bound<'py, PyDict>> {
    let c = py.detach(|| Catalog::open(&paths));
    catalog_to_py(py, &c)
}

#[pyfunction]
fn als_issues(py: Python<'_>, catalog: &Bound<'_, PyDict>, tolerance: f64) -> PyResult<Vec<(String, String)>> {
    let c = catalog_from_py(catalog)?;
    Ok(py.detach(|| c.issues(tolerance)).into_iter().map(|i| (i.kind, i.message)).collect())
}

#[pyfunction]
fn als_report(py: Python<'_>, catalog: &Bound<'_, PyDict>, tolerance: f64) -> PyResult<String> {
    let c = catalog_from_py(catalog)?;
    Ok(py.detach(|| c.report(tolerance)))
}

#[pyfunction]
fn als_overlaps(catalog: &Bound<'_, PyDict>, tolerance: f64) -> PyResult<Vec<(usize, usize, f64)>> {
    Ok(catalog_from_py(catalog)?.overlaps(tolerance))
}

#[pyfunction]
fn als_gaps(catalog: &Bound<'_, PyDict>, tolerance: f64) -> PyResult<Vec<((f64, f64, f64, f64), f64)>> {
    Ok(catalog_from_py(catalog)?.gaps(tolerance).into_iter().map(|(b, a)| ((b[0], b[1], b[2], b[3]), a)).collect())
}

#[pyfunction]
#[pyo3(signature = (catalog, chunk_size, buffer, origin=None))]
fn als_plan<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, chunk_size: Option<f64>, buffer: f64, origin: Option<(f64, f64)>) -> PyResult<Bound<'py, PyList>> {
    let c = catalog_from_py(catalog)?;
    let chunks = als::plan(&c, layout(chunk_size, origin), buffer).map_err(err)?;
    let out = PyList::empty(py);
    for ch in &chunks {
        out.append(chunk_to_py(py, ch)?)?;
    }
    Ok(out)
}

#[pyfunction]
fn als_workers(est_points: Vec<u64>, workers: usize, bytes_per_point: u64) -> PyResult<usize> {
    als::workers_for_estimates(&est_points, workers, bytes_per_point).map_err(err)
}

/// A chunk's points and its buffer flags.
#[pyfunction]
fn als_read_chunk<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, chunk: &Bound<'_, PyDict>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>, Bound<'py, PyArray1<bool>>)> {
    let (c, ch) = (catalog_from_py(catalog)?, chunk_from_py(chunk)?);
    let data = py.detach(|| als::read_chunk(&c, &ch)).map_err(err)?;
    let (xyz, attrs) = cloud_to_py(py, &data.cloud)?;
    Ok((xyz, attrs, data.buffer.into_pyarray(py)))
}

#[pyfunction]
fn als_read_region<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, bounds: (f64, f64, f64, f64)) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let c = catalog_from_py(catalog)?;
    let cloud = py.detach(|| als::read_region(&c, [bounds.0, bounds.1, bounds.2, bounds.3])).map_err(err)?;
    cloud_to_py(py, &cloud)
}

/// The catalogue grid at `resolution`, as an all-NaN raster dict.
#[pyfunction]
fn als_grid<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, resolution: f64) -> PyResult<Bound<'py, PyDict>> {
    let c = catalog_from_py(catalog)?;
    raster_to_py(py, &als::catalog_grid(&c, resolution).map_err(err)?)
}

/// Join `(data, xmin, ymin, resolution, core)` rasters onto the catalogue grid.
#[pyfunction]
fn als_mosaic<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, resolution: f64, parts: Vec<(PyReadonlyArray2<f64>, f64, f64, f64, (f64, f64, f64, f64))>) -> PyResult<Bound<'py, PyDict>> {
    let c = catalog_from_py(catalog)?;
    let grid = als::catalog_grid(&c, resolution).map_err(err)?;
    let parts: Vec<(sylva_rs::Raster, [f64; 4])> = parts.into_iter().map(|(d, x, y, r, b)| (raster_from_py(d, x, y, r), [b.0, b.1, b.2, b.3])).collect();
    let m = py.detach(|| als::mosaic(&grid, &parts)).map_err(err)?;
    raster_to_py(py, &m)
}

/// Where a chunk's output goes (refusing to overwrite a tile of the catalogue).
#[pyfunction]
fn als_output_path(catalog: &Bound<'_, PyDict>, out_dir: PathBuf, name: &str, ext: &str) -> PyResult<String> {
    let c = catalog_from_py(catalog)?;
    Ok(als::output_path(&c, &out_dir, name, ext).map_err(err)?.to_string_lossy().to_string())
}

/// Write a cloud in the format, scale and CRS of tile `like`.
#[pyfunction]
#[pyo3(signature = (catalog, like, path, xyz, attrs=None))]
fn als_write_like(py: Python<'_>, catalog: &Bound<'_, PyDict>, like: usize, path: PathBuf, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    let c = catalog_from_py(catalog)?;
    let tile = c.tiles.get(like).ok_or_else(|| PyValueError::new_err(format!("no tile {like}")))?.clone();
    let cloud = cloud_from_py(xyz, attrs)?;
    py.detach(|| als::write_like(&cloud, &path, &tile)).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (catalog, out_dir, method, last_returns, format, chunk_size, buffer, workers, cloth_resolution, rigidness, class_threshold, iterations, time_step, cell_size, max_window, slope, initial_distance, max_distance))]
fn als_classify_ground(py: Python<'_>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, method: &str, last_returns: bool, format: Option<String>, chunk_size: Option<f64>, buffer: f64, workers: usize, cloth_resolution: f64, rigidness: usize, class_threshold: f64, iterations: usize, time_step: f64, cell_size: f64, max_window: f64, slope: f64, initial_distance: f64, max_distance: f64) -> PyResult<Vec<String>> {
    let c = catalog_from_py(catalog)?;
    let m = match method {
        "csf" => GroundMethod::Csf(CsfParams { cloth_resolution, rigidness, class_threshold, iterations, time_step }),
        "pmf" => GroundMethod::Pmf(PmfParams { cell_size, max_window, slope, initial_distance, max_distance }),
        other => return Err(PyValueError::new_err(format!("unknown method {other:?}; expected 'csf' or 'pmf'"))),
    };
    let opts = run_options(chunk_size, buffer, workers);
    Ok(paths_to_py(py.detach(|| als_ops::classify_ground(&c, &out_dir, &m, last_returns, format.as_deref(), &opts)).map_err(err)?))
}

fn dtm_method(method: &str, power: f64, k: usize, max_distance: Option<f64>) -> PyResult<DtmMethod> {
    Ok(match method {
        "lowest" => DtmMethod::Lowest,
        other => DtmMethod::Grid(GridParams { method: GridMethod::parse(other).map_err(|_| PyValueError::new_err(format!("unknown method {other:?}; expected 'lowest', 'tin', 'natural' or 'idw'")))?, power, k, max_distance }),
    })
}

#[pyfunction]
#[pyo3(signature = (catalog, resolution, method, power, k, max_distance, chunk_size, buffer, workers))]
fn als_dtm<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, resolution: f64, method: &str, power: f64, k: usize, max_distance: Option<f64>, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let c = catalog_from_py(catalog)?;
    let m = dtm_method(method, power, k, max_distance)?;
    let opts = run_options(chunk_size, buffer, workers);
    raster_to_py(py, &py.detach(|| als_ops::dtm(&c, resolution, &m, &opts)).map_err(err)?)
}

pub(crate) fn heights(mode: &str, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, dtm_resolution: f64) -> PyResult<Heights> {
    Ok(match (mode, dtm) {
        ("z", _) => Heights::Z,
        ("auto", _) => Heights::Auto { resolution: dtm_resolution },
        ("dtm", Some((d, x, y, r))) => Heights::Dtm(raster_from_py(d, x, y, r)),
        (other, _) => return Err(PyValueError::new_err(format!("unknown height source {other:?}"))),
    })
}

#[pyfunction]
#[pyo3(signature = (catalog, resolution, mode, dtm, dtm_resolution, min_height, drop_noise, chunk_size, buffer, workers))]
fn als_chm<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, resolution: f64, mode: &str, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, dtm_resolution: f64, min_height: f64, drop_noise: bool, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let c = catalog_from_py(catalog)?;
    let h = heights(mode, dtm, dtm_resolution)?;
    let opts = run_options(chunk_size, buffer, workers);
    raster_to_py(py, &py.detach(|| als_ops::chm(&c, resolution, &h, min_height, drop_noise, &opts)).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (catalog, out_dir, mode, dtm, dtm_resolution, replace_z, format, chunk_size, buffer, workers))]
fn als_normalize(py: Python<'_>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, mode: &str, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, dtm_resolution: f64, replace_z: bool, format: Option<String>, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Vec<String>> {
    let c = catalog_from_py(catalog)?;
    let h = heights(mode, dtm, dtm_resolution)?;
    let opts = run_options(chunk_size, buffer, workers);
    Ok(paths_to_py(py.detach(|| als_ops::normalize(&c, &out_dir, &h, replace_z, format.as_deref(), &opts)).map_err(err)?))
}

#[pyfunction]
#[pyo3(signature = (catalog, out_dir, method, k, std_ratio, radius, min_neighbors, classify, format, chunk_size, buffer, workers))]
fn als_filter(py: Python<'_>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, method: &str, k: usize, std_ratio: f64, radius: f64, min_neighbors: usize, classify: bool, format: Option<String>, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Vec<String>> {
    let c = catalog_from_py(catalog)?;
    let m = match method {
        "sor" => NoiseMethod::Sor { k, std_ratio },
        "ror" => NoiseMethod::Ror { radius, min_neighbors },
        other => return Err(PyValueError::new_err(format!("unknown method {other:?}; expected 'sor' or 'ror'"))),
    };
    let opts = run_options(chunk_size, buffer, workers);
    Ok(paths_to_py(py.detach(|| als_ops::filter_noise(&c, &out_dir, &m, classify, format.as_deref(), &opts)).map_err(err)?))
}

#[pyfunction]
#[pyo3(signature = (catalog, out_dir, size, buffer, origin, format, workers))]
fn als_retile(py: Python<'_>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, size: f64, buffer: f64, origin: Option<(f64, f64)>, format: Option<String>, workers: usize) -> PyResult<Vec<String>> {
    let c = catalog_from_py(catalog)?;
    Ok(paths_to_py(py.detach(|| als_ops::retile(&c, &out_dir, size, buffer, origin, format.as_deref(), workers)).map_err(err)?))
}

#[pyfunction]
#[pyo3(signature = (catalog, out_dir, method, fraction, seed, size, format, chunk_size, workers))]
fn als_decimate(py: Python<'_>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, method: &str, fraction: f64, seed: u64, size: f64, format: Option<String>, chunk_size: Option<f64>, workers: usize) -> PyResult<Vec<String>> {
    let c = catalog_from_py(catalog)?;
    let m = match method {
        "random" => Decimation::Random { fraction, seed },
        "voxel" => Decimation::Voxel { size },
        "highest" => Decimation::Highest { size },
        other => return Err(PyValueError::new_err(format!("unknown method {other:?}; expected 'random', 'voxel' or 'highest'"))),
    };
    let opts = run_options(chunk_size, 0.0, workers);
    Ok(paths_to_py(py.detach(|| als_ops::decimate(&c, &out_dir, &m, format.as_deref(), &opts)).map_err(err)?))
}

#[pyfunction]
#[pyo3(signature = (xyz, attrs, out_dir, size, origin, format, point_format, scale, epsg))]
fn als_write_tiles(py: Python<'_>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, out_dir: PathBuf, size: f64, origin: Option<(f64, f64)>, format: &str, point_format: u8, scale: f64, epsg: Option<u16>) -> PyResult<Vec<(String, usize)>> {
    let cloud = cloud_from_py(xyz, attrs)?;
    #[allow(clippy::needless_update)]
    let opts = LasWriteOptions { point_format, scale, ..Default::default() };
    let out = py.detach(|| als_ops::write_tiles(&cloud, &out_dir, size, origin, format, &opts, epsg)).map_err(err)?;
    Ok(out.into_iter().map(|(p, n)| (p.to_string_lossy().to_string(), n)).collect())
}

fn trajectory_to_py<'py>(py: Python<'py>, t: &Trajectory) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("time", t.time.clone().into_pyarray(py))?;
    d.set_item("x", t.x.clone().into_pyarray(py))?;
    d.set_item("y", t.y.clone().into_pyarray(py))?;
    d.set_item("z", t.z.clone().into_pyarray(py))?;
    d.set_item("roll", t.roll.clone().into_pyarray(py))?;
    d.set_item("pitch", t.pitch.clone().into_pyarray(py))?;
    d.set_item("heading", t.heading.clone().into_pyarray(py))?;
    d.set_item("line", t.line.clone().into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (xyz, attrs, altitude, speed, line_spacing, heading, scan_pattern, scan_angle, scan_rate, pulse_rate, divergence, footprint_samples, max_returns, min_separation, detection_threshold, range_noise, attitude, target_radius, terrain_slope, bounds, clip, margin, turn_time, start_time, trajectory_rate, seed))]
fn synthetic_als_flight<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, altitude: f64, speed: f64, line_spacing: f64, heading: f64, scan_pattern: &str, scan_angle: f64, scan_rate: f64, pulse_rate: f64, divergence: f64, footprint_samples: usize, max_returns: usize, min_separation: f64, detection_threshold: f64, range_noise: f64, attitude: (f64, f64, f64), target_radius: f64, terrain_slope: f64, bounds: Option<(f64, f64, f64, f64)>, clip: bool, margin: f64, turn_time: f64, start_time: f64, trajectory_rate: f64, seed: u64) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>, Bound<'py, PyDict>, u64)> {
    let scene = cloud_from_py(xyz, attrs)?;
    let p = FlightParams {
        altitude,
        speed,
        line_spacing,
        heading,
        pattern: ScanPattern::parse(scan_pattern).map_err(err)?,
        scan_angle,
        scan_rate,
        pulse_rate,
        divergence_mrad: divergence,
        footprint_samples,
        max_returns,
        min_separation,
        detection_threshold,
        range_noise,
        attitude: [attitude.0, attitude.1, attitude.2],
        target_radius,
        terrain_slope,
        bounds: bounds.map(|b| [b.0, b.1, b.2, b.3]),
        clip,
        margin,
        turn_time,
        start_time,
        trajectory_rate,
        seed,
    };
    let f = py.detach(|| synthetic_als::fly(&scene, &p)).map_err(err)?;
    let (xyz, attrs) = cloud_to_py(py, &f.points)?;
    Ok((xyz, attrs, trajectory_to_py(py, &f.trajectory)?, f.n_pulses))
}

/// Sensor positions at `times` from a trajectory's time, x, y, z and line columns.
#[pyfunction]
fn synthetic_trajectory_positions<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>, z: PyReadonlyArray1<f64>, line: PyReadonlyArray1<u16>, times: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t = Trajectory { time: time.as_array().to_vec(), x: x.as_array().to_vec(), y: y.as_array().to_vec(), z: z.as_array().to_vec(), line: line.as_array().to_vec(), ..Default::default() };
    if [t.x.len(), t.y.len(), t.z.len(), t.line.len()].iter().any(|&n| n != t.time.len()) {
        return Err(PyValueError::new_err("trajectory columns differ in length"));
    }
    if !t.time.windows(2).all(|w| w[0] <= w[1]) {
        return Err(PyValueError::new_err("trajectory times must be sorted"));
    }
    let times = times.as_array().to_vec();
    Ok(xyz_to_py(py, &py.detach(|| t.positions(&times))))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(als_catalog, m)?,
        wrap_pyfunction!(als_issues, m)?,
        wrap_pyfunction!(als_report, m)?,
        wrap_pyfunction!(als_overlaps, m)?,
        wrap_pyfunction!(als_gaps, m)?,
        wrap_pyfunction!(als_plan, m)?,
        wrap_pyfunction!(als_workers, m)?,
        wrap_pyfunction!(als_read_chunk, m)?,
        wrap_pyfunction!(als_read_region, m)?,
        wrap_pyfunction!(als_grid, m)?,
        wrap_pyfunction!(als_mosaic, m)?,
        wrap_pyfunction!(als_output_path, m)?,
        wrap_pyfunction!(als_write_like, m)?,
        wrap_pyfunction!(als_classify_ground, m)?,
        wrap_pyfunction!(als_dtm, m)?,
        wrap_pyfunction!(als_chm, m)?,
        wrap_pyfunction!(als_normalize, m)?,
        wrap_pyfunction!(als_filter, m)?,
        wrap_pyfunction!(als_retile, m)?,
        wrap_pyfunction!(als_decimate, m)?,
        wrap_pyfunction!(als_write_tiles, m)?,
        wrap_pyfunction!(synthetic_als_flight, m)?,
        wrap_pyfunction!(synthetic_trajectory_positions, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("ALS_BYTES_PER_POINT", als::BYTES_PER_POINT)?;
    Ok(())
}
