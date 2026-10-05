// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::als::tiles: tiles from scans and tiled point operations.
//!
//! A catalogue crosses the boundary as the dict of `sylva.als.Catalog._core`.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::io::las::LasWriteOptions;
use sylva_rs::trees::stems::StemParams;
use sylva_rs::als::tiles::{self, Feature, RunInfo, ScanTiler, ScanTiling};

use crate::als_py::catalog_from_py;
use crate::{cloud_from_py, err, matrix_from_py, tree_to_py, xyz_from_py};

fn info_to_py<'py>(py: Python<'py>, i: &RunInfo) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("chunks", i.chunks)?;
    d.set_item("max_points", i.max_points)?;
    d.set_item("points_read", i.points_read)?;
    d.set_item("rereads", i.rereads)?;
    d.set_item("widened_points", i.widened_points)?;
    Ok(d)
}

fn paths_to_py(paths: Vec<PathBuf>) -> Vec<String> {
    paths.into_iter().map(|p| p.to_string_lossy().to_string()).collect()
}

type Written<'py> = (Vec<String>, Bound<'py, PyDict>);

fn written<'py>(py: Python<'py>, out: (Vec<PathBuf>, RunInfo)) -> PyResult<Written<'py>> {
    Ok((paths_to_py(out.0), info_to_py(py, &out.1)?))
}

/// First point per voxel of a grid with a corner at `origin`, as sorted indices.
#[pyfunction]
fn voxel_downsample_indices_at<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64, origin: (f64, f64, f64)) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let p = xyz_from_py(xyz)?;
    let idx = py.detach(|| sylva_rs::filters::voxel_downsample_indices_at(&p, &[origin.0, origin.1, origin.2], voxel_size));
    Ok(idx.into_pyarray(py))
}

#[pyfunction]
fn tiles_thin<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, voxel_size: f64, origin: (f64, f64, f64), format: Option<String>, workers: usize) -> PyResult<Written<'py>> {
    let c = catalog_from_py(catalog)?;
    let out = py.detach(|| tiles::thin(&c, &out_dir, voxel_size, [origin.0, origin.1, origin.2], format.as_deref(), workers)).map_err(err)?;
    written(py, out)
}

#[pyfunction]
fn tiles_sor<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, k: usize, std_ratio: f64, classify: bool, format: Option<String>, buffer: f64, workers: usize) -> PyResult<Written<'py>> {
    let c = catalog_from_py(catalog)?;
    let out = py.detach(|| tiles::sor(&c, &out_dir, k, std_ratio, classify, format.as_deref(), buffer, workers)).map_err(err)?;
    written(py, out)
}

#[pyfunction]
fn tiles_ror<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, radius: f64, min_neighbors: usize, classify: bool, format: Option<String>, buffer: f64, workers: usize) -> PyResult<Written<'py>> {
    let c = catalog_from_py(catalog)?;
    let out = py.detach(|| tiles::ror(&c, &out_dir, radius, min_neighbors, classify, format.as_deref(), buffer, workers)).map_err(err)?;
    written(py, out)
}

#[pyfunction]
fn tiles_features<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, k: usize, feature: &str, format: Option<String>, buffer: f64, workers: usize) -> PyResult<Written<'py>> {
    let c = catalog_from_py(catalog)?;
    let f = match feature {
        "normals" => Feature::Normals,
        "shape" => Feature::Shape,
        other => return Err(PyValueError::new_err(format!("unknown feature {other:?}; expected 'normals' or 'shape'"))),
    };
    let out = py.detach(|| tiles::features(&c, &out_dir, k, f, format.as_deref(), buffer, workers)).map_err(err)?;
    written(py, out)
}

fn stem_params(d: Option<&Bound<'_, PyDict>>) -> PyResult<StemParams> {
    let mut p = StemParams::default();
    let Some(d) = d else { return Ok(p) };
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        macro_rules! set {
            ($($f:ident),*) => {
                match key.as_str() {
                    $(stringify!($f) => p.$f = v.extract()?,)*
                    other => return Err(PyValueError::new_err(format!("unknown stem detection parameter {other:?}"))),
                }
            };
        }
        set!(slice_min, slice_max, slice_thickness, slice_step, reference_height, min_radius, max_radius, cluster_cell, min_cluster_points, max_cluster_extent, ransac_iterations, ransac_tolerance, max_circles_per_cluster, min_circle_inliers, min_coverage, min_arc_deg, max_circle_rmse, link_radius, link_radius_ratio, min_slices, max_lean_deg, link_radius_abs, prefilter, prefilter_k, prefilter_max_nz, prefilter_max_variation, seed, ransac_block, ransac_presample, recluster_wide, cluster_grid_at_slice_min, band_top_inclusive, shared_rng, taper_weight_power, min_total_points, cluster_seeds);
    }
    Ok(p)
}

#[pyfunction]
#[pyo3(signature = (catalog, height_attr, buffer, workers, params=None))]
fn tiles_detect_stems<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, height_attr: &str, buffer: f64, workers: usize, params: Option<&Bound<'_, PyDict>>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyDict>)> {
    let c = catalog_from_py(catalog)?;
    let p = stem_params(params)?;
    let (found, info) = py.detach(|| tiles::detect_stems(&c, height_attr, &p, buffer, workers)).map_err(err)?;
    let list = PyList::empty(py);
    for s in &found {
        let d = tree_to_py(py, &s.tree)?;
        d.set_item("z_ref", s.z)?;
        d.set_item("axis", s.axis.to_vec())?;
        d.set_item("coverage", s.coverage)?;
        list.append(d)?;
    }
    Ok((list, info_to_py(py, &info)?))
}

/// Tiles built from scans added one at a time (sylva_rs::als::tiles::ScanTiler).
#[pyclass(name = "ScanTiler")]
struct PyScanTiler {
    inner: Option<ScanTiler>,
}

impl PyScanTiler {
    fn tiler(&mut self) -> PyResult<&mut ScanTiler> {
        self.inner.as_mut().ok_or_else(|| PyValueError::new_err("the tiles have already been written"))
    }
}

#[pymethods]
impl PyScanTiler {
    #[new]
    #[pyo3(signature = (out_dir, tile_size, voxel_size, origin, bounds, format, point_format, scale, epsg))]
    fn new(out_dir: PathBuf, tile_size: f64, voxel_size: Option<f64>, origin: (f64, f64, f64), bounds: Option<(f64, f64, f64, f64)>, format: String, point_format: u8, scale: f64, epsg: Option<u16>) -> PyResult<Self> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(PyValueError::new_err(format!("scale must be a positive number, got {scale}")));
        }
        #[allow(clippy::needless_update)]
        let las = LasWriteOptions { point_format, scale, ..Default::default() };
        let params = ScanTiling { tile_size, voxel_size, origin: [origin.0, origin.1, origin.2], bounds: bounds.map(|b| [b.0, b.1, b.2, b.3]), format, las, epsg };
        Ok(PyScanTiler { inner: Some(ScanTiler::new(&out_dir, params).map_err(err)?) })
    }

    /// Add a scan given as arrays; returns the points it contributes.
    #[pyo3(signature = (xyz, attrs=None, matrix=None))]
    fn add_cloud(&mut self, py: Python<'_>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, matrix: Option<PyReadonlyArray2<f64>>) -> PyResult<usize> {
        let cloud = cloud_from_py(xyz, attrs)?;
        let t = matrix_from_py(matrix)?;
        let tiler = self.tiler()?;
        py.detach(|| tiler.add(cloud, t.as_ref())).map_err(err)
    }

    /// Add a scan read from a file; returns the points it contributes.
    #[pyo3(signature = (path, matrix=None))]
    fn add_file(&mut self, py: Python<'_>, path: PathBuf, matrix: Option<PyReadonlyArray2<f64>>) -> PyResult<usize> {
        let t = matrix_from_py(matrix)?;
        let tiler = self.tiler()?;
        py.detach(|| {
            let cloud = sylva_rs::io::read(&path)?;
            tiler.add(cloud, t.as_ref())
        })
        .map_err(err)
    }

    /// Write the tiles; returns `[(path, n_points)]` and the run counts.
    fn finish<'py>(&mut self, py: Python<'py>, workers: usize) -> PyResult<(Vec<(String, usize)>, Bound<'py, PyDict>)> {
        let tiler = self.inner.take().ok_or_else(|| PyValueError::new_err("the tiles have already been written"))?;
        let (w, info) = py.detach(|| tiler.finish(workers)).map_err(err)?;
        Ok((w.into_iter().map(|(p, n)| (p.to_string_lossy().to_string(), n)).collect(), info_to_py(py, &info)?))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(voxel_downsample_indices_at, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_thin, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_sor, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_ror, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_features, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_detect_stems, m)?)?;
    m.add_class::<PyScanTiler>()?;
    Ok(())
}
