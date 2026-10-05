// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::als::tiles_trees: tiled segmentation and tree stores.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::path::PathBuf;

use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::als::tiles::RunInfo;
use sylva_rs::als::tiles_trees::{self as tt, MergeSettings, TreeEntry, TreeTiling};
use sylva_rs::trees::prune::PruneParams;
use sylva_rs::trees::SegmentParams;

use crate::als_py::catalog_from_py;
use crate::{attr_from_py, cloud_to_py, err, tree_to_py, trees_from_py};

fn info_to_py<'py>(py: Python<'py>, i: &RunInfo) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("chunks", i.chunks)?;
    d.set_item("max_points", i.max_points)?;
    d.set_item("points_read", i.points_read)?;
    d.set_item("rereads", i.rereads)?;
    d.set_item("widened_points", i.widened_points)?;
    Ok(d)
}

fn origin(v: &Bound<'_, PyAny>) -> PyResult<Option<[f64; 3]>> {
    if v.is_none() {
        return Ok(None);
    }
    let o: (f64, f64, f64) = v.extract()?;
    Ok(Some([o.0, o.1, o.2]))
}

/// Graph settings from a dict of `trees.segment_trees` keywords, over `p`.
fn graph_params(mut p: SegmentParams, d: &Bound<'_, PyDict>, what: &str) -> PyResult<SegmentParams> {
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "k" => p.k = v.extract()?,
            "max_edge" => p.max_edge = v.extract()?,
            "voxel_size" => p.voxel_size = v.extract()?,
            "seed_height" => p.seed_height = v.extract()?,
            "seed_radius" => p.seed_radius = v.extract()?,
            "seed_ring" => p.seed_ring = v.extract()?,
            "power" => p.power = v.extract()?,
            "angle_penalty" => p.angle_penalty = v.extract()?,
            "cut_above_ground" => p.cut_above_ground = v.extract()?,
            "voxel_origin" => p.voxel_origin = origin(&v)?,
            _ if what == "merge" => return Err(PyValueError::new_err(format!("unknown merge_branches parameter {key:?}"))),
            "gravity" => p.gravity = v.extract()?,
            "height_prior" => p.height_prior = v.extract()?,
            "height_prior_radius" => p.height_prior_radius = v.extract()?,
            "height_prior_power" => p.height_prior_power = v.extract()?,
            "low_height" => p.low_height = v.extract()?,
            "low_radius" => p.low_radius = v.extract()?,
            "wood_costs" => p.wood_costs = v.extract()?,
            "wood_k" => p.wood_k = v.extract()?,
            "wood_threshold" => p.wood_threshold = v.extract()?,
            "understorey_height" => p.understorey_height = v.extract()?,
            "understorey_band" => p.understorey_band = v.extract()?,
            _ => return Err(PyValueError::new_err(format!("unknown segment_trees parameter {key:?}"))),
        }
    }
    Ok(p)
}

/// Merge settings from a dict of `trees.merge_branches` keywords, with its
/// defaults (those of the `merge_branches` binding).
fn merge_settings(d: &Bound<'_, PyDict>) -> PyResult<MergeSettings> {
    let graph = SegmentParams { k: 10, max_edge: 1.0, voxel_size: 0.1, seed_height: 1.5, seed_radius: 0.5, seed_ring: true, power: 3.0, angle_penalty: true, gravity: 0.0, cut_above_ground: 0.25, height_prior: false, height_prior_radius: 1.5, low_height: 0.5, low_radius: 1.0, wood_costs: false, wood_k: 20, wood_threshold: 0.9, voxel_origin: Some([0.0; 3]), ..Default::default() };
    let mut m = MergeSettings { graph, ground_height: 0.5, trunk_scale: 1.5, trunk_min: 0.15, search_radius: 6.0 };
    let rest = PyDict::new(d.py());
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "ground_height" => m.ground_height = v.extract()?,
            "trunk_scale" => m.trunk_scale = v.extract()?,
            "trunk_min" => m.trunk_min = v.extract()?,
            "search_radius" => m.search_radius = v.extract()?,
            _ => rest.set_item(k, v)?,
        }
    }
    m.graph = graph_params(m.graph, &rest, "merge")?;
    Ok(m)
}

fn prune_params(d: &Bound<'_, PyDict>) -> PyResult<PruneParams> {
    let mut p = PruneParams::default();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "min_height" => p.min_height = v.extract()?,
            "merge_radius" => p.merge_radius = v.extract()?,
            "max_dbh" => p.max_dbh = v.extract()?,
            "min_quality_short" => p.min_quality_short = v.extract()?,
            "short_slices" => p.short_slices = v.extract()?,
            "min_slenderness" => p.min_slenderness = v.extract()?,
            "slender_min_dbh" => p.slender_min_dbh = v.extract()?,
            other => return Err(PyValueError::new_err(format!("unknown prune_trees parameter {other:?}"))),
        }
    }
    Ok(p)
}

type Segmented<'py> = (Bound<'py, PyList>, Vec<i64>, Vec<String>, u64, Bound<'py, PyDict>);

#[pyfunction]
#[pyo3(signature = (catalog, stems, out_dir, format, height_attr, segment, merge, percentile, prune, buffer, max_buffer, edge_margin, attribute, workers))]
fn tiles_segment_trees<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, stems: &Bound<'_, PyList>, out_dir: PathBuf, format: Option<String>, height_attr: String, segment: &Bound<'_, PyDict>, merge: Option<&Bound<'_, PyDict>>, percentile: f64, prune: Option<&Bound<'_, PyDict>>, buffer: f64, max_buffer: f64, edge_margin: f64, attribute: String, workers: usize) -> PyResult<Segmented<'py>> {
    let c = catalog_from_py(catalog)?;
    let stems = trees_from_py(stems)?;
    let seg = graph_params(SegmentParams { voxel_origin: Some([0.0; 3]), ..Default::default() }, segment, "segment")?;
    let s = TreeTiling { height_attr, segment: seg, merge: merge.map(merge_settings).transpose()?, percentile, prune: prune.map(prune_params).transpose()?, buffer, max_buffer, edge_margin, attribute };
    let r = py.detach(|| tt::segment_trees(&c, &stems, &out_dir, format.as_deref(), &s, workers)).map_err(err)?;
    let list = PyList::empty(py);
    for (i, t) in &r.trees {
        list.append((*i, tree_to_py(py, t)?))?;
    }
    let paths = r.paths.iter().map(|p| p.to_string_lossy().to_string()).collect();
    Ok((list, r.at_edge, paths, r.conflicts, info_to_py(py, &r.info)?))
}

fn entry_to_py<'py>(py: Python<'py>, e: &TreeEntry) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("tree_id", e.tree_id)?;
    d.set_item("n_points", e.n_points)?;
    d.set_item("bounds", e.bounds.to_vec())?;
    d.set_item("parts", e.parts.clone())?;
    Ok(d)
}

#[pyfunction]
fn tiles_split_trees<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, out_dir: PathBuf, attribute: &str, workers: usize) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyDict>)> {
    let c = catalog_from_py(catalog)?;
    let (entries, info) = py.detach(|| tt::split_trees(&c, &out_dir, attribute, workers)).map_err(err)?;
    let list = PyList::empty(py);
    for e in &entries {
        list.append(entry_to_py(py, e)?)?;
    }
    Ok((list, info_to_py(py, &info)?))
}

#[pyfunction]
fn tree_store_read<'py>(py: Python<'py>, store: PathBuf) -> PyResult<(Vec<String>, Bound<'py, PyList>)> {
    let (names, entries) = tt::read_store(&store).map_err(err)?;
    let list = PyList::empty(py);
    for e in &entries {
        list.append(entry_to_py(py, e)?)?;
    }
    Ok((names, list))
}

#[pyfunction]
fn tree_store_tree<'py>(py: Python<'py>, store: PathBuf, tree_id: i64) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let c = py.detach(|| tt::read_tree(&store, tree_id)).map_err(err)?;
    cloud_to_py(py, &c)
}

#[pyfunction]
#[pyo3(signature = (catalog, tree_id, attribute, bounds=None))]
fn tiles_read_tree<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, tree_id: i64, attribute: &str, bounds: Option<(f64, f64, f64, f64)>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let c = catalog_from_py(catalog)?;
    let b = bounds.map(|b| [b.0, b.1, b.2, b.3]);
    let cloud = py.detach(|| tt::read_tree_from_tiles(&c, tree_id, attribute, b)).map_err(err)?;
    cloud_to_py(py, &cloud)
}

#[pyfunction]
fn tree_store_write_values(py: Python<'_>, store: PathBuf, tree_id: i64, name: &str, values: &Bound<'_, PyAny>) -> PyResult<()> {
    let a = attr_from_py(values)?;
    py.detach(|| tt::write_tree_values(&store, tree_id, name, &a)).map_err(err)
}

#[pyfunction]
fn tree_store_has_values(store: PathBuf, tree_id: i64, name: &str) -> PyResult<bool> {
    Ok(tt::read_tree_values(&store, tree_id, name).map_err(err)?.is_some())
}

#[pyfunction]
fn tree_store_values<'py>(py: Python<'py>, store: PathBuf, tree_id: i64, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let a = py.detach(|| tt::read_tree_values(&store, tree_id, name)).map_err(err)?;
    Ok(a.map(|a| crate::attr_to_py(py, &a)))
}

/// The type code of a NumPy dtype name, as the scratch files store it.
fn dtype_code(name: &str) -> PyResult<u8> {
    Ok(match name {
        "float64" => 0,
        "float32" => 1,
        "int64" => 2,
        "int32" => 3,
        "uint32" => 4,
        "uint16" => 5,
        "uint8" => 6,
        "int8" => 7,
        "bool" => 8,
        other => return Err(PyValueError::new_err(format!("values of type {other} cannot be written to tiles"))),
    })
}

#[pyfunction]
fn tiles_write_back<'py>(py: Python<'py>, catalog: &Bound<'_, PyDict>, store: PathBuf, out_dir: PathBuf, attribute: &str, name: &str, default: f64, dtype: &str, format: Option<String>, workers: usize) -> PyResult<(Vec<String>, Bound<'py, PyDict>)> {
    let c = catalog_from_py(catalog)?;
    let code = dtype_code(dtype)?;
    let (paths, info) = py.detach(|| tt::write_back(&c, &store, &out_dir, attribute, name, default, code, format.as_deref(), workers)).map_err(err)?;
    Ok((paths.into_iter().map(|p| p.to_string_lossy().to_string()).collect(), info_to_py(py, &info)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tiles_segment_trees, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_split_trees, m)?)?;
    m.add_function(wrap_pyfunction!(tree_store_read, m)?)?;
    m.add_function(wrap_pyfunction!(tree_store_tree, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_read_tree, m)?)?;
    m.add_function(wrap_pyfunction!(tree_store_write_values, m)?)?;
    m.add_function(wrap_pyfunction!(tree_store_has_values, m)?)?;
    m.add_function(wrap_pyfunction!(tree_store_values, m)?)?;
    m.add_function(wrap_pyfunction!(tiles_write_back, m)?)?;
    Ok(())
}
