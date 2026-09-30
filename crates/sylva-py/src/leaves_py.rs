// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva.leaves: leaf/wood labels, angle distributions, leaf
//! area grids (passed as origin, voxel size and a `(nz, ny, nx)` density),
//! leaf shapes (vertices, faces, length, width), leaf insertion and the OBJ
//! reader and writers.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyFileNotFoundError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::leaf_model::{self as lm, LeafAreaGrid, LeafShape};
use sylva_rs::leaves::{self, LeafAngles};
use sylva_rs::mesh_io::{self, ObjMesh};
use sylva_rs::qsm;

use crate::voxels_py::PyRayVoxels;
use crate::{err, qsm_from_rows, xyz_from_py, xyz_to_py};

type Vector<'py> = Bound<'py, PyArray1<f64>>;
type Matrix<'py> = Bound<'py, PyArray2<f64>>;
type Faces<'py> = Bound<'py, PyArray2<u32>>;
type Grid<'py> = (Vector<'py>, f64, Bound<'py, PyArray3<f64>>);
type Shape<'py> = (Matrix<'py>, Faces<'py>, f64, f64);

fn faces_from_py(f: PyReadonlyArray2<u32>) -> PyResult<Vec<[u32; 3]>> {
    let a = f.as_array();
    if a.ncols() != 3 {
        return Err(PyValueError::new_err("faces must have three columns"));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn faces_to_py<'py>(py: Python<'py>, f: &[[u32; 3]]) -> PyResult<Bound<'py, PyArray2<u32>>> {
    PyArray1::from_vec(py, f.iter().flatten().copied().collect::<Vec<u32>>()).reshape([f.len(), 3])
}

fn grid_from_py(origin: PyReadonlyArray1<f64>, voxel_size: f64, density: PyReadonlyArray3<f64>) -> PyResult<LeafAreaGrid> {
    let o = origin.as_array();
    if o.len() != 3 {
        return Err(PyValueError::new_err("origin must have three values"));
    }
    let d = density.as_array();
    let s = d.shape();
    LeafAreaGrid::new([o[0], o[1], o[2]], voxel_size, [s[0], s[1], s[2]], d.iter().copied().collect()).map_err(err)
}

fn grid_to_py<'py>(py: Python<'py>, g: LeafAreaGrid) -> PyResult<Grid<'py>> {
    Ok((g.origin.to_vec().into_pyarray(py), g.voxel_size, PyArray1::from_vec(py, g.density).reshape(g.shape)?))
}

fn shape_from_py(vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, length: f64, width: f64) -> PyResult<LeafShape> {
    LeafShape::new(Some(xyz_from_py(vertices)?), Some(faces_from_py(faces)?), length, width).map_err(err)
}

fn shape_to_py<'py>(py: Python<'py>, s: &LeafShape) -> PyResult<Shape<'py>> {
    Ok((xyz_to_py(py, &s.vertices), faces_to_py(py, &s.faces)?, s.length, s.width))
}

fn angles_from_py(bin_centres: PyReadonlyArray1<f64>, density: PyReadonlyArray1<f64>) -> LeafAngles {
    LeafAngles { bin_centres: bin_centres.as_array().to_vec(), density: density.as_array().to_vec(), mean: 0.0, std: 0.0, beta_a: 0.0, beta_b: 0.0, chi: 1.0, de_wit: None }
}

// ------------------------------------------------------------ leaf / wood

#[pyfunction]
#[pyo3(signature = (xyz, voxel_size=0.02, k=20, high_threshold=0.85, medium_threshold=0.75, scale_radius=0.1, graph_k=10, max_edge=1.0, base_height=0.25, target_res=0.2, min_passage=3, assign_dist=0.05, assign_scale=0.0, component_res=0.05, component_min=200, sor_k=50, sor_std=1.0, dilate_dist=0.03, passage=true))]
#[allow(clippy::too_many_arguments)]
fn classify_leaf_wood<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64, k: usize, high_threshold: f64, medium_threshold: f64, scale_radius: f64, graph_k: usize, max_edge: f64, base_height: f64, target_res: f64, min_passage: usize, assign_dist: f64, assign_scale: f64, component_res: f64, component_min: usize, sor_k: usize, sor_std: f64, dilate_dist: f64, passage: bool) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    let params = qsm::wood::WoodParams { k, high_threshold, medium_threshold, scale_radius, graph_k, max_edge, base_height, target_res, min_passage, assign_dist, assign_scale, component_res, component_min, sor_k, sor_std, dilate_dist, passage };
    Ok(py.detach(|| leaves::classify_leaf_wood(&p, voxel_size, &params)).into_pyarray(py))
}

/// Graph-based leaf/wood separation; `intervals` and `max_angle` default to
/// the authors' settings for the tree's height.
#[pyfunction]
#[pyo3(signature = (xyz, voxel_size=0.02, graph_k=8, max_edge=1.0, base_height=0.25, intervals=None, max_angle=None, linearity=0.9, circle_error=0.2, min_points=10))]
#[allow(clippy::too_many_arguments)]
fn classify_leaf_wood_gbs<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64, graph_k: usize, max_edge: f64, base_height: f64, intervals: Option<Vec<f64>>, max_angle: Option<f64>, linearity: f64, circle_error: f64, min_points: usize) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    let base = qsm::wood::GbsParams { graph_k, max_edge, base_height, linearity, circle_error, min_points, ..Default::default() };
    let params = lm::gbs_params_for(&p, intervals, max_angle, base);
    if params.intervals.is_empty() || params.intervals.iter().any(|v| v.is_nan() || *v <= 0.0) {
        return Err(PyValueError::new_err("intervals must be positive"));
    }
    Ok(py.detach(|| leaves::classify_leaf_wood_gbs(&p, voxel_size, &params)).into_pyarray(py))
}

type PassageScores<'py> = (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
type GbsScores<'py> = (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// [`classify_leaf_wood`] with the per-point wood confidence and cues:
/// `(mask, confidence, anisotropy, passage share)`.
#[pyfunction]
#[pyo3(signature = (xyz, voxel_size=0.02, k=20, high_threshold=0.85, medium_threshold=0.75, scale_radius=0.1, graph_k=10, max_edge=1.0, base_height=0.25, target_res=0.2, min_passage=3, assign_dist=0.05, assign_scale=0.0, component_res=0.05, component_min=200, sor_k=50, sor_std=1.0, dilate_dist=0.03, passage=true))]
#[allow(clippy::too_many_arguments)]
fn classify_leaf_wood_scores<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64, k: usize, high_threshold: f64, medium_threshold: f64, scale_radius: f64, graph_k: usize, max_edge: f64, base_height: f64, target_res: f64, min_passage: usize, assign_dist: f64, assign_scale: f64, component_res: f64, component_min: usize, sor_k: usize, sor_std: f64, dilate_dist: f64, passage: bool) -> PyResult<PassageScores<'py>> {
    let p = xyz_from_py(xyz)?;
    let params = qsm::wood::WoodParams { k, high_threshold, medium_threshold, scale_radius, graph_k, max_edge, base_height, target_res, min_passage, assign_dist, assign_scale, component_res, component_min, sor_k, sor_std, dilate_dist, passage };
    let (s, conf) = py.detach(|| {
        let s = leaves::classify_leaf_wood_scores(&p, voxel_size, &params);
        let conf = leaves::passage_confidence(&s);
        (s, conf)
    });
    Ok((s.mask.into_pyarray(py), conf.into_pyarray(py), s.anisotropy.into_pyarray(py), s.passage.into_pyarray(py)))
}

/// [`classify_leaf_wood_gbs`] with the per-point wood confidence and the
/// share of shell scales at which each point was wood: `(mask, confidence, votes)`.
#[pyfunction]
#[pyo3(signature = (xyz, voxel_size=0.02, graph_k=8, max_edge=1.0, base_height=0.25, intervals=None, max_angle=None, linearity=0.9, circle_error=0.2, min_points=10))]
#[allow(clippy::too_many_arguments)]
fn classify_leaf_wood_gbs_scores<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64, graph_k: usize, max_edge: f64, base_height: f64, intervals: Option<Vec<f64>>, max_angle: Option<f64>, linearity: f64, circle_error: f64, min_points: usize) -> PyResult<GbsScores<'py>> {
    let p = xyz_from_py(xyz)?;
    let base = qsm::wood::GbsParams { graph_k, max_edge, base_height, linearity, circle_error, min_points, ..Default::default() };
    let params = lm::gbs_params_for(&p, intervals, max_angle, base);
    if params.intervals.is_empty() || params.intervals.iter().any(|v| v.is_nan() || *v <= 0.0) {
        return Err(PyValueError::new_err("intervals must be positive"));
    }
    let (mask, conf, votes) = py.detach(|| {
        let (mask, votes) = leaves::classify_leaf_wood_gbs_scores(&p, voxel_size, &params);
        let conf = leaves::gbs_confidence(&mask, &votes);
        (mask, conf, votes)
    });
    Ok((mask.into_pyarray(py), conf.into_pyarray(py), votes.into_pyarray(py)))
}

// ------------------------------------------------------------ angles

#[pyfunction]
#[pyo3(signature = (xyz, k=12))]
fn leaf_inclinations<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, k: usize) -> PyResult<(Vector<'py>, Matrix<'py>)> {
    let p = xyz_from_py(xyz)?;
    let (incl, normals) = py.detach(|| leaves::inclinations(&p, k));
    Ok((incl.into_pyarray(py), xyz_to_py(py, &normals)))
}

fn leaf_angles_to_py<'py>(py: Python<'py>, a: &LeafAngles) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("bin_centres", a.bin_centres.clone().into_pyarray(py))?;
    d.set_item("density", a.density.clone().into_pyarray(py))?;
    d.set_item("mean", a.mean)?;
    d.set_item("std", a.std)?;
    d.set_item("beta_a", a.beta_a)?;
    d.set_item("beta_b", a.beta_b)?;
    d.set_item("chi", a.chi)?;
    d.set_item("de_wit", a.de_wit)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (inclination, weights=None, n_bins=18))]
fn leaf_angle_distribution<'py>(py: Python<'py>, inclination: PyReadonlyArray1<f64>, weights: Option<PyReadonlyArray1<f64>>, n_bins: usize) -> PyResult<Bound<'py, PyDict>> {
    let incl = inclination.as_array().to_vec();
    let w = weights.map(|w| w.as_array().to_vec());
    if let Some(w) = &w {
        if w.len() != incl.len() {
            return Err(PyValueError::new_err("weights must match inclination"));
        }
    }
    leaf_angles_to_py(py, &leaves::angle_distribution(&incl, w.as_deref(), n_bins))
}

/// A de Wit (1965) distribution by name; KeyError for an unknown one.
#[pyfunction]
#[pyo3(signature = (name="spherical", n_bins=18))]
fn leaf_de_wit<'py>(py: Python<'py>, name: &str, n_bins: usize) -> PyResult<Bound<'py, PyDict>> {
    let a = lm::de_wit(name, n_bins).ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
    leaf_angles_to_py(py, &a)
}

#[pyfunction]
fn leaf_projection_histogram(bin_centres: PyReadonlyArray1<f64>, density: PyReadonlyArray1<f64>, beam_zenith: PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    let a = angles_from_py(bin_centres, density);
    Ok(beam_zenith.as_array().iter().map(|&t| leaves::projection(&a, t)).collect())
}

// ------------------------------------------------------------ leaf area

#[pyfunction]
#[pyo3(signature = (xyz, res=0.01, k=12))]
fn point_leaf_area<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, res: f64, k: usize) -> PyResult<(Matrix<'py>, Vector<'py>, Vector<'py>)> {
    let p = xyz_from_py(xyz)?;
    let (pts, area, incl) = py.detach(|| leaves::point_leaf_area(&p, res, k));
    Ok((xyz_to_py(py, &pts), area.into_pyarray(py), incl.into_pyarray(py)))
}

/// Leaf area density from leaf points: `(origin, voxel_size, density)`.
#[pyfunction]
#[pyo3(signature = (xyz, voxel_size=0.25, res=0.0, k=12))]
fn leaf_area_density<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64, res: f64, k: usize) -> PyResult<Grid<'py>> {
    let p = xyz_from_py(xyz)?;
    let g = py.detach(|| lm::leaf_area_density(&p, voxel_size, res, k)).map_err(err)?;
    grid_to_py(py, g)
}

/// Leaf area per voxel (m2), `(nz, ny, nx)`.
#[pyfunction]
fn leaf_grid_area<'py>(py: Python<'py>, origin: PyReadonlyArray1<f64>, voxel_size: f64, density: PyReadonlyArray3<f64>) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let g = grid_from_py(origin, voxel_size, density)?;
    PyArray1::from_vec(py, g.area()).reshape(g.shape)
}

#[pyfunction]
fn leaf_grid_total_area(origin: PyReadonlyArray1<f64>, voxel_size: f64, density: PyReadonlyArray3<f64>) -> PyResult<f64> {
    Ok(grid_from_py(origin, voxel_size, density)?.total_area())
}

/// The density rescaled to a total leaf area.
#[pyfunction]
fn leaf_grid_scaled<'py>(py: Python<'py>, origin: PyReadonlyArray1<f64>, voxel_size: f64, density: PyReadonlyArray3<f64>, total_area: f64) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let g = grid_from_py(origin, voxel_size, density)?.scaled_to(total_area);
    PyArray1::from_vec(py, g.density).reshape(g.shape)
}

/// `(z, area)` per layer.
#[pyfunction]
fn leaf_grid_profile<'py>(py: Python<'py>, origin: PyReadonlyArray1<f64>, voxel_size: f64, density: PyReadonlyArray3<f64>) -> PyResult<(Vector<'py>, Vector<'py>)> {
    let (z, a) = grid_from_py(origin, voxel_size, density)?.profile();
    Ok((z.into_pyarray(py), a.into_pyarray(py)))
}

/// `(centres, area)` of the voxels holding leaf area.
#[pyfunction]
fn leaf_grid_cells<'py>(py: Python<'py>, origin: PyReadonlyArray1<f64>, voxel_size: f64, density: PyReadonlyArray3<f64>) -> PyResult<(Matrix<'py>, Vector<'py>)> {
    let (c, a) = grid_from_py(origin, voxel_size, density)?.cells();
    Ok((xyz_to_py(py, &c), a.into_pyarray(py)))
}

/// Leaf area density of a ray-traced grid: `(origin, voxel_size, density)`.
#[pyfunction]
fn leaf_grid_from_voxels<'py>(py: Python<'py>, grid: PyRef<'_, PyRayVoxels>, field: &str) -> PyResult<Grid<'py>> {
    grid_to_py(py, LeafAreaGrid::from_voxels(&grid.inner, field).map_err(err)?)
}

// ------------------------------------------------------------ leaf shapes

/// Check a shape; `(vertices, faces)` with the built-in blade's where None.
#[pyfunction]
#[pyo3(signature = (vertices, faces, length, width))]
fn leaf_shape_check<'py>(py: Python<'py>, vertices: Option<PyReadonlyArray2<f64>>, faces: Option<PyReadonlyArray2<u32>>, length: f64, width: f64) -> PyResult<(Matrix<'py>, Faces<'py>)> {
    let v = vertices.map(xyz_from_py).transpose()?;
    let f = faces.map(faces_from_py).transpose()?;
    let s = LeafShape::new(v, f, length, width).map_err(err)?;
    Ok((xyz_to_py(py, &s.vertices), faces_to_py(py, &s.faces)?))
}

#[pyfunction]
fn leaf_shape_area(vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, length: f64, width: f64) -> PyResult<f64> {
    Ok(shape_from_py(vertices, faces, length, width)?.area())
}

/// `(length, width)` giving the shape a one-sided area.
#[pyfunction]
fn leaf_shape_scaled(vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, length: f64, width: f64, area: f64) -> PyResult<(f64, f64)> {
    let s = shape_from_py(vertices, faces, length, width)?.scaled_to(area).map_err(err)?;
    Ok((s.length, s.width))
}

#[pyfunction]
#[pyo3(signature = (vertices, faces, length=None, width=None, normalise=true))]
fn leaf_shape_from_mesh<'py>(py: Python<'py>, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, length: Option<f64>, width: Option<f64>, normalise: bool) -> PyResult<Shape<'py>> {
    let s = LeafShape::from_mesh(xyz_from_py(vertices)?, faces_from_py(faces)?, length, width, normalise).map_err(err)?;
    shape_to_py(py, &s)
}

#[pyfunction]
#[pyo3(signature = (path, length=None, width=None, normalise=true))]
fn leaf_shape_from_obj<'py>(py: Python<'py>, path: PathBuf, length: Option<f64>, width: Option<f64>, normalise: bool) -> PyResult<Shape<'py>> {
    let s = LeafShape::from_obj(&path, length, width, normalise).map_err(|e| match e {
        sylva_rs::Error::Io(io) if io.kind() == std::io::ErrorKind::NotFound => PyFileNotFoundError::new_err(format!("No such file or directory: {:?}", path.display().to_string())),
        e => err(e),
    })?;
    shape_to_py(py, &s)
}

#[pyfunction]
#[pyo3(signature = (length, width, vertices=None, faces=None))]
fn single_leaf_area(length: f64, width: f64, vertices: Option<PyReadonlyArray2<f64>>, faces: Option<PyReadonlyArray2<u32>>) -> PyResult<f64> {
    let shape = match (vertices, faces) {
        (Some(v), Some(f)) => Some(shape_from_py(v, f, length, width)?),
        _ => None,
    };
    lm::single_leaf_area(length, width, shape.as_ref()).map_err(err)
}

// ------------------------------------------------------------ insertion

/// Leaves for a QSM from a grid (`density` given) or a total area.
#[pyfunction]
#[pyo3(signature = (origin, voxel_size, density, total_area, seeds, bin_centres, angle_density, cylinders, vertices, faces, length, width, max_branch_distance=0.5, jitter=0.01, seed=1))]
#[allow(clippy::too_many_arguments)]
fn add_leaves<'py>(py: Python<'py>, origin: PyReadonlyArray1<f64>, voxel_size: f64, density: Option<PyReadonlyArray3<f64>>, total_area: f64, seeds: PyReadonlyArray2<f64>, bin_centres: PyReadonlyArray1<f64>, angle_density: PyReadonlyArray1<f64>, cylinders: PyReadonlyArray2<f64>, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, length: f64, width: f64, max_branch_distance: f64, jitter: f64, seed: u64) -> PyResult<Bound<'py, PyDict>> {
    let grid = density.map(|d| grid_from_py(origin, voxel_size, d)).transpose()?;
    let seeds = xyz_from_py(seeds)?;
    let angles = angles_from_py(bin_centres, angle_density);
    let model = if cylinders.as_array().nrows() > 0 { qsm_from_rows(cylinders)? } else { Default::default() };
    let opts = lm::AddLeaves { shape: shape_from_py(vertices, faces, length, width)?, max_branch_distance, jitter, seed };
    let area = match &grid {
        Some(g) => lm::LeafArea::Grid(g),
        None => lm::LeafArea::Total(total_area),
    };
    let mesh = py.detach(|| lm::add_leaves(area, &seeds, &angles, &model.cylinders, &opts)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("vertices", xyz_to_py(py, &mesh.vertices))?;
    d.set_item("faces", faces_to_py(py, &mesh.faces)?)?;
    d.set_item("centres", xyz_to_py(py, &mesh.centres))?;
    d.set_item("normals", xyz_to_py(py, &mesh.normals))?;
    d.set_item("inclination", mesh.inclination.clone().into_pyarray(py))?;
    d.set_item("cylinder", mesh.cylinder.clone().into_pyarray(py))?;
    d.set_item("leaf_area", mesh.leaf_area)?;
    Ok(d)
}

// ------------------------------------------------------------ OBJ

/// Write `(vertices, faces)` pairs to one OBJ file, one named object each.
#[pyfunction]
fn write_obj(path: PathBuf, meshes: &Bound<'_, PyList>, names: Vec<String>) -> PyResult<()> {
    if names.len() != meshes.len() {
        return Err(PyValueError::new_err("one name per mesh"));
    }
    let mut parts = Vec::new();
    for m in meshes.iter() {
        let (v, f): (PyReadonlyArray2<f64>, PyReadonlyArray2<u32>) = m.extract()?;
        parts.push((xyz_from_py(v)?, faces_from_py(f)?));
    }
    let objs: Vec<ObjMesh> = parts.iter().zip(&names).map(|((v, f), n)| ObjMesh { name: n, vertices: v, faces: f }).collect();
    mesh_io::write_obj(path, &objs).map_err(err)
}

/// Wood cylinders and leaves to one OBJ, as objects `wood` and `leaves`.
#[pyfunction]
#[pyo3(signature = (path, cylinders, vertices, faces, sides=12, contiguous=false))]
fn write_tree_obj(path: PathBuf, cylinders: PyReadonlyArray2<f64>, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, sides: usize, contiguous: bool) -> PyResult<()> {
    let q = qsm_from_rows(cylinders)?;
    lm::write_tree_obj(path, &q, sides, contiguous, &xyz_from_py(vertices)?, &faces_from_py(faces)?).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(classify_leaf_wood, m)?,
        wrap_pyfunction!(classify_leaf_wood_gbs, m)?,
        wrap_pyfunction!(classify_leaf_wood_scores, m)?,
        wrap_pyfunction!(classify_leaf_wood_gbs_scores, m)?,
        wrap_pyfunction!(leaf_inclinations, m)?,
        wrap_pyfunction!(leaf_angle_distribution, m)?,
        wrap_pyfunction!(leaf_de_wit, m)?,
        wrap_pyfunction!(leaf_projection_histogram, m)?,
        wrap_pyfunction!(point_leaf_area, m)?,
        wrap_pyfunction!(leaf_area_density, m)?,
        wrap_pyfunction!(leaf_grid_area, m)?,
        wrap_pyfunction!(leaf_grid_total_area, m)?,
        wrap_pyfunction!(leaf_grid_scaled, m)?,
        wrap_pyfunction!(leaf_grid_profile, m)?,
        wrap_pyfunction!(leaf_grid_cells, m)?,
        wrap_pyfunction!(leaf_grid_from_voxels, m)?,
        wrap_pyfunction!(leaf_shape_check, m)?,
        wrap_pyfunction!(leaf_shape_area, m)?,
        wrap_pyfunction!(leaf_shape_scaled, m)?,
        wrap_pyfunction!(leaf_shape_from_mesh, m)?,
        wrap_pyfunction!(leaf_shape_from_obj, m)?,
        wrap_pyfunction!(single_leaf_area, m)?,
        wrap_pyfunction!(add_leaves, m)?,
        wrap_pyfunction!(write_obj, m)?,
        wrap_pyfunction!(write_tree_obj, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
