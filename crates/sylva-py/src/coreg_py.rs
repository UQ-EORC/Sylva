// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for coregistration's numerical layers: rigid transforms
//! (sylva_rs::coreg::transforms), reflective targets (coreg_reflectors), the
//! pose graph (coreg_posegraph) and joint refinement (coreg_refine).
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use std::path::PathBuf;

use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector6};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::coreg::posegraph as pg;
use sylva_rs::coreg::reflectors as rf;
use sylva_rs::coreg::refine as refine;
use sylva_rs::coreg::transforms as tf;
use sylva_rs::Point;

use crate::{err, xyz_from_py, xyz_to_py};

fn fixed<const R: usize, const C: usize>(a: &PyReadonlyArray2<f64>, what: &str) -> PyResult<nalgebra::SMatrix<f64, R, C>> {
    let v = a.as_array();
    if v.shape() != [R, C] {
        return Err(PyValueError::new_err(format!("{what} must have shape ({R}, {C}), got {:?}", v.shape())));
    }
    Ok(nalgebra::SMatrix::from_fn(|r, c| v[[r, c]]))
}

fn mat4(a: &PyReadonlyArray2<f64>) -> PyResult<Matrix4<f64>> {
    fixed::<4, 4>(a, "a transform")
}

fn mats_from_py<const N: usize>(a: &PyReadonlyArray3<f64>, what: &str) -> PyResult<Vec<nalgebra::SMatrix<f64, N, N>>> {
    let v = a.as_array();
    if v.shape()[1] != N || v.shape()[2] != N {
        return Err(PyValueError::new_err(format!("{what} must have shape (n, {N}, {N})")));
    }
    Ok((0..v.shape()[0]).map(|k| nalgebra::SMatrix::from_fn(|r, c| v[[k, r, c]])).collect())
}

fn mat_to_py<'py, const R: usize, const C: usize>(py: Python<'py>, m: &nalgebra::SMatrix<f64, R, C>) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = (0..R).flat_map(|r| (0..C).map(move |c| m[(r, c)])).collect();
    PyArray1::from_vec(py, flat).reshape([R, C]).expect("reshape")
}

fn mats_to_py<'py>(py: Python<'py>, m: &[Matrix4<f64>]) -> Bound<'py, PyArray3<f64>> {
    let flat: Vec<f64> = m.iter().flat_map(|t| (0..4).flat_map(move |r| (0..4).map(move |c| t[(r, c)]))).collect();
    PyArray1::from_vec(py, flat).reshape([m.len(), 4, 4]).expect("reshape")
}

fn vec_to_py<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, v)
}

// ----------------------------------------------------------------- transforms

#[pyfunction]
fn coreg_skew<'py>(py: Python<'py>, v: [f64; 3]) -> Bound<'py, PyArray2<f64>> {
    mat_to_py(py, &tf::skew(&Vector3::from(v)))
}

#[pyfunction]
fn coreg_so3_exp<'py>(py: Python<'py>, w: [f64; 3]) -> Bound<'py, PyArray2<f64>> {
    mat_to_py(py, &tf::so3_exp(&Vector3::from(w)))
}

#[pyfunction]
fn coreg_so3_log<'py>(py: Python<'py>, r: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r: Matrix3<f64> = fixed::<3, 3>(&r, "a rotation")?;
    Ok(vec_to_py(py, tf::so3_log(&r).as_slice()))
}

#[pyfunction]
fn coreg_rotation_angle(r: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let r: Matrix3<f64> = fixed::<3, 3>(&r, "a rotation")?;
    Ok(tf::so3_log(&r).norm())
}

#[pyfunction]
fn coreg_se3_exp<'py>(py: Python<'py>, xi: [f64; 6]) -> Bound<'py, PyArray2<f64>> {
    mat_to_py(py, &tf::se3_exp(&Vector6::from(xi)))
}

#[pyfunction]
fn coreg_se3_log<'py>(py: Python<'py>, t: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(vec_to_py(py, tf::se3_log(&mat4(&t)?).as_slice()))
}

#[pyfunction]
fn coreg_invert<'py>(py: Python<'py>, t: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(mat_to_py(py, &tf::invert(&mat4(&t)?)))
}

#[pyfunction]
fn coreg_transform_points<'py>(py: Python<'py>, t: PyReadonlyArray2<f64>, points: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (t, p) = (mat4(&t)?, xyz_from_py(points)?);
    Ok(xyz_to_py(py, &py.detach(|| tf::transform_points(&t, &p))))
}

#[pyfunction]
fn coreg_transform_vectors<'py>(py: Python<'py>, t: PyReadonlyArray2<f64>, vectors: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (t, v) = (mat4(&t)?, xyz_from_py(vectors)?);
    Ok(xyz_to_py(py, &py.detach(|| tf::transform_vectors(&t, &v))))
}

#[pyfunction]
fn coreg_yaw_transform<'py>(py: Python<'py>, yaw: f64, tx: f64, ty: f64, tz: f64) -> Bound<'py, PyArray2<f64>> {
    mat_to_py(py, &tf::yaw_transform(yaw, tx, ty, tz))
}

#[pyfunction]
#[pyo3(signature = (source, target, weights=None))]
fn coreg_kabsch<'py>(py: Python<'py>, source: PyReadonlyArray2<f64>, target: PyReadonlyArray2<f64>, weights: Option<PyReadonlyArray1<f64>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (s, t) = (xyz_from_py(source)?, xyz_from_py(target)?);
    let w = weights.map(|w| w.as_array().to_vec());
    Ok(mat_to_py(py, &tf::kabsch(&s, &t, w.as_deref()).map_err(err)?))
}

#[pyfunction]
#[pyo3(signature = (source, target, weights=None))]
fn coreg_kabsch_2d_yaw<'py>(py: Python<'py>, source: PyReadonlyArray2<f64>, target: PyReadonlyArray2<f64>, weights: Option<PyReadonlyArray1<f64>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (s, t) = (xyz_from_py(source)?, xyz_from_py(target)?);
    let w = weights.map(|w| w.as_array().to_vec());
    Ok(mat_to_py(py, &tf::kabsch_2d_yaw(&s, &t, w.as_deref()).map_err(err)?))
}

#[pyfunction]
fn coreg_transform_difference(a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>) -> PyResult<(f64, f64)> {
    Ok(tf::transform_difference(&mat4(&a)?, &mat4(&b)?))
}

// ----------------------------------------------------------------- reflectors

type ReflectorTuple = (f64, f64, f64, f64, f64, i64, String);

fn reflector_tuples(v: Vec<rf::Reflector>) -> Vec<ReflectorTuple> {
    v.into_iter().map(|r| (r.x, r.y, r.z, r.reflectance, r.diameter, r.n_points, r.name)).collect()
}

/// `(x, y, z, reflectance, diameter, n_points, name)` per target.
#[pyfunction]
fn coreg_read_tiepoint_list(path: PathBuf) -> PyResult<Vec<ReflectorTuple>> {
    Ok(reflector_tuples(rf::read_tiepoint_list(path).map_err(err)?))
}

#[pyfunction]
fn coreg_read_reflector_list(path: PathBuf) -> PyResult<Vec<ReflectorTuple>> {
    Ok(reflector_tuples(rf::read_reflector_list(path).map_err(err)?))
}

#[pyfunction]
fn coreg_detect_reflectors(py: Python<'_>, xyz: PyReadonlyArray2<f64>, reflectance: Option<PyReadonlyArray1<f64>>, min_reflectance: f64, cluster_radius: f64, min_points: usize, max_extent: f64) -> PyResult<Vec<ReflectorTuple>> {
    let p = xyz_from_py(xyz)?;
    let r = reflectance.map(|r| r.as_array().to_vec());
    let found = py.detach(|| rf::detect_reflectors(&p, r.as_deref(), min_reflectance, cluster_radius, min_points, max_extent)).map_err(err)?;
    Ok(reflector_tuples(found))
}

/// A dict with `transform`, `n_inliers`, `rmse`, `correspondences` and `success`.
#[pyfunction]
fn coreg_match_reflectors<'py>(py: Python<'py>, source: PyReadonlyArray2<f64>, target: PyReadonlyArray2<f64>, tolerance: f64, min_inliers: i64, distance_tolerance: f64) -> PyResult<Bound<'py, PyDict>> {
    let (s, t) = (xyz_from_py(source)?, xyz_from_py(target)?);
    let m = py.detach(|| rf::match_reflectors(&s, &t, tolerance, min_inliers, distance_tolerance)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("transform", mat_to_py(py, &m.transform))?;
    d.set_item("n_inliers", m.n_inliers)?;
    d.set_item("rmse", m.rmse)?;
    let flat: Vec<i64> = m.correspondences.iter().flat_map(|c| [c[0] as i64, c[1] as i64]).collect();
    d.set_item("correspondences", PyArray1::from_vec(py, flat).reshape([m.correspondences.len(), 2])?)?;
    d.set_item("success", m.success)?;
    Ok(d)
}

// ----------------------------------------------------------------- pose graph

#[pyfunction]
fn coreg_default_information<'py>(py: Python<'py>, rmse: f64, fitness: f64, n_correspondences: i64, extent: f64) -> Bound<'py, PyArray2<f64>> {
    mat_to_py(py, &pg::default_information(rmse, fitness, n_correspondences, extent))
}

#[pyfunction]
fn coreg_adjoint<'py>(py: Python<'py>, t: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(mat_to_py(py, &pg::adjoint(&mat4(&t)?)))
}

#[pyfunction]
fn coreg_plane_edge_information<'py>(py: Python<'py>, hessian: PyReadonlyArray2<f64>, sigma: f64, n: i64, transform: PyReadonlyArray2<f64>, patch_points: f64, min_sigma: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let h: Matrix6<f64> = fixed::<6, 6>(&hessian, "hessian")?;
    Ok(mat_to_py(py, &pg::plane_edge_information(&h, sigma, n, &mat4(&transform)?, patch_points, min_sigma)))
}

fn edges_from_py(n: usize, i: Vec<usize>, j: Vec<usize>, transforms: &PyReadonlyArray3<f64>, information: Option<&PyReadonlyArray3<f64>>, weights: Option<Vec<f64>>) -> PyResult<Vec<pg::Edge>> {
    let t = mats_from_py::<4>(transforms, "transforms")?;
    let info = match information {
        Some(a) => mats_from_py::<6>(a, "information")?,
        None => vec![Matrix6::identity(); t.len()],
    };
    let w = weights.unwrap_or_else(|| vec![1.0; t.len()]);
    if i.len() != t.len() || j.len() != t.len() || info.len() != t.len() || w.len() != t.len() {
        return Err(PyValueError::new_err("one i, j, transform, information and weight per edge"));
    }
    if i.iter().chain(&j).any(|&k| k >= n) {
        return Err(PyValueError::new_err("edge endpoints out of range"));
    }
    Ok((0..t.len()).map(|k| pg::Edge { i: i[k], j: j[k], transform: t[k], information: info[k], weight: w[k] }).collect())
}

fn poses_from_py(poses: &PyReadonlyArray3<f64>, n: usize) -> PyResult<Vec<Matrix4<f64>>> {
    let p = mats_from_py::<4>(poses, "poses")?;
    if p.len() != n {
        return Err(PyValueError::new_err("one pose per node"));
    }
    Ok(p)
}

fn fixed_from_py(nodes: Vec<usize>, poses: &PyReadonlyArray3<f64>, n: usize) -> PyResult<Vec<(usize, Matrix4<f64>)>> {
    let p = mats_from_py::<4>(poses, "fixed poses")?;
    if p.len() != nodes.len() || nodes.iter().any(|&k| k >= n) {
        return Err(PyValueError::new_err("fixed nodes and poses do not match"));
    }
    Ok(nodes.into_iter().zip(p).collect())
}

/// The 6-vector error of edges `(i, j, transform)` under `poses`, one row per edge.
#[pyfunction]
fn coreg_posegraph_residuals<'py>(py: Python<'py>, i: Vec<usize>, j: Vec<usize>, transforms: PyReadonlyArray3<f64>, poses: PyReadonlyArray3<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = mats_from_py::<4>(&poses, "poses")?;
    let edges = edges_from_py(p.len(), i, j, &transforms, None, None)?;
    let flat: Vec<f64> = edges.iter().flat_map(|e| pg::residual(e, &p).iter().copied().collect::<Vec<_>>()).collect();
    PyArray1::from_vec(py, flat).reshape([edges.len(), 6])
}

/// Sum of squared Mahalanobis edge errors.
#[pyfunction]
fn coreg_posegraph_total_error(i: Vec<usize>, j: Vec<usize>, transforms: PyReadonlyArray3<f64>, information: PyReadonlyArray3<f64>, poses: PyReadonlyArray3<f64>) -> PyResult<f64> {
    let p = mats_from_py::<4>(&poses, "poses")?;
    let edges = edges_from_py(p.len(), i, j, &transforms, Some(&information), None)?;
    let all: Vec<usize> = (0..edges.len()).collect();
    Ok(pg::total_error(&edges, &all, &p))
}

#[pyfunction]
fn coreg_posegraph_components(n: usize, i: Vec<usize>, j: Vec<usize>) -> PyResult<Vec<Vec<usize>>> {
    if i.len() != j.len() || i.iter().chain(&j).any(|&k| k >= n) {
        return Err(PyValueError::new_err("edge endpoints out of range"));
    }
    let edges: Vec<pg::Edge> = i.iter().zip(&j).map(|(&a, &b)| pg::Edge { i: a, j: b, transform: Matrix4::identity(), information: Matrix6::identity(), weight: 1.0 }).collect();
    Ok(pg::components(n, &edges))
}

#[pyfunction]
fn coreg_posegraph_initialise<'py>(py: Python<'py>, n: usize, i: Vec<usize>, j: Vec<usize>, transforms: PyReadonlyArray3<f64>, weights: Vec<f64>, reference: usize, fixed_nodes: Vec<usize>, fixed_poses: PyReadonlyArray3<f64>) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let edges = edges_from_py(n, i, j, &transforms, None, Some(weights))?;
    let fixed = fixed_from_py(fixed_nodes, &fixed_poses, n)?;
    if reference >= n {
        return Err(PyValueError::new_err("reference node index out of range"));
    }
    Ok(mats_to_py(py, &pg::initialise(n, &edges, reference, &fixed)))
}

/// Levenberg-Marquardt with outlier rejection: a dict with `poses`,
/// `iterations`, `converged`, `initial_error`, `final_error`,
/// `rejected_edges` and `edge_errors`.
#[pyfunction]
fn coreg_posegraph_optimise<'py>(py: Python<'py>, n: usize, i: Vec<usize>, j: Vec<usize>, transforms: PyReadonlyArray3<f64>, information: PyReadonlyArray3<f64>, weights: Vec<f64>, reference: usize, fixed_nodes: Vec<usize>, fixed_poses: PyReadonlyArray3<f64>, poses: PyReadonlyArray3<f64>, max_iterations: usize, tolerance: f64, huber_delta: f64, reject_outliers: bool, outlier_sigma: f64, max_rejection_passes: i64) -> PyResult<Bound<'py, PyDict>> {
    let edges = edges_from_py(n, i, j, &transforms, Some(&information), Some(weights))?;
    let fixed = fixed_from_py(fixed_nodes, &fixed_poses, n)?;
    let start = poses_from_py(&poses, n)?;
    if reference >= n {
        return Err(PyValueError::new_err("reference node index out of range"));
    }
    let params = pg::OptimiseParams { max_iterations, tolerance, huber_delta, reject_outliers, outlier_sigma, max_rejection_passes };
    let r = py.detach(|| pg::optimise(n, &edges, reference, &fixed, start, &params));
    let d = PyDict::new(py);
    d.set_item("poses", mats_to_py(py, &r.poses))?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    d.set_item("initial_error", r.initial_error)?;
    d.set_item("final_error", r.final_error)?;
    d.set_item("rejected_edges", r.rejected_edges)?;
    d.set_item("edge_errors", r.edge_errors.into_pyarray(py))?;
    Ok(d)
}

/// Mean terrain slope (degrees) of an elevation grid.
#[pyfunction]
fn coreg_ground_slope_deg(elevation: PyReadonlyArray2<f64>, cell_size: f64) -> PyResult<f64> {
    let (ny, nx) = (elevation.shape()[0], elevation.shape()[1]);
    let e: Vec<f64> = elevation.as_array().iter().copied().collect();
    sylva_rs::coreg::ground::slope_deg(&e, ny, nx, cell_size).map_err(err)
}

// --------------------------------------------------------- joint refinement

/// Joint refinement: a dict with `poses`, `shifts`, `rotations`,
/// `residual_before`, `residual_after` and `correspondences`. `log` is
/// called with each progress message.
#[pyfunction]
#[pyo3(signature = (points, poses, edges, stems, reference, voxel_sizes, max_distances, rounds, iterations, points_per_scan, correspondences_per_pair, min_planarity, normal_neighbours, max_normal_angle_deg, stem_weight, stem_radius, stem_scale, min_voxel_points, robust_scale, prior_translation, prior_rotation_deg, max_step_translation, max_step_rotation_deg, seed, log=None))]
fn coreg_refine_joint<'py>(py: Python<'py>, points: Vec<PyReadonlyArray2<f64>>, poses: PyReadonlyArray3<f64>, edges: Vec<(usize, usize)>, stems: Vec<PyReadonlyArray2<f64>>, reference: usize, voxel_sizes: Vec<f64>, max_distances: Vec<f64>, rounds: usize, iterations: usize, points_per_scan: usize, correspondences_per_pair: usize, min_planarity: f64, normal_neighbours: usize, max_normal_angle_deg: f64, stem_weight: f64, stem_radius: f64, stem_scale: f64, min_voxel_points: usize, robust_scale: f64, prior_translation: f64, prior_rotation_deg: f64, max_step_translation: f64, max_step_rotation_deg: f64, seed: u64, log: Option<Py<PyAny>>) -> PyResult<Bound<'py, PyDict>> {
    let pts: Vec<Vec<Point>> = points.into_iter().map(xyz_from_py).collect::<PyResult<_>>()?;
    let st: Vec<Vec<Point>> = stems.into_iter().map(xyz_from_py).collect::<PyResult<_>>()?;
    let start = mats_from_py::<4>(&poses, "poses")?;
    let params = refine::RefineParams { voxel_sizes, max_distances, rounds, iterations, points_per_scan, correspondences_per_pair, min_planarity, normal_neighbours, max_normal_angle_deg, stem_weight, stem_radius, stem_scale, min_voxel_points, robust_scale, prior_translation, prior_rotation_deg, max_step_translation, max_step_rotation_deg, seed };
    let mut failed: Option<PyErr> = None;
    let r = py.detach(|| {
        let mut say = |msg: &str| {
            if let (Some(f), None) = (&log, &failed) {
                if let Err(e) = Python::attach(|py| f.call1(py, (msg,))) {
                    failed = Some(e);
                }
            }
        };
        refine::refine_joint(&pts, &start, &edges, &st, reference, &params, &mut say)
    });
    if let Some(e) = failed {
        return Err(e);
    }
    let r = r.map_err(err)?;
    let d = PyDict::new(py);
    let poses = PyList::empty(py);
    for p in &r.poses {
        poses.append(mat_to_py(py, p))?;
    }
    d.set_item("poses", poses)?;
    d.set_item("shifts", r.shifts.into_pyarray(py))?;
    d.set_item("rotations", r.rotations.into_pyarray(py))?;
    d.set_item("residual_before", r.residual_before)?;
    d.set_item("residual_after", r.residual_after)?;
    d.set_item("correspondences", r.correspondences)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(coreg_skew, m)?,
        wrap_pyfunction!(coreg_so3_exp, m)?,
        wrap_pyfunction!(coreg_so3_log, m)?,
        wrap_pyfunction!(coreg_rotation_angle, m)?,
        wrap_pyfunction!(coreg_se3_exp, m)?,
        wrap_pyfunction!(coreg_se3_log, m)?,
        wrap_pyfunction!(coreg_invert, m)?,
        wrap_pyfunction!(coreg_transform_points, m)?,
        wrap_pyfunction!(coreg_transform_vectors, m)?,
        wrap_pyfunction!(coreg_yaw_transform, m)?,
        wrap_pyfunction!(coreg_kabsch, m)?,
        wrap_pyfunction!(coreg_kabsch_2d_yaw, m)?,
        wrap_pyfunction!(coreg_transform_difference, m)?,
        wrap_pyfunction!(coreg_read_tiepoint_list, m)?,
        wrap_pyfunction!(coreg_read_reflector_list, m)?,
        wrap_pyfunction!(coreg_detect_reflectors, m)?,
        wrap_pyfunction!(coreg_match_reflectors, m)?,
        wrap_pyfunction!(coreg_default_information, m)?,
        wrap_pyfunction!(coreg_adjoint, m)?,
        wrap_pyfunction!(coreg_plane_edge_information, m)?,
        wrap_pyfunction!(coreg_posegraph_residuals, m)?,
        wrap_pyfunction!(coreg_posegraph_total_error, m)?,
        wrap_pyfunction!(coreg_posegraph_components, m)?,
        wrap_pyfunction!(coreg_posegraph_initialise, m)?,
        wrap_pyfunction!(coreg_posegraph_optimise, m)?,
        wrap_pyfunction!(coreg_ground_slope_deg, m)?,
        wrap_pyfunction!(coreg_refine_joint, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
