// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the coregistration pipeline (sylva_rs::coreg::pipeline and
//! sylva_rs::coreg::survey).
//!
//! Scans, pairs and surveys cross as plain tuples and dicts that
//! `sylva/coreg/pipeline.py` builds from its dataclasses and builds them
//! back from. Progress messages come back through
//! [`sylva_rs::util::relay::relay`]: the pipeline runs on a worker thread and each
//! message is handed to the Python callable on the calling thread.
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use std::path::PathBuf;

use nalgebra::{Matrix4, Matrix6};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use sylva_rs::coreg::matching::StemMatch;
use sylva_rs::coreg::ground::GroundModel;
use sylva_rs::coreg::icp::{IcpConfig, IcpResult, PlaneInformation};
use sylva_rs::coreg::pipeline as cp;
use sylva_rs::coreg::posegraph as pg;
use sylva_rs::coreg::reflectors::{Reflector, ReflectorMatch};
use sylva_rs::coreg::stemmap::StemRecord;
use sylva_rs::coreg::survey as cs;
use sylva_rs::util::relay::{quiet, relay, Log};
use sylva_rs::trees::stems::StemParams;
use sylva_rs::{Point, Transform};

use crate::{err, xyz_from_py, xyz_to_py, PyCoregIcpTarget};

type Mat4 = Matrix4<f64>;

// ---------------------------------------------------------------- plumbing

fn prepare_err(e: cp::PrepareError) -> PyErr {
    match e {
        cp::PrepareError::MissingAttribute(m) => PyKeyError::new_err(m),
        cp::PrepareError::Core(e) => err(e),
    }
}

fn item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_string()))
}

fn get<'py, T>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    item(d, key)?.extract().map_err(Into::into)
}

/// A value that may be missing or None.
fn opt<'py, T>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> <T as FromPyObject<'a, 'py>>::Error: Into<PyErr>,
{
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract().map_err(Into::into)?)),
        _ => Ok(None),
    }
}

fn truthy(d: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    item(d, key)?.is_truthy()
}

fn mat4(a: &Bound<'_, PyAny>) -> PyResult<Mat4> {
    let a: PyReadonlyArray2<f64> = a.extract()?;
    let v = a.as_array();
    if v.shape() != [4, 4] {
        return Err(PyValueError::new_err(format!("a transform must have shape (4, 4), got {:?}", v.shape())));
    }
    Ok(Matrix4::from_fn(|r, c| v[[r, c]]))
}

fn mat4_to_py<'py>(py: Python<'py>, m: &Mat4) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = (0..4).flat_map(|r| (0..4).map(move |c| m[(r, c)])).collect();
    PyArray1::from_vec(py, flat).reshape([4, 4]).expect("reshape")
}

fn mat6_to_py<'py>(py: Python<'py>, m: &Matrix6<f64>) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = (0..6).flat_map(|r| (0..6).map(move |c| m[(r, c)])).collect();
    PyArray1::from_vec(py, flat).reshape([6, 6]).expect("reshape")
}

fn points(a: &Bound<'_, PyAny>) -> PyResult<Vec<Point>> {
    xyz_from_py(a.extract()?)
}

fn f32_points_to_py<'py>(py: Python<'py>, p: &[Point]) -> Bound<'py, PyArray2<f32>> {
    let flat: Vec<f32> = p.iter().flat_map(|q| q.iter().map(|&v| v as f32)).collect();
    PyArray1::from_vec(py, flat).reshape([p.len(), 3]).expect("reshape")
}

/// Run `work` with a [`Log`] that calls `log` (a Python callable, or None
/// for silence) on this thread; the first exception the callable raises is
/// raised once the work is done.
fn with_log<T: Send>(py: Python<'_>, log: Option<Py<PyAny>>, work: impl FnOnce(Log) -> T + Send) -> PyResult<T> {
    let Some(log) = log else { return Ok(py.detach(|| work(&quiet))) };
    let mut failed: Option<PyErr> = None;
    let out = py.detach(|| {
        relay(work, |msg| {
            if failed.is_none() {
                if let Err(e) = Python::attach(|py| log.call1(py, (msg,))) {
                    failed = Some(e);
                }
            }
        })
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

// ------------------------------------------------------------------ config

fn stem_params(d: &Bound<'_, PyDict>) -> PyResult<StemParams> {
    Ok(StemParams {
        slice_min: get(d, "slice_min")?,
        slice_max: get(d, "slice_max")?,
        slice_thickness: get(d, "slice_thickness")?,
        slice_step: get(d, "slice_step")?,
        reference_height: get(d, "reference_height")?,
        min_radius: get(d, "min_radius")?,
        max_radius: get(d, "max_radius")?,
        cluster_cell: get(d, "cluster_cell")?,
        min_cluster_points: get(d, "min_cluster_points")?,
        max_cluster_extent: get(d, "max_cluster_extent")?,
        ransac_iterations: get(d, "ransac_iterations")?,
        ransac_tolerance: get(d, "ransac_tolerance")?,
        max_circles_per_cluster: get(d, "max_circles_per_cluster")?,
        min_circle_inliers: get(d, "min_circle_inliers")?,
        min_coverage: get(d, "min_coverage")?,
        min_arc_deg: get(d, "min_arc_deg")?,
        max_circle_rmse: get(d, "max_circle_rmse")?,
        link_radius: get(d, "link_radius")?,
        link_radius_ratio: get(d, "link_radius_ratio")?,
        min_slices: get(d, "min_slices")?,
        max_lean_deg: get(d, "max_lean_deg")?,
        link_radius_abs: get(d, "link_radius_abs")?,
        prefilter: get(d, "prefilter")?,
        prefilter_k: get(d, "prefilter_k")?,
        prefilter_max_nz: get(d, "prefilter_max_nz")?,
        prefilter_max_variation: get(d, "prefilter_max_variation")?,
        seed: get(d, "seed")?,
        ransac_block: get(d, "ransac_block")?,
        ransac_presample: get(d, "ransac_presample")?,
        recluster_wide: get(d, "recluster_wide")?,
        cluster_grid_at_slice_min: get(d, "cluster_grid_at_slice_min")?,
        band_top_inclusive: get(d, "band_top_inclusive")?,
        shared_rng: get(d, "shared_rng")?,
        taper_weight_power: get(d, "taper_weight_power")?,
        min_total_points: get(d, "min_total_points")?,
        cluster_seeds: false,
    })
}

fn match_params(d: &Bound<'_, PyDict>) -> PyResult<sylva_rs::coreg::matching::MatchParams> {
    Ok(sylva_rs::coreg::matching::MatchParams {
        min_pair_distance: get(d, "min_pair_distance")?,
        max_pair_distance: get(d, "max_pair_distance")?,
        pair_distance_tolerance: get(d, "pair_distance_tolerance")?,
        inlier_tolerance: get(d, "inlier_tolerance")?,
        diameter_rel_tolerance: get(d, "diameter_rel_tolerance")?,
        diameter_abs_tolerance: get(d, "diameter_abs_tolerance")?,
        use_diameters: truthy(d, "use_diameters")?,
        max_stems: get(d, "max_stems")?,
        max_hypotheses: get(d, "max_hypotheses")?,
        min_inliers: get(d, "min_inliers")?,
        early_exit_inliers: get(d, "early_exit_inliers")?,
        distinct_translation: get(d, "distinct_translation")?,
        distinct_yaw_deg: get(d, "distinct_yaw_deg")?,
        refine_iterations: get(d, "refine_iterations")?,
    })
}

fn icp_config(d: &Bound<'_, PyDict>) -> PyResult<IcpConfig> {
    Ok(IcpConfig {
        voxel_sizes: get(d, "voxel_sizes")?,
        max_distances: opt(d, "max_distances")?,
        max_iterations: get(d, "max_iterations")?,
        method: get(d, "method")?,
        robust: get(d, "robust")?,
        robust_scale: get(d, "robust_scale")?,
        trim_fraction: get(d, "trim_fraction")?,
        trim_ramp: get(d, "trim_ramp")?,
        min_planarity: get(d, "min_planarity")?,
        normal_neighbours: get(d, "normal_neighbours")?,
        translation_tolerance: get(d, "translation_tolerance")?,
        rotation_tolerance: get(d, "rotation_tolerance")?,
        fitness_threshold: get(d, "fitness_threshold")?,
        damping: get(d, "damping")?,
        max_points: get(d, "max_points")?,
        plateau_tolerance: get(d, "plateau_tolerance")?,
        plateau_patience: get(d, "plateau_patience")?,
        seed: get(d, "seed")?,
    })
}

fn reading(d: &Bound<'_, PyDict>) -> PyResult<cp::ReadingOptions> {
    let mut o = cp::ReadingOptions { any: !d.is_empty(), ..Default::default() };
    if let Some(v) = opt::<PathBuf>(d, "library")? {
        o.library = Some(v);
    }
    if let Some(v) = d.get_item("drop_pseudo_echoes")? {
        o.drop_pseudo_echoes = v.is_truthy()?;
    }
    if let Some(v) = opt::<String>(d, "echoes")? {
        o.echoes = v;
    }
    if let Some(v) = opt::<usize>(d, "stride")? {
        o.stride = v;
    }
    if let Some(v) = opt::<usize>(d, "shot_stride")? {
        o.shot_stride = v;
    }
    for (g, name) in cp::GATES.iter().enumerate() {
        o.bounds[g] = (opt(d, &format!("min_{name}"))?, opt(d, &format!("max_{name}"))?);
    }
    Ok(o)
}

/// A `CoregConfig` from the dict `sylva.coreg.pipeline._config_core` builds.
fn config(d: &Bound<'_, PyDict>) -> PyResult<cp::CoregConfig> {
    Ok(cp::CoregConfig {
        ground_cell_size: get(d, "ground_cell_size")?,
        ground_min_coverage: opt(d, "ground_min_coverage")?,
        stems: stem_params(&item(d, "stems")?.cast_into()?)?,
        matching: match_params(&item(d, "matching")?.cast_into()?)?,
        icp: icp_config(&item(d, "icp")?.cast_into()?)?,
        icp_voxel: get(d, "icp_voxel")?,
        icp_min_planarity: get(d, "icp_min_planarity")?,
        icp_max_height: get(d, "icp_max_height")?,
        use_reflectors: truthy(d, "use_reflectors")?,
        min_reflector_matches: get(d, "min_reflector_matches")?,
        reflector_tolerance: get(d, "reflector_tolerance")?,
        trusted_reflector_matches: get(d, "trusted_reflector_matches")?,
        trusted_reflector_rmse: get(d, "trusted_reflector_rmse")?,
        min_match_inliers: get(d, "min_match_inliers")?,
        max_match_ambiguity: get(d, "max_match_ambiguity")?,
        ambiguity_margin: get(d, "ambiguity_margin")?,
        max_match_rmse: get(d, "max_match_rmse")?,
        max_coarse_stem_rmse: get(d, "max_coarse_stem_rmse")?,
        height_from_ground: truthy(d, "height_from_ground")?,
        ground_radius: get(d, "ground_radius")?,
        min_ground_cells: get(d, "min_ground_cells")?,
        max_ground_disagreement: opt(d, "max_ground_disagreement")?,
        max_pair_distance: get(d, "max_pair_distance")?,
        screen_pairs: truthy(d, "screen_pairs")?,
        max_pairs_per_scan: opt(d, "max_pairs_per_scan")?.unwrap_or(0),
        min_icp_fitness: get(d, "min_icp_fitness")?,
        max_icp_rmse: get(d, "max_icp_rmse")?,
        min_icp_fitness_above_ground: get(d, "min_icp_fitness_above_ground")?,
        fitness_min_height: get(d, "fitness_min_height")?,
        stem_agreement_tolerance: get(d, "stem_agreement_tolerance")?,
        max_coarse_to_fine_shift: get(d, "max_coarse_to_fine_shift")?,
        recover_unregistered: truthy(d, "recover_unregistered")?,
        recovery_rounds: get(d, "recovery_rounds")?,
        recovery_neighbours: get(d, "recovery_neighbours")?,
        refine_multiview: truthy(d, "refine_multiview")?,
        refinement_rounds: get(d, "refinement_rounds")?,
        refinement_voxel_sizes: get(d, "refinement_voxel_sizes")?,
        refinement_max_distances: get(d, "refinement_max_distances")?,
        refinement_points_per_scan: get(d, "refinement_points_per_scan")?,
        refinement_stem_weight: get(d, "refinement_stem_weight")?,
        refinement_stem_radius: get(d, "refinement_stem_radius")?,
        refinement_min_voxel_points: get(d, "refinement_min_voxel_points")?,
        refinement_max_shift: get(d, "refinement_max_shift")?,
        optimise_globally: truthy(d, "optimise_globally")?,
        reference_scan: get(d, "reference_scan")?,
        reject_outlier_edges: truthy(d, "reject_outlier_edges")?,
        information_patch_points: get(d, "information_patch_points")?,
        information_min_sigma: get(d, "information_min_sigma")?,
        max_prior_shift: get(d, "max_prior_shift")?,
        max_prior_rotation: opt(d, "max_prior_rotation")?,
        workers: get(d, "workers")?,
        memory_per_worker_gb: opt(d, "memory_per_worker_gb")?.unwrap_or(0.0),
        riscan_filter: get(d, "riscan_filter")?,
        reading: reading(&item(d, "riegl_options")?.cast_into()?)?,
        min_points_per_scan: get(d, "min_points_per_scan")?,
        max_points_per_scan: opt(d, "max_points_per_scan")?.unwrap_or(0),
    })
}

// ------------------------------------------------------------------- scans

type ReflectorTuple = (f64, f64, f64, f64, f64, i64, String);

fn reflectors_from_py(v: Vec<ReflectorTuple>) -> Vec<Reflector> {
    v.into_iter().map(|(x, y, z, reflectance, diameter, n_points, name)| Reflector { x, y, z, reflectance, diameter, n_points, name }).collect()
}

fn stems_from_py(a: PyReadonlyArray2<f64>) -> PyResult<Vec<StemRecord>> {
    let v = a.as_array();
    if v.nrows() > 0 && v.ncols() != 13 {
        return Err(PyValueError::new_err("stems must have 13 columns"));
    }
    Ok(v.rows().into_iter().map(|r| StemRecord { x: r[0], y: r[1], z: r[2], dbh: r[3], axis: [r[4], r[5], r[6]], reference_height: r[7], n_slices: r[8] as i64, n_points: r[9] as i64, rmse: r[10], coverage: r[11], lean_deg: r[12] }).collect())
}

fn stems_to_py<'py>(py: Python<'py>, stems: &[StemRecord]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = stems.iter().flat_map(|s| [s.x, s.y, s.z, s.dbh, s.axis[0], s.axis[1], s.axis[2], s.reference_height, s.n_slices as f64, s.n_points as f64, s.rmse, s.coverage, s.lean_deg]).collect();
    PyArray1::from_vec(py, flat).reshape([stems.len(), 13]).expect("reshape")
}

fn ground_from_py(t: &Bound<'_, PyAny>) -> PyResult<Option<GroundModel>> {
    if t.is_none() {
        return Ok(None);
    }
    let (elevation, origin, cell_size, observed): (PyReadonlyArray2<f64>, (f64, f64), f64, PyReadonlyArray2<bool>) = t.extract()?;
    let (ny, nx) = (elevation.shape()[0], elevation.shape()[1]);
    if observed.shape() != elevation.shape() {
        return Err(PyValueError::new_err("elevation and observed must have the same shape"));
    }
    Ok(Some(GroundModel { nx, ny, elevation: elevation.as_array().iter().cloned().collect(), origin: [origin.0, origin.1], cell_size, observed: observed.as_array().iter().cloned().collect() }))
}

fn ground_to_py<'py>(py: Python<'py>, g: &GroundModel) -> PyResult<Bound<'py, PyTuple>> {
    let e = PyArray1::from_vec(py, g.elevation.clone()).reshape([g.ny, g.nx])?;
    let o = PyArray1::from_vec(py, g.observed.clone()).reshape([g.ny, g.nx])?;
    PyTuple::new(py, [e.into_any(), (g.origin[0], g.origin[1]).into_pyobject(py)?.into_any(), g.cell_size.into_pyobject(py)?.into_any(), o.into_any()])
}

/// A scan from the tuple `_scan_core` builds.
fn scan_from_py(t: &Bound<'_, PyAny>) -> PyResult<cp::ScanFeatures> {
    let t = t.cast::<PyTuple>()?;
    if t.len() != 13 {
        return Err(PyValueError::new_err("a scan is a tuple of 13 fields"));
    }
    let heights: PyReadonlyArray1<f32> = t.get_item(7)?.extract()?;
    let origin: Vec<f64> = t.get_item(9)?.extract()?;
    if origin.len() != 3 {
        return Err(PyValueError::new_err("a scanner origin has three coordinates"));
    }
    Ok(cp::ScanFeatures {
        name: t.get_item(0)?.extract()?,
        n_points: t.get_item(1)?.extract()?,
        ground: ground_from_py(&t.get_item(2)?)?,
        stems: stems_from_py(t.get_item(3)?.extract()?)?,
        stem_map_name: t.get_item(4)?.extract()?,
        icp_points: points(&t.get_item(5)?)?,
        reflectors: reflectors_from_py(t.get_item(6)?.extract()?),
        icp_heights: heights.as_array().to_vec(),
        levelling: mat4(&t.get_item(8)?)?,
        origin: [origin[0], origin[1], origin[2]],
        source: t.get_item(10)?.extract()?,
        seconds: t.get_item(11)?.extract()?,
        error: t.get_item(12)?.extract()?,
    })
}

fn scans_from_py(list: &Bound<'_, PyList>) -> PyResult<Vec<cp::ScanFeatures>> {
    list.iter().map(|s| scan_from_py(&s)).collect()
}

fn scan_to_py<'py>(py: Python<'py>, s: &cp::ScanFeatures) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", &s.name)?;
    d.set_item("n_points", s.n_points)?;
    d.set_item("ground", s.ground.as_ref().map(|g| ground_to_py(py, g)).transpose()?)?;
    d.set_item("stems", stems_to_py(py, &s.stems))?;
    d.set_item("stem_map_name", &s.stem_map_name)?;
    d.set_item("icp_points", f32_points_to_py(py, &s.icp_points))?;
    let refl: Vec<ReflectorTuple> = s.reflectors.iter().map(|r| (r.x, r.y, r.z, r.reflectance, r.diameter, r.n_points, r.name.clone())).collect();
    d.set_item("reflectors", refl)?;
    d.set_item("icp_heights", PyArray1::from_slice(py, &s.icp_heights))?;
    d.set_item("levelling", mat4_to_py(py, &s.levelling))?;
    d.set_item("origin", PyArray1::from_slice(py, &s.origin))?;
    d.set_item("source", &s.source)?;
    d.set_item("seconds", s.seconds)?;
    d.set_item("error", &s.error)?;
    Ok(d)
}

/// A scan input: a path (str), `(n, 3)` points, or `("failed", reason, source)`.
fn input_from_py(v: &Bound<'_, PyAny>) -> PyResult<cp::ScanInput> {
    if let Ok(t) = v.cast::<PyTuple>() {
        let (_, reason, source): (String, String, Option<PathBuf>) = t.extract()?;
        return Ok(cp::ScanInput::Failed { reason, source });
    }
    if let Ok(p) = v.extract::<PathBuf>() {
        return Ok(cp::ScanInput::Path(p));
    }
    Ok(cp::ScanInput::Points(points(v)?))
}

// ------------------------------------------------------------------- pairs

fn stem_match_to_py<'py>(py: Python<'py>, r: &StemMatch) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("transform", mat4_to_py(py, &r.transform.0))?;
    d.set_item("n_inliers", r.n_inliers)?;
    d.set_item("inlier_rmse", r.inlier_rmse)?;
    d.set_item("score", r.score)?;
    let c: Vec<i64> = r.correspondences.iter().flat_map(|&(i, j)| [i as i64, j as i64]).collect();
    d.set_item("correspondences", PyArray1::from_vec(py, c).reshape([r.correspondences.len(), 2])?)?;
    d.set_item("n_source", r.n_source)?;
    d.set_item("n_target", r.n_target)?;
    d.set_item("success", r.success)?;
    d.set_item("ambiguity", r.ambiguity)?;
    d.set_item("rival", r.rival.as_ref().map(|rv| stem_match_to_py(py, rv)).transpose()?)?;
    Ok(d)
}

fn pairs_from_array(a: PyReadonlyArray2<i64>) -> PyResult<Vec<(usize, usize)>> {
    let v = a.as_array();
    if v.nrows() > 0 && v.ncols() != 2 {
        return Err(PyValueError::new_err("correspondences must have two columns"));
    }
    v.rows().into_iter().map(|r| Ok((usize::try_from(r[0]).map_err(|_| PyValueError::new_err("negative index"))?, usize::try_from(r[1]).map_err(|_| PyValueError::new_err("negative index"))?))).collect()
}

fn stem_match_from_py(v: &Bound<'_, PyAny>) -> PyResult<Option<StemMatch>> {
    if v.is_none() {
        return Ok(None);
    }
    let d = v.cast::<PyDict>()?;
    Ok(Some(StemMatch {
        transform: Transform(mat4(&item(d, "transform")?)?),
        n_inliers: get(d, "n_inliers")?,
        inlier_rmse: get(d, "inlier_rmse")?,
        score: get(d, "score")?,
        correspondences: pairs_from_array(get(d, "correspondences")?)?,
        n_source: get(d, "n_source")?,
        n_target: get(d, "n_target")?,
        success: truthy(d, "success")?,
        ambiguity: get(d, "ambiguity")?,
        rival: stem_match_from_py(&item(d, "rival")?)?.map(Box::new),
    }))
}

fn reflector_match_to_py<'py>(py: Python<'py>, r: &ReflectorMatch) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("transform", mat4_to_py(py, &r.transform))?;
    d.set_item("n_inliers", r.n_inliers)?;
    d.set_item("rmse", r.rmse)?;
    let c: Vec<i64> = r.correspondences.iter().flat_map(|c| [c[0] as i64, c[1] as i64]).collect();
    d.set_item("correspondences", PyArray1::from_vec(py, c).reshape([r.correspondences.len(), 2])?)?;
    d.set_item("success", r.success)?;
    Ok(d)
}

fn reflector_match_from_py(v: &Bound<'_, PyAny>) -> PyResult<Option<ReflectorMatch>> {
    if v.is_none() {
        return Ok(None);
    }
    let d = v.cast::<PyDict>()?;
    Ok(Some(ReflectorMatch {
        transform: mat4(&item(d, "transform")?)?,
        n_inliers: get(d, "n_inliers")?,
        rmse: get(d, "rmse")?,
        correspondences: pairs_from_array(get(d, "correspondences")?)?.into_iter().map(|(a, b)| [a, b]).collect(),
        success: truthy(d, "success")?,
    }))
}

fn icp_to_py<'py>(py: Python<'py>, r: &IcpResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("transform", mat4_to_py(py, &r.transform))?;
    d.set_item("fitness", r.fitness)?;
    d.set_item("inlier_rmse", r.inlier_rmse)?;
    d.set_item("n_correspondences", r.n_correspondences)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    d.set_item("history", r.history.clone())?;
    match &r.information {
        Some(info) => {
            d.set_item("hessian", mat6_to_py(py, &info.hessian))?;
            d.set_item("plane_sigma", info.sigma)?;
            d.set_item("plane_n", info.n)?;
        }
        None => d.set_item("hessian", py.None())?,
    }
    Ok(d)
}

fn icp_from_py(v: &Bound<'_, PyAny>) -> PyResult<Option<IcpResult>> {
    if v.is_none() {
        return Ok(None);
    }
    let d = v.cast::<PyDict>()?;
    let information = match opt::<PyReadonlyArray2<f64>>(d, "hessian")? {
        Some(h) => {
            let a = h.as_array();
            if a.shape() != [6, 6] {
                return Err(PyValueError::new_err("a hessian must have shape (6, 6)"));
            }
            Some(PlaneInformation { hessian: Matrix6::from_fn(|r, c| a[[r, c]]), sigma: get(d, "plane_sigma")?, n: get(d, "plane_n")? })
        }
        None => None,
    };
    Ok(Some(IcpResult { transform: mat4(&item(d, "transform")?)?, fitness: get(d, "fitness")?, inlier_rmse: get(d, "inlier_rmse")?, n_correspondences: get(d, "n_correspondences")?, iterations: get(d, "iterations")?, converged: truthy(d, "converged")?, history: get(d, "history")?, information }))
}

fn pair_to_py<'py>(py: Python<'py>, p: &cp::PairResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("i", p.i)?;
    d.set_item("j", p.j)?;
    d.set_item("name_i", &p.name_i)?;
    d.set_item("name_j", &p.name_j)?;
    d.set_item("transform", mat4_to_py(py, &p.transform))?;
    d.set_item("coarse_transform", mat4_to_py(py, &p.coarse_transform))?;
    d.set_item("match", p.stem_match.as_ref().map(|m| stem_match_to_py(py, m)).transpose()?)?;
    d.set_item("reflector_match", p.reflector_match.as_ref().map(|m| reflector_match_to_py(py, m)).transpose()?)?;
    d.set_item("icp", p.icp.as_ref().map(|m| icp_to_py(py, m)).transpose()?)?;
    d.set_item("success", p.success)?;
    d.set_item("reason", &p.reason)?;
    d.set_item("seconds", p.seconds)?;
    d.set_item("matched_source", xyz_to_py(py, &p.matched_source))?;
    d.set_item("matched_target", xyz_to_py(py, &p.matched_target))?;
    d.set_item("coarse_stem_rmse", p.coarse_stem_rmse)?;
    d.set_item("fine_stem_rmse", p.fine_stem_rmse)?;
    d.set_item("fitness_above", p.fitness_above)?;
    d.set_item("rival", p.rival.as_ref().map(|m| stem_match_to_py(py, m)).transpose()?)?;
    d.set_item("used_icp", p.used_icp)?;
    d.set_item("ground_offset", p.ground_offset)?;
    d.set_item("trusted", p.trusted)?;
    Ok(d)
}

/// A pair from the dict `_pair_core` builds.
fn pair_from_py(v: &Bound<'_, PyAny>) -> PyResult<cp::PairResult> {
    let d = v.cast::<PyDict>()?;
    Ok(cp::PairResult {
        i: get(d, "i")?,
        j: get(d, "j")?,
        name_i: get(d, "name_i")?,
        name_j: get(d, "name_j")?,
        transform: mat4(&item(d, "transform")?)?,
        coarse_transform: mat4(&item(d, "coarse_transform")?)?,
        stem_match: stem_match_from_py(&item(d, "match")?)?,
        reflector_match: reflector_match_from_py(&item(d, "reflector_match")?)?,
        icp: icp_from_py(&item(d, "icp")?)?,
        success: truthy(d, "success")?,
        reason: get(d, "reason")?,
        seconds: get(d, "seconds")?,
        matched_source: points(&item(d, "matched_source")?)?,
        matched_target: points(&item(d, "matched_target")?)?,
        coarse_stem_rmse: get(d, "coarse_stem_rmse")?,
        fine_stem_rmse: get(d, "fine_stem_rmse")?,
        fitness_above: get(d, "fitness_above")?,
        rival: stem_match_from_py(&item(d, "rival")?)?,
        used_icp: truthy(d, "used_icp")?,
        ground_offset: get(d, "ground_offset")?,
        trusted: truthy(d, "trusted")?,
    })
}

fn pairs_from_py(list: &Bound<'_, PyList>) -> PyResult<Vec<cp::PairResult>> {
    list.iter().map(|p| pair_from_py(&p)).collect()
}

fn targets_from_py<'a>(scans: &'a [cp::ScanFeatures], poses: &[Mat4]) -> Vec<(&'a cp::ScanFeatures, Mat4)> {
    scans.iter().zip(poses.iter().copied()).collect()
}

fn mats_from_list(list: &Bound<'_, PyList>) -> PyResult<Vec<Mat4>> {
    list.iter().map(|m| mat4(&m)).collect()
}

// --------------------------------------------------------------- functions

/// `prepare_scan`: a scan dict for `_scan_from_core`.
#[pyfunction]
#[pyo3(signature = (cloud, config, name, reflectors, levelling=None, origin=None))]
fn coreg_prepare_scan<'py>(py: Python<'py>, cloud: &Bound<'py, PyAny>, config: &Bound<'py, PyDict>, name: String, reflectors: Vec<ReflectorTuple>, levelling: Option<&Bound<'py, PyAny>>, origin: Option<[f64; 3]>) -> PyResult<Bound<'py, PyDict>> {
    let cfg = self::config(config)?;
    let input = input_from_py(cloud)?;
    let refl = reflectors_from_py(reflectors);
    let level = levelling.map(mat4).transpose()?;
    let scan = py.detach(|| cp::prepare_scan(&input, &cfg, &name, &refl, level.as_ref(), origin)).map_err(prepare_err)?;
    scan_to_py(py, &scan)
}

/// `_read_scan`: the filtered points of a scan file.
#[pyfunction]
fn coreg_read_scan<'py>(py: Python<'py>, path: PathBuf, config: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cfg = self::config(config)?;
    let pts = py.detach(|| cp::read_scan(&path, &cfg)).map_err(prepare_err)?;
    Ok(xyz_to_py(py, &pts))
}

/// `_refit_visible_ground`: the refitted ground, or None if the fit stands.
#[pyfunction]
fn coreg_refit_visible_ground<'py>(py: Python<'py>, points: PyReadonlyArray2<f64>, scanner: [f64; 3], config: &Bound<'py, PyDict>) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let cfg = self::config(config)?;
    let p = xyz_from_py(points)?;
    let g = py.detach(|| cp::refit_visible_ground(&p, scanner, &cfg)).map_err(err)?;
    g.as_ref().map(|g| ground_to_py(py, g)).transpose()
}

#[pyfunction]
fn coreg_terrain_samples<'py>(py: Python<'py>, scan: &Bound<'py, PyAny>, radius: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = scan_from_py(scan)?;
    Ok(xyz_to_py(py, &cp::terrain_samples(&s, radius)))
}

/// `_height_offset`, `targets` as a list of scans and one of their poses.
#[pyfunction]
fn coreg_height_offset(py: Python<'_>, source: &Bound<'_, PyAny>, world_from_source: &Bound<'_, PyAny>, targets: &Bound<'_, PyList>, poses: &Bound<'_, PyList>, config: &Bound<'_, PyDict>) -> PyResult<f64> {
    let (s, t, cfg) = (scan_from_py(source)?, mat4(world_from_source)?, self::config(config)?);
    let (scans, poses) = (scans_from_py(targets)?, mats_from_list(poses)?);
    Ok(py.detach(|| cp::height_offset(&s, &t, &targets_from_py(&scans, &poses), &cfg)))
}

#[pyfunction]
fn coreg_on_ground<'py>(py: Python<'py>, transform: &Bound<'py, PyAny>, source: &Bound<'py, PyAny>, targets: &Bound<'py, PyList>, poses: &Bound<'py, PyList>, config: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (t, s, cfg) = (mat4(transform)?, scan_from_py(source)?, self::config(config)?);
    let (scans, poses) = (scans_from_py(targets)?, mats_from_list(poses)?);
    let out = py.detach(|| cp::on_ground(&t, &s, &targets_from_py(&scans, &poses), &cfg));
    Ok(mat4_to_py(py, &out))
}

#[pyfunction]
fn coreg_prior_ok(pose: &Bound<'_, PyAny>, prior: &Bound<'_, PyAny>, origin: [f64; 3], config: &Bound<'_, PyDict>) -> PyResult<(bool, String)> {
    Ok(cp::prior_ok(&mat4(pose)?, &mat4(prior)?, origin, &self::config(config)?))
}

#[pyfunction]
fn coreg_stem_median_residual(transform: &Bound<'_, PyAny>, matched_source: PyReadonlyArray2<f64>, matched_target: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let (ms, mt) = (xyz_from_py(matched_source)?, xyz_from_py(matched_target)?);
    if ms.len() != mt.len() {
        return Err(PyValueError::new_err("matched stems must pair up"));
    }
    Ok(cp::stem_median_residual(&mat4(transform)?, &ms, &mt))
}

/// `_within_reach`: `(kept pairs, messages)`.
#[pyfunction]
fn coreg_within_reach(pairs: Vec<(usize, usize)>, positions: Option<Vec<Vec<f64>>>, limit: f64) -> PyResult<(Vec<(usize, usize)>, Vec<String>)> {
    let messages = std::sync::Mutex::new(Vec::new());
    let log = |m: &str| messages.lock().expect("log").push(m.to_string());
    let kept = cs::within_reach(&pairs, positions.as_deref(), limit, &log).map_err(err)?;
    Ok((kept, messages.into_inner().expect("log")))
}

/// `_limit_per_scan` on `(i, j, inliers)`: the indices kept.
#[pyfunction]
fn coreg_limit_per_scan(pairs: Vec<(usize, usize, usize)>, limit: i64, n_scans: usize) -> PyResult<Vec<usize>> {
    if pairs.iter().any(|&(i, j, _)| i >= n_scans || j >= n_scans) {
        return Err(PyValueError::new_err("list index out of range"));
    }
    Ok(cs::limit_per_scan(&pairs, limit, n_scans))
}

/// `_combined_stem_map`: stem rows, best first.
#[pyfunction]
fn coreg_combined_stem_map<'py>(py: Python<'py>, scans: &Bound<'py, PyList>, registered: Vec<bool>, poses: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (s, p) = (scans_from_py(scans)?, mats_from_list(poses)?);
    if registered.len() > s.len() || p.len() < registered.len() {
        return Err(PyValueError::new_err("list index out of range"));
    }
    Ok(stems_to_py(py, &cs::combined_stem_map(&s, &registered, &p)))
}

/// `register_pair`: a pair dict.
#[pyfunction]
#[pyo3(signature = (source, target, config, initial=None, stem_match=None, i=0, j=1, target_icp=None))]
fn coreg_register_pair<'py>(py: Python<'py>, source: &Bound<'py, PyAny>, target: &Bound<'py, PyAny>, config: &Bound<'py, PyDict>, initial: Option<&Bound<'py, PyAny>>, stem_match: Option<&Bound<'py, PyAny>>, i: i64, j: i64, target_icp: Option<PyRef<'py, PyCoregIcpTarget>>) -> PyResult<Bound<'py, PyDict>> {
    let (s, t, cfg) = (scan_from_py(source)?, scan_from_py(target)?, self::config(config)?);
    let initial = initial.map(mat4).transpose()?;
    let m = stem_match.map(stem_match_from_py).transpose()?.flatten();
    let prepared = target_icp.map(|t| t.inner.clone());
    let p = py.detach(|| cp::register_pair(&s, &t, &cfg, initial.as_ref(), m.as_ref(), i, j, prepared.as_deref())).map_err(err)?;
    pair_to_py(py, &p)
}

/// `_trust_reflectors` on a pair dict: the pair after the decision.
#[pyfunction]
fn coreg_trust_reflectors<'py>(py: Python<'py>, pair: &Bound<'py, PyAny>, coarse: &Bound<'py, PyAny>, shift: f64, config: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let mut p = pair_from_py(pair)?;
    cp::trust_reflectors(&mut p, &mat4(coarse)?, shift, &self::config(config)?);
    pair_to_py(py, &p)
}

#[pyfunction]
fn coreg_pair_summary(pair: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(pair_from_py(pair)?.summary())
}

/// `place_from_prior`: `(pair dict, indices used)`.
#[pyfunction]
#[pyo3(signature = (scan, survey, poses, prior, config, neighbours=None))]
fn coreg_place_from_prior<'py>(py: Python<'py>, scan: &Bound<'py, PyAny>, survey: &Bound<'py, PyList>, poses: &Bound<'py, PyList>, prior: &Bound<'py, PyAny>, config: &Bound<'py, PyDict>, neighbours: Option<i64>) -> PyResult<(Bound<'py, PyDict>, Vec<usize>)> {
    let (s, cfg, prior) = (scan_from_py(scan)?, self::config(config)?, mat4(prior)?);
    let (others, poses) = (scans_from_py(survey)?, mats_from_list(poses)?);
    if poses.len() < others.len() {
        return Err(PyValueError::new_err("list index out of range"));
    }
    let refs: Vec<&cp::ScanFeatures> = others.iter().collect();
    let (r, used) = py.detach(|| cs::place_from_prior(&s, &refs, &poses, &prior, &cfg, neighbours)).map_err(err)?;
    Ok((pair_to_py(py, &r)?, used))
}

fn optimisation_to_py<'py>(py: Python<'py>, o: &pg::Optimisation) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("iterations", o.iterations)?;
    d.set_item("converged", o.converged)?;
    d.set_item("initial_error", o.initial_error)?;
    d.set_item("final_error", o.final_error)?;
    d.set_item("rejected_edges", o.rejected_edges.clone())?;
    d.set_item("edge_errors", o.edge_errors.clone().into_pyarray(py))?;
    Ok(d)
}

fn survey_to_py<'py>(py: Python<'py>, r: &cs::SurveyResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let pairs = PyList::empty(py);
    for p in &r.pairs {
        pairs.append(pair_to_py(py, p)?)?;
    }
    d.set_item("pairs", pairs)?;
    let poses = PyList::empty(py);
    for p in &r.poses {
        poses.append(mat4_to_py(py, p))?;
    }
    d.set_item("poses", poses)?;
    d.set_item("reference", r.reference)?;
    d.set_item("optimisation", r.optimisation.as_ref().map(|o| optimisation_to_py(py, o)).transpose()?)?;
    d.set_item("registered", r.registered.clone())?;
    d.set_item("seconds", r.seconds)?;
    d.set_item("edge_to_pair", r.edge_to_pair.clone())?;
    Ok(d)
}

fn options_from_py(pairs: Option<Vec<(usize, usize)>>, positions: Option<Vec<Vec<f64>>>, fixed: Vec<(usize, Bound<'_, PyAny>)>, priors: Option<Vec<Option<Bound<'_, PyAny>>>>) -> PyResult<cs::SurveyOptions> {
    Ok(cs::SurveyOptions {
        pairs,
        approximate_positions: positions,
        fixed: fixed.iter().map(|(k, p)| Ok((*k, mat4(p)?))).collect::<PyResult<_>>()?,
        priors: priors.map(|v| v.iter().map(|p| p.as_ref().map(mat4).transpose()).collect::<PyResult<_>>()).transpose()?,
    })
}

/// `coregister_prepared`: a survey dict.
#[pyfunction]
#[pyo3(signature = (scans, config, pairs=None, positions=None, fixed=Vec::new(), priors=None, log=None, already=0.0))]
fn coreg_coregister_prepared<'py>(py: Python<'py>, scans: &Bound<'py, PyList>, config: &Bound<'py, PyDict>, pairs: Option<Vec<(usize, usize)>>, positions: Option<Vec<Vec<f64>>>, fixed: Vec<(usize, Bound<'py, PyAny>)>, priors: Option<Vec<Option<Bound<'py, PyAny>>>>, log: Option<Py<PyAny>>, already: f64) -> PyResult<Bound<'py, PyDict>> {
    let (s, cfg) = (scans_from_py(scans)?, self::config(config)?);
    let opts = options_from_py(pairs, positions, fixed, priors)?;
    let r = with_log(py, log, |log| cs::coregister_prepared(&s, &cfg, &opts, log, already))?.map_err(err)?;
    survey_to_py(py, &r)
}

/// `coregister`: `(scan dicts, survey dict)`.
#[pyfunction]
#[pyo3(signature = (inputs, config, names, reflectors, levelling, pairs=None, positions=None, fixed=Vec::new(), priors=None, log=None))]
fn coreg_coregister<'py>(py: Python<'py>, inputs: &Bound<'py, PyList>, config: &Bound<'py, PyDict>, names: Vec<String>, reflectors: Vec<Vec<ReflectorTuple>>, levelling: Vec<Option<Bound<'py, PyAny>>>, pairs: Option<Vec<(usize, usize)>>, positions: Option<Vec<Vec<f64>>>, fixed: Vec<(usize, Bound<'py, PyAny>)>, priors: Option<Vec<Option<Bound<'py, PyAny>>>>, log: Option<Py<PyAny>>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyDict>)> {
    let cfg = self::config(config)?;
    if reflectors.len() != inputs.len() || levelling.len() != inputs.len() {
        return Err(PyValueError::new_err("one set of reflectors and one levelling per scan"));
    }
    let specs: Vec<cs::ScanSpec> = inputs
        .iter()
        .zip(reflectors)
        .zip(&levelling)
        .map(|((input, r), l)| Ok(cs::ScanSpec { input: input_from_py(&input)?, reflectors: reflectors_from_py(r), levelling: l.as_ref().map(mat4).transpose()? }))
        .collect::<PyResult<_>>()?;
    let opts = options_from_py(pairs, positions, fixed, priors)?;
    let (scans, r) = with_log(py, log, |log| cs::coregister(&specs, &cfg, Some(&names), &opts, log))?.map_err(err)?;
    let list = PyList::empty(py);
    for s in &scans {
        list.append(scan_to_py(py, s)?)?;
    }
    Ok((list, survey_to_py(py, &r)?))
}

/// `merge_clouds`: `(xyz, scan_id)`.
#[pyfunction]
fn coreg_merge_clouds<'py>(py: Python<'py>, inputs: &Bound<'py, PyList>, poses: &Bound<'py, PyList>, levellings: &Bound<'py, PyList>, registered: Vec<bool>, only_registered: bool, voxel: Option<f64>, config: &Bound<'py, PyDict>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i32>>)> {
    let cfg = self::config(config)?;
    let inputs: Vec<cp::ScanInput> = inputs.iter().map(|i| input_from_py(&i)).collect::<PyResult<_>>()?;
    let (poses, levellings) = (mats_from_list(poses)?, mats_from_list(levellings)?);
    let cloud = py.detach(|| cs::merge_clouds(&inputs, &poses, &levellings, &registered, only_registered, voxel, &cfg)).map_err(prepare_err)?;
    let ids = match cloud.attr("scan_id") {
        Some(sylva_rs::pointcloud::Attr::I32(v)) => v.clone(),
        _ => Vec::new(),
    };
    Ok((xyz_to_py(py, &cloud.xyz), ids.into_pyarray(py)))
}

type ScanSummaryTuple<'py> = (String, i64, i64, String, Option<String>, Bound<'py, PyAny>);

fn summaries_from_py(v: Vec<ScanSummaryTuple<'_>>) -> PyResult<Vec<cs::ScanSummary>> {
    v.into_iter().map(|(name, n_points, n_stems, error, source, levelling)| Ok(cs::ScanSummary { name, n_points, n_stems, error, source, levelling: mat4(&levelling)? })).collect()
}

type OptimisationTuple = (usize, bool, f64, f64, Vec<usize>);

fn optimisation_from_py(o: Option<OptimisationTuple>) -> Option<pg::Optimisation> {
    o.map(|(iterations, converged, initial_error, final_error, rejected_edges)| pg::Optimisation { poses: Vec::new(), iterations, converged, initial_error, final_error, rejected_edges, edge_errors: Vec::new() })
}

/// `SurveyResult.report`.
#[pyfunction]
fn coreg_survey_report(scans: Vec<ScanSummaryTuple<'_>>, pairs: &Bound<'_, PyList>, poses: &Bound<'_, PyList>, reference: i64, optimisation: Option<OptimisationTuple>, registered: Vec<bool>, seconds: f64, edge_to_pair: Vec<usize>) -> PyResult<String> {
    let (s, p, poses) = (summaries_from_py(scans)?, pairs_from_py(pairs)?, mats_from_list(poses)?);
    cs::report(&s, &p, &poses, reference, optimisation_from_py(optimisation).as_ref(), &registered, seconds, &edge_to_pair).map_err(err)
}

/// `SurveyResult.consistency`: `[(i, j, value)]` in dict order.
#[pyfunction]
fn coreg_survey_consistency(pairs: &Bound<'_, PyList>, poses: &Bound<'_, PyList>, robust: bool) -> PyResult<Vec<(i64, i64, f64)>> {
    let (p, poses) = (pairs_from_py(pairs)?, mats_from_list(poses)?);
    Ok(cs::consistency(&p, &poses, robust).map_err(err)?.into_iter().map(|((i, j), v)| (i, j, v)).collect())
}

/// `SurveyResult.save`.
#[pyfunction]
fn coreg_survey_save(path: PathBuf, scans: Vec<ScanSummaryTuple<'_>>, pairs: &Bound<'_, PyList>, poses: &Bound<'_, PyList>, reference: i64, registered: Vec<bool>, seconds: f64) -> PyResult<()> {
    let (s, p, poses) = (summaries_from_py(scans)?, pairs_from_py(pairs)?, mats_from_list(poses)?);
    let text = cs::survey_json(&s, &p, &poses, reference, &registered, seconds).map_err(err)?;
    cs::save_survey(&path, &text).map_err(err)
}

/// `load_transforms`: `[(name, rows)]` in dict order.
#[pyfunction]
fn coreg_load_transforms(path: PathBuf) -> PyResult<Vec<(String, Vec<Vec<f64>>)>> {
    cs::load_transforms(&path).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(coreg_prepare_scan, m)?,
        wrap_pyfunction!(coreg_read_scan, m)?,
        wrap_pyfunction!(coreg_refit_visible_ground, m)?,
        wrap_pyfunction!(coreg_terrain_samples, m)?,
        wrap_pyfunction!(coreg_height_offset, m)?,
        wrap_pyfunction!(coreg_on_ground, m)?,
        wrap_pyfunction!(coreg_prior_ok, m)?,
        wrap_pyfunction!(coreg_stem_median_residual, m)?,
        wrap_pyfunction!(coreg_within_reach, m)?,
        wrap_pyfunction!(coreg_limit_per_scan, m)?,
        wrap_pyfunction!(coreg_combined_stem_map, m)?,
        wrap_pyfunction!(coreg_register_pair, m)?,
        wrap_pyfunction!(coreg_trust_reflectors, m)?,
        wrap_pyfunction!(coreg_pair_summary, m)?,
        wrap_pyfunction!(coreg_place_from_prior, m)?,
        wrap_pyfunction!(coreg_coregister_prepared, m)?,
        wrap_pyfunction!(coreg_coregister, m)?,
        wrap_pyfunction!(coreg_merge_clouds, m)?,
        wrap_pyfunction!(coreg_survey_report, m)?,
        wrap_pyfunction!(coreg_survey_consistency, m)?,
        wrap_pyfunction!(coreg_survey_save, m)?,
        wrap_pyfunction!(coreg_load_transforms, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
