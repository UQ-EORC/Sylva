// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the epoch, tree and summary parts of sylva_rs::change and
//! for its synthetic epochs.
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::change::{epochs, provenance, summary, synthetic, trees};
use sylva_rs::coreg_ground::GroundModel;
use sylva_rs::{Point, Transform};

use crate::{cloud_to_py, err, shots_to_py, xyz_from_py};

fn transform_from_py(m: PyReadonlyArray2<f64>) -> PyResult<Transform> {
    let a = m.as_array();
    if a.shape() != [4, 4] {
        return Err(PyValueError::new_err(format!("a transform must have shape (4, 4), got {:?}", a.shape())));
    }
    Transform::from_row_major(&a.iter().copied().collect::<Vec<_>>()).map_err(err)
}

fn transform_to_py<'py>(py: Python<'py>, t: &Transform) -> Bound<'py, PyArray2<f64>> {
    PyArray1::from_vec(py, t.to_row_major().to_vec()).reshape([4, 4]).expect("reshape")
}

fn rows_from_py<const N: usize>(a: PyReadonlyArray2<f64>, what: &str) -> PyResult<Vec<[f64; N]>> {
    let v = a.as_array();
    if v.ncols() != N {
        return Err(PyValueError::new_err(format!("{what} must have shape (n, {N}), got (n, {})", v.ncols())));
    }
    Ok(v.rows().into_iter().map(|r| std::array::from_fn(|k| r[k])).collect())
}

fn matrix_to_py<'py>(py: Python<'py>, rows: &[Vec<f64>], ncols: usize) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    PyArray1::from_vec(py, flat).reshape([rows.len(), ncols]).expect("reshape")
}

// ------------------------------------------------------------- synthetic

#[pyfunction]
fn change_forest_epochs<'py>(
    py: Python<'py>,
    n_trees: usize,
    size: f64,
    min_spacing: f64,
    deaths: usize,
    recruits: usize,
    replaced: usize,
    small_increments: usize,
    dbh_increment: (f64, f64),
    height_increment: (f64, f64),
    branch_removals: usize,
    foliage_box: Option<[f64; 6]>,
    foliage_fraction: f64,
    tree_shift: f64,
    offset: PyReadonlyArray2<f64>,
    range_noise: [f64; 2],
    scan_positions: Vec<[f64; 2]>,
    scan_jitter: f64,
    resolution_deg: f64,
    max_echoes: usize,
    ground_density: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = synthetic::EpochParams {
        n_trees,
        size,
        min_spacing,
        deaths,
        recruits,
        replaced,
        small_increments,
        dbh_increment,
        height_increment,
        branch_removals,
        foliage_box,
        foliage_fraction,
        tree_shift,
        offset: transform_from_py(offset)?,
        range_noise,
        scan_positions,
        scan_jitter,
        resolution_deg,
        max_echoes,
        ground_density,
        seed,
    };
    let f = py.detach(|| synthetic::forest_epochs(&p)).map_err(err)?;
    let d = PyDict::new(py);
    let clouds = PyList::empty(py);
    for c in &f.clouds {
        clouds.append(cloud_to_py(py, c)?)?;
    }
    d.set_item("clouds", clouds)?;
    let shots = PyList::empty(py);
    for s in &f.shots {
        shots.append(shots_to_py(py, s)?)?;
    }
    d.set_item("shots", shots)?;
    let tables = PyList::empty(py);
    for set in &f.trees {
        let t = PyDict::new(py);
        t.set_item("tree_id", set.iter().map(|r| r.tree_id).collect::<Vec<_>>().into_pyarray(py))?;
        for (name, get) in [
            ("x", (|r: &synthetic::TrueTree| r.x) as fn(&synthetic::TrueTree) -> f64),
            ("y", |r| r.y),
            ("z0", |r| r.z0),
            ("dbh", |r| r.dbh),
            ("height", |r| r.height),
            ("stem_volume", |r| r.stem_volume),
            ("wood_volume", |r| r.wood_volume),
            ("leaf_area", |r| r.leaf_area),
        ] {
            t.set_item(name, set.iter().map(get).collect::<Vec<_>>().into_pyarray(py))?;
        }
        t.set_item("n_limbs", set.iter().map(|r| r.n_limbs as i64).collect::<Vec<_>>().into_pyarray(py))?;
        tables.append(t)?;
    }
    d.set_item("trees", tables)?;
    let changes = PyList::empty(py);
    for c in &f.changes {
        let v = PyDict::new(py);
        v.set_item("kind", c.kind)?;
        v.set_item("tree_id", c.tree_id)?;
        for (k, x) in &c.values {
            v.set_item(*k, *x)?;
        }
        changes.append(v)?;
    }
    d.set_item("changes", changes)?;
    d.set_item("transform", transform_to_py(py, &f.transform))?;
    let origins = PyList::empty(py);
    for o in &f.origins {
        origins.append(crate::xyz_to_py(py, o))?;
    }
    d.set_item("origins", origins)?;
    Ok(d)
}

// ---------------------------------------------------------------- epochs

fn ground_from_py(g: Option<(PyReadonlyArray2<f64>, f64, f64, f64, PyReadonlyArray2<bool>)>) -> PyResult<Option<GroundModel>> {
    let Some((e, x0, y0, cs, obs)) = g else { return Ok(None) };
    let (e, obs) = (e.as_array(), obs.as_array());
    if e.shape() != obs.shape() {
        return Err(PyValueError::new_err("ground elevation and observed differ in shape"));
    }
    Ok(Some(GroundModel { ny: e.nrows(), nx: e.ncols(), elevation: e.iter().copied().collect(), origin: [x0, y0], cell_size: cs, observed: obs.iter().copied().collect() }))
}

#[pyfunction]
fn change_align_on_stable<'py>(
    py: Python<'py>,
    ref_stems: PyReadonlyArray2<f64>,
    new_stems: PyReadonlyArray2<f64>,
    ref_ground: Option<(PyReadonlyArray2<f64>, f64, f64, f64, PyReadonlyArray2<bool>)>,
    new_ground: Option<(PyReadonlyArray2<f64>, f64, f64, f64, PyReadonlyArray2<bool>)>,
    initial: PyReadonlyArray2<f64>,
    evaluate: PyReadonlyArray2<f64>,
    use_stems: bool,
    use_ground: bool,
    stem_tolerance: f64,
    dbh_tolerance: f64,
    ground_spacing: f64,
    ground_block: f64,
    stem_floor: f64,
    ground_floor: f64,
    huber: f64,
    iterations: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (rs, ns) = (rows_from_py::<4>(ref_stems, "ref_stems")?, rows_from_py::<4>(new_stems, "new_stems")?);
    let (rg, ng) = (ground_from_py(ref_ground)?, ground_from_py(new_ground)?);
    let init = transform_from_py(initial)?;
    let ev: Vec<Point> = xyz_from_py(evaluate)?;
    let p = epochs::AlignParams { use_stems, use_ground, stem_tolerance, dbh_tolerance, ground_spacing, ground_block, stem_floor, ground_floor, huber, iterations };
    let samples = ng.as_ref().map(|g| epochs::ground_samples(g, ground_spacing)).unwrap_or_default();
    let a = py.detach(|| epochs::align_on_stable(&rs, &ns, rg.as_ref(), &samples, &init, &ev, &p)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("transform", transform_to_py(py, &a.transform))?;
    let cov: Vec<Vec<f64>> = (0..6).map(|r| (0..6).map(|c| a.covariance[(r, c)]).collect()).collect();
    d.set_item("covariance", matrix_to_py(py, &cov, 6))?;
    d.set_item("centre", a.centre.to_vec())?;
    d.set_item("sigma_xyz", a.sigma_xyz.to_vec())?;
    d.set_item("registration_sigma", a.registration_sigma)?;
    d.set_item("sigma_horizontal", a.sigma_horizontal)?;
    d.set_item("sigma_vertical", a.sigma_vertical)?;
    let flat: Vec<i64> = a.stem_pairs.iter().flat_map(|&(i, j)| [i as i64, j as i64]).collect();
    d.set_item("stem_pairs", PyArray1::from_vec(py, flat).reshape([a.stem_pairs.len(), 2])?)?;
    let res: Vec<Vec<f64>> = a.stem_residuals.iter().map(|r| r.to_vec()).collect();
    d.set_item("stem_residuals", matrix_to_py(py, &res, 2))?;
    d.set_item("n_ground", a.n_ground)?;
    d.set_item("ground_residuals", a.ground_residuals.clone().into_pyarray(py))?;
    d.set_item("stem_sigma", a.stem_sigma)?;
    d.set_item("ground_sigma", a.ground_sigma)?;
    d.set_item("iterations", a.iterations)?;
    Ok(d)
}

// ----------------------------------------------------------------- trees

#[pyfunction]
fn change_match_trees<'py>(py: Python<'py>, a: PyReadonlyArray2<f64>, b: PyReadonlyArray2<f64>, max_distance: f64, dbh_tolerance: f64, max_shrink: f64, dbh_weight: f64, height_weight: f64, merge_factor: f64) -> PyResult<Bound<'py, PyDict>> {
    let rows = |v: Vec<[f64; 4]>| v.into_iter().map(|r| trees::TreeRow { x: r[0], y: r[1], dbh: r[2], height: r[3] }).collect::<Vec<_>>();
    let (ta, tb) = (rows(rows_from_py::<4>(a, "a")?), rows(rows_from_py::<4>(b, "b")?));
    let p = trees::MatchParams { max_distance, dbh_tolerance, max_shrink, dbh_weight, height_weight, merge_factor };
    let m = py.detach(|| trees::match_trees(&ta, &tb, &p)).map_err(err)?;
    let d = PyDict::new(py);
    let flat: Vec<i64> = m.pairs.iter().flat_map(|&(i, j)| [i as i64, j as i64]).collect();
    d.set_item("pairs", PyArray1::from_vec(py, flat).reshape([m.pairs.len(), 2])?)?;
    d.set_item("distance", m.distance.into_pyarray(py))?;
    d.set_item("cost", m.cost.into_pyarray(py))?;
    d.set_item("status_a", m.status_a.iter().map(|s| s.name()).collect::<Vec<_>>())?;
    d.set_item("status_b", m.status_b.iter().map(|s| s.name()).collect::<Vec<_>>())?;
    let rel = |v: &[Option<usize>]| v.iter().map(|r| r.map(|x| x as i64).unwrap_or(-1)).collect::<Vec<_>>();
    d.set_item("related_a", rel(&m.related_a).into_pyarray(py))?;
    d.set_item("related_b", rel(&m.related_b).into_pyarray(py))?;
    Ok(d)
}

fn measures_to_py<'py>(py: Python<'py>, m: &[trees::TreeMeasure], n_slices: usize) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("diameter", matrix_to_py(py, &m.iter().map(|t| t.diameter.clone()).collect::<Vec<_>>(), n_slices))?;
    d.set_item("diameter_se", matrix_to_py(py, &m.iter().map(|t| t.diameter_se.clone()).collect::<Vec<_>>(), n_slices))?;
    for (name, get) in [
        ("dbh", (|t: &trees::TreeMeasure| t.dbh) as fn(&trees::TreeMeasure) -> f64),
        ("dbh_se", |t| t.dbh_se),
        ("taper", |t| t.taper),
        ("height", |t| t.height),
        ("height_se", |t| t.height_se),
        ("crown_area", |t| t.crown_area),
        ("crown_volume", |t| t.crown_volume),
    ] {
        d.set_item(name, m.iter().map(get).collect::<Vec<_>>().into_pyarray(py))?;
    }
    d.set_item("n_points", m.iter().map(|t| t.n_points as i64).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
fn change_tree_increments<'py>(
    py: Python<'py>,
    xyz_a: PyReadonlyArray2<f64>,
    heights_a: PyReadonlyArray1<f64>,
    labels_a: PyReadonlyArray1<i64>,
    trees_a: PyReadonlyArray2<f64>,
    sigma_a: f64,
    xyz_b: PyReadonlyArray2<f64>,
    heights_b: PyReadonlyArray1<f64>,
    labels_b: PyReadonlyArray1<i64>,
    trees_b: PyReadonlyArray2<f64>,
    sigma_b: f64,
    pairs: PyReadonlyArray2<i64>,
    registration_sigma: f64,
    slice_heights: Vec<f64>,
    slice_thickness: f64,
    search_radius: f64,
    min_slices: usize,
    confidence: f64,
    max_dbh_increment: f64,
    top_points: usize,
    height_error: f64,
    top_radius: f64,
    top_gap: f64,
    slice_correlation: f64,
    dbh_accuracy: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(PyValueError::new_err(format!("confidence must be in (0, 1), got {confidence}")));
    }
    let p = trees::IncrementParams { slice_heights, slice_thickness, search_radius, min_slices, confidence, max_dbh_increment, top_points, height_error, top_radius, top_gap, inlier_threshold: (3.0 * sigma_a.max(sigma_b)).max(0.01), slice_correlation, dbh_accuracy };
    if !(0.0..=1.0).contains(&slice_correlation) || dbh_accuracy.is_nan() || dbh_accuracy < 0.0 {
        return Err(PyValueError::new_err("slice_correlation must be in [0, 1] and dbh_accuracy >= 0"));
    }
    if height_error.is_nan() || height_error < 0.0 {
        return Err(PyValueError::new_err(format!("height_error must be >= 0, got {height_error}")));
    }
    let pa = xyz_from_py(xyz_a)?;
    let pb = xyz_from_py(xyz_b)?;
    let (ha, hb) = (heights_a.as_array().to_vec(), heights_b.as_array().to_vec());
    let (la, lb) = (labels_a.as_array().to_vec(), labels_b.as_array().to_vec());
    let ids = |v: Vec<[f64; 3]>| v.into_iter().map(|r| (r[0] as i64, r[1], r[2])).collect::<Vec<_>>();
    let (ta, tb) = (ids(rows_from_py::<3>(trees_a, "trees_a")?), ids(rows_from_py::<3>(trees_b, "trees_b")?));
    let pv = pairs.as_array();
    if pv.ncols() != 2 {
        return Err(PyValueError::new_err("pairs must have shape (n, 2)"));
    }
    let pr: Vec<(usize, usize)> = pv.rows().into_iter().map(|r| (r[0], r[1])).map(|(i, j)| (i as usize, j as usize)).collect();
    if pr.iter().any(|&(i, j)| i >= ta.len() || j >= tb.len()) {
        return Err(PyValueError::new_err("a pair refers to a tree that is not in the tables"));
    }
    let (ma, mb, inc) = py
        .detach(|| -> sylva_rs::Result<_> {
            let ma = trees::measure_trees(&pa, &ha, &la, &ta, sigma_a, &p)?;
            let mb = trees::measure_trees(&pb, &hb, &lb, &tb, sigma_b, &p)?;
            let inc: Vec<trees::Increment> = pr.iter().map(|&(i, j)| trees::increment(&ma[i], &mb[j], registration_sigma, &p)).collect();
            Ok((ma, mb, inc))
        })
        .map_err(err)?;
    let k = p.slice_heights.len();
    let d = PyDict::new(py);
    d.set_item("a", measures_to_py(py, &ma, k)?)?;
    d.set_item("b", measures_to_py(py, &mb, k)?)?;
    let c = PyDict::new(py);
    for (name, get) in [
        ("d_dbh", (|t: &trees::Increment| t.d_dbh) as fn(&trees::Increment) -> f64),
        ("d_dbh_se", |t| t.d_dbh_se),
        ("d_dbh_mdi", |t| t.d_dbh_mdi),
        ("d_height", |t| t.d_height),
        ("d_height_se", |t| t.d_height_se),
        ("d_height_mdi", |t| t.d_height_mdi),
        ("d_crown_area", |t| t.d_crown_area),
        ("d_crown_volume", |t| t.d_crown_volume),
    ] {
        c.set_item(name, inc.iter().map(get).collect::<Vec<_>>().into_pyarray(py))?;
    }
    c.set_item("n_slices", inc.iter().map(|t| t.n_slices as i64).collect::<Vec<_>>().into_pyarray(py))?;
    c.set_item("dbh_change", inc.iter().map(|t| t.dbh_change.name()).collect::<Vec<_>>())?;
    c.set_item("height_change", inc.iter().map(|t| t.height_change.name()).collect::<Vec<_>>())?;
    c.set_item("implausible", inc.iter().map(|t| t.implausible).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("increments", c)?;
    Ok(d)
}

// --------------------------------------------------------------- summary

#[pyfunction]
fn change_plot_summary(py: Python<'_>, survivors: PyReadonlyArray2<f64>, dead: PyReadonlyArray2<f64>, recruits: PyReadonlyArray2<f64>, n_ambiguous: usize, area: f64, years: f64, n_draws: usize, confidence: f64, form_factor: f64, wood_density: f64, seed: u64) -> PyResult<Vec<(String, String, f64, f64, f64)>> {
    let s: Vec<summary::Survivor> = rows_from_py::<8>(survivors, "survivors")?
        .into_iter()
        .map(|r| summary::Survivor { dbh: r[0], dbh_se: r[1], height: r[2], height_se: r[3], d_dbh: r[4], d_dbh_se: r[5], d_height: r[6], d_height_se: r[7] })
        .collect();
    let single = |v: Vec<[f64; 4]>| v.into_iter().map(|r| summary::Single { dbh: r[0], dbh_se: r[1], height: r[2], height_se: r[3] }).collect::<Vec<_>>();
    let (d, r) = (single(rows_from_py::<4>(dead, "dead")?), single(rows_from_py::<4>(recruits, "recruits")?));
    let p = summary::SummaryParams { area, years, n_draws, confidence, form_factor, wood_density, seed };
    let q = py.detach(|| summary::plot_summary(&s, &d, &r, n_ambiguous, &p)).map_err(err)?;
    Ok(q.into_iter().map(|x| (x.name.to_string(), x.unit.to_string(), x.estimate, x.low, x.high)).collect())
}

#[pyfunction]
fn change_provenance_differences(a: &str, b: &str, rtol: f64) -> PyResult<Vec<(String, Option<String>, Option<String>)>> {
    provenance::differences(a, b, rtol).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(change_forest_epochs, m)?,
        wrap_pyfunction!(change_align_on_stable, m)?,
        wrap_pyfunction!(change_match_trees, m)?,
        wrap_pyfunction!(change_tree_increments, m)?,
        wrap_pyfunction!(change_plot_summary, m)?,
        wrap_pyfunction!(change_provenance_differences, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
