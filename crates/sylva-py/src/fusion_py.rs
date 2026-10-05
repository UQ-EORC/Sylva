// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::fusion (TLS and ALS together).
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use nalgebra::Matrix4;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::fusion::{link, merge, register as reg, synthetic, upscale};

use crate::{cloud_from_py, err, raster_from_py, shots_to_py, xyz_from_py};

fn mat4_from_py(m: PyReadonlyArray2<f64>) -> PyResult<Matrix4<f64>> {
    let a = m.as_array();
    if a.shape() != [4, 4] {
        return Err(PyValueError::new_err(format!("a transform must have shape (4, 4), got {:?}", a.shape())));
    }
    Ok(Matrix4::from_row_slice(&a.iter().copied().collect::<Vec<_>>()))
}

fn mat_to_py<'py>(py: Python<'py>, m: &[f64], rows: usize, cols: usize) -> Bound<'py, PyArray2<f64>> {
    PyArray1::from_vec(py, m.to_vec()).reshape([rows, cols]).expect("reshape")
}

fn mat4_to_py<'py>(py: Python<'py>, m: &Matrix4<f64>) -> Bound<'py, PyArray2<f64>> {
    let v: Vec<f64> = (0..4).flat_map(|r| (0..4).map(move |c| m[(r, c)])).collect();
    mat_to_py(py, &v, 4, 4)
}

fn bools(a: PyReadonlyArray1<bool>) -> Vec<bool> {
    a.as_array().to_vec()
}

// ----------------------------------------------------------- registration

#[pyfunction]
#[pyo3(signature = (tls, tls_ground, als, als_ground, initial, resolution, returns_per_cell, coarse_resolution, search_radius, heading_range, heading_step, dtm_weight, min_height, min_overlap, n_candidates, refine, icp_voxel_sizes, icp_max_distances, min_planarity, max_refine_shift, max_refine_turn, jackknife))]
fn fusion_register<'py>(
    py: Python<'py>,
    tls: PyReadonlyArray2<f64>,
    tls_ground: PyReadonlyArray1<bool>,
    als: PyReadonlyArray2<f64>,
    als_ground: PyReadonlyArray1<bool>,
    initial: PyReadonlyArray2<f64>,
    resolution: f64,
    returns_per_cell: f64,
    coarse_resolution: f64,
    search_radius: f64,
    heading_range: f64,
    heading_step: f64,
    dtm_weight: f64,
    min_height: f64,
    min_overlap: f64,
    n_candidates: usize,
    refine: &str,
    icp_voxel_sizes: Vec<f64>,
    icp_max_distances: Vec<f64>,
    min_planarity: f64,
    max_refine_shift: f64,
    max_refine_turn: f64,
    jackknife: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p = reg::RegisterParams {
        resolution,
        returns_per_cell,
        coarse_resolution,
        search_radius,
        heading_range,
        heading_step,
        dtm_weight,
        min_height,
        min_overlap,
        n_candidates,
        refine: reg::Refine::parse(refine).map_err(err)?,
        icp_voxel_sizes,
        icp_max_distances,
        min_planarity,
        max_refine_shift,
        max_refine_turn,
        jackknife,
    };
    let (t, tg, a, ag) = (xyz_from_py(tls)?, bools(tls_ground), xyz_from_py(als)?, bools(als_ground));
    let init = mat4_from_py(initial)?;
    let r = py.detach(|| reg::register(&t, &tg, &a, &ag, &init, &p)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("resolution", r.resolution)?;
    d.set_item("coarse_resolution", r.coarse_resolution)?;
    d.set_item("als_density", r.als_density)?;
    d.set_item("transform", mat4_to_py(py, &r.transform))?;
    d.set_item("search_transform", mat4_to_py(py, &r.search_transform))?;
    d.set_item("pivot", r.pivot.to_vec().into_pyarray(py))?;
    d.set_item("heading", r.heading)?;
    d.set_item("shift", r.shift.to_vec().into_pyarray(py))?;
    let c = PyDict::new(py);
    for (name, get) in [
        ("heading", (|q: &reg::Candidate| q.heading) as fn(&reg::Candidate) -> f64),
        ("dx", |q| q.dx),
        ("dy", |q| q.dy),
        ("score", |q| q.score),
        ("chm_r", |q| q.chm_r),
        ("dtm_r", |q| q.dtm_r),
    ] {
        c.set_item(name, r.candidates.iter().map(get).collect::<Vec<_>>().into_pyarray(py))?;
    }
    c.set_item("overlap", r.candidates.iter().map(|q| q.overlap as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("candidates", c)?;
    d.set_item("ambiguity", r.ambiguity)?;
    match &r.icp {
        Some(i) => {
            let v = PyDict::new(py);
            v.set_item("fitness", i.fitness)?;
            v.set_item("rmse", i.rmse)?;
            v.set_item("n", i.n)?;
            v.set_item("iterations", i.iterations)?;
            v.set_item("converged", i.converged)?;
            v.set_item("accepted", i.accepted)?;
            v.set_item("shift", i.shift)?;
            v.set_item("turn", i.turn)?;
            v.set_item("start", i.start)?;
            d.set_item("icp", v)?;
        }
        None => d.set_item("icp", py.None())?,
    }
    match &r.covariance {
        Some(m) => {
            let v: Vec<f64> = (0..6).flat_map(|i| (0..6).map(move |j| m[(i, j)])).collect();
            d.set_item("covariance", mat_to_py(py, &v, 6, 6))?;
        }
        None => d.set_item("covariance", py.None())?,
    }
    match &r.jackknife {
        Some(j) => d.set_item("jackknife", j.to_vec().into_pyarray(py))?,
        None => d.set_item("jackknife", py.None())?,
    }
    let s = &r.residuals;
    let res = PyDict::new(py);
    res.set_item("ground_n", s.ground_n)?;
    res.set_item("ground_median", s.ground_median)?;
    res.set_item("ground_nmad", s.ground_nmad)?;
    res.set_item("ground_rmse", s.ground_rmse)?;
    res.set_item("canopy_n", s.canopy_n)?;
    res.set_item("canopy_median", s.canopy_median)?;
    res.set_item("canopy_p90", s.canopy_p90)?;
    d.set_item("residuals", res)?;
    d.set_item("n_tls", r.n_tls)?;
    d.set_item("n_als", r.n_als)?;
    Ok(d)
}

// ------------------------------------------------------------------ trees

#[pyfunction]
#[pyo3(signature = (tls, als, crowns, max_distance, crown_buffer, height_weight, dbh_weight, max_cost, top_tolerance))]
fn fusion_link_trees<'py>(py: Python<'py>, tls: PyReadonlyArray2<f64>, als: PyReadonlyArray2<f64>, crowns: Vec<PyReadonlyArray2<f64>>, max_distance: f64, crown_buffer: f64, height_weight: f64, dbh_weight: f64, max_cost: f64, top_tolerance: f64) -> PyResult<Bound<'py, PyDict>> {
    let (t, a) = (tls.as_array(), als.as_array());
    if t.ncols() != 6 || a.ncols() != 3 {
        return Err(PyValueError::new_err("tls must be (n, 6) [x, y, dbh, height, volume, top_seen] and als (m, 3) [x, y, height]"));
    }
    if crowns.len() != a.nrows() {
        return Err(PyValueError::new_err(format!("{} crowns for {} ALS trees", crowns.len(), a.nrows())));
    }
    let tls_rows: Vec<link::TlsTree> = t.rows().into_iter().map(|r| link::TlsTree { x: r[0], y: r[1], dbh: r[2], height: r[3], volume: r[4], top_seen: r[5] }).collect();
    let mut als_rows = Vec::with_capacity(a.nrows());
    for (r, c) in a.rows().into_iter().zip(&crowns) {
        let c = c.as_array();
        if c.nrows() > 0 && c.ncols() != 2 {
            return Err(PyValueError::new_err("crowns must be (k, 2) arrays"));
        }
        als_rows.push(link::AlsTree { x: r[0], y: r[1], height: r[2], crown: c.rows().into_iter().map(|v| [v[0], v[1]]).collect() });
    }
    let p = link::LinkParams { max_distance, crown_buffer, height_weight, dbh_weight, max_cost, top_tolerance };
    let l = link::link_trees(&tls_rows, &als_rows, &p).map_err(err)?;
    let idx = |v: &[Option<usize>]| v.iter().map(|o| o.map(|i| i as i64).unwrap_or(-1)).collect::<Vec<_>>();
    let d = PyDict::new(py);
    d.set_item("status", l.tls_status.iter().map(|s| s.name()).collect::<Vec<_>>())?;
    d.set_item("als", idx(&l.tls_als).into_pyarray(py))?;
    d.set_item("distance", l.distance.into_pyarray(py))?;
    d.set_item("inside", l.inside.into_pyarray(py))?;
    d.set_item("als_height", l.als_height.into_pyarray(py))?;
    d.set_item("height", l.height.into_pyarray(py))?;
    d.set_item("height_source", l.height_source)?;
    d.set_item("top_seen", l.top_seen.into_pyarray(py))?;
    d.set_item("flag", l.flag)?;
    d.set_item("cost", l.cost.into_pyarray(py))?;
    d.set_item("als_tls", idx(&l.als_tls).into_pyarray(py))?;
    let stems = PyList::empty(py);
    for s in &l.als_stems {
        stems.append(s.iter().map(|&i| i as i64).collect::<Vec<_>>().into_pyarray(py))?;
    }
    d.set_item("als_stems", stems)?;
    Ok(d)
}

// ------------------------------------------------------------ clouds, profiles

#[pyfunction]
#[pyo3(signature = (xyz, heights, split_height, split_raster, fallback, below))]
fn fusion_select<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, split_height: Option<f64>, split_raster: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, fallback: f64, below: bool) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let pts = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let split = match (split_height, split_raster) {
        (Some(v), None) => merge::Split::Height(v),
        (None, Some((d, x, y, r))) => merge::Split::Raster { raster: raster_from_py(d, x, y, r), fallback },
        _ => return Err(PyValueError::new_err("give a split height or a split raster")),
    };
    let keep = py.detach(|| merge::select(&pts, &h, &split, below)).map_err(err)?;
    Ok(keep.into_pyarray(py))
}

fn flat3(a: &Option<PyReadonlyArray3<f64>>) -> Option<Vec<f64>> {
    a.as_ref().map(|v| v.as_array().iter().copied().collect())
}

#[pyfunction]
#[pyo3(signature = (origin, voxel_size, beams, pad, hits, path, mask, dtm, bin_size, min_beams, g))]
fn fusion_layer_stats<'py>(
    py: Python<'py>,
    origin: (f64, f64, f64),
    voxel_size: f64,
    beams: PyReadonlyArray3<f64>,
    pad: Option<PyReadonlyArray3<f64>>,
    hits: Option<PyReadonlyArray3<f64>>,
    path: Option<PyReadonlyArray3<f64>>,
    mask: Option<PyReadonlyArray2<bool>>,
    dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>,
    bin_size: f64,
    min_beams: f64,
    g: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let s = beams.shape();
    let shape = [s[0], s[1], s[2]];
    for a in [&pad, &hits, &path].into_iter().flatten() {
        if a.shape() != s {
            return Err(PyValueError::new_err(format!("every voxel field must have shape {s:?}, got {:?}", a.shape())));
        }
    }
    let b: Vec<f64> = beams.as_array().iter().copied().collect();
    let (pv, hv, qv) = (flat3(&pad), flat3(&hits), flat3(&path));
    let m: Option<Vec<bool>> = mask.map(|m| m.as_array().iter().copied().collect());
    let dtm = dtm.map(|(d, x, y, r)| raster_from_py(d, x, y, r));
    let v = merge::VoxelFields { origin: [origin.0, origin.1, origin.2], voxel_size, shape, beams: &b, pad: pv.as_deref(), hits: hv.as_deref(), path: qv.as_deref() };
    let st = py.detach(|| merge::layer_stats(&v, m.as_deref(), dtm.as_ref(), bin_size, min_beams, g)).map_err(err)?;
    stats_to_py(py, &st)
}

fn stats_to_py<'py>(py: Python<'py>, s: &merge::LayerStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("height", s.height.clone().into_pyarray(py))?;
    d.set_item("pad", s.pad.clone().into_pyarray(py))?;
    d.set_item("beams", s.beams.clone().into_pyarray(py))?;
    d.set_item("observed", s.observed.clone().into_pyarray(py))?;
    d.set_item("n_voxels", s.n_voxels.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(d)
}

fn stats_from_py(d: &Bound<'_, PyDict>) -> PyResult<merge::LayerStats> {
    let get = |k: &str| -> PyResult<Vec<f64>> {
        let v = d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("layer statistics need {k:?}")))?;
        Ok(v.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec())
    };
    let s = merge::LayerStats { height: get("height")?, pad: get("pad")?, beams: get("beams")?, observed: get("observed")?, n_voxels: Vec::new() };
    let n = s.height.len();
    if s.pad.len() != n || s.beams.len() != n || s.observed.len() != n {
        return Err(PyValueError::new_err("layer statistics must have one value per bin"));
    }
    let nv = match d.get_item("n_voxels")? {
        Some(v) => v.extract::<PyReadonlyArray1<i64>>()?.as_array().iter().map(|&x| x.max(0) as usize).collect(),
        None => vec![0; n],
    };
    Ok(merge::LayerStats { n_voxels: nv, ..s })
}

#[pyfunction]
#[pyo3(signature = (tls, als, bin_size, mode, min_observed))]
fn fusion_fuse<'py>(py: Python<'py>, tls: Bound<'py, PyDict>, als: Bound<'py, PyDict>, bin_size: f64, mode: &str, min_observed: f64) -> PyResult<Bound<'py, PyDict>> {
    let (t, a) = (stats_from_py(&tls)?, stats_from_py(&als)?);
    let f = merge::fuse(&t, &a, bin_size, merge::FuseMode::parse(mode).map_err(err)?, min_observed).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("height", f.height.into_pyarray(py))?;
    d.set_item("pad", f.pad.into_pyarray(py))?;
    d.set_item("weight_tls", f.weight_tls.into_pyarray(py))?;
    d.set_item("weight_als", f.weight_als.into_pyarray(py))?;
    d.set_item("tls", stats_to_py(py, &f.tls)?)?;
    d.set_item("als", stats_to_py(py, &f.als)?)?;
    d.set_item("split_height", f.split_height)?;
    Ok(d)
}

// --------------------------------------------------------------- upscaling

fn columns(x: PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    let a = x.as_array();
    (0..a.ncols()).map(|j| a.column(j).to_vec()).collect()
}

#[pyfunction]
#[pyo3(signature = (y, x, form))]
fn fusion_fit<'py>(py: Python<'py>, y: PyReadonlyArray1<f64>, x: PyReadonlyArray2<f64>, form: &str) -> PyResult<Bound<'py, PyDict>> {
    let yv = y.as_array().to_vec();
    if x.as_array().nrows() != yv.len() {
        return Err(PyValueError::new_err(format!("{} rows of predictors for {} plot values", x.as_array().nrows(), yv.len())));
    }
    let f = upscale::fit(&yv, &columns(x), upscale::Form::parse(form).map_err(err)?).map_err(err)?;
    let k = f.coef.len();
    let d = PyDict::new(py);
    d.set_item("form", f.form.name())?;
    d.set_item("coef", f.coef.into_pyarray(py))?;
    d.set_item("se", f.se.into_pyarray(py))?;
    d.set_item("sigma", f.sigma)?;
    d.set_item("df", f.df)?;
    d.set_item("n", f.n)?;
    d.set_item("r2", f.r2)?;
    d.set_item("adj_r2", f.adj_r2)?;
    d.set_item("xtx_inv", mat_to_py(py, &f.xtx_inv, k, k))?;
    d.set_item("correction", f.correction)?;
    d.set_item("fitted", f.fitted.into_pyarray(py))?;
    d.set_item("loo", f.loo.into_pyarray(py))?;
    d.set_item("loo_rmse", f.loo_rmse)?;
    d.set_item("loo_bias", f.loo_bias)?;
    d.set_item("loo_r2", f.loo_r2)?;
    d.set_item("loo_rrmse", f.loo_rrmse)?;
    d.set_item("x_min", f.x_min.into_pyarray(py))?;
    d.set_item("x_max", f.x_max.into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (form, coef, sigma, df, xtx_inv, x_min, x_max, x, level))]
fn fusion_predict<'py>(py: Python<'py>, form: &str, coef: Vec<f64>, sigma: f64, df: usize, xtx_inv: PyReadonlyArray2<f64>, x_min: Vec<f64>, x_max: Vec<f64>, x: PyReadonlyArray2<f64>, level: f64) -> PyResult<Bound<'py, PyDict>> {
    let k = coef.len();
    let inv = xtx_inv.as_array();
    if inv.shape() != [k, k] || x_min.len() + 1 != k || x_max.len() + 1 != k {
        return Err(PyValueError::new_err("the model's arrays do not agree in size"));
    }
    let f = upscale::Fit {
        form: upscale::Form::parse(form).map_err(err)?,
        coef,
        se: Vec::new(),
        sigma,
        df,
        n: 0,
        r2: f64::NAN,
        adj_r2: f64::NAN,
        xtx_inv: inv.iter().copied().collect(),
        correction: f64::NAN,
        fitted: Vec::new(),
        loo: Vec::new(),
        loo_rmse: f64::NAN,
        loo_bias: f64::NAN,
        loo_r2: f64::NAN,
        loo_rrmse: f64::NAN,
        x_min,
        x_max,
    };
    let cols = columns(x);
    let p = py.detach(|| upscale::predict(&f, &cols, level)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("mean", p.mean.into_pyarray(py))?;
    d.set_item("se", p.se.into_pyarray(py))?;
    d.set_item("lower", p.lower.into_pyarray(py))?;
    d.set_item("upper", p.upper.into_pyarray(py))?;
    d.set_item("extrapolated", p.extrapolated.into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
fn fusion_t_quantile(p: f64, df: f64) -> f64 {
    upscale::t_quantile(p, df)
}

#[pyfunction]
#[pyo3(signature = (dbh, volume, area, wood_density, min_dbh))]
fn fusion_plot_summary<'py>(py: Python<'py>, dbh: PyReadonlyArray1<f64>, volume: PyReadonlyArray1<f64>, area: f64, wood_density: f64, min_dbh: f64) -> PyResult<Bound<'py, PyDict>> {
    let s = upscale::plot_summary(dbh.as_slice()?, volume.as_slice()?, area, wood_density, min_dbh).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("agb", s.agb)?;
    d.set_item("volume", s.volume)?;
    d.set_item("basal_area", s.basal_area)?;
    d.set_item("stems", s.stems)?;
    d.set_item("n_trees", s.n_trees)?;
    d.set_item("missing_volume", s.missing_volume)?;
    Ok(d)
}

// --------------------------------------------------------------- synthetic

#[pyfunction]
#[pyo3(signature = (xyz, attrs, origins, resolution_deg, max_zenith_deg, target_radius, terrain_slope, min_range, max_range))]
fn fusion_scan_spheres<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<Bound<'py, PyDict>>, origins: PyReadonlyArray2<f64>, resolution_deg: f64, max_zenith_deg: f64, target_radius: f64, terrain_slope: f64, min_range: f64, max_range: f64) -> PyResult<Bound<'py, PyList>> {
    let scene = cloud_from_py(xyz, attrs.as_ref())?;
    let o = xyz_from_py(origins)?;
    let p = synthetic::SphereScan { resolution_deg, max_zenith_deg, target_radius, terrain_slope, min_range, max_range };
    let shots = py.detach(|| synthetic::scan_spheres(&scene, &o, &p)).map_err(err)?;
    let out = PyList::empty(py);
    for s in &shots {
        out.append(shots_to_py(py, s)?)?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(fusion_register, m)?,
        wrap_pyfunction!(fusion_link_trees, m)?,
        wrap_pyfunction!(fusion_select, m)?,
        wrap_pyfunction!(fusion_layer_stats, m)?,
        wrap_pyfunction!(fusion_fuse, m)?,
        wrap_pyfunction!(fusion_fit, m)?,
        wrap_pyfunction!(fusion_predict, m)?,
        wrap_pyfunction!(fusion_t_quantile, m)?,
        wrap_pyfunction!(fusion_plot_summary, m)?,
        wrap_pyfunction!(fusion_scan_spheres, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
