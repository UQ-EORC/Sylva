// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::change::als: change between airborne surveys.
//!
//! Rasters cross as `(data, xmin, ymin, resolution)` tuples one way and
//! dicts the other, catalogues as the dicts of `als_py`, and an alignment as
//! the dict `change_als_align` returns.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::als::canopy::ProfileGrid;
use sylva_rs::als::metrics::MetricParams;
use sylva_rs::change::als::align::{self, AlignParams, Alignment, Model};
use sylva_rs::change::als::gaps::{self, GapParams, Gaps};
use sylva_rs::change::als::metrics::{self, MetricChangeParams};
use sylva_rs::change::als::surface::{self, DtmKind, EpochStats, Surface, SurfaceParams};
use sylva_rs::change::als::trees::{self, AlsTree, CanopyChange, TreeChangeParams, TreeChangeRow};
use sylva_rs::change::als::Harmonise;
use sylva_rs::masks::Polygon;
use sylva_rs::Raster;

use crate::als_py::{catalog_from_py, layout};
use crate::{err, raster_from_py, raster_to_py};

type RasterArg<'py> = (PyReadonlyArray2<'py, f64>, f64, f64, f64);

fn raster(r: RasterArg) -> Raster {
    raster_from_py(r.0, r.1, r.2, r.3)
}

fn arr3<'py>(py: Python<'py>, v: &[[f64; 3]]) -> PyResult<Bound<'py, PyAny>> {
    let n = v.len();
    Ok(PyArray1::from_vec(py, v.iter().flatten().copied().collect()).reshape([n, 3])?.into_any())
}

fn model(name: &str, smoothing: f64) -> PyResult<Model> {
    Ok(match name {
        "constant" => Model::Constant,
        "blocks" => Model::Blocks,
        "field" => Model::Field { smoothing },
        other => return Err(PyValueError::new_err(format!("model must be 'constant', 'blocks' or 'field', got {other:?}"))),
    })
}

fn alignment_to_py<'py>(py: Python<'py>, a: &Alignment) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("xmin", a.xmin)?;
    d.set_item("ymin", a.ymin)?;
    d.set_item("block_size", a.block_size)?;
    d.set_item("nx", a.nx)?;
    d.set_item("ny", a.ny)?;
    d.set_item("model", a.model.name())?;
    d.set_item("smoothing", match a.model {
        Model::Field { smoothing } => smoothing,
        _ => f64::NAN,
    })?;
    d.set_item("values", arr3(py, &a.values)?)?;
    d.set_item("sigmas", arr3(py, &a.sigmas)?)?;
    d.set_item("global", a.global.to_vec())?;
    d.set_item("global_sigma", a.global_sigma.to_vec())?;
    d.set_item("birge", a.birge.to_vec())?;
    d.set_item("noise_a", a.noise_a)?;
    d.set_item("noise_b", a.noise_b)?;
    d.set_item("residuals", arr3(py, &a.residuals())?)?;
    let n = a.nx * a.ny;
    let f = |g: &dyn Fn(&align::BlockFit) -> f64| -> Vec<f64> { (0..n).map(|i| a.blocks.get(i).and_then(|b| b.as_ref()).map_or(f64::NAN, g)).collect() };
    let b = PyDict::new(py);
    b.set_item("fitted", (0..n).map(|i| a.blocks.get(i).is_some_and(|b| b.is_some())).collect::<Vec<bool>>().into_pyarray(py))?;
    b.set_item("n_samples", f(&|b| b.n_samples as f64).into_pyarray(py))?;
    b.set_item("n_used", f(&|b| b.n_used as f64).into_pyarray(py))?;
    b.set_item("n_eff", f(&|b| b.n_eff as f64).into_pyarray(py))?;
    let off: Vec<[f64; 3]> = (0..n).map(|i| a.blocks.get(i).and_then(|b| b.as_ref()).map_or([f64::NAN; 3], |b| b.offset)).collect();
    let sd: Vec<[f64; 3]> = (0..n).map(|i| a.blocks.get(i).and_then(|b| b.as_ref()).map_or([f64::NAN; 3], |b| [0, 1, 2].map(|k| b.cov[k][k].max(0.0).sqrt()))).collect();
    b.set_item("offset", arr3(py, &off)?)?;
    b.set_item("sigma", arr3(py, &sd)?)?;
    b.set_item("median_before", f(&|b| b.median_before).into_pyarray(py))?;
    b.set_item("spread_before", f(&|b| b.spread_before).into_pyarray(py))?;
    b.set_item("spread_after", f(&|b| b.spread_after).into_pyarray(py))?;
    b.set_item("horizontal_determined", (0..n).map(|i| a.blocks.get(i).and_then(|b| b.as_ref()).is_some_and(|b| b.horizontal_determined)).collect::<Vec<bool>>().into_pyarray(py))?;
    b.set_item("noise_a", f(&|b| b.noise_a).into_pyarray(py))?;
    b.set_item("noise_b", f(&|b| b.noise_b).into_pyarray(py))?;
    b.set_item("iterations", f(&|b| b.iterations as f64).into_pyarray(py))?;
    d.set_item("blocks", b)?;
    Ok(d)
}

pub(crate) fn alignment_from_py(d: &Bound<'_, PyDict>) -> PyResult<Alignment> {
    let get = |k: &str| d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("alignment is missing {k:?}")));
    let v3 = |k: &str| -> PyResult<Vec<[f64; 3]>> {
        let a: PyReadonlyArray2<f64> = get(k)?.extract()?;
        let a = a.as_array();
        if a.ncols() != 3 {
            return Err(PyValueError::new_err(format!("alignment {k:?} must have 3 columns")));
        }
        Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
    };
    let g3 = |k: &str| -> PyResult<[f64; 3]> {
        let v: Vec<f64> = get(k)?.extract()?;
        v.try_into().map_err(|_| PyValueError::new_err(format!("alignment {k:?} must have 3 values")))
    };
    let name: String = get("model")?.extract()?;
    let m = model(&name, get("smoothing")?.extract()?)?;
    Alignment::from_field(get("xmin")?.extract()?, get("ymin")?.extract()?, get("block_size")?.extract()?, get("nx")?.extract()?, get("ny")?.extract()?, m, v3("values")?, v3("sigmas")?, g3("global")?, g3("global_sigma")?).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (catalog_a, catalog_b, block_size, stable_classes, sample_spacing, radius, min_neighbours, max_roughness, max_slope, max_offset, horizontal, horizontal_prior, correlation_length, min_samples, huber, iterations, model_name, smoothing, blocks_per_chunk, workers))]
fn change_als_align<'py>(py: Python<'py>, catalog_a: &Bound<'_, PyDict>, catalog_b: &Bound<'_, PyDict>, block_size: f64, stable_classes: Vec<u8>, sample_spacing: f64, radius: f64, min_neighbours: usize, max_roughness: f64, max_slope: f64, max_offset: f64, horizontal: bool, horizontal_prior: f64, correlation_length: f64, min_samples: usize, huber: f64, iterations: usize, model_name: &str, smoothing: f64, blocks_per_chunk: usize, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (catalog_from_py(catalog_a)?, catalog_from_py(catalog_b)?);
    let p = AlignParams { block_size, stable_classes, sample_spacing, radius, min_neighbours, max_roughness, max_slope, max_offset, horizontal, horizontal_prior, correlation_length, min_samples, huber, iterations, model: model(model_name, smoothing)? };
    let al = py.detach(|| align::align(&a, &b, &p, blocks_per_chunk, workers)).map_err(err)?;
    alignment_to_py(py, &al)
}

/// Offsets and their standard deviations at points.
#[pyfunction]
fn change_als_offsets<'py>(py: Python<'py>, alignment: &Bound<'_, PyDict>, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let al = alignment_from_py(alignment)?;
    let (x, y) = (x.as_array(), y.as_array());
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y differ in length"));
    }
    let o: Vec<[f64; 3]> = x.iter().zip(y.iter()).map(|(&a, &b)| al.offset_at(a, b)).collect();
    let s: Vec<[f64; 3]> = x.iter().zip(y.iter()).map(|(&a, &b)| al.sigma_at(a, b)).collect();
    Ok((arr3(py, &o)?, arr3(py, &s)?))
}

fn stats_to_py<'py>(py: Python<'py>, s: &EpochStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("returns", s.n_returns)?;
    d.set_item("pulses", s.n_first)?;
    d.set_item("area", s.area)?;
    d.set_item("pulse_density", s.pulse_density())?;
    d.set_item("returns_per_pulse", s.returns_per_pulse())?;
    d.set_item("single_return_share", s.single_share())?;
    d.set_item("mean_abs_scan_angle", s.mean_abs_scan_angle())?;
    d.set_item("max_returns", s.max_returns)?;
    Ok(d)
}

fn dtm_kind(s: &str) -> PyResult<DtmKind> {
    match s {
        "plane" => Ok(DtmKind::Plane),
        "tin" => Ok(DtmKind::Tin),
        "lowest" => Ok(DtmKind::Lowest),
        other => Err(PyValueError::new_err(format!("dtm_method must be 'plane', 'tin' or 'lowest', got {other:?}"))),
    }
}

#[pyfunction]
#[pyo3(signature = (catalog_a, catalog_b, alignment, surface_name, resolution, dtm_resolution, dtm_method, first_returns, subcircle, min_returns, noise_a, noise_b, interpolation_error, confidence, harmonise_cell, seed, horizontal_sigma, vertical_sigma, chunk_size, buffer, workers))]
fn change_als_surface<'py>(py: Python<'py>, catalog_a: &Bound<'_, PyDict>, catalog_b: &Bound<'_, PyDict>, alignment: Option<&Bound<'_, PyDict>>, surface_name: &str, resolution: f64, dtm_resolution: f64, dtm_method: &str, first_returns: bool, subcircle: f64, min_returns: usize, noise_a: f64, noise_b: f64, interpolation_error: f64, confidence: f64, harmonise_cell: Option<f64>, seed: u64, horizontal_sigma: f64, vertical_sigma: f64, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (catalog_from_py(catalog_a)?, catalog_from_py(catalog_b)?);
    let al = alignment.map(alignment_from_py).transpose()?;
    let p = SurfaceParams {
        surface: Surface::parse(surface_name).map_err(err)?,
        resolution,
        dtm_resolution,
        dtm: dtm_kind(dtm_method)?,
        first_returns,
        subcircle,
        min_returns,
        noise_a,
        noise_b,
        interpolation_error,
        confidence,
        harmonise: harmonise_cell.map(|cell| Harmonise { cell, seed }),
        horizontal_sigma,
        vertical_sigma,
    };
    let s = py.detach(|| surface::surface_change(&a, &b, al.as_ref(), &p, layout(chunk_size, None), buffer, workers)).map_err(err)?;
    let d = PyDict::new(py);
    for (k, r) in [("a", &s.a), ("b", &s.b), ("difference", &s.difference), ("bias", &s.bias), ("lower", &s.lower), ("upper", &s.upper), ("lod", &s.lod), ("sigma", &s.sigma), ("sigma_a", &s.sigma_a), ("sigma_b", &s.sigma_b), ("pulses_a", &s.pulses_a), ("pulses_b", &s.pulses_b), ("density_a", &s.density_a), ("density_b", &s.density_b)] {
        d.set_item(k, raster_to_py(py, r)?)?;
    }
    d.set_item("classes", PyArray1::from_vec(py, s.classes.clone()).reshape([s.a.nrows, s.a.ncols])?)?;
    d.set_item("volume_gained", s.volume_gained)?;
    d.set_item("volume_lost", s.volume_lost)?;
    d.set_item("area_changed", s.area_changed)?;
    d.set_item("area_compared", s.area_compared)?;
    d.set_item("raw_a", stats_to_py(py, &s.raw_a)?)?;
    d.set_item("raw_b", stats_to_py(py, &s.raw_b)?)?;
    d.set_item("stats_a", stats_to_py(py, &s.stats_a)?)?;
    d.set_item("stats_b", stats_to_py(py, &s.stats_b)?)?;
    d.set_item("median_bias", s.median_bias)?;
    d.set_item("notes", s.notes.clone())?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (catalog, other, out_dir, cell, seed, format, chunk_size, workers))]
fn change_als_harmonise(py: Python<'_>, catalog: &Bound<'_, PyDict>, other: &Bound<'_, PyDict>, out_dir: PathBuf, cell: f64, seed: u64, format: Option<String>, chunk_size: Option<f64>, workers: usize) -> PyResult<Vec<String>> {
    let (a, b) = (catalog_from_py(catalog)?, catalog_from_py(other)?);
    let h = Harmonise { cell, seed };
    let paths = py.detach(|| surface::harmonise_catalog(&a, &b, &out_dir, &h, format.as_deref(), layout(chunk_size, None), workers)).map_err(err)?;
    Ok(paths.into_iter().map(|p| p.to_string_lossy().to_string()).collect())
}

fn polygons_to_py<'py>(py: Python<'py>, polys: &[Polygon]) -> PyResult<Bound<'py, PyList>> {
    let ring = |r: &[[f64; 2]]| -> PyResult<Bound<'py, PyAny>> { Ok(PyArray1::from_vec(py, r.iter().flatten().copied().collect()).reshape([r.len(), 2])?.into_any()) };
    let out = PyList::empty(py);
    for p in polys {
        let holes = PyList::empty(py);
        for h in &p.holes {
            holes.append(ring(h)?)?;
        }
        out.append((ring(&p.exterior)?, holes))?;
    }
    Ok(out)
}

fn gaps_to_py<'py>(py: Python<'py>, g: &Gaps, nrows: usize, ncols: usize) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("labels", PyArray1::from_vec(py, g.labels.clone()).reshape([nrows, ncols])?)?;
    d.set_item("id", g.gaps.iter().map(|x| x.id as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("n_cells", g.gaps.iter().map(|x| x.n_cells as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("area", g.gaps.iter().map(|x| x.area).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("x", g.gaps.iter().map(|x| x.x).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("y", g.gaps.iter().map(|x| x.y).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("mean_height", g.gaps.iter().map(|x| x.mean_height).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("max_height", g.gaps.iter().map(|x| x.max_height).collect::<Vec<_>>().into_pyarray(py))?;
    let polys = PyList::empty(py);
    for x in &g.gaps {
        polys.append(polygons_to_py(py, &x.polygons)?)?;
    }
    d.set_item("polygons", polys)?;
    d.set_item("area_with_data", g.area_with_data)?;
    Ok(d)
}

#[pyfunction]
fn change_als_gaps<'py>(py: Python<'py>, chm: RasterArg, height: f64, min_area: f64, max_area: f64, connectivity: u8) -> PyResult<Bound<'py, PyDict>> {
    let r = raster(chm);
    let p = GapParams { height, min_area, max_area, connectivity };
    let g = py.detach(|| gaps::find_gaps(&r, &p)).map_err(err)?;
    gaps_to_py(py, &g, r.nrows, r.ncols)
}

#[pyfunction]
#[pyo3(signature = (chm_a, chm_b, classes, height, min_area, max_area, connectivity))]
fn change_als_gap_change<'py>(py: Python<'py>, chm_a: RasterArg, chm_b: RasterArg, classes: Option<PyReadonlyArray2<u8>>, height: f64, min_area: f64, max_area: f64, connectivity: u8) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (raster(chm_a), raster(chm_b));
    let cls: Option<Vec<u8>> = classes.map(|c| c.as_array().iter().copied().collect());
    let p = GapParams { height, min_area, max_area, connectivity };
    let c = py.detach(|| gaps::gap_change(&a, &b, cls.as_deref(), &p)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("a", gaps_to_py(py, &c.a, a.nrows, a.ncols)?)?;
    d.set_item("b", gaps_to_py(py, &c.b, a.nrows, a.ncols)?)?;
    d.set_item("cells", PyArray1::from_vec(py, c.cells.clone()).reshape([a.nrows, a.ncols])?)?;
    d.set_item("status_a", c.status_a.clone())?;
    d.set_item("closed_area", c.closed_area.clone().into_pyarray(py))?;
    d.set_item("status_b", c.status_b.clone())?;
    d.set_item("formed_area", c.formed_area.clone().into_pyarray(py))?;
    d.set_item("areas", c.areas.to_vec())?;
    d.set_item("cell_names", gaps::GAP_CELLS.to_vec())?;
    Ok(d)
}

#[pyfunction]
fn change_als_size_exponent(sizes: PyReadonlyArray1<f64>, xmin: f64) -> (f64, f64, usize) {
    gaps::size_exponent(sizes.as_array().as_slice().unwrap_or(&sizes.as_array().to_vec()), xmin)
}

fn trees_from_py(d: &Bound<'_, PyDict>) -> PyResult<Vec<AlsTree>> {
    let get = |k: &str| d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("trees are missing {k:?}")));
    let f = |k: &str| -> PyResult<Vec<f64>> { Ok(get(k)?.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec()) };
    let id: Vec<i64> = get("id")?.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec();
    let (x, y, h, area) = (f("x")?, f("y")?, f("height")?, f("crown_area")?);
    let crowns: Vec<PyReadonlyArray2<f64>> = get("crowns")?.extract()?;
    let n = id.len();
    if [x.len(), y.len(), h.len(), area.len(), crowns.len()].iter().any(|&k| k != n) {
        return Err(PyValueError::new_err("tree columns differ in length"));
    }
    (0..n)
        .map(|i| {
            let c = crowns[i].as_array();
            let crown = if c.ncols() == 2 { c.rows().into_iter().map(|r| [r[0], r[1]]).collect() } else { Vec::new() };
            Ok(AlsTree { id: id[i], x: x[i], y: y[i], height: h[i], crown_area: area[i], crown })
        })
        .collect()
}

fn rows_to_py<'py>(py: Python<'py>, rows: &[TreeChangeRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id_a", rows.iter().map(|r| r.id_a).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("id_b", rows.iter().map(|r| r.id_b).collect::<Vec<_>>().into_pyarray(py))?;
    let f = |g: &dyn Fn(&TreeChangeRow) -> f64| rows.iter().map(g).collect::<Vec<f64>>();
    for (k, v) in [
        ("x", f(&|r| r.x)),
        ("y", f(&|r| r.y)),
        ("distance", f(&|r| r.distance)),
        ("height_a", f(&|r| r.height_a)),
        ("height_b", f(&|r| r.height_b)),
        ("dh", f(&|r| r.dh)),
        ("sigma", f(&|r| r.sigma)),
        ("lod", f(&|r| r.lod)),
        ("crown_area_a", f(&|r| r.crown_area_a)),
        ("crown_area_b", f(&|r| r.crown_area_b)),
        ("crown_loss", f(&|r| r.crown_loss)),
        ("crown_gain", f(&|r| r.crown_gain)),
        ("observed", f(&|r| r.observed)),
    ] {
        d.set_item(k, v.into_pyarray(py))?;
    }
    d.set_item("dh_change", rows.iter().map(|r| r.dh_change).collect::<Vec<_>>())?;
    d.set_item("status", rows.iter().map(|r| r.status).collect::<Vec<_>>())?;
    Ok(d)
}

fn static_str(s: &str) -> &'static str {
    const ALL: [&str; 13] = ["survivor", "damaged", "dead", "undetected", "unobserved", "recruit", "released", "growth", "decrease", "below_detection", "unmeasured", "", "?"];
    ALL.iter().find(|&&a| a == s).copied().unwrap_or("?")
}

fn rows_from_py(d: &Bound<'_, PyDict>) -> PyResult<Vec<TreeChangeRow>> {
    let get = |k: &str| d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("tree change is missing {k:?}")));
    let f = |k: &str| -> PyResult<Vec<f64>> { Ok(get(k)?.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec()) };
    let i = |k: &str| -> PyResult<Vec<i64>> { Ok(get(k)?.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec()) };
    let (id_a, id_b) = (i("id_a")?, i("id_b")?);
    let (x, y, dist, ha, hb, dh, s, lod, caa, cab, cl, cg, obs) = (f("x")?, f("y")?, f("distance")?, f("height_a")?, f("height_b")?, f("dh")?, f("sigma")?, f("lod")?, f("crown_area_a")?, f("crown_area_b")?, f("crown_loss")?, f("crown_gain")?, f("observed")?);
    let status: Vec<String> = get("status")?.extract()?;
    let dhc: Vec<String> = get("dh_change")?.extract()?;
    let n = id_a.len();
    if [id_b.len(), x.len(), status.len(), dhc.len(), dh.len()].iter().any(|&k| k != n) {
        return Err(PyValueError::new_err("tree change columns differ in length"));
    }
    Ok((0..n)
        .map(|k| TreeChangeRow { id_a: id_a[k], id_b: id_b[k], x: x[k], y: y[k], distance: dist[k], height_a: ha[k], height_b: hb[k], dh: dh[k], sigma: s[k], lod: lod[k], dh_change: static_str(&dhc[k]), status: static_str(&status[k]), crown_area_a: caa[k], crown_area_b: cab[k], crown_loss: cl[k], crown_gain: cg[k], observed: obs[k] })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (trees_a, trees_b, chm_a, chm_b, sigma_a, sigma_b, classes, alignment, max_distance, max_growth, max_drop, height_weight, confidence, dead_fraction, damage_fraction, min_observed))]
fn change_als_trees<'py>(py: Python<'py>, trees_a: &Bound<'_, PyDict>, trees_b: &Bound<'_, PyDict>, chm_a: RasterArg, chm_b: RasterArg, sigma_a: RasterArg, sigma_b: RasterArg, classes: PyReadonlyArray2<u8>, alignment: Option<&Bound<'_, PyDict>>, max_distance: f64, max_growth: f64, max_drop: f64, height_weight: f64, confidence: f64, dead_fraction: f64, damage_fraction: f64, min_observed: f64) -> PyResult<Bound<'py, PyDict>> {
    let (ta, tb) = (trees_from_py(trees_a)?, trees_from_py(trees_b)?);
    let (ca, cb, sa, sb) = (raster(chm_a), raster(chm_b), raster(sigma_a), raster(sigma_b));
    let cls: Vec<u8> = classes.as_array().iter().copied().collect();
    let al = alignment.map(alignment_from_py).transpose()?;
    let p = TreeChangeParams { max_distance, max_growth, max_drop, height_weight, confidence, dead_fraction, damage_fraction, min_observed };
    let canopy = CanopyChange { chm_a: &ca, chm_b: &cb, sigma_a: &sa, sigma_b: &sb, classes: &cls };
    let rows = py.detach(|| trees::tree_change(&ta, &tb, &canopy, al.as_ref(), &p)).map_err(err)?;
    rows_to_py(py, &rows)
}

#[pyfunction]
#[pyo3(signature = (rows, mask, years))]
fn change_als_tree_summary<'py>(py: Python<'py>, rows: &Bound<'_, PyDict>, mask: Option<PyReadonlyArray1<bool>>, years: Option<f64>) -> PyResult<Bound<'py, PyDict>> {
    let r = rows_from_py(rows)?;
    let m: Option<Vec<bool>> = mask.map(|m| m.as_array().to_vec());
    if m.as_ref().is_some_and(|m| m.len() != r.len()) {
        return Err(PyValueError::new_err("the mask must have one value per row"));
    }
    let sel: Vec<&TreeChangeRow> = r.iter().enumerate().filter(|(k, _)| m.as_ref().is_none_or(|m| m[*k])).map(|(_, x)| x).collect();
    let s = trees::summarise(&sel, years);
    let d = PyDict::new(py);
    for (k, v) in [("survivors", s.survivors), ("damaged", s.damaged), ("dead", s.dead), ("recruits", s.recruits), ("released", s.released), ("undetected_a", s.undetected_a), ("undetected_b", s.undetected_b), ("unobserved_a", s.unobserved_a), ("unobserved_b", s.unobserved_b), ("n_growth", s.n_growth), ("n_growth_detected", s.n_growth_detected)] {
        d.set_item(k, v)?;
    }
    for (k, v) in [("mean_growth", s.mean_growth), ("growth_se", s.growth_se), ("growth_measurement_se", s.growth_measurement_se), ("crown_area_dead", s.crown_area_dead), ("crown_area_damaged", s.crown_area_damaged), ("mortality_rate", s.mortality_rate), ("recruitment_rate", s.recruitment_rate)] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

#[pyfunction]
fn change_als_tree_grid<'py>(py: Python<'py>, rows: &Bound<'_, PyDict>, xmin: f64, ymin: f64, resolution: f64, nrows: usize, ncols: usize) -> PyResult<Bound<'py, PyDict>> {
    let r = rows_from_py(rows)?;
    let layers = trees::grid_summary(&r, (xmin, ymin, resolution, nrows, ncols)).map_err(err)?;
    let d = PyDict::new(py);
    for (name, l) in trees::GRID_LAYERS.iter().zip(&layers) {
        d.set_item(*name, raster_to_py(py, l)?)?;
    }
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (catalog_a, catalog_b, alignment, resolution, names, threshold, entropy_bin, cover_break, min_height, drop_noise, dtm_resolution, dtm_method, permutations, stratum, seed, confidence, harmonise_cell, harmonise_seed, chunk_size, buffer, workers))]
fn change_als_metrics<'py>(py: Python<'py>, catalog_a: &Bound<'_, PyDict>, catalog_b: &Bound<'_, PyDict>, alignment: Option<&Bound<'_, PyDict>>, resolution: f64, names: Option<Vec<String>>, threshold: f64, entropy_bin: f64, cover_break: f64, min_height: Option<f64>, drop_noise: bool, dtm_resolution: f64, dtm_method: &str, permutations: usize, stratum: f64, seed: u64, confidence: f64, harmonise_cell: Option<f64>, harmonise_seed: u64, chunk_size: Option<f64>, buffer: f64, workers: usize) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (catalog_from_py(catalog_a)?, catalog_from_py(catalog_b)?);
    let al = alignment.map(alignment_from_py).transpose()?;
    let p = MetricChangeParams {
        metrics: MetricParams { threshold, entropy_bin, cover_break, min_height, drop_noise, ..Default::default() },
        names,
        resolution,
        dtm_resolution,
        dtm: dtm_kind(dtm_method)?,
        permutations,
        stratum,
        seed,
        confidence,
        harmonise: harmonise_cell.map(|cell| Harmonise { cell, seed: harmonise_seed }),
    };
    let m = py.detach(|| metrics::metric_change(&a, &b, al.as_ref(), &p, layout(chunk_size, None), buffer, workers)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("names", m.names.clone())?;
    for (k, v) in [("a", &m.a), ("b", &m.b), ("difference", &m.difference), ("bias", &m.bias), ("lower", &m.lower), ("upper", &m.upper), ("lod", &m.lod), ("sigma", &m.sigma)] {
        let l = PyList::empty(py);
        for r in v {
            l.append(raster_to_py(py, r)?)?;
        }
        d.set_item(k, l)?;
    }
    let l = PyList::empty(py);
    let (nr, nc) = (m.a.first().map_or(0, |r| r.nrows), m.a.first().map_or(0, |r| r.ncols));
    for c in &m.classes {
        l.append(PyArray1::from_vec(py, c.clone()).reshape([nr, nc])?)?;
    }
    d.set_item("classes", l)?;
    Ok(d)
}

fn profile_from_py(d: &Bound<'_, PyDict>) -> PyResult<ProfileGrid> {
    let get = |k: &str| d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("profile is missing {k:?}")));
    let w: PyReadonlyArray3<f64> = get("weight")?.extract()?;
    let wk: PyReadonlyArray3<f64> = get("weight_k")?.extract()?;
    let s = w.shape().to_vec();
    if s.len() != 3 || wk.shape() != s.as_slice() || s[0] < 3 {
        return Err(PyValueError::new_err("weight and weight_k must be (nz + 2, ny, nx) arrays of one shape"));
    }
    Ok(ProfileGrid {
        xmin: get("xmin")?.extract()?,
        ymin: get("ymin")?.extract()?,
        resolution: get("resolution")?.extract()?,
        nx: s[2],
        ny: s[1],
        nz: s[0] - 2,
        min_height: get("min_height")?.extract()?,
        bin_size: get("bin_size")?.extract()?,
        weight: w.as_array().iter().copied().collect(),
        weight_k: wk.as_array().iter().copied().collect(),
        n_skipped: 0,
    })
}

#[pyfunction]
fn change_als_pai<'py>(py: Python<'py>, profile_a: &Bound<'_, PyDict>, profile_b: &Bound<'_, PyDict>, confidence: f64) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (profile_from_py(profile_a)?, profile_from_py(profile_b)?);
    let c = py.detach(|| metrics::pai_change(&a, &b, confidence)).map_err(err)?;
    let d = PyDict::new(py);
    for (k, r) in [("pai_a", &c.pai_a), ("pai_b", &c.pai_b), ("sigma_a", &c.sigma_a), ("sigma_b", &c.sigma_b), ("difference", &c.difference), ("sigma", &c.sigma), ("lod", &c.lod)] {
        d.set_item(k, raster_to_py(py, r)?)?;
    }
    d.set_item("classes", PyArray1::from_vec(py, c.classes.clone()).reshape([c.pai_a.nrows, c.pai_a.ncols])?)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (profile_a, profile_b, mask=None))]
fn change_als_profile<'py>(py: Python<'py>, profile_a: &Bound<'_, PyDict>, profile_b: &Bound<'_, PyDict>, mask: Option<PyReadonlyArray2<bool>>) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (profile_from_py(profile_a)?, profile_from_py(profile_b)?);
    let m: Option<Vec<bool>> = mask.map(|m| m.as_array().iter().copied().collect());
    let (h, pa, pb, diff, s) = metrics::profile_change(&a, &b, m.as_deref()).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("height", h.into_pyarray(py))?;
    d.set_item("pad_a", pa.into_pyarray(py))?;
    d.set_item("pad_b", pb.into_pyarray(py))?;
    d.set_item("difference", diff.into_pyarray(py))?;
    d.set_item("sigma", s.into_pyarray(py))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(change_als_align, m)?,
        wrap_pyfunction!(change_als_offsets, m)?,
        wrap_pyfunction!(change_als_surface, m)?,
        wrap_pyfunction!(change_als_harmonise, m)?,
        wrap_pyfunction!(change_als_gaps, m)?,
        wrap_pyfunction!(change_als_gap_change, m)?,
        wrap_pyfunction!(change_als_size_exponent, m)?,
        wrap_pyfunction!(change_als_trees, m)?,
        wrap_pyfunction!(change_als_tree_summary, m)?,
        wrap_pyfunction!(change_als_tree_grid, m)?,
        wrap_pyfunction!(change_als_metrics, m)?,
        wrap_pyfunction!(change_als_pai, m)?,
        wrap_pyfunction!(change_als_profile, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("CHANGE_ALS_CLASSES", surface::CLASSES.to_vec())?;
    m.add("CHANGE_ALS_PAI_CLASSES", metrics::PAI_CLASSES.to_vec())?;
    Ok(())
}
