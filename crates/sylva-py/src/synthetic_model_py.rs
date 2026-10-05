// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the realistic synthetic trees, plots and beam scanner.
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sylva_rs::qsm::Qsm;
use sylva_rs::synthetic::plot::{self as plot_rs, DbhDistribution, PlotSpec, Terrain};
use sylva_rs::synthetic::scan::{self, BeamScan, Reflectance};
use sylva_rs::synthetic::tree::{self, Archetype, Leaf, TreeSpec};
use sylva_rs::voxel::Lad;
use sylva_rs::Point;

use crate::{cloud_from_py, cloud_to_py, err, shots_to_py};

fn rows<'py, const N: usize>(py: Python<'py>, r: &[[f64; N]]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = r.iter().flat_map(|v| v.iter().copied()).collect();
    PyArray1::from_vec(py, flat).reshape([r.len(), N]).expect("reshape")
}

fn points<'py>(py: Python<'py>, p: impl Iterator<Item = Point>) -> Bound<'py, PyArray2<f64>> {
    rows(py, &p.collect::<Vec<_>>())
}

fn lad_from(name: Option<String>, params: Vec<f64>) -> PyResult<Option<Lad>> {
    match name {
        None => Ok(None),
        Some(n) => {
            let n = if n == "beta" { "twoParamBeta".to_string() } else { n };
            Lad::parse(&n, &params).map(Some).map_err(err)
        }
    }
}

fn leaves_to_py<'py>(py: Python<'py>, leaves: &[Leaf], d: &Bound<'py, PyDict>) -> PyResult<()> {
    d.set_item("leaf_centre", points(py, leaves.iter().map(|l| l.centre)))?;
    d.set_item("leaf_normal", points(py, leaves.iter().map(|l| l.normal)))?;
    d.set_item("leaf_axis", points(py, leaves.iter().map(|l| l.axis)))?;
    d.set_item("leaf_length", leaves.iter().map(|l| l.length).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("leaf_width", leaves.iter().map(|l| l.width).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("leaf_area", leaves.iter().map(|l| l.area()).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("leaf_cylinder", leaves.iter().map(|l| l.cylinder as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("leaf_epicormic", leaves.iter().map(|l| l.epicormic).collect::<Vec<_>>().into_pyarray(py))?;
    Ok(())
}

#[pyfunction]
fn synthetic_tree_model<'py>(py: Python<'py>, archetype: &str, x: f64, y: f64, z0: f64, dbh: f64, height: f64, crown_radius: Option<f64>, crown_base: Option<f64>, max_order: u32, lean_deg: Option<f64>, sweep: Option<f64>, butt_swell: Option<f64>, buttresses: usize, buttress_height: f64, buttress_extent: f64, ellipticity: f64, bark_depth: f64, leaf_area: Option<f64>, leaf_size: Option<(f64, f64)>, lad: Option<String>, lad_params: Vec<f64>, epicormic: f64, point_density: f64, pipe_exponent: f64, min_radius: f64, breast_height: f64, seed: u64) -> PyResult<Bound<'py, PyDict>> {
    let mut s = TreeSpec::new(Archetype::parse(archetype).map_err(err)?, [x, y, z0], dbh, height, seed);
    s.crown_radius = crown_radius;
    s.crown_base = crown_base;
    s.max_order = max_order;
    s.lean_deg = lean_deg;
    s.sweep = sweep;
    s.butt_swell = butt_swell;
    s.buttresses = buttresses;
    s.buttress_height = buttress_height;
    s.buttress_extent = buttress_extent;
    s.ellipticity = ellipticity;
    s.bark_depth = bark_depth;
    s.leaf_area = leaf_area;
    s.leaf_size = leaf_size;
    s.lad = lad_from(lad, lad_params)?;
    s.epicormic = epicormic;
    s.point_density = point_density;
    s.pipe_exponent = pipe_exponent;
    s.min_radius = min_radius;
    s.breast_height = breast_height;
    let t = py.detach(|| tree::tree_model(&s)).map_err(err)?;
    let d = PyDict::new(py);
    let (xyz, attrs) = cloud_to_py(py, &t.cloud)?;
    d.set_item("xyz", xyz)?;
    d.set_item("attrs", attrs)?;
    d.set_item("cylinders", rows(py, &t.qsm.to_rows()))?;
    d.set_item("cylinder_epicormic", t.cylinder_epicormic.clone().into_pyarray(py))?;
    leaves_to_py(py, &t.leaves, &d)?;
    d.set_item("archetype", t.archetype.name())?;
    d.set_item("base", t.base.to_vec())?;
    d.set_item("stem_bh", t.stem_bh.to_vec())?;
    d.set_item("dbh", t.dbh)?;
    d.set_item("height", t.height)?;
    d.set_item("crown_base", t.crown_base)?;
    d.set_item("crown_area", t.crown_area)?;
    d.set_item("crown_extent", t.crown_extent.to_vec())?;
    d.set_item("total_leaf_area", t.leaf_area)?;
    d.set_item("epicormic_leaf_area", t.epicormic_leaf_area)?;
    d.set_item("lad", t.lad.name())?;
    Ok(d)
}

/// The architectural constants of an archetype that users may want to see.
#[pyfunction]
fn synthetic_archetype<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    let f = Archetype::parse(name).map_err(err)?.form();
    let d = PyDict::new(py);
    d.set_item("crown_base", f.crown_base)?;
    d.set_item("crown_radius", f.crown_radius)?;
    d.set_item("leader", f.leader)?;
    d.set_item("lai", f.lai)?;
    d.set_item("leaf_k", f.leaf_k)?;
    d.set_item("crown_k", f.crown_k)?;
    d.set_item("allometry", f.allometry)?;
    d.set_item("leaf_size", f.leaf)?;
    d.set_item("lad", f.lad)?;
    d.set_item("butt_swell", f.butt_swell)?;
    d.set_item("lean", f.lean)?;
    d.set_item("sweep", f.sweep)?;
    Ok(d)
}

#[pyfunction]
fn synthetic_plot<'py>(py: Python<'py>, size: f64, density: f64, dbh_kind: &str, dbh_params: Vec<f64>, min_dbh: f64, archetypes: Vec<(String, f64)>, height_noise: f64, slope: f64, aspect_deg: f64, roughness: f64, roughness_length: f64, shrubs: f64, grass_cover: f64, grass_height: f64, logs: f64, stumps: f64, ground_density: f64, margin: f64, point_density: f64, max_order: u32, lad: Option<String>, lad_params: Vec<f64>, epicormic: f64, seed: u64) -> PyResult<Bound<'py, PyDict>> {
    let mut s = PlotSpec::new(seed);
    let need = |n: usize| if dbh_params.len() == n { Ok(()) } else { Err(PyValueError::new_err(format!("the {dbh_kind} diameter distribution takes {n} parameter(s), got {}", dbh_params.len()))) };
    s.dbh = match dbh_kind {
        "weibull" => {
            need(2)?;
            DbhDistribution::Weibull { shape: dbh_params[0], scale: dbh_params[1] }
        }
        "reverse_j" => {
            need(1)?;
            DbhDistribution::ReverseJ { mean: dbh_params[0] }
        }
        "given" => DbhDistribution::Given(dbh_params),
        other => return Err(PyValueError::new_err(format!("unknown diameter distribution {other:?}; expected 'weibull', 'reverse_j' or a list of diameters"))),
    };
    s.size = size;
    s.density = density;
    s.min_dbh = min_dbh;
    s.archetypes = archetypes.iter().map(|(n, w)| Archetype::parse(n).map(|a| (a, *w))).collect::<sylva_rs::error::Result<_>>().map_err(err)?;
    s.height_noise = height_noise;
    s.slope = slope;
    s.aspect_deg = aspect_deg;
    s.roughness = roughness;
    s.roughness_length = roughness_length;
    s.shrubs = shrubs;
    s.grass_cover = grass_cover;
    s.grass_height = grass_height;
    s.logs = logs;
    s.stumps = stumps;
    s.ground_density = ground_density;
    s.margin = margin;
    s.point_density = point_density;
    s.max_order = max_order;
    s.lad = lad_from(lad, lad_params)?;
    s.epicormic = epicormic;
    let p = py.detach(|| plot_rs::plot(&s)).map_err(err)?;
    let d = PyDict::new(py);
    let (xyz, attrs) = cloud_to_py(py, &p.cloud)?;
    d.set_item("xyz", xyz)?;
    d.set_item("attrs", attrs)?;
    let t = PyDict::new(py);
    let col = |f: &dyn Fn(&plot_rs::PlotTree) -> f64| p.trees.iter().map(f).collect::<Vec<f64>>();
    t.set_item("tree_id", p.trees.iter().map(|t| t.id as i64).collect::<Vec<_>>().into_pyarray(py))?;
    t.set_item("archetype", p.trees.iter().map(|t| t.archetype.name()).collect::<Vec<_>>())?;
    for (name, v) in [
        ("x", col(&|t| t.base[0])), ("y", col(&|t| t.base[1])), ("z", col(&|t| t.base[2])),
        ("x_bh", col(&|t| t.stem_bh[0])), ("y_bh", col(&|t| t.stem_bh[1])),
        ("dbh", col(&|t| t.dbh)), ("height", col(&|t| t.height)), ("crown_base", col(&|t| t.crown_base)), ("crown_area", col(&|t| t.crown_area)),
        ("wood_volume", col(&|t| t.wood_volume)), ("stem_volume", col(&|t| t.stem_volume)), ("branch_volume", col(&|t| t.branch_volume)), ("leaf_area", col(&|t| t.leaf_area)),
    ] {
        t.set_item(name, v.into_pyarray(py))?;
    }
    d.set_item("trees", t)?;
    d.set_item("cylinders", rows(py, &Qsm { cylinders: p.cylinders.clone() }.to_rows()))?;
    d.set_item("cylinder_tree", p.cylinder_tree.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    leaves_to_py(py, &p.leaves, &d)?;
    d.set_item("leaf_tree", p.leaf_tree.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    let dw = PyDict::new(py);
    dw.set_item("kind", p.dead_wood.iter().map(|w| if w.kind == 0 { "log" } else { "stump" }).collect::<Vec<_>>())?;
    dw.set_item("start", points(py, p.dead_wood.iter().map(|w| w.start)))?;
    dw.set_item("axis", points(py, p.dead_wood.iter().map(|w| w.axis)))?;
    dw.set_item("length", p.dead_wood.iter().map(|w| w.length).collect::<Vec<_>>().into_pyarray(py))?;
    dw.set_item("radius", p.dead_wood.iter().map(|w| w.radius).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("dead_wood", dw)?;
    d.set_item("shrubs", rows(py, &p.shrubs))?;
    d.set_item("grass", rows(py, &p.grass))?;
    d.set_item("terrain_modes", rows(py, &p.terrain.modes))?;
    d.set_item("terrain_slope", p.terrain.slope)?;
    d.set_item("terrain_aspect", p.terrain.aspect_deg)?;
    Ok(d)
}

/// Height of a plot's terrain at each `(x, y)`.
#[pyfunction]
fn synthetic_plot_terrain<'py>(py: Python<'py>, modes: PyReadonlyArray2<f64>, slope: f64, aspect_deg: f64, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = modes.as_array();
    if m.ncols() != 4 && m.nrows() > 0 {
        return Err(PyValueError::new_err("terrain modes must have shape (n, 4)"));
    }
    let t = Terrain { slope, aspect_deg, modes: m.rows().into_iter().map(|r| [r[0], r[1], r[2], r[3]]).collect() };
    let (x, y) = (x.as_array(), y.as_array());
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y differ in length"));
    }
    Ok(x.iter().zip(y.iter()).map(|(&a, &b)| t.height(a, b)).collect::<Vec<_>>().into_pyarray(py))
}

#[pyfunction]
fn synthetic_scan_beam<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, origin: (f64, f64, f64), resolution_deg: f64, min_zenith_deg: f64, max_zenith_deg: f64, max_echoes: usize, echo_separation: f64, range_noise: f64, range_noise_slope: f64, beam_divergence: f64, exit_diameter: f64, footprint_samples: usize, mixed_pixels: bool, target_radius: Option<f64>, oriented: bool, detection_threshold: f64, reflectance_map: Vec<(i64, f64)>, reflectance_default: f64, reflectance_attr: Option<String>, reflectance_constant: Option<f64>, max_range: f64, seed: u64) -> PyResult<Bound<'py, PyDict>> {
    let cloud = cloud_from_py(xyz, attrs)?;
    let mut p = BeamScan::new([origin.0, origin.1, origin.2]);
    p.resolution_deg = resolution_deg;
    p.min_zenith_deg = min_zenith_deg;
    p.max_zenith_deg = max_zenith_deg;
    p.max_echoes = max_echoes;
    p.echo_separation = echo_separation;
    p.range_noise = range_noise;
    p.range_noise_slope = range_noise_slope;
    p.divergence_mrad = beam_divergence;
    p.exit_diameter = exit_diameter;
    p.footprint_samples = footprint_samples;
    p.mixed_pixels = mixed_pixels;
    p.target_radius = target_radius;
    p.oriented = oriented;
    p.detection_threshold = detection_threshold;
    p.reflectance = match (reflectance_attr, reflectance_constant) {
        (Some(a), _) => Reflectance::Attribute(a),
        (None, Some(c)) => Reflectance::Constant(c),
        (None, None) => Reflectance::ByClass(reflectance_map, reflectance_default),
    };
    p.max_range = max_range;
    p.seed = seed;
    let s = py.detach(|| scan::scan_beam(&cloud, &p)).map_err(err)?;
    shots_to_py(py, &s)
}

#[pyfunction]
fn synthetic_scanner_preset<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    let p = scan::scanner_preset(name).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("min_zenith_deg", p.min_zenith_deg)?;
    d.set_item("max_zenith_deg", p.max_zenith_deg)?;
    d.set_item("beam_divergence", p.divergence_mrad)?;
    d.set_item("exit_diameter", p.exit_diameter)?;
    d.set_item("range_noise", p.range_noise)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(synthetic_tree_model, m)?,
        wrap_pyfunction!(synthetic_archetype, m)?,
        wrap_pyfunction!(synthetic_plot, m)?,
        wrap_pyfunction!(synthetic_plot_terrain, m)?,
        wrap_pyfunction!(synthetic_scan_beam, m)?,
        wrap_pyfunction!(synthetic_scanner_preset, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
