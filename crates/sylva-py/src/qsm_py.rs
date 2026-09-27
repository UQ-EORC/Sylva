// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva.qsm: the cylinder table's totals and cuts, the CSV
//! reader, the mesh writers, buttress meshing and fusion, and the plot loop
//! with its table and files. Models cross as `(n, 12)` float arrays; a plot
//! crosses as a list of `(tree_id, cylinders, points, height, buttress)`
//! tuples, the buttress `(vertices, faces, volume, top, top_z)` or None.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::qsm::buttress::{Buttress, ButtressParams};
use sylva_rs::qsm::QsmParams;
use sylva_rs::qsm_ops::{self as ops, Row, Segment};
use sylva_rs::qsm_plot::{self as plot, ButtressView, PlotEntry, PlotParams};
use sylva_rs::{mesh_io, Point};

use crate::{err, xyz_from_py, xyz_to_py};

type Matrix<'py> = Bound<'py, PyArray2<f64>>;

fn rows_from_py(cylinders: PyReadonlyArray2<f64>) -> PyResult<Vec<Row>> {
    let a = cylinders.as_array();
    if a.ncols() != 12 {
        return Err(PyValueError::new_err("cylinder array must have 12 columns"));
    }
    Ok(a.rows().into_iter().map(|r| std::array::from_fn(|k| r[k])).collect())
}

fn rows_to_py<'py>(py: Python<'py>, rows: &[Row]) -> PyResult<Matrix<'py>> {
    PyArray1::from_vec(py, rows.iter().flatten().copied().collect::<Vec<f64>>()).reshape([rows.len(), 12])
}

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

fn segments_from_py(s: PyReadonlyArray3<f64>) -> PyResult<Vec<Segment>> {
    let a = s.as_array();
    if a.shape()[1] != 2 || a.shape()[2] != 2 {
        return Err(PyValueError::new_err("a section must have shape (n, 2, 2)"));
    }
    Ok((0..a.shape()[0]).map(|i| [[a[[i, 0, 0]], a[[i, 0, 1]]], [a[[i, 1, 0]], a[[i, 1, 1]]]]).collect())
}

fn segments_to_py<'py>(py: Python<'py>, s: &[Segment]) -> PyResult<Bound<'py, PyArray3<f64>>> {
    PyArray1::from_vec(py, s.iter().flatten().flatten().copied().collect::<Vec<f64>>()).reshape([s.len(), 2, 2])
}

fn io_err(path: &std::path::Path, e: sylva_rs::Error) -> PyErr {
    match e {
        sylva_rs::Error::Io(io) if io.kind() == std::io::ErrorKind::NotFound => PyFileNotFoundError::new_err(format!("{} not found.", path.display())),
        e => err(e),
    }
}

// ------------------------------------------------------------ the table

/// Volumes, lengths and the highest branch order of a model.
#[pyfunction]
fn qsm_totals<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyDict>> {
    let t = ops::totals(&rows_from_py(cylinders)?);
    let d = PyDict::new(py);
    d.set_item("total_volume", t.total_volume)?;
    d.set_item("stem_volume", t.stem_volume)?;
    d.set_item("branch_volume", t.branch_volume)?;
    d.set_item("total_length", t.total_length)?;
    d.set_item("max_branch_order", t.max_branch_order)?;
    Ok(d)
}

#[pyfunction]
fn qsm_ends<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Matrix<'py>> {
    Ok(xyz_to_py(py, &ops::ends(&rows_from_py(cylinders)?)))
}

#[pyfunction]
fn qsm_volumes<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(ops::volumes(&rows_from_py(cylinders)?).into_pyarray(py))
}

#[pyfunction]
fn qsm_volume_above(cylinders: PyReadonlyArray2<f64>, z: f64) -> PyResult<f64> {
    Ok(ops::volume_above(&rows_from_py(cylinders)?, z))
}

#[pyfunction]
fn qsm_above<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>, z: f64) -> PyResult<Matrix<'py>> {
    rows_to_py(py, &ops::above(&rows_from_py(cylinders)?, z))
}

#[pyfunction]
fn qsm_read_csv<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Matrix<'py>> {
    let rows = ops::read_csv(&path).map_err(|e| io_err(&path, e))?;
    rows_to_py(py, &rows)
}

// ------------------------------------------------------------ meshes

#[pyfunction]
#[pyo3(signature = (path, cylinders, sides=12, contiguous=false))]
fn qsm_write_obj(path: PathBuf, cylinders: PyReadonlyArray2<f64>, sides: usize, contiguous: bool) -> PyResult<()> {
    ops::write_model_obj(path, &rows_from_py(cylinders)?, sides, contiguous).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (path, cylinders, sides=12, color=None, contiguous=false))]
fn qsm_write_ply(path: PathBuf, cylinders: PyReadonlyArray2<f64>, sides: usize, color: Option<[u8; 3]>, contiguous: bool) -> PyResult<()> {
    ops::write_model_ply(path, &rows_from_py(cylinders)?, sides, color, contiguous).map_err(err)
}

/// A binary PLY triangle mesh; `faces` are int32 and `face_rgb` one colour per face.
#[pyfunction]
#[pyo3(signature = (path, vertices, faces, face_rgb=None))]
fn write_ply_mesh(path: PathBuf, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<i32>, face_rgb: Option<PyReadonlyArray2<u8>>) -> PyResult<()> {
    let v = xyz_from_py(vertices)?;
    let f = faces.as_array();
    if f.ncols() != 3 {
        return Err(PyValueError::new_err("faces must have three columns"));
    }
    let f: Vec<[i32; 3]> = f.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let rgb = match face_rgb {
        Some(c) => {
            let c = c.as_array();
            if c.ncols() != 3 {
                return Err(PyValueError::new_err("face_rgb must have three columns"));
            }
            Some(c.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect::<Vec<[u8; 3]>>())
        }
        None => None,
    };
    mesh_io::write_ply(path, &v, &f, rgb.as_deref()).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, cx, cy, ground_z=None, resolution=0.02, slice=0.05, close_radius=0.08, max_radius=4.0, max_height=6.0, top=None, solidity=0.9, round_run=4, min_top=0.5, min_points=30, max_flare=1.0, smooth=10))]
#[allow(clippy::too_many_arguments)]
fn qsm_buttress_mesh<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, cx: f64, cy: f64, ground_z: Option<f64>, resolution: f64, slice: f64, close_radius: f64, max_radius: f64, max_height: f64, top: Option<f64>, solidity: f64, round_run: usize, min_top: f64, min_points: usize, max_flare: f64, smooth: usize) -> PyResult<Bound<'py, PyDict>> {
    let pts = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let p = ButtressParams { resolution, slice, close_radius, max_radius, max_height, top, solidity, round_run, min_top, min_points, max_flare, smooth };
    let b = py.detach(|| plot::buttress_mesh(&pts, &h, cx, cy, ground_z, &p)).map_err(err)?;
    buttress_to_py(py, &b)
}

fn buttress_to_py<'py>(py: Python<'py>, b: &Buttress) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("vertices", xyz_to_py(py, &b.vertices))?;
    let f: Vec<i64> = b.faces.iter().flatten().map(|&x| x as i64).collect();
    d.set_item("faces", PyArray1::from_vec(py, f).reshape([b.faces.len(), 3])?)?;
    d.set_item("volume", b.volume)?;
    d.set_item("top", b.top)?;
    d.set_item("top_z", b.top_z)?;
    d.set_item("heights", b.heights.clone().into_pyarray(py))?;
    d.set_item("areas", b.areas.clone().into_pyarray(py))?;
    d.set_item("solidities", b.solidities.clone().into_pyarray(py))?;
    d.set_item("open", b.open.clone().into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
fn qsm_section<'py>(py: Python<'py>, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, z: f64) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let s = ops::section(&xyz_from_py(vertices)?, &faces_from_py(faces)?, z).map_err(err)?;
    segments_to_py(py, &s)
}

#[pyfunction]
fn qsm_inside<'py>(py: Python<'py>, section: PyReadonlyArray3<f64>, points: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let s = segments_from_py(section)?;
    let p = points.as_array();
    if p.ncols() < 2 {
        return Err(PyValueError::new_err("points must be (n, 2)"));
    }
    let p: Vec<[f64; 2]> = p.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    Ok(ops::inside(&s, &p).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (base, wood, cell=0.02))]
fn qsm_join_fit(base: PyReadonlyArray3<f64>, wood: PyReadonlyArray3<f64>, cell: f64) -> PyResult<(f64, f64)> {
    Ok(ops::join_fit(&segments_from_py(base)?, &segments_from_py(wood)?, cell))
}

#[pyfunction]
#[pyo3(signature = (vertices, faces, volume, top_z, cylinders, sides=12, contiguous=true, overlap=0.1))]
#[allow(clippy::too_many_arguments)]
fn qsm_fuse<'py>(py: Python<'py>, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, volume: f64, top_z: f64, cylinders: PyReadonlyArray2<f64>, sides: usize, contiguous: bool, overlap: f64) -> PyResult<Bound<'py, PyDict>> {
    let (v, f, rows) = (xyz_from_py(vertices)?, faces_from_py(faces)?, rows_from_py(cylinders)?);
    let t = py.detach(|| ops::fuse(&v, &f, volume, top_z, &rows, sides, contiguous, overlap)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("vertices", xyz_to_py(py, &t.vertices))?;
    d.set_item("faces", faces_to_py(py, &t.faces)?)?;
    d.set_item("part", t.part.into_pyarray(py))?;
    d.set_item("buttress_volume", t.buttress_volume)?;
    d.set_item("wood_volume", t.wood_volume)?;
    d.set_item("top_z", t.top_z)?;
    d.set_item("offset", t.offset)?;
    d.set_item("overhang", t.overhang)?;
    Ok(d)
}

#[pyfunction]
fn tree_mesh_write_obj(path: PathBuf, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, part: PyReadonlyArray1<u8>) -> PyResult<()> {
    ops::write_tree_mesh_obj(path, &xyz_from_py(vertices)?, &faces_from_py(faces)?, &part.as_array().to_vec()).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (path, vertices, faces, part, color=None))]
fn tree_mesh_write_ply(path: PathBuf, vertices: PyReadonlyArray2<f64>, faces: PyReadonlyArray2<u32>, part: PyReadonlyArray1<u8>, color: Option<[u8; 3]>) -> PyResult<()> {
    ops::write_tree_mesh_ply(path, &xyz_from_py(vertices)?, &faces_from_py(faces)?, &part.as_array().to_vec(), color).map_err(err)
}

// ------------------------------------------------------------ plots

/// QSM settings from a dict holding every field of `QsmParams`.
fn qsm_params_from_py(d: &Bound<'_, PyDict>) -> PyResult<QsmParams> {
    let mut p = QsmParams::default();
    macro_rules! take {
        ($($f:ident),*) => {
            $(p.$f = d.get_item(stringify!($f))?.ok_or_else(|| PyValueError::new_err(concat!("missing QSM setting ", stringify!($f))))?.extract()?;)*
        };
    }
    take!(k, max_edge, bin_length, min_points, ransac_threshold, max_radius, taper_limit, max_rmse, smooth_steps, apex_radius, min_arc_deg, min_inlier_fraction, prune_points, fit_min_points, crop_length, butt_height, relative_tolerance, base_radius, allometry_tolerance, buttress_equivalent_area, buttress_max_inlier_fraction, pipe_slack, branch_min_inlier_fraction, spacing_scale, radius_power, power_above_spacing, sensor_noise, cluster_eps, centre_fit_points, radius_smooth_steps, butt_swell, butt_vertical_run, butt_max_lean_deg, chain_max_d, fourier_min_radius);
    Ok(p)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn qsm_build_plot<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, labels: PyReadonlyArray1<i64>, heights: Option<PyReadonlyArray1<f64>>, stem_ids: Vec<i64>, stem_xy: Vec<(f64, f64)>, voxel_size: f64, wood: bool, buttress: bool, min_points: f64, params: &Bound<'_, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let pts = xyz_from_py(xyz)?;
    let labels = labels.as_array().to_vec();
    let h = heights.map(|h| h.as_array().to_vec());
    if stem_ids.len() != stem_xy.len() {
        return Err(PyValueError::new_err("one stem position per stem id"));
    }
    let stems: Vec<(i64, [f64; 2])> = stem_ids.into_iter().zip(stem_xy).map(|(t, (x, y))| (t, [x, y])).collect();
    let p = PlotParams { voxel_size, wood, buttress, min_points, qsm: qsm_params_from_py(params)? };
    let r = py.detach(|| plot::build_plot(&pts, &labels, h.as_deref(), &stems, &p)).map_err(err)?;
    let share = py.detach(|| plot::median_measured_length(r.models.iter().map(|m| m.1.as_slice())));
    let d = PyDict::new(py);
    let models = PyList::empty(py);
    for (t, rows) in &r.models {
        models.append((*t, rows_to_py(py, rows)?))?;
    }
    let bases = PyList::empty(py);
    for (t, b) in &r.buttresses {
        bases.append((*t, buttress_to_py(py, b)?))?;
    }
    d.set_item("models", models)?;
    d.set_item("buttresses", bases)?;
    d.set_item("skipped", r.skipped)?;
    d.set_item("points", r.points)?;
    d.set_item("heights", r.heights)?;
    d.set_item("median_measured_length", share)?;
    Ok(d)
}

type ButtressArgs<'py> = (PyReadonlyArray2<'py, f64>, PyReadonlyArray2<'py, u32>, f64, f64, f64);
type EntryArgs<'py> = (i64, PyReadonlyArray2<'py, f64>, Option<i64>, Option<f64>, Option<ButtressArgs<'py>>);

type OwnedButtress = (Vec<Point>, Vec<[u32; 3]>, f64, f64, f64);

struct Owned {
    tree_id: i64,
    rows: Vec<Row>,
    points: Option<i64>,
    height: Option<f64>,
    buttress: Option<OwnedButtress>,
}

fn entries_from_py(entries: Vec<EntryArgs<'_>>) -> PyResult<Vec<Owned>> {
    entries
        .into_iter()
        .map(|(tree_id, cyl, points, height, b)| {
            let buttress = match b {
                Some((v, f, volume, top, top_z)) => Some((xyz_from_py(v)?, faces_from_py(f)?, volume, top, top_z)),
                None => None,
            };
            Ok(Owned { tree_id, rows: rows_from_py(cyl)?, points, height, buttress })
        })
        .collect()
}

fn views(owned: &[Owned]) -> Vec<PlotEntry<'_>> {
    owned
        .iter()
        .map(|o| PlotEntry {
            tree_id: o.tree_id,
            rows: &o.rows,
            points: o.points,
            height: o.height,
            buttress: o.buttress.as_ref().map(|(v, f, volume, top, top_z)| ButtressView { vertices: v, faces: f, volume: *volume, top: *top, top_z: *top_z }),
        })
        .collect()
}

#[pyfunction]
fn plot_total_volume(entries: Vec<EntryArgs<'_>>) -> PyResult<f64> {
    Ok(plot::total_volume(&views(&entries_from_py(entries)?)))
}

/// The plot table as `(tree_id, points, volume_m3, dbh_m, height_m,
/// n_cylinders, measured_volume, measured_length, buttress_m3,
/// buttress_top_m)` tuples, None where a value is missing.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn plot_table(py: Python<'_>, entries: Vec<EntryArgs<'_>>) -> PyResult<Vec<(i64, Option<i64>, f64, f64, f64, usize, f64, f64, Option<f64>, Option<f64>)>> {
    let owned = entries_from_py(entries)?;
    let rows = py.detach(|| plot::table(&views(&owned)));
    Ok(rows.into_iter().map(|r| (r.tree_id, r.points, r.volume_m3, r.dbh_m, r.height_m, r.n_cylinders, r.measured_volume, r.measured_length, r.buttress_m3, r.buttress_top_m)).collect())
}

#[pyfunction]
fn plot_write_csv(py: Python<'_>, path: PathBuf, entries: Vec<EntryArgs<'_>>) -> PyResult<()> {
    let owned = entries_from_py(entries)?;
    py.detach(|| plot::write_table_csv(&path, &views(&owned))).map_err(err)
}

#[pyfunction]
fn plot_write_meshes(py: Python<'_>, directory: PathBuf, entries: Vec<EntryArgs<'_>>, fmt: &str, sides: usize, contiguous: bool, prefix: &str) -> PyResult<Vec<PathBuf>> {
    let owned = entries_from_py(entries)?;
    py.detach(|| plot::write_meshes(&directory, &views(&owned), fmt, sides, contiguous, prefix)).map_err(err)
}

#[pyfunction]
fn plot_write_cylinders(py: Python<'_>, directory: PathBuf, entries: Vec<EntryArgs<'_>>, prefix: &str) -> PyResult<()> {
    let owned = entries_from_py(entries)?;
    py.detach(|| plot::write_cylinders(&directory, &views(&owned), prefix)).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(qsm_totals, m)?,
        wrap_pyfunction!(qsm_ends, m)?,
        wrap_pyfunction!(qsm_volumes, m)?,
        wrap_pyfunction!(qsm_volume_above, m)?,
        wrap_pyfunction!(qsm_above, m)?,
        wrap_pyfunction!(qsm_read_csv, m)?,
        wrap_pyfunction!(qsm_write_obj, m)?,
        wrap_pyfunction!(qsm_write_ply, m)?,
        wrap_pyfunction!(write_ply_mesh, m)?,
        wrap_pyfunction!(qsm_buttress_mesh, m)?,
        wrap_pyfunction!(qsm_section, m)?,
        wrap_pyfunction!(qsm_inside, m)?,
        wrap_pyfunction!(qsm_join_fit, m)?,
        wrap_pyfunction!(qsm_fuse, m)?,
        wrap_pyfunction!(tree_mesh_write_obj, m)?,
        wrap_pyfunction!(tree_mesh_write_ply, m)?,
        wrap_pyfunction!(qsm_build_plot, m)?,
        wrap_pyfunction!(plot_total_volume, m)?,
        wrap_pyfunction!(plot_table, m)?,
        wrap_pyfunction!(plot_write_csv, m)?,
        wrap_pyfunction!(plot_write_meshes, m)?,
        wrap_pyfunction!(plot_write_cylinders, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
