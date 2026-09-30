// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Python bindings for sylva-rs.
//!
//! Point clouds cross the boundary as `(xyz: ndarray (N,3) float64, attrs: dict[str, ndarray])`
//! and shots as a dict of arrays; see `python/sylva` for the friendly wrappers.

use std::collections::BTreeMap;
use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::pointcloud::Attr;

mod canopy_py;
mod leaves_py;
mod quality_py;
mod trees_py;
mod filters_py;
mod limits_py;
mod raster_py;
mod registration_py;
mod shots_py;
mod riscan_py;
mod coreg_py;
mod voxels_py;
mod synthetic_py;
mod qsm_py;
use sylva_rs::{canopy, cluster, coreg, coreg_geometry, coreg_ground, coreg_icp as coreg_icp_rs, filters, ground, io, qsm, registration, trees, Point, PointCloud, Raster, Shots, Transform};
mod coreg_pipeline_py;
mod interpolate_py;
mod masks_py;
mod coords_py;
mod als_metrics_py;
mod als_py;
mod als_canopy_py;
mod als_trees_py;
mod change_als_py;
mod change_points_py;
mod change_qsm_py;
mod change_trees_py;
mod waveform_py;
mod fusion_py;

fn err(e: sylva_rs::Error) -> PyErr {
    match e {
        sylva_rs::Error::Io(_) | sylva_rs::Error::File { .. } | sylva_rs::Error::Las(_) => PyIOError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

// ----------------------------------------------------------------- converters

fn xyz_from_py(xyz: PyReadonlyArray2<f64>) -> PyResult<Vec<Point>> {
    let a = xyz.as_array();
    if a.ncols() != 3 {
        return Err(PyValueError::new_err(format!("xyz must have shape (N, 3), got (N, {})", a.ncols())));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn xyz_to_py<'py>(py: Python<'py>, xyz: &[Point]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = xyz.iter().flat_map(|p| p.iter().cloned()).collect();
    PyArray1::from_vec(py, flat).reshape([xyz.len(), 3]).expect("reshape")
}

fn attr_from_py(v: &Bound<'_, PyAny>) -> PyResult<Attr> {
    macro_rules! try_dtype {
        ($($t:ty => $variant:ident),*) => {
            $(if let Ok(a) = v.extract::<PyReadonlyArray1<$t>>() {
                return Ok(Attr::$variant(a.as_array().to_vec()));
            })*
        };
    }
    try_dtype!(f64 => F64, f32 => F32, i64 => I64, i32 => I32, u32 => U32, u16 => U16, u8 => U8, i8 => I8, bool => Bool);
    // Fall back through numpy for other dtypes (e.g. int16, uint64) by asking for float64.
    let np = v.py().import("numpy")?;
    let arr = np.call_method1("asarray", (v,))?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    let wide_unsigned = kind == "u" && arr.getattr("dtype")?.getattr("itemsize")?.extract::<usize>()? > 4;
    if wide_unsigned {
        // uint64 ids must not wrap into 32 bits.
        let cast = arr.call_method1("astype", ("int64",))?;
        return Ok(Attr::I64(cast.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec()));
    }
    let conv = match kind.as_str() {
        "i" => "int64",
        "u" => "uint32",
        "b" => "bool",
        _ => "float64",
    };
    let cast = arr.call_method1("astype", (conv,))?;
    match kind.as_str() {
        "i" => Ok(Attr::I64(cast.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec())),
        "u" => Ok(Attr::U32(cast.extract::<PyReadonlyArray1<u32>>()?.as_array().to_vec())),
        "b" => Ok(Attr::Bool(cast.extract::<PyReadonlyArray1<bool>>()?.as_array().to_vec())),
        _ => Ok(Attr::F64(cast.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec())),
    }
}

fn attr_to_py<'py>(py: Python<'py>, a: &Attr) -> Bound<'py, PyAny> {
    match a {
        Attr::F64(v) => v.clone().into_pyarray(py).into_any(),
        Attr::F32(v) => v.clone().into_pyarray(py).into_any(),
        Attr::I64(v) => v.clone().into_pyarray(py).into_any(),
        Attr::I32(v) => v.clone().into_pyarray(py).into_any(),
        Attr::U32(v) => v.clone().into_pyarray(py).into_any(),
        Attr::U16(v) => v.clone().into_pyarray(py).into_any(),
        Attr::U8(v) => v.clone().into_pyarray(py).into_any(),
        Attr::I8(v) => v.clone().into_pyarray(py).into_any(),
        Attr::Bool(v) => v.clone().into_pyarray(py).into_any(),
    }
}

fn attrs_from_py(attrs: Option<&Bound<'_, PyDict>>) -> PyResult<BTreeMap<String, Attr>> {
    let mut out = BTreeMap::new();
    if let Some(d) = attrs {
        for (k, v) in d.iter() {
            out.insert(k.extract::<String>()?, attr_from_py(&v)?);
        }
    }
    Ok(out)
}

fn attrs_to_py<'py>(py: Python<'py>, attrs: &BTreeMap<String, Attr>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in attrs {
        d.set_item(k, attr_to_py(py, v))?;
    }
    Ok(d)
}

fn cloud_from_py(xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<PointCloud> {
    PointCloud::with_attrs(xyz_from_py(xyz)?, attrs_from_py(attrs)?).map_err(err)
}

fn cloud_to_py<'py>(py: Python<'py>, c: &PointCloud) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    Ok((xyz_to_py(py, &c.xyz), attrs_to_py(py, &c.attrs)?))
}

fn matrix_from_py(m: Option<PyReadonlyArray2<f64>>) -> PyResult<Option<Transform>> {
    match m {
        None => Ok(None),
        Some(m) => {
            let a = m.as_array();
            if a.shape() != [4, 4] {
                return Err(PyValueError::new_err("transform must be 4x4"));
            }
            let flat: Vec<f64> = a.iter().cloned().collect();
            Transform::from_row_major(&flat).map(Some).map_err(err)
        }
    }
}

fn matrix_to_py<'py>(py: Python<'py>, t: &Transform) -> Bound<'py, PyArray2<f64>> {
    PyArray1::from_vec(py, t.to_row_major().to_vec()).reshape([4, 4]).expect("reshape")
}

fn raster_from_py(data: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64) -> Raster {
    let a = data.as_array();
    Raster { data: a.iter().cloned().collect(), nrows: a.nrows(), ncols: a.ncols(), xmin, ymin, resolution }
}

fn raster_to_py<'py>(py: Python<'py>, r: &Raster) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("data", PyArray1::from_vec(py, r.data.clone()).reshape([r.nrows, r.ncols])?)?;
    d.set_item("xmin", r.xmin)?;
    d.set_item("ymin", r.ymin)?;
    d.set_item("resolution", r.resolution)?;
    Ok(d)
}

fn shots_from_py(d: &Bound<'_, PyDict>) -> PyResult<Shots> {
    let get = |k: &str| d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("shots dict missing {k:?}")));
    let origin = xyz_from_py(get("origin")?.extract()?)?;
    let direction = xyz_from_py(get("direction")?.extract()?)?;
    let start = get("echo_start")?.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec();
    let count = get("echo_count")?.extract::<PyReadonlyArray1<i64>>()?.as_array().to_vec();
    let echo_range: Vec<f64> = get("echo_range")?.extract::<PyReadonlyArray1<f64>>()?.as_array().to_vec();
    let echo_attrs = match d.get_item("echo_attrs")? {
        Some(a) => attrs_from_py(Some(a.cast::<PyDict>()?))?,
        None => BTreeMap::new(),
    };
    // Everything downstream indexes echoes by these; check once here so a
    // malformed Shots is a ValueError, not a panic.
    let n = origin.len();
    if direction.len() != n || start.len() != n || count.len() != n {
        return Err(PyValueError::new_err("shots: origin, direction, echo_start and echo_count must have one row per pulse"));
    }
    let n_echo = echo_range.len() as i64;
    for (&s0, &c) in start.iter().zip(&count) {
        if s0 < 0 || c < 0 || c > u32::MAX as i64 || s0 + c > n_echo {
            return Err(PyValueError::new_err("shots: echo_start / echo_count point outside echo_range"));
        }
    }
    for (name, a) in &echo_attrs {
        if a.len() != echo_range.len() {
            return Err(PyValueError::new_err(format!("shots: echo attribute {name:?} has {} values for {} echoes", a.len(), echo_range.len())));
        }
    }
    let echo_start: Vec<usize> = start.iter().map(|&v| v as usize).collect();
    let echo_count: Vec<u32> = count.iter().map(|&v| v as u32).collect();
    Ok(Shots { origin, direction, echo_start, echo_count, echo_range, echo_attrs })
}

fn shots_to_py<'py>(py: Python<'py>, s: &Shots) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("origin", xyz_to_py(py, &s.origin))?;
    d.set_item("direction", xyz_to_py(py, &s.direction))?;
    d.set_item("echo_start", s.echo_start.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("echo_count", s.echo_count.iter().map(|&v| v as i64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("echo_range", s.echo_range.clone().into_pyarray(py))?;
    d.set_item("echo_attrs", attrs_to_py(py, &s.echo_attrs)?)?;
    Ok(d)
}

fn tree_to_py<'py>(py: Python<'py>, t: &trees::Tree) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("tree_id", t.tree_id)?;
    d.set_item("x", t.x)?;
    d.set_item("y", t.y)?;
    d.set_item("dbh", t.dbh)?;
    d.set_item("height", t.height)?;
    d.set_item("n_points", t.n_points)?;
    d.set_item("inlier_fraction", t.inlier_fraction)?;
    d.set_item("n_slices", t.n_slices)?;
    d.set_item("rmse", t.rmse)?;
    d.set_item("lean_deg", t.lean_deg)?;
    d.set_item("quality", t.quality)?;
    Ok(d)
}

fn trees_from_py(list: &Bound<'_, PyList>) -> PyResult<Vec<trees::Tree>> {
    list.iter()
        .map(|t| {
            let d = t.cast::<PyDict>()?;
            let f = |k: &str, default: f64| -> PyResult<f64> { Ok(d.get_item(k)?.map(|v| v.extract::<f64>()).transpose()?.unwrap_or(default)) };
            Ok(trees::Tree {
                tree_id: d.get_item("tree_id")?.map(|v| v.extract::<i64>()).transpose()?.unwrap_or(0),
                x: f("x", 0.0)?,
                y: f("y", 0.0)?,
                dbh: f("dbh", f64::NAN)?,
                height: f("height", f64::NAN)?,
                n_points: d.get_item("n_points")?.map(|v| v.extract::<usize>()).transpose()?.unwrap_or(0),
                inlier_fraction: f("inlier_fraction", f64::NAN)?,
                n_slices: d.get_item("n_slices")?.map(|v| v.extract::<usize>()).transpose()?.unwrap_or(0),
                rmse: f("rmse", f64::NAN)?,
                lean_deg: f("lean_deg", f64::NAN)?,
                quality: f("quality", f64::NAN)?,
            })
        })
        .collect()
}

fn bounds_opt(b: Option<(f64, f64, f64, f64)>) -> Option<(f64, f64, f64, f64)> {
    b
}

// ------------------------------------------------------------------------- io

#[pyfunction]
fn read<'py>(py: Python<'py>, path: PathBuf) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let c = py.detach(|| io::read(&path)).map_err(err)?;
    cloud_to_py(py, &c)
}

#[pyfunction]
#[pyo3(signature = (path, xyz, attrs=None, point_format=6, scale=0.001, binary=true, crs_wkt=None))]
fn write(py: Python<'_>, path: PathBuf, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, point_format: u8, scale: f64, binary: bool, crs_wkt: Option<String>) -> PyResult<()> {
    let c = cloud_from_py(xyz, attrs)?;
    py.detach(|| io::write_with(&c, &path, &io::WriteOptions { point_format, scale, binary, crs_wkt })).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (path, columns=None))]
fn read_ascii<'py>(py: Python<'py>, path: PathBuf, columns: Option<Vec<String>>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let c = py.detach(|| io::ascii::read_ascii(&path, columns.as_deref())).map_err(err)?;
    cloud_to_py(py, &c)
}

fn rxp_opts(library: Option<PathBuf>, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: usize, max_points: Option<usize>, echoes: String, shot_stride: usize) -> io::riegl::RxpOptions {
    io::riegl::RxpOptions { library, drop_pseudo_echoes, min_range, max_range, stride: stride.max(1), max_points, echoes, shot_stride: shot_stride.max(1) }
}

#[pyfunction]
#[pyo3(signature = (path, library=None, drop_pseudo_echoes=true, min_range=0.5, max_range=f64::INFINITY, stride=1, max_points=None, echoes="all".to_string(), shot_stride=1))]
#[allow(clippy::too_many_arguments)]
fn read_rxp<'py>(py: Python<'py>, path: PathBuf, library: Option<PathBuf>, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: usize, max_points: Option<usize>, echoes: String, shot_stride: usize) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let opts = rxp_opts(library, drop_pseudo_echoes, min_range, max_range, stride, max_points, echoes, shot_stride);
    let c = py.detach(|| io::riegl::read_rxp(&path, &opts)).map_err(err)?;
    cloud_to_py(py, &c)
}

#[pyfunction]
#[pyo3(signature = (path, library=None, drop_pseudo_echoes=true, min_range=0.5, max_range=f64::INFINITY, stride=1, max_points=None, echoes="all".to_string(), shot_stride=1))]
#[allow(clippy::too_many_arguments)]
fn read_rxp_shots<'py>(py: Python<'py>, path: PathBuf, library: Option<PathBuf>, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: usize, max_points: Option<usize>, echoes: String, shot_stride: usize) -> PyResult<Bound<'py, PyDict>> {
    let opts = rxp_opts(library, drop_pseudo_echoes, min_range, max_range, stride, max_points, echoes, shot_stride);
    let s = py.detach(|| io::riegl::read_rxp_shots(&path, &opts)).map_err(err)?;
    shots_to_py(py, &s)
}

#[pyfunction]
#[pyo3(signature = (hint=None))]
fn find_rivlib(hint: Option<PathBuf>) -> PyResult<PathBuf> {
    io::riegl::find_rivlib(hint.as_deref()).map_err(err)
}

#[pyfunction]
fn read_matrix_file<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(matrix_to_py(py, &Transform::read_matrix_file(path).map_err(err)?))
}

// ---------------------------------------------------------------------- shots

#[pyfunction]
#[pyo3(signature = (xyz, attrs=None, origin=(0.0, 0.0, 0.0)))]
fn shots_from_pointcloud<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, origin: (f64, f64, f64)) -> PyResult<Bound<'py, PyDict>> {
    let c = cloud_from_py(xyz, attrs)?;
    shots_to_py(py, &Shots::from_pointcloud(&c, [origin.0, origin.1, origin.2]))
}

#[pyfunction]
#[pyo3(signature = (xyz, attrs=None))]
fn shots_from_ray_cloud<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<Bound<'py, PyDict>> {
    let c = cloud_from_py(xyz, attrs)?;
    shots_to_py(py, &Shots::from_ray_cloud(&c).map_err(err)?)
}

#[pyfunction]
fn shots_to_pointcloud<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let s = shots_from_py(shots)?;
    cloud_to_py(py, &s.to_pointcloud())
}

#[pyfunction]
fn shots_transform<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, matrix: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyDict>> {
    let s = shots_from_py(shots)?;
    let t = matrix_from_py(Some(matrix))?.unwrap();
    shots_to_py(py, &s.transformed(&t))
}

// -------------------------------------------------------------------- filters

#[pyfunction]
fn voxel_downsample_indices<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let p = xyz_from_py(xyz)?;
    let idx = py.detach(|| filters::voxel_downsample_indices(&p, voxel_size));
    Ok(idx.into_iter().map(|i| i as i64).collect::<Vec<_>>().into_pyarray(py))
}

#[pyfunction]
fn voxel_centroids<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let c = PointCloud::new(xyz_from_py(xyz)?);
    let out = py.detach(|| filters::voxel_downsample(&c, voxel_size, true));
    Ok(xyz_to_py(py, &out.xyz))
}

#[pyfunction]
fn random_indices<'py>(py: Python<'py>, total: usize, n: usize, seed: u64) -> Bound<'py, PyArray1<i64>> {
    filters::random_indices(total, n, seed).into_iter().map(|i| i as i64).collect::<Vec<_>>().into_pyarray(py)
}

#[pyfunction]
fn min_distance_indices<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, distance: f64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let p = xyz_from_py(xyz)?;
    let idx = py.detach(|| filters::min_distance_indices(&p, distance));
    Ok(idx.into_iter().map(|i| i as i64).collect::<Vec<_>>().into_pyarray(py))
}

#[pyfunction]
fn statistical_outlier_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, k: usize, std_ratio: f64) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    Ok(py.detach(|| filters::statistical_outlier_mask(&p, k, std_ratio)).into_pyarray(py))
}

#[pyfunction]
fn radius_outlier_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, radius: f64, min_neighbors: usize) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    Ok(py.detach(|| filters::radius_outlier_mask(&p, radius, min_neighbors)).into_pyarray(py))
}

#[pyfunction]
fn estimate_normals<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, k: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = xyz_from_py(xyz)?;
    let n = py.detach(|| filters::estimate_normals(&p, k));
    Ok(xyz_to_py(py, &n))
}

#[pyfunction]
fn planarity_linearity<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, k: usize) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let p = xyz_from_py(xyz)?;
    let (pl, li) = py.detach(|| filters::planarity_linearity(&p, k));
    Ok((pl.into_pyarray(py), li.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (xyz, k=20, high_threshold=0.85, medium_threshold=0.75, scale_radius=0.0, graph_k=10, max_edge=1.0, base_height=0.25, target_res=0.2, min_passage=3, assign_dist=0.05, assign_scale=0.0, component_res=0.05, component_min=200, sor_k=50, sor_std=1.0, dilate_dist=0.03, passage=true))]
#[allow(clippy::too_many_arguments)]
fn wood_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, k: usize, high_threshold: f64, medium_threshold: f64, scale_radius: f64, graph_k: usize, max_edge: f64, base_height: f64, target_res: f64, min_passage: usize, assign_dist: f64, assign_scale: f64, component_res: f64, component_min: usize, sor_k: usize, sor_std: f64, dilate_dist: f64, passage: bool) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    let params = qsm::wood::WoodParams { k, high_threshold, medium_threshold, scale_radius, graph_k, max_edge, base_height, target_res, min_passage, assign_dist, assign_scale, component_res, component_min, sor_k, sor_std, dilate_dist, passage };
    Ok(py.detach(|| qsm::wood::wood_mask(&p, &params)).into_pyarray(py))
}

#[pyfunction]
fn euclidean_clusters<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, radius: f64, min_points: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let p = xyz_from_py(xyz)?;
    Ok(py.detach(|| cluster::euclidean_clusters(&p, radius, min_points)).into_pyarray(py))
}

/// Points within `radius` of each point, itself included.
#[pyfunction]
fn count_within<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, radius: f64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let p = xyz_from_py(xyz)?;
    let counts = py.detach(|| filters::count_within(&p, radius));
    Ok(counts.into_iter().map(|c| c as i64).collect::<Vec<_>>().into_pyarray(py))
}

#[pyfunction]
fn knn<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, queries: PyReadonlyArray2<f64>, k: usize) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)> {
    let p = xyz_from_py(xyz)?;
    let q = xyz_from_py(queries)?;
    let (d, i) = py.detach(|| filters::knn(&p, &q, k));
    Ok((PyArray1::from_vec(py, d).reshape([q.len(), k])?, PyArray1::from_vec(py, i).reshape([q.len(), k])?))
}

// --------------------------------------------------------------------- ground

#[pyfunction]
#[pyo3(signature = (xyz, cloth_resolution=0.5, rigidness=2, class_threshold=0.3, iterations=500, time_step=0.65))]
fn csf_ground_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, cloth_resolution: f64, rigidness: usize, class_threshold: f64, iterations: usize, time_step: f64) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    let params = ground::CsfParams { cloth_resolution, rigidness, class_threshold, iterations, time_step };
    Ok(py.detach(|| ground::csf_ground_mask(&p, &params)).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (xyz, cell_size=0.5, max_window=10.0, slope=0.3, initial_distance=0.15, max_distance=2.0))]
fn pmf_ground_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, cell_size: f64, max_window: f64, slope: f64, initial_distance: f64, max_distance: f64) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    let params = ground::PmfParams { cell_size, max_window, slope, initial_distance, max_distance };
    Ok(py.detach(|| ground::pmf_ground_mask(&p, &params)).map_err(err)?.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (ground_xyz, resolution=0.5, bounds=None))]
fn make_dtm<'py>(py: Python<'py>, ground_xyz: PyReadonlyArray2<f64>, resolution: f64, bounds: Option<(f64, f64, f64, f64)>) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(ground_xyz)?;
    let r = py.detach(|| ground::make_dtm(&p, resolution, bounds_opt(bounds))).map_err(err)?;
    raster_to_py(py, &r)
}

#[pyfunction]
fn raster_sample<'py>(py: Python<'py>, data: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r = raster_from_py(data, xmin, ymin, resolution);
    let xs = x.as_array();
    let ys = y.as_array();
    Ok(r.sample_many(xs.iter().cloned().zip(ys.iter().cloned())).into_pyarray(py))
}

#[pyfunction]
fn raster_fill_nearest<'py>(py: Python<'py>, data: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut r = raster_from_py(data, 0.0, 0.0, 1.0);
    r.fill_nearest();
    Ok(PyArray1::from_vec(py, r.data).reshape([r.nrows, r.ncols])?)
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, resolution=0.5, bounds=None, min_height=0.0))]
fn make_chm<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, resolution: f64, bounds: Option<(f64, f64, f64, f64)>, min_height: f64) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let r = py.detach(|| ground::make_chm(&p, &h, resolution, bounds_opt(bounds), min_height)).map_err(err)?;
    raster_to_py(py, &r)
}

#[pyfunction]
#[pyo3(signature = (path, data, xmin, ymin, resolution, nodata=-9999.0))]
fn write_ascii_grid(path: PathBuf, data: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64, nodata: f64) -> PyResult<()> {
    raster_from_py(data, xmin, ymin, resolution).write_ascii_grid(path, nodata).map_err(err)
}

#[pyfunction]
fn read_ascii_grid<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    raster_to_py(py, &Raster::read_ascii_grid(path).map_err(err)?)
}

// --------------------------------------------------------------------- canopy

#[pyfunction]
#[pyo3(signature = (xyz, voxel_size, origin=None, shape=None))]
fn voxelize<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel_size: f64, origin: Option<(f64, f64, f64)>, shape: Option<(usize, usize, usize)>) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let g = py.detach(|| canopy::voxelize(&p, voxel_size, origin.map(|o| [o.0, o.1, o.2]), shape.map(|s| [s.0, s.1, s.2]))).map_err(err)?;
    let d = PyDict::new(py);
    // Return counts as (nz, ny, nx) so numpy indexing is [k, j, i]; flat layout matches.
    let counts: Vec<i64> = g.counts.iter().map(|&c| c as i64).collect();
    d.set_item("counts", PyArray1::from_vec(py, counts).reshape([g.shape[2], g.shape[1], g.shape[0]])?)?;
    d.set_item("origin", g.origin.to_vec().into_pyarray(py))?;
    d.set_item("voxel_size", g.voxel_size)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, voxel_size=0.5, max_height=None, clumping=1.0))]
fn pad_profile_voxel<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, voxel_size: f64, max_height: Option<f64>, clumping: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let (z, pad) = py.detach(|| canopy::pad_profile_voxel(&p, &h, voxel_size, max_height, clumping)).map_err(err)?;
    Ok((z.into_pyarray(py), pad.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (shots, echo_heights, min_height=0.0, zenith_edges=None))]
fn gap_fraction_zenith<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, echo_heights: PyReadonlyArray1<f64>, min_height: f64, zenith_edges: Option<PyReadonlyArray1<f64>>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let s = shots_from_py(shots)?;
    let h = echo_heights.as_array().to_vec();
    let edges: Vec<f64> = zenith_edges.map(|e| e.as_array().to_vec()).unwrap_or_else(|| (0..=18).map(|i| i as f64 * 5.0).collect());
    let (c, g) = canopy::gap_fraction_zenith(&s, &h, min_height, &edges);
    Ok((c.into_pyarray(py), g.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (zenith_deg, gap_fraction, method="hinge"))]
fn lai_from_gap_fraction(zenith_deg: PyReadonlyArray1<f64>, gap_fraction: PyReadonlyArray1<f64>, method: &str) -> PyResult<f64> {
    canopy::lai_from_gap_fraction(&zenith_deg.as_array().to_vec(), &gap_fraction.as_array().to_vec(), method).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (chm, threshold=2.0))]
fn canopy_cover(chm: PyReadonlyArray2<f64>, threshold: f64) -> f64 {
    canopy::canopy_cover(&chm.as_array().iter().cloned().collect::<Vec<_>>(), threshold)
}

#[pyfunction]
#[pyo3(signature = (shots, voxel_size, origin=None, shape=None, min_hits=2))]
fn density_grid<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, voxel_size: f64, origin: Option<(f64, f64, f64)>, shape: Option<(usize, usize, usize)>, min_hits: u32) -> PyResult<Bound<'py, PyDict>> {
    let s = shots_from_py(shots)?;
    let g = py.detach(|| match (origin, shape) {
        (Some(o), Some(sh)) => {
            let mut g = canopy::DensityGrid::new([o.0, o.1, o.2], voxel_size, [sh.0, sh.1, sh.2]);
            g.add_shots(&s);
            Ok(g)
        }
        _ => canopy::density_grid_from_shots(&s, voxel_size),
    })
    .map_err(err)?;
    let d = PyDict::new(py);
    let shp = [g.shape[2], g.shape[1], g.shape[0]];
    d.set_item("n_rays", PyArray1::from_vec(py, g.n_rays.iter().map(|&v| v as i64).collect()).reshape(shp)?)?;
    d.set_item("n_hits", PyArray1::from_vec(py, g.n_hits.iter().map(|&v| v as i64).collect()).reshape(shp)?)?;
    d.set_item("path_length", PyArray1::from_vec(py, g.path_length.clone()).reshape(shp)?)?;
    d.set_item("density", PyArray1::from_vec(py, g.density(min_hits)).reshape(shp)?)?;
    d.set_item("profile", g.vertical_profile(min_hits).into_pyarray(py))?;
    d.set_item("origin", g.origin.to_vec().into_pyarray(py))?;
    d.set_item("voxel_size", g.voxel_size)?;
    Ok(d)
}


#[pyfunction]
#[pyo3(signature = (shots, path, double=false, row_group_size=1048576, zstd_level=3, origin_tolerance=1e-3))]
fn write_shots(py: Python<'_>, shots: &Bound<'_, PyDict>, path: PathBuf, double: bool, row_group_size: usize, zstd_level: i32, origin_tolerance: f64) -> PyResult<()> {
    let s = shots_from_py(shots)?;
    let opts = io::shots::ShotsWriteOptions { double, row_group_size, zstd_level, origin_tolerance };
    py.detach(|| io::shots::write_shots(&s, &path, &opts)).map_err(err)
}

/// Read a shots file, or only the row groups listed in `groups`.
#[pyfunction]
#[pyo3(signature = (path, groups=None))]
fn read_shots<'py>(py: Python<'py>, path: PathBuf, groups: Option<Vec<usize>>) -> PyResult<Bound<'py, PyDict>> {
    let s = py
        .detach(|| {
            let file = io::shots::ShotsFile::open(&path)?;
            match groups {
                None => file.read_all(),
                Some(g) => file.read_groups(&g),
            }
        })
        .map_err(err)?;
    shots_to_py(py, &s)
}

/// Header of a shots file: counts, row groups, echo bounds, scanner positions, attributes.
#[pyfunction]
fn shots_info<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyDict>> {
    let file = io::shots::ShotsFile::open(&path).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("n_shots", file.n_shots)?;
    d.set_item("n_echoes", file.n_echoes)?;
    d.set_item("n_groups", file.n_groups())?;
    d.set_item("bounds", (file.bounds.0.to_vec(), file.bounds.1.to_vec()))?;
    d.set_item("scans", xyz_to_py(py, file.scans()))?;
    d.set_item("echo_attrs", file.attr_names().collect::<Vec<_>>())?;
    Ok(d)
}


// --------------------------------------------------------------- registration

#[pyfunction]
fn kabsch<'py>(py: Python<'py>, source: PyReadonlyArray2<f64>, target: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t = registration::kabsch(&xyz_from_py(source)?, &xyz_from_py(target)?).map_err(err)?;
    Ok(matrix_to_py(py, &t))
}

#[pyfunction]
#[pyo3(signature = (source, target, init=None, max_correspondence_distance=0.5, max_iterations=50, tolerance=1e-6, method="point", trim=1.0, normal_k=12))]
#[allow(clippy::too_many_arguments)]
fn icp<'py>(py: Python<'py>, source: PyReadonlyArray2<f64>, target: PyReadonlyArray2<f64>, init: Option<PyReadonlyArray2<f64>>, max_correspondence_distance: f64, max_iterations: usize, tolerance: f64, method: &str, trim: f64, normal_k: usize) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let s = PointCloud::new(xyz_from_py(source)?);
    let t = PointCloud::new(xyz_from_py(target)?);
    let init = matrix_from_py(init)?;
    let params = registration::IcpParams { max_correspondence_distance, max_iterations, tolerance, method: method.to_string(), trim, normal_k };
    let r = py.detach(|| registration::icp(&s, &t, init, &params)).map_err(err)?;
    let info = PyDict::new(py);
    info.set_item("rmse", r.rmse)?;
    info.set_item("iterations", r.iterations)?;
    info.set_item("n_correspondences", r.n_correspondences)?;
    Ok((matrix_to_py(py, &r.transform), info))
}

fn stem_match_to_py<'py>(py: Python<'py>, r: &coreg::StemMatch) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("transform", matrix_to_py(py, &r.transform))?;
    d.set_item("n_inliers", r.n_inliers)?;
    d.set_item("inlier_rmse", r.inlier_rmse)?;
    d.set_item("score", r.score)?;
    let c: Vec<i64> = r.correspondences.iter().flat_map(|&(i, j)| [i as i64, j as i64]).collect();
    d.set_item("correspondences", numpy::ndarray::Array2::from_shape_vec((r.correspondences.len(), 2), c).unwrap().into_pyarray(py))?;
    d.set_item("n_source", r.n_source)?;
    d.set_item("n_target", r.n_target)?;
    d.set_item("success", r.success)?;
    d.set_item("ambiguity", r.ambiguity)?;
    match &r.rival {
        Some(rv) => d.set_item("rival", stem_match_to_py(py, rv)?)?,
        None => d.set_item("rival", py.None())?,
    }
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (source, source_diameters, source_qualities, target, target_diameters, target_qualities, min_pair_distance=2.0, max_pair_distance=35.0, pair_distance_tolerance=0.25, inlier_tolerance=0.40, diameter_rel_tolerance=0.30, diameter_abs_tolerance=0.04, use_diameters=true, max_stems=70, max_hypotheses=60000, min_inliers=4, early_exit_inliers=40, distinct_translation=1.0, distinct_yaw_deg=5.0, refine_iterations=6))]
#[allow(clippy::too_many_arguments)]
fn match_stem_maps<'py>(py: Python<'py>, source: PyReadonlyArray2<f64>, source_diameters: PyReadonlyArray1<f64>, source_qualities: PyReadonlyArray1<f64>, target: PyReadonlyArray2<f64>, target_diameters: PyReadonlyArray1<f64>, target_qualities: PyReadonlyArray1<f64>, min_pair_distance: f64, max_pair_distance: f64, pair_distance_tolerance: f64, inlier_tolerance: f64, diameter_rel_tolerance: f64, diameter_abs_tolerance: f64, use_diameters: bool, max_stems: usize, max_hypotheses: usize, min_inliers: usize, early_exit_inliers: usize, distinct_translation: f64, distinct_yaw_deg: f64, refine_iterations: usize) -> PyResult<Bound<'py, PyDict>> {
    let src = coreg::StemMap { positions: xyz_from_py(source)?, diameters: source_diameters.as_array().to_vec(), qualities: source_qualities.as_array().to_vec() };
    let dst = coreg::StemMap { positions: xyz_from_py(target)?, diameters: target_diameters.as_array().to_vec(), qualities: target_qualities.as_array().to_vec() };
    if src.diameters.len() != src.len() || src.qualities.len() != src.len() || dst.diameters.len() != dst.len() || dst.qualities.len() != dst.len() {
        return Err(PyValueError::new_err("diameters and qualities must have one value per stem"));
    }
    let p = coreg::MatchParams { min_pair_distance, max_pair_distance, pair_distance_tolerance, inlier_tolerance, diameter_rel_tolerance, diameter_abs_tolerance, use_diameters, max_stems, max_hypotheses, min_inliers, early_exit_inliers, distinct_translation, distinct_yaw_deg, refine_iterations };
    let r = py.detach(|| coreg::match_stem_maps(&src, &dst, &p));
    stem_match_to_py(py, &r)
}

// ------------------------------------------------------- coreg ground model

/// View a C-contiguous `(N, K)` float64 array as `&[[f64; K]]`, or copy it
/// row by row when it is not contiguous.
fn rows_from_py<'a, const K: usize>(a: &'a PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<std::borrow::Cow<'a, [[f64; K]]>> {
    let v = a.as_array();
    if v.ncols() != K {
        return Err(PyValueError::new_err(format!("{name} must have shape (N, {K}), got (N, {})", v.ncols())));
    }
    if let Ok(s) = a.as_slice() {
        // SAFETY: `[f64; K]` has the size and alignment of K consecutive f64.
        let rows = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const [f64; K], s.len() / K) };
        return Ok(std::borrow::Cow::Borrowed(rows));
    }
    Ok(std::borrow::Cow::Owned(v.rows().into_iter().map(|r| std::array::from_fn(|k| r[k])).collect()))
}

#[pyfunction]
#[pyo3(signature = (xyz, cell_size=0.5, percentile=5.0, max_slope=1.0, smooth_cells=3, opening_cells=5, min_points_per_cell=1, pit_depth=3.0, pit_window=9, max_points=8_000_000, seed=0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn coreg_fit_ground<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, cell_size: f64, percentile: f64, max_slope: f64, smooth_cells: usize, opening_cells: usize, min_points_per_cell: usize, pit_depth: f64, pit_window: usize, max_points: usize, seed: u64) -> PyResult<(Bound<'py, PyArray2<f64>>, (f64, f64), Bound<'py, PyArray2<bool>>)> {
    let pts = rows_from_py::<3>(&xyz, "xyz")?;
    let p = coreg_ground::GroundParams { cell_size, percentile, max_slope, smooth_cells, opening_cells, min_points_per_cell, pit_depth, pit_window, max_points: (max_points > 0).then_some(max_points), seed };
    let pts: &[[f64; 3]] = &pts;
    let g = py.detach(|| coreg_ground::fit_ground(pts, &p)).map_err(err)?;
    let (ny, nx) = (g.ny, g.nx);
    Ok((PyArray1::from_vec(py, g.elevation).reshape([ny, nx])?, (g.origin[0], g.origin[1]), PyArray1::from_vec(py, g.observed).reshape([ny, nx])?))
}

#[pyfunction]
fn coreg_ground_height<'py>(py: Python<'py>, elevation: PyReadonlyArray2<f64>, x0: f64, y0: f64, cell_size: f64, xy: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (ny, nx) = (elevation.shape()[0], elevation.shape()[1]);
    if ny == 0 || nx == 0 {
        return Err(PyValueError::new_err("elevation grid is empty"));
    }
    let e: Vec<f64> = elevation.as_array().iter().cloned().collect();
    let q = rows_from_py::<2>(&xy, "xy")?;
    let q: &[[f64; 2]] = &q;
    Ok(py.detach(|| coreg_ground::height_at_many(&e, nx, ny, [x0, y0], cell_size, q)).into_pyarray(py))
}

#[pyfunction]
fn coreg_ground_support<'py>(py: Python<'py>, observed: PyReadonlyArray2<bool>, x0: f64, y0: f64, cell_size: f64, xy: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let (ny, nx) = (observed.shape()[0], observed.shape()[1]);
    if ny == 0 || nx == 0 {
        return Err(PyValueError::new_err("observed grid is empty"));
    }
    let o: Vec<bool> = observed.as_array().iter().cloned().collect();
    let q = rows_from_py::<2>(&xy, "xy")?;
    let q: &[[f64; 2]] = &q;
    Ok(py.detach(|| coreg_ground::support_many(&o, nx, ny, [x0, y0], cell_size, q)).into_pyarray(py))
}

// ------------------------------------------------------------ co-registration

fn matrix4_from_py(m: PyReadonlyArray2<f64>) -> PyResult<coreg_icp_rs::Mat4> {
    Ok(matrix_from_py(Some(m))?.expect("matrix").0)
}

fn matrix4_to_py<'py>(py: Python<'py>, m: &coreg_icp_rs::Mat4) -> Bound<'py, PyArray2<f64>> {
    matrix_to_py(py, &Transform(*m))
}

/// Voxel downsampling with per-voxel counts: `(points, counts)`.
#[pyfunction]
#[pyo3(signature = (xyz, voxel, centroid=true))]
fn coreg_voxel_centroids<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, voxel: f64, centroid: bool) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
    if voxel.is_nan() || voxel <= 0.0 {
        return Err(PyValueError::new_err("voxel size must be positive"));
    }
    let p = xyz_from_py(xyz)?;
    let (c, n) = py.detach(|| coreg_geometry::voxel_centroids(&p, voxel, centroid));
    Ok((xyz_to_py(py, &c), n.into_iter().map(|v| v as i64).collect::<Vec<_>>().into_pyarray(py)))
}

/// Local PCA: `(normals, planarity, valid, evals)` with ascending eigenvalues.
#[pyfunction]
#[pyo3(signature = (xyz, k=20, radius=None))]
fn coreg_local_pca<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, k: usize, radius: Option<f64>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<bool>>, Bound<'py, PyArray2<f64>>)> {
    let p = xyz_from_py(xyz)?;
    let (normals, planarity, valid, evals) = py.detach(|| {
        if p.len() < 3 {
            let n = p.len();
            return (vec![[0.0; 3]; n], vec![0.0; n], vec![false; n], vec![[0.0; 3]; n]);
        }
        let pca = coreg_geometry::local_pca(&p, k, radius);
        let (n, pl) = coreg_geometry::normals_from_pca(&pca);
        (n, pl, pca.valid, pca.evals)
    });
    Ok((xyz_to_py(py, &normals), planarity.into_pyarray(py), valid.into_pyarray(py), xyz_to_py(py, &evals)))
}

/// Planarity filter: the locally planar points (or voxel centroids).
#[pyfunction]
#[pyo3(signature = (xyz, min_planarity=0.35, voxel=Some(0.05), k=20, radius=Some(0.15)))]
fn coreg_planar_filter<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, min_planarity: f64, voxel: Option<f64>, k: usize, radius: Option<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if matches!(voxel, Some(v) if v < 0.0) {
        return Err(PyValueError::new_err("voxel size must be positive"));
    }
    let p = xyz_from_py(xyz)?;
    let out = py.detach(|| coreg_geometry::planar_filter(&p, min_planarity, voxel, k, radius));
    Ok(xyz_to_py(py, &out))
}

/// Point-to-plane / point-to-point ICP over a voxel pyramid.
#[pyfunction]
#[pyo3(signature = (source, target, initial=None, voxel_sizes=vec![0.30, 0.15, 0.07, 0.05], max_distances=Some(vec![0.80, 0.40, 0.20, 0.12]), max_iterations=30, method="point_to_plane", robust="huber", robust_scale=0.05, trim_fraction=0.85, trim_ramp=3, min_planarity=0.25, normal_neighbours=20, translation_tolerance=1e-4, rotation_tolerance=2e-5, fitness_threshold=0.10, damping=1e-6, max_points=120_000, plateau_tolerance=0.0, plateau_patience=3, seed=0, prepared=None))]
#[allow(clippy::too_many_arguments)]
fn coreg_icp<'py>(py: Python<'py>, source: PyReadonlyArray2<f64>, target: Option<PyReadonlyArray2<f64>>, initial: Option<PyReadonlyArray2<f64>>, voxel_sizes: Vec<f64>, max_distances: Option<Vec<f64>>, max_iterations: usize, method: &str, robust: &str, robust_scale: f64, trim_fraction: f64, trim_ramp: usize, min_planarity: f64, normal_neighbours: usize, translation_tolerance: f64, rotation_tolerance: f64, fitness_threshold: f64, damping: f64, max_points: usize, plateau_tolerance: f64, plateau_patience: usize, seed: u64, prepared: Option<PyRef<'_, PyCoregIcpTarget>>) -> PyResult<Bound<'py, PyDict>> {
    let s = xyz_from_py(source)?;
    let init = initial.map(matrix4_from_py).transpose()?;
    let cfg = coreg_icp_rs::IcpConfig { voxel_sizes, max_distances, max_iterations, method: method.to_string(), robust: robust.to_string(), robust_scale, trim_fraction, trim_ramp, min_planarity, normal_neighbours, translation_tolerance, rotation_tolerance, fitness_threshold, damping, max_points, plateau_tolerance, plateau_patience, seed };
    let r = match (prepared, target) {
        (Some(pt), _) => {
            let pt = pt.inner.clone();
            py.detach(|| coreg_icp_rs::icp_prepared(&s, &pt, init, &cfg)).map_err(err)?
        }
        (None, Some(target)) => {
            let t = xyz_from_py(target)?;
            py.detach(|| coreg_icp_rs::icp(&s, &t, init, &cfg)).map_err(err)?
        }
        (None, None) => return Err(PyValueError::new_err("give a target or a prepared target")),
    };
    let d = PyDict::new(py);
    d.set_item("transform", matrix4_to_py(py, &r.transform))?;
    d.set_item("fitness", r.fitness)?;
    d.set_item("inlier_rmse", r.inlier_rmse)?;
    d.set_item("n_correspondences", r.n_correspondences)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("converged", r.converged)?;
    d.set_item("history", r.history)?;
    set_plane_information(py, &d, r.information.as_ref())?;
    Ok(d)
}

fn set_plane_information(py: Python<'_>, d: &Bound<'_, PyDict>, info: Option<&coreg_icp_rs::PlaneInformation>) -> PyResult<()> {
    match info {
        Some(info) => {
            let h: Vec<Vec<f64>> = (0..6).map(|r| (0..6).map(|c| info.hessian[(r, c)]).collect()).collect();
            d.set_item("hessian", PyArray2::from_vec2(py, &h)?)?;
            d.set_item("plane_sigma", info.sigma)?;
            d.set_item("plane_n", info.n)?;
        }
        None => d.set_item("hessian", py.None())?,
    }
    Ok(())
}

/// Point-to-plane information of `transform` (`coreg_icp_rs::plane_information`):
/// a dict with `hessian` (None if unavailable), `plane_sigma` and `plane_n`.
#[pyfunction]
#[pyo3(signature = (source, target, transform, voxel_sizes=vec![0.30, 0.15, 0.07, 0.05], max_distances=Some(vec![0.80, 0.40, 0.20, 0.12]), robust="huber", robust_scale=0.05, trim_fraction=0.85, trim_ramp=3, min_planarity=0.25, normal_neighbours=20, max_points=120_000, seed=0, prepared=None))]
#[allow(clippy::too_many_arguments)]
fn coreg_plane_information<'py>(py: Python<'py>, source: PyReadonlyArray2<f64>, target: Option<PyReadonlyArray2<f64>>, transform: PyReadonlyArray2<f64>, voxel_sizes: Vec<f64>, max_distances: Option<Vec<f64>>, robust: &str, robust_scale: f64, trim_fraction: f64, trim_ramp: usize, min_planarity: f64, normal_neighbours: usize, max_points: usize, seed: u64, prepared: Option<PyRef<'_, PyCoregIcpTarget>>) -> PyResult<Bound<'py, PyDict>> {
    let s = xyz_from_py(source)?;
    let m = matrix4_from_py(transform)?;
    let cfg = coreg_icp_rs::IcpConfig { voxel_sizes, max_distances, robust: robust.to_string(), robust_scale, trim_fraction, trim_ramp, min_planarity, normal_neighbours, max_points, seed, ..Default::default() };
    let info = match (prepared, target) {
        (Some(pt), _) => {
            let pt = pt.inner.clone();
            if !pt.matches(&cfg) {
                return Err(PyValueError::new_err("the prepared ICP target was built with other pyramid settings"));
            }
            py.detach(|| coreg_icp_rs::plane_information(&s, &pt, &m, &cfg))
        }
        (None, Some(target)) => {
            let t = xyz_from_py(target)?;
            py.detach(|| coreg_icp_rs::plane_information(&s, &coreg_icp_rs::IcpTarget::new(&t, &cfg), &m, &cfg))
        }
        (None, None) => return Err(PyValueError::new_err("give a target or a prepared target")),
    };
    let d = PyDict::new(py);
    set_plane_information(py, &d, info.as_ref())?;
    Ok(d)
}

/// Registration quality: `(fitness, inlier_rmse, n_inliers)`.
#[pyfunction]
#[pyo3(signature = (source, target, transform, threshold=0.10, max_points=200_000, voxel=Some(0.05), seed=0))]
#[allow(clippy::too_many_arguments)]
fn coreg_evaluate(py: Python<'_>, source: PyReadonlyArray2<f64>, target: PyReadonlyArray2<f64>, transform: PyReadonlyArray2<f64>, threshold: f64, max_points: usize, voxel: Option<f64>, seed: u64) -> PyResult<(f64, f64, usize)> {
    let s = xyz_from_py(source)?;
    let t = xyz_from_py(target)?;
    let m = matrix4_from_py(transform)?;
    if matches!(voxel, Some(v) if v < 0.0) {
        return Err(PyValueError::new_err("voxel size must be positive"));
    }
    Ok(py.detach(|| coreg_icp_rs::evaluate_registration(&s, &t, &m, threshold, max_points, voxel, seed)))
}

/// k-d tree with scipy `cKDTree.query(k=1)` conventions; 2-D points are
/// padded with z = 0.
/// A target scan's ICP pyramid, built once and reused for every scan registered
/// against it (`coreg_icp(..., prepared=...)`).
#[pyclass(name = "CoregIcpTarget", frozen)]
struct PyCoregIcpTarget {
    inner: std::sync::Arc<coreg_icp_rs::IcpTarget>,
}

#[pymethods]
impl PyCoregIcpTarget {
    #[new]
    #[pyo3(signature = (target, voxel_sizes, max_points=120000, method="point_to_plane", normal_neighbours=20, seed=0))]
    fn new(py: Python<'_>, target: PyReadonlyArray2<f64>, voxel_sizes: Vec<f64>, max_points: usize, method: &str, normal_neighbours: usize, seed: u64) -> PyResult<Self> {
        let t = xyz_from_py(target)?;
        let cfg = coreg_icp_rs::IcpConfig { voxel_sizes, max_points, method: method.to_string(), normal_neighbours, seed, ..Default::default() };
        let inner = py.detach(|| coreg_icp_rs::IcpTarget::new(&t, &cfg));
        Ok(PyCoregIcpTarget { inner: std::sync::Arc::new(inner) })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(name = "CoregKdTree")]
struct PyCoregKdTree {
    tree: coreg_geometry::CoregTree,
    dim: usize,
}

fn padded_from_py(a: PyReadonlyArray2<f64>, what: &str) -> PyResult<(Vec<Point>, usize)> {
    let a = a.as_array();
    match a.ncols() {
        3 => Ok((a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect(), 3)),
        2 => Ok((a.rows().into_iter().map(|r| [r[0], r[1], 0.0]).collect(), 2)),
        c => Err(PyValueError::new_err(format!("{what} must have shape (N, 2) or (N, 3), got (N, {c})"))),
    }
}

#[pymethods]
impl PyCoregKdTree {
    #[new]
    fn new(py: Python<'_>, points: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let (p, dim) = padded_from_py(points, "points")?;
        let tree = py.detach(|| coreg_geometry::CoregTree::new(&p));
        Ok(PyCoregKdTree { tree, dim })
    }

    #[getter]
    fn n(&self) -> usize {
        self.tree.len()
    }

    #[getter]
    fn m(&self) -> usize {
        self.dim
    }

    fn __len__(&self) -> usize {
        self.tree.len()
    }

    /// Nearest neighbour of each query: `(dist, index)`, with `inf` / `n`
    /// where none lies strictly within `distance_upper_bound`.
    #[pyo3(signature = (queries, distance_upper_bound=f64::INFINITY))]
    fn query<'py>(&self, py: Python<'py>, queries: PyReadonlyArray2<f64>, distance_upper_bound: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
        let (q, dim) = padded_from_py(queries, "queries")?;
        if dim != self.dim {
            return Err(PyValueError::new_err(format!("queries have {dim} columns but the tree has {}", self.dim)));
        }
        let (d, i) = py.detach(|| self.tree.query(&q, distance_upper_bound));
        Ok((d.into_pyarray(py), i.into_iter().map(|v| v as i64).collect::<Vec<_>>().into_pyarray(py)))
    }
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, cx, cy, ground_z, resolution=0.02, slice=0.05, close_radius=0.08, max_radius=4.0, max_height=6.0, top=None, solidity=0.9, round_run=4, min_top=0.5, min_points=30, max_flare=1.0, smooth=10))]
#[allow(clippy::too_many_arguments)]
fn buttress_mesh<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, cx: f64, cy: f64, ground_z: f64, resolution: f64, slice: f64, close_radius: f64, max_radius: f64, max_height: f64, top: Option<f64>, solidity: f64, round_run: usize, min_top: f64, min_points: usize, max_flare: f64, smooth: usize) -> PyResult<Bound<'py, PyDict>> {
    let pts = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    if h.len() != pts.len() {
        return Err(PyValueError::new_err("heights must have one value per point"));
    }
    if resolution <= 0.0 || slice <= 0.0 || max_height <= 0.0 {
        return Err(PyValueError::new_err("resolution, slice and max_height must be positive"));
    }
    let p = qsm::buttress::ButtressParams { resolution, slice, close_radius, max_radius, max_height, top, solidity, round_run, min_top, min_points, max_flare, smooth };
    let b = py.detach(|| qsm::buttress::buttress_mesh(&pts, &h, cx, cy, ground_z, &p));
    let d = PyDict::new(py);
    let v: Vec<f64> = b.vertices.iter().flatten().copied().collect();
    d.set_item("vertices", numpy::ndarray::Array2::from_shape_vec((b.vertices.len(), 3), v).unwrap().into_pyarray(py))?;
    let f: Vec<i64> = b.faces.iter().flatten().map(|&x| x as i64).collect();
    d.set_item("faces", numpy::ndarray::Array2::from_shape_vec((b.faces.len(), 3), f).unwrap().into_pyarray(py))?;
    d.set_item("volume", b.volume)?;
    d.set_item("top", b.top)?;
    d.set_item("top_z", b.top_z)?;
    d.set_item("heights", b.heights.into_pyarray(py))?;
    d.set_item("areas", b.areas.into_pyarray(py))?;
    d.set_item("solidities", b.solidities.into_pyarray(py))?;
    d.set_item("open", b.open.into_pyarray(py))?;
    Ok(d)
}

// ---------------------------------------------------------------------- trees

fn xy_from_py(xy: PyReadonlyArray2<f64>) -> PyResult<Vec<[f64; 2]>> {
    let a = xy.as_array();
    if a.ncols() != 2 {
        return Err(PyValueError::new_err("xy must have shape (N, 2)"));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1]]).collect())
}

#[pyfunction]
fn fit_circle(xy: PyReadonlyArray2<f64>) -> PyResult<(f64, f64, f64, f64)> {
    trees::fit_circle(&xy_from_py(xy)?).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (xy, threshold=0.01, iterations=200, min_radius=0.02, max_radius=1.5, seed=0))]
fn fit_circle_ransac<'py>(py: Python<'py>, xy: PyReadonlyArray2<f64>, threshold: f64, iterations: usize, min_radius: f64, max_radius: f64, seed: u64) -> PyResult<(f64, f64, f64, Bound<'py, PyArray1<bool>>)> {
    let p = trees::RansacCircleParams { threshold, iterations, min_radius, max_radius, seed };
    let (cx, cy, r, inl) = trees::fit_circle_ransac(&xy_from_py(xy)?, &p).map_err(err)?;
    Ok((cx, cy, r, inl.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, slice_min=1.0, slice_max=5.0, slice_thickness=0.3, slice_step=0.25, reference_height=1.3, min_radius=0.015, max_radius=0.75, cluster_cell=0.06, min_cluster_points=12, max_cluster_extent=2.0, ransac_iterations=120, ransac_tolerance=0.02, max_circles_per_cluster=3, min_circle_inliers=10, min_coverage=0.12, min_arc_deg=0.0, max_circle_rmse=0.02, link_radius=0.2, link_radius_ratio=0.45, min_slices=3, max_lean_deg=25.0, link_radius_abs=0.02, prefilter=true, prefilter_k=16, prefilter_max_nz=0.6, prefilter_max_variation=0.15, seed=0, ransac_block=0, ransac_presample=false, recluster_wide=true, cluster_grid_at_slice_min=false, band_top_inclusive=false, shared_rng=false, taper_weight_power=1.0, min_total_points=0))]
#[allow(clippy::too_many_arguments)]
fn detect_stems<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, slice_min: f64, slice_max: f64, slice_thickness: f64, slice_step: f64, reference_height: f64, min_radius: f64, max_radius: f64, cluster_cell: f64, min_cluster_points: usize, max_cluster_extent: f64, ransac_iterations: usize, ransac_tolerance: f64, max_circles_per_cluster: usize, min_circle_inliers: usize, min_coverage: f64, min_arc_deg: f64, max_circle_rmse: f64, link_radius: f64, link_radius_ratio: f64, min_slices: usize, max_lean_deg: f64, link_radius_abs: f64, prefilter: bool, prefilter_k: usize, prefilter_max_nz: f64, prefilter_max_variation: f64, seed: u64, ransac_block: usize, ransac_presample: bool, recluster_wide: bool, cluster_grid_at_slice_min: bool, band_top_inclusive: bool, shared_rng: bool, taper_weight_power: f64, min_total_points: usize) -> PyResult<Bound<'py, PyList>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    if h.len() != p.len() {
        return Err(PyValueError::new_err("heights must have one value per point"));
    }
    let params = sylva_rs::stems::StemParams { slice_min, slice_max, slice_thickness, slice_step, reference_height, min_radius, max_radius, cluster_cell, min_cluster_points, max_cluster_extent, ransac_iterations, ransac_tolerance, max_circles_per_cluster, min_circle_inliers, min_coverage, min_arc_deg, max_circle_rmse, link_radius, link_radius_ratio, min_slices, max_lean_deg, link_radius_abs, prefilter, prefilter_k, prefilter_max_nz, prefilter_max_variation, seed, ransac_block, ransac_presample, recluster_wide, cluster_grid_at_slice_min, band_top_inclusive, shared_rng, taper_weight_power, min_total_points };
    let found = py.detach(|| sylva_rs::stems::detect_stems_full(&p, &h, &params));
    let list = PyList::empty(py);
    for s in &found {
        let d = tree_to_py(py, &s.tree)?;
        // Height above ground of the reported position (the reference height).
        d.set_item("z_ref", s.z)?;
        d.set_item("axis", s.axis.to_vec())?;
        d.set_item("coverage", s.coverage)?;
        list.append(d)?;
    }
    Ok(list)
}

/// The detector settings used for coregistration stem maps, as keyword
/// arguments of `detect_stems`.
#[pyfunction]
fn stems_coreg_defaults<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let p = sylva_rs::stems::StemParams::coreg();
    let d = PyDict::new(py);
    d.set_item("slice_min", p.slice_min)?;
    d.set_item("slice_max", p.slice_max)?;
    d.set_item("slice_thickness", p.slice_thickness)?;
    d.set_item("slice_step", p.slice_step)?;
    d.set_item("reference_height", p.reference_height)?;
    d.set_item("min_radius", p.min_radius)?;
    d.set_item("max_radius", p.max_radius)?;
    d.set_item("cluster_cell", p.cluster_cell)?;
    d.set_item("min_cluster_points", p.min_cluster_points)?;
    d.set_item("max_cluster_extent", p.max_cluster_extent)?;
    d.set_item("ransac_iterations", p.ransac_iterations)?;
    d.set_item("ransac_tolerance", p.ransac_tolerance)?;
    d.set_item("max_circles_per_cluster", p.max_circles_per_cluster)?;
    d.set_item("min_circle_inliers", p.min_circle_inliers)?;
    d.set_item("min_coverage", p.min_coverage)?;
    d.set_item("min_arc_deg", p.min_arc_deg)?;
    d.set_item("max_circle_rmse", p.max_circle_rmse)?;
    d.set_item("link_radius", p.link_radius)?;
    d.set_item("link_radius_ratio", p.link_radius_ratio)?;
    d.set_item("min_slices", p.min_slices)?;
    d.set_item("max_lean_deg", p.max_lean_deg)?;
    d.set_item("link_radius_abs", p.link_radius_abs)?;
    d.set_item("prefilter", p.prefilter)?;
    d.set_item("prefilter_k", p.prefilter_k)?;
    d.set_item("prefilter_max_nz", p.prefilter_max_nz)?;
    d.set_item("prefilter_max_variation", p.prefilter_max_variation)?;
    d.set_item("seed", p.seed)?;
    d.set_item("ransac_block", p.ransac_block)?;
    d.set_item("ransac_presample", p.ransac_presample)?;
    d.set_item("recluster_wide", p.recluster_wide)?;
    d.set_item("cluster_grid_at_slice_min", p.cluster_grid_at_slice_min)?;
    d.set_item("band_top_inclusive", p.band_top_inclusive)?;
    d.set_item("shared_rng", p.shared_rng)?;
    d.set_item("taper_weight_power", p.taper_weight_power)?;
    d.set_item("min_total_points", p.min_total_points)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, trees_list, slab_heights, slab_half=0.1, trunk_scale=1.5, trunk_min=0.15, prefilter=true))]
#[allow(clippy::too_many_arguments)]
fn stem_root_support<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, trees_list: &Bound<'_, PyList>, slab_heights: PyReadonlyArray1<f64>, slab_half: f64, trunk_scale: f64, trunk_min: f64, prefilter: bool) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let t = trees_from_py(trees_list)?;
    let slabs = slab_heights.as_array().to_vec();
    let params = sylva_rs::stems::StemParams { prefilter, ..Default::default() };
    let out = py.detach(|| sylva_rs::stems::stem_root_support(&p, &h, &t, &params, &slabs, slab_half, trunk_scale, trunk_min));
    let flat: Vec<i64> = out.iter().flat_map(|r| r.iter().map(|&v| v as i64)).collect();
    Ok(PyArray1::from_vec(py, flat).reshape([t.len(), slabs.len()])?)
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, cx, cy, at_heights, slice_thickness=0.1, search_radius=0.75))]
#[allow(clippy::too_many_arguments)]
fn dbh_profile<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, cx: f64, cy: f64, at_heights: PyReadonlyArray1<f64>, slice_thickness: f64, search_radius: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let at = at_heights.as_array().to_vec();
    Ok(py.detach(|| trees::dbh_profile(&p, &h, cx, cy, &at, slice_thickness, search_radius)).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, trees_list, k=6, max_edge=1.0, voxel_size=0.03, seed_height=1.5, seed_radius=0.25, seed_ring=true, power=6.0, angle_penalty=true, gravity=0.0, cut_above_ground=0.25, height_prior=true, height_prior_radius=1.5, height_prior_power=1.0, low_height=0.5, low_radius=1.0, wood_costs=false, wood_k=20, wood_threshold=0.9, understorey_height=10.0, understorey_band=0.5))]
#[allow(clippy::too_many_arguments)]
fn segment_trees<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, trees_list: &Bound<'_, PyList>, k: usize, max_edge: f64, voxel_size: f64, seed_height: f64, seed_radius: f64, seed_ring: bool, power: f64, angle_penalty: bool, gravity: f64, cut_above_ground: f64, height_prior: bool, height_prior_radius: f64, height_prior_power: f64, low_height: f64, low_radius: f64, wood_costs: bool, wood_k: usize, wood_threshold: f64, understorey_height: f64, understorey_band: f64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let t = trees_from_py(trees_list)?;
    let params = trees::SegmentParams { k, max_edge, voxel_size, seed_height, seed_radius, seed_ring, power, angle_penalty, gravity, cut_above_ground, height_prior, height_prior_radius, height_prior_power, low_height, low_radius, wood_costs, wood_k, wood_threshold, understorey_height, understorey_band };
    Ok(py.detach(|| trees::segment_trees(&p, &h, &t, &params)).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, trees_list, k=10, max_edge=1.0, voxel_size=0.1, seed_height=1.5, seed_radius=0.5, seed_ring=true, power=3.0, angle_penalty=true, cut_above_ground=0.25, ground_height=0.5, trunk_scale=1.5, trunk_min=0.15, search_radius=6.0))]
#[allow(clippy::too_many_arguments)]
fn merge_branches<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, trees_list: &Bound<'_, PyList>, k: usize, max_edge: f64, voxel_size: f64, seed_height: f64, seed_radius: f64, seed_ring: bool, power: f64, angle_penalty: bool, cut_above_ground: f64, ground_height: f64, trunk_scale: f64, trunk_min: f64, search_radius: f64) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyArray1<i64>>)> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let t = trees_from_py(trees_list)?;
    let params = trees::SegmentParams { k, max_edge, voxel_size, seed_height, seed_radius, seed_ring, power, angle_penalty, gravity: 0.0, cut_above_ground, height_prior: false, height_prior_radius: 1.5, low_height: 0.5, low_radius: 1.0, wood_costs: false, wood_k: 20, wood_threshold: 0.9, ..Default::default() };
    let (kept, merged) = py.detach(|| trees::merge_branches(&p, &h, &t, &params, ground_height, trunk_scale, trunk_min, search_radius));
    let list = PyList::empty(py);
    for tr in &kept {
        list.append(tree_to_py(py, tr)?)?;
    }
    Ok((list, merged.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (heights, labels, trees_list, percentile=100.0))]
fn tree_heights<'py>(py: Python<'py>, heights: PyReadonlyArray1<f64>, labels: PyReadonlyArray1<i64>, trees_list: &Bound<'_, PyList>, percentile: f64) -> PyResult<Bound<'py, PyList>> {
    let mut t = trees_from_py(trees_list)?;
    trees::tree_heights(&heights.as_array().to_vec(), &labels.as_array().to_vec(), &mut t, percentile);
    let list = PyList::empty(py);
    for tr in &t {
        list.append(tree_to_py(py, tr)?)?;
    }
    Ok(list)
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, labels, tree_id, crown_base_fraction=0.1))]
fn crown_metrics<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, labels: PyReadonlyArray1<i64>, tree_id: i64, crown_base_fraction: f64) -> PyResult<Option<Bound<'py, PyDict>>> {
    let p = xyz_from_py(xyz)?;
    let m = trees::crown_metrics(&p, &heights.as_array().to_vec(), &labels.as_array().to_vec(), tree_id, crown_base_fraction);
    match m {
        None => Ok(None),
        Some([area, base, depth, diam]) => {
            let d = PyDict::new(py);
            d.set_item("crown_area", area)?;
            d.set_item("crown_base_height", base)?;
            d.set_item("crown_depth", depth)?;
            d.set_item("crown_diameter", diam)?;
            Ok(Some(d))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, labels, crown_base_fraction=0.1))]
fn crown_metrics_all<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, labels: PyReadonlyArray1<i64>, crown_base_fraction: f64) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let l = labels.as_array().to_vec();
    let all = py.detach(|| trees::crown_metrics_all(&p, &h, &l, crown_base_fraction));
    let out = PyDict::new(py);
    for (id, [area, base, depth, diam]) in all {
        let d = PyDict::new(py);
        d.set_item("crown_area", area)?;
        d.set_item("crown_base_height", base)?;
        d.set_item("crown_depth", depth)?;
        d.set_item("crown_diameter", diam)?;
        out.set_item(id, d)?;
    }
    Ok(out)
}

#[pyfunction]
fn convex_hull_area(xy: PyReadonlyArray2<f64>) -> PyResult<f64> {
    Ok(trees::convex_hull_area(&xy_from_py(xy)?))
}

// ------------------------------------------------------------------------ qsm

#[pyfunction]
#[pyo3(signature = (xyz, axis_init=None))]
fn fit_cylinder<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, axis_init: Option<(f64, f64, f64)>) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let f = qsm::fit_cylinder(&p, axis_init.map(|a| [a.0, a.1, a.2])).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("point", f.point.to_vec().into_pyarray(py))?;
    d.set_item("axis", f.axis.to_vec().into_pyarray(py))?;
    d.set_item("radius", f.radius)?;
    d.set_item("rmse", f.rmse)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (xyz, threshold=0.02, iterations=100, sample_size=12, seed=0))]
fn fit_cylinder_ransac<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, threshold: f64, iterations: usize, sample_size: usize, seed: u64) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let (f, inl) = qsm::fit_cylinder_ransac(&p, threshold, iterations, sample_size, seed).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("point", f.point.to_vec().into_pyarray(py))?;
    d.set_item("axis", f.axis.to_vec().into_pyarray(py))?;
    d.set_item("radius", f.radius)?;
    d.set_item("rmse", f.rmse)?;
    d.set_item("inliers", inl.into_pyarray(py))?;
    Ok(d)
}

fn qsm_to_py<'py>(py: Python<'py>, q: &qsm::Qsm) -> PyResult<Bound<'py, PyDict>> {
    let rows = q.to_rows();
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().cloned()).collect();
    let d = PyDict::new(py);
    d.set_item("cylinders", PyArray1::from_vec(py, flat).reshape([rows.len(), 12])?)?;
    d.set_item("total_volume", q.total_volume())?;
    d.set_item("stem_volume", q.stem_volume())?;
    d.set_item("branch_volume", q.branch_volume())?;
    d.set_item("total_length", q.total_length())?;
    d.set_item("max_branch_order", q.max_branch_order())?;
    d.set_item("dbh", q.dbh())?;
    Ok(d)
}

fn qsm_from_rows(rows: PyReadonlyArray2<f64>) -> PyResult<qsm::Qsm> {
    let a = rows.as_array();
    if a.ncols() != 12 {
        return Err(PyValueError::new_err("cylinder array must have 12 columns"));
    }
    let q = qsm::Qsm {
        cylinders: a
            .rows()
            .into_iter()
            .map(|r| qsm::Cylinder {
                start: [r[0], r[1], r[2]],
                axis: [r[3], r[4], r[5]],
                length: r[6],
                radius: r[7],
                parent: r[8] as i64,
                branch_order: r[9] as u32,
                branch_id: r[10] as u32,
                n_points: r[11] as usize,
            })
            .collect(),
    };
    Ok(q)
}

#[pyfunction]
#[pyo3(signature = (xyz, base_xy=None, k=15, max_edge=1.0, bin_length=0.1, min_points=1, ransac_threshold=0.02, max_radius=1.0, taper_limit=1.1, max_rmse=0.03, smooth_steps=10, apex_radius=0.0025, min_arc_deg=90.0, min_inlier_fraction=0.05, prune_points=5, fit_min_points=50, crop_length=0.0, butt_height=0.6, relative_tolerance=0.08, base_radius=0.0, allometry_tolerance=0.3, stem_radius_cap=0.0, buttress_equivalent_area=true, buttress_max_inlier_fraction=0.3, pipe_slack=1.2, branch_min_inlier_fraction=0.3, spacing_scale=1.5, radius_power=0.0, power_above_spacing=0.025, sensor_noise=0.02, cluster_eps=0.1, centre_fit_points=100, radius_smooth_steps=15, butt_swell=1.1, butt_vertical_run=4, butt_max_lean_deg=50.0, chain_max_d=0.1, fourier_min_radius=0.15, min_weight=0.0, min_mean_weight=0.5, weights=None))]
#[allow(clippy::too_many_arguments)]
fn build_qsm<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, base_xy: Option<(f64, f64)>, k: usize, max_edge: f64, bin_length: f64, min_points: usize, ransac_threshold: f64, max_radius: f64, taper_limit: f64, max_rmse: f64, smooth_steps: usize, apex_radius: f64, min_arc_deg: f64, min_inlier_fraction: f64, prune_points: usize, fit_min_points: usize, crop_length: f64, butt_height: f64, relative_tolerance: f64, base_radius: f64, allometry_tolerance: f64, stem_radius_cap: f64, buttress_equivalent_area: bool, buttress_max_inlier_fraction: f64, pipe_slack: f64, branch_min_inlier_fraction: f64, spacing_scale: f64, radius_power: f64, power_above_spacing: f64, sensor_noise: f64, cluster_eps: f64, centre_fit_points: usize, radius_smooth_steps: usize, butt_swell: f64, butt_vertical_run: usize, butt_max_lean_deg: f64, chain_max_d: f64, fourier_min_radius: f64, min_weight: f64, min_mean_weight: f64, weights: Option<PyReadonlyArray1<f64>>) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let params = qsm::QsmParams { k, max_edge, bin_length, min_points, ransac_threshold, max_radius, taper_limit, max_rmse, smooth_steps, apex_radius, min_arc_deg, min_inlier_fraction, prune_points, fit_min_points, crop_length, butt_height, relative_tolerance, base_radius, allometry_tolerance, stem_radius_cap, buttress_equivalent_area, buttress_max_inlier_fraction, pipe_slack, branch_min_inlier_fraction, spacing_scale, radius_power, power_above_spacing, sensor_noise, cluster_eps, centre_fit_points, radius_smooth_steps, butt_swell, butt_vertical_run, butt_max_lean_deg, chain_max_d, fourier_min_radius, min_weight, min_mean_weight };
    let weights = weights.map(|w| w.as_array().to_vec());
    let base_xy = base_xy.map(|b| [b.0, b.1]);
    let q = py.detach(|| match &weights { Some(w) => qsm::build_qsm_weighted(&p, base_xy, &params, w), None => qsm::build_qsm(&p, base_xy, &params) }).map_err(err)?;
    qsm_to_py(py, &q)
}

#[pyfunction]
#[pyo3(signature = (xyz, base_xy=None, k=15, max_edge=1.0, bin_length=0.1))]
fn skeletonize<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, base_xy: Option<(f64, f64)>, k: usize, max_edge: f64, bin_length: f64) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let params = qsm::QsmParams { k, max_edge, bin_length, ..Default::default() };
    let s = py.detach(|| qsm::skeletonize(&p, base_xy.map(|b| [b.0, b.1]), &params)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("segment_id", s.segment_id.into_pyarray(py))?;
    d.set_item("geodesic", s.geodesic.into_pyarray(py))?;
    d.set_item("centres", xyz_to_py(py, &s.centres))?;
    let edges: Vec<i64> = s.edges.iter().flat_map(|(c, p)| [*c as i64, *p as i64]).collect();
    d.set_item("edges", PyArray1::from_vec(py, edges).reshape([s.edges.len(), 2])?)?;
    Ok(d)
}


fn crown_shape_to_py<'py>(py: Python<'py>, c: &trees::CrownShape) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("projected_area", c.projected_area)?;
    d.set_item("diameter", c.diameter)?;
    d.set_item("max_width", c.max_width)?;
    d.set_item("volume", c.volume)?;
    d.set_item("surface", c.surface)?;
    d.set_item("base_height", c.base_height)?;
    d.set_item("top_height", c.top_height)?;
    d.set_item("offset", c.offset)?;
    d.set_item("offset_direction", c.offset_direction)?;
    d.set_item("asymmetry", c.asymmetry)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (xyz, base_xy=None, z_min=f64::NEG_INFINITY, slice=0.5))]
fn crown_shape<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, base_xy: Option<(f64, f64)>, z_min: f64, slice: f64) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    if !(slice > 0.0) {
        return Err(PyValueError::new_err("slice must be positive"));
    }
    let c = py.detach(|| trees::crown_shape(&p, base_xy.map(|(x, y)| [x, y]), z_min, slice));
    crown_shape_to_py(py, &c)
}

#[pyfunction]
#[pyo3(signature = (cylinders, crown_branch_length=1.0, crown_slice=0.5))]
fn qsm_metrics<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>, crown_branch_length: f64, crown_slice: f64) -> PyResult<Bound<'py, PyDict>> {
    let q = qsm_from_rows(cylinders)?;
    q.check().map_err(err)?;
    let m = qsm::metrics::tree_metrics(&q, crown_branch_length, crown_slice.max(1e-3));
    let d = PyDict::new(py);
    d.set_item("height", m.height)?;
    d.set_item("dbh", m.dbh)?;
    d.set_item("total_volume", m.total_volume)?;
    d.set_item("stem_volume", m.stem_volume)?;
    d.set_item("branch_volume", m.branch_volume)?;
    d.set_item("total_length", m.total_length)?;
    d.set_item("stem_length", m.stem_length)?;
    d.set_item("max_order", m.max_order)?;
    d.set_item("n_branches_by_order", m.n_branches_by_order.clone())?;
    d.set_item("length_by_order", m.length_by_order.clone())?;
    d.set_item("volume_by_order", m.volume_by_order.clone())?;
    d.set_item("n_tips", m.n_tips)?;
    d.set_item("path_fraction", m.path_fraction)?;
    d.set_item("crown_base_height", m.crown_base_height)?;
    d.set_item("lean", m.lean)?;
    d.set_item("lean_direction", m.lean_direction)?;
    d.set_item("sweep", m.sweep)?;
    d.set_item("taper_heights", m.taper_heights.clone().into_pyarray(py))?;
    d.set_item("taper_radii", m.taper_radii.clone().into_pyarray(py))?;
    d.set_item("crown", crown_shape_to_py(py, &m.crown)?)?;
    d.set_item("measured_volume_fraction", m.measured_volume_fraction)?;
    d.set_item("measured_length_fraction", m.measured_length_fraction)?;
    d.set_item("median_insertion_angle", m.median_insertion_angle)?;
    d.set_item("median_branch_zenith", m.median_branch_zenith)?;
    Ok(d)
}

#[pyfunction]
fn qsm_branches<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyDict>> {
    let q = qsm_from_rows(cylinders)?;
    q.check().map_err(err)?;
    let b = qsm::metrics::branches(&q);
    let d = PyDict::new(py);
    macro_rules! col {
        ($name:literal, $f:expr) => {
            d.set_item($name, b.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    col!("id", |x| x.id as i64);
    col!("order", |x| x.order as i64);
    col!("parent", |x| x.parent);
    col!("n_cylinders", |x| x.n_cylinders as i64);
    col!("length", |x| x.length);
    col!("volume", |x| x.volume);
    col!("base_radius", |x| x.base_radius);
    col!("mean_radius", |x| x.mean_radius);
    col!("base_height", |x| x.base_height);
    col!("tip_height", |x| x.tip_height);
    col!("insertion_angle", |x| x.insertion_angle);
    col!("zenith", |x| x.zenith);
    col!("azimuth", |x| x.azimuth);
    col!("tortuosity", |x| x.tortuosity);
    col!("n_children", |x| x.n_children as i64);
    col!("measured_fraction", |x| x.measured_fraction);
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (shots, echo_heights, zenith_edges, n_azimuth, height_bin, n_heights, min_height=0.0, fired_per_ring=None))]
#[allow(clippy::too_many_arguments)]
fn pgap_histogram<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, echo_heights: PyReadonlyArray1<f64>, zenith_edges: Vec<f64>, n_azimuth: usize, height_bin: f64, n_heights: usize, min_height: f64, fired_per_ring: Option<Vec<f64>>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let s = shots_from_py(shots)?;
    let h = echo_heights.as_array().to_vec();
    if h.len() != s.echo_range.len() {
        return Err(PyValueError::new_err("echo_heights must have one value per echo"));
    }
    if zenith_edges.len() < 2 || zenith_edges.windows(2).any(|w| !(w[1] > w[0])) {
        return Err(PyValueError::new_err("zenith_edges must be increasing"));
    }
    if !(height_bin > 0.0) || n_heights == 0 {
        return Err(PyValueError::new_err("height_bin and n_heights must be positive"));
    }
    let mut hist = canopy::PgapHistogram::new(zenith_edges, n_azimuth, height_bin, n_heights);
    py.detach(|| hist.add(&s, &h, min_height, fired_per_ring.as_deref()));
    Ok((hist.hits.into_pyarray(py), hist.shots.into_pyarray(py)))
}


#[pyfunction]
#[pyo3(signature = (xyz, heights, stems, scan_ids=None, height_min=1.0, height_max=3.0, step=0.25, thickness=0.1, min_radius=0.05, max_radius=1.0, min_arc=270.0, min_inlier_fraction=0.5, cut_min=0.05, cut_fraction=0.3, min_scan_points=30, iterations=3))]
#[allow(clippy::too_many_arguments)]
fn stem_noise<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, stems: PyReadonlyArray2<f64>, scan_ids: Option<PyReadonlyArray1<i64>>, height_min: f64, height_max: f64, step: f64, thickness: f64, min_radius: f64, max_radius: f64, min_arc: f64, min_inlier_fraction: f64, cut_min: f64, cut_fraction: f64, min_scan_points: usize, iterations: usize) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let ids = scan_ids.map(|s| s.as_array().to_vec());
    if h.len() != p.len() || ids.as_ref().is_some_and(|s| s.len() != p.len()) {
        return Err(PyValueError::new_err("heights and scan_ids need one value per point"));
    }
    let st = stems.as_array();
    if st.ncols() < 2 {
        return Err(PyValueError::new_err("stems must be (n, 2): x, y"));
    }
    let stems: Vec<[f64; 2]> = st.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    let params = sylva_rs::quality::NoiseParams { height_min, height_max, step, thickness, min_radius, max_radius, min_arc, min_inlier_fraction, cut_min, cut_fraction, min_scan_points };
    let r = py.detach(|| sylva_rs::quality::stem_noise(&p, &h, ids.as_deref(), &stems, &params, iterations));
    let d = PyDict::new(py);
    let sl = PyDict::new(py);
    macro_rules! col {
        ($dict:expr, $src:expr, $name:literal, $f:expr) => {
            $dict.set_item($name, $src.iter().map($f).collect::<Vec<_>>().into_pyarray(py))?;
        };
    }
    col!(sl, r.slices, "stem", |x| x.stem as i64);
    col!(sl, r.slices, "height", |x| x.height);
    col!(sl, r.slices, "cx", |x| x.cx);
    col!(sl, r.slices, "cy", |x| x.cy);
    col!(sl, r.slices, "radius", |x| x.radius);
    col!(sl, r.slices, "n_points", |x| x.n_points as i64);
    col!(sl, r.slices, "sigma", |x| x.sigma);
    col!(sl, r.slices, "sigma_first", |x| x.sigma_first);
    col!(sl, r.slices, "arc", |x| x.arc);
    col!(sl, r.slices, "tail_fraction", |x| x.tail_fraction);
    let ss = PyDict::new(py);
    col!(ss, r.scan_slices, "scan", |x| x.scan);
    col!(ss, r.scan_slices, "slice", |x| x.slice as i64);
    col!(ss, r.scan_slices, "n_points", |x| x.n_points as i64);
    col!(ss, r.scan_slices, "median_residual", |x| x.median_residual);
    col!(ss, r.scan_slices, "sigma_within", |x| x.sigma_within);
    col!(ss, r.scan_slices, "sigma_local", |x| x.sigma_local);
    let sc = PyDict::new(py);
    col!(sc, r.scans, "scan", |x| x.scan);
    col!(sc, r.scans, "n_points", |x| x.n_points as i64);
    col!(sc, r.scans, "n_slices", |x| x.n_slices as i64);
    col!(sc, r.scans, "tx", |x| x.tx);
    col!(sc, r.scans, "ty", |x| x.ty);
    col!(sc, r.scans, "sigma_within", |x| x.sigma_within);
    col!(sc, r.scans, "sigma_local", |x| x.sigma_local);
    d.set_item("slices", sl)?;
    d.set_item("scan_slices", ss)?;
    d.set_item("scans", sc)?;
    d.set_item("residual", r.residual.into_pyarray(py))?;
    Ok(d)
}

#[pyfunction]
fn qsm_summary<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyDict>> {
    qsm_to_py(py, &qsm_from_rows(cylinders)?)
}

/// Memory the system says is free (bytes), or None where it will not say.
#[pyfunction]
fn memory_available() -> Option<u64> {
    sylva_rs::limits::available()
}

/// The most one allocation may ask for (bytes), or None when nothing is known.
#[pyfunction]
fn memory_budget() -> Option<u64> {
    sylva_rs::limits::budget()
}

/// Set that budget in bytes; 0 goes back to reading the system.
#[pyfunction]
fn set_memory_budget(bytes: u64) {
    sylva_rs::limits::set_budget(bytes);
}

/// Raise if `cells * per_cell` bytes would not fit, naming what and what to try.
#[pyfunction]
fn memory_check(cells: u128, per_cell: u64, what: &str, hint: &str) -> PyResult<()> {
    sylva_rs::limits::check_cells(cells, per_cell, what, hint).map_err(err)
}

/// The stages running now, as (label, done, total). Safe to call from
/// another thread while the work runs: the core holds the counts in atomics.
#[pyfunction]
fn progress_state() -> Vec<(String, u64, u64)> {
    sylva_rs::progress::state()
}

/// A stage opened by Python (a loop over trees, scans, files). The handle
/// closes it; dropping it takes it off the list.
#[pyclass(name = "ProgressTask")]
struct PyProgressTask {
    task: Option<sylva_rs::progress::Task>,
}

#[pymethods]
impl PyProgressTask {
    #[new]
    #[pyo3(signature = (label, total=0))]
    fn new(label: &str, total: u64) -> Self {
        PyProgressTask { task: Some(sylva_rs::progress::start(label, total)) }
    }

    #[pyo3(signature = (n=1))]
    fn inc(&self, n: u64) {
        if let Some(t) = &self.task {
            t.inc(n);
        }
    }

    fn set(&self, done: u64) {
        if let Some(t) = &self.task {
            t.set(done);
        }
    }

    fn set_total(&self, total: u64) {
        if let Some(t) = &self.task {
            t.set_total(total);
        }
    }

    /// End the stage.
    fn close(&mut self) {
        self.task = None;
    }
}

#[pyfunction]
#[pyo3(signature = (cylinders, sides=12, contiguous=false))]
fn qsm_mesh<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>, sides: usize, contiguous: bool) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<u32>>)> {
    let q = qsm_from_rows(cylinders)?;
    let (v, t, o) = if contiguous { q.mesh_contiguous(sides) } else { q.mesh(sides) };
    let flat_t: Vec<u32> = t.iter().flat_map(|f| f.iter().cloned()).collect();
    let nt = t.len();
    Ok((xyz_to_py(py, &v), PyArray1::from_vec(py, flat_t).reshape([nt, 3])?, o.into_pyarray(py)))
}

#[pyfunction]
fn qsm_write_csv(cylinders: PyReadonlyArray2<f64>, path: PathBuf) -> PyResult<()> {
    qsm_from_rows(cylinders)?.write_csv(path).map_err(err)
}

#[pyfunction]
fn qsm_write_treefile(cylinders: PyReadonlyArray2<f64>, path: PathBuf) -> PyResult<()> {
    qsm_from_rows(cylinders)?.write_treefile(path).map_err(err)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    canopy_py::register(m)?;
    quality_py::register(m)?;
    trees_py::register(m)?;
    filters_py::register(m)?;
    limits_py::register(m)?;
    raster_py::register(m)?;
    registration_py::register(m)?;
    shots_py::register(m)?;
    riscan_py::register(m)?;
    coreg_py::register(m)?;
    leaves_py::register(m)?;
    voxels_py::register(m)?;
    synthetic_py::register(m)?;
    qsm_py::register(m)?;
    coreg_pipeline_py::register(m)?;
    interpolate_py::register(m)?;
    masks_py::register(m)?;
    coords_py::register(m)?;
    als_py::register(m)?;
    als_metrics_py::register(m)?;
    als_canopy_py::register(m)?;
    als_trees_py::register(m)?;
    change_als_py::register(m)?;
    change_points_py::register(m)?;
    change_qsm_py::register(m)?;
    change_trees_py::register(m)?;
    waveform_py::register(m)?;
    fusion_py::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyProgressTask>()?;
    m.add_class::<PyCoregKdTree>()?;
    m.add_class::<PyCoregIcpTarget>()?;
    for f in [
        wrap_pyfunction!(read, m)?,
        wrap_pyfunction!(write, m)?,
        wrap_pyfunction!(read_ascii, m)?,
        wrap_pyfunction!(read_rxp, m)?,
        wrap_pyfunction!(read_rxp_shots, m)?,
        wrap_pyfunction!(find_rivlib, m)?,
        wrap_pyfunction!(read_matrix_file, m)?,
        wrap_pyfunction!(shots_from_pointcloud, m)?,
        wrap_pyfunction!(shots_from_ray_cloud, m)?,
        wrap_pyfunction!(shots_to_pointcloud, m)?,
        wrap_pyfunction!(shots_transform, m)?,
        wrap_pyfunction!(voxel_downsample_indices, m)?,
        wrap_pyfunction!(voxel_centroids, m)?,
        wrap_pyfunction!(random_indices, m)?,
        wrap_pyfunction!(min_distance_indices, m)?,
        wrap_pyfunction!(statistical_outlier_mask, m)?,
        wrap_pyfunction!(radius_outlier_mask, m)?,
        wrap_pyfunction!(estimate_normals, m)?,
        wrap_pyfunction!(planarity_linearity, m)?,
        wrap_pyfunction!(wood_mask, m)?,
        wrap_pyfunction!(crown_shape, m)?,
        wrap_pyfunction!(qsm_metrics, m)?,
        wrap_pyfunction!(qsm_branches, m)?,
        wrap_pyfunction!(pgap_histogram, m)?,
        wrap_pyfunction!(stem_noise, m)?,
        wrap_pyfunction!(euclidean_clusters, m)?,
        wrap_pyfunction!(knn, m)?,
        wrap_pyfunction!(count_within, m)?,
        wrap_pyfunction!(csf_ground_mask, m)?,
        wrap_pyfunction!(pmf_ground_mask, m)?,
        wrap_pyfunction!(make_dtm, m)?,
        wrap_pyfunction!(raster_sample, m)?,
        wrap_pyfunction!(raster_fill_nearest, m)?,
        wrap_pyfunction!(make_chm, m)?,
        wrap_pyfunction!(write_ascii_grid, m)?,
        wrap_pyfunction!(read_ascii_grid, m)?,
        wrap_pyfunction!(voxelize, m)?,
        wrap_pyfunction!(pad_profile_voxel, m)?,
        wrap_pyfunction!(gap_fraction_zenith, m)?,
        wrap_pyfunction!(lai_from_gap_fraction, m)?,
        wrap_pyfunction!(canopy_cover, m)?,
        wrap_pyfunction!(density_grid, m)?,
        wrap_pyfunction!(write_shots, m)?,
        wrap_pyfunction!(read_shots, m)?,
        wrap_pyfunction!(shots_info, m)?,
        wrap_pyfunction!(kabsch, m)?,
        wrap_pyfunction!(icp, m)?,
        wrap_pyfunction!(match_stem_maps, m)?,
        wrap_pyfunction!(coreg_fit_ground, m)?,
        wrap_pyfunction!(coreg_ground_height, m)?,
        wrap_pyfunction!(coreg_ground_support, m)?,
        wrap_pyfunction!(coreg_voxel_centroids, m)?,
        wrap_pyfunction!(coreg_local_pca, m)?,
        wrap_pyfunction!(coreg_planar_filter, m)?,
        wrap_pyfunction!(coreg_icp, m)?,
        wrap_pyfunction!(coreg_evaluate, m)?,
        wrap_pyfunction!(coreg_plane_information, m)?,
        wrap_pyfunction!(buttress_mesh, m)?,
        wrap_pyfunction!(fit_circle, m)?,
        wrap_pyfunction!(fit_circle_ransac, m)?,
        wrap_pyfunction!(detect_stems, m)?,
        wrap_pyfunction!(stems_coreg_defaults, m)?,
        wrap_pyfunction!(dbh_profile, m)?,
        wrap_pyfunction!(segment_trees, m)?,
        wrap_pyfunction!(merge_branches, m)?,
        wrap_pyfunction!(stem_root_support, m)?,
        wrap_pyfunction!(tree_heights, m)?,
        wrap_pyfunction!(crown_metrics, m)?,
        wrap_pyfunction!(crown_metrics_all, m)?,
        wrap_pyfunction!(convex_hull_area, m)?,
        wrap_pyfunction!(fit_cylinder, m)?,
        wrap_pyfunction!(fit_cylinder_ransac, m)?,
        wrap_pyfunction!(build_qsm, m)?,
        wrap_pyfunction!(skeletonize, m)?,
        wrap_pyfunction!(qsm_summary, m)?,
        wrap_pyfunction!(qsm_write_csv, m)?,
        wrap_pyfunction!(qsm_mesh, m)?,
        wrap_pyfunction!(progress_state, m)?,
        wrap_pyfunction!(memory_available, m)?,
        wrap_pyfunction!(memory_budget, m)?,
        wrap_pyfunction!(set_memory_budget, m)?,
        wrap_pyfunction!(memory_check, m)?,
        wrap_pyfunction!(qsm_write_treefile, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}

#[allow(dead_code)]
fn _touch(a: &Bound<'_, PyArray3<f64>>) -> usize {
    a.len()
}
