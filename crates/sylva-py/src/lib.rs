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
use sylva_rs::{canopy, cluster, filters, ground, io, qsm, registration, trees, voxel, Point, PointCloud, Raster, Shots, Transform};

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
#[pyo3(signature = (path, xyz, attrs=None, point_format=6, scale=0.001, binary=true))]
fn write(py: Python<'_>, path: PathBuf, xyz: PyReadonlyArray2<f64>, attrs: Option<&Bound<'_, PyDict>>, point_format: u8, scale: f64, binary: bool) -> PyResult<()> {
    let c = cloud_from_py(xyz, attrs)?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    py.detach(|| match ext.as_str() {
        "las" | "laz" => io::las::write_las(&c, &path, &io::las::LasWriteOptions { point_format, scale }),
        "ply" => io::ply::write_ply(&c, &path, binary),
        _ => io::write(&c, &path),
    })
    .map_err(err)
}

#[pyfunction]
#[pyo3(signature = (path, columns=None))]
fn read_ascii<'py>(py: Python<'py>, path: PathBuf, columns: Option<Vec<String>>) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let c = py.detach(|| io::ascii::read_ascii(&path, columns.as_deref())).map_err(err)?;
    cloud_to_py(py, &c)
}

fn rxp_opts(library: Option<PathBuf>, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: usize, max_points: Option<usize>, echoes: String) -> io::riegl::RxpOptions {
    io::riegl::RxpOptions { library, drop_pseudo_echoes, min_range, max_range, stride: stride.max(1), max_points, echoes }
}

#[pyfunction]
#[pyo3(signature = (path, library=None, drop_pseudo_echoes=true, min_range=0.5, max_range=f64::INFINITY, stride=1, max_points=None, echoes="all".to_string()))]
#[allow(clippy::too_many_arguments)]
fn read_rxp<'py>(py: Python<'py>, path: PathBuf, library: Option<PathBuf>, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: usize, max_points: Option<usize>, echoes: String) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    let opts = rxp_opts(library, drop_pseudo_echoes, min_range, max_range, stride, max_points, echoes);
    let c = py.detach(|| io::riegl::read_rxp(&path, &opts)).map_err(err)?;
    cloud_to_py(py, &c)
}

#[pyfunction]
#[pyo3(signature = (path, library=None, drop_pseudo_echoes=true, min_range=0.5, max_range=f64::INFINITY, stride=1, max_points=None, echoes="all".to_string()))]
#[allow(clippy::too_many_arguments)]
fn read_rxp_shots<'py>(py: Python<'py>, path: PathBuf, library: Option<PathBuf>, drop_pseudo_echoes: bool, min_range: f64, max_range: f64, stride: usize, max_points: Option<usize>, echoes: String) -> PyResult<Bound<'py, PyDict>> {
    let opts = rxp_opts(library, drop_pseudo_echoes, min_range, max_range, stride, max_points, echoes);
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
#[pyo3(signature = (xyz, k=20, high_threshold=0.85, medium_threshold=0.75, graph_k=10, max_edge=1.0, base_height=0.25, target_res=0.2, min_passage=3, assign_dist=0.05, assign_scale=0.0, component_res=0.05, component_min=200, sor_k=50, sor_std=1.0, dilate_dist=0.03, passage=true))]
#[allow(clippy::too_many_arguments)]
fn wood_mask<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, k: usize, high_threshold: f64, medium_threshold: f64, graph_k: usize, max_edge: f64, base_height: f64, target_res: f64, min_passage: usize, assign_dist: f64, assign_scale: f64, component_res: f64, component_min: usize, sor_k: usize, sor_std: f64, dilate_dist: f64, passage: bool) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let p = xyz_from_py(xyz)?;
    let params = qsm::wood::WoodParams { k, high_threshold, medium_threshold, graph_k, max_edge, base_height, target_res, min_passage, assign_dist, assign_scale, component_res, component_min, sor_k, sor_std, dilate_dist, passage };
    Ok(py.detach(|| qsm::wood::wood_mask(&p, &params)).into_pyarray(py))
}

#[pyfunction]
fn euclidean_clusters<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, radius: f64, min_points: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let p = xyz_from_py(xyz)?;
    Ok(py.detach(|| cluster::euclidean_clusters(&p, radius, min_points)).into_pyarray(py))
}

#[pyfunction]
fn knn<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, queries: PyReadonlyArray2<f64>, k: usize) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)> {
    let p = xyz_from_py(xyz)?;
    let q = xyz_from_py(queries)?;
    let (d, i): (Vec<f64>, Vec<i64>) = py.detach(|| {
        use rayon::prelude::*;
        let tree = sylva_rs::spatial::KdTree::new(&p);
        let rows: Vec<Vec<(usize, f64)>> = q.par_iter().map(|x| tree.knn(x, k)).collect();
        let mut d = Vec::with_capacity(q.len() * k);
        let mut i = Vec::with_capacity(q.len() * k);
        for r in rows {
            for j in 0..k {
                match r.get(j) {
                    Some(&(idx, dist)) => {
                        d.push(dist);
                        i.push(idx as i64);
                    }
                    None => {
                        d.push(f64::INFINITY);
                        i.push(-1);
                    }
                }
            }
        }
        (d, i)
    });
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

// --------------------------------------------------------------- ray voxels

/// Ray-traced voxel statistics. The grid stays on the Rust side; arrays are
/// copied out on request, shaped `(nz, ny, nx)`.
#[pyclass(name = "RayVoxels")]
struct PyRayVoxels {
    inner: voxel::RayVoxels,
}

#[pymethods]
impl PyRayVoxels {
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.origin.to_vec().into_pyarray(py)
    }

    #[getter]
    fn voxel_size(&self) -> f64 {
        self.inner.voxel_size
    }

    /// `(nx, ny, nz)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.inner.shape;
        (s[0], s[1], s[2])
    }

    #[getter]
    fn has_leaf(&self) -> bool {
        self.inner.has_leaf
    }

    #[getter]
    fn has_wood(&self) -> bool {
        self.inner.has_wood
    }

    /// Names accepted by `field`.
    fn field_names(&self) -> Vec<&'static str> {
        let v = &self.inner;
        let mut names: Vec<&'static str> = voxel::I::ALL.iter().map(|f| f.name()).collect();
        names.extend(voxel::F::ALL.iter().filter(|f| !v.f[**f as usize].is_empty()).map(|f| f.name()));
        for (name, on) in [("ppl_lambda", v.ppl_lambda.is_some()), ("wood_volume", v.wood_volume.is_some()), ("predominant_tree", v.predominant_tree.is_some()), ("subvoxel_counts", v.subvoxel_counts.is_some()), ("ground_height", v.ground_height.is_some())] {
            if on {
                names.push(name);
            }
        }
        names
    }

    /// A raw accumulator as an array.
    fn field<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let v = &self.inner;
        let shp = [v.shape[2], v.shape[1], v.shape[0]];
        let missing = || PyValueError::new_err(format!("no voxel field {name:?} (see field_names())"));
        if let Some(f) = voxel::I::ALL.iter().find(|f| f.name() == name) {
            return Ok(PyArray1::from_vec(py, v.i[*f as usize].clone()).reshape(shp)?.into_any());
        }
        if let Some(f) = voxel::F::ALL.iter().find(|f| f.name() == name) {
            let data = &v.f[*f as usize];
            if data.is_empty() {
                return Err(missing());
            }
            return Ok(PyArray1::from_vec(py, data.clone()).reshape(shp)?.into_any());
        }
        match name {
            "ppl_lambda" => Ok(PyArray1::from_vec(py, v.ppl_lambda.clone().ok_or_else(missing)?).reshape(shp)?.into_any()),
            "wood_volume" => Ok(PyArray1::from_vec(py, v.wood_volume.clone().ok_or_else(missing)?).reshape(shp)?.into_any()),
            "predominant_tree" => Ok(PyArray1::from_vec(py, v.predominant_tree.clone().ok_or_else(missing)?).reshape(shp)?.into_any()),
            "ground_height" => Ok(PyArray1::from_vec(py, v.ground_height.clone().ok_or_else(missing)?).reshape([shp[1], shp[2]])?.into_any()),
            "subvoxel_counts" => {
                let n_sub = v.params.subvoxel_split.pow(3);
                Ok(PyArray1::from_vec(py, v.subvoxel_counts.clone().ok_or_else(missing)?).reshape([shp[0], shp[1], shp[2], n_sub])?.into_any())
            }
            _ => Err(missing()),
        }
    }

    /// A derived per-voxel quantity (`pad_fpl`, `attenuation_ppl`, `transmittance`, ...).
    fn metric<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let v = &self.inner;
        let data = py.detach(|| v.metric(name)).map_err(err)?;
        let shp = [v.shape[2], v.shape[1], v.shape[0]];
        if name == "state" {
            return Ok(PyArray1::from_vec(py, data.iter().map(|&s| s as u8).collect()).reshape(shp)?.into_any());
        }
        Ok(PyArray1::from_vec(py, data).reshape(shp)?.into_any())
    }

    fn metric_names(&self) -> Vec<&'static str> {
        voxel::RayVoxels::METRICS.to_vec()
    }

    /// Per-tree inclination distributions: `{tree_id: {...}}`.
    fn tree_iad<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (tid, t) in &self.inner.tree_iad {
            let d = PyDict::new(py);
            d.set_item("bin_centres", t.bin_centres.clone().into_pyarray(py))?;
            for (k, h) in [("liad", &t.liad), ("wiad", &t.wiad), ("piad", &t.piad), ("liad_bailey", &t.liad_bailey), ("wiad_bailey", &t.wiad_bailey), ("piad_bailey", &t.piad_bailey)] {
                d.set_item(k, h.clone().into_pyarray(py))?;
            }
            for (k, g) in [("g_leaf", t.g_leaf), ("g_wood", t.g_wood), ("g_plant", t.g_plant), ("bailey_g_leaf", t.bailey_g_leaf), ("bailey_g_wood", t.bailey_g_wood)] {
                d.set_item(k, g)?;
            }
            d.set_item("leaf_hits", t.leaf_hits)?;
            d.set_item("wood_hits", t.wood_hits)?;
            d.set_item("liad_de_wit", t.liad_de_wit)?;
            d.set_item("wiad_de_wit", t.wiad_de_wit)?;
            d.set_item("piad_de_wit", t.piad_de_wit)?;
            out.set_item(*tid, d)?;
        }
        Ok(out)
    }

    /// Rasterise QSM cylinders (12-column rows) into per-voxel woody volume.
    fn add_wood_volume(&mut self, py: Python<'_>, cylinders: PyReadonlyArray2<f64>) -> PyResult<()> {
        let q = qsm_from_rows(cylinders)?;
        let v = &mut self.inner;
        py.detach(|| v.add_wood_volume(&q.cylinders));
        Ok(())
    }

    /// Write `.vox` (AMAPVox) or `.txt`; returns the number of voxels written.
    #[pyo3(signature = (path, format="vox", include_unobserved=false, filled_only=false))]
    fn write(&self, py: Python<'_>, path: PathBuf, format: &str, include_unobserved: bool, filled_only: bool) -> PyResult<usize> {
        let opts = voxel::WriteOptions { include_unobserved, filled_only };
        let v = &self.inner;
        match format {
            "vox" => py.detach(|| v.write_vox(&path, opts)).map_err(err),
            "text" => py.detach(|| v.write_text(&path, opts)).map_err(err),
            other => Err(PyValueError::new_err(format!("unknown voxel format {other:?} (vox|text)"))),
        }
    }

    fn write_iad_csv(&self, path: PathBuf) -> PyResult<()> {
        self.inner.write_iad_csv(path).map_err(err)
    }
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

/// Voxelise a shots file one batch of row groups at a time.
#[pyfunction]
#[pyo3(signature = (path, voxel_size=0.1, bounds=None, dtm=None, class_attr="classification", ground_class=None, ground_distance=0.2, leaf_classes=vec![], wood_classes=vec![], tree_attr="tree_id", intensity_attr="intensity", weighting="equal", occlusion=false, flat_top=false, neighbour_prior_min_rays=0, beam=None, subvoxel_split=0, subvoxel_min_beams=10, average_leaf_area=0.005, lad="spherical", lad_params=vec![], attenuation=vec!["fpl".to_string()], inclination=false, n_iad_bins=18, knn_normal=10, triangle_lmax=0.05, unbounded_range=f64::INFINITY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ray_voxelize_file(py: Python<'_>, path: PathBuf, voxel_size: f64, bounds: Option<((f64, f64, f64), (f64, f64, f64))>, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, class_attr: &str, ground_class: Option<i64>, ground_distance: f64, leaf_classes: Vec<i64>, wood_classes: Vec<i64>, tree_attr: &str, intensity_attr: &str, weighting: &str, occlusion: bool, flat_top: bool, neighbour_prior_min_rays: u32, beam: Option<(f64, f64)>, subvoxel_split: usize, subvoxel_min_beams: u8, average_leaf_area: f64, lad: &str, lad_params: Vec<f64>, attenuation: Vec<String>, inclination: bool, n_iad_bins: usize, knn_normal: usize, triangle_lmax: f64, unbounded_range: f64) -> PyResult<PyRayVoxels> {
    let params = voxel::VoxelParams {
        voxel_size,
        bounds: bounds.map(|(lo, hi)| ([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2])),
        weighting: voxel::WeightMethod::parse(weighting).map_err(err)?,
        occlusion,
        flat_top,
        neighbour_prior_min_rays,
        beam: beam.map(|(diameter, divergence)| voxel::BeamSpec { diameter, divergence }),
        subvoxel_split,
        subvoxel_min_beams,
        average_leaf_area,
        lad: voxel::Lad::parse(lad, &lad_params).map_err(err)?,
        attenuation: attenuation.iter().map(|m| voxel::Attenuation::parse(m)).collect::<Result<_, _>>().map_err(err)?,
        inclination,
        n_iad_bins,
        knn_normal,
        triangle_lmax,
        unbounded_range,
    };
    let labels = voxel::EchoLabels { class_attr: class_attr.into(), ground_class, ground_distance, leaf_classes, wood_classes, tree_attr: tree_attr.into(), intensity_attr: intensity_attr.into() };
    let dtm = dtm.map(|(data, xmin, ymin, res)| raster_from_py(data, xmin, ymin, res));
    let inner = py.detach(|| voxel::voxelize_file(&io::shots::ShotsFile::open(&path)?, &params, &labels, dtm.as_ref())).map_err(err)?;
    Ok(PyRayVoxels { inner })
}

#[pyfunction]
#[pyo3(signature = (shots, voxel_size=0.1, bounds=None, ground=None, foliage=None, intensity=None, tree_id=None, dtm=None, weighting="equal", occlusion=false, flat_top=false, neighbour_prior_min_rays=0, beam=None, subvoxel_split=0, subvoxel_min_beams=10, average_leaf_area=0.005, lad="spherical", lad_params=vec![], attenuation=vec!["fpl".to_string()], inclination=false, n_iad_bins=18, knn_normal=10, triangle_lmax=0.05, unbounded_range=f64::INFINITY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ray_voxelize(py: Python<'_>, shots: &Bound<'_, PyDict>, voxel_size: f64, bounds: Option<((f64, f64, f64), (f64, f64, f64))>, ground: Option<PyReadonlyArray1<bool>>, foliage: Option<PyReadonlyArray1<u8>>, intensity: Option<PyReadonlyArray1<f64>>, tree_id: Option<PyReadonlyArray1<i32>>, dtm: Option<(PyReadonlyArray2<f64>, f64, f64, f64)>, weighting: &str, occlusion: bool, flat_top: bool, neighbour_prior_min_rays: u32, beam: Option<(f64, f64)>, subvoxel_split: usize, subvoxel_min_beams: u8, average_leaf_area: f64, lad: &str, lad_params: Vec<f64>, attenuation: Vec<String>, inclination: bool, n_iad_bins: usize, knn_normal: usize, triangle_lmax: f64, unbounded_range: f64) -> PyResult<PyRayVoxels> {
    let s = shots_from_py(shots)?;
    let params = voxel::VoxelParams {
        voxel_size,
        bounds: bounds.map(|(lo, hi)| ([lo.0, lo.1, lo.2], [hi.0, hi.1, hi.2])),
        weighting: voxel::WeightMethod::parse(weighting).map_err(err)?,
        occlusion,
        flat_top,
        neighbour_prior_min_rays,
        beam: beam.map(|(diameter, divergence)| voxel::BeamSpec { diameter, divergence }),
        subvoxel_split,
        subvoxel_min_beams,
        average_leaf_area,
        lad: voxel::Lad::parse(lad, &lad_params).map_err(err)?,
        attenuation: attenuation.iter().map(|m| voxel::Attenuation::parse(m)).collect::<Result<_, _>>().map_err(err)?,
        inclination,
        n_iad_bins,
        knn_normal,
        triangle_lmax,
        unbounded_range,
    };
    let ground = ground.map(|a| a.as_array().to_vec());
    let foliage = foliage.map(|a| a.as_array().to_vec());
    let intensity = intensity.map(|a| a.as_array().to_vec());
    let tree_id = tree_id.map(|a| a.as_array().to_vec());
    let dtm = dtm.map(|(data, xmin, ymin, res)| raster_from_py(data, xmin, ymin, res));
    let inner = py
        .detach(|| {
            let inputs = voxel::VoxelInputs { shots: &s, ground: ground.as_deref(), foliage: foliage.as_deref(), intensity: intensity.as_deref(), tree_id: tree_id.as_deref(), dtm: dtm.as_ref() };
            voxel::voxelize(&inputs, &params)
        })
        .map_err(err)?;
    Ok(PyRayVoxels { inner })
}

/// `(beam diameter at exit [m], divergence [rad])` of a scanner known to AMAPVox.
#[pyfunction]
fn laser_spec(name: &str) -> Option<(f64, f64)> {
    voxel::laser_spec(name)
}

/// Projection function G of an analytic leaf angle distribution at beam zenith `theta` (rad).
#[pyfunction]
#[pyo3(signature = (theta, lad="spherical", lad_params=vec![]))]
fn leaf_projection<'py>(py: Python<'py>, theta: PyReadonlyArray1<f64>, lad: &str, lad_params: Vec<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lad = voxel::Lad::parse(lad, &lad_params).map_err(err)?;
    Ok(theta.as_array().iter().map(|&t| voxel::compute_g(t, &lad)).collect::<Vec<_>>().into_pyarray(py))
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
#[pyo3(signature = (xyz, heights, slice_min=1.0, slice_max=5.0, slice_thickness=0.3, slice_step=0.25, reference_height=1.3, min_radius=0.015, max_radius=0.75, cluster_cell=0.06, min_cluster_points=12, max_cluster_extent=2.0, ransac_iterations=120, ransac_tolerance=0.02, max_circles_per_cluster=3, min_circle_inliers=10, min_coverage=0.12, min_arc_deg=0.0, max_circle_rmse=0.02, link_radius=0.2, link_radius_ratio=0.45, min_slices=3, max_lean_deg=25.0, link_radius_abs=0.02, prefilter=true, prefilter_k=16, prefilter_max_nz=0.6, prefilter_max_variation=0.15, seed=0))]
#[allow(clippy::too_many_arguments)]
fn detect_stems<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, slice_min: f64, slice_max: f64, slice_thickness: f64, slice_step: f64, reference_height: f64, min_radius: f64, max_radius: f64, cluster_cell: f64, min_cluster_points: usize, max_cluster_extent: f64, ransac_iterations: usize, ransac_tolerance: f64, max_circles_per_cluster: usize, min_circle_inliers: usize, min_coverage: f64, min_arc_deg: f64, max_circle_rmse: f64, link_radius: f64, link_radius_ratio: f64, min_slices: usize, max_lean_deg: f64, link_radius_abs: f64, prefilter: bool, prefilter_k: usize, prefilter_max_nz: f64, prefilter_max_variation: f64, seed: u64) -> PyResult<Bound<'py, PyList>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let params = sylva_rs::stems::StemParams { slice_min, slice_max, slice_thickness, slice_step, reference_height, min_radius, max_radius, cluster_cell, min_cluster_points, max_cluster_extent, ransac_iterations, ransac_tolerance, max_circles_per_cluster, min_circle_inliers, min_coverage, min_arc_deg, max_circle_rmse, link_radius, link_radius_ratio, min_slices, max_lean_deg, link_radius_abs, prefilter, prefilter_k, prefilter_max_nz, prefilter_max_variation, seed };
    let found = py.detach(|| trees::detect_stems(&p, &h, &params));
    let list = PyList::empty(py);
    for t in &found {
        list.append(tree_to_py(py, t)?)?;
    }
    Ok(list)
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
#[pyo3(signature = (xyz, heights, trees_list, k=10, max_edge=1.0, voxel_size=0.05, seed_height=1.5, seed_radius=0.5, power=3.0, angle_penalty=true, gravity=0.0, cut_above_ground=0.25, height_prior=true, height_prior_radius=1.5, low_height=0.5, low_radius=1.0, wood_costs=false, wood_k=20, wood_threshold=0.9))]
#[allow(clippy::too_many_arguments)]
fn segment_trees<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, trees_list: &Bound<'_, PyList>, k: usize, max_edge: f64, voxel_size: f64, seed_height: f64, seed_radius: f64, power: f64, angle_penalty: bool, gravity: f64, cut_above_ground: f64, height_prior: bool, height_prior_radius: f64, low_height: f64, low_radius: f64, wood_costs: bool, wood_k: usize, wood_threshold: f64) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let t = trees_from_py(trees_list)?;
    let params = trees::SegmentParams { k, max_edge, voxel_size, seed_height, seed_radius, power, angle_penalty, gravity, cut_above_ground, height_prior, height_prior_radius, low_height, low_radius, wood_costs, wood_k, wood_threshold };
    Ok(py.detach(|| trees::segment_trees(&p, &h, &t, &params)).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (xyz, heights, trees_list, k=10, max_edge=1.0, voxel_size=0.1, seed_height=1.5, seed_radius=0.5, power=3.0, angle_penalty=true, cut_above_ground=0.25, ground_height=0.5, trunk_scale=1.5, trunk_min=0.15, search_radius=6.0))]
#[allow(clippy::too_many_arguments)]
fn merge_branches<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, heights: PyReadonlyArray1<f64>, trees_list: &Bound<'_, PyList>, k: usize, max_edge: f64, voxel_size: f64, seed_height: f64, seed_radius: f64, power: f64, angle_penalty: bool, cut_above_ground: f64, ground_height: f64, trunk_scale: f64, trunk_min: f64, search_radius: f64) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyArray1<i64>>)> {
    let p = xyz_from_py(xyz)?;
    let h = heights.as_array().to_vec();
    let t = trees_from_py(trees_list)?;
    let params = trees::SegmentParams { k, max_edge, voxel_size, seed_height, seed_radius, power, angle_penalty, gravity: 0.0, cut_above_ground, height_prior: false, height_prior_radius: 1.5, low_height: 0.5, low_radius: 1.0, wood_costs: false, wood_k: 20, wood_threshold: 0.9 };
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
    Ok(qsm::Qsm {
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
    })
}

#[pyfunction]
#[pyo3(signature = (xyz, base_xy=None, k=15, max_edge=1.0, bin_length=0.1, min_points=1, ransac_threshold=0.02, max_radius=1.0, taper_limit=1.1, max_rmse=0.03, smooth_steps=10, apex_radius=0.0025, min_arc_deg=90.0, min_inlier_fraction=0.05, prune_points=5, fit_min_points=50, crop_length=0.0, butt_height=0.6, relative_tolerance=0.08, base_radius=0.0, allometry_tolerance=0.3, buttress_equivalent_area=true, buttress_max_inlier_fraction=0.3, pipe_slack=1.2, branch_min_inlier_fraction=0.3, cluster_eps=0.1, centre_fit_points=100, radius_smooth_steps=15, butt_swell=1.1, butt_vertical_run=4, butt_max_lean_deg=50.0, chain_max_d=0.1, fourier_min_radius=0.15))]
#[allow(clippy::too_many_arguments)]
fn build_qsm<'py>(py: Python<'py>, xyz: PyReadonlyArray2<f64>, base_xy: Option<(f64, f64)>, k: usize, max_edge: f64, bin_length: f64, min_points: usize, ransac_threshold: f64, max_radius: f64, taper_limit: f64, max_rmse: f64, smooth_steps: usize, apex_radius: f64, min_arc_deg: f64, min_inlier_fraction: f64, prune_points: usize, fit_min_points: usize, crop_length: f64, butt_height: f64, relative_tolerance: f64, base_radius: f64, allometry_tolerance: f64, buttress_equivalent_area: bool, buttress_max_inlier_fraction: f64, pipe_slack: f64, branch_min_inlier_fraction: f64, cluster_eps: f64, centre_fit_points: usize, radius_smooth_steps: usize, butt_swell: f64, butt_vertical_run: usize, butt_max_lean_deg: f64, chain_max_d: f64, fourier_min_radius: f64) -> PyResult<Bound<'py, PyDict>> {
    let p = xyz_from_py(xyz)?;
    let params = qsm::QsmParams { k, max_edge, bin_length, min_points, ransac_threshold, max_radius, taper_limit, max_rmse, smooth_steps, apex_radius, min_arc_deg, min_inlier_fraction, prune_points, fit_min_points, crop_length, butt_height, relative_tolerance, base_radius, allometry_tolerance, buttress_equivalent_area, buttress_max_inlier_fraction, pipe_slack, branch_min_inlier_fraction, cluster_eps, centre_fit_points, radius_smooth_steps, butt_swell, butt_vertical_run, butt_max_lean_deg, chain_max_d, fourier_min_radius };
    let q = py.detach(|| qsm::build_qsm(&p, base_xy.map(|b| [b.0, b.1]), &params)).map_err(err)?;
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

#[pyfunction]
fn qsm_summary<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyDict>> {
    qsm_to_py(py, &qsm_from_rows(cylinders)?)
}

#[pyfunction]
#[pyo3(signature = (cylinders, sides=12))]
fn qsm_mesh<'py>(py: Python<'py>, cylinders: PyReadonlyArray2<f64>, sides: usize) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<u32>>)> {
    let (v, t, o) = qsm_from_rows(cylinders)?.mesh(sides);
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
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyRayVoxels>()?;
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
        wrap_pyfunction!(euclidean_clusters, m)?,
        wrap_pyfunction!(knn, m)?,
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
        wrap_pyfunction!(ray_voxelize, m)?,
        wrap_pyfunction!(ray_voxelize_file, m)?,
        wrap_pyfunction!(write_shots, m)?,
        wrap_pyfunction!(read_shots, m)?,
        wrap_pyfunction!(shots_info, m)?,
        wrap_pyfunction!(laser_spec, m)?,
        wrap_pyfunction!(leaf_projection, m)?,
        wrap_pyfunction!(kabsch, m)?,
        wrap_pyfunction!(icp, m)?,
        wrap_pyfunction!(fit_circle, m)?,
        wrap_pyfunction!(fit_circle_ransac, m)?,
        wrap_pyfunction!(detect_stems, m)?,
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
