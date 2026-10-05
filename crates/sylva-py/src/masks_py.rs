// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::geo::masks.
//!
//! Polygon layers cross the boundary flattened: `coords (M, 2)`,
//! `ring_start (R + 1)` into coords, `part_start (P + 1)` into rings (the
//! first ring of each part is its exterior) and `feature_start (F + 1)` into
//! parts.

use std::collections::BTreeMap;

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::geo::masks::{self, Expr, MultiPolygon, Polygon, PolygonIndex, Property, RasterTest};
use sylva_rs::Point;

use crate::{attr_from_py, err, raster_from_py};

/// The rows of a C-contiguous `(N, 3)` float64 array, without a copy.
fn points<'a>(xyz: &'a PyReadonlyArray2<'_, f64>) -> PyResult<&'a [Point]> {
    if xyz.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!("xyz must have shape (N, 3), got (N, {})", xyz.shape()[1])));
    }
    let flat = xyz.as_slice().map_err(|_| PyValueError::new_err("xyz must be C-contiguous"))?;
    Ok(flat.as_chunks::<3>().0)
}

fn offsets(a: &PyReadonlyArray1<'_, i64>, len: usize, what: &str) -> PyResult<Vec<usize>> {
    let v = a.as_slice()?;
    let bad = || PyValueError::new_err(format!("{what} must start at 0, not decrease and end at {len}"));
    if v.first() != Some(&0) || v.last() != Some(&(len as i64)) || v.windows(2).any(|w| w[1] < w[0]) {
        return Err(bad());
    }
    Ok(v.iter().map(|&x| x as usize).collect())
}

pub(crate) fn features_from_flat(coords: PyReadonlyArray2<f64>, ring_start: PyReadonlyArray1<i64>, part_start: PyReadonlyArray1<i64>, feature_start: PyReadonlyArray1<i64>) -> PyResult<Vec<MultiPolygon>> {
    if coords.shape()[1] != 2 {
        return Err(PyValueError::new_err("coords must have shape (M, 2)"));
    }
    let c = coords.as_array();
    let xy: Vec<[f64; 2]> = c.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    let rings = offsets(&ring_start, xy.len(), "ring_start")?;
    let parts = offsets(&part_start, rings.len() - 1, "part_start")?;
    let feats = offsets(&feature_start, parts.len() - 1, "feature_start")?;
    let ring = |r: usize| xy[rings[r]..rings[r + 1]].to_vec();
    Ok(feats
        .windows(2)
        .map(|f| MultiPolygon {
            parts: (f[0]..f[1])
                .filter(|&p| parts[p + 1] > parts[p])
                .map(|p| Polygon { exterior: ring(parts[p]), holes: (parts[p] + 1..parts[p + 1]).map(ring).collect() })
                .collect(),
        })
        .collect())
}

/// Index of the first polygon containing each point's x, y, or -1.
#[pyfunction]
fn mask_polygon_index<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>, coords: PyReadonlyArray2<f64>, ring_start: PyReadonlyArray1<i64>, part_start: PyReadonlyArray1<i64>, feature_start: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let features = features_from_flat(coords, ring_start, part_start, feature_start)?;
    let index = PolygonIndex::new(&features).map_err(err)?;
    let p = points(&xyz)?;
    Ok(py.detach(|| index.feature_of(p)).into_pyarray(py))
}

fn property_to_py<'py>(py: Python<'py>, v: &Property) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Property::Null => py.None().into_bound(py),
        Property::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Property::Int(i) => i.into_pyobject(py)?.into_any(),
        Property::Float(x) => x.into_pyobject(py)?.into_any(),
        Property::Str(s) => s.into_pyobject(py)?.into_any(),
    })
}

/// Read a polygon layer; returns the flattened polygons, a list of
/// property dicts and the CRS text (or None).
#[pyfunction]
#[pyo3(signature = (path, layer=None))]
fn read_polygons<'py>(py: Python<'py>, path: std::path::PathBuf, layer: Option<String>) -> PyResult<Bound<'py, PyDict>> {
    let l = py.detach(|| masks::read_polygons(&path, layer.as_deref())).map_err(err)?;
    let (mut coords, mut ring_start, mut part_start, mut feature_start) = (Vec::<f64>::new(), vec![0i64], vec![0i64], vec![0i64]);
    let props = PyList::empty(py);
    for f in &l.features {
        for p in &f.geometry.parts {
            for r in std::iter::once(&p.exterior).chain(&p.holes) {
                coords.extend(r.iter().flatten());
                ring_start.push((coords.len() / 2) as i64);
            }
            part_start.push(ring_start.len() as i64 - 1);
        }
        feature_start.push(part_start.len() as i64 - 1);
        let d = PyDict::new(py);
        for (k, v) in &f.properties {
            d.set_item(k, property_to_py(py, v)?)?;
        }
        props.append(d)?;
    }
    let n = coords.len() / 2;
    let out = PyDict::new(py);
    out.set_item("coords", PyArray1::from_vec(py, coords).reshape([n, 2])?)?;
    out.set_item("ring_start", ring_start.into_pyarray(py))?;
    out.set_item("part_start", part_start.into_pyarray(py))?;
    out.set_item("feature_start", feature_start.into_pyarray(py))?;
    out.set_item("properties", props)?;
    out.set_item("crs", l.crs)?;
    Ok(out)
}

/// Points whose raster cell is valid and passes the range or value test.
#[pyfunction]
#[pyo3(signature = (xyz, data, xmin, ymin, resolution, min=None, max=None, values=None))]
#[allow(clippy::too_many_arguments)]
fn mask_raster<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>, data: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64, min: Option<f64>, max: Option<f64>, values: Option<Vec<f64>>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let r = raster_from_py(data, xmin, ymin, resolution);
    let test = match (min, max, values) {
        (None, None, None) => RasterTest::Valid,
        (min, max, None) => RasterTest::Range { min, max },
        (None, None, Some(v)) => RasterTest::Values(v),
        _ => return Err(PyValueError::new_err("give min and/or max, or values, not both")),
    };
    let p = points(&xyz)?;
    Ok(py.detach(|| masks::raster_mask(p, &r, &test)).map_err(err)?.into_pyarray(py))
}

/// Evaluate a mask expression on a cloud.
#[pyfunction]
fn mask_expression<'py>(py: Python<'py>, expr: &str, xyz: PyReadonlyArray2<'py, f64>, attrs: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let e = Expr::parse(expr).map_err(err)?;
    let available: Vec<String> = attrs.keys().iter().map(|k| k.extract::<String>()).collect::<PyResult<_>>()?;
    e.check_names(&available).map_err(err)?;
    // Only the attributes the expression uses cross the boundary.
    let mut used = BTreeMap::new();
    for name in e.names() {
        if let Some(v) = attrs.get_item(&name)? {
            used.insert(name, attr_from_py(&v)?);
        }
    }
    let p = points(&xyz)?;
    Ok(py.detach(|| e.mask(p, &used)).map_err(err)?.into_pyarray(py))
}

/// Points within `distance` of any point of `other`.
#[pyfunction]
fn mask_near<'py>(py: Python<'py>, xyz: PyReadonlyArray2<'py, f64>, other: PyReadonlyArray2<'py, f64>, distance: f64, horizontal: bool) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let (p, o) = (points(&xyz)?, points(&other)?);
    Ok(py.detach(|| masks::near_mask(p, o, distance, horizontal)).map_err(err)?.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mask_polygon_index, m)?)?;
    m.add_function(wrap_pyfunction!(read_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(mask_raster, m)?)?;
    m.add_function(wrap_pyfunction!(mask_expression, m)?)?;
    m.add_function(wrap_pyfunction!(mask_near, m)?)?;
    Ok(())
}
