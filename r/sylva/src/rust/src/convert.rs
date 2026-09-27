// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Conversions between R values and sylva-rs types.

use std::collections::BTreeMap;
use std::collections::HashMap;

use extendr_api::prelude::*;
use sylva_rs::pointcloud::Attr;
use sylva_rs::{Point, PointCloud, Raster, Shots};

pub type Result<T> = std::result::Result<T, Error>;

pub fn err(e: sylva_rs::Error) -> Error {
    Error::Other(e.to_string())
}

pub fn fail<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Other(msg.into()))
}

// ----------------------------------------------------------------- converters

pub fn xyz_from_r(xyz: &Robj) -> Result<Vec<Point>> {
    let m: RMatrix<f64> = xyz.try_into().map_err(|_| Error::Other("xyz must be a double matrix with 3 columns".into()))?;
    if m.ncols() != 3 {
        return Err(Error::Other(format!("xyz must have 3 columns, got {}", m.ncols())));
    }
    let n = m.nrows();
    let d = m.data();
    Ok((0..n).map(|i| [d[i], d[n + i], d[2 * n + i]]).collect())
}

pub fn xyz_to_r(xyz: &[Point]) -> Robj {
    let n = xyz.len();
    RMatrix::new_matrix(n, 3, |r, c| xyz[r][c]).into()
}

pub fn attr_to_r(a: &Attr) -> Robj {
    match a {
        Attr::F64(v) => v.clone().into(),
        Attr::F32(v) => v.iter().map(|&x| x as f64).collect::<Vec<f64>>().into(),
        Attr::I64(v) => v.iter().map(|&x| x as f64).collect::<Vec<f64>>().into(),
        Attr::U32(v) => v.iter().map(|&x| x as f64).collect::<Vec<f64>>().into(),
        Attr::I32(v) => v.clone().into(),
        Attr::U16(v) => v.iter().map(|&x| x as i32).collect::<Vec<i32>>().into(),
        Attr::U8(v) => v.iter().map(|&x| x as i32).collect::<Vec<i32>>().into(),
        Attr::I8(v) => v.iter().map(|&x| x as i32).collect::<Vec<i32>>().into(),
        Attr::Bool(v) => v.iter().map(|&x| Rbool::from(x)).collect::<Logicals>().into(),
    }
}

pub fn attr_from_r(name: &str, v: &Robj) -> Result<Attr> {
    if let Some(x) = v.as_real_slice() {
        return Ok(Attr::F64(x.to_vec()));
    }
    if let Some(x) = v.as_integer_slice() {
        return Ok(Attr::I32(x.to_vec()));
    }
    if let Some(x) = v.as_logical_slice() {
        return Ok(Attr::Bool(x.iter().map(|b| b.is_true()).collect()));
    }
    Err(Error::Other(format!("attribute `{name}` must be a double, integer or logical vector")))
}

pub fn cloud_from_r(cloud: &List) -> Result<PointCloud> {
    let map: HashMap<&str, Robj> = cloud.clone().try_into()?;
    let xyz = xyz_from_r(map.get("xyz").ok_or_else(|| Error::Other("cloud has no `xyz`".into()))?)?;
    let mut attrs = BTreeMap::new();
    if let Some(a) = map.get("attrs") {
        if !a.is_null() {
            let list: List = a.try_into()?;
            for (name, v) in list.iter() {
                attrs.insert(name.to_string(), attr_from_r(name, &v)?);
            }
        }
    }
    PointCloud::with_attrs(xyz, attrs).map_err(err)
}

pub fn cloud_to_r(cloud: &PointCloud) -> List {
    let names: Vec<&str> = cloud.attrs.keys().map(|s| s.as_str()).collect();
    let values: Vec<Robj> = cloud.attrs.values().map(attr_to_r).collect();
    let attrs = List::from_names_and_values(names, values).expect("names match values");
    list!(xyz = xyz_to_r(&cloud.xyz), attrs = attrs)
}

/// A double vector (integers are widened).
pub fn doubles(v: &Robj, what: &str) -> Result<Vec<f64>> {
    if let Some(x) = v.as_real_slice() {
        return Ok(x.to_vec());
    }
    if let Some(x) = v.as_integer_slice() {
        return Ok(x.iter().map(|&i| i as f64).collect());
    }
    fail(format!("{what} must be numeric"))
}

fn field<'a>(map: &'a HashMap<&str, Robj>, k: &str, what: &str) -> Result<&'a Robj> {
    map.get(k).ok_or_else(|| Error::Other(format!("{what} has no `{k}`")))
}

/// Shots from `list(origin, direction, echo_start, echo_count, echo_range,
/// echo_attrs)`, with `echo_start` 0-based as in the Python package and the
/// shots file format.
pub fn shots_from_r(shots: &List) -> Result<Shots> {
    let map: HashMap<&str, Robj> = shots.clone().try_into()?;
    let origin = xyz_from_r(field(&map, "origin", "shots")?)?;
    let direction = xyz_from_r(field(&map, "direction", "shots")?)?;
    let start = doubles(field(&map, "echo_start", "shots")?, "echo_start")?;
    let count = doubles(field(&map, "echo_count", "shots")?, "echo_count")?;
    let echo_range = doubles(field(&map, "echo_range", "shots")?, "echo_range")?;
    let mut echo_attrs = BTreeMap::new();
    if let Some(a) = map.get("echo_attrs") {
        if !a.is_null() {
            let list: List = a.try_into()?;
            for (name, v) in list.iter() {
                echo_attrs.insert(name.to_string(), attr_from_r(name, &v)?);
            }
        }
    }
    let n = origin.len();
    if direction.len() != n || start.len() != n || count.len() != n {
        return fail("shots: origin, direction, echo_start and echo_count must have one row per pulse");
    }
    let n_echo = echo_range.len() as f64;
    for (&s0, &c) in start.iter().zip(&count) {
        if s0 < 0.0 || c < 0.0 || s0 + c > n_echo || s0.fract() != 0.0 || c.fract() != 0.0 {
            return fail("shots: echo_start / echo_count point outside echo_range");
        }
    }
    for (name, a) in &echo_attrs {
        if a.len() != echo_range.len() {
            return fail(format!("shots: echo attribute `{name}` has {} values for {} echoes", a.len(), echo_range.len()));
        }
    }
    Ok(Shots {
        origin,
        direction,
        echo_start: start.iter().map(|&v| v as usize).collect(),
        echo_count: count.iter().map(|&v| v as u32).collect(),
        echo_range,
        echo_attrs,
    })
}

pub fn shots_to_r(s: &Shots) -> List {
    let names: Vec<&str> = s.echo_attrs.keys().map(|k| k.as_str()).collect();
    let values: Vec<Robj> = s.echo_attrs.values().map(attr_to_r).collect();
    let attrs = List::from_names_and_values(names, values).expect("names match values");
    list!(
        origin = xyz_to_r(&s.origin),
        direction = xyz_to_r(&s.direction),
        echo_start = s.echo_start.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
        echo_count = s.echo_count.iter().map(|&v| v as i32).collect::<Vec<i32>>(),
        echo_range = s.echo_range.clone(),
        echo_attrs = attrs
    )
}

/// A raster from its R pieces: `data` a numeric matrix, rows from `ymin` up.
pub fn raster_from_r(data: &Robj, xmin: f64, ymin: f64, resolution: f64) -> Result<Raster> {
    let m: RMatrix<f64> = data.try_into().map_err(|_| Error::Other("raster data must be a double matrix".into()))?;
    let (nr, nc) = (m.nrows(), m.ncols());
    let d = m.data();
    // R is column-major; the core raster is row-major.
    let data = (0..nr).flat_map(|r| (0..nc).map(move |c| d[c * nr + r])).collect();
    Ok(Raster { data, nrows: nr, ncols: nc, xmin, ymin, resolution })
}

/// A numeric matrix from row-major values.
pub fn matrix_from_rows(rows: usize, cols: usize, v: &[f64]) -> Robj {
    RMatrix::new_matrix(rows, cols, |r, c| v[r * cols + c]).into()
}

/// A 4 x 4 transform from an R matrix (column-major, as R stores it).
pub fn matrix4_from_r(m: &Robj) -> Result<sylva_rs::Transform> {
    let v = doubles(m, "matrix")?;
    if v.len() != 16 || m.dim().map(|d| d.len() == 2 && d.iter().all(|x| x.0 == 4)) == Some(false) {
        return fail("matrix must be 4 x 4");
    }
    Ok(sylva_rs::Transform(nalgebra::Matrix4::from_column_slice(&v)))
}

/// A 4 x 4 R matrix from a transform.
pub fn matrix4_to_r(t: &sylva_rs::Transform) -> Robj {
    RMatrix::new_matrix(4, 4, |r, c| t.0[(r, c)]).into()
}

/// A raster as `list(data, xmin, ymin, resolution)`, `data` with rows from `ymin` up.
pub fn raster_to_r(r: &Raster) -> List {
    list!(data = matrix_from_rows(r.nrows, r.ncols, &r.data), xmin = r.xmin, ymin = r.ymin, resolution = r.resolution)
}

/// `NULL` or four numbers `(xmin, ymin, xmax, ymax)`.
pub fn bounds_from_r(b: &Robj) -> Result<Option<(f64, f64, f64, f64)>> {
    if b.is_null() {
        return Ok(None);
    }
    let v = doubles(b, "bounds")?;
    if v.len() != 4 {
        return fail("bounds must be c(xmin, ymin, xmax, ymax)");
    }
    Ok(Some((v[0], v[1], v[2], v[3])))
}

/// Three numbers as a point.
pub fn point_from_r(v: &[f64], what: &str) -> Result<Point> {
    if v.len() != 3 {
        return fail(format!("{what} must have three values"));
    }
    Ok([v[0], v[1], v[2]])
}

/// 0-based indices as R's 1-based positions (doubles, so no size limit).
pub fn positions(idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| i as f64 + 1.0).collect()
}

/// Booleans as an R logical vector.
pub fn logicals(v: &[bool]) -> Logicals {
    v.iter().map(|&b| Rbool::from(b)).collect()
/// `NULL` or the first value of a numeric vector.
pub fn optional_f64(v: &Robj, what: &str) -> Result<Option<f64>> {
    if v.is_null() {
        return Ok(None);
    }
    doubles(v, what)?.first().copied().map(Some).ok_or_else(|| Error::Other(format!("{what} is empty")))
}

/// Triangles from an `n x 3` matrix of 0-based vertex indices.
pub fn faces_from_r(faces: &Robj) -> Result<Vec<[u32; 3]>> {
    let (n, d): (usize, Vec<f64>) = if let Ok(m) = RMatrix::<f64>::try_from(faces) {
        if m.ncols() != 3 {
            return fail("faces must have 3 columns");
        }
        (m.nrows(), m.data().to_vec())
    } else if let Ok(m) = RMatrix::<i32>::try_from(faces) {
        if m.ncols() != 3 {
            return fail("faces must have 3 columns");
        }
        (m.nrows(), m.data().iter().map(|&v| v as f64).collect())
    } else {
        return fail("faces must be a numeric matrix with 3 columns");
    };
    if d.iter().any(|&v| !(v >= 0.0 && v <= u32::MAX as f64 && v.fract() == 0.0)) {
        return fail("faces must hold non-negative whole indices");
    }
    Ok((0..n).map(|i| [d[i] as u32, d[n + i] as u32, d[2 * n + i] as u32]).collect())
}

/// An `n x 3` integer matrix of 0-based vertex indices.
pub fn faces_to_r(faces: &[[u32; 3]]) -> Robj {
    RMatrix::new_matrix(faces.len(), 3, |r, c| faces[r][c] as i32).into()
}
