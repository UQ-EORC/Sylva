// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! R bindings for sylva-rs.
//!
//! Point clouds cross the boundary as `list(xyz = <n x 3 double matrix>,
//! attrs = <named list of vectors>)`; see `R/` for the friendly wrappers.
//! Attribute types map to R as: floats and 64-bit or unsigned 32-bit
//! integers to double, smaller integers to integer, booleans to logical.

use std::collections::BTreeMap;

use extendr_api::prelude::*;
use sylva_rs::pointcloud::Attr;
use sylva_rs::{filters, io, Point, PointCloud};

type Result<T> = std::result::Result<T, Error>;

fn err(e: sylva_rs::Error) -> Error {
    Error::Other(e.to_string())
}

// ----------------------------------------------------------------- converters

fn xyz_from_r(xyz: &Robj) -> Result<Vec<Point>> {
    let m: RMatrix<f64> = xyz.try_into().map_err(|_| Error::Other("xyz must be a double matrix with 3 columns".into()))?;
    if m.ncols() != 3 {
        return Err(Error::Other(format!("xyz must have 3 columns, got {}", m.ncols())));
    }
    let n = m.nrows();
    let d = m.data();
    Ok((0..n).map(|i| [d[i], d[n + i], d[2 * n + i]]).collect())
}

fn xyz_to_r(xyz: &[Point]) -> Robj {
    let n = xyz.len();
    RMatrix::new_matrix(n, 3, |r, c| xyz[r][c]).into()
}

fn attr_to_r(a: &Attr) -> Robj {
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

fn attr_from_r(name: &str, v: &Robj) -> Result<Attr> {
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

fn cloud_from_r(cloud: &List) -> Result<PointCloud> {
    let map: std::collections::HashMap<&str, Robj> = cloud.clone().try_into()?;
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

fn cloud_to_r(cloud: &PointCloud) -> List {
    let names: Vec<&str> = cloud.attrs.keys().map(|s| s.as_str()).collect();
    let values: Vec<Robj> = cloud.attrs.values().map(attr_to_r).collect();
    let attrs = List::from_names_and_values(names, values).expect("names match values");
    list!(xyz = xyz_to_r(&cloud.xyz), attrs = attrs)
}

// ------------------------------------------------------------------------ I/O

/// Read a point cloud (LAS/LAZ, PLY, ASCII or RIEGL RXP).
/// @noRd
#[extendr]
fn core_read(path: &str) -> Result<List> {
    Ok(cloud_to_r(&io::read(path).map_err(err)?))
}

/// Write a point cloud, the format taken from the extension.
/// @noRd
#[extendr]
fn core_write(cloud: List, path: &str) -> Result<()> {
    io::write(&cloud_from_r(&cloud)?, path).map_err(err)
}

/// Check and normalise a cloud built in R (lengths, types).
/// @noRd
#[extendr]
fn core_cloud(cloud: List) -> Result<List> {
    Ok(cloud_to_r(&cloud_from_r(&cloud)?))
}

// -------------------------------------------------------------------- filters

/// One point per voxel: the first, or the centroid.
/// @noRd
#[extendr]
fn core_voxel_downsample(cloud: List, voxel_size: f64, centroid: bool) -> Result<List> {
    if !(voxel_size > 0.0) {
        return Err(Error::Other("voxel_size must be positive".into()));
    }
    Ok(cloud_to_r(&filters::voxel_downsample(&cloud_from_r(&cloud)?, voxel_size, centroid)))
}

extendr_module! {
    mod sylva;
    fn core_read;
    fn core_write;
    fn core_cloud;
    fn core_voxel_downsample;
}
