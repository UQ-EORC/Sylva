// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Point cloud readers and writers, dispatched on file extension.
//!
//! | Format                    | read | write |
//! |---------------------------|------|-------|
//! | `.las` / `.laz`           | yes  | yes   |
//! | `.ply` (ascii/binary)     | yes  | yes   |
//! | `.xyz` `.txt` `.csv` `.pts` | yes | yes  |
//! | `.rxp` (RIEGL, RiVLib)    | yes  | no    |
//!
//! Pulse data ([`crate::Shots`]) has its own Parquet-based format, see [`shots`].

pub mod ascii;
pub mod geotiff;
pub mod las;
pub mod mesh;
pub mod ply;
pub mod riegl;
pub mod shots;

use std::path::Path;

use crate::error::{Error, Result};
use crate::PointCloud;

fn ext(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

/// Read a point cloud, choosing the reader from the extension.
pub fn read(path: impl AsRef<Path>) -> Result<PointCloud> {
    let path = path.as_ref();
    match ext(path).as_str() {
        "las" | "laz" => las::read_las(path),
        "ply" => ply::read_ply(path),
        "xyz" | "txt" | "csv" | "asc" | "pts" => ascii::read_ascii(path, None),
        "rxp" => riegl::read_rxp(path, &riegl::RxpOptions::default()),
        other => Err(Error::UnsupportedFormat(other.to_string())),
    }
}

/// Options for [`write_with`]; each applies to one family of formats.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// LAS point data record format (LAS/LAZ only).
    pub point_format: u8,
    /// LAS coordinate quantisation in metres (LAS/LAZ only).
    pub scale: f64,
    /// Binary little-endian PLY rather than ASCII (PLY only).
    pub binary: bool,
    /// CRS stored as a WKT VLR (LAS/LAZ only; other formats have no place for it).
    pub crs_wkt: Option<String>,
}

impl Default for WriteOptions {
    fn default() -> Self {
        let las = las::LasWriteOptions::default();
        WriteOptions { point_format: las.point_format, scale: las.scale, binary: true, crs_wkt: None }
    }
}

/// Write a point cloud, choosing the writer from the extension.
pub fn write(cloud: &PointCloud, path: impl AsRef<Path>) -> Result<()> {
    write_with(cloud, path, &WriteOptions::default())
}

/// Write a point cloud, choosing the writer from the extension, with options.
pub fn write_with(cloud: &PointCloud, path: impl AsRef<Path>, opts: &WriteOptions) -> Result<()> {
    let path = path.as_ref();
    match ext(path).as_str() {
        "las" | "laz" => las::write_las(cloud, path, &las::LasWriteOptions { point_format: opts.point_format, scale: opts.scale, crs_wkt: opts.crs_wkt.clone() }),
        "ply" => ply::write_ply(cloud, path, opts.binary),
        "xyz" | "txt" | "asc" | "pts" => ascii::write_ascii(cloud, path, " ", true, 4),
        "csv" => ascii::write_ascii(cloud, path, ",", true, 4),
        other => Err(Error::UnsupportedFormat(other.to_string())),
    }
}
