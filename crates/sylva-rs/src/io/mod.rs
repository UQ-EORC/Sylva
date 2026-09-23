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
pub mod las;
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

/// Write a point cloud, choosing the writer from the extension.
pub fn write(cloud: &PointCloud, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    match ext(path).as_str() {
        "las" | "laz" => las::write_las(cloud, path, &las::LasWriteOptions::default()),
        "ply" => ply::write_ply(cloud, path, true),
        "xyz" | "txt" | "asc" | "pts" => ascii::write_ascii(cloud, path, " ", true, 4),
        "csv" => ascii::write_ascii(cloud, path, ",", true, 4),
        other => Err(Error::UnsupportedFormat(other.to_string())),
    }
}
