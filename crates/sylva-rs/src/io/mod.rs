// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Point cloud readers and writers, dispatched on file extension.
//!
//! | Format                    | read | write |
//! |---------------------------|------|-------|
//! | `.las` / `.laz`           | yes  | yes   |
//! | `.ply` (ascii/binary)     | yes  | yes   |
//! | `.xyz` `.txt` `.csv` `.asc` `.pts` | yes | yes |
//! | `.rxp` (RIEGL, RiVLib)    | yes  | no    |
//! | `.rdbx` (RIEGL, librdb)   | yes  | no    |
//!
//! Pulse data ([`crate::Shots`]) has its own Parquet-based format, see [`shots`].

pub mod ascii;
pub mod geotiff;
pub mod las;
pub mod mesh;
pub mod ply;
pub mod rdb;
pub mod mta;
pub mod mta_model;
pub mod riegl;
pub mod rxp_mta;
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
        "rdbx" => rdb::read_rdbx(path, &rdb::RdbxOptions::default()),
        other => Err(Error::UnsupportedFormat(other.to_string())),
    }
}

/// [`read`] keeping only the attributes named in `keep` (None keeps them
/// all; names the file does not have are ignored). LAS/LAZ never allocates
/// the other columns; the other formats read everything and drop them.
pub fn read_attrs(path: impl AsRef<Path>, keep: Option<&[&str]>) -> Result<PointCloud> {
    let path = path.as_ref();
    if matches!(ext(path).as_str(), "las" | "laz") {
        return las::read_las_attrs(path, keep);
    }
    let mut cloud = read(path)?;
    if let Some(keep) = keep {
        cloud.attrs.retain(|k, _| keep.contains(&k.as_str()));
    }
    Ok(cloud)
}

/// Write the points of the LAS/LAZ file `src` to `path` with the columns of
/// `extra` added (one value per point, in file order), a batch at a time:
/// [`las::write_las_annotated`] with [`WriteOptions`]. `lo` is the minimum
/// corner for the header's offsets (None: the source header's).
pub fn write_annotated(src: impl AsRef<Path>, path: impl AsRef<Path>, opts: &WriteOptions, extra: &std::collections::BTreeMap<String, crate::pointcloud::Attr>, lo: Option<crate::Point>) -> Result<()> {
    let (src, path) = (src.as_ref(), path.as_ref());
    match (ext(src).as_str(), ext(path).as_str()) {
        ("las" | "laz", "las" | "laz") => las::write_las_annotated(src, path, &las::LasWriteOptions { point_format: opts.point_format, scale: opts.scale, crs_wkt: opts.crs_wkt.clone() }, extra, lo),
        ("las" | "laz", other) | (other, _) => Err(Error::UnsupportedFormat(other.to_string())),
    }
}

/// Options for [`write_with`]; each applies to one family of formats.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// LAS point data record format (LAS/LAZ only); a cloud with colour or GPS
    /// time is written in the nearest format that holds them ([`las::format_for`]).
    pub point_format: u8,
    /// Coordinate resolution in the units of the coordinates: the LAS/LAZ
    /// quantisation step, and for text files the decimals written, one more
    /// than the scale has ([`text_decimals`]: 0.001 m gives 4 decimals,
    /// 1e-7 degrees gives 8).
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

/// Decimals a text file's coordinates are written with for a coordinate
/// resolution of `scale` (in the units of the coordinates): one more than the
/// scale's own, so 0.001 (millimetres) gives 4, a tenth of a millimetre, and
/// 1e-7 (degrees of a geographic CRS, about a centimetre) gives 8. Between 0
/// and 17; 4 for a scale that is not a positive number.
pub fn text_decimals(scale: f64) -> usize {
    if !(scale > 0.0) || !scale.is_finite() {
        return 4;
    }
    ((-scale.log10() - 1e-9).ceil() + 1.0).clamp(0.0, 17.0) as usize
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
        "xyz" | "txt" | "asc" | "pts" => ascii::write_ascii(cloud, path, " ", true, text_decimals(opts.scale)),
        "csv" => ascii::write_ascii(cloud, path, ",", true, text_decimals(opts.scale)),
        other => Err(Error::UnsupportedFormat(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_coordinates_follow_the_scale() {
        assert_eq!([0.001, 0.0001, 1e-7, 0.01, 1.0, 0.005].map(text_decimals), [4, 5, 8, 3, 1, 4]);
        assert_eq!([0.0, -1.0, f64::NAN, 1e-30].map(text_decimals), [4, 4, 4, 17]);
        // Longitude and latitude at four decimals are good to eleven metres;
        // at the 1e-7 degrees the Python package asks for, to a centimetre.
        let cloud = PointCloud::new(vec![[153.0123456789, -27.4987654321, 12.0]]);
        let path = std::env::temp_dir().join(format!("sylva-io-text-{}.csv", std::process::id()));
        write_with(&cloud, &path, &WriteOptions { scale: 1e-7, ..Default::default() }).unwrap();
        let back = read(&path).unwrap();
        assert!((back.xyz[0][0] - cloud.xyz[0][0]).abs() < 1e-8 && (back.xyz[0][1] - cloud.xyz[0][1]).abs() < 1e-8);
        write(&cloud, &path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().nth(1), Some("153.0123,-27.4988,12.0000"), "metres stay at 0.1 mm");
        let _ = std::fs::remove_file(&path);
    }
}
