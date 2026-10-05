// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Rasters as GeoTIFF, and the `.prj` beside an ESRI ASCII grid, without
//! GDAL.
//!
//! [`write_geotiff`] writes one uncompressed float32 band, north-up, in
//! strips, with NaN as nodata (the `GDAL_NODATA` tag). The CRS goes in the
//! GeoKey directory (OGC GeoTIFF 1.1): an EPSG code as
//! `ProjectedCSTypeGeoKey` or `GeographicTypeGeoKey` (with
//! `VerticalCSTypeGeoKey` for the vertical part of a compound code). A CRS
//! without an EPSG code would need its projection spelt out key by key, so
//! its WKT goes instead in a GDAL `.aux.xml` beside the file, which GDAL
//! (QGIS, rasterio, R's terra) reads as the CRS. A file of 4 GB or more is
//! written as BigTIFF.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::geo::crs::Crs;
use crate::error::{Error, Result};
use crate::raster::Raster;

const SHORT: u16 = 3;
const LONG: u16 = 4;
const DOUBLE: u16 = 12;
const ASCII: u16 = 2;
const LONG8: u16 = 16;

/// Bytes of image data per strip, about.
const STRIP_BYTES: usize = 1 << 18;

/// One IFD entry: its values already encoded little-endian.
struct Entry {
    tag: u16,
    typ: u16,
    count: u64,
    data: Vec<u8>,
}

fn shorts(tag: u16, v: &[u16]) -> Entry {
    Entry { tag, typ: SHORT, count: v.len() as u64, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
}

fn longs(tag: u16, v: &[u64], big: bool) -> Entry {
    if big {
        Entry { tag, typ: LONG8, count: v.len() as u64, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    } else {
        Entry { tag, typ: LONG, count: v.len() as u64, data: v.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect() }
    }
}

fn doubles(tag: u16, v: &[f64]) -> Entry {
    Entry { tag, typ: DOUBLE, count: v.len() as u64, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
}

fn ascii(tag: u16, s: &str) -> Entry {
    let mut data = s.as_bytes().to_vec();
    data.push(0);
    Entry { tag, typ: ASCII, count: data.len() as u64, data }
}

/// The GeoKey directory for `crs`, and the WKT to put in the `.aux.xml`
/// when the CRS has no EPSG code.
fn geokeys(crs: Option<&str>) -> Result<(Vec<u16>, Option<String>)> {
    // (key, location, count, value)
    let mut keys: Vec<[u16; 4]> = vec![[1025, 0, 1, 1]]; // GTRasterTypeGeoKey: PixelIsArea
    let mut aux = None;
    if let Some(text) = crs.map(str::trim).filter(|t| !t.is_empty()) {
        let c = Crs::parse(text)?;
        match c.epsg.and_then(|e| u16::try_from(e).ok()) {
            Some(code) => {
                let geographic = c.is_geographic()?;
                keys.push([1024, 0, 1, if geographic { 2 } else { 1 }]); // GTModelTypeGeoKey
                keys.push([if geographic { 2048 } else { 3072 }, 0, 1, code]);
                if let Some(v) = c.vertical_epsg.and_then(|v| u16::try_from(v).ok()) {
                    keys.push([4096, 0, 1, v]); // VerticalCSTypeGeoKey
                }
            }
            None => aux = Some(c.to_wkt().ok_or_else(|| Error::invalid(format!("the CRS {text:?} has no EPSG code or WKT to write")))?),
        }
    }
    keys.sort_by_key(|k| k[0]);
    let mut dir = vec![1, 1, 0, keys.len() as u16];
    dir.extend(keys.iter().flatten());
    Ok((dir, aux))
}

/// The GDAL `.aux.xml` beside a raster (`chm.tif` -> `chm.tif.aux.xml`).
pub fn aux_xml_path(path: impl AsRef<Path>) -> PathBuf {
    let mut p = path.as_ref().as_os_str().to_owned();
    p.push(".aux.xml");
    PathBuf::from(p)
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Write `r` as a single-band float32 GeoTIFF, north-up, NaN for nodata,
/// with `crs` (an EPSG code, PROJ string with an EPSG equivalent, or WKT)
/// when given.
pub fn write_geotiff(r: &Raster, path: impl AsRef<Path>, crs: Option<&str>) -> Result<()> {
    if r.nrows == 0 || r.ncols == 0 {
        return Err(Error::invalid("the raster is empty"));
    }
    if r.ncols > u32::MAX as usize || r.nrows > u32::MAX as usize {
        return Err(Error::invalid(format!("a {} x {} raster is too large for TIFF", r.nrows, r.ncols)));
    }
    let row_bytes = r.ncols * 4;
    let rows_per_strip = (STRIP_BYTES / row_bytes).clamp(1, r.nrows);
    let n_strips = r.nrows.div_ceil(rows_per_strip);
    let image_bytes = (r.nrows * row_bytes) as u64;
    let (geo, aux) = geokeys(crs)?;
    // Image data, small values and the directory well under 4 GB use classic TIFF.
    let big = image_bytes + (n_strips as u64) * 16 + 65536 >= u32::MAX as u64;
    let header_len: u64 = if big { 16 } else { 8 };
    let counts: Vec<u64> = (0..n_strips).map(|k| (rows_per_strip.min(r.nrows - k * rows_per_strip) * row_bytes) as u64).collect();
    let offsets: Vec<u64> = counts.iter().scan(header_len, |pos, &c| { let o = *pos; *pos += c; Some(o) }).collect();

    let mut entries = vec![
        longs(256, &[r.ncols as u64], false),
        longs(257, &[r.nrows as u64], false),
        shorts(258, &[32]),
        shorts(259, &[1]),
        shorts(262, &[1]),
        longs(273, &offsets, big),
        shorts(277, &[1]),
        longs(278, &[rows_per_strip as u64], false),
        longs(279, &counts, big),
        shorts(284, &[1]),
        shorts(339, &[3]),
        doubles(33550, &[r.resolution, r.resolution, 0.0]),
        doubles(33922, &[0.0, 0.0, 0.0, r.xmin, r.ymax(), 0.0]),
        shorts(34735, &geo),
    ];
    entries.push(ascii(42113, "nan"));
    entries.sort_by_key(|e| e.tag);

    // Values too long to sit in their entry go after the image, then the directory.
    let inline = if big { 8 } else { 4 };
    let mut pos = header_len + image_bytes;
    let mut blob_at = Vec::with_capacity(entries.len());
    for e in &entries {
        if e.data.len() > inline {
            pos += pos % 2;
            blob_at.push(Some(pos));
            pos += e.data.len() as u64;
        } else {
            blob_at.push(None);
        }
    }
    pos += pos % 2;
    let ifd_at = pos;
    if !big && ifd_at + 2 + 12 * entries.len() as u64 + 4 > u32::MAX as u64 {
        return Err(Error::invalid("internal error: classic TIFF chosen for a file over 4 GB"));
    }

    let path = path.as_ref();
    let mut f = BufWriter::with_capacity(1 << 20, File::create(path).map_err(|e| Error::file(path, e.to_string()))?);
    if big {
        f.write_all(b"II")?;
        f.write_all(&43u16.to_le_bytes())?;
        f.write_all(&8u16.to_le_bytes())?;
        f.write_all(&0u16.to_le_bytes())?;
        f.write_all(&ifd_at.to_le_bytes())?;
    } else {
        f.write_all(b"II")?;
        f.write_all(&42u16.to_le_bytes())?;
        f.write_all(&(ifd_at as u32).to_le_bytes())?;
    }
    let mut row = Vec::with_capacity(row_bytes);
    for rr in (0..r.nrows).rev() {
        row.clear();
        for c in 0..r.ncols {
            row.extend_from_slice(&(r.get(rr, c) as f32).to_le_bytes());
        }
        f.write_all(&row)?;
    }
    let mut written = header_len + image_bytes;
    for (e, at) in entries.iter().zip(&blob_at) {
        if let Some(at) = at {
            while written < *at {
                f.write_all(&[0])?;
                written += 1;
            }
            f.write_all(&e.data)?;
            written += e.data.len() as u64;
        }
    }
    while written < ifd_at {
        f.write_all(&[0])?;
        written += 1;
    }
    if big {
        f.write_all(&(entries.len() as u64).to_le_bytes())?;
    } else {
        f.write_all(&(entries.len() as u16).to_le_bytes())?;
    }
    for (e, at) in entries.iter().zip(&blob_at) {
        f.write_all(&e.tag.to_le_bytes())?;
        f.write_all(&e.typ.to_le_bytes())?;
        let mut value = vec![0u8; inline];
        match at {
            Some(o) if big => value.copy_from_slice(&o.to_le_bytes()),
            Some(o) => value.copy_from_slice(&(*o as u32).to_le_bytes()),
            None => value[..e.data.len()].copy_from_slice(&e.data),
        }
        if big {
            f.write_all(&e.count.to_le_bytes())?;
        } else {
            f.write_all(&(e.count as u32).to_le_bytes())?;
        }
        f.write_all(&value)?;
    }
    if big {
        f.write_all(&0u64.to_le_bytes())?;
    } else {
        f.write_all(&0u32.to_le_bytes())?;
    }
    f.flush()?;
    // A stale sidecar would give the file a CRS it no longer has.
    let sidecar = aux_xml_path(path);
    match aux {
        Some(wkt) => std::fs::write(&sidecar, format!("<PAMDataset>\n  <SRS>{}</SRS>\n</PAMDataset>\n", xml_escape(&wkt)))?,
        None if sidecar.exists() => std::fs::remove_file(&sidecar)?,
        None => {}
    }
    Ok(())
}

/// The `.prj` beside a raster file (`dtm.asc` -> `dtm.prj`).
pub fn prj_path(path: impl AsRef<Path>) -> PathBuf {
    path.as_ref().with_extension("prj")
}

/// Write the WKT of `crs` (an EPSG code, PROJ string with an EPSG
/// equivalent, or WKT) to the `.prj` beside `path`, as GIS software reads
/// the CRS of an ESRI ASCII grid.
pub fn write_prj(path: impl AsRef<Path>, crs: &str) -> Result<()> {
    let c = Crs::parse(crs)?;
    let wkt = c.to_wkt().ok_or_else(|| Error::invalid(format!("the CRS {crs:?} has no WKT form to write to a .prj")))?;
    std::fs::write(prj_path(path), wkt)?;
    Ok(())
}

/// The CRS in the `.prj` beside `path`, if there is one: `EPSG:<code>` when
/// its WKT names an EPSG code, else the WKT.
pub fn read_prj(path: impl AsRef<Path>) -> Result<Option<String>> {
    let p = prj_path(path);
    if !p.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&p)?;
    let wkt = text.trim().trim_end_matches('\0').to_string();
    if wkt.is_empty() {
        return Ok(None);
    }
    Ok(Some(match crate::als::epsg_from_wkt(&wkt) {
        Some(code) => format!("EPSG:{code}"),
        None => wkt,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tags of a classic little-endian TIFF: tag -> (type, count, value bytes).
    fn tags(bytes: &[u8]) -> std::collections::BTreeMap<u16, (u16, u32, Vec<u8>)> {
        let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        assert_eq!(&bytes[..4], b"II\x2a\x00");
        let ifd = u32_at(4) as usize;
        let n = u16_at(ifd) as usize;
        let mut out = std::collections::BTreeMap::new();
        for k in 0..n {
            let e = ifd + 2 + 12 * k;
            let (tag, typ, count) = (u16_at(e), u16_at(e + 2), u32_at(e + 4));
            let size = match typ { 2 => 1, 3 => 2, 4 => 4, 12 => 8, _ => panic!("type {typ}") } * count as usize;
            let data = if size <= 4 { bytes[e + 8..e + 8 + size].to_vec() } else { let o = u32_at(e + 8) as usize; bytes[o..o + size].to_vec() };
            out.insert(tag, (typ, count, data));
        }
        out
    }

    fn as_u16(v: &[u8]) -> Vec<u16> {
        v.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    fn as_u32(v: &[u8]) -> Vec<u32> {
        v.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn as_f64(v: &[u8]) -> Vec<f64> {
        v.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("sylva-geotiff-{name}-{}", std::process::id()))
    }

    #[test]
    fn a_geotiff_holds_the_grid_north_up_with_its_crs() {
        let mut r = Raster::filled(3, 70_000, 500_000.0, 6_960_000.0, 0.5, 1.0);
        r.set(0, 0, 7.0); // south-west
        r.set(2, 1, 9.0); // north row, second column
        r.set(1, 5, f64::NAN);
        let path = tmp("epsg").with_extension("tif");
        write_geotiff(&r, &path, Some("EPSG:28356")).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let t = tags(&bytes);
        assert_eq!(as_u32(&t[&256].2), [70_000]);
        assert_eq!(as_u32(&t[&257].2), [3]);
        assert_eq!(as_u16(&t[&339].2), [3]);
        assert_eq!(as_f64(&t[&33550].2), [0.5, 0.5, 0.0]);
        assert_eq!(as_f64(&t[&33922].2), [0.0, 0.0, 0.0, 500_000.0, 6_960_001.5, 0.0]);
        assert_eq!(as_u16(&t[&34735].2), [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 28356]);
        assert_eq!(&t[&42113].2, b"nan\0");
        // Pixels, north row first, read through the strips.
        let offsets = as_u32(&t[&273].2);
        let counts = as_u32(&t[&279].2);
        assert!(offsets.len() > 1, "70,000 columns need a strip per row");
        let mut pixels = Vec::new();
        for (o, c) in offsets.iter().zip(&counts) {
            pixels.extend(bytes[*o as usize..(*o + *c) as usize].chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
        }
        assert_eq!(pixels.len(), 3 * 70_000);
        assert_eq!(pixels[1], 9.0);
        assert_eq!(pixels[2 * 70_000], 7.0);
        assert!(pixels[70_000 + 5].is_nan());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_crs_without_an_epsg_code_is_written_as_wkt() {
        let wkt = r#"PROJCS["custom TM",GEOGCS["GDA94",DATUM["Geocentric_Datum_of_Australia_1994",SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",152],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1]]"#;
        let r = Raster::filled(2, 2, 0.0, 0.0, 1.0, 0.0);
        let path = tmp("wkt").with_extension("tif");
        write_geotiff(&r, &path, Some(wkt)).unwrap();
        let t = tags(&std::fs::read(&path).unwrap());
        assert_eq!(as_u16(&t[&34735].2), [1, 1, 0, 1, 1025, 0, 1, 1]);
        let aux = std::fs::read_to_string(aux_xml_path(&path)).unwrap();
        assert!(aux.contains("<SRS>PROJCS[&quot;custom TM&quot;"), "{aux}");
        // No CRS at all: only the raster type, and the old sidecar goes.
        write_geotiff(&r, &path, None).unwrap();
        assert!(!aux_xml_path(&path).exists());
        let t = tags(&std::fs::read(&path).unwrap());
        assert_eq!(as_u16(&t[&34735].2), [1, 1, 0, 1, 1025, 0, 1, 1]);
        assert!(write_geotiff(&r, &path, Some("not a crs")).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_prj_round_trips_the_crs() {
        let asc = tmp("prj").with_extension("asc");
        assert_eq!(read_prj(&asc).unwrap(), None);
        write_prj(&asc, "EPSG:28356").unwrap();
        assert!(std::fs::read_to_string(prj_path(&asc)).unwrap().contains("MGA zone 56"));
        assert_eq!(read_prj(&asc).unwrap().as_deref(), Some("EPSG:28356"));
        let _ = std::fs::remove_file(prj_path(&asc));
    }
}
