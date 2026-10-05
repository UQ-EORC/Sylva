// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! LAS / LAZ via the `las` crate, with typed extra-bytes dimensions
//! (ASPRS LAS 1.4 R15 specification; LAZ is Isenburg's 2013 LASzip).

use std::path::Path;

use byteorder::{ByteOrder, LittleEndian};
use las::point::{Classification, Format};
use las::{Builder, Point as LasPoint, Reader, Transform as LasTransform, Vector, Vlr, Writer};

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::{Point, PointCloud};

const EXTRA_BYTES_USER_ID: &str = "LASF_Spec";
const EXTRA_BYTES_RECORD_ID: u16 = 4;

/// One extra-bytes dimension as described by the LAS 1.4 spec VLR.
///
/// LAS carries a fixed set of fields, so anything else - a ray's deviation, a
/// tree id, a leaf/wood label - rides in "extra bytes" appended to each point
/// record. A header record then says what those bytes mean: a name, a type
/// code and a width. Reading them is therefore two steps: parse that record
/// (`parse_extra_bytes_vlr`), then cut each point's trailing bytes up
/// accordingly (`decode_dim`).
#[derive(Debug, Clone)]
struct ExtraDim {
    name: String,
    data_type: u8,
    size: usize,
}

/// Bytes taken by one of the specification's numbered types, if it is one we
/// can read; `None` for the deprecated array types.
fn dim_size(data_type: u8) -> Option<usize> {
    Some(match data_type {
        1 | 2 => 1,
        3 | 4 => 2,
        5 | 6 => 4,
        7 | 8 => 8,
        9 => 4,
        10 => 8,
        _ => return None,
    })
}

/// The dimensions described by the extra-bytes record, in order.
///
/// Each description is exactly 192 bytes, so the record is read as a run of
/// them: `chunks_exact(192)` hands over one at a time and ignores any
/// trailing partial chunk. A dimension of a type we cannot decode is still
/// listed, with its width, because its bytes have to be stepped over to reach
/// the dimensions after it.
fn parse_extra_bytes_vlr(vlr: &Vlr) -> Vec<ExtraDim> {
    let mut dims = Vec::new();
    for rec in vlr.data.chunks_exact(192) {
        let data_type = rec[2];
        let name_end = rec[4..36].iter().position(|&b| b == 0).unwrap_or(32);
        let name = String::from_utf8_lossy(&rec[4..4 + name_end]).to_string();
        match dim_size(data_type) {
            Some(size) => dims.push(ExtraDim { name, data_type, size }),
            None => {
                // Undocumented/deprecated array types: skip by options byte size.
                let opts = rec[3];
                dims.push(ExtraDim { name, data_type: 0, size: opts as usize });
            }
        }
    }
    dims
}

fn decode_dim(dim: &ExtraDim, bytes: &[u8]) -> f64 {
    match dim.data_type {
        1 => bytes[0] as f64,
        2 => bytes[0] as i8 as f64,
        3 => LittleEndian::read_u16(bytes) as f64,
        4 => LittleEndian::read_i16(bytes) as f64,
        5 => LittleEndian::read_u32(bytes) as f64,
        6 => LittleEndian::read_i32(bytes) as f64,
        7 => LittleEndian::read_u64(bytes) as f64,
        8 => LittleEndian::read_i64(bytes) as f64,
        9 => LittleEndian::read_f32(bytes) as f64,
        10 => LittleEndian::read_f64(bytes),
        _ => f64::NAN,
    }
}

fn attr_from_dim(dim: &ExtraDim, values: Vec<f64>) -> Attr {
    match dim.data_type {
        1 => Attr::U8(values.iter().map(|&v| v as u8).collect()),
        2 => Attr::I8(values.iter().map(|&v| v as i8).collect()),
        3 => Attr::U16(values.iter().map(|&v| v as u16).collect()),
        4 | 6 => Attr::I32(values.iter().map(|&v| v as i32).collect()),
        5 => Attr::U32(values.iter().map(|&v| v as u32).collect()),
        7 | 8 => Attr::I64(values.iter().map(|&v| v as i64).collect()),
        9 => Attr::F32(values.iter().map(|&v| v as f32).collect()),
        _ => Attr::F64(values),
    }
}

/// Read a LAS/LAZ file. Standard dimensions become attributes named as in
/// laspy (`intensity`, `return_number`, `classification`, `gps_time`, ...).
pub fn read_las(path: impl AsRef<Path>) -> Result<PointCloud> {
    read_las_impl(path.as_ref(), |_| true, true)
}

/// Points read per batch by [`read_las_where`].
const READ_BATCH: u64 = 1 << 20;

/// Read the points of a LAS/LAZ file for which `keep(&[x, y, z])` is true,
/// with the attributes of [`read_las`]. The file is streamed in batches, so
/// only the kept points are held in memory (LAZ is still decompressed in
/// full: without a spatial index every point has to be decoded to be tested).
pub fn read_las_where(path: impl AsRef<Path>, keep: impl FnMut(&Point) -> bool) -> Result<PointCloud> {
    read_las_impl(path.as_ref(), keep, false)
}

/// `reserve_all` sizes the columns for every point up front (right when all
/// are kept); otherwise they grow as points are kept.
fn read_las_impl(path: &Path, mut keep: impl FnMut(&Point) -> bool, reserve_all: bool) -> Result<PointCloud> {
    let mut reader = Reader::from_path(path)?;
    let header = reader.header().clone();
    let total = header.number_of_points();
    let mut cols = Columns::new(&header, if reserve_all { total as usize } else { 0 });
    let mut left = total;
    while left > 0 {
        let pd = reader.read_points(left.min(READ_BATCH))?;
        if pd.is_empty() {
            break;
        }
        left -= pd.len() as u64;
        for p in pd.points() {
            let p = p?;
            if keep(&[p.x, p.y, p.z]) {
                cols.push(&p);
            }
        }
    }
    Ok(cols.into_cloud())
}

/// Stream a LAS/LAZ file in batches of up to `batch` points, handing each to
/// `f` as a cloud with the attributes of [`read_las`], in file order. Only
/// one batch is held in memory at a time.
///
/// `f` is a callback: this function reads a batch, hands it over, and the
/// memory is reused for the next one, so a survey far larger than memory can
/// be processed a piece at a time. `FnMut` means `f` may keep and update
/// state of its own between batches - a running total, a writer - and the
/// `?` on the call means a batch that fails stops the read there.
pub fn read_las_batches(path: impl AsRef<Path>, batch: u64, mut f: impl FnMut(PointCloud) -> Result<()>) -> Result<()> {
    let mut reader = Reader::from_path(path.as_ref())?;
    let header = reader.header().clone();
    let batch = batch.max(1);
    let mut left = header.number_of_points();
    while left > 0 {
        let pd = reader.read_points(left.min(batch))?;
        if pd.is_empty() {
            break;
        }
        left -= pd.len() as u64;
        let mut cols = Columns::new(&header, pd.len());
        for p in pd.points() {
            cols.push(&p?);
        }
        f(cols.into_cloud())?;
    }
    Ok(())
}

/// The columns of [`read_las`], filled one point at a time.
struct Columns {
    extra_dims: Vec<ExtraDim>,
    xyz: Vec<Point>,
    intensity: Vec<u16>,
    return_number: Vec<u8>,
    number_of_returns: Vec<u8>,
    classification: Vec<u8>,
    scan_angle: Vec<f32>,
    user_data: Vec<u8>,
    point_source_id: Vec<u16>,
    gps_time: Option<Vec<f64>>,
    color: Option<(Vec<u16>, Vec<u16>, Vec<u16>)>,
    extra: Vec<Vec<f64>>,
}

impl Columns {
    fn new(header: &las::Header, n: usize) -> Columns {
        let format = header.point_format();
        let extra_dims: Vec<ExtraDim> = header
            .all_vlrs()
            .filter(|v| v.user_id == EXTRA_BYTES_USER_ID && v.record_id == EXTRA_BYTES_RECORD_ID)
            .flat_map(parse_extra_bytes_vlr)
            .collect();
        Columns {
            extra: extra_dims.iter().map(|_| Vec::with_capacity(n)).collect(),
            extra_dims,
            xyz: Vec::with_capacity(n),
            intensity: Vec::with_capacity(n),
            return_number: Vec::with_capacity(n),
            number_of_returns: Vec::with_capacity(n),
            classification: Vec::with_capacity(n),
            scan_angle: Vec::with_capacity(n),
            user_data: Vec::with_capacity(n),
            point_source_id: Vec::with_capacity(n),
            gps_time: if format.has_gps_time { Some(Vec::with_capacity(n)) } else { None },
            color: if format.has_color { Some((Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n))) } else { None },
        }
    }

    fn push(&mut self, p: &LasPoint) {
        self.xyz.push([p.x, p.y, p.z]);
        self.intensity.push(p.intensity);
        self.return_number.push(p.return_number);
        self.number_of_returns.push(p.number_of_returns);
        self.classification.push(u8::from(p.classification));
        self.scan_angle.push(p.scan_angle);
        self.user_data.push(p.user_data);
        self.point_source_id.push(p.point_source_id);
        if let (Some(v), Some(t)) = (&mut self.gps_time, p.gps_time) {
            v.push(t);
        }
        if let (Some((r, g, b)), Some(c)) = (&mut self.color, p.color) {
            r.push(c.red);
            g.push(c.green);
            b.push(c.blue);
        }
        let mut off = 0;
        for (k, dim) in self.extra_dims.iter().enumerate() {
            if off + dim.size <= p.extra_bytes.len() {
                self.extra[k].push(decode_dim(dim, &p.extra_bytes[off..off + dim.size]));
            }
            off += dim.size;
        }
    }

    fn into_cloud(self) -> PointCloud {
        let mut cloud = PointCloud::new(self.xyz);
        cloud.attrs.insert("intensity".into(), Attr::U16(self.intensity));
        cloud.attrs.insert("return_number".into(), Attr::U8(self.return_number));
        cloud.attrs.insert("number_of_returns".into(), Attr::U8(self.number_of_returns));
        cloud.attrs.insert("classification".into(), Attr::U8(self.classification));
        cloud.attrs.insert("scan_angle".into(), Attr::F32(self.scan_angle));
        cloud.attrs.insert("user_data".into(), Attr::U8(self.user_data));
        cloud.attrs.insert("point_source_id".into(), Attr::U16(self.point_source_id));
        if let Some(t) = self.gps_time {
            cloud.attrs.insert("gps_time".into(), Attr::F64(t));
        }
        if let Some((r, g, b)) = self.color {
            cloud.attrs.insert("red".into(), Attr::U16(r));
            cloud.attrs.insert("green".into(), Attr::U16(g));
            cloud.attrs.insert("blue".into(), Attr::U16(b));
        }
        for (dim, values) in self.extra_dims.iter().zip(self.extra) {
            if dim.data_type != 0 && values.len() == cloud.len() {
                cloud.attrs.insert(dim.name.clone(), attr_from_dim(dim, values));
            }
        }
        cloud
    }
}

#[derive(Debug, Clone)]
pub struct LasWriteOptions {
    pub point_format: u8,
    pub scale: f64,
    /// CRS to store as an OGC WKT VLR (see [`crate::geo::crs::Crs::to_wkt`]).
    pub crs_wkt: Option<String>,
}

impl Default for LasWriteOptions {
    fn default() -> Self {
        LasWriteOptions { point_format: 6, scale: 0.001, crs_wkt: None }
    }
}

const STANDARD: &[&str] = &[
    "intensity", "return_number", "number_of_returns", "classification", "scan_angle", "user_data",
    "point_source_id", "gps_time", "red", "green", "blue",
];

fn extra_dim_for(attr: &Attr) -> (u8, usize) {
    match attr {
        Attr::U8(_) | Attr::Bool(_) => (1, 1),
        Attr::I8(_) => (2, 1),
        Attr::U16(_) => (3, 2),
        Attr::I32(_) => (6, 4),
        Attr::U32(_) => (5, 4),
        Attr::I64(_) => (8, 8),
        Attr::F32(_) => (9, 4),
        Attr::F64(_) => (10, 8),
    }
}

fn encode_dim(attr: &Attr, i: usize, out: &mut Vec<u8>) {
    match attr {
        Attr::U8(v) => out.push(v[i]),
        Attr::Bool(v) => out.push(v[i] as u8),
        Attr::I8(v) => out.push(v[i] as u8),
        Attr::U16(v) => out.extend_from_slice(&v[i].to_le_bytes()),
        Attr::I32(v) => out.extend_from_slice(&v[i].to_le_bytes()),
        Attr::U32(v) => out.extend_from_slice(&v[i].to_le_bytes()),
        Attr::I64(v) => out.extend_from_slice(&v[i].to_le_bytes()),
        Attr::F32(v) => out.extend_from_slice(&v[i].to_le_bytes()),
        Attr::F64(v) => out.extend_from_slice(&v[i].to_le_bytes()),
    }
}

/// Write a LAS/LAZ file (compression chosen from the extension). Non-standard
/// attributes are stored as extra-bytes dimensions.
pub fn write_las(cloud: &PointCloud, path: impl AsRef<Path>, opts: &LasWriteOptions) -> Result<()> {
    write_las_with_vlrs(cloud, path, opts, &[])
}

/// [`write_las`], also storing `vlrs` in the header: the coordinate system
/// records of the file a cloud was read from, say, so that they carry over.
/// A WKT CRS record sets the header's WKT flag.
pub fn write_las_with_vlrs(cloud: &PointCloud, path: impl AsRef<Path>, opts: &LasWriteOptions, vlrs: &[Vlr]) -> Result<()> {
    let path = path.as_ref();
    let mut builder = Builder::from((1, 4));
    let mut format = Format::new(opts.point_format)?;
    if format.has_color && !cloud.attrs.contains_key("red") {
        // fine: colours default to zero
    }
    let extras: Vec<(&String, &Attr, u8, usize)> = cloud
        .attrs
        .iter()
        .filter(|(k, _)| !STANDARD.contains(&k.as_str()))
        .map(|(k, a)| {
            let (t, s) = extra_dim_for(a);
            (k, a, t, s)
        })
        .collect();
    format.extra_bytes = extras.iter().map(|e| e.3 as u16).sum();
    format.is_compressed = path.extension().map(|e| e.eq_ignore_ascii_case("laz")).unwrap_or(false);
    builder.point_format = format;

    if !extras.is_empty() {
        let mut data = Vec::with_capacity(192 * extras.len());
        for (name, _, dtype, _) in &extras {
            let mut rec = [0u8; 192];
            rec[2] = *dtype;
            let bytes = name.as_bytes();
            let n = bytes.len().min(31);
            rec[4..4 + n].copy_from_slice(&bytes[..n]);
            let desc = b"sylva";
            rec[160..160 + desc.len()].copy_from_slice(desc);
            data.extend_from_slice(&rec);
        }
        builder.vlrs.push(Vlr {
            user_id: EXTRA_BYTES_USER_ID.to_string(),
            record_id: EXTRA_BYTES_RECORD_ID,
            description: "Extra Bytes".to_string(),
            data,
        });
    }

    if let Some(wkt) = opts.crs_wkt.as_deref().filter(|w| !w.trim().is_empty()) {
        // LAS 1.4 R15: the WKT is a null-terminated string in a LASF_Projection
        // record 2112, with the global-encoding WKT bit set.
        let mut data = wkt.trim().as_bytes().to_vec();
        data.push(0);
        builder.vlrs.push(Vlr { user_id: "LASF_Projection".to_string(), record_id: 2112, description: "OGC WKT".to_string(), data });
        builder.has_wkt_crs = true;
    }

    let (lo, _) = cloud.bounds().unwrap_or(([0.0; 3], [0.0; 3]));
    builder.transforms = Vector {
        x: LasTransform { scale: opts.scale, offset: lo[0].floor() },
        y: LasTransform { scale: opts.scale, offset: lo[1].floor() },
        z: LasTransform { scale: opts.scale, offset: lo[2].floor() },
    };
    builder.generating_software = format!("sylva {}", env!("CARGO_PKG_VERSION"));
    for v in vlrs {
        builder.has_wkt_crs |= v.is_wkt_crs();
        builder.vlrs.push(v.clone());
    }
    let header = builder.into_header()?;
    let has_gps = header.point_format().has_gps_time;
    let has_color = header.point_format().has_color;
    let mut writer = Writer::from_path(path, header)?;

    let get = |name: &str| cloud.attrs.get(name);
    let intensity = get("intensity");
    let return_number = get("return_number");
    let number_of_returns = get("number_of_returns");
    let classification = get("classification");
    let scan_angle = get("scan_angle");
    let user_data = get("user_data");
    let point_source_id = get("point_source_id");
    let gps_time = get("gps_time");
    let (red, green, blue) = (get("red"), get("green"), get("blue"));

    for i in 0..cloud.len() {
        let mut p = LasPoint { x: cloud.xyz[i][0], y: cloud.xyz[i][1], z: cloud.xyz[i][2], ..Default::default() };
        if let Some(a) = intensity {
            p.intensity = a.get_f64(i) as u16;
        }
        if let Some(a) = return_number {
            p.return_number = (a.get_f64(i) as u8).clamp(0, 15);
        }
        if let Some(a) = number_of_returns {
            p.number_of_returns = (a.get_f64(i) as u8).clamp(0, 15);
        }
        if let Some(a) = classification {
            p.classification = Classification::new(a.get_f64(i) as u8).unwrap_or_default();
        }
        if let Some(a) = scan_angle {
            p.scan_angle = a.get_f64(i) as f32;
        }
        if let Some(a) = user_data {
            p.user_data = a.get_f64(i) as u8;
        }
        if let Some(a) = point_source_id {
            p.point_source_id = a.get_f64(i) as u16;
        }
        if has_gps {
            p.gps_time = Some(gps_time.map(|a| a.get_f64(i)).unwrap_or(0.0));
        }
        if has_color {
            p.color = Some(las::Color {
                red: red.map(|a| a.get_f64(i) as u16).unwrap_or(0),
                green: green.map(|a| a.get_f64(i) as u16).unwrap_or(0),
                blue: blue.map(|a| a.get_f64(i) as u16).unwrap_or(0),
            });
        }
        let mut eb = Vec::with_capacity(extras.iter().map(|e| e.3).sum());
        for (_, attr, _, _) in &extras {
            encode_dim(attr, i, &mut eb);
        }
        p.extra_bytes = eb;
        writer.write_point(p)?;
    }
    writer.close()?;
    Ok(())
}

pub(crate) fn _unused(_: Error) {}
