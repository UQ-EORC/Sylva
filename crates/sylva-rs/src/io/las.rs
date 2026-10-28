// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! LAS / LAZ via the `las` crate, with typed extra-bytes dimensions
//! (ASPRS LAS 1.4 R15 specification; LAZ is Isenburg's 2013 LASzip).

use std::collections::BTreeMap;
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
    read_las_impl(path.as_ref(), |_| true, true, None)
}

/// [`read_las`] keeping only the attributes named in `keep` (None keeps
/// them all; names the file does not have are ignored). The other columns
/// are never allocated, so a cloud that is processed as coordinates alone
/// costs its coordinates alone.
pub fn read_las_attrs(path: impl AsRef<Path>, keep: Option<&[&str]>) -> Result<PointCloud> {
    read_las_impl(path.as_ref(), |_| true, true, keep)
}

/// Points read per batch by [`read_las_where`].
const READ_BATCH: u64 = 1 << 20;

/// Read the points of a LAS/LAZ file for which `keep(&[x, y, z])` is true,
/// with the attributes of [`read_las`]. The file is streamed in batches, so
/// only the kept points are held in memory (LAZ is still decompressed in
/// full: without a spatial index every point has to be decoded to be tested).
pub fn read_las_where(path: impl AsRef<Path>, keep: impl FnMut(&Point) -> bool) -> Result<PointCloud> {
    read_las_impl(path.as_ref(), keep, false, None)
}

/// `reserve_all` sizes the columns for every point up front (right when all
/// are kept); otherwise they grow as points are kept. `attrs` names the
/// attributes to keep, None for all.
fn read_las_impl(path: &Path, mut keep: impl FnMut(&Point) -> bool, reserve_all: bool, attrs: Option<&[&str]>) -> Result<PointCloud> {
    let mut reader = Reader::from_path(path)?;
    let header = reader.header().clone();
    let total = header.number_of_points();
    let mut cols = Columns::new(&header, if reserve_all { total as usize } else { 0 }, attrs);
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
        let mut cols = Columns::new(&header, pd.len(), None);
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
    want: Wanted,
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

/// Which columns a read keeps (the coordinates always are).
struct Wanted {
    intensity: bool,
    return_number: bool,
    number_of_returns: bool,
    classification: bool,
    scan_angle: bool,
    user_data: bool,
    point_source_id: bool,
    gps_time: bool,
    color: bool,
    extra: Vec<bool>,
}

impl Columns {
    /// Columns for `n` points, keeping the attributes named in `attrs`
    /// (None for all). A column not kept is never allocated.
    fn new(header: &las::Header, n: usize, attrs: Option<&[&str]>) -> Columns {
        let format = header.point_format();
        let extra_dims: Vec<ExtraDim> = header
            .all_vlrs()
            .filter(|v| v.user_id == EXTRA_BYTES_USER_ID && v.record_id == EXTRA_BYTES_RECORD_ID)
            .flat_map(parse_extra_bytes_vlr)
            .collect();
        let wanted = |name: &str| attrs.is_none_or(|a| a.contains(&name));
        let want = Wanted {
            intensity: wanted("intensity"),
            return_number: wanted("return_number"),
            number_of_returns: wanted("number_of_returns"),
            classification: wanted("classification"),
            scan_angle: wanted("scan_angle"),
            user_data: wanted("user_data"),
            point_source_id: wanted("point_source_id"),
            gps_time: format.has_gps_time && wanted("gps_time"),
            color: format.has_color && (wanted("red") || wanted("green") || wanted("blue")),
            extra: extra_dims.iter().map(|d| wanted(&d.name)).collect(),
        };
        fn col<T>(keep: bool, n: usize) -> Vec<T> {
            if keep { Vec::with_capacity(n) } else { Vec::new() }
        }
        Columns {
            extra: want.extra.iter().map(|&w| col(w, n)).collect(),
            extra_dims,
            xyz: Vec::with_capacity(n),
            intensity: col(want.intensity, n),
            return_number: col(want.return_number, n),
            number_of_returns: col(want.number_of_returns, n),
            classification: col(want.classification, n),
            scan_angle: col(want.scan_angle, n),
            user_data: col(want.user_data, n),
            point_source_id: col(want.point_source_id, n),
            gps_time: if want.gps_time { Some(Vec::with_capacity(n)) } else { None },
            color: if want.color { Some((Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n))) } else { None },
            want,
        }
    }

    fn push(&mut self, p: &LasPoint) {
        self.xyz.push([p.x, p.y, p.z]);
        if self.want.intensity {
            self.intensity.push(p.intensity);
        }
        if self.want.return_number {
            self.return_number.push(p.return_number);
        }
        if self.want.number_of_returns {
            self.number_of_returns.push(p.number_of_returns);
        }
        if self.want.classification {
            self.classification.push(u8::from(p.classification));
        }
        if self.want.scan_angle {
            self.scan_angle.push(p.scan_angle);
        }
        if self.want.user_data {
            self.user_data.push(p.user_data);
        }
        if self.want.point_source_id {
            self.point_source_id.push(p.point_source_id);
        }
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
            if self.want.extra[k] && off + dim.size <= p.extra_bytes.len() {
                self.extra[k].push(decode_dim(dim, &p.extra_bytes[off..off + dim.size]));
            }
            off += dim.size;
        }
    }

    fn into_cloud(self) -> PointCloud {
        let mut cloud = PointCloud::new(self.xyz);
        let want = &self.want;
        let mut put = |keep: bool, name: &str, attr: Attr| {
            if keep {
                cloud.attrs.insert(name.into(), attr);
            }
        };
        put(want.intensity, "intensity", Attr::U16(self.intensity));
        put(want.return_number, "return_number", Attr::U8(self.return_number));
        put(want.number_of_returns, "number_of_returns", Attr::U8(self.number_of_returns));
        put(want.classification, "classification", Attr::U8(self.classification));
        put(want.scan_angle, "scan_angle", Attr::F32(self.scan_angle));
        put(want.user_data, "user_data", Attr::U8(self.user_data));
        put(want.point_source_id, "point_source_id", Attr::U16(self.point_source_id));
        if let Some(t) = self.gps_time {
            cloud.attrs.insert("gps_time".into(), Attr::F64(t));
        }
        if let Some((r, g, b)) = self.color {
            cloud.attrs.insert("red".into(), Attr::U16(r));
            cloud.attrs.insert("green".into(), Attr::U16(g));
            cloud.attrs.insert("blue".into(), Attr::U16(b));
        }
        for ((dim, values), &keep) in self.extra_dims.iter().zip(self.extra).zip(&want.extra) {
            if keep && dim.data_type != 0 && values.len() == cloud.len() {
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
    let (lo, _) = cloud.bounds().unwrap_or(([0.0; 3], [0.0; 3]));
    let mut writer = LasWriter::create(path, opts, vlrs, &extra_schema(&cloud.attrs), lo)?;
    writer.write(cloud)?;
    writer.close()
}

/// Write the points of `src` (a LAS/LAZ file) to `path` with the columns
/// of `extra` added, one value per point of `src` in file order, a batch at
/// a time. A cloud that was processed as coordinates alone gets its labels
/// this way without the whole file, attributes and all, being held in
/// memory: the output is that of [`write_las`] on the file's cloud with
/// `extra` inserted into its attributes. `lo` is the minimum corner the
/// header's offsets are taken from, as [`write_las`] takes it from the
/// cloud's bounds; None takes the source header's.
pub fn write_las_annotated(src: impl AsRef<Path>, path: impl AsRef<Path>, opts: &LasWriteOptions, extra: &BTreeMap<String, Attr>, lo: Option<Point>) -> Result<()> {
    let src = src.as_ref();
    let header = Reader::from_path(src)?.header().clone();
    let n = header.number_of_points() as usize;
    for (k, a) in extra {
        if a.len() != n {
            return Err(Error::Invalid(format!("{k} has {} values for the {n} points of {}", a.len(), src.display())));
        }
    }
    let lo = lo.unwrap_or_else(|| {
        let b = header.bounds();
        if n == 0 { [0.0; 3] } else { [b.min.x, b.min.y, b.min.z] }
    });
    // The attribute set of the file, from its header, with `extra` over it.
    let mut attrs = Columns::new(&header, 0, None).into_cloud().attrs;
    for (k, a) in extra {
        attrs.insert(k.clone(), a.slice(0..0));
    }
    let mut writer = LasWriter::create(path, opts, &[], &extra_schema(&attrs), lo)?;
    let mut off = 0;
    read_las_batches(src, READ_BATCH, |mut batch| {
        let m = batch.len();
        for (k, a) in extra {
            batch.attrs.insert(k.clone(), a.slice(off..off + m));
        }
        off += m;
        writer.write(&batch)
    })?;
    writer.close()
}

/// The extra-bytes dimensions a cloud's attributes take: the non-standard
/// ones in name order, each with its LAS type code and width.
pub fn extra_schema(attrs: &BTreeMap<String, Attr>) -> Vec<(String, u8, usize)> {
    attrs
        .iter()
        .filter(|(k, _)| !STANDARD.contains(&k.as_str()))
        .map(|(k, a)| {
            let (t, size) = extra_dim_for(a);
            (k.clone(), t, size)
        })
        .collect()
}

/// A LAS/LAZ file being written one cloud at a time, so that a cloud larger
/// than memory is written in pieces. The header - point format, scale and
/// offset, the extra-bytes record - is fixed when the writer is created,
/// and every cloud written must carry the extra attributes it names, with
/// the types it gave them.
pub struct LasWriter {
    writer: Writer<std::io::BufWriter<std::fs::File>>,
    extras: Vec<(String, u8, usize)>,
    has_gps: bool,
    has_color: bool,
}

impl LasWriter {
    /// Open `path` with a header for `opts`, the extra `vlrs`, the extra-bytes
    /// dimensions `extras` ([`extra_schema`]) and coordinate offsets at the
    /// floor of `lo`.
    pub fn create(path: impl AsRef<Path>, opts: &LasWriteOptions, vlrs: &[Vlr], extras: &[(String, u8, usize)], lo: Point) -> Result<LasWriter> {
        let path = path.as_ref();
        let mut builder = Builder::from((1, 4));
        let mut format = Format::new(opts.point_format)?;
        format.extra_bytes = extras.iter().map(|e| e.2 as u16).sum();
        format.is_compressed = path.extension().map(|e| e.eq_ignore_ascii_case("laz")).unwrap_or(false);
        builder.point_format = format;

        if !extras.is_empty() {
            let mut data = Vec::with_capacity(192 * extras.len());
            for (name, dtype, _) in extras {
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
        let writer = Writer::from_path(path, header)?;
        Ok(LasWriter { writer, extras: extras.to_vec(), has_gps, has_color })
    }

    /// Append every point of `cloud`.
    pub fn write(&mut self, cloud: &PointCloud) -> Result<()> {
        let mut extra_attrs: Vec<&Attr> = Vec::with_capacity(self.extras.len());
        for (name, dtype, size) in &self.extras {
            let a = cloud.attrs.get(name).ok_or_else(|| Error::Invalid(format!("attribute {name} missing from a cloud written to the file")))?;
            if extra_dim_for(a) != (*dtype, *size) {
                return Err(Error::Invalid(format!("attribute {name} is {} in a cloud written to the file, not the type the header gives it", a.dtype())));
            }
            extra_attrs.push(a);
        }
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
        let eb_len: usize = self.extras.iter().map(|e| e.2).sum();

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
            if self.has_gps {
                p.gps_time = Some(gps_time.map(|a| a.get_f64(i)).unwrap_or(0.0));
            }
            if self.has_color {
                p.color = Some(las::Color {
                    red: red.map(|a| a.get_f64(i) as u16).unwrap_or(0),
                    green: green.map(|a| a.get_f64(i) as u16).unwrap_or(0),
                    blue: blue.map(|a| a.get_f64(i) as u16).unwrap_or(0),
                });
            }
            let mut eb = Vec::with_capacity(eb_len);
            for attr in &extra_attrs {
                encode_dim(attr, i, &mut eb);
            }
            p.extra_bytes = eb;
            self.writer.write_point(p)?;
        }
        Ok(())
    }

    /// Finish the file (the header's point count and bounds are written now).
    pub fn close(mut self) -> Result<()> {
        self.writer.close()?;
        Ok(())
    }
}

/// One LAS/LAZ file from several, streamed file by file, so a cloud larger
/// than memory is assembled from its tiles. Every input must carry the same
/// attributes (names and types) as the first; the header's offset is the
/// floor of the union of the inputs' bounds.
pub fn write_las_concat(inputs: &[std::path::PathBuf], path: impl AsRef<Path>, opts: &LasWriteOptions) -> Result<()> {
    let Some(first_path) = inputs.first() else {
        return Err(Error::Invalid("no input files".to_string()));
    };
    let mut lo = [f64::INFINITY; 3];
    for f in inputs {
        let b = Reader::from_path(f)?.header().bounds();
        lo = [lo[0].min(b.min.x), lo[1].min(b.min.y), lo[2].min(b.min.z)];
    }
    let first = read_las(first_path)?;
    let mut writer = LasWriter::create(path, opts, &[], &extra_schema(&first.attrs), lo)?;
    writer.write(&first)?;
    for f in &inputs[1..] {
        let c = read_las(f)?;
        for (k, a) in &first.attrs {
            match c.attrs.get(k) {
                Some(b) if b.dtype() == a.dtype() => {}
                _ => return Err(Error::Invalid(format!("{}: attribute {k} differs from {}", f.display(), first_path.display()))),
            }
        }
        writer.write(&c)?;
    }
    writer.close()
}

pub(crate) fn _unused(_: Error) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory under the system's, named for the test.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(name: &str) -> TempDir {
            let p = std::env::temp_dir().join(format!("sylva-las-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn cloud(n: usize) -> PointCloud {
        let mut c = PointCloud::new((0..n).map(|i| [i as f64 * 0.37 - 3.0, (i % 7) as f64 * 1.5, (i * i % 11) as f64 * 0.1]).collect());
        c.attrs.insert("intensity".into(), Attr::U16((0..n).map(|i| (i * 13 % 65536) as u16).collect()));
        c.attrs.insert("return_number".into(), Attr::U8((0..n).map(|i| (i % 3 + 1) as u8).collect()));
        c.attrs.insert("number_of_returns".into(), Attr::U8(vec![3; n]));
        c.attrs.insert("classification".into(), Attr::U8((0..n).map(|i| if i % 5 == 0 { 2 } else { 1 }).collect()));
        c.attrs.insert("gps_time".into(), Attr::F64((0..n).map(|i| 1e5 + i as f64 * 1e-4).collect()));
        c.attrs.insert("amplitude".into(), Attr::F32((0..n).map(|i| i as f32 * 0.01).collect()));
        c.attrs.insert("deviation".into(), Attr::U16((0..n).map(|i| (i % 50) as u16).collect()));
        c
    }

    #[test]
    fn annotated_streaming_write_is_the_whole_cloud_write() {
        let dir = TempDir::new("annotated");
        let n = 50_000;
        let c = cloud(n);
        let src = dir.path().join("src.laz");
        write_las(&c, &src, &LasWriteOptions::default()).unwrap();
        let labels = Attr::I32((0..n).map(|i| (i % 9) as i32 - 1).collect());
        let mut whole = read_las(&src).unwrap();
        whole.attrs.insert("tree_id".into(), labels.clone());
        let a = dir.path().join("whole.laz");
        write_las(&whole, &a, &LasWriteOptions { crs_wkt: Some("LOCAL_CS[\"x\"]".into()), ..Default::default() }).unwrap();
        let b = dir.path().join("streamed.laz");
        let extra: BTreeMap<String, Attr> = [("tree_id".to_string(), labels)].into_iter().collect();
        let (lo, _) = whole.bounds().unwrap();
        write_las_annotated(&src, &b, &LasWriteOptions { crs_wkt: Some("LOCAL_CS[\"x\"]".into()), ..Default::default() }, &extra, Some(lo)).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap(), "streamed file differs from the whole-cloud one");
        // The source header's minimum is the cloud's, so None gives the same file.
        let d = dir.path().join("streamed2.laz");
        write_las_annotated(&src, &d, &LasWriteOptions { crs_wkt: Some("LOCAL_CS[\"x\"]".into()), ..Default::default() }, &extra, None).unwrap();
        assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&d).unwrap());
        let wrong: BTreeMap<String, Attr> = [("tree_id".to_string(), Attr::I32(vec![0; 3]))].into_iter().collect();
        assert!(write_las_annotated(&src, dir.path().join("bad.laz"), &LasWriteOptions::default(), &wrong, None).is_err());
    }

    #[test]
    fn batches_cross_the_streamed_write() {
        // Points beyond one READ_BATCH, so the annotated write spans batches.
        let dir = TempDir::new("batches");
        let n = READ_BATCH as usize + 1000;
        let mut c = PointCloud::new((0..n).map(|i| [(i % 1000) as f64 * 0.01, (i / 1000) as f64 * 0.01, 0.0]).collect());
        c.attrs.insert("intensity".into(), Attr::U16((0..n).map(|i| (i % 65536) as u16).collect()));
        let src = dir.path().join("src.las");
        write_las(&c, &src, &LasWriteOptions::default()).unwrap();
        let labels = Attr::I32((0..n).map(|i| (i % 1_000_003) as i32).collect());
        let extra: BTreeMap<String, Attr> = [("tree_id".to_string(), labels.clone())].into_iter().collect();
        let out = dir.path().join("out.las");
        write_las_annotated(&src, &out, &LasWriteOptions::default(), &extra, None).unwrap();
        let back = read_las(&out).unwrap();
        assert_eq!(back.len(), n);
        assert_eq!(back.attrs["tree_id"], labels);
        assert_eq!(back.attrs["intensity"], c.attrs["intensity"]);
    }

    #[test]
    fn reading_some_attributes_allocates_only_those() {
        let dir = TempDir::new("attrs");
        let c = cloud(1000);
        let src = dir.path().join("c.laz");
        write_las(&c, &src, &LasWriteOptions::default()).unwrap();
        let all = read_las(&src).unwrap();
        let some = read_las_attrs(&src, Some(&["amplitude", "classification", "height", "red"])).unwrap();
        assert_eq!(some.xyz, all.xyz);
        assert_eq!(some.attrs.keys().cloned().collect::<Vec<_>>(), ["amplitude", "classification"]);
        assert_eq!(some.attrs["amplitude"], all.attrs["amplitude"]);
        assert_eq!(some.attrs["classification"], all.attrs["classification"]);
        let none = read_las_attrs(&src, Some(&[])).unwrap();
        assert!(none.attrs.is_empty() && none.len() == 1000);
        assert_eq!(read_las_attrs(&src, None).unwrap(), all);
    }
}
