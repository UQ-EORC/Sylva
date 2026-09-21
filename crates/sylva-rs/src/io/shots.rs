//! Native pulse storage: [`Shots`] as a Parquet file.
//!
//! One row per pulse, in row groups that can be read independently:
//!
//! | column | type | |
//! |---|---|---|
//! | `scan` | int32 | index into the scanner positions in the file metadata |
//! | `zenith`, `azimuth` | float / double | beam direction (rad; zenith from +z, azimuth `atan2(x, y)`) |
//! | `range` | list&lt;float / double&gt; | echo ranges, nearest first; empty for a pulse without a return |
//! | *attr* | list&lt;typed&gt; | one list column per echo attribute |
//!
//! With more than 65 536 distinct origins (mobile or airborne platforms) the
//! `scan` column is replaced by `origin_x`, `origin_y`, `origin_z`: int32 in
//! units of `sylva.origin_scale` (0.1 mm, or 1 µm in double precision, unless the
//! platform travels too far for that) from `sylva.origin_offset`, delta encoded, so a smooth trajectory
//! costs a few bits per pulse.
//! A pulse without an echo costs its two angles and nothing else, so there is
//! no need for far "sky" points, and an echo is a range rather than three
//! coordinates. In single precision (the default) echo positions are
//! reproduced to about 0.01 mm per 100 m of range.
//!
//! File metadata keys: `sylva.shots` (format version), `sylva.precision`,
//! `sylva.scans` (`x y z;x y z;...`), `sylva.origin_offset`, `sylva.origin_scale`, `sylva.attrs` (`name:dtype,...`),
//! `sylva.bounds` (echo bounding box, `x0 y0 z0 x1 y1 z1`), `sylva.n_echoes`.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::column::reader::ColumnReader;
use parquet::data_type::{BoolType, DataType, DoubleType, FloatType, Int32Type, Int64Type};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
use parquet::schema::parser::parse_message_type;

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::{Point, Shots};

const VERSION: &str = "1";
const MAX_SCANS: usize = 1 << 16;

impl From<parquet::errors::ParquetError> for Error {
    fn from(e: parquet::errors::ParquetError) -> Self {
        Error::invalid(format!("parquet: {e}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShotsWriteOptions {
    /// Store angles and ranges as `f64` (exact round trip) instead of `f32`.
    pub double: bool,
    /// Pulses per row group, the unit of streaming reads.
    pub row_group_size: usize,
    pub zstd_level: i32,
    /// Origins closer than this (m) are one scanner position, their mean.
    /// Ray clouds carry the origin of every ray with rounding noise (LAZ ray
    /// clouds store it as a float32 offset from the end point); echoes stay
    /// where they are, only the beam origin moves, by less than this. 0 keeps
    /// origins exact, which usually means storing one per pulse.
    pub origin_tolerance: f64,
}

impl Default for ShotsWriteOptions {
    fn default() -> Self {
        ShotsWriteOptions { double: false, row_group_size: 1 << 20, zstd_level: 3, origin_tolerance: 1e-3 }
    }
}

fn physical(a: &Attr) -> &'static str {
    match a {
        Attr::F64(_) => "double",
        Attr::F32(_) => "float",
        Attr::I64(_) => "int64",
        Attr::I32(_) => "int32",
        Attr::U32(_) => "int32 element (INTEGER(32,false))",
        Attr::U16(_) => "int32 element (INTEGER(16,false))",
        Attr::U8(_) => "int32 element (INTEGER(8,false))",
        Attr::I8(_) => "int32 element (INTEGER(8,true))",
        Attr::Bool(_) => "boolean",
    }
}

fn list_field(name: &str, element: &str) -> String {
    let element = if element.contains("element") { element.to_string() } else { format!("{element} element") };
    format!("  required group {name} (LIST) {{ repeated group list {{ required {element}; }} }}\n")
}

/// Beam direction to `(zenith, azimuth)`.
fn angles(d: &Point) -> (f64, f64) {
    (d[2].clamp(-1.0, 1.0).acos(), d[0].atan2(d[1]))
}

fn direction(zenith: f64, azimuth: f64) -> Point {
    let (sz, cz) = zenith.sin_cos();
    let (sa, ca) = azimuth.sin_cos();
    [sz * sa, sz * ca, cz]
}

fn write_col<T: DataType>(rg: &mut SerializedRowGroupWriter<'_, File>, values: &[T::T], levels: Option<(&[i16], &[i16])>) -> Result<()> {
    let mut col = rg.next_column()?.ok_or_else(|| Error::invalid("parquet: schema has fewer columns than the data"))?;
    col.typed::<T>().write_batch(values, levels.map(|l| l.0), levels.map(|l| l.1))?;
    col.close()?;
    Ok(())
}

/// Write `shots` to a Parquet file (see the module docs for the layout).
pub fn write_shots(shots: &Shots, path: impl AsRef<Path>, opts: &ShotsWriteOptions) -> Result<()> {
    let path = path.as_ref();
    let n = shots.n_shots();
    for name in shots.echo_attrs.keys() {
        let ok = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !ok || matches!(name.as_str(), "scan" | "zenith" | "azimuth" | "range" | "origin_x" | "origin_y" | "origin_z" | "list" | "element") {
            return Err(Error::invalid(format!("echo attribute {name:?} cannot be a column name")));
        }
    }

    // Scanner positions, in order of first use.
    let tol = opts.origin_tolerance.max(0.0);
    let key_of = |o: &Point| -> [u64; 3] {
        if tol > 0.0 {
            [(o[0] / tol).round() as i64 as u64, (o[1] / tol).round() as i64 as u64, (o[2] / tol).round() as i64 as u64]
        } else {
            [o[0].to_bits(), o[1].to_bits(), o[2].to_bits()]
        }
    };
    let mut sums: Vec<(Point, f64)> = Vec::new();
    // First origin of each position, and whether every member equals it.
    let mut first: Vec<(Point, bool)> = Vec::new();
    let mut scan_of: Vec<i32> = Vec::with_capacity(n);
    let mut seen: HashMap<[u64; 3], i32> = HashMap::new();
    for o in &shots.origin {
        let next = sums.len() as i32;
        let id = *seen.entry(key_of(o)).or_insert(next);
        if id == next {
            if sums.len() == MAX_SCANS {
                break;
            }
            sums.push(([0.0; 3], 0.0));
            first.push((*o, true));
        }
        first[id as usize].1 &= first[id as usize].0 == *o;
        let s = &mut sums[id as usize];
        for k in 0..3 {
            s.0[k] += o[k];
        }
        s.1 += 1.0;
        scan_of.push(id);
    }
    let per_shot_origin = scan_of.len() < n;
    let scans: Vec<Point> = sums.iter().zip(&first).map(|((s, c), (o, same))| if *same { *o } else { [s[0] / c, s[1] / c, s[2] / c] }).collect();

    // Beams re-aimed from their scanner position at the unchanged echoes.
    let snapped;
    let shots = if per_shot_origin || tol == 0.0 {
        shots
    } else {
        let mut out = shots.clone();
        for s in 0..n {
            let (o, d, scan) = (shots.origin[s], shots.direction[s], scans[scan_of[s] as usize]);
            let (e0, c) = (shots.echo_start[s], shots.echo_count[s] as usize);
            out.origin[s] = scan;
            for e in e0..e0 + c {
                let r = shots.echo_range[e];
                let v = [o[0] + d[0] * r - scan[0], o[1] + d[1] * r - scan[1], o[2] + d[2] * r - scan[2]];
                out.echo_range[e] = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if e + 1 == e0 + c && out.echo_range[e] > 0.0 {
                    out.direction[s] = [v[0] / out.echo_range[e], v[1] / out.echo_range[e], v[2] / out.echo_range[e]];
                }
            }
        }
        snapped = out;
        &snapped
    };

    let real = if opts.double { "double" } else { "float" };
    let mut schema = String::from("message sylva_shots {\n");
    if per_shot_origin {
        schema += "  required int32 origin_x;\n  required int32 origin_y;\n  required int32 origin_z;\n";
    } else {
        schema += "  required int32 scan;\n";
    }
    schema += &format!("  required {real} zenith;\n  required {real} azimuth;\n");
    schema += &list_field("range", real);
    for (name, a) in &shots.echo_attrs {
        schema += &list_field(name, physical(a));
    }
    schema += "}\n";
    let schema = Arc::new(parse_message_type(&schema)?);

    let xyz_bounds = {
        let xyz = shots.echo_xyz();
        (crate::spatial::min_corner(&xyz), crate::spatial::max_corner(&xyz))
    };
    let join = |p: &Point| format!("{} {} {}", p[0], p[1], p[2]);
    // Per-pulse origins are integers in units of `origin_scale` from the first
    // origin: along a trajectory they change slowly, so their deltas pack into a few bits.
    let origin_offset = shots.origin.first().copied().unwrap_or([0.0; 3]);
    let reach = shots.origin.iter().flat_map(|o| (0..3).map(move |k| (o[k] - origin_offset[k]).abs())).fold(0.0f64, f64::max);
    let origin_scale = (reach / 2.0e9).max(if opts.double { 1e-6 } else { 1e-4 });
    let kv = vec![
        KeyValue::new("sylva.origin_offset".into(), join(&origin_offset)),
        KeyValue::new("sylva.origin_scale".into(), origin_scale.to_string()),
        KeyValue::new("sylva.shots".into(), VERSION.to_string()),
        KeyValue::new("sylva.precision".into(), if opts.double { "double" } else { "single" }.to_string()),
        KeyValue::new("sylva.scans".into(), if per_shot_origin { String::new() } else { scans.iter().map(join).collect::<Vec<_>>().join(";") }),
        KeyValue::new("sylva.attrs".into(), shots.echo_attrs.iter().map(|(k, a)| format!("{k}:{}", a.dtype())).collect::<Vec<_>>().join(",")),
        KeyValue::new("sylva.bounds".into(), format!("{} {}", join(&xyz_bounds.0), join(&xyz_bounds.1))),
        KeyValue::new("sylva.n_echoes".into(), shots.n_echoes().to_string()),
    ];
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(opts.zstd_level)?))
        .set_key_value_metadata(Some(kv))
        // Dictionaries suit the scan index and integer attributes; floats split their bytes instead.
        .set_column_encoding("origin_x".into(), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled("origin_x".into(), false)
        .set_column_encoding("origin_y".into(), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled("origin_y".into(), false)
        .set_column_encoding("origin_z".into(), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled("origin_z".into(), false)
        .set_column_encoding("zenith".into(), Encoding::BYTE_STREAM_SPLIT)
        .set_column_dictionary_enabled("zenith".into(), false)
        .set_column_encoding("azimuth".into(), Encoding::BYTE_STREAM_SPLIT)
        .set_column_dictionary_enabled("azimuth".into(), false)
        .set_column_encoding("range.list.element".into(), Encoding::BYTE_STREAM_SPLIT)
        .set_column_dictionary_enabled("range.list.element".into(), false)
        .build();
    let file = File::create(path).map_err(|e| Error::file(path, e.to_string()))?;
    let mut writer = SerializedFileWriter::new(file, schema, Arc::new(props))?;

    let group = opts.row_group_size.max(1);
    for s0 in (0..n.max(1)).step_by(group) {
        let s1 = (s0 + group).min(n);
        let mut rg = writer.next_row_group()?;
        if per_shot_origin {
            for k in 0..3 {
                let v: Vec<i32> = shots.origin[s0..s1].iter().map(|o| ((o[k] - origin_offset[k]) / origin_scale).round() as i32).collect();
                write_col::<Int32Type>(&mut rg, &v, None)?;
            }
        } else {
            write_col::<Int32Type>(&mut rg, &scan_of[s0..s1], None)?;
        }
        for k in 0..2 {
            let v: Vec<f64> = shots.direction[s0..s1].iter().map(|d| if k == 0 { angles(d).0 } else { angles(d).1 }).collect();
            if opts.double {
                write_col::<DoubleType>(&mut rg, &v, None)?;
            } else {
                write_col::<FloatType>(&mut rg, &v.iter().map(|&x| x as f32).collect::<Vec<_>>(), None)?;
            }
        }
        // List levels, shared by every echo column: an empty list is one
        // entry at definition level 0; elements are at level 1, repeating after the first.
        let (e0, e1) = if s0 < n { (shots.echo_start[s0], shots.echo_start[s1 - 1] + shots.echo_count[s1 - 1] as usize) } else { (0, 0) };
        let mut def = Vec::with_capacity(e1 - e0);
        let mut rep = Vec::with_capacity(e1 - e0);
        for &c in &shots.echo_count[s0..s1] {
            if c == 0 {
                def.push(0);
                rep.push(0);
            }
            for j in 0..c {
                def.push(1);
                rep.push((j > 0) as i16);
            }
        }
        let lv = Some((&def[..], &rep[..]));
        if opts.double {
            write_col::<DoubleType>(&mut rg, &shots.echo_range[e0..e1], lv)?;
        } else {
            write_col::<FloatType>(&mut rg, &shots.echo_range[e0..e1].iter().map(|&x| x as f32).collect::<Vec<_>>(), lv)?;
        }
        for a in shots.echo_attrs.values() {
            let as_i32 = |it: &mut dyn Iterator<Item = i32>| it.collect::<Vec<i32>>();
            match a {
                Attr::F64(v) => write_col::<DoubleType>(&mut rg, &v[e0..e1], lv)?,
                Attr::F32(v) => write_col::<FloatType>(&mut rg, &v[e0..e1], lv)?,
                Attr::I64(v) => write_col::<Int64Type>(&mut rg, &v[e0..e1], lv)?,
                Attr::I32(v) => write_col::<Int32Type>(&mut rg, &v[e0..e1], lv)?,
                Attr::U32(v) => write_col::<Int32Type>(&mut rg, &as_i32(&mut v[e0..e1].iter().map(|&x| x as i32)), lv)?,
                Attr::U16(v) => write_col::<Int32Type>(&mut rg, &as_i32(&mut v[e0..e1].iter().map(|&x| x as i32)), lv)?,
                Attr::U8(v) => write_col::<Int32Type>(&mut rg, &as_i32(&mut v[e0..e1].iter().map(|&x| x as i32)), lv)?,
                Attr::I8(v) => write_col::<Int32Type>(&mut rg, &as_i32(&mut v[e0..e1].iter().map(|&x| x as i32)), lv)?,
                Attr::Bool(v) => write_col::<BoolType>(&mut rg, &v[e0..e1], lv)?,
            }
        }
        rg.close()?;
    }
    writer.close()?;
    Ok(())
}

/// An open shots file; row groups are read on demand.
pub struct ShotsFile {
    path: std::path::PathBuf,
    reader: SerializedFileReader<File>,
    scans: Vec<Point>,
    /// Origins are stored per pulse (`origin_x/y/z`) rather than as a `scan` index.
    per_pulse: bool,
    attrs: Vec<(String, String)>,
    double: bool,
    origin_offset: Point,
    origin_scale: f64,
    /// Bounding box of the echoes.
    pub bounds: (Point, Point),
    pub n_shots: usize,
    pub n_echoes: usize,
}

fn read_values<T: DataType>(col: ColumnReader, rows: usize, lists: bool, get: impl FnOnce(ColumnReader) -> Option<parquet::column::reader::ColumnReaderImpl<T>>) -> Result<(Vec<T::T>, Vec<i16>)> {
    let mut r = get(col).ok_or_else(|| Error::invalid("parquet: unexpected column type"))?;
    let (mut values, mut def, mut rep) = (Vec::new(), Vec::new(), Vec::new());
    let mut done = 0;
    while done < rows {
        let (records, _, _) = r.read_records(rows - done, lists.then_some(&mut def), lists.then_some(&mut rep), &mut values)?;
        if records == 0 {
            break;
        }
        done += records;
    }
    if done != rows {
        return Err(Error::invalid("parquet: column is shorter than its row group"));
    }
    Ok((values, rep.iter().zip(&def).map(|(&r, &d)| if r == 0 { d } else { 2 }).collect()))
}

macro_rules! reader_of {
    ($variant:ident) => {
        |c| match c {
            ColumnReader::$variant(r) => Some(r),
            _ => None,
        }
    };
}

impl ShotsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| Error::file(path, e.to_string()))?;
        let reader = SerializedFileReader::new(file)?;
        let meta = reader.metadata().file_metadata();
        let kv: BTreeMap<&str, &str> = meta.key_value_metadata().into_iter().flatten().map(|k| (k.key.as_str(), k.value.as_deref().unwrap_or(""))).collect();
        let bad = |what: &str| Error::file(path, format!("not a sylva shots file ({what})"));
        if kv.get("sylva.shots") != Some(&VERSION) {
            return Err(bad("missing or unknown sylva.shots version"));
        }
        let point = |s: &str| -> Option<Point> {
            let v: Vec<f64> = s.split_whitespace().map(|t| t.parse().ok()).collect::<Option<_>>()?;
            (v.len() == 3).then(|| [v[0], v[1], v[2]])
        };
        let scans: Vec<Point> = kv.get("sylva.scans").copied().unwrap_or("").split(';').filter(|s| !s.is_empty()).map(point).collect::<Option<_>>().ok_or_else(|| bad("scans"))?;
        let attrs: Vec<(String, String)> = kv.get("sylva.attrs").copied().unwrap_or("").split(',').filter(|s| !s.is_empty()).map(|s| s.split_once(':').map(|(a, b)| (a.to_string(), b.to_string()))).collect::<Option<_>>().ok_or_else(|| bad("attrs"))?;
        let b: Vec<f64> = kv.get("sylva.bounds").copied().unwrap_or("").split_whitespace().filter_map(|t| t.parse().ok()).collect();
        if b.len() != 6 {
            return Err(bad("bounds"));
        }
        // Read the storage mode from the schema: a file of zero shots has no
        // scans either, yet is written with the `scan` column.
        let per_pulse = meta.schema_descr().columns().iter().any(|c| c.name() == "origin_x");
        Ok(ShotsFile {
            path: path.to_path_buf(),
            scans,
            per_pulse,
            attrs,
            double: kv.get("sylva.precision") == Some(&"double"),
            origin_offset: kv.get("sylva.origin_offset").and_then(|s| point(s)).ok_or_else(|| bad("origin_offset"))?,
            origin_scale: kv.get("sylva.origin_scale").and_then(|s| s.parse().ok()).ok_or_else(|| bad("origin_scale"))?,
            bounds: ([b[0], b[1], b[2]], [b[3], b[4], b[5]]),
            n_shots: meta.num_rows() as usize,
            n_echoes: kv.get("sylva.n_echoes").and_then(|s| s.parse().ok()).ok_or_else(|| bad("n_echoes"))?,
            reader,
        })
    }

    /// A second handle on the same file. Handles share no file position, so
    /// each thread reading row groups needs its own.
    pub fn reopen(&self) -> Result<Self> {
        ShotsFile::open(&self.path)
    }

    pub fn n_groups(&self) -> usize {
        self.reader.num_row_groups()
    }

    /// Scanner positions (empty when origins are stored per pulse).
    pub fn scans(&self) -> &[Point] {
        &self.scans
    }

    pub fn attr_names(&self) -> impl Iterator<Item = &str> {
        self.attrs.iter().map(|a| a.0.as_str())
    }

    /// Read one row group.
    pub fn read_group(&self, group: usize) -> Result<Shots> {
        let rg = self.reader.get_row_group(group)?;
        let rows = rg.metadata().num_rows() as usize;
        let mut col = 0;
        let mut next = || {
            col += 1;
            rg.get_column_reader(col - 1)
        };
        let reals = |c: ColumnReader, lists: bool| -> Result<(Vec<f64>, Vec<i16>)> {
            if self.double {
                read_values::<DoubleType>(c, rows, lists, reader_of!(DoubleColumnReader))
            } else {
                let (v, l) = read_values::<FloatType>(c, rows, lists, reader_of!(FloatColumnReader))?;
                Ok((v.into_iter().map(|x| x as f64).collect(), l))
            }
        };
        let mut shots = Shots::default();
        if self.per_pulse {
            let mut xyz = Vec::new();
            for _ in 0..3 {
                xyz.push(read_values::<Int32Type>(next()?, rows, false, reader_of!(Int32ColumnReader))?.0);
            }
            let at = |k: usize, i: usize| self.origin_offset[k] + xyz[k][i] as f64 * self.origin_scale;
            shots.origin = (0..rows).map(|i| [at(0, i), at(1, i), at(2, i)]).collect();
        } else {
            let (ids, _) = read_values::<Int32Type>(next()?, rows, false, reader_of!(Int32ColumnReader))?;
            shots.origin = ids.iter().map(|&i| self.scans.get(i as usize).copied().ok_or_else(|| Error::invalid("shots file: scan index out of range"))).collect::<Result<_>>()?;
        }
        let zenith = reals(next()?, false)?.0;
        let azimuth = reals(next()?, false)?.0;
        shots.direction = zenith.iter().zip(&azimuth).map(|(&z, &a)| direction(z, a)).collect();

        // Levels: 0 = empty list, 1 = first element of a list, 2 = further element.
        let (range, levels) = reals(next()?, true)?;
        shots.echo_range = range;
        let mut at = 0;
        for &l in &levels {
            match l {
                2 => *shots.echo_count.last_mut().ok_or_else(|| Error::invalid("shots file: bad list levels"))? += 1,
                _ => {
                    shots.echo_start.push(at);
                    shots.echo_count.push(l as u32);
                }
            }
            at += (l > 0) as usize;
        }
        for (name, dtype) in &self.attrs {
            let c = next()?;
            let ints = |c| read_values::<Int32Type>(c, rows, true, reader_of!(Int32ColumnReader)).map(|v| v.0);
            let attr = match dtype.as_str() {
                "float64" => Attr::F64(read_values::<DoubleType>(c, rows, true, reader_of!(DoubleColumnReader))?.0),
                "float32" => Attr::F32(read_values::<FloatType>(c, rows, true, reader_of!(FloatColumnReader))?.0),
                "int64" => Attr::I64(read_values::<Int64Type>(c, rows, true, reader_of!(Int64ColumnReader))?.0),
                "int32" => Attr::I32(ints(c)?),
                "uint32" => Attr::U32(ints(c)?.into_iter().map(|x| x as u32).collect()),
                "uint16" => Attr::U16(ints(c)?.into_iter().map(|x| x as u16).collect()),
                "uint8" => Attr::U8(ints(c)?.into_iter().map(|x| x as u8).collect()),
                "int8" => Attr::I8(ints(c)?.into_iter().map(|x| x as i8).collect()),
                "bool" => Attr::Bool(read_values::<BoolType>(c, rows, true, reader_of!(BoolColumnReader))?.0),
                other => return Err(Error::invalid(format!("shots file: unknown dtype {other:?}"))),
            };
            if attr.len() != shots.echo_range.len() {
                return Err(Error::invalid(format!("shots file: attribute {name:?} does not match the echoes")));
            }
            shots.echo_attrs.insert(name.clone(), attr);
        }
        Ok(shots)
    }

    /// Read every row group into one [`Shots`].
    pub fn read_all(&self) -> Result<Shots> {
        self.read_groups(&(0..self.n_groups()).collect::<Vec<_>>())
    }

    /// Read some row groups into one [`Shots`].
    pub fn read_groups(&self, groups: &[usize]) -> Result<Shots> {
        let mut out = Shots::default();
        for (name, dtype) in &self.attrs {
            let empty = match dtype.as_str() {
                "float64" => Attr::F64(vec![]),
                "float32" => Attr::F32(vec![]),
                "int64" => Attr::I64(vec![]),
                "int32" => Attr::I32(vec![]),
                "uint32" => Attr::U32(vec![]),
                "uint16" => Attr::U16(vec![]),
                "uint8" => Attr::U8(vec![]),
                "int8" => Attr::I8(vec![]),
                _ => Attr::Bool(vec![]),
            };
            out.echo_attrs.insert(name.clone(), empty);
        }
        for &g in groups {
            if g >= self.n_groups() {
                return Err(Error::invalid(format!("shots file has {} row groups, asked for {g}", self.n_groups())));
            }
            append(&mut out, self.read_group(g)?)?;
        }
        Ok(out)
    }
}

/// Append `part` to `out` (same echo attributes).
pub(crate) fn append(out: &mut Shots, part: Shots) -> Result<()> {
    if out.n_shots() == 0 {
        *out = part;
        return Ok(());
    }
    let offset = out.echo_range.len();
    out.origin.extend(part.origin);
    out.direction.extend(part.direction);
    out.echo_start.extend(part.echo_start.iter().map(|s| s + offset));
    out.echo_count.extend(part.echo_count);
    out.echo_range.extend(part.echo_range);
    for (k, a) in &part.echo_attrs {
        out.echo_attrs.get_mut(k).ok_or_else(|| Error::invalid(format!("shots file: attribute {k:?} is not in every row group")))?.extend(a)?;
    }
    Ok(())
}

/// Read a whole shots file.
pub fn read_shots(path: impl AsRef<Path>) -> Result<Shots> {
    ShotsFile::open(path)?.read_all()
}
