// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! PLY reader/writer for the `vertex` element (ASCII and binary).
//!
//! raycloudtools ray clouds (`nx, ny, nz` = vector to sensor, `alpha` =
//! intensity, 0 = unbounded) read as ordinary attributes and can be turned
//! into [`crate::Shots`] with `Shots::from_ray_cloud`.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use byteorder::{BigEndian, ByteOrder, LittleEndian};

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::{Point, PointCloud};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyType {
    fn parse(s: &str) -> Option<PlyType> {
        Some(match s {
            "char" | "int8" => PlyType::I8,
            "uchar" | "uint8" => PlyType::U8,
            "short" | "int16" => PlyType::I16,
            "ushort" | "uint16" => PlyType::U16,
            "int" | "int32" => PlyType::I32,
            "uint" | "uint32" => PlyType::U32,
            "float" | "float32" => PlyType::F32,
            "double" | "float64" => PlyType::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            PlyType::I8 | PlyType::U8 => 1,
            PlyType::I16 | PlyType::U16 => 2,
            PlyType::I32 | PlyType::U32 | PlyType::F32 => 4,
            PlyType::F64 => 8,
        }
    }

    fn name(self) -> &'static str {
        match self {
            PlyType::I8 => "char",
            PlyType::U8 => "uchar",
            PlyType::I16 => "short",
            PlyType::U16 => "ushort",
            PlyType::I32 => "int",
            PlyType::U32 => "uint",
            PlyType::F32 => "float",
            PlyType::F64 => "double",
        }
    }

    fn read<B: ByteOrder>(self, b: &[u8]) -> f64 {
        match self {
            PlyType::I8 => b[0] as i8 as f64,
            PlyType::U8 => b[0] as f64,
            PlyType::I16 => B::read_i16(b) as f64,
            PlyType::U16 => B::read_u16(b) as f64,
            PlyType::I32 => B::read_i32(b) as f64,
            PlyType::U32 => B::read_u32(b) as f64,
            PlyType::F32 => B::read_f32(b) as f64,
            PlyType::F64 => B::read_f64(b),
        }
    }

    fn to_attr(self, v: Vec<f64>) -> Attr {
        match self {
            PlyType::I8 => Attr::I8(v.iter().map(|&x| x as i8).collect()),
            PlyType::U8 => Attr::U8(v.iter().map(|&x| x as u8).collect()),
            PlyType::I16 | PlyType::I32 => Attr::I32(v.iter().map(|&x| x as i32).collect()),
            PlyType::U16 => Attr::U16(v.iter().map(|&x| x as u16).collect()),
            PlyType::U32 => Attr::U32(v.iter().map(|&x| x as u32).collect()),
            PlyType::F32 => Attr::F32(v.iter().map(|&x| x as f32).collect()),
            PlyType::F64 => Attr::F64(v),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Ascii,
    BinaryLe,
    BinaryBe,
}

struct Element {
    name: String,
    count: usize,
    /// `(name, type, list count type)`; list properties carry `Some(count_type)`.
    props: Vec<(String, PlyType, Option<PlyType>)>,
}

fn parse_header<R: BufRead>(r: &mut R, path: &Path) -> Result<(Format, Vec<Element>)> {
    let mut line = String::new();
    r.read_line(&mut line)?;
    if line.trim() != "ply" {
        return Err(Error::file(path, "not a PLY file"));
    }
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(Error::file(path, "unexpected end of PLY header"));
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.first().copied() {
            None | Some("comment") | Some("obj_info") => {}
            Some("format") => {
                format = Some(match parts.get(1).copied() {
                    Some("ascii") => Format::Ascii,
                    Some("binary_little_endian") => Format::BinaryLe,
                    Some("binary_big_endian") => Format::BinaryBe,
                    other => return Err(Error::file(path, format!("unknown PLY format {other:?}"))),
                })
            }
            Some("element") => elements.push(Element {
                name: parts.get(1).unwrap_or(&"").to_string(),
                count: parts.get(2).and_then(|c| c.parse().ok()).unwrap_or(0),
                props: Vec::new(),
            }),
            Some("property") => {
                let el = elements.last_mut().ok_or_else(|| Error::file(path, "property before element"))?;
                if parts.get(1) == Some(&"list") {
                    let ct = PlyType::parse(parts[2]).ok_or_else(|| Error::file(path, "bad list type"))?;
                    let vt = PlyType::parse(parts[3]).ok_or_else(|| Error::file(path, "bad list type"))?;
                    el.props.push((parts[4].to_string(), vt, Some(ct)));
                } else {
                    let t = PlyType::parse(parts[1]).ok_or_else(|| Error::file(path, format!("bad type {}", parts[1])))?;
                    el.props.push((parts[2].to_string(), t, None));
                }
            }
            Some("end_header") => break,
            Some(other) => return Err(Error::file(path, format!("unexpected header line {other:?}"))),
        }
    }
    Ok((format.ok_or_else(|| Error::file(path, "missing format line"))?, elements))
}

/// Read the `vertex` element of a PLY file.
pub fn read_ply(path: impl AsRef<Path>) -> Result<PointCloud> {
    let path = path.as_ref();
    let mut r = BufReader::new(std::fs::File::open(path)?);
    let (format, elements) = parse_header(&mut r, path)?;
    let mut columns: Option<Vec<Vec<f64>>> = None;
    let mut vertex: Option<&Element> = None;
    for el in &elements {
        let is_vertex = el.name == "vertex";
        if is_vertex && el.props.iter().any(|p| p.2.is_some()) {
            return Err(Error::file(path, "list properties in vertex element are not supported"));
        }
        match format {
            Format::Ascii => {
                let mut cols: Vec<Vec<f64>> = if is_vertex { el.props.iter().map(|_| Vec::with_capacity(el.count)).collect() } else { Vec::new() };
                let mut line = String::new();
                for _ in 0..el.count {
                    line.clear();
                    r.read_line(&mut line)?;
                    if is_vertex {
                        for (k, tok) in line.split_whitespace().enumerate().take(el.props.len()) {
                            cols[k].push(tok.parse().map_err(|_| Error::file(path, format!("bad value {tok:?}")))?);
                        }
                    }
                }
                if is_vertex {
                    columns = Some(cols);
                    vertex = Some(el);
                    break;
                }
            }
            Format::BinaryLe | Format::BinaryBe => {
                if el.props.iter().any(|p| p.2.is_some()) {
                    if columns.is_some() {
                        break;
                    }
                    return Err(Error::file(path, format!("cannot skip binary list element {:?}", el.name)));
                }
                let row: usize = el.props.iter().map(|p| p.1.size()).sum();
                let mut buf = vec![0u8; row * el.count];
                r.read_exact(&mut buf)?;
                if is_vertex {
                    let mut cols: Vec<Vec<f64>> = el.props.iter().map(|_| Vec::with_capacity(el.count)).collect();
                    for rec in buf.chunks_exact(row) {
                        let mut off = 0;
                        for (k, (_, t, _)) in el.props.iter().enumerate() {
                            let b = &rec[off..off + t.size()];
                            cols[k].push(if format == Format::BinaryLe { t.read::<LittleEndian>(b) } else { t.read::<BigEndian>(b) });
                            off += t.size();
                        }
                    }
                    columns = Some(cols);
                    vertex = Some(el);
                    break;
                }
            }
        }
    }
    let (Some(cols), Some(el)) = (columns, vertex) else {
        return Err(Error::file(path, "no vertex element"));
    };
    let idx = |n: &str| el.props.iter().position(|p| p.0 == n).ok_or_else(|| Error::file(path, format!("missing vertex property {n}")));
    let (ix, iy, iz) = (idx("x")?, idx("y")?, idx("z")?);
    let n = el.count;
    let xyz: Vec<Point> = (0..n).map(|i| [cols[ix][i], cols[iy][i], cols[iz][i]]).collect();
    let mut cloud = PointCloud::new(xyz);
    for (k, (name, t, _)) in el.props.iter().enumerate() {
        if k == ix || k == iy || k == iz {
            continue;
        }
        cloud.attrs.insert(name.clone(), t.to_attr(cols[k].clone()));
    }
    Ok(cloud)
}

fn ply_type_of(attr: &Attr) -> PlyType {
    match attr {
        Attr::F64(_) => PlyType::F64,
        Attr::F32(_) => PlyType::F32,
        Attr::I64(_) | Attr::I32(_) => PlyType::I32,
        Attr::U32(_) => PlyType::U32,
        Attr::U16(_) => PlyType::U16,
        Attr::U8(_) | Attr::Bool(_) => PlyType::U8,
        Attr::I8(_) => PlyType::I8,
    }
}

fn write_value<W: Write>(w: &mut W, t: PlyType, v: f64) -> std::io::Result<()> {
    match t {
        PlyType::I8 => w.write_all(&[(v as i8) as u8]),
        PlyType::U8 => w.write_all(&[v as u8]),
        PlyType::I16 => w.write_all(&(v as i16).to_le_bytes()),
        PlyType::U16 => w.write_all(&(v as u16).to_le_bytes()),
        PlyType::I32 => w.write_all(&(v as i32).to_le_bytes()),
        PlyType::U32 => w.write_all(&(v as u32).to_le_bytes()),
        PlyType::F32 => w.write_all(&(v as f32).to_le_bytes()),
        PlyType::F64 => w.write_all(&v.to_le_bytes()),
    }
}

/// Write a PLY with `x y z` as doubles and all attributes as vertex properties.
pub fn write_ply(cloud: &PointCloud, path: impl AsRef<Path>, binary: bool) -> Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    let props: Vec<(&String, &Attr, PlyType)> = cloud.attrs.iter().map(|(k, a)| (k, a, ply_type_of(a))).collect();
    writeln!(w, "ply")?;
    writeln!(w, "format {} 1.0", if binary { "binary_little_endian" } else { "ascii" })?;
    writeln!(w, "comment written by sylva")?;
    writeln!(w, "element vertex {}", cloud.len())?;
    for c in ["x", "y", "z"] {
        writeln!(w, "property double {c}")?;
    }
    for (name, _, t) in &props {
        writeln!(w, "property {} {}", t.name(), name)?;
    }
    writeln!(w, "end_header")?;
    for i in 0..cloud.len() {
        let p = cloud.xyz[i];
        if binary {
            for v in p {
                w.write_all(&v.to_le_bytes())?;
            }
            for (_, a, t) in &props {
                write_value(&mut w, *t, a.get_f64(i))?;
            }
        } else {
            write!(w, "{} {} {}", p[0], p[1], p[2])?;
            for (_, a, t) in &props {
                match t {
                    PlyType::F32 | PlyType::F64 => write!(w, " {}", a.get_f64(i))?,
                    _ => write!(w, " {}", a.get_f64(i) as i64)?,
                }
            }
            writeln!(w)?;
        }
    }
    w.flush()?;
    Ok(())
}
