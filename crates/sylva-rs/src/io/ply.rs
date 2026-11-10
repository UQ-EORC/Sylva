// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! PLY reader/writer for the `vertex` element (ASCII and binary).
//!
//! raycloudtools ray clouds (`nx, ny, nz` = vector to sensor, `alpha` =
//! intensity, 0 = unbounded) read as ordinary attributes and can be turned
//! into [`crate::Shots`] with `Shots::from_ray_cloud`.

use std::io::{BufRead, BufReader, BufWriter, Read, Seek, Write};
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

/// Read the header, stopping at `end_header` with the reader left at the
/// first record.
///
/// A PLY header declares its format (ASCII or binary, and which byte order)
/// and then a run of elements, each with a count and a list of properties.
/// Nothing about the records can be known before this is read, which is why
/// it comes back as a description for `read_ply` to act on.
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
            Some("element") => {
                let (Some(name), Some(count)) = (parts.get(1), parts.get(2).and_then(|c| c.parse().ok())) else {
                    return Err(Error::file(path, format!("bad element line {:?}", line.trim())));
                };
                elements.push(Element { name: name.to_string(), count, props: Vec::new() });
            }
            Some("property") => {
                let el = elements.last_mut().ok_or_else(|| Error::file(path, "property before element"))?;
                let bad = || Error::file(path, format!("bad property line {:?}", line.trim()));
                if parts.get(1) == Some(&"list") {
                    let (Some(ct), Some(vt), Some(name)) = (parts.get(2), parts.get(3), parts.get(4)) else { return Err(bad()) };
                    let ct = PlyType::parse(ct).ok_or_else(|| Error::file(path, "bad list type"))?;
                    let vt = PlyType::parse(vt).ok_or_else(|| Error::file(path, "bad list type"))?;
                    el.props.push((name.to_string(), vt, Some(ct)));
                } else {
                    let (Some(t), Some(name)) = (parts.get(1), parts.get(2)) else { return Err(bad()) };
                    let t = PlyType::parse(t).ok_or_else(|| Error::file(path, format!("bad type {t}")))?;
                    el.props.push((name.to_string(), t, None));
                }
            }
            Some("end_header") => break,
            Some(other) => return Err(Error::file(path, format!("unexpected header line {other:?}"))),
        }
    }
    Ok((format.ok_or_else(|| Error::file(path, "missing format line"))?, elements))
}

/// Read the `vertex` element of a PLY file.
///
/// A PLY header lists its elements (`vertex`, often `face`) and the properties
/// of each. The points live in `vertex`, but its records can only be found by
/// reading the elements declared before it, so every element is walked in
/// order; the ones that are not `vertex` are read past and discarded. Each
/// property becomes a column, whatever its stored type, and the columns become
/// coordinates and attributes afterwards.
pub fn read_ply(path: impl AsRef<Path>) -> Result<PointCloud> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = BufReader::new(file);
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
                // A record is at least two bytes of text, so the file's length
                // bounds what a header's count can honestly reserve.
                let room = el.count.min(file_len as usize / 2);
                let mut cols: Vec<Vec<f64>> = if is_vertex { el.props.iter().map(|_| Vec::with_capacity(room)).collect() } else { Vec::new() };
                let mut line = String::new();
                for row in 0..el.count {
                    line.clear();
                    if r.read_line(&mut line)? == 0 {
                        return Err(Error::file(path, format!("truncated: element {:?} declares {} records and the file ends after {row}", el.name, el.count)));
                    }
                    if is_vertex {
                        let mut got = 0;
                        for (k, tok) in line.split_whitespace().enumerate().take(el.props.len()) {
                            cols[k].push(tok.parse().map_err(|_| Error::file(path, format!("bad value {tok:?}")))?);
                            got += 1;
                        }
                        if got != el.props.len() {
                            return Err(Error::file(path, format!("vertex {row} has {got} values for {} properties", el.props.len())));
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
                // The count is the header's word; check it against the bytes
                // that are there before allocating for it.
                let left = file_len.saturating_sub(r.stream_position()?);
                let bytes = row.checked_mul(el.count).filter(|&b| b as u64 <= left).ok_or_else(|| {
                    Error::file(path, format!("truncated: element {:?} declares {} records of {row} bytes and {left} bytes remain", el.name, el.count))
                })?;
                let mut buf = vec![0u8; bytes];
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

/// The PLY type an attribute is written as. PLY has no 64-bit integer, so a
/// 64-bit column goes out as `int` when every value fits 32 bits and as
/// `double` otherwise, which holds integers exactly up to 2^53;
/// [`write_ply`] refuses values beyond that.
fn ply_type_of(attr: &Attr) -> PlyType {
    match attr {
        Attr::F64(_) => PlyType::F64,
        Attr::F32(_) => PlyType::F32,
        Attr::I64(v) if v.iter().any(|&x| x < i32::MIN as i64 || x > i32::MAX as i64) => PlyType::F64,
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

/// Write a PLY with `x y z` as doubles and all attributes as vertex
/// properties. A 64-bit integer attribute is written as `int` when its values
/// fit 32 bits, else as `double` (PLY has no 64-bit integer type); a value
/// beyond 2^53, which a double cannot hold exactly, is an error.
pub fn write_ply(cloud: &PointCloud, path: impl AsRef<Path>, binary: bool) -> Result<()> {
    for (name, a) in &cloud.attrs {
        if let Attr::I64(v) = a {
            if let Some(x) = v.iter().find(|x| x.unsigned_abs() > 1 << 53) {
                return Err(Error::Invalid(format!("attribute {name} holds {x}, which a PLY file cannot store exactly (no 64-bit integer type; doubles are exact to 2^53)")));
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("sylva-ply-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join("t.ply")
    }

    #[test]
    fn malformed_files_are_errors() {
        let read = |name: &str, bytes: &[u8]| {
            let p = tmp(name);
            std::fs::write(&p, bytes).unwrap();
            let r = read_ply(&p);
            let _ = std::fs::remove_dir_all(p.parent().unwrap());
            r
        };
        let head = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        // Fewer records than declared, and a record with a value missing.
        let e = read("short", format!("{head}0 0 0\n1 1 1\n").as_bytes()).unwrap_err().to_string();
        assert!(e.contains("truncated"), "{e}");
        let e = read("ragged", format!("{head}0 0 0\n1 1\n2 2 2\n").as_bytes()).unwrap_err().to_string();
        assert!(e.contains("vertex 1 has 2 values"), "{e}");
        assert_eq!(read("whole", format!("{head}0 0 0\n1 1 1\n2 2 2\n").as_bytes()).unwrap().len(), 3);
        // Header lines with words missing.
        for bad in ["property float\n", "property\n", "property list uchar int\n", "element vertex\n", "element vertex many\n"] {
            let text = format!("ply\nformat ascii 1.0\nelement vertex 1\n{bad}end_header\n0 0 0\n");
            assert!(read("header", text.as_bytes()).is_err(), "{bad:?}");
        }
        // A count the file has no bytes for is refused before it is allocated.
        let huge = "ply\nformat binary_little_endian 1.0\nelement vertex 4000000000000000\nproperty double x\nproperty double y\nproperty double z\nend_header\n";
        let e = read("huge", huge.as_bytes()).unwrap_err().to_string();
        assert!(e.contains("truncated"), "{e}");
        assert!(read("huge-ascii", huge.replace("binary_little_endian", "ascii").as_bytes()).is_err());
    }

    #[test]
    fn sixty_four_bit_integers_keep_their_values() {
        let mut c = PointCloud::new(vec![[0.0; 3], [1.0; 3]]);
        c.attrs.insert("tree_id".into(), Attr::I64(vec![5_000_000_000, -7]));
        c.attrs.insert("small".into(), Attr::I64(vec![12, -3]));
        for binary in [true, false] {
            let p = tmp(if binary { "i64b" } else { "i64a" });
            write_ply(&c, &p, binary).unwrap();
            let back = read_ply(&p).unwrap();
            assert_eq!(back.attrs["tree_id"].to_f64(), vec![5.0e9, -7.0], "binary {binary}");
            assert_eq!(back.attrs["small"], Attr::I32(vec![12, -3]));
            c.attrs.insert("beyond".into(), Attr::I64(vec![(1 << 53) + 1, 0]));
            assert!(write_ply(&c, &p, binary).is_err());
            c.attrs.remove("beyond");
            let _ = std::fs::remove_dir_all(p.parent().unwrap());
        }
    }
}
