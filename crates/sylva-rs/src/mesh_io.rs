// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Wavefront OBJ files for triangle meshes: wood and leaf meshes out, a
//! single leaf blade in.
//!
//! Several meshes go to one file as named objects (`o wood`, `o leaves`,
//! `tree_1`, ...), vertices with four decimals (a tenth of a millimetre)
//! and 1-based face indices running on across the objects.

use std::fmt::Write as _;
use std::path::Path;

use crate::{Error, Point, Result};

/// One named object of an OBJ file; `faces` index `vertices` from 0.
#[derive(Debug, Clone, Copy)]
pub struct ObjMesh<'a> {
    pub name: &'a str,
    pub vertices: &'a [Point],
    pub faces: &'a [[u32; 3]],
}

/// `x` with four decimals, as C's `%.4f`: correctly rounded, ties to even,
/// `nan`, `inf` and `-inf` spelled as C spells them.
pub fn fixed4(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    format!("{x:.4}")
}

/// The text of an OBJ file holding `meshes`.
pub fn obj_text(meshes: &[ObjMesh]) -> String {
    let mut s = String::from("# sylva QSM mesh\n");
    let mut offset = 1u64;
    for m in meshes {
        let _ = writeln!(s, "o {}", m.name);
        for v in m.vertices {
            let _ = writeln!(s, "v {} {} {}", fixed4(v[0]), fixed4(v[1]), fixed4(v[2]));
        }
        for f in m.faces {
            let _ = writeln!(s, "f {} {} {}", f[0] as u64 + offset, f[1] as u64 + offset, f[2] as u64 + offset);
        }
        offset += m.vertices.len() as u64;
    }
    s
}

/// Write `meshes` to one OBJ file, one named object each.
pub fn write_obj(path: impl AsRef<Path>, meshes: &[ObjMesh]) -> Result<()> {
    std::fs::write(path, obj_text(meshes))?;
    Ok(())
}

/// Vertices and triangles of OBJ text. Only `v` and `f` lines are read;
/// polygons are fanned into triangles from their first corner, texture and
/// normal indices are ignored, and negative indices count back from the
/// last vertex read so far. Vertices given with two coordinates get z = 0.
pub fn parse_obj(text: &str) -> Result<(Vec<Point>, Vec<[i64; 3]>)> {
    let mut v: Vec<Point> = Vec::new();
    let mut dims: Option<usize> = None;
    let mut f: Vec<[i64; 3]> = Vec::new();
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        if w.is_empty() {
            continue;
        }
        if w[0] == "v" {
            let c = w[1..].iter().take(3).map(|x| x.parse::<f64>().map_err(|_| Error::invalid(format!("could not convert string to float: {x:?}")))).collect::<Result<Vec<f64>>>()?;
            if dims.is_some_and(|d| d != c.len()) || c.len() < 2 {
                return Err(Error::invalid("vertices must be (V, 3) or (V, 2)"));
            }
            dims = Some(c.len());
            v.push([c[0], c[1], c.get(2).copied().unwrap_or(0.0)]);
        } else if w[0] == "f" {
            let idx = w[1..]
                .iter()
                .map(|p| {
                    let s = p.split('/').next().unwrap_or("");
                    s.parse::<i64>().map_err(|_| Error::invalid(format!("invalid literal for int() with base 10: {s:?}")))
                })
                .collect::<Result<Vec<i64>>>()?;
            let idx: Vec<i64> = idx.iter().map(|&i| if i > 0 { i - 1 } else { v.len() as i64 + i }).collect();
            for t in 1..idx.len().saturating_sub(1) {
                f.push([idx[0], idx[t], idx[t + 1]]);
            }
        }
    }
    Ok((v, f))
}

/// Read an OBJ file with [`parse_obj`]; an error if it holds no faces.
pub fn read_obj(path: impl AsRef<Path>) -> Result<(Vec<Point>, Vec<[u32; 3]>)> {
    let path = path.as_ref();
    let (v, f) = parse_obj(&std::fs::read_to_string(path)?)?;
    if f.is_empty() {
        return Err(Error::invalid(format!("{} holds no faces", path.display())));
    }
    let faces = f
        .iter()
        .map(|t| {
            let mut out = [0u32; 3];
            for k in 0..3 {
                out[k] = u32::try_from(t[k]).map_err(|_| Error::invalid(format!("{}: face index {} out of range", path.display(), t[k] + 1)))?;
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((v, faces))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_decimals_as_c() {
        assert_eq!(fixed4(0.03125), "0.0312");
        assert_eq!(fixed4(0.09375), "0.0938");
        assert_eq!(fixed4(-0.00004), "-0.0000");
        assert_eq!(fixed4(-0.0), "-0.0000");
        assert_eq!(fixed4(1e9), "1000000000.0000");
        assert_eq!(fixed4(1.23455), "1.2346");
        assert_eq!(fixed4(f64::NAN), "nan");
    }

    #[test]
    fn objects_number_their_faces_on() {
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let f = [[0, 1, 2]];
        let t = obj_text(&[ObjMesh { name: "a", vertices: &v, faces: &f }, ObjMesh { name: "b", vertices: &v, faces: &f }]);
        assert!(t.starts_with("# sylva QSM mesh\no a\nv 0.0000 0.0000 0.0000\n"));
        assert!(t.ends_with("o b\nv 0.0000 0.0000 0.0000\nv 1.0000 0.0000 0.0000\nv 0.0000 1.0000 0.0000\nf 4 5 6\n"));
    }

    #[test]
    fn polygons_fan_and_negative_indices_count_back() {
        let (v, f) = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1 4/1\nf -4 -2 -1\n").unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(f, vec![[0, 1, 2], [0, 2, 3], [0, 2, 3]]);
        let (v, _) = parse_obj("v 1 2\nv 3 4\n").unwrap();
        assert_eq!(v[1], [3.0, 4.0, 0.0]);
        assert!(parse_obj("v 1 2 3\nv 1 2\n").is_err());
    }
}
