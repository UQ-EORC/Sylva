// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Core point cloud container.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::transform::Transform;
use crate::Point;

/// A typed per-point attribute column.
///
/// Types mirror what LAS and numpy commonly carry so files round-trip
/// without silent widening.
#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
    I8(Vec<i8>),
    Bool(Vec<bool>),
}

macro_rules! attr_dispatch {
    ($self:expr, $v:ident => $body:expr) => {
        match $self {
            Attr::F64($v) => $body,
            Attr::F32($v) => $body,
            Attr::I64($v) => $body,
            Attr::I32($v) => $body,
            Attr::U32($v) => $body,
            Attr::U16($v) => $body,
            Attr::U8($v) => $body,
            Attr::I8($v) => $body,
            Attr::Bool($v) => $body,
        }
    };
}

impl Attr {
    pub fn len(&self) -> usize {
        attr_dispatch!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Short dtype name (numpy style).
    pub fn dtype(&self) -> &'static str {
        match self {
            Attr::F64(_) => "float64",
            Attr::F32(_) => "float32",
            Attr::I64(_) => "int64",
            Attr::I32(_) => "int32",
            Attr::U32(_) => "uint32",
            Attr::U16(_) => "uint16",
            Attr::U8(_) => "uint8",
            Attr::I8(_) => "int8",
            Attr::Bool(_) => "bool",
        }
    }

    /// Value at `i` widened to f64.
    pub fn get_f64(&self, i: usize) -> f64 {
        match self {
            Attr::F64(v) => v[i],
            Attr::F32(v) => v[i] as f64,
            Attr::I64(v) => v[i] as f64,
            Attr::I32(v) => v[i] as f64,
            Attr::U32(v) => v[i] as f64,
            Attr::U16(v) => v[i] as f64,
            Attr::U8(v) => v[i] as f64,
            Attr::I8(v) => v[i] as f64,
            Attr::Bool(v) => v[i] as u8 as f64,
        }
    }

    /// Whole column widened to f64.
    pub fn to_f64(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.get_f64(i)).collect()
    }

    /// Select rows by index.
    pub fn take(&self, idx: &[usize]) -> Attr {
        attr_dispatch!(self, v => {
            let out: Vec<_> = idx.iter().map(|&i| v[i].clone()).collect();
            out.into()
        })
    }

    /// Append another column of the same type.
    pub fn extend(&mut self, other: &Attr) -> Result<()> {
        match (self, other) {
            (Attr::F64(a), Attr::F64(b)) => a.extend_from_slice(b),
            (Attr::F32(a), Attr::F32(b)) => a.extend_from_slice(b),
            (Attr::I64(a), Attr::I64(b)) => a.extend_from_slice(b),
            (Attr::I32(a), Attr::I32(b)) => a.extend_from_slice(b),
            (Attr::U32(a), Attr::U32(b)) => a.extend_from_slice(b),
            (Attr::U16(a), Attr::U16(b)) => a.extend_from_slice(b),
            (Attr::U8(a), Attr::U8(b)) => a.extend_from_slice(b),
            (Attr::I8(a), Attr::I8(b)) => a.extend_from_slice(b),
            (Attr::Bool(a), Attr::Bool(b)) => a.extend_from_slice(b),
            (a, b) => {
                return Err(Error::invalid(format!(
                    "cannot concatenate {} with {}",
                    a.dtype(),
                    b.dtype()
                )))
            }
        }
        Ok(())
    }
}

macro_rules! attr_from {
    ($($t:ty => $variant:ident),* $(,)?) => {
        $(impl From<Vec<$t>> for Attr {
            fn from(v: Vec<$t>) -> Self { Attr::$variant(v) }
        })*
    };
}
attr_from!(f64 => F64, f32 => F32, i64 => I64, i32 => I32, u32 => U32, u16 => U16, u8 => U8,
           i8 => I8, bool => Bool);

/// A set of 3-D points with optional per-point attributes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PointCloud {
    pub xyz: Vec<Point>,
    pub attrs: BTreeMap<String, Attr>,
}

impl PointCloud {
    pub fn new(xyz: Vec<Point>) -> Self {
        PointCloud { xyz, attrs: BTreeMap::new() }
    }

    /// Build from coordinates and attributes, checking lengths.
    pub fn with_attrs(xyz: Vec<Point>, attrs: BTreeMap<String, Attr>) -> Result<Self> {
        for (name, a) in &attrs {
            if a.len() != xyz.len() {
                return Err(Error::invalid(format!(
                    "attribute {name:?} has length {}, expected {}",
                    a.len(),
                    xyz.len()
                )));
            }
        }
        Ok(PointCloud { xyz, attrs })
    }

    pub fn len(&self) -> usize {
        self.xyz.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xyz.is_empty()
    }

    /// Insert or replace an attribute.
    pub fn set_attr(&mut self, name: impl Into<String>, attr: impl Into<Attr>) -> Result<()> {
        let attr = attr.into();
        if attr.len() != self.len() {
            return Err(Error::invalid(format!(
                "attribute has length {}, expected {}",
                attr.len(),
                self.len()
            )));
        }
        self.attrs.insert(name.into(), attr);
        Ok(())
    }

    pub fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.get(name)
    }

    /// Attribute widened to f64, if present.
    pub fn attr_f64(&self, name: &str) -> Option<Vec<f64>> {
        self.attrs.get(name).map(Attr::to_f64)
    }

    /// Heights: the `height` attribute if present, otherwise z.
    pub fn heights(&self, attr: &str) -> Vec<f64> {
        self.attr_f64(attr).unwrap_or_else(|| self.xyz.iter().map(|p| p[2]).collect())
    }

    pub fn x(&self) -> impl Iterator<Item = f64> + '_ {
        self.xyz.iter().map(|p| p[0])
    }

    pub fn y(&self) -> impl Iterator<Item = f64> + '_ {
        self.xyz.iter().map(|p| p[1])
    }

    pub fn z(&self) -> impl Iterator<Item = f64> + '_ {
        self.xyz.iter().map(|p| p[2])
    }

    /// `(min, max)` corners; `None` for an empty cloud.
    pub fn bounds(&self) -> Option<(Point, Point)> {
        if self.xyz.is_empty() {
            return None;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &self.xyz {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Some((lo, hi))
    }

    /// Subset by explicit row indices.
    pub fn take(&self, idx: &[usize]) -> PointCloud {
        PointCloud {
            xyz: idx.iter().map(|&i| self.xyz[i]).collect(),
            attrs: self.attrs.iter().map(|(k, a)| (k.clone(), a.take(idx))).collect(),
        }
    }

    /// Subset by boolean mask.
    pub fn filter(&self, mask: &[bool]) -> PointCloud {
        let idx: Vec<usize> = mask.iter().enumerate().filter(|(_, &m)| m).map(|(i, _)| i).collect();
        self.take(&idx)
    }

    /// Apply a rigid/affine transform, returning a new cloud.
    pub fn transformed(&self, t: &Transform) -> PointCloud {
        PointCloud { xyz: self.xyz.iter().map(|p| t.apply(p)).collect(), attrs: self.attrs.clone() }
    }

    /// Apply a transform in place.
    pub fn transform_in_place(&mut self, t: &Transform) {
        for p in &mut self.xyz {
            *p = t.apply(p);
        }
    }

    /// Concatenate clouds; only attributes present in all inputs are kept.
    pub fn concatenate(clouds: &[PointCloud]) -> Result<PointCloud> {
        let Some(first) = clouds.first() else {
            return Ok(PointCloud::default());
        };
        let mut out = PointCloud::new(Vec::with_capacity(clouds.iter().map(|c| c.len()).sum()));
        let common: Vec<String> = first
            .attrs
            .keys()
            .filter(|k| clouds.iter().all(|c| c.attrs.contains_key(*k)))
            .cloned()
            .collect();
        for c in clouds {
            out.xyz.extend_from_slice(&c.xyz);
        }
        for name in common {
            let mut col = first.attrs[&name].clone();
            for c in &clouds[1..] {
                col.extend(&c.attrs[&name])?;
            }
            out.attrs.insert(name, col);
        }
        Ok(out)
    }

    /// Centroid of the cloud.
    pub fn centroid(&self) -> Point {
        let n = self.len().max(1) as f64;
        let mut c = [0.0; 3];
        for p in &self.xyz {
            for k in 0..3 {
                c[k] += p[k];
            }
        }
        [c[0] / n, c[1] / n, c[2] / n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_type() -> Vec<Attr> {
        vec![
            Attr::F64(vec![1.5, -2.0, 3.0]),
            Attr::F32(vec![1.5, -2.0, 3.0]),
            Attr::I64(vec![1, -2, 3]),
            Attr::I32(vec![1, -2, 3]),
            Attr::U32(vec![1, 2, u32::MAX]),
            Attr::U16(vec![1, 2, 3]),
            Attr::U8(vec![1, 2, 255]),
            Attr::I8(vec![1, -2, -128]),
            Attr::Bool(vec![true, false, true]),
        ]
    }

    #[test]
    fn attributes_of_every_type() {
        let names: Vec<&str> = every_type().iter().map(Attr::dtype).collect();
        assert_eq!(names, ["float64", "float32", "int64", "int32", "uint32", "uint16", "uint8", "int8", "bool"]);
        let widened: Vec<Vec<f64>> = every_type().iter().map(Attr::to_f64).collect();
        assert_eq!(widened[2], [1.0, -2.0, 3.0]);
        assert_eq!(widened[4], [1.0, 2.0, u32::MAX as f64]);
        assert_eq!(widened[7], [1.0, -2.0, -128.0]);
        assert_eq!(widened[8], [1.0, 0.0, 1.0]);
        for a in every_type() {
            assert_eq!((a.len(), a.is_empty()), (3, false));
            let t = a.take(&[2, 0]);
            assert_eq!(t.dtype(), a.dtype());
            assert_eq!(t.to_f64(), [a.get_f64(2), a.get_f64(0)]);
            assert!(a.take(&[]).is_empty());
            let mut twice = a.clone();
            twice.extend(&a).unwrap();
            assert_eq!(twice.len(), 6);
            assert_eq!(twice.to_f64()[3..], a.to_f64()[..]);
        }
        let mut a = Attr::U8(vec![1]);
        let e = a.extend(&Attr::I8(vec![1])).unwrap_err().to_string();
        assert_eq!(e, "cannot concatenate uint8 with int8");
    }

    fn cloud() -> PointCloud {
        let xyz = vec![[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [1.0, -2.0, 3.0]];
        let mut c = PointCloud::new(xyz);
        c.set_attr("height", vec![0.5f32, 1.5, 2.5]).unwrap();
        c.set_attr("class", vec![2u8, 5, 5]).unwrap();
        c
    }

    #[test]
    fn lengths_are_checked() {
        let mut c = cloud();
        assert_eq!(c.set_attr("bad", vec![1.0f64]).unwrap_err().to_string(), "attribute has length 1, expected 3");
        let mut attrs = BTreeMap::new();
        attrs.insert("a".to_string(), Attr::F64(vec![1.0]));
        let e = PointCloud::with_attrs(vec![[0.0; 3]; 2], attrs).unwrap_err().to_string();
        assert_eq!(e, "attribute \"a\" has length 1, expected 2");
        assert!(c.attr("bad").is_none() && c.attr_f64("missing").is_none());
    }

    #[test]
    fn coordinates_heights_bounds_and_centroid() {
        let c = cloud();
        assert_eq!(c.x().collect::<Vec<_>>(), [0.0, 2.0, 1.0]);
        assert_eq!(c.y().collect::<Vec<_>>(), [0.0, 4.0, -2.0]);
        assert_eq!(c.z().collect::<Vec<_>>(), [0.0, 6.0, 3.0]);
        assert_eq!(c.heights("height"), [0.5, 1.5, 2.5]);
        assert_eq!(c.heights("nope"), [0.0, 6.0, 3.0], "falls back to z");
        assert_eq!(c.bounds(), Some(([0.0, -2.0, 0.0], [2.0, 4.0, 6.0])));
        assert_eq!(c.centroid(), [1.0, 2.0 / 3.0, 3.0]);
        let empty = PointCloud::default();
        assert!(empty.is_empty() && empty.bounds().is_none());
        assert_eq!(empty.centroid(), [0.0; 3]);
    }

    #[test]
    fn subsets_keep_attributes_in_step() {
        let c = cloud();
        let f = c.filter(&[true, false, true]);
        assert_eq!(f.xyz, [c.xyz[0], c.xyz[2]]);
        assert_eq!(f.attr_f64("height").unwrap(), [0.5, 2.5]);
        assert_eq!(f.attr("class"), Some(&Attr::U8(vec![2, 5])));
        assert!(c.filter(&[false; 3]).is_empty());
    }

    #[test]
    fn transforms_move_points_not_attributes() {
        let c = cloud();
        let t = Transform::rotation_z(90.0).compose(&Transform::translation(1.0, 2.0, 3.0));
        let moved = c.transformed(&t);
        let mut in_place = c.clone();
        in_place.transform_in_place(&t);
        assert_eq!(moved, in_place);
        assert_eq!(moved.attrs, c.attrs);
        for (p, q) in c.xyz.iter().zip(&moved.xyz) {
            assert_eq!(*q, t.apply(p));
        }
        // A quarter turn about z after the shift: (x, y, z) -> (-(y + 2), x + 1, z + 3).
        let q = moved.xyz[1];
        assert!((q[0] + 6.0).abs() < 1e-12 && (q[1] - 3.0).abs() < 1e-12 && (q[2] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn concatenate_keeps_the_common_attributes() {
        let a = cloud();
        let mut b = PointCloud::new(vec![[9.0, 9.0, 9.0]]);
        b.set_attr("height", vec![7.0f32]).unwrap();
        let c = PointCloud::concatenate(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(c.len(), 4);
        assert_eq!(c.xyz[3], [9.0, 9.0, 9.0]);
        assert_eq!(c.attrs.keys().collect::<Vec<_>>(), ["height"], "class is not in both");
        assert_eq!(c.attr_f64("height").unwrap(), [0.5, 1.5, 2.5, 7.0]);
        b.set_attr("height", vec![7.0f64]).unwrap();
        assert_eq!(PointCloud::concatenate(&[a, b]).unwrap_err().to_string(), "cannot concatenate float32 with float64");
        assert_eq!(PointCloud::concatenate(&[]).unwrap(), PointCloud::default());
    }
}
