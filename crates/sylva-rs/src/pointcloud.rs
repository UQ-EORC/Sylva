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
