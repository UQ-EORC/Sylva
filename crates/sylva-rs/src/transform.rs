// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Rigid / affine 4x4 transforms.

use nalgebra::{Matrix3, Matrix4, Vector3};

use crate::error::{Error, Result};
use crate::Point;

/// A 4x4 homogeneous transform (row-major semantics: `p' = M p`).
///
/// The usual 4x4: the top-left 3x3 rotates (and, for an affine one, scales or
/// shears), the last column translates, and the fourth row is `0 0 0 1`. A
/// point is carried by multiplying on the left, so `a * b` applies `b` first
/// and then `a` - the order to keep in mind when composing a chain of
/// registrations.
///
/// `Transform(pub Matrix4<f64>)` is a named wrapper around one matrix: a
/// distinct type, so a transform cannot be mistaken for any other 4x4, with
/// the matrix itself reachable as `.0` when nalgebra's own operations are
/// wanted. Being `Copy` means it is duplicated on assignment like a number
/// rather than moved, which is why transforms are passed around by value
/// throughout the core.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform(pub Matrix4<f64>);

impl Default for Transform {
    fn default() -> Self {
        Transform(Matrix4::identity())
    }
}

impl Transform {
    pub fn identity() -> Self {
        Self::default()
    }

    /// From a rotation matrix and translation vector.
    pub fn from_rt(r: Matrix3<f64>, t: Vector3<f64>) -> Self {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&t);
        Transform(m)
    }

    /// From 16 row-major values.
    pub fn from_row_major(v: &[f64]) -> Result<Self> {
        if v.len() != 16 {
            return Err(Error::invalid("transform needs 16 values"));
        }
        Ok(Transform(Matrix4::from_row_slice(v)))
    }

    pub fn to_row_major(&self) -> [f64; 16] {
        let mut out = [0.0; 16];
        for r in 0..4 {
            for c in 0..4 {
                out[r * 4 + c] = self.0[(r, c)];
            }
        }
        out
    }

    pub fn translation(dx: f64, dy: f64, dz: f64) -> Self {
        Self::from_rt(Matrix3::identity(), Vector3::new(dx, dy, dz))
    }

    /// Rotation about z by `deg` degrees.
    pub fn rotation_z(deg: f64) -> Self {
        let a = deg.to_radians();
        let (s, c) = a.sin_cos();
        Self::from_rt(Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0), Vector3::zeros())
    }

    /// RIEGL / RiSCAN convention: `Rz(yaw) Ry(pitch) Rx(roll)` with angles in degrees.
    pub fn from_roll_pitch_yaw(roll: f64, pitch: f64, yaw: f64) -> Self {
        let rx = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, roll.to_radians().cos(), -roll.to_radians().sin(),
            0.0, roll.to_radians().sin(), roll.to_radians().cos(),
        );
        let ry = Matrix3::new(
            pitch.to_radians().cos(), 0.0, pitch.to_radians().sin(),
            0.0, 1.0, 0.0,
            -pitch.to_radians().sin(), 0.0, pitch.to_radians().cos(),
        );
        let rz = Matrix3::new(
            yaw.to_radians().cos(), -yaw.to_radians().sin(), 0.0,
            yaw.to_radians().sin(), yaw.to_radians().cos(), 0.0,
            0.0, 0.0, 1.0,
        );
        Self::from_rt(rz * ry * rx, Vector3::zeros())
    }

    pub fn rotation(&self) -> Matrix3<f64> {
        self.0.fixed_view::<3, 3>(0, 0).into_owned()
    }

    pub fn translation_vec(&self) -> Vector3<f64> {
        self.0.fixed_view::<3, 1>(0, 3).into_owned()
    }

    #[inline]
    pub fn apply(&self, p: &Point) -> Point {
        let m = &self.0;
        [
            m[(0, 0)] * p[0] + m[(0, 1)] * p[1] + m[(0, 2)] * p[2] + m[(0, 3)],
            m[(1, 0)] * p[0] + m[(1, 1)] * p[1] + m[(1, 2)] * p[2] + m[(1, 3)],
            m[(2, 0)] * p[0] + m[(2, 1)] * p[1] + m[(2, 2)] * p[2] + m[(2, 3)],
        ]
    }

    /// Rotate a direction (no translation).
    #[inline]
    pub fn apply_dir(&self, d: &Point) -> Point {
        let m = &self.0;
        [
            m[(0, 0)] * d[0] + m[(0, 1)] * d[1] + m[(0, 2)] * d[2],
            m[(1, 0)] * d[0] + m[(1, 1)] * d[1] + m[(1, 2)] * d[2],
            m[(2, 0)] * d[0] + m[(2, 1)] * d[1] + m[(2, 2)] * d[2],
        ]
    }

    /// `self ∘ other` (apply `other` first).
    pub fn compose(&self, other: &Transform) -> Transform {
        Transform(self.0 * other.0)
    }

    pub fn inverse(&self) -> Result<Transform> {
        self.0.try_inverse().map(Transform).ok_or_else(|| Error::invalid("singular transform"))
    }

    /// Read a whitespace-delimited 4x4 matrix file (RIEGL `.dat` SOP/POP).
    pub fn read_matrix_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| Error::file(path.as_ref(), e.to_string()))?;
        let vals: Vec<f64> = text
            .split_whitespace()
            .map(|t| t.parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::file(path.as_ref(), format!("bad matrix value: {e}")))?;
        if vals.len() != 16 {
            return Err(Error::file(path.as_ref(), format!("expected 16 values, got {}", vals.len())));
        }
        Self::from_row_major(&vals)
    }
}

#[inline]
pub(crate) fn sub(a: &Point, b: &Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn dot(a: &Point, b: &Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn norm(a: &Point) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub(crate) fn scale(a: &Point, s: f64) -> Point {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub(crate) fn add(a: &Point, b: &Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn cross(a: &Point, b: &Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
pub(crate) fn normalize(a: &Point) -> Point {
    let n = norm(a);
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_files_round_trip_and_name_themselves_in_errors() {
        let dir = std::env::temp_dir().join(format!("sylva-matrix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t = Transform::from_roll_pitch_yaw(1.0, -2.0, 30.0).compose(&Transform::translation(10.0, -5.0, 2.5));
        let text: Vec<String> = t.to_row_major().iter().map(|v| format!("{v:.17e}")).collect();
        let path = dir.join("sop.dat");
        std::fs::write(&path, text.chunks(4).map(|r| r.join(" ")).collect::<Vec<_>>().join("\n")).unwrap();
        assert_eq!(Transform::read_matrix_file(&path).unwrap().to_row_major(), t.to_row_major());
        let prefix = |p: &std::path::Path| format!("{}: ", p.display());
        std::fs::write(&path, "1 0 0 0 0 1 0 0 0 0 1 0").unwrap();
        assert_eq!(Transform::read_matrix_file(&path).unwrap_err().to_string(), format!("{}expected 16 values, got 12", prefix(&path)));
        std::fs::write(&path, "1 0 0 x").unwrap();
        assert!(Transform::read_matrix_file(&path).unwrap_err().to_string().starts_with(&format!("{}bad matrix value", prefix(&path))));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(Transform::read_matrix_file(&path).unwrap_err().to_string().starts_with(&prefix(&path)));
    }
}
