// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Shifting, rotating and recentring coordinates.
//!
//! Every operation is a [`Transform`], so shifts and rotations compose with
//! registration results and RiSCAN matrices. Two details keep repeated round
//! trips free of drift beyond float rounding: rotations by whole multiples of
//! 90 degrees use exact sines and cosines (0 and 1, not 6e-17), and a pure
//! translation is applied as one addition per coordinate rather than as a
//! matrix product.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::transform::Transform;
use crate::Point;

use nalgebra::{Matrix3, Vector3};

/// Points per parallel work unit; results do not depend on it.
const CHUNK: usize = 1 << 16;

/// Sine and cosine of an angle in degrees, exact at multiples of 90 degrees.
pub fn sin_cos_deg(deg: f64) -> (f64, f64) {
    let r = deg.rem_euclid(360.0);
    if r == 0.0 {
        (0.0, 1.0)
    } else if r == 90.0 {
        (1.0, 0.0)
    } else if r == 180.0 {
        (0.0, -1.0)
    } else if r == 270.0 {
        (-1.0, 0.0)
    } else {
        deg.to_radians().sin_cos()
    }
}

/// Rotation by `deg` degrees (counter-clockwise looking down the axis towards
/// the origin, i.e. right-handed) about `axis` through the point `about`
/// (the origin when `None`).
///
/// # Errors
/// If the angle or axis is not finite, or the axis has zero length.
pub fn rotation(axis: [f64; 3], deg: f64, about: Option<Point>) -> Result<Transform> {
    if !deg.is_finite() {
        return Err(Error::invalid(format!("rotation angle must be finite, got {deg}")));
    }
    if axis.iter().any(|v| !v.is_finite()) {
        return Err(Error::invalid("rotation axis must be finite"));
    }
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if n == 0.0 {
        return Err(Error::invalid("rotation axis must not be the zero vector"));
    }
    let (s, c) = sin_cos_deg(deg);
    // Keep coordinate axes exact: dividing by a norm of 1 is exact already,
    // and a lone non-zero component is normalised to exactly +-1.
    let nonzero = axis.iter().filter(|v| **v != 0.0).count();
    let k = if nonzero == 1 { axis.map(|v| if v == 0.0 { 0.0 } else { v.signum() }) } else { axis.map(|v| v / n) };
    let (x, y, z) = (k[0], k[1], k[2]);
    let t = 1.0 - c;
    // Rodrigues: R = c I + s [k]x + (1 - c) k k^T; the diagonal entry of the
    // rotation axis itself is exactly 1, so that coordinate never changes.
    let d = |k: f64| if k * k == 1.0 { 1.0 } else { c + k * k * t };
    let r = Matrix3::new(
        d(x), x * y * t - z * s, x * z * t + y * s,
        y * x * t + z * s, d(y), y * z * t - x * s,
        z * x * t - y * s, z * y * t + x * s, d(z),
    );
    let tr = match about {
        None => Vector3::zeros(),
        Some(a) => {
            if a.iter().any(|v| !v.is_finite()) {
                return Err(Error::invalid("rotation centre must be finite"));
            }
            let a = Vector3::new(a[0], a[1], a[2]);
            a - r * a
        }
    };
    Ok(Transform::from_rt(r, tr))
}

/// Parse an axis name (`"x"`, `"y"`, `"z"`, case-insensitive) to a unit vector.
pub fn axis_from_name(name: &str) -> Result<[f64; 3]> {
    match name.trim().to_ascii_lowercase().as_str() {
        "x" => Ok([1.0, 0.0, 0.0]),
        "y" => Ok([0.0, 1.0, 0.0]),
        "z" => Ok([0.0, 0.0, 1.0]),
        other => Err(Error::invalid(format!("axis must be \"x\", \"y\", \"z\" or a 3-vector, got {other:?}"))),
    }
}

/// True when the transform is the identity plus a translation, so it can be
/// applied exactly as an addition.
pub fn is_pure_translation(t: &Transform) -> bool {
    let m = &t.0;
    (0..3).all(|r| (0..3).all(|c| m[(r, c)] == if r == c { 1.0 } else { 0.0 }))
        && m[(3, 0)] == 0.0
        && m[(3, 1)] == 0.0
        && m[(3, 2)] == 0.0
        && m[(3, 3)] == 1.0
}

/// Apply a transform to every point, in parallel.
///
/// A pure translation is applied as `p + t` (exact up to one rounding per
/// coordinate); anything else through the upper 3 x 4 block with
/// [`Transform::apply`], so results match `PointCloud.transform` to the bit.
pub fn apply_in_place(xyz: &mut [Point], t: &Transform) {
    let m = t.0;
    if is_pure_translation(t) {
        let (dx, dy, dz) = (m[(0, 3)], m[(1, 3)], m[(2, 3)]);
        xyz.par_chunks_mut(CHUNK).for_each(|chunk| {
            for p in chunk {
                p[0] += dx;
                p[1] += dy;
                p[2] += dz;
            }
        });
        return;
    }
    xyz.par_chunks_mut(CHUNK).for_each(|chunk| {
        for p in chunk {
            *p = t.apply(p);
        }
    });
}

/// [`apply_in_place`] on a copy.
pub fn apply(xyz: &[Point], t: &Transform) -> Vec<Point> {
    let mut out = xyz.to_vec();
    apply_in_place(&mut out, t);
    out
}

/// Default origin for recentring: the minimum corner of the finite points,
/// rounded down to whole metres (zeros for an empty cloud or one with no
/// finite coordinate on an axis).
///
/// Whole-metre origins keep the offset exactly representable and the local
/// coordinates short, so float precision is kept for projected coordinates
/// in the millions of metres.
pub fn recentre_origin(xyz: &[Point]) -> Point {
    let lo = xyz
        .par_chunks(CHUNK)
        .map(|chunk| {
            let mut lo = [f64::INFINITY; 3];
            for p in chunk {
                for k in 0..3 {
                    if p[k].is_finite() && p[k] < lo[k] {
                        lo[k] = p[k];
                    }
                }
            }
            lo
        })
        .reduce(|| [f64::INFINITY; 3], |a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]);
    lo.map(|v| if v.is_finite() { v.floor() } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_turns_are_exact() {
        let t = rotation([0.0, 0.0, 1.0], 90.0, None).unwrap();
        assert_eq!(t.apply(&[1.0, 2.0, 3.0]), [-2.0, 1.0, 3.0]);
        let t = rotation([1.0, 0.0, 0.0], -90.0, None).unwrap();
        assert_eq!(t.apply(&[1.0, 2.0, 3.0]), [1.0, 3.0, -2.0]);
        let t = rotation([0.0, 2.0, 0.0], 180.0, Some([1.0, 0.0, 1.0])).unwrap();
        assert_eq!(t.apply(&[2.0, 5.0, 1.0]), [0.0, 5.0, 1.0]);
        for k in -8..=8 {
            let (s, c) = sin_cos_deg(90.0 * k as f64);
            assert!(s.abs() == 0.0 || s.abs() == 1.0);
            assert!(c.abs() == 0.0 || c.abs() == 1.0);
        }
    }

    #[test]
    fn matches_rotation_z_and_rpy() {
        let a = rotation([0.0, 0.0, 1.0], 33.0, None).unwrap();
        let b = Transform::rotation_z(33.0);
        assert!((a.0 - b.0).abs().max() < 1e-15);
        let a = rotation([1.0, 0.0, 0.0], 12.0, None).unwrap();
        let b = Transform::from_roll_pitch_yaw(12.0, 0.0, 0.0);
        assert!((a.0 - b.0).abs().max() < 1e-15);
        let a = rotation([0.0, 1.0, 0.0], -7.0, None).unwrap();
        let b = Transform::from_roll_pitch_yaw(0.0, -7.0, 0.0);
        assert!((a.0 - b.0).abs().max() < 1e-15);
    }

    #[test]
    fn arbitrary_axis_is_a_rotation_about_that_axis() {
        let axis = [1.0, 2.0, -0.5];
        let t = rotation(axis, 41.0, Some([10.0, -3.0, 2.0])).unwrap();
        let r = t.rotation();
        assert!((r.transpose() * r - Matrix3::identity()).abs().max() < 1e-15);
        assert!((r.determinant() - 1.0).abs() < 1e-15);
        // Points on the axis through the centre do not move.
        let p = [10.0 + 2.0, -3.0 + 4.0, 2.0 - 1.0];
        let q = t.apply(&p);
        for k in 0..3 {
            assert!((p[k] - q[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn round_trips_do_not_drift() {
        let pts: Vec<Point> = (0..1000).map(|i| [500_000.0 + i as f64 * 0.37, 6_900_000.0 - i as f64 * 0.11, 100.0 + (i % 7) as f64]).collect();
        let mut xyz = pts.clone();
        let about = Some([500_100.0, 6_899_950.0, 0.0]);
        for _ in 0..100 {
            apply_in_place(&mut xyz, &rotation([0.0, 0.0, 1.0], 17.3, about).unwrap());
            apply_in_place(&mut xyz, &rotation([0.0, 0.0, 1.0], -17.3, about).unwrap());
            apply_in_place(&mut xyz, &Transform::translation(-500_000.0, -6_900_000.0, -100.0));
            apply_in_place(&mut xyz, &Transform::translation(500_000.0, 6_900_000.0, 100.0));
        }
        let worst = xyz.iter().zip(&pts).map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
        assert!(worst < 1e-6, "drift {worst}");
        // Whole-metre shifts of millimetre data round-trip exactly.
        let mut xyz = pts.clone();
        apply_in_place(&mut xyz, &Transform::translation(-500_000.0, -6_900_000.0, -100.0));
        apply_in_place(&mut xyz, &Transform::translation(500_000.0, 6_900_000.0, 100.0));
        assert_eq!(xyz, pts);
    }

    #[test]
    fn apply_matches_transform_apply() {
        let t = Transform::from_roll_pitch_yaw(3.0, -2.0, 40.0).compose(&Transform::translation(1.0, 2.0, 3.0));
        let pts: Vec<Point> = (0..200_000).map(|i| [i as f64 * 0.001, (i % 97) as f64, -(i as f64).sqrt()]).collect();
        let out = apply(&pts, &t);
        for (p, q) in pts.iter().zip(&out) {
            assert_eq!(t.apply(p), *q);
        }
    }

    #[test]
    fn origin_ignores_nan_and_handles_empty() {
        assert_eq!(recentre_origin(&[]), [0.0; 3]);
        let pts = vec![[f64::NAN, 5.5, -0.5], [500_000.7, 6.2, 3.0], [500_001.0, f64::INFINITY, 2.0]];
        assert_eq!(recentre_origin(&pts), [500_000.0, 5.0, -1.0]);
        assert_eq!(recentre_origin(&[[f64::NAN; 3]]), [0.0; 3]);
    }

    #[test]
    fn invalid_rotations() {
        assert!(rotation([0.0; 3], 10.0, None).is_err());
        assert!(rotation([0.0, 0.0, 1.0], f64::NAN, None).is_err());
        assert!(rotation([0.0, 0.0, 1.0], 1.0, Some([f64::NAN, 0.0, 0.0])).is_err());
        assert!(axis_from_name("w").is_err());
        assert_eq!(axis_from_name(" Y ").unwrap(), [0.0, 1.0, 0.0]);
    }
}
