// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Rigid transforms (SO(3) and SE(3)) for coregistration.
//!
//! Transforms are 4x4 homogeneous matrices in column-vector convention,
//! `q = T [p, 1]`. Twists are ordered `xi = [wx, wy, wz, tx, ty, tz]`,
//! rotation first. The exponential and logarithm maps, the weighted Kabsch
//! (1976) fit and the point transform are those ICP uses
//! ([`crate::coreg::icp`]); the yaw-only fit is the stem matcher's
//! ([`crate::coreg::matching::kabsch_yaw`]). Arithmetic follows the NumPy code it
//! replaced (`sylva.coreg.transforms`).

use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Vector3, Vector6};

pub use crate::coreg::icp::{kabsch_weighted, left_jacobian, se3_exp, skew, so3_exp, so3_log, transform_points, Mat4};
use crate::coreg::matching::numpy_sum;
use crate::error::{Error, Result};
use crate::Point;

/// `np.linalg.solve(a, b)` for a square system: LU with partial pivoting
/// (LAPACK's `dgesv`). `None` for an exactly singular matrix.
pub fn solve(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut x = b.clone();
    for k in 0..n {
        let mut p = k;
        let mut best = lu[(k, k)].abs();
        for i in k + 1..n {
            let v = lu[(i, k)].abs();
            if v > best || (best.is_nan() && !v.is_nan()) {
                best = v;
                p = i;
            }
        }
        if lu[(p, k)] == 0.0 {
            return None;
        }
        if p != k {
            lu.swap_rows(p, k);
            x.swap_rows(p, k);
        }
        let inv = 1.0 / lu[(k, k)];
        for i in k + 1..n {
            lu[(i, k)] *= inv;
        }
        for j in k + 1..n {
            let u = lu[(k, j)];
            if u != 0.0 {
                for i in k + 1..n {
                    lu[(i, j)] -= lu[(i, k)] * u;
                }
            }
        }
    }
    for j in 0..n {
        let v = x[j];
        if v != 0.0 {
            for i in j + 1..n {
                x[i] -= v * lu[(i, j)];
            }
        }
    }
    for j in (0..n).rev() {
        x[j] /= lu[(j, j)];
        let v = x[j];
        if v != 0.0 {
            for i in 0..j {
                x[i] -= v * lu[(i, j)];
            }
        }
    }
    Some(x)
}

/// A 3x3 system (the left Jacobian in [`se3_log`]).
fn solve3(a: &Matrix3<f64>, b: &Vector3<f64>) -> Vector3<f64> {
    let m = DMatrix::from_fn(3, 3, |r, c| a[(r, c)]);
    match solve(&m, &DVector::from_column_slice(b.as_slice())) {
        Some(x) => Vector3::new(x[0], x[1], x[2]),
        None => Vector3::from_element(f64::NAN),
    }
}

/// Twist `[w, t]` of a transform; the inverse of [`se3_exp`].
pub fn se3_log(t: &Mat4) -> Vector6<f64> {
    let r: Matrix3<f64> = t.fixed_view::<3, 3>(0, 0).into();
    let w = so3_log(&r);
    let v = solve3(&left_jacobian(&w), &Vector3::new(t[(0, 3)], t[(1, 3)], t[(2, 3)]));
    Vector6::new(w[0], w[1], w[2], v[0], v[1], v[2])
}

/// Inverse of a rigid transform.
pub fn invert(t: &Mat4) -> Mat4 {
    let mut out = Matrix4::identity();
    for i in 0..3 {
        for j in 0..3 {
            out[(i, j)] = t[(j, i)];
        }
    }
    for i in 0..3 {
        out[(i, 3)] = -(t[(0, i)] * t[(0, 3)] + t[(1, i)] * t[(1, 3)] + t[(2, i)] * t[(2, 3)]);
    }
    out
}

/// Rotate directions (the translation is ignored).
pub fn transform_vectors(t: &Mat4, v: &[Point]) -> Vec<Point> {
    v.iter()
        .map(|p| {
            let mut o = [0.0; 3];
            for (i, oi) in o.iter_mut().enumerate() {
                *oi = p[0] * t[(i, 0)] + p[1] * t[(i, 1)] + p[2] * t[(i, 2)];
            }
            o
        })
        .collect()
}

/// A rotation of `yaw` radians about +z followed by a translation.
pub fn yaw_transform(yaw: f64, tx: f64, ty: f64, tz: f64) -> Mat4 {
    let (c, s) = (yaw.cos(), yaw.sin());
    Matrix4::new(c, -s, 0.0, tx, s, c, 0.0, ty, 0.0, 0.0, 1.0, tz, 0.0, 0.0, 0.0, 1.0)
}

/// Least-squares rigid transform mapping `source` onto `target`, optionally
/// weighted; reflections are suppressed.
pub fn kabsch(source: &[Point], target: &[Point], weights: Option<&[f64]>) -> Result<Mat4> {
    if source.len() != target.len() {
        return Err(Error::invalid("source and target must both be (N, 3) arrays of equal length"));
    }
    if source.len() < 3 {
        return Err(Error::invalid("at least 3 correspondences are required"));
    }
    let ones;
    let w = match weights {
        Some(w) => {
            if w.len() != source.len() {
                return Err(Error::invalid("weights must have one value per point"));
            }
            w
        }
        None => {
            ones = vec![1.0; source.len()];
            &ones
        }
    };
    if numpy_sum(w) <= 1e-12 {
        return Err(Error::invalid("weights sum to zero"));
    }
    kabsch_weighted(source, target, w).ok_or_else(|| Error::invalid("SVD did not converge"))
}

/// Least-squares yaw and translation (4 degrees of freedom) between point
/// sets, the right estimator for levelled scans.
pub fn kabsch_2d_yaw(source: &[Point], target: &[Point], weights: Option<&[f64]>) -> Result<Mat4> {
    if source.len() != target.len() || source.len() < 2 {
        return Err(Error::invalid("source and target must be (N, 3) arrays of equal length, N >= 2"));
    }
    let Some(w) = weights else {
        return Ok(crate::coreg::matching::kabsch_yaw(source, target).0);
    };
    if w.len() != source.len() {
        return Err(Error::invalid("weights must have one value per point"));
    }
    let total = numpy_sum(w);
    if total <= 1e-12 {
        return Err(Error::invalid("weights sum to zero"));
    }
    let w: Vec<f64> = w.iter().map(|v| v / total).collect();
    let mut ms = [0.0; 3];
    let mut md = [0.0; 3];
    for (i, ((a, b), wi)) in source.iter().zip(target).zip(&w).enumerate() {
        for k in 0..3 {
            if i == 0 {
                ms[k] = wi * a[k];
                md[k] = wi * b[k];
            } else {
                ms[k] += wi * a[k];
                md[k] += wi * b[k];
            }
        }
    }
    let mut num = Vec::with_capacity(w.len());
    let mut den = Vec::with_capacity(w.len());
    for ((a, b), wi) in source.iter().zip(target).zip(&w) {
        let (ax, ay) = (a[0] - ms[0], a[1] - ms[1]);
        let (bx, by) = (b[0] - md[0], b[1] - md[1]);
        num.push(wi * (ax * by - ay * bx));
        den.push(wi * (ax * bx + ay * by));
    }
    let yaw = numpy_sum(&num).atan2(numpy_sum(&den));
    let r = so3_exp(&Vector3::new(0.0, 0.0, yaw));
    let t = Vector3::new(md[0] - r[(0, 0)].mul_add(ms[0], r[(0, 1)] * ms[1]), md[1] - r[(1, 0)].mul_add(ms[0], r[(1, 1)] * ms[1]), md[2] - ms[2]);
    let mut out = Matrix4::identity();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    out.fixed_view_mut::<3, 1>(0, 3).copy_from(&t);
    Ok(out)
}

/// Rotation magnitude of a transform, in radians (exact near zero).
pub fn rotation_angle(t: &Mat4) -> f64 {
    let r: Matrix3<f64> = t.fixed_view::<3, 3>(0, 0).into();
    so3_log(&r).norm()
}

/// `(rotation_rad, translation_m)` of the discrepancy `A^-1 B`.
pub fn transform_difference(a: &Mat4, b: &Mat4) -> (f64, f64) {
    let d = invert(a) * b;
    (rotation_angle(&d), (d[(0, 3)] * d[(0, 3)] + d[(1, 3)] * d[(1, 3)] + d[(2, 3)] * d[(2, 3)]).sqrt())
}

/// A 4x4 transform from 16 row-major values.
pub fn mat4_from_rows(v: &[f64]) -> Result<Mat4> {
    if v.len() != 16 {
        return Err(Error::invalid(format!("a transform needs 16 values, got {}", v.len())));
    }
    Ok(Matrix4::from_row_slice(v))
}

/// The 16 row-major values of a 4x4 transform.
pub fn mat4_to_rows(t: &Mat4) -> [f64; 16] {
    let mut out = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[4 * r + c] = t[(r, c)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Mat4, b: &Mat4, tol: f64) -> bool {
        (a - b).abs().max() < tol
    }

    #[test]
    fn exp_and_log_are_inverse() {
        for xi in [Vector6::new(0.3, -0.2, 1.1, 4.0, -2.0, 0.5), Vector6::new(1e-10, 0.0, -2e-10, 1.0, 2.0, 3.0), Vector6::new(0.0, 0.0, 3.1, 0.0, 1.0, 0.0)] {
            let back = se3_log(&se3_exp(&xi));
            assert!((back - xi).abs().max() < 1e-9, "{xi:?} -> {back:?}");
        }
    }

    #[test]
    fn invert_undoes_a_transform() {
        let t = se3_exp(&Vector6::new(0.1, 0.4, -0.7, 3.0, -1.0, 2.0));
        assert!(close(&(invert(&t) * t), &Matrix4::identity(), 1e-12));
        let (r, d) = transform_difference(&t, &t);
        assert!(r < 1e-7 && d < 1e-12);
    }

    #[test]
    fn kabsch_recovers_a_transform_and_suppresses_reflections() {
        let t = se3_exp(&Vector6::new(0.2, -0.1, 0.9, 1.0, 2.0, -3.0));
        let src: Vec<Point> = (0..10).map(|i| [(i as f64).sin() * 5.0, (i as f64 * 1.7).cos() * 4.0, i as f64 * 0.3]).collect();
        let dst = transform_points(&t, &src);
        assert!(close(&kabsch(&src, &dst, None).unwrap(), &t, 1e-9));
        let w: Vec<f64> = (0..10).map(|i| 1.0 + i as f64).collect();
        assert!(close(&kabsch(&src, &dst, Some(&w)).unwrap(), &t, 1e-9));
        let mirrored: Vec<Point> = src.iter().map(|p| [p[0], p[1], -p[2]]).collect();
        let r = kabsch(&src, &mirrored, None).unwrap();
        let rot: Matrix3<f64> = r.fixed_view::<3, 3>(0, 0).into();
        assert!((rot.determinant() - 1.0).abs() < 1e-9);
        assert!(kabsch(&src[..2], &dst[..2], None).is_err());
        assert!(kabsch(&src, &dst, Some(&[0.0; 10])).is_err());
    }

    #[test]
    fn yaw_fit_ignores_tilt() {
        let t = yaw_transform(0.8, 2.0, -1.0, 0.3);
        let src: Vec<Point> = (0..8).map(|i| [(i as f64).sin() * 5.0, (i as f64 * 1.3).cos() * 4.0, 0.0]).collect();
        let dst = transform_points(&t, &src);
        assert!(close(&kabsch_2d_yaw(&src, &dst, None).unwrap(), &t, 1e-9));
        assert!(close(&kabsch_2d_yaw(&src, &dst, Some(&[2.0; 8])).unwrap(), &t, 1e-9));
        assert!(kabsch_2d_yaw(&src[..1], &dst[..1], None).is_err());
    }

    #[test]
    fn solve_pivots() {
        let a = DMatrix::from_row_slice(3, 3, &[0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 1.0]);
        let x = solve(&a, &DVector::from_vec(vec![5.0, 3.0, 6.0])).unwrap();
        let back = &a * &x;
        assert!((back - DVector::from_vec(vec![5.0, 3.0, 6.0])).abs().max() < 1e-12);
        assert!(solve(&DMatrix::zeros(2, 2), &DVector::zeros(2)).is_none());
    }
}
