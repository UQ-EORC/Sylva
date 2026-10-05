// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Cylinder fitting.

use crate::error::{Error, Result};
use crate::filters::Rng;
use crate::util::optim::levenberg_marquardt;
use crate::transform::{cross, dot, norm, normalize, scale, sub};
use crate::Point;

#[derive(Debug, Clone, PartialEq)]
pub struct CylinderFit {
    /// A point on the axis (centroid projected onto it).
    pub point: Point,
    /// Unit axis direction (z >= 0).
    pub axis: Point,
    pub radius: f64,
    pub rmse: f64,
}

fn axis_from_pca(xyz: &[Point]) -> Point {
    let n = xyz.len() as f64;
    let mut c = [0.0; 3];
    for p in xyz {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    for p in xyz {
        let d = nalgebra::Vector3::new(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        cov += d * d.transpose();
    }
    let eig = cov.symmetric_eigen();
    let imax = (0..3).max_by(|&a, &b| eig.eigenvalues[a].partial_cmp(&eig.eigenvalues[b]).unwrap()).unwrap();
    let v = eig.eigenvectors.column(imax);
    normalize(&[v[0], v[1], v[2]])
}

fn perp_basis(axis: &Point) -> (Point, Point) {
    let helper = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = normalize(&cross(axis, &helper));
    let v = cross(axis, &u);
    (u, v)
}

/// Signed radial distance of each point from the cylinder surface.
pub fn cylinder_residuals(xyz: &[Point], point: &Point, axis: &Point, radius: f64) -> Vec<f64> {
    let axis = normalize(axis);
    xyz.iter()
        .map(|p| {
            let d = sub(p, point);
            let along = dot(&d, &axis);
            let radial = sub(&d, &scale(&axis, along));
            norm(&radial) - radius
        })
        .collect()
}

/// Least-squares cylinder fit (5 parameters: two axis angles, two axis
/// offsets, radius) initialised from PCA.
pub fn fit_cylinder(xyz: &[Point], axis_init: Option<Point>) -> Result<CylinderFit> {
    if xyz.len() < 6 {
        return Err(Error::invalid("need >= 6 points for a cylinder fit"));
    }
    let n = xyz.len() as f64;
    let mut centroid = [0.0; 3];
    for p in xyz {
        for k in 0..3 {
            centroid[k] += p[k] / n;
        }
    }
    let axis0 = axis_init.map(|a| normalize(&a)).unwrap_or_else(|| axis_from_pca(xyz));
    let theta0 = axis0[2].clamp(-1.0, 1.0).acos();
    let phi0 = axis0[1].atan2(axis0[0]);
    let r0 = cylinder_residuals(xyz, &centroid, &axis0, 0.0).iter().sum::<f64>() / n;
    let (u, v) = perp_basis(&axis0);
    let unpack = |p: &[f64]| -> (Point, Point, f64) {
        let (theta, phi) = (p[0], p[1]);
        let axis = [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()];
        let pt = [
            centroid[0] + p[2] * u[0] + p[3] * v[0],
            centroid[1] + p[2] * u[1] + p[3] * v[1],
            centroid[2] + p[2] * u[2] + p[3] * v[2],
        ];
        (pt, axis, p[4])
    };
    let res = levenberg_marquardt(
        |p, out| {
            let (pt, axis, r) = unpack(p);
            *out = cylinder_residuals(xyz, &pt, &axis, r);
        },
        &[theta0, phi0, 0.0, 0.0, r0],
        100,
        1e-10,
    );
    let (pt, mut axis, r) = unpack(&res.x);
    if axis[2] < 0.0 {
        axis = scale(&axis, -1.0);
    }
    Ok(CylinderFit { point: pt, axis, radius: r.abs(), rmse: res.rmse })
}

/// RANSAC wrapper around [`fit_cylinder`]: returns the fit and the inlier mask.
pub fn fit_cylinder_ransac(xyz: &[Point], threshold: f64, iterations: usize, sample_size: usize, seed: u64) -> Result<(CylinderFit, Vec<bool>)> {
    let n = xyz.len();
    if n < sample_size.max(6) {
        let f = fit_cylinder(xyz, None)?;
        return Ok((f, vec![true; n]));
    }
    let mut rng = Rng::new(seed);
    let mut best: Vec<bool> = Vec::new();
    let mut best_n = 0;
    let axis_hint = axis_from_pca(xyz);
    for _ in 0..iterations {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..sample_size {
            let j = i + rng.below(n - i);
            idx.swap(i, j);
        }
        let sample: Vec<Point> = idx[..sample_size].iter().map(|&i| xyz[i]).collect();
        let Ok(f) = fit_cylinder(&sample, Some(axis_hint)) else { continue };
        let res = cylinder_residuals(xyz, &f.point, &f.axis, f.radius);
        let mask: Vec<bool> = res.iter().map(|r| r.abs() < threshold).collect();
        let cnt = mask.iter().filter(|&&b| b).count();
        if cnt > best_n {
            best_n = cnt;
            best = mask;
        }
    }
    if best_n < 6 {
        let f = fit_cylinder(xyz, None)?;
        return Ok((f, vec![true; n]));
    }
    let inl: Vec<Point> = xyz.iter().zip(&best).filter(|(_, &b)| b).map(|(p, _)| *p).collect();
    let f = fit_cylinder(&inl, Some(axis_hint))?;
    Ok((f, best))
}
