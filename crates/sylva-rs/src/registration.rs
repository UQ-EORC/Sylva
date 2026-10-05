// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Rigid registration of scan positions.

use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::filters::estimate_normals;
use crate::pointcloud::Attr;
use crate::util::spatial::KdTree;
use crate::transform::{cross, dot, sub, Transform};
use crate::{Point, PointCloud};

/// Least-squares rigid transform mapping `source` onto `target` (Kabsch 1976,
/// with Umeyama's 1991 reflection correction).
pub fn kabsch(source: &[Point], target: &[Point]) -> Result<Transform> {
    if source.len() != target.len() || source.len() < 3 {
        return Err(Error::invalid("kabsch needs >= 3 paired points"));
    }
    let n = source.len() as f64;
    let mut cs = Vector3::zeros();
    let mut ct = Vector3::zeros();
    for (s, t) in source.iter().zip(target) {
        cs += Vector3::from(*s);
        ct += Vector3::from(*t);
    }
    cs /= n;
    ct /= n;
    let mut h = Matrix3::zeros();
    for (s, t) in source.iter().zip(target) {
        h += (Vector3::from(*s) - cs) * (Vector3::from(*t) - ct).transpose();
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let d = (vt.transpose() * u.transpose()).determinant().signum();
    let r = vt.transpose() * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d)) * u.transpose();
    Ok(Transform::from_rt(r, ct - r * cs))
}

#[derive(Debug, Clone)]
pub struct IcpParams {
    pub max_correspondence_distance: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    /// `point` or `plane`.
    pub method: String,
    /// Fraction of closest correspondences kept (trimmed ICP); 1.0 = all.
    pub trim: f64,
    pub normal_k: usize,
}

impl Default for IcpParams {
    fn default() -> Self {
        IcpParams { max_correspondence_distance: 0.5, max_iterations: 50, tolerance: 1e-6, method: "point".into(), trim: 1.0, normal_k: 12 }
    }
}

#[derive(Debug, Clone)]
pub struct IcpResult {
    pub transform: Transform,
    pub rmse: f64,
    pub iterations: usize,
    pub n_correspondences: usize,
}

fn normals_of(cloud: &PointCloud, k: usize) -> Vec<Point> {
    match (cloud.attr("nx"), cloud.attr("ny"), cloud.attr("nz")) {
        (Some(nx), Some(ny), Some(nz)) => (0..cloud.len()).map(|i| [nx.get_f64(i), ny.get_f64(i), nz.get_f64(i)]).collect(),
        _ => estimate_normals(&cloud.xyz, k),
    }
}

/// Iterative closest point: point-to-point (Besl & McKay 1992) or
/// point-to-plane (Chen & Medioni 1992), with an optional trimmed
/// fraction of correspondences (Chetverikov et al. 2002).
pub fn icp(source: &PointCloud, target: &PointCloud, init: Option<Transform>, p: &IcpParams) -> Result<IcpResult> {
    let tree = KdTree::new(&target.xyz);
    let normals = if p.method == "plane" { Some(normals_of(target, p.normal_k)) } else { None };
    let mut t = init.unwrap_or_default();
    let mut prev_rmse = f64::INFINITY;
    let mut rmse = f64::INFINITY;
    let mut n_corr = 0;
    let mut iterations = 0;
    for it in 1..=p.max_iterations {
        iterations = it;
        let moved: Vec<Point> = source.xyz.iter().map(|q| t.apply(q)).collect();
        let mut corr: Vec<(usize, usize, f64)> = moved
            .par_iter()
            .enumerate()
            .filter_map(|(i, q)| tree.nearest(q).filter(|(_, d)| *d <= p.max_correspondence_distance).map(|(j, d)| (i, j, d)))
            .collect();
        if p.trim < 1.0 && !corr.is_empty() {
            corr.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
            let keep = ((corr.len() as f64 * p.trim).round() as usize).max(3);
            corr.truncate(keep);
        }
        n_corr = corr.len();
        if n_corr < 3 {
            return Err(Error::invalid("too few correspondences; increase max_correspondence_distance"));
        }
        rmse = (corr.iter().map(|c| c.2 * c.2).sum::<f64>() / n_corr as f64).sqrt();
        let src: Vec<Point> = corr.iter().map(|c| moved[c.0]).collect();
        let dst: Vec<Point> = corr.iter().map(|c| target.xyz[c.1]).collect();
        let step = match &normals {
            None => kabsch(&src, &dst)?,
            Some(n) => {
                let nn: Vec<Point> = corr.iter().map(|c| n[c.1]).collect();
                point_to_plane_step(&src, &dst, &nn)?
            }
        };
        t = step.compose(&t);
        if (prev_rmse - rmse).abs() < p.tolerance {
            break;
        }
        prev_rmse = rmse;
    }
    Ok(IcpResult { transform: t, rmse, iterations, n_correspondences: n_corr })
}

/// Linearised point-to-plane solve (Low 2004).
fn point_to_plane_step(p: &[Point], q: &[Point], n: &[Point]) -> Result<Transform> {
    let mut ata = Matrix6::<f64>::zeros();
    let mut atb = Vector6::<f64>::zeros();
    for i in 0..p.len() {
        let c = cross(&p[i], &n[i]);
        let row = Vector6::new(c[0], c[1], c[2], n[i][0], n[i][1], n[i][2]);
        let b = dot(&sub(&q[i], &p[i]), &n[i]);
        ata += row * row.transpose();
        atb += row * b;
    }
    let x = ata.lu().solve(&atb).ok_or_else(|| Error::invalid("singular point-to-plane system"))?;
    let (rx, ry, rz) = (x[0], x[1], x[2]);
    let rxm = Matrix3::new(1.0, 0.0, 0.0, 0.0, rx.cos(), -rx.sin(), 0.0, rx.sin(), rx.cos());
    let rym = Matrix3::new(ry.cos(), 0.0, ry.sin(), 0.0, 1.0, 0.0, -ry.sin(), 0.0, ry.cos());
    let rzm = Matrix3::new(rz.cos(), -rz.sin(), 0.0, rz.sin(), rz.cos(), 0.0, 0.0, 0.0, 1.0);
    Ok(Transform::from_rt(rzm * rym * rxm, Vector3::new(x[3], x[4], x[5])))
}

/// Transform each cloud and concatenate, adding a `scan_id` attribute.
pub fn merge_scans(clouds: &[PointCloud], transforms: Option<&[Transform]>) -> Result<PointCloud> {
    let mut parts = Vec::with_capacity(clouds.len());
    for (i, c) in clouds.iter().enumerate() {
        let mut c = match transforms {
            Some(t) => c.transformed(&t[i]),
            None => c.clone(),
        };
        c.attrs.insert("scan_id".into(), Attr::I32(vec![i as i32; c.len()]));
        parts.push(c);
    }
    PointCloud::concatenate(&parts)
}
