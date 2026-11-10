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
/// Settings of [`icp`]: how correspondences are chosen, when to stop, and
/// which error metric is minimised.
pub struct IcpParams {
    /// Pairs further apart than this (m) are ignored; the start must be within about this of the solution.
    pub max_correspondence_distance: f64,
    /// Iteration limit.
    pub max_iterations: usize,
    /// Stop when the RMSE changes by less than this (m) between iterations.
    pub tolerance: f64,
    /// `"point"` (point-to-point) or `"plane"` (point-to-plane); anything else
    /// is an error, not a silent point-to-point run.
    pub method: String,
    /// Fraction of closest correspondences kept (trimmed ICP); 1.0 = all.
    pub trim: f64,
    /// Neighbours for the PCA normals of the `plane` method when the target has no `nx`, `ny`, `nz` columns.
    pub normal_k: usize,
}

impl Default for IcpParams {
    fn default() -> Self {
        IcpParams { max_correspondence_distance: 0.5, max_iterations: 50, tolerance: 1e-6, method: "point".into(), trim: 1.0, normal_k: 12 }
    }
}

#[derive(Debug, Clone)]
/// What [`icp`] converged to.
pub struct IcpResult {
    /// Transform mapping `source` onto `target`, the initial transform included.
    pub transform: Transform,
    /// Root-mean-square distance of the kept correspondences at the last iteration (m).
    pub rmse: f64,
    /// Iterations run before converging or reaching `max_iterations`.
    pub iterations: usize,
    /// Correspondences kept at the last iteration, after trimming.
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
///
/// With `method = "plane"` the target's normals are its `nx`, `ny`, `nz`
/// attributes if it has all three, else PCA normals over `normal_k`
/// neighbours.
///
/// # Errors
/// If `method` is neither `"point"` nor `"plane"`, or fewer than 3
/// correspondences lie within `max_correspondence_distance` (m).
pub fn icp(source: &PointCloud, target: &PointCloud, init: Option<Transform>, p: &IcpParams) -> Result<IcpResult> {
    if !matches!(p.method.as_str(), "point" | "plane") {
        return Err(Error::invalid(format!("unknown ICP method {:?}: use \"point\" or \"plane\"", p.method)));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A tilted plane, 0.1 m grid.
    fn plane() -> Vec<Point> {
        (0..40).flat_map(|i| (0..40).map(move |j| [0.1 * i as f64, 0.1 * j as f64, 0.02 * i as f64])).collect()
    }

    #[test]
    fn an_unknown_method_is_an_error_not_point_to_point() {
        let cloud = PointCloud::new(plane());
        // The coregistration ICP's names, easily typed here by mistake, used to
        // run point-to-point without a word.
        for method in ["point_to_plane", "point_to_point", "Plane", ""] {
            let p = IcpParams { method: method.into(), ..Default::default() };
            let e = icp(&cloud, &cloud, None, &p).unwrap_err().to_string();
            assert!(e.contains("unknown ICP method"), "{method}: {e}");
        }
        for method in ["point", "plane"] {
            let p = IcpParams { method: method.into(), max_iterations: 2, ..Default::default() };
            assert!(icp(&cloud, &cloud, None, &p).is_ok());
        }
    }

    /// Three faces of a corner, 0.1 m grid: every direction is constrained.
    fn corner() -> Vec<Point> {
        let mut pts = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                let (u, v) = (0.1 * i as f64, 0.1 * j as f64);
                pts.extend([[u, v, 0.0], [u, 0.0, v + 0.1], [0.0, u + 0.1, v + 0.1]]);
            }
        }
        pts
    }

    #[test]
    fn the_targets_own_normals_are_used_for_point_to_plane() {
        // With PCA normals a 2 cm offset is recovered. Normals of NaN in the
        // target's nx, ny, nz cannot give a transform at all, which shows they
        // were the ones used.
        let pts = corner();
        let start = Some(Transform::translation(0.02, -0.02, 0.02));
        let p = IcpParams { method: "plane".into(), max_correspondence_distance: 0.2, ..Default::default() };
        let source = PointCloud::new(pts.clone());
        let pca = icp(&source, &PointCloud::new(pts.clone()), start, &p).unwrap();
        assert!(pca.transform.translation_vec().norm() < 5e-3, "{}", pca.transform.translation_vec());
        let mut target = PointCloud::new(pts.clone());
        for name in ["nx", "ny", "nz"] {
            target.set_attr(name, Attr::F64(vec![f64::NAN; pts.len()])).unwrap();
        }
        let usable = icp(&source, &target, start, &p).is_ok_and(|r| r.transform.0.iter().all(|v| v.is_finite()));
        assert!(!usable);
    }
}
