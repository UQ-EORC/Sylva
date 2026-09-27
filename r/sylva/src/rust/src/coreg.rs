// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for coregistration: rigid transforms, reflective targets, the
//! pose graph, joint refinement, stem maps, stem matching, ICP, local
//! geometry and the terrain model. Indices cross 0-based; `R/coreg*.R`
//! presents them 1-based. Transforms are 4x4 matrices, point sets n x 3.

use std::collections::HashMap;

use extendr_api::prelude::*;
use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector6};
use sylva_rs::coreg as cm;
use sylva_rs::coreg_geometry as geo;
use sylva_rs::coreg_ground as gr;
use sylva_rs::coreg_icp as icp;
use sylva_rs::coreg_posegraph as pg;
use sylva_rs::coreg_reflectors as rf;
use sylva_rs::coreg_refine as refine;
use sylva_rs::coreg_stemmap as sm;
use sylva_rs::coreg_transforms as tf;
use sylva_rs::stems::{detect_stems_full, StemParams};
use sylva_rs::Point;

use crate::convert::{doubles, err, fail, xyz_from_r, xyz_to_r, Result};

// ----------------------------------------------------------------- converters

fn square<const N: usize>(m: &Robj, what: &str) -> Result<nalgebra::SMatrix<f64, N, N>> {
    let v: RMatrix<f64> = m.try_into().map_err(|_| Error::Other(format!("{what} must be a {N} x {N} numeric matrix")))?;
    if v.nrows() != N || v.ncols() != N {
        return fail(format!("{what} must be a {N} x {N} matrix, got {} x {}", v.nrows(), v.ncols()));
    }
    Ok(nalgebra::SMatrix::from_column_slice(v.data()))
}

fn mat4(m: &Robj) -> Result<Matrix4<f64>> {
    square::<4>(m, "a transform")
}

fn mat_to_r<const R: usize, const C: usize>(m: &nalgebra::SMatrix<f64, R, C>) -> Robj {
    RMatrix::new_matrix(R, C, |r, c| m[(r, c)]).into()
}

fn mats<const N: usize>(list: &List, what: &str) -> Result<Vec<nalgebra::SMatrix<f64, N, N>>> {
    list.values().map(|m| square::<N>(&m, what)).collect()
}

fn mats_to_r(m: &[Matrix4<f64>]) -> List {
    List::from_values(m.iter().map(mat_to_r))
}

fn indices(v: &Robj, what: &str) -> Result<Vec<usize>> {
    doubles(v, what)?
        .into_iter()
        .map(|x| if x >= 0.0 && x.fract() == 0.0 { Ok(x as usize) } else { fail(format!("{what} must be non-negative whole numbers")) })
        .collect()
}

fn opt_doubles(v: &Robj, what: &str) -> Result<Option<Vec<f64>>> {
    if v.is_null() {
        Ok(None)
    } else {
        doubles(v, what).map(Some)
    }
}

fn opt_f64(v: &Robj, what: &str) -> Result<Option<f64>> {
    Ok(opt_doubles(v, what)?.and_then(|x| x.first().copied()))
}

fn clouds(list: &List) -> Result<Vec<Vec<Point>>> {
    list.values().map(|m| xyz_from_r(&m)).collect()
}

/// Points of a 2- or 3-column matrix, 2-D ones at z = 0.
fn padded(m: &Robj, what: &str) -> Result<(Vec<Point>, usize)> {
    let v: RMatrix<f64> = m.try_into().map_err(|_| Error::Other(format!("{what} must be a numeric matrix")))?;
    let (n, d) = (v.nrows(), v.data());
    match v.ncols() {
        3 => Ok(((0..n).map(|i| [d[i], d[n + i], d[2 * n + i]]).collect(), 3)),
        2 => Ok(((0..n).map(|i| [d[i], d[n + i], 0.0]).collect(), 2)),
        c => fail(format!("{what} must have 2 or 3 columns, got {c}")),
    }
}

fn xy_rows(m: &Robj) -> Result<Vec<[f64; 2]>> {
    let (p, _) = padded(m, "xy")?;
    Ok(p.into_iter().map(|q| [q[0], q[1]]).collect())
}

/// A row-major grid from an R matrix (rows are y).
fn grid_from_r<T: Copy>(data: &[T], ny: usize, nx: usize) -> Vec<T> {
    (0..ny).flat_map(|r| (0..nx).map(move |c| data[c * ny + r])).collect()
}

fn params(config: &List) -> Result<HashMap<&str, Robj>> {
    Ok(config.clone().try_into()?)
}

fn get<'a>(m: &'a HashMap<&str, Robj>, k: &str) -> Result<&'a Robj> {
    m.get(k).ok_or_else(|| Error::Other(format!("configuration has no `{k}`")))
}

fn num(m: &HashMap<&str, Robj>, k: &str) -> Result<f64> {
    doubles(get(m, k)?, k)?.first().copied().ok_or_else(|| Error::Other(format!("`{k}` is empty")))
}

fn count(m: &HashMap<&str, Robj>, k: &str) -> Result<usize> {
    let v = num(m, k)?;
    if v < 0.0 {
        return fail(format!("`{k}` must not be negative"));
    }
    Ok(v as usize)
}

fn flag(m: &HashMap<&str, Robj>, k: &str) -> Result<bool> {
    get(m, k)?.as_bool().ok_or_else(|| Error::Other(format!("`{k}` must be TRUE or FALSE")))
}

fn text(m: &HashMap<&str, Robj>, k: &str) -> Result<String> {
    get(m, k)?.as_str().map(str::to_string).ok_or_else(|| Error::Other(format!("`{k}` must be a string")))
}

// ----------------------------------------------------------------- transforms

/// @noRd
#[extendr]
fn core_coreg_skew(v: &[f64]) -> Result<Robj> {
    if v.len() != 3 {
        return fail("a 3-vector is needed");
    }
    Ok(mat_to_r(&tf::skew(&Vector3::from_column_slice(v))))
}

/// @noRd
#[extendr]
fn core_coreg_so3_exp(w: &[f64]) -> Result<Robj> {
    if w.len() != 3 {
        return fail("a rotation vector has 3 values");
    }
    Ok(mat_to_r(&tf::so3_exp(&Vector3::from_column_slice(w))))
}

/// @noRd
#[extendr]
fn core_coreg_so3_log(r: Robj) -> Result<Vec<f64>> {
    let r: Matrix3<f64> = square::<3>(&r, "a rotation")?;
    Ok(tf::so3_log(&r).as_slice().to_vec())
}

/// @noRd
#[extendr]
fn core_coreg_rotation_angle(r: Robj) -> Result<f64> {
    let r: Matrix3<f64> = square::<3>(&r, "a rotation")?;
    Ok(tf::so3_log(&r).norm())
}

/// @noRd
#[extendr]
fn core_coreg_se3_exp(xi: &[f64]) -> Result<Robj> {
    if xi.len() != 6 {
        return fail("a twist has 6 values");
    }
    Ok(mat_to_r(&tf::se3_exp(&Vector6::from_column_slice(xi))))
}

/// @noRd
#[extendr]
fn core_coreg_se3_log(t: Robj) -> Result<Vec<f64>> {
    Ok(tf::se3_log(&mat4(&t)?).as_slice().to_vec())
}

/// @noRd
#[extendr]
fn core_coreg_invert(t: Robj) -> Result<Robj> {
    Ok(mat_to_r(&tf::invert(&mat4(&t)?)))
}

/// @noRd
#[extendr]
fn core_coreg_transform_points(t: Robj, points: Robj) -> Result<Robj> {
    Ok(xyz_to_r(&tf::transform_points(&mat4(&t)?, &xyz_from_r(&points)?)))
}

/// @noRd
#[extendr]
fn core_coreg_transform_vectors(t: Robj, vectors: Robj) -> Result<Robj> {
    Ok(xyz_to_r(&tf::transform_vectors(&mat4(&t)?, &xyz_from_r(&vectors)?)))
}

/// @noRd
#[extendr]
fn core_coreg_yaw_transform(yaw: f64, tx: f64, ty: f64, tz: f64) -> Robj {
    mat_to_r(&tf::yaw_transform(yaw, tx, ty, tz))
}

/// @noRd
#[extendr]
fn core_coreg_kabsch(source: Robj, target: Robj, weights: Robj) -> Result<Robj> {
    let w = opt_doubles(&weights, "weights")?;
    Ok(mat_to_r(&tf::kabsch(&xyz_from_r(&source)?, &xyz_from_r(&target)?, w.as_deref()).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_coreg_kabsch_2d_yaw(source: Robj, target: Robj, weights: Robj) -> Result<Robj> {
    let w = opt_doubles(&weights, "weights")?;
    Ok(mat_to_r(&tf::kabsch_2d_yaw(&xyz_from_r(&source)?, &xyz_from_r(&target)?, w.as_deref()).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_coreg_transform_difference(a: Robj, b: Robj) -> Result<Vec<f64>> {
    let (r, t) = tf::transform_difference(&mat4(&a)?, &mat4(&b)?);
    Ok(vec![r, t])
}

// ----------------------------------------------------------------- reflectors

fn reflectors_to_r(v: Vec<rf::Reflector>) -> List {
    list!(
        x = v.iter().map(|r| r.x).collect::<Vec<_>>(),
        y = v.iter().map(|r| r.y).collect::<Vec<_>>(),
        z = v.iter().map(|r| r.z).collect::<Vec<_>>(),
        reflectance = v.iter().map(|r| r.reflectance).collect::<Vec<_>>(),
        diameter = v.iter().map(|r| r.diameter).collect::<Vec<_>>(),
        n_points = v.iter().map(|r| r.n_points as f64).collect::<Vec<_>>(),
        name = v.iter().map(|r| r.name.as_str()).collect::<Strings>()
    )
}

/// @noRd
#[extendr]
fn core_coreg_read_tiepoint_list(path: &str) -> Result<List> {
    Ok(reflectors_to_r(rf::read_tiepoint_list(path).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_coreg_read_reflector_list(path: &str) -> Result<List> {
    Ok(reflectors_to_r(rf::read_reflector_list(path).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_coreg_detect_reflectors(xyz: Robj, reflectance: Robj, min_reflectance: f64, cluster_radius: f64, min_points: f64, max_extent: f64) -> Result<List> {
    let r = opt_doubles(&reflectance, "reflectance")?;
    let found = rf::detect_reflectors(&xyz_from_r(&xyz)?, r.as_deref(), min_reflectance, cluster_radius, min_points.max(0.0) as usize, max_extent).map_err(err)?;
    Ok(reflectors_to_r(found))
}

/// @noRd
#[extendr]
fn core_coreg_match_reflectors(source: Robj, target: Robj, tolerance: f64, min_inliers: f64, distance_tolerance: f64) -> Result<List> {
    let m = rf::match_reflectors(&xyz_from_r(&source)?, &xyz_from_r(&target)?, tolerance, min_inliers as i64, distance_tolerance).map_err(err)?;
    let c = &m.correspondences;
    Ok(list!(
        transform = mat_to_r(&m.transform),
        n_inliers = m.n_inliers as f64,
        rmse = m.rmse,
        correspondences = RMatrix::new_matrix(c.len(), 2, |r, k| c[r][k] as f64),
        success = m.success
    ))
}

// ----------------------------------------------------------------- pose graph

/// @noRd
#[extendr]
fn core_coreg_default_information(rmse: f64, fitness: f64, n_correspondences: f64, extent: f64) -> Robj {
    mat_to_r(&pg::default_information(rmse, fitness, n_correspondences as i64, extent))
}

/// @noRd
#[extendr]
fn core_coreg_adjoint(t: Robj) -> Result<Robj> {
    Ok(mat_to_r(&pg::adjoint(&mat4(&t)?)))
}

/// @noRd
#[extendr]
fn core_coreg_plane_edge_information(hessian: Robj, sigma: f64, n: f64, transform: Robj, patch_points: f64, min_sigma: f64) -> Result<Robj> {
    let h: Matrix6<f64> = square::<6>(&hessian, "hessian")?;
    Ok(mat_to_r(&pg::plane_edge_information(&h, sigma, n as i64, &mat4(&transform)?, patch_points, min_sigma)))
}

fn edges_from_r(n: usize, i: &Robj, j: &Robj, transforms: &List, information: Option<&List>, weights: Option<&Robj>) -> Result<Vec<pg::Edge>> {
    let (i, j) = (indices(i, "i")?, indices(j, "j")?);
    let t = mats::<4>(transforms, "an edge transform")?;
    let info = match information {
        Some(l) => mats::<6>(l, "an information matrix")?,
        None => vec![Matrix6::identity(); t.len()],
    };
    let w = match weights {
        Some(w) => doubles(w, "weights")?,
        None => vec![1.0; t.len()],
    };
    if i.len() != t.len() || j.len() != t.len() || info.len() != t.len() || w.len() != t.len() {
        return fail("one i, j, transform, information and weight per edge");
    }
    if i.iter().chain(&j).any(|&k| k >= n) {
        return fail("edge endpoints out of range");
    }
    Ok((0..t.len()).map(|k| pg::Edge { i: i[k], j: j[k], transform: t[k], information: info[k], weight: w[k] }).collect())
}

fn fixed_from_r(n: usize, nodes: &Robj, poses: &List) -> Result<Vec<(usize, Matrix4<f64>)>> {
    let k = indices(nodes, "fixed nodes")?;
    let p = mats::<4>(poses, "a fixed pose")?;
    if k.len() != p.len() || k.iter().any(|&v| v >= n) {
        return fail("fixed nodes and poses do not match");
    }
    Ok(k.into_iter().zip(p).collect())
}

/// @noRd
#[extendr]
fn core_coreg_posegraph_residuals(i: Robj, j: Robj, transforms: List, poses: List) -> Result<Robj> {
    let p = mats::<4>(&poses, "a pose")?;
    let edges = edges_from_r(p.len(), &i, &j, &transforms, None, None)?;
    let r: Vec<Vector6<f64>> = edges.iter().map(|e| pg::residual(e, &p)).collect();
    Ok(RMatrix::new_matrix(r.len(), 6, |a, b| r[a][b]).into())
}

/// @noRd
#[extendr]
fn core_coreg_posegraph_total_error(i: Robj, j: Robj, transforms: List, information: List, poses: List) -> Result<f64> {
    let p = mats::<4>(&poses, "a pose")?;
    let edges = edges_from_r(p.len(), &i, &j, &transforms, Some(&information), None)?;
    let all: Vec<usize> = (0..edges.len()).collect();
    Ok(pg::total_error(&edges, &all, &p))
}

/// @noRd
#[extendr]
fn core_coreg_posegraph_components(n: f64, i: Robj, j: Robj) -> Result<List> {
    let n = n as usize;
    let (i, j) = (indices(&i, "i")?, indices(&j, "j")?);
    if i.len() != j.len() || i.iter().chain(&j).any(|&k| k >= n) {
        return fail("edge endpoints out of range");
    }
    let edges: Vec<pg::Edge> = i.iter().zip(&j).map(|(&a, &b)| pg::Edge { i: a, j: b, transform: Matrix4::identity(), information: Matrix6::identity(), weight: 1.0 }).collect();
    Ok(List::from_values(pg::components(n, &edges).into_iter().map(|c| c.into_iter().map(|k| k as f64).collect::<Vec<_>>())))
}

/// @noRd
#[extendr]
fn core_coreg_posegraph_initialise(n: f64, i: Robj, j: Robj, transforms: List, weights: Robj, reference: f64, fixed_nodes: Robj, fixed_poses: List) -> Result<List> {
    let n = n as usize;
    let edges = edges_from_r(n, &i, &j, &transforms, None, Some(&weights))?;
    let fixed = fixed_from_r(n, &fixed_nodes, &fixed_poses)?;
    if reference as usize >= n {
        return fail("reference node out of range");
    }
    Ok(mats_to_r(&pg::initialise(n, &edges, reference as usize, &fixed)))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_coreg_posegraph_optimise(n: f64, i: Robj, j: Robj, transforms: List, information: List, weights: Robj, reference: f64, fixed_nodes: Robj, fixed_poses: List, poses: List, max_iterations: f64, tolerance: f64, huber_delta: f64, reject_outliers: bool, outlier_sigma: f64, max_rejection_passes: f64) -> Result<List> {
    let n = n as usize;
    let edges = edges_from_r(n, &i, &j, &transforms, Some(&information), Some(&weights))?;
    let fixed = fixed_from_r(n, &fixed_nodes, &fixed_poses)?;
    let start = mats::<4>(&poses, "a pose")?;
    if start.len() != n || reference as usize >= n {
        return fail("one pose per node, and a reference among them");
    }
    let p = pg::OptimiseParams { max_iterations: max_iterations.max(0.0) as usize, tolerance, huber_delta, reject_outliers, outlier_sigma, max_rejection_passes: max_rejection_passes as i64 };
    let r = pg::optimise(n, &edges, reference as usize, &fixed, start, &p);
    Ok(list!(
        poses = mats_to_r(&r.poses),
        iterations = r.iterations as f64,
        converged = r.converged,
        initial_error = r.initial_error,
        final_error = r.final_error,
        rejected_edges = r.rejected_edges.iter().map(|&k| k as f64).collect::<Vec<_>>(),
        edge_errors = r.edge_errors
    ))
}

// --------------------------------------------------------- joint refinement

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_coreg_refine_joint(points: List, poses: List, edges_i: Robj, edges_j: Robj, stems: List, reference: f64, voxel_sizes: &[f64], max_distances: &[f64], rounds: f64, iterations: f64, points_per_scan: f64, correspondences_per_pair: f64, min_planarity: f64, normal_neighbours: f64, max_normal_angle_deg: f64, stem_weight: f64, stem_radius: f64, stem_scale: f64, min_voxel_points: f64, robust_scale: f64, prior_translation: f64, prior_rotation_deg: f64, max_step_translation: f64, max_step_rotation_deg: f64, seed: f64, log: Robj) -> Result<List> {
    let pts = clouds(&points)?;
    let st = clouds(&stems)?;
    let start = mats::<4>(&poses, "a pose")?;
    let (ei, ej) = (indices(&edges_i, "edges")?, indices(&edges_j, "edges")?);
    if ei.len() != ej.len() {
        return fail("edges need two columns");
    }
    let edges: Vec<(usize, usize)> = ei.into_iter().zip(ej).collect();
    let u = |v: f64| v.max(0.0) as usize;
    let p = refine::RefineParams {
        voxel_sizes: voxel_sizes.to_vec(),
        max_distances: max_distances.to_vec(),
        rounds: u(rounds),
        iterations: u(iterations),
        points_per_scan: u(points_per_scan),
        correspondences_per_pair: u(correspondences_per_pair),
        min_planarity,
        normal_neighbours: u(normal_neighbours),
        max_normal_angle_deg,
        stem_weight,
        stem_radius,
        stem_scale,
        min_voxel_points: u(min_voxel_points),
        robust_scale,
        prior_translation,
        prior_rotation_deg,
        max_step_translation,
        max_step_rotation_deg,
        seed: seed.max(0.0) as u64,
    };
    let callback = if log.is_null() { None } else { Some(log.as_function().ok_or_else(|| Error::Other("log must be a function".into()))?) };
    let mut failed: Option<Error> = None;
    let mut say = |msg: &str| {
        if let (Some(f), None) = (&callback, &failed) {
            if let Err(e) = f.call(pairlist!(msg)) {
                failed = Some(e);
            }
        }
    };
    let r = refine::refine_joint(&pts, &start, &edges, &st, reference as usize, &p, &mut say).map_err(err)?;
    if let Some(e) = failed {
        return Err(e);
    }
    Ok(list!(
        poses = mats_to_r(&r.poses),
        shifts = r.shifts,
        rotations = r.rotations,
        residual_before = r.residual_before,
        residual_after = r.residual_after,
        correspondences = r.correspondences as f64
    ))
}

// ------------------------------------------------------------------ stem maps

fn stems_from_r(stems: &List) -> Result<Vec<sm::StemRecord>> {
    let m: HashMap<&str, Robj> = stems.clone().try_into()?;
    let col = |k: &str| -> Result<Vec<f64>> { doubles(m.get(k).ok_or_else(|| Error::Other(format!("stems have no `{k}`")))?, k) };
    let (x, y, z, dbh) = (col("x")?, col("y")?, col("z")?, col("dbh")?);
    let (ax, ay, az) = (col("axis_x")?, col("axis_y")?, col("axis_z")?);
    let (rh, ns, np) = (col("reference_height")?, col("n_slices")?, col("n_points")?);
    let (rmse, cov, lean) = (col("rmse")?, col("coverage")?, col("lean_deg")?);
    let n = x.len();
    if [&y, &z, &dbh, &ax, &ay, &az, &rh, &ns, &np, &rmse, &cov, &lean].iter().any(|c| c.len() != n) {
        return fail("stem columns differ in length");
    }
    Ok((0..n)
        .map(|k| sm::StemRecord { x: x[k], y: y[k], z: z[k], dbh: dbh[k], axis: [ax[k], ay[k], az[k]], reference_height: rh[k], n_slices: ns[k] as i64, n_points: np[k] as i64, rmse: rmse[k], coverage: cov[k], lean_deg: lean[k] })
        .collect())
}

fn stems_to_r(s: &[sm::StemRecord]) -> List {
    let c = |f: &dyn Fn(&sm::StemRecord) -> f64| s.iter().map(f).collect::<Vec<f64>>();
    list!(
        x = c(&|r| r.x),
        y = c(&|r| r.y),
        z = c(&|r| r.z),
        dbh = c(&|r| r.dbh),
        axis_x = c(&|r| r.axis[0]),
        axis_y = c(&|r| r.axis[1]),
        axis_z = c(&|r| r.axis[2]),
        reference_height = c(&|r| r.reference_height),
        n_slices = c(&|r| r.n_slices as f64),
        n_points = c(&|r| r.n_points as f64),
        rmse = c(&|r| r.rmse),
        coverage = c(&|r| r.coverage),
        lean_deg = c(&|r| r.lean_deg)
    )
}

/// @noRd
#[extendr]
fn core_coreg_stem_quality(rmse: &[f64], coverage: &[f64], n_slices: &[f64]) -> Result<Vec<f64>> {
    if rmse.len() != coverage.len() || rmse.len() != n_slices.len() {
        return fail("rmse, coverage and n_slices differ in length");
    }
    Ok((0..rmse.len()).map(|k| sm::stem_quality(rmse[k], coverage[k], n_slices[k])).collect())
}

/// @noRd
#[extendr]
fn core_coreg_read_stem_map(path: &str) -> Result<List> {
    let (name, stems) = sm::read_stem_map(path).map_err(err)?;
    Ok(list!(name = name, stems = stems_to_r(&stems)))
}

/// @noRd
#[extendr]
fn core_coreg_write_stem_map(path: &str, name: &str, stems: List) -> Result<()> {
    sm::write_stem_map(path, name, &stems_from_r(&stems)?).map_err(err)
}

/// @noRd
#[extendr]
fn core_coreg_detect_stems(points: Robj, heights: &[f64], config: List) -> Result<List> {
    let pts = xyz_from_r(&points)?;
    if heights.len() != pts.len() {
        return fail("heights must have one value per point");
    }
    let m = params(&config)?;
    let mut p = StemParams::tlsalign();
    p.slice_min = num(&m, "slice_min_height")?;
    p.slice_max = num(&m, "slice_max_height")?;
    p.slice_thickness = num(&m, "slice_thickness")?;
    p.slice_step = num(&m, "slice_step")?;
    p.reference_height = num(&m, "reference_height")?;
    p.min_radius = num(&m, "min_radius")?;
    p.max_radius = num(&m, "max_radius")?;
    p.cluster_cell = num(&m, "cluster_cell")?;
    p.min_cluster_points = count(&m, "min_cluster_points")?;
    p.max_cluster_extent = num(&m, "max_cluster_extent")?;
    p.ransac_iterations = count(&m, "ransac_iterations")?;
    p.ransac_tolerance = num(&m, "ransac_tolerance")?;
    p.max_circles_per_cluster = count(&m, "max_circles_per_cluster")?;
    p.min_circle_inliers = count(&m, "min_circle_inliers")?;
    p.min_coverage = num(&m, "min_coverage")?;
    p.max_circle_rmse = num(&m, "max_circle_rmse")?;
    p.link_radius = num(&m, "link_radius")?;
    p.link_radius_ratio = num(&m, "link_radius_ratio")?;
    p.min_slices = count(&m, "min_slices")?;
    p.max_lean_deg = num(&m, "max_lean_deg")?;
    p.seed = count(&m, "seed")? as u64;
    let found = detect_stems_full(&pts, heights, &p);
    let recs: Vec<sm::StemRecord> = found
        .iter()
        .map(|s| sm::StemRecord { x: s.tree.x, y: s.tree.y, z: p.reference_height, dbh: s.tree.dbh, axis: s.axis, reference_height: p.reference_height, n_slices: s.tree.n_slices as i64, n_points: s.tree.n_points as i64, rmse: s.tree.rmse, coverage: s.coverage, lean_deg: s.tree.lean_deg })
        .collect();
    Ok(stems_to_r(&recs))
}

// ------------------------------------------------------------------- matching

fn stem_match_to_r(r: &cm::StemMatch) -> List {
    let c = &r.correspondences;
    list!(
        transform = mat_to_r(&r.transform.0),
        n_inliers = r.n_inliers as f64,
        inlier_rmse = r.inlier_rmse,
        score = r.score,
        correspondences = RMatrix::new_matrix(c.len(), 2, |a, b| if b == 0 { c[a].0 as f64 } else { c[a].1 as f64 }),
        n_source = r.n_source as f64,
        n_target = r.n_target as f64,
        success = r.success,
        ambiguity = r.ambiguity,
        rival = r.rival.as_ref().map_or(Robj::from(()), |rv| stem_match_to_r(rv).into())
    )
}

/// @noRd
#[extendr]
fn core_coreg_match_stem_maps(source: Robj, source_diameters: &[f64], source_qualities: &[f64], target: Robj, target_diameters: &[f64], target_qualities: &[f64], config: List) -> Result<List> {
    let src = cm::StemMap { positions: xyz_from_r(&source)?, diameters: source_diameters.to_vec(), qualities: source_qualities.to_vec() };
    let dst = cm::StemMap { positions: xyz_from_r(&target)?, diameters: target_diameters.to_vec(), qualities: target_qualities.to_vec() };
    if src.diameters.len() != src.len() || src.qualities.len() != src.len() || dst.diameters.len() != dst.len() || dst.qualities.len() != dst.len() {
        return fail("diameters and qualities must have one value per stem");
    }
    let m = params(&config)?;
    let p = cm::MatchParams {
        min_pair_distance: num(&m, "min_pair_distance")?,
        max_pair_distance: num(&m, "max_pair_distance")?,
        pair_distance_tolerance: num(&m, "pair_distance_tolerance")?,
        inlier_tolerance: num(&m, "inlier_tolerance")?,
        diameter_rel_tolerance: num(&m, "diameter_rel_tolerance")?,
        diameter_abs_tolerance: num(&m, "diameter_abs_tolerance")?,
        use_diameters: flag(&m, "use_diameters")?,
        max_stems: count(&m, "max_stems")?,
        max_hypotheses: count(&m, "max_hypotheses")?,
        min_inliers: count(&m, "min_inliers")?,
        early_exit_inliers: count(&m, "early_exit_inliers")?,
        distinct_translation: num(&m, "distinct_translation")?,
        distinct_yaw_deg: num(&m, "distinct_yaw_deg")?,
        refine_iterations: count(&m, "refine_iterations")?,
    };
    Ok(stem_match_to_r(&cm::match_stem_maps(&src, &dst, &p)))
}

// ------------------------------------------------------------------------ ICP

fn icp_config(config: &List) -> Result<icp::IcpConfig> {
    let m = params(config)?;
    let md = get(&m, "max_distances")?;
    Ok(icp::IcpConfig {
        voxel_sizes: doubles(get(&m, "voxel_sizes")?, "voxel_sizes")?,
        max_distances: if md.is_null() { None } else { Some(doubles(md, "max_distances")?) },
        max_iterations: count(&m, "max_iterations")?,
        method: text(&m, "method")?,
        robust: text(&m, "robust")?,
        robust_scale: num(&m, "robust_scale")?,
        trim_fraction: num(&m, "trim_fraction")?,
        trim_ramp: count(&m, "trim_ramp")?,
        min_planarity: num(&m, "min_planarity")?,
        normal_neighbours: count(&m, "normal_neighbours")?,
        translation_tolerance: num(&m, "translation_tolerance")?,
        rotation_tolerance: num(&m, "rotation_tolerance")?,
        fitness_threshold: num(&m, "fitness_threshold")?,
        damping: num(&m, "damping")?,
        max_points: count(&m, "max_points")?,
        plateau_tolerance: num(&m, "plateau_tolerance")?,
        plateau_patience: count(&m, "plateau_patience")?,
        seed: count(&m, "seed")? as u64,
    })
}

fn plane_information_to_r(info: Option<&icp::PlaneInformation>) -> Robj {
    match info {
        Some(i) => list!(hessian = mat_to_r(&i.hessian), sigma = i.sigma, n = i.n as f64).into(),
        None => Robj::from(()),
    }
}

fn prepared_target(prepared: &Robj) -> Result<Option<ExternalPtr<icp::IcpTarget>>> {
    if prepared.is_null() {
        return Ok(None);
    }
    let p: ExternalPtr<icp::IcpTarget> = prepared.try_into().map_err(|_| Error::Other("not a prepared ICP target".into()))?;
    Ok(Some(p))
}

/// @noRd
#[extendr]
fn core_coreg_icp_target(points: Robj, config: List) -> Result<Robj> {
    let cfg = icp_config(&config)?;
    let t = icp::IcpTarget::new(&xyz_from_r(&points)?, &cfg);
    Ok(ExternalPtr::new(t).into())
}

/// @noRd
#[extendr]
fn core_coreg_icp_target_len(prepared: Robj) -> Result<f64> {
    Ok(prepared_target(&prepared)?.map_or(0, |p| p.len()) as f64)
}

/// @noRd
#[extendr]
fn core_coreg_icp(source: Robj, target: Robj, prepared: Robj, initial: Robj, config: List) -> Result<List> {
    let s = xyz_from_r(&source)?;
    let init = if initial.is_null() { None } else { Some(mat4(&initial)?) };
    let cfg = icp_config(&config)?;
    let r = match prepared_target(&prepared)? {
        Some(p) => icp::icp_prepared(&s, &p, init, &cfg),
        None => icp::icp(&s, &xyz_from_r(&target)?, init, &cfg),
    }
    .map_err(err)?;
    Ok(list!(
        transform = mat_to_r(&r.transform),
        fitness = r.fitness,
        inlier_rmse = r.inlier_rmse,
        n_correspondences = r.n_correspondences as f64,
        iterations = r.iterations as f64,
        converged = r.converged,
        history = r.history,
        information = plane_information_to_r(r.information.as_ref())
    ))
}

/// @noRd
#[extendr]
fn core_coreg_plane_information(source: Robj, target: Robj, prepared: Robj, transform: Robj, config: List) -> Result<Robj> {
    let s = xyz_from_r(&source)?;
    let t = mat4(&transform)?;
    let cfg = icp_config(&config)?;
    let info = match prepared_target(&prepared)? {
        Some(p) => {
            if !p.matches(&cfg) {
                return fail("the prepared ICP target was built with other pyramid settings");
            }
            icp::plane_information(&s, &p, &t, &cfg)
        }
        None => icp::plane_information(&s, &icp::IcpTarget::new(&xyz_from_r(&target)?, &cfg), &t, &cfg),
    };
    Ok(plane_information_to_r(info.as_ref()))
}

/// @noRd
#[extendr]
fn core_coreg_evaluate(source: Robj, target: Robj, transform: Robj, threshold: f64, max_points: f64, voxel: Robj, seed: f64) -> Result<List> {
    let voxel = opt_f64(&voxel, "voxel")?;
    if matches!(voxel, Some(v) if v < 0.0) {
        return fail("voxel size must be positive");
    }
    let (f, r, n) = icp::evaluate_registration(&xyz_from_r(&source)?, &xyz_from_r(&target)?, &mat4(&transform)?, threshold, max_points.max(0.0) as usize, voxel, seed.max(0.0) as u64);
    Ok(list!(fitness = f, inlier_rmse = r, n_inliers = n as f64))
}

// ------------------------------------------------------------------- geometry

/// @noRd
#[extendr]
fn core_coreg_kd_tree(points: Robj) -> Result<List> {
    let (p, dim) = padded(&points, "points")?;
    Ok(list!(tree = ExternalPtr::new(geo::CoregTree::new(&p)), dim = dim as f64, n = p.len() as f64))
}

/// @noRd
#[extendr]
fn core_coreg_kd_query(tree: Robj, dim: f64, queries: Robj, distance_upper_bound: f64) -> Result<List> {
    let t: ExternalPtr<geo::CoregTree> = tree.try_into().map_err(|_| Error::Other("not a k-d tree".into()))?;
    let (q, d) = padded(&queries, "queries")?;
    if d != dim as usize {
        return fail(format!("queries have {d} columns but the tree has {dim}"));
    }
    let (dist, idx) = t.query(&q, distance_upper_bound);
    Ok(list!(distance = dist, index = idx.into_iter().map(|k| k as f64).collect::<Vec<_>>()))
}

/// @noRd
#[extendr]
fn core_coreg_estimate_normals(points: Robj, k: f64, radius: Robj) -> Result<List> {
    let (n, p) = geo::estimate_normals(&xyz_from_r(&points)?, k.max(0.0) as usize, opt_f64(&radius, "radius")?);
    Ok(list!(normals = xyz_to_r(&n), planarity = p))
}

/// @noRd
#[extendr]
fn core_coreg_planar_filter(points: Robj, min_planarity: f64, voxel: Robj, k: f64, radius: Robj) -> Result<Robj> {
    let voxel = opt_f64(&voxel, "voxel")?;
    if matches!(voxel, Some(v) if v < 0.0) {
        return fail("voxel size must be positive");
    }
    Ok(xyz_to_r(&geo::planar_filter(&xyz_from_r(&points)?, min_planarity, voxel, k.max(0.0) as usize, opt_f64(&radius, "radius")?)))
}

/// @noRd
#[extendr]
fn core_coreg_voxel_centroids(points: Robj, voxel: f64, centroid: bool) -> Result<List> {
    if voxel.is_nan() || voxel <= 0.0 {
        return fail("voxel size must be positive");
    }
    let (p, c) = geo::voxel_centroids(&xyz_from_r(&points)?, voxel, centroid);
    Ok(list!(points = xyz_to_r(&p), counts = c.into_iter().map(|v| v as f64).collect::<Vec<_>>()))
}

// --------------------------------------------------------------------- ground

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_coreg_fit_ground(points: Robj, cell_size: f64, percentile: f64, max_slope: f64, smooth_cells: f64, opening_cells: f64, min_points_per_cell: f64, pit_depth: f64, pit_window: f64, max_points: f64, seed: f64) -> Result<List> {
    let u = |v: f64| v.max(0.0) as usize;
    let p = gr::GroundParams { cell_size, percentile, max_slope, smooth_cells: u(smooth_cells), opening_cells: u(opening_cells), min_points_per_cell: u(min_points_per_cell), pit_depth, pit_window: u(pit_window), max_points: (max_points > 0.0).then_some(u(max_points)), seed: seed.max(0.0) as u64 };
    let g = gr::fit_ground(&xyz_from_r(&points)?, &p).map_err(err)?;
    let (ny, nx) = (g.ny, g.nx);
    Ok(list!(
        elevation = RMatrix::new_matrix(ny, nx, |r, c| g.elevation[r * nx + c]),
        origin = vec![g.origin[0], g.origin[1]],
        observed = RMatrix::new_matrix(ny, nx, |r, c| Rbool::from(g.observed[r * nx + c]))
    ))
}

fn elevation_from_r(elevation: &Robj) -> Result<(Vec<f64>, usize, usize)> {
    let m: RMatrix<f64> = elevation.try_into().map_err(|_| Error::Other("elevation must be a numeric matrix".into()))?;
    let (ny, nx) = (m.nrows(), m.ncols());
    if ny == 0 || nx == 0 {
        return fail("elevation grid is empty");
    }
    Ok((grid_from_r(m.data(), ny, nx), ny, nx))
}

/// @noRd
#[extendr]
fn core_coreg_ground_height(elevation: Robj, x0: f64, y0: f64, cell_size: f64, xy: Robj) -> Result<Vec<f64>> {
    let (e, ny, nx) = elevation_from_r(&elevation)?;
    Ok(gr::height_at_many(&e, nx, ny, [x0, y0], cell_size, &xy_rows(&xy)?))
}

/// @noRd
#[extendr]
fn core_coreg_ground_support(observed: Robj, x0: f64, y0: f64, cell_size: f64, xy: Robj) -> Result<Robj> {
    let m: RMatrix<Rbool> = observed.try_into().map_err(|_| Error::Other("observed must be a logical matrix".into()))?;
    let (ny, nx) = (m.nrows(), m.ncols());
    if ny == 0 || nx == 0 {
        return fail("observed grid is empty");
    }
    let o: Vec<bool> = grid_from_r(m.data(), ny, nx).into_iter().map(|b| b.is_true()).collect();
    let s = gr::support_many(&o, nx, ny, [x0, y0], cell_size, &xy_rows(&xy)?);
    Ok(s.into_iter().map(Rbool::from).collect::<Logicals>().into())
}

/// @noRd
#[extendr]
fn core_coreg_ground_slope_deg(elevation: Robj, cell_size: f64) -> Result<f64> {
    let (e, ny, nx) = elevation_from_r(&elevation)?;
    gr::slope_deg(&e, ny, nx, cell_size).map_err(err)
}

extendr_module! {
    mod coreg;
    fn core_coreg_skew;
    fn core_coreg_so3_exp;
    fn core_coreg_so3_log;
    fn core_coreg_rotation_angle;
    fn core_coreg_se3_exp;
    fn core_coreg_se3_log;
    fn core_coreg_invert;
    fn core_coreg_transform_points;
    fn core_coreg_transform_vectors;
    fn core_coreg_yaw_transform;
    fn core_coreg_kabsch;
    fn core_coreg_kabsch_2d_yaw;
    fn core_coreg_transform_difference;
    fn core_coreg_read_tiepoint_list;
    fn core_coreg_read_reflector_list;
    fn core_coreg_detect_reflectors;
    fn core_coreg_match_reflectors;
    fn core_coreg_default_information;
    fn core_coreg_adjoint;
    fn core_coreg_plane_edge_information;
    fn core_coreg_posegraph_residuals;
    fn core_coreg_posegraph_total_error;
    fn core_coreg_posegraph_components;
    fn core_coreg_posegraph_initialise;
    fn core_coreg_posegraph_optimise;
    fn core_coreg_refine_joint;
    fn core_coreg_stem_quality;
    fn core_coreg_read_stem_map;
    fn core_coreg_write_stem_map;
    fn core_coreg_detect_stems;
    fn core_coreg_match_stem_maps;
    fn core_coreg_icp_target;
    fn core_coreg_icp_target_len;
    fn core_coreg_icp;
    fn core_coreg_plane_information;
    fn core_coreg_evaluate;
    fn core_coreg_kd_tree;
    fn core_coreg_kd_query;
    fn core_coreg_estimate_normals;
    fn core_coreg_planar_filter;
    fn core_coreg_voxel_centroids;
    fn core_coreg_fit_ground;
    fn core_coreg_ground_height;
    fn core_coreg_ground_support;
    fn core_coreg_ground_slope_deg;
}
