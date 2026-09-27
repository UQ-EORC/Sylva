// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for the coregistration pipeline (sylva_rs::coreg_pipeline and
//! sylva_rs::coreg_survey). Scans, pairs and surveys cross as plain lists
//! with 0-based indices; `R/coreg_pipeline.R` builds its classed objects
//! from them and presents indices 1-based. Progress messages reach the R
//! callback on R's thread through [`sylva_rs::relay::relay`]: the pipeline
//! runs on a worker thread, and R is only ever called from this one.

use std::collections::HashMap;
use std::path::PathBuf;

use extendr_api::prelude::*;
use nalgebra::{Matrix4, Matrix6};
use sylva_rs::coreg::StemMatch;
use sylva_rs::coreg_ground::GroundModel;
use sylva_rs::coreg_icp::{IcpResult, PlaneInformation};
use sylva_rs::coreg_pipeline as cp;
use sylva_rs::coreg_posegraph as pg;
use sylva_rs::coreg_reflectors::{Reflector, ReflectorMatch};
use sylva_rs::coreg_survey as cs;
use sylva_rs::relay::{quiet, relay, Log};
use sylva_rs::{Point, Transform};

use crate::convert::{doubles, err, fail, xyz_from_r, xyz_to_r, Result};
use crate::coreg::{count, flag, get, grid_from_r, icp_config, mat4, mat_to_r, match_params, num, opt_doubles, params, plane_information_to_r, prepared_target, reflectors_to_r, stem_match_to_r, stem_params, stems_from_r, stems_to_r, text};

type Mat4 = Matrix4<f64>;

// ---------------------------------------------------------------- plumbing

fn field<'a>(m: &'a HashMap<&str, Robj>, k: &str, what: &str) -> Result<&'a Robj> {
    m.get(k).ok_or_else(|| Error::Other(format!("{what} has no `{k}`")))
}

fn map<'a>(v: &'a Robj, what: &str) -> Result<HashMap<&'a str, Robj>> {
    let l: List = v.try_into().map_err(|_| Error::Other(format!("{what} must be a list")))?;
    l.try_into()
}

fn scalar(v: &Robj, what: &str) -> Result<f64> {
    doubles(v, what)?.first().copied().ok_or_else(|| Error::Other(format!("`{what}` is empty")))
}

fn string(v: &Robj, what: &str) -> Result<String> {
    v.as_str().map(str::to_string).ok_or_else(|| Error::Other(format!("`{what}` must be a string")))
}

fn opt_string(v: &Robj, what: &str) -> Result<Option<String>> {
    if v.is_null() {
        Ok(None)
    } else {
        string(v, what).map(Some)
    }
}

fn boolean(v: &Robj, what: &str) -> Result<bool> {
    v.as_bool().ok_or_else(|| Error::Other(format!("`{what}` must be TRUE or FALSE")))
}

fn index(v: f64) -> i64 {
    v as i64
}

fn mats(list: &Robj, what: &str) -> Result<Vec<Mat4>> {
    let l: List = list.try_into().map_err(|_| Error::Other(format!("{what} must be a list of 4 x 4 matrices")))?;
    l.values().map(|m| mat4(&m)).collect()
}

fn mats_to_r(m: &[Mat4]) -> List {
    List::from_values(m.iter().map(mat_to_r))
}

fn opt_i64(v: i64) -> f64 {
    v as f64
}

/// Run `work` with a [`Log`] that calls the R function `log` (or nothing
/// for `NULL`) on R's thread; the first error it raises is returned once the
/// work is done.
fn with_log<T: Send>(log: &Robj, work: impl FnOnce(Log) -> T + Send) -> Result<T> {
    if log.is_null() {
        return Ok(work(&quiet));
    }
    let f = log.as_function().ok_or_else(|| Error::Other("progress must be a function".into()))?;
    let mut failed: Option<Error> = None;
    let out = relay(work, |msg| {
        if failed.is_none() {
            if let Err(e) = f.call(pairlist!(msg)) {
                failed = Some(e);
            }
        }
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

// ------------------------------------------------------------------ config

fn reading(v: &Robj) -> Result<cp::ReadingOptions> {
    let mut o = cp::ReadingOptions::default();
    if v.is_null() {
        return Ok(o);
    }
    let l: List = v.try_into().map_err(|_| Error::Other("riegl_options must be a named list".into()))?;
    o.any = !l.is_empty();
    let m: HashMap<&str, Robj> = l.try_into()?;
    if let Some(p) = m.get("library").filter(|p| !p.is_null()) {
        o.library = Some(PathBuf::from(string(p, "library")?));
    }
    if let Some(v) = m.get("drop_pseudo_echoes") {
        o.drop_pseudo_echoes = boolean(v, "drop_pseudo_echoes")?;
    }
    if let Some(v) = m.get("echoes").filter(|v| !v.is_null()) {
        o.echoes = string(v, "echoes")?;
    }
    if let Some(v) = m.get("stride").filter(|v| !v.is_null()) {
        o.stride = scalar(v, "stride")?.max(0.0) as usize;
    }
    if let Some(v) = m.get("shot_stride").filter(|v| !v.is_null()) {
        o.shot_stride = scalar(v, "shot_stride")?.max(0.0) as usize;
    }
    for (g, name) in cp::GATES.iter().enumerate() {
        let bound = |side: &str| -> Result<Option<f64>> {
            match m.get(format!("{side}_{name}").as_str()) {
                Some(v) if !v.is_null() => Ok(Some(scalar(v, name)?)),
                _ => Ok(None),
            }
        };
        o.bounds[g] = (bound("min")?, bound("max")?);
    }
    Ok(o)
}

fn opt_num(m: &HashMap<&str, Robj>, k: &str) -> Result<Option<f64>> {
    let v = get(m, k)?;
    if v.is_null() {
        Ok(None)
    } else {
        num(m, k).map(Some)
    }
}

fn int(m: &HashMap<&str, Robj>, k: &str) -> Result<i64> {
    Ok(num(m, k)? as i64)
}

/// A `CoregConfig` from a `coreg_config()`.
fn config(v: &List) -> Result<cp::CoregConfig> {
    let m = params(v)?;
    let sub = |k: &str| -> Result<List> { get(&m, k)?.try_into().map_err(|_| Error::Other(format!("`{k}` must be a configuration list"))) };
    Ok(cp::CoregConfig {
        ground_cell_size: num(&m, "ground_cell_size")?,
        ground_min_coverage: opt_num(&m, "ground_min_coverage")?,
        stems: stem_params(&sub("stems")?)?,
        matching: match_params(&sub("matching")?)?,
        icp: icp_config(&sub("icp")?)?,
        icp_voxel: num(&m, "icp_voxel")?,
        icp_min_planarity: num(&m, "icp_min_planarity")?,
        icp_max_height: num(&m, "icp_max_height")?,
        use_reflectors: flag(&m, "use_reflectors")?,
        min_reflector_matches: int(&m, "min_reflector_matches")?,
        reflector_tolerance: num(&m, "reflector_tolerance")?,
        trusted_reflector_matches: int(&m, "trusted_reflector_matches")?,
        trusted_reflector_rmse: num(&m, "trusted_reflector_rmse")?,
        min_match_inliers: int(&m, "min_match_inliers")?,
        max_match_ambiguity: num(&m, "max_match_ambiguity")?,
        ambiguity_margin: num(&m, "ambiguity_margin")?,
        max_match_rmse: num(&m, "max_match_rmse")?,
        max_coarse_stem_rmse: num(&m, "max_coarse_stem_rmse")?,
        height_from_ground: flag(&m, "height_from_ground")?,
        ground_radius: num(&m, "ground_radius")?,
        min_ground_cells: int(&m, "min_ground_cells")?,
        max_ground_disagreement: opt_num(&m, "max_ground_disagreement")?,
        max_pair_distance: num(&m, "max_pair_distance")?,
        screen_pairs: flag(&m, "screen_pairs")?,
        max_pairs_per_scan: opt_num(&m, "max_pairs_per_scan")?.map_or(0, |v| v as i64),
        min_icp_fitness: num(&m, "min_icp_fitness")?,
        max_icp_rmse: num(&m, "max_icp_rmse")?,
        min_icp_fitness_above_ground: num(&m, "min_icp_fitness_above_ground")?,
        fitness_min_height: num(&m, "fitness_min_height")?,
        stem_agreement_tolerance: num(&m, "stem_agreement_tolerance")?,
        max_coarse_to_fine_shift: num(&m, "max_coarse_to_fine_shift")?,
        recover_unregistered: flag(&m, "recover_unregistered")?,
        recovery_rounds: int(&m, "recovery_rounds")?,
        recovery_neighbours: int(&m, "recovery_neighbours")?,
        refine_multiview: flag(&m, "refine_multiview")?,
        refinement_rounds: int(&m, "refinement_rounds")?,
        refinement_voxel_sizes: doubles(get(&m, "refinement_voxel_sizes")?, "refinement_voxel_sizes")?,
        refinement_max_distances: doubles(get(&m, "refinement_max_distances")?, "refinement_max_distances")?,
        refinement_points_per_scan: count(&m, "refinement_points_per_scan")?,
        refinement_stem_weight: num(&m, "refinement_stem_weight")?,
        refinement_stem_radius: num(&m, "refinement_stem_radius")?,
        refinement_min_voxel_points: count(&m, "refinement_min_voxel_points")?,
        refinement_max_shift: num(&m, "refinement_max_shift")?,
        optimise_globally: flag(&m, "optimise_globally")?,
        reference_scan: int(&m, "reference_scan")?,
        reject_outlier_edges: flag(&m, "reject_outlier_edges")?,
        information_patch_points: num(&m, "information_patch_points")?,
        information_min_sigma: num(&m, "information_min_sigma")?,
        max_prior_shift: num(&m, "max_prior_shift")?,
        max_prior_rotation: opt_num(&m, "max_prior_rotation")?,
        workers: int(&m, "workers")?,
        memory_per_worker_gb: opt_num(&m, "memory_per_worker_gb")?.unwrap_or(0.0),
        riscan_filter: text(&m, "riscan_filter")?,
        reading: reading(get(&m, "riegl_options")?)?,
        min_points_per_scan: int(&m, "min_points_per_scan")?,
        max_points_per_scan: opt_num(&m, "max_points_per_scan")?.map_or(0, |v| v.max(0.0) as usize),
    })
}

// ------------------------------------------------------------------- scans

fn reflectors_from_r(v: &Robj) -> Result<Vec<Reflector>> {
    if v.is_null() {
        return Ok(Vec::new());
    }
    let m = map(v, "reflectors")?;
    let col = |k: &str| doubles(field(&m, k, "reflectors")?, k);
    let (x, y, z) = (col("x")?, col("y")?, col("z")?);
    let (reflectance, diameter, n_points) = (col("reflectance")?, col("diameter")?, col("n_points")?);
    let names: Vec<String> = field(&m, "name", "reflectors")?.as_str_iter().map(|it| it.map(str::to_string).collect()).unwrap_or_default();
    if [&y, &z, &reflectance, &diameter, &n_points].iter().any(|c| c.len() != x.len()) || names.len() != x.len() {
        return fail("reflector columns differ in length");
    }
    Ok((0..x.len()).map(|k| Reflector { x: x[k], y: y[k], z: z[k], reflectance: reflectance[k], diameter: diameter[k], n_points: n_points[k] as i64, name: names[k].clone() }).collect())
}

fn ground_from_r(v: &Robj) -> Result<Option<GroundModel>> {
    if v.is_null() {
        return Ok(None);
    }
    let m = map(v, "a ground model")?;
    let e: RMatrix<f64> = field(&m, "elevation", "a ground model")?.try_into().map_err(|_| Error::Other("elevation must be a numeric matrix".into()))?;
    let (ny, nx) = (e.nrows(), e.ncols());
    let o: Vec<bool> = field(&m, "observed", "a ground model")?.as_logical_slice().ok_or_else(|| Error::Other("observed must be a logical matrix".into()))?.iter().map(|b| b.is_true()).collect();
    if o.len() != ny * nx {
        return fail("elevation and observed differ in size");
    }
    let origin = doubles(field(&m, "origin", "a ground model")?, "origin")?;
    if origin.len() < 2 {
        return fail("a ground origin has two coordinates");
    }
    Ok(Some(GroundModel { nx, ny, elevation: grid_from_r(e.data(), ny, nx), origin: [origin[0], origin[1]], cell_size: scalar(field(&m, "cell_size", "a ground model")?, "cell_size")?, observed: grid_from_r(&o, ny, nx) }))
}

fn ground_to_r(g: &GroundModel) -> List {
    list!(
        elevation = RMatrix::new_matrix(g.ny, g.nx, |r, c| g.elevation[r * g.nx + c]),
        origin = vec![g.origin[0], g.origin[1]],
        cell_size = g.cell_size,
        observed = RMatrix::new_matrix(g.ny, g.nx, |r, c| Rbool::from(g.observed[r * g.nx + c]))
    )
}

/// A scan from the list `scan_core()` builds.
fn scan_from_r(v: &Robj) -> Result<cp::ScanFeatures> {
    let m = map(v, "a scan")?;
    let f = |k: &str| field(&m, k, "a scan");
    let origin = doubles(f("origin")?, "origin")?;
    if origin.len() != 3 {
        return fail("a scanner origin has three coordinates");
    }
    let stems: List = f("stems")?.try_into().map_err(|_| Error::Other("stems must be a list of columns".into()))?;
    Ok(cp::ScanFeatures {
        name: string(f("name")?, "name")?,
        n_points: scalar(f("n_points")?, "n_points")? as usize,
        ground: ground_from_r(f("ground")?)?,
        stems: stems_from_r(&stems)?,
        stem_map_name: string(f("stem_map_name")?, "stem_map_name")?,
        icp_points: xyz_from_r(f("icp_points")?)?,
        reflectors: reflectors_from_r(f("reflectors")?)?,
        icp_heights: doubles(f("icp_heights")?, "icp_heights")?.into_iter().map(|h| h as f32).collect(),
        levelling: mat4(f("levelling")?)?,
        origin: [origin[0], origin[1], origin[2]],
        source: opt_string(f("source")?, "source")?,
        seconds: scalar(f("seconds")?, "seconds")?,
        error: string(f("error")?, "error")?,
    })
}

fn scans_from_r(v: &Robj) -> Result<Vec<cp::ScanFeatures>> {
    let l: List = v.try_into().map_err(|_| Error::Other("scans must be a list".into()))?;
    l.values().map(|s| scan_from_r(&s)).collect()
}

fn f32_points_to_r(p: &[Point]) -> Robj {
    xyz_to_r(p)
}

fn scan_to_r(s: &cp::ScanFeatures) -> List {
    list!(
        name = s.name.as_str(),
        n_points = s.n_points as f64,
        ground = s.ground.as_ref().map_or(Robj::from(()), |g| ground_to_r(g).into()),
        stems = stems_to_r(&s.stems),
        stem_map_name = s.stem_map_name.as_str(),
        icp_points = f32_points_to_r(&s.icp_points),
        reflectors = reflectors_to_r(s.reflectors.clone()),
        icp_heights = s.icp_heights.iter().map(|&h| h as f64).collect::<Vec<f64>>(),
        levelling = mat_to_r(&s.levelling),
        origin = s.origin.to_vec(),
        source = s.source.as_deref().map_or(Robj::from(()), Robj::from),
        seconds = s.seconds,
        error = s.error.as_str()
    )
}

/// A scan input: a path (string), n x 3 points, or `list(failed = reason, source = path)`.
fn input_from_r(v: &Robj) -> Result<cp::ScanInput> {
    if let Some(s) = v.as_str() {
        return Ok(cp::ScanInput::Path(PathBuf::from(s)));
    }
    if v.is_list() {
        let m = map(v, "a failed scan")?;
        return Ok(cp::ScanInput::Failed { reason: string(field(&m, "failed", "a failed scan")?, "failed")?, source: opt_string(field(&m, "source", "a failed scan")?, "source")?.map(PathBuf::from) });
    }
    Ok(cp::ScanInput::Points(xyz_from_r(v)?))
}

fn inputs_from_r(v: &Robj) -> Result<Vec<cp::ScanInput>> {
    let l: List = v.try_into().map_err(|_| Error::Other("clouds must be a list".into()))?;
    l.values().map(|c| input_from_r(&c)).collect()
}

// ------------------------------------------------------------------- pairs

fn pairs_from_matrix(v: &Robj, what: &str) -> Result<Vec<(usize, usize)>> {
    if v.is_null() {
        return Ok(Vec::new());
    }
    let m: RMatrix<f64> = v.try_into().map_err(|_| Error::Other(format!("{what} must be a two-column numeric matrix")))?;
    if m.nrows() > 0 && m.ncols() != 2 {
        return fail(format!("{what} must have two columns"));
    }
    let (n, d) = (m.nrows(), m.data());
    (0..n).map(|r| {
        let (a, b) = (d[r], d[n + r]);
        if a < 0.0 || b < 0.0 {
            return fail(format!("{what} must be non-negative"));
        }
        Ok((a as usize, b as usize))
    }).collect()
}

fn stem_match_from_r(v: &Robj) -> Result<Option<StemMatch>> {
    if v.is_null() {
        return Ok(None);
    }
    let m = map(v, "a match")?;
    let f = |k: &str| field(&m, k, "a match");
    Ok(Some(StemMatch {
        transform: Transform(mat4(f("transform")?)?),
        n_inliers: scalar(f("n_inliers")?, "n_inliers")? as usize,
        inlier_rmse: scalar(f("inlier_rmse")?, "inlier_rmse")?,
        score: scalar(f("score")?, "score")?,
        correspondences: pairs_from_matrix(f("correspondences")?, "correspondences")?,
        n_source: scalar(f("n_source")?, "n_source")? as usize,
        n_target: scalar(f("n_target")?, "n_target")? as usize,
        success: boolean(f("success")?, "success")?,
        ambiguity: scalar(f("ambiguity")?, "ambiguity")?,
        rival: stem_match_from_r(f("rival")?)?.map(Box::new),
    }))
}

fn reflector_match_to_r(r: &ReflectorMatch) -> List {
    let c = &r.correspondences;
    list!(
        transform = mat_to_r(&r.transform),
        n_inliers = r.n_inliers as f64,
        rmse = r.rmse,
        correspondences = RMatrix::new_matrix(c.len(), 2, |a, b| c[a][b] as f64),
        success = r.success
    )
}

fn reflector_match_from_r(v: &Robj) -> Result<Option<ReflectorMatch>> {
    if v.is_null() {
        return Ok(None);
    }
    let m = map(v, "a reflector match")?;
    let f = |k: &str| field(&m, k, "a reflector match");
    Ok(Some(ReflectorMatch {
        transform: mat4(f("transform")?)?,
        n_inliers: scalar(f("n_inliers")?, "n_inliers")? as usize,
        rmse: scalar(f("rmse")?, "rmse")?,
        correspondences: pairs_from_matrix(f("correspondences")?, "correspondences")?.into_iter().map(|(a, b)| [a, b]).collect(),
        success: boolean(f("success")?, "success")?,
    }))
}

fn icp_to_r(r: &IcpResult) -> List {
    list!(
        transform = mat_to_r(&r.transform),
        fitness = r.fitness,
        inlier_rmse = r.inlier_rmse,
        n_correspondences = r.n_correspondences as f64,
        iterations = r.iterations as f64,
        converged = r.converged,
        history = r.history.clone(),
        information = plane_information_to_r(r.information.as_ref())
    )
}

fn icp_from_r(v: &Robj) -> Result<Option<IcpResult>> {
    if v.is_null() {
        return Ok(None);
    }
    let m = map(v, "an ICP result")?;
    let f = |k: &str| field(&m, k, "an ICP result");
    let info = f("information")?;
    let information = if info.is_null() {
        None
    } else {
        let i = map(info, "plane information")?;
        let h: RMatrix<f64> = field(&i, "hessian", "plane information")?.try_into().map_err(|_| Error::Other("a hessian must be a 6 x 6 matrix".into()))?;
        if h.nrows() != 6 || h.ncols() != 6 {
            return fail("a hessian must be a 6 x 6 matrix");
        }
        Some(PlaneInformation { hessian: Matrix6::from_column_slice(h.data()), sigma: scalar(field(&i, "sigma", "plane information")?, "sigma")?, n: scalar(field(&i, "n", "plane information")?, "n")? as usize })
    };
    Ok(Some(IcpResult {
        transform: mat4(f("transform")?)?,
        fitness: scalar(f("fitness")?, "fitness")?,
        inlier_rmse: scalar(f("inlier_rmse")?, "inlier_rmse")?,
        n_correspondences: scalar(f("n_correspondences")?, "n_correspondences")? as usize,
        iterations: scalar(f("iterations")?, "iterations")? as usize,
        converged: boolean(f("converged")?, "converged")?,
        history: doubles(f("history")?, "history")?,
        information,
    }))
}

fn opt_list<T>(v: Option<&T>, f: impl Fn(&T) -> List) -> Robj {
    v.map_or(Robj::from(()), |x| f(x).into())
}

fn pair_to_r(p: &cp::PairResult) -> List {
    list!(
        i = opt_i64(p.i),
        j = opt_i64(p.j),
        name_i = p.name_i.as_str(),
        name_j = p.name_j.as_str(),
        transform = mat_to_r(&p.transform),
        coarse_transform = mat_to_r(&p.coarse_transform),
        match_result = opt_list(p.stem_match.as_ref(), stem_match_to_r),
        reflector_match = opt_list(p.reflector_match.as_ref(), reflector_match_to_r),
        icp = opt_list(p.icp.as_ref(), icp_to_r),
        success = p.success,
        reason = p.reason.as_str(),
        seconds = p.seconds,
        matched_source = xyz_to_r(&p.matched_source),
        matched_target = xyz_to_r(&p.matched_target),
        coarse_stem_rmse = p.coarse_stem_rmse,
        fine_stem_rmse = p.fine_stem_rmse,
        fitness_above = p.fitness_above,
        rival = opt_list(p.rival.as_ref(), stem_match_to_r),
        used_icp = p.used_icp,
        ground_offset = p.ground_offset,
        trusted = p.trusted
    )
}

/// A pair from the list `pair_core()` builds.
fn pair_from_r(v: &Robj) -> Result<cp::PairResult> {
    let m = map(v, "a pair")?;
    let f = |k: &str| field(&m, k, "a pair");
    Ok(cp::PairResult {
        i: index(scalar(f("i")?, "i")?),
        j: index(scalar(f("j")?, "j")?),
        name_i: string(f("name_i")?, "name_i")?,
        name_j: string(f("name_j")?, "name_j")?,
        transform: mat4(f("transform")?)?,
        coarse_transform: mat4(f("coarse_transform")?)?,
        stem_match: stem_match_from_r(f("match_result")?)?,
        reflector_match: reflector_match_from_r(f("reflector_match")?)?,
        icp: icp_from_r(f("icp")?)?,
        success: boolean(f("success")?, "success")?,
        reason: string(f("reason")?, "reason")?,
        seconds: scalar(f("seconds")?, "seconds")?,
        matched_source: xyz_from_r(f("matched_source")?)?,
        matched_target: xyz_from_r(f("matched_target")?)?,
        coarse_stem_rmse: scalar(f("coarse_stem_rmse")?, "coarse_stem_rmse")?,
        fine_stem_rmse: scalar(f("fine_stem_rmse")?, "fine_stem_rmse")?,
        fitness_above: scalar(f("fitness_above")?, "fitness_above")?,
        rival: stem_match_from_r(f("rival")?)?,
        used_icp: boolean(f("used_icp")?, "used_icp")?,
        ground_offset: scalar(f("ground_offset")?, "ground_offset")?,
        trusted: boolean(f("trusted")?, "trusted")?,
    })
}

fn pairs_from_r(v: &Robj) -> Result<Vec<cp::PairResult>> {
    let l: List = v.try_into().map_err(|_| Error::Other("pairs must be a list".into()))?;
    l.values().map(|p| pair_from_r(&p)).collect()
}

fn optimisation_to_r(o: &pg::Optimisation) -> List {
    list!(
        iterations = o.iterations as f64,
        converged = o.converged,
        initial_error = o.initial_error,
        final_error = o.final_error,
        rejected_edges = o.rejected_edges.iter().map(|&k| k as f64).collect::<Vec<f64>>(),
        edge_errors = o.edge_errors.clone()
    )
}

fn optimisation_from_r(v: &Robj) -> Result<Option<pg::Optimisation>> {
    if v.is_null() {
        return Ok(None);
    }
    let m = map(v, "an optimisation")?;
    let f = |k: &str| field(&m, k, "an optimisation");
    Ok(Some(pg::Optimisation {
        poses: Vec::new(),
        iterations: scalar(f("iterations")?, "iterations")? as usize,
        converged: boolean(f("converged")?, "converged")?,
        initial_error: scalar(f("initial_error")?, "initial_error")?,
        final_error: scalar(f("final_error")?, "final_error")?,
        rejected_edges: doubles(f("rejected_edges")?, "rejected_edges")?.into_iter().map(|k| k as usize).collect(),
        edge_errors: Vec::new(),
    }))
}

fn survey_to_r(r: &cs::SurveyResult) -> List {
    list!(
        pairs = List::from_values(r.pairs.iter().map(pair_to_r)),
        poses = mats_to_r(&r.poses),
        reference = r.reference as f64,
        optimisation = opt_list(r.optimisation.as_ref(), optimisation_to_r),
        registered = r.registered.clone(),
        seconds = r.seconds,
        edge_to_pair = r.edge_to_pair.iter().map(|&k| k as f64).collect::<Vec<f64>>()
    )
}

fn options_from_r(pairs: &Robj, positions: &Robj, fixed_nodes: &Robj, fixed_poses: &Robj, priors: &Robj) -> Result<cs::SurveyOptions> {
    let positions = if positions.is_null() {
        None
    } else {
        let m: RMatrix<f64> = positions.try_into().map_err(|_| Error::Other("approximate positions must be a numeric matrix".into()))?;
        let (n, c, d) = (m.nrows(), m.ncols(), m.data());
        if n == 0 { None } else { Some((0..n).map(|r| (0..c).map(|k| d[k * n + r]).collect()).collect()) }
    };
    let nodes: Vec<usize> = opt_doubles(fixed_nodes, "fixed")?.unwrap_or_default().into_iter().map(|k| k as usize).collect();
    let poses = if fixed_poses.is_null() { Vec::new() } else { mats(fixed_poses, "fixed poses")? };
    if nodes.len() != poses.len() {
        return fail("one pose per fixed scan");
    }
    let priors = if priors.is_null() {
        None
    } else {
        let l: List = priors.try_into().map_err(|_| Error::Other("priors must be a list".into()))?;
        Some(l.values().map(|p| if p.is_null() { Ok(None) } else { mat4(&p).map(Some) }).collect::<Result<Vec<_>>>()?)
    };
    Ok(cs::SurveyOptions { pairs: if pairs.is_null() { None } else { Some(pairs_from_matrix(pairs, "pairs")?) }, approximate_positions: positions, fixed: nodes.into_iter().zip(poses).collect(), priors })
}

fn prepare_err(e: cp::PrepareError) -> Error {
    Error::Other(e.to_string())
}

// --------------------------------------------------------------- functions

/// @noRd
#[extendr]
fn core_coreg_prepare_scan(cloud: Robj, config: List, name: &str, reflectors: Robj, levelling: Robj, origin: Robj) -> Result<List> {
    let cfg = self::config(&config)?;
    let input = input_from_r(&cloud)?;
    let refl = reflectors_from_r(&reflectors)?;
    let level = if levelling.is_null() { None } else { Some(mat4(&levelling)?) };
    let origin = match opt_doubles(&origin, "origin")? {
        Some(o) if o.len() == 3 => Some([o[0], o[1], o[2]]),
        Some(_) => return fail("origin must have three coordinates"),
        None => None,
    };
    let scan = cp::prepare_scan(&input, &cfg, name, &refl, level.as_ref(), origin).map_err(prepare_err)?;
    Ok(scan_to_r(&scan))
}

/// @noRd
#[extendr]
fn core_coreg_read_scan(path: &str, config: List) -> Result<Robj> {
    let cfg = self::config(&config)?;
    Ok(xyz_to_r(&cp::read_scan(std::path::Path::new(path), &cfg).map_err(prepare_err)?))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_coreg_register_pair(source: Robj, target: Robj, config: List, initial: Robj, stem_match: Robj, i: f64, j: f64, prepared: Robj) -> Result<List> {
    let (s, t, cfg) = (scan_from_r(&source)?, scan_from_r(&target)?, self::config(&config)?);
    let initial = if initial.is_null() { None } else { Some(mat4(&initial)?) };
    let m = stem_match_from_r(&stem_match)?;
    let pyramid = prepared_target(&prepared)?;
    let p = cp::register_pair(&s, &t, &cfg, initial.as_ref(), m.as_ref(), index(i), index(j), pyramid.as_deref()).map_err(err)?;
    Ok(pair_to_r(&p))
}

/// @noRd
#[extendr]
fn core_coreg_pair_summary(pair: Robj) -> Result<String> {
    Ok(pair_from_r(&pair)?.summary())
}

/// @noRd
#[extendr]
fn core_coreg_place_from_prior(scan: Robj, survey: Robj, poses: Robj, prior: Robj, config: List, neighbours: Robj) -> Result<List> {
    let (s, cfg, prior) = (scan_from_r(&scan)?, self::config(&config)?, mat4(&prior)?);
    let (others, poses) = (scans_from_r(&survey)?, if poses.is_null() { Vec::new() } else { mats(&poses, "poses")? });
    if poses.len() < others.len() {
        return fail("one pose per registered scan");
    }
    let refs: Vec<&cp::ScanFeatures> = others.iter().collect();
    let n = opt_doubles(&neighbours, "neighbours")?.and_then(|v| v.first().copied()).map(|v| v as i64);
    let (r, used) = cs::place_from_prior(&s, &refs, &poses, &prior, &cfg, n).map_err(err)?;
    Ok(list!(result = pair_to_r(&r), used = used.iter().map(|&k| k as f64).collect::<Vec<f64>>()))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_coreg_coregister_prepared(scans: Robj, config: List, pairs: Robj, positions: Robj, fixed_nodes: Robj, fixed_poses: Robj, priors: Robj, log: Robj, already: f64) -> Result<List> {
    let (s, cfg) = (scans_from_r(&scans)?, self::config(&config)?);
    let opts = options_from_r(&pairs, &positions, &fixed_nodes, &fixed_poses, &priors)?;
    let r = with_log(&log, |log| cs::coregister_prepared(&s, &cfg, &opts, log, already))?.map_err(err)?;
    Ok(survey_to_r(&r))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_coreg_coregister(inputs: Robj, config: List, names: Robj, reflectors: List, levelling: List, pairs: Robj, positions: Robj, fixed_nodes: Robj, fixed_poses: Robj, priors: Robj, log: Robj) -> Result<List> {
    let cfg = self::config(&config)?;
    let inputs = inputs_from_r(&inputs)?;
    if reflectors.len() != inputs.len() || levelling.len() != inputs.len() {
        return fail("one set of reflectors and one levelling per scan");
    }
    let specs: Vec<cs::ScanSpec> = inputs
        .into_iter()
        .zip(reflectors.values())
        .zip(levelling.values())
        .map(|((input, r), l)| Ok(cs::ScanSpec { input, reflectors: reflectors_from_r(&r)?, levelling: if l.is_null() { None } else { Some(mat4(&l)?) } }))
        .collect::<Result<_>>()?;
    let names: Vec<String> = if names.is_null() { Vec::new() } else { names.as_str_iter().map(|it| it.map(str::to_string).collect()).ok_or_else(|| Error::Other("names must be a character vector".into()))? };
    let opts = options_from_r(&pairs, &positions, &fixed_nodes, &fixed_poses, &priors)?;
    let (scans, r) = with_log(&log, |log| cs::coregister(&specs, &cfg, Some(&names), &opts, log))?.map_err(err)?;
    Ok(list!(scans = List::from_values(scans.iter().map(scan_to_r)), survey = survey_to_r(&r)))
}

/// @noRd
#[extendr]
fn core_coreg_merge_clouds(inputs: Robj, poses: Robj, levellings: Robj, registered: &[Rbool], only_registered: bool, voxel: Robj, config: List) -> Result<List> {
    let cfg = self::config(&config)?;
    let inputs = inputs_from_r(&inputs)?;
    let (poses, levellings) = (mats(&poses, "poses")?, mats(&levellings, "levellings")?);
    let registered: Vec<bool> = registered.iter().map(|b| b.is_true()).collect();
    let voxel = opt_doubles(&voxel, "voxel")?.and_then(|v| v.first().copied());
    let cloud = cs::merge_clouds(&inputs, &poses, &levellings, &registered, only_registered, voxel, &cfg).map_err(prepare_err)?;
    let ids: Vec<f64> = match cloud.attr("scan_id") {
        Some(sylva_rs::pointcloud::Attr::I32(v)) => v.iter().map(|&k| k as f64).collect(),
        _ => Vec::new(),
    };
    Ok(list!(xyz = xyz_to_r(&cloud.xyz), scan_id = ids))
}

fn summaries_from_r(v: &Robj) -> Result<Vec<cs::ScanSummary>> {
    let l: List = v.try_into().map_err(|_| Error::Other("scans must be a list".into()))?;
    l.values()
        .map(|s| {
            let m = map(&s, "a scan")?;
            let f = |k: &str| field(&m, k, "a scan");
            Ok(cs::ScanSummary { name: string(f("name")?, "name")?, n_points: scalar(f("n_points")?, "n_points")? as i64, n_stems: scalar(f("n_stems")?, "n_stems")? as i64, error: string(f("error")?, "error")?, source: opt_string(f("source")?, "source")?, levelling: mat4(f("levelling")?)? })
        })
        .collect()
}

fn registered_from_r(v: &[Rbool]) -> Vec<bool> {
    v.iter().map(|b| b.is_true()).collect()
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_coreg_survey_report(scans: Robj, pairs: Robj, poses: Robj, reference: f64, optimisation: Robj, registered: &[Rbool], seconds: f64, edge_to_pair: &[f64]) -> Result<String> {
    let (s, p, poses) = (summaries_from_r(&scans)?, pairs_from_r(&pairs)?, mats(&poses, "poses")?);
    let o = optimisation_from_r(&optimisation)?;
    let edges: Vec<usize> = edge_to_pair.iter().map(|&k| k as usize).collect();
    cs::report(&s, &p, &poses, index(reference), o.as_ref(), &registered_from_r(registered), seconds, &edges).map_err(err)
}

/// @noRd
#[extendr]
fn core_coreg_survey_consistency(pairs: Robj, poses: Robj, robust: bool) -> Result<List> {
    let (p, poses) = (pairs_from_r(&pairs)?, mats(&poses, "poses")?);
    let c = cs::consistency(&p, &poses, robust).map_err(err)?;
    Ok(list!(i = c.iter().map(|x| x.0 .0 as f64).collect::<Vec<f64>>(), j = c.iter().map(|x| x.0 .1 as f64).collect::<Vec<f64>>(), value = c.iter().map(|x| x.1).collect::<Vec<f64>>()))
}

/// @noRd
#[extendr]
fn core_coreg_survey_save(path: &str, scans: Robj, pairs: Robj, poses: Robj, reference: f64, registered: &[Rbool], seconds: f64) -> Result<()> {
    let (s, p, poses) = (summaries_from_r(&scans)?, pairs_from_r(&pairs)?, mats(&poses, "poses")?);
    let text = cs::survey_json(&s, &p, &poses, index(reference), &registered_from_r(registered), seconds).map_err(err)?;
    cs::save_survey(std::path::Path::new(path), &text).map_err(err)
}

/// @noRd
#[extendr]
fn core_coreg_load_transforms(path: &str) -> Result<List> {
    let loaded = cs::load_transforms(std::path::Path::new(path)).map_err(err)?;
    let names: Vec<&str> = loaded.iter().map(|(n, _)| n.as_str()).collect();
    let values: Vec<Robj> = loaded
        .iter()
        .map(|(_, rows)| {
            let (r, c) = (rows.len(), rows.first().map_or(0, Vec::len));
            if rows.iter().any(|row| row.len() != c) {
                return fail("`world_from_scan` rows differ in length");
            }
            Ok(RMatrix::new_matrix(r, c, |a, b| rows[a][b]).into())
        })
        .collect::<Result<_>>()?;
    List::from_names_and_values(names, values)
}

/// @noRd
#[extendr]
fn core_coreg_transform_for(pose: Robj, levelling: Robj) -> Result<Robj> {
    Ok(mat_to_r(&cp::matmul(&mat4(&pose)?, &mat4(&levelling)?)))
}

extendr_module! {
    mod coreg_pipeline;
    fn core_coreg_prepare_scan;
    fn core_coreg_read_scan;
    fn core_coreg_register_pair;
    fn core_coreg_pair_summary;
    fn core_coreg_place_from_prior;
    fn core_coreg_coregister_prepared;
    fn core_coreg_coregister;
    fn core_coreg_merge_clouds;
    fn core_coreg_survey_report;
    fn core_coreg_survey_consistency;
    fn core_coreg_survey_save;
    fn core_coreg_load_transforms;
    fn core_coreg_transform_for;
}
