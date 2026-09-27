// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for trees: circle fits, stem detection, segmentation, heights,
//! crowns, pruning, basal area and buttresses. Trees cross as a list of
//! equal-length columns (a data frame in `R/trees.R`).
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use extendr_api::prelude::*;
use sylva_rs::buttress_detect as bd;
use sylva_rs::stems::StemParams;
use sylva_rs::tree_prune as tp;
use sylva_rs::trees::{self, SegmentParams, Tree};

use crate::convert::{doubles, err, fail, xyz_from_r, Result};

const TREE_COLUMNS: [&str; 11] = ["tree_id", "x", "y", "dbh", "height", "n_points", "inlier_fraction", "n_slices", "rmse", "lean_deg", "quality"];

pub fn xy_from_r(xy: &Robj) -> Result<Vec<[f64; 2]>> {
    let m: RMatrix<f64> = xy.try_into().map_err(|_| Error::Other("xy must be a double matrix with 2 columns".into()))?;
    if m.ncols() != 2 {
        return fail(format!("xy must have 2 columns, got {}", m.ncols()));
    }
    let n = m.nrows();
    let d = m.data();
    Ok((0..n).map(|i| [d[i], d[n + i]]).collect())
}

fn heights_for(points: usize, heights: &[f64]) -> Result<Vec<f64>> {
    if heights.len() != points {
        return fail("heights must have one value per point");
    }
    Ok(heights.to_vec())
}

/// Trees from a list of columns; missing columns take the Python defaults.
pub fn trees_from_r(trees: &List) -> Result<Vec<Tree>> {
    let map: HashMap<&str, Robj> = trees.clone().try_into()?;
    let x = doubles(map.get("x").ok_or_else(|| Error::Other("trees have no `x`".into()))?, "x")?;
    let n = x.len();
    let col = |k: &str, default: f64| -> Result<Vec<f64>> {
        match map.get(k) {
            Some(v) => {
                let c = doubles(v, k)?;
                if c.len() != n {
                    return fail(format!("trees: column `{k}` has {} values for {n} trees", c.len()));
                }
                Ok(c)
            }
            None => Ok(vec![default; n]),
        }
    };
    let (id, y, dbh, height, np, inl, ns, rmse, lean, q) = (col("tree_id", 0.0)?, col("y", 0.0)?, col("dbh", f64::NAN)?, col("height", f64::NAN)?, col("n_points", 0.0)?, col("inlier_fraction", f64::NAN)?, col("n_slices", 0.0)?, col("rmse", f64::NAN)?, col("lean_deg", f64::NAN)?, col("quality", f64::NAN)?);
    Ok((0..n)
        .map(|i| Tree { tree_id: id[i] as i64, x: x[i], y: y[i], dbh: dbh[i], height: height[i], n_points: np[i].max(0.0) as usize, inlier_fraction: inl[i], n_slices: ns[i].max(0.0) as usize, rmse: rmse[i], lean_deg: lean[i], quality: q[i] })
        .collect())
}

pub fn trees_to_r(trees: &[Tree]) -> List {
    let f = |g: &dyn Fn(&Tree) -> f64| trees.iter().map(g).collect::<Vec<f64>>();
    let values: Vec<Robj> = vec![
        f(&|t| t.tree_id as f64).into(),
        f(&|t| t.x).into(),
        f(&|t| t.y).into(),
        f(&|t| t.dbh).into(),
        f(&|t| t.height).into(),
        f(&|t| t.n_points as f64).into(),
        f(&|t| t.inlier_fraction).into(),
        f(&|t| t.n_slices as f64).into(),
        f(&|t| t.rmse).into(),
        f(&|t| t.lean_deg).into(),
        f(&|t| t.quality).into(),
    ];
    List::from_names_and_values(TREE_COLUMNS, values).expect("names match values")
}

fn labels_from_r(labels: &Robj, n: usize) -> Result<Vec<i64>> {
    let l: Vec<i64> = doubles(labels, "labels")?.iter().map(|&v| if v.is_nan() { -1 } else { v as i64 }).collect();
    if n != usize::MAX && l.len() != n {
        return fail("labels must have one value per point");
    }
    Ok(l)
}

fn labels_to_r(labels: &[i64]) -> Vec<i32> {
    labels.iter().map(|&v| v as i32).collect()
}

fn opt_pair(v: &Robj, what: &str) -> Result<Option<[f64; 2]>> {
    if v.is_null() {
        return Ok(None);
    }
    let d = doubles(v, what)?;
    if d.len() < 2 {
        return fail(format!("{what} must have two values"));
    }
    Ok(Some([d[0], d[1]]))
}

// -------------------------------------------------------------------- circles

/// @noRd
#[extendr]
fn core_fit_circle(xy: Robj) -> Result<Vec<f64>> {
    let (cx, cy, r, rmse) = trees::fit_circle(&xy_from_r(&xy)?).map_err(err)?;
    Ok(vec![cx, cy, r, rmse])
}

/// @noRd
#[extendr]
fn core_fit_circle_ransac(xy: Robj, threshold: f64, iterations: f64, min_radius: f64, max_radius: f64, seed: f64) -> Result<List> {
    let p = trees::RansacCircleParams { threshold, iterations: iterations.max(0.0) as usize, min_radius, max_radius, seed: seed as u64 };
    let (cx, cy, r, inl) = trees::fit_circle_ransac(&xy_from_r(&xy)?, &p).map_err(err)?;
    Ok(list!(cx = cx, cy = cy, r = r, inliers = inl.iter().map(|&b| Rbool::from(b)).collect::<Logicals>()))
}

/// @noRd
#[extendr]
fn core_convex_hull_area(xy: Robj) -> Result<f64> {
    Ok(trees::convex_hull_area(&xy_from_r(&xy)?))
}

// ----------------------------------------------------------- stems and trees

/// Stem detection settings from a named list over the defaults.
fn stem_params(params: &List) -> Result<StemParams> {
    let mut p = StemParams::default();
    for (name, v) in params.iter() {
        let d = doubles(&v, name).or_else(|_| v.as_logical_slice().map(|b| b.iter().map(|x| if x.is_true() { 1.0 } else { 0.0 }).collect()).ok_or_else(|| Error::Other(format!("`{name}` must be numeric or logical"))))?;
        let Some(&x) = d.first() else { return fail(format!("`{name}` is empty")) };
        let u = x.max(0.0) as usize;
        let b = x != 0.0;
        match name {
            "slice_min" => p.slice_min = x,
            "slice_max" => p.slice_max = x,
            "slice_thickness" => p.slice_thickness = x,
            "slice_step" => p.slice_step = x,
            "reference_height" => p.reference_height = x,
            "min_radius" => p.min_radius = x,
            "max_radius" => p.max_radius = x,
            "cluster_cell" => p.cluster_cell = x,
            "min_cluster_points" => p.min_cluster_points = u,
            "max_cluster_extent" => p.max_cluster_extent = x,
            "ransac_iterations" => p.ransac_iterations = u,
            "ransac_tolerance" => p.ransac_tolerance = x,
            "max_circles_per_cluster" => p.max_circles_per_cluster = u,
            "min_circle_inliers" => p.min_circle_inliers = u,
            "min_coverage" => p.min_coverage = x,
            "min_arc_deg" => p.min_arc_deg = x,
            "max_circle_rmse" => p.max_circle_rmse = x,
            "link_radius" => p.link_radius = x,
            "link_radius_ratio" => p.link_radius_ratio = x,
            "link_radius_abs" => p.link_radius_abs = x,
            "min_slices" => p.min_slices = u,
            "max_lean_deg" => p.max_lean_deg = x,
            "prefilter" => p.prefilter = b,
            "prefilter_k" => p.prefilter_k = u,
            "prefilter_max_nz" => p.prefilter_max_nz = x,
            "prefilter_max_variation" => p.prefilter_max_variation = x,
            "seed" => p.seed = u as u64,
            "ransac_block" => p.ransac_block = u,
            "ransac_presample" => p.ransac_presample = b,
            "recluster_wide" => p.recluster_wide = b,
            "cluster_grid_at_slice_min" => p.cluster_grid_at_slice_min = b,
            "band_top_inclusive" => p.band_top_inclusive = b,
            "shared_rng" => p.shared_rng = b,
            "taper_weight_power" => p.taper_weight_power = x,
            "min_total_points" => p.min_total_points = u,
            _ => return fail(format!("detect_stems has no parameter `{name}`")),
        }
    }
    Ok(p)
}

/// @noRd
#[extendr]
fn core_detect_stems(xyz: Robj, heights: &[f64], params: List) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    let h = heights_for(p.len(), heights)?;
    let found = sylva_rs::stems::detect_stems(&p, &h, &stem_params(&params)?);
    Ok(trees_to_r(&found))
}

/// @noRd
#[extendr]
fn core_dbh_profile(xyz: Robj, heights: &[f64], cx: f64, cy: f64, at_heights: &[f64], slice_thickness: f64, search_radius: f64) -> Result<Vec<f64>> {
    let p = xyz_from_r(&xyz)?;
    let h = heights_for(p.len(), heights)?;
    Ok(trees::dbh_profile(&p, &h, cx, cy, at_heights, slice_thickness, search_radius))
}

/// Graph settings from a named list (every field of SegmentParams).
fn segment_params(g: &List) -> Result<SegmentParams> {
    let m: HashMap<&str, Robj> = g.clone().try_into()?;
    let num = |k: &str| -> Result<f64> {
        let v = m.get(k).ok_or_else(|| Error::Other(format!("graph settings have no `{k}`")))?;
        if let Some(b) = v.as_logical_slice() {
            return Ok(if b.first().is_some_and(|x| x.is_true()) { 1.0 } else { 0.0 });
        }
        doubles(v, k)?.first().copied().ok_or_else(|| Error::Other(format!("`{k}` is empty")))
    };
    Ok(SegmentParams {
        k: num("k")?.max(0.0) as usize,
        max_edge: num("max_edge")?,
        voxel_size: num("voxel_size")?,
        seed_height: num("seed_height")?,
        seed_radius: num("seed_radius")?,
        seed_ring: num("seed_ring")? != 0.0,
        power: num("power")?,
        angle_penalty: num("angle_penalty")? != 0.0,
        gravity: num("gravity")?,
        cut_above_ground: num("cut_above_ground")?,
        height_prior: num("height_prior")? != 0.0,
        height_prior_radius: num("height_prior_radius")?,
        height_prior_power: num("height_prior_power")?,
        low_height: num("low_height")?,
        low_radius: num("low_radius")?,
        wood_costs: num("wood_costs")? != 0.0,
        wood_k: num("wood_k")?.max(0.0) as usize,
        wood_threshold: num("wood_threshold")?,
        understorey_height: num("understorey_height")?,
        understorey_band: num("understorey_band")?,
    })
}

/// @noRd
#[extendr]
fn core_segment_trees(xyz: Robj, heights: &[f64], trees: List, graph: List) -> Result<Vec<i32>> {
    let p = xyz_from_r(&xyz)?;
    let h = heights_for(p.len(), heights)?;
    let t = trees_from_r(&trees)?;
    Ok(labels_to_r(&trees::segment_trees(&p, &h, &t, &segment_params(&graph)?)))
}

/// @noRd
#[extendr]
fn core_merge_branches(xyz: Robj, heights: &[f64], trees: List, graph: List, ground_height: f64, trunk_scale: f64, trunk_min: f64, search_radius: f64) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    let h = heights_for(p.len(), heights)?;
    let t = trees_from_r(&trees)?;
    let (kept, merged) = trees::merge_branches(&p, &h, &t, &segment_params(&graph)?, ground_height, trunk_scale, trunk_min, search_radius);
    Ok(list!(trees = trees_to_r(&kept), merged_into = merged.iter().map(|&v| v as f64).collect::<Vec<f64>>()))
}

/// @noRd
#[extendr]
fn core_tree_heights(heights: &[f64], labels: Robj, trees: List, percentile: f64) -> Result<List> {
    let l = labels_from_r(&labels, heights.len())?;
    let mut t = trees_from_r(&trees)?;
    trees::tree_heights(heights, &l, &mut t, percentile);
    Ok(trees_to_r(&t))
}

fn crown_list(m: [f64; 4]) -> List {
    list!(crown_area = m[0], crown_base_height = m[1], crown_depth = m[2], crown_diameter = m[3])
}

/// @noRd
#[extendr]
fn core_crown_metrics(xyz: Robj, heights: &[f64], labels: Robj, tree_id: f64, crown_base_fraction: f64) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    let h = heights_for(p.len(), heights)?;
    let l = labels_from_r(&labels, p.len())?;
    Ok(trees::crown_metrics(&p, &h, &l, tree_id as i64, crown_base_fraction).map(crown_list).unwrap_or_else(|| List::new(0)))
}

/// @noRd
#[extendr]
fn core_crown_metrics_all(xyz: Robj, heights: &[f64], labels: Robj, crown_base_fraction: f64) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    let h = heights_for(p.len(), heights)?;
    let l = labels_from_r(&labels, p.len())?;
    let all = trees::crown_metrics_all(&p, &h, &l, crown_base_fraction);
    let col = |k: usize| all.iter().map(|(_, m)| m[k]).collect::<Vec<f64>>();
    Ok(list!(
        tree_id = all.iter().map(|(id, _)| *id as f64).collect::<Vec<f64>>(),
        crown_area = col(0),
        crown_base_height = col(1),
        crown_depth = col(2),
        crown_diameter = col(3)
    ))
}

/// @noRd
#[extendr]
fn core_crown_shape(xyz: Robj, base_xy: Robj, z_min: f64, slice: f64) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    if slice.is_nan() || slice <= 0.0 {
        return fail("slice_height must be positive");
    }
    let c = trees::crown_shape(&p, opt_pair(&base_xy, "base_xy")?, z_min, slice);
    Ok(list!(
        projected_area = c.projected_area,
        diameter = c.diameter,
        max_width = c.max_width,
        volume = c.volume,
        surface = c.surface,
        base_height = c.base_height,
        top_height = c.top_height,
        offset = c.offset,
        offset_direction = c.offset_direction,
        asymmetry = c.asymmetry
    ))
}

// -------------------------------------------------- pruning, basal area, buttresses

/// @noRd
#[extendr]
fn core_prune_trees(trees: List, labels: Robj, min_height: f64, merge_radius: f64, max_dbh: Robj, min_quality_short: f64, short_slices: f64) -> Result<List> {
    let t = trees_from_r(&trees)?;
    let l = labels_from_r(&labels, usize::MAX)?;
    let max_dbh = if max_dbh.is_null() { None } else { doubles(&max_dbh, "max_dbh")?.first().copied() };
    let p = tp::PruneParams { min_height, merge_radius, max_dbh, min_quality_short, short_slices: short_slices as i64 };
    let (kept, lab) = tp::prune_trees(&t, &l, &p).map_err(err)?;
    let rows: Vec<f64> = kept.iter().map(|(i, _)| *i as f64 + 1.0).collect();
    let survivors: Vec<Tree> = kept.into_iter().map(|(_, t)| t).collect();
    Ok(list!(row = rows, trees = trees_to_r(&survivors), labels = labels_to_r(&lab)))
}

/// @noRd
#[extendr]
fn core_basal_area(dbh: &[f64], area: f64, min_dbh: f64) -> Result<f64> {
    tp::basal_area(dbh, area, min_dbh).map_err(err)
}

/// @noRd
#[extendr]
fn core_detect_buttress(xyz: Robj, heights: &[f64], base_xy: Robj, max_radius: f64, slice_height: f64, max_height: f64, low: f64, bins: f64, max_circle_fit: f64, min_ridges: f64, bark_only: bool, voxel: f64) -> Result<List> {
    let p = xyz_from_r(&xyz)?;
    let h = heights_for(p.len(), heights)?;
    let params = bd::ButtressParams { max_radius, slice_height, max_height, low, bins: bins.max(0.0) as usize, max_circle_fit, min_ridges: min_ridges.max(0.0) as usize, bark_only, voxel };
    let b = bd::detect_buttress(&p, &h, opt_pair(&base_xy, "base_xy")?, &params).map_err(err)?;
    Ok(list!(
        buttressed = b.buttressed,
        base_circle_fit = b.base_circle_fit,
        stem_circle_fit = b.stem_circle_fit,
        stem_radius = b.stem_radius,
        ridges = b.ridges as i32,
        ridge_share = b.ridge_share,
        spread = b.spread,
        top = b.top,
        centre = b.centre.to_vec()
    ))
}

extendr_module! {
    mod trees;
    fn core_fit_circle;
    fn core_fit_circle_ransac;
    fn core_convex_hull_area;
    fn core_detect_stems;
    fn core_dbh_profile;
    fn core_segment_trees;
    fn core_merge_branches;
    fn core_tree_heights;
    fn core_crown_metrics;
    fn core_crown_metrics_all;
    fn core_crown_shape;
    fn core_prune_trees;
    fn core_basal_area;
    fn core_detect_buttress;
}
