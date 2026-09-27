// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for quantitative structure models: cylinder fits, skeletons,
//! QSMs (an `n x 12` double matrix of cylinders), their totals, cuts,
//! metrics, meshes (vertices and 0-based faces) and files, buttress meshes
//! joined to them, and the plot loop with its table and files. A plot
//! crosses as a list of trees, each `list(tree_id, cylinders, points,
//! height, buttress)` with `NULL` for what is missing.
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use extendr_api::prelude::*;
use sylva_rs::leaf_model::qsm_from_rows;
use sylva_rs::mesh_io;
use sylva_rs::qsm::buttress::{Buttress, ButtressParams};
use sylva_rs::qsm::metrics::{branches, tree_metrics};
use sylva_rs::qsm::wood::{wood_mask, WoodParams};
use sylva_rs::qsm::{self, QsmParams};
use sylva_rs::qsm_ops::{self as ops, Row};
use sylva_rs::qsm_plot::{self as plot, ButtressView, PlotEntry, PlotParams};
use sylva_rs::trees::CrownShape;
use sylva_rs::Point;

use crate::convert::{doubles, err, fail, faces_from_r, faces_to_r, matrix_from_rows, optional_f64, xyz_from_r, xyz_to_r, Result};

/// Cylinder rows from an `n x 12` double matrix.
fn rows_from_r(cylinders: &Robj) -> Result<Vec<Row>> {
    if cylinders.is_null() {
        return Ok(Vec::new());
    }
    let m: RMatrix<f64> = cylinders.try_into().map_err(|_| Error::Other("cylinders must be a double matrix with 12 columns".into()))?;
    if m.ncols() != 12 {
        return fail("cylinder array must have 12 columns");
    }
    let n = m.nrows();
    let d = m.data();
    Ok((0..n).map(|i| std::array::from_fn(|c| d[c * n + i])).collect())
}

fn rows_to_r(rows: &[Row]) -> Robj {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    matrix_from_rows(rows.len(), 12, &flat)
}

fn xy_from_r(v: &Robj, what: &str) -> Result<Option<[f64; 2]>> {
    if v.is_null() {
        return Ok(None);
    }
    let d = doubles(v, what)?;
    if d.len() < 2 {
        return fail(format!("{what} must have two values"));
    }
    Ok(Some([d[0], d[1]]))
}

fn rgb_from_r(v: &Robj) -> Result<Option<[u8; 3]>> {
    if v.is_null() {
        return Ok(None);
    }
    let d = doubles(v, "color")?;
    if d.len() != 3 || d.iter().any(|c| !(0.0..=255.0).contains(c)) {
        return fail("color must be three values from 0 to 255");
    }
    Ok(Some([d[0] as u8, d[1] as u8, d[2] as u8]))
}

fn count(v: f64, what: &str) -> Result<usize> {
    if v.is_nan() || v < 0.0 || v.fract() != 0.0 {
        return fail(format!("{what} must be a whole number, at least 0"));
    }
    Ok(v as usize)
}

// ------------------------------------------------------------ fits

/// @noRd
#[extendr]
fn core_fit_cylinder(xyz: Robj, axis_init: Robj) -> Result<List> {
    let axis = if axis_init.is_null() {
        None
    } else {
        let a = doubles(&axis_init, "axis_init")?;
        if a.len() != 3 {
            return fail("axis_init must have three values");
        }
        Some([a[0], a[1], a[2]])
    };
    let f = qsm::fit_cylinder(&xyz_from_r(&xyz)?, axis).map_err(err)?;
    Ok(list!(point = f.point.to_vec(), axis = f.axis.to_vec(), radius = f.radius, rmse = f.rmse))
}

/// @noRd
#[extendr]
fn core_fit_cylinder_ransac(xyz: Robj, threshold: f64, iterations: f64, sample_size: f64, seed: f64) -> Result<List> {
    let (f, inl) = qsm::fit_cylinder_ransac(&xyz_from_r(&xyz)?, threshold, count(iterations, "iterations")?, count(sample_size, "sample_size")?, seed as u64).map_err(err)?;
    Ok(list!(point = f.point.to_vec(), axis = f.axis.to_vec(), radius = f.radius, rmse = f.rmse, inliers = inl))
}

/// QSM settings from a named list; each name must be a `QsmParams` field.
fn qsm_params_from_r(params: &List) -> Result<QsmParams> {
    let mut p = QsmParams::default();
    for (k, v) in params.iter() {
        let x = optional_f64(&v, k)?.ok_or_else(|| Error::Other(format!("{k} must not be NULL")))?;
        match k {
            "k" => p.k = count(x, k)?,
            "max_edge" => p.max_edge = x,
            "bin_length" => p.bin_length = x,
            "min_points" => p.min_points = count(x, k)?,
            "ransac_threshold" => p.ransac_threshold = x,
            "max_radius" => p.max_radius = x,
            "taper_limit" => p.taper_limit = x,
            "max_rmse" => p.max_rmse = x,
            "smooth_steps" => p.smooth_steps = count(x, k)?,
            "apex_radius" => p.apex_radius = x,
            "min_arc_deg" => p.min_arc_deg = x,
            "min_inlier_fraction" => p.min_inlier_fraction = x,
            "prune_points" => p.prune_points = count(x, k)?,
            "fit_min_points" => p.fit_min_points = count(x, k)?,
            "crop_length" => p.crop_length = x,
            "butt_height" => p.butt_height = x,
            "relative_tolerance" => p.relative_tolerance = x,
            "base_radius" => p.base_radius = x,
            "allometry_tolerance" => p.allometry_tolerance = x,
            "buttress_equivalent_area" => p.buttress_equivalent_area = x != 0.0,
            "buttress_max_inlier_fraction" => p.buttress_max_inlier_fraction = x,
            "pipe_slack" => p.pipe_slack = x,
            "branch_min_inlier_fraction" => p.branch_min_inlier_fraction = x,
            "spacing_scale" => p.spacing_scale = x,
            "radius_power" => p.radius_power = x,
            "power_above_spacing" => p.power_above_spacing = x,
            "sensor_noise" => p.sensor_noise = x,
            "cluster_eps" => p.cluster_eps = x,
            "centre_fit_points" => p.centre_fit_points = count(x, k)?,
            "radius_smooth_steps" => p.radius_smooth_steps = count(x, k)?,
            "butt_swell" => p.butt_swell = x,
            "butt_vertical_run" => p.butt_vertical_run = count(x, k)?,
            "butt_max_lean_deg" => p.butt_max_lean_deg = x,
            "chain_max_d" => p.chain_max_d = x,
            "fourier_min_radius" => p.fourier_min_radius = x,
            _ => return fail(format!("unknown QSM setting `{k}`")),
        }
    }
    Ok(p)
}

/// @noRd
#[extendr]
fn core_build_qsm(xyz: Robj, base_xy: Robj, params: List) -> Result<Robj> {
    let p = qsm_params_from_r(&params)?;
    let q = qsm::build_qsm(&xyz_from_r(&xyz)?, xy_from_r(&base_xy, "base_xy")?, &p).map_err(err)?;
    Ok(rows_to_r(&q.to_rows()))
}

/// @noRd
#[extendr]
fn core_skeletonize(xyz: Robj, base_xy: Robj, k: f64, max_edge: f64, bin_length: f64) -> Result<List> {
    let p = QsmParams { k: count(k, "k")?, max_edge, bin_length, ..Default::default() };
    let s = qsm::skeletonize(&xyz_from_r(&xyz)?, xy_from_r(&base_xy, "base_xy")?, &p).map_err(err)?;
    let edges: Vec<f64> = s.edges.iter().flat_map(|(c, p)| [*c as f64, *p as f64]).collect();
    Ok(list!(
        segment_id = s.segment_id.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
        geodesic = s.geodesic,
        centres = xyz_to_r(&s.centres),
        edges = matrix_from_rows(s.edges.len(), 2, &edges)
    ))
}

/// @noRd
#[extendr]
fn core_wood_mask(xyz: Robj, params: List) -> Result<Vec<bool>> {
    let mut p = WoodParams::default();
    for (k, v) in params.iter() {
        let x = optional_f64(&v, k)?.ok_or_else(|| Error::Other(format!("{k} must not be NULL")))?;
        match k {
            "k" => p.k = count(x, k)?,
            "threshold" => p.high_threshold = x,
            "medium_threshold" => p.medium_threshold = x,
            "scale_radius" => p.scale_radius = x,
            "passage" => p.passage = x != 0.0,
            "min_passage" => p.min_passage = count(x, k)?,
            "target_res" => p.target_res = x,
            "graph_k" => p.graph_k = count(x, k)?,
            "max_edge" => p.max_edge = x,
            "base_height" => p.base_height = x,
            "assign_dist" => p.assign_dist = x,
            "assign_scale" => p.assign_scale = x,
            "component_res" => p.component_res = x,
            "component_min" => p.component_min = count(x, k)?,
            "sor_k" => p.sor_k = count(x, k)?,
            "sor_std" => p.sor_std = x,
            "dilate_dist" => p.dilate_dist = x,
            _ => return fail(format!("unknown wood filter setting `{k}`")),
        }
    }
    Ok(wood_mask(&xyz_from_r(&xyz)?, &p))
}

// ------------------------------------------------------------ models

/// @noRd
#[extendr]
fn core_qsm_totals(cylinders: Robj) -> Result<List> {
    let t = ops::totals(&rows_from_r(&cylinders)?);
    Ok(list!(total_volume = t.total_volume, stem_volume = t.stem_volume, branch_volume = t.branch_volume, total_length = t.total_length, max_branch_order = t.max_branch_order))
}

/// @noRd
#[extendr]
fn core_qsm_ends(cylinders: Robj) -> Result<Robj> {
    Ok(xyz_to_r(&ops::ends(&rows_from_r(&cylinders)?)))
}

/// @noRd
#[extendr]
fn core_qsm_volumes(cylinders: Robj) -> Result<Vec<f64>> {
    Ok(ops::volumes(&rows_from_r(&cylinders)?))
}

/// @noRd
#[extendr]
fn core_qsm_dbh(cylinders: Robj) -> Result<f64> {
    Ok(qsm_from_rows(&rows_from_r(&cylinders)?).dbh())
}

/// @noRd
#[extendr]
fn core_qsm_volume_above(cylinders: Robj, z: f64) -> Result<f64> {
    Ok(ops::volume_above(&rows_from_r(&cylinders)?, z))
}

/// @noRd
#[extendr]
fn core_qsm_above(cylinders: Robj, z: f64) -> Result<Robj> {
    Ok(rows_to_r(&ops::above(&rows_from_r(&cylinders)?, z)))
}

fn crown_to_r(c: &CrownShape) -> List {
    list!(projected_area = c.projected_area, diameter = c.diameter, max_width = c.max_width, volume = c.volume, surface = c.surface, base_height = c.base_height, top_height = c.top_height, offset = c.offset, offset_direction = c.offset_direction, asymmetry = c.asymmetry)
}

fn usizes(v: &[usize]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// @noRd
#[extendr]
fn core_qsm_metrics(cylinders: Robj, crown_branch_length: f64, crown_slice: f64) -> Result<List> {
    let m = tree_metrics(&qsm_from_rows(&rows_from_r(&cylinders)?), crown_branch_length, crown_slice.max(1e-3));
    let l = List::from_names_and_values(
        [
            "height", "dbh", "total_volume", "stem_volume", "branch_volume", "total_length", "stem_length", "max_order", "n_branches_by_order", "length_by_order", "volume_by_order", "n_tips", "path_fraction", "crown_base_height", "lean", "lean_direction", "sweep", "taper_heights", "taper_radii", "crown",
            "measured_volume_fraction", "measured_length_fraction", "median_insertion_angle", "median_branch_zenith",
        ],
        [
            Robj::from(m.height),
            m.dbh.into(),
            m.total_volume.into(),
            m.stem_volume.into(),
            m.branch_volume.into(),
            m.total_length.into(),
            m.stem_length.into(),
            (m.max_order as f64).into(),
            usizes(&m.n_branches_by_order).into(),
            m.length_by_order.clone().into(),
            m.volume_by_order.clone().into(),
            (m.n_tips as f64).into(),
            m.path_fraction.into(),
            m.crown_base_height.into(),
            m.lean.into(),
            m.lean_direction.into(),
            m.sweep.into(),
            m.taper_heights.clone().into(),
            m.taper_radii.clone().into(),
            crown_to_r(&m.crown).into(),
            m.measured_volume_fraction.into(),
            m.measured_length_fraction.into(),
            m.median_insertion_angle.into(),
            m.median_branch_zenith.into(),
        ],
    )?;
    Ok(l)
}

/// @noRd
#[extendr]
fn core_qsm_branches(cylinders: Robj) -> Result<List> {
    let b = branches(&qsm_from_rows(&rows_from_r(&cylinders)?));
    macro_rules! col {
        ($f:expr) => {
            b.iter().map($f).collect::<Vec<f64>>()
        };
    }
    Ok(list!(
        id = col!(|x| x.id as f64),
        order = col!(|x| x.order as f64),
        parent = col!(|x| x.parent as f64),
        n_cylinders = col!(|x| x.n_cylinders as f64),
        length = col!(|x| x.length),
        volume = col!(|x| x.volume),
        base_radius = col!(|x| x.base_radius),
        mean_radius = col!(|x| x.mean_radius),
        base_height = col!(|x| x.base_height),
        tip_height = col!(|x| x.tip_height),
        insertion_angle = col!(|x| x.insertion_angle),
        zenith = col!(|x| x.zenith),
        azimuth = col!(|x| x.azimuth),
        tortuosity = col!(|x| x.tortuosity),
        n_children = col!(|x| x.n_children as f64),
        measured_fraction = col!(|x| x.measured_fraction)
    ))
}

/// @noRd
#[extendr]
fn core_qsm_summary(cylinders: Robj) -> Result<List> {
    let rows = rows_from_r(&cylinders)?;
    let q = qsm_from_rows(&rows);
    Ok(list!(n_cylinders = rows.len() as f64, total_volume = q.total_volume(), stem_volume = q.stem_volume(), branch_volume = q.branch_volume(), total_length = q.total_length(), max_branch_order = q.max_branch_order() as f64, dbh = q.dbh()))
}

/// @noRd
#[extendr]
fn core_qsm_mesh(cylinders: Robj, sides: f64, contiguous: bool) -> Result<List> {
    let (v, f, o) = ops::mesh(&rows_from_r(&cylinders)?, count(sides, "sides")?, contiguous);
    Ok(list!(vertices = xyz_to_r(&v), faces = faces_to_r(&f), owner = o.iter().map(|&x| x as f64).collect::<Vec<f64>>()))
}

/// @noRd
#[extendr]
fn core_qsm_read_csv(path: &str) -> Result<Robj> {
    Ok(rows_to_r(&ops::read_csv(path).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_qsm_write_csv(cylinders: Robj, path: &str) -> Result<()> {
    qsm_from_rows(&rows_from_r(&cylinders)?).write_csv(path).map_err(err)
}

/// @noRd
#[extendr]
fn core_qsm_write_treefile(cylinders: Robj, path: &str) -> Result<()> {
    qsm_from_rows(&rows_from_r(&cylinders)?).write_treefile(path).map_err(err)
}

/// @noRd
#[extendr]
fn core_qsm_write_obj(path: &str, cylinders: Robj, sides: f64, contiguous: bool) -> Result<()> {
    ops::write_model_obj(path, &rows_from_r(&cylinders)?, count(sides, "sides")?, contiguous).map_err(err)
}

/// @noRd
#[extendr]
fn core_qsm_write_ply(path: &str, cylinders: Robj, sides: f64, color: Robj, contiguous: bool) -> Result<()> {
    ops::write_model_ply(path, &rows_from_r(&cylinders)?, count(sides, "sides")?, rgb_from_r(&color)?, contiguous).map_err(err)
}

/// @noRd
#[extendr]
fn core_write_ply_mesh(path: &str, vertices: Robj, faces: Robj, face_rgb: Robj) -> Result<()> {
    let f: Vec<[i32; 3]> = faces_from_r(&faces)?.iter().map(|t| [t[0] as i32, t[1] as i32, t[2] as i32]).collect();
    let rgb = if face_rgb.is_null() {
        None
    } else {
        let m: RMatrix<f64> = face_rgb.try_into().map_err(|_| Error::Other("face_rgb must be a numeric matrix with 3 columns".into()))?;
        let (n, d) = (m.nrows(), m.data());
        if m.ncols() != 3 || d.iter().any(|c| !(0.0..=255.0).contains(c)) {
            return fail("face_rgb must have 3 columns of values from 0 to 255");
        }
        Some((0..n).map(|i| [d[i] as u8, d[n + i] as u8, d[2 * n + i] as u8]).collect::<Vec<[u8; 3]>>())
    };
    mesh_io::write_ply(path, &xyz_from_r(&vertices)?, &f, rgb.as_deref()).map_err(err)
}

// ------------------------------------------------------------ buttresses

fn buttress_to_r(b: &Buttress) -> List {
    list!(
        vertices = xyz_to_r(&b.vertices),
        faces = faces_to_r(&b.faces),
        volume = b.volume,
        top = b.top,
        top_z = b.top_z,
        heights = b.heights.clone(),
        areas = b.areas.clone(),
        solidities = b.solidities.clone(),
        open = b.open.clone()
    )
}

/// @noRd
#[extendr]
fn core_qsm_buttress_mesh(xyz: Robj, heights: &[f64], cx: f64, cy: f64, ground_z: Robj, resolution: f64, slice: f64, close_radius: f64, max_radius: f64, max_height: f64, top: Robj, solidity: f64, max_flare: f64, smooth: f64) -> Result<List> {
    let p = ButtressParams { resolution, slice, close_radius, max_radius, max_height, top: optional_f64(&top, "top")?, solidity, max_flare, smooth: count(smooth, "smooth")?, ..Default::default() };
    let b = plot::buttress_mesh(&xyz_from_r(&xyz)?, heights, cx, cy, optional_f64(&ground_z, "ground_z")?, &p).map_err(err)?;
    Ok(buttress_to_r(&b))
}

/// @noRd
#[extendr]
fn core_qsm_fuse(vertices: Robj, faces: Robj, volume: f64, top_z: f64, cylinders: Robj, sides: f64, contiguous: bool, overlap: f64) -> Result<List> {
    let t = ops::fuse(&xyz_from_r(&vertices)?, &faces_from_r(&faces)?, volume, top_z, &rows_from_r(&cylinders)?, count(sides, "sides")?, contiguous, overlap).map_err(err)?;
    Ok(list!(
        vertices = xyz_to_r(&t.vertices),
        faces = faces_to_r(&t.faces),
        part = t.part.iter().map(|&p| p as i32).collect::<Vec<i32>>(),
        buttress_volume = t.buttress_volume,
        wood_volume = t.wood_volume,
        top_z = t.top_z,
        offset = t.offset,
        overhang = t.overhang
    ))
}

fn parts_from_r(part: &[f64]) -> Result<Vec<u8>> {
    part.iter().map(|&p| if (0.0..=255.0).contains(&p) { Ok(p as u8) } else { fail("part must hold labels from 0 to 255") }).collect()
}

/// @noRd
#[extendr]
fn core_tree_mesh_write_obj(path: &str, vertices: Robj, faces: Robj, part: &[f64]) -> Result<()> {
    ops::write_tree_mesh_obj(path, &xyz_from_r(&vertices)?, &faces_from_r(&faces)?, &parts_from_r(part)?).map_err(err)
}

/// @noRd
#[extendr]
fn core_tree_mesh_write_ply(path: &str, vertices: Robj, faces: Robj, part: &[f64], color: Robj) -> Result<()> {
    ops::write_tree_mesh_ply(path, &xyz_from_r(&vertices)?, &faces_from_r(&faces)?, &parts_from_r(part)?, rgb_from_r(&color)?).map_err(err)
}

// ------------------------------------------------------------ plots

/// @noRd
#[extendr]
fn core_qsm_build_plot(xyz: Robj, labels: &[f64], heights: Robj, stem_ids: &[f64], stem_x: &[f64], stem_y: &[f64], voxel_size: f64, wood: bool, buttress: bool, min_points: f64, params: List) -> Result<List> {
    if labels.iter().any(|l| !l.is_nan() && l.fract() != 0.0) {
        return fail("labels must be whole numbers");
    }
    let labels: Vec<i64> = labels.iter().map(|&l| if l.is_nan() { -1 } else { l as i64 }).collect();
    let h = if heights.is_null() { None } else { Some(doubles(&heights, "heights")?) };
    if stem_x.len() != stem_ids.len() || stem_y.len() != stem_ids.len() {
        return fail("one stem position per stem id");
    }
    let stems: Vec<(i64, [f64; 2])> = (0..stem_ids.len()).map(|i| (stem_ids[i] as i64, [stem_x[i], stem_y[i]])).collect();
    let p = PlotParams { voxel_size, wood, buttress, min_points, qsm: qsm_params_from_r(&params)? };
    let r = plot::build_plot(&xyz_from_r(&xyz)?, &labels, h.as_deref(), &stems, &p).map_err(err)?;
    let share = plot::median_measured_length(r.models.iter().map(|m| m.1.as_slice()));
    let ids = |v: &[i64]| v.iter().map(|&t| t as f64).collect::<Vec<f64>>();
    Ok(list!(
        model_ids = ids(&r.models.iter().map(|m| m.0).collect::<Vec<i64>>()),
        models = List::from_values(r.models.iter().map(|m| rows_to_r(&m.1))),
        buttress_ids = ids(&r.buttresses.iter().map(|b| b.0).collect::<Vec<i64>>()),
        buttresses = List::from_values(r.buttresses.iter().map(|b| buttress_to_r(&b.1))),
        skipped_ids = ids(&r.skipped.iter().map(|s| s.0).collect::<Vec<i64>>()),
        skipped = r.skipped.iter().map(|s| s.1.clone()).collect::<Vec<String>>(),
        point_ids = ids(&r.points.iter().map(|s| s.0).collect::<Vec<i64>>()),
        points = r.points.iter().map(|s| s.1 as f64).collect::<Vec<f64>>(),
        height_ids = ids(&r.heights.iter().map(|s| s.0).collect::<Vec<i64>>()),
        heights = r.heights.iter().map(|s| s.1).collect::<Vec<f64>>(),
        median_measured_length = share.map_or_else(|| Robj::from(()), Robj::from)
    ))
}

type OwnedButtress = (Vec<Point>, Vec<[u32; 3]>, f64, f64, f64);

struct Owned {
    tree_id: i64,
    rows: Vec<Row>,
    points: Option<i64>,
    height: Option<f64>,
    buttress: Option<OwnedButtress>,
}

fn entries_from_r(entries: &List) -> Result<Vec<Owned>> {
    entries
        .values()
        .map(|e| {
            let l: HashMap<&str, Robj> = List::try_from(&e)?.try_into()?;
            let get = |k: &str| l.get(k).cloned().unwrap_or_else(|| Robj::from(()));
            let tree_id = optional_f64(&get("tree_id"), "tree_id")?.ok_or_else(|| Error::Other("a tree needs a tree_id".into()))? as i64;
            let buttress = {
                let b = get("buttress");
                if b.is_null() {
                    None
                } else {
                    let m: HashMap<&str, Robj> = List::try_from(&b)?.try_into()?;
                    let f = |k: &str| -> Result<Robj> { m.get(k).cloned().ok_or_else(|| Error::Other(format!("buttress has no `{k}`"))) };
                    let num = |k: &str| -> Result<f64> { optional_f64(&f(k)?, k)?.ok_or_else(|| Error::Other(format!("buttress `{k}` is NULL"))) };
                    Some((xyz_from_r(&f("vertices")?)?, faces_from_r(&f("faces")?)?, num("volume")?, num("top")?, num("top_z")?))
                }
            };
            Ok(Owned { tree_id, rows: rows_from_r(&get("cylinders"))?, points: optional_f64(&get("points"), "points")?.map(|p| p as i64), height: optional_f64(&get("height"), "height")?, buttress })
        })
        .collect()
}

fn views(owned: &[Owned]) -> Vec<PlotEntry<'_>> {
    owned
        .iter()
        .map(|o| PlotEntry {
            tree_id: o.tree_id,
            rows: &o.rows,
            points: o.points,
            height: o.height,
            buttress: o.buttress.as_ref().map(|(v, f, volume, top, top_z)| ButtressView { vertices: v, faces: f, volume: *volume, top: *top, top_z: *top_z }),
        })
        .collect()
}

/// @noRd
#[extendr]
fn core_plot_volumes(entries: List) -> Result<Vec<f64>> {
    let owned = entries_from_r(&entries)?;
    Ok(views(&owned).iter().map(plot::tree_volume).collect())
}

/// @noRd
#[extendr]
fn core_plot_total_volume(entries: List) -> Result<f64> {
    Ok(plot::total_volume(&views(&entries_from_r(&entries)?)))
}

/// @noRd
#[extendr]
fn core_plot_table(entries: List) -> Result<List> {
    let rows = plot::table(&views(&entries_from_r(&entries)?));
    let opt = |v: Option<f64>| v.unwrap_or(f64::NAN);
    Ok(list!(
        tree_id = rows.iter().map(|r| r.tree_id as f64).collect::<Vec<f64>>(),
        points = rows.iter().map(|r| r.points.map_or(f64::NAN, |p| p as f64)).collect::<Vec<f64>>(),
        volume_m3 = rows.iter().map(|r| r.volume_m3).collect::<Vec<f64>>(),
        dbh_m = rows.iter().map(|r| r.dbh_m).collect::<Vec<f64>>(),
        height_m = rows.iter().map(|r| r.height_m).collect::<Vec<f64>>(),
        n_cylinders = rows.iter().map(|r| r.n_cylinders as f64).collect::<Vec<f64>>(),
        measured_volume = rows.iter().map(|r| r.measured_volume).collect::<Vec<f64>>(),
        measured_length = rows.iter().map(|r| r.measured_length).collect::<Vec<f64>>(),
        buttress_m3 = rows.iter().map(|r| opt(r.buttress_m3)).collect::<Vec<f64>>(),
        buttress_top_m = rows.iter().map(|r| opt(r.buttress_top_m)).collect::<Vec<f64>>(),
        has_points = rows.iter().map(|r| r.points.is_some()).collect::<Vec<bool>>(),
        has_buttress = rows.iter().map(|r| r.buttress_m3.is_some()).collect::<Vec<bool>>()
    ))
}

/// @noRd
#[extendr]
fn core_plot_write_csv(path: &str, entries: List) -> Result<()> {
    plot::write_table_csv(path, &views(&entries_from_r(&entries)?)).map_err(err)
}

/// @noRd
#[extendr]
fn core_plot_write_meshes(directory: &str, entries: List, fmt: &str, sides: f64, contiguous: bool, prefix: &str) -> Result<Vec<String>> {
    let owned = entries_from_r(&entries)?;
    let files = plot::write_meshes(directory, &views(&owned), fmt, count(sides, "sides")?, contiguous, prefix).map_err(err)?;
    Ok(files.iter().map(|f| f.to_string_lossy().into_owned()).collect())
}

/// @noRd
#[extendr]
fn core_plot_write_cylinders(directory: &str, entries: List, prefix: &str) -> Result<()> {
    plot::write_cylinders(directory, &views(&entries_from_r(&entries)?), prefix).map_err(err)
}

extendr_module! {
    mod qsm;
    fn core_fit_cylinder;
    fn core_fit_cylinder_ransac;
    fn core_build_qsm;
    fn core_skeletonize;
    fn core_wood_mask;
    fn core_qsm_totals;
    fn core_qsm_ends;
    fn core_qsm_volumes;
    fn core_qsm_dbh;
    fn core_qsm_volume_above;
    fn core_qsm_above;
    fn core_qsm_metrics;
    fn core_qsm_branches;
    fn core_qsm_summary;
    fn core_qsm_mesh;
    fn core_qsm_read_csv;
    fn core_qsm_write_csv;
    fn core_qsm_write_treefile;
    fn core_qsm_write_obj;
    fn core_qsm_write_ply;
    fn core_write_ply_mesh;
    fn core_qsm_buttress_mesh;
    fn core_qsm_fuse;
    fn core_tree_mesh_write_obj;
    fn core_tree_mesh_write_ply;
    fn core_qsm_build_plot;
    fn core_plot_volumes;
    fn core_plot_total_volume;
    fn core_plot_table;
    fn core_plot_write_csv;
    fn core_plot_write_meshes;
    fn core_plot_write_cylinders;
}
