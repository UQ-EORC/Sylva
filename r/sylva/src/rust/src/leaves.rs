// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for foliage: leaf/wood labels, angle distributions, leaf area
//! grids (origin, voxel size, `dim = c(nz, ny, nx)` and a row-major
//! density), leaf shapes (vertices and 0-based faces), leaf insertion and
//! the OBJ reader and writers.
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use extendr_api::prelude::*;
use sylva_rs::leaf_model::{self as lm, LeafAreaGrid, LeafShape};
use sylva_rs::leaves::{self, LeafAngles};
use sylva_rs::mesh_io::{self, ObjMesh};
use sylva_rs::qsm::wood::{GbsParams, WoodParams};

use crate::convert::{doubles, err, fail, faces_from_r, faces_to_r, optional_f64, xyz_from_r, xyz_to_r, Result};
use crate::voxels::RayVoxelGrid;

fn options(opts: &List, allowed: &[&str]) -> Result<HashMap<String, Robj>> {
    let mut out = HashMap::new();
    for (k, v) in opts.iter() {
        if !allowed.contains(&k) {
            return fail(format!("unknown option `{k}`"));
        }
        out.insert(k.to_string(), v);
    }
    Ok(out)
}

fn num(m: &HashMap<String, Robj>, k: &str, default: f64) -> Result<f64> {
    match m.get(k) {
        Some(v) => Ok(optional_f64(v, k)?.unwrap_or(default)),
        None => Ok(default),
    }
}

fn angles_to_r(a: &LeafAngles) -> List {
    list!(bin_centres = a.bin_centres.clone(), density = a.density.clone(), mean = a.mean, std = a.std, beta_a = a.beta_a, beta_b = a.beta_b, chi = a.chi, de_wit = a.de_wit.map_or_else(|| Robj::from(()), Robj::from))
}

fn angles_from_r(bin_centres: &[f64], density: &[f64]) -> LeafAngles {
    LeafAngles { bin_centres: bin_centres.to_vec(), density: density.to_vec(), mean: 0.0, std: 0.0, beta_a: 0.0, beta_b: 0.0, chi: 1.0, de_wit: None }
}

pub fn grid_from_r(origin: &[f64], voxel_size: f64, dim: &[f64], density: &[f64]) -> Result<LeafAreaGrid> {
    if origin.len() != 3 || dim.len() != 3 {
        return fail("origin and dim must have three values");
    }
    LeafAreaGrid::new([origin[0], origin[1], origin[2]], voxel_size, [dim[0] as usize, dim[1] as usize, dim[2] as usize], density.to_vec()).map_err(err)
}

fn grid_to_r(g: LeafAreaGrid) -> List {
    list!(origin = g.origin.to_vec(), voxel_size = g.voxel_size, dim = g.shape.iter().map(|&v| v as f64).collect::<Vec<f64>>(), density = g.density)
}

fn shape_from_r(vertices: &Robj, faces: &Robj, length: f64, width: f64) -> Result<LeafShape> {
    let v = if vertices.is_null() { None } else { Some(xyz_from_r(vertices)?) };
    let f = if faces.is_null() { None } else { Some(faces_from_r(faces)?) };
    LeafShape::new(v, f, length, width).map_err(err)
}

fn shape_to_r(s: &LeafShape) -> List {
    list!(vertices = xyz_to_r(&s.vertices), faces = faces_to_r(&s.faces), length = s.length, width = s.width)
}

// ------------------------------------------------------------ leaf / wood

/// @noRd
#[extendr]
fn core_classify_leaf_wood(xyz: Robj, voxel_size: f64, method: &str, opts: List) -> Result<Vec<bool>> {
    let p = xyz_from_r(&xyz)?;
    match method {
        "gbs" => {
            let m = options(&opts, &["intervals", "max_angle", "linearity", "circle_error", "graph_k", "max_edge", "base_height", "min_points"])?;
            let d = GbsParams::default();
            let base = GbsParams { graph_k: num(&m, "graph_k", d.graph_k as f64)? as usize, max_edge: num(&m, "max_edge", d.max_edge)?, base_height: num(&m, "base_height", d.base_height)?, linearity: num(&m, "linearity", d.linearity)?, circle_error: num(&m, "circle_error", d.circle_error)?, min_points: num(&m, "min_points", d.min_points as f64)? as usize, ..d };
            let intervals = match m.get("intervals") {
                Some(v) if !v.is_null() => Some(doubles(v, "intervals")?),
                _ => None,
            };
            let max_angle = match m.get("max_angle") {
                Some(v) => optional_f64(v, "max_angle")?,
                None => None,
            };
            let params = lm::gbs_params_for(&p, intervals, max_angle, base);
            if params.intervals.is_empty() || params.intervals.iter().any(|v| v.is_nan() || *v <= 0.0) {
                return fail("intervals must be positive");
            }
            Ok(leaves::classify_leaf_wood_gbs(&p, voxel_size, &params))
        }
        "passage" => {
            let keys = ["k", "high_threshold", "medium_threshold", "scale_radius", "graph_k", "max_edge", "base_height", "target_res", "min_passage", "assign_dist", "assign_scale", "component_res", "component_min", "sor_k", "sor_std", "dilate_dist", "passage"];
            let m = options(&opts, &keys)?;
            let d = WoodParams { scale_radius: 0.1, ..Default::default() };
            let passage = match m.get("passage") {
                Some(v) => v.as_bool().ok_or_else(|| Error::Other("passage must be TRUE or FALSE".into()))?,
                None => d.passage,
            };
            let params = WoodParams {
                k: num(&m, "k", d.k as f64)? as usize,
                high_threshold: num(&m, "high_threshold", d.high_threshold)?,
                medium_threshold: num(&m, "medium_threshold", d.medium_threshold)?,
                scale_radius: num(&m, "scale_radius", d.scale_radius)?,
                graph_k: num(&m, "graph_k", d.graph_k as f64)? as usize,
                max_edge: num(&m, "max_edge", d.max_edge)?,
                base_height: num(&m, "base_height", d.base_height)?,
                target_res: num(&m, "target_res", d.target_res)?,
                min_passage: num(&m, "min_passage", d.min_passage as f64)? as usize,
                assign_dist: num(&m, "assign_dist", d.assign_dist)?,
                assign_scale: num(&m, "assign_scale", d.assign_scale)?,
                component_res: num(&m, "component_res", d.component_res)?,
                component_min: num(&m, "component_min", d.component_min as f64)? as usize,
                sor_k: num(&m, "sor_k", d.sor_k as f64)? as usize,
                sor_std: num(&m, "sor_std", d.sor_std)?,
                dilate_dist: num(&m, "dilate_dist", d.dilate_dist)?,
                passage,
            };
            Ok(leaves::classify_leaf_wood(&p, voxel_size, &params))
        }
        _ => fail("method must be 'passage' or 'gbs'"),
    }
}

// ------------------------------------------------------------ angles

/// @noRd
#[extendr]
fn core_leaf_inclinations(xyz: Robj, k: i32) -> Result<List> {
    let (incl, normals) = leaves::inclinations(&xyz_from_r(&xyz)?, k.max(0) as usize);
    Ok(list!(inclination = incl, normals = xyz_to_r(&normals)))
}

/// @noRd
#[extendr]
fn core_point_leaf_area(xyz: Robj, res: f64, k: i32) -> Result<List> {
    let (pts, area, incl) = leaves::point_leaf_area(&xyz_from_r(&xyz)?, res, k.max(0) as usize);
    Ok(list!(points = xyz_to_r(&pts), area = area, inclination = incl))
}

/// @noRd
#[extendr]
fn core_leaf_angle_distribution(inclination: &[f64], weights: Robj, n_bins: i32) -> Result<List> {
    let w = if weights.is_null() { None } else { Some(doubles(&weights, "weights")?) };
    if w.as_ref().is_some_and(|w| w.len() != inclination.len()) {
        return fail("weights must match inclination");
    }
    Ok(angles_to_r(&leaves::angle_distribution(inclination, w.as_deref(), n_bins.max(0) as usize)))
}

/// @noRd
#[extendr]
fn core_leaf_de_wit(name: &str, n_bins: i32) -> Result<List> {
    lm::de_wit(name, n_bins.max(0) as usize).map(|a| angles_to_r(&a)).ok_or_else(|| Error::Other(format!("unknown de Wit type `{name}` (one of {})", lm::DE_WIT_TYPES.join(", "))))
}

/// @noRd
#[extendr]
fn core_leaf_projection_histogram(bin_centres: &[f64], density: &[f64], beam_zenith: &[f64]) -> Vec<f64> {
    let a = angles_from_r(bin_centres, density);
    beam_zenith.iter().map(|&t| leaves::projection(&a, t)).collect()
}

// ------------------------------------------------------------ leaf area grids

/// @noRd
#[extendr]
fn core_leaf_area_density(xyz: Robj, voxel_size: f64, res: f64, k: i32) -> Result<List> {
    Ok(grid_to_r(lm::leaf_area_density(&xyz_from_r(&xyz)?, voxel_size, res, k.max(0) as usize).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_leaf_grid_area(origin: &[f64], voxel_size: f64, dim: &[f64], density: &[f64]) -> Result<Vec<f64>> {
    Ok(grid_from_r(origin, voxel_size, dim, density)?.area())
}

/// @noRd
#[extendr]
fn core_leaf_grid_total_area(origin: &[f64], voxel_size: f64, dim: &[f64], density: &[f64]) -> Result<f64> {
    Ok(grid_from_r(origin, voxel_size, dim, density)?.total_area())
}

/// @noRd
#[extendr]
fn core_leaf_grid_scaled(origin: &[f64], voxel_size: f64, dim: &[f64], density: &[f64], total_area: f64) -> Result<Vec<f64>> {
    Ok(grid_from_r(origin, voxel_size, dim, density)?.scaled_to(total_area).density)
}

/// @noRd
#[extendr]
fn core_leaf_grid_profile(origin: &[f64], voxel_size: f64, dim: &[f64], density: &[f64]) -> Result<List> {
    let (z, area) = grid_from_r(origin, voxel_size, dim, density)?.profile();
    Ok(list!(z = z, area = area))
}

/// @noRd
#[extendr]
fn core_leaf_grid_cells(origin: &[f64], voxel_size: f64, dim: &[f64], density: &[f64]) -> Result<List> {
    let (c, a) = grid_from_r(origin, voxel_size, dim, density)?.cells();
    Ok(list!(centres = xyz_to_r(&c), area = a))
}

/// @noRd
#[extendr]
fn core_leaf_grid_from_voxels(grid: &RayVoxelGrid, field: &str) -> Result<List> {
    Ok(grid_to_r(LeafAreaGrid::from_voxels(&grid.inner, field).map_err(err)?))
}

// ------------------------------------------------------------ leaf shapes

/// @noRd
#[extendr]
fn core_leaf_shape(vertices: Robj, faces: Robj, length: f64, width: f64) -> Result<List> {
    Ok(shape_to_r(&shape_from_r(&vertices, &faces, length, width)?))
}

/// @noRd
#[extendr]
fn core_leaf_shape_area(vertices: Robj, faces: Robj, length: f64, width: f64) -> Result<f64> {
    Ok(shape_from_r(&vertices, &faces, length, width)?.area())
}

/// @noRd
#[extendr]
fn core_leaf_shape_scaled(vertices: Robj, faces: Robj, length: f64, width: f64, area: f64) -> Result<List> {
    Ok(shape_to_r(&shape_from_r(&vertices, &faces, length, width)?.scaled_to(area).map_err(err)?))
}

/// @noRd
#[extendr]
fn core_leaf_shape_from_mesh(vertices: Robj, faces: Robj, length: Robj, width: Robj, normalise: bool) -> Result<List> {
    let s = LeafShape::from_mesh(xyz_from_r(&vertices)?, faces_from_r(&faces)?, optional_f64(&length, "length")?, optional_f64(&width, "width")?, normalise).map_err(err)?;
    Ok(shape_to_r(&s))
}

/// @noRd
#[extendr]
fn core_leaf_shape_from_obj(path: &str, length: Robj, width: Robj, normalise: bool) -> Result<List> {
    let s = LeafShape::from_obj(path, optional_f64(&length, "length")?, optional_f64(&width, "width")?, normalise).map_err(err)?;
    Ok(shape_to_r(&s))
}

/// @noRd
#[extendr]
fn core_single_leaf_area(length: f64, width: f64, vertices: Robj, faces: Robj) -> Result<f64> {
    let shape = if vertices.is_null() { None } else { Some(shape_from_r(&vertices, &faces, length, width)?) };
    lm::single_leaf_area(length, width, shape.as_ref()).map_err(err)
}

// ------------------------------------------------------------ insertion

/// @noRd
#[extendr]
fn core_add_leaves(grid: Robj, total_area: f64, seeds: Robj, bin_centres: &[f64], angle_density: &[f64], cylinders: Robj, vertices: Robj, faces: Robj, length: f64, width: f64, max_branch_distance: f64, jitter: f64, seed: f64) -> Result<List> {
    let grid = if grid.is_null() {
        None
    } else {
        let g: HashMap<&str, Robj> = List::try_from(&grid)?.try_into()?;
        let get = |k: &str| -> Result<Vec<f64>> { doubles(g.get(k).ok_or_else(|| Error::Other(format!("grid has no `{k}`")))?, k) };
        Some(grid_from_r(&get("origin")?, get("voxel_size")?[0], &get("dim")?, &get("density")?)?)
    };
    let seeds = xyz_from_r(&seeds)?;
    let model = crate::voxels::qsm_from_r(&cylinders)?;
    let angles = angles_from_r(bin_centres, angle_density);
    let opts = lm::AddLeaves { shape: shape_from_r(&vertices, &faces, length, width)?, max_branch_distance, jitter, seed: seed as u64 };
    let area = match &grid {
        Some(g) => lm::LeafArea::Grid(g),
        None => lm::LeafArea::Total(total_area),
    };
    let m = lm::add_leaves(area, &seeds, &angles, &model.cylinders, &opts).map_err(err)?;
    Ok(list!(
        vertices = xyz_to_r(&m.vertices),
        faces = faces_to_r(&m.faces),
        centres = xyz_to_r(&m.centres),
        normals = xyz_to_r(&m.normals),
        inclination = m.inclination,
        cylinder = m.cylinder.iter().map(|&c| c as f64).collect::<Vec<f64>>(),
        leaf_area = m.leaf_area
    ))
}

// ------------------------------------------------------------ OBJ

/// @noRd
#[extendr]
fn core_write_obj(path: &str, meshes: List, names: Vec<String>) -> Result<()> {
    if names.len() != meshes.len() {
        return fail("one name per mesh");
    }
    let mut parts = Vec::new();
    for (_, m) in meshes.iter() {
        let l: HashMap<&str, Robj> = List::try_from(&m)?.try_into()?;
        let v = xyz_from_r(l.get("vertices").ok_or_else(|| Error::Other("mesh has no `vertices`".into()))?)?;
        let f = faces_from_r(l.get("faces").ok_or_else(|| Error::Other("mesh has no `faces`".into()))?)?;
        parts.push((v, f));
    }
    let objs: Vec<ObjMesh> = parts.iter().zip(&names).map(|((v, f), n)| ObjMesh { name: n, vertices: v, faces: f }).collect();
    mesh_io::write_obj(path, &objs).map_err(err)
}

/// @noRd
#[extendr]
fn core_write_tree_obj(path: &str, cylinders: Robj, vertices: Robj, faces: Robj, sides: i32, contiguous: bool) -> Result<()> {
    let q = crate::voxels::qsm_from_r(&cylinders)?;
    lm::write_tree_obj(path, &q, sides.max(0) as usize, contiguous, &xyz_from_r(&vertices)?, &faces_from_r(&faces)?).map_err(err)
}

extendr_module! {
    mod leaves;
    fn core_classify_leaf_wood;
    fn core_leaf_inclinations;
    fn core_point_leaf_area;
    fn core_leaf_angle_distribution;
    fn core_leaf_de_wit;
    fn core_leaf_projection_histogram;
    fn core_leaf_area_density;
    fn core_leaf_grid_area;
    fn core_leaf_grid_total_area;
    fn core_leaf_grid_scaled;
    fn core_leaf_grid_profile;
    fn core_leaf_grid_cells;
    fn core_leaf_grid_from_voxels;
    fn core_leaf_shape;
    fn core_leaf_shape_area;
    fn core_leaf_shape_scaled;
    fn core_leaf_shape_from_mesh;
    fn core_leaf_shape_from_obj;
    fn core_single_leaf_area;
    fn core_add_leaves;
    fn core_write_obj;
    fn core_write_tree_obj;
}
