// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for RiSCAN PRO projects, export settings and RiSCAN's import
//! filter. Matrices cross as R matrices; `R/riscan.R` builds the objects.
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use extendr_api::prelude::*;
use sylva_rs::riscan;

use crate::convert::{cloud_from_r, cloud_to_r, doubles, err, fail, matrix_from_rows, xyz_from_r, Result};

fn null() -> Robj {
    Robj::from(())
}

fn opt<T>(v: Option<T>, f: impl FnOnce(T) -> Robj) -> Robj {
    v.map(f).unwrap_or_else(null)
}

fn path_str(p: &std::path::Path) -> String {
    p.to_string_lossy().into_owned()
}

fn f32_rows(xyz: &Robj) -> Result<Vec<[f32; 3]>> {
    Ok(xyz_from_r(xyz)?.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect())
}

fn position_to_r(q: &riscan::ScanPosition) -> List {
    let pattern = opt(q.pattern, |s| {
        list!(theta_start = s.theta_start, theta_delta = s.theta_delta, theta_count = s.theta_count as f64,
              phi_start = s.phi_start, phi_delta = s.phi_delta, phi_count = s.phi_count as f64).into()
    });
    list!(
        name = q.name.as_str(),
        rxp = opt(q.rxp(), |p| path_str(p).into()),
        sop = opt(q.sop, |m| matrix_from_rows(4, 4, &m)),
        scans = q.scans.iter().map(|p| path_str(p)).collect::<Vec<_>>(),
        instrument = opt(q.instrument.as_deref(), |s| s.into()),
        pattern = pattern,
        tiepoints = opt(q.tiepoints.as_deref(), |p| path_str(p).into()),
        gnss = opt(q.gnss, |g| g.to_vec().into()),
        attitude = opt(q.attitude, |a| matrix_from_rows(3, 3, &a))
    )
}

/// @noRd
#[extendr]
fn core_riscan_read_project(path: &str) -> Result<List> {
    let p = riscan::read_project(path).map_err(err)?;
    let positions = List::from_values(p.positions.iter().map(position_to_r));
    Ok(list!(path = path_str(&p.path), name = p.name.as_str(), pop = opt(p.pop, |m| matrix_from_rows(4, 4, &m)), positions = positions))
}

/// @noRd
#[extendr]
fn core_riscan_rotation_zyx(roll: f64, pitch: f64, yaw: f64) -> Robj {
    matrix_from_rows(4, 4, &riscan::rotation_zyx(roll, pitch, yaw))
}

/// GNSS fixes as local metres; `coordinates` an `n x 3` matrix, NA rows
/// for positions without a fix.
/// @noRd
#[extendr]
fn core_riscan_gnss_to_local(coordinates: Robj) -> Result<Robj> {
    let rows = if coordinates.nrows() == 0 { vec![] } else { xyz_from_r(&coordinates)? };
    let c: Vec<Option<[f64; 3]>> = rows.into_iter().map(|r| if r.iter().any(|v| v.is_nan()) { None } else { Some(r) }).collect();
    let out = riscan::gnss_to_local(&c);
    Ok(matrix_from_rows(out.len(), 3, &out.concat()))
}

/// @noRd
#[extendr]
fn core_riscan_read_export_settings(path: &str) -> Result<List> {
    let s = riscan::read_export_settings(path).map_err(err)?;
    Ok(list!(name = s.iter().map(|x| x.0.clone()).collect::<Vec<_>>(), min = s.iter().map(|x| x.1).collect::<Vec<_>>(), max = s.iter().map(|x| x.2).collect::<Vec<_>>()))
}

/// @noRd
#[extendr]
fn core_riscan_export_settings_mask(names: Vec<String>, min: &[f64], max: &[f64], xyz: Robj, attributes: List) -> Result<Robj> {
    if names.len() != min.len() || names.len() != max.len() {
        return fail("settings need one minimum and maximum per attribute");
    }
    let settings: Vec<(String, f64, f64)> = names.into_iter().zip(min.iter().zip(max)).map(|(n, (&lo, &hi))| (n, lo, hi)).collect();
    let p = xyz_from_r(&xyz)?;
    let mut attrs = HashMap::new();
    for (name, v) in attributes.iter() {
        attrs.insert(name.to_string(), doubles(&v, name)?);
    }
    let keep = riscan::export_settings_mask(&settings, &p, |n| attrs.get(n).map(|v| v.as_slice())).map_err(err)?;
    Ok(keep.iter().map(|&k| Rbool::from(k)).collect::<Logicals>().into())
}

/// @noRd
#[extendr]
fn core_riscan_angular_steps(xyz: Robj, sample: f64) -> Result<Vec<f64>> {
    let (t, p) = riscan::angular_steps(&f32_rows(&xyz)?, sample.max(0.0) as usize).map_err(err)?;
    Ok(vec![t, p])
}

/// @noRd
#[extendr]
fn core_riscan_like_mask(xyz: Robj, amplitude: &[f64], mode: &str, min_range: f64, window_steps: f64, window_range: f64, min_neighbours: f64, weak_db: f64, steps: Robj) -> Result<Robj> {
    let steps = if steps.is_null() {
        None
    } else {
        let s = doubles(&steps, "steps")?;
        if s.len() != 2 {
            return fail("steps must be c(theta_step, phi_step)");
        }
        Some((s[0], s[1]))
    };
    let legacy = riscan::LegacyFilter { window_steps, window_range, min_neighbours: min_neighbours.max(0.0) as usize, weak_db, steps };
    let a: Vec<f32> = amplitude.iter().map(|&v| v as f32).collect();
    let keep = riscan::riscan_like_mask(&f32_rows(&xyz)?, &a, mode, min_range, &legacy).map_err(err)?;
    Ok(keep.iter().map(|&k| Rbool::from(k)).collect::<Logicals>().into())
}

/// @noRd
#[extendr]
fn core_riscan_read_tiepoints(path: &str) -> Result<List> {
    let t = riscan::read_tiepoint_list(path).map_err(err)?;
    Ok(list!(
        x = t.iter().map(|r| r.position[0]).collect::<Vec<_>>(),
        y = t.iter().map(|r| r.position[1]).collect::<Vec<_>>(),
        z = t.iter().map(|r| r.position[2]).collect::<Vec<_>>(),
        reflectance = t.iter().map(|r| r.reflectance).collect::<Vec<_>>(),
        diameter = t.iter().map(|r| r.diameter).collect::<Vec<_>>(),
        n_points = t.iter().map(|r| r.point_count as f64).collect::<Vec<_>>(),
        name = t.iter().map(|r| r.name.clone()).collect::<Vec<_>>()
    ))
}

/// A point cloud under a 4x4 matrix (row-major values).
/// @noRd
#[extendr]
fn core_riscan_transform_cloud(cloud: List, matrix: &[f64]) -> Result<List> {
    let t = sylva_rs::Transform::from_row_major(matrix).map_err(err)?;
    Ok(cloud_to_r(&cloud_from_r(&cloud)?.transformed(&t)))
}

extendr_module! {
    mod riscan;
    fn core_riscan_read_project;
    fn core_riscan_rotation_zyx;
    fn core_riscan_gnss_to_local;
    fn core_riscan_read_export_settings;
    fn core_riscan_export_settings_mask;
    fn core_riscan_angular_steps;
    fn core_riscan_like_mask;
    fn core_riscan_read_tiepoints;
    fn core_riscan_transform_cloud;
}
