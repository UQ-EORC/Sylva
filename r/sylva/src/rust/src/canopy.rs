// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for gap profiles, fired pulses, ground planes and density grids.
//! Multi-dimensional arrays cross as flat row-major vectors; `R/canopy.R`
//! gives them R's dimensions.

use std::collections::HashMap;

use extendr_api::prelude::*;
use sylva_rs::canopy as cn;
use sylva_rs::canopy_profile as cp;

use crate::convert::{doubles, err, fail, matrix_from_rows, raster_from_r, shots_from_r, xyz_from_r, Result};

fn opt_f64(v: &Robj) -> Result<Option<f64>> {
    if v.is_null() {
        Ok(None)
    } else {
        Ok(Some(doubles(v, "value")?[0]))
    }
}

fn pattern_from_r(p: &List) -> Result<cp::ScanPattern> {
    let m: HashMap<&str, Robj> = p.clone().try_into()?;
    let get = |k: &str| -> Result<f64> {
        let v = m.get(k).ok_or_else(|| Error::Other(format!("pattern has no `{k}`")))?;
        Ok(doubles(v, k)?[0])
    };
    Ok(cp::ScanPattern {
        theta_start: get("theta_start")?,
        theta_delta: get("theta_delta")?,
        theta_count: get("theta_count")? as usize,
        phi_count: get("phi_count")? as usize,
    })
}

fn zenith(direction: &Robj) -> Result<Vec<f64>> {
    Ok(cp::zenith_deg(&xyz_from_r(direction)?))
}

fn pair(v: &[f64], what: &str) -> Result<(f64, f64)> {
    if v.len() != 2 {
        return fail(format!("{what} must have two values"));
    }
    Ok((v[0], v[1]))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_pgap_histogram(shots: List, echo_heights: &[f64], zenith_edges: &[f64], n_azimuth: i32, height_bin: f64, n_heights: i32, min_height: f64, fired_per_ring: Robj) -> Result<List> {
    let s = shots_from_r(&shots)?;
    if echo_heights.len() != s.echo_range.len() {
        return fail("echo_heights must have one value per echo");
    }
    if zenith_edges.len() < 2 || zenith_edges.windows(2).any(|w| !(w[1] > w[0])) {
        return fail("zenith_edges must be increasing");
    }
    let fired = if fired_per_ring.is_null() { None } else { Some(doubles(&fired_per_ring, "fired_per_ring")?) };
    let mut hist = cn::PgapHistogram::new(zenith_edges.to_vec(), n_azimuth as usize, height_bin, n_heights as usize);
    hist.add(&s, echo_heights, min_height, fired.as_deref());
    Ok(list!(hits = hist.hits, shots = hist.shots))
}

/// @noRd
#[extendr]
fn core_pulses_per_line(direction: Robj, pattern: List, quantile: f64, shot_stride: i32) -> Result<f64> {
    Ok(cp::pulses_per_line(&zenith(&direction)?, &pattern_from_r(&pattern)?, quantile, shot_stride.max(1) as usize) as f64)
}

/// @noRd
#[extendr]
fn core_expected_per_zenith(pattern: List, zenith_edges: &[f64], pulses_per_line: Robj) -> Result<Vec<f64>> {
    let ppl = opt_f64(&pulses_per_line)?.map(|v| v as usize);
    Ok(cp::expected_per_zenith(&pattern_from_r(&pattern)?, zenith_edges, ppl))
}

/// @noRd
#[extendr]
fn core_fired_pulses_per_ring(direction: Robj, pattern: List, zenith_edges: &[f64], shot_stride: i32, ground_zenith: &[f64]) -> Result<Vec<f64>> {
    Ok(cp::fired_pulses_per_ring(&zenith(&direction)?, &pattern_from_r(&pattern)?, zenith_edges, shot_stride.max(1) as usize, pair(ground_zenith, "ground_zenith")?))
}

/// @noRd
#[extendr]
fn core_fired_pulses_from_points(direction: Robj, zenith_edges: &[f64], ground_zenith: &[f64], limit_quantile: f64, field_of_view: Robj) -> Result<Vec<f64>> {
    cp::fired_pulses_from_points(&zenith(&direction)?, zenith_edges, pair(ground_zenith, "ground_zenith")?, limit_quantile, opt_f64(&field_of_view)?).map_err(err)
}

/// @noRd
#[extendr]
fn core_gap_fraction_pattern(shots: List, echo_heights: &[f64], pattern: List, min_height: f64, zenith_edges: &[f64], pulses_per_line: f64) -> Result<List> {
    let s = shots_from_r(&shots)?;
    let (c, g) = cp::gap_fraction_pattern(&s, echo_heights, &pattern_from_r(&pattern)?, min_height, zenith_edges, pulses_per_line as usize).map_err(err)?;
    Ok(list!(centres = c, gap = g))
}

/// @noRd
#[extendr]
fn core_gap_fraction_zenith(shots: List, echo_heights: &[f64], min_height: f64, zenith_edges: &[f64]) -> Result<List> {
    let s = shots_from_r(&shots)?;
    if echo_heights.len() != s.echo_range.len() {
        return fail("echo_heights must have one value per echo");
    }
    let (c, g) = cn::gap_fraction_zenith(&s, echo_heights, min_height, zenith_edges);
    Ok(list!(centres = c, gap = g))
}

/// @noRd
#[extendr]
fn core_fit_ground_plane(points: Robj, cell: f64, centre: Robj, radius: Robj, iterations: i32) -> Result<Vec<f64>> {
    let p = xyz_from_r(&points)?;
    let c = if centre.is_null() { None } else { Some(pair(&doubles(&centre, "centre")?, "centre")?) };
    let r = opt_f64(&radius)?;
    let (c, r) = if c.is_some() && r.is_some() { (c, r) } else { (None, None) };
    Ok(cp::fit_ground_plane(&p, cell, c, r, iterations.max(0) as usize).map_err(err)?.to_vec())
}

/// @noRd
#[extendr]
fn core_vertical_profile(heights: &[f64], bin_size: f64, max_height: Robj) -> Result<List> {
    let (b, c) = cp::vertical_profile(heights, bin_size, opt_f64(&max_height)?);
    Ok(list!(bins = b, counts = c.into_iter().map(|v| v as f64).collect::<Vec<f64>>()))
}

// ------------------------------------------------------------- gap profiles

struct Gap {
    edges: Vec<f64>,
    na: usize,
    nh: usize,
    height_bin: f64,
    min_height: f64,
    hits: Vec<f64>,
    shots: Vec<f64>,
}

impl Gap {
    fn arrays(&self) -> cp::GapArrays<'_> {
        cp::GapArrays { zenith_edges: &self.edges, n_azimuth: self.na, height_bin: self.height_bin, n_heights: self.nh, hits: &self.hits, shots: &self.shots, min_height: self.min_height }
    }
}

fn gap(zenith_edges: &[f64], n_azimuth: i32, height_bin: f64, n_heights: i32, min_height: f64, hits: &[f64], shots: &[f64]) -> Gap {
    Gap { edges: zenith_edges.to_vec(), na: n_azimuth as usize, nh: n_heights as usize, height_bin, min_height, hits: hits.to_vec(), shots: shots.to_vec() }
}

fn scan_list(l: &List, what: &str) -> Result<Vec<Vec<f64>>> {
    l.values().map(|v| doubles(&v, what)).collect()
}

fn refs(v: &[Vec<f64>]) -> Vec<&[f64]> {
    v.iter().map(|x| x.as_slice()).collect()
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_gap_pgap(zenith_edges: &[f64], n_azimuth: i32, height_bin: f64, n_heights: i32, min_height: f64, hits: &[f64], shots: &[f64]) -> Result<Robj> {
    let g = gap(zenith_edges, n_azimuth, height_bin, n_heights, min_height, hits, shots);
    let p = g.arrays().pgap().map_err(err)?;
    Ok(matrix_from_rows(g.edges.len() - 1, g.nh, &p))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_gap_pai_profile(zenith_edges: &[f64], n_azimuth: i32, height_bin: f64, n_heights: i32, min_height: f64, hits: &[f64], shots: &[f64], method: &str, derivative: bool) -> Result<Vec<f64>> {
    let g = gap(zenith_edges, n_azimuth, height_bin, n_heights, min_height, hits, shots);
    let a = g.arrays();
    if derivative { a.pavd_profile(method) } else { a.pai_profile(method) }.map_err(err)
}

/// @noRd
#[extendr]
fn core_gap_clumping(zenith_edges: &[f64], n_azimuth: i32, scan_hits: List, scan_shots: List, scan_low: List, zenith: f64) -> Result<f64> {
    let (h, s, l) = (scan_list(&scan_hits, "scan_hits")?, scan_list(&scan_shots, "scan_shots")?, scan_list(&scan_low, "scan_low")?);
    Ok(cp::clumping(zenith_edges, n_azimuth as usize, &refs(&h), &refs(&s), &refs(&l), zenith))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_gap_report(zenith_edges: &[f64], n_azimuth: i32, height_bin: f64, n_heights: i32, min_height: f64, hits: &[f64], shots: &[f64], scan_hits: List, scan_shots: List, scan_low: List, top_fraction: f64, saturation_gap: f64) -> Result<List> {
    let g = gap(zenith_edges, n_azimuth, height_bin, n_heights, min_height, hits, shots);
    let (h, s, l) = (scan_list(&scan_hits, "scan_hits")?, scan_list(&scan_shots, "scan_shots")?, scan_list(&scan_low, "scan_low")?);
    let r = cp::gap_report(&g.arrays(), &refs(&h), &refs(&s), &refs(&l), top_fraction, saturation_gap).map_err(err)?;
    Ok(list!(
        saturated = r.saturated,
        gap_57 = r.gap_57,
        pai_hinge = r.pai_hinge,
        pai_linear = r.pai_linear,
        pai_weighted = r.pai_weighted,
        mla_linear = r.mla_linear,
        clumping = r.clumping,
        pai_hinge_corrected = r.pai_hinge_corrected,
        canopy_height = r.canopy_height,
        closure_57 = r.closure_57,
        cover = r.cover,
        cover_zenith = r.cover_zenith,
        n_scans = r.n_scans as i32,
        pulses = r.pulses,
        height = r.height,
        pai_hinge_profile = r.pai_hinge_profile,
        pavd_hinge = r.pavd_hinge,
        pai_linear_profile = r.pai_linear_profile,
        pavd_linear = r.pavd_linear
    ))
}

// ----------------------------------------------------------- density grids

/// @noRd
#[extendr]
fn core_density_grid(shots: List, voxel_size: f64, origin: Robj, shape: Robj, min_hits: i32) -> Result<List> {
    let s = shots_from_r(&shots)?;
    let g = if origin.is_null() || shape.is_null() {
        cn::density_grid_from_shots(&s, voxel_size).map_err(err)?
    } else {
        let o = doubles(&origin, "origin")?;
        let sh = doubles(&shape, "shape")?;
        if o.len() != 3 || sh.len() != 3 {
            return fail("origin and shape must have three values");
        }
        let mut g = cn::DensityGrid::new([o[0], o[1], o[2]], voxel_size, [sh[0] as usize, sh[1] as usize, sh[2] as usize]);
        g.add_shots(&s);
        g
    };
    let m = min_hits.max(0) as u32;
    Ok(list!(
        dim = vec![g.shape[2] as f64, g.shape[1] as f64, g.shape[0] as f64],
        n_rays = g.n_rays.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
        n_hits = g.n_hits.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
        path_length = g.path_length.clone(),
        density = g.density(m),
        profile = g.vertical_profile(m),
        origin = g.origin.to_vec(),
        voxel_size = g.voxel_size
    ))
}

struct Grid {
    shape: [usize; 3],
    origin: [f64; 3],
    voxel_size: f64,
    n_rays: Vec<f64>,
    n_hits: Vec<f64>,
    path_length: Vec<f64>,
    density: Vec<f64>,
}

impl Grid {
    fn from_r(g: &List) -> Result<Grid> {
        let m: HashMap<&str, Robj> = g.clone().try_into()?;
        let get = |k: &str| -> Result<Vec<f64>> { doubles(m.get(k).ok_or_else(|| Error::Other(format!("grid has no `{k}`")))?, k) };
        let dim = get("dim")?;
        let origin = get("origin")?;
        let n = (dim[0] * dim[1] * dim[2]) as usize;
        let grid = Grid {
            shape: [dim[0] as usize, dim[1] as usize, dim[2] as usize],
            origin: [origin[0], origin[1], origin[2]],
            voxel_size: get("voxel_size")?[0],
            n_rays: get("n_rays")?,
            n_hits: get("n_hits")?,
            path_length: get("path_length")?,
            density: get("density")?,
        };
        if [&grid.n_rays, &grid.n_hits, &grid.path_length, &grid.density].iter().any(|v| v.len() != n) {
            return fail("grid arrays do not match its dimensions");
        }
        Ok(grid)
    }

    fn arrays(&self) -> cp::GridArrays<'_> {
        cp::GridArrays { shape: self.shape, origin: self.origin, voxel_size: self.voxel_size, n_rays: &self.n_rays, n_hits: &self.n_hits, path_length: &self.path_length, density: &self.density }
    }
}

/// @noRd
#[extendr]
fn core_grid_height_above(grid: List, dtm: Robj, xmin: f64, ymin: f64, resolution: f64) -> Result<Vec<f64>> {
    Ok(Grid::from_r(&grid)?.arrays().height_above(&raster_from_r(&dtm, xmin, ymin, resolution)?))
}

/// @noRd
#[extendr]
fn core_grid_mask_ground(grid: List, dtm: Robj, xmin: f64, ymin: f64, resolution: f64, margin: f64) -> Result<List> {
    let (d, p) = Grid::from_r(&grid)?.arrays().mask_ground(&raster_from_r(&dtm, xmin, ymin, resolution)?, margin);
    Ok(list!(density = d, profile = p))
}

/// @noRd
#[extendr]
#[allow(clippy::too_many_arguments)]
fn core_grid_profile_above_ground(grid: List, dtm: Robj, xmin: f64, ymin: f64, resolution: f64, bin_size: f64, margin: f64, pooled: bool, max_height: Robj) -> Result<List> {
    let (b, p) = Grid::from_r(&grid)?.arrays().profile_above_ground(&raster_from_r(&dtm, xmin, ymin, resolution)?, bin_size, opt_f64(&max_height)?, margin, pooled);
    Ok(list!(bins = b, pad = p))
}

extendr_module! {
    mod canopy;
    fn core_pgap_histogram;
    fn core_pulses_per_line;
    fn core_expected_per_zenith;
    fn core_fired_pulses_per_ring;
    fn core_fired_pulses_from_points;
    fn core_gap_fraction_pattern;
    fn core_gap_fraction_zenith;
    fn core_fit_ground_plane;
    fn core_vertical_profile;
    fn core_gap_pgap;
    fn core_gap_pai_profile;
    fn core_gap_clumping;
    fn core_gap_report;
    fn core_density_grid;
    fn core_grid_height_above;
    fn core_grid_mask_ground;
    fn core_grid_profile_above_ground;
}
