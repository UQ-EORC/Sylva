// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Bindings for sylva_rs::canopy_profile: gap profiles, fired pulses,
//! ground planes and density-grid profiles.
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sylva_rs::canopy_profile as cp;

use crate::{err, raster_from_py, shots_from_py, xyz_from_py};

pub(crate) fn pattern_from_py(d: &Bound<'_, PyDict>) -> PyResult<cp::ScanPattern> {
    let get = |k: &str| d.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("pattern has no {k:?}")));
    Ok(cp::ScanPattern {
        theta_start: get("theta_start")?.extract()?,
        theta_delta: get("theta_delta")?.extract()?,
        theta_count: get("theta_count")?.extract::<f64>()? as usize,
        phi_count: get("phi_count")?.extract::<f64>()? as usize,
    })
}

fn zenith_of(direction: PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
    Ok(cp::zenith_deg(&xyz_from_py(direction)?))
}

fn vec1<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    v.into_pyarray(py)
}

#[pyfunction]
fn canopy_pulses_per_line(direction: PyReadonlyArray2<f64>, pattern: &Bound<'_, PyDict>, quantile: f64, shot_stride: usize) -> PyResult<usize> {
    Ok(cp::pulses_per_line(&zenith_of(direction)?, &pattern_from_py(pattern)?, quantile, shot_stride))
}

#[pyfunction]
#[pyo3(signature = (pattern, zenith_edges, pulses_per_line=None))]
fn canopy_expected_per_zenith<'py>(py: Python<'py>, pattern: &Bound<'_, PyDict>, zenith_edges: Vec<f64>, pulses_per_line: Option<usize>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(vec1(py, cp::expected_per_zenith(&pattern_from_py(pattern)?, &zenith_edges, pulses_per_line)))
}

#[pyfunction]
fn canopy_fired_pulses_per_ring<'py>(py: Python<'py>, direction: PyReadonlyArray2<f64>, pattern: &Bound<'_, PyDict>, zenith_edges: Vec<f64>, shot_stride: usize, ground_zenith: (f64, f64)) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let zen = zenith_of(direction)?;
    Ok(vec1(py, cp::fired_pulses_per_ring(&zen, &pattern_from_py(pattern)?, &zenith_edges, shot_stride, ground_zenith)))
}

#[pyfunction]
#[pyo3(signature = (direction, zenith_edges, ground_zenith, limit_quantile, field_of_view=None))]
fn canopy_fired_pulses_from_points<'py>(py: Python<'py>, direction: PyReadonlyArray2<f64>, zenith_edges: Vec<f64>, ground_zenith: (f64, f64), limit_quantile: f64, field_of_view: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let zen = zenith_of(direction)?;
    Ok(vec1(py, cp::fired_pulses_from_points(&zen, &zenith_edges, ground_zenith, limit_quantile, field_of_view).map_err(err)?))
}

#[pyfunction]
fn canopy_gap_fraction_pattern<'py>(py: Python<'py>, shots: &Bound<'_, PyDict>, echo_heights: PyReadonlyArray1<f64>, pattern: &Bound<'_, PyDict>, min_height: f64, zenith_edges: Vec<f64>, pulses_per_line: usize) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let s = shots_from_py(shots)?;
    let (c, g) = cp::gap_fraction_pattern(&s, &echo_heights.as_array().to_vec(), &pattern_from_py(pattern)?, min_height, &zenith_edges, pulses_per_line).map_err(err)?;
    Ok((vec1(py, c), vec1(py, g)))
}

#[pyfunction]
#[pyo3(signature = (points, cell, centre=None, radius=None, iterations=20))]
fn canopy_fit_ground_plane<'py>(py: Python<'py>, points: PyReadonlyArray2<f64>, cell: f64, centre: Option<(f64, f64)>, radius: Option<f64>, iterations: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = xyz_from_py(points)?;
    Ok(vec1(py, cp::fit_ground_plane(&p, cell, centre, radius, iterations).map_err(err)?.to_vec()))
}

#[pyfunction]
#[pyo3(signature = (heights, bin_size, max_height=None))]
fn canopy_vertical_profile<'py>(py: Python<'py>, heights: PyReadonlyArray1<f64>, bin_size: f64, max_height: Option<f64>) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>) {
    let (b, c) = cp::vertical_profile(&heights.as_array().to_vec(), bin_size, max_height);
    (vec1(py, b), c.into_iter().map(|v| v as i64).collect::<Vec<_>>().into_pyarray(py))
}

// ------------------------------------------------------------- gap profiles

/// The pooled arrays of a Python GapProfile.
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
    fn from_py(zenith_edges: Vec<f64>, height_bin: f64, min_height: f64, hits: PyReadonlyArray3<f64>, shots: PyReadonlyArray2<f64>) -> PyResult<Gap> {
        let dims = hits.shape().to_vec();
        if dims[0] + 1 != zenith_edges.len() || shots.shape() != [dims[0], dims[1]] {
            return Err(PyValueError::new_err("hits must be (rings, sectors, heights) and shots (rings, sectors)"));
        }
        Ok(Gap {
            edges: zenith_edges,
            na: dims[1],
            nh: dims[2],
            height_bin,
            min_height,
            hits: hits.as_array().iter().copied().collect(),
            shots: shots.as_array().iter().copied().collect(),
        })
    }

    fn arrays(&self) -> cp::GapArrays<'_> {
        cp::GapArrays { zenith_edges: &self.edges, n_azimuth: self.na, height_bin: self.height_bin, n_heights: self.nh, hits: &self.hits, shots: &self.shots, min_height: self.min_height }
    }
}

fn refs(v: &[Vec<f64>]) -> Vec<&[f64]> {
    v.iter().map(|x| x.as_slice()).collect()
}

fn scan_arrays(list: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f64>>> {
    list.iter().map(|a| Ok(a.extract::<PyReadonlyArray2<f64>>()?.as_array().iter().copied().collect())).collect()
}

#[pyfunction]
fn canopy_gap_pgap<'py>(py: Python<'py>, zenith_edges: Vec<f64>, height_bin: f64, min_height: f64, hits: PyReadonlyArray3<f64>, shots: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let g = Gap::from_py(zenith_edges, height_bin, min_height, hits, shots)?;
    let p = g.arrays().pgap().map_err(err)?;
    PyArray1::from_vec(py, p).reshape([g.edges.len() - 1, g.nh])
}

#[pyfunction]
fn canopy_gap_pai_profile<'py>(py: Python<'py>, zenith_edges: Vec<f64>, height_bin: f64, min_height: f64, hits: PyReadonlyArray3<f64>, shots: PyReadonlyArray2<f64>, method: &str, derivative: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = Gap::from_py(zenith_edges, height_bin, min_height, hits, shots)?;
    let a = g.arrays();
    Ok(vec1(py, if derivative { a.pavd_profile(method) } else { a.pai_profile(method) }.map_err(err)?))
}

#[pyfunction]
fn canopy_gap_clumping(zenith_edges: Vec<f64>, n_azimuth: usize, scan_hits: &Bound<'_, PyList>, scan_shots: &Bound<'_, PyList>, scan_low: &Bound<'_, PyList>, zenith: f64) -> PyResult<f64> {
    let (h, s, l) = (scan_arrays(scan_hits)?, scan_arrays(scan_shots)?, scan_arrays(scan_low)?);
    Ok(cp::clumping(&zenith_edges, n_azimuth, &refs(&h), &refs(&s), &refs(&l), zenith))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn canopy_gap_report<'py>(py: Python<'py>, zenith_edges: Vec<f64>, height_bin: f64, min_height: f64, hits: PyReadonlyArray3<f64>, shots: PyReadonlyArray2<f64>, scan_hits: &Bound<'_, PyList>, scan_shots: &Bound<'_, PyList>, scan_low: &Bound<'_, PyList>, top_fraction: f64, saturation_gap: f64) -> PyResult<Bound<'py, PyDict>> {
    let g = Gap::from_py(zenith_edges, height_bin, min_height, hits, shots)?;
    let (h, s, l) = (scan_arrays(scan_hits)?, scan_arrays(scan_shots)?, scan_arrays(scan_low)?);
    let r = cp::gap_report(&g.arrays(), &refs(&h), &refs(&s), &refs(&l), top_fraction, saturation_gap).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("saturated", r.saturated)?;
    d.set_item("gap_57", r.gap_57)?;
    d.set_item("pai_hinge", r.pai_hinge)?;
    d.set_item("pai_linear", r.pai_linear)?;
    d.set_item("pai_weighted", r.pai_weighted)?;
    d.set_item("mla_linear", r.mla_linear)?;
    d.set_item("clumping", r.clumping)?;
    d.set_item("pai_hinge_corrected", r.pai_hinge_corrected)?;
    d.set_item("canopy_height", r.canopy_height)?;
    d.set_item("closure_57", r.closure_57)?;
    d.set_item("cover", r.cover)?;
    d.set_item("cover_zenith", r.cover_zenith)?;
    d.set_item("n_scans", r.n_scans)?;
    d.set_item("pulses", r.pulses)?;
    d.set_item("height", vec1(py, r.height))?;
    d.set_item("pai_hinge_profile", vec1(py, r.pai_hinge_profile))?;
    d.set_item("pavd_hinge", vec1(py, r.pavd_hinge))?;
    d.set_item("pai_linear_profile", vec1(py, r.pai_linear_profile))?;
    d.set_item("pavd_linear", vec1(py, r.pavd_linear))?;
    Ok(d)
}

// ----------------------------------------------------------- density grids

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
    fn from_py(origin: (f64, f64, f64), voxel_size: f64, n_rays: PyReadonlyArray3<f64>, n_hits: PyReadonlyArray3<f64>, path_length: PyReadonlyArray3<f64>, density: PyReadonlyArray3<f64>) -> PyResult<Grid> {
        let s = n_rays.shape().to_vec();
        for a in [n_hits.shape(), path_length.shape(), density.shape()] {
            if a != s.as_slice() {
                return Err(PyValueError::new_err("grid arrays must share one shape"));
            }
        }
        let flat = |a: PyReadonlyArray3<f64>| a.as_array().iter().copied().collect::<Vec<f64>>();
        Ok(Grid {
            shape: [s[0], s[1], s[2]],
            origin: [origin.0, origin.1, origin.2],
            voxel_size,
            n_rays: flat(n_rays),
            n_hits: flat(n_hits),
            path_length: flat(path_length),
            density: flat(density),
        })
    }

    fn arrays(&self) -> cp::GridArrays<'_> {
        cp::GridArrays { shape: self.shape, origin: self.origin, voxel_size: self.voxel_size, n_rays: &self.n_rays, n_hits: &self.n_hits, path_length: &self.path_length, density: &self.density }
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn canopy_grid_height_above<'py>(py: Python<'py>, origin: (f64, f64, f64), voxel_size: f64, n_rays: PyReadonlyArray3<f64>, n_hits: PyReadonlyArray3<f64>, path_length: PyReadonlyArray3<f64>, density: PyReadonlyArray3<f64>, dtm: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let g = Grid::from_py(origin, voxel_size, n_rays, n_hits, path_length, density)?;
    let h = g.arrays().height_above(&raster_from_py(dtm, xmin, ymin, resolution));
    PyArray1::from_vec(py, h).reshape(g.shape)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn canopy_grid_mask_ground<'py>(py: Python<'py>, origin: (f64, f64, f64), voxel_size: f64, n_rays: PyReadonlyArray3<f64>, n_hits: PyReadonlyArray3<f64>, path_length: PyReadonlyArray3<f64>, density: PyReadonlyArray3<f64>, dtm: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64, margin: f64) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray1<f64>>)> {
    let g = Grid::from_py(origin, voxel_size, n_rays, n_hits, path_length, density)?;
    let (d, p) = g.arrays().mask_ground(&raster_from_py(dtm, xmin, ymin, resolution), margin);
    Ok((PyArray1::from_vec(py, d).reshape(g.shape)?, vec1(py, p)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (origin, voxel_size, n_rays, n_hits, path_length, density, dtm, xmin, ymin, resolution, bin_size, margin, pooled, max_height=None))]
fn canopy_grid_profile_above_ground<'py>(py: Python<'py>, origin: (f64, f64, f64), voxel_size: f64, n_rays: PyReadonlyArray3<f64>, n_hits: PyReadonlyArray3<f64>, path_length: PyReadonlyArray3<f64>, density: PyReadonlyArray3<f64>, dtm: PyReadonlyArray2<f64>, xmin: f64, ymin: f64, resolution: f64, bin_size: f64, margin: f64, pooled: bool, max_height: Option<f64>) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let g = Grid::from_py(origin, voxel_size, n_rays, n_hits, path_length, density)?;
    let (b, p) = g.arrays().profile_above_ground(&raster_from_py(dtm, xmin, ymin, resolution), bin_size, max_height, margin, pooled);
    Ok((vec1(py, b), vec1(py, p)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(canopy_pulses_per_line, m)?,
        wrap_pyfunction!(canopy_expected_per_zenith, m)?,
        wrap_pyfunction!(canopy_fired_pulses_per_ring, m)?,
        wrap_pyfunction!(canopy_fired_pulses_from_points, m)?,
        wrap_pyfunction!(canopy_gap_fraction_pattern, m)?,
        wrap_pyfunction!(canopy_fit_ground_plane, m)?,
        wrap_pyfunction!(canopy_vertical_profile, m)?,
        wrap_pyfunction!(canopy_gap_pgap, m)?,
        wrap_pyfunction!(canopy_gap_pai_profile, m)?,
        wrap_pyfunction!(canopy_gap_clumping, m)?,
        wrap_pyfunction!(canopy_gap_report, m)?,
        wrap_pyfunction!(canopy_grid_height_above, m)?,
        wrap_pyfunction!(canopy_grid_mask_ground, m)?,
        wrap_pyfunction!(canopy_grid_profile_above_ground, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
