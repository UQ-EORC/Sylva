// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Offsets between two airborne surveys, estimated on stable surfaces.
//!
//! Every stable return of the second survey (ground, and optionally roads
//! or roofs) is compared with a plane fitted to the first survey's stable
//! returns around it. A second survey displaced by `(dx, dy, dz)` sees the
//! surface `s` at `(x, y)` where the first sees it at `(x - dx, y - dy)`,
//! so its height above the plane is
//!
//! ```text
//! r = dz - gx dx - gy dy
//! ```
//!
//! with `(gx, gy)` the plane's gradient: the vertical offset comes from
//! every sample, the horizontal ones from the variety of slopes and aspects
//! (terrain, roof planes), as in the co-registration of elevation models by
//! Nuth and Kääb (2011). The offsets are solved by iteratively reweighted
//! least squares (Huber 1964 weights), refitting the planes at the moved
//! position until the solution settles, one block of the area at a time.
//!
//! Where the slopes do not fix a horizontal offset (flat ground, or a slope
//! of one aspect, along which a horizontal shift looks like a vertical
//! one), a Gaussian prior of standard deviation `horizontal_prior` keeps it
//! near zero, and its uncertainty stays near the prior; a block reports
//! whether its horizontal offset was determined by the data.
//!
//! Stable returns within `correlation_length` of each other do not err
//! independently (the planes overlap, and surveys err in patches), so the
//! samples of a block weigh as much as the number of such cells they
//! occupy. The blocks are then combined into one offset for the whole area
//! (`Constant`), kept apart (`Blocks`), or smoothed into a field
//! (`Field`) by pooling the blocks' information with Gaussian weights; the
//! uncertainty of a combination is inflated by the Birge ratio of the blocks
//! about it, so that offsets that vary between blocks more than their
//! uncertainties allow widen the uncertainty rather than being averaged away.

use std::collections::BTreeMap;

use crate::als::{in_core, plan, run, workers_for_estimates, Catalog, Layout, BYTES_PER_POINT};
use crate::error::{Error, Result};
use crate::Point;

use super::{canonical, classes, est_other, fit_plane, inv3, median, read_other, Grid2};

/// How the block estimates become the offset applied at a point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Model {
    /// One offset for the whole area.
    Constant,
    /// Each block its own offset; blocks without an estimate take the
    /// constant one.
    Blocks,
    /// A smooth field: at each block centre the blocks' information pooled
    /// with Gaussian weights of standard deviation `smoothing` (m),
    /// interpolated bilinearly between centres.
    Field { smoothing: f64 },
}

impl Model {
    pub fn name(&self) -> &'static str {
        match self {
            Model::Constant => "constant",
            Model::Blocks => "blocks",
            Model::Field { .. } => "field",
        }
    }
}

/// Settings of [`align`].
#[derive(Debug, Clone)]
pub struct AlignParams {
    /// Side of the blocks (m), on a grid through a multiple of it.
    pub block_size: f64,
    /// Classes of the stable returns (2 ground; 6 buildings, 11 roads).
    pub stable_classes: Vec<u8>,
    /// One sample of the second survey per square of this side (m).
    pub sample_spacing: f64,
    /// Radius (m) of the first survey's returns fitted by a plane.
    pub radius: f64,
    /// Fewest returns in a plane.
    pub min_neighbours: usize,
    /// Planes whose residual RMS exceeds this (m) are not planar enough:
    /// breaklines, roof edges, low vegetation classified as ground.
    pub max_roughness: f64,
    /// Planes steeper than this gradient are left out.
    pub max_slope: f64,
    /// Largest offset (m) the search allows; also widens the chunks' buffer.
    pub max_offset: f64,
    /// Estimate horizontal offsets (else only the vertical one).
    pub horizontal: bool,
    /// Standard deviation (m) of the prior on each horizontal offset.
    pub horizontal_prior: f64,
    /// Samples within squares of this side (m) count as one independent sample.
    pub correlation_length: f64,
    /// Fewest usable samples in a block.
    pub min_samples: usize,
    /// Huber constant, in robust standard deviations.
    pub huber: f64,
    /// Largest number of refits.
    pub iterations: usize,
    pub model: Model,
}

impl Default for AlignParams {
    fn default() -> Self {
        AlignParams {
            block_size: 100.0,
            stable_classes: vec![2],
            sample_spacing: 1.0,
            radius: 1.5,
            min_neighbours: 6,
            max_roughness: 0.15,
            max_slope: 1.0,
            max_offset: 2.0,
            horizontal: true,
            horizontal_prior: 0.3,
            correlation_length: 5.0,
            min_samples: 30,
            huber: 1.5,
            iterations: 20,
            model: Model::Field { smoothing: 100.0 },
        }
    }
}

impl AlignParams {
    pub fn check(&self) -> Result<()> {
        let pos = |name: &str, v: f64| -> Result<()> {
            if v.is_finite() && v > 0.0 { Ok(()) } else { Err(Error::invalid(format!("{name} must be a positive number, got {v}"))) }
        };
        pos("block_size", self.block_size)?;
        pos("sample_spacing", self.sample_spacing)?;
        pos("radius", self.radius)?;
        pos("max_roughness", self.max_roughness)?;
        pos("max_slope", self.max_slope)?;
        pos("max_offset", self.max_offset)?;
        pos("horizontal_prior", self.horizontal_prior)?;
        pos("correlation_length", self.correlation_length)?;
        pos("huber", self.huber)?;
        if let Model::Field { smoothing } = self.model {
            pos("smoothing", smoothing)?;
        }
        if self.stable_classes.is_empty() {
            return Err(Error::invalid("stable_classes must name at least one class"));
        }
        if self.min_neighbours < 3 {
            return Err(Error::invalid(format!("min_neighbours must be at least 3, got {}", self.min_neighbours)));
        }
        if self.min_samples < 3 {
            return Err(Error::invalid(format!("min_samples must be at least 3, got {}", self.min_samples)));
        }
        if self.iterations == 0 {
            return Err(Error::invalid("iterations must be at least 1"));
        }
        Ok(())
    }

    /// The prior's information matrix: `1 / prior²` on x and y (a hard zero
    /// without horizontal estimation), nothing on z.
    fn prior(&self) -> [[f64; 3]; 3] {
        let p = if self.horizontal { 1.0 / (self.horizontal_prior * self.horizontal_prior) } else { 1e12 };
        [[p, 0.0, 0.0], [0.0, p, 0.0], [0.0, 0.0, 0.0]]
    }
}

/// The estimate of one block.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockFit {
    /// Samples of the second survey tried, and those with a usable plane.
    pub n_samples: usize,
    pub n_used: usize,
    /// Independent samples they count as (occupied correlation cells).
    pub n_eff: usize,
    /// `(dx, dy, dz)` of the second survey relative to the first (m), and
    /// its covariance (with the horizontal prior).
    pub offset: [f64; 3],
    pub cov: [[f64; 3]; 3],
    /// The data's information matrix and `information * offset`, without
    /// the prior: what blocks pool.
    pub info: [[f64; 3]; 3],
    pub rhs: [f64; 3],
    /// Median height of the samples above the first survey's planes before
    /// any offset (a first look at the vertical offset), and the robust
    /// standard deviation of those heights before and after the offset.
    pub median_before: f64,
    pub spread_before: f64,
    pub spread_after: f64,
    /// Whether the data, not the prior, fixed both horizontal offsets
    /// (standard deviations under half the prior's).
    pub horizontal_determined: bool,
    /// Median residual RMS of the planes fitted to each survey's stable
    /// returns: return noise plus roughness at the scale of `radius`.
    pub noise_a: f64,
    pub noise_b: f64,
    /// Refits used.
    pub iterations: usize,
}

/// Offsets between two surveys over an area, on a grid of blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct Alignment {
    pub xmin: f64,
    pub ymin: f64,
    pub block_size: f64,
    pub nx: usize,
    pub ny: usize,
    pub model: Model,
    /// Estimate of each block (row-major, row 0 at the south), None where
    /// there were too few samples; empty for an alignment rebuilt from its
    /// field alone.
    pub blocks: Vec<Option<BlockFit>>,
    /// Offset the model gives at each block centre, and its standard deviation.
    pub values: Vec<[f64; 3]>,
    pub sigmas: Vec<[f64; 3]>,
    /// The constant offset over all blocks, and its standard deviation.
    pub global: [f64; 3],
    pub global_sigma: [f64; 3],
    /// Birge ratio of the blocks about the model, per component (1 when
    /// they agree within their uncertainties; the uncertainties were
    /// multiplied by it where larger).
    pub birge: [f64; 3],
    /// Medians over the blocks of the per-survey plane residuals (m).
    pub noise_a: f64,
    pub noise_b: f64,
}

fn diag_sd(c: &[[f64; 3]; 3]) -> [f64; 3] {
    [c[0][0].max(0.0).sqrt(), c[1][1].max(0.0).sqrt(), c[2][2].max(0.0).sqrt()]
}

fn add(a: &mut [[f64; 3]; 3], b: &[[f64; 3]; 3], k: f64) {
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] += k * b[i][j];
        }
    }
}

fn mat_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

impl Alignment {
    /// An alignment from its field alone (as stored by the Python layer).
    #[allow(clippy::too_many_arguments)]
    pub fn from_field(xmin: f64, ymin: f64, block_size: f64, nx: usize, ny: usize, model: Model, values: Vec<[f64; 3]>, sigmas: Vec<[f64; 3]>, global: [f64; 3], global_sigma: [f64; 3]) -> Result<Alignment> {
        if values.len() != nx * ny || sigmas.len() != nx * ny {
            return Err(Error::invalid(format!("an alignment of {nx} x {ny} blocks needs {} offsets, got {} and {}", nx * ny, values.len(), sigmas.len())));
        }
        if !(block_size.is_finite() && block_size > 0.0) || nx == 0 || ny == 0 {
            return Err(Error::invalid("an alignment needs a positive block size and at least one block"));
        }
        if values.iter().chain(&[global]).any(|v| v.iter().any(|c| !c.is_finite())) {
            return Err(Error::invalid("alignment offsets must be finite"));
        }
        Ok(Alignment { xmin, ymin, block_size, nx, ny, model, blocks: Vec::new(), values, sigmas, global, global_sigma, birge: [1.0; 3], noise_a: f64::NAN, noise_b: f64::NAN })
    }

    /// A constant offset (for tests and for a known shift).
    pub fn constant(offset: [f64; 3], sigma: [f64; 3]) -> Alignment {
        Alignment { xmin: 0.0, ymin: 0.0, block_size: 1.0, nx: 1, ny: 1, model: Model::Constant, blocks: Vec::new(), values: vec![offset], sigmas: vec![sigma], global: offset, global_sigma: sigma, birge: [1.0; 3], noise_a: f64::NAN, noise_b: f64::NAN }
    }

    fn interp(&self, v: &[[f64; 3]], x: f64, y: f64) -> [f64; 3] {
        match self.model {
            Model::Constant => v[0],
            Model::Blocks => {
                let c = (((x - self.xmin) / self.block_size).floor().max(0.0) as usize).min(self.nx - 1);
                let r = (((y - self.ymin) / self.block_size).floor().max(0.0) as usize).min(self.ny - 1);
                v[r * self.nx + c]
            }
            Model::Field { .. } => {
                let fc = ((x - self.xmin) / self.block_size - 0.5).clamp(0.0, (self.nx - 1) as f64);
                let fr = ((y - self.ymin) / self.block_size - 0.5).clamp(0.0, (self.ny - 1) as f64);
                let (c0, r0) = (fc.floor() as usize, fr.floor() as usize);
                let (c1, r1) = ((c0 + 1).min(self.nx - 1), (r0 + 1).min(self.ny - 1));
                let (tx, ty) = (fc - c0 as f64, fr - r0 as f64);
                let g = |r: usize, c: usize| v[r * self.nx + c];
                [0, 1, 2].map(|k| g(r0, c0)[k] * (1.0 - tx) * (1.0 - ty) + g(r0, c1)[k] * tx * (1.0 - ty) + g(r1, c0)[k] * (1.0 - tx) * ty + g(r1, c1)[k] * tx * ty)
            }
        }
    }

    /// `(dx, dy, dz)` of the second survey relative to the first at `(x, y)`:
    /// a point `p` of the second survey belongs at `p - offset`.
    pub fn offset_at(&self, x: f64, y: f64) -> [f64; 3] {
        if matches!(self.model, Model::Constant) {
            return self.global;
        }
        self.interp(&self.values, x, y)
    }

    /// Standard deviation of [`Alignment::offset_at`].
    pub fn sigma_at(&self, x: f64, y: f64) -> [f64; 3] {
        if matches!(self.model, Model::Constant) {
            return self.global_sigma;
        }
        self.interp(&self.sigmas, x, y)
    }

    /// The largest horizontal offset anywhere (m).
    pub fn max_horizontal(&self) -> f64 {
        self.values.iter().chain(std::iter::once(&self.global)).map(|v| v[0].hypot(v[1])).fold(0.0, f64::max)
    }

    /// Block posterior minus the model's offset at its centre, per block
    /// (NaN where the block has no estimate).
    pub fn residuals(&self) -> Vec<[f64; 3]> {
        (0..self.nx * self.ny)
            .map(|i| match self.blocks.get(i).and_then(|b| b.as_ref()) {
                Some(b) => [0, 1, 2].map(|k| b.offset[k] - self.values[i][k]),
                None => [f64::NAN; 3],
            })
            .collect()
    }
}

/// One sample per `spacing` square: the point nearest the square's centre
/// (ties to the first in canonical order), in order of the squares.
fn samples(points: &[Point], spacing: f64) -> Vec<Point> {
    let mut best: BTreeMap<(i64, i64), (f64, Point)> = BTreeMap::new();
    for p in points {
        let key = ((p[0] / spacing).floor() as i64, (p[1] / spacing).floor() as i64);
        let (cx, cy) = ((key.0 as f64 + 0.5) * spacing, (key.1 as f64 + 0.5) * spacing);
        let d = (p[0] - cx).hypot(p[1] - cy);
        match best.get(&key) {
            Some((d0, _)) if *d0 <= d => {}
            _ => {
                best.insert(key, (d, *p));
            }
        }
    }
    best.into_values().map(|(_, p)| p).collect()
}

/// One sample's comparison: its height above the reference plane, the
/// plane's gradient and residual RMS, and its correlation cell.
struct Row {
    r: f64,
    gx: f64,
    gy: f64,
    cov_g: [[f64; 2]; 2],
    rms: f64,
    cell: (i64, i64),
}

fn rows_at(samples: &[Point], a: &[Point], ga: &Grid2, theta: &[f64; 3], p: &AlignParams) -> Vec<Row> {
    let mut out = Vec::with_capacity(samples.len());
    for s in samples {
        let (qx, qy) = (s[0] - theta[0], s[1] - theta[1]);
        let idx = ga.within(a, qx, qy, p.radius);
        if idx.len() < p.min_neighbours {
            continue;
        }
        let Some(pl) = fit_plane(a, &idx, qx, qy) else { continue };
        if pl.rms > p.max_roughness || pl.gx.hypot(pl.gy) > p.max_slope {
            continue;
        }
        let cl = p.correlation_length;
        out.push(Row { r: s[2] - theta[2] - pl.h, gx: pl.gx, gy: pl.gy, cov_g: pl.cov_g, rms: pl.rms, cell: ((qx / cl).floor() as i64, (qy / cl).floor() as i64) });
    }
    out
}

/// Robust centre and spread (1.4826 MAD) of the rows' heights.
fn centre_spread(rows: &[Row]) -> (f64, f64) {
    let mut r: Vec<f64> = rows.iter().map(|w| w.r).collect();
    let med = median(&mut r);
    let mut dev: Vec<f64> = rows.iter().map(|w| (w.r - med).abs()).collect();
    (med, 1.4826 * median(&mut dev))
}

/// Information matrix and vector of the rows, weighted by Huber and
/// scaled for spatial correlation.
///
/// The gradients are themselves estimates, and their noise alone would
/// seem to fix a horizontal offset (and bias it towards zero, as noise in a
/// regressor does). The moment matrix is therefore corrected by the
/// gradients' own covariance (the moment correction for errors in
/// variables; Fuller 1987), and directions whose remaining information is
/// within three times the sampling fluctuation of that correction are
/// given none: there the prior decides.
fn normal_equations(rows: &[Row], p: &AlignParams) -> ([[f64; 3]; 3], [f64; 3], f64, usize) {
    let (med, spread) = centre_spread(rows);
    let s = spread.max(1e-3);
    let cells: std::collections::BTreeSet<(i64, i64)> = rows.iter().map(|w| w.cell).collect();
    let n_eff = cells.len();
    let f = n_eff as f64 / rows.len() as f64;
    let mut n = [[0.0; 3]; 3];
    let mut u = [0.0; 3];
    let mut fluct = 0.0;
    for w in rows {
        let e = ((w.r - med) / s).abs();
        let wt = if e <= p.huber { 1.0 } else { p.huber / e };
        let j = [-w.gx, -w.gy, 1.0];
        for a in 0..3 {
            for b in 0..3 {
                let c = if a < 2 && b < 2 { w.cov_g[a][b] } else { 0.0 };
                n[a][b] += f * wt * (j[a] * j[b] - c) / (s * s);
            }
            u[a] += f * wt * j[a] * w.r / (s * s);
        }
        fluct += wt * wt * 2.0 * (w.cov_g[0][0].powi(2) + w.cov_g[1][1].powi(2));
    }
    let tol = 3.0 * f * fluct.sqrt() / (s * s);
    let (n, u) = clamp_psd(&n, &u, tol);
    (n, u, spread, n_eff)
}

/// A symmetric matrix with its eigenvalues below `tol` set to zero, and a
/// vector with its components along their eigenvectors removed: the data
/// say nothing in those directions, neither curvature nor pull.
fn clamp_psd(m: &[[f64; 3]; 3], u: &[f64; 3], tol: f64) -> ([[f64; 3]; 3], [f64; 3]) {
    let e = nalgebra::SymmetricEigen::new(nalgebra::Matrix3::from_fn(|i, j| 0.5 * (m[i][j] + m[j][i])));
    let mut out = [[0.0; 3]; 3];
    let mut uo = [0.0; 3];
    for k in 0..3 {
        let l = e.eigenvalues[k];
        if l <= tol {
            continue;
        }
        let v = e.eigenvectors.column(k);
        let along = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
        for i in 0..3 {
            uo[i] += along * v[i];
            for j in 0..3 {
                out[i][j] += l * v[i] * v[j];
            }
        }
    }
    (out, uo)
}

/// Estimate one block's offset from its samples (second survey, raw
/// coordinates) against the first survey's stable returns `a`; `b` are the
/// second survey's stable returns (for its own noise).
fn fit_block(samp: &[Point], a: &[Point], ga: &Grid2, b: &[Point], gb: &Grid2, p: &AlignParams) -> Option<BlockFit> {
    let prior = p.prior();
    let mut theta = [0.0; 3];
    let mut first: Option<(f64, f64)> = None;
    let mut iterations = 0;
    for it in 0..p.iterations {
        iterations = it + 1;
        let rows = rows_at(samp, a, ga, &theta, p);
        if rows.len() < p.min_samples {
            return None;
        }
        if first.is_none() {
            first = Some(centre_spread(&rows));
        }
        let (n, u, _, _) = normal_equations(&rows, p);
        let mut m = n;
        add(&mut m, &prior, 1.0);
        let pt = mat_vec(&prior, &theta);
        let rhs = [0, 1, 2].map(|k| u[k] - pt[k]);
        let inv = inv3(&m)?;
        let d = mat_vec(&inv, &rhs);
        for k in 0..3 {
            theta[k] += d[k];
        }
        if theta.iter().any(|v| !v.is_finite()) || theta[0].hypot(theta[1]) > p.max_offset || theta[2].abs() > p.max_offset {
            return None;
        }
        if d.iter().all(|v| v.abs() < 1e-4) {
            break;
        }
    }
    // The information of the data at the solution.
    let rows = rows_at(samp, a, ga, &theta, p);
    if rows.len() < p.min_samples {
        return None;
    }
    let (n, u, spread_after, n_eff) = normal_equations(&rows, p);
    let mut m = n;
    add(&mut m, &prior, 1.0);
    let cov = inv3(&m)?;
    let pt = mat_vec(&prior, &theta);
    let d = mat_vec(&cov, &[0, 1, 2].map(|k| u[k] - pt[k]));
    let offset = [0, 1, 2].map(|k| theta[k] + d[k]);
    let nt = mat_vec(&n, &theta);
    let rhs = [0, 1, 2].map(|k| nt[k] + u[k]);
    let sd = diag_sd(&cov);
    let determined = p.horizontal && sd[0] < 0.5 * p.horizontal_prior && sd[1] < 0.5 * p.horizontal_prior;
    let mut rms_a: Vec<f64> = rows.iter().map(|w| w.rms).collect();
    let mut rms_b: Vec<f64> = samp
        .iter()
        .filter_map(|s| {
            let idx = gb.within(b, s[0], s[1], p.radius);
            if idx.len() < p.min_neighbours {
                return None;
            }
            fit_plane(b, &idx, s[0], s[1]).filter(|pl| pl.rms <= p.max_roughness && pl.gx.hypot(pl.gy) <= p.max_slope).map(|pl| pl.rms)
        })
        .collect();
    let (median_before, spread_before) = first.unwrap_or((f64::NAN, f64::NAN));
    Some(BlockFit {
        n_samples: samp.len(),
        n_used: rows.len(),
        n_eff,
        offset,
        cov,
        info: n,
        rhs,
        median_before,
        spread_before,
        spread_after,
        horizontal_determined: determined,
        noise_a: median(&mut rms_a),
        noise_b: median(&mut rms_b),
        iterations,
    })
}

/// Pool blocks `(index, weight)`: `(A^-1 sum k h, A^-1 (sum k² N + P) A^-1)`
/// with `A = sum k N + P`.
fn pool(blocks: &[Option<BlockFit>], members: &[(usize, f64)], prior: &[[f64; 3]; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
    let mut a = *prior;
    let mut mid = *prior;
    let mut h = [0.0; 3];
    let mut wsum = 0.0;
    for &(i, k) in members {
        if let Some(b) = &blocks[i] {
            add(&mut a, &b.info, k);
            add(&mut mid, &b.info, k * k);
            for (hc, rc) in h.iter_mut().zip(&b.rhs) {
                *hc += k * rc;
            }
            wsum += k;
        }
    }
    if wsum < 1e-6 {
        return None;
    }
    let inv = inv3(&a)?;
    Some((mat_vec(&inv, &h), mat_mul(&mat_mul(&inv, &mid), &inv)))
}

/// Birge ratio per component of the fitted blocks about `values`, over the
/// blocks whose data fixed that component (standard deviation under half
/// the prior's, for the horizontal ones).
fn birge(blocks: &[Option<BlockFit>], values: &[[f64; 3]], prior: f64) -> [f64; 3] {
    let mut out = [1.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let (mut chi, mut n) = (0.0, 0usize);
        for (i, b) in blocks.iter().enumerate() {
            let Some(b) = b else { continue };
            if k < 2 && b.cov[k][k].max(0.0).sqrt() >= 0.5 * prior {
                continue;
            }
            let var = b.cov[k][k];
            if var > 0.0 {
                chi += (b.offset[k] - values[i][k]).powi(2) / var;
                n += 1;
            }
        }
        if n >= 2 {
            *o = (chi / (n - 1) as f64).sqrt().max(1.0);
        }
    }
    out
}

fn scaled(c: &[[f64; 3]; 3], r: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut o = *c;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] *= r[i] * r[j];
        }
    }
    o
}

/// Combine block estimates into an [`Alignment`].
pub fn combine(xmin: f64, ymin: f64, nx: usize, ny: usize, blocks: Vec<Option<BlockFit>>, p: &AlignParams) -> Result<Alignment> {
    let prior = p.prior();
    let fitted: Vec<(usize, f64)> = (0..blocks.len()).filter(|&i| blocks[i].is_some()).map(|i| (i, 1.0)).collect();
    if fitted.is_empty() {
        return Err(Error::invalid(format!(
            "no block of {} m had {} usable samples of the stable classes {:?} in both surveys; check the classes, or use larger blocks",
            p.block_size, p.min_samples, p.stable_classes
        )));
    }
    let (global, gcov) = pool(&blocks, &fitted, &prior).ok_or_else(|| Error::invalid("the stable samples do not determine the vertical offset"))?;
    let gb = birge(&blocks, &vec![global; blocks.len()], p.horizontal_prior);
    let gcov = scaled(&gcov, &gb);
    let global_sigma = diag_sd(&gcov);
    let n = nx * ny;
    let (values, sigmas, bratio) = match p.model {
        Model::Constant => (vec![global; n], vec![global_sigma; n], gb),
        Model::Blocks => {
            let v: Vec<[f64; 3]> = (0..n).map(|i| blocks[i].as_ref().map_or(global, |b| b.offset)).collect();
            let s: Vec<[f64; 3]> = (0..n).map(|i| blocks[i].as_ref().map_or(global_sigma, |b| diag_sd(&b.cov))).collect();
            (v, s, [1.0; 3])
        }
        Model::Field { smoothing } => {
            let reach = (3.0 * smoothing / p.block_size).ceil() as i64;
            let mut v = vec![global; n];
            let mut c = vec![gcov; n];
            for r in 0..ny as i64 {
                for col in 0..nx as i64 {
                    let mut members = Vec::new();
                    for dr in -reach..=reach {
                        for dc in -reach..=reach {
                            let (rr, cc) = (r + dr, col + dc);
                            if rr < 0 || cc < 0 || rr >= ny as i64 || cc >= nx as i64 {
                                continue;
                            }
                            let d2 = ((dr * dr + dc * dc) as f64) * p.block_size * p.block_size;
                            members.push((rr as usize * nx + cc as usize, (-0.5 * d2 / (smoothing * smoothing)).exp()));
                        }
                    }
                    if let Some((val, cov)) = pool(&blocks, &members, &prior) {
                        // A centre whose neighbours barely determine dz takes the global offset.
                        if cov[2][2].is_finite() && cov[2][2] < 1e6 {
                            let i = r as usize * nx + col as usize;
                            v[i] = val;
                            c[i] = cov;
                        }
                    }
                }
            }
            let br = birge(&blocks, &v, p.horizontal_prior);
            (v, c.iter().map(|m| diag_sd(&scaled(m, &br))).collect(), br)
        }
    };
    let mut na: Vec<f64> = blocks.iter().flatten().map(|b| b.noise_a).filter(|v| v.is_finite()).collect();
    let mut nb: Vec<f64> = blocks.iter().flatten().map(|b| b.noise_b).filter(|v| v.is_finite()).collect();
    Ok(Alignment { xmin, ymin, block_size: p.block_size, nx, ny, model: p.model, blocks, values, sigmas, global, global_sigma, birge: bratio, noise_a: median(&mut na), noise_b: median(&mut nb) })
}

/// The block grid of a catalogue: south-west corner at its minimum snapped
/// down to a multiple of `size`.
pub fn block_grid(cat: &Catalog, size: f64) -> Result<(f64, f64, usize, usize)> {
    let b = cat.xy_bounds().ok_or_else(|| Error::invalid("the catalogue has no tiles"))?;
    let xmin = (b[0] / size).floor() * size;
    let ymin = (b[1] / size).floor() * size;
    let nx = ((b[2] - xmin) / size).floor() as usize + 1;
    let ny = ((b[3] - ymin) / size).floor() as usize + 1;
    crate::util::limits::check_cells(nx as u128 * ny as u128, 400, &format!("{nx} x {ny} alignment blocks"), "a larger block_size")?;
    Ok((xmin, ymin, nx, ny))
}

/// Estimate the offsets of survey `b` relative to survey `a` over the whole
/// catalogue of `a`; see the module documentation. The chunks are squares of
/// whole blocks (`blocks_per_chunk` on a side), so each block's estimate
/// uses exactly the returns in it and within `radius + max_offset` of it,
/// whatever the chunks and workers.
pub fn align(cat_a: &Catalog, cat_b: &Catalog, p: &AlignParams, blocks_per_chunk: usize, workers: usize) -> Result<Alignment> {
    p.check()?;
    cat_b.check_usable()?;
    let (xmin, ymin, nx, ny) = block_grid(cat_a, p.block_size)?;
    let k = blocks_per_chunk.max(1);
    let size = k as f64 * p.block_size;
    let buffer = p.radius + p.max_offset + 1e-6;
    let chunks = plan(cat_a, Layout::Grid { size, origin: Some((xmin, ymin)) }, buffer)?;
    let est: Vec<u64> = chunks.iter().map(|c| c.est_points + est_other(cat_b, c, 0.0)).collect();
    let w = workers_for_estimates(&est, workers, BYTES_PER_POINT)?;
    let stable = |c: u8| p.stable_classes.contains(&c);
    let parts = run(cat_a, &chunks, w, "aligning surveys", |chunk, data| {
        let cls_a = classes(&data.cloud);
        if data.cloud.attr("classification").is_none() {
            return Err(Error::invalid("the first survey has no 'classification'; classify ground first (als.classify_ground)"));
        }
        let a = canonical((0..data.cloud.len()).filter(|&i| stable(cls_a[i])).map(|i| data.cloud.xyz[i]).collect());
        let (cb, _) = read_other(cat_b, chunk, p.radius, None)?;
        if !cb.is_empty() && cb.attr("classification").is_none() {
            return Err(Error::invalid("the second survey has no 'classification'; classify ground first (als.classify_ground)"));
        }
        let cls_b = classes(&cb);
        let b = canonical((0..cb.len()).filter(|&i| stable(cls_b[i])).map(|i| cb.xyz[i]).collect());
        let (ga, gb) = (Grid2::new(&a, p.radius), Grid2::new(&b, p.radius));
        let mut out = Vec::new();
        let c0 = ((chunk.core[0] - xmin) / p.block_size).round() as usize;
        let r0 = ((chunk.core[1] - ymin) / p.block_size).round() as usize;
        for r in r0..(r0 + k).min(ny) {
            for c in c0..(c0 + k).min(nx) {
                let core = [xmin + c as f64 * p.block_size, ymin + r as f64 * p.block_size, xmin + (c + 1) as f64 * p.block_size, ymin + (r + 1) as f64 * p.block_size];
                let own: Vec<Point> = b.iter().filter(|q| in_core(&core, q[0], q[1])).copied().collect();
                let samp = samples(&own, p.sample_spacing);
                out.push((r * nx + c, fit_block(&samp, &a, &ga, &b, &gb, p)));
            }
        }
        Ok(out)
    })?;
    let mut blocks: Vec<Option<BlockFit>> = vec![None; nx * ny];
    for (i, f) in parts.into_iter().flatten().flatten() {
        blocks[i] = f;
    }
    combine(xmin, ymin, nx, ny, blocks, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stable points of a surface with slopes of every aspect, and the same
    /// surface displaced.
    fn surface(n: usize, shift: [f64; 3], seed: u64) -> Vec<Point> {
        let mut out = Vec::new();
        let mut s = seed;
        let mut rnd = || {
            s = crate::change::als::mix(s);
            crate::change::als::unit(s)
        };
        let f = |x: f64, y: f64| 0.3 * (x / 7.0).sin() + 0.4 * (y / 5.0).cos() + 0.02 * x;
        for _ in 0..n {
            let (x, y) = (rnd() * 60.0, rnd() * 60.0);
            let noise = (rnd() - 0.5) * 0.02;
            out.push([x + shift[0], y + shift[1], f(x, y) + shift[2] + noise]);
        }
        out
    }

    #[test]
    fn a_known_shift_is_recovered_with_its_uncertainty() {
        let p = AlignParams { block_size: 60.0, ..Default::default() };
        let truth = [0.4, -0.25, 0.15];
        let a = canonical(surface(60_000, [0.0; 3], 1));
        let b = canonical(surface(30_000, truth, 2));
        let (ga, gb) = (Grid2::new(&a, p.radius), Grid2::new(&b, p.radius));
        let own: Vec<Point> = b.iter().filter(|q| q[0] > 5.0 && q[0] < 55.0 && q[1] > 5.0 && q[1] < 55.0).copied().collect();
        let samp = samples(&own, p.sample_spacing);
        let fit = fit_block(&samp, &a, &ga, &b, &gb, &p).expect("a fit");
        let sd = diag_sd(&fit.cov);
        for k in 0..3 {
            assert!((fit.offset[k] - truth[k]).abs() < 4.0 * sd[k] + 0.01, "component {k}: {} vs {} (sd {})", fit.offset[k], truth[k], sd[k]);
        }
        assert!(fit.horizontal_determined);
        assert!(sd[2] < 0.01, "{sd:?}");
        assert!(fit.spread_after < fit.spread_before);
        // Pooled over one block, the constant model is that block.
        let al = combine(0.0, 0.0, 1, 1, vec![Some(fit.clone())], &AlignParams { model: Model::Constant, ..p.clone() }).unwrap();
        for k in 0..3 {
            assert!((al.global[k] - fit.offset[k]).abs() < 1e-3);
        }
        assert_eq!(al.offset_at(10.0, 10.0), al.global);
    }

    #[test]
    fn a_uniform_slope_leaves_the_offset_along_it_to_the_prior() {
        // Along a slope of one aspect a horizontal shift looks like a
        // vertical one: the noisy gradients must not pretend otherwise.
        let p = AlignParams { block_size: 60.0, min_samples: 10, ..Default::default() };
        let tilted = |shift: [f64; 3], seed: u64| -> Vec<Point> {
            surface(40_000, [0.0; 3], seed).into_iter().map(|q| {
                let noise = q[2] - (0.3 * (q[0] / 7.0).sin() + 0.4 * (q[1] / 5.0).cos() + 0.02 * q[0]);
                [q[0] + shift[0], q[1] + shift[1], 1.0 + 0.05 * q[0] + shift[2] + noise]
            }).collect()
        };
        let a = canonical(tilted([0.0; 3], 3));
        let b = canonical(tilted([0.5, 0.0, 0.2], 4));
        let (ga, gb) = (Grid2::new(&a, p.radius), Grid2::new(&b, p.radius));
        let own: Vec<Point> = b.iter().filter(|q| q[0] > 5.0 && q[0] < 55.0 && q[1] > 5.0 && q[1] < 55.0).copied().collect();
        let fit = fit_block(&samples(&own, p.sample_spacing), &a, &ga, &b, &gb, &p).expect("a fit");
        let sd = diag_sd(&fit.cov);
        assert!(!fit.horizontal_determined, "{sd:?}");
        assert!(sd[0] > 0.8 * p.horizontal_prior, "{sd:?}");
        // dz - 0.05 dx is what the data fix; dx itself stays near the prior's 0.
        assert!((fit.offset[2] - 0.05 * fit.offset[0] - (0.2 - 0.05 * 0.5)).abs() < 0.005, "{:?}", fit.offset);
        assert!((fit.offset[2] - 0.2).abs() < 3.0 * sd[2], "{:?} {sd:?}", fit.offset);
        // A flat surface fixes neither horizontal offset.
        let flat = |shift: [f64; 3], seed: u64| -> Vec<Point> { surface(20_000, [shift[0], shift[1], 0.0], seed).into_iter().map(|q| [q[0], q[1], 1.0 + shift[2]]).collect() };
        let a = canonical(flat([0.0; 3], 3));
        let b = canonical(flat([0.5, 0.0, 0.2], 4));
        let (ga, gb) = (Grid2::new(&a, p.radius), Grid2::new(&b, p.radius));
        let fit = fit_block(&samples(&b, p.sample_spacing), &a, &ga, &b, &gb, &p).expect("a fit");
        assert!(!fit.horizontal_determined);
        assert!(fit.offset[0].abs() < 0.05, "{:?}", fit.offset);
        assert!((fit.offset[2] - 0.2).abs() < 1e-3);
    }

    #[test]
    fn the_field_interpolates_between_blocks() {
        let fit = |dz: f64| BlockFit { n_samples: 100, n_used: 100, n_eff: 50, offset: [0.0, 0.0, dz], cov: [[0.09, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 1e-4]], info: [[0.0; 3], [0.0; 3], [0.0, 0.0, 1e4]], rhs: [0.0, 0.0, 1e4 * dz], median_before: dz, spread_before: 0.1, spread_after: 0.05, horizontal_determined: false, noise_a: 0.02, noise_b: 0.03, iterations: 2 };
        let blocks = vec![Some(fit(0.0)), None, Some(fit(0.2))];
        let p = AlignParams { block_size: 10.0, model: Model::Field { smoothing: 1.0 }, ..Default::default() };
        let al = combine(0.0, 0.0, 3, 1, blocks.clone(), &p).unwrap();
        assert!((al.values[0][2]).abs() < 1e-6 && (al.values[2][2] - 0.2).abs() < 1e-6);
        assert!((al.values[1][2] - 0.1).abs() < 1e-6, "{:?}", al.values);
        assert!((al.offset_at(10.0, 5.0)[2] - 0.05).abs() < 1e-6);
        let c = combine(0.0, 0.0, 3, 1, blocks.clone(), &AlignParams { model: Model::Constant, ..p.clone() }).unwrap();
        assert!((c.global[2] - 0.1).abs() < 1e-9);
        // The two blocks disagree by 20 sigma: the constant offset's
        // uncertainty is widened by the Birge ratio.
        assert!(c.birge[2] > 10.0 && c.global_sigma[2] > 0.05, "{:?} {:?}", c.birge, c.global_sigma);
        let b = combine(0.0, 0.0, 3, 1, blocks, &AlignParams { model: Model::Blocks, ..p }).unwrap();
        assert_eq!(b.offset_at(25.0, 5.0)[2], 0.2);
        assert_eq!(b.offset_at(15.0, 5.0)[2], c.global[2]);
        assert!(combine(0.0, 0.0, 1, 1, vec![None], &AlignParams::default()).is_err());
    }
}
