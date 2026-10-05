// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Area-based metrics of airborne lidar: the standard set of lidR's
//! `stdmetrics` (Roussel et al. 2020) computed from heights above ground,
//! on a grid over a whole catalogue ([`grid_metrics`]), for plots
//! ([`plot_metrics`]) or for one cloud ([`cloud_metrics`]).
//!
//! # Definitions
//!
//! For the `n` retained returns with heights `z` (sorted ascending), mean
//! `m` and central sums `S_k = sum((z - m)^k)`:
//!
//! - `zmax`, `zmean`; `zsd = sqrt(S_2 / (n - 1))` (NaN for one return);
//!   `zskew = (S_3 / n) / (S_2 / n)^1.5` and `zkurt = n S_4 / S_2^2`, the
//!   moment estimators lidR uses (SciPy's `skew` and `kurtosis(fisher=False)`
//!   with `bias=True`; the kurtosis is not the excess).
//! - `zentropy`: lidR's `entropy(z, by)`, the Shannon index of the heights
//!   in bins of `by` m from 0 to `ceiling(zmax / by) * by`, divided by that
//!   of a uniform distribution over the same bins, `-sum(p ln p) / ln(k)`.
//!   NaN when `zmax < 2 by` or a height is negative (see
//!   [`MetricParams::clamp_negative`]). As in lidR the bins are
//!   half-open `[a, b)`, so a return exactly at the top edge is not counted.
//! - `pzabovezmean`, `pzabove<t>`: percentage of returns above the mean and
//!   above `t` m.
//! - `zq5` .. `zq95`: quantiles at 5 % steps, linear interpolation between
//!   order statistics (R's type 7, NumPy's default).
//! - `zpcum1` .. `zpcum9`: cumulative percentage of returns in the lower
//!   `k` tenths of `[0, zmax)` (lidR's breaks `seq(0, zmax, zmax / 10)`;
//!   returns at `zmax` or below 0 are not counted); all 0 when `zmax <= 0`.
//! - `cover`: percentage of first returns above the cover break;
//!   `gap_fraction`: share (0-1) of first returns at or below it. Without
//!   `return_number` every return counts as a first return.
//! - `itot`, `imax`, `imean`, `isd`, `iskew`, `ikurt`: the same statistics
//!   of `intensity`; `ipground`: percentage of the total intensity from
//!   ground returns (class 2); `ipcumzq10` .. `ipcumzq90`: percentage of the
//!   total intensity from returns at or below the 10, 30, 50, 70 and 90 %
//!   height quantiles.
//! - `p1th` .. `p5th`: percentage of returns that are the first .. fifth
//!   return; `pground`: percentage of ground returns (class 2).
//!
//! # Chunks and edges
//!
//! Grid cells lie on the catalogue grid of [`crate::als::catalog_grid`]. A
//! chunk computes the cells that lie wholly inside its core grown by 1.25
//! cells, from every point of the cell (the buffer is at least 1.5 cells), so
//! a cell's value never depends on where the chunks meet. Within a cell the
//! returns are put in a canonical order (height, x, y, then the attributes)
//! before anything is summed, so the value is the same to the bit from any
//! chunk and any number of threads.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::als::{catalog_grid, est_points, in_core, plan, run, workers_for, Catalog, Chunk, Layout, BYTES_PER_POINT};
use crate::als::ops::{chunk_heights, Heights, RunOptions, HIGH_NOISE_CLASS, NOISE_CLASS};
use crate::error::{Error, Result};
use crate::geo::masks::{MultiPolygon, PolygonIndex};
use crate::util::numeric::{pairwise_sum, quantile_sorted};
use crate::raster::Raster;
use crate::PointCloud;

/// ASPRS ground class.
const GROUND: f64 = 2.0;

/// Settings of the metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricParams {
    /// Height (m) of `pzabove<threshold>` (lidR's `th`, 2 m).
    pub threshold: f64,
    /// Bin width (m) of `zentropy` (lidR's `dz`, 1 m).
    pub entropy_bin: f64,
    /// Height (m) above which a first return is canopy for `cover` and `gap_fraction`.
    pub cover_break: f64,
    /// Returns lower than this height are left out of every metric.
    pub min_height: Option<f64>,
    /// Leave out returns classified as noise (7 or 18).
    pub drop_noise: bool,
    /// Set heights below 0 to 0 (before `min_height` is applied), as lidR
    /// users do with `Z[Z < 0] <- 0`. Ground returns a few centimetres
    /// below the DTM are in nearly every cell, and make `zentropy` NaN.
    pub clamp_negative: bool,
}

impl Default for MetricParams {
    fn default() -> Self {
        MetricParams { threshold: 2.0, entropy_bin: 1.0, cover_break: 2.0, min_height: None, drop_noise: true, clamp_negative: false }
    }
}

impl MetricParams {
    pub fn check(&self) -> Result<()> {
        if !self.threshold.is_finite() {
            return Err(Error::invalid(format!("threshold must be a finite height, got {}", self.threshold)));
        }
        if !(self.entropy_bin.is_finite() && self.entropy_bin > 0.0) {
            return Err(Error::invalid(format!("entropy_bin must be a positive number, got {}", self.entropy_bin)));
        }
        if !self.cover_break.is_finite() {
            return Err(Error::invalid(format!("cover_break must be a finite height, got {}", self.cover_break)));
        }
        if let Some(m) = self.min_height {
            if !m.is_finite() {
                return Err(Error::invalid(format!("min_height must be a finite height, got {m}")));
            }
        }
        Ok(())
    }
}

/// Which optional attributes the returns have; each adds its metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Available {
    pub intensity: bool,
    pub returns: bool,
    pub classification: bool,
}

impl Available {
    /// Every LAS point format has all three.
    pub const ALL: Available = Available { intensity: true, returns: true, classification: true };

    /// What a cloud has, by attribute name.
    pub fn of(cloud: &PointCloud) -> Available {
        Available { intensity: cloud.attr("intensity").is_some(), returns: cloud.attr("return_number").is_some(), classification: cloud.attr("classification").is_some() }
    }
}

/// Probabilities of `zq5` .. `zq95`.
fn z_probs() -> impl Iterator<Item = (usize, f64)> {
    (1..=19).map(|k| (5 * k, k as f64 / 20.0))
}

/// Probabilities of `ipcumzq10` .. `ipcumzq90`.
const I_PROBS: [(usize, f64); 5] = [(10, 0.1), (30, 0.3), (50, 0.5), (70, 0.7), (90, 0.9)];

/// Names of the metrics, in the order [`compute`] gives them.
pub fn metric_names(avail: Available, params: &MetricParams) -> Vec<String> {
    let mut v: Vec<String> = ["n", "zmax", "zmean", "zsd", "zskew", "zkurt", "zentropy", "pzabovezmean"].iter().map(|s| s.to_string()).collect();
    v.push(format!("pzabove{}", params.threshold));
    v.extend(z_probs().map(|(k, _)| format!("zq{k}")));
    v.extend((1..=9).map(|k| format!("zpcum{k}")));
    v.push("cover".into());
    v.push("gap_fraction".into());
    if avail.intensity {
        v.extend(["itot", "imax", "imean", "isd", "iskew", "ikurt"].iter().map(|s| s.to_string()));
        if avail.classification {
            v.push("ipground".into());
        }
        v.extend(I_PROBS.iter().map(|(k, _)| format!("ipcumzq{k}")));
    }
    if avail.returns {
        v.extend((1..=5).map(|k| format!("p{k}th")));
    }
    if avail.classification {
        v.push("pground".into());
    }
    v
}

/// How many metrics [`metric_names`] lists.
fn metric_count(avail: Available) -> usize {
    let mut n = 9 + 19 + 9 + 2;
    if avail.intensity {
        n += 6 + 5 + avail.classification as usize;
    }
    n + 5 * avail.returns as usize + avail.classification as usize
}

/// Indices into `all` of the `wanted` names (all of them when None).
pub fn select(all: &[String], wanted: Option<&[String]>) -> Result<Vec<usize>> {
    let Some(w) = wanted else { return Ok((0..all.len()).collect()) };
    if w.is_empty() {
        return Err(Error::invalid("no metric asked for"));
    }
    w.iter().map(|name| all.iter().position(|a| a == name).ok_or_else(|| Error::invalid(format!("unknown metric {name:?}; available: {}", all.join(", "))))).collect()
}

/// One return: height above ground and the attributes the metrics use
/// (NaN when absent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Return {
    pub z: f64,
    pub intensity: f64,
    pub return_number: f64,
    pub classification: f64,
}

fn cmp_returns(a: &Return, b: &Return) -> Ordering {
    a.z.total_cmp(&b.z).then(a.intensity.total_cmp(&b.intensity)).then(a.return_number.total_cmp(&b.return_number)).then(a.classification.total_cmp(&b.classification))
}

/// Mean, `S_2`, `S_3`, `S_4` (central sums) of `v`, summed pairwise as NumPy sums.
fn moments(v: &[f64]) -> (f64, f64, f64, f64) {
    let n = v.len() as f64;
    let mean = pairwise_sum(v) / n;
    let d2: Vec<f64> = v.iter().map(|x| (x - mean) * (x - mean)).collect();
    let d3: Vec<f64> = v.iter().zip(&d2).map(|(x, d)| d * (x - mean)).collect();
    let d4: Vec<f64> = d2.iter().map(|d| d * d).collect();
    (mean, pairwise_sum(&d2), pairwise_sum(&d3), pairwise_sum(&d4))
}

/// `sd`, `skew`, `kurt` from [`moments`] of `n` values.
fn shape(n: usize, s2: f64, s3: f64, s4: f64) -> (f64, f64, f64) {
    let nf = n as f64;
    let sd = if n > 1 { (s2 / (nf - 1.0)).sqrt() } else { f64::NAN };
    (sd, (s3 / nf) / (s2 / nf).powf(1.5), nf * s4 / (s2 * s2))
}

/// R's `seq(0, to, by)` for positive `by`: `(0:n) * by` capped at `to`, with
/// `n = as.integer(to / by + 1e-10)`.
fn r_seq(to: f64, by: f64) -> Vec<f64> {
    let n = (to / by + 1e-10) as usize;
    (0..=n).map(|k| (k as f64 * by).min(to)).collect()
}

/// R's `fast_table(findInterval(z, breaks), breaks.len() - 1)` as lidR uses
/// it: counts of the half-open intervals between consecutive breaks.
fn interval_counts(z: &[f64], breaks: &[f64]) -> Vec<f64> {
    let k = breaks.len() - 1;
    let mut counts = vec![0.0; k];
    for &v in z {
        let i = breaks.partition_point(|&b| b <= v);
        if i >= 1 && i <= k {
            counts[i - 1] += 1.0;
        }
    }
    counts
}

/// lidR's `entropy(z, by)`: normalised Shannon index of the heights in bins
/// of `by` from 0. `z` sorted ascending.
pub fn entropy(z: &[f64], by: f64) -> f64 {
    let (Some(&lo), Some(&zmax)) = (z.first(), z.last()) else { return f64::NAN };
    if zmax < 2.0 * by || lo < 0.0 {
        return f64::NAN;
    }
    let breaks = r_seq((zmax / by).ceil() * by, by);
    let counts = interval_counts(z, &breaks);
    let total: f64 = counts.iter().sum();
    let k = counts.len() as f64;
    let s: f64 = counts.iter().filter(|&&c| c > 0.0).map(|&c| c / total).map(|p| p * p.ln()).sum();
    let reference: f64 = counts.iter().map(|_| (1.0 / k) * (1.0 / k).ln()).sum();
    -s / -reference
}

/// lidR's cumulative height deciles `zpcum1` .. `zpcum9`. `z` sorted ascending.
fn zpcum(z: &[f64]) -> [f64; 9] {
    let zmax = *z.last().expect("non-empty");
    if zmax <= 0.0 {
        return [0.0; 9];
    }
    let breaks = r_seq(zmax, zmax / 10.0);
    let mut counts = interval_counts(z, &breaks);
    counts.resize(10, 0.0);
    let total: f64 = counts.iter().sum();
    let mut out = [0.0; 9];
    let mut acc = 0.0;
    for k in 0..9 {
        acc += counts[k] / total * 100.0;
        out[k] = acc;
    }
    out
}

fn percent(count: usize, n: usize) -> f64 {
    count as f64 / n as f64 * 100.0
}

/// Every metric of [`metric_names`] for these returns (put in canonical
/// order first). With no returns, `n` is 0 and everything else NaN.
pub fn compute(returns: &mut [Return], avail: Available, params: &MetricParams) -> Vec<f64> {
    let len = metric_count(avail);
    let n = returns.len();
    if n == 0 {
        let mut v = vec![f64::NAN; len];
        v[0] = 0.0;
        return v;
    }
    returns.sort_unstable_by(cmp_returns);
    let mut out = Vec::with_capacity(len);
    let z: Vec<f64> = returns.iter().map(|r| r.z).collect();
    let zmax = z[n - 1];
    let (zmean, s2, s3, s4) = moments(&z);
    let (zsd, zskew, zkurt) = shape(n, s2, s3, s4);
    out.extend([n as f64, zmax, zmean, zsd, zskew, zkurt, entropy(&z, params.entropy_bin)]);
    out.push(percent(z.iter().filter(|&&v| v > zmean).count(), n));
    out.push(percent(z.iter().filter(|&&v| v > params.threshold).count(), n));
    out.extend(z_probs().map(|(_, p)| quantile_sorted(&z, p)));
    out.extend(zpcum(&z));
    let first = |r: &Return| !avail.returns || r.return_number == 1.0;
    let n_first = returns.iter().filter(|r| first(r)).count();
    let above = returns.iter().filter(|r| first(r) && r.z > params.cover_break).count();
    if n_first == 0 {
        out.extend([f64::NAN, f64::NAN]);
    } else {
        out.push(percent(above, n_first));
        out.push((n_first - above) as f64 / n_first as f64);
    }
    if avail.intensity {
        let i: Vec<f64> = returns.iter().map(|r| r.intensity).collect();
        let itot = pairwise_sum(&i);
        let (imean, s2, s3, s4) = moments(&i);
        let (isd, iskew, ikurt) = shape(n, s2, s3, s4);
        let imax = i.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        out.extend([itot, imax, imean, isd, iskew, ikurt]);
        if avail.classification {
            let g: Vec<f64> = returns.iter().filter(|r| r.classification == GROUND).map(|r| r.intensity).collect();
            out.push(pairwise_sum(&g) / itot * 100.0);
        }
        for (_, p) in I_PROBS {
            let q = quantile_sorted(&z, p);
            // The returns are sorted by height, so those at or below q come first.
            let k = z.partition_point(|&v| v <= q);
            out.push(pairwise_sum(&i[..k]) / itot * 100.0);
        }
    }
    if avail.returns {
        for k in 1..=5 {
            out.push(percent(returns.iter().filter(|r| r.return_number == k as f64).count(), n));
        }
    }
    if avail.classification {
        out.push(percent(returns.iter().filter(|r| r.classification == GROUND).count(), n));
    }
    debug_assert_eq!(out.len(), len);
    out
}

// ------------------------------------------------------------------ heights and returns

/// Where heights above ground come from.
#[derive(Debug, Clone)]
pub enum HeightSource {
    /// As for the other catalogue operations: z, z minus a DTM, or z minus a
    /// DTM made per chunk from its ground points.
    Heights(Heights),
    /// A point attribute holding the height (as `als.normalize` writes it).
    Attribute(String),
}

impl HeightSource {
    fn check(&self) -> Result<()> {
        match self {
            HeightSource::Heights(Heights::Auto { resolution }) if !(resolution.is_finite() && *resolution > 0.0) => Err(Error::invalid(format!("dtm_resolution must be a positive number, got {resolution}"))),
            HeightSource::Heights(Heights::Dtm(r)) if r.data.is_empty() => Err(Error::invalid("the DTM is empty")),
            HeightSource::Attribute(a) if a.is_empty() => Err(Error::invalid("the height attribute needs a name")),
            _ => Ok(()),
        }
    }

    fn is_auto(&self) -> bool {
        matches!(self, HeightSource::Heights(Heights::Auto { .. }))
    }
}

const NO_GROUND: &str = "no chunk has 3 ground points (classification 2); classify ground first (als.classify_ground), or give a DTM";

/// Heights of every point of a chunk (clamped at 0 with
/// `params.clamp_negative`); None when it has too little ground for
/// [`Heights::Auto`].
fn heights_for(cat: &Catalog, chunk: &Chunk, cloud: &PointCloud, src: &HeightSource, params: &MetricParams) -> Result<Option<Vec<f64>>> {
    let h = match src {
        HeightSource::Heights(h) => chunk_heights(cat, chunk, cloud, h)?,
        HeightSource::Attribute(name) => Some(cloud.attr(name).map(|a| a.to_f64()).ok_or_else(|| Error::invalid(format!("the tiles have no {name:?} attribute")))?),
    };
    Ok(h.map(|h| clamped(h, params)))
}

/// `h` with negative heights set to 0 when `params.clamp_negative`.
fn clamped(mut h: Vec<f64>, params: &MetricParams) -> Vec<f64> {
    if params.clamp_negative {
        h.iter_mut().filter(|v| **v < 0.0).for_each(|v| *v = 0.0);
    }
    h
}

/// Indices of the points the metrics use: finite height, at least
/// `min_height`, and not noise when `drop_noise`.
pub fn kept(cloud: &PointCloud, heights: &[f64], params: &MetricParams) -> Vec<usize> {
    let cls = cloud.attr("classification").filter(|_| params.drop_noise);
    (0..cloud.len())
        .filter(|&i| {
            let h = heights[i];
            h.is_finite()
                && params.min_height.is_none_or(|m| h >= m)
                && cls.is_none_or(|c| {
                    let v = c.get_f64(i);
                    v != NOISE_CLASS as f64 && v != HIGH_NOISE_CLASS as f64
                })
        })
        .collect()
}

/// The points' columns the metrics and the canonical order read.
struct Columns<'a> {
    cloud: &'a PointCloud,
    heights: &'a [f64],
    intensity: Option<&'a crate::pointcloud::Attr>,
    return_number: Option<&'a crate::pointcloud::Attr>,
    classification: Option<&'a crate::pointcloud::Attr>,
}

impl<'a> Columns<'a> {
    fn new(cloud: &'a PointCloud, heights: &'a [f64]) -> Self {
        Columns { cloud, heights, intensity: cloud.attr("intensity"), return_number: cloud.attr("return_number"), classification: cloud.attr("classification") }
    }

    fn ret(&self, i: usize) -> Return {
        let get = |a: Option<&crate::pointcloud::Attr>| a.map_or(f64::NAN, |c| c.get_f64(i));
        Return { z: self.heights[i], intensity: get(self.intensity), return_number: get(self.return_number), classification: get(self.classification) }
    }

    /// Canonical order of two points: height, x, y, then the attributes.
    fn cmp(&self, a: usize, b: usize) -> Ordering {
        let (p, q) = (self.cloud.xyz[a], self.cloud.xyz[b]);
        self.heights[a].total_cmp(&self.heights[b]).then(p[0].total_cmp(&q[0])).then(p[1].total_cmp(&q[1])).then_with(|| cmp_returns(&self.ret(a), &self.ret(b)))
    }

    fn returns(&self, idx: &[usize]) -> Vec<Return> {
        idx.iter().map(|&i| self.ret(i)).collect()
    }
}

/// Metrics of one cloud, with `heights` per point. Returns the names and
/// values; the optional metrics are those of the attributes the cloud has.
pub fn cloud_metrics(cloud: &PointCloud, heights: &[f64], params: &MetricParams) -> Result<(Vec<String>, Vec<f64>)> {
    params.check()?;
    if heights.len() != cloud.len() {
        return Err(Error::invalid(format!("{} heights for {} points", heights.len(), cloud.len())));
    }
    let avail = Available::of(cloud);
    let heights = clamped(heights.to_vec(), params);
    let cols = Columns::new(cloud, &heights);
    let mut r = cols.returns(&kept(cloud, &heights, params));
    Ok((metric_names(avail, params), compute(&mut r, avail, params)))
}

// ------------------------------------------------------------------ grid

/// The buffer a grid run uses: at least 1.5 cells.
pub fn metrics_buffer(buffer: f64, resolution: f64) -> f64 {
    buffer.max(1.5 * resolution)
}

/// Chunks for [`grid_metrics`] (with the buffer of [`metrics_buffer`]).
pub fn metrics_plan(cat: &Catalog, layout: Layout, buffer: f64, resolution: f64) -> Result<Vec<Chunk>> {
    if !(resolution.is_finite() && resolution > 0.0) {
        return Err(Error::invalid(format!("resolution must be a positive number, got {resolution}")));
    }
    plan(cat, layout, metrics_buffer(buffer, resolution))
}

/// A chunk's retained points grouped by the grid cells it computes.
#[derive(Debug, Clone, Default)]
pub struct ChunkCells {
    /// Point indices into the chunk's cloud, cell by cell, each cell's
    /// points in canonical order.
    pub order: Vec<usize>,
    /// Height of each point of `order`.
    pub heights: Vec<f64>,
    /// Row-major index on the catalogue grid of each cell.
    pub cells: Vec<usize>,
    /// Cell `g` holds `order[starts[g]..starts[g + 1]]`.
    pub starts: Vec<usize>,
    /// Whether each cell's centre is in the chunk's core (such a chunk's
    /// value is preferred when several compute a cell).
    pub central: Vec<bool>,
}

/// Group a chunk's points by the cells of `grid` lying wholly inside its
/// core grown by 1.25 cells (and inside its buffered box). None when the
/// heights cannot be had (too little ground for [`Heights::Auto`]).
pub fn chunk_cells(cat: &Catalog, chunk: &Chunk, cloud: &PointCloud, grid: &Raster, heights: &HeightSource, params: &MetricParams) -> Result<Option<ChunkCells>> {
    let Some(h) = heights_for(cat, chunk, cloud, heights, params)? else { return Ok(None) };
    let res = grid.resolution;
    let (core, outer) = (chunk.core, chunk.outer);
    let pad = 1.25 * res;
    let b = [(core[0] - pad).max(outer[0]), (core[1] - pad).max(outer[1]), (core[2] + pad).min(outer[2]), (core[3] + pad).min(outer[3])];
    let c0 = ((b[0] - grid.xmin) / res).ceil().max(0.0) as i64;
    let r0 = ((b[1] - grid.ymin) / res).ceil().max(0.0) as i64;
    let c1 = (((b[2] - grid.xmin) / res).floor() as i64 - 1).min(grid.ncols as i64 - 1);
    let r1 = (((b[3] - grid.ymin) / res).floor() as i64 - 1).min(grid.nrows as i64 - 1);
    let mut keyed: Vec<(usize, usize)> = Vec::new();
    if c1 >= c0 && r1 >= r0 {
        for i in kept(cloud, &h, params) {
            let p = cloud.xyz[i];
            let (r, c) = grid.cell_index(p[0], p[1]);
            if r >= r0 && r <= r1 && c >= c0 && c <= c1 {
                keyed.push((r as usize * grid.ncols + c as usize, i));
            }
        }
    }
    let cols = Columns::new(cloud, &h);
    keyed.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| cols.cmp(a.1, b.1)));
    let mut out = ChunkCells::default();
    for (k, &(cell, i)) in keyed.iter().enumerate() {
        if k == 0 || keyed[k - 1].0 != cell {
            out.cells.push(cell);
            out.starts.push(k);
            let (x, y) = grid.cell_center(cell / grid.ncols, cell % grid.ncols);
            out.central.push(in_core(&core, x, y));
        }
        out.order.push(i);
        out.heights.push(h[i]);
    }
    out.starts.push(keyed.len());
    Ok(Some(out))
}

/// Metric rasters on the catalogue grid, one per name.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricRasters {
    pub names: Vec<String>,
    pub rasters: Vec<Raster>,
}

/// Join per-chunk cell values (in chunk order) onto `grid`: each cell from
/// the first chunk whose core holds its centre, else from the first chunk
/// that computed it. `parts` holds `(cells, central, values)` with
/// `values.len() == cells.len() * k`.
pub fn assemble(grid: &Raster, k: usize, parts: &[(Vec<usize>, Vec<bool>, Vec<f64>)]) -> Vec<Raster> {
    let n = grid.nrows * grid.ncols;
    let mut rank = vec![u8::MAX; n];
    let mut data = vec![f64::NAN; n * k];
    for (cells, central, values) in parts {
        for (g, &cell) in cells.iter().enumerate() {
            let r = if central[g] { 0 } else { 1 };
            if r < rank[cell] {
                rank[cell] = r;
                data[cell * k..(cell + 1) * k].copy_from_slice(&values[g * k..(g + 1) * k]);
            }
        }
    }
    (0..k).map(|j| Raster { data: (0..n).map(|c| data[c * k + j]).collect(), ..grid.clone() }).collect()
}

/// Rasters of the metrics `names` (all of [`metric_names`] if None) at
/// `resolution` over the whole catalogue. Cells without returns are NaN, as
/// are those of chunks without enough ground for [`Heights::Auto`]. The
/// values do not depend on the chunks or the number of workers (with
/// [`Heights::Auto`], away from where the per-chunk DTMs differ).
pub fn grid_metrics(cat: &Catalog, resolution: f64, heights: &HeightSource, params: &MetricParams, names: Option<&[String]>, opts: &RunOptions) -> Result<MetricRasters> {
    params.check()?;
    heights.check()?;
    let grid = catalog_grid(cat, resolution)?;
    let avail = Available::ALL;
    let all = metric_names(avail, params);
    let pick = select(&all, names)?;
    let chunks = metrics_plan(cat, opts.layout, opts.buffer, resolution)?;
    let w = workers_for(&chunks, opts.workers, BYTES_PER_POINT)?;
    let parts = run(cat, &chunks, w, "ALS metrics", |chunk, data| {
        let Some(cells) = chunk_cells(cat, chunk, &data.cloud, &grid, heights, params)? else { return Ok(None) };
        let cols = Columns::new(&data.cloud, &[]);
        let values: Vec<Vec<f64>> = (0..cells.cells.len())
            .into_par_iter()
            .map(|g| {
                let (s, e) = (cells.starts[g], cells.starts[g + 1]);
                let mut r: Vec<Return> = (s..e).map(|k| Return { z: cells.heights[k], ..ret_attrs(&cols, cells.order[k]) }).collect();
                let v = compute(&mut r, avail, params);
                pick.iter().map(|&j| v[j]).collect()
            })
            .collect();
        Ok(Some((cells.cells, cells.central, values.concat())))
    })?;
    let parts: Vec<_> = parts.into_iter().flatten().flatten().collect();
    if parts.is_empty() && heights.is_auto() {
        return Err(Error::invalid(NO_GROUND));
    }
    let rasters = assemble(&grid, pick.len(), &parts);
    Ok(MetricRasters { names: pick.iter().map(|&j| all[j].clone()).collect(), rasters })
}

/// A return's attributes (its height is filled in by the caller).
fn ret_attrs(cols: &Columns, i: usize) -> Return {
    let get = |a: Option<&crate::pointcloud::Attr>| a.map_or(f64::NAN, |c| c.get_f64(i));
    Return { z: f64::NAN, intensity: get(cols.intensity), return_number: get(cols.return_number), classification: get(cols.classification) }
}

// ------------------------------------------------------------------ plots

/// A plot to extract.
#[derive(Debug, Clone)]
pub enum Plot {
    /// A circle; a point on the circle is inside.
    Circle { x: f64, y: f64, radius: f64 },
    /// Polygons (closed sets, holes excluded), as [`crate::geo::masks`] tests them.
    Polygon(MultiPolygon),
}

enum Shape {
    Circle { x: f64, y: f64, r2: f64 },
    Polygon(PolygonIndex),
}

struct Prepared {
    /// `[xmin, ymin, xmax, ymax]`; None for a plot that holds nothing.
    bbox: Option<[f64; 4]>,
    shape: Shape,
}

impl Prepared {
    fn new(k: usize, p: &Plot) -> Result<Prepared> {
        Ok(match p {
            Plot::Circle { x, y, radius } => {
                if !(x.is_finite() && y.is_finite() && radius.is_finite() && *radius > 0.0) {
                    return Err(Error::invalid(format!("plot {k}: a circle needs a finite centre and a positive radius, got ({x}, {y}) and {radius}")));
                }
                Prepared { bbox: Some([x - radius, y - radius, x + radius, y + radius]), shape: Shape::Circle { x: *x, y: *y, r2: radius * radius } }
            }
            Plot::Polygon(mp) => {
                let index = PolygonIndex::new(std::slice::from_ref(mp)).map_err(|e| Error::invalid(format!("plot {k}: {e}")))?;
                let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                for v in mp.parts.iter().flat_map(|p| p.exterior.iter()) {
                    b = [b[0].min(v[0]), b[1].min(v[1]), b[2].max(v[0]), b[3].max(v[1])];
                }
                Prepared { bbox: b[0].is_finite().then_some(b), shape: Shape::Polygon(index) }
            }
        })
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        match &self.shape {
            Shape::Circle { x: cx, y: cy, r2 } => (x - cx) * (x - cx) + (y - cy) * (y - cy) <= *r2,
            Shape::Polygon(index) => index.locate(x, y).is_some(),
        }
    }
}

fn meets(a: &[f64; 4], b: &[f64; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Run `f(plot, cloud, heights, idx)` on every plot, where `idx` are the
/// indices into `cloud` of the retained points inside the plot, in
/// canonical order, and `heights` the height of every point of `cloud`.
/// Plots are grouped by the tiles they overlap, and each group is read
/// once (only those tiles, only the points in the group's box, grown by
/// `buffer` for [`Heights::Auto`]) and processed as a chunk of
/// [`crate::als::run`]. Returns None for plots that overlap no tile or
/// whose group has no points or too little ground for [`Heights::Auto`].
pub fn plot_points<T: Send>(cat: &Catalog, plots: &[Plot], heights: &HeightSource, params: &MetricParams, buffer: f64, workers: usize, f: impl Fn(usize, &PointCloud, &[f64], &[usize]) -> Result<T> + Sync) -> Result<Vec<Option<T>>> {
    params.check()?;
    heights.check()?;
    cat.check_usable()?;
    if !(buffer.is_finite() && buffer >= 0.0) {
        return Err(Error::invalid(format!("buffer must be a non-negative number of metres, got {buffer}")));
    }
    let prepared: Vec<Prepared> = plots.iter().enumerate().map(|(k, p)| Prepared::new(k, p)).collect::<Result<_>>()?;
    let pad = if heights.is_auto() { buffer } else { 0.0 };
    let tile_boxes: Vec<[f64; 4]> = cat.tiles.iter().map(|t| t.xy()).collect();
    let files_of = |b: &[f64; 4]| -> Vec<usize> { (0..cat.tiles.len()).filter(|&i| cat.tiles[i].n_points > 0 && meets(&tile_boxes[i], b)).collect() };
    let mut groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for (k, p) in prepared.iter().enumerate() {
        if let Some(b) = p.bbox {
            let files = files_of(&b);
            if !files.is_empty() {
                groups.entry(files).or_default().push(k);
            }
        }
    }
    let groups: Vec<Vec<usize>> = groups.into_values().collect();
    let chunks: Vec<Chunk> = groups
        .iter()
        .enumerate()
        .map(|(g, members)| {
            let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
            for &k in members {
                let p = prepared[k].bbox.expect("grouped plots have a box");
                b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[2]), b[3].max(p[3])];
            }
            let outer = [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad];
            // The core is closed on every side (a point on a plot's eastern
            // or northern edge is its own).
            let core = [b[0], b[1], b[2].next_up(), b[3].next_up()];
            let files = files_of(&outer);
            Chunk { index: g, core, outer, own: None, est_points: est_points(cat, &outer, &files), files, name: format!("plots {}", members[0]) }
        })
        .collect();
    let w = workers_for(&chunks, workers, BYTES_PER_POINT)?;
    let parts = run(cat, &chunks, w, "ALS plots", |chunk, data| {
        let cloud = &data.cloud;
        let Some(h) = heights_for(cat, chunk, cloud, heights, params)? else { return Ok(None) };
        let mut by_x = kept(cloud, &h, params);
        by_x.sort_unstable_by(|&a, &b| cloud.xyz[a][0].total_cmp(&cloud.xyz[b][0]).then(a.cmp(&b)));
        let cols = Columns::new(cloud, &h);
        let out: Vec<(usize, T)> = groups[chunk.index]
            .par_iter()
            .map(|&k| {
                let p = &prepared[k];
                let b = p.bbox.expect("grouped plots have a box");
                let lo = by_x.partition_point(|&i| cloud.xyz[i][0] < b[0]);
                let hi = by_x.partition_point(|&i| cloud.xyz[i][0] <= b[2]);
                let mut idx: Vec<usize> = by_x[lo..hi]
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let q = cloud.xyz[i];
                        q[1] >= b[1] && q[1] <= b[3] && p.contains(q[0], q[1])
                    })
                    .collect();
                idx.sort_unstable_by(|&a, &b| cols.cmp(a, b));
                Ok((k, f(k, cloud, &h, &idx)?))
            })
            .collect::<Result<_>>()?;
        Ok(Some(out))
    })?;
    let mut results: Vec<Option<T>> = (0..plots.len()).map(|_| None).collect();
    for (k, v) in parts.into_iter().flatten().flatten().flatten() {
        results[k] = Some(v);
    }
    Ok(results)
}

/// Metrics `names` (all of [`metric_names`] if None) of every plot, one row
/// per plot. A plot without returns has `n` 0 and NaN elsewhere.
pub fn plot_metrics(cat: &Catalog, plots: &[Plot], heights: &HeightSource, params: &MetricParams, names: Option<&[String]>, buffer: f64, workers: usize) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
    let avail = Available::ALL;
    let all = metric_names(avail, params);
    let pick = select(&all, names)?;
    let rows = plot_points(cat, plots, heights, params, buffer, workers, |_, cloud, h, idx| {
        let mut r = Columns::new(cloud, h).returns(idx);
        let v = compute(&mut r, avail, params);
        Ok(pick.iter().map(|&j| v[j]).collect::<Vec<f64>>())
    })?;
    let empty: Vec<f64> = {
        let v = compute(&mut [], avail, params);
        pick.iter().map(|&j| v[j]).collect()
    };
    Ok((pick.iter().map(|&j| all[j].clone()).collect(), rows.into_iter().map(|r| r.unwrap_or_else(|| empty.clone())).collect()))
}

/// Plot metrics as CSV text: a `plot` column (0-based), an `id` column
/// when `ids` are given, then one column per metric; NaN is an empty cell.
pub fn metrics_csv(names: &[String], ids: Option<&[String]>, rows: &[Vec<f64>]) -> Result<String> {
    if let Some(ids) = ids {
        if ids.len() != rows.len() {
            return Err(Error::invalid(format!("{} ids for {} plots", ids.len(), rows.len())));
        }
    }
    let quote = |s: &str| if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() };
    let mut out = String::from("plot");
    if ids.is_some() {
        out.push_str(",id");
    }
    for n in names {
        out.push(',');
        out.push_str(&quote(n));
    }
    out.push('\n');
    for (k, row) in rows.iter().enumerate() {
        if row.len() != names.len() {
            return Err(Error::invalid(format!("plot {k} has {} values for {} metrics", row.len(), names.len())));
        }
        out.push_str(&k.to_string());
        if let Some(ids) = ids {
            out.push(',');
            out.push_str(&quote(&ids[k]));
        }
        for v in row {
            out.push(',');
            if !v.is_nan() {
                out.push_str(&format!("{v:?}"));
            }
        }
        out.push('\n');
    }
    Ok(out)
}

/// Write [`metrics_csv`] to `path`.
pub fn write_metrics_csv(path: impl AsRef<std::path::Path>, names: &[String], ids: Option<&[String]>, rows: &[Vec<f64>]) -> Result<()> {
    std::fs::write(path, metrics_csv(names, ids, rows)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::als::read_region;
    use crate::als::ops::write_tiles;
    use crate::io::las::LasWriteOptions;
    use crate::geo::masks::Polygon;
    use crate::util::nprandom::Generator;
    use crate::pointcloud::Attr;
    use std::path::{Path, PathBuf};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
    }

    fn get(names: &[String], v: &[f64], name: &str) -> f64 {
        v[names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no {name}"))]
    }

    fn rets(z: &[f64]) -> Vec<Return> {
        z.iter().map(|&z| Return { z, intensity: f64::NAN, return_number: f64::NAN, classification: f64::NAN }).collect()
    }

    #[test]
    fn statistics_of_a_known_sample() {
        let p = MetricParams::default();
        let avail = Available { intensity: false, returns: false, classification: false };
        let names = metric_names(avail, &p);
        // 0, 1, ..., 10: mean 5, sd sqrt(11), symmetric so zero skewness.
        let z: Vec<f64> = (0..=10).map(|v| v as f64).collect();
        let v = compute(&mut rets(&z), avail, &p);
        assert_eq!(v.len(), names.len());
        assert_eq!(get(&names, &v, "n"), 11.0);
        assert_eq!(get(&names, &v, "zmean"), 5.0);
        assert!(close(get(&names, &v, "zsd"), 11f64.sqrt(), 1e-15));
        assert!(get(&names, &v, "zskew").abs() < 1e-15);
        // Pearson kurtosis of a discrete uniform on 11 values: m4 / m2^2 with m2 = 10, m4 = 178.
        assert!(close(get(&names, &v, "zkurt"), 178.0 / 100.0, 1e-14));
        assert!(close(get(&names, &v, "zq25"), 2.5, 1e-15));
        assert!(close(get(&names, &v, "zq95"), 9.5, 1e-15));
        assert!(close(get(&names, &v, "pzabove2"), 8.0 / 11.0 * 100.0, 1e-15));
        assert!(close(get(&names, &v, "pzabovezmean"), 5.0 / 11.0 * 100.0, 1e-15));
        // Entropy: bins [0,1) .. [9,10), one return each (10 is on the top edge
        // and not counted), so exactly uniform.
        assert!(close(get(&names, &v, "zentropy"), 1.0, 1e-15));
        // zpcum: breaks 0, 1, ..., 10; one return in each tenth.
        for k in 1..=9 {
            assert!(close(get(&names, &v, &format!("zpcum{k}")), k as f64 * 10.0, 1e-13), "zpcum{k}");
        }
        assert!(close(get(&names, &v, "cover"), 8.0 / 11.0 * 100.0, 1e-15));
        assert!(close(get(&names, &v, "gap_fraction"), 3.0 / 11.0, 1e-15));
        // Order does not matter.
        let mut rev: Vec<f64> = z.iter().rev().cloned().collect();
        rev.swap(2, 7);
        assert_eq!(compute(&mut rets(&rev), avail, &p), v);
    }

    #[test]
    fn entropy_follows_lidr() {
        // All in one bin of [0, 5): log(1) = 0.
        assert_eq!(entropy(&[0.5, 0.6, 0.7, 4.5], 1.0), -(0.75f64 * 0.75f64.ln() + 0.25 * 0.25f64.ln()) / 5f64.ln());
        // Too low, or a negative height: NaN.
        assert!(entropy(&[0.1, 1.9], 1.0).is_nan());
        assert!(entropy(&[-0.1, 5.0], 1.0).is_nan());
        assert!(entropy(&[], 1.0).is_nan());
    }

    #[test]
    fn negative_heights_can_be_clamped() {
        // Ground returns just under the DTM make zentropy NaN unless clamped.
        let cloud = PointCloud::new(vec![[0.0; 3]; 5]);
        let h = [-0.03, -0.01, 0.0, 6.5, 12.2];
        let get = |p: &MetricParams, name: &str| {
            let (names, v) = cloud_metrics(&cloud, &h, p).unwrap();
            v[names.iter().position(|n| n == name).unwrap()]
        };
        let lidr = MetricParams::default();
        let clamp = MetricParams { clamp_negative: true, ..Default::default() };
        assert!(get(&lidr, "zentropy").is_nan());
        let mut z = [0.0, 0.0, 0.0, 6.5, 12.2];
        z.sort_by(f64::total_cmp);
        assert_eq!(get(&clamp, "zentropy"), entropy(&z, 1.0));
        assert_eq!(get(&clamp, "n"), 5.0);
        assert_eq!(get(&clamp, "zq5"), 0.0);
        // min_height=0 drops them instead.
        assert_eq!(get(&MetricParams { min_height: Some(0.0), ..Default::default() }, "n"), 3.0);
    }

    #[test]
    fn empty_single_and_flat_samples() {
        let p = MetricParams::default();
        let a = Available::ALL;
        let names = metric_names(a, &p);
        let v = compute(&mut [], a, &p);
        assert_eq!(v[0], 0.0);
        assert!(v[1..].iter().all(|x| x.is_nan()));
        let mut one = vec![Return { z: 3.0, intensity: 10.0, return_number: 1.0, classification: 1.0 }];
        let v = compute(&mut one, a, &p);
        assert_eq!(get(&names, &v, "zmax"), 3.0);
        assert!(get(&names, &v, "zsd").is_nan() && get(&names, &v, "zskew").is_nan());
        assert_eq!(get(&names, &v, "zq50"), 3.0);
        assert_eq!(get(&names, &v, "p1th"), 100.0);
        assert_eq!(get(&names, &v, "cover"), 100.0);
        assert_eq!(get(&names, &v, "itot"), 10.0);
        assert_eq!(get(&names, &v, "ipground"), 0.0);
        // Everything on the ground: zpcum is 0, entropy NaN.
        let mut flat = rets(&[0.0, 0.0, -0.2]);
        let v = compute(&mut flat, Available { intensity: false, returns: false, classification: false }, &p);
        assert_eq!(v[metric_names(Available { intensity: false, returns: false, classification: false }, &p).iter().position(|n| n == "zpcum5").unwrap()], 0.0);
    }

    #[test]
    fn names_and_selection() {
        let p = MetricParams { threshold: 2.5, ..Default::default() };
        let names = metric_names(Available::ALL, &p);
        assert!(names.contains(&"pzabove2.5".to_string()));
        assert_eq!(names.len(), 9 + 19 + 9 + 2 + 7 + 5 + 5 + 1);
        for (i, r, c) in [(false, false, false), (true, false, false), (true, false, true), (false, true, true), (true, true, true)] {
            let a = Available { intensity: i, returns: r, classification: c };
            assert_eq!(metric_count(a), metric_names(a, &p).len());
        }
        assert_eq!(select(&names, Some(&["zmax".to_string()])).unwrap(), vec![1]);
        assert!(select(&names, Some(&["zmx".to_string()])).is_err());
        assert!(MetricParams { entropy_bin: 0.0, ..Default::default() }.check().is_err());
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sylva-alsm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// 120 x 120 m of returns with heights, intensities, return numbers and
    /// classes, as 3 x 3 tiles of 40 m (z is the height).
    fn scene(dir: &Path) -> Catalog {
        let mut rng = Generator::new(11);
        let n = 60_000;
        let u = rng.uniform_n(0.0, 1.0, 5 * n);
        let mut xyz = Vec::with_capacity(n);
        let (mut int, mut rn, mut cls) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let (x, y) = (120.0 * u[5 * i], 120.0 * u[5 * i + 1]);
            let ground = u[5 * i + 2] < 0.3;
            let z = if ground { 0.0 } else { 25.0 * u[5 * i + 3] * (1.0 + (x / 30.0).sin()) / 2.0 };
            xyz.push([x, y, z]);
            int.push((50.0 + 200.0 * u[5 * i + 4]) as u16);
            rn.push(1 + (u[5 * i + 4] * 3.0) as u8);
            cls.push(if ground { 2u8 } else if i % 97 == 0 { 7 } else { 1 });
        }
        let mut cloud = PointCloud::new(xyz);
        cloud.attrs.insert("intensity".into(), Attr::U16(int));
        cloud.attrs.insert("return_number".into(), Attr::U8(rn));
        cloud.attrs.insert("classification".into(), Attr::U8(cls));
        #[allow(clippy::needless_update)]
        let opts = LasWriteOptions { point_format: 6, scale: 0.001, ..Default::default() };
        let tiles = write_tiles(&cloud, dir, 40.0, Some((0.0, 0.0)), "laz", &opts, None).unwrap();
        Catalog::open(&tiles.iter().map(|t| t.0.clone()).collect::<Vec<_>>())
    }

    #[test]
    fn a_grid_does_not_depend_on_chunks_or_workers_and_matches_the_merged_cloud() {
        let d = tmp("grid");
        let cat = scene(&d);
        let h = HeightSource::Heights(Heights::Z);
        let p = MetricParams::default();
        let res = 7.0; // cells do not line up with the 40 m tiles
        let one = grid_metrics(&cat, res, &h, &p, None, &RunOptions { buffer: 0.0, workers: 1, ..Default::default() }).unwrap();
        let four = grid_metrics(&cat, res, &h, &p, None, &RunOptions { layout: Layout::Grid { size: 25.0, origin: None }, buffer: 0.0, workers: 4 }).unwrap();
        let same = |a: &Raster, b: &Raster| a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
        for (a, b) in one.rasters.iter().zip(&four.rasters) {
            assert!(same(a, b));
        }
        // Every cell against the metrics of the merged cloud's points in it.
        let merged = read_region(&cat, [-1.0, -1.0, 121.0, 121.0]).unwrap();
        let grid = &one.rasters[0];
        let heights: Vec<f64> = merged.xyz.iter().map(|q| q[2]).collect();
        let mut cells: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();
        for i in kept(&merged, &heights, &p) {
            cells.entry(grid.cell_index(merged.xyz[i][0], merged.xyz[i][1])).or_default().push(i);
        }
        let cols = Columns::new(&merged, &heights);
        for ((r, c), idx) in &cells {
            let v = compute(&mut cols.returns(idx), Available::ALL, &p);
            for (j, raster) in one.rasters.iter().enumerate() {
                let got = raster.get(*r as usize, *c as usize);
                assert!(close(got, v[j], 1e-12), "{} at {r} {c}: {got} vs {}", one.names[j], v[j]);
            }
        }
        let n_cells = one.rasters[0].data.iter().filter(|v| v.is_finite()).count();
        assert_eq!(n_cells, cells.len());
        // A subset of metrics comes back in the order asked.
        let two = grid_metrics(&cat, res, &h, &p, Some(&["zq95".into(), "n".into()]), &RunOptions::default()).unwrap();
        assert_eq!(two.names, vec!["zq95", "n"]);
        assert!(same(&two.rasters[1], &one.rasters[0]));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn plots_match_the_points_inside_them() {
        let d = tmp("plots");
        let cat = scene(&d);
        let h = HeightSource::Heights(Heights::Z);
        let p = MetricParams::default();
        let square = MultiPolygon { parts: vec![Polygon { exterior: vec![[35.0, 35.0], [55.0, 35.0], [55.0, 55.0], [35.0, 55.0]], holes: vec![vec![[40.0, 40.0], [45.0, 40.0], [45.0, 45.0], [40.0, 45.0]]] }] };
        let plots = vec![Plot::Circle { x: 40.0, y: 80.0, radius: 11.3 }, Plot::Polygon(square), Plot::Circle { x: 500.0, y: 500.0, radius: 5.0 }, Plot::Circle { x: 5.0, y: 5.0, radius: 3.0 }];
        let (names, rows) = plot_metrics(&cat, &plots, &h, &p, None, 0.0, 2).unwrap();
        let merged = read_region(&cat, [-1.0, -1.0, 121.0, 121.0]).unwrap();
        let heights: Vec<f64> = merged.xyz.iter().map(|q| q[2]).collect();
        let cols = Columns::new(&merged, &heights);
        let keep = kept(&merged, &heights, &p);
        let inside_circle = |x: f64, y: f64, r: f64| -> Vec<usize> { keep.iter().copied().filter(|&i| (merged.xyz[i][0] - x).powi(2) + (merged.xyz[i][1] - y).powi(2) <= r * r).collect() };
        let inside_square: Vec<usize> = keep
            .iter()
            .copied()
            .filter(|&i| {
                let (x, y) = (merged.xyz[i][0], merged.xyz[i][1]);
                (35.0..=55.0).contains(&x) && (35.0..=55.0).contains(&y) && !(x > 40.0 && x < 45.0 && y > 40.0 && y < 45.0)
            })
            .collect();
        for (k, idx) in [(0, inside_circle(40.0, 80.0, 11.3)), (1, inside_square), (3, inside_circle(5.0, 5.0, 3.0))] {
            let v = compute(&mut cols.returns(&idx), Available::ALL, &p);
            assert_eq!(rows[k][0], idx.len() as f64);
            for j in 0..names.len() {
                assert!(close(rows[k][j], v[j], 1e-12), "plot {k} {}", names[j]);
            }
        }
        assert_eq!(rows[2][0], 0.0);
        assert!(rows[2][1].is_nan());
        assert!(plot_metrics(&cat, &[Plot::Circle { x: 1.0, y: 1.0, radius: 0.0 }], &h, &p, None, 0.0, 1).is_err());
        let csv = metrics_csv(&names[..2], Some(&["a,b".into(), "c".into(), "d".into(), "e".into()]), &rows.iter().map(|r| r[..2].to_vec()).collect::<Vec<_>>()).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "plot,id,n,zmax");
        assert!(lines[1].starts_with("0,\"a,b\","));
        assert_eq!(lines[3], "2,d,0.0,");
        let _ = std::fs::remove_dir_all(&d);
    }
}
