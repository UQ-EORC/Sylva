// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Ray-based canopy structure from airborne and UAV lidar.
//!
//! Three parts, each usable on one cloud or on a whole catalogue through
//! the chunk engine of [`crate::als`]:
//!
//! * **Pulses.** [`reconstruct`] groups discrete returns into pulses by GPS
//!   time (and flight line), puts each pulse's origin at the sensor
//!   position interpolated from the trajectory, and gives [`Shots`] that the
//!   ray tracer of [`crate::voxel`] takes as it takes terrestrial scans.
//!   Returns missing from a pulse are counted, and pulses that returned
//!   nothing can be inferred where the regular firing leaves a hole of a
//!   few pulses (see [`PulseParams::fill_missing`]).
//! * **Gap-fraction profiles.** [`profile_cloud`] and [`profile_catalog`]
//!   accumulate, per grid cell and height layer, the share of the pulses
//!   stopped there, and turn it into plant area density by inverting the
//!   Beer-Lambert law layer by layer: the method of MacArthur & Horn
//!   (1969), as in Bouvier et al. (2015), with each return's
//!   extinction `G(θ) / cos θ` at its own beam zenith θ, so that oblique
//!   pulses, which cross more foliage per metre of height, are not read as
//!   denser canopy.
//! * **Ray-traced voxels.** [`voxelize_catalog`] reconstructs the pulses of
//!   each chunk and traces them with [`crate::voxel`] on the part of one
//!   catalogue-wide voxel lattice that the chunk's core owns.
//!
//! Discrete-return data cannot give back everything a pulse did: a return
//! below the detection threshold, returns closer than the range resolution
//! (merged into one) and the energy each return carried are lost; a pulse
//! that returned nothing leaves no record at all, and only the regularity of
//! the firing tells where one is missing.

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::als::{catalog_grid, plan, run, workers_for_estimates, Catalog, BYTES_PER_POINT};
use crate::als::ops::{chunk_heights, noise_mask, Heights, RunOptions};
use crate::als::trajectory::{estimate, line_extents, merge_extents, pulse_lines, thin, EstimateParams, Estimated, Trajectory};
use crate::error::{Error, Result};
use crate::raster::Raster;
use crate::shots::Shots;
use crate::voxel::{compute_g, EchoLabels, Lad, VoxelParams};
use crate::util::limits;
use crate::voxel::grid as voxel_grid;
use crate::{Point, PointCloud};

// ------------------------------------------------------------------ pulses

/// Settings of [`reconstruct`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseParams {
    /// Longest gap between trajectory samples (s) to interpolate across;
    /// None for ten median sample intervals.
    pub max_gap: Option<f64>,
    /// Seconds added to the returns' `gps_time` to put them on the
    /// trajectory's clock.
    pub time_offset: f64,
    /// Add the pulses that returned nothing where a line's firing shows a
    /// hole of at most `max_fill` pulses.
    pub fill_missing: bool,
    pub max_fill: usize,
    /// Leave out pulses with fewer returns than `number_of_returns` says.
    pub drop_incomplete: bool,
}

impl Default for PulseParams {
    fn default() -> Self {
        PulseParams { max_gap: None, time_offset: 0.0, fill_missing: false, max_fill: 8, drop_incomplete: false }
    }
}

/// What [`reconstruct`] found.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PulseReport {
    /// Returns read.
    pub n_returns: usize,
    /// Pulses in the output, including the inferred ones.
    pub n_pulses: usize,
    /// Pulses with fewer returns than `number_of_returns` says.
    pub n_incomplete: usize,
    /// Returns missing from those pulses.
    pub n_missing_returns: usize,
    /// Incomplete pulses left out (`drop_incomplete`).
    pub n_dropped: usize,
    /// Groups of returns sharing a GPS time that repeat a return number
    /// and so were split into several pulses.
    pub n_split: usize,
    /// Returns left out because the trajectory does not cover their time.
    pub n_unpositioned: usize,
    /// Pulses without a return that were inferred.
    pub n_filled: usize,
    /// Median time between successive pulses of a line (s).
    pub pulse_interval: f64,
    /// Median and 95th percentile of the distance (m) from the sensor
    /// position to the line through a pulse's first and last returns, over
    /// pulses whose returns are at least 1 m apart. A correct trajectory
    /// gives centimetres to decimetres; metres point to a time or height
    /// offset between the trajectory and the points.
    pub line_offset_median: f64,
    pub line_offset_p95: f64,
}

fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    *v.select_nth_unstable_by(k, f64::total_cmp).1
}

fn norm(v: &Point) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sub(a: &Point, b: &Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// GPS seconds of the week of an adjusted standard GPS time (GPS time
/// minus 10⁹ s, as LAS files with the global encoding bit set store it).
pub fn week_seconds(adjusted_standard: f64) -> f64 {
    (adjusted_standard + 1e9).rem_euclid(604_800.0)
}

fn no_overlap_error(times: &[f64], traj: &Trajectory, offset: f64) -> Error {
    let (lo, hi) = times.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| (a.min(t), b.max(t)));
    let (t0, t1) = (traj.time[0], traj.time[traj.len() - 1]);
    let mut msg = format!("no return falls within the trajectory: the returns' gps_time (+ time_offset {offset}) runs from {:.3} to {:.3} s, the trajectory from {t0:.3} to {t1:.3} s", lo + offset, hi + offset);
    let w = week_seconds(lo);
    if lo.is_finite() && w >= t0 - 60.0 && w <= t1 + 60.0 {
        msg.push_str(&format!(". The returns look like adjusted standard GPS time and the trajectory like seconds of the week: pass time_offset = {:.6}", w - lo));
    }
    Error::invalid(msg)
}

/// Group the returns of `cloud` into pulses with their origin at the
/// sensor (see the module docs). Needs `gps_time`; uses `point_source_id`
/// (the flight line), `return_number` and `number_of_returns` when present.
/// Every attribute of the points becomes an echo attribute. Pulses come out
/// ordered by flight line, then time; inferred pulses without returns sit
/// in time order among them.
///
/// Each pulse's direction is the unit vector from the sensor to its
/// farthest return, and every return's range is its distance from the
/// sensor.
///
/// # Errors
/// No `gps_time`, or no return within the trajectory (the message suggests
/// the offset when the clocks look like adjusted standard time against
/// seconds of the week).
pub fn reconstruct(cloud: &PointCloud, traj: &Trajectory, p: &PulseParams) -> Result<(Shots, PulseReport)> {
    let n = cloud.len();
    let mut report = PulseReport { n_returns: n, pulse_interval: f64::NAN, line_offset_median: f64::NAN, line_offset_p95: f64::NAN, ..Default::default() };
    if n == 0 {
        return Ok((Shots::default(), report));
    }
    let time = cloud.attr_f64("gps_time").ok_or_else(|| Error::invalid("the points have no gps_time attribute, so their pulses cannot be told apart"))?;
    let line = cloud.attr_f64("point_source_id");
    let rn = cloud.attr_f64("return_number");
    let nr = cloud.attr_f64("number_of_returns");
    let max_gap = p.max_gap.unwrap_or_else(|| traj.default_max_gap());
    if max_gap.is_nan() || max_gap < 0.0 {
        return Err(Error::invalid(format!("max_gap must be a non-negative number of seconds, got {max_gap}")));
    }
    if !p.time_offset.is_finite() {
        return Err(Error::invalid("time_offset must be finite"));
    }
    let lid = |i: usize| line.as_ref().map_or(0, |l| l[i] as i64);
    let rank = |i: usize| rn.as_ref().map_or(0.0, |r| r[i]);
    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_by(|&a, &b| lid(a).cmp(&lid(b)).then(time[a].total_cmp(&time[b])).then(rank(a).total_cmp(&rank(b))).then(a.cmp(&b)));

    // Pulses as ranges of `order`: equal (line, time), split where a return number repeats.
    let mut pulses: Vec<(usize, usize)> = Vec::new();
    let mut s = 0;
    while s < n {
        let mut e = s + 1;
        let mut split = false;
        while e < n && lid(order[e]) == lid(order[s]) && time[order[e]] == time[order[s]] {
            if rn.is_some() && rank(order[e]) <= rank(order[e - 1]) {
                pulses.push((s, e));
                s = e;
                split = true;
            }
            e += 1;
        }
        report.n_split += split as usize;
        pulses.push((s, e));
        s = e;
    }
    let ptime: Vec<f64> = pulses.iter().map(|&(s, _)| time[order[s]] + p.time_offset).collect();
    let origins = traj.positions(&ptime, max_gap);
    if origins.iter().all(|o| o[0].is_nan()) {
        return Err(no_overlap_error(&ptime.iter().map(|t| t - p.time_offset).collect::<Vec<_>>(), traj, p.time_offset));
    }

    struct Pulse {
        line: i64,
        time: f64,
        origin: Point,
        dir: Point,
        echoes: Vec<(f64, usize)>,
    }
    let mut kept: Vec<Pulse> = Vec::with_capacity(pulses.len());
    let mut offsets = Vec::new();
    for (k, &(s, e)) in pulses.iter().enumerate() {
        let o = origins[k];
        if o[0].is_nan() {
            report.n_unpositioned += e - s;
            continue;
        }
        let members = &order[s..e];
        let expected = nr.as_ref().map_or(members.len(), |v| members.iter().map(|&i| v[i] as usize).max().unwrap_or(0));
        if members.len() < expected {
            report.n_incomplete += 1;
            report.n_missing_returns += expected - members.len();
            if p.drop_incomplete {
                report.n_dropped += 1;
                continue;
            }
        }
        let mut echoes: Vec<(f64, usize)> = members.iter().map(|&i| (norm(&sub(&cloud.xyz[i], &o)), i)).collect();
        echoes.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let (r_far, far) = *echoes.last().expect("a pulse has a return");
        if r_far.is_nan() || r_far <= 0.0 {
            report.n_unpositioned += e - s;
            continue;
        }
        let v = sub(&cloud.xyz[far], &o);
        let dir = [v[0] / r_far, v[1] / r_far, v[2] / r_far];
        if echoes.len() >= 2 {
            let (first, last) = (cloud.xyz[echoes[0].1], cloud.xyz[far]);
            let d = sub(&first, &last);
            let sep = norm(&d);
            if sep >= 1.0 {
                let u = [d[0] / sep, d[1] / sep, d[2] / sep];
                let w = sub(&o, &last);
                let t = w[0] * u[0] + w[1] * u[1] + w[2] * u[2];
                offsets.push(norm(&[w[0] - t * u[0], w[1] - t * u[1], w[2] - t * u[2]]));
            }
        }
        kept.push(Pulse { line: lid(members[0]), time: time[members[0]], origin: o, dir, echoes });
    }
    report.line_offset_median = quantile(offsets.clone(), 0.5);
    report.line_offset_p95 = quantile(offsets, 0.95);

    // Pulse interval and angular step, from successive pulses of a line.
    let diffs: Vec<f64> = kept.windows(2).filter(|w| w[0].line == w[1].line).map(|w| w[1].time - w[0].time).filter(|&d| d > 0.0).collect();
    report.pulse_interval = quantile(diffs, 0.5);
    let mut filled: Vec<Pulse> = Vec::new();
    if p.fill_missing && report.pulse_interval.is_finite() {
        let dt = report.pulse_interval;
        let angle = |a: &Point, b: &Point| (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0).acos();
        let steps: Vec<f64> = kept.windows(2).filter(|w| w[0].line == w[1].line && ((w[1].time - w[0].time) / dt - 1.0).abs() < 0.1).map(|w| angle(&w[0].dir, &w[1].dir)).collect();
        let step = quantile(steps, 0.5);
        let mut times = Vec::new();
        let mut spans = Vec::new();
        for (a, w) in kept.windows(2).enumerate() {
            if w[0].line != w[1].line {
                continue;
            }
            let ratio = (w[1].time - w[0].time) / dt;
            let k = ratio.round();
            if k < 2.0 || k > p.max_fill as f64 + 1.0 || (ratio - k).abs() > 0.1 {
                continue;
            }
            // The mirror must have swept steadily across the hole: no turn and no jump.
            let turn = angle(&w[0].dir, &w[1].dir);
            if turn.is_nan() || turn > 1.5 * k * step + 1e-6 {
                continue;
            }
            for j in 1..k as usize {
                times.push(w[0].time + (w[1].time - w[0].time) * j as f64 / k + p.time_offset);
                spans.push((a, j as f64 / k));
            }
        }
        let origins = traj.positions(&times, max_gap);
        for ((a, f), (o, t)) in spans.into_iter().zip(origins.into_iter().zip(times)) {
            if o[0].is_nan() {
                continue;
            }
            let (da, db) = (kept[a].dir, kept[a + 1].dir);
            let d = [da[0] + f * (db[0] - da[0]), da[1] + f * (db[1] - da[1]), da[2] + f * (db[2] - da[2])];
            let l = norm(&d);
            filled.push(Pulse { line: kept[a].line, time: t - p.time_offset, origin: o, dir: [d[0] / l, d[1] / l, d[2] / l], echoes: Vec::new() });
        }
        report.n_filled = filled.len();
    }
    let mut all: Vec<Pulse> = kept.into_iter().chain(filled).collect();
    all.sort_by(|a, b| a.line.cmp(&b.line).then(a.time.total_cmp(&b.time)).then(b.echoes.len().cmp(&a.echoes.len())));

    let mut shots = Shots::default();
    let mut echo_idx = Vec::with_capacity(n);
    for pl in &all {
        shots.origin.push(pl.origin);
        shots.direction.push(pl.dir);
        shots.echo_start.push(shots.echo_range.len());
        shots.echo_count.push(pl.echoes.len() as u32);
        for &(r, i) in &pl.echoes {
            shots.echo_range.push(r);
            echo_idx.push(i);
        }
    }
    shots.echo_attrs = cloud.attrs.iter().map(|(k, v)| (k.clone(), v.take(&echo_idx))).collect();
    report.n_pulses = shots.n_shots();
    Ok((shots, report))
}

/// [`crate::als::trajectory::estimate`] over a whole catalogue: the pulse
/// lines of every tile (its own points only, so that a pulse split between
/// tiles is not paired across them), thinned per tile to what the estimate
/// uses.
pub fn estimate_catalog(cat: &Catalog, p: &EstimateParams, workers: usize) -> Result<Estimated> {
    let chunks = plan(cat, crate::als::Layout::Tiles, 0.0)?;
    let w = workers_for_estimates(&chunks.iter().map(|c| c.est_points).collect::<Vec<_>>(), workers, BYTES_PER_POINT)?;
    let parts = run(cat, &chunks, w, "pulse lines", |_, data| {
        let own = data.cloud.take(&data.core_indices());
        Ok((thin(pulse_lines(&own, p.min_separation)?, p.interval, p.max_pulses), line_extents(&own)?))
    })?;
    let (mut lines, mut extents) = (Vec::new(), Vec::new());
    for (l, e) in parts.into_iter().flatten() {
        lines.extend(l);
        extents.extend(e);
    }
    estimate(&thin(lines, p.interval, p.max_pulses), &merge_extents(extents), p)
}

// ---------------------------------------------------------------- profiles

/// Share of its pulse each return stands for in a gap-fraction profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnWeight {
    /// `1 / number_of_returns` (Armston et al. 2013): a pulse's returns
    /// together count as one pulse.
    Equal,
    /// First returns only, each a whole pulse (MacArthur & Horn 1969).
    First,
    /// Every return a whole pulse (the profile of Bouvier et al. 2015).
    All,
}

impl ReturnWeight {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "equal" => ReturnWeight::Equal,
            "first" => ReturnWeight::First,
            "all" => ReturnWeight::All,
            other => return Err(Error::invalid(format!("unknown return weighting {other:?}; expected 'equal', 'first' or 'all'"))),
        })
    }
}

/// Projection of the foliage on the beam.
#[derive(Debug, Clone, PartialEq)]
pub enum Projection {
    /// `G(θ)` of a leaf angle distribution.
    Lad(Lad),
    /// A constant `G` (the extinction coefficient `k`).
    Constant(f64),
}

/// Where each return's beam zenith (off-nadir angle) comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum Angles {
    /// From the sensor position interpolated at its time.
    Trajectory { trajectory: Trajectory, max_gap: Option<f64>, time_offset: f64 },
    /// The absolute LAS `scan_angle` (degrees from nadir; includes roll, not pitch).
    ScanAngle,
    /// Every beam vertical: no correction.
    Nadir,
}

/// Default [`ProfileParams::top_quantile`]: one part in 100,000 of the
/// weight above `min_height` may lie above the top layer.
pub const DEFAULT_TOP_QUANTILE: f64 = 0.99999;

/// Settings of the gap-fraction profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileParams {
    /// Cell size (m).
    pub resolution: f64,
    /// Height (m) of the bottom of the lowest layer; returns below it,
    /// ground returns included, are the pulses that got through.
    pub min_height: f64,
    /// Layer thickness (m).
    pub bin_size: f64,
    /// Top of the highest layer (m); returns above it count as intercepted
    /// above every layer. None: up to the height below which `top_quantile`
    /// of the weight above `min_height` lies (see [`ProfileGrid::trim_top`]).
    pub max_height: Option<f64>,
    /// Share (0-1] of the weight above `min_height` the layers must hold
    /// when `max_height` is None; 1 reaches the highest return. Just below
    /// 1, a few stray returns far above the canopy (birds, haze, a mast)
    /// do not stretch the profile with empty layers.
    pub top_quantile: f64,
    /// Leave out returns classified as noise (7 or 18).
    pub drop_noise: bool,
    pub weighting: ReturnWeight,
    pub projection: Projection,
    pub angles: Angles,
    /// Returns whose beam is further than this from nadir (degrees) are left out.
    pub max_zenith: f64,
    /// Count each return in the cell where its beam meets the ground
    /// (known only from a trajectory) rather than where the return is. Which
    /// pulses a cell holds then depends on their geometry alone, not on
    /// whether the canopy stopped them.
    pub anchor_ground: bool,
}

impl Default for ProfileParams {
    fn default() -> Self {
        ProfileParams { resolution: 10.0, min_height: 1.0, bin_size: 1.0, max_height: None, top_quantile: DEFAULT_TOP_QUANTILE, drop_noise: true, weighting: ReturnWeight::Equal, projection: Projection::Lad(Lad::Spherical), angles: Angles::Nadir, max_zenith: 90.0, anchor_ground: true }
    }
}

impl ProfileParams {
    fn check(&self) -> Result<()> {
        if !(self.resolution.is_finite() && self.resolution > 0.0) {
            return Err(Error::invalid(format!("resolution must be a positive number of metres, got {}", self.resolution)));
        }
        if !(self.bin_size.is_finite() && self.bin_size > 0.0) {
            return Err(Error::invalid(format!("bin_size must be a positive number of metres, got {}", self.bin_size)));
        }
        if !self.min_height.is_finite() {
            return Err(Error::invalid("min_height must be finite"));
        }
        if let Some(m) = self.max_height {
            if !(m.is_finite() && m > self.min_height) {
                return Err(Error::invalid(format!("max_height must be above min_height, got {m}")));
            }
        }
        if !(self.top_quantile > 0.0 && self.top_quantile <= 1.0) {
            return Err(Error::invalid(format!("top_quantile must be in (0, 1], got {}", self.top_quantile)));
        }
        if let Projection::Constant(g) = self.projection {
            if !(g.is_finite() && g > 0.0) {
                return Err(Error::invalid(format!("the projection G must be positive, got {g}")));
            }
        }
        if !(self.max_zenith > 0.0 && self.max_zenith <= 90.0) {
            return Err(Error::invalid(format!("max_zenith must be in (0, 90] degrees, got {}", self.max_zenith)));
        }
        Ok(())
    }

    /// Layers between `min_height` and `max_height`, or enough to hold `top`.
    fn n_layers(&self, top: f64) -> usize {
        let n = match self.max_height {
            Some(hi) => ((hi - self.min_height) / self.bin_size).ceil(),
            None => ((top - self.min_height) / self.bin_size).floor() + 1.0,
        };
        n.max(1.0) as usize
    }
}

/// Weighted returns per cell and height layer. Layer 0 holds what is below
/// `min_height`, layers `1..=nz` the bins of `bin_size` above it, layer
/// `nz + 1` what is above `max_height`. `weight` sums each return's share
/// of its pulse, `weight_k` that share times the return's extinction per
/// unit vertical plant area, `G(θ) / cos θ`. Arrays are `(nz + 2, ny, nx)`,
/// row 0 at the south.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProfileGrid {
    pub xmin: f64,
    pub ymin: f64,
    pub resolution: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub min_height: f64,
    pub bin_size: f64,
    pub weight: Vec<f64>,
    pub weight_k: Vec<f64>,
    /// Returns left out: no position on the trajectory, or beyond `max_zenith`.
    pub n_skipped: usize,
}

impl ProfileGrid {
    fn empty(xmin: f64, ymin: f64, nx: usize, ny: usize, nz: usize, p: &ProfileParams) -> Result<ProfileGrid> {
        let cells = (nz as u128 + 2) * ny as u128 * nx as u128;
        limits::check_cells(cells, 16, &format!("a {nx} x {ny} x {} profile grid", nz + 2), "a coarser resolution or bin_size")?;
        let n = cells as usize;
        Ok(ProfileGrid { xmin, ymin, resolution: p.resolution, nx, ny, nz, min_height: p.min_height, bin_size: p.bin_size, weight: vec![0.0; n], weight_k: vec![0.0; n], n_skipped: 0 })
    }

    /// Grow to `nz` layers (only when no `max_height` fixes them).
    fn with_layers(mut self, nz: usize) -> ProfileGrid {
        if nz <= self.nz {
            return self;
        }
        let per = self.nx * self.ny;
        let grow = |v: &mut Vec<f64>, old: usize| {
            // The "above" layer of the smaller grid is empty without max_height.
            v.truncate((old + 1) * per);
            v.resize((nz + 2) * per, 0.0);
        };
        grow(&mut self.weight, self.nz);
        grow(&mut self.weight_k, self.nz);
        self.nz = nz;
        self
    }

    /// Drop the top layers that together hold no more than `1 - quantile`
    /// of the weight above `min_height`, folding their returns into the
    /// layer above the top (intercepted above every layer). The layers kept
    /// are unchanged: a layer's transmittance depends only on the returns at
    /// and below it. At least one layer is kept.
    pub fn trim_top(&mut self, quantile: f64) {
        let per = self.nx * self.ny;
        let layer_sum = |l: usize| self.weight[l * per..(l + 1) * per].iter().sum::<f64>();
        let sums: Vec<f64> = (1..=self.nz + 1).map(layer_sum).collect();
        let allowed = (1.0 - quantile) * sums.iter().sum::<f64>();
        // Layers 1..=nz are sums[0..nz]; sums[nz] is the layer above.
        let mut keep = self.nz;
        let mut above = sums[self.nz];
        while keep > 1 && above + sums[keep - 1] <= allowed {
            above += sums[keep - 1];
            keep -= 1;
        }
        if keep == self.nz {
            return;
        }
        for v in [&mut self.weight, &mut self.weight_k] {
            for l in keep + 1..=self.nz + 1 {
                for i in 0..per {
                    let w = v[l * per + i];
                    if l != keep + 1 {
                        v[(keep + 1) * per + i] += w;
                    }
                }
            }
            v.truncate((keep + 2) * per);
        }
        self.nz = keep;
    }
}

/// The weight each return carries.
fn return_weights(cloud: &PointCloud, w: ReturnWeight) -> Result<Vec<f64>> {
    let need = |name: &str| cloud.attr_f64(name).ok_or_else(|| Error::invalid(format!("the {w:?} weighting needs the {name} attribute").to_lowercase()));
    Ok(match w {
        ReturnWeight::All => vec![1.0; cloud.len()],
        ReturnWeight::First => need("return_number")?.iter().map(|&r| if r <= 1.0 { 1.0 } else { 0.0 }).collect(),
        ReturnWeight::Equal => need("number_of_returns")?.iter().map(|&r| 1.0 / r.max(1.0)).collect(),
    })
}

/// Off-nadir beam angle (rad) of each return, and the horizontal distance
/// its beam travels per metre of descent (`[dx, dy]`, zero without a
/// trajectory); NaN where the angle cannot be had.
fn beam_geometry(cloud: &PointCloud, angles: &Angles) -> Result<(Vec<f64>, Vec<[f64; 2]>)> {
    let flat = |th: Vec<f64>| { let n = th.len(); (th, vec![[0.0; 2]; n]) };
    Ok(match angles {
        Angles::Nadir => flat(vec![0.0; cloud.len()]),
        Angles::ScanAngle => flat(cloud.attr_f64("scan_angle").ok_or_else(|| Error::invalid("the points have no scan_angle attribute; give a trajectory, or angles='none'"))?.iter().map(|a| a.abs().to_radians()).collect()),
        Angles::Trajectory { trajectory, max_gap, time_offset } => {
            let t: Vec<f64> = cloud.attr_f64("gps_time").ok_or_else(|| Error::invalid("the points have no gps_time attribute to place them on the trajectory"))?.iter().map(|t| t + time_offset).collect();
            let o = trajectory.positions(&t, max_gap.unwrap_or_else(|| trajectory.default_max_gap()));
            if !cloud.is_empty() && o.iter().all(|p| p[0].is_nan()) {
                return Err(no_overlap_error(&t.iter().map(|v| v - time_offset).collect::<Vec<_>>(), trajectory, *time_offset));
            }
            cloud.xyz.iter().zip(&o).map(|(q, s)| {
                let v = sub(s, q);
                let r = norm(&v);
                if r > 0.0 && v[2] > 0.0 { ((v[2] / r).clamp(-1.0, 1.0).acos(), [-v[0] / v[2], -v[1] / v[2]]) } else { (f64::NAN, [0.0; 2]) }
            }).unzip()
        }
    })
}

/// Add the returns `idx` of a cloud (with their heights) to `g`, which
/// holds the cells from `offset` (column, row) of a grid anchored at
/// `anchor`; returns outside it are ignored. Cells are found from the
/// anchor of the whole grid, so that a part of it bins every return as the
/// whole grid would.
fn accumulate(g: &mut ProfileGrid, cloud: &PointCloud, heights: &[f64], idx: &[usize], p: &ProfileParams, anchor: (f64, f64), offset: (usize, usize)) -> Result<()> {
    let w = return_weights(cloud, p.weighting)?;
    let (theta, drift) = beam_geometry(cloud, &p.angles)?;
    let max_z = p.max_zenith.to_radians();
    let per = g.nx * g.ny;
    for &i in idx {
        let (h, th) = (heights[i], theta[i]);
        if !h.is_finite() || th.is_nan() || th > max_z || th >= std::f64::consts::FRAC_PI_2 {
            g.n_skipped += 1;
            continue;
        }
        if w[i] == 0.0 {
            continue;
        }
        // Where the pulse meets the ground, or where the return is.
        let reach = if p.anchor_ground { h.max(0.0) } else { 0.0 };
        let c = ((cloud.xyz[i][0] + reach * drift[i][0] - anchor.0) / g.resolution).floor() - offset.0 as f64;
        let r = ((cloud.xyz[i][1] + reach * drift[i][1] - anchor.1) / g.resolution).floor() - offset.1 as f64;
        if c < 0.0 || r < 0.0 || c >= g.nx as f64 || r >= g.ny as f64 {
            continue;
        }
        let layer = if h < g.min_height {
            0
        } else {
            let l = ((h - g.min_height) / g.bin_size).floor() as usize + 1;
            if l > g.nz { g.nz + 1 } else { l }
        };
        let gv = match &p.projection {
            Projection::Constant(v) => *v,
            Projection::Lad(l) => compute_g(th, l),
        };
        let k = layer * per + r as usize * g.nx + c as usize;
        g.weight[k] += w[i];
        g.weight_k[k] += w[i] * gv / th.cos();
    }
    Ok(())
}

/// Grid layout over `[xmin, ymin, xmax, ymax]`: corner snapped down to a
/// multiple of the resolution, as the catalogue grid.
fn layout(b: [f64; 4], res: f64) -> (f64, f64, usize, usize) {
    let xmin = (b[0] / res).floor() * res;
    let ymin = (b[1] / res).floor() * res;
    (xmin, ymin, ((b[2] - xmin) / res).floor() as usize + 1, ((b[3] - ymin) / res).floor() as usize + 1)
}

fn top_height(heights: &[f64], idx: &[usize]) -> f64 {
    idx.iter().map(|&i| heights[i]).filter(|h| h.is_finite()).fold(f64::NEG_INFINITY, f64::max)
}

/// The profile grid of one cloud with the given heights above ground, over
/// `bounds` (by default the cloud's extent).
pub fn profile_cloud(cloud: &PointCloud, heights: &[f64], p: &ProfileParams, bounds: Option<[f64; 4]>) -> Result<ProfileGrid> {
    p.check()?;
    if heights.len() != cloud.len() {
        return Err(Error::invalid(format!("{} heights for {} points", heights.len(), cloud.len())));
    }
    let b = match bounds {
        Some(b) => b,
        None => {
            let (lo, hi) = cloud.bounds().ok_or_else(|| Error::invalid("the cloud is empty"))?;
            [lo[0], lo[1], hi[0], hi[1]]
        }
    };
    if !(b[2] >= b[0] && b[3] >= b[1]) {
        return Err(Error::invalid(format!("bounds must be (xmin, ymin, xmax, ymax), got {b:?}")));
    }
    let (xmin, ymin, nx, ny) = layout(b, p.resolution);
    let idx = without_noise(cloud, (0..cloud.len()).collect(), p);
    let nz = p.n_layers(top_height(heights, &idx).max(p.min_height));
    let mut g = ProfileGrid::empty(xmin, ymin, nx, ny, nz, p)?;
    accumulate(&mut g, cloud, heights, &idx, p, (xmin, ymin), (0, 0))?;
    if p.max_height.is_none() {
        g.trim_top(p.top_quantile);
    }
    Ok(g)
}

/// `idx` without the returns classified as noise, when `p.drop_noise`.
fn without_noise(cloud: &PointCloud, idx: Vec<usize>, p: &ProfileParams) -> Vec<usize> {
    match noise_mask(cloud).filter(|_| p.drop_noise) {
        Some(noise) => idx.into_iter().filter(|&i| !noise[i]).collect(),
        None => idx,
    }
}

/// The profile grid of a whole catalogue, on the catalogue grid at
/// `p.resolution`. Every point is counted once, by the chunk whose core
/// holds it; its height comes from `heights` with the chunk's buffer.
/// Cells near a chunk edge sum the counts of the chunks around them, so
/// the result does not depend on the chunks (with `anchor_ground`, as long
/// as no beam drifts further than the buffer between a return and the
/// ground; returns that would are left out).
pub fn profile_catalog(cat: &Catalog, heights: &Heights, p: &ProfileParams, opts: &RunOptions) -> Result<ProfileGrid> {
    p.check()?;
    let grid = catalog_grid(cat, p.resolution)?;
    let chunks = plan(cat, opts.layout, opts.buffer)?;
    let w = workers_for_estimates(&chunks.iter().map(|c| c.est_points).collect::<Vec<_>>(), opts.workers, BYTES_PER_POINT)?;
    let parts = run(cat, &chunks, w, "canopy profiles", |chunk, data| {
        let Some(h) = chunk_heights(cat, chunk, &data.cloud, heights)? else { return Ok(None) };
        let idx = without_noise(&data.cloud, data.core_indices(), p);
        // The buffered box: a return of the core can be counted where its beam meets the ground.
        let cell = |v: f64, o: f64, n: usize| (((v - o) / p.resolution).floor().max(0.0) as usize).min(n - 1);
        let (c0, c1) = (cell(chunk.outer[0], grid.xmin, grid.ncols), cell(chunk.outer[2], grid.xmin, grid.ncols));
        let (r0, r1) = (cell(chunk.outer[1], grid.ymin, grid.nrows), cell(chunk.outer[3], grid.ymin, grid.nrows));
        let nz = p.n_layers(top_height(&h, &idx).max(p.min_height));
        let mut g = ProfileGrid::empty(grid.xmin + c0 as f64 * p.resolution, grid.ymin + r0 as f64 * p.resolution, c1 + 1 - c0, r1 + 1 - r0, nz, p)?;
        accumulate(&mut g, &data.cloud, &h, &idx, p, (grid.xmin, grid.ymin), (c0, r0))?;
        Ok(Some((c0, r0, g)))
    })?;
    let parts: Vec<(usize, usize, ProfileGrid)> = parts.into_iter().flatten().flatten().collect();
    if parts.is_empty() && matches!(heights, Heights::Auto { .. }) {
        return Err(Error::invalid("no chunk has enough ground points (classification 2) for a DTM; classify the ground first or give a DTM"));
    }
    let nz = parts.iter().map(|x| x.2.nz).max().unwrap_or_else(|| p.n_layers(p.min_height));
    let mut out = ProfileGrid::empty(grid.xmin, grid.ymin, grid.ncols, grid.nrows, nz, p)?;
    let per = out.nx * out.ny;
    for (c0, r0, g) in parts {
        let g = g.with_layers(nz);
        out.n_skipped += g.n_skipped;
        for l in 0..nz + 2 {
            for r in 0..g.ny {
                for c in 0..g.nx {
                    let (src, dst) = (l * g.nx * g.ny + r * g.nx + c, l * per + (r0 + r) * out.nx + c0 + c);
                    out.weight[dst] += g.weight[src];
                    out.weight_k[dst] += g.weight_k[src];
                }
            }
        }
    }
    if p.max_height.is_none() {
        out.trim_top(p.top_quantile);
    }
    Ok(out)
}

/// Plant area density (m²/m³) of each layer of one column, from its
/// `nz + 2` weights and weighted extinctions. The transmittance of layer
/// `i` is the weight below it over the weight at or below it, and its
/// density `-ln T / (k̄ dz)` with `k̄` the mean extinction of the returns at
/// or below it. A layer nothing passed is given `T = 0.5 / max(E, 1)` (half
/// a pulse of the `E` that entered), a lower bound on its density; a layer
/// no pulse reached is NaN.
pub fn column_pad(weight: &[f64], weight_k: &[f64], bin_size: f64) -> Vec<f64> {
    let nz = weight.len().saturating_sub(2);
    let mut out = Vec::with_capacity(nz);
    let (mut below, mut below_k) = (weight[0], weight_k[0]);
    for i in 1..=nz {
        let e = below + weight[i];
        let ek = below_k + weight_k[i];
        out.push(if e > 0.0 && ek > 0.0 {
            let t = (below / e).max(0.5 / e.max(1.0));
            // + 0.0 turns the -0.0 of a clear layer into 0.
            -t.ln() / (ek / e * bin_size) + 0.0
        } else {
            f64::NAN
        });
        below = e;
        below_k = ek;
    }
    out
}

/// Gap probability at each layer boundary of one column, from the ground
/// up: the weight below `min_height + i * bin_size` over the whole weight,
/// `i = 0..=nz`.
pub fn column_pgap(weight: &[f64]) -> Vec<f64> {
    let total: f64 = weight.iter().sum();
    let mut acc = 0.0;
    let nz = weight.len().saturating_sub(2);
    (0..=nz)
        .map(|i| {
            acc += weight[i];
            if total > 0.0 { acc / total } else { f64::NAN }
        })
        .collect()
}

/// Plant area index of one column: `-ln P / k̄` with `P` the share of the
/// weight below `min_height` and `k̄` the mean extinction of all returns;
/// saturated as in [`column_pad`]; NaN without returns.
pub fn column_pai(weight: &[f64], weight_k: &[f64]) -> f64 {
    let total: f64 = weight.iter().sum();
    let total_k: f64 = weight_k.iter().sum();
    if !(total > 0.0 && total_k > 0.0) {
        return f64::NAN;
    }
    let pg = (weight[0] / total).max(0.5 / total.max(1.0));
    -pg.ln() / (total_k / total) + 0.0
}

// ------------------------------------------------------- profile metrics

/// One height stratum of the profile metrics: its edges (m) and the layers
/// (1-based, as in the counts) whose middle falls inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Stratum {
    lo: f64,
    hi: f64,
    first: usize,
    last: usize,
}

/// The strata of `size` m from the ground that hold at least one of the
/// `nz` layers above `min_height`. A layer belongs to the stratum its
/// middle falls in, so strata that are not a whole number of layers keep
/// every layer exactly once.
fn strata(min_height: f64, bin_size: f64, nz: usize, size: f64) -> Vec<Stratum> {
    let mut out: Vec<Stratum> = Vec::new();
    for layer in 1..=nz {
        let mid = min_height + (layer as f64 - 0.5) * bin_size;
        let k = (mid / size).floor();
        let (lo, hi) = (k * size, (k + 1.0) * size);
        match out.last_mut() {
            Some(s) if s.lo == lo => s.last = layer,
            _ => out.push(Stratum { lo, hi, first: layer, last: layer }),
        }
    }
    out
}

fn check_strata(bin_size: f64, size: f64) -> Result<()> {
    if !(size.is_finite() && size >= bin_size) {
        return Err(Error::invalid(format!("strata must be at least the layer thickness ({bin_size} m), got {size}")));
    }
    Ok(())
}

/// Number of metrics before the per-stratum ones.
const PROFILE_SCALARS: usize = 8;

/// Names of the metrics [`column_metrics`] gives for a profile of `nz`
/// layers of `bin_size` m above `min_height`, with strata of `strata` m.
///
/// - `pulses`: the weight of the column, the number of pulses with the
///   `equal` weighting.
/// - `pai`: plant area index above `min_height` (as [`column_pai`]).
/// - `cover`: one minus the gap probability at `min_height`.
/// - `fhd`: foliage height diversity, the Shannon index `-sum(p ln p)` of
///   the shares `p` of the plant area in each layer (MacArthur & MacArthur
///   1961; GEDI L2B's `fhd_normal`). 0 for a column with no plant area.
/// - `pad_max`, `height_pad_max`: the densest layer's plant area density
///   and the height of its middle.
/// - `height_pad_mean`, `height_pad_sd`: mean and standard deviation of
///   height weighted by plant area, the centre and the spread of the canopy.
/// - per stratum `[a, b)`: `pavd_<a>_<b>`, its mean plant area density
///   (as GEDI's `pavd_z`); `pai_above_<h>`, the plant area index above the
///   bottom `h` of its lowest layer (as `pai_z`); `cover_above_<h>`, the
///   cover at that height (as `cover_z`).
pub fn profile_metric_names(min_height: f64, bin_size: f64, nz: usize, strata_size: f64) -> Result<Vec<String>> {
    check_strata(bin_size, strata_size)?;
    let mut v: Vec<String> = ["pulses", "pai", "cover", "fhd", "pad_max", "height_pad_max", "height_pad_mean", "height_pad_sd"].iter().map(|s| s.to_string()).collect();
    debug_assert_eq!(v.len(), PROFILE_SCALARS);
    let st = strata(min_height, bin_size, nz, strata_size);
    v.extend(st.iter().map(|s| format!("pavd_{}_{}", s.lo, s.hi)));
    v.extend(st.iter().map(|s| format!("pai_above_{}", min_height + (s.first - 1) as f64 * bin_size)));
    v.extend(st.iter().map(|s| format!("cover_above_{}", min_height + (s.first - 1) as f64 * bin_size)));
    Ok(v)
}

/// The metrics of [`profile_metric_names`] for one column of `nz + 2`
/// weights and weighted extinctions. A column without returns is NaN
/// throughout but for `pulses` (0); a column with returns and no plant area
/// has 0 plant area, density and diversity and NaN heights. Layers no pulse
/// reached are left out of the sums.
pub fn column_metrics(weight: &[f64], weight_k: &[f64], min_height: f64, bin_size: f64, strata_size: f64) -> Vec<f64> {
    let nz = weight.len().saturating_sub(2);
    let st = strata(min_height, bin_size, nz, strata_size);
    let n = PROFILE_SCALARS + 3 * st.len();
    let total: f64 = weight.iter().sum();
    if total.is_nan() || total <= 0.0 {
        let mut v = vec![f64::NAN; n];
        v[0] = 0.0;
        return v;
    }
    let pad = column_pad(weight, weight_k, bin_size);
    let pgap = column_pgap(weight);
    let mid = |layer: usize| min_height + (layer as f64 - 0.5) * bin_size;
    // Plant area of each layer (m²/m²), 0 where no pulse reached it.
    let area: Vec<f64> = pad.iter().map(|&p| if p.is_finite() { p * bin_size } else { 0.0 }).collect();
    let sum_area: f64 = area.iter().sum();

    let mut out = Vec::with_capacity(n);
    out.push(total);
    out.push(column_pai(weight, weight_k));
    out.push(1.0 - pgap[0]);
    if sum_area > 0.0 {
        let fhd: f64 = area.iter().filter(|&&a| a > 0.0).map(|&a| a / sum_area).map(|p| -p * p.ln()).sum();
        let (imax, pmax) = pad.iter().enumerate().filter(|(_, p)| p.is_finite()).fold((0, f64::NEG_INFINITY), |acc, (i, &p)| if p > acc.1 { (i, p) } else { acc });
        let mean: f64 = area.iter().enumerate().map(|(i, a)| a * mid(i + 1)).sum::<f64>() / sum_area;
        let var: f64 = area.iter().enumerate().map(|(i, a)| a * (mid(i + 1) - mean).powi(2)).sum::<f64>() / sum_area;
        out.extend([fhd + 0.0, pmax, mid(imax + 1), mean, var.sqrt()]);
    } else {
        out.extend([0.0, 0.0, f64::NAN, f64::NAN, f64::NAN]);
    }
    for s in &st {
        let reached: Vec<usize> = (s.first..=s.last).filter(|&l| pad[l - 1].is_finite()).collect();
        let thick = reached.len() as f64 * bin_size;
        out.push(if thick > 0.0 { reached.iter().map(|&l| area[l - 1]).sum::<f64>() / thick } else { f64::NAN });
    }
    for s in &st {
        out.push(area[s.first - 1..].iter().sum());
    }
    for s in &st {
        out.push(1.0 - pgap[s.first - 1]);
    }
    debug_assert_eq!(out.len(), n);
    out
}

impl ProfileGrid {
    fn column(&self, v: &[f64], cell: usize) -> Vec<f64> {
        let per = self.nx * self.ny;
        (0..self.nz + 2).map(|l| v[l * per + cell]).collect()
    }

    /// Plant area density of every layer and cell, `(nz, ny, nx)`.
    pub fn pad(&self) -> Vec<f64> {
        let per = self.nx * self.ny;
        let cols: Vec<Vec<f64>> = (0..per).into_par_iter().map(|c| column_pad(&self.column(&self.weight, c), &self.column(&self.weight_k, c), self.bin_size)).collect();
        let mut out = vec![0.0; self.nz * per];
        for (c, col) in cols.iter().enumerate() {
            for (l, v) in col.iter().enumerate() {
                out[l * per + c] = *v;
            }
        }
        out
    }

    /// Plant area index of every cell, `(ny, nx)`.
    pub fn pai(&self) -> Vec<f64> {
        (0..self.nx * self.ny).into_par_iter().map(|c| column_pai(&self.column(&self.weight, c), &self.column(&self.weight_k, c))).collect()
    }

    /// Canopy cover of every cell: one minus the gap probability at `min_height`.
    pub fn cover(&self) -> Vec<f64> {
        (0..self.nx * self.ny)
            .map(|c| {
                let w = self.column(&self.weight, c);
                let total: f64 = w.iter().sum();
                if total > 0.0 { 1.0 - w[0] / total } else { f64::NAN }
            })
            .collect()
    }

    /// The layer sums over the cells where `mask` is true (all when None):
    /// the pooled column of an area.
    pub fn pooled(&self, mask: Option<&[bool]>) -> Result<(Vec<f64>, Vec<f64>)> {
        let per = self.nx * self.ny;
        if let Some(m) = mask {
            if m.len() != per {
                return Err(Error::invalid(format!("the mask has {} cells, the grid {per}", m.len())));
            }
        }
        let sum = |v: &[f64]| -> Vec<f64> { (0..self.nz + 2).map(|l| (0..per).filter(|&c| mask.is_none_or(|m| m[c])).map(|c| v[l * per + c]).sum()).collect() };
        Ok((sum(&self.weight), sum(&self.weight_k)))
    }

    /// The profile metrics of every cell ([`profile_metric_names`]):
    /// the names, and the values as `(k, ny, nx)`.
    pub fn metrics(&self, strata_size: f64) -> Result<(Vec<String>, Vec<f64>)> {
        let names = profile_metric_names(self.min_height, self.bin_size, self.nz, strata_size)?;
        let per = self.nx * self.ny;
        let cols: Vec<Vec<f64>> = (0..per).into_par_iter().map(|c| column_metrics(&self.column(&self.weight, c), &self.column(&self.weight_k, c), self.min_height, self.bin_size, strata_size)).collect();
        let mut out = vec![0.0; names.len() * per];
        for (c, col) in cols.iter().enumerate() {
            for (k, v) in col.iter().enumerate() {
                out[k * per + c] = *v;
            }
        }
        Ok((names, out))
    }

    /// The profile metrics of areas, each given by the cells (row-major
    /// indices) whose counts are pooled into one column: one row per area.
    pub fn area_metrics(&self, areas: &[Vec<usize>], strata_size: f64) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
        let names = profile_metric_names(self.min_height, self.bin_size, self.nz, strata_size)?;
        let per = self.nx * self.ny;
        if let Some(&c) = areas.iter().flatten().find(|&&c| c >= per) {
            return Err(Error::invalid(format!("cell {c} is outside the {per} cells of the grid")));
        }
        let rows = areas
            .par_iter()
            .map(|cells| {
                let sum = |v: &[f64]| -> Vec<f64> { (0..self.nz + 2).map(|l| cells.iter().map(|&c| v[l * per + c]).sum()).collect() };
                column_metrics(&sum(&self.weight), &sum(&self.weight_k), self.min_height, self.bin_size, strata_size)
            })
            .collect();
        Ok((names, rows))
    }
}

// ------------------------------------------------------------ ray tracing

/// Voxel fields of a whole catalogue on one lattice.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CatalogVoxels {
    pub origin: Point,
    pub voxel_size: f64,
    /// `[nx, ny, nz]`.
    pub shape: [usize; 3],
    /// Each requested field or metric, `(nz, ny, nx)` flattened, NaN in
    /// columns no chunk traced.
    pub fields: Vec<(String, Vec<f64>)>,
    /// Pulses traced, summed over chunks (a pulse in several chunks' reach
    /// counts in each).
    pub n_pulses: usize,
    /// Pulses left out for want of a trajectory position.
    pub n_unpositioned: usize,
    /// Largest horizontal distance (m) a traced pulse covered between the
    /// top of the grid and its last return: a buffer at least this wide
    /// brings every pulse that crosses a chunk's core into the chunk.
    pub reach: f64,
}

/// Bytes a traced voxel holds while tracing, for sizing the work.
const BYTES_PER_VOXEL: u64 = 400;

/// Everything [`voxelize_chunk`] needs besides the points.
#[derive(Debug, Clone)]
pub struct TraceSettings<'a> {
    pub trajectory: &'a Trajectory,
    pub pulses: PulseParams,
    pub voxel: VoxelParams,
    pub labels: EchoLabels,
    pub dtm: Option<&'a Raster>,
    pub fields: Vec<String>,
}

/// The lattice: `origin`, and `[nx, ny, nz]`.
fn lattice(b: [f64; 6], vs: f64) -> Result<(Point, [usize; 3])> {
    if !(vs.is_finite() && vs > 0.0) {
        return Err(Error::invalid(format!("voxel_size must be a positive number of metres, got {vs}")));
    }
    let o = [(b[0] / vs).floor() * vs, (b[1] / vs).floor() * vs, (b[2] / vs).floor() * vs];
    let n = [((b[3] - o[0]) / vs).floor() as usize + 1, ((b[4] - o[1]) / vs).floor() as usize + 1, ((b[5] - o[2]) / vs).floor() as usize + 1];
    Ok((o, n))
}

/// Trace the pulses of `cloud` on the cells `[i0, i1) x [j0, j1)` of the
/// lattice (all layers) and return the requested fields for them, the
/// pulses traced and left out, and the reach.
#[allow(clippy::type_complexity)]
fn voxelize_part(cloud: &PointCloud, s: &TraceSettings, origin: Point, nz: usize, cells: [usize; 4]) -> Result<Option<(Vec<Vec<f64>>, usize, usize, f64)>> {
    let (shots, report) = reconstruct(cloud, s.trajectory, &s.pulses)?;
    if shots.n_shots() == 0 {
        return Ok(None);
    }
    let vs = s.voxel.voxel_size;
    let [i0, i1, j0, j1] = cells;
    let lo = [origin[0] + i0 as f64 * vs, origin[1] + j0 as f64 * vs, origin[2]];
    // Voxelizer takes ceil((hi - lo) / size) voxels: half a voxel short of the edge is exact.
    let hi = [lo[0] + ((i1 - i0) as f64 - 0.5) * vs, lo[1] + ((j1 - j0) as f64 - 0.5) * vs, lo[2] + (nz as f64 - 0.5) * vs];
    let mut params = s.voxel.clone();
    params.bounds = Some((lo, hi));
    let vox = voxel_grid::voxelize_labelled(&shots, &params, &s.labels, None, None, s.dtm)?;
    debug_assert_eq!(vox.shape, [i1 - i0, j1 - j0, nz]);
    let fields = s.fields.iter().map(|f| vox.values(f)).collect::<Result<Vec<_>>>()?;
    let top = lo[2] + nz as f64 * vs;
    let mut reach = 0.0f64;
    for k in 0..shots.n_shots() {
        let c = shots.echo_count[k] as usize;
        if c == 0 {
            continue;
        }
        let d = shots.direction[k];
        let e = shots.echo_start[k] + c - 1;
        let z_last = shots.origin[k][2] + d[2] * shots.echo_range[e];
        if d[2] < 0.0 && top > z_last {
            reach = reach.max((top - z_last) * (d[0] * d[0] + d[1] * d[1]).sqrt() / -d[2]);
        }
    }
    Ok(Some((fields, shots.n_shots(), report.n_unpositioned, reach)))
}

fn check_fields(fields: &[String]) -> Result<()> {
    if fields.is_empty() {
        return Err(Error::invalid("ask for at least one voxel field"));
    }
    Ok(())
}

/// Ray-traced voxels of one cloud on a lattice anchored at multiples of the
/// voxel size, over `bounds` (`[xmin, ymin, zmin, xmax, ymax, zmax]`, by
/// default the cloud's extent).
pub fn voxelize_cloud(cloud: &PointCloud, s: &TraceSettings, bounds: Option<[f64; 6]>) -> Result<CatalogVoxels> {
    check_fields(&s.fields)?;
    let b = match bounds {
        Some(b) => b,
        None => {
            let (lo, hi) = cloud.bounds().ok_or_else(|| Error::invalid("the cloud is empty"))?;
            [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
        }
    };
    let vs = s.voxel.voxel_size;
    let (origin, shape) = lattice(b, vs)?;
    let mut out = CatalogVoxels { origin, voxel_size: vs, shape, ..Default::default() };
    match voxelize_part(cloud, s, origin, shape[2], [0, shape[0], 0, shape[1]])? {
        Some((f, n, u, reach)) => {
            out.fields = s.fields.iter().cloned().zip(f).collect();
            out.n_pulses = n;
            out.n_unpositioned = u;
            out.reach = reach;
        }
        None => {
            let n = shape.iter().product();
            out.fields = s.fields.iter().map(|f| (f.clone(), vec![f64::NAN; n])).collect();
        }
    }
    Ok(out)
}

/// Distance from `(x, y)` to a half-open core box: 0 inside.
fn dist_to_core(b: &[f64; 4], x: f64, y: f64) -> f64 {
    if crate::als::in_core(b, x, y) {
        return 0.0;
    }
    let dx = (b[0] - x).max(0.0).max(x - b[2]);
    let dy = (b[1] - y).max(0.0).max(y - b[3]);
    (dx * dx + dy * dy).sqrt().max(f64::MIN_POSITIVE)
}

/// Ray-traced voxels of a whole catalogue. The lattice is anchored at the
/// catalogue's minimum corner snapped down to a multiple of the voxel size
/// and spans `z_range` (by default the catalogue's z range from the
/// headers). Each chunk reconstructs the pulses of its core and buffer
/// points and traces them on the columns around its core; each column is
/// taken from the chunk whose (half-open) core is nearest its centre. A pulse crossing a core is
/// traced by that chunk as long as one of its returns lies within the
/// buffer, which [`CatalogVoxels::reach`] checks.
pub fn voxelize_catalog(cat: &Catalog, s: &TraceSettings, z_range: Option<(f64, f64)>, opts: &RunOptions) -> Result<CatalogVoxels> {
    check_fields(&s.fields)?;
    let b = cat.bounds().ok_or_else(|| Error::invalid("the catalogue has no tiles"))?;
    let (z0, z1) = z_range.unwrap_or((b[2], b[5]));
    if z1.is_nan() || z0.is_nan() || z1 <= z0 {
        return Err(Error::invalid(format!("z_range must run upwards, got ({z0}, {z1})")));
    }
    let vs = s.voxel.voxel_size;
    let (origin, shape) = lattice([b[0], b[1], z0, b[3], b[4], z1], vs)?;
    let [nx, ny, nz] = shape;
    let n = nx as u128 * ny as u128 * nz as u128;
    limits::check_cells(n, 8 * s.fields.len() as u64, &format!("{} voxel field(s) of {nx} x {ny} x {nz}", s.fields.len()), "a larger voxel_size, fewer fields or a z_range")?;
    let chunks = plan(cat, opts.layout, opts.buffer)?;
    // Each chunk traces the columns whose centres lie within a voxel and a
    // metre of its core (tile extents from headers can leave slivers
    // between tiles); each column is then taken from the chunk whose core
    // is nearest its centre, as the rasters of a catalogue are.
    let grow = vs + 1.0;
    let span = |a: f64, b: f64, o: f64, n: usize| -> (usize, usize) {
        let lo = ((a - grow - o) / vs - 0.5).ceil().max(0.0) as usize;
        let hi = (((b + grow - o) / vs - 0.5).ceil().max(0.0) as usize).min(n);
        (lo.min(n), hi)
    };
    let cells: Vec<[usize; 4]> = chunks.iter().map(|c| {
        let (i0, i1) = span(c.core[0], c.core[2], origin[0], nx);
        let (j0, j1) = span(c.core[1], c.core[3], origin[1], ny);
        [i0, i1, j0, j1]
    }).collect();
    // A chunk holds its points and its voxels.
    let est: Vec<u64> = chunks.iter().zip(&cells).map(|(c, k)| c.est_points + ((k[1] - k[0]) as u64 * (k[3] - k[2]) as u64 * nz as u64 * BYTES_PER_VOXEL).div_ceil(BYTES_PER_POINT)).collect();
    let w = workers_for_estimates(&est, opts.workers, BYTES_PER_POINT)?;
    let parts = run(cat, &chunks, w, "tracing ALS chunks", |chunk, data| {
        let k = cells[chunk.index];
        if k[1] <= k[0] || k[3] <= k[2] {
            return Ok(None);
        }
        voxelize_part(&data.cloud, s, origin, nz, k)
    })?;
    let mut out = CatalogVoxels { origin, voxel_size: vs, shape, ..Default::default() };
    let mut fields: Vec<Vec<f64>> = s.fields.iter().map(|_| vec![f64::NAN; n as usize]).collect();
    let mut best = vec![f64::INFINITY; nx * ny];
    for (ci, part) in parts.into_iter().enumerate() {
        let Some((f, np, nu, reach)) = part.flatten() else { continue };
        out.n_pulses += np;
        out.n_unpositioned += nu;
        out.reach = out.reach.max(reach);
        let [i0, i1, j0, j1] = cells[ci];
        let (cw, ch) = (i1 - i0, j1 - j0);
        let core = chunks[ci].core;
        for j in 0..ch {
            for i in 0..cw {
                // Nearest core wins (zero inside, half-open); ties go to the earlier chunk.
                let col = (j0 + j) * nx + i0 + i;
                let (x, y) = (origin[0] + ((i0 + i) as f64 + 0.5) * vs, origin[1] + ((j0 + j) as f64 + 0.5) * vs);
                let d = dist_to_core(&core, x, y);
                if d >= best[col] {
                    continue;
                }
                best[col] = d;
                for (dst, src) in fields.iter_mut().zip(&f) {
                    for k in 0..nz {
                        dst[k * nx * ny + col] = src[k * cw * ch + j * cw + i];
                    }
                }
            }
        }
    }
    out.fields = s.fields.iter().cloned().zip(fields).collect();
    Ok(out)
}

/// Mean of `values` per height bin of `bin_size` over the voxels with at
/// least `min_beams` pulses (`beams`), heights being those of the voxel
/// centres above `dtm` (above the grid floor without one). Returns the
/// lower edge of each bin from 0 and the means (NaN for empty bins).
pub fn height_profile(v: &CatalogVoxels, values: &[f64], beams: &[f64], dtm: Option<&Raster>, bin_size: f64, min_beams: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    let [nx, ny, nz] = v.shape;
    let n = nx * ny * nz;
    if values.len() != n || beams.len() != n {
        return Err(Error::invalid("values and beams must be per-voxel arrays of the grid"));
    }
    if !(bin_size.is_finite() && bin_size > 0.0) {
        return Err(Error::invalid(format!("bin_size must be positive, got {bin_size}")));
    }
    let vs = v.voxel_size;
    let ground: Vec<f64> = (0..nx * ny).map(|c| match dtm {
        Some(d) => d.sample(v.origin[0] + ((c % nx) as f64 + 0.5) * vs, v.origin[1] + ((c / nx) as f64 + 0.5) * vs),
        None => v.origin[2],
    }).collect();
    let mut sums: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
    for k in 0..nz {
        let zc = v.origin[2] + (k as f64 + 0.5) * vs;
        for (c, g) in ground.iter().enumerate() {
            let i = k * nx * ny + c;
            if beams[i].is_nan() || beams[i] < min_beams || !values[i].is_finite() || !g.is_finite() {
                continue;
            }
            let h = zc - g;
            if h < 0.0 {
                continue;
            }
            let e = sums.entry((h / bin_size).floor() as i64).or_insert((0.0, 0));
            e.0 += values[i];
            e.1 += 1;
        }
    }
    let top = sums.keys().next_back().copied().unwrap_or(-1);
    let heights = (0..=top).map(|b| b as f64 * bin_size).collect();
    let means = (0..=top).map(|b| sums.get(&b).map_or(f64::NAN, |(s, c)| s / *c as f64)).collect();
    Ok((heights, means))
}

/// Column sums of `values x voxel_size` over the voxels with at least
/// `min_beams` pulses and centres at least `min_height` above `dtm`
/// (anywhere without one): the plant area index of each column from a PAD
/// field. `(ny, nx)`; NaN for columns with no such voxel.
pub fn column_sums(v: &CatalogVoxels, values: &[f64], beams: &[f64], dtm: Option<&Raster>, min_height: f64, min_beams: f64) -> Result<Vec<f64>> {
    let [nx, ny, nz] = v.shape;
    let n = nx * ny * nz;
    if values.len() != n || beams.len() != n {
        return Err(Error::invalid("values and beams must be per-voxel arrays of the grid"));
    }
    let vs = v.voxel_size;
    Ok((0..nx * ny)
        .into_par_iter()
        .map(|c| {
            let g = dtm.map_or(f64::NEG_INFINITY, |d| d.sample(v.origin[0] + ((c % nx) as f64 + 0.5) * vs, v.origin[1] + ((c / nx) as f64 + 0.5) * vs));
            let mut sum = 0.0;
            let mut any = false;
            for k in 0..nz {
                let i = k * nx * ny + c;
                let zc = v.origin[2] + (k as f64 + 0.5) * vs;
                if beams[i] >= min_beams && values[i].is_finite() && zc - g >= min_height {
                    sum += values[i] * vs;
                    any = true;
                }
            }
            if any { sum } else { f64::NAN }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointcloud::Attr;

    fn traj() -> Trajectory {
        let time: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
        let xyz = time.iter().map(|&t| [10.0 * t, 0.0, 100.0]).collect();
        Trajectory::new(time, xyz, None, None, None).unwrap()
    }

    /// Pulses every 1 ms along the track, straight down to z = 0, with a
    /// second return 5 m up on every other pulse.
    fn cloud(skip: &[usize]) -> PointCloud {
        let mut xyz = Vec::new();
        let (mut t, mut rn, mut nr) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..200usize {
            if skip.contains(&i) {
                continue;
            }
            let ti = 1.0 + i as f64 * 0.001;
            let x = 10.0 * ti;
            let a = ((i % 50) as f64 / 49.0 - 0.5) * 0.4;
            let (dx, dz) = (a.sin(), -a.cos());
            let ground = [x + 100.0 * dx / -dz, 0.0, 0.0];
            let two = i % 2 == 0;
            if two {
                xyz.push([x + 95.0 * dx / -dz, 0.0, 5.0]);
                t.push(ti);
                rn.push(1u8);
                nr.push(2u8);
            }
            xyz.push(ground);
            t.push(ti);
            rn.push(if two { 2 } else { 1 });
            nr.push(if two { 2 } else { 1 });
        }
        let mut c = PointCloud::new(xyz);
        c.attrs.insert("gps_time".into(), Attr::F64(t));
        c.attrs.insert("return_number".into(), Attr::U8(rn));
        c.attrs.insert("number_of_returns".into(), Attr::U8(nr));
        c
    }

    #[test]
    fn pulses_start_at_the_sensor() {
        let (s, r) = reconstruct(&cloud(&[]), &traj(), &PulseParams::default()).unwrap();
        assert_eq!(s.n_shots(), 200);
        assert_eq!(s.n_echoes(), 300);
        assert_eq!(r.n_incomplete, 0);
        assert!(r.line_offset_median < 1e-6, "{r:?}");
        assert!((r.pulse_interval - 0.001).abs() < 1e-9);
        let xyz = s.echo_xyz();
        let c = cloud(&[]);
        // Every return comes back where it was.
        let mut a: Vec<[i64; 3]> = xyz.iter().map(|p| p.map(|v| (v * 1e6).round() as i64)).collect();
        let mut b: Vec<[i64; 3]> = c.xyz.iter().map(|p| p.map(|v| (v * 1e6).round() as i64)).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn missing_returns_and_pulses_are_found() {
        // Drop the second return of pulse 10 and pulses 20..23 entirely.
        let mut c = cloud(&[21, 22, 23]);
        let t = c.attr_f64("gps_time").unwrap();
        let drop: Vec<bool> = (0..c.len()).map(|i| (t[i] - 1.010).abs() < 1e-9 && c.xyz[i][2] == 0.0).collect();
        c = c.filter(&drop.iter().map(|d| !d).collect::<Vec<_>>());
        let p = PulseParams { fill_missing: true, ..Default::default() };
        let (s, r) = reconstruct(&c, &traj(), &p).unwrap();
        assert_eq!(r.n_incomplete, 1);
        assert_eq!(r.n_missing_returns, 1);
        assert_eq!(r.n_filled, 3);
        assert_eq!(s.n_shots(), 200);
        assert_eq!(s.echo_count.iter().filter(|&&c| c == 0).count(), 3);
        let dropped = reconstruct(&c, &traj(), &PulseParams { drop_incomplete: true, ..Default::default() }).unwrap();
        assert_eq!(dropped.0.n_shots(), 196);
    }

    #[test]
    fn a_wrong_clock_is_explained() {
        let mut c = cloud(&[]);
        let t: Vec<f64> = c.attr_f64("gps_time").unwrap().iter().map(|v| v + 1e6).collect();
        c.attrs.insert("gps_time".into(), Attr::F64(t));
        assert!(reconstruct(&c, &traj(), &PulseParams::default()).is_err());
    }

    #[test]
    fn a_turbid_column_gives_its_density() {
        // 10 000 vertical pulses into a layer of PAD 0.5 between 10 and 20 m (G = 0.5).
        let (pad, g) = (0.5, 0.5);
        let n = 10_000;
        let mut xyz = Vec::new();
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            // Depth of the first hit into the layer, by inverting the exponential.
            let depth = -u.ln() / (g * pad);
            let z = if depth < 10.0 { 20.0 - depth } else { 0.0 };
            xyz.push([0.5, 0.5, z]);
        }
        let c = PointCloud::new(xyz.clone());
        let h: Vec<f64> = xyz.iter().map(|p| p[2]).collect();
        let p = ProfileParams { resolution: 1.0, bin_size: 2.0, weighting: ReturnWeight::All, max_height: Some(24.0), ..Default::default() };
        let grid = profile_cloud(&c, &h, &p, None).unwrap();
        let pads = grid.pad();
        // Layers 1-9 m, ..., the 10-20 m layers hold the canopy.
        for (l, v) in pads.iter().enumerate() {
            let lo = 1.0 + 2.0 * l as f64;
            let want = if (11.0..19.0).contains(&lo) { pad } else if lo + 2.0 <= 10.0 || lo >= 21.0 { 0.0 } else { f64::NAN };
            if want.is_finite() {
                assert!((v - want).abs() < 0.02, "layer {lo}: {v}");
            }
        }
        let pai = grid.pai()[0];
        assert!((pai - pad * 10.0).abs() < 0.1, "{pai}");
    }

    #[test]
    fn oblique_beams_are_corrected() {
        // The same layer seen at 30 degrees: -ln P doubles with 1 / cos, the estimate does not.
        let (pad, g, th) = (0.3, 0.5, 30f64.to_radians());
        let n = 20_000;
        let mut xyz = Vec::new();
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            let path = -u.ln() / (g * pad);
            let depth = path * th.cos();
            xyz.push([5.0, 5.0, if depth < 10.0 { 20.0 - depth } else { 0.0 }]);
        }
        let mut c = PointCloud::new(xyz.clone());
        c.attrs.insert("scan_angle".into(), Attr::F32(vec![30.0; n]));
        let h: Vec<f64> = xyz.iter().map(|p| p[2]).collect();
        let base = ProfileParams { resolution: 20.0, bin_size: 1.0, weighting: ReturnWeight::All, ..Default::default() };
        let corrected = profile_cloud(&c, &h, &ProfileParams { angles: Angles::ScanAngle, ..base.clone() }, None).unwrap().pai()[0];
        let naive = profile_cloud(&c, &h, &base, None).unwrap().pai()[0];
        assert!((corrected - 3.0).abs() < 0.05, "{corrected}");
        assert!((naive - 3.0 / th.cos()).abs() < 0.05, "{naive}");
    }

    #[test]
    fn saturated_and_empty_layers() {
        let w = [0.0, 0.0, 4.0, 0.0];
        let k = [0.0, 0.0, 2.0, 0.0];
        let pad = column_pad(&w, &k, 1.0);
        assert!(pad[0].is_nan());
        assert!((pad[1] - (8.0f64).ln() / 0.5).abs() < 1e-12);
        assert!(column_pai(&[0.0; 4], &[0.0; 4]).is_nan());
        assert_eq!(column_pgap(&[1.0, 1.0, 2.0, 0.0]), vec![0.25, 0.5, 1.0]);
    }

    #[test]
    fn strata_keep_every_layer_once() {
        let st = strata(1.0, 1.0, 12, 5.0);
        let spans: Vec<(f64, f64, usize, usize)> = st.iter().map(|s| (s.lo, s.hi, s.first, s.last)).collect();
        assert_eq!(spans, vec![(0.0, 5.0, 1, 4), (5.0, 10.0, 5, 9), (10.0, 15.0, 10, 12)]);
        // A stratum below min_height has no layer and no metric.
        assert_eq!(strata(6.0, 1.0, 4, 5.0)[0].lo, 5.0);
        let names = profile_metric_names(1.0, 1.0, 12, 5.0).unwrap();
        assert_eq!(&names[PROFILE_SCALARS..PROFILE_SCALARS + 3], ["pavd_0_5", "pavd_5_10", "pavd_10_15"]);
        assert!(names.contains(&"pai_above_5".to_string()) && names.contains(&"cover_above_1".to_string()));
        assert!(profile_metric_names(1.0, 1.0, 12, 0.5).is_err());
    }

    /// Value of metric `name` in a column of `nz` layers of 1 m from 0 m,
    /// strata of 1 m; vertical beams with G = 0.5.
    fn metric(weight: &[f64], name: &str) -> f64 {
        let k: Vec<f64> = weight.iter().map(|w| w * 0.5).collect();
        let names = profile_metric_names(0.0, 1.0, weight.len() - 2, 1.0).unwrap();
        column_metrics(weight, &k, 0.0, 1.0, 1.0)[names.iter().position(|n| n == name).unwrap()]
    }

    #[test]
    fn one_dense_layer() {
        // Half the pulses stop in the second layer, none in the first.
        let w = [50.0, 0.0, 50.0, 0.0];
        let pad = 2.0 * 2f64.ln();
        assert_eq!(metric(&w, "pulses"), 100.0);
        assert!((metric(&w, "pai") - pad).abs() < 1e-12);
        assert!((metric(&w, "cover") - 0.5).abs() < 1e-12);
        assert_eq!(metric(&w, "fhd"), 0.0);
        assert!((metric(&w, "pad_max") - pad).abs() < 1e-12);
        assert_eq!(metric(&w, "height_pad_max"), 1.5);
        assert_eq!(metric(&w, "height_pad_mean"), 1.5);
        assert_eq!(metric(&w, "height_pad_sd"), 0.0);
        assert_eq!(metric(&w, "pavd_0_1"), 0.0);
        assert!((metric(&w, "pavd_1_2") - pad).abs() < 1e-12);
        assert!((metric(&w, "pai_above_1") - pad).abs() < 1e-12);
        assert!((metric(&w, "cover_above_1") - 0.5).abs() < 1e-12);
    }

    #[test]
    fn equal_plant_area_in_two_layers() {
        // Each layer stops half of what reaches it: equal density in both.
        let w = [25.0, 25.0, 50.0, 0.0];
        assert!((metric(&w, "fhd") - 2f64.ln()).abs() < 1e-12);
        assert_eq!(metric(&w, "height_pad_mean"), 1.0);
        assert!((metric(&w, "height_pad_sd") - 0.5).abs() < 1e-12);
        assert!((metric(&w, "pai_above_0") - 2.0 * metric(&w, "pai_above_1")).abs() < 1e-12);
        assert!((metric(&w, "cover_above_1") - 0.5).abs() < 1e-12);
    }

    #[test]
    fn open_and_empty_columns() {
        // Every pulse reached the ground: no plant area, no heights.
        let w = [10.0, 0.0, 0.0, 0.0];
        assert_eq!(metric(&w, "pai"), 0.0);
        assert_eq!(metric(&w, "fhd"), 0.0);
        assert!(metric(&w, "height_pad_mean").is_nan());
        let empty = [0.0; 4];
        assert_eq!(metric(&empty, "pulses"), 0.0);
        assert!(metric(&empty, "pai").is_nan() && metric(&empty, "fhd").is_nan());
    }

    #[test]
    fn pooled_areas_add_their_cells() {
        let mut g = ProfileGrid { nx: 2, ny: 1, nz: 2, bin_size: 1.0, resolution: 1.0, ..Default::default() };
        // Layers are the slowest axis: (nz + 2, ny, nx).
        g.weight = vec![50.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 100.0];
        g.weight_k = g.weight.iter().map(|w| w * 0.5).collect();
        let (names, rows) = g.area_metrics(&[vec![0], vec![0, 1]], 1.0).unwrap();
        let pulses = names.iter().position(|n| n == "pulses").unwrap();
        assert_eq!((rows[0][pulses], rows[1][pulses]), (100.0, 200.0));
        assert!(g.area_metrics(&[vec![2]], 1.0).is_err());
        let (_, per_cell) = g.metrics(1.0).unwrap();
        assert_eq!(per_cell.len(), names.len() * 2);
    }

    #[test]
    fn stray_returns_and_noise_do_not_stretch_the_profile() {
        // A turbid layer from 10 to 20 m, as above, and a bird at 80 m.
        let n = 10_000;
        let mut xyz = Vec::new();
        for i in 0..n {
            let depth = -((i as f64 + 0.5) / n as f64).ln() / 0.25;
            xyz.push([0.5, 0.5, if depth < 10.0 { 20.0 - depth } else { 0.0 }]);
        }
        xyz.push([0.5, 0.5, 80.0]);
        let mut c = PointCloud::new(xyz.clone());
        let h: Vec<f64> = xyz.iter().map(|p| p[2]).collect();
        let p = ProfileParams { resolution: 1.0, weighting: ReturnWeight::All, top_quantile: 0.999, ..Default::default() };
        let full = profile_cloud(&c, &h, &ProfileParams { top_quantile: 1.0, ..p.clone() }, None).unwrap();
        let trimmed = profile_cloud(&c, &h, &p, None).unwrap();
        // Layers of 1 m from 1 m: 19 reach the canopy top at 20 m, 80 the bird.
        assert_eq!(full.nz, 80);
        // The top canopy layer (19-20 m) holds more than the 0.1 % allowed, so it stays.
        assert_eq!(trimmed.nz, 19);
        // The layers kept are unchanged, and the bird is still counted above them.
        assert_eq!(&full.pad()[..trimmed.nz], &trimmed.pad()[..]);
        assert_eq!(trimmed.weight.iter().sum::<f64>(), full.weight.iter().sum::<f64>());
        assert!((trimmed.pai()[0] - 5.0).abs() < 0.1, "{}", trimmed.pai()[0]);
        // As noise (class 7) the bird is left out, and nothing is trimmed.
        let mut cls = vec![1u8; n + 1];
        cls[n] = 7;
        c.attrs.insert("classification".into(), Attr::U8(cls));
        let clean = profile_cloud(&c, &h, &ProfileParams { top_quantile: 1.0, ..p.clone() }, None).unwrap();
        assert_eq!(clean.nz, 19);
        assert_eq!(clean.weight.iter().sum::<f64>(), n as f64);
        let kept = profile_cloud(&c, &h, &ProfileParams { top_quantile: 1.0, drop_noise: false, ..p.clone() }, None).unwrap();
        assert_eq!(kept.nz, 80);
        assert!(profile_cloud(&c, &h, &ProfileParams { top_quantile: 0.0, ..p }, None).is_err());
    }
}
