// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Change between the quantitative structure models of two epochs.
//!
//! [`compare_qsms`] takes two cylinder models of one tree, in one frame (the
//! epochs aligned beforehand), and reports:
//!
//! * the stem radius change by height (the taper increment), from a line
//!   fitted to each model's measured stem radii in every height bin, with an
//!   uncertainty from the scatter about those lines;
//! * branch matching by base position, direction and parent, giving matched
//!   branches with their growth, lost branches and new branches;
//! * volume change split into the stem and each branch order, and into a
//!   trusted part (where both models fitted that part to points) and an
//!   untrusted remainder (parts whose radius came from the taper and
//!   pipe-model priors, or that only one model measured);
//! * height, DBH and crown change.
//!
//! A cylinder counts as measured when its `n_points` is positive, as in
//! [`crate::qsm::metrics`]. With a ray-traced state grid of the later epoch
//! ([`StateGrid`]), a lost branch whose space the later scans did not observe
//! is reported as [`BranchStatus::Unobserved`], not lost, and one whose space
//! holds returns as [`BranchStatus::Present`] (there, but not modelled). A
//! grid of the earlier epoch checks new branches the same way.
//!
//! [`compare_plot`] runs the comparison over the trees of a plot, given a
//! tree match (survivor pairs, deaths and recruits), and tabulates the
//! changes with plot totals.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::path::Path;

use rayon::prelude::*;

use crate::qsm::metrics::{branches, chains, tree_metrics};
use crate::qsm::{Cylinder, Qsm};
use crate::transform::{add, dot, norm, scale, sub};
use crate::{Error, Point, Result};

/// Voxel states of a ray-traced grid (as `sylva.voxels.STATES`).
pub const UNOBSERVED: u8 = 0;
pub const OCCLUDED: u8 = 1;
pub const EMPTY: u8 = 2;
pub const FILLED: u8 = 3;

/// Voxel states of a ray-traced grid: 0 unobserved, 1 occluded (reached
/// only by pulses already stopped), 2 empty (crossed, no return), 3 filled.
#[derive(Debug, Clone, PartialEq)]
pub struct StateGrid {
    /// Minimum corner of the grid.
    pub origin: Point,
    pub voxel_size: f64,
    /// `[nx, ny, nz]`.
    pub shape: [usize; 3],
    /// One state per voxel, x fastest then y then z (a C-ordered
    /// `(nz, ny, nx)` array).
    pub state: Vec<u8>,
}

impl StateGrid {
    /// A grid, checked: positive voxel size, finite origin, one state per voxel.
    pub fn new(origin: Point, voxel_size: f64, shape: [usize; 3], state: Vec<u8>) -> Result<Self> {
        if !(voxel_size > 0.0 && voxel_size.is_finite()) {
            return Err(Error::invalid(format!("voxel_size must be positive, got {voxel_size}")));
        }
        if origin.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("grid origin must be finite"));
        }
        let n = shape[0] * shape[1] * shape[2];
        if state.len() != n {
            return Err(Error::invalid(format!("the grid has {n} voxels but {} states", state.len())));
        }
        Ok(StateGrid { origin, voxel_size, shape, state })
    }

    /// State of the voxel holding `p`; unobserved outside the grid.
    pub fn at(&self, p: &Point) -> u8 {
        let mut ijk = [0usize; 3];
        for k in 0..3 {
            let f = ((p[k] - self.origin[k]) / self.voxel_size).floor();
            if !(f >= 0.0 && f < self.shape[k] as f64) {
                return UNOBSERVED;
            }
            ijk[k] = f as usize;
        }
        self.state[(ijk[2] * self.shape[1] + ijk[1]) * self.shape[0] + ijk[0]]
    }
}

/// Settings of [`compare_qsms`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompareParams {
    /// Height bins of the stem profile (m).
    pub height_step: f64,
    /// Farthest two branch bases may be apart to match (m).
    pub max_base_distance: f64,
    /// Largest angle between two branch directions to match (deg).
    pub max_angle: f64,
    /// Added to a pair's matching cost when their parents are not matched to
    /// each other (the cost is base distance over `max_base_distance` plus
    /// angle over `max_angle`).
    pub parent_penalty: f64,
    /// Length of branch past its first cylinder over which its direction is
    /// taken (m).
    pub direction_reach: f64,
    /// Share of a stem bin's length, or of a branch's length, that must be
    /// measured in a model for that part to be trusted.
    pub min_measured: f64,
    /// Measured stem cylinders a bin needs in each model to be trusted.
    pub min_fits: usize,
    /// Radius uncertainty of one fit added in quadrature (m).
    pub radius_sigma: f64,
    /// Bins further than this many of their own uncertainties from the mean
    /// increment are left out of it.
    pub clip: f64,
    /// Share of a branch's samples a grid must have observed; below it the
    /// branch is unobserved.
    pub min_observed: f64,
    /// Share of the observed samples holding returns above which the branch
    /// is present rather than lost (or new).
    pub max_filled: f64,
    /// A model's height is trusted when a measured cylinder reaches within
    /// this distance of its top (m).
    pub top_band: f64,
    /// As in [`tree_metrics`].
    pub crown_branch_length: f64,
    pub crown_slice: f64,
}

impl Default for CompareParams {
    fn default() -> Self {
        CompareParams {
            height_step: 1.0,
            max_base_distance: 0.5,
            max_angle: 35.0,
            parent_penalty: 1.0,
            direction_reach: 1.0,
            min_measured: 0.5,
            min_fits: 3,
            radius_sigma: 0.001,
            clip: 3.0,
            min_observed: 0.5,
            max_filled: 0.5,
            top_band: 1.0,
            crown_branch_length: 1.0,
            crown_slice: 0.5,
        }
    }
}

impl CompareParams {
    /// Checks every setting, naming the first bad one.
    pub fn check(&self) -> Result<()> {
        let positive = [("height_step", self.height_step), ("max_base_distance", self.max_base_distance), ("max_angle", self.max_angle), ("crown_slice", self.crown_slice), ("clip", self.clip)];
        for (name, v) in positive {
            if !(v > 0.0 && v.is_finite()) {
                return Err(Error::invalid(format!("{name} must be positive, got {v}")));
            }
        }
        let non_negative = [("parent_penalty", self.parent_penalty), ("direction_reach", self.direction_reach), ("radius_sigma", self.radius_sigma), ("top_band", self.top_band), ("crown_branch_length", self.crown_branch_length)];
        for (name, v) in non_negative {
            if !(v >= 0.0 && v.is_finite()) {
                return Err(Error::invalid(format!("{name} must be zero or positive, got {v}")));
            }
        }
        for (name, v) in [("min_measured", self.min_measured), ("min_observed", self.min_observed), ("max_filled", self.max_filled)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(Error::invalid(format!("{name} must be between 0 and 1, got {v}")));
            }
        }
        if self.min_fits < 1 {
            return Err(Error::invalid("min_fits must be at least 1"));
        }
        Ok(())
    }
}

/// One height bin of the stem profile.
#[derive(Debug, Clone, PartialEq)]
pub struct TaperBin {
    /// Bottom and top of the bin above the base (m).
    pub z0: f64,
    pub z1: f64,
    /// Stem radius at the bin centre in each model (m), from a line through
    /// its measured radii; NaN without measured cylinders.
    pub radius_a: f64,
    pub radius_b: f64,
    /// `radius_b - radius_a` (m) and its standard uncertainty (m).
    pub increment: f64,
    pub sigma: f64,
    /// Measured stem cylinders centred in the bin.
    pub n_fits_a: usize,
    pub n_fits_b: usize,
    /// Share of the stem length in the bin that is measured.
    pub measured_a: f64,
    pub measured_b: f64,
    /// Stem volume in the bin (m3), cylinders split by height.
    pub volume_a: f64,
    pub volume_b: f64,
    /// Both models fitted the bin to points (`min_fits`, `min_measured`).
    pub fitted: bool,
    /// Fitted, and within `clip` of its uncertainty from the mean increment:
    /// the bin enters the mean increment and the trusted volume change.
    pub trusted: bool,
}

/// Two branches matched between the epochs.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchMatch {
    pub id_a: u32,
    pub id_b: u32,
    pub order_a: u32,
    pub order_b: u32,
    /// Distance between the bases (m) and angle between the directions (deg).
    pub base_distance: f64,
    pub angle: f64,
    /// The parents are matched to each other (or both are roots).
    pub parent_consistent: bool,
    pub length_a: f64,
    pub length_b: f64,
    pub volume_a: f64,
    pub volume_b: f64,
    pub mean_radius_a: f64,
    pub mean_radius_b: f64,
    /// Distance between the two tips (m).
    pub tip_shift: f64,
    /// Measured share of the length in each model.
    pub measured_a: f64,
    pub measured_b: f64,
    /// Standard uncertainty of `volume_b - volume_a` from the radius scatter (m3).
    pub volume_sigma: f64,
    /// Both models measured at least `min_measured` of the branch.
    pub trusted: bool,
}

/// What became of an unmatched branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchStatus {
    /// In the earlier model only; with a grid of the later epoch, its space
    /// was observed and empty.
    Lost,
    /// In the later model only; with a grid of the earlier epoch, its space
    /// was observed and empty there.
    New,
    /// The other epoch's grid did not observe the branch's space.
    Unobserved,
    /// The other epoch's grid has returns in the branch's space: the branch
    /// is there, but the other model does not have it.
    Present,
}

impl BranchStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            BranchStatus::Lost => "lost",
            BranchStatus::New => "new",
            BranchStatus::Unobserved => "unobserved",
            BranchStatus::Present => "present",
        }
    }
}

/// A branch in one model without a match in the other.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchChange {
    pub id: u32,
    pub order: u32,
    /// Branch it grows from (-1 for a root).
    pub parent: i64,
    pub base: Point,
    pub length: f64,
    pub volume: f64,
    pub measured: f64,
    pub status: BranchStatus,
    /// Share of samples along the branch the other epoch's grid observed,
    /// and share of those holding returns; NaN without a grid.
    pub observed_share: f64,
    pub filled_share: f64,
    /// Standard uncertainty of the volume from the radius scatter (m3).
    pub volume_sigma: f64,
    /// Measured and confirmed: status lost (or new) with at least
    /// `min_measured` of its length measured.
    pub trusted: bool,
}

/// Volume change of one branch order (0 the stem).
#[derive(Debug, Clone, PartialEq)]
pub struct OrderChange {
    pub order: u32,
    pub volume_a: f64,
    pub volume_b: f64,
    /// `volume_b - volume_a` split into the part both models measured and
    /// the rest.
    pub trusted_change: f64,
    pub untrusted_change: f64,
}

/// Result of [`compare_qsms`].
#[derive(Debug, Clone, PartialEq)]
pub struct QsmChange {
    /// Height of the stem base shared by both profiles (the earlier model's).
    pub base_z: f64,
    pub height_a: f64,
    pub height_b: f64,
    pub height_trusted: bool,
    pub dbh_a: f64,
    pub dbh_b: f64,
    pub dbh_trusted: bool,
    pub crown_area_a: f64,
    pub crown_area_b: f64,
    pub crown_volume_a: f64,
    pub crown_volume_b: f64,
    pub crown_trusted: bool,
    pub volume_a: f64,
    pub volume_b: f64,
    /// Measured share of each model's volume.
    pub measured_volume_a: f64,
    pub measured_volume_b: f64,
    /// `volume_b - volume_a` = `trusted_change + untrusted_change` (m3).
    pub trusted_change: f64,
    pub untrusted_change: f64,
    /// Standard uncertainty of `trusted_change` (m3).
    pub trusted_sigma: f64,
    pub orders: Vec<OrderChange>,
    pub taper: Vec<TaperBin>,
    /// Weighted mean increment over the trusted bins (m), its uncertainty,
    /// and the number of bins; NaN without trusted bins.
    pub taper_increment: f64,
    pub taper_sigma: f64,
    pub n_taper_bins: usize,
    pub matched: Vec<BranchMatch>,
    pub lost: Vec<BranchChange>,
    pub new: Vec<BranchChange>,
}

impl QsmChange {
    /// Total volume change (m3).
    pub fn change(&self) -> f64 {
        self.volume_b - self.volume_a
    }
}

/// One branch with the geometry the matching needs.
#[derive(Debug, Clone)]
struct Geom {
    id: u32,
    order: u32,
    parent: i64,
    base: Point,
    dir: Point,
    tip: Point,
    cylinders: Vec<usize>,
    length: f64,
    volume: f64,
    mean_radius: f64,
    measured: f64,
    /// Scatter of the measured radii about a line along the branch (m); NaN
    /// with fewer than three.
    scatter: f64,
    /// Radius of the cylinder the branch grows from (0 for a root).
    parent_radius: f64,
}

/// Least-squares line through `(x, y)`: value at `x = 0` and the residual
/// standard deviation (`n - 2` degrees of freedom); the mean and NaN for
/// fewer than three points or no spread in x.
fn line_at_zero(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / nf;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / nf;
    if n < 3 {
        return (my, f64::NAN);
    }
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let slope = if sxx > 1e-12 { sxy / sxx } else { 0.0 };
    let dof = if sxx > 1e-12 { nf - 2.0 } else { nf - 1.0 };
    let ss: f64 = pts.iter().map(|p| (p.1 - my - slope * (p.0 - mx)).powi(2)).sum();
    (my - slope * mx, (ss / dof).sqrt())
}

fn angle_deg(a: &Point, b: &Point) -> f64 {
    let c = dot(a, b) / (norm(a) * norm(b)).max(1e-18);
    c.clamp(-1.0, 1.0).acos().to_degrees()
}

fn geometry(q: &Qsm, reach: f64) -> Vec<Geom> {
    let cyl = &q.cylinders;
    let table = branches(q);
    let chains = chains(cyl);
    table
        .into_iter()
        .zip(chains)
        .map(|(b, (id, order))| {
            debug_assert_eq!(b.id, id);
            let first = &cyl[order[0]];
            // Direction over `reach` past the first cylinder, which only joins
            // the branch to its parent's axis.
            let skip = usize::from(order.len() > 1);
            let from = if skip == 1 { first.end() } else { first.start };
            let mut far = cyl[*order.last().unwrap()].end();
            let mut walked = 0.0;
            for &i in order.iter().skip(skip) {
                let c = &cyl[i];
                if walked + c.length >= reach {
                    far = add(&c.start, &scale(&c.axis, (reach - walked).max(0.0)));
                    break;
                }
                walked += c.length;
            }
            let mut dir = sub(&far, &from);
            if norm(&dir).is_nan() || norm(&dir) <= 1e-9 {
                dir = first.axis;
            }
            let mut s = 0.0;
            let mut pts = Vec::new();
            for &i in &order {
                let c = &cyl[i];
                if c.n_points > 0 {
                    pts.push((s + c.length / 2.0, c.radius));
                }
                s += c.length;
            }
            let tip = cyl[*order.last().unwrap()].end();
            Geom {
                id,
                order: b.order,
                parent: b.parent,
                base: first.start,
                dir,
                tip,
                length: b.length,
                volume: b.volume,
                mean_radius: b.mean_radius,
                measured: b.measured_fraction,
                scatter: line_at_zero(&pts).1,
                parent_radius: if first.parent >= 0 { cyl[first.parent as usize].radius } else { 0.0 },
                cylinders: order,
            }
        })
        .collect()
}

/// Greedy matching, one order of the earlier model at a time so that
/// parents are matched before their children: the cheapest admissible pair
/// first. Returns `(index_a, index_b, distance, angle, parent_consistent)`.
fn match_branches(a: &[Geom], b: &[Geom], p: &CompareParams) -> Vec<(usize, usize, f64, f64, bool)> {
    let mut to_b: HashMap<u32, u32> = HashMap::new();
    let mut taken_b = vec![false; b.len()];
    let mut out = Vec::new();
    let mut levels: Vec<u32> = a.iter().map(|g| g.order).collect();
    levels.sort_unstable();
    levels.dedup();
    for level in levels {
        let mut cand: Vec<(f64, usize, usize, f64, f64, bool)> = a
            .par_iter()
            .enumerate()
            .filter(|(_, ga)| ga.order == level)
            .flat_map_iter(|(ia, ga)| {
                let to_b = &to_b;
                let taken_b = &taken_b;
                b.iter().enumerate().filter_map(move |(ib, gb)| {
                    if taken_b[ib] {
                        return None;
                    }
                    let d = norm(&sub(&ga.base, &gb.base));
                    if d.is_nan() || d > p.max_base_distance {
                        return None;
                    }
                    let ang = angle_deg(&ga.dir, &gb.dir);
                    if ang.is_nan() || ang > p.max_angle {
                        return None;
                    }
                    let consistent = match (ga.parent, gb.parent) {
                        (pa, pb) if pa < 0 && pb < 0 => true,
                        (pa, pb) if pa >= 0 && pb >= 0 => to_b.get(&(pa as u32)) == Some(&(pb as u32)),
                        _ => false,
                    };
                    let cost = d / p.max_base_distance + ang / p.max_angle + if consistent { 0.0 } else { p.parent_penalty };
                    Some((cost, ia, ib, d, ang, consistent))
                })
            })
            .collect();
        cand.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
        let mut taken_a = vec![false; a.len()];
        for (_, ia, ib, d, ang, consistent) in cand {
            if taken_a[ia] || taken_b[ib] {
                continue;
            }
            taken_a[ia] = true;
            taken_b[ib] = true;
            to_b.insert(a[ia].id, b[ib].id);
            out.push((ia, ib, d, ang, consistent));
        }
    }
    out.sort_by_key(|m| (m.0, m.1));
    out
}

/// Share of samples along a branch that `grid` observed, and share of those
/// holding returns. Samples every half voxel along the cylinder axes, leaving
/// out the part inside the parent (within its radius plus a voxel of the
/// base); the tip alone if nothing else is left.
fn observation(g: &Geom, cyl: &[Cylinder], grid: &StateGrid) -> (f64, f64) {
    let step = grid.voxel_size / 2.0;
    let skip = g.parent_radius + grid.voxel_size;
    let mut samples: Vec<Point> = Vec::new();
    for &i in &g.cylinders {
        let c = &cyl[i];
        let n = (c.length / step).ceil().max(1.0) as usize;
        for k in 0..n {
            let s = add(&c.start, &scale(&c.axis, c.length * (k as f64 + 0.5) / n as f64));
            if g.parent < 0 || norm(&sub(&s, &g.base)) > skip {
                samples.push(s);
            }
        }
    }
    if samples.is_empty() {
        samples.push(g.tip);
    }
    let mut count = [0usize; 4];
    for s in &samples {
        count[(grid.at(s) as usize).min(3)] += 1;
    }
    let observed = count[EMPTY as usize] + count[FILLED as usize];
    let filled = if observed > 0 { count[FILLED as usize] as f64 / observed as f64 } else { f64::NAN };
    (observed as f64 / samples.len() as f64, filled)
}

fn unmatched(g: &Geom, cyl: &[Cylinder], grid: Option<&StateGrid>, absent: BranchStatus, p: &CompareParams) -> BranchChange {
    let (observed_share, filled_share, status) = match grid {
        None => (f64::NAN, f64::NAN, absent),
        Some(grid) => {
            let (o, f) = observation(g, cyl, grid);
            let status = if o < p.min_observed {
                BranchStatus::Unobserved
            } else if f > p.max_filled {
                BranchStatus::Present
            } else {
                absent
            };
            (o, f, status)
        }
    };
    let scatter = if g.scatter.is_finite() { g.scatter } else { 0.0 };
    BranchChange {
        id: g.id,
        order: g.order,
        parent: g.parent,
        base: g.base,
        length: g.length,
        volume: g.volume,
        measured: g.measured,
        status,
        observed_share,
        filled_share,
        volume_sigma: 2.0 * PI * g.mean_radius * g.length * (scatter * scatter + p.radius_sigma * p.radius_sigma).sqrt(),
        trusted: status == absent && g.measured >= p.min_measured,
    }
}

/// Per-bin sums of a model's stem (order 0) cylinders: volume, length,
/// measured length, and the measured radii as `(height - bin centre, radius)`.
struct StemBins {
    volume: Vec<f64>,
    length: Vec<f64>,
    measured: Vec<f64>,
    fits: Vec<Vec<(f64, f64)>>,
}

fn stem_bins(q: &Qsm, base_z: f64, step: f64, n: usize) -> StemBins {
    let mut out = StemBins { volume: vec![0.0; n], length: vec![0.0; n], measured: vec![0.0; n], fits: vec![Vec::new(); n] };
    let bin_of = |z: f64| ((z / step).floor().max(0.0) as usize).min(n.saturating_sub(1));
    for c in q.cylinders.iter().filter(|c| c.branch_order == 0) {
        let (zs, ze) = (c.start[2] - base_z, c.end()[2] - base_z);
        let (lo, hi) = (zs.min(ze), zs.max(ze));
        if !(lo.is_finite() && hi.is_finite()) || n == 0 {
            continue;
        }
        let m = if c.n_points > 0 { c.length } else { 0.0 };
        if hi - lo < 1e-12 {
            let k = bin_of(lo);
            out.volume[k] += c.volume();
            out.length[k] += c.length;
            out.measured[k] += m;
        } else {
            for k in bin_of(lo)..=bin_of(hi) {
                let (b0, b1) = (k as f64 * step, (k + 1) as f64 * step);
                // The first and last bins take whatever lies below or above them.
                let b0 = if k == 0 { f64::NEG_INFINITY } else { b0 };
                let b1 = if k + 1 == n { f64::INFINITY } else { b1 };
                let share = (hi.min(b1) - lo.max(b0)).max(0.0) / (hi - lo);
                out.volume[k] += c.volume() * share;
                out.length[k] += c.length * share;
                out.measured[k] += m * share;
            }
        }
        if c.n_points > 0 {
            let mid = (zs + ze) / 2.0;
            let k = bin_of(mid);
            out.fits[k].push((mid - (k as f64 + 0.5) * step, c.radius));
        }
    }
    out
}

fn base_z(q: &Qsm) -> f64 {
    q.cylinders.iter().filter(|c| c.parent < 0).map(|c| c.start[2]).fold(f64::INFINITY, f64::min)
}

fn top_z(q: &Qsm) -> f64 {
    q.cylinders.iter().map(|c| c.start[2].max(c.end()[2])).fold(f64::NEG_INFINITY, f64::max)
}

/// A measured cylinder reaches within `band` of the model's top.
fn top_measured(q: &Qsm, band: f64) -> bool {
    let top = top_z(q);
    q.cylinders.iter().any(|c| c.n_points > 0 && c.start[2].max(c.end()[2]) >= top - band)
}

fn volume_by_order(q: &Qsm, n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for c in &q.cylinders {
        v[(c.branch_order as usize).min(n - 1)] += c.volume();
    }
    v
}

fn measured_share<'a>(cyl: impl Iterator<Item = &'a Cylinder>, volume: bool) -> f64 {
    let (mut all, mut measured) = (0.0, 0.0);
    for c in cyl {
        let x = if volume { c.volume() } else { c.length };
        all += x;
        if c.n_points > 0 {
            measured += x;
        }
    }
    if all > 0.0 { measured / all } else { 0.0 }
}

fn finite_or_nan(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::NAN }
}

/// Weighted mean increment over the fitted bins, weights `1 / sigma^2`,
/// iteratively leaving out bins more than `clip` of their own uncertainty
/// from the mean (a stem tip tapered to the apex radius, a fit through a
/// branch junction). The uncertainty is the weighted-mean error scaled up by
/// the Birge ratio `sqrt(chi^2 / (n - 1))` when the bins scatter more than
/// their uncertainties say. Marks the bins kept as trusted and returns
/// `(mean, sigma, n_trusted)`; NaN without fitted bins.
fn taper_mean(taper: &mut [TaperBin], clip: f64) -> (f64, f64, usize) {
    // A floor of 0.01 mm keeps a bin with exactly collinear radii and no
    // `radius_sigma` from taking all the weight.
    let w = |t: &TaperBin| 1.0 / t.sigma.max(1e-5).powi(2);
    let mean_of = |taper: &[TaperBin]| -> (f64, f64, usize) {
        let (mut sw, mut swx, mut n) = (0.0, 0.0, 0usize);
        for t in taper.iter().filter(|t| t.trusted) {
            sw += w(t);
            swx += w(t) * t.increment;
            n += 1;
        }
        if n == 0 {
            return (f64::NAN, f64::NAN, 0);
        }
        let mean = swx / sw;
        let chi2: f64 = taper.iter().filter(|t| t.trusted).map(|t| w(t) * (t.increment - mean).powi(2)).sum();
        let birge = if n >= 2 { (chi2 / (n - 1) as f64).sqrt().max(1.0) } else { 1.0 };
        (mean, birge / sw.sqrt(), n)
    };
    for t in taper.iter_mut() {
        t.trusted = t.fitted;
    }
    for _ in 0..taper.len() {
        let (mean, _, n) = mean_of(taper);
        if n <= 1 {
            break;
        }
        let mut changed = false;
        for t in taper.iter_mut().filter(|t| t.trusted) {
            if (t.increment - mean).abs() > clip * t.sigma.max(1e-5) {
                t.trusted = false;
                changed = true;
            }
        }
        if !changed || !taper.iter().any(|t| t.trusted) {
            break;
        }
    }
    if !taper.iter().any(|t| t.trusted) {
        // Every bin disagreed with the others: fall back to all fitted bins.
        for t in taper.iter_mut() {
            t.trusted = t.fitted;
        }
    }
    mean_of(taper)
}

/// Compares two models of one tree, `a` the earlier. `grid_b` (a ray-traced
/// grid of the later epoch) checks lost branches and `grid_a` new ones; see
/// the module notes.
pub fn compare_qsms(a: &Qsm, b: &Qsm, grid_a: Option<&StateGrid>, grid_b: Option<&StateGrid>, p: &CompareParams) -> Result<QsmChange> {
    p.check()?;
    a.check()?;
    b.check()?;
    let bz = if a.is_empty() { base_z(b) } else { base_z(a) };
    let ma = tree_metrics(a, p.crown_branch_length, p.crown_slice);
    let mb = tree_metrics(b, p.crown_branch_length, p.crown_slice);

    // Stem profile.
    let top = [a, b].iter().filter(|q| !q.is_empty()).flat_map(|q| q.cylinders.iter().filter(|c| c.branch_order == 0)).map(|c| c.start[2].max(c.end()[2]) - bz).fold(0.0f64, f64::max);
    let n_bins = if bz.is_finite() && top > 0.0 { (top / p.height_step).ceil().max(1.0) as usize } else { 0 };
    let sa = stem_bins(a, bz, p.height_step, n_bins);
    let sb = stem_bins(b, bz, p.height_step, n_bins);
    let mut taper = Vec::with_capacity(n_bins);
    for k in 0..n_bins {
        let (ra, sda) = line_at_zero(&sa.fits[k]);
        let (rb, sdb) = line_at_zero(&sb.fits[k]);
        let share = |m: f64, l: f64| if l > 0.0 { m / l } else { 0.0 };
        let (measured_a, measured_b) = (share(sa.measured[k], sa.length[k]), share(sb.measured[k], sb.length[k]));
        let rs2 = p.radius_sigma * p.radius_sigma;
        let sigma = (sda * sda + sdb * sdb + 2.0 * rs2).sqrt();
        let trusted = sa.fits[k].len() >= p.min_fits && sb.fits[k].len() >= p.min_fits && measured_a >= p.min_measured && measured_b >= p.min_measured && sigma.is_finite() && (rb - ra).is_finite();
        taper.push(TaperBin {
            z0: k as f64 * p.height_step,
            z1: (k + 1) as f64 * p.height_step,
            radius_a: ra,
            radius_b: rb,
            increment: rb - ra,
            sigma,
            n_fits_a: sa.fits[k].len(),
            n_fits_b: sb.fits[k].len(),
            measured_a,
            measured_b,
            volume_a: sa.volume[k],
            volume_b: sb.volume[k],
            fitted: trusted,
            trusted,
        });
    }
    let (taper_increment, taper_sigma, n_taper_bins) = taper_mean(&mut taper, p.clip);
    let mut var = 0.0;
    let mut stem_trusted = 0.0;
    for (k, t) in taper.iter().enumerate() {
        if t.trusted {
            stem_trusted += t.volume_b - t.volume_a;
            let r = (t.radius_a + t.radius_b) / 2.0;
            let l = (sa.length[k] + sb.length[k]) / 2.0;
            var += (2.0 * PI * r * l * t.sigma).powi(2);
        }
    }
    let stem_sigma = var.sqrt();

    // Branches.
    let ga = geometry(a, p.direction_reach);
    let gb = geometry(b, p.direction_reach);
    let pairs = match_branches(&ga, &gb, p);
    let mut matched_a = vec![false; ga.len()];
    let mut matched_b = vec![false; gb.len()];
    let n_orders = (ma.max_order.max(mb.max_order) as usize + 1).max(1);
    let mut trusted_by = vec![0.0; n_orders];
    trusted_by[0] = stem_trusted;
    let rs2 = p.radius_sigma * p.radius_sigma;
    let matched: Vec<BranchMatch> = pairs
        .iter()
        .map(|&(ia, ib, d, ang, consistent)| {
            matched_a[ia] = true;
            matched_b[ib] = true;
            let (x, y) = (&ga[ia], &gb[ib]);
            let s2 = |s: f64| if s.is_finite() { s * s } else { 0.0 };
            // The stem's uncertainty comes from its height bins.
            let volume_sigma = if x.order == 0 { stem_sigma } else { 2.0 * PI * (x.mean_radius + y.mean_radius) / 2.0 * x.length.min(y.length) * (s2(x.scatter) + s2(y.scatter) + 2.0 * rs2).sqrt() };
            BranchMatch {
                id_a: x.id,
                id_b: y.id,
                order_a: x.order,
                order_b: y.order,
                base_distance: d,
                angle: ang,
                parent_consistent: consistent,
                length_a: x.length,
                length_b: y.length,
                volume_a: x.volume,
                volume_b: y.volume,
                mean_radius_a: x.mean_radius,
                mean_radius_b: y.mean_radius,
                tip_shift: norm(&sub(&y.tip, &x.tip)),
                measured_a: x.measured,
                measured_b: y.measured,
                volume_sigma,
                trusted: x.measured >= p.min_measured && y.measured >= p.min_measured,
            }
        })
        .collect();
    for m in &matched {
        if m.order_a >= 1 && m.trusted {
            trusted_by[m.order_a as usize] += m.volume_b - m.volume_a;
            var += m.volume_sigma * m.volume_sigma;
        }
    }
    let lost: Vec<BranchChange> = ga.iter().zip(&matched_a).filter(|(_, &m)| !m).map(|(g, _)| unmatched(g, &a.cylinders, grid_b, BranchStatus::Lost, p)).collect();
    let new: Vec<BranchChange> = gb.iter().zip(&matched_b).filter(|(_, &m)| !m).map(|(g, _)| unmatched(g, &b.cylinders, grid_a, BranchStatus::New, p)).collect();
    for c in &lost {
        if c.order >= 1 && c.trusted {
            trusted_by[c.order as usize] -= c.volume;
            var += c.volume_sigma * c.volume_sigma;
        }
    }
    for c in &new {
        if c.order >= 1 && c.trusted {
            trusted_by[c.order as usize] += c.volume;
            var += c.volume_sigma * c.volume_sigma;
        }
    }
    let va = volume_by_order(a, n_orders);
    let vb = volume_by_order(b, n_orders);
    let orders: Vec<OrderChange> = (0..n_orders)
        .map(|k| OrderChange { order: k as u32, volume_a: va[k], volume_b: vb[k], trusted_change: trusted_by[k], untrusted_change: (vb[k] - va[k]) - trusted_by[k] })
        .collect();
    let trusted_change: f64 = trusted_by.iter().sum();
    let volume_a = a.total_volume();
    let volume_b = b.total_volume();

    let dbh_bin = (1.3 / p.height_step).floor() as usize;
    let unobserved = lost.iter().chain(&new).any(|c| c.status == BranchStatus::Unobserved);
    let branch_measured = |q: &Qsm| measured_share(q.cylinders.iter().filter(|c| c.branch_order >= 1), false);
    Ok(QsmChange {
        base_z: finite_or_nan(bz),
        height_a: finite_or_nan(ma.height),
        height_b: finite_or_nan(mb.height),
        height_trusted: !a.is_empty() && !b.is_empty() && top_measured(a, p.top_band) && top_measured(b, p.top_band),
        dbh_a: ma.dbh,
        dbh_b: mb.dbh,
        dbh_trusted: taper.get(dbh_bin).is_some_and(|t| t.trusted),
        crown_area_a: ma.crown.projected_area,
        crown_area_b: mb.crown.projected_area,
        crown_volume_a: ma.crown.volume,
        crown_volume_b: mb.crown.volume,
        crown_trusted: branch_measured(a) >= p.min_measured && branch_measured(b) >= p.min_measured && !unobserved,
        volume_a,
        volume_b,
        measured_volume_a: measured_share(a.cylinders.iter(), true),
        measured_volume_b: measured_share(b.cylinders.iter(), true),
        trusted_change,
        untrusted_change: (volume_b - volume_a) - trusted_change,
        trusted_sigma: var.sqrt(),
        orders,
        taper,
        taper_increment,
        taper_sigma,
        n_taper_bins,
        matched,
        lost,
        new,
    })
}

/// What happened to a tree between the epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fate {
    Survivor,
    Death,
    Recruit,
}

impl Fate {
    pub fn as_str(self) -> &'static str {
        match self {
            Fate::Survivor => "survivor",
            Fate::Death => "death",
            Fate::Recruit => "recruit",
        }
    }
}

/// One tree of a plot: its ids and models in each epoch (None where it has
/// no id or was not modelled).
#[derive(Debug, Clone, Copy)]
pub struct PlotTree<'a> {
    pub fate: Fate,
    pub id_a: Option<i64>,
    pub id_b: Option<i64>,
    pub a: Option<&'a Qsm>,
    pub b: Option<&'a Qsm>,
}

/// One row of the plot table.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotRow {
    pub fate: Fate,
    pub id_a: Option<i64>,
    pub id_b: Option<i64>,
    pub volume_a: f64,
    pub volume_b: f64,
    pub change: f64,
    pub trusted_change: f64,
    pub untrusted_change: f64,
    pub trusted_sigma: f64,
    /// Stem (order 0) and branch (order >= 1) parts of `change`.
    pub stem_change: f64,
    pub branch_change: f64,
    pub taper_increment: f64,
    pub taper_sigma: f64,
    pub height_a: f64,
    pub height_b: f64,
    pub n_matched: usize,
    pub n_lost: usize,
    pub n_unobserved: usize,
    pub n_present: usize,
    pub n_new: usize,
    /// Why a row has no comparison (a missing model), else empty.
    pub note: String,
}

/// Plot totals of [`compare_plot`] (m3); NaN rows are left out and counted
/// in `n_unmodelled`.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotTotals {
    pub n_survivors: usize,
    pub n_deaths: usize,
    pub n_recruits: usize,
    pub n_unmodelled: usize,
    /// Volume change of the survivors, its trusted part and that part's
    /// standard uncertainty.
    pub growth: f64,
    pub growth_trusted: f64,
    pub growth_sigma: f64,
    /// Volume of the dead trees in the earlier epoch and of the recruits in
    /// the later one (both positive).
    pub mortality: f64,
    pub recruitment: f64,
    /// `growth - mortality + recruitment`, and its trusted part.
    pub net: f64,
    pub net_trusted: f64,
}

/// Result of [`compare_plot`]: a row and (for compared survivors) a full
/// comparison per tree, in the order given, and the totals.
#[derive(Debug, Clone)]
pub struct PlotChange {
    pub rows: Vec<PlotRow>,
    pub changes: Vec<Option<QsmChange>>,
    pub totals: PlotTotals,
}

fn plot_row(t: &PlotTree, grid_a: Option<&StateGrid>, grid_b: Option<&StateGrid>, p: &CompareParams) -> Result<(PlotRow, Option<QsmChange>)> {
    let nan = f64::NAN;
    let mut row = PlotRow {
        fate: t.fate,
        id_a: t.id_a,
        id_b: t.id_b,
        volume_a: nan,
        volume_b: nan,
        change: nan,
        trusted_change: nan,
        untrusted_change: nan,
        trusted_sigma: nan,
        stem_change: nan,
        branch_change: nan,
        taper_increment: nan,
        taper_sigma: nan,
        height_a: nan,
        height_b: nan,
        n_matched: 0,
        n_lost: 0,
        n_unobserved: 0,
        n_present: 0,
        n_new: 0,
        note: String::new(),
    };
    // A dead tree or a recruit counts in full; its measured volume is the
    // trusted part.
    let whole = |q: &Qsm, sign: f64, row: &mut PlotRow| {
        let v = q.total_volume();
        let m = measured_share(q.cylinders.iter(), true) * v;
        let stem = q.stem_volume();
        row.change = sign * v;
        row.trusted_change = sign * m;
        row.untrusted_change = sign * (v - m);
        row.trusted_sigma = 0.0;
        row.stem_change = sign * stem;
        row.branch_change = sign * (v - stem);
    };
    match (t.fate, t.a, t.b) {
        (Fate::Survivor, Some(a), Some(b)) => {
            let c = compare_qsms(a, b, grid_a, grid_b, p)?;
            row.volume_a = c.volume_a;
            row.volume_b = c.volume_b;
            row.change = c.change();
            row.trusted_change = c.trusted_change;
            row.untrusted_change = c.untrusted_change;
            row.trusted_sigma = c.trusted_sigma;
            let stem = c.orders.first().map_or(0.0, |o| o.volume_b - o.volume_a);
            row.stem_change = stem;
            row.branch_change = row.change - stem;
            row.taper_increment = c.taper_increment;
            row.taper_sigma = c.taper_sigma;
            row.height_a = c.height_a;
            row.height_b = c.height_b;
            row.n_matched = c.matched.len();
            row.n_lost = c.lost.iter().filter(|x| x.status == BranchStatus::Lost).count();
            row.n_unobserved = c.lost.iter().chain(&c.new).filter(|x| x.status == BranchStatus::Unobserved).count();
            row.n_present = c.lost.iter().chain(&c.new).filter(|x| x.status == BranchStatus::Present).count();
            row.n_new = c.new.iter().filter(|x| x.status == BranchStatus::New).count();
            return Ok((row, Some(c)));
        }
        (Fate::Survivor, a, b) => {
            row.note = match (a.is_some(), b.is_some()) {
                (false, false) => "no model in either epoch",
                (false, true) => "no model in epoch a",
                _ => "no model in epoch b",
            }
            .into();
            row.volume_a = a.map_or(nan, Qsm::total_volume);
            row.volume_b = b.map_or(nan, Qsm::total_volume);
        }
        (Fate::Death, Some(a), _) => {
            row.volume_a = a.total_volume();
            row.volume_b = 0.0;
            row.height_a = finite_or_nan(top_z(a) - base_z(a));
            whole(a, -1.0, &mut row);
        }
        (Fate::Recruit, _, Some(b)) => {
            row.volume_a = 0.0;
            row.volume_b = b.total_volume();
            row.height_b = finite_or_nan(top_z(b) - base_z(b));
            whole(b, 1.0, &mut row);
        }
        (Fate::Death, None, _) => row.note = "no model in epoch a".into(),
        (Fate::Recruit, _, None) => row.note = "no model in epoch b".into(),
    }
    Ok((row, None))
}

/// Compares the models of every tree of a plot (in parallel; the result does
/// not depend on the thread count) and totals the changes.
pub fn compare_plot(trees: &[PlotTree], grid_a: Option<&StateGrid>, grid_b: Option<&StateGrid>, p: &CompareParams) -> Result<PlotChange> {
    p.check()?;
    let done: Vec<(PlotRow, Option<QsmChange>)> = trees.par_iter().map(|t| plot_row(t, grid_a, grid_b, p)).collect::<Result<_>>()?;
    let mut totals = PlotTotals { n_survivors: 0, n_deaths: 0, n_recruits: 0, n_unmodelled: 0, growth: 0.0, growth_trusted: 0.0, growth_sigma: 0.0, mortality: 0.0, recruitment: 0.0, net: 0.0, net_trusted: 0.0 };
    let mut var = 0.0;
    for (r, _) in &done {
        match r.fate {
            Fate::Survivor => totals.n_survivors += 1,
            Fate::Death => totals.n_deaths += 1,
            Fate::Recruit => totals.n_recruits += 1,
        }
        if !r.change.is_finite() {
            totals.n_unmodelled += 1;
            continue;
        }
        match r.fate {
            Fate::Survivor => {
                totals.growth += r.change;
                totals.growth_trusted += r.trusted_change;
                var += r.trusted_sigma * r.trusted_sigma;
            }
            Fate::Death => totals.mortality -= r.change,
            Fate::Recruit => totals.recruitment += r.change,
        }
        totals.net_trusted += r.trusted_change;
    }
    totals.growth_sigma = var.sqrt();
    totals.net = totals.growth - totals.mortality + totals.recruitment;
    let (rows, changes) = done.into_iter().unzip();
    Ok(PlotChange { rows, changes, totals })
}

/// Columns of [`plot_table_csv`].
pub const PLOT_COLUMNS: [&str; 21] = [
    "fate", "tree_id_a", "tree_id_b", "volume_a_m3", "volume_b_m3", "change_m3", "trusted_change_m3", "untrusted_change_m3", "trusted_sigma_m3", "stem_change_m3", "branch_change_m3", "taper_increment_m", "taper_sigma_m", "height_a_m", "height_b_m", "n_matched", "n_lost", "n_unobserved", "n_present", "n_new", "note",
];

/// The plot rows as CSV text (`\n` line ends; a missing id or a NaN is an
/// empty cell).
pub fn plot_table_csv(rows: &[PlotRow]) -> String {
    let mut s = PLOT_COLUMNS.join(",") + "\n";
    let f = |x: f64| if x.is_finite() { format!("{x}") } else { String::new() };
    let id = |x: Option<i64>| x.map(|v| v.to_string()).unwrap_or_default();
    for r in rows {
        let _ = writeln!(
            s,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.fate.as_str(),
            id(r.id_a),
            id(r.id_b),
            f(r.volume_a),
            f(r.volume_b),
            f(r.change),
            f(r.trusted_change),
            f(r.untrusted_change),
            f(r.trusted_sigma),
            f(r.stem_change),
            f(r.branch_change),
            f(r.taper_increment),
            f(r.taper_sigma),
            f(r.height_a),
            f(r.height_b),
            r.n_matched,
            r.n_lost,
            r.n_unobserved,
            r.n_present,
            r.n_new,
            r.note
        );
    }
    s
}

/// Write [`plot_table_csv`] to `path`.
pub fn write_plot_csv(path: impl AsRef<Path>, rows: &[PlotRow]) -> Result<()> {
    std::fs::write(path, plot_table_csv(rows))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn cyl(start: Point, axis: Point, length: f64, radius: f64, parent: i64, order: u32, branch: u32, n_points: usize) -> Cylinder {
        let n = norm(&axis);
        Cylinder { start, axis: scale(&axis, 1.0 / n), length, radius, parent, branch_order: order, branch_id: branch, n_points }
    }

    /// A 10 m stem of 0.1 m cylinders tapering from `r0` by 0.01 per metre,
    /// plus radius `dr`, and horizontal limbs (height, azimuth deg, length).
    fn model(r0: f64, dr: f64, limbs: &[(f64, f64, f64)], stem_points: impl Fn(f64) -> usize) -> Qsm {
        let mut c = Vec::new();
        for k in 0..100 {
            let z = k as f64 * 0.1;
            c.push(cyl([0.0, 0.0, z], [0.0, 0.0, 1.0], 0.1, r0 - 0.01 * (z + 0.05) + dr, k as i64 - 1, 0, 0, stem_points(z)));
        }
        for (j, &(h, az, len)) in limbs.iter().enumerate() {
            let parent = (h / 0.1).round() as i64 - 1;
            let (s, co) = az.to_radians().sin_cos();
            let n = (len / 0.1).round() as usize;
            for k in 0..n {
                let start = [co * 0.1 * k as f64, s * 0.1 * k as f64, h];
                let p = if k == 0 { parent } else { c.len() as i64 - 1 };
                c.push(cyl(start, [co, s, 0.0], 0.1, 0.03 - 0.001 * k as f64, p, 1, j as u32 + 1, 60));
            }
        }
        Qsm { cylinders: c }
    }

    #[test]
    fn thickening_is_the_taper_increment() {
        let a = model(0.2, 0.0, &[(5.0, 0.0, 2.0)], |_| 80);
        let b = model(0.2, 0.01, &[(5.0, 0.0, 2.0)], |_| 80);
        let c = compare_qsms(&a, &b, None, None, &CompareParams::default()).unwrap();
        assert_eq!(c.taper.len(), 10);
        assert!(c.taper.iter().all(|t| t.trusted && (t.increment - 0.01).abs() < 1e-9));
        assert!((c.taper_increment - 0.01).abs() < 1e-9);
        // Stem volume change is pi ((r + dr)^2 - r^2) summed over the cylinders.
        let expect: f64 = (0..100).map(|k| PI * 0.1 * ((0.2 - 0.01 * (k as f64 * 0.1 + 0.05) + 0.01f64).powi(2) - (0.2 - 0.01 * (k as f64 * 0.1 + 0.05)).powi(2))).sum();
        assert!((c.orders[0].trusted_change - expect).abs() < 1e-12);
        assert!(c.orders[0].untrusted_change.abs() < 1e-12);
        assert_eq!(c.matched.len(), 2);
        assert!(c.lost.is_empty() && c.new.is_empty());
        assert!((c.trusted_change - c.change()).abs() < 1e-12);
        assert!(c.dbh_trusted && c.height_trusted);
    }

    #[test]
    fn prior_filled_bins_are_not_trusted() {
        let a = model(0.2, 0.0, &[], |_| 80);
        // The later model has no fits between 4 and 6 m.
        let b = model(0.2, 0.01, &[], |z| if (4.0..6.0).contains(&z) { 0 } else { 80 });
        let c = compare_qsms(&a, &b, None, None, &CompareParams::default()).unwrap();
        for t in &c.taper {
            assert_eq!(t.trusted, !(4.0..6.0).contains(&t.z0), "bin at {}", t.z0);
        }
        let untrusted: f64 = c.taper.iter().filter(|t| !t.trusted).map(|t| t.volume_b - t.volume_a).sum();
        assert!((c.orders[0].untrusted_change - untrusted).abs() < 1e-12);
        assert_eq!(c.n_taper_bins, 8);
    }

    #[test]
    fn a_bin_out_of_line_is_fitted_but_not_trusted() {
        let a = model(0.2, 0.0, &[], |_| 80);
        let mut b = model(0.2, 0.01, &[], |_| 80);
        // Radii between 8 and 9 m run 3 cm wide in the later model.
        for c in b.cylinders.iter_mut().filter(|c| (8.0..9.0).contains(&c.start[2])) {
            c.radius += 0.03;
        }
        let c = compare_qsms(&a, &b, None, None, &CompareParams::default()).unwrap();
        let odd = &c.taper[8];
        assert!(odd.fitted && !odd.trusted && (odd.increment - 0.04).abs() < 1e-9);
        assert!(c.taper.iter().enumerate().all(|(k, t)| t.trusted == (k != 8)));
        assert!((c.taper_increment - 0.01).abs() < 1e-9 && c.n_taper_bins == 9);
        assert!((c.orders[0].untrusted_change - (odd.volume_b - odd.volume_a)).abs() < 1e-12);
    }

    #[test]
    fn branches_match_are_lost_and_new() {
        let a = model(0.2, 0.0, &[(4.0, 0.0, 1.5), (6.0, 120.0, 1.5)], |_| 80);
        // Limb 1 grows by 0.5 m, limb 2 is gone, a limb appears at 8 m.
        let b = model(0.2, 0.0, &[(4.0, 0.0, 2.0), (8.0, 240.0, 1.0)], |_| 80);
        let c = compare_qsms(&a, &b, None, None, &CompareParams::default()).unwrap();
        let limb = c.matched.iter().find(|m| m.order_a == 1).unwrap();
        assert!((limb.length_b - limb.length_a - 0.5).abs() < 1e-9 && limb.trusted && limb.parent_consistent);
        assert_eq!(c.lost.len(), 1);
        assert_eq!((c.lost[0].id, c.lost[0].status), (2, BranchStatus::Lost));
        assert!(c.lost[0].observed_share.is_nan());
        assert_eq!(c.new.len(), 1);
        assert_eq!(c.new[0].status, BranchStatus::New);
        let o = &c.orders[1];
        assert!((o.trusted_change - (o.volume_b - o.volume_a)).abs() < 1e-12);
    }

    #[test]
    fn a_grid_tells_lost_from_unobserved_and_present() {
        let a = model(0.2, 0.0, &[(6.0, 0.0, 1.5)], |_| 80);
        let b = model(0.2, 0.0, &[], |_| 80);
        let shape = [40, 40, 120];
        let make = |s: u8| StateGrid::new([-2.0, -2.0, -1.0], 0.1, shape, vec![s; shape[0] * shape[1] * shape[2]]).unwrap();
        let p = CompareParams::default();
        for (state, status, trusted) in [(EMPTY, BranchStatus::Lost, true), (OCCLUDED, BranchStatus::Unobserved, false), (UNOBSERVED, BranchStatus::Unobserved, false), (FILLED, BranchStatus::Present, false)] {
            let c = compare_qsms(&a, &b, None, Some(&make(state)), &p).unwrap();
            assert_eq!((c.lost[0].status, c.lost[0].trusted), (status, trusted));
        }
        // Outside the grid is unobserved.
        let far = StateGrid::new([50.0, 50.0, 50.0], 0.1, [2, 2, 2], vec![EMPTY; 8]).unwrap();
        let c = compare_qsms(&a, &b, None, Some(&far), &p).unwrap();
        assert_eq!(c.lost[0].status, BranchStatus::Unobserved);
        assert!(!c.crown_trusted);
    }

    #[test]
    fn empty_models_and_bad_settings() {
        let e = Qsm::default();
        let c = compare_qsms(&e, &e, None, None, &CompareParams::default()).unwrap();
        assert!(c.taper.is_empty() && c.matched.is_empty() && c.taper_increment.is_nan() && c.height_a.is_nan());
        let a = model(0.2, 0.0, &[(5.0, 0.0, 1.0)], |_| 80);
        let c = compare_qsms(&a, &e, None, None, &CompareParams::default()).unwrap();
        assert_eq!(c.lost.len(), 2);
        assert!((c.change() + a.total_volume()).abs() < 1e-12);
        let bad = CompareParams { height_step: 0.0, ..CompareParams::default() };
        assert!(compare_qsms(&a, &a, None, None, &bad).is_err());
        let bad = CompareParams { min_measured: 1.5, ..CompareParams::default() };
        assert!(compare_qsms(&a, &a, None, None, &bad).is_err());
        assert!(StateGrid::new([0.0; 3], 0.1, [2, 2, 2], vec![0; 7]).is_err());
    }

    #[test]
    fn plot_rows_and_totals() {
        let a = model(0.2, 0.0, &[(5.0, 0.0, 1.0)], |_| 80);
        let b = model(0.2, 0.01, &[(5.0, 0.0, 1.0)], |_| 80);
        let trees = [
            PlotTree { fate: Fate::Survivor, id_a: Some(1), id_b: Some(7), a: Some(&a), b: Some(&b) },
            PlotTree { fate: Fate::Death, id_a: Some(2), id_b: None, a: Some(&a), b: None },
            PlotTree { fate: Fate::Recruit, id_a: None, id_b: Some(9), a: None, b: Some(&b) },
            PlotTree { fate: Fate::Survivor, id_a: Some(3), id_b: Some(8), a: Some(&a), b: None },
        ];
        let r = compare_plot(&trees, None, None, &CompareParams::default()).unwrap();
        let t = &r.totals;
        assert_eq!((t.n_survivors, t.n_deaths, t.n_recruits, t.n_unmodelled), (2, 1, 1, 1));
        assert!((t.mortality - a.total_volume()).abs() < 1e-12 && (t.recruitment - b.total_volume()).abs() < 1e-12);
        assert!((t.growth - (b.total_volume() - a.total_volume())).abs() < 1e-12);
        assert!((t.net - (t.growth - t.mortality + t.recruitment)).abs() < 1e-12);
        assert_eq!(r.rows[3].note, "no model in epoch b");
        assert!(r.changes[0].is_some() && r.changes[1].is_none());
        let csv = plot_table_csv(&r.rows);
        assert!(csv.starts_with("fate,tree_id_a") && csv.contains("\ndeath,2,,") && csv.lines().count() == 5);
    }
}
