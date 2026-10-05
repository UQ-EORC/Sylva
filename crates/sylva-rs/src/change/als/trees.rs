// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Airborne trees of two surveys matched, with height growth, mortality,
//! damage and recruitment.
//!
//! The trees of each survey (from `als.find_trees`) are matched by the
//! optimal assignment of [`crate::change::trees::match_trees`], with the
//! tree height standing in for the diameter: a pair must lie within
//! `max_distance` and its height may grow by at most `max_growth` or fall by
//! at most `max_drop` of the larger height. The canopy height change of the
//! same area ([`super::surface`]) then says what became of the trees left
//! over and supplies the uncertainty of each height:
//!
//! * a matched tree is a **survivor**, or **damaged** when a significant
//!   loss covers `damage_fraction` of its crown or its height fell by more
//!   than its level of detection;
//! * an unmatched tree of the first survey is **damaged** when a tree of the
//!   second survey stands within twice `max_distance` of its top on
//!   significantly lowered canopy (a broken or collapsed crown), **dead** when
//!   a significant loss covers `dead_fraction` of its crown, **unobserved**
//!   when less than `min_observed` of its crown has data in both surveys,
//!   and otherwise **undetected**: its canopy is still there, but no tree
//!   was found on it (merged with a neighbour, or its top missed);
//! * an unmatched tree of the second survey is a **recruit** when its top
//!   rose significantly from below `1 - max_growth` of its height (more than
//!   the crown of an existing tree could grow), **released** when the canopy
//!   above it fell (an understorey tree exposed by the loss of a neighbour),
//!   **unobserved** without data, and otherwise **undetected**: its top did
//!   not change significantly (the first survey missed the tree), or rose
//!   no more than a growing crown does (a top split off a neighbour's
//!   growing crown).
//!
//! A height is the value of its top's CHM cell, and its uncertainty is that
//! cell's standard deviation in [`super::surface::SurfaceChange`]: the return
//! noise, the sampling of the apex (the spacing of the highest returns) and
//! the DTM error. Crown areas and crown loss are given as measured, without
//! a level of detection.

use crate::change::trees::{match_trees, MatchParams, TreeRow};
use crate::error::{Error, Result};
use crate::raster::Raster;

use super::gaps::inside;
use super::{z_of, Alignment};

/// A tree of one survey.
#[derive(Debug, Clone, PartialEq)]
pub struct AlsTree {
    pub id: i64,
    pub x: f64,
    pub y: f64,
    pub height: f64,
    pub crown_area: f64,
    /// Crown outline (may be empty).
    pub crown: Vec<[f64; 2]>,
}

/// Settings of [`tree_change`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeChangeParams {
    pub max_distance: f64,
    pub max_growth: f64,
    pub max_drop: f64,
    /// Weight of the squared relative height difference in the matching
    /// cost, next to the squared distance over `max_distance`.
    pub height_weight: f64,
    pub confidence: f64,
    pub dead_fraction: f64,
    pub damage_fraction: f64,
    pub min_observed: f64,
}

impl Default for TreeChangeParams {
    fn default() -> Self {
        TreeChangeParams { max_distance: 1.5, max_growth: 0.3, max_drop: 0.2, height_weight: 1.0, confidence: 0.95, dead_fraction: 0.5, damage_fraction: 0.3, min_observed: 0.5 }
    }
}

impl TreeChangeParams {
    pub fn check(&self) -> Result<()> {
        for (name, v) in [("max_distance", self.max_distance), ("max_growth", self.max_growth)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::invalid(format!("{name} must be a positive number, got {v}")));
            }
        }
        if self.max_drop.is_nan() || self.max_drop < 0.0 || self.max_drop >= 1.0 {
            return Err(Error::invalid(format!("max_drop must be in [0, 1), got {}", self.max_drop)));
        }
        if self.height_weight.is_nan() || self.height_weight < 0.0 {
            return Err(Error::invalid(format!("height_weight must be zero or more, got {}", self.height_weight)));
        }
        for (name, v) in [("dead_fraction", self.dead_fraction), ("damage_fraction", self.damage_fraction), ("min_observed", self.min_observed)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(Error::invalid(format!("{name} must be between 0 and 1, got {v}")));
            }
        }
        z_of(self.confidence)?;
        Ok(())
    }
}

/// The canopy change the trees are read against, on one grid.
pub struct CanopyChange<'a> {
    pub chm_a: &'a Raster,
    pub chm_b: &'a Raster,
    pub sigma_a: &'a Raster,
    pub sigma_b: &'a Raster,
    /// Codes of [`super::surface::CLASSES`].
    pub classes: &'a [u8],
}

/// One row of the comparison: a tree of the first survey (with its partner,
/// if any) or a tree found only in the second.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeChangeRow {
    /// Ids in each survey (0 for none).
    pub id_a: i64,
    pub id_b: i64,
    /// Top in the first survey's frame (the second survey's for its own trees).
    pub x: f64,
    pub y: f64,
    /// Horizontal distance between the partners' tops (m).
    pub distance: f64,
    pub height_a: f64,
    pub height_b: f64,
    /// `height_b - height_a`, its standard deviation and level of detection.
    pub dh: f64,
    pub sigma: f64,
    pub lod: f64,
    /// "growth", "decrease", "below_detection" or "unmeasured"; empty
    /// without a partner.
    pub dh_change: &'static str,
    pub status: &'static str,
    pub crown_area_a: f64,
    pub crown_area_b: f64,
    /// Shares of the crown (the first survey's, else the second's) with
    /// significant loss and gain, among its cells with data; and the share
    /// of its cells with data in both surveys.
    pub crown_loss: f64,
    pub crown_gain: f64,
    pub observed: f64,
}

/// Cells of the grid whose centres lie in a crown (or, without an outline,
/// within `radius` of the top).
fn crown_cells(t: &AlsTree, x: f64, y: f64, dx: f64, dy: f64, g: &Raster) -> Vec<usize> {
    let ring: Vec<[f64; 2]> = t.crown.iter().map(|v| [v[0] - dx, v[1] - dy]).collect();
    let res = g.resolution;
    let (bx0, by0, bx1, by1, disc) = if ring.len() >= 3 {
        let (mut a, mut b, mut c, mut d) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for v in &ring {
            a = a.min(v[0]);
            b = b.min(v[1]);
            c = c.max(v[0]);
            d = d.max(v[1]);
        }
        (a, b, c, d, None)
    } else {
        let r = if t.crown_area.is_finite() && t.crown_area > 0.0 { (t.crown_area / std::f64::consts::PI).sqrt() } else { res };
        let r = r.max(0.5 * res);
        (x - r, y - r, x + r, y + r, Some(r))
    };
    let (r0, c0) = g.cell_index(bx0, by0);
    let (r1, c1) = g.cell_index(bx1, by1);
    let mut out = Vec::new();
    for r in r0.max(0)..=r1.min(g.nrows as i64 - 1) {
        for c in c0.max(0)..=c1.min(g.ncols as i64 - 1) {
            let (cx, cy) = g.cell_center(r as usize, c as usize);
            let ok = match disc {
                Some(rad) => (cx - x).hypot(cy - y) <= rad,
                None => inside(&ring, cx, cy),
            };
            if ok {
                out.push(r as usize * g.ncols + c as usize);
            }
        }
    }
    if out.is_empty() {
        let (r, c) = g.cell_index(x, y);
        if g.in_bounds(r, c) {
            out.push(r as usize * g.ncols + c as usize);
        }
    }
    out
}

/// `(observed, loss, gain)` shares of a set of cells.
fn crown_shares(cells: &[usize], classes: &[u8]) -> (f64, f64, f64) {
    if cells.is_empty() {
        return (0.0, f64::NAN, f64::NAN);
    }
    let seen = cells.iter().filter(|&&k| classes[k] != 0).count();
    let loss = cells.iter().filter(|&&k| classes[k] == 3).count();
    let gain = cells.iter().filter(|&&k| classes[k] == 2).count();
    let obs = seen as f64 / cells.len() as f64;
    if seen == 0 {
        return (obs, f64::NAN, f64::NAN);
    }
    (obs, loss as f64 / seen as f64, gain as f64 / seen as f64)
}

/// A raster's value in the cell of `(x, y)`, else the largest finite value
/// of its 3 x 3 neighbourhood.
fn at_top(r: &Raster, x: f64, y: f64) -> f64 {
    let (row, col) = r.cell_index(x, y);
    if r.in_bounds(row, col) {
        let v = r.get(row as usize, col as usize);
        if v.is_finite() {
            return v;
        }
    }
    let mut best = f64::NAN;
    for dr in -1..=1 {
        for dc in -1..=1 {
            if r.in_bounds(row + dr, col + dc) {
                let v = r.get((row + dr) as usize, (col + dc) as usize);
                if v.is_finite() && (best.is_nan() || v > best) {
                    best = v;
                }
            }
        }
    }
    best
}

fn class_at(g: &Raster, classes: &[u8], x: f64, y: f64) -> u8 {
    let (r, c) = g.cell_index(x, y);
    if g.in_bounds(r, c) { classes[r as usize * g.ncols + c as usize] } else { 0 }
}

/// Compare the trees of two surveys; see the module documentation.
/// `alignment` moves the second survey's trees into the first's frame.
pub fn tree_change(a: &[AlsTree], b: &[AlsTree], canopy: &CanopyChange, alignment: Option<&Alignment>, p: &TreeChangeParams) -> Result<Vec<TreeChangeRow>> {
    p.check()?;
    let g = canopy.chm_a;
    let n = g.nrows * g.ncols;
    for (name, r) in [("chm_b", canopy.chm_b), ("sigma_a", canopy.sigma_a), ("sigma_b", canopy.sigma_b)] {
        if r.nrows != g.nrows || r.ncols != g.ncols || (r.xmin - g.xmin).abs() > 1e-9 || (r.ymin - g.ymin).abs() > 1e-9 {
            return Err(Error::invalid(format!("{name} is not on the grid of chm_a")));
        }
    }
    if canopy.classes.len() != n {
        return Err(Error::invalid(format!("{} classes for {n} cells", canopy.classes.len())));
    }
    if a.iter().chain(b).any(|t| !(t.x.is_finite() && t.y.is_finite() && t.height.is_finite())) {
        return Err(Error::invalid("tree positions and heights must be finite"));
    }
    let z = z_of(p.confidence)?;
    // The second survey's trees in the first survey's frame.
    let shift: Vec<[f64; 3]> = b.iter().map(|t| alignment.map_or([0.0; 3], |al| al.offset_at(t.x, t.y))).collect();
    let bx: Vec<(f64, f64)> = b.iter().zip(&shift).map(|(t, s)| (t.x - s[0], t.y - s[1])).collect();
    let rows_a: Vec<TreeRow> = a.iter().map(|t| TreeRow { x: t.x, y: t.y, dbh: t.height, height: t.height }).collect();
    let rows_b: Vec<TreeRow> = b.iter().zip(&bx).map(|(t, &(x, y))| TreeRow { x, y, dbh: t.height, height: t.height }).collect();
    let mp = MatchParams { max_distance: p.max_distance, dbh_tolerance: p.max_growth, max_shrink: p.max_drop, dbh_weight: p.height_weight, height_weight: 0.0, merge_factor: f64::INFINITY };
    let m = match_trees(&rows_a, &rows_b, &mp)?;
    let mut partner_a: Vec<Option<usize>> = vec![None; a.len()];
    let mut partner_b: Vec<Option<usize>> = vec![None; b.len()];
    for &(i, j) in &m.pairs {
        partner_a[i] = Some(j);
        partner_b[j] = Some(i);
    }
    let sig_a = |i: usize| at_top(canopy.sigma_a, a[i].x, a[i].y);
    let sig_b = |j: usize| at_top(canopy.sigma_b, bx[j].0, bx[j].1);
    let increment = |i: usize, j: usize| -> (f64, f64, f64, &'static str) {
        let dh = b[j].height - a[i].height;
        let s = sig_a(i).hypot(sig_b(j));
        if !s.is_finite() {
            return (dh, f64::NAN, f64::NAN, "unmeasured");
        }
        let lod = z * s;
        (dh, s, lod, if dh > lod { "growth" } else if dh < -lod { "decrease" } else { "below_detection" })
    };
    // Unmatched trees of the second survey standing on significantly lowered canopy.
    let lowered: Vec<usize> = (0..b.len()).filter(|&j| partner_b[j].is_none() && class_at(g, canopy.classes, bx[j].0, bx[j].1) == 3).collect();
    let mut taken = vec![false; b.len()];
    let mut out = Vec::with_capacity(a.len() + b.len());
    for (i, t) in a.iter().enumerate() {
        let cells = crown_cells(t, t.x, t.y, 0.0, 0.0, g);
        let (observed, loss, gain) = crown_shares(&cells, canopy.classes);
        let mut row = TreeChangeRow { id_a: t.id, id_b: 0, x: t.x, y: t.y, distance: f64::NAN, height_a: t.height, height_b: f64::NAN, dh: f64::NAN, sigma: f64::NAN, lod: f64::NAN, dh_change: "", status: "", crown_area_a: t.crown_area, crown_area_b: f64::NAN, crown_loss: loss, crown_gain: gain, observed };
        let pair = |row: &mut TreeChangeRow, j: usize| {
            let (dh, s, lod, c) = increment(i, j);
            row.id_b = b[j].id;
            row.height_b = b[j].height;
            row.crown_area_b = b[j].crown_area;
            row.distance = (t.x - bx[j].0).hypot(t.y - bx[j].1);
            row.dh = dh;
            row.sigma = s;
            row.lod = lod;
            row.dh_change = c;
        };
        if let Some(j) = partner_a[i] {
            pair(&mut row, j);
            row.status = if row.dh_change == "decrease" || loss >= p.damage_fraction { "damaged" } else { "survivor" };
        } else {
            // The nearest lowered tree of the second survey near its top.
            let near = lowered.iter().copied().filter(|&j| !taken[j]).map(|j| (j, (t.x - bx[j].0).hypot(t.y - bx[j].1))).filter(|&(_, d)| d <= 2.0 * p.max_distance).min_by(|x, y| x.1.total_cmp(&y.1));
            if let Some((j, _)) = near.filter(|_| observed >= p.min_observed) {
                taken[j] = true;
                pair(&mut row, j);
                row.status = "damaged";
            } else if observed < p.min_observed {
                row.status = "unobserved";
            } else if loss >= p.dead_fraction {
                row.status = "dead";
            } else {
                row.status = "undetected";
            }
        }
        out.push(row);
    }
    for (j, t) in b.iter().enumerate() {
        if partner_b[j].is_some() || taken[j] {
            continue;
        }
        let (x, y) = bx[j];
        let cells = crown_cells(t, x, y, shift[j][0], shift[j][1], g);
        let (observed, loss, gain) = crown_shares(&cells, canopy.classes);
        // A recruit's top rose by more than an existing crown can grow:
        // the canopy under it was below (1 - max_growth) of its height.
        let below = at_top(canopy.chm_a, x, y) < (1.0 - p.max_growth) * t.height;
        let status = match class_at(g, canopy.classes, x, y) {
            0 => "unobserved",
            2 if below => "recruit",
            3 => "released",
            _ => "undetected",
        };
        let status = if observed < p.min_observed && status != "recruit" { "unobserved" } else { status };
        out.push(TreeChangeRow { id_a: 0, id_b: t.id, x, y, distance: f64::NAN, height_a: f64::NAN, height_b: t.height, dh: f64::NAN, sigma: f64::NAN, lod: f64::NAN, dh_change: "", status, crown_area_a: f64::NAN, crown_area_b: t.crown_area, crown_loss: loss, crown_gain: gain, observed });
    }
    Ok(out)
}

/// Totals of a tree comparison.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TreeSummary {
    pub survivors: usize,
    pub damaged: usize,
    pub dead: usize,
    pub recruits: usize,
    pub released: usize,
    pub undetected_a: usize,
    pub undetected_b: usize,
    pub unobserved_a: usize,
    pub unobserved_b: usize,
    /// Mean height change of the survivors with a measured change, its
    /// standard error from their scatter and from their measurement
    /// uncertainties alone, and their number.
    pub mean_growth: f64,
    pub growth_se: f64,
    pub growth_measurement_se: f64,
    pub n_growth: usize,
    /// Survivors whose growth exceeds its level of detection.
    pub n_growth_detected: usize,
    /// Crown area (m²) of the dead trees, and the lost part of the damaged
    /// trees' crowns.
    pub crown_area_dead: f64,
    pub crown_area_damaged: f64,
    /// Annual mortality and recruitment rates (Sheil, Burslem and Alder
    /// 1995), NaN without `years`.
    pub mortality_rate: f64,
    pub recruitment_rate: f64,
}

/// Totals over some rows (all of them, or those in an area).
pub fn summarise(rows: &[&TreeChangeRow], years: Option<f64>) -> TreeSummary {
    let mut s = TreeSummary::default();
    let mut dh = Vec::new();
    let mut var = 0.0;
    for r in rows {
        let from_a = r.id_a != 0;
        match (r.status, from_a) {
            ("survivor", _) => {
                s.survivors += 1;
                if r.dh.is_finite() && r.sigma.is_finite() {
                    dh.push(r.dh);
                    var += r.sigma * r.sigma;
                    if r.dh_change == "growth" {
                        s.n_growth_detected += 1;
                    }
                }
            }
            ("damaged", _) => {
                s.damaged += 1;
                if r.crown_area_a.is_finite() && r.crown_loss.is_finite() {
                    s.crown_area_damaged += r.crown_area_a * r.crown_loss;
                }
            }
            ("dead", _) => {
                s.dead += 1;
                if r.crown_area_a.is_finite() {
                    s.crown_area_dead += r.crown_area_a;
                }
            }
            ("recruit", _) => s.recruits += 1,
            ("released", _) => s.released += 1,
            ("undetected", true) => s.undetected_a += 1,
            ("undetected", false) => s.undetected_b += 1,
            ("unobserved", true) => s.unobserved_a += 1,
            ("unobserved", false) => s.unobserved_b += 1,
            _ => {}
        }
    }
    s.n_growth = dh.len();
    if !dh.is_empty() {
        let n = dh.len() as f64;
        let mean = dh.iter().sum::<f64>() / n;
        s.mean_growth = mean;
        s.growth_se = if dh.len() > 1 { (dh.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt() } else { f64::NAN };
        s.growth_measurement_se = var.sqrt() / n;
    } else {
        s.mean_growth = f64::NAN;
        s.growth_se = f64::NAN;
        s.growth_measurement_se = f64::NAN;
    }
    s.mortality_rate = f64::NAN;
    s.recruitment_rate = f64::NAN;
    if let Some(t) = years.filter(|t| *t > 0.0) {
        let n0 = (s.survivors + s.damaged + s.dead) as f64;
        let n1 = (s.survivors + s.damaged + s.recruits + s.released) as f64;
        if n0 > 0.0 {
            s.mortality_rate = 1.0 - (1.0 - s.dead as f64 / n0).powf(1.0 / t);
        }
        if n1 > 0.0 {
            s.recruitment_rate = 1.0 - (1.0 - s.recruits as f64 / n1).powf(1.0 / t);
        }
    }
    s
}

/// Names of the layers of [`grid_summary`].
pub const GRID_LAYERS: [&str; 7] = ["survivors", "damaged", "dead", "recruits", "mean_growth", "growth_se", "crown_area_lost"];

/// Totals per cell of a grid `(xmin, ymin, resolution, nrows, ncols)`, each
/// row in the cell of its top: one raster per name of [`GRID_LAYERS`]
/// (counts; mean growth and its standard error, NaN without a survivor;
/// crown area of the dead plus the lost part of the damaged, m²).
pub fn grid_summary(rows: &[TreeChangeRow], grid: (f64, f64, f64, usize, usize)) -> Result<Vec<Raster>> {
    let (xmin, ymin, res, nr, nc) = grid;
    if !(res.is_finite() && res > 0.0) || nr == 0 || nc == 0 {
        return Err(Error::invalid("the summary grid needs a positive resolution and at least one cell"));
    }
    crate::util::limits::check_cells(nr as u128 * nc as u128, 56, &format!("a {nr} x {nc} summary grid"), "a coarser resolution")?;
    let mut members: Vec<Vec<&TreeChangeRow>> = vec![Vec::new(); nr * nc];
    for r in rows {
        let (row, col) = (((r.y - ymin) / res).floor(), ((r.x - xmin) / res).floor());
        if row >= 0.0 && col >= 0.0 && (row as usize) < nr && (col as usize) < nc {
            members[row as usize * nc + col as usize].push(r);
        }
    }
    let mut layers = vec![vec![0.0; nr * nc]; GRID_LAYERS.len()];
    for (k, m) in members.iter().enumerate() {
        let s = summarise(m, None);
        let v = [s.survivors as f64, s.damaged as f64, s.dead as f64, s.recruits as f64, s.mean_growth, s.growth_se, s.crown_area_dead + s.crown_area_damaged];
        for (l, x) in v.iter().enumerate() {
            layers[l][k] = *x;
        }
    }
    Ok(layers.into_iter().map(|data| Raster { data, nrows: nr, ncols: nc, xmin, ymin, resolution: res }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(id: i64, x: f64, y: f64, h: f64) -> AlsTree {
        let r = 2.0;
        let crown = (0..12).map(|k| {
            let a = k as f64 * std::f64::consts::PI / 6.0;
            [x + r * a.cos(), y + r * a.sin()]
        }).collect();
        AlsTree { id, x, y, height: h, crown_area: std::f64::consts::PI * r * r, crown }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn fates_follow_the_canopy_change() {
        // 30 x 30 m at 0.5 m; the canopy lost where tree 2 stood, and a new
        // crown appeared at (25, 25).
        let mut chm_a = Raster::filled(60, 60, 0.0, 0.0, 0.5, 20.0);
        let sig = Raster::filled(60, 60, 0.0, 0.0, 0.5, 0.1);
        let mut cls = vec![1u8; 3600];
        let mut chm_b = chm_a.clone();
        let g = chm_a.clone();
        for k in 0..3600 {
            let (x, y) = g.cell_center(k / 60, k % 60);
            if (x - 15.0).hypot(y - 5.0) <= 2.5 {
                cls[k] = 3;
                chm_b.data[k] = 0.2;
            }
            if (x - 25.0).hypot(y - 25.0) <= 1.0 {
                cls[k] = 2;
                chm_a.data[k] = 1.0;
            }
            if (x - 5.0).hypot(y - 25.0) <= 2.5 {
                cls[k] = 0;
            }
        }
        let canopy = CanopyChange { chm_a: &chm_a, chm_b: &chm_b, sigma_a: &sig, sigma_b: &sig, classes: &cls };
        let a = vec![tree(1, 5.0, 5.0, 20.0), tree(2, 15.0, 5.0, 22.0), tree(3, 25.0, 5.0, 18.0), tree(4, 5.0, 25.0, 19.0), tree(5, 15.0, 15.0, 21.0)];
        let b = vec![tree(11, 5.3, 5.1, 20.8), tree(13, 25.2, 4.9, 18.1), tree(16, 25.0, 25.0, 6.0)];
        let rows = tree_change(&a, &b, &canopy, None, &TreeChangeParams::default()).unwrap();
        let st: Vec<(&str, &str)> = rows.iter().map(|r| (r.status, r.dh_change)).collect();
        assert_eq!(st, vec![("survivor", "growth"), ("dead", ""), ("survivor", "below_detection"), ("unobserved", ""), ("undetected", ""), ("recruit", "")]);
        assert!((rows[0].dh - 0.8).abs() < 1e-12 && (rows[0].sigma - 0.1f64.hypot(0.1)).abs() < 1e-12);
        assert_eq!(rows[5].id_b, 16);
        let s = summarise(&rows.iter().collect::<Vec<_>>(), Some(5.0));
        assert_eq!((s.survivors, s.dead, s.recruits, s.unobserved_a, s.undetected_a), (2, 1, 1, 1, 1));
        assert!((s.mean_growth - 0.45).abs() < 1e-12);
        assert!((s.mortality_rate - (1.0 - (2.0f64 / 3.0).powf(0.2))).abs() < 1e-12);
        let gs = grid_summary(&rows, (0.0, 0.0, 10.0, 3, 3)).unwrap();
        assert_eq!(gs[2].get(0, 1), 1.0);
        assert!(gs[4].get(0, 0) > 0.7);
        // A known shift of the second survey is taken out before matching.
        let shifted: Vec<AlsTree> = b.iter().map(|t| AlsTree { x: t.x + 1.0, y: t.y - 0.5, crown: t.crown.iter().map(|v| [v[0] + 1.0, v[1] - 0.5]).collect(), ..t.clone() }).collect();
        let al = Alignment::constant([1.0, -0.5, 3.0], [0.0; 3]);
        let rows2 = tree_change(&a, &shifted, &canopy, Some(&al), &TreeChangeParams::default()).unwrap();
        assert_eq!(rows2.iter().map(|r| r.status).collect::<Vec<_>>(), rows.iter().map(|r| r.status).collect::<Vec<_>>());
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn a_broken_top_is_damage_not_death() {
        let g = Raster::filled(20, 20, 0.0, 0.0, 1.0, 20.0);
        let sig = Raster::filled(20, 20, 0.0, 0.0, 1.0, 0.2);
        let mut cls = vec![1u8; 400];
        for k in 0..400 {
            let (x, y) = g.cell_center(k / 20, k % 20);
            if (x - 10.0).hypot(y - 10.0) <= 2.0 {
                cls[k] = 3;
            }
        }
        let canopy = CanopyChange { chm_a: &g, chm_b: &g, sigma_a: &sig, sigma_b: &sig, classes: &cls };
        let rows = tree_change(&[tree(1, 10.0, 10.0, 20.0)], &[tree(2, 10.5, 10.4, 12.0)], &canopy, None, &TreeChangeParams::default()).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].status, rows[0].dh_change, rows[0].id_b), ("damaged", "decrease", 2));
    }

    /// A tree with no crown outline: its crown is the disc of its crown area.
    fn bare(id: i64, x: f64, y: f64, h: f64) -> AlsTree {
        AlsTree { id, x, y, height: h, crown_area: std::f64::consts::PI, crown: vec![] }
    }

    #[test]
    fn released_unmeasured_and_disc_crowns() {
        let chm = Raster::filled(20, 20, 0.0, 0.0, 1.0, 20.0);
        let mut sig = Raster::filled(20, 20, 0.0, 0.0, 1.0, 0.2);
        let mut cls = vec![1u8; 400];
        // No height uncertainty anywhere near (2.5, 2.5): the change there is unmeasured.
        for r in 0..5 {
            for c in 0..5 {
                sig.data[r * 20 + c] = f64::NAN;
            }
        }
        // Canopy lost at (15.5, 15.5): an understorey tree found there only in the second survey.
        cls[15 * 20 + 15] = 3;
        // A disc of radius 1 m: its own cell and the four at 1 m, three of the five lost.
        for (r, c) in [(10, 10), (10, 11), (11, 10)] {
            cls[r * 20 + c] = 3;
        }
        let canopy = CanopyChange { chm_a: &chm, chm_b: &chm, sigma_a: &sig, sigma_b: &sig, classes: &cls };
        let a = [bare(1, 2.5, 2.5, 20.0), bare(2, 10.5, 10.5, 20.0)];
        let b = [bare(11, 2.6, 2.5, 20.3), bare(12, 10.5, 10.6, 20.1), bare(13, 15.5, 15.5, 8.0)];
        let rows = tree_change(&a, &b, &canopy, None, &TreeChangeParams::default()).unwrap();
        let st: Vec<(i64, i64, &str, &str)> = rows.iter().map(|r| (r.id_a, r.id_b, r.status, r.dh_change)).collect();
        assert_eq!(st, vec![(1, 11, "survivor", "unmeasured"), (2, 12, "damaged", "below_detection"), (0, 13, "released", "")]);
        assert!(rows[0].sigma.is_nan() && (rows[0].dh - 0.3).abs() < 1e-12);
        assert!((rows[1].crown_loss - 0.6).abs() < 1e-12 && rows[1].observed == 1.0);
        let s = summarise(&rows.iter().collect::<Vec<_>>(), Some(2.0));
        assert_eq!((s.survivors, s.damaged, s.released, s.n_growth), (1, 1, 1, 0));
        assert!(s.mean_growth.is_nan() && s.growth_se.is_nan());
        assert!((s.crown_area_damaged - 0.6 * std::f64::consts::PI).abs() < 1e-12);
        // Released trees are recruited into the canopy but are not recruits.
        assert_eq!((s.mortality_rate, s.recruitment_rate), (0.0, 0.0));
    }

    #[test]
    fn bad_settings_and_inputs_are_named() {
        let err = |p: TreeChangeParams| p.check().unwrap_err().to_string();
        let d = TreeChangeParams::default();
        assert_eq!(err(TreeChangeParams { max_distance: 0.0, ..d }), "max_distance must be a positive number, got 0");
        assert_eq!(err(TreeChangeParams { max_growth: f64::NAN, ..d }), "max_growth must be a positive number, got NaN");
        assert_eq!(err(TreeChangeParams { max_drop: 1.0, ..d }), "max_drop must be in [0, 1), got 1");
        assert_eq!(err(TreeChangeParams { height_weight: -1.0, ..d }), "height_weight must be zero or more, got -1");
        assert_eq!(err(TreeChangeParams { damage_fraction: 1.5, ..d }), "damage_fraction must be between 0 and 1, got 1.5");
        assert_eq!(err(TreeChangeParams { confidence: 1.0, ..d }), "confidence must be between 0 and 1, got 1");
        let g = Raster::filled(4, 4, 0.0, 0.0, 1.0, 10.0);
        let off = Raster::filled(4, 4, 0.5, 0.0, 1.0, 10.0);
        let cls = vec![1u8; 16];
        let t = [bare(1, 1.5, 1.5, 10.0)];
        let run = |b: &Raster, s: &Raster, c: &[u8], trees: &[AlsTree]| tree_change(trees, &t, &CanopyChange { chm_a: &g, chm_b: b, sigma_a: s, sigma_b: s, classes: c }, None, &d).unwrap_err().to_string();
        assert_eq!(run(&off, &g, &cls, &t), "chm_b is not on the grid of chm_a");
        assert_eq!(run(&g, &off, &cls, &t), "sigma_a is not on the grid of chm_a");
        assert_eq!(run(&g, &g, &cls[..15], &t), "15 classes for 16 cells");
        assert_eq!(run(&g, &g, &cls, &[bare(1, f64::NAN, 1.5, 10.0)]), "tree positions and heights must be finite");
        let e = grid_summary(&[], (0.0, 0.0, 0.0, 2, 2)).unwrap_err().to_string();
        assert_eq!(e, "the summary grid needs a positive resolution and at least one cell");
    }
}
