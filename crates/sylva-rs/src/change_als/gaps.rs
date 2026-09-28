// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by
// law.
//! Canopy gaps and their dynamics.
//!
//! A gap is a connected region of CHM cells no higher than `height` whose
//! area lies between `min_area` and `max_area`, as ForestGapR's
//! `getForestGaps` defines it (Silva et al. 2019): cells are joined across
//! edges and corners (`connectivity` 8, ForestGapR's choice) or across edges
//! only (4). Between two surveys a cell forms a gap when it is in a gap of
//! the second survey but not of the first and its CHM fell significantly,
//! and closes when the reverse holds and its CHM rose significantly
//! (Silva et al.'s `GapChangeDec` without a significance test counts every
//! transition). A cell that crossed the threshold without a significant
//! change is uncertain rather than formed or closed.
//!
//! Gap sizes follow a power law above some size in many forests (Fisher et
//! al. 2008; Asner et al. 2013); [`size_exponent`] estimates its exponent by
//! maximum likelihood (Clauset, Shalizi and Newman 2009).

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::masks::Polygon;
use crate::raster::Raster;

/// Settings of [`find_gaps`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GapParams {
    /// Cells no higher than this (m) are gap.
    pub height: f64,
    pub min_area: f64,
    pub max_area: f64,
    /// 8 (edges and corners) or 4 (edges).
    pub connectivity: u8,
}

impl Default for GapParams {
    fn default() -> Self {
        GapParams { height: 2.0, min_area: 10.0, max_area: f64::INFINITY, connectivity: 8 }
    }
}

impl GapParams {
    pub fn check(&self) -> Result<()> {
        if !self.height.is_finite() {
            return Err(Error::invalid(format!("height must be a finite number, got {}", self.height)));
        }
        if self.min_area.is_nan() || self.min_area < 0.0 || self.max_area.is_nan() || self.max_area < self.min_area {
            return Err(Error::invalid(format!("need 0 <= min_area <= max_area, got {} and {}", self.min_area, self.max_area)));
        }
        if self.connectivity != 4 && self.connectivity != 8 {
            return Err(Error::invalid(format!("connectivity must be 4 or 8, got {}", self.connectivity)));
        }
        Ok(())
    }
}

/// One gap.
#[derive(Debug, Clone, PartialEq)]
pub struct Gap {
    pub id: u32,
    pub n_cells: usize,
    /// Area (m²).
    pub area: f64,
    /// Centre of its cells.
    pub x: f64,
    pub y: f64,
    /// Mean and largest CHM height of its cells (m).
    pub mean_height: f64,
    pub max_height: f64,
    /// Its outline: one polygon per part (parts touch only at corners),
    /// exteriors counter-clockwise, holes clockwise, along cell edges.
    pub polygons: Vec<Polygon>,
}

/// The gaps of one CHM.
#[derive(Debug, Clone, PartialEq)]
pub struct Gaps {
    /// Gap of each cell (0 for none), on the CHM's grid.
    pub labels: Vec<u32>,
    pub gaps: Vec<Gap>,
    /// Area (m²) of the cells with a CHM value.
    pub area_with_data: f64,
}

impl Gaps {
    /// Share of the area with data that is gap.
    pub fn gap_fraction(&self) -> f64 {
        let a: f64 = self.gaps.iter().map(|g| g.area).sum();
        if self.area_with_data > 0.0 { a / self.area_with_data } else { f64::NAN }
    }
}

/// Label connected regions of `mask` (row-major, `nr x nc`), in order of
/// their first cell; returns the labels and the cells of each region.
fn label(mask: &[bool], nr: usize, nc: usize, eight: bool) -> (Vec<u32>, Vec<Vec<usize>>) {
    let mut lab = vec![0u32; nr * nc];
    let mut regions: Vec<Vec<usize>> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..nr * nc {
        if !mask[start] || lab[start] != 0 {
            continue;
        }
        let id = regions.len() as u32 + 1;
        let mut cells = Vec::new();
        lab[start] = id;
        stack.push(start);
        while let Some(k) = stack.pop() {
            cells.push(k);
            let (r, c) = ((k / nc) as i64, (k % nc) as i64);
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if (dr == 0 && dc == 0) || (!eight && dr != 0 && dc != 0) {
                        continue;
                    }
                    let (rr, cc) = (r + dr, c + dc);
                    if rr < 0 || cc < 0 || rr >= nr as i64 || cc >= nc as i64 {
                        continue;
                    }
                    let j = rr as usize * nc + cc as usize;
                    if mask[j] && lab[j] == 0 {
                        lab[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
        cells.sort_unstable();
        regions.push(cells);
    }
    (lab, regions)
}

fn ring_area(r: &[[f64; 2]]) -> f64 {
    let n = r.len();
    (0..n).map(|i| r[i][0] * r[(i + 1) % n][1] - r[(i + 1) % n][0] * r[i][1]).sum::<f64>() / 2.0
}

pub(crate) fn inside(r: &[[f64; 2]], x: f64, y: f64) -> bool {
    let n = r.len();
    let mut c = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (r[i], r[j]);
        if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
            c = !c;
        }
        j = i;
    }
    c
}

/// Outline of a region of cells as polygons along cell edges. Edges run
/// with the region on their left; at a corner where two region cells meet
/// diagonally the trace turns left, so that rings never touch themselves.
pub fn outline(cells: &[usize], ncols: usize, xmin: f64, ymin: f64, res: f64) -> Vec<Polygon> {
    let set: std::collections::HashSet<usize> = cells.iter().copied().collect();
    let has = |r: i64, c: i64| r >= 0 && c >= 0 && (c as usize) < ncols && set.contains(&(r as usize * ncols + c as usize));
    // Directed edges between integer corners (col, row).
    let mut edges: Vec<((i64, i64), (i64, i64))> = Vec::new();
    for &k in cells {
        let (r, c) = ((k / ncols) as i64, (k % ncols) as i64);
        if !has(r - 1, c) {
            edges.push(((c, r), (c + 1, r)));
        }
        if !has(r, c + 1) {
            edges.push(((c + 1, r), (c + 1, r + 1)));
        }
        if !has(r + 1, c) {
            edges.push(((c + 1, r + 1), (c, r + 1)));
        }
        if !has(r, c - 1) {
            edges.push(((c, r + 1), (c, r)));
        }
    }
    edges.sort_unstable();
    let mut from: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        from.entry(e.0).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut rings: Vec<Vec<(i64, i64)>> = Vec::new();
    for s in 0..edges.len() {
        if used[s] {
            continue;
        }
        let mut ring = Vec::new();
        let mut e = s;
        loop {
            used[e] = true;
            ring.push(edges[e].0);
            let (a, b) = edges[e];
            let d = (b.0 - a.0, b.1 - a.1);
            let cand: Vec<usize> = from.get(&b).map(|v| v.iter().copied().filter(|&j| !used[j] || j == s).collect()).unwrap_or_default();
            if cand.is_empty() {
                break;
            }
            let next = if cand.len() == 1 {
                cand[0]
            } else {
                // Prefer the left turn: direction (-dy, dx).
                let left = (-d.1, d.0);
                *cand.iter().find(|&&j| {
                    let (p, q) = edges[j];
                    (q.0 - p.0, q.1 - p.1) == left
                }).unwrap_or(&cand[0])
            };
            if next == s {
                break;
            }
            e = next;
        }
        // Drop corners where the direction does not change.
        let n = ring.len();
        let simple: Vec<(i64, i64)> = (0..n)
            .filter(|&i| {
                let (p, q, r) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
                (q.0 - p.0) * (r.1 - q.1) - (q.1 - p.1) * (r.0 - q.0) != 0
            })
            .map(|i| ring[i])
            .collect();
        rings.push(simple);
    }
    let to_xy = |v: &(i64, i64)| [xmin + v.0 as f64 * res, ymin + v.1 as f64 * res];
    let mut exteriors: Vec<Polygon> = Vec::new();
    let mut holes: Vec<(Vec<[f64; 2]>, [f64; 2])> = Vec::new();
    for ring in &rings {
        let xy: Vec<[f64; 2]> = ring.iter().map(to_xy).collect();
        if ring_area(&xy) > 0.0 {
            exteriors.push(Polygon { exterior: xy, holes: Vec::new() });
        } else if ring.len() >= 2 {
            // The region cell to the left of the hole's first edge.
            let (p, q) = (ring[0], ring[1]);
            let (dx, dy) = ((q.0 - p.0).signum(), (q.1 - p.1).signum());
            let (mx, my) = (p.0 as f64 + 0.5 * dx as f64, p.1 as f64 + 0.5 * dy as f64);
            let (cx, cy) = (mx - 0.5 * dy as f64, my + 0.5 * dx as f64);
            holes.push((xy, [xmin + cx * res, ymin + cy * res]));
        }
    }
    for (h, pt) in holes {
        let mut best: Option<(usize, f64)> = None;
        for (i, e) in exteriors.iter().enumerate() {
            if inside(&e.exterior, pt[0], pt[1]) {
                let a = ring_area(&e.exterior);
                if best.is_none_or(|b| a < b.1) {
                    best = Some((i, a));
                }
            }
        }
        if let Some((i, _)) = best {
            exteriors[i].holes.push(h);
        }
    }
    exteriors
}

/// Canopy gaps of a CHM; see the module documentation. NaN cells are never gap.
pub fn find_gaps(chm: &Raster, p: &GapParams) -> Result<Gaps> {
    p.check()?;
    let (nr, nc, res) = (chm.nrows, chm.ncols, chm.resolution);
    if chm.data.len() != nr * nc {
        return Err(Error::invalid("the CHM's data length does not match its shape"));
    }
    let cell = res * res;
    let mask: Vec<bool> = chm.data.iter().map(|v| v.is_finite() && *v <= p.height).collect();
    let (_, regions) = label(&mask, nr, nc, p.connectivity == 8);
    let mut labels = vec![0u32; nr * nc];
    let mut gaps = Vec::new();
    for cells in regions {
        let area = cells.len() as f64 * cell;
        if area < p.min_area || area > p.max_area {
            continue;
        }
        let id = gaps.len() as u32 + 1;
        let (mut sx, mut sy, mut sh, mut mh) = (0.0, 0.0, 0.0, f64::NEG_INFINITY);
        for &k in &cells {
            labels[k] = id;
            let (x, y) = chm.cell_center(k / nc, k % nc);
            sx += x;
            sy += y;
            sh += chm.data[k];
            mh = mh.max(chm.data[k]);
        }
        let n = cells.len() as f64;
        gaps.push(Gap { id, n_cells: cells.len(), area, x: sx / n, y: sy / n, mean_height: sh / n, max_height: mh, polygons: outline(&cells, nc, chm.xmin, chm.ymin, res) });
    }
    let area_with_data = chm.data.iter().filter(|v| v.is_finite()).count() as f64 * cell;
    Ok(Gaps { labels, gaps, area_with_data })
}

/// Maximum-likelihood exponent `alpha` of a continuous power law
/// `p(x) ~ x^-alpha` over the sizes at or above `xmin`, its standard error
/// `(alpha - 1) / sqrt(n)`, and `n` (Clauset et al. 2009, eq. 3.1). NaN with
/// fewer than two sizes.
pub fn size_exponent(sizes: &[f64], xmin: f64) -> (f64, f64, usize) {
    let v: Vec<f64> = sizes.iter().copied().filter(|&s| s >= xmin && s.is_finite()).collect();
    let n = v.len();
    let s: f64 = v.iter().map(|x| (x / xmin).ln()).sum();
    if n < 2 || s.is_nan() || s <= 0.0 || xmin.is_nan() || xmin <= 0.0 {
        return (f64::NAN, f64::NAN, n);
    }
    let alpha = 1.0 + n as f64 / s;
    (alpha, (alpha - 1.0) / (n as f64).sqrt(), n)
}

/// Names of the cell codes of a gap change.
pub const GAP_CELLS: [&str; 6] = ["canopy", "stable_gap", "formed", "closed", "uncertain", "no_data"];

/// Gaps of two surveys compared.
#[derive(Debug, Clone, PartialEq)]
pub struct GapChange {
    pub a: Gaps,
    pub b: Gaps,
    /// Code of [`GAP_CELLS`] per cell.
    pub cells: Vec<u8>,
    /// Per gap of the first survey: "closed" (no part is gap any more),
    /// "shrunk", "stable" or "uncertain" (its changes are not significant
    /// or not observed); its area closed (m²).
    pub status_a: Vec<&'static str>,
    pub closed_area: Vec<f64>,
    /// Per gap of the second survey: "new", "expanded", "stable" or
    /// "uncertain"; its area formed (m²).
    pub status_b: Vec<&'static str>,
    pub formed_area: Vec<f64>,
    /// Area (m²) per cell code.
    pub areas: [f64; 6],
}

/// Gaps of two CHMs on one grid, and the change between them. `classes`
/// holds the significance of each cell's change (codes of
/// [`super::surface::CLASSES`]); without it every transition counts.
pub fn gap_change(chm_a: &Raster, chm_b: &Raster, classes: Option<&[u8]>, p: &GapParams) -> Result<GapChange> {
    if chm_a.nrows != chm_b.nrows || chm_a.ncols != chm_b.ncols || (chm_a.xmin - chm_b.xmin).abs() > 1e-9 || (chm_a.ymin - chm_b.ymin).abs() > 1e-9 || (chm_a.resolution - chm_b.resolution).abs() > 1e-12 {
        return Err(Error::invalid("the two CHMs must be on the same grid"));
    }
    let n = chm_a.data.len();
    if let Some(c) = classes {
        if c.len() != n {
            return Err(Error::invalid(format!("{} classes for {n} cells", c.len())));
        }
    }
    let ga = find_gaps(chm_a, p)?;
    let gb = find_gaps(chm_b, p)?;
    let mut cells = vec![0u8; n];
    for (k, cell) in cells.iter_mut().enumerate() {
        let (ia, ib) = (ga.labels[k] > 0, gb.labels[k] > 0);
        let data = chm_a.data[k].is_finite() && chm_b.data[k].is_finite();
        let cls = classes.map(|c| c[k]);
        *cell = if !data {
            5
        } else if ia && ib {
            1
        } else if ib {
            match cls {
                None | Some(3) => 2,
                Some(0) => 5,
                _ => 4,
            }
        } else if ia {
            match cls {
                None | Some(2) => 3,
                Some(0) => 5,
                _ => 4,
            }
        } else {
            0
        };
    }
    let cell = chm_a.resolution * chm_a.resolution;
    let mut areas = [0.0; 6];
    for &c in &cells {
        areas[c as usize] += cell;
    }
    let tally = |g: &Gaps, other: &Gaps, code: u8| -> Vec<(usize, usize, usize)> {
        let mut t = vec![(0usize, 0usize, 0usize); g.gaps.len()];
        for (k, &l) in g.labels.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let e = &mut t[l as usize - 1];
            e.0 += 1;
            if other.labels[k] > 0 {
                e.1 += 1;
            }
            if cells[k] == code {
                e.2 += 1;
            }
        }
        t
    };
    let (ta, tb) = (tally(&ga, &gb, 3), tally(&gb, &ga, 2));
    let status = |overlap: usize, changed: usize, gone: &'static str, partly: &'static str| -> &'static str {
        match (overlap > 0, changed > 0) {
            (false, true) => gone,
            (true, true) => partly,
            (true, false) => "stable",
            (false, false) => "uncertain",
        }
    };
    Ok(GapChange {
        status_a: ta.iter().map(|t| status(t.1, t.2, "closed", "shrunk")).collect(),
        closed_area: ta.iter().map(|t| t.2 as f64 * cell).collect(),
        status_b: tb.iter().map(|t| status(t.1, t.2, "new", "expanded")).collect(),
        formed_area: tb.iter().map(|t| t.2 as f64 * cell).collect(),
        a: ga,
        b: gb,
        cells,
        areas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chm(rows: &[&str]) -> Raster {
        // Rows given north first; '.' is gap (0 m), '#' canopy (20 m), '?' NaN.
        let nr = rows.len();
        let nc = rows[0].len();
        let mut r = Raster::filled(nr, nc, 0.0, 0.0, 1.0, 0.0);
        for (i, line) in rows.iter().enumerate() {
            for (c, ch) in line.chars().enumerate() {
                r.set(nr - 1 - i, c, match ch {
                    '.' => 0.5,
                    '?' => f64::NAN,
                    _ => 20.0,
                });
            }
        }
        r
    }

    #[test]
    fn gaps_are_connected_low_regions_with_outlines() {
        let r = chm(&["#########", "#...#####", "#.#.##..#", "#...##..#", "#####.###", "#########"]);
        let p = GapParams { min_area: 1.0, ..Default::default() };
        let g = find_gaps(&r, &p).unwrap();
        // The ring with an island, and the 2 x 2 square joined to the single cell at a corner.
        assert_eq!(g.gaps.len(), 2);
        let ring = g.gaps.iter().find(|x| x.n_cells == 8).unwrap();
        assert_eq!(ring.polygons.len(), 1);
        assert_eq!(ring.polygons[0].holes.len(), 1);
        let a: f64 = ring.polygons.iter().map(|q| ring_area(&q.exterior) + q.holes.iter().map(|h| ring_area(h)).sum::<f64>()).sum();
        assert!((a - 8.0).abs() < 1e-12, "{a}");
        let sq = g.gaps.iter().find(|x| x.n_cells == 5).unwrap();
        // Two parts touching at a corner.
        assert_eq!(sq.polygons.len(), 2);
        let parts: Vec<f64> = sq.polygons.iter().map(|q| ring_area(&q.exterior)).collect();
        assert!(parts.contains(&4.0) && parts.contains(&1.0), "{parts:?}");
        assert_eq!(sq.polygons.iter().find(|q| ring_area(&q.exterior) == 4.0).unwrap().exterior.len(), 4, "collinear corners dropped");
        // Edges only: the corner cell is its own region, too small here.
        let g4 = find_gaps(&r, &GapParams { connectivity: 4, min_area: 2.0, ..p }).unwrap();
        assert_eq!(g4.gaps.iter().map(|x| x.n_cells).collect::<Vec<_>>(), vec![8, 4]);
        assert!((g.gap_fraction() - 13.0 / 54.0).abs() < 1e-12);
    }

    #[test]
    fn formation_and_closure_need_significant_change() {
        let a = chm(&["######", "#..###", "#..###", "######"]);
        let b = chm(&["######", "######", "###..#", "###..#"]);
        let p = GapParams { min_area: 1.0, ..Default::default() };
        let c = gap_change(&a, &b, None, &p).unwrap();
        assert_eq!(c.status_a, vec!["closed"]);
        assert_eq!(c.status_b, vec!["new"]);
        assert_eq!(c.areas[2], 4.0);
        assert_eq!(c.areas[3], 4.0);
        // Only one of the new gap's cells changed significantly.
        let mut cls = vec![1u8; 24];
        let k = 3; // row 0 (south), col 3
        cls[k] = 3;
        for (i, &v) in a.data.iter().enumerate() {
            if v < 2.0 {
                cls[i] = 2;
            }
        }
        let c = gap_change(&a, &b, Some(&cls), &p).unwrap();
        assert_eq!(c.areas[2], 1.0);
        assert_eq!(c.areas[4], 3.0);
        assert_eq!(c.status_b, vec!["new"]);
        let (alpha, se, n) = size_exponent(&[10.0, 20.0, 40.0, 80.0], 10.0);
        assert_eq!(n, 4);
        assert!((alpha - (1.0 + 4.0 / (6.0 * 2.0f64.ln()))).abs() < 1e-12, "{alpha}");
        assert!(se > 0.0);
        assert!(size_exponent(&[5.0], 1.0).0.is_nan());
    }
}
