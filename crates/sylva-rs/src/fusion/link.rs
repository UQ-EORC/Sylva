// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! TLS stems linked to ALS trees.
//!
//! A TLS tree (stem position, DBH, height) and an ALS tree (top, height,
//! crown outline) can be the same tree when the stem lies inside the crown
//! (or within `crown_buffer` of its outline) or within `max_distance` of the
//! top. Among those pairs an optimal assignment (Kuhn 1955; Munkres 1957,
//! as in [`crate::change::trees::assign`]) minimises the total cost, a
//! stem left without an ALS tree costing `max_cost`. A pair costs
//!
//! `(d / D)² + height_weight * ((h_als - h_tls) / h_als)² + dbh_weight * (1 - dbh / dbh_max)²`
//!
//! with `d` the stem-to-top distance, `D` the larger of `max_distance` and
//! the crown's equivalent radius, and `dbh_max` the largest DBH among the
//! stems that could be that ALS tree: the stem that is tallest by the TLS
//! and thickest under a crown is its tree, since the dominant tree of a
//! crown is usually both. (A stem without a TLS height is charged
//! `dh = 0.3`, a typical shortfall of a TLS height under a closed canopy.)
//! Each ALS tree is linked to at most one stem. The stems under a crown that
//! are not its tree are reported with it:
//!
//! * `suppressed`: shorter than the ALS tree by more than `top_tolerance`
//!   and 10 %, an understorey or overtopped tree the ALS cannot see;
//! * `codominant`: about as tall, so the crown found by the ALS holds two
//!   canopy trees (the ALS segmentation merged them).
//!
//! The per-tree table keeps the TLS DBH and takes the height from the
//! instrument that saw the top: for a linked tree the ALS height, unless the
//! TLS is known to have seen the top too and measured it taller (both are
//! lower bounds, the ALS for missing the apex, the TLS for occlusion; a TLS
//! height can also be too tall where its segmentation gave a tree part of a
//! neighbour's crown, so an inferred sighting does not overrule the ALS). Whether the TLS saw the
//! top is given per tree (e.g. from `voxels.tree_sampling`) or, for a linked
//! tree, inferred from the TLS height reaching the ALS height within
//! `top_tolerance`.

use std::collections::HashMap;

use crate::change::trees::assign;
use crate::error::{Error, Result};

/// A tree from the TLS: stem position in the ALS frame, DBH (m), height
/// (m), wood volume (m³), and whether its top was seen (1, 0 or NaN for
/// unknown).
#[derive(Debug, Clone, Copy)]
pub struct TlsTree {
    pub x: f64,
    pub y: f64,
    pub dbh: f64,
    pub height: f64,
    pub volume: f64,
    pub top_seen: f64,
}

/// A tree from the ALS: top position, height and crown outline (fewer than
/// three vertices for none).
#[derive(Debug, Clone)]
pub struct AlsTree {
    pub x: f64,
    pub y: f64,
    pub height: f64,
    pub crown: Vec<[f64; 2]>,
}

/// Settings of [`link_trees`].
#[derive(Debug, Clone)]
pub struct LinkParams {
    pub max_distance: f64,
    pub crown_buffer: f64,
    pub height_weight: f64,
    pub dbh_weight: f64,
    /// Cost of leaving a stem without an ALS tree: no pair costing more is made.
    pub max_cost: f64,
    pub top_tolerance: f64,
}

impl Default for LinkParams {
    fn default() -> Self {
        LinkParams { max_distance: 3.0, crown_buffer: 0.5, height_weight: 2.0, dbh_weight: 2.0, max_cost: 2.0, top_tolerance: 1.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsStatus {
    Matched,
    Suppressed,
    Codominant,
    Unlinked,
}

impl TlsStatus {
    pub fn name(self) -> &'static str {
        match self {
            TlsStatus::Matched => "matched",
            TlsStatus::Suppressed => "suppressed",
            TlsStatus::Codominant => "codominant",
            TlsStatus::Unlinked => "unlinked",
        }
    }
}

/// Result of [`link_trees`].
#[derive(Debug, Clone, Default)]
pub struct Links {
    /// Per TLS tree: status, the ALS tree it is linked to or stands under,
    /// the distance from its stem to that tree's top, whether the stem is
    /// inside that crown, and that tree's height.
    pub tls_status: Vec<TlsStatus>,
    pub tls_als: Vec<Option<usize>>,
    pub distance: Vec<f64>,
    pub inside: Vec<bool>,
    pub als_height: Vec<f64>,
    /// Combined height, where it came from (`"als"`, `"tls"`, `"none"`),
    /// whether the TLS saw the top (1, 0, NaN) and a flag: `"top_seen"`,
    /// `"als_height"` (TLS top not seen, ALS height used), `"top_not_seen"`
    /// (TLS top not seen and no ALS height: a lower bound) or `"top_unknown"`.
    pub height: Vec<f64>,
    pub height_source: Vec<&'static str>,
    pub top_seen: Vec<f64>,
    pub flag: Vec<&'static str>,
    /// Per ALS tree: its linked stem and every stem linked to it or under it.
    pub als_tls: Vec<Option<usize>>,
    pub als_stems: Vec<Vec<usize>>,
    /// Assignment cost of each linked TLS tree (NaN otherwise).
    pub cost: Vec<f64>,
}

/// `(inside, distance to the outline)` of a point and a polygon.
pub fn polygon_relation(x: f64, y: f64, poly: &[[f64; 2]]) -> (bool, f64) {
    let n = poly.len();
    if n < 3 {
        return (false, f64::INFINITY);
    }
    let mut inside = false;
    let mut dmin = f64::INFINITY;
    for k in 0..n {
        let (a, b) = (poly[k], poly[(k + 1) % n]);
        if (a[1] > y) != (b[1] > y) {
            let xc = a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if x < xc {
                inside = !inside;
            }
        }
        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
        let len2 = ex * ex + ey * ey;
        let t = if len2 > 0.0 { (((x - a[0]) * ex + (y - a[1]) * ey) / len2).clamp(0.0, 1.0) } else { 0.0 };
        dmin = dmin.min((x - a[0] - t * ex).hypot(y - a[1] - t * ey));
    }
    (inside, dmin)
}

fn polygon_area(poly: &[[f64; 2]]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let n = poly.len();
    (0..n).map(|k| poly[k][0] * poly[(k + 1) % n][1] - poly[(k + 1) % n][0] * poly[k][1]).sum::<f64>().abs() * 0.5
}

fn find(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
        r = parent[r];
    }
    let mut k = i;
    while parent[k] != r {
        let next = parent[k];
        parent[k] = r;
        k = next;
    }
    r
}

/// Minimum-cost matching of the allowed pairs `(i, j, cost)` between `na`
/// and `nb` items, solved in each group of items linked by allowed pairs.
/// With `unmatched`, leaving an item of the first set without a partner
/// costs that much, so no pair costing more is made; without it, as many
/// pairs as possible are made first.
pub fn assign_pairs(na: usize, nb: usize, edges: &[(usize, usize, f64)], unmatched: Option<f64>) -> Vec<(usize, usize)> {
    let mut parent: Vec<usize> = (0..na + nb).collect();
    for &(i, j, _) in edges {
        let (ri, rj) = (find(&mut parent, i), find(&mut parent, na + j));
        if ri != rj {
            parent[ri.max(rj)] = ri.min(rj);
        }
    }
    let mut groups: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
    for i in 0..na {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().0.push(i);
    }
    for j in 0..nb {
        let r = find(&mut parent, na + j);
        groups.entry(r).or_default().1.push(j);
    }
    let cost_of: HashMap<(usize, usize), f64> = edges.iter().map(|&(i, j, c)| ((i, j), c)).collect();
    let mut roots: Vec<usize> = groups.keys().copied().collect();
    roots.sort_unstable();
    const FORBIDDEN: f64 = 1e9;
    let mut pairs = Vec::new();
    for r in roots {
        let (ga, gb) = &groups[&r];
        if ga.is_empty() || gb.is_empty() {
            continue;
        }
        if let Some(c0) = unmatched {
            // One column per ALS item, then one "unmatched" column per row.
            let cost: Vec<Vec<f64>> = ga
                .iter()
                .map(|&i| gb.iter().map(|&j| cost_of.get(&(i, j)).copied().unwrap_or(FORBIDDEN)).chain(std::iter::repeat_n(c0, ga.len())).collect())
                .collect();
            for (k, c) in assign(&cost).into_iter().enumerate() {
                if c < gb.len() && cost[k][c] < FORBIDDEN {
                    pairs.push((ga[k], gb[c]));
                }
            }
            continue;
        }
        let transpose = ga.len() > gb.len();
        let (rows, cols) = if transpose { (gb, ga) } else { (ga, gb) };
        let cost: Vec<Vec<f64>> = rows
            .iter()
            .map(|&ri| cols.iter().map(|&ci| {
                let (i, j) = if transpose { (ci, ri) } else { (ri, ci) };
                cost_of.get(&(i, j)).copied().unwrap_or(FORBIDDEN)
            }).collect())
            .collect();
        for (k, c) in assign(&cost).into_iter().enumerate() {
            if cost[k][c] < FORBIDDEN {
                pairs.push(if transpose { (cols[c], rows[k]) } else { (rows[k], cols[c]) });
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

/// Link TLS trees to ALS trees; see the module documentation.
///
/// # Errors
/// For a non-finite position or a setting out of range.
pub fn link_trees(tls: &[TlsTree], als: &[AlsTree], p: &LinkParams) -> Result<Links> {
    let nonneg = |v: f64| v.is_finite() && v >= 0.0;
    if !(p.max_distance.is_finite() && p.max_distance > 0.0) || !(p.max_cost.is_finite() && p.max_cost > 0.0) || !nonneg(p.crown_buffer) || !nonneg(p.height_weight) || !nonneg(p.dbh_weight) || !nonneg(p.top_tolerance) {
        return Err(Error::invalid("max_distance and max_cost must be positive; crown_buffer, height_weight, dbh_weight and top_tolerance >= 0"));
    }
    if tls.iter().any(|t| !t.x.is_finite() || !t.y.is_finite()) || als.iter().any(|t| !t.x.is_finite() || !t.y.is_finite()) {
        return Err(Error::invalid("tree positions must be finite"));
    }
    if als.iter().any(|t| t.crown.iter().any(|v| !v[0].is_finite() || !v[1].is_finite())) {
        return Err(Error::invalid("crown outlines must be finite"));
    }
    let (na, nb) = (tls.len(), als.len());
    // Candidate ALS trees of each stem, through a grid on the tops.
    let reach: Vec<f64> = als.iter().map(|a| {
        let far = a.crown.iter().map(|v| (v[0] - a.x).hypot(v[1] - a.y)).fold(0.0, f64::max);
        far.max(p.max_distance) + p.crown_buffer
    }).collect();
    let cell = reach.iter().copied().fold(p.max_distance, f64::max);
    let key = |x: f64, y: f64| ((x / cell).floor() as i64, (y / cell).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (j, a) in als.iter().enumerate() {
        grid.entry(key(a.x, a.y)).or_default().push(j);
    }
    let size: Vec<f64> = als.iter().map(|a| (polygon_area(&a.crown) / std::f64::consts::PI).sqrt().max(p.max_distance)).collect();
    // Per stem: (j, distance to top, inside, inside or near the crown).
    let mut rel: Vec<Vec<(usize, f64, bool, bool)>> = vec![Vec::new(); na];
    for (i, t) in tls.iter().enumerate() {
        let (kx, ky) = key(t.x, t.y);
        let mut near: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                near.extend(grid.get(&(kx + dx, ky + dy)).map(|v| v.as_slice()).unwrap_or(&[]));
            }
        }
        near.sort_unstable();
        for j in near {
            let a = &als[j];
            let d = (t.x - a.x).hypot(t.y - a.y);
            let (inside, edge) = polygon_relation(t.x, t.y, &a.crown);
            let under = if a.crown.len() >= 3 { inside || edge <= p.crown_buffer } else { d <= p.max_distance };
            if under || d <= p.max_distance {
                rel[i].push((j, d, inside, under));
            }
        }
    }
    // The thickest stem that could be each ALS tree.
    let mut dbh_max = vec![f64::NAN; nb];
    for (i, r) in rel.iter().enumerate() {
        for &(j, ..) in r {
            if tls[i].dbh.is_finite() && !(dbh_max[j] >= tls[i].dbh) {
                dbh_max[j] = tls[i].dbh;
            }
        }
    }
    let mut edges = Vec::new();
    for (i, r) in rel.iter().enumerate() {
        let t = &tls[i];
        for &(j, d, ..) in r {
            let a = &als[j];
            let dh = if !(a.height.is_finite() && a.height > 0.0) { 0.0 } else if t.height.is_finite() { (a.height - t.height) / a.height } else { 0.3 };
            let dd = if t.dbh.is_finite() && dbh_max[j] > 0.0 { 1.0 - t.dbh / dbh_max[j] } else { 0.0 };
            edges.push((i, j, (d / size[j]).powi(2) + p.height_weight * dh * dh + p.dbh_weight * dd * dd));
        }
    }
    let pairs = assign_pairs(na, nb, &edges, Some(p.max_cost));
    let cost_of: HashMap<(usize, usize), f64> = edges.iter().map(|&(i, j, c)| ((i, j), c)).collect();
    let mut out = Links {
        tls_status: vec![TlsStatus::Unlinked; na],
        tls_als: vec![None; na],
        distance: vec![f64::NAN; na],
        inside: vec![false; na],
        als_height: vec![f64::NAN; na],
        height: vec![f64::NAN; na],
        height_source: vec!["none"; na],
        top_seen: vec![f64::NAN; na],
        flag: vec!["top_unknown"; na],
        als_tls: vec![None; nb],
        als_stems: vec![Vec::new(); nb],
        cost: vec![f64::NAN; na],
    };
    for &(i, j) in &pairs {
        out.tls_status[i] = TlsStatus::Matched;
        out.tls_als[i] = Some(j);
        out.als_tls[j] = Some(i);
        out.cost[i] = cost_of[&(i, j)];
    }
    for i in 0..na {
        if out.tls_status[i] == TlsStatus::Matched {
            continue;
        }
        // The crown it stands under: inside first, then the nearest top.
        let pick = rel[i].iter().filter(|r| r.3).min_by(|a, b| (!a.2).cmp(&!b.2).then(a.1.total_cmp(&b.1)).then(a.0.cmp(&b.0)));
        if let Some(&(j, _, _, _)) = pick {
            let (h, ha) = (tls[i].height, als[j].height);
            let codominant = h.is_finite() && ha.is_finite() && h >= ha - p.top_tolerance.max(0.1 * ha);
            out.tls_status[i] = if codominant { TlsStatus::Codominant } else { TlsStatus::Suppressed };
            out.tls_als[i] = Some(j);
        }
    }
    for (i, t) in tls.iter().enumerate() {
        if let Some(j) = out.tls_als[i] {
            let a = &als[j];
            out.distance[i] = (t.x - a.x).hypot(t.y - a.y);
            out.inside[i] = polygon_relation(t.x, t.y, &a.crown).0;
            out.als_height[i] = a.height;
            out.als_stems[j].push(i);
        }
        let mut seen = t.top_seen;
        if out.tls_status[i] == TlsStatus::Matched {
            let ha = out.als_height[i];
            if seen.is_nan() && ha.is_finite() {
                seen = if t.height.is_finite() && t.height >= ha - p.top_tolerance { 1.0 } else { 0.0 };
            }
            // Only a top the TLS is known to have seen (not one inferred from
            // the heights agreeing) can overrule the ALS height.
            if ha.is_finite() && !(t.top_seen == 1.0 && t.height.is_finite() && t.height > ha) {
                out.height[i] = ha;
                out.height_source[i] = "als";
                out.flag[i] = if seen == 1.0 { "top_seen" } else { "als_height" };
            } else {
                out.height[i] = t.height;
                out.height_source[i] = if t.height.is_finite() { "tls" } else { "none" };
                out.flag[i] = if seen == 1.0 { "top_seen" } else if seen == 0.0 { "top_not_seen" } else { "top_unknown" };
            }
        } else {
            out.height[i] = t.height;
            out.height_source[i] = if t.height.is_finite() { "tls" } else { "none" };
            out.flag[i] = if seen == 1.0 { "top_seen" } else if seen == 0.0 { "top_not_seen" } else { "top_unknown" };
        }
        out.top_seen[i] = seen;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(cx: f64, cy: f64, r: f64) -> Vec<[f64; 2]> {
        vec![[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]
    }

    fn stem(x: f64, y: f64, h: f64) -> TlsTree {
        TlsTree { x, y, dbh: 0.3, height: h, volume: f64::NAN, top_seen: f64::NAN }
    }

    #[test]
    fn polygon_relations() {
        let sq = square(0.0, 0.0, 1.0);
        assert_eq!(polygon_relation(0.0, 0.0, &sq), (true, 1.0));
        let (inside, d) = polygon_relation(2.0, 0.0, &sq);
        assert!(!inside && (d - 1.0).abs() < 1e-12);
        assert!((polygon_area(&sq) - 4.0).abs() < 1e-12);
        assert!(!polygon_relation(0.0, 0.0, &sq[..2]).0);
    }

    #[test]
    fn the_tallest_stem_under_a_crown_is_its_tree() {
        let als = vec![
            AlsTree { x: 0.0, y: 0.0, height: 20.0, crown: square(0.0, 0.0, 3.0) },
            AlsTree { x: 10.0, y: 0.0, height: 15.0, crown: square(10.0, 0.0, 2.5) },
            AlsTree { x: 30.0, y: 0.0, height: 18.0, crown: square(30.0, 0.0, 3.0) },
        ];
        let tls = vec![
            stem(1.5, 0.5, 8.0),   // suppressed under tree 0, nearer its top than the dominant stem
            stem(-1.8, -1.0, 19.6), // the dominant tree of crown 0, top seen
            stem(10.3, 0.2, 14.0), // tree 1
            stem(11.5, 1.0, 14.8), // as tall, under the same crown: codominant
            stem(20.0, 0.0, 9.0),  // under no crown
        ];
        let l = link_trees(&tls, &als, &LinkParams::default()).unwrap();
        let st: Vec<&str> = l.tls_status.iter().map(|s| s.name()).collect();
        assert_eq!(st, ["suppressed", "matched", "matched", "codominant", "unlinked"]);
        assert_eq!(l.tls_als, [Some(0), Some(0), Some(1), Some(1), None]);
        assert_eq!(l.als_tls, [Some(1), Some(2), None]);
        assert_eq!((l.height[2], l.flag[2]), (15.0, "top_seen"));
        assert_eq!(l.als_stems, [vec![0, 1], vec![2, 3], vec![]]);
        assert_eq!(l.height[1], 20.0);
        assert_eq!(l.flag[1], "top_seen");
        assert_eq!(l.top_seen[1], 1.0);
        assert_eq!(l.height[0], 8.0);
        assert_eq!(l.height_source[0], "tls");
        assert_eq!(l.flag[4], "top_unknown");
        assert!(l.inside[0] && l.distance[1] > 2.0);
        // The TLS saw the top and measured it taller: its height is kept.
        let mut t2 = tls.clone();
        t2[1].height = 20.5;
        t2[1].top_seen = 1.0;
        let l2 = link_trees(&t2, &als, &LinkParams::default()).unwrap();
        assert_eq!((l2.height[1], l2.height_source[1]), (20.5, "tls"));
        // Top given as not seen: the ALS height replaces the TLS one.
        t2[1].top_seen = 0.0;
        t2[1].height = 14.0;
        let l3 = link_trees(&t2, &als, &LinkParams::default()).unwrap();
        assert_eq!((l3.height[1], l3.flag[1]), (20.0, "als_height"));
    }

    #[test]
    fn the_thickest_stem_wins_without_heights() {
        let als = vec![AlsTree { x: 0.0, y: 0.0, height: 20.0, crown: square(0.0, 0.0, 4.0) }];
        let mut thin = stem(1.0, 0.0, f64::NAN);
        thin.dbh = 0.15;
        let mut thick = stem(-1.5, 0.0, f64::NAN);
        thick.dbh = 0.45;
        let l = link_trees(&[thin, thick], &als, &LinkParams::default()).unwrap();
        assert_eq!(l.als_tls, [Some(1)]);
        assert_eq!(l.tls_status[0], TlsStatus::Suppressed);
        // Without the DBH term, the nearer stem would be taken.
        let l = link_trees(&[thin, thick], &als, &LinkParams { dbh_weight: 0.0, ..Default::default() }).unwrap();
        assert_eq!(l.als_tls, [Some(0)]);
    }

    #[test]
    fn without_crowns_distance_decides() {
        let als = vec![AlsTree { x: 0.0, y: 0.0, height: 20.0, crown: vec![] }, AlsTree { x: 4.0, y: 0.0, height: 20.0, crown: vec![] }];
        let tls = vec![stem(3.0, 0.0, 19.0), stem(0.5, 0.0, 19.0)];
        let l = link_trees(&tls, &als, &LinkParams::default()).unwrap();
        // Nearest-neighbour would give both stems to one top; the assignment
        // gives each its own.
        assert_eq!(l.tls_als, [Some(1), Some(0)]);
        assert!(link_trees(&[], &als, &LinkParams::default()).unwrap().als_tls.iter().all(|v| v.is_none()));
        assert!(link_trees(&tls, &[], &LinkParams::default()).unwrap().tls_status.iter().all(|s| *s == TlsStatus::Unlinked));
        // A stem too unlike the tree stays unmatched, under it.
        let short = vec![stem(2.9, 0.0, 5.0)];
        let l = link_trees(&short, &als[..1], &LinkParams::default()).unwrap();
        assert_eq!((l.tls_status[0], l.tls_als[0]), (TlsStatus::Suppressed, Some(0)));
        let bad = LinkParams { max_distance: 0.0, ..Default::default() };
        assert!(link_trees(&tls, &als, &bad).is_err());
        assert!(link_trees(&[stem(f64::NAN, 0.0, 1.0)], &als, &LinkParams::default()).is_err());
    }
}
