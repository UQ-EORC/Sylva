// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Tree lists after segmentation: pruning short candidates and duplicates
//! (with the point labels renumbered to match), and plot basal area.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::util::numeric::pairwise_sum;
use crate::trees::Tree;

/// Settings of [`prune_trees`].
#[derive(Debug, Clone)]
pub struct PruneParams {
    /// Trees lower than this (m) are dropped; NaN height counts as low.
    pub min_height: f64,
    /// Stems closer than this (m) are merged into the one with more points.
    pub merge_radius: f64,
    /// Stems wider than this (m) are dropped.
    pub max_dbh: Option<f64>,
    /// Minimum quality of candidates with fewer than `short_slices` layers.
    pub min_quality_short: f64,
    pub short_slices: i64,
    /// Stems wider than `slender_min_dbh` with height / DBH below this are
    /// dropped (shrubs and stumps fitted as stems); 0 disables.
    pub min_slenderness: f64,
    pub slender_min_dbh: f64,
}

impl Default for PruneParams {
    fn default() -> Self {
        PruneParams { min_height: 3.0, merge_radius: 0.2, max_dbh: None, min_quality_short: 0.0, short_slices: 4, min_slenderness: 0.0, slender_min_dbh: 0.2 }
    }
}

/// A map that keeps its keys in first-insertion order, as a Python dict does.
struct OrderedMap {
    keys: Vec<i64>,
    values: HashMap<i64, i64>,
}

impl OrderedMap {
    fn new() -> Self {
        OrderedMap { keys: Vec::new(), values: HashMap::new() }
    }

    fn insert(&mut self, k: i64, v: i64) {
        if self.values.insert(k, v).is_none() {
            self.keys.push(k);
        }
    }

    fn iter(&self) -> impl Iterator<Item = (i64, i64)> + '_ {
        self.keys.iter().map(|k| (*k, self.values[k]))
    }
}

/// A NumPy index into `len` values: negative indices count from the end.
fn wrap_index(i: i64, len: usize) -> Result<usize> {
    let j = if i < 0 { i + len as i64 } else { i };
    if j < 0 || j >= len as i64 {
        return Err(Error::invalid(format!("tree id {i} is out of range for the labels")));
    }
    Ok(j as usize)
}

/// Survivors of [`prune_trees`], each with the index of the input tree it
/// came from, and the renumbered labels.
pub type Pruned = (Vec<(usize, Tree)>, Vec<i64>);

/// Drop short candidates and merge duplicates after segmentation.
///
/// Trees lower than `min_height` (NaN counts as low), wider than `max_dbh`,
/// or supported by fewer than `short_slices` layers with a quality below
/// `min_quality_short`, or wider than `slender_min_dbh` with a height / DBH
/// ratio below `min_slenderness`, are removed. The rest are taken by decreasing point
/// count; one within `merge_radius` of an earlier survivor is absorbed by
/// it. Survivors are ordered by decreasing DBH (stable; a NaN DBH goes
/// last) and renumbered 1..n, and the labels follow: points of absorbed
/// trees take the survivor's id, points of dropped trees -1.
///
/// Returns each survivor with the index of the input tree it came from
/// (so that callers can carry per-tree extras along), its new id and
/// recounted `n_points`, and the new labels.
pub fn prune_trees(trees: &[Tree], labels: &[i64], p: &PruneParams) -> Result<Pruned> {
    let Some(&label_max) = labels.iter().max() else {
        return Err(Error::invalid("labels are empty"));
    };
    let mut keep: Vec<usize> = (0..trees.len())
        .filter(|&i| {
            let t = &trees[i];
            let slender = p.min_slenderness <= 0.0 || !(t.dbh > p.slender_min_dbh) || t.height / t.dbh >= p.min_slenderness;
            t.height >= p.min_height && p.max_dbh.is_none_or(|m| t.dbh <= m) && (t.n_slices as i64 >= p.short_slices || t.quality >= p.min_quality_short) && slender
        })
        .collect();
    keep.sort_by(|&a, &b| trees[b].n_points.cmp(&trees[a].n_points));
    let mut survivors: Vec<usize> = Vec::new();
    let mut absorbed = OrderedMap::new();
    for &i in &keep {
        let t = &trees[i];
        match survivors.iter().find(|&&s| (t.x - trees[s].x).hypot(t.y - trees[s].y) <= p.merge_radius) {
            Some(&s) => absorbed.insert(t.tree_id, trees[s].tree_id),
            None => survivors.push(i),
        }
    }
    // Largest DBH first; NaN compares equal to everything in Python, here it goes last.
    survivors.sort_by(|&a, &b| {
        let (da, db) = (trees[a].dbh, trees[b].dbh);
        match (da.is_nan(), db.is_nan()) {
            (false, false) => db.partial_cmp(&da).unwrap(),
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
        }
    });
    let mut new_id = OrderedMap::new();
    for (k, &i) in survivors.iter().enumerate() {
        new_id.insert(trees[i].tree_id, k as i64 + 1);
    }
    let id_max = trees.iter().map(|t| t.tree_id).max().unwrap_or(0);
    let size = label_max.max(id_max) + 2;
    if size <= 0 {
        return Err(Error::invalid("no labels or tree ids to prune"));
    }
    let mut lut = vec![-1i64; size as usize];
    for (old, new) in new_id.iter() {
        lut[wrap_index(old, size as usize)?] = new;
    }
    for (old, target) in absorbed.iter() {
        lut[wrap_index(old, size as usize)?] = new_id.values[&target];
    }
    let last = lut.len() as i64 - 1;
    let out_labels: Vec<i64> = labels.iter().map(|&l| if l >= 0 { lut[l.clamp(0, last) as usize] } else { -1 }).collect();
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &l in &out_labels {
        *counts.entry(l).or_insert(0) += 1;
    }
    let out = survivors
        .iter()
        .map(|&i| {
            let id = new_id.values[&trees[i].tree_id];
            let t = Tree { tree_id: id, n_points: counts.get(&id).copied().unwrap_or(0), ..trees[i].clone() };
            (i, t)
        })
        .collect();
    Ok((out, out_labels))
}

/// Basal area (m²/ha) of stems with `dbh >= min_dbh` over `area` m²; NaN
/// DBHs are left out.
pub fn basal_area(dbh: &[f64], area: f64, min_dbh: f64) -> Result<f64> {
    if area.is_nan() || area <= 0.0 {
        return Err(Error::invalid(format!("area must be positive, got {area:?}")));
    }
    let cross: Vec<f64> = dbh.iter().filter(|d| d.is_finite() && **d >= min_dbh).map(|d| std::f64::consts::PI * ((d / 2.0) * (d / 2.0))).collect();
    Ok(pairwise_sum(&cross) / area * 1e4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(id: i64, x: f64, dbh: f64, height: f64, n_points: usize) -> Tree {
        Tree { tree_id: id, x, y: 0.0, dbh, height, n_points, inlier_fraction: 0.5, n_slices: 5, rmse: 0.01, lean_deg: 1.0, quality: 0.5 }
    }

    #[test]
    fn prune_merges_duplicates_and_drops_short_trees() {
        let trees = vec![tree(1, 0.0, 0.3, 20.0, 3), tree(2, 0.1, 0.5, 20.0, 1), tree(3, 5.0, 0.6, 20.0, 2), tree(4, 9.0, 0.4, 1.0, 1)];
        let labels = vec![1, 1, 1, 2, 3, 3, 4, -1];
        let (out, lab) = prune_trees(&trees, &labels, &PruneParams::default()).unwrap();
        // Tree 3 (DBH 0.6) comes first, tree 1 absorbs tree 2, tree 4 is too low.
        let ids: Vec<(usize, i64, usize)> = out.iter().map(|(i, t)| (*i, t.tree_id, t.n_points)).collect();
        assert_eq!(ids, vec![(2, 1, 2), (0, 2, 4)]);
        assert_eq!(lab, vec![2, 2, 2, 2, 1, 1, -1, -1]);
    }

    #[test]
    fn prune_drops_squat_stems_only_when_asked() {
        // A shrub fitted as a 1.2 m stem 3 m tall, a real tree (0.8 m, 40 m),
        // a thin sapling below the DBH gate (0.1 m, 1.5 m) and a NaN DBH.
        let trees = vec![tree(1, 0.0, 1.2, 3.0, 1), tree(2, 5.0, 0.8, 40.0, 1), tree(3, 10.0, 0.1, 1.5 + 2.0, 1), tree(4, 15.0, f64::NAN, 5.0, 1)];
        let labels = vec![1, 2, 3, 4];
        let (all, _) = prune_trees(&trees, &labels, &PruneParams { min_height: 0.0, ..PruneParams::default() }).unwrap();
        assert_eq!(all.len(), 4);
        let p = PruneParams { min_height: 0.0, min_slenderness: 10.0, ..PruneParams::default() };
        let (out, lab) = prune_trees(&trees, &labels, &p).unwrap();
        let kept: Vec<usize> = out.iter().map(|(i, _)| *i).collect();
        assert_eq!(kept, vec![1, 2, 3]);
        assert_eq!(lab[0], -1);
    }

    #[test]
    fn prune_needs_labels() {
        assert!(prune_trees(&[], &[], &PruneParams::default()).is_err());
    }

    #[test]
    fn basal_area_skips_nan_and_small_stems() {
        let ba = basal_area(&[0.2, 0.4, f64::NAN, 0.05], 1000.0, 0.1).unwrap();
        let want = std::f64::consts::PI * (0.1f64.powi(2) + 0.2f64.powi(2)) / 1000.0 * 1e4;
        assert!((ba - want).abs() < 1e-12);
        assert!(basal_area(&[0.2], 0.0, 0.0).is_err());
        assert_eq!(basal_area(&[], 1.0, 0.0).unwrap(), 0.0);
    }
}
