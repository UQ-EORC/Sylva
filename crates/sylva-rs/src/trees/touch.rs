// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Merging segmented trees that lie against each other.
//!
//! Stem detection favours recall, so some candidates are not trees of their
//! own: a second circle on a wide trunk, a low fork, a whorl of branches
//! fitted as a stem. [`segment_trees`](crate::trees::segment_trees) grows
//! each of them like any other stem and it comes away holding a piece of
//! its parent's crown. Such a piece lies against the parent along much of
//! its extent, whereas two separate trees touch only where their crowns
//! meet. [`merge_touching`] measures that and joins the pieces back.

use std::collections::{BTreeMap, HashMap, HashSet};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::filters::voxel_downsample_indices;
use crate::trees::Tree;
use crate::util::spatial::KdTree;
use crate::Point;

/// Settings of [`merge_touching`].
#[derive(Debug, Clone)]
pub struct TouchParams {
    /// A tree is merged into a larger one when at least this share (0..1) of
    /// its nodes touch the larger tree; 0 or less merges nothing.
    pub threshold: f64,
    /// Spacing of the thinned cloud the contact is measured on (m): one node
    /// per voxel, so that the share does not depend on the point density.
    pub voxel_size: f64,
    /// A node touches another tree when one of its `neighbours` nearest
    /// nodes, no farther than this (m), belongs to that tree.
    pub reach: f64,
    /// Points below this height above ground (m) are left out: near the
    /// ground every tree keeps a disc of litter and grass about its stem,
    /// and neighbouring discs touch whatever the trees do.
    pub min_height: f64,
    /// Nearest nodes examined about each node (count).
    pub neighbours: usize,
    /// Two trees are merged only if the highest node of each is at least
    /// this high (m above ground). Low clumps that touch a trunk are mostly
    /// shrubs and saplings: left alone they are dropped by
    /// [`prune_trees`](crate::trees::prune::prune_trees), merged they would
    /// add understorey to the tree.
    pub min_top: f64,
}

impl Default for TouchParams {
    fn default() -> Self {
        TouchParams { threshold: 0.1, voxel_size: 0.05, reach: 0.1, min_height: 0.5, neighbours: 8, min_top: 3.0 }
    }
}

/// How much each pair of neighbouring trees touch, on the thinned cloud.
#[derive(Debug, Clone, Default)]
pub struct Contacts {
    /// Nodes of each tree (tree id -> count).
    pub nodes: BTreeMap<i64, usize>,
    /// `(a, b)` -> nodes of tree `a` that touch tree `b`. Not symmetric.
    pub touching: BTreeMap<(i64, i64), usize>,
    /// Height above ground of each tree's highest node (m).
    pub top: BTreeMap<i64, f64>,
}

/// Count the contacts between the labelled trees: the points at or above
/// `min_height` with a positive label are thinned to one node per
/// `voxel_size` voxel (the first point of each), and a node of tree `a`
/// touches tree `b` when one of its `neighbours` nearest nodes within
/// `reach` carries the label `b`.
pub fn contacts(points: &[Point], heights: &[f64], labels: &[i64], p: &TouchParams) -> Contacts {
    let sel: Vec<usize> = (0..points.len()).filter(|&i| labels[i] > 0 && heights[i] >= p.min_height).collect();
    let sub: Vec<Point> = sel.iter().map(|&i| points[i]).collect();
    let keep = voxel_downsample_indices(&sub, p.voxel_size);
    let nodes: Vec<Point> = keep.iter().map(|&i| sub[i]).collect();
    let lab: Vec<i64> = keep.iter().map(|&i| labels[sel[i]]).collect();
    let mut out = Contacts::default();
    for (&l, &i) in lab.iter().zip(&keep) {
        *out.nodes.entry(l).or_insert(0) += 1;
        let top = out.top.entry(l).or_insert(f64::NEG_INFINITY);
        *top = top.max(heights[sel[i]]);
    }
    if nodes.is_empty() {
        return out;
    }
    let tree = KdTree::new(&nodes);
    // For each node, the other trees it touches (each once).
    let touched: Vec<Vec<i64>> = nodes
        .par_iter()
        .enumerate()
        .map(|(i, q)| {
            let mut others: Vec<i64> = tree.knn(q, p.neighbours + 1).into_iter().filter(|&(j, d)| j != i && d <= p.reach && lab[j] != lab[i]).map(|(j, _)| lab[j]).collect();
            others.sort_unstable();
            others.dedup();
            others
        })
        .collect();
    for (i, others) in touched.iter().enumerate() {
        for &o in others {
            *out.touching.entry((lab[i], o)).or_insert(0) += 1;
        }
    }
    out
}

/// Which trees to merge, from their [`Contacts`]: `tree id -> the id it ends
/// up in`, for the merged trees only.
///
/// Greedy agglomeration. Of all pairs of touching trees, the one in which the
/// largest share of the smaller tree's nodes touch the larger is merged
/// first, if that share reaches `threshold`; the merged tree keeps the
/// larger one's id, the two node counts add, and so do their contacts with
/// every other tree. This repeats until no pair reaches the threshold.
/// A pair is merged only if both its trees reach `min_top`, and a tree in
/// `protected` is never merged into another. Ties go to the lower id of
/// the smaller tree, then of the larger.
pub fn merge_plan(c: &Contacts, threshold: f64, min_top: f64, protected: &HashSet<i64>) -> HashMap<i64, i64> {
    let mut into: HashMap<i64, i64> = HashMap::new();
    if !(threshold > 0.0) {
        return into;
    }
    let mut n: BTreeMap<i64, f64> = c.nodes.iter().map(|(&k, &v)| (k, v as f64)).collect();
    let mut top: BTreeMap<i64, f64> = c.top.clone();
    // nb[a][b]: nodes of a touching b; both directions are kept as rows.
    let mut nb: BTreeMap<i64, BTreeMap<i64, f64>> = BTreeMap::new();
    for (&(a, b), &v) in &c.touching {
        nb.entry(a).or_default().insert(b, v as f64);
        nb.entry(b).or_default().entry(a).or_insert(0.0);
    }
    loop {
        let mut best: Option<(f64, i64, i64)> = None;
        for (&a, row) in &nb {
            if protected.contains(&a) || !(top.get(&a).copied().unwrap_or(f64::NEG_INFINITY) >= min_top) {
                continue;
            }
            for (&b, &t) in row {
                // `a` is the smaller of the pair (the lower id of two equals),
                // and the larger must be a tree too: a tall piece is not
                // merged into a low clump, which would carry the clump past
                // pruning.
                if a == b || n[&a] > n[&b] || (n[&a] == n[&b] && a > b) || !(top.get(&b).copied().unwrap_or(f64::NEG_INFINITY) >= min_top) {
                    continue;
                }
                let share = t / n[&a].max(1.0);
                if share >= threshold && best.is_none_or(|(s, ..)| share > s) {
                    best = Some((share, a, b));
                }
            }
        }
        let Some((_, small, big)) = best else { break };
        into.insert(small, big);
        let n_small = n[&small];
        *n.get_mut(&big).unwrap() += n_small;
        let t_small = top.get(&small).copied().unwrap_or(f64::NEG_INFINITY);
        let t_big = top.entry(big).or_insert(f64::NEG_INFINITY);
        *t_big = t_big.max(t_small);
        let row = nb.remove(&small).unwrap_or_default();
        for (&o, &t) in &row {
            let back = nb.get_mut(&o).and_then(|r| r.remove(&small)).unwrap_or(0.0);
            if o != big {
                // Nodes of the merged tree touching `o`, and of `o` touching it.
                *nb.entry(big).or_default().entry(o).or_insert(0.0) += t;
                let cap = n[&o];
                let e = nb.entry(o).or_default().entry(big).or_insert(0.0);
                *e = (*e + back).min(cap);
            }
        }
        if let Some(r) = nb.get_mut(&big) {
            r.remove(&small);
        }
    }
    // Follow each merged tree to the tree it finally belongs to.
    let keys: Vec<i64> = into.keys().copied().collect();
    let mut out = HashMap::with_capacity(keys.len());
    for k in keys {
        let mut r = k;
        while let Some(&next) = into.get(&r) {
            r = next;
        }
        out.insert(k, r);
    }
    out
}

/// Merge trees that lie against a larger neighbour, after segmentation.
///
/// The contact between every pair of labelled trees is counted on a thinned
/// cloud ([`contacts`]) and trees are merged while the share of the smaller
/// tree's nodes touching the larger reaches `threshold` ([`merge_plan`]).
/// The points of a merged tree take the id of the tree it joins, which is
/// always the larger of the two at the time. Trees that do not reach
/// `min_top` take no part, and are left for pruning to drop.
///
/// Returns the surviving trees, in their input order and otherwise
/// unchanged (their `height` and `n_points` are those of before the merge:
/// run [`tree_heights`](crate::trees::tree_heights) afterwards), the new
/// labels, and for every input tree the id it now belongs to (its own if it
/// survives).
///
/// On the four reference plots of the tree benchmark, two held-out plots
/// and eight simulated ones, a threshold of 0.1 removed spurious trees and
/// lost none in any run; see *Benchmarks > Tree detection*.
pub fn merge_touching(points: &[Point], heights: &[f64], labels: &[i64], trees: &[Tree], p: &TouchParams) -> Result<(Vec<Tree>, Vec<i64>, Vec<i64>)> {
    if heights.len() != points.len() || labels.len() != points.len() {
        return Err(Error::invalid(format!("heights ({}) and labels ({}) must have one entry per point ({})", heights.len(), labels.len(), points.len())));
    }
    if !(p.voxel_size > 0.0) || !(p.reach > 0.0) {
        return Err(Error::invalid(format!("voxel_size and reach must be positive, got {} and {}", p.voxel_size, p.reach)));
    }
    let c = contacts(points, heights, labels, p);
    // Only labels that belong to a listed tree can be merged or merged into.
    let known: HashSet<i64> = trees.iter().map(|t| t.tree_id).collect();
    let foreign: HashSet<i64> = c.nodes.keys().copied().filter(|k| !known.contains(k)).collect();
    let c = if foreign.is_empty() {
        c
    } else {
        Contacts {
            nodes: c.nodes.into_iter().filter(|(k, _)| !foreign.contains(k)).collect(),
            touching: c.touching.into_iter().filter(|((a, b), _)| !foreign.contains(a) && !foreign.contains(b)).collect(),
            top: c.top.into_iter().filter(|(k, _)| !foreign.contains(k)).collect(),
        }
    };
    let plan = merge_plan(&c, p.threshold, p.min_top, &HashSet::new());
    let merged_into: Vec<i64> = trees.iter().map(|t| plan.get(&t.tree_id).copied().unwrap_or(t.tree_id)).collect();
    let kept: Vec<Tree> = trees.iter().filter(|t| !plan.contains_key(&t.tree_id)).cloned().collect();
    let out: Vec<i64> = labels.iter().map(|l| plan.get(l).copied().unwrap_or(*l)).collect();
    Ok((kept, out, merged_into))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(id: i64, x: f64) -> Tree {
        Tree { tree_id: id, x, y: 0.0, dbh: 0.2, height: f64::NAN, n_points: 0, inlier_fraction: 0.5, n_slices: 5, rmse: 0.01, lean_deg: 1.0, quality: 0.5 }
    }

    /// Spacing of the test points (m): wider than a voxel, so thinning keeps every one.
    const S: f64 = 0.06;

    /// A vertical sheet of `columns` columns of points `S` apart, from `x0`, heights 1 to 3.3 m.
    fn sheet(x0: f64, columns: usize, label: i64, pts: &mut Vec<Point>, h: &mut Vec<f64>, lab: &mut Vec<i64>) {
        for c in 0..columns {
            for r in 0..40 {
                let z = 1.0 + S * r as f64;
                pts.push([x0 + S * c as f64, 0.0, z]);
                h.push(z);
                lab.push(label);
            }
        }
    }

    #[test]
    fn a_piece_lying_against_a_tree_joins_it_and_a_neighbour_does_not() {
        // Tree 1: a sheet of 34 columns. Tree 2: four more columns of the
        // same sheet that a second stem took. Tree 3: a sheet as large as
        // tree 1 whose edge column stands next to tree 1's other edge.
        let (mut pts, mut h, mut lab) = (Vec::new(), Vec::new(), Vec::new());
        sheet(0.0, 34, 1, &mut pts, &mut h, &mut lab);
        sheet(34.0 * S, 4, 2, &mut pts, &mut h, &mut lab);
        sheet(-34.0 * S, 34, 3, &mut pts, &mut h, &mut lab);
        let trees = vec![tree(1, 1.0), tree(2, 2.1), tree(3, -1.0)];
        let c = contacts(&pts, &h, &lab, &TouchParams::default());
        assert_eq!(c.nodes[&1], 34 * 40);
        assert!((c.top[&2] - (1.0 + 39.0 * S)).abs() < 1e-9);
        // One column in four of the piece touches tree 1; one in 34 of tree 3.
        let share = |a: i64, b: i64| c.touching.get(&(a, b)).copied().unwrap_or(0) as f64 / c.nodes[&a] as f64;
        assert!((share(2, 1) - 0.25).abs() < 1e-9, "{}", share(2, 1));
        assert!((share(3, 1) - 1.0 / 34.0).abs() < 1e-9, "{}", share(3, 1));
        let (kept, out, into) = merge_touching(&pts, &h, &lab, &trees, &TouchParams::default()).unwrap();
        assert_eq!(kept.iter().map(|t| t.tree_id).collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(into, vec![1, 1, 3]);
        assert!(out.iter().zip(&lab).all(|(o, l)| *o == if *l == 2 { 1 } else { *l }));
        // A piece that does not reach `min_top` is left for pruning.
        let (kept, _, _) = merge_touching(&pts, &h, &lab, &trees, &TouchParams { min_top: 4.0, ..Default::default() }).unwrap();
        assert_eq!(kept.len(), 3);
        // A threshold of 0 merges nothing, and so does one above every share.
        for t in [0.0, 0.9] {
            let (kept, out, _) = merge_touching(&pts, &h, &lab, &trees, &TouchParams { threshold: t, ..Default::default() }).unwrap();
            assert_eq!(kept.len(), 3);
            assert_eq!(out, lab);
        }
    }

    #[test]
    fn the_larger_tree_survives_and_chains_resolve_to_one_id() {
        // 3 lies against 2, which lies against 1: all end in tree 1.
        let c = Contacts {
            nodes: [(1, 1000), (2, 100), (3, 10)].into_iter().collect(),
            touching: [((2, 1), 30), ((1, 2), 30), ((3, 2), 5), ((2, 3), 5)].into_iter().collect(),
            top: [(1, 20.0), (2, 8.0), (3, 2.0)].into_iter().collect(),
        };
        let plan = merge_plan(&c, 0.1, 0.0, &HashSet::new());
        assert_eq!(plan.get(&3), Some(&1));
        assert_eq!(plan.get(&2), Some(&1));
        assert_eq!(plan.len(), 2);
        // A protected tree stays, and what lay against it stays with it.
        let plan = merge_plan(&c, 0.1, 0.0, &[2].into_iter().collect());
        assert_eq!(plan.get(&3), Some(&2));
        assert_eq!(plan.len(), 1);
        // A clump 2 m tall is left for pruning; the 8 m piece still joins its tree.
        let plan = merge_plan(&c, 0.1, 3.0, &HashSet::new());
        assert_eq!(plan.get(&2), Some(&1));
        assert_eq!(plan.len(), 1);
        // Nor is a tall piece merged into a low clump that has more nodes.
        let c = Contacts {
            nodes: [(1, 50), (2, 400)].into_iter().collect(),
            touching: [((1, 2), 20), ((2, 1), 20)].into_iter().collect(),
            top: [(1, 6.0), (2, 2.5)].into_iter().collect(),
        };
        assert!(merge_plan(&c, 0.1, 3.0, &HashSet::new()).is_empty());
        assert_eq!(merge_plan(&c, 0.1, 0.0, &HashSet::new()).get(&1), Some(&2));
    }

    #[test]
    fn points_near_the_ground_and_unlabelled_points_do_not_count() {
        // Two stems 12 cm apart. Below 0.5 m the points between them belong
        // to tree 2, so there the trees are side by side; above it those
        // points are unlabelled, and the trees are out of each other's reach.
        let (mut pts, mut h, mut lab) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..40 {
            let z = S * k as f64;
            for (x, l) in [(0.0, 1), (S, if z < 0.5 { 2 } else { -1 }), (2.0 * S, 2)] {
                pts.push([x, 0.0, z]);
                h.push(z);
                lab.push(l);
            }
        }
        let c = contacts(&pts, &h, &lab, &TouchParams::default());
        assert!(c.touching.is_empty(), "{:?}", c.touching);
        // With the ground layer counted they do touch.
        let c = contacts(&pts, &h, &lab, &TouchParams { min_height: 0.0, ..Default::default() });
        assert!(c.touching[&(1, 2)] > 0);
        // One height and one label per point.
        assert!(merge_touching(&pts, &h[..3], &lab, &[tree(1, 0.0)], &TouchParams::default()).is_err());
    }
}
