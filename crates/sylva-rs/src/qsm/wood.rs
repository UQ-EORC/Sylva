// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Leaf / wood separation for a single tree.
//!
//! Local anisotropy alone (planarity + linearity over `k` neighbours) drops
//! sparsely scanned trunks: an occluded stem seen as a few scattered patches
//! is neither planar nor linear at the neighbourhood scale, and without it
//! the QSM has no measurements on the very cylinders that carry the volume.
//! The answer here is topological: every path from the base to any part of
//! the crown runs through the trunk and the branches, so points that many
//! shortest paths pass through are wood whatever their local shape (the
//! path-frequency cue of Vicari et al. 2019).
//!
//! Steps: passage counting on a kNN graph from
//! the base, with coarse target cells so the count does not depend on point
//! density; neighbours of passage wood; high-likelihood anisotropy in
//! connected components that are large or touch passage wood; medium
//! likelihood only where it connects to wood already found; a final
//! dilation for thin branches.

use std::collections::HashSet;

use rayon::prelude::*;

use crate::cluster::{connected_components, dijkstra, knn_graph, radius_graph};
use crate::filters::{planarity_linearity, statistical_outlier_mask, voxel_downsample_indices};
use crate::spatial::KdTree;
use crate::Point;

#[derive(Debug, Clone)]
pub struct WoodParams {
    /// Neighbourhood for the anisotropy likelihood.
    pub k: usize,
    /// Planarity + linearity above this is high wood likelihood.
    pub high_threshold: f64,
    /// ... and above this medium likelihood, kept only next to wood after
    /// outlier removal. At or above `high_threshold` the step is off.
    pub medium_threshold: f64,
    /// Radius of the second, wider anisotropy neighbourhood (m): larger than a
    /// leaf, smaller than the spacing of branches. 0 (the default) uses `k`
    /// neighbours only: the wider scale labels foliage far better (leaf recall
    /// 0.1 -> 0.6 on synthetic leaf-on trees) but removes wood a QSM needs
    /// (harvest volume rRMSE 19.9 -> 22.9 %), so `classify_leaf_wood` turns it
    /// on and the QSM input filter does not.
    pub scale_radius: f64,
    /// kNN graph for the passage paths.
    pub graph_k: usize,
    pub max_edge: f64,
    /// Points within this height of the lowest point seed the paths.
    pub base_height: f64,
    /// Target cell size: one path per occupied cell.
    pub target_res: f64,
    /// A point crossed by at least this many target paths is wood.
    pub min_passage: usize,
    /// Neighbours of passage wood within this distance are wood too ...
    pub assign_dist: f64,
    /// ... plus this fraction of the tree height times the square root of
    /// the share of target paths the point carries (0 for a fixed reach).
    pub assign_scale: f64,
    /// Connectivity resolution and minimum size of a free-standing
    /// high-likelihood component.
    pub component_res: f64,
    pub component_min: usize,
    /// Statistical outlier removal on the medium-likelihood set before its
    /// components are formed: scattered foliage goes, bark
    /// patches stay.
    pub sor_k: usize,
    pub sor_std: f64,
    /// Final dilation radius.
    pub dilate_dist: f64,
    /// 0 disables the passage step (anisotropy only, the old behaviour).
    pub passage: bool,
}

impl Default for WoodParams {
    fn default() -> Self {
        WoodParams { k: 20, high_threshold: 0.85, medium_threshold: 0.75, scale_radius: 0.0, graph_k: 10, max_edge: 1.0, base_height: 0.25, target_res: 0.2, min_passage: 3, assign_dist: 0.05, assign_scale: 0.0, component_res: 0.05, component_min: 200, sor_k: 50, sor_std: 1.0, dilate_dist: 0.03, passage: true }
    }
}

/// Planarity + linearity, `1 - l_min / l_max`, of a neighbourhood; at most 64
/// of its points are used.
fn anisotropy(points: &[Point], nbrs: &[(usize, f64)]) -> f64 {
    if nbrs.len() < 4 {
        return 0.0;
    }
    let step = nbrs.len().div_ceil(64);
    let sel: Vec<usize> = nbrs.iter().step_by(step).map(|&(i, _)| i).collect();
    let n = sel.len() as f64;
    let mut c = [0.0; 3];
    for &i in &sel {
        for a in 0..3 {
            c[a] += points[i][a] / n;
        }
    }
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    for &i in &sel {
        let d = [points[i][0] - c[0], points[i][1] - c[1], points[i][2] - c[2]];
        for a in 0..3 {
            for b in 0..3 {
                cov[(a, b)] += d[a] * d[b];
            }
        }
    }
    let e = cov.symmetric_eigen().eigenvalues;
    let (lo, hi) = (e.min(), e.max());
    if hi > 1e-14 { 1.0 - (lo / hi).max(0.0) } else { 0.0 }
}

/// Per-point wood mask for one tree's points.
pub fn wood_mask(points: &[Point], p: &WoodParams) -> Vec<bool> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let (pl, li) = planarity_linearity(points, p.k);
    let mut like: Vec<f64> = pl.iter().zip(&li).map(|(a, b)| a + b).collect();
    // A single leaf is as planar as bark over a few centimetres. Over a
    // neighbourhood wider than a leaf, foliage is a jumble of orientations
    // while a trunk stays planar and a branch linear: keep the lower of the
    // two scores.
    if p.scale_radius > 0.0 {
        let tree = KdTree::new(points);
        let wide: Vec<f64> = points.par_iter().map(|q| anisotropy(points, &tree.within(q, p.scale_radius))).collect();
        for (l, w) in like.iter_mut().zip(wide) {
            *l = l.min(w);
        }
    }
    let mut wood = vec![false; n];

    // 1. Passage: shortest paths from the base; a point's passage count is
    // the number of target cells whose path runs through it.
    if p.passage && n > p.graph_k {
        let graph = knn_graph(points, p.graph_k, p.max_edge);
        // Seed from the lowest zone of the main component: an isolated clump
        // of ground remnants below the trunk would otherwise start the paths
        // and nothing above it would count.
        let (_, comp) = connected_components(&graph);
        let main = super::model::largest_component(&comp);
        let z0 = points.iter().zip(&comp).filter(|(_, &c)| c == main).map(|(q, _)| q[2]).fold(f64::INFINITY, f64::min);
        let sources: Vec<usize> = (0..n).filter(|&i| comp[i] == main && points[i][2] <= z0 + p.base_height).collect();
        let (dist, _, pred) = dijkstra(&graph, &sources);
        let targets: HashSet<usize> = voxel_downsample_indices(points, p.target_res).into_iter().collect();
        let mut order: Vec<usize> = (0..n).filter(|&i| dist[i].is_finite()).collect();
        order.sort_by(|&a, &b| dist[b].partial_cmp(&dist[a]).unwrap());
        let mut passage = vec![0usize; n];
        for &i in &order {
            if targets.contains(&i) {
                passage[i] += 1;
            }
            if pred[i] != usize::MAX {
                passage[pred[i]] += passage[i];
            }
        }
        // Neighbours of a passage point are wood too. A path runs along one
        // side of a stem, so on a thin or rough trunk a fixed reach keeps
        // little more than the points the paths step on. `assign_scale` lets
        // the reach grow with the share of the tree a point carries (by the
        // pipe model a stem's cross-section scales with what it supports):
        // at 0.03 the trunk reaches 3 % of the tree height across and a twig
        // barely beyond `assign_dist`. It is off by default: on the harvest
        // benchmark it improved bias and DBH but widened the volume scatter.
        let z1 = points.iter().zip(&comp).filter(|(_, &c)| c == main).map(|(q, _)| q[2]).fold(f64::NEG_INFINITY, f64::max);
        let height = (z1 - z0).max(0.0);
        let n_targets = targets.len().max(1) as f64;
        let tree = KdTree::new(points);
        let mut seed = vec![false; n];
        for i in 0..n {
            if passage[i] >= p.min_passage {
                seed[i] = true;
                let share = (passage[i] as f64 / n_targets).min(1.0);
                let reach = p.assign_dist + p.assign_scale * height * share.sqrt();
                for (j, _) in tree.within(&points[i], reach) {
                    seed[j] = true;
                }
            }
        }
        wood = seed;
    }

    // 2. High likelihood: connected components (26-connectivity at
    // `component_res`) that are large or contain passage wood.
    let high: Vec<usize> = (0..n).filter(|&i| wood[i] || like[i] > p.high_threshold).collect();
    wood = keep_components(points, &high, &wood, p.component_res, p.component_min, n);

    // 3. Medium likelihood, only in components that already hold wood.
    if p.medium_threshold < p.high_threshold {
        let mut medium: Vec<usize> = (0..n).filter(|&i| wood[i] || like[i] > p.medium_threshold).collect();
        if p.sor_k > 0 {
            let sub: Vec<Point> = medium.iter().map(|&i| points[i]).collect();
            let keep = statistical_outlier_mask(&sub, p.sor_k, p.sor_std);
            medium = medium.iter().zip(&keep).filter(|(&i, &k)| k || wood[i]).map(|(&i, _)| i).collect();
        }
        wood = keep_components(points, &medium, &wood, p.component_res, usize::MAX, n);
    }

    // 4. Dilation for thin branches.
    if p.dilate_dist > 0.0 {
        let tree = KdTree::new(points);
        let mut out = wood.clone();
        for i in 0..n {
            if wood[i] {
                for (j, _) in tree.within(&points[i], p.dilate_dist) {
                    out[j] = true;
                }
            }
        }
        wood = out;
    }
    wood
}

/// Components of `subset` (indices into `points`) under `res` connectivity;
/// a component survives when it contains a point already in `wood` or has
/// at least `min_size` points. Returns the new mask.
fn keep_components(points: &[Point], subset: &[usize], wood: &[bool], res: f64, min_size: usize, n: usize) -> Vec<bool> {
    let mut out = vec![false; n];
    if subset.is_empty() {
        return out;
    }
    let sub: Vec<Point> = subset.iter().map(|&i| points[i]).collect();
    let graph = radius_graph(&sub, res * 3f64.sqrt());
    let (count, label) = connected_components(&graph);
    let mut size = vec![0usize; count];
    let mut has_wood = vec![false; count];
    for (k, &i) in subset.iter().enumerate() {
        size[label[k]] += 1;
        if wood[i] {
            has_wood[label[k]] = true;
        }
    }
    for (k, &i) in subset.iter().enumerate() {
        if has_wood[label[k]] || size[label[k]] >= min_size {
            out[i] = true;
        }
    }
    out
}

/// Graph-based leaf / wood separation after Tian & Li (2022, IEEE TGRS 60,
/// "GBSeparation").
///
/// Shortest paths from the base give every point a path length and an
/// incoming growth direction. Edges are cut where they are long for their
/// neighbourhood or join points growing in different directions, which
/// severs leaves from the branch they hang on before any shape is judged.
/// The cut graph is then split into shells of path length at several scales;
/// a connected piece of a shell is wood when it runs the whole shell along
/// its growth direction and is either cylindrical (a circle fits its
/// cross-section) or linear. A piece thicker than the wood below it on its
/// path is rejected. Wood then spreads down every path to the base, to
/// graph neighbours no further from the base, and to close neighbours.
///
/// The steps and thresholds follow the authors' Python implementation
/// (Tian & Li, GBSeparation, 2022, <https://doi.org/10.5281/zenodo.6837613>,
/// CC BY 4.0), rewritten in Rust with union-find in place of networkx.
#[derive(Debug, Clone)]
pub struct GbsParams {
    pub graph_k: usize,
    pub max_edge: f64,
    pub base_height: f64,
    /// Shell thicknesses in path length (m). Larger trees want larger ones
    /// (the authors use 0.5-3 m).
    pub intervals: Vec<f64>,
    /// Largest angle (rad) between the growth directions of two joined points.
    pub max_angle: f64,
    /// Share of variance along the growth direction for a linear piece.
    pub linearity: f64,
    /// Largest circle-fit RMSE relative to the radius for a cylindrical piece.
    pub circle_error: f64,
    /// Smallest piece considered (also at least 1 / 20 000 of the cloud).
    pub min_points: usize,
}

impl Default for GbsParams {
    fn default() -> Self {
        GbsParams { graph_k: 8, max_edge: 1.0, base_height: 0.25, intervals: vec![0.1, 0.2, 0.3, 0.5, 1.0], max_angle: std::f64::consts::FRAC_PI_4, linearity: 0.9, circle_error: 0.2, min_points: 10 }
    }
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Shape class of one piece: `Some(r)` cylindrical with radius `r`,
/// `Some(-linearity)` linear, `None` neither.
fn classify_piece(points: &[Point], members: &[usize], step: &[Point], interval: f64, p: &GbsParams) -> Option<f64> {
    use crate::transform::{cross, dot, norm, normalize, scale, sub};
    let mut axis = [0.0; 3];
    for &i in members {
        for a in 0..3 {
            axis[a] += step[i][a];
        }
    }
    if norm(&axis) < 1e-12 {
        return None;
    }
    let axis = normalize(&axis);
    let n = members.len() as f64;
    let mut c = [0.0; 3];
    for &i in members {
        for a in 0..3 {
            c[a] += points[i][a] / n;
        }
    }
    let helper = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = normalize(&cross(&axis, &helper));
    let v = cross(&axis, &u);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut s_ax, mut s_u, mut s_v) = (0.0, 0.0, 0.0);
    let mut xy: Vec<[f64; 2]> = Vec::with_capacity(members.len());
    for &i in members {
        let d = sub(&points[i], &c);
        let (t, a, b) = (dot(&d, &axis), dot(&d, &u), dot(&d, &v));
        lo = lo.min(t);
        hi = hi.max(t);
        s_ax += t * t;
        s_u += a * a;
        s_v += b * b;
        xy.push([a, b]);
    }
    // The piece must run the whole shell along its growth direction.
    let extent = hi - lo;
    if extent < 0.75 * interval || extent > 1.25 * interval {
        return None;
    }
    let total = s_ax + s_u + s_v;
    if total <= 0.0 {
        return None;
    }
    // Cylindrical: the cross-section has some spread and a circle fits it.
    let spread = s_u.min(s_v) / total;
    if spread > 0.01 {
        if let Some((cx, cy, r)) = crate::stems::fit_circle_refined(&xy) {
            if r > 0.0 {
                let rmse = (xy.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).powi(2)).sum::<f64>() / n).sqrt();
                if rmse / r < p.circle_error {
                    return Some(r);
                }
            }
        }
    }
    let linearity = s_ax / total;
    let _ = scale;
    (linearity > p.linearity).then_some(-linearity)
}

/// Per-point wood mask (`true` = wood) for one tree, graph-based.
pub fn gbs_mask(points: &[Point], p: &GbsParams) -> Vec<bool> {
    use crate::transform::{dot, norm, sub};
    let n = points.len();
    if n <= p.graph_k {
        return vec![false; n];
    }
    let graph = knn_graph(points, p.graph_k, p.max_edge);
    let (_, comp) = connected_components(&graph);
    let main = super::model::largest_component(&comp);
    let z0 = points.iter().zip(&comp).filter(|(_, &c)| c == main).map(|(q, _)| q[2]).fold(f64::INFINITY, f64::min);
    let sources: Vec<usize> = (0..n).filter(|&i| comp[i] == main && points[i][2] <= z0 + p.base_height).collect();
    let (dist, _, pred) = dijkstra(&graph, &sources);
    // Last step of each point's path: its length and direction of growth.
    let step: Vec<Point> = (0..n).map(|i| if pred[i] != usize::MAX { sub(&points[i], &points[pred[i]]) } else { [0.0, 0.0, 0.0] }).collect();
    let step_len: Vec<f64> = step.iter().map(norm).collect();
    let keep_edge = |i: usize, j: usize, w: f64| -> bool {
        if pred[i] == usize::MAX || pred[j] == usize::MAX {
            return true; // base points
        }
        if w > 2.0 * step_len[i].min(step_len[j]) {
            return false;
        }
        let c = (dot(&step[i], &step[j]) / (step_len[i] * step_len[j]).max(1e-18)).clamp(-1.0, 1.0);
        let a = c.acos();
        a.min(std::f64::consts::PI - a) <= p.max_angle
    };
    let min_piece = p.min_points.max(n / 20_000);
    let mut init = vec![false; n];
    for &interval in &p.intervals {
        let bin: Vec<i64> = dist.iter().map(|d| if d.is_finite() { (d / interval).floor() as i64 } else { -1 }).collect();
        let mut parent: Vec<usize> = (0..n).collect();
        for i in 0..n {
            if bin[i] < 0 {
                continue;
            }
            for (j, w) in graph.edges(i) {
                if j > i && bin[j] == bin[i] && keep_edge(i, j, w) {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    if a != b {
                        parent[a] = b;
                    }
                }
            }
        }
        let mut pieces: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n {
            if bin[i] >= 0 {
                let r = find(&mut parent, i);
                pieces.entry(r).or_default().push(i);
            }
        }
        let mut pieces: Vec<Vec<usize>> = pieces.into_values().filter(|m| m.len() >= min_piece).collect();
        // The taper check below visits the pieces in turn and reads what it
        // has already decided: take them in a fixed order, not the hash
        // map's, which changes from one process to the next.
        pieces.sort_unstable_by_key(|m| m[0]);
        let mut class: Vec<Option<f64>> = pieces.par_iter().map(|m| classify_piece(points, m, &step, interval, p)).collect();
        let mut piece_of = vec![usize::MAX; n];
        for (k, m) in pieces.iter().enumerate() {
            for &i in m {
                piece_of[i] = k;
            }
        }
        // Taper: a piece may not be thicker than the first wood piece below it.
        for _ in 0..3 {
            for k in 0..pieces.len() {
                let Some(c) = class[k] else { continue };
                let mut i = pieces[k][0];
                while pred[i] != usize::MAX {
                    i = pred[i];
                    let q = piece_of[i];
                    if q != usize::MAX && q != k {
                        if let Some(below) = class[q] {
                            if c > below {
                                class[k] = None;
                            }
                            break;
                        }
                    }
                }
            }
        }
        for (k, m) in pieces.iter().enumerate() {
            if class[k].is_some() {
                for &i in m {
                    init[i] = true;
                }
            }
        }
    }
    // Wood runs down every path to the base ...
    let mut wood = vec![false; n];
    for i in 0..n {
        if init[i] {
            let mut j = i;
            while !wood[j] {
                wood[j] = true;
                if pred[j] == usize::MAX {
                    break;
                }
                j = pred[j];
            }
        }
    }
    // ... spreads to graph neighbours no further from the base ...
    let mut frontier: Vec<usize> = (0..n).filter(|&i| wood[i]).collect();
    for _ in 0..100 {
        let mut next = Vec::new();
        for &i in &frontier {
            for (j, _) in graph.edges(i) {
                if !wood[j] && dist[j] <= dist[i] {
                    wood[j] = true;
                    next.push(j);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    // ... and to neighbours closer than twice the point's own path step.
    let seed: Vec<usize> = (0..n).filter(|&i| wood[i] && pred[i] != usize::MAX).collect();
    for i in seed {
        for (j, w) in graph.edges(i) {
            if !wood[j] && w < 2.0 * step_len[i] {
                wood[j] = true;
            }
        }
    }
    for &s in &sources {
        wood[s] = true;
    }
    wood
}
