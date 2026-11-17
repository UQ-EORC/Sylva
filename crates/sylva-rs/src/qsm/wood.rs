// Sylva: LiDAR processing for forest ecology and remote sensing research.
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
//! the crown runs through the trunk and the branches, so points that
//! shortest paths run on through are wood whatever their local shape (the
//! path-frequency cue of Vicari et al. 2019).
//!
//! [`wood_scores`] measures that as a length: over the tree of shortest
//! paths from the base, the *support* of a point is the length of path from
//! it to the farthest point whose path runs through it. Points that support
//! at least `min_support` are wood; the wood then grows over the graph
//! through points whose neighbourhood is planar or linear.
//!
//! [`gbs_mask`] is the graph-based separation of Tian and Li (2022).

use rayon::prelude::*;

use crate::cluster::{connected_components, dijkstra, directed_knn_graph_bridged, knn_graph};
use crate::filters::planarity_linearity;
use crate::util::spatial::KdTree;
use crate::Point;

#[derive(Debug, Clone)]
/// Settings of [`wood_mask`] and [`wood_scores`].
pub struct WoodParams {
    /// Neighbourhood for the anisotropy.
    pub k: usize,
    /// Planarity + linearity above this is wood-like: the wood the paths
    /// find grows through such points (without `paths`, they are the wood).
    pub threshold: f64,
    /// Radius of the second, wider anisotropy neighbourhood (m): larger than a
    /// leaf, smaller than the spacing of branches. 0 (the default) uses `k`
    /// neighbours only: the wider scale labels foliage better but removes
    /// wood a QSM needs, so `classify_leaf_wood` turns it on and the QSM
    /// input filter does not.
    pub scale_radius: f64,
    /// Neighbours of a point in the graph the paths run over.
    pub graph_k: usize,
    /// Longest graph edge (m); pieces of the cloud within this of each other
    /// are joined.
    pub max_edge: f64,
    /// Points within this height of the lowest point start the paths (m).
    pub base_height: f64,
    /// A point is wood when the paths through it run on at least this far
    /// beyond it (m).
    pub min_support: f64,
    /// Points within this fraction of a wood point's support are wood too
    /// (0, the default, for none): by the pipe model a stem is as thick as
    /// what it carries is long, so the reach is a metre on a trunk that
    /// supports 30 m at 0.03, and nothing on a twig.
    pub reach_scale: f64,
    /// Longest graph step the wood may grow across (m).
    pub grow_reach: f64,
    /// Graph steps taken beyond the grown wood whatever the anisotropy, for
    /// twigs too thin to have any (count).
    pub grow_hops: usize,
    /// false leaves the paths out: the wood is then the wood-like points alone.
    pub paths: bool,
}

impl Default for WoodParams {
    fn default() -> Self {
        WoodParams { k: 20, threshold: 0.85, scale_radius: 0.0, graph_k: 8, max_edge: 1.0, base_height: 0.25, min_support: 0.3, reach_scale: 0.0, grow_reach: 0.05, grow_hops: 1, paths: true }
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
    if hi > 0.0 { (1.0 - lo / hi).clamp(0.0, 1.0) } else { 0.0 }
}

/// Per-point wood mask for one tree's points.
pub fn wood_mask(points: &[Point], p: &WoodParams) -> Vec<bool> {
    wood_scores(points, p).mask
}

/// The wood filter's decision and the two per-point cues it is made from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WoodScores {
    /// The filter's wood mask ([`wood_mask`]).
    pub mask: Vec<bool>,
    /// Local anisotropy, planarity + linearity (`1 - l_min / l_max`), in
    /// `[0, 1]`; the lower of the two scales with `scale_radius`.
    pub anisotropy: Vec<f64>,
    /// Support of the point as a share of the longest path from the base,
    /// in `[0, 1]` (0 everywhere without the paths).
    pub support: Vec<f64>,
}

/// [`wood_mask`] with the anisotropy and the support of every point.
///
/// The wood is what the shortest paths from the base run on through (the
/// path-frequency cue of Vicari et al. 2019, measured as a length), grown
/// over the same graph through wood-like points.
pub fn wood_scores(points: &[Point], p: &WoodParams) -> WoodScores {
    let n = points.len();
    if n == 0 {
        return WoodScores::default();
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
    let mut share = vec![0.0; n];

    // 1. Support. Over the tree of shortest paths from the base, a point
    // carries the points whose paths run through it, and its support is the
    // length of path from it to the farthest of them. A leaf, or a bark
    // point where a path ends, supports nothing; a point on a branch
    // supports the branch beyond it. Being a length, the measure does not
    // depend on how densely the tree was sampled.
    if p.paths && n > p.graph_k {
        // The graph joins its pieces wherever two come within `max_edge` of
        // each other, so a trunk cut off from its crown by an occluded
        // stretch still carries the crown's paths. An edge costs its length.
        let graph = directed_knn_graph_bridged(points, p.graph_k, p.max_edge, 1.0, None);
        // Start from the lowest zone of the main component: an isolated clump
        // of ground remnants below the trunk would otherwise start the paths
        // and nothing above it would count.
        let (_, comp) = connected_components(&graph);
        let main = super::model::largest_component(&comp);
        let z0 = points.iter().zip(&comp).filter(|(_, &c)| c == main).map(|(q, _)| q[2]).fold(f64::INFINITY, f64::min);
        let sources: Vec<usize> = (0..n).filter(|&i| comp[i] == main && points[i][2] <= z0 + p.base_height).collect();
        let (dist, _, pred) = dijkstra(&graph, &sources);
        // Farthest first, so that every point is finished before the point
        // its path goes on to.
        let mut order: Vec<usize> = (0..n).filter(|&i| dist[i].is_finite()).collect();
        order.sort_by(|&a, &b| dist[b].partial_cmp(&dist[a]).unwrap());
        let mut far = dist.clone();
        for &i in &order {
            if pred[i] != usize::MAX {
                far[pred[i]] = far[pred[i]].max(far[i]);
            }
        }
        let longest = order.first().map(|&i| dist[i]).unwrap_or(0.0).max(f64::MIN_POSITIVE);
        let tree = (p.reach_scale > 0.0).then(|| KdTree::new(points));
        for &i in &order {
            let support = far[i] - dist[i];
            share[i] = (support / longest).clamp(0.0, 1.0);
            if support >= p.min_support {
                wood[i] = true;
                if let Some(tree) = &tree {
                    for (j, _) in tree.within(&points[i], p.reach_scale * support) {
                        wood[j] = true;
                    }
                }
            }
        }
        // 2. Growth: from the wood the paths found, over the same graph,
        // through points whose anisotropy is wood-like, a step of at most
        // `grow_reach` at a time; then `grow_hops` steps more whatever the
        // anisotropy, for twigs too thin to have any.
        let mut front: Vec<usize> = (0..n).filter(|&i| wood[i]).collect();
        while let Some(i) = front.pop() {
            for (j, w) in graph.edges(i) {
                if !wood[j] && w <= p.grow_reach && like[j] > p.threshold {
                    wood[j] = true;
                    front.push(j);
                }
            }
        }
        for _ in 0..p.grow_hops {
            let add: Vec<usize> = (0..n).filter(|&i| wood[i]).flat_map(|i| graph.edges(i).filter(|&(j, w)| !wood[j] && w <= p.grow_reach).map(|(j, _)| j).collect::<Vec<_>>()).collect();
            for j in add {
                wood[j] = true;
            }
        }
    } else {
        // Without the paths: the wood-like points themselves.
        for i in 0..n {
            wood[i] = like[i] > p.threshold;
        }
    }
    WoodScores { mask: wood, anisotropy: like, support: share }
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
    /// Neighbours per point in the kNN graph the paths run on.
    pub graph_k: usize,
    /// Longest graph edge (m).
    pub max_edge: f64,
    /// Points within this height (m) of the lowest point of the main component seed the paths.
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
        if let Some((cx, cy, r)) = crate::trees::stems::fit_circle_refined(&xy) {
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
    gbs_scores(points, p).0
}

/// [`gbs_mask`] and, per point, the share of the shell scales (`intervals`)
/// at which the point's piece was classified wood before the labels were
/// spread along the paths: 1 for a piece that is cylindrical or linear at
/// every scale, 0 for a point that is wood only by being on a path or next
/// to wood.
pub fn gbs_scores(points: &[Point], p: &GbsParams) -> (Vec<bool>, Vec<f64>) {
    use crate::transform::{dot, norm, sub};
    let n = points.len();
    if n <= p.graph_k {
        return (vec![false; n], vec![0.0; n]);
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
    let mut votes = vec![0u32; n];
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
                    votes[i] += 1;
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
    let scales = p.intervals.len().max(1) as f64;
    (wood, votes.into_iter().map(|v| v as f64 / scales).collect())
}
