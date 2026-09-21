//! Leaf / wood separation for a single tree.
//!
//! Local anisotropy alone (planarity + linearity over `k` neighbours) drops
//! sparsely scanned trunks: an occluded stem seen as a few scattered patches
//! is neither planar nor linear at the neighbourhood scale, and without it
//! the QSM has no measurements on the very cylinders that carry the volume.
//! The answer here is topological: every path from the base to any part of
//! the crown runs through the trunk and the branches, so points that many
//! shortest paths pass through are wood whatever their local shape.
//!
//! Steps: passage counting on a kNN graph from
//! the base, with coarse target cells so the count does not depend on point
//! density; neighbours of passage wood; high-likelihood anisotropy in
//! connected components that are large or touch passage wood; medium
//! likelihood only where it connects to wood already found; a final
//! dilation for thin branches.

use std::collections::HashSet;

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
        WoodParams { k: 20, high_threshold: 0.85, medium_threshold: 0.75, graph_k: 10, max_edge: 1.0, base_height: 0.25, target_res: 0.2, min_passage: 3, assign_dist: 0.05, assign_scale: 0.0, component_res: 0.05, component_min: 200, sor_k: 50, sor_std: 1.0, dilate_dist: 0.03, passage: true }
    }
}

/// Per-point wood mask for one tree's points.
pub fn wood_mask(points: &[Point], p: &WoodParams) -> Vec<bool> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let (pl, li) = planarity_linearity(points, p.k);
    let like: Vec<f64> = pl.iter().zip(&li).map(|(a, b)| a + b).collect();
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
