// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Tree-level processing: stem detection, DBH, segmentation, heights, crowns.

pub mod prune;
pub mod stems;

use std::collections::HashMap;

use rayon::prelude::*;

use crate::cluster::{dijkstra_gravity, dijkstra_scaled, directed_knn_graph, directed_knn_graph_bridged, directed_knn_graph_wood};
use crate::error::{Error, Result};
use crate::filters::Rng;
use crate::util::spatial::first_in_voxel;
use crate::util::optim::levenberg_marquardt;
use crate::util::spatial::KdTree;
use crate::Point;

/// Summary of a detected tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub tree_id: i64,
    pub x: f64,
    pub y: f64,
    pub dbh: f64,
    pub height: f64,
    pub n_points: usize,
    /// Fraction of the circumference observed (angular coverage), 0..1.
    pub inlier_fraction: f64,
    pub n_slices: usize,
    pub rmse: f64,
    pub lean_deg: f64,
    /// Heuristic 0..1 confidence from fit residual, coverage and slice support.
    pub quality: f64,
}

// ------------------------------------------------------------------ circles

/// Kåsa (1976) algebraic circle fit: `(cx, cy, r)`.
pub fn fit_circle_algebraic(xy: &[[f64; 2]]) -> Result<(f64, f64, f64)> {
    if xy.len() < 3 {
        return Err(Error::invalid("need >= 3 points"));
    }
    // Normal equations for [x y 1] [a b c]^T = x^2 + y^2.
    let mut ata = nalgebra::Matrix3::<f64>::zeros();
    let mut atb = nalgebra::Vector3::<f64>::zeros();
    for p in xy {
        let row = nalgebra::Vector3::new(p[0], p[1], 1.0);
        ata += row * row.transpose();
        atb += row * (p[0] * p[0] + p[1] * p[1]);
    }
    let s = ata.lu().solve(&atb).ok_or_else(|| Error::invalid("degenerate circle fit"))?;
    let cx = s[0] / 2.0;
    let cy = s[1] / 2.0;
    let r = (s[2] + cx * cx + cy * cy).max(0.0).sqrt();
    Ok((cx, cy, r))
}

/// Geometric (LM) circle fit initialised algebraically: `(cx, cy, r, rmse)`.
pub fn fit_circle(xy: &[[f64; 2]]) -> Result<(f64, f64, f64, f64)> {
    let (cx, cy, r) = fit_circle_algebraic(xy)?;
    let res = levenberg_marquardt(
        |p, out| {
            out.clear();
            out.extend(xy.iter().map(|q| ((q[0] - p[0]).hypot(q[1] - p[1])) - p[2]));
        },
        &[cx, cy, r],
        50,
        1e-10,
    );
    Ok((res.x[0], res.x[1], res.x[2].abs(), res.rmse))
}

fn circle_through_3(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<(f64, f64, f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-12 {
        return None;
    }
    let (a2, b2, c2) = (a[0] * a[0] + a[1] * a[1], b[0] * b[0] + b[1] * b[1], c[0] * c[0] + c[1] * c[1]);
    let ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
    let uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
    Some((ux, uy, (a[0] - ux).hypot(a[1] - uy)))
}

#[derive(Debug, Clone)]
pub struct RansacCircleParams {
    pub threshold: f64,
    pub iterations: usize,
    pub min_radius: f64,
    pub max_radius: f64,
    pub seed: u64,
}

impl Default for RansacCircleParams {
    fn default() -> Self {
        RansacCircleParams { threshold: 0.01, iterations: 200, min_radius: 0.02, max_radius: 1.5, seed: 0 }
    }
}

/// RANSAC circle fit (Fischler & Bolles 1981): `(cx, cy, r, inlier mask)`,
/// refit on the inliers.
pub fn fit_circle_ransac(xy: &[[f64; 2]], p: &RansacCircleParams) -> Result<(f64, f64, f64, Vec<bool>)> {
    let n = xy.len();
    if n < 3 {
        return Err(Error::invalid("need >= 3 points"));
    }
    let mut rng = Rng::new(p.seed);
    let mut best: Vec<bool> = Vec::new();
    let mut best_n = 0usize;
    for _ in 0..p.iterations {
        let (i, j, k) = (rng.below(n), rng.below(n), rng.below(n));
        if i == j || j == k || i == k {
            continue;
        }
        let Some((cx, cy, r)) = circle_through_3(xy[i], xy[j], xy[k]) else { continue };
        if r < p.min_radius || r > p.max_radius {
            continue;
        }
        let cnt = xy.iter().filter(|q| (((q[0] - cx).hypot(q[1] - cy)) - r).abs() < p.threshold).count();
        if cnt > best_n {
            best_n = cnt;
            best = xy.iter().map(|q| (((q[0] - cx).hypot(q[1] - cy)) - r).abs() < p.threshold).collect();
        }
    }
    if best_n < 3 {
        return Err(Error::invalid("RANSAC found no valid circle"));
    }
    let inl: Vec<[f64; 2]> = xy.iter().zip(&best).filter(|(_, &b)| b).map(|(q, _)| *q).collect();
    let (cx, cy, r, _) = fit_circle(&inl)?;
    Ok((cx, cy, r, best))
}

// ---------------------------------------------------------------- detection

pub use crate::trees::stems::{detect_stems, StemParams};

/// Stem diameter at each of `at_heights` (taper). NaN where no fit is possible.
pub fn dbh_profile(points: &[Point], heights: &[f64], cx: f64, cy: f64, at_heights: &[f64], slice_thickness: f64, search_radius: f64) -> Vec<f64> {
    let ransac = RansacCircleParams::default();
    at_heights
        .iter()
        .map(|&hh| {
            let xy: Vec<[f64; 2]> = (0..points.len())
                .filter(|&i| (heights[i] - hh).abs() <= slice_thickness / 2.0 && (points[i][0] - cx).hypot(points[i][1] - cy) <= search_radius)
                .map(|i| [points[i][0], points[i][1]])
                .collect();
            if xy.len() < 10 {
                return f64::NAN;
            }
            fit_circle_ransac(&xy, &ransac).map(|(_, _, r, _)| 2.0 * r).unwrap_or(f64::NAN)
        })
        .collect()
}

// ------------------------------------------------------------- segmentation

#[derive(Debug, Clone)]
pub struct SegmentParams {
    pub k: usize,
    /// Neighbours farther than this are not linked.
    pub max_edge: f64,
    /// Build the graph on a voxel downsample of this size (0 = full cloud).
    pub voxel_size: f64,
    pub seed_height: f64,
    pub seed_radius: f64,
    /// Edge cost exponent on distance.
    pub power: f64,
    /// Penalise edges by their angle from vertical (up free, down x100).
    pub angle_penalty: bool,
    /// raycloudtools gravity factor: cost x (1 + g * lateral^2 from the seed); 0 disables.
    pub gravity: f64,
    /// Points below this height are left unassigned (-1) and kept out of
    /// the graph: the terrain model's own noise band, 0.1 m, so that the
    /// ground surface is not in the graph while a stem is labelled to within
    /// a hand of the ground. With the ground given as a mask instead
    /// (`segment_trees_masked`) it can be 0.
    pub cut_above_ground: f64,
    /// Scale each tree's path costs by `1 / max(height_estimate, 2)`, the
    /// height estimate being the 95th-percentile point height within
    /// `height_prior_radius` of the stem (raycloudtools' per-root scaling), so
    /// tall trees win contested crown points over understorey stems.
    pub height_prior: bool,
    pub height_prior_radius: f64,
    /// Seed a ring around the stem surface rather than a disc around its
    /// axis, so a trunk wider than `seed_radius` still gets seeds.
    pub seed_ring: bool,
    /// How hard the height prior leans: cost is divided by `h^this`. 1 gives
    /// a tall tree its full advantage, which in a dense stand hands it its
    /// suppressed neighbour's crown as well; 0 is no prior at all.
    pub height_prior_power: f64,
    /// Points below this height stay labelled only within `low_radius`
    /// (or 1.5 DBH) of their tree's base: the ground remnants, litter and
    /// understorey the graph reaches along the surface are left unassigned
    /// instead of being modelled as part of the tree.
    pub low_height: f64,
    pub low_radius: f64,
    /// Classify graph nodes as wood by local anisotropy (`1 - l_min / l_max`
    /// over `wood_k` neighbours) above `wood_threshold`, and
    /// apply wood / leaf cost factors so foliage cannot bridge trees.
    pub wood_costs: bool,
    pub wood_k: usize,
    pub wood_threshold: f64,
    /// Understorey competes for points (after raycloudtools, where every
    /// point's path runs to the ground and small plants keep their own).
    /// Graph nodes up to `understorey_band` above `cut_above_ground`, and
    /// further than `max(low_radius, 1.5 DBH)` from every stem, become extra
    /// sources whose path costs are scaled as for a tree this tall (with
    /// `height_prior`); what they claim is left unassigned. 0, the default,
    /// disables: on the four CHERLET reference plots the sources gained no
    /// tree and lost up to four, and what they take from a crown they take
    /// through foliage regardless of how they are costed.
    pub understorey_height: f64,
    pub understorey_band: f64,
    /// Corner of the voxel grid of the graph; None anchors it at the
    /// cloud's minimum corner rounded down to whole metres, so that nothing
    /// about the cut or the thinning moves it. A fixed corner puts every
    /// part of a plot on one grid, so tiles of the plot get the graph nodes
    /// the whole plot has (`crate::als::tiles_trees`).
    pub voxel_origin: Option<Point>,
    /// Join the graph's components wherever two come within `max_edge` of
    /// each other, so that a crown reaches its trunk through sparsely
    /// sampled branches whatever points the thinning kept
    /// (`crate::cluster::directed_knn_graph_bridged`).
    pub bridge: bool,
}

impl Default for SegmentParams {
    fn default() -> Self {
        SegmentParams { k: 6, max_edge: 1.0, voxel_size: 0.03, seed_height: 1.5, seed_radius: 0.25, power: 6.0, angle_penalty: true, gravity: 0.0, cut_above_ground: 0.1, seed_ring: true, height_prior: true, height_prior_radius: 1.5, height_prior_power: 1.0, low_height: 0.5, low_radius: 1.0, wood_costs: false, wood_k: 20, wood_threshold: 0.9, understorey_height: 0.0, understorey_band: 0.5, voxel_origin: None, bridge: true }
    }
}

/// The points that become graph nodes: those at or above
/// `cut_above_ground`, thinned to the first of each `voxel_size` voxel.
pub fn graph_nodes(points: &[Point], heights: &[f64], p: &SegmentParams) -> Vec<usize> {
    graph_nodes_masked(points, heights, p, None)
}

/// [`graph_nodes`] leaving out the points flagged in `exclude` as well: the
/// ground as a classifier found it, so that the surface is out of the graph
/// whatever its height, and the cut can sit at the ground itself.
pub fn graph_nodes_masked(points: &[Point], heights: &[f64], p: &SegmentParams, exclude: Option<&[bool]>) -> Vec<usize> {
    let above: Vec<usize> = (0..points.len()).filter(|&i| heights[i] >= p.cut_above_ground && exclude.is_none_or(|e| !e[i])).collect();
    if p.voxel_size.is_nan() || p.voxel_size <= 0.0 {
        return above;
    }
    // Thinned in place: the voxel of each point is found through its index,
    // with no copy of the points above the cut and no list per voxel.
    let origin = p.voxel_origin.unwrap_or_else(|| {
        let lo = crate::util::spatial::min_corner(points);
        [lo[0].floor(), lo[1].floor(), lo[2].floor()]
    });
    first_in_voxel(points, above, &origin, p.voxel_size)
}

/// Assign points to the nearest stem by shortest path through a kNN graph
/// (multi-source Dijkstra from stem seeds), after raycloudtools' `rayextract
/// trees` (Devereux et al. 2026). Unreachable points get `-1`.
pub fn segment_trees(points: &[Point], heights: &[f64], trees: &[Tree], p: &SegmentParams) -> Vec<i64> {
    segment_trees_masked(points, heights, trees, p, None)
}

/// [`segment_trees`] with the points flagged in `exclude` kept out of the
/// graph and left unassigned (-1): pass the ground classification, and set
/// `cut_above_ground` to 0, to label stems down to the last point above the
/// ground rather than to a fixed height above it.
pub fn segment_trees_masked(points: &[Point], heights: &[f64], trees: &[Tree], p: &SegmentParams, exclude: Option<&[bool]>) -> Vec<i64> {
    let task = crate::util::progress::start("segmenting trees", 3);
    let work_idx = graph_nodes_masked(points, heights, p, exclude);
    let work: Vec<Point> = work_idx.iter().map(|&i| points[i]).collect();
    let hw: Vec<f64> = work_idx.iter().map(|&i| heights[i]).collect();
    task.inc(1); // thinned
    let labels_work = segment_nodes(&work, &hw, trees, p);
    task.inc(1); // paths
    let labels = label_points_masked(points, heights, &work, &labels_work, trees, p, exclude);
    task.inc(1); // labelled
    labels
}

/// The sources of [`segment_nodes`]' search that come from the stems: node
/// indices, each with the id of its tree, the stem position and the cost
/// scale of the tree's height prior.
pub struct Seeds {
    pub seeds: Vec<usize>,
    pub seed_tree: Vec<i64>,
    pub seed_xy: Vec<[f64; 2]>,
    pub seed_scale: Vec<f64>,
}

/// Nodes in a flat 2-D grid of square cells, so that the nodes within a
/// radius of a point are found among the cells the radius reaches instead
/// of by a pass over them all. `order` holds the node indices sorted by
/// cell, `starts[k]..starts[k + 1]` the run of cell `cells[k]`.
struct FlatGrid {
    cell: f64,
    cells: Vec<[i64; 2]>,
    starts: Vec<usize>,
    order: Vec<u32>,
}

impl FlatGrid {
    fn key(x: f64, y: f64, cell: f64) -> [i64; 2] {
        [(x / cell).floor() as i64, (y / cell).floor() as i64]
    }

    fn new(points: &[Point], cell: f64) -> FlatGrid {
        let mut keyed: Vec<([i64; 2], u32)> = points.iter().enumerate().map(|(i, q)| (Self::key(q[0], q[1], cell), i as u32)).collect();
        keyed.par_sort_unstable();
        let mut cells = Vec::new();
        let mut starts = Vec::new();
        let mut order = Vec::with_capacity(keyed.len());
        for (k, i) in keyed {
            if cells.last() != Some(&k) {
                cells.push(k);
                starts.push(order.len());
            }
            order.push(i);
        }
        starts.push(order.len());
        FlatGrid { cell, cells, starts, order }
    }

    /// Every node that could lie within `r` of `(x, y)`: the nodes of the
    /// cells the disc touches, in no particular order. The caller applies
    /// the exact test. The range is padded a little so that rounding in
    /// `x - r` cannot leave out a node exactly `r` away.
    fn candidates(&self, x: f64, y: f64, r: f64) -> impl Iterator<Item = usize> + '_ {
        let pad = r * 1e-9 + 1e-9;
        let lo = Self::key(x - r - pad, y - r - pad, self.cell);
        let hi = Self::key(x + r + pad, y + r + pad, self.cell);
        (lo[0]..=hi[0])
            .flat_map(move |cx| (lo[1]..=hi[1]).map(move |cy| [cx, cy]))
            .filter_map(move |k| self.cells.binary_search(&k).ok())
            .flat_map(move |slot| self.order[self.starts[slot]..self.starts[slot + 1]].iter().map(|&i| i as usize))
    }
}

/// The seeds of each stem among the graph nodes (`work`, heights `hw`), in
/// the order the search takes them: tree by tree, each tree's seed nodes by
/// index. With `height_prior`, a tree's cost scale is `1 / h^power` for `h`
/// the 95th-percentile node height within `height_prior_radius` of its stem
/// (at least 2 m). The seed nodes are those on the bark ring: distance from
/// the stem axis within `seed_radius` of half the DBH (of the axis itself
/// without `seed_ring`), at any height below `seed_height + 0.5`.
pub fn collect_seeds(work: &[Point], hw: &[f64], trees: &[Tree], p: &SegmentParams) -> Seeds {
    let mut out = Seeds { seeds: Vec::new(), seed_tree: Vec::new(), seed_xy: Vec::new(), seed_scale: Vec::new() };
    if trees.is_empty() || work.is_empty() {
        return out;
    }
    // One grid serves both queries: cells as wide as the widest, so a disc
    // of either radius lies within the three-by-three cells about its centre.
    let ring = trees.iter().map(|t| if p.seed_ring { 0.5 * t.dbh } else { 0.0 } + p.seed_radius).fold(0.0, f64::max);
    let widest = if p.height_prior { ring.max(p.height_prior_radius) } else { ring };
    let grid = FlatGrid::new(work, widest.max(1e-3));
    let mut found: Vec<usize> = Vec::new();
    for t in trees {
        let scale = if p.height_prior {
            let mut hs: Vec<f64> = grid.candidates(t.x, t.y, p.height_prior_radius).filter(|&i| (work[i][0] - t.x).hypot(work[i][1] - t.y) <= p.height_prior_radius).map(|i| hw[i]).collect();
            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let h95 = hs.get(((hs.len() as f64 * 0.95) as usize).min(hs.len().saturating_sub(1))).copied().unwrap_or(2.0);
            1.0 / h95.max(2.0).powf(p.height_prior_power)
        } else {
            1.0
        };
        // Seed the bark, not the axis: on a stem wider than `seed_radius` a
        // disc around the centre lies inside the trunk, where a scan has no
        // points at all, and the tree starts with nothing (r-lidar's arbor seeds
        // a whole synthetic trunk shell for the same reason).
        let inner = if p.seed_ring { 0.5 * t.dbh } else { 0.0 };
        // The whole shell from the ground (every node is above the cut) up
        // to half a metre above `seed_height`: the trunk belongs to its tree
        // at no cost, rather than being grown into from breast height against
        // the angle penalty.
        found.clear();
        found.extend(grid.candidates(t.x, t.y, inner + p.seed_radius).filter(|&i| {
            let d = (work[i][0] - t.x).hypot(work[i][1] - t.y);
            (d - inner).abs() <= p.seed_radius && hw[i] < p.seed_height + 0.5
        }));
        found.sort_unstable();
        for &i in &found {
            out.seeds.push(i);
            out.seed_tree.push(t.tree_id);
            out.seed_xy.push([t.x, t.y]);
            out.seed_scale.push(scale);
        }
    }
    out
}

/// The graph part of [`segment_trees`] on its nodes (`work`, the points
/// [`graph_nodes`] keeps, with heights `hw`): the tree id whose seeds reach
/// each node most cheaply, -1 for nodes no seed reaches and for those the
/// understorey claims. Nothing is thinned here.
pub fn segment_nodes(work: &[Point], hw: &[f64], trees: &[Tree], p: &SegmentParams) -> Vec<i64> {
    let wood: Option<Vec<bool>> = if p.wood_costs {
        let (_, vals) = crate::filters::local_pca(work, p.wood_k);
        Some(vals.iter().map(|[l1, _, l3]| *l3 > 1e-14 && (l3 - l1) / l3 > p.wood_threshold).collect())
    } else {
        None
    };
    let graph = if p.bridge { directed_knn_graph_bridged(work, p.k, p.max_edge, p.power, p.angle_penalty, wood.as_deref()) } else { directed_knn_graph_wood(work, p.k, p.max_edge, p.power, p.angle_penalty, wood.as_deref()) };
    let Seeds { mut seeds, mut seed_tree, mut seed_xy, mut seed_scale } = collect_seeds(work, hw, trees, p);
    if seeds.is_empty() {
        return vec![-1; work.len()];
    }
    if p.understorey_height > 0.0 {
        // Near-ground nodes away from every stem: sources labelled -1.
        let stems = KdTree::new(&trees.iter().map(|t| [t.x, t.y, 0.0]).collect::<Vec<_>>());
        let clear: Vec<f64> = trees.iter().map(|t| p.low_radius.max(1.5 * t.dbh)).collect();
        let reach = clear.iter().cloned().fold(0.0, f64::max);
        let scale = if p.height_prior { 1.0 / p.understorey_height } else { 1.0 };
        for (i, q) in work.iter().enumerate() {
            if hw[i] > p.cut_above_ground + p.understorey_band {
                continue;
            }
            let flat = [q[0], q[1], 0.0];
            let near = stems.within(&flat, reach).into_iter().any(|(j, d)| d <= clear[j]);
            if !near {
                seeds.push(i);
                seed_tree.push(-1);
                seed_xy.push([q[0], q[1]]);
                seed_scale.push(scale);
            }
        }
    }
    let (dist, src, _) = dijkstra_scaled(&graph, &seeds, Some(&seed_xy), p.gravity, if p.height_prior { Some(&seed_scale) } else { None });
    (0..work.len()).map(|i| if dist[i].is_finite() { seed_tree[src[i]] } else { -1 }).collect()
}

/// The last step of [`segment_trees`]: each point takes the label of its
/// nearest graph node (`work`, labelled by [`segment_nodes`]); points below
/// `cut_above_ground`, and those below `low_height` farther than
/// `max(low_radius, 1.5 DBH)` from their tree's stem, get -1.
pub fn label_points(points: &[Point], heights: &[f64], work: &[Point], labels_work: &[i64], trees: &[Tree], p: &SegmentParams) -> Vec<i64> {
    label_points_masked(points, heights, work, labels_work, trees, p, None)
}

/// [`label_points`] with the points flagged in `exclude` left at -1.
pub fn label_points_masked(points: &[Point], heights: &[f64], work: &[Point], labels_work: &[i64], trees: &[Tree], p: &SegmentParams, exclude: Option<&[bool]>) -> Vec<i64> {
    let tree = KdTree::new(work);
    let base: std::collections::HashMap<i64, (f64, f64, f64)> = trees.iter().map(|t| (t.tree_id, (t.x, t.y, p.low_radius.max(1.5 * t.dbh)))).collect();
    (0..points.len())
        .into_par_iter()
        .map(|i| {
            let (q, h) = (&points[i], heights[i]);
            if h < p.cut_above_ground || exclude.is_some_and(|e| e[i]) {
                return -1;
            }
            let l = tree.nearest(q).map(|(j, _)| labels_work[j]).unwrap_or(-1);
            if l >= 0 && h < p.low_height {
                if let Some(&(x, y, r)) = base.get(&l) {
                    if (q[0] - x).hypot(q[1] - y) > r {
                        return -1;
                    }
                }
            }
            l
        })
        .collect()
}

/// Drop stem candidates that are really branches or secondary stems of
/// another candidate.
///
/// After raycloudtools (Devereux et al. 2026), every point's least-cost path
/// to the ground is found
/// (multi-source Dijkstra from all points below `ground_height` on the
/// upward-cheap directed graph). A candidate whose seed's path to the ground
/// passes through another candidate's trunk region (within
/// `max(trunk_scale * radius, trunk_min)` of that axis, below that
/// candidate's seed height) is a branch of it and is removed. Returns the
/// surviving trees and, for every input tree, the id it was merged into.
pub fn merge_branches(points: &[Point], heights: &[f64], trees: &[Tree], p: &SegmentParams, ground_height: f64, trunk_scale: f64, trunk_min: f64, search_radius: f64) -> (Vec<Tree>, Vec<i64>) {
    let parent = branch_parents(points, heights, trees, p, ground_height, trunk_scale, trunk_min, search_radius, None);
    resolve_branches(trees, &parent)
}

/// The first half of [`merge_branches`]: for each candidate, the candidate
/// whose trunk its seed's path to the ground runs through (the index into
/// `trees`), if any. With `only`, candidates not flagged in it are not
/// traced (their entry is None) but still count as trunks for the others.
#[allow(clippy::too_many_arguments)]
pub fn branch_parents(points: &[Point], heights: &[f64], trees: &[Tree], p: &SegmentParams, ground_height: f64, trunk_scale: f64, trunk_min: f64, search_radius: f64, only: Option<&[bool]>) -> Vec<Option<usize>> {
    let mut parent: Vec<Option<usize>> = vec![None; trees.len()];
    let work_idx = graph_nodes(points, heights, p);
    let work: Vec<Point> = work_idx.iter().map(|&i| points[i]).collect();
    let hw: Vec<f64> = work_idx.iter().map(|&i| heights[i]).collect();
    let graph = if p.bridge { directed_knn_graph_bridged(&work, p.k, p.max_edge, p.power, p.angle_penalty, None) } else { directed_knn_graph(&work, p.k, p.max_edge, p.power, p.angle_penalty) };
    let sources: Vec<usize> = (0..work.len()).filter(|&i| hw[i] < ground_height).collect();
    if sources.is_empty() {
        return parent;
    }
    let (dist, _, pred) = dijkstra_gravity(&graph, &sources, None, 0.0);
    let traced = |ci: usize| only.is_none_or(|o| o[ci]);
    // Seed node per candidate: nearest graph node to (x, y, seed_height) that is connected to ground.
    let seed_node: Vec<Option<usize>> = trees
        .iter()
        .enumerate()
        .map(|(ci, t)| {
            if !traced(ci) {
                return None;
            }
            let mut best: Option<(usize, f64)> = None;
            for i in 0..work.len() {
                if (work[i][0] - t.x).hypot(work[i][1] - t.y) > p.seed_radius || (hw[i] - p.seed_height).abs() > 0.5 || !dist[i].is_finite() {
                    continue;
                }
                if best.map(|b| dist[i] < b.1).unwrap_or(true) {
                    best = Some((i, dist[i]));
                }
            }
            best.map(|b| b.0)
        })
        .collect();
    for (ci, t) in trees.iter().enumerate() {
        let Some(mut node) = seed_node[ci] else { continue };
        // Walk the predecessor chain to the ground.
        let mut guard = 0;
        while pred[node] != usize::MAX && guard < 1_000_000 {
            node = pred[node];
            guard += 1;
            let q = work[node];
            let h = hw[node];
            for (oi, o) in trees.iter().enumerate() {
                if oi == ci || (o.x - t.x).hypot(o.y - t.y) > search_radius {
                    continue;
                }
                let r = (trunk_scale * o.dbh / 2.0).max(trunk_min);
                if h >= p.cut_above_ground && h <= p.seed_height + 0.5 && (q[0] - o.x).hypot(q[1] - o.y) <= r {
                    parent[ci] = Some(oi);
                }
            }
            if parent[ci].is_some() {
                break;
            }
        }
    }
    parent
}

/// The second half of [`merge_branches`]: follow the parents of
/// [`branch_parents`] to the candidate each one belongs to, and return the
/// survivors and, for every candidate, the id it now belongs to.
pub fn resolve_branches(trees: &[Tree], parent: &[Option<usize>]) -> (Vec<Tree>, Vec<i64>) {
    // Resolve chains (a -> b -> c) and mutual pairs (keep the higher quality one).
    let mut target: Vec<usize> = (0..trees.len()).collect();
    for ci in 0..trees.len() {
        let mut cur = ci;
        let mut seen = 0;
        while let Some(pi) = parent[cur] {
            if pi == ci || seen > trees.len() {
                // cycle: keep the better candidate (`cur` is the member that
                // points back at `ci`); ties go to the lower index so both
                // ends of a mutual pair agree.
                let (qa, qb) = (trees[ci].quality, trees[cur].quality);
                cur = if qa > qb || (qa == qb && ci <= cur) { ci } else { cur };
                break;
            }
            cur = pi;
            seen += 1;
        }
        target[ci] = cur;
    }
    let survivors: Vec<Tree> = trees.iter().enumerate().filter(|(i, _)| target[*i] == *i).map(|(_, t)| t.clone()).collect();
    let merged_into: Vec<i64> = target.iter().map(|&ti| trees[ti].tree_id).collect();
    (survivors, merged_into)
}

/// Fill `height` and `n_points` for each tree from segmentation labels.
pub fn tree_heights(heights: &[f64], labels: &[i64], trees: &mut [Tree], percentile: f64) {
    // One pass over the labels, bucketing each tree's heights, rather than
    // a pass per tree. A tree id that several entries share takes the first.
    let mut slot: HashMap<i64, usize> = HashMap::new();
    for (k, t) in trees.iter().enumerate() {
        slot.entry(t.tree_id).or_insert(k);
    }
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); trees.len()];
    for (&l, &h) in labels.iter().zip(heights) {
        if let Some(&k) = slot.get(&l) {
            buckets[k].push(h);
        }
    }
    for (t, k) in trees.iter_mut().zip(0..) {
        let mut hs = std::mem::take(&mut buckets[slot[&t.tree_id]]);
        if slot[&t.tree_id] != k {
            // A later duplicate of an id: the same points as the first.
            hs = labels.iter().zip(heights).filter(|(&l, _)| l == t.tree_id).map(|(_, &h)| h).collect();
        }
        t.n_points = hs.len();
        if hs.is_empty() {
            t.height = f64::NAN;
            continue;
        }
        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = ((percentile / 100.0) * (hs.len() - 1) as f64).round() as usize;
        t.height = hs[pos.min(hs.len() - 1)];
    }
}

/// Convex hull (Andrew's 1979 monotone chain) of 2-D points, counter-clockwise, without
/// repeating the first vertex. Fewer than 3 distinct points give them back.
pub fn convex_hull(xy: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut pts: Vec<[f64; 2]> = xy.iter().copied().filter(|p| p[0].is_finite() && p[1].is_finite()).collect();
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap().then(a[1].partial_cmp(&b[1]).unwrap()));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: &[f64; 2], a: &[f64; 2], b: &[f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut lower: Vec<[f64; 2]> = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper: Vec<[f64; 2]> = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.into_iter().chain(upper).collect()
}

/// Area of a polygon (shoelace).
pub fn polygon_area(poly: &[[f64; 2]]) -> f64 {
    let mut area = 0.0;
    for i in 0..poly.len() {
        let j = (i + 1) % poly.len();
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    area.abs() / 2.0
}

/// Whether `p` is inside (or on) a counter-clockwise convex polygon.
pub fn in_convex_polygon(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    (0..poly.len()).all(|i| {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12
    })
}

/// Convex hull area of 2-D points (NaN for fewer than 3).
pub fn convex_hull_area(xy: &[[f64; 2]]) -> f64 {
    let hull = convex_hull(xy);
    if hull.len() < 3 {
        return f64::NAN;
    }
    polygon_area(&hull)
}

/// Shape of a crown from its points (or any 3-D outline of it).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CrownShape {
    /// Vertical projection: convex hull area (m2), its equivalent diameter
    /// and greatest width (m).
    pub projected_area: f64,
    pub diameter: f64,
    pub max_width: f64,
    /// Stacked convex hulls of horizontal slices (m3): follows the crown's
    /// taper and skirt, unlike one 3-D hull.
    pub volume: f64,
    /// Outer surface of the stacked hulls (m2): side walls plus top.
    pub surface: f64,
    pub base_height: f64,
    pub top_height: f64,
    /// Horizontal offset of the crown's area centroid from `base_xy` (m),
    /// its direction (deg, counter-clockwise from +x) and the offset relative
    /// to the equivalent crown radius.
    pub offset: f64,
    pub offset_direction: f64,
    pub asymmetry: f64,
}

/// Crown shape from points between `z_min` and the top: projection, stacked
/// slice hulls every `slice` metres, and asymmetry about `base_xy` (the stem
/// position; the lowest point's position when `None`).
pub fn crown_shape(points: &[Point], base_xy: Option<[f64; 2]>, z_min: f64, slice: f64) -> CrownShape {
    let crown: Vec<&Point> = points.iter().filter(|p| p[2] >= z_min && p.iter().all(|v| v.is_finite())).collect();
    if crown.len() < 3 || !(slice > 0.0) {
        return CrownShape { projected_area: f64::NAN, diameter: f64::NAN, max_width: f64::NAN, volume: f64::NAN, surface: f64::NAN, base_height: z_min, top_height: f64::NAN, offset: f64::NAN, offset_direction: f64::NAN, asymmetry: f64::NAN };
    }
    let top = crown.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
    let base = crown.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
    let xy: Vec<[f64; 2]> = crown.iter().map(|p| [p[0], p[1]]).collect();
    let hull = convex_hull(&xy);
    let area = if hull.len() >= 3 { polygon_area(&hull) } else { 0.0 };
    let mut max_width = 0.0f64;
    for i in 0..hull.len() {
        for j in i + 1..hull.len() {
            max_width = max_width.max((hull[i][0] - hull[j][0]).hypot(hull[i][1] - hull[j][1]));
        }
    }
    // Area centroid of the projection.
    let (mut cx, mut cy, mut a2) = (0.0, 0.0, 0.0);
    for i in 0..hull.len() {
        let (p, q) = (hull[i], hull[(i + 1) % hull.len()]);
        let c = p[0] * q[1] - q[0] * p[1];
        cx += (p[0] + q[0]) * c;
        cy += (p[1] + q[1]) * c;
        a2 += c;
    }
    let centroid = if a2.abs() > 1e-12 { [cx / (3.0 * a2), cy / (3.0 * a2)] } else { [xy.iter().map(|p| p[0]).sum::<f64>() / xy.len() as f64, xy.iter().map(|p| p[1]).sum::<f64>() / xy.len() as f64] };
    let lowest = crown.iter().min_by(|a, b| a[2].partial_cmp(&b[2]).unwrap()).unwrap();
    let origin = base_xy.unwrap_or([lowest[0], lowest[1]]);
    let (dx, dy) = (centroid[0] - origin[0], centroid[1] - origin[1]);
    // Stacked slice hulls.
    let n_slices = (((top - base) / slice).ceil() as usize).max(1);
    let mut slices: Vec<Vec<[f64; 2]>> = vec![Vec::new(); n_slices];
    for p in &crown {
        let k = (((p[2] - base) / slice) as usize).min(n_slices - 1);
        slices[k].push([p[0], p[1]]);
    }
    let (mut volume, mut surface) = (0.0, 0.0);
    let mut top_area = 0.0;
    for s in &slices {
        let h = convex_hull(s);
        if h.len() >= 3 {
            let a = polygon_area(&h);
            let perimeter: f64 = (0..h.len()).map(|i| (h[i][0] - h[(i + 1) % h.len()][0]).hypot(h[i][1] - h[(i + 1) % h.len()][1])).sum();
            volume += a * slice;
            surface += perimeter * slice;
            top_area = a;
        }
    }
    surface += top_area;
    let radius = (area / std::f64::consts::PI).sqrt();
    CrownShape {
        projected_area: area,
        diameter: 2.0 * radius,
        max_width,
        volume,
        surface,
        base_height: base,
        top_height: top,
        offset: dx.hypot(dy),
        offset_direction: dy.atan2(dx).to_degrees(),
        asymmetry: if radius > 0.0 { dx.hypot(dy) / radius } else { f64::NAN },
    }
}

/// Crown metrics for every label in one pass over the cloud.
/// Returns `(tree_id, metrics)` pairs for labels with enough points.
pub fn crown_metrics_all(points: &[Point], heights: &[f64], labels: &[i64], crown_base_fraction: f64) -> Vec<(i64, [f64; 4])> {
    let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if l >= 0 {
            groups.entry(l).or_default().push(i);
        }
    }
    let mut ids: Vec<i64> = groups.keys().cloned().collect();
    ids.sort_unstable();
    ids.par_iter()
        .filter_map(|&id| crown_metrics_indices(points, heights, &groups[&id], crown_base_fraction).map(|m| (id, m)))
        .collect()
}

/// Crown metrics for one tree: `(crown_area, crown_base_height, crown_depth, crown_diameter)`.
pub fn crown_metrics(points: &[Point], heights: &[f64], labels: &[i64], tree_id: i64, crown_base_fraction: f64) -> Option<[f64; 4]> {
    let idx: Vec<usize> = (0..points.len()).filter(|&i| labels[i] == tree_id).collect();
    crown_metrics_indices(points, heights, &idx, crown_base_fraction)
}

fn crown_metrics_indices(points: &[Point], heights: &[f64], idx: &[usize], crown_base_fraction: f64) -> Option<[f64; 4]> {
    if idx.len() < 4 {
        return None;
    }
    let top = idx.iter().map(|&i| heights[i]).fold(f64::NEG_INFINITY, f64::max);
    let nb = ((top / 0.5).ceil() as usize).max(1);
    let mut counts = vec![0usize; nb];
    for &i in idx {
        let b = ((heights[i] / 0.5) as usize).min(nb - 1);
        counts[b] += 1;
    }
    let thresh = crown_base_fraction * *counts.iter().max().unwrap() as f64;
    let start = nb / 4;
    let base_idx = (start..nb).find(|&b| counts[b] as f64 >= thresh).unwrap_or(start);
    let crown_base = base_idx as f64 * 0.5;
    let xy: Vec<[f64; 2]> = idx.iter().filter(|&&i| heights[i] >= crown_base).map(|&i| [points[i][0], points[i][1]]).collect();
    let area = convex_hull_area(&xy);
    let diam = if area.is_finite() { 2.0 * (area / std::f64::consts::PI).sqrt() } else { f64::NAN };
    Some([area, crown_base, top - crown_base, diam])
}

#[cfg(test)]
mod graph_node_tests {
    use super::*;

    /// `graph_nodes` as it was: copy the points above the cut, thin the copy.
    fn reference(points: &[Point], heights: &[f64], p: &SegmentParams) -> Vec<usize> {
        let above: Vec<usize> = (0..points.len()).filter(|&i| heights[i] >= p.cut_above_ground).collect();
        if p.voxel_size.is_nan() || p.voxel_size <= 0.0 {
            return above;
        }
        let above_pts: Vec<Point> = above.iter().map(|&i| points[i]).collect();
        let keep = match &p.voxel_origin {
            Some(o) => crate::filters::voxel_downsample_indices_at(&above_pts, o, p.voxel_size),
            None => crate::filters::voxel_downsample_indices(&above_pts, p.voxel_size),
        };
        keep.into_iter().map(|i| above[i]).collect()
    }

    #[test]
    fn nodes_are_those_of_the_copying_thinning() {
        let mut rng = Rng::new(11);
        let mut unit = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let mut pts: Vec<Point> = (0..6000).map(|_| [5.0 * unit() - 2.0, 5.0 * unit(), 8.0 * unit() - 0.5]).collect();
        for i in 0..300 {
            pts[5000 + i] = pts[i];
        }
        let heights: Vec<f64> = pts.iter().enumerate().map(|(i, q)| if i % 97 == 0 { f64::NAN } else { q[2] }).collect();
        for (voxel, origin) in [(0.03, None), (0.1, Some([0.0, 0.0, 0.0])), (0.0, None), (0.25, Some([1.0, -1.0, 0.0]))] {
            let p = SegmentParams { voxel_size: voxel, voxel_origin: origin, ..Default::default() };
            assert_eq!(graph_nodes(&pts, &heights, &p), reference(&pts, &heights, &p));
        }
        let none = SegmentParams { cut_above_ground: 100.0, ..Default::default() };
        assert!(graph_nodes(&pts, &heights, &none).is_empty());
    }
}

#[cfg(test)]
mod seed_and_height_tests {
    use super::*;

    fn forest(n: usize, seed: u64) -> (Vec<Point>, Vec<f64>, Vec<Tree>) {
        let mut rng = Rng::new(seed);
        let mut unit = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let mut pts: Vec<Point> = (0..n).map(|_| [20.0 * unit() - 10.0, 20.0 * unit() - 10.0, 12.0 * unit()]).collect();
        let trees: Vec<Tree> = (0..9)
            .map(|i| Tree { tree_id: i as i64 + 1, x: 6.0 * (i % 3) as f64 - 6.0, y: 6.0 * (i / 3) as f64 - 6.0, dbh: 0.1 + 0.15 * (i % 4) as f64, height: f64::NAN, n_points: 0, inlier_fraction: 1.0, n_slices: 5, rmse: 0.01, lean_deg: 0.0, quality: 0.9 })
            .collect();
        // Bark rings at seed height around every stem, plus a few exact-distance points.
        for t in &trees {
            for k in 0..60 {
                let a = k as f64 * 0.1047;
                let r = 0.5 * t.dbh + 0.02 * (k % 3) as f64;
                pts.push([t.x + r * a.cos(), t.y + r * a.sin(), 1.5 + 0.3 * ((k % 5) as f64 - 2.0)]);
            }
            pts.push([t.x + 1.5, t.y, 3.0]); // exactly height_prior_radius away
            pts.push([t.x + 0.5 * t.dbh + 0.25, t.y, 1.5]); // exactly on the ring edge
        }
        let hw = pts.iter().map(|q| q[2]).collect();
        (pts, hw, trees)
    }

    /// The seed collection as it was: a pass over every node per tree.
    fn reference(work: &[Point], hw: &[f64], trees: &[Tree], p: &SegmentParams) -> Seeds {
        let mut out = Seeds { seeds: Vec::new(), seed_tree: Vec::new(), seed_xy: Vec::new(), seed_scale: Vec::new() };
        for t in trees {
            let scale = if p.height_prior {
                let mut hs: Vec<f64> = work.iter().zip(hw).filter(|(q, _)| (q[0] - t.x).hypot(q[1] - t.y) <= p.height_prior_radius).map(|(_, &h)| h).collect();
                hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let h95 = hs.get(((hs.len() as f64 * 0.95) as usize).min(hs.len().saturating_sub(1))).copied().unwrap_or(2.0);
                1.0 / h95.max(2.0).powf(p.height_prior_power)
            } else {
                1.0
            };
            let inner = if p.seed_ring { 0.5 * t.dbh } else { 0.0 };
            for (i, q) in work.iter().enumerate() {
                let d = (q[0] - t.x).hypot(q[1] - t.y);
                if (d - inner).abs() <= p.seed_radius && hw[i] < p.seed_height + 0.5 {
                    out.seeds.push(i);
                    out.seed_tree.push(t.tree_id);
                    out.seed_xy.push([t.x, t.y]);
                    out.seed_scale.push(scale);
                }
            }
        }
        out
    }

    #[test]
    fn gridded_seeds_are_the_scanned_seeds() {
        for (seed, prior, ring) in [(1, true, true), (2, false, true), (3, true, false), (4, false, false)] {
            let (pts, hw, trees) = forest(20_000, seed);
            let p = SegmentParams { height_prior: prior, seed_ring: ring, ..Default::default() };
            let a = collect_seeds(&pts, &hw, &trees, &p);
            let b = reference(&pts, &hw, &trees, &p);
            assert!(!a.seeds.is_empty());
            assert_eq!(a.seeds, b.seeds);
            assert_eq!(a.seed_tree, b.seed_tree);
            assert_eq!(a.seed_xy, b.seed_xy);
            assert!(a.seed_scale.iter().zip(&b.seed_scale).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        let (pts, hw, _) = forest(100, 5);
        assert!(collect_seeds(&pts, &hw, &[], &SegmentParams::default()).seeds.is_empty());
        assert!(collect_seeds(&[], &[], &forest(10, 6).2, &SegmentParams::default()).seeds.is_empty());
    }

    #[test]
    fn an_exclusion_mask_keeps_points_out_of_graph_and_labels() {
        let (pts, hw, trees) = forest(8000, 21);
        let p = SegmentParams { voxel_size: 0.05, ..Default::default() };
        // No mask is the unmasked function, bit for bit.
        assert_eq!(segment_trees_masked(&pts, &hw, &trees, &p, None), segment_trees(&pts, &hw, &trees, &p));
        assert_eq!(graph_nodes_masked(&pts, &hw, &p, None), graph_nodes(&pts, &hw, &p));
        // Masked points are neither nodes nor labelled, and the rest still are.
        let mask: Vec<bool> = pts.iter().map(|q| q[2] < 0.6 && q[0] > 0.0).collect();
        let nodes = graph_nodes_masked(&pts, &hw, &p, Some(&mask));
        assert!(!nodes.is_empty() && nodes.iter().all(|&i| !mask[i]));
        let labels = segment_trees_masked(&pts, &hw, &trees, &p, Some(&mask));
        assert!(labels.iter().zip(&mask).all(|(&l, &m)| !m || l == -1));
        assert!(labels.iter().any(|&l| l >= 0));
        // With the cut at the ground and the mask doing its job, the stems reach lower.
        let low = SegmentParams { cut_above_ground: 0.0, voxel_size: 0.05, ..Default::default() };
        let ground: Vec<bool> = pts.iter().map(|q| q[2] < 0.05).collect();
        let labels_low = segment_trees_masked(&pts, &hw, &trees, &low, Some(&ground));
        let labelled_below_cut = labels_low.iter().zip(&pts).filter(|(&l, q)| l >= 0 && q[2] < 0.25).count();
        assert!(labelled_below_cut > 0);
        assert!(labels_low.iter().zip(&ground).all(|(&l, &g)| !g || l == -1));
    }

    #[test]
    fn one_pass_heights_are_the_per_tree_heights() {
        let (pts, hw, mut trees) = forest(5000, 7);
        let labels: Vec<i64> = (0..pts.len()).map(|i| (i % 12) as i64 - 2).collect(); // ids 1..9 plus -2, -1, 0 and 10 (no tree)
        let mut expect = trees.clone();
        for t in expect.iter_mut() {
            let mut hs: Vec<f64> = labels.iter().zip(&hw).filter(|(&l, _)| l == t.tree_id).map(|(_, &h)| h).collect();
            t.n_points = hs.len();
            if hs.is_empty() {
                t.height = f64::NAN;
                continue;
            }
            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pos = ((99.0 / 100.0) * (hs.len() - 1) as f64).round() as usize;
            t.height = hs[pos.min(hs.len() - 1)];
        }
        tree_heights(&hw, &labels, &mut trees, 99.0);
        for (a, b) in trees.iter().zip(&expect) {
            assert_eq!(a.n_points, b.n_points);
            assert_eq!(a.height.to_bits(), b.height.to_bits());
        }
        // A tree with no points, and a duplicated id.
        let mut odd = vec![trees[0].clone(), Tree { tree_id: 77, ..trees[1].clone() }, trees[0].clone()];
        tree_heights(&hw, &labels, &mut odd, 100.0);
        assert_eq!(odd[0].n_points, odd[2].n_points);
        assert_eq!(odd[0].height.to_bits(), odd[2].height.to_bits());
        assert_eq!(odd[1].n_points, 0);
        assert!(odd[1].height.is_nan());
    }
}
