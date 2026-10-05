// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Neighbourhood graphs, connected components and shortest paths.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::util::spatial::KdTree;
use crate::Point;

/// Undirected sparse graph in CSR form with edge weights.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub offsets: Vec<usize>,
    pub neighbors: Vec<u32>,
    pub weights: Vec<f64>,
    /// Optional node xy positions (needed for the gravity term).
    pub xy: Vec<[f64; 2]>,
}

impl Graph {
    pub fn n(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    fn position(&self, i: usize) -> [f64; 2] {
        self.xy.get(i).copied().unwrap_or([0.0, 0.0])
    }

    pub fn edges(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (a, b) = (self.offsets[i], self.offsets[i + 1]);
        self.neighbors[a..b].iter().zip(&self.weights[a..b]).map(|(&j, &w)| (j as usize, w))
    }

    /// Build from an adjacency list, symmetrising.
    fn from_adjacency(mut adj: Vec<Vec<(u32, f64)>>) -> Graph {
        let n = adj.len();
        let mut extra: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for (i, list) in adj.iter().enumerate() {
            for &(j, w) in list {
                extra[j as usize].push((i as u32, w));
            }
        }
        for (i, list) in adj.iter_mut().enumerate() {
            list.append(&mut extra[i]);
            list.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            list.dedup_by(|a, b| a.0 == b.0);
        }
        let mut offsets = Vec::with_capacity(n + 1);
        let mut neighbors = Vec::new();
        let mut weights = Vec::new();
        offsets.push(0);
        for list in &adj {
            for &(j, w) in list {
                neighbors.push(j);
                weights.push(w);
            }
            offsets.push(neighbors.len());
        }
        Graph { offsets, neighbors, weights, xy: Vec::new() }
    }
}

/// Directed graph from per-node adjacency lists (no symmetrisation).
pub fn graph_from_directed(adj: Vec<Vec<(u32, f64)>>) -> Graph {
    let mut offsets = Vec::with_capacity(adj.len() + 1);
    let mut neighbors = Vec::new();
    let mut weights = Vec::new();
    offsets.push(0);
    for list in &adj {
        for &(j, w) in list {
            neighbors.push(j);
            weights.push(w);
        }
        offsets.push(neighbors.len());
    }
    Graph { offsets, neighbors, weights, xy: Vec::new() }
}

/// What a k-nearest-neighbour graph over `n` points will cost, and whether it
/// fits: each edge is stored both ways, as a `u32` and an `f64`, plus the
/// per-node vectors that build it.
fn check_graph(n: usize, k: usize) -> crate::error::Result<()> {
    let edges = (n as u128) * (2 * k as u128 + 2);
    crate::util::limits::check_cells(
        edges,
        (std::mem::size_of::<u32>() + std::mem::size_of::<f64>()) as u64 + 8,
        &format!("a {k}-neighbour graph over {n} points"),
        "thinning the cloud (filters.voxel_downsample), a smaller k, or working tile by tile",
    )
}

/// Directed kNN graph with a direction-dependent cost:
/// `cost(from -> to) = d^power * penalty(angle from vertical)`, where the
/// penalty is `min(exp(0.046051 * deg), 100)` -- travelling up is free,
/// horizontal costs x63 and downward x100 -- so shortest paths from stem
/// seeds climb through the tree rather than leaking across the ground.
/// The neighbour set is symmetrised so the graph is connected both ways.
pub fn directed_knn_graph(points: &[Point], k: usize, max_distance: f64, power: f64, angle_penalty: bool) -> Graph {
    directed_knn_graph_wood(points, k, max_distance, power, angle_penalty, None)
}

/// [`directed_knn_graph`] with wood / leaf cost factors when a per-node
/// `wood` mask is given: wood->wood x0.1, leaf->leaf x20, wood->leaf x1000,
/// leaf->wood x1. Paths run along the woody skeleton and may enter it from
/// foliage, but almost never leave it, so foliage cannot bridge two trees.
pub fn directed_knn_graph_wood(points: &[Point], k: usize, max_distance: f64, power: f64, angle_penalty: bool, wood: Option<&[bool]>) -> Graph {
    if let Err(e) = check_graph(points.len(), k) {
        // The callers of this one cannot carry an error; refusing loudly is
        // still better than the machine going down.
        panic!("{e}");
    }
    let tree = KdTree::new(points);
    let mut adj: Vec<Vec<u32>> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| tree.knn(p, k + 1).into_iter().filter(|&(j, d)| j != i && d <= max_distance).map(|(j, _)| j as u32).collect())
        .collect();
    let mut extra: Vec<Vec<u32>> = vec![Vec::new(); points.len()];
    for (i, list) in adj.iter().enumerate() {
        for &j in list {
            extra[j as usize].push(i as u32);
        }
    }
    for (i, list) in adj.iter_mut().enumerate() {
        list.append(&mut extra[i]);
        list.sort_unstable();
        list.dedup();
    }
    let cost = |from: usize, to: usize| -> f64 {
        let d = crate::transform::sub(&points[to], &points[from]);
        let len = crate::transform::norm(&d).max(1e-12);
        let mut c = len.powf(power);
        if angle_penalty {
            let cos_theta = (d[2] / len).clamp(-1.0, 1.0);
            let deg = cos_theta.acos().to_degrees();
            c *= (0.046051 * deg).exp().min(100.0);
        }
        if let Some(w) = wood {
            c *= match (w[from], w[to]) {
                (true, true) => 0.1,
                (false, false) => 20.0,
                (true, false) => 1000.0,
                (false, true) => 1.0,
            };
        }
        c
    };
    let weighted: Vec<Vec<(u32, f64)>> = adj
        .into_par_iter()
        .enumerate()
        .map(|(i, list)| list.into_iter().map(|j| (j, cost(i, j as usize))).collect())
        .collect();
    let mut g = graph_from_directed(weighted);
    g.xy = points.iter().map(|p| [p[0], p[1]]).collect();
    g
}

/// Symmetric k-nearest-neighbour graph; edges longer than `max_distance` are dropped.
pub fn knn_graph(points: &[Point], k: usize, max_distance: f64) -> Graph {
    if let Err(e) = check_graph(points.len(), k) {
        panic!("{e}");
    }
    let tree = KdTree::new(points);
    let adj: Vec<Vec<(u32, f64)>> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            tree.knn(p, k + 1)
                .into_iter()
                .filter(|&(j, d)| j != i && d <= max_distance)
                .map(|(j, d)| (j as u32, d.max(1e-12)))
                .collect()
        })
        .collect();
    Graph::from_adjacency(adj)
}

/// Graph linking all point pairs closer than `radius`.
pub fn radius_graph(points: &[Point], radius: f64) -> Graph {
    let tree = KdTree::new(points);
    let adj: Vec<Vec<(u32, f64)>> = points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            tree.within(p, radius)
                .into_iter()
                .filter(|&(j, _)| j != i)
                .map(|(j, d)| (j as u32, d.max(1e-12)))
                .collect()
        })
        .collect();
    Graph::from_adjacency(adj)
}

/// Connected component label per node (0-based, arbitrary order).
pub fn connected_components(graph: &Graph) -> (usize, Vec<usize>) {
    let n = graph.n();
    let mut label = vec![usize::MAX; n];
    let mut count = 0;
    let mut stack = Vec::new();
    for s in 0..n {
        if label[s] != usize::MAX {
            continue;
        }
        label[s] = count;
        stack.push(s);
        while let Some(i) = stack.pop() {
            for (j, _) in graph.edges(i) {
                if label[j] == usize::MAX {
                    label[j] = count;
                    stack.push(j);
                }
            }
        }
        count += 1;
    }
    (count, label)
}

/// Euclidean clustering: connected components of the radius graph.
///
/// Returns a label per point; clusters smaller than `min_points` get `-1`.
/// Labels are ordered by decreasing cluster size.
pub fn euclidean_clusters(points: &[Point], radius: f64, min_points: usize) -> Vec<i64> {
    if points.is_empty() {
        return Vec::new();
    }
    let (count, labels) = connected_components(&radius_graph(points, radius));
    let mut sizes = vec![0usize; count];
    for &l in &labels {
        sizes[l] += 1;
    }
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(sizes[c]));
    let mut remap = vec![-1i64; count];
    let mut next = 0;
    for c in order {
        if sizes[c] >= min_points {
            remap[c] = next;
            next += 1;
        }
    }
    labels.iter().map(|&l| remap[l]).collect()
}

#[derive(Copy, Clone, PartialEq)]
struct HeapItem {
    dist: f64,
    node: usize,
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Multi-source Dijkstra.
///
/// Returns per node `(distance, source index into `sources`, predecessor)`;
/// unreachable nodes have infinite distance and `usize::MAX` markers.
pub fn dijkstra(graph: &Graph, sources: &[usize]) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    dijkstra_gravity(graph, sources, None, 0.0)
}

/// [`dijkstra`] with raycloudtools' gravity term (Devereux et al. 2026): each edge cost is scaled by
/// `1 + gravity * lateral^2`, the squared horizontal offset of the node being
/// expanded from the seed its path started at (`seed_xy[source]`), so long
/// horizontal reaches away from a stem are discouraged.
pub fn dijkstra_gravity(graph: &Graph, sources: &[usize], seed_xy: Option<&[[f64; 2]]>, gravity: f64) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    dijkstra_scaled(graph, sources, seed_xy, gravity, None)
}

/// [`dijkstra_gravity`] with an additional per-source cost multiplier
/// (`source_scale[source]`), raycloudtools' `score /= node.radius`: giving
/// taller trees a smaller scale lets them win contested crown points.
pub fn dijkstra_scaled(graph: &Graph, sources: &[usize], seed_xy: Option<&[[f64; 2]]>, gravity: f64, source_scale: Option<&[f64]>) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let n = graph.n();
    let mut dist = vec![f64::INFINITY; n];
    let mut src = vec![usize::MAX; n];
    let mut pred = vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();
    for (si, &s) in sources.iter().enumerate() {
        if dist[s] > 0.0 {
            dist[s] = 0.0;
            src[s] = si;
            heap.push(HeapItem { dist: 0.0, node: s });
        }
    }
    while let Some(HeapItem { dist: d, node: i }) = heap.pop() {
        if d > dist[i] {
            continue;
        }
        let mut scale = match (seed_xy, gravity > 0.0) {
            (Some(sxy), true) if src[i] != usize::MAX => {
                let s = sxy[src[i]];
                let p = graph.position(i);
                let (dx, dy) = (p[0] - s[0], p[1] - s[1]);
                1.0 + gravity * (dx * dx + dy * dy)
            }
            _ => 1.0,
        };
        if let Some(ss) = source_scale {
            if src[i] != usize::MAX {
                scale *= ss[src[i]];
            }
        }
        for (j, w) in graph.edges(i) {
            let nd = d + w * scale;
            if nd < dist[j] {
                dist[j] = nd;
                src[j] = src[i];
                pred[j] = i;
                heap.push(HeapItem { dist: nd, node: j });
            }
        }
    }
    (dist, src, pred)
}
