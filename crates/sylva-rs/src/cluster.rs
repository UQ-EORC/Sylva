// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// The bend cost of the directed graph (`Bend`) follows raycloudtools (CSIRO);
// see THIRD_PARTY_NOTICES.md.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Neighbourhood graphs, connected components and shortest paths.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::util::spatial::KdTree;
use crate::Point;

/// Sparse graph in CSR form.
///
/// Edge weights are either stored (`weights`, one per entry of `neighbors`)
/// or, for the directed kNN graphs of [`directed_knn_graph`], computed from
/// the node positions when an edge is visited: a shortest-path search
/// relaxes every edge once, so storing the weight saves no work and costs
/// eight bytes an edge, which on a plot of a few hundred million points is
/// most of the memory.
#[derive(Debug, Clone, Default)]
pub struct Graph<'a> {
    /// CSR row starts: node `i`'s neighbours are entries `offsets[i]..offsets[i + 1]` of `neighbors` (length n + 1).
    pub offsets: Vec<usize>,
    /// Neighbour node indices of every node, concatenated in node order.
    pub neighbors: Vec<u32>,
    /// One weight per neighbour; empty when `cost` computes them.
    pub weights: Vec<f64>,
    /// Optional node xy positions (needed for the gravity term when the
    /// positions are not in `cost`).
    pub xy: Vec<[f64; 2]>,
    cost: Option<EdgeCost<'a>>,
}

/// How a directed graph costs a change of direction, after the shortest
/// paths of raycloudtools' `rayextract trees` (Lowe and Stepanas 2021;
/// Devereux et al. 2026; `raysegment.cpp`), where
/// an edge costs its squared length over the squared cosine of the angle it
/// makes with the path that arrives at its start.
///
/// An edge of length `d` costs `d^power / max(floor, cos(phi))^self.power`
/// (`power` the graph's length exponent), with `phi` the angle between the
/// edge and a reference direction: the vertical, turned by `path_weight`
/// towards the direction the path arrived in (0 is the vertical alone, 1
/// the path's direction alone, as in raycloudtools). An edge along the
/// reference direction is not penalised; `floor` bounds the factor for
/// edges across or against it at `1 / floor^power`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bend {
    /// Exponent on the cosine of the bend (raycloudtools: 2).
    pub power: f64,
    /// Smallest cosine counted, which caps the factor (0-1).
    pub floor: f64,
    /// Share of the path's own direction in the reference direction (0-1).
    pub path_weight: f64,
}

impl Default for Bend {
    /// Sideways and downward edges cost at most 39 times an upward one:
    /// measured at least as good as Sylva 1.1's cost on dense, thinned and
    /// simulated plots alike.
    fn default() -> Self {
        Bend { power: 4.0, floor: 0.4, path_weight: 0.3 }
    }
}

/// The edge cost of a directed graph, computed from the node positions.
#[derive(Debug, Clone)]
struct EdgeCost<'a> {
    points: &'a [Point],
    power: f64,
    bend: Option<Bend>,
}

const UP: [f64; 3] = [0.0, 0.0, 1.0];

impl EdgeCost<'_> {
    /// The cost of the edge `from -> to` when the reference direction at
    /// `from` is the unit vector `dir`.
    #[inline]
    fn cost(&self, from: usize, to: usize, dir: &[f64; 3]) -> f64 {
        let d = crate::transform::sub(&self.points[to], &self.points[from]);
        let len = crate::transform::norm(&d).max(1e-12);
        let c = len.powf(self.power);
        match self.bend {
            Some(b) => {
                let cos = ((d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2]) / len).max(b.floor);
                c / cos.powf(b.power)
            }
            None => c,
        }
    }

    /// The reference direction at node `i` given the search's predecessors:
    /// the vertical, turned towards the direction from `i`'s grandparent
    /// (or parent) to `i` by the bend's `path_weight`.
    #[inline]
    fn direction(&self, i: usize, pred: &[usize]) -> [f64; 3] {
        let Some(b) = self.bend else { return UP };
        let p = pred[i];
        if b.path_weight <= 0.0 || p == usize::MAX {
            return UP;
        }
        let from = if pred[p] != usize::MAX { pred[p] } else { p };
        let d = crate::transform::sub(&self.points[i], &self.points[from]);
        let len = crate::transform::norm(&d);
        if len < 1e-12 {
            return UP;
        }
        let w = b.path_weight.min(1.0);
        let v = [w * d[0] / len, w * d[1] / len, w * d[2] / len + (1.0 - w)];
        let n = crate::transform::norm(&v);
        if n < 1e-12 { UP } else { [v[0] / n, v[1] / n, v[2] / n] }
    }
}

impl<'a> Graph<'a> {
    /// Number of nodes: one less than the length of `offsets`, 0 for an empty graph.
    pub fn n(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    fn position(&self, i: usize) -> [f64; 2] {
        match &self.cost {
            Some(c) => [c.points[i][0], c.points[i][1]],
            None => self.xy.get(i).copied().unwrap_or([0.0, 0.0]),
        }
    }

    /// Edge weight of the `e`-th entry of `neighbors`, leaving node `i`.
    #[inline]
    fn weight(&self, i: usize, e: usize) -> f64 {
        match &self.cost {
            Some(c) => c.cost(i, self.neighbors[e] as usize, &UP),
            None => self.weights[e],
        }
    }

    /// The edges leaving node `i` as `(neighbour, weight)` pairs, the weight stored
    /// or computed from the positions as the graph was built (for a graph
    /// with a [`Bend`], taken against the vertical: a shortest-path search
    /// uses the direction its path arrived in).
    pub fn edges(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let (a, b) = (self.offsets[i], self.offsets[i + 1]);
        (a..b).map(move |e| (self.neighbors[e] as usize, self.weight(i, e)))
    }

    /// Build from an adjacency list, symmetrising.
    fn from_adjacency(mut adj: Vec<Vec<(u32, f64)>>) -> Graph<'static> {
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
        graph_from_directed(adj)
    }

    /// The symmetrised neighbour sets of a k-nearest-neighbour search, as
    /// CSR offsets and neighbours: each node's list holds the nodes it
    /// found and the nodes that found it, sorted by index, each once. Built
    /// in flat arrays - a slot table of the search results, a reverse pass,
    /// a sort of each node's run - rather than a vector per node, which for
    /// a few neighbours costs more in headers and slack than in neighbours.
    fn symmetric_knn_csr(points: &[Point], k: usize, max_distance: f64, bridge: bool) -> (Vec<usize>, Vec<u32>) {
        let n = points.len();
        let tree = KdTree::new(points);
        // Up to k + 1 results survive the filter when the point's own entry
        // is not among the k + 1 nearest (coincident points), so a slot
        // table of k + 1 per node holds every list; u32::MAX marks an empty slot.
        let slots = k + 1;
        let mut found = vec![u32::MAX; n * slots];
        found.par_chunks_mut(slots).enumerate().for_each(|(i, row)| {
            let mut m = 0;
            for (j, d) in tree.knn(&points[i], k + 1) {
                if j != i && d <= max_distance {
                    row[m] = j as u32;
                    m += 1;
                }
            }
        });
        drop(tree);
        let extra = if bridge { bridges(points, &found, slots, max_distance) } else { Vec::new() };
        // Degree of each node: what it found plus what found it, and the bridges.
        let mut degree = vec![0usize; n];
        for (i, row) in found.chunks(slots).enumerate() {
            for &j in row {
                if j == u32::MAX {
                    break;
                }
                degree[i] += 1;
                degree[j as usize] += 1;
            }
        }
        for &(i, j) in &extra {
            degree[i as usize] += 1;
            degree[j as usize] += 1;
        }
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0usize);
        for &d in &degree {
            offsets.push(offsets.last().unwrap() + d);
        }
        let mut neighbors = vec![0u32; offsets[n]];
        // Forward entries first, in each node's run, then the reverse ones.
        let mut cursor: Vec<usize> = offsets[..n].to_vec();
        for (i, row) in found.chunks(slots).enumerate() {
            for &j in row {
                if j == u32::MAX {
                    break;
                }
                neighbors[cursor[i]] = j;
                cursor[i] += 1;
            }
        }
        for (i, row) in found.chunks(slots).enumerate() {
            for &j in row {
                if j == u32::MAX {
                    break;
                }
                let j = j as usize;
                neighbors[cursor[j]] = i as u32;
                cursor[j] += 1;
            }
        }
        for &(i, j) in &extra {
            neighbors[cursor[i as usize]] = j;
            cursor[i as usize] += 1;
            neighbors[cursor[j as usize]] = i;
            cursor[j as usize] += 1;
        }
        drop(found);
        drop(cursor);
        // Sort and dedup each run in place; `kept[i]` is its new length.
        let mut kept = vec![0usize; n];
        {
            let mut runs: Vec<&mut [u32]> = Vec::with_capacity(n);
            let mut rest: &mut [u32] = &mut neighbors;
            for i in 0..n {
                let (run, tail) = rest.split_at_mut(degree[i]);
                runs.push(run);
                rest = tail;
            }
            runs.into_par_iter().zip(kept.par_iter_mut()).for_each(|(run, kept)| {
                run.sort_unstable();
                let mut m = 0;
                for e in 0..run.len() {
                    if e == 0 || run[e] != run[e - 1] {
                        run[m] = run[e];
                        m += 1;
                    }
                }
                *kept = m;
            });
        }
        // Close the gaps the duplicates left: each run moves down to the end
        // of the previous one (the runs' old starts are the degree sums).
        let mut write = 0;
        let mut start = 0;
        for i in 0..n {
            if write != start {
                neighbors.copy_within(start..start + kept[i], write);
            }
            write += kept[i];
            start += degree[i];
            offsets[i + 1] = write;
        }
        neighbors.truncate(write);
        neighbors.shrink_to_fit();
        (offsets, neighbors)
    }
}

/// Root of `x` in a union-find forest, halving the path on the way.
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize];
        x = p;
    }
    x
}

/// The edges that join the connected components of a k-nearest-neighbour
/// graph wherever two of them come within `max_distance`: for every pair
/// of components within reach, the closest pair of nodes between them,
/// repeated until no two components are that close. The graph then
/// connects whatever the cloud connects at `max_distance`, however the
/// points were thinned: a crown joined to its trunk by a few sparsely
/// sampled branches is no longer at the mercy of which point each voxel
/// kept, which is what made a k-neighbour graph fall apart there. A piece
/// near two trees is joined to both, and the path costs decide which it
/// belongs to, as they would in a radius graph. `found` is the slot table
/// of the search, `slots` wide per node.
///
/// The closest pairs come from a search over a bounding-box tree of the
/// nodes in which every box records the component its nodes belong to,
/// when they all belong to one: two boxes of the same component have
/// nothing to offer and are skipped whole, so the inside of a tree costs
/// nothing and the work goes where components meet. A radius search from
/// every node would visit the whole metre around each of them, which on a
/// dense plot is thousands of nodes a node and hours a pass.
fn bridges(points: &[Point], found: &[u32], slots: usize, max_distance: f64) -> Vec<(u32, u32)> {
    let n = points.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for (i, row) in found.chunks(slots).enumerate() {
        for &j in row {
            if j == u32::MAX {
                break;
            }
            let (a, b) = (find(&mut parent, i as u32), find(&mut parent, j));
            if a != b {
                parent[a as usize] = b;
            }
        }
    }
    let mut extra = Vec::new();
    if n == 0 {
        return extra;
    }
    let tree = BoxTree::new(points);
    // Each pass joins every component to its neighbours within reach;
    // chains of components take a few passes, and the count is bounded by
    // the number of components, which halves at least each pass.
    for _ in 0..64 {
        let comp: Vec<u32> = (0..n as u32).map(|i| find(&mut parent, i)).collect();
        if comp.iter().all(|&c| c == comp[0]) {
            break;
        }
        let best = tree.closest_pairs(&comp, max_distance);
        if best.is_empty() {
            break;
        }
        // Every pair's edge goes in (so a piece near two trees reaches
        // both); the unions decide whether another pass is needed.
        let mut joined = false;
        let mut picks: Vec<(f64, u32, u32)> = best.into_values().collect();
        picks.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (_, i, j) in picks {
            extra.push((i, j));
            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
            if a != b {
                parent[a as usize] = b;
                joined = true;
            }
        }
        if !joined {
            break;
        }
    }
    extra
}

/// The closest pair of nodes per pair of components: `(distance, i, j)`
/// under the component pair `(a, b)` with `a < b`.
type ClosestPairs = std::collections::HashMap<(u32, u32), (f64, u32, u32)>;

/// A bounding-box tree over a point set: a k-d split down to leaves of at
/// most [`BoxTree::LEAF`] points, each node holding the tight box of its
/// points. Built once per graph; the component flags that drive
/// [`BoxTree::closest_pairs`] are recomputed per pass.
struct BoxTree<'a> {
    points: &'a [Point],
    /// Point indices, permuted so that every node holds a contiguous run.
    perm: Vec<u32>,
    nodes: Vec<BoxNode>,
}

#[derive(Clone)]
struct BoxNode {
    lo: Point,
    hi: Point,
    /// Run of `perm` this node holds.
    start: u32,
    end: u32,
    /// First child; the second is `left + 1`. `u32::MAX` for a leaf.
    left: u32,
}

/// Marker for a box whose points span more than one component.
const MIXED: u32 = u32::MAX;

impl<'a> BoxTree<'a> {
    const LEAF: usize = 16;

    fn new(points: &'a [Point]) -> Self {
        let perm: Vec<u32> = (0..points.len() as u32).collect();
        let mut nodes = Vec::with_capacity(2 * points.len() / Self::LEAF + 1);
        let root = BoxNode { lo: [0.0; 3], hi: [0.0; 3], start: 0, end: points.len() as u32, left: u32::MAX };
        nodes.push(root);
        let mut tree = BoxTree { points, perm, nodes };
        tree.build(0);
        tree
    }

    /// Lay out the subtree at `node`: box the run, split the longest axis
    /// at its median point, recurse. Children follow their parent in
    /// `nodes`, so a reverse scan visits every child before its parent.
    fn build(&mut self, node: usize) {
        let (start, end) = (self.nodes[node].start as usize, self.nodes[node].end as usize);
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &i in &self.perm[start..end] {
            for k in 0..3 {
                lo[k] = lo[k].min(self.points[i as usize][k]);
                hi[k] = hi[k].max(self.points[i as usize][k]);
            }
        }
        self.nodes[node].lo = lo;
        self.nodes[node].hi = hi;
        if end - start <= Self::LEAF {
            return;
        }
        let axis = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap();
        let mid = (start + end) / 2;
        let points = self.points;
        self.perm[start..end].select_nth_unstable_by(mid - start, |&a, &b| points[a as usize][axis].total_cmp(&points[b as usize][axis]));
        let left = self.nodes.len() as u32;
        self.nodes[node].left = left;
        self.nodes.push(BoxNode { lo, hi, start: start as u32, end: mid as u32, left: u32::MAX });
        self.nodes.push(BoxNode { lo, hi, start: mid as u32, end: end as u32, left: u32::MAX });
        self.build(left as usize);
        self.build(left as usize + 1);
    }

    /// The component of each box: its nodes' one component, or `MIXED`.
    fn flags(&self, comp: &[u32]) -> Vec<u32> {
        let mut flag = vec![MIXED; self.nodes.len()];
        for (b, node) in self.nodes.iter().enumerate().rev() {
            flag[b] = if node.left == u32::MAX {
                let mut run = self.perm[node.start as usize..node.end as usize].iter().map(|&i| comp[i as usize]);
                let first = run.next().unwrap_or(MIXED);
                if run.all(|c| c == first) {
                    first
                } else {
                    MIXED
                }
            } else {
                let (l, r) = (flag[node.left as usize], flag[node.left as usize + 1]);
                if l == r {
                    l
                } else {
                    MIXED
                }
            };
        }
        flag
    }

    /// Squared distance between two boxes (zero when they overlap).
    fn gap(&self, a: usize, b: usize) -> f64 {
        let (a, b) = (&self.nodes[a], &self.nodes[b]);
        (0..3)
            .map(|k| {
                let d = (a.lo[k] - b.hi[k]).max(b.lo[k] - a.hi[k]).max(0.0);
                d * d
            })
            .sum()
    }

    /// The closest pair of nodes between every two components that come
    /// within `max_distance`, for the component labels `comp`.
    fn closest_pairs(&self, comp: &[u32], max_distance: f64) -> ClosestPairs {
        let flag = self.flags(comp);
        let limit = max_distance * max_distance;
        // Widen the root pair into enough independent box pairs to share
        // out, then search each and keep the closest per component pair.
        let mut pairs = vec![(0usize, 0usize)];
        let mut next = Vec::new();
        for _ in 0..7 {
            next.clear();
            for &(a, b) in &pairs {
                self.expand(a, b, &flag, limit, &mut next);
            }
            std::mem::swap(&mut pairs, &mut next);
        }
        pairs
            .par_iter()
            .fold(ClosestPairs::new, |mut best, &(a, b)| {
                self.search(a, b, comp, &flag, limit, &mut best);
                best
            })
            .reduce(ClosestPairs::new, |mut x, y| {
                for (key, v) in y {
                    let e = x.entry(key).or_insert(v);
                    if v < *e {
                        *e = v;
                    }
                }
                x
            })
    }

    /// Whether the pair of boxes can hold a bridge at all.
    fn worth(&self, a: usize, b: usize, flag: &[u32], limit: f64) -> bool {
        !(flag[a] != MIXED && flag[a] == flag[b]) && self.gap(a, b) <= limit
    }

    /// The child pairs of a box pair, or the pair itself when it has no
    /// children to open, dropping what `worth` rules out.
    fn expand(&self, a: usize, b: usize, flag: &[u32], limit: f64, out: &mut Vec<(usize, usize)>) {
        if !self.worth(a, b, flag, limit) {
            return;
        }
        let (la, lb) = (self.nodes[a].left, self.nodes[b].left);
        match (la != u32::MAX, lb != u32::MAX) {
            (false, false) => out.push((a, b)),
            (true, false) => out.extend([(la as usize, b), (la as usize + 1, b)]),
            (false, true) => out.extend([(a, lb as usize), (a, lb as usize + 1)]),
            (true, true) if a == b => out.extend([(la as usize, la as usize), (la as usize + 1, la as usize + 1), (la as usize, la as usize + 1)]),
            (true, true) => out.extend([(la as usize, lb as usize), (la as usize, lb as usize + 1), (la as usize + 1, lb as usize), (la as usize + 1, lb as usize + 1)]),
        }
    }

    /// Search a pair of boxes for closest pairs between components, into `best`.
    fn search(&self, a: usize, b: usize, comp: &[u32], flag: &[u32], limit: f64, best: &mut ClosestPairs) {
        if !self.worth(a, b, flag, limit) {
            return;
        }
        if flag[a] != MIXED && flag[b] != MIXED {
            // Two boxes of two components: nothing to gain past the pair
            // already found for them.
            let key = if flag[a] < flag[b] { (flag[a], flag[b]) } else { (flag[b], flag[a]) };
            if best.get(&key).is_some_and(|e| e.0 * e.0 <= self.gap(a, b)) {
                return;
            }
        }
        let (la, lb) = (self.nodes[a].left, self.nodes[b].left);
        if la == u32::MAX && lb == u32::MAX {
            let (na, nb) = (&self.nodes[a], &self.nodes[b]);
            for &i in &self.perm[na.start as usize..na.end as usize] {
                let ci = comp[i as usize];
                for &j in &self.perm[nb.start as usize..nb.end as usize] {
                    if a == b && j <= i {
                        continue;
                    }
                    let cj = comp[j as usize];
                    if ci == cj {
                        continue;
                    }
                    let (p, q) = (&self.points[i as usize], &self.points[j as usize]);
                    let d2 = (0..3).map(|k| (p[k] - q[k]) * (p[k] - q[k])).sum::<f64>();
                    if d2 > limit {
                        continue;
                    }
                    let (key, i, j) = if ci < cj { ((ci, cj), i, j) } else { ((cj, ci), j, i) };
                    let d = d2.sqrt();
                    let e = best.entry(key).or_insert((f64::INFINITY, 0, 0));
                    if d < e.0 {
                        *e = (d, i, j);
                    }
                }
            }
            return;
        }
        let mut children = Vec::with_capacity(4);
        self.expand(a, b, flag, limit, &mut children);
        for (x, y) in children {
            self.search(x, y, comp, flag, limit, best);
        }
    }

}

/// Directed graph from per-node adjacency lists (no symmetrisation).
pub fn graph_from_directed(adj: Vec<Vec<(u32, f64)>>) -> Graph<'static> {
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
    Graph { offsets, neighbors, weights, xy: Vec::new(), cost: None }
}

/// What a k-nearest-neighbour graph over `n` points will cost, and whether it
/// fits. The symmetrised neighbour lists hold up to `2k + 2` entries a node
/// as `u32`, built from a slot table of `k + 1` a node, with the degrees,
/// offsets and the kd-tree's copy of the points beside them (about 60 bytes
/// a node); a graph with stored weights adds an `f64` an entry.
fn check_graph(n: usize, k: usize, stored_weights: bool) -> crate::error::Result<()> {
    let per_edge = std::mem::size_of::<u32>() as u64 + if stored_weights { std::mem::size_of::<f64>() as u64 } else { 0 };
    let edges = (n as u128) * (3 * k as u128 + 3);
    let bytes = edges.saturating_mul(per_edge as u128).saturating_add((n as u128) * 60);
    crate::util::limits::check_cells(
        bytes,
        1,
        &format!("a {k}-neighbour graph over {n} points"),
        "thinning the cloud (filters.voxel_downsample), a smaller k, or working tile by tile",
    )
}

/// Directed kNN graph whose cost depends on direction:
/// `cost(from -> to) = d^power`, divided with a [`Bend`] by a power of the
/// cosine of the angle the edge makes with the vertical and the arriving
/// path, so that shortest paths from stem seeds climb through a tree and do
/// not leak across the ground. The neighbour set is symmetrised so the
/// graph is connected both ways.
///
/// The graph borrows `points`: edge costs are computed from them as the
/// edges are walked, not stored.
pub fn directed_knn_graph(points: &[Point], k: usize, max_distance: f64, power: f64, bend: Option<Bend>) -> Graph<'_> {
    directed_knn_graph_full(points, k, max_distance, power, bend, false)
}

/// [`directed_knn_graph`] with its components joined wherever two come
/// within `max_distance` of each other (see `bridges`), so that connectivity
/// is that of the cloud at `max_distance` and not of the thinning.
pub fn directed_knn_graph_bridged(points: &[Point], k: usize, max_distance: f64, power: f64, bend: Option<Bend>) -> Graph<'_> {
    directed_knn_graph_full(points, k, max_distance, power, bend, true)
}

fn directed_knn_graph_full(points: &[Point], k: usize, max_distance: f64, power: f64, bend: Option<Bend>, bridge: bool) -> Graph<'_> {
    if let Err(e) = check_graph(points.len(), k, false) {
        // The callers of this one cannot carry an error; refusing loudly is
        // still better than the machine going down.
        panic!("{e}");
    }
    let (offsets, neighbors) = Graph::symmetric_knn_csr(points, k, max_distance, bridge);
    Graph { offsets, neighbors, weights: Vec::new(), xy: Vec::new(), cost: Some(EdgeCost { points, power, bend }) }
}

/// Symmetric k-nearest-neighbour graph; edges longer than `max_distance` are dropped.
pub fn knn_graph(points: &[Point], k: usize, max_distance: f64) -> Graph<'static> {
    if let Err(e) = check_graph(points.len(), k, true) {
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

/// Graph linking all point pairs closer than `radius` (m), each edge stored
/// with its length.
///
/// The number of edges is not known from the point count - it grows with
/// the cube of `radius` in a dense cloud - so the neighbours are counted
/// first and the graph refused when it would not fit in memory (about 40
/// bytes an edge while it is built). To label connected pieces only,
/// [`euclidean_clusters`] stores no edges at all.
///
/// # Panics
/// When the graph would exceed the memory budget of [`crate::util::limits`],
/// with a message saying what it needs.
pub fn radius_graph(points: &[Point], radius: f64) -> Graph<'static> {
    let tree = KdTree::new(points);
    let edges: u128 = points.par_iter().map(|p| tree.count_within(p, radius).saturating_sub(1) as u128).sum();
    if let Err(e) = crate::util::limits::check_cells(edges, 40, &format!("a graph of {edges} edges within {radius} m over {} points", points.len()), "a smaller radius or a thinned cloud (filters.voxel_downsample)") {
        // As for the k-neighbour graphs: refusing loudly is better than the
        // machine going down.
        panic!("{e}");
    }
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

/// Connected pieces of the points joined wherever two are within `radius`
/// (m): the number of pieces and the piece of each point, numbered in the
/// order their first points come.
///
/// The pieces are those of [`connected_components`] on [`radius_graph`],
/// found without the graph: the neighbours of a batch of points are
/// searched, merged into a union-find forest and dropped, so memory stays
/// at a few words a point however many neighbours each has.
pub fn radius_components(points: &[Point], radius: f64) -> (usize, Vec<usize>) {
    let n = points.len();
    assert!(n <= u32::MAX as usize, "clustering is limited to {} points, got {n}", u32::MAX);
    let tree = KdTree::new(points);
    let mut parent: Vec<u32> = (0..n as u32).collect();
    // Batches small enough that their neighbour lists are a modest block
    // even at a thousand neighbours a point.
    const BATCH: usize = 1 << 14;
    let mut start = 0;
    while start < n {
        let end = (start + BATCH).min(n);
        let found: Vec<Vec<u32>> = (start..end).into_par_iter().map(|i| tree.within(&points[i], radius).into_iter().filter(|&(j, _)| j > i).map(|(j, _)| j as u32).collect()).collect();
        for (k, list) in found.into_iter().enumerate() {
            let a = find(&mut parent, (start + k) as u32);
            for j in list {
                let b = find(&mut parent, j);
                if a != b {
                    parent[b as usize] = a;
                }
            }
        }
        start = end;
    }
    let mut id = vec![usize::MAX; n];
    let mut label = vec![0usize; n];
    let mut count = 0;
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        if id[r] == usize::MAX {
            id[r] = count;
            count += 1;
        }
        label[i] = id[r];
    }
    (count, label)
}

/// Euclidean clustering: the points joined wherever two are within `radius`
/// (m), as connected components of the radius graph.
///
/// Returns a label per point; clusters smaller than `min_points` get `-1`.
/// Labels are ordered by decreasing cluster size, equal sizes in the order
/// of their first points. No graph is stored ([`radius_components`]), so a
/// dense cloud with thousands of neighbours a point costs time, not memory.
pub fn euclidean_clusters(points: &[Point], radius: f64, min_points: usize) -> Vec<i64> {
    if points.is_empty() {
        return Vec::new();
    }
    let (count, labels) = radius_components(points, radius);
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
    dijkstra_reach(graph, sources, seed_xy, gravity, source_scale, None)
}

/// [`dijkstra_scaled`] with a horizontal reach per source: a path from source
/// `s` does not enter a node farther than `reach[s]` (m, horizontal) from
/// `seed_xy[s]`, so every node a source claims lies within its reach. An
/// infinite reach is no limit; without `seed_xy` the reach is ignored.
pub fn dijkstra_reach(graph: &Graph, sources: &[usize], seed_xy: Option<&[[f64; 2]]>, gravity: f64, source_scale: Option<&[f64]>, reach: Option<&[f64]>) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
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
        // With a bend that follows the path, an edge's cost depends on how
        // the search arrived at `i`, which is settled once `i` is taken off
        // the heap.
        let dir = match &graph.cost {
            Some(c) => c.direction(i, &pred),
            None => UP,
        };
        // The reach of the path's source, squared, and its stem.
        let bound = match (seed_xy, reach) {
            (Some(sxy), Some(r)) if src[i] != usize::MAX && r[src[i]].is_finite() => Some((sxy[src[i]], r[src[i]] * r[src[i]])),
            _ => None,
        };
        for e in graph.offsets[i]..graph.offsets[i + 1] {
            let j = graph.neighbors[e] as usize;
            if let Some((s, r2)) = bound {
                let q = graph.position(j);
                let (dx, dy) = (q[0] - s[0], q[1] - s[1]);
                if dx * dx + dy * dy > r2 {
                    continue;
                }
            }
            let w = match &graph.cost {
                Some(c) => c.cost(i, j, &dir),
                None => graph.weights[e],
            };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::Rng;

    /// The graph as it was built before the flat CSR: a vector per node,
    /// symmetrised, sorted and deduplicated, with the costs stored.
    fn reference(points: &[Point], k: usize, max_distance: f64, power: f64, bend: Option<Bend>) -> Graph<'static> {
        let tree = KdTree::new(points);
        let mut adj: Vec<Vec<u32>> = points.iter().enumerate().map(|(i, p)| tree.knn(p, k + 1).into_iter().filter(|&(j, d)| j != i && d <= max_distance).map(|(j, _)| j as u32).collect()).collect();
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
        let cost = EdgeCost { points, power, bend };
        let weighted: Vec<Vec<(u32, f64)>> = adj.into_iter().enumerate().map(|(i, list)| list.into_iter().map(|j| (j, cost.cost(i, j as usize, &UP))).collect()).collect();
        let mut g = graph_from_directed(weighted);
        g.xy = points.iter().map(|p| [p[0], p[1]]).collect();
        g
    }

    fn cloud(n: usize, seed: u64, coincident: bool) -> Vec<Point> {
        let mut rng = Rng::new(seed);
        let mut unit = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let mut pts: Vec<Point> = (0..n).map(|_| [4.0 * unit(), 4.0 * unit(), 6.0 * unit()]).collect();
        if coincident {
            // Clumps of identical points, so a point's own entry can fall
            // outside its k + 1 nearest.
            for i in 0..n / 4 {
                pts[n / 2 + i] = pts[i];
                pts[n / 4 * 3 + i % (n / 4)] = pts[i];
            }
        }
        pts
    }

    fn same(a: &Graph<'_>, b: &Graph<'_>) {
        assert_eq!(a.offsets, b.offsets);
        assert_eq!(a.neighbors, b.neighbors);
        for i in 0..a.n() {
            let ea: Vec<(usize, f64)> = a.edges(i).collect();
            let eb: Vec<(usize, f64)> = b.edges(i).collect();
            assert_eq!(ea.len(), eb.len());
            for (x, y) in ea.iter().zip(&eb) {
                assert_eq!(x.0, y.0);
                assert!(x.1.to_bits() == y.1.to_bits(), "node {i}: {} vs {}", x.1, y.1);
            }
            assert_eq!(a.position(i), b.position(i));
        }
    }

    #[test]
    fn flat_csr_matches_the_vector_build() {
        for (seed, coincident, max_edge) in [(1, false, 1.0), (2, true, 1.0), (3, false, 0.3), (4, true, 0.05)] {
            let pts = cloud(400, seed, coincident);
            // A bend against the vertical alone is a cost per edge, which
            // a stored graph can hold.
            let vertical = Some(Bend { path_weight: 0.0, ..Bend::default() });
            let g = directed_knn_graph(&pts, 6, max_edge, 6.0, vertical);
            same(&g, &reference(&pts, 6, max_edge, 6.0, vertical));
            let g = directed_knn_graph(&pts, 10, max_edge, 3.0, None);
            same(&g, &reference(&pts, 10, max_edge, 3.0, None));
        }
    }

    #[test]
    fn paths_are_those_of_the_stored_graph() {
        let pts = cloud(600, 9, true);
        let vertical = Some(Bend { path_weight: 0.0, ..Bend::default() });
        let g = directed_knn_graph(&pts, 6, 1.0, 6.0, vertical);
        let r = reference(&pts, 6, 1.0, 6.0, vertical);
        let sources = [0usize, 7, 123];
        let xy = [[pts[0][0], pts[0][1]], [pts[7][0], pts[7][1]], [pts[123][0], pts[123][1]]];
        let scale = [0.5, 1.0, 0.25];
        let a = dijkstra_scaled(&g, &sources, Some(&xy), 0.3, Some(&scale));
        let b = dijkstra_scaled(&r, &sources, Some(&xy), 0.3, Some(&scale));
        assert!(a.0.iter().zip(&b.0).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
        assert_eq!(connected_components(&g), connected_components(&r));
    }

    #[test]
    fn bridges_join_what_the_cloud_joins() {
        // Two dense vertical lines 0.8 m apart: every node's six nearest are
        // on its own line, so the kNN graph has two components; bridged within
        // 1 m it has one, within 0.5 m still two.
        let mut pts: Vec<Point> = (0..200).map(|i| [0.0, 0.0, i as f64 * 0.02]).collect();
        pts.extend((0..200).map(|i| [0.8, 0.0, i as f64 * 0.02]));
        let plain = directed_knn_graph(&pts, 6, 1.0, 6.0, Some(Bend::default()));
        assert_eq!(connected_components(&plain).0, 2);
        let bridged = directed_knn_graph_bridged(&pts, 6, 1.0, 6.0, Some(Bend::default()));
        assert_eq!(connected_components(&bridged).0, 1);
        assert_eq!(bridged.neighbors.len(), plain.neighbors.len() + 2);
        let apart = directed_knn_graph_bridged(&pts, 6, 0.5, 6.0, Some(Bend::default()));
        assert_eq!(connected_components(&apart).0, 2);
        // A chain of four clumps joins in passes, and a lone far point stays apart.
        let mut chain: Vec<Point> = Vec::new();
        for c in 0..4 {
            chain.extend((0..50).map(|i| [c as f64 * 0.9, 0.0, i as f64 * 0.01]));
        }
        chain.push([10.0, 10.0, 0.0]);
        assert_eq!(connected_components(&directed_knn_graph_bridged(&chain, 6, 1.0, 6.0, Some(Bend::default()))).0, 2);
        // Bridging an already connected graph adds nothing.
        let dense = cloud(300, 4, false);
        let same = directed_knn_graph_bridged(&dense, 6, 1.0, 6.0, Some(Bend::default()));
        let base = directed_knn_graph(&dense, 6, 1.0, 6.0, Some(Bend::default()));
        assert_eq!(connected_components(&base).0, 1);
        assert_eq!(same.neighbors, base.neighbors);
    }

    #[test]
    fn clusters_without_a_graph_are_those_of_the_graph() {
        // Clumps of very different sizes at UTM-scale coordinates, with coincident points.
        let mut rng = Rng::new(21);
        let mut unit = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let mut pts: Vec<Point> = Vec::new();
        for c in 0..40 {
            let centre = [512_000.0 + 30.0 * unit(), 6_234_000.0 + 30.0 * unit(), 300.0 + 5.0 * unit()];
            for _ in 0..(5 + 37 * c % 211) {
                pts.push([centre[0] + 0.6 * unit(), centre[1] + 0.6 * unit(), centre[2] + 0.6 * unit()]);
            }
        }
        for i in 0..50 {
            pts.push(pts[i * 3]);
        }
        for radius in [0.05, 0.2, 0.7] {
            let (count, labels) = radius_components(&pts, radius);
            assert_eq!((count, labels.clone()), connected_components(&radius_graph(&pts, radius)), "radius {radius}");
            // The labels of the clustering, as it gave them from the graph.
            let mut sizes = vec![0usize; count];
            labels.iter().for_each(|&l| sizes[l] += 1);
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by_key(|&c| std::cmp::Reverse(sizes[c]));
            let mut remap = vec![-1i64; count];
            let mut next = 0;
            for c in order {
                if sizes[c] >= 10 {
                    remap[c] = next;
                    next += 1;
                }
            }
            assert_eq!(euclidean_clusters(&pts, radius, 10), labels.iter().map(|&l| remap[l]).collect::<Vec<_>>());
        }
        assert_eq!(radius_components(&[], 1.0), (0, Vec::new()));
    }

    #[test]
    fn the_bend_sets_what_a_turn_costs() {
        // Two points 0.1 m apart, side by side and one above the other.
        let side: [Point; 2] = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]];
        let up: [Point; 2] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]];
        let bend = Bend { power: 2.0, floor: 0.25, path_weight: 1.0 };
        let cost = |points: &[Point], from: usize, to: usize, dir: [f64; 3]| EdgeCost { points, power: 2.0, bend: Some(bend) }.cost(from, to, &dir);
        let base = 0.01;
        let close = |a: f64, b: f64| (a / b - 1.0).abs() < 1e-9;
        assert!(close(cost(&up, 0, 1, UP), base)); // along the reference direction: free
        assert!(close(cost(&side, 0, 1, UP), base * 16.0)); // across it: the floor, 1 / 0.25^2
        assert!(close(cost(&up, 1, 0, UP), base * 16.0)); // against it: the floor too
        assert!(close(cost(&side, 0, 1, [1.0, 0.0, 0.0]), base)); // a path already heading sideways goes on freely
        let diag = [std::f64::consts::FRAC_1_SQRT_2, 0.0, std::f64::consts::FRAC_1_SQRT_2];
        assert!(close(cost(&up, 0, 1, diag), base * 2.0)); // 45 degrees off: 1 / cos^2
        // Without a bend the cost is the length term alone.
        assert!(close(EdgeCost { points: &side, power: 2.0, bend: None }.cost(0, 1, &UP), base));
        // The reference direction: up at a source, the path's with weight 1,
        // between the two otherwise.
        let chain: [Point; 3] = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]];
        let pred = [usize::MAX, 0, 1];
        let c = EdgeCost { points: &chain, power: 2.0, bend: Some(bend) };
        assert_eq!(c.direction(0, &pred), UP);
        assert!(close(c.direction(2, &pred)[0], 1.0));
        let half = EdgeCost { points: &chain, power: 2.0, bend: Some(Bend { path_weight: 0.5, ..bend }) }.direction(2, &pred);
        assert!(close(half[0], half[2]) && half[1] == 0.0);
        assert_eq!(EdgeCost { points: &chain, power: 2.0, bend: Some(Bend { path_weight: 0.0, ..bend }) }.direction(2, &pred), UP);
    }

    #[test]
    fn a_search_that_follows_its_path_runs_along_a_branch() {
        // A stem to z = 1 and a horizontal branch from its top. Against the
        // vertical alone every step along the branch is a sideways one;
        // a search that follows its path pays for the turn and then goes on
        // at the cost of going straight.
        let mut pts: Vec<Point> = (0..=10).map(|k| [0.0, 0.0, 0.1 * k as f64]).collect();
        pts.extend((1..=20).map(|k| [0.1 * k as f64, 0.0, 1.0]));
        let far = pts.len() - 1;
        let reach = |path_weight: f64| {
            let g = directed_knn_graph(&pts, 2, 0.15, 2.0, Some(Bend { power: 2.0, floor: 0.25, path_weight }));
            dijkstra(&g, &[0]).0[far]
        };
        let stem = 10.0 * 0.01;
        assert!((reach(0.0) - (stem + 20.0 * 0.01 * 16.0)).abs() < 1e-9);
        // the turn at the floor, one step at 45 degrees (1 / cos^2 = 2), then 18 straight on
        assert!((reach(1.0) - (stem + 0.01 * 16.0 + 0.01 * 2.0 + 18.0 * 0.01)).abs() < 1e-9);
    }

    #[test]
    fn empty_and_tiny_graphs() {
        let g = directed_knn_graph(&[], 6, 1.0, 6.0, Some(Bend::default()));
        assert_eq!(g.n(), 0);
        let one = [[0.0, 0.0, 0.0]];
        let g = directed_knn_graph(&one, 6, 1.0, 6.0, Some(Bend::default()));
        assert_eq!(g.n(), 1);
        assert_eq!(g.edges(0).count(), 0);
    }
}
