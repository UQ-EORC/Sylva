//! Skeleton extraction and QSM assembly.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use rayon::prelude::*;

use crate::cluster::{connected_components, dijkstra, knn_graph, Graph};
use crate::error::{Error, Result};
use crate::transform::{add, dot, scale, sub};
use crate::Point;

#[derive(Debug, Clone, PartialEq)]
pub struct Cylinder {
    pub start: Point,
    pub axis: Point,
    pub length: f64,
    pub radius: f64,
    pub parent: i64,
    pub branch_order: u32,
    pub branch_id: u32,
    pub n_points: usize,
}

impl Cylinder {
    pub fn end(&self) -> Point {
        add(&self.start, &scale(&self.axis, self.length))
    }

    pub fn volume(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius * self.length
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Qsm {
    pub cylinders: Vec<Cylinder>,
}

impl Qsm {
    pub fn len(&self) -> usize {
        self.cylinders.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cylinders.is_empty()
    }

    pub fn total_volume(&self) -> f64 {
        self.cylinders.iter().map(Cylinder::volume).sum()
    }

    pub fn stem_volume(&self) -> f64 {
        self.cylinders.iter().filter(|c| c.branch_order == 0).map(Cylinder::volume).sum()
    }

    pub fn branch_volume(&self) -> f64 {
        self.total_volume() - self.stem_volume()
    }

    pub fn total_length(&self) -> f64 {
        self.cylinders.iter().map(|c| c.length).sum()
    }

    pub fn max_branch_order(&self) -> u32 {
        self.cylinders.iter().map(|c| c.branch_order).max().unwrap_or(0)
    }

    /// DBH: diameter interpolated at 1.3 m above the base along the main
    /// stem (the order-0 chain from the lowest root), falling back to the
    /// nearest stem cylinder when 1.3 m is outside it.
    pub fn dbh(&self) -> f64 {
        if self.cylinders.is_empty() {
            return f64::NAN;
        }
        let base_z = self.cylinders.iter().filter(|c| c.parent < 0).map(|c| c.start[2]).fold(f64::INFINITY, f64::min);
        let target = base_z + 1.3;
        let stem: Vec<&Cylinder> = self.cylinders.iter().filter(|c| c.branch_order == 0).collect();
        // Cylinder spanning the target height.
        for c in &stem {
            let (z0, z1) = (c.start[2].min(c.end()[2]), c.start[2].max(c.end()[2]));
            if target >= z0 && target <= z1 {
                return 2.0 * c.radius;
            }
        }
        stem.iter()
            .min_by(|a, b| {
                let da = ((a.start[2] + a.end()[2]) / 2.0 - target).abs();
                let db = ((b.start[2] + b.end()[2]) / 2.0 - target).abs();
                da.partial_cmp(&db).unwrap()
            })
            .map(|c| 2.0 * c.radius)
            .unwrap_or(f64::NAN)
    }

    /// Triangle mesh of all cylinders (`sides` facets each, with end caps).
    /// Returns `(vertices, triangles as vertex indices, cylinder index per triangle)`.
    pub fn mesh(&self, sides: usize) -> (Vec<Point>, Vec<[u32; 3]>, Vec<u32>) {
        let sides = sides.max(3);
        let mut verts: Vec<Point> = Vec::with_capacity(self.cylinders.len() * (2 * sides + 2));
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(self.cylinders.len() * 4 * sides);
        let mut owner: Vec<u32> = Vec::with_capacity(self.cylinders.len() * 4 * sides);
        for (ci, c) in self.cylinders.iter().enumerate() {
            let axis = crate::transform::normalize(&c.axis);
            let helper = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let u = crate::transform::normalize(&crate::transform::cross(&axis, &helper));
            let v = crate::transform::cross(&axis, &u);
            let base = verts.len() as u32;
            let end = c.end();
            for ring in [c.start, end] {
                for k in 0..sides {
                    let a = k as f64 / sides as f64 * std::f64::consts::TAU;
                    let (sa, ca) = a.sin_cos();
                    verts.push([
                        ring[0] + c.radius * (ca * u[0] + sa * v[0]),
                        ring[1] + c.radius * (ca * u[1] + sa * v[1]),
                        ring[2] + c.radius * (ca * u[2] + sa * v[2]),
                    ]);
                }
            }
            verts.push(c.start);
            verts.push(end);
            let (c0, c1) = (base + 2 * sides as u32, base + 2 * sides as u32 + 1);
            for k in 0..sides as u32 {
                let k1 = (k + 1) % sides as u32;
                let (a0, a1, b0, b1) = (base + k, base + k1, base + sides as u32 + k, base + sides as u32 + k1);
                tris.push([a0, b0, a1]);
                tris.push([a1, b0, b1]);
                tris.push([c0, a1, a0]);
                tris.push([c1, b0, b1]);
                owner.extend([ci as u32; 4]);
            }
        }
        (verts, tris, owner)
    }

    /// `(n, 12)` row-major array: start(3), axis(3), length, radius, parent, order, branch, n_points.
    pub fn to_rows(&self) -> Vec<[f64; 12]> {
        self.cylinders
            .iter()
            .map(|c| [
                c.start[0], c.start[1], c.start[2], c.axis[0], c.axis[1], c.axis[2], c.length, c.radius,
                c.parent as f64, c.branch_order as f64, c.branch_id as f64, c.n_points as f64,
            ])
            .collect()
    }

    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(w, "sx,sy,sz,ax,ay,az,length,radius,parent,branch_order,branch_id,n_points")?;
        for r in self.to_rows() {
            let s: Vec<String> = r.iter().map(|v| format!("{v:.6}")).collect();
            writeln!(w, "{}", s.join(","))?;
        }
        Ok(())
    }

    /// Write in the raycloudtools `_trees.txt` format (one tree; segments are
    /// cylinder end points with `x,y,z,radius,parent_id`).
    pub fn write_treefile(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(w, "# Tree file written by sylva: x,y,z,radius,parent_id,section_id")?;
        writeln!(w, "x,y,z,radius,parent_id,section_id")?;
        let mut line = Vec::new();
        // Root node: start of the first cylinder without a parent.
        let root = self.cylinders.iter().position(|c| c.parent < 0).unwrap_or(0);
        if let Some(r) = self.cylinders.get(root) {
            line.push(format!("{},{},{},{},-1,0", r.start[0], r.start[1], r.start[2], r.radius));
        }
        for (i, c) in self.cylinders.iter().enumerate() {
            let e = c.end();
            let parent = if c.parent < 0 { 0 } else { c.parent + 1 };
            line.push(format!("{},{},{},{},{},{}", e[0], e[1], e[2], c.radius, parent, i + 1));
        }
        writeln!(w, "{}", line.join(", "))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct QsmParams {
    pub k: usize,
    pub max_edge: f64,
    pub bin_length: f64,
    pub min_points: usize,
    pub ransac_threshold: f64,
    /// No cylinder may be wider than this (set from the measured DBH).
    pub max_radius: f64,
    /// A child cylinder may be at most this times its parent's radius
    /// (monotonic taper); wider fits are clamped, not dropped.
    pub taper_limit: f64,
    /// Circle fits with RMSE above this are rejected (node left unmeasured).
    pub max_rmse: f64,
    /// Laplacian smoothing passes over skeleton node positions.
    pub smooth_steps: usize,
    /// Smallest radius a tip may have.
    pub apex_radius: f64,
    /// Longest contiguous arc (degrees) a cross-section circle must cover.
    pub min_arc_deg: f64,
    /// Fraction of a segment's points that must be circle inliers.
    pub min_inlier_fraction: f64,
    /// Unmeasured leaf segments with fewer points than this are pruned as foliage.
    pub prune_points: usize,
    /// Segments with fewer points are not fitted directly; their
    /// radius comes from the taper model.
    pub fit_min_points: usize,
    /// Leafy branch tips shorter than this (subtree length, m) are not
    /// reconstructed (raycloudtools `crop_length`).
    pub crop_length: f64,
    /// Below this height above the base only the largest connected component
    /// per bin is kept.
    pub butt_height: f64,
    /// Refit band around the RANSAC circle as a fraction of its radius.
    pub relative_tolerance: f64,
    /// Base (breast-height) radius anchoring the allometric taper prior; 0 =
    /// estimate from the strongest measurement.
    pub base_radius: f64,
    /// Weak measurements further than this fraction from the prior are replaced.
    pub allometry_tolerance: f64,
    /// Use an equivalent-area star-polygon radius where a circle explains
    /// fewer than `buttress_max_inlier_fraction` of a section's points
    /// (buttressed or fluted stems).
    pub buttress_equivalent_area: bool,
    pub buttress_max_inlier_fraction: f64,
    /// Children's summed cross-section area may exceed the parent's by this
    /// factor at a fork (Leonardo's rule); 0 disables.
    pub pipe_slack: f64,
    /// Branch (order >= 1) circle fits need at least this inlier fraction.
    pub branch_min_inlier_fraction: f64,
    /// Spatial radius for clustering points within a geodesic shell (a
    /// DBSCAN eps); 0 falls back to components of the geodesic graph.
    pub cluster_eps: f64,
    /// Clusters with at least this many points get a circle-fitted centre; 0 disables.
    pub centre_fit_points: usize,
    /// Taubin smoothing passes over radii along each axis.
    pub radius_smooth_steps: usize,
    /// Unmeasured nodes below the lowest accepted stem fit may exceed it by
    /// at most this factor (butt swell).
    pub butt_swell: f64,
    /// Straighten the butt: cut the stem below the first run of this many
    /// near-vertical cylinders and replace it by a vertical stump;
    /// 0 disables.
    pub butt_vertical_run: usize,
    /// Cylinders within this many degrees of vertical count as vertical.
    pub butt_max_lean_deg: f64,
    /// Skeleton chaining radius: the next node is the nearest unplaced centre
    /// of a higher shell within this distance.
    pub chain_max_d: f64,
    /// Sections with a circle radius at least this large and points in at
    /// least 300 deg of angular bins take the equivalent-area radius of a
    /// Fourier contour instead of the circle;
    /// 0 disables.
    pub fourier_min_radius: f64,
}

impl Default for QsmParams {
    fn default() -> Self {
        QsmParams { k: 15, max_edge: 1.0, bin_length: 0.1, min_points: 1, ransac_threshold: 0.02, max_radius: 1.0, taper_limit: 1.1, max_rmse: 0.03, smooth_steps: 10, apex_radius: 0.0025, min_arc_deg: 90.0, min_inlier_fraction: 0.05, prune_points: 5, fit_min_points: 50, crop_length: 0.0, butt_height: 0.6, relative_tolerance: 0.08, base_radius: 0.0, allometry_tolerance: 0.3, buttress_equivalent_area: true, buttress_max_inlier_fraction: 0.3, pipe_slack: 1.2, branch_min_inlier_fraction: 0.3, cluster_eps: 0.1, centre_fit_points: 100, radius_smooth_steps: 15, butt_swell: 1.1, butt_vertical_run: 4, butt_max_lean_deg: 50.0, chain_max_d: 0.1, fourier_min_radius: 0.15 }
    }
}

#[derive(Debug, Clone)]
pub struct Skeleton {
    /// Per-point segment id (-1 if disconnected from the base).
    pub segment_id: Vec<i64>,
    /// Per-point geodesic distance from the base.
    pub geodesic: Vec<f64>,
    /// Segment centroids.
    pub centres: Vec<Point>,
    /// `(child, parent)` links.
    pub edges: Vec<(usize, usize)>,
}

/// Connected components of `members` under a spatial radius `eps` (a
/// per-shell DBSCAN with minPts = 1), independent of the geodesic graph.
fn eps_components(xyz: &[Point], members: &[usize], eps: f64) -> Vec<usize> {
    let pts: Vec<Point> = members.iter().map(|&i| xyz[i]).collect();
    let tree = crate::spatial::KdTree::new(&pts);
    let mut label = vec![usize::MAX; pts.len()];
    let mut count = 0;
    let mut stack = Vec::new();
    for start in 0..pts.len() {
        if label[start] != usize::MAX {
            continue;
        }
        label[start] = count;
        stack.push(start);
        while let Some(i) = stack.pop() {
            for (j, _) in tree.within(&pts[i], eps) {
                if label[j] == usize::MAX {
                    label[j] = count;
                    stack.push(j);
                }
            }
        }
        count += 1;
    }
    label
}

#[allow(dead_code)]
fn subgraph_components(graph: &Graph, members: &[usize]) -> Vec<usize> {
    let pos: HashMap<usize, usize> = members.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); members.len()];
    for (k, &i) in members.iter().enumerate() {
        for (j, _) in graph.edges(i) {
            if let Some(&kj) = pos.get(&j) {
                adj[k].push(kj);
            }
        }
    }
    let mut label = vec![usize::MAX; members.len()];
    let mut count = 0;
    let mut stack = Vec::new();
    for s in 0..members.len() {
        if label[s] != usize::MAX {
            continue;
        }
        label[s] = count;
        stack.push(s);
        while let Some(i) = stack.pop() {
            for &j in &adj[i] {
                if label[j] == usize::MAX {
                    label[j] = count;
                    stack.push(j);
                }
            }
        }
        count += 1;
    }
    label
}

/// Graph-based skeleton of a single tree.
pub fn skeletonize(xyz: &[Point], base_xy: Option<[f64; 2]>, p: &QsmParams) -> Result<Skeleton> {
    if xyz.is_empty() {
        return Err(Error::invalid("empty cloud"));
    }
    let graph = knn_graph(xyz, p.k, p.max_edge);
    // The root is the lowest zone of the tree's main graph component; a
    // stray clump of ground remnants nearest the stem position must not
    // become the base of a 30-cylinder "tree".
    let (_, comp) = connected_components(&graph);
    let main = largest_component(&comp);
    let zmin = xyz.iter().zip(&comp).filter(|(_, &c)| c == main).map(|(q, _)| q[2]).fold(f64::INFINITY, f64::min);
    let base_xy = base_xy.unwrap_or_else(|| {
        let i = (0..xyz.len()).filter(|&i| comp[i] == main).min_by(|&a, &b| xyz[a][2].partial_cmp(&xyz[b][2]).unwrap()).unwrap();
        [xyz[i][0], xyz[i][1]]
    });
    let base = (0..xyz.len())
        .filter(|&i| comp[i] == main && xyz[i][2] <= zmin + 0.5)
        .min_by(|&a, &b| {
            let da = (xyz[a][0] - base_xy[0]).hypot(xyz[a][1] - base_xy[1]);
            let db = (xyz[b][0] - base_xy[0]).hypot(xyz[b][1] - base_xy[1]);
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    // Grow the shells from the whole base ring, not from one point: from a
    // single surface point the first half-circumference of geodesic shells
    // are skewed patches rather than cross-sections, which inflates the butt.
    let near = |i: usize| (xyz[i][0] - xyz[base][0]).hypot(xyz[i][1] - xyz[base][1]) <= 1.0;
    let z_low = (0..xyz.len()).filter(|&i| comp[i] == main && near(i)).map(|i| xyz[i][2]).fold(f64::INFINITY, f64::min);
    let sources: Vec<usize> = (0..xyz.len()).filter(|&i| comp[i] == main && near(i) && xyz[i][2] <= z_low + p.bin_length).collect();
    let (geod, _, pred) = dijkstra(&graph, &sources);
    let max_bin = geod.iter().filter(|d| d.is_finite()).map(|d| (d / p.bin_length).floor() as usize).max().unwrap_or(0);
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); max_bin + 1];
    for (i, d) in geod.iter().enumerate() {
        if d.is_finite() {
            bins[(d / p.bin_length).floor() as usize].push(i);
        }
    }
    let mut segment_id = vec![-1i64; xyz.len()];
    let mut next = 0i64;
    let butt_bins = ((p.butt_height / p.bin_length).ceil() as usize).max(1);
    for (b, members) in bins.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let comp = if p.cluster_eps > 0.0 { eps_components(xyz, members, p.cluster_eps) } else { subgraph_components(&graph, members) };
        let n = comp.iter().max().unwrap() + 1;
        if b < butt_bins && n > 1 {
            // Butt cleaning: near the ground keep only
            // the largest component so understorey and ground remnants do not
            // seed extra roots.
            let mut sizes = vec![0usize; n];
            for &c in &comp {
                sizes[c] += 1;
            }
            let biggest = (0..n).max_by_key(|&c| sizes[c]).unwrap();
            for (k, &i) in members.iter().enumerate() {
                if comp[k] == biggest {
                    segment_id[i] = next;
                }
            }
            next += 1;
            continue;
        }
        for (k, &i) in members.iter().enumerate() {
            segment_id[i] = next + comp[k] as i64;
        }
        next += n as i64;
    }
    let n_seg = next as usize;
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_seg];
    for (i, &s) in segment_id.iter().enumerate() {
        if s >= 0 {
            members[s as usize].push(i);
        }
    }
    // Node centre: the mean, or for well-populated clusters the centre of a
    // circle fitted in XY, which de-biases one-sided scans of stems.
    let centres: Vec<Point> = members
        .par_iter()
        .map(|m| {
            let n = m.len().max(1) as f64;
            let mut c = [0.0; 3];
            for &i in m {
                for k in 0..3 {
                    c[k] += xyz[i][k] / n;
                }
            }
            if p.centre_fit_points > 0 && m.len() >= p.centre_fit_points {
                let xy: Vec<[f64; 2]> = m.iter().map(|&i| [xyz[i][0], xyz[i][1]]).collect();
                let cp = crate::stems::StemParams { min_radius: p.apex_radius, max_radius: p.max_radius, ransac_tolerance: 0.04, ransac_iterations: 120, ..Default::default() };
                let mut rng = crate::filters::Rng::new(m.len() as u64);
                if let Some((cx, cy, r, mask)) = crate::stems::ransac_circle(&xy, &cp, &mut rng) {
                    let inl: Vec<[f64; 2]> = xy.iter().zip(&mask).filter(|(_, &k)| k).map(|(q, _)| *q).collect();
                    let (_, arc) = crate::stems::angular_coverage(&inl, cx, cy);
                    if arc >= 120.0 && inl.len() * 2 >= xy.len() && (cx - c[0]).hypot(cy - c[1]) <= r {
                        return [cx, cy, c[2]];
                    }
                }
            }
            c
        })
        .collect();
    // Parent links by greedy chain growing: from the current node take
    // the nearest unplaced centre of a higher shell within `chain_max_d`;
    // when stuck, the unplaced centre of the lowest shell attaches to its
    // nearest placed centre and the chain continues from there. Following
    // the closest centre keeps the trunk on its core through buttress zones
    // instead of hopping between arms of the same shell.
    let seg_bin: Vec<i64> = (0..n_seg)
        .map(|s| members[s].iter().map(|&i| (geod[i] / p.bin_length).floor() as i64).min().unwrap_or(0))
        .collect();
    let _ = &pred;
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n_seg);
    if n_seg > 0 {
        let tree = crate::spatial::KdTree::new(&centres);
        let mut done = vec![false; n_seg];
        let mut cur = (0..n_seg).min_by(|&a, &b| centres[a][2].partial_cmp(&centres[b][2]).unwrap()).unwrap();
        done[cur] = true;
        let mut order: Vec<usize> = (0..n_seg).collect();
        order.sort_by_key(|&s| seg_bin[s]);
        let mut next_orphan = 0usize;
        let mut placed: Vec<usize> = vec![cur];
        let max_d = p.chain_max_d.max(p.bin_length);
        let mut remaining = n_seg - 1;
        while remaining > 0 {
            let hits = tree.within(&centres[cur], max_d);
            let best = hits
                .iter()
                .filter(|&&(q, _)| !done[q] && seg_bin[q] > seg_bin[cur])
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|&(q, _)| q);
            let (child, parent) = match best {
                Some(q) => (q, cur),
                None => {
                    while next_orphan < n_seg && done[order[next_orphan]] {
                        next_orphan += 1;
                    }
                    if next_orphan >= n_seg {
                        break;
                    }
                    let o = order[next_orphan];
                    let near = placed
                        .par_iter()
                        .map(|&q| (q, crate::transform::norm(&sub(&centres[q], &centres[o]))))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(q, _)| q)
                        .unwrap_or(cur);
                    (o, near)
                }
            };
            done[child] = true;
            placed.push(child);
            remaining -= 1;
            edges.push((child, parent));
            cur = child;
        }
    }
    Ok(Skeleton { segment_id, geodesic: geod, centres, edges })
}

/// Order / branch labels and root-to-tip chains: at every node the child with
/// the largest `key` continues its parent's axis, the rest start new axes.
fn assign_axes(n_seg: usize, topo: &[usize], roots: &[usize], children: &[Vec<usize>], key: &[(f64, f64)]) -> (Vec<u32>, Vec<u32>, Vec<Vec<usize>>, Vec<usize>) {
    let mut order = vec![0u32; n_seg];
    let mut branch = vec![0u32; n_seg];
    let mut next_branch = 1u32;
    let mut chains: Vec<Vec<usize>> = Vec::new();
    let mut chain_of = vec![usize::MAX; n_seg];
    for &root in roots {
        chains.push(vec![root]);
        chain_of[root] = chains.len() - 1;
    }
    for &s in topo {
        let mut kids = children[s].clone();
        kids.sort_by(|&a, &b| key[b].partial_cmp(&key[a]).unwrap_or(std::cmp::Ordering::Equal));
        for (i, c) in kids.into_iter().enumerate() {
            if i == 0 {
                order[c] = order[s];
                branch[c] = branch[s];
                let ch = chain_of[s];
                chains[ch].push(c);
                chain_of[c] = ch;
            } else {
                order[c] = order[s] + 1;
                branch[c] = next_branch;
                next_branch += 1;
                chains.push(vec![c]);
                chain_of[c] = chains.len() - 1;
            }
        }
    }
    (order, branch, chains, chain_of)
}

/// Assemble a cylinder model from a skeleton.
///
/// 1. Segment centres are smoothed along parent-child chains
///    (`smooth_steps` passes), and each node's axis is the skeleton direction
///    through it -- not a per-segment cylinder fit, which wobbles on short
///    sections of thick stems.
/// 2. The radius at each node is a RANSAC circle fitted to the segment's
///    points projected onto the plane perpendicular to that axis, accepted
///    only with enough inliers, a contiguous arc and a small residual;
///    otherwise the node is unmeasured.
/// 3. Along every root-to-tip path radii are made non-increasing by weighted
///    isotonic regression (pool-adjacent-violators, weights = inlier counts),
///    unmeasured nodes are interpolated, tips get `apex_radius`, and a child
///    branch may not be wider than its parent.
/// 4. Cylinders join consecutive nodes; unsupported leaf segments (no
///    accepted radius, few points) are pruned as foliage.
pub fn fit_cylinders(xyz: &[Point], skel: &Skeleton, p: &QsmParams) -> Result<Qsm> {
    let n_seg = skel.centres.len();
    if n_seg == 0 {
        return Ok(Qsm::default());
    }
    let parent_of: HashMap<usize, usize> = skel.edges.iter().cloned().collect();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n_seg];
    for &(c, par) in &skel.edges {
        children[par].push(c);
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_seg];
    for (i, &s) in skel.segment_id.iter().enumerate() {
        if s >= 0 {
            members[s as usize].push(i);
        }
    }
    // Topological order (roots first).
    let roots: Vec<usize> = (0..n_seg).filter(|s| !parent_of.contains_key(s)).collect();
    let mut topo: Vec<usize> = Vec::with_capacity(n_seg);
    let mut stack = roots.clone();
    while let Some(s) = stack.pop() {
        topo.push(s);
        stack.extend(children[s].iter().copied());
    }

    // 1. Taubin-smooth centres along chains (lambda 0.5, mu -0.53),
    // with roots and tips pinned so the chain does not shrink.
    let pinned: Vec<bool> = (0..n_seg).map(|s| !parent_of.contains_key(&s) || children[s].is_empty()).collect();
    let mut centre: Vec<Point> = skel.centres.clone();
    let laplacian = |c: &[Point], s: usize| -> Point {
        let mut acc = [0.0; 3];
        let mut n = 0.0;
        if let Some(&par) = parent_of.get(&s) {
            acc = add(&acc, &c[par]);
            n += 1.0;
        }
        for &ch in &children[s] {
            acc = add(&acc, &c[ch]);
            n += 1.0;
        }
        if n == 0.0 {
            [0.0; 3]
        } else {
            sub(&scale(&acc, 1.0 / n), &c[s])
        }
    };
    for _ in 0..p.smooth_steps {
        for factor in [0.5, -0.53] {
            let prev = centre.clone();
            for s in 0..n_seg {
                if !pinned[s] {
                    centre[s] = add(&prev[s], &scale(&laplacian(&prev, s), factor));
                }
            }
        }
    }
    // Axis: direction from parent centre to the mean of the children (or to self).
    let axis: Vec<Point> = (0..n_seg)
        .map(|s| {
            let from = parent_of.get(&s).map(|&par| centre[par]).unwrap_or(centre[s]);
            let to = if children[s].is_empty() {
                centre[s]
            } else {
                let mut cm = [0.0; 3];
                for &c in &children[s] {
                    cm = add(&cm, &centre[c]);
                }
                scale(&cm, 1.0 / children[s].len() as f64)
            };
            let d = sub(&to, &from);
            if crate::transform::norm(&d) < 1e-9 {
                [0.0, 0.0, 1.0]
            } else {
                crate::transform::normalize(&d)
            }
        })
        .collect();

    // 2. Radius per node from a perpendicular circle fit.
    let circle_params = crate::stems::StemParams {
        min_radius: p.apex_radius,
        max_radius: p.max_radius,
        ransac_iterations: 120,
        ransac_tolerance: p.ransac_threshold,
        ..Default::default()
    };
    let fits: Vec<Option<(f64, usize, f64, f64)>> = (0..n_seg)
        .into_par_iter()
        .map(|s| {
            if members[s].len() < p.fit_min_points {
                return None;
            }
            let (u, v) = perp_basis(&axis[s]);
            let xy: Vec<[f64; 2]> = members[s]
                .iter()
                .map(|&i| {
                    let d = sub(&xyz[i], &centre[s]);
                    [dot(&d, &u), dot(&d, &v)]
                })
                .collect();
            let mut rng = crate::filters::Rng::new(s as u64 + 1);
            let (cx0, cy0, r0, _) = crate::stems::ransac_circle(&xy, &circle_params, &mut rng)?;
            // Rough bark on thick stems spans more than the RANSAC band, and a
            // tight band then favours a sub-arc and a small radius. Refit on
            // every point within a radius-relative band (>= 8 % of r).
            let band = p.ransac_threshold.max(p.relative_tolerance * r0);
            let inl: Vec<[f64; 2]> = xy.iter().filter(|q| ((q[0] - cx0).hypot(q[1] - cy0) - r0).abs() <= band).cloned().collect();
            let (cx, cy, r) = crate::stems::fit_circle_refined(&inl).unwrap_or((cx0, cy0, r0));
            let inl: Vec<[f64; 2]> = xy.iter().filter(|q| ((q[0] - cx).hypot(q[1] - cy) - r).abs() <= band).cloned().collect();
            if inl.len() < p.min_points {
                return None;
            }
            let rmse = (inl.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).powi(2)).sum::<f64>() / inl.len() as f64).sqrt();
            let (_, arc) = crate::stems::angular_coverage(&inl, cx, cy);
            let frac = inl.len() as f64 / xy.len() as f64;
            let ok = arc >= p.min_arc_deg && rmse <= p.max_rmse.max(band) && frac >= p.min_inlier_fraction;
            if ok && frac >= p.buttress_max_inlier_fraction {
                // Thick sections seen all round: fit a
                // Fourier contour and take its equivalent-area radius, which
                // is what the volume needs on fluted stems where the circle
                // through the ridges overstates the section.
                if p.fourier_min_radius > 0.0 && r >= p.fourier_min_radius {
                    if let Some(req) = fourier_area_radius(&xy, cx, cy, BINS * 5 / 6) {
                        if req >= p.apex_radius && req <= p.max_radius {
                            return Some((req, inl.len(), arc, frac));
                        }
                    }
                }
                return Some((r, inl.len(), arc, frac));
            }
            // Buttress / fluted section: a circle explains few of the points.
            // Use the equivalent-area radius of a Fourier contour;
            // reported as a weak measurement.
            if p.buttress_equivalent_area && xy.len() >= p.fit_min_points {
                let n = xy.len() as f64;
                let (mx, my) = (xy.iter().map(|q| q[0]).sum::<f64>() / n, xy.iter().map(|q| q[1]).sum::<f64>() / n);
                if let Some(req) = fourier_area_radius(&xy, mx, my, BINS * 5 / 6) {
                    if req >= p.apex_radius && req <= p.max_radius {
                        return Some((req, xy.len().min(p.fit_min_points), 0.0, frac.max(1e-3)));
                    }
                }
            }
            ok.then_some((r, inl.len(), arc, frac))
        })
        .collect();

    // 3. Regularise radii along root-to-tip paths.
    //
    // Subtree length (longest path to a tip) drives an allometric prior
    // r(w) = apex + (w / w0)^1.1 (r0 - apex) anchored on the base radius
    // (a conic allometry). Weak measurements (short arc or few inliers)
    // more than 30 % off the prior are replaced by it; chains are then made
    // non-increasing by weighted isotonic regression, gaps interpolated and
    // unmeasured chains take the prior.
    let mut sub_len = vec![0.0f64; n_seg];
    for &s in topo.iter().rev() {
        let mut best = 0.0f64;
        for &c in &children[s] {
            best = best.max(sub_len[c] + crate::transform::norm(&sub(&centre[c], &centre[s])));
        }
        sub_len[s] = best;
    }
    let w0 = roots.iter().map(|&r| sub_len[r]).fold(0.0, f64::max).max(1e-6);
    // Fit the prior's base radius to the decent measurements (weighted median
    // of r_i / (w_i / w0)^1.1) rather than trusting one
    // diameter; fall back to the given base radius (measured DBH) when there are
    // too few, so a wrong DBH cannot inflate a whole stem.
    let mut est: Vec<(f64, f64)> = (0..n_seg)
        .filter_map(|s| {
            let (r, n, arc, frac) = fits[s]?;
            let scale = (sub_len[s] / w0).clamp(1e-3, 1.0).powf(1.1);
            (arc >= 180.0 && frac >= 0.5 && sub_len[s] > 0.2 * w0).then_some(((r - p.apex_radius) / scale + p.apex_radius, n as f64))
        })
        .collect();
    let r0 = if est.len() >= 5 {
        est.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let total: f64 = est.iter().map(|e| e.1).sum();
        let mut acc = 0.0;
        let mut med = est[est.len() / 2].0;
        for (r, w) in &est {
            acc += w;
            if acc >= total / 2.0 {
                med = *r;
                break;
            }
        }
        med.clamp(p.apex_radius * 2.0, p.max_radius)
    } else if p.base_radius > 0.0 {
        p.base_radius
    } else {
        fits.iter().filter_map(|f| f.map(|(r, ..)| r)).fold(0.0, f64::max).max(p.apex_radius * 4.0)
    };
    let allo = |s: usize| -> f64 { p.apex_radius + (sub_len[s] / w0).clamp(0.0, 1.0).powf(1.1) * (r0 - p.apex_radius) };
    // Axes: at each node the child with the
    // highest subtree top continues the axis (ties: longest subtree); the
    // others start branches one order higher. Once radii exist the axes are
    // reassigned by subtree volume.
    let mut sub_top = vec![f64::NEG_INFINITY; n_seg];
    for &s in topo.iter().rev() {
        sub_top[s] = centre[s][2];
        for &c in &children[s] {
            sub_top[s] = sub_top[s].max(sub_top[c]);
        }
    }
    let key_top: Vec<(f64, f64)> = (0..n_seg).map(|s| (sub_top[s], sub_len[s])).collect();
    let (mut order, _, mut chains, _) = assign_axes(n_seg, &topo, &roots, &children, &key_top);
    // Weak branch fits: a circle that explains under 30 % of a crown
    // segment's points is usually foliage or several twigs, not a branch.
    let mut fits = fits;
    for s in 0..n_seg {
        if order[s] >= 1 {
            if let Some((_, _, arc, frac)) = fits[s] {
                if frac < p.branch_min_inlier_fraction || arc <= 0.0 {
                    fits[s] = None;
                }
            }
        }
    }
    // Main stem (order 0). The prior inside the measured range is a
    // polynomial: a quadratic in subtree length through the apex,
    // r = t + a s + b s^2, least-squares fitted to the stem's own accepted
    // circles above breast height and kept only when it does not shrink
    // toward the base; otherwise the tree allometry. Weak fits far from the
    // prior are replaced, gaps interpolated (non-increasing), and nodes above
    // the last measurement are left to the pipe model below, like every
    // unmeasured branch.
    let mut prior = vec![f64::NAN; n_seg];
    for &s in &topo {
        if order[s] == 0 {
            prior[s] = match parent_of.get(&s) {
                None => allo(s),
                Some(&par) => allo(s).min(prior[par]),
            };
        }
    }
    let mut radius: Vec<f64> = vec![f64::NAN; n_seg];
    let mut weight: Vec<f64> = vec![0.0; n_seg];
    let accept = |s: usize, est: f64, radius: &mut [f64], weight: &mut [f64]| {
        if let Some((r, n, arc, frac)) = fits[s] {
            let strong = arc >= 300.0 && frac >= 0.7;
            if !strong && (r - est).abs() > p.allometry_tolerance * est {
                radius[s] = est;
                weight[s] = 1.0;
            } else {
                radius[s] = r;
                weight[s] = n as f64 * if strong { 2.0 } else { 1.0 };
            }
        }
    };
    for chain in chains.iter().filter(|c| order[c[0]] == 0) {
        let mut path = 0.0;
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for (k, &q) in chain.iter().enumerate() {
            if k > 0 {
                path += crate::transform::norm(&sub(&centre[q], &centre[chain[k - 1]]));
            }
            if let Some((r, ..)) = fits[q] {
                if path > 1.3 {
                    pts.push((sub_len[q], r - p.apex_radius));
                }
            }
        }
        if pts.len() >= 7 {
            let (mut x2, mut x3, mut x4, mut xy, mut x2y) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
            for &(x, y) in &pts {
                x2 += x * x;
                x3 += x * x * x;
                x4 += x * x * x * x;
                xy += x * y;
                x2y += x * x * y;
                x_min = x_min.min(x);
                x_max = x_max.max(x);
            }
            let det = x2 * x4 - x3 * x3;
            if det.abs() > 1e-12 {
                let a = (xy * x4 - x2y * x3) / det;
                let b = (x2 * x2y - xy * x3) / det;
                if a + 2.0 * b * x_min >= -1e-9 && a + 2.0 * b * x_max >= -1e-9 {
                    for &q in chain {
                        let x = sub_len[q];
                        if x >= x_min {
                            prior[q] = (p.apex_radius + a * x + b * x * x).clamp(p.apex_radius, p.max_radius);
                        }
                    }
                }
            }
        }
        let Some(k_last) = chain.iter().rposition(|&q| fits[q].is_some()) else { continue };
        let measured = &chain[..=k_last];
        for &q in measured {
            accept(q, prior[q], &mut radius, &mut weight);
        }
        isotonic_fill(measured, &mut radius, &weight, p.apex_radius);
        for &q in measured {
            if !radius[q].is_finite() {
                radius[q] = prior[q];
            }
        }
    }
    // Branches: pipe-model reconstruction, top-down. Each node is
    // anchored on its parent's final radius r_a and subtree length s_a: on a
    // straight run the radius tapers linearly with subtree length,
    // r = t + (r_a - t) s / s_a; at a fork (several unmeasured children, or
    // any measured one) the parent's cross-section is shared among all
    // children by subtree length, r_i = r_a s_i / sqrt(sum s_j^2), so area is
    // conserved through every fork rather than created at each side branch.
    // Measured branch nodes keep their circle unless it is weak and far off.
    for &s in &topo {
        if radius[s].is_finite() {
            continue;
        }
        let Some(&par) = parent_of.get(&s) else {
            radius[s] = if prior[s].is_finite() { prior[s] } else { p.apex_radius };
            continue;
        };
        let r_a = radius[par];
        let s_a = sub_len[par];
        let sibs = &children[par];
        let measured = sibs.iter().filter(|&&c| fits[c].is_some()).count();
        let is_fork = sibs.len() - measured > 1 || measured > 0;
        let est = if !r_a.is_finite() || s_a <= 0.0 {
            p.apex_radius
        } else if !is_fork {
            p.apex_radius + (r_a - p.apex_radius) * (sub_len[s] / s_a).clamp(0.0, 1.0)
        } else {
            let ss: f64 = sibs.iter().map(|&c| sub_len[c] * sub_len[c]).sum();
            if ss > 0.0 { r_a * sub_len[s] / ss.sqrt() } else { p.apex_radius }
        };
        let est = est.clamp(p.apex_radius, r_a.max(p.apex_radius));
        prior[s] = est;
        accept(s, est, &mut radius, &mut weight);
        if !radius[s].is_finite() {
            radius[s] = est;
            weight[s] = 0.5;
        }
    }
    for chain in chains.iter().filter(|c| order[c[0]] > 0) {
        isotonic_fill(chain, &mut radius, &weight, p.apex_radius);
    }
    for chain in chains.iter().filter(|c| order[c[0]] == 0) {
        // Keep the stem non-increasing across the measured / pipe-model join.
        for k in 1..chain.len() {
            radius[chain[k]] = radius[chain[k]].min(radius[chain[k - 1]]);
        }
    }
    // Below the lowest accepted stem measurement the allometry would keep
    // growing into the butt; a real butt rarely
    // exceeds the trunk above by much, so cap that zone at `butt_swell` x it.
    // Applied to every node below the lowest *strong* fit, measured or not:
    // weak circles and star-polygon areas on fluted sections both overshoot.
    for chain in chains.iter().filter(|c| order[c[0]] == 0) {
        let strong = |q: usize| matches!(fits[q], Some((_, _, arc, frac)) if arc >= 180.0 && frac >= 0.5);
        let Some(k0) = chain.iter().position(|&q| strong(q)) else { continue };
        let cap = radius[chain[k0]] * p.butt_swell;
        for &q in &chain[..k0] {
            // Predict the butt from the stem prior (the quadratic through
            // the measurements above breast height when there is one),
            // never below the lowest strong fit nor above the swell cap.
            let pred = if prior[q].is_finite() { prior[q] } else { radius[q] };
            radius[q] = pred.clamp(radius[chain[k0]], cap);
        }
    }
    for &s in &topo {
        if let Some(&par) = parent_of.get(&s) {
            if !(radius[s] <= p.taper_limit * radius[par]) {
                radius[s] = radius[par];
            }
        }
    }
    // Taubin-smooth radii along each axis (lambda 0.5,
    // mu -0.7), ends pinned, floored at half the chain minimum.
    for chain in &chains {
        if chain.len() < 3 || p.radius_smooth_steps == 0 {
            continue;
        }
        let floor = chain.iter().map(|&q| radius[q]).fold(f64::INFINITY, f64::min) * 0.5;
        for _ in 0..p.radius_smooth_steps {
            for factor in [0.5, -0.7] {
                let prev: Vec<f64> = chain.iter().map(|&q| radius[q]).collect();
                for k in 1..chain.len() - 1 {
                    let lap = (prev[k - 1] + prev[k + 1]) / 2.0 - prev[k];
                    radius[chain[k]] = (prev[k] + factor * lap).max(floor);
                }
            }
        }
    }

    // Leonardo's rule at forks (as in raycloudtools): the
    // children's cross-section area may not exceed the parent's by more
    // than `pipe_slack`; unmeasured children are scaled down first, then all.
    if p.pipe_slack > 0.0 {
        for &s in &topo {
            let kids: Vec<usize> = children[s].iter().copied().filter(|&c| radius[c].is_finite()).collect();
            if kids.len() < 2 {
                continue;
            }
            let parent_area = radius[s] * radius[s];
            let total: f64 = kids.iter().map(|&c| radius[c] * radius[c]).sum();
            if total <= p.pipe_slack * parent_area {
                continue;
            }
            let measured: Vec<usize> = kids.iter().copied().filter(|&c| fits[c].is_some()).collect();
            let unmeasured: Vec<usize> = kids.iter().copied().filter(|&c| fits[c].is_none()).collect();
            let m_area: f64 = measured.iter().map(|&c| radius[c] * radius[c]).sum();
            let u_area: f64 = unmeasured.iter().map(|&c| radius[c] * radius[c]).sum();
            let budget = (p.pipe_slack * parent_area - m_area).max(0.0);
            if u_area > budget && u_area > 0.0 {
                let f = (budget / u_area).sqrt();
                for &c in &unmeasured {
                    radius[c] = (radius[c] * f).max(p.apex_radius);
                }
            }
            let total: f64 = kids.iter().map(|&c| radius[c] * radius[c]).sum();
            if total > p.pipe_slack * parent_area {
                let f = (p.pipe_slack * parent_area / total).sqrt();
                for &c in &kids {
                    radius[c] = (radius[c] * f).max(p.apex_radius);
                }
            }
        }
    }
    for r in &mut radius {
        if !r.is_finite() {
            *r = p.apex_radius;
        }
        *r = r.clamp(p.apex_radius, p.max_radius);
    }
    let branch: Vec<u32>;
    // Reassign axes by subtree volume now that radii exist:
    // with co-dominant limbs the woodier one is the
    // stem, not the one that happens to reach higher.
    {
        let mut sub_vol = vec![0.0f64; n_seg];
        for &s in topo.iter().rev() {
            let len = parent_of.get(&s).map(|&q| crate::transform::norm(&sub(&centre[s], &centre[q]))).unwrap_or(p.bin_length);
            sub_vol[s] += radius[s] * radius[s] * len;
            if let Some(&q) = parent_of.get(&s) {
                sub_vol[q] += sub_vol[s];
            }
        }
        let key_vol: Vec<(f64, f64)> = (0..n_seg).map(|s| (sub_vol[s], sub_len[s])).collect();
        let (o2, b2, c2, _) = assign_axes(n_seg, &topo, &roots, &children, &key_vol);
        order = o2;
        branch = b2;
        chains = c2;
    }
    let _ = &chains;

    // 4. Prune unsupported foliage: leaf segments without an accepted fit and
    // few points, repeated until stable.
    let mut alive = vec![true; n_seg];
    loop {
        let mut changed = false;
        for s in 0..n_seg {
            if !alive[s] || fits[s].is_some() || parent_of.get(&s).is_none() {
                continue;
            }
            let has_live_child = children[s].iter().any(|&c| alive[c]);
            if !has_live_child && members[s].len() < p.prune_points {
                alive[s] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Crop leafy tips: remove segments whose remaining subtree (in bins) is
    // shorter than crop_length, unless they are on a root chain's start.
    if p.crop_length > 0.0 {
        let mut sub_len = vec![0.0f64; n_seg];
        for &s in topo.iter().rev() {
            let mut best = 0.0f64;
            for &c in &children[s] {
                if alive[c] {
                    best = best.max(sub_len[c] + crate::transform::norm(&sub(&centre[c], &centre[s])));
                }
            }
            sub_len[s] = best;
        }
        for &s in &topo {
            if alive[s] && parent_of.contains_key(&s) && sub_len[s] < p.crop_length && fits[s].is_none() {
                alive[s] = false;
            }
        }
    }

    // Cylinders from parent centre to own centre (roots: half a bin below).
    let mut cyls: Vec<Option<Cylinder>> = vec![None; n_seg];
    for &s in &topo {
        if !alive[s] {
            continue;
        }
        let start = match parent_of.get(&s) {
            Some(&par) => centre[par],
            None => sub(&centre[s], &scale(&axis[s], p.bin_length * 0.5)),
        };
        let d = sub(&centre[s], &start);
        let length = crate::transform::norm(&d);
        if length < 1e-6 {
            continue;
        }
        cyls[s] = Some(Cylinder {
            start,
            axis: scale(&d, 1.0 / length),
            length,
            radius: radius[s],
            parent: parent_of.get(&s).map(|&q| q as i64).unwrap_or(-1),
            branch_order: order[s],
            branch_id: branch[s],
            n_points: fits[s].map(|f| f.1).unwrap_or(0),
        });
    }
    // Nearest live ancestor as parent, then compact.
    let keep: Vec<usize> = (0..n_seg).filter(|&s| cyls[s].is_some()).collect();
    let remap: HashMap<usize, i64> = keep.iter().enumerate().map(|(new, &old)| (old, new as i64)).collect();
    let cylinders = keep
        .iter()
        .map(|&s| {
            let mut c = cyls[s].clone().unwrap();
            let mut a = parent_of.get(&s).copied();
            while let Some(q) = a {
                if cyls[q].is_some() {
                    break;
                }
                a = parent_of.get(&q).copied();
            }
            c.parent = a.and_then(|q| remap.get(&q).copied()).unwrap_or(-1);
            c
        })
        .collect();
    let mut qsm = Qsm { cylinders };
    if p.butt_vertical_run > 0 {
        straighten_butt(&mut qsm, p);
    }
    Ok(qsm)
}

/// Butt straightening: where the main stem starts by
/// wandering through buttress arms, cut it below the first run of
/// `butt_vertical_run` cylinders within `butt_max_lean_deg` of vertical and
/// replace that part by a straight vertical stump down to the base height,
/// flaring 1 % per shell. Branches that hung off the removed part reattach
/// to the stump top.
fn straighten_butt(qsm: &mut Qsm, p: &QsmParams) {
    let cyl = &qsm.cylinders;
    let Some(root) = cyl.iter().position(|c| c.parent < 0 && c.branch_order == 0) else { return };
    // Prolong down to the trunk's own root, not to the lowest cylinder of
    // the tree: a stray low branch must not pull the stump below the base.
    let base_z = cyl[root].start[2].min(cyl[root].end()[2]);
    let stem_id = cyl[root].branch_id;
    let mut chain = vec![root];
    while chain.len() <= cyl.len() {
        let last = *chain.last().unwrap() as i64;
        match cyl.iter().position(|c| c.parent == last && c.branch_id == stem_id) {
            Some(i) => chain.push(i),
            None => break,
        }
    }
    let cos_max = p.butt_max_lean_deg.to_radians().cos();
    let run = p.butt_vertical_run;
    if chain.len() < run {
        return;
    }
    let Some(k0) = (0..=chain.len() - run).find(|&k| (0..run).all(|j| cyl[chain[k + j]].axis[2].abs() >= cos_max)) else { return };
    if k0 == 0 {
        return;
    }
    let removed: std::collections::HashSet<usize> = chain[..k0].iter().copied().collect();
    let keep_first = chain[k0];
    let top = cyl[keep_first].start;
    let r_top = cyl[keep_first].radius;
    let height = (top[2] - base_z).max(0.0);
    let n_stump = ((height / p.bin_length).ceil() as usize).max(1);
    let mut new: Vec<Cylinder> = (0..n_stump)
        .map(|i| {
            let z0 = base_z + i as f64 * height / n_stump as f64;
            let z1 = base_z + (i + 1) as f64 * height / n_stump as f64;
            Cylinder {
                start: [top[0], top[1], z0],
                axis: [0.0, 0.0, 1.0],
                length: z1 - z0,
                radius: (r_top * 1.01f64.powi((n_stump - i) as i32)).min(p.max_radius),
                parent: i as i64 - 1,
                branch_order: 0,
                branch_id: stem_id,
                n_points: 0,
            }
        })
        .collect();
    let mut remap: HashMap<usize, i64> = HashMap::new();
    for (i, c) in cyl.iter().enumerate() {
        if !removed.contains(&i) {
            remap.insert(i, new.len() as i64);
            new.push(c.clone());
        }
    }
    let stump_top = n_stump as i64 - 1;
    for c in new.iter_mut().skip(n_stump) {
        c.parent = match c.parent {
            q if q < 0 => stump_top,
            q if removed.contains(&(q as usize)) => stump_top,
            q => remap.get(&(q as usize)).copied().unwrap_or(stump_top),
        };
    }
    qsm.cylinders = new;
}

const BINS: usize = 72;

/// Fourier cross-section: polar radii about `(cx, cy)`, binned
/// (median per 5 deg bin, empty bins filled with the overall median once at
/// least `min_filled` bins have points), a 10-harmonic Fourier series
/// through them, and the equivalent-area radius of that contour
/// (area = 1/2 int r^2 dtheta = pi (c0^2 + 1/2 sum a_k^2 + b_k^2)). Medians
/// rather than the outermost point per bin keep arm tips and noise from
/// inflating the section.
fn fourier_area_radius(xy: &[[f64; 2]], cx: f64, cy: f64, min_filled: usize) -> Option<f64> {
    const HARMONICS: usize = 10;
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); BINS];
    for q in xy {
        let a = (q[1] - cy).atan2(q[0] - cx);
        let b = (((a + std::f64::consts::PI) / std::f64::consts::TAU) * BINS as f64) as usize % BINS;
        bins[b].push((q[0] - cx).hypot(q[1] - cy));
    }
    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let mut rr: Vec<f64> = bins.iter_mut().map(|v| if v.is_empty() { f64::NAN } else { median(v) }).collect();
    if rr.iter().filter(|v| v.is_finite()).count() < min_filled {
        return None;
    }
    let mut all: Vec<f64> = rr.iter().copied().filter(|v| v.is_finite()).collect();
    let fill = median(&mut all);
    for v in rr.iter_mut() {
        if !v.is_finite() {
            *v = fill;
        }
    }
    let n = BINS as f64;
    let c0 = rr.iter().sum::<f64>() / n;
    let mut power = c0 * c0;
    for k in 1..=HARMONICS {
        let (mut ak, mut bk) = (0.0, 0.0);
        for (i, &v) in rr.iter().enumerate() {
            let th = k as f64 * std::f64::consts::TAU * i as f64 / n;
            ak += v * th.cos();
            bk += v * th.sin();
        }
        ak *= 2.0 / n;
        bk *= 2.0 / n;
        power += 0.5 * (ak * ak + bk * bk);
    }
    Some(power.sqrt())
}

/// Label of the component with the most members.
pub(crate) fn largest_component(comp: &[usize]) -> usize {
    let mut size: Vec<usize> = Vec::new();
    for &c in comp {
        if c >= size.len() {
            size.resize(c + 1, 0);
        }
        size[c] += 1;
    }
    (0..size.len()).max_by_key(|&c| size[c]).unwrap_or(0)
}

fn perp_basis(axis: &Point) -> (Point, Point) {
    let helper = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = crate::transform::normalize(&crate::transform::cross(axis, &helper));
    let v = crate::transform::cross(axis, &u);
    (u, v)
}

/// Non-increasing weighted isotonic regression along `chain` (base first);
/// NaN radii are interpolated between measured neighbours and the tip end
/// falls to `apex_radius` when nothing is measured beyond it.
fn isotonic_fill(chain: &[usize], radius: &mut [f64], weight: &[f64], apex_radius: f64) {
    let measured: Vec<(usize, f64, f64)> = chain
        .iter()
        .enumerate()
        .filter(|(_, &s)| radius[s].is_finite() && weight[s] > 0.0)
        .map(|(k, &s)| (k, radius[s], weight[s]))
        .collect();
    if measured.is_empty() {
        return; // filled from the parent later by the taper pass
    }
    // Pool-adjacent-violators for a non-increasing sequence.
    let mut blocks: Vec<(f64, f64, usize, usize)> = Vec::new(); // (weighted mean, weight, k_start, k_end)
    for &(k, r, w) in &measured {
        blocks.push((r, w, k, k));
        while blocks.len() >= 2 {
            let n = blocks.len();
            let (r1, w1, k1, _) = blocks[n - 2];
            let (r2, w2, _, k2b) = blocks[n - 1];
            if r2 <= r1 {
                break;
            }
            let w = w1 + w2;
            let r = (r1 * w1 + r2 * w2) / w;
            blocks.truncate(n - 2);
            blocks.push((r, w, k1, k2b));
        }
    }
    let mut fitted = vec![f64::NAN; chain.len()];
    for (r, _, ka, kb) in &blocks {
        for k in *ka..=*kb {
            if weight[chain[k]] > 0.0 {
                fitted[k] = *r;
            }
        }
    }
    // Interpolate gaps; extrapolate flat at the base, towards apex at the tip.
    let first = fitted.iter().position(|v| v.is_finite()).unwrap();
    let last = fitted.iter().rposition(|v| v.is_finite()).unwrap();
    for k in 0..first {
        fitted[k] = fitted[first];
    }
    let mut k = first;
    while k < last {
        let mut j = k + 1;
        while !fitted[j].is_finite() {
            j += 1;
        }
        for m in k + 1..j {
            let t = (m - k) as f64 / (j - k) as f64;
            fitted[m] = fitted[k] * (1.0 - t) + fitted[j] * t;
        }
        k = j;
    }
    let tail = chain.len() - 1 - last;
    for (i, k) in (last + 1..chain.len()).enumerate() {
        let t = (i + 1) as f64 / (tail + 1) as f64;
        fitted[k] = fitted[last] * (1.0 - t) + apex_radius * t;
    }
    for (k, &s) in chain.iter().enumerate() {
        radius[s] = fitted[k];
    }
}

/// [`skeletonize`] then [`fit_cylinders`].
pub fn build_qsm(xyz: &[Point], base_xy: Option<[f64; 2]>, p: &QsmParams) -> Result<Qsm> {
    let skel = skeletonize(xyz, base_xy, p)?;
    fit_cylinders(xyz, &skel, p)
}

#[allow(dead_code)]
fn _unused(g: &Graph) -> usize {
    connected_components(g).0
}
