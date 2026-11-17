// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Skeleton extraction and QSM assembly.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use rayon::prelude::*;

use crate::cluster::{connected_components, dijkstra, directed_knn_graph_bridged, Graph};
use crate::error::{Error, Result};
use crate::transform::{add, dot, scale, sub};
use crate::Point;

#[derive(Debug, Clone, PartialEq)]
/// One cylinder of a QSM: a straight tube from `start` along `axis`,
/// linked to its parent by index.
pub struct Cylinder {
    /// Start point (m), in the point cloud's frame.
    pub start: Point,
    /// Unit direction from start to end.
    pub axis: Point,
    /// Length along the axis (m).
    pub length: f64,
    /// Radius (m).
    pub radius: f64,
    /// Index of the parent cylinder in the model; negative (-1) for a root.
    pub parent: i64,
    /// Branch order: 0 for the stem, 1 for branches off it, and so on.
    pub branch_order: u32,
    /// Id of the branch (chain of cylinders) this one belongs to.
    pub branch_id: u32,
    /// Points the radius measurement rests on (circle inliers for a circle fit); 0 when the radius came from the taper prior.
    pub n_points: usize,
}

impl Cylinder {
    /// The far end of the cylinder, `start + axis * length`.
    pub fn end(&self) -> Point {
        add(&self.start, &scale(&self.axis, self.length))
    }

    /// Volume `pi r^2 length` (m3).
    pub fn volume(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius * self.length
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
/// A quantitative structure model of one tree: its cylinders, each linked to
/// its parent, as built by [`build_qsm`] or read from a cylinder table.
pub struct Qsm {
    /// The cylinders, in the order `parent` indexes them.
    pub cylinders: Vec<Cylinder>,
}

impl Qsm {
    /// Every parent index names a cylinder of this model (negative: a root).
    /// Models built from outside data are checked, since the metrics follow
    /// parents without bounds checks.
    pub fn check(&self) -> Result<()> {
        let n = self.cylinders.len() as i64;
        match self.cylinders.iter().position(|c| c.parent >= n) {
            Some(i) => Err(Error::invalid(format!("cylinder {i} has parent {} but the model has {n} cylinders", self.cylinders[i].parent))),
            None => Ok(()),
        }
    }

    /// Number of cylinders.
    pub fn len(&self) -> usize {
        self.cylinders.len()
    }

    /// Whether the model has no cylinders.
    pub fn is_empty(&self) -> bool {
        self.cylinders.is_empty()
    }

    /// Sum of all cylinder volumes (m3).
    pub fn total_volume(&self) -> f64 {
        self.cylinders.iter().map(Cylinder::volume).sum()
    }

    /// Volume of the order-0 (stem) cylinders (m3).
    pub fn stem_volume(&self) -> f64 {
        self.cylinders.iter().filter(|c| c.branch_order == 0).map(Cylinder::volume).sum()
    }

    /// Volume of every cylinder of order 1 and above (m3): total minus stem.
    pub fn branch_volume(&self) -> f64 {
        self.total_volume() - self.stem_volume()
    }

    /// Sum of all cylinder lengths (m).
    pub fn total_length(&self) -> f64 {
        self.cylinders.iter().map(|c| c.length).sum()
    }

    /// Highest branch order of any cylinder; 0 for an empty model.
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
            let helper = least_aligned(&axis);
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
                tris.push([a0, a1, b0]);
                tris.push([a1, b1, b0]);
                tris.push([c0, a1, a0]);
                tris.push([c1, b0, b1]);
                owner.extend([ci as u32; 4]);
            }
        }
        (verts, tris, owner)
    }

    /// Triangle mesh with one continuous tube per branch instead of one closed
    /// tube per cylinder: consecutive cylinders of a branch share a ring, so
    /// there are no caps inside a branch and the surface runs unbroken from
    /// its base to its tip. The ring frame is carried along the branch, so the
    /// facets do not twist, and a shared ring has the radius of the cylinder
    /// that starts at it.
    ///
    /// Each branch is still its own closed surface, pushed into its parent
    /// rather than welded to it: the result is one watertight component per
    /// branch. Welding them into a single solid needs a boolean union, which
    /// this does not do.
    pub fn mesh_contiguous(&self, sides: usize) -> (Vec<Point>, Vec<[u32; 3]>, Vec<u32>) {
        let sides = sides.max(3);
        let n = self.cylinders.len();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, c) in self.cylinders.iter().enumerate() {
            if c.parent >= 0 && (c.parent as usize) < n {
                children[c.parent as usize].push(i);
            }
        }
        // The child that continues the same branch, if any.
        let next: Vec<Option<usize>> = (0..n)
            .map(|i| children[i].iter().copied().find(|&j| self.cylinders[j].branch_id == self.cylinders[i].branch_id))
            .collect();
        let mut head = vec![true; n];
        for (i, nx) in next.iter().enumerate() {
            if let Some(j) = nx {
                if self.cylinders[*j].branch_id == self.cylinders[i].branch_id {
                    head[*j] = false;
                }
            }
        }
        let mut verts: Vec<Point> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut owner: Vec<u32> = Vec::new();
        let ring = |verts: &mut Vec<Point>, centre: Point, u: Point, v: Point, r: f64| -> u32 {
            let base = verts.len() as u32;
            for k in 0..sides {
                let a = k as f64 / sides as f64 * std::f64::consts::TAU;
                let (sa, ca) = a.sin_cos();
                verts.push([
                    centre[0] + r * (ca * u[0] + sa * v[0]),
                    centre[1] + r * (ca * u[1] + sa * v[1]),
                    centre[2] + r * (ca * u[2] + sa * v[2]),
                ]);
            }
            base
        };
        for start in 0..n {
            if !head[start] {
                continue;
            }
            let mut i = start;
            let mut axis = crate::transform::normalize(&self.cylinders[i].axis);
            let helper = least_aligned(&axis);
            let mut u = crate::transform::normalize(&crate::transform::cross(&axis, &helper));
            let mut v = crate::transform::cross(&axis, &u);
            let mut base = ring(&mut verts, self.cylinders[i].start, u, v, self.cylinders[i].radius);
            // Cap the base of the branch.
            let c0 = verts.len() as u32;
            verts.push(self.cylinders[i].start);
            for k in 0..sides as u32 {
                tris.push([c0, base + (k + 1) % sides as u32, base + k]);
                owner.push(i as u32);
            }
            loop {
                let c = &self.cylinders[i];
                let new_axis = crate::transform::normalize(&c.axis);
                // Carry the frame across the joint: project u onto the new normal plane.
                let dot = crate::transform::dot(&u, &new_axis);
                let mut nu = [u[0] - dot * new_axis[0], u[1] - dot * new_axis[1], u[2] - dot * new_axis[2]];
                if crate::transform::norm(&nu) < 1e-9 {
                    let helper = least_aligned(&new_axis);
                    nu = crate::transform::cross(&new_axis, &helper);
                }
                u = crate::transform::normalize(&nu);
                v = crate::transform::cross(&new_axis, &u);
                axis = new_axis;
                let follow = next[i];
                // A shared ring has the radius of the cylinder that starts at it.
                let r_end = match follow {
                    Some(j) => self.cylinders[j].radius,
                    None => c.radius,
                };
                let top = ring(&mut verts, c.end(), u, v, r_end);
                for k in 0..sides as u32 {
                    let k1 = (k + 1) % sides as u32;
                    tris.push([base + k, base + k1, top + k]);
                    tris.push([base + k1, top + k1, top + k]);
                    owner.extend([i as u32; 2]);
                }
                match follow {
                    Some(j) => {
                        base = top;
                        i = j;
                    }
                    None => {
                        // Cap the tip.
                        let c1 = verts.len() as u32;
                        verts.push(c.end());
                        for k in 0..sides as u32 {
                            tris.push([c1, top + k, top + (k + 1) % sides as u32]);
                            owner.push(i as u32);
                        }
                        break;
                    }
                }
            }
            let _ = axis;
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

    /// Write the cylinder table as CSV with a header, one row per cylinder, values to 6 decimals.
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
/// Settings of [`build_qsm`] and [`skeletonize`].
pub struct QsmParams {
    /// Neighbours per point in the kNN graph the skeleton is built on.
    pub k: usize,
    /// Longest graph edge (m); raise it where occlusion leaves gaps along branches.
    pub max_edge: f64,
    /// Geodesic shell width (m); sets the cylinder length.
    pub bin_length: f64,
    /// Fewest circle inliers (or inlier weight) for a section fit to count as a measurement.
    pub min_points: usize,
    /// Circle RANSAC inlier distance (m); the refit band is at least this wide.
    pub ransac_threshold: f64,
    /// No cylinder may be wider than this (set from the measured DBH).
    pub max_radius: f64,
    /// A child cylinder may be at most this times its parent's radius
    /// (monotonic taper); wider fits are clamped, not dropped.
    pub taper_limit: f64,
    /// Circle fits with RMSE above this are rejected (node left unmeasured).
    pub max_rmse: f64,
    /// Smoothing passes over the skeleton's node positions: each moves a
    /// node half way to the mean of its neighbours.
    pub centre_passes: usize,
    /// Radius of a twig: what a tip is given, and the least any cylinder has (m).
    pub twig_radius: f64,
    /// Longest contiguous arc (degrees) a cross-section circle must cover: a
    /// circle through a shorter arc runs wide.
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
    /// Refit band around the RANSAC circle as a fraction of its radius.
    pub relative_tolerance: f64,
    /// A measured radius at breast height (DBH / 2). The stem's taper is
    /// tied to it when the stem has too few sections of its own, or always
    /// with `stem_radius_cap`; 0 for none.
    pub base_radius: f64,
    /// A section is set aside when its radius lies further than this many
    /// robust standard deviations from the taper of its own axis.
    pub outlier_sigma: f64,
    /// With a `base_radius` (a measured DBH), tie the stem's taper to it and
    /// set aside every main-stem circle wider than this many times it,
    /// however well fitted: foliage sheathing a bole (epicormic regrowth) fits wide,
    /// well-supported circles, while a stem is nowhere above breast height
    /// much wider than at it. 0 disables.
    pub stem_radius_cap: f64,
    /// Children's summed cross-section area may exceed the parent's by this
    /// factor at a fork (Leonardo's rule); 0 disables.
    pub pipe_slack: f64,
    /// Scale the band, the shell length and the fit threshold to the cloud's
    /// own point spacing, so a thinned cloud does not silently fall back to
    /// the priors. 0 leaves the given values alone.
    pub spacing_scale: f64,
    /// Radius from a power mean of the distances to the axis, after
    /// raycloudtools (Devereux et al. 2026): `r = (mean d^q)^(1/q)`. A low power is dominated by the
    /// near points, so foliage pulls the radius down rather than up, and
    /// nothing has to lie within a band for it to work. 0 uses the circle fit.
    pub radius_power: f64,
    /// Spacing (m) at or above which the power mean takes over from the circle
    /// fit, when `radius_power` is 0 and scaling is on. Circle fits are the
    /// better of the two while the points are fine enough to fit circles to,
    /// and the worse once they are not. 0 never switches.
    pub power_above_spacing: f64,
    /// Range noise floor in the weight a power-mean measurement carries
    /// (`accuracy = r / (mean |d - r| + noise)`), after raycloudtools.
    pub sensor_noise: f64,
    /// Unmeasured nodes below the lowest accepted stem fit may exceed it by
    /// at most this factor (butt swell).
    pub butt_swell: f64,
    /// Weighted fits only ([`super::weighted::build_qsm_weighted`]): points
    /// with a lower weight are dropped before the graph is built.
    pub min_weight: f64,
    /// Weighted fits only: a circle whose inliers have a lower mean weight
    /// is not a measurement (the section is left to the priors).
    pub min_mean_weight: f64,
}

impl Default for QsmParams {
    fn default() -> Self {
        QsmParams { k: 15, max_edge: 1.0, bin_length: 0.125, min_points: 1, ransac_threshold: 0.02, max_radius: 1.0, taper_limit: 1.1, max_rmse: 0.03, centre_passes: 4, twig_radius: 0.002, min_arc_deg: 180.0, min_inlier_fraction: 0.05, prune_points: 5, fit_min_points: 30, crop_length: 0.0, relative_tolerance: 0.08, base_radius: 0.0, outlier_sigma: 3.0, stem_radius_cap: 0.0, pipe_slack: 1.2, spacing_scale: 1.5, radius_power: 0.0, power_above_spacing: 0.025, sensor_noise: 0.02, butt_swell: 1.1, min_weight: 0.0, min_mean_weight: 0.5 }
    }
}

#[derive(Debug, Clone)]
/// Graph skeleton of one tree from [`skeletonize`]: connected pieces of
/// geodesic shells as segments, their centres and the parent links.
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

/// Connected components of `members` in the graph: the pieces of a shell.
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

/// Graph-based skeleton of a single tree: level sets of geodesic distance
/// from the base, split into connected pieces (Verroust & Lazarus 2000;
/// Xu et al. 2007).
///
/// A cluster hangs from a cluster of a lower shell that its points' shortest
/// paths come through, the nearest of them when there are several.
pub fn skeletonize(xyz: &[Point], base_xy: Option<[f64; 2]>, p: &QsmParams) -> Result<Skeleton> {
    if xyz.is_empty() {
        return Err(Error::invalid("empty cloud"));
    }
    // The graph joins its pieces wherever two come within `max_edge` of each
    // other, so a crown cut off from its trunk by an occluded stretch is
    // still part of the tree. An edge costs its length.
    let graph = directed_knn_graph_bridged(xyz, p.k, p.max_edge, 1.0, None);
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
    for members in &bins {
        if members.is_empty() {
            continue;
        }
        // The pieces of a shell are its connected components in the graph
        // (Verroust and Lazarus 2000).
        let comp = subgraph_components(&graph, members);
        let n = comp.iter().max().unwrap() + 1;
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
    // Node centre: the mean of the cluster. A stem seen from one side has
    // its mean off the axis; the section fits move the node onto the circle
    // later ([`fit_cylinders`]).
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
            c
        })
        .collect();
    // Parent links.
    let seg_bin: Vec<i64> = (0..n_seg)
        .map(|s| members[s].iter().map(|&i| (geod[i] / p.bin_length).floor() as i64).min().unwrap_or(0))
        .collect();
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(n_seg);
    if n_seg > 0 {
        // Each cluster hangs from a cluster of a lower shell that its
        // points' shortest paths come through, the nearest when there are
        // several: the paths give the connection, and the nearest centre
        // keeps the trunk on its core where paths fan out over a flared
        // base. A cluster with no such path (its predecessors were cleaned
        // away) hangs from the nearest centre of a lower shell. The lowest
        // cluster is the root.
        let root = (0..n_seg).min_by(|&a, &b| (seg_bin[a], centres[a][2]).partial_cmp(&(seg_bin[b], centres[b][2])).unwrap()).unwrap();
        let mut by_bin: Vec<usize> = (0..n_seg).collect();
        by_bin.sort_by_key(|&q| seg_bin[q]);
        for q in 0..n_seg {
            if q == root {
                continue;
            }
            let mut through: Vec<usize> = Vec::new();
            for &i in &members[q] {
                let pi = pred[i];
                if pi != usize::MAX && segment_id[pi] >= 0 {
                    let t = segment_id[pi] as usize;
                    if t != q && seg_bin[t] < seg_bin[q] {
                        through.push(t);
                    }
                }
            }
            through.sort_unstable();
            through.dedup();
            let dist = |t: usize| crate::transform::norm(&sub(&centres[t], &centres[q]));
            let nearest = |it: &mut dyn Iterator<Item = usize>| it.min_by(|&x, &y| dist(x).partial_cmp(&dist(y)).unwrap().then_with(|| x.cmp(&y)));
            let parent = nearest(&mut through.iter().copied())
                .or_else(|| nearest(&mut by_bin.iter().copied().take_while(|&t| seg_bin[t] < seg_bin[q])))
                .unwrap_or(root);
            edges.push((q, parent));
        }
    }
    Ok(Skeleton { segment_id, geodesic: geod, centres, edges })
}

/// Order / branch labels and root-to-tip chains: at every node the child with
/// the largest `key` continues its parent's axis, the rest start new axes.
///
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

/// The taper of one axis: `ln r = ln_k + b ln x`, with `x` the length a node
/// carries, and the robust spread `sigma` of its sections about it (in `ln r`).
#[derive(Debug, Clone, Copy)]
struct Taper {
    ln_k: f64,
    b: f64,
    sigma: f64,
}

impl Taper {
    /// Fewest sections for an axis to fit its own exponent and to judge them by it.
    const ENOUGH: usize = 4;

    /// `ln r` at `ln x`.
    fn at(&self, ln_x: f64) -> f64 {
        self.ln_k + self.b * ln_x
    }

    /// The taper through the observations `(ln x, ln r)`: with at least
    /// [`Taper::ENOUGH`] of them the exponent is the median of the slopes
    /// between pairs (Theil 1950; Sen 1968), held between a quarter and
    /// two; with fewer it is 1, a radius in proportion to the length
    /// carried. The level is the median of what is left, and `sigma` 1.4826
    /// times the median absolute residual, never under `floor`. No
    /// observations give a twig.
    fn through(obs: &[(f64, f64)], floor: f64) -> Taper {
        let median = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        if obs.is_empty() {
            return Taper { ln_k: (1e-3f64).ln(), b: 1.0, sigma: floor };
        }
        let mut b = 1.0;
        if obs.len() >= Taper::ENOUGH {
            // Every pair up to a few hundred observations, a stride of them beyond.
            let step = obs.len().div_ceil(300);
            let pick: Vec<(f64, f64)> = obs.iter().step_by(step).copied().collect();
            let mut slopes: Vec<f64> = Vec::with_capacity(pick.len() * (pick.len() - 1) / 2);
            for (i, a) in pick.iter().enumerate() {
                for c in &pick[i + 1..] {
                    if (c.0 - a.0).abs() > 1e-6 {
                        slopes.push((c.1 - a.1) / (c.0 - a.0));
                    }
                }
            }
            if !slopes.is_empty() {
                b = median(&mut slopes).clamp(0.25, 2.0);
            }
        }
        let ln_k = median(&mut obs.iter().map(|o| o.1 - b * o.0).collect());
        let sigma = (1.4826 * median(&mut obs.iter().map(|o| (o.1 - ln_k - b * o.0).abs()).collect())).max(floor);
        Taper { ln_k, b, sigma }
    }
}

/// Assemble a cylinder model from a skeleton.
///
/// 1. Segment centres are smoothed along parent-child chains
///    (`centre_passes` passes), and each node's axis is the skeleton direction
///    through it -- not a per-segment cylinder fit, which wobbles on short
///    sections of thick stems.
/// 2. The radius at each node is a RANSAC circle fitted to the segment's
///    points projected onto the plane perpendicular to that axis, accepted
///    only with enough inliers, a contiguous arc and a small residual;
///    otherwise the node is unmeasured.
/// 3. Each axis gets a robust taper of its own against the length its
///    nodes carry; sections far off it are set aside, the rest made
///    non-increasing by weighted isotonic regression, and what is
///    unmeasured follows the taper or the length carried. A child branch
///    may not be much wider than its parent.
/// 4. Cylinders join consecutive nodes; unsupported leaf segments (no
///    accepted radius, few points) are pruned as foliage.
pub fn fit_cylinders(xyz: &[Point], skel: &Skeleton, p: &QsmParams) -> Result<Qsm> {
    fit_cylinders_with(xyz, skel, p, None)
}

/// [`fit_cylinders`], with a wood weight per point when `weights` is given:
/// each section's radius then comes from the weighted circle fit of
/// [`super::weighted::section_fit`], and an unmeasured leaf segment is pruned
/// as foliage when its summed weight is below `prune_points`.
///
/// The rule for unmeasured radii follows raycloudtools (Lowe and Stepanas
/// 2021; Devereux et al. 2026; `raytrees.cpp`), marked where it occurs below.
pub(crate) fn fit_cylinders_with(xyz: &[Point], skel: &Skeleton, p: &QsmParams, weights: Option<&[f64]>) -> Result<Qsm> {
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

    // 1. Smooth the centres: each pass moves a node half way to the mean of
    // its neighbours in the skeleton (its parent and its children). Roots
    // and tips stay where they are, so no axis shortens.
    let pinned: Vec<bool> = (0..n_seg).map(|s| !parent_of.contains_key(&s) || children[s].is_empty()).collect();
    let smooth = |centre: &mut Vec<Point>| {
        for _ in 0..p.centre_passes {
            let prev = centre.clone();
            for s in 0..n_seg {
                if pinned[s] {
                    continue;
                }
                let mut acc = prev[parent_of[&s]];
                for &ch in &children[s] {
                    acc = add(&acc, &prev[ch]);
                }
                let mean = scale(&acc, 1.0 / (1 + children[s].len()) as f64);
                centre[s] = scale(&add(&prev[s], &mean), 0.5);
            }
        }
    };
    let mut centre: Vec<Point> = skel.centres.clone();
    smooth(&mut centre);
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
    let circle_params = crate::trees::stems::StemParams {
        min_radius: p.twig_radius,
        max_radius: p.max_radius,
        ransac_iterations: 120,
        ransac_tolerance: p.ransac_threshold,
        ..Default::default()
    };
    type Fit = Option<(f64, usize, f64, f64)>;
    let fitted: Vec<(Fit, Option<[f64; 2]>)> = (0..n_seg)
        .into_par_iter()
        .map(|s| -> (Fit, Option<[f64; 2]>) {
            if members[s].len() < p.fit_min_points {
                return (None, None);
            }
            let (u, v) = perp_basis(&axis[s]);
            let xy: Vec<[f64; 2]> = members[s]
                .iter()
                .map(|&i| {
                    let d = sub(&xyz[i], &centre[s]);
                    [dot(&d, &u), dot(&d, &v)]
                })
                .collect();
            if let Some(w) = weights {
                let ws: Vec<f64> = members[s].iter().map(|&i| w[i]).collect();
                return super::weighted::section_fit(&xy, &ws, s as u64 + 1, p, &circle_params);
            }
            if p.radius_power > 0.0 {
                return (power_mean_radius(&xy, p), None);
            }
            let mut rng = crate::filters::Rng::new(s as u64 + 1);
            let Some((cx0, cy0, r0, _)) = crate::trees::stems::ransac_circle(&xy, &circle_params, &mut rng) else { return (None, None) };
            // Rough bark on thick stems spans more than the RANSAC band, and a
            // tight band then favours a sub-arc and a small radius. Refit on
            // every point within a radius-relative band (>= 8 % of r).
            let band = p.ransac_threshold.max(p.relative_tolerance * r0);
            let inl: Vec<[f64; 2]> = xy.iter().filter(|q| ((q[0] - cx0).hypot(q[1] - cy0) - r0).abs() <= band).cloned().collect();
            let (cx, cy, r) = crate::trees::stems::fit_circle_refined(&inl).unwrap_or((cx0, cy0, r0));
            let inl: Vec<[f64; 2]> = xy.iter().filter(|q| ((q[0] - cx).hypot(q[1] - cy) - r).abs() <= band).cloned().collect();
            if inl.len() < p.min_points {
                return (None, None);
            }
            let rmse = (inl.iter().map(|q| ((q[0] - cx).hypot(q[1] - cy) - r).powi(2)).sum::<f64>() / inl.len() as f64).sqrt();
            let arc = crate::trees::stems::angular_extent(&inl, cx, cy);
            let frac = inl.len() as f64 / xy.len() as f64;
            let ok = arc >= p.min_arc_deg && rmse <= p.max_rmse.max(band) && frac >= p.min_inlier_fraction;
            if ok { (Some((r, inl.len(), arc, frac)), Some([cx, cy])) } else { (None, None) }
        })
        .collect();
    let mut fits: Vec<Fit> = fitted.iter().map(|f| f.0).collect();
    // Move each fitted node onto the centre of its circle (the mean of a
    // stem seen from one side lies off its axis), and smooth again.
    for s in 0..n_seg {
        if let (Some((r, ..)), Some(c)) = (fitted[s].0, fitted[s].1) {
            if c[0].hypot(c[1]) <= r {
                let (u, v) = perp_basis(&axis[s]);
                centre[s] = add(&centre[s], &add(&scale(&u, c[0]), &scale(&v, c[1])));
            }
        }
    }
    smooth(&mut centre);

    // 3. Radii along the axes.
    //
    // Each axis gets a taper of its own, r = k x^b, fitted robustly to its
    // accepted sections, where x is the length a node carries: the longest
    // way from it to a tip, times the share of the cross-section that its
    // length claimed at every fork on the way from the root (raycloudtools,
    // Lowe and Stepanas 2021; Devereux et al. 2026; `raytrees.cpp`: a radius
    // in proportion to length, a fork sharing its area by length so that
    // area is conserved, as in the pipe model of Shinozaki et al. 1964). A
    // section whose radius lies far off the taper of its own axis is set
    // aside; what is unmeasured takes the taper, and an axis with nothing
    // measured takes its parent's, which with b = 1 is raycloudtools' rule.
    let mut sub_len = vec![0.0f64; n_seg];
    for &s in topo.iter().rev() {
        let mut best = 0.0f64;
        for &c in &children[s] {
            best = best.max(sub_len[c] + crate::transform::norm(&sub(&centre[c], &centre[s])));
        }
        sub_len[s] = best;
    }
    // Axes: at each node the child whose subtree holds the most points
    // continues the axis (ties: the longest way to a tip); the others start
    // branches one order higher. Points measure scanned surface, so the
    // thicker and longer limb is the stem, and a long thin side branch is
    // not.
    let mut sub_pts = vec![0.0f64; n_seg];
    for &s in topo.iter().rev() {
        sub_pts[s] += members[s].len() as f64;
        if let Some(&q) = parent_of.get(&s) {
            sub_pts[q] += sub_pts[s];
        }
    }
    let key_pts: Vec<(f64, f64)> = (0..n_seg).map(|s| (sub_pts[s], sub_len[s])).collect();
    let (order, branch, chains, chain_of) = assign_axes(n_seg, &topo, &roots, &children, &key_pts);
    // The length a node carries. A tip carries its own shell.
    let mut share = vec![1.0f64; n_seg];
    for &s in &topo {
        let kids = &children[s];
        let total = kids.iter().map(|&c| (sub_len[c] + p.bin_length).powi(2)).sum::<f64>().sqrt();
        for &c in kids {
            share[c] = share[s] * if kids.len() > 1 && total > 0.0 { (sub_len[c] + p.bin_length) / total } else { 1.0 };
        }
    }
    let carried: Vec<f64> = (0..n_seg).map(|s| (sub_len[s] + p.bin_length) * share[s]).collect();
    // With a trusted DBH, main-stem circles far wider than it are not the
    // stem (foliage around it).
    let capped = p.stem_radius_cap > 0.0 && p.base_radius > 0.0;
    if capped {
        for s in 0..n_seg {
            if order[s] == 0 && matches!(fits[s], Some((r, ..)) if r > p.stem_radius_cap * p.base_radius) {
                fits[s] = None;
            }
        }
    }
    let mut radius: Vec<f64> = vec![f64::NAN; n_seg];
    let mut weight: Vec<f64> = vec![0.0; n_seg];
    // Segments whose own circle was set aside: their cylinders report no
    // points, so they do not count as measured.
    let mut set_aside: Vec<bool> = vec![false; n_seg];
    let mut tapers: Vec<Option<Taper>> = vec![None; chains.len()];
    let mut by_order: Vec<usize> = (0..chains.len()).collect();
    by_order.sort_by_key(|&c| order[chains[c][0]]);
    // The tree's own taper, for an axis that has neither sections nor a parent with any.
    let overall = Taper::through(&fits.iter().enumerate().filter_map(|(s, f)| f.map(|(r, ..)| (carried[s].ln(), r.ln()))).collect::<Vec<_>>(), p.relative_tolerance);
    for &ci in &by_order {
        let chain = &chains[ci];
        let stem = order[chain[0]] == 0;
        let obs: Vec<(usize, f64, f64)> = chain.iter().filter_map(|&q| fits[q].map(|(r, ..)| (q, carried[q].ln(), r.ln()))).collect();
        let pairs: Vec<(f64, f64)> = obs.iter().map(|o| (o.1, o.2)).collect();
        let mut taper = if pairs.is_empty() { None } else { Some(Taper::through(&pairs, p.relative_tolerance)) };
        // A measured DBH ties the stem's taper down where the stem has few
        // sections of its own, or always when it is trusted (`capped`).
        if stem && p.base_radius > 0.0 && (capped || pairs.len() < Taper::ENOUGH) {
            let z0 = centre[chain[0]][2];
            let q = *chain.iter().min_by(|&&a, &&b| (centre[a][2] - z0 - 1.3).abs().partial_cmp(&(centre[b][2] - z0 - 1.3).abs()).unwrap()).unwrap();
            let b = taper.map(|t| t.b).unwrap_or(1.0);
            taper = Some(Taper { ln_k: p.base_radius.ln() - b * carried[q].ln(), b, sigma: taper.map(|t| t.sigma).unwrap_or(p.relative_tolerance) });
        }
        // No section on this axis: the taper of the axis it hangs from.
        let taper = taper.or_else(|| parent_of.get(&chain[0]).and_then(|par| tapers[chain_of[*par]])).unwrap_or(overall);
        tapers[ci] = Some(taper);
        for &(q, x, y) in &obs {
            let (r, n, arc, _) = fits[q].unwrap();
            if pairs.len() >= Taper::ENOUGH && (y - taper.at(x)).abs() > p.outlier_sigma * taper.sigma {
                set_aside[q] = true;
            } else {
                radius[q] = r;
                weight[q] = n as f64 * (arc / 360.0).clamp(0.1, 1.0);
            }
        }
        // Between its accepted sections the axis is made non-increasing and
        // its gaps are interpolated; below the lowest it keeps that radius,
        // or on the stem the taper's, up to `butt_swell` times it. Beyond the
        // last section, and on an axis with none, the radius is in
        // proportion to the length carried, from the last measured node or
        // from the node the axis hangs from: the taper is fitted on thick,
        // measured wood and is not extrapolated to twigs.
        let model = |q: usize| taper.at(carried[q].ln()).exp().max(p.twig_radius);
        let along = |radius: &[f64], from: usize, q: usize| (radius[from] * carried[q] / carried[from]).max(p.twig_radius);
        match chain.iter().rposition(|&q| weight[q] > 0.0) {
            Some(k_last) => {
                let k_first = chain.iter().position(|&q| weight[q] > 0.0).unwrap();
                isotonic_fill(&chain[..=k_last], &mut radius, &weight, p.twig_radius);
                if stem {
                    let low = radius[chain[k_first]];
                    for &q in &chain[..k_first] {
                        radius[q] = model(q).clamp(low, low * p.butt_swell.max(1.0));
                    }
                }
                let last = chain[k_last];
                for k in k_last + 1..chain.len() {
                    radius[chain[k]] = along(&radius, last, chain[k]).min(radius[chain[k - 1]]);
                }
            }
            None => {
                let from = parent_of.get(&chain[0]).copied().filter(|&par| radius[par].is_finite());
                for (k, &q) in chain.iter().enumerate() {
                    let r = match from {
                        Some(par) => along(&radius, par, q),
                        None => model(q),
                    };
                    radius[q] = if k == 0 { r } else { r.min(radius[chain[k - 1]]) };
                }
            }
        }
    }
    // A child is never much wider than what it hangs from.
    for &s in &topo {
        if let Some(&par) = parent_of.get(&s) {
            if !(radius[s] <= p.taper_limit * radius[par]) {
                radius[s] = radius[par];
            }
        }
    }
    // Leonardo's rule at forks (as in raycloudtools, Devereux et al. 2026): the
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
                    radius[c] = (radius[c] * f).max(p.twig_radius);
                }
            }
            let total: f64 = kids.iter().map(|&c| radius[c] * radius[c]).sum();
            if total > p.pipe_slack * parent_area {
                let f = (p.pipe_slack * parent_area / total).sqrt();
                for &c in &kids {
                    radius[c] = (radius[c] * f).max(p.twig_radius);
                }
            }
        }
    }
    for r in &mut radius {
        if !r.is_finite() {
            *r = p.twig_radius;
        }
        *r = r.clamp(p.twig_radius, p.max_radius);
    }
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
            let few = match weights {
                None => members[s].len() < p.prune_points,
                Some(w) => members[s].iter().map(|&i| w[i]).sum::<f64>() < p.prune_points as f64,
            };
            if !has_live_child && few {
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
            n_points: if set_aside[s] { 0 } else { fits[s].map(|f| f.1).unwrap_or(0) },
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
    Ok(Qsm { cylinders })
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
    let helper = least_aligned(&axis);
    let u = crate::transform::normalize(&crate::transform::cross(axis, &helper));
    let v = crate::transform::cross(axis, &u);
    (u, v)
}

/// The coordinate axis furthest from `axis`, to build a frame around it.
pub(crate) fn least_aligned(axis: &Point) -> Point {
    let a = [axis[0].abs(), axis[1].abs(), axis[2].abs()];
    let k = if a[0] <= a[1] && a[0] <= a[2] { 0 } else if a[1] <= a[2] { 1 } else { 2 };
    let mut e = [0.0; 3];
    e[k] = 1.0;
    e
}

/// Non-increasing weighted isotonic regression along `chain` (base first);
/// NaN radii are interpolated between measured neighbours and the tip end
/// falls to `twig_radius` when nothing is measured beyond it.
fn isotonic_fill(chain: &[usize], radius: &mut [f64], weight: &[f64], twig_radius: f64) {
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
        fitted[k] = fitted[last] * (1.0 - t) + twig_radius * t;
    }
    for (k, &s) in chain.iter().enumerate() {
        radius[s] = fitted[k];
    }
}

/// Radius of a cross-section from a power mean of the distances to its axis,
/// after raycloudtools (`raytrees.cpp`, `power = 0.25`; Devereux et al. 2026).
///
/// `r = (mean d^q)^(1/q)` with `q` well below 1 is dominated by the nearest
/// points, so a shell that caught foliage or a neighbour reads slightly small
/// rather than wildly large: the opposite failure to a circle fit, which is
/// dragged outward by the same points and, on a sparse shell, is free to pick
/// any wide circle at all. Nothing has to lie within a band, so it degrades
/// gently as the cloud thins.
///
/// Returns the same tuple as the circle fit: radius, points, angular coverage
/// and a confidence in `[0, 1]` from `r / (mean |d - r| + sensor_noise)`,
/// which falls as the section's points scatter.
pub(super) fn power_mean_radius(xy: &[[f64; 2]], p: &QsmParams) -> Option<(f64, usize, f64, f64)> {
    let n = xy.len();
    if n < 3 {
        return None;
    }
    // `xy` is already in the section's own frame, so the axis is the origin:
    // the points' centroid sits on the surface when a scan only sees one side.
    let mut d: Vec<f64> = xy.iter().map(|q| q[0].hypot(q[1])).collect();
    // Drop what is clearly not this section's bark before averaging: a shell
    // that caught a branch or a leaf cluster has a tail far off the axis, and
    // even a low power mean is moved by enough of it (raycloudtools trims the
    // same way in `removeDistantPoints`).
    let mut sorted = d.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    if med > 0.0 {
        d.retain(|&x| x <= 2.0 * med);
    }
    let n = d.len();
    if n < 3 {
        return None;
    }
    let q = p.radius_power;
    let mean = d.iter().map(|&x| x.max(1e-6).powf(q)).sum::<f64>() / n as f64;
    let r = mean.powf(1.0 / q);
    if !(r.is_finite() && r >= p.twig_radius && r <= p.max_radius) {
        return None;
    }
    let e = d.iter().map(|&x| (x - r).abs()).sum::<f64>() / n as f64;
    let accuracy = r / (e + p.sensor_noise);
    let arc = crate::trees::stems::angular_extent(xy, 0.0, 0.0);
    // Map the accuracy onto the [0, 1] confidence the rest of the model reads
    // as an inlier fraction: 1 is a section whose points sit on a circle to
    // within the noise floor.
    let frac = (accuracy / (accuracy + 4.0)).clamp(1e-3, 1.0);
    Some((r, n, arc, frac))
}

/// The parameters that are tied to how far apart the points are, rescaled to
/// this cloud.
///
/// A circle band of 2 cm and shells of 10 cm suit a centimetre cloud. Thin it
/// to 5 cm and no point lies within the band of an honest circle, so the fit
/// falls to whatever wide circle catches strays, and the shells hold fewer
/// points than `fit_min_points`, so most sections are never measured at all.
/// Both failures are silent and both inflate. Here the band, the shell length
/// and the count are set from the cloud's own median spacing, and none of them
/// is ever made tighter than asked for.
pub fn scaled_to_spacing(xyz: &[Point], p: &QsmParams) -> QsmParams {
    let mut out = p.clone();
    if p.spacing_scale <= 0.0 || xyz.len() < 100 {
        return out;
    }
    let spacing = crate::leaves::median_spacing(xyz);
    if !(spacing > 0.0) {
        return out;
    }
    out.ransac_threshold = p.ransac_threshold.max(p.spacing_scale * spacing);
    out.max_rmse = p.max_rmse.max(out.ransac_threshold);
    // A shell must be a few points deep, and its circumference a few points
    // round, before a radius means anything.
    out.bin_length = p.bin_length.max(4.0 * spacing);
    // Points per shell fall with the square of the spacing.
    let coarser = (spacing / 0.01).max(1.0);
    out.fit_min_points = ((p.fit_min_points as f64 / (coarser * coarser)).round() as usize).max(8);
    // Past a few centimetres a circle has too little to bite on, and the
    // robust mean is both less biased and far less prone to a runaway.
    if out.radius_power <= 0.0 && p.power_above_spacing > 0.0 && spacing >= p.power_above_spacing {
        out.radius_power = 0.25;
    }
    out
}

/// [`skeletonize`] then [`fit_cylinders`].
pub fn build_qsm(xyz: &[Point], base_xy: Option<[f64; 2]>, p: &QsmParams) -> Result<Qsm> {
    let task = crate::util::progress::start("building a QSM", 2);
    let p = &scaled_to_spacing(xyz, p);
    let skel = skeletonize(xyz, base_xy, p)?;
    task.inc(1); // skeleton
    let qsm = fit_cylinders(xyz, &skel, p);
    task.inc(1); // cylinders
    qsm
}

#[allow(dead_code)]
fn _unused(g: &Graph) -> usize {
    connected_components(g).0
}
