// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Two epochs of one synthetic plot with known changes.
//!
//! Each tree is a fixed structure (a tapered stem, limbs at fixed heights,
//! leaf discs at fixed offsets from the limb tips) that is sampled afresh in
//! each epoch. Between the epochs survivors grow by a known diameter and
//! height increment (a uniform layer of wood along the stem, a longer
//! leader and slightly longer limbs), some trees die, new trees appear,
//! limbs are removed and the foliage in a known box is thinned. Both epochs
//! are scanned from several positions with [`crate::synthetic::scan`], so
//! occlusion is real, with each epoch's own Gaussian range noise, and the
//! second epoch is delivered in a frame displaced by a known rigid
//! transform, as an independently registered survey would be.
//!
//! The true tree tables are in the frame of the first epoch; every stem
//! stands vertically at its `(x, y)`, and its DBH is the diameter 1.3 m
//! above the terrain at the stem centre.

use std::f64::consts::PI;

use crate::util::nprandom::Generator;
use crate::pointcloud::Attr;
use crate::synthetic::{scan, terrain_height, LEAF_RADIUS};
use crate::{Error, Point, PointCloud, Result, Shots, Transform};
use super::{positive, non_negative};

const POINTS_PER_LEAF: usize = 12;
const SLOPE: f64 = 0.05;
const SCANNER_HEIGHT: f64 = 1.5;

/// Settings of [`forest_epochs`].
#[derive(Debug, Clone)]
pub struct EpochParams {
    /// Trees standing in the first epoch.
    pub n_trees: usize,
    /// Side of the square plot (m); trees stand at least 2 m inside it.
    pub size: f64,
    /// Least distance between stems (m).
    pub min_spacing: f64,
    /// Trees of the first epoch that are gone in the second.
    pub deaths: usize,
    /// New trees in the second epoch.
    pub recruits: usize,
    /// Of the deaths and recruits, pairs in which the recruit stands 0.3 to
    /// 0.6 m from the dead stem (a tree felled and another grown nearby).
    pub replaced: usize,
    /// Survivors whose DBH grows by only 0.5 mm and height by 1 cm.
    pub small_increments: usize,
    /// Mean and standard deviation of the DBH increment (m) of survivors.
    pub dbh_increment: (f64, f64),
    /// Mean and standard deviation of the height increment (m) of survivors.
    pub height_increment: (f64, f64),
    /// Survivors that lose their largest limb (with its leaves).
    pub branch_removals: usize,
    /// Box `[xmin, ymin, zmin, xmax, ymax, zmax]` in which `foliage_fraction`
    /// of the leaf discs disappear; `None` puts it on the crown of one
    /// survivor.
    pub foliage_box: Option<[f64; 6]>,
    pub foliage_fraction: f64,
    /// Horizontal displacement (m, random direction) of each survivor's
    /// stem between the epochs; 0 keeps stems in place.
    pub tree_shift: f64,
    /// Rigid transform from the true frame to the frame epoch 2 is
    /// delivered in.
    pub offset: Transform,
    /// Range noise (m, one standard deviation) of the two epochs.
    pub range_noise: [f64; 2],
    /// Scanner positions as fractions of the plot side; the scanner stands
    /// 1.5 m above the terrain.
    pub scan_positions: Vec<[f64; 2]>,
    /// Standard deviation (m) of the horizontal offset of each epoch-2
    /// scanner from the epoch-1 position.
    pub scan_jitter: f64,
    /// Angular step of the scans (degrees).
    pub resolution_deg: f64,
    /// Echoes recorded per pulse. [`scan`] places every echo of a pulse
    /// along the direction of its last one, so with more than one echo the
    /// first echoes at a stem's silhouette are displaced sideways, which
    /// widens stem circles by a few millimetres; one echo keeps every point
    /// on its surface.
    pub max_echoes: usize,
    /// Terrain points per m², before scanning.
    pub ground_density: f64,
    pub seed: u64,
}

impl Default for EpochParams {
    fn default() -> Self {
        let offset = Transform::translation(0.8, -0.5, 0.15).compose(&Transform::from_roll_pitch_yaw(0.1, -0.15, 1.5));
        EpochParams {
            n_trees: 16,
            size: 30.0,
            min_spacing: 2.5,
            deaths: 2,
            recruits: 2,
            replaced: 1,
            small_increments: 2,
            dbh_increment: (0.012, 0.004),
            height_increment: (0.6, 0.2),
            branch_removals: 1,
            foliage_box: None,
            foliage_fraction: 0.5,
            tree_shift: 0.0,
            offset,
            range_noise: [0.003, 0.005],
            scan_positions: vec![[0.5, 0.5], [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]],
            scan_jitter: 0.5,
            resolution_deg: 0.25,
            max_echoes: 1,
            ground_density: 60.0,
            seed: 0,
        }
    }
}

/// A tree of one epoch, as built (ground truth).
#[derive(Debug, Clone, PartialEq)]
pub struct TrueTree {
    /// Identity of the tree, the same in both epochs.
    pub tree_id: i64,
    pub x: f64,
    pub y: f64,
    /// Terrain elevation at the stem centre.
    pub z0: f64,
    /// Stem diameter 1.3 m above `z0` (m).
    pub dbh: f64,
    /// Top of the tree above `z0` (m).
    pub height: f64,
    /// Volume of the stem (m³).
    pub stem_volume: f64,
    /// Stem and limbs (m³).
    pub wood_volume: f64,
    /// One-sided leaf area (m²).
    pub leaf_area: f64,
    pub n_limbs: usize,
}

/// One known change between the epochs.
#[derive(Debug, Clone, PartialEq)]
pub struct TrueChange {
    /// `growth`, `death`, `recruit`, `branch_removed` or `foliage_thinned`.
    pub kind: &'static str,
    /// The tree concerned (-1 for the foliage box as a whole).
    pub tree_id: i64,
    /// Named values of the change (increments, volumes, positions, box).
    pub values: Vec<(&'static str, f64)>,
}

/// The two scanned epochs and their truth.
#[derive(Debug, Clone)]
pub struct ForestEpochs {
    /// The echoes of all scans of each epoch; epoch 2 in its displaced frame.
    pub clouds: Vec<PointCloud>,
    /// The pulses of each epoch (misses included), in the same frames.
    pub shots: Vec<Shots>,
    /// The trees standing in each epoch, in the true frame.
    pub trees: Vec<Vec<TrueTree>>,
    pub changes: Vec<TrueChange>,
    /// Transform taking epoch 2 back onto epoch 1 (the inverse of the offset).
    pub transform: Transform,
    /// Scanner positions of each epoch, in each epoch's delivered frame.
    pub origins: Vec<Vec<Point>>,
}

#[derive(Debug, Clone)]
struct Limb {
    /// Height of the limb base above the stem base.
    h: f64,
    dir: Point,
    length: f64,
    radius: f64,
    /// Leaf discs: centre offset from the tip and unit normal.
    discs: Vec<(Point, Point)>,
}

#[derive(Debug, Clone)]
struct Shape {
    id: i64,
    x: f64,
    y: f64,
    z0: f64,
    r13: f64,
    /// Radius lost per metre of height.
    taper: f64,
    height: f64,
    limbs: Vec<Limb>,
}

fn norm(v: &Point) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross(a: &Point, b: &Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(v: Point) -> Point {
    let n = norm(&v);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Two unit vectors completing `axis` to an orthonormal frame.
fn frame(axis: &Point) -> (Point, Point) {
    let helper = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = unit(cross(axis, &helper));
    (u, cross(axis, &u))
}

impl Shape {
    #[allow(clippy::too_many_arguments)]
    fn new(rng: &mut Generator, id: i64, x: f64, y: f64, dbh: f64, height: f64, n_limbs: usize, leaf_points: usize) -> Shape {
        let r13 = dbh / 2.0;
        let taper = 0.8 * r13 / (height - 1.3).max(1.0);
        let nb = n_limbs as f64;
        let mut limbs = Vec::with_capacity(n_limbs);
        for b in 0..n_limbs {
            let bf = b as f64;
            let h = height * (0.45 + 0.4 * (bf + 0.5) / nb);
            let az = 2.4 * bf + rng.uniform(-0.3, 0.3);
            let length = height * rng.uniform(0.16, 0.26);
            let dir = unit([az.cos(), az.sin(), rng.uniform(0.25, 0.6)]);
            let radius = r13 * 0.3 * (1.0 - 0.5 * bf / nb);
            let tip_z = h + length * dir[2];
            let mut discs = Vec::new();
            for _ in 0..leaf_points / n_limbs.max(1) / POINTS_PER_LEAF {
                let o = [rng.normal(0.0, height * 0.05), rng.normal(0.0, height * 0.05), rng.normal(0.0, height * 0.05)];
                let n = unit([rng.normal(0.0, 1.0), rng.normal(0.0, 1.0), rng.normal(0.0, 1.0) + 1e-9]);
                // The leader is the top of the tree: no leaf above it.
                if tip_z + o[2] + LEAF_RADIUS < height - 0.05 {
                    discs.push((o, n));
                }
            }
            limbs.push(Limb { h, dir, length, radius, discs });
        }
        Shape { id, x, y, z0: terrain_height(x, y, SLOPE), r13, taper, height, limbs }
    }

    fn radius_at(&self, z: f64) -> f64 {
        (self.r13 - self.taper * (z - 1.3)).max(0.004)
    }

    fn base(&self, l: &Limb) -> Point {
        [self.x, self.y, self.z0 + l.h]
    }

    fn tip(&self, l: &Limb) -> Point {
        let b = self.base(l);
        [b[0] + l.length * l.dir[0], b[1] + l.length * l.dir[1], b[2] + l.length * l.dir[2]]
    }

    /// The same tree `dd` wider at every height (below the old top), `dh`
    /// taller and with limbs lengthened in proportion.
    fn grown(&self, dd: f64, dh: f64) -> Shape {
        let mut s = self.clone();
        s.r13 += dd / 2.0;
        s.height += dh;
        let g = dh / self.height;
        for l in &mut s.limbs {
            l.length *= 1.0 + g;
            l.radius += 0.3 * dd / 2.0;
        }
        s
    }

    fn stem_volume(&self) -> f64 {
        let n = 400;
        let dz = self.height / n as f64;
        (0..n).map(|i| PI * self.radius_at((i as f64 + 0.5) * dz).powi(2) * dz).sum()
    }

    fn limb_volume(l: &Limb) -> f64 {
        let (r0, r1) = (l.radius, l.radius * 0.3);
        PI * l.length / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)
    }

    fn truth(&self) -> TrueTree {
        let stem = self.stem_volume();
        let n_discs: usize = self.limbs.iter().map(|l| l.discs.len()).sum();
        TrueTree {
            tree_id: self.id,
            x: self.x,
            y: self.y,
            z0: self.z0,
            dbh: 2.0 * self.radius_at(1.3),
            height: self.height,
            stem_volume: stem,
            wood_volume: stem + self.limbs.iter().map(Shape::limb_volume).sum::<f64>(),
            leaf_area: n_discs as f64 * PI * LEAF_RADIUS * LEAF_RADIUS,
            n_limbs: self.limbs.iter().filter(|l| l.length > 0.0).count(),
        }
    }

    /// Points on the surfaces: stem, limbs and leaves, with their class
    /// (5 wood, 4 leaf) and limb index (-1 on the stem).
    fn sample(&self, rng: &mut Generator, out: &mut Sampled) {
        let density = 3000.0;
        // Stem: area-proportional point count, uniform in height and angle.
        let area = 2.0 * PI * self.radius_at(self.height / 2.0) * self.height;
        let n = (density * area) as usize;
        for _ in 0..n {
            let z = rng.uniform(0.0, self.height);
            let a = rng.uniform(0.0, 2.0 * PI);
            let r = self.radius_at(z);
            out.push([self.x + r * a.cos(), self.y + r * a.sin(), self.z0 + z], 5, self.id, -1);
        }
        out.push([self.x, self.y, self.z0 + self.height], 5, self.id, -1);
        for (k, l) in self.limbs.iter().enumerate() {
            if l.length <= 0.0 {
                continue;
            }
            let (u, v) = frame(&l.dir);
            let base = self.base(l);
            let n = (density * 2.0 * PI * 0.65 * l.radius * l.length).max(20.0) as usize;
            for _ in 0..n {
                let t = rng.uniform(0.0, 1.0);
                let a = rng.uniform(0.0, 2.0 * PI);
                let r = l.radius * (1.0 - 0.7 * t);
                let (c, s) = (a.cos() * r, a.sin() * r);
                let tl = t * l.length;
                out.push(std::array::from_fn(|j| base[j] + tl * l.dir[j] + c * u[j] + s * v[j]), 5, self.id, k as i32);
            }
            let tip = self.tip(l);
            for (o, nrm) in &l.discs {
                let c = [tip[0] + o[0], tip[1] + o[1], tip[2] + o[2]];
                let (e1, e2) = frame(nrm);
                for _ in 0..POINTS_PER_LEAF {
                    let rd = LEAF_RADIUS * rng.uniform(0.0, 1.0).sqrt();
                    let an = rng.uniform(0.0, 2.0 * PI);
                    let (p, q) = (rd * an.cos(), rd * an.sin());
                    out.push(std::array::from_fn(|j| c[j] + p * e1[j] + q * e2[j]), 4, self.id, k as i32);
                }
            }
        }
    }
}

#[derive(Default)]
struct Sampled {
    xyz: Vec<Point>,
    cls: Vec<u8>,
    tree: Vec<i32>,
    branch: Vec<i32>,
}

impl Sampled {
    fn push(&mut self, p: Point, cls: u8, tree: i64, branch: i32) {
        self.xyz.push(p);
        self.cls.push(cls);
        self.tree.push(tree as i32);
        self.branch.push(branch);
    }

    fn into_cloud(self) -> PointCloud {
        let mut c = PointCloud::new(self.xyz);
        c.attrs.insert("classification".into(), Attr::U8(self.cls));
        c.attrs.insert("tree_id".into(), Attr::I32(self.tree));
        c.attrs.insert("branch_id".into(), Attr::I32(self.branch));
        c
    }
}

fn check(p: &EpochParams) -> Result<()> {
    if p.size.is_nan() || p.size <= 4.0 {
        return Err(Error::invalid(format!("size must exceed 4 m, got {}", p.size)));
    }
    if p.deaths > p.n_trees {
        return Err(Error::invalid(format!("deaths ({}) exceed n_trees ({})", p.deaths, p.n_trees)));
    }
    if p.replaced > p.deaths || p.replaced > p.recruits {
        return Err(Error::invalid("replaced cannot exceed deaths or recruits"));
    }
    if p.deaths + p.branch_removals + p.small_increments > p.n_trees {
        return Err(Error::invalid("deaths + branch_removals + small_increments exceed n_trees"));
    }
    if !(0.0..=1.0).contains(&p.foliage_fraction) {
        return Err(Error::invalid(format!("foliage_fraction must be in [0, 1], got {}", p.foliage_fraction)));
    }
    if p.range_noise.iter().any(|s| !non_negative(*s)) || !positive(p.resolution_deg) || !non_negative(p.ground_density) {
        return Err(Error::invalid("range_noise and ground_density must be >= 0 and resolution_deg > 0"));
    }
    if p.max_echoes == 0 {
        return Err(Error::invalid("max_echoes must be at least 1"));
    }
    if p.scan_positions.is_empty() {
        return Err(Error::invalid("need at least one scan position"));
    }
    for v in [p.dbh_increment.0, p.dbh_increment.1, p.height_increment.0, p.height_increment.1, p.tree_shift, p.scan_jitter, p.min_spacing] {
        if !v.is_finite() || v < 0.0 {
            return Err(Error::invalid("increments, spacing, shift and jitter must be finite and >= 0"));
        }
    }
    Ok(())
}

/// A position at least `spacing` from every stem in `taken`, inside the
/// plot (2 m margin).
fn place(rng: &mut Generator, taken: &[(f64, f64)], size: f64, spacing: f64) -> Result<(f64, f64)> {
    for _ in 0..20000 {
        let (x, y) = (rng.uniform(2.0, size - 2.0), rng.uniform(2.0, size - 2.0));
        if taken.iter().all(|&(a, b)| (x - a).hypot(y - b) >= spacing) {
            return Ok((x, y));
        }
    }
    Err(Error::invalid("cannot place the trees at this spacing; lower n_trees or min_spacing, or enlarge size"))
}

/// Allometric height (m) for a DBH (m), with a random factor.
fn height_for(rng: &mut Generator, dbh: f64) -> f64 {
    (1.3 + 20.0 * (1.0 - (-5.0 * dbh).exp())) * rng.uniform(0.9, 1.1)
}

fn scan_epoch(cloud: &PointCloud, origins: &[Point], p: &EpochParams, sigma: f64, rng: &mut Generator) -> Result<Shots> {
    let mut parts = Vec::with_capacity(origins.len());
    for (k, o) in origins.iter().enumerate() {
        let mut s = scan(cloud, *o, p.resolution_deg, 130.0, p.max_echoes, 0.5);
        for r in &mut s.echo_range {
            *r += rng.normal(0.0, sigma);
        }
        s.echo_attrs.insert("scan_id".into(), Attr::I32(vec![k as i32; s.n_echoes()]));
        parts.push(s);
    }
    Shots::concatenate(&parts.iter().collect::<Vec<_>>())
}

/// Scanner positions: the fractions of the plot side, moved to stand at
/// least 1 m from every stem.
fn scanners(p: &EpochParams, stems: &[(f64, f64)], jitter: Option<&mut Generator>) -> Vec<Point> {
    let mut rng = jitter;
    p.scan_positions
        .iter()
        .map(|f| {
            let (mut x, mut y) = (f[0] * p.size, f[1] * p.size);
            if let Some(r) = rng.as_deref_mut() {
                x += r.normal(0.0, p.scan_jitter);
                y += r.normal(0.0, p.scan_jitter);
            }
            for _ in 0..4 {
                for &(a, b) in stems {
                    let d = (x - a).hypot(y - b);
                    if d < 1.0 {
                        let (ux, uy) = if d > 1e-9 { ((x - a) / d, (y - b) / d) } else { (1.0, 0.0) };
                        (x, y) = (a + ux, b + uy);
                    }
                }
            }
            [x, y, terrain_height(x, y, SLOPE) + SCANNER_HEIGHT]
        })
        .collect()
}

fn ground(rng: &mut Generator, p: &EpochParams, out: &mut Sampled) {
    let margin = 4.0;
    let side = p.size + 2.0 * margin;
    let n = (p.ground_density * side * side) as usize;
    for _ in 0..n {
        let (x, y) = (rng.uniform(-margin, p.size + margin), rng.uniform(-margin, p.size + margin));
        out.push([x, y, terrain_height(x, y, SLOPE) + rng.normal(0.0, 0.01)], 2, 0, -1);
    }
}

/// Two scanned epochs of one synthetic plot with known changes; see the
/// module documentation.
pub fn forest_epochs(p: &EpochParams) -> Result<ForestEpochs> {
    check(p)?;
    let mut rng = Generator::new(p.seed);
    // Epoch-1 trees.
    let mut taken: Vec<(f64, f64)> = Vec::new();
    let mut first: Vec<Shape> = Vec::with_capacity(p.n_trees);
    for i in 0..p.n_trees {
        let (x, y) = place(&mut rng, &taken, p.size, p.min_spacing)?;
        taken.push((x, y));
        let dbh = rng.uniform(0.15, 0.5);
        let h = height_for(&mut rng, dbh);
        first.push(Shape::new(&mut rng, i as i64 + 1, x, y, dbh, h, 6, 18000));
    }
    // Roles: a random order of the trees, cut into deaths, limb losses and
    // small increments.
    let order = if p.n_trees > 0 { rng.choice(p.n_trees, p.n_trees) } else { Vec::new() };
    let dead: Vec<usize> = order[..p.deaths].to_vec();
    let pruned: Vec<usize> = order[p.deaths..p.deaths + p.branch_removals].to_vec();
    let small: Vec<usize> = order[p.deaths + p.branch_removals..p.deaths + p.branch_removals + p.small_increments].to_vec();
    let mut changes = Vec::new();
    let mut second: Vec<Shape> = Vec::new();
    for (i, s) in first.iter().enumerate() {
        if dead.contains(&i) {
            let t = s.truth();
            changes.push(TrueChange { kind: "death", tree_id: s.id, values: vec![("x", s.x), ("y", s.y), ("dbh", t.dbh), ("height", t.height), ("wood_volume", t.wood_volume)] });
            continue;
        }
        let (dd, dh) = if small.contains(&i) {
            (0.0005, 0.01)
        } else {
            (rng.normal(p.dbh_increment.0, p.dbh_increment.1).max(0.001), rng.normal(p.height_increment.0, p.height_increment.1).max(0.05))
        };
        let mut g = s.grown(dd, dh);
        if p.tree_shift > 0.0 {
            let a = rng.uniform(0.0, 2.0 * PI);
            g.x += p.tree_shift * a.cos();
            g.y += p.tree_shift * a.sin();
            g.z0 = terrain_height(g.x, g.y, SLOPE);
        }
        let (t1, t2) = (s.truth(), g.truth());
        changes.push(TrueChange { kind: "growth", tree_id: s.id, values: vec![("d_dbh", t2.dbh - t1.dbh), ("d_height", t2.height - t1.height), ("d_stem_volume", t2.stem_volume - t1.stem_volume), ("shift", (g.x - s.x).hypot(g.y - s.y))] });
        if pruned.contains(&i) && !g.limbs.is_empty() {
            let k = (0..g.limbs.len()).max_by(|&a, &b| Shape::limb_volume(&g.limbs[a]).total_cmp(&Shape::limb_volume(&g.limbs[b]))).unwrap_or(0);
            let l = g.limbs.remove(k);
            let (b, tip) = (g.base(&l), g.tip(&l));
            changes.push(TrueChange {
                kind: "branch_removed",
                tree_id: s.id,
                values: vec![("branch_id", k as f64), ("volume", Shape::limb_volume(&l)), ("leaf_area", l.discs.len() as f64 * PI * LEAF_RADIUS * LEAF_RADIUS), ("base_x", b[0]), ("base_y", b[1]), ("base_z", b[2]), ("tip_x", tip[0]), ("tip_y", tip[1]), ("tip_z", tip[2])],
            });
            // Keep the other limbs' ids: the removed one leaves a gap.
            g.limbs.insert(k, Limb { length: 0.0, radius: 0.0, discs: Vec::new(), ..l });
        }
        second.push(g);
    }
    // Recruits: the replaced ones next to a dead stem, the rest anywhere free.
    let mut taken2: Vec<(f64, f64)> = first.iter().map(|s| (s.x, s.y)).chain(second.iter().map(|s| (s.x, s.y))).collect();
    for r in 0..p.recruits {
        let (x, y) = if r < p.replaced {
            let d = &first[dead[r]];
            let a = rng.uniform(0.0, 2.0 * PI);
            let dist = rng.uniform(0.3, 0.6);
            (d.x + dist * a.cos(), d.y + dist * a.sin())
        } else {
            place(&mut rng, &taken2, p.size, p.min_spacing)?
        };
        taken2.push((x, y));
        let dbh = rng.uniform(0.08, 0.12);
        let h = rng.uniform(4.0, 6.0);
        let s = Shape::new(&mut rng, (p.n_trees + r + 1) as i64, x, y, dbh, h, 4, 4000);
        let t = s.truth();
        let mut values = vec![("x", x), ("y", y), ("dbh", t.dbh), ("height", t.height), ("wood_volume", t.wood_volume)];
        if r < p.replaced {
            values.push(("replaces", first[dead[r]].id as f64));
        }
        changes.push(TrueChange { kind: "recruit", tree_id: s.id, values });
        second.push(s);
    }
    // Foliage thinning in a box.
    let bx = match p.foliage_box {
        Some(b) => Some(b),
        None => {
            let pick = order[p.deaths + p.branch_removals + p.small_increments..].first().and_then(|&i| second.iter().find(|s| s.id == first[i].id));
            pick.map(|s| [s.x - 1.5, s.y - 1.5, s.z0 + 0.6 * s.height, s.x + 1.5, s.y + 1.5, s.z0 + s.height + 0.5])
        }
    };
    if let Some(b) = bx {
        let mut removed_total = 0usize;
        for s in &mut second {
            let tips: Vec<Point> = s.limbs.iter().map(|l| s.tip(l)).collect();
            let mut removed = 0usize;
            for (l, tip) in s.limbs.iter_mut().zip(&tips) {
                let before = l.discs.len();
                let mut kept = Vec::with_capacity(before);
                for d in l.discs.drain(..) {
                    let c = [tip[0] + d.0[0], tip[1] + d.0[1], tip[2] + d.0[2]];
                    let inside = (0..3).all(|j| c[j] >= b[j] && c[j] <= b[j + 3]);
                    if inside && rng.random() < p.foliage_fraction {
                        removed += 1;
                    } else {
                        kept.push(d);
                    }
                }
                l.discs = kept;
            }
            if removed > 0 {
                changes.push(TrueChange { kind: "foliage_thinned", tree_id: s.id, values: vec![("leaf_area", removed as f64 * PI * LEAF_RADIUS * LEAF_RADIUS), ("discs", removed as f64)] });
            }
            removed_total += removed;
        }
        changes.push(TrueChange {
            kind: "foliage_thinned",
            tree_id: -1,
            values: vec![("xmin", b[0]), ("ymin", b[1]), ("zmin", b[2]), ("xmax", b[3]), ("ymax", b[4]), ("zmax", b[5]), ("fraction", p.foliage_fraction), ("leaf_area", removed_total as f64 * PI * LEAF_RADIUS * LEAF_RADIUS)],
        });
    }
    // Sample, scan and displace.
    let mut clouds = Vec::with_capacity(2);
    let mut all_shots = Vec::with_capacity(2);
    let mut origins = Vec::with_capacity(2);
    let stems: Vec<(f64, f64)> = first.iter().chain(&second).map(|s| (s.x, s.y)).collect();
    for (e, shapes) in [&first, &second].into_iter().enumerate() {
        let mut srng = Generator::new(p.seed.wrapping_mul(7919).wrapping_add(e as u64 + 1));
        let mut pts = Sampled::default();
        ground(&mut srng, p, &mut pts);
        for s in shapes.iter() {
            s.sample(&mut srng, &mut pts);
        }
        let cloud = pts.into_cloud();
        let o = scanners(p, &stems, if e == 1 { Some(&mut srng) } else { None });
        let mut shots = scan_epoch(&cloud, &o, p, p.range_noise[e], &mut srng)?;
        let mut o = o;
        if e == 1 {
            shots = shots.transformed(&p.offset);
            o = o.iter().map(|q| p.offset.apply(q)).collect();
        }
        clouds.push(shots.to_pointcloud());
        all_shots.push(shots);
        origins.push(o);
    }
    Ok(ForestEpochs {
        clouds,
        shots: all_shots,
        trees: vec![first.iter().map(Shape::truth).collect(), second.iter().map(Shape::truth).collect()],
        changes,
        transform: p.offset.inverse()?,
        origins,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> EpochParams {
        EpochParams { n_trees: 5, size: 16.0, deaths: 1, recruits: 1, replaced: 1, small_increments: 1, branch_removals: 1, scan_positions: vec![[0.5, 0.5]], resolution_deg: 1.0, ground_density: 5.0, ..Default::default() }
    }

    #[test]
    fn truth_tables_follow_the_changes() {
        let f = forest_epochs(&small()).unwrap();
        assert_eq!(f.trees[0].len(), 5);
        assert_eq!(f.trees[1].len(), 5);
        let kinds = |k: &str| f.changes.iter().filter(|c| c.kind == k).count();
        assert_eq!((kinds("death"), kinds("recruit"), kinds("growth"), kinds("branch_removed")), (1, 1, 4, 1));
        // Growth is the difference of the two tables.
        for c in f.changes.iter().filter(|c| c.kind == "growth") {
            let a = f.trees[0].iter().find(|t| t.tree_id == c.tree_id).unwrap();
            let b = f.trees[1].iter().find(|t| t.tree_id == c.tree_id).unwrap();
            assert!((b.dbh - a.dbh - c.values[0].1).abs() < 1e-12);
            assert!(b.dbh > a.dbh && b.height > a.height);
        }
        // The replacing recruit stands next to the dead stem.
        let d = f.changes.iter().find(|c| c.kind == "death").unwrap();
        let r = f.changes.iter().find(|c| c.kind == "recruit").unwrap();
        let dist = (d.values[0].1 - r.values[0].1).hypot(d.values[1].1 - r.values[1].1);
        assert!((0.3..=0.6).contains(&dist));
        assert!(f.clouds[0].len() > 1000 && f.clouds[1].len() > 1000);
        assert!(f.clouds[1].attrs.contains_key("scan_id") && f.clouds[1].attrs.contains_key("branch_id"));
    }

    #[test]
    fn epoch_two_is_displaced_by_the_offset() {
        let mut p = small();
        p.range_noise = [0.0, 0.0];
        let f = forest_epochs(&p).unwrap();
        // The scanner of epoch 2, moved back, stands on the terrain + 1.5 m.
        let o = f.transform.apply(&f.origins[1][0]);
        assert!((o[2] - terrain_height(o[0], o[1], SLOPE) - SCANNER_HEIGHT).abs() < 1e-9);
        let back = f.transform.compose(&p.offset);
        assert!((back.0 - nalgebra::Matrix4::identity()).abs().max() < 1e-12);
    }

    #[test]
    fn same_seed_same_epochs() {
        let a = forest_epochs(&small()).unwrap();
        let b = forest_epochs(&small()).unwrap();
        assert_eq!(a.clouds[1].xyz, b.clouds[1].xyz);
        assert_eq!(a.trees, b.trees);
    }

    #[test]
    fn rejects_impossible_settings() {
        assert!(forest_epochs(&EpochParams { deaths: 9, n_trees: 3, ..small() }).is_err());
        assert!(forest_epochs(&EpochParams { replaced: 2, ..small() }).is_err());
        assert!(forest_epochs(&EpochParams { n_trees: 400, size: 10.0, ..small() }).is_err());
    }
}
