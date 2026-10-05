// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A synthetic forest plot with its truth.
//!
//! The stand has a stem density (stems per hectare) and a diameter
//! distribution (Weibull, reverse-J or given diameters, truncated at
//! `min_dbh`); heights follow the archetype's Chapman-Richards
//! height-diameter curve `1.3 + a (1 - exp(-b D))^c` (D in cm) with
//! lognormal scatter; crowns and leaf area follow the allometries of
//! [`crate::synthetic::tree`]. Archetypes are mixed by weight. Stems are
//! placed uniformly at random but never closer than `0.75 (D_i + D_j) +
//! 0.2` m (a hard-core process, largest trees first). The terrain is a plane
//! of given slope and aspect plus micro-relief: a Gaussian random field of
//! standard deviation `roughness` and correlation length
//! `roughness_length`, the sum of 64 random Fourier modes. Understorey
//! shrubs, grass tufts, fallen logs and stumps are optional.
//!
//! Every tree is grown with its own seed, drawn in order from the plot's
//! seed, and the trees are grown in parallel; the result does not depend on
//! the number of threads.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::util::nprandom::Generator;
use crate::pointcloud::Attr;
use crate::qsm::Cylinder;
use crate::synthetic::tree::{insert_normals, jittered_grid, tree_model, Archetype, Leaf, SynthTree, TreeSpec, LABEL_DEAD_WOOD, LABEL_GROUND, LABEL_UNDERSTOREY};
use crate::voxel::Lad;
use crate::util::limits;
use crate::{Point, PointCloud};

/// Diameter distribution of a plot.
#[derive(Debug, Clone, PartialEq)]
pub enum DbhDistribution {
    /// Weibull with `shape` and `scale` (m), truncated below at `min_dbh`.
    Weibull { shape: f64, scale: f64 },
    /// Negative exponential above `min_dbh` with mean excess `mean` (m): the
    /// reverse-J of uneven-aged stands (de Liocourt's constant quotient).
    ReverseJ { mean: f64 },
    /// These diameters (m), one tree each; the density is then ignored.
    Given(Vec<f64>),
}

/// Terrain of a plot: a plane plus a random field.
#[derive(Debug, Clone, PartialEq)]
pub struct Terrain {
    /// Rise per metre towards `aspect_deg`.
    pub slope: f64,
    /// Direction of steepest ascent, degrees clockwise from north (+y).
    pub aspect_deg: f64,
    /// `(kx, ky, phase, amplitude)` of the Fourier modes.
    pub modes: Vec<[f64; 4]>,
}

impl Terrain {
    pub fn new(slope: f64, aspect_deg: f64, roughness: f64, length: f64, seed: u64) -> Terrain {
        const K: usize = 64;
        let mut rng = Generator::new(seed);
        let modes = if roughness > 0.0 && length > 0.0 {
            (0..K).map(|_| [rng.normal(0.0, 1.0) / length, rng.normal(0.0, 1.0) / length, rng.uniform(0.0, 2.0 * PI), roughness * (2.0 / K as f64).sqrt()]).collect()
        } else {
            Vec::new()
        };
        Terrain { slope, aspect_deg, modes }
    }

    pub fn height(&self, x: f64, y: f64) -> f64 {
        let a = self.aspect_deg.to_radians();
        let mut z = self.slope * (x * a.sin() + y * a.cos());
        for m in &self.modes {
            z += m[3] * (m[0] * x + m[1] * y + m[2]).cos();
        }
        z
    }

    /// `(dz/dx, dz/dy)`.
    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        let a = self.aspect_deg.to_radians();
        let (mut gx, mut gy) = (self.slope * a.sin(), self.slope * a.cos());
        for m in &self.modes {
            let s = -m[3] * (m[0] * x + m[1] * y + m[2]).sin();
            gx += s * m[0];
            gy += s * m[1];
        }
        (gx, gy)
    }
}

/// Settings of [`plot`].
#[derive(Debug, Clone)]
pub struct PlotSpec {
    /// Side of the square plot (m); trees stand within it.
    pub size: f64,
    /// Stems per hectare.
    pub density: f64,
    pub dbh: DbhDistribution,
    pub min_dbh: f64,
    /// Archetypes and their weights.
    pub archetypes: Vec<(Archetype, f64)>,
    /// Standard deviation of the log of height about the allometry.
    pub height_noise: f64,
    pub slope: f64,
    pub aspect_deg: f64,
    pub roughness: f64,
    pub roughness_length: f64,
    /// Understorey shrubs per hectare.
    pub shrubs: f64,
    /// Fraction of the ground covered by grass tufts.
    pub grass_cover: f64,
    /// Tallest grass blade (m).
    pub grass_height: f64,
    /// Fallen logs and stumps per hectare.
    pub logs: f64,
    pub stumps: f64,
    /// Terrain points per m², over the plot and `margin` m beyond it.
    pub ground_density: f64,
    pub margin: f64,
    /// Points per m² of wood, leaf, understorey and dead wood.
    pub point_density: f64,
    pub max_order: u32,
    pub lad: Option<Lad>,
    /// Epicormic shoots per metre of bole.
    pub epicormic: f64,
    pub seed: u64,
}

impl PlotSpec {
    pub fn new(seed: u64) -> PlotSpec {
        PlotSpec { size: 30.0, density: 600.0, dbh: DbhDistribution::Weibull { shape: 1.8, scale: 0.22 }, min_dbh: 0.07, archetypes: vec![(Archetype::Broadleaf, 1.0)], height_noise: 0.1, slope: 0.1, aspect_deg: 90.0, roughness: 0.05, roughness_length: 2.0, shrubs: 400.0, grass_cover: 0.2, grass_height: 0.5, logs: 60.0, stumps: 40.0, ground_density: 400.0, margin: 2.0, point_density: 1000.0, max_order: 3, lad: None, epicormic: 0.0, seed }
    }

    fn check(&self) -> Result<()> {
        let positive = [("size", self.size), ("min_dbh", self.min_dbh), ("point_density", self.point_density), ("roughness_length", self.roughness_length), ("grass_height", self.grass_height)];
        for (name, v) in positive {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::invalid(format!("{name} must be a positive number, got {v}")));
            }
        }
        let non_negative = [("density", self.density), ("height_noise", self.height_noise), ("roughness", self.roughness), ("shrubs", self.shrubs), ("logs", self.logs), ("stumps", self.stumps), ("ground_density", self.ground_density), ("margin", self.margin), ("epicormic", self.epicormic)];
        for (name, v) in non_negative {
            if !(v.is_finite() && v >= 0.0) {
                return Err(Error::invalid(format!("{name} must be zero or more, got {v}")));
            }
        }
        if !(0.0..=1.0).contains(&self.grass_cover) {
            return Err(Error::invalid(format!("grass_cover must be between 0 and 1, got {}", self.grass_cover)));
        }
        if !(self.slope.is_finite() && self.slope.abs() <= 1.0 && self.aspect_deg.is_finite()) {
            return Err(Error::invalid(format!("slope must be within ±1 and aspect finite, got {} and {}", self.slope, self.aspect_deg)));
        }
        if self.archetypes.is_empty() || self.archetypes.iter().any(|a| !(a.1.is_finite() && a.1 >= 0.0)) || self.archetypes.iter().map(|a| a.1).sum::<f64>() <= 0.0 {
            return Err(Error::invalid("archetypes need non-negative weights with a positive sum"));
        }
        match &self.dbh {
            DbhDistribution::Weibull { shape, scale } if !(*shape > 0.0 && *scale > 0.0 && shape.is_finite() && scale.is_finite()) => Err(Error::invalid(format!("Weibull shape and scale must be positive, got {shape} and {scale}"))),
            DbhDistribution::ReverseJ { mean } if !(*mean > 0.0 && mean.is_finite()) => Err(Error::invalid(format!("the reverse-J mean must be positive, got {mean}"))),
            DbhDistribution::Given(d) if d.iter().any(|v| !(v.is_finite() && *v > 0.0)) => Err(Error::invalid("given diameters must be positive")),
            _ => Ok(()),
        }
    }
}

/// Truth of one tree of a plot.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotTree {
    pub id: i32,
    pub archetype: Archetype,
    /// Stem base on the terrain.
    pub base: Point,
    /// Stem axis 1.3 m along the stem above the terrain.
    pub stem_bh: Point,
    pub dbh: f64,
    /// Highest point above the terrain at the stem.
    pub height: f64,
    pub crown_base: f64,
    pub crown_area: f64,
    pub wood_volume: f64,
    pub stem_volume: f64,
    pub branch_volume: f64,
    pub leaf_area: f64,
}

/// A fallen log (kind 0) or a stump (kind 1): a cylinder.
#[derive(Debug, Clone, PartialEq)]
pub struct DeadWood {
    pub kind: u8,
    pub start: Point,
    pub axis: Point,
    pub length: f64,
    pub radius: f64,
}

/// A plot and its truth.
#[derive(Debug, Clone)]
pub struct Plot {
    /// Points with `classification` (2 ground, 3 understorey, 4 leaf, 5
    /// wood and dead wood), `label` (1 ground, 2 stem, 3 branch, 4 leaf, 5
    /// understorey, 6 dead wood), `tree_id` (0 off the trees), `branch_order`
    /// (-1 off the trees), `cylinder` and `leaf` (rows of the tables, -1
    /// elsewhere) and `epicormic`.
    pub cloud: PointCloud,
    pub trees: Vec<PlotTree>,
    /// Wood cylinders of all trees; `parent` refers to rows of the same tree.
    pub cylinders: Vec<Cylinder>,
    pub cylinder_tree: Vec<i32>,
    pub leaves: Vec<Leaf>,
    pub leaf_tree: Vec<i32>,
    pub dead_wood: Vec<DeadWood>,
    /// Shrubs as `(x, y, height, leaf area)`.
    pub shrubs: Vec<[f64; 4]>,
    /// Grass tufts as `(x, y, radius)`.
    pub grass: Vec<[f64; 3]>,
    pub terrain: Terrain,
}

fn draw_dbh(rng: &mut Generator, d: &DbhDistribution, min_dbh: f64) -> f64 {
    for _ in 0..10_000 {
        let v = match d {
            DbhDistribution::Weibull { shape, scale } => scale * (-(1.0 - rng.random()).ln()).powf(1.0 / shape),
            DbhDistribution::ReverseJ { mean } => min_dbh - mean * (1.0 - rng.random()).ln(),
            DbhDistribution::Given(_) => unreachable!(),
        };
        if v >= min_dbh {
            return v;
        }
    }
    min_dbh
}

fn pick(rng: &mut Generator, weights: &[(Archetype, f64)]) -> Archetype {
    let total: f64 = weights.iter().map(|w| w.1).sum();
    let mut u = rng.random() * total;
    for w in weights {
        if u < w.1 {
            return w.0;
        }
        u -= w.1;
    }
    weights.iter().rev().find(|w| w.1 > 0.0).unwrap().0
}

struct Part {
    xyz: Vec<Point>,
    nrm: Vec<Point>,
    cls: Vec<u8>,
    label: Vec<u8>,
}

impl Part {
    fn new() -> Part {
        Part { xyz: Vec::new(), nrm: Vec::new(), cls: Vec::new(), label: Vec::new() }
    }

    fn push(&mut self, p: Point, n: Point, cls: u8, label: u8) {
        self.xyz.push(p);
        self.nrm.push(n);
        self.cls.push(cls);
        self.label.push(label);
    }
}

/// Points on a cylinder above the terrain.
#[allow(clippy::too_many_arguments)]
fn cylinder_points(rng: &mut Generator, part: &mut Part, t: &Terrain, start: Point, axis: Point, length: f64, radius: f64, density: f64, cap: bool) {
    let x = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let d = |a: &Point, b: &Point| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let ux = d(&x, &axis);
    let u = [x[0] - ux * axis[0], x[1] - ux * axis[1], x[2] - ux * axis[2]];
    let nu = d(&u, &u).sqrt();
    let u = [u[0] / nu, u[1] / nu, u[2] / nu];
    let v = [axis[1] * u[2] - axis[2] * u[1], axis[2] * u[0] - axis[0] * u[2], axis[0] * u[1] - axis[1] * u[0]];
    let emit = |part: &mut Part, q: Point, n: Point| {
        if q[2] >= t.height(q[0], q[1]) {
            part.push(q, n, 5, LABEL_DEAD_WOOD);
        }
    };
    for [arc, s] in jittered_grid(rng, density * 2.0 * PI * radius * length, 2.0 * PI * radius, length) {
        let a = arc / radius;
        let (c, sn) = (radius * a.cos(), radius * a.sin());
        emit(part, std::array::from_fn(|k| start[k] + s * axis[k] + c * u[k] + sn * v[k]), std::array::from_fn(|k| a.cos() * u[k] + a.sin() * v[k]));
    }
    if cap {
        let n = (density * PI * radius * radius).round() as usize;
        for _ in 0..n {
            let (r, a) = (radius * rng.random().sqrt(), rng.uniform(0.0, 2.0 * PI));
            let (c, sn) = (r * a.cos(), r * a.sin());
            emit(part, std::array::from_fn(|k| start[k] + length * axis[k] + c * u[k] + sn * v[k]), axis);
        }
    }
}

/// Build a plot (see the module documentation).
pub fn plot(spec: &PlotSpec) -> Result<Plot> {
    spec.check()?;
    let area = spec.size * spec.size;
    let mut rng = Generator::new(spec.seed);
    let terrain = Terrain::new(spec.slope, spec.aspect_deg, spec.roughness, spec.roughness_length, rng.next_u64());
    // ---- stems: diameters, archetypes, heights, positions
    let mut dbh: Vec<f64> = match &spec.dbh {
        DbhDistribution::Given(d) => d.clone(),
        d => {
            let n = (spec.density * area / 1e4).round() as usize;
            (0..n).map(|_| draw_dbh(&mut rng, d, spec.min_dbh)).collect()
        }
    };
    limits::check_cells(dbh.len() as u128, 1 << 20, &format!("a plot of {} trees", dbh.len()), "a lower density or a smaller plot")?;
    dbh.sort_by(|a, b| b.total_cmp(a));
    let too_dense = || Error::invalid(format!("could not place {} stems of these diameters without overlap in a {} m plot; lower the density", dbh.len(), spec.size));
    // Discs of half the smallest spacing cannot overlap; beyond a packing fraction of 0.6 give up at once.
    if dbh.iter().map(|d| PI * (0.375 * d + 0.1).powi(2)).sum::<f64>() > 0.6 * area {
        return Err(too_dense());
    }
    let cell = 1.5 * dbh.first().copied().unwrap_or(0.0) + 0.2;
    let nc = ((spec.size / cell).ceil() as usize).max(1);
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); nc * nc];
    let cell_of = |v: f64| ((v / cell) as usize).min(nc - 1);
    let mut placed: Vec<(f64, f64, f64)> = Vec::new();
    for &d in &dbh {
        let mut ok = None;
        for _ in 0..2000 {
            let (x, y) = (rng.uniform(0.0, spec.size), rng.uniform(0.0, spec.size));
            let (cx, cy) = (cell_of(x), cell_of(y));
            let free = (cy.saturating_sub(1)..=(cy + 1).min(nc - 1)).all(|gy| (cx.saturating_sub(1)..=(cx + 1).min(nc - 1)).all(|gx| grid[gy * nc + gx].iter().all(|&j| {
                let (px, py, pd) = placed[j];
                ((x - px).powi(2) + (y - py).powi(2)).sqrt() >= 0.75 * (d + pd) + 0.2
            })));
            if free {
                ok = Some((x, y));
                grid[cy * nc + cx].push(placed.len());
                break;
            }
        }
        let (x, y) = ok.ok_or_else(too_dense)?;
        placed.push((x, y, d));
    }
    let mut specs: Vec<(TreeSpec, f64)> = Vec::with_capacity(placed.len());
    for &(x, y, d) in &placed {
        let arch = pick(&mut rng, &spec.archetypes);
        let (a, b, c) = arch.form().allometry;
        let h = (1.3 + a * (1.0 - (-b * d * 100.0).exp()).powf(c)) * (spec.height_noise * rng.normal(0.0, 1.0)).exp();
        let h = h.max(2.0).max(5.0 * d);
        let (gx, gy) = terrain.gradient(x, y);
        // Sink the stem base so that the downhill side reaches the ground.
        let sink = (gx.hypot(gy)) * 0.6 * d * 1.5 + 0.02;
        let z = terrain.height(x, y);
        let mut ts = TreeSpec::new(arch, [x, y, z - sink], d, h + sink, rng.next_u64());
        ts.breast_height = 1.3 + sink;
        ts.crown_base = Some(arch.form().crown_base * h + sink);
        ts.point_density = spec.point_density;
        ts.max_order = spec.max_order;
        ts.lad = spec.lad;
        ts.epicormic = spec.epicormic;
        specs.push((ts, sink));
    }
    // ---- understorey, dead wood, grass positions and seeds (drawn before growing)
    let n_shrubs = (spec.shrubs * area / 1e4).round() as usize;
    let near_stem = |x: f64, y: f64, r: f64| placed.iter().any(|&(px, py, pd)| ((x - px).powi(2) + (y - py).powi(2)).sqrt() < r + pd);
    let mut shrub_specs = Vec::new();
    for _ in 0..n_shrubs {
        let (mut x, mut y) = (rng.uniform(0.0, spec.size), rng.uniform(0.0, spec.size));
        for _ in 0..100 {
            if !near_stem(x, y, 0.6) {
                break;
            }
            (x, y) = (rng.uniform(0.0, spec.size), rng.uniform(0.0, spec.size));
        }
        let h = rng.uniform(0.5, 2.5);
        let d = (0.012 * h).max(0.008);
        let mut ts = TreeSpec::new(Archetype::Shrub, [x, y, terrain.height(x, y) - 0.02], d, h, rng.next_u64());
        ts.point_density = spec.point_density;
        ts.max_order = 2.min(spec.max_order);
        ts.lad = spec.lad;
        ts.leaf_area = Some(Archetype::Shrub.form().lai * PI * (0.4 * h).powi(2) * 0.5);
        shrub_specs.push(ts);
    }
    // ---- grow the trees and shrubs in parallel
    let grown: Vec<Result<SynthTree>> = specs.par_iter().map(|(s, _)| tree_model(s)).collect();
    let shrubs_grown: Vec<Result<SynthTree>> = shrub_specs.par_iter().map(tree_model).collect();

    let mut xyz: Vec<Point> = Vec::new();
    let mut nrm: Vec<Point> = Vec::new();
    let (mut cls, mut label, mut tree_id, mut order, mut cyl, mut leaf, mut epi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    // Ground.
    let lo = -spec.margin;
    let hi = spec.size + spec.margin;
    let n_ground = spec.ground_density * (hi - lo) * (hi - lo);
    limits::check_cells(n_ground as u128, 24, &format!("{n_ground:.0} ground points"), "a lower ground_density")?;
    for [x, y] in jittered_grid(&mut rng, n_ground, hi - lo, hi - lo) {
        let (x, y) = (x + lo, y + lo);
        xyz.push([x, y, terrain.height(x, y)]);
        let (gx, gy) = terrain.gradient(x, y);
        let n = (gx * gx + gy * gy + 1.0).sqrt();
        nrm.push([-gx / n, -gy / n, 1.0 / n]);
    }
    let n_ground = xyz.len();
    let fill = |v: &mut Vec<i32>, n: usize, x: i32| v.extend(std::iter::repeat_n(x, n));
    cls.extend(std::iter::repeat_n(2u8, n_ground));
    label.extend(std::iter::repeat_n(LABEL_GROUND, n_ground));
    fill(&mut tree_id, n_ground, 0);
    fill(&mut order, n_ground, -1);
    fill(&mut cyl, n_ground, -1);
    fill(&mut leaf, n_ground, -1);
    epi.extend(std::iter::repeat_n(0u8, n_ground));

    let mut trees = Vec::new();
    let (mut cylinders, mut cylinder_tree, mut leaves, mut leaf_tree) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, (t, (ts, sink))) in grown.into_iter().zip(&specs).enumerate() {
        let t = t?;
        let id = i as i32 + 1;
        let (c0, l0) = (cylinders.len() as i32, leaves.len() as i32);
        let a = |name: &str| t.cloud.attr(name).expect("tree attribute");
        let (Attr::U8(tc), Attr::U8(tl), Attr::I32(to), Attr::I32(tcy), Attr::I32(tle), Attr::U8(te)) = (a("classification"), a("label"), a("branch_order"), a("cylinder"), a("leaf"), a("epicormic")) else { unreachable!() };
        let tn = tree_normals(&t.cloud);
        let mut counts = vec![0usize; t.qsm.cylinders.len()];
        for (k, p) in t.cloud.xyz.iter().enumerate() {
            if p[2] < terrain.height(p[0], p[1]) {
                continue;
            }
            xyz.push(*p);
            nrm.push(tn[k]);
            cls.push(tc[k]);
            label.push(tl[k]);
            tree_id.push(id);
            order.push(to[k]);
            cyl.push(if tcy[k] >= 0 { counts[tcy[k] as usize] += 1; tcy[k] + c0 } else { -1 });
            leaf.push(if tle[k] >= 0 { tle[k] + l0 } else { -1 });
            epi.push(te[k]);
        }
        let stem_volume: f64 = t.qsm.cylinders.iter().filter(|c| c.branch_order == 0).map(|c| c.volume()).sum();
        let wood_volume = t.wood_volume();
        trees.push(PlotTree { id, archetype: t.archetype, base: [ts.base[0], ts.base[1], ts.base[2] + sink], stem_bh: t.stem_bh, dbh: t.dbh, height: t.height - sink, crown_base: t.crown_base - sink, crown_area: t.crown_area, wood_volume, stem_volume, branch_volume: wood_volume - stem_volume, leaf_area: t.leaf_area });
        for (c, n) in t.qsm.cylinders.iter().zip(counts) {
            cylinders.push(Cylinder { n_points: n, ..c.clone() });
            cylinder_tree.push(id);
        }
        leaf_tree.extend(std::iter::repeat_n(id, t.leaves.len()));
        leaves.extend(t.leaves.into_iter().map(|mut l| {
            l.cylinder += c0 as usize;
            l
        }));
    }
    let mut shrubs = Vec::new();
    for (t, ts) in shrubs_grown.into_iter().zip(&shrub_specs) {
        let t = t?;
        let tn = tree_normals(&t.cloud);
        for (k, p) in t.cloud.xyz.iter().enumerate() {
            if p[2] < terrain.height(p[0], p[1]) {
                continue;
            }
            xyz.push(*p);
            nrm.push(tn[k]);
            cls.push(3);
            label.push(LABEL_UNDERSTOREY);
        }
        let n = xyz.len() - tree_id.len();
        fill(&mut tree_id, n, 0);
        fill(&mut order, n, -1);
        fill(&mut cyl, n, -1);
        fill(&mut leaf, n, -1);
        epi.extend(std::iter::repeat_n(0u8, n));
        shrubs.push([ts.base[0], ts.base[1], t.height, t.leaf_area]);
    }
    // ---- grass, logs and stumps
    let mut part = Part::new();
    let tuft_r = 0.15;
    let n_tufts = (spec.grass_cover * area / (PI * tuft_r * tuft_r)).round() as usize;
    let mut grass = Vec::new();
    for _ in 0..n_tufts {
        let (cx, cy) = (rng.uniform(0.0, spec.size), rng.uniform(0.0, spec.size));
        if near_stem(cx, cy, 0.3) {
            continue;
        }
        grass.push([cx, cy, tuft_r]);
        let blades = 12 + (rng.random() * 12.0) as usize;
        for _ in 0..blades {
            let (r, a) = (0.5 * tuft_r * rng.random().sqrt(), rng.uniform(0.0, 2.0 * PI));
            let (bx, by) = (cx + r * a.cos(), cy + r * a.sin());
            let bz = terrain.height(bx, by);
            let len = spec.grass_height * rng.uniform(0.4, 1.0);
            let lean = rng.uniform(0.1, 0.7);
            let az = a + rng.uniform(-0.5, 0.5);
            let width = 0.005;
            let n = random_count(&mut rng, spec.point_density * len * width);
            for _ in 0..n {
                let s = rng.random();
                // The blade bends outwards: its angle from the vertical grows along it.
                let ang = lean * (1.0 + 1.5 * s);
                let horiz = len * s * ang.sin();
                let w = rng.uniform(-0.5, 0.5) * width;
                let p = [bx + horiz * az.cos() - w * az.sin(), by + horiz * az.sin() + w * az.cos(), bz + len * s * ang.cos()];
                // Normal: across the blade's width and its tangent.
                let n = [-ang.cos() * az.cos(), -ang.cos() * az.sin(), ang.sin()];
                if p[2] >= terrain.height(p[0], p[1]) {
                    part.push(p, n, 3, LABEL_UNDERSTOREY);
                }
            }
        }
    }
    let mut dead_wood = Vec::new();
    let n_logs = (spec.logs * area / 1e4).round() as usize;
    for _ in 0..n_logs {
        let len = rng.uniform(2.0, 8.0);
        let radius = rng.uniform(0.05, 0.25);
        let (cx, cy, az) = (rng.uniform(0.0, spec.size), rng.uniform(0.0, spec.size), rng.uniform(0.0, 2.0 * PI));
        let (dx, dy) = (0.5 * len * az.cos(), 0.5 * len * az.sin());
        let a = [cx - dx, cy - dy, terrain.height(cx - dx, cy - dy) + radius * 0.8];
        let b = [cx + dx, cy + dy, terrain.height(cx + dx, cy + dy) + radius * 0.8];
        let v = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let axis = [v[0] / l, v[1] / l, v[2] / l];
        cylinder_points(&mut rng, &mut part, &terrain, a, axis, l, radius, spec.point_density, false);
        dead_wood.push(DeadWood { kind: 0, start: a, axis, length: l, radius });
    }
    let n_stumps = (spec.stumps * area / 1e4).round() as usize;
    for _ in 0..n_stumps {
        let radius = rng.uniform(0.1, 0.3);
        let (mut x, mut y) = (rng.uniform(0.0, spec.size), rng.uniform(0.0, spec.size));
        for _ in 0..100 {
            if !near_stem(x, y, radius + 0.3) {
                break;
            }
            (x, y) = (rng.uniform(0.0, spec.size), rng.uniform(0.0, spec.size));
        }
        let h = rng.uniform(0.2, 1.0);
        let z = terrain.height(x, y);
        let start = [x, y, z - 0.1];
        cylinder_points(&mut rng, &mut part, &terrain, start, [0.0, 0.0, 1.0], h + 0.1, radius, spec.point_density, true);
        dead_wood.push(DeadWood { kind: 1, start, axis: [0.0, 0.0, 1.0], length: h + 0.1, radius });
    }
    let n = part.xyz.len();
    xyz.extend(part.xyz);
    nrm.extend(part.nrm);
    cls.extend(part.cls);
    label.extend(part.label);
    fill(&mut tree_id, n, 0);
    fill(&mut order, n, -1);
    fill(&mut cyl, n, -1);
    fill(&mut leaf, n, -1);
    epi.extend(std::iter::repeat_n(0u8, n));

    let mut cloud = PointCloud::new(xyz);
    cloud.attrs.insert("classification".into(), Attr::U8(cls));
    cloud.attrs.insert("label".into(), Attr::U8(label));
    cloud.attrs.insert("tree_id".into(), Attr::I32(tree_id));
    cloud.attrs.insert("branch_order".into(), Attr::I32(order));
    cloud.attrs.insert("cylinder".into(), Attr::I32(cyl));
    cloud.attrs.insert("leaf".into(), Attr::I32(leaf));
    cloud.attrs.insert("epicormic".into(), Attr::U8(epi));
    insert_normals(&mut cloud, &nrm);
    Ok(Plot { cloud, trees, cylinders, cylinder_tree, leaves, leaf_tree, dead_wood, shrubs, grass, terrain })
}

fn tree_normals(c: &PointCloud) -> Vec<Point> {
    let get = |name: &str| match c.attr(name) {
        Some(Attr::F32(v)) => v.clone(),
        _ => vec![0.0; c.len()],
    };
    let (x, y, z) = (get("normal_x"), get("normal_y"), get("normal_z"));
    (0..c.len()).map(|i| [x[i] as f64, y[i] as f64, z[i] as f64]).collect()
}

fn random_count(rng: &mut Generator, x: f64) -> usize {
    let f = x.max(0.0).floor();
    f as usize + usize::from(rng.random() < x - f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(seed: u64) -> PlotSpec {
        let mut s = PlotSpec::new(seed);
        s.size = 12.0;
        s.density = 400.0;
        s.point_density = 60.0;
        s.ground_density = 20.0;
        s.max_order = 2;
        s
    }

    #[test]
    fn a_small_plot_and_its_truth() {
        let p = plot(&small(1)).unwrap();
        assert_eq!(p.trees.len(), (400.0 * 144.0 / 1e4f64).round() as usize);
        let Some(Attr::U8(label)) = p.cloud.attr("label") else { panic!() };
        for l in 1..=6u8 {
            assert!(label.contains(&l), "label {l} missing");
        }
        // No two stems overlap at the base.
        for (i, a) in p.trees.iter().enumerate() {
            for b in &p.trees[i + 1..] {
                assert!((a.base[0] - b.base[0]).hypot(a.base[1] - b.base[1]) > 0.5 * (a.dbh + b.dbh));
            }
            assert!(a.dbh >= 0.07 - 1e-9 && a.height > 2.0);
        }
        // Nothing lies below the terrain.
        assert!(p.cloud.xyz.iter().all(|q| q[2] >= p.terrain.height(q[0], q[1]) - 1e-9));
        // Table sums.
        let v: f64 = p.cylinders.iter().zip(&p.cylinder_tree).filter(|(_, &t)| t == 1).map(|(c, _)| c.volume()).sum();
        assert!((v - p.trees[0].wood_volume).abs() < 1e-12);
    }

    #[test]
    fn terrain_roughness_has_its_spread() {
        let t = Terrain::new(0.0, 0.0, 0.1, 1.0, 5);
        let z: Vec<f64> = (0..200).flat_map(|i| (0..200).map(move |j| (i as f64 * 0.5, j as f64 * 0.5))).map(|(x, y)| t.height(x, y)).collect();
        let m = z.iter().sum::<f64>() / z.len() as f64;
        let sd = (z.iter().map(|v| (v - m).powi(2)).sum::<f64>() / z.len() as f64).sqrt();
        assert!((sd - 0.1).abs() < 0.025, "{sd}");
        let t = Terrain::new(0.2, 90.0, 0.0, 1.0, 5);
        assert!((t.height(10.0, 3.0) - 2.0).abs() < 1e-12);
        let (gx, gy) = t.gradient(1.0, 1.0);
        assert!((gx - 0.2).abs() < 1e-12 && gy.abs() < 1e-12);
    }

    #[test]
    fn deterministic_and_checked() {
        let a = plot(&small(2)).unwrap();
        let b = plot(&small(2)).unwrap();
        assert_eq!(a.cloud.xyz, b.cloud.xyz);
        let mut s = small(2);
        s.grass_cover = 2.0;
        assert!(plot(&s).is_err());
        let mut s = small(2);
        s.density = 1e6;
        assert!(plot(&s).is_err());
    }
}
