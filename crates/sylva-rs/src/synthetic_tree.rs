// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Synthetic trees with a known architecture, and their truth.
//!
//! A tree is grown from an archetype (a broadleaf, a conifer with a central
//! leader and whorls, a eucalypt with a clear bole and a sparse crown, a
//! savanna tree with a low forked crown, or a shrub):
//!
//! - **Stem.** The axis leans (`lean_deg` towards a random azimuth) and
//!   curves smoothly (three sinusoids per horizontal component, slope
//!   amplitude `sweep`). Below the lowest branch the radius follows
//!   `r(s) = R g(s) / g(s_ref)` along the stem length `s`, with
//!   `g(s) = (1 + b exp(-s / 0.6)) ((S - s) / (S - s_ref))^p` (butt swell
//!   `b`, taper exponent `p`, stem length `S`, `s_ref` 1.3 m), so that the
//!   diameter at `s_ref` is `dbh`. The cross-section can be elliptical, carry
//!   bark fissures of a given depth (both rescaled to keep the area of the
//!   round section) and buttress flanges below `buttress_height`, which add
//!   area.
//! - **Branches.** First-order branches leave the stem from the crown base
//!   up (in whorls for the conifer, spirally at 137.5 degrees otherwise),
//!   their length set by the crown envelope of the archetype at their
//!   height; forking archetypes end the stem in codominant limbs. Each axis
//!   of order `k` carries children of order `k + 1` up to `max_order`, their
//!   length a fixed ratio of the parent's (shorter towards its tip), at the
//!   archetype's branching angle, alternately to either side. Axes bend
//!   towards the vertical (or droop) by a tropism per metre.
//! - **Radii: pipe model** (Shinozaki et al. 1964). Above the lowest branch
//!   the cross-sectional area of every segment is proportional to the leaf
//!   area it carries, `r = c W^(1/e)` with `e = pipe_exponent` (2 for the
//!   pipe model) and `c` set so the radius is continuous with the stem
//!   taper where the crown begins.
//! - **Leaves** are planar elliptical blades of known length and width
//!   (area `π L W / 4`) on the distal 80 % of the terminal twigs, their
//!   number set by the leaf area (by default the archetype's crown leaf area
//!   index times the crown's projected area), their normals drawn from a
//!   leaf angle distribution ([`Lad`]: the de Wit types, ellipsoidal or
//!   Goel and Strebel's two-parameter beta) and uniform in azimuth.
//! - **Epicormic shoots** (optional) are short leafy shoots along the bole
//!   below the crown, as after fire.
//!
//! The wood is a table of cylinders (a [`Qsm`]); points are drawn on exactly
//! those cylinders (on the shaped section for the stem), so the table's
//! volume is the volume of the sampled surface. Leaf points are drawn on the
//! blades. Random numbers come from [`Generator`], so a seed gives the same
//! tree on every machine and thread count.

use std::f64::consts::PI;

use crate::error::{Error, Result};
use crate::nprandom::Generator;
use crate::pointcloud::Attr;
use crate::qsm::{Cylinder, Qsm};
use crate::voxel::Lad;
use crate::{Point, PointCloud};

/// Point labels of the synthetic scenes (`label` attribute).
pub const LABEL_GROUND: u8 = 1;
pub const LABEL_STEM: u8 = 2;
pub const LABEL_BRANCH: u8 = 3;
pub const LABEL_LEAF: u8 = 4;
pub const LABEL_UNDERSTOREY: u8 = 5;
pub const LABEL_DEAD_WOOD: u8 = 6;

/// Samples of a stem cross-section's radius around the axis.
const SECTION: usize = 256;

/// Tree forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archetype {
    Broadleaf,
    Conifer,
    Eucalypt,
    Savanna,
    Shrub,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Envelope {
    Cone,
    Ellipsoid,
    Sparse,
    Flat,
}

/// Architectural constants of an archetype. Lengths that set the density
/// of branching are for a 20 m tree and scale with height.
#[derive(Debug, Clone)]
pub struct Form {
    /// Height of the lowest first-order branch, fraction of the height.
    pub crown_base: f64,
    /// Crown radius, fraction of the height.
    pub crown_radius: f64,
    /// Where the stem ends (and forks), fraction of the height.
    pub leader: f64,
    forks: usize,
    fork_angle: f64,
    whorl: usize,
    internode: f64,
    /// Branching angle from the parent axis (degrees) of orders 1 to 4.
    angle: [f64; 4],
    /// Bending towards the vertical per metre of orders 1 to 4 (negative droops).
    tropism: [f64; 4],
    /// Child length over parent length for orders 2 to 4 (index 0 unused).
    ratio: [f64; 4],
    /// Spacing of children along axes of order 1 to 3 (m, 20 m tree).
    spacing: [f64; 3],
    envelope: Envelope,
    /// Leaf area per unit of crown projected area.
    pub lai: f64,
    /// Leaf area per squared DBH (m² m⁻²): the pipe model's leaf area to
    /// sapwood area, as an allometry.
    pub leaf_k: f64,
    /// Crown radius allometry `crown_k (0.6 + 12 dbh)` (m).
    pub crown_k: f64,
    /// Height-diameter allometry `1.3 + a (1 - exp(-b D))^c`, D in cm.
    pub allometry: (f64, f64, f64),
    /// Leaf blade length and width (m).
    pub leaf: (f64, f64),
    /// Default leaf angle distribution.
    pub lad: &'static str,
    taper: f64,
    /// Relative extra radius at the ground.
    pub butt_swell: f64,
    /// Default lean (degrees).
    pub lean: f64,
    /// Default slope amplitude of the stem's curvature.
    pub sweep: f64,
}

impl Archetype {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "broadleaf" => Ok(Archetype::Broadleaf),
            "conifer" => Ok(Archetype::Conifer),
            "eucalypt" => Ok(Archetype::Eucalypt),
            "savanna" => Ok(Archetype::Savanna),
            "shrub" => Ok(Archetype::Shrub),
            _ => Err(Error::invalid(format!("unknown archetype {name:?}; expected 'broadleaf', 'conifer', 'eucalypt', 'savanna' or 'shrub'"))),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Archetype::Broadleaf => "broadleaf",
            Archetype::Conifer => "conifer",
            Archetype::Eucalypt => "eucalypt",
            Archetype::Savanna => "savanna",
            Archetype::Shrub => "shrub",
        }
    }

    pub fn form(&self) -> Form {
        match self {
            Archetype::Broadleaf => Form { leaf_k: 1500.0, crown_k: 1.0, allometry: (28.0, 0.035, 1.0), crown_base: 0.35, crown_radius: 0.25, leader: 0.8, forks: 2, fork_angle: 20.0, whorl: 1, internode: 0.45, angle: [55.0, 45.0, 40.0, 40.0], tropism: [0.12, 0.08, 0.04, 0.0], ratio: [0.0, 0.5, 0.45, 0.4], spacing: [0.45, 0.2, 0.08], envelope: Envelope::Ellipsoid, lai: 4.0, leaf: (0.10, 0.05), lad: "planophile", taper: 0.8, butt_swell: 0.25, lean: 2.0, sweep: 0.02 },
            Archetype::Conifer => Form { leaf_k: 2000.0, crown_k: 0.5, allometry: (35.0, 0.03, 1.1), crown_base: 0.25, crown_radius: 0.14, leader: 1.0, forks: 0, fork_angle: 0.0, whorl: 5, internode: 0.5, angle: [80.0, 55.0, 45.0, 40.0], tropism: [-0.05, 0.0, 0.0, 0.0], ratio: [0.0, 0.4, 0.4, 0.35], spacing: [0.3, 0.12, 0.05], envelope: Envelope::Cone, lai: 6.0, leaf: (0.05, 0.012), lad: "spherical", taper: 1.0, butt_swell: 0.15, lean: 1.0, sweep: 0.01 },
            Archetype::Eucalypt => Form { leaf_k: 600.0, crown_k: 0.8, allometry: (40.0, 0.025, 1.0), crown_base: 0.55, crown_radius: 0.2, leader: 0.65, forks: 3, fork_angle: 35.0, whorl: 1, internode: 1.2, angle: [45.0, 40.0, 40.0, 40.0], tropism: [0.1, 0.05, 0.0, -0.1], ratio: [0.0, 0.6, 0.5, 0.4], spacing: [0.7, 0.3, 0.12], envelope: Envelope::Sparse, lai: 1.5, leaf: (0.15, 0.03), lad: "erectophile", taper: 0.7, butt_swell: 0.3, lean: 4.0, sweep: 0.03 },
            Archetype::Savanna => Form { leaf_k: 900.0, crown_k: 1.6, allometry: (12.0, 0.05, 1.0), crown_base: 0.2, crown_radius: 0.5, leader: 0.3, forks: 3, fork_angle: 45.0, whorl: 1, internode: 0.8, angle: [60.0, 50.0, 45.0, 45.0], tropism: [-0.02, -0.05, 0.0, 0.0], ratio: [0.0, 0.5, 0.45, 0.4], spacing: [0.5, 0.2, 0.08], envelope: Envelope::Flat, lai: 1.8, leaf: (0.04, 0.015), lad: "planophile", taper: 0.6, butt_swell: 0.3, lean: 6.0, sweep: 0.04 },
            Archetype::Shrub => Form { leaf_k: 3000.0, crown_k: 1.0, allometry: (3.0, 0.3, 1.0), crown_base: 0.1, crown_radius: 0.4, leader: 0.08, forks: 4, fork_angle: 25.0, whorl: 1, internode: 0.3, angle: [50.0, 45.0, 40.0, 40.0], tropism: [0.2, 0.1, 0.0, 0.0], ratio: [0.0, 0.5, 0.45, 0.4], spacing: [0.2, 0.08, 0.04], envelope: Envelope::Ellipsoid, lai: 3.0, leaf: (0.05, 0.025), lad: "spherical", taper: 0.6, butt_swell: 0.1, lean: 5.0, sweep: 0.05 },
        }
    }
}

/// Settings of [`tree_model`]. `None` takes the archetype's value.
#[derive(Debug, Clone)]
pub struct TreeSpec {
    pub archetype: Archetype,
    /// Stem base.
    pub base: Point,
    /// Diameter at 1.3 m along the stem (m); for trees shorter than 2.6 m,
    /// at half the stem length.
    pub dbh: f64,
    /// Height (m); the realised height is in [`SynthTree::height`].
    pub height: f64,
    /// Crown radius (m).
    pub crown_radius: Option<f64>,
    /// Height of the lowest first-order branch above the base (m).
    pub crown_base: Option<f64>,
    /// Highest branch order (1 to 4).
    pub max_order: u32,
    pub lean_deg: Option<f64>,
    pub sweep: Option<f64>,
    pub butt_swell: Option<f64>,
    /// Number of buttress flanges (0 for none).
    pub buttresses: usize,
    /// Height the flanges reach (m).
    pub buttress_height: f64,
    /// Relative extra radius of a flange at the ground.
    pub buttress_extent: f64,
    /// `a / r - 1` of the stem section (0 for round).
    pub ellipticity: f64,
    /// Depth of bark fissures (m).
    pub bark_depth: f64,
    /// Total one-sided leaf area of the crown (m²).
    pub leaf_area: Option<f64>,
    /// Leaf blade length and width (m).
    pub leaf_size: Option<(f64, f64)>,
    /// Leaf angle distribution; the archetype's by default.
    pub lad: Option<Lad>,
    /// Epicormic shoots per metre of bole (0 for none).
    pub epicormic: f64,
    /// Points per m² of wood surface and of leaf (one side).
    pub point_density: f64,
    pub pipe_exponent: f64,
    /// Smallest radius of a twig (m).
    pub min_radius: f64,
    /// Distance along the stem at which `dbh` applies (m).
    pub breast_height: f64,
    pub seed: u64,
}

impl TreeSpec {
    pub fn new(archetype: Archetype, base: Point, dbh: f64, height: f64, seed: u64) -> Self {
        TreeSpec { archetype, base, dbh, height, crown_radius: None, crown_base: None, max_order: 3, lean_deg: None, sweep: None, butt_swell: None, buttresses: 0, buttress_height: 1.0, buttress_extent: 0.5, ellipticity: 0.0, bark_depth: 0.0, leaf_area: None, leaf_size: None, lad: None, epicormic: 0.0, point_density: 2000.0, pipe_exponent: 2.0, min_radius: 0.0015, breast_height: 1.3, seed }
    }

    fn check(&self) -> Result<()> {
        let positive = [("dbh", self.dbh), ("height", self.height), ("breast_height", self.breast_height), ("point_density", self.point_density), ("pipe_exponent", self.pipe_exponent), ("min_radius", self.min_radius)];
        for (name, v) in positive {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::invalid(format!("{name} must be a positive number, got {v}")));
            }
        }
        let non_negative = [("buttress_height", self.buttress_height), ("buttress_extent", self.buttress_extent), ("ellipticity", self.ellipticity), ("bark_depth", self.bark_depth), ("epicormic", self.epicormic)];
        for (name, v) in non_negative {
            if !(v.is_finite() && v >= 0.0) {
                return Err(Error::invalid(format!("{name} must be zero or more, got {v}")));
            }
        }
        for (name, v) in [("crown_radius", self.crown_radius), ("leaf_area", self.leaf_area), ("lean_deg", self.lean_deg), ("sweep", self.sweep), ("butt_swell", self.butt_swell), ("crown_base", self.crown_base)] {
            if let Some(v) = v {
                if !(v.is_finite() && v >= 0.0) {
                    return Err(Error::invalid(format!("{name} must be zero or more, got {v}")));
                }
            }
        }
        if let Some((l, w)) = self.leaf_size {
            if !(l.is_finite() && w.is_finite() && l > 0.0 && w > 0.0) {
                return Err(Error::invalid(format!("leaf_size must be two positive lengths, got ({l}, {w})")));
            }
        }
        if !(1..=4).contains(&self.max_order) {
            return Err(Error::invalid(format!("max_order must be 1 to 4, got {}", self.max_order)));
        }
        if self.ellipticity >= 0.9 {
            return Err(Error::invalid(format!("ellipticity must be below 0.9, got {}", self.ellipticity)));
        }
        if self.dbh > self.height {
            return Err(Error::invalid(format!("dbh ({}) must be smaller than the height ({})", self.dbh, self.height)));
        }
        if !self.base.iter().all(|v| v.is_finite()) {
            return Err(Error::invalid("the base position must be finite"));
        }
        Ok(())
    }
}

/// A leaf of a synthetic tree: an elliptical blade.
#[derive(Debug, Clone, PartialEq)]
pub struct Leaf {
    pub centre: Point,
    /// Unit normal (pointing up).
    pub normal: Point,
    /// Unit direction of the blade's length (from the petiole to the tip).
    pub axis: Point,
    pub length: f64,
    pub width: f64,
    /// Row of the cylinder the leaf is attached to.
    pub cylinder: usize,
    pub epicormic: bool,
}

impl Leaf {
    /// One-sided area of the blade, `π L W / 4`.
    pub fn area(&self) -> f64 {
        PI / 4.0 * self.length * self.width
    }
}

/// A generated tree and its truth.
#[derive(Debug, Clone)]
pub struct SynthTree {
    /// Points with `classification` (4 leaf, 5 wood), `label` (2 stem, 3
    /// branch, 4 leaf), `branch_order` (of the cylinder, or of the twig a
    /// leaf hangs from), `cylinder` (row in `qsm`, -1 for leaves), `leaf`
    /// (row in `leaves`, -1 for wood) and `epicormic` (1 on epicormic
    /// shoots and their leaves).
    pub cloud: PointCloud,
    /// The wood as cylinders; `n_points` is the number of points drawn on each.
    pub qsm: Qsm,
    /// Whether each cylinder is part of an epicormic shoot.
    pub cylinder_epicormic: Vec<bool>,
    pub leaves: Vec<Leaf>,
    pub archetype: Archetype,
    pub base: Point,
    /// Stem axis at breast height (`s_ref` along the stem).
    pub stem_bh: Point,
    /// Area-equivalent diameter of the stem section at `s_ref` (m).
    pub dbh: f64,
    /// Highest point of wood or leaf above the base (m).
    pub height: f64,
    /// Height above the base of the lowest first-order branch that is not epicormic (m).
    pub crown_base: f64,
    /// Area of the convex hull of the crown seen from above (m²).
    pub crown_area: f64,
    /// `(xmin, ymin, xmax, ymax)` of the crown.
    pub crown_extent: [f64; 4],
    pub leaf_area: f64,
    pub epicormic_leaf_area: f64,
    pub lad: Lad,
}

impl SynthTree {
    pub fn wood_volume(&self) -> f64 {
        self.qsm.cylinders.iter().map(|c| c.volume()).sum()
    }
}

// ------------------------------------------------------------ vectors

fn add(a: &Point, b: &Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: &Point, s: f64) -> Point {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: &Point, b: &Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &Point, b: &Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: &Point) -> Point {
    let n = dot(a, a).sqrt();
    if n > 0.0 { scale(a, 1.0 / n) } else { [0.0, 0.0, 1.0] }
}

/// Two unit vectors perpendicular to `d` and to each other; the first is
/// horizontal unless `d` is vertical.
fn basis(d: &Point) -> (Point, Point) {
    let h = cross(&[0.0, 0.0, 1.0], d);
    let e1 = if dot(&h, &h) > 1e-12 { unit(&h) } else { [1.0, 0.0, 0.0] };
    (e1, cross(d, &e1))
}

/// Stem section frame: `u` is x projected off the axis (y if the axis is along x).
fn section_frame(d: &Point) -> (Point, Point) {
    let x = if d[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = unit(&add(&x, &scale(d, -dot(&x, d))));
    (u, cross(d, &u))
}

/// `floor(x)` plus one with probability `frac(x)`.
fn random_round(rng: &mut Generator, x: f64) -> usize {
    let f = x.max(0.0).floor();
    f as usize + usize::from(rng.random() < x - f)
}

/// About `n` points (a random rounding) spread over `[0, w] x [0, h]` on a
/// jittered grid of nearly square cells, one uniform point per cell: a
/// surface sample without the holes and clumps of independent points, so
/// that a ray-cast scan of it sees a closed surface.
pub fn jittered_grid(rng: &mut Generator, n: f64, w: f64, h: f64) -> Vec<[f64; 2]> {
    let n = random_round(rng, n);
    if n == 0 || !(w > 0.0 && h > 0.0) {
        return Vec::new();
    }
    let nw = ((n as f64 * w / h).sqrt().round() as usize).clamp(1, n);
    let nh = (n as f64 / nw as f64).round().max(1.0) as usize;
    let (cw, ch) = (w / nw as f64, h / nh as f64);
    let mut out = Vec::with_capacity(nw * nh);
    for j in 0..nh {
        for i in 0..nw {
            out.push([(i as f64 + rng.random()) * cw, (j as f64 + rng.random()) * ch]);
        }
    }
    out
}

/// Inverse cumulative table of a leaf angle distribution on `[0, π/2]`.
pub struct LadSampler {
    cdf: Vec<f64>,
}

impl LadSampler {
    const BINS: usize = 4096;

    pub fn new(lad: &Lad) -> Self {
        let h = PI / 2.0 / Self::BINS as f64;
        let mut cdf = Vec::with_capacity(Self::BINS + 1);
        cdf.push(0.0);
        let mut acc = 0.0;
        for i in 0..Self::BINS {
            // Simpson's rule on each bin, safe at the end points of the beta density.
            let a = i as f64 * h;
            let f = |t: f64| {
                let v = lad.pdf(t.clamp(1e-9, PI / 2.0 - 1e-9));
                if v.is_finite() { v.max(0.0) } else { 0.0 }
            };
            acc += h / 6.0 * (f(a) + 4.0 * f(a + h / 2.0) + f(a + h));
            cdf.push(acc);
        }
        let total = acc;
        for v in &mut cdf {
            *v /= total;
        }
        LadSampler { cdf }
    }

    /// Leaf inclination (rad) for a uniform draw `u`.
    pub fn inclination(&self, u: f64) -> f64 {
        let k = self.cdf.partition_point(|&c| c < u).clamp(1, Self::BINS);
        let (c0, c1) = (self.cdf[k - 1], self.cdf[k]);
        let f = if c1 > c0 { (u - c0) / (c1 - c0) } else { 0.5 };
        ((k - 1) as f64 + f) * PI / 2.0 / Self::BINS as f64
    }
}

// ------------------------------------------------------------ growth

struct Seg {
    start: Point,
    dir: Point,
    len: f64,
    parent: i64,
    order: u32,
    axis: usize,
    /// Distance along the stem of the segment's middle (stem only).
    s_mid: f64,
    radius: f64,
    /// Leaf area hanging from the segment.
    leaf_w: f64,
    /// Stem cross-section radii at `SECTION` angles (stem only).
    section: Vec<f64>,
}

struct AxisRec {
    order: u32,
    segs: Vec<usize>,
    length: f64,
    epicormic: bool,
    has_children: bool,
}

struct Grower {
    segs: Vec<Seg>,
    axes: Vec<AxisRec>,
}

impl Grower {
    /// Grow an axis of `length` from `start` along `dir`; its segments are
    /// appended and the axis is returned by index.
    #[allow(clippy::too_many_arguments)]
    fn grow(&mut self, rng: &mut Generator, start: Point, dir: Point, length: f64, order: u32, parent: i64, epicormic: bool, tropism: f64) -> usize {
        let step = if order <= 1 { 0.25 } else { 0.1 };
        let n = ((length / step).ceil() as usize).clamp(if order <= 1 { 2 } else { 1 }, if order <= 1 { 12 } else { 6 });
        let dl = length / n as f64;
        let id = self.axes.len();
        let mut rec = AxisRec { order, segs: Vec::with_capacity(n), length, epicormic, has_children: false };
        let (mut p, mut d, mut par) = (start, unit(&dir), parent);
        for _ in 0..n {
            let k = self.segs.len();
            self.segs.push(Seg { start: p, dir: d, len: dl, parent: par, order, axis: id, s_mid: 0.0, radius: 0.0, leaf_w: 0.0, section: Vec::new() });
            rec.segs.push(k);
            par = k as i64;
            p = add(&p, &scale(&d, dl));
            let w = 0.05 * dl.sqrt();
            let jitter = [rng.normal(0.0, w), rng.normal(0.0, w), rng.normal(0.0, w)];
            let mut nd = add(&add(&d, &[0.0, 0.0, tropism * dl]), &jitter);
            // Do not let limbs grow steeply into the ground.
            nd[2] = nd[2].max(-0.5 * (nd[0] * nd[0] + nd[1] * nd[1]).sqrt());
            d = unit(&nd);
        }
        self.axes.push(rec);
        id
    }

    /// Position and segment at distance `s` along axis `a`.
    fn along(&self, a: usize, s: f64) -> (Point, usize) {
        let rec = &self.axes[a];
        let mut acc = 0.0;
        for &k in &rec.segs {
            let sg = &self.segs[k];
            if s <= acc + sg.len || k == *rec.segs.last().unwrap() {
                let t = (s - acc).clamp(0.0, sg.len);
                return (add(&sg.start, &scale(&sg.dir, t)), k);
            }
            acc += sg.len;
        }
        unreachable!("axes have segments")
    }
}

/// Crown radius profile of an archetype at relative height `q` in the crown.
fn envelope(e: Envelope, q: f64) -> f64 {
    let q = q.clamp(0.0, 1.0);
    match e {
        Envelope::Cone => 0.08 + 0.92 * (1.0 - q).powf(0.9),
        Envelope::Ellipsoid => 0.15 + 0.85 * (1.0 - (2.0 * q - 1.0).powi(2)).max(0.0).sqrt(),
        Envelope::Sparse => 0.2 + 0.4 * (PI * q).sin(),
        Envelope::Flat => 0.5,
    }
}

/// Direction at angle `a` (rad) from `d`, at azimuth `phi` about it.
fn branch_dir(d: &Point, a: f64, phi: f64) -> Point {
    let (e1, e2) = basis(d);
    let w = add(&scale(&e1, phi.cos()), &scale(&e2, phi.sin()));
    unit(&add(&scale(d, a.cos()), &scale(&w, a.sin())))
}

/// Grow a tree (see the module documentation).
pub fn tree_model(spec: &TreeSpec) -> Result<SynthTree> {
    spec.check()?;
    let form = spec.archetype.form();
    let lad = spec.lad.unwrap_or_else(|| Lad::parse(form.lad, &[]).expect("archetype LAD"));
    if let Lad::TwoParamBeta(a, b) = lad {
        if !(a > 0.0 && b > 0.0) {
            return Err(Error::invalid("beta LAD parameters must be positive"));
        }
    }
    let mut rng = Generator::new(spec.seed);
    let h = spec.height;
    let k = (h / 20.0).clamp(0.25, 2.5);
    let base = spec.base;
    let (leaf_len, leaf_wid) = spec.leaf_size.unwrap_or(form.leaf);
    let crown_r = spec.crown_radius.unwrap_or_else(|| (form.crown_radius * h).min(form.crown_k * (0.6 + 12.0 * spec.dbh)));
    let z_top = (form.leader * h).max(0.05 * h);
    let z_cb = spec.crown_base.unwrap_or(form.crown_base * h).min(z_top);

    // ---- stem axis
    let lean = spec.lean_deg.unwrap_or(form.lean).to_radians();
    let lean_az = rng.uniform(0.0, 2.0 * PI);
    let sweep = spec.sweep.unwrap_or(form.sweep);
    let waves: Vec<[f64; 3]> = (0..6).map(|_| [rng.uniform(0.4, 1.5) * h, rng.uniform(0.0, 2.0 * PI), rng.normal(0.0, 1.0) / 3f64.sqrt()]).collect();
    let tilt = |s: f64| -> Point {
        let w = |j: usize| waves[j][2] * (2.0 * PI * s / waves[j][0] + waves[j][1]).sin() - waves[j][2] * waves[j][1].sin();
        let tx = lean.tan() * lean_az.cos() + sweep * (w(0) + w(1) + w(2));
        let ty = lean.tan() * lean_az.sin() + sweep * (w(3) + w(4) + w(5));
        unit(&[tx, ty, 1.0])
    };
    let s_ref = spec.breast_height.min(0.5 * z_top);
    let ds_target = (h / 250.0).clamp(0.02, 0.1);
    let n_ref = ((s_ref / ds_target - 0.5).round()).max(0.0);
    let ds = s_ref / (n_ref + 0.5);
    let n_ref = n_ref as usize;
    let mut g = Grower { segs: Vec::new(), axes: Vec::new() };
    let mut stem = AxisRec { order: 0, segs: Vec::new(), length: 0.0, epicormic: false, has_children: true };
    let (mut p, mut s) = (base, 0.0);
    while p[2] - base[2] < z_top || stem.segs.len() <= n_ref {
        let d = tilt(s + ds / 2.0);
        let kseg = g.segs.len();
        g.segs.push(Seg { start: p, dir: d, len: ds, parent: kseg as i64 - 1, order: 0, axis: 0, s_mid: s + ds / 2.0, radius: 0.0, leaf_w: 0.0, section: Vec::new() });
        stem.segs.push(kseg);
        p = add(&p, &scale(&d, ds));
        s += ds;
    }
    stem.length = s;
    let stem_len = s;
    g.axes.push(stem);
    let stem_top = p;
    let stem_bh = add(&g.segs[n_ref].start, &scale(&g.segs[n_ref].dir, ds / 2.0));

    // ---- first-order branches along the stem
    let rad = |deg: f64| deg.to_radians();
    let mut node_z = z_cb;
    let mut phi = rng.uniform(0.0, 2.0 * PI);
    let crown_top = h;
    let mut lowest_branch: Option<usize> = None;
    while node_z < z_top - 0.02 * h || (form.forks == 0 && node_z < z_top) {
        // Stem segment at this height.
        let kseg = *g.axes[0].segs.iter().find(|&&j| g.segs[j].start[2] + g.segs[j].dir[2] * g.segs[j].len - base[2] >= node_z).unwrap_or(g.axes[0].segs.last().unwrap());
        let sg = &g.segs[kseg];
        let t = ((node_z - (sg.start[2] - base[2])) / sg.dir[2]).clamp(0.0, sg.len);
        let at = add(&sg.start, &scale(&sg.dir, t));
        let sdir = sg.dir;
        let q = (node_z - z_cb) / (crown_top - z_cb).max(1e-9);
        let n_here = form.whorl;
        let whorl_rot = rng.uniform(0.0, 2.0 * PI);
        for w in 0..n_here {
            let az = if n_here > 1 { whorl_rot + 2.0 * PI * w as f64 / n_here as f64 + rng.uniform(-0.2, 0.2) } else {
                phi += rad(137.5) + rng.uniform(-0.2, 0.2);
                phi
            };
            let a = rad(form.angle[0] + rng.uniform(-8.0, 8.0));
            let reach = crown_r * envelope(form.envelope, q);
            let len = (reach / a.sin() * rng.uniform(0.85, 1.15)).max(0.2 * k);
            // Azimuth about the stem measured from its horizontal basis vector.
            let dir = branch_dir(&sdir, a, az);
            g.grow(&mut rng, at, dir, len, 1, kseg as i64, false, form.tropism[0]);
            lowest_branch = Some(lowest_branch.map_or(kseg, |b: usize| b.min(kseg)));
        }
        node_z += form.internode * k * rng.uniform(0.8, 1.2);
    }
    // ---- forks at the stem top
    let last = *g.axes[0].segs.last().unwrap();
    if form.forks > 0 {
        let fa = rad(form.fork_angle);
        let top_target = if spec.archetype == Archetype::Savanna { 0.8 } else { 0.9 } * h;
        let len = ((top_target - (stem_top[2] - base[2])).max(0.1 * h) / fa.cos()).max(0.1);
        let rot = rng.uniform(0.0, 2.0 * PI);
        let sdir = g.segs[last].dir;
        for f in 0..form.forks {
            let az = rot + 2.0 * PI * f as f64 / form.forks as f64 + rng.uniform(-0.3, 0.3);
            let a = fa * rng.uniform(0.8, 1.2);
            let dir = branch_dir(&sdir, a, az);
            let lf = len * rng.uniform(0.9, 1.1);
            g.grow(&mut rng, stem_top, dir, lf, 1, last as i64, false, form.tropism[0]);
            lowest_branch = Some(lowest_branch.map_or(last, |b: usize| b.min(last)));
        }
    }
    // ---- higher orders
    let mut ai = 1;
    while ai < g.axes.len() {
        let (order, length, epi) = (g.axes[ai].order, g.axes[ai].length, g.axes[ai].epicormic);
        if !epi && order < spec.max_order {
            let spacing = form.spacing[(order - 1) as usize] * k;
            let n = ((0.75 * length / spacing).floor() as usize).min(12);
            let mut side = if rng.random() < 0.5 { 0.0 } else { PI };
            for j in 0..n {
                let u = 0.2 + 0.75 * (j as f64 + rng.uniform(0.2, 0.8)) / n as f64;
                let (at, kseg) = g.along(ai, u * length);
                let pdir = g.segs[kseg].dir;
                let a = rad(form.angle[order as usize] + rng.uniform(-10.0, 10.0));
                side += PI;
                let az = side + rng.uniform(-0.5, 0.5);
                let len = (form.ratio[order as usize] * length * (1.0 - 0.5 * u) * rng.uniform(0.8, 1.2)).max(0.02);
                let dir = branch_dir(&pdir, a, az);
                g.grow(&mut rng, at, dir, len, order + 1, kseg as i64, false, form.tropism[order as usize]);
            }
            g.axes[ai].has_children = n > 0;
        }
        ai += 1;
    }
    // ---- epicormic shoots on the bole below the crown
    let s_cb = {
        let lb = lowest_branch.unwrap_or(last);
        g.segs[lb].s_mid
    };
    let mut epi_axes = Vec::new();
    if spec.epicormic > 0.0 && s_cb > 0.6 {
        let n = random_round(&mut rng, spec.epicormic * (s_cb - 0.5));
        for _ in 0..n {
            let s = rng.uniform(0.5, s_cb);
            let (at, kseg) = g.along(0, s);
            let dir = branch_dir(&g.segs[kseg].dir, rad(60.0 + rng.uniform(-15.0, 15.0)), rng.uniform(0.0, 2.0 * PI));
            let len = rng.uniform(0.1, 0.4) * k.clamp(0.5, 1.0);
            epi_axes.push(g.grow(&mut rng, at, dir, len, 1, kseg as i64, true, 0.5));
        }
    }

    // ---- leaves
    let sampler = LadSampler::new(&lad);
    let leaf_area_one = PI / 4.0 * leaf_len * leaf_wid;
    let twigs: Vec<usize> = (1..g.axes.len()).filter(|&a| !g.axes[a].epicormic && !g.axes[a].has_children).collect();
    let total_len: f64 = twigs.iter().map(|&a| g.axes[a].length).sum();
    let target = spec.leaf_area.unwrap_or((form.lai * PI * crown_r * crown_r).min(form.leaf_k * spec.dbh * spec.dbh));
    let mut leaves: Vec<Leaf> = Vec::new();
    let place = |g: &mut Grower, rng: &mut Generator, a: usize, n: usize, from: f64, epicormic: bool, leaves: &mut Vec<Leaf>| {
        let length = g.axes[a].length;
        for _ in 0..n {
            let (at, kseg) = g.along(a, rng.uniform(from, 1.0) * length);
            let sd = g.segs[kseg].dir;
            let (e1, e2) = basis(&sd);
            let pa = rng.uniform(0.0, 2.0 * PI);
            let qdir = add(&scale(&e1, pa.cos()), &scale(&e2, pa.sin()));
            let incl = sampler.inclination(rng.random());
            let az = rng.uniform(0.0, 2.0 * PI);
            let normal = [incl.sin() * az.cos(), incl.sin() * az.sin(), incl.cos()];
            let proj = add(&qdir, &scale(&normal, -dot(&qdir, &normal)));
            let axis = if dot(&proj, &proj) > 1e-8 { unit(&proj) } else { basis(&normal).0 };
            let centre = add(&at, &scale(&axis, 0.6 * leaf_len));
            g.segs[kseg].leaf_w += leaf_area_one;
            leaves.push(Leaf { centre, normal, axis, length: leaf_len, width: leaf_wid, cylinder: kseg, epicormic });
        }
    };
    if total_len > 0.0 {
        for &a in &twigs {
            let n = random_round(&mut rng, target * g.axes[a].length / total_len / leaf_area_one);
            place(&mut g, &mut rng, a, n, 0.2, false, &mut leaves);
        }
    }
    for &a in &epi_axes {
        let area = rng.uniform(0.03, 0.1);
        let n = random_round(&mut rng, area / leaf_area_one);
        place(&mut g, &mut rng, a, n, 0.0, true, &mut leaves);
    }

    // ---- radii: taper below the crown, pipe model above
    let swell = spec.butt_swell.unwrap_or(form.butt_swell);
    let r_dbh = spec.dbh / 2.0;
    let big_s = stem_len.max(s_ref + 0.1);
    let prof = |s: f64| (1.0 + swell * (-s / 0.6).exp()) * ((big_s - s).max(0.0) / (big_s - s_ref)).powf(form.taper);
    let profile = |s: f64| r_dbh * prof(s) / prof(s_ref);
    let nseg = g.segs.len();
    let mut w_down: Vec<f64> = g.segs.iter().map(|sg| sg.leaf_w).collect();
    for i in (0..nseg).rev() {
        let p = g.segs[i].parent;
        if p >= 0 {
            w_down[p as usize] += w_down[i];
        }
    }
    let stem_segs = g.axes[0].segs.clone();
    let is_epi_seg: Vec<bool> = g.segs.iter().map(|sg| g.axes[sg.axis].epicormic).collect();
    // Crown leaf area through each stem segment, without the epicormic shoots.
    let mut w_crown = w_down.clone();
    for &a in &epi_axes {
        let first = g.axes[a].segs[0];
        let w = w_down[first];
        let mut p = g.segs[first].parent;
        while p >= 0 {
            w_crown[p as usize] -= w;
            p = g.segs[p as usize].parent;
        }
    }
    let ib = lowest_branch.map(|b| stem_segs.iter().position(|&j| j == b).unwrap_or(0));
    let e = spec.pipe_exponent;
    let (anchor, c) = match ib {
        Some(ib) if w_crown[stem_segs[ib]] > 0.0 => {
            let a = ib.max(n_ref.min(stem_segs.len() - 1));
            let sa = stem_segs[a];
            let r_a = profile(g.segs[sa].s_mid).max(spec.min_radius);
            (Some(ib), r_a / w_crown[sa].max(1e-12).powf(1.0 / e))
        }
        _ => (None, 0.0),
    };
    for (pos, &j) in stem_segs.iter().enumerate() {
        let taper = profile(g.segs[j].s_mid).max(spec.min_radius);
        g.segs[j].radius = match anchor {
            Some(ib) if pos > ib => (c * w_crown[j].max(0.0).powf(1.0 / e)).max(spec.min_radius),
            _ => taper,
        };
    }
    let c_epi = if c > 0.0 { c } else { profile(s_cb) / w_down[stem_segs[0]].max(1e-12).powf(1.0 / e) };
    for i in 0..nseg {
        if g.segs[i].order > 0 {
            let cc = if is_epi_seg[i] { c_epi } else { c };
            g.segs[i].radius = (cc * w_down[i].powf(1.0 / e)).max(spec.min_radius);
        }
    }
    // Stem sections: ellipse, bark fissures, buttress flanges.
    let ell_az = rng.uniform(0.0, PI);
    let bark_phase = rng.uniform(0.0, 2.0 * PI);
    let n_ridges = ((2.0 * PI * r_dbh / 0.04).round() as usize).max(8) as f64;
    let flanges: Vec<f64> = (0..spec.buttresses).map(|j| 2.0 * PI * j as f64 / spec.buttresses as f64 + rng.uniform(-0.3, 0.3)).collect();
    let flange_w = 0.18;
    let theta = |i: usize| 2.0 * PI * i as f64 / SECTION as f64;
    for &j in &stem_segs {
        let (r_nom, sm) = (g.segs[j].radius, g.segs[j].s_mid);
        let twist = 0.1 * sm;
        let mut raw: Vec<f64> = (0..SECTION).map(|i| r_nom * (1.0 + spec.ellipticity * (2.0 * (theta(i) - ell_az - twist)).cos())).collect();
        if spec.bark_depth > 0.0 {
            let ridge: Vec<f64> = (0..SECTION).map(|i| (n_ridges / 2.0 * theta(i) + 1.2 * (2.0 * PI * sm / 1.7 + bark_phase).sin()).sin().abs().powf(0.3)).collect();
            let mean = ridge.iter().sum::<f64>() / SECTION as f64;
            for i in 0..SECTION {
                raw[i] += spec.bark_depth * (ridge[i] - mean);
            }
        }
        let rms = (raw.iter().map(|v| v * v).sum::<f64>() / SECTION as f64).sqrt();
        let mut sec: Vec<f64> = raw.iter().map(|v| (v * r_nom / rms).max(0.2 * r_nom)).collect();
        let amp = spec.buttress_extent * (1.0 - sm / spec.buttress_height.max(1e-9)).max(0.0).powi(2);
        if amp > 0.0 && !flanges.is_empty() {
            for (i, v) in sec.iter_mut().enumerate() {
                let bump: f64 = flanges.iter().map(|&f| {
                    let d = (theta(i) - f + PI).rem_euclid(2.0 * PI) - PI;
                    (-(d / flange_w).powi(2)).exp()
                }).sum();
                *v *= 1.0 + amp * bump;
            }
        }
        let req = (sec.iter().map(|v| v * v).sum::<f64>() / SECTION as f64).sqrt();
        g.segs[j].radius = req;
        g.segs[j].section = sec;
    }
    let dbh = 2.0 * g.segs[stem_segs[n_ref]].radius;

    // ---- points
    let dens = spec.point_density;
    let mut xyz: Vec<Point> = Vec::new();
    let mut nrm: Vec<Point> = Vec::new();
    let (mut cls, mut label, mut order_attr, mut cyl_attr, mut leaf_attr, mut epi_attr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut n_points = vec![0usize; nseg];
    for i in 0..nseg {
        let sg = &g.segs[i];
        let first_of_axis = sg.order > 0 && (sg.parent < 0 || g.segs[sg.parent as usize].axis != sg.axis);
        let parent = if first_of_axis { Some(&g.segs[sg.parent as usize]) } else { None };
        if sg.order == 0 {
            let (u, v) = section_frame(&sg.dir);
            let pts: Vec<[f64; 2]> = (0..SECTION).map(|m| [sg.section[m] * theta(m).cos(), sg.section[m] * theta(m).sin()]).collect();
            let mut cum = vec![0.0; SECTION + 1];
            for m in 0..SECTION {
                let a = pts[m];
                let b = pts[(m + 1) % SECTION];
                cum[m + 1] = cum[m] + ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            }
            let perim = cum[SECTION];
            let cells = jittered_grid(&mut rng, dens * perim * sg.len, perim, sg.len);
            let n = cells.len();
            for &[target, t] in &cells {
                let m = cum.partition_point(|&c| c <= target).clamp(1, SECTION) - 1;
                let f = (target - cum[m]) / (cum[m + 1] - cum[m]).max(1e-15);
                let a = pts[m];
                let b = pts[(m + 1) % SECTION];
                let (x2, y2) = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]));
                let c = add(&sg.start, &scale(&sg.dir, t));
                xyz.push(add(&c, &add(&scale(&u, x2), &scale(&v, y2))));
                // Outward normal of the (counter-clockwise) section edge.
                nrm.push(unit(&add(&scale(&u, b[1] - a[1]), &scale(&v, a[0] - b[0]))));
            }
            n_points[i] = n;
            label.extend(std::iter::repeat_n(LABEL_STEM, n));
        } else {
            let (u, v) = section_frame(&sg.dir);
            let circ = 2.0 * PI * sg.radius;
            let mut kept = 0;
            for [arc, t] in jittered_grid(&mut rng, dens * circ * sg.len, circ, sg.len) {
                let a = arc / sg.radius;
                let c = add(&sg.start, &scale(&sg.dir, t));
                let q = add(&c, &add(&scale(&u, sg.radius * a.cos()), &scale(&v, sg.radius * a.sin())));
                if let Some(ps) = parent {
                    // Leave out what lies inside the parent.
                    let w = [q[0] - ps.start[0], q[1] - ps.start[1], q[2] - ps.start[2]];
                    let along = dot(&w, &ps.dir);
                    let perp = dot(&w, &w) - along * along;
                    if along > -ps.radius && along < ps.len + ps.radius && perp < ps.radius * ps.radius {
                        continue;
                    }
                }
                xyz.push(q);
                nrm.push(add(&scale(&u, a.cos()), &scale(&v, a.sin())));
                kept += 1;
            }
            n_points[i] = kept;
            label.extend(std::iter::repeat_n(LABEL_BRANCH, kept));
        }
        let n = n_points[i];
        cls.extend(std::iter::repeat_n(5u8, n));
        order_attr.extend(std::iter::repeat_n(sg.order as i32, n));
        cyl_attr.extend(std::iter::repeat_n(i as i32, n));
        leaf_attr.extend(std::iter::repeat_n(-1i32, n));
        epi_attr.extend(std::iter::repeat_n(u8::from(is_epi_seg[i]), n));
    }
    for (li, lf) in leaves.iter().enumerate() {
        let b = cross(&lf.normal, &lf.axis);
        // A jittered grid over the blade's bounding rectangle, kept inside the ellipse.
        let mut n = 0;
        for [p, q] in jittered_grid(&mut rng, dens * lf.length * lf.width, lf.length, lf.width) {
            let (p, q) = (p - 0.5 * lf.length, q - 0.5 * lf.width);
            if (2.0 * p / lf.length).powi(2) + (2.0 * q / lf.width).powi(2) <= 1.0 {
                xyz.push(add(&lf.centre, &add(&scale(&lf.axis, p), &scale(&b, q))));
                nrm.push(lf.normal);
                n += 1;
            }
        }
        cls.extend(std::iter::repeat_n(4u8, n));
        label.extend(std::iter::repeat_n(LABEL_LEAF, n));
        order_attr.extend(std::iter::repeat_n(g.segs[lf.cylinder].order as i32, n));
        cyl_attr.extend(std::iter::repeat_n(-1i32, n));
        leaf_attr.extend(std::iter::repeat_n(li as i32, n));
        epi_attr.extend(std::iter::repeat_n(u8::from(lf.epicormic), n));
    }
    let mut cloud = PointCloud::new(xyz);
    cloud.attrs.insert("classification".into(), Attr::U8(cls));
    cloud.attrs.insert("label".into(), Attr::U8(label));
    cloud.attrs.insert("branch_order".into(), Attr::I32(order_attr));
    cloud.attrs.insert("cylinder".into(), Attr::I32(cyl_attr));
    cloud.attrs.insert("leaf".into(), Attr::I32(leaf_attr));
    cloud.attrs.insert("epicormic".into(), Attr::U8(epi_attr));
    insert_normals(&mut cloud, &nrm);

    // ---- truth
    let cylinders: Vec<Cylinder> = g.segs.iter().enumerate().map(|(i, sg)| Cylinder { start: sg.start, axis: sg.dir, length: sg.len, radius: sg.radius, parent: sg.parent, branch_order: sg.order, branch_id: sg.axis as u32, n_points: n_points[i] }).collect();
    let mut top = f64::NEG_INFINITY;
    for sg in &g.segs {
        top = top.max(sg.start[2] + sg.dir[2] * sg.len);
    }
    for lf in &leaves {
        let b = cross(&lf.normal, &lf.axis);
        top = top.max(lf.centre[2] + 0.5 * (lf.length * lf.axis[2]).hypot(lf.width * b[2]));
    }
    let crown_base = lowest_branch.map_or(stem_top[2] - base[2], |b| {
        let sg = &g.segs[b];
        sg.start[2] + sg.dir[2] * sg.len - base[2]
    });
    let mut crown_xy: Vec<[f64; 2]> = leaves.iter().filter(|l| !l.epicormic).map(|l| [l.centre[0], l.centre[1]]).collect();
    for (i, sg) in g.segs.iter().enumerate() {
        if sg.order > 0 && !is_epi_seg[i] {
            let e = add(&sg.start, &scale(&sg.dir, sg.len));
            crown_xy.push([e[0], e[1]]);
        }
    }
    let crown_area = hull_area(&crown_xy);
    let mut ext = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in &crown_xy {
        ext = [ext[0].min(p[0]), ext[1].min(p[1]), ext[2].max(p[0]), ext[3].max(p[1])];
    }
    if crown_xy.is_empty() {
        ext = [base[0], base[1], base[0], base[1]];
    }
    let leaf_area = leaves.iter().map(|l| l.area()).sum();
    let epicormic_leaf_area = leaves.iter().filter(|l| l.epicormic).map(|l| l.area()).sum();
    Ok(SynthTree { cloud, qsm: Qsm { cylinders }, cylinder_epicormic: is_epi_seg, leaves, archetype: spec.archetype, base, stem_bh, dbh, height: top - base[2], crown_base, crown_area, crown_extent: ext, leaf_area, epicormic_leaf_area, lad })
}

/// Store unit surface normals as the `normal_x`, `normal_y` and `normal_z`
/// attributes (float32), which [`crate::synthetic_scan`] uses to orient the
/// surface patch each point stands for.
pub fn insert_normals(cloud: &mut PointCloud, nrm: &[Point]) {
    for (k, name) in ["normal_x", "normal_y", "normal_z"].iter().enumerate() {
        cloud.attrs.insert((*name).into(), Attr::F32(nrm.iter().map(|n| n[k] as f32).collect()));
    }
}

/// Area of the convex hull of `xy` (Andrew's monotone chain).
pub fn hull_area(xy: &[[f64; 2]]) -> f64 {
    let mut p: Vec<[f64; 2]> = xy.iter().copied().filter(|v| v[0].is_finite() && v[1].is_finite()).collect();
    if p.len() < 3 {
        return 0.0;
    }
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let turn = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut hull: Vec<[f64; 2]> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in iter {
            while hull.len() >= start + 2 && turn(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0 {
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    let n = hull.len();
    (0..n).map(|i| hull[i][0] * hull[(i + 1) % n][1] - hull[(i + 1) % n][0] * hull[i][1]).sum::<f64>().abs() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(a: Archetype) -> TreeSpec {
        let mut s = TreeSpec::new(a, [1.0, 2.0, 0.5], 0.3, 15.0, 3);
        s.point_density = 300.0;
        s
    }

    #[test]
    fn every_archetype_grows() {
        for a in [Archetype::Broadleaf, Archetype::Conifer, Archetype::Eucalypt, Archetype::Savanna] {
            let t = tree_model(&spec(a)).unwrap();
            assert!((t.dbh - 0.3).abs() < 1e-9, "{a:?} dbh {}", t.dbh);
            assert!(t.height > 0.8 * 15.0 && t.height < 1.2 * 15.0, "{a:?} height {}", t.height);
            assert!(t.leaves.len() > 100, "{a:?}");
            assert!(t.qsm.cylinders.iter().any(|c| c.branch_order == 3));
            let n: usize = t.qsm.cylinders.iter().map(|c| c.n_points).sum();
            let Some(Attr::U8(cls)) = t.cloud.attr("classification") else { panic!() };
            assert_eq!(cls.iter().filter(|&&c| c == 5).count(), n);
            // Pipe model: area is conserved at every branching.
            let rows = &t.qsm.cylinders;
            let mut child_area = vec![0.0; rows.len()];
            for c in rows.iter().filter(|c| c.parent >= 0) {
                child_area[c.parent as usize] += c.radius * c.radius;
            }
            for (i, c) in rows.iter().enumerate() {
                let clamped = rows.iter().any(|d| d.parent == i as i64 && d.radius <= 0.0015 * (1.0 + 1e-12));
                if c.branch_order > 0 && child_area[i] > 0.0 && !clamped {
                    assert!(child_area[i] <= c.radius * c.radius * (1.0 + 1e-9), "{a:?} cylinder {i}");
                }
            }
        }
    }

    #[test]
    fn a_bare_stem_is_a_frustum_stack() {
        let mut s = TreeSpec::new(Archetype::Conifer, [0.0; 3], 0.4, 10.0, 1);
        s.crown_base = Some(10.0);
        s.lean_deg = Some(0.0);
        s.sweep = Some(0.0);
        s.butt_swell = Some(0.0);
        s.point_density = 50.0;
        let t = tree_model(&s).unwrap();
        // No branch fits between the crown base and the top: a straight tapered stem.
        assert!(t.qsm.cylinders.iter().all(|c| c.branch_order == 0));
        let v: f64 = t.wood_volume();
        // Cone-like taper (exponent 1) of radius 0.2 at 1.3 m: V ≈ π r0² S / 3 with r0 = 0.2 S / (S - 1.3).
        let big_s = t.qsm.cylinders.iter().map(|c| c.length).sum::<f64>();
        let r0 = 0.2 * big_s / (big_s - 1.3);
        let exact = PI * r0 * r0 * big_s / 3.0;
        assert!((v - exact).abs() / exact < 0.01, "{v} vs {exact}");
    }

    #[test]
    fn leaf_angles_follow_the_distribution() {
        let mut s = spec(Archetype::Broadleaf);
        s.lad = Some(Lad::Erectophile);
        s.point_density = 1.0;
        let t = tree_model(&s).unwrap();
        let mean = t.leaves.iter().map(|l| l.normal[2].clamp(-1.0, 1.0).acos()).sum::<f64>() / t.leaves.len() as f64;
        // Erectophile mean inclination: π/4 + 1/π = 63.24 degrees.
        assert!((mean.to_degrees() - 63.24).abs() < 1.0, "{}", mean.to_degrees());
    }

    #[test]
    fn same_seed_same_tree() {
        let a = tree_model(&spec(Archetype::Eucalypt)).unwrap();
        let b = tree_model(&spec(Archetype::Eucalypt)).unwrap();
        assert_eq!(a.cloud.xyz, b.cloud.xyz);
        let mut s = spec(Archetype::Eucalypt);
        s.seed = 4;
        assert_ne!(tree_model(&s).unwrap().cloud.xyz.len(), a.cloud.xyz.len());
    }

    #[test]
    fn rejects_bad_input() {
        let mut s = spec(Archetype::Broadleaf);
        s.max_order = 5;
        assert!(tree_model(&s).is_err());
        let mut s = spec(Archetype::Broadleaf);
        s.dbh = f64::NAN;
        assert!(tree_model(&s).is_err());
        assert!(Archetype::parse("palm").is_err());
    }

    #[test]
    fn hull_of_a_square() {
        assert!((hull_area(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]) - 1.0).abs() < 1e-12);
    }
}
