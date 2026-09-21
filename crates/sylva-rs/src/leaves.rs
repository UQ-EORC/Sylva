//! Foliage for a structure model: leaf angle distribution, leaf area from
//! leaf points, and leaf meshes placed on a QSM.
//!
//! A QSM describes the wood. To make a whole-tree model for radiative
//! transfer or visualisation the leaves are added as flat polygons whose
//! total area, spatial distribution and orientation follow what was measured:
//!
//! * **angles** -- the inclination of a leaf is the angle between its normal
//!   and the vertical; normals come from a PCA of each leaf point's
//!   neighbours, their inclinations are binned over `[0, 90]` degrees and
//!   summarised by the mean, a two-parameter beta distribution (Goel &
//!   Strebel 1984), Campbell's ellipsoidal `chi` (1990) and the nearest de
//!   Wit type;
//! * **area** -- a leaf area per voxel, either from ray-traced leaf area
//!   density (`voxel`) or, without pulses, from the leaf points themselves: a
//!   surface thinned to one point per cube of side `res` crosses
//!   `(|nx| + |ny| + |nz|) / res^2` cubes per unit area, so each thinned point
//!   stands for `res^2 / (|nx| + |ny| + |nz|)` of surface. That counts what
//!   the scanner saw: occluded foliage is missing, so it is a lower bound;
//! * **insertion** -- each voxel receives leaves until its area is met,
//!   centred on leaf points of that voxel (or uniformly inside it), with
//!   normals drawn from the angle distribution and a uniform azimuth, and the
//!   blade pointing away from the nearest branch of the QSM. Leaves may
//!   intersect each other; nothing here resolves collisions.

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

use rayon::prelude::*;

use crate::filters::{local_pca, voxel_downsample_indices, Rng};
use crate::qsm::model::Cylinder;
use crate::qsm::wood::{wood_mask, WoodParams};
use crate::spatial::KdTree;
use crate::transform::{add, cross, dot, norm, normalize, scale, sub};
use crate::voxel::metrics::classify_de_wit;
use crate::Point;

/// Inclination (rad, `[0, pi/2]`) of the local surface normal at every point,
/// from a PCA over `k` neighbours, with the normal itself.
pub fn inclinations(points: &[Point], k: usize) -> (Vec<f64>, Vec<Point>) {
    let (normals, _) = local_pca(points, k.max(3));
    let incl = normals.iter().map(|n| n[2].abs().min(1.0).acos()).collect();
    (incl, normals)
}

/// A leaf inclination angle distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct LeafAngles {
    /// Bin centres (rad) over `[0, pi/2]`.
    pub bin_centres: Vec<f64>,
    /// Probability per bin (sums to 1).
    pub density: Vec<f64>,
    pub mean: f64,
    pub std: f64,
    /// Beta distribution on `t = 2 theta / pi`: `f(t) ~ t^(a-1) (1-t)^(b-1)`.
    pub beta_a: f64,
    pub beta_b: f64,
    /// Campbell's ellipsoidal parameter (1 spherical, > 1 planophile).
    pub chi: f64,
    pub de_wit: Option<&'static str>,
}

/// Bin inclinations (rad), optionally weighted (e.g. by the area each point
/// stands for), and fit the summary parameters.
pub fn angle_distribution(inclination: &[f64], weights: Option<&[f64]>, n_bins: usize) -> LeafAngles {
    let n_bins = n_bins.max(1);
    let mut density = vec![0.0; n_bins];
    let (mut sw, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for (i, &th) in inclination.iter().enumerate() {
        if !th.is_finite() {
            continue;
        }
        let w = weights.map_or(1.0, |w| w[i]);
        let th = th.clamp(0.0, FRAC_PI_2);
        density[((th / FRAC_PI_2 * n_bins as f64) as usize).min(n_bins - 1)] += w;
        sw += w;
        s1 += w * th;
        s2 += w * th * th;
    }
    let bin_centres: Vec<f64> = (0..n_bins).map(|b| (b as f64 + 0.5) * FRAC_PI_2 / n_bins as f64).collect();
    if sw <= 0.0 {
        // Nothing to go on: spherical, whose density is sin(theta).
        let mut d: Vec<f64> = bin_centres.iter().map(|t| t.sin()).collect();
        let s: f64 = d.iter().sum();
        d.iter_mut().for_each(|v| *v /= s);
        return LeafAngles { bin_centres, density: d, mean: 1.0, std: 0.0, beta_a: f64::NAN, beta_b: f64::NAN, chi: 1.0, de_wit: Some("spherical") };
    }
    density.iter_mut().for_each(|v| *v /= sw);
    let mean = s1 / sw;
    let var = (s2 / sw - mean * mean).max(0.0);
    let t = mean / FRAC_PI_2;
    let vt = var / (FRAC_PI_2 * FRAC_PI_2);
    let c = if vt > 0.0 { t * (1.0 - t) / vt - 1.0 } else { f64::NAN };
    // Campbell (1990): mean inclination = 9.65 (3 + chi)^-1.65 rad.
    let chi = ((mean.max(1e-6) / 9.65).powf(-1.0 / 1.65) - 3.0).max(0.01);
    let de_wit = classify_de_wit(&bin_centres, &density);
    LeafAngles { bin_centres, density, mean, std: var.sqrt(), beta_a: t * c, beta_b: (1.0 - t) * c, chi, de_wit }
}

impl LeafAngles {
    fn unit(rng: &mut Rng) -> f64 {
        (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Draw an inclination (rad): a bin by its probability, uniform within it.
    fn sample(&self, rng: &mut Rng) -> f64 {
        let u = Self::unit(rng);
        let mut acc = 0.0;
        let n = self.density.len();
        let width = FRAC_PI_2 / n as f64;
        for (b, &p) in self.density.iter().enumerate() {
            acc += p;
            if u <= acc || b == n - 1 {
                return (b as f64 + Self::unit(rng)) * width;
            }
        }
        FRAC_PI_2 / 2.0
    }
}

/// Leaf area seen in the points: thin them to one per cube of side `res`,
/// estimate each survivor's normal over `k` neighbours, and give it the area
/// `res^2 / (|nx| + |ny| + |nz|)`. Returns the thinned points, their area and
/// their inclination (rad).
///
/// `res <= 0` picks it from the data: 3.5 times the median nearest-neighbour
/// spacing. The estimate is a box count and has no plateau: smaller cubes fall
/// between the points and undercount, larger ones overcount at leaf edges and
/// through the range noise. On unoccluded synthetic leaves of 2-8 cm the
/// default is within about 15 % from 5 000 to 80 000 points per m2.
pub fn point_leaf_area(points: &[Point], res: f64, k: usize) -> (Vec<Point>, Vec<f64>, Vec<f64>) {
    let res = if res > 0.0 { res } else { 3.5 * median_spacing(points) };
    if !(res > 0.0) {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let keep = voxel_downsample_indices(points, res);
    let thin: Vec<Point> = keep.iter().map(|&i| points[i]).collect();
    if thin.len() < 3 {
        return (thin, Vec::new(), Vec::new());
    }
    let (incl, normals) = inclinations(&thin, k.min(thin.len()));
    let area = normals.iter().map(|n| res * res / (n[0].abs() + n[1].abs() + n[2].abs()).max(1.0)).collect();
    (thin, area, incl)
}

/// Median distance to the nearest neighbour, from up to 20 000 evenly taken points.
pub fn median_spacing(points: &[Point]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let tree = KdTree::new(points);
    let step = points.len().div_ceil(20_000);
    let mut d: Vec<f64> = points.par_iter().step_by(step).filter_map(|p| tree.knn(p, 2).get(1).map(|&(_, d)| d)).collect();
    if d.is_empty() {
        return 0.0;
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d[d.len() / 2]
}

/// Wood (`true`) / leaf (`false`) for every point: the classification runs on
/// the cloud thinned to `voxel_size` and each point takes the label of its
/// nearest thinned neighbour.
pub fn classify_leaf_wood(points: &[Point], voxel_size: f64, params: &WoodParams) -> Vec<bool> {
    classify_thinned(points, voxel_size, |thin| wood_mask(thin, params))
}

/// The same with the graph-based separation ([`crate::qsm::wood::gbs_mask`]).
pub fn classify_leaf_wood_gbs(points: &[Point], voxel_size: f64, params: &crate::qsm::wood::GbsParams) -> Vec<bool> {
    classify_thinned(points, voxel_size, |thin| crate::qsm::wood::gbs_mask(thin, params))
}

fn classify_thinned(points: &[Point], voxel_size: f64, classify: impl Fn(&[Point]) -> Vec<bool>) -> Vec<bool> {
    if points.is_empty() {
        return Vec::new();
    }
    if !(voxel_size > 0.0) {
        return classify(points);
    }
    let keep = voxel_downsample_indices(points, voxel_size);
    let thin: Vec<Point> = keep.iter().map(|&i| points[i]).collect();
    let mask = classify(&thin);
    let tree = KdTree::new(&thin);
    points.par_iter().map(|p| tree.nearest(p).map(|(j, _)| mask[j]).unwrap_or(false)).collect()
}

#[derive(Debug, Clone)]
pub struct LeafParams {
    /// Blade length and greatest width (m).
    pub length: f64,
    pub width: f64,
    /// Leaves whose centre is within this distance of a cylinder axis point
    /// away from it; further ones take a random in-plane direction.
    pub max_branch_distance: f64,
    /// Random displacement of a leaf centre from its seed point (m).
    pub jitter: f64,
    pub seed: u64,
}

impl Default for LeafParams {
    fn default() -> Self {
        LeafParams { length: 0.08, width: 0.04, max_branch_distance: 0.5, jitter: 0.01, seed: 1 }
    }
}

/// Outline of a unit leaf in its own plane, base at the origin, tip at
/// `(1, 0)`, half-width 0.5 at 45 % of the length: `(along, across)`.
const OUTLINE: [(f64, f64); 6] = [(0.0, 0.0), (0.2, 0.36), (0.45, 0.5), (1.0, 0.0), (0.45, -0.5), (0.2, -0.36)];

/// One-sided area of a leaf of the given length and width.
pub fn single_leaf_area(length: f64, width: f64) -> f64 {
    let mut a = 0.0;
    for i in 0..OUTLINE.len() {
        let (x0, y0) = OUTLINE[i];
        let (x1, y1) = OUTLINE[(i + 1) % OUTLINE.len()];
        a += x0 * y1 - x1 * y0;
    }
    0.5 * a.abs() * length * width
}

#[derive(Debug, Clone, Default)]
pub struct LeafMesh {
    pub vertices: Vec<Point>,
    pub faces: Vec<[u32; 3]>,
    /// Per leaf: centre, unit normal, inclination (rad), nearest cylinder (-1 if none in reach).
    pub centres: Vec<Point>,
    pub normals: Vec<Point>,
    pub inclination: Vec<f64>,
    pub cylinder: Vec<i64>,
    /// One-sided area of one leaf (m2).
    pub leaf_area: f64,
}

impl LeafMesh {
    pub fn n_leaves(&self) -> usize {
        self.centres.len()
    }

    pub fn total_area(&self) -> f64 {
        self.leaf_area * self.n_leaves() as f64
    }
}

/// Place leaves. `cells` are voxel centres with the one-sided leaf area (m2)
/// each must hold; `seeds` are leaf points (may be empty) that leaves are
/// centred on where a cell has some; `cylinders` the QSM (may be empty).
pub fn insert_leaves(cells: &[(Point, f64)], voxel_size: f64, seeds: &[Point], angles: &LeafAngles, cylinders: &[Cylinder], p: &LeafParams) -> LeafMesh {
    let leaf_area = single_leaf_area(p.length, p.width);
    let mut mesh = LeafMesh { leaf_area, ..Default::default() };
    if !(leaf_area > 0.0) || !(voxel_size > 0.0) {
        return mesh;
    }
    let key = |q: &Point| [(q[0] / voxel_size).floor() as i64, (q[1] / voxel_size).floor() as i64, (q[2] / voxel_size).floor() as i64];
    let mut by_cell: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, s) in seeds.iter().enumerate() {
        by_cell.entry(key(s)).or_default().push(i);
    }
    // Axis samples every 5 cm to find the nearest branch.
    let mut axis_pts: Vec<Point> = Vec::new();
    let mut axis_cyl: Vec<usize> = Vec::new();
    for (ci, c) in cylinders.iter().enumerate() {
        let n = (c.length / 0.05).ceil().max(1.0) as usize;
        for s in 0..=n {
            axis_pts.push(add(&c.start, &scale(&c.axis, c.length * s as f64 / n as f64)));
            axis_cyl.push(ci);
        }
    }
    let axis_tree = (!axis_pts.is_empty()).then(|| KdTree::new(&axis_pts));

    let mut rng = Rng::new(p.seed.max(1));
    let unit = |rng: &mut Rng| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut carry = 0.0; // fractional leaves carried between cells so the total is met
    for (centre, area) in cells {
        if !(*area > 0.0) {
            continue;
        }
        let want = area / leaf_area + carry;
        let n = want.floor() as usize;
        carry = want - n as f64;
        let local = by_cell.get(&key(centre));
        for _ in 0..n {
            let mut c = match local {
                Some(idx) if !idx.is_empty() => seeds[idx[rng.below(idx.len())]],
                _ => [centre[0] + (unit(&mut rng) - 0.5) * voxel_size, centre[1] + (unit(&mut rng) - 0.5) * voxel_size, centre[2] + (unit(&mut rng) - 0.5) * voxel_size],
            };
            for k in 0..3 {
                c[k] += (unit(&mut rng) - 0.5) * 2.0 * p.jitter;
            }
            let theta = angles.sample(&mut rng);
            let phi = unit(&mut rng) * TAU;
            let normal = [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()];
            // Blade direction in the leaf plane: away from the nearest branch, else random.
            let mut parent = -1i64;
            let mut away: Option<Point> = None;
            if let Some(tree) = &axis_tree {
                if let Some((j, d)) = tree.nearest(&c) {
                    if d <= p.max_branch_distance {
                        parent = axis_cyl[j] as i64;
                        away = Some(sub(&c, &axis_pts[j]));
                    }
                }
            }
            let in_plane = |v: &Point| sub(v, &scale(&normal, dot(v, &normal)));
            let mut dir = away.map(|v| in_plane(&v)).filter(|v| norm(v) > 1e-9);
            if dir.is_none() {
                let a = unit(&mut rng) * TAU;
                let helper = if normal[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
                let u = normalize(&cross(&normal, &helper));
                let v = cross(&normal, &u);
                dir = Some(add(&scale(&u, a.cos()), &scale(&v, a.sin())));
            }
            let dir = normalize(&dir.unwrap());
            let side = cross(&normal, &dir);
            let base = sub(&c, &scale(&dir, 0.5 * p.length));
            let v0 = mesh.vertices.len() as u32;
            for (along, across) in OUTLINE {
                mesh.vertices.push(add(&add(&base, &scale(&dir, along * p.length)), &scale(&side, across * p.width)));
            }
            for t in 1..OUTLINE.len() as u32 - 1 {
                mesh.faces.push([v0, v0 + t, v0 + t + 1]);
            }
            mesh.centres.push(c);
            mesh.normals.push(normal);
            mesh.inclination.push(theta);
            mesh.cylinder.push(parent);
        }
    }
    mesh
}

/// Beam projection `G(theta)` of the distribution for a beam at zenith
/// `theta` (rad): the mean of `|cos angle(beam, normal)|` over the histogram
/// with a uniform azimuth (Wilson 1960 kernel).
pub fn projection(angles: &LeafAngles, beam_zenith: f64) -> f64 {
    let (st, ct) = beam_zenith.sin_cos();
    angles
        .bin_centres
        .iter()
        .zip(&angles.density)
        .map(|(&tl, &w)| {
            let (sl, cl) = tl.sin_cos();
            let a = ct * cl;
            let b = st * sl;
            let g = if b.abs() <= a.abs() || b.abs() < 1e-12 {
                a.abs()
            } else {
                let psi = (-a / b).clamp(-1.0, 1.0).acos();
                (a * (2.0 * psi / PI - 1.0) + 2.0 / PI * b * psi.sin()).abs()
            };
            w * g
        })
        .sum()
}
