// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Virtual scans of a synthetic plot with known stems, for testing registration.
//!
//! The sampler behind `sylva.coreg.simulate.scan_plot`. Field data rarely
//! comes with ground truth; these scans reproduce what breaks forest
//! registration: point density and noise that fall off with range,
//! grazing-incidence drop-out and, above all, mutual occlusion between
//! stems, which is why two scans of one plot may share only a fraction of
//! their stems. The plot itself (tree placement, DBH, lean, terrain
//! coefficients) is drawn in Python, where it is a few dozen numbers; the
//! points are sampled here.
//!
//! The random draws come from [`Generator`], NumPy's `default_rng`, in the
//! order the NumPy implementation made them, and every coordinate is
//! computed with the same operations in the same order, so a seed gives the
//! scan the Python package always gave, to the bit (`tests/parity`).

use std::f64::consts::PI;

use crate::util::nprandom::Generator;
use crate::Point;

/// A leaning, tapering stem carrying a crown (`sylva.coreg.simulate.Tree`).
#[derive(Clone, Debug)]
pub struct Tree {
    /// Stem base x in the plot frame (m).
    pub x: f64,
    /// Stem base y in the plot frame (m).
    pub y: f64,
    /// Terrain elevation at the base (m).
    pub base_z: f64,
    /// Diameter at breast height (m).
    pub dbh: f64,
    /// Tree height above the base (m).
    pub height: f64,
    /// Unit vector along the stem axis; `[0, 0, 1]` is upright.
    pub lean: [f64; 3],
    /// Radius lost per metre of height (m / m).
    pub taper: f64,
    /// Height of the crown base above the stem base (m).
    pub crown_base: f64,
}

impl Tree {
    /// Stem radius (m) at height `h` above the base: half the DBH at 1.3 m,
    /// narrowing by `taper` per metre, never below 0.01 m.
    pub fn radius_at(&self, h: f64) -> f64 {
        (0.5 * self.dbh - self.taper * (h - 1.3)).max(0.01)
    }

    /// The point on the stem axis `h` metres above the base, in the plot
    /// frame: the base displaced along `lean` until it has risen `h`.
    pub fn axis_point(&self, h: f64) -> Point {
        let s = h / self.lean[2].max(1e-6);
        [self.x + s * self.lean[0], self.y + s * self.lean[1], self.base_z + s * self.lean[2]]
    }
}

/// Terrain elevation (m) at `(x, y)`: a plane plus two sinusoids, from the
/// seven coefficients of `Plot.terrain_coeffs`.
pub fn terrain(a: &[f64; 7], x: f64, y: f64) -> f64 {
    a[0] + a[1] * x + a[2] * y + a[3] * (x / 12.0 + a[5]).sin() + a[4] * (y / 9.0 + a[6]).cos()
}

/// Settings of [`scan_plot`]; the defaults are those of the Python function.
#[derive(Clone, Debug)]
pub struct ScanParams {
    /// Trees further away (m) are not seen; ground and understorey are
    /// sampled out to this range, and the range noise doubles between the
    /// scanner and it.
    pub max_range: f64,
    /// Scanner height above the ground (m).
    pub scanner_height: f64,
    /// Beam spacing (radians): a stem returns about one point per
    /// `angular_step` squared of the solid angle it fills, before
    /// grazing-incidence drop-out.
    pub angular_step: f64,
    /// Standard deviation (m) of the Gaussian noise on each coordinate at
    /// zero range; it grows linearly to twice that at `max_range`.
    pub range_noise: f64,
    /// Ground returns per m² of the disc of radius `max_range` round the
    /// scanner, drawn uniform in range (so denser near the scanner), with
    /// 0.015 m of vertical noise.
    pub ground_density: f64,
    /// Crown points per metre of crown depth for a tree within 8 m, falling
    /// as 1 / range beyond that and capped at 8000 per tree.
    pub foliage_density: f64,
    /// Low-vegetation returns per m² of the same disc as `ground_density`,
    /// their heights exponential with a 0.45 m mean: the source of false stems.
    pub understorey_density: f64,
    /// Drop points shadowed by stems (opaque vertical cylinders in plan
    /// view); foliage is shadowed with probability 0.55, as a leaky crown.
    pub occlusion: bool,
    /// Return the points in the plot frame; otherwise the scanner position is
    /// subtracted, giving a scanner-centred frame with the plot's orientation.
    pub world_frame: bool,
    /// Stem points a tree needs after occlusion to count as visible.
    pub min_stem_returns: usize,
}

impl Default for ScanParams {
    fn default() -> Self {
        ScanParams {
            max_range: 30.0,
            scanner_height: 1.6,
            angular_step: 0.0012,
            range_noise: 0.004,
            ground_density: 150.0,
            foliage_density: 2500.0,
            understorey_density: 70.0,
            occlusion: true,
            world_frame: true,
            min_stem_returns: 150,
        }
    }
}

/// One virtual scan.
#[derive(Clone, Debug, Default)]
pub struct Scan {
    /// The points, in the plot frame or the scanner's (`ScanParams::world_frame`).
    pub xyz: Vec<Point>,
    /// Per point: the tree index, or -1 ground, -2 foliage, -3 understorey.
    pub label: Vec<i32>,
    /// Range from the scanner (m) before the noise was added.
    pub range: Vec<f64>,
    /// Per tree: did at least `min_stem_returns` stem points survive occlusion?
    pub visible: Vec<bool>,
    /// Scanner position in the plot frame (m).
    pub scanner: Point,
}

/// Python's `int(np.clip(x, lo, hi))` for a non-negative `x`.
fn clip_int(x: f64, lo: f64, hi: f64) -> usize {
    x.max(lo).min(hi) as usize
}

/// A virtual scan of the plot from `scanner_xy`: stems, crowns, ground and
/// understorey, occluded and noised, with the trees that stayed visible.
///
/// `trees` and `size` are the plot's; `coeffs` its terrain. `seed` seeds
/// every draw.
pub fn scan_plot(trees: &[Tree], size: f64, coeffs: &[f64; 7], scanner_xy: [f64; 2], seed: u64, p: &ScanParams) -> Scan {
    let mut rng = Generator::new(seed);
    let scanner = [scanner_xy[0], scanner_xy[1], terrain(coeffs, scanner_xy[0], scanner_xy[1]) + p.scanner_height];
    let blockers: Vec<[f64; 3]> = trees.iter().map(|t| [t.x, t.y, 0.5 * t.dbh]).collect();
    let mut xyz: Vec<Point> = Vec::new();
    let mut label: Vec<i32> = Vec::new();
    let mut visible = vec![false; trees.len()];
    for (i, tree) in trees.iter().enumerate() {
        let mut pts = sample_stem(tree, &scanner, &mut rng, p.max_range, p.angular_step);
        if pts.is_empty() {
            continue;
        }
        if p.occlusion {
            let mask = visible_mask(&pts, &scanner, &blockers, Some(i), 1.0);
            pts = keep(pts, &mask);
        }
        visible[i] = pts.len() >= p.min_stem_returns;
        label.extend(std::iter::repeat_n(i as i32, pts.len()));
        xyz.extend(pts);
    }
    let mut foliage = sample_foliage(trees, &scanner, &mut rng, p.max_range, p.foliage_density);
    if p.occlusion && !foliage.is_empty() {
        let mask = visible_mask(&foliage, &scanner, &blockers, None, 0.55);
        foliage = keep(foliage, &mask);
    }
    let mut ground = sample_ground(size, coeffs, &scanner, &mut rng, p.max_range, p.ground_density);
    if p.occlusion && !ground.is_empty() {
        let mask = visible_mask(&ground, &scanner, &blockers, None, 1.0);
        ground = keep(ground, &mask);
    }
    let under = sample_understorey(size, coeffs, &scanner, &mut rng, p.max_range, p.understorey_density);
    for (part, l) in [(foliage, -2), (ground, -1), (under, -3)] {
        label.extend(std::iter::repeat_n(l, part.len()));
        xyz.extend(part);
    }
    if xyz.is_empty() {
        return Scan { visible, scanner, ..Scan::default() };
    }
    // The range before the noise, then noise growing with range: one draw per
    // coordinate in point order, as NumPy fills an (n, 3) array.
    let range: Vec<f64> = xyz.iter().map(|q| norm3(q[0] - scanner[0], q[1] - scanner[1], q[2] - scanner[2])).collect();
    for (q, &r) in xyz.iter_mut().zip(&range) {
        let f = 1.0 + r / p.max_range;
        for k in 0..3 {
            q[k] += rng.normal(0.0, p.range_noise) * f;
        }
    }
    if !p.world_frame {
        for q in &mut xyz {
            for k in 0..3 {
                q[k] -= scanner[k];
            }
        }
    }
    Scan { xyz, label, range, visible, scanner }
}

/// `np.linalg.norm` of a 3-vector: the squares summed left to right.
fn norm3(x: f64, y: f64, z: f64) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

/// The points whose flag is set.
fn keep(pts: Vec<Point>, mask: &[bool]) -> Vec<Point> {
    pts.into_iter().zip(mask).filter_map(|(q, &m)| if m { Some(q) } else { None }).collect()
}

/// The scanner-facing half of a stem at a realistic angular density: points
/// on the stem surface, thinned by grazing incidence.
fn sample_stem(tree: &Tree, scanner: &Point, rng: &mut Generator, max_range: f64, angular_step: f64) -> Vec<Point> {
    let distance = (tree.x - scanner[0]).hypot(tree.y - scanner[1]);
    if distance > max_range || distance < 0.3 {
        return Vec::new();
    }
    let top = tree.crown_base.min(tree.height);
    let n_vert = clip_int(top / (distance * angular_step), 8.0, 4000.0);
    let n_horiz = clip_int(tree.dbh * PI / (distance * angular_step), 4.0, 800.0);
    let n = clip_int((n_vert * n_horiz) as f64, 0.0, 60000.0);
    if n < 10 {
        return Vec::new();
    }
    let h = rng.uniform_n(0.05, top, n);
    let offset = rng.uniform_n(-PI / 2.0, PI / 2.0, n);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Grazing incidence drops out: kept with probability cos(offset)^0.7.
        let kept = rng.random() < offset[i].cos().powf(0.7);
        if !kept {
            continue;
        }
        let axis = tree.axis_point(h[i]);
        let radius = tree.radius_at(h[i]);
        let bearing = (scanner[1] - axis[1]).atan2(scanner[0] - axis[0]);
        let theta = bearing + offset[i];
        out.push([axis[0] + radius * theta.cos(), axis[1] + radius * theta.sin(), axis[2]]);
    }
    out
}

/// Crown points: a cone of foliage narrowing to the top, denser for near trees.
fn sample_foliage(trees: &[Tree], scanner: &Point, rng: &mut Generator, max_range: f64, density: f64) -> Vec<Point> {
    let mut out = Vec::new();
    for tree in trees {
        let d = (tree.x - scanner[0]).hypot(tree.y - scanner[1]);
        if d > max_range {
            continue;
        }
        let n = clip_int(density * (tree.height - tree.crown_base) / (d / 8.0).max(1.0), 0.0, 8000.0);
        if n < 5 {
            continue;
        }
        let crown_r = 0.25 * tree.height * rng.uniform(0.6, 1.0);
        let h = rng.uniform_n(tree.crown_base, tree.height, n);
        let angle = rng.uniform_n(0.0, 2.0 * PI, n);
        let rad: Vec<f64> = (0..n)
            .map(|i| {
                let spread = crown_r * (1.0 - (h[i] - tree.crown_base) / (tree.height - tree.crown_base).max(1e-6)).sqrt();
                spread * rng.random().sqrt()
            })
            .collect();
        let dz = rng.normal_n(0.0, 0.3, n);
        for i in 0..n {
            let c = tree.axis_point(h[i]);
            out.push([c[0] + rad[i] * angle[i].cos(), c[1] + rad[i] * angle[i].sin(), c[2] + dz[i]]);
        }
    }
    out
}

/// Positions on the disc of radius `max_range` round the scanner, uniform in
/// range (the 1 / r thinning of grazing returns), kept within 5 m of the plot.
fn disc_positions(size: f64, scanner: &Point, rng: &mut Generator, max_range: f64, density: f64) -> Vec<[f64; 2]> {
    let n = (density * PI * max_range.powi(2)) as usize;
    let r = rng.uniform_n(0.5, max_range, n);
    let phi = rng.uniform_n(0.0, 2.0 * PI, n);
    let limit = size / 2.0 + 5.0;
    (0..n)
        .map(|i| [scanner[0] + r[i] * phi[i].cos(), scanner[1] + r[i] * phi[i].sin()])
        .filter(|q| q[0].abs() <= limit && q[1].abs() <= limit)
        .collect()
}

/// Ground returns: the terrain with 0.015 m of vertical noise.
fn sample_ground(size: f64, coeffs: &[f64; 7], scanner: &Point, rng: &mut Generator, max_range: f64, density: f64) -> Vec<Point> {
    let xy = disc_positions(size, scanner, rng, max_range, density);
    let noise = rng.normal_n(0.0, 0.015, xy.len());
    xy.iter().zip(&noise).map(|(q, &dz)| [q[0], q[1], terrain(coeffs, q[0], q[1]) + dz]).collect()
}

/// Low vegetation, heights exponential with a 0.45 m mean: the main source
/// of false stems.
fn sample_understorey(size: f64, coeffs: &[f64; 7], scanner: &Point, rng: &mut Generator, max_range: f64, density: f64) -> Vec<Point> {
    let xy = disc_positions(size, scanner, rng, max_range, density);
    let height = rng.exponential_n(0.45, xy.len());
    xy.iter().zip(&height).map(|(q, &dz)| [q[0], q[1], terrain(coeffs, q[0], q[1]) + dz]).collect()
}

/// Points not shadowed by a stem: the blockers are opaque vertical cylinders
/// `[x, y, radius]` in plan view, `skip` the tree being sampled. With
/// `fraction` below 1 a blocker shadows a point with that probability only
/// (a leaky crown), drawn from a generator seeded by the blocker's index.
fn visible_mask(points: &[Point], scanner: &Point, blockers: &[[f64; 3]], skip: Option<usize>, fraction: f64) -> Vec<bool> {
    let n = points.len();
    let mut visible = vec![true; n];
    if blockers.is_empty() || n == 0 {
        return visible;
    }
    let geometry: Vec<([f64; 2], f64)> = points
        .iter()
        .map(|q| {
            let d = [q[0] - scanner[0], q[1] - scanner[1]];
            let length = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-9);
            ([d[0] / length, d[1] / length], length)
        })
        .collect();
    for (j, b) in blockers.iter().enumerate() {
        if Some(j) == skip {
            continue;
        }
        let tb = [b[0] - scanner[0], b[1] - scanner[1]];
        let mut leak = (fraction < 1.0).then(|| Generator::new(j as u64));
        for i in 0..n {
            let (dir, length) = geometry[i];
            // `direction @ to_blocker`: the fused multiply-add is how the BLAS
            // behind NumPy's matmul rounds a two-term dot product.
            let along = dir[0].mul_add(tb[0], dir[1] * tb[1]);
            let perp = (dir[0] * tb[1] - dir[1] * tb[0]).abs();
            let mut hit = perp < b[2] && along > 0.05 && along < length - b[2];
            if let Some(g) = leak.as_mut() {
                hit &= g.random() < fraction;
            }
            if hit {
                visible[i] = false;
            }
        }
    }
    visible
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plot() -> Vec<Tree> {
        (0..6)
            .map(|k| {
                let a = k as f64 * 1.1;
                Tree { x: 6.0 * a.cos(), y: 6.0 * a.sin(), base_z: 0.0, dbh: 0.3 + 0.05 * k as f64, height: 18.0, lean: [0.0, 0.0, 1.0], taper: 0.008, crown_base: 8.0 }
            })
            .collect()
    }

    #[test]
    fn a_scan_has_every_class_and_is_repeatable() {
        let p = ScanParams { angular_step: 0.01, max_range: 15.0, ground_density: 5.0, foliage_density: 50.0, understorey_density: 3.0, ..ScanParams::default() };
        let coeffs = [0.0, 0.01, -0.02, 0.3, 0.2, 0.1, 0.4];
        let a = scan_plot(&plot(), 20.0, &coeffs, [0.0, 0.0], 3, &p);
        let b = scan_plot(&plot(), 20.0, &coeffs, [0.0, 0.0], 3, &p);
        assert_eq!(a.xyz, b.xyz);
        assert_eq!(a.xyz.len(), a.label.len());
        assert_eq!(a.range.len(), a.label.len());
        for l in [-3, -2, -1, 0, 5] {
            assert!(a.label.contains(&l), "label {l}");
        }
        assert!(a.visible.iter().all(|&v| v), "{:?}", a.visible);
        assert!((a.scanner[2] - (terrain(&coeffs, 0.0, 0.0) + 1.6)).abs() < 1e-12);
    }

    #[test]
    fn occlusion_hides_the_far_side_of_a_stem() {
        // A blocker right in front of a point hides it; one behind does not.
        let scanner = [0.0, 0.0, 1.6];
        let pts = [[10.0, 0.0, 1.0], [-10.0, 0.0, 1.0]];
        let blockers = [[5.0, 0.0, 0.2]];
        assert_eq!(visible_mask(&pts, &scanner, &blockers, None, 1.0), vec![false, true]);
        assert_eq!(visible_mask(&pts, &scanner, &blockers, Some(0), 1.0), vec![true, true]);
    }

    #[test]
    fn scanner_frame_is_the_plot_frame_shifted() {
        let p = ScanParams { angular_step: 0.02, ground_density: 2.0, foliage_density: 20.0, understorey_density: 1.0, ..ScanParams::default() };
        let coeffs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let world = scan_plot(&plot(), 20.0, &coeffs, [2.0, -1.0], 1, &p);
        let local = scan_plot(&plot(), 20.0, &coeffs, [2.0, -1.0], 1, &ScanParams { world_frame: false, ..p });
        assert_eq!(world.xyz.len(), local.xyz.len());
        for (w, l) in world.xyz.iter().zip(&local.xyz) {
            for k in 0..3 {
                assert!((w[k] - world.scanner[k] - l[k]).abs() < 1e-9);
            }
        }
    }
}
