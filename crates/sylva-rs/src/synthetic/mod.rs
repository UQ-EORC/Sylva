// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Small synthetic scenes for examples, tutorials and tests.
//!
//! Nothing here is a forest model; the shapes are simple enough that the
//! right answer is known (stem positions, diameters, heights, leaf area).
//! The random draws come from [`Generator`], NumPy's `default_rng`, in the
//! order the NumPy implementation made them, and every coordinate is
//! computed with the same operations in the same order, so that a seed
//! gives the scene the Python package always gave, to the bit.

pub mod als;
pub mod plot;
pub mod scan;
pub mod tree;

use std::f64::consts::PI;

use crate::util::nprandom::Generator;
use crate::pointcloud::Attr;
use crate::shots::ops::{np_remainder, packed_starts};
use crate::{Point, PointCloud, Shots};

/// Radius (m) of the leaf discs of [`tree`]; each disc is 12 points.
pub const LEAF_RADIUS: f64 = 0.08;
const POINTS_PER_LEAF: usize = 12;

/// `(x, y, dbh, height)` of the trees in [`forest`] by default.
pub const DEFAULT_TREES: [(f64, f64, f64, f64); 4] = [(5.0, 5.0, 0.30, 12.0), (14.0, 6.0, 0.20, 9.0), (8.0, 15.0, 0.45, 15.0), (15.5, 15.0, 0.25, 11.0)];

/// Ground elevation of the synthetic scenes: `slope * x + 0.2 * sin(y / 3)`.
pub fn terrain_height(x: f64, y: f64, slope: f64) -> f64 {
    slope * x + 0.2 * (y / 3.0).sin()
}

/// `np.linalg.norm` of a 3-vector: squares summed left to right.
fn norm(v: &Point) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// `np.cross` of 3-vectors, in NumPy's order of operations.
fn cross(a: &Point, b: &Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn div(v: &Point, s: f64) -> Point {
    [v[0] / s, v[1] / s, v[2] / s]
}

/// `n` points on a tapered cylinder surface from `start` along `axis`,
/// radius `r0` to `r1`, with Gaussian radial noise.
#[allow(clippy::too_many_arguments)]
fn cylinder(rng: &mut Generator, start: Point, axis: Point, length: f64, r0: f64, r1: f64, n: usize, noise: f64) -> Vec<Point> {
    let axis = div(&axis, norm(&axis));
    let helper = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = cross(&axis, &helper);
    let u = div(&u, norm(&u));
    let v = cross(&axis, &u);
    let t = rng.uniform_n(0.0, 1.0, n);
    let a = rng.uniform_n(0.0, 2.0 * PI, n);
    let dr = rng.normal_n(0.0, noise, n);
    (0..n)
        .map(|i| {
            let r = r0 + (r1 - r0) * t[i] + dr[i];
            let tl = t[i] * length;
            let (c, s) = (a[i].cos() * r, a[i].sin() * r);
            std::array::from_fn(|k| start[k] + tl * axis[k] + c * u[k] + s * v[k])
        })
        .collect()
}

/// One tree: a tapered stem, `n_branches` limbs in the upper half and small
/// flat leaf discs around the limb ends. `classification` is 5 for wood and
/// 4 for leaves. `dbh` is the diameter at the base (the stem tapers to a
/// quarter of it); `leaf_points` is approximate (12 per disc).
#[allow(clippy::too_many_arguments)]
pub fn tree(x: f64, y: f64, dbh: f64, height: f64, z0: f64, n_branches: usize, leaf_points: usize, seed: u64) -> PointCloud {
    let mut rng = Generator::new(seed);
    let r = dbh / 2.0;
    let stem_points = (2500.0 * height) as usize;
    let mut wood = cylinder(&mut rng, [x, y, z0], [0.0, 0.0, 1.0], height * 0.9, r * 1.05, r * 0.25, stem_points, 0.003);
    let mut leaves: Vec<Point> = Vec::new();
    let nb = n_branches as f64;
    for b in 0..n_branches {
        let bf = b as f64;
        let h = height * (0.45 + 0.45 * (bf + 0.5) / nb);
        let az = 2.4 * bf + rng.uniform(-0.3, 0.3);
        let length = height * rng.uniform(0.16, 0.26);
        let axis = [az.cos(), az.sin(), rng.uniform(0.25, 0.6)];
        let rb = r * 0.3 * (1.0 - 0.5 * bf / nb);
        let base = [x, y, z0 + h];
        let limb = cylinder(&mut rng, base, axis, length, rb, rb * 0.3, (1500.0 * length) as usize, 0.003);
        // The limb point farthest from its base (the first, on ties).
        let mut tip = limb.first().copied().unwrap_or(base);
        let mut best = f64::NEG_INFINITY;
        for p in &limb {
            let d = norm(&[p[0] - base[0], p[1] - base[1], p[2] - base[2]]);
            if d > best || (d.is_nan() && !best.is_nan()) {
                best = d;
                tip = *p;
            }
        }
        wood.extend_from_slice(&limb);
        // Leaves: small randomly oriented discs scattered around the limb tip.
        let k = leaf_points / n_branches;
        let offsets = rng.normal_n(0.0, height * 0.07, k / POINTS_PER_LEAF * 3);
        for o in offsets.as_chunks::<3>().0 {
            let c = [tip[0] + o[0], tip[1] + o[1], tip[2] + o[2]];
            let normal = [rng.normal(0.0, 1.0), rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)];
            let normal = div(&normal, norm(&normal));
            let e1 = cross(&normal, &[0.0, 0.0, 1.0]);
            let e1 = div(&e1, norm(&e1) + 1e-12);
            let e2 = cross(&normal, &e1);
            let rad: Vec<f64> = rng.uniform_n(0.0, 1.0, POINTS_PER_LEAF).iter().map(|u| LEAF_RADIUS * u.sqrt()).collect();
            let ang = rng.uniform_n(0.0, 2.0 * PI, POINTS_PER_LEAF);
            for (&rd, &an) in rad.iter().zip(&ang) {
                let (p, q) = (rd * an.cos(), rd * an.sin());
                leaves.push(std::array::from_fn(|j| c[j] + p * e1[j] + q * e2[j]));
            }
        }
    }
    let mut cls = vec![5u8; wood.len()];
    cls.resize(wood.len() + leaves.len(), 4);
    wood.extend(leaves);
    let mut cloud = PointCloud::new(wood);
    cloud.attrs.insert("classification".into(), Attr::U8(cls));
    cloud
}

/// True one-sided leaf area (m²) of a synthetic cloud, from the number of
/// leaf points (`classification == 4`).
pub fn leaf_area(classification: &[f64]) -> f64 {
    let n = classification.iter().filter(|&&c| c == 4.0).count();
    n as f64 / POINTS_PER_LEAF as f64 * PI * (LEAF_RADIUS * LEAF_RADIUS)
}

/// Sloped terrain ([`terrain_height`] with slope 0.05) with `trees`
/// (`(x, y, dbh, height)`, default [`DEFAULT_TREES`]) standing on it, each
/// a [`tree`] with seed `seed + i`. The ground spans `margin` m beyond the
/// `size` m square. Attributes: `classification` (2 ground, 4 leaf, 5 wood)
/// and `tree_id` (0 for ground, then 1.. in list order).
pub fn forest(trees: &[(f64, f64, f64, f64)], size: f64, ground_points: usize, margin: f64, seed: u64) -> PointCloud {
    let mut rng = Generator::new(seed);
    let xy = rng.uniform_n(-margin, size + margin, 2 * ground_points);
    let dz = rng.normal_n(0.0, 0.01, ground_points);
    let mut xyz: Vec<Point> = (0..ground_points).map(|i| {
        let (x, y) = (xy[2 * i], xy[2 * i + 1]);
        [x, y, terrain_height(x, y, 0.05) + dz[i]]
    }).collect();
    let mut cls = vec![2u8; ground_points];
    let mut ids = vec![0i32; ground_points];
    for (i, &(x, y, dbh, h)) in trees.iter().enumerate() {
        let t = tree(x, y, dbh, h, terrain_height(x, y, 0.05), 6, 18000, seed + i as u64 + 1);
        ids.resize(ids.len() + t.len(), i as i32 + 1);
        if let Some(Attr::U8(c)) = t.attrs.get("classification") {
            cls.extend_from_slice(c);
        }
        xyz.extend(t.xyz);
    }
    let mut cloud = PointCloud::new(xyz);
    cloud.attrs.insert("classification".into(), Attr::U8(cls));
    cloud.attrs.insert("tree_id".into(), Attr::I32(ids));
    cloud
}

/// `np.floor(v).astype(np.int64)` as x86 NumPy casts it (NaN and overflow
/// become `i64::MIN`).
fn floor_i64(v: f64) -> i64 {
    let f = v.floor();
    if f.is_nan() || !(-9.223372036854776e18..9.223372036854776e18).contains(&f) {
        i64::MIN
    } else {
        f as i64
    }
}

/// A pseudo terrestrial scan of `cloud` from `origin`: pulses on a regular
/// zenith / azimuth grid of `resolution_deg`, from straight up to
/// `max_zenith_deg`. The points in a pulse's angular cell (farther than
/// 0.1 m) are its candidate targets: the nearest gives the first echo and
/// further ones at least `echo_separation` m beyond the last give up to
/// `max_echoes` echoes. Cells without a point are pulses with no return.
/// A pulse with echoes is aimed at its farthest one, so that echo positions
/// reproduce the points; echo attributes are copied from the points.
pub fn scan(cloud: &PointCloud, origin: Point, resolution_deg: f64, max_zenith_deg: f64, max_echoes: usize, echo_separation: f64) -> Shots {
    let to_deg = 180.0 / PI;
    let to_rad = PI / 180.0;
    let n = cloud.len();
    let d: Vec<Point> = cloud.xyz.iter().map(|p| [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]).collect();
    let range: Vec<f64> = d.iter().map(norm).collect();
    let n_zen = (max_zenith_deg / resolution_deg).round_ties_even() as i64;
    let n_az = (360.0 / resolution_deg).round_ties_even() as i64;
    let mut cell = vec![0i64; n];
    let mut ok = Vec::with_capacity(n);
    for i in 0..n {
        let r = range[i];
        let rm = if r.is_nan() || r >= 1e-12 { r } else { 1e-12 };
        let zen = (d[i][2] / rm).clamp(-1.0, 1.0).acos() * to_deg;
        let az = np_remainder(d[i][0].atan2(d[i][1]) * to_deg, 360.0);
        let iz = floor_i64(zen / resolution_deg);
        let ia = floor_i64(az / resolution_deg).min(n_az - 1);
        cell[i] = iz.wrapping_mul(n_az).wrapping_add(ia);
        if iz < n_zen && r > 0.1 {
            ok.push(i);
        }
    }
    // Nearest first within each cell; ties keep point order.
    ok.sort_by(|&a, &b| cell[a].cmp(&cell[b]).then(range[a].partial_cmp(&range[b]).unwrap_or(std::cmp::Ordering::Equal)));
    let n_cells = (n_zen.max(0) * n_az.max(0)) as usize;
    let mut count = vec![0u32; n_cells];
    let mut echo_points = Vec::new();
    let (mut last_cell, mut last_range, mut taken) = (-1i64, 0.0f64, 0usize);
    for &i in &ok {
        let c = cell[i];
        if c != last_cell {
            (last_cell, taken, last_range) = (c, 0, f64::NEG_INFINITY);
        }
        if taken < max_echoes && range[i] - last_range >= echo_separation {
            echo_points.push(i);
            count[c as usize] += 1;
            taken += 1;
            last_range = range[i];
        }
    }
    let start = packed_starts(&count);
    let mut direction = Vec::with_capacity(n_cells);
    for iz in 0..n_zen.max(0) {
        let z = (iz as f64 + 0.5) * resolution_deg * to_rad;
        for ia in 0..n_az.max(0) {
            let a = (ia as f64 + 0.5) * resolution_deg * to_rad;
            direction.push([z.sin() * a.sin(), z.sin() * a.cos(), z.cos()]);
        }
    }
    for c in 0..n_cells {
        if count[c] > 0 {
            let far = echo_points[start[c] + count[c] as usize - 1];
            direction[c] = div(&d[far], range[far]);
        }
    }
    Shots {
        origin: vec![origin; n_cells],
        direction,
        echo_start: start,
        echo_count: count,
        echo_range: echo_points.iter().map(|&i| range[i]).collect(),
        echo_attrs: cloud.attrs.iter().map(|(k, v)| (k.clone(), v.take(&echo_points))).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_has_its_parts() {
        let t = tree(1.0, 2.0, 0.3, 4.0, 0.5, 3, 360, 0);
        let Some(Attr::U8(cls)) = t.attrs.get("classification") else { panic!() };
        let leaves = cls.iter().filter(|&&c| c == 4).count();
        assert_eq!(leaves, 3 * (120 / 12) * 12);
        assert_eq!(cls.len(), t.len());
        // Stem points are within the stem radius (plus noise) of the axis.
        let r = ((t.xyz[0][0] - 1.0).powi(2) + (t.xyz[0][1] - 2.0).powi(2)).sqrt();
        assert!(r < 0.3 && r > 0.02);
        let la = leaf_area(&cls.iter().map(|&c| c as f64).collect::<Vec<_>>());
        assert!((la - 30.0 * PI * 0.0064).abs() < 1e-12);
    }

    #[test]
    fn tree_matches_numpy() {
        // synthetic.tree(height=1.0, n_branches=1, leaf_points=12, seed=0): len, xyz[[0, 2600, -1]]
        let t = tree(0.0, 0.0, 0.3, 1.0, 0.0, 1, 12, 0);
        assert_eq!(t.len(), 2815);
        assert_eq!(t.xyz[0], [0.0802232858035399, -0.00820980323646008, 0.5732655185893089]);
        assert_eq!(t.xyz[2600], [0.04108589345700356, -0.03459132135523987, 0.6827043921762495]);
        assert_eq!(t.xyz[2814], [0.32253282545314765, -0.11437589571671343, 0.8088992708733204]);
    }

    #[test]
    fn scan_finds_near_echoes_first() {
        let mut cloud = PointCloud::new(vec![[0.0, 5.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.2, 0.0], [0.0, 0.05, 0.0]]);
        cloud.attrs.insert("id".into(), Attr::I32(vec![0, 1, 2, 3]));
        let s = scan(&cloud, [0.0, 0.0, 0.0], 45.0, 135.0, 2, 0.5);
        assert_eq!(s.n_shots(), 3 * 8);
        assert_eq!(s.echo_range, vec![3.0, 5.0]);
        assert_eq!(s.echo_attrs["id"], Attr::I32(vec![1, 0]));
        let c = s.echo_count.iter().position(|&c| c > 0).unwrap();
        assert_eq!(s.echo_count[c], 2);
        assert_eq!(s.direction[c], [0.0, 1.0, 0.0]);
        assert!((s.direction[0][2] - 22.5f64.to_radians().cos()).abs() < 1e-15);
    }
}
