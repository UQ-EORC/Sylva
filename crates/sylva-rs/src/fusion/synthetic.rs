// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A terrestrial scanner that sees a synthetic scene as the airborne
//! simulator ([`crate::synthetic::als::fly`]) does.
//!
//! Every scene point is a sphere of `target_radius` (points of class 2, the
//! scene's ground, are ignored) and the ground is the analytic terrain of
//! [`crate::synthetic::terrain_height`]. Pulses leave the scanner on a
//! regular zenith and azimuth grid (as [`crate::synthetic::scan`] fires
//! them) as thin rays: a ray stops at the first sphere it meets or at the
//! terrain, whichever is nearer, and gives one echo there, or none if it
//! leaves the scene. A layer of spheres of density `n` per m³ is then a
//! turbid medium of plant area density `2 π r² n` (spherical leaf angles,
//! G = 0.5) for both scanners, so their profiles can be checked against the
//! same known foliage.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::synthetic::terrain_height;
use crate::{Point, PointCloud, Shots};

/// Settings of [`scan_spheres`].
#[derive(Debug, Clone)]
pub struct SphereScan {
    pub resolution_deg: f64,
    pub max_zenith_deg: f64,
    pub target_radius: f64,
    pub terrain_slope: f64,
    /// Echoes nearer than this (m) are ignored.
    pub min_range: f64,
    /// Rays are followed this far (m).
    pub max_range: f64,
}

impl Default for SphereScan {
    fn default() -> Self {
        SphereScan { resolution_deg: 0.25, max_zenith_deg: 130.0, target_radius: 0.03, terrain_slope: 0.05, min_range: 0.1, max_range: 200.0 }
    }
}

/// Scene points binned in a regular 3-D grid; each point is listed in every
/// cell its sphere reaches.
struct Grid {
    lo: Point,
    cell: f64,
    n: [usize; 3],
    start: Vec<u32>,
    idx: Vec<u32>,
    r: f64,
}

impl Grid {
    fn new(points: &[Point], targets: &[usize], r: f64) -> Result<Grid> {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &i in targets {
            for k in 0..3 {
                lo[k] = lo[k].min(points[i][k] - r);
                hi[k] = hi[k].max(points[i][k] + r);
            }
        }
        if targets.is_empty() {
            return Ok(Grid { lo: [0.0; 3], cell: 1.0, n: [0; 3], start: vec![0], idx: vec![], r });
        }
        let mut cell = (4.0 * r).max(0.25);
        let dims = |c: f64| -> [usize; 3] { std::array::from_fn(|k| ((hi[k] - lo[k]) / c).floor() as usize + 1) };
        while dims(cell).iter().map(|&v| v as u128).product::<u128>() > 60_000_000 {
            cell *= 1.5;
        }
        let n = dims(cell);
        if targets.len() as u128 * 8 > u32::MAX as u128 {
            return Err(Error::invalid("too many scene points for the synthetic scanner"));
        }
        let span = |v: f64, k: usize| (((v - r - lo[k]) / cell).floor().max(0.0) as usize, (((v + r - lo[k]) / cell).floor() as usize).min(n[k] - 1));
        let cells_of = |p: &Point| -> ([usize; 2], [usize; 2], [usize; 2]) {
            let (a, b) = span(p[0], 0);
            let (c, d) = span(p[1], 1);
            let (e, f) = span(p[2], 2);
            ([a, b], [c, d], [e, f])
        };
        let total = n[0] * n[1] * n[2];
        let mut count = vec![0u32; total + 1];
        for &i in targets {
            let (x, y, z) = cells_of(&points[i]);
            for iz in z[0]..=z[1] {
                for iy in y[0]..=y[1] {
                    for ix in x[0]..=x[1] {
                        count[(iz * n[1] + iy) * n[0] + ix + 1] += 1;
                    }
                }
            }
        }
        for k in 1..count.len() {
            count[k] += count[k - 1];
        }
        let mut fill = count.clone();
        let mut idx = vec![0u32; count[total] as usize];
        for &i in targets {
            let (x, y, z) = cells_of(&points[i]);
            for iz in z[0]..=z[1] {
                for iy in y[0]..=y[1] {
                    for ix in x[0]..=x[1] {
                        let c = (iz * n[1] + iy) * n[0] + ix;
                        idx[fill[c] as usize] = i as u32;
                        fill[c] += 1;
                    }
                }
            }
        }
        Ok(Grid { lo, cell, n, start: count, idx, r })
    }

    /// The first sphere on `o + t d` with `t_min < t < t_max`: `(t, point)`.
    fn cast(&self, points: &[Point], o: &Point, d: &Point, t_min: f64, t_max: f64) -> Option<(f64, usize)> {
        if self.n[0] == 0 {
            return None;
        }
        // Clip the ray to the grid's box.
        let (mut t0, mut t1) = (0.0f64, t_max);
        for k in 0..3 {
            let hi = self.lo[k] + self.n[k] as f64 * self.cell;
            if d[k].abs() < 1e-15 {
                if o[k] < self.lo[k] || o[k] > hi {
                    return None;
                }
            } else {
                let (a, b) = ((self.lo[k] - o[k]) / d[k], (hi - o[k]) / d[k]);
                t0 = t0.max(a.min(b));
                t1 = t1.min(a.max(b));
            }
        }
        if t0 >= t1 {
            return None;
        }
        let p0: Point = std::array::from_fn(|k| o[k] + (t0 + 1e-9) * d[k]);
        let mut c: [i64; 3] = std::array::from_fn(|k| (((p0[k] - self.lo[k]) / self.cell).floor() as i64).clamp(0, self.n[k] as i64 - 1));
        let step: [i64; 3] = std::array::from_fn(|k| if d[k] > 0.0 { 1 } else { -1 });
        let mut next: [f64; 3] = std::array::from_fn(|k| {
            if d[k].abs() < 1e-15 {
                f64::INFINITY
            } else {
                let edge = self.lo[k] + (c[k] + if d[k] > 0.0 { 1 } else { 0 }) as f64 * self.cell;
                (edge - o[k]) / d[k]
            }
        });
        let delta: [f64; 3] = std::array::from_fn(|k| if d[k].abs() < 1e-15 { f64::INFINITY } else { self.cell / d[k].abs() });
        let r2 = self.r * self.r;
        let mut best: Option<(f64, usize)> = None;
        loop {
            let cell = ((c[2] as usize * self.n[1]) + c[1] as usize) * self.n[0] + c[0] as usize;
            for j in self.start[cell] as usize..self.start[cell + 1] as usize {
                let i = self.idx[j] as usize;
                let q = points[i];
                let v = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
                let tc = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
                let perp2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - tc * tc;
                if perp2 < r2 {
                    let t = tc - (r2 - perp2).sqrt();
                    if t > t_min && t < t_max && best.is_none_or(|(bt, bi)| t < bt || (t == bt && i < bi)) {
                        best = Some((t, i));
                    }
                }
            }
            let exit = next[0].min(next[1]).min(next[2]);
            if best.is_some_and(|(bt, _)| bt <= exit) || exit >= t1 {
                return best;
            }
            let k = if next[0] <= next[1] && next[0] <= next[2] { 0 } else if next[1] <= next[2] { 1 } else { 2 };
            c[k] += step[k];
            if c[k] < 0 || c[k] >= self.n[k] as i64 {
                return best;
            }
            next[k] += delta[k];
        }
    }
}

/// Range along `o + t d` to the terrain before `t_max`, by steps no longer
/// than the height above it over the function's largest slope along the ray
/// (so the surface is never stepped over), then bisection.
fn terrain_hit(o: &Point, d: &Point, slope: f64, t_max: f64) -> Option<f64> {
    let f = |t: f64| o[2] + t * d[2] - terrain_height(o[0] + t * d[0], o[1] + t * d[1], slope);
    let lip = d[2].abs() + slope.abs() * d[0].abs() + 0.2 / 3.0 * d[1].abs() + 1e-12;
    let mut t = 0.0;
    let mut v = f(t);
    if v <= 0.0 {
        return None;
    }
    for _ in 0..100_000 {
        let step = (v / lip).max(1e-4);
        let nt = t + step;
        if nt > t_max {
            return None;
        }
        let nv = f(nt);
        if nv <= 0.0 {
            let (mut a, mut b) = (t, nt);
            for _ in 0..60 {
                let m = 0.5 * (a + b);
                if f(m) > 0.0 { a = m } else { b = m }
            }
            return Some(0.5 * (a + b));
        }
        if nv < 1e-9 {
            return Some(nt);
        }
        (t, v) = (nt, nv);
    }
    None
}

/// Scan `scene` from each origin; one [`Shots`] per origin, every pulse
/// fired (those with no echo too). Echoes carry `classification` (2 for the
/// terrain, the scene's class otherwise) and, when the scene has it,
/// `tree_id` (0 for the terrain).
///
/// # Errors
/// For settings out of range or an origin below the terrain.
pub fn scan_spheres(scene: &PointCloud, origins: &[Point], p: &SphereScan) -> Result<Vec<Shots>> {
    if !(p.resolution_deg > 0.0 && p.resolution_deg <= 10.0) || !(p.max_zenith_deg > 0.0 && p.max_zenith_deg <= 180.0) {
        return Err(Error::invalid("resolution_deg must be in (0, 10] and max_zenith_deg in (0, 180]"));
    }
    if !(p.target_radius > 0.0 && p.target_radius.is_finite()) || !p.terrain_slope.is_finite() || !(p.min_range >= 0.0) || !(p.max_range > p.min_range) {
        return Err(Error::invalid("target_radius must be positive, min_range >= 0 and max_range above it"));
    }
    for o in origins {
        if !o.iter().all(|v| v.is_finite()) || o[2] <= terrain_height(o[0], o[1], p.terrain_slope) {
            return Err(Error::invalid(format!("scanner origin {o:?} is not above the terrain")));
        }
    }
    let class = scene.attrs.get("classification");
    let tree = scene.attrs.get("tree_id");
    let targets: Vec<usize> = (0..scene.len()).filter(|&i| class.is_none_or(|c| c.get_f64(i) != 2.0) && scene.xyz[i].iter().all(|v| v.is_finite())).collect();
    let grid = Grid::new(&scene.xyz, &targets, p.target_radius)?;
    let to_rad = std::f64::consts::PI / 180.0;
    let n_zen = (p.max_zenith_deg / p.resolution_deg).round() as usize;
    let n_az = (360.0 / p.resolution_deg).round() as usize;
    let directions: Vec<Point> = (0..n_zen)
        .flat_map(|iz| {
            let z = (iz as f64 + 0.5) * p.resolution_deg * to_rad;
            (0..n_az).map(move |ia| {
                let a = (ia as f64 + 0.5) * 360.0 / n_az as f64 * to_rad;
                [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()]
            })
        })
        .collect();
    let mut out = Vec::with_capacity(origins.len());
    for o in origins {
        // (range, scene point or usize::MAX for the terrain)
        let hits: Vec<Option<(f64, usize)>> = directions
            .par_iter()
            .map(|d| {
                let s = grid.cast(&scene.xyz, o, d, p.min_range, p.max_range);
                let limit = s.map(|v| v.0).unwrap_or(p.max_range);
                match terrain_hit(o, d, p.terrain_slope, limit) {
                    Some(t) if t > p.min_range => Some((t, usize::MAX)),
                    _ => s,
                }
            })
            .collect();
        let count: Vec<u32> = hits.iter().map(|h| h.is_some() as u32).collect();
        let mut start = Vec::with_capacity(hits.len());
        let mut acc = 0usize;
        for &c in &count {
            start.push(acc);
            acc += c as usize;
        }
        let echoes: Vec<(f64, usize)> = hits.iter().flatten().copied().collect();
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert("classification".to_string(), Attr::U8(echoes.iter().map(|&(_, i)| if i == usize::MAX { 2 } else { class.map(|c| c.get_f64(i) as u8).unwrap_or(4) }).collect()));
        if let Some(t) = tree {
            attrs.insert("tree_id".to_string(), Attr::I64(echoes.iter().map(|&(_, i)| if i == usize::MAX { 0 } else { t.get_f64(i) as i64 }).collect()));
        }
        out.push(Shots {
            origin: vec![*o; directions.len()],
            direction: directions.clone(),
            echo_start: start,
            echo_count: count,
            echo_range: echoes.iter().map(|e| e.0).collect(),
            echo_attrs: attrs,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_meet_the_nearest_sphere_and_the_terrain() {
        let xyz = vec![[0.0, 5.0, 1.5], [0.0, 3.0, 1.5], [0.0, 0.0, 10.0], [0.0, 0.0, -50.0]];
        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert("classification".to_string(), Attr::U8(vec![4, 5, 4, 2]));
        attrs.insert("tree_id".to_string(), Attr::I64(vec![1, 2, 3, 0]));
        let scene = PointCloud { xyz, attrs };
        let g = Grid::new(&scene.xyz, &[0, 1, 2], 0.1).unwrap();
        // North, horizontally: the sphere at y = 3 first.
        let (t, i) = g.cast(&scene.xyz, &[0.0, 0.0, 1.5], &[0.0, 1.0, 0.0], 0.0, 100.0).unwrap();
        assert_eq!(i, 1);
        assert!((t - 2.9).abs() < 1e-12);
        // Straight up.
        let (t, i) = g.cast(&scene.xyz, &[0.0, 0.0, 1.5], &[0.0, 0.0, 1.0], 0.0, 100.0).unwrap();
        assert!(i == 2 && (t - 8.4).abs() < 1e-12);
        assert!(g.cast(&scene.xyz, &[0.0, 0.0, 1.5], &[1.0, 0.0, 0.0], 0.0, 100.0).is_none());
        // Terrain: flat at z = 0.2 sin(y / 3); straight down from 1.5 m.
        let t = terrain_hit(&[0.0, 0.0, 1.5], &[0.0, 0.0, -1.0], 0.0, 100.0).unwrap();
        assert!((t - 1.5).abs() < 1e-9);
        let d = [0.6, 0.0, -0.8];
        let t = terrain_hit(&[0.0, 0.0, 1.5], &d, 0.05, 100.0).unwrap();
        let hit = [0.6 * t, 0.0, 1.5 - 0.8 * t];
        assert!((hit[2] - terrain_height(hit[0], hit[1], 0.05)).abs() < 1e-9);
        assert!(terrain_hit(&[0.0, 0.0, 1.5], &[0.0, 0.0, 1.0], 0.05, 100.0).is_none());
        let shots = scan_spheres(&scene, &[[0.0, 0.0, 1.5]], &SphereScan { resolution_deg: 2.0, max_zenith_deg: 180.0, target_radius: 0.1, terrain_slope: 0.0, ..Default::default() }).unwrap();
        let s = &shots[0];
        assert_eq!(s.n_shots(), 90 * 180);
        let Some(Attr::U8(c)) = s.echo_attrs.get("classification") else { panic!() };
        // Every downward pulse has an echo, on the terrain but for the two
        // just below the horizon that meet the sphere at y = 3; the scene's
        // ground point is ignored.
        assert!(s.direction.iter().zip(&s.echo_count).all(|(d, &n)| d[2] >= 0.0 || n == 1));
        assert_eq!(c.iter().filter(|&&v| v == 2).count(), 45 * 180 - 2);
        assert!(scan_spheres(&scene, &[[0.0, 0.0, -1.0]], &SphereScan::default()).is_err());
    }

    #[test]
    fn a_sphere_layer_attenuates_as_a_turbid_medium() {
        // Spheres of radius r, n per m³, in a slab from 5 to 10 m: a vertical
        // ray crosses 5 m and passes with probability exp(-n π r² 5).
        let (r, n_per) = (0.05, 20.0);
        let mut s = 99u64;
        let mut u = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let vol = 60.0 * 60.0 * 5.0;
        let xyz: Vec<Point> = (0..(n_per * vol) as usize).map(|_| [-30.0 + 60.0 * u(), -30.0 + 60.0 * u(), 5.0 + 5.0 * u()]).collect();
        let scene = PointCloud::new(xyz);
        let p = SphereScan { resolution_deg: 0.5, max_zenith_deg: 20.0, target_radius: r, terrain_slope: 0.0, ..Default::default() };
        let shots = scan_spheres(&scene, &[[0.0, 0.0, 1.5]], &p).unwrap();
        let s = &shots[0];
        let d: Vec<f64> = s.direction.iter().map(|d| d[2]).collect();
        // Expected gap fraction, per pulse by its path length through the slab.
        let expected: f64 = d.iter().map(|&c| (-n_per * std::f64::consts::PI * r * r * 5.0 / c).exp()).sum::<f64>() / d.len() as f64;
        let gap = s.echo_count.iter().filter(|&&c| c == 0).count() as f64 / d.len() as f64;
        assert!((gap - expected).abs() < 0.02, "{gap} vs {expected}");
    }
}
