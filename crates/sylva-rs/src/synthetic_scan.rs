// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A terrestrial scanner with a finite beam, cast into a point scene.
//!
//! Pulses leave `origin` on a regular zenith / azimuth grid (the two-axis
//! pattern of a terrestrial scanner, as in [`crate::synthetic::scan`]). A
//! pulse is a cone of full divergence `divergence` leaving an aperture of
//! `exit_diameter`, so that its footprint at range `R` has diameter
//! `exit_diameter + R divergence`; it is sampled by `footprint_samples`
//! equal-energy sub-beams spread over the footprint disc on a sunflower
//! (Vogel 1979) pattern. Each point stands for a small patch of surface of
//! radius `target_radius`: a disc facing along the point's normal (the
//! `normal_x`, `normal_y`, `normal_z` attributes when the cloud has them,
//! as the synthetic trees and plots do, else a local PCA over 10 neighbours
//! where the neighbourhood is planar), or a sphere where no normal is
//! known. A sub-beam stops at the first patch it meets: at the disc's plane,
//! or, for a sphere, at the point's closest approach, so a surface is not
//! brought forward by the radius. Discs seen edge-on are not hit, so a
//! stem's silhouette is not widened by the radius (a sphere widens it).
//!
//! The receiver separates echoes as a discrete-return scanner does: hits
//! closer in range than `echo_separation` form one echo. With
//! `mixed_pixels` its range is the energy-weighted mean of the hits' ranges
//! (energy: the sub-beam's share times the target's reflectance), which is
//! how a footprint straddling an edge produces a point in the empty space
//! between foreground and background; without it the echo takes the range
//! of its strongest hit. Echoes with less energy than `detection_threshold`
//! are lost; at most `max_echoes` are kept, nearest first. Gaussian range
//! noise of standard deviation `range_noise + range_noise_slope * R` is then
//! added along the beam. Every pulse is kept, with or without echoes, so
//! the free space it crossed is known.
//!
//! Echo positions lie on the pulse's central axis. Echo attributes are those
//! of the point with the most energy in the echo, plus `reflectance` (dB,
//! `10 log10` of the echo energy: 0 dB is a white target filling the
//! footprint, as RIEGL's calibrated reflectance), `footprint_fraction` (the
//! share of sub-beams in the echo) and `range_spread` (m, the spread of the
//! hits' ranges within the echo, before noise).
//!
//! Random numbers come from [`Generator`], `max_echoes` normal draws per
//! pulse in firing order, so that a seed gives the same scan on any number
//! of threads.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::nprandom::Generator;
use crate::pointcloud::Attr;
use crate::shots_ops::packed_starts;
use crate::{limits, Point, PointCloud, Shots};

/// Reflectance of the scene points.
#[derive(Debug, Clone, PartialEq)]
pub enum Reflectance {
    /// By `classification`: `(class, reflectance)` pairs and a default for
    /// other classes and clouds without the attribute.
    ByClass(Vec<(i64, f64)>, f64),
    /// From a per-point attribute.
    Attribute(String),
    /// The same for every point.
    Constant(f64),
}

impl Default for Reflectance {
    /// Diffuse reflectance near 1550 nm (the wavelength of RIEGL's VZ
    /// scanners): ground 0.3, understorey 0.3, leaf 0.3, wood 0.5, other 0.4.
    fn default() -> Self {
        Reflectance::ByClass(vec![(2, 0.3), (3, 0.3), (4, 0.3), (5, 0.5)], 0.4)
    }
}

/// Settings of [`scan_beam`]; angles in degrees, lengths in metres.
#[derive(Debug, Clone)]
pub struct BeamScan {
    pub origin: Point,
    pub resolution_deg: f64,
    pub min_zenith_deg: f64,
    pub max_zenith_deg: f64,
    pub max_echoes: usize,
    pub echo_separation: f64,
    pub range_noise: f64,
    pub range_noise_slope: f64,
    /// Full beam divergence (mrad).
    pub divergence_mrad: f64,
    pub exit_diameter: f64,
    pub footprint_samples: usize,
    pub mixed_pixels: bool,
    /// Radius of the surface patch a point stands for; `None` takes 1.75 times
    /// the median spacing of the scene.
    pub target_radius: Option<f64>,
    /// Orient the patches (discs) where a normal is known; spheres otherwise.
    pub oriented: bool,
    pub detection_threshold: f64,
    pub reflectance: Reflectance,
    pub max_range: f64,
    pub seed: u64,
}

impl BeamScan {
    pub fn new(origin: Point) -> BeamScan {
        BeamScan { origin, resolution_deg: 0.25, min_zenith_deg: 0.0, max_zenith_deg: 130.0, max_echoes: 2, echo_separation: 0.5, range_noise: 0.0, range_noise_slope: 0.0, divergence_mrad: 0.0, exit_diameter: 0.0, footprint_samples: 1, mixed_pixels: true, target_radius: None, oriented: true, detection_threshold: 0.0, reflectance: Reflectance::default(), max_range: f64::INFINITY, seed: 0 }
    }
}

/// Field of view and beam of a scanner model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScannerPreset {
    pub min_zenith_deg: f64,
    pub max_zenith_deg: f64,
    pub divergence_mrad: f64,
    pub exit_diameter: f64,
    pub range_noise: f64,
}

/// Names accepted by [`scanner_preset`].
pub const SCANNERS: [&str; 3] = ["vz400", "vz400i", "vz2000i"];

/// Nominal figures of RIEGL scanners from their data sheets: vertical field
/// of view +60 to -40 degrees (zenith 30 to 130), beam divergence 0.35 mrad
/// (VZ-400, VZ-400i) or 0.27 mrad (VZ-2000i), ranging precision 3 mm. The
/// exit diameter (7 mm) is a nominal value.
pub fn scanner_preset(name: &str) -> Result<ScannerPreset> {
    let p = |div: f64| ScannerPreset { min_zenith_deg: 30.0, max_zenith_deg: 130.0, divergence_mrad: div, exit_diameter: 0.007, range_noise: 0.003 };
    match name.to_ascii_lowercase().replace(['-', ' ', '_'], "").as_str() {
        "vz400" | "vz400i" => Ok(p(0.35)),
        "vz2000i" => Ok(p(0.27)),
        _ => Err(Error::invalid(format!("unknown scanner {name:?}; expected one of {SCANNERS:?}"))),
    }
}

/// Sub-beam offsets in the unit disc: the centre for one sample, else a
/// sunflower pattern of equal-area parts.
pub fn footprint_offsets(n: usize) -> Vec<[f64; 2]> {
    if n <= 1 {
        return vec![[0.0, 0.0]];
    }
    let golden = PI * (3.0 - 5f64.sqrt());
    (0..n).map(|k| {
        let r = ((k as f64 + 0.5) / n as f64).sqrt();
        let a = k as f64 * golden;
        [r * a.cos(), r * a.sin()]
    }).collect()
}

/// Points indexed in a regular 3-D grid for ray casting; each point is
/// listed in every cell its sphere of radius `r` reaches.
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
            return Ok(Grid { lo: [0.0; 3], cell: 1.0, n: [0, 0, 0], start: vec![0], idx: vec![], r });
        }
        let vol = (0..3).map(|k| (hi[k] - lo[k]).max(r)).product::<f64>();
        let cell = (2.0 * r).max((vol / (1u64 << 24) as f64).cbrt());
        let n: [usize; 3] = std::array::from_fn(|k| (((hi[k] - lo[k]) / cell).floor() as usize + 1).max(1));
        let cells = n[0] * n[1] * n[2];
        limits::check_cells(cells as u128 + 8 * targets.len() as u128, 4, "the ray-casting index of the scene", "a smaller scene or a larger target_radius")?;
        let span = |v: f64, k: usize| (((v - r - lo[k]) / cell).floor().max(0.0) as usize, (((v + r - lo[k]) / cell).floor() as usize).min(n[k] - 1));
        let mut count = vec![0u32; cells + 1];
        let each = |i: usize, f: &mut dyn FnMut(usize)| {
            let p = points[i];
            let (x0, x1) = span(p[0], 0);
            let (y0, y1) = span(p[1], 1);
            let (z0, z1) = span(p[2], 2);
            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        f((z * n[1] + y) * n[0] + x);
                    }
                }
            }
        };
        for &i in targets {
            each(i, &mut |c| count[c + 1] += 1);
        }
        for k in 1..count.len() {
            count[k] += count[k - 1];
        }
        let mut fill = count.clone();
        let mut idx = vec![0u32; count[cells] as usize];
        for &i in targets {
            each(i, &mut |c| {
                idx[fill[c] as usize] = i as u32;
                fill[c] += 1;
            });
        }
        Ok(Grid { lo, cell, n, start: count, idx, r })
    }

    /// First patch the ray `o + t d` meets at `t` in `(t_min, t_max)`:
    /// `(range, index)`. A point with a unit normal is a disc of radius `r`
    /// facing along it; with a zero normal, a sphere (range to the point's
    /// closest approach).
    fn cast(&self, points: &[Point], normals: &[Point], o: &Point, d: &Point, t_min: f64, t_max: f64) -> Option<(f64, usize)> {
        if self.n[0] == 0 {
            return None;
        }
        // Clip to the grid box.
        let (mut ta, mut tb) = (0.0f64, t_max);
        for k in 0..3 {
            let hi = self.lo[k] + self.n[k] as f64 * self.cell;
            if d[k] == 0.0 {
                if o[k] < self.lo[k] || o[k] > hi {
                    return None;
                }
            } else {
                let (t0, t1) = ((self.lo[k] - o[k]) / d[k], (hi - o[k]) / d[k]);
                ta = ta.max(t0.min(t1));
                tb = tb.min(t0.max(t1));
            }
        }
        if ta >= tb {
            return None;
        }
        let p0: Point = std::array::from_fn(|k| o[k] + ta * d[k]);
        let mut c: [i64; 3] = std::array::from_fn(|k| (((p0[k] - self.lo[k]) / self.cell).floor() as i64).clamp(0, self.n[k] as i64 - 1));
        let step: [i64; 3] = std::array::from_fn(|k| if d[k] > 0.0 { 1 } else { -1 });
        let mut t_next: [f64; 3] = std::array::from_fn(|k| {
            if d[k] == 0.0 {
                f64::INFINITY
            } else {
                let edge = self.lo[k] + (c[k] + i64::from(step[k] > 0)) as f64 * self.cell;
                (edge - o[k]) / d[k]
            }
        });
        let dt: [f64; 3] = std::array::from_fn(|k| if d[k] == 0.0 { f64::INFINITY } else { self.cell / d[k].abs() });
        let r2 = self.r * self.r;
        let mut best: Option<(f64, f64, usize)> = None;
        loop {
            let t_exit = t_next[0].min(t_next[1]).min(t_next[2]).min(tb);
            let cid = ((c[2] as usize * self.n[1]) + c[1] as usize) * self.n[0] + c[0] as usize;
            for &j in &self.idx[self.start[cid] as usize..self.start[cid + 1] as usize] {
                let i = j as usize;
                let q = points[i];
                let v = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
                let n = normals[i];
                let (te, tc) = if n == [0.0; 3] {
                    let tc = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
                    let perp2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - tc * tc;
                    if perp2 >= r2 {
                        continue;
                    }
                    (tc - (r2 - perp2).sqrt(), tc)
                } else {
                    let den = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
                    if den.abs() < 1e-9 {
                        continue;
                    }
                    let t = (v[0] * n[0] + v[1] * n[1] + v[2] * n[2]) / den;
                    let w = [o[0] + t * d[0] - q[0], o[1] + t * d[1] - q[1], o[2] + t * d[2] - q[2]];
                    if w[0] * w[0] + w[1] * w[1] + w[2] * w[2] >= r2 {
                        continue;
                    }
                    (t, t)
                };
                if te > t_min && te < t_max && best.is_none_or(|(bt, _, bi)| te < bt || (te == bt && i < bi)) {
                    best = Some((te, tc, i));
                }
            }
            if best.is_some_and(|(bt, _, _)| bt <= t_exit) || t_exit >= tb {
                break;
            }
            let k = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] { 0 } else if t_next[1] <= t_next[2] { 1 } else { 2 };
            c[k] += step[k];
            if c[k] < 0 || c[k] >= self.n[k] as i64 {
                break;
            }
            t_next[k] += dt[k];
        }
        best.map(|(_, tc, i)| (tc, i))
    }
}

fn check(p: &BeamScan) -> Result<()> {
    if !p.origin.iter().all(|v| v.is_finite()) {
        return Err(Error::invalid("origin must be finite"));
    }
    if !(p.resolution_deg.is_finite() && p.resolution_deg > 0.0) {
        return Err(Error::invalid(format!("resolution_deg must be positive, got {}", p.resolution_deg)));
    }
    if !(0.0..180.0).contains(&p.min_zenith_deg) || !(p.max_zenith_deg > p.min_zenith_deg && p.max_zenith_deg <= 180.0) {
        return Err(Error::invalid(format!("zenith range must satisfy 0 <= min < max <= 180, got {} to {}", p.min_zenith_deg, p.max_zenith_deg)));
    }
    let non_negative = [("echo_separation", p.echo_separation), ("range_noise", p.range_noise), ("range_noise_slope", p.range_noise_slope), ("beam_divergence", p.divergence_mrad), ("exit_diameter", p.exit_diameter), ("detection_threshold", p.detection_threshold)];
    for (name, v) in non_negative {
        if !(v.is_finite() && v >= 0.0) {
            return Err(Error::invalid(format!("{name} must be zero or more, got {v}")));
        }
    }
    if p.max_range.is_nan() || p.max_range <= 0.0 {
        return Err(Error::invalid(format!("max_range must be positive, got {}", p.max_range)));
    }
    if !(1..=1000).contains(&p.footprint_samples) {
        return Err(Error::invalid(format!("footprint_samples must be 1 to 1000, got {}", p.footprint_samples)));
    }
    if p.max_echoes == 0 {
        return Err(Error::invalid("max_echoes must be at least 1"));
    }
    if let Some(r) = p.target_radius {
        if !(r.is_finite() && r > 0.0) {
            return Err(Error::invalid(format!("target_radius must be positive, got {r}")));
        }
    }
    Ok(())
}

fn point_reflectance(cloud: &PointCloud, r: &Reflectance) -> Result<Vec<f64>> {
    let n = cloud.len();
    let v: Vec<f64> = match r {
        Reflectance::Constant(c) => vec![*c; n],
        Reflectance::Attribute(name) => {
            let a = cloud.attr(name).ok_or_else(|| Error::invalid(format!("the cloud has no attribute {name:?} for the reflectance")))?;
            (0..n).map(|i| a.get_f64(i)).collect()
        }
        Reflectance::ByClass(map, default) => match cloud.attr("classification") {
            Some(c) => (0..n).map(|i| {
                let k = c.get_f64(i);
                map.iter().find(|m| m.0 as f64 == k).map_or(*default, |m| m.1)
            }).collect(),
            None => vec![*default; n],
        },
    };
    if v.iter().any(|x| !(x.is_finite() && *x >= 0.0)) {
        return Err(Error::invalid("reflectance must be zero or more and finite"));
    }
    Ok(v)
}

/// Unit normals of the target points (zero for a sphere): from the
/// `normal_*` attributes, else from a local PCA where it is planar.
fn patch_normals(cloud: &PointCloud, targets: &[usize], oriented: bool) -> Vec<Point> {
    let mut out = vec![[0.0; 3]; cloud.len()];
    if !oriented {
        return out;
    }
    if let (Some(x), Some(y), Some(z)) = (cloud.attr("normal_x"), cloud.attr("normal_y"), cloud.attr("normal_z")) {
        for &i in targets {
            let n = [x.get_f64(i), y.get_f64(i), z.get_f64(i)];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if l.is_finite() && l > 0.5 {
                out[i] = [n[0] / l, n[1] / l, n[2] / l];
            }
        }
        return out;
    }
    let pts: Vec<Point> = targets.iter().map(|&i| cloud.xyz[i]).collect();
    if pts.len() < 10 {
        return out;
    }
    let (n, planarity) = crate::coreg_geometry::estimate_normals(&pts, 10, None);
    for (k, &i) in targets.iter().enumerate() {
        if planarity[k] >= 0.5 {
            out[i] = n[k];
        }
    }
    out
}

/// One echo before noise: range, energy, footprint fraction, spread, point.
struct Echo {
    range: f64,
    energy: f64,
    fraction: f64,
    spread: f64,
    point: usize,
}

/// Scan `cloud` with a finite beam (see the module documentation).
pub fn scan_beam(cloud: &PointCloud, p: &BeamScan) -> Result<Shots> {
    check(p)?;
    let rho = point_reflectance(cloud, &p.reflectance)?;
    let targets: Vec<usize> = (0..cloud.len()).filter(|&i| cloud.xyz[i].iter().all(|v| v.is_finite()) && rho[i] > 0.0).collect();
    let r = match p.target_radius {
        Some(r) => r,
        None => {
            let pts: Vec<Point> = targets.iter().map(|&i| cloud.xyz[i]).collect();
            let s = crate::leaves::median_spacing(&pts);
            if s > 0.0 { 1.75 * s } else { 0.01 }
        }
    };
    let normals = patch_normals(cloud, &targets, p.oriented);
    let grid = Grid::new(&cloud.xyz, &targets, r)?;
    let res = p.resolution_deg;
    let n_zen = ((p.max_zenith_deg - p.min_zenith_deg) / res).round().max(1.0) as usize;
    let n_az = (360.0 / res).round().max(1.0) as usize;
    let n_pulses = n_zen * n_az;
    limits::check_cells(n_pulses as u128, 64 + 8 * p.max_echoes as u64, &format!("a scan of {n_pulses} pulses"), "a coarser resolution_deg")?;
    let offsets = footprint_offsets(p.footprint_samples);
    let share = 1.0 / offsets.len() as f64;
    let half_div = p.divergence_mrad * 1e-3 / 2.0;
    let half_exit = p.exit_diameter / 2.0;
    let o = p.origin;
    let mut rng = Generator::new(p.seed);
    let mut direction = Vec::with_capacity(n_pulses);
    let mut per_pulse: Vec<Vec<(Echo, f64)>> = Vec::with_capacity(n_pulses);
    const BLOCK: usize = 1 << 15;
    let mut j0 = 0;
    while j0 < n_pulses {
        let j1 = (j0 + BLOCK).min(n_pulses);
        let noise = if p.range_noise > 0.0 || p.range_noise_slope > 0.0 { rng.normal_n(0.0, 1.0, (j1 - j0) * p.max_echoes) } else { vec![0.0; (j1 - j0) * p.max_echoes] };
        let block: Vec<(Point, Vec<(Echo, f64)>)> = (j0..j1).into_par_iter().map(|j| {
            let (iz, ia) = (j / n_az, j % n_az);
            let z = (p.min_zenith_deg + (iz as f64 + 0.5) * res).to_radians();
            let a = ((ia as f64 + 0.5) * res).to_radians();
            let d = [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()];
            let h = if d[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
            let e1 = {
                let c = [d[1] * h[2] - d[2] * h[1], d[2] * h[0] - d[0] * h[2], d[0] * h[1] - d[1] * h[0]];
                let nn = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                [c[0] / nn, c[1] / nn, c[2] / nn]
            };
            let e2 = [d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0]];
            let mut hits: Vec<(f64, usize)> = offsets.iter().filter_map(|off| {
                let lat: Point = std::array::from_fn(|k| off[0] * e1[k] + off[1] * e2[k]);
                let os: Point = std::array::from_fn(|k| o[k] + half_exit * lat[k]);
                let ds: Point = std::array::from_fn(|k| d[k] + half_div * lat[k]);
                let nn = (ds[0] * ds[0] + ds[1] * ds[1] + ds[2] * ds[2]).sqrt();
                let ds = [ds[0] / nn, ds[1] / nn, ds[2] / nn];
                grid.cast(&cloud.xyz, &normals, &os, &ds, 0.1, p.max_range)
            }).collect();
            hits.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut echoes: Vec<Echo> = Vec::new();
            let mut g0 = 0;
            while g0 < hits.len() {
                let mut g1 = g0 + 1;
                while g1 < hits.len() && hits[g1].0 - hits[g0].0 < p.echo_separation {
                    g1 += 1;
                }
                let group = &hits[g0..g1];
                let w: Vec<f64> = group.iter().map(|h| share * rho[h.1]).collect();
                let energy: f64 = w.iter().sum();
                let mut best = 0;
                for k in 1..group.len() {
                    if w[k] > w[best] {
                        best = k;
                    }
                }
                let range = if p.mixed_pixels && energy > 0.0 { group.iter().zip(&w).map(|(h, w)| h.0 * w).sum::<f64>() / energy } else { group[best].0 };
                if energy >= p.detection_threshold && energy > 0.0 {
                    echoes.push(Echo { range, energy, fraction: share * group.len() as f64, spread: group[group.len() - 1].0 - group[0].0, point: group[best].1 });
                }
                g0 = g1;
            }
            echoes.truncate(p.max_echoes);
            let nz = &noise[(j - j0) * p.max_echoes..];
            let mut out: Vec<(Echo, f64)> = echoes.into_iter().enumerate().map(|(k, e)| {
                let sigma = p.range_noise + p.range_noise_slope * e.range;
                let noisy = e.range + sigma * nz[k];
                (e, noisy)
            }).collect();
            out.sort_by(|a, b| a.1.total_cmp(&b.1));
            (d, out)
        }).collect();
        for (d, e) in block {
            direction.push(d);
            per_pulse.push(e);
        }
        j0 = j1;
    }
    let count: Vec<u32> = per_pulse.iter().map(|e| e.len() as u32).collect();
    let echo_start = packed_starts(&count);
    let flat: Vec<&(Echo, f64)> = per_pulse.iter().flatten().collect();
    let idx: Vec<usize> = flat.iter().map(|e| e.0.point).collect();
    let mut echo_attrs: std::collections::BTreeMap<String, Attr> = cloud.attrs.iter().map(|(k, v)| (k.clone(), v.take(&idx))).collect();
    echo_attrs.insert("reflectance".into(), Attr::F32(flat.iter().map(|e| (10.0 * e.0.energy.log10()) as f32).collect()));
    echo_attrs.insert("footprint_fraction".into(), Attr::F32(flat.iter().map(|e| e.0.fraction as f32).collect()));
    echo_attrs.insert("range_spread".into(), Attr::F32(flat.iter().map(|e| e.0.spread as f32).collect()));
    Ok(Shots { origin: vec![o; n_pulses], direction, echo_start, echo_count: count, echo_range: flat.iter().map(|e| e.1).collect(), echo_attrs })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense square of points at `y`, facing the scanner at the origin.
    fn wall(y: f64, x0: f64, x1: f64, z0: f64, z1: f64, step: f64) -> Vec<Point> {
        let mut v = Vec::new();
        let mut x = x0;
        while x <= x1 {
            let mut z = z0;
            while z <= z1 {
                v.push([x, y, z]);
                z += step;
            }
            x += step;
        }
        v
    }

    fn params() -> BeamScan {
        let mut p = BeamScan::new([0.0, 0.0, 0.0]);
        p.oriented = false;
        p.resolution_deg = 0.2;
        p.min_zenith_deg = 80.0;
        p.max_zenith_deg = 100.0;
        p.target_radius = Some(0.004);
        p
    }

    #[test]
    fn a_plain_wall_is_at_its_range() {
        let cloud = PointCloud::new(wall(10.0, -1.0, 1.0, -1.0, 1.0, 0.005));
        let s = scan_beam(&cloud, &params()).unwrap();
        assert!(s.echo_range.len() > 50);
        for c in 0..s.n_shots() {
            for e in 0..s.echo_count[c] as usize {
                let r = s.echo_range[s.echo_start[c] + e];
                // Range times the y component of the direction is the wall's distance.
                assert!((r * s.direction[c][1] - 10.0).abs() < 0.01, "{r}");
            }
        }
    }

    #[test]
    fn range_noise_has_its_spread() {
        let cloud = PointCloud::new(wall(10.0, -2.0, 2.0, -1.5, 1.5, 0.01));
        let mut p = params();
        p.target_radius = Some(0.008);
        p.range_noise = 0.01;
        p.seed = 3;
        let s = scan_beam(&cloud, &p).unwrap();
        let err: Vec<f64> = (0..s.n_shots()).filter(|&c| s.echo_count[c] > 0).map(|c| s.echo_range[s.echo_start[c]] * s.direction[c][1] - 10.0).collect();
        let n = err.len() as f64;
        let m = err.iter().sum::<f64>() / n;
        let sd = (err.iter().map(|e| (e - m).powi(2)).sum::<f64>() / n).sqrt();
        assert!(n > 1000.0 && (sd - 0.01).abs() < 0.001 && m.abs() < 0.001, "{n} {m} {sd}");
    }

    #[test]
    fn a_wide_beam_mixes_an_edge() {
        // Foreground half-plane (x < 0) at 10 m, background at 10.3 m.
        let mut pts = wall(10.0, -3.0, 0.0, -1.0, 1.0, 0.005);
        pts.extend(wall(10.3, -3.0, 3.0, -1.0, 1.0, 0.005));
        let cloud = PointCloud::new(pts);
        let mut p = params();
        p.divergence_mrad = 3.0;
        p.footprint_samples = 19;
        let mixed = |s: &Shots| (0..s.n_shots()).filter(|&c| s.echo_count[c] > 0).filter(|&c| {
            let d = s.echo_range[s.echo_start[c]] * s.direction[c][1];
            d > 10.02 && d < 10.28
        }).count();
        let s = scan_beam(&cloud, &p).unwrap();
        assert!(mixed(&s) > 0);
        p.mixed_pixels = false;
        assert_eq!(mixed(&scan_beam(&cloud, &p).unwrap()), 0);
        // With a fine range resolution the two surfaces give two echoes instead.
        p.mixed_pixels = true;
        p.echo_separation = 0.1;
        let s = scan_beam(&cloud, &p).unwrap();
        assert_eq!(mixed(&s), 0);
        assert!(s.echo_count.contains(&2));
    }

    #[test]
    fn misses_are_kept_and_presets_parse() {
        let cloud = PointCloud::new(vec![[0.0, 5.0, 0.0]]);
        let mut p = params();
        p.target_radius = Some(0.01);
        let s = scan_beam(&cloud, &p).unwrap();
        assert_eq!(s.n_shots(), 100 * 1800);
        assert!(s.echo_count.iter().filter(|&&c| c > 0).count() <= 4);
        assert_eq!(scanner_preset("VZ-400i").unwrap().divergence_mrad, 0.35);
        assert!(scanner_preset("p20").is_err());
        p.range_noise = -1.0;
        assert!(scan_beam(&cloud, &p).is_err());
    }

    #[test]
    fn discs_keep_a_cylinder_its_width() {
        // A vertical cylinder of radius 0.2 m at 5 m, sampled with its normals.
        let mut pts = Vec::new();
        let mut nrm = Vec::new();
        for i in 0..400 {
            for j in 0..60 {
                let a = 2.0 * PI * i as f64 / 400.0;
                pts.push([0.2 * a.cos(), 5.0 + 0.2 * a.sin(), -0.3 + 0.01 * j as f64]);
                nrm.push([a.cos(), a.sin(), 0.0]);
            }
        }
        let mut cloud = PointCloud::new(pts);
        crate::synthetic_tree::insert_normals(&mut cloud, &nrm);
        let mut p = params();
        p.oriented = true;
        p.resolution_deg = 0.05;
        p.min_zenith_deg = 89.9;
        p.max_zenith_deg = 90.1;
        p.target_radius = Some(0.02);
        let width = |s: &Shots| {
            let xs: Vec<f64> = (0..s.n_shots()).filter(|&c| s.echo_count[c] > 0).map(|c| s.echo_range[s.echo_start[c]] * s.direction[c][0]).collect();
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - xs.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        let w = width(&scan_beam(&cloud, &p).unwrap());
        assert!((w - 0.4).abs() < 0.01, "{w}");
        p.oriented = false;
        let w = width(&scan_beam(&cloud, &p).unwrap());
        assert!(w > 0.42, "{w}");
    }

    #[test]
    fn offsets_cover_the_disc() {
        let o = footprint_offsets(100);
        assert!(o.iter().all(|v| v[0].hypot(v[1]) <= 1.0));
        let m: f64 = o.iter().map(|v| v[0]).sum::<f64>() / 100.0;
        assert!(m.abs() < 0.02);
    }
}
