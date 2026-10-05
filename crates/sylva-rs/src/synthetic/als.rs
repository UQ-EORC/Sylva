// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A synthetic airborne laser scanner flown over a synthetic scene.
//!
//! The aircraft flies parallel, alternating flight lines at constant speed
//! and height; a mirror sweeps the beam across the track while pulses leave
//! at a fixed rate. Each pulse is a cone of the beam divergence, sampled by
//! a few equal-energy sub-beams; a sub-beam stops at the first scene point
//! it passes within `target_radius` of, or at the analytic terrain of
//! [`crate::synthetic::terrain_height`]. The energy stopped along the pulse
//! is grouped into returns the way a discrete-return receiver separates
//! echoes: hits closer than `min_separation` in range merge into one
//! return at their energy-weighted mean range, and returns below the
//! detection threshold are lost.
//!
//! # Geometry and timing
//!
//! Map frame: x east, y north, z up (the scene's frame). Body frame of the
//! aircraft: x forward, y to the right wing, z down. The attitude is
//! aerospace roll `φ` (right wing down positive), pitch `θ` (nose up
//! positive) and heading `ψ` (clockwise from north, degrees in the
//! trajectory), and a body vector `b` points in the map along
//!
//! ```text
//! d = M · Rz(ψ) · Ry(θ) · Rx(φ) · b,     M (north-east-down to east-north-up) = [[0,1,0],[1,0,0],[0,0,-1]]
//! ```
//!
//! The beam leaves the scanner, which sits at the trajectory position (no
//! lever arm or boresight offset), along `b = (0, sin α, cos α)` for mirror
//! angle `α` (positive to the right; nadir is `α = 0`). Every return lies on
//! that central axis: `position = origin(t) + range · d`, where `t` is the
//! pulse's `gps_time`; all returns of a pulse share it. The LAS
//! `scan_angle` of a return is `α - φ` in degrees: the angle of the beam
//! from the vertical, including the roll of the aircraft, as the LAS
//! specification defines it (rolling right wing down turns the beam left).
//! The mirror angle is therefore `scan_angle + roll`.
//!
//! Line `k` (from 0) starts at `t0_k = start_time + k · (T + turn_time)`,
//! where `T` is the time to fly the line, and pulses are fired at
//! `t0_k + j / pulse_rate`. Position within a line is a straight segment at
//! constant speed, so interpolating the trajectory linearly in time gives
//! the exact origin; roll, pitch and heading are small sinusoids about
//! level flight along the line (periods 7.3, 11.1 and 13.7 s, amplitudes
//! given, phases drawn from the seed), exact at the trajectory samples.
//! The mirror angle `α` at time `t` since the line start, with
//! `u = t · scan_rate` sweeps done: a rotating polygon sweeps left to right
//! and jumps back, `α = A (2 frac(u) - 1)`; an oscillating mirror sweeps
//! left to right and back at constant angular speed (a zigzag on the
//! ground), one sweep per `1 / scan_rate` s.
//!
//! Random numbers come from [`Generator`] (NumPy's `default_rng`): three
//! attitude phases, then the range noise of each pulse, `max_returns`
//! normal draws per pulse in firing order, so that a seed gives the same
//! flight whatever the number of threads.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::util::nprandom::Generator;
use crate::pointcloud::Attr;
use crate::util::{limits, progress};
use crate::{Point, PointCloud};

/// How the mirror moves the beam across the track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanPattern {
    /// Back and forth at constant angular speed: a zigzag on the ground.
    Oscillating,
    /// Always left to right (a rotating polygon): parallel lines on the ground.
    Rotating,
}

impl ScanPattern {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "oscillating" => Ok(ScanPattern::Oscillating),
            "rotating" => Ok(ScanPattern::Rotating),
            _ => Err(Error::invalid(format!("unknown scan pattern {name:?}; expected 'oscillating' or 'rotating'"))),
        }
    }
}

/// Settings of [`fly`]. Lengths in metres, times in seconds, angles in
/// degrees unless the name says otherwise.
#[derive(Debug, Clone)]
pub struct FlightParams {
    /// Flying height above z = 0 (the scene's datum).
    pub altitude: f64,
    /// Ground speed (m/s).
    pub speed: f64,
    /// Distance between neighbouring flight lines.
    pub line_spacing: f64,
    /// Direction of the first line, clockwise from north; lines alternate.
    pub heading: f64,
    pub pattern: ScanPattern,
    /// Largest mirror angle either side of nadir.
    pub scan_angle: f64,
    /// Sweeps across the track per second.
    pub scan_rate: f64,
    /// Pulses per second.
    pub pulse_rate: f64,
    /// Full beam divergence (mrad); the footprint radius at range `R` is `R · divergence / 2`.
    pub divergence_mrad: f64,
    /// Sub-beams sampling the footprint: 1, 7 or 19.
    pub footprint_samples: usize,
    /// Returns recorded per pulse at most (1-15).
    pub max_returns: usize,
    /// Hits closer than this in range make one return.
    pub min_separation: f64,
    /// Share of the pulse energy a return needs to be recorded.
    pub detection_threshold: f64,
    /// Standard deviation of the range noise.
    pub range_noise: f64,
    /// Amplitudes of roll, pitch and heading about level flight.
    pub attitude: [f64; 3],
    /// Radius of the sphere each scene point stands for.
    pub target_radius: f64,
    /// Slope of the terrain ([`crate::synthetic::terrain_height`]).
    pub terrain_slope: f64,
    /// Area to cover, `[xmin, ymin, xmax, ymax]`; the scene's extent if None.
    pub bounds: Option<[f64; 4]>,
    /// Keep only returns inside `bounds`.
    pub clip: bool,
    /// Distance flown before and after the area on each line.
    pub margin: f64,
    /// Time between lines (the laser is off while turning).
    pub turn_time: f64,
    /// GPS time of the first pulse.
    pub start_time: f64,
    /// Trajectory samples per second.
    pub trajectory_rate: f64,
    pub seed: u64,
}

impl Default for FlightParams {
    fn default() -> Self {
        FlightParams {
            altitude: 80.0,
            speed: 10.0,
            line_spacing: 40.0,
            heading: 0.0,
            pattern: ScanPattern::Oscillating,
            scan_angle: 30.0,
            scan_rate: 80.0,
            pulse_rate: 50_000.0,
            divergence_mrad: 1.0,
            footprint_samples: 7,
            max_returns: 5,
            min_separation: 1.0,
            detection_threshold: 0.1,
            range_noise: 0.02,
            attitude: [0.5, 0.3, 0.2],
            target_radius: 0.03,
            terrain_slope: 0.05,
            bounds: None,
            clip: true,
            margin: 10.0,
            turn_time: 10.0,
            start_time: 0.0,
            trajectory_rate: 100.0,
            seed: 0,
        }
    }
}

/// The platform's path, sampled while the laser is on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trajectory {
    pub time: Vec<f64>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    /// Degrees, right wing down positive.
    pub roll: Vec<f64>,
    /// Degrees, nose up positive.
    pub pitch: Vec<f64>,
    /// Degrees clockwise from north, in `[0, 360)`.
    pub heading: Vec<f64>,
    /// Flight line (the `point_source_id` of its returns).
    pub line: Vec<u16>,
}

impl Trajectory {
    /// Sensor position at each time, by linear interpolation between the
    /// samples of the line flying at that time (exact, since lines are
    /// straight and flown at constant speed). NaN for times when no line
    /// was being flown.
    pub fn positions(&self, times: &[f64]) -> Vec<Point> {
        times
            .iter()
            .map(|&t| {
                let k = self.time.partition_point(|&s| s <= t);
                let nan = [f64::NAN; 3];
                if k == 0 {
                    return nan;
                }
                let i = k - 1;
                if self.time[i] == t {
                    return [self.x[i], self.y[i], self.z[i]];
                }
                if k >= self.time.len() || self.line[k] != self.line[i] {
                    return nan;
                }
                let f = (t - self.time[i]) / (self.time[k] - self.time[i]);
                [self.x[i] + f * (self.x[k] - self.x[i]), self.y[i] + f * (self.y[k] - self.y[i]), self.z[i] + f * (self.z[k] - self.z[i])]
            })
            .collect()
    }
}

/// A simulated flight: the returns and the trajectory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Flight {
    /// Returns with `gps_time`, `return_number`, `number_of_returns`,
    /// `scan_angle`, `intensity`, `point_source_id`, `classification` (what
    /// was hit: 2 ground, else the scene point's class) and, if the scene
    /// has it, `tree_id`.
    pub points: PointCloud,
    pub trajectory: Trajectory,
    /// Pulses fired.
    pub n_pulses: u64,
}

/// One flight line.
#[derive(Debug, Clone)]
struct Line {
    id: u16,
    start: [f64; 2],
    dir: [f64; 2],
    heading: f64,
    t0: f64,
    t1: f64,
}

/// The lines, timing and attitude of a flight.
struct Plan {
    lines: Vec<Line>,
    phase: [f64; 3],
}

const PERIODS: [f64; 3] = [7.3, 11.1, 13.7];

impl Plan {
    fn new(p: &FlightParams, b: [f64; 4], phase: [f64; 3]) -> Plan {
        let h = p.heading.to_radians();
        let along = [h.sin(), h.cos()];
        let across = [h.cos(), -h.sin()];
        let corners = [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]];
        let proj = |v: &[f64; 2], q: &[f64; 2]| v[0] * q[0] + v[1] * q[1];
        let (mut amin, mut amax, mut cmin, mut cmax) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for q in &corners {
            amin = amin.min(proj(&along, q));
            amax = amax.max(proj(&along, q));
            cmin = cmin.min(proj(&across, q));
            cmax = cmax.max(proj(&across, q));
        }
        let n = ((cmax - cmin) / p.line_spacing).floor() as usize + 1;
        let length = amax - amin + 2.0 * p.margin;
        let duration = length / p.speed;
        let cmid = 0.5 * (cmin + cmax);
        let lines = (0..n)
            .map(|k| {
                let c = cmid + (k as f64 - (n as f64 - 1.0) / 2.0) * p.line_spacing;
                let forward = k % 2 == 0;
                let a0 = if forward { amin - p.margin } else { amax + p.margin };
                let s = if forward { 1.0 } else { -1.0 };
                let t0 = p.start_time + k as f64 * (duration + p.turn_time);
                Line {
                    id: k as u16 + 1,
                    start: [a0 * along[0] + c * across[0], a0 * along[1] + c * across[1]],
                    dir: [s * along[0], s * along[1]],
                    heading: (p.heading + if forward { 0.0 } else { 180.0 }).rem_euclid(360.0),
                    t0,
                    t1: t0 + duration,
                }
            })
            .collect();
        Plan { lines, phase }
    }

    /// Position and attitude (roll, pitch, heading in degrees) on `line` at `t`.
    fn pose(&self, p: &FlightParams, line: &Line, t: f64) -> (Point, [f64; 3]) {
        let s = (t - line.t0) * p.speed;
        let pos = [line.start[0] + s * line.dir[0], line.start[1] + s * line.dir[1], p.altitude];
        let wave = |k: usize| (2.0 * PI * t / PERIODS[k] + self.phase[k]).sin();
        let att = [p.attitude[0] * wave(0), p.attitude[1] * wave(1), (line.heading + p.attitude[2] * wave(2)).rem_euclid(360.0)];
        (pos, att)
    }
}

/// Mirror angle (degrees) `dt` seconds after a line starts.
fn mirror_angle(p: &FlightParams, dt: f64) -> f64 {
    let u = dt * p.scan_rate;
    match p.pattern {
        ScanPattern::Rotating => p.scan_angle * (2.0 * (u - u.floor()) - 1.0),
        ScanPattern::Oscillating => {
            let v = u / 2.0 - (u / 2.0).floor();
            p.scan_angle * if v < 0.5 { 4.0 * v - 1.0 } else { 3.0 - 4.0 * v }
        }
    }
}

/// Body-to-map rotation for roll, pitch, heading in degrees (see the module notes).
pub fn body_to_map(roll: f64, pitch: f64, heading: f64) -> [[f64; 3]; 3] {
    let (sr, cr) = roll.to_radians().sin_cos();
    let (sp, cp) = pitch.to_radians().sin_cos();
    let (sh, ch) = heading.to_radians().sin_cos();
    // Rz(ψ) Ry(θ) Rx(φ), body to north-east-down.
    let ned = [
        [ch * cp, ch * sp * sr - sh * cr, ch * sp * cr + sh * sr],
        [sh * cp, sh * sp * sr + ch * cr, sh * sp * cr - ch * sr],
        [-sp, cp * sr, cp * cr],
    ];
    // North-east-down to east-north-up.
    [ned[1], ned[0], [-ned[2][0], -ned[2][1], -ned[2][2]]]
}

fn apply(m: &[[f64; 3]; 3], v: &Point) -> Point {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// Beam direction in the body frame for mirror angle `alpha` (degrees).
pub fn beam_body(alpha: f64) -> Point {
    let (s, c) = alpha.to_radians().sin_cos();
    [0.0, s, c]
}

/// Sub-beam offsets as fractions of the footprint radius: equal-area parts
/// of the footprint disc, each sampled at (about) its centroid.
fn footprint_offsets(n: usize) -> Vec<[f64; 2]> {
    let ring = |k: usize, r: f64, rot: f64| (0..k).map(move |i| {
        let a = rot + 2.0 * PI * i as f64 / k as f64;
        [r * a.cos(), r * a.sin()]
    });
    let mut v = vec![[0.0, 0.0]];
    match n {
        7 => v.extend(ring(6, 0.7, 0.0)),
        19 => {
            v.extend(ring(6, 0.43, 0.0));
            v.extend(ring(12, 0.81, PI / 12.0));
        }
        _ => {}
    }
    v
}

/// Scene points indexed in vertical columns for ray casting: each point is
/// listed in every column its disc of radius `r` reaches, sorted by z from
/// the top down.
struct Columns {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    start: Vec<u32>,
    z: Vec<f64>,
    idx: Vec<u32>,
    zmin: f64,
    zmax: f64,
    r: f64,
}

impl Columns {
    fn new(points: &[Point], targets: &[usize], r: f64) -> Result<Columns> {
        let cell = (4.0 * r).max(0.1);
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &i in targets {
            for k in 0..3 {
                lo[k] = lo[k].min(points[i][k]);
                hi[k] = hi[k].max(points[i][k]);
            }
        }
        if targets.is_empty() {
            return Ok(Columns { x0: 0.0, y0: 0.0, cell, nx: 0, ny: 0, start: vec![0], z: vec![], idx: vec![], zmin: 0.0, zmax: 0.0, r });
        }
        let (x0, y0) = (lo[0] - r, lo[1] - r);
        let nx = ((hi[0] + r - x0) / cell).floor() as usize + 1;
        let ny = ((hi[1] + r - y0) / cell).floor() as usize + 1;
        limits::check_cells(nx as u128 * ny as u128, 4, &format!("a {nx} x {ny} column index of the scene"), "a smaller scene or a larger target_radius")?;
        let span = |v: f64, o: f64, n: usize| (((v - r - o) / cell).floor().max(0.0) as usize, (((v + r - o) / cell).floor() as usize).min(n - 1));
        let mut count = vec![0u32; nx * ny + 1];
        for &i in targets {
            let (c0, c1) = span(points[i][0], x0, nx);
            let (r0, r1) = span(points[i][1], y0, ny);
            for row in r0..=r1 {
                for c in c0..=c1 {
                    count[row * nx + c + 1] += 1;
                }
            }
        }
        for k in 1..count.len() {
            count[k] += count[k - 1];
        }
        let total = count[nx * ny] as usize;
        let mut fill = count.clone();
        let mut idx = vec![0u32; total];
        for &i in targets {
            let (c0, c1) = span(points[i][0], x0, nx);
            let (r0, r1) = span(points[i][1], y0, ny);
            for row in r0..=r1 {
                for c in c0..=c1 {
                    let k = row * nx + c;
                    idx[fill[k] as usize] = i as u32;
                    fill[k] += 1;
                }
            }
        }
        for k in 0..nx * ny {
            idx[count[k] as usize..count[k + 1] as usize].sort_by(|&a, &b| points[b as usize][2].total_cmp(&points[a as usize][2]).then(a.cmp(&b)));
        }
        let z = idx.iter().map(|&i| points[i as usize][2]).collect();
        Ok(Columns { x0, y0, cell, nx, ny, start: count, z, idx, zmin: lo[2], zmax: hi[2], r })
    }

    /// The first scene point hit by the ray `o + t d` (`d` a unit vector
    /// pointing down) before `t_max`: `(t, point index)`.
    fn cast(&self, points: &[Point], o: &Point, d: &Point, t_max: f64) -> Option<(f64, usize)> {
        if self.nx == 0 || d[2] >= 0.0 {
            return None;
        }
        let r = self.r;
        let t_top = ((o[2] - (self.zmax + r)) / -d[2]).max(0.0);
        let t_end = ((o[2] - (self.zmin - r)) / -d[2]).min(t_max);
        if t_top >= t_end {
            return None;
        }
        let at = |t: f64| [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        let p0 = at(t_top);
        let mut cx = ((p0[0] - self.x0) / self.cell).floor() as i64;
        let mut cy = ((p0[1] - self.y0) / self.cell).floor() as i64;
        let step = |dv: f64| if dv > 0.0 { 1i64 } else { -1 };
        let (sx, sy) = (step(d[0]), step(d[1]));
        let boundary_t = |c: i64, s: i64, origin: f64, ov: f64, dv: f64| -> f64 {
            if dv == 0.0 {
                return f64::INFINITY;
            }
            let edge = origin + (c + if s > 0 { 1 } else { 0 }) as f64 * self.cell;
            (edge - ov) / dv
        };
        let mut tx = boundary_t(cx, sx, self.x0, o[0], d[0]);
        let mut ty = boundary_t(cy, sy, self.y0, o[1], d[1]);
        let dtx = if d[0] == 0.0 { f64::INFINITY } else { self.cell / d[0].abs() };
        let dty = if d[1] == 0.0 { f64::INFINITY } else { self.cell / d[1].abs() };
        let mut ta = t_top;
        let mut best: Option<(f64, usize)> = None;
        let r2 = r * r;
        loop {
            let tb = tx.min(ty).min(t_end);
            if cx >= 0 && cy >= 0 && (cx as usize) < self.nx && (cy as usize) < self.ny {
                let k = cy as usize * self.nx + cx as usize;
                let (s, e) = (self.start[k] as usize, self.start[k + 1] as usize);
                let ztop = o[2] + ta * d[2] + r;
                let zbot = o[2] + tb * d[2] - r;
                // Sorted from the top down: skip what is above the band.
                let first = s + self.z[s..e].partition_point(|&z| z > ztop);
                for j in first..e {
                    if self.z[j] < zbot {
                        break;
                    }
                    let i = self.idx[j] as usize;
                    let q = points[i];
                    let v = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
                    let tc = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
                    let perp2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - tc * tc;
                    if perp2 < r2 {
                        let t = tc - (r2 - perp2).sqrt();
                        if t > 0.0 && t < t_max && best.is_none_or(|(bt, bi)| t < bt || (t == bt && i < bi)) {
                            best = Some((t, i));
                        }
                    }
                }
            }
            if best.is_some_and(|(bt, _)| bt <= tb) || tb >= t_end {
                break;
            }
            ta = tb;
            if tx < ty {
                cx += sx;
                tx += dtx;
            } else {
                cy += sy;
                ty += dty;
            }
        }
        best
    }
}

/// Range along `o + t d` to the terrain `slope · x + 0.2 sin(y / 3)`:
/// Newton's method inside a bracket that is halved whenever a Newton step
/// would leave it.
fn terrain_range(o: &Point, d: &Point, slope: f64) -> f64 {
    let f = |t: f64| o[2] + t * d[2] - crate::synthetic::terrain_height(o[0] + t * d[0], o[1] + t * d[1], slope);
    let df = |t: f64| d[2] - slope * d[0] - 0.2 / 3.0 * ((o[1] + t * d[1]) / 3.0).cos() * d[1];
    let (mut lo, mut hi) = (0.0, (o[2] + 1.0).max(1.0) / -d[2]);
    while f(hi) > 0.0 {
        lo = hi;
        hi *= 2.0;
    }
    let mut t = 0.5 * (lo + hi);
    for _ in 0..100 {
        let v = f(t);
        if v > 0.0 {
            lo = t;
        } else {
            hi = t;
        }
        let g = df(t);
        let mut nt = if g != 0.0 { t - v / g } else { f64::NAN };
        if !(nt > lo && nt < hi) {
            nt = 0.5 * (lo + hi);
        }
        if (nt - t).abs() < 1e-12 * t.max(1.0) {
            return nt;
        }
        t = nt;
    }
    t
}

fn check(p: &FlightParams) -> Result<()> {
    let positive = [("altitude", p.altitude), ("speed", p.speed), ("line_spacing", p.line_spacing), ("scan_rate", p.scan_rate), ("pulse_rate", p.pulse_rate), ("target_radius", p.target_radius), ("trajectory_rate", p.trajectory_rate)];
    for (name, v) in positive {
        if !(v.is_finite() && v > 0.0) {
            return Err(Error::invalid(format!("{name} must be a positive number, got {v}")));
        }
    }
    let non_negative = [("divergence", p.divergence_mrad), ("min_separation", p.min_separation), ("range_noise", p.range_noise), ("margin", p.margin), ("turn_time", p.turn_time)];
    for (name, v) in non_negative {
        if !(v.is_finite() && v >= 0.0) {
            return Err(Error::invalid(format!("{name} must be zero or more, got {v}")));
        }
    }
    if !(p.scan_angle > 0.0 && p.scan_angle <= 75.0) {
        return Err(Error::invalid(format!("scan_angle must be in (0, 75] degrees, got {}", p.scan_angle)));
    }
    if ![1, 7, 19].contains(&p.footprint_samples) {
        return Err(Error::invalid(format!("footprint_samples must be 1, 7 or 19, got {}", p.footprint_samples)));
    }
    if !(1..=15).contains(&p.max_returns) {
        return Err(Error::invalid(format!("max_returns must be 1 to 15, got {}", p.max_returns)));
    }
    if !(0.0..=1.0).contains(&p.detection_threshold) {
        return Err(Error::invalid(format!("detection_threshold must be between 0 and 1, got {}", p.detection_threshold)));
    }
    if p.attitude.iter().any(|a| !(a.is_finite() && a.abs() <= 10.0)) {
        return Err(Error::invalid(format!("attitude amplitudes must be within 10 degrees, got {:?}", p.attitude)));
    }
    if !(p.terrain_slope.is_finite() && p.terrain_slope.abs() <= 0.5) {
        return Err(Error::invalid(format!("terrain_slope must be within ±0.5, got {}", p.terrain_slope)));
    }
    if !(p.heading.is_finite() && p.start_time.is_finite()) {
        return Err(Error::invalid("heading and start_time must be finite"));
    }
    Ok(())
}

/// Reflectance of what a sub-beam hit: ground, leaf (class 4), wood (5) or other.
fn reflectance(class: u8) -> f64 {
    match class {
        2 => 0.25,
        4 => 0.45,
        5 => 0.35,
        _ => 0.3,
    }
}

/// Hits merged into one return: sum of weight times range, sum of weight,
/// and the hits `(range, point index or usize::MAX for the terrain)`.
type Echo = (f64, f64, Vec<(f64, usize)>);

/// One recorded return before it becomes a point.
struct Return {
    xyz: Point,
    t: f64,
    number: u8,
    count: u8,
    scan_angle: f32,
    intensity: u16,
    line: u16,
    class: u8,
    tree: i32,
}

/// Fly over `scene` (points with an optional `classification`, whose
/// class-2 points are left out as the analytic terrain replaces them, and
/// optional `tree_id`) and record what the scanner sees.
pub fn fly(scene: &PointCloud, p: &FlightParams) -> Result<Flight> {
    check(p)?;
    let cls: Vec<u8> = match scene.attr("classification") {
        Some(c) => (0..scene.len()).map(|i| c.get_f64(i) as u8).collect(),
        None => vec![1; scene.len()],
    };
    let tree: Option<Vec<i32>> = scene.attr("tree_id").map(|a| (0..scene.len()).map(|i| a.get_f64(i) as i32).collect());
    let targets: Vec<usize> = (0..scene.len()).filter(|&i| cls[i] != 2 && scene.xyz[i].iter().all(|v| v.is_finite())).collect();
    let bounds = match p.bounds {
        Some(b) => {
            if !(b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]) {
                return Err(Error::invalid(format!("bounds must be finite (xmin, ymin, xmax, ymax) with max > min, got {b:?}")));
            }
            b
        }
        None => {
            let (lo, hi) = scene.bounds().ok_or_else(|| Error::invalid("the scene is empty; give bounds to fly over bare terrain"))?;
            [lo[0], lo[1], hi[0], hi[1]]
        }
    };
    let cols = Columns::new(&scene.xyz, &targets, p.target_radius)?;
    if !targets.is_empty() && cols.zmax + p.target_radius >= p.altitude {
        return Err(Error::invalid(format!("the aircraft at {} m would fly through the scene, which reaches {:.1} m", p.altitude, cols.zmax)));
    }
    let terrain_top = p.terrain_slope.abs() * bounds[0].abs().max(bounds[2].abs()) + 0.2;
    if terrain_top >= p.altitude {
        return Err(Error::invalid(format!("the aircraft at {} m would fly into the terrain", p.altitude)));
    }

    let mut rng = Generator::new(p.seed);
    let phase = [rng.uniform(0.0, 2.0 * PI), rng.uniform(0.0, 2.0 * PI), rng.uniform(0.0, 2.0 * PI)];
    let plan = Plan::new(p, bounds, phase);
    let per_line: Vec<u64> = plan.lines.iter().map(|l| ((l.t1 - l.t0) * p.pulse_rate).floor() as u64).collect();
    let n_pulses: u64 = per_line.iter().sum();
    limits::check_cells(n_pulses as u128 * 2, 64, &format!("a flight of {n_pulses} pulses"), "a lower pulse_rate, a smaller area or a wider line_spacing")?;

    let offsets = footprint_offsets(p.footprint_samples);
    let weight = 1.0 / offsets.len() as f64;
    let half_div = p.divergence_mrad * 1e-3 / 2.0;
    let task = progress::start("flying", n_pulses);
    let mut out: Vec<Return> = Vec::new();
    const BLOCK: u64 = 1 << 16;
    for line in &plan.lines {
        let n = ((line.t1 - line.t0) * p.pulse_rate).floor() as u64;
        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + BLOCK).min(n);
            let noise = if p.range_noise > 0.0 { rng.normal_n(0.0, p.range_noise, ((j1 - j0) as usize) * p.max_returns) } else { vec![0.0; ((j1 - j0) as usize) * p.max_returns] };
            let block: Vec<Vec<Return>> = (j0..j1)
                .into_par_iter()
                .map(|j| {
                    let t = line.t0 + j as f64 / p.pulse_rate;
                    let (o, att) = plan.pose(p, line, t);
                    let alpha = mirror_angle(p, t - line.t0);
                    let m = body_to_map(att[0], att[1], att[2]);
                    let b = beam_body(alpha);
                    let d = apply(&m, &b);
                    // Sub-beams: the central beam tilted along track and within the scan plane.
                    let e1 = [1.0, 0.0, 0.0];
                    let e2 = [0.0, b[2], -b[1]];
                    let mut hits: Vec<(f64, usize)> = offsets
                        .iter()
                        .map(|off| {
                            let (a1, a2) = (off[0] * half_div, off[1] * half_div);
                            let bs: Point = std::array::from_fn(|k| b[k] + a1 * e1[k] + a2 * e2[k]);
                            let nrm = (bs[0] * bs[0] + bs[1] * bs[1] + bs[2] * bs[2]).sqrt();
                            let ds = apply(&m, &[bs[0] / nrm, bs[1] / nrm, bs[2] / nrm]);
                            let tg = terrain_range(&o, &ds, p.terrain_slope);
                            match cols.cast(&scene.xyz, &o, &ds, tg) {
                                Some((th, i)) => (th, i),
                                None => (tg, usize::MAX),
                            }
                        })
                        .collect();
                    hits.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    // Group echoes closer than the range resolution.
                    let mut groups: Vec<Echo> = Vec::new();
                    for h in hits {
                        match groups.last_mut() {
                            Some(g) if h.0 - g.2[0].0 < p.min_separation => {
                                g.0 += weight * h.0;
                                g.1 += weight;
                                g.2.push(h);
                            }
                            _ => groups.push((weight * h.0, weight, vec![h])),
                        }
                    }
                    let kept: Vec<&Echo> = groups.iter().filter(|g| g.1 >= p.detection_threshold - 1e-12).take(p.max_returns).collect();
                    let count = kept.len() as u8;
                    let noise = &noise[((j - j0) as usize) * p.max_returns..];
                    kept.iter()
                        .enumerate()
                        .map(|(k, g)| {
                            let range = g.0 / g.1 + noise[k];
                            // The class with the most energy in the return; the first hit on ties.
                            let class_of = |i: usize| if i == usize::MAX { 2u8 } else { cls[i] };
                            let mut best = (0usize, g.2[0].1);
                            for &(_, i) in &g.2 {
                                let c = class_of(i);
                                let n = g.2.iter().filter(|h| class_of(h.1) == c).count();
                                if n > best.0 {
                                    best = (n, i);
                                }
                            }
                            let class = class_of(best.1);
                            let tid = match (&tree, best.1) {
                                (Some(ids), i) if i != usize::MAX => ids[i],
                                _ => 0,
                            };
                            let inten = 65535.0 * 0.5 * g.1 * reflectance(class) * (p.altitude / range).powi(2);
                            Return {
                                xyz: [o[0] + range * d[0], o[1] + range * d[1], o[2] + range * d[2]],
                                t,
                                number: k as u8 + 1,
                                count,
                                scan_angle: (alpha - att[0]) as f32,
                                intensity: inten.round().clamp(0.0, 65535.0) as u16,
                                line: line.id,
                                class,
                                tree: tid,
                            }
                        })
                        .collect()
                })
                .collect();
            for r in block.into_iter().flatten() {
                if !p.clip || (r.xyz[0] >= bounds[0] && r.xyz[0] <= bounds[2] && r.xyz[1] >= bounds[1] && r.xyz[1] <= bounds[3]) {
                    out.push(r);
                }
            }
            task.inc(j1 - j0);
            j0 = j1;
        }
    }

    let mut points = PointCloud::new(out.iter().map(|r| r.xyz).collect());
    points.attrs.insert("gps_time".into(), Attr::F64(out.iter().map(|r| r.t).collect()));
    points.attrs.insert("return_number".into(), Attr::U8(out.iter().map(|r| r.number).collect()));
    points.attrs.insert("number_of_returns".into(), Attr::U8(out.iter().map(|r| r.count).collect()));
    points.attrs.insert("scan_angle".into(), Attr::F32(out.iter().map(|r| r.scan_angle).collect()));
    points.attrs.insert("intensity".into(), Attr::U16(out.iter().map(|r| r.intensity).collect()));
    points.attrs.insert("point_source_id".into(), Attr::U16(out.iter().map(|r| r.line).collect()));
    points.attrs.insert("classification".into(), Attr::U8(out.iter().map(|r| r.class).collect()));
    if tree.is_some() {
        points.attrs.insert("tree_id".into(), Attr::I32(out.iter().map(|r| r.tree).collect()));
    }

    let mut traj = Trajectory::default();
    for line in &plan.lines {
        let n = ((line.t1 - line.t0) * p.trajectory_rate).floor() as u64;
        let mut times: Vec<f64> = (0..=n).map(|k| line.t0 + k as f64 / p.trajectory_rate).collect();
        if line.t1 - times[times.len() - 1] > 1e-9 {
            times.push(line.t1);
        }
        for t in times {
            let (pos, att) = plan.pose(p, line, t);
            traj.time.push(t);
            traj.x.push(pos[0]);
            traj.y.push(pos[1]);
            traj.z.push(pos[2]);
            traj.roll.push(att[0]);
            traj.pitch.push(att[1]);
            traj.heading.push(att[2]);
            traj.line.push(line.id);
        }
    }
    Ok(Flight { points, trajectory: traj, n_pulses })
}

// ------------------------------------------------------------------ the trees of a scene

/// A tree of a synthetic scene: the truth that tree detection and crown
/// delineation are checked against.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneTree {
    /// The scene's `tree_id`.
    pub id: i32,
    /// Stem position: the mean x, y of the tree's wood points (all its
    /// points without a `classification`) within 1 m of its lowest point.
    pub stem: [f64; 2],
    /// The tree's highest point.
    pub top: Point,
    /// Height of the top above the terrain beneath it
    /// ([`crate::synthetic::terrain_height`] with `terrain_slope`).
    pub height: f64,
    /// Convex hull of the tree's points seen from above, counter-clockwise.
    pub crown: Vec<[f64; 2]>,
    /// Area of `crown` (m²).
    pub crown_area: f64,
}

/// The trees of a scene made by [`crate::synthetic::forest`] (points with
/// a `tree_id`, 0 or less for no tree), in order of id.
pub fn scene_trees(scene: &PointCloud, terrain_slope: f64) -> Result<Vec<SceneTree>> {
    let ids = scene.attr("tree_id").ok_or_else(|| Error::invalid("the scene has no 'tree_id'; make it with synthetic.forest"))?;
    let cls = scene.attr("classification");
    let mut groups: std::collections::BTreeMap<i32, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..scene.len() {
        let id = ids.get_f64(i) as i32;
        if id > 0 && scene.xyz[i].iter().all(|v| v.is_finite()) {
            groups.entry(id).or_default().push(i);
        }
    }
    Ok(groups
        .into_par_iter()
        .map(|(id, idx)| {
            let p = &scene.xyz;
            let zmin = idx.iter().map(|&i| p[i][2]).fold(f64::INFINITY, f64::min);
            let wood = |i: usize| cls.is_none_or(|c| c.get_f64(i) == 5.0);
            let low: Vec<usize> = idx.iter().copied().filter(|&i| p[i][2] <= zmin + 1.0 && wood(i)).collect();
            let low = if low.is_empty() { idx.iter().copied().filter(|&i| p[i][2] <= zmin + 1.0).collect() } else { low };
            let stem = [low.iter().map(|&i| p[i][0]).sum::<f64>() / low.len() as f64, low.iter().map(|&i| p[i][1]).sum::<f64>() / low.len() as f64];
            let top = idx.iter().map(|&i| p[i]).fold([f64::NAN, f64::NAN, f64::NEG_INFINITY], |a, q| if q[2] > a[2] { q } else { a });
            let xy: Vec<[f64; 2]> = idx.iter().map(|&i| [p[i][0], p[i][1]]).collect();
            let crown = crate::trees::convex_hull(&xy);
            let crown_area = if crown.len() >= 3 { crate::trees::polygon_area(&crown) } else { 0.0 };
            SceneTree { id, stem, top, height: top[2] - crate::synthetic::terrain_height(top[0], top[1], terrain_slope), crown, crown_area }
        })
        .collect())
}

/// `n` trees `(x, y, dbh, height)` for [`crate::synthetic::forest`], at
/// random in the `size` m square, no two stems closer than `min_spacing`,
/// heights uniform in `heights` and `dbh = 0.1 + 0.015 * height`. Stems
/// are drawn uniformly and rejected when too close to one already placed.
pub fn stand(n: usize, size: f64, min_spacing: f64, heights: (f64, f64), seed: u64) -> Result<Vec<(f64, f64, f64, f64)>> {
    if !(size.is_finite() && size > 0.0) || !(min_spacing.is_finite() && min_spacing >= 0.0) {
        return Err(Error::invalid(format!("size must be positive and min_spacing zero or more, got {size} and {min_spacing}")));
    }
    if !(heights.0.is_finite() && heights.1.is_finite() && heights.0 > 1.0 && heights.1 >= heights.0) {
        return Err(Error::invalid(format!("heights must be (low, high) with 1 < low <= high, got {heights:?}")));
    }
    let mut rng = Generator::new(seed);
    let mut out: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(n);
    let mut tries = 0usize;
    while out.len() < n {
        tries += 1;
        if tries > 1000 * n.max(1) {
            return Err(Error::invalid(format!("could not place {n} trees {min_spacing} m apart in a {size} m square (placed {})", out.len())));
        }
        let (x, y) = (rng.uniform(0.0, size), rng.uniform(0.0, size));
        if out.iter().any(|t| (t.0 - x).hypot(t.1 - y) < min_spacing) {
            continue;
        }
        let h = rng.uniform(heights.0, heights.1);
        out.push((x, y, 0.1 + 0.015 * h, h));
    }
    Ok(out)
}

/// Shape of the crowns of [`crown_forest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrownForm {
    /// An ellipsoid of revolution: horizontal semi-axis the crown radius,
    /// vertical semi-axis half the crown length, its top at the tree height.
    Ellipsoid,
    /// A cone with its apex at the tree height and its base, of the crown
    /// radius, a crown length below.
    Cone,
}

impl CrownForm {
    pub fn parse(s: &str) -> Result<CrownForm> {
        match s {
            "ellipsoid" => Ok(CrownForm::Ellipsoid),
            "cone" => Ok(CrownForm::Cone),
            _ => Err(Error::invalid(format!("unknown crown shape {s:?}; expected 'ellipsoid' or 'cone'"))),
        }
    }

    /// Is the point at horizontal distance `d` from the axis and `u` below
    /// the top inside a crown of radius `r` and length `l`?
    fn contains(self, d: f64, u: f64, r: f64, l: f64) -> bool {
        if !(0.0..=l).contains(&u) {
            return false;
        }
        match self {
            CrownForm::Ellipsoid => {
                let c = (u - l / 2.0) / (l / 2.0);
                (d / r).powi(2) + c * c <= 1.0
            }
            CrownForm::Cone => d <= r * u / l,
        }
    }

    /// Volume of a crown of radius `r` and length `l`.
    fn volume(self, r: f64, l: f64) -> f64 {
        match self {
            CrownForm::Ellipsoid => 4.0 / 3.0 * PI * r * r * l / 2.0,
            CrownForm::Cone => PI * r * r * l / 3.0,
        }
    }
}

/// A scene of trees with solid crowns, for airborne lidar: each tree
/// `(x, y, dbh, height)` has a stem (points on a cylinder of diameter
/// `dbh`, `classification` 5) from the terrain to its crown and a crown of
/// `form` filled uniformly with `density` leaf points per m³
/// (`classification` 4), of radius `crown_radius * height` and length
/// `crown_length * height`, its top `height` above the terrain at the
/// stem. The terrain is that of [`crate::synthetic::forest`] (slope 0.05),
/// with `ground_points` points over the `size` m square grown by `margin`.
/// `tree_id` is 0 for ground, then 1.. in list order. The crown of tree
/// `i` is drawn from seed `seed + i + 1`. Unlike the trees of
/// [`crate::synthetic::tree`], whose leaves cluster at the ends of a few
/// limbs, these crowns have the closed, convex outline that airborne
/// tree detection assumes, and a known projected area `π (crown_radius *
/// height)²`.
#[allow(clippy::too_many_arguments)]
pub fn crown_forest(trees: &[(f64, f64, f64, f64)], form: CrownForm, crown_radius: f64, crown_length: f64, density: f64, size: f64, ground_points: usize, margin: f64, seed: u64) -> Result<PointCloud> {
    if !(crown_radius.is_finite() && crown_radius > 0.0 && crown_length.is_finite() && crown_length > 0.0 && crown_length <= 1.0 && density.is_finite() && density > 0.0) {
        return Err(Error::invalid(format!("crown_radius and density must be positive and crown_length in (0, 1], got {crown_radius}, {density} and {crown_length}")));
    }
    if trees.iter().any(|t| !(t.0.is_finite() && t.1.is_finite() && t.2.is_finite() && t.2 >= 0.0 && t.3.is_finite() && t.3 > 0.0)) {
        return Err(Error::invalid("every tree needs finite x, y, a dbh of 0 or more and a positive height"));
    }
    let total: f64 = trees.iter().map(|t| form.volume(crown_radius * t.3, crown_length * t.3) * density).sum();
    limits::check((total as u64 + ground_points as u64).saturating_mul(64), &format!("a scene of about {} points", total as u64), "a lower density or fewer trees")?;
    let mut ground = crate::synthetic::forest(&[], size, ground_points, margin, seed);
    let parts: Vec<(Vec<Point>, Vec<u8>)> = trees
        .par_iter()
        .enumerate()
        .map(|(i, &(x, y, dbh, h))| {
            let mut rng = Generator::new(seed + i as u64 + 1);
            let z0 = crate::synthetic::terrain_height(x, y, 0.05);
            let (r, l) = (crown_radius * h, crown_length * h);
            let mut pts = Vec::new();
            let mut cls = Vec::new();
            // Stem up to the crown base.
            let stem_len = h - l;
            let ns = (200.0 * stem_len).round().max(0.0) as usize;
            if dbh > 0.0 {
                let t = rng.uniform_n(0.0, 1.0, ns);
                let a = rng.uniform_n(0.0, 2.0 * PI, ns);
                for k in 0..ns {
                    pts.push([x + dbh / 2.0 * a[k].cos(), y + dbh / 2.0 * a[k].sin(), z0 + t[k] * stem_len]);
                    cls.push(5u8);
                }
            }
            // Crown: uniform in its bounding cylinder, kept inside the form.
            let want = (form.volume(r, l) * density).round() as usize;
            let mut got = 0;
            while got < want {
                let v = rng.uniform_n(0.0, 1.0, 3 * 1024);
                for c in v.as_chunks::<3>().0 {
                    if got == want {
                        break;
                    }
                    let (d, a, u) = (r * c[0].sqrt(), 2.0 * PI * c[1], l * c[2]);
                    if form.contains(d, u, r, l) {
                        pts.push([x + d * a.cos(), y + d * a.sin(), z0 + h - u]);
                        cls.push(4u8);
                        got += 1;
                    }
                }
            }
            (pts, cls)
        })
        .collect();
    let mut cls = match ground.attrs.remove("classification") {
        Some(Attr::U8(c)) => c,
        _ => vec![2; ground.len()],
    };
    let mut ids = vec![0i32; ground.len()];
    for (i, (p, c)) in parts.into_iter().enumerate() {
        ids.resize(ids.len() + p.len(), i as i32 + 1);
        ground.xyz.extend(p);
        cls.extend(c);
    }
    ground.attrs.insert("classification".into(), Attr::U8(cls));
    ground.attrs.insert("tree_id".into(), Attr::I32(ids));
    Ok(ground)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rotation_points_the_beam_where_it_should() {
        let down = apply(&body_to_map(0.0, 0.0, 0.0), &beam_body(0.0));
        assert!((down[2] + 1.0).abs() < 1e-15 && down[0].abs() < 1e-15);
        // Flying north, a positive mirror angle looks east (the right wing).
        let right = apply(&body_to_map(0.0, 0.0, 0.0), &beam_body(30.0));
        assert!((right[0] - 0.5).abs() < 1e-12 && right[1].abs() < 1e-12);
        // Flying east, the right wing is south; rolling right wing down turns
        // the scanner, and the beam, back towards the left.
        let east = apply(&body_to_map(5.0, 0.0, 90.0), &beam_body(10.0));
        assert!((east[1] + 5f64.to_radians().sin()).abs() < 1e-12 && east[0].abs() < 1e-12, "{east:?}");
        // Nose up turns the belly, and the nadir beam, forwards.
        let ahead = apply(&body_to_map(0.0, 10.0, 0.0), &beam_body(0.0));
        assert!((ahead[1] - 10f64.to_radians().sin()).abs() < 1e-12);
    }

    #[test]
    fn mirror_patterns_sweep_the_full_angle() {
        let mut p = FlightParams { scan_rate: 10.0, scan_angle: 20.0, ..Default::default() };
        assert_eq!(mirror_angle(&p, 0.0), -20.0);
        assert!((mirror_angle(&p, 0.05) - 0.0).abs() < 1e-12);
        assert!((mirror_angle(&p, 0.1) - 20.0).abs() < 1e-9);
        assert!((mirror_angle(&p, 0.15) - 0.0).abs() < 1e-9);
        p.pattern = ScanPattern::Rotating;
        assert!((mirror_angle(&p, 0.099999) - 20.0).abs() < 1e-3);
        assert!((mirror_angle(&p, 0.1) + 20.0).abs() < 1e-9);
    }

    #[test]
    fn terrain_is_hit_on_the_surface() {
        let o = [3.0, 7.0, 60.0];
        let d = apply(&body_to_map(1.0, -2.0, 33.0), &beam_body(25.0));
        let t = terrain_range(&o, &d, 0.05);
        let q = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
        assert!((q[2] - crate::synthetic::terrain_height(q[0], q[1], 0.05)).abs() < 1e-9);
    }

    #[test]
    fn rays_stop_at_the_first_point() {
        let pts = vec![[0.0, 0.0, 5.0], [0.0, 0.0, 10.0], [3.0, 0.0, 12.0]];
        let cols = Columns::new(&pts, &[0, 1, 2], 0.05).unwrap();
        let hit = cols.cast(&pts, &[0.0, 0.0, 50.0], &[0.0, 0.0, -1.0], 100.0).unwrap();
        assert_eq!(hit.1, 1);
        assert!((hit.0 - 39.95).abs() < 1e-12);
        let s = (0.5f64).sqrt();
        let slanted = cols.cast(&pts, &[3.0 - 38.0, 0.0, 50.0], &[s, 0.0, -s], 100.0).unwrap();
        assert_eq!(slanted.1, 2);
        assert!(cols.cast(&pts, &[1.0, 1.0, 50.0], &[0.0, 0.0, -1.0], 100.0).is_none());
        assert!(cols.cast(&pts, &[0.0, 0.0, 50.0], &[0.0, 0.0, -1.0], 39.0).is_none());
    }

    #[test]
    fn a_flight_is_reproducible_and_geometrically_exact() {
        let scene = crate::synthetic::forest(&crate::synthetic::DEFAULT_TREES, 20.0, 1000, 4.0, 0);
        let p = FlightParams { pulse_rate: 5000.0, altitude: 60.0, line_spacing: 15.0, ..Default::default() };
        let f = fly(&scene, &p).unwrap();
        assert_eq!(f, fly(&scene, &p).unwrap());
        let n = f.points.len();
        assert!(n > 10_000, "{n}");
        let lines = f.trajectory.line.iter().max().copied().unwrap();
        assert_eq!(lines, 2);
        let Some(Attr::F64(t)) = f.points.attrs.get("gps_time") else { panic!() };
        let origins = f.trajectory.positions(t);
        let Some(Attr::F32(sa)) = f.points.attrs.get("scan_angle") else { panic!() };
        for i in (0..n).step_by(97) {
            let o = origins[i];
            let q = f.points.xyz[i];
            let v = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            // The beam direction from the attitude and the mirror angle.
            let k = f.trajectory.time.partition_point(|&s| s <= t[i]) - 1;
            let roll = f.trajectory.roll[k];
            let m = body_to_map(roll, f.trajectory.pitch[k], f.trajectory.heading[k]);
            let d = apply(&m, &beam_body(sa[i] as f64 + roll));
            let cos = (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]) / r;
            assert!(cos > 1.0 - 1e-6, "point {i}: {cos}");
        }
        let Some(Attr::U8(nr)) = f.points.attrs.get("number_of_returns") else { panic!() };
        assert!(nr.iter().any(|&c| c > 1));
    }

    #[test]
    fn scene_trees_know_their_tops_and_crowns() {
        let trees = stand(6, 30.0, 6.0, (10.0, 20.0), 2).unwrap();
        assert_eq!(trees.len(), 6);
        for (i, a) in trees.iter().enumerate() {
            for b in &trees[i + 1..] {
                assert!((a.0 - b.0).hypot(a.1 - b.1) >= 6.0);
            }
        }
        let scene = crate::synthetic::forest(&trees, 30.0, 1000, 4.0, 0);
        let st = scene_trees(&scene, 0.05).unwrap();
        assert_eq!(st.len(), 6);
        for (t, s) in trees.iter().zip(&st) {
            assert!((s.stem[0] - t.0).abs() < 0.05 && (s.stem[1] - t.1).abs() < 0.05, "{s:?}");
            assert!(s.height > 0.8 * t.3 && s.height < 1.3 * t.3, "{} vs {}", s.height, t.3);
            assert!(s.crown_area > 1.0);
        }
        assert!(stand(100, 10.0, 5.0, (10.0, 20.0), 0).is_err());
    }

    #[test]
    fn crown_forest_has_solid_crowns_of_known_size() {
        let trees = [(10.0, 10.0, 0.3, 20.0), (25.0, 12.0, 0.2, 12.0)];
        for form in [CrownForm::Ellipsoid, CrownForm::Cone] {
            let s = crown_forest(&trees, form, 0.25, 0.5, 40.0, 40.0, 500, 0.0, 1).unwrap();
            let st = scene_trees(&s, 0.05).unwrap();
            assert_eq!(st.len(), 2);
            for (t, s) in trees.iter().zip(&st) {
                let r = 0.25 * t.3;
                assert!((s.stem[0] - t.0).abs() < 0.05 && (s.stem[1] - t.1).abs() < 0.05);
                assert!(s.height > t.3 - 0.6 && s.height <= t.3 + 0.2, "{form:?} {}", s.height);
                let a = PI * r * r;
                assert!(s.crown_area < a && s.crown_area > 0.85 * a, "{form:?} {} vs {a}", s.crown_area);
            }
            let want = form.volume(5.0, 10.0) * 40.0;
            let got = (0..s.len()).filter(|&i| s.attr("tree_id").unwrap().get_f64(i) == 1.0 && s.attr("classification").unwrap().get_f64(i) == 4.0).count();
            assert!((got as f64 - want).abs() <= 1.0);
        }
        assert!(crown_forest(&trees, CrownForm::Cone, 0.0, 0.5, 40.0, 40.0, 10, 0.0, 1).is_err());
        assert!(CrownForm::parse("sphere").is_err());
    }
}
