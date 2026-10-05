// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Flight trajectories of airborne and UAV laser scanners.
//!
//! Ray-based canopy metrics need the position of the scanner when each
//! pulse left. Survey providers deliver it as a trajectory: an SBET
//! ("smoothed best estimate of trajectory", the binary format of Applanix
//! POSPac and most GNSS/INS software) or a text table of time, position
//! and attitude. [`Trajectory::positions`] interpolates it at the returns'
//! GPS times.
//!
//! Without a trajectory, [`estimate`] recovers an approximate one from the
//! returns themselves: every pulse with two or more returns defines a line
//! through the scanner, and within a short time window the scanner moves
//! along a straight line, so its position and velocity follow from a linear
//! least-squares intersection of those lines (the idea of Gatziolis &
//! McGaughey 2019 and of lidR's `track_sensor`, Roussel et al. 2020, with
//! the motion within the window modelled rather than ignored).

use std::io::Read;
use std::path::Path;

use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::{Point, PointCloud};

/// A trajectory: strictly increasing times with positions and, optionally,
/// attitude (degrees; heading clockwise from north).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trajectory {
    pub time: Vec<f64>,
    pub xyz: Vec<Point>,
    pub roll: Option<Vec<f64>>,
    pub pitch: Option<Vec<f64>>,
    pub heading: Option<Vec<f64>>,
}

impl Trajectory {
    /// Check and order the samples: sorted by time, a sample whose time
    /// repeats an earlier one is dropped.
    ///
    /// # Errors
    /// Fewer than two samples, arrays of different lengths, or non-finite
    /// times or positions.
    pub fn new(time: Vec<f64>, xyz: Vec<Point>, roll: Option<Vec<f64>>, pitch: Option<Vec<f64>>, heading: Option<Vec<f64>>) -> Result<Trajectory> {
        let n = time.len();
        if xyz.len() != n {
            return Err(Error::invalid(format!("the trajectory has {n} times but {} positions", xyz.len())));
        }
        for (name, a) in [("roll", &roll), ("pitch", &pitch), ("heading", &heading)] {
            if let Some(a) = a {
                if a.len() != n {
                    return Err(Error::invalid(format!("the trajectory has {n} times but {} {name} values", a.len())));
                }
            }
        }
        if let Some(i) = (0..n).find(|&i| !time[i].is_finite() || !xyz[i].iter().all(|v| v.is_finite())) {
            return Err(Error::invalid(format!("trajectory sample {i} has a non-finite time or position")));
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then(a.cmp(&b)));
        order.dedup_by(|b, a| time[*a] == time[*b]);
        if order.len() < 2 {
            return Err(Error::invalid(format!("a trajectory needs at least two samples at different times, got {}", order.len())));
        }
        let take = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        Ok(Trajectory {
            time: take(&time),
            xyz: order.iter().map(|&i| xyz[i]).collect(),
            roll: roll.as_deref().map(take),
            pitch: pitch.as_deref().map(take),
            heading: heading.as_deref().map(take),
        })
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }

    /// Median time between samples.
    pub fn median_interval(&self) -> f64 {
        let mut d: Vec<f64> = self.time.windows(2).map(|w| w[1] - w[0]).collect();
        if d.is_empty() {
            return f64::NAN;
        }
        let m = d.len() / 2;
        *d.select_nth_unstable_by(m, f64::total_cmp).1
    }

    /// The default largest gap between samples to interpolate across: ten
    /// median intervals.
    pub fn default_max_gap(&self) -> f64 {
        10.0 * self.median_interval()
    }

    /// Bracketing samples and weight of the later one for time `t`; None
    /// outside the trajectory or across a gap longer than `max_gap`.
    fn bracket(&self, t: f64, max_gap: f64) -> Option<(usize, usize, f64)> {
        if !t.is_finite() {
            return None;
        }
        let k = self.time.partition_point(|&s| s <= t);
        if k == 0 {
            return None;
        }
        let i = k - 1;
        if self.time[i] == t {
            return Some((i, i, 0.0));
        }
        if k >= self.time.len() || self.time[k] - self.time[i] > max_gap {
            return None;
        }
        Some((i, k, (t - self.time[i]) / (self.time[k] - self.time[i])))
    }

    /// Sensor position at each time, linearly interpolated; NaN outside the
    /// trajectory or within a gap between samples longer than `max_gap` (s).
    pub fn positions(&self, times: &[f64], max_gap: f64) -> Vec<Point> {
        times
            .par_iter()
            .with_min_len(4096)
            .map(|&t| match self.bracket(t, max_gap) {
                None => [f64::NAN; 3],
                Some((i, k, f)) => {
                    let (a, b) = (self.xyz[i], self.xyz[k]);
                    [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])]
                }
            })
            .collect()
    }

    /// Roll, pitch and heading (degrees) at each time, linearly
    /// interpolated (heading along the shorter arc, wrapped to `[0, 360)`);
    /// None when the trajectory has no attitude.
    pub fn attitude(&self, times: &[f64], max_gap: f64) -> Option<Vec<[f64; 3]>> {
        let (r, p, h) = (self.roll.as_ref()?, self.pitch.as_ref()?, self.heading.as_ref()?);
        Some(
            times
                .par_iter()
                .with_min_len(4096)
                .map(|&t| match self.bracket(t, max_gap) {
                    None => [f64::NAN; 3],
                    Some((i, k, f)) => {
                        let dh = (h[k] - h[i] + 540.0).rem_euclid(360.0) - 180.0;
                        [r[i] + f * (r[k] - r[i]), p[i] + f * (p[k] - p[i]), (h[i] + f * dh).rem_euclid(360.0)]
                    }
                })
                .collect(),
        )
    }
}

// ------------------------------------------------------------------ files

/// Bytes per SBET record: 17 little-endian doubles.
pub const SBET_RECORD: usize = 136;

/// The records of an SBET file, angles in degrees.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sbet {
    /// GPS seconds of the week.
    pub time: Vec<f64>,
    pub latitude: Vec<f64>,
    pub longitude: Vec<f64>,
    /// Ellipsoidal height (m).
    pub height: Vec<f64>,
    pub roll: Vec<f64>,
    pub pitch: Vec<f64>,
    /// True heading, clockwise from north in `[0, 360)`: the platform
    /// heading minus the wander angle.
    pub heading: Vec<f64>,
}

/// Read an SBET file: records of 17 little-endian doubles, time (GPS
/// seconds of the week), latitude, longitude (radians), ellipsoidal height,
/// three velocities, roll, pitch, platform heading, wander angle (radians),
/// three accelerations and three angular rates.
///
/// # Errors
/// A missing file, or one whose size is not a whole number of records.
pub fn read_sbet(path: impl AsRef<Path>) -> Result<Sbet> {
    let path = path.as_ref();
    let mut bytes = Vec::new();
    std::fs::File::open(path).and_then(|mut f| f.read_to_end(&mut bytes)).map_err(|e| Error::file(path, e.to_string()))?;
    if bytes.is_empty() || bytes.len() % SBET_RECORD != 0 {
        return Err(Error::file(path, format!("{} bytes is not a whole number of {SBET_RECORD}-byte SBET records", bytes.len())));
    }
    let n = bytes.len() / SBET_RECORD;
    let mut s = Sbet::default();
    let field = |r: usize, k: usize| f64::from_le_bytes(bytes[r * SBET_RECORD + 8 * k..r * SBET_RECORD + 8 * k + 8].try_into().expect("8 bytes"));
    for r in 0..n {
        s.time.push(field(r, 0));
        s.latitude.push(field(r, 1).to_degrees());
        s.longitude.push(field(r, 2).to_degrees());
        s.height.push(field(r, 3));
        s.roll.push(field(r, 7).to_degrees());
        s.pitch.push(field(r, 8).to_degrees());
        s.heading.push((field(r, 9) - field(r, 10)).to_degrees().rem_euclid(360.0));
    }
    Ok(s)
}

/// Write an SBET file (velocities, accelerations, angular rates and the
/// wander angle are written as 0). Angles in degrees.
pub fn write_sbet(path: impl AsRef<Path>, s: &Sbet) -> Result<()> {
    let path = path.as_ref();
    let n = s.time.len();
    for (name, v) in [("latitude", &s.latitude), ("longitude", &s.longitude), ("height", &s.height), ("roll", &s.roll), ("pitch", &s.pitch), ("heading", &s.heading)] {
        if v.len() != n {
            return Err(Error::invalid(format!("SBET {name} has {} values for {n} times", v.len())));
        }
    }
    let mut out = Vec::with_capacity(n * SBET_RECORD);
    for i in 0..n {
        let mut rec = [0.0f64; 17];
        rec[0] = s.time[i];
        rec[1] = s.latitude[i].to_radians();
        rec[2] = s.longitude[i].to_radians();
        rec[3] = s.height[i];
        rec[7] = s.roll[i].to_radians();
        rec[8] = s.pitch[i].to_radians();
        rec[9] = s.heading[i].to_radians();
        for v in rec {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, out).map_err(|e| Error::file(path, e.to_string()))
}

/// Which trajectory quantity a text column holds, from its name (case and
/// punctuation ignored): `time` / `gps_time` / `t` / `timestamp`, `x` /
/// `easting`, `y` / `northing`, `z` / `height` / `altitude` / `elevation`,
/// `roll`, `pitch`, `heading` / `yaw`.
pub fn column_role(name: &str) -> Option<&'static str> {
    let key: String = name.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
    Some(match key.as_str() {
        "time" | "gpstime" | "t" | "timestamp" | "gpst" | "sec" | "seconds" => "time",
        "x" | "easting" | "east" | "e" => "x",
        "y" | "northing" | "north" | "n" => "y",
        "z" | "height" | "altitude" | "alt" | "elevation" | "h" | "ellipsoidheight" => "z",
        "roll" | "r" => "roll",
        "pitch" | "p" => "pitch",
        "heading" | "yaw" | "azimuth" => "heading",
        _ => return None,
    })
}

/// A text table: column names (from a header line, else `col0`, `col1`,
/// ...) and the columns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Table {
    pub names: Vec<String>,
    pub columns: Vec<Vec<f64>>,
}

/// Read a delimited text table of numbers. Blank lines and lines starting
/// with `#` or `//` are skipped; fields are separated by commas, semicolons
/// or white space (whichever the first data line uses); a first line that
/// is not all numbers is the header.
///
/// # Errors
/// A missing file, no data, or rows with differing numbers of fields or
/// fields that are not numbers (the line is named).
pub fn read_table(path: impl AsRef<Path>) -> Result<Table> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| Error::file(path, e.to_string()))?;
    parse_table(&text).map_err(|e| match e {
        Error::Invalid(m) => Error::file(path, m),
        other => other,
    })
}

/// [`read_table`] on text in memory.
pub fn parse_table(text: &str) -> Result<Table> {
    let mut lines = text.lines().enumerate().map(|(i, l)| (i + 1, l.trim())).filter(|(_, l)| !l.is_empty() && !l.starts_with('#') && !l.starts_with("//"));
    let Some((_, first)) = lines.next() else { return Err(Error::invalid("the trajectory file has no data")) };
    let delim = if first.contains(',') {
        Some(',')
    } else if first.contains(';') {
        Some(';')
    } else {
        None
    };
    let split = |l: &str| -> Vec<String> {
        match delim {
            Some(c) => l.split(c).map(|s| s.trim().trim_matches('"').to_string()).collect(),
            None => l.split_whitespace().map(|s| s.trim_matches('"').to_string()).collect(),
        }
    };
    let head = split(first);
    let numeric = head.iter().all(|s| s.parse::<f64>().is_ok());
    let names: Vec<String> = if numeric { (0..head.len()).map(|i| format!("col{i}")).collect() } else { head.clone() };
    let mut columns = vec![Vec::new(); names.len()];
    let mut push = |no: usize, fields: Vec<String>| -> Result<()> {
        if fields.len() != names.len() {
            return Err(Error::invalid(format!("line {no} has {} fields, expected {}", fields.len(), names.len())));
        }
        for (c, f) in fields.iter().enumerate() {
            columns[c].push(f.parse::<f64>().map_err(|_| Error::invalid(format!("line {no}: {f:?} is not a number")))?);
        }
        Ok(())
    };
    if numeric {
        push(1, head)?;
    }
    for (no, l) in lines {
        push(no, split(l))?;
    }
    if columns.first().is_none_or(|c| c.is_empty()) {
        return Err(Error::invalid("the trajectory file has a header but no data"));
    }
    Ok(Table { names, columns })
}

// ------------------------------------------------------------- estimation

/// Settings of [`estimate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimateParams {
    /// Length (s) of the time windows; one position per window and line.
    pub interval: f64,
    /// Fewest usable pulses in a window.
    pub min_pulses: usize,
    /// Least distance (m) between a pulse's first and last returns for it
    /// to be used: the direction of a shorter pair is too uncertain.
    pub min_separation: f64,
    /// Most pulses used per window (those with the widest separation).
    pub max_pulses: usize,
    /// Longest time (s) the first and last windows of a line are
    /// extrapolated, with their fitted velocity, towards the line's first
    /// and last returns (over open ground a line has no multiple returns).
    pub extend: f64,
}

impl Default for EstimateParams {
    fn default() -> Self {
        EstimateParams { interval: 0.5, min_pulses: 30, min_separation: 2.0, max_pulses: 4000, extend: 2.0 }
    }
}

/// First and last `gps_time` of every flight line (`point_source_id`, or
/// one line without it), as `(line, first, last)` in line order.
pub fn line_extents(cloud: &PointCloud) -> Result<Vec<(i64, f64, f64)>> {
    let time = cloud.attr_f64("gps_time").ok_or_else(|| Error::invalid("the points have no gps_time attribute, so their pulses cannot be told apart"))?;
    let line = cloud.attr_f64("point_source_id");
    Ok(merge_extents(time.iter().enumerate().filter(|(_, t)| t.is_finite()).map(|(i, &t)| (line.as_ref().map_or(0, |l| l[i] as i64), t, t))))
}

/// Merge [`line_extents`] of several parts.
pub fn merge_extents(parts: impl IntoIterator<Item = (i64, f64, f64)>) -> Vec<(i64, f64, f64)> {
    let mut ext: std::collections::BTreeMap<i64, (f64, f64)> = std::collections::BTreeMap::new();
    for (l, a, b) in parts {
        let e = ext.entry(l).or_insert((a, b));
        e.0 = e.0.min(a);
        e.1 = e.1.max(b);
    }
    ext.into_iter().map(|(l, (a, b))| (l, a, b)).collect()
}

/// One pulse's line: time, flight line, a point on it (the last return),
/// the unit direction towards the sensor, and the first-to-last distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseLine {
    pub time: f64,
    pub line: i64,
    pub point: Point,
    pub dir: Point,
    pub separation: f64,
}

/// The lines of every multi-return pulse of a cloud whose first and last
/// returns are at least `min_separation` apart. Returns of a pulse share
/// `gps_time` (and `point_source_id`, the flight line, when present); the
/// first and last are told by `return_number`, else by height.
pub fn pulse_lines(cloud: &PointCloud, min_separation: f64) -> Result<Vec<PulseLine>> {
    let time = cloud.attr_f64("gps_time").ok_or_else(|| Error::invalid("the points have no gps_time attribute, so their pulses cannot be told apart"))?;
    let line = cloud.attr_f64("point_source_id");
    let rn = cloud.attr_f64("return_number");
    let n = cloud.len();
    let key = |i: usize| (line.as_ref().map_or(0, |l| l[i] as i64), time[i]);
    let mut order: Vec<usize> = (0..n).filter(|&i| time[i].is_finite()).collect();
    order.par_sort_by(|&a, &b| {
        let (ka, kb) = (key(a), key(b));
        ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then(a.cmp(&b))
    });
    let mut out = Vec::new();
    let mut s = 0;
    while s < order.len() {
        let mut e = s + 1;
        while e < order.len() && key(order[e]) == key(order[s]) {
            e += 1;
        }
        if e - s >= 2 {
            // First: lowest return number (else highest point); last: the opposite.
            let rank = |i: usize| rn.as_ref().map_or(-cloud.xyz[i][2], |r| r[i]);
            let g = &order[s..e];
            let first = *g.iter().min_by(|&&a, &&b| rank(a).total_cmp(&rank(b)).then(a.cmp(&b))).expect("non-empty");
            let last = *g.iter().max_by(|&&a, &&b| rank(a).total_cmp(&rank(b)).then(b.cmp(&a))).expect("non-empty");
            let (p, q) = (cloud.xyz[first], cloud.xyz[last]);
            let v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
            let sep = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if sep >= min_separation && v[2] > 0.0 {
                let (l, t) = key(first);
                out.push(PulseLine { time: t, line: l, point: q, dir: [v[0] / sep, v[1] / sep, v[2] / sep], separation: sep });
            }
        }
        s = e;
    }
    Ok(out)
}

/// Keep at most `max` lines per flight line and `interval` window (GPS
/// time anchored at multiples of `interval`): those with the widest
/// separation, ties by time. [`estimate`] fits no more than that, so
/// thinning each tile first bounds the memory without changing the fits
/// (only the times of the samples at each line's ends can move, to the
/// first and last pulse kept).
pub fn thin(mut lines: Vec<PulseLine>, interval: f64, max: usize) -> Vec<PulseLine> {
    let win = |l: &PulseLine| (l.line, (l.time / interval).floor() as i64);
    lines.sort_by(|a, b| win(a).cmp(&win(b)).then(b.separation.total_cmp(&a.separation)).then(a.time.total_cmp(&b.time)));
    let mut out = Vec::with_capacity(lines.len());
    let mut s = 0;
    while s < lines.len() {
        let mut e = s + 1;
        while e < lines.len() && win(&lines[e]) == win(&lines[s]) {
            e += 1;
        }
        out.extend_from_slice(&lines[s..(s + max).min(e)]);
        s = e;
    }
    out
}

/// A trajectory estimated from the returns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Estimated {
    pub trajectory: Trajectory,
    /// Flight line of each sample.
    pub line: Vec<i64>,
    /// Pulses used for each sample.
    pub n_pulses: Vec<usize>,
    /// Root mean square distance (m) of those pulses' lines from the fitted
    /// position.
    pub rms: Vec<f64>,
}

/// Least-squares position `a` and velocity `b` of a sensor moving as
/// `a + b (t - tc)` that the pulse lines pass closest to, weighting each
/// line by its squared separation. None when the lines do not fix it.
fn fit_window(lines: &[&PulseLine], tc: f64) -> Option<(Point, Point)> {
    let mut m = SMatrix::<f64, 6, 6>::zeros();
    let mut rhs = SVector::<f64, 6>::zeros();
    for l in lines {
        let w = l.separation * l.separation;
        let tau = l.time - tc;
        let d = l.dir;
        let mut p = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                p[r][c] = if r == c { 1.0 } else { 0.0 } - d[r] * d[c];
            }
        }
        let pq: [f64; 3] = std::array::from_fn(|r| (0..3).map(|c| p[r][c] * l.point[c]).sum());
        for r in 0..3 {
            for c in 0..3 {
                let v = w * p[r][c];
                m[(r, c)] += v;
                m[(r, c + 3)] += tau * v;
                m[(r + 3, c)] += tau * v;
                m[(r + 3, c + 3)] += tau * tau * v;
            }
            rhs[r] += w * pq[r];
            rhs[r + 3] += w * tau * pq[r];
        }
    }
    // Scale the velocity unknowns so the system is well conditioned.
    let span = lines.iter().map(|l| (l.time - tc).abs()).fold(0.0, f64::max).max(1e-6);
    let mut s = SMatrix::<f64, 6, 6>::identity();
    for k in 3..6 {
        s[(k, k)] = 1.0 / span;
    }
    let ms = s * m * s;
    let eig = ms.symmetric_eigenvalues();
    let (lo, hi) = eig.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v.abs())));
    if lo.is_nan() || lo <= 1e-9 * hi {
        return None;
    }
    let y = ms.cholesky()?.solve(&(s * rhs));
    let x = s * y;
    Some(([x[0], x[1], x[2]], [x[3], x[4], x[5]]))
}

fn line_distance(l: &PulseLine, o: &Point) -> f64 {
    let v = [o[0] - l.point[0], o[1] - l.point[1], o[2] - l.point[2]];
    let t = v[0] * l.dir[0] + v[1] * l.dir[1] + v[2] * l.dir[2];
    let perp = [v[0] - t * l.dir[0], v[1] - t * l.dir[1], v[2] - t * l.dir[2]];
    (perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt()
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let m = v.len() / 2;
    *v.select_nth_unstable_by(m, f64::total_cmp).1
}

/// Estimate the trajectory from pulse lines (see the module docs). Windows
/// of `interval` s are anchored at multiples of `interval` in GPS time,
/// per flight line; each gives a sample at the mean time of its pulses,
/// and the first and last windows of a line also give samples at the
/// line's first and last returns (from `extents`, see [`line_extents`]),
/// extrapolated with the fitted velocity by at most `extend` s beyond the
/// window's own pulses, so that the whole line can be interpolated. A window's fit is
/// repeated once without the lines more than three median distances (and
/// 5 cm) from the first fit, and rejected if the sensor comes out below
/// the returns.
///
/// # Errors
/// Bad settings, or no window with enough pulses.
pub fn estimate(lines: &[PulseLine], extents: &[(i64, f64, f64)], params: &EstimateParams) -> Result<Estimated> {
    if !(params.interval.is_finite() && params.interval > 0.0) {
        return Err(Error::invalid(format!("interval must be a positive number of seconds, got {}", params.interval)));
    }
    if params.min_pulses < 3 || params.max_pulses < params.min_pulses {
        return Err(Error::invalid("min_pulses must be at least 3 and at most max_pulses"));
    }
    if params.extend.is_nan() || params.extend < 0.0 {
        return Err(Error::invalid(format!("extend must be a non-negative number of seconds, got {}", params.extend)));
    }
    // Windows keyed by (line, window index), in order.
    let mut keyed: Vec<(i64, i64, usize)> = lines.iter().enumerate().map(|(i, l)| (l.line, (l.time / params.interval).floor() as i64, i)).collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(lines[a.2].time.total_cmp(&lines[b.2].time)).then(a.2.cmp(&b.2)));
    let mut windows: Vec<(i64, Vec<usize>)> = Vec::new();
    let mut s = 0;
    while s < keyed.len() {
        let mut e = s + 1;
        while e < keyed.len() && keyed[e].0 == keyed[s].0 && keyed[e].1 == keyed[s].1 {
            e += 1;
        }
        windows.push((keyed[s].0, keyed[s..e].iter().map(|k| k.2).collect()));
        s = e;
    }
    struct Fit {
        line: i64,
        tc: f64,
        t0: f64,
        t1: f64,
        a: Point,
        b: Point,
        n: usize,
        rms: f64,
    }
    let fits: Vec<Option<Fit>> = windows
        .par_iter()
        .map(|(line, members)| {
            if members.len() < params.min_pulses {
                return None;
            }
            let mut use_: Vec<&PulseLine> = members.iter().map(|&i| &lines[i]).collect();
            if use_.len() > params.max_pulses {
                use_.sort_by(|a, b| b.separation.total_cmp(&a.separation).then(a.time.total_cmp(&b.time)));
                use_.truncate(params.max_pulses);
            }
            let tc = use_.iter().map(|l| l.time).sum::<f64>() / use_.len() as f64;
            let (a, b) = fit_window(&use_, tc)?;
            let at = |l: &PulseLine| -> Point { let tau = l.time - tc; [a[0] + b[0] * tau, a[1] + b[1] * tau, a[2] + b[2] * tau] };
            let d: Vec<f64> = use_.iter().map(|l| line_distance(l, &at(l))).collect();
            let cut = (3.0 * median(d.clone())).max(0.05);
            let kept: Vec<&PulseLine> = use_.iter().zip(&d).filter(|(_, &v)| v <= cut).map(|(l, _)| *l).collect();
            if kept.len() < params.min_pulses {
                return None;
            }
            let (a, b) = fit_window(&kept, tc)?;
            let at = |l: &PulseLine| -> Point { let tau = l.time - tc; [a[0] + b[0] * tau, a[1] + b[1] * tau, a[2] + b[2] * tau] };
            let rms = (kept.iter().map(|l| line_distance(l, &at(l)).powi(2)).sum::<f64>() / kept.len() as f64).sqrt();
            let top = kept.iter().map(|l| l.point[2] + l.dir[2] * l.separation).fold(f64::NEG_INFINITY, f64::max);
            if a[2].is_nan() || a[2] <= top {
                return None;
            }
            let t0 = members.iter().map(|&i| lines[i].time).fold(f64::INFINITY, f64::min);
            let t1 = members.iter().map(|&i| lines[i].time).fold(f64::NEG_INFINITY, f64::max);
            Some(Fit { line: *line, tc, t0, t1, a, b, n: kept.len(), rms })
        })
        .collect();
    let mut fits: Vec<Fit> = fits.into_iter().flatten().collect();
    // A window whose position is far off the straight line through the
    // others of its flight line (a partial window at a line's end, a narrow
    // fan of lines) is dropped: more than five median residuals and a metre.
    let mut keep = vec![true; fits.len()];
    let mut s = 0;
    while s < fits.len() {
        let mut e = s + 1;
        while e < fits.len() && fits[e].line == fits[s].line {
            e += 1;
        }
        if e - s >= 3 {
            let w = &fits[s..e];
            let sw: f64 = w.iter().map(|f| f.n as f64).sum();
            let tm = w.iter().map(|f| f.n as f64 * f.tc).sum::<f64>() / sw;
            let stt: f64 = w.iter().map(|f| f.n as f64 * (f.tc - tm).powi(2)).sum();
            let am: Point = std::array::from_fn(|k| w.iter().map(|f| f.n as f64 * f.a[k]).sum::<f64>() / sw);
            let v: Point = std::array::from_fn(|k| if stt > 0.0 { w.iter().map(|f| f.n as f64 * (f.tc - tm) * (f.a[k] - am[k])).sum::<f64>() / stt } else { 0.0 });
            let r: Vec<f64> = w.iter().map(|f| (0..3).map(|k| (f.a[k] - am[k] - v[k] * (f.tc - tm)).powi(2)).sum::<f64>().sqrt()).collect();
            let cut = (5.0 * median(r.clone())).max(1.0);
            for (j, rj) in r.iter().enumerate() {
                keep[s + j] = *rj <= cut;
            }
        }
        s = e;
    }
    let mut k = 0;
    fits.retain(|_| {
        k += 1;
        keep[k - 1]
    });
    if fits.is_empty() {
        return Err(Error::invalid(format!(
            "no {} s window has {} pulses with returns at least {} m apart; the trajectory cannot be estimated from these points (too few multiple returns: give the trajectory, or a longer interval)",
            params.interval, params.min_pulses, params.min_separation
        )));
    }
    let mut out = Estimated::default();
    let (mut time, mut xyz) = (Vec::new(), Vec::new());
    let mut push = |t: f64, p: Point, f: &Fit| {
        time.push(t);
        xyz.push(p);
        out.line.push(f.line);
        out.n_pulses.push(f.n);
        out.rms.push(f.rms);
    };
    // The ends of a line are extrapolated with the line's velocity: the
    // slope of its window positions against time (weighted by pulses), which
    // is far steadier than one short window's own velocity.
    let velocity = |line: i64, own: Point| -> Point {
        let w: Vec<&Fit> = fits.iter().filter(|f| f.line == line).collect();
        if w.len() < 2 {
            return own;
        }
        let sw: f64 = w.iter().map(|f| f.n as f64).sum();
        let tm = w.iter().map(|f| f.n as f64 * f.tc).sum::<f64>() / sw;
        let stt: f64 = w.iter().map(|f| f.n as f64 * (f.tc - tm).powi(2)).sum();
        if stt.is_nan() || stt <= 0.0 {
            return own;
        }
        std::array::from_fn(|k| {
            let am = w.iter().map(|f| f.n as f64 * f.a[k]).sum::<f64>() / sw;
            w.iter().map(|f| f.n as f64 * (f.tc - tm) * (f.a[k] - am)).sum::<f64>() / stt
        })
    };
    let at = |f: &Fit, t: f64| -> Point {
        let v = velocity(f.line, f.b);
        let tau = t - f.tc;
        [f.a[0] + v[0] * tau, f.a[1] + v[1] * tau, f.a[2] + v[2] * tau]
    };
    for (k, f) in fits.iter().enumerate() {
        let first = k == 0 || fits[k - 1].line != f.line;
        let last = k + 1 == fits.len() || fits[k + 1].line != f.line;
        let ext = extents.iter().find(|e| e.0 == f.line);
        if first {
            let t = ext.map_or(f.t0, |e| e.1.max(f.t0 - params.extend)).min(f.t0);
            if t < f.tc {
                push(t, at(f, t), f);
            }
        }
        push(f.tc, f.a, f);
        if last {
            let t = ext.map_or(f.t1, |e| e.2.min(f.t1 + params.extend)).max(f.t1);
            if t > f.tc {
                push(t, at(f, t), f);
            }
        }
    }
    // Lines are fitted separately but share one clock: order by time.
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then(a.cmp(&b)));
    order.dedup_by(|b, a| time[*a] == time[*b]);
    let pick_f = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let trajectory = Trajectory { time: pick_f(&time), xyz: order.iter().map(|&i| xyz[i]).collect(), roll: None, pitch: None, heading: None };
    Ok(Estimated {
        line: order.iter().map(|&i| out.line[i]).collect(),
        n_pulses: order.iter().map(|&i| out.n_pulses[i]).collect(),
        rms: pick_f(&out.rms),
        trajectory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight() -> Trajectory {
        let time: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let xyz = time.iter().map(|&t| [10.0 * t, 0.0, 100.0]).collect();
        let heading = vec![350.0, 355.0, 0.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        Trajectory::new(time, xyz, Some(vec![0.0; 11]), Some(vec![0.0; 11]), Some(heading)).unwrap()
    }

    #[test]
    fn interpolates_positions_and_wraps_heading() {
        let t = straight();
        let p = t.positions(&[0.5, 10.0, -1.0, 10.5], 2.0);
        assert!((p[0][0] - 5.0).abs() < 1e-12 && p[1][0] == 100.0);
        assert!(p[2][0].is_nan() && p[3][0].is_nan());
        let a = t.attitude(&[1.5], 2.0).unwrap();
        assert!((a[0][2] - 357.5).abs() < 1e-9, "{:?}", a);
    }

    #[test]
    fn gaps_are_not_bridged() {
        let t = Trajectory::new(vec![0.0, 1.0, 5.0, 6.0], vec![[0.0; 3], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]], None, None, None).unwrap();
        assert_eq!(t.median_interval(), 1.0);
        let p = t.positions(&[0.5, 3.0, 5.5], 2.0);
        assert!(p[0][0].is_finite() && p[1][0].is_nan() && p[2][0].is_finite());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Trajectory::new(vec![0.0], vec![[0.0; 3]], None, None, None).is_err());
        assert!(Trajectory::new(vec![0.0, f64::NAN], vec![[0.0; 3]; 2], None, None, None).is_err());
        assert!(Trajectory::new(vec![0.0, 1.0], vec![[0.0; 3]], None, None, None).is_err());
        // Unsorted input is sorted; repeated times are dropped.
        let t = Trajectory::new(vec![2.0, 0.0, 2.0, 1.0], vec![[2.0, 0.0, 0.0], [0.0; 3], [9.0, 0.0, 0.0], [1.0, 0.0, 0.0]], None, None, None).unwrap();
        assert_eq!(t.time, vec![0.0, 1.0, 2.0]);
        assert_eq!(t.xyz[2][0], 2.0);
    }

    #[test]
    fn tables_parse_with_or_without_header() {
        let t = parse_table("# comment\ntime, x, y, z\n0, 1, 2, 3\n1, 2, 3, 4\n").unwrap();
        assert_eq!(t.names, vec!["time", "x", "y", "z"]);
        assert_eq!(t.columns[3], vec![3.0, 4.0]);
        let t = parse_table("0 1 2 3\n1  2 3 4\n").unwrap();
        assert_eq!(t.names[0], "col0");
        assert_eq!(t.columns[1], vec![1.0, 2.0]);
        assert!(parse_table("0 1 2\n1 2\n").is_err());
        assert!(parse_table("a,b\n1,x\n").is_err());
        assert_eq!(column_role("GPS_Time"), Some("time"));
        assert_eq!(column_role("Easting"), Some("x"));
        assert_eq!(column_role("yaw"), Some("heading"));
    }

    #[test]
    fn sbet_round_trips() {
        let dir = std::env::temp_dir().join(format!("sylva_sbet_{}", std::process::id()));
        let s = Sbet { time: vec![1.0, 2.0], latitude: vec![-27.5, -27.4], longitude: vec![153.0, 153.1], height: vec![100.0, 101.0], roll: vec![1.0, 2.0], pitch: vec![-1.0, 0.5], heading: vec![10.0, 350.0] };
        write_sbet(&dir, &s).unwrap();
        let r = read_sbet(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        for (a, b) in [(&s.latitude, &r.latitude), (&s.heading, &r.heading), (&s.roll, &r.roll)] {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-9);
            }
        }
        assert_eq!(r.time, s.time);
    }

    #[test]
    fn recovers_a_moving_sensor_from_pulse_lines() {
        // Sensor at (0, 20 t, 100); pulses fan across the track to the ground.
        let mut lines = Vec::new();
        for i in 0..400 {
            let t = i as f64 * 0.001;
            let o = [0.0, 20.0 * t, 100.0];
            let a = ((i % 40) as f64 / 39.0 - 0.5) * 0.9;
            let d = [a.sin(), 0.0, -a.cos()];
            let q = [o[0] + 100.0 / a.cos() * d[0], o[1], 0.0];
            lines.push(PulseLine { time: t, line: 1, point: q, dir: [-d[0], -d[1], -d[2]], separation: 10.0 });
        }
        let e = estimate(&lines, &[(1, -1.0, 0.5)], &EstimateParams { interval: 1.0, min_pulses: 30, ..Default::default() }).unwrap();
        assert_eq!(e.trajectory.time[0], -1.0);
        let p = e.trajectory.positions(&[0.2], 1.0)[0];
        assert!((p[1] - 4.0).abs() < 1e-6 && (p[2] - 100.0).abs() < 1e-6 && p[0].abs() < 1e-6, "{p:?}");
    }
}
