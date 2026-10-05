// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Retro-reflective targets: reading them, finding them, matching them.
//!
//! A target is a point, not a fitted cylinder, so it is located to
//! millimetres, and three shared targets fix all six degrees of freedom where
//! a stem map fixes four. Targets come from a RIEGL `.tpl` tie-point list, a
//! RiSCAN PRO `.rfl` reflector list, or the cloud itself by return strength.
//! Two target sets are aligned by an exhaustive search over congruent
//! triangles, each scored by the Kabsch (1976) fit of the targets it brings
//! into agreement. The readers keep the Python package's reading of odd
//! input (Python's `float()` and `int()` on each field).

use std::path::Path;

use crate::cluster::euclidean_clusters;
use crate::coreg::matching::numpy_sum;
use crate::coreg::transforms::{kabsch, transform_points, Mat4};
use crate::error::{Error, Result};
use crate::util::json::{self, Json};
use crate::util::numeric::median;
use crate::Point;

/// One retro-reflective target, in the scan's own frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Reflector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub reflectance: f64,
    pub diameter: f64,
    pub n_points: i64,
    pub name: String,
}

impl Reflector {
    pub fn position(&self) -> Point {
        [self.x, self.y, self.z]
    }
}

// ------------------------------------------------------ Python's conversions

/// Python's `str.strip()` whitespace.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// Digits with single underscores between them, as Python's numeric
/// literals allow; `None` for a misplaced underscore.
fn without_underscores(s: &str) -> Option<String> {
    if !s.contains('_') {
        return Some(s.to_string());
    }
    let b: Vec<char> = s.chars().collect();
    for (i, &c) in b.iter().enumerate() {
        if c == '_' && !(i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit()) {
            return None;
        }
    }
    Some(s.replace('_', ""))
}

/// Python's `float(str)`.
fn py_float_str(s: &str) -> Option<f64> {
    let t = without_underscores(py_strip(s))?;
    let body = t.strip_prefix(['+', '-']).unwrap_or(&t);
    let lower = body.to_ascii_lowercase();
    if matches!(lower.as_str(), "nan" | "inf" | "infinity") {
        return t.to_ascii_lowercase().parse().ok();
    }
    let ok = !body.is_empty() && body.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-')) && body.chars().any(|c| c.is_ascii_digit());
    if !ok {
        return None;
    }
    t.parse().ok()
}

/// Python's `float(x)` on a decoded JSON value: `None` where Python raises.
fn py_float(v: &Json) -> Option<f64> {
    match v {
        Json::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Json::Int(i) => Some(*i as f64),
        Json::Float(f) => Some(*f),
        Json::Str(s) => py_float_str(s),
        _ => None,
    }
}

/// Python's `int(float)`: truncation, an error for NaN and infinities.
fn py_int_of_float(f: f64) -> Result<i64> {
    if f.is_finite() {
        Ok(f.trunc() as i64)
    } else {
        Err(Error::invalid(format!("cannot convert float {} to integer", json::py_float_repr(f))))
    }
}

/// Python's `int(x)` on a decoded JSON value.
fn py_int(v: &Json) -> Result<i64> {
    match v {
        Json::Bool(b) => Ok(*b as i64),
        Json::Int(i) => Ok(*i),
        Json::Float(f) => py_int_of_float(*f),
        Json::Str(s) => without_underscores(py_strip(s)).and_then(|t| t.parse::<i64>().ok()).ok_or_else(|| Error::invalid(format!("invalid literal for int() with base 10: {s:?}"))),
        _ => Err(Error::invalid("int() argument must be a string or a real number")),
    }
}

/// Python's `str(x)` on a decoded JSON value.
fn py_str(v: &Json) -> String {
    match v {
        Json::Null => "None".into(),
        Json::Bool(b) => if *b { "True" } else { "False" }.into(),
        Json::Int(i) => i.to_string(),
        Json::Float(f) => json::py_float_repr(*f),
        Json::Str(s) => s.clone(),
        other => json::to_string_indented(other, 0).replace('\n', ""),
    }
}

// ------------------------------------------------------------------- readers

/// Read a RIEGL `.tpl` tie-point list (JSON). Empty for a missing or
/// unreadable file (a position that found no targets is normal); entries
/// without a position are skipped. An error only where a present field
/// cannot be read as a number.
pub fn read_tiepoint_list(path: impl AsRef<Path>) -> Result<Vec<Reflector>> {
    let Ok(bytes) = std::fs::read(path) else { return Ok(Vec::new()) };
    let Ok(text) = String::from_utf8(bytes) else { return Ok(Vec::new()) };
    let Ok(Json::Array(entries)) = json::parse(&text) else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    let empty = Json::Object(Vec::new());
    for entry in &entries {
        let cartesian = match entry {
            Json::Object(_) => entry.get("positionCartesian").filter(|c| c.truthy()).unwrap_or(&empty),
            _ => &empty,
        };
        let coord = |k: &str| cartesian.get(k).and_then(py_float);
        let (Some(x), Some(y), Some(z)) = (coord("x"), coord("y"), coord("z")) else { continue };
        let number = |k: &str| -> Result<f64> {
            match entry.get(k) {
                None => Ok(f64::NAN),
                Some(v) => py_float(v).ok_or_else(|| Error::invalid(format!("could not convert {} to float", py_str(v)))),
            }
        };
        out.push(Reflector {
            x,
            y,
            z,
            reflectance: number("reflectance")?,
            diameter: number("diameter")?,
            n_points: entry.get("pointcount").map_or(Ok(0), py_int)?,
            name: entry.get("name").map_or(String::new(), py_str),
        });
    }
    Ok(out)
}

/// Python's `str.splitlines()`.
fn py_splitlines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let brk = matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if brk {
            out.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(_, '\n')) = chars.peek() {
                    chars.next();
                    end += 1;
                }
            }
            start = end;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// Read a RiSCAN PRO `.rfl` reflector list (`RieglRflID`).
///
/// Each `ReflectorN=` line holds the fields named by the `ReflectorIdx=`
/// line, with x, y, z in the scanner's own frame. Empty for a missing file;
/// rows without a readable position are skipped.
pub fn read_reflector_list(path: impl AsRef<Path>) -> Result<Vec<Reflector>> {
    let Ok(bytes) = std::fs::read(path) else { return Ok(Vec::new()) };
    let text = String::from_utf8_lossy(&bytes);
    let mut columns: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for line in py_splitlines(&text) {
        let (key, value) = line.split_once('=').unwrap_or((line, ""));
        let key = py_strip(key);
        if key == "ReflectorIdx" {
            columns = value.split(',').map(|c| py_strip(c).to_lowercase()).collect();
            continue;
        }
        if !key.starts_with("Reflector") || columns.is_empty() {
            continue;
        }
        let mut row: Vec<(&str, &str)> = Vec::new();
        for (c, v) in columns.iter().zip(value.split(',')) {
            let v = py_strip(v);
            match row.iter_mut().find(|(k, _)| k == c) {
                Some(slot) => slot.1 = v,
                None => row.push((c.as_str(), v)),
            }
        }
        let field = |k: &str| row.iter().find(|(c, _)| *c == k).map(|(_, v)| *v);
        let num = |k: &str, default: f64| field(k).and_then(py_float_str).unwrap_or(default);
        let (Some(x), Some(y), Some(z)) = (field("x").and_then(py_float_str), field("y").and_then(py_float_str), field("z").and_then(py_float_str)) else { continue };
        out.push(Reflector {
            x,
            y,
            z,
            reflectance: num("reflectance", f64::NAN),
            diameter: num("diameter", f64::NAN),
            n_points: py_int_of_float(num("points", 0.0))?,
            name: field("name").unwrap_or("").to_string(),
        });
    }
    Ok(out)
}

// ----------------------------------------------------------------- detection

/// Find retro-reflective targets in a scan by their return strength: returns
/// at least `min_reflectance` bright, single-link clustered at
/// `cluster_radius`, clusters of `min_points` or more and no wider than
/// `max_extent`. Each target sits at the cluster's centroid weighted by
/// reflectance above the cluster's faintest return.
pub fn detect_reflectors(xyz: &[Point], reflectance: Option<&[f64]>, min_reflectance: f64, cluster_radius: f64, min_points: usize, max_extent: f64) -> Result<Vec<Reflector>> {
    let Some(values) = reflectance else { return Ok(Vec::new()) };
    if xyz.is_empty() {
        return Ok(Vec::new());
    }
    if values.len() != xyz.len() {
        return Err(Error::invalid("reflectance must have one value per point"));
    }
    let bright: Vec<usize> = (0..xyz.len()).filter(|&i| values[i] >= min_reflectance).collect();
    if bright.is_empty() {
        return Ok(Vec::new());
    }
    let points: Vec<Point> = bright.iter().map(|&i| xyz[i]).collect();
    let values: Vec<f64> = bright.iter().map(|&i| values[i]).collect();
    let labels = euclidean_clusters(&points, cluster_radius, min_points);
    let n_labels = labels.iter().copied().max().map_or(0, |m| (m + 1).max(0) as usize);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_labels];
    for (i, &l) in labels.iter().enumerate() {
        if l >= 0 {
            members[l as usize].push(i);
        }
    }
    let mut out = Vec::new();
    for m in members.iter().filter(|m| !m.is_empty()) {
        let mut lo = points[m[0]];
        let mut hi = points[m[0]];
        for &i in &m[1..] {
            for k in 0..3 {
                // np.min / np.max propagate NaN.
                let v = points[i][k];
                if v < lo[k] || v.is_nan() {
                    lo[k] = if lo[k].is_nan() { lo[k] } else { v };
                }
                if v > hi[k] || v.is_nan() {
                    hi[k] = if hi[k].is_nan() { hi[k] } else { v };
                }
            }
        }
        let extent = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let widest = if extent.iter().any(|e| e.is_nan()) { f64::NAN } else { extent[0].max(extent[1]).max(extent[2]) };
        if widest > max_extent {
            continue;
        }
        let v: Vec<f64> = m.iter().map(|&i| values[i]).collect();
        let vmin = v.iter().copied().fold(f64::INFINITY, f64::min);
        let w: Vec<f64> = v.iter().map(|x| x - vmin + 1.0).collect();
        let scale = numpy_sum(&w);
        let mut centre = [0.0; 3];
        for (j, &i) in m.iter().enumerate() {
            for k in 0..3 {
                let p = points[i][k] * w[j];
                centre[k] = if j == 0 { p } else { centre[k] + p };
            }
        }
        out.push(Reflector {
            x: centre[0] / scale,
            y: centre[1] / scale,
            z: centre[2] / scale,
            reflectance: median(&v),
            diameter: (extent[0] * extent[0] + extent[1] * extent[1]).sqrt(),
            n_points: m.len() as i64,
            name: String::new(),
        });
    }
    Ok(out)
}

// ------------------------------------------------------------------ matching

/// Alignment of two target sets; `transform` maps source into target.
#[derive(Debug, Clone)]
pub struct ReflectorMatch {
    pub transform: Mat4,
    pub n_inliers: usize,
    pub rmse: f64,
    /// `(source, target)` indices of the matched targets.
    pub correspondences: Vec<[usize; 2]>,
    pub success: bool,
}

impl ReflectorMatch {
    fn failure(transform: Mat4, n_inliers: usize) -> Self {
        ReflectorMatch { transform, n_inliers, rmse: f64::INFINITY, correspondences: Vec::new(), success: false }
    }
}

fn dist(a: &Point, b: &Point) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn collinear(a: &Point, b: &Point, c: &Point, tolerance: f64) -> bool {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let x = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    let area = 0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let (ab, ac, bc) = (dist(b, a), dist(c, a), dist(c, b));
    let mut longest = ab;
    for d in [ac, bc] {
        if d > longest {
            longest = d;
        }
    }
    area < tolerance * (longest * longest)
}

fn score(src: &[Point], dst: &[Point], s: [usize; 3], d: [usize; 3], tolerance: f64) -> Result<ReflectorMatch> {
    let Ok(t) = kabsch(&s.map(|i| src[i]), &d.map(|i| dst[i]), None) else {
        return Ok(ReflectorMatch::failure(Mat4::identity(), 0));
    };
    let moved = transform_points(&t, src);
    let mut nearest = vec![0usize; src.len()];
    let mut smallest = vec![0.0; src.len()];
    for (i, m) in moved.iter().enumerate() {
        let mut best = (0, f64::NAN);
        for (j, q) in dst.iter().enumerate() {
            let g = dist(m, q);
            if j == 0 || g < best.1 || (g.is_nan() && !best.1.is_nan()) {
                best = (j, g);
            }
        }
        nearest[i] = best.0;
        smallest[i] = best.1;
    }
    let hit: Vec<bool> = smallest.iter().map(|&g| g <= tolerance).collect();
    let n_hit = hit.iter().filter(|&&h| h).count();
    if n_hit < 3 {
        return Ok(ReflectorMatch::failure(t, n_hit));
    }
    let mut order: Vec<usize> = (0..src.len()).collect();
    order.sort_by(|&a, &b| smallest[a].total_cmp(&smallest[b]));
    let mut pairs: Vec<[usize; 2]> = Vec::new();
    let mut seen = vec![false; dst.len()];
    for s in order {
        if hit[s] && !seen[nearest[s]] {
            seen[nearest[s]] = true;
            pairs.push([s, nearest[s]]);
        }
    }
    if pairs.len() < 3 {
        return Ok(ReflectorMatch::failure(t, pairs.len()));
    }
    // By source index, not by residual: candidate transforms that tie up to
    // rounding would otherwise order the same pairs differently.
    pairs.sort_unstable();
    let a: Vec<Point> = pairs.iter().map(|p| src[p[0]]).collect();
    let b: Vec<Point> = pairs.iter().map(|p| dst[p[1]]).collect();
    let refined = kabsch(&a, &b, None)?;
    let sq: Vec<f64> = transform_points(&refined, &a).iter().zip(&b).map(|(p, q)| dist(p, q).powi(2)).collect();
    let rmse = (numpy_sum(&sq) / sq.len() as f64).sqrt();
    Ok(ReflectorMatch { transform: refined, n_inliers: pairs.len(), rmse, correspondences: pairs, success: false })
}

/// Align two target sets with no initial guess.
///
/// Source triangles (sides of at least 0.3 m, not collinear) are searched
/// exhaustively against target triangles whose sides agree within
/// `distance_tolerance`; each candidate is scored by how many targets land
/// within `tolerance` of a partner, ties going to the lower RMSE. Success
/// needs `min_inliers` targets.
pub fn match_reflectors(src: &[Point], dst: &[Point], tolerance: f64, min_inliers: i64, distance_tolerance: f64) -> Result<ReflectorMatch> {
    let failure = ReflectorMatch::failure(Mat4::identity(), 0);
    if src.len() < 3 || dst.len() < 3 {
        return Ok(failure);
    }
    let n = dst.len();
    let dd: Vec<f64> = (0..n * n).map(|k| dist(&dst[k / n], &dst[k % n])).collect();
    let mut best = failure;
    for a in 0..src.len() {
        for b in a + 1..src.len() {
            let d_ab = dist(&src[a], &src[b]);
            for c in b + 1..src.len() {
                let d_ac = dist(&src[a], &src[c]);
                let d_bc = dist(&src[b], &src[c]);
                let mut shortest = d_ab;
                for d in [d_ac, d_bc] {
                    if d < shortest {
                        shortest = d;
                    }
                }
                if shortest < 0.3 || collinear(&src[a], &src[b], &src[c], 0.05) {
                    continue;
                }
                for i in 0..n {
                    for j in 0..n {
                        if j == i || (dd[i * n + j] - d_ab).abs() > distance_tolerance {
                            continue;
                        }
                        for k in 0..n {
                            if k == i || k == j || (dd[i * n + k] - d_ac).abs() > distance_tolerance || (dd[j * n + k] - d_bc).abs() > distance_tolerance {
                                continue;
                            }
                            let cand = score(src, dst, [a, b, c], [i, j, k], tolerance)?;
                            if cand.n_inliers > best.n_inliers || (cand.n_inliers == best.n_inliers && cand.rmse < best.rmse) {
                                best = cand;
                            }
                        }
                    }
                }
            }
        }
    }
    if (best.n_inliers as i64) < min_inliers {
        return Ok(ReflectorMatch::failure(Mat4::identity(), 0));
    }
    best.success = true;
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coreg::transforms::{invert, se3_exp};
    use nalgebra::Vector6;

    fn targets() -> Vec<Point> {
        vec![[3.0, 4.0, 1.0], [-8.0, 2.0, 0.5], [10.0, -6.0, 2.0], [-4.0, -9.0, 1.5], [0.0, 12.0, -1.0], [15.0, 7.0, 0.0]]
    }

    #[test]
    fn a_full_transform_is_recovered() {
        let t = se3_exp(&Vector6::new(0.05, -0.03, 1.2, 4.0, -3.0, 0.5));
        let dst = targets();
        let src = transform_points(&invert(&t), &dst);
        let m = match_reflectors(&src, &dst, 0.05, 3, 0.03).unwrap();
        assert!(m.success && m.n_inliers == 6);
        assert!((m.transform - t).abs().max() < 1e-9);
        assert!(!match_reflectors(&src[..2], &dst, 0.05, 3, 0.03).unwrap().success);
    }

    #[test]
    fn collinear_targets_are_refused() {
        let line: Vec<Point> = (0..5).map(|i| [2.5 * i as f64, 0.0, 0.0]).collect();
        let moved: Vec<Point> = line.iter().map(|p| [p[0] + 1.0, p[1] + 1.0, p[2] + 1.0]).collect();
        assert!(!match_reflectors(&line, &moved, 0.05, 3, 0.03).unwrap().success);
    }

    #[test]
    fn bright_clusters_are_targets() {
        let mut xyz = Vec::new();
        let mut refl = Vec::new();
        for i in 0..400 {
            xyz.push([(i % 20) as f64, (i / 20) as f64, 0.0]);
            refl.push(-10.0);
        }
        for i in 0..10 {
            xyz.push([5.0 + 0.01 * i as f64, 5.0, 1.0]);
            refl.push(10.0 + i as f64);
        }
        let found = detect_reflectors(&xyz, Some(&refl), 5.0, 0.15, 8, 0.5).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].n_points, 10);
        assert!((found[0].y - 5.0).abs() < 1e-12);
        assert!(detect_reflectors(&xyz, None, 5.0, 0.15, 8, 0.5).unwrap().is_empty());
    }

    #[test]
    fn python_number_parsing() {
        assert_eq!(py_float_str(" -2 "), Some(-2.0));
        assert_eq!(py_float_str("1_000.5"), Some(1000.5));
        assert_eq!(py_float_str("1e3"), Some(1000.0));
        assert!(py_float_str("Infinity").unwrap().is_infinite());
        assert_eq!(py_float_str("one"), None);
        assert_eq!(py_float_str(""), None);
        assert_eq!(py_float_str("_1"), None);
        assert_eq!(py_splitlines("a\r\nb\rc\n"), vec!["a", "b", "c"]);
    }
}
