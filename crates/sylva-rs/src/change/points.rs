// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Surface change between two epochs: point distances and rasters of
//! difference.
//!
//! * [`c2c`]: distance from every point of the later epoch to the nearest
//!   point of the earlier one (cloud-to-cloud).
//! * [`m3c2`]: the Multiscale Model to Model Cloud Comparison of Lague,
//!   Brodu and Leroux (2013): at each core point a normal is fitted to the
//!   reference cloud at the normal scale, both clouds are projected onto
//!   that normal inside a cylinder of the projection scale, and the distance
//!   between the two mean positions is compared with its 95 % level of
//!   detection, `1.96 (sqrt(s_a^2 / n_a + s_b^2 / n_b) + reg)` (their
//!   eq. 1), where `s` is the spread of the projected positions, `n` their
//!   count and `reg` the registration error.
//! * [`dod`]: the difference of two rasters (DTM or CHM of difference) with
//!   a level of detection per cell, either given or propagated from the
//!   uncertainty of each surface (`1.96 sqrt(s_a^2 + s_b^2)`).
//!
//! Every core point or cell is computed on its own, in parallel, so the
//! results do not depend on the number of threads.

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Matrix3, SymmetricEigen};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::raster::Raster;
use crate::Point;

/// Two-sided 95 % quantile of the standard normal distribution.
pub const Z95: f64 = 1.959_963_984_540_054;

/// Core points handled by one parallel task.
const CHUNK: usize = 4096;

fn finite(p: &Point) -> bool {
    p.iter().all(|v| v.is_finite())
}

/// k-d tree over the finite points of a cloud, remembering their original
/// indices; `None` when there is no finite point.
struct Tree {
    tree: ImmutableKdTree<f64, 3>,
    points: Vec<Point>,
}

impl Tree {
    fn new(points: &[Point]) -> Option<Self> {
        let kept: Vec<Point> = points.iter().filter(|p| finite(p)).copied().collect();
        if kept.is_empty() {
            return None;
        }
        let tree = ImmutableKdTree::new_from_slice_parallel(&kept).expect("kd-tree build");
        Some(Tree { tree, points: kept })
    }

    /// Points within `radius` of `p`, in the tree's (deterministic) order.
    fn within(&self, p: &Point, radius: f64) -> impl Iterator<Item = &Point> + '_ {
        self.tree.query(p).within::<SquaredEuclidean<f64>>(radius * radius).unsorted().execute().into_iter().map(move |r| &self.points[r.item as usize])
    }

    fn nearest(&self, p: &Point) -> f64 {
        self.tree.query(p).nearest_one::<SquaredEuclidean<f64>>().execute().distance.sqrt()
    }
}

// ------------------------------------------------------------------ c2c

/// Distance from every point of `compared` to the nearest finite point of
/// `reference`; NaN for a non-finite point, when `reference` has no finite
/// point, or beyond `max_distance`.
///
/// # Errors
/// `max_distance` negative or NaN.
pub fn c2c(reference: &[Point], compared: &[Point], max_distance: Option<f64>) -> Result<Vec<f64>> {
    if let Some(d) = max_distance {
        if d.is_nan() || d < 0.0 {
            return Err(Error::invalid(format!("max_distance must be a non-negative number, got {d}")));
        }
    }
    let Some(tree) = Tree::new(reference) else {
        return Ok(vec![f64::NAN; compared.len()]);
    };
    let limit = max_distance.unwrap_or(f64::INFINITY);
    Ok(compared
        .par_iter()
        .with_min_len(CHUNK)
        .map(|p| {
            if !finite(p) {
                return f64::NAN;
            }
            let d = tree.nearest(p);
            if d <= limit {
                d
            } else {
                f64::NAN
            }
        })
        .collect())
}

// ------------------------------------------------------------------ m3c2

/// How M3C2 normals are oriented (a fitted normal has no sign of its own).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Orientation {
    /// Positive along this direction (e.g. `[0, 0, 1]`, up).
    Direction(Point),
    /// Pointing from the core point towards this location (e.g. the scanner).
    Towards(Point),
}

/// Parameters of [`m3c2`]. Scales are diameters, as in Lague et al. (2013).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct M3c2Params {
    /// Diameter `D` of the neighbourhood the normal is fitted to (m).
    pub normal_scale: f64,
    /// Diameter `d` of the projection cylinder (m).
    pub projection_scale: f64,
    /// Half-length of the cylinder: how far along the normal, either way,
    /// points are searched (m).
    pub max_depth: f64,
    /// Registration error (1 sigma, m), added to the level of detection.
    pub registration_sigma: f64,
    /// Fewest points of each cloud in the cylinder for a level of detection.
    pub min_points: usize,
    pub orientation: Orientation,
}

impl M3c2Params {
    fn check(&self) -> Result<()> {
        for (name, v) in [("normal_scale", self.normal_scale), ("projection_scale", self.projection_scale), ("max_depth", self.max_depth)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(Error::invalid(format!("{name} must be a positive number, got {v}")));
            }
        }
        if !(self.registration_sigma.is_finite() && self.registration_sigma >= 0.0) {
            return Err(Error::invalid(format!("registration_sigma must be finite and non-negative, got {}", self.registration_sigma)));
        }
        let o = match self.orientation {
            Orientation::Direction(d) => {
                if d.iter().all(|&v| v == 0.0) {
                    return Err(Error::invalid("orientation direction must not be zero"));
                }
                d
            }
            Orientation::Towards(p) => p,
        };
        if !finite(&o) {
            return Err(Error::invalid("orientation must be finite"));
        }
        Ok(())
    }
}

/// Per-core-point results of [`m3c2`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct M3c2 {
    /// Signed distance from the reference to the compared surface along the
    /// normal (m); NaN where either cylinder is empty or there is no normal.
    pub distance: Vec<f64>,
    /// 95 % level of detection (m); NaN with fewer than `min_points` (or 2)
    /// points of either cloud in the cylinder.
    pub lod: Vec<f64>,
    /// `|distance| > lod`.
    pub significant: Vec<bool>,
    /// Unit normal used at each core point (NaN where none could be fitted).
    pub normal: Vec<Point>,
    /// Points of each cloud in the cylinder.
    pub n_a: Vec<u32>,
    pub n_b: Vec<u32>,
    /// Standard deviation (n - 1) of each cloud's positions along the normal.
    pub spread_a: Vec<f64>,
    pub spread_b: Vec<f64>,
}

/// Unit normal of the points (smallest principal axis), oriented; None with
/// fewer than three points or a degenerate neighbourhood.
fn fit_normal<'a>(pts: impl Iterator<Item = &'a Point>, core: &Point, orientation: &Orientation) -> Option<Point> {
    let local: Vec<[f64; 3]> = pts.map(|p| [p[0] - core[0], p[1] - core[1], p[2] - core[2]]).collect();
    if local.len() < 3 {
        return None;
    }
    let n = local.len() as f64;
    let mut m = [0.0; 3];
    for p in &local {
        for k in 0..3 {
            m[k] += p[k];
        }
    }
    for v in &mut m {
        *v /= n;
    }
    let mut c = Matrix3::<f64>::zeros();
    for p in &local {
        let d = [p[0] - m[0], p[1] - m[1], p[2] - m[2]];
        for i in 0..3 {
            for j in 0..3 {
                c[(i, j)] += d[i] * d[j];
            }
        }
    }
    let e = SymmetricEigen::new(c / n);
    let mut ord = [0usize, 1, 2];
    ord.sort_by(|&a, &b| e.eigenvalues[a].total_cmp(&e.eigenvalues[b]));
    let l1 = e.eigenvalues[ord[1]];
    if l1.is_nan() || l1 <= 0.0 {
        return None;
    }
    let col = e.eigenvectors.column(ord[0]);
    let len = (col[0] * col[0] + col[1] * col[1] + col[2] * col[2]).sqrt();
    if !(len > 0.0 && len.is_finite()) {
        return None;
    }
    let mut v = [col[0] / len, col[1] / len, col[2] / len];
    let refv = match orientation {
        Orientation::Direction(d) => *d,
        Orientation::Towards(t) => [t[0] - core[0], t[1] - core[1], t[2] - core[2]],
    };
    if v[0] * refv[0] + v[1] * refv[1] + v[2] * refv[2] < 0.0 {
        v = [-v[0], -v[1], -v[2]];
    }
    Some(v)
}

/// Positions along `normal` of the points of `tree` inside the cylinder of
/// radius `r` and half-length `h` centred on `core`.
///
/// The cylinder is cut into slabs of length `2 r` along its axis; each slab
/// lies inside a sphere of radius `r sqrt(2)` around its centre, which one
/// k-d tree query returns, and a point is kept by the slab its axial
/// position falls in, so none is counted twice. Long, thin cylinders are thus
/// searched without the cost of one sphere of radius `h`.
fn cylinder(tree: &Tree, core: &Point, normal: &Point, r: f64, h: f64, out: &mut Vec<f64>) {
    out.clear();
    let s = r;
    let m = (h / s).ceil().max(1.0) as usize;
    let radius = (r * r + s * s).sqrt() * (1.0 + 1e-12);
    let r2 = r * r;
    for k in 0..m {
        let tk = -h + s * (2 * k + 1) as f64;
        let c = [core[0] + normal[0] * tk, core[1] + normal[1] * tk, core[2] + normal[2] * tk];
        for p in tree.within(&c, radius) {
            let d = [p[0] - core[0], p[1] - core[1], p[2] - core[2]];
            let t = d[0] * normal[0] + d[1] * normal[1] + d[2] * normal[2];
            if t.abs() > h {
                continue;
            }
            let slab = (((t + h) / (2.0 * s)).floor().max(0.0) as usize).min(m - 1);
            if slab != k {
                continue;
            }
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - t * t <= r2 {
                out.push(t);
            }
        }
    }
}

/// Mean and sample standard deviation (NaN below two values).
fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, f64::NAN);
    }
    let ss: f64 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
    (mean, (ss / (n - 1) as f64).sqrt())
}

/// M3C2 distances from cloud `a` (reference) to cloud `b` at `core` points.
///
/// Normals are fitted to the points of `a` within `normal_scale / 2` of each
/// core point, unless `normals` are given (one per core point; they are
/// normalised and not re-oriented).
///
/// # Errors
/// Invalid parameters, or `normals` of the wrong length.
pub fn m3c2(a: &[Point], b: &[Point], core: &[Point], normals: Option<&[Point]>, params: &M3c2Params) -> Result<M3c2> {
    params.check()?;
    if let Some(nv) = normals {
        if nv.len() != core.len() {
            return Err(Error::invalid(format!("normals has {} rows for {} core points", nv.len(), core.len())));
        }
    }
    let n = core.len();
    let ta = Tree::new(a);
    let tb = Tree::new(b);
    let r = params.projection_scale / 2.0;
    let h = params.max_depth;
    let min_points = params.min_points.max(2);
    type Row = (f64, f64, Point, u32, u32, f64, f64);
    let rows: Vec<Row> = (0..n)
        .into_par_iter()
        .with_min_len(CHUNK / 16)
        .map_init(
            || (Vec::new(), Vec::new()),
            |(ba, bb), i| {
                let nan = (f64::NAN, f64::NAN, [f64::NAN; 3], 0, 0, f64::NAN, f64::NAN);
                let c = &core[i];
                if !finite(c) {
                    return nan;
                }
                let normal = match normals {
                    Some(nv) => {
                        let v = nv[i];
                        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                        (len > 0.0 && len.is_finite()).then(|| [v[0] / len, v[1] / len, v[2] / len])
                    }
                    None => ta.as_ref().and_then(|t| fit_normal(t.within(c, params.normal_scale / 2.0), c, &params.orientation)),
                };
                let Some(normal) = normal else { return nan };
                match &ta {
                    Some(t) => cylinder(t, c, &normal, r, h, ba),
                    None => ba.clear(),
                }
                match &tb {
                    Some(t) => cylinder(t, c, &normal, r, h, bb),
                    None => bb.clear(),
                }
                let (ma, sa) = mean_sd(ba);
                let (mb, sb) = mean_sd(bb);
                let (na, nb) = (ba.len(), bb.len());
                let dist = mb - ma;
                let lod = if na >= min_points && nb >= min_points {
                    Z95 * ((sa * sa / na as f64 + sb * sb / nb as f64).sqrt() + params.registration_sigma)
                } else {
                    f64::NAN
                };
                (dist, lod, normal, na as u32, nb as u32, sa, sb)
            },
        )
        .collect();
    let mut out = M3c2 {
        distance: Vec::with_capacity(n),
        lod: Vec::with_capacity(n),
        significant: Vec::with_capacity(n),
        normal: Vec::with_capacity(n),
        n_a: Vec::with_capacity(n),
        n_b: Vec::with_capacity(n),
        spread_a: Vec::with_capacity(n),
        spread_b: Vec::with_capacity(n),
    };
    for (d, l, nv, na, nb, sa, sb) in rows {
        out.distance.push(d);
        out.lod.push(l);
        out.significant.push(d.abs() > l);
        out.normal.push(nv);
        out.n_a.push(na);
        out.n_b.push(nb);
        out.spread_a.push(sa);
        out.spread_b.push(sb);
    }
    Ok(out)
}

// ------------------------------------------------------------------ dod

/// An uncertainty or threshold: one value everywhere or one per cell of a
/// raster on the lattice of the surfaces.
#[derive(Debug, Clone, PartialEq)]
pub enum CellValue {
    Scalar(f64),
    Grid(Raster),
}

impl CellValue {
    fn at(&self, x: f64, y: f64) -> f64 {
        match self {
            CellValue::Scalar(v) => *v,
            CellValue::Grid(r) => {
                let (row, col) = r.cell_index(x, y);
                if r.in_bounds(row, col) {
                    r.get(row as usize, col as usize)
                } else {
                    f64::NAN
                }
            }
        }
    }

    fn check(&self, name: &str, res: f64, xmin: f64, ymin: f64) -> Result<()> {
        match self {
            CellValue::Scalar(v) if v.is_nan() || *v < 0.0 => Err(Error::invalid(format!("{name} must be non-negative, got {v}"))),
            CellValue::Scalar(_) => Ok(()),
            CellValue::Grid(r) => {
                if r.data.len() != r.nrows * r.ncols {
                    return Err(Error::invalid(format!("{name}: data length does not match its shape")));
                }
                aligned(res, xmin, ymin, r).map_err(|m| Error::invalid(format!("{name} is not on the lattice of the surfaces: {m}")))
            }
        }
    }
}

/// Whether raster `r` lies on the lattice of resolution `res` through
/// `(xmin, ymin)`.
fn aligned(res: f64, xmin: f64, ymin: f64, r: &Raster) -> std::result::Result<(), String> {
    if !(r.resolution.is_finite() && r.resolution > 0.0) {
        return Err(format!("resolution must be positive, got {}", r.resolution));
    }
    if (r.resolution - res).abs() > 1e-9 * res {
        return Err(format!("resolution {} differs from {res}", r.resolution));
    }
    for (o, o0) in [(r.xmin, xmin), (r.ymin, ymin)] {
        let k = (o - o0) / res;
        if (k - k.round()).abs() > 1e-6 {
            return Err(format!("corner offset {} is not a whole number of cells", o - o0));
        }
    }
    Ok(())
}

/// A difference of two surfaces with its level of detection.
#[derive(Debug, Clone, PartialEq)]
pub struct Dod {
    /// `b - a` on the overlap of the two rasters; NaN where either is NaN.
    pub difference: Raster,
    /// Level of detection per cell (m); NaN where it is unknown.
    pub lod: Raster,
    /// `|difference| > lod` (false where either is NaN), row-major.
    pub significant: Vec<bool>,
    /// Volumes (m³, cell area times difference) of the significant cells:
    /// raised, lowered (positive) and net (raised - lowered).
    pub volume_gained: f64,
    pub volume_lost: f64,
    pub net_volume: f64,
    /// Area (m²) of significant change, and of cells with a finite
    /// difference and level of detection.
    pub area_changed: f64,
    pub area_compared: f64,
}

/// Raster of difference `b - a` with a significance mask.
///
/// The two rasters must share their resolution and lattice (corners a whole
/// number of cells apart); the result covers their overlap. The level of
/// detection is `min_detectable` if given, otherwise `1.96 sqrt(s_a^2 +
/// s_b^2)` from the standard deviations of the two surfaces (either may be
/// omitted when the other is given, and is then taken as equal to it).
/// Per-cell values come from rasters on the same lattice (NaN outside
/// them).
///
/// # Errors
/// Rasters that do not share a lattice or do not overlap, negative or NaN
/// scalars, or neither or both of `min_detectable` and the sigmas.
pub fn dod(a: &Raster, b: &Raster, min_detectable: Option<&CellValue>, sigma_a: Option<&CellValue>, sigma_b: Option<&CellValue>) -> Result<Dod> {
    for (name, r) in [("raster_a", a), ("raster_b", b)] {
        if r.data.len() != r.nrows * r.ncols {
            return Err(Error::invalid(format!("{name}: data length does not match its shape")));
        }
        if !(r.resolution.is_finite() && r.resolution > 0.0) {
            return Err(Error::invalid(format!("{name}: resolution must be positive, got {}", r.resolution)));
        }
        if !(r.xmin.is_finite() && r.ymin.is_finite()) {
            return Err(Error::invalid(format!("{name}: corner must be finite")));
        }
    }
    let res = a.resolution;
    aligned(res, a.xmin, a.ymin, b).map_err(|m| Error::invalid(format!("raster_b is not on the lattice of raster_a: {m}")))?;
    match (min_detectable, sigma_a, sigma_b) {
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => return Err(Error::invalid("give min_detectable or sigma_a / sigma_b, not both")),
        (None, None, None) => return Err(Error::invalid("give min_detectable or sigma_a / sigma_b: every change needs a level of detection")),
        _ => (),
    }
    for (name, v) in [("min_detectable", min_detectable), ("sigma_a", sigma_a), ("sigma_b", sigma_b)] {
        if let Some(v) = v {
            v.check(name, res, a.xmin, a.ymin)?;
        }
    }
    // Overlap in cells of a.
    let off = |o: f64, o0: f64| ((o - o0) / res).round() as i64;
    let (bc, br) = (off(b.xmin, a.xmin), off(b.ymin, a.ymin));
    let c0 = bc.max(0);
    let r0 = br.max(0);
    let c1 = (a.ncols as i64).min(bc + b.ncols as i64);
    let r1 = (a.nrows as i64).min(br + b.nrows as i64);
    if c1 <= c0 || r1 <= r0 {
        return Err(Error::invalid("the two rasters do not overlap"));
    }
    let (ncols, nrows) = ((c1 - c0) as usize, (r1 - r0) as usize);
    let xmin = a.xmin + c0 as f64 * res;
    let ymin = a.ymin + r0 as f64 * res;
    let cells: Vec<(f64, f64)> = (0..nrows * ncols)
        .into_par_iter()
        .map(|i| {
            let (r, c) = (i / ncols, i % ncols);
            let (ra, ca) = (r + r0 as usize, c + c0 as usize);
            let (rb, cb) = ((ra as i64 - br) as usize, (ca as i64 - bc) as usize);
            let d = b.get(rb, cb) - a.get(ra, ca);
            let (x, y) = (xmin + (c as f64 + 0.5) * res, ymin + (r as f64 + 0.5) * res);
            let lod = match min_detectable {
                Some(m) => m.at(x, y),
                None => {
                    let sa = sigma_a.or(sigma_b).map_or(f64::NAN, |s| s.at(x, y));
                    let sb = sigma_b.or(sigma_a).map_or(f64::NAN, |s| s.at(x, y));
                    Z95 * (sa * sa + sb * sb).sqrt()
                }
            };
            (if d.is_finite() { d } else { f64::NAN }, lod)
        })
        .collect();
    let area = res * res;
    let (mut gained, mut lost, mut n_changed, mut n_compared) = (0.0, 0.0, 0usize, 0usize);
    let mut significant = Vec::with_capacity(cells.len());
    for &(d, l) in &cells {
        let s = d.abs() > l;
        significant.push(s);
        if d.is_finite() && l.is_finite() {
            n_compared += 1;
        }
        if s {
            n_changed += 1;
            if d > 0.0 {
                gained += d * area;
            } else {
                lost -= d * area;
            }
        }
    }
    Ok(Dod {
        difference: Raster { data: cells.iter().map(|c| c.0).collect(), nrows, ncols, xmin, ymin, resolution: res },
        lod: Raster { data: cells.iter().map(|c| c.1).collect(), nrows, ncols, xmin, ymin, resolution: res },
        significant,
        volume_gained: gained,
        volume_lost: lost,
        net_volume: gained - lost,
        area_changed: n_changed as f64 * area,
        area_compared: n_compared as f64 * area,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nprandom::Generator;

    fn plane(n_side: usize, step: f64, z: f64, noise: f64, seed: u64) -> Vec<Point> {
        let mut g = Generator::new(seed);
        let mut out = Vec::with_capacity(n_side * n_side);
        for i in 0..n_side {
            for j in 0..n_side {
                let e = g.normal(0.0, noise);
                out.push([i as f64 * step + 0.3 * step * g.random(), j as f64 * step + 0.3 * step * g.random(), z + e]);
            }
        }
        out
    }

    fn params() -> M3c2Params {
        M3c2Params { normal_scale: 1.0, projection_scale: 0.5, max_depth: 1.0, registration_sigma: 0.0, min_points: 4, orientation: Orientation::Direction([0.0, 0.0, 1.0]) }
    }

    #[test]
    fn c2c_matches_brute_force() {
        let a = plane(20, 0.1, 0.0, 0.01, 1);
        let b = plane(20, 0.1, 0.05, 0.01, 2);
        let d = c2c(&a, &b, None).unwrap();
        for (p, &di) in b.iter().zip(&d) {
            let best = a.iter().map(|q| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()).fold(f64::INFINITY, f64::min);
            assert!((best - di).abs() < 1e-12);
        }
        let capped = c2c(&a, &b, Some(0.0)).unwrap();
        assert!(capped.iter().all(|v| v.is_nan()));
        assert!(c2c(&[], &b, None).unwrap().iter().all(|v| v.is_nan()));
        assert!(c2c(&a, &b, Some(-1.0)).is_err());
    }

    #[test]
    fn m3c2_recovers_a_shift_and_its_sign() {
        let a = plane(60, 0.05, 0.0, 0.005, 3);
        let b = plane(60, 0.05, 0.1, 0.005, 4);
        let core = vec![[1.5, 1.5, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 0.0]];
        let r = m3c2(&a, &b, &core, None, &params()).unwrap();
        for i in 0..3 {
            assert!((r.distance[i] - 0.1).abs() < 0.005, "{}", r.distance[i]);
            assert!(r.significant[i]);
            assert!(r.normal[i][2] > 0.99);
            assert!(r.n_a[i] > 50 && r.n_b[i] > 50);
        }
        // Reversed orientation flips the sign.
        let mut p = params();
        p.orientation = Orientation::Towards([1.5, 1.5, -10.0]);
        let r2 = m3c2(&a, &b, &core, None, &p).unwrap();
        assert!((r2.distance[0] + r.distance[0]).abs() < 1e-12);
    }

    #[test]
    fn cylinder_slabs_count_each_point_once() {
        let a = plane(40, 0.05, 0.0, 0.3, 5);
        let tree = Tree::new(&a).unwrap();
        let core = [1.0, 1.0, 0.0];
        let nrm = [0.0, 0.6, 0.8];
        let mut got = Vec::new();
        cylinder(&tree, &core, &nrm, 0.3, 2.5, &mut got);
        let mut want: Vec<f64> = a
            .iter()
            .filter_map(|p| {
                let d = [p[0] - core[0], p[1] - core[1], p[2] - core[2]];
                let t = d[0] * nrm[0] + d[1] * nrm[1] + d[2] * nrm[2];
                (t.abs() <= 2.5 && d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - t * t <= 0.09).then_some(t)
            })
            .collect();
        got.sort_by(f64::total_cmp);
        want.sort_by(f64::total_cmp);
        assert_eq!(got, want);
        assert!(!got.is_empty());
    }

    #[test]
    fn m3c2_lod_and_registration() {
        let a = plane(60, 0.05, 0.0, 0.01, 6);
        let b = plane(60, 0.05, 0.0, 0.01, 7);
        let core = vec![[1.5, 1.5, 0.0]];
        let r = m3c2(&a, &b, &core, None, &params()).unwrap();
        let expect = Z95 * (r.spread_a[0].powi(2) / r.n_a[0] as f64 + r.spread_b[0].powi(2) / r.n_b[0] as f64).sqrt();
        assert!((r.lod[0] - expect).abs() < 1e-12);
        let mut p = params();
        p.registration_sigma = 0.02;
        let r2 = m3c2(&a, &b, &core, None, &p).unwrap();
        assert!((r2.lod[0] - r.lod[0] - Z95 * 0.02).abs() < 1e-12);
        // Too few points: no level of detection, never significant.
        p.min_points = 100_000;
        let r3 = m3c2(&a, &b, &core, None, &p).unwrap();
        assert!(r3.lod[0].is_nan() && !r3.significant[0] && r3.distance[0].is_finite());
    }

    #[test]
    fn m3c2_edge_cases() {
        let a = plane(10, 0.1, 0.0, 0.0, 8);
        let core = vec![[0.5, 0.5, 0.0], [f64::NAN, 0.0, 0.0], [50.0, 50.0, 0.0]];
        let r = m3c2(&a, &[], &core, None, &params()).unwrap();
        assert!(r.distance.iter().all(|v| v.is_nan()));
        assert_eq!(r.n_b, vec![0, 0, 0]);
        let r = m3c2(&[], &a, &core, Some(&[[0.0, 0.0, 2.0]; 3]), &params()).unwrap();
        assert_eq!(r.normal[0], [0.0, 0.0, 1.0]);
        assert!(m3c2(&a, &a, &core, Some(&[[0.0, 0.0, 1.0]]), &params()).is_err());
        let mut p = params();
        p.projection_scale = 0.0;
        assert!(m3c2(&a, &a, &core, None, &p).is_err());
        p = params();
        p.orientation = Orientation::Direction([0.0; 3]);
        assert!(m3c2(&a, &a, &core, None, &p).is_err());
    }

    #[test]
    fn m3c2_is_thread_count_independent() {
        let a = plane(50, 0.05, 0.0, 0.02, 9);
        let b = plane(50, 0.05, 0.03, 0.02, 10);
        let run = |t: usize| rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap().install(|| m3c2(&a, &b, &a, None, &params()).unwrap());
        let (r1, r4) = (run(1), run(4));
        assert_eq!(format!("{:?}", r1.distance), format!("{:?}", r4.distance));
        assert_eq!(format!("{:?}", r1.lod), format!("{:?}", r4.lod));
    }

    fn raster(nrows: usize, ncols: usize, xmin: f64, ymin: f64, f: impl Fn(usize, usize) -> f64) -> Raster {
        let mut r = Raster::filled(nrows, ncols, xmin, ymin, 1.0, 0.0);
        for i in 0..nrows {
            for j in 0..ncols {
                r.set(i, j, f(i, j));
            }
        }
        r
    }

    #[test]
    fn dod_overlap_threshold_and_volumes() {
        let a = raster(4, 5, 0.0, 0.0, |_, _| 1.0);
        let b = raster(4, 5, 1.0, 1.0, |i, j| if i == 0 && j == 0 { 1.5 } else if i == 1 && j == 1 { 0.2 } else { 1.05 });
        let d = dod(&a, &b, Some(&CellValue::Scalar(0.1)), None, None).unwrap();
        assert_eq!((d.difference.nrows, d.difference.ncols, d.difference.xmin, d.difference.ymin), (3, 4, 1.0, 1.0));
        assert!((d.difference.get(0, 0) - 0.5).abs() < 1e-12);
        assert_eq!(d.significant.iter().filter(|&&s| s).count(), 2);
        assert!((d.volume_gained - 0.5).abs() < 1e-12 && (d.volume_lost - 0.8).abs() < 1e-12);
        assert!((d.net_volume + 0.3).abs() < 1e-12);
        assert_eq!((d.area_changed, d.area_compared), (2.0, 12.0));
        let s = dod(&a, &b, None, Some(&CellValue::Scalar(0.1)), None).unwrap();
        assert!((s.lod.data[0] - Z95 * 0.1 * 2f64.sqrt()).abs() < 1e-12);
        let g = CellValue::Grid(raster(2, 2, 1.0, 1.0, |_, _| 0.01));
        let s = dod(&a, &b, None, Some(&g), Some(&CellValue::Scalar(0.0))).unwrap();
        assert!((s.lod.get(0, 0) - Z95 * 0.01).abs() < 1e-12 && s.lod.get(2, 3).is_nan() && !s.significant[2 * 4 + 3]);
    }

    #[test]
    fn dod_rejects_bad_input() {
        let a = raster(3, 3, 0.0, 0.0, |_, _| 0.0);
        let far = raster(3, 3, 10.0, 0.0, |_, _| 0.0);
        let off = raster(3, 3, 0.5, 0.0, |_, _| 0.0);
        let t = CellValue::Scalar(0.1);
        assert!(dod(&a, &far, Some(&t), None, None).is_err());
        assert!(dod(&a, &off, Some(&t), None, None).is_err());
        assert!(dod(&a, &a, None, None, None).is_err());
        assert!(dod(&a, &a, Some(&t), Some(&t), None).is_err());
        assert!(dod(&a, &a, Some(&CellValue::Scalar(-1.0)), None, None).is_err());
    }
}
